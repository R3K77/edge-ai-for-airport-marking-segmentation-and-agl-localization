#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_thresholds(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("No thresholds parsed")
    return vals


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    x, y = center
    h, w = heatmap.shape
    diameter = 2 * radius + 1
    sigma = max(diameter / 6.0, 1e-6)

    xs = np.arange(0, diameter, 1, float)
    ys = xs[:, np.newaxis]
    x0 = y0 = radius
    g = np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma * sigma))

    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    masked_hm = heatmap[y - top:y + bottom, x - left:x + right]
    masked_g = g[radius - top:radius + bottom, radius - left:radius + right]

    if masked_hm.size > 0 and masked_g.size > 0:
        np.maximum(masked_hm, masked_g, out=masked_hm)


def max_pool_nms(hm: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(hm, kernel_size=kernel, stride=1, padding=pad)
    keep = (pooled == hm).float()
    return hm * keep


def greedy_match_points(gt_points, pred_points, match_radius_px: float):
    matched_gt = set()
    matched_pred = set()
    dists = []

    if not gt_points and not pred_points:
        return 0, 0, 0, dists, matched_gt, matched_pred

    pairs = []
    for gi, gt in enumerate(gt_points):
        gx, gy = gt
        for pi, pred in enumerate(pred_points):
            px, py = pred["x"], pred["y"]
            d = math.hypot(px - gx, py - gy)
            if d <= match_radius_px:
                pairs.append((d, gi, pi))

    pairs.sort(key=lambda t: t[0])

    for d, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        dists.append(d)

    tp = len(matched_gt)
    fp = len(pred_points) - tp
    fn = len(gt_points) - tp
    return tp, fp, fn, dists, matched_gt, matched_pred


def save_confusion_matrix(cm: np.ndarray, output_path: Path, labels: list[str]) -> None:
    fig = plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Image-level confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("GT")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _extract_points_from_label_json(data: Any) -> list[list[float]]:
    points = []

    if isinstance(data, dict) and "points" in data:
        raw = data["points"]
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    x = item.get("x")
                    y = item.get("y")
                    if x is not None and y is not None:
                        points.append([float(x), float(y)])
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    points.append([float(item[0]), float(item[1])])

    if isinstance(data, dict) and "shapes" in data:
        for shape in data["shapes"]:
            if not isinstance(shape, dict):
                continue
            if shape.get("shape_type") != "point":
                continue
            pts = shape.get("points", [])
            if isinstance(pts, list) and pts:
                p = pts[0]
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    points.append([float(p[0]), float(p[1])])

    return points


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    is_positive: bool


def _load_split_samples(split_dir: Path) -> list[Sample]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected {images_dir} and {labels_dir}")

    samples = []
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = labels_dir / f"{img_path.stem}.json"
        if not label_path.exists():
            continue

        with label_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        pts = _extract_points_from_label_json(data)
        samples.append(Sample(img_path, label_path, len(pts) > 0))

    if not samples:
        raise RuntimeError(f"No samples found in {split_dir}")

    return samples


class PointHeatmapDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        img_size: int,
        down_ratio: int,
        gaussian_radius: int,
        flip_aug: bool = False,
        normalize: bool = True,
    ) -> None:
        self.split_dir = split_dir
        self.samples = _load_split_samples(split_dir)
        self.img_size = img_size
        self.down_ratio = down_ratio
        self.hm_size = img_size // down_ratio
        self.gaussian_radius = gaussian_radius
        self.flip_aug = flip_aug
        self.normalize = normalize

        self.positive_count = sum(1 for s in self.samples if s.is_positive)
        self.negative_count = len(self.samples) - self.positive_count

    def __len__(self) -> int:
        return len(self.samples)

    def _load_points(self, label_path: Path) -> list[list[float]]:
        with label_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _extract_points_from_label_json(data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
            img_np = np.asarray(img, dtype=np.float32) / 255.0

        points = self._load_points(sample.label_path)

        sx = self.img_size / max(orig_w, 1)
        sy = self.img_size / max(orig_h, 1)
        points_resized = [[p[0] * sx, p[1] * sy] for p in points]

        if self.flip_aug and random.random() < 0.5:
            img_np = img_np[:, ::-1, :].copy()
            points_resized = [[self.img_size - 1 - p[0], p[1]] for p in points_resized]

        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_np = (img_np - mean) / std

        heatmap = np.zeros((1, self.hm_size, self.hm_size), dtype=np.float32)
        offset = np.zeros((2, self.hm_size, self.hm_size), dtype=np.float32)
        offset_mask = np.zeros((1, self.hm_size, self.hm_size), dtype=np.float32)

        gt_points_out = []

        for px, py in points_resized:
            fx = px / self.down_ratio
            fy = py / self.down_ratio
            ix = int(np.clip(np.floor(fx), 0, self.hm_size - 1))
            iy = int(np.clip(np.floor(fy), 0, self.hm_size - 1))

            draw_gaussian(heatmap[0], (ix, iy), self.gaussian_radius)
            offset[0, iy, ix] = fx - ix
            offset[1, iy, ix] = fy - iy
            offset_mask[0, iy, ix] = 1.0
            gt_points_out.append([float(px), float(py)])

        return {
            "image": torch.from_numpy(img_np.transpose(2, 0, 1)),
            "heatmap": torch.from_numpy(heatmap),
            "offset": torch.from_numpy(offset),
            "offset_mask": torch.from_numpy(offset_mask),
            "gt_points": gt_points_out,
            "image_path": str(sample.image_path),
            "orig_size": torch.tensor([float(orig_w), float(orig_h)], dtype=torch.float32),
            "is_positive": sample.is_positive,
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "heatmap": torch.stack([b["heatmap"] for b in batch], dim=0),
        "offset": torch.stack([b["offset"] for b in batch], dim=0),
        "offset_mask": torch.stack([b["offset_mask"] for b in batch], dim=0),
        "gt_points": [b["gt_points"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
        "orig_size": torch.stack([b["orig_size"] for b in batch], dim=0),
        "is_positive": [b["is_positive"] for b in batch],
    }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetPoint(nn.Module):
    def __init__(self, base_ch: int = 32, out_stride: int = 4) -> None:
        super().__init__()
        if out_stride not in {2, 4}:
            raise ValueError("out_stride must be 2 or 4")

        self.out_stride = out_stride

        self.inc = ConvBlock(3, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4)

        if out_stride == 2:
            self.up3 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
            head_ch = base_ch * 2
        else:
            self.up3 = None
            head_ch = base_ch * 4

        self.hm_head = nn.Sequential(
            nn.Conv2d(head_ch, head_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, 1, 1),
        )
        self.off_head = nn.Sequential(
            nn.Conv2d(head_ch, head_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, 2, 1),
        )

    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)

        if self.out_stride == 2:
            x = self.up3(x, x2)

        hm = self.hm_head(x)
        off = self.off_head(x)
        return hm, off


def weighted_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 30.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    prob = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)

    pos_mask = (target >= 0.999).float()
    neg_mask = (target < 0.999).float()

    pos_loss = -torch.log(prob) * torch.pow(1 - prob, 2) * pos_mask * pos_weight
    neg_weights = torch.pow(1 - target, 4)
    neg_loss = -torch.log(1 - prob) * torch.pow(prob, 2) * neg_weights * neg_mask * neg_weight

    denom = pos_mask.sum() + 1.0
    return (pos_loss.sum() + neg_loss.sum()) / denom


def offset_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target) * mask
    denom = mask.sum() * pred.shape[1] + 1e-6
    return diff.sum() / denom


def decode_points(hm_logits, off, score_threshold, topk, down_ratio, decoder_nms_kernel, min_distance_px):
    hm = torch.sigmoid(hm_logits)
    hm = max_pool_nms(hm, kernel=decoder_nms_kernel)

    b, c, h, w = hm.shape
    hm_flat = hm.view(b, -1)
    scores, inds = torch.topk(hm_flat, k=min(topk, hm_flat.shape[1]), dim=1)

    results = []
    for bi in range(b):
        preds = []
        for s, ind in zip(scores[bi], inds[bi]):
            score = float(s.item())
            if score < score_threshold:
                continue

            idx = int(ind.item())
            iy = idx // w
            ix = idx % w

            dx = float(off[bi, 0, iy, ix].item())
            dy = float(off[bi, 1, iy, ix].item())

            px = (ix + dx) * down_ratio
            py = (iy + dy) * down_ratio

            too_close = False
            for old in preds:
                if math.hypot(px - old["x"], py - old["y"]) < min_distance_px:
                    too_close = True
                    break
            if too_close:
                continue

            preds.append({"x": px, "y": py, "score": score})
        results.append(preds)

    return results


def save_debug_overlays(output_dir: Path, predictions_dump, max_images: int, img_size: int) -> None:
    dbg_dir = output_dir / "debug_overlays"
    ensure_dir(dbg_dir)

    for i, rec in enumerate(predictions_dump[:max_images]):
        image_path = Path(rec["image_path"])
        if not image_path.exists():
            continue

        orig_w, orig_h = rec["orig_size"]
        sx = orig_w / img_size
        sy = orig_h / img_size

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)

            for gi, gt in enumerate(rec["gt_points"]):
                x = gt[0] * sx
                y = gt[1] * sy
                r = 6
                color = (0, 255, 0) if gi in rec["matched_gt"] else (255, 255, 0)
                draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)

            for pi, pred in enumerate(rec["pred_points"]):
                x = pred["x"] * sx
                y = pred["y"] * sy
                r = 5
                color = (255, 0, 0) if pi not in rec["matched_pred"] else (0, 255, 255)
                draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)

            img.save(dbg_dir / f"{i:03d}_{image_path.name}")


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    thresholds,
    topk,
    match_radius_px,
    down_ratio,
    decoder_nms_kernel,
    min_distance_px,
    split_name,
    output_dir,
    save_overlays,
    max_overlay_images,
    selection_mode,
    selection_fp_weight,
    selection_precision_weight,
    show_progress: bool = True,
):
    model.eval()

    hm_losses = []
    off_losses = []
    predictions_by_thr = {t: [] for t in thresholds}
    max_scores = []

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            total=len(loader),
            desc=f"Eval {split_name}",
            dynamic_ncols=True,
            leave=False,
        )

    for batch in iterator:
        image = batch["image"].to(device)
        target_hm = batch["heatmap"].to(device)
        target_off = batch["offset"].to(device)
        off_mask = batch["offset_mask"].to(device)

        hm_logits, off = model(image)

        hm_loss = weighted_heatmap_loss(hm_logits, target_hm, pos_weight=30.0)
        off_loss = offset_l1_loss(off, target_off, off_mask)

        hm_losses.append(float(hm_loss.item()))
        off_losses.append(float(off_loss.item()))

        hm_prob = torch.sigmoid(hm_logits)
        max_scores.extend(hm_prob.amax(dim=(1, 2, 3)).detach().cpu().numpy().tolist())

        for thr in thresholds:
            decoded = decode_points(
                hm_logits,
                off,
                thr,
                topk,
                down_ratio,
                decoder_nms_kernel,
                min_distance_px,
            )
            for i in range(len(decoded)):
                predictions_by_thr[thr].append(
                    {
                        "image_path": batch["image_path"][i],
                        "gt_points": batch["gt_points"][i],
                        "pred_points": decoded[i],
                        "orig_size": batch["orig_size"][i].cpu().numpy().tolist(),
                    }
                )

    best_thr = thresholds[0]
    best_metrics = None
    best_dump = None
    best_selection_score = -1e18

    for thr in thresholds:
        tp = fp = fn = 0
        dists_all = []
        cm = np.zeros((2, 2), dtype=int)
        dump = []

        for rec in predictions_by_thr[thr]:
            tpi, fpi, fni, dists, matched_gt, matched_pred = greedy_match_points(
                rec["gt_points"], rec["pred_points"], match_radius_px
            )
            tp += tpi
            fp += fpi
            fn += fni
            dists_all.extend(dists)

            gt_has = int(len(rec["gt_points"]) > 0)
            pred_has = int(len(rec["pred_points"]) > 0)
            cm[gt_has, pred_has] += 1

            dump.append(
                {
                    "image_path": rec["image_path"],
                    "gt_points": rec["gt_points"],
                    "pred_points": rec["pred_points"],
                    "num_gt": len(rec["gt_points"]),
                    "num_pred": len(rec["pred_points"]),
                    "orig_size": rec["orig_size"],
                    "matched_gt": sorted(list(matched_gt)),
                    "matched_pred": sorted(list(matched_pred)),
                }
            )

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        mle = float(np.mean(dists_all)) if dists_all else None

        neg_fp_images = int(cm[0, 1])

        if selection_mode == "fp_aware":
            selection_score = f1 + selection_precision_weight * precision - selection_fp_weight * neg_fp_images
        else:
            selection_score = f1

        metrics = {
            "split": split_name,
            "point_level": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_localization_error_px": mle,
                "matched_points": len(dists_all),
            },
            "image_level_confusion_matrix": {
                "labels": {
                    "rows_gt": ["negative", "positive"],
                    "cols_pred": ["negative", "positive"],
                },
                "matrix": cm.tolist(),
            },
            "decode": {
                "score_threshold": thr,
                "topk": topk,
                "match_radius_px": match_radius_px,
                "down_ratio": down_ratio,
                "decoder_nms_kernel": decoder_nms_kernel,
                "min_distance_px": min_distance_px,
                "mean_max_heatmap_score": float(np.mean(max_scores)) if max_scores else None,
            },
            "loss": {
                "heatmap": float(np.mean(hm_losses)) if hm_losses else None,
                "offset": float(np.mean(off_losses)) if off_losses else None,
                "total": (float(np.mean(hm_losses)) if hm_losses else 0.0)
                + (float(np.mean(off_losses)) if off_losses else 0.0),
            },
            "selection": {
                "mode": selection_mode,
                "score": selection_score,
                "neg_fp_images": neg_fp_images,
                "fp_weight": selection_fp_weight,
                "precision_weight": selection_precision_weight,
            },
        }

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_thr = thr
            best_metrics = metrics
            best_dump = dump

    assert best_metrics is not None and best_dump is not None

    ensure_dir(output_dir)
    with (output_dir / f"metrics_{split_name}.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / f"predictions_{split_name}.json").open("w", encoding="utf-8") as f:
        json.dump(best_dump, f, ensure_ascii=False, indent=2)

    cm = np.array(best_metrics["image_level_confusion_matrix"]["matrix"], dtype=int)
    save_confusion_matrix(
        cm,
        output_dir / f"confusion_matrix_{split_name}.png",
        ["negative", "positive"],
    )

    if save_overlays:
        save_debug_overlays(output_dir, best_dump, max_overlay_images, loader.dataset.img_size)

    return best_metrics, best_thr, best_selection_score


def make_loader(dataset, batch_size, shuffle, num_workers, balanced_sampler):
    if balanced_sampler:
        weights = []
        pos_count = max(dataset.positive_count, 1)
        neg_count = max(dataset.negative_count, 1)
        for s in dataset.samples:
            weights.append(1.0 / pos_count if s.is_positive else 1.0 / neg_count)

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def save_checkpoint(path, model, optimizer, epoch, best_val_f1, best_threshold, best_selection_score):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_f1": float(best_val_f1),
        "best_threshold": float(best_threshold),
        "best_selection_score": float(best_selection_score),
    }
    torch.save(ckpt, path)


def load_best_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net heatmap point detector for AGL lamps.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--flip-aug", action="store_true")
    p.add_argument("--save-debug-overlays", action="store_true")
    p.add_argument("--max-overlay-images", type=int, default=24)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--down-ratio", type=int, default=4)
    p.add_argument("--gaussian-radius", type=int, default=2)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--hm-pos-weight", type=float, default=30.0)
    p.add_argument("--off-loss-weight", type=float, default=0.25)
    p.add_argument("--match-radius-px", type=float, default=12.0)

    p.add_argument("--decoder-topk", type=int, default=24)
    p.add_argument("--decoder-nms-kernel", type=int, default=7)
    p.add_argument("--min-distance-px", type=float, default=16.0)
    p.add_argument("--eval-thresholds", type=str, default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")

    p.add_argument("--balanced-sampler", action="store_true")

    p.add_argument("--selection-mode", type=str, default="fp_aware", choices=["f1", "fp_aware"])
    p.add_argument("--selection-fp-weight", type=float, default=0.10)
    p.add_argument("--selection-precision-weight", type=float, default=0.10)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.output_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise SystemExit(f"Output dir exists: {args.output_dir}. Use --overwrite.")

    ensure_dir(args.output_dir)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    train_ds = PointHeatmapDataset(
        args.dataset_root / "train",
        args.img_size,
        args.down_ratio,
        args.gaussian_radius,
        flip_aug=args.flip_aug,
    )
    val_ds = PointHeatmapDataset(
        args.dataset_root / "val",
        args.img_size,
        args.down_ratio,
        args.gaussian_radius,
        flip_aug=False,
    )
    test_ds = PointHeatmapDataset(
        args.dataset_root / "test",
        args.img_size,
        args.down_ratio,
        args.gaussian_radius,
        flip_aug=False,
    )

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, args.balanced_sampler)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, False)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, False)

    model = UNetPoint(base_ch=args.base_ch, out_stride=args.down_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    thresholds = parse_thresholds(args.eval_thresholds)

    print(f"Device: {device}")
    print(f"Train images: {len(train_ds)} | positive: {train_ds.positive_count} | negative: {train_ds.negative_count}")
    print(f"Val images:   {len(val_ds)} | positive: {val_ds.positive_count} | negative: {val_ds.negative_count}")
    print(f"Test images:  {len(test_ds)} | positive: {test_ds.positive_count} | negative: {test_ds.negative_count}")
    print(f"Model: U-Net | down_ratio={args.down_ratio} | base_ch={args.base_ch}")
    print(f"Eval thresholds: {thresholds}")
    print(f"Decoder: topk={args.decoder_topk}, nms_kernel={args.decoder_nms_kernel}, min_distance_px={args.min_distance_px}")
    print(f"Loss weights: hm_pos={args.hm_pos_weight}, off={args.off_loss_weight}")
    print(f"Balanced sampler: {args.balanced_sampler}")

    history = []
    best_val_f1 = -1.0
    best_selection_score = -1e18
    best_thr = thresholds[0]

    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_hm_losses = []
        train_off_losses = []

        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch:03d}/{args.epochs}",
            dynamic_ncols=True,
            leave=False,
        )

        for step, batch in enumerate(pbar, start=1):
            image = batch["image"].to(device)
            target_hm = batch["heatmap"].to(device)
            target_off = batch["offset"].to(device)
            off_mask = batch["offset_mask"].to(device)

            hm_logits, off = model(image)

            hm_loss = weighted_heatmap_loss(hm_logits, target_hm, pos_weight=args.hm_pos_weight)
            off_loss = offset_l1_loss(off, target_off, off_mask)
            loss = hm_loss + args.off_loss_weight * off_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_hm_losses.append(float(hm_loss.item()))
            train_off_losses.append(float(off_loss.item()))

            if step % 5 == 0 or step == len(train_loader):
                avg_loss = sum(train_losses[-20:]) / min(len(train_losses), 20)
                avg_hm = sum(train_hm_losses[-20:]) / min(len(train_hm_losses), 20)
                avg_off = sum(train_off_losses[-20:]) / min(len(train_off_losses), 20)
                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    hm=f"{avg_hm:.4f}",
                    off=f"{avg_off:.4f}",
                    lr=f"{lr_now:.2e}",
                )

        val_output_dir = args.output_dir / "val_eval_latest"
        val_metrics, val_thr, val_selection_score = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            thresholds=thresholds,
            topk=args.decoder_topk,
            match_radius_px=args.match_radius_px,
            down_ratio=args.down_ratio,
            decoder_nms_kernel=args.decoder_nms_kernel,
            min_distance_px=args.min_distance_px,
            split_name="val",
            output_dir=val_output_dir,
            save_overlays=False,
            max_overlay_images=args.max_overlay_images,
            selection_mode=args.selection_mode,
            selection_fp_weight=args.selection_fp_weight,
            selection_precision_weight=args.selection_precision_weight,
            show_progress=True,
        )

        val_f1 = val_metrics["point_level"]["f1"]
        val_prec = val_metrics["point_level"]["precision"]
        val_rec = val_metrics["point_level"]["recall"]
        val_loss_total = val_metrics["loss"]["total"]
        val_maxscore = val_metrics["decode"]["mean_max_heatmap_score"]

        hist_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "train_loss_heatmap": float(np.mean(train_hm_losses)) if train_hm_losses else None,
            "train_loss_offset": float(np.mean(train_off_losses)) if train_off_losses else None,
            "val_loss": val_loss_total,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_threshold": val_thr,
            "val_maxscore": val_maxscore,
            "selection_score": val_selection_score,
        }
        history.append(hist_row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={hist_row['train_loss']:.4f} "
            f"val_loss={val_loss_total:.4f} "
            f"thr={val_thr:.3f} "
            f"val_f1={val_f1:.4f} "
            f"val_prec={val_prec:.4f} "
            f"val_rec={val_rec:.4f} "
            f"val_maxscore={val_maxscore:.4f} "
            f"sel={val_selection_score:.4f}"
        )

        save_checkpoint(last_path, model, optimizer, epoch, best_val_f1, best_thr, best_selection_score)

        if val_selection_score > best_selection_score:
            best_selection_score = val_selection_score
            best_val_f1 = val_f1
            best_thr = val_thr
            save_checkpoint(best_path, model, optimizer, epoch, best_val_f1, best_thr, best_selection_score)

        with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best threshold: {best_thr:.3f}")

    ckpt = load_best_checkpoint(best_path, model, device)
    best_thr = float(ckpt.get("best_threshold", best_thr))

    test_output_dir = args.output_dir / "test_eval_best"
    test_metrics, _, _ = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        thresholds=[best_thr],
        topk=args.decoder_topk,
        match_radius_px=args.match_radius_px,
        down_ratio=args.down_ratio,
        decoder_nms_kernel=args.decoder_nms_kernel,
        min_distance_px=args.min_distance_px,
        split_name="test",
        output_dir=test_output_dir,
        save_overlays=args.save_debug_overlays,
        max_overlay_images=args.max_overlay_images,
        selection_mode=args.selection_mode,
        selection_fp_weight=args.selection_fp_weight,
        selection_precision_weight=args.selection_precision_weight,
        show_progress=True,
    )

    summary = {
        "model": {
            "name": "UNet-Point",
            "base_ch": args.base_ch,
            "down_ratio": args.down_ratio,
            "gaussian_radius": args.gaussian_radius,
        },
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "decoder": {
            "thresholds": thresholds,
            "best_threshold": best_thr,
            "topk": args.decoder_topk,
            "decoder_nms_kernel": args.decoder_nms_kernel,
            "min_distance_px": args.min_distance_px,
            "match_radius_px": args.match_radius_px,
        },
        "loss_weights": {
            "hm_pos_weight": args.hm_pos_weight,
            "off_loss_weight": args.off_loss_weight,
        },
        "balanced_sampler": args.balanced_sampler,
        "selection": {
            "mode": args.selection_mode,
            "selection_fp_weight": args.selection_fp_weight,
            "selection_precision_weight": args.selection_precision_weight,
            "best_selection_score": best_selection_score,
        },
        "best_val_f1": best_val_f1,
        "final_test_f1": test_metrics["point_level"]["f1"],
        "final_test_precision": test_metrics["point_level"]["precision"],
        "final_test_recall": test_metrics["point_level"]["recall"],
    }

    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Final metrics saved to: {test_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())