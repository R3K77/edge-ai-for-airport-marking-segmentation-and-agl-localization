#!/usr/bin/env python3
"""
GUI-assisted exporter for test image set predictions.

Workflow:
1. Open GUI.
2. Choose:
   - input folder with test images (or test root containing "images")
   - HRNet checkpoint
   - U-Net checkpoint
   - output folder
   - detection parameters
3. Load first image and optionally draw ROI with mouse.
4. Click "Start export".
5. Script saves results separately for HRNet and U-Net.

Saved outputs:
  output_root/
    HRNet/
      combined/
    UNet/
      combined/

Each combined image is saved as a PNG file in full input-image resolution for each panel.
The layout is vertical:
  top    = detections overlay
  bottom = heatmap

If ROI is not selected, the whole image is used.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MODELS
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, 3, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        out = self.act(out)
        return out


class HRFuse2(nn.Module):
    def __init__(self, ch_high: int, ch_low: int) -> None:
        super().__init__()
        self.low_to_high = nn.Sequential(
            nn.Conv2d(ch_low, ch_high, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_high),
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(ch_high, ch_low, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_low),
        )
        self.act_high = nn.ReLU(inplace=True)
        self.act_low = nn.ReLU(inplace=True)

    def forward(self, xh: torch.Tensor, xl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xl_up = F.interpolate(self.low_to_high(xl), size=xh.shape[-2:], mode="bilinear", align_corners=False)
        xh_down = self.high_to_low(xh)
        xh = self.act_high(xh + xl_up)
        xl = self.act_low(xl + xh_down)
        return xh, xl


class HRStage2(nn.Module):
    def __init__(self, ch_high: int, ch_low: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.high_blocks = nn.Sequential(*[BasicBlock(ch_high) for _ in range(num_blocks)])
        self.low_blocks = nn.Sequential(*[BasicBlock(ch_low) for _ in range(num_blocks)])
        self.fuse = HRFuse2(ch_high, ch_low)

    def forward(self, xh: torch.Tensor, xl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xh = self.high_blocks(xh)
        xl = self.low_blocks(xl)
        return self.fuse(xh, xl)


class HRNetLitePoint(nn.Module):
    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, base_ch, 3, 2),
            ConvBNAct(base_ch, base_ch, 3, 2),
            BasicBlock(base_ch),
            BasicBlock(base_ch),
        )
        self.to_low = ConvBNAct(base_ch, base_ch * 2, 3, 2)
        self.stage1 = HRStage2(base_ch, base_ch * 2, num_blocks=2)
        self.stage2 = HRStage2(base_ch, base_ch * 2, num_blocks=2)
        self.stage3 = HRStage2(base_ch, base_ch * 2, num_blocks=2)

        self.head_pre = nn.Sequential(
            ConvBNAct(base_ch + base_ch * 2, base_ch * 2, 3, 1),
            BasicBlock(base_ch * 2),
        )
        self.hm_head = nn.Sequential(
            ConvBNAct(base_ch * 2, base_ch, 3, 1),
            nn.Conv2d(base_ch, 1, kernel_size=1, stride=1, padding=0),
        )
        self.off_head = nn.Sequential(
            ConvBNAct(base_ch * 2, base_ch, 3, 1),
            nn.Conv2d(base_ch, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xh = self.stem(x)
        xl = self.to_low(xh)
        xh, xl = self.stage1(xh, xl)
        xh, xl = self.stage2(xh, xl)
        xh, xl = self.stage3(xh, xl)
        xl_up = F.interpolate(xl, size=xh.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.head_pre(torch.cat([xh, xl_up], dim=1))
        hm = self.hm_head(feat)
        off = self.off_head(feat)
        return hm, off


class UNetConvBlock(nn.Module):
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


class UNetDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = UNetConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UNetUp(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = UNetConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetPoint(nn.Module):
    def __init__(self, base_ch: int = 16, out_stride: int = 4) -> None:
        super().__init__()
        if out_stride not in {2, 4}:
            raise ValueError("out_stride must be 2 or 4")

        self.out_stride = out_stride
        self.inc = UNetConvBlock(3, base_ch)
        self.down1 = UNetDown(base_ch, base_ch * 2)
        self.down2 = UNetDown(base_ch * 2, base_ch * 4)
        self.down3 = UNetDown(base_ch * 4, base_ch * 8)
        self.down4 = UNetDown(base_ch * 8, base_ch * 16)
        self.up1 = UNetUp(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up2 = UNetUp(base_ch * 8, base_ch * 4, base_ch * 4)

        if out_stride == 2:
            self.up3 = UNetUp(base_ch * 4, base_ch * 2, base_ch * 2)
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


# ============================================================
# HELPERS
# ============================================================

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(folder: Path) -> list[Path]:
    folder = folder.expanduser().resolve()
    images_dir = folder / "images"
    search_dir = images_dir if images_dir.is_dir() else folder
    return sorted([p for p in search_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_first_image_from_folder(folder: Path) -> tuple[np.ndarray, int, int, Path]:
    images = collect_images(folder)
    if not images:
        raise RuntimeError("Nie znaleziono obrazów w wybranym folderze.")
    first = images[0]
    frame = cv2.imread(str(first))
    if frame is None:
        raise RuntimeError(f"Nie udało się odczytać obrazu: {first}")
    h, w = frame.shape[:2]
    return frame, w, h, first


def preprocess_rgb(rgb: np.ndarray, img_size: int) -> torch.Tensor:
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def max_pool_nms(hm: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(hm, kernel_size=kernel, stride=1, padding=pad)
    keep = (pooled == hm).float()
    return hm * keep


def decode_points(hm_logits: torch.Tensor, off: torch.Tensor, score_threshold: float, topk: int, down_ratio: int, decoder_nms_kernel: int, min_distance_px: float) -> list[dict[str, float]]:
    hm = torch.sigmoid(hm_logits)
    hm = max_pool_nms(hm, kernel=decoder_nms_kernel)
    b, c, h, w = hm.shape
    assert b == 1 and c == 1

    hm_flat = hm.view(1, -1)
    k = min(topk, hm_flat.shape[1])
    scores, inds = torch.topk(hm_flat, k=k, dim=1)

    preds: list[dict[str, float]] = []
    for s, ind in zip(scores[0], inds[0]):
        score = float(s.item())
        if score < score_threshold:
            continue
        idx = int(ind.item())
        iy = idx // w
        ix = idx % w
        dx = float(off[0, 0, iy, ix].item())
        dy = float(off[0, 1, iy, ix].item())
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
    return preds


def make_heatmap_preview(hm_logits: torch.Tensor, out_w: int, out_h: int, show_nms: bool, nms_kernel: int) -> np.ndarray:
    hm = torch.sigmoid(hm_logits)
    if show_nms:
        hm = max_pool_nms(hm, kernel=nms_kernel)
    hm_np = hm[0, 0].detach().cpu().numpy()
    hm_np = np.clip(hm_np, 0.0, 1.0)
    hm_u8 = (hm_np * 255.0).astype(np.uint8)
    hm_u8 = cv2.resize(hm_u8, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    max_score = float(hm_np.max()) if hm_np.size else 0.0
    cv2.putText(hm_color, f"heatmap max={max_score:.3f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return hm_color


def clamp_roi(roi: Optional[tuple[int, int, int, int]], w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return (x1, y1, x2, y2)


def draw_predictions(vis: np.ndarray, preds: list[dict[str, float]], color=(0, 0, 255)) -> None:
    for p in preds:
        x0 = int(round(p["x"]))
        y0 = int(round(p["y"]))
        cv2.circle(vis, (x0, y0), 8, color, 2)
        cv2.putText(vis, f"{p['score']:.2f}", (x0 + 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


@dataclass
class DetectorConfig:
    img_size: int
    down_ratio: int
    threshold: float
    topk: int
    nms_kernel: int
    min_distance_px: float
    show_nms_heatmap: bool = False


class PointDetectorWrapper:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model: Optional[nn.Module] = None
        self.model_name: str = "?"
        self.base_ch: int = 0
        self.config: Optional[DetectorConfig] = None

    def _infer_model_type(self, state_dict: dict[str, torch.Tensor]) -> str:
        keys = list(state_dict.keys())
        if any(k.startswith("stem.") for k in keys):
            return "hrnet"
        if any(k.startswith("inc.") for k in keys):
            return "unet"
        raise RuntimeError("Nie udało się automatycznie rozpoznać architektury.")

    def _infer_base_ch(self, state_dict: dict[str, torch.Tensor], model_type: str) -> int:
        if model_type == "hrnet":
            return int(state_dict["stem.0.block.0.weight"].shape[0])
        if model_type == "unet":
            return int(state_dict["inc.block.0.weight"].shape[0])
        raise RuntimeError("Nieznany typ modelu")

    def _infer_unet_out_stride(self, state_dict: dict[str, torch.Tensor]) -> int:
        if any(k.startswith("up3.") for k in state_dict.keys()):
            return 2
        return 4

    def load_checkpoint(self, checkpoint_path: str | Path, model_choice: str, base_ch_override: Optional[int], down_ratio_override: Optional[int], config: DetectorConfig) -> None:
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        model_type = self._infer_model_type(state_dict) if model_choice == "auto" else model_choice.lower()
        base_ch = self._infer_base_ch(state_dict, model_type) if not base_ch_override else int(base_ch_override)

        if model_type == "hrnet":
            model = HRNetLitePoint(base_ch=base_ch)
            down_ratio = 4 if not down_ratio_override else int(down_ratio_override)
        elif model_type == "unet":
            down_ratio = self._infer_unet_out_stride(state_dict) if not down_ratio_override else int(down_ratio_override)
            model = UNetPoint(base_ch=base_ch, out_stride=down_ratio)
        else:
            raise RuntimeError(f"Nieobsługiwany model: {model_type}")

        model = model.to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self.model_name = model_type
        self.base_ch = base_ch
        self.config = DetectorConfig(config.img_size, down_ratio, config.threshold, config.topk, config.nms_kernel, config.min_distance_px, config.show_nms_heatmap)

    @torch.no_grad()
    def infer_crop(self, crop_bgr: np.ndarray, global_offset_xy: tuple[int, int]) -> tuple[np.ndarray, list[dict[str, float]]]:
        if self.model is None or self.config is None:
            raise RuntimeError("Model nie został wczytany.")

        crop_h, crop_w = crop_bgr.shape[:2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = preprocess_rgb(crop_rgb, self.config.img_size).to(self.device)
        hm_logits, off = self.model(x)

        preds_local = decode_points(hm_logits, off, self.config.threshold, self.config.topk, self.config.down_ratio, self.config.nms_kernel, self.config.min_distance_px)

        sx = crop_w / self.config.img_size
        sy = crop_h / self.config.img_size
        ox, oy = global_offset_xy
        preds_global: list[dict[str, float]] = []
        for p in preds_local:
            preds_global.append({"x": p["x"] * sx + ox, "y": p["y"] * sy + oy, "score": p["score"]})

        hm_vis = make_heatmap_preview(hm_logits, crop_w, crop_h, self.config.show_nms_heatmap, self.config.nms_kernel)
        return hm_vis, preds_global


# ============================================================
# EXPORT
# ============================================================

def ensure_dirs(root: Path, model_label: str) -> dict[str, Path]:
    paths = {
        "root": root / model_label,
        "combined": root / model_label / "combined",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def export_image_folder(*, input_folder: Path, hrnet: PointDetectorWrapper, unet: PointDetectorWrapper, output_root: Path, roi: Optional[tuple[int, int, int, int]]) -> None:
    images = collect_images(input_folder)
    if not images:
        raise RuntimeError("Nie znaleziono obrazów do przetworzenia.")

    hr_dirs = ensure_dirs(output_root, "HRNet")
    un_dirs = ensure_dirs(output_root, "UNet")

    print(f"Znaleziono {len(images)} obrazów.")
    if roi is not None:
        print(f"ROI: {roi}")

    t0 = time.perf_counter()
    for idx, image_path in enumerate(images, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Pominięto: {image_path}")
            continue

        h, w = frame.shape[:2]
        roi_used = clamp_roi(roi, w, h)
        if roi_used is None:
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1, y1, x2, y2 = roi_used

        crop_bgr = frame[y1:y2, x1:x2]
        hr_heat_crop, hr_preds = hrnet.infer_crop(crop_bgr, (x1, y1))
        un_heat_crop, un_preds = unet.infer_crop(crop_bgr, (x1, y1))

        hr_overlay = frame.copy()
        un_overlay = frame.copy()
        if roi_used is not None:
            cv2.rectangle(hr_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(un_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        draw_predictions(hr_overlay, hr_preds)
        draw_predictions(un_overlay, un_preds)

        hr_heat_full = np.zeros_like(frame)
        un_heat_full = np.zeros_like(frame)
        hr_heat_full[y1:y2, x1:x2] = hr_heat_crop
        un_heat_full[y1:y2, x1:x2] = un_heat_crop
        if roi_used is not None:
            cv2.rectangle(hr_heat_full, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(un_heat_full, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(hr_overlay, f"HRNet detections ({len(hr_preds)})", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hr_heat_full, "HRNet heatmap", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(un_overlay, f"U-Net detections ({len(un_preds)})", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(un_heat_full, "U-Net heatmap", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        stem = image_path.stem

        # Combined output in full per-panel resolution: vertical stack preserves full image width.
        hr_combined = np.vstack([hr_overlay, hr_heat_full])
        un_combined = np.vstack([un_overlay, un_heat_full])

        # Always save as PNG to avoid lossy JPEG compression.
        cv2.imwrite(str(hr_dirs["combined"] / f"{stem}_hrnet_combined.png"), hr_combined)
        cv2.imwrite(str(un_dirs["combined"] / f"{stem}_unet_combined.png"), un_combined)

        if idx % 10 == 0 or idx == len(images):
            elapsed = time.perf_counter() - t0
            print(f"Przetworzono {idx}/{len(images)} obrazów | czas: {elapsed:.1f} s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nGotowe. Wyniki zapisano w: {output_root}")
    print(f"Liczba obrazów: {len(images)}")
    print(f"Czas całkowity: {elapsed:.1f} s")


# ============================================================
# GUI
# ============================================================

@dataclass
class ExportSettings:
    input_folder: Path
    hrnet_ckpt: Path
    unet_ckpt: Path
    output_folder: Path
    roi: Optional[tuple[int, int, int, int]]
    hr_model: str
    un_model: str
    hr_base_ch: Optional[int]
    hr_down_ratio: Optional[int]
    un_base_ch: Optional[int]
    un_down_ratio: Optional[int]
    img_size: int
    hr_threshold: float
    hr_topk: int
    hr_min_distance: float
    hr_nms_kernel: int
    un_threshold: float
    un_topk: int
    un_min_distance: float
    un_nms_kernel: int


class ConfigApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Eksport wyników dla zbioru testowego")
        self.root.geometry("1500x950")
        self.result: Optional[ExportSettings] = None

        self.input_var = tk.StringVar()
        self.hrnet_var = tk.StringVar()
        self.unet_var = tk.StringVar()
        self.output_var = tk.StringVar()

        self.hr_model_var = tk.StringVar(value="auto")
        self.un_model_var = tk.StringVar(value="auto")
        self.hr_base_ch_var = tk.IntVar(value=0)
        self.hr_down_ratio_var = tk.IntVar(value=0)
        self.un_base_ch_var = tk.IntVar(value=0)
        self.un_down_ratio_var = tk.IntVar(value=0)

        self.img_size_var = tk.IntVar(value=1024)
        self.hr_threshold_var = tk.DoubleVar(value=0.50)
        self.hr_topk_var = tk.IntVar(value=24)
        self.hr_min_dist_var = tk.DoubleVar(value=16.0)
        self.hr_nms_var = tk.IntVar(value=7)
        self.un_threshold_var = tk.DoubleVar(value=0.50)
        self.un_topk_var = tk.IntVar(value=24)
        self.un_min_dist_var = tk.DoubleVar(value=16.0)
        self.un_nms_var = tk.IntVar(value=7)

        self.first_frame_bgr: Optional[np.ndarray] = None
        self.first_frame_w = 0
        self.first_frame_h = 0
        self.roi_xyxy: Optional[tuple[int, int, int, int]] = None
        self.last_photo = None
        self.display_scale = 1.0
        self.display_x0 = 0
        self.display_y0 = 0
        self.display_w = 0
        self.display_h = 0
        self.roi_start_xy: Optional[tuple[int, int]] = None
        self.roi_preview_rect_id: Optional[int] = None

        self._build_ui()
        self._bind_canvas()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main, width=360)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Pliki i foldery", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self._path_row(left, "Folder testowy", self.input_var, self.pick_input_folder)
        self._path_row(left, "HRNet .pt", self.hrnet_var, self.pick_hrnet_file)
        self._path_row(left, "U-Net .pt", self.unet_var, self.pick_unet_file)
        self._path_row(left, "Folder wyjściowy", self.output_var, self.pick_output_folder)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        hr_box = ttk.LabelFrame(left, text="HRNet")
        hr_box.pack(fill=tk.X, pady=4)
        ttk.Combobox(hr_box, textvariable=self.hr_model_var, values=["auto", "hrnet"], state="readonly").pack(fill=tk.X, pady=2)
        r = ttk.Frame(hr_box); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="base_ch").pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self.hr_base_ch_var, width=8).pack(side=tk.RIGHT)
        r = ttk.Frame(hr_box); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="down_ratio").pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self.hr_down_ratio_var, width=8).pack(side=tk.RIGHT)

        self._scale(left, "Input size", self.img_size_var, 256, 1536)
        self._scale(left, "HR threshold", self.hr_threshold_var, 0.01, 0.90)
        self._scale(left, "HR Top-K", self.hr_topk_var, 1, 100)
        self._scale(left, "HR min distance", self.hr_min_dist_var, 1, 50)
        self._scale(left, "HR NMS kernel", self.hr_nms_var, 3, 11)

        un_box = ttk.LabelFrame(left, text="U-Net")
        un_box.pack(fill=tk.X, pady=4)
        ttk.Combobox(un_box, textvariable=self.un_model_var, values=["auto", "unet"], state="readonly").pack(fill=tk.X, pady=2)
        r = ttk.Frame(un_box); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="base_ch").pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self.un_base_ch_var, width=8).pack(side=tk.RIGHT)
        r = ttk.Frame(un_box); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="down_ratio").pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self.un_down_ratio_var, width=8).pack(side=tk.RIGHT)

        self._scale(left, "UN threshold", self.un_threshold_var, 0.01, 0.90)
        self._scale(left, "UN Top-K", self.un_topk_var, 1, 100)
        self._scale(left, "UN min distance", self.un_min_dist_var, 1, 50)
        self._scale(left, "UN NMS kernel", self.un_nms_var, 3, 11)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(left, text="Wczytaj pierwszy obraz", command=self.load_first_frame_action).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Wyczyść ROI", command=self.clear_roi).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Start export", command=self.start_export).pack(fill=tk.X, pady=(12, 2))

        self.info_var = tk.StringVar(value="1) wybierz folder testowy\n2) wybierz checkpointy\n3) wczytaj pierwszy obraz\n4) zaznacz ROI (opcjonalnie)\n5) Start export")
        ttk.Label(left, textvariable=self.info_var, wraplength=330).pack(anchor="w", pady=(10, 0))

        self.canvas = tk.Canvas(right, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _path_row(self, parent, label, variable, command):
        box = ttk.LabelFrame(parent, text=label)
        box.pack(fill=tk.X, pady=3)
        ttk.Entry(box, textvariable=variable).pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(box, text="Wybierz", command=command).pack(fill=tk.X, padx=4, pady=(0, 4))

    def _scale(self, parent, label, variable, from_, to):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text=label).pack(anchor="w")
        ttk.Scale(frame, orient=tk.HORIZONTAL, variable=variable, from_=from_, to=to).pack(fill=tk.X)
        ttk.Label(frame, textvariable=variable).pack(anchor="e")

    def _bind_canvas(self):
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def pick_input_folder(self):
        path = filedialog.askdirectory(title="Wybierz folder testowy")
        if path:
            self.input_var.set(path)

    def pick_output_folder(self):
        path = filedialog.askdirectory(title="Wybierz folder wyjściowy")
        if path:
            self.output_var.set(path)

    def pick_hrnet_file(self):
        path = filedialog.askopenfilename(title="Wybierz checkpoint HRNet", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.hrnet_var.set(path)

    def pick_unet_file(self):
        path = filedialog.askopenfilename(title="Wybierz checkpoint U-Net", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.unet_var.set(path)

    def load_first_frame_action(self):
        folder = self.input_var.get().strip()
        if not folder:
            messagebox.showinfo("Info", "Najpierw wybierz folder testowy.")
            return
        try:
            frame, w, h, first_path = load_first_image_from_folder(Path(folder))
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
            return
        self.first_frame_bgr = frame
        self.first_frame_w = w
        self.first_frame_h = h
        self.roi_xyxy = None
        self.info_var.set(f"Pierwszy obraz: {first_path.name}")
        self.show_first_frame()

    def clear_roi(self):
        self.roi_xyxy = None
        self.roi_start_xy = None
        if self.roi_preview_rect_id is not None:
            self.canvas.delete(self.roi_preview_rect_id)
            self.roi_preview_rect_id = None
        if self.first_frame_bgr is not None:
            self.show_first_frame()

    def show_first_frame(self):
        if self.first_frame_bgr is None:
            return
        vis = self.first_frame_bgr.copy()
        if self.roi_xyxy is not None:
            x1, y1, x2, y2 = self.roi_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, "Zaznacz ROI myszą. Bez ROI = cały obraz.", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.display_scale = scale
        self.display_x0 = (canvas_w - new_w) // 2
        self.display_y0 = (canvas_h - new_h) // 2
        self.display_w = new_w
        self.display_h = new_h
        self.last_photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image(self.display_x0, self.display_y0, anchor=tk.NW, image=self.last_photo)

        if self.roi_xyxy is not None:
            x1, y1, x2, y2 = self.roi_xyxy
            cx1 = self.display_x0 + int(round(x1 * self.display_scale))
            cy1 = self.display_y0 + int(round(y1 * self.display_scale))
            cx2 = self.display_x0 + int(round(x2 * self.display_scale))
            cy2 = self.display_y0 + int(round(y2 * self.display_scale))
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="yellow", width=2)

    def _canvas_to_frame_xy(self, cx: int, cy: int) -> Optional[tuple[int, int]]:
        if self.first_frame_bgr is None:
            return None
        if cx < self.display_x0 or cy < self.display_y0 or cx >= self.display_x0 + self.display_w or cy >= self.display_y0 + self.display_h:
            return None
        local_x = cx - self.display_x0
        local_y = cy - self.display_y0
        fx = int(round(local_x / max(self.display_scale, 1e-6)))
        fy = int(round(local_y / max(self.display_scale, 1e-6)))
        fx = max(0, min(self.first_frame_w - 1, fx))
        fy = max(0, min(self.first_frame_h - 1, fy))
        return fx, fy

    def on_canvas_press(self, event):
        pt = self._canvas_to_frame_xy(event.x, event.y)
        if pt is not None:
            self.roi_start_xy = pt

    def on_canvas_drag(self, event):
        if self.roi_start_xy is None:
            return
        pt = self._canvas_to_frame_xy(event.x, event.y)
        if pt is None:
            return
        x1, y1 = self.roi_start_xy
        x2, y2 = pt
        cx1 = self.display_x0 + int(round(min(x1, x2) * self.display_scale))
        cy1 = self.display_y0 + int(round(min(y1, y2) * self.display_scale))
        cx2 = self.display_x0 + int(round(max(x1, x2) * self.display_scale))
        cy2 = self.display_y0 + int(round(max(y1, y2) * self.display_scale))
        if self.roi_preview_rect_id is not None:
            self.canvas.delete(self.roi_preview_rect_id)
        self.roi_preview_rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="yellow", width=2, dash=(4, 2))

    def on_canvas_release(self, event):
        if self.roi_start_xy is None:
            return
        pt = self._canvas_to_frame_xy(event.x, event.y)
        if pt is None:
            return
        x1, y1 = self.roi_start_xy
        x2, y2 = pt
        roi = clamp_roi((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), self.first_frame_w, self.first_frame_h)
        self.roi_start_xy = None
        if self.roi_preview_rect_id is not None:
            self.canvas.delete(self.roi_preview_rect_id)
            self.roi_preview_rect_id = None
        if roi is None:
            self.info_var.set("ROI odrzucone: zbyt mały obszar.")
            return
        self.roi_xyxy = roi
        self.info_var.set(f"ROI: {roi}")
        self.show_first_frame()

    def start_export(self):
        input_folder = Path(self.input_var.get().strip())
        hrnet_ckpt = Path(self.hrnet_var.get().strip())
        unet_ckpt = Path(self.unet_var.get().strip())
        output_folder = Path(self.output_var.get().strip())

        missing = []
        if not input_folder.exists():
            missing.append("folder testowy")
        if not hrnet_ckpt.exists():
            missing.append("HRNet checkpoint")
        if not unet_ckpt.exists():
            missing.append("U-Net checkpoint")
        if not output_folder.exists():
            missing.append("folder wyjściowy")
        if missing:
            messagebox.showerror("Błąd", "Brakuje: " + ", ".join(missing))
            return

        self.result = ExportSettings(
            input_folder=input_folder,
            hrnet_ckpt=hrnet_ckpt,
            unet_ckpt=unet_ckpt,
            output_folder=output_folder,
            roi=self.roi_xyxy,
            hr_model=self.hr_model_var.get(),
            un_model=self.un_model_var.get(),
            hr_base_ch=self.hr_base_ch_var.get() if self.hr_base_ch_var.get() > 0 else None,
            hr_down_ratio=self.hr_down_ratio_var.get() if self.hr_down_ratio_var.get() > 0 else None,
            un_base_ch=self.un_base_ch_var.get() if self.un_base_ch_var.get() > 0 else None,
            un_down_ratio=self.un_down_ratio_var.get() if self.un_down_ratio_var.get() > 0 else None,
            img_size=int(self.img_size_var.get()),
            hr_threshold=float(self.hr_threshold_var.get()),
            hr_topk=int(self.hr_topk_var.get()),
            hr_min_distance=float(self.hr_min_dist_var.get()),
            hr_nms_kernel=int(self.hr_nms_var.get()),
            un_threshold=float(self.un_threshold_var.get()),
            un_topk=int(self.un_topk_var.get()),
            un_min_distance=float(self.un_min_dist_var.get()),
            un_nms_kernel=int(self.un_nms_var.get()),
        )
        self.root.destroy()


def run_gui() -> Optional[ExportSettings]:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    app = ConfigApp(root)
    root.mainloop()
    return app.result


def main() -> int:
    settings = run_gui()
    if settings is None:
        print("Anulowano.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hr_cfg = DetectorConfig(settings.img_size, 4, settings.hr_threshold, settings.hr_topk, settings.hr_nms_kernel if settings.hr_nms_kernel % 2 == 1 else settings.hr_nms_kernel + 1, settings.hr_min_distance, False)
    un_cfg = DetectorConfig(settings.img_size, 4, settings.un_threshold, settings.un_topk, settings.un_nms_kernel if settings.un_nms_kernel % 2 == 1 else settings.un_nms_kernel + 1, settings.un_min_distance, False)

    hrnet = PointDetectorWrapper(device)
    unet = PointDetectorWrapper(device)
    print("Wczytywanie modelu HRNet...")
    hrnet.load_checkpoint(settings.hrnet_ckpt, settings.hr_model, settings.hr_base_ch, settings.hr_down_ratio, hr_cfg)
    print(f"HRNet | arch={hrnet.model_name} | base_ch={hrnet.base_ch} | down_ratio={hrnet.config.down_ratio}")
    print("Wczytywanie modelu U-Net...")
    unet.load_checkpoint(settings.unet_ckpt, settings.un_model, settings.un_base_ch, settings.un_down_ratio, un_cfg)
    print(f"U-Net | arch={unet.model_name} | base_ch={unet.base_ch} | down_ratio={unet.config.down_ratio}")

    print("Start eksportu...")
    export_image_folder(input_folder=settings.input_folder, hrnet=hrnet, unet=unet, output_root=settings.output_folder, roi=settings.roi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
