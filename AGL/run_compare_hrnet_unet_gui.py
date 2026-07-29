#!/usr/bin/env python3
"""
Compare HRNet-Lite and U-Net on the same video/frame/ROI.

Layout:
  top-left     = HRNet detections
  top-right    = HRNet heatmap
  bottom-left  = U-Net detections
  bottom-right = U-Net heatmap

Requirements:
    pip install torch opencv-python pillow numpy
And ffmpeg + ffprobe available in PATH.
"""

from __future__ import annotations

import json
import math
import queue
import subprocess
import threading
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


# ---------------- HRNet-Lite ----------------

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


# ---------------- U-Net ----------------

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
# VIDEO
# ============================================================

def ffprobe_video(video_path: str) -> tuple[int, int, float, int | None, float | None]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]

    width = int(stream["width"])
    height = int(stream["height"])

    rate = stream["r_frame_rate"]
    num, den = rate.split("/")
    fps = float(num) / max(float(den), 1.0)

    nb_frames = stream.get("nb_frames")
    duration = stream.get("duration")

    if nb_frames is not None and str(nb_frames).isdigit():
        total_frames = int(nb_frames)
    elif duration is not None:
        total_frames = int(float(duration) * fps)
    else:
        total_frames = None

    duration_s = float(duration) if duration is not None else None
    return width, height, fps, total_frames, duration_s


class FFmpegFrameReader:
    def __init__(
        self,
        video_path: str,
        width: int,
        height: int,
        queue_size: int = 8,
        seek_seconds: float = 0.0,
        start_frame_index: int = 0,
    ) -> None:
        self.video_path = video_path
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.thread: Optional[threading.Thread] = None
        self.process: Optional[subprocess.Popen] = None
        self.stop_event = threading.Event()
        self.started = False
        self.seek_seconds = max(0.0, float(seek_seconds))
        self.start_frame_index = max(0, int(start_frame_index))

    def start(self) -> None:
        if self.started:
            return
        self.started = True

        cmd = ["ffmpeg"]
        if self.seek_seconds > 0:
            cmd += ["-ss", f"{self.seek_seconds:.3f}"]
        cmd += [
            "-i", self.video_path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-loglevel", "error",
            "-"
        ]

        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        local_idx = 0
        try:
            while not self.stop_event.is_set():
                raw = self.process.stdout.read(self.frame_size)
                if len(raw) != self.frame_size:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3)).copy()
                frame_idx = self.start_frame_index + local_idx
                while not self.stop_event.is_set():
                    try:
                        self.queue.put((frame_idx, frame), timeout=0.05)
                        break
                    except queue.Full:
                        continue
                local_idx += 1
        finally:
            try:
                self.queue.put(None, timeout=0.05)
            except Exception:
                pass
            try:
                if self.process and self.process.stdout:
                    self.process.stdout.close()
            except Exception:
                pass
            try:
                if self.process:
                    self.process.wait(timeout=1)
            except Exception:
                pass

    def read_nowait(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return "EMPTY"

    def stop(self) -> None:
        self.stop_event.set()
        try:
            if self.process:
                self.process.kill()
        except Exception:
            pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)


# ============================================================
# INFERENCE
# ============================================================

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def max_pool_nms(hm: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(hm, kernel_size=kernel, stride=1, padding=pad)
    keep = (pooled == hm).float()
    return hm * keep

def preprocess_rgb(rgb: np.ndarray, img_size: int) -> torch.Tensor:
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def decode_points(
    hm_logits: torch.Tensor,
    off: torch.Tensor,
    score_threshold: float,
    topk: int,
    down_ratio: int,
    decoder_nms_kernel: int,
    min_distance_px: float,
) -> list[dict[str, float]]:
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


def make_heatmap_preview(
    hm_logits: torch.Tensor,
    out_w: int,
    out_h: int,
    show_nms: bool,
    nms_kernel: int,
) -> np.ndarray:
    hm = torch.sigmoid(hm_logits)
    if show_nms:
        hm = max_pool_nms(hm, kernel=nms_kernel)

    hm_np = hm[0, 0].detach().cpu().numpy()
    hm_np = np.clip(hm_np, 0.0, 1.0)

    hm_u8 = (hm_np * 255.0).astype(np.uint8)
    hm_u8 = cv2.resize(hm_u8, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    max_score = float(hm_np.max()) if hm_np.size else 0.0
    cv2.putText(
        hm_color,
        f"heatmap max={max_score:.3f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return hm_color


def clamp_roi(roi, w: int, h: int):
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


def parse_timecode_to_seconds(text: str) -> float:
    text = text.strip()
    if not text:
        raise ValueError("Puste pole czasu")
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError("Niepoprawny format czasu")


@dataclass
class DetectorConfig:
    img_size: int = 1024
    down_ratio: int = 4
    threshold: float = 0.50
    topk: int = 24
    nms_kernel: int = 7
    min_distance_px: float = 16.0
    show_nms_heatmap: bool = False


class PointDetectorWrapper:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model: Optional[nn.Module] = None
        self.model_name: str = "?"
        self.base_ch: int = 0
        self.config = DetectorConfig()
        self.ckpt_path: Optional[Path] = None

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
        raise RuntimeError("Nieznany model_type")

    def _infer_unet_out_stride(self, state_dict: dict[str, torch.Tensor]) -> int:
        if any(k.startswith("up3.") for k in state_dict.keys()):
            return 2
        return 4

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model_choice: str = "auto",
        base_ch_override: int | None = None,
        down_ratio_override: int | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        if model_choice == "auto":
            model_type = self._infer_model_type(state_dict)
        else:
            model_type = model_choice.lower()

        if base_ch_override is None or base_ch_override <= 0:
            base_ch = self._infer_base_ch(state_dict, model_type)
        else:
            base_ch = int(base_ch_override)

        if model_type == "hrnet":
            model = HRNetLitePoint(base_ch=base_ch)
            down_ratio = 4 if down_ratio_override is None or down_ratio_override <= 0 else int(down_ratio_override)
        elif model_type == "unet":
            if down_ratio_override is None or down_ratio_override <= 0:
                down_ratio = self._infer_unet_out_stride(state_dict)
            else:
                down_ratio = int(down_ratio_override)
            model = UNetPoint(base_ch=base_ch, out_stride=down_ratio)
        else:
            raise RuntimeError(f"Nieobsługiwany model: {model_type}")

        model = model.to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        self.model = model
        self.model_name = model_type
        self.base_ch = base_ch
        self.ckpt_path = checkpoint_path
        self.config.down_ratio = down_ratio

    @torch.no_grad()
    def infer_crop(
        self,
        crop_bgr: np.ndarray,
        global_offset_xy: tuple[int, int],
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        crop_h, crop_w = crop_bgr.shape[:2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = preprocess_rgb(crop_rgb, self.config.img_size).to(self.device)

        hm_logits, off = self.model(x)
        preds_local = decode_points(
            hm_logits=hm_logits,
            off=off,
            score_threshold=self.config.threshold,
            topk=self.config.topk,
            down_ratio=self.config.down_ratio,
            decoder_nms_kernel=self.config.nms_kernel,
            min_distance_px=self.config.min_distance_px,
        )

        sx = crop_w / self.config.img_size
        sy = crop_h / self.config.img_size
        ox, oy = global_offset_xy

        preds_global: list[dict[str, float]] = []
        for p in preds_local:
            gx = p["x"] * sx + ox
            gy = p["y"] * sy + oy
            preds_global.append({"x": gx, "y": gy, "score": p["score"]})

        hm_vis = make_heatmap_preview(
            hm_logits=hm_logits,
            out_w=crop_w,
            out_h=crop_h,
            show_nms=self.config.show_nms_heatmap,
            nms_kernel=self.config.nms_kernel,
        )

        return hm_vis, preds_global


# ============================================================
# GUI
# ============================================================

class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AGL Compare: HRNet vs U-Net")
        self.root.geometry("1600x900")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hrnet = PointDetectorWrapper(self.device)
        self.unet = PointDetectorWrapper(self.device)

        self.video_path: Optional[Path] = None
        self.video_width: Optional[int] = None
        self.video_height: Optional[int] = None

        self.reader: Optional[FFmpegFrameReader] = None
        self.video_fps = 25.0
        self.total_frames: Optional[int] = None
        self.duration_s: Optional[float] = None
        self.frame_index = -1

        self.playing = False
        self.last_photo = None
        self.last_tick = 0.0
        self.last_frame_bgr: Optional[np.ndarray] = None

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.save_output_var = tk.BooleanVar(value=False)
        self.output_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Wczytaj checkpoint HRNet, U-Net i wideo.")
        self.current_time_var = tk.StringVar(value="00:00 / --:--")

        self.seek_var = tk.StringVar(value="0")
        self.hr_show_nms_var = tk.BooleanVar(value=False)
        self.un_show_nms_var = tk.BooleanVar(value=False)

        self.model_choice_var_hr = tk.StringVar(value="auto")
        self.model_choice_var_un = tk.StringVar(value="auto")
        self.hr_base_ch = tk.IntVar(value=0)
        self.hr_down_ratio = tk.IntVar(value=0)
        self.un_base_ch = tk.IntVar(value=0)
        self.un_down_ratio = tk.IntVar(value=0)

        self.roi_xyxy: tuple[int, int, int, int] | None = None
        self.roi_select_mode = False
        self.roi_start_xy: Optional[tuple[int, int]] = None
        self.roi_preview_rect_id: Optional[int] = None

        self.display_scale = 1.0
        self.display_x0 = 0
        self.display_y0 = 0
        self.display_panel_w = 0
        self.display_panel_h = 0

        self._build_ui()
        self._bind_canvas_events()
        self._schedule_update()

    def _on_left_inner_configure(self, event=None):
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_left_canvas_configure(self, event):
        self.left_canvas.itemconfigure(self.left_window, width=event.width)

    def _bind_mousewheel(self, event=None):
        self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.left_canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.left_canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, event=None):
        self.left_canvas.unbind_all("<MouseWheel>")
        self.left_canvas.unbind_all("<Button-4>")
        self.left_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.left_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.left_canvas.yview_scroll(1, "units")

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ---------- LEFT: scrollable fixed-width panel ----------
        left_outer = ttk.Frame(main, width=300)
        left_outer.pack(side=tk.LEFT, fill=tk.Y)
        left_outer.pack_propagate(False)

        self.left_canvas = tk.Canvas(left_outer, highlightthickness=0, width=300)
        left_scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.left_inner = ttk.Frame(self.left_canvas)
        self.left_window = self.left_canvas.create_window((0, 0), window=self.left_inner, anchor="nw")

        self.left_inner.bind("<Configure>", self._on_left_inner_configure)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_configure)

        self.left_canvas.bind("<Enter>", self._bind_mousewheel)
        self.left_canvas.bind("<Leave>", self._unbind_mousewheel)
        self.left_inner.bind("<Enter>", self._bind_mousewheel)
        self.left_inner.bind("<Leave>", self._unbind_mousewheel)

        # ---------- RIGHT ----------
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        left = self.left_inner

        ttk.Label(left, text="Checkpoints", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))

        hr_box = ttk.LabelFrame(left, text="HRNet")
        hr_box.pack(fill=tk.X, pady=4)
        ttk.Combobox(hr_box, textvariable=self.model_choice_var_hr, values=["auto", "hrnet"], state="readonly").pack(fill=tk.X, pady=2)
        row = ttk.Frame(hr_box); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="base_ch").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.hr_base_ch, width=8).pack(side=tk.RIGHT)
        row = ttk.Frame(hr_box); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="down_ratio").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.hr_down_ratio, width=8).pack(side=tk.RIGHT)
        ttk.Button(hr_box, text="Wczytaj HRNet .pt", command=self.load_hrnet).pack(fill=tk.X, pady=2)

        un_box = ttk.LabelFrame(left, text="U-Net")
        un_box.pack(fill=tk.X, pady=4)
        ttk.Combobox(un_box, textvariable=self.model_choice_var_un, values=["auto", "unet"], state="readonly").pack(fill=tk.X, pady=2)
        row = ttk.Frame(un_box); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="base_ch").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.un_base_ch, width=8).pack(side=tk.RIGHT)
        row = ttk.Frame(un_box); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="down_ratio").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.un_down_ratio, width=8).pack(side=tk.RIGHT)
        ttk.Button(un_box, text="Wczytaj U-Net .pt", command=self.load_unet).pack(fill=tk.X, pady=2)

        ttk.Button(left, text="Wczytaj wideo", command=self.load_video).pack(fill=tk.X, pady=(6, 2))

        ttk.Checkbutton(left, text="Zapisz wynikowe wideo 4-panelowe", variable=self.save_output_var).pack(anchor="w", pady=(8, 0))
        ttk.Button(left, text="Wybierz plik wyjściowy .mp4", command=self.choose_output_path).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Seek / Nawigacja", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=2)
        ttk.Entry(row, textvariable=self.seek_var, width=14).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Idź", command=self.seek_to_timecode).pack(side=tk.LEFT, padx=(6, 0))
        row2 = ttk.Frame(left)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="-5s", command=lambda: self.seek_relative(-5)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row2, text="+5s", command=lambda: self.seek_relative(5)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Label(left, textvariable=self.current_time_var).pack(anchor="w", pady=(4, 0))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="ROI", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Button(left, text="Włącz wybór ROI", command=self.enable_roi_select).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Wyczyść ROI", command=self.clear_roi).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Wspólne parametry", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self.img_size_var = tk.IntVar(value=1024)
        self._add_scale(left, "Input size", self.img_size_var, 256, 1536)

        ttk.Label(left, text="HRNet parametry", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8, 6))
        self.hr_threshold = tk.DoubleVar(value=0.50)
        self.hr_topk = tk.IntVar(value=24)
        self.hr_min_dist = tk.DoubleVar(value=16.0)
        self.hr_nms = tk.IntVar(value=7)
        self._add_scale(left, "HR threshold", self.hr_threshold, 0.01, 0.90)
        self._add_scale(left, "HR Top-K", self.hr_topk, 1, 100)
        self._add_scale(left, "HR min distance", self.hr_min_dist, 1, 50)
        self._add_scale(left, "HR NMS kernel", self.hr_nms, 3, 11)
        ttk.Checkbutton(left, text="HR heatmap po NMS", variable=self.hr_show_nms_var, command=self.apply_gui_config).pack(anchor="w", pady=(4, 0))

        ttk.Label(left, text="U-Net parametry", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8, 6))
        self.un_threshold = tk.DoubleVar(value=0.50)
        self.un_topk = tk.IntVar(value=24)
        self.un_min_dist = tk.DoubleVar(value=16.0)
        self.un_nms = tk.IntVar(value=7)
        self._add_scale(left, "UN threshold", self.un_threshold, 0.01, 0.90)
        self._add_scale(left, "UN Top-K", self.un_topk, 1, 100)
        self._add_scale(left, "UN min distance", self.un_min_dist, 1, 50)
        self._add_scale(left, "UN NMS kernel", self.un_nms, 3, 11)
        ttk.Checkbutton(left, text="UN heatmap po NMS", variable=self.un_show_nms_var, command=self.apply_gui_config).pack(anchor="w", pady=(4, 0))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(left, text="Start / Resume", command=self.play).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Pause", command=self.pause).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Stop", command=self.stop).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Jedna klatka", command=self.step_once).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left, textvariable=self.status_var, wraplength=260).pack(anchor="w")

        self.canvas = tk.Canvas(right, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _bind_canvas_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def _add_scale(self, parent, label, variable, from_, to):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text=label).pack(anchor="w")
        scale = ttk.Scale(frame, orient=tk.HORIZONTAL, variable=variable, from_=from_, to=to)
        scale.pack(fill=tk.X)
        ttk.Label(frame, textvariable=variable).pack(anchor="e")

    def load_hrnet(self) -> None:
        path = filedialog.askopenfilename(title="Wybierz checkpoint HRNet", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.hrnet.load_checkpoint(
                path,
                model_choice=self.model_choice_var_hr.get(),
                base_ch_override=self.hr_base_ch.get() if self.hr_base_ch.get() > 0 else None,
                down_ratio_override=self.hr_down_ratio.get() if self.hr_down_ratio.get() > 0 else None,
            )
            self.status_var.set(f"Wczytano HRNet: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Błąd HRNet", f"Nie udało się wczytać HRNet:\n{e}")

    def load_unet(self) -> None:
        path = filedialog.askopenfilename(title="Wybierz checkpoint U-Net", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.unet.load_checkpoint(
                path,
                model_choice=self.model_choice_var_un.get(),
                base_ch_override=self.un_base_ch.get() if self.un_base_ch.get() > 0 else None,
                down_ratio_override=self.un_down_ratio.get() if self.un_down_ratio.get() > 0 else None,
            )
            self.status_var.set(f"Wczytano U-Net: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Błąd U-Net", f"Nie udało się wczytać U-Net:\n{e}")

    def load_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz wideo",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")],
        )
        if not path:
            return

        self.stop(full_reset=False)
        self.video_path = Path(path)

        try:
            w, h, fps, total_frames, duration_s = ffprobe_video(str(self.video_path))
        except Exception as e:
            messagebox.showerror("Błąd ffprobe", f"Nie udało się odczytać parametrów wideo:\n{e}")
            return

        self.video_width = w
        self.video_height = h
        self.video_fps = max(fps, 1.0)
        self.total_frames = total_frames
        self.duration_s = duration_s
        self.frame_index = -1
        self.roi_xyxy = None

        try:
            self.reader = FFmpegFrameReader(str(self.video_path), w, h)
            self.reader.start()
        except Exception as e:
            messagebox.showerror("Błąd ffmpeg", f"Nie udało się uruchomić czytnika ffmpeg:\n{e}")
            self.reader = None
            return

        self._prepare_writer(w, h, self.video_fps)
        self._update_time_label()
        self.status_var.set(f"Wczytano wideo: {self.video_path.name} | {w}x{h} | {self.video_fps:.2f} FPS")
        self.step_once()

    def choose_output_path(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Wybierz plik wynikowy",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
        )
        if path:
            self.output_path_var.set(path)
            self.status_var.set(f"Plik wynikowy: {Path(path).name}")

    def _prepare_writer(self, width: int, height: int, fps: float) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if not self.save_output_var.get():
            return
        if not self.output_path_var.get():
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.output_path_var.get(), fourcc, fps, (width * 2, height * 2))

    def apply_gui_config(self) -> None:
        img_size = max(256, int(self.img_size_var.get()))

        self.hrnet.config.img_size = img_size
        self.hrnet.config.threshold = float(self.hr_threshold.get())
        self.hrnet.config.topk = int(self.hr_topk.get())
        self.hrnet.config.min_distance_px = float(self.hr_min_dist.get())
        hr_nms = int(self.hr_nms.get())
        if hr_nms % 2 == 0:
            hr_nms += 1
        self.hrnet.config.nms_kernel = max(3, hr_nms)
        self.hrnet.config.show_nms_heatmap = bool(self.hr_show_nms_var.get())

        self.unet.config.img_size = img_size
        self.unet.config.threshold = float(self.un_threshold.get())
        self.unet.config.topk = int(self.un_topk.get())
        self.unet.config.min_distance_px = float(self.un_min_dist.get())
        un_nms = int(self.un_nms.get())
        if un_nms % 2 == 0:
            un_nms += 1
        self.unet.config.nms_kernel = max(3, un_nms)
        self.unet.config.show_nms_heatmap = bool(self.un_show_nms_var.get())

    def enable_roi_select(self) -> None:
        self.roi_select_mode = True
        self.status_var.set("Tryb ROI: przeciągnij po panelu GÓRNYM-LEWYM.")

    def clear_roi(self) -> None:
        self.roi_xyxy = None
        self.roi_select_mode = False
        self.roi_start_xy = None
        self.status_var.set("ROI wyczyszczone.")
        if self.last_frame_bgr is not None:
            self.render_current_frame(self.last_frame_bgr)

    def _canvas_to_frame_xy(self, cx: int, cy: int) -> tuple[int, int] | None:
        if self.video_width is None or self.video_height is None:
            return None
        if cx < self.display_x0 or cy < self.display_y0:
            return None
        if cx >= self.display_x0 + self.display_panel_w:
            return None
        if cy >= self.display_y0 + self.display_panel_h:
            return None

        local_x = cx - self.display_x0
        local_y = cy - self.display_y0

        fx = int(round(local_x / max(self.display_scale, 1e-6)))
        fy = int(round(local_y / max(self.display_scale, 1e-6)))
        fx = max(0, min(self.video_width - 1, fx))
        fy = max(0, min(self.video_height - 1, fy))
        return fx, fy

    def on_canvas_press(self, event) -> None:
        if not self.roi_select_mode:
            return
        pt = self._canvas_to_frame_xy(event.x, event.y)
        if pt is None:
            return
        self.roi_start_xy = pt

    def on_canvas_drag(self, event) -> None:
        if not self.roi_select_mode or self.roi_start_xy is None:
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

    def on_canvas_release(self, event) -> None:
        if not self.roi_select_mode or self.roi_start_xy is None:
            return
        pt = self._canvas_to_frame_xy(event.x, event.y)
        if pt is None:
            return

        x1, y1 = self.roi_start_xy
        x2, y2 = pt
        roi = clamp_roi((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), self.video_width or 0, self.video_height or 0)

        self.roi_start_xy = None
        self.roi_select_mode = False

        if self.roi_preview_rect_id is not None:
            self.canvas.delete(self.roi_preview_rect_id)
            self.roi_preview_rect_id = None

        if roi is None:
            self.status_var.set("ROI odrzucone: za małe.")
            return

        self.roi_xyxy = roi
        self.status_var.set(f"ROI ustawione: {roi}")
        if self.last_frame_bgr is not None:
            self.render_current_frame(self.last_frame_bgr)

    def play(self) -> None:
        if self.reader is None:
            messagebox.showinfo("Info", "Najpierw wczytaj wideo.")
            return
        if self.hrnet.model is None or self.unet.model is None:
            messagebox.showinfo("Info", "Najpierw wczytaj oba modele.")
            return
        self.apply_gui_config()
        self.playing = True
        self.last_tick = time.perf_counter()
        self.status_var.set("Odtwarzanie...")

    def pause(self) -> None:
        self.playing = False
        self.status_var.set("Pauza.")

    def stop(self, full_reset: bool = True) -> None:
        self.playing = False
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.frame_index = -1

        if full_reset:
            self.video_path = None
            self.video_width = None
            self.video_height = None
            self.total_frames = None
            self.duration_s = None
            self.last_frame_bgr = None
            self.roi_xyxy = None
            self.current_time_var.set("00:00 / --:--")

        self.status_var.set("Zatrzymano.")

    def seek_relative(self, delta_seconds: float) -> None:
        if self.video_path is None:
            return
        current_sec = max(0.0, self.frame_index / max(self.video_fps, 1.0)) if self.frame_index >= 0 else 0.0
        self.seek_to_seconds(current_sec + delta_seconds)

    def seek_to_timecode(self) -> None:
        if self.video_path is None:
            messagebox.showinfo("Info", "Najpierw wczytaj wideo.")
            return
        try:
            sec = parse_timecode_to_seconds(self.seek_var.get())
        except Exception as e:
            messagebox.showerror("Błąd czasu", f"Niepoprawny czas:\n{e}")
            return
        self.seek_to_seconds(sec)

    def seek_to_seconds(self, sec: float) -> None:
        if self.video_path is None or self.video_width is None or self.video_height is None:
            return

        was_playing = self.playing
        self.playing = False

        if self.duration_s is not None:
            sec = max(0.0, min(float(sec), max(0.0, self.duration_s - 0.001)))
        else:
            sec = max(0.0, float(sec))

        if self.reader is not None:
            self.reader.stop()
            self.reader = None

        start_frame_index = int(sec * self.video_fps)
        self.reader = FFmpegFrameReader(
            str(self.video_path),
            self.video_width,
            self.video_height,
            seek_seconds=sec,
            start_frame_index=start_frame_index,
        )
        self.reader.start()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self._prepare_writer(self.video_width, self.video_height, self.video_fps)

        self.frame_index = start_frame_index - 1
        self._update_time_label()
        self.status_var.set(f"Przeskok do {sec:.2f}s")
        self.step_once()

        if was_playing:
            self.playing = True
            self.last_tick = time.perf_counter()

    def _update_time_label(self) -> None:
        if self.frame_index >= 0:
            cur = self.frame_index / max(self.video_fps, 1.0)
        else:
            cur = 0.0

        def fmt(t: float | None) -> str:
            if t is None:
                return "--:--"
            t = max(0.0, float(t))
            total = int(t)
            h = total // 3600
            m = (total % 3600) // 60
            s = total % 60
            if h > 0:
                return f"{h:02d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        self.current_time_var.set(f"{fmt(cur)} / {fmt(self.duration_s)}")

    def render_current_frame(self, frame: np.ndarray) -> None:
        if self.hrnet.model is None or self.unet.model is None:
            return

        self.apply_gui_config()

        orig_h, orig_w = frame.shape[:2]
        roi = clamp_roi(self.roi_xyxy, orig_w, orig_h)
        if roi is None:
            x1, y1, x2, y2 = 0, 0, orig_w, orig_h
        else:
            x1, y1, x2, y2 = roi

        crop_bgr = frame[y1:y2, x1:x2]

        t0 = time.perf_counter()
        hr_heat_crop, hr_preds = self.hrnet.infer_crop(crop_bgr, (x1, y1))
        un_heat_crop, un_preds = self.unet.infer_crop(crop_bgr, (x1, y1))
        infer_ms = (time.perf_counter() - t0) * 1000.0

        hr_vis = frame.copy()
        un_vis = frame.copy()

        if roi is not None:
            cv2.rectangle(hr_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(un_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for p in hr_preds:
            x0 = int(round(p["x"]))
            y0 = int(round(p["y"]))
            cv2.circle(hr_vis, (x0, y0), 8, (0, 0, 255), 2)
            cv2.putText(hr_vis, f"{p['score']:.2f}", (x0 + 10, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        for p in un_preds:
            x0 = int(round(p["x"]))
            y0 = int(round(p["y"]))
            cv2.circle(un_vis, (x0, y0), 8, (0, 0, 255), 2)
            cv2.putText(un_vis, f"{p['score']:.2f}", (x0 + 10, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        hr_heat_full = np.zeros_like(frame)
        un_heat_full = np.zeros_like(frame)
        hr_heat_full[y1:y2, x1:x2] = hr_heat_crop
        un_heat_full[y1:y2, x1:x2] = un_heat_crop

        if roi is not None:
            cv2.rectangle(hr_heat_full, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(un_heat_full, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(hr_vis, f"HRNet detections ({len(hr_preds)})", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hr_heat_full, "HRNet heatmap", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(un_vis, f"U-Net detections ({len(un_preds)})", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(un_heat_full, "U-Net heatmap", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        top = np.hstack([hr_vis, hr_heat_full])
        bottom = np.hstack([un_vis, un_heat_full])
        combined = np.vstack([top, bottom])

        if self.video_writer is not None:
            self.video_writer.write(combined)

        self.show_combined(combined)

        frame_info = f"frame={self.frame_index}"
        if self.total_frames:
            frame_info += f"/{self.total_frames}"
        roi_txt = " | ROI" if self.roi_xyxy is not None else ""
        self.status_var.set(
            f"{frame_info} | HR det={len(hr_preds)} | UN det={len(un_preds)} | "
            f"infer={infer_ms:.1f} ms{roi_txt}"
        )

    def step_once(self) -> None:
        if self.reader is None:
            return
        if self.hrnet.model is None or self.unet.model is None:
            messagebox.showinfo("Info", "Najpierw wczytaj oba modele.")
            return

        item = self.reader.read_nowait()
        if item == "EMPTY":
            return
        if item is None:
            self.playing = False
            self.status_var.set("Koniec wideo.")
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            return

        frame_idx, frame = item
        self.frame_index = frame_idx
        self.last_frame_bgr = frame.copy()
        self._update_time_label()
        self.render_current_frame(frame)

    def show_combined(self, combined_bgr: np.ndarray) -> None:
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)

        rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # top-left quadrant corresponds to original frame
        self.display_scale = scale
        self.display_x0 = (canvas_w - new_w) // 2
        self.display_y0 = (canvas_h - new_h) // 2
        self.display_panel_w = new_w // 2
        self.display_panel_h = new_h // 2

        img = Image.fromarray(resized)
        self.last_photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(self.display_x0, self.display_y0, anchor=tk.NW, image=self.last_photo)

        roi = self.roi_xyxy
        if roi is not None:
            x1, y1, x2, y2 = roi
            cx1 = self.display_x0 + int(round(x1 * self.display_scale))
            cy1 = self.display_y0 + int(round(y1 * self.display_scale))
            cx2 = self.display_x0 + int(round(x2 * self.display_scale))
            cy2 = self.display_y0 + int(round(y2 * self.display_scale))
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="yellow", width=2)

    def _schedule_update(self) -> None:
        if self.playing and self.reader is not None and self.hrnet.model is not None and self.unet.model is not None:
            now = time.perf_counter()
            frame_period = 1.0 / max(self.video_fps, 1.0)
            if now - self.last_tick >= frame_period:
                self.step_once()
                self.last_tick = now
        self.root.after(5, self._schedule_update)


def main() -> int:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    app = App(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())