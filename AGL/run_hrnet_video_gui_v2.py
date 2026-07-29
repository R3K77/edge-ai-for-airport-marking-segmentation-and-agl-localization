#!/usr/bin/env python3
"""
Improved GUI video runner for HRNet-Lite detector.

What is fixed compared to the previous version:
- uses ffprobe + ffmpeg rawvideo reader instead of cv2.VideoCapture
- respects source video FPS instead of stepping every 10 ms
- does not "finish too quickly" just because the UI loop is fast
- more robust on MP4/H.264/H.265 files that OpenCV sometimes reads badly
- keeps the app open at end-of-video

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


# ----------------------------
# Model definition (HRNet-Lite)
# ----------------------------

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


# ----------------------------
# ffmpeg / video utils
# ----------------------------

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
    def __init__(self, video_path: str, width: int, height: int, queue_size: int = 8) -> None:
        self.video_path = video_path
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.thread: Optional[threading.Thread] = None
        self.process: Optional[subprocess.Popen] = None
        self.stop_event = threading.Event()
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        cmd = [
            "ffmpeg",
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
        frame_idx = 0
        try:
            while not self.stop_event.is_set():
                raw = self.process.stdout.read(self.frame_size)
                if len(raw) != self.frame_size:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3)).copy()
                while not self.stop_event.is_set():
                    try:
                        self.queue.put((frame_idx, frame), timeout=0.05)
                        break
                    except queue.Full:
                        continue
                frame_idx += 1
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
            item = self.queue.get_nowait()
            return item
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


# ----------------------------
# Inference utils
# ----------------------------

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def max_pool_nms(hm: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(hm, kernel_size=kernel, stride=1, padding=pad)
    keep = (pooled == hm).float()
    return hm * keep


def preprocess_bgr(frame_bgr: np.ndarray, img_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0)
    return x, (w, h)


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


@dataclass
class InferenceConfig:
    img_size: int = 1024
    down_ratio: int = 4
    base_ch: int = 32
    threshold: float = 0.35
    topk: int = 24
    nms_kernel: int = 7
    min_distance_px: float = 16.0


class HRNetVideoDetector:
    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model: Optional[HRNetLitePoint] = None
        self.config = InferenceConfig()
        self.ckpt_path: Optional[Path] = None

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = HRNetLitePoint(base_ch=32).to(self.device)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        self.ckpt_path = checkpoint_path

    @torch.no_grad()
    def infer_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]], float]:
        if self.model is None:
            raise RuntimeError("Load a checkpoint first.")

        x, (orig_w, orig_h) = preprocess_bgr(frame_bgr, self.config.img_size)
        x = x.to(self.device)

        t0 = time.perf_counter()
        hm_logits, off = self.model(x)
        preds = decode_points(
            hm_logits=hm_logits,
            off=off,
            score_threshold=self.config.threshold,
            topk=self.config.topk,
            down_ratio=self.config.down_ratio,
            decoder_nms_kernel=self.config.nms_kernel,
            min_distance_px=self.config.min_distance_px,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        sx = orig_w / self.config.img_size
        sy = orig_h / self.config.img_size

        vis = frame_bgr.copy()
        for p in preds:
            x0 = int(round(p["x"] * sx))
            y0 = int(round(p["y"] * sy))
            score = p["score"]
            cv2.circle(vis, (x0, y0), 8, (0, 0, 255), 2)
            cv2.putText(vis, f"{score:.2f}", (x0 + 10, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        return vis, preds, infer_ms


# ----------------------------
# GUI
# ----------------------------

class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AGL HRNet Video Viewer v2")
        self.root.geometry("1450x920")

        self.detector = HRNetVideoDetector(device="cuda")

        self.video_path: Optional[Path] = None
        self.reader: Optional[FFmpegFrameReader] = None
        self.video_fps = 25.0
        self.total_frames: Optional[int] = None
        self.duration_s: Optional[float] = None
        self.frame_index = -1

        self.playing = False
        self.last_photo = None
        self.last_tick = 0.0

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.save_output_var = tk.BooleanVar(value=False)
        self.output_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Wczytaj model i wideo.")

        self._build_ui()
        self._schedule_update()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Model / Wideo", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Button(left, text="Wczytaj checkpoint .pt", command=self.load_model).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Wczytaj wideo", command=self.load_video).pack(fill=tk.X, pady=2)

        ttk.Checkbutton(left, text="Zapisz wynikowe wideo", variable=self.save_output_var).pack(anchor="w", pady=(8, 0))
        ttk.Button(left, text="Wybierz plik wyjściowy .mp4", command=self.choose_output_path).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Parametry detekcji", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))

        self.threshold_var = tk.DoubleVar(value=0.35)
        self.topk_var = tk.IntVar(value=24)
        self.min_dist_var = tk.DoubleVar(value=16.0)
        self.nms_var = tk.IntVar(value=7)
        self.img_size_var = tk.IntVar(value=1024)

        self._add_scale(left, "Threshold", self.threshold_var, 0.01, 0.90)
        self._add_scale(left, "Top-K", self.topk_var, 1, 100)
        self._add_scale(left, "Min distance px", self.min_dist_var, 1, 50)
        self._add_scale(left, "NMS kernel", self.nms_var, 3, 11)
        self._add_scale(left, "Input size", self.img_size_var, 256, 1536)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(left, text="Start / Resume", command=self.play).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Pause", command=self.pause).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Stop", command=self.stop).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Jedna klatka", command=self.step_once).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left, textvariable=self.status_var, wraplength=320).pack(anchor="w")

        self.canvas = tk.Canvas(right, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _add_scale(self, parent, label, variable, from_, to):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text=label).pack(anchor="w")
        scale = ttk.Scale(frame, orient=tk.HORIZONTAL, variable=variable, from_=from_, to=to)
        scale.pack(fill=tk.X)
        ttk.Label(frame, textvariable=variable).pack(anchor="e")

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz checkpoint modelu",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.detector.load_checkpoint(path)
            self.status_var.set(f"Wczytano model: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Błąd modelu", f"Nie udało się wczytać modelu:\n{e}")

    def load_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz wideo",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")],
        )
        if not path:
            return

        self.stop()
        self.video_path = Path(path)
        try:
            w, h, fps, total_frames, duration_s = ffprobe_video(str(self.video_path))
        except Exception as e:
            messagebox.showerror("Błąd ffprobe", f"Nie udało się odczytać parametrów wideo:\n{e}")
            return

        self.video_fps = max(fps, 1.0)
        self.total_frames = total_frames
        self.duration_s = duration_s
        self.frame_index = -1

        try:
            self.reader = FFmpegFrameReader(str(self.video_path), w, h)
            self.reader.start()
        except Exception as e:
            messagebox.showerror("Błąd ffmpeg", f"Nie udało się uruchomić czytnika ffmpeg:\n{e}")
            self.reader = None
            return

        self._prepare_writer(w, h, self.video_fps)
        self.status_var.set(
            f"Wczytano wideo: {self.video_path.name} | {w}x{h} | {self.video_fps:.2f} FPS"
        )
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
        self.video_writer = cv2.VideoWriter(self.output_path_var.get(), fourcc, fps, (width, height))

    def apply_gui_config(self) -> None:
        self.detector.config.threshold = float(self.threshold_var.get())
        self.detector.config.topk = int(self.topk_var.get())
        self.detector.config.min_distance_px = float(self.min_dist_var.get())

        nms = int(self.nms_var.get())
        if nms % 2 == 0:
            nms += 1
        self.detector.config.nms_kernel = max(3, nms)
        self.detector.config.img_size = max(256, int(self.img_size_var.get()))

    def play(self) -> None:
        if self.reader is None:
            messagebox.showinfo("Info", "Najpierw wczytaj wideo.")
            return
        if self.detector.model is None:
            messagebox.showinfo("Info", "Najpierw wczytaj model.")
            return
        self.apply_gui_config()
        self.playing = True
        self.last_tick = time.perf_counter()
        self.status_var.set("Odtwarzanie...")

    def pause(self) -> None:
        self.playing = False
        self.status_var.set("Pauza.")

    def stop(self) -> None:
        self.playing = False
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.status_var.set("Zatrzymano.")
        self.frame_index = -1

    def step_once(self) -> None:
        if self.reader is None:
            return
        if self.detector.model is None:
            messagebox.showinfo("Info", "Najpierw wczytaj model.")
            return

        self.apply_gui_config()

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

        vis, preds, infer_ms = self.detector.infer_frame(frame)

        if self.video_writer is not None:
            self.video_writer.write(vis)

        self.show_frame(vis)

        frame_info = f"frame={frame_idx}"
        if self.total_frames:
            frame_info += f"/{self.total_frames}"

        self.status_var.set(
            f"{frame_info} | det={len(preds)} | infer={infer_ms:.1f} ms | thr={self.detector.config.threshold:.2f}"
        )

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        img = Image.fromarray(resized)
        self.last_photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        x0 = (canvas_w - new_w) // 2
        y0 = (canvas_h - new_h) // 2
        self.canvas.create_image(x0, y0, anchor=tk.NW, image=self.last_photo)

    def _schedule_update(self) -> None:
        if self.playing and self.reader is not None and self.detector.model is not None:
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
