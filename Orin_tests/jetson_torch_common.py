#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wspólne funkcje dla benchmarków PyTorch/TorchScript na NVIDIA Jetson AGX Orin.
Umie ładować .pt/.pth jako TorchScript, pełny nn.Module albo state_dict + factory.
Zbiera metryki Jetsona z: psutil, torch.cuda, sysfs, tegrastats, nvpmodel, jetson_clocks.
"""
from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

import torch

os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "32768")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
POINT_DOWN_RATIO = 4


# ============================================================
# Dataclassy
# ============================================================

@dataclass
class PointModelConfig:
    threshold: float = 0.526133828996282
    topk: int = 24
    min_distance_px: float = 16.0
    nms_kernel: int = 7
    crop_bottom_ratio: float = 0.60


@dataclass
class SegModelConfig:
    crop_bottom_ratio: float = 0.60
    sample_stride_x: int = 14
    sample_stride_y: int = 12


@dataclass
class ProjectionConfig:
    d_min: float = 0.5
    d_max: float = 5.5
    width_near_m: float = 1.5
    width_far_m: float = 9.0
    distance_mode: str = "reciprocal"
    gamma: float = 2.0


@dataclass
class GPSPerFrameRow:
    frame: int
    pts_sec: float
    time_utc: str
    lat: Optional[float]
    lon: Optional[float]
    alt_m: Optional[float]
    speed_kmh: Optional[float]
    bearing_deg: Optional[float]


@dataclass
class GPSBundle:
    rows: Dict[int, GPSPerFrameRow]
    raw_points_csv: Path
    per_frame_csv: Path
    fps: float
    total_frames: int


# ============================================================
# Ogólne I/O
# ============================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_key(text: Any) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9A-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_cmd(cmd: Sequence[str], timeout: Optional[float] = None, check: bool = False) -> str:
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=check)
        return p.stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def append_dict_row_csv(path: Path, row: Dict[str, Any], header_cache: Dict[Path, List[str]]) -> None:
    ensure_dir(path.parent)
    if path not in header_cache:
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open("r", newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                try:
                    header_cache[path] = next(r)
                except StopIteration:
                    header_cache[path] = list(row.keys())
        else:
            header_cache[path] = list(row.keys())
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=header_cache[path]).writeheader()
    # Jeżeli pojawiły się nowe kolumny, CSV pozostaje stabilny; nowe klucze trafią do osobnego *_resources.csv pisanego na końcu.
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=header_cache[path], extrasaction="ignore").writerow(row)


def parse_size(s: str) -> Tuple[int, int, int]:
    # Akceptuje HxW albo HxWxC.
    parts = [p.strip() for p in re.split(r"[xX,]", str(s)) if p.strip()]
    if len(parts) == 2:
        h, w = int(parts[0]), int(parts[1])
        c = 3
    elif len(parts) == 3:
        h, w, c = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise ValueError(f"Niepoprawny rozmiar wejścia: {s}. Użyj np. 256x512 albo 256x512x3.")
    if h <= 0 or w <= 0 or c not in (1, 3):
        raise ValueError(f"Niepoprawny rozmiar wejścia: {s}")
    return h, w, c


# ============================================================
# Wideo / ffmpeg / exiftool
# ============================================================

def ffprobe_video(video_path: Path) -> Tuple[int, int, float, Optional[int]]:
    if not have_cmd("ffprobe"):
        raise RuntimeError("Nie znaleziono ffprobe w PATH. Zainstaluj FFmpeg.")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", str(video_path),
    ]
    out = run_cmd(cmd, check=True)
    data = json.loads(out)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    nb_frames = stream.get("nb_frames")
    duration = stream.get("duration")
    if nb_frames is not None and str(nb_frames).isdigit():
        total_frames = int(nb_frames)
    elif duration is not None and fps > 0:
        total_frames = int(round(float(duration) * fps))
    else:
        total_frames = None
    return width, height, fps, total_frames


def ffmpeg_frame_reader(video_path: Path, width: int, height: int, frame_step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
    if not have_cmd("ffmpeg"):
        raise RuntimeError("Nie znaleziono ffmpeg w PATH. Zainstaluj FFmpeg.")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path)]
    if frame_step > 1:
        cmd += ["-vf", f"select=not(mod(n\\,{frame_step}))", "-vsync", "0"]
    cmd += ["-an", "-dn", "-sn", "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    frame_size = width * height * 3
    out_idx = 0
    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            yield out_idx * max(1, frame_step), frame
            out_idx += 1
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            _ = proc.stderr.read()
            proc.stderr.close()
        proc.wait()


def make_video_writer(base_path: Path, width: int, height: int, fps: float, mode: str):
    if mode == "none":
        return None, None
    if mode == "mjpg":
        out_path = base_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    elif mode == "mp4":
        out_path = base_path.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        raise ValueError(f"Nieznany video_mode: {mode}")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Nie można otworzyć VideoWriter: {out_path}")
    return writer, out_path


# ============================================================
# GPS z metadanych wideo
# ============================================================

def parse_exif_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    s = value.strip()
    s = re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", s)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def try_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except Exception:
        return None


def haversine_m(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[float]:
    if not a or not b:
        return None
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    r = 6371000.0
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def bearing_deg(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[float]:
    if not a or not b:
        return None
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def interp(ts: Sequence[float], vs: Sequence[Optional[float]], t: float) -> Optional[float]:
    import bisect
    i = bisect.bisect_left(ts, t)
    if i == 0:
        return vs[0]
    if i >= len(ts):
        return vs[-1]
    t0, t1 = ts[i - 1], ts[i]
    v0, v1 = vs[i - 1], vs[i]
    if t1 == t0:
        return v0
    w = (t - t0) / (t1 - t0)
    if v0 is None or v1 is None:
        return v0 if w <= 0.5 else v1
    return v0 + w * (v1 - v0)


def interp_geo(ts: Sequence[float], lats: Sequence[float], lons: Sequence[float], t: float) -> Tuple[Optional[float], Optional[float]]:
    return interp(ts, lats, t), interp(ts, lons, t)


def exiftool_points_to_csv(video_path: Path, out_csv: Path) -> int:
    if not have_cmd("exiftool"):
        raise RuntimeError("Nie znaleziono exiftool w PATH. Zainstaluj: sudo apt install libimage-exiftool-perl")
    fmt_path = out_csv.parent / "_points_tmp.fmt"
    fmt_text = "#[HEAD]\ntime_utc,sample_time,lat,lon,alt,speed\n#[BODY]\n$gpsdatetime,$sampletime,$gpslatitude,$gpslongitude,$gpsaltitude,$gpsspeed\n#[TAIL]\n"
    fmt_path.write_text(fmt_text, encoding="ascii")
    cmd = ["exiftool", "-ee3", "-api", "largefilesupport=1", "-n", "-d", "%Y-%m-%d %H:%M:%S.%3fZ", "-p", str(fmt_path), str(video_path)]
    out = run_cmd(cmd, check=True)
    out_csv.write_text(out, encoding="utf-8")
    fmt_path.unlink(missing_ok=True)
    count = 0
    for line in out.splitlines():
        s = line.strip()
        if s and not s.lower().startswith("time_utc"):
            count += 1
    return count


def read_points_csv_clean(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("time_utc"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            tstr, sample, lat, lon, alt, speed = parts[:6]
            dt = parse_exif_dt(tstr)
            st = try_float(sample)
            la = try_float(lat)
            lo = try_float(lon)
            al = try_float(alt)
            sp = try_float(speed)
            if dt and st is not None and la is not None and lo is not None:
                rows.append((st, dt, la, lo, al, sp))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    base_dt = rows[0][1]
    ts_abs = [(base_dt.timestamp() + r[0]) for r in rows]
    lats = [r[2] for r in rows]
    lons = [r[3] for r in rows]
    alts = [r[4] for r in rows]
    speeds = [r[5] for r in rows]
    return ts_abs, lats, lons, alts, speeds, base_dt


def video_fps_and_frames(video_path: Path) -> Tuple[float, int]:
    _w, _h, fps, frames = ffprobe_video(video_path)
    if fps > 0 and frames is not None and frames > 0:
        return fps, frames
    raise RuntimeError("Nie udało się ustalić FPS ani liczby klatek przez ffprobe.")


def write_per_frame_csv(out_csv: Path, ts_abs: Sequence[float], lats: Sequence[float], lons: Sequence[float], alts: Sequence[Optional[float]], speeds: Sequence[Optional[float]], base_dt: datetime, fps: float, frames: int, max_gap_s: float) -> Dict[int, GPSPerFrameRow]:
    g_start, g_end = ts_abs[0], ts_abs[-1]
    v_start = base_dt.timestamp()
    rows: Dict[int, GPSPerFrameRow] = {}
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "pts_sec", "time_utc", "lat", "lon", "alt_m", "speed_kmh", "bearing_deg"])
        writer.writeheader()
        for i in range(frames):
            t = v_start + i / fps
            if t < g_start - max_gap_s or t > g_end + max_gap_s:
                lat = lon = alt = speed = br = None
            else:
                lat, lon = interp_geo(ts_abs, lats, lons, t)
                alt = interp(ts_abs, alts, t)
                if any(s is not None for s in speeds):
                    speed = interp(ts_abs, speeds, t)
                else:
                    t0 = max(g_start, t - 0.25)
                    t1 = min(g_end, t + 0.25)
                    p0 = interp_geo(ts_abs, lats, lons, t0)
                    p1 = interp_geo(ts_abs, lats, lons, t1)
                    d = haversine_m(p0, p1)
                    speed = d / (t1 - t0) if (d is not None and t1 > t0) else None
                tb0 = max(g_start, t - 0.1)
                tb1 = min(g_end, t + 0.1)
                br = bearing_deg(interp_geo(ts_abs, lats, lons, tb0), interp_geo(ts_abs, lats, lons, tb1))
            iso = datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            row = GPSPerFrameRow(i, round(i / fps, 6), iso, lat, lon, alt, speed, br)
            rows[i] = row
            writer.writerow({
                "frame": i,
                "pts_sec": f"{row.pts_sec:.6f}",
                "time_utc": row.time_utc,
                "lat": f"{lat:.7f}" if lat is not None else "",
                "lon": f"{lon:.7f}" if lon is not None else "",
                "alt_m": f"{alt:.2f}" if alt is not None else "",
                "speed_kmh": f"{speed:.3f}" if speed is not None else "",
                "bearing_deg": f"{br:.1f}" if br is not None else "",
            })
    return rows


def extract_gps_from_video(video_path: Path, out_dir: Path, max_gap_s: float) -> GPSBundle:
    ensure_dir(out_dir)
    raw_csv = out_dir / f"{video_path.stem}_gps_points.csv"
    per_csv = out_dir / f"{video_path.stem}_per_frame.csv"
    point_count = exiftool_points_to_csv(video_path, raw_csv)
    if point_count < 2:
        raise RuntimeError("Za mało punktów GPS w metadanych wideo.")
    cleaned = read_points_csv_clean(raw_csv)
    if cleaned is None:
        raise RuntimeError("Nie udało się sparsować punktów GPS.")
    ts_abs, lats, lons, alts, speeds, base_dt = cleaned
    fps, total_frames = video_fps_and_frames(video_path)
    rows = write_per_frame_csv(per_csv, ts_abs, lats, lons, alts, speeds, base_dt, fps, total_frames, max_gap_s)
    return GPSBundle(rows=rows, raw_points_csv=raw_csv, per_frame_csv=per_csv, fps=fps, total_frames=total_frames)


# ============================================================
# Model PyTorch/TorchScript
# ============================================================

def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)  # PyTorch >= 2.0
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def clean_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        kk = str(k)
        for prefix in ("module.", "model.", "net."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        out[kk] = v
    return out


def resolve_factory(factory: str):
    # factory może mieć postać /path/to/models.py:make_model albo package.module:make_model
    if ":" not in factory:
        raise ValueError("factory musi mieć format 'plik.py:funkcja' albo 'pakiet.modul:funkcja'")
    module_ref, func_name = factory.split(":", 1)
    if module_ref.endswith(".py") or "/" in module_ref:
        module_path = Path(module_ref).expanduser().resolve()
        spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Nie można załadować factory z {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    else:
        module = importlib.import_module(module_ref)
    fn = getattr(module, func_name)
    return fn


def extract_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "net", "model_weights", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(isinstance(k, str) for k in obj.keys()) and any(torch.is_tensor(v) for v in obj.values()):
            return obj
    return None


class TorchRunner:
    def __init__(self, model_path: Path, input_shape: Tuple[int, int, int], kind: str, device: str = "cuda", precision: str = "fp32", factory: Optional[str] = None, factory_args_json: Optional[str] = None):
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.kind = kind
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.precision = precision
        self.factory = factory
        self.load_mode = ""
        self.model = self._load_model(factory_args_json=factory_args_json)
        self.model.eval()
        self.model.to(self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            try:
                self.model.half()
            except Exception:
                pass
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _load_model(self, factory_args_json: Optional[str] = None):
        # 1) TorchScript
        try:
            model = torch.jit.load(str(self.model_path), map_location="cpu")
            self.load_mode = "torchscript"
            return model
        except Exception:
            pass
        # 2) Pełny nn.Module lub checkpoint
        obj = _torch_load(self.model_path, map_location="cpu")
        if isinstance(obj, torch.nn.Module):
            self.load_mode = "full_nn_module"
            return obj
        if isinstance(obj, dict):
            for key in ("model", "net", "module"):
                if key in obj and isinstance(obj[key], torch.nn.Module):
                    self.load_mode = f"checkpoint_{key}_nn_module"
                    return obj[key]
        state_dict = extract_state_dict(obj)
        if state_dict is None:
            raise RuntimeError(
                f"Nie rozpoznano formatu modelu: {self.model_path}. Obsługiwane: TorchScript, pełny nn.Module albo state_dict + --*_factory."
            )
        if not self.factory:
            raise RuntimeError(
                f"{self.model_path} wygląda na state_dict/checkpoint bez architektury. Podaj factory, np. --hrnet_factory /home/jetson/models.py:make_hrnet."
            )
        fn = resolve_factory(self.factory)
        kwargs = json.loads(factory_args_json) if factory_args_json else {}
        model = fn(**kwargs)
        missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=False)
        self.load_mode = f"state_dict_factory_missing={len(missing)}_unexpected={len(unexpected)}"
        if missing or unexpected:
            print(f"[WARN] load_state_dict strict=False dla {self.model_path}: missing={len(missing)}, unexpected={len(unexpected)}")
            if len(missing) <= 10:
                print(f"       missing: {missing}")
            if len(unexpected) <= 10:
                print(f"       unexpected: {unexpected}")
        return model

    def warmup(self, n: int = 5) -> None:
        if n <= 0:
            return
        h, w, c = self.input_shape
        dummy = np.zeros((h, w, c), dtype=np.float32)
        for _ in range(n):
            _ = self.infer(dummy)

    def infer(self, input_tensor_hwc: np.ndarray) -> Dict[str, np.ndarray]:
        x = torch.from_numpy(input_tensor_hwc).permute(2, 0, 1).unsqueeze(0).contiguous()
        x = x.to(self.device, non_blocking=True)
        if self.precision == "fp16" and self.device.type == "cuda":
            x = x.half()
        else:
            x = x.float()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        with torch.inference_mode():
            y = self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return torch_outputs_to_numpy_dict(y)


def torch_outputs_to_numpy_dict(y: Any) -> Dict[str, np.ndarray]:
    if isinstance(y, dict):
        items = list(y.items())
    elif isinstance(y, (list, tuple)):
        items = [(f"out{i}", v) for i, v in enumerate(y)]
    else:
        items = [("output", y)]
    out: Dict[str, np.ndarray] = {}
    for name, value in items:
        if torch.is_tensor(value):
            out[str(name)] = value.detach().float().cpu().numpy()
        elif isinstance(value, np.ndarray):
            out[str(name)] = value
        else:
            try:
                out[str(name)] = torch.as_tensor(value).detach().float().cpu().numpy()
            except Exception as exc:
                raise RuntimeError(f"Nieobsługiwany typ wyjścia modelu {name}: {type(value)}") from exc
    return out


# ============================================================
# Preprocess / decode / render
# ============================================================

def preprocess_seg(crop_bgr: np.ndarray, input_shape: Tuple[int, int, int]) -> np.ndarray:
    in_h, in_w, in_c = input_shape
    if in_c != 3:
        raise RuntimeError(f"Model segmentacyjny oczekuje 3 kanałów, dostał: {input_shape}")
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.float32) / 255.0


def preprocess_point(crop_bgr: np.ndarray, input_shape: Tuple[int, int, int]) -> np.ndarray:
    in_h, in_w, in_c = input_shape
    if in_c != 3:
        raise RuntimeError(f"Model punktowy oczekuje 3 kanałów, dostał: {input_shape}")
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    return (x - MEAN) / STD


def squeeze_batch(x: np.ndarray) -> np.ndarray:
    while x.ndim > 0 and x.shape[0] == 1:
        x = x[0]
    return x


def to_chw_feature(x: np.ndarray) -> np.ndarray:
    x = squeeze_batch(x)
    if x.ndim == 2:
        return x[None, :, :]
    if x.ndim != 3:
        raise RuntimeError(f"Nieoczekiwany rozmiar tensora: {x.shape}")
    if x.shape[0] in (1, 2, 3):
        return x
    if x.shape[-1] in (1, 2, 3):
        return np.transpose(x, (2, 0, 1))
    return np.transpose(x, (2, 0, 1))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def max_pool_nms_numpy(hm: np.ndarray, kernel: int = 5) -> np.ndarray:
    hm2 = hm.astype(np.float32, copy=False)
    pooled = cv2.dilate(hm2, np.ones((kernel, kernel), np.uint8))
    keep = (hm2 >= pooled - 1e-7).astype(np.float32)
    return hm2 * keep


def decode_segmentation(outputs: Dict[str, np.ndarray]) -> np.ndarray:
    if len(outputs) != 1:
        raise RuntimeError(f"Model segmentacyjny powinien mieć 1 wyjście, ma: {list(outputs.keys())}")
    logits = next(iter(outputs.values()))
    chw = to_chw_feature(logits)
    return np.argmax(chw, axis=0).astype(np.uint8)


def decode_points_outputs(outputs: Dict[str, np.ndarray], threshold: float, topk: int, nms_kernel: int, min_distance_px: float) -> List[Dict[str, float]]:
    if len(outputs) != 2:
        raise RuntimeError(f"Model punktowy powinien mieć 2 wyjścia, ma: {list(outputs.keys())}")
    tensors = {name: to_chw_feature(arr) for name, arr in outputs.items()}
    hm = None
    off = None
    for arr in tensors.values():
        if arr.shape[0] == 1:
            hm = arr
        elif arr.shape[0] == 2:
            off = arr
    if hm is None or off is None:
        raise RuntimeError(f"Nie rozpoznano heatmapy i offsetów z wyjść: {[v.shape for v in tensors.values()]}")
    hm_map = sigmoid(hm[0])
    hm_map = max_pool_nms_numpy(hm_map, kernel=nms_kernel)
    h, w = hm_map.shape
    flat = hm_map.reshape(-1)
    k = min(topk, flat.size)
    if k <= 0:
        return []
    inds = np.argpartition(-flat, k - 1)[:k]
    inds = inds[np.argsort(-flat[inds])]
    preds: List[Dict[str, float]] = []
    for idx in inds:
        score = float(flat[idx])
        if score < threshold:
            continue
        iy = int(idx // w)
        ix = int(idx % w)
        dx = float(off[0, iy, ix])
        dy = float(off[1, iy, ix])
        px = (ix + dx) * POINT_DOWN_RATIO
        py = (iy + dy) * POINT_DOWN_RATIO
        too_close = any(math.hypot(px - old["x"], py - old["y"]) < min_distance_px for old in preds)
        if not too_close:
            preds.append({"x": px, "y": py, "score": score})
    return preds


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask == 0] = (0, 0, 0)
    out[mask == 1] = (0, 255, 0)
    out[mask == 2] = (0, 0, 255)
    return out


def render_points_panel(frame_bgr: np.ndarray, crop_box: Tuple[int, int, int, int], preds_small: List[Dict[str, float]], model_name: str, model_input_w: int, model_input_h: int) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    crop = frame_bgr[y1:y2, x1:x2].copy()
    sx = crop.shape[1] / float(model_input_w)
    sy = crop.shape[0] / float(model_input_h)
    for p in preds_small:
        gx = int(round(p["x"] * sx))
        gy = int(round(p["y"] * sy))
        cv2.circle(crop, (gx, gy), 6, (0, 0, 255), 2)
        cv2.putText(crop, f"{p['score']:.2f}", (gx + 8, gy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(crop, f"{model_name} | points={len(preds_small)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return crop


def render_seg_panel(frame_bgr: np.ndarray, crop_box: Tuple[int, int, int, int], seg_mask_small: np.ndarray, model_name: str, alpha: float = 0.45) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    crop = frame_bgr[y1:y2, x1:x2].copy()
    mask = cv2.resize(seg_mask_small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = colorize_mask(mask)
    overlay = cv2.addWeighted(crop, 1.0, color, alpha, 0.0)
    cv2.putText(overlay, f"{model_name} | seg", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


# ============================================================
# Geometria / projekcja
# ============================================================

def latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    r = 6378137.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = dlon * r * math.cos(math.radians(lat0))
    y = dlat * r
    return x, y


def row_to_distance(y: int, roi_h: int, cfg: ProjectionConfig) -> float:
    if roi_h <= 1:
        return cfg.d_min
    t = (roi_h - 1 - y) / (roi_h - 1)
    if cfg.distance_mode == "linear":
        return cfg.d_min + t * (cfg.d_max - cfg.d_min)
    eps = 0.08
    z = (1.0 / (eps + (1.0 - t))) - 1.0
    z0 = (1.0 / (eps + 1.0)) - 1.0
    z1 = (1.0 / (eps + 1e-6)) - 1.0
    zn = (z - z0) / (z1 - z0 + 1e-9)
    zn = float(np.clip(zn, 0.0, 1.0))
    zn = zn ** (1.0 / cfg.gamma)
    return cfg.d_min + zn * (cfg.d_max - cfg.d_min)


def get_row_distance_lut(roi_h: int, cfg: ProjectionConfig, cache: Dict[Tuple[Any, ...], np.ndarray]) -> np.ndarray:
    key = (roi_h, cfg.d_min, cfg.d_max, cfg.distance_mode, cfg.gamma)
    if key not in cache:
        cache[key] = np.array([row_to_distance(y, roi_h, cfg) for y in range(roi_h)], dtype=np.float32)
    return cache[key]


def width_at_distance(forward_dist: float, cfg: ProjectionConfig) -> float:
    if cfg.d_max <= 1e-6:
        return cfg.width_near_m
    t = float(np.clip(forward_dist / cfg.d_max, 0.0, 1.0))
    return cfg.width_near_m + t * (cfg.width_far_m - cfg.width_near_m)


def pixel_to_lateral(x: float, roi_w: int, forward_dist: float, cfg: ProjectionConfig) -> float:
    if roi_w <= 1:
        return 0.0
    x_norm = (x / (roi_w - 1)) * 2.0 - 1.0
    width_m = width_at_distance(forward_dist, cfg)
    return x_norm * (width_m / 2.0)


def local_offset_to_world(base_x: float, base_y: float, heading_deg: float, dx_right: float, dy_forward: float) -> Tuple[float, float]:
    th = math.radians(heading_deg)
    fwd_e = math.sin(th)
    fwd_n = math.cos(th)
    right_e = math.sin(th + math.pi / 2.0)
    right_n = math.cos(th + math.pi / 2.0)
    world_x = base_x + dy_forward * fwd_e + dx_right * right_e
    world_y = base_y + dy_forward * fwd_n + dx_right * right_n
    return world_x, world_y


# ============================================================
# Metryki Jetsona
# ============================================================

def read_text_safe(path: Path) -> Optional[str]:
    try:
        return path.read_text(errors="ignore").strip()
    except Exception:
        return None


def read_number_safe(path: Path) -> Optional[float]:
    txt = read_text_safe(path)
    if txt is None or txt == "":
        return None
    try:
        return float(txt)
    except Exception:
        return None


def collect_thermal_zones() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        typ = read_text_safe(z / "type") or z.name
        temp = read_number_safe(z / "temp")
        key = sanitize_key(typ)
        if temp is not None:
            # Linux thermal zwykle jest w milistopniach C.
            out[f"thermal_{key}_c"] = temp / 1000.0 if abs(temp) > 200 else temp
    return out


def collect_hwmon() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name = sanitize_key(read_text_safe(h / "name") or h.name)
        for p in h.glob("temp*_input"):
            idx = p.name.replace("temp", "").replace("_input", "")
            label = sanitize_key(read_text_safe(h / f"temp{idx}_label") or idx)
            val = read_number_safe(p)
            if val is not None:
                out[f"hwmon_{name}_temp_{label}_c"] = val / 1000.0 if abs(val) > 200 else val
        for p in h.glob("power*_input"):
            idx = p.name.replace("power", "").replace("_input", "")
            label = sanitize_key(read_text_safe(h / f"power{idx}_label") or idx)
            val = read_number_safe(p)
            if val is not None:
                # hwmon power_input zwykle w mikrowatach.
                out[f"hwmon_{name}_power_{label}_w"] = val / 1_000_000.0
        for p in h.glob("curr*_input"):
            idx = p.name.replace("curr", "").replace("_input", "")
            label = sanitize_key(read_text_safe(h / f"curr{idx}_label") or idx)
            val = read_number_safe(p)
            if val is not None:
                out[f"hwmon_{name}_current_{label}_raw"] = val
        for p in h.glob("in*_input"):
            idx = p.name.replace("in", "").replace("_input", "")
            label = sanitize_key(read_text_safe(h / f"in{idx}_label") or idx)
            val = read_number_safe(p)
            if val is not None:
                out[f"hwmon_{name}_voltage_{label}_raw"] = val
        for p in h.glob("fan*_input"):
            idx = p.name.replace("fan", "").replace("_input", "")
            val = read_number_safe(p)
            if val is not None:
                out[f"hwmon_{name}_fan_{idx}_rpm"] = val
    return out


def collect_devfreq() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in sorted(Path("/sys/class/devfreq").glob("*")):
        if not d.is_dir():
            continue
        key = sanitize_key(d.name)
        for fname in ("cur_freq", "min_freq", "max_freq", "target_freq", "trans_stat"):
            if fname == "trans_stat":
                continue
            val = read_number_safe(d / fname)
            if val is not None:
                out[f"devfreq_{key}_{fname}_hz"] = val
        gov = read_text_safe(d / "governor")
        if gov:
            out[f"devfreq_{key}_governor"] = gov
        load = read_text_safe(d / "load")
        if load:
            nums = re.findall(r"\d+", load)
            if nums:
                out[f"devfreq_{key}_load_raw"] = float(nums[0])
    # Częsty path dla GPU load na Jetsonie.
    for p in [Path("/sys/devices/gpu.0/load"), Path("/sys/kernel/debug/gpu/load")]:
        val = read_number_safe(p)
        if val is not None:
            out[f"jetson_gpu_load_raw_{sanitize_key(str(p))}"] = val
    return out


def collect_torch_cuda_metrics() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            out["torch_cuda_device_index"] = dev
            out["torch_cuda_memory_allocated_mb"] = torch.cuda.memory_allocated(dev) / (1024 * 1024)
            out["torch_cuda_memory_reserved_mb"] = torch.cuda.memory_reserved(dev) / (1024 * 1024)
            out["torch_cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
            out["torch_cuda_max_memory_reserved_mb"] = torch.cuda.max_memory_reserved(dev) / (1024 * 1024)
        except Exception as exc:
            out["torch_cuda_metrics_error"] = str(exc)
    return out


def collect_process_system_metrics(proc: Any = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if psutil is None:
        return out
    try:
        if proc is None:
            proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        out["process_rss_mb"] = mem.rss / (1024 * 1024)
        out["process_vms_mb"] = mem.vms / (1024 * 1024)
        out["process_cpu_percent"] = proc.cpu_percent(interval=None)
        out["process_num_threads"] = proc.num_threads()
    except Exception:
        pass
    try:
        out["system_cpu_percent"] = psutil.cpu_percent(interval=None)
        for i, v in enumerate(psutil.cpu_percent(interval=None, percpu=True)):
            out[f"system_cpu_core_{i}_percent"] = v
        vm = psutil.virtual_memory()
        out["system_ram_total_mb"] = vm.total / (1024 * 1024)
        out["system_ram_available_mb"] = vm.available / (1024 * 1024)
        out["system_ram_used_mb"] = vm.used / (1024 * 1024)
        out["system_ram_percent"] = vm.percent
        sm = psutil.swap_memory()
        out["system_swap_total_mb"] = sm.total / (1024 * 1024)
        out["system_swap_used_mb"] = sm.used / (1024 * 1024)
        out["system_swap_percent"] = sm.percent
        la = os.getloadavg()
        out["loadavg_1m"] = la[0]
        out["loadavg_5m"] = la[1]
        out["loadavg_15m"] = la[2]
    except Exception:
        pass
    return out


def collect_resource_snapshot(frame_idx: int, proc: Any = None, tag: str = "") -> Dict[str, Any]:
    row: Dict[str, Any] = {"frame_idx": frame_idx, "sample_time_utc": now_utc()}
    if tag:
        row["tag"] = tag
    row.update(collect_process_system_metrics(proc))
    row.update(collect_torch_cuda_metrics())
    row.update(collect_thermal_zones())
    row.update(collect_hwmon())
    row.update(collect_devfreq())
    return row


def summarize_numeric_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    cols: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if isinstance(v, bool):
                continue
            try:
                if v is None or v == "":
                    continue
                fv = float(v)
                if math.isfinite(fv):
                    cols[k].append(fv)
            except Exception:
                continue
    out: Dict[str, Any] = {}
    for k, vals in cols.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        prefix = sanitize_key(k)
        out[f"{prefix}_count"] = int(arr.size)
        out[f"{prefix}_mean"] = float(np.mean(arr))
        out[f"{prefix}_std"] = float(np.std(arr))
        out[f"{prefix}_min"] = float(np.min(arr))
        out[f"{prefix}_p50"] = float(np.percentile(arr, 50))
        out[f"{prefix}_p90"] = float(np.percentile(arr, 90))
        out[f"{prefix}_p95"] = float(np.percentile(arr, 95))
        out[f"{prefix}_p99"] = float(np.percentile(arr, 99))
        out[f"{prefix}_max"] = float(np.max(arr))
    return out


def add_timing_summary(metrics: Dict[str, Any], timings: Dict[str, List[float]]) -> None:
    for name, values in timings.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        metrics[f"{name}_count"] = int(arr.size)
        metrics[f"{name}_mean_ms"] = float(np.mean(arr))
        metrics[f"{name}_std_ms"] = float(np.std(arr))
        metrics[f"{name}_min_ms"] = float(np.min(arr))
        metrics[f"{name}_p50_ms"] = float(np.percentile(arr, 50))
        metrics[f"{name}_p90_ms"] = float(np.percentile(arr, 90))
        metrics[f"{name}_p95_ms"] = float(np.percentile(arr, 95))
        metrics[f"{name}_p99_ms"] = float(np.percentile(arr, 99))
        metrics[f"{name}_max_ms"] = float(np.max(arr))


def collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "time_utc": now_utc(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_cuda_available": torch.cuda.is_available(),
        "cwd": os.getcwd(),
    }
    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            info.update({
                "cuda_device_index": dev,
                "cuda_device_name": torch.cuda.get_device_name(dev),
                "cuda_total_memory_mb": props.total_memory / (1024 * 1024),
                "cuda_multi_processor_count": props.multi_processor_count,
                "cuda_capability": f"{props.major}.{props.minor}",
            })
        except Exception as exc:
            info["cuda_device_error"] = str(exc)
    for p in [Path("/etc/nv_tegra_release"), Path("/etc/nvidia-container-runtime/host-files-for-container.d/l4t.csv")]:
        txt = read_text_safe(p)
        if txt:
            info[sanitize_key(str(p))] = txt[:4000]
    info["uname_a"] = run_cmd(["uname", "-a"], timeout=3)
    info["lscpu"] = run_cmd(["lscpu"], timeout=5)
    info["nvpmodel_q"] = run_cmd(["nvpmodel", "-q"], timeout=5) if have_cmd("nvpmodel") else "nvpmodel not found"
    info["jetson_clocks_show"] = run_cmd(["jetson_clocks", "--show"], timeout=8) if have_cmd("jetson_clocks") else "jetson_clocks not found"
    info["tegrastats_available"] = have_cmd("tegrastats")
    info["initial_resource_snapshot"] = collect_resource_snapshot(-1, None, tag="initial")
    return info


class TegrastatsLogger:
    def __init__(self, out_dir: Path, interval_ms: int = 1000, enabled: bool = True, prefix: str = "tegrastats"):
        self.out_dir = out_dir
        self.interval_ms = interval_ms
        self.enabled = enabled and have_cmd("tegrastats")
        self.prefix = prefix
        self.proc: Optional[subprocess.Popen] = None
        self.raw_log = out_dir / f"{prefix}_raw.log"
        self.parsed_csv = out_dir / f"{prefix}_parsed.csv"

    def start(self) -> None:
        if not self.enabled:
            return
        ensure_dir(self.out_dir)
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms), "--logfile", str(self.raw_log)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            print(f"[WARN] Nie udało się uruchomić tegrastats: {exc}")
            self.proc = None

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        self.parse_to_csv()

    def parse_to_csv(self) -> None:
        if not self.raw_log.exists():
            return
        rows = []
        for line_idx, line in enumerate(self.raw_log.read_text(errors="ignore").splitlines()):
            row = parse_tegrastats_line(line)
            if row:
                row["line_idx"] = line_idx
                row["raw"] = line
                rows.append(row)
        write_rows_csv(self.parsed_csv, rows)


def parse_tegrastats_line(line: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"sample_time_utc_parse": now_utc()}
    m = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if m:
        row["tegrastats_ram_used_mb"] = float(m.group(1))
        row["tegrastats_ram_total_mb"] = float(m.group(2))
    m = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
    if m:
        row["tegrastats_swap_used_mb"] = float(m.group(1))
        row["tegrastats_swap_total_mb"] = float(m.group(2))
    m = re.search(r"CPU\s+\[([^\]]+)\]", line)
    if m:
        vals = m.group(1).split(",")
        usages = []
        freqs = []
        for i, token in enumerate(vals):
            token = token.strip()
            if "off" in token.lower():
                row[f"tegrastats_cpu{i}_off"] = True
                continue
            mm = re.search(r"(\d+)%@([0-9]+)", token)
            if mm:
                u = float(mm.group(1)); f = float(mm.group(2))
                row[f"tegrastats_cpu{i}_percent"] = u
                row[f"tegrastats_cpu{i}_freq_mhz"] = f
                usages.append(u); freqs.append(f)
        if usages:
            row["tegrastats_cpu_mean_percent"] = float(np.mean(usages))
            row["tegrastats_cpu_max_percent"] = float(np.max(usages))
        if freqs:
            row["tegrastats_cpu_mean_freq_mhz"] = float(np.mean(freqs))
    for name in ["GR3D_FREQ", "EMC_FREQ", "VIC_FREQ", "NVENC", "NVDEC", "NVJPG", "NVDLA0", "NVDLA1", "PVA0_FREQ", "PVA1_FREQ"]:
        mm = re.search(name + r"\s+([0-9]+)%?(?:@|\[|\s)?([0-9,]+)?", line)
        if mm:
            key = sanitize_key(name)
            row[f"tegrastats_{key}_percent"] = float(mm.group(1))
            if mm.group(2):
                freqs = [float(x) for x in re.findall(r"\d+", mm.group(2))]
                if freqs:
                    row[f"tegrastats_{key}_freq_mhz_mean"] = float(np.mean(freqs))
                    row[f"tegrastats_{key}_freq_mhz_max"] = float(np.max(freqs))
    # Temperatury typu CPU@45.5C GPU@44C tj@46.5C
    for name, val in re.findall(r"([A-Za-z0-9_]+)@([0-9.]+)C", line):
        row[f"tegrastats_temp_{sanitize_key(name)}_c"] = float(val)
    # Moc typu VDD_IN 4215mW/4215mW albo POM_5V_IN 1000/1200
    for name, cur, avg in re.findall(r"([A-Za-z0-9_]+)\s+([0-9]+)m?W/([0-9]+)m?W", line):
        key = sanitize_key(name)
        row[f"tegrastats_power_{key}_current_mw"] = float(cur)
        row[f"tegrastats_power_{key}_average_mw"] = float(avg)
    return row
