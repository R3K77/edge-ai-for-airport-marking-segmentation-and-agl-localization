#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zintegrowany pipeline dla Raspberry Pi:
- wejście: pojedynczy plik wideo MP4 z telemetrią GPS osadzoną w metadanych,
- ekstrakcja GPS z pliku wideo przez ExifTool,
- interpolacja GPS do każdej klatki,
- inferencja dwoma modelami ONNX:
    1) HRNet-Lite-Point (detekcja lamp jako punkty),
    2) LinkNet + MobileNetV2 (segmentacja oznakowania poziomego),
- zapis wspólnego CSV z detekcjami,
- zapis CSV z trajektorią,
- zapis mapy PNG.

Założenia wynikające z dostarczonych skryptów:
- GPS pobierany jest z metadanych osadzonych w pliku wideo przez exiftool,
- dla segmentacji stosowany jest dolny obszar obrazu (ROI),
- dla punktów stosowany jest ten sam dolny obszar obrazu,
- pozycja obiektu w świecie wyznaczana jest przybliżeniem perspektywicznym,
  tak jak w skrypcie generującym mapę z oznakowania poziomego.

Wymagania systemowe:
- exiftool w PATH,
- Python: opencv-python, numpy, pandas, matplotlib, onnxruntime.

Przykład:
python3 airport_pipeline_raspberry.py \
  --video /home/pi/dane/GH010453.MP4 \
  --hrnet_model /home/pi/modele/hrnet-lite-point_raspb.onnx \
  --linknet_model /home/pi/modele/linknet_mobilenetv2.onnx \
  --out_dir /home/pi/wyniki/pipeline_run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd


# ============================================================
# Stałe domyślne
# ============================================================

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
POINT_DOWN_RATIO = 4


# ============================================================
# Narzędzia pomocnicze
# ============================================================


def have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_cmd(cmd: Sequence[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Błąd uruchomienia polecenia: {' '.join(cmd)}\n{exc.output}"
        ) from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# GPS z metadanych wideo
# ============================================================


def parse_exif_dt(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    s = value.strip()
    s = re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", s)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def try_float(value: str | None) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except Exception:
        return None


def haversine_m(a: Tuple[float, float] | None, b: Tuple[float, float] | None) -> Optional[float]:
    if not a or not b:
        return None
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    r = 6371000.0
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def bearing_deg(a: Tuple[float, float] | None, b: Tuple[float, float] | None) -> Optional[float]:
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
    fmt_path = out_csv.parent / "_points_tmp.fmt"
    fmt_text = (
        "#[HEAD]\n"
        "time_utc,sample_time,lat,lon,alt,speed\n"
        "#[BODY]\n"
        "$gpsdatetime,$sampletime,$gpslatitude,$gpslongitude,$gpsaltitude,$gpsspeed\n"
        "#[TAIL]\n"
    )
    fmt_path.write_text(fmt_text, encoding="ascii")

    cmd = [
        "exiftool",
        "-ee3",
        "-api",
        "largefilesupport=1",
        "-n",
        "-d",
        "%Y-%m-%d %H:%M:%S.%3fZ",
        "-p",
        str(fmt_path),
        str(video_path),
    ]
    out = run_cmd(cmd)
    out_csv.write_text(out, encoding="utf-8")
    fmt_path.unlink(missing_ok=True)

    count = 0
    for line in out.splitlines():
        s = line.strip()
        if not s or s.lower().startswith("time_utc"):
            continue
        count += 1
    return count


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
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps > 0 and frames > 0:
            return fps, frames

    out = run_cmd(["exiftool", "-n", "-s3", "-VideoFrameRate", "-MediaDuration", str(video_path)]).splitlines()
    fps = float(out[0].strip()) if out and out[0].strip() else 0.0
    dur = out[1].strip() if len(out) > 1 else ""
    m = re.match(r"(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", dur)
    if fps > 0 and m:
        hh = float(m.group(1) or 0)
        mm = float(m.group(2))
        ss = float(m.group(3))
        seconds = hh * 3600 + mm * 60 + ss
        frames = int(round(seconds * fps))
        if frames > 0:
            return fps, frames

    raise RuntimeError("Nie udało się ustalić FPS ani liczby klatek.")


def write_per_frame_csv(
    out_csv: Path,
    ts_abs: Sequence[float],
    lats: Sequence[float],
    lons: Sequence[float],
    alts: Sequence[Optional[float]],
    speeds: Sequence[Optional[float]],
    base_dt: datetime,
    fps: float,
    frames: int,
    max_gap_s: float,
) -> Dict[int, GPSPerFrameRow]:
    g_start, g_end = ts_abs[0], ts_abs[-1]
    v_start = base_dt.timestamp()

    rows: Dict[int, GPSPerFrameRow] = {}
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "pts_sec", "time_utc", "lat", "lon", "alt_m", "speed_kmh", "bearing_deg"],
        )
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
                br = bearing_deg(
                    interp_geo(ts_abs, lats, lons, tb0),
                    interp_geo(ts_abs, lats, lons, tb1),
                )

            iso = datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            row = GPSPerFrameRow(
                frame=i,
                pts_sec=round(i / fps, 6),
                time_utc=iso,
                lat=lat,
                lon=lon,
                alt_m=alt,
                speed_kmh=speed,
                bearing_deg=br,
            )
            rows[i] = row
            writer.writerow(
                {
                    "frame": row.frame,
                    "pts_sec": f"{row.pts_sec:.6f}",
                    "time_utc": row.time_utc,
                    "lat": f"{row.lat:.7f}" if row.lat is not None else "",
                    "lon": f"{row.lon:.7f}" if row.lon is not None else "",
                    "alt_m": f"{row.alt_m:.2f}" if row.alt_m is not None else "",
                    "speed_kmh": f"{row.speed_kmh:.3f}" if row.speed_kmh is not None else "",
                    "bearing_deg": f"{row.bearing_deg:.1f}" if row.bearing_deg is not None else "",
                }
            )

    return rows


def extract_gps_from_video(video_path: Path, out_dir: Path, max_gap_s: float) -> GPSBundle:
    ensure_dir(out_dir)
    base = video_path.stem
    raw_csv = out_dir / f"{base}_gps_points.csv"
    per_csv = out_dir / f"{base}_per_frame.csv"

    point_count = exiftool_points_to_csv(video_path, raw_csv)
    if point_count < 2:
        raise RuntimeError("Za mało punktów GPS w metadanych wideo.")

    cleaned = read_points_csv_clean(raw_csv)
    if cleaned is None:
        raise RuntimeError("Nie udało się sparsować punktów GPS z CSV pośredniego.")

    ts_abs, lats, lons, alts, speeds, base_dt = cleaned
    fps, total_frames = video_fps_and_frames(video_path)
    rows = write_per_frame_csv(
        out_csv=per_csv,
        ts_abs=ts_abs,
        lats=lats,
        lons=lons,
        alts=alts,
        speeds=speeds,
        base_dt=base_dt,
        fps=fps,
        frames=total_frames,
        max_gap_s=max_gap_s,
    )

    return GPSBundle(
        rows=rows,
        raw_points_csv=raw_csv,
        per_frame_csv=per_csv,
        fps=fps,
        total_frames=total_frames,
    )


# ============================================================
# ONNX i dekodowanie modeli
# ============================================================


class ONNXRunner:
    def __init__(self, model_path: Path, intra_threads: int = 0, inter_threads: int = 0):
        session_options = ort.SessionOptions()
        if intra_threads > 0:
            session_options.intra_op_num_threads = intra_threads
        if inter_threads > 0:
            session_options.inter_op_num_threads = inter_threads

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self._parse_input_shape(self.input_meta.shape)

    @staticmethod
    def _parse_input_shape(shape_raw) -> Tuple[int, int, int]:
        if len(shape_raw) != 4:
            raise RuntimeError(f"Nieoczekiwany rozmiar wejścia ONNX: {shape_raw}")
        _, a, b, c = shape_raw
        if a in (1, 3):
            return (b, c, a)
        if c in (1, 3):
            return (a, b, c)
        raise RuntimeError(f"Nie da się określić układu wejścia z: {shape_raw}")

    def infer(self, input_tensor_hwc: np.ndarray) -> Dict[str, np.ndarray]:
        x = np.transpose(input_tensor_hwc, (2, 0, 1))[None].astype(np.float32)
        outputs = self.session.run(self.output_names, {self.input_name: x})
        return {name: arr for name, arr in zip(self.output_names, outputs)}


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
    d_min: float = 2.0
    d_max: float = 45.0
    width_near_m: float = 8.0
    width_far_m: float = 35.0
    distance_mode: str = "reciprocal"
    gamma: float = 2.0


@dataclass
class InferenceTimes:
    pre_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0


@dataclass
class FrameDetection:
    frame_idx: int
    timestamp_s: float
    time_utc: str
    lat: Optional[float]
    lon: Optional[float]
    bearing_deg: Optional[float]
    speed_kmh: Optional[float]
    det_type: str
    cls_id: int
    cls_name: str
    score: Optional[float]
    x_full_px: float
    y_full_px: float
    x_crop_px: float
    y_crop_px: float
    forward_m: float
    lateral_m: float
    x_m: float
    y_m: float


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


def preprocess_point(crop_bgr: np.ndarray, input_shape: Tuple[int, int, int]) -> np.ndarray:
    in_h, in_w, in_c = input_shape
    if in_c != 3:
        raise RuntimeError(f"Model punktowy oczekuje 3 kanałów, dostał: {input_shape}")
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x


def preprocess_seg(crop_bgr: np.ndarray, input_shape: Tuple[int, int, int]) -> np.ndarray:
    in_h, in_w, in_c = input_shape
    if in_c != 3:
        raise RuntimeError(f"Model segmentacyjny oczekuje 3 kanałów, dostał: {input_shape}")
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.float32) / 255.0


def decode_points_outputs(
    outputs: Dict[str, np.ndarray],
    threshold: float,
    topk: int,
    nms_kernel: int,
    min_distance_px: float,
) -> List[Dict[str, float]]:
    if len(outputs) != 2:
        raise RuntimeError(f"Model punktowy powinien mieć 2 wyjścia, ma: {list(outputs.keys())}")

    tensors = {name: to_chw_feature(arr) for name, arr in outputs.items()}
    hm = None
    off = None
    for _, arr in tensors.items():
        if arr.shape[0] == 1:
            hm = arr
        elif arr.shape[0] == 2:
            off = arr

    if hm is None or off is None:
        raise RuntimeError("Nie udało się rozpoznać mapy ciepła i mapy przesunięć.")

    hm_map = sigmoid(hm[0])
    hm_map = max_pool_nms_numpy(hm_map, kernel=nms_kernel)

    h, w = hm_map.shape
    flat = hm_map.reshape(-1)
    k = min(topk, flat.size)
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

        too_close = False
        for old in preds:
            if math.hypot(px - old["x"], py - old["y"]) < min_distance_px:
                too_close = True
                break
        if too_close:
            continue

        preds.append({"x": px, "y": py, "score": score})

    return preds


def decode_segmentation(outputs: Dict[str, np.ndarray]) -> np.ndarray:
    if len(outputs) != 1:
        raise RuntimeError(f"Model segmentacyjny powinien mieć 1 wyjście, ma: {list(outputs.keys())}")
    logits = next(iter(outputs.values()))
    chw = to_chw_feature(logits)
    return np.argmax(chw, axis=0).astype(np.uint8)


# ============================================================
# Geometria i projekcja na mapę
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


def get_row_distance_lut(roi_h: int, cfg: ProjectionConfig, cache: Dict[Tuple, np.ndarray]) -> np.ndarray:
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


def local_offset_to_world(
    base_x: float,
    base_y: float,
    heading_deg: float,
    dx_right: float,
    dy_forward: float,
) -> Tuple[float, float]:
    th = math.radians(heading_deg)
    fwd_e = math.sin(th)
    fwd_n = math.cos(th)
    right_e = math.sin(th + math.pi / 2.0)
    right_n = math.cos(th + math.pi / 2.0)
    world_x = base_x + dy_forward * fwd_e + dx_right * right_e
    world_y = base_y + dy_forward * fwd_n + dx_right * right_n
    return world_x, world_y


# ============================================================
# Pipeline główny
# ============================================================


def color_for_class(cls_name: str) -> str:
    if cls_name == "white_mark":
        return "white"
    if cls_name == "yellow_mark":
        return "gold"
    if cls_name == "agl_light":
        return "red"
    return "cyan"


def process_video(
    video_path: Path,
    gps_bundle: GPSBundle,
    hrnet_runner: ONNXRunner,
    linknet_runner: ONNXRunner,
    out_dir: Path,
    point_cfg: PointModelConfig,
    seg_cfg: SegModelConfig,
    proj_cfg: ProjectionConfig,
    max_frames: Optional[int] = None,
    frame_step: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć wideo: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = gps_bundle.fps

    all_detections: List[FrameDetection] = []
    traj_rows: List[dict] = []
    row_distance_cache: Dict[Tuple, np.ndarray] = {}

    processed_frames = 0
    read_frames = 0
    sum_hr_pre = sum_hr_inf = sum_hr_post = 0.0
    sum_seg_pre = sum_seg_inf = sum_seg_post = 0.0
    t0 = time.perf_counter()

    valid_origin = None
    for idx in sorted(gps_bundle.rows.keys()):
        row = gps_bundle.rows[idx]
        if row.lat is not None and row.lon is not None:
            valid_origin = (row.lat, row.lon)
            break
    if valid_origin is None:
        raise RuntimeError("Brak poprawnego punktu GPS do zbudowania układu lokalnego.")
    lat0, lon0 = valid_origin

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_idx = read_frames
        read_frames += 1

        if frame_step > 1 and (frame_idx % frame_step != 0):
            continue
        if max_frames is not None and processed_frames >= max_frames:
            break

        gps = gps_bundle.rows.get(frame_idx)
        if gps is None or gps.lat is None or gps.lon is None or gps.bearing_deg is None:
            continue

        base_x, base_y = latlon_to_local_xy(gps.lat, gps.lon, lat0, lon0)
        traj_rows.append(
            {
                "frame": frame_idx,
                "time_utc": gps.time_utc,
                "lat": gps.lat,
                "lon": gps.lon,
                "bearing_deg": gps.bearing_deg,
                "speed_kmh": gps.speed_kmh,
                "x_m": base_x,
                "y_m": base_y,
            }
        )

        # ---------------- model punktowy: lampy ----------------
        y1_p = int(round(height * (1.0 - point_cfg.crop_bottom_ratio)))
        crop_p = frame_bgr[y1_p:height, 0:width]
        p_in_h, p_in_w, _ = hrnet_runner.input_shape

        t_pre = time.perf_counter()
        inp_point = preprocess_point(crop_p, hrnet_runner.input_shape)
        sum_hr_pre += (time.perf_counter() - t_pre) * 1000.0

        t_inf = time.perf_counter()
        out_point = hrnet_runner.infer(inp_point)
        sum_hr_inf += (time.perf_counter() - t_inf) * 1000.0

        t_post = time.perf_counter()
        point_preds = decode_points_outputs(
            out_point,
            threshold=point_cfg.threshold,
            topk=point_cfg.topk,
            nms_kernel=point_cfg.nms_kernel,
            min_distance_px=point_cfg.min_distance_px,
        )
        sum_hr_post += (time.perf_counter() - t_post) * 1000.0

        crop_h_p = height - y1_p
        crop_w_p = width
        sx_p = crop_w_p / float(p_in_w)
        sy_p = crop_h_p / float(p_in_h)
        row_dist_lut_p = get_row_distance_lut(crop_h_p, proj_cfg, row_distance_cache)

        for pred in point_preds:
            x_crop = float(pred["x"] * sx_p)
            y_crop = float(pred["y"] * sy_p)
            x_full = float(x_crop)
            y_full = float(y1_p + y_crop)

            y_crop_int = int(np.clip(round(y_crop), 0, crop_h_p - 1))
            forward_m = float(row_dist_lut_p[y_crop_int])
            lateral_m = float(pixel_to_lateral(x_crop, crop_w_p, forward_m, proj_cfg))
            world_x, world_y = local_offset_to_world(base_x, base_y, gps.bearing_deg, lateral_m, forward_m)

            all_detections.append(
                FrameDetection(
                    frame_idx=frame_idx,
                    timestamp_s=frame_idx / fps,
                    time_utc=gps.time_utc,
                    lat=gps.lat,
                    lon=gps.lon,
                    bearing_deg=gps.bearing_deg,
                    speed_kmh=gps.speed_kmh,
                    det_type="point",
                    cls_id=10,
                    cls_name="agl_light",
                    score=float(pred["score"]),
                    x_full_px=x_full,
                    y_full_px=y_full,
                    x_crop_px=x_crop,
                    y_crop_px=y_crop,
                    forward_m=forward_m,
                    lateral_m=lateral_m,
                    x_m=world_x,
                    y_m=world_y,
                )
            )

        # ---------------- model segmentacyjny: oznakowanie ----------------
        y1_s = int(round(height * (1.0 - seg_cfg.crop_bottom_ratio)))
        crop_s = frame_bgr[y1_s:height, 0:width]
        s_in_h, s_in_w, _ = linknet_runner.input_shape

        t_pre = time.perf_counter()
        inp_seg = preprocess_seg(crop_s, linknet_runner.input_shape)
        sum_seg_pre += (time.perf_counter() - t_pre) * 1000.0

        t_inf = time.perf_counter()
        out_seg = linknet_runner.infer(inp_seg)
        sum_seg_inf += (time.perf_counter() - t_inf) * 1000.0

        t_post = time.perf_counter()
        seg_mask_small = decode_segmentation(out_seg)
        seg_mask = cv2.resize(seg_mask_small, (crop_s.shape[1], crop_s.shape[0]), interpolation=cv2.INTER_NEAREST)
        sum_seg_post += (time.perf_counter() - t_post) * 1000.0

        ys_small, xs_small = np.where(seg_mask[::seg_cfg.sample_stride_y, ::seg_cfg.sample_stride_x] > 0)
        if xs_small.size > 0:
            xs = (xs_small.astype(np.int32) * seg_cfg.sample_stride_x).astype(np.int32)
            ys = (ys_small.astype(np.int32) * seg_cfg.sample_stride_y).astype(np.int32)
            cls = seg_mask[ys, xs].astype(np.uint8)

            row_dist_lut_s = get_row_distance_lut(crop_s.shape[0], proj_cfg, row_distance_cache)
            for x_crop_i, y_crop_i, cls_i in zip(xs.tolist(), ys.tolist(), cls.tolist()):
                if cls_i not in (1, 2):
                    continue
                forward_m = float(row_dist_lut_s[int(np.clip(y_crop_i, 0, crop_s.shape[0] - 1))])
                lateral_m = float(pixel_to_lateral(float(x_crop_i), crop_s.shape[1], forward_m, proj_cfg))
                world_x, world_y = local_offset_to_world(base_x, base_y, gps.bearing_deg, lateral_m, forward_m)
                cls_name = "white_mark" if cls_i == 1 else "yellow_mark"

                all_detections.append(
                    FrameDetection(
                        frame_idx=frame_idx,
                        timestamp_s=frame_idx / fps,
                        time_utc=gps.time_utc,
                        lat=gps.lat,
                        lon=gps.lon,
                        bearing_deg=gps.bearing_deg,
                        speed_kmh=gps.speed_kmh,
                        det_type="segmentation",
                        cls_id=int(cls_i),
                        cls_name=cls_name,
                        score=None,
                        x_full_px=float(x_crop_i),
                        y_full_px=float(y1_s + y_crop_i),
                        x_crop_px=float(x_crop_i),
                        y_crop_px=float(y_crop_i),
                        forward_m=forward_m,
                        lateral_m=lateral_m,
                        x_m=world_x,
                        y_m=world_y,
                    )
                )

        processed_frames += 1
        if processed_frames % 20 == 0:
            elapsed = time.perf_counter() - t0
            fps_proc = processed_frames / max(elapsed, 1e-6)
            print(
                f"\r[INFO] Przetworzono {processed_frames} klatek | wydajność pipeline: {fps_proc:.2f} kl./s",
                end="",
                flush=True,
            )

    cap.release()
    print()

    det_df = pd.DataFrame([d.__dict__ for d in all_detections])
    traj_df = pd.DataFrame(traj_rows)

    elapsed = time.perf_counter() - t0
    metrics = {
        "processed_frames": float(processed_frames),
        "pipeline_fps": processed_frames / max(elapsed, 1e-6),
        "avg_hrnet_pre_ms": sum_hr_pre / max(processed_frames, 1),
        "avg_hrnet_infer_ms": sum_hr_inf / max(processed_frames, 1),
        "avg_hrnet_post_ms": sum_hr_post / max(processed_frames, 1),
        "avg_linknet_pre_ms": sum_seg_pre / max(processed_frames, 1),
        "avg_linknet_infer_ms": sum_seg_inf / max(processed_frames, 1),
        "avg_linknet_post_ms": sum_seg_post / max(processed_frames, 1),
        "elapsed_s": elapsed,
    }
    return det_df, traj_df, metrics


# ============================================================
# Zapis wyników
# ============================================================


def save_metrics(metrics: Dict[str, float], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, f"{value:.6f}"])


def save_detection_csv(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        out_path.write_text("", encoding="utf-8")
        return
    ordered_cols = [
        "frame_idx",
        "timestamp_s",
        "time_utc",
        "lat",
        "lon",
        "bearing_deg",
        "speed_kmh",
        "det_type",
        "cls_id",
        "cls_name",
        "score",
        "x_full_px",
        "y_full_px",
        "x_crop_px",
        "y_crop_px",
        "forward_m",
        "lateral_m",
        "x_m",
        "y_m",
    ]
    cols = [c for c in ordered_cols if c in df.columns]
    df[cols].to_csv(out_path, index=False)


def save_trajectory_csv(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        out_path.write_text("", encoding="utf-8")
        return
    df.to_csv(out_path, index=False)


def save_map_png(det_df: pd.DataFrame, traj_df: pd.DataFrame, out_path: Path, dpi: int = 320) -> None:
    fig, ax = plt.subplots(figsize=(20, 16))

    if not traj_df.empty:
        ax.plot(traj_df["x_m"], traj_df["y_m"], color="black", linewidth=1.0, label="Trajektoria GPS")

    if not det_df.empty:
        for cls_name, label in [
            ("white_mark", "Oznakowanie białe"),
            ("yellow_mark", "Oznakowanie żółte"),
            ("agl_light", "Lampy AGL"),
        ]:
            sub = det_df[det_df["cls_name"] == cls_name]
            if not sub.empty:
                ax.scatter(
                    sub["x_m"],
                    sub["y_m"],
                    s=0.6 if cls_name != "agl_light" else 8.0,
                    c=color_for_class(cls_name),
                    edgecolors="none",
                    label=label,
                )

    ax.set_facecolor("#2f4f2f")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Mapa oznakowania poziomego i lamp AGL z GPS + analiza wizyjna")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_overlay_preview(
    video_path: Path,
    out_path: Path,
    hrnet_runner: ONNXRunner,
    linknet_runner: ONNXRunner,
    point_cfg: PointModelConfig,
    seg_cfg: SegModelConfig,
    frame_idx: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return

    h, w = frame.shape[:2]

    # segmentacja
    y1_s = int(round(h * (1.0 - seg_cfg.crop_bottom_ratio)))
    crop_s = frame[y1_s:h, 0:w]
    inp_seg = preprocess_seg(crop_s, linknet_runner.input_shape)
    out_seg = linknet_runner.infer(inp_seg)
    seg_mask_small = decode_segmentation(out_seg)
    seg_mask = cv2.resize(seg_mask_small, (crop_s.shape[1], crop_s.shape[0]), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros_like(crop_s)
    color_mask[seg_mask == 1] = (255, 255, 255)
    color_mask[seg_mask == 2] = (0, 255, 255)
    overlay = crop_s.copy()
    fg = seg_mask > 0
    overlay[fg] = ((0.45 * overlay[fg]) + (0.55 * color_mask[fg])).astype(np.uint8)

    frame_overlay = frame.copy()
    frame_overlay[y1_s:h, 0:w] = overlay

    # punkty lamp
    y1_p = int(round(h * (1.0 - point_cfg.crop_bottom_ratio)))
    crop_p = frame[y1_p:h, 0:w]
    inp_point = preprocess_point(crop_p, hrnet_runner.input_shape)
    out_point = hrnet_runner.infer(inp_point)
    point_preds = decode_points_outputs(
        out_point,
        threshold=point_cfg.threshold,
        topk=point_cfg.topk,
        nms_kernel=point_cfg.nms_kernel,
        min_distance_px=point_cfg.min_distance_px,
    )

    p_in_h, p_in_w, _ = hrnet_runner.input_shape
    sx = crop_p.shape[1] / float(p_in_w)
    sy = crop_p.shape[0] / float(p_in_h)

    for pred in point_preds:
        x = int(round(pred["x"] * sx))
        y = int(round(y1_p + pred["y"] * sy))
        cv2.circle(frame_overlay, (x, y), 7, (0, 0, 255), 2)
        cv2.putText(
            frame_overlay,
            f"{pred['score']:.2f}",
            (x + 6, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), frame_overlay)


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pełny pipeline wideo + GPS + dwa modele ONNX dla Raspberry Pi.")
    parser.add_argument("--video", required=True, help="Ścieżka do pliku MP4 z osadzonym GPS.")
    parser.add_argument("--hrnet_model", required=True, help="Ścieżka do modelu HRNet-Lite-Point w ONNX.")
    parser.add_argument("--linknet_model", required=True, help="Ścieżka do modelu LinkNet + MobileNetV2 w ONNX.")
    parser.add_argument("--out_dir", required=True, help="Folder wyjściowy.")

    parser.add_argument("--max_gap_s", type=float, default=10.0, help="Maksymalna luka GPS przy interpolacji [s].")
    parser.add_argument("--max_frames", type=int, default=0, help="Maksymalna liczba analizowanych klatek. 0 = bez limitu.")
    parser.add_argument("--frame_step", type=int, default=1, help="Analiza co N-tą klatkę.")

    parser.add_argument("--point_crop_bottom_ratio", type=float, default=0.60, help="Udział dolnej części obrazu dla modelu punktowego.")
    parser.add_argument("--point_threshold", type=float, default=0.526133828996282, help="Próg detekcji dla HRNet-Lite-Point.")
    parser.add_argument("--point_topk", type=int, default=24, help="Maksymalna liczba kandydatów punktowych na klatkę.")
    parser.add_argument("--point_min_distance_px", type=float, default=16.0, help="Minimalna odległość między punktami [px].")
    parser.add_argument("--point_nms_kernel", type=int, default=7, help="Rozmiar okna NMS dla punktów.")

    parser.add_argument("--seg_crop_bottom_ratio", type=float, default=0.60, help="Udział dolnej części obrazu dla segmentacji.")
    parser.add_argument("--seg_stride_x", type=int, default=14, help="Próbkowanie maski segmentacyjnej w osi X.")
    parser.add_argument("--seg_stride_y", type=int, default=12, help="Próbkowanie maski segmentacyjnej w osi Y.")

    parser.add_argument("--d_min", type=float, default=2.0, help="Odległość dla dolnego wiersza ROI [m].")
    parser.add_argument("--d_max", type=float, default=45.0, help="Odległość dla górnego wiersza ROI [m].")
    parser.add_argument("--width_near_m", type=float, default=8.0, help="Szerokość obserwowanego pasa przy d_min [m].")
    parser.add_argument("--width_far_m", type=float, default=35.0, help="Szerokość obserwowanego pasa przy d_max [m].")
    parser.add_argument("--distance_mode", choices=["linear", "reciprocal"], default="reciprocal", help="Model zależności odległości od wiersza obrazu.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Krzywizna dla trybu reciprocal.")

    parser.add_argument("--onnx_intra_threads", type=int, default=0, help="Liczba wątków intra-op w ONNX Runtime.")
    parser.add_argument("--onnx_inter_threads", type=int, default=0, help="Liczba wątków inter-op w ONNX Runtime.")
    parser.add_argument("--save_preview", action="store_true", help="Zapisz dodatkowy podgląd PNG z nałożeniem masek i punktów.")
    parser.add_argument("--preview_frame", type=int, default=0, help="Numer klatki do zapisu podglądu.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not have_cmd("exiftool"):
        print("[BŁĄD] Nie znaleziono exiftool w PATH.", file=sys.stderr)
        return 1

    video_path = Path(args.video)
    hrnet_model = Path(args.hrnet_model)
    linknet_model = Path(args.linknet_model)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not video_path.exists():
        print(f"[BŁĄD] Nie ma pliku wideo: {video_path}", file=sys.stderr)
        return 1
    if not hrnet_model.exists():
        print(f"[BŁĄD] Nie ma modelu HRNet: {hrnet_model}", file=sys.stderr)
        return 1
    if not linknet_model.exists():
        print(f"[BŁĄD] Nie ma modelu LinkNet: {linknet_model}", file=sys.stderr)
        return 1

    point_cfg = PointModelConfig(
        threshold=args.point_threshold,
        topk=args.point_topk,
        min_distance_px=args.point_min_distance_px,
        nms_kernel=args.point_nms_kernel,
        crop_bottom_ratio=args.point_crop_bottom_ratio,
    )
    seg_cfg = SegModelConfig(
        crop_bottom_ratio=args.seg_crop_bottom_ratio,
        sample_stride_x=args.seg_stride_x,
        sample_stride_y=args.seg_stride_y,
    )
    proj_cfg = ProjectionConfig(
        d_min=args.d_min,
        d_max=args.d_max,
        width_near_m=args.width_near_m,
        width_far_m=args.width_far_m,
        distance_mode=args.distance_mode,
        gamma=args.gamma,
    )

    print("[INFO] Ekstrakcja GPS z wideo...")
    gps_bundle = extract_gps_from_video(video_path, out_dir, max_gap_s=args.max_gap_s)
    print(f"[INFO] Zapisano GPS per frame: {gps_bundle.per_frame_csv}")

    print("[INFO] Ładowanie modelu HRNet-Lite-Point...")
    hrnet_runner = ONNXRunner(hrnet_model, intra_threads=args.onnx_intra_threads, inter_threads=args.onnx_inter_threads)
    print("[INFO] Ładowanie modelu LinkNet + MobileNetV2...")
    linknet_runner = ONNXRunner(linknet_model, intra_threads=args.onnx_intra_threads, inter_threads=args.onnx_inter_threads)

    print("[INFO] Uruchamianie pełnego pipeline'u...")
    det_df, traj_df, metrics = process_video(
        video_path=video_path,
        gps_bundle=gps_bundle,
        hrnet_runner=hrnet_runner,
        linknet_runner=linknet_runner,
        out_dir=out_dir,
        point_cfg=point_cfg,
        seg_cfg=seg_cfg,
        proj_cfg=proj_cfg,
        max_frames=(args.max_frames if args.max_frames > 0 else None),
        frame_step=max(1, args.frame_step),
    )

    detections_csv = out_dir / f"{video_path.stem}_detections_combined.csv"
    trajectory_csv = out_dir / f"{video_path.stem}_trajectory.csv"
    metrics_csv = out_dir / f"{video_path.stem}_pipeline_metrics.csv"
    map_png = out_dir / f"{video_path.stem}_map.png"

    save_detection_csv(det_df, detections_csv)
    save_trajectory_csv(traj_df, trajectory_csv)
    save_metrics(metrics, metrics_csv)
    save_map_png(det_df, traj_df, map_png)

    if args.save_preview:
        preview_png = out_dir / f"{video_path.stem}_preview.png"
        save_overlay_preview(
            video_path=video_path,
            out_path=preview_png,
            hrnet_runner=hrnet_runner,
            linknet_runner=linknet_runner,
            point_cfg=point_cfg,
            seg_cfg=seg_cfg,
            frame_idx=args.preview_frame,
        )
        print(f"[INFO] Zapisano podgląd: {preview_png}")

    print(f"[INFO] Zapisano CSV detekcji: {detections_csv}")
    print(f"[INFO] Zapisano CSV trajektorii: {trajectory_csv}")
    print(f"[INFO] Zapisano metryki pipeline'u: {metrics_csv}")
    print(f"[INFO] Zapisano mapę PNG: {map_png}")
    print("[INFO] Gotowe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
