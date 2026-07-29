#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zintegrowany pipeline dla Raspberry Pi / Hailo:
- wejście: pojedynczy plik wideo MP4 z telemetrią GPS osadzoną w metadanych,
- ekstrakcja GPS z pliku wideo przez ExifTool,
- interpolacja GPS do każdej klatki,
- inferencja dwoma modelami Hailo HEF przez HailoRT / pyHailoRT:
    1) HRNet-Lite-Point HEF (detekcja lamp jako punkty),
    2) LinkNet + MobileNetV2 HEF (segmentacja oznakowania poziomego),
- zapis wspólnego CSV z detekcjami,
- zapis CSV z trajektorią,
- zapis mapy PNG,
- zapis podstawowych metryk pipeline'u,
- zapis rozszerzonych metryk wydajnościowych do osobnych plików z dopiskiem extended.

Założenia wynikające z dostarczonych skryptów:
- GPS pobierany jest z metadanych osadzonych w pliku wideo przez exiftool,
- dla segmentacji stosowany jest dolny obszar obrazu (ROI),
- dla punktów stosowany jest ten sam dolny obszar obrazu,
- pozycja obiektu w świecie wyznaczana jest przybliżeniem perspektywicznym,
  tak jak w skrypcie generującym mapę z oznakowania poziomego.

Wymagania systemowe:
- exiftool w PATH,
- ffmpeg i ffprobe w PATH,
- Python: opencv-python, numpy, pandas, matplotlib, hailort / hailo_platform,
- opcjonalnie: psutil dla metryk CPU/RAM.

Przykład:
python3 airport_pipeline_hailo_hef_extended.py \
  --video /home/pi/dane/GH010453.MP4 \
  --hrnet_hef /home/pi/modele/hrnet-lite-point.hef \
  --linknet_hef /home/pi/modele/linknet_mobilenetv2.hef \
  --out_dir /home/pi/wyniki/pipeline_run
"""

from __future__ import annotations

import argparse
import csv
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

os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "32768")

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def ffprobe_video(video_path: Path) -> Tuple[int, int, float, Optional[int]]:
    if not have_cmd("ffprobe"):
        raise RuntimeError("Nie znaleziono ffprobe w PATH. Zainstaluj FFmpeg i dodaj folder bin do PATH.")
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(video_path)]
    data = json.loads(run_cmd(cmd))
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
    """Czyta klatki przez ffmpeg jako surowe BGR, omijając problemy OpenCV z metadanymi GoPro."""
    if not have_cmd("ffmpeg"):
        raise RuntimeError("Nie znaleziono ffmpeg w PATH. Zainstaluj FFmpeg i dodaj folder bin do PATH.")

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
        err = b""
        if proc.stderr is not None:
            err = proc.stderr.read()
            proc.stderr.close()
        proc.wait()
        if proc.returncode not in (0, None):
            msg = err.decode("utf-8", errors="ignore").strip()
            if msg:
                print(f"[WARN] ffmpeg zakończył się kodem {proc.returncode}: {msg}")


def read_one_frame_ffmpeg(video_path: Path, frame_idx: int, width: int, height: int, fps: float) -> Optional[np.ndarray]:
    if not have_cmd("ffmpeg"):
        return None
    t = max(0.0, frame_idx / max(fps, 1e-9))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{t:.6f}", "-i", str(video_path),
        "-an", "-dn", "-sn", "-map", "0:v:0",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    frame_size = width * height * 3
    assert proc.stdout is not None
    raw = proc.stdout.read(frame_size)
    proc.stdout.close()
    if proc.stderr is not None:
        proc.stderr.close()
    proc.wait()
    if len(raw) != frame_size:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()


# ============================================================
# Metryki rozszerzone
# ============================================================


def add_timing_summary(metrics: Dict[str, object], timings: Dict[str, List[float]]) -> None:
    """Dodaje statystyki min/mean/std/p50/p90/p95/p99/max dla list czasów w ms."""
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


def read_cpu_temp_c() -> Optional[float]:
    """Próbuje odczytać temperaturę CPU. Działa głównie na Raspberry Pi / Linux."""
    thermal_paths = [
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/class/hwmon/hwmon0/temp1_input"),
    ]
    for thermal_path in thermal_paths:
        if thermal_path.exists():
            try:
                return float(thermal_path.read_text().strip()) / 1000.0
            except Exception:
                pass

    if have_cmd("vcgencmd"):
        try:
            out = run_cmd(["vcgencmd", "measure_temp"])
            m = re.search(r"temp=([\d.]+)", out)
            if m:
                return float(m.group(1))
        except Exception:
            pass

    return None


def read_rpi_throttled() -> Optional[str]:
    """Zwraca surowy wynik vcgencmd get_throttled, np. throttled=0x0."""
    if not have_cmd("vcgencmd"):
        return None
    try:
        return run_cmd(["vcgencmd", "get_throttled"]).strip()
    except Exception:
        return None


def parse_rpi_throttled_flags(raw: Optional[str]) -> Dict[str, object]:
    """
    Interpretuje najważniejsze bity z vcgencmd get_throttled.
    Bity:
    0 undervoltage now, 1 frequency capped now, 2 throttled now, 3 soft temp limit now,
    16 undervoltage occurred, 17 frequency capped occurred, 18 throttling occurred,
    19 soft temp limit occurred.
    """
    result: Dict[str, object] = {}
    if not raw:
        return result
    m = re.search(r"0x([0-9a-fA-F]+)", raw)
    if not m:
        result["rpi_throttled_raw"] = raw
        return result
    value = int(m.group(1), 16)
    result["rpi_throttled_raw"] = raw
    result["rpi_throttled_int"] = value
    result["rpi_undervoltage_now"] = bool(value & (1 << 0))
    result["rpi_frequency_capped_now"] = bool(value & (1 << 1))
    result["rpi_throttled_now"] = bool(value & (1 << 2))
    result["rpi_soft_temp_limit_now"] = bool(value & (1 << 3))
    result["rpi_undervoltage_occurred"] = bool(value & (1 << 16))
    result["rpi_frequency_capped_occurred"] = bool(value & (1 << 17))
    result["rpi_throttled_occurred"] = bool(value & (1 << 18))
    result["rpi_soft_temp_limit_occurred"] = bool(value & (1 << 19))
    return result


def collect_resource_snapshot(frame_idx: int, proc) -> Dict[str, object]:
    """Zbiera próbkę zasobów. psutil jest opcjonalny."""
    row: Dict[str, object] = {
        "frame_idx": frame_idx,
        "sample_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cpu_temp_c": read_cpu_temp_c(),
    }

    if psutil is not None and proc is not None:
        try:
            mem = proc.memory_info()
            row["process_rss_mb"] = mem.rss / (1024 * 1024)
            row["process_vms_mb"] = mem.vms / (1024 * 1024)
            row["process_cpu_percent"] = proc.cpu_percent(interval=None)
        except Exception:
            pass
        try:
            row["system_cpu_percent"] = psutil.cpu_percent(interval=None)
            row["system_ram_percent"] = psutil.virtual_memory().percent
        except Exception:
            pass

    return row


def summarize_numeric_column(metrics: Dict[str, object], df: pd.DataFrame, col: str, prefix: str) -> None:
    if df.empty or col not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return
    metrics[f"{prefix}_mean"] = float(s.mean())
    metrics[f"{prefix}_std"] = float(s.std(ddof=0))
    metrics[f"{prefix}_min"] = float(s.min())
    metrics[f"{prefix}_p50"] = float(s.quantile(0.50))
    metrics[f"{prefix}_p90"] = float(s.quantile(0.90))
    metrics[f"{prefix}_p95"] = float(s.quantile(0.95))
    metrics[f"{prefix}_p99"] = float(s.quantile(0.99))
    metrics[f"{prefix}_max"] = float(s.max())


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
    try:
        _w, _h, fps, frames = ffprobe_video(video_path)
        if fps > 0 and frames is not None and frames > 0:
            return fps, frames
    except Exception as exc:
        print(f"[WARN] ffprobe nie ustalił liczby klatek: {exc}")

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

    raise RuntimeError("Nie udało się ustalić FPS ani liczby klatek. Zainstaluj FFmpeg/ffprobe.")


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
# HailoRT / HEF i dekodowanie modeli
# ============================================================


class HailoEnvironment:
    """Wspólny kontekst HailoRT dla kilku modeli HEF."""

    def __init__(self, interface: str = "PCIe", scheduling_algorithm: str = "ROUND_ROBIN"):
        try:
            import hailo_platform as hp  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Nie udało się zaimportować hailo_platform. "
                "Na Raspberry Pi z AI Kit zwykle trzeba mieć zainstalowany pakiet hailo-all "
                "oraz uruchamiać skrypt w środowisku widzącym systemowe site-packages, "
                "np. python3 -m venv --system-site-packages venv."
            ) from exc

        self.hp = hp
        self.interface_name = interface
        self.scheduling_algorithm_name = scheduling_algorithm

        params = hp.VDevice.create_params()
        if hasattr(hp, "HailoSchedulingAlgorithm") and scheduling_algorithm:
            if not hasattr(hp.HailoSchedulingAlgorithm, scheduling_algorithm):
                allowed = [x for x in dir(hp.HailoSchedulingAlgorithm) if x.isupper()]
                raise RuntimeError(
                    f"Nieznany HailoSchedulingAlgorithm={scheduling_algorithm}. Dostępne: {allowed}"
                )
            params.scheduling_algorithm = getattr(hp.HailoSchedulingAlgorithm, scheduling_algorithm)

        if not hasattr(hp.HailoStreamInterface, interface):
            allowed = [x for x in dir(hp.HailoStreamInterface) if x.isupper() or x == "PCIe"]
            raise RuntimeError(f"Nieznany HailoStreamInterface={interface}. Dostępne: {allowed}")
        self.interface = getattr(hp.HailoStreamInterface, interface)

        # Przy aktywnym schedulerze HailoRT nie należy wołać network_group.activate()
        # przy każdej inferencji. W nowszych wersjach HailoRT takie wywołania są
        # oznaczone jako deprecated i będą błędem.
        sched = (scheduling_algorithm or "").upper()
        self.scheduler_enabled = sched not in ("", "NONE", "DISABLED", "NO_SCHEDULING")
        self.target = hp.VDevice(params=params)
        self._runners: List[Any] = []

    def register_runner(self, runner: Any) -> None:
        self._runners.append(runner)

    def close(self) -> None:
        for runner in reversed(getattr(self, "_runners", [])):
            try:
                runner.close()
            except Exception:
                pass
        try:
            self.target.release()
        except Exception:
            pass


def _hailo_format_name(fmt: Any) -> str:
    try:
        return str(fmt.name)
    except Exception:
        return str(fmt)


class HailoRunner:
    """
    Minimalny wrapper na Hailo HEF.

    Zakłada jeden input i dowolną liczbę outputów. Wejście do infer() jest w układzie HWC,
    a outputy zwracane są jako dict[name -> ndarray], tak jak wcześniej przy HailoRunner.
    """

    def __init__(
        self,
        model_path: Path,
        env: HailoEnvironment,
        input_format: str = "auto",
        output_format: str = "FLOAT32",
    ):
        self.model_path = Path(model_path)
        self.env = env
        hp = env.hp

        self.hef = hp.HEF(str(self.model_path))
        self.input_infos = list(self.hef.get_input_vstream_infos())
        self.output_infos = list(self.hef.get_output_vstream_infos())
        if len(self.input_infos) != 1:
            raise RuntimeError(f"Ten pipeline obsługuje jeden input HEF, znaleziono: {len(self.input_infos)}")
        if not self.output_infos:
            raise RuntimeError("HEF nie ma output vstreamów.")

        self.input_info = self.input_infos[0]
        self.input_name = self.input_info.name
        self.output_names = [o.name for o in self.output_infos]
        self.input_shape = self._parse_input_shape(self.input_info.shape)
        self.input_format = self._resolve_format(input_format, self.input_info)
        self.output_format = self._resolve_format(output_format, self.output_infos[0])
        self.input_format_name = _hailo_format_name(self.input_format)
        self.output_format_name = _hailo_format_name(self.output_format)

        configure_params = hp.ConfigureParams.create_from_hef(hef=self.hef, interface=env.interface)
        network_groups = env.target.configure(self.hef, configure_params)
        if not network_groups:
            raise RuntimeError(f"Nie udało się skonfigurować HEF: {self.model_path}")
        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_params = self._make_input_params()
        self.output_vstream_params = self._make_output_params()

        self._activation_cm = None
        self._infer_pipeline_cm = None
        self._infer_pipeline = None
        env.register_runner(self)
        self.open()

    @staticmethod
    def _parse_input_shape(shape_raw) -> Tuple[int, int, int]:
        shape = tuple(int(x) for x in shape_raw)
        if len(shape) == 3:
            a, b, c = shape
            if c in (1, 3):
                return (a, b, c)  # HWC, typowe dla Hailo vstream
            if a in (1, 3):
                return (b, c, a)  # CHW
        if len(shape) == 4:
            n, a, b, c = shape
            if c in (1, 3):
                return (a, b, c)  # NHWC
            if a in (1, 3):
                return (b, c, a)  # NCHW
        raise RuntimeError(f"Nie da się określić układu wejścia HEF z: {shape_raw}")

    def _resolve_format(self, requested: str, stream_info: Any):
        hp = self.env.hp
        req = (requested or "auto").upper()
        if req == "AUTO":
            try:
                return stream_info.format.type
            except Exception:
                return None
        if not hasattr(hp.FormatType, req):
            allowed = [x for x in dir(hp.FormatType) if x.isupper()]
            raise RuntimeError(f"Nieznany FormatType={requested}. Dostępne: {allowed}")
        return getattr(hp.FormatType, req)

    def _make_input_params(self):
        hp = self.env.hp
        try:
            if self.input_format is None:
                return hp.InputVStreamParams.make(self.network_group)
            return hp.InputVStreamParams.make(self.network_group, format_type=self.input_format)
        except TypeError:
            return hp.InputVStreamParams.make(self.network_group)

    def _make_output_params(self):
        hp = self.env.hp
        try:
            if self.output_format is None:
                return hp.OutputVStreamParams.make(self.network_group)
            return hp.OutputVStreamParams.make(self.network_group, format_type=self.output_format)
        except TypeError:
            return hp.OutputVStreamParams.make(self.network_group)

    def resolve_preprocess_mode(self, model_kind: str, requested: str) -> str:
        req = (requested or "auto").lower()
        if req != "auto":
            return req
        fmt = self.input_format_name.upper()
        if "UINT8" in fmt:
            return "rgb_uint8"
        if model_kind == "point":
            return "imagenet_float"
        return "rgb_float01"

    def _prepare_input_dtype(self, x: np.ndarray) -> np.ndarray:
        fmt = self.input_format_name.upper()
        if "UINT8" in fmt:
            if x.dtype != np.uint8:
                x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
            return np.ascontiguousarray(x)
        if "UINT16" in fmt:
            if x.dtype != np.uint16:
                x = np.clip(np.rint(x), 0, 65535).astype(np.uint16)
            return np.ascontiguousarray(x)
        return np.ascontiguousarray(x.astype(np.float32, copy=False))

    def open(self) -> None:
        """Otwiera VStreams raz na cały czas życia runnera.

        Przy schedulerze HailoRT, np. ROUND_ROBIN, nie wywołujemy activate(), bo
        HailoRT zgłasza to jako deprecated. Bez schedulera zostawiamy activate(),
        ale tylko raz, a nie dla każdej klatki.
        """
        if self._infer_pipeline is not None:
            return

        hp = self.env.hp
        if not getattr(self.env, "scheduler_enabled", False):
            self._activation_cm = self.network_group.activate(self.network_group_params)
            self._activation_cm.__enter__()

        self._infer_pipeline_cm = hp.InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params,
        )
        self._infer_pipeline = self._infer_pipeline_cm.__enter__()

    def close(self) -> None:
        if self._infer_pipeline_cm is not None:
            try:
                self._infer_pipeline_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._infer_pipeline_cm = None
            self._infer_pipeline = None

        if self._activation_cm is not None:
            try:
                self._activation_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._activation_cm = None

    def infer(self, input_tensor_hwc: np.ndarray) -> Dict[str, np.ndarray]:
        x = self._prepare_input_dtype(input_tensor_hwc)
        if tuple(x.shape) != tuple(self.input_shape):
            raise RuntimeError(f"Niepoprawny kształt wejścia dla HEF: got={x.shape}, expected={self.input_shape}")

        if self._infer_pipeline is None:
            self.open()

        input_data = {self.input_name: np.expand_dims(x, axis=0)}
        results = self._infer_pipeline.infer(input_data)
        return {name: results[name] for name in self.output_names if name in results}


def hailort_version_string() -> str:
    if have_cmd("hailortcli"):
        try:
            return run_cmd(["hailortcli", "--version"]).strip()
        except Exception:
            pass
    return ""

@dataclass
class PointModelConfig:
    threshold: float = 0.526133828996282
    topk: int = 24
    min_distance_px: float = 16.0
    nms_kernel: int = 7
    crop_bottom_ratio: float = 0.50


@dataclass
class SegModelConfig:
    crop_bottom_ratio: float = 0.50
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


@dataclass
class PipelineProcessResult:
    det_df: pd.DataFrame
    traj_df: pd.DataFrame
    metrics: Dict[str, float]
    extended_metrics: Dict[str, object]
    frame_metrics_df: pd.DataFrame
    resource_metrics_df: pd.DataFrame


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


def preprocess_image_for_hailo(crop_bgr: np.ndarray, input_shape: Tuple[int, int, int], mode: str) -> np.ndarray:
    in_h, in_w, in_c = input_shape
    if in_c != 3:
        raise RuntimeError(f"Model oczekuje 3 kanałów, dostał: {input_shape}")

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    mode = (mode or "rgb_uint8").lower()
    if mode == "rgb_uint8":
        return np.ascontiguousarray(rgb.astype(np.uint8))
    if mode == "rgb_float01":
        return np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
    if mode == "imagenet_float":
        x = rgb.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return np.ascontiguousarray(x.astype(np.float32))

    raise RuntimeError(
        f"Nieznany tryb preprocessingu: {mode}. "
        "Dostępne: auto, rgb_uint8, rgb_float01, imagenet_float."
    )


def preprocess_point(crop_bgr: np.ndarray, input_shape: Tuple[int, int, int], mode: str = "imagenet_float") -> np.ndarray:
    return preprocess_image_for_hailo(crop_bgr, input_shape, mode)


def preprocess_seg(crop_bgr: np.ndarray, input_shape: Tuple[int, int, int], mode: str = "rgb_float01") -> np.ndarray:
    return preprocess_image_for_hailo(crop_bgr, input_shape, mode)


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
    hrnet_runner: HailoRunner,
    linknet_runner: HailoRunner,
    out_dir: Path,
    point_cfg: PointModelConfig,
    seg_cfg: SegModelConfig,
    proj_cfg: ProjectionConfig,
    max_frames: Optional[int] = None,
    frame_step: int = 1,
    resource_sample_every: int = 10,
    point_preprocess_mode: str = "auto",
    seg_preprocess_mode: str = "auto",
) -> PipelineProcessResult:
    width, height, fps_probe, _total_probe = ffprobe_video(video_path)
    fps = gps_bundle.fps if gps_bundle.fps > 0 else fps_probe
    point_mode = hrnet_runner.resolve_preprocess_mode("point", point_preprocess_mode)
    seg_mode = linknet_runner.resolve_preprocess_mode("seg", seg_preprocess_mode)

    all_detections: List[FrameDetection] = []
    traj_rows: List[dict] = []
    frame_metric_rows: List[Dict[str, object]] = []
    resource_rows: List[Dict[str, object]] = []
    row_distance_cache: Dict[Tuple, np.ndarray] = {}
    timings: Dict[str, List[float]] = defaultdict(list)

    processed_frames = 0
    read_frames = 0
    candidate_frames = 0
    skipped_no_gps = 0
    skipped_by_max_frames = 0

    # Zachowane oryginalne sumy metryk.
    sum_hr_pre = sum_hr_inf = sum_hr_post = 0.0
    sum_seg_pre = sum_seg_inf = sum_seg_post = 0.0
    t0 = time.perf_counter()

    proc = None
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            proc.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            proc = None

    rpi_throttled_start = read_rpi_throttled()

    valid_origin = None
    for idx in sorted(gps_bundle.rows.keys()):
        row = gps_bundle.rows[idx]
        if row.lat is not None and row.lon is not None:
            valid_origin = (row.lat, row.lon)
            break
    if valid_origin is None:
        raise RuntimeError("Brak poprawnego punktu GPS do zbudowania układu lokalnego.")
    lat0, lon0 = valid_origin

    reader = ffmpeg_frame_reader(video_path, width, height, frame_step=max(1, frame_step))
    while True:
        t_read = time.perf_counter()
        try:
            frame_idx, frame_bgr = next(reader)
        except StopIteration:
            break
        read_ms = (time.perf_counter() - t_read) * 1000.0
        timings["ffmpeg_decode_read"].append(read_ms)
        candidate_frames += 1
        read_frames = frame_idx + 1

        if max_frames is not None and processed_frames >= max_frames:
            skipped_by_max_frames += 1
            break

        t_frame_total = time.perf_counter()

        gps = gps_bundle.rows.get(frame_idx)
        if gps is None or gps.lat is None or gps.lon is None or gps.bearing_deg is None:
            skipped_no_gps += 1
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
        inp_point = preprocess_point(crop_p, hrnet_runner.input_shape, mode=point_mode)
        hr_pre_ms = (time.perf_counter() - t_pre) * 1000.0
        sum_hr_pre += hr_pre_ms
        timings["hrnet_pre"].append(hr_pre_ms)

        t_inf = time.perf_counter()
        out_point = hrnet_runner.infer(inp_point)
        hr_inf_ms = (time.perf_counter() - t_inf) * 1000.0
        sum_hr_inf += hr_inf_ms
        timings["hrnet_infer"].append(hr_inf_ms)

        t_post = time.perf_counter()
        point_preds = decode_points_outputs(
            out_point,
            threshold=point_cfg.threshold,
            topk=point_cfg.topk,
            nms_kernel=point_cfg.nms_kernel,
            min_distance_px=point_cfg.min_distance_px,
        )
        hr_post_ms = (time.perf_counter() - t_post) * 1000.0
        sum_hr_post += hr_post_ms
        timings["hrnet_post"].append(hr_post_ms)

        crop_h_p = height - y1_p
        crop_w_p = width
        sx_p = crop_w_p / float(p_in_w)
        sy_p = crop_h_p / float(p_in_h)
        row_dist_lut_p = get_row_distance_lut(crop_h_p, proj_cfg, row_distance_cache)

        point_detection_count = 0
        for pred in point_preds:
            point_detection_count += 1
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
        inp_seg = preprocess_seg(crop_s, linknet_runner.input_shape, mode=seg_mode)
        seg_pre_ms = (time.perf_counter() - t_pre) * 1000.0
        sum_seg_pre += seg_pre_ms
        timings["linknet_pre"].append(seg_pre_ms)

        t_inf = time.perf_counter()
        out_seg = linknet_runner.infer(inp_seg)
        seg_inf_ms = (time.perf_counter() - t_inf) * 1000.0
        sum_seg_inf += seg_inf_ms
        timings["linknet_infer"].append(seg_inf_ms)

        t_post = time.perf_counter()
        seg_mask_small = decode_segmentation(out_seg)
        seg_mask = cv2.resize(seg_mask_small, (crop_s.shape[1], crop_s.shape[0]), interpolation=cv2.INTER_NEAREST)
        seg_post_ms = (time.perf_counter() - t_post) * 1000.0
        sum_seg_post += seg_post_ms
        timings["linknet_post"].append(seg_post_ms)

        segmentation_sample_count = 0
        white_mark_sample_count = 0
        yellow_mark_sample_count = 0

        ys_small, xs_small = np.where(seg_mask[::seg_cfg.sample_stride_y, ::seg_cfg.sample_stride_x] > 0)
        if xs_small.size > 0:
            xs = (xs_small.astype(np.int32) * seg_cfg.sample_stride_x).astype(np.int32)
            ys = (ys_small.astype(np.int32) * seg_cfg.sample_stride_y).astype(np.int32)
            cls = seg_mask[ys, xs].astype(np.uint8)

            row_dist_lut_s = get_row_distance_lut(crop_s.shape[0], proj_cfg, row_distance_cache)
            for x_crop_i, y_crop_i, cls_i in zip(xs.tolist(), ys.tolist(), cls.tolist()):
                if cls_i not in (1, 2):
                    continue
                segmentation_sample_count += 1
                if cls_i == 1:
                    white_mark_sample_count += 1
                elif cls_i == 2:
                    yellow_mark_sample_count += 1

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

        frame_total_ms = (time.perf_counter() - t_frame_total) * 1000.0
        timings["frame_total"].append(frame_total_ms)

        frame_metric_rows.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / fps,
                "time_utc": gps.time_utc,
                "ffmpeg_decode_read_ms": read_ms,
                "hrnet_pre_ms": hr_pre_ms,
                "hrnet_infer_ms": hr_inf_ms,
                "hrnet_post_ms": hr_post_ms,
                "linknet_pre_ms": seg_pre_ms,
                "linknet_infer_ms": seg_inf_ms,
                "linknet_post_ms": seg_post_ms,
                "frame_total_ms": frame_total_ms,
                "point_detections": point_detection_count,
                "segmentation_samples": segmentation_sample_count,
                "white_mark_samples": white_mark_sample_count,
                "yellow_mark_samples": yellow_mark_sample_count,
                "total_detections_or_samples": point_detection_count + segmentation_sample_count,
                "speed_kmh": gps.speed_kmh,
                "bearing_deg": gps.bearing_deg,
            }
        )

        if resource_sample_every > 0 and processed_frames % resource_sample_every == 0:
            resource_rows.append(collect_resource_snapshot(frame_idx, proc))

        processed_frames += 1
        if processed_frames % 20 == 0:
            elapsed = time.perf_counter() - t0
            fps_proc = processed_frames / max(elapsed, 1e-6)
            print(
                f"\r[INFO] Przetworzono {processed_frames} klatek | wydajność pipeline: {fps_proc:.2f} kl./s",
                end="",
                flush=True,
            )

    print()

    det_df = pd.DataFrame([d.__dict__ for d in all_detections])
    traj_df = pd.DataFrame(traj_rows)
    frame_metrics_df = pd.DataFrame(frame_metric_rows)
    resource_metrics_df = pd.DataFrame(resource_rows)

    elapsed = time.perf_counter() - t0

    # Oryginalne metryki zostają zachowane bez usuwania.
    metrics: Dict[str, float] = {
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

    extended_metrics: Dict[str, object] = dict(metrics)
    add_timing_summary(extended_metrics, timings)

    extended_metrics["candidate_frames"] = int(candidate_frames)
    extended_metrics["read_frames_last_index_plus_one"] = int(read_frames)
    extended_metrics["skipped_no_gps_frames"] = int(skipped_no_gps)
    extended_metrics["skipped_by_max_frames"] = int(skipped_by_max_frames)
    extended_metrics["gps_coverage_ratio_processed_over_candidate"] = float(processed_frames / max(candidate_frames, 1))
    extended_metrics["input_video_fps"] = float(fps)
    extended_metrics["frame_step"] = int(max(1, frame_step))
    extended_metrics["analyzed_frames_per_second"] = float(processed_frames / max(elapsed, 1e-6))
    extended_metrics["ms_per_analyzed_frame"] = float(1000.0 * elapsed / max(processed_frames, 1))
    extended_metrics["video_seconds_processed_nominal"] = float(processed_frames * max(1, frame_step) / max(fps, 1e-9))
    extended_metrics["video_real_time_factor"] = float(
        (processed_frames * max(1, frame_step) / max(fps, 1e-9)) / max(elapsed, 1e-6)
    )
    extended_metrics["resource_sample_every"] = int(resource_sample_every)
    extended_metrics["rpi_throttled_start"] = rpi_throttled_start or ""
    rpi_throttled_end = read_rpi_throttled()
    extended_metrics["rpi_throttled_end"] = rpi_throttled_end or ""
    extended_metrics.update({f"start_{k}": v for k, v in parse_rpi_throttled_flags(rpi_throttled_start).items()})
    extended_metrics.update({f"end_{k}": v for k, v in parse_rpi_throttled_flags(rpi_throttled_end).items()})

    if not frame_metrics_df.empty:
        summarize_numeric_column(extended_metrics, frame_metrics_df, "point_detections", "point_detections_per_frame")
        summarize_numeric_column(extended_metrics, frame_metrics_df, "segmentation_samples", "segmentation_samples_per_frame")
        summarize_numeric_column(extended_metrics, frame_metrics_df, "white_mark_samples", "white_mark_samples_per_frame")
        summarize_numeric_column(extended_metrics, frame_metrics_df, "yellow_mark_samples", "yellow_mark_samples_per_frame")
        summarize_numeric_column(extended_metrics, frame_metrics_df, "total_detections_or_samples", "total_detections_or_samples_per_frame")
        extended_metrics["frames_without_point_detections"] = int((frame_metrics_df["point_detections"] == 0).sum())
        extended_metrics["frames_without_segmentation_samples"] = int((frame_metrics_df["segmentation_samples"] == 0).sum())
        extended_metrics["frames_without_any_detection_or_sample"] = int((frame_metrics_df["total_detections_or_samples"] == 0).sum())

    if not det_df.empty and "cls_name" in det_df.columns:
        extended_metrics["detections_total_rows"] = int(len(det_df))
        extended_metrics["detections_agl_light_rows"] = int((det_df["cls_name"] == "agl_light").sum())
        extended_metrics["detections_white_mark_rows"] = int((det_df["cls_name"] == "white_mark").sum())
        extended_metrics["detections_yellow_mark_rows"] = int((det_df["cls_name"] == "yellow_mark").sum())
    else:
        extended_metrics["detections_total_rows"] = 0
        extended_metrics["detections_agl_light_rows"] = 0
        extended_metrics["detections_white_mark_rows"] = 0
        extended_metrics["detections_yellow_mark_rows"] = 0

    if not traj_df.empty:
        extended_metrics["trajectory_rows"] = int(len(traj_df))
    else:
        extended_metrics["trajectory_rows"] = 0

    if not resource_metrics_df.empty:
        summarize_numeric_column(extended_metrics, resource_metrics_df, "process_rss_mb", "process_rss_mb")
        summarize_numeric_column(extended_metrics, resource_metrics_df, "process_vms_mb", "process_vms_mb")
        summarize_numeric_column(extended_metrics, resource_metrics_df, "process_cpu_percent", "process_cpu_percent")
        summarize_numeric_column(extended_metrics, resource_metrics_df, "system_cpu_percent", "system_cpu_percent")
        summarize_numeric_column(extended_metrics, resource_metrics_df, "system_ram_percent", "system_ram_percent")
        summarize_numeric_column(extended_metrics, resource_metrics_df, "cpu_temp_c", "cpu_temp_c")

    return PipelineProcessResult(
        det_df=det_df,
        traj_df=traj_df,
        metrics=metrics,
        extended_metrics=extended_metrics,
        frame_metrics_df=frame_metrics_df,
        resource_metrics_df=resource_metrics_df,
    )


# ============================================================
# Zapis wyników
# ============================================================


def save_metrics(metrics: Dict[str, object], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                writer.writerow([key, f"{float(value):.6f}"])
            else:
                writer.writerow([key, str(value)])


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
    hrnet_runner: HailoRunner,
    linknet_runner: HailoRunner,
    point_cfg: PointModelConfig,
    seg_cfg: SegModelConfig,
    frame_idx: int,
    point_preprocess_mode: str = "auto",
    seg_preprocess_mode: str = "auto",
) -> None:
    try:
        w, h, fps, _ = ffprobe_video(video_path)
        frame = read_one_frame_ffmpeg(video_path, frame_idx, w, h, fps)
    except Exception:
        frame = None
    if frame is None:
        print("[WARN] Nie udało się odczytać klatki podglądu przez ffmpeg.")
        return

    h, w = frame.shape[:2]

    # segmentacja
    y1_s = int(round(h * (1.0 - seg_cfg.crop_bottom_ratio)))
    crop_s = frame[y1_s:h, 0:w]
    seg_mode = linknet_runner.resolve_preprocess_mode("seg", seg_preprocess_mode)
    inp_seg = preprocess_seg(crop_s, linknet_runner.input_shape, mode=seg_mode)
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
    point_mode = hrnet_runner.resolve_preprocess_mode("point", point_preprocess_mode)
    inp_point = preprocess_point(crop_p, hrnet_runner.input_shape, mode=point_mode)
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
# Okna wyboru plików / folderu dla Windows
# ============================================================


def choose_paths_with_dialogs(args: argparse.Namespace) -> argparse.Namespace:
    """Uzupełnia brakujące ścieżki przez standardowe okna systemowe."""
    need_dialog = not args.video or not args.hrnet_model or not args.linknet_model or not args.out_dir
    if not need_dialog:
        return args

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:
        raise RuntimeError(
            "Nie udało się uruchomić okien wyboru plików. "
            "Podaj ścieżki ręcznie przez argumenty --video, --hrnet_model, --linknet_model i --out_dir."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.update()

    try:
        if not args.video:
            args.video = filedialog.askopenfilename(
                title="Wybierz plik wideo GoPro / MP4 z danymi GPS",
                filetypes=[("Pliki wideo", "*.mp4 *.MP4 *.mov *.MOV"), ("Wszystkie pliki", "*.*")],
            )
            if not args.video:
                raise RuntimeError("Nie wybrano pliku wideo.")

        if not args.hrnet_model:
            args.hrnet_model = filedialog.askopenfilename(
                title="Wybierz model HRNet-Lite-Point (.hef)",
                filetypes=[("Modele Hailo HEF", "*.hef *.HEF"), ("Wszystkie pliki", "*.*")],
            )
            if not args.hrnet_model:
                raise RuntimeError("Nie wybrano modelu HRNet-Lite-Point HEF.")

        if not args.linknet_model:
            args.linknet_model = filedialog.askopenfilename(
                title="Wybierz model LinkNet + MobileNetV2 (.hef)",
                filetypes=[("Modele Hailo HEF", "*.hef *.HEF"), ("Wszystkie pliki", "*.*")],
            )
            if not args.linknet_model:
                raise RuntimeError("Nie wybrano modelu LinkNet + MobileNetV2 HEF.")

        if not args.out_dir:
            args.out_dir = filedialog.askdirectory(
                title="Wybierz folder wyjściowy na CSV, metryki i mapę"
            )
            if not args.out_dir:
                raise RuntimeError("Nie wybrano folderu wyjściowego.")

        try:
            import tkinter.ttk as ttk
            win = tk.Toplevel(root)
            win.title("Parametry trapezu / projekcji na mapę")
            win.resizable(False, False)
            ttk.Label(win, text="Ustawienia trapezu jak w gps_ground_map.py.\nDomyślne: frame_step=2, roi_top_ratio=0.5, d_min=0.5, d_max=5.5, width_near=1.5, width_far=9.0.", justify="left").grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 8), sticky="w")
            fields = [
                ("frame_step", "Analizuj co N-tą klatkę", args.frame_step),
                ("roi_top_ratio", "roi_top_ratio", 1.0 - args.seg_crop_bottom_ratio),
                ("d_min", "d_min [m]", args.d_min),
                ("d_max", "d_max [m]", args.d_max),
                ("width_near_m", "width_near_m [m]", args.width_near_m),
                ("width_far_m", "width_far_m [m]", args.width_far_m),
                ("seg_stride_x", "Próbkowanie segmentacji X", args.seg_stride_x),
                ("seg_stride_y", "Próbkowanie segmentacji Y", args.seg_stride_y),
                ("gamma", "gamma", args.gamma),
            ]
            entries = {}
            for i, (key, label, val) in enumerate(fields, start=1):
                ttk.Label(win, text=label).grid(row=i, column=0, padx=12, pady=4, sticky="w")
                e = ttk.Entry(win, width=18)
                e.insert(0, str(val))
                e.grid(row=i, column=1, padx=12, pady=4, sticky="e")
                entries[key] = e
            ttk.Label(win, text="distance_mode").grid(row=len(fields)+1, column=0, padx=12, pady=4, sticky="w")
            mode_var = tk.StringVar(value=args.distance_mode)
            ttk.Combobox(win, textvariable=mode_var, values=["reciprocal", "linear"], state="readonly", width=15).grid(row=len(fields)+1, column=1, padx=12, pady=4, sticky="e")
            def _float(key):
                return float(entries[key].get().replace(",", "."))
            def _int(key):
                return int(float(entries[key].get().replace(",", ".")))
            def accept():
                try:
                    args.frame_step = max(1, _int("frame_step"))
                    roi_top_ratio = _float("roi_top_ratio")
                    if not (0.0 <= roi_top_ratio < 1.0):
                        raise ValueError("roi_top_ratio musi być w zakresie [0, 1).")
                    args.seg_crop_bottom_ratio = 1.0 - roi_top_ratio
                    args.point_crop_bottom_ratio = 1.0 - roi_top_ratio
                    args.d_min = _float("d_min")
                    args.d_max = _float("d_max")
                    args.width_near_m = _float("width_near_m")
                    args.width_far_m = _float("width_far_m")
                    args.seg_stride_x = max(1, _int("seg_stride_x"))
                    args.seg_stride_y = max(1, _int("seg_stride_y"))
                    args.gamma = _float("gamma")
                    args.distance_mode = mode_var.get()
                    win.destroy()
                except Exception as exc:
                    messagebox.showerror("Błąd parametrów", str(exc))
            btn = ttk.Frame(win)
            btn.grid(row=len(fields)+2, column=0, columnspan=2, padx=12, pady=12, sticky="e")
            ttk.Button(btn, text="Uruchom z tymi parametrami", command=accept).pack(side="left", padx=4)
            ttk.Button(btn, text="Zostaw domyślne", command=win.destroy).pack(side="left", padx=4)
            win.grab_set()
            root.wait_window(win)
        except Exception as exc:
            try:
                messagebox.showwarning("Parametry trapezu", f"Nie udało się otworzyć okna parametrów. Używam wartości domyślnych.\n\n{exc}")
            except Exception:
                pass

        try:
            messagebox.showinfo(
                "Pipeline lotniskowy",
                "Wybrano pliki i parametry. Przetwarzanie rozpocznie się w oknie konsoli."
            )
        except Exception:
            pass
    finally:
        root.destroy()

    return args


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pełny pipeline wideo + GPS + dwa modele Hailo HEF. Na Linux/Raspberry Pi podaj ścieżki przez CLI.")
    parser.add_argument("--video", default=None, help="Ścieżka do pliku MP4 z osadzonym GPS.")
    parser.add_argument("--hrnet_hef", "--hrnet_model", dest="hrnet_model", default=None, help="Ścieżka do modelu HRNet-Lite-Point w formacie .hef.")
    parser.add_argument("--linknet_hef", "--linknet_model", dest="linknet_model", default=None, help="Ścieżka do modelu LinkNet + MobileNetV2 w formacie .hef.")
    parser.add_argument("--out_dir", default=None, help="Folder wyjściowy.")

    parser.add_argument("--max_gap_s", type=float, default=10.0, help="Maksymalna luka GPS przy interpolacji [s].")
    parser.add_argument("--max_frames", type=int, default=0, help="Maksymalna liczba analizowanych klatek. 0 = bez limitu.")
    parser.add_argument("--frame_step", type=int, default=2, help="Analiza co N-tą klatkę.")

    parser.add_argument("--point_crop_bottom_ratio", type=float, default=0.50, help="Udział dolnej części obrazu dla modelu punktowego. Odpowiada roi_top_ratio=0.5.")
    parser.add_argument("--point_threshold", type=float, default=0.526133828996282, help="Próg detekcji dla HRNet-Lite-Point.")
    parser.add_argument("--point_topk", type=int, default=24, help="Maksymalna liczba kandydatów punktowych na klatkę.")
    parser.add_argument("--point_min_distance_px", type=float, default=16.0, help="Minimalna odległość między punktami [px].")
    parser.add_argument("--point_nms_kernel", type=int, default=7, help="Rozmiar okna NMS dla punktów.")

    parser.add_argument("--seg_crop_bottom_ratio", type=float, default=0.50, help="Udział dolnej części obrazu dla segmentacji. Odpowiada roi_top_ratio=0.5.")
    parser.add_argument("--seg_stride_x", type=int, default=14, help="Próbkowanie maski segmentacyjnej w osi X.")
    parser.add_argument("--seg_stride_y", type=int, default=12, help="Próbkowanie maski segmentacyjnej w osi Y.")

    parser.add_argument("--d_min", type=float, default=0.5, help="Odległość dla dolnego wiersza ROI [m].")
    parser.add_argument("--d_max", type=float, default=5.5, help="Odległość dla górnego wiersza ROI [m].")
    parser.add_argument("--width_near_m", type=float, default=1.5, help="Szerokość obserwowanego pasa przy d_min [m].")
    parser.add_argument("--width_far_m", type=float, default=9.0, help="Szerokość obserwowanego pasa przy d_max [m].")
    parser.add_argument("--distance_mode", choices=["linear", "reciprocal"], default="reciprocal", help="Model zależności odległości od wiersza obrazu.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Krzywizna dla trybu reciprocal.")

    parser.add_argument("--hailo_interface", default="PCIe", help="HailoStreamInterface, zwykle PCIe na Raspberry Pi AI Kit.")
    parser.add_argument("--hailo_scheduling_algorithm", default="ROUND_ROBIN", help="HailoSchedulingAlgorithm, np. ROUND_ROBIN albo NONE.")
    parser.add_argument("--hailo_input_format", default="auto", choices=["auto", "UINT8", "UINT16", "FLOAT32"], help="Format wejściowego VStream. auto = format z HEF.")
    parser.add_argument("--hailo_output_format", default="FLOAT32", choices=["auto", "UINT8", "UINT16", "FLOAT32"], help="Format wyjściowego VStream. FLOAT32 ułatwia dekodowanie po stronie CPU.")
    parser.add_argument("--hailo_point_preprocess", default="auto", choices=["auto", "rgb_uint8", "rgb_float01", "imagenet_float"], help="Preprocessing wejścia HRNet HEF.")
    parser.add_argument("--hailo_seg_preprocess", default="auto", choices=["auto", "rgb_uint8", "rgb_float01", "imagenet_float"], help="Preprocessing wejścia LinkNet HEF.")

    parser.add_argument("--resource_sample_every", type=int, default=10, help="Co ile przetworzonych klatek zapisywać próbkę CPU/RAM/temperatury do resource_metrics_extended.csv. 0 = wyłącz.")
    parser.add_argument("--save_preview", action="store_true", help="Zapisz dodatkowy podgląd PNG z nałożeniem masek i punktów.")
    parser.add_argument("--preview_frame", type=int, default=0, help="Numer klatki do zapisu podglądu.")
    return parser.parse_args()


def main() -> int:
    args = choose_paths_with_dialogs(parse_args())

    if not have_cmd("exiftool"):
        print("[BŁĄD] Nie znaleziono exiftool w PATH.", file=sys.stderr)
        return 1
    if not have_cmd("ffmpeg") or not have_cmd("ffprobe"):
        print("[BŁĄD] Nie znaleziono ffmpeg/ffprobe w PATH. Zainstaluj FFmpeg i dodaj folder bin do PATH.", file=sys.stderr)
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
        print(f"[BŁĄD] Nie ma pliku HEF HRNet: {hrnet_model}", file=sys.stderr)
        return 1
    if not linknet_model.exists():
        print(f"[BŁĄD] Nie ma pliku HEF LinkNet: {linknet_model}", file=sys.stderr)
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

    extended_outer_metrics: Dict[str, object] = {}
    total_script_t0 = time.perf_counter()
    hailo_env: Optional[HailoEnvironment] = None

    print("[INFO] Ekstrakcja GPS z wideo...")
    t_gps = time.perf_counter()
    gps_bundle = extract_gps_from_video(video_path, out_dir, max_gap_s=args.max_gap_s)
    gps_extract_s = time.perf_counter() - t_gps
    extended_outer_metrics["gps_extract_s"] = gps_extract_s
    print(f"[INFO] Zapisano GPS per frame: {gps_bundle.per_frame_csv}")

    try:
        print("[INFO] Inicjalizacja HailoRT / VDevice...")
        t_hailo_env = time.perf_counter()
        hailo_env = HailoEnvironment(
            interface=args.hailo_interface,
            scheduling_algorithm=args.hailo_scheduling_algorithm,
        )
        extended_outer_metrics["hailo_env_init_s"] = time.perf_counter() - t_hailo_env

        print("[INFO] Ładowanie modelu HRNet-Lite-Point HEF...")
        t_load = time.perf_counter()
        hrnet_runner = HailoRunner(
            hrnet_model,
            env=hailo_env,
            input_format=args.hailo_input_format,
            output_format=args.hailo_output_format,
        )
        hrnet_load_s = time.perf_counter() - t_load
        extended_outer_metrics["hrnet_load_s"] = hrnet_load_s

        print("[INFO] Ładowanie modelu LinkNet + MobileNetV2 HEF...")
        t_load = time.perf_counter()
        linknet_runner = HailoRunner(
            linknet_model,
            env=hailo_env,
            input_format=args.hailo_input_format,
            output_format=args.hailo_output_format,
        )
        linknet_load_s = time.perf_counter() - t_load
        extended_outer_metrics["linknet_load_s"] = linknet_load_s

        point_mode = hrnet_runner.resolve_preprocess_mode("point", args.hailo_point_preprocess)
        seg_mode = linknet_runner.resolve_preprocess_mode("seg", args.hailo_seg_preprocess)

        print("[INFO] Ustawienia Hailo:")
        print(f"       interface={args.hailo_interface}")
        print(f"       scheduling_algorithm={args.hailo_scheduling_algorithm}")
        print(f"       input_format={hrnet_runner.input_format_name} / {linknet_runner.input_format_name}")
        print(f"       output_format={hrnet_runner.output_format_name} / {linknet_runner.output_format_name}")
        print(f"       point_preprocess={point_mode}")
        print(f"       seg_preprocess={seg_mode}")

        print("[INFO] Ustawienia trapezu / projekcji:")
        print(f"       frame_step={args.frame_step}")
        print(f"       roi_top_ratio={1.0 - seg_cfg.crop_bottom_ratio:.3f}  (crop_bottom_ratio={seg_cfg.crop_bottom_ratio:.3f})")
        print(f"       d_min={proj_cfg.d_min}, d_max={proj_cfg.d_max}")
        print(f"       width_near_m={proj_cfg.width_near_m}, width_far_m={proj_cfg.width_far_m}")
        print(f"       distance_mode={proj_cfg.distance_mode}, gamma={proj_cfg.gamma}")

        print("[INFO] Uruchamianie pełnego pipeline'u Hailo...")
        result = process_video(
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
            resource_sample_every=max(0, args.resource_sample_every),
            point_preprocess_mode=args.hailo_point_preprocess,
            seg_preprocess_mode=args.hailo_seg_preprocess,
        )

        det_df = result.det_df
        traj_df = result.traj_df
        metrics = result.metrics
        extended_metrics = dict(result.extended_metrics)
        extended_metrics.update(extended_outer_metrics)

        detections_csv = out_dir / f"{video_path.stem}_detections_combined.csv"
        trajectory_csv = out_dir / f"{video_path.stem}_trajectory.csv"
        metrics_csv = out_dir / f"{video_path.stem}_pipeline_metrics.csv"
        metrics_extended_csv = out_dir / f"{video_path.stem}_pipeline_metrics_extended.csv"
        frame_metrics_extended_csv = out_dir / f"{video_path.stem}_frame_metrics_extended.csv"
        resource_metrics_extended_csv = out_dir / f"{video_path.stem}_resource_metrics_extended.csv"
        map_png = out_dir / f"{video_path.stem}_map.png"

        t_save = time.perf_counter()
        save_detection_csv(det_df, detections_csv)
        extended_metrics["detection_csv_save_s"] = time.perf_counter() - t_save

        t_save = time.perf_counter()
        save_trajectory_csv(traj_df, trajectory_csv)
        extended_metrics["trajectory_csv_save_s"] = time.perf_counter() - t_save

        t_save = time.perf_counter()
        save_metrics(metrics, metrics_csv)
        extended_metrics["basic_metrics_csv_save_s"] = time.perf_counter() - t_save

        t_save = time.perf_counter()
        save_map_png(det_df, traj_df, map_png)
        extended_metrics["map_png_save_s"] = time.perf_counter() - t_save

        if args.save_preview:
            preview_png = out_dir / f"{video_path.stem}_preview.png"
            t_save = time.perf_counter()
            save_overlay_preview(
                video_path=video_path,
                out_path=preview_png,
                hrnet_runner=hrnet_runner,
                linknet_runner=linknet_runner,
                point_cfg=point_cfg,
                seg_cfg=seg_cfg,
                frame_idx=args.preview_frame,
                point_preprocess_mode=args.hailo_point_preprocess,
                seg_preprocess_mode=args.hailo_seg_preprocess,
            )
            extended_metrics["preview_png_save_s"] = time.perf_counter() - t_save
            print(f"[INFO] Zapisano podgląd: {preview_png}")

        # Zapis per-frame i resource extended do osobnych plików.
        t_save = time.perf_counter()
        if result.frame_metrics_df.empty:
            frame_metrics_extended_csv.write_text("", encoding="utf-8")
        else:
            result.frame_metrics_df.to_csv(frame_metrics_extended_csv, index=False)
        extended_metrics["frame_metrics_extended_csv_save_s"] = time.perf_counter() - t_save

        t_save = time.perf_counter()
        if result.resource_metrics_df.empty:
            resource_metrics_extended_csv.write_text("", encoding="utf-8")
        else:
            result.resource_metrics_df.to_csv(resource_metrics_extended_csv, index=False)
        extended_metrics["resource_metrics_extended_csv_save_s"] = time.perf_counter() - t_save

        # Metadane eksperymentu i parametrów uruchomienia.
        extended_metrics["total_script_elapsed_s"] = time.perf_counter() - total_script_t0
        extended_metrics["video_path"] = str(video_path)
        extended_metrics["hrnet_hef_path"] = str(hrnet_model)
        extended_metrics["linknet_hef_path"] = str(linknet_model)
        extended_metrics["out_dir"] = str(out_dir)
        extended_metrics["video_file_size_mb"] = video_path.stat().st_size / (1024 * 1024)
        extended_metrics["hrnet_hef_size_mb"] = hrnet_model.stat().st_size / (1024 * 1024)
        extended_metrics["linknet_hef_size_mb"] = linknet_model.stat().st_size / (1024 * 1024)
        extended_metrics["opencv_version"] = cv2.__version__
        extended_metrics["hailortcli_version"] = hailort_version_string()
        extended_metrics["python_version"] = sys.version.replace("\n", " ")
        extended_metrics["platform"] = platform.platform()
        extended_metrics["machine"] = platform.machine()
        extended_metrics["processor"] = platform.processor()
        extended_metrics["psutil_available"] = psutil is not None
        extended_metrics["hrnet_input_shape"] = str(hrnet_runner.input_shape)
        extended_metrics["linknet_input_shape"] = str(linknet_runner.input_shape)
        extended_metrics["hrnet_input_name"] = str(hrnet_runner.input_name)
        extended_metrics["linknet_input_name"] = str(linknet_runner.input_name)
        extended_metrics["hrnet_output_names"] = str(hrnet_runner.output_names)
        extended_metrics["linknet_output_names"] = str(linknet_runner.output_names)
        extended_metrics["hailo_interface"] = str(args.hailo_interface)
        extended_metrics["hailo_scheduling_algorithm"] = str(args.hailo_scheduling_algorithm)
        extended_metrics["hailo_input_format_arg"] = str(args.hailo_input_format)
        extended_metrics["hailo_output_format_arg"] = str(args.hailo_output_format)
        extended_metrics["hrnet_input_format"] = str(hrnet_runner.input_format_name)
        extended_metrics["linknet_input_format"] = str(linknet_runner.input_format_name)
        extended_metrics["hrnet_output_format"] = str(hrnet_runner.output_format_name)
        extended_metrics["linknet_output_format"] = str(linknet_runner.output_format_name)
        extended_metrics["hailo_point_preprocess_arg"] = str(args.hailo_point_preprocess)
        extended_metrics["hailo_seg_preprocess_arg"] = str(args.hailo_seg_preprocess)
        extended_metrics["resolved_point_preprocess"] = str(point_mode)
        extended_metrics["resolved_seg_preprocess"] = str(seg_mode)
        extended_metrics["max_gap_s"] = float(args.max_gap_s)
        extended_metrics["max_frames_arg"] = int(args.max_frames)
        extended_metrics["point_crop_bottom_ratio"] = float(args.point_crop_bottom_ratio)
        extended_metrics["point_threshold"] = float(args.point_threshold)
        extended_metrics["point_topk"] = int(args.point_topk)
        extended_metrics["point_min_distance_px"] = float(args.point_min_distance_px)
        extended_metrics["point_nms_kernel"] = int(args.point_nms_kernel)
        extended_metrics["seg_crop_bottom_ratio"] = float(args.seg_crop_bottom_ratio)
        extended_metrics["seg_stride_x"] = int(args.seg_stride_x)
        extended_metrics["seg_stride_y"] = int(args.seg_stride_y)
        extended_metrics["d_min"] = float(args.d_min)
        extended_metrics["d_max"] = float(args.d_max)
        extended_metrics["width_near_m"] = float(args.width_near_m)
        extended_metrics["width_far_m"] = float(args.width_far_m)
        extended_metrics["distance_mode"] = str(args.distance_mode)
        extended_metrics["gamma"] = float(args.gamma)
        extended_metrics["save_preview"] = bool(args.save_preview)
        extended_metrics["preview_frame"] = int(args.preview_frame)

        t_save = time.perf_counter()
        save_metrics(extended_metrics, metrics_extended_csv)
        extended_metrics_save_s = time.perf_counter() - t_save

        print(f"[INFO] Zapisano CSV detekcji: {detections_csv}")
        print(f"[INFO] Zapisano CSV trajektorii: {trajectory_csv}")
        print(f"[INFO] Zapisano metryki pipeline'u: {metrics_csv}")
        print(f"[INFO] Zapisano metryki rozszerzone: {metrics_extended_csv}")
        print(f"[INFO] Zapisano metryki per-frame extended: {frame_metrics_extended_csv}")
        print(f"[INFO] Zapisano metryki zasobów extended: {resource_metrics_extended_csv}")
        print(f"[INFO] Zapisano mapę PNG: {map_png}")
        print(f"[INFO] Czas zapisu metrics_extended CSV: {extended_metrics_save_s:.6f} s")
        print("[INFO] Gotowe.")
        return 0
    finally:
        if hailo_env is not None:
            hailo_env.close()



if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        print(f"[BŁĄD] {exc}", file=sys.stderr)
        if os.name == "nt":
            input("\nNaciśnij Enter, aby zamknąć okno...")
        raise SystemExit(1)

    if os.name == "nt":
        input("\nGotowe. Naciśnij Enter, aby zamknąć okno...")
    raise SystemExit(code)
