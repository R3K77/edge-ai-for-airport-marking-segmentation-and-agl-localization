from __future__ import annotations

import csv
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import psutil
import hailo_platform as hpf


# ============================================================
# KONFIG
# ============================================================

VIDEO_PATH = Path("/home/r3k7/seg_run/video/sample.mp4")   # <- zmień jeśli trzeba
OUT_DIR = Path("/home/r3k7/seg_run/out_csv_metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_BOTTOM_RATIO = 0.60
MAX_FRAMES = None   # np. 300 do szybkiego testu

# Parametry punktowych modeli z GUI
HR_THRESHOLD = 0.2
HR_TOPK = 24
HR_MIN_DISTANCE = 16.0
HR_NMS_KERNEL = 7

UN_THRESHOLD = 0.2
UN_TOPK = 24
UN_MIN_DISTANCE = 16.0
UN_NMS_KERNEL = 7

POINT_DOWN_RATIO = 4

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ModelSpec:
    name: str
    hef_path: str
    kind: str   # "point" albo "seg"


MODELS: List[ModelSpec] = [
    ModelSpec(
        name="hrnet_point",
        hef_path="/home/r3k7/seg_run/modele/hrnet-lite-point_raspb_h8.hef",
        kind="point",
    ),
    ModelSpec(
        name="unet_point",
        hef_path="/home/r3k7/seg_run/modele/unet-point_raspb_h8.hef",
        kind="point",
    ),
    ModelSpec(
        name="linknet_seg",
        hef_path="/home/r3k7/seg_run/modele/linknet_mobilenetv2_h8.hef",
        kind="seg",
    ),
    ModelSpec(
        name="unet_seg",
        hef_path="/home/r3k7/seg_run/modele/unet_mobilenetv2_h8.hef",
        kind="seg",
    ),
]


# ============================================================
# HAILO
# ============================================================

class HailoRunner:
    def __init__(self, hef_path: str):
        self.hef_path = hef_path
        self.hef = hpf.HEF(hef_path)
        self.target = hpf.VDevice()
        self.configure_params = hpf.ConfigureParams.create_from_hef(
            self.hef,
            interface=hpf.HailoStreamInterface.PCIe,
        )
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_infos = self.hef.get_input_vstream_infos()
        self.output_infos = self.hef.get_output_vstream_infos()

        self.input_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=hpf.FormatType.FLOAT32,
        )
        self.output_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=hpf.FormatType.FLOAT32,
        )

        self.input_info = self.input_infos[0]
        self.input_name = self.input_info.name
        self.input_shape = tuple(self.input_info.shape)

        self._activation_cm = None
        self._pipeline = None

    def __enter__(self):
        self._activation_cm = self.network_group.activate(self.network_group_params)
        self._activation_cm.__enter__()
        self._pipeline = hpf.InferVStreams(
            self.network_group,
            self.input_params,
            self.output_params,
        )
        self._pipeline.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pipeline is not None:
            self._pipeline.__exit__(exc_type, exc_val, exc_tb)
        if self._activation_cm is not None:
            self._activation_cm.__exit__(exc_type, exc_val, exc_tb)
        self.target.release()

    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        batched = np.expand_dims(input_tensor.astype(np.float32, copy=False), axis=0)
        return self._pipeline.infer({self.input_name: batched})


# ============================================================
# HELPERS
# ============================================================

def hwc_from_hailo_input_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(shape) != 3:
        raise RuntimeError(f"Unexpected input shape: {shape}")

    a, b, c = shape
    if c in (1, 3):
        return a, b, c
    if a in (1, 3):
        return b, c, a
    return a, b, c


def squeeze_batch(x: np.ndarray) -> np.ndarray:
    while x.ndim > 0 and x.shape[0] == 1:
        x = x[0]
    return x


def to_chw_feature(x: np.ndarray) -> np.ndarray:
    x = squeeze_batch(x)

    if x.ndim == 2:
        return x[None, :, :]

    if x.ndim != 3:
        raise RuntimeError(f"Unexpected feature shape: {x.shape}")

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


def frame_idx_to_ts(frame_idx: int, fps: float) -> float:
    return frame_idx / fps if fps > 0 else 0.0


def get_cpu_temp_c() -> float | None:
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_path.exists():
        try:
            raw = thermal_path.read_text().strip()
            return float(raw) / 1000.0
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True,
            text=True,
            check=False,
        )
        txt = result.stdout.strip()
        if "temp=" in txt:
            txt = txt.replace("temp=", "").replace("'C", "").strip()
            return float(txt)
    except Exception:
        pass

    return None


def sample_system_metrics(proc: psutil.Process) -> tuple[float, float, float | None]:
    try:
        cpu_percent = proc.cpu_percent(interval=None)
    except Exception:
        cpu_percent = 0.0

    try:
        ram_mb = proc.memory_info().rss / (1024 * 1024)
    except Exception:
        ram_mb = 0.0

    cpu_temp_c = get_cpu_temp_c()
    return cpu_percent, ram_mb, cpu_temp_c


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_seg(crop_bgr: np.ndarray, input_shape: Tuple[int, int, int]) -> np.ndarray:
    in_h, in_w, in_c = hwc_from_hailo_input_shape(input_shape)
    if in_c != 3:
        raise RuntimeError(f"Expected 3 channels, got {input_shape}")

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    return x


def preprocess_point(crop_bgr: np.ndarray, input_shape: Tuple[int, int, int]) -> np.ndarray:
    in_h, in_w, in_c = hwc_from_hailo_input_shape(input_shape)
    if in_c != 3:
        raise RuntimeError(f"Expected 3 channels, got {input_shape}")

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x


# ============================================================
# DECODE
# ============================================================

def decode_segmentation(outputs: Dict[str, np.ndarray]) -> np.ndarray:
    if len(outputs) != 1:
        raise RuntimeError(f"Segmentation model expected 1 output, got {list(outputs.keys())}")

    logits = next(iter(outputs.values()))
    chw = to_chw_feature(logits)
    pred = np.argmax(chw, axis=0).astype(np.uint8)
    return pred


def decode_points_outputs(
    outputs: Dict[str, np.ndarray],
    threshold: float,
    topk: int,
    nms_kernel: int,
    min_distance_px: float,
) -> List[Dict[str, float]]:
    if len(outputs) != 2:
        raise RuntimeError(f"Point model expected 2 outputs, got {list(outputs.keys())}")

    tensors = {name: to_chw_feature(arr) for name, arr in outputs.items()}

    hm = None
    off = None
    for _, arr in tensors.items():
        c = arr.shape[0]
        if c == 1:
            hm = arr
        elif c == 2:
            off = arr

    if hm is None or off is None:
        raise RuntimeError(f"Could not identify hm/off outputs from {[v.shape for v in tensors.values()]}")

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


# ============================================================
# CSV
# ============================================================

def write_point_csv_header(path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "frame_idx",
            "timestamp_s",
            "det_idx",
            "x_full_px",
            "y_full_px",
            "x_crop_px",
            "y_crop_px",
            "score",
            "threshold",
            "topk",
            "nms_kernel",
            "min_distance_px",
            "model_input_w",
            "model_input_h",
            "crop_y1",
            "crop_y2",
            "video_width",
            "video_height",
        ])


def append_point_rows(
    path: Path,
    model_name: str,
    frame_idx: int,
    timestamp_s: float,
    preds_small: List[Dict[str, float]],
    crop_box: Tuple[int, int, int, int],
    video_size: Tuple[int, int],
    threshold: float,
    topk: int,
    nms_kernel: int,
    min_distance_px: float,
    model_input_w: int,
    model_input_h: int,
):
    x1, y1, x2, y2 = crop_box
    video_w, video_h = video_size
    crop_h = y2 - y1
    crop_w = x2 - x1

    sx = crop_w / float(model_input_w)
    sy = crop_h / float(model_input_h)

    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not preds_small:
            w.writerow([
                model_name, frame_idx, f"{timestamp_s:.6f}", -1,
                "", "", "", "", "",
                threshold, topk, nms_kernel, min_distance_px,
                model_input_w, model_input_h,
                y1, y2, video_w, video_h
            ])
            return

        for det_idx, p in enumerate(preds_small):
            x_crop = float(p["x"] * sx)
            y_crop = float(p["y"] * sy)
            x_full = float(x1 + x_crop)
            y_full = float(y1 + y_crop)

            w.writerow([
                model_name,
                frame_idx,
                f"{timestamp_s:.6f}",
                det_idx,
                f"{x_full:.3f}",
                f"{y_full:.3f}",
                f"{x_crop:.3f}",
                f"{y_crop:.3f}",
                f"{p['score']:.6f}",
                threshold,
                topk,
                nms_kernel,
                min_distance_px,
                model_input_w,
                model_input_h,
                y1,
                y2,
                video_w,
                video_h,
            ])


def write_seg_csv_header(path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "frame_idx",
            "timestamp_s",
            "class0_pixels",
            "class1_pixels",
            "class2_pixels",
            "class1_ratio",
            "class2_ratio",
            "crop_width",
            "crop_height",
            "video_width",
            "video_height",
            "crop_y1",
            "crop_y2",
            "model_input_w",
            "model_input_h",
        ])


def append_seg_row(
    path: Path,
    model_name: str,
    frame_idx: int,
    timestamp_s: float,
    seg_mask_small: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    video_size: Tuple[int, int],
    model_input_w: int,
    model_input_h: int,
):
    x1, y1, x2, y2 = crop_box
    video_w, video_h = video_size
    crop_h = y2 - y1
    crop_w = x2 - x1

    c0 = int((seg_mask_small == 0).sum())
    c1 = int((seg_mask_small == 1).sum())
    c2 = int((seg_mask_small == 2).sum())
    total = max(seg_mask_small.size, 1)

    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            model_name,
            frame_idx,
            f"{timestamp_s:.6f}",
            c0,
            c1,
            c2,
            f"{c1 / total:.6f}",
            f"{c2 / total:.6f}",
            crop_w,
            crop_h,
            video_w,
            video_h,
            y1,
            y2,
            model_input_w,
            model_input_h,
        ])


# ============================================================
# RUN
# ============================================================

def process_model(spec: ModelSpec, video_path: Path, out_dir: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    y1 = int(round(height * (1.0 - CROP_BOTTOM_RATIO)))
    crop_box = (0, y1, width, height)

    if spec.kind == "point":
        csv_path = out_dir / f"{spec.name}_detections.csv"
        write_point_csv_header(csv_path)
    else:
        csv_path = out_dir / f"{spec.name}_frames.csv"
        write_seg_csv_header(csv_path)

    preprocess_times = []
    infer_times = []
    post_times = []
    total_times = []
    cpu_samples = []
    ram_samples = []
    temp_samples = []

    frames_done = 0
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)

    with HailoRunner(spec.hef_path) as runner:
        in_h, in_w, _ = hwc_from_hailo_input_shape(runner.input_shape)

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if MAX_FRAMES is not None and frames_done >= MAX_FRAMES:
                break

            t_all0 = time.perf_counter()

            x1, y1, x2, y2 = crop_box
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            timestamp_s = frame_idx_to_ts(frames_done, fps_in)

            t0 = time.perf_counter()
            if spec.kind == "seg":
                inp = preprocess_seg(crop_bgr, runner.input_shape)
            else:
                inp = preprocess_point(crop_bgr, runner.input_shape)
            preprocess_times.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            outputs = runner.infer(inp)
            infer_times.append((time.perf_counter() - t1) * 1000.0)

            t2 = time.perf_counter()
            if spec.kind == "seg":
                seg = decode_segmentation(outputs)
                append_seg_row(
                    csv_path,
                    spec.name,
                    frames_done,
                    timestamp_s,
                    seg,
                    crop_box,
                    (width, height),
                    in_w,
                    in_h,
                )
            else:
                if spec.name == "hrnet_point":
                    preds = decode_points_outputs(
                        outputs,
                        threshold=HR_THRESHOLD,
                        topk=HR_TOPK,
                        nms_kernel=HR_NMS_KERNEL,
                        min_distance_px=HR_MIN_DISTANCE,
                    )
                    append_point_rows(
                        csv_path,
                        spec.name,
                        frames_done,
                        timestamp_s,
                        preds,
                        crop_box,
                        (width, height),
                        HR_THRESHOLD,
                        HR_TOPK,
                        HR_NMS_KERNEL,
                        HR_MIN_DISTANCE,
                        in_w,
                        in_h,
                    )
                else:
                    preds = decode_points_outputs(
                        outputs,
                        threshold=UN_THRESHOLD,
                        topk=UN_TOPK,
                        nms_kernel=UN_NMS_KERNEL,
                        min_distance_px=UN_MIN_DISTANCE,
                    )
                    append_point_rows(
                        csv_path,
                        spec.name,
                        frames_done,
                        timestamp_s,
                        preds,
                        crop_box,
                        (width, height),
                        UN_THRESHOLD,
                        UN_TOPK,
                        UN_NMS_KERNEL,
                        UN_MIN_DISTANCE,
                        in_w,
                        in_h,
                    )
            post_times.append((time.perf_counter() - t2) * 1000.0)

            cpu_percent, ram_mb, cpu_temp_c = sample_system_metrics(proc)
            cpu_samples.append(cpu_percent)
            ram_samples.append(ram_mb)
            if cpu_temp_c is not None:
                temp_samples.append(cpu_temp_c)

            total_times.append((time.perf_counter() - t_all0) * 1000.0)
            frames_done += 1

            if frames_done % 50 == 0:
                avg_total = 1000.0 / max(np.mean(total_times), 1e-9)
                print(f"[{spec.name}] processed={frames_done} avg_pipeline_fps={avg_total:.2f}")

    cap.release()

    avg_pre = float(np.mean(preprocess_times)) if preprocess_times else None
    avg_inf = float(np.mean(infer_times)) if infer_times else None
    avg_post = float(np.mean(post_times)) if post_times else None
    avg_total = float(np.mean(total_times)) if total_times else None
    pipeline_fps = 1000.0 / avg_total if avg_total and avg_total > 0 else None
    infer_fps_only = 1000.0 / avg_inf if avg_inf and avg_inf > 0 else None

    return {
        "model": spec.name,
        "kind": spec.kind,
        "csv_path": str(csv_path),
        "frames_processed": frames_done,
        "avg_preprocess_ms": avg_pre,
        "avg_infer_ms": avg_inf,
        "avg_postprocess_ms": avg_post,
        "avg_total_ms": avg_total,
        "infer_fps_only": infer_fps_only,
        "pipeline_fps": pipeline_fps,
        "avg_cpu_percent": float(np.mean(cpu_samples)) if cpu_samples else None,
        "avg_ram_mb": float(np.mean(ram_samples)) if ram_samples else None,
        "avg_cpu_temp_c": float(np.mean(temp_samples)) if temp_samples else None,
    }


def write_summary_csv(results: List[dict], out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Missing input video: {VIDEO_PATH}")

    results = []
    for spec in MODELS:
        print(f"\n==== RUN {spec.name} ====")
        result = process_model(spec, VIDEO_PATH, OUT_DIR)
        results.append(result)

    summary_csv = OUT_DIR / "benchmark_summary.csv"
    write_summary_csv(results, summary_csv)

    print("\n==== SUMMARY ====")
    for r in results:
        print(
            f"{r['model']:14s} | kind={r['kind']:5s} | "
            f"infer_fps={r['infer_fps_only']:.2f} | "
            f"pipeline_fps={r['pipeline_fps']:.2f} | "
            f"pre={r['avg_preprocess_ms']:.2f} ms | "
            f"infer={r['avg_infer_ms']:.2f} ms | "
            f"post={r['avg_postprocess_ms']:.2f} ms | "
            f"total={r['avg_total_ms']:.2f} ms | "
            f"cpu={r['avg_cpu_percent']:.2f}% | "
            f"ram={r['avg_ram_mb']:.2f} MB | "
            f"temp={r['avg_cpu_temp_c']:.2f} C"
        )

    print(f"\nSummary CSV: {summary_csv}")
    print(f"Outputs dir : {OUT_DIR}")


if __name__ == "__main__":
    main()
