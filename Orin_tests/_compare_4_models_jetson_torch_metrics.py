#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analog skryptów Raspberry `_compare_4_models_cpu_csv_metrics.py` oraz
`_compare_4_models_cpu_csv_and_video_metrics.py`, ale dla NVIDIA Jetson + PyTorch.

Najważniejsze założenie zgodności z Raspberry:
- odczyt klatki z wideo NIE wchodzi do avg_total_ms / pipeline_fps,
- avg_total_ms = preprocess + inferencja + postprocess + zapis CSV/wideo,
- pipeline_fps = 1000 / avg_total_ms,
- infer_fps_only = 1000 / avg_infer_ms.

Dodatkowo zapisuje `overall_wall_clock_fps`, czyli pełny czas skryptu z dekodowaniem FFmpeg.
Tego pola nie porównuj bezpośrednio z Raspberry, jeżeli w Raspberry używasz `pipeline_fps`.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from jetson_torch_common import (
    TorchRunner,
    TegrastatsLogger,
    add_timing_summary,
    collect_resource_snapshot,
    collect_system_info,
    decode_points_outputs,
    decode_segmentation,
    ensure_dir,
    ffmpeg_frame_reader,
    ffprobe_video,
    make_video_writer,
    parse_size,
    preprocess_point,
    preprocess_seg,
    render_points_panel,
    render_seg_panel,
    summarize_numeric_rows,
    write_json,
    write_rows_csv,
)


@dataclass
class ModelSpec:
    name: str
    model_path: Path
    kind: str  # "point" albo "seg"
    input_shape: Tuple[int, int, int]
    factory: Optional[str] = None
    factory_args_json: Optional[str] = None
    threshold: float = 0.5
    topk: int = 24
    min_distance_px: float = 16.0
    nms_kernel: int = 7
    crop_bottom_ratio: float = 0.60


def frame_idx_to_ts(frame_idx: int, fps: float) -> float:
    return frame_idx / fps if fps > 0 else 0.0


def write_point_csv_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "model", "frame_idx", "timestamp_s", "det_idx",
            "x_full_px", "y_full_px", "x_crop_px", "y_crop_px", "score",
            "threshold", "topk", "nms_kernel", "min_distance_px",
            "model_input_w", "model_input_h", "crop_y1", "crop_y2",
            "video_width", "video_height",
        ])


def append_point_rows(
    path: Path,
    model_name: str,
    frame_idx: int,
    timestamp_s: float,
    preds_small: List[Dict[str, float]],
    crop_box: Tuple[int, int, int, int],
    video_size: Tuple[int, int],
    cfg: ModelSpec,
) -> None:
    x1, y1, x2, y2 = crop_box
    video_w, video_h = video_size
    crop_h = y2 - y1
    crop_w = x2 - x1
    in_h, in_w, _ = cfg.input_shape
    sx = crop_w / float(in_w)
    sy = crop_h / float(in_h)

    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not preds_small:
            w.writerow([
                model_name, frame_idx, f"{timestamp_s:.6f}", -1,
                "", "", "", "", "",
                cfg.threshold, cfg.topk, cfg.nms_kernel, cfg.min_distance_px,
                in_w, in_h, y1, y2, video_w, video_h,
            ])
            return
        for det_idx, p in enumerate(preds_small):
            x_crop = float(p["x"] * sx)
            y_crop = float(p["y"] * sy)
            x_full = float(x1 + x_crop)
            y_full = float(y1 + y_crop)
            w.writerow([
                model_name, frame_idx, f"{timestamp_s:.6f}", det_idx,
                f"{x_full:.3f}", f"{y_full:.3f}", f"{x_crop:.3f}", f"{y_crop:.3f}", f"{p['score']:.6f}",
                cfg.threshold, cfg.topk, cfg.nms_kernel, cfg.min_distance_px,
                in_w, in_h, y1, y2, video_w, video_h,
            ])


def write_seg_csv_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "model", "frame_idx", "timestamp_s",
            "class0_pixels", "class1_pixels", "class2_pixels",
            "class1_ratio", "class2_ratio",
            "crop_width", "crop_height", "video_width", "video_height",
            "crop_y1", "crop_y2", "model_input_w", "model_input_h",
        ])


def append_seg_row(
    path: Path,
    model_name: str,
    frame_idx: int,
    timestamp_s: float,
    seg_mask_small: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    video_size: Tuple[int, int],
    input_shape: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = crop_box
    video_w, video_h = video_size
    in_h, in_w, _ = input_shape
    c0 = int((seg_mask_small == 0).sum())
    c1 = int((seg_mask_small == 1).sum())
    c2 = int((seg_mask_small == 2).sum())
    total = max(int(seg_mask_small.size), 1)
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            model_name, frame_idx, f"{timestamp_s:.6f}",
            c0, c1, c2, f"{c1 / total:.6f}", f"{c2 / total:.6f}",
            x2 - x1, y2 - y1, video_w, video_h, y1, y2, in_w, in_h,
        ])


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _fps_from_ms(ms: Optional[float]) -> Optional[float]:
    return 1000.0 / ms if ms and ms > 0 else None


def process_model(spec: ModelSpec, video_path: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    model_dir = out_dir / spec.name
    ensure_dir(model_dir)

    width, height, fps_in, total_frames_probe = ffprobe_video(video_path)
    if args.max_seconds and args.max_seconds > 0:
        max_original_frame_idx = int(round(fps_in * args.max_seconds))
    else:
        max_original_frame_idx = total_frames_probe if total_frames_probe and total_frames_probe > 0 else None

    y1 = int(round(height * (1.0 - spec.crop_bottom_ratio)))
    crop_box = (0, y1, width, height)
    crop_h = height - y1
    crop_w = width

    if spec.kind == "point":
        csv_path = model_dir / f"{spec.name}_detections.csv"
        write_point_csv_header(csv_path)
    else:
        csv_path = model_dir / f"{spec.name}_frames.csv"
        write_seg_csv_header(csv_path)

    video_writer, video_path_out = make_video_writer(
        model_dir / spec.name,
        crop_w,
        crop_h,
        fps_in / max(1, args.frame_step),
        args.video_mode,
    )

    print(f"\n==== RUN {spec.name} ====")
    print(f"[INFO] model={spec.model_path} kind={spec.kind} input={spec.input_shape} device={args.device} precision={args.precision}")

    t_load = time.perf_counter()
    runner = TorchRunner(
        model_path=spec.model_path,
        input_shape=spec.input_shape,
        kind=spec.kind,
        device=args.device,
        precision=args.precision,
        factory=spec.factory,
        factory_args_json=spec.factory_args_json,
    )
    load_s = time.perf_counter() - t_load
    print(f"[INFO] load_mode={runner.load_mode} load_s={load_s:.3f}")

    if args.warmup > 0:
        print(f"[INFO] Warmup: {args.warmup} inferencji")
        runner.warmup(args.warmup)

    proc = None
    if args.collect_psutil:
        try:
            import psutil  # type: ignore
            proc = psutil.Process(os.getpid())
            proc.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            proc = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tegra = TegrastatsLogger(
        model_dir,
        interval_ms=args.tegrastats_interval_ms,
        enabled=not args.disable_tegrastats,
        prefix=f"tegrastats_{spec.name}",
    )

    preprocess_times: List[float] = []
    infer_times: List[float] = []
    post_times: List[float] = []
    total_times: List[float] = []
    video_read_times: List[float] = []
    cpu_samples: List[float] = []
    ram_samples: List[float] = []
    temp_samples: List[float] = []

    frame_rows: List[Dict[str, Any]] = []
    resource_rows: List[Dict[str, Any]] = []
    infer_errors = 0
    frames_done = 0
    last_frame_idx = -1

    tegra.start()
    t_script0 = time.perf_counter()

    try:
        reader = ffmpeg_frame_reader(video_path, width, height, frame_step=max(1, args.frame_step))
        for frame_idx, frame_bgr in reader:
            last_frame_idx = frame_idx
            if max_original_frame_idx is not None and frame_idx >= max_original_frame_idx:
                break

            t_read0 = time.perf_counter()
            # Klatka jest już odczytana z generatora; read_ms mierzy wyłącznie narzut odbioru z pipe.
            read_ms = (time.perf_counter() - t_read0) * 1000.0
            video_read_times.append(read_ms)

            timestamp_s = frame_idx_to_ts(frame_idx, fps_in)
            x1, y1, x2, y2 = crop_box
            crop_bgr = frame_bgr[y1:y2, x1:x2]

            # ZGODNIE Z RASPBERRY: total zaczyna się PO odczycie klatki.
            t_all0 = time.perf_counter()

            t0 = time.perf_counter()
            inp = preprocess_seg(crop_bgr, runner.input_shape) if spec.kind == "seg" else preprocess_point(crop_bgr, runner.input_shape)
            pre_ms = (time.perf_counter() - t0) * 1000.0
            preprocess_times.append(pre_ms)

            t1 = time.perf_counter()
            try:
                outputs = runner.infer(inp)
            except Exception as exc:
                infer_errors += 1
                print(f"[BŁĄD] Inferencja {spec.name}, frame={frame_idx}: {exc}")
                if args.stop_on_error:
                    raise
                continue
            inf_ms = (time.perf_counter() - t1) * 1000.0
            infer_times.append(inf_ms)

            t2 = time.perf_counter()
            detections_or_pixels = 0
            if spec.kind == "seg":
                seg = decode_segmentation(outputs)
                append_seg_row(csv_path, spec.name, frame_idx, timestamp_s, seg, crop_box, (width, height), runner.input_shape)
                detections_or_pixels = int((seg > 0).sum())
                if video_writer is not None:
                    video_writer.write(render_seg_panel(frame_bgr, crop_box, seg, spec.name, alpha=args.seg_alpha))
            else:
                preds = decode_points_outputs(outputs, spec.threshold, spec.topk, spec.nms_kernel, spec.min_distance_px)
                append_point_rows(csv_path, spec.name, frame_idx, timestamp_s, preds, crop_box, (width, height), spec)
                detections_or_pixels = len(preds)
                if video_writer is not None:
                    in_h, in_w, _ = runner.input_shape
                    video_writer.write(render_points_panel(frame_bgr, crop_box, preds, spec.name, in_w, in_h))
            post_ms = (time.perf_counter() - t2) * 1000.0
            post_times.append(post_ms)

            if args.resource_sample_every > 0 and frames_done % args.resource_sample_every == 0:
                resource = collect_resource_snapshot(frame_idx, proc, tag=spec.name)
                resource_rows.append(resource)
                if "process_cpu_percent" in resource and resource["process_cpu_percent"] not in (None, ""):
                    cpu_samples.append(float(resource["process_cpu_percent"]))
                if "process_rss_mb" in resource and resource["process_rss_mb"] not in (None, ""):
                    ram_samples.append(float(resource["process_rss_mb"]))
                temp_key = None
                for k in resource.keys():
                    if k.endswith("thermal_zone0_temp_c") or k.endswith("cpu_temp_c") or k == "cpu_temp_c":
                        temp_key = k
                        break
                if temp_key and resource.get(temp_key) not in (None, ""):
                    try:
                        temp_samples.append(float(resource[temp_key]))
                    except Exception:
                        pass

            total_ms = (time.perf_counter() - t_all0) * 1000.0
            total_times.append(total_ms)

            row: Dict[str, Any] = {
                "model": spec.name,
                "frame_idx": frame_idx,
                "processed_idx": frames_done,
                "timestamp_s": timestamp_s,
                "video_read_ms": read_ms,
                "preprocess_ms": pre_ms,
                "infer_ms": inf_ms,
                "postprocess_ms": post_ms,
                "total_frame_ms": total_ms,
                "detections_or_positive_pixels": detections_or_pixels,
                "pipeline_fps_instant_raspberry_like": 1000.0 / total_ms if total_ms > 0 else "",
                "infer_fps_instant": 1000.0 / inf_ms if inf_ms > 0 else "",
            }
            if torch.cuda.is_available():
                row["torch_cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                row["torch_cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
                row["torch_cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            frame_rows.append(row)

            frames_done += 1
            if frames_done % args.print_every == 0:
                avg_total = _mean(total_times)
                avg_inf = _mean(infer_times)
                print(
                    f"[{spec.name}] processed={frames_done} "
                    f"pipeline_fps={_fps_from_ms(avg_total):.2f} "
                    f"infer_fps={_fps_from_ms(avg_inf):.2f} "
                    f"infer_ms={avg_inf:.2f} total_ms={avg_total:.2f}"
                )
    finally:
        if video_writer is not None:
            video_writer.release()
        tegra.stop()

    elapsed_s = time.perf_counter() - t_script0
    frame_metrics_csv = model_dir / f"{spec.name}_frame_metrics.csv"
    resources_csv = model_dir / f"{spec.name}_resource_samples.csv"
    write_rows_csv(frame_metrics_csv, frame_rows)
    write_rows_csv(resources_csv, resource_rows)
    write_json(model_dir / "system_info_before.json", collect_system_info())
    write_json(model_dir / "system_info_after.json", collect_system_info())

    avg_pre = _mean(preprocess_times)
    avg_inf = _mean(infer_times)
    avg_post = _mean(post_times)
    avg_total = _mean(total_times)

    summary: Dict[str, Any] = {
        "model": spec.name,
        "kind": spec.kind,
        "model_path": str(spec.model_path),
        "factory": spec.factory or "",
        "load_mode": runner.load_mode,
        "csv_path": str(csv_path),
        "frame_metrics_csv": str(frame_metrics_csv),
        "resource_samples_csv": str(resources_csv),
        "video_path": str(video_path_out) if video_path_out is not None else "",
        "frames_processed": frames_done,
        "frames_read_or_last_frame_idx": last_frame_idx,
        "max_seconds": args.max_seconds,
        "frame_step": args.frame_step,
        "input_h": runner.input_shape[0],
        "input_w": runner.input_shape[1],
        "input_c": runner.input_shape[2],
        "device": str(runner.device),
        "precision": args.precision,
        "load_s": load_s,
        "infer_errors": infer_errors,
        # Kolumny analogiczne do Raspberry:
        "avg_preprocess_ms": avg_pre,
        "avg_infer_ms": avg_inf,
        "avg_postprocess_ms": avg_post,
        "avg_total_ms": avg_total,
        "infer_fps_only": _fps_from_ms(avg_inf),
        "pipeline_fps": _fps_from_ms(avg_total),
        "avg_cpu_percent": float(np.mean(cpu_samples)) if cpu_samples else None,
        "avg_ram_mb": float(np.mean(ram_samples)) if ram_samples else None,
        "avg_cpu_temp_c": float(np.mean(temp_samples)) if temp_samples else None,
        # Dodatkowe aliasy/metryki Jetsona:
        "preprocess_mean_ms": avg_pre,
        "infer_mean_ms": avg_inf,
        "postprocess_mean_ms": avg_post,
        "total_frame_mean_ms": avg_total,
        "overall_wall_clock_fps": frames_done / elapsed_s if elapsed_s > 0 else None,
        "elapsed_s": elapsed_s,
        "tegrastats_raw_log": str(tegra.raw_log) if tegra.raw_log.exists() else "",
        "tegrastats_parsed_csv": str(tegra.parsed_csv) if tegra.parsed_csv.exists() else "",
    }

    timings = {
        "video_read": video_read_times,
        "preprocess": preprocess_times,
        "infer": infer_times,
        "postprocess": post_times,
        "total_frame": total_times,
    }
    add_timing_summary(summary, timings)
    summary.update({f"resource_{k}": v for k, v in summarize_numeric_rows(resource_rows).items()})
    summary.update({f"frame_{k}": v for k, v in summarize_numeric_rows(frame_rows).items()})
    write_json(model_dir / f"{spec.name}_summary.json", summary)
    return summary


def maybe_add_model(
    specs: List[ModelSpec],
    name: str,
    kind: str,
    model_path: Optional[str],
    input_str: str,
    factory: Optional[str],
    factory_args: Optional[str],
    threshold: float,
    topk: int,
    min_dist: float,
    nms: int,
    crop_bottom_ratio: float,
) -> None:
    if not model_path:
        return
    specs.append(ModelSpec(
        name=name,
        model_path=Path(model_path).expanduser(),
        kind=kind,
        input_shape=parse_size(input_str),
        factory=factory,
        factory_args_json=factory_args,
        threshold=threshold,
        topk=topk,
        min_distance_px=min_dist,
        nms_kernel=nms,
        crop_bottom_ratio=crop_bottom_ratio,
    ))


def parse_extra_model(text: str, default_crop: float) -> ModelSpec:
    vals: Dict[str, str] = {}
    for part in text.split(","):
        if "=" not in part:
            raise ValueError(f"Niepoprawny --extra_model fragment: {part}")
        k, v = part.split("=", 1)
        vals[k.strip()] = v.strip()
    for req in ("name", "path", "kind", "input"):
        if req not in vals:
            raise ValueError(f"--extra_model wymaga {req}: {text}")
    return ModelSpec(
        name=vals["name"],
        model_path=Path(vals["path"]).expanduser(),
        kind=vals["kind"],
        input_shape=parse_size(vals["input"]),
        factory=vals.get("factory") or None,
        factory_args_json=vals.get("factory_args_json") or None,
        threshold=float(vals.get("threshold", "0.5")),
        topk=int(vals.get("topk", "24")),
        min_distance_px=float(vals.get("min_distance_px", "16")),
        nms_kernel=int(vals.get("nms_kernel", "7")),
        crop_bottom_ratio=float(vals.get("crop_bottom_ratio", str(default_crop))),
    )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Analog benchmarków Raspberry dla pojedynczych modeli, Jetson + PyTorch")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--out_dir", type=Path)
    ap.add_argument("--gui", action="store_true", help="Uruchom proste okno wyboru plików")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--max_seconds", type=float, default=120.0, help="0 lub <=0 oznacza całe wideo")
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--video_mode", default="none", choices=["none", "mjpg", "mp4"])
    ap.add_argument("--seg_alpha", type=float, default=0.45)
    ap.add_argument("--crop_bottom_ratio", type=float, default=0.60)
    ap.add_argument("--resource_sample_every", type=int, default=1)
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--tegrastats_interval_ms", type=int, default=1000)
    ap.add_argument("--disable_tegrastats", action="store_true")
    ap.add_argument("--collect_psutil", action="store_true", default=True)
    ap.add_argument("--stop_on_error", action="store_true")

    ap.add_argument("--hrnet_point_model")
    ap.add_argument("--hrnet_point_factory")
    ap.add_argument("--hrnet_point_factory_args_json")
    ap.add_argument("--hrnet_point_input", default="256x512")
    ap.add_argument("--hr_threshold", type=float, default=0.5)
    ap.add_argument("--hr_topk", type=int, default=24)
    ap.add_argument("--hr_min_distance", type=float, default=16.0)
    ap.add_argument("--hr_nms_kernel", type=int, default=7)

    ap.add_argument("--unet_point_model")
    ap.add_argument("--unet_point_factory")
    ap.add_argument("--unet_point_factory_args_json")
    ap.add_argument("--unet_point_input", default="256x512")
    ap.add_argument("--un_threshold", type=float, default=0.3772490706319703)
    ap.add_argument("--un_topk", type=int, default=24)
    ap.add_argument("--un_min_distance", type=float, default=16.0)
    ap.add_argument("--un_nms_kernel", type=int, default=7)

    ap.add_argument("--linknet_seg_model")
    ap.add_argument("--linknet_seg_factory")
    ap.add_argument("--linknet_seg_factory_args_json")
    ap.add_argument("--linknet_seg_input", default="384x384")

    ap.add_argument("--unet_seg_model")
    ap.add_argument("--unet_seg_factory")
    ap.add_argument("--unet_seg_factory_args_json")
    ap.add_argument("--unet_seg_input", default="384x384")

    ap.add_argument("--extra_model", action="append", default=[], help="Format: name=...,path=...,kind=point|seg,input=HxW[,factory=plik.py:fn,threshold=...]")
    return ap


def infer_defaults_for_gui(model_path: Path, kind: str) -> Dict[str, str]:
    name = model_path.name.lower()
    result = {"input": "384x384", "factory": "", "threshold": "0.5", "crop": "0.60"}
    if kind == "point":
        result["input"] = "256x512"
        result["crop"] = "0.60"
        if "hrnet" in name:
            result["factory"] = "./model_defs_points.py:make_hrnet_point"
            result["threshold"] = "0.5"
    else:
        result["input"] = "384x384"
        result["crop"] = "0.50"
        if "linknet" in name:
            result["factory"] = "./model_defs_ground_map.py:make_ground_map_linknet_mobilenetv2"
        elif "unet" in name:
            result["factory"] = "./model_defs_ground_map.py:make_ground_map_unet_mobilenetv2"
    return result


def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        print(f"[BŁĄD] Brak tkinter: {exc}. Zainstaluj: sudo apt install python3-tk", file=sys.stderr)
        return 10

    root = tk.Tk()
    root.title("Jetson single model benchmark analog Raspberry")
    root.geometry("760x470")

    video_var = tk.StringVar()
    model_var = tk.StringVar()
    out_var = tk.StringVar(value=str(Path.cwd() / "wyniki_jetson_single"))
    kind_var = tk.StringVar(value="point")
    input_var = tk.StringVar(value="256x512")
    factory_var = tk.StringVar(value="./model_defs_points.py:make_hrnet_point")
    max_seconds_var = tk.StringVar(value="30")
    frame_step_var = tk.StringVar(value="2")
    crop_var = tk.StringVar(value="0.60")
    threshold_var = tk.StringVar(value="0.5")
    video_mode_var = tk.StringVar(value="none")
    disable_tegrastats_var = tk.BooleanVar(value=False)

    def choose_video():
        p = filedialog.askopenfilename(title="Wybierz wideo", filetypes=[("Video", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI"), ("All", "*")])
        if p:
            video_var.set(p)

    def choose_model():
        p = filedialog.askopenfilename(title="Wybierz model", filetypes=[("PyTorch", "*.pt *.pth"), ("All", "*")])
        if p:
            model_var.set(p)
            defs = infer_defaults_for_gui(Path(p), kind_var.get())
            input_var.set(defs["input"]); factory_var.set(defs["factory"]); threshold_var.set(defs["threshold"]); crop_var.set(defs["crop"])

    def choose_out():
        p = filedialog.askdirectory(title="Wybierz folder wyników")
        if p:
            out_var.set(p)

    def kind_changed(*_):
        if model_var.get():
            defs = infer_defaults_for_gui(Path(model_var.get()), kind_var.get())
            input_var.set(defs["input"]); factory_var.set(defs["factory"]); threshold_var.set(defs["threshold"]); crop_var.set(defs["crop"])

    kind_var.trace_add("write", kind_changed)

    rows = [
        ("Wideo", video_var, choose_video),
        ("Model", model_var, choose_model),
        ("Folder wyników", out_var, choose_out),
    ]
    for i, (label, var, cmd) in enumerate(rows):
        tk.Label(root, text=label, anchor="w").grid(row=i, column=0, sticky="w", padx=8, pady=6)
        tk.Entry(root, textvariable=var, width=78).grid(row=i, column=1, padx=8, pady=6)
        tk.Button(root, text="Wybierz", command=cmd).grid(row=i, column=2, padx=8, pady=6)

    r = len(rows)
    tk.Label(root, text="Typ modelu").grid(row=r, column=0, sticky="w", padx=8, pady=6)
    ttk.Combobox(root, textvariable=kind_var, values=["point", "seg"], state="readonly", width=15).grid(row=r, column=1, sticky="w", padx=8, pady=6)
    r += 1
    for label, var in [
        ("Input HxW", input_var),
        ("Factory plik.py:funkcja", factory_var),
        ("Max seconds: 0=całe", max_seconds_var),
        ("Frame step", frame_step_var),
        ("Crop bottom ratio", crop_var),
        ("Threshold punktów", threshold_var),
    ]:
        tk.Label(root, text=label).grid(row=r, column=0, sticky="w", padx=8, pady=5)
        tk.Entry(root, textvariable=var, width=42).grid(row=r, column=1, sticky="w", padx=8, pady=5)
        r += 1

    tk.Label(root, text="Video mode").grid(row=r, column=0, sticky="w", padx=8, pady=5)
    ttk.Combobox(root, textvariable=video_mode_var, values=["none", "mjpg", "mp4"], state="readonly", width=15).grid(row=r, column=1, sticky="w", padx=8, pady=5)
    r += 1
    tk.Checkbutton(root, text="Wyłącz tegrastats", variable=disable_tegrastats_var).grid(row=r, column=1, sticky="w", padx=8, pady=5)
    r += 1

    def start():
        if not video_var.get() or not model_var.get() or not out_var.get():
            messagebox.showerror("Brak danych", "Wybierz wideo, model i folder wyników.")
            return
        cmd = [sys.executable, str(Path(__file__).resolve()), "--video", video_var.get(), "--out_dir", out_var.get(), "--max_seconds", max_seconds_var.get(), "--frame_step", frame_step_var.get(), "--crop_bottom_ratio", crop_var.get(), "--precision", "fp16", "--video_mode", video_mode_var.get(), "--resource_sample_every", "30", "--print_every", "20"]
        if disable_tegrastats_var.get():
            cmd.append("--disable_tegrastats")
        factory = factory_var.get().strip()
        if kind_var.get() == "point":
            cmd += ["--hrnet_point_model", model_var.get(), "--hrnet_point_input", input_var.get(), "--hr_threshold", threshold_var.get()]
            if factory:
                cmd += ["--hrnet_point_factory", factory]
        else:
            # Nazwa opcji zależy od architektury, ale obie prowadzą do kind=seg.
            if "unet" in Path(model_var.get()).name.lower() and "linknet" not in Path(model_var.get()).name.lower():
                cmd += ["--unet_seg_model", model_var.get(), "--unet_seg_input", input_var.get()]
                if factory:
                    cmd += ["--unet_seg_factory", factory]
            else:
                cmd += ["--linknet_seg_model", model_var.get(), "--linknet_seg_input", input_var.get()]
                if factory:
                    cmd += ["--linknet_seg_factory", factory]
        root.destroy()
        print("[GUI] Uruchamiam:")
        print(" ".join(cmd))
        raise SystemExit(subprocess.call(cmd))

    tk.Button(root, text="START", command=start, height=2, width=20).grid(row=r, column=1, sticky="e", padx=8, pady=12)
    root.mainloop()
    return 0


def main() -> int:
    args = build_argparser().parse_args()
    if args.gui or (args.video is None and args.out_dir is None and not any([args.hrnet_point_model, args.unet_point_model, args.linknet_seg_model, args.unet_seg_model, args.extra_model])):
        return run_gui()

    if args.video is None or args.out_dir is None:
        print("[BŁĄD] Podaj --video i --out_dir albo uruchom z --gui.", file=sys.stderr)
        return 1
    if not args.video.exists():
        print(f"[BŁĄD] Nie ma pliku wideo: {args.video}", file=sys.stderr)
        return 2
    ensure_dir(args.out_dir)

    specs: List[ModelSpec] = []
    maybe_add_model(specs, "hrnet_point_jetson_torch", "point", args.hrnet_point_model, args.hrnet_point_input, args.hrnet_point_factory, args.hrnet_point_factory_args_json, args.hr_threshold, args.hr_topk, args.hr_min_distance, args.hr_nms_kernel, args.crop_bottom_ratio)
    maybe_add_model(specs, "unet_point_jetson_torch", "point", args.unet_point_model, args.unet_point_input, args.unet_point_factory, args.unet_point_factory_args_json, args.un_threshold, args.un_topk, args.un_min_distance, args.un_nms_kernel, args.crop_bottom_ratio)
    maybe_add_model(specs, "linknet_seg_jetson_torch", "seg", args.linknet_seg_model, args.linknet_seg_input, args.linknet_seg_factory, args.linknet_seg_factory_args_json, 0.0, 0, 0.0, 0, args.crop_bottom_ratio)
    maybe_add_model(specs, "unet_seg_jetson_torch", "seg", args.unet_seg_model, args.unet_seg_input, args.unet_seg_factory, args.unet_seg_factory_args_json, 0.0, 0, 0.0, 0, args.crop_bottom_ratio)
    for txt in args.extra_model:
        specs.append(parse_extra_model(txt, args.crop_bottom_ratio))

    if not specs:
        print("[BŁĄD] Nie podano modelu. Użyj np. --hrnet_point_model albo --gui.", file=sys.stderr)
        return 3
    for s in specs:
        if not s.model_path.exists():
            print(f"[BŁĄD] Nie ma modelu: {s.model_path}", file=sys.stderr)
            return 4
        if s.kind not in ("point", "seg"):
            print(f"[BŁĄD] Niepoprawny kind dla {s.name}: {s.kind}", file=sys.stderr)
            return 5

    write_json(args.out_dir / "system_info_start.json", collect_system_info())
    all_summaries: List[Dict[str, Any]] = []
    for spec in specs:
        all_summaries.append(process_model(spec, args.video, args.out_dir, args))
    write_rows_csv(args.out_dir / "benchmark_summary.csv", all_summaries)
    write_json(args.out_dir / "benchmark_summary.json", all_summaries)
    write_json(args.out_dir / "system_info_end.json", collect_system_info())

    print("\n==== SUMMARY — ANALOGICZNIE DO RASPBERRY ====")
    for r in all_summaries:
        print(
            f"{r['model']:28s} | frames={r['frames_processed']} | "
            f"infer_fps={r['infer_fps_only']:.2f} | pipeline_fps={r['pipeline_fps']:.2f} | "
            f"pre={r['avg_preprocess_ms']:.2f} ms | infer={r['avg_infer_ms']:.2f} ms | "
            f"post={r['avg_postprocess_ms']:.2f} ms | total={r['avg_total_ms']:.2f} ms | "
            f"wall_fps={r['overall_wall_clock_fps']:.2f}"
        )
    print(f"\nSummary CSV: {args.out_dir / 'benchmark_summary.csv'}")
    print(f"Outputs dir : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
