#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zintegrowany pipeline dla NVIDIA Jetson AGX Orin / TensorRT FP16 .engine:
- wejście: plik MP4 z telemetrią GPS osadzoną w metadanych,
- ekstrakcja GPS przez ExifTool i interpolacja GPS do każdej klatki,
- inferencja dwoma modelami TensorRT .engine:
    1) model punktowy, np. HRNet-Lite-Point,
    2) model segmentacyjny, np. LinkNet + MobileNetV2,
- zapis wspólnego CSV z detekcjami i próbkami segmentacji,
- zapis CSV trajektorii,
- zapis mapy PNG,
- zapis metryk czasu per frame,
- zapis rozbudowanych metryk Jetsona: psutil, sysfs thermal/hwmon/devfreq, tegrastats, nvpmodel, jetson_clocks.

Modele muszą być wcześniej zbudowane do TensorRT .engine, np. przez trtexec:
  --hrnet_engine ./modele_trt/hrnet-lite-point_fp16.engine
  --linknet_engine ./modele_trt/linknet_mobilenetv2_fp16.engine

Przykład:
python3 jetson_pipeline_tensorrt_analog_raspberry.py \
  --video ./nagrania_testowe/GH010453.MP4 \
  --out_dir ./wyniki_pipeline_trt/GH010453_hrnet_linknet \
  --hrnet_engine ./modele_trt/hrnet-lite-point_fp16.engine \
  --hrnet_input 256x512 \
  --linknet_engine ./modele_trt/linknet_mobilenetv2_fp16.engine \
  --linknet_input 384x384 \
  --precision fp16
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from jetson_trt_common import (
    PointModelConfig,
    SegModelConfig,
    ProjectionConfig,
    TensorRTRunner,
    TegrastatsLogger,
    add_timing_summary,
    collect_resource_snapshot,
    collect_system_info,
    decode_points_outputs,
    decode_segmentation,
    ensure_dir,
    extract_gps_from_video,
    ffmpeg_frame_reader,
    ffprobe_video,
    get_row_distance_lut,
    latlon_to_local_xy,
    local_offset_to_world,
    make_video_writer,
    parse_size,
    pixel_to_lateral,
    preprocess_point,
    preprocess_seg,
    render_points_panel,
    render_seg_panel,
    summarize_numeric_rows,
    write_json,
    write_rows_csv,
)


def append_rows_stream(path: Path, rows: List[Dict[str, Any]], header_state: Dict[str, List[str]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    key = str(path)
    if key not in header_state:
        fieldnames: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    fieldnames.append(k)
                    seen.add(k)
        header_state[key] = fieldnames
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_state[key], extrasaction="ignore")
        w.writerows(rows)


def save_map_png(detections_csv: Path, traj_csv: Path, out_png: Path, max_points: int = 250000) -> None:
    """Mapa PNG w stylu zgodnym z pipeline Raspberry/Hailo."""
    try:
        import pandas as pd  # type: ignore
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[WARN] Nie zapisano mapy PNG, brak pandas/matplotlib: {exc}")
        return

    if not detections_csv.exists() or detections_csv.stat().st_size == 0:
        return

    det_df = pd.read_csv(detections_csv)
    traj_df = pd.read_csv(traj_csv) if traj_csv.exists() and traj_csv.stat().st_size > 0 else pd.DataFrame()

    if len(det_df) > max_points:
        det_df = det_df.sample(max_points, random_state=42)

    fig, ax = plt.subplots(figsize=(20, 16))

    if not traj_df.empty and {"x_m", "y_m"}.issubset(traj_df.columns):
        ax.plot(
            traj_df["x_m"],
            traj_df["y_m"],
            color="black",
            linewidth=1.0,
            label="Trajektoria GPS",
        )

    if not det_df.empty and {"x_m", "y_m", "cls_name"}.issubset(det_df.columns):
        styles = [
            ("white_mark", "Oznakowanie białe", "white", 0.6),
            ("yellow_mark", "Oznakowanie żółte", "gold", 0.6),
            ("agl_light", "Lampy AGL", "red", 8.0),
        ]
        for cls_name, label, color, size in styles:
            sub = det_df[det_df["cls_name"] == cls_name]
            if not sub.empty:
                ax.scatter(
                    sub["x_m"],
                    sub["y_m"],
                    s=size,
                    c=color,
                    edgecolors="none",
                    label=label,
                )

        known = {"white_mark", "yellow_mark", "agl_light"}
        other = det_df[~det_df["cls_name"].isin(known)]
        if not other.empty:
            ax.scatter(other["x_m"], other["y_m"], s=0.6, c="cyan", edgecolors="none", label="Inne")

    ax.set_facecolor("#2f4f2f")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Mapa oznakowania poziomego i lamp AGL z GPS + analiza wizyjna")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def process_video(args: argparse.Namespace, hrnet_runner: TensorRTRunner, linknet_runner: TensorRTRunner, gps_bundle) -> Dict[str, Any]:
    out_dir: Path = args.out_dir
    width, height, fps_probe, _total_probe = ffprobe_video(args.video)
    fps = gps_bundle.fps if gps_bundle.fps > 0 else fps_probe

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

    detections_csv = out_dir / f"{args.video.stem}_detections_combined_jetson_trt.csv"
    traj_csv = out_dir / f"{args.video.stem}_trajectory_jetson_trt.csv"
    frame_metrics_csv = out_dir / f"{args.video.stem}_frame_metrics_jetson_trt.csv"
    resource_csv = out_dir / f"{args.video.stem}_resource_samples_jetson_trt.csv"
    map_png = out_dir / f"{args.video.stem}_map_jetson_trt.png"

    for p in (detections_csv, traj_csv, frame_metrics_csv, resource_csv):
        if p.exists():
            p.unlink()
    header_state: Dict[str, List[str]] = {}

    # Pochodzenie lokalnego układu współrzędnych: pierwszy poprawny GPS.
    valid_origin = None
    for idx in sorted(gps_bundle.rows.keys()):
        row = gps_bundle.rows[idx]
        if row.lat is not None and row.lon is not None:
            valid_origin = (row.lat, row.lon)
            break
    if valid_origin is None:
        raise RuntimeError("Brak poprawnego punktu GPS do zbudowania układu lokalnego.")
    lat0, lon0 = valid_origin

    proc = None
    if args.collect_psutil:
        try:
            import psutil  # type: ignore
            proc = psutil.Process(os.getpid())
            proc.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            proc = None


    tegra = TegrastatsLogger(out_dir, interval_ms=args.tegrastats_interval_ms, enabled=not args.disable_tegrastats, prefix="tegrastats_pipeline")
    tegra.start()

    video_writer = None
    video_path_out = None
    if args.video_mode != "none":
        # Dwa panele: punktowy i segmentacyjny jeden pod drugim.
        y1_v = int(round(height * (1.0 - max(point_cfg.crop_bottom_ratio, seg_cfg.crop_bottom_ratio))))
        crop_h_v = height - y1_v
        video_writer, video_path_out = make_video_writer(out_dir / f"{args.video.stem}_pipeline_overlay", width, crop_h_v * 2, fps / max(1, args.frame_step), args.video_mode)

    timings: Dict[str, List[float]] = defaultdict(list)
    resource_rows_buffer: List[Dict[str, Any]] = []
    frame_rows_buffer: List[Dict[str, Any]] = []
    traj_rows_buffer: List[Dict[str, Any]] = []
    det_rows_buffer: List[Dict[str, Any]] = []
    row_distance_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

    processed_frames = 0
    candidate_frames = 0
    skipped_no_gps = 0
    skipped_by_max_frames = 0
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
    flush_every = max(1, args.flush_every)
    t0_script = time.perf_counter()

    try:
        reader = ffmpeg_frame_reader(args.video, width, height, frame_step=max(1, args.frame_step))
        for frame_idx, frame_bgr in reader:
            candidate_frames += 1
            if max_frames is not None and processed_frames >= max_frames:
                skipped_by_max_frames += 1
                break

            t_frame0 = time.perf_counter()
            gps = gps_bundle.rows.get(frame_idx)
            if gps is None or gps.lat is None or gps.lon is None or gps.bearing_deg is None:
                skipped_no_gps += 1
                continue

            base_x, base_y = latlon_to_local_xy(gps.lat, gps.lon, lat0, lon0)
            traj_rows_buffer.append({
                "frame": frame_idx,
                "timestamp_s": frame_idx / fps,
                "time_utc": gps.time_utc,
                "lat": gps.lat,
                "lon": gps.lon,
                "alt_m": gps.alt_m,
                "bearing_deg": gps.bearing_deg,
                "speed_kmh": gps.speed_kmh,
                "x_m": base_x,
                "y_m": base_y,
            })

            # ---------------- model punktowy ----------------
            y1_p = int(round(height * (1.0 - point_cfg.crop_bottom_ratio)))
            crop_p = frame_bgr[y1_p:height, 0:width]
            p_in_h, p_in_w, _ = hrnet_runner.input_shape

            t_pre = time.perf_counter()
            inp_point = preprocess_point(crop_p, hrnet_runner.input_shape)
            hr_pre_ms = (time.perf_counter() - t_pre) * 1000.0
            timings["hrnet_pre"].append(hr_pre_ms)

            t_inf = time.perf_counter()
            out_point = hrnet_runner.infer(inp_point)
            hr_inf_ms = (time.perf_counter() - t_inf) * 1000.0
            timings["hrnet_infer"].append(hr_inf_ms)

            t_post = time.perf_counter()
            point_preds = decode_points_outputs(out_point, point_cfg.threshold, point_cfg.topk, point_cfg.nms_kernel, point_cfg.min_distance_px)
            hr_post_ms = (time.perf_counter() - t_post) * 1000.0
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
                x_full = x_crop
                y_full = float(y1_p + y_crop)
                y_crop_int = int(np.clip(round(y_crop), 0, crop_h_p - 1))
                forward_m = float(row_dist_lut_p[y_crop_int])
                lateral_m = float(pixel_to_lateral(x_crop, crop_w_p, forward_m, proj_cfg))
                world_x, world_y = local_offset_to_world(base_x, base_y, gps.bearing_deg, lateral_m, forward_m)
                det_rows_buffer.append({
                    "frame_idx": frame_idx,
                    "timestamp_s": frame_idx / fps,
                    "time_utc": gps.time_utc,
                    "lat": gps.lat,
                    "lon": gps.lon,
                    "bearing_deg": gps.bearing_deg,
                    "speed_kmh": gps.speed_kmh,
                    "det_type": "point",
                    "model": "hrnet_point_jetson_trt",
                    "cls_id": 10,
                    "cls_name": "agl_light",
                    "score": float(pred["score"]),
                    "x_full_px": x_full,
                    "y_full_px": y_full,
                    "x_crop_px": x_crop,
                    "y_crop_px": y_crop,
                    "forward_m": forward_m,
                    "lateral_m": lateral_m,
                    "x_m": world_x,
                    "y_m": world_y,
                })

            # ---------------- model segmentacyjny ----------------
            y1_s = int(round(height * (1.0 - seg_cfg.crop_bottom_ratio)))
            crop_s = frame_bgr[y1_s:height, 0:width]

            t_pre = time.perf_counter()
            inp_seg = preprocess_seg(crop_s, linknet_runner.input_shape)
            seg_pre_ms = (time.perf_counter() - t_pre) * 1000.0
            timings["linknet_pre"].append(seg_pre_ms)

            t_inf = time.perf_counter()
            out_seg = linknet_runner.infer(inp_seg)
            seg_inf_ms = (time.perf_counter() - t_inf) * 1000.0
            timings["linknet_infer"].append(seg_inf_ms)

            t_post = time.perf_counter()
            seg_mask_small = decode_segmentation(out_seg)
            seg_mask = cv2.resize(seg_mask_small, (crop_s.shape[1], crop_s.shape[0]), interpolation=cv2.INTER_NEAREST)
            seg_post_ms = (time.perf_counter() - t_post) * 1000.0
            timings["linknet_post"].append(seg_post_ms)

            # ---------------- wektoryzowane mapowanie segmentacji ----------------
            # Ten blok jest krytyczny wydajnościowo. W pierwszej wersji każdy piksel-próbka
            # segmentacji był przeliczany w pętli Pythona przez pixel_to_lateral() i
            # local_offset_to_world(). Przy setkach/tysiącach punktów na klatkę to dominowało
            # nad inferencją TensorRT. Tutaj geometria jest liczona wektorowo w NumPy.
            t_seg_map = time.perf_counter()
            segmentation_sample_count = 0
            white_mark_sample_count = 0
            yellow_mark_sample_count = 0
            ys_small, xs_small = np.where(seg_mask[::seg_cfg.sample_stride_y, ::seg_cfg.sample_stride_x] > 0)
            if xs_small.size > 0:
                xs = (xs_small.astype(np.int32) * seg_cfg.sample_stride_x).astype(np.int32)
                ys = (ys_small.astype(np.int32) * seg_cfg.sample_stride_y).astype(np.int32)
                cls = seg_mask[ys, xs].astype(np.uint8)
                keep = (cls == 1) | (cls == 2)
                if np.any(keep):
                    xs = xs[keep]
                    ys = ys[keep]
                    cls = cls[keep]

                    segmentation_sample_count = int(cls.size)
                    white_mark_sample_count = int(np.count_nonzero(cls == 1))
                    yellow_mark_sample_count = int(np.count_nonzero(cls == 2))

                    row_dist_lut_s = get_row_distance_lut(crop_s.shape[0], proj_cfg, row_distance_cache)
                    ys_clip = np.clip(ys, 0, crop_s.shape[0] - 1)
                    forward = row_dist_lut_s[ys_clip].astype(np.float32, copy=False)

                    # odpowiednik pixel_to_lateral(), ale wektorowo
                    if crop_s.shape[1] <= 1:
                        lateral = np.zeros_like(forward, dtype=np.float32)
                    else:
                        x_norm = (xs.astype(np.float32) / float(crop_s.shape[1] - 1)) * 2.0 - 1.0
                        if proj_cfg.d_max <= 1e-6:
                            width_m = np.full_like(forward, proj_cfg.width_near_m, dtype=np.float32)
                        else:
                            tt = np.clip(forward / float(proj_cfg.d_max), 0.0, 1.0)
                            width_m = proj_cfg.width_near_m + tt * (proj_cfg.width_far_m - proj_cfg.width_near_m)
                        lateral = x_norm * (width_m / 2.0)

                    # odpowiednik local_offset_to_world(), ale sin/cos tylko raz na klatkę
                    th = math.radians(float(gps.bearing_deg))
                    fwd_e = math.sin(th)
                    fwd_n = math.cos(th)
                    right_e = math.sin(th + math.pi / 2.0)
                    right_n = math.cos(th + math.pi / 2.0)
                    world_x = base_x + forward * fwd_e + lateral * right_e
                    world_y = base_y + forward * fwd_n + lateral * right_n

                    timestamp_s = frame_idx / fps
                    common = {
                        "frame_idx": frame_idx,
                        "timestamp_s": timestamp_s,
                        "time_utc": gps.time_utc,
                        "lat": gps.lat,
                        "lon": gps.lon,
                        "bearing_deg": gps.bearing_deg,
                        "speed_kmh": gps.speed_kmh,
                        "det_type": "segmentation",
                        "model": "linknet_seg_jetson_trt",
                        "score": "",
                    }
                    det_rows_buffer.extend({
                        **common,
                        "cls_id": int(ci),
                        "cls_name": "white_mark" if int(ci) == 1 else "yellow_mark",
                        "x_full_px": float(xi),
                        "y_full_px": float(y1_s + yi),
                        "x_crop_px": float(xi),
                        "y_crop_px": float(yi),
                        "forward_m": float(fw),
                        "lateral_m": float(la),
                        "x_m": float(wx),
                        "y_m": float(wy),
                    } for xi, yi, ci, fw, la, wx, wy in zip(xs, ys, cls, forward, lateral, world_x, world_y))
            seg_map_ms = (time.perf_counter() - t_seg_map) * 1000.0
            timings["segmentation_map_append"].append(seg_map_ms)

            if video_writer is not None:
                # Render wspólny z wyrównaniem do jednego cropu od y1_v.
                y1_v = int(round(height * (1.0 - max(point_cfg.crop_bottom_ratio, seg_cfg.crop_bottom_ratio))))
                crop_box_v = (0, y1_v, width, height)
                p_panel = render_points_panel(frame_bgr, (0, y1_p, width, height), point_preds, "hrnet_point", p_in_w, p_in_h)
                s_panel = render_seg_panel(frame_bgr, (0, y1_s, width, height), seg_mask_small, "linknet_seg", alpha=args.seg_alpha)
                target_h = height - y1_v
                p_panel = cv2.resize(p_panel, (width, target_h), interpolation=cv2.INTER_LINEAR)
                s_panel = cv2.resize(s_panel, (width, target_h), interpolation=cv2.INTER_LINEAR)
                video_writer.write(np.vstack([p_panel, s_panel]))

            frame_total_ms = (time.perf_counter() - t_frame0) * 1000.0
            timings["frame_total"].append(frame_total_ms)
            measured_ms = hr_pre_ms + hr_inf_ms + hr_post_ms + seg_pre_ms + seg_inf_ms + seg_post_ms + seg_map_ms
            other_ms = frame_total_ms - measured_ms
            timings["frame_other"].append(other_ms)
            frame_row: Dict[str, Any] = {
                "frame_idx": frame_idx,
                "processed_idx": processed_frames,
                "timestamp_s": frame_idx / fps,
                "time_utc": gps.time_utc,
                "hrnet_pre_ms": hr_pre_ms,
                "hrnet_infer_ms": hr_inf_ms,
                "hrnet_post_ms": hr_post_ms,
                "linknet_pre_ms": seg_pre_ms,
                "linknet_infer_ms": seg_inf_ms,
                "linknet_post_ms": seg_post_ms,
                "segmentation_map_append_ms": seg_map_ms,
                "frame_other_ms": other_ms,
                "frame_total_ms": frame_total_ms,
                "point_detections": point_detection_count,
                "segmentation_samples": segmentation_sample_count,
                "white_mark_samples": white_mark_sample_count,
                "yellow_mark_samples": yellow_mark_sample_count,
                "speed_kmh": gps.speed_kmh,
                "bearing_deg": gps.bearing_deg,
                "pipeline_fps_instant": 1000.0 / frame_total_ms if frame_total_ms > 0 else "",
            }
            frame_rows_buffer.append(frame_row)

            if args.resource_sample_every > 0 and processed_frames % args.resource_sample_every == 0:
                resource_rows_buffer.append(collect_resource_snapshot(frame_idx, proc, tag="pipeline"))

            processed_frames += 1
            if len(det_rows_buffer) >= flush_every:
                append_rows_stream(detections_csv, det_rows_buffer, header_state); det_rows_buffer.clear()
            if len(traj_rows_buffer) >= flush_every:
                append_rows_stream(traj_csv, traj_rows_buffer, header_state); traj_rows_buffer.clear()
            if len(frame_rows_buffer) >= flush_every:
                append_rows_stream(frame_metrics_csv, frame_rows_buffer, header_state); frame_rows_buffer.clear()
            if len(resource_rows_buffer) >= max(1, min(flush_every, 100)):
                append_rows_stream(resource_csv, resource_rows_buffer, header_state); resource_rows_buffer.clear()

            if processed_frames % args.print_every == 0:
                elapsed = time.perf_counter() - t0_script
                fps_proc = processed_frames / max(elapsed, 1e-9)
                hr_mean = float(np.mean(timings["hrnet_infer"])) if timings["hrnet_infer"] else float("nan")
                ln_mean = float(np.mean(timings["linknet_infer"])) if timings["linknet_infer"] else float("nan")
                print(f"[INFO] frames={processed_frames} fps={fps_proc:.2f} hrnet_inf={hr_mean:.2f}ms linknet_inf={ln_mean:.2f}ms det_rows~{len(det_rows_buffer)}")
    finally:
        if video_writer is not None:
            video_writer.release()
        tegra.stop()
        append_rows_stream(detections_csv, det_rows_buffer, header_state)
        append_rows_stream(traj_csv, traj_rows_buffer, header_state)
        append_rows_stream(frame_metrics_csv, frame_rows_buffer, header_state)
        append_rows_stream(resource_csv, resource_rows_buffer, header_state)

    elapsed_s = time.perf_counter() - t0_script
    # Do podsumowania wczytujemy metryki liczbowo z CSV, aby uniknąć trzymania wszystkiego w RAM.
    frame_rows_for_summary: List[Dict[str, Any]] = []
    resource_rows_for_summary: List[Dict[str, Any]] = []
    try:
        import pandas as pd  # type: ignore
        if frame_metrics_csv.exists() and frame_metrics_csv.stat().st_size > 0:
            frame_rows_for_summary = pd.read_csv(frame_metrics_csv).to_dict("records")
        if resource_csv.exists() and resource_csv.stat().st_size > 0:
            resource_rows_for_summary = pd.read_csv(resource_csv).to_dict("records")
    except Exception:
        pass

    metrics: Dict[str, Any] = {
        "video": str(args.video),
        "out_dir": str(out_dir),
        "processed_frames": processed_frames,
        "candidate_frames": candidate_frames,
        "skipped_no_gps": skipped_no_gps,
        "skipped_by_max_frames": skipped_by_max_frames,
        "elapsed_s": elapsed_s,
        "overall_pipeline_fps": processed_frames / elapsed_s if elapsed_s > 0 else None,
        "hrnet_engine": str(args.hrnet_engine),
        "hrnet_load_mode": hrnet_runner.load_mode,
        "hrnet_input_shape": str(hrnet_runner.input_shape),
        "linknet_engine": str(args.linknet_engine),
        "linknet_load_mode": linknet_runner.load_mode,
        "linknet_input_shape": str(linknet_runner.input_shape),
        "device": "cuda",
        "precision": args.precision,
        "detections_csv": str(detections_csv),
        "trajectory_csv": str(traj_csv),
        "frame_metrics_csv": str(frame_metrics_csv),
        "resource_samples_csv": str(resource_csv),
        "map_png": str(map_png),
        "tegrastats_raw_log": str(tegra.raw_log) if tegra.raw_log.exists() else "",
        "tegrastats_parsed_csv": str(tegra.parsed_csv) if tegra.parsed_csv.exists() else "",
        "overlay_video": str(video_path_out) if video_path_out is not None else "",
        "gps_raw_csv": str(gps_bundle.raw_points_csv),
        "gps_per_frame_csv": str(gps_bundle.per_frame_csv),
    }
    add_timing_summary(metrics, timings)

    # Aliasy analogiczne do skryptów Raspberry: FPS liczony z czasu przetwarzania klatki,
    # bez dekodowania/odczytu wideo. overall_pipeline_fps zostaje jako wall-clock.
    frame_total_mean = metrics.get("frame_total_mean_ms")
    hrnet_infer_mean = metrics.get("hrnet_infer_mean_ms")
    linknet_infer_mean = metrics.get("linknet_infer_mean_ms")
    try:
        metrics["pipeline_fps"] = 1000.0 / float(frame_total_mean) if frame_total_mean else None
    except Exception:
        metrics["pipeline_fps"] = None
    try:
        metrics["hrnet_infer_fps_only"] = 1000.0 / float(hrnet_infer_mean) if hrnet_infer_mean else None
    except Exception:
        metrics["hrnet_infer_fps_only"] = None
    try:
        metrics["linknet_infer_fps_only"] = 1000.0 / float(linknet_infer_mean) if linknet_infer_mean else None
    except Exception:
        metrics["linknet_infer_fps_only"] = None

    metrics.update({f"frame_{k}": v for k, v in summarize_numeric_rows(frame_rows_for_summary).items()})
    metrics.update({f"resource_{k}": v for k, v in summarize_numeric_rows(resource_rows_for_summary).items()})

    write_json(out_dir / f"{args.video.stem}_pipeline_metrics_jetson_trt.json", metrics)
    write_rows_csv(out_dir / f"{args.video.stem}_pipeline_metrics_jetson_trt.csv", [metrics])
    save_map_png(detections_csv, traj_csv, map_png, max_points=args.max_map_points)
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pełny pipeline GPS + HRNet + LinkNet dla NVIDIA Jetson AGX Orin, TensorRT .engine")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)

    ap.add_argument("--hrnet_engine", required=True, type=Path, help="HRNet-Lite-Point TensorRT .engine")
    ap.add_argument("--hrnet_input", default="256x512", help="HWC input modelu punktowego, np. 256x512")
    ap.add_argument("--linknet_engine", required=True, type=Path, help="LinkNet MobileNetV2 TensorRT .engine")
    ap.add_argument("--linknet_input", default="384x384", help="HWC input modelu segmentacyjnego, np. 384x384")

    # Argumenty zachowane dla kompatybilności z komendami PyTorch; TensorRT zawsze używa CUDA.
    ap.add_argument("--device", default="cuda", choices=["cuda"])
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16"], help="Metadane wyniku; precyzja wynika z engine'u")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=0, help="0 oznacza całe wideo")
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument("--flush_every", type=int, default=1000)
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--trt_verbose", action="store_true")

    ap.add_argument("--point_threshold", type=float, default=0.5)
    ap.add_argument("--point_topk", type=int, default=24)
    ap.add_argument("--point_min_distance_px", type=float, default=16.0)
    ap.add_argument("--point_nms_kernel", type=int, default=7)
    ap.add_argument("--point_crop_bottom_ratio", type=float, default=0.60)

    ap.add_argument("--seg_crop_bottom_ratio", type=float, default=0.60)
    ap.add_argument("--seg_stride_x", type=int, default=14)
    ap.add_argument("--seg_stride_y", type=int, default=12)
    ap.add_argument("--seg_alpha", type=float, default=0.45)

    ap.add_argument("--d_min", type=float, default=0.5)
    ap.add_argument("--d_max", type=float, default=5.5)
    ap.add_argument("--width_near_m", type=float, default=1.5)
    ap.add_argument("--width_far_m", type=float, default=9.0)
    ap.add_argument("--distance_mode", choices=["linear", "reciprocal"], default="reciprocal")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--max_gap_s", type=float, default=10.0)

    ap.add_argument("--resource_sample_every", type=int, default=30)
    ap.add_argument("--tegrastats_interval_ms", type=int, default=1000)
    ap.add_argument("--disable_tegrastats", action="store_true")
    ap.add_argument("--collect_psutil", action="store_true", default=True)
    ap.add_argument("--video_mode", default="none", choices=["none", "mjpg", "mp4"])
    ap.add_argument("--max_map_points", type=int, default=250000)
    return ap

def main() -> int:
    args = build_argparser().parse_args()
    ensure_dir(args.out_dir)
    if not args.video.exists():
        print(f"[BŁĄD] Nie ma pliku wideo: {args.video}", file=sys.stderr)
        return 1
    if not args.hrnet_engine.exists():
        print(f"[BŁĄD] Nie ma engine punktowego: {args.hrnet_engine}", file=sys.stderr)
        return 2
    if not args.linknet_engine.exists():
        print(f"[BŁĄD] Nie ma engine segmentacyjnego: {args.linknet_engine}", file=sys.stderr)
        return 3

    write_json(args.out_dir / "system_info_start.json", collect_system_info())

    print("[INFO] Ekstrakcja GPS z wideo...")
    t_gps = time.perf_counter()
    gps_bundle = extract_gps_from_video(args.video, args.out_dir, max_gap_s=args.max_gap_s)
    gps_extract_s = time.perf_counter() - t_gps
    print(f"[INFO] GPS per frame: {gps_bundle.per_frame_csv} ({gps_extract_s:.2f}s)")

    print("[INFO] Ładowanie engine punktowego TensorRT...")
    t_load = time.perf_counter()
    hrnet_runner = TensorRTRunner(args.hrnet_engine, parse_size(args.hrnet_input), kind="point", verbose=args.trt_verbose)
    hrnet_load_s = time.perf_counter() - t_load
    print(f"[INFO] HRNet/TRT load_mode={hrnet_runner.load_mode} input={hrnet_runner.input_shape} load_s={hrnet_load_s:.3f}")

    print("[INFO] Ładowanie engine segmentacyjnego TensorRT...")
    t_load = time.perf_counter()
    linknet_runner = TensorRTRunner(args.linknet_engine, parse_size(args.linknet_input), kind="seg", verbose=args.trt_verbose)
    linknet_load_s = time.perf_counter() - t_load
    print(f"[INFO] LinkNet/TRT load_mode={linknet_runner.load_mode} input={linknet_runner.input_shape} load_s={linknet_load_s:.3f}")

    if args.warmup > 0:
        print(f"[INFO] Warmup: {args.warmup} inferencji na każdy model")
        hrnet_runner.warmup(args.warmup)
        linknet_runner.warmup(args.warmup)

    metrics = process_video(args, hrnet_runner, linknet_runner, gps_bundle)
    metrics["gps_extract_s"] = gps_extract_s
    metrics["hrnet_load_s"] = hrnet_load_s
    metrics["linknet_load_s"] = linknet_load_s
    write_json(args.out_dir / f"{args.video.stem}_pipeline_metrics_jetson_trt.json", metrics)
    write_json(args.out_dir / "system_info_end.json", collect_system_info())

    print("\n==== SUMMARY — ANALOGICZNIE DO RASPBERRY ====")
    print(f"frames_processed       : {metrics.get('processed_frames')}")
    print(f"pipeline_fps           : {metrics.get('pipeline_fps')}  # 1000/frame_total_mean_ms, bez dekodowania wideo")
    print(f"overall_wall_clock_fps : {metrics.get('overall_pipeline_fps')}  # pełny czas skryptu")
    print(f"hrnet_infer_mean_ms    : {metrics.get('hrnet_infer_mean_ms'):.3f}")
    print(f"hrnet_infer_fps_only   : {metrics.get('hrnet_infer_fps_only')}")
    print(f"linknet_infer_mean_ms  : {metrics.get('linknet_infer_mean_ms'):.3f}")
    print(f"linknet_infer_fps_only : {metrics.get('linknet_infer_fps_only')}")
    print(f"detections_csv         : {metrics.get('detections_csv')}")
    print(f"trajectory_csv         : {metrics.get('trajectory_csv')}")
    print(f"frame_metrics_csv      : {metrics.get('frame_metrics_csv')}")
    print(f"resource_samples_csv   : {metrics.get('resource_samples_csv')}")
    print(f"tegrastats_raw_log     : {metrics.get('tegrastats_raw_log')}")
    print(f"map_png                : {metrics.get('map_png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
