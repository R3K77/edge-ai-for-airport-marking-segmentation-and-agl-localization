
#!/usr/bin/env python3
"""
Prepare a point-detection dataset from LabelMe annotations for a CenterNet-like model.

Input:
  A folder that recursively contains images and matching LabelMe JSON files
  with the same basename, e.g.:
    img_001.jpg
    img_001.json

What counts as a sample:
  - image exists
  - matching JSON exists
  - JSON parses correctly
  - points are read from shapes with label == allowed label(s) and shape_type == "point"
  - JSON with zero valid points is still included as a negative sample

What this script creates:
output_root/
  train/
    images/
    labels/
    index.json
  val/
    images/
    labels/
    index.json
  test/
    images/
    labels/
    index.json
  reports/
    split_report.json

Each label JSON is simplified to:
{
  "image_filename": "abc.jpg",
  "width": 1920,
  "height": 1080,
  "num_points": 3,
  "points": [
    {"x": 123.4, "y": 456.7, "label": "agl_light"}
  ]
}
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    image_path: Path
    json_path: Path
    rel_stem: str
    width: int
    height: int
    points: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare point dataset from LabelMe JSON.")
    parser.add_argument("input_dir", type=Path, help="Folder with images and LabelMe JSON files.")
    parser.add_argument("output_dir", type=Path, help="Folder to create the split dataset in.")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio. Default: 0.7")
    parser.add_argument("--val", type=float, default=0.2, help="Val ratio. Default: 0.2")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio. Default: 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output_dir first if it exists.")
    parser.add_argument(
        "--allowed-labels",
        nargs="+",
        default=["agl_light"],
        help="Only points with these labels will be included. Default: agl_light",
    )
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}")
    for name, value in [("train", train), ("val", val), ("test", test)]:
        if value < 0:
            raise ValueError(f"Ratio {name} cannot be negative.")


def find_matching_json(image_path: Path) -> Path | None:
    candidate = image_path.with_suffix(".json")
    if candidate.exists():
        return candidate
    return None


def load_points_from_labelme(
    json_path: Path,
    allowed_labels: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    points_out: list[dict[str, Any]] = []

    for shape in shapes:
        label = shape.get("label", "")
        shape_type = shape.get("shape_type", "")
        pts = shape.get("points", [])

        if label not in allowed_labels:
            continue

        is_point = shape_type == "point" or (shape_type == "" and len(pts) == 1)
        if not is_point:
            continue

        if not pts or len(pts[0]) < 2:
            continue

        x = float(pts[0][0])
        y = float(pts[0][1])

        entry = {
            "x": x,
            "y": y,
            "label": label,
        }

        # Keep optional metadata if present
        if "flags" in shape and isinstance(shape["flags"], dict):
            entry["flags"] = shape["flags"]
        if "group_id" in shape:
            entry["group_id"] = shape["group_id"]
        if "description" in shape:
            entry["description"] = shape["description"]

        points_out.append(entry)

    return points_out, data


def discover_samples(input_dir: Path, allowed_labels: set[str]) -> tuple[list[Sample], dict[str, Any]]:
    samples: list[Sample] = []
    skipped_no_json: list[str] = []
    skipped_invalid_json: list[dict[str, str]] = []
    skipped_image_errors: list[dict[str, str]] = []
    ignored_labelme_without_image: list[str] = []

    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    seen_jsons: set[Path] = set()

    for image_path in image_paths:
        json_path = find_matching_json(image_path)
        if json_path is None:
            skipped_no_json.append(str(image_path))
            continue

        try:
            points, raw_json = load_points_from_labelme(json_path, allowed_labels)
        except Exception as e:  # noqa: BLE001
            skipped_invalid_json.append(
                {"image_path": str(image_path), "json_path": str(json_path), "error": str(e)}
            )
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:  # noqa: BLE001
            skipped_image_errors.append({"image_path": str(image_path), "error": str(e)})
            continue

        rel_stem = str(image_path.relative_to(input_dir).with_suffix("")).replace("\\", "/")
        samples.append(
            Sample(
                image_path=image_path,
                json_path=json_path,
                rel_stem=rel_stem,
                width=width,
                height=height,
                points=points,
            )
        )
        seen_jsons.add(json_path.resolve())

    # JSONs without a matching image are reported
    for json_path in sorted(input_dir.rglob("*.json")):
        if json_path.resolve() in seen_jsons:
            continue
        has_image = any(json_path.with_suffix(ext).exists() for ext in IMAGE_EXTS)
        if not has_image:
            ignored_labelme_without_image.append(str(json_path))

    report = {
        "skipped_no_json": skipped_no_json,
        "skipped_invalid_json": skipped_invalid_json,
        "skipped_image_errors": skipped_image_errors,
        "ignored_labelme_without_image": ignored_labelme_without_image,
    }
    return samples, report


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:n_train + n_val + n_test],
    }


def copy_sample(sample: Sample, split_dir: Path) -> dict[str, Any]:
    image_ext = sample.image_path.suffix.lower()
    safe_name = sample.rel_stem.replace("/", "__")
    dst_image_name = safe_name + image_ext
    dst_label_name = safe_name + ".json"

    dst_image = split_dir / "images" / dst_image_name
    dst_label = split_dir / "labels" / dst_label_name

    dst_image.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(sample.image_path, dst_image)

    label_data = {
        "image_filename": dst_image_name,
        "source_image": str(sample.image_path),
        "source_json": str(sample.json_path),
        "width": sample.width,
        "height": sample.height,
        "num_points": len(sample.points),
        "points": sample.points,
    }
    with dst_label.open("w", encoding="utf-8") as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)

    return {
        "image": f"images/{dst_image_name}",
        "label": f"labels/{dst_label_name}",
        "width": sample.width,
        "height": sample.height,
        "num_points": len(sample.points),
    }


def write_index(split_dir: Path, items: list[dict[str, Any]]) -> None:
    with (split_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input dir does not exist or is not a folder: {input_dir}", file=sys.stderr)
        return 2

    try:
        validate_ratios(args.train, args.val, args.test)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            print(f"[ERROR] Output dir exists: {output_dir}. Use --overwrite.", file=sys.stderr)
            return 2

    allowed_labels = set(args.allowed_labels)
    samples, discovery_report = discover_samples(input_dir, allowed_labels)

    if len(samples) == 0:
        print("[ERROR] No valid image+json samples found.", file=sys.stderr)
        return 3

    split_data = split_samples(samples, args.train, args.val, args.test, args.seed)

    counts = {}
    for split_name, split_samples_list in split_data.items():
        split_dir = output_dir / split_name
        items = [copy_sample(sample, split_dir) for sample in split_samples_list]
        write_index(split_dir, items)
        counts[split_name] = len(items)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "allowed_labels": sorted(allowed_labels),
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "counts": {
            "total_samples": len(samples),
            "train": counts.get("train", 0),
            "val": counts.get("val", 0),
            "test": counts.get("test", 0),
            "positive_images_total": sum(1 for s in samples if len(s.points) > 0),
            "negative_images_total": sum(1 for s in samples if len(s.points) == 0),
        },
        **discovery_report,
    }

    with (reports_dir / "split_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Total valid samples: {len(samples)}")
    print(f"Train: {counts.get('train', 0)}")
    print(f"Val:   {counts.get('val', 0)}")
    print(f"Test:  {counts.get('test', 0)}")
    print(f"Positive images: {report['counts']['positive_images_total']}")
    print(f"Negative images: {report['counts']['negative_images_total']}")
    print(f"Output: {output_dir.resolve()}")
    print("Note: images without JSON were skipped on purpose.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
