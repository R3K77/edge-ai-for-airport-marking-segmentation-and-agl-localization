#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from jetson_torch_common import parse_size
from jetson_trt_common import TensorRTRunner


def main() -> int:
    ap = argparse.ArgumentParser(description="Szybki test ładowania i inferencji TensorRT .engine")
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--input", required=True, help="Rozmiar HxW, np. 256x512 albo 384x384")
    ap.add_argument("--kind", default="model")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    runner = TensorRTRunner(args.engine, parse_size(args.input), kind=args.kind, verbose=args.verbose)
    print("engine:", args.engine)
    print("input_name:", runner.input_name)
    print("input_shape_hwc:", runner.input_shape)
    print("input_binding_shape:", runner.input_binding_shape)
    print("input_layout:", runner.input_layout)
    print("outputs:", runner.output_names)

    x = np.zeros(runner.input_shape, dtype=np.float32)
    out = runner.infer(x)
    for k, v in out.items():
        print(k, v.shape, v.dtype, "min", float(np.nanmin(v)), "max", float(np.nanmax(v)))
    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
