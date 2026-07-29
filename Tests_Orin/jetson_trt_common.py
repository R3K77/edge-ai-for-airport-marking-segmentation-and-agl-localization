#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wspólne funkcje dla pełnego pipeline Jetson + TensorRT.

Ten plik używa istniejącej logiki z jetson_torch_common.py dla:
- GPS / ExifTool,
- preprocessu,
- dekodowania wyjść HRNet/LinkNet,
- geometrii i mapowania,
- CSV/metryk/tegrastats.

Dodaje TensorRTRunner, który ładuje gotowe pliki .engine i odpala inferencję
przez TensorRT Python API + CUDA Driver API przez ctypes, bez zależności od PyCUDA.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# TensorRT 8.5 na JetPack 5.1.x używa starego aliasu np.bool w trt.nptype().
# NumPy 1.24 usunął ten alias, więc dokładamy go lokalnie przed pierwszym
# wywołaniem TensorRT. Bez tego test engine kończy się AttributeError: np.bool.
if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]
if "int" not in np.__dict__:
    np.int = int  # type: ignore[attr-defined]
if "float" not in np.__dict__:
    np.float = float  # type: ignore[attr-defined]
if "complex" not in np.__dict__:
    np.complex = complex  # type: ignore[attr-defined]
if "object" not in np.__dict__:
    np.object = object  # type: ignore[attr-defined]
if "str" not in np.__dict__:
    np.str = str  # type: ignore[attr-defined]

# Re-używamy wszystkich stabilnych funkcji z dotychczasowego pipeline'u.
from jetson_torch_common import (  # noqa: F401
    PointModelConfig,
    SegModelConfig,
    ProjectionConfig,
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


class CUDADriverError(RuntimeError):
    pass


class CUDADriver:
    """Minimalny wrapper CUDA Driver API przez ctypes.

    Dzięki temu pipeline nie wymaga instalacji pycuda. Używane funkcje:
    cuInit, cuDevicePrimaryCtxRetain, cuCtxSetCurrent, cuMemAlloc,
    cuMemcpyHtoDAsync, cuMemcpyDtoHAsync, cuStreamCreate/Synchronize/Destroy.
    """

    def __init__(self) -> None:
        try:
            self.lib = ctypes.CDLL("libcuda.so")
        except OSError:
            self.lib = ctypes.CDLL("libcuda.so.1")
        self._bind()
        self.check(self.lib.cuInit(0), "cuInit")
        self._ensure_context()

    def _get(self, name: str, fallback: Optional[str] = None):
        if hasattr(self.lib, name):
            return getattr(self.lib, name)
        if fallback and hasattr(self.lib, fallback):
            return getattr(self.lib, fallback)
        raise AttributeError(name)

    def _bind(self) -> None:
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int

        self.lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cuCtxGetCurrent.restype = ctypes.c_int
        self.lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.lib.cuCtxSetCurrent.restype = ctypes.c_int

        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGet.restype = ctypes.c_int
        self.lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int

        self.cuMemAlloc = self._get("cuMemAlloc_v2", "cuMemAlloc")
        self.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.cuMemAlloc.restype = ctypes.c_int
        self.cuMemFree = self._get("cuMemFree_v2", "cuMemFree")
        self.cuMemFree.argtypes = [ctypes.c_uint64]
        self.cuMemFree.restype = ctypes.c_int

        self.cuMemcpyHtoDAsync = self._get("cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync")
        self.cuMemcpyHtoDAsync.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        self.cuMemcpyHtoDAsync.restype = ctypes.c_int
        self.cuMemcpyDtoHAsync = self._get("cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync")
        self.cuMemcpyDtoHAsync.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
        self.cuMemcpyDtoHAsync.restype = ctypes.c_int

        self.lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.lib.cuStreamCreate.restype = ctypes.c_int
        self.lib.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cuStreamSynchronize.restype = ctypes.c_int
        self.cuStreamDestroy = self._get("cuStreamDestroy_v2", "cuStreamDestroy")
        self.cuStreamDestroy.argtypes = [ctypes.c_void_p]
        self.cuStreamDestroy.restype = ctypes.c_int

    @staticmethod
    def check(code: int, where: str) -> None:
        if int(code) != 0:
            raise CUDADriverError(f"CUDA Driver API error {code} at {where}")

    def _ensure_context(self) -> None:
        ctx = ctypes.c_void_p()
        self.check(self.lib.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if ctx.value:
            self.ctx = ctx
            return
        dev = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        new_ctx = ctypes.c_void_p()
        self.check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(new_ctx), dev.value), "cuDevicePrimaryCtxRetain")
        self.check(self.lib.cuCtxSetCurrent(new_ctx), "cuCtxSetCurrent")
        self.ctx = new_ctx

    def stream_create(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.check(self.lib.cuStreamCreate(ctypes.byref(stream), 0), "cuStreamCreate")
        return stream

    def stream_synchronize(self, stream: ctypes.c_void_p) -> None:
        self.check(self.lib.cuStreamSynchronize(stream), "cuStreamSynchronize")

    def stream_destroy(self, stream: ctypes.c_void_p) -> None:
        if stream and stream.value:
            self.check(self.cuStreamDestroy(stream), "cuStreamDestroy")

    def mem_alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_uint64()
        self.check(self.cuMemAlloc(ctypes.byref(ptr), int(nbytes)), "cuMemAlloc")
        return int(ptr.value)

    def mem_free(self, ptr: int) -> None:
        if ptr:
            self.check(self.cuMemFree(ctypes.c_uint64(int(ptr))), "cuMemFree")

    def memcpy_htod_async(self, dst_device: int, src_host_ptr: int, nbytes: int, stream: ctypes.c_void_p) -> None:
        self.check(
            self.cuMemcpyHtoDAsync(
                ctypes.c_uint64(int(dst_device)),
                ctypes.c_void_p(int(src_host_ptr)),
                ctypes.c_size_t(int(nbytes)),
                stream,
            ),
            "cuMemcpyHtoDAsync",
        )

    def memcpy_dtoh_async(self, dst_host_ptr: int, src_device: int, nbytes: int, stream: ctypes.c_void_p) -> None:
        self.check(
            self.cuMemcpyDtoHAsync(
                ctypes.c_void_p(int(dst_host_ptr)),
                ctypes.c_uint64(int(src_device)),
                ctypes.c_size_t(int(nbytes)),
                stream,
            ),
            "cuMemcpyDtoHAsync",
        )


class TensorRTRunner:
    """Runner TensorRT dla jednego engine'u .engine.

    Wersja naprawiona: buforami CUDA i strumieniem zarządza PyTorch, a nie
    ręczny wrapper CUDA Driver API przez ctypes. Na Jetsonie eliminuje to błąd
    `CUDA Driver API error 709 at cuStreamCreate`, wynikający z konfliktu/
    niepoprawnego stanu kontekstu CUDA przy ręcznym tworzeniu strumienia.

    Wejście do infer() jest w formacie HWC float32, identycznie jak dla
    TorchRunner:
    - point: wynik preprocess_point(crop, input_shape), HWC normalized ImageNet,
    - seg: wynik preprocess_seg(crop, input_shape), HWC RGB/255.

    Runner sam transponuje HWC -> NCHW, jeśli engine ma wejście NCHW.
    """

    def __init__(
        self,
        engine_path: Path,
        input_shape: Optional[Tuple[int, int, int]] = None,
        kind: str = "model",
        verbose: bool = False,
    ) -> None:
        import tensorrt as trt  # type: ignore
        import torch  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() == False. TensorRT pipeline wymaga CUDA.")

        # Wymuszamy inicjalizację kontekstu CUDA przez PyTorch. Ten sam kontekst
        # wykorzystuje TensorRT przy execute_async_v2.
        torch.cuda.init()
        torch.cuda.set_device(0)

        self.trt = trt
        self.torch = torch
        self.engine_path = Path(engine_path)
        self.kind = kind
        self.device = "cuda"
        self.precision = "engine"
        self.load_mode = "tensorrt_engine_torch_cuda_buffers"
        self.verbose = verbose

        logger_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        self.logger = trt.Logger(logger_level)
        runtime = trt.Runtime(self.logger)
        engine_bytes = self.engine_path.read_bytes()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Nie udało się zdeserializować TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Nie udało się utworzyć execution context: {self.engine_path}")

        self.num_bindings = int(self.engine.num_bindings)
        self.input_indices = [i for i in range(self.num_bindings) if self.engine.binding_is_input(i)]
        self.output_indices = [i for i in range(self.num_bindings) if not self.engine.binding_is_input(i)]
        if len(self.input_indices) != 1:
            raise RuntimeError(f"Pipeline obsługuje dokładnie 1 input, engine ma: {len(self.input_indices)}")
        self.input_index = self.input_indices[0]
        self.input_name = self.engine.get_binding_name(self.input_index)
        self.output_names = [self.engine.get_binding_name(i) for i in self.output_indices]

        engine_input_shape = tuple(int(x) for x in self.engine.get_binding_shape(self.input_index))
        if input_shape is None:
            if any(d <= 0 for d in engine_input_shape):
                raise RuntimeError(
                    f"Engine ma dynamiczne wejście {engine_input_shape}. Podaj --*_input, np. 256x512 albo 384x384."
                )
            self.input_shape = self._hwc_from_binding_shape(engine_input_shape)
        else:
            self.input_shape = tuple(int(x) for x in input_shape)  # HWC

        self.input_binding_shape = self._binding_shape_from_hwc(self.input_shape, engine_input_shape)
        if any(d <= 0 for d in engine_input_shape):
            ok = self.context.set_binding_shape(self.input_index, self.input_binding_shape)
            if not ok:
                raise RuntimeError(f"Nie udało się ustawić binding shape {self.input_binding_shape} dla {self.input_name}")

        self.input_layout = self._detect_layout(self.input_binding_shape)
        self.binding_shapes: Dict[int, Tuple[int, ...]] = {}
        self.binding_np_dtypes: Dict[int, Any] = {}
        self.binding_torch_dtypes: Dict[int, Any] = {}
        self.device_tensors: Dict[int, Any] = {}
        self.bindings: List[int] = [0] * self.num_bindings
        self._allocate_torch_buffers()

    @staticmethod
    def _detect_layout(shape: Sequence[int]) -> str:
        if len(shape) == 4:
            if shape[1] in (1, 3):
                return "NCHW"
            if shape[-1] in (1, 3):
                return "NHWC"
        if len(shape) == 3:
            if shape[0] in (1, 3):
                return "CHW"
            if shape[-1] in (1, 3):
                return "HWC"
        raise RuntimeError(f"Nie da się rozpoznać layoutu wejścia TensorRT: {tuple(shape)}")

    @staticmethod
    def _hwc_from_binding_shape(shape: Sequence[int]) -> Tuple[int, int, int]:
        shape = tuple(int(x) for x in shape)
        if len(shape) == 4:
            _, a, b, c = shape
            if a in (1, 3):
                return (b, c, a)
            if c in (1, 3):
                return (a, b, c)
        if len(shape) == 3:
            a, b, c = shape
            if a in (1, 3):
                return (b, c, a)
            if c in (1, 3):
                return (a, b, c)
        raise RuntimeError(f"Nie da się określić HWC z binding shape: {shape}")

    @staticmethod
    def _binding_shape_from_hwc(input_shape_hwc: Sequence[int], engine_shape: Sequence[int]) -> Tuple[int, ...]:
        h, w, c = [int(x) for x in input_shape_hwc]
        shape = tuple(int(x) for x in engine_shape)
        if len(shape) == 4:
            if shape[1] in (1, 3) or shape[1] <= 0:
                return (1, c, h, w)
            if shape[-1] in (1, 3) or shape[-1] <= 0:
                return (1, h, w, c)
        if len(shape) == 3:
            if shape[0] in (1, 3) or shape[0] <= 0:
                return (c, h, w)
            if shape[-1] in (1, 3) or shape[-1] <= 0:
                return (h, w, c)
        raise RuntimeError(f"Nieobsługiwany binding shape wejścia: {shape}")

    def _binding_shape_runtime(self, idx: int) -> Tuple[int, ...]:
        shape = tuple(int(x) for x in self.context.get_binding_shape(idx))
        if any(d <= 0 for d in shape):
            shape = tuple(int(x) for x in self.engine.get_binding_shape(idx))
        if any(d <= 0 for d in shape):
            raise RuntimeError(f"Dynamiczny output binding bez ustalonego kształtu: idx={idx}, shape={shape}")
        return shape

    def _torch_dtype_from_np(self, dtype: Any):
        torch = self.torch
        dtype = np.dtype(dtype)
        if dtype == np.dtype(np.float32):
            return torch.float32
        if dtype == np.dtype(np.float16):
            return torch.float16
        if dtype == np.dtype(np.int32):
            return torch.int32
        if dtype == np.dtype(np.int8):
            return torch.int8
        if dtype == np.dtype(np.uint8):
            return torch.uint8
        if dtype == np.dtype(np.bool_):
            return torch.bool
        raise RuntimeError(f"Nieobsługiwany dtype TensorRT binding: {dtype}")

    def _allocate_torch_buffers(self) -> None:
        trt = self.trt
        torch = self.torch
        for idx in range(self.num_bindings):
            name = self.engine.get_binding_name(idx)
            np_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(idx)))
            torch_dtype = self._torch_dtype_from_np(np_dtype)
            if idx == self.input_index:
                shape = self.input_binding_shape
            else:
                shape = self._binding_shape_runtime(idx)
            tensor = torch.empty(tuple(int(x) for x in shape), device="cuda", dtype=torch_dtype)
            self.device_tensors[idx] = tensor
            self.bindings[idx] = int(tensor.data_ptr())
            self.binding_shapes[idx] = tuple(int(x) for x in shape)
            self.binding_np_dtypes[idx] = np_dtype
            self.binding_torch_dtypes[idx] = torch_dtype
            if self.verbose:
                print(f"[TRT] binding idx={idx} name={name} shape={shape} np_dtype={np_dtype} torch_dtype={torch_dtype} ptr={self.bindings[idx]}")

    def _prepare_input(self, input_tensor_hwc: np.ndarray) -> np.ndarray:
        h, w, c = self.input_shape
        if tuple(input_tensor_hwc.shape) != (h, w, c):
            raise RuntimeError(f"Niepoprawny input dla {self.engine_path.name}: got={input_tensor_hwc.shape}, expected={(h, w, c)}")

        if self.input_layout == "NCHW":
            x = np.transpose(input_tensor_hwc, (2, 0, 1))[None]
        elif self.input_layout == "NHWC":
            x = input_tensor_hwc[None]
        elif self.input_layout == "CHW":
            x = np.transpose(input_tensor_hwc, (2, 0, 1))
        elif self.input_layout == "HWC":
            x = input_tensor_hwc
        else:
            raise RuntimeError(f"Nieobsługiwany layout: {self.input_layout}")

        dtype = self.binding_np_dtypes[self.input_index]
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    def infer(self, input_tensor_hwc: np.ndarray) -> Dict[str, np.ndarray]:
        torch = self.torch
        x_np = self._prepare_input(input_tensor_hwc)
        in_tensor = self.device_tensors[self.input_index]

        # Kopiowanie host->GPU przez PyTorch, na bieżącym strumieniu CUDA.
        x_cpu = torch.from_numpy(x_np)
        in_tensor.copy_(x_cpu.to(device="cuda", dtype=in_tensor.dtype), non_blocking=False)

        stream = torch.cuda.current_stream()
        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=int(stream.cuda_stream))
        if not ok:
            raise RuntimeError(f"TensorRT execute_async_v2 zwróciło False dla {self.engine_path}")

        stream.synchronize()

        out: Dict[str, np.ndarray] = {}
        for idx in self.output_indices:
            name = self.engine.get_binding_name(idx)
            arr = self.device_tensors[idx].detach().cpu().numpy().reshape(self.binding_shapes[idx]).copy()
            if np.issubdtype(arr.dtype, np.floating) and arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            out[name] = arr
        return out

    def warmup(self, n: int = 5) -> None:
        if n <= 0:
            return
        x = np.zeros(self.input_shape, dtype=np.float32)
        for _ in range(int(n)):
            self.infer(x)

    def close(self) -> None:
        # Bufory zwalnia GC/PyTorch. Synchronizacja ogranicza ryzyko użycia
        # bufora po zakończeniu programu.
        try:
            self.torch.cuda.synchronize()
        except Exception:
            pass
        self.device_tensors = {}
        self.bindings = []

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
