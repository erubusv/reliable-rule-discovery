from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

try:
    from . import _cpu_native
except ImportError:  # Source-tree and unsupported-platform reference path.
    _cpu_native = None

_CUDA = None


def cpu_available() -> bool:
    return _cpu_native is not None


def _cuda_library():
    global _CUDA
    if _CUDA is False:
        return None
    if _CUDA is None:
        path = Path(__file__).with_name("libcrbstpp_cuda.so")
        if not path.is_file():
            _CUDA = False
            return None
        library = ctypes.CDLL(str(path))
        function = library.crbstpp_cuda_moments
        pointer = ctypes.POINTER(ctypes.c_double)
        function.argtypes = [
            ctypes.c_int, pointer, pointer, pointer, ctypes.c_int64, ctypes.c_int64,
            pointer, pointer,
        ]
        function.restype = ctypes.c_int
        _CUDA = library
    return _CUDA


def cuda_available() -> bool:
    return _cuda_library() is not None


def moments(
    x: np.ndarray, first: np.ndarray, second: np.ndarray, *, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    gradient = np.empty(x.shape[1], dtype=np.float64)
    hessian = np.empty((x.shape[1], x.shape[1]), dtype=np.float64)
    if device.startswith("cuda") and _cuda_library() is not None:
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        pointer = ctypes.POINTER(ctypes.c_double)
        status = _CUDA.crbstpp_cuda_moments(
            index,
            x.ctypes.data_as(pointer), first.ctypes.data_as(pointer), second.ctypes.data_as(pointer),
            x.shape[0], x.shape[1], gradient.ctypes.data_as(pointer), hessian.ctypes.data_as(pointer),
        )
        if status == 0:
            return gradient, hessian
    if _cpu_native is not None:
        _cpu_native.moments(x, first, second, gradient, hessian)
        return gradient, hessian
    return x.T @ first, x.T @ (second[:, None] * x)


def kernel_contributions(
    entities: np.ndarray,
    times: np.ndarray,
    spans: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    offsets: np.ndarray,
    basis: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if _cpu_native is None:
        return None
    entities = np.ascontiguousarray(entities, dtype=np.int64)
    times = np.ascontiguousarray(times, dtype=np.int64)
    spans = np.ascontiguousarray(spans, dtype=np.int64)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    capacity = len(entities) * basis.shape[1]
    rows = np.empty(capacity, dtype=np.int64)
    values = np.empty((capacity, basis.shape[0]), dtype=np.float64)
    count = int(_cpu_native.kernel_contributions(
        entities, times, spans, starts, ends, offsets, basis, int(window), rows, values
    ))
    return rows[:count], values[:count]

