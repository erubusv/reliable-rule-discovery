from __future__ import annotations

import ctypes
import threading
from pathlib import Path

import numpy as np

try:
    from . import _cpu_native
except ImportError:  # Source-tree and unsupported-platform reference path.
    _cpu_native = None

try:  # Conda NumPy uses MKL on the target workstation.
    import mkl as _mkl
except ImportError:  # Wheels using OpenBLAS remain fully supported.
    _mkl = None

_CUDA = None
_CUDA_LOCK = threading.Lock()


def cpu_available() -> bool:
    return _cpu_native is not None


def configure_cpu_threads(count: int) -> None:
    if int(count) < 1:
        raise ValueError("CPU thread count must be positive")
    if _cpu_native is not None:
        _cpu_native.set_num_threads(int(count))
    if _mkl is not None:
        _mkl.set_num_threads_local(int(count))


def _cuda_library():
    global _CUDA
    if _CUDA is False:
        return None
    if _CUDA is None:
        with _CUDA_LOCK:
            if _CUDA is None:
                path = Path(__file__).with_name("libcrbstpp_cuda.so")
                if not path.is_file():
                    _CUDA = False
                    return None
                library = ctypes.CDLL(str(path))
                function = library.crbstpp_cuda_moments
                pointer = ctypes.POINTER(ctypes.c_double)
                function.argtypes = [
                    ctypes.c_int,
                    pointer,
                    pointer,
                    pointer,
                    ctypes.c_int64,
                    ctypes.c_int64,
                    pointer,
                    pointer,
                ]
                function.restype = ctypes.c_int
                batch_function = library.crbstpp_cuda_moments_batch
                batch_function.argtypes = [
                    ctypes.c_int,
                    pointer,
                    pointer,
                    pointer,
                    ctypes.c_int64,
                    ctypes.c_int64,
                    ctypes.c_int64,
                    pointer,
                    pointer,
                ]
                batch_function.restype = ctypes.c_int
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
        gradient.fill(0.0)
        hessian.fill(0.0)
        tile_rows = max(1, min(262_144, 128 * 1024**2 // max(8, 8 * x.shape[1])))
        status = 0
        for start in range(0, x.shape[0], tile_rows):
            end = min(x.shape[0], start + tile_rows)
            tile_x = np.ascontiguousarray(x[start:end])
            tile_first = np.ascontiguousarray(first[start:end])
            tile_second = np.ascontiguousarray(second[start:end])
            tile_gradient = np.empty_like(gradient)
            tile_hessian = np.empty_like(hessian)
            status = _CUDA.crbstpp_cuda_moments(
                index,
                tile_x.ctypes.data_as(pointer),
                tile_first.ctypes.data_as(pointer),
                tile_second.ctypes.data_as(pointer),
                tile_x.shape[0],
                tile_x.shape[1],
                tile_gradient.ctypes.data_as(pointer),
                tile_hessian.ctypes.data_as(pointer),
            )
            if status != 0:
                break
            gradient += tile_gradient
            hessian += tile_hessian
        if status == 0:
            return gradient, hessian
    if _cpu_native is not None:
        _cpu_native.moments(x, first, second, gradient, hessian)
        return gradient, hessian
    return x.T @ first, x.T @ (second[:, None] * x)


def moments_batch(
    x: np.ndarray, first: np.ndarray, second: np.ndarray, *, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Moments for B candidate blocks sharing row derivatives."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    if x.ndim != 3 or first.shape != (x.shape[1],) or second.shape != (x.shape[1],):
        raise ValueError("batched moment buffer shape mismatch")
    gradient = np.empty((x.shape[0], x.shape[2]), dtype=np.float64)
    hessian = np.empty((x.shape[0], x.shape[2], x.shape[2]), dtype=np.float64)
    if device.startswith("cuda") and _cuda_library() is not None:
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        pointer = ctypes.POINTER(ctypes.c_double)
        status = _CUDA.crbstpp_cuda_moments_batch(
            index,
            x.ctypes.data_as(pointer),
            first.ctypes.data_as(pointer),
            second.ctypes.data_as(pointer),
            x.shape[0],
            x.shape[1],
            x.shape[2],
            gradient.ctypes.data_as(pointer),
            hessian.ctypes.data_as(pointer),
        )
        if status == 0:
            return gradient, hessian
    for index in range(x.shape[0]):
        gradient[index], hessian[index] = moments(x[index], first, second, device="cpu")
    return gradient, hessian


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
    count = int(
        _cpu_native.kernel_contributions(
            entities,
            times,
            spans,
            starts,
            ends,
            offsets,
            basis,
            int(window),
            rows,
            values,
        )
    )
    return rows[:count], values[:count]


def accumulate_kernel(
    entities: np.ndarray,
    times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    offsets: np.ndarray,
    basis: np.ndarray,
    lookup: np.ndarray,
    accumulator: np.ndarray,
) -> bool:
    """Accumulate one disjoint completion-span slice in place."""
    if _cpu_native is None or not hasattr(_cpu_native, "accumulate_kernel"):
        return False
    entities = np.ascontiguousarray(entities, dtype=np.int64)
    times = np.ascontiguousarray(times, dtype=np.int64)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    lookup = np.ascontiguousarray(lookup, dtype=np.int64)
    if accumulator.dtype != np.float64 or not accumulator.flags.c_contiguous:
        raise ValueError("kernel accumulator must be C-contiguous float64")
    _cpu_native.accumulate_kernel(
        entities,
        times,
        starts,
        ends,
        offsets,
        basis,
        lookup,
        accumulator,
    )
    return True


def completion_events(
    sources: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if _cpu_native is None:
        return None
    if not 1 <= len(sources) <= 3:
        raise ValueError("completion requires one to three predicate sources")
    lengths = np.asarray([len(entities) for entities, _ in sources], dtype=np.int64)
    offsets = np.zeros(len(sources) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    if int(offsets[-1]) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    entities = np.ascontiguousarray(
        np.concatenate([item[0] for item in sources]), dtype=np.int64
    )
    times = np.ascontiguousarray(
        np.concatenate([item[1] for item in sources]), dtype=np.int64
    )
    capacity = len(entities)
    output_entities = np.empty(capacity, dtype=np.int64)
    output_times = np.empty(capacity, dtype=np.int64)
    output_spans = np.empty(capacity, dtype=np.int64)
    count = int(
        _cpu_native.completion_events(
            entities,
            times,
            offsets,
            output_entities,
            output_times,
            output_spans,
        )
    )
    return (
        output_entities[:count],
        output_times[:count],
        output_spans[:count],
    )


def future_rows(
    entities: np.ndarray,
    times: np.ndarray,
    spans: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    offsets: np.ndarray,
    *,
    window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if _cpu_native is None:
        return None
    entities = np.ascontiguousarray(entities, dtype=np.int64)
    times = np.ascontiguousarray(times, dtype=np.int64)
    spans = np.ascontiguousarray(spans, dtype=np.int64)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    output = np.empty(len(entities) * int(horizon), dtype=np.int64)
    output_spans = np.empty_like(output)
    count = int(
        _cpu_native.future_rows(
            entities,
            times,
            spans,
            starts,
            ends,
            offsets,
            int(window),
            int(horizon),
            output,
            output_spans,
        )
    )
    return output[:count], output_spans[:count]
