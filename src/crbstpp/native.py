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


def bounded_span_order(spans: np.ndarray, maximum_span: int) -> np.ndarray | None:
    """Stable indices of nonnegative spans at most ``maximum_span``.

    The compiled counting pass is exact and linear in the number of spans plus
    the (small) integer window range.  ``None`` requests the NumPy fallback.
    """
    if _cpu_native is None or not hasattr(_cpu_native, "bounded_span_order"):
        return None
    if int(maximum_span) < 0:
        raise ValueError("maximum span must be nonnegative")
    # Continuous-time ticks can be arbitrarily fine; counting sort is only a
    # memory win when the integer range is compact.  Falling back changes no
    # ordering or values.
    if int(maximum_span) > 1_000_000:
        return None
    spans = np.ascontiguousarray(spans, dtype=np.int64)
    output = np.empty(len(spans), dtype=np.int64)
    count = int(_cpu_native.bounded_span_order(spans, int(maximum_span), output))
    return output[:count]


def sorted_unique_union(parts: list[np.ndarray]) -> np.ndarray | None:
    """Merge sorted unique int64 arrays exactly without concatenating/sorting."""
    if _cpu_native is None or not hasattr(_cpu_native, "sorted_unique_union"):
        return None
    arrays = [np.ascontiguousarray(part, dtype=np.int64) for part in parts]
    if not arrays:
        return np.zeros(0, dtype=np.int64)
    if len(arrays) == 1:
        return arrays[0]
    output = np.empty(sum(len(part) for part in arrays), dtype=np.int64)
    count = int(_cpu_native.sorted_unique_union(arrays, output))
    return output[:count]


def configure_cpu_threads(count: int) -> None:
    if int(count) < 1:
        raise ValueError("CPU thread count must be positive")
    if _cpu_native is not None:
        _cpu_native.set_num_threads(int(count))
    if _mkl is not None:
        _mkl.set_num_threads_local(int(count))


def aggregate_design_rows(
    x: np.ndarray,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    copy_input: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum likelihood weights for identical design rows without approximation."""
    x = np.array(x, dtype=np.float64, order="C", copy=copy_input)
    exposure = np.array(exposure_weight, dtype=np.float64, order="C", copy=copy_input)
    noevent = np.array(noevent_weight, dtype=np.float64, order="C", copy=copy_input)
    event = np.array(event_weight, dtype=np.float64, order="C", copy=copy_input)
    if x.ndim != 2 or any(
        weight.shape != (x.shape[0],) for weight in (exposure, noevent, event)
    ):
        raise ValueError("design aggregation shape mismatch")
    if _cpu_native is not None and hasattr(_cpu_native, "aggregate_design_rows"):
        count = int(_cpu_native.aggregate_design_rows(x, exposure, noevent, event))
        return (
            x[:count].copy(),
            exposure[:count].copy(),
            noevent[:count].copy(),
            event[:count].copy(),
        )
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    return (
        unique,
        np.bincount(inverse, weights=exposure, minlength=len(unique)),
        np.bincount(inverse, weights=noevent, minlength=len(unique)),
        np.bincount(inverse, weights=event, minlength=len(unique)),
    )


def aggregate_design_rows_with_groups(
    x: np.ndarray,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    copy_input: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate rows and retain each input row's exact output-group index."""
    x = np.array(x, dtype=np.float64, order="C", copy=copy_input)
    exposure = np.array(
        exposure_weight, dtype=np.float64, order="C", copy=copy_input
    )
    noevent = np.array(noevent_weight, dtype=np.float64, order="C", copy=copy_input)
    event = np.array(event_weight, dtype=np.float64, order="C", copy=copy_input)
    if x.ndim != 2 or any(
        weight.shape != (x.shape[0],) for weight in (exposure, noevent, event)
    ):
        raise ValueError("design aggregation shape mismatch")
    if _cpu_native is not None and hasattr(_cpu_native, "aggregate_design_rows"):
        groups = np.empty(x.shape[0], dtype=np.int64)
        count = int(
            _cpu_native.aggregate_design_rows(
                x, exposure, noevent, event, groups
            )
        )
        return (
            x[:count].copy(),
            exposure[:count].copy(),
            noevent[:count].copy(),
            event[:count].copy(),
            groups,
        )
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    return (
        unique,
        np.bincount(inverse, weights=exposure, minlength=len(unique)),
        np.bincount(inverse, weights=noevent, minlength=len(unique)),
        np.bincount(inverse, weights=event, minlength=len(unique)),
        inverse.astype(np.int64, copy=False),
    )


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
                    ctypes.c_int,
                    pointer,
                    pointer,
                    pointer,
                ]
                batch_function.restype = ctypes.c_int
                sparse_batch = library.crbstpp_cuda_sparse_moments_batch
                int64_pointer = ctypes.POINTER(ctypes.c_int64)
                sparse_batch.argtypes = [
                    ctypes.c_int,
                    int64_pointer,
                    pointer,
                    pointer,
                    pointer,
                    int64_pointer,
                    ctypes.c_int64,
                    ctypes.c_int64,
                    ctypes.c_int64,
                    ctypes.c_int64,
                    pointer,
                    pointer,
                    pointer,
                ]
                sparse_batch.restype = ctypes.c_int
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
        # cuBLAS reuses each tile across all Hessian columns.  Larger tiles
        # amortize allocation/transfer/handle overhead while x+weighted-x stay
        # below roughly 1 GiB on each 24 GiB device.
        tile_rows = max(
            1,
            min(1_048_576, 512 * 1024**2 // max(8, 8 * x.shape[1])),
        )
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
    # Host-resident exact-fit matrices are substantially faster through the
    # installed MKL/OpenBLAS DGEMM than through a scalar d^2 row scan.  Tiling
    # bounds the temporary weighted-design buffer without changing any sums
    # or the optimization problem.
    gradient.fill(0.0)
    hessian.fill(0.0)
    tile_rows = max(
        1,
        min(x.shape[0], 512 * 1024**2 // max(8, 8 * x.shape[1])),
    )
    for start in range(0, x.shape[0], tile_rows):
        end = min(x.shape[0], start + tile_rows)
        tile = x[start:end]
        gradient += tile.T @ first[start:end]
        hessian += tile.T @ (second[start:end, None] * tile)
    return gradient, hessian


def moments_batch(
    x: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    device: str = "cpu",
    return_second_gradient: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moments for B candidate blocks sharing row derivatives."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    if x.ndim != 3 or first.shape != (x.shape[1],) or second.shape != (x.shape[1],):
        raise ValueError("batched moment buffer shape mismatch")
    gradient = np.empty((x.shape[0], x.shape[2]), dtype=np.float64)
    hessian = np.empty((x.shape[0], x.shape[2], x.shape[2]), dtype=np.float64)
    second_gradient = np.empty_like(gradient)
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
            int(return_second_gradient),
            gradient.ctypes.data_as(pointer),
            hessian.ctypes.data_as(pointer),
            second_gradient.ctypes.data_as(pointer),
        )
        if status == 0:
            return (
                (gradient, hessian, second_gradient)
                if return_second_gradient
                else (gradient, hessian)
            )
    for index in range(x.shape[0]):
        gradient[index], hessian[index] = moments(x[index], first, second, device="cpu")
        if return_second_gradient:
            second_gradient[index] = x[index].T @ second
    return (
        (gradient, hessian, second_gradient)
        if return_second_gradient
        else (gradient, hessian)
    )


def sparse_moments_batch(
    candidates: list[
        tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]
    ],
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Exact ragged sparse moments for equally shaped candidate block sets.

    Each block tuple contains sorted rows, M-knot values, and the first/second
    likelihood derivatives evaluated at those rows.  The CUDA kernel performs
    deterministic per-output reductions and never constructs a dense row union.
    """
    library = _cuda_library()
    if (
        not candidates
        or not device.startswith("cuda")
        or library is None
        or not hasattr(library, "crbstpp_cuda_sparse_moments_batch")
    ):
        return None
    block_count = len(candidates[0])
    if block_count < 1 or any(len(candidate) != block_count for candidate in candidates):
        raise ValueError("ragged candidates must have one common block count")
    knot_count = candidates[0][0][1].shape[1]
    rows_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    offsets = [0]
    for candidate in candidates:
        for rows, values, first, second in candidate:
            rows = np.ascontiguousarray(rows, dtype=np.int64)
            values = np.ascontiguousarray(values, dtype=np.float64)
            first = np.ascontiguousarray(first, dtype=np.float64)
            second = np.ascontiguousarray(second, dtype=np.float64)
            if (
                values.shape != (len(rows), knot_count)
                or first.shape != (len(rows),)
                or second.shape != (len(rows),)
            ):
                raise ValueError("ragged sparse block shape mismatch")
            rows_parts.append(rows)
            value_parts.append(values)
            first_parts.append(first)
            second_parts.append(second)
            offsets.append(offsets[-1] + len(rows))
    total_rows = offsets[-1]
    if total_rows:
        rows = np.ascontiguousarray(np.concatenate(rows_parts), dtype=np.int64)
        values = np.ascontiguousarray(np.concatenate(value_parts), dtype=np.float64)
        first = np.ascontiguousarray(np.concatenate(first_parts), dtype=np.float64)
        second = np.ascontiguousarray(np.concatenate(second_parts), dtype=np.float64)
    else:
        rows = np.zeros(0, dtype=np.int64)
        values = np.zeros((0, knot_count), dtype=np.float64)
        first = np.zeros(0, dtype=np.float64)
        second = np.zeros(0, dtype=np.float64)
    block_offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    dimensions = block_count * knot_count
    gradient = np.empty((len(candidates), dimensions), dtype=np.float64)
    hessian = np.empty(
        (len(candidates), dimensions, dimensions), dtype=np.float64
    )
    cross = np.empty_like(gradient)
    pointer = ctypes.POINTER(ctypes.c_double)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    status = library.crbstpp_cuda_sparse_moments_batch(
        index,
        rows.ctypes.data_as(int64_pointer),
        values.ctypes.data_as(pointer),
        first.ctypes.data_as(pointer),
        second.ctypes.data_as(pointer),
        block_offsets.ctypes.data_as(int64_pointer),
        total_rows,
        len(candidates),
        block_count,
        knot_count,
        gradient.ctypes.data_as(pointer),
        hessian.ctypes.data_as(pointer),
        cross.ctypes.data_as(pointer),
    )
    return (gradient, hessian, cross) if status == 0 else None


def nonnegative_quadratic_gains(
    gradients: np.ndarray, hessians: np.ndarray
) -> np.ndarray | None:
    """Exact active-set gains for a batch of tiny nonnegative quadratics."""
    gradients = np.ascontiguousarray(gradients, dtype=np.float64)
    hessians = np.ascontiguousarray(hessians, dtype=np.float64)
    if (
        gradients.ndim != 2
        or hessians.shape
        != (gradients.shape[0], gradients.shape[1], gradients.shape[1])
    ):
        raise ValueError("quadratic gain batch shape mismatch")
    if _cpu_native is None or not hasattr(
        _cpu_native, "nonnegative_quadratic_gains"
    ):
        return None
    output = np.empty(gradients.shape[0], dtype=np.float64)
    _cpu_native.nonnegative_quadratic_gains(gradients, hessians, output)
    return output


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
    if lookup.dtype not in (np.int32, np.int64):
        lookup = np.asarray(lookup, dtype=np.int64)
    lookup = np.ascontiguousarray(lookup)
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


def fill_candidate_batch(
    destination: np.ndarray,
    maximum_rows: np.ndarray,
    lookup: np.ndarray | None,
    blocks: list[tuple[np.ndarray, np.ndarray]],
    batch_indices: np.ndarray,
    column_offsets: np.ndarray,
    tile_start: int,
) -> bool:
    """Fill one dense hierarchy-pricing tile from sparse blocks in C++."""
    if (
        _cpu_native is None
        or not hasattr(_cpu_native, "fill_candidate_batch")
        or lookup is None
    ):
        return False
    if destination.dtype != np.float64 or not destination.flags.c_contiguous:
        raise ValueError("candidate destination must be C-contiguous float64")
    maximum_rows = np.ascontiguousarray(maximum_rows, dtype=np.int64)
    lookup = np.ascontiguousarray(lookup)
    rows = [np.ascontiguousarray(item[0], dtype=np.int64) for item in blocks]
    values = [np.ascontiguousarray(item[1], dtype=np.float64) for item in blocks]
    batch_indices = np.ascontiguousarray(batch_indices, dtype=np.int64)
    column_offsets = np.ascontiguousarray(column_offsets, dtype=np.int64)
    _cpu_native.fill_candidate_batch(
        destination,
        maximum_rows,
        lookup,
        rows,
        values,
        batch_indices,
        column_offsets,
        int(tile_start),
    )
    return True


def sparse_joint_moments(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    lookup: np.ndarray | None,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Exact gradient, Fisher matrix and intercept cross of sparse blocks."""
    if (
        _cpu_native is None
        or not hasattr(_cpu_native, "sparse_joint_moments")
        or lookup is None
        or not blocks
    ):
        return None
    rows = [np.ascontiguousarray(item[0], dtype=np.int64) for item in blocks]
    values = [np.ascontiguousarray(item[1], dtype=np.float64) for item in blocks]
    width = values[0].shape[1]
    dimension = len(blocks) * width
    lookup = np.ascontiguousarray(lookup)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    gradient = np.empty(dimension, dtype=np.float64)
    hessian = np.empty((dimension, dimension), dtype=np.float64)
    cross = np.empty(dimension, dtype=np.float64)
    _cpu_native.sparse_joint_moments(
        rows,
        values,
        lookup,
        first,
        second,
        gradient,
        hessian,
        cross,
    )
    return gradient, hessian, cross


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


def completion_window_counts(
    sources: list[tuple[np.ndarray, np.ndarray]],
    ends: np.ndarray,
    windows: np.ndarray,
) -> np.ndarray | None:
    """Count productive completions for nested W without materializing them."""
    if _cpu_native is None or not hasattr(_cpu_native, "completion_window_counts"):
        return None
    if not 1 <= len(sources) <= 3:
        raise ValueError("completion requires one to three predicate sources")
    windows = np.ascontiguousarray(windows, dtype=np.int64)
    if windows.ndim != 1 or not len(windows):
        raise ValueError("completion windows must be a nonempty vector")
    lengths = np.asarray([len(entities) for entities, _ in sources], dtype=np.int64)
    offsets = np.zeros(len(sources) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    entities = np.ascontiguousarray(
        np.concatenate([item[0] for item in sources]), dtype=np.int64
    )
    times = np.ascontiguousarray(
        np.concatenate([item[1] for item in sources]), dtype=np.int64
    )
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    counts = np.empty(len(windows), dtype=np.int64)
    _cpu_native.completion_window_counts(
        entities, times, offsets, ends, windows, counts
    )
    return counts


def response_min_spans(
    entities: np.ndarray,
    times: np.ndarray,
    spans: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    offsets: np.ndarray,
    *,
    horizon: int,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return exact response rows/minimum spans through a bounded dense bitmap."""
    if _cpu_native is None or not hasattr(_cpu_native, "response_min_spans"):
        return None
    entities = np.ascontiguousarray(entities, dtype=np.int64)
    times = np.ascontiguousarray(times, dtype=np.int64)
    spans = np.ascontiguousarray(spans, dtype=np.int64)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    sentinel = np.iinfo(np.int32).max
    threshold = np.full(int(n_grid), sentinel, dtype=np.int32)
    _cpu_native.response_min_spans(
        entities, times, spans, starts, ends, offsets, int(horizon), threshold
    )
    active = threshold != sentinel
    return (
        np.flatnonzero(active).astype(np.int64, copy=False),
        threshold[active].astype(np.int64),
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
