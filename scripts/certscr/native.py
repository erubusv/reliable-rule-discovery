from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Sequence

import numpy as np


_BUILD_GUARD = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_LOAD_ATTEMPTED = False


def _load_completion_library() -> ctypes.CDLL | None:
    global _LIBRARY, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LIBRARY
    with _BUILD_GUARD:
        if _LOAD_ATTEMPTED:
            return _LIBRARY
        source = Path(__file__).with_name("native_completion.cpp")
        try:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            target = Path(tempfile.gettempdir()) / f"certscr_completion_{digest}.so"
            if not target.exists():
                temporary = target.with_name(
                    f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                subprocess.run(
                    [
                        "g++",
                        "-O3",
                        "-std=c++17",
                        "-shared",
                        "-fPIC",
                        str(source),
                        "-o",
                        str(temporary),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            library = ctypes.CDLL(str(target))
            function = library.certscr_linear_completions
            int_pointer = ctypes.POINTER(ctypes.c_int32)
            function.argtypes = [
                ctypes.c_int32,
                ctypes.POINTER(int_pointer),
                ctypes.POINTER(int_pointer),
                ctypes.POINTER(ctypes.c_int64),
                int_pointer,
                ctypes.c_int64,
                ctypes.c_int32,
                int_pointer,
                int_pointer,
                int_pointer,
                ctypes.c_int64,
            ]
            function.restype = ctypes.c_int64
            sorted_union = library.certscr_sorted_unique_int64_union
            int64_pointer = ctypes.POINTER(ctypes.c_int64)
            sorted_union.argtypes = [
                ctypes.POINTER(int64_pointer),
                int64_pointer,
                ctypes.c_int32,
                int64_pointer,
                ctypes.c_int64,
            ]
            sorted_union.restype = ctypes.c_int64
            grid_sequences = library.certscr_sorted_grid_sequences
            grid_sequences.argtypes = [
                int64_pointer,
                ctypes.c_int64,
                int64_pointer,
                ctypes.c_int64,
                int_pointer,
            ]
            grid_sequences.restype = ctypes.c_int64
            union_positions = (
                library.certscr_sorted_unique_int64_union_with_positions
            )
            union_positions.argtypes = [
                ctypes.POINTER(int64_pointer),
                int64_pointer,
                ctypes.c_int32,
                int64_pointer,
                ctypes.POINTER(int_pointer),
                ctypes.c_int64,
                ctypes.c_int32,
            ]
            union_positions.restype = ctypes.c_int64
            component_integral = library.certscr_sparse_component_integral
            component_integral.argtypes = [
                int64_pointer,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int64,
                ctypes.c_double,
                int64_pointer,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                int64_pointer,
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_double),
            ]
            component_integral.restype = ctypes.c_int64
            add_predictor = library.certscr_add_sparse_linear_predictor
            add_predictor.argtypes = [
                int_pointer,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int64,
            ]
            add_predictor.restype = ctypes.c_int64
            grouping = library.certscr_group_float32_rows
            grouping.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
            ]
            grouping.restype = ctypes.c_int64
            partitioned_grouping = library.certscr_group_float32_rows_partitioned
            partitioned_grouping.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int64),
            ]
            partitioned_grouping.restype = ctypes.c_int64
            pair_grouping = library.certscr_group_float32_rows_pair
            pair_grouping.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            pair_grouping.restype = ctypes.c_int64
            sparse_block = library.certscr_sparse_kernel_block
            sparse_block.argtypes = [
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int64,
            ]
            sparse_block.restype = ctypes.c_int64
            cone_fit = library.certscr_fit_prepared_cone
            double_pointer = ctypes.POINTER(ctypes.c_double)
            cone_fit.argtypes = [
                double_pointer,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int32,
                double_pointer,
                double_pointer,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_double,
                double_pointer,
                double_pointer,
                double_pointer,
                double_pointer,
                int_pointer,
            ]
            cone_fit.restype = ctypes.c_int64
            _LIBRARY = library
        except (OSError, AttributeError, subprocess.SubprocessError):
            # The NumPy implementation remains an exact portable fallback.
            _LIBRARY = None
        _LOAD_ATTEMPTED = True
        return _LIBRARY


def linear_completions(
    source_sequences: Sequence[np.ndarray],
    source_times: Sequence[np.ndarray],
    sequence_lookup: np.ndarray,
    *,
    max_span: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Native O(total source events) latest-witness sweep, if available."""
    library = _load_completion_library()
    if library is None:
        return None
    q = len(source_sequences)
    if q < 1 or q > 3 or len(source_times) != q:
        raise ValueError("native completion supports one to three aligned sources")
    sequences = [np.ascontiguousarray(value, dtype=np.int32) for value in source_sequences]
    times = [np.ascontiguousarray(value, dtype=np.int32) for value in source_times]
    if any(len(seq) != len(time) for seq, time in zip(sequences, times, strict=True)):
        raise ValueError("source sequence/time arrays differ in length")
    lookup = np.ascontiguousarray(sequence_lookup, dtype=np.int32)
    capacity = int(sum(len(value) for value in sequences))
    if capacity == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty.copy(), empty.copy()
    output_sequence = np.empty(capacity, dtype=np.int32)
    output_time = np.empty(capacity, dtype=np.int32)
    output_span = np.empty(capacity, dtype=np.int32)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    sequence_pointers = (int_pointer * q)(
        *(value.ctypes.data_as(int_pointer) for value in sequences)
    )
    time_pointers = (int_pointer * q)(
        *(value.ctypes.data_as(int_pointer) for value in times)
    )
    lengths = np.asarray([len(value) for value in sequences], dtype=np.int64)
    written = int(
        library.certscr_linear_completions(
            q,
            sequence_pointers,
            time_pointers,
            lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            lookup.ctypes.data_as(int_pointer),
            len(lookup),
            -1 if max_span is None else int(max_span),
            output_sequence.ctypes.data_as(int_pointer),
            output_time.ctypes.data_as(int_pointer),
            output_span.ctypes.data_as(int_pointer),
            capacity,
        )
    )
    if written < 0:
        raise RuntimeError(f"native completion sweep failed with status {written}")
    # These arrays own their buffers, so resize can trim conservative capacity
    # in place (or realloc once) without allocating three simultaneous copies.
    output_sequence.resize(written, refcheck=False)
    output_time.resize(written, refcheck=False)
    output_span.resize(written, refcheck=False)
    return output_sequence, output_time, output_span


def sorted_unique_int64_union(
    arrays: Sequence[np.ndarray],
) -> np.ndarray | None:
    """Union strictly increasing int64 arrays without concatenate/sort copies."""
    library = _load_completion_library()
    if library is None:
        return None
    values = [
        np.ascontiguousarray(array, dtype=np.int64).reshape(-1)
        for array in arrays
        if len(array)
    ]
    if not values:
        return np.zeros(0, dtype=np.int64)
    if len(values) == 1:
        return values[0].copy()
    # The allocation-free head scan wins consistently for the q<=3
    # antecedent/completion path.  Wider support unions are faster in NumPy's
    # vectorized contiguous sort on the target CPU.
    if len(values) > 3:
        return None
    capacity = int(sum(len(value) for value in values))
    output = np.empty(capacity, dtype=np.int64)
    pointer = ctypes.POINTER(ctypes.c_int64)
    pointers = (pointer * len(values))(
        *(value.ctypes.data_as(pointer) for value in values)
    )
    lengths = np.asarray([len(value) for value in values], dtype=np.int64)
    written = int(
        library.certscr_sorted_unique_int64_union(
            pointers,
            lengths.ctypes.data_as(pointer),
            len(values),
            output.ctypes.data_as(pointer),
            capacity,
        )
    )
    if written < 0:
        raise RuntimeError(f"native sorted union failed with status {written}")
    output.resize(written, refcheck=False)
    return output


def sorted_grid_sequences(
    offsets: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray | None:
    """Map sorted grid rows to contiguous sequence intervals in one sweep."""
    library = _load_completion_library()
    if library is None:
        return None
    boundaries = np.ascontiguousarray(offsets, dtype=np.int64).reshape(-1)
    requested = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1)
    if len(boundaries) < 2:
        raise ValueError("grid sequence offsets require at least one interval")
    output = np.empty(len(requested), dtype=np.int32)
    if not len(requested):
        return output
    written = int(
        library.certscr_sorted_grid_sequences(
            boundaries.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(boundaries) - 1,
            requested.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(requested),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
    )
    if written < 0:
        raise RuntimeError(f"native grid-sequence sweep failed with status {written}")
    return output


def sorted_unique_int64_union_with_positions(
    arrays: Sequence[np.ndarray],
    *,
    assume_sorted: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]] | None:
    """Union sorted arrays and map every input value to its union row."""
    library = _load_completion_library()
    if library is None:
        return None
    values = [
        np.ascontiguousarray(array, dtype=np.int64).reshape(-1)
        for array in arrays
    ]
    capacity = int(sum(len(value) for value in values))
    if capacity > np.iinfo(np.int32).max:
        return None
    if not values:
        return np.zeros(0, dtype=np.int64), ()
    output = np.empty(capacity, dtype=np.int64)
    mapped = tuple(np.empty(len(value), dtype=np.int32) for value in values)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    input_pointers = (int64_pointer * len(values))(
        *(value.ctypes.data_as(int64_pointer) for value in values)
    )
    output_position_pointers = (int_pointer * len(values))(
        *(position.ctypes.data_as(int_pointer) for position in mapped)
    )
    lengths = np.asarray([len(value) for value in values], dtype=np.int64)
    written = int(
        library.certscr_sorted_unique_int64_union_with_positions(
            input_pointers,
            lengths.ctypes.data_as(int64_pointer),
            len(values),
            output.ctypes.data_as(int64_pointer),
            output_position_pointers,
            capacity,
            0 if assume_sorted else 1,
        )
    )
    if written < 0:
        raise RuntimeError(
            f"native sorted union/layout failed with status {written}"
        )
    output.resize(written, refcheck=False)
    return output, mapped


def sparse_component_integral(
    summary_indices: np.ndarray,
    summary_eta: np.ndarray,
    inactive_eta: float,
    block_indices: np.ndarray,
    block_values: np.ndarray,
    coefficients: np.ndarray,
    row_weights: np.ndarray | None,
    sequence_offsets: np.ndarray,
    *,
    assume_sorted: bool = False,
) -> np.ndarray | None:
    """Integrate one sparse scalar block per sequence in a fused native sweep."""
    library = _load_completion_library()
    if library is None:
        return None
    summary_rows = np.ascontiguousarray(summary_indices, dtype=np.int64)
    eta = np.ascontiguousarray(summary_eta, dtype=np.float64)
    rows = np.ascontiguousarray(block_indices, dtype=np.int64)
    values = np.ascontiguousarray(block_values, dtype=np.float32)
    beta = np.ascontiguousarray(coefficients, dtype=np.float64)
    weights = (
        None
        if row_weights is None
        else np.ascontiguousarray(row_weights, dtype=np.float64)
    )
    offsets = np.ascontiguousarray(sequence_offsets, dtype=np.int64)
    if (
        summary_rows.ndim != 1
        or eta.shape != summary_rows.shape
        or rows.ndim != 1
        or values.ndim != 2
        or values.shape[0] != len(rows)
        or beta.shape != (values.shape[1],)
        or (weights is not None and weights.shape != rows.shape)
        or offsets.ndim != 1
        or len(offsets) < 2
    ):
        raise ValueError("native sparse component inputs are not aligned")
    output = np.zeros(len(offsets) - 1, dtype=np.float64)
    if not len(rows):
        return output
    written = int(
        library.certscr_sparse_component_integral(
            summary_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            eta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(summary_rows),
            float(inactive_eta),
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(rows),
            values.shape[1],
            beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            (
                None
                if weights is None
                else weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            ),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(offsets) - 1,
            0 if assume_sorted else 1,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if written < 0:
        raise RuntimeError(
            f"native sparse component integration failed with status {written}"
        )
    return output


def add_sparse_linear_predictor(
    positions: np.ndarray,
    values: np.ndarray,
    coefficients: np.ndarray,
    output: np.ndarray,
    *,
    scale: float = 1.0,
) -> bool:
    """Fused float32 sparse-row matvec and float64 scatter-add, if available."""
    library = _load_completion_library()
    if library is None:
        return False
    mapped = np.ascontiguousarray(positions, dtype=np.int32)
    matrix = np.ascontiguousarray(values, dtype=np.float32)
    beta = np.ascontiguousarray(coefficients, dtype=np.float64)
    target = np.asarray(output)
    if (
        mapped.ndim != 1
        or matrix.ndim != 2
        or matrix.shape[0] != len(mapped)
        or beta.shape != (matrix.shape[1],)
        or target.ndim != 1
        or target.dtype != np.float64
        or not target.flags.c_contiguous
    ):
        raise ValueError("native sparse predictor inputs are not aligned")
    written = int(
        library.certscr_add_sparse_linear_predictor(
            mapped.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(mapped),
            matrix.shape[1],
            beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            float(scale),
            target.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(target),
        )
    )
    if written < 0:
        raise RuntimeError(
            f"native sparse predictor accumulation failed with status {written}"
        )
    return True


def aggregate_float32_rows(
    matrix: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Group bit-identical rows in expected O(rows), with an exact fallback."""
    library = _load_completion_library()
    if library is None:
        return None
    design = np.ascontiguousarray(matrix, dtype=np.float32)
    mass = np.ascontiguousarray(weights, dtype=np.float64)
    if design.ndim != 2 or mass.shape != (len(design),):
        raise ValueError("native grouping input dimensions differ")
    if design.shape[1] < 1:
        raise ValueError("native grouping requires at least one design column")
    if not len(design):
        return design.copy(), mass.copy()
    output_design = np.empty_like(design)
    output_mass = np.empty_like(mass)
    written = int(
        library.certscr_group_float32_rows(
            design.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(design),
            design.shape[1],
            output_design.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if written < 0:
        raise RuntimeError(f"native row grouping failed with status {written}")
    output_design.resize((written, design.shape[1]), refcheck=False)
    output_mass.resize(written, refcheck=False)
    return output_design, output_mass


def aggregate_float32_rows_partitioned(
    matrix: np.ndarray,
    n_events: int,
    event_weights: np.ndarray,
    grid_weights: np.ndarray,
    *,
    inplace: bool = False,
) -> tuple[np.ndarray, int, np.ndarray] | None:
    """Group event/grid rows separately into one exact output allocation."""
    library = _load_completion_library()
    if library is None:
        return None
    design = np.ascontiguousarray(matrix, dtype=np.float32)
    event_mass = np.ascontiguousarray(event_weights, dtype=np.float64)
    grid_mass = np.ascontiguousarray(grid_weights, dtype=np.float64)
    event_count = int(n_events)
    if (
        design.ndim != 2
        or not 0 <= event_count <= len(design)
        or event_mass.shape != (event_count,)
        or grid_mass.shape != (len(design) - event_count,)
    ):
        raise ValueError("native partitioned grouping input dimensions differ")
    if design.shape[1] < 1:
        raise ValueError("native partitioned grouping requires a design column")
    if not len(design):
        return design.copy(), 0, np.zeros(0, dtype=np.float64)
    output_design = (
        design
        if inplace and design.flags.owndata and design.flags.writeable
        else np.empty_like(design)
    )
    output_mass = np.empty(len(design), dtype=np.float64)
    output_events = ctypes.c_int64(0)
    written = int(
        library.certscr_group_float32_rows_partitioned(
            design.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            event_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            grid_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(design),
            event_count,
            design.shape[1],
            output_design.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(output_events),
        )
    )
    if written < 0:
        raise RuntimeError(f"native partitioned row grouping failed with status {written}")
    output_design.resize((written, design.shape[1]), refcheck=False)
    output_mass.resize(written, refcheck=False)
    return output_design, int(output_events.value), output_mass


def aggregate_float32_rows_pair(
    matrix: np.ndarray,
    first_weights: np.ndarray,
    second_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Group rows once while aggregating two exact sufficient-statistic masses."""
    library = _load_completion_library()
    if library is None:
        return None
    design = np.ascontiguousarray(matrix, dtype=np.float32)
    first = np.ascontiguousarray(first_weights, dtype=np.float64)
    second = np.ascontiguousarray(second_weights, dtype=np.float64)
    if (
        design.ndim != 2
        or first.shape != (len(design),)
        or second.shape != (len(design),)
    ):
        raise ValueError("native paired grouping input dimensions differ")
    if design.shape[1] < 1:
        raise ValueError("native paired grouping requires a design column")
    if not len(design):
        return design.copy(), first.copy(), second.copy()
    output_design = np.empty_like(design)
    output_first = np.empty_like(first)
    output_second = np.empty_like(second)
    written = int(
        library.certscr_group_float32_rows_pair(
            design.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            first.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            second.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(design),
            design.shape[1],
            output_design.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_first.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            output_second.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if written < 0:
        raise RuntimeError(f"native paired row grouping failed with status {written}")
    output_design.resize((written, design.shape[1]), refcheck=False)
    output_first.resize(written, refcheck=False)
    output_second.resize(written, refcheck=False)
    return output_design, output_first, output_second


def sparse_kernel_block(
    base_indices: np.ndarray,
    occurrence_times: np.ndarray,
    end_times: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Accumulate a discrete kernel block without occurrence×lag temporaries."""
    library = _load_completion_library()
    if library is None:
        return None
    base = np.ascontiguousarray(base_indices, dtype=np.int64)
    occurrence = np.ascontiguousarray(occurrence_times, dtype=np.int64)
    ends = np.ascontiguousarray(end_times, dtype=np.int64)
    kernels = np.ascontiguousarray(basis, dtype=np.float32)
    if (
        base.shape != occurrence.shape
        or base.shape != ends.shape
        or base.ndim != 1
        or kernels.ndim != 2
    ):
        raise ValueError("native sparse-kernel inputs are not aligned")
    if not len(base):
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, kernels.shape[0]), dtype=np.float32),
        )
    valid_lengths = np.clip(ends - occurrence, 0, kernels.shape[1])
    capacity = int(np.sum(valid_lengths, dtype=np.int64))
    if capacity == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, kernels.shape[0]), dtype=np.float32),
        )
    output_indices = np.empty(capacity, dtype=np.int64)
    output_values = np.empty((capacity, kernels.shape[0]), dtype=np.float32)
    written = int(
        library.certscr_sparse_kernel_block(
            base.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            occurrence.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(base),
            kernels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kernels.shape[0],
            kernels.shape[1],
            output_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            output_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            capacity,
        )
    )
    if written < 0:
        raise RuntimeError(f"native sparse-kernel accumulation failed with status {written}")
    output_indices.resize(written, refcheck=False)
    output_values.resize((written, kernels.shape[0]), refcheck=False)
    return output_indices, output_values


def fit_prepared_cone_float64(
    design: np.ndarray,
    n_events: int,
    event_weights: np.ndarray,
    grid_weights: np.ndarray,
    constrained_start: int,
    initial_values: np.ndarray,
    *,
    occurrence_likelihood: str,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, float, float, int] | None:
    """Solve one grouped cone GLM in native float64, if KKT-converged.

    The native routine implements the same convex objective, nonnegative rule
    cone and Armijo/KKT contract as the portable NumPy solver.  Returning
    ``None`` is fail-open: callers rerun the established scalar path, so an
    unavailable compiler, singular Newton system or nonconvergence can never
    change model admission.
    """
    library = _load_completion_library()
    if library is None:
        return None
    matrix = np.ascontiguousarray(design, dtype=np.float64)
    event_mass = np.ascontiguousarray(event_weights, dtype=np.float64)
    grid_mass = np.ascontiguousarray(grid_weights, dtype=np.float64)
    initial = np.ascontiguousarray(initial_values, dtype=np.float64)
    event_count = int(n_events)
    boundary = int(constrained_start)
    if (
        matrix.ndim != 2
        or not 0 <= event_count <= len(matrix)
        or event_mass.shape != (event_count,)
        or grid_mass.shape != (len(matrix) - event_count,)
        or initial.shape != (matrix.shape[1],)
        or not 1 <= boundary <= matrix.shape[1]
    ):
        raise ValueError("native cone-fit inputs are not aligned")
    likelihood_code = {
        "poisson": 0,
        "first_event_cloglog": 1,
    }.get(str(occurrence_likelihood))
    if likelihood_code is None:
        raise ValueError(f"unknown occurrence likelihood: {occurrence_likelihood}")
    output = np.empty_like(initial)
    objective = ctypes.c_double(float("inf"))
    kkt = ctypes.c_double(float("inf"))
    iterations = ctypes.c_int32(0)
    pointer = ctypes.POINTER(ctypes.c_double)
    status = int(
        library.certscr_fit_prepared_cone(
            matrix.ctypes.data_as(pointer),
            len(matrix),
            event_count,
            matrix.shape[1],
            event_mass.ctypes.data_as(pointer),
            grid_mass.ctypes.data_as(pointer),
            boundary,
            likelihood_code,
            int(max_iter),
            float(tolerance),
            initial.ctypes.data_as(pointer),
            output.ctypes.data_as(pointer),
            ctypes.byref(objective),
            ctypes.byref(kkt),
            ctypes.byref(iterations),
        )
    )
    if status == 1:
        return None
    if status < 0:
        raise RuntimeError(f"native prepared-cone fit failed with status {status}")
    return output, float(objective.value), float(kkt.value), int(iterations.value)
