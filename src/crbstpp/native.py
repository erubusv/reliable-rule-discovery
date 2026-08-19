from __future__ import annotations

import ctypes
import itertools
import threading
from dataclasses import dataclass
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
_DERIVATIVE_TOKENS = itertools.count(1)
_GEOMETRY_TOKENS = itertools.count(1)
_VALIDATION_LOCK = threading.Lock()
_VALIDATED_IMPLICIT_SOURCES: set[int] = set()
_VALIDATED_IMPLICIT_DERIVATIVES: set[int] = set()
_LIBC = None
_POISSON_LIKELIHOODS = frozenset({"poisson", "continuous_poisson"})


@dataclass(frozen=True)
class SparseMomentGeometry:
    rows: np.ndarray
    values: np.ndarray
    block_offsets: np.ndarray
    block_count: int
    knot_count: int
    token: int


def cpu_available() -> bool:
    return _cpu_native is not None


def trim_host_allocator() -> bool:
    """Return unused glibc arenas after a large exact-fit wave.

    NumPy/PyTorch temporaries can be freed while glibc retains their
    multi-GiB arenas. Trimming changes no live array or arithmetic;
    unsupported allocators simply return ``False``.
    """

    global _LIBC
    try:
        if _LIBC is None:
            library = ctypes.CDLL(None)
            function = getattr(library, "malloc_trim", None)
            if function is None:
                _LIBC = False
                return False
            function.argtypes = [ctypes.c_size_t]
            function.restype = ctypes.c_int
            _LIBC = (library, function)
        if _LIBC is False:
            return False
        return bool(_LIBC[1](0))
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def incremental_kernel_available() -> bool:
    """Whether exact nested-W touched-row tracking is compiled in."""
    return _cpu_native is not None and hasattr(_cpu_native, "kernel_touched_positions")


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


def completion_entity_offsets(
    entities: np.ndarray,
    pattern_starts: np.ndarray,
    pattern_ends: np.ndarray,
    entity_count: int,
    *,
    workers: int,
) -> np.ndarray | None:
    """Build exact absolute per-entity offsets for a compact motif wave."""
    function = (
        None
        if _cpu_native is None
        else getattr(_cpu_native, "completion_entity_offsets", None)
    )
    if function is None:
        return None
    entities = np.ascontiguousarray(entities, dtype=np.int32)
    pattern_starts = np.ascontiguousarray(pattern_starts, dtype=np.int64)
    pattern_ends = np.ascontiguousarray(pattern_ends, dtype=np.int64)
    if (
        pattern_starts.ndim != 1
        or pattern_ends.shape != pattern_starts.shape
        or not len(pattern_starts)
        or int(entity_count) < 1
        or int(workers) < 1
    ):
        raise ValueError("compact completion offset arguments are invalid")
    output = np.empty(
        (len(pattern_starts), int(entity_count) + 1), dtype=np.int64
    )
    function(
        entities,
        pattern_starts,
        pattern_ends,
        int(entity_count),
        int(workers),
        output,
    )
    return output


def completion_entity_profiles(
    entities: np.ndarray,
    times: np.ndarray,
    spans: np.ndarray,
    pattern_starts: np.ndarray,
    pattern_ends: np.ndarray,
    entity_ends: np.ndarray,
    *,
    workers: int,
    output_entities: np.ndarray,
    output_minimum_spans: np.ndarray,
) -> np.ndarray | None:
    """Build exact strict-future minimum-span profiles for all patterns."""
    function = (
        None
        if _cpu_native is None
        else getattr(_cpu_native, "completion_entity_profiles", None)
    )
    if function is None:
        return None
    entities = np.ascontiguousarray(entities, dtype=np.int32)
    times = np.ascontiguousarray(times, dtype=np.int64)
    spans = np.ascontiguousarray(spans, dtype=np.int64)
    pattern_starts = np.ascontiguousarray(pattern_starts, dtype=np.int64)
    pattern_ends = np.ascontiguousarray(pattern_ends, dtype=np.int64)
    entity_ends = np.ascontiguousarray(entity_ends, dtype=np.int64)
    output_entities = np.asarray(output_entities)
    output_minimum_spans = np.asarray(output_minimum_spans)
    if (
        times.shape != entities.shape
        or spans.shape != entities.shape
        or pattern_starts.ndim != 1
        or pattern_ends.shape != pattern_starts.shape
        or entity_ends.ndim != 1
        or output_entities.dtype != np.int32
        or output_minimum_spans.dtype != np.int64
        or output_entities.shape != entities.shape
        or output_minimum_spans.shape != entities.shape
        or not output_entities.flags.c_contiguous
        or not output_minimum_spans.flags.c_contiguous
        or int(workers) < 1
    ):
        raise ValueError("compact completion profile arguments are invalid")
    counts = np.empty(len(pattern_starts), dtype=np.int64)
    function(
        entities,
        times,
        spans,
        pattern_starts,
        pattern_ends,
        entity_ends,
        int(workers),
        counts,
        output_entities,
        output_minimum_spans,
    )
    return counts


def candidate_entities_from_profiles(
    profile_entities: np.ndarray,
    minimum_spans: np.ndarray,
    pattern_starts: np.ndarray,
    pattern_counts: np.ndarray,
    candidate_patterns: np.ndarray,
    thresholds: np.ndarray,
    *,
    workers: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pack exact W-admissible entities for a one-block candidate wave."""
    function = (
        None
        if _cpu_native is None
        else getattr(_cpu_native, "candidate_entities_from_profiles", None)
    )
    if function is None:
        return None
    profile_entities = np.ascontiguousarray(profile_entities, dtype=np.int32)
    minimum_spans = np.ascontiguousarray(minimum_spans, dtype=np.int64)
    pattern_starts = np.ascontiguousarray(pattern_starts, dtype=np.int64)
    pattern_counts = np.ascontiguousarray(pattern_counts, dtype=np.int64)
    candidate_patterns = np.ascontiguousarray(candidate_patterns, dtype=np.int32)
    thresholds = np.ascontiguousarray(thresholds, dtype=np.int64)
    if (
        minimum_spans.shape != profile_entities.shape
        or pattern_counts.shape != pattern_starts.shape
        or candidate_patterns.ndim != 1
        or thresholds.shape != candidate_patterns.shape
        or not len(candidate_patterns)
        or int(workers) < 1
        or np.any(candidate_patterns < 0)
        or np.any(candidate_patterns >= len(pattern_starts))
    ):
        raise ValueError("candidate entity profile arguments are invalid")
    upper = int(np.sum(pattern_counts[candidate_patterns], dtype=np.int64))
    offsets = np.empty(len(candidate_patterns) + 1, dtype=np.int64)
    entities = np.empty(upper, dtype=np.int32)
    count = int(
        function(
            profile_entities,
            minimum_spans,
            pattern_starts,
            pattern_counts,
            candidate_patterns,
            thresholds,
            int(workers),
            offsets,
            entities,
        )
    )
    return offsets, entities[:count]


def configure_cpu_threads(count: int) -> None:
    if int(count) < 1:
        raise ValueError("CPU thread count must be positive")
    if _cpu_native is not None:
        _cpu_native.set_num_threads(int(count))
    if _mkl is not None:
        _mkl.set_num_threads_local(int(count))


def fused_likelihood_value_eta_gradient(
    x: np.ndarray,
    beta: np.ndarray,
    primary_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    likelihood: str,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Exact OpenMP NLL, eta and gradient without a Fisher matrix."""
    function = (
        None
        if _cpu_native is None
        else getattr(_cpu_native, "likelihood_value_eta_gradient", None)
    )
    if function is None:
        return None
    x = np.ascontiguousarray(x, dtype=np.float64)
    beta = np.ascontiguousarray(beta, dtype=np.float64)
    primary = np.ascontiguousarray(primary_weight, dtype=np.float64)
    event = np.ascontiguousarray(event_weight, dtype=np.float64)
    if (
        x.ndim != 2
        or beta.shape != (x.shape[1],)
        or primary.shape != (x.shape[0],)
        or event.shape != (x.shape[0],)
    ):
        raise ValueError("fused likelihood buffers are misaligned")
    mode = (
        1
        if likelihood in _POISSON_LIKELIHOODS
        else 2 if likelihood == "first_event_cloglog" else None
    )
    if mode is None:
        return None
    eta = np.empty(x.shape[0], dtype=np.float64)
    gradient = np.empty(x.shape[1], dtype=np.float64)
    nll = float(function(x, beta, primary, event, mode, eta, gradient))
    return nll, eta, gradient


def design_column_cross(x: np.ndarray, column: int) -> np.ndarray | None:
    """Return one exact column of X'X using a fused OpenMP row pass."""
    function = (
        None
        if _cpu_native is None
        else getattr(_cpu_native, "design_column_cross", None)
    )
    if function is None:
        return None
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2 or not 0 <= int(column) < x.shape[1]:
        raise ValueError("design cross column is invalid")
    output = np.empty(x.shape[1], dtype=np.float64)
    function(x, int(column), output)
    return output


def fill_cloglog_mixed_conjugate(
    dual: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    output: np.ndarray,
) -> bool:
    """Fill mixed-row cloglog conjugates with the compiled float64 solver.

    ``False`` requests the algebraically identical Python reference fallback.
    Non-mixed positions are deliberately left unchanged.
    """
    if _cpu_native is None or not hasattr(_cpu_native, "cloglog_mixed_conjugate"):
        return False
    dual = np.ascontiguousarray(dual, dtype=np.float64)
    noevent = np.ascontiguousarray(noevent_weight, dtype=np.float64)
    event = np.ascontiguousarray(event_weight, dtype=np.float64)
    if (
        output.dtype != np.float64
        or not output.flags.c_contiguous
        or output.shape != dual.shape
        or noevent.shape != dual.shape
        or event.shape != dual.shape
    ):
        raise ValueError("compiled cloglog conjugate shape mismatch")
    _cpu_native.cloglog_mixed_conjugate(dual, noevent, event, output)
    return True


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
        if count == x.shape[0]:
            return x, exposure, noevent, event
        original_columns = x.shape[1]
        try:
            # The compiled aggregator already compacted these owning scratch
            # buffers in place.  Shrinking their logical allocation avoids
            # copying the complete compact matrix and three weight vectors.
            x.resize((count, original_columns), refcheck=False)
            exposure.resize((count,), refcheck=False)
            noevent.resize((count,), refcheck=False)
            event.resize((count,), refcheck=False)
            return x, exposure, noevent, event
        except ValueError:
            # Non-owning public inputs retain the exact former copy path.
            pass
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


def aggregate_quotient_rows(
    old_groups: np.ndarray,
    values: np.ndarray,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    worker_count: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Exactly aggregate ``(old_group, values...)`` without a signature copy.

    This is the state-quotient Add-bound hot path. The native implementation
    partitions exact hash classes across CPU workers; collisions are resolved
    by elementwise equality, so parallelism never changes the quotient.
    """
    old_groups = np.ascontiguousarray(old_groups, dtype=np.int64)
    values = np.ascontiguousarray(values, dtype=np.float64)
    exposure = np.ascontiguousarray(exposure_weight, dtype=np.float64)
    noevent = np.ascontiguousarray(noevent_weight, dtype=np.float64)
    event = np.ascontiguousarray(event_weight, dtype=np.float64)
    if values.ndim != 2 or old_groups.shape != (len(values),) or any(
        weight.shape != (len(values),) for weight in (exposure, noevent, event)
    ):
        raise ValueError("quotient aggregation shape mismatch")
    if len(old_groups) and int(np.min(old_groups)) < 0:
        raise ValueError("quotient groups must be nonnegative")
    group_count = int(np.max(old_groups, initial=-1)) + 1
    if worker_count < 0:
        raise ValueError("quotient worker count must be nonnegative")
    if _cpu_native is not None and hasattr(_cpu_native, "aggregate_quotient_rows"):
        output_exposure = np.empty(len(values), dtype=np.float64)
        output_noevent = np.empty(len(values), dtype=np.float64)
        output_event = np.empty(len(values), dtype=np.float64)
        removed_exposure = np.empty(group_count, dtype=np.float64)
        removed_noevent = np.empty(group_count, dtype=np.float64)
        removed_event = np.empty(group_count, dtype=np.float64)
        count = int(
            _cpu_native.aggregate_quotient_rows(
                old_groups,
                values,
                exposure,
                noevent,
                event,
                output_exposure,
                output_noevent,
                output_event,
                removed_exposure,
                removed_noevent,
                removed_event,
                int(worker_count),
            )
        )
        return (
            output_exposure[:count],
            output_noevent[:count],
            output_event[:count],
            removed_exposure,
            removed_noevent,
            removed_event,
        )
    signatures = np.empty((len(values), values.shape[1] + 1), dtype=np.float64)
    signatures[:, 0] = old_groups
    signatures[:, 1:] = values
    _, output_exposure, output_noevent, output_event = aggregate_design_rows(
        signatures,
        exposure,
        noevent,
        event,
        copy_input=False,
    )
    return (
        output_exposure,
        output_noevent,
        output_event,
        np.bincount(old_groups, weights=exposure, minlength=group_count),
        np.bincount(old_groups, weights=noevent, minlength=group_count),
        np.bincount(old_groups, weights=event, minlength=group_count),
    )


def subtract_group_weights(
    groups: np.ndarray,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    base_exposure: np.ndarray,
    base_noevent: np.ndarray,
    base_event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract three likelihood weights by group in stable input order."""
    groups = np.ascontiguousarray(groups, dtype=np.int64)
    exposure = np.ascontiguousarray(exposure_weight, dtype=np.float64)
    noevent = np.ascontiguousarray(noevent_weight, dtype=np.float64)
    event = np.ascontiguousarray(event_weight, dtype=np.float64)
    output_exposure = np.array(base_exposure, dtype=np.float64, order="C", copy=True)
    output_noevent = np.array(base_noevent, dtype=np.float64, order="C", copy=True)
    output_event = np.array(base_event, dtype=np.float64, order="C", copy=True)
    count = len(output_exposure)
    if any(
        weight.shape != groups.shape for weight in (exposure, noevent, event)
    ) or output_noevent.shape != (count,) or output_event.shape != (count,):
        raise ValueError("group weight shape mismatch")
    if _cpu_native is not None and hasattr(_cpu_native, "subtract_group_weights"):
        _cpu_native.subtract_group_weights(
            groups,
            exposure,
            noevent,
            event,
            output_exposure,
            output_noevent,
            output_event,
        )
        return output_exposure, output_noevent, output_event
    if len(groups) and (int(groups.min()) < 0 or int(groups.max()) >= count):
        raise ValueError("group weight index is out of range")
    np.add.at(output_exposure, groups, -exposure)
    np.add.at(output_noevent, groups, -noevent)
    np.add.at(output_event, groups, -event)
    return output_exposure, output_noevent, output_event


def entity_loss_contrast(
    null_group_eta: np.ndarray,
    full_group_eta: np.ndarray,
    null_baseline_loss: np.ndarray,
    full_baseline_loss: np.ndarray,
    active_rows: np.ndarray,
    active_design_groups: np.ndarray,
    active_baseline_groups: np.ndarray,
    entity_offsets: np.ndarray,
    entity_weights: np.ndarray,
    target_rows: np.ndarray,
    target_counts: np.ndarray,
    *,
    tick_exposure: float,
    likelihood: str,
    workers: int,
) -> np.ndarray | None:
    """Stream an exact active-row NLL contrast into entity totals.

    ``None`` requests the NumPy reference path when the installed extension
    predates this operator.  The implementation consumes the compact design
    predictors and row-to-group maps directly, so it never materializes an
    ``n_active x 2`` predictor or dense event vector.
    """

    function = (
        None
        if _cpu_native is None
        else getattr(_cpu_native, "entity_loss_contrast", None)
    )
    if function is None:
        return None
    null_group_eta = np.ascontiguousarray(null_group_eta, dtype=np.float64)
    full_group_eta = np.ascontiguousarray(full_group_eta, dtype=np.float64)
    null_baseline_loss = np.ascontiguousarray(null_baseline_loss, dtype=np.float64)
    full_baseline_loss = np.ascontiguousarray(full_baseline_loss, dtype=np.float64)
    active_rows = np.ascontiguousarray(active_rows, dtype=np.int64)
    active_design_groups = np.ascontiguousarray(
        active_design_groups, dtype=np.int32
    )
    active_baseline_groups = np.ascontiguousarray(
        active_baseline_groups, dtype=np.int32
    )
    entity_offsets = np.ascontiguousarray(entity_offsets, dtype=np.int64)
    entity_weights = np.ascontiguousarray(entity_weights, dtype=np.float64)
    target_rows = np.ascontiguousarray(target_rows, dtype=np.int64)
    target_counts = np.ascontiguousarray(target_counts, dtype=np.float64)
    output = np.empty(len(entity_weights), dtype=np.float64)
    if likelihood not in {*_POISSON_LIKELIHOODS, "first_event_cloglog"}:
        raise ValueError(f"unknown likelihood: {likelihood}")
    function(
        null_group_eta,
        full_group_eta,
        null_baseline_loss,
        full_baseline_loss,
        active_rows,
        active_design_groups,
        active_baseline_groups,
        entity_offsets,
        entity_weights,
        target_rows,
        target_counts,
        float(tick_exposure),
        0 if likelihood in _POISSON_LIKELIHOODS else 1,
        max(0, int(workers)),
        output,
    )
    return output


def dependency_row_derivatives(
    group_eta: np.ndarray,
    active_rows: np.ndarray,
    active_design_groups: np.ndarray,
    active_baseline_groups: np.ndarray,
    entity_offsets: np.ndarray,
    entity_starts: np.ndarray,
    entity_weights: np.ndarray,
    dependency_codes: np.ndarray,
    target_rows: np.ndarray,
    target_counts: np.ndarray,
    baseline_first: np.ndarray,
    *,
    origin: int,
    tick_exposure: float,
    likelihood: str,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Create exact active-row dependency statistics without dense gathers."""

    function = (
        None
        if _cpu_native is None
        else getattr(_cpu_native, "dependency_row_derivatives", None)
    )
    if function is None:
        return None
    group_eta = np.ascontiguousarray(group_eta, dtype=np.float64)
    active_rows = np.ascontiguousarray(active_rows, dtype=np.int64)
    active_design_groups = np.ascontiguousarray(
        active_design_groups, dtype=np.int32
    )
    active_baseline_groups = np.ascontiguousarray(
        active_baseline_groups, dtype=np.int32
    )
    entity_offsets = np.ascontiguousarray(entity_offsets, dtype=np.int64)
    entity_starts = np.ascontiguousarray(entity_starts, dtype=np.int64)
    entity_weights = np.ascontiguousarray(entity_weights, dtype=np.float64)
    dependency_codes = np.ascontiguousarray(dependency_codes, dtype=np.int32)
    target_rows = np.ascontiguousarray(target_rows, dtype=np.int64)
    target_counts = np.ascontiguousarray(target_counts, dtype=np.float64)
    baseline_first = np.ascontiguousarray(baseline_first, dtype=np.float64)
    if likelihood not in {*_POISSON_LIKELIHOODS, "first_event_cloglog"}:
        raise ValueError(f"unknown likelihood: {likelihood}")
    count = len(active_rows)
    first = np.empty(count, dtype=np.float64)
    entity_cluster = np.empty(count, dtype=np.int32)
    time_cluster = np.empty(count, dtype=np.int32)
    default_first = np.empty(count, dtype=np.float64)
    function(
        group_eta,
        active_rows,
        active_design_groups,
        active_baseline_groups,
        entity_offsets,
        entity_starts,
        entity_weights,
        dependency_codes,
        target_rows,
        target_counts,
        baseline_first,
        int(origin),
        float(tick_exposure),
        0 if likelihood in _POISSON_LIKELIHOODS else 1,
        max(1, int(workers)),
        first,
        entity_cluster,
        time_cluster,
        default_first,
    )
    return first, entity_cluster, time_cluster, default_first


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
    exposure = np.array(exposure_weight, dtype=np.float64, order="C", copy=copy_input)
    noevent = np.array(noevent_weight, dtype=np.float64, order="C", copy=copy_input)
    event = np.array(event_weight, dtype=np.float64, order="C", copy=copy_input)
    if x.ndim != 2 or any(
        weight.shape != (x.shape[0],) for weight in (exposure, noevent, event)
    ):
        raise ValueError("design aggregation shape mismatch")
    if _cpu_native is not None and hasattr(_cpu_native, "aggregate_design_rows"):
        groups = np.empty(x.shape[0], dtype=np.int64)
        count = int(
            _cpu_native.aggregate_design_rows(x, exposure, noevent, event, groups)
        )
        if count == x.shape[0]:
            return x, exposure, noevent, event, groups
        original_columns = x.shape[1]
        try:
            x.resize((count, original_columns), refcheck=False)
            exposure.resize((count,), refcheck=False)
            noevent.resize((count,), refcheck=False)
            event.resize((count,), refcheck=False)
            return x, exposure, noevent, event, groups
        except ValueError:
            pass
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
                resident_moments = getattr(
                    library, "crbstpp_cuda_moments_resident", None
                )
                if resident_moments is not None:
                    resident_moments.argtypes = [
                        ctypes.c_int,
                        ctypes.c_int64,
                        pointer,
                        pointer,
                        pointer,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        pointer,
                        pointer,
                    ]
                    resident_moments.restype = ctypes.c_int
                resident_eta = getattr(library, "crbstpp_cuda_eta_resident", None)
                if resident_eta is not None:
                    resident_eta.argtypes = [
                        ctypes.c_int,
                        ctypes.c_int64,
                        pointer,
                        pointer,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        pointer,
                    ]
                    resident_eta.restype = ctypes.c_int
                resident_poisson = getattr(
                    library, "crbstpp_cuda_poisson_objective_resident", None
                )
                if resident_poisson is not None:
                    resident_poisson.argtypes = [
                        ctypes.c_int,
                        ctypes.c_int64,
                        pointer,
                        pointer,
                        pointer,
                        pointer,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int,
                        pointer,
                        pointer,
                        pointer,
                        pointer,
                    ]
                    resident_poisson.restype = ctypes.c_int
                resident_cloglog = getattr(
                    library, "crbstpp_cuda_cloglog_objective_resident", None
                )
                if resident_cloglog is not None:
                    resident_cloglog.argtypes = [
                        ctypes.c_int,
                        ctypes.c_int64,
                        pointer,
                        pointer,
                        pointer,
                        pointer,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int,
                        pointer,
                        pointer,
                        pointer,
                        pointer,
                    ]
                    resident_cloglog.restype = ctypes.c_int
                resident_projected = getattr(
                    library,
                    "crbstpp_cuda_projected_objective_resident",
                    None,
                )
                if resident_projected is not None:
                    resident_projected.argtypes = [
                        ctypes.c_int,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        pointer,
                        pointer,
                        ctypes.POINTER(ctypes.c_int64),
                        pointer,
                        pointer,
                        pointer,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int,
                        ctypes.c_int,
                        pointer,
                        pointer,
                        pointer,
                    ]
                    resident_projected.restype = ctypes.c_int
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
                indexed_sparse_batch = getattr(
                    library, "crbstpp_cuda_sparse_moments_indexed_batch", None
                )
                if indexed_sparse_batch is not None:
                    indexed_sparse_batch.argtypes = [
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
                        ctypes.c_int64,
                        ctypes.c_int64,
                        pointer,
                        pointer,
                        pointer,
                    ]
                    indexed_sparse_batch.restype = ctypes.c_int
                resident_sparse = getattr(
                    library,
                    "crbstpp_cuda_sparse_moments_indexed_resident",
                    None,
                )
                if resident_sparse is not None:
                    resident_sparse.argtypes = [
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
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        pointer,
                        pointer,
                        pointer,
                    ]
                    resident_sparse.restype = ctypes.c_int
                implicit_batch = getattr(
                    library, "crbstpp_cuda_implicit_moments_batch", None
                )
                if implicit_batch is not None:
                    int_pointer = ctypes.POINTER(ctypes.c_int)
                    uint8_pointer = ctypes.POINTER(ctypes.c_uint8)
                    implicit_batch.argtypes = [
                        ctypes.c_int,
                        int64_pointer,
                        int64_pointer,
                        int64_pointer,
                        ctypes.c_int,
                        int64_pointer,
                        int64_pointer,
                        int64_pointer,
                        pointer,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int,
                        ctypes.c_int,
                        int_pointer,
                        int_pointer,
                        int64_pointer,
                        int64_pointer,
                        int_pointer,
                        int64_pointer,
                        int_pointer,
                        ctypes.c_int,
                        ctypes.c_int,
                        pointer,
                        pointer,
                        uint8_pointer,
                        pointer,
                        pointer,
                        ctypes.c_int,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        int_pointer,
                        pointer,
                        ctypes.c_int64,
                        ctypes.c_int,
                        int_pointer,
                        ctypes.c_int,
                        ctypes.c_int,
                        uint8_pointer,
                        int_pointer,
                        ctypes.c_int,
                        ctypes.c_int,
                        int_pointer,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        pointer,
                        pointer,
                        pointer,
                        pointer,
                        pointer,
                    ]
                    implicit_batch.restype = ctypes.c_int
                implicit_objective = getattr(
                    library,
                    "crbstpp_cuda_implicit_objective_batch",
                    None,
                )
                if implicit_objective is not None:
                    int_pointer = ctypes.POINTER(ctypes.c_int)
                    implicit_objective.argtypes = [
                        ctypes.c_int,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_int,
                        ctypes.c_int,
                        int_pointer,
                        int_pointer,
                        int64_pointer,
                        int64_pointer,
                        int_pointer,
                        int64_pointer,
                        int_pointer,
                        ctypes.c_int,
                        ctypes.c_int,
                        pointer,
                        pointer,
                        ctypes.c_int64,
                        ctypes.c_int,
                        ctypes.c_int,
                        pointer,
                    ]
                    implicit_objective.restype = ctypes.c_int
                release_workspace = getattr(
                    library, "crbstpp_cuda_release_workspace", None
                )
                if release_workspace is not None:
                    release_workspace.argtypes = [ctypes.c_int]
                    release_workspace.restype = ctypes.c_int
                _CUDA = library
    return _CUDA


def cuda_available() -> bool:
    return _cuda_library() is not None


def release_cuda_workspaces(devices: tuple[str, ...]) -> bool:
    """Release persistent native GPU scratch after a completed pipeline phase."""
    library = _cuda_library()
    function = (
        None
        if library is None
        else getattr(library, "crbstpp_cuda_release_workspace", None)
    )
    if function is None:
        return False
    indices = sorted(
        {
            int(device.split(":", 1)[1]) if ":" in device else 0
            for device in devices
            if device.startswith("cuda")
        }
    )
    return all(int(function(index)) == 0 for index in indices)


def resident_eta(
    x: np.ndarray,
    beta: np.ndarray,
    *,
    device: str,
    matrix_token: int,
) -> np.ndarray | None:
    """Evaluate X@beta while an exact-fit design remains resident on one GPU."""
    library = _cuda_library()
    function = (
        None if library is None else getattr(library, "crbstpp_cuda_eta_resident", None)
    )
    x = np.ascontiguousarray(x, dtype=np.float64)
    beta = np.ascontiguousarray(beta, dtype=np.float64)
    # X and its Hessian-weighted scratch coexist.  A 6-GiB design consumes at
    # most 12 GiB with that scratch, leaving half of a 24-GiB device for
    # derivatives and implicit pricing buffers.
    if (
        function is None
        or not device.startswith("cuda")
        or x.ndim != 2
        or beta.shape != (x.shape[1],)
        or x.nbytes > 6 * 1024**3
    ):
        return None
    eta = np.empty(x.shape[0], dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    status = function(
        index,
        int(matrix_token),
        x.ctypes.data_as(pointer),
        beta.ctypes.data_as(pointer),
        x.shape[0],
        x.shape[1],
        eta.ctypes.data_as(pointer),
    )
    return eta if status == 0 else None


def resident_poisson_objective(
    x: np.ndarray,
    beta: np.ndarray,
    exposure_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    device: str,
    matrix_token: int,
    compute_moments: bool | int,
    return_eta: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray | None] | None:
    """Evaluate a complete Poisson Newton state without host round trips.

    The immutable design and likelihood weights stay resident on one GPU.
    Intensity, NLL, score weights, Fisher weights and (when requested)
    gradient/Hessian are produced in one CUDA/cuBLAS call.  Only the scalar,
    small moments and optionally ``eta`` return to Python.
    """

    library = _cuda_library()
    function = (
        None
        if library is None
        else getattr(library, "crbstpp_cuda_poisson_objective_resident", None)
    )
    x = np.ascontiguousarray(x, dtype=np.float64)
    beta = np.ascontiguousarray(beta, dtype=np.float64)
    exposure = np.ascontiguousarray(exposure_weight, dtype=np.float64)
    event = np.ascontiguousarray(event_weight, dtype=np.float64)
    if (
        function is None
        or not device.startswith("cuda")
        or x.ndim != 2
        or beta.shape != (x.shape[1],)
        or exposure.shape != (x.shape[0],)
        or event.shape != (x.shape[0],)
        or x.nbytes > 6 * 1024**3
    ):
        return None
    nll = np.empty(1, dtype=np.float64)
    eta = np.empty(x.shape[0], dtype=np.float64) if return_eta else None
    moment_mode = int(compute_moments)
    gradient = np.empty(x.shape[1], dtype=np.float64) if moment_mode else None
    hessian = (
        np.empty((x.shape[1], x.shape[1]), dtype=np.float64)
        if moment_mode == 1
        else None
    )
    pointer = ctypes.POINTER(ctypes.c_double)
    null_pointer = pointer()
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    status = function(
        index,
        int(matrix_token),
        x.ctypes.data_as(pointer),
        beta.ctypes.data_as(pointer),
        exposure.ctypes.data_as(pointer),
        event.ctypes.data_as(pointer),
        x.shape[0],
        x.shape[1],
        moment_mode,
        nll.ctypes.data_as(pointer),
        null_pointer if eta is None else eta.ctypes.data_as(pointer),
        null_pointer if gradient is None else gradient.ctypes.data_as(pointer),
        null_pointer if hessian is None else hessian.ctypes.data_as(pointer),
    )
    if status != 0:
        return None
    return float(nll[0]), eta, gradient, hessian


def resident_projected_objective(
    x: np.ndarray,
    beta: np.ndarray,
    columns: np.ndarray,
    scales: np.ndarray,
    primary_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    likelihood: str,
    device: str,
    matrix_token: int,
    projection_token: int,
    compute_moments: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None] | None:
    """Evaluate one exact signed projection of a resident source design."""

    library = _cuda_library()
    function = (
        None
        if library is None
        else getattr(
            library, "crbstpp_cuda_projected_objective_resident", None
        )
    )
    x = np.ascontiguousarray(x, dtype=np.float64)
    beta = np.ascontiguousarray(beta, dtype=np.float64)
    columns = np.ascontiguousarray(columns, dtype=np.int64)
    scales = np.ascontiguousarray(scales, dtype=np.float64)
    primary = np.ascontiguousarray(primary_weight, dtype=np.float64)
    event = np.ascontiguousarray(event_weight, dtype=np.float64)
    projected_bytes = int(x.shape[0]) * int(len(columns)) * 8 if x.ndim == 2 else 0
    if (
        function is None
        or not device.startswith("cuda")
        or x.ndim != 2
        or beta.shape != (len(columns),)
        or columns.ndim != 1
        or scales.shape != columns.shape
        or primary.shape != (x.shape[0],)
        or event.shape != (x.shape[0],)
        or not len(columns)
        or np.any(columns < 0)
        or np.any(columns >= x.shape[1])
        or x.nbytes > 6 * 1024**3
        or x.nbytes + 2 * projected_bytes > 14 * 1024**3
    ):
        return None
    nll = np.empty(1, dtype=np.float64)
    gradient = np.empty(len(columns), dtype=np.float64) if compute_moments else None
    hessian = (
        np.empty((len(columns), len(columns)), dtype=np.float64)
        if compute_moments
        else None
    )
    pointer = ctypes.POINTER(ctypes.c_double)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    null_pointer = pointer()
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    mode = 1 if likelihood in _POISSON_LIKELIHOODS else 2
    status = function(
        index,
        int(matrix_token),
        int(projection_token),
        x.ctypes.data_as(pointer),
        beta.ctypes.data_as(pointer),
        columns.ctypes.data_as(int64_pointer),
        scales.ctypes.data_as(pointer),
        primary.ctypes.data_as(pointer),
        event.ctypes.data_as(pointer),
        x.shape[0],
        x.shape[1],
        len(columns),
        mode,
        int(bool(compute_moments)),
        nll.ctypes.data_as(pointer),
        null_pointer if gradient is None else gradient.ctypes.data_as(pointer),
        null_pointer if hessian is None else hessian.ctypes.data_as(pointer),
    )
    if status != 0:
        return None
    return float(nll[0]), gradient, hessian


def resident_cloglog_objective(
    x: np.ndarray,
    beta: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    device: str,
    matrix_token: int,
    compute_moments: bool | int,
    return_eta: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray | None] | None:
    """Evaluate an exact first-event cloglog Newton state on a resident GPU."""

    library = _cuda_library()
    function = (
        None
        if library is None
        else getattr(library, "crbstpp_cuda_cloglog_objective_resident", None)
    )
    x = np.ascontiguousarray(x, dtype=np.float64)
    beta = np.ascontiguousarray(beta, dtype=np.float64)
    noevent = np.ascontiguousarray(noevent_weight, dtype=np.float64)
    event = np.ascontiguousarray(event_weight, dtype=np.float64)
    if (
        function is None
        or not device.startswith("cuda")
        or x.ndim != 2
        or beta.shape != (x.shape[1],)
        or noevent.shape != (x.shape[0],)
        or event.shape != (x.shape[0],)
        or x.nbytes > 6 * 1024**3
    ):
        return None
    nll = np.empty(1, dtype=np.float64)
    eta = np.empty(x.shape[0], dtype=np.float64) if return_eta else None
    moment_mode = int(compute_moments)
    gradient = np.empty(x.shape[1], dtype=np.float64) if moment_mode else None
    hessian = (
        np.empty((x.shape[1], x.shape[1]), dtype=np.float64)
        if moment_mode == 1
        else None
    )
    pointer = ctypes.POINTER(ctypes.c_double)
    null_pointer = pointer()
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    status = function(
        index,
        int(matrix_token),
        x.ctypes.data_as(pointer),
        beta.ctypes.data_as(pointer),
        noevent.ctypes.data_as(pointer),
        event.ctypes.data_as(pointer),
        x.shape[0],
        x.shape[1],
        moment_mode,
        nll.ctypes.data_as(pointer),
        null_pointer if eta is None else eta.ctypes.data_as(pointer),
        null_pointer if gradient is None else gradient.ctypes.data_as(pointer),
        null_pointer if hessian is None else hessian.ctypes.data_as(pointer),
    )
    if status != 0:
        return None
    return float(nll[0]), eta, gradient, hessian


def moments(
    x: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    device: str = "cpu",
    matrix_token: int | None = None,
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
        resident_function = getattr(_CUDA, "crbstpp_cuda_moments_resident", None)
        if (
            matrix_token is not None
            and resident_function is not None
            and x.nbytes <= 6 * 1024**3
        ):
            status = resident_function(
                index,
                int(matrix_token),
                x.ctypes.data_as(pointer),
                first.ctypes.data_as(pointer),
                second.ctypes.data_as(pointer),
                x.shape[0],
                x.shape[1],
                gradient.ctypes.data_as(pointer),
                hessian.ctypes.data_as(pointer),
            )
            if status == 0:
                return gradient, hessian
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


def accumulate_cluster_scores(
    design: np.ndarray,
    row_groups: np.ndarray,
    clusters: np.ndarray,
    multipliers: np.ndarray,
    output: np.ndarray,
) -> None:
    """Add ``multiplier[row] * design[group[row]]`` by cluster exactly."""

    design = np.ascontiguousarray(design, dtype=np.float64)
    row_groups = np.ascontiguousarray(row_groups, dtype=np.int32)
    clusters = np.ascontiguousarray(clusters, dtype=np.int32)
    multipliers = np.ascontiguousarray(multipliers, dtype=np.float64)
    if (
        design.ndim != 2
        or row_groups.ndim != 1
        or clusters.shape != row_groups.shape
        or multipliers.shape != row_groups.shape
        or output.dtype != np.float64
        or output.ndim != 2
        or output.shape[1] != design.shape[1]
        or output.shape[0] < int(np.max(clusters, initial=-1)) + 1
        or not output.flags.c_contiguous
    ):
        raise ValueError("cluster score buffers are misaligned")
    function = (
        None
        if _cpu_native is None
        else getattr(_cpu_native, "accumulate_cluster_scores", None)
    )
    if function is not None:
        function(design, row_groups, clusters, multipliers, output)
        return
    # Reference fallback is intentionally column-wise: it bounds peak memory
    # and is byte-for-byte deterministic for tests without the extension.
    for column in range(design.shape[1]):
        output[:, column] += np.bincount(
            clusters,
            weights=multipliers * design[row_groups, column],
            minlength=output.shape[0],
        )


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


def continuous_single_block_moments(
    completion_entities: np.ndarray,
    completion_times: np.ndarray,
    completion_spans: np.ndarray,
    candidate_starts: np.ndarray,
    candidate_ends: np.ndarray,
    candidate_windows: np.ndarray,
    entity_ends: np.ndarray,
    grid_offsets: np.ndarray,
    row_times: np.ndarray,
    knot_edges: np.ndarray,
    knot_scales: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    group_by_row: np.ndarray,
    current_x: np.ndarray,
    current_columns: np.ndarray | None = None,
    *,
    candidate_minimum_spans: np.ndarray | None = None,
    workers: int = 0,
    gradient_only: bool = False,
    prefix_first: np.ndarray | None = None,
    prefix_second: np.ndarray | None = None,
    group_run_starts: np.ndarray | None = None,
    group_run_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Exact continuous moments without expanding completion intervals.

    The compiled sweep integrates the same piecewise-constant M-knot columns
    used by ``ResponseEngine``.  Only their representation changes: interval
    endpoints and derivative prefix sums replace candidate-by-row matrices.
    Unsupported source installations fail open to the existing exact path.
    """
    if _cpu_native is None or not hasattr(
        _cpu_native, "continuous_single_block_moments"
    ):
        return None
    completion_entities = np.ascontiguousarray(completion_entities, dtype=np.int32)
    completion_times = np.ascontiguousarray(completion_times, dtype=np.int64)
    completion_spans = np.ascontiguousarray(completion_spans, dtype=np.int64)
    candidate_starts = np.ascontiguousarray(candidate_starts, dtype=np.int64)
    candidate_ends = np.ascontiguousarray(candidate_ends, dtype=np.int64)
    candidate_windows = np.ascontiguousarray(candidate_windows, dtype=np.int64)
    minimum_spans = (
        np.full(len(candidate_windows), -1, dtype=np.int64)
        if candidate_minimum_spans is None
        else np.ascontiguousarray(candidate_minimum_spans, dtype=np.int64)
    )
    entity_ends = np.ascontiguousarray(entity_ends, dtype=np.int64)
    grid_offsets = np.ascontiguousarray(grid_offsets, dtype=np.int64)
    row_times = np.ascontiguousarray(row_times, dtype=np.int64)
    knot_edges = np.ascontiguousarray(knot_edges, dtype=np.int64)
    knot_scales = np.ascontiguousarray(knot_scales, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    group_by_row = np.ascontiguousarray(group_by_row, dtype=np.int32)
    current_x = np.ascontiguousarray(current_x, dtype=np.float64)
    columns = (
        np.arange(current_x.shape[1], dtype=np.int32)
        if current_columns is None
        else np.ascontiguousarray(current_columns, dtype=np.int32)
    )
    if (
        completion_times.shape != completion_entities.shape
        or completion_spans.shape != completion_entities.shape
        or candidate_ends.shape != candidate_starts.shape
        or candidate_windows.shape != candidate_starts.shape
        or minimum_spans.shape != candidate_starts.shape
        or np.any(minimum_spans >= candidate_windows)
        or first.shape != row_times.shape
        or second.shape != row_times.shape
        or group_by_row.shape != row_times.shape
        or knot_edges.shape != (len(knot_scales) + 1,)
        or current_x.ndim != 2
        or np.any(columns < 0)
        or np.any(columns >= current_x.shape[1])
    ):
        raise ValueError("continuous moment buffers do not align")
    if prefix_first is None or prefix_second is None:
        prefix_first = np.empty(len(first) + 1, dtype=np.float64)
        prefix_second = np.empty(len(second) + 1, dtype=np.float64)
        prefix_first[0] = 0.0
        prefix_second[0] = 0.0
        np.cumsum(first, out=prefix_first[1:])
        np.cumsum(second, out=prefix_second[1:])
    else:
        prefix_first = np.ascontiguousarray(prefix_first, dtype=np.float64)
        prefix_second = np.ascontiguousarray(prefix_second, dtype=np.float64)
        if prefix_first.shape != (len(first) + 1,) or prefix_second.shape != (
            len(second) + 1,
        ):
            raise ValueError("continuous derivative prefixes do not align")
    if group_run_starts is None or group_run_ids is None:
        changes = np.empty(len(group_by_row), dtype=np.bool_)
        changes[0] = True
        changes[1:] = group_by_row[1:] != group_by_row[:-1]
        run_starts = np.ascontiguousarray(np.flatnonzero(changes), dtype=np.int64)
        run_ids = np.ascontiguousarray(group_by_row[run_starts], dtype=np.int32)
    else:
        run_starts = np.ascontiguousarray(group_run_starts, dtype=np.int64)
        run_ids = np.ascontiguousarray(group_run_ids, dtype=np.int32)
        if run_starts.shape != run_ids.shape:
            raise ValueError("continuous group runs do not align")
    candidate_count = len(candidate_starts)
    knot_count = len(knot_scales)
    gradient = np.empty((candidate_count, knot_count), dtype=np.float64)
    hessian = np.empty(
        (candidate_count, 0 if gradient_only else knot_count,
         0 if gradient_only else knot_count),
        dtype=np.float64,
    )
    # The native sweep always computes Fisher moments; retain a private output
    # for gradient-only callers so the public return contract remains intact.
    native_hessian = (
        np.empty((candidate_count, knot_count, knot_count), dtype=np.float64)
        if gradient_only
        else hessian
    )
    cross = np.empty((candidate_count, len(columns), knot_count), dtype=np.float64)
    _cpu_native.continuous_single_block_moments(
        completion_entities,
        completion_times,
        completion_spans,
        candidate_starts,
        candidate_ends,
        candidate_windows,
        minimum_spans,
        entity_ends,
        grid_offsets,
        row_times,
        knot_edges,
        knot_scales,
        prefix_first,
        prefix_second,
        run_starts,
        run_ids,
        current_x,
        columns,
        gradient,
        native_hessian,
        cross,
        int(workers),
        int(gradient_only),
    )
    if gradient_only:
        cross = np.empty((candidate_count, 0, knot_count), dtype=np.float64)
    return gradient, hessian, cross


def continuous_single_block_profiles(
    completion_entities: np.ndarray,
    completion_times: np.ndarray,
    completion_spans: np.ndarray,
    candidate_starts: np.ndarray,
    candidate_ends: np.ndarray,
    candidate_windows: np.ndarray,
    entity_ends: np.ndarray,
    grid_offsets: np.ndarray,
    row_times: np.ndarray,
    knot_edges: np.ndarray,
    knot_scales: np.ndarray,
    coefficients: np.ndarray,
    *,
    workers: int = 0,
) -> np.ndarray | None:
    """Evaluate fitted continuous rule effects directly on risk intervals."""
    if _cpu_native is None or not hasattr(
        _cpu_native, "continuous_single_block_profiles"
    ):
        return None
    completion_entities = np.ascontiguousarray(completion_entities, dtype=np.int32)
    completion_times = np.ascontiguousarray(completion_times, dtype=np.int64)
    completion_spans = np.ascontiguousarray(completion_spans, dtype=np.int64)
    candidate_starts = np.ascontiguousarray(candidate_starts, dtype=np.int64)
    candidate_ends = np.ascontiguousarray(candidate_ends, dtype=np.int64)
    candidate_windows = np.ascontiguousarray(candidate_windows, dtype=np.int64)
    entity_ends = np.ascontiguousarray(entity_ends, dtype=np.int64)
    grid_offsets = np.ascontiguousarray(grid_offsets, dtype=np.int64)
    row_times = np.ascontiguousarray(row_times, dtype=np.int64)
    knot_edges = np.ascontiguousarray(knot_edges, dtype=np.int64)
    knot_scales = np.ascontiguousarray(knot_scales, dtype=np.float64)
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
    candidates = len(candidate_starts)
    if (
        completion_times.shape != completion_entities.shape
        or completion_spans.shape != completion_entities.shape
        or candidate_ends.shape != candidate_starts.shape
        or candidate_windows.shape != candidate_starts.shape
        or knot_edges.shape != (len(knot_scales) + 1,)
        or coefficients.shape != (candidates, len(knot_scales))
        or grid_offsets.shape != (len(entity_ends) + 1,)
        or int(grid_offsets[-1]) != len(row_times)
    ):
        raise ValueError("continuous profile buffers do not align")
    output = np.empty((candidates, len(row_times)), dtype=np.float64)
    _cpu_native.continuous_single_block_profiles(
        completion_entities,
        completion_times,
        completion_spans,
        candidate_starts,
        candidate_ends,
        candidate_windows,
        entity_ends,
        grid_offsets,
        row_times,
        knot_edges,
        knot_scales,
        coefficients,
        output,
        int(workers),
    )
    return output




def continuous_additive_support_profiles(
    completion_entities: np.ndarray,
    completion_times: np.ndarray,
    completion_spans: np.ndarray,
    component_starts: np.ndarray,
    component_ends: np.ndarray,
    component_windows: np.ndarray,
    support_offsets: np.ndarray,
    entity_ends: np.ndarray,
    grid_offsets: np.ndarray,
    row_times: np.ndarray,
    knot_edges: np.ndarray,
    knot_scales: np.ndarray,
    coefficients: np.ndarray,
    *,
    workers: int = 0,
) -> np.ndarray | None:
    """Evaluate additive support effects without component-sized dense slabs."""
    if _cpu_native is None or not hasattr(
        _cpu_native, "continuous_additive_support_profiles"
    ):
        return None
    completion_entities = np.ascontiguousarray(completion_entities, dtype=np.int32)
    completion_times = np.ascontiguousarray(completion_times, dtype=np.int64)
    completion_spans = np.ascontiguousarray(completion_spans, dtype=np.int64)
    component_starts = np.ascontiguousarray(component_starts, dtype=np.int64)
    component_ends = np.ascontiguousarray(component_ends, dtype=np.int64)
    component_windows = np.ascontiguousarray(component_windows, dtype=np.int64)
    support_offsets = np.ascontiguousarray(support_offsets, dtype=np.int64)
    entity_ends = np.ascontiguousarray(entity_ends, dtype=np.int64)
    grid_offsets = np.ascontiguousarray(grid_offsets, dtype=np.int64)
    row_times = np.ascontiguousarray(row_times, dtype=np.int64)
    knot_edges = np.ascontiguousarray(knot_edges, dtype=np.int64)
    knot_scales = np.ascontiguousarray(knot_scales, dtype=np.float64)
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
    components = len(component_starts)
    supports = len(support_offsets) - 1
    if (
        supports < 0
        or completion_times.shape != completion_entities.shape
        or completion_spans.shape != completion_entities.shape
        or component_ends.shape != component_starts.shape
        or component_windows.shape != component_starts.shape
        or knot_edges.shape != (len(knot_scales) + 1,)
        or coefficients.shape != (components, len(knot_scales))
        or grid_offsets.shape != (len(entity_ends) + 1,)
        or int(grid_offsets[-1]) != len(row_times)
        or support_offsets.shape != (supports + 1,)
        or int(support_offsets[0]) != 0
        or int(support_offsets[-1]) != components
        or np.any(support_offsets[1:] < support_offsets[:-1])
    ):
        raise ValueError("continuous additive profile buffers do not align")
    output = np.empty((supports, len(row_times)), dtype=np.float64)
    _cpu_native.continuous_additive_support_profiles(
        completion_entities,
        completion_times,
        completion_spans,
        component_starts,
        component_ends,
        component_windows,
        support_offsets,
        entity_ends,
        grid_offsets,
        row_times,
        knot_edges,
        knot_scales,
        coefficients,
        output,
        int(workers),
    )
    return output

def continuous_single_block_profile_distances(
    completion_entities: np.ndarray,
    completion_times: np.ndarray,
    completion_spans: np.ndarray,
    candidate_starts: np.ndarray,
    candidate_ends: np.ndarray,
    candidate_windows: np.ndarray,
    entity_ends: np.ndarray,
    grid_offsets: np.ndarray,
    row_times: np.ndarray,
    knot_edges: np.ndarray,
    knot_scales: np.ndarray,
    coefficients: np.ndarray,
    sqrt_fisher: np.ndarray,
    current_profile: tuple[np.ndarray, np.ndarray],
    right_profiles: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    tolerance: float,
    workers: int = 0,
) -> np.ndarray | None:
    """Evaluate exact continuous profiles and Fisher distances in one pass.

    Only the final candidate-by-reference distances are retained.  Dense
    current/reference workspaces are built once, while each native worker owns
    one row vector.  This preserves the scalar profile threshold and ascending
    squared-distance accumulation without allocating a candidates-by-rows
    slab.
    """
    if _cpu_native is None or not hasattr(
        _cpu_native, "continuous_single_block_profile_distances"
    ):
        return None
    completion_entities = np.ascontiguousarray(completion_entities, dtype=np.int32)
    completion_times = np.ascontiguousarray(completion_times, dtype=np.int64)
    completion_spans = np.ascontiguousarray(completion_spans, dtype=np.int64)
    candidate_starts = np.ascontiguousarray(candidate_starts, dtype=np.int64)
    candidate_ends = np.ascontiguousarray(candidate_ends, dtype=np.int64)
    candidate_windows = np.ascontiguousarray(candidate_windows, dtype=np.int64)
    entity_ends = np.ascontiguousarray(entity_ends, dtype=np.int64)
    grid_offsets = np.ascontiguousarray(grid_offsets, dtype=np.int64)
    row_times = np.ascontiguousarray(row_times, dtype=np.int64)
    knot_edges = np.ascontiguousarray(knot_edges, dtype=np.int64)
    knot_scales = np.ascontiguousarray(knot_scales, dtype=np.float64)
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
    sqrt_fisher = np.ascontiguousarray(sqrt_fisher, dtype=np.float64)
    candidates = len(candidate_starts)
    row_count = len(row_times)
    right = tuple(
        (
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(values, dtype=np.float64),
        )
        for rows, values in right_profiles
    )
    current_rows = np.ascontiguousarray(current_profile[0], dtype=np.int64)
    current_values = np.ascontiguousarray(current_profile[1], dtype=np.float64)
    invalid = (
        completion_times.shape != completion_entities.shape
        or completion_spans.shape != completion_entities.shape
        or candidate_ends.shape != candidate_starts.shape
        or candidate_windows.shape != candidate_starts.shape
        or knot_edges.shape != (len(knot_scales) + 1,)
        or coefficients.shape != (candidates, len(knot_scales))
        or grid_offsets.shape != (len(entity_ends) + 1,)
        or int(grid_offsets[-1]) != row_count
        or sqrt_fisher.shape != (row_count,)
        or current_rows.shape != current_values.shape
        or not right
        or not np.array_equal(right[0][0], current_rows)
        or not np.array_equal(right[0][1], current_values)
        or any(rows.shape != values.shape for rows, values in right)
        or np.any(current_rows < 0)
        or np.any(current_rows >= row_count)
        or any(np.any(rows < 0) or np.any(rows >= row_count) for rows, _ in right)
    )
    if invalid:
        # A resumed search can restore a valid sparse reference family whose
        # cached row basis is not the fused evaluator's current-first layout.
        # The public contract for unsupported layouts is fail-open: the caller
        # evaluates the same profiles with the canonical scalar exact path.
        # Raising here used to abort an otherwise valid deterministic resume.
        return None
    # This workspace is bounded by the already frozen reference family, not by
    # the candidate dictionary.  Refuse oversized allocations and retain the
    # original exact fallback.
    dense_bytes = row_count * max(1, len(right)) * 8
    if dense_bytes > 3 * 1024**3:
        return None
    current = np.zeros(row_count, dtype=np.float64)
    current[current_rows] = current_values
    references = np.zeros((row_count, len(right)), dtype=np.float64)
    for index, (rows, values) in enumerate(right):
        references[rows, index] = values
    output = np.empty((candidates, 1), dtype=np.float64)
    _cpu_native.continuous_single_block_profile_distances(
        completion_entities,
        completion_times,
        completion_spans,
        candidate_starts,
        candidate_ends,
        candidate_windows,
        entity_ends,
        grid_offsets,
        row_times,
        knot_edges,
        knot_scales,
        coefficients,
        sqrt_fisher,
        current,
        references,
        float(tolerance),
        output,
        int(workers),
    )
    return output


def dense_increment_sparse_distances(
    dense_increments: np.ndarray,
    current_profile: tuple[np.ndarray, np.ndarray],
    right_profiles: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    tolerance: float,
    workers: int = 0,
) -> np.ndarray | None:
    """Merge dense increments with one sparse profile and batch distances.

    The native operator applies the same finite/tolerance mask, sorted sparse
    sum, and squared-distance reduction as the scalar reference.  It avoids
    materializing one multi-million-row union in Python for every candidate.
    """
    if _cpu_native is None or not hasattr(
        _cpu_native, "dense_increment_sparse_distances"
    ):
        return None
    dense = np.ascontiguousarray(dense_increments, dtype=np.float64)
    current_rows = np.ascontiguousarray(current_profile[0], dtype=np.int64)
    current_values = np.ascontiguousarray(current_profile[1], dtype=np.float64)
    right = tuple(
        (
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(values, dtype=np.float64),
        )
        for rows, values in right_profiles
    )
    if dense.ndim != 2 or current_rows.shape != current_values.shape:
        raise ValueError("dense increment and current profile do not align")
    output = np.empty((len(dense), len(right)), dtype=np.float64)
    _cpu_native.dense_increment_sparse_distances(
        dense,
        current_rows,
        current_values,
        tuple(item[0] for item in right),
        tuple(item[1] for item in right),
        float(tolerance),
        output,
        int(workers),
    )
    return output

def sparse_squared_distances(
    left_profiles: tuple[tuple[np.ndarray, np.ndarray], ...],
    right_profiles: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    workers: int = 0,
) -> np.ndarray | None:
    """Return all squared Euclidean distances without dense union arrays."""
    if _cpu_native is None or not hasattr(
        _cpu_native, "sparse_squared_distances"
    ):
        return None
    left = tuple(
        (
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(values, dtype=np.float64),
        )
        for rows, values in left_profiles
    )
    right = tuple(
        (
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(values, dtype=np.float64),
        )
        for rows, values in right_profiles
    )
    output = np.empty((len(left), len(right)), dtype=np.float64)
    _cpu_native.sparse_squared_distances(
        tuple(item[0] for item in left),
        tuple(item[1] for item in left),
        tuple(item[0] for item in right),
        tuple(item[1] for item in right),
        output,
        int(workers),
    )
    return output


def sparse_moments_batch(
    candidates: list[tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]],
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
    if block_count < 1 or any(
        len(candidate) != block_count for candidate in candidates
    ):
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
    hessian = np.empty((len(candidates), dimensions, dimensions), dtype=np.float64)
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


def sparse_moments_indexed_batch(
    candidates: list[tuple[tuple[np.ndarray, np.ndarray], ...]],
    first: np.ndarray,
    second: np.ndarray,
    *,
    device: str,
    derivative_token: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Exact ragged moments with one shared derivative grid.

    Sparse blocks carry only global row indices and M-knot values.  The CUDA
    kernels index the shared first/second derivative arrays directly, avoiding
    candidate/block-wise derivative gathers and transfers.  Reduction order is
    identical to :func:`sparse_moments_batch`.
    """
    library = _cuda_library()
    if (
        not candidates
        or not device.startswith("cuda")
        or library is None
        or not hasattr(library, "crbstpp_cuda_sparse_moments_indexed_batch")
    ):
        return None
    block_count = len(candidates[0])
    if block_count < 1 or any(
        len(candidate) != block_count for candidate in candidates
    ):
        raise ValueError("ragged candidates must have one common block count")
    knot_count = candidates[0][0][1].shape[1]
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("shared derivative grid shape mismatch")
    rows_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    offsets = [0]
    for candidate in candidates:
        for rows, values in candidate:
            rows = np.ascontiguousarray(rows, dtype=np.int64)
            values = np.ascontiguousarray(values, dtype=np.float64)
            if values.shape != (len(rows), knot_count):
                raise ValueError("ragged sparse block shape mismatch")
            if len(rows) and (int(rows[0]) < 0 or int(rows[-1]) >= len(first)):
                raise ValueError("sparse row is outside the derivative grid")
            rows_parts.append(rows)
            value_parts.append(values)
            offsets.append(offsets[-1] + len(rows))
    total_rows = offsets[-1]
    if total_rows:
        rows = np.ascontiguousarray(np.concatenate(rows_parts), dtype=np.int64)
        values = np.ascontiguousarray(np.concatenate(value_parts), dtype=np.float64)
    else:
        rows = np.zeros(0, dtype=np.int64)
        values = np.zeros((0, knot_count), dtype=np.float64)
    block_offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    dimensions = block_count * knot_count
    gradient = np.empty((len(candidates), dimensions), dtype=np.float64)
    hessian = np.empty((len(candidates), dimensions, dimensions), dtype=np.float64)
    cross = np.empty_like(gradient)
    pointer = ctypes.POINTER(ctypes.c_double)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    status = library.crbstpp_cuda_sparse_moments_indexed_batch(
        index,
        rows.ctypes.data_as(int64_pointer),
        values.ctypes.data_as(pointer),
        first.ctypes.data_as(pointer),
        second.ctypes.data_as(pointer),
        block_offsets.ctypes.data_as(int64_pointer),
        total_rows,
        len(first),
        -1 if derivative_token is None else int(derivative_token),
        len(candidates),
        block_count,
        knot_count,
        gradient.ctypes.data_as(pointer),
        hessian.ctypes.data_as(pointer),
        cross.ctypes.data_as(pointer),
    )
    return (gradient, hessian, cross) if status == 0 else None


def sparse_moment_geometry(
    blocks: tuple[tuple[np.ndarray, np.ndarray], ...],
    signs: tuple[int, ...] | None = None,
) -> SparseMomentGeometry:
    """Pack one immutable sparse block design for resident Newton moments.

    Signs are applied directly into the packed destination.  Building a
    separate ``-values`` array for every inhibition block before concatenating
    them used three simultaneous copies (raw, signed, packed) of a long-horizon
    support.  ``np.negative(..., out=...)`` keeps the identical float64 values
    while bounding peak storage to raw plus packed geometry.
    """
    if not blocks:
        raise ValueError("sparse model geometry requires at least one block")
    block_signs = (1,) * len(blocks) if signs is None else tuple(signs)
    if len(block_signs) != len(blocks) or any(sign not in {-1, 1} for sign in block_signs):
        raise ValueError("sparse model geometry requires aligned signs")
    knot_count = int(np.asarray(blocks[0][1]).shape[1])
    rows_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    offsets = [0]
    for raw_rows, raw_values in blocks:
        rows = np.ascontiguousarray(raw_rows, dtype=np.int64)
        values = np.ascontiguousarray(raw_values, dtype=np.float64)
        if values.shape != (len(rows), knot_count):
            raise ValueError("sparse model block shape mismatch")
        if len(rows) > 1 and np.any(rows[1:] <= rows[:-1]):
            raise ValueError("sparse model rows must be sorted and unique")
        rows_parts.append(rows)
        value_parts.append(values)
        offsets.append(offsets[-1] + len(rows))
    if offsets[-1]:
        rows = np.empty(offsets[-1], dtype=np.int64)
        values = np.empty((offsets[-1], knot_count), dtype=np.float64)
        for index, (source_rows, source_values, sign) in enumerate(
            zip(rows_parts, value_parts, block_signs, strict=True)
        ):
            left = offsets[index]
            right = offsets[index + 1]
            rows[left:right] = source_rows
            if sign > 0:
                values[left:right] = source_values
            else:
                np.negative(source_values, out=values[left:right])
    else:
        rows = np.zeros(0, dtype=np.int64)
        values = np.zeros((0, knot_count), dtype=np.float64)
    return SparseMomentGeometry(
        rows=rows,
        values=values,
        block_offsets=np.ascontiguousarray(offsets, dtype=np.int64),
        block_count=len(blocks),
        knot_count=knot_count,
        token=next(_GEOMETRY_TOKENS),
    )


def sparse_model_moments(
    geometry: SparseMomentGeometry,
    first: np.ndarray,
    second: np.ndarray,
    *,
    device: str,
    derivative_token: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Evaluate one exact sparse design while retaining its geometry on CUDA."""
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("sparse model derivative grid shape mismatch")
    if len(geometry.rows) and (
        int(np.min(geometry.rows)) < 0 or int(np.max(geometry.rows)) >= len(first)
    ):
        raise ValueError("sparse model row is outside the derivative grid")
    if device == "cpu":
        # Exact CPU fallback.  Each block stores sorted unique row indices.
        # Pairwise intersections avoid constructing an active_rows-by-p
        # dense design; support dimensions are small, so this is both bounded
        # in memory and faster than copying a full sufficient-statistic matrix.
        block_count = geometry.block_count
        knot_count = geometry.knot_count
        dimension = block_count * knot_count
        gradient = np.zeros(dimension, dtype=np.float64)
        hessian = np.zeros((dimension, dimension), dtype=np.float64)
        cross = np.zeros(dimension, dtype=np.float64)
        block_rows: list[np.ndarray] = []
        block_values: list[np.ndarray] = []
        for block_index in range(block_count):
            left = int(geometry.block_offsets[block_index])
            right = int(geometry.block_offsets[block_index + 1])
            rows = geometry.rows[left:right]
            values = geometry.values[left:right]
            block_rows.append(rows)
            block_values.append(values)
            target = slice(
                block_index * knot_count, (block_index + 1) * knot_count
            )
            if not len(rows):
                continue
            gradient[target] = values.T @ first[rows]
            cross[target] = values.T @ second[rows]
            hessian[target, target] = values.T @ (second[rows, None] * values)
        for left_index in range(block_count):
            left_rows = block_rows[left_index]
            if not len(left_rows):
                continue
            left_values = block_values[left_index]
            left_target = slice(
                left_index * knot_count, (left_index + 1) * knot_count
            )
            for right_index in range(left_index + 1, block_count):
                right_rows = block_rows[right_index]
                if not len(right_rows):
                    continue
                common, left_positions, right_positions = np.intersect1d(
                    left_rows,
                    right_rows,
                    assume_unique=True,
                    return_indices=True,
                )
                if not len(common):
                    continue
                right_values = block_values[right_index]
                right_target = slice(
                    right_index * knot_count, (right_index + 1) * knot_count
                )
                cross_hessian = left_values[left_positions].T @ (
                    second[common, None] * right_values[right_positions]
                )
                hessian[left_target, right_target] = cross_hessian
                hessian[right_target, left_target] = cross_hessian.T
        return gradient, hessian, cross
    library = _cuda_library()
    function = (
        None
        if library is None
        else getattr(
            library,
            "crbstpp_cuda_sparse_moments_indexed_resident",
            None,
        )
    )
    if not device.startswith("cuda") or function is None:
        return None
    dimension = geometry.block_count * geometry.knot_count
    gradient = np.empty((1, dimension), dtype=np.float64)
    hessian = np.empty((1, dimension, dimension), dtype=np.float64)
    cross = np.empty((1, dimension), dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    device_index = int(device.split(":", 1)[1]) if ":" in device else 0
    status = function(
        device_index,
        geometry.rows.ctypes.data_as(int64_pointer),
        geometry.values.ctypes.data_as(pointer),
        first.ctypes.data_as(pointer),
        second.ctypes.data_as(pointer),
        geometry.block_offsets.ctypes.data_as(int64_pointer),
        len(geometry.rows),
        len(first),
        int(derivative_token),
        int(geometry.token),
        1,
        geometry.block_count,
        geometry.knot_count,
        gradient.ctypes.data_as(pointer),
        hessian.ctypes.data_as(pointer),
        cross.ctypes.data_as(pointer),
    )
    return (
        (gradient[0], hessian[0], cross[0])
        if status == 0
        else None
    )


def implicit_moments_batch(
    source_offsets: np.ndarray,
    source_times: np.ndarray,
    source_spans: np.ndarray | None,
    starts: np.ndarray,
    ends: np.ndarray,
    grid_offsets: np.ndarray,
    basis: np.ndarray,
    block_predicates: np.ndarray,
    block_orders: np.ndarray,
    block_windows: np.ndarray,
    block_counts: np.ndarray,
    candidate_entity_offsets: np.ndarray,
    candidate_entities: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    group_by_row: np.ndarray,
    current_x: np.ndarray,
    *,
    source_token: int,
    derivative_token: int,
    device: str,
    compact_poisson_events: np.ndarray | None = None,
    compact_cloglog_event_deltas: tuple[np.ndarray, np.ndarray] | None = None,
    completion_mode: bool | int = False,
    block_minimum_spans: np.ndarray | None = None,
    current_columns: np.ndarray | None = None,
    validated_source_offsets: bool = False,
    gradient_only: bool = False,
    baseline_group_by_row: np.ndarray | None = None,
    baseline_group_count: int | None = None,
    baseline_dimension: int | None = None,
    current_signed_state: np.ndarray | None = None,
    collect_footprint_stats: bool = False,
    relaxation_group_by_current_group: np.ndarray | None = None,
    relaxation_group_count: int | None = None,
    collect_group_footprint_stats: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | None
):
    """Exact completion moments without a global expanded response matrix.

    Immutable time/span values and likelihood derivatives remain resident on
    one GPU.  ``source_offsets`` is wave-local metadata and is uploaded on
    every call; it may select disjoint absolute slices of the immutable value
    vectors under the same ``source_token``.  Candidate response blocks are
    generated in bounded entity tiles, consumed by
    gradient/Fisher/current-cross reductions, and immediately reused.
    Unsupported installations fail open to the sparse reference path.
    """
    library = _cuda_library()
    if (
        not device.startswith("cuda")
        or library is None
        or not hasattr(library, "crbstpp_cuda_implicit_moments_batch")
    ):
        return None
    source_offsets = np.ascontiguousarray(source_offsets, dtype=np.int64)
    source_times = np.ascontiguousarray(source_times, dtype=np.int64)
    spans = (
        np.ascontiguousarray(source_spans, dtype=np.int64)
        if source_spans is not None
        else np.zeros(0, dtype=np.int64)
    )
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    grid_offsets = np.ascontiguousarray(grid_offsets, dtype=np.int64)
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    block_predicates = np.ascontiguousarray(block_predicates, dtype=np.int32)
    block_orders = np.ascontiguousarray(block_orders, dtype=np.int32)
    block_windows = np.ascontiguousarray(block_windows, dtype=np.int64)
    minimum_spans = (
        np.full(block_windows.shape, -1, dtype=np.int64)
        if block_minimum_spans is None
        else np.ascontiguousarray(block_minimum_spans, dtype=np.int64)
    )
    block_counts = np.ascontiguousarray(block_counts, dtype=np.int32)
    candidate_entity_offsets = np.ascontiguousarray(
        candidate_entity_offsets, dtype=np.int64
    )
    candidate_entities = np.ascontiguousarray(candidate_entities, dtype=np.int32)
    first = np.ascontiguousarray(first, dtype=np.float64)
    second = np.ascontiguousarray(second, dtype=np.float64)
    group_by_row = np.ascontiguousarray(group_by_row, dtype=np.int32)
    current_x = np.ascontiguousarray(current_x, dtype=np.float64)
    baseline_groups = (
        np.ascontiguousarray(baseline_group_by_row, dtype=np.int32)
        if baseline_group_by_row is not None
        else np.zeros(0, dtype=np.int32)
    )
    baseline_count = int(baseline_group_count or 0)
    effective_baseline_dimension = int(baseline_dimension or 0)
    signed_state = (
        np.ascontiguousarray(current_signed_state, dtype=np.uint8)
        if current_signed_state is not None
        else np.zeros(0, dtype=np.uint8)
    )
    relaxation_groups = (
        np.ascontiguousarray(relaxation_group_by_current_group, dtype=np.int32)
        if relaxation_group_by_current_group is not None
        else np.zeros(0, dtype=np.int32)
    )
    relaxation_count = int(relaxation_group_count or 0)
    selected_current_columns = np.ascontiguousarray(
        (
            np.zeros(0, dtype=np.int32)
            if gradient_only
            else (
                np.arange(current_x.shape[1], dtype=np.int32)
                if current_columns is None
                else current_columns
            )
        ),
        dtype=np.int32,
    )
    compact_cloglog = compact_cloglog_event_deltas is not None
    compact_poisson = compact_poisson_events is not None and not compact_cloglog
    compact_mode = 2 if compact_cloglog else (1 if compact_poisson else 0)
    event_by_row = (
        np.ascontiguousarray(compact_poisson_events, dtype=np.uint8)
        if compact_mode
        else np.zeros(0, dtype=np.uint8)
    )
    event_first_delta = (
        np.ascontiguousarray(compact_cloglog_event_deltas[0], dtype=np.float64)
        if compact_cloglog
        else np.zeros(0, dtype=np.float64)
    )
    event_second_delta = (
        np.ascontiguousarray(compact_cloglog_event_deltas[1], dtype=np.float64)
        if compact_cloglog
        else np.zeros(0, dtype=np.float64)
    )
    if starts.ndim != 1 or ends.shape != starts.shape:
        raise ValueError("implicit context entity shape mismatch")
    entity_count = len(starts)
    if grid_offsets.shape != (entity_count + 1,):
        raise ValueError("implicit context offset shape mismatch")
    completion_mode_value = int(completion_mode)
    if completion_mode_value not in (0, 1, 2):
        raise ValueError("implicit completion mode must be 0, 1, or 2")
    expected_offset_width = 2 if completion_mode_value == 2 else entity_count + 1
    if source_offsets.ndim != 2 or source_offsets.shape[1] != expected_offset_width:
        raise ValueError("implicit source offset shape mismatch")
    if basis.ndim != 2 or basis.shape[1] < 1:
        raise ValueError("implicit kernel basis shape mismatch")
    if (
        block_predicates.ndim != 3
        or block_predicates.shape[2] != 3
        or block_orders.shape != block_predicates.shape[:2]
        or block_windows.shape != block_predicates.shape[:2]
        or minimum_spans.shape != block_predicates.shape[:2]
        or block_counts.shape != (block_predicates.shape[0],)
    ):
        raise ValueError("implicit candidate metadata shape mismatch")
    candidates, maximum_blocks = block_predicates.shape[:2]
    if candidates < 1 or maximum_blocks < 1:
        raise ValueError("implicit candidate batch cannot be empty")
    if (
        candidate_entity_offsets.shape != (candidates + 1,)
        or int(candidate_entity_offsets[0]) != 0
        or int(candidate_entity_offsets[-1]) != len(candidate_entities)
        or np.any(candidate_entity_offsets[1:] < candidate_entity_offsets[:-1])
    ):
        raise ValueError("implicit candidate entity index is invalid")
    if len(candidate_entities) and (
        int(candidate_entities.min()) < 0
        or int(candidate_entities.max()) >= entity_count
    ):
        raise ValueError("implicit candidate entity is out of range")
    if np.any(block_counts < 1) or np.any(block_counts > maximum_blocks):
        raise ValueError("implicit candidate block count is invalid")
    rows = int(grid_offsets[-1])
    expected_derivatives = current_x.shape[0] if compact_mode else rows
    if first.shape != (expected_derivatives,) or second.shape != first.shape:
        raise ValueError("implicit derivative shape mismatch")
    if group_by_row.shape != (rows,):
        raise ValueError("implicit current-design group map shape mismatch")
    if collect_footprint_stats and (
        baseline_groups.shape != (rows,)
        or signed_state.shape != (current_x.shape[0],)
        or baseline_count < 1
        or effective_baseline_dimension < 1
        or effective_baseline_dimension > current_x.shape[1]
        or compact_mode == 0
        or (len(baseline_groups) and (
            int(baseline_groups.min()) < 0
            or int(baseline_groups.max()) >= baseline_count
        ))
    ):
        raise ValueError("implicit footprint-statistic metadata is invalid")
    if collect_group_footprint_stats and (
        relaxation_groups.shape != (current_x.shape[0],)
        or relaxation_count < 1
        or (
            len(relaxation_groups)
            and (
                int(relaxation_groups.min()) < -1
                or int(relaxation_groups.max()) >= relaxation_count
            )
        )
        or compact_mode == 0
    ):
        raise ValueError("implicit group-footprint metadata is invalid")
    if compact_mode and event_by_row.shape != (rows,):
        raise ValueError("compact event counts must cover the grid")
    if compact_cloglog and (
        event_first_delta.shape != (current_x.shape[0],)
        or event_second_delta.shape != event_first_delta.shape
    ):
        raise ValueError(
            "compact cloglog event corrections must align with design groups"
        )
    if current_x.ndim != 2 or current_x.shape[1] < 1:
        raise ValueError("implicit current design shape mismatch")
    if (
        selected_current_columns.ndim != 1
        or (not gradient_only and not len(selected_current_columns))
        or np.any(selected_current_columns < 0)
        or np.any(selected_current_columns >= current_x.shape[1])
        or len(np.unique(selected_current_columns))
        != len(selected_current_columns)
    ):
        raise ValueError("implicit current columns are invalid")
    with _VALIDATION_LOCK:
        source_validated = int(source_token) in _VALIDATED_IMPLICIT_SOURCES
        derivative_validated = (
            int(derivative_token) in _VALIDATED_IMPLICIT_DERIVATIVES
        )
    if not derivative_validated:
        if len(group_by_row) and (
            int(group_by_row.min()) < 0
            or int(group_by_row.max()) >= current_x.shape[0]
        ):
            raise ValueError("implicit current-design group is out of range")
        with _VALIDATION_LOCK:
            _VALIDATED_IMPLICIT_DERIVATIVES.add(int(derivative_token))
    # Offsets are wave-local even when immutable time/span values reuse one
    # resident token.  Validate every wave; no contiguity across predicate
    # rows is required because each row may select a disjoint global slice.
    if not validated_source_offsets:
        if (
            np.any(source_offsets[:, 1:] < source_offsets[:, :-1])
            or int(source_offsets.min(initial=0)) < 0
            or int(source_offsets.max(initial=0)) > len(source_times)
        ):
            raise ValueError("implicit source offsets are outside immutable values")
    if completion_mode_value and spans.shape != source_times.shape:
        raise ValueError("completion spans must align with completion times")
    if not source_validated:
        with _VALIDATION_LOCK:
            _VALIDATED_IMPLICIT_SOURCES.add(int(source_token))
    active_metadata = np.arange(maximum_blocks)[None, :] < block_counts[:, None]
    if np.any(block_orders[active_metadata] < 1) or np.any(
        block_orders[active_metadata] > 3
    ):
        raise ValueError("implicit antecedent order must lie in [1, 3]")
    if np.any(block_windows[active_metadata] < 0):
        raise ValueError("implicit formation windows must be nonnegative")
    if np.any(minimum_spans[active_metadata] >= block_windows[active_metadata]):
        raise ValueError("implicit formation-band lower spans must be smaller")
    active_predicates = block_predicates[active_metadata]
    active_orders = block_orders[active_metadata]
    for row, order in zip(active_predicates, active_orders, strict=True):
        if np.any(row[:order] < 0) or np.any(row[:order] >= source_offsets.shape[0]):
            raise ValueError("implicit predicate index is out of range")
    dimensions = maximum_blocks * basis.shape[0]
    gradient = np.empty((candidates, dimensions), dtype=np.float64)
    hessian = np.empty(
        (candidates, 0, 0) if gradient_only else (candidates, dimensions, dimensions),
        dtype=np.float64,
    )
    cross = np.empty(
        (candidates, 0, dimensions)
        if gradient_only
        else (candidates, len(selected_current_columns), dimensions),
        dtype=np.float64,
    )
    footprint_stats = np.empty(
        (candidates, 4, baseline_count, 2)
        if collect_footprint_stats
        else (0, 4, 0, 2),
        dtype=np.float64,
    )
    group_footprint_stats = np.empty(
        (candidates, relaxation_count, 2)
        if collect_group_footprint_stats
        else (0, 0, 2),
        dtype=np.float64,
    )
    pointer = ctypes.POINTER(ctypes.c_double)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    uint8_pointer = ctypes.POINTER(ctypes.c_uint8)
    device_index = int(device.split(":", 1)[1]) if ":" in device else 0
    status = library.crbstpp_cuda_implicit_moments_batch(
        device_index,
        source_offsets.ctypes.data_as(int64_pointer),
        source_times.ctypes.data_as(int64_pointer),
        (
            spans.ctypes.data_as(int64_pointer)
            if completion_mode
            else ctypes.cast(0, int64_pointer)
        ),
        completion_mode_value,
        starts.ctypes.data_as(int64_pointer),
        ends.ctypes.data_as(int64_pointer),
        grid_offsets.ctypes.data_as(int64_pointer),
        basis.ctypes.data_as(pointer),
        int(source_token),
        source_offsets.shape[0],
        entity_count,
        len(source_times),
        basis.shape[0],
        basis.shape[1],
        block_predicates.ctypes.data_as(int_pointer),
        block_orders.ctypes.data_as(int_pointer),
        minimum_spans.ctypes.data_as(int64_pointer),
        block_windows.ctypes.data_as(int64_pointer),
        block_counts.ctypes.data_as(int_pointer),
        candidate_entity_offsets.ctypes.data_as(int64_pointer),
        candidate_entities.ctypes.data_as(int_pointer),
        candidates,
        maximum_blocks,
        first.ctypes.data_as(pointer),
        second.ctypes.data_as(pointer),
        event_by_row.ctypes.data_as(uint8_pointer),
        event_first_delta.ctypes.data_as(pointer),
        event_second_delta.ctypes.data_as(pointer),
        int(compact_mode),
        int(derivative_token),
        rows,
        group_by_row.ctypes.data_as(int_pointer),
        current_x.ctypes.data_as(pointer),
        current_x.shape[0],
        current_x.shape[1],
        (
            baseline_groups.ctypes.data_as(int_pointer)
            if collect_footprint_stats
            else ctypes.cast(0, int_pointer)
        ),
        baseline_count if collect_footprint_stats else 1,
        effective_baseline_dimension if collect_footprint_stats else 1,
        (
            signed_state.ctypes.data_as(uint8_pointer)
            if collect_footprint_stats
            else ctypes.cast(0, uint8_pointer)
        ),
        (
            relaxation_groups.ctypes.data_as(int_pointer)
            if collect_group_footprint_stats
            else ctypes.cast(0, int_pointer)
        ),
        relaxation_count if collect_group_footprint_stats else 0,
        int(collect_group_footprint_stats),
        selected_current_columns.ctypes.data_as(int_pointer),
        len(selected_current_columns),
        int(gradient_only),
        int(collect_footprint_stats),
        gradient.ctypes.data_as(pointer),
        hessian.ctypes.data_as(pointer),
        cross.ctypes.data_as(pointer),
        footprint_stats.ctypes.data_as(pointer),
        group_footprint_stats.ctypes.data_as(pointer),
    )
    if status != 0:
        return None
    if collect_footprint_stats and collect_group_footprint_stats:
        return gradient, hessian, cross, footprint_stats, group_footprint_stats
    if collect_footprint_stats:
        return gradient, hessian, cross, footprint_stats
    if collect_group_footprint_stats:
        return gradient, hessian, cross, group_footprint_stats
    return gradient, hessian, cross


def implicit_objective_batch(
    block_predicates: np.ndarray,
    block_orders: np.ndarray,
    block_windows: np.ndarray,
    block_counts: np.ndarray,
    candidate_entity_offsets: np.ndarray,
    candidate_entities: np.ndarray,
    coefficients: np.ndarray,
    group_eta: np.ndarray,
    *,
    block_minimum_spans: np.ndarray | None = None,
    likelihood: str,
    source_token: int,
    derivative_token: int,
    entity_count: int,
    current_groups: int,
    knot_count: int,
    lag: int,
    maximum_entity_rows: int,
    device: str,
) -> np.ndarray | None:
    """Evaluate exact parent-frozen NLL changes in one resident GPU pass.

    Immutable source streams, entity grids, compact likelihood derivatives and
    current design groups must already be resident from
    :func:`implicit_moments_batch` under the supplied tokens.  Unsupported or
    stale workspaces fail open to the ordinary sparse evaluator.
    """
    library = _cuda_library()
    if (
        not device.startswith("cuda")
        or library is None
        or not hasattr(library, "crbstpp_cuda_implicit_objective_batch")
    ):
        return None
    predicates = np.ascontiguousarray(block_predicates, dtype=np.int32)
    orders = np.ascontiguousarray(block_orders, dtype=np.int32)
    windows = np.ascontiguousarray(block_windows, dtype=np.int64)
    minimum_spans = (
        np.full(windows.shape, -1, dtype=np.int64)
        if block_minimum_spans is None
        else np.ascontiguousarray(block_minimum_spans, dtype=np.int64)
    )
    counts = np.ascontiguousarray(block_counts, dtype=np.int32)
    entity_offsets = np.ascontiguousarray(
        candidate_entity_offsets, dtype=np.int64
    )
    entities = np.ascontiguousarray(candidate_entities, dtype=np.int32)
    beta = np.ascontiguousarray(coefficients, dtype=np.float64)
    eta = np.ascontiguousarray(group_eta, dtype=np.float64)
    likelihood_mode = {
        "poisson": 1,
        "first_event_cloglog": 2,
    }.get(str(likelihood))
    if likelihood_mode is None:
        return None
    if (
        predicates.ndim != 3
        or predicates.shape[2] != 3
        or orders.shape != predicates.shape[:2]
        or windows.shape != predicates.shape[:2]
        or minimum_spans.shape != predicates.shape[:2]
        or counts.shape != (predicates.shape[0],)
    ):
        raise ValueError("implicit objective metadata shape mismatch")
    candidates, maximum_blocks = predicates.shape[:2]
    if candidates < 1 or maximum_blocks < 1:
        raise ValueError("implicit objective batch cannot be empty")
    if beta.shape != (candidates, maximum_blocks * int(knot_count)):
        raise ValueError("implicit objective coefficient shape mismatch")
    if int(current_groups) < 1:
        raise ValueError("implicit objective current group count is invalid")
    if likelihood_mode == 2 and eta.shape != (int(current_groups),):
        raise ValueError("cloglog group predictor shape mismatch")
    if likelihood_mode == 1 and eta.shape not in {
        (0,),
        (int(current_groups),),
    }:
        raise ValueError("Poisson group predictor shape mismatch")
    if np.any(counts < 1) or np.any(counts > maximum_blocks):
        raise ValueError("implicit objective block count is invalid")
    if (
        entity_offsets.shape != (candidates + 1,)
        or int(entity_offsets[0]) != 0
        or int(entity_offsets[-1]) != len(entities)
        or np.any(entity_offsets[1:] < entity_offsets[:-1])
        or np.any(entities < 0)
        or np.any(entities >= int(entity_count))
    ):
        raise ValueError("implicit objective active-entity index is invalid")
    active = np.arange(maximum_blocks)[None, :] < counts[:, None]
    if np.any(orders[active] < 1) or np.any(orders[active] > 3):
        raise ValueError("implicit objective antecedent order is invalid")
    if np.any(minimum_spans[active] >= windows[active]):
        raise ValueError("implicit objective band lower spans are invalid")
    output = np.empty(candidates, dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    device_index = int(device.split(":", 1)[1]) if ":" in device else 0
    status = library.crbstpp_cuda_implicit_objective_batch(
        device_index,
        int(source_token),
        int(derivative_token),
        int(entity_count),
        int(knot_count),
        int(lag),
        predicates.ctypes.data_as(int_pointer),
        orders.ctypes.data_as(int_pointer),
        minimum_spans.ctypes.data_as(int64_pointer),
        windows.ctypes.data_as(int64_pointer),
        counts.ctypes.data_as(int_pointer),
        entity_offsets.ctypes.data_as(int64_pointer),
        entities.ctypes.data_as(int_pointer),
        candidates,
        maximum_blocks,
        beta.ctypes.data_as(pointer),
        (
            eta.ctypes.data_as(pointer)
            if len(eta)
            else ctypes.cast(0, pointer)
        ),
        int(current_groups),
        int(likelihood_mode),
        int(maximum_entity_rows),
        output.ctypes.data_as(pointer),
    )
    return output if status == 0 else None


def implicit_poisson_objective_batch(
    block_predicates: np.ndarray,
    block_orders: np.ndarray,
    block_windows: np.ndarray,
    block_counts: np.ndarray,
    candidate_entity_offsets: np.ndarray,
    candidate_entities: np.ndarray,
    coefficients: np.ndarray,
    *,
    source_token: int,
    derivative_token: int,
    entity_count: int,
    current_groups: int,
    knot_count: int,
    lag: int,
    maximum_entity_rows: int,
    device: str,
) -> np.ndarray | None:
    """Backward-compatible Poisson specialization of the generic evaluator."""
    return implicit_objective_batch(
        block_predicates,
        block_orders,
        block_windows,
        block_counts,
        candidate_entity_offsets,
        candidate_entities,
        coefficients,
        np.zeros(0, dtype=np.float64),
        likelihood="poisson",
        source_token=source_token,
        derivative_token=derivative_token,
        entity_count=entity_count,
        current_groups=current_groups,
        knot_count=knot_count,
        lag=lag,
        maximum_entity_rows=maximum_entity_rows,
        device=device,
    )


def new_derivative_token() -> int:
    """Return a process-unique token for one immutable CUDA derivative grid."""
    return next(_DERIVATIVE_TOKENS)


def nonnegative_quadratic_gains(
    gradients: np.ndarray, hessians: np.ndarray
) -> np.ndarray | None:
    """Exact active-set gains for a batch of tiny nonnegative quadratics."""
    gradients = np.ascontiguousarray(gradients, dtype=np.float64)
    hessians = np.ascontiguousarray(hessians, dtype=np.float64)
    if gradients.ndim != 2 or hessians.shape != (
        gradients.shape[0],
        gradients.shape[1],
        gradients.shape[1],
    ):
        raise ValueError("quadratic gain batch shape mismatch")
    if _cpu_native is None or not hasattr(_cpu_native, "nonnegative_quadratic_gains"):
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
    *,
    worker_count: int = 0,
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
    if worker_count < 0:
        raise ValueError("kernel worker count must be nonnegative")
    _cpu_native.accumulate_kernel(
        entities,
        times,
        starts,
        ends,
        offsets,
        basis,
        lookup,
        accumulator,
        int(worker_count),
    )
    return True


def kernel_touched_positions(
    entities: np.ndarray,
    times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    offsets: np.ndarray,
    lookup: np.ndarray,
    *,
    horizon: int,
    marks: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray | None:
    """Return sorted accumulator positions changed by one completion slice.

    ``marks`` and ``positions`` are caller-owned reusable scratch arrays.  The
    compiled routine clears every mark before returning, including its error
    path, so one scratch allocation can safely serve all nested-W families.
    """
    if _cpu_native is None or not hasattr(_cpu_native, "kernel_touched_positions"):
        return None
    entities = np.ascontiguousarray(entities, dtype=np.int64)
    times = np.ascontiguousarray(times, dtype=np.int64)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    if lookup.dtype not in (np.int32, np.int64):
        lookup = np.asarray(lookup, dtype=np.int64)
    lookup = np.ascontiguousarray(lookup)
    if marks.dtype != np.uint8 or not marks.flags.c_contiguous:
        raise ValueError("kernel mark scratch must be C-contiguous uint8")
    if positions.dtype != np.int64 or not positions.flags.c_contiguous:
        raise ValueError("kernel position scratch must be C-contiguous int64")
    if len(positions) < len(marks):
        raise ValueError("kernel position scratch is smaller than marks")
    count = int(
        _cpu_native.kernel_touched_positions(
            entities,
            times,
            starts,
            ends,
            offsets,
            lookup,
            int(horizon),
            marks,
            positions,
        )
    )
    return positions[:count]


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


def fill_pricing_values(
    query_rows: np.ndarray,
    source_rows: np.ndarray,
    source_values: np.ndarray,
    output: np.ndarray,
    *,
    lookup: np.ndarray | None = None,
) -> bool:
    """Fill a hierarchy-pricing scratch matrix without Python index arrays."""
    if _cpu_native is None or not hasattr(_cpu_native, "fill_pricing_values"):
        return False
    query_rows = np.ascontiguousarray(query_rows, dtype=np.int64)
    source_rows = np.ascontiguousarray(source_rows, dtype=np.int64)
    source_values = np.ascontiguousarray(source_values, dtype=np.float64)
    if output.dtype != np.float64 or not output.flags.c_contiguous:
        raise ValueError("pricing output must be C-contiguous float64")
    if lookup is not None:
        lookup = np.ascontiguousarray(lookup)
    _cpu_native.fill_pricing_values(
        query_rows, source_rows, source_values, lookup, output
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
    sources: list[
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    *,
    relation: str = "unordered",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if relation not in {"atomic", "unordered", "ordered"}:
        raise ValueError("unknown completion relation")
    if _cpu_native is None:
        return _ordered_completion_events(sources) if relation == "ordered" else None
    if relation == "ordered" and not hasattr(_cpu_native, "ordered_completion_events"):
        return _ordered_completion_events(sources)
    if not 1 <= len(sources) <= 3:
        raise ValueError("completion requires one to three predicate sources")
    normalized = []
    synthetic_offset = 0
    for source in sources:
        length = len(source[0])
        primitive = (
            np.asarray(source[2])
            if len(source) == 3
            else np.arange(
                synthetic_offset, synthetic_offset + length, dtype=np.int64
            )
        )
        normalized.append((np.asarray(source[0]), np.asarray(source[1]), primitive))
        synthetic_offset += length
    lengths = np.asarray([len(entities) for entities, _, _ in normalized], dtype=np.int64)
    offsets = np.zeros(len(sources) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    if int(offsets[-1]) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    entities = np.ascontiguousarray(
        np.concatenate([item[0] for item in normalized]), dtype=np.int64
    )
    times = np.ascontiguousarray(
        np.concatenate([item[1] for item in normalized]), dtype=np.int64
    )
    primitive_ids = np.ascontiguousarray(
        np.concatenate([item[2] for item in normalized]), dtype=np.int64
    )
    capacity = len(entities)
    output_entities = np.empty(capacity, dtype=np.int64)
    output_times = np.empty(capacity, dtype=np.int64)
    output_spans = np.empty(capacity, dtype=np.int64)
    native_function = (
        _cpu_native.ordered_completion_events
        if relation == "ordered"
        else _cpu_native.completion_events
    )
    count = int(
        native_function(
            entities,
            times,
            primitive_ids,
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


def observed_temporal_motifs(
    entities: np.ndarray,
    times: np.ndarray,
    predicates: np.ndarray,
    primitive_ids: np.ndarray,
    *,
    predicate_count: int,
    q_max: int,
    maximum_span: int,
    allow_unordered: bool,
    allow_ordered: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return conservative completion counts from one target-blind pass.

    Positive entries are the former observed-motif masks.  Counts may exceed
    the canonical latest-witness completion count when several primitives at
    one tick witness the same motif; this is intentional and makes
    ``count * impact_lag`` a safe response-footprint cardinality bound.
    """
    if _cpu_native is None or not hasattr(_cpu_native, "observed_temporal_motifs"):
        return None
    predicate_count = int(predicate_count)
    pair_size = predicate_count * predicate_count
    triplet_size = pair_size * predicate_count
    atomic = np.zeros(predicate_count, dtype=np.int64)
    unordered_pair = np.zeros(pair_size, dtype=np.int64)
    ordered_pair = np.zeros(pair_size, dtype=np.int64)
    unordered_triplet = np.zeros(triplet_size, dtype=np.int64)
    ordered_triplet = np.zeros(triplet_size, dtype=np.int64)
    _cpu_native.observed_temporal_motifs(
        np.ascontiguousarray(entities, dtype=np.int64),
        np.ascontiguousarray(times, dtype=np.int64),
        np.ascontiguousarray(predicates, dtype=np.int64),
        np.ascontiguousarray(primitive_ids, dtype=np.int64),
        predicate_count,
        int(q_max),
        int(maximum_span),
        bool(allow_unordered),
        bool(allow_ordered),
        atomic,
        unordered_pair,
        ordered_pair,
        unordered_triplet,
        ordered_triplet,
    )
    return atomic, unordered_pair, ordered_pair, unordered_triplet, ordered_triplet


def _ordered_completion_events(
    sources: list[
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strict ordered completions without using coincident witnesses.

    A completion occurs at every distinct final-source tick for which a strict
    chain exists.  The latest feasible predecessor chain minimizes the span;
    retaining that span is sufficient for every nested formation window.
    """
    if not 2 <= len(sources) <= 3:
        raise ValueError("ordered completion requires two or three sources")
    normalized = [
        (
            np.asarray(source[0], dtype=np.int64),
            np.asarray(source[1], dtype=np.int64),
        )
        for source in sources
    ]
    by_source: list[dict[int, np.ndarray]] = []
    for entities, times in normalized:
        mapping: dict[int, np.ndarray] = {}
        if len(entities):
            unique, first = np.unique(entities, return_index=True)
            for index, entity in enumerate(unique):
                right = first[index + 1] if index + 1 < len(first) else len(entities)
                mapping[int(entity)] = np.unique(times[first[index] : right])
        by_source.append(mapping)
    eligible = set(by_source[0])
    for mapping in by_source[1:]:
        eligible.intersection_update(mapping)
    output_entities: list[int] = []
    output_times: list[int] = []
    output_spans: list[int] = []
    for entity in sorted(eligible):
        first_times = by_source[0][entity]
        second_times = by_source[1][entity]
        if len(sources) == 2:
            for terminal in second_times:
                position = int(np.searchsorted(first_times, terminal, side="left")) - 1
                if position >= 0:
                    output_entities.append(entity)
                    output_times.append(int(terminal))
                    output_spans.append(int(terminal - first_times[position]))
            continue
        third_times = by_source[2][entity]
        valid_second: list[int] = []
        valid_first: list[int] = []
        for middle in second_times:
            position = int(np.searchsorted(first_times, middle, side="left")) - 1
            if position >= 0:
                valid_second.append(int(middle))
                valid_first.append(int(first_times[position]))
        if not valid_second:
            continue
        middle_array = np.asarray(valid_second, dtype=np.int64)
        first_array = np.asarray(valid_first, dtype=np.int64)
        for terminal in third_times:
            position = int(np.searchsorted(middle_array, terminal, side="left")) - 1
            if position >= 0:
                output_entities.append(entity)
                output_times.append(int(terminal))
                output_spans.append(int(terminal - first_array[position]))
    return (
        np.asarray(output_entities, dtype=np.int64),
        np.asarray(output_times, dtype=np.int64),
        np.asarray(output_spans, dtype=np.int64),
    )


def completion_window_counts(
    sources: list[
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    ends: np.ndarray,
    windows: np.ndarray,
    *,
    relation: str = "unordered",
) -> np.ndarray | None:
    """Count productive completions for nested W without materializing them."""
    profile = completion_window_profile(sources, ends, windows, relation=relation)
    return None if profile is None else profile[0]


def completion_window_profile(
    sources: list[
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    ends: np.ndarray,
    windows: np.ndarray,
    *,
    relation: str = "unordered",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Count completions and return each entity's minimum productive span.

    The two outputs are produced by one streaming source merge.  A finite
    minimum span is an exact certificate that the antecedent can activate for
    every W at least that large; no completion arrays are materialized.
    """
    if relation == "ordered":
        compiled = completion_events(sources, relation="ordered")
        if compiled is None:
            return None
        entities, times, spans = compiled
        windows = np.ascontiguousarray(windows, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        counts = np.zeros(len(windows), dtype=np.int64)
        minimum = np.full(len(ends), np.iinfo(np.int64).max, dtype=np.int64)
        if len(entities):
            productive = times <= ends[entities]
            entities, spans = entities[productive], spans[productive]
            for index, window in enumerate(windows):
                counts[index] = int(np.count_nonzero(spans <= int(window)))
            if len(entities):
                np.minimum.at(minimum, entities, spans)
        return counts, minimum
    if relation not in {"atomic", "unordered"}:
        raise ValueError("unknown completion relation")
    if _cpu_native is None or not hasattr(_cpu_native, "completion_window_counts"):
        return None
    if not 1 <= len(sources) <= 3:
        raise ValueError("completion requires one to three predicate sources")
    windows = np.ascontiguousarray(windows, dtype=np.int64)
    if windows.ndim != 1 or not len(windows):
        raise ValueError("completion windows must be a nonempty vector")
    normalized = []
    synthetic_offset = 0
    for source in sources:
        length = len(source[0])
        primitive = (
            np.asarray(source[2])
            if len(source) == 3
            else np.arange(
                synthetic_offset, synthetic_offset + length, dtype=np.int64
            )
        )
        normalized.append((np.asarray(source[0]), np.asarray(source[1]), primitive))
        synthetic_offset += length
    lengths = np.asarray([len(entities) for entities, _, _ in normalized], dtype=np.int64)
    offsets = np.zeros(len(sources) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    entities = np.ascontiguousarray(
        np.concatenate([item[0] for item in normalized]), dtype=np.int64
    )
    times = np.ascontiguousarray(
        np.concatenate([item[1] for item in normalized]), dtype=np.int64
    )
    primitive_ids = np.ascontiguousarray(
        np.concatenate([item[2] for item in normalized]), dtype=np.int64
    )
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    counts = np.empty(len(windows), dtype=np.int64)
    minimum_spans = np.empty(len(ends), dtype=np.int64)
    _cpu_native.completion_window_counts(
        entities,
        times,
        primitive_ids,
        offsets,
        ends,
        windows,
        counts,
        minimum_spans,
    )
    return counts, minimum_spans


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


def safe_shell_counts_sources(
    source_offsets: np.ndarray,
    source_times: np.ndarray,
    source_primitives: np.ndarray,
    antecedent_predicates: np.ndarray,
    antecedent_orders: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    grid_offsets: np.ndarray,
    windows: np.ndarray,
    window_counts: np.ndarray,
    signed_state: np.ndarray,
    baseline_groups: np.ndarray,
    event_state: np.ndarray,
    run_ends: np.ndarray,
    *,
    horizon: int,
    group_count: int,
    exposure: float,
    weight: float,
) -> np.ndarray | None:
    """Fuse latest-witness completion and exact safe-shell aggregation.

    This returns the same first-admitting-W sufficient statistics as
    :func:`safe_shell_counts`, but consumes primitive predicate streams
    directly and never materializes an antecedent-wide completion array.
    """

    if _cpu_native is None or not hasattr(
        _cpu_native, "safe_shell_counts_sources"
    ):
        return None
    source_offsets = np.ascontiguousarray(source_offsets, dtype=np.int64)
    source_times = np.ascontiguousarray(source_times, dtype=np.int64)
    source_primitives = np.ascontiguousarray(source_primitives, dtype=np.int64)
    antecedent_predicates = np.ascontiguousarray(
        antecedent_predicates, dtype=np.int32
    )
    antecedent_orders = np.ascontiguousarray(antecedent_orders, dtype=np.int32)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    grid_offsets = np.ascontiguousarray(grid_offsets, dtype=np.int64)
    windows = np.ascontiguousarray(windows, dtype=np.int64)
    window_counts = np.ascontiguousarray(window_counts, dtype=np.int32)
    signed_state = np.ascontiguousarray(signed_state, dtype=np.uint8)
    baseline_groups = np.ascontiguousarray(baseline_groups, dtype=np.int32)
    event_state = np.ascontiguousarray(event_state, dtype=np.uint8)
    run_ends = np.ascontiguousarray(run_ends, dtype=np.int32)
    if antecedent_predicates.shape != (len(antecedent_orders), 3):
        raise ValueError("source safe-shell antecedent table is invalid")
    if windows.ndim != 2 or window_counts.shape != (windows.shape[0],):
        raise ValueError("source safe-shell W metadata is invalid")
    if windows.shape[0] != len(antecedent_orders):
        raise ValueError("source safe-shell antecedent/W tables do not align")
    output = np.zeros(
        (windows.shape[0], 4, windows.shape[1], 3, int(group_count)),
        dtype=np.float64,
    )
    _cpu_native.safe_shell_counts_sources(
        source_offsets,
        source_times,
        source_primitives,
        antecedent_predicates,
        antecedent_orders,
        starts,
        ends,
        grid_offsets,
        windows,
        window_counts,
        signed_state,
        baseline_groups,
        event_state,
        run_ends,
        int(horizon),
        int(group_count),
        float(exposure),
        float(weight),
        output,
    )
    return output


def label_run_ends(
    signed_state: np.ndarray,
    baseline_groups: np.ndarray,
    event_state: np.ndarray,
    grid_offsets: np.ndarray,
) -> np.ndarray | None:
    """Return the exclusive end of each immutable row-label run."""

    if _cpu_native is None or not hasattr(_cpu_native, "label_run_ends"):
        return None
    signed_state = np.ascontiguousarray(signed_state, dtype=np.uint8)
    baseline_groups = np.ascontiguousarray(baseline_groups, dtype=np.int32)
    event_state = np.ascontiguousarray(event_state, dtype=np.uint8)
    grid_offsets = np.ascontiguousarray(grid_offsets, dtype=np.int64)
    output = np.empty(len(signed_state), dtype=np.int32)
    _cpu_native.label_run_ends(
        signed_state,
        baseline_groups,
        event_state,
        grid_offsets,
        output,
    )
    return output


def safe_shell_counts(
    completion_offsets: np.ndarray,
    completion_times: np.ndarray,
    completion_spans: np.ndarray,
    antecedent_indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    grid_offsets: np.ndarray,
    windows: np.ndarray,
    window_counts: np.ndarray,
    signed_state: np.ndarray,
    baseline_groups: np.ndarray,
    event_state: np.ndarray,
    *,
    horizon: int,
    group_count: int,
    exposure: float,
    weight: float,
) -> np.ndarray | None:
    """Aggregate exact nested-W row counts without dense response materialization.

    The output axes are antecedent, signed-current-state category, first
    admitting W, exposure/noevent/event and baseline group.  It is the same
    sufficient statistic used by the localized/directional relaxation; only
    its execution representation differs.
    """

    if _cpu_native is None or not hasattr(_cpu_native, "safe_shell_counts"):
        return None
    completion_offsets = np.ascontiguousarray(completion_offsets, dtype=np.int64)
    completion_times = np.ascontiguousarray(completion_times, dtype=np.int64)
    completion_spans = np.ascontiguousarray(completion_spans, dtype=np.int64)
    antecedent_indices = np.ascontiguousarray(antecedent_indices, dtype=np.int32)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    grid_offsets = np.ascontiguousarray(grid_offsets, dtype=np.int64)
    windows = np.ascontiguousarray(windows, dtype=np.int64)
    window_counts = np.ascontiguousarray(window_counts, dtype=np.int32)
    signed_state = np.ascontiguousarray(signed_state, dtype=np.uint8)
    baseline_groups = np.ascontiguousarray(baseline_groups, dtype=np.int32)
    event_state = np.ascontiguousarray(event_state, dtype=np.uint8)
    if windows.ndim != 2 or window_counts.shape != (windows.shape[0],):
        raise ValueError("safe shell W metadata is invalid")
    output = np.zeros(
        (windows.shape[0], 4, windows.shape[1], 3, int(group_count)),
        dtype=np.float64,
    )
    _cpu_native.safe_shell_counts(
        completion_offsets,
        completion_times,
        completion_spans,
        antecedent_indices,
        starts,
        ends,
        grid_offsets,
        windows,
        window_counts,
        signed_state,
        baseline_groups,
        event_state,
        int(horizon),
        int(group_count),
        float(exposure),
        float(weight),
        output,
    )
    return output


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
