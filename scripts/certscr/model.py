from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .data import QueryContext
from .marked import MarkHeadFit
from .native import (
    aggregate_float32_rows,
    aggregate_float32_rows_pair,
    aggregate_float32_rows_partitioned,
    fit_prepared_cone_float64,
    sorted_unique_int64_union,
    sorted_unique_int64_union_with_positions,
)
from .occurrence import Antecedent, RuleIdentity, SparseKernelResponse


ClosureTerm = tuple[Antecedent, int]
DesignBlock = np.ndarray | SparseKernelResponse
ControlDesign = DesignBlock | Sequence[DesignBlock]


def _sorted_unique_indices(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Union sorted integer index vectors in one allocation/sort pass."""
    nonempty = [
        np.asarray(part, dtype=np.int64).reshape(-1)
        for part in parts
        if len(part)
    ]
    if not nonempty:
        return np.zeros(0, dtype=np.int64)
    if len(nonempty) == 1:
        return np.unique(nonempty[0]).astype(np.int64, copy=False)
    native = sorted_unique_int64_union(nonempty)
    if native is not None:
        return native
    return np.unique(np.concatenate(nonempty)).astype(np.int64, copy=False)


@dataclass(frozen=True)
class FitResult:
    rules: tuple[RuleIdentity, ...]
    closure_terms: tuple[ClosureTerm, ...]
    alpha: float
    gamma: np.ndarray
    theta: np.ndarray
    nll: float
    kkt_residual: float
    converged: bool
    iterations: int
    device: str
    # ``nll`` is the objective used for support discovery.  It equals
    # ``intensity_nll`` for an unmarked process and intensity_nll + mark_nll
    # for a marked process.  The occurrence and mark heads remain separately
    # available because financial certification tests two distinct estimands.
    intensity_nll: float | None = None
    mark_fit: MarkHeadFit | None = None

    @property
    def amplitudes(self) -> np.ndarray:
        return np.sum(self.theta, axis=1) if self.theta.size else np.zeros(0, dtype=np.float64)

    @property
    def shapes(self) -> np.ndarray:
        if not self.theta.size:
            return self.theta.copy()
        total = np.sum(self.theta, axis=1, keepdims=True)
        return np.divide(self.theta, total, out=np.zeros_like(self.theta), where=total > 0)


@dataclass(frozen=True)
class GroupSaturatedPoissonBound:
    """Exact lower bound obtained by saturating identical design rows.

    Rows with an identical augmented design necessarily share one linear
    predictor in the fitted TPP.  Giving every distinct row pattern its own
    unrestricted predictor relaxes the model, so its optimum is a lower bound
    on every signed/nonnegative kernel fit using that design.  The bound is
    algebraic; it does not use a local quadratic approximation.
    """

    lower_bound: float
    group_count: int
    active_grid_rows: int
    finite: bool


def _saturated_group_terms(
    noevent_mass: np.ndarray,
    event_mass: np.ndarray,
    occurrence_likelihood: str,
) -> tuple[np.ndarray, bool]:
    """Infimum of each independently saturated grouped occurrence model."""
    noevent = np.asarray(noevent_mass, dtype=np.float64)
    events = np.asarray(event_mass, dtype=np.float64)
    likelihood = _validate_occurrence_likelihood(occurrence_likelihood)
    terms = np.zeros_like(events)
    if likelihood == "poisson":
        positive = events > 0.0
        if np.any(positive & (noevent <= 0.0)):
            return terms, False
        terms[positive] = events[positive] * (
            1.0 - np.log(events[positive] / noevent[positive])
        )
        return terms, True
    both = (events > 0.0) & (noevent > 0.0)
    # If a saturated pattern occurs only in event or only in no-event cells,
    # its Bernoulli NLL has infimum zero. With both masses present, the exact
    # optimum satisfies exp(eta)=log(1+y/n0).
    if np.any(both):
        hazard = np.log1p(events[both] / noevent[both])
        eta = np.log(hazard)
        terms[both] = (
            noevent[both] * hazard
            + events[both] * cloglog_event_nll(eta)
        )
    return terms, True


@dataclass(frozen=True)
class PreparedFixedSupportDesign:
    """Exact grouped sufficient statistics reusable across KKT and fitting."""

    design: np.ndarray
    n_events: int
    event_weights: np.ndarray
    grid_weights: np.ndarray
    constrained_start: int
    control_width: int
    knot_count: int
    active_grid_rows: int
    rules: tuple[RuleIdentity, ...]
    # ``poisson`` is the exact event-time counting-process likelihood used by
    # recurrent streams. ``first_event_cloglog`` is the exact grouped
    # likelihood when a terminal event is only observed inside a unit-width
    # reporting interval (for example a Freddie Mac reporting month).
    occurrence_likelihood: str = "poisson"


OCCURRENCE_LIKELIHOODS = frozenset({"poisson", "first_event_cloglog"})


def _validate_occurrence_likelihood(value: str) -> str:
    likelihood = str(value)
    if likelihood not in OCCURRENCE_LIKELIHOODS:
        raise ValueError(f"unknown occurrence likelihood: {likelihood}")
    return likelihood


def _cloglog_event_terms_numpy(
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable event-cell NLL, gradient and Hessian for a cloglog hazard.

    With unit-interval integrated hazard ``mu=exp(eta)``, an observed terminal
    event has probability ``1-exp(-mu)``.  The returned derivatives are with
    respect to ``eta`` and the Hessian is nonnegative.
    """
    values = np.asarray(eta, dtype=np.float64)
    loss = np.empty_like(values)
    gradient = np.empty_like(values)
    hessian = np.empty_like(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mu = np.exp(values)
    small = mu < 1.0e-4
    regular = (~small) & (mu <= 50.0) & np.isfinite(mu)
    large = (~small) & (~regular)
    if np.any(small):
        m = mu[small]
        # Bernoulli-number expansions avoid cancellation at rare-event
        # intercepts, exactly where mortgage first-event models operate.
        loss[small] = -np.log(m) + 0.5 * m - (m * m) / 24.0
        gradient[small] = -1.0 + 0.5 * m - (m * m) / 12.0
        hessian[small] = 0.5 * m - (m * m) / 6.0
    if np.any(regular):
        m = mu[regular]
        denominator = np.expm1(m)
        loss[regular] = -np.log(-np.expm1(-m))
        gradient[regular] = -m / denominator
        hessian[regular] = (
            -m / denominator
            + (m * m) * np.exp(m) / (denominator * denominator)
        )
    if np.any(large):
        finite_eta = np.isfinite(values[large])
        # At very large positive eta the event probability is numerically one.
        # A nonfinite/negative input remains invalid and is caught by callers.
        loss[large] = np.where(finite_eta & (values[large] > 0.0), 0.0, math.inf)
        gradient[large] = np.where(finite_eta & (values[large] > 0.0), 0.0, math.nan)
        hessian[large] = np.where(finite_eta & (values[large] > 0.0), 0.0, math.nan)
    return loss, gradient, np.maximum(hessian, 0.0)


def cloglog_event_nll(eta: np.ndarray) -> np.ndarray:
    """Return ``-log(1-exp(-exp(eta)))`` without rare-event cancellation."""
    return _cloglog_event_terms_numpy(eta)[0]


def cloglog_event_terms(
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Public stable event-cell NLL, gradient, and observed information."""
    return _cloglog_event_terms_numpy(eta)


def promote_prepared_design_float64(
    prepared: PreparedFixedSupportDesign,
) -> PreparedFixedSupportDesign:
    """Promote a grouped design once before repeated host-float64 operations.

    Float32 response values are exactly representable in float64.  Promotion
    therefore changes neither a design entry nor the fitted problem, while
    allowing the caller to release the float32 matrix before the Newton loop
    instead of retaining both matrices for the whole solve.
    """
    if prepared.design.dtype == np.float64:
        return prepared
    return replace(
        prepared,
        design=prepared.design.astype(np.float64, copy=True),
    )


def project_prepared_support_design(
    prepared: PreparedFixedSupportDesign,
    rules: Sequence[RuleIdentity],
    *,
    source_closure_terms: Sequence[ClosureTerm] | None = None,
    target_closure_terms: Sequence[ClosureTerm] | None = None,
    regroup: bool = False,
) -> PreparedFixedSupportDesign:
    """Select an exact nested support from an already grouped full design.

    Rows grouped by the full column set remain a valid (possibly finer) exact
    sufficient-statistic partition after columns are removed.  Reusing that
    partition avoids rebuilding a nested drop model from the full time grid;
    no row, exposure mass, event mass, objective term, or constraint changes.
    """
    target_rules = tuple(rules)
    full_rules = tuple(prepared.rules)
    full_index = {rule: index for index, rule in enumerate(full_rules)}
    if any(rule not in full_index for rule in target_rules):
        raise ValueError("nested support contains a rule absent from full design")
    knot_count = int(prepared.knot_count)
    closure_projection = (
        source_closure_terms is not None or target_closure_terms is not None
    )
    if closure_projection and (
        source_closure_terms is None or target_closure_terms is None
    ):
        raise ValueError("source and target closure terms must be supplied together")
    column_scales: list[float] = []
    if not closure_projection:
        columns = list(range(int(prepared.constrained_start)))
        column_scales.extend([1.0] * len(columns))
        target_control_width = int(prepared.control_width)
        target_constrained_start = int(prepared.constrained_start)
    else:
        source_closure = tuple(sorted(source_closure_terms or ()))
        target_closure = tuple(sorted(target_closure_terms or ()))
        fixed_control_width = int(prepared.control_width) - len(source_closure) * knot_count
        if fixed_control_width < 0:
            raise ValueError("source closure width exceeds prepared nuisance width")
        columns = list(range(1 + fixed_control_width))
        column_scales.extend([1.0] * len(columns))
        source_closure_index = {
            term: index for index, term in enumerate(source_closure)
        }
        source_rule_term = {
            (rule.antecedent, int(rule.window)): (index, int(rule.sign))
            for index, rule in enumerate(full_rules)
        }
        for term in target_closure:
            closure_index = source_closure_index.get(term)
            if closure_index is not None:
                left = 1 + fixed_control_width + closure_index * knot_count
                scale = 1.0
            else:
                represented = source_rule_term.get(term)
                if represented is None:
                    raise ValueError(
                        "target closure term is absent from the full design"
                    )
                rule_index, rule_sign = represented
                left = int(prepared.constrained_start) + rule_index * knot_count
                # Rule blocks in the augmented design already carry their sign;
                # hierarchy nuisance blocks are unsigned.
                scale = float(rule_sign)
            columns.extend(range(left, left + knot_count))
            column_scales.extend([scale] * knot_count)
        target_control_width = fixed_control_width + len(target_closure) * knot_count
        target_constrained_start = 1 + target_control_width
    for rule in target_rules:
        left = int(prepared.constrained_start) + full_index[rule] * knot_count
        columns.extend(range(left, left + knot_count))
        column_scales.extend([1.0] * knot_count)
    selected = np.ascontiguousarray(
        prepared.design[:, np.asarray(columns, dtype=np.int64)],
        dtype=prepared.design.dtype,
    )
    scales = np.asarray(column_scales, dtype=selected.dtype)
    if np.any(scales != 1.0):
        selected *= scales.reshape(1, -1)
    selected_n_events = int(prepared.n_events)
    selected_event_weights = prepared.event_weights
    selected_grid_weights = prepared.grid_weights
    if regroup:
        (
            selected,
            selected_n_events,
            selected_event_weights,
            selected_grid_weights,
        ) = aggregate_duplicate_design_rows(
            selected,
            selected_n_events,
            selected_event_weights,
            selected_grid_weights,
            inplace=True,
        )
    return PreparedFixedSupportDesign(
        design=selected,
        n_events=selected_n_events,
        event_weights=selected_event_weights,
        grid_weights=selected_grid_weights,
        constrained_start=target_constrained_start,
        control_width=target_control_width,
        knot_count=knot_count if target_rules else 0,
        # This field is reporting metadata only.  The inherited full-support
        # count is a conservative count of the ungrouped rows represented by
        # this finer sufficient-statistic partition.
        active_grid_rows=int(prepared.active_grid_rows),
        rules=target_rules,
        occurrence_likelihood=prepared.occurrence_likelihood,
    )


def _validated_cluster_weights(
    ctx: QueryContext,
    cluster_weights: np.ndarray | None,
) -> np.ndarray:
    if cluster_weights is None:
        return np.ones(ctx.n_sequences, dtype=np.float64)
    weights = np.asarray(cluster_weights, dtype=np.float64)
    if weights.shape != (ctx.n_sequences,):
        raise ValueError("cluster weights must match the query-context sequences")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("cluster weights must be finite, nonnegative, and have positive mass")
    return weights


def canonical_nll(
    eta: np.ndarray,
    ctx: QueryContext,
    cluster_weights: np.ndarray | None = None,
    *,
    occurrence_likelihood: str = "poisson",
) -> float:
    eta = np.asarray(eta, dtype=np.float64)
    if len(eta) != ctx.n_queries:
        raise ValueError("eta/query length mismatch")
    weights = _validated_cluster_weights(ctx, cluster_weights)
    likelihood = _validate_occurrence_likelihood(occurrence_likelihood)
    event_weights = weights[ctx.event_sequence_local]
    grid_eta = eta[ctx.n_events :]
    with np.errstate(over="ignore", invalid="ignore"):
        grid_intensity = np.exp(grid_eta)
    if np.any(~np.isfinite(grid_intensity)):
        return math.inf
    exposure = float(
        np.dot(
            weights,
            ctx.aggregate_weighted_grid(grid_intensity),
        )
    )
    if likelihood == "poisson":
        return exposure - float(np.dot(event_weights, eta[: ctx.n_events]))
    event_grid_hazard = np.exp(
        grid_eta[np.asarray(ctx.event_grid_rows, dtype=np.int64)]
    )
    event_loss = cloglog_event_nll(eta[: ctx.n_events])
    return (
        exposure
        - float(np.dot(event_weights, event_grid_hazard))
        + float(np.dot(event_weights, event_loss))
    )


def assemble_design(
    controls: np.ndarray,
    rule_features: Sequence[np.ndarray],
    rules: Sequence[RuleIdentity],
) -> tuple[np.ndarray, int]:
    controls = np.asarray(controls, dtype=np.float32)
    if controls.ndim != 2:
        raise ValueError("control design must be a matrix")
    if len(rule_features) != len(rules):
        raise ValueError("rule feature/rule length mismatch")
    n = controls.shape[0]
    parts = [np.ones((n, 1), dtype=np.float32)]
    if controls.shape[1]:
        parts.append(controls)
    for rule, feature in zip(rules, rule_features, strict=True):
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim != 2 or feature.shape[0] != n:
            raise ValueError("invalid rule feature matrix")
        parts.append(float(rule.sign) * feature)
    return np.concatenate(parts, axis=1), 1 + controls.shape[1]


def compress_zero_grid_rows(
    design: np.ndarray,
    n_events: int,
    grid_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Exactly aggregate grid rows whose non-intercept design is all zero.

    Such rows all have ``eta=alpha``.  The Poisson exposure, gradient and
    Fisher information therefore depend on them only through the sum of their
    weights.  Event rows are never aggregated because downstream event terms
    use their individual weights.  This is an algebraic sufficient-statistic
    reduction, not an approximation.
    """
    design = np.asarray(design, dtype=np.float32)
    weights = np.asarray(grid_weights, dtype=np.float64)
    n_events = int(n_events)
    if design.ndim != 2 or not 0 <= n_events <= len(design):
        raise ValueError("invalid design/event boundary")
    if weights.shape != (len(design) - n_events,):
        raise ValueError("grid weights do not match design")
    grid = design[n_events:]
    if not len(grid) or design.shape[1] <= 1:
        active = np.zeros(len(grid), dtype=bool)
    else:
        active = np.any(grid[:, 1:] != 0.0, axis=1)
    # A zero-weight quadrature row contributes exactly zero even when its
    # covariates are active.  Keeping it only enlarges the design/Hessian.
    active &= weights > 0.0
    inactive_weight = float(np.sum(weights[~active], dtype=np.float64))
    active_count = int(np.sum(active))
    if inactive_weight <= 0.0:
        return design, weights, active_count
    zero_row = np.zeros((1, design.shape[1]), dtype=np.float32)
    zero_row[0, 0] = 1.0
    compressed = np.concatenate(
        (design[:n_events], grid[active], zero_row),
        axis=0,
    )
    compressed_weights = np.concatenate(
        (weights[active], np.asarray([inactive_weight], dtype=np.float64))
    )
    return compressed, compressed_weights, active_count


def assemble_compressed_design(
    controls: ControlDesign,
    rule_features: Sequence[DesignBlock],
    rules: Sequence[RuleIdentity],
    *,
    n_events: int,
    grid_weights: np.ndarray | None = None,
    base_grid_weights: np.ndarray | None = None,
    grid_sequence_local: np.ndarray | None = None,
    sequence_weights: np.ndarray | None = None,
    weighted_exposure: float | None = None,
    grid_context: QueryContext | None = None,
    excluded_grid_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build only event rows, active grid rows, and one exact zero-grid row.

    This is algebraically identical to ``assemble_design`` followed by
    ``compress_zero_grid_rows`` but never materializes the usually enormous
    dense full-grid design first.
    """
    control_blocks = _validated_control_blocks(controls)
    if len(rule_features) != len(rules):
        raise ValueError("rule feature/rule length mismatch")
    factorized_weights = grid_weights is None
    if factorized_weights:
        if sequence_weights is None or weighted_exposure is None:
            raise ValueError("factorized grid weights are incomplete")
        sequence_weight_values = np.asarray(sequence_weights, dtype=np.float64)
        if grid_context is not None:
            n_grid = int(grid_context.n_grid)
            base_weights = None
            sequence_local = None
        else:
            if base_grid_weights is None or grid_sequence_local is None:
                raise ValueError("factorized grid weights are incomplete")
            base_weights = np.asarray(base_grid_weights)
            sequence_local = np.asarray(grid_sequence_local)
            if base_weights.ndim != 1 or sequence_local.shape != base_weights.shape:
                raise ValueError("factorized grid weights have inconsistent rows")
            n_grid = int(len(base_weights))
        total_weight = float(weighted_exposure)
    else:
        weights = np.asarray(grid_weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError("grid weights must be a vector")
        n_grid = int(len(weights))
        total_weight = float(np.sum(weights, dtype=np.float64))
    n_queries = int(control_blocks[0].shape[0]) if control_blocks else int(
        rule_features[0].shape[0] if rule_features else n_events + n_grid
    )
    n_events = int(n_events)
    if not 0 <= n_events <= n_queries:
        raise ValueError("invalid event boundary")
    if n_grid != n_queries - n_events:
        raise ValueError("grid weights do not match query rows")
    excluded = np.asarray(
        () if excluded_grid_rows is None else excluded_grid_rows,
        dtype=np.int64,
    ).reshape(-1)
    if len(excluded):
        if np.any(excluded < 0) or np.any(excluded >= n_grid):
            raise ValueError("excluded grid row is outside the query grid")
        excluded = np.unique(excluded)

    # Keep signs as metadata and apply them while filling the one final design
    # allocation.  Materializing signed copies of every sparse block doubled
    # response memory traffic before assembly even started.
    signed_features: list[tuple[DesignBlock, np.float32]] = []
    for rule, feature in zip(rules, rule_features, strict=True):
        if isinstance(feature, SparseKernelResponse):
            values: DesignBlock = feature
        else:
            values = np.asarray(feature, dtype=np.float32)
        if len(values.shape) != 2 or values.shape[0] != n_queries:
            raise ValueError("invalid rule feature matrix")
        signed_features.append((values, np.float32(rule.sign)))
    nonintercept: list[tuple[DesignBlock, np.float32]] = [
        *((block, np.float32(1.0)) for block in control_blocks),
        *signed_features,
    ]
    sparse_layout: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
    all_sparse = bool(nonintercept) and all(
        isinstance(block, SparseKernelResponse)
        for block, _sign in nonintercept
    )
    if all_sparse:
        sparse_blocks = tuple(
            block
            for block, _sign in nonintercept
            if isinstance(block, SparseKernelResponse)
        )
        native_layout = sorted_unique_int64_union_with_positions(
            [block.grid_indices for block in sparse_blocks],
            assume_sorted=True,
        )
    else:
        sparse_blocks = ()
        native_layout = None
    if native_layout is not None:
        active_indices, raw_sparse_positions = native_layout
    else:
        active_index_parts: list[np.ndarray] = []
        for block, _sign in nonintercept:
            if block.shape[1]:
                if isinstance(block, SparseKernelResponse):
                    block_indices = block.grid_indices[
                        np.any(block.grid_values != 0.0, axis=1)
                    ]
                else:
                    block_indices = np.flatnonzero(
                        np.any(block[n_events:] != 0.0, axis=1)
                    ).astype(np.int64, copy=False)
                active_index_parts.append(block_indices)
        active_indices = _sorted_unique_indices(active_index_parts)
    if factorized_weights:
        if grid_context is not None:
            active_weights = grid_context.grid_weights_at(
                active_indices, assume_valid=True
            ) * (
                sequence_weight_values[
                    grid_context.grid_sequences_at(
                        active_indices,
                        assume_valid=True,
                        assume_sorted=True,
                    )
                ]
            )
        else:
            assert base_weights is not None and sequence_local is not None
            active_weights = (
                base_weights[active_indices].astype(np.float64)
                * sequence_weight_values[sequence_local[active_indices]]
            )
    else:
        active_weights = weights[active_indices]
    positive_weight = active_weights > 0.0
    if len(excluded) and len(active_indices):
        excluded_positions = np.searchsorted(excluded, active_indices)
        safe_excluded = np.minimum(excluded_positions, len(excluded) - 1)
        positive_weight &= (excluded_positions >= len(excluded)) | (
            excluded[safe_excluded] != active_indices
        )
    if native_layout is not None:
        sparse_layout = tuple(
            (
                positions[positive_weight[positions]].astype(
                    np.int64, copy=False
                ),
                positive_weight[positions],
            )
            for positions in raw_sparse_positions
        )
        if np.any(~positive_weight):
            old_to_new = np.full(len(positive_weight), -1, dtype=np.int64)
            old_to_new[positive_weight] = np.arange(
                int(np.sum(positive_weight)), dtype=np.int64
            )
            sparse_layout = tuple(
                (old_to_new[target], source_keep)
                for target, source_keep in sparse_layout
            )
    active_indices = active_indices[positive_weight]
    active_weights = active_weights[positive_weight]
    inactive_weight = float(
        total_weight - np.sum(active_weights, dtype=np.float64)
    )
    roundoff = np.finfo(np.float64).eps * max(1.0, abs(total_weight)) * 64.0
    if inactive_weight < 0.0 and inactive_weight >= -roundoff:
        inactive_weight = 0.0
    if inactive_weight < 0.0:
        raise ValueError("active grid exposure exceeds total exposure")

    nonzero_rows = n_events + len(active_indices)
    has_zero_row = inactive_weight > 0.0
    total_rows = nonzero_rows + int(has_zero_row)
    design_width = 1 + sum(block.shape[1] for block, _sign in nonintercept)
    # The array starts at zero, so sparse blocks only write their matched rows.
    # This replaces one selected matrix and two concatenation copies per block.
    design = np.zeros((total_rows, design_width), dtype=np.float32)
    design[:, 0] = 1.0
    column = 1
    for block_index, (block, sign) in enumerate(nonintercept):
        if not block.shape[1]:
            continue
        right = column + int(block.shape[1])
        if isinstance(block, SparseKernelResponse):
            if n_events:
                if sign == np.float32(1.0):
                    design[:n_events, column:right] = block.event_values
                else:
                    np.multiply(
                        block.event_values,
                        sign,
                        out=design[:n_events, column:right],
                    )
            if len(active_indices) and len(block.grid_indices):
                if sparse_layout is not None:
                    target_positions, source_keep = sparse_layout[block_index]
                    target = n_events + target_positions
                    values = block.grid_values[source_keep]
                else:
                    positions = np.searchsorted(active_indices, block.grid_indices)
                    safe = np.minimum(positions, len(active_indices) - 1)
                    matched = (positions < len(active_indices)) & (
                        active_indices[safe] == block.grid_indices
                    )
                    target = n_events + positions[matched]
                    values = block.grid_values[matched]
                if sign == np.float32(1.0):
                    design[target, column:right] = values
                else:
                    design[target, column:right] = sign * values
        else:
            if n_events:
                design[:n_events, column:right] = sign * block[:n_events]
            if len(active_indices):
                design[n_events:nonzero_rows, column:right] = (
                    sign * block[n_events + active_indices]
                )
        column = right
    if column != design_width:
        raise ValueError("assembled design width mismatch")
    if has_zero_row:
        active_weights = np.concatenate(
            (active_weights, np.asarray([inactive_weight], dtype=np.float64))
        )
    control_width = sum(block.shape[1] for block in control_blocks)
    return design, active_weights, 1 + control_width, int(len(active_indices))


def group_saturated_poisson_lower_bound(
    ctx: QueryContext,
    controls: ControlDesign,
    rule_features: Sequence[DesignBlock],
    rules: Sequence[RuleIdentity],
    *,
    cluster_weights: np.ndarray | None = None,
    sequence_exposures: np.ndarray | None = None,
    prepared_design: PreparedFixedSupportDesign | None = None,
    occurrence_likelihood: str = "poisson",
) -> GroupSaturatedPoissonBound:
    """Return a safe lower bound for an augmented Poisson TPP objective.

    Only nonzero grid rows are materialized.  All-zero rows form one exact
    design group, and nonzero rows are grouped by their complete bit-identical
    float32 design.  For group exposure ``w`` and weighted event count ``y``,
    minimizing over an independent log intensity gives

        y * (1 - log(y / w))  if y > 0, else 0.

    The independently saturated group model contains the requested linear
    model, hence the sum is a global lower bound and can safely reject a model
    whose best possible block-MDL score is nonpositive.
    """

    rules = tuple(rules)
    control_blocks = _validated_control_blocks(controls)
    if len(rule_features) != len(rules):
        raise ValueError("rule feature/rule length mismatch")
    n_queries = int(
        control_blocks[0].shape[0]
        if control_blocks
        else rule_features[0].shape[0]
        if rule_features
        else ctx.n_queries
    )
    if n_queries != ctx.n_queries:
        raise ValueError("safe-bound design/query length mismatch")

    if prepared_design is not None:
        prepared = prepared_design
        if (
            tuple(prepared.rules) != rules
            or prepared.control_width
            != sum(block.shape[1] for block in control_blocks)
            or prepared.knot_count
            != (int(rule_features[0].shape[1]) if rule_features else 0)
        ):
            raise ValueError("prepared safe-bound design does not match model blocks")
        grid_mass = np.concatenate(
            (
                np.zeros(prepared.n_events, dtype=np.float64),
                prepared.grid_weights,
            )
        )
        event_mass = np.concatenate(
            (
                prepared.event_weights,
                np.zeros(len(prepared.grid_weights), dtype=np.float64),
            )
        )
        grouped = aggregate_float32_rows_pair(
            prepared.design,
            grid_mass,
            event_mass,
        )
        if grouped is None:
            row_type = np.dtype(
                (
                    np.void,
                    prepared.design.dtype.itemsize * prepared.design.shape[1],
                )
            )
            row_bytes = np.ascontiguousarray(prepared.design).view(row_type).reshape(-1)
            _patterns, inverse = np.unique(row_bytes, return_inverse=True)
            exposure = np.bincount(
                inverse, weights=grid_mass, minlength=len(_patterns)
            ).astype(np.float64, copy=False)
            events = np.bincount(
                inverse, weights=event_mass, minlength=len(_patterns)
            ).astype(np.float64, copy=False)
            group_count = int(len(_patterns))
        else:
            _group_design, exposure, events = grouped
            group_count = int(len(exposure))
        likelihood = prepared.occurrence_likelihood
        if likelihood != _validate_occurrence_likelihood(occurrence_likelihood):
            raise ValueError("prepared safe-bound likelihood does not match request")
        terms, finite = _saturated_group_terms(exposure, events, likelihood)
        if not finite:
            return GroupSaturatedPoissonBound(
                -math.inf,
                group_count,
                prepared.active_grid_rows,
                False,
            )
        lower_bound = float(np.sum(terms, dtype=np.float64))
        return GroupSaturatedPoissonBound(
            lower_bound=lower_bound,
            group_count=group_count,
            active_grid_rows=prepared.active_grid_rows,
            finite=math.isfinite(lower_bound),
        )

    if _validate_occurrence_likelihood(occurrence_likelihood) != "poisson":
        raise ValueError(
            "first-event safe bounds require a prepared design with event cells excluded"
        )

    signed_features: list[DesignBlock] = []
    for rule, raw in zip(rules, rule_features, strict=True):
        feature: DesignBlock = (
            raw
            if isinstance(raw, SparseKernelResponse)
            else np.asarray(raw, dtype=np.float32)
        )
        if len(feature.shape) != 2 or feature.shape[0] != n_queries:
            raise ValueError("invalid safe-bound rule feature matrix")
        if isinstance(feature, SparseKernelResponse):
            signed_features.append(
                SparseKernelResponse(
                    n_events=feature.n_events,
                    n_grid=feature.n_grid,
                    grid_indices=feature.grid_indices,
                    grid_values=np.float32(rule.sign) * feature.grid_values,
                    event_values=np.float32(rule.sign) * feature.event_values,
                )
            )
        else:
            signed_features.append(np.float32(rule.sign) * feature)
    blocks = [*control_blocks, *signed_features]

    sequence_weights = _validated_cluster_weights(ctx, cluster_weights)
    if sequence_exposures is None:
        per_sequence_exposure = ctx.sequence_exposures()
    else:
        per_sequence_exposure = np.asarray(sequence_exposures, dtype=np.float64)
    if per_sequence_exposure.shape != (ctx.n_sequences,):
        raise ValueError("sequence exposure length mismatch")
    total_grid_exposure = float(per_sequence_exposure @ sequence_weights)
    active_index_parts: list[np.ndarray] = []
    for block in blocks:
        if block.shape[1]:
            if isinstance(block, SparseKernelResponse):
                block_indices = block.grid_indices[
                    np.any(block.grid_values != 0.0, axis=1)
                ]
            else:
                block_indices = np.flatnonzero(
                    np.any(block[ctx.n_events :] != 0.0, axis=1)
                ).astype(np.int64, copy=False)
            active_index_parts.append(block_indices)
    active_indices = _sorted_unique_indices(active_index_parts)
    active_grid_weights = ctx.grid_weights_at(
        active_indices, assume_valid=True
    ) * sequence_weights[
        ctx.grid_sequences_at(
            active_indices, assume_valid=True, assume_sorted=True
        )
    ]
    positive_weight = active_grid_weights > 0.0
    active_indices = active_indices[positive_weight]
    active_grid_weights = active_grid_weights[positive_weight]

    event_grid_indices = ctx.event_grid_rows
    event_weights = sequence_weights[ctx.event_sequence_local]
    active_event_counts = np.zeros(len(active_indices), dtype=np.float64)
    unmatched_event_weight = 0.0
    if len(event_grid_indices):
        if len(active_indices):
            positions = np.searchsorted(active_indices, event_grid_indices)
            safe_positions = np.minimum(positions, len(active_indices) - 1)
            matched = (positions < len(active_indices)) & (
                active_indices[safe_positions] == event_grid_indices
            )
        else:
            positions = np.zeros(len(event_grid_indices), dtype=np.int64)
            matched = np.zeros(len(event_grid_indices), dtype=bool)
        if np.any(matched):
            active_event_counts = np.bincount(
                positions[matched],
                weights=event_weights[matched],
                minlength=len(active_indices),
            ).astype(np.float64, copy=False)
        unmatched_event_weight = float(np.sum(event_weights[~matched], dtype=np.float64))
        if np.any(~matched):
            # An unmatched event is valid only on the common all-zero design.
            # A zero-exposure nonzero row would have no finite saturated bound.
            query_rows = ctx.n_events + event_grid_indices[~matched]
            nonzero_unmatched = False
            for block in blocks:
                if not block.shape[1]:
                    continue
                if isinstance(block, SparseKernelResponse):
                    values = block.event_values[~matched]
                else:
                    values = block[query_rows]
                if np.any(values != 0.0):
                    nonzero_unmatched = True
                    break
            if nonzero_unmatched:
                return GroupSaturatedPoissonBound(-math.inf, 0, len(active_indices), False)

    group_exposure: list[np.ndarray] = []
    group_events: list[np.ndarray] = []
    group_count = 0
    if len(active_indices):
        query_rows = ctx.n_events + active_indices
        gathered: list[np.ndarray] = []
        for block in blocks:
            if not block.shape[1]:
                continue
            if isinstance(block, SparseKernelResponse):
                values = np.zeros((len(active_indices), block.shape[1]), dtype=np.float32)
                if len(block.grid_indices):
                    positions = np.searchsorted(block.grid_indices, active_indices)
                    safe = np.minimum(positions, len(block.grid_indices) - 1)
                    matched_rows = (positions < len(block.grid_indices)) & (
                        block.grid_indices[safe] == active_indices
                    )
                    values[matched_rows] = block.grid_values[positions[matched_rows]]
                gathered.append(values)
            else:
                gathered.append(block[query_rows])
        nonintercept = np.concatenate(gathered, axis=1)
        contiguous = np.ascontiguousarray(nonintercept, dtype=np.float32)
        native_groups = aggregate_float32_rows_pair(
            contiguous,
            active_grid_weights,
            active_event_counts,
        )
        if native_groups is not None:
            _group_design, exposure_mass, event_mass = native_groups
            group_count += int(len(exposure_mass))
            group_exposure.append(exposure_mass)
            group_events.append(event_mass)
        else:
            row_bytes = contiguous.view(
                np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
            ).reshape(-1)
            _patterns, inverse = np.unique(row_bytes, return_inverse=True)
            group_count += int(len(_patterns))
            group_exposure.append(
                np.bincount(
                    inverse,
                    weights=active_grid_weights,
                    minlength=len(_patterns),
                ).astype(np.float64, copy=False)
            )
            group_events.append(
                np.bincount(
                    inverse,
                    weights=active_event_counts,
                    minlength=len(_patterns),
                ).astype(np.float64, copy=False)
            )

    inactive_exposure = float(
        total_grid_exposure
        - np.sum(active_grid_weights, dtype=np.float64)
    )
    roundoff = (
        np.finfo(np.float64).eps
        * max(1.0, abs(total_grid_exposure))
        * 64.0
    )
    if inactive_exposure < 0.0 and inactive_exposure >= -roundoff:
        inactive_exposure = 0.0
    if inactive_exposure < 0.0:
        raise ValueError("active grid exposure exceeds total exposure")
    if inactive_exposure > 0.0 or unmatched_event_weight > 0.0:
        group_count += 1
        group_exposure.append(np.asarray([inactive_exposure], dtype=np.float64))
        group_events.append(np.asarray([unmatched_event_weight], dtype=np.float64))
    if not group_exposure:
        return GroupSaturatedPoissonBound(-math.inf, 0, len(active_indices), False)

    exposure = np.concatenate(group_exposure)
    events = np.concatenate(group_events)
    terms, finite = _saturated_group_terms(
        exposure,
        events,
        occurrence_likelihood,
    )
    if not finite:
        return GroupSaturatedPoissonBound(-math.inf, group_count, len(active_indices), False)
    lower_bound = float(np.sum(terms, dtype=np.float64))
    return GroupSaturatedPoissonBound(
        lower_bound=lower_bound,
        group_count=group_count,
        active_grid_rows=len(active_indices),
        finite=math.isfinite(lower_bound),
    )


def _validated_control_blocks(controls: ControlDesign) -> tuple[DesignBlock, ...]:
    raw_blocks = (
        (controls,)
        if isinstance(controls, (np.ndarray, SparseKernelResponse))
        else tuple(controls)
    )
    blocks: list[DesignBlock] = []
    n_rows: int | None = None
    for raw in raw_blocks:
        block: DesignBlock = (
            raw
            if isinstance(raw, SparseKernelResponse)
            else np.asarray(raw, dtype=np.float32)
        )
        if len(block.shape) != 2:
            raise ValueError("control design blocks must be matrices")
        if n_rows is None:
            n_rows = int(block.shape[0])
        elif block.shape[0] != n_rows:
            raise ValueError("control design blocks must share their query rows")
        if block.shape[1]:
            blocks.append(block)
    return tuple(blocks)


def aggregate_duplicate_design_rows(
    design: np.ndarray,
    n_events: int,
    event_weights: np.ndarray,
    grid_weights: np.ndarray,
    *,
    inplace: bool = False,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Aggregate bit-identical event/grid rows into exact sufficient statistics."""
    matrix = np.ascontiguousarray(design, dtype=np.float32)
    event_weight = np.asarray(event_weights, dtype=np.float64)
    grid_weight = np.asarray(grid_weights, dtype=np.float64)
    if event_weight.shape != (int(n_events),):
        raise ValueError("event weights do not align with design")
    if grid_weight.shape != (len(matrix) - int(n_events),):
        raise ValueError("grid weights do not align with design")

    partitioned = aggregate_float32_rows_partitioned(
        matrix,
        int(n_events),
        event_weight,
        grid_weight,
        inplace=inplace,
    )
    if partitioned is not None:
        grouped_design, grouped_events, grouped_weights = partitioned
        return (
            grouped_design,
            int(grouped_events),
            grouped_weights[:grouped_events],
            grouped_weights[grouped_events:],
        )

    def aggregate(block: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(block):
            return block.copy(), weights.copy()
        native = aggregate_float32_rows(block, weights)
        if native is not None:
            return native
        row_type = np.dtype((np.void, block.dtype.itemsize * block.shape[1]))
        row_bytes = block.view(row_type).reshape(-1)
        _patterns, first, inverse = np.unique(
            row_bytes,
            return_index=True,
            return_inverse=True,
        )
        aggregated = np.bincount(
            inverse,
            weights=weights,
            minlength=len(first),
        ).astype(np.float64, copy=False)
        positive = aggregated > 0.0
        return block[first[positive]], aggregated[positive]

    event_design, event_mass = aggregate(matrix[:n_events], event_weight)
    grid_design, grid_mass = aggregate(matrix[n_events:], grid_weight)
    return (
        np.concatenate((event_design, grid_design), axis=0),
        int(len(event_design)),
        event_mass,
        grid_mass,
    )


def prepare_fixed_support_design(
    ctx: QueryContext,
    controls: ControlDesign,
    rule_features: Sequence[DesignBlock],
    rules: Sequence[RuleIdentity],
    *,
    cluster_weights: np.ndarray | None = None,
    sequence_exposures: np.ndarray | None = None,
    occurrence_likelihood: str = "poisson",
) -> PreparedFixedSupportDesign:
    """Assemble and group a fixed-support design exactly once."""
    likelihood = _validate_occurrence_likelihood(occurrence_likelihood)
    rules = tuple(rules)
    control_blocks = _validated_control_blocks(controls)
    control_width = sum(block.shape[1] for block in control_blocks)
    feature_widths = {int(feature.shape[1]) for feature in rule_features}
    if len(feature_widths) > 1:
        raise ValueError("every rule block must use the same kernel dimension")
    if any(block.shape[0] != ctx.n_queries for block in control_blocks):
        raise ValueError("control design/query length mismatch")
    sequence_weights = _validated_cluster_weights(ctx, cluster_weights)
    event_weights = sequence_weights[ctx.event_sequence_local]
    excluded_grid_rows: np.ndarray | None = None
    if likelihood == "first_event_cloglog":
        event_counts = np.bincount(
            ctx.event_sequence_local,
            minlength=ctx.n_sequences,
        )
        if np.any(event_counts > 1):
            raise ValueError(
                "first_event_cloglog requires at most one target per sequence"
            )
        event_grid_rows = np.asarray(ctx.event_grid_rows, dtype=np.int64)
        if len(event_grid_rows) != ctx.n_events or len(np.unique(event_grid_rows)) != len(event_grid_rows):
            raise ValueError("first-event targets must map one-to-one to reporting cells")
        event_cell_widths = ctx.grid_weights_at(event_grid_rows, assume_valid=True)
        if np.any(event_cell_widths != 1.0):
            raise ValueError("first_event_cloglog currently requires unit reporting intervals")
        excluded_grid_rows = event_grid_rows
    exposures = (
        ctx.sequence_exposures()
        if sequence_exposures is None
        else np.asarray(sequence_exposures, dtype=np.float64)
    )
    if exposures.shape != (ctx.n_sequences,):
        raise ValueError("sequence exposure length mismatch")
    weighted_grid_exposure = float(exposures @ sequence_weights)
    if likelihood == "first_event_cloglog":
        # Event reporting cells contribute a Bernoulli/cloglog event term, not
        # the no-event survival term represented by the grid compensator.
        weighted_grid_exposure -= float(np.sum(event_weights, dtype=np.float64))
        if weighted_grid_exposure <= 0.0:
            raise ValueError("first-event fit requires positive no-event exposure")
    design, grid_weights, constrained_start, active_grid_rows = (
        assemble_compressed_design(
            control_blocks,
            rule_features,
            rules,
            n_events=ctx.n_events,
            sequence_weights=sequence_weights,
            weighted_exposure=weighted_grid_exposure,
            grid_context=ctx,
            excluded_grid_rows=excluded_grid_rows,
        )
    )
    design, n_events, event_weights, grid_weights = aggregate_duplicate_design_rows(
        design,
        ctx.n_events,
        event_weights,
        grid_weights,
        inplace=True,
    )
    knot_count = int(rule_features[0].shape[1]) if rule_features else 0
    return PreparedFixedSupportDesign(
        design=design,
        n_events=int(n_events),
        event_weights=event_weights,
        grid_weights=grid_weights,
        constrained_start=int(constrained_start),
        control_width=int(control_width),
        knot_count=knot_count,
        active_grid_rows=int(active_grid_rows),
        rules=rules,
        occurrence_likelihood=likelihood,
    )


def _prepared_objective_grad_hessian_numpy(
    prepared: PreparedFixedSupportDesign,
    values: np.ndarray,
    *,
    need_hessian: bool = True,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    """Exact objective derivatives for either supported occurrence model."""
    design = prepared.design.astype(np.float64, copy=False)
    values = np.asarray(values, dtype=np.float64)
    event_design = design[: prepared.n_events]
    grid_design = design[prepared.n_events :]
    event_eta = event_design @ values
    grid_eta = grid_design @ values
    with np.errstate(over="ignore", invalid="ignore"):
        grid_mu = prepared.grid_weights * np.exp(grid_eta)
    objective = float(np.sum(grid_mu, dtype=np.float64))
    gradient = grid_design.T @ grid_mu
    hessian = (
        grid_design.T @ (grid_design * grid_mu[:, None])
        if need_hessian
        else None
    )
    if prepared.occurrence_likelihood == "poisson":
        if prepared.n_events:
            event_sufficient = event_design.T @ prepared.event_weights
            objective -= float(np.dot(prepared.event_weights, event_eta))
            gradient -= event_sufficient
        return objective, gradient, hessian
    event_loss, event_gradient, event_hessian = _cloglog_event_terms_numpy(event_eta)
    weighted_event_gradient = prepared.event_weights * event_gradient
    objective += float(np.dot(prepared.event_weights, event_loss))
    gradient += event_design.T @ weighted_event_gradient
    if need_hessian and hessian is not None:
        hessian += event_design.T @ (
            event_design * (prepared.event_weights * event_hessian)[:, None]
        )
    return objective, gradient, hessian


def _prepared_objective_numpy(
    prepared: PreparedFixedSupportDesign,
    values: np.ndarray,
) -> float:
    design = prepared.design.astype(np.float64, copy=False)
    event_eta = design[: prepared.n_events] @ values
    grid_eta = design[prepared.n_events :] @ values
    with np.errstate(over="ignore", invalid="ignore"):
        objective = float(
            np.dot(prepared.grid_weights, np.exp(grid_eta))
        )
    if prepared.occurrence_likelihood == "poisson":
        return objective - float(np.dot(prepared.event_weights, event_eta))
    return objective + float(
        np.dot(prepared.event_weights, cloglog_event_nll(event_eta))
    )


def _prepared_intercept_optimum(prepared: PreparedFixedSupportDesign) -> tuple[float, float]:
    events = float(np.sum(prepared.event_weights, dtype=np.float64))
    noevent = float(np.sum(prepared.grid_weights, dtype=np.float64))
    if events <= 0.0 or noevent <= 0.0:
        raise ValueError("a finite occurrence fit requires event and no-event mass")
    if prepared.occurrence_likelihood == "poisson":
        alpha = math.log(events) - math.log(noevent)
        return alpha, events * (1.0 - math.log(events / noevent))
    integrated_hazard = math.log1p(events / noevent)
    alpha = math.log(integrated_hazard)
    event_loss = float(cloglog_event_nll(np.asarray([alpha]))[0])
    return alpha, noevent * integrated_hazard + events * event_loss


def fixed_support_projected_kkt(
    prepared: PreparedFixedSupportDesign,
    fit: FitResult,
    *,
    tolerance: float,
) -> tuple[bool, float, float]:
    """Certify the configured full-cone KKT condition in host float64."""
    if tuple(fit.rules) != tuple(prepared.rules):
        raise ValueError("KKT fit/rule identities do not match the prepared design")
    design = prepared.design.astype(np.float64, copy=False)
    p = int(design.shape[1])
    values = np.zeros(p, dtype=np.float64)
    values[0] = float(fit.alpha)
    if len(fit.gamma) != prepared.control_width:
        raise ValueError("KKT fit/control dimension mismatch")
    values[1 : prepared.constrained_start] = fit.gamma
    expected_theta = p - prepared.constrained_start
    if fit.theta.size != expected_theta:
        raise ValueError("KKT fit/rule dimension mismatch")
    values[prepared.constrained_start :] = fit.theta.reshape(-1)
    objective, gradient, hessian = _prepared_objective_grad_hessian_numpy(
        prepared,
        values,
    )
    if (
        not math.isfinite(objective)
        or np.any(~np.isfinite(gradient))
        or hessian is None
        or np.any(~np.isfinite(hessian))
    ):
        return False, math.inf, math.inf
    fisher_diagonal = np.diag(hessian)
    projected = gradient.copy()
    if prepared.constrained_start < p:
        theta = values[prepared.constrained_start :]
        boundary_scale = max(1.0, float(np.max(np.abs(theta), initial=0.0)))
        boundary_tolerance = (
            np.finfo(np.float64).eps * max(1, len(theta)) * boundary_scale
        )
        projected[prepared.constrained_start :] = np.where(
            theta > boundary_tolerance,
            gradient[prepared.constrained_start :],
            np.minimum(gradient[prepared.constrained_start :], 0.0),
        )
    scale = np.sqrt(np.maximum(fisher_diagonal, np.finfo(np.float64).tiny))
    residual = float(np.max(np.abs(projected) / scale, initial=0.0))
    return bool(math.isfinite(residual) and residual <= float(tolerance)), residual, objective


def _objective_grad_hessian(
    x: object,
    params: object,
    n_events: int,
    event_weights: object,
    grid_weights: object,
) -> tuple[object, object, object, object]:
    eta = x.matmul(params)
    event_term = (
        torch.dot(event_weights, eta[:n_events])
        if n_events
        else torch.zeros((), dtype=x.dtype, device=x.device)
    )
    grid_eta = eta[n_events:]
    intensity_weight = grid_weights * torch.exp(grid_eta)
    objective = torch.sum(intensity_weight) - event_term
    grad = x[n_events:].T.matmul(intensity_weight)
    if n_events:
        grad = grad - x[:n_events].T.matmul(event_weights)
    hessian = x[n_events:].T.matmul(x[n_events:] * intensity_weight.reshape(-1, 1))
    return objective, grad, hessian, eta


def _objective_only(
    x: object,
    params: object,
    n_events: int,
    event_weights: object,
    grid_weights: object,
) -> object:
    """Poisson-process objective without trial-step gradient/Hessian work."""
    eta = x.matmul(params)
    event_term = (
        torch.dot(event_weights, eta[:n_events])
        if n_events
        else torch.zeros((), dtype=x.dtype, device=x.device)
    )
    grid_eta = eta[n_events:]
    return torch.sum(grid_weights * torch.exp(grid_eta)) - event_term


def _objective_at_eta(
    eta: object,
    n_events: int,
    event_weights: object,
    grid_weights: object,
) -> object:
    """Poisson objective for a precomputed predictor."""
    event_term = (
        torch.dot(event_weights, eta[:n_events])
        if n_events
        else torch.zeros((), dtype=eta.dtype, device=eta.device)
    )
    return torch.sum(grid_weights * torch.exp(eta[n_events:])) - event_term


def _fit_fixed_support_numpy64(
    prepared: PreparedFixedSupportDesign,
    rules: Sequence[RuleIdentity],
    closure_terms: Sequence[ClosureTerm],
    *,
    max_iter: int,
    tolerance: float,
    initial: FitResult | None,
) -> FitResult:
    """Host-float64 active-set Newton without tensor launch/sync overhead."""
    rules = tuple(rules)
    x = prepared.design.astype(np.float64, copy=False)
    ew = prepared.event_weights.astype(np.float64, copy=False)
    gw = prepared.grid_weights.astype(np.float64, copy=False)
    n_events = int(prepared.n_events)
    constrained_start = int(prepared.constrained_start)
    p = int(x.shape[1])
    grid_design = x[n_events:]
    values = np.zeros(p, dtype=np.float64)
    intercept, intercept_value = _prepared_intercept_optimum(prepared)
    values[0] = intercept
    default_values = values.copy()
    if initial is not None:
        values[0] = float(initial.alpha)
        if len(initial.gamma) == prepared.control_width and prepared.control_width:
            values[1:constrained_start] = initial.gamma
        if initial.theta.size and rules:
            knot_count = int(prepared.knot_count)
            theta_view = values[constrained_start:].reshape(len(rules), knot_count)
            initial_index = {rule: index for index, rule in enumerate(initial.rules)}
            for new_index, rule in enumerate(rules):
                old_index = initial_index.get(rule)
                if old_index is not None and initial.theta.shape[1] == knot_count:
                    theta_view[new_index] = initial.theta[old_index]
        warm_value = _prepared_objective_numpy(prepared, values)
        if not math.isfinite(warm_value) or warm_value > intercept_value:
            values = default_values

    # The loop and the returned convergence flag must enforce the same public
    # KKT tolerance.  A hidden sqrt(eps) early-stop previously terminated a fit
    # and then reported it unconverged when callers requested a stricter value.
    effective_tolerance = float(tolerance)
    iterations = 0
    converged = False
    final_kkt = math.inf
    final_state_valid = False
    for iterations in range(1, int(max_iter) + 1):
        objective, gradient, hessian_or_none = (
            _prepared_objective_grad_hessian_numpy(prepared, values)
        )
        if hessian_or_none is None:
            raise RuntimeError("fixed-support Hessian was not computed")
        hessian = hessian_or_none
        final_state_valid = True
        if (
            not math.isfinite(objective)
            or np.any(~np.isfinite(gradient))
            or np.any(~np.isfinite(hessian))
        ):
            raise FloatingPointError("nonfinite fixed-support objective")
        projected = gradient.copy()
        boundary_tolerance = 0.0
        if constrained_start < p:
            theta = values[constrained_start:]
            boundary_scale = max(1.0, float(np.max(np.abs(theta), initial=0.0)))
            boundary_tolerance = (
                np.finfo(np.float64).eps * max(1, len(theta)) * boundary_scale
            )
            projected[constrained_start:] = np.where(
                theta > boundary_tolerance,
                gradient[constrained_start:],
                np.minimum(gradient[constrained_start:], 0.0),
            )
        fisher_scale = np.sqrt(
            np.maximum(np.diag(hessian), np.finfo(np.float64).tiny)
        )
        final_kkt = float(
            np.max(np.abs(projected) / fisher_scale, initial=0.0)
        )
        if final_kkt <= effective_tolerance:
            converged = True
            break

        active = np.ones(p, dtype=bool)
        if constrained_start < p:
            active[constrained_start:] = (
                (values[constrained_start:] > boundary_tolerance)
                | (gradient[constrained_start:] < -boundary_tolerance)
            )
        active_index = np.flatnonzero(active)
        active_hessian = hessian[np.ix_(active_index, active_index)]
        active_gradient = gradient[active_index]
        diagonal_scale = np.sqrt(
            np.maximum(np.diag(active_hessian), np.finfo(np.float64).tiny)
        )
        standardized_hessian = active_hessian / np.outer(
            diagonal_scale, diagonal_scale
        )
        standardized_gradient = active_gradient / diagonal_scale
        try:
            # Full-rank Fisher systems dominate accepted fits. Cholesky gives
            # the same Newton direction without a complete eigendecomposition;
            # singular/semidefinite systems retain the rank-aware eigen path.
            factor = np.linalg.cholesky(standardized_hessian)
            standardized_direction = np.linalg.solve(
                factor.T,
                np.linalg.solve(factor, -standardized_gradient),
            )
        except np.linalg.LinAlgError:
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(standardized_hessian)
                largest = max(
                    float(np.max(eigenvalues, initial=0.0)),
                    np.finfo(np.float64).tiny,
                )
                rank_tolerance = (
                    np.finfo(np.float64).eps
                    * max(1, len(active_index))
                    * largest
                )
                inverse = np.divide(
                    1.0,
                    eigenvalues,
                    out=np.zeros_like(eigenvalues),
                    where=eigenvalues > rank_tolerance,
                )
                standardized_direction = -eigenvectors @ (
                    inverse * (eigenvectors.T @ standardized_gradient)
                )
            except np.linalg.LinAlgError:
                standardized_direction = -standardized_gradient / np.maximum(
                    np.diag(standardized_hessian),
                    np.finfo(np.float64).tiny,
                )
        active_direction = standardized_direction / diagonal_scale
        direction = np.zeros(p, dtype=np.float64)
        direction[active_index] = active_direction
        if constrained_start < p:
            blocked = (
                values[constrained_start:] <= boundary_tolerance
            ) & (direction[constrained_start:] < 0.0)
            direction[constrained_start:][blocked] = 0.0
        slope = float(gradient @ direction)
        if not math.isfinite(slope) or slope >= 0.0:
            direction = -projected
            slope = float(gradient @ direction)

        max_step = 1.0
        if constrained_start < p:
            negative = (
                direction[constrained_start:] < 0.0
            ) & (values[constrained_start:] > boundary_tolerance)
            if np.any(negative):
                max_step = min(
                    max_step,
                    float(
                        np.min(
                            -values[constrained_start:][negative]
                            / direction[constrained_start:][negative]
                        )
                    ),
                )
        step = max(max_step, 1.0e-12)
        accepted = False
        for _line_iteration in range(40):
            trial = values + step * direction
            clamped = False
            if constrained_start < p:
                constrained_trial = np.maximum(
                    trial[constrained_start:], 0.0
                )
                clamped = bool(
                    np.any(constrained_trial != trial[constrained_start:])
                )
                if clamped:
                    trial[constrained_start:] = constrained_trial
            actual_slope = (
                float(gradient @ (trial - values)) if clamped else step * slope
            )
            trial_objective = _prepared_objective_numpy(prepared, trial)
            if (
                math.isfinite(trial_objective)
                and trial_objective <= objective + 1.0e-4 * actual_slope
            ):
                values = trial
                final_state_valid = False
                accepted = True
                break
            step *= 0.5
        if not accepted:
            largest = float(np.linalg.norm(hessian, ord=2))
            pg_step = 1.0 / max(largest, 1.0e-8)
            for _line_iteration in range(40):
                trial = values - pg_step * projected
                if constrained_start < p:
                    trial[constrained_start:] = np.maximum(
                        trial[constrained_start:], 0.0
                    )
                trial_objective = _prepared_objective_numpy(prepared, trial)
                if math.isfinite(trial_objective) and trial_objective < objective:
                    values = trial
                    final_state_valid = False
                    accepted = True
                    break
                pg_step *= 0.5
            if not accepted:
                break

    if final_state_valid:
        fisher_diagonal = np.diag(hessian)
        final_objective = objective
    else:
        final_objective, gradient, final_hessian = (
            _prepared_objective_grad_hessian_numpy(prepared, values)
        )
        if final_hessian is None:
            raise RuntimeError("fixed-support Hessian was not computed")
        fisher_diagonal = np.diag(final_hessian)
    projected = gradient.copy()
    if constrained_start < p:
        theta_values = values[constrained_start:]
        boundary_scale = max(
            1.0, float(np.max(np.abs(theta_values), initial=0.0))
        )
        boundary_tolerance = (
            np.finfo(np.float64).eps
            * max(1, len(theta_values))
            * boundary_scale
        )
        projected[constrained_start:] = np.where(
            theta_values > boundary_tolerance,
            gradient[constrained_start:],
            np.minimum(gradient[constrained_start:], 0.0),
        )
    final_kkt = float(
        np.max(
            np.abs(projected)
            / np.sqrt(np.maximum(fisher_diagonal, np.finfo(np.float64).tiny)),
            initial=0.0,
        )
    )
    nll = float(final_objective)
    knot_count = int(prepared.knot_count)
    theta = (
        values[constrained_start:].reshape(len(rules), knot_count).copy()
        if rules
        else np.zeros((0, 0), dtype=np.float64)
    )
    result = FitResult(
        rules=rules,
        closure_terms=tuple(sorted(closure_terms)),
        alpha=float(values[0]),
        gamma=values[1:constrained_start].copy(),
        theta=theta,
        nll=nll,
        kkt_residual=final_kkt,
        converged=bool(final_kkt <= float(tolerance)),
        iterations=int(iterations),
        device="cpu",
        intensity_nll=nll,
        mark_fit=None,
    )
    if not result.converged and initial is not None:
        # A feasible warm start can have a lower objective than the intercept yet
        # still lead finite-iteration Newton/Armijo into a stalled path after the
        # design changes.  Convex optimum and the public KKT contract must not
        # depend on initialization.  Retry the identical problem once from its
        # exact intercept optimum; the recursive call has ``initial=None`` and
        # therefore cannot recurse again.
        cold = _fit_fixed_support_numpy64(
            prepared,
            rules,
            closure_terms,
            max_iter=max_iter,
            tolerance=tolerance,
            initial=None,
        )
        if cold.converged or (
            not result.converged
            and (float(cold.kkt_residual), float(cold.nll))
            < (float(result.kkt_residual), float(result.nll))
        ):
            return cold
    return result


def _fit_fixed_support_native64(
    prepared: PreparedFixedSupportDesign,
    rules: Sequence[RuleIdentity],
    closure_terms: Sequence[ClosureTerm],
    *,
    max_iter: int,
    tolerance: float,
    initial: FitResult | None,
) -> FitResult | None:
    """Run the compiled active-set solver and certify it in host float64.

    This is a pure execution backend.  A result is returned only when the
    existing Python KKT implementation certifies the same prepared objective
    at the requested tolerance.  Every other status falls back to
    ``_fit_fixed_support_numpy64`` in the caller.
    """
    rules = tuple(rules)
    p = int(prepared.design.shape[1])
    constrained_start = int(prepared.constrained_start)
    values = np.zeros(p, dtype=np.float64)
    intercept, intercept_value = _prepared_intercept_optimum(prepared)
    values[0] = intercept
    default_values = values.copy()
    if initial is not None:
        values[0] = float(initial.alpha)
        if len(initial.gamma) == prepared.control_width and prepared.control_width:
            values[1:constrained_start] = initial.gamma
        if initial.theta.size and rules:
            knot_count = int(prepared.knot_count)
            theta_view = values[constrained_start:].reshape(len(rules), knot_count)
            initial_index = {rule: index for index, rule in enumerate(initial.rules)}
            for new_index, rule in enumerate(rules):
                old_index = initial_index.get(rule)
                if old_index is not None and initial.theta.shape[1] == knot_count:
                    theta_view[new_index] = initial.theta[old_index]
        if constrained_start < p:
            values[constrained_start:] = np.maximum(
                values[constrained_start:], 0.0
            )
        warm_value = _prepared_objective_numpy(prepared, values)
        if not math.isfinite(warm_value) or warm_value > intercept_value:
            values = default_values
    native = fit_prepared_cone_float64(
        prepared.design,
        prepared.n_events,
        prepared.event_weights,
        prepared.grid_weights,
        constrained_start,
        values,
        occurrence_likelihood=prepared.occurrence_likelihood,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    if native is None:
        return None
    fitted_values, _native_objective, _native_kkt, iterations = native
    knot_count = int(prepared.knot_count)
    theta = (
        fitted_values[constrained_start:].reshape(len(rules), knot_count).copy()
        if rules
        else np.zeros((0, 0), dtype=np.float64)
    )
    provisional = FitResult(
        rules=rules,
        closure_terms=tuple(sorted(closure_terms)),
        alpha=float(fitted_values[0]),
        gamma=fitted_values[1:constrained_start].copy(),
        theta=theta,
        nll=math.inf,
        kkt_residual=math.inf,
        converged=False,
        iterations=int(iterations),
        device="cpu-native",
        intensity_nll=math.inf,
        mark_fit=None,
    )
    converged, kkt, objective = fixed_support_projected_kkt(
        prepared,
        provisional,
        tolerance=float(tolerance),
    )
    if not converged:
        return None
    return replace(
        provisional,
        nll=float(objective),
        intensity_nll=float(objective),
        kkt_residual=float(kkt),
        converged=True,
    )


def fit_fixed_support(
    ctx: QueryContext,
    controls: ControlDesign,
    rule_features: Sequence[DesignBlock],
    rules: Sequence[RuleIdentity],
    *,
    device: str = "cuda",
    dtype: str = "float32",
    max_iter: int = 80,
    tolerance: float = 2.0e-5,
    initial: FitResult | None = None,
    closure_terms: Sequence[ClosureTerm] = (),
    cluster_weights: np.ndarray | None = None,
    sequence_exposures: np.ndarray | None = None,
    prepared_design: PreparedFixedSupportDesign | None = None,
    occurrence_likelihood: str = "poisson",
) -> FitResult:
    if torch is None:
        raise RuntimeError("PyTorch is required")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    rules = tuple(rules)
    control_blocks = _validated_control_blocks(controls)
    control_width = sum(block.shape[1] for block in control_blocks)
    if prepared_design is None:
        prepared_design = prepare_fixed_support_design(
            ctx,
            control_blocks,
            rule_features,
            rules,
            cluster_weights=cluster_weights,
            sequence_exposures=sequence_exposures,
            occurrence_likelihood=occurrence_likelihood,
        )
    if (
        prepared_design.control_width != control_width
        or tuple(prepared_design.rules) != rules
        or prepared_design.knot_count
        != (int(rule_features[0].shape[1]) if rule_features else 0)
    ):
        raise ValueError("prepared fixed-support design does not match model blocks")
    requested_likelihood = _validate_occurrence_likelihood(occurrence_likelihood)
    if prepared_design.occurrence_likelihood != requested_likelihood:
        raise ValueError("prepared design occurrence likelihood does not match fit")
    if prepared_design.occurrence_likelihood == "first_event_cloglog":
        # The exact interval-event derivatives are implemented in the host
        # float64 active-set solvers.  The native path is accepted only after
        # the ordinary host-float64 projected KKT check; otherwise the
        # established NumPy solver reruns the identical convex problem.
        prepared_design = promote_prepared_design_float64(prepared_design)
        native_fit = (
            _fit_fixed_support_native64(
                prepared_design,
                rules,
                closure_terms,
                max_iter=max_iter,
                tolerance=tolerance,
                initial=initial,
            )
            # Direct native loops remove launch overhead for very small GLMs.
            # Above this structural width, NumPy's optimized BLAS wins on the
            # audited host (p=17/33); the backend choice observes dimensions
            # only and both paths retain the same host KKT certificate.
            if prepared_design.design.shape[1] <= 12
            else None
        )
        if native_fit is not None:
            return native_fit
        return _fit_fixed_support_numpy64(
            prepared_design,
            rules,
            closure_terms,
            max_iter=max_iter,
            tolerance=tolerance,
            initial=initial,
        )
    if str(device).startswith("cpu") and dtype == "float64":
        prepared_design = promote_prepared_design_float64(prepared_design)
        native_fit = (
            _fit_fixed_support_native64(
                prepared_design,
                rules,
                closure_terms,
                max_iter=max_iter,
                tolerance=tolerance,
                initial=initial,
            )
            if prepared_design.design.shape[1] <= 12
            else None
        )
        if native_fit is not None:
            return native_fit
        return _fit_fixed_support_numpy64(
            prepared_design,
            rules,
            closure_terms,
            max_iter=max_iter,
            tolerance=tolerance,
            initial=initial,
        )
    solve_design = prepared_design.design
    solve_n_events = prepared_design.n_events
    event_weight_np = prepared_design.event_weights
    solve_grid_weight_np = prepared_design.grid_weights
    constrained_start = prepared_design.constrained_start
    x = torch.as_tensor(solve_design, dtype=torch_dtype, device=device)
    ew = torch.as_tensor(event_weight_np, dtype=torch_dtype, device=device)
    gw = torch.as_tensor(solve_grid_weight_np, dtype=torch_dtype, device=device)
    p = x.shape[1]
    params = torch.zeros(p, dtype=torch_dtype, device=device)
    weighted_events = float(np.sum(event_weight_np, dtype=np.float64))
    weighted_exposure = float(np.sum(solve_grid_weight_np, dtype=np.float64))
    if weighted_events <= 0.0:
        raise ValueError("a finite point-process intercept requires at least one weighted target event")
    if weighted_exposure <= 0.0:
        raise ValueError("a point-process fit requires positive weighted exposure")
    # Compute the exact intercept-only optimum in log space.  A hard lower
    # bound on the event rate slows genuinely rare-event fits and is unnecessary
    # because both validated masses are strictly positive.
    params[0] = math.log(weighted_events) - math.log(weighted_exposure)
    default_params = params.clone()
    if initial is not None:
        params[0] = float(initial.alpha)
        if len(initial.gamma) == control_width and control_width:
            params[1:constrained_start] = torch.as_tensor(initial.gamma, dtype=torch_dtype, device=device)
        if initial.theta.size and rule_features:
            knot_count = int(rule_features[0].shape[1])
            theta_view = params[constrained_start:].reshape(len(rules), knot_count)
            initial_index = {rule: idx for idx, rule in enumerate(initial.rules)}
            for new_index, rule in enumerate(rules):
                old_index = initial_index.get(rule)
                if old_index is not None and initial.theta.shape[1] == knot_count:
                    theta_view[new_index] = torch.as_tensor(
                        initial.theta[old_index],
                        dtype=torch_dtype,
                        device=device,
                    )

        # Removing an inhibitory rule can make the linear predictor of a
        # transferred full-model warm start arbitrarily large.  The target
        # objective is still well-defined: only that particular warm start is
        # unsafe.  Back off on the line segment to the finite intercept-only
        # initialization.  This changes initialization, never the fitted
        # objective or feasible parameter space.
        warm_objective = _objective_only(x, params, solve_n_events, ew, gw)
        warm_value = float(warm_objective.detach().cpu().item())
        intercept_only_value = weighted_events * (
            1.0 - math.log(weighted_events / weighted_exposure)
        )
        if not math.isfinite(warm_value) or warm_value > intercept_only_value:
            # A finite transferred fit can still be arbitrarily poor after a
            # model change.  Starting from the lower-objective feasible point
            # is an exact convex-optimization initialization rule: it changes
            # neither the objective nor its feasible minimizer.
            params = default_params

    converged = False
    final_kkt = math.inf
    # Float32 may leave the accelerator phase at its numerical floor, but is
    # then polished and certified in host float64 below.  Float64 must never
    # stop above the same public tolerance used by the returned KKT flag.
    effective_kkt_tolerance = (
        max(
            float(tolerance),
            math.sqrt(torch.finfo(torch_dtype).eps)
            * math.sqrt(max(1, int(p))),
        )
        if torch_dtype == torch.float32
        else float(tolerance)
    )
    iterations = 0
    final_state: tuple[object, object, object, object] | None = None
    for iterations in range(1, max_iter + 1):
        objective, grad, hessian, _eta = _objective_grad_hessian(x, params, solve_n_events, ew, gw)
        final_state = (objective, grad, hessian, _eta)
        if not bool(torch.isfinite(objective).item()) or not bool(torch.all(torch.isfinite(grad)).item()):
            raise FloatingPointError("nonfinite fixed-support objective")
        projected = grad.clone()
        if constrained_start < p:
            theta = params[constrained_start:]
            gtheta = grad[constrained_start:]
            boundary_scale = max(1.0, float(torch.max(torch.abs(theta)).detach().cpu().item()))
            boundary_tolerance = torch.finfo(torch_dtype).eps * max(1, int(theta.numel())) * boundary_scale
            projected[constrained_start:] = torch.where(
                theta > boundary_tolerance,
                gtheta,
                torch.minimum(gtheta, torch.zeros_like(gtheta)),
            )
        fisher_scale = torch.sqrt(
            torch.clamp(torch.diag(hessian), min=torch.finfo(torch_dtype).tiny)
        )
        final_kkt = float(
            torch.max(torch.abs(projected) / fisher_scale).detach().cpu().item()
        )
        if final_kkt <= effective_kkt_tolerance:
            converged = True
            break

        active = torch.ones(p, dtype=torch.bool, device=device)
        if constrained_start < p:
            active[constrained_start:] = (
                (params[constrained_start:] > boundary_tolerance)
                | (grad[constrained_start:] < -boundary_tolerance)
            )
        active_idx = torch.nonzero(active, as_tuple=False).reshape(-1)
        h_active = hessian.index_select(0, active_idx).index_select(1, active_idx)
        g_active = grad.index_select(0, active_idx)
        try:
            # Solve in Fisher-diagonally standardized coordinates.  Directly
            # thresholding raw Hessian eigenvalues makes the Newton direction
            # depend on arbitrary feature units and can erase rare kernels.
            diagonal_scale = torch.sqrt(
                torch.clamp(torch.diag(h_active), min=torch.finfo(torch_dtype).tiny)
            )
            h_standardized = h_active / (
                diagonal_scale.reshape(-1, 1) * diagonal_scale.reshape(1, -1)
            )
            g_standardized = g_active / diagonal_scale
            eigenvalues, eigenvectors = torch.linalg.eigh(h_standardized)
            largest = torch.max(eigenvalues) if eigenvalues.numel() else torch.tensor(0.0, dtype=torch_dtype, device=device)
            rank_tolerance = (
                torch.finfo(torch_dtype).eps
                * max(1.0, float(len(active_idx)))
                * torch.clamp(largest, min=torch.finfo(torch_dtype).tiny)
            )
            inverse = torch.where(eigenvalues > rank_tolerance, 1.0 / eigenvalues, torch.zeros_like(eigenvalues))
            d_standardized = -eigenvectors.matmul(
                inverse * eigenvectors.T.matmul(g_standardized)
            )
            d_active = d_standardized / diagonal_scale
        except RuntimeError:
            d_active = -g_active / torch.clamp(
                torch.diag(h_active), min=torch.finfo(torch_dtype).tiny
            )
        direction = torch.zeros_like(params)
        direction[active_idx] = d_active
        if constrained_start < p:
            theta = params[constrained_start:]
            dtheta = direction[constrained_start:]
            # A coupled Newton solve can point an active coordinate out of the
            # feasible cone at theta=0.  Such a coordinate must be held at the
            # boundary; allowing it into the ratio test would collapse the
            # step for every other coordinate to approximately zero.
            blocked = (theta <= boundary_tolerance) & (dtheta < 0)
            dtheta[blocked] = 0.0
        directional_derivative = torch.dot(grad, direction)
        if not bool(torch.isfinite(directional_derivative).item()) or float(directional_derivative.item()) >= 0.0:
            direction = -projected
            directional_derivative = torch.dot(grad, direction)

        max_step = 1.0
        if constrained_start < p:
            theta = params[constrained_start:]
            dtheta = direction[constrained_start:]
            negative = (dtheta < 0) & (theta > boundary_tolerance)
            if bool(torch.any(negative).item()):
                bound = torch.min(-theta[negative] / dtheta[negative])
                # Step exactly to the first cone boundary.  The former 0.995
                # fraction-to-boundary rule left the same coefficient barely
                # positive and repeatedly solved an unchanged active system.
                # Projection below handles roundoff; the feasible set and KKT
                # solution are unchanged.
                max_step = min(max_step, float(bound.detach().cpu().item()))
        accepted = False
        current_value = float(objective.detach().cpu().item())
        slope = float(directional_derivative.detach().cpu().item())
        # X @ direction is invariant throughout Armijo backtracking.  Reusing
        # it avoids one full design matvec for every rejected trial step.
        direction_eta = x.matmul(direction)
        step = max(max_step, 1.0e-12)
        for _ in range(40):
            raw_trial = params + step * direction
            trial = raw_trial
            clamped = False
            if constrained_start < p:
                constrained_trial = torch.clamp(
                    raw_trial[constrained_start:], min=0.0
                )
                clamped = bool(
                    torch.any(
                        constrained_trial != raw_trial[constrained_start:]
                    ).item()
                )
                if clamped:
                    trial = raw_trial.clone()
                    trial[constrained_start:] = constrained_trial
            # Fraction-to-boundary normally makes the clamp a no-op, allowing
            # Xd reuse.  If the minimum trial step crosses an extremely small
            # coefficient through zero, evaluate the predictor and Armijo
            # slope at the parameters that will actually be stored.
            if clamped:
                actual_delta = trial - params
                trial_eta = _eta + x.matmul(actual_delta)
                armijo_delta = float(torch.dot(grad, actual_delta).item())
            else:
                trial_eta = _eta + step * direction_eta
                armijo_delta = step * slope
            trial_obj = _objective_at_eta(
                trial_eta,
                solve_n_events,
                ew,
                gw,
            )
            if bool(torch.isfinite(trial_obj).item()) and float(trial_obj.item()) <= current_value + 1.0e-4 * armijo_delta:
                params = trial
                final_state = None
                accepted = True
                break
            step *= 0.5
        if not accepted:
            # Globally safe projected-gradient fallback using the local Hessian
            # spectral bound. Backtracking below still verifies descent.
            largest = float(torch.linalg.norm(hessian, ord=2).detach().cpu().item())
            pg_step = 1.0 / max(largest, 1.0e-8)
            for _ in range(40):
                trial = params - pg_step * projected
                if constrained_start < p:
                    trial[constrained_start:] = torch.clamp(trial[constrained_start:], min=0.0)
                trial_obj = _objective_only(x, trial, solve_n_events, ew, gw)
                if bool(torch.isfinite(trial_obj).item()) and float(trial_obj.item()) < current_value:
                    params = trial
                    final_state = None
                    accepted = True
                    break
                pg_step *= 0.5
            if not accepted:
                break

    # A converged/no-descent exit has already evaluated derivatives at the
    # final parameters.  Reusing that exact tensor state saves one complete
    # design matvec and Hessian construction per fit without changing a single
    # objective, constraint, or stopping decision.
    if final_state is None:
        objective, grad, final_hessian, eta = _objective_grad_hessian(
            x, params, solve_n_events, ew, gw
        )
    else:
        objective, grad, final_hessian, eta = final_state
    projected = grad.clone()
    if constrained_start < p:
        final_theta = params[constrained_start:]
        boundary_scale = max(1.0, float(torch.max(torch.abs(final_theta)).detach().cpu().item()))
        boundary_tolerance = torch.finfo(torch_dtype).eps * max(1, int(final_theta.numel())) * boundary_scale
        projected[constrained_start:] = torch.where(
            final_theta > boundary_tolerance,
            grad[constrained_start:],
            torch.minimum(grad[constrained_start:], torch.zeros_like(grad[constrained_start:])),
        )
    final_fisher_scale = torch.sqrt(
        torch.clamp(torch.diag(final_hessian), min=torch.finfo(torch_dtype).tiny)
    )
    final_kkt = float(
        torch.max(torch.abs(projected) / final_fisher_scale).detach().cpu().item()
    )
    converged = bool(final_kkt <= effective_kkt_tolerance)
    values = params.detach().cpu().to(torch.float64).numpy()
    gamma = values[1:constrained_start].copy()
    knot_count = rule_features[0].shape[1] if rule_features else 0
    theta = values[constrained_start:].reshape(len(rules), knot_count).copy() if rules else np.zeros((0, 0), dtype=np.float64)
    # Re-evaluate in float64 on the host without materializing a second full
    # float64 design matrix; this is also the persisted objective.
    solve_eta64 = solve_design @ values
    solve_nll64 = float(
        np.dot(solve_grid_weight_np, np.exp(solve_eta64[solve_n_events :]))
        - np.dot(event_weight_np, solve_eta64[:solve_n_events])
    )
    polish_updates = 0
    if torch_dtype == torch.float32 and final_kkt > float(tolerance):
        # Float32 reductions over millions of rows can stall a few ulps from
        # the optimum.  Polish only such stalled fits with a small, chunked
        # host-float64 Newton solve.  The full design remains float32; only a
        # bounded block and the p-by-p Fisher matrix are promoted at once.
        grid_design = solve_design[solve_n_events:]
        chunk_size = 1_000_000
        event_design64 = (
            solve_design[:solve_n_events].astype(np.float64, copy=False)
            if solve_n_events
            else np.zeros((0, p), dtype=np.float64)
        )
        converged = False
        # Use the remaining configured Newton-update budget, followed by one
        # derivative-only pass.  A fixed small polishing count can reject a
        # valid rare rule merely because float32 stopped a few ulps early.
        max_polish_updates = max(0, int(max_iter) - int(iterations))
        for _polish_iteration in range(max_polish_updates + 1):
            grad64 = np.zeros(p, dtype=np.float64)
            fisher64 = np.zeros((p, p), dtype=np.float64)
            if solve_n_events:
                grad64 -= event_design64.T @ event_weight_np
            grid_mu64 = solve_grid_weight_np * np.exp(solve_eta64[solve_n_events:])
            for left in range(0, len(grid_design), chunk_size):
                right = min(left + chunk_size, len(grid_design))
                block = grid_design[left:right].astype(np.float64, copy=False)
                weights_block = grid_mu64[left:right]
                grad64 += block.T @ weights_block
                fisher64 += block.T @ (block * weights_block.reshape(-1, 1))
            projected64 = grad64.copy()
            boundary_tolerance64 = 0.0
            if constrained_start < p:
                theta64 = values[constrained_start:]
                boundary_scale64 = max(1.0, float(np.max(np.abs(theta64), initial=0.0)))
                boundary_tolerance64 = (
                    np.finfo(np.float64).eps * max(1, len(theta64)) * boundary_scale64
                )
                projected64[constrained_start:] = np.where(
                    theta64 > boundary_tolerance64,
                    grad64[constrained_start:],
                    np.minimum(grad64[constrained_start:], 0.0),
                )
            host_scale = np.sqrt(
                np.maximum(np.diag(fisher64), np.finfo(np.float64).tiny)
            )
            final_kkt = float(
                np.max(np.abs(projected64) / host_scale, initial=0.0)
            )
            if final_kkt <= tolerance:
                converged = True
                break
            if _polish_iteration == max_polish_updates:
                break

            active64 = np.ones(p, dtype=bool)
            if constrained_start < p:
                active64[constrained_start:] = (
                    (values[constrained_start:] > boundary_tolerance64)
                    | (grad64[constrained_start:] < -boundary_tolerance64)
                )
            active_idx64 = np.flatnonzero(active64)
            h_active64 = fisher64[np.ix_(active_idx64, active_idx64)]
            g_active64 = grad64[active_idx64]
            dscale64 = np.sqrt(
                np.maximum(np.diag(h_active64), np.finfo(np.float64).tiny)
            )
            h_standardized64 = h_active64 / np.outer(dscale64, dscale64)
            g_standardized64 = g_active64 / dscale64
            direction64 = np.zeros(p, dtype=np.float64)
            try:
                eigval64, eigvec64 = np.linalg.eigh(h_standardized64)
                rank_tol64 = (
                    np.finfo(np.float64).eps
                    * max(1, len(active_idx64))
                    * max(float(np.max(eigval64, initial=0.0)), np.finfo(np.float64).tiny)
                )
                inverse64 = np.divide(
                    1.0,
                    eigval64,
                    out=np.zeros_like(eigval64),
                    where=eigval64 > rank_tol64,
                )
                d_standardized64 = -eigvec64 @ (
                    inverse64 * (eigvec64.T @ g_standardized64)
                )
                direction64[active_idx64] = d_standardized64 / dscale64
            except np.linalg.LinAlgError:
                direction64[active_idx64] = -g_active64 / np.maximum(
                    np.diag(h_active64), np.finfo(np.float64).tiny
                )
            if constrained_start < p:
                blocked64 = (
                    values[constrained_start:] <= boundary_tolerance64
                ) & (direction64[constrained_start:] < 0.0)
                direction64[constrained_start:][blocked64] = 0.0
            slope64 = float(grad64 @ direction64)
            if not math.isfinite(slope64) or slope64 >= 0.0:
                direction64 = -projected64 / np.maximum(
                    np.diag(fisher64), np.finfo(np.float64).tiny
                )
                slope64 = float(grad64 @ direction64)
            max_step64 = 1.0
            if constrained_start < p:
                negative64 = (
                    direction64[constrained_start:] < 0.0
                ) & (values[constrained_start:] > boundary_tolerance64)
                if np.any(negative64):
                    max_step64 = min(
                        max_step64,
                        float(
                            np.min(
                                -values[constrained_start:][negative64]
                                / direction64[constrained_start:][negative64]
                            )
                        ),
                    )
            step64 = max(max_step64, 1.0e-12)
            accepted64 = False
            for _ in range(40):
                trial_values64 = values + step64 * direction64
                if constrained_start < p:
                    trial_values64[constrained_start:] = np.maximum(
                        trial_values64[constrained_start:], 0.0
                    )
                # Clamping can only change constrained coordinates; recompute
                # the exact direction when a boundary is reached.
                actual_delta64 = trial_values64 - values
                trial_eta64 = solve_eta64 + solve_design @ actual_delta64
                with np.errstate(over="ignore", invalid="ignore"):
                    trial_nll64 = float(
                        np.dot(solve_grid_weight_np, np.exp(trial_eta64[solve_n_events:]))
                        - np.dot(event_weight_np, trial_eta64[:solve_n_events])
                    )
                if math.isfinite(trial_nll64) and trial_nll64 <= solve_nll64 + 1.0e-4 * step64 * slope64:
                    values = trial_values64
                    solve_eta64 = trial_eta64
                    solve_nll64 = trial_nll64
                    accepted64 = True
                    polish_updates += 1
                    break
                step64 *= 0.5
            if not accepted64:
                break
        gamma = values[1:constrained_start].copy()
        theta = (
            values[constrained_start:].reshape(len(rules), knot_count).copy()
            if rules
            else np.zeros((0, 0), dtype=np.float64)
        )
    # ``assemble_compressed_design`` is an exact sufficient-statistic
    # reduction: every omitted row has the same intercept-only predictor and
    # its weights were summed into the final zero row.  Rebuilding the entire
    # full-grid eta here duplicated an O(NM) scan after every support fit.
    nll64 = float(solve_nll64)
    # The public convergence contract is always the user-specified KKT
    # tolerance.  The looser float32 floor only decides when to leave the GPU
    # phase and enter mixed-precision refinement; it must never certify a fit.
    converged = bool(final_kkt <= float(tolerance))
    return FitResult(
        rules=rules,
        closure_terms=tuple((tuple(antecedent), int(window)) for antecedent, window in closure_terms),
        alpha=float(values[0]),
        gamma=gamma,
        theta=theta,
        nll=float(nll64),
        kkt_residual=final_kkt,
        converged=bool(converged),
        iterations=int(iterations + polish_updates),
        device=str(device),
        intensity_nll=float(nll64),
        mark_fit=None,
    )


def fit_unconstrained_prepared_batch(
    ctx: QueryContext,
    controls: Sequence[ControlDesign],
    prepared_designs: Sequence[PreparedFixedSupportDesign],
    closure_terms: Sequence[Sequence[ClosureTerm]],
    *,
    device: str = "cpu",
    dtype: str = "float64",
    max_iter: int = 80,
    tolerance: float = 2.0e-5,
) -> list[FitResult]:
    """Solve independent unconstrained Poisson GLMs in one exact batch.

    Identity profiling fits many hierarchy-null models with equal parameter
    width and different formation windows.  Padding zero-weight rows lets one
    batched Newton/eigendecomposition evaluate those *same* objectives while
    amortizing Python, BLAS and accelerator-launch overhead.  Any item that
    does not pass the ordinary host-float64 KKT check is rerun by
    :func:`fit_fixed_support`, so batching cannot weaken the solver contract.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required")
    batch_size = len(prepared_designs)
    if not (
        batch_size == len(controls) == len(closure_terms)
    ):
        raise ValueError("batched null-fit inputs must have equal length")
    if batch_size == 0:
        return []
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    np_dtype = np.float64 if torch_dtype == torch.float64 else np.float32

    widths = {int(item.design.shape[1]) for item in prepared_designs}
    if len(widths) != 1:
        raise ValueError("a batched null solve requires one parameter width")
    p = widths.pop()
    for item in prepared_designs:
        if item.rules or item.constrained_start != p:
            raise ValueError("batched solver accepts unconstrained null models only")
        if item.occurrence_likelihood != "poisson":
            raise ValueError("batched null solver currently accepts Poisson designs only")

    max_event_rows = max(int(item.n_events) for item in prepared_designs)
    max_grid_rows = max(
        int(len(item.grid_weights)) for item in prepared_designs
    )
    event_x_np = np.zeros(
        (batch_size, max_event_rows, p), dtype=np_dtype
    )
    grid_x_np = np.zeros(
        (batch_size, max_grid_rows, p), dtype=np_dtype
    )
    event_w_np = np.zeros((batch_size, max_event_rows), dtype=np_dtype)
    grid_w_np = np.zeros((batch_size, max_grid_rows), dtype=np_dtype)
    for index, item in enumerate(prepared_designs):
        event_count = int(item.n_events)
        grid_count = int(len(item.grid_weights))
        event_x_np[index, :event_count] = item.design[:event_count]
        grid_x_np[index, :grid_count] = item.design[event_count:]
        event_w_np[index, :event_count] = item.event_weights
        grid_w_np[index, :grid_count] = item.grid_weights

    event_x = torch.as_tensor(event_x_np, dtype=torch_dtype, device=device)
    grid_x = torch.as_tensor(grid_x_np, dtype=torch_dtype, device=device)
    event_w = torch.as_tensor(event_w_np, dtype=torch_dtype, device=device)
    grid_w = torch.as_tensor(grid_w_np, dtype=torch_dtype, device=device)
    weighted_events = torch.sum(event_w, dim=1)
    weighted_exposure = torch.sum(grid_w, dim=1)
    if bool(torch.any(weighted_events <= 0).item()) or bool(
        torch.any(weighted_exposure <= 0).item()
    ):
        raise ValueError("batched point-process fits require event and exposure mass")

    params = torch.zeros((batch_size, p), dtype=torch_dtype, device=device)
    params[:, 0] = torch.log(weighted_events) - torch.log(weighted_exposure)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    failed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    iteration_count = torch.zeros(batch_size, dtype=torch.int64, device=device)
    epsilon = torch.finfo(torch_dtype).eps
    effective_tolerance = float(tolerance)

    for iteration in range(1, int(max_iter) + 1):
        active_index = torch.nonzero(
            (~done) & (~failed), as_tuple=False
        ).reshape(-1)
        if not active_index.numel():
            break
        active_event_x = event_x.index_select(0, active_index)
        active_grid_x = grid_x.index_select(0, active_index)
        active_event_w = event_w.index_select(0, active_index)
        active_grid_w = grid_w.index_select(0, active_index)
        active_params = params.index_select(0, active_index)
        event_eta = torch.bmm(
            active_event_x, active_params.unsqueeze(2)
        ).squeeze(2)
        grid_eta = torch.bmm(
            active_grid_x, active_params.unsqueeze(2)
        ).squeeze(2)
        grid_mu = active_grid_w * torch.exp(grid_eta)
        objective = torch.sum(grid_mu, dim=1) - torch.sum(
            active_event_w * event_eta, dim=1
        )
        gradient = torch.bmm(
            active_grid_x.transpose(1, 2), grid_mu.unsqueeze(2)
        ).squeeze(2) - torch.bmm(
            active_event_x.transpose(1, 2), active_event_w.unsqueeze(2)
        ).squeeze(2)
        hessian = torch.bmm(
            active_grid_x.transpose(1, 2),
            active_grid_x * grid_mu.unsqueeze(2),
        )
        fisher_scale = torch.sqrt(
            torch.clamp(torch.diagonal(hessian, dim1=1, dim2=2), min=epsilon)
        )
        residual = torch.amax(torch.abs(gradient) / fisher_scale, dim=1)
        finite = (
            torch.isfinite(objective)
            & torch.all(torch.isfinite(gradient), dim=1)
            & torch.all(torch.all(torch.isfinite(hessian), dim=2), dim=1)
        )
        newly_done_local = finite & (residual <= effective_tolerance)
        newly_failed_local = ~finite
        if bool(torch.any(newly_done_local).item()):
            newly_done_index = active_index[newly_done_local]
            iteration_count[newly_done_index] = iteration
            done[newly_done_index] = True
        if bool(torch.any(newly_failed_local).item()):
            failed[active_index[newly_failed_local]] = True
        work_local = torch.nonzero(
            finite & (~newly_done_local), as_tuple=False
        ).reshape(-1)
        if not work_local.numel():
            continue
        work_index = active_index.index_select(0, work_local)
        active_hessian = hessian.index_select(0, work_local)
        active_gradient = gradient.index_select(0, work_local)
        diagonal_scale = torch.sqrt(
            torch.clamp(
                torch.diagonal(active_hessian, dim1=1, dim2=2),
                min=epsilon,
            )
        )
        standardized_hessian = active_hessian / (
            diagonal_scale.unsqueeze(2) * diagonal_scale.unsqueeze(1)
        )
        standardized_gradient = active_gradient / diagonal_scale
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(standardized_hessian)
            largest = torch.amax(eigenvalues, dim=1)
            rank_tolerance = (
                epsilon
                * max(1.0, float(p))
                * torch.clamp(largest, min=epsilon)
            )
            inverse = torch.where(
                eigenvalues > rank_tolerance.unsqueeze(1),
                1.0 / eigenvalues,
                torch.zeros_like(eigenvalues),
            )
            projected_gradient = torch.bmm(
                eigenvectors.transpose(1, 2),
                standardized_gradient.unsqueeze(2),
            ).squeeze(2)
            standardized_direction = -torch.bmm(
                eigenvectors,
                (inverse * projected_gradient).unsqueeze(2),
            ).squeeze(2)
            active_direction = standardized_direction / diagonal_scale
        except RuntimeError:
            active_direction = -active_gradient / torch.clamp(
                torch.diagonal(active_hessian, dim1=1, dim2=2),
                min=epsilon,
            )

        slope = torch.sum(active_gradient * active_direction, dim=1)
        bad_direction = (~torch.isfinite(slope)) | (slope >= 0.0)
        if bool(torch.any(bad_direction).item()):
            active_direction[bad_direction] = -active_gradient[bad_direction]
            slope = torch.sum(active_gradient * active_direction, dim=1)

        work_event_x = active_event_x.index_select(0, work_local)
        work_grid_x = active_grid_x.index_select(0, work_local)
        work_event_w = active_event_w.index_select(0, work_local)
        work_grid_w = active_grid_w.index_select(0, work_local)
        work_event_eta = event_eta.index_select(0, work_local)
        work_grid_eta = grid_eta.index_select(0, work_local)
        work_objective = objective.index_select(0, work_local)
        event_direction = torch.bmm(
            work_event_x, active_direction.unsqueeze(2)
        ).squeeze(2)
        grid_direction = torch.bmm(
            work_grid_x, active_direction.unsqueeze(2)
        ).squeeze(2)
        steps = torch.ones(len(work_index), dtype=torch_dtype, device=device)
        accepted = torch.zeros(len(work_index), dtype=torch.bool, device=device)
        for _line_iteration in range(40):
            pending_index = torch.nonzero(
                ~accepted, as_tuple=False
            ).reshape(-1)
            if not pending_index.numel():
                break
            pending_steps = steps.index_select(0, pending_index)
            trial_event_eta = work_event_eta.index_select(
                0, pending_index
            ) + pending_steps.unsqueeze(1) * event_direction.index_select(
                0, pending_index
            )
            trial_grid_eta = work_grid_eta.index_select(
                0, pending_index
            ) + pending_steps.unsqueeze(1) * grid_direction.index_select(
                0, pending_index
            )
            trial_objective = torch.sum(
                work_grid_w.index_select(0, pending_index)
                * torch.exp(trial_grid_eta),
                dim=1,
            ) - torch.sum(
                work_event_w.index_select(0, pending_index)
                * trial_event_eta,
                dim=1,
            )
            acceptable_pending = (
                torch.isfinite(trial_objective)
                & (
                    trial_objective
                    <= work_objective.index_select(0, pending_index)
                    + 1.0e-4
                    * pending_steps
                    * slope.index_select(0, pending_index)
                )
            )
            if bool(torch.any(acceptable_pending).item()):
                accepted_local = pending_index[acceptable_pending]
                accepted_global = work_index[accepted_local]
                params[accepted_global] = (
                    params[accepted_global]
                    + steps[accepted_local].unsqueeze(1)
                    * active_direction[accepted_local]
                )
                accepted[accepted_local] = True
            rejected_local = pending_index[~acceptable_pending]
            steps[rejected_local] *= 0.5
        no_descent_local = torch.nonzero(
            ~accepted, as_tuple=False
        ).reshape(-1)
        if no_descent_local.numel():
            no_descent_global = work_index[no_descent_local]
            iteration_count[no_descent_global] = iteration
            failed[no_descent_global] = True

    unfinished = (~done) & (~failed)
    iteration_count[unfinished] = int(max_iter)

    values_batch = params.detach().cpu().to(torch.float64).numpy()
    iteration_values = iteration_count.detach().cpu().numpy()
    results: list[FitResult] = []
    for index, (prepared, terms, control_design) in enumerate(
        zip(prepared_designs, closure_terms, controls, strict=True)
    ):
        values = values_batch[index]
        provisional = FitResult(
            rules=(),
            closure_terms=tuple(sorted(terms)),
            alpha=float(values[0]),
            gamma=values[1:].copy(),
            theta=np.zeros((0, 0), dtype=np.float64),
            nll=math.inf,
            kkt_residual=math.inf,
            converged=False,
            iterations=int(iteration_values[index]),
            device=str(device),
            intensity_nll=math.inf,
            mark_fit=None,
        )
        converged, kkt, objective = fixed_support_projected_kkt(
            prepared,
            provisional,
            tolerance=float(tolerance),
        )
        if converged:
            results.append(
                replace(
                    provisional,
                    nll=float(objective),
                    intensity_nll=float(objective),
                    kkt_residual=float(kkt),
                    converged=True,
                )
            )
            continue
        # Fail closed to the established scalar solver.  Supplying the batch
        # iterate is only a warm start; the scalar path re-applies its ordinary
        # Armijo, cone and host-float64 KKT checks.
        results.append(
            fit_fixed_support(
                ctx,
                control_design,
                [],
                [],
                device=device,
                dtype=dtype,
                max_iter=max_iter,
                tolerance=tolerance,
                initial=provisional,
                closure_terms=terms,
                prepared_design=prepared,
            )
        )
    return results


def predict_eta(
    fit: FitResult,
    controls: np.ndarray,
    rule_features: Sequence[np.ndarray],
) -> np.ndarray:
    controls = np.asarray(controls)
    if controls.ndim != 2 or controls.shape[1] != len(fit.gamma):
        raise ValueError("fit/design mismatch")
    if len(rule_features) != len(fit.rules) or fit.theta.shape[0] != len(fit.rules):
        raise ValueError("fit/design mismatch")
    eta = np.full(controls.shape[0], float(fit.alpha), dtype=np.float64)
    if controls.shape[1]:
        eta += controls @ fit.gamma
    for index, (rule, feature) in enumerate(zip(fit.rules, rule_features, strict=True)):
        feature = np.asarray(feature)
        if (
            feature.ndim != 2
            or feature.shape[0] != controls.shape[0]
            or feature.shape[1] != fit.theta.shape[1]
        ):
            raise ValueError("fit/design mismatch")
        eta += float(rule.sign) * (feature @ fit.theta[index])
    return eta


def cluster_nll(
    eta: np.ndarray,
    ctx: QueryContext,
    *,
    occurrence_likelihood: str = "poisson",
) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    likelihood = _validate_occurrence_likelihood(occurrence_likelihood)
    grid_hazard = np.exp(eta[ctx.n_events :])
    grid_loss = ctx.aggregate_weighted_grid(grid_hazard)
    if likelihood == "poisson":
        event_values = -eta[: ctx.n_events]
    else:
        event_grid_rows = np.asarray(ctx.event_grid_rows, dtype=np.int64)
        grid_loss = grid_loss - np.bincount(
            ctx.event_sequence_local,
            weights=grid_hazard[event_grid_rows],
            minlength=ctx.n_sequences,
        )
        event_values = cloglog_event_nll(eta[: ctx.n_events])
    event_loss = np.bincount(
        ctx.event_sequence_local,
        weights=event_values,
        minlength=ctx.n_sequences,
    )
    return event_loss + grid_loss


def cluster_exposure(ctx: QueryContext) -> np.ndarray:
    return ctx.sequence_exposures().copy()
