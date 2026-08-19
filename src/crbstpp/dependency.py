from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .likelihood import loss_rows
from .native import accumulate_cluster_scores, dependency_row_derivatives, moments
from .response import Context, ModelMatrix, ResponseEngine
from .solver import FitResult


@dataclass(frozen=True)
class DependencyComplexity:
    effective_dimension: float
    parameter_rank: int
    entity_clusters: int
    calendar_clusters: int
    hac_width: int
    raw_minimum_eigenvalue: float
    score_residual: float
    method: str = "two_way_entity_calendar_godambe_clbic"

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "effective_dimension": self.effective_dimension,
            "parameter_rank": self.parameter_rank,
            "entity_clusters": self.entity_clusters,
            "calendar_clusters": self.calendar_clusters,
            "hac_width": self.hac_width,
            "raw_minimum_eigenvalue": self.raw_minimum_eigenvalue,
            "score_residual": self.score_residual,
            "psd_correction": "negative_eigenvalues_clipped_to_zero",
        }


def _continuous_dependency_scores(
    engine: ResponseEngine,
    context: Context,
    matrix: ModelMatrix,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact full-grid scores grouped by continuous sampling unit.

    A raw continuous clock cannot be expanded into regular calendar ticks.
    Instead, every likelihood row is assigned to its pre-registered
    ``dependency_group`` (a trading day for WSELOB).  This changes only the
    covariance used by the MDL code.  The TPP likelihood and its MLE are
    untouched.
    """

    dimension = matrix.dimension
    coefficients = np.asarray(coefficients, dtype=np.float64)
    eta = engine.linear_predictor(context, matrix, coefficients)
    exposure = context.all_row_weights()
    event = np.zeros(context.n_grid, dtype=np.float64)
    if len(context.target_rows):
        event[context.target_rows] = context.target_counts
    _, first, _ = loss_rows(
        eta,
        likelihood=context.dataset.likelihood,
        exposure_weight=exposure,
        noevent_weight=exposure,
        event_weight=event,
    )

    entity_codes, entity_count = dependency_cluster_codes(context)
    scores = np.zeros((entity_count, dimension), dtype=np.float64)

    # Baseline columns are one-hot.  Accumulating one entity at a time avoids
    # allocating an observation-by-parameter dense design.
    for local in range(len(context.entity_codes)):
        left = int(context.offsets[local])
        right = int(context.offsets[local + 1])
        if right <= left:
            continue
        rows = np.arange(left, right, dtype=np.int64)
        groups = context.temporal_baseline_groups_at_rows(
            rows, time_bins=engine.baseline_time_bins
        )
        np.add.at(scores[int(entity_codes[local])], groups, first[left:right])

    # Controls, hierarchy nuisance and reportable rules use the same sparse
    # signed blocks as the exact likelihood evaluator, including total-state
    # masking.  Consequently this score cannot silently use a different rule
    # representation from Add/Drop fitting.
    for block, sign, coefficient_slice in engine._frozen_blocks(context, matrix):
        if not len(block.rows):
            continue
        local = np.searchsorted(context.offsets, block.rows, side="right") - 1
        clusters = entity_codes[local]
        contribution = (
            float(sign) * first[block.rows, None] * np.asarray(block.values)
        )
        for column, global_column in enumerate(
            range(coefficient_slice.start, coefficient_slice.stop)
        ):
            np.add.at(scores[:, global_column], clusters, contribution[:, column])

    return scores, first


def dependency_cluster_codes(context: Context) -> tuple[np.ndarray, int]:
    """Return contiguous sampling-unit codes for the current context."""

    groups = context.dataset.dependency_groups
    raw = (
        context.entity_codes.astype(np.int64, copy=False)
        if groups is None
        else groups[context.entity_codes].astype(np.int64, copy=False)
    )
    _, inverse = np.unique(raw, return_inverse=True)
    codes = np.ascontiguousarray(inverse, dtype=np.int32)
    return codes, int(np.max(codes, initial=-1)) + 1


def prediction_opportunity_count(context: Context, impact_lag: int) -> int:
    """Return the target-blind number of independent prediction horizons.

    A first-event likelihood has one prediction episode per population entity.
    A recurrent likelihood can issue another prediction after one impact
    horizon, so its opportunity count is total observed exposure divided by
    that horizon.  Raw timestamp units are converted with ``ticks_per_unit``.
    """

    horizon = int(impact_lag)
    if horizon < 1:
        raise ValueError("impact_lag must be positive")
    if context.dataset.likelihood == "first_event_cloglog":
        return max(2, int(context.population_entities))
    ticks = np.maximum(
        0,
        np.asarray(context.ends, dtype=np.int64)
        - np.asarray(context.starts, dtype=np.int64),
    )
    exposure = float(np.sum(ticks, dtype=np.float64)) / float(
        context.dataset.ticks_per_unit
    )
    return max(2, int(math.floor(exposure / float(horizon))))


def _baseline_derivatives(
    engine: ResponseEngine,
    context: Context,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    free = engine.free_baseline_dimension
    exposure = np.full(free, engine.tick_exposure, dtype=np.float64)
    noevent = np.full(free, engine.tick_exposure, dtype=np.float64)
    if context.dataset.likelihood == "first_event_cloglog":
        exposure.fill(1.0)
        noevent.fill(1.0)
    _, first, second = loss_rows(
        np.asarray(coefficients[:free], dtype=np.float64),
        likelihood=context.dataset.likelihood,
        exposure_weight=exposure,
        noevent_weight=noevent,
        event_weight=np.zeros(free, dtype=np.float64),
    )
    return first, second


def _bartlett_meat(scores: np.ndarray, width: int) -> np.ndarray:
    """PSD overlapping-block representation of a Bartlett HAC meat."""

    scores = np.asarray(scores, dtype=np.float64)
    count, dimension = scores.shape
    if not count:
        return np.zeros((dimension, dimension), dtype=np.float64)
    width = min(max(1, int(width)), count)
    if width == 1:
        return scores.T @ scores
    prefix = np.zeros((count + 1, dimension), dtype=np.float64)
    np.cumsum(scores, axis=0, out=prefix[1:])
    blocks = prefix[width:] - prefix[:-width]
    # Division by H makes the independent-score expectation match the cell
    # meat away from the two deterministic endpoints.
    return (blocks.T @ blocks) / float(width)


def model_dependency_complexity(
    engine: ResponseEngine,
    context: Context,
    matrix: ModelMatrix,
    fit: FitResult,
    *,
    dependence_horizon_ticks: int,
    require_converged: bool = True,
) -> DependencyComplexity:
    """Exact sandwich effective dimension at a fixed-support MLE.

    The model likelihood is not changed.  Regular-grid datasets use the
    pre-registered financial sampling unit and calendar tick.  Continuous
    datasets use their declared dependency group because raw timestamps do
    not define a dense calendar grid.
    """

    if (require_converged and not fit.converged) or matrix.dimension != len(
        fit.coefficients
    ):
        raise ValueError("dependency complexity requires a compatible fitted point")
    if matrix.x.shape[0] == 0:
        raise ValueError("dependency complexity requires a materialized design")
    dimension = matrix.dimension
    free = engine.free_baseline_dimension
    rule_start = matrix.baseline_dimension + matrix.closure_dimension
    if dimension <= rule_start:
        _, entity_count = dependency_cluster_codes(context)
        if context.dataset.likelihood == "continuous_poisson":
            return DependencyComplexity(
                0.0,
                0,
                entity_count,
                0,
                0,
                0.0,
                0.0,
                method="one_way_dependency_group_godambe_clbic",
            )
        origin = int(np.min(context.starts)) if len(context.starts) else 0
        stop = int(np.max(context.ends)) + 1 if len(context.ends) else origin
        return DependencyComplexity(
            0.0,
            0,
            entity_count,
            max(0, stop - origin),
            max(1, int(dependence_horizon_ticks)),
            0.0,
            0.0,
        )
    beta = np.asarray(fit.coefficients, dtype=np.float64)
    eta = matrix.x @ beta
    _, grouped_first, grouped_second = loss_rows(
        eta,
        likelihood=context.dataset.likelihood,
        exposure_weight=matrix.exposure_weight,
        noevent_weight=matrix.noevent_weight,
        event_weight=matrix.event_weight,
    )
    gradient, hessian = moments(
        matrix.x,
        grouped_first,
        grouped_second,
        device="cpu",
    )
    # Complexity is the conditional effective dimension of the reportable
    # rule block, not ``effective_df(full) - effective_df(baseline)``.  The
    # latter can be negative when a rule explains clustered residuals and
    # would fail to charge a common shock.  Efficiently residualizing rule
    # scores against the fitted baseline/control nuisance gives the exact
    # nonnegative composite-likelihood dimension of the added model block.
    active = beta > max(1.0e-10, 10.0 * np.finfo(np.float64).eps)
    nuisance_mask = np.arange(dimension) < rule_start
    nuisance_mask[:free] = True
    nuisance_indices = np.flatnonzero(nuisance_mask & (active | (np.arange(dimension) < free)))
    interest_indices = np.flatnonzero((~nuisance_mask) & active)
    if not len(interest_indices):
        # The rule can be inactive at its standalone optimum yet become
        # conditionally useful in a larger rule set.  Report the actual
        # registered sampling-unit count rather than the misleading value
        # zero.  The caller already applies the structural-dimension floor,
        # so this diagnostic correction cannot make such a rule free.
        _, entity_count = dependency_cluster_codes(context)
        return DependencyComplexity(
            0.0,
            0,
            entity_count,
            0,
            0 if context.dataset.likelihood == "continuous_poisson" else 1,
            0.0,
            0.0,
            method=(
                "one_way_dependency_group_godambe_clbic"
                if context.dataset.likelihood == "continuous_poisson"
                else "two_way_entity_calendar_godambe_clbic"
            ),
        )
    indices = np.concatenate((nuisance_indices, interest_indices))

    if context.dataset.likelihood == "continuous_poisson":
        entity_scores, _ = _continuous_dependency_scores(
            engine, context, matrix, beta
        )
        selected_entity = entity_scores[:, indices]
        full_meat = selected_entity.T @ selected_entity
        full_meat = 0.5 * (full_meat + full_meat.T)

        selected_hessian = hessian[np.ix_(indices, indices)]
        selected_hessian = 0.5 * (selected_hessian + selected_hessian.T)
        nuisance_count = len(nuisance_indices)
        interest_count = len(interest_indices)
        if nuisance_count:
            h_nn = selected_hessian[:nuisance_count, :nuisance_count]
            h_rn = selected_hessian[nuisance_count:, :nuisance_count]
            nuisance_inverse = np.linalg.pinv(
                h_nn,
                rcond=np.finfo(np.float64).eps * max(1, nuisance_count),
                hermitian=True,
            )
            projection = h_rn @ nuisance_inverse
            transform = np.concatenate(
                (-projection, np.eye(interest_count, dtype=np.float64)), axis=1
            )
            efficient_hessian = (
                selected_hessian[nuisance_count:, nuisance_count:]
                - projection
                @ selected_hessian[:nuisance_count, nuisance_count:]
            )
        else:
            transform = np.eye(interest_count, dtype=np.float64)
            efficient_hessian = selected_hessian
        efficient_hessian = 0.5 * (efficient_hessian + efficient_hessian.T)
        meat = transform @ full_meat @ transform.T
        meat = 0.5 * (meat + meat.T)
        eigenvalues, eigenvectors = np.linalg.eigh(meat)
        minimum = float(np.min(eigenvalues, initial=0.0))
        positive = np.maximum(eigenvalues, 0.0)
        meat = (eigenvectors * positive[None, :]) @ eigenvectors.T
        inverse = np.linalg.pinv(
            efficient_hessian,
            rcond=np.finfo(np.float64).eps * max(1, interest_count),
            hermitian=True,
        )
        effective = float(np.trace(inverse @ meat))
        if not math.isfinite(effective):
            raise FloatingPointError("nonfinite dependency effective dimension")
        effective = max(0.0, effective)
        efficient_gradient = transform @ gradient[indices]
        efficient_entity = selected_entity @ transform.T
        gradient_scale = max(
            1.0, float(np.linalg.norm(efficient_gradient, ord=np.inf))
        )
        score_residual = float(
            np.linalg.norm(
                np.sum(efficient_entity, axis=0) - efficient_gradient,
                ord=np.inf,
            )
            / gradient_scale
        )
        _, entity_count = dependency_cluster_codes(context)
        return DependencyComplexity(
            effective_dimension=effective,
            parameter_rank=int(np.linalg.matrix_rank(efficient_hessian)),
            entity_clusters=entity_count,
            calendar_clusters=0,
            hac_width=0,
            raw_minimum_eigenvalue=minimum,
            score_residual=score_residual,
            method="one_way_dependency_group_godambe_clbic",
        )

    entity_codes, entity_count = dependency_cluster_codes(context)
    origin = int(np.min(context.starts)) if len(context.starts) else 0
    stop = int(np.max(context.ends)) + 1 if len(context.ends) else origin
    tick_count = max(0, stop - origin)
    entity_scores = np.zeros((entity_count, dimension), dtype=np.float64)
    time_difference = np.zeros((tick_count + 1, free), dtype=np.float64)
    baseline_first, _ = _baseline_derivatives(engine, context, beta)
    segment_entity, segment_left, segment_right, segment_group = (
        context.temporal_baseline_segments(time_bins=engine.baseline_time_bins)
    )
    if len(segment_entity):
        segment_weight = context.entity_weights[segment_entity]
        segment_length = (segment_right - segment_left).astype(np.float64)
        segment_score = (
            segment_length * segment_weight * baseline_first[segment_group]
        )
        np.add.at(
            entity_scores,
            (entity_codes[segment_entity], segment_group),
            segment_score,
        )
        endpoints = segment_weight * baseline_first[segment_group]
        np.add.at(
            time_difference,
            (segment_left - origin, segment_group),
            endpoints,
        )
        np.add.at(
            time_difference,
            (segment_right - origin, segment_group),
            -endpoints,
        )
    time_scores = np.zeros((tick_count, dimension), dtype=np.float64)
    if tick_count:
        time_scores[:, :free] = np.cumsum(time_difference[:-1], axis=0)

    rows = np.asarray(matrix.active_rows, dtype=np.int64)
    cell_meat = np.zeros((dimension, dimension), dtype=np.float64)
    if len(segment_entity):
        baseline_cell_mass = np.bincount(
            segment_group,
            weights=(
                segment_length
                * context.entity_weights[segment_entity] ** 2
                * baseline_first[segment_group] ** 2
            ),
            minlength=free,
        )
        cell_meat[np.arange(free), np.arange(free)] = baseline_cell_mass
    if len(rows):
        groups = np.asarray(matrix.active_design_groups, dtype=np.int32)
        # The installed native operator assumes every row inside an entity's
        # observation bounds has unit opportunity.  Dynamic observation masks
        # must use the exact row-weighted reference calculation below.
        unit_row_exposure = context.baseline_row_exposure is None or np.all(
            context.baseline_row_exposure == 1.0
        )
        compiled_rows = (
            dependency_row_derivatives(
                eta,
                rows,
                groups,
                matrix.active_baseline_groups,
                context.offsets,
                context.starts,
                context.entity_weights,
                entity_codes,
                context.target_rows,
                context.target_counts,
                baseline_first,
                origin=origin,
                tick_exposure=engine.tick_exposure,
                likelihood=context.dataset.likelihood,
                workers=0,
            )
            if unit_row_exposure
            else None
        )
        if compiled_rows is None:
            local, times = context.rows_to_entity_time(rows)
            weights = context.weights_at_rows(rows)
            event = context.target_counts_at_sorted_rows(rows)
            exposure = engine.tick_exposure * weights
            noevent = (
                exposure - event
                if context.dataset.likelihood == "first_event_cloglog"
                else exposure
            )
            _, active_first, _ = loss_rows(
                eta[groups],
                likelihood=context.dataset.likelihood,
                exposure_weight=exposure,
                noevent_weight=noevent,
                event_weight=event,
            )
            active_entity_cluster = entity_codes[local]
            active_time_cluster = np.ascontiguousarray(times - origin, dtype=np.int32)
            baseline_group = context.temporal_baseline_groups_at_rows(
                rows, time_bins=engine.baseline_time_bins
            )
            default_first = baseline_first[baseline_group] * weights
        else:
            (
                active_first,
                active_entity_cluster,
                active_time_cluster,
                default_first,
            ) = compiled_rows
            baseline_group = np.asarray(
                matrix.active_baseline_groups, dtype=np.int32
            )
        accumulate_cluster_scores(
            matrix.x, groups, active_entity_cluster, active_first, entity_scores
        )
        accumulate_cluster_scores(
            matrix.x, groups, active_time_cluster, active_first, time_scores
        )
        np.add.at(
            entity_scores,
            (active_entity_cluster, baseline_group),
            -default_first,
        )
        np.add.at(
            time_scores,
            (active_time_cluster, baseline_group),
            -default_first,
        )
        active_square_mass = np.bincount(
            groups,
            weights=active_first * active_first,
            minlength=len(matrix.x),
        )
        _, active_outer = moments(
            matrix.x,
            np.zeros(len(matrix.x), dtype=np.float64),
            active_square_mass,
            device="cpu",
        )
        cell_meat += active_outer
        removed_default = np.bincount(
            baseline_group,
            weights=default_first * default_first,
            minlength=free,
        )
        cell_meat[np.arange(free), np.arange(free)] -= removed_default

    selected_entity = entity_scores[:, indices]
    selected_time = time_scores[:, indices]
    entity_meat = selected_entity.T @ selected_entity
    width = min(max(1, int(dependence_horizon_ticks)), max(1, tick_count))
    time_meat = _bartlett_meat(selected_time, width)
    selected_cell = cell_meat[np.ix_(indices, indices)]
    full_meat = entity_meat + time_meat - selected_cell
    full_meat = 0.5 * (full_meat + full_meat.T)

    selected_hessian = hessian[np.ix_(indices, indices)]
    selected_hessian = 0.5 * (selected_hessian + selected_hessian.T)
    nuisance_count = len(nuisance_indices)
    interest_count = len(interest_indices)
    if nuisance_count:
        h_nn = selected_hessian[:nuisance_count, :nuisance_count]
        h_rn = selected_hessian[nuisance_count:, :nuisance_count]
        nuisance_inverse = np.linalg.pinv(
            h_nn,
            rcond=np.finfo(np.float64).eps * max(1, nuisance_count),
            hermitian=True,
        )
        projection = h_rn @ nuisance_inverse
        transform = np.concatenate(
            (-projection, np.eye(interest_count, dtype=np.float64)), axis=1
        )
        efficient_hessian = (
            selected_hessian[nuisance_count:, nuisance_count:]
            - projection @ selected_hessian[:nuisance_count, nuisance_count:]
        )
    else:
        transform = np.eye(interest_count, dtype=np.float64)
        efficient_hessian = selected_hessian
    efficient_hessian = 0.5 * (efficient_hessian + efficient_hessian.T)
    meat = transform @ full_meat @ transform.T
    meat = 0.5 * (meat + meat.T)
    eigenvalues, eigenvectors = np.linalg.eigh(meat)
    minimum = float(np.min(eigenvalues, initial=0.0))
    positive = np.maximum(eigenvalues, 0.0)
    meat = (eigenvectors * positive[None, :]) @ eigenvectors.T

    inverse = np.linalg.pinv(
        efficient_hessian,
        rcond=np.finfo(np.float64).eps * max(1, interest_count),
        hermitian=True,
    )
    effective = float(np.trace(inverse @ meat))
    if not math.isfinite(effective):
        raise FloatingPointError("nonfinite dependency effective dimension")
    effective = max(0.0, effective)
    efficient_gradient = transform @ gradient[indices]
    efficient_entity = selected_entity @ transform.T
    gradient_scale = max(
        1.0, float(np.linalg.norm(efficient_gradient, ord=np.inf))
    )
    score_residual = float(
        np.linalg.norm(
            np.sum(efficient_entity, axis=0) - efficient_gradient, ord=np.inf
        )
        / gradient_scale
    )
    return DependencyComplexity(
        effective_dimension=effective,
        parameter_rank=int(np.linalg.matrix_rank(efficient_hessian)),
        entity_clusters=entity_count,
        calendar_clusters=tick_count,
        hac_width=width,
        raw_minimum_eigenvalue=minimum,
        score_residual=score_residual,
    )
