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

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "two_way_entity_calendar_godambe_clbic",
            "effective_dimension": self.effective_dimension,
            "parameter_rank": self.parameter_rank,
            "entity_clusters": self.entity_clusters,
            "calendar_clusters": self.calendar_clusters,
            "hac_width": self.hac_width,
            "raw_minimum_eigenvalue": self.raw_minimum_eigenvalue,
            "score_residual": self.score_residual,
            "psd_correction": "negative_eigenvalues_clipped_to_zero",
        }


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
    """Exact two-way sandwich effective dimension at a fixed-support MLE.

    The model likelihood is not changed.  Scores are allocated to the
    pre-registered financial sampling unit and calendar tick, with a Bartlett
    window spanning the complete rule-formation plus impact horizon.  The
    entity-time cell meat is subtracted once, as required by two-way cluster
    inclusion--exclusion.
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
        return DependencyComplexity(0.0, 0, 0, 0, 1, 0.0, 0.0)
    indices = np.concatenate((nuisance_indices, interest_indices))

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
