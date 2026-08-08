from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .evidence import (
    evidence_diagnostics,
    frequency_channel_evidence,
)
from .likelihood import loss_rows, loss_value_rows
from .objective import SupportRecord
from .report import Certificate
from .reliability import (
    density_ratio_robust_test,
    DensityRatioRobustTest,
    EnvironmentSpec,
    multinomial_l1_radius,
    worst_case_total_variation_mean,
)
from .response import Context, ModelMatrix, ResponseEngine
from .rules import (
    ClosureTerm,
    EMPTY_SUPPORT,
    RuleIdentity,
    Support,
    hierarchy_branch_drop,
    hierarchy_branch_null_closure,
    hierarchy_closure,
)
from .search import SupportOptimizer, support_key
from .solver import FitResult

# Backward-compatible private test hooks; implementation is shared with search.
_multinomial_l1_radius = multinomial_l1_radius
_worst_case_total_variation_mean = worst_case_total_variation_mean


@dataclass(frozen=True)
class EffectTest:
    mean: float
    standard_error: float
    statistic: float
    pvalue: float
    testable: bool


@dataclass(frozen=True)
class _BootstrapComponent:
    statistic: float
    standardized_influence: np.ndarray


@dataclass(frozen=True)
class CertifiedModel:
    record: SupportRecord
    certificate: Certificate
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class CertificationResult:
    models: tuple[CertifiedModel, ...]
    certified: tuple[CertifiedModel, ...]
    selected: tuple[CertifiedModel, ...]
    family_size: int


def one_sided_mean_test(values: np.ndarray, threshold: float = 0.0) -> EffectTest:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)] - float(threshold)
    if len(values) < 2:
        return EffectTest(math.nan, math.inf, -math.inf, 1.0, False)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    if (
        not math.isfinite(standard_deviation)
        or standard_deviation <= np.finfo(float).eps
    ):
        return EffectTest(mean, math.inf, -math.inf, 1.0, False)
    standard_error = standard_deviation / math.sqrt(len(values))
    statistic = mean / standard_error
    pvalue = 0.5 * math.erfc(statistic / math.sqrt(2.0))
    return EffectTest(mean, standard_error, statistic, min(1.0, max(0.0, pvalue)), True)


def _mean_test_component(
    values: np.ndarray, threshold: float = 0.0
) -> tuple[EffectTest, _BootstrapComponent | None]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)] - float(threshold)
    test = one_sided_mean_test(finite)
    if not test.testable or len(finite) != len(values):
        return test, None
    centered = finite - float(np.mean(finite))
    deviation = float(np.std(finite, ddof=1))
    influence = np.ascontiguousarray(centered / deviation, dtype=np.float32)
    return test, _BootstrapComponent(test.statistic, influence)


def _scalar_direction_score(
    engine: ResponseEngine,
    context: Context,
    null_matrix: ModelMatrix,
    null_fit: FitResult,
    full_matrix: ModelMatrix,
    full_fit: FitResult,
    rows: np.ndarray | None = None,
    *,
    dependence_horizon_ticks: int | None = None,
    frequency_channel: str | None = None,
) -> tuple[EffectTest, _BootstrapComponent | None, dict[str, object]]:
    r"""Test one frozen D_fit predictor direction on independent D_cert.

    Discovery freezes either a scalar normalized kernel or a full ``M``-knot
    kernel for every rule by MDL.  On
    certification data, re-estimating those ``M`` coefficients and evaluating
    the same observations would invalidate the held-out test and needlessly
    disadvantage sparse higher-order rules.  We test the scalar path

    ``eta(theta) = eta_null + theta * (eta_full - eta_null), theta >= 0``.

    Both endpoint predictors were fitted using D_fit only.  The
    entity-clustered score at ``theta=0`` is therefore a valid
    one-degree-of-freedom D_cert test without fitting to D_cert.  The one-step
    amplitude below is diagnostic only; certified supports receive their full
    exact M-knot refit later on D_fit+D_cert.
    """
    if null_matrix.dimension != len(null_fit.coefficients):
        raise ValueError("scalar-direction null dimension mismatch")
    if full_matrix.dimension != len(full_fit.coefficients):
        raise ValueError("scalar-direction full dimension mismatch")
    restricted = rows is not None
    if rows is None:
        null_rows, _ = engine.frozen_active_predictor(
            context, null_matrix, null_fit.coefficients
        )
        full_rows, _ = engine.frozen_active_predictor(
            context, full_matrix, full_fit.coefficients
        )
        rows = np.union1d(null_rows, full_rows).astype(np.int64, copy=False)
    else:
        rows = np.unique(np.asarray(rows, dtype=np.int64))
    if dependence_horizon_ticks is not None:
        evidence = frequency_channel_evidence(
            engine,
            context,
            null_matrix,
            null_fit,
            full_matrix,
            full_fit,
            rows=rows,
            dependence_horizon_ticks=dependence_horizon_ticks,
            selected_channel=frequency_channel,
        )
        # Romano--Wolf continues to use the natural independent entity
        # clusters for the original frozen-direction score.  The extra
        # calendar-dependence channel has a different sampling unit and is
        # combined through the support max-p/Holm gate below, never by
        # pretending that calendar windows are independent entities.
        test, component = _mean_test_component(evidence.raw_entity_score)
        total_score = float(np.sum(evidence.raw_entity_score))
        total_information = float(evidence.raw_information)
        one_step_amplitude = (
            max(0.0, total_score / total_information)
            if math.isfinite(total_information)
            and total_information > np.finfo(float).eps
            else math.nan
        )
        diagnostics = {
            "method": "entity_clustered_frozen_shape_scalar_score",
            "degrees_of_freedom": 1,
            "shape_source": "D_fit_full_M_knot_exact_fit",
            "amplitude_fitted_on_D_cert": False,
            "one_step_amplitude": one_step_amplitude,
            "total_score": total_score,
            "conditional_information": total_information,
            "active_rows": int(len(rows)),
            "restricted_to_preregistered_impact_footprint": bool(restricted),
            "frequency_effect_separation": evidence_diagnostics(evidence),
        }
        return test, component, diagnostics
    null_eta = engine.frozen_linear_predictor_at_rows(
        context, null_matrix, null_fit.coefficients, rows
    )
    full_eta = engine.frozen_linear_predictor_at_rows(
        context, full_matrix, full_fit.coefficients, rows
    )
    direction = full_eta - null_eta
    free_dimension = engine.free_baseline_dimension
    intercept_direction = (
        full_fit.coefficients[:free_dimension] - null_fit.coefficients[:free_dimension]
    )
    exposure = (
        1.0 / context.dataset.ticks_per_unit
        if context.dataset.likelihood == "poisson"
        else 1.0
    )
    _, baseline_first, baseline_second = loss_rows(
        np.asarray(null_fit.coefficients[:free_dimension], dtype=np.float64),
        likelihood=context.dataset.likelihood,
        exposure_weight=np.full(free_dimension, exposure, dtype=np.float64),
        noevent_weight=np.full(free_dimension, exposure, dtype=np.float64),
        event_weight=np.zeros(free_dimension, dtype=np.float64),
    )
    if restricted:
        entity_gradient = np.zeros(len(context.entity_codes), dtype=np.float64)
        entity_information = np.zeros(len(context.entity_codes), dtype=np.float64)
    else:
        baseline_counts = engine.entity_age_counts(context)
        entity_gradient = context.entity_weights * (baseline_counts @ (
            baseline_first * intercept_direction
        ))
        entity_information = context.entity_weights * (baseline_counts @ (
            baseline_second * intercept_direction * intercept_direction
        ))
    if len(rows):
        event = _events_at_rows(context, rows)
        row_weight = context.weights_at_rows(rows)
        exposure_weight = exposure * row_weight
        noevent_weight = (
            exposure_weight - event
            if context.dataset.likelihood == "first_event_cloglog"
            else exposure_weight
        )
        _, first, second = loss_rows(
            null_eta,
            likelihood=context.dataset.likelihood,
            exposure_weight=exposure_weight,
            noevent_weight=noevent_weight,
            event_weight=event,
        )
        if restricted:
            gradient_correction = first * direction
            information_correction = second * direction * direction
        else:
            row_strata = context.temporal_baseline_groups_at_rows(
                rows, time_bins=engine.baseline_time_bins
            )
            gradient_correction = (
                first * direction
                - row_weight
                * baseline_first[row_strata]
                * intercept_direction[row_strata]
            )
            information_correction = (
                second * direction * direction
                - row_weight
                * baseline_second[row_strata]
                * intercept_direction[row_strata]
                * intercept_direction[row_strata]
            )
        local, _ = context.rows_to_entity_time(rows)
        boundaries = np.flatnonzero(np.r_[True, local[1:] != local[:-1]])
        entity_gradient[local[boundaries]] += np.add.reduceat(
            gradient_correction, boundaries
        )
        entity_information[local[boundaries]] += np.add.reduceat(
            information_correction, boundaries
        )
    # Positive score points from the frozen null toward the frozen full
    # predictor and hence represents an improving log-likelihood direction.
    entity_score = np.ascontiguousarray(-entity_gradient, dtype=np.float64)
    test, component = _mean_test_component(entity_score)
    total_score = float(np.sum(entity_score))
    total_information = float(np.sum(entity_information))
    one_step_amplitude = (
        max(0.0, total_score / total_information)
        if math.isfinite(total_information) and total_information > np.finfo(float).eps
        else math.nan
    )
    diagnostics: dict[str, object] = {
        "method": "entity_clustered_frozen_shape_scalar_score",
        "degrees_of_freedom": 1,
        "shape_source": "D_fit_full_M_knot_exact_fit",
        "amplitude_fitted_on_D_cert": False,
        "one_step_amplitude": one_step_amplitude,
        "total_score": total_score,
        "conditional_information": total_information,
        "active_rows": int(len(rows)),
        "restricted_to_preregistered_impact_footprint": bool(restricted),
    }
    return test, component, diagnostics


def _density_ratio_component(
    values: np.ndarray, *, alpha: float, threshold: float = 0.0
) -> tuple[DensityRatioRobustTest, _BootstrapComponent | None]:
    test = density_ratio_robust_test(
        values,
        alpha=alpha,
        threshold=threshold,
    )
    sample = np.asarray(values, dtype=np.float64)
    if not test.testable or not np.all(np.isfinite(sample)):
        return test, None
    mean = float(np.mean(sample))
    centered = sample - mean
    variance = float(np.mean(centered * centered))
    deviation = math.sqrt(max(0.0, variance))
    radius_scale = math.sqrt(float(test.radius))
    influence = centered - (
        radius_scale * ((centered * centered) - variance) / (2.0 * deviation)
    )
    influence_deviation = float(np.std(influence, ddof=1))
    if (
        not math.isfinite(influence_deviation)
        or influence_deviation <= np.finfo(float).eps
    ):
        return test, None
    standardized = np.ascontiguousarray(
        influence / influence_deviation, dtype=np.float32
    )
    return test, _BootstrapComponent(float(test.statistic), standardized)


def _events_at_rows(context: Context, rows: np.ndarray) -> np.ndarray:
    return context.target_counts_at_sorted_rows(rows)


def _loss_at_rows(context: Context, eta: np.ndarray, rows: np.ndarray) -> np.ndarray:
    event = _events_at_rows(context, rows)
    exposure = context.weights_at_rows(rows) * (
        1.0 / context.dataset.ticks_per_unit
        if context.dataset.likelihood == "poisson"
        else 1.0
    )
    noevent = (
        exposure - event
        if context.dataset.likelihood == "first_event_cloglog"
        else exposure
    )
    values = loss_value_rows(
        eta,
        likelihood=context.dataset.likelihood,
        exposure_weight=exposure,
        noevent_weight=noevent,
        event_weight=event,
    )
    return values


def _entity_losses_sparse(
    engine: ResponseEngine,
    context: Context,
    matrix: ModelMatrix,
    fit: FitResult,
) -> np.ndarray:
    """Exact entity losses without materializing the complete time grid."""
    baseline_dimension = matrix.baseline_dimension
    if baseline_dimension != engine.baseline_dimension:
        raise ValueError("matrix baseline-control dimension mismatch")
    baseline_loss = engine.tick_exposure * np.exp(
        np.clip(
            np.asarray(fit.coefficients[: engine.free_baseline_dimension]),
            -745.0,
            700.0,
        )
    )
    entity = context.entity_weights * (engine.entity_age_counts(context) @ baseline_loss)
    rows = matrix.active_rows
    if not len(rows):
        return entity
    group_eta = matrix.x @ np.asarray(fit.coefficients, dtype=np.float64)
    eta = group_eta[matrix.active_design_groups]
    full_loss = _loss_at_rows(context, eta, rows)
    default_loss = (
        baseline_loss[matrix.active_baseline_groups]
        * context.weights_at_rows(rows)
    )
    local, _ = context.rows_to_entity_time(rows)
    delta = full_loss - default_loss
    boundaries = np.flatnonzero(np.r_[True, local[1:] != local[:-1]])
    entity[local[boundaries]] += np.add.reduceat(delta, boundaries)
    return entity


def _entity_losses_frozen(
    engine: ResponseEngine,
    context: Context,
    matrix: ModelMatrix,
    fit: FitResult,
) -> np.ndarray:
    """Exact entity losses for fixed coefficients without a dense design."""
    baseline_loss = engine.tick_exposure * np.exp(
        np.clip(
            np.asarray(fit.coefficients[: engine.free_baseline_dimension]),
            -745.0,
            700.0,
        )
    )
    entity = context.entity_weights * (engine.entity_age_counts(context) @ baseline_loss)
    rows, eta = engine.frozen_active_predictor(context, matrix, fit.coefficients)
    if not len(rows):
        return entity
    full_loss = _loss_at_rows(context, eta, rows)
    default_loss = (
        baseline_loss[
            context.temporal_baseline_groups_at_rows(
                rows, time_bins=engine.baseline_time_bins
            )
        ]
        * context.weights_at_rows(rows)
    )
    local, _ = context.rows_to_entity_time(rows)
    delta = full_loss - default_loss
    boundaries = np.flatnonzero(np.r_[True, local[1:] != local[:-1]])
    entity[local[boundaries]] += np.add.reduceat(delta, boundaries)
    return entity


def _calendar_block_losses_sparse(
    engine: ResponseEngine,
    context: Context,
    matrix: ModelMatrix,
    fit: FitResult,
    environments: EnvironmentSpec,
) -> np.ndarray:
    """Return full-horizon-equivalent entity losses in fixed calendar blocks.

    Calendar F3 environments are used only for recurrent fixed-population
    panels.  Each cell contains one entity's loss inside one target-time block,
    rescaled by total horizon / block width.  Consequently every block effect
    is on the same scale as F1 and the support MDL threshold.  Histories before
    a boundary remain available through the already constructed model matrix;
    only the target-time loss is assigned to a block.
    """
    edges = environments.calendar_edges
    if edges is None:
        raise ValueError("calendar block loss requires calendar edges")
    edges = np.asarray(edges, dtype=np.int64)
    if (
        context.dataset.likelihood != "poisson"
        or edges.ndim != 1
        or len(edges) < 3
        or np.any(edges[1:] <= edges[:-1])
    ):
        raise ValueError("invalid recurrent calendar-block environment")
    block_count = len(edges) - 1
    baseline_rate = (1.0 / context.dataset.ticks_per_unit) * np.exp(
        np.clip(
            np.asarray(fit.coefficients[: engine.free_baseline_dimension]),
            -745.0,
            700.0,
        )
    )
    losses = np.zeros((len(context.entity_codes), block_count), dtype=np.float64)
    segment_entity, segment_left, segment_right, segment_group = (
        context.temporal_baseline_segments(time_bins=engine.baseline_time_bins)
    )
    for block in range(block_count):
        overlap = np.maximum(
            0,
            np.minimum(segment_right, edges[block + 1])
            - np.maximum(segment_left, edges[block]),
        ).astype(np.float64)
        np.add.at(
            losses[:, block],
            segment_entity,
            overlap
            * baseline_rate[segment_group]
            * context.entity_weights[segment_entity],
        )
    rows = matrix.active_rows
    if len(rows):
        group_eta = matrix.x @ np.asarray(fit.coefficients, dtype=np.float64)
        eta = group_eta[matrix.active_design_groups]
        full_loss = _loss_at_rows(context, eta, rows)
        groups = context.temporal_baseline_groups_at_rows(
            rows, time_bins=engine.baseline_time_bins
        )
        default_loss = baseline_rate[groups] * context.weights_at_rows(rows)
        local, times = context.rows_to_entity_time(rows)
        blocks = np.searchsorted(edges, times, side="right") - 1
        valid = (blocks >= 0) & (blocks < block_count)
        if not np.all(valid):
            raise ValueError("active model row lies outside calendar F3 blocks")
        np.add.at(losses, (local, blocks), full_loss - default_loss)
    widths = np.diff(edges).astype(np.float64)
    total_width = float(edges[-1] - edges[0])
    losses *= total_width / widths[None, :]
    flattened = np.ascontiguousarray(losses.reshape(-1), dtype=np.float64)
    if flattened.shape != environments.inverse.shape:
        raise ValueError("calendar losses do not align with F3 environments")
    return flattened


def _environment_losses_sparse(
    engine: ResponseEngine,
    context: Context,
    matrix: ModelMatrix,
    fit: FitResult,
    environments: EnvironmentSpec,
    *,
    entity_losses: np.ndarray | None = None,
) -> np.ndarray:
    if environments.calendar_edges is not None:
        return _calendar_block_losses_sparse(
            engine, context, matrix, fit, environments
        )
    if entity_losses is not None:
        return entity_losses
    return _entity_losses_sparse(engine, context, matrix, fit)


def _branch_drop(support: Support, root: RuleIdentity) -> Support:
    return hierarchy_branch_drop(support, root)


def _branch_null_closure(
    full_closure: tuple[ClosureTerm, ...], drop_support: Support, root: RuleIdentity
) -> tuple[ClosureTerm, ...]:
    return hierarchy_branch_null_closure(full_closure, drop_support, root)


def _evaluate_frozen(
    engine: ResponseEngine,
    context: Context,
    fit_matrix: ModelMatrix,
    fit: FitResult,
) -> tuple[ModelMatrix, np.ndarray]:
    for term, sign in zip(fit_matrix.closure, fit_matrix.closure_signs, strict=True):
        engine.set_closure_sign(term, sign)
    matrix = engine.model_metadata(
        fit_matrix.support, forced_closure=fit_matrix.closure
    )
    if matrix.dimension != len(fit.coefficients):
        raise ValueError("frozen model dimension changed across split")
    entity_loss = _entity_losses_frozen(engine, context, matrix, fit)
    return matrix, entity_loss


def _fit_on_discovery(
    optimizer: SupportOptimizer,
    support: Support,
    *,
    closure: tuple[ClosureTerm, ...] | None = None,
    source: SupportRecord | None = None,
) -> tuple[ModelMatrix, FitResult]:
    resolved = hierarchy_closure(support) if closure is None else closure
    # Certification consumes only the frozen D_fit coefficients/layout before
    # evaluating them on D_cert.  Asking the matrix-returning API to serve an
    # already cached optimum projected and aggregated a multi-GiB D_fit matrix
    # that `_evaluate_frozen` immediately discarded.  The result-only cache
    # preserves the identical exact fit and materializes a matrix only when the
    # optimum is genuinely missing.
    fit = optimizer.fit_fixed_results_many(
        [(support, resolved)],
        sources=[source],
    )[0]
    matrix = optimizer.engine.model_metadata(
        support,
        forced_closure=resolved,
    )
    return matrix, fit


def _holm_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = sorted(range(count), key=lambda index: (pvalues[index], index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _romano_wolf_adjust(
    components: list[tuple[_BootstrapComponent, ...] | None],
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    """Gaussian entity-cluster multiplier Romano--Wolf step-down max-T.

    Each support is one max-p IUT and therefore has the scalar statistic
    ``min(component T)``.  Bootstrap component errors share the same entity
    multipliers across the complete frozen family.  The observed gaps from the
    limiting component are retained when recomputing the bootstrap minimum;
    centering every component at the boundary would incorrectly treat strong
    F1/F2/F3 alternatives as additional null components.
    """
    count = len(components)
    if not count:
        return []
    if any(item is None or not item for item in components):
        valid = [
            index for index, item in enumerate(components) if item is not None and item
        ]
    else:
        valid = list(range(count))
    adjusted = np.ones(count, dtype=np.float64)
    if not valid:
        return adjusted.tolist()
    entity_counts = {
        len(component.standardized_influence)
        for index in valid
        for component in (components[index] or ())
    }
    if len(entity_counts) != 1:
        raise ValueError("Romano-Wolf components must share entity clusters")
    entity_count = entity_counts.pop()
    flat: list[_BootstrapComponent] = []
    support_indices: list[np.ndarray] = []
    observed = np.full(count, -math.inf, dtype=np.float64)
    gaps: list[np.ndarray] = []
    for item in components:
        if item is None or not item:
            support_indices.append(np.zeros(0, dtype=np.int64))
            gaps.append(np.zeros(0, dtype=np.float64))
            continue
        left = len(flat)
        flat.extend(item)
        indices = np.arange(left, len(flat), dtype=np.int64)
        statistics = np.asarray(
            [component.statistic for component in item], dtype=np.float64
        )
        minimum = float(np.min(statistics))
        support_indices.append(indices)
        observed[len(support_indices) - 1] = minimum
        gaps.append(statistics - minimum)
    influence = np.asarray(
        [component.standardized_influence for component in flat],
        dtype=np.float32,
    )
    component_count = len(flat)
    covariance = np.zeros((component_count, component_count), dtype=np.float64)
    # Float32 storage bounds family memory; tiled float64 accumulation retains
    # deterministic covariance accuracy without a second family-sized array.
    tile = 16_384
    for left in range(0, entity_count, tile):
        right = min(entity_count, left + tile)
        values = influence[:, left:right].astype(np.float64)
        covariance += values @ values.T
    covariance /= float(entity_count)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    root = eigenvectors * np.sqrt(eigenvalues)[None, :]

    order = np.asarray(
        sorted(valid, key=lambda index: (-observed[index], index)),
        dtype=np.int64,
    )
    exceedances = np.zeros(len(order), dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    batch_size = min(10_000, max(1_000, int(resamples)))
    for left in range(0, int(resamples), batch_size):
        batch = min(batch_size, int(resamples) - left)
        normal = generator.standard_normal((batch, component_count)) @ root.T
        support_bootstrap = np.empty((batch, len(order)), dtype=np.float64)
        for position, support_index in enumerate(order):
            indices = support_indices[int(support_index)]
            support_bootstrap[:, position] = np.min(
                normal[:, indices] + gaps[int(support_index)][None, :],
                axis=1,
            )
        # suffix maxima implement every step-down remaining family in one pass.
        suffix = np.maximum.accumulate(support_bootstrap[:, ::-1], axis=1)[:, ::-1]
        for rank, support_index in enumerate(order):
            exceedances[rank] += int(
                np.count_nonzero(suffix[:, rank] >= observed[int(support_index)])
            )
    running = 0.0
    for rank, support_index in enumerate(order):
        pvalue = (1.0 + float(exceedances[rank])) / (float(resamples) + 1.0)
        running = max(running, pvalue)
        adjusted[int(support_index)] = min(1.0, running)
    return adjusted.tolist()


def compact_certified_models(
    models: tuple[CertifiedModel, ...],
) -> tuple[CertifiedModel, ...]:
    """Retain one D_fit-MDL representative along each certified nested chain.

    Search performs the same deterministic compaction before certification.
    Repeating it here makes resumed older checkpoints safe without using
    D_cert performance to choose among already-tested alternatives.
    """
    reliable = tuple(model for model in models if model.certificate.certified)
    ordered = sorted(
        reliable,
        key=lambda model: (
            len(model.record.support.rules),
            model.record.support.rules,
        ),
    )
    selected: list[CertifiedModel] = []
    for model in ordered:
        rules = set(model.record.support.rules)
        dominated = any(
            (
                set(other.record.support.rules) < rules
                or rules < set(other.record.support.rules)
            )
            and (
                other.record.score > model.record.score + 1.0e-12
                or (
                    abs(other.record.score - model.record.score) <= 1.0e-12
                    and (
                        len(other.record.support.rules),
                        other.record.support.rules,
                    )
                    < (
                        len(model.record.support.rules),
                        model.record.support.rules,
                    )
                )
            )
            for other in ordered
        )
        if not dominated:
            selected.append(model)
    selected.sort(key=lambda model: model.record.support.rules)
    return tuple(selected)


def _fixed_contextual_rule_probability_contribution(
    engine: ResponseEngine,
    context: Context,
    matrix: ModelMatrix,
    fit: FitResult,
    *,
    rule_index: int,
    rows: np.ndarray,
    full_eta: np.ndarray | None = None,
    hierarchical_total: bool = False,
) -> np.ndarray:
    """Return a direct modifier or its complete hierarchical context effect."""
    output = np.zeros(len(context.entity_codes), dtype=np.float64)
    if not len(rows):
        return output
    if not 0 <= rule_index < len(matrix.support.rules):
        raise IndexError("reported-rule index is outside the frozen support")
    contextual = (
        engine.frozen_hierarchical_total_contribution_at_rows(
            context,
            matrix,
            fit.coefficients,
            rule_index=rule_index,
            rows=rows,
        )
        if hierarchical_total
        else engine.frozen_contextual_rule_contribution_at_rows(
            context,
            matrix,
            fit.coefficients,
            rule_index=rule_index,
            rows=rows,
        )
    )
    if full_eta is None:
        full_eta = engine.frozen_linear_predictor_at_rows(
            context,
            matrix,
            fit.coefficients,
            rows,
        )
    else:
        full_eta = np.asarray(full_eta, dtype=np.float64)
        if full_eta.shape != rows.shape:
            raise ValueError("frozen full predictor does not align with footprint")
    without_eta = full_eta - contextual
    local, _ = context.rows_to_entity_time(rows)
    observed = (
        np.ones(len(rows), dtype=np.float64)
        if context.baseline_row_exposure is None
        else context.baseline_row_exposure[rows]
    )
    scale = float(context.dataset.ticks_per_unit)
    full_hazard = (
        np.bincount(
            local,
            weights=observed * np.exp(np.clip(full_eta, -745.0, 700.0)),
            minlength=len(output),
        )
        / scale
    )
    without_hazard = (
        np.bincount(
            local,
            weights=observed * np.exp(np.clip(without_eta, -745.0, 700.0)),
            minlength=len(output),
        )
        / scale
    )
    full_probability = -np.expm1(-full_hazard)
    without_probability = -np.expm1(-without_hazard)
    output[:] = full_probability - without_probability
    return output


def _frozen_reported_direction(values: np.ndarray) -> int:
    """Choose one external total direction from D_fit without using D_cert."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0
    mean = float(np.mean(finite))
    scale = max(1.0, float(np.max(np.abs(finite), initial=0.0)))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if mean > tolerance:
        return 1
    if mean < -tolerance:
        return -1
    return 0


def certify_family(
    optimizer: SupportOptimizer,
    certification_context: Context,
    family: tuple[SupportRecord, ...],
    config: RunConfig,
) -> CertificationResult:
    cert_engine = ResponseEngine(
        certification_context.dataset,
        lag=config.impact_lag,
        knot_count=config.knot_count,
        baseline_time_bins=config.baseline_time_bins,
        effect_model=config.effect_model,
        # Certification is streamed support by support.  A smaller response
        # LRU is sufficient and prevents D_fit, D_cert and the current dense
        # matrices from simultaneously approaching the process memory limit.
        cache_bytes=max(64 * 1024**2, config.cache_bytes // 16),
    )
    for term, sign in optimizer._closure_signs.items():
        cert_engine.set_closure_sign(term, sign)
    # F1 records ordinary held-out generalization.  F3 is a stronger
    # distributionally robust support-level gate on the same entity gains.
    # F1 remains an explicit fail-closed contract check, while only the
    # nonredundant F2 and F3 influence components enter the support IUT and
    # Romano--Wolf family calibration.
    test_alpha = config.alpha

    # The empty F1/F3 null is common to the complete frozen family.  Fit and
    # evaluate it once; unlike the former global job table, no family-sized
    # collection of comparison matrices is retained.
    _, null_fit = _fit_on_discovery(optimizer, EMPTY_SUPPORT, closure=())
    null_cert_matrix = cert_engine.model_metadata(EMPTY_SUPPORT, forced_closure=())
    if null_cert_matrix.dimension != len(null_fit.coefficients):
        raise ValueError("frozen baseline dimension changed across split")
    null_entity = _entity_losses_frozen(
        cert_engine, certification_context, null_cert_matrix, null_fit
    )
    # Repeated branch identities cache only an entity-loss vector.  Dense
    # matrices are always local to one comparison and are released immediately
    # after its horizon diagnostic.
    cert_evaluation_cache: OrderedDict[
        tuple[Support, tuple[ClosureTerm, ...]], np.ndarray
    ] = OrderedDict()
    cert_evaluation_cache_bytes = 0
    cert_evaluation_cache_limit = max(1, min(512 * 1024**2, config.cache_bytes // 32))

    def evaluate_comparison(
        key: tuple[Support, tuple[ClosureTerm, ...]],
        fit: FitResult,
    ) -> tuple[ModelMatrix, FitResult, np.ndarray]:
        nonlocal cert_evaluation_cache_bytes
        support, closure = key
        cert_matrix = cert_engine.model_metadata(support, forced_closure=closure)
        if cert_matrix.dimension != len(fit.coefficients):
            raise ValueError("frozen comparison dimension changed across split")
        entity_loss = cert_evaluation_cache.get(key)
        if entity_loss is None:
            entity_loss = _entity_losses_frozen(
                cert_engine, certification_context, cert_matrix, fit
            )
            if entity_loss.nbytes <= cert_evaluation_cache_limit:
                cert_evaluation_cache[key] = entity_loss
                cert_evaluation_cache_bytes += entity_loss.nbytes
                while (
                    cert_evaluation_cache_bytes > cert_evaluation_cache_limit
                    and len(cert_evaluation_cache) > 1
                ):
                    _, removed = cert_evaluation_cache.popitem(last=False)
                    cert_evaluation_cache_bytes -= removed.nbytes
        else:
            cert_evaluation_cache.move_to_end(key)
        return cert_matrix, fit, entity_loss

    interim: list[
        tuple[
            SupportRecord,
            float,
            dict[str, object],
            tuple[str, ...],
            tuple[_BootstrapComponent, ...] | None,
        ]
    ] = []
    for record in family:
        reasons: list[str] = []
        diagnostics: dict[str, object] = {}
        full_matrix, full_entity = _evaluate_frozen(
            cert_engine, certification_context, record.matrix, record.fit
        )
        diagnostics["cert_mean_nll"] = float(np.mean(full_entity))
        branch_keys: list[tuple[Support, tuple[ClosureTerm, ...]]] = []
        for root in record.support.rules:
            drop_support = _branch_drop(record.support, root)
            drop_closure = optimizer.branch_challenger_closure(record.support, root)
            branch_keys.append((drop_support, drop_closure))

        # Rebuild exactly one D_fit full design only when a branch optimum is
        # genuinely missing.  Common singleton nulls and repeated branch keys
        # return their cached coefficients without reconstructing the source.
        discovery_source: SupportRecord | None = None
        discovery_full_matrix: ModelMatrix | None = None
        if not optimizer.fixed_fit_results_cached(branch_keys):
            for term, sign in zip(
                record.matrix.closure, record.matrix.closure_signs, strict=True
            ):
                optimizer.engine.set_closure_sign(term, sign)
            discovery_full_matrix = optimizer.engine.model_matrix(
                optimizer.context,
                record.support,
                forced_closure=record.matrix.closure,
            )
            if discovery_full_matrix.dimension != len(record.fit.coefficients):
                raise ValueError(
                    "frozen discovery dimension changed during certification"
                )
            discovery_source = SupportRecord(
                record.support,
                discovery_full_matrix,
                record.fit,
                record.penalty,
                record.score,
                record.rule_score,
                record.closure_null_nll,
                record.rule_score_upper,
            )
        branch_models: list[FitResult] = []
        for left in range(0, len(branch_keys), config.exact_workers):
            wave = branch_keys[left : left + config.exact_workers]
            fitted_wave = optimizer.fit_fixed_results_many(
                wave,
                sources=[discovery_source] * len(wave),
            )
            branch_models.extend(fitted_wave)
            del fitted_wave
        del discovery_source, discovery_full_matrix
        if not null_fit.converged:
            reasons.append("baseline_null_nonconvergence")
            interim.append((record, 1.0, diagnostics, tuple(reasons), None))
            del full_matrix, full_entity
            continue
        support_gain = null_entity - full_entity
        f1, f1_component = _mean_test_component(support_gain)
        diagnostics["f1"] = {
            **f1.__dict__,
            "method": "entity_clustered_frozen_predictor_nll",
        }
        # Retain the old diagnostic key for result-schema readers.  It now
        # exactly equals the F1 gate instead of being a nonbinding diagnostic.
        diagnostics["f1_fixed_predictor_nll"] = f1.__dict__
        if not f1.testable:
            reasons.append("f1_not_testable")
        elif f1.pvalue > test_alpha:
            reasons.append("f1_heldout_nll_gain_not_positive")
        certification_entities = len(certification_context.entity_codes)
        certification_penalty = optimizer.objective.penalty_for_dimension(
            record.support,
            record.matrix.dimension - record.matrix.baseline_dimension,
            n_entities=certification_entities,
        )
        discovery_mdl_materiality = certification_penalty / (
            2.0 * max(1, certification_entities)
        )
        # Complexity and model selection were already paid for on independent
        # D_fit.  The predictor is frozen on D_cert, so charging its parameter
        # code again would be a second, order-dependent penalty that
        # systematically disadvantages pair/triplet and multi-rule supports.
        # F3 tests distributional robustness of the held-out gain itself.
        f3_materiality = 0.0
        f3_test, f3_component = _density_ratio_component(
            support_gain,
            alpha=test_alpha,
            threshold=f3_materiality,
        )
        support_f3_pvalue = f3_test.pvalue
        distribution_shift_testable = f3_test.testable
        support_f3 = bool(
            distribution_shift_testable and support_f3_pvalue <= test_alpha
        )
        if not distribution_shift_testable:
            reasons.append("f3_distribution_shift_not_testable")
        if not support_f3:
            if distribution_shift_testable:
                reasons.append("f3_density_ratio_robust_gain_not_positive")
        rule_pvalues: list[float] = []
        rule_components: list[tuple[_BootstrapComponent | None, ...]] = []
        rule_diagnostics: list[dict[str, object]] = []
        for rule_index, (root, branch_key, drop_fit) in enumerate(
            zip(
                record.support.rules,
                branch_keys,
                branch_models,
                strict=True,
            )
        ):
            discovery_footprint = optimizer.engine.footprint_rows(
                optimizer.context,
                root,
                config.early_warning_horizon,
            )
            discovery_full_eta = optimizer.engine.frozen_linear_predictor_at_rows(
                optimizer.context,
                record.matrix,
                record.fit.coefficients,
                discovery_footprint,
            )
            discovery_total_probability = (
                _fixed_contextual_rule_probability_contribution(
                    optimizer.engine,
                    optimizer.context,
                    record.matrix,
                    record.fit,
                    rule_index=rule_index,
                    rows=discovery_footprint,
                    full_eta=discovery_full_eta,
                    hierarchical_total=True,
                )
            )
            reported_total_sign = _frozen_reported_direction(
                discovery_total_probability
            )
            reported_total_direction = {
                -1: "inhibition",
                0: "unidentified",
                1: "excitation",
            }[reported_total_sign]
            discovery_total_mean = (
                float(np.mean(discovery_total_probability))
                if len(discovery_total_probability)
                else 0.0
            )
            del discovery_full_eta, discovery_total_probability
            if reported_total_sign == 0 and not root.hierarchical:
                reasons.append(f"f2_total_direction_unidentified_on_fit:{root}")
            semantic_type = (
                "additive_hierarchical_modifier"
                if root.hierarchical
                else "support_conditional_additive_rule"
                if root.support_additive
                else "main_or_total_state_rule"
            )
            if (
                not root.hierarchical
                and reported_total_sign != int(root.sign)
            ):
                reasons.append(f"f2_total_state_direction_conflict:{root}")
            if not drop_fit.converged:
                rule_pvalues.append(1.0)
                rule_components.append((None,))
                reasons.append(f"branch_nonconvergence:{root}")
                rule_diagnostics.append(
                    {
                        "rule": repr(root),
                        "scalar_direction": None,
                        "global": None,
                        "horizon": None,
                        "probability": None,
                        "footprint_rows": 0,
                        "pvalue": 1.0,
                        "reason": drop_fit.message,
                        "rule_sign": int(root.sign),
                        "rule_direction": (
                            "excitation" if root.sign > 0 else "inhibition"
                        ),
                        "reported_total_sign": int(reported_total_sign),
                        "reported_total_direction": reported_total_direction,
                        "semantic_type": semantic_type,
                        "fit_total_probability_mean": discovery_total_mean,
                    }
                )
                continue
            drop_cert_matrix = cert_engine.model_metadata(
                branch_key[0], forced_closure=branch_key[1]
            )
            if drop_cert_matrix.dimension != len(drop_fit.coefficients):
                raise ValueError("frozen branch-null dimension changed across split")
            dependence_horizon_ticks = (
                int(root.window) + int(config.impact_lag)
            ) * certification_context.dataset.ticks_per_unit
            frequency_channel: str | None = None
            discovery_frequency_diagnostics: dict[str, object] | None = None
            if config.frequency_effect_separation:
                discovery_drop_matrix = optimizer.engine.model_metadata(
                    branch_key[0], forced_closure=branch_key[1]
                )
                discovery_frequency = frequency_channel_evidence(
                    optimizer.engine,
                    optimizer.context,
                    discovery_drop_matrix,
                    drop_fit,
                    record.matrix,
                    record.fit,
                    rows=discovery_footprint,
                    dependence_horizon_ticks=(
                        int(root.window) + int(config.impact_lag)
                    )
                    * optimizer.context.dataset.ticks_per_unit,
                )
                frequency_channel = discovery_frequency.selected_channel
                discovery_frequency_diagnostics = evidence_diagnostics(
                    discovery_frequency
                )
                del discovery_drop_matrix, discovery_frequency
            footprint = cert_engine.footprint_rows(
                certification_context, root, config.early_warning_horizon
            )
            scalar_test, scalar_component, scalar_diagnostics = _scalar_direction_score(
                cert_engine,
                certification_context,
                drop_cert_matrix,
                drop_fit,
                full_matrix,
                record.fit,
                rows=footprint,
                dependence_horizon_ticks=(
                    dependence_horizon_ticks
                    if config.frequency_effect_separation
                    else None
                ),
                frequency_channel=frequency_channel,
            )
            full_eta = cert_engine.frozen_linear_predictor_at_rows(
                certification_context,
                full_matrix,
                record.fit.coefficients,
                footprint,
            )
            raw_total_probability = _fixed_contextual_rule_probability_contribution(
                cert_engine,
                certification_context,
                full_matrix,
                record.fit,
                rule_index=rule_index,
                rows=footprint,
                full_eta=full_eta,
                hierarchical_total=True,
            )
            raw_direct_probability = _fixed_contextual_rule_probability_contribution(
                cert_engine,
                certification_context,
                full_matrix,
                record.fit,
                rule_index=rule_index,
                rows=footprint,
                full_eta=full_eta,
                hierarchical_total=False,
            )
            probability_difference = float(root.sign) * raw_direct_probability
            probability_test, probability_component = _mean_test_component(
                probability_difference
            )
            # D_fit estimates the complete M-knot state shape.  Independent
            # D_cert requires both (i) a positive score from the one-rule Drop
            # null toward that frozen shape and (ii) a direct fixed-predictor
            # probability contribution with the preregistered sign.  Their
            # max-p is the rule-level IUT: neither a compensating refit nor an
            # opposite held-out direction can certify the rule.
            frequency_diagnostics = scalar_diagnostics.get(
                "frequency_effect_separation"
            )
            frequency_test = (
                frequency_diagnostics.get("selected_test", {})
                if isinstance(frequency_diagnostics, dict)
                else {}
            )
            frequency_pvalue = float(frequency_test.get("pvalue", 0.0))
            frequency_testable = bool(frequency_test.get("testable", True))
            pvalue = max(
                scalar_test.pvalue,
                probability_test.pvalue,
                frequency_pvalue,
            )
            rule_pvalues.append(pvalue)
            rule_components.append((scalar_component, probability_component))
            rule_diagnostics.append(
                {
                    "rule": repr(root),
                    "scalar_direction": {
                        **scalar_test.__dict__,
                        **scalar_diagnostics,
                    },
                    "fit_frequency_effect_separation": (
                        discovery_frequency_diagnostics
                    ),
                    "global": None,
                    "horizon": scalar_test.__dict__,
                    "probability": probability_test.__dict__,
                    "total_contextual_probability": {
                        **probability_test.__dict__,
                        "raw_cert_mean": (
                            float(np.mean(raw_total_probability))
                            if len(raw_total_probability)
                            else 0.0
                        ),
                        "fit_mean": discovery_total_mean,
                        "reported_sign": int(reported_total_sign),
                        "reported_direction": reported_total_direction,
                        "method": (
                            "fixed_predictor_additive_hierarchical_total_"
                            "probability_contrast"
                        ),
                    },
                    "direct_modifier_probability": {
                        **probability_test.__dict__,
                        "raw_cert_mean": (
                            float(np.mean(raw_direct_probability))
                            if len(raw_direct_probability)
                            else 0.0
                        ),
                        "reported_sign": int(root.sign),
                        "reported_direction": (
                            "excitation" if root.sign > 0 else "inhibition"
                        ),
                    },
                    "rule_sign": int(root.sign),
                    "rule_direction": ("excitation" if root.sign > 0 else "inhibition"),
                    "reported_total_sign": int(reported_total_sign),
                    "reported_total_direction": reported_total_direction,
                    "semantic_type": semantic_type,
                    "fit_total_probability_mean": discovery_total_mean,
                    "footprint_rows": int(len(footprint)),
                    "pvalue": pvalue,
                }
            )
            if not scalar_test.testable:
                reasons.append(f"f2_scalar_direction_not_testable:{root}")
            if not probability_test.testable:
                reasons.append(f"f2_probability_direction_not_testable:{root}")
            if config.frequency_effect_separation and not frequency_testable:
                reasons.append(f"f2_frequency_channel_not_testable:{root}")
            elif config.frequency_effect_separation and frequency_pvalue > test_alpha:
                reasons.append(f"f2_frequency_channel_not_positive:{root}")
            del raw_total_probability, raw_direct_probability, probability_difference
            del drop_cert_matrix
        diagnostics["rules"] = rule_diagnostics
        f3 = support_f3
        diagnostics["f3"] = {
            "passed": f3,
            "distribution_shift_testable": distribution_shift_testable,
            "method": "entity_cluster_pearson_density_ratio_dro",
            "ambiguity": "mean density ratio 1; mean squared deviation at most rho",
            "entity_count": f3_test.entity_count,
            "ambiguity_chi_square_radius": f3_test.radius,
            "support_mean_gain": f3_test.mean_gain,
            "support_gain_standard_deviation": f3_test.standard_deviation,
            "support_robust_gain": f3_test.robust_gain,
            "support_robust_standard_error": f3_test.standard_error,
            "support_robust_lcb": f3_test.lower_confidence_bound,
            "certification_materiality_threshold": f3_materiality,
            "discovery_mdl_penalty_reference": certification_penalty,
            "discovery_mdl_materiality_reference_per_entity": (
                discovery_mdl_materiality
            ),
            "pvalue": support_f3_pvalue,
            "alpha": test_alpha,
        }
        support_pvalue = max(
            [*rule_pvalues, support_f3_pvalue],
            default=1.0,
        )
        bootstrap_components = (
            None
            if f3_component is None
            or any(
                component is None
                for components in rule_components
                for component in components
            )
            else (
                *(
                    component
                    for components in rule_components
                    for component in components
                    if component is not None
                ),
                f3_component,
            )
        )
        interim.append(
            (
                record,
                support_pvalue,
                diagnostics,
                tuple(reasons),
                bootstrap_components,
            )
        )
        del full_matrix, full_entity
    holm_adjusted = _holm_adjust([item[1] for item in interim])
    adjusted = _romano_wolf_adjust(
        [item[4] for item in interim],
        resamples=config.romano_wolf_resamples,
        seed=config.romano_wolf_seed,
    )
    models: list[CertifiedModel] = []
    f0 = (
        all(
            optimizer.context.dataset.f0_contract.get(name) is True
            for name in (
                "dynamic_predicates",
                "outcome_blind_predicate_construction",
                "direct_target_proxy_excluded_from_reported_dictionary",
                "strict_future_effect_required",
                "atomic_predicates",
                "primitive_event_provenance",
            )
        )
        and optimizer.context.dataset.f0_contract.get(
            # Independence is a certification prerequisite, not a dataset-schema
            # prerequisite (IBM is deliberately loadable but uncertifiable).  A
            # missing declaration must therefore fail closed, never silently pass.
            "independent_certification_units",
            False,
        )
        is True
    )
    for (
        (record, pvalue, diagnostics, reasons, _),
        adjusted_pvalue,
        holm_adjusted_pvalue,
    ) in zip(interim, adjusted, holm_adjusted, strict=True):
        # Monte-Carlo max-T calibration and analytic marginal p-values use
        # different finite-sample approximations.  An adjusted family p-value
        # must never be smaller than its unadjusted support IUT p-value.
        adjusted_pvalue = max(float(pvalue), float(adjusted_pvalue))
        if config.frequency_effect_separation:
            # Calendar-window HAC and entity-cluster components deliberately
            # use different sampling units.  Romano--Wolf retains dependence
            # gains among the original entity components; Holm on the complete
            # support max-p family safely covers the added calendar component.
            # Taking the larger value cannot weaken the existing correction.
            adjusted_pvalue = max(adjusted_pvalue, float(holm_adjusted_pvalue))
        final_reasons = reasons if f0 else (*reasons, "f0_contract_failed")
        if adjusted_pvalue > test_alpha:
            final_reasons = (
                *final_reasons,
                "romano_wolf_family_not_significant",
            )
        f1_pvalue = (
            float(diagnostics.get("f1", {}).get("pvalue", 1.0))
            if isinstance(diagnostics.get("f1"), dict)
            else 1.0
        )
        rule_pvalues = tuple(
            float(item["pvalue"]) for item in diagnostics.get("rules", [])
        )
        f3 = bool(diagnostics.get("f3", {}).get("passed", False))
        certified = bool(
            f0 and f3 and not final_reasons and adjusted_pvalue <= test_alpha
        )
        certificate = Certificate(
            support_key=support_key(record.support),
            f0=f0,
            f1_pvalue=f1_pvalue,
            f2_pvalues=rule_pvalues,
            f3=f3,
            family_pvalue=pvalue,
            holm_adjusted_pvalue=holm_adjusted_pvalue,
            certified=certified,
            reasons=final_reasons,
            family_adjusted_pvalue=adjusted_pvalue,
            multiplicity_method=(
                "romano_wolf_stepdown_max_t_plus_holm_frequency_channel"
                if config.frequency_effect_separation
                else "romano_wolf_stepdown_max_t"
            ),
            romano_wolf_resamples=config.romano_wolf_resamples,
        )
        models.append(CertifiedModel(record, certificate, diagnostics))
    certified_models = tuple(model for model in models if model.certificate.certified)
    selected_models = compact_certified_models(tuple(models))
    return CertificationResult(
        tuple(models), certified_models, selected_models, len(models)
    )
