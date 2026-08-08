from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .likelihood import loss_rows
from .response import Context, ModelMatrix, ResponseEngine
from .solver import FitResult


@dataclass(frozen=True)
class ChannelTest:
    """One-sided score test for one preselected evidence channel."""

    mean: float
    standard_error: float
    statistic: float
    pvalue: float
    testable: bool
    units: int


@dataclass(frozen=True)
class RiskSetDerivatives:
    """Exact null derivatives aggregated by calendar tick.

    The arrays are obtained from the sparse fitted model without constructing
    an entity-by-time matrix.  ``first`` and ``second`` already include entity
    weights and every event/control/rule contribution of the frozen null.
    The remaining fields retain the sparse pieces needed to allocate a common
    calendar-time direction back to entities exactly.
    """

    origin: int
    first: np.ndarray
    second: np.ndarray
    segment_entities: np.ndarray
    segment_left: np.ndarray
    segment_right: np.ndarray
    segment_groups: np.ndarray
    baseline_first: np.ndarray
    active_entities: np.ndarray
    active_times: np.ndarray
    active_first_correction: np.ndarray


@dataclass(frozen=True)
class FrequencyEvidence:
    """Curvature-orthogonal systemic/relative decomposition of a direction."""

    raw_test: ChannelTest
    systemic_test: ChannelTest
    relative_test: ChannelTest
    selected_channel: str
    selected_test: ChannelTest
    raw_entity_score: np.ndarray
    systemic_entity_score: np.ndarray
    relative_entity_score: np.ndarray
    systemic_tick_score: np.ndarray
    raw_information: float
    systemic_information: float
    relative_information: float
    active_rows: int
    risk_ticks: int
    dependence_horizon_ticks: int


def _one_sided_mean_test(values: np.ndarray) -> ChannelTest:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return ChannelTest(math.nan, math.inf, -math.inf, 1.0, False, len(values))
    mean = float(np.mean(values))
    deviation = float(np.std(values, ddof=1))
    if not math.isfinite(deviation) or deviation <= np.finfo(float).eps:
        return ChannelTest(mean, math.inf, -math.inf, 1.0, False, len(values))
    standard_error = deviation / math.sqrt(len(values))
    statistic = mean / standard_error
    pvalue = 0.5 * math.erfc(statistic / math.sqrt(2.0))
    return ChannelTest(
        mean,
        standard_error,
        statistic,
        min(1.0, max(0.0, pvalue)),
        True,
        len(values),
    )


def overlapping_block_mean_test(values: np.ndarray, horizon: int) -> ChannelTest:
    r"""One-sided mean test with an overlapping-block long-run variance.

    For calendar scores ``u_t`` and block length ``H``, this uses

    ``Omega = sum_s (sum_{t=s}^{s+H-1}(u_t-u_bar))^2 / (H*(T-H+1))``.

    This is the scalar Bartlett/HAC estimator written as overlapping blocks.
    There is no arbitrary block origin: a direction occurring near a nominal
    boundary contributes to every dependency window containing that tick.
    """

    sample = np.asarray(values, dtype=np.float64)
    sample = np.where(np.isfinite(sample), sample, 0.0)
    count = len(sample)
    if count < 2:
        return ChannelTest(math.nan, math.inf, -math.inf, 1.0, False, count)
    width = min(max(1, int(horizon)), count)
    mean = float(np.mean(sample))
    centered = sample - mean
    prefix = np.empty(count + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(centered, out=prefix[1:])
    block_sums = prefix[width:] - prefix[:-width]
    denominator = float(width * len(block_sums))
    long_run_variance = float(np.dot(block_sums, block_sums) / denominator)
    if (
        not math.isfinite(long_run_variance)
        or long_run_variance <= np.finfo(float).eps
    ):
        return ChannelTest(mean, math.inf, -math.inf, 1.0, False, count)
    standard_error = math.sqrt(long_run_variance / count)
    statistic = mean / standard_error
    pvalue = 0.5 * math.erfc(statistic / math.sqrt(2.0))
    return ChannelTest(
        mean,
        standard_error,
        statistic,
        min(1.0, max(0.0, pvalue)),
        True,
        count,
    )


def prepare_risk_set_derivatives(
    engine: ResponseEngine,
    context: Context,
    null_matrix: ModelMatrix,
    null_fit: FitResult,
) -> RiskSetDerivatives:
    """Aggregate exact null score/curvature over the calendar risk set."""

    if null_matrix.dimension != len(null_fit.coefficients):
        raise ValueError("risk-set null dimension mismatch")
    origin = int(np.min(context.starts)) if len(context.starts) else 0
    stop = int(np.max(context.ends)) + 1 if len(context.ends) else origin
    tick_count = max(0, stop - origin)
    free = engine.free_baseline_dimension
    exposure = float(engine.tick_exposure)
    _, baseline_first, baseline_second = loss_rows(
        np.asarray(null_fit.coefficients[:free], dtype=np.float64),
        likelihood=context.dataset.likelihood,
        exposure_weight=np.full(free, exposure, dtype=np.float64),
        noevent_weight=np.full(free, exposure, dtype=np.float64),
        event_weight=np.zeros(free, dtype=np.float64),
    )
    segment_entity, segment_left, segment_right, segment_group = (
        context.temporal_baseline_segments(time_bins=engine.baseline_time_bins)
    )
    first_difference = np.zeros(tick_count + 1, dtype=np.float64)
    second_difference = np.zeros(tick_count + 1, dtype=np.float64)
    if len(segment_entity):
        segment_weight = context.entity_weights[segment_entity]
        segment_first = baseline_first[segment_group] * segment_weight
        segment_second = baseline_second[segment_group] * segment_weight
        left_index = segment_left - origin
        right_index = segment_right - origin
        np.add.at(first_difference, left_index, segment_first)
        np.add.at(first_difference, right_index, -segment_first)
        np.add.at(second_difference, left_index, segment_second)
        np.add.at(second_difference, right_index, -segment_second)

    active_rows, active_eta = engine.frozen_active_predictor(
        context, null_matrix, null_fit.coefficients
    )
    if len(active_rows):
        active_event = context.target_counts_at_sorted_rows(active_rows)
        active_weight = context.weights_at_rows(active_rows)
        exposure_weight = exposure * active_weight
        noevent_weight = (
            exposure_weight - active_event
            if context.dataset.likelihood == "first_event_cloglog"
            else exposure_weight
        )
        _, active_first, active_second = loss_rows(
            active_eta,
            likelihood=context.dataset.likelihood,
            exposure_weight=exposure_weight,
            noevent_weight=noevent_weight,
            event_weight=active_event,
        )
        active_entity, active_time = context.rows_to_entity_time(active_rows)
        active_group = context.temporal_baseline_groups_at_rows(
            active_rows, time_bins=engine.baseline_time_bins
        )
        default_first = baseline_first[active_group] * active_weight
        default_second = baseline_second[active_group] * active_weight
        first_correction = active_first - default_first
        second_correction = active_second - default_second
        tick_index = active_time - origin
        np.add.at(first_difference, tick_index, first_correction)
        np.add.at(first_difference, tick_index + 1, -first_correction)
        np.add.at(second_difference, tick_index, second_correction)
        np.add.at(second_difference, tick_index + 1, -second_correction)
    else:
        active_entity = np.zeros(0, dtype=np.int32)
        active_time = np.zeros(0, dtype=np.int64)
        first_correction = np.zeros(0, dtype=np.float64)

    total_first = np.cumsum(first_difference[:-1])
    total_second = np.cumsum(second_difference[:-1])
    scale = max(1.0, float(np.max(np.abs(total_second), initial=0.0)))
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    if np.any(total_second < -tolerance):
        raise AssertionError("risk-set conditional Fisher became negative")
    total_second = np.maximum(total_second, 0.0)
    return RiskSetDerivatives(
        origin=origin,
        first=np.ascontiguousarray(total_first),
        second=np.ascontiguousarray(total_second),
        segment_entities=segment_entity,
        segment_left=segment_left,
        segment_right=segment_right,
        segment_groups=segment_group,
        baseline_first=np.ascontiguousarray(baseline_first),
        active_entities=np.ascontiguousarray(active_entity, dtype=np.int32),
        active_times=np.ascontiguousarray(active_time, dtype=np.int64),
        active_first_correction=np.ascontiguousarray(first_correction),
    )


def frequency_channel_evidence(
    engine: ResponseEngine,
    context: Context,
    null_matrix: ModelMatrix,
    null_fit: FitResult,
    full_matrix: ModelMatrix,
    full_fit: FitResult,
    *,
    rows: np.ndarray,
    dependence_horizon_ticks: int,
    prepared: RiskSetDerivatives | None = None,
    selected_channel: str | None = None,
) -> FrequencyEvidence:
    r"""Separate a frozen rule direction into systemic and relative evidence.

    At each calendar tick, the fitted direction ``X_it`` is decomposed with
    the exact null curvature ``w_it``:

    ``mu_t = sum_i w_it X_it / sum_i w_it``;
    ``X_sys=mu_t`` and ``X_rel=X_it-mu_t``.

    Consequently the two directions are Fisher-orthogonal and their
    information adds exactly.  The systemic channel is tested over overlapping
    calendar windows; the relative channel is tested over entity histories.
    ``selected_channel=None`` deterministically freezes the larger D_fit
    statistic, with ``relative`` winning an exact tie.
    """

    rows = np.unique(np.asarray(rows, dtype=np.int64))
    risk = prepared or prepare_risk_set_derivatives(
        engine, context, null_matrix, null_fit
    )
    entity_count = len(context.entity_codes)
    empty_entities = np.zeros(entity_count, dtype=np.float64)
    if not len(rows):
        unavailable = ChannelTest(math.nan, math.inf, -math.inf, 1.0, False, 0)
        choice = selected_channel or "relative"
        return FrequencyEvidence(
            unavailable,
            unavailable,
            unavailable,
            choice,
            unavailable,
            empty_entities,
            empty_entities.copy(),
            empty_entities.copy(),
            np.zeros_like(risk.first),
            0.0,
            0.0,
            0.0,
            0,
            len(risk.first),
            int(dependence_horizon_ticks),
        )

    null_eta = engine.frozen_linear_predictor_at_rows(
        context, null_matrix, null_fit.coefficients, rows
    )
    full_eta = engine.frozen_linear_predictor_at_rows(
        context, full_matrix, full_fit.coefficients, rows
    )
    direction = full_eta - null_eta
    event = context.target_counts_at_sorted_rows(rows)
    row_weight = context.weights_at_rows(rows)
    exposure_weight = engine.tick_exposure * row_weight
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
    local, times = context.rows_to_entity_time(rows)
    time_index = times - risk.origin
    if np.any(time_index < 0) or np.any(time_index >= len(risk.first)):
        raise AssertionError("direction footprint lies outside the risk calendar")

    numerator = np.zeros_like(risk.second)
    np.add.at(numerator, time_index, second * direction)
    mu = np.zeros_like(risk.second)
    positive_risk = risk.second > np.finfo(np.float64).eps
    mu[positive_risk] = numerator[positive_risk] / risk.second[positive_risk]

    raw_row_score = -first * direction
    raw_entity = np.bincount(
        local,
        weights=raw_row_score,
        minlength=entity_count,
    ).astype(np.float64)
    systemic_tick = -mu * risk.first

    prefix = np.empty(len(mu) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(mu, out=prefix[1:])
    segment_mu = (
        prefix[risk.segment_right - risk.origin]
        - prefix[risk.segment_left - risk.origin]
    )
    segment_score = (
        -context.entity_weights[risk.segment_entities]
        * risk.baseline_first[risk.segment_groups]
        * segment_mu
    )
    systemic_entity = np.bincount(
        risk.segment_entities,
        weights=segment_score,
        minlength=entity_count,
    ).astype(np.float64)
    if len(risk.active_entities):
        active_score = (
            -risk.active_first_correction
            * mu[risk.active_times - risk.origin]
        )
        systemic_entity += np.bincount(
            risk.active_entities,
            weights=active_score,
            minlength=entity_count,
        )
    relative_entity = raw_entity - systemic_entity

    raw_information = float(np.dot(second, direction * direction))
    systemic_information = float(np.dot(risk.second, mu * mu))
    relative_information = raw_information - systemic_information
    info_scale = max(1.0, abs(raw_information), abs(systemic_information))
    if relative_information < -256.0 * np.finfo(np.float64).eps * info_scale:
        raise AssertionError("systemic/relative Fisher decomposition is not nested")
    relative_information = max(0.0, relative_information)

    raw_test = _one_sided_mean_test(raw_entity)
    systemic_test = overlapping_block_mean_test(
        systemic_tick, dependence_horizon_ticks
    )
    relative_test = _one_sided_mean_test(relative_entity)
    if selected_channel is None:
        systemic_stat = (
            systemic_test.statistic if systemic_test.testable else -math.inf
        )
        relative_stat = relative_test.statistic if relative_test.testable else -math.inf
        choice = "systemic" if systemic_stat > relative_stat else "relative"
    else:
        choice = str(selected_channel)
        if choice not in {"systemic", "relative"}:
            raise ValueError("selected frequency channel must be systemic or relative")
    selected = systemic_test if choice == "systemic" else relative_test
    return FrequencyEvidence(
        raw_test=raw_test,
        systemic_test=systemic_test,
        relative_test=relative_test,
        selected_channel=choice,
        selected_test=selected,
        raw_entity_score=np.ascontiguousarray(raw_entity),
        systemic_entity_score=np.ascontiguousarray(systemic_entity),
        relative_entity_score=np.ascontiguousarray(relative_entity),
        systemic_tick_score=np.ascontiguousarray(systemic_tick),
        raw_information=raw_information,
        systemic_information=systemic_information,
        relative_information=relative_information,
        active_rows=len(rows),
        risk_ticks=len(risk.first),
        dependence_horizon_ticks=min(
            max(1, int(dependence_horizon_ticks)), max(1, len(risk.first))
        ),
    )


def evidence_diagnostics(evidence: FrequencyEvidence) -> dict[str, object]:
    return {
        "method": "risk_set_curvature_orthogonal_systemic_relative_score",
        "selected_channel": evidence.selected_channel,
        "selected_test": evidence.selected_test.__dict__,
        "raw": evidence.raw_test.__dict__,
        "systemic": evidence.systemic_test.__dict__,
        "relative": evidence.relative_test.__dict__,
        "raw_information": evidence.raw_information,
        "systemic_information": evidence.systemic_information,
        "relative_information": evidence.relative_information,
        "active_rows": evidence.active_rows,
        "risk_ticks": evidence.risk_ticks,
        "dependence_horizon_ticks": evidence.dependence_horizon_ticks,
        "channel_selection_source": "D_fit_only",
    }
