from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import QueryContext


@dataclass(frozen=True)
class MarkHeadFit:
    """Conditional log-normal mark head with a fixed variance.

    The temporal activation shape is supplied by the occurrence head.  Only a
    separate scalar coefficient per discovered rule is estimated here.  The
    variance is estimated once from the D_fit baseline and then frozen, so
    support comparisons cannot improve merely by changing their noise scale.
    """

    intercept: float
    nuisance_beta: np.ndarray
    rule_beta: np.ndarray
    variance: float
    unit: float
    nll: float
    rank: int
    converged: bool


@dataclass(frozen=True)
class MarkBaseResidualizer:
    """Reusable weighted least-squares factorization of intercept+nuisance."""

    base_design: np.ndarray
    weighted_base: np.ndarray
    sqrt_weight: np.ndarray
    weighted_pinv: np.ndarray
    rank: int


def _cluster_weights(ctx: QueryContext, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones(ctx.n_sequences, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (ctx.n_sequences,):
        raise ValueError("mark cluster weights must match the query context")
    if np.any(~np.isfinite(values)) or np.any(values < 0) or not np.any(values > 0):
        raise ValueError("mark cluster weights must be finite, nonnegative, and nonzero")
    return values


def mark_design(
    nuisance_design: np.ndarray,
    rule_activations: Sequence[np.ndarray],
) -> tuple[np.ndarray, int]:
    nuisance = np.asarray(nuisance_design, dtype=np.float64)
    if nuisance.ndim != 2:
        raise ValueError("mark nuisance design must be a matrix")
    parts = [np.ones((len(nuisance), 1), dtype=np.float64), nuisance]
    for activation in rule_activations:
        values = np.asarray(activation, dtype=np.float64).reshape(-1)
        if len(values) != len(nuisance):
            raise ValueError("mark activation/query length mismatch")
        parts.append(values.reshape(-1, 1))
    return np.concatenate(parts, axis=1), 1 + nuisance.shape[1]


def make_mark_base_residualizer(
    ctx: QueryContext,
    nuisance_design: np.ndarray,
    *,
    cluster_weights: np.ndarray | None = None,
) -> MarkBaseResidualizer:
    """Factor the fixed mark nuisance design once for exact FWL reuse."""
    nuisance = np.asarray(nuisance_design, dtype=np.float64)
    if nuisance.ndim != 2 or nuisance.shape[0] not in {ctx.n_events, ctx.n_queries}:
        raise ValueError("mark nuisance design must contain event rows or all query rows")
    base_design, _rule_start = mark_design(nuisance[: ctx.n_events], [])
    sequence_weights = _cluster_weights(ctx, cluster_weights)
    event_weights = sequence_weights[ctx.event_sequence_local]
    sqrt_weight = np.sqrt(event_weights)
    weighted_base = base_design * sqrt_weight[:, None]
    u, singular, vt = np.linalg.svd(weighted_base, full_matrices=False)
    if singular.size:
        cutoff = (
            np.finfo(np.float64).eps
            * max(weighted_base.shape)
            * float(singular[0])
        )
        inverse = np.divide(
            1.0,
            singular,
            out=np.zeros_like(singular),
            where=singular > cutoff,
        )
        weighted_pinv = (vt.T * inverse) @ u.T
        rank = int(np.sum(singular > cutoff))
    else:
        weighted_pinv = np.zeros((weighted_base.shape[1], weighted_base.shape[0]))
        rank = 0
    for value in (base_design, weighted_base, sqrt_weight, weighted_pinv):
        value.setflags(write=False)
    return MarkBaseResidualizer(
        base_design=base_design,
        weighted_base=weighted_base,
        sqrt_weight=sqrt_weight,
        weighted_pinv=weighted_pinv,
        rank=rank,
    )


def fit_mark_head(
    ctx: QueryContext,
    nuisance_design: np.ndarray,
    rule_activations: Sequence[np.ndarray],
    *,
    unit: float,
    variance: float | None,
    cluster_weights: np.ndarray | None = None,
    base_residualizer: MarkBaseResidualizer | None = None,
) -> MarkHeadFit:
    if ctx.event_marks is None:
        raise ValueError("marked fitting requires event marks")
    if len(ctx.event_marks) != ctx.n_events or ctx.n_events == 0:
        raise ValueError("marked fitting requires one mark per target event")
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError("mark unit must be finite and positive")
    marks = np.asarray(ctx.event_marks, dtype=np.float64)
    if np.any(~np.isfinite(marks)) or np.any(marks <= 0):
        raise ValueError("financial event marks must be finite and strictly positive")

    nuisance = np.asarray(nuisance_design, dtype=np.float64)
    if nuisance.ndim != 2 or nuisance.shape[0] not in {ctx.n_events, ctx.n_queries}:
        raise ValueError("mark nuisance design must contain event rows or all query rows")
    event_activations: list[np.ndarray] = []
    for raw in rule_activations:
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if len(values) not in {ctx.n_events, ctx.n_queries}:
            raise ValueError("mark activation must contain event rows or all query rows")
        event_activations.append(values[: ctx.n_events])
    event_design, rule_start = mark_design(nuisance[: ctx.n_events], event_activations)
    y = np.log(marks / float(unit))
    sequence_weights = _cluster_weights(ctx, cluster_weights)
    event_weights = sequence_weights[ctx.event_sequence_local]
    weight_sum = float(np.sum(event_weights))
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("marked fitting requires positive target-event weight mass")
    residualizer = base_residualizer or make_mark_base_residualizer(
        ctx,
        nuisance,
        cluster_weights=cluster_weights,
    )
    if (
        residualizer.base_design.shape != (ctx.n_events, rule_start)
        or not np.array_equal(
            residualizer.base_design,
            event_design[:, :rule_start],
        )
    ):
        raise ValueError("mark base residualizer/design mismatch")
    sqrt_weight = residualizer.sqrt_weight
    weighted_y = y * sqrt_weight
    base_beta_y = residualizer.weighted_pinv @ weighted_y
    if event_activations:
        rules = event_design[:, rule_start:]
        weighted_rules = rules * sqrt_weight[:, None]
        base_beta_rules = residualizer.weighted_pinv @ weighted_rules
        residual_y = weighted_y - residualizer.weighted_base @ base_beta_y
        residual_rules = weighted_rules - residualizer.weighted_base @ base_beta_rules
        rule_beta, _residuals, rule_rank, _singular = np.linalg.lstsq(
            residual_rules,
            residual_y,
            rcond=None,
        )
        base_beta = residualizer.weighted_pinv @ (
            weighted_y - weighted_rules @ rule_beta
        )
        beta = np.concatenate((base_beta, rule_beta))
        rank = int(residualizer.rank + rule_rank)
    else:
        beta = base_beta_y
        rank = int(residualizer.rank)
    fitted = event_design @ beta
    residual = y - fitted
    if variance is None:
        estimate = float(np.dot(event_weights, residual * residual) / weight_sum)
        centered = y - float(np.dot(event_weights, y) / weight_sum)
        reference = float(np.dot(event_weights, centered * centered) / weight_sum)
        numerical_floor = np.finfo(np.float64).eps * max(reference, 1.0)
        variance_value = max(estimate, numerical_floor)
    else:
        variance_value = float(variance)
        if not math.isfinite(variance_value) or variance_value <= 0:
            raise ValueError("fixed mark variance must be finite and positive")
    normal_nll = 0.5 * (
        math.log(2.0 * math.pi * variance_value) + residual * residual / variance_value
    )
    # y=log(m/unit), hence |dy/dm|=1/m.  Retaining the Jacobian makes this the
    # exact conditional log-normal mark likelihood rather than a transformed
    # squared-error surrogate.
    event_nll = normal_nll + np.log(marks)
    nll = float(np.dot(event_weights, event_nll))
    finite = bool(np.all(np.isfinite(beta)) and math.isfinite(nll))
    return MarkHeadFit(
        intercept=float(beta[0]),
        nuisance_beta=beta[1:rule_start].astype(np.float64, copy=True),
        rule_beta=beta[rule_start:].astype(np.float64, copy=True),
        variance=variance_value,
        unit=float(unit),
        nll=nll,
        rank=int(rank),
        converged=finite,
    )


def predict_mark_log_mean(
    fit: MarkHeadFit,
    nuisance_design: np.ndarray,
    rule_activations: Sequence[np.ndarray],
) -> np.ndarray:
    nuisance = np.asarray(nuisance_design, dtype=np.float64)
    if nuisance.ndim != 2:
        raise ValueError("mark nuisance design must be a matrix")
    if len(fit.nuisance_beta) != nuisance.shape[1] or len(fit.rule_beta) != len(rule_activations):
        raise ValueError("mark fit/design mismatch")
    result = np.full(len(nuisance), fit.intercept, dtype=np.float64)
    if nuisance.shape[1]:
        result += nuisance @ fit.nuisance_beta
    for coefficient, activation in zip(fit.rule_beta, rule_activations, strict=True):
        values = np.asarray(activation, dtype=np.float64).reshape(-1)
        if len(values) != len(nuisance):
            raise ValueError("mark activation/query length mismatch")
        result += float(coefficient) * values
    return result


def event_mark_log_density(
    fit: MarkHeadFit,
    ctx: QueryContext,
    nuisance_design: np.ndarray,
    rule_activations: Sequence[np.ndarray],
) -> np.ndarray:
    if ctx.event_marks is None:
        raise ValueError("mark density requires event marks")
    marks = np.asarray(ctx.event_marks, dtype=np.float64)
    mu = predict_mark_log_mean(
        fit,
        np.asarray(nuisance_design)[: ctx.n_events],
        [np.asarray(values)[: ctx.n_events] for values in rule_activations],
    )
    y = np.log(marks / fit.unit)
    return (
        -0.5 * (math.log(2.0 * math.pi * fit.variance) + (y - mu) ** 2 / fit.variance)
        - np.log(marks)
    )


def expected_mark(
    fit: MarkHeadFit,
    nuisance_design: np.ndarray,
    rule_activations: Sequence[np.ndarray],
) -> np.ndarray:
    mu = predict_mark_log_mean(fit, nuisance_design, rule_activations)
    with np.errstate(over="ignore", invalid="ignore"):
        return fit.unit * np.exp(mu + 0.5 * fit.variance)


def mark_score_moments(
    fit: MarkHeadFit,
    ctx: QueryContext,
    nuisance_design: np.ndarray,
    rule_activations: Sequence[np.ndarray],
    candidate_event_feature: np.ndarray,
    *,
    cluster_weights: np.ndarray | None = None,
    base_residualizer: MarkBaseResidualizer | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient/Fisher moments for adding one unrestricted mark coefficient.

    ``candidate_event_feature`` contains the M candidate kernel columns on the
    event rows.  Its temporal shape is selected jointly with the occurrence
    head; only the scalar mark coefficient is unrestricted in sign.
    """
    feature = np.asarray(candidate_event_feature, dtype=np.float64)
    if feature.ndim != 2 or feature.shape[0] != ctx.n_events:
        raise ValueError("candidate mark feature must have one row per target event")
    marks = np.asarray(ctx.event_marks, dtype=np.float64) if ctx.event_marks is not None else None
    if marks is None or len(marks) != ctx.n_events or np.any(marks <= 0):
        raise ValueError("mark score requires positive event marks")
    event_nuisance = np.asarray(nuisance_design)[: ctx.n_events]
    event_activations = [np.asarray(values)[: ctx.n_events] for values in rule_activations]
    mu = predict_mark_log_mean(fit, event_nuisance, event_activations)
    y = np.log(marks / fit.unit)
    sequence_weights = _cluster_weights(ctx, cluster_weights)
    weights = sequence_weights[ctx.event_sequence_local] / fit.variance
    # The fitted null contains an intercept, hierarchy/control terms and any
    # existing rule activations.  Profile information for a new coefficient is
    # the Schur complement, not raw X'WX; failing to residualize systematically
    # understates the local mark gain for correlated financial patterns.
    base_design, _rule_start = mark_design(event_nuisance, event_activations)
    sqrt_weight = np.sqrt(weights)
    weighted_feature = feature * sqrt_weight[:, None]
    if base_residualizer is not None:
        # The residualizer uses cluster weights rather than weights/variance;
        # the common positive 1/variance scale leaves the projection unchanged.
        if (
            base_residualizer.base_design.shape != base_design.shape
            or not np.array_equal(base_residualizer.base_design, base_design)
        ):
            raise ValueError("mark score residualizer/design mismatch")
        coefficients = base_residualizer.weighted_pinv @ (
            feature * base_residualizer.sqrt_weight[:, None]
        )
    else:
        weighted_base = base_design * sqrt_weight[:, None]
        coefficients = np.linalg.lstsq(
            weighted_base,
            weighted_feature,
            rcond=None,
        )[0]
    residual_feature = feature - base_design @ coefficients
    gradient = residual_feature.T @ (weights * (mu - y))
    information = residual_feature.T @ (weights[:, None] * residual_feature)
    return gradient, 0.5 * (information + information.T)


def cluster_mark_nll(
    fit: MarkHeadFit,
    ctx: QueryContext,
    nuisance_design: np.ndarray,
    rule_activations: Sequence[np.ndarray],
) -> np.ndarray:
    log_density = event_mark_log_density(fit, ctx, nuisance_design, rule_activations)
    return np.bincount(
        ctx.event_sequence_local,
        weights=-log_density,
        minlength=ctx.n_sequences,
    ).astype(np.float64)


def cluster_financial_mean_loss(
    eta: np.ndarray,
    mark_mean: np.ndarray,
    ctx: QueryContext,
    *,
    unit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Squared loss for the conditional mean of cumulative marked exposure."""
    if ctx.event_marks is None:
        raise ValueError("financial exposure loss requires event marks")
    eta = np.asarray(eta, dtype=np.float64)
    mark_mean = np.asarray(mark_mean, dtype=np.float64)
    if len(eta) != ctx.n_queries or len(mark_mean) != ctx.n_queries:
        raise ValueError("financial exposure query length mismatch")
    observed = np.bincount(
        ctx.event_sequence_local,
        weights=ctx.event_marks,
        minlength=ctx.n_sequences,
    ).astype(np.float64)
    expected = ctx.aggregate_weighted_grid(
        np.exp(eta[ctx.n_events :]) * mark_mean[ctx.n_events :]
    ).astype(np.float64)
    loss = ((observed - expected) / float(unit)) ** 2
    return loss, observed, expected
