from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .data import QueryContext


@dataclass(frozen=True)
class MeanTest:
    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    lower_bound: float
    n_clusters: int


def _query_cluster_weights(ctx: QueryContext, cluster_weights: np.ndarray | None) -> np.ndarray:
    if cluster_weights is None:
        return np.ones(ctx.n_sequences, dtype=np.float64)
    weights = np.asarray(cluster_weights, dtype=np.float64)
    if weights.shape != (ctx.n_sequences,):
        raise ValueError("cluster weights must match the query context")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("cluster weights must be finite, nonnegative, and have positive mass")
    return weights


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 300
    eps = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            return h
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, df: int) -> float:
    if df <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if value == 0.0:
        return 0.5
    x = float(df) / (float(df) + float(value) ** 2)
    tail = 0.5 * _regularized_beta(x, 0.5 * float(df), 0.5)
    return 1.0 - tail if value > 0 else tail


@lru_cache(maxsize=256)
def student_t_ppf(probability: float, df: int) -> float:
    if not 0.0 < probability < 1.0:
        if probability == 0.0:
            return -math.inf
        if probability == 1.0:
            return math.inf
        raise ValueError("probability must lie in [0,1]")
    if probability == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while student_t_cdf(lo, df) > probability:
        lo *= 2.0
    while student_t_cdf(hi, df) < probability:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < probability:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def one_sided_mean_test(values: Sequence[float], *, null: float = 0.0, alpha: float = 0.05) -> MeanTest:
    arr = np.asarray(values, dtype=np.float64)
    # Silently deleting a nonfinite cluster changes the tested population in
    # an outcome-dependent way.  A failed numerical contribution therefore
    # invalidates the test instead of being complete-case filtered.
    if arr.ndim != 1 or np.any(~np.isfinite(arr)):
        return MeanTest(math.nan, math.inf, -math.inf, 1.0, -math.inf, int(arr.size))
    n = int(arr.size)
    if n < 2:
        return MeanTest(float(np.mean(arr)) if n else math.nan, math.inf, -math.inf, 1.0, -math.inf, n)
    estimate = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd <= np.finfo(np.float64).eps:
        if estimate > null:
            return MeanTest(estimate, 0.0, math.inf, 0.0, estimate, n)
        return MeanTest(estimate, 0.0, -math.inf, 1.0, estimate, n)
    se = sd / math.sqrt(n)
    statistic = (estimate - float(null)) / se
    p_value = float(1.0 - student_t_cdf(statistic, n - 1))
    lower = estimate - student_t_ppf(1.0 - alpha, n - 1) * se
    return MeanTest(estimate, se, statistic, p_value, lower, n)


def one_sided_mean_test_zero_padded(
    nonzero_values: Sequence[float],
    *,
    total_count: int,
    null: float = 0.0,
    alpha: float = 0.05,
) -> MeanTest:
    """One-sample t test after implicit zero padding.

    Early-warning contrasts are identically zero for entities outside a rule's
    future footprint.  Computing their mean and sample variance from the active
    entity values plus an exact zero count avoids allocating one full vector per
    rule/support comparison.  This is the same test on the same observations;
    only its sufficient statistics are evaluated directly.
    """
    values = np.asarray(nonzero_values, dtype=np.float64)
    n = int(total_count)
    if (
        values.ndim != 1
        or n < len(values)
        or np.any(~np.isfinite(values))
    ):
        return MeanTest(math.nan, math.inf, -math.inf, 1.0, -math.inf, max(0, n))
    if n < 2:
        estimate = float(values[0]) if n == 1 and len(values) else 0.0 if n == 1 else math.nan
        return MeanTest(estimate, math.inf, -math.inf, 1.0, -math.inf, n)
    estimate = float(np.sum(values, dtype=np.float64) / n)
    centered_sum_squares = float(
        np.sum((values - estimate) ** 2, dtype=np.float64)
        + (n - len(values)) * estimate * estimate
    )
    sd = math.sqrt(max(0.0, centered_sum_squares / (n - 1)))
    if sd <= np.finfo(np.float64).eps:
        if estimate > null:
            return MeanTest(estimate, 0.0, math.inf, 0.0, estimate, n)
        return MeanTest(estimate, 0.0, -math.inf, 1.0, estimate, n)
    se = sd / math.sqrt(n)
    statistic = (estimate - float(null)) / se
    p_value = float(1.0 - student_t_cdf(statistic, n - 1))
    lower = estimate - student_t_ppf(1.0 - alpha, n - 1) * se
    return MeanTest(estimate, se, statistic, p_value, lower, n)


def equivalence_mean_test(values: Sequence[float], *, tolerance: float, alpha: float = 0.05) -> dict:
    if tolerance <= 0:
        raise ValueError("equivalence tolerance must be positive")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or np.any(~np.isfinite(arr)):
        return {
            "estimate": None,
            "p_value": 1.0,
            "passed": False,
            "n_clusters": int(arr.size),
            "invalid_reason": "nonfinite cluster value",
        }
    n = int(arr.size)
    if n < 2:
        return {"estimate": float(np.mean(arr)) if n else None, "p_value": 1.0, "passed": False, "n_clusters": n}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd <= np.finfo(np.float64).eps:
        p = 0.0 if abs(mean) < tolerance else 1.0
        return {"estimate": mean, "p_value": p, "passed": bool(p <= alpha), "n_clusters": n}
    se = sd / math.sqrt(n)
    p_lower = float(1.0 - student_t_cdf((mean + tolerance) / se, n - 1))
    p_upper = float(student_t_cdf((mean - tolerance) / se, n - 1))
    p_value = max(p_lower, p_upper)
    return {
        "estimate": mean,
        "standard_error": se,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_value": p_value,
        "passed": bool(p_value <= alpha),
        "n_clusters": n,
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    if p.size == 0:
        return p.copy()
    p = np.clip(np.where(np.isfinite(p), p, 1.0), 0.0, 1.0)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * sorted_p)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def efficient_residual(
    z: np.ndarray,
    nuisance: np.ndarray,
    eta_null: np.ndarray,
    ctx: QueryContext,
    cluster_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    z = np.asarray(z, dtype=np.float64)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    if z.shape != (ctx.n_queries,):
        raise ValueError("efficient-score activation has wrong shape")
    if nuisance.ndim != 2 or nuisance.shape[0] != ctx.n_queries:
        raise ValueError("nuisance design has wrong shape")
    grid = slice(ctx.n_events, None)
    sequence_weights = _query_cluster_weights(ctx, cluster_weights)
    weights = (
        ctx.expand_sequence_values(sequence_weights).astype(np.float64, copy=False)
        * np.exp(np.asarray(eta_null[grid], dtype=np.float64))
    )
    if int(getattr(ctx.grid_weights, "nbytes", 0)) > 0:
        weights *= np.asarray(ctx.grid_weights, dtype=np.float64)
    if nuisance.shape[1] == 0:
        residual = z.copy()
        information = float(np.dot(weights, residual[grid] ** 2))
        return residual, information, 0
    f_grid = nuisance[grid]
    gram = f_grid.T @ (weights.reshape(-1, 1) * f_grid)
    rhs = f_grid.T @ (weights * z[grid])
    left, singular, right = np.linalg.svd(gram, full_matrices=False)
    if singular.size:
        tolerance = np.finfo(np.float64).eps * max(gram.shape) * singular[0]
        rank = int(np.sum(singular > tolerance))
        inverse = np.divide(
            1.0,
            singular,
            out=np.zeros_like(singular),
            where=singular > tolerance,
        )
        coefficients = right.T @ (inverse * (left.T @ rhs))
    else:
        rank = 0
        coefficients = np.zeros(gram.shape[1], dtype=np.float64)
    residual = z - nuisance @ coefficients
    information = float(np.dot(weights, residual[grid] ** 2))
    return residual, information, rank


def efficient_information_matrix(
    x_rule: np.ndarray,
    nuisance: np.ndarray | Sequence[np.ndarray],
    eta_null: np.ndarray,
    ctx: QueryContext,
    cluster_weights: np.ndarray | None = None,
    *,
    projected_shape: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fisher-residualize a rule block without arbitrary rank cutoffs.

    When ``projected_shape`` is supplied, only the residual in that one kernel
    direction is returned.  The full M-by-M information is still exact, while
    avoiding an N-by-M float64 residual copy during certification.
    """
    x_rule = np.asarray(x_rule)
    if x_rule.ndim != 2 or x_rule.shape[0] != ctx.n_queries:
        raise ValueError("rule block has wrong shape")
    shape = None
    if projected_shape is not None:
        shape = np.asarray(projected_shape, dtype=np.float64).reshape(-1)
        if shape.shape != (x_rule.shape[1],) or np.any(~np.isfinite(shape)):
            raise ValueError("projected Fisher shape must be a finite rule-block vector")
    raw_blocks = (nuisance,) if isinstance(nuisance, np.ndarray) else tuple(nuisance)
    nuisance_blocks: list[np.ndarray] = []
    for raw in raw_blocks:
        block = np.asarray(raw)
        if block.ndim != 2 or block.shape[0] != ctx.n_queries:
            raise ValueError("nuisance design blocks have wrong shape")
        if block.shape[1]:
            nuisance_blocks.append(block)
    nuisance_width = sum(block.shape[1] for block in nuisance_blocks)
    grid = slice(ctx.n_events, None)
    sequence_weights = _query_cluster_weights(ctx, cluster_weights)
    weights = (
        ctx.expand_sequence_values(sequence_weights).astype(np.float64, copy=False)
        * np.exp(np.asarray(eta_null[grid], dtype=np.float64))
    )
    if int(getattr(ctx.grid_weights, "nbytes", 0)) > 0:
        weights *= np.asarray(ctx.grid_weights, dtype=np.float64)
    if nuisance_width:
        gram = np.zeros((nuisance_width, nuisance_width), dtype=np.float64)
        rhs = np.zeros((nuisance_width, x_rule.shape[1]), dtype=np.float64)
        # Bound weighted-design temporaries while accumulating exact sufficient
        # statistics.  Chunking changes summation order only, not the model.
        chunk_size = 1_000_000
        for left in range(ctx.n_events, ctx.n_queries, chunk_size):
            right = min(left + chunk_size, ctx.n_queries)
            f_block = np.concatenate(
                [block[left:right] for block in nuisance_blocks], axis=1
            ).astype(np.float64, copy=False)
            x_block = x_rule[left:right].astype(np.float64, copy=False)
            w_block = weights[left - ctx.n_events : right - ctx.n_events]
            weighted_f = w_block.reshape(-1, 1) * f_block
            gram += f_block.T @ weighted_f
            rhs += weighted_f.T @ x_block
        left, singular, right = np.linalg.svd(gram, full_matrices=False)
        if singular.size and singular[0] > 0:
            tolerance = np.finfo(np.float64).eps * max(gram.shape) * singular[0]
            nuisance_rank = int(np.sum(singular > tolerance))
            inverse = np.divide(
                1.0,
                singular,
                out=np.zeros_like(singular),
                where=singular > tolerance,
            )
            coefficients = right.T @ (
                inverse[:, None] * (left.T @ rhs)
            )
        else:
            nuisance_rank = 0
            coefficients = np.zeros(
                (nuisance_width, x_rule.shape[1]), dtype=np.float64
            )
    else:
        nuisance_rank = 0
        coefficients = np.zeros((0, x_rule.shape[1]), dtype=np.float64)
    residual = (
        np.empty(ctx.n_queries, dtype=np.float64)
        if shape is not None
        else np.empty((ctx.n_queries, x_rule.shape[1]), dtype=np.float64)
    )
    information = np.zeros((x_rule.shape[1], x_rule.shape[1]), dtype=np.float64)
    for left in range(0, ctx.n_queries, 1_000_000):
        right = min(left + 1_000_000, ctx.n_queries)
        residual_block = x_rule[left:right].astype(np.float64, copy=True)
        if nuisance_width:
            f_block = np.concatenate(
                [block[left:right] for block in nuisance_blocks], axis=1
            ).astype(np.float64, copy=False)
            residual_block -= f_block @ coefficients
        if shape is None:
            residual[left:right] = residual_block
        else:
            residual[left:right] = residual_block @ shape
        grid_left = max(left, ctx.n_events)
        if right > grid_left:
            local_left = grid_left - left
            grid_residual = residual_block[local_left:]
            w_block = weights[
                grid_left - ctx.n_events : right - ctx.n_events
            ]
            information += grid_residual.T @ (
                w_block.reshape(-1, 1) * grid_residual
            )
    information = 0.5 * (information + information.T)
    return residual, information, nuisance_rank


def cluster_directional_score(
    residual: np.ndarray,
    eta_null: np.ndarray,
    ctx: QueryContext,
    *,
    sign: int,
    cluster_weights: np.ndarray | None = None,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    eta_null = np.asarray(eta_null, dtype=np.float64)
    event = np.bincount(
        ctx.event_sequence_local,
        weights=residual[: ctx.n_events],
        minlength=ctx.n_sequences,
    )
    grid = ctx.aggregate_weighted_grid(
        np.exp(eta_null[ctx.n_events :]) * residual[ctx.n_events :]
    )
    sequence_weights = _query_cluster_weights(ctx, cluster_weights)
    return sequence_weights * float(sign) * (event - grid)


def numeric_information_positive(
    information: float,
    z: np.ndarray,
    ctx: QueryContext,
    *,
    eta_null: np.ndarray,
    cluster_weights: np.ndarray | None = None,
) -> bool:
    """Whether residual Fisher information is positive beyond roundoff.

    The error scale must use the same intensity and cluster weights as the
    Fisher projection.  Comparing weighted residual information with an
    unweighted exposure norm can otherwise reject or accept a rule solely
    because IPW/financial units changed.
    """
    grid_z = np.asarray(z[ctx.n_events :], dtype=np.float64)
    eta = np.asarray(eta_null, dtype=np.float64)
    if eta.shape != (ctx.n_queries,):
        raise ValueError("null eta must align with the query context")
    sequence_weights = _query_cluster_weights(ctx, cluster_weights)
    with np.errstate(over="ignore", invalid="ignore"):
        raw_by_sequence = ctx.aggregate_weighted_grid(
            np.exp(eta[ctx.n_events :]) * grid_z * grid_z
        )
        raw_information = float(np.dot(sequence_weights, raw_by_sequence))
    if not math.isfinite(raw_information):
        return False
    # Standard forward-error scale for a length-n nonnegative dot product.
    n = max(1, len(grid_z))
    eps = np.finfo(np.float64).eps
    gamma_n = (n * eps) / max(1.0 - n * eps, eps)
    tolerance = gamma_n * max(1.0, raw_information)
    return bool(np.isfinite(information) and information > tolerance)


def numeric_information_positive_from_raw(
    information: float,
    raw_information: float,
    n_grid: int,
) -> bool:
    """The same roundoff contract when raw Fisher information is pre-aggregated."""
    if not math.isfinite(raw_information):
        return False
    n = max(1, int(n_grid))
    eps = np.finfo(np.float64).eps
    gamma_n = (n * eps) / max(1.0 - n * eps, eps)
    tolerance = gamma_n * max(1.0, float(raw_information))
    return bool(np.isfinite(information) and information > tolerance)
