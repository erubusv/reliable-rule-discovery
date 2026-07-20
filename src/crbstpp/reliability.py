from __future__ import annotations

import math
from statistics import NormalDist
from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .response import Context


@dataclass(frozen=True)
class EnvironmentSpec:
    """Outcome-blind temporal environments and their finite-sample mixture set."""

    inverse: np.ndarray
    labels: np.ndarray
    counts: np.ndarray
    probabilities: np.ndarray
    l1_radius: float
    source: str


def one_sided_mean_pvalue(values: np.ndarray) -> tuple[float, bool]:
    """Normal one-sided mean-test p-value used by fit-side vector pricing."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 1.0, False
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    if (
        not math.isfinite(standard_deviation)
        or standard_deviation <= np.finfo(float).eps
    ):
        return 1.0, False
    statistic = mean / (standard_deviation / math.sqrt(len(values)))
    pvalue = 0.5 * math.erfc(statistic / math.sqrt(2.0))
    return min(1.0, max(0.0, pvalue)), True


def multinomial_l1_radius(n: int, environments: int, alpha: float) -> float:
    """Bretagnolle--Huber--Carol confidence radius for mixture weights."""
    if n < 1:
        raise ValueError("environment calibration requires observations")
    if environments <= 1:
        return 0.0
    radius = math.sqrt(
        2.0
        * (float(environments) * math.log(2.0) + math.log(1.0 / float(alpha)))
        / float(n)
    )
    return min(2.0, radius)


def environment_spec(
    context: Context, config: RunConfig, *, alpha: float | None = None
) -> EnvironmentSpec:
    """Construct the same pre-registered environments for search and certification."""
    confidence_alpha = config.alpha if alpha is None else float(alpha)
    if not 0.0 < confidence_alpha < 1.0:
        raise ValueError("environment confidence alpha must lie in (0, 1)")
    raw = context.dataset.split_groups[context.entity_codes]
    source = "dataset.split_groups"
    if len(np.unique(raw)) <= 1:
        width = max(
            1,
            (max(config.formation_windows) + config.impact_lag)
            * context.dataset.ticks_per_unit,
        )
        origin = int(np.min(context.starts))
        raw = (context.starts - origin) // width
        source = "entity_start_time/formation_plus_impact_horizon"
    labels, inverse, counts = np.unique(raw, return_inverse=True, return_counts=True)
    counts = counts.astype(np.float64)
    probabilities = counts / float(np.sum(counts))
    return EnvironmentSpec(
        inverse=np.asarray(inverse, dtype=np.int64),
        labels=np.asarray(labels),
        counts=counts,
        probabilities=probabilities,
        # Half of alpha covers estimation of the empirical mixture weights;
        # the other half is reserved for simultaneous cohort-mean LCBs.
        l1_radius=multinomial_l1_radius(
            len(raw), len(labels), confidence_alpha / 2.0
        ),
        source=source,
    )


def worst_case_total_variation_mean(
    values: np.ndarray,
    probabilities: np.ndarray,
    l1_radius: float,
) -> float:
    """Exact minimum mean over an L1 ball on a finite environment simplex."""
    values = np.asarray(values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 1
        or probabilities.shape != values.shape
        or not len(values)
        or np.any(probabilities < 0.0)
        or not np.isclose(float(np.sum(probabilities)), 1.0)
        or not 0.0 <= l1_radius <= 2.0
    ):
        raise ValueError("invalid finite-environment ambiguity problem")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("finite-environment values may contain only finite or -inf values")
    if np.any(np.isneginf(values) & (probabilities > 0.0)):
        return -math.inf
    if len(values) == 1 or l1_radius == 0.0:
        return float(probabilities @ values)
    weights = probabilities.copy()
    low_order = np.argsort(values, kind="stable")
    high_order = low_order[::-1]
    budget = min(1.0, 0.5 * float(l1_radius))
    low_index = high_index = 0
    tolerance = 16.0 * np.finfo(float).eps
    while budget > tolerance:
        while (
            low_index < len(values) and weights[low_order[low_index]] >= 1.0 - tolerance
        ):
            low_index += 1
        while high_index < len(values) and weights[high_order[high_index]] <= tolerance:
            high_index += 1
        if low_index >= len(values) or high_index >= len(values):
            break
        low = int(low_order[low_index])
        high = int(high_order[high_index])
        if low == high or values[low] >= values[high]:
            break
        moved = min(budget, 1.0 - weights[low], weights[high])
        if moved <= tolerance:
            break
        weights[low] += moved
        weights[high] -= moved
        budget -= moved
    return float(weights @ values)


def environment_robust_mean(
    values: np.ndarray, environments: EnvironmentSpec
) -> tuple[float, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != environments.inverse.shape:
        raise ValueError("entity gain and environment arrays must align")
    sums = np.bincount(
        environments.inverse,
        weights=values,
        minlength=len(environments.labels),
    )
    means = sums / environments.counts
    worst = worst_case_total_variation_mean(
        means, environments.probabilities, environments.l1_radius
    )
    return worst, means


def environment_robust_lcb(
    values: np.ndarray,
    environments: EnvironmentSpec,
    *,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Worst mixture mean after simultaneous environment-mean uncertainty.

    The previous F3 implementation perturbed only empirical cohort proportions
    and treated estimated cohort effects as known.  This function first forms
    Bonferroni simultaneous one-sided normal lower bounds for every cohort mean,
    then minimizes those bounds over the finite-sample mixture ambiguity set.
    The split ``alpha/2`` is paired with the BHC mixture-radius calibration in
    :func:`environment_spec`.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != environments.inverse.shape:
        raise ValueError("entity gain and environment arrays must align")
    count = len(environments.labels)
    if count <= 1:
        return -math.inf, np.full(count, math.nan), np.full(count, -math.inf)
    sums = np.bincount(
        environments.inverse, weights=values, minlength=count
    ).astype(np.float64)
    squared = np.bincount(
        environments.inverse, weights=values * values, minlength=count
    ).astype(np.float64)
    means = sums / environments.counts
    numerators = np.maximum(
        0.0, squared - environments.counts * means * means
    )
    variances = np.divide(
        numerators,
        environments.counts - 1.0,
        out=np.full(count, math.inf, dtype=np.float64),
        where=environments.counts > 1.0,
    )
    standard_errors = np.sqrt(variances / environments.counts)
    tail = float(alpha) / (2.0 * count)
    critical = NormalDist().inv_cdf(1.0 - tail)
    lower = means - critical * standard_errors
    lower[~np.isfinite(lower)] = -math.inf
    worst = worst_case_total_variation_mean(
        lower, environments.probabilities, environments.l1_radius
    )
    return worst, means, lower
