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
    calibration_observations: int | None = None
    calendar_edges: np.ndarray | None = None


@dataclass(frozen=True)
class DensityRatioRobustTest:
    """One-sided entity-cluster test of a Pearson-DRO worst-case gain."""

    entity_count: int
    mean_gain: float
    standard_deviation: float
    radius: float
    robust_gain: float
    standard_error: float
    statistic: float
    pvalue: float
    lower_confidence_bound: float
    testable: bool


def density_ratio_robust_test(
    values: np.ndarray,
    *,
    alpha: float,
    threshold: float = 0.0,
) -> DensityRatioRobustTest:
    r"""Test a support gain against local entity-level distribution shift.

    Let ``x_i`` be the held-out NLL gain for independent entity ``i``.  F3
    uses the Pearson density-ratio ambiguity set

    .. math::

       \mathcal Q_\rho=\{w_i\ge0:\bar w=1,
       \overline{(w_i-1)^2}\le\rho\}.

    Cauchy--Schwarz gives the safe lower bound
    ``mean(x) - sqrt(rho) * std(x)`` for the worst reweighted mean.  The
    radius ``z_(1-alpha)^2 / n`` is the usual local empirical-DRO confidence
    scale; it is outcome-independent and therefore does not manufacture
    calendar regimes.  A cluster influence-function standard error supplies
    the one-sided p-value and influence statistic used by the support IUT and
    Romano--Wolf family correction.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("density-ratio F3 alpha must lie in (0, 1)")
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    count = int(len(sample))
    if count < 3:
        return DensityRatioRobustTest(
            count,
            math.nan,
            math.nan,
            math.nan,
            -math.inf,
            math.inf,
            -math.inf,
            1.0,
            -math.inf,
            False,
        )
    mean = float(np.mean(sample))
    centered = sample - mean
    variance = float(np.mean(centered * centered))
    standard_deviation = math.sqrt(max(0.0, variance))
    if (
        not math.isfinite(standard_deviation)
        or standard_deviation <= np.finfo(float).eps
    ):
        return DensityRatioRobustTest(
            count,
            mean,
            standard_deviation,
            math.nan,
            -math.inf,
            math.inf,
            -math.inf,
            1.0,
            -math.inf,
            False,
        )
    critical = NormalDist().inv_cdf(1.0 - float(alpha))
    radius = (critical * critical) / float(count)
    radius_scale = math.sqrt(radius)
    robust_gain = mean - radius_scale * standard_deviation

    # Influence function of mean(X) - sqrt(rho) * sd(X), with rho fixed at
    # the pre-registered sample-size/alpha calibration above.
    influence = centered - (
        radius_scale
        * ((centered * centered) - variance)
        / (2.0 * standard_deviation)
    )
    influence_sd = float(np.std(influence, ddof=1))
    standard_error = influence_sd / math.sqrt(count)
    if (
        not math.isfinite(standard_error)
        or standard_error <= np.finfo(float).eps
    ):
        return DensityRatioRobustTest(
            count,
            mean,
            standard_deviation,
            radius,
            robust_gain,
            math.inf,
            -math.inf,
            1.0,
            -math.inf,
            False,
        )
    statistic = (robust_gain - float(threshold)) / standard_error
    pvalue = 0.5 * math.erfc(statistic / math.sqrt(2.0))
    lower = robust_gain - critical * standard_error
    return DensityRatioRobustTest(
        count,
        mean,
        standard_deviation,
        radius,
        robust_gain,
        standard_error,
        statistic,
        min(1.0, max(0.0, pvalue)),
        lower,
        True,
    )


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
    calendar_edges: np.ndarray | None = None
    if (
        len(np.unique(raw)) <= 1
        and context.dataset.likelihood == "poisson"
        and len(context.starts)
        and np.all(context.starts == context.starts[0])
        and np.all(context.ends == context.ends[0])
    ):
        # Fixed-population recurrent panels such as IBM AML have no staggered
        # entity-entry cohort.  F3 must therefore compare calendar regimes,
        # not manufacture entity cohorts from identical start times.  The
        # minimum block width is the complete dependency span of one rule:
        # formation may look back max(W), and its kernel may affect the next
        # impact_lag ticks.  Equal-width integer blocks use every observation
        # and are determined without targets or fitted effects.
        dependency = max(
            1,
            (max(config.formation_windows) + config.impact_lag)
            * context.dataset.ticks_per_unit,
        )
        origin = int(context.starts[0])
        stop = int(context.ends[0]) + 1
        span = stop - origin
        count = span // dependency
        if count >= 2:
            quotient, remainder = divmod(span, count)
            widths = np.full(count, quotient, dtype=np.int64)
            widths[:remainder] += 1
            calendar_edges = np.r_[
                np.asarray([origin], dtype=np.int64),
                origin + np.cumsum(widths, dtype=np.int64),
            ]
            raw = np.tile(np.arange(count, dtype=np.int64), len(context.entity_codes))
            labels = np.arange(count, dtype=np.int64)
            counts = np.full(count, len(context.entity_codes), dtype=np.float64)
            probabilities = widths.astype(np.float64) / float(span)
            return EnvironmentSpec(
                inverse=raw,
                labels=labels,
                counts=counts,
                probabilities=probabilities,
                l1_radius=multinomial_l1_radius(
                    len(context.entity_codes), count, confidence_alpha / 2.0
                ),
                source="calendar_time_blocks/max_formation_plus_impact_horizon",
                calibration_observations=len(context.entity_codes),
                calendar_edges=calendar_edges,
            )
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
        calibration_observations=len(raw),
        calendar_edges=calendar_edges,
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


def environment_robust_pvalue(
    values: np.ndarray,
    environments: EnvironmentSpec,
    *,
    threshold: float = 0.0,
) -> tuple[float, bool]:
    """Invert the finite-cohort robust lower bound into one component p-value.

    Both sources of F3 uncertainty vary with the candidate level ``alpha``:
    the BHC ambiguity radius for cohort proportions and the simultaneous
    one-sided cohort-mean bounds.  Their robust lower bound is monotone in
    ``alpha``.  Bisection therefore returns the smallest level at which the
    worst-mixture gain clears ``threshold``.  This lets F1, every F2 branch and
    F3 form one max-p intersection-union test before the frozen support family
    receives its multiplicity correction.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != environments.inverse.shape:
        raise ValueError("entity gain and environment arrays must align")
    count = len(environments.labels)
    if count <= 1 or len(values) < 2:
        return 1.0, False
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
    if not np.all(np.isfinite(standard_errors)):
        return 1.0, False
    observations = (
        int(np.sum(environments.counts))
        if environments.calibration_observations is None
        else int(environments.calibration_observations)
    )

    def passes(alpha: float) -> bool:
        critical = NormalDist().inv_cdf(1.0 - alpha / (2.0 * count))
        lower = means - critical * standard_errors
        radius = multinomial_l1_radius(observations, count, alpha / 2.0)
        worst = worst_case_total_variation_mean(
            lower, environments.probabilities, radius
        )
        return bool(math.isfinite(worst) and worst > float(threshold))

    upper = 1.0 - 8.0 * np.finfo(float).eps
    if not passes(upper):
        return 1.0, True
    lower = max(np.finfo(float).tiny, 1.0e-15)
    if passes(lower):
        return 0.0, True
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        if passes(middle):
            upper = middle
        else:
            lower = middle
    return float(upper), True
