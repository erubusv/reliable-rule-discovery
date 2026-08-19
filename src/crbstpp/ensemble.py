from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csc_matrix

from .config import RunConfig
from .likelihood import cloglog_event_terms, is_poisson_likelihood, loss_rows
from .native import configure_cpu_threads, sorted_unique_union
from .response import Context, ModelMatrix, ResponseEngine
from .rules import EMPTY_SUPPORT, ClosureTerm, RuleIdentity, Support
from .solver import FitResult, fit_model_matrix_continued


@dataclass(frozen=True)
class EnsembleResult:
    supports: tuple[Support, ...]
    fits: tuple[FitResult, ...]
    weights: np.ndarray
    baseline_weight: float
    baseline_fit: FitResult | None
    train_nll: float
    test_nll: float | None
    baseline_test_nll: float | None
    converged: bool
    combination: str = "support_intensity_simplex"
    rule_effects: tuple[RuleIdentity, ...] = ()
    rule_effect_sources: tuple[Support, ...] = ()
    rule_effect_weights: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    support_simplex_train_nll: float | None = None
    support_simplex_test_nll: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "support_count": len(self.supports),
            # Positional weights are meaningless without the frozen support
            # identity at the same index.  Persist the mapping explicitly so
            # inspection and prediction cannot silently associate a weight
            # with the wrong model.
            "supports": [
                [
                    {
                        "antecedent": list(rule.antecedent),
                        "window": rule.window,
                        "sign": rule.sign,
                        "kernel_rank": rule.kernel_rank,
                        "relation": rule.relation,
                        "hierarchical": rule.hierarchical,
                        "support_additive": rule.support_additive,
                        "history_marks": [list(mark) for mark in rule.history_marks],
                    }
                    for rule in support.rules
                ]
                for support in self.supports
            ],
            "weights": self.weights.tolist(),
            "baseline_weight": self.baseline_weight,
            "train_nll": self.train_nll,
            "test_nll": self.test_nll,
            "baseline_test_nll": self.baseline_test_nll,
            "converged": self.converged,
            "combination": self.combination,
            "rule_effects": [
                {
                    "antecedent": list(rule.antecedent),
                    "window": rule.window,
                    "sign": rule.sign,
                    "kernel_rank": rule.kernel_rank,
                    "relation": rule.relation,
                    "hierarchical": rule.hierarchical,
                    "support_additive": rule.support_additive,
                    "history_marks": [list(mark) for mark in rule.history_marks],
                    "source_support": [
                        {
                            "antecedent": list(source_rule.antecedent),
                            "window": source_rule.window,
                            "sign": source_rule.sign,
                            "kernel_rank": source_rule.kernel_rank,
                            "relation": source_rule.relation,
                            "hierarchical": source_rule.hierarchical,
                            "support_additive": source_rule.support_additive,
                            "history_marks": [
                                list(mark) for mark in source_rule.history_marks
                            ],
                        }
                        for source_rule in source.rules
                    ],
                    "weight": float(weight),
                }
                for rule, source, weight in zip(
                    self.rule_effects,
                    self.rule_effect_sources,
                    self.rule_effect_weights,
                    strict=True,
                )
            ],
            "active_rule_effect_count": int(
                np.count_nonzero(self.rule_effect_weights > 1.0e-12)
            ),
            "support_simplex_train_nll": self.support_simplex_train_nll,
            "support_simplex_test_nll": self.support_simplex_test_nll,
        }


@dataclass(frozen=True)
class _RuleEffectStack:
    rules: tuple[RuleIdentity, ...]
    sources: tuple[Support, ...]
    source_indices: tuple[tuple[int, int], ...]
    weights: np.ndarray
    nll: float
    converged: bool
    iterations: int = 0
    warm_start_stationary: bool = False
    device_resident: bool = False


@dataclass(frozen=True)
class _SparseComponents:
    active_intensity: np.ndarray
    baseline_intensity: np.ndarray
    active_event: np.ndarray
    active_noevent: np.ndarray
    inactive_baseline_groups: np.ndarray
    likelihood: str
    tick_exposure: float


@dataclass(frozen=True)
class _MixtureSufficientStatistics:
    """Exact simplex-weight likelihood compressed to target rows.

    For both supported likelihoods every non-target contribution is linear in
    the mixed intensity.  Only target rows are nonlinear.  Consequently one
    component-wise linear coefficient plus the target-row intensities is a
    lossless sufficient statistic for all simplex objective, gradient and KKT
    evaluations.
    """

    linear: np.ndarray
    target_intensity: np.ndarray
    target_weight: np.ndarray
    likelihood: str


@dataclass(frozen=True)
class _SparseProfile:
    """One fitted model represented only on rows differing from its intercept."""

    rows: np.ndarray
    intensity: np.ndarray
    baseline_intensity: np.ndarray


@dataclass(frozen=True)
class _SimplexFit:
    weights: np.ndarray
    nll: float
    converged: bool


@dataclass(frozen=True)
class _IntensityFamilySelection:
    """Finite Add/Drop-stationary intensity-mixture family."""

    indices: tuple[int, ...]
    weights: np.ndarray
    nll: float
    initial_score: float
    score: float
    audits: int
    moves: int


def _fit_frozen_model_with_retry(
    matrix: ModelMatrix,
    *,
    likelihood: str,
    tolerance: float,
    max_iter: int,
    warm_start: np.ndarray | None,
    device: str,
    allow_cpu_fallback: bool = True,
) -> FitResult:
    """Fit one frozen ensemble component without silently losing the model.

    A D_fit optimum can lie numerically on a nonnegative-cone boundary.  When
    continued on D_fit+D_cert, tiny positive warm coefficients may put
    linearly dependent columns on the critical face even though a cold solve
    reaches the same predictive optimum on a lower-dimensional face.  Retry
    from the canonical cold point before treating that model as unidentified.
    A CPU retry distinguishes a device-specific numerical failure from a
    genuine fixed-support failure.  Every attempt solves the identical convex
    likelihood and therefore changes neither the estimator nor its KKT/rank
    requirements.
    """

    attempts = [warm_start]
    if warm_start is not None:
        attempts.append(None)
    devices = [device]
    if allow_cpu_fallback and device != "cpu":
        devices.append("cpu")
    failures: list[str] = []
    for retry_device in devices:
        for start in attempts:
            fit = fit_model_matrix_continued(
                matrix,
                likelihood=likelihood,
                tolerance=tolerance,
                max_iter=max_iter,
                warm_start=start,
                device=retry_device,
            )
            if fit.converged:
                return fit
            label = "warm" if start is not None else "cold"
            failures.append(f"{retry_device}/{label}: {fit.message}")
        # A cold start is canonical.  Do not repeat an absent warm start twice
        # when the caller did not supply one.
        attempts = [None]
    raise RuntimeError(
        "frozen ensemble component failed exact refit after fail-open retries: "
        + "; ".join(failures)
    )


def _simplex_projection(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values.copy()
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    valid = ordered - cumulative / np.arange(1, len(values) + 1) > 0
    rho = np.flatnonzero(valid)[-1]
    threshold = cumulative[rho] / (rho + 1)
    return np.maximum(values - threshold, 0.0)


def _simplex_kkt(weights: np.ndarray, gradient: np.ndarray, objective: float) -> float:
    """Scale-free KKT residual for minimization on the probability simplex."""
    active = weights > 1.0e-10
    if not np.any(active):
        return math.inf
    multiplier = float(np.mean(gradient[active]))
    active_residual = float(np.max(np.abs(gradient[active] - multiplier)))
    inactive = ~active
    inactive_residual = (
        float(np.max(np.maximum(multiplier - gradient[inactive], 0.0)))
        if np.any(inactive)
        else 0.0
    )
    feasibility = max(
        abs(float(np.sum(weights)) - 1.0),
        max(0.0, -float(np.min(weights))),
    )
    return max(active_residual, inactive_residual) / max(
        1.0, abs(objective), feasibility
    )


def _events_at_rows(context: Context, rows: np.ndarray) -> np.ndarray:
    return context.target_counts_at_sorted_rows(rows)


def _sparse_profile(
    engine: ResponseEngine,
    context: Context,
    matrix: ModelMatrix,
    fit: FitResult,
) -> _SparseProfile:
    if matrix.baseline_dimension != engine.baseline_dimension:
        raise ValueError("matrix baseline-control dimension mismatch")
    if matrix.x.shape[0]:
        rows = np.ascontiguousarray(matrix.active_rows, dtype=np.int64)
        eta = engine.linear_predictor_at_rows(context, matrix, fit.coefficients, rows)
    else:
        rows, eta = engine.frozen_active_predictor(context, matrix, fit.coefficients)
    return _SparseProfile(
        rows,
        np.exp(np.clip(eta, -745.0, 700.0)),
        np.exp(
            np.clip(
                fit.coefficients[: engine.free_baseline_dimension],
                -745.0,
                700.0,
            )
        ),
    )


def _components_from_profiles(
    engine: ResponseEngine,
    context: Context,
    profiles: list[_SparseProfile],
) -> _SparseComponents:
    if not profiles:
        raise ValueError("at least one sparse profile is required")
    active_parts = [context.target_rows]
    active_parts.extend(profile.rows for profile in profiles if len(profile.rows))
    rows = sorted_unique_union(active_parts)
    if rows is None:
        rows = np.unique(np.concatenate(active_parts))
    # Profiles use the engine's structural x age x calendar baseline.  Using
    # the legacy static-stratum mapping here assigns all active exposure to the
    # first cells while `_baseline_totals` remains temporal, which can make
    # active mass exceed total observation mass.
    row_groups = context.temporal_baseline_groups_at_rows(
        rows, time_bins=engine.baseline_time_bins
    )
    active_by_group = np.bincount(
        row_groups,
        weights=context.weights_at_rows(rows),
        minlength=engine.free_baseline_dimension,
    ).astype(np.float64)
    inactive_by_group = engine._baseline_totals(context) - active_by_group
    if np.any(inactive_by_group < -1.0e-8):
        raise AssertionError("ensemble active rows exceed the observation grid")
    inactive_by_group = np.maximum(inactive_by_group, 0.0)
    active_intensity = np.empty((len(profiles), len(rows)), dtype=np.float64)
    baseline_intensity = np.empty(
        (len(profiles), engine.free_baseline_dimension), dtype=np.float64
    )
    for index, profile in enumerate(profiles):
        baseline_intensity[index] = profile.baseline_intensity
        active_intensity[index] = profile.baseline_intensity[row_groups]
        if len(profile.rows):
            positions = np.searchsorted(rows, profile.rows)
            active_intensity[index, positions] = profile.intensity
    event = _events_at_rows(context, rows)
    noevent = (
        context.weights_at_rows(rows) - event
        if context.dataset.likelihood == "first_event_cloglog"
        else context.weights_at_rows(rows)
    )
    return _SparseComponents(
        active_intensity,
        baseline_intensity,
        event,
        noevent,
        inactive_by_group,
        context.dataset.likelihood,
        1.0 / context.dataset.ticks_per_unit
        if context.dataset.likelihood == "poisson"
        else 1.0,
    )


def _sparse_components(
    engine: ResponseEngine,
    context: Context,
    matrices: list[ModelMatrix],
    fits: list[FitResult],
) -> _SparseComponents:
    return _components_from_profiles(
        engine,
        context,
        [
            _sparse_profile(engine, context, matrix, fit)
            for matrix, fit in zip(matrices, fits, strict=True)
        ],
    )


def _mixture_nll_gradient(
    components: _SparseComponents, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    active = np.maximum(weights @ components.active_intensity, np.finfo(float).tiny)
    baseline = np.maximum(weights @ components.baseline_intensity, np.finfo(float).tiny)
    if is_poisson_likelihood(components.likelihood):
        nll = float(
            components.tick_exposure * (components.active_noevent @ active)
            - components.active_event @ np.log(active)
            + components.tick_exposure
            * (components.inactive_baseline_groups @ baseline)
        )
        active_derivative = (
            components.tick_exposure * components.active_noevent
            - components.active_event / active
        )
    else:
        event_value, event_first_eta, _ = cloglog_event_terms(np.log(active))
        nll = float(
            components.active_noevent @ active
            + components.active_event @ event_value
            + components.inactive_baseline_groups @ baseline
        )
        active_derivative = (
            components.active_noevent
            + components.active_event * event_first_eta / active
        )
    gradient = (
        components.active_intensity @ active_derivative
        + components.tick_exposure
        * (components.baseline_intensity @ components.inactive_baseline_groups)
    )
    return nll, gradient


def _mixture_sufficient_statistics(
    components: _SparseComponents,
) -> _MixtureSufficientStatistics:
    target = components.active_event > 0.0
    target_intensity = np.ascontiguousarray(
        components.active_intensity[:, target], dtype=np.float64
    )
    target_weight = np.ascontiguousarray(
        components.active_event[target], dtype=np.float64
    )
    if is_poisson_likelihood(components.likelihood):
        linear = components.tick_exposure * (
            components.active_intensity @ components.active_noevent
            + components.baseline_intensity @ components.inactive_baseline_groups
        )
    else:
        linear = (
            components.active_intensity @ components.active_noevent
            + components.baseline_intensity @ components.inactive_baseline_groups
        )
    return _MixtureSufficientStatistics(
        np.ascontiguousarray(linear, dtype=np.float64),
        target_intensity,
        target_weight,
        components.likelihood,
    )


def _mixture_statistics_nll_gradient(
    statistics: _MixtureSufficientStatistics,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    intensity = np.maximum(
        weights @ statistics.target_intensity,
        np.finfo(float).tiny,
    )
    linear_value = float(statistics.linear @ weights)
    if is_poisson_likelihood(statistics.likelihood):
        nll = linear_value - float(statistics.target_weight @ np.log(intensity))
        gradient = statistics.linear - statistics.target_intensity @ (
            statistics.target_weight / intensity
        )
    else:
        event_value, event_first_eta, _ = cloglog_event_terms(np.log(intensity))
        nll = linear_value + float(statistics.target_weight @ event_value)
        gradient = statistics.linear + statistics.target_intensity @ (
            statistics.target_weight * event_first_eta / intensity
        )
    return nll, np.ascontiguousarray(gradient, dtype=np.float64)


def _subset_mixture_statistics(
    statistics: _MixtureSufficientStatistics,
    indices: tuple[int, ...],
) -> _MixtureSufficientStatistics:
    selected = np.asarray(indices, dtype=np.int64)
    return _MixtureSufficientStatistics(
        np.ascontiguousarray(statistics.linear[selected]),
        np.ascontiguousarray(statistics.target_intensity[selected]),
        statistics.target_weight,
        statistics.likelihood,
    )


def _fit_simplex_statistics(
    statistics: _MixtureSufficientStatistics,
    *,
    tolerance: float,
    initial: np.ndarray | None = None,
) -> _SimplexFit:
    component_count = int(statistics.linear.shape[0])
    if component_count < 1:
        raise ValueError("at least one mixture component is required")
    weights = (
        np.full(component_count, 1.0 / component_count, dtype=np.float64)
        if initial is None
        else _simplex_projection(np.asarray(initial, dtype=np.float64))
    )
    if weights.shape != (component_count,):
        raise ValueError("mixture initial weight dimension mismatch")
    result = minimize(
        lambda value: _mixture_statistics_nll_gradient(statistics, value),
        weights,
        jac=True,
        bounds=[(0.0, 1.0)] * component_count,
        constraints={
            "type": "eq",
            "fun": lambda value: float(np.sum(value) - 1.0),
            "jac": lambda value: np.ones_like(value),
        },
        method="SLSQP",
        options={
            "ftol": tolerance,
            "maxiter": 2_000,
            "disp": False,
        },
    )
    if result.x is not None and np.all(np.isfinite(result.x)):
        weights = _simplex_projection(np.asarray(result.x, dtype=np.float64))
    nll, gradient = _mixture_statistics_nll_gradient(statistics, weights)
    converged = _simplex_kkt(weights, gradient, nll) <= 10.0 * tolerance
    for _ in range(0 if converged else 2_000):
        nll, gradient = _mixture_statistics_nll_gradient(statistics, weights)
        if _simplex_kkt(weights, gradient, nll) <= 10.0 * tolerance:
            converged = True
            break
        step = 1.0 / max(1.0, float(np.linalg.norm(gradient)))
        accepted = False
        for _ in range(40):
            trial = _simplex_projection(weights - step * gradient)
            trial_nll, _ = _mixture_statistics_nll_gradient(statistics, trial)
            displacement = trial - weights
            if trial_nll <= nll + 1.0e-4 * float(gradient @ displacement):
                weights, accepted = trial, True
                break
            step *= 0.5
        if not accepted:
            break
    nll, gradient = _mixture_statistics_nll_gradient(statistics, weights)
    converged = _simplex_kkt(weights, gradient, nll) <= 10.0 * tolerance
    return _SimplexFit(weights, nll, converged)


def _fit_simplex_components(
    components: _SparseComponents,
    *,
    tolerance: float,
    initial: np.ndarray | None = None,
) -> _SimplexFit:
    """Exactly optimize fixed component intensities on the simplex.

    The component kernels are frozen; only their nonnegative convex weights
    are optimized.  The Poisson and cloglog mixture NLLs are convex in those
    weights, so the verified simplex KKT residual is a global certificate for
    this small subproblem.
    """
    return _fit_simplex_statistics(
        _mixture_sufficient_statistics(components),
        tolerance=tolerance,
        initial=initial,
    )


def _fit_indexed_simplex_components(
    components: _SparseComponents,
    indices: tuple[int, ...],
    *,
    tolerance: float,
    initial: np.ndarray | None = None,
) -> _SimplexFit:
    """Fit a component subset with one bounded row compaction."""
    component_count = int(components.active_intensity.shape[0])
    selected = np.asarray(indices, dtype=np.int64)
    if (
        not len(selected)
        or np.any(selected < 0)
        or np.any(selected >= component_count)
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("indexed mixture requires unique in-range components")
    if np.array_equal(selected, np.arange(component_count, dtype=np.int64)):
        return _fit_simplex_components(
            components,
            tolerance=tolerance,
            initial=initial,
        )
    # A BLAS product with a full K-vector of mostly zero weights still scans
    # all K intensity rows on every SLSQP iteration.  Copy the selected rows
    # once, then solve the identical smaller convex problem.  The full-family
    # fit above remains zero-copy, so peak storage is bounded by
    # ``full table + largest proper subset``.
    subset = _SparseComponents(
        active_intensity=np.ascontiguousarray(components.active_intensity[selected]),
        baseline_intensity=np.ascontiguousarray(
            components.baseline_intensity[selected]
        ),
        active_event=components.active_event,
        active_noevent=components.active_noevent,
        inactive_baseline_groups=components.inactive_baseline_groups,
        likelihood=components.likelihood,
        tick_exposure=components.tick_exposure,
    )
    return _fit_simplex_components(
        subset,
        tolerance=tolerance,
        initial=initial,
    )


def _select_intensity_family(
    components: _SparseComponents,
    penalties: np.ndarray,
    *,
    baseline_nll: float,
    n_entities: int,
    tolerance: float,
    search_tolerance: float,
) -> _IntensityFamilySelection:
    r"""Select a coded Add/Drop-stationary intensity-mixture family.

    For a frozen family ``F`` this optimizes exactly

    .. math::

       \lambda_{F,w}(t)=\sum_{S\in F}w_S\lambda_S(t)

    and scores it by twice its NLL gain minus all support codes,
    ``(|F|-1) log(N)`` for the extra independently fitted component
    intercepts, and ``(|F|-1) log(N)`` for the free simplex weights. Candidate fits reuse the
    current weights but acceptance depends only on the verified simplex KKT,
    so warm starts alter runtime rather than the selected optimum.
    """
    model_count = int(components.active_intensity.shape[0])
    codes = np.ascontiguousarray(penalties, dtype=np.float64)
    if model_count < 1 or codes.shape != (model_count,):
        raise ValueError("intensity family penalties must align with components")
    if not np.all(np.isfinite(codes)):
        raise ValueError("intensity family penalties must be finite")
    if not math.isfinite(baseline_nll) or n_entities < 1:
        raise ValueError("intensity family requires a finite baseline")
    weight_unit = math.log(max(2, int(n_entities)))
    statistics = _mixture_sufficient_statistics(components)
    cache: dict[tuple[int, ...], tuple[float, _SimplexFit | None]] = {}
    audits = 0
    moves = 0

    def evaluate(
        indices: tuple[int, ...],
        initial: np.ndarray | None = None,
    ) -> tuple[float, _SimplexFit | None]:
        nonlocal audits
        cached = cache.get(indices)
        if cached is not None:
            return cached
        audits += 1
        if not indices:
            result: tuple[float, _SimplexFit | None] = (0.0, None)
            cache[indices] = result
            return result
        fitted = _fit_simplex_statistics(
            _subset_mixture_statistics(statistics, indices),
            tolerance=tolerance,
            initial=initial,
        )
        if not fitted.converged or not math.isfinite(fitted.nll):
            result = (-math.inf, fitted)
            cache[indices] = result
            return result
        code = float(np.sum(codes[np.asarray(indices, dtype=np.int64)]))
        # Each support penalty excludes the one baseline intercept shared by
        # the null model.  A K-component family contains K independently
        # fitted intercepts, hence K-1 additional intercept parameters as
        # well as K-1 free simplex weights.
        code += 2 * max(0, len(indices) - 1) * weight_unit
        score = 2.0 * (float(baseline_nll) - fitted.nll) - code
        result = (float(score), fitted)
        cache[indices] = result
        return result

    def inherited_initial(
        source: tuple[int, ...],
        source_fit: _SimplexFit | None,
        target: tuple[int, ...],
    ) -> np.ndarray | None:
        if source_fit is None:
            return None
        weights = np.zeros(len(target), dtype=np.float64)
        positions = {index: position for position, index in enumerate(target)}
        for index, weight in zip(source, source_fit.weights, strict=True):
            position = positions.get(index)
            if position is not None:
                weights[position] = weight
        total = float(np.sum(weights))
        return None if total <= np.finfo(float).eps else weights / total

    current = tuple(range(model_count))
    current_score, current_fit = evaluate(current)
    initial_score = float(current_score)
    if current_fit is None or not math.isfinite(current_score):
        return _IntensityFamilySelection(
            (),
            np.zeros(0, dtype=np.float64),
            math.inf,
            initial_score,
            -math.inf,
            audits,
            moves,
        )
    zero_tolerance = max(1.0e-12, 10.0 * tolerance)
    zero_pruned_exact = False
    positive = tuple(
        index
        for index, weight in zip(current, current_fit.weights, strict=True)
        if weight > zero_tolerance
    )
    if positive and len(positive) < len(current):
        reduced_score, reduced_fit = evaluate(
            positive,
            inherited_initial(current, current_fit, positive),
        )
        if reduced_fit is not None and reduced_score > current_score + search_tolerance:
            nll_slack = max(
                1.0e-9,
                10.0 * tolerance,
                256.0
                * np.finfo(np.float64).eps
                * max(1.0, abs(current_fit.nll), abs(reduced_fit.nll)),
            )
            zero_pruned_exact = abs(reduced_fit.nll - current_fit.nll) <= nll_slack
            current, current_score, current_fit = (
                positive,
                reduced_score,
                reduced_fit,
            )
            moves += 1

    while True:
        alternatives: list[tuple[float, tuple[int, ...], _SimplexFit | None]] = []
        for index in current:
            trial = tuple(item for item in current if item != index)
            score, fitted = evaluate(
                trial,
                inherited_initial(current, current_fit, trial),
            )
            alternatives.append((score, trial, fitted))
        improving_drops = [
            item
            for item in alternatives
            if item[2] is not None and item[0] > current_score + search_tolerance
        ]
        # If the only move so far removed exact zero-weight components, the
        # reduced predictor is the verified optimum of the original full
        # simplex.  Every excluded Add therefore has nonnegative directional
        # derivative and an additional positive model code: it cannot improve
        # the family score.  Audit Drops only.  Once a positive-weight model is
        # removed, that KKT certificate no longer applies and the complete Add
        # audit below is restored.
        if zero_pruned_exact:
            if not improving_drops:
                break
            next_score, next_family, next_fit = min(
                improving_drops,
                key=lambda item: (-item[0], item[1]),
            )
            if next_fit is None:
                raise AssertionError("improving Drop is missing its simplex fit")
            current, current_score, current_fit = (
                next_family,
                next_score,
                next_fit,
            )
            moves += 1
            zero_pruned_exact = False
            continue
        current_set = set(current)
        for index in range(model_count):
            if index in current_set:
                continue
            trial = tuple(sorted((*current, index)))
            score, fitted = evaluate(
                trial,
                inherited_initial(current, current_fit, trial),
            )
            alternatives.append((score, trial, fitted))
        improving = improving_drops + [
            item
            for item in alternatives[len(current) :]
            if item[2] is not None and item[0] > current_score + search_tolerance
        ]
        if not improving:
            break
        next_score, next_family, next_fit = min(
            improving,
            key=lambda item: (-item[0], item[1]),
        )
        if next_fit is None:
            raise AssertionError("improving family is missing its simplex fit")
        current, current_score, current_fit = (
            next_family,
            next_score,
            next_fit,
        )
        moves += 1

    return _IntensityFamilySelection(
        current,
        np.asarray(current_fit.weights, dtype=np.float64),
        float(current_fit.nll),
        initial_score,
        float(current_score),
        audits,
        moves,
    )


def _fixed_model_mixture(
    engine: ResponseEngine,
    context: Context,
    matrices: list[ModelMatrix],
    fits: list[FitResult],
    *,
    tolerance: float,
) -> _SimplexFit:
    """Fit simplex weights for already fitted support models."""
    if len(matrices) != len(fits) or not matrices:
        raise ValueError("fixed mixture requires aligned nonempty models")
    return _fit_simplex_components(
        _sparse_components(engine, context, matrices, fits),
        tolerance=tolerance,
    )


def _fit_sparse_profiles(
    engine: ResponseEngine,
    context: Context,
    profiles: list[_SparseProfile],
    *,
    tolerance: float,
) -> _SimplexFit:
    """Optimize a fixed-model mixture after dense matrices were released."""
    return _fit_simplex_components(
        _components_from_profiles(engine, context, profiles),
        tolerance=tolerance,
    )


def _rule_effect_design(
    engine: ResponseEngine,
    context: Context,
    support_matrices: list[ModelMatrix],
    support_fits: list[FitResult],
    *,
    frozen_sources: tuple[tuple[int, int], ...] | None = None,
    deduplicate_rules: bool = True,
) -> tuple[
    tuple[RuleIdentity, ...],
    tuple[Support, ...],
    tuple[tuple[int, int], ...],
    np.ndarray,
    csc_matrix,
]:
    """Build sparse fitted-effect columns on one common baseline.

    Final legacy stacking uses one canonical source for each rule identity.
    Unified family discovery instead sets ``deduplicate_rules=False`` so the
    same identity fitted inside two different rule sets remains two distinct
    columns.  This distinction is required for an exact joint-versus-separate
    comparison: the fitted effect of a rule may change after another rule is
    added to its set.
    """

    if frozen_sources is None:
        if deduplicate_rules:
            candidates: dict[RuleIdentity, tuple[int, int]] = {}
            for support_index, matrix in enumerate(support_matrices):
                for rule_index, rule in enumerate(matrix.support.rules):
                    incumbent = candidates.get(rule)
                    if incumbent is None:
                        candidates[rule] = (support_index, rule_index)
                        continue
                    incumbent_support = support_matrices[incumbent[0]].support
                    candidate_key = (
                        len(matrix.support.rules),
                        matrix.support.rules,
                        rule_index,
                    )
                    incumbent_key = (
                        len(incumbent_support.rules),
                        incumbent_support.rules,
                        incumbent[1],
                    )
                    if candidate_key < incumbent_key:
                        candidates[rule] = (support_index, rule_index)
            rules = tuple(sorted(candidates))
            sources = tuple(candidates[rule] for rule in rules)
        else:
            sources = tuple(
                (support_index, rule_index)
                for support_index, matrix in enumerate(support_matrices)
                for rule_index, _ in enumerate(matrix.support.rules)
            )
            rules = tuple(
                support_matrices[support_index].support.rules[rule_index]
                for support_index, rule_index in sources
            )
    else:
        sources = tuple(frozen_sources)
        rules = tuple(
            support_matrices[support_index].support.rules[rule_index]
            for support_index, rule_index in sources
        )

    retained_rules: list[RuleIdentity] = []
    retained_sources: list[tuple[int, int]] = []
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for rule, (support_index, rule_index) in zip(rules, sources, strict=True):
        matrix = support_matrices[support_index]
        fit = support_fits[support_index]
        response_blocks = engine.total_state_rule_blocks(context, matrix.support)
        rows = np.ascontiguousarray(response_blocks[rule_index].rows, dtype=np.int64)
        if not len(rows):
            if frozen_sources is not None:
                retained_rules.append(rule)
                retained_sources.append((support_index, rule_index))
                blocks.append(
                    (
                        rows,
                        np.zeros(0, dtype=np.float64),
                    )
                )
            continue
        values = engine.frozen_contextual_rule_contribution_at_rows(
            context,
            matrix,
            fit.coefficients,
            rule_index=rule_index,
            rows=rows,
        )
        if not np.any(np.abs(values) > 0.0):
            if frozen_sources is not None:
                retained_rules.append(rule)
                retained_sources.append((support_index, rule_index))
                blocks.append((rows, np.zeros(len(rows), dtype=np.float64)))
            continue
        retained_rules.append(rule)
        retained_sources.append((support_index, rule_index))
        blocks.append((rows, np.ascontiguousarray(values, dtype=np.float64)))
    if not blocks:
        return (), (), (), np.zeros(0, dtype=np.int64), csc_matrix((0, 0))

    nonempty = [item[0] for item in blocks if len(item[0])]
    rows = sorted_unique_union(nonempty) if nonempty else None
    if rows is None:
        rows = (
            np.unique(np.concatenate(nonempty))
            if nonempty
            else np.zeros(0, dtype=np.int64)
        )
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    positions = [np.searchsorted(rows, item[0]) for item in blocks]
    row_index = np.concatenate(positions) if positions else np.zeros(0, dtype=np.int64)
    column_index = np.concatenate(
        [
            np.full(len(item[0]), index, dtype=np.int32)
            for index, item in enumerate(blocks)
        ]
    )
    values = np.concatenate([item[1] for item in blocks])
    design = csc_matrix(
        (values, (row_index, column_index)),
        shape=(len(rows), len(blocks)),
    )
    return (
        tuple(retained_rules),
        tuple(
            support_matrices[support_index].support
            for support_index, _ in retained_sources
        ),
        tuple(retained_sources),
        rows,
        design,
    )


def _fit_rule_effect_stack(
    engine: ResponseEngine,
    context: Context,
    baseline_matrix: ModelMatrix,
    baseline_fit: FitResult,
    support_matrices: list[ModelMatrix],
    support_fits: list[FitResult],
    *,
    tolerance: float,
    deduplicate_rules: bool = True,
    precomputed_design: tuple[
        tuple[RuleIdentity, ...],
        tuple[Support, ...],
        tuple[tuple[int, int], ...],
        np.ndarray,
        csc_matrix,
    ]
    | None = None,
    initial_weights: np.ndarray | None = None,
    prefer_warm_newton: bool = False,
    one_step_only: bool = False,
    evaluation_device: str | None = None,
) -> _RuleEffectStack:
    """Fit the exact common-baseline nonnegative rule-effect stack."""

    if precomputed_design is None:
        rules, sources, source_indices, rows, design = _rule_effect_design(
            engine,
            context,
            support_matrices,
            support_fits,
            deduplicate_rules=deduplicate_rules,
        )
    else:
        rules, sources, source_indices, rows, design = precomputed_design
        if design.shape != (len(rows), len(rules)):
            raise ValueError("precomputed rule-effect design shape mismatch")
        if len(sources) != len(rules) or len(source_indices) != len(rules):
            raise ValueError("precomputed rule-effect identities are misaligned")
    if not rules:
        return _RuleEffectStack(
            (), (), (), np.zeros(0, dtype=np.float64), baseline_fit.nll, True
        )
    eta0 = engine.frozen_linear_predictor_at_rows(
        context, baseline_matrix, baseline_fit.coefficients, rows
    )
    row_weight = context.weights_at_rows(rows)
    event_weight = context.target_counts_at_sorted_rows(rows)
    noevent_weight = (
        row_weight - event_weight
        if context.dataset.likelihood == "first_event_cloglog"
        else row_weight
    )
    exposure_weight = engine.tick_exposure * row_weight
    baseline_rows = float(
        np.sum(
            loss_rows(
                eta0,
                likelihood=context.dataset.likelihood,
                exposure_weight=exposure_weight,
                noevent_weight=noevent_weight,
                event_weight=event_weight,
            )[0]
        )
    )

    # Fixed-support Newton already keeps its small-dimensional design resident
    # on CUDA.  Rule-effect stacking has the same geometry but historically
    # fell back to repeated CPU sparse matvecs.  Materialize this tiny-column
    # design once on the assigned GPU and return only scalar/O(p)/O(p^2)
    # results.  Any upload or device failure falls back to the byte-for-byte
    # CPU formulas below.
    torch_state = None
    if (
        evaluation_device is not None
        and str(evaluation_device).startswith("cuda")
        and len(rows)
        and design.shape[1]
    ):
        try:
            import torch

            torch_device = torch.device(str(evaluation_device))
            torch_design = torch.zeros(
                design.shape,
                dtype=torch.float64,
                device=torch_device,
            )
            sparse = design.tocsc()
            for column in range(sparse.shape[1]):
                left = int(sparse.indptr[column])
                right = int(sparse.indptr[column + 1])
                if left == right:
                    continue
                positions = torch.as_tensor(
                    sparse.indices[left:right],
                    dtype=torch.int64,
                    device=torch_device,
                )
                column_values = torch.as_tensor(
                    sparse.data[left:right],
                    dtype=torch.float64,
                    device=torch_device,
                )
                torch_design[positions, column] = column_values
            torch_state = (
                torch,
                torch_design,
                torch.as_tensor(eta0, dtype=torch.float64, device=torch_device),
                torch.as_tensor(
                    exposure_weight, dtype=torch.float64, device=torch_device
                ),
                torch.as_tensor(
                    event_weight, dtype=torch.float64, device=torch_device
                ),
            )
        except (ImportError, RuntimeError, MemoryError):
            torch_state = None

    def derivatives(
        weights: np.ndarray, *, with_hessian: bool
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        if torch_state is not None and is_poisson_likelihood(
            context.dataset.likelihood
        ):
            torch, torch_design, torch_eta0, torch_exposure, torch_events = (
                torch_state
            )
            coefficients = torch.as_tensor(
                np.ascontiguousarray(weights),
                dtype=torch.float64,
                device=torch_design.device,
            )
            eta_device = torch_eta0 + torch_design.mv(coefficients)
            intensity = torch.exp(eta_device)
            first_device = torch_exposure * intensity - torch_events
            value_device = (
                torch_exposure.dot(intensity) - torch_events.dot(eta_device)
            )
            gradient_device = torch_design.T.mv(first_device)
            if not with_hessian:
                return (
                    float(value_device.item()),
                    gradient_device.cpu().numpy(),
                    None,
                )
            second_device = torch_exposure * intensity
            dimension = int(torch_design.shape[1])
            hessian_device = torch.zeros(
                (dimension, dimension),
                dtype=torch.float64,
                device=torch_design.device,
            )
            bytes_per_row = 8 * max(1, dimension)
            row_chunk = max(
                1,
                min(
                    len(rows),
                    max(262_144, (512 * 1024**2) // bytes_per_row),
                ),
            )
            for row_left in range(0, len(rows), row_chunk):
                row_right = min(len(rows), row_left + row_chunk)
                block = torch_design[row_left:row_right]
                hessian_device.add_(
                    block.T.mm(block * second_device[row_left:row_right, None])
                )
            hessian = hessian_device.cpu().numpy()
            return (
                float(value_device.item()),
                gradient_device.cpu().numpy(),
                0.5 * (hessian + hessian.T),
            )
        eta = eta0 + np.asarray(design @ weights, dtype=np.float64).reshape(-1)
        values, first, second = loss_rows(
            eta,
            likelihood=context.dataset.likelihood,
            exposure_weight=exposure_weight,
            noevent_weight=noevent_weight,
            event_weight=event_weight,
        )
        gradient = np.asarray(design.T @ first).reshape(-1)
        if not with_hessian:
            return float(np.sum(values)), gradient, None
        weighted_design = design.multiply(second[:, None])
        hessian = np.asarray((design.T @ weighted_design).toarray(), dtype=np.float64)
        hessian = 0.5 * (hessian + hessian.T)
        return float(np.sum(values)), gradient, hessian

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _ = derivatives(weights, with_hessian=False)
        return value, gradient

    def kkt_residual(weights: np.ndarray, gradient: np.ndarray) -> float:
        active = weights > max(1.0e-10, 10.0 * tolerance)
        active_residual = (
            float(np.max(np.abs(gradient[active]))) if np.any(active) else 0.0
        )
        inactive_residual = (
            float(np.max(np.maximum(-gradient[~active], 0.0)))
            if np.any(~active)
            else 0.0
        )
        return max(active_residual, inactive_residual)

    if initial_weights is None:
        initial = np.ones(len(rules), dtype=np.float64)
    else:
        initial = np.asarray(initial_weights, dtype=np.float64).reshape(-1)
        if len(initial) != len(rules):
            raise ValueError("rule-effect warm start dimension mismatch")
        if not np.all(np.isfinite(initial)):
            raise ValueError("rule-effect warm start must be finite")
        initial = np.maximum(0.0, initial).copy()

    # A neighbouring family differs by only one rule-set block.  Mapping the
    # previous exact solution onto the unchanged columns gives a feasible
    # point.  If the newly introduced zero columns also satisfy cone KKT, this
    # point is already the global optimum of the augmented convex problem and
    # no optimizer call is necessary.
    rows_nll, gradient, _ = derivatives(initial, with_hessian=False)
    full_nll = float(baseline_fit.nll + rows_nll - baseline_rows)
    threshold = 10.0 * tolerance * max(1.0, abs(full_nll))
    warm_start_stationary = bool(
        initial_weights is not None
        and kkt_residual(initial, gradient) <= threshold
    )
    skip_lbfgs = warm_start_stationary
    def projected_newton_refine(
        weights: np.ndarray,
        rows_nll: float,
        gradient: np.ndarray,
        *,
        max_steps: int,
    ) -> tuple[np.ndarray, float, np.ndarray, int]:
        """Damped exact Newton refinement on the nonnegative cone."""

        used = 0
        for _ in range(max_steps):
            full_value = float(baseline_fit.nll + rows_nll - baseline_rows)
            threshold_value = 10.0 * tolerance * max(1.0, abs(full_value))
            if kkt_residual(weights, gradient) <= threshold_value:
                break
            rows_nll, gradient, hessian = derivatives(
                weights, with_hessian=True
            )
            assert hessian is not None
            free = (weights > max(1.0e-10, 10.0 * tolerance)) | (
                gradient < 0.0
            )
            if not np.any(free):
                break
            indices = np.flatnonzero(free)
            block = hessian[np.ix_(indices, indices)]
            scale = max(1.0, float(np.max(np.diag(block))))
            block = block + np.eye(len(indices), dtype=np.float64) * (
                np.finfo(np.float64).eps * scale
            )
            try:
                direction_free = -np.linalg.solve(block, gradient[indices])
            except np.linalg.LinAlgError:
                diagonal = np.maximum(
                    np.diag(block), np.finfo(np.float64).eps * scale
                )
                direction_free = -gradient[indices] / diagonal
            direction = np.zeros_like(weights)
            direction[indices] = direction_free
            projected = np.maximum(0.0, weights + direction)
            direction = projected - weights
            directional = float(gradient @ direction)
            if not np.isfinite(directional) or directional >= 0.0:
                diagonal = np.maximum(
                    np.diag(hessian), np.finfo(np.float64).eps * scale
                )
                projected = np.maximum(0.0, weights - gradient / diagonal)
                direction = projected - weights
                directional = float(gradient @ direction)
            if not np.isfinite(directional) or directional >= 0.0:
                break
            accepted = False
            step = 1.0
            for _ in range(60):
                trial = np.maximum(0.0, weights + step * direction)
                trial_nll, trial_gradient, _ = derivatives(
                    trial, with_hessian=False
                )
                if (
                    np.isfinite(trial_nll)
                    and trial_nll
                    <= rows_nll + 1.0e-4 * step * directional
                ):
                    weights = trial
                    rows_nll = trial_nll
                    gradient = trial_gradient
                    accepted = True
                    used += 1
                    break
                step *= 0.5
            if not accepted:
                break
        return weights, rows_nll, gradient, used

    iterations = 0
    weights = initial
    if one_step_only:
        # A route/terminal Family Block-MDL audit asks whether one feasible
        # projected block step improves the common family objective.  It does
        # not require the whole neighbouring stacking cone to be optimized.
        # The line search evaluates the original likelihood, so a positive
        # score is a genuine feasible improvement.  Full optimization remains
        # mandatory after such a move is selected.
        weights, rows_nll, gradient, used = projected_newton_refine(
            weights, rows_nll, gradient, max_steps=1
        )
        iterations += used
        full_nll = float(baseline_fit.nll + rows_nll - baseline_rows)
        converged = kkt_residual(weights, gradient) <= 10.0 * tolerance * max(
            1.0, abs(full_nll)
        )
        return _RuleEffectStack(
            rules,
            sources,
            source_indices,
            weights,
            full_nll,
            converged,
            iterations,
            warm_start_stationary,
            torch_state is not None,
        )
    if not skip_lbfgs and initial_weights is not None and prefer_warm_newton:
        # A mapped neighbouring optimum is normally already in the local
        # quadratic basin.  Newton reaches the same cone-KKT solution in a few
        # passes; any difficult case falls back to the original L-BFGS path.
        weights, rows_nll, gradient, used = projected_newton_refine(
            weights, rows_nll, gradient, max_steps=24
        )
        iterations += used
        full_nll = float(baseline_fit.nll + rows_nll - baseline_rows)
        threshold = 10.0 * tolerance * max(1.0, abs(full_nll))
        skip_lbfgs = bool(
            kkt_residual(weights, gradient) <= threshold
        )
    if not skip_lbfgs:
        result = minimize(
            objective,
            weights,
            jac=True,
            bounds=[(0.0, None)] * len(rules),
            method="L-BFGS-B",
            # scipy's ftol is relative to the *summed* NLL.  Reusing the model
            # tolerance here made a large Home Credit objective stop while its
            # projected gradient was still material.  Keep objective stopping at
            # machine precision and let the explicit cone-KKT check below decide.
            options={
                "ftol": 1.0e-15,
                "gtol": tolerance,
                "maxiter": 2_000,
                "maxls": 100,
            },
        )
        iterations += int(getattr(result, "nit", 0) or 0)
        weights = np.maximum(
            0.0,
            np.asarray(
                result.x if result.x is not None else weights,
                dtype=np.float64,
            ),
        )
        rows_nll, gradient, _ = derivatives(weights, with_hessian=False)
        full_nll = float(baseline_fit.nll + rows_nll - baseline_rows)
        threshold = 10.0 * tolerance * max(1.0, abs(full_nll))

    # Finish any line-search endpoint with the same exact projected Newton
    # KKT refinement used above.
    weights, rows_nll, gradient, used = projected_newton_refine(
        weights, rows_nll, gradient, max_steps=64
    )
    iterations += used
    full_nll = float(baseline_fit.nll + rows_nll - baseline_rows)
    converged = kkt_residual(weights, gradient) <= 10.0 * tolerance * max(
        1.0, abs(full_nll)
    )
    return _RuleEffectStack(
        rules,
        sources,
        source_indices,
        weights,
        full_nll,
        converged,
        iterations,
        warm_start_stationary,
        torch_state is not None,
    )


def _evaluate_rule_effect_stack(
    engine: ResponseEngine,
    context: Context,
    baseline_matrix: ModelMatrix,
    baseline_fit: FitResult,
    support_matrices: list[ModelMatrix],
    support_fits: list[FitResult],
    stack: _RuleEffectStack,
    *,
    baseline_nll: float,
) -> float:
    if not stack.rules:
        return float(baseline_nll)
    rules, _, _, rows, design = _rule_effect_design(
        engine,
        context,
        support_matrices,
        support_fits,
        frozen_sources=stack.source_indices,
    )
    if rules != stack.rules:
        raise AssertionError("test rule-effect design changed its frozen identity")
    eta0 = engine.frozen_linear_predictor_at_rows(
        context, baseline_matrix, baseline_fit.coefficients, rows
    )
    row_weight = context.weights_at_rows(rows)
    event_weight = context.target_counts_at_sorted_rows(rows)
    noevent_weight = (
        row_weight - event_weight
        if context.dataset.likelihood == "first_event_cloglog"
        else row_weight
    )
    exposure_weight = engine.tick_exposure * row_weight
    baseline_rows = float(
        np.sum(
            loss_rows(
                eta0,
                likelihood=context.dataset.likelihood,
                exposure_weight=exposure_weight,
                noevent_weight=noevent_weight,
                event_weight=event_weight,
            )[0]
        )
    )
    eta = eta0 + np.asarray(design @ stack.weights).reshape(-1)
    stacked_rows = float(
        np.sum(
            loss_rows(
                eta,
                likelihood=context.dataset.likelihood,
                exposure_weight=exposure_weight,
                noevent_weight=noevent_weight,
                event_weight=event_weight,
            )[0]
        )
    )
    return float(baseline_nll + stacked_rows - baseline_rows)


def fit_ensemble(
    dataset_context: Context,
    test_context: Context | None,
    supports: tuple[Support, ...],
    config: RunConfig,
    *,
    closure_signs: dict[ClosureTerm, int] | None = None,
    baseline_warm_start: np.ndarray | None = None,
    support_warm_starts: dict[Support, np.ndarray] | None = None,
    mixture_warm_start: dict[Support, float] | None = None,
) -> EnsembleResult:
    engine = ResponseEngine(
        dataset_context.dataset,
        lag=config.impact_lag,
        knot_count=config.knot_count,
        baseline_time_bins=config.baseline_time_bins,
        effect_model=config.effect_model,
        cache_bytes=config.cache_bytes,
    )
    for term, sign in (closure_signs or {}).items():
        engine.set_closure_sign(term, sign)
    baseline_matrix = engine.model_matrix(dataset_context, EMPTY_SUPPORT)
    baseline_fit = _fit_frozen_model_with_retry(
        baseline_matrix,
        likelihood=dataset_context.dataset.likelihood,
        tolerance=config.solver_tolerance,
        max_iter=config.solver_max_iter,
        warm_start=baseline_warm_start,
        device=(config.pricing_devices or ("cpu",))[0],
    )
    support_fits: list[FitResult] = []
    support_matrices: list[ModelMatrix] = []
    cuda_slots = len(
        {device for device in config.pricing_devices if device.startswith("cuda")}
    )
    physical_exact_workers = (
        min(config.exact_workers, cuda_slots) if cuda_slots else config.exact_workers
    )
    for start in range(0, len(supports), physical_exact_workers):
        support_wave = supports[start : start + physical_exact_workers]
        devices = config.pricing_devices or ("cpu",)
        threads_per_fit = max(1, config.pricing_workers // physical_exact_workers)

        def build_and_fit(
            item: tuple[int, Support],
        ) -> tuple[ModelMatrix, FitResult | None, RuntimeError | None]:
            index, support = item
            configure_cpu_threads(threads_per_fit)
            # Certified supports are immutable total-rule models.  The
            # ResponseEngine synchronizes its shared response caches; once
            # built, each fixed design is owned by this worker's solver.
            matrix = engine.model_matrix(dataset_context, support)
            try:
                fit = _fit_frozen_model_with_retry(
                    matrix,
                    likelihood=dataset_context.dataset.likelihood,
                    tolerance=config.solver_tolerance,
                    max_iter=config.solver_max_iter,
                    warm_start=(
                        None
                        if support_warm_starts is None
                        else support_warm_starts.get(support)
                    ),
                    device=devices[index % len(devices)],
                    allow_cpu_fallback=False,
                )
            except RuntimeError as error:
                # Do not launch a CPU fallback while another CUDA worker is
                # still mutating process-global native thread/workspace state.
                # Return the failure and retry it serially after the wave has
                # joined.  The retry solves the identical fixed design.
                return matrix, None, error
            return matrix, fit, None

        indexed_wave = list(enumerate(support_wave))
        if len(indexed_wave) == 1:
            completed_wave = [build_and_fit(indexed_wave[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=len(indexed_wave),
                thread_name_prefix="crbstpp-ensemble",
            ) as executor:
                completed_wave = list(executor.map(build_and_fit, indexed_wave))
        matrix_wave = [item[0] for item in completed_wave]
        fit_wave: list[FitResult] = []
        for wave_index, (support, (matrix, fit, parallel_error)) in enumerate(
            zip(support_wave, completed_wave, strict=True)
        ):
            if fit is None:
                configure_cpu_threads(config.pricing_workers)
                try:
                    # Rebuild the same sufficient statistics canonically.
                    # Incremental group splitting is algebraically exact, but
                    # its different summation order can leave a boundary KKT
                    # residual above a strict tolerance on very uneven
                    # weights.  The canonical design is a numerical fail-open,
                    # not a different model or objective.
                    matrix = engine.model_matrix(
                        dataset_context,
                        support,
                        _allow_extension=False,
                    )
                    matrix_wave[wave_index] = matrix
                    fit = _fit_frozen_model_with_retry(
                        matrix,
                        likelihood=dataset_context.dataset.likelihood,
                        tolerance=config.solver_tolerance,
                        max_iter=config.solver_max_iter,
                        warm_start=None,
                        device="cpu",
                    )
                except RuntimeError as serial_error:
                    raise RuntimeError(
                        "certified support failed frozen ensemble refit: "
                        f"support={support!r}; parallel={parallel_error}; "
                        f"serial_cpu={serial_error}"
                    ) from serial_error
            fit_wave.append(fit)
        for matrix, fit in zip(matrix_wave, fit_wave, strict=True):
            support_fits.append(fit)
            support_matrices.append(matrix)
    if len(support_matrices) != len(supports):
        raise AssertionError("certified support was omitted from ensemble refit")
    matrices = [baseline_matrix, *support_matrices]
    fits = [baseline_fit, *support_fits]
    retained_supports = tuple(matrix.support for matrix in support_matrices)
    model_specs = tuple((matrix.support, matrix.closure) for matrix in matrices)
    components = (
        None
        if config.posthoc_rule_effect_stacking
        else _sparse_components(engine, dataset_context, matrices, fits)
    )
    rule_stack = (
        _fit_rule_effect_stack(
            engine,
            dataset_context,
            baseline_matrix,
            baseline_fit,
            support_matrices,
            support_fits,
            tolerance=config.solver_tolerance,
            deduplicate_rules=False,
        )
        if (
            config.rule_effect_stacking_search
            or config.posthoc_rule_effect_stacking
        )
        else None
    )
    # Sparse mixture components and fitted matrices are self-contained.  The
    # potentially multi-GiB response/completion LRU is no longer useful and
    # must not overlap the independently cached D_test engine below.
    engine.clear_caches()
    # Only frozen coefficients, support identities and mixture intensities are
    # needed from this point.  Release all training ModelMatrix objects before
    # simplex optimization and D_test construction.
    del baseline_matrix, support_matrices, matrices
    simplex = None
    if not config.posthoc_rule_effect_stacking:
        simplex_initial: np.ndarray | None = None
        if mixture_warm_start:
            support_initial = np.asarray(
                [
                    max(0.0, float(mixture_warm_start.get(support, 0.0)))
                    for support in retained_supports
                ],
                dtype=np.float64,
            )
            support_total = float(np.sum(support_initial))
            if support_total > 0.0:
                baseline_initial = max(0.0, 1.0 - support_total)
                simplex_initial = np.r_[baseline_initial, support_initial]
                simplex_initial /= float(np.sum(simplex_initial))
        if components is None:
            raise AssertionError("support-simplex components were not built")
        simplex = _fit_simplex_components(
            components,
            tolerance=config.solver_tolerance,
            initial=simplex_initial,
        )
    if (
        simplex is not None
        and config.ensemble_irreducible_family
        and retained_supports
    ):
        zero_tolerance = max(1.0e-12, 10.0 * config.solver_tolerance)
        active_positions = tuple(
            index
            for index, weight in enumerate(simplex.weights[1:])
            if float(weight) > zero_tolerance
        )
        if len(active_positions) < len(retained_supports):
            # Components are fixed after D_fit+D_cert support refitting, so
            # zero-weight pruning needs only another exact convex simplex fit;
            # no support model is fitted twice. Baseline remains component 0.
            component_indices = (0,) + tuple(
                position + 1 for position in active_positions
            )
            statistics = _mixture_sufficient_statistics(components)
            initial = np.asarray(
                simplex.weights[np.asarray(component_indices, dtype=np.int64)],
                dtype=np.float64,
            )
            initial /= float(np.sum(initial))
            simplex = _fit_simplex_statistics(
                _subset_mixture_statistics(statistics, component_indices),
                tolerance=config.solver_tolerance,
                initial=initial,
            )
            retained_supports = tuple(
                retained_supports[position] for position in active_positions
            )
            support_fits = [support_fits[position] for position in active_positions]
            model_specs = (model_specs[0],) + tuple(
                model_specs[position + 1] for position in active_positions
            )
            fits = [baseline_fit, *support_fits]
    weights = (
        simplex.weights
        if simplex is not None
        else np.r_[1.0, np.zeros(len(retained_supports), dtype=np.float64)]
    )
    converged = True if simplex is None else simplex.converged
    support_simplex_train_nll = None if simplex is None else simplex.nll
    train_nll = (
        rule_stack.nll
        if rule_stack is not None
        else float(support_simplex_train_nll)
    )
    # The learned simplex weights are frozen before D_test.  Training
    # intensities can therefore be released, preventing train/test
    # ``models x active_rows`` arrays from overlapping at peak memory.
    components = None
    test_nll = baseline_test_nll = None
    support_simplex_test_nll = None
    if test_context is not None:
        test_engine = ResponseEngine(
            test_context.dataset,
            lag=config.impact_lag,
            knot_count=config.knot_count,
            baseline_time_bins=config.baseline_time_bins,
            effect_model=config.effect_model,
            cache_bytes=config.cache_bytes,
        )
        for term, sign in (closure_signs or {}).items():
            test_engine.set_closure_sign(term, sign)
        test_matrices = [
            test_engine.model_metadata(support, forced_closure=closure)
            for support, closure in model_specs
        ]
        test_components = None
        if simplex is not None:
            test_components = _sparse_components(
                test_engine, test_context, test_matrices, fits
            )
            support_simplex_test_nll = _mixture_nll_gradient(
                test_components, weights
            )[0]
        baseline_components = _sparse_components(
            test_engine,
            test_context,
            test_matrices[:1],
            fits[:1],
        )
        baseline_test_nll = _mixture_nll_gradient(
            baseline_components, np.ones(1, dtype=np.float64)
        )[0]
        if rule_stack is not None:
            test_nll = _evaluate_rule_effect_stack(
                test_engine,
                test_context,
                test_matrices[0],
                baseline_fit,
                test_matrices[1:],
                support_fits,
                rule_stack,
                baseline_nll=baseline_test_nll,
            )
        else:
            test_nll = support_simplex_test_nll
        test_engine.clear_caches()
        del test_matrices
    return EnsembleResult(
        retained_supports,
        tuple(support_fits),
        weights[1:],
        float(weights[0]),
        baseline_fit,
        train_nll,
        test_nll,
        baseline_test_nll,
        converged and (rule_stack is None or rule_stack.converged),
        combination=(
            "common_baseline_nonnegative_rule_effect_stack"
            if rule_stack is not None
            else "support_intensity_simplex"
        ),
        rule_effects=() if rule_stack is None else rule_stack.rules,
        rule_effect_sources=() if rule_stack is None else rule_stack.sources,
        rule_effect_weights=(
            np.zeros(0, dtype=np.float64) if rule_stack is None else rule_stack.weights
        ),
        support_simplex_train_nll=support_simplex_train_nll,
        support_simplex_test_nll=support_simplex_test_nll,
    )
