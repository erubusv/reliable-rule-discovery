from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import RunConfig
from .likelihood import cloglog_event_terms
from .response import Context, ModelMatrix, ResponseEngine
from .rules import EMPTY_SUPPORT, Support
from .solver import FitResult, fit_model_matrices, fit_model_matrix_continued


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
        }


@dataclass(frozen=True)
class _SparseComponents:
    active_intensity: np.ndarray
    baseline_intensity: np.ndarray
    active_event: np.ndarray
    active_noevent: np.ndarray
    inactive_by_age: np.ndarray
    likelihood: str
    tick_exposure: float


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


def _simplex_kkt(
    weights: np.ndarray, gradient: np.ndarray, objective: float
) -> float:
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
    return max(active_residual, inactive_residual) / max(1.0, abs(objective), feasibility)


def _events_at_rows(context: Context, rows: np.ndarray) -> np.ndarray:
    event = np.zeros(len(rows), dtype=np.float64)
    if not len(rows) or not len(context.target_rows):
        return event
    positions = np.searchsorted(context.target_rows, rows)
    matched = positions < len(context.target_rows)
    safe = np.minimum(positions, len(context.target_rows) - 1)
    matched &= context.target_rows[safe] == rows
    event[matched] = context.target_counts[positions[matched]]
    return event


def _sparse_components(
    engine: ResponseEngine,
    context: Context,
    matrices: list[ModelMatrix],
    fits: list[FitResult],
) -> _SparseComponents:
    active_parts = [context.target_rows]
    active_parts.extend(
        matrix.active_rows for matrix in matrices if len(matrix.active_rows)
    )
    rows = (
        np.unique(np.concatenate(active_parts))
        if active_parts
        else np.zeros(0, dtype=np.int64)
    )
    local, times = context.rows_to_entity_time(rows)
    ages = context.baseline_origins[local] + times - context.starts[local]
    age_bins = np.minimum(ages // engine.lag, engine.baseline_dimension - 1).astype(
        np.int64
    )
    active_by_age = np.bincount(age_bins, minlength=engine.baseline_dimension).astype(
        np.float64
    )
    inactive_by_age = engine._baseline_totals(context) - active_by_age
    if np.any(inactive_by_age < -1.0e-8):
        raise AssertionError("ensemble active rows exceed the observation grid")
    inactive_by_age = np.maximum(inactive_by_age, 0.0)
    active_intensity = []
    baseline_intensity = []
    for matrix, fit in zip(matrices, fits, strict=True):
        eta = engine.linear_predictor_at_rows(context, matrix, fit.coefficients, rows)
        active_intensity.append(np.exp(np.clip(eta, -745.0, 700.0)))
        baseline_dimension = matrix.free_dimension - matrix.closure_dimension
        baseline_eta = np.full(
            engine.baseline_dimension, fit.coefficients[0], dtype=np.float64
        )
        if baseline_dimension > 1:
            baseline_eta[1:baseline_dimension] += fit.coefficients[1:baseline_dimension]
        baseline_intensity.append(np.exp(np.clip(baseline_eta, -745.0, 700.0)))
    event = _events_at_rows(context, rows)
    noevent = (
        1.0 - event
        if context.dataset.likelihood == "first_event_cloglog"
        else np.ones(len(rows))
    )
    return _SparseComponents(
        np.asarray(active_intensity, dtype=np.float64),
        np.asarray(baseline_intensity, dtype=np.float64),
        event,
        noevent,
        inactive_by_age,
        context.dataset.likelihood,
        1.0 / context.dataset.ticks_per_unit
        if context.dataset.likelihood == "poisson"
        else 1.0,
    )


def _mixture_nll_gradient(
    components: _SparseComponents, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    active = np.maximum(weights @ components.active_intensity, np.finfo(float).tiny)
    baseline = np.maximum(weights @ components.baseline_intensity, np.finfo(float).tiny)
    if components.likelihood == "poisson":
        nll = float(
            components.tick_exposure * np.sum(active)
            - components.active_event @ np.log(active)
            + components.tick_exposure * (components.inactive_by_age @ baseline)
        )
        active_derivative = components.tick_exposure - components.active_event / active
    else:
        event_value, event_first_eta, _ = cloglog_event_terms(np.log(active))
        nll = float(
            components.active_noevent @ active
            + components.active_event @ event_value
            + components.inactive_by_age @ baseline
        )
        active_derivative = (
            components.active_noevent
            + components.active_event * event_first_eta / active
        )
    gradient = (
        components.active_intensity @ active_derivative
        + components.tick_exposure
        * (components.baseline_intensity @ components.inactive_by_age)
    )
    return nll, gradient


def _evaluate_mixture(
    engine: ResponseEngine,
    context: Context,
    matrices: list[ModelMatrix],
    fits: list[FitResult],
    weights: np.ndarray,
) -> float:
    return _mixture_nll_gradient(
        _sparse_components(engine, context, matrices, fits), weights
    )[0]


def fit_ensemble(
    dataset_context: Context,
    test_context: Context | None,
    supports: tuple[Support, ...],
    config: RunConfig,
) -> EnsembleResult:
    engine = ResponseEngine(
        dataset_context.dataset,
        lag=config.impact_lag,
        knot_count=config.knot_count,
        cache_bytes=config.cache_bytes,
    )
    baseline_matrix = engine.model_matrix(dataset_context, EMPTY_SUPPORT)
    baseline_fit = fit_model_matrix_continued(
        baseline_matrix,
        likelihood=dataset_context.dataset.likelihood,
        tolerance=config.solver_tolerance,
        max_iter=config.solver_max_iter,
        device=(config.pricing_devices or ("cpu",))[0],
    )
    if not baseline_fit.converged:
        return EnsembleResult(
            (), (), np.zeros(0), 0.0, baseline_fit, math.inf, None, None, False
        )
    support_fits: list[FitResult] = []
    support_matrices: list[ModelMatrix] = []
    for start in range(0, len(supports), config.exact_workers):
        support_wave = supports[start : start + config.exact_workers]
        matrix_wave = [
            engine.model_matrix(dataset_context, support) for support in support_wave
        ]
        fit_wave = fit_model_matrices(
            matrix_wave,
            likelihood=dataset_context.dataset.likelihood,
            tolerance=config.solver_tolerance,
            max_iter=config.solver_max_iter,
            workers=config.exact_workers,
            cpu_threads_per_worker=max(
                1, config.pricing_workers // config.exact_workers
            ),
            devices=config.pricing_devices or ("cpu",),
        )
        for matrix, fit in zip(matrix_wave, fit_wave, strict=True):
            if fit.converged:
                support_fits.append(fit)
                support_matrices.append(matrix)
    matrices = [baseline_matrix, *support_matrices]
    fits = [baseline_fit, *support_fits]
    components = _sparse_components(engine, dataset_context, matrices, fits)
    weights = np.full(len(fits), 1.0 / len(fits), dtype=np.float64)
    result = minimize(
        lambda value: _mixture_nll_gradient(components, value),
        weights,
        jac=True,
        bounds=[(0.0, 1.0)] * len(weights),
        constraints={
            "type": "eq",
            "fun": lambda value: float(np.sum(value) - 1.0),
            "jac": lambda value: np.ones_like(value),
        },
        method="SLSQP",
        options={
            "ftol": config.solver_tolerance,
            "maxiter": 2_000,
            "disp": False,
        },
    )
    if result.x is not None and np.all(np.isfinite(result.x)):
        weights = _simplex_projection(np.asarray(result.x, dtype=np.float64))
    nll, gradient = _mixture_nll_gradient(components, weights)
    converged = _simplex_kkt(weights, gradient, nll) <= (
        10.0 * config.solver_tolerance
    )
    # SLSQP is normally definitive, but retain a deterministic projected-
    # gradient fallback and accept only an explicitly verified simplex KKT.
    for _ in range(0 if converged else 2_000):
        nll, gradient = _mixture_nll_gradient(components, weights)
        if _simplex_kkt(weights, gradient, nll) <= 10.0 * config.solver_tolerance:
            converged = True
            break
        step = 1.0 / max(1.0, float(np.linalg.norm(gradient)))
        accepted = False
        trial_nll = nll
        for _ in range(40):
            trial = _simplex_projection(weights - step * gradient)
            trial_nll, _ = _mixture_nll_gradient(components, trial)
            displacement = trial - weights
            if trial_nll <= nll + 1.0e-4 * float(gradient @ displacement):
                weights, accepted = trial, True
                break
            step *= 0.5
        if not accepted:
            converged = _simplex_kkt(weights, gradient, nll) <= (
                10.0 * config.solver_tolerance
            )
            break
        _, trial_gradient = _mixture_nll_gradient(components, weights)
        if _simplex_kkt(weights, trial_gradient, trial_nll) <= (
            10.0 * config.solver_tolerance
        ):
            converged = True
            break
    train_nll, _ = _mixture_nll_gradient(components, weights)
    test_nll = baseline_test_nll = None
    if test_context is not None:
        test_engine = ResponseEngine(
            test_context.dataset,
            lag=config.impact_lag,
            knot_count=config.knot_count,
            cache_bytes=config.cache_bytes,
        )
        test_matrices = [
            test_engine.model_matrix(
                test_context, matrix.support, forced_closure=matrix.closure
            )
            for matrix in matrices
        ]
        test_nll = _evaluate_mixture(
            test_engine, test_context, test_matrices, fits, weights
        )
        baseline_test_matrix = test_engine.model_matrix(test_context, EMPTY_SUPPORT)
        baseline_test_nll = _evaluate_mixture(
            test_engine,
            test_context,
            [baseline_test_matrix],
            [baseline_fit],
            np.ones(1, dtype=np.float64),
        )
    retained_supports = tuple(matrix.support for matrix in support_matrices)
    return EnsembleResult(
        retained_supports,
        tuple(support_fits),
        weights[1:],
        float(weights[0]),
        baseline_fit,
        train_nll,
        test_nll,
        baseline_test_nll,
        converged,
    )
