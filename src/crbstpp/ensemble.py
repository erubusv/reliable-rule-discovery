from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .likelihood import cloglog_event_terms
from .response import Context, ResponseEngine
from .rules import Support
from .solver import FitResult, fit_model_matrix


@dataclass(frozen=True)
class EnsembleResult:
    supports: tuple[Support, ...]
    fits: tuple[FitResult, ...]
    weights: np.ndarray
    train_nll: float
    test_nll: float | None
    baseline_test_nll: float | None
    converged: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "support_count": len(self.supports),
            "weights": self.weights.tolist(),
            "train_nll": self.train_nll,
            "test_nll": self.test_nll,
            "baseline_test_nll": self.baseline_test_nll,
            "converged": self.converged,
        }


def _simplex_projection(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    valid = ordered - cumulative / np.arange(1, len(values) + 1) > 0
    rho = np.flatnonzero(valid)[-1]
    threshold = cumulative[rho] / (rho + 1)
    return np.maximum(values - threshold, 0.0)


def _mixture_nll_gradient(
    component_intensity: np.ndarray,
    context: Context,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    mixture = np.maximum(weights @ component_intensity, np.finfo(float).tiny)
    target = np.zeros(context.n_grid, dtype=np.float64)
    if len(context.target_rows):
        np.add.at(target, context.target_rows, context.target_counts)
    if context.dataset.likelihood == "poisson":
        nll = float(np.sum(mixture) - target @ np.log(mixture))
        derivative = np.ones(context.n_grid) - target / mixture
    else:
        noevent = 1.0 - target
        event_value, event_first_eta, _ = cloglog_event_terms(np.log(mixture))
        nll = float(noevent @ mixture + target @ event_value)
        derivative = noevent + target * event_first_eta / mixture
    return nll, component_intensity @ derivative


def fit_ensemble(
    dataset_context: Context,
    test_context: Context | None,
    supports: tuple[Support, ...],
    config: RunConfig,
) -> EnsembleResult:
    if not supports:
        return EnsembleResult((), (), np.zeros(0), math.inf, None, None, False)
    engine = ResponseEngine(
        dataset_context.dataset,
        lag=config.impact_lag,
        knot_count=config.knot_count,
        cache_bytes=config.cache_bytes,
    )
    fits: list[FitResult] = []
    intensities: list[np.ndarray] = []
    matrices = []
    for support in supports:
        matrix = engine.model_matrix(dataset_context, support)
        fit = fit_model_matrix(
            matrix,
            likelihood=dataset_context.dataset.likelihood,
            tolerance=config.solver_tolerance,
            max_iter=config.solver_max_iter,
        )
        if not fit.converged:
            continue
        fits.append(fit)
        matrices.append(matrix)
        eta = engine.linear_predictor(dataset_context, matrix, fit.coefficients)
        intensities.append(np.exp(np.clip(eta, -745.0, 700.0)))
    if not fits:
        return EnsembleResult((), (), np.zeros(0), math.inf, None, None, False)
    components = np.stack(intensities)
    weights = np.full(len(fits), 1.0 / len(fits), dtype=np.float64)
    previous = math.inf
    converged = False
    for _ in range(500):
        nll, gradient = _mixture_nll_gradient(components, dataset_context, weights)
        step = 1.0 / max(1.0, float(np.linalg.norm(gradient)))
        accepted = False
        for _ in range(40):
            trial = _simplex_projection(weights - step * gradient)
            trial_nll, _ = _mixture_nll_gradient(components, dataset_context, trial)
            if trial_nll <= nll - 1.0e-4 * float(np.sum((weights - trial) ** 2)) / step:
                weights, previous, accepted = trial, nll, True
                break
            step *= 0.5
        if not accepted:
            break
        if abs(previous - trial_nll) <= config.solver_tolerance * max(1.0, abs(previous)):
            converged = True
            break
    train_nll, _ = _mixture_nll_gradient(components, dataset_context, weights)
    test_nll = baseline_test_nll = None
    if test_context is not None:
        test_engine = ResponseEngine(
            test_context.dataset, lag=config.impact_lag, knot_count=config.knot_count,
            cache_bytes=config.cache_bytes,
        )
        test_components = []
        for matrix, fit in zip(matrices, fits, strict=True):
            test_matrix = test_engine.model_matrix(
                test_context, matrix.support, forced_closure=matrix.closure
            )
            eta = test_engine.linear_predictor(test_context, test_matrix, fit.coefficients)
            test_components.append(np.exp(np.clip(eta, -745.0, 700.0)))
        test_nll, _ = _mixture_nll_gradient(np.stack(test_components), test_context, weights)
        baseline_matrix = engine.model_matrix(dataset_context, Support(()))
        baseline_fit = fit_model_matrix(
            baseline_matrix, likelihood=dataset_context.dataset.likelihood,
            tolerance=config.solver_tolerance, max_iter=config.solver_max_iter,
        )
        baseline_test_matrix = test_engine.model_matrix(test_context, Support(()))
        baseline_eta = test_engine.linear_predictor(
            test_context, baseline_test_matrix, baseline_fit.coefficients
        )
        baseline_intensity = np.exp(np.clip(baseline_eta, -745.0, 700.0))[None, :]
        baseline_test_nll, _ = _mixture_nll_gradient(
            baseline_intensity, test_context, np.ones(1)
        )
    retained_supports = tuple(matrix.support for matrix in matrices)
    return EnsembleResult(
        retained_supports, tuple(fits), weights, train_nll,
        test_nll, baseline_test_nll, converged,
    )

