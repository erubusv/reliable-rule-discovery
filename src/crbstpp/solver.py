from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linprog

from .likelihood import loss_rows
from .native import configure_cpu_threads, moments
from .response import ModelMatrix


@dataclass(frozen=True)
class FitResult:
    coefficients: np.ndarray
    nll: float
    converged: bool
    iterations: int
    projected_kkt: float
    rank: int
    recession: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": self.coefficients.tolist(),
            "nll": self.nll if math.isfinite(self.nll) else None,
            "converged": self.converged,
            "iterations": self.iterations,
            "projected_kkt": self.projected_kkt
            if math.isfinite(self.projected_kkt)
            else None,
            "rank": self.rank,
            "recession": self.recession,
            "message": self.message,
        }


@dataclass(frozen=True)
class OneStepState:
    """Feasible Newton point plus derivatives reused by its dual check."""

    fit: FitResult
    dual_vector: np.ndarray
    gradient: np.ndarray


def _objective(
    matrix: ModelMatrix,
    likelihood: str,
    beta: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    eta = matrix.x @ beta
    rows, first, second = loss_rows(
        eta,
        likelihood=likelihood,
        exposure_weight=matrix.exposure_weight,
        noevent_weight=matrix.noevent_weight,
        event_weight=matrix.event_weight,
    )
    gradient, hessian = moments(matrix.x, first, second, device=device)
    return float(np.sum(rows)), gradient, hessian, eta


def _value(matrix: ModelMatrix, likelihood: str, beta: np.ndarray) -> float:
    """Objective-only path for line search; derivatives are not consumed."""
    eta = matrix.x @ beta
    rows, _, _ = loss_rows(
        eta,
        likelihood=likelihood,
        exposure_weight=matrix.exposure_weight,
        noevent_weight=matrix.noevent_weight,
        event_weight=matrix.event_weight,
    )
    return float(np.sum(rows))


def projected_kkt(beta: np.ndarray, gradient: np.ndarray, free_dimension: int) -> float:
    residual = gradient.copy()
    constrained = np.arange(len(beta)) >= int(free_dimension)
    at_boundary = constrained & (beta <= 1.0e-12)
    residual[at_boundary] = np.minimum(residual[at_boundary], 0.0)
    return float(np.max(np.abs(residual))) if len(residual) else 0.0


def _axis_recession(matrix: ModelMatrix, likelihood: str) -> bool:
    """Detect every coordinate recession ray in one tiled matrix pass."""
    x = matrix.x
    dimension = matrix.dimension
    tolerance = 1.0e-14
    minimum = np.full(dimension, np.inf, dtype=np.float64)
    maximum = np.full(dimension, -np.inf, dtype=np.float64)
    event_minimum = np.full(dimension, np.inf, dtype=np.float64)
    event_maximum = np.full(dimension, -np.inf, dtype=np.float64)
    noevent_minimum = np.full(dimension, np.inf, dtype=np.float64)
    noevent_maximum = np.full(dimension, -np.inf, dtype=np.float64)
    # Bound boolean-index temporaries while reducing every column together.
    tile_rows = max(1, min(len(x), 64 * 1024**2 // max(8, 8 * dimension)))
    for start in range(0, len(x), tile_rows):
        end = min(len(x), start + tile_rows)
        tile = x[start:end]
        minimum = np.minimum(minimum, np.min(tile, axis=0))
        maximum = np.maximum(maximum, np.max(tile, axis=0))
        event = matrix.event_weight[start:end] > 0
        if np.any(event):
            selected = tile[event]
            event_minimum = np.minimum(event_minimum, np.min(selected, axis=0))
            event_maximum = np.maximum(event_maximum, np.max(selected, axis=0))
        if likelihood != "poisson":
            noevent = matrix.noevent_weight[start:end] > 0
            if np.any(noevent):
                selected = tile[noevent]
                noevent_minimum = np.minimum(
                    noevent_minimum, np.min(selected, axis=0)
                )
                noevent_maximum = np.maximum(
                    noevent_maximum, np.max(selected, axis=0)
                )
    if likelihood == "poisson":
        event_absolute = np.maximum(
            np.abs(event_minimum), np.abs(event_maximum)
        )
        event_absolute[~np.isfinite(event_absolute)] = 0.0
        positive = (
            (maximum <= tolerance)
            & (event_absolute <= tolerance)
            & (minimum < -tolerance)
        )
        negative = (
            (minimum >= -tolerance)
            & (event_absolute <= tolerance)
            & (maximum > tolerance)
        )
    else:
        positive = (
            (noevent_maximum <= tolerance)
            & (event_minimum >= -tolerance)
            & (
                (noevent_minimum < -tolerance)
                | (event_maximum > tolerance)
            )
        )
        negative = (
            (noevent_minimum >= -tolerance)
            & (event_maximum <= tolerance)
            & (
                (noevent_maximum > tolerance)
                | (event_minimum < -tolerance)
            )
        )
    negative[matrix.free_dimension :] = False
    return bool(np.any(positive | negative))


def _general_recession_design(
    x: np.ndarray,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    free_dimension: int,
    likelihood: str,
) -> bool:
    """Detect an arbitrary feasible recession ray by a bounded cone LP.

    Coordinate checks miss combinations such as ``d=(1,-1)``.  Intersecting
    the recession cone with [-1,1] for free coefficients and [0,1] for
    nonnegative coefficients is lossless: every nonzero ray has a scaled point
    in that box.  The linear objective is strictly negative exactly when at
    least one likelihood term improves along the ray.
    """
    x = np.asarray(x, dtype=np.float64)
    dimension = x.shape[1]
    event = event_weight > 0.0
    noevent = noevent_weight > 0.0
    if likelihood == "poisson":
        constraints = [x]
        if np.any(event):
            # Event rows must have Xd=0; Xd<=0 is already included above.
            constraints.append(-x[event])
        improving = ~event & (exposure_weight > 0.0)
        if not np.any(improving):
            return False
        objective = x[improving].T @ exposure_weight[improving]
    else:
        constraints = []
        if np.any(noevent):
            constraints.append(x[noevent])
        if np.any(event):
            constraints.append(-x[event])
        noevent_only = noevent & ~event
        event_only = event & ~noevent
        if not np.any(noevent_only) and not np.any(event_only):
            return False
        objective = np.zeros(dimension, dtype=np.float64)
        if np.any(noevent_only):
            objective += x[noevent_only].T @ noevent_weight[noevent_only]
        if np.any(event_only):
            objective -= x[event_only].T @ event_weight[event_only]
    if not constraints or not np.any(objective):
        return False
    bounds = [(-1.0, 1.0)] * free_dimension + [
        (0.0, 1.0)
    ] * (dimension - free_dimension)
    result = linprog(
        np.ascontiguousarray(objective),
        A_ub=np.ascontiguousarray(np.vstack(constraints)),
        b_ub=np.zeros(sum(len(value) for value in constraints), dtype=np.float64),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success or result.fun is None or not math.isfinite(result.fun):
        # Failure to construct a certificate must not label a finite model as
        # recessionary.  The caller retains its original non-convergence.
        return False
    scale = max(1.0, float(np.linalg.norm(objective, ord=1)))
    return bool(result.fun < -1.0e-10 * scale)


def _general_recession(matrix: ModelMatrix, likelihood: str) -> bool:
    return _general_recession_design(
        matrix.x,
        matrix.exposure_weight,
        matrix.noevent_weight,
        matrix.event_weight,
        matrix.free_dimension,
        likelihood,
    )


def _failed_fit(
    matrix: ModelMatrix,
    likelihood: str,
    beta: np.ndarray,
    nll: float,
    iteration: int,
    kkt: float,
    rank: int,
    message: str,
) -> FitResult:
    recession = _general_recession(matrix, likelihood)
    return FitResult(
        beta,
        math.inf if recession else nll,
        False,
        iteration,
        math.inf if recession else kkt,
        0 if recession else rank,
        recession,
        "nonattained combined recession direction" if recession else message,
    )


def _checked_rank(hessian: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(hessian))


def one_step_model_matrix(
    matrix: ModelMatrix,
    *,
    likelihood: str,
    warm_start: np.ndarray,
    tolerance: float,
    device: str = "cpu",
) -> OneStepState:
    """Take exactly one feasible projected-Newton step on a full support.

    This routine is deliberately *not* an optimizer.  It supplies a cheap,
    feasible primal point for conditional add/drop proposals.  Callers attach a
    Fenchel certificate to the returned point and keep it outside the exact-fit
    cache.  Consequently a one-step proposal can never be mistaken for a
    converged fixed-support estimator.

    The step is the same active-set Newton direction and monotone Armijo search
    used by :func:`fit_model_matrix`.  If the warm point already satisfies KKT,
    it is returned unchanged, but ``converged`` remains false by construction;
    only the full solver may create an exact cached result.
    """
    beta = np.asarray(warm_start, dtype=np.float64).copy()
    if beta.shape != (matrix.dimension,) or not np.all(np.isfinite(beta)):
        raise ValueError("one-step warm start does not match the model matrix")
    if matrix.dimension > matrix.free_dimension:
        beta[matrix.free_dimension :] = np.maximum(
            beta[matrix.free_dimension :], 0.0
        )
    nll, gradient, hessian, _ = _objective(
        matrix, likelihood, beta, device=device
    )
    if not (
        math.isfinite(nll)
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(hessian))
    ):
        return OneStepState(
            FitResult(
                beta,
                math.inf,
                False,
                0,
                math.inf,
                0,
                False,
                "nonfinite conditional one-step derivatives",
            ),
            np.zeros(len(matrix.x), dtype=np.float64),
            np.full(matrix.dimension, math.inf, dtype=np.float64),
        )
    initial_kkt = projected_kkt(beta, gradient, matrix.free_dimension)
    if initial_kkt <= tolerance:
        eta = matrix.x @ beta
        _, dual_vector, _ = loss_rows(
            eta,
            likelihood=likelihood,
            exposure_weight=matrix.exposure_weight,
            noevent_weight=matrix.noevent_weight,
            event_weight=matrix.event_weight,
        )
        return OneStepState(
            FitResult(
                beta,
                nll,
                False,
                0,
                initial_kkt,
                _checked_rank(hessian),
                False,
                "conditional one-step warm point already satisfies KKT",
            ),
            np.asarray(dual_vector, dtype=np.float64),
            gradient,
        )

    active = np.arange(matrix.dimension) < matrix.free_dimension
    active |= (beta > 1.0e-12) | (gradient < 0.0)
    indices = np.flatnonzero(active)
    if not len(indices):
        eta = matrix.x @ beta
        _, dual_vector, _ = loss_rows(
            eta,
            likelihood=likelihood,
            exposure_weight=matrix.exposure_weight,
            noevent_weight=matrix.noevent_weight,
            event_weight=matrix.event_weight,
        )
        return OneStepState(
            FitResult(
                beta,
                nll,
                False,
                0,
                initial_kkt,
                0,
                False,
                "empty conditional one-step active set",
            ),
            np.asarray(dual_vector, dtype=np.float64),
            gradient,
        )
    sub_hessian = hessian[np.ix_(indices, indices)]
    try:
        active_direction = np.linalg.solve(sub_hessian, -gradient[indices])
    except np.linalg.LinAlgError:
        active_direction = np.linalg.lstsq(
            sub_hessian, -gradient[indices], rcond=None
        )[0]
    direction = np.zeros(matrix.dimension, dtype=np.float64)
    direction[indices] = active_direction
    if float(gradient @ direction) >= 0.0:
        direction = -gradient
        direction[matrix.free_dimension :] = np.where(
            (beta[matrix.free_dimension :] <= 1.0e-12)
            & (direction[matrix.free_dimension :] < 0.0),
            0.0,
            direction[matrix.free_dimension :],
        )
    step = 1.0
    accepted = False
    for _ in range(60):
        trial = beta + step * direction
        trial[matrix.free_dimension :] = np.maximum(
            trial[matrix.free_dimension :], 0.0
        )
        trial_nll = _value(matrix, likelihood, trial)
        displacement = trial - beta
        if math.isfinite(trial_nll) and trial_nll <= nll + 1.0e-4 * float(
            gradient @ displacement
        ):
            beta = trial
            accepted = True
            break
        step *= 0.5
    if not accepted:
        eta = matrix.x @ beta
        _, dual_vector, _ = loss_rows(
            eta,
            likelihood=likelihood,
            exposure_weight=matrix.exposure_weight,
            noevent_weight=matrix.noevent_weight,
            event_weight=matrix.event_weight,
        )
        return OneStepState(
            FitResult(
                beta,
                nll,
                False,
                0,
                initial_kkt,
                _checked_rank(hessian),
                False,
                "conditional one-step line search failed",
            ),
            np.asarray(dual_vector, dtype=np.float64),
            gradient,
        )

    final_nll, final_gradient, final_hessian, final_eta = _objective(
        matrix, likelihood, beta, device=device
    )
    _, dual_vector, _ = loss_rows(
        final_eta,
        likelihood=likelihood,
        exposure_weight=matrix.exposure_weight,
        noevent_weight=matrix.noevent_weight,
        event_weight=matrix.event_weight,
    )
    final_kkt = projected_kkt(beta, final_gradient, matrix.free_dimension)
    return OneStepState(
        FitResult(
            beta,
            final_nll,
            False,
            1,
            final_kkt,
            _checked_rank(final_hessian),
            False,
            "conditional one-step feasible point",
        ),
        np.asarray(dual_vector, dtype=np.float64),
        final_gradient,
    )


def fit_model_matrix(
    matrix: ModelMatrix,
    *,
    likelihood: str,
    tolerance: float,
    max_iter: int,
    warm_start: np.ndarray | None = None,
    device: str = "cpu",
) -> FitResult:
    dimension = matrix.dimension
    beta = np.zeros(dimension, dtype=np.float64)
    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=np.float64)
        beta[: min(len(warm), dimension)] = warm[: min(len(warm), dimension)]
    if dimension > matrix.free_dimension:
        beta[matrix.free_dimension :] = np.maximum(beta[matrix.free_dimension :], 0.0)
    # Intercept initialization from the empirical rate prevents needless
    # overflow in the first Newton iteration.
    total_exposure = float(np.sum(matrix.exposure_weight))
    total_events = float(np.sum(matrix.event_weight))
    if warm_start is None and total_exposure > 0:
        beta[0] = math.log(max(total_events, 0.5) / total_exposure)
    recession = _axis_recession(matrix, likelihood)
    if recession:
        return FitResult(
            beta,
            math.inf,
            False,
            0,
            math.inf,
            0,
            True,
            "nonattained recession direction",
        )
    previous = math.inf
    rank = 0
    for iteration in range(1, int(max_iter) + 1):
        nll, gradient, hessian, _ = _objective(
            matrix, likelihood, beta, device=device
        )
        if (
            not math.isfinite(nll)
            or not np.all(np.isfinite(gradient))
            or not np.all(np.isfinite(hessian))
        ):
            return _failed_fit(
                matrix,
                likelihood,
                beta,
                nll,
                iteration,
                math.inf,
                rank,
                "nonfinite objective derivatives",
            )
        kkt = projected_kkt(beta, gradient, matrix.free_dimension)
        if kkt <= tolerance:
            rank = _checked_rank(hessian)
            if rank < dimension:
                return FitResult(
                    beta,
                    nll,
                    False,
                    iteration,
                    kkt,
                    rank,
                    False,
                    "rank-deficient fixed-support information",
                )
            return FitResult(beta, nll, True, iteration, kkt, rank, False, "converged")
        active = np.arange(dimension) < matrix.free_dimension
        active |= (beta > 1.0e-12) | (gradient < 0.0)
        indices = np.flatnonzero(active)
        if not len(indices):
            return FitResult(
                beta, nll, False, iteration, kkt, 0, False, "empty Newton active set"
            )
        sub_hessian = hessian[np.ix_(indices, indices)]
        rank = _checked_rank(sub_hessian)
        try:
            direction_active = np.linalg.solve(sub_hessian, -gradient[indices])
        except np.linalg.LinAlgError:
            direction_active = np.linalg.lstsq(
                sub_hessian, -gradient[indices], rcond=None
            )[0]
        direction = np.zeros(dimension, dtype=np.float64)
        direction[indices] = direction_active
        directional = float(gradient @ direction)
        if directional >= 0:
            direction = -gradient
            direction[matrix.free_dimension :] = np.where(
                (beta[matrix.free_dimension :] <= 1.0e-12)
                & (direction[matrix.free_dimension :] < 0),
                0.0,
                direction[matrix.free_dimension :],
            )
            directional = float(gradient @ direction)
        step = 1.0
        accepted = False
        for _ in range(60):
            trial = beta + step * direction
            trial[matrix.free_dimension :] = np.maximum(
                trial[matrix.free_dimension :], 0.0
            )
            trial_nll = _value(matrix, likelihood, trial)
            displacement = trial - beta
            if math.isfinite(trial_nll) and trial_nll <= nll + 1.0e-4 * float(
                gradient @ displacement
            ):
                beta = trial
                previous = nll
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return _failed_fit(
                matrix,
                likelihood,
                beta,
                nll,
                iteration,
                kkt,
                rank,
                "line search failed",
            )
        if abs(previous - trial_nll) <= tolerance * max(1.0, abs(previous)):
            final_nll, final_gradient, final_hessian, _ = _objective(
                matrix, likelihood, beta, device=device
            )
            final_kkt = projected_kkt(beta, final_gradient, matrix.free_dimension)
            if final_kkt <= 10.0 * tolerance:
                final_rank = _checked_rank(final_hessian)
                if final_rank < dimension:
                    return FitResult(
                        beta,
                        final_nll,
                        False,
                        iteration,
                        final_kkt,
                        final_rank,
                        False,
                        "rank-deficient fixed-support information",
                    )
                return FitResult(
                    beta,
                    final_nll,
                    True,
                    iteration,
                    final_kkt,
                    final_rank,
                    False,
                    "converged by objective and KKT",
                )
    nll, gradient, hessian, _ = _objective(
        matrix, likelihood, beta, device=device
    )
    return _failed_fit(
        matrix,
        likelihood,
        beta,
        nll,
        int(max_iter),
        projected_kkt(beta, gradient, matrix.free_dimension),
        _checked_rank(hessian),
        "maximum iterations reached",
    )


def fit_model_matrix_continued(
    matrix: ModelMatrix,
    *,
    likelihood: str,
    tolerance: float,
    max_iter: int,
    warm_start: np.ndarray | None = None,
    device: str = "cpu",
) -> FitResult:
    """Continue an exact Newton fit across iteration windows until certified.

    ``max_iter`` is a checkpoint interval, not a statistical rejection rule.
    A finite, full-rank convex model must not become uncertifiable merely
    because it needed one more Newton window.  Genuine recession, rank failure,
    line-search failure and nonfinite arithmetic still fail closed.  Repeated
    identical terminal states are reported as numerical stagnation rather than
    looping indefinitely.
    """
    total_iterations = 0
    start = warm_start
    previous: tuple[float, float, bytes] | None = None
    repeated = 0
    while True:
        fit = fit_model_matrix(
            matrix,
            likelihood=likelihood,
            tolerance=tolerance,
            max_iter=max_iter,
            warm_start=start,
            device=device,
        )
        total_iterations += fit.iterations
        if fit.converged or fit.message != "maximum iterations reached":
            return replace(fit, iterations=total_iterations)
        state = (
            float(fit.nll),
            float(fit.projected_kkt),
            np.ascontiguousarray(fit.coefficients).tobytes(),
        )
        if previous == state:
            repeated += 1
            if repeated >= 2:
                return replace(
                    fit,
                    iterations=total_iterations,
                    message="continued Newton stagnation before KKT convergence",
                )
        else:
            repeated = 0
        previous = state
        start = fit.coefficients


def fit_offset_design(
    x: np.ndarray,
    offset: np.ndarray,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    likelihood: str,
    free_dimension: int,
    tolerance: float,
    max_iter: int,
    device: str = "cpu",
    warm_start: np.ndarray | None = None,
) -> FitResult:
    """Exactly optimize a small new block around a frozen linear predictor."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    offset = np.ascontiguousarray(offset, dtype=np.float64)
    exposure_weight = np.ascontiguousarray(exposure_weight, dtype=np.float64)
    noevent_weight = np.ascontiguousarray(noevent_weight, dtype=np.float64)
    event_weight = np.ascontiguousarray(event_weight, dtype=np.float64)
    rows, dimension = x.shape
    if any(
        value.shape != (rows,)
        for value in (offset, exposure_weight, noevent_weight, event_weight)
    ) or not 0 <= free_dimension <= dimension:
        raise ValueError("offset block design shape mismatch")
    if warm_start is None:
        beta = np.zeros(dimension, dtype=np.float64)
    else:
        beta = np.asarray(warm_start, dtype=np.float64).copy()
        if beta.shape != (dimension,) or not np.all(np.isfinite(beta)):
            raise ValueError("invalid offset-block warm start")
        if np.any(beta[free_dimension:] < 0.0):
            raise ValueError("offset-block warm start violates the nonnegative cone")

    def failed(
        nll: float, iteration: int, kkt: float, rank: int, message: str
    ) -> FitResult:
        recession = _general_recession_design(
            x,
            exposure_weight,
            noevent_weight,
            event_weight,
            free_dimension,
            likelihood,
        )
        return FitResult(
            beta,
            math.inf if recession else nll,
            False,
            iteration,
            math.inf if recession else kkt,
            0 if recession else rank,
            recession,
            "nonattained combined offset-block recession direction"
            if recession
            else message,
        )
    # Recession directions depend only on design signs and event/no-event
    # support; the finite frozen offset does not change them.
    for index in range(dimension):
        signs = (-1.0, 1.0) if index < free_dimension else (1.0,)
        for sign in signs:
            direction = sign * x[:, index]
            if not np.any(direction):
                continue
            event = event_weight > 0
            if likelihood == "poisson":
                valid = np.all(direction <= 1.0e-14) and np.all(
                    np.abs(direction[event]) <= 1.0e-14
                )
                strict = np.any(direction < -1.0e-14)
            else:
                noevent = noevent_weight > 0
                valid = np.all(direction[noevent] <= 1.0e-14) and np.all(
                    direction[event] >= -1.0e-14
                )
                strict = np.any(direction[noevent] < -1.0e-14) or np.any(
                    direction[event] > 1.0e-14
                )
            if valid and strict:
                return FitResult(
                    beta,
                    math.inf,
                    False,
                    0,
                    math.inf,
                    0,
                    True,
                    "nonattained offset-block recession direction",
                )

    previous = math.inf
    rank = 0
    for iteration in range(1, int(max_iter) + 1):
        eta = offset + x @ beta
        values, first, second = loss_rows(
            eta,
            likelihood=likelihood,
            exposure_weight=exposure_weight,
            noevent_weight=noevent_weight,
            event_weight=event_weight,
        )
        nll = float(np.sum(values))
        gradient, hessian = moments(x, first, second, device=device)
        if not (
            math.isfinite(nll)
            and np.all(np.isfinite(gradient))
            and np.all(np.isfinite(hessian))
        ):
            return failed(
                nll,
                iteration,
                math.inf,
                rank,
                "nonfinite offset-block derivatives",
            )
        kkt = projected_kkt(beta, gradient, free_dimension)
        if kkt <= tolerance:
            rank = _checked_rank(hessian)
            return FitResult(
                beta,
                nll,
                rank == dimension,
                iteration,
                kkt,
                rank,
                False,
                "converged" if rank == dimension else "rank-deficient offset block",
            )
        active = np.arange(dimension) < free_dimension
        active |= (beta > 1.0e-12) | (gradient < 0.0)
        indices = np.flatnonzero(active)
        sub_hessian = hessian[np.ix_(indices, indices)]
        rank = _checked_rank(sub_hessian)
        try:
            step_active = np.linalg.solve(sub_hessian, -gradient[indices])
        except np.linalg.LinAlgError:
            step_active = np.linalg.lstsq(
                sub_hessian, -gradient[indices], rcond=None
            )[0]
        direction = np.zeros(dimension, dtype=np.float64)
        direction[indices] = step_active
        if float(gradient @ direction) >= 0:
            direction = -gradient
            direction[free_dimension:] = np.where(
                (beta[free_dimension:] <= 1.0e-12)
                & (direction[free_dimension:] < 0),
                0.0,
                direction[free_dimension:],
            )
        accepted = False
        for _ in range(60):
            trial = beta + direction
            trial[free_dimension:] = np.maximum(trial[free_dimension:], 0.0)
            trial_values, _, _ = loss_rows(
                offset + x @ trial,
                likelihood=likelihood,
                exposure_weight=exposure_weight,
                noevent_weight=noevent_weight,
                event_weight=event_weight,
            )
            trial_nll = float(np.sum(trial_values))
            if math.isfinite(trial_nll) and trial_nll <= nll + 1.0e-4 * float(
                gradient @ (trial - beta)
            ):
                beta = trial
                previous = nll
                accepted = True
                break
            direction *= 0.5
        if not accepted:
            return failed(
                nll,
                iteration,
                kkt,
                rank,
                "offset-block line search failed",
            )
        if abs(previous - trial_nll) <= tolerance * max(1.0, abs(previous)):
            # Let the next iteration perform the definitive projected-KKT and
            # rank checks; no approximate early acceptance is used.
            continue
    eta = offset + x @ beta
    values, first, second = loss_rows(
        eta,
        likelihood=likelihood,
        exposure_weight=exposure_weight,
        noevent_weight=noevent_weight,
        event_weight=event_weight,
    )
    gradient, hessian = moments(x, first, second, device=device)
    return failed(
        float(np.sum(values)),
        int(max_iter),
        projected_kkt(beta, gradient, free_dimension),
        _checked_rank(hessian),
        "offset-block maximum iterations reached",
    )


def fit_offset_design_continued(
    x: np.ndarray,
    offset: np.ndarray,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    likelihood: str,
    free_dimension: int,
    tolerance: float,
    max_iter: int,
    device: str = "cpu",
    warm_start: np.ndarray | None = None,
) -> FitResult:
    """Continue an exact restricted-block fit beyond a checkpoint window."""
    total_iterations = 0
    start = warm_start
    previous: tuple[float, float, bytes] | None = None
    repeated = 0
    while True:
        fit = fit_offset_design(
            x,
            offset,
            exposure_weight,
            noevent_weight,
            event_weight,
            likelihood=likelihood,
            free_dimension=free_dimension,
            tolerance=tolerance,
            max_iter=max_iter,
            device=device,
            warm_start=start,
        )
        total_iterations += fit.iterations
        if fit.converged or fit.message != "offset-block maximum iterations reached":
            return replace(fit, iterations=total_iterations)
        state = (
            float(fit.nll),
            float(fit.projected_kkt),
            np.ascontiguousarray(fit.coefficients).tobytes(),
        )
        if previous == state:
            repeated += 1
            if repeated >= 2:
                return replace(
                    fit,
                    iterations=total_iterations,
                    message=(
                        "continued offset-block Newton stagnation before "
                        "KKT convergence"
                    ),
                )
        else:
            repeated = 0
        previous = state
        start = fit.coefficients


def fit_model_matrices(
    matrices: list[ModelMatrix] | tuple[ModelMatrix, ...],
    *,
    likelihood: str,
    tolerance: float,
    max_iter: int,
    workers: int,
    cpu_threads_per_worker: int,
    warm_starts: list[np.ndarray | None] | tuple[np.ndarray | None, ...] | None = None,
    devices: tuple[str, ...] = ("cpu",),
) -> list[FitResult]:
    """Deterministic concurrent wrapper around the exact fixed-model solver."""
    ordered = list(matrices)
    starts = [None] * len(ordered) if warm_starts is None else list(warm_starts)
    if len(starts) != len(ordered):
        raise ValueError("warm-start batch length mismatch")

    if not devices:
        devices = ("cpu",)

    def solve(
        item: tuple[int, ModelMatrix, np.ndarray | None]
    ) -> FitResult:
        configure_cpu_threads(max(1, int(cpu_threads_per_worker)))
        index, matrix, warm = item
        return fit_model_matrix_continued(
            matrix,
            likelihood=likelihood,
            tolerance=tolerance,
            max_iter=max_iter,
            warm_start=warm,
            device=devices[index % len(devices)],
        )

    jobs = [
        (index, matrix, warm)
        for index, (matrix, warm) in enumerate(zip(ordered, starts, strict=True))
    ]
    if len(jobs) <= 1 or workers <= 1:
        return [solve(job) for job in jobs]
    with ThreadPoolExecutor(
        max_workers=min(int(workers), len(jobs)),
        thread_name_prefix="crbstpp-newton",
    ) as executor:
        return list(executor.map(solve, jobs))
