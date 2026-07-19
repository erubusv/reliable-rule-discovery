from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

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


def _objective(
    matrix: ModelMatrix, likelihood: str, beta: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    eta = matrix.x @ beta
    rows, first, second = loss_rows(
        eta,
        likelihood=likelihood,
        exposure_weight=matrix.exposure_weight,
        noevent_weight=matrix.noevent_weight,
        event_weight=matrix.event_weight,
    )
    gradient, hessian = moments(matrix.x, first, second, device="cpu")
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
    """Detect every coordinate recession ray before Newton is attempted."""
    for index in range(matrix.dimension):
        signs = (-1.0, 1.0) if index < matrix.free_dimension else (1.0,)
        for sign in signs:
            direction = sign * matrix.x[:, index]
            if not np.any(direction):
                continue
            event = matrix.event_weight > 0
            if likelihood == "poisson":
                valid = np.all(direction <= 1.0e-14) and np.all(
                    np.abs(direction[event]) <= 1.0e-14
                )
                strict = np.any(direction < -1.0e-14)
            else:
                noevent = matrix.noevent_weight > 0
                valid = np.all(direction[noevent] <= 1.0e-14) and np.all(
                    direction[event] >= -1.0e-14
                )
                strict = np.any(direction[noevent] < -1.0e-14) or np.any(
                    direction[event] > 1.0e-14
                )
            if valid and strict:
                return True
    return False


def _checked_rank(hessian: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(hessian))


def fit_model_matrix(
    matrix: ModelMatrix,
    *,
    likelihood: str,
    tolerance: float,
    max_iter: int,
    warm_start: np.ndarray | None = None,
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
        nll, gradient, hessian, _ = _objective(matrix, likelihood, beta)
        if (
            not math.isfinite(nll)
            or not np.all(np.isfinite(gradient))
            or not np.all(np.isfinite(hessian))
        ):
            return FitResult(
                beta,
                nll,
                False,
                iteration,
                math.inf,
                rank,
                False,
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
            return FitResult(
                beta, nll, False, iteration, kkt, rank, False, "line search failed"
            )
        if abs(previous - trial_nll) <= tolerance * max(1.0, abs(previous)):
            final_nll, final_gradient, final_hessian, _ = _objective(
                matrix, likelihood, beta
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
    nll, gradient, hessian, _ = _objective(matrix, likelihood, beta)
    return FitResult(
        beta,
        nll,
        False,
        int(max_iter),
        projected_kkt(beta, gradient, matrix.free_dimension),
        _checked_rank(hessian),
        False,
        "maximum iterations reached",
    )


def fit_model_matrices(
    matrices: list[ModelMatrix] | tuple[ModelMatrix, ...],
    *,
    likelihood: str,
    tolerance: float,
    max_iter: int,
    workers: int,
    cpu_threads_per_worker: int,
    warm_starts: list[np.ndarray | None] | tuple[np.ndarray | None, ...] | None = None,
) -> list[FitResult]:
    """Deterministic concurrent wrapper around the exact fixed-model solver."""
    ordered = list(matrices)
    starts = [None] * len(ordered) if warm_starts is None else list(warm_starts)
    if len(starts) != len(ordered):
        raise ValueError("warm-start batch length mismatch")

    def solve(item: tuple[ModelMatrix, np.ndarray | None]) -> FitResult:
        configure_cpu_threads(max(1, int(cpu_threads_per_worker)))
        matrix, warm = item
        return fit_model_matrix(
            matrix,
            likelihood=likelihood,
            tolerance=tolerance,
            max_iter=max_iter,
            warm_start=warm,
        )

    jobs = list(zip(ordered, starts, strict=True))
    if len(jobs) <= 1 or workers <= 1:
        return [solve(job) for job in jobs]
    with ThreadPoolExecutor(
        max_workers=min(int(workers), len(jobs)),
        thread_name_prefix="crbstpp-newton",
    ) as executor:
        return list(executor.map(solve, jobs))
