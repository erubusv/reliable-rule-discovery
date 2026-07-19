from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .likelihood import conjugate_sum, loss_rows
from .response import ModelMatrix


@dataclass(frozen=True)
class DualCertificate:
    feasible: bool
    nll_lower_bound: float
    equality_residual: float
    inequality_residual: float
    domain_margin: float
    iterations: int


def _clip_domain(
    dual: np.ndarray, matrix: ModelMatrix, likelihood: str, margin: float
) -> np.ndarray:
    output = dual.copy()
    if likelihood == "poisson":
        output = np.maximum(output, -matrix.event_weight + margin)
        return output
    if likelihood != "first_event_cloglog":
        raise ValueError(likelihood)
    output = np.maximum(output, -matrix.event_weight + margin)
    finite_upper = matrix.noevent_weight == 0
    output[finite_upper] = np.minimum(output[finite_upper], -margin)
    return output


def dual_certificate(
    matrix: ModelMatrix,
    *,
    likelihood: str,
    beta: np.ndarray,
    tolerance: float = 1.0e-10,
    max_iter: int = 500,
) -> DualCertificate:
    """Construct a numerically verified Fenchel-dual feasible point.

    Projection failure is not an algorithmic rejection: callers must fail open
    to an exact primal fit.  A certificate is used only after all affine,
    inequality and conjugate-domain residuals have been rechecked.
    """
    eta = matrix.x @ np.asarray(beta, dtype=np.float64)
    _, first, _ = loss_rows(
        eta,
        likelihood=likelihood,
        exposure_weight=matrix.exposure_weight,
        noevent_weight=matrix.noevent_weight,
        event_weight=matrix.event_weight,
    )
    dual = np.asarray(first, dtype=np.float64)
    free = matrix.x[:, : matrix.free_dimension]
    cone = matrix.x[:, matrix.free_dimension :]
    margin = 0.0
    dual = _clip_domain(dual, matrix, likelihood, margin)
    if free.shape[1]:
        left, singular, _ = np.linalg.svd(free, full_matrices=False)
        cutoff = max(free.shape) * np.finfo(float).eps * max(1.0, float(singular[0]))
        free_basis = left[:, singular > cutoff]
    else:
        free_basis = np.zeros((len(dual), 0), dtype=np.float64)
    for iteration in range(1, int(max_iter) + 1):
        if free_basis.shape[1]:
            dual -= free_basis @ (free_basis.T @ dual)
        if cone.shape[1]:
            dots = cone.T @ dual
            for index in np.flatnonzero(dots < 0.0):
                column = cone[:, index]
                direction = column
                if free_basis.shape[1]:
                    direction = column - free_basis @ (free_basis.T @ column)
                norm = float(direction @ direction)
                if norm > 0:
                    dual += (-float(column @ dual) / norm) * direction
        dual = _clip_domain(dual, matrix, likelihood, margin)
        equality = float(np.max(np.abs(free.T @ dual))) if free.shape[1] else 0.0
        inequality = float(np.min(cone.T @ dual)) if cone.shape[1] else math.inf
        if equality <= tolerance and inequality >= -tolerance:
            value = conjugate_sum(
                dual,
                likelihood=likelihood,
                exposure_weight=matrix.exposure_weight,
                noevent_weight=matrix.noevent_weight,
                event_weight=matrix.event_weight,
            )
            domain_margin = float(np.min(dual + matrix.event_weight))
            if likelihood == "first_event_cloglog" and np.any(matrix.noevent_weight == 0):
                domain_margin = min(
                    domain_margin,
                    float(np.min(-dual[matrix.noevent_weight == 0])),
                )
            feasible = bool(math.isfinite(value) and domain_margin >= -tolerance)
            return DualCertificate(
                feasible, -value if feasible else -math.inf,
                equality, inequality, domain_margin, iteration,
            )
    equality = float(np.max(np.abs(free.T @ dual))) if free.shape[1] else 0.0
    inequality = float(np.min(cone.T @ dual)) if cone.shape[1] else math.inf
    return DualCertificate(False, -math.inf, equality, inequality, -math.inf, int(max_iter))
