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


@dataclass(frozen=True)
class DualGeometry:
    free_dimension: int
    nonzero_columns: np.ndarray
    column_scale: np.ndarray
    gram_inverse: np.ndarray


def dual_geometry(matrix: ModelMatrix) -> DualGeometry:
    """Precompute the tiny projection geometry shared by sign identities."""
    free = matrix.x[:, : matrix.free_dimension]
    if not free.shape[1]:
        return DualGeometry(
            0,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
        )
    scale = np.linalg.norm(free, axis=0)
    columns = np.flatnonzero(scale > np.finfo(float).tiny)
    selected_scale = scale[columns]
    if not len(columns):
        return DualGeometry(
            matrix.free_dimension,
            columns,
            selected_scale,
            np.zeros((0, 0), dtype=np.float64),
        )
    selected = free[:, columns]
    gram = (selected.T @ selected) / (selected_scale[:, None] * selected_scale[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    cutoff = (
        max(free.shape)
        * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(eigenvalues))))
    )
    keep = eigenvalues > cutoff
    inverse = (
        (eigenvectors[:, keep] / eigenvalues[keep]) @ eigenvectors[:, keep].T
        if np.any(keep)
        else np.zeros_like(gram)
    )
    return DualGeometry(matrix.free_dimension, columns, selected_scale, inverse)


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
    geometry: DualGeometry | None = None,
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
    initial_equality = float(np.max(np.abs(free.T @ dual))) if free.shape[1] else 0.0
    initial_inequality = float(np.min(cone.T @ dual)) if cone.shape[1] else math.inf
    if initial_equality <= tolerance and initial_inequality >= -tolerance:
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
            feasible,
            -value if feasible else -math.inf,
            initial_equality,
            initial_inequality,
            domain_margin,
            0,
        )
    # Projection onto the free-column orthogonal complement used to build an
    # N x rank left-singular-vector matrix for every proposal.  The same exact
    # projection can be represented by the tiny, column-scaled Gram matrix.
    # This changes neither the dual feasible set nor the certified bound and
    # removes the dominant candidate-wise allocation on large event grids.
    projection = dual_geometry(matrix) if geometry is None else geometry
    if projection.free_dimension != matrix.free_dimension:
        raise ValueError("dual projection geometry does not match model matrix")
    selected_free = free[:, projection.nonzero_columns]

    def remove_free_projection(vector: np.ndarray) -> np.ndarray:
        if not len(projection.nonzero_columns):
            return vector
        normalized_dot = (selected_free.T @ vector) / projection.column_scale
        coefficient = projection.gram_inverse @ normalized_dot
        return vector - selected_free @ (coefficient / projection.column_scale)

    for iteration in range(1, int(max_iter) + 1):
        dual = remove_free_projection(dual)
        if cone.shape[1]:
            dots = cone.T @ dual
            for index in np.flatnonzero(dots < 0.0):
                column = cone[:, index]
                direction = remove_free_projection(column)
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
            if likelihood == "first_event_cloglog" and np.any(
                matrix.noevent_weight == 0
            ):
                domain_margin = min(
                    domain_margin,
                    float(np.min(-dual[matrix.noevent_weight == 0])),
                )
            feasible = bool(math.isfinite(value) and domain_margin >= -tolerance)
            return DualCertificate(
                feasible,
                -value if feasible else -math.inf,
                equality,
                inequality,
                domain_margin,
                iteration,
            )
    equality = float(np.max(np.abs(free.T @ dual))) if free.shape[1] else 0.0
    inequality = float(np.min(cone.T @ dual)) if cone.shape[1] else math.inf
    return DualCertificate(
        False, -math.inf, equality, inequality, -math.inf, int(max_iter)
    )
