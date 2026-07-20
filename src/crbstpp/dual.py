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


def design_dual_geometry(x: np.ndarray, free_dimension: int) -> DualGeometry:
    """Precompute the exact small Gram geometry for free dual constraints."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not 0 <= int(free_dimension) <= x.shape[1]:
        raise ValueError("invalid dual design geometry")
    free = x[:, : int(free_dimension)]
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
            int(free_dimension),
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
    return DualGeometry(int(free_dimension), columns, selected_scale, inverse)


def dual_geometry(matrix: ModelMatrix) -> DualGeometry:
    """Precompute the tiny projection geometry shared by sign identities."""
    return design_dual_geometry(matrix.x, matrix.free_dimension)


def _clip_domain(
    dual: np.ndarray,
    *,
    event_weight: np.ndarray,
    noevent_weight: np.ndarray,
    likelihood: str,
    margin: float,
) -> np.ndarray:
    output = dual.copy()
    if likelihood == "poisson":
        output = np.maximum(output, -event_weight + margin)
        return output
    if likelihood != "first_event_cloglog":
        raise ValueError(likelihood)
    output = np.maximum(output, -event_weight + margin)
    finite_upper = noevent_weight == 0
    output[finite_upper] = np.minimum(output[finite_upper], -margin)
    return output


def offset_dual_certificate(
    x: np.ndarray,
    offset: np.ndarray | float,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
    *,
    likelihood: str,
    beta: np.ndarray,
    free_dimension: int,
    tolerance: float = 1.0e-10,
    max_iter: int = 500,
    geometry: DualGeometry | None = None,
) -> DualCertificate:
    """Certify a lower bound for a convex likelihood with a fixed offset.

    The primal is ``min f(offset + X beta)`` with the first
    ``free_dimension`` coefficients unrestricted and all remaining coefficients
    nonnegative.  Its dual objective is ``u.T @ offset - f*(u)`` under
    ``X_free.T @ u = 0`` and ``X_cone.T @ u >= 0``.  Every returned bound is
    rechecked against those constraints and the conjugate domain.

    Projection failure is not an algorithmic rejection: callers must fail open
    to an exact primal fit.  A certificate is used only after all affine,
    inequality and conjugate-domain residuals have been rechecked.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    zero_offset = bool(offset.ndim == 0 and float(offset) == 0.0)
    exposure_weight = np.asarray(exposure_weight, dtype=np.float64)
    noevent_weight = np.asarray(noevent_weight, dtype=np.float64)
    event_weight = np.asarray(event_weight, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    rows, dimension = x.shape
    if (
        (not zero_offset and offset.shape != (rows,))
        or exposure_weight.shape != (rows,)
        or noevent_weight.shape != (rows,)
        or event_weight.shape != (rows,)
        or beta.shape != (dimension,)
        or not 0 <= int(free_dimension) <= dimension
    ):
        raise ValueError("offset dual certificate shape mismatch")
    eta = x @ beta if zero_offset else offset + x @ beta
    _, first, _ = loss_rows(
        eta,
        likelihood=likelihood,
        exposure_weight=exposure_weight,
        noevent_weight=noevent_weight,
        event_weight=event_weight,
    )
    dual = np.asarray(first, dtype=np.float64)
    free = x[:, : int(free_dimension)]
    cone = x[:, int(free_dimension) :]
    margin = 0.0
    dual = _clip_domain(
        dual,
        event_weight=event_weight,
        noevent_weight=noevent_weight,
        likelihood=likelihood,
        margin=margin,
    )

    def make_certificate(
        vector: np.ndarray,
        equality: float,
        inequality: float,
        iterations: int,
    ) -> DualCertificate:
        value = conjugate_sum(
            vector,
            likelihood=likelihood,
            exposure_weight=exposure_weight,
            noevent_weight=noevent_weight,
            event_weight=event_weight,
        )
        domain_margin = float(np.min(vector + event_weight))
        if likelihood == "first_event_cloglog" and np.any(noevent_weight == 0):
            domain_margin = min(
                domain_margin,
                float(np.min(-vector[noevent_weight == 0])),
            )
        lower_bound = -value if zero_offset else float(vector @ offset) - value
        feasible = bool(
            math.isfinite(lower_bound)
            and equality <= tolerance
            and inequality >= -tolerance
            and domain_margin >= -tolerance
        )
        return DualCertificate(
            feasible,
            lower_bound if feasible else -math.inf,
            equality,
            inequality,
            domain_margin,
            iterations,
        )

    initial_equality = float(np.max(np.abs(free.T @ dual))) if free.shape[1] else 0.0
    initial_inequality = float(np.min(cone.T @ dual)) if cone.shape[1] else math.inf
    if initial_equality <= tolerance and initial_inequality >= -tolerance:
        return make_certificate(dual, initial_equality, initial_inequality, 0)
    # Projection onto the free-column orthogonal complement used to build an
    # N x rank left-singular-vector matrix for every proposal.  The same exact
    # projection can be represented by the tiny, column-scaled Gram matrix.
    # This changes neither the dual feasible set nor the certified bound and
    # removes the dominant candidate-wise allocation on large event grids.
    projection = (
        design_dual_geometry(x, int(free_dimension)) if geometry is None else geometry
    )
    if projection.free_dimension != int(free_dimension):
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
        dual = _clip_domain(
            dual,
            event_weight=event_weight,
            noevent_weight=noevent_weight,
            likelihood=likelihood,
            margin=margin,
        )
        equality = float(np.max(np.abs(free.T @ dual))) if free.shape[1] else 0.0
        inequality = float(np.min(cone.T @ dual)) if cone.shape[1] else math.inf
        if equality <= tolerance and inequality >= -tolerance:
            return make_certificate(dual, equality, inequality, iteration)
    equality = float(np.max(np.abs(free.T @ dual))) if free.shape[1] else 0.0
    inequality = float(np.min(cone.T @ dual)) if cone.shape[1] else math.inf
    return DualCertificate(
        False, -math.inf, equality, inequality, -math.inf, int(max_iter)
    )


def dual_certificate(
    matrix: ModelMatrix,
    *,
    likelihood: str,
    beta: np.ndarray,
    tolerance: float = 1.0e-10,
    max_iter: int = 500,
    geometry: DualGeometry | None = None,
) -> DualCertificate:
    """Construct a verified Fenchel certificate for a full model matrix."""
    return offset_dual_certificate(
        matrix.x,
        np.asarray(0.0),
        matrix.exposure_weight,
        matrix.noevent_weight,
        matrix.event_weight,
        likelihood=likelihood,
        beta=beta,
        free_dimension=matrix.free_dimension,
        tolerance=tolerance,
        max_iter=max_iter,
        geometry=geometry,
    )
