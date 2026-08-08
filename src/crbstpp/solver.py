from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix, vstack as sparse_vstack

from .likelihood import (
    cloglog_event_terms,
    is_poisson_likelihood,
    loss_grid_sparse_event_derivatives,
    loss_rows,
    loss_value_rows,
)
from .native import (
    SparseMomentGeometry,
    configure_cpu_threads,
    design_column_cross,
    fused_likelihood_value_eta_gradient,
    moments,
    new_derivative_token,
    release_cuda_workspaces,
    resident_cloglog_objective,
    resident_eta,
    resident_poisson_objective,
    resident_projected_objective,
    sparse_model_moments,
    sparse_moment_geometry,
    sorted_unique_union,
)
from .response import Context, ModelMatrix, SparseBlock


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


@dataclass(frozen=True)
class _ResidentLikelihoodShard:
    x: np.ndarray
    primary_weight: np.ndarray
    event_weight: np.ndarray
    likelihood: str
    device: str
    token: int


class ProjectedDesignEvaluator:
    """OOM-bounded multi-GPU evaluator shared by projected submodels.

    A large parent design is row-sharded, never copied, across physical CUDA
    devices.  Each device retains at most one shard and returns only a scalar
    and the small source-coordinate gradient/Hessian.  Fixed-order host
    reduction preserves deterministic float64 accumulation.  Unsupported
    configurations transparently use the ordinary source-matrix evaluator.
    """

    _MAX_SHARD_BYTES = 4 * 1024**3

    def __init__(
        self,
        matrix: ModelMatrix,
        *,
        likelihood: str,
        devices: tuple[str, ...],
    ):
        self.matrix = matrix
        self.likelihood = str(likelihood)
        self.devices = tuple(dict.fromkeys(devices))
        self._pool: ThreadPoolExecutor | None = None
        self._shards: tuple[_ResidentLikelihoodShard, ...] = ()
        self._column_cross: dict[int, np.ndarray] = {}
        self._axis_extrema: tuple[np.ndarray, ...] | None = None
        self._projection_key: tuple[bytes, bytes] | None = None
        self._projection_token = -1
        cuda_devices = tuple(
            device for device in self.devices if device.startswith("cuda")
        )
        if (
            not (
                is_poisson_likelihood(self.likelihood)
                or self.likelihood == "first_event_cloglog"
            )
            or not cuda_devices
            or not len(matrix.x)
        ):
            return
        shard_count = min(len(cuda_devices), len(matrix.x))
        row_edges = np.linspace(0, len(matrix.x), shard_count + 1, dtype=np.int64)
        maximum_bytes = max(
            int(matrix.x[int(row_edges[index]) : int(row_edges[index + 1])].nbytes)
            for index in range(shard_count)
        )
        if maximum_bytes > self._MAX_SHARD_BYTES:
            return
        # Pricing kernels and exact-shard kernels have disjoint lifetimes.
        # Releasing the former before upload keeps the peak below the
        # per-device 24-GiB budget; source host slices below are views.
        release_cuda_workspaces(cuda_devices)
        shards = []
        for index in range(shard_count):
            start = int(row_edges[index])
            end = int(row_edges[index + 1])
            shards.append(
                _ResidentLikelihoodShard(
                    x=matrix.x[start:end],
                    primary_weight=(
                        matrix.exposure_weight[start:end]
                        if is_poisson_likelihood(self.likelihood)
                        else matrix.noevent_weight[start:end]
                    ),
                    event_weight=matrix.event_weight[start:end],
                    likelihood=self.likelihood,
                    device=cuda_devices[index],
                    token=new_derivative_token(),
                )
            )
        self._shards = tuple(shards)
        if len(self._shards) > 1:
            self._pool = ThreadPoolExecutor(
                max_workers=len(self._shards),
                thread_name_prefix="crbstpp-projected-shard",
            )

    @property
    def sharded(self) -> bool:
        return bool(self._shards)

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    @property
    def maximum_shard_bytes(self) -> int:
        return max((shard.x.nbytes for shard in self._shards), default=0)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> ProjectedDesignEvaluator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _evaluate_shard(
        task: tuple[_ResidentLikelihoodShard, np.ndarray, bool],
    ) -> tuple[float, np.ndarray | None, np.ndarray | None] | None:
        shard, beta, compute_moments = task
        objective = (
            resident_poisson_objective
            if is_poisson_likelihood(shard.likelihood)
            else resident_cloglog_objective
        )
        result = objective(
            shard.x,
            beta,
            shard.primary_weight,
            shard.event_weight,
            device=shard.device,
            matrix_token=shard.token,
            compute_moments=compute_moments,
            return_eta=False,
        )
        if result is None:
            return None
        nll, _, gradient, hessian = result
        return nll, gradient, hessian

    @staticmethod
    def _evaluate_dual_shard(
        task: tuple[_ResidentLikelihoodShard, np.ndarray],
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        shard, beta = task
        objective = (
            resident_poisson_objective
            if is_poisson_likelihood(shard.likelihood)
            else resident_cloglog_objective
        )
        result = objective(
            shard.x,
            beta,
            shard.primary_weight,
            shard.event_weight,
            device=shard.device,
            matrix_token=shard.token,
            # Native mode 2 computes eta and X'u but skips the X'WX Hessian.
            compute_moments=2,
            return_eta=True,
        )
        if result is None:
            return None
        nll, eta, gradient, _ = result
        if eta is None or gradient is None:
            return None
        return nll, eta, gradient

    def _resident_parts(
        self,
        beta: np.ndarray,
        *,
        compute_moments: bool,
    ) -> list[tuple[float, np.ndarray | None, np.ndarray | None]] | None:
        if not self._shards:
            return None
        tasks = [
            (shard, np.asarray(beta, dtype=np.float64), compute_moments)
            for shard in self._shards
        ]
        if self._pool is None:
            raw = [self._evaluate_shard(tasks[0])]
        else:
            raw = list(self._pool.map(self._evaluate_shard, tasks))
        if any(item is None for item in raw):
            return None
        return [item for item in raw if item is not None]

    @staticmethod
    def _evaluate_projected_shard(
        task: tuple[
            _ResidentLikelihoodShard,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
            bool,
        ],
    ) -> tuple[float, np.ndarray | None, np.ndarray | None] | None:
        shard, beta, columns, scales, projection_token, compute_moments = task
        return resident_projected_objective(
            shard.x,
            beta,
            columns,
            scales,
            shard.primary_weight,
            shard.event_weight,
            likelihood=shard.likelihood,
            device=shard.device,
            matrix_token=shard.token,
            projection_token=projection_token,
            compute_moments=compute_moments,
        )

    def _projected_parts(
        self,
        beta: np.ndarray,
        columns: np.ndarray,
        scales: np.ndarray,
        *,
        compute_moments: bool,
    ) -> list[tuple[float, np.ndarray | None, np.ndarray | None]] | None:
        if not self._shards:
            return None
        columns = np.ascontiguousarray(columns, dtype=np.int64)
        scales = np.ascontiguousarray(scales, dtype=np.float64)
        key = (columns.tobytes(), scales.tobytes())
        if key != self._projection_key:
            self._projection_key = key
            self._projection_token = new_derivative_token()
        tasks = [
            (
                shard,
                np.asarray(beta, dtype=np.float64),
                columns,
                scales,
                self._projection_token,
                compute_moments,
            )
            for shard in self._shards
        ]
        if self._pool is None:
            raw = [self._evaluate_projected_shard(tasks[0])]
        else:
            raw = list(self._pool.map(self._evaluate_projected_shard, tasks))
        if any(item is None for item in raw):
            return None
        return [item for item in raw if item is not None]

    def projected_objective(
        self, beta: np.ndarray, columns: np.ndarray, scales: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        parts = self._projected_parts(
            beta, columns, scales, compute_moments=True
        )
        if parts is None:
            return None
        nll = float(math.fsum(item[0] for item in parts))
        gradient = np.zeros(len(columns), dtype=np.float64)
        hessian = np.zeros((len(columns), len(columns)), dtype=np.float64)
        for _, part_gradient, part_hessian in parts:
            if part_gradient is None or part_hessian is None:
                return None
            gradient += part_gradient
            hessian += part_hessian
        return nll, gradient, hessian

    def projected_value(
        self, beta: np.ndarray, columns: np.ndarray, scales: np.ndarray
    ) -> float | None:
        parts = self._projected_parts(
            beta, columns, scales, compute_moments=False
        )
        if parts is None:
            return None
        return float(math.fsum(item[0] for item in parts))

    def objective(self, beta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        parts = self._resident_parts(beta, compute_moments=True)
        if parts is None:
            nll, gradient, hessian, _ = _objective(
                self.matrix, self.likelihood, beta, device="cpu"
            )
            return nll, gradient, hessian
        nll = float(math.fsum(item[0] for item in parts))
        gradient = np.zeros(self.matrix.dimension, dtype=np.float64)
        hessian = np.zeros(
            (self.matrix.dimension, self.matrix.dimension), dtype=np.float64
        )
        for _, part_gradient, part_hessian in parts:
            if part_gradient is None or part_hessian is None:
                raise AssertionError("resident projected shard lost its moments")
            gradient += part_gradient
            hessian += part_hessian
        return nll, gradient, hessian

    def value(self, beta: np.ndarray) -> float:
        parts = self._resident_parts(beta, compute_moments=False)
        if parts is None:
            return _value(self.matrix, self.likelihood, beta, device="cpu")
        return float(math.fsum(item[0] for item in parts))

    def dual_state(
        self, beta: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        """Return exact eta and source gradient from the resident row shards."""
        primary = (
            self.matrix.exposure_weight
            if is_poisson_likelihood(self.likelihood)
            else self.matrix.noevent_weight
        )
        fused = fused_likelihood_value_eta_gradient(
            self.matrix.x,
            np.asarray(beta, dtype=np.float64),
            primary,
            self.matrix.event_weight,
            likelihood=self.likelihood,
        )
        if fused is not None:
            return fused
        if not self._shards:
            return None
        tasks = [(shard, np.asarray(beta, dtype=np.float64)) for shard in self._shards]
        raw = (
            [self._evaluate_dual_shard(tasks[0])]
            if self._pool is None
            else list(self._pool.map(self._evaluate_dual_shard, tasks))
        )
        if any(item is None for item in raw):
            return None
        parts = [item for item in raw if item is not None]
        eta = np.concatenate([item[1] for item in parts])
        gradient = np.zeros(self.matrix.dimension, dtype=np.float64)
        for _, _, part_gradient in parts:
            gradient += part_gradient
        return float(math.fsum(item[0] for item in parts)), eta, gradient

    def source_column_cross(self, column: int) -> np.ndarray | None:
        """Return and cache one source X'X column without forming the Gram."""
        column = int(column)
        cached = self._column_cross.get(column)
        if cached is None:
            cached = design_column_cross(self.matrix.x, column)
            if cached is None:
                return None
            self._column_cross[column] = cached
        return cached

    def projected_axis_recession(
        self,
        columns: np.ndarray,
        scales: np.ndarray,
        free_dimension: int,
    ) -> bool:
        """Check projected coordinate rays from one cached source scan.

        Representation audit fits many signed projections of the same source
        design.  Scanning every row again for every projection used to cost as
        much as the Newton solve itself.  The source-column extrema below are
        sufficient for the exact coordinate-ray test and are computed once
        per shared design.  Signed projections are handled algebraically, so
        this is identical to :func:`_projected_axis_recession`.
        """

        if self._axis_extrema is None:
            dimension = self.matrix.dimension
            minimum = np.full(dimension, np.inf, dtype=np.float64)
            maximum = np.full(dimension, -np.inf, dtype=np.float64)
            event_minimum = np.full(dimension, np.inf, dtype=np.float64)
            event_maximum = np.full(dimension, -np.inf, dtype=np.float64)
            noevent_minimum = np.full(dimension, np.inf, dtype=np.float64)
            noevent_maximum = np.full(dimension, -np.inf, dtype=np.float64)
            tile_rows = max(
                1,
                min(
                    len(self.matrix.x),
                    64 * 1024**2 // max(8, 8 * dimension),
                ),
            )
            for start in range(0, len(self.matrix.x), tile_rows):
                end = min(len(self.matrix.x), start + tile_rows)
                tile = self.matrix.x[start:end]
                minimum = np.minimum(minimum, np.min(tile, axis=0))
                maximum = np.maximum(maximum, np.max(tile, axis=0))
                event = self.matrix.event_weight[start:end] > 0
                if np.any(event):
                    selected = tile[event]
                    event_minimum = np.minimum(event_minimum, np.min(selected, axis=0))
                    event_maximum = np.maximum(event_maximum, np.max(selected, axis=0))
                if not is_poisson_likelihood(self.likelihood):
                    noevent = self.matrix.noevent_weight[start:end] > 0
                    if np.any(noevent):
                        selected = tile[noevent]
                        noevent_minimum = np.minimum(
                            noevent_minimum, np.min(selected, axis=0)
                        )
                        noevent_maximum = np.maximum(
                            noevent_maximum, np.max(selected, axis=0)
                        )
            self._axis_extrema = (
                minimum,
                maximum,
                event_minimum,
                event_maximum,
                noevent_minimum,
                noevent_maximum,
            )

        (
            source_minimum,
            source_maximum,
            source_event_minimum,
            source_event_maximum,
            source_noevent_minimum,
            source_noevent_maximum,
        ) = self._axis_extrema
        columns = np.asarray(columns, dtype=np.int64)
        scales = np.asarray(scales, dtype=np.float64)

        def signed_extrema(
            lower: np.ndarray, upper: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            selected_lower = lower[columns]
            selected_upper = upper[columns]
            positive = scales > 0.0
            return (
                np.where(positive, scales * selected_lower, scales * selected_upper),
                np.where(positive, scales * selected_upper, scales * selected_lower),
            )

        minimum, maximum = signed_extrema(source_minimum, source_maximum)
        event_minimum, event_maximum = signed_extrema(
            source_event_minimum, source_event_maximum
        )
        noevent_minimum, noevent_maximum = signed_extrema(
            source_noevent_minimum, source_noevent_maximum
        )
        tolerance = 1.0e-14
        if is_poisson_likelihood(self.likelihood):
            event_absolute = np.maximum(np.abs(event_minimum), np.abs(event_maximum))
            event_absolute[~np.isfinite(event_absolute)] = 0.0
            positive_ray = (
                (maximum <= tolerance)
                & (event_absolute <= tolerance)
                & (minimum < -tolerance)
            )
            negative_ray = (
                (minimum >= -tolerance)
                & (event_absolute <= tolerance)
                & (maximum > tolerance)
            )
        else:
            positive_ray = (
                (noevent_maximum <= tolerance)
                & (event_minimum >= -tolerance)
                & ((noevent_minimum < -tolerance) | (event_maximum > tolerance))
            )
            negative_ray = (
                (noevent_minimum >= -tolerance)
                & (event_maximum <= tolerance)
                & ((noevent_maximum > tolerance) | (event_minimum < -tolerance))
            )
        negative_ray[int(free_dimension) :] = False
        return bool(np.any(positive_ray | negative_ray))


def _objective(
    matrix: ModelMatrix,
    likelihood: str,
    beta: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    matrix_token = matrix.resident_token
    if is_poisson_likelihood(likelihood):
        resident = resident_poisson_objective(
            matrix.x,
            beta,
            matrix.exposure_weight,
            matrix.event_weight,
            device=device,
            matrix_token=matrix_token,
            compute_moments=True,
            return_eta=True,
        )
        if resident is not None:
            nll, eta, gradient, hessian = resident
            if eta is None or gradient is None or hessian is None:
                raise AssertionError("resident Poisson moments are incomplete")
            return nll, gradient, hessian, eta
    elif likelihood == "first_event_cloglog":
        resident = resident_cloglog_objective(
            matrix.x,
            beta,
            matrix.noevent_weight,
            matrix.event_weight,
            device=device,
            matrix_token=matrix_token,
            compute_moments=True,
            return_eta=True,
        )
        if resident is not None:
            nll, eta, gradient, hessian = resident
            if eta is None or gradient is None or hessian is None:
                raise AssertionError("resident cloglog moments are incomplete")
            return nll, gradient, hessian, eta
    eta = resident_eta(
        matrix.x,
        beta,
        device=device,
        matrix_token=matrix_token,
    )
    if eta is None:
        eta = matrix.x @ beta
    rows, first, second = loss_rows(
        eta,
        likelihood=likelihood,
        exposure_weight=matrix.exposure_weight,
        noevent_weight=matrix.noevent_weight,
        event_weight=matrix.event_weight,
    )
    gradient, hessian = moments(
        matrix.x,
        first,
        second,
        device=device,
        matrix_token=matrix_token,
    )
    return float(np.sum(rows)), gradient, hessian, eta


def _value(
    matrix: ModelMatrix,
    likelihood: str,
    beta: np.ndarray,
    *,
    device: str = "cpu",
) -> float:
    """Objective-only path for line search; derivatives are not consumed."""
    if is_poisson_likelihood(likelihood):
        resident = resident_poisson_objective(
            matrix.x,
            beta,
            matrix.exposure_weight,
            matrix.event_weight,
            device=device,
            matrix_token=matrix.resident_token,
            compute_moments=False,
            return_eta=False,
        )
        if resident is not None:
            return resident[0]
    elif likelihood == "first_event_cloglog":
        resident = resident_cloglog_objective(
            matrix.x,
            beta,
            matrix.noevent_weight,
            matrix.event_weight,
            device=device,
            matrix_token=matrix.resident_token,
            compute_moments=False,
            return_eta=False,
        )
        if resident is not None:
            return resident[0]
    eta = resident_eta(
        matrix.x,
        beta,
        device=device,
        matrix_token=matrix.resident_token,
    )
    if eta is None:
        eta = matrix.x @ beta
    rows = loss_value_rows(
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
        if not is_poisson_likelihood(likelihood):
            noevent = matrix.noevent_weight[start:end] > 0
            if np.any(noevent):
                selected = tile[noevent]
                noevent_minimum = np.minimum(noevent_minimum, np.min(selected, axis=0))
                noevent_maximum = np.maximum(noevent_maximum, np.max(selected, axis=0))
    if is_poisson_likelihood(likelihood):
        event_absolute = np.maximum(np.abs(event_minimum), np.abs(event_maximum))
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
            & ((noevent_minimum < -tolerance) | (event_maximum > tolerance))
        )
        negative = (
            (noevent_minimum >= -tolerance)
            & (event_maximum <= tolerance)
            & ((noevent_maximum > tolerance) | (event_minimum < -tolerance))
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
    if is_poisson_likelihood(likelihood):
        improving = ~event & (exposure_weight > 0.0)
        if not np.any(improving):
            return False
        objective = x[improving].T @ exposure_weight[improving]
    else:
        noevent_only = noevent & ~event
        event_only = event & ~noevent
        if not np.any(noevent_only) and not np.any(event_only):
            return False
        objective = np.zeros(dimension, dtype=np.float64)
        if np.any(noevent_only):
            objective += x[noevent_only].T @ noevent_weight[noevent_only]
        if np.any(event_only):
            objective -= x[event_only].T @ event_weight[event_only]
    if not np.any(objective):
        return False
    event_indices = np.flatnonzero(event)
    noevent_indices = np.flatnonzero(noevent)
    bounds = [(-1.0, 1.0)] * free_dimension + [(0.0, 1.0)] * (
        dimension - free_dimension
    )
    objective = np.ascontiguousarray(objective)

    def unconstrained_box_minimum() -> np.ndarray:
        direction = np.zeros(dimension, dtype=np.float64)
        free = objective[:free_dimension]
        direction[:free_dimension] = np.where(
            free > 0.0,
            -1.0,
            np.where(free < 0.0, 1.0, 0.0),
        )
        constrained = objective[free_dimension:]
        direction[free_dimension:] = np.where(constrained < 0.0, 1.0, 0.0)
        return direction

    def worst_constraint(
        direction: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:
        predictor = x @ direction
        worst = -math.inf
        row: np.ndarray | None = None

        def consider(values: np.ndarray, rows: np.ndarray, sign: float) -> None:
            nonlocal worst, row
            if not len(values):
                return
            index = int(np.argmax(values))
            value = float(values[index])
            if value > worst:
                worst = value
                row = np.ascontiguousarray(sign * rows[index])

        if is_poisson_likelihood(likelihood):
            consider(predictor, x, 1.0)
            if len(event_indices):
                values = -predictor[event_indices]
                index = int(event_indices[int(np.argmax(values))])
                value = float(-predictor[index])
                if value > worst:
                    worst = value
                    row = np.ascontiguousarray(-x[index])
        else:
            if len(noevent_indices):
                values = predictor[noevent_indices]
                index = int(noevent_indices[int(np.argmax(values))])
                value = float(predictor[index])
                if value > worst:
                    worst = value
                    row = np.ascontiguousarray(x[index])
            if len(event_indices):
                values = -predictor[event_indices]
                index = int(event_indices[int(np.argmax(values))])
                value = float(-predictor[index])
                if value > worst:
                    worst = value
                    row = np.ascontiguousarray(-x[index])
        return worst, row

    # The recession cone lives in a low-dimensional coefficient space while
    # Freddie matrices can contain millions of likelihood groups.  Solve the
    # same box-normalized LP by deterministic constraint generation: a relaxed
    # optimum either proves non-recession immediately or contributes its most
    # violated row.  Only a pathological tail falls back to the complete sparse
    # HiGHS problem.  No row, direction or acceptance threshold is sampled.
    feasibility_tolerance = 1.0e-9
    decision_scale = max(1.0, float(np.linalg.norm(objective, ord=1)))
    recession_threshold = -1.0e-10 * decision_scale
    direction = unconstrained_box_minimum()
    cuts: list[np.ndarray] = []
    max_cuts = max(16, 4 * dimension)
    result = None
    for _ in range(max_cuts):
        relaxed_value = float(objective @ direction)
        # The cut problem is a relaxation of the complete cone.  A
        # nonnegative relaxed optimum proves that the more constrained full LP
        # cannot contain an improving recession direction.
        if relaxed_value >= recession_threshold:
            return False
        violation, cut = worst_constraint(direction)
        if violation <= feasibility_tolerance:
            return True
        if cut is None or any(np.array_equal(cut, previous) for previous in cuts):
            break
        cuts.append(cut)
        cut_matrix = np.ascontiguousarray(np.vstack(cuts))
        result = linprog(
            objective,
            A_ub=cut_matrix,
            b_ub=np.zeros(len(cuts), dtype=np.float64),
            bounds=bounds,
            method="highs",
            options={
                "presolve": True,
                "primal_feasibility_tolerance": feasibility_tolerance,
                "dual_feasibility_tolerance": feasibility_tolerance,
            },
        )
        if (
            not result.success
            or result.fun is None
            or result.x is None
            or not math.isfinite(result.fun)
        ):
            break
        direction = np.ascontiguousarray(result.x, dtype=np.float64)

    # Fail open from the cutting-plane accelerator to the mathematically
    # identical complete LP.  Sparse assembly prevents the old dense vstack
    # from multiplying memory use on a failed full-data candidate.
    sparse_x = csr_matrix(x)
    if is_poisson_likelihood(likelihood):
        sparse_parts = [sparse_x]
        if len(event_indices):
            # Event rows must have Xd=0; Xd<=0 is already included above.
            sparse_parts.append(-sparse_x[event_indices])
    else:
        sparse_parts = []
        if len(noevent_indices):
            sparse_parts.append(sparse_x[noevent_indices])
        if len(event_indices):
            sparse_parts.append(-sparse_x[event_indices])
    if not sparse_parts:
        return False
    sparse_constraints = sparse_vstack(sparse_parts, format="csr")
    result = linprog(
        objective,
        A_ub=sparse_constraints,
        b_ub=np.zeros(sparse_constraints.shape[0], dtype=np.float64),
        bounds=bounds,
        method="highs",
        options={
            "presolve": True,
            "primal_feasibility_tolerance": feasibility_tolerance,
            "dual_feasibility_tolerance": feasibility_tolerance,
        },
    )
    if not result.success or result.fun is None or not math.isfinite(result.fun):
        # Failure to construct a certificate must not label a finite model as
        # recessionary.  The caller retains its original non-convergence.
        return False
    return bool(result.fun < recession_threshold)


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
    *,
    diagnose_recession: bool = True,
) -> FitResult:
    # Fast discovery rejects every nonconverged support regardless of whether
    # the failure is numerical or caused by a recession direction.  Running
    # the complete cone LP in that case changes only the diagnostic label, not
    # any search/certification decision, and can scan a multi-gigabyte design
    # hundreds of times.  Exact/reporting callers retain the full diagnosis.
    recession = _general_recession(matrix, likelihood) if diagnose_recession else False
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


def _critical_cone_rank(
    beta: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    *,
    free_dimension: int,
    tolerance: float,
) -> tuple[int, int]:
    """Return information rank on the constrained optimum's critical face.

    A zero nonnegative coefficient with a strictly positive gradient is fixed
    at the boundary by first-order KKT conditions.  Requiring curvature in
    that infeasible negative direction incorrectly rejects a unique,
    well-attained constrained MLE.  Free, positive, and zero-gradient boundary
    coordinates form the critical cone and are the only directions in which
    Fisher rank is required.
    """

    beta = np.asarray(beta, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    critical = np.arange(len(beta)) < int(free_dimension)
    critical |= beta > 1.0e-12
    critical |= gradient <= max(float(tolerance), 1.0e-12)
    indices = np.flatnonzero(critical)
    if not len(indices):
        return 0, 0
    return (
        _checked_rank(hessian[np.ix_(indices, indices)]),
        int(len(indices)),
    )


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
    feasible primal point for conditional add/drop/identity proposals.  Callers
    attach a Fenchel certificate to the returned point and keep it outside the
    exact-fit cache.  Consequently a one-step proposal can never be mistaken
    for a converged fixed-support estimator.

    The step is the same active-set Newton direction and monotone Armijo search
    used by :func:`fit_model_matrix`.  If the warm point already satisfies KKT,
    it is returned unchanged, but ``converged`` remains false by construction;
    only the full solver may create an exact cached result.
    """
    beta = np.asarray(warm_start, dtype=np.float64).copy()
    if beta.shape != (matrix.dimension,) or not np.all(np.isfinite(beta)):
        raise ValueError("one-step warm start does not match the model matrix")
    if matrix.dimension > matrix.free_dimension:
        beta[matrix.free_dimension :] = np.maximum(beta[matrix.free_dimension :], 0.0)
    nll, gradient, hessian, _ = _objective(matrix, likelihood, beta, device=device)
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
        active_direction = np.linalg.lstsq(sub_hessian, -gradient[indices], rcond=None)[
            0
        ]
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
        trial[matrix.free_dimension :] = np.maximum(trial[matrix.free_dimension :], 0.0)
        trial_nll = _value(matrix, likelihood, trial, device=device)
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
    _check_recession: bool = True,
    _diagnose_recession_on_failure: bool = True,
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
    if warm_start is None and total_exposure > 0:
        # Free baseline columns are disjoint one-hot strata.  Initialize every
        # stratum at its own empirical event rate; the legacy one-intercept
        # case is exactly the first iteration of this loop.
        for column in range(matrix.free_dimension):
            member = matrix.x[:, column] != 0.0
            exposure = float(np.sum(matrix.exposure_weight[member]))
            events = float(np.sum(matrix.event_weight[member]))
            if exposure > 0.0:
                beta[column] = math.log(max(events, 0.5) / exposure)
    # Recession is a property of the fixed design, likelihood and cone; it is
    # independent of the Newton warm point.  The continued wrapper checks it
    # in its first window and skips the identical tall matrix scan thereafter.
    recession = _axis_recession(matrix, likelihood) if _check_recession else False
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
        nll, gradient, hessian, _ = _objective(matrix, likelihood, beta, device=device)
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
                diagnose_recession=_diagnose_recession_on_failure,
            )
        kkt = projected_kkt(beta, gradient, matrix.free_dimension)
        if kkt <= tolerance:
            rank, critical_dimension = _critical_cone_rank(
                beta,
                gradient,
                hessian,
                free_dimension=matrix.free_dimension,
                tolerance=tolerance,
            )
            if rank < critical_dimension:
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
            trial_nll = _value(matrix, likelihood, trial, device=device)
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
                diagnose_recession=_diagnose_recession_on_failure,
            )
        if abs(previous - trial_nll) <= tolerance * max(1.0, abs(previous)):
            final_nll, final_gradient, final_hessian, _ = _objective(
                matrix, likelihood, beta, device=device
            )
            final_kkt = projected_kkt(beta, final_gradient, matrix.free_dimension)
            if final_kkt <= 10.0 * tolerance:
                final_rank, critical_dimension = _critical_cone_rank(
                    beta,
                    final_gradient,
                    final_hessian,
                    free_dimension=matrix.free_dimension,
                    tolerance=10.0 * tolerance,
                )
                if final_rank < critical_dimension:
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
    nll, gradient, hessian, _ = _objective(matrix, likelihood, beta, device=device)
    return _failed_fit(
        matrix,
        likelihood,
        beta,
        nll,
        int(max_iter),
        projected_kkt(beta, gradient, matrix.free_dimension),
        _checked_rank(hessian),
        "maximum iterations reached",
        diagnose_recession=_diagnose_recession_on_failure,
    )


def fit_sparse_grid_model(
    context: Context,
    blocks: tuple[SparseBlock, ...],
    signs: tuple[int, ...],
    *,
    likelihood: str,
    tick_exposure: float,
    tolerance: float,
    max_iter: int,
    warm_start: np.ndarray | None = None,
    device: str = "cpu",
    baseline_group_count: int = 1,
    baseline_time_bins: int = 1,
    shared_active_rows: np.ndarray | None = None,
    shared_active_lookup: np.ndarray | None = None,
    shared_active_baseline_groups: np.ndarray | None = None,
    shared_inactive_exposure: np.ndarray | None = None,
) -> FitResult:
    """Exactly fit stratified intercepts plus signed sparse response blocks.

    The ordinary fixed-support solver materializes a dense row-by-parameter
    sufficient-statistic matrix.  Financial event histories are sparse, so
    almost all of those copied entries are zero.  This solver evaluates the
    identical full-grid likelihood and Fisher system from resident columnar
    blocks.  It uses the same projected Newton, Armijo and KKT criteria as
    :func:`fit_model_matrix`; unsupported devices or numerical failures return
    a non-converged result so callers can fail open to the dense reference.
    """
    baseline_group_count = int(baseline_group_count)
    if baseline_group_count < 1:
        raise ValueError("sparse grid fit requires at least one baseline group")
    if (
        not blocks
        or len(blocks) != len(signs)
        or any(sign not in {-1, 1} for sign in signs)
    ):
        raise ValueError("sparse grid fit requires aligned signed blocks")
    knot_count = int(blocks[0].values.shape[1])
    if knot_count < 1 or any(
        block.values.shape != (len(block.rows), knot_count) for block in blocks
    ):
        raise ValueError("sparse grid blocks must share one knot dimension")
    if context.uniform_entity_weight is None:
        return FitResult(
            np.zeros(
                baseline_group_count + len(blocks) * knot_count,
                dtype=np.float64,
            ),
            math.inf,
            False,
            0,
            math.inf,
            0,
            False,
            "nonuniform entity weights require dense exact fallback",
        )
    if context.baseline_row_exposure is not None:
        # Structural grids may retain rows at which the data source was not
        # observed. Their declared exposure is zero, so deleting them from
        # every sparse response block leaves the likelihood, gradient and
        # Hessian unchanged. Positive fractional exposures remain in the exact
        # row-wise exposure vector below.
        row_exposure = context.baseline_row_exposure
        filtered_blocks: list[SparseBlock] = []
        for block in blocks:
            observed = row_exposure[block.rows] > 0.0
            if np.all(observed):
                filtered_blocks.append(block)
            else:
                filtered_blocks.append(
                    SparseBlock(
                        np.ascontiguousarray(block.rows[observed]),
                        np.ascontiguousarray(block.values[observed]),
                    )
                )
        blocks = tuple(filtered_blocks)
    raw_geometry = tuple((block.rows, block.values) for block in blocks)
    geometry = sparse_moment_geometry(raw_geometry, signs=signs)
    del raw_geometry
    dimension = baseline_group_count + geometry.block_count * geometry.knot_count
    beta = np.zeros(dimension, dtype=np.float64)
    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=np.float64)
        if warm.shape != (dimension,) or not np.all(np.isfinite(warm)):
            raise ValueError("sparse grid warm start does not match the model")
        beta[:] = warm
    beta[baseline_group_count:] = np.maximum(beta[baseline_group_count:], 0.0)
    baseline_totals = context.weighted_baseline_totals(
        baseline_group_count, time_bins=baseline_time_bins
    )
    total_exposure = float(tick_exposure) * baseline_totals
    target_groups = context.temporal_baseline_groups_at_rows(
        context.target_rows, time_bins=baseline_time_bins
    )
    total_events = np.bincount(
        target_groups,
        weights=context.target_counts,
        minlength=baseline_group_count,
    ).astype(np.float64)
    if warm_start is None:
        positive = total_exposure > 0.0
        positive_groups = np.flatnonzero(positive)
        beta[positive_groups] = np.log(
            np.maximum(total_events[positive], 0.5) / total_exposure[positive]
        )

    # Coordinate recession is cheap in the sparse representation.  General
    # combined-ray diagnosis remains the dense fail-open caller's job after a
    # numerical/nonconvergence return, exactly as in fast fixed-support fits.
    for block_index in range(geometry.block_count):
        left = int(geometry.block_offsets[block_index])
        right = int(geometry.block_offsets[block_index + 1])
        rows = geometry.rows[left:right]
        values = geometry.values[left:right]
        if not len(rows):
            continue
        target_positions = np.searchsorted(rows, context.target_rows)
        target_matched = target_positions < len(rows)
        safe = np.minimum(target_positions, len(rows) - 1)
        target_matched &= rows[safe] == context.target_rows
        target_values = (
            values[target_positions[target_matched]]
            if np.any(target_matched)
            else np.zeros((0, knot_count), dtype=np.float64)
        )
        if is_poisson_likelihood(likelihood):
            event_absolute = np.max(np.abs(target_values), axis=0, initial=0.0)
            minimum = np.minimum(0.0, np.min(values, axis=0, initial=0.0))
            maximum = np.maximum(0.0, np.max(values, axis=0, initial=0.0))
            recession = (
                (maximum <= 1.0e-14)
                & (event_absolute <= 1.0e-14)
                & (minimum < -1.0e-14)
            )
        else:
            event_rows = np.zeros(len(rows), dtype=bool)
            if np.any(target_matched):
                event_rows[target_positions[target_matched]] = True
            # Boolean-indexing the complete block allocated another dense
            # values copy solely to obtain four extrema. Accumulate the exact
            # same minima/maxima in bounded tiles instead.
            noevent_minimum = np.zeros(knot_count, dtype=np.float64)
            noevent_maximum = np.zeros(knot_count, dtype=np.float64)
            extrema_tile = 1_048_576
            for tile_left in range(0, len(values), extrema_tile):
                tile_right = min(len(values), tile_left + extrema_tile)
                tile_mask = ~event_rows[tile_left:tile_right]
                if not np.any(tile_mask):
                    continue
                tile_values = values[tile_left:tile_right][tile_mask]
                noevent_minimum = np.minimum(
                    noevent_minimum,
                    np.min(tile_values, axis=0, initial=0.0),
                )
                noevent_maximum = np.maximum(
                    noevent_maximum,
                    np.max(tile_values, axis=0, initial=0.0),
                )
            event_minimum = np.minimum(0.0, np.min(target_values, axis=0, initial=0.0))
            event_maximum = np.maximum(0.0, np.max(target_values, axis=0, initial=0.0))
            recession = (
                (noevent_maximum <= 1.0e-14)
                & (event_minimum >= -1.0e-14)
                & ((noevent_minimum < -1.0e-14) | (event_maximum > 1.0e-14))
            )
        if np.any(recession):
            return FitResult(
                beta,
                math.inf,
                False,
                0,
                math.inf,
                0,
                True,
                "nonattained sparse coordinate recession direction",
            )

    # All non-intercept columns are exactly zero outside the union of sparse
    # block rows.  Include target rows in that union and represent every
    # remaining observation by one analytic intercept-only mass.  This is an
    # exact sufficient-statistic reduction: it changes neither the likelihood
    # nor any Newton derivative, while avoiding two O(n_grid) arrays.
    if shared_active_rows is None:
        active_parts = [context.target_rows]
        for block_index in range(geometry.block_count):
            left = int(geometry.block_offsets[block_index])
            right = int(geometry.block_offsets[block_index + 1])
            if right > left:
                active_parts.append(geometry.rows[left:right])
        active_rows = sorted_unique_union(active_parts)
        if active_rows is None:
            active_rows = np.unique(np.concatenate(active_parts))
    else:
        active_rows = np.ascontiguousarray(shared_active_rows, dtype=np.int64)
    # Every packed block row is a member of ``active_rows``.  A dense int32
    # inverse map changes the old O(nnz log n_active) binary searches into one
    # O(n_grid + nnz) gather.  Even the full Home Credit grid needs only tens
    # of MiB here, while the avoided dense design is several GiB.
    if shared_active_lookup is None:
        active_lookup = np.full(context.n_grid, -1, dtype=np.int32)
        active_lookup[active_rows] = np.arange(len(active_rows), dtype=np.int32)
    else:
        active_lookup = np.asarray(shared_active_lookup, dtype=np.int32)
        if active_lookup.shape != (context.n_grid,):
            raise ValueError("shared sparse active lookup has invalid shape")
    compact_rows = active_lookup[geometry.rows].astype(np.int64, copy=False)
    if len(compact_rows) and np.any(compact_rows < 0):
        raise AssertionError("sparse block row missing from active union")
    packed_values = geometry.values
    packed_offsets = geometry.block_offsets
    packed_block_count = geometry.block_count
    packed_knot_count = geometry.knot_count
    compact_geometry = SparseMomentGeometry(
        rows=compact_rows,
        values=packed_values,
        block_offsets=packed_offsets,
        block_count=packed_block_count,
        knot_count=packed_knot_count,
        token=new_derivative_token(),
    )
    # The compact geometry no longer needs the original packed global-row
    # array. Drop that owner before Newton temporaries are allocated.
    del geometry
    target_positions = np.searchsorted(active_rows, context.target_rows)
    if np.any(target_positions >= len(active_rows)) or not np.array_equal(
        active_rows[target_positions], context.target_rows
    ):
        raise AssertionError("sparse target row missing from active union")
    active_baseline_groups = (
        context.temporal_baseline_groups_at_rows(
            active_rows, time_bins=baseline_time_bins
        )
        if shared_active_baseline_groups is None
        else np.asarray(shared_active_baseline_groups, dtype=np.int32)
    )
    if active_baseline_groups.shape != active_rows.shape:
        raise ValueError("shared sparse baseline groups do not match active rows")
    if np.any(active_baseline_groups < 0) or np.any(
        active_baseline_groups >= baseline_group_count
    ):
        raise ValueError("shared sparse baseline group lies outside dictionary")
    active_weight = context.weights_at_rows(active_rows)
    active_exposure = float(tick_exposure) * active_weight
    if shared_inactive_exposure is None:
        active_weight_by_group = np.bincount(
            active_baseline_groups,
            weights=active_weight,
            minlength=baseline_group_count,
        )
        inactive_exposure = float(tick_exposure) * np.maximum(
            baseline_totals - active_weight_by_group, 0.0
        )
    else:
        inactive_exposure = np.asarray(shared_inactive_exposure, dtype=np.float64)
        if inactive_exposure.shape != (baseline_group_count,):
            raise ValueError("shared sparse inactive exposure has invalid shape")
    eta = np.empty(len(active_rows), dtype=np.float64)
    mean = np.empty(len(active_rows), dtype=np.float64)

    # Keep the complete sparse predictor geometry resident for Newton and,
    # crucially, Armijo value-only trials.  The former implementation rebuilt
    # an O(n_active) NumPy predictor for as many as 60 line-search steps; CUDA
    # then sat idle even though the subsequent moments were device-resident.
    # This path evaluates the identical float64 formulas and falls back to the
    # NumPy reference when PyTorch/CUDA is unavailable or the upload fails.
    torch_state = None
    torch_design = None
    if device.startswith("cuda") and len(active_rows):
        try:
            import torch

            torch_device = torch.device(device)
            torch_rows = torch.as_tensor(
                compact_geometry.rows, dtype=torch.int64, device=torch_device
            )
            torch_values = torch.as_tensor(
                compact_geometry.values, dtype=torch.float64, device=torch_device
            )
            torch_groups = torch.as_tensor(
                active_baseline_groups, dtype=torch.int64, device=torch_device
            )
            torch_targets = torch.as_tensor(
                target_positions, dtype=torch.int64, device=torch_device
            )
            torch_target_counts = torch.as_tensor(
                context.target_counts, dtype=torch.float64, device=torch_device
            )
            torch_active_exposure = torch.as_tensor(
                active_exposure, dtype=torch.float64, device=torch_device
            )
            torch_inactive_exposure = torch.as_tensor(
                inactive_exposure, dtype=torch.float64, device=torch_device
            )
            torch_state = (
                torch,
                torch_device,
                torch_rows,
                torch_values,
                torch_groups,
                torch_targets,
                torch_target_counts,
                torch_active_exposure,
                torch_inactive_exposure,
            )
            # The model dimension is tiny (M columns per active block), while
            # repeatedly gathering every sparse block for each Newton and
            # Armijo evaluation is memory-bandwidth bound.  Materialize the
            # candidate's compact active-row design once on its assigned GPU.
            # This is still the exact same design: rows outside ``active_rows``
            # are represented analytically by ``inactive_exposure`` below.
            # Allocation failure is an execution-only fallback to the resident
            # sparse representation and cannot change the fitted objective.
            torch_design = torch.zeros(
                (
                    len(active_rows),
                    compact_geometry.block_count * knot_count,
                ),
                dtype=torch.float64,
                device=torch_device,
            )
            for block_index in range(compact_geometry.block_count):
                left = int(compact_geometry.block_offsets[block_index])
                right = int(compact_geometry.block_offsets[block_index + 1])
                if right <= left:
                    continue
                column_left = block_index * knot_count
                torch_design[
                    torch_rows[left:right],
                    column_left : column_left + knot_count,
                ] = torch_values[left:right]
        except (ImportError, RuntimeError, MemoryError):
            torch_state = None
            torch_design = None

    def torch_predictor(vector: np.ndarray):
        if torch_state is None:
            return None
        (
            torch,
            torch_device,
            torch_rows,
            torch_values,
            torch_groups,
            _,
            _,
            _,
            _,
        ) = torch_state
        coefficients = torch.as_tensor(
            np.ascontiguousarray(vector), dtype=torch.float64, device=torch_device
        )
        current_eta = coefficients[:baseline_group_count][torch_groups].clone()
        if torch_design is not None:
            current_eta.add_(torch_design.mv(coefficients[baseline_group_count:]))
            return current_eta
        for block_index in range(compact_geometry.block_count):
            left = int(compact_geometry.block_offsets[block_index])
            right = int(compact_geometry.block_offsets[block_index + 1])
            if right <= left:
                continue
            coefficient_left = baseline_group_count + block_index * knot_count
            contribution = torch_values[left:right].mv(
                coefficients[coefficient_left : coefficient_left + knot_count]
            )
            current_eta.index_add_(0, torch_rows[left:right], contribution)
        return current_eta

    def torch_values(vectors: np.ndarray) -> np.ndarray | None:
        """Evaluate a small ordered Armijo step batch on the resident grid."""
        if torch_state is None:
            return None
        (
            torch,
            torch_device,
            torch_rows,
            torch_values_array,
            torch_groups,
            torch_targets,
            torch_target_counts,
            torch_active_exposure,
            torch_inactive_exposure,
        ) = torch_state
        coefficients = torch.as_tensor(
            np.ascontiguousarray(vectors), dtype=torch.float64, device=torch_device
        )
        batch = int(coefficients.shape[0])
        batch_eta = coefficients[:, :baseline_group_count][
            :, torch_groups
        ].T.contiguous()
        if torch_design is not None:
            batch_eta.add_(torch_design.mm(coefficients[:, baseline_group_count:].T))
        else:
            for block_index in range(compact_geometry.block_count):
                left = int(compact_geometry.block_offsets[block_index])
                right = int(compact_geometry.block_offsets[block_index + 1])
                if right <= left:
                    continue
                coefficient_left = baseline_group_count + block_index * knot_count
                contribution = torch_values_array[left:right].mm(
                    coefficients[:, coefficient_left : coefficient_left + knot_count].T
                )
                batch_eta.index_add_(0, torch_rows[left:right], contribution)
        if is_poisson_likelihood(likelihood):
            result = (
                torch.exp(coefficients[:, :baseline_group_count]).mm(
                    torch_inactive_exposure[:, None]
                )[:, 0]
                + (torch_active_exposure[:, None] * torch.exp(batch_eta)).sum(dim=0)
                - (batch_eta[torch_targets] * torch_target_counts[:, None]).sum(dim=0)
            )
            return result.cpu().numpy()
        if likelihood != "first_event_cloglog":
            return np.full(batch, math.inf, dtype=np.float64)
        intensity = torch.exp(torch.clamp(batch_eta, -745.0, 700.0))
        target_intensity = intensity[torch_targets]
        event_value = torch.empty_like(target_intensity)
        small = target_intensity < 1.0e-4
        large = target_intensity > 40.0
        middle = ~(small | large)
        if bool(small.any()):
            values = target_intensity[small]
            event_value[small] = (
                -torch.log(torch.clamp(values, min=torch.finfo(torch.float64).tiny))
                + values / 2.0
                - values * values / 24.0
            )
        if bool(middle.any()):
            values = target_intensity[middle]
            event_value[middle] = -torch.log(-torch.expm1(-values))
        if bool(large.any()):
            values = target_intensity[large]
            event_value[large] = -torch.log1p(-torch.exp(-values))
        inactive = torch.exp(
            torch.clamp(coefficients[:, :baseline_group_count], max=700.0)
        ).mm(torch_inactive_exposure[:, None])[:, 0]
        result = (
            inactive
            + (torch_active_exposure[:, None] * intensity).sum(dim=0)
            + ((event_value - target_intensity) * torch_target_counts[:, None]).sum(
                dim=0
            )
        )
        return result.cpu().numpy()

    def predictor(vector: np.ndarray) -> np.ndarray:
        device_eta = torch_predictor(vector)
        if device_eta is not None:
            eta[:] = device_eta.cpu().numpy()
            return eta
        eta[:] = vector[active_baseline_groups]
        for block_index in range(compact_geometry.block_count):
            left = int(compact_geometry.block_offsets[block_index])
            right = int(compact_geometry.block_offsets[block_index + 1])
            if right <= left:
                continue
            coefficient_left = baseline_group_count + block_index * knot_count
            eta[compact_geometry.rows[left:right]] += (
                compact_geometry.values[left:right]
                @ vector[coefficient_left : coefficient_left + knot_count]
            )
        return eta

    def value(vector: np.ndarray) -> float:
        device_eta = torch_predictor(vector)
        if device_eta is not None:
            (
                torch,
                torch_device,
                _,
                _,
                _,
                torch_targets,
                torch_target_counts,
                torch_active_exposure,
                torch_inactive_exposure,
            ) = torch_state
            coefficients = torch.as_tensor(
                np.ascontiguousarray(vector),
                dtype=torch.float64,
                device=torch_device,
            )
            if is_poisson_likelihood(likelihood):
                result = (
                    torch_inactive_exposure.dot(
                        torch.exp(coefficients[:baseline_group_count])
                    )
                    + torch_active_exposure.dot(torch.exp(device_eta))
                    - torch_target_counts.dot(device_eta[torch_targets])
                )
                return float(result.item())
            if likelihood != "first_event_cloglog":
                return math.inf
            intensity = torch.exp(torch.clamp(device_eta, -745.0, 700.0))
            target_intensity = intensity[torch_targets]
            event_value = torch.empty_like(target_intensity)
            small = target_intensity < 1.0e-4
            large = target_intensity > 40.0
            middle = ~(small | large)
            if bool(small.any()):
                values = target_intensity[small]
                event_value[small] = (
                    -torch.log(torch.clamp(values, min=torch.finfo(torch.float64).tiny))
                    + values / 2.0
                    - values * values / 24.0
                )
            if bool(middle.any()):
                values = target_intensity[middle]
                event_value[middle] = -torch.log(-torch.expm1(-values))
            if bool(large.any()):
                values = target_intensity[large]
                event_value[large] = -torch.log1p(-torch.exp(-values))
            result = (
                torch_inactive_exposure.dot(
                    torch.exp(
                        torch.minimum(
                            coefficients[:baseline_group_count],
                            torch.full_like(coefficients[:baseline_group_count], 700.0),
                        )
                    )
                )
                + torch_active_exposure.dot(intensity)
                + torch_target_counts.dot(event_value - target_intensity)
            )
            return float(result.item())
        current_eta = predictor(vector)
        if is_poisson_likelihood(likelihood):
            with np.errstate(over="ignore"):
                np.exp(current_eta, out=mean)
                inactive_intensity = np.exp(vector[:baseline_group_count])
            return float(
                inactive_exposure @ inactive_intensity
                + active_exposure @ mean
                - context.target_counts @ current_eta[target_positions]
            )
        if likelihood != "first_event_cloglog":
            return math.inf
        np.clip(current_eta, -745.0, 700.0, out=mean)
        np.exp(mean, out=mean)
        event_value, _, _ = cloglog_event_terms(current_eta[target_positions])
        return float(
            inactive_exposure @ np.exp(np.minimum(vector[:baseline_group_count], 700.0))
            + active_exposure @ mean
            + context.target_counts @ (event_value - mean[target_positions])
        )

    def torch_objective(
        vector: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        """Evaluate the exact sparse Newton system without host grid copies.

        ``torch_design`` is the compact active-row representation of precisely
        the same signed sparse columns used by ``sparse_model_moments``.  Only
        the final O(p) gradient and O(p^2) Hessian cross the PCIe boundary.
        The old CUDA path copied the O(n_active) predictor to NumPy, formed two
        derivative grids on the host, and uploaded both grids again on every
        Newton iteration; that transfer dominated final representation audits.
        """

        if torch_state is None or torch_design is None:
            return None
        (
            torch,
            torch_device,
            _,
            _,
            torch_groups,
            torch_targets,
            torch_target_counts,
            torch_active_exposure,
            torch_inactive_exposure,
        ) = torch_state
        coefficients = torch.as_tensor(
            np.ascontiguousarray(vector), dtype=torch.float64, device=torch_device
        )
        current_eta = coefficients[:baseline_group_count][torch_groups].clone()
        current_eta.add_(torch_design.mv(coefficients[baseline_group_count:]))

        if is_poisson_likelihood(likelihood):
            intensity = torch.exp(current_eta)
            first_device = torch_active_exposure * intensity
            second_device = first_device.clone()
            if int(torch_targets.numel()):
                first_device.index_add_(0, torch_targets, -torch_target_counts)
            inactive_mean = torch_inactive_exposure * torch.exp(
                coefficients[:baseline_group_count]
            )
            nll_device = (
                inactive_mean.sum()
                + torch_active_exposure.dot(intensity)
                - torch_target_counts.dot(current_eta[torch_targets])
            )
        elif likelihood == "first_event_cloglog":
            intensity = torch.exp(torch.clamp(current_eta, -745.0, 700.0))
            first_device = torch_active_exposure * intensity
            second_device = first_device.clone()
            target_intensity = intensity[torch_targets]
            event_value = torch.empty_like(target_intensity)
            event_first = torch.empty_like(target_intensity)
            event_second = torch.empty_like(target_intensity)
            small = target_intensity < 1.0e-4
            large = target_intensity > 40.0
            middle = ~(small | large)
            if bool(small.any()):
                values = target_intensity[small]
                square = values * values
                event_value[small] = (
                    -torch.log(torch.clamp(values, min=torch.finfo(torch.float64).tiny))
                    + values / 2.0
                    - square / 24.0
                )
                event_first[small] = -1.0 + values / 2.0 - square / 12.0
                event_second[small] = values / 2.0 - square / 6.0
            if bool(middle.any()):
                values = target_intensity[middle]
                denominator = torch.expm1(values)
                exponential = denominator + 1.0
                event_value[middle] = -torch.log(-torch.expm1(-values))
                event_first[middle] = -values / denominator
                event_second[middle] = (
                    values
                    * ((values - 1.0) * exponential + 1.0)
                    / (denominator * denominator)
                )
            if bool(large.any()):
                values = target_intensity[large]
                tail = torch.exp(-values)
                event_value[large] = -torch.log1p(-tail)
                event_first[large] = (
                    -values
                    * tail
                    / torch.clamp(1.0 - tail, min=torch.finfo(torch.float64).tiny)
                )
                large_second = torch.zeros_like(values)
                safe_large = values < 100.0
                if bool(safe_large.any()):
                    selected = values[safe_large]
                    large_second[safe_large] = (
                        selected * (selected - 1.0) * tail[safe_large]
                    )
                event_second[large] = large_second
            event_second.clamp_(min=0.0)
            if int(torch_targets.numel()):
                first_device.index_add_(
                    0,
                    torch_targets,
                    torch_target_counts * (event_first - target_intensity),
                )
                second_device.index_add_(
                    0,
                    torch_targets,
                    torch_target_counts * (event_second - target_intensity),
                )
            inactive_mean = torch_inactive_exposure * torch.exp(
                torch.clamp(coefficients[:baseline_group_count], max=700.0)
            )
            nll_device = (
                inactive_mean.sum()
                + torch_active_exposure.dot(intensity)
                + torch_target_counts.dot(event_value - target_intensity)
            )
        else:
            return None

        response_dimension = int(torch_design.shape[1])
        gradient_device = torch.empty(
            dimension, dtype=torch.float64, device=torch_device
        )
        gradient_device[:baseline_group_count] = (
            torch.zeros(
                baseline_group_count, dtype=torch.float64, device=torch_device
            ).scatter_add_(0, torch_groups, first_device)
            + inactive_mean
        )
        gradient_device[baseline_group_count:] = torch_design.T.mv(first_device)

        hessian_device = torch.zeros(
            (dimension, dimension), dtype=torch.float64, device=torch_device
        )
        baseline_second = (
            torch.zeros(
                baseline_group_count, dtype=torch.float64, device=torch_device
            ).scatter_add_(0, torch_groups, second_device)
            + inactive_mean
        )
        baseline_indices = torch.arange(
            baseline_group_count, dtype=torch.int64, device=torch_device
        )
        hessian_device[baseline_indices, baseline_indices] = baseline_second

        # Bound the only temporary dense allocation independently of the data
        # size.  The accumulation is over disjoint row chunks and retains the
        # exact float64 Fisher formula X' diag(second) X.
        response_hessian = hessian_device[baseline_group_count:, baseline_group_count:]
        baseline_cross = hessian_device[:baseline_group_count, baseline_group_count:]
        # Use the largest row tile whose weighted-design temporary stays below
        # 512 MiB.  The previous fixed 262k tile synchronized Python with CUDA
        # hundreds of times on Aave even though both 24-GiB devices had ample
        # headroom.  Tile size changes only the reduction schedule, not the
        # objective or accepted convergence criterion.
        bytes_per_row = 8 * max(1, response_dimension)
        row_chunk = max(
            1,
            min(
                len(active_rows),
                max(262_144, (512 * 1024**2) // bytes_per_row),
            ),
        )
        for row_left in range(0, len(active_rows), row_chunk):
            row_right = min(len(active_rows), row_left + row_chunk)
            design_chunk = torch_design[row_left:row_right]
            weighted = design_chunk * second_device[row_left:row_right, None]
            response_hessian.add_(design_chunk.T.mm(weighted))
            groups_chunk = torch_groups[row_left:row_right]
            # This is the same grouped sum used by the baseline gradient
            # above, now applied to all response columns in one operation.
            # The former Python loop rescanned this complete row tile once per
            # baseline group. ``index_add_`` visits each weighted row once and
            # leaves the likelihood, Hessian and KKT equations unchanged.
            baseline_cross.index_add_(0, groups_chunk, weighted)
        hessian_device[baseline_group_count:, :baseline_group_count] = baseline_cross.T
        # ``response_dimension`` is intentionally asserted after construction:
        # it catches an accidental mismatch between the compact design and the
        # coefficient layout before a numerically plausible fit can escape.
        if response_dimension != dimension - baseline_group_count:
            raise AssertionError("resident sparse response dimension mismatch")
        return (
            float(nll_device.item()),
            gradient_device.cpu().numpy(),
            hessian_device.cpu().numpy(),
        )

    def objective(
        vector: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        device_result = torch_objective(vector)
        if device_result is not None:
            return device_result
        current_eta = predictor(vector)
        first, second = loss_grid_sparse_event_derivatives(
            current_eta,
            likelihood=likelihood,
            exposure=active_exposure,
            event_rows=target_positions,
            event_counts=context.target_counts,
        )
        block_moments = sparse_model_moments(
            compact_geometry,
            first,
            second,
            device=device,
            derivative_token=new_derivative_token(),
        )
        if block_moments is None:
            return None
        block_gradient, block_hessian, intercept_cross = block_moments
        gradient = np.empty(dimension, dtype=np.float64)
        with np.errstate(over="ignore"):
            inactive_mean = inactive_exposure * np.exp(vector[:baseline_group_count])
        gradient[:baseline_group_count] = (
            np.bincount(
                active_baseline_groups,
                weights=first,
                minlength=baseline_group_count,
            )
            + inactive_mean
        )
        gradient[baseline_group_count:] = block_gradient
        hessian = np.zeros((dimension, dimension), dtype=np.float64)
        baseline_second = (
            np.bincount(
                active_baseline_groups,
                weights=second,
                minlength=baseline_group_count,
            )
            + inactive_mean
        )
        hessian[np.arange(baseline_group_count), np.arange(baseline_group_count)] = (
            baseline_second
        )
        if baseline_group_count == 1:
            hessian[0, baseline_group_count:] = intercept_cross
        else:
            for block_index in range(compact_geometry.block_count):
                left = int(compact_geometry.block_offsets[block_index])
                right = int(compact_geometry.block_offsets[block_index + 1])
                if right <= left:
                    continue
                rows = compact_geometry.rows[left:right]
                weighted_values = (
                    compact_geometry.values[left:right] * second[rows, None]
                )
                groups = active_baseline_groups[rows]
                column_left = baseline_group_count + block_index * knot_count
                # Each response block has only M kernel columns.  Accumulate
                # those columns by group instead of scanning the same sparse
                # block once for every baseline group.  np.bincount preserves
                # the complete input and computes the identical grouped cross
                # moments used by the reference formula.
                for column in range(knot_count):
                    hessian[
                        :baseline_group_count, column_left + column
                    ] = np.bincount(
                        groups,
                        weights=weighted_values[:, column],
                        minlength=baseline_group_count,
                    )
        hessian[baseline_group_count:, :baseline_group_count] = hessian[
            :baseline_group_count, baseline_group_count:
        ].T
        hessian[baseline_group_count:, baseline_group_count:] = block_hessian
        return value(vector), gradient, 0.5 * (hessian + hessian.T)

    previous = math.inf
    rank = 0
    for iteration in range(1, int(max_iter) + 1):
        evaluated = objective(beta)
        if evaluated is None:
            return FitResult(
                beta,
                math.inf,
                False,
                iteration,
                math.inf,
                rank,
                False,
                "resident sparse CUDA moments unavailable",
            )
        nll, gradient, hessian = evaluated
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
                "nonfinite sparse objective derivatives",
            )
        kkt = projected_kkt(beta, gradient, baseline_group_count)
        if kkt <= tolerance:
            rank, critical_dimension = _critical_cone_rank(
                beta,
                gradient,
                hessian,
                free_dimension=baseline_group_count,
                tolerance=tolerance,
            )
            if rank < critical_dimension:
                return FitResult(
                    beta,
                    nll,
                    False,
                    iteration,
                    kkt,
                    rank,
                    False,
                    "rank-deficient sparse fixed-support information",
                )
            return FitResult(beta, nll, True, iteration, kkt, rank, False, "converged")
        active = np.arange(dimension) < baseline_group_count
        active |= (beta > 1.0e-12) | (gradient < 0.0)
        indices = np.flatnonzero(active)
        if not len(indices):
            return FitResult(
                beta,
                nll,
                False,
                iteration,
                kkt,
                0,
                False,
                "empty sparse Newton active set",
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
        with np.errstate(over="ignore", invalid="ignore"):
            directional = float(gradient @ direction)
        if (
            not np.all(np.isfinite(direction))
            or not math.isfinite(directional)
            or directional >= 0.0
        ):
            direction = -gradient
            direction[baseline_group_count:] = np.where(
                (beta[baseline_group_count:] <= 1.0e-12)
                & (direction[baseline_group_count:] < 0.0),
                0.0,
                direction[baseline_group_count:],
            )
        # An ill-conditioned Fisher block can produce a formally descending
        # Newton vector with coefficients near float overflow.  Backtracking
        # then spends all 60 trials rediscovering that the first dozens are
        # invalid.  A deterministic infinity-norm trust radius is a standard
        # damped-Newton reparameterization: it changes only the step length,
        # not the convex objective, feasible cone, KKT condition, or optimum.
        direction_norm = float(np.linalg.norm(direction, ord=np.inf))
        trust_radius = max(
            1.0,
            1.0 + float(np.linalg.norm(beta, ord=np.inf)),
        )
        if math.isfinite(direction_norm) and direction_norm > trust_radius:
            direction *= trust_radius / direction_norm

        def armijo_search(
            candidate_direction: np.ndarray,
        ) -> tuple[np.ndarray | None, float]:
            if not np.all(np.isfinite(candidate_direction)):
                return None, math.inf
            if torch_state is not None:
                for left_power in range(0, 60, 4):
                    powers = np.arange(
                        left_power, min(60, left_power + 4), dtype=np.int64
                    )
                    steps = np.ldexp(np.ones(len(powers), dtype=np.float64), -powers)
                    trials = (
                        beta[None, :] + steps[:, None] * candidate_direction[None, :]
                    )
                    trials[:, baseline_group_count:] = np.maximum(
                        trials[:, baseline_group_count:], 0.0
                    )
                    trial_values = torch_values(trials)
                    if trial_values is None:
                        break
                    for trial, candidate_nll in zip(trials, trial_values, strict=True):
                        displacement = trial - beta
                        with np.errstate(over="ignore", invalid="ignore"):
                            armijo_slope = float(gradient @ displacement)
                        if (
                            math.isfinite(float(candidate_nll))
                            and math.isfinite(armijo_slope)
                            and float(candidate_nll) <= nll + 1.0e-4 * armijo_slope
                        ):
                            return trial, float(candidate_nll)
                # The resident evaluator has tested the complete ordered step
                # set.  Repeating those same 60 values on CPU was the observed
                # 16-minute fallback loop and cannot create a mathematical
                # Armijo step absent a backend arithmetic disagreement.
                return None, math.inf
            for step_power in range(60):
                step = math.ldexp(1.0, -step_power)
                trial = beta + step * candidate_direction
                trial[baseline_group_count:] = np.maximum(
                    trial[baseline_group_count:], 0.0
                )
                candidate_nll = value(trial)
                displacement = trial - beta
                with np.errstate(over="ignore", invalid="ignore"):
                    armijo_slope = float(gradient @ displacement)
                if (
                    math.isfinite(candidate_nll)
                    and math.isfinite(armijo_slope)
                    and candidate_nll <= nll + 1.0e-4 * armijo_slope
                ):
                    return trial, float(candidate_nll)
            return None, math.inf

        accepted = False
        trial, trial_nll = armijo_search(direction)
        if trial is None:
            # Retry with a diagonally scaled projected-gradient direction.
            # It is a strict descent direction whenever the KKT residual is
            # nonzero and avoids rejecting an otherwise finite MLE because one
            # Newton system was ill-conditioned.
            diagonal = np.maximum(np.diag(hessian), 1.0e-12)
            stable_direction = -gradient / diagonal
            stable_direction[baseline_group_count:] = np.where(
                (beta[baseline_group_count:] <= 1.0e-12)
                & (stable_direction[baseline_group_count:] < 0.0),
                0.0,
                stable_direction[baseline_group_count:],
            )
            stable_norm = float(np.linalg.norm(stable_direction, ord=np.inf))
            if math.isfinite(stable_norm) and stable_norm > trust_radius:
                stable_direction *= trust_radius / stable_norm
            trial, trial_nll = armijo_search(stable_direction)
        if trial is not None:
            beta = trial
            previous = nll
            accepted = True
        if not accepted:
            return FitResult(
                beta,
                nll,
                False,
                iteration,
                kkt,
                rank,
                False,
                "sparse line search failed",
            )
        if abs(previous - trial_nll) <= tolerance * max(1.0, abs(previous)):
            final = objective(beta)
            if final is None:
                break
            final_nll, final_gradient, final_hessian = final
            final_kkt = projected_kkt(beta, final_gradient, baseline_group_count)
            if final_kkt <= 10.0 * tolerance:
                final_rank, critical_dimension = _critical_cone_rank(
                    beta,
                    final_gradient,
                    final_hessian,
                    free_dimension=baseline_group_count,
                    tolerance=10.0 * tolerance,
                )
                if final_rank < critical_dimension:
                    return FitResult(
                        beta,
                        final_nll,
                        False,
                        iteration,
                        final_kkt,
                        final_rank,
                        False,
                        "rank-deficient sparse fixed-support information",
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
    final = objective(beta)
    if final is None:
        return FitResult(
            beta,
            math.inf,
            False,
            int(max_iter),
            math.inf,
            rank,
            False,
            "resident sparse CUDA moments unavailable",
        )
    nll, gradient, hessian = final
    return FitResult(
        beta,
        nll,
        False,
        int(max_iter),
        projected_kkt(beta, gradient, baseline_group_count),
        _checked_rank(hessian),
        False,
        "maximum sparse iterations reached",
    )


def fit_sparse_grid_models_shared(
    context: Context,
    block_families: tuple[tuple[SparseBlock, ...], ...],
    sign_families: tuple[tuple[int, ...], ...],
    *,
    likelihood: str,
    tick_exposure: float,
    tolerance: float,
    max_iter: int,
    baseline_group_count: int,
    baseline_time_bins: int = 1,
    warm_starts: tuple[np.ndarray | None, ...] | None = None,
    devices: tuple[str, ...] = ("cpu",),
    max_workers: int | None = None,
) -> list[FitResult]:
    """Exactly fit sparse support projections on one shared active grid.

    The union of target and response rows, its O(n_grid) inverse lookup,
    baseline strata, and inactive-stratum exposure are support independent.
    Building them once removes the dominant repeated work in a representation
    audit.  Each support still runs the same float64 projected Newton/KKT
    solver on its own signed blocks and therefore has the same optimum as
    :func:`fit_sparse_grid_model` called separately.
    """

    if len(block_families) != len(sign_families):
        raise ValueError("shared sparse families are not aligned")
    if not block_families:
        return []
    starts = (
        (None,) * len(block_families) if warm_starts is None else tuple(warm_starts)
    )
    if len(starts) != len(block_families):
        raise ValueError("shared sparse warm starts are not aligned")
    if context.uniform_entity_weight is None:
        return [
            FitResult(
                np.zeros(
                    int(baseline_group_count)
                    + sum(block.values.shape[1] for block in blocks),
                    dtype=np.float64,
                ),
                math.inf,
                False,
                0,
                math.inf,
                0,
                False,
                "nonuniform entity weights require dense exact fallback",
            )
            for blocks in block_families
        ]
    if context.baseline_row_exposure is not None:
        row_exposure = context.baseline_row_exposure
        block_families = tuple(
            tuple(
                block
                if np.all(row_exposure[block.rows] > 0.0)
                else SparseBlock(
                    np.ascontiguousarray(
                        block.rows[row_exposure[block.rows] > 0.0]
                    ),
                    np.ascontiguousarray(
                        block.values[row_exposure[block.rows] > 0.0]
                    ),
                )
                for block in blocks
            )
            for blocks in block_families
        )

    active_parts: list[np.ndarray] = [context.target_rows]
    seen_blocks: set[int] = set()
    for blocks in block_families:
        for block in blocks:
            identity = id(block.rows)
            if identity in seen_blocks or not len(block.rows):
                continue
            seen_blocks.add(identity)
            active_parts.append(block.rows)
    active_rows = sorted_unique_union(active_parts)
    if active_rows is None:
        active_rows = np.unique(np.concatenate(active_parts))
    active_rows = np.ascontiguousarray(active_rows, dtype=np.int64)
    active_lookup = np.full(context.n_grid, -1, dtype=np.int32)
    active_lookup[active_rows] = np.arange(len(active_rows), dtype=np.int32)
    active_groups = context.temporal_baseline_groups_at_rows(
        active_rows, time_bins=baseline_time_bins
    )
    baseline_totals = context.weighted_baseline_totals(
        baseline_group_count, time_bins=baseline_time_bins
    )
    active_weight = context.weights_at_rows(active_rows)
    active_weight_by_group = np.bincount(
        active_groups,
        weights=active_weight,
        minlength=int(baseline_group_count),
    )
    inactive_exposure = float(tick_exposure) * np.maximum(
        baseline_totals - active_weight_by_group, 0.0
    )

    physical_devices = tuple(dict.fromkeys(devices)) or ("cpu",)
    cuda_devices = tuple(
        device for device in physical_devices if device.startswith("cuda")
    )
    execution_devices: tuple[str, ...]
    if cuda_devices:
        # A sparse fit alternates CPU construction/small Newton algebra with
        # CUDA reductions.  One producer per device leaves gaps between those
        # phases.  Derive the producer count from currently free memory and
        # the caller's declared exact-worker budget.  The former hard cap of
        # three producers per GPU left half of a 12-worker/two-GPU Aave audit
        # idle even though the measured peak stayed below half of device RAM.
        # Concurrency changes only the execution schedule; each candidate
        # retains its own exact float64 objective, line search and KKT test.
        largest_response_dimension = max(
            sum(int(block.values.shape[1]) for block in blocks)
            for blocks in block_families
        )
        largest_geometry_bytes = max(
            sum(block.rows.nbytes + block.values.nbytes for block in blocks)
            for blocks in block_families
        )
        # Resident design + predictor/derivative/Armijo buffers + the bounded
        # 512-MiB Hessian tile.  The factor of 1.25 covers allocator rounding
        # and library workspaces without relying on a data-specific threshold.
        estimated_peak = int(
            1.25
            * (
                len(active_rows) * 8 * (largest_response_dimension + 10)
                + largest_geometry_bytes
                + 512 * 1024**2
            )
        )
        capacities: list[int] = []
        try:
            import torch

            for device in cuda_devices:
                with torch.cuda.device(torch.device(device)):
                    free_bytes, _ = torch.cuda.mem_get_info()
                # Keep 1 GiB uncommitted for the context/native pricing cache.
                usable = max(0, int(free_bytes) - 1024**3)
                capacities.append(max(1, usable // max(1, estimated_peak)))
        except (ImportError, RuntimeError):
            capacities = [1] * len(cuda_devices)
        worker_limit = len(block_families)
        if max_workers is not None:
            worker_limit = min(worker_limit, max(1, int(max_workers)))
        # Allocate slots round-robin so both devices remain active even when
        # their free-memory capacities differ.  This is deterministic and
        # never exceeds either the memory-derived capacity or worker budget.
        remaining = list(capacities)
        scheduled: list[str] = []
        while len(scheduled) < worker_limit and any(value > 0 for value in remaining):
            for index, device in enumerate(cuda_devices):
                if len(scheduled) >= worker_limit:
                    break
                if remaining[index] <= 0:
                    continue
                scheduled.append(device)
                remaining[index] -= 1
        execution_devices = tuple(scheduled) or (cuda_devices[0],)
    else:
        cpu_workers = len(block_families)
        if max_workers is not None:
            cpu_workers = min(cpu_workers, max(1, int(max_workers)))
        execution_devices = (physical_devices[0],) * max(1, cpu_workers)
    output: list[FitResult | None] = [None] * len(block_families)
    next_index = 0
    queue_lock = threading.Lock()

    def consume(device: str) -> None:
        nonlocal next_index
        configure_cpu_threads(1)
        while True:
            with queue_lock:
                if next_index >= len(block_families):
                    return
                index = next_index
                next_index += 1
            output[index] = fit_sparse_grid_model(
                context,
                block_families[index],
                sign_families[index],
                likelihood=likelihood,
                tick_exposure=tick_exposure,
                tolerance=tolerance,
                max_iter=max_iter,
                warm_start=starts[index],
                device=device,
                baseline_group_count=baseline_group_count,
                baseline_time_bins=baseline_time_bins,
                shared_active_rows=active_rows,
                shared_active_lookup=active_lookup,
                shared_active_baseline_groups=active_groups,
                shared_inactive_exposure=inactive_exposure,
            )

    if len(execution_devices) == 1:
        consume(execution_devices[0])
    else:
        with ThreadPoolExecutor(
            max_workers=len(execution_devices),
            thread_name_prefix="crbstpp-shared-sparse-grid",
        ) as executor:
            list(executor.map(consume, execution_devices))
    if any(result is None for result in output):
        raise AssertionError("shared sparse-grid queue lost a fit")
    return [result for result in output if result is not None]


def _projected_axis_recession(
    matrix: ModelMatrix,
    columns: np.ndarray,
    scales: np.ndarray,
    *,
    likelihood: str,
    free_dimension: int,
) -> bool:
    """Detect coordinate recession rays without materializing ``X[:, columns]``.

    The projected design is exactly ``X_p = X[:, columns] * scales``.  Only a
    bounded row tile is gathered at a time, so a Drop audit never owns another
    observation-sized model matrix.
    """
    dimension = len(columns)
    tolerance = 1.0e-14
    minimum = np.full(dimension, np.inf, dtype=np.float64)
    maximum = np.full(dimension, -np.inf, dtype=np.float64)
    event_minimum = np.full(dimension, np.inf, dtype=np.float64)
    event_maximum = np.full(dimension, -np.inf, dtype=np.float64)
    noevent_minimum = np.full(dimension, np.inf, dtype=np.float64)
    noevent_maximum = np.full(dimension, -np.inf, dtype=np.float64)
    tile_rows = max(
        1,
        min(len(matrix.x), 64 * 1024**2 // max(8, 8 * dimension)),
    )
    for start in range(0, len(matrix.x), tile_rows):
        end = min(len(matrix.x), start + tile_rows)
        tile = matrix.x[start:end, columns] * scales
        minimum = np.minimum(minimum, np.min(tile, axis=0))
        maximum = np.maximum(maximum, np.max(tile, axis=0))
        event = matrix.event_weight[start:end] > 0
        if np.any(event):
            selected = tile[event]
            event_minimum = np.minimum(event_minimum, np.min(selected, axis=0))
            event_maximum = np.maximum(event_maximum, np.max(selected, axis=0))
        if not is_poisson_likelihood(likelihood):
            noevent = matrix.noevent_weight[start:end] > 0
            if np.any(noevent):
                selected = tile[noevent]
                noevent_minimum = np.minimum(noevent_minimum, np.min(selected, axis=0))
                noevent_maximum = np.maximum(noevent_maximum, np.max(selected, axis=0))
    if is_poisson_likelihood(likelihood):
        event_absolute = np.maximum(np.abs(event_minimum), np.abs(event_maximum))
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
            & ((noevent_minimum < -tolerance) | (event_maximum > tolerance))
        )
        negative = (
            (noevent_minimum >= -tolerance)
            & (event_maximum <= tolerance)
            & ((noevent_maximum > tolerance) | (event_minimum < -tolerance))
        )
    negative[int(free_dimension) :] = False
    return bool(np.any(positive | negative))


def fit_projected_model_matrix(
    matrix: ModelMatrix,
    columns: np.ndarray,
    scales: np.ndarray,
    *,
    likelihood: str,
    free_dimension: int,
    tolerance: float,
    max_iter: int,
    warm_start: np.ndarray | None = None,
    device: str = "cpu",
    _check_recession: bool = True,
    evaluator: ProjectedDesignEvaluator | None = None,
) -> FitResult:
    """Exactly fit a signed column projection while sharing the source design.

    For ``X_p = X[:, columns] diag(scales)``, every objective evaluation is
    performed on the already resident source matrix with
    ``beta_source[columns] = scales * beta_projected``.  Gradients and Hessians
    are then mapped back by the chain rule.  This is algebraically the same
    constrained Newton problem as fitting a materialized projected matrix; it
    changes storage and execution only.

    The caller must fail open to a materialized fit when this accelerator does
    not converge.  In particular, combined recession diagnosis is intentionally
    left to that fallback so this routine never constructs the projected tall
    design merely to label a failed candidate.
    """
    columns = np.ascontiguousarray(columns, dtype=np.int64)
    scales = np.ascontiguousarray(scales, dtype=np.float64)
    if columns.ndim != 1 or scales.shape != columns.shape or not len(columns):
        raise ValueError("projected columns/scales must be nonempty vectors")
    if np.any(columns < 0) or np.any(columns >= matrix.dimension):
        raise ValueError("projected column lies outside the source matrix")
    if len(np.unique(columns)) != len(columns):
        raise ValueError("projected columns must be unique")
    if not np.all(np.isfinite(scales)) or np.any(scales == 0.0):
        raise ValueError("projected scales must be finite and nonzero")
    dimension = len(columns)
    free_dimension = int(free_dimension)
    if not 0 <= free_dimension <= dimension:
        raise ValueError("invalid projected free dimension")
    if evaluator is not None and (
        evaluator.matrix is not matrix or evaluator.likelihood != likelihood
    ):
        raise ValueError("projected evaluator does not match the source problem")

    beta = np.zeros(dimension, dtype=np.float64)
    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=np.float64)
        if warm.shape != (dimension,) or not np.all(np.isfinite(warm)):
            raise ValueError("projected warm start does not match the design")
        beta[:] = warm
    if dimension > free_dimension:
        beta[free_dimension:] = np.maximum(beta[free_dimension:], 0.0)
    total_exposure = float(np.sum(matrix.exposure_weight))
    total_events = float(np.sum(matrix.event_weight))
    if warm_start is None and total_exposure > 0.0:
        beta[0] = math.log(max(total_events, 0.5) / total_exposure)

    def embed(projected_beta: np.ndarray) -> np.ndarray:
        source_beta = np.zeros(matrix.dimension, dtype=np.float64)
        source_beta[columns] = scales * projected_beta
        return source_beta

    def objective(
        projected_beta: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if evaluator is not None and len(columns) < matrix.dimension:
            projected = evaluator.projected_objective(
                projected_beta, columns, scales
            )
            if projected is not None:
                return projected
        source_beta = embed(projected_beta)
        if evaluator is None:
            nll, source_gradient, source_hessian, _ = _objective(
                matrix,
                likelihood,
                source_beta,
                device=device,
            )
        else:
            nll, source_gradient, source_hessian = evaluator.objective(source_beta)
        gradient = scales * source_gradient[columns]
        hessian = (
            source_hessian[np.ix_(columns, columns)] * scales[:, None] * scales[None, :]
        )
        return nll, gradient, hessian

    def value(projected_beta: np.ndarray) -> float:
        if evaluator is not None and len(columns) < matrix.dimension:
            projected = evaluator.projected_value(
                projected_beta, columns, scales
            )
            if projected is not None:
                return projected
        source_beta = embed(projected_beta)
        if evaluator is None:
            return _value(
                matrix,
                likelihood,
                source_beta,
                device=device,
            )
        return evaluator.value(source_beta)

    has_axis_recession = False
    if _check_recession:
        has_axis_recession = (
            evaluator.projected_axis_recession(columns, scales, free_dimension)
            if evaluator is not None
            else _projected_axis_recession(
                matrix,
                columns,
                scales,
                likelihood=likelihood,
                free_dimension=free_dimension,
            )
        )
    if has_axis_recession:
        return FitResult(
            beta,
            math.inf,
            False,
            0,
            math.inf,
            0,
            True,
            "nonattained projected coordinate recession direction",
        )

    previous = math.inf
    rank = 0
    for iteration in range(1, int(max_iter) + 1):
        nll, gradient, hessian = objective(beta)
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
                "nonfinite projected objective derivatives",
            )
        kkt = projected_kkt(beta, gradient, free_dimension)
        if kkt <= tolerance:
            rank, critical_dimension = _critical_cone_rank(
                beta,
                gradient,
                hessian,
                free_dimension=free_dimension,
                tolerance=tolerance,
            )
            if rank < critical_dimension:
                return FitResult(
                    beta,
                    nll,
                    False,
                    iteration,
                    kkt,
                    rank,
                    False,
                    "rank-deficient projected-support information",
                )
            return FitResult(
                beta,
                nll,
                True,
                iteration,
                kkt,
                rank,
                False,
                "projected view converged",
            )
        active = np.arange(dimension) < free_dimension
        active |= (beta > 1.0e-12) | (gradient < 0.0)
        indices = np.flatnonzero(active)
        if not len(indices):
            return FitResult(
                beta,
                nll,
                False,
                iteration,
                kkt,
                0,
                False,
                "empty projected Newton active set",
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
        if directional >= 0.0:
            direction = -gradient
            direction[free_dimension:] = np.where(
                (beta[free_dimension:] <= 1.0e-12) & (direction[free_dimension:] < 0.0),
                0.0,
                direction[free_dimension:],
            )
        accepted = False
        step = 1.0
        for _ in range(60):
            trial = beta + step * direction
            trial[free_dimension:] = np.maximum(trial[free_dimension:], 0.0)
            trial_nll = value(trial)
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
                beta,
                nll,
                False,
                iteration,
                kkt,
                rank,
                False,
                "projected line search failed",
            )
        if abs(previous - trial_nll) <= tolerance * max(1.0, abs(previous)):
            final_nll, final_gradient, final_hessian = objective(beta)
            final_kkt = projected_kkt(beta, final_gradient, free_dimension)
            if final_kkt <= 10.0 * tolerance:
                final_rank, critical_dimension = _critical_cone_rank(
                    beta,
                    final_gradient,
                    final_hessian,
                    free_dimension=free_dimension,
                    tolerance=10.0 * tolerance,
                )
                if final_rank < critical_dimension:
                    return FitResult(
                        beta,
                        final_nll,
                        False,
                        iteration,
                        final_kkt,
                        final_rank,
                        False,
                        "rank-deficient projected-support information",
                    )
                return FitResult(
                    beta,
                    final_nll,
                    True,
                    iteration,
                    final_kkt,
                    final_rank,
                    False,
                    "projected view converged by objective and KKT",
                )
    nll, gradient, hessian = objective(beta)
    return FitResult(
        beta,
        nll,
        False,
        int(max_iter),
        projected_kkt(beta, gradient, free_dimension),
        _checked_rank(hessian),
        False,
        "projected maximum iterations reached",
    )




def _fit_model_matrix_lbfgsb_polish(
    matrix: ModelMatrix,
    *,
    likelihood: str,
    tolerance: float,
    max_iter: int,
    warm_start: np.ndarray,
    device: str,
) -> FitResult:
    """Polish a stalled convex cone fit with a different exact solver.

    Projected Newton and L-BFGS-B optimize the identical likelihood over the
    identical free/nonnegative coefficient set.  The latter is used only when
    repeated Newton windows return a byte-identical state before satisfying
    KKT.  Its output is accepted only after an independent full objective,
    projected-KKT and critical-cone rank check, so this is a numerical
    fail-open rather than a relaxed convergence rule.
    """

    start = np.ascontiguousarray(warm_start, dtype=np.float64).copy()
    if start.shape != (matrix.dimension,) or not np.all(np.isfinite(start)):
        raise ValueError("L-BFGS-B polish warm start does not match the model")
    start[matrix.free_dimension :] = np.maximum(
        start[matrix.free_dimension :], 0.0
    )
    bounds = [
        (None, None) if index < matrix.free_dimension else (0.0, None)
        for index in range(matrix.dimension)
    ]

    def value_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        nll, gradient, _, _ = _objective(
            matrix,
            likelihood,
            np.ascontiguousarray(beta, dtype=np.float64),
            device=device,
        )
        return float(nll), np.ascontiguousarray(gradient, dtype=np.float64)

    result = minimize(
        value_gradient,
        start,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "ftol": np.finfo(np.float64).eps,
            "gtol": float(tolerance),
            "maxiter": max(200, 4 * int(max_iter)),
            "maxls": 100,
            "maxcor": 20,
        },
    )
    beta = np.ascontiguousarray(result.x, dtype=np.float64)
    beta[matrix.free_dimension :] = np.maximum(
        beta[matrix.free_dimension :], 0.0
    )
    nll, gradient, hessian, _ = _objective(
        matrix, likelihood, beta, device=device
    )
    kkt = projected_kkt(beta, gradient, matrix.free_dimension)
    rank, critical_dimension = _critical_cone_rank(
        beta,
        gradient,
        hessian,
        free_dimension=matrix.free_dimension,
        tolerance=10.0 * tolerance,
    )
    converged = bool(
        math.isfinite(nll)
        and np.all(np.isfinite(beta))
        and kkt <= 10.0 * tolerance
        and rank >= critical_dimension
    )
    return FitResult(
        beta,
        float(nll),
        converged,
        int(getattr(result, "nit", 0)),
        float(kkt),
        int(rank),
        False,
        (
            "converged by exact L-BFGS-B KKT polish"
            if converged
            else "L-BFGS-B polish did not satisfy exact KKT/rank: "
            f"{result.message}"
        ),
    )


def fit_model_matrix_continued(
    matrix: ModelMatrix,
    *,
    likelihood: str,
    tolerance: float,
    max_iter: int,
    warm_start: np.ndarray | None = None,
    device: str = "cpu",
    _check_recession: bool = True,
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
    polish_attempted = False
    recession_checked = not bool(_check_recession)
    while True:
        fit = fit_model_matrix(
            matrix,
            likelihood=likelihood,
            tolerance=tolerance,
            max_iter=max_iter,
            warm_start=start,
            device=device,
            _check_recession=not recession_checked,
        )
        recession_checked = True
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
                if not polish_attempted:
                    polish_attempted = True
                    polished = _fit_model_matrix_lbfgsb_polish(
                        matrix,
                        likelihood=likelihood,
                        tolerance=tolerance,
                        max_iter=max_iter,
                        warm_start=fit.coefficients,
                        device=device,
                    )
                    total_iterations += polished.iterations
                    if polished.converged:
                        return replace(polished, iterations=total_iterations)
                    # A finite lower point can move Newton off its stalled
                    # face. Continue from it once; otherwise report the more
                    # informative polished KKT failure without cycling.
                    slack = 128.0 * np.finfo(np.float64).eps * max(
                        1.0, abs(fit.nll), abs(polished.nll)
                    )
                    if (
                        math.isfinite(polished.nll)
                        and polished.nll <= fit.nll + slack
                        and not np.array_equal(
                            polished.coefficients, fit.coefficients
                        )
                    ):
                        start = polished.coefficients
                        previous = None
                        repeated = 0
                        continue
                    return replace(
                        polished,
                        iterations=total_iterations,
                        message=(
                            "continued Newton stagnation; exact L-BFGS-B "
                            f"polish failed: {polished.message}"
                        ),
                    )
                return replace(
                    fit,
                    iterations=total_iterations,
                    message=(
                        "continued Newton stagnation after exact L-BFGS-B polish"
                    ),
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
    if (
        any(
            value.shape != (rows,)
            for value in (offset, exposure_weight, noevent_weight, event_weight)
        )
        or not 0 <= free_dimension <= dimension
    ):
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
            if is_poisson_likelihood(likelihood):
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
    matrix_token = new_derivative_token()

    def linear_predictor(value: np.ndarray) -> np.ndarray:
        product = resident_eta(
            x,
            value,
            device=device,
            matrix_token=matrix_token,
        )
        if product is None:
            product = x @ value
        return offset + product

    for iteration in range(1, int(max_iter) + 1):
        eta = linear_predictor(beta)
        values, first, second = loss_rows(
            eta,
            likelihood=likelihood,
            exposure_weight=exposure_weight,
            noevent_weight=noevent_weight,
            event_weight=event_weight,
        )
        nll = float(np.sum(values))
        gradient, hessian = moments(
            x,
            first,
            second,
            device=device,
            matrix_token=matrix_token,
        )
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
            rank, critical_dimension = _critical_cone_rank(
                beta,
                gradient,
                hessian,
                free_dimension=free_dimension,
                tolerance=tolerance,
            )
            converged = rank == critical_dimension
            return FitResult(
                beta,
                nll,
                converged,
                iteration,
                kkt,
                rank,
                False,
                "converged" if converged else "rank-deficient offset block",
            )
        active = np.arange(dimension) < free_dimension
        active |= (beta > 1.0e-12) | (gradient < 0.0)
        indices = np.flatnonzero(active)
        sub_hessian = hessian[np.ix_(indices, indices)]
        rank = _checked_rank(sub_hessian)
        try:
            step_active = np.linalg.solve(sub_hessian, -gradient[indices])
        except np.linalg.LinAlgError:
            step_active = np.linalg.lstsq(sub_hessian, -gradient[indices], rcond=None)[
                0
            ]
        direction = np.zeros(dimension, dtype=np.float64)
        direction[indices] = step_active
        if float(gradient @ direction) >= 0:
            direction = -gradient
            direction[free_dimension:] = np.where(
                (beta[free_dimension:] <= 1.0e-12) & (direction[free_dimension:] < 0),
                0.0,
                direction[free_dimension:],
            )
        accepted = False
        for _ in range(60):
            trial = beta + direction
            trial[free_dimension:] = np.maximum(trial[free_dimension:], 0.0)
            trial_values = loss_value_rows(
                linear_predictor(trial),
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
    eta = linear_predictor(beta)
    values, first, second = loss_rows(
        eta,
        likelihood=likelihood,
        exposure_weight=exposure_weight,
        noevent_weight=noevent_weight,
        event_weight=event_weight,
    )
    gradient, hessian = moments(
        x,
        first,
        second,
        device=device,
        matrix_token=matrix_token,
    )
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
    # CUDA devices are physical execution slots.  Oversubscribing cuda:0 or
    # padding a two-GPU wave with a CPU straggler cannot improve latency and
    # used to make the deterministic map wait for its slowest job.  CPU-only
    # execution retains the requested concurrency.
    physical_workers = (
        min(int(workers), len(devices))
        if any(device.startswith("cuda") for device in devices)
        else int(workers)
    )

    def solve(
        item: tuple[int, ModelMatrix, np.ndarray | None],
        *,
        device: str | None = None,
    ) -> FitResult:
        configure_cpu_threads(max(1, int(cpu_threads_per_worker)))
        index, matrix, warm = item
        return fit_model_matrix_continued(
            matrix,
            likelihood=likelihood,
            tolerance=tolerance,
            max_iter=max_iter,
            warm_start=warm,
            device=devices[index % len(devices)] if device is None else device,
        )

    jobs = [
        (index, matrix, warm)
        for index, (matrix, warm) in enumerate(zip(ordered, starts, strict=True))
    ]
    if len(jobs) <= 1 or physical_workers <= 1:
        return [solve(job) for job in jobs]
    cuda_devices = tuple(device for device in devices if device.startswith("cuda"))
    if cuda_devices:
        # A generic executor assigns the next indexed job to whichever thread
        # finishes first, while ``index % devices`` can point that job back to
        # a GPU which is still busy.  Keep one sequential consumer per
        # physical GPU and restore input order.  This removes accidental
        # device oversubscription and the corresponding OOM/reupload stalls
        # without changing any fitted model.
        output: list[FitResult | None] = [None] * len(jobs)
        next_index = 0
        queue_lock = threading.Lock()

        def consume(device: str) -> None:
            nonlocal next_index
            while True:
                with queue_lock:
                    if next_index >= len(jobs):
                        return
                    index = next_index
                    next_index += 1
                output[index] = solve(jobs[index], device=device)

        with ThreadPoolExecutor(
            max_workers=min(physical_workers, len(cuda_devices)),
            thread_name_prefix="crbstpp-newton",
        ) as executor:
            list(executor.map(consume, cuda_devices[:physical_workers]))
        if any(item is None for item in output):
            raise AssertionError("dynamic matrix-fit queue lost a result")
        return [item for item in output if item is not None]
    with ThreadPoolExecutor(
        max_workers=min(physical_workers, len(jobs)),
        thread_name_prefix="crbstpp-newton",
    ) as executor:
        return list(executor.map(solve, jobs))
