from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .data import QueryContext
from .model import (
    OCCURRENCE_LIKELIHOODS,
    cloglog_event_nll,
    cluster_exposure,
    cluster_nll,
)
from .statistics import equivalence_mean_test, one_sided_mean_test


@dataclass(frozen=True)
class EnsembleFit:
    weights: np.ndarray
    nll: float
    converged: bool
    iterations: int
    projected_residual: float
    device: str


def _simplex_project(values: object) -> object:
    ordered, _ = torch.sort(values, descending=True)
    cssv = torch.cumsum(ordered, dim=0) - 1.0
    indices = torch.arange(1, len(values) + 1, dtype=values.dtype, device=values.device)
    condition = ordered - cssv / indices > 0
    rho = int(torch.nonzero(condition, as_tuple=False).reshape(-1)[-1].item())
    threshold = cssv[rho] / float(rho + 1)
    return torch.clamp(values - threshold, min=0.0)


def fit_intensity_ensemble(
    component_eta: Sequence[np.ndarray],
    ctx: QueryContext,
    *,
    component_event_log_density: Sequence[np.ndarray] | None = None,
    component_grid_integrals: np.ndarray | None = None,
    device: str = "cuda",
    max_iter: int = 500,
    tolerance: float = 1.0e-7,
    cluster_weights: np.ndarray | None = None,
    occurrence_likelihood: str = "poisson",
) -> EnsembleFit:
    if not component_eta:
        raise ValueError("ensemble needs at least one component")
    if torch is None:
        raise RuntimeError("PyTorch is required")
    if occurrence_likelihood not in OCCURRENCE_LIKELIHOODS:
        raise ValueError("unknown occurrence likelihood")
    if (
        occurrence_likelihood == "first_event_cloglog"
        and component_event_log_density is not None
    ):
        raise ValueError("marked densities are not supported by first-event cloglog")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    eta_values = [np.asarray(eta, dtype=np.float64) for eta in component_eta]
    expected_rows = ctx.n_events if component_grid_integrals is not None else ctx.n_queries
    if any(values.shape != (expected_rows,) for values in eta_values):
        raise ValueError("component query/event length mismatch")
    if cluster_weights is None:
        cluster_weights = np.ones(ctx.n_sequences, dtype=np.float64)
    cluster_weights = np.asarray(cluster_weights, dtype=np.float64)
    if (
        cluster_weights.shape != (ctx.n_sequences,)
        or np.any(~np.isfinite(cluster_weights))
        or np.any(cluster_weights < 0)
        or not np.any(cluster_weights > 0)
    ):
        raise ValueError("ensemble cluster weights must be a finite nonnegative sequence vector")
    event_weights = torch.as_tensor(
        cluster_weights[ctx.event_sequence_local], dtype=torch.float64, device=device
    )
    event_log_component = np.stack(
        [values[: ctx.n_events] for values in eta_values],
        axis=1,
    )
    if component_event_log_density is not None:
        log_density = np.stack(
            [np.asarray(values, dtype=np.float64) for values in component_event_log_density],
            axis=1,
        )
        if log_density.shape != event_log_component.shape:
            raise ValueError("component mark-density/event-intensity dimensions differ")
        # The event density of a marked component is lambda_k(t) f_k(m|t).
        # The integrated compensator remains integral lambda_k because every
        # conditional mark density integrates to one.
        event_log_component += log_density
    if ctx.n_events:
        event_log_scale_np = np.max(event_log_component, axis=1)
        if np.any(~np.isfinite(event_log_scale_np)):
            raise FloatingPointError("nonfinite ensemble event log density")
        event_scaled_np = np.exp(event_log_component - event_log_scale_np[:, None])
    else:
        event_log_scale_np = np.zeros(0, dtype=np.float64)
        event_scaled_np = np.zeros((0, len(eta_values)), dtype=np.float64)
    event_intensity = torch.as_tensor(event_scaled_np, dtype=torch.float64, device=device)
    event_log_scale = torch.as_tensor(event_log_scale_np, dtype=torch.float64, device=device)
    event_hazard_np = np.exp(event_log_component)
    event_hazard = torch.as_tensor(event_hazard_np, dtype=torch.float64, device=device)

    if component_grid_integrals is None:
        grid_integrals_np = np.empty(len(eta_values), dtype=np.float64)
        for index, values in enumerate(eta_values):
            grid_eta = values[ctx.n_events :]
            with np.errstate(over="ignore", invalid="ignore"):
                intensity = np.exp(grid_eta)
            if np.any(~np.isfinite(intensity)):
                raise FloatingPointError("ensemble component grid intensity overflow")
            grid_integrals_np[index] = float(
                np.dot(cluster_weights, ctx.aggregate_weighted_grid(intensity))
            )
    else:
        grid_integrals_np = np.asarray(component_grid_integrals, dtype=np.float64)
        if (
            grid_integrals_np.shape != (len(eta_values),)
            or np.any(~np.isfinite(grid_integrals_np))
            or np.any(grid_integrals_np < 0)
        ):
            raise ValueError("component grid integrals must be finite nonnegative scalars")
    if occurrence_likelihood == "first_event_cloglog" and ctx.n_events:
        # Supplied component integrals cover all reporting cells. Replace each
        # terminal event cell's survival contribution by its event probability.
        grid_integrals_np = grid_integrals_np - (
            cluster_weights[ctx.event_sequence_local, None] * event_hazard_np
        ).sum(axis=0)
        roundoff = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(grid_integrals_np)) * 64.0
        grid_integrals_np = np.where(
            (grid_integrals_np < 0.0) & (grid_integrals_np >= -roundoff),
            0.0,
            grid_integrals_np,
        )
        if np.any(grid_integrals_np < 0.0):
            raise ValueError("event-cell hazard exceeds component grid integral")
    grid_integrals = torch.as_tensor(grid_integrals_np, dtype=torch.float64, device=device)
    k = len(eta_values)
    weights = torch.full((k,), 1.0 / float(k), dtype=torch.float64, device=device)

    def value_grad_hessian(w: object) -> tuple[object, object, object]:
        if occurrence_likelihood == "first_event_cloglog":
            mixture_hazard = torch.clamp(
                event_hazard.matmul(w), min=torch.finfo(torch.float64).tiny
            )
            denominator = torch.expm1(mixture_hazard)
            event_loss = -torch.log(-torch.expm1(-mixture_hazard))
            first = -1.0 / denominator
            second = torch.exp(mixture_hazard) / (denominator * denominator)
            value = torch.dot(event_weights, event_loss) + torch.dot(grid_integrals, w)
            grad = event_hazard.T.matmul(event_weights * first) + grid_integrals
            hessian = event_hazard.T.matmul(
                event_hazard * (event_weights * second).reshape(-1, 1)
            )
            return value, grad, hessian
        mixture = torch.clamp(
            event_intensity.matmul(w), min=torch.finfo(torch.float64).tiny
        )
        value = -torch.dot(
            event_weights,
            torch.log(mixture) + event_log_scale,
        ) + torch.dot(grid_integrals, w)
        ratios = event_intensity / mixture.reshape(-1, 1)
        grad = -torch.sum(event_weights.reshape(-1, 1) * ratios, dim=0) + grid_integrals
        hessian = ratios.T.matmul(event_weights.reshape(-1, 1) * ratios)
        return value, grad, hessian

    def value_only(w: object) -> object:
        if occurrence_likelihood == "first_event_cloglog":
            mixture_hazard = torch.clamp(
                event_hazard.matmul(w), min=torch.finfo(torch.float64).tiny
            )
            return torch.dot(
                event_weights,
                -torch.log(-torch.expm1(-mixture_hazard)),
            ) + torch.dot(grid_integrals, w)
        mixture = torch.clamp(
            event_intensity.matmul(w), min=torch.finfo(torch.float64).tiny
        )
        return -torch.dot(
            event_weights,
            torch.log(mixture) + event_log_scale,
        ) + torch.dot(grid_integrals, w)

    converged = False
    residual = math.inf
    iterations = 0
    final_state: tuple[object, object, object] | None = None
    for iterations in range(1, max_iter + 1):
        value, grad, hessian = value_grad_hessian(weights)
        final_state = (value, grad, hessian)
        residual_step = 1.0 / max(float(torch.linalg.norm(grad).item()), 1.0)
        residual = float(
            torch.max(
                torch.abs(
                    _simplex_project(weights - residual_step * grad) - weights
                )
            ).item()
        ) / residual_step
        target_residual = tolerance * max(
            1.0, abs(float(value.item())) / max(1, ctx.n_queries)
        )
        if residual <= target_residual:
            converged = True
            break

        # Equality-constrained Newton on the current simplex face.  Pure
        # projected gradient is globally valid but converges extremely slowly
        # when several support models are identical or nearly collinear.  The
        # tangent-space pseudoinverse below preserves the same convex optimum
        # and handles such singular component families without a ridge.
        w_np = weights.detach().cpu().numpy()
        g_np = grad.detach().cpu().numpy()
        h_np = hessian.detach().cpu().numpy()
        boundary = 64.0 * np.finfo(np.float64).eps * max(1, k)
        active = w_np > boundary
        if not np.any(active):
            active[int(np.argmax(w_np))] = True
        active_before_entry = active.copy()
        active_gradient = float(np.mean(g_np[active]))
        inactive = np.flatnonzero(~active)
        entered: int | None = None
        if len(inactive):
            candidate = int(inactive[np.argmin(g_np[inactive])])
            if g_np[candidate] < active_gradient:
                active[candidate] = True
                entered = candidate
        active_indices = np.flatnonzero(active)
        direction_np = np.zeros(k, dtype=np.float64)
        if len(active_indices) >= 2:
            active_count = len(active_indices)
            tangent = np.zeros((active_count, active_count - 1), dtype=np.float64)
            tangent[np.arange(active_count - 1), np.arange(active_count - 1)] = 1.0
            tangent[-1, :] = -1.0
            tangent_hessian = tangent.T @ h_np[np.ix_(active_indices, active_indices)] @ tangent
            tangent_gradient = tangent.T @ g_np[active_indices]
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(
                    0.5 * (tangent_hessian + tangent_hessian.T)
                )
                largest = max(
                    float(np.max(eigenvalues, initial=0.0)),
                    np.finfo(np.float64).tiny,
                )
                cutoff = np.finfo(np.float64).eps * max(1, active_count) * largest
                inverse = np.divide(
                    1.0,
                    eigenvalues,
                    out=np.zeros_like(eigenvalues),
                    where=eigenvalues > cutoff,
                )
                tangent_step = -eigenvectors @ (
                    inverse * (eigenvectors.T @ tangent_gradient)
                )
                direction_np[active_indices] = tangent @ tangent_step
            except np.linalg.LinAlgError:
                direction_np[:] = 0.0

        descent = float(g_np @ direction_np)
        infeasible_entry = bool(
            entered is not None and direction_np[entered] <= 0.0
        )
        zero_points_outward = bool(
            np.any((w_np <= boundary) & (direction_np < 0.0))
        )
        if (
            not math.isfinite(descent)
            or descent >= 0.0
            or infeasible_entry
            or zero_points_outward
        ):
            # Exact feasible pair direction for admitting the most violated
            # inactive component.  It is a descent direction precisely when
            # that component's reduced gradient violates simplex KKT.
            direction_np[:] = 0.0
            donors = np.flatnonzero(active_before_entry)
            receivers = np.flatnonzero(~active_before_entry)
            if len(donors) and len(receivers):
                receiver = int(receivers[np.argmin(g_np[receivers])])
                donor = int(donors[np.argmax(g_np[donors])])
                if g_np[receiver] < g_np[donor]:
                    direction_np[receiver] = 1.0
                    direction_np[donor] = -1.0
            descent = float(g_np @ direction_np)

        accepted = False
        if math.isfinite(descent) and descent < 0.0 and np.any(direction_np):
            negative = direction_np < 0.0
            max_step = float(
                np.min(-w_np[negative] / direction_np[negative])
            ) if np.any(negative) else 1.0
            step = min(1.0, max_step)
            direction = torch.as_tensor(
                direction_np, dtype=torch.float64, device=device
            )
            current_value = float(value.item())
            for _ in range(40):
                proposal = weights + step * direction
                proposal = torch.clamp(proposal, min=0.0)
                proposal = proposal / torch.sum(proposal)
                delta = proposal - weights
                trial_value = value_only(proposal)
                if (
                    bool(torch.isfinite(trial_value).item())
                    and float(trial_value.item())
                    <= current_value + 1.0e-4 * float(torch.dot(grad, delta).item())
                ):
                    weights = proposal
                    final_state = None
                    accepted = True
                    break
                step *= 0.5
        if accepted:
            continue

        # Globally convergent projected-gradient fallback.  This also covers a
        # one-vertex face whose only improving move has not yet entered due to
        # roundoff in the active-set comparison.
        eigen_max = (
            float(torch.linalg.eigvalsh(hessian)[-1].item())
            if k > 1
            else float(hessian[0, 0].item())
        )
        step = 1.0 / max(eigen_max, 1.0e-12)
        accepted = False
        for _ in range(30):
            proposal = _simplex_project(weights - step * grad)
            trial_value = value_only(proposal)
            delta = proposal - weights
            upper = value + torch.dot(grad, delta) + torch.dot(delta, delta) / (2.0 * step)
            if float(trial_value.item()) <= float(upper.item()) + 1.0e-12:
                weights = proposal
                final_state = None
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    if final_state is None:
        value, grad, _hessian = value_grad_hessian(weights)
    else:
        value, grad, _hessian = final_state
    step = 1.0 / max(float(torch.linalg.norm(grad).item()), 1.0)
    residual = float(torch.max(torch.abs(_simplex_project(weights - step * grad) - weights)).item()) / step
    converged = bool(
        residual
        <= tolerance
        * max(1.0, abs(float(value.item())) / max(1, ctx.n_queries))
    )
    return EnsembleFit(
        weights=weights.detach().cpu().numpy(),
        nll=float(value.detach().cpu().item()),
        converged=bool(converged),
        iterations=int(iterations),
        projected_residual=float(residual),
        device=device,
    )


def mixture_eta(component_eta: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    values = [np.asarray(eta, dtype=np.float64) for eta in component_eta]
    coefficients = np.asarray(weights, dtype=np.float64)
    if not values or coefficients.shape != (len(values),):
        raise ValueError("mixture component/weight mismatch")
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("mixture component shapes differ")
    if any(np.any(~np.isfinite(value)) for value in values):
        raise ValueError("mixture component log intensities must be finite")
    if np.any(~np.isfinite(coefficients)) or np.any(coefficients < 0) or not np.any(coefficients > 0):
        raise ValueError("mixture weights must be finite, nonnegative, and nonzero")
    if not math.isclose(float(np.sum(coefficients)), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("mixture weights must sum to one")
    maximum = np.maximum.reduce(
        [value for value, coefficient in zip(values, coefficients, strict=True) if coefficient > 0]
    )
    scaled = np.zeros_like(maximum)
    for value, coefficient in zip(values, coefficients, strict=True):
        if coefficient > 0:
            scaled += coefficient * np.exp(value - maximum)
    if np.any(~np.isfinite(scaled)) or np.any(scaled <= 0.0):
        raise FloatingPointError("nonpositive scaled mixture intensity")
    return maximum + np.log(scaled)


def evaluate_ensemble(
    ensemble_eta: np.ndarray,
    baseline_eta: np.ndarray,
    ctx: QueryContext,
    *,
    contribution_threshold: float = 0.0,
    calibration_tolerance: float | None = None,
    alpha: float = 0.05,
    cluster_weights: np.ndarray | None = None,
    occurrence_likelihood: str = "poisson",
) -> dict:
    ensemble_eta = np.asarray(ensemble_eta, dtype=np.float64)
    baseline_eta = np.asarray(baseline_eta, dtype=np.float64)
    if (
        ensemble_eta.shape != (ctx.n_queries,)
        or baseline_eta.shape != (ctx.n_queries,)
        or np.any(~np.isfinite(ensemble_eta))
        or np.any(~np.isfinite(baseline_eta))
    ):
        raise ValueError("ensemble evaluation requires finite query-aligned log intensities")

    def losses(eta: np.ndarray) -> np.ndarray:
        values = cluster_nll(
            eta,
            ctx,
            occurrence_likelihood=occurrence_likelihood,
        )
        if cluster_weights is None:
            return values
        weights = np.asarray(cluster_weights, dtype=np.float64)
        if weights.shape != (ctx.n_sequences,) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("evaluation cluster weights must be a finite nonnegative sequence vector")
        return weights * values

    contribution = one_sided_mean_test(
        losses(baseline_eta) - losses(ensemble_eta),
        null=contribution_threshold,
        alpha=alpha,
    )
    exposure = cluster_exposure(ctx)
    expected = ctx.aggregate_weighted_grid(np.exp(ensemble_eta[ctx.n_events :]))
    observed = np.bincount(ctx.event_sequence_local, minlength=ctx.n_sequences).astype(np.float64)
    calibration_error = np.divide(observed - expected, exposure, out=np.zeros_like(expected), where=exposure > 0)
    if occurrence_likelihood == "first_event_cloglog":
        if calibration_tolerance is not None:
            raise ValueError(
                "hazard intensity is not an entity-level probability calibration target"
            )
        calibration = {
            "gated": False,
            "estimate": None,
            "note": "probability calibration requires per-cell survival probabilities",
        }
    elif calibration_tolerance is None:
        calibration = {"gated": False, "estimate": float(np.mean(calibration_error))}
    else:
        calibration = equivalence_mean_test(calibration_error, tolerance=calibration_tolerance, alpha=alpha)
        calibration["gated"] = True
    return {
        "n_sequences": ctx.n_sequences,
        "n_events": ctx.n_events,
        "contribution": {
            "estimate": contribution.estimate,
            "standard_error": contribution.standard_error,
            "p_value": contribution.p_value,
            "lower_bound": contribution.lower_bound,
        },
        "calibration": calibration,
    }


def evaluate_ensemble_sufficient(
    ensemble_event_eta: np.ndarray,
    baseline_event_eta: np.ndarray,
    ensemble_cluster_intensity: np.ndarray,
    baseline_cluster_intensity: np.ndarray,
    ctx: QueryContext,
    *,
    contribution_threshold: float = 0.0,
    calibration_tolerance: float | None = None,
    alpha: float = 0.05,
    cluster_weights: np.ndarray | None = None,
    occurrence_likelihood: str = "poisson",
) -> dict:
    """Exact ensemble evaluation from event and compensator sufficient statistics."""
    ensemble_event = np.asarray(ensemble_event_eta, dtype=np.float64)
    baseline_event = np.asarray(baseline_event_eta, dtype=np.float64)
    ensemble_intensity = np.asarray(ensemble_cluster_intensity, dtype=np.float64)
    baseline_intensity = np.asarray(baseline_cluster_intensity, dtype=np.float64)
    if (
        ensemble_event.shape != (ctx.n_events,)
        or baseline_event.shape != (ctx.n_events,)
        or ensemble_intensity.shape != (ctx.n_sequences,)
        or baseline_intensity.shape != (ctx.n_sequences,)
        or np.any(~np.isfinite(ensemble_event))
        or np.any(~np.isfinite(baseline_event))
        or np.any(~np.isfinite(ensemble_intensity))
        or np.any(~np.isfinite(baseline_intensity))
    ):
        raise ValueError("ensemble sufficient statistics must be finite and aligned")

    def losses(event_eta: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        if occurrence_likelihood == "poisson":
            event_values = -event_eta
            noevent_intensity = intensity
        elif occurrence_likelihood == "first_event_cloglog":
            event_values = cloglog_event_nll(event_eta)
            noevent_intensity = intensity - np.bincount(
                ctx.event_sequence_local,
                weights=np.exp(event_eta),
                minlength=ctx.n_sequences,
            )
        else:
            raise ValueError("unknown occurrence likelihood")
        event = np.bincount(
            ctx.event_sequence_local,
            weights=event_values,
            minlength=ctx.n_sequences,
        )
        values = event + noevent_intensity
        if cluster_weights is None:
            return values
        weights = np.asarray(cluster_weights, dtype=np.float64)
        if (
            weights.shape != (ctx.n_sequences,)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0)
        ):
            raise ValueError(
                "evaluation cluster weights must be a finite nonnegative sequence vector"
            )
        return weights * values

    contribution = one_sided_mean_test(
        losses(baseline_event, baseline_intensity)
        - losses(ensemble_event, ensemble_intensity),
        null=contribution_threshold,
        alpha=alpha,
    )
    observed = np.bincount(
        ctx.event_sequence_local, minlength=ctx.n_sequences
    ).astype(np.float64)
    exposure = ctx.sequence_exposures()
    calibration_error = np.divide(
        observed - ensemble_intensity,
        exposure,
        out=np.zeros_like(ensemble_intensity),
        where=exposure > 0,
    )
    if occurrence_likelihood == "first_event_cloglog":
        if calibration_tolerance is not None:
            raise ValueError(
                "hazard intensity is not an entity-level probability calibration target"
            )
        calibration = {
            "gated": False,
            "estimate": None,
            "note": "probability calibration requires per-cell survival probabilities",
        }
    elif calibration_tolerance is None:
        calibration = {"gated": False, "estimate": float(np.mean(calibration_error))}
    else:
        calibration = equivalence_mean_test(
            calibration_error,
            tolerance=calibration_tolerance,
            alpha=alpha,
        )
        calibration["gated"] = True
    return {
        "n_sequences": ctx.n_sequences,
        "n_events": ctx.n_events,
        "contribution": {
            "estimate": contribution.estimate,
            "standard_error": contribution.standard_error,
            "p_value": contribution.p_value,
            "lower_bound": contribution.lower_bound,
        },
        "calibration": calibration,
    }
