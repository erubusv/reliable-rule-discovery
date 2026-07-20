from __future__ import annotations

import math

import numpy as np


def cloglog_event_terms(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = np.asarray(eta, dtype=np.float64)
    x = np.exp(np.clip(eta, -745.0, 700.0))
    value = np.empty_like(x)
    first = np.empty_like(x)
    second = np.empty_like(x)
    small = x < 1.0e-4
    large = x > 40.0
    middle = ~(small | large)
    xs = x[small]
    value[small] = (
        -np.log(np.maximum(xs, np.finfo(float).tiny)) + xs / 2.0 - xs * xs / 24.0
    )
    first[small] = -1.0 + xs / 2.0 - xs * xs / 12.0
    second[small] = xs / 2.0 - xs * xs / 6.0
    xm = x[middle]
    denominator = np.expm1(xm)
    value[middle] = -np.log(-np.expm1(-xm))
    first[middle] = -xm / denominator
    exponential = denominator + 1.0
    second[middle] = xm * ((xm - 1.0) * exponential + 1.0) / (denominator * denominator)
    xl = x[large]
    tail = np.exp(-xl)
    value[large] = -np.log1p(-tail)
    first[large] = -xl * tail / np.maximum(1.0 - tail, np.finfo(float).tiny)
    safe_large = xl < 100.0
    second_large = np.zeros_like(xl)
    second_large[safe_large] = (
        xl[safe_large] * (xl[safe_large] - 1.0) * tail[safe_large]
    )
    second[large] = second_large
    return value, first, np.maximum(second, 0.0)


def loss_rows(
    eta: np.ndarray,
    *,
    likelihood: str,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = np.asarray(eta, dtype=np.float64)
    intensity = np.exp(np.clip(eta, -745.0, 700.0))
    if likelihood == "poisson":
        value = exposure_weight * intensity - event_weight * eta
        first = exposure_weight * intensity - event_weight
        second = exposure_weight * intensity
        return value, first, second
    if likelihood != "first_event_cloglog":
        raise ValueError(f"unknown likelihood: {likelihood}")
    event_value, event_first, event_second = cloglog_event_terms(eta)
    return (
        noevent_weight * intensity + event_weight * event_value,
        noevent_weight * intensity + event_weight * event_first,
        noevent_weight * intensity + event_weight * event_second,
    )


def loss_grid_sparse_event_derivatives(
    eta: np.ndarray,
    *,
    likelihood: str,
    exposure: float,
    event_rows: np.ndarray,
    event_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact grid derivatives using only two dense output buffers."""
    eta = np.asarray(eta, dtype=np.float64)
    rows = np.asarray(event_rows, dtype=np.int64)
    counts = np.asarray(event_counts, dtype=np.float64)
    if rows.shape != counts.shape or (
        len(rows) and (rows[0] < 0 or rows[-1] >= len(eta))
    ):
        raise ValueError("sparse event rows do not match the predictor grid")
    first = float(exposure) * np.exp(np.clip(eta, -745.0, 700.0))
    second = first.copy()
    if not len(rows):
        return first, second
    if likelihood == "poisson":
        first[rows] -= counts
        return first, second
    if likelihood != "first_event_cloglog":
        raise ValueError(f"unknown likelihood: {likelihood}")
    _, event_first, event_second = cloglog_event_terms(eta[rows])
    noevent_value = np.exp(np.clip(eta[rows], -745.0, 700.0))
    first[rows] += counts * (event_first - noevent_value)
    second[rows] += counts * (event_second - noevent_value)
    return first, second


def poisson_conjugate(
    dual: np.ndarray, exposure_weight: np.ndarray, event_weight: np.ndarray
) -> np.ndarray:
    dual = np.asarray(dual, dtype=np.float64)
    mass = dual + event_weight
    if np.any(mass < 0) or np.any(exposure_weight <= 0):
        return np.full_like(dual, np.inf)
    output = np.zeros_like(dual)
    positive = mass > 0
    output[positive] = mass[positive] * (
        np.log(mass[positive] / exposure_weight[positive]) - 1.0
    )
    return output


def _mixed_cloglog_value_gradient(
    eta: float, noevent: float, event: float
) -> tuple[float, float]:
    values, first, _ = cloglog_event_terms(np.asarray([eta]))
    intensity = math.exp(min(eta, 700.0))
    return noevent * intensity + event * float(
        values[0]
    ), noevent * intensity + event * float(first[0])


def cloglog_conjugate(
    dual: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
) -> np.ndarray:
    """Conjugate of aggregated Bernoulli-cloglog row losses.

    CRBS matrices contain event-only active rows and aggregated no-event rows.
    Those cases are handled vectorially; the general mixed-row fallback is
    retained as a checked reference path.
    """
    dual = np.asarray(dual, dtype=np.float64)
    noevent_weight = np.asarray(noevent_weight, dtype=np.float64)
    event_weight = np.asarray(event_weight, dtype=np.float64)
    output = np.full_like(dual, np.inf)
    noevent_only = (noevent_weight > 0) & (event_weight == 0)
    if np.any(noevent_only):
        u = dual[noevent_only]
        exposure = noevent_weight[noevent_only]
        valid = u >= 0
        values = np.full_like(u, np.inf)
        positive = valid & (u > 0)
        values[valid & ~positive] = 0.0
        values[positive] = u[positive] * (
            np.log(u[positive] / exposure[positive]) - 1.0
        )
        output[noevent_only] = values
    event_only = (event_weight > 0) & (noevent_weight == 0)
    if np.any(event_only):
        u = dual[event_only]
        event = event_weight[event_only]
        valid = (u >= -event) & (u <= 0)
        values = np.full_like(u, np.inf)
        boundary = valid & ((np.abs(u + event) <= 1.0e-14) | (np.abs(u) <= 1.0e-14))
        values[boundary] = 0.0
        interior = valid & ~boundary
        if np.any(interior):
            target = u[interior] / event[interior]
            low = np.full_like(target, -745.0)
            high = np.full_like(target, math.log(700.0))
            for _ in range(64):
                middle = 0.5 * (low + high)
                _, derivative, _ = cloglog_event_terms(middle)
                move_low = derivative < target
                low = np.where(move_low, middle, low)
                high = np.where(move_low, high, middle)
            eta = 0.5 * (low + high)
            event_value, _, _ = cloglog_event_terms(eta)
            values[interior] = u[interior] * eta - event[interior] * event_value
        output[event_only] = values
    empty = (noevent_weight == 0) & (event_weight == 0)
    output[empty & (dual == 0)] = 0.0
    mixed = (noevent_weight > 0) & (event_weight > 0)
    for index in np.flatnonzero(mixed):
        u = float(dual[index])
        noevent = float(noevent_weight[index])
        event = float(event_weight[index])
        lower_domain = -event
        upper_domain = math.inf if noevent > 0 else 0.0
        if u < lower_domain or u > upper_domain:
            output[index] = math.inf
            continue
        if abs(u - lower_domain) <= 1.0e-14 or (
            math.isfinite(upper_domain) and abs(u - upper_domain) <= 1.0e-14
        ):
            output[index] = 0.0
            continue
        low, high = -50.0, 50.0
        _, low_gradient = _mixed_cloglog_value_gradient(low, noevent, event)
        _, high_gradient = _mixed_cloglog_value_gradient(high, noevent, event)
        while low_gradient > u and low > -740.0:
            low *= 2.0
            _, low_gradient = _mixed_cloglog_value_gradient(low, noevent, event)
        while high_gradient < u and high < 700.0:
            high *= 2.0
            _, high_gradient = _mixed_cloglog_value_gradient(high, noevent, event)
        if not low_gradient <= u <= high_gradient:
            output[index] = math.inf
            continue
        for _ in range(100):
            middle = 0.5 * (low + high)
            _, gradient = _mixed_cloglog_value_gradient(middle, noevent, event)
            if gradient < u:
                low = middle
            else:
                high = middle
        eta = 0.5 * (low + high)
        value, _ = _mixed_cloglog_value_gradient(eta, noevent, event)
        output[index] = u * eta - value
    return output


def conjugate_sum(
    dual: np.ndarray,
    *,
    likelihood: str,
    exposure_weight: np.ndarray,
    noevent_weight: np.ndarray,
    event_weight: np.ndarray,
) -> float:
    if likelihood == "poisson":
        values = poisson_conjugate(dual, exposure_weight, event_weight)
    elif likelihood == "first_event_cloglog":
        values = cloglog_conjugate(dual, noevent_weight, event_weight)
    else:
        raise ValueError(likelihood)
    return float(np.sum(values)) if np.all(np.isfinite(values)) else math.inf
