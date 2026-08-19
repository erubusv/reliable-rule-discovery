from __future__ import annotations

import math

import numpy as np


def _aligned_binary(y_true: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("binary outcomes and probabilities must be aligned vectors")
    if np.any((y != 0) & (y != 1)) or np.any(~np.isfinite(p)):
        raise ValueError("invalid binary predictions")
    return y, np.clip(p, 1.0e-12, 1.0 - 1.0e-12)


def binary_log_loss(y_true: np.ndarray, probability: np.ndarray) -> float:
    y, p = _aligned_binary(y_true, probability)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    y, p = _aligned_binary(y_true, probability)
    return float(np.mean(np.square(p - y)))


def roc_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Exact Mann--Whitney AUROC with average ranks for tied scores."""

    y, p = _aligned_binary(y_true, probability)
    positive = int(np.count_nonzero(y))
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        return math.nan
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    left = 0
    while left < len(p):
        right = left + 1
        while right < len(p) and sorted_p[right] == sorted_p[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + 1 + right)
        left = right
    rank_sum = float(np.sum(ranks[y == 1]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray
) -> dict[str, float | int | None]:
    auc = roc_auc(y_true, probability)
    return {
        "n": int(len(y_true)),
        "targets": int(np.sum(y_true)),
        "auroc": float(auc) if math.isfinite(auc) else None,
        "brier": brier_score(y_true, probability),
        "binary_nll": binary_log_loss(y_true, probability),
    }
