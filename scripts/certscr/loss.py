from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import QueryContext
from .model import cluster_nll


@dataclass(frozen=True)
class ClusterLoss:
    """Cluster-separable loss used for support selection and certification.

    Financial grounding is an explicit property of the supplied loss, never
    inferred from a statistically significant predictive score.
    """

    name: str
    financially_grounded: bool
    global_sequence_weights: np.ndarray | None = None

    def weights(self, ctx: QueryContext) -> np.ndarray:
        if self.global_sequence_weights is None:
            return np.ones(ctx.n_sequences, dtype=np.float64)
        weights = np.asarray(self.global_sequence_weights, dtype=np.float64)
        if weights.ndim != 1 or np.max(ctx.global_sequence_ids) >= len(weights):
            raise ValueError("financial loss weights do not cover the query context")
        selected = weights[ctx.global_sequence_ids]
        if np.any(~np.isfinite(selected)) or np.any(selected < 0) or not np.any(selected > 0):
            raise ValueError("financial loss weights must be finite, nonnegative, and not all zero")
        return selected

    def values(self, eta: np.ndarray, ctx: QueryContext) -> np.ndarray:
        return self.weights(ctx) * cluster_nll(eta, ctx)


def predictive_nll_loss() -> ClusterLoss:
    return ClusterLoss(name="tpp_negative_log_likelihood", financially_grounded=False)


def financial_weighted_nll_loss(weights: np.ndarray, *, weight_name: str) -> ClusterLoss:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("financial weights must be a finite nonnegative vector with positive mass")
    return ClusterLoss(
        name=f"financial_weighted_tpp_nll[{weight_name}]",
        financially_grounded=True,
        global_sequence_weights=weights.copy(),
    )
