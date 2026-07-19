from __future__ import annotations

import math
from dataclasses import dataclass

from .response import ModelMatrix
from .rules import Support
from .solver import FitResult


@dataclass(frozen=True)
class ObjectiveSpec:
    n_entities: int
    skeleton_count: int
    knot_count: int
    window_count_by_order: tuple[int, int, int]

    def penalty(self, support: Support, matrix: ModelMatrix, baseline_dimension: int) -> float:
        size = len(support.rules)
        if size == 0:
            return 0.0
        if not 0 <= size <= self.skeleton_count:
            raise ValueError("support size exceeds skeleton dictionary")
        parameter_dimension = matrix.dimension - int(baseline_dimension)
        parameter_code = parameter_dimension * math.log(max(2, int(self.n_entities)))
        support_code = 2.0 * (
            math.lgamma(self.skeleton_count + 1)
            - math.lgamma(size + 1)
            - math.lgamma(self.skeleton_count - size + 1)
        )
        identity_code = 0.0
        for rule in support.rules:
            count = self.window_count_by_order[rule.order - 1]
            identity_code += 2.0 * math.log(2 * count)
        return float(parameter_code + support_code + identity_code)


@dataclass(frozen=True)
class SupportRecord:
    support: Support
    matrix: ModelMatrix
    fit: FitResult
    penalty: float
    score: float


def support_score(
    *, baseline_nll: float, fit_nll: float, penalty: float
) -> float:
    if not math.isfinite(fit_nll):
        return -math.inf
    return float(2.0 * (baseline_nll - fit_nll) - penalty)

