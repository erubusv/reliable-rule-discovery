from __future__ import annotations

import math
from dataclasses import dataclass

from .response import ModelMatrix
from .rules import Support, hierarchy_closure
from .solver import FitResult


@dataclass(frozen=True)
class ObjectiveSpec:
    n_entities: int
    skeleton_count: int
    knot_count: int
    window_count_by_order: tuple[int, int, int]

    def penalty_for_dimension(
        self,
        support: Support,
        parameter_dimension: int,
        *,
        n_entities: int | None = None,
    ) -> float:
        size = len(support.rules)
        if size == 0:
            return 0.0
        if not 0 <= size <= self.skeleton_count:
            raise ValueError("support size exceeds skeleton dictionary")
        sample_size = self.n_entities if n_entities is None else int(n_entities)
        parameter_code = parameter_dimension * math.log(max(2, sample_size))
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

    def structural_penalty(self, support: Support) -> float:
        parameter_dimension = self.knot_count * (
            len(support.rules) + len(hierarchy_closure(support))
        )
        return self.penalty_for_dimension(support, parameter_dimension)

    def reported_branch_penalty(
        self,
        support: Support,
        parameter_dimension: int,
        *,
        n_entities: int | None = None,
    ) -> float:
        """Code length of reported branches relative to their shared closure null.

        The closure nuisance is present in both models and must not be charged in
        the contrast.  Reported kernel parameters, support identity and W/sign
        identities remain part of the code.  ``n_entities`` permits the held-out
        certification threshold to use the certification sample size rather than
        silently reusing the discovery size.
        """
        size = len(support.rules)
        if size == 0:
            return 0.0
        sample_size = self.n_entities if n_entities is None else int(n_entities)
        parameter_code = int(parameter_dimension) * math.log(max(2, sample_size))
        support_code = 2.0 * (
            math.lgamma(self.skeleton_count + 1)
            - math.lgamma(size + 1)
            - math.lgamma(self.skeleton_count - size + 1)
        )
        identity_code = sum(
            2.0 * math.log(2 * self.window_count_by_order[rule.order - 1])
            for rule in support.rules
        )
        return float(parameter_code + support_code + identity_code)

    def penalty(
        self, support: Support, matrix: ModelMatrix, baseline_dimension: int
    ) -> float:
        parameter_dimension = matrix.dimension - int(baseline_dimension)
        penalty = self.penalty_for_dimension(support, parameter_dimension)
        expected = self.structural_penalty(support)
        if not math.isclose(penalty, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise AssertionError("closure complexity is not counted exactly once")
        return penalty


@dataclass(frozen=True)
class SupportRecord:
    support: Support
    matrix: ModelMatrix
    fit: FitResult
    penalty: float
    score: float


def support_score(*, baseline_nll: float, fit_nll: float, penalty: float) -> float:
    if not math.isfinite(fit_nll):
        return -math.inf
    return float(2.0 * (baseline_nll - fit_nll) - penalty)
