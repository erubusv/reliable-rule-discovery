from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .response import ModelMatrix
from .rules import PatternKey, Support, hierarchy_closure
from .solver import FitResult


@dataclass(frozen=True, eq=False)
class ObjectiveSpec:
    n_entities: int
    skeleton_count: int
    knot_count: int
    window_count_by_order: tuple[int, int, int]
    window_count_by_antecedent: tuple[tuple[tuple[int, ...], int], ...] = ()
    window_count_by_pattern: tuple[tuple[PatternKey, int], ...] = ()
    kernel_family_count: int = 1
    history_identity_count_by_pattern: tuple[tuple[PatternKey, int], ...] = ()
    _pattern_window_lookup: dict[PatternKey, int] = field(init=False, repr=False)
    _antecedent_window_lookup: dict[tuple[int, ...], int] = field(
        init=False, repr=False
    )
    _history_identity_lookup: dict[PatternKey, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # A relation-aware q<=3 dictionary can contain tens of thousands of
        # patterns.  The former linear scan made each MDL evaluation O(K), and
        # the generated dataclass hash traversed the same K-entry tuple again
        # before every @lru_cache lookup.  Objective specifications are
        # immutable run-local objects, so identity hashing plus immutable
        # lookup tables preserves every numerical value while making each
        # window-code query O(1).
        object.__setattr__(
            self,
            "_pattern_window_lookup",
            {pattern: int(count) for pattern, count in self.window_count_by_pattern},
        )
        object.__setattr__(
            self,
            "_antecedent_window_lookup",
            {
                antecedent: int(count)
                for antecedent, count in self.window_count_by_antecedent
            },
        )
        object.__setattr__(
            self,
            "_history_identity_lookup",
            {
                pattern: int(count)
                for pattern, count in self.history_identity_count_by_pattern
            },
        )

    def window_count(
        self, antecedent: tuple[int, ...], relation: str = "unordered"
    ) -> int:
        normalized = "atomic" if len(antecedent) == 1 else str(relation)
        pattern_count = self._pattern_window_lookup.get((normalized, antecedent))
        if pattern_count is not None:
            return pattern_count
        antecedent_count = self._antecedent_window_lookup.get(antecedent)
        if antecedent_count is not None:
            return antecedent_count
        return int(self.window_count_by_order[len(antecedent) - 1])

    def history_identity_count(self, pattern: PatternKey) -> int:
        return max(1, int(self._history_identity_lookup.get(pattern, 1)))

    @lru_cache(maxsize=262_144)
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
            count = self.window_count(rule.antecedent, rule.relation)
            identity_code += 2.0 * math.log(
                2
                * count
                * max(1, int(self.kernel_family_count))
                * self.history_identity_count(rule.pattern_key)
            )
        return float(parameter_code + support_code + identity_code)

    def penalty_for_effective_dimension(
        self,
        support: Support,
        effective_dimension: float,
        *,
        n_entities: int | None = None,
    ) -> float:
        """Dependency-aware code length for an exactly fitted support.

        The identity code is unchanged.  The parameter code uses the
        Godambe effective dimension, lower-bounded by the declared structural
        dimension.  Finite-sample negative covariance fluctuations can
        therefore never make dependent evidence cheaper than ordinary BIC.
        This also leaves :meth:`structural_penalty` as a rigorous lower bound
        for every existing likelihood-relaxation upper certificate.
        """

        size = len(support.rules)
        if size == 0:
            return 0.0
        if not math.isfinite(effective_dimension) or effective_dimension < 0.0:
            raise ValueError("effective dimension must be finite and nonnegative")
        structural = float(self.parameter_dimension(support))
        dimension = max(structural, float(effective_dimension))
        sample_size = self.n_entities if n_entities is None else int(n_entities)
        parameter_code = dimension * math.log(max(2, sample_size))
        support_code = 2.0 * (
            math.lgamma(self.skeleton_count + 1)
            - math.lgamma(size + 1)
            - math.lgamma(self.skeleton_count - size + 1)
        )
        identity_code = sum(
            2.0
            * math.log(
                2
                * self.window_count(rule.antecedent, rule.relation)
                * max(1, int(self.kernel_family_count))
                * self.history_identity_count(rule.pattern_key)
            )
            for rule in support.rules
        )
        return float(parameter_code + support_code + identity_code)

    @lru_cache(maxsize=262_144)
    def parameter_dimension(self, support: Support) -> int:
        return (
            sum(rule.kernel_dimension(self.knot_count) for rule in support.rules)
            + len(hierarchy_closure(support)) * self.knot_count
        )

    @lru_cache(maxsize=262_144)
    def structural_penalty(self, support: Support) -> float:
        return self.penalty_for_dimension(support, self.parameter_dimension(support))

    @lru_cache(maxsize=262_144)
    def proposal_penalty(self, support: Support) -> float:
        """One-amplitude code used only to order discovery proposals.

        A finite rule identity first enters column generation through one
        nonnegative amplitude along a normalized score direction.  Charging
        one degree of freedom per proposed rule makes this relaxation
        identical for singleton, pair and triplet antecedents.  It is never a
        reportable-model penalty: every accepted support is still refitted
        with all ``knot_count`` coefficients and evaluated by
        :meth:`structural_penalty`.
        """

        return self.penalty_for_dimension(support, len(support.rules))

    @lru_cache(maxsize=262_144)
    def proposal_add_penalty_delta(
        self, support: Support, pattern: PatternKey
    ) -> float:
        """Exact one-amplitude code increment without building a child.

        Proposal pricing may inspect tens of thousands of W/sign identities
        at one support.  All identities of one antecedent pattern have the
        same one-amplitude code, yet the former implementation constructed
        and hashed a complete ``Support`` (and its hierarchy) for every one.
        This is the algebraic difference of :meth:`proposal_penalty` for
        ``support + pattern``: one parameter, the adjacent support-size code,
        and the new finite identity code.  It is independent of the selected
        W/sign value, exactly as the original code length is.
        """

        pattern = (str(pattern[0]), tuple(pattern[1]))
        if pattern in support.patterns:
            raise ValueError("proposal pattern is already active")
        size = len(support.rules)
        if not 0 <= size < self.skeleton_count:
            raise ValueError("proposal support size exceeds skeleton dictionary")
        parameter_delta = math.log(max(2, self.n_entities))
        # C(K,s+1) / C(K,s) = (K-s)/(s+1).
        support_delta = 2.0 * math.log((self.skeleton_count - size) / (size + 1))
        count = self.window_count(pattern[1], pattern[0])
        identity_delta = 2.0 * math.log(
            2
            * count
            * max(1, int(self.kernel_family_count))
            * self.history_identity_count(pattern)
        )
        return float(parameter_delta + support_delta + identity_delta)

    @lru_cache(maxsize=262_144)
    def reported_branch_penalty(
        self,
        support: Support,
        parameter_dimension: int,
        *,
        n_entities: int | None = None,
    ) -> float:
        """Code length of the visible total-state rules.

        The method name is retained for compatibility with certification code;
        without hidden closure this is the ordinary support code.
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
            2.0
            * math.log(
                2
                * self.window_count(rule.antecedent, rule.relation)
                * max(1, int(self.kernel_family_count))
                * self.history_identity_count(rule.pattern_key)
            )
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
            raise AssertionError("rule and shared hierarchy complexity are not exact")
        return penalty


@dataclass(frozen=True)
class SupportRecord:
    support: Support
    matrix: ModelMatrix
    fit: FitResult
    penalty: float
    score: float
    rule_score: float | None = None
    closure_null_nll: float | None = None
    rule_score_upper: float | None = None
    dependency_effective_dimension: float | None = None
    dependency_diagnostics: dict[str, object] | None = None

    @property
    def discovery_score(self) -> float:
        """Common-baseline total-state MDL used by support discovery.

        Provisional one-step states expose a feasible endpoint until terminal
        exact correction.  Every reportable record is exact and has
        ``rule_score`` populated; fast route edges may remain provisional.
        """
        return self.score if self.rule_score is None else self.rule_score

    @property
    def discovery_upper_score(self) -> float:
        """Certified upper endpoint of the common total-state objective.

        Exact records set both endpoints to the same value.  A route state may
        carry a primal/dual interval until the terminal exact correction.
        """
        if self.rule_score_upper is not None:
            return self.rule_score_upper
        return self.discovery_score


def freeze_support_record(record: SupportRecord) -> SupportRecord:
    """Remove observation-sized arrays while preserving a fitted model.

    Search and certification need the support identity, block layout, frozen
    closure signs and coefficients after a model has been fitted.  They do not
    need to retain every candidate's multi-million-row design matrix.  A
    metadata-only ``ModelMatrix`` keeps those invariants and makes accidental
    numerical use fail by shape rather than silently holding the original
    allocation alive.
    """
    if record.matrix.x.shape[0] == 0:
        return record
    empty_float = np.empty(0, dtype=np.float64)
    empty_int64 = np.empty(0, dtype=np.int64)
    empty_int32 = np.empty(0, dtype=np.int32)
    matrix = ModelMatrix(
        x=np.empty((0, record.matrix.dimension), dtype=np.float64),
        exposure_weight=empty_float,
        noevent_weight=empty_float,
        event_weight=empty_float,
        free_dimension=record.matrix.free_dimension,
        closure_dimension=record.matrix.closure_dimension,
        rule_slices=record.matrix.rule_slices,
        support=record.matrix.support,
        closure=record.matrix.closure,
        closure_signs=record.matrix.closure_signs,
        active_rows=empty_int64,
        active_design_groups=empty_int32,
        active_baseline_groups=empty_int32,
        aggregate_baseline_groups=empty_int32,
        control_dimension=record.matrix.control_dimension,
    )
    return SupportRecord(
        support=record.support,
        matrix=matrix,
        fit=record.fit,
        penalty=record.penalty,
        score=record.score,
        rule_score=record.rule_score,
        closure_null_nll=record.closure_null_nll,
        rule_score_upper=record.rule_score_upper,
        dependency_effective_dimension=record.dependency_effective_dimension,
        dependency_diagnostics=record.dependency_diagnostics,
    )


def support_score(*, baseline_nll: float, fit_nll: float, penalty: float) -> float:
    if not math.isfinite(fit_nll):
        return -math.inf
    return float(2.0 * (baseline_nll - fit_nll) - penalty)


def block_mdl_delta(
    *,
    likelihood_gain: float,
    parent_penalty: float,
    child_penalty: float,
) -> float:
    """Return the Block-MDL change produced by one model move.

    ``likelihood_gain`` is ``NLL(parent) - NLL(child)``. The helper is used
    by exact fits and by quadratic likelihood relaxations alike, preventing a
    proposal score from silently changing the model-selection objective.
    A relaxed caller may pass a rigorous lower bound on ``child_penalty``;
    the returned value is then an upper bound on this same Block-MDL delta,
    not a different one-df objective.
    """

    if not math.isfinite(likelihood_gain):
        return math.inf if likelihood_gain > 0.0 else -math.inf
    if not math.isfinite(parent_penalty) or not math.isfinite(child_penalty):
        return -math.inf
    return float(
        2.0 * float(likelihood_gain) - (float(child_penalty) - float(parent_penalty))
    )


def family_block_mdl_threshold(
    *,
    parent_score: float,
    separate_score: float | None = None,
) -> float:
    """Return the incumbent score a joint child must beat."""

    threshold = float(parent_score)
    if separate_score is not None:
        threshold = max(threshold, float(separate_score))
    return threshold


def family_block_mdl_delta(
    *,
    parent_score: float,
    child_score: float,
    separate_score: float | None = None,
    branch_delta: float | None = None,
) -> float:
    """Return the constrained Family Block-MDL change of one move.

    The joint child must beat every already feasible family representation.
    A high-order Add must additionally have positive closure-matched branch
    contribution. Taking the minimum encodes the conjunction exactly.
    """

    threshold = family_block_mdl_threshold(
        parent_score=parent_score,
        separate_score=separate_score,
    )
    if not math.isfinite(threshold):
        return -math.inf
    if not math.isfinite(child_score):
        return math.inf if child_score > 0.0 else -math.inf
    delta = float(child_score) - threshold
    if branch_delta is not None:
        delta = min(delta, float(branch_delta))
    return float(delta)


def relaxed_family_block_mdl_delta(
    *,
    parent_score: float,
    block_delta: float,
    separate_score: float | None = None,
    branch_delta: float | None = None,
) -> float:
    """Map a quadratic Block-MDL delta to the same family objective.

    ``block_delta`` may be a local approximation or a valid upper endpoint.
    Exact/dual validation keeps its original role. Proposal ranking,
    admission and terminal block audit nevertheless subtract the identical
    family hurdle and apply the identical closure-matched constraint.
    """

    child_score = float(parent_score) + float(block_delta)
    return family_block_mdl_delta(
        parent_score=parent_score,
        child_score=child_score,
        separate_score=separate_score,
        branch_delta=branch_delta,
    )
