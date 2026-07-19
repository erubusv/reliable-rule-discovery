from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .config import RunConfig
from .dual import DualCertificate, dual_certificate
from .objective import ObjectiveSpec, SupportRecord, support_score
from .response import Context, ModelMatrix, ResponseEngine
from .rules import (
    EMPTY_SUPPORT,
    RuleIdentity,
    Support,
    identities_for,
    one_exchange_neighbors,
    rule_dictionary,
    skeletons,
)
from .solver import FitResult, fit_model_matrix


@dataclass(frozen=True)
class ProposalBounds:
    support: Support
    lower_score: float
    upper_score: float
    lower_gain: float
    upper_gain: float
    dual: DualCertificate | None
    warm_coefficients: np.ndarray


@dataclass
class SearchDiagnostics:
    exact_fits: int = 0
    fit_cache_hits: int = 0
    bound_evaluations: int = 0
    dual_screens: int = 0
    dual_fail_open: int = 0
    accepted_moves: int = 0
    states: int = 0
    transition_cache_hits: int = 0


@dataclass(frozen=True)
class SearchResult:
    family: tuple[SupportRecord, ...]
    terminals: tuple[SupportRecord, ...]
    positive_atoms: tuple[SupportRecord, ...]
    paths: tuple[dict[str, object], ...]
    diagnostics: SearchDiagnostics


class SupportOptimizer:
    def __init__(self, context: Context, config: RunConfig):
        self.context = context
        self.config = config
        self.engine = ResponseEngine(
            context.dataset,
            lag=config.impact_lag,
            knot_count=config.knot_count,
            cache_bytes=config.cache_bytes,
        )
        self.skeletons = skeletons(context.dataset.n_predicates, config.q_max)
        self.dictionary = rule_dictionary(
            context.dataset.n_predicates, config.q_max, config.formation_windows
        )
        self.objective = ObjectiveSpec(
            n_entities=len(context.entity_codes),
            skeleton_count=len(self.skeletons),
            knot_count=config.knot_count,
            window_count_by_order=(1, len(config.formation_windows), len(config.formation_windows)),
        )
        self.records: dict[Support, SupportRecord] = {}
        self.diagnostics = SearchDiagnostics()
        baseline_matrix = self.engine.model_matrix(context, EMPTY_SUPPORT)
        baseline_fit = fit_model_matrix(
            baseline_matrix,
            likelihood=context.dataset.likelihood,
            tolerance=config.solver_tolerance,
            max_iter=config.solver_max_iter,
        )
        if not baseline_fit.converged:
            raise RuntimeError(f"baseline fit failed: {baseline_fit.message}")
        self.baseline_dimension = baseline_matrix.dimension
        self.baseline_nll = baseline_fit.nll
        self.records[EMPTY_SUPPORT] = SupportRecord(
            EMPTY_SUPPORT, baseline_matrix, baseline_fit, 0.0, 0.0
        )
        self.diagnostics.exact_fits += 1

    @staticmethod
    def _block_map(matrix: ModelMatrix) -> tuple[dict[object, slice], dict[object, slice]]:
        closure_map: dict[object, slice] = {}
        left = matrix.free_dimension - matrix.closure_dimension
        knot_count = matrix.rule_slices[0].stop - matrix.rule_slices[0].start if matrix.rule_slices else (
            matrix.closure_dimension // len(matrix.closure) if matrix.closure else 0
        )
        for index, term in enumerate(matrix.closure):
            closure_map[term] = slice(left + index * knot_count, left + (index + 1) * knot_count)
        rule_map = {rule: block for rule, block in zip(matrix.support.rules, matrix.rule_slices, strict=True)}
        return closure_map, rule_map

    def warm_start(self, source: SupportRecord, target: ModelMatrix) -> np.ndarray:
        output = np.zeros(target.dimension, dtype=np.float64)
        baseline = min(self.baseline_dimension, len(source.fit.coefficients), target.dimension)
        output[:baseline] = source.fit.coefficients[:baseline]
        source_closure, source_rules = self._block_map(source.matrix)
        target_closure, target_rules = self._block_map(target)
        for term, destination in target_closure.items():
            origin = source_closure.get(term)
            if origin is not None:
                output[destination] = source.fit.coefficients[origin]
        for rule, destination in target_rules.items():
            origin = source_rules.get(rule)
            if origin is not None:
                output[destination] = source.fit.coefficients[origin]
        output[target.free_dimension:] = np.maximum(output[target.free_dimension:], 0.0)
        return output

    def fit(self, support: Support, source: SupportRecord | None = None) -> SupportRecord:
        cached = self.records.get(support)
        if cached is not None:
            self.diagnostics.fit_cache_hits += 1
            return cached
        matrix = self.engine.model_matrix(self.context, support)
        warm = None if source is None else self.warm_start(source, matrix)
        fit = fit_model_matrix(
            matrix,
            likelihood=self.context.dataset.likelihood,
            tolerance=self.config.solver_tolerance,
            max_iter=self.config.solver_max_iter,
            warm_start=warm,
        )
        penalty = self.objective.penalty(support, matrix, self.baseline_dimension)
        score = support_score(
            baseline_nll=self.baseline_nll, fit_nll=fit.nll, penalty=penalty
        ) if fit.converged else -math.inf
        record = SupportRecord(support, matrix, fit, penalty, score)
        self.records[support] = record
        self.diagnostics.exact_fits += 1
        return record

    def bounds(self, current: SupportRecord, trial: Support) -> ProposalBounds:
        self.diagnostics.bound_evaluations += 1
        matrix = self.engine.model_matrix(self.context, trial)
        warm = self.warm_start(current, matrix)
        # One projected Newton iteration produces a feasible primal point and
        # therefore a rigorous lower bound on the optimized support score.
        one_step = fit_model_matrix(
            matrix,
            likelihood=self.context.dataset.likelihood,
            tolerance=self.config.solver_tolerance,
            max_iter=1,
            warm_start=warm,
        )
        feasible = one_step.coefficients
        from .likelihood import loss_rows
        eta = matrix.x @ feasible
        rows, _, _ = loss_rows(
            eta,
            likelihood=self.context.dataset.likelihood,
            exposure_weight=matrix.exposure_weight,
            noevent_weight=matrix.noevent_weight,
            event_weight=matrix.event_weight,
        )
        feasible_nll = float(np.sum(rows))
        penalty = self.objective.penalty(trial, matrix, self.baseline_dimension)
        lower_score = support_score(
            baseline_nll=self.baseline_nll, fit_nll=feasible_nll, penalty=penalty
        )
        certificate = dual_certificate(
            matrix,
            likelihood=self.context.dataset.likelihood,
            beta=feasible,
            tolerance=min(1.0e-10, self.config.solver_tolerance * 0.01),
        )
        if certificate.feasible:
            upper_score = support_score(
                baseline_nll=self.baseline_nll,
                fit_nll=certificate.nll_lower_bound,
                penalty=penalty,
            )
        else:
            self.diagnostics.dual_fail_open += 1
            upper_score = math.inf
        return ProposalBounds(
            trial, lower_score, upper_score,
            lower_score - current.score, upper_score - current.score,
            certificate, feasible,
        )

    def best_exact_neighbor(
        self, current: SupportRecord, proposals: list[Support]
    ) -> tuple[SupportRecord | None, float]:
        unique = sorted(set(proposals), key=lambda support: support.rules)
        bounded = [self.bounds(current, support) for support in unique]
        bounded.sort(key=lambda item: (-item.upper_score, item.support.rules))
        incumbent: SupportRecord | None = None
        incumbent_score = current.score
        for item in bounded:
            if item.upper_score <= incumbent_score + self.config.search_tolerance:
                if math.isfinite(item.upper_score):
                    self.diagnostics.dual_screens += 1
                continue
            record = self.fit(item.support, current)
            if record.score > incumbent_score + self.config.search_tolerance:
                incumbent, incumbent_score = record, record.score
        return incumbent, max(0.0, incumbent_score - current.score)

    def splice_proposals(self, current: SupportRecord) -> list[Support]:
        existing = {rule.antecedent for rule in current.support.rules}
        add_trials = [
            current.support.add(rule)
            for rule in self.dictionary
            if rule.antecedent not in existing
        ]
        if not add_trials:
            return []
        # The local primal gain orders all blocks; no block is removed from the
        # one-exchange audit.  The ordering is used only to form deterministic
        # multi-block escape proposals.
        add_bounds = [self.bounds(current, support) for support in add_trials]
        best_by_antecedent: dict[tuple[int, ...], ProposalBounds] = {}
        for item in add_bounds:
            rule = next(rule for rule in item.support.rules if rule.antecedent not in existing)
            previous = best_by_antecedent.get(rule.antecedent)
            if previous is None or (item.lower_gain, tuple(item.support.rules)) > (
                previous.lower_gain, tuple(previous.support.rules)
            ):
                best_by_antecedent[rule.antecedent] = item
        additions = sorted(
            best_by_antecedent.values(), key=lambda item: (-item.lower_gain, item.support.rules)
        )
        added_rules = [
            next(rule for rule in item.support.rules if rule.antecedent not in existing)
            for item in additions
        ]
        proposals: set[Support] = set()
        # Add-only splices expose pure synergistic structures.
        base_rules = list(current.support.rules)
        for size in range(2, len(added_rules) + 1):
            proposals.add(Support.of((*base_rules, *added_rules[:size])))
        if current.support.rules:
            drop_records = [self.fit(current.support.drop(rule), current) for rule in current.support.rules]
            sacrifices = sorted(
                zip(current.support.rules, drop_records, strict=True),
                key=lambda item: (current.score - item[1].score, item[0]),
            )
            maximum = min(len(sacrifices), len(added_rules))
            for size in range(1, maximum + 1):
                removed = {item[0] for item in sacrifices[:size]}
                proposals.add(Support.of(
                    [rule for rule in current.support.rules if rule not in removed]
                    + added_rules[:size]
                ))
        proposals.discard(current.support)
        return sorted(proposals, key=lambda support: support.rules)

    def search(self) -> SearchResult:
        positive_atoms: dict[Support, SupportRecord] = {}
        starts: list[Support] = [EMPTY_SUPPORT]
        # One exact best W/sign identity per skeleton is retained as a basin
        # witness. Negative atoms are also starts so suppressor combinations
        # are reachable without heredity assumptions.
        empty = self.records[EMPTY_SUPPORT]
        for antecedent in self.skeletons:
            candidates = [Support.of((rule,)) for rule in identities_for(antecedent, self.config.formation_windows)]
            records = [self.fit(support, empty) for support in candidates]
            best = max(records, key=lambda record: (record.score, tuple(record.support.rules)))
            if best.fit.converged:
                starts.append(best.support)
            for record in records:
                if record.score > 0:
                    positive_atoms[record.support] = record
        transition_cache: dict[Support, tuple[Support | None, float]] = {}
        terminals: dict[Support, SupportRecord] = {}
        paths: list[dict[str, object]] = []
        for start in starts:
            current = self.fit(start, empty) if start.rules else empty
            moves: list[dict[str, object]] = []
            while True:
                self.diagnostics.states += 1
                transition = transition_cache.get(current.support)
                if transition is not None:
                    self.diagnostics.transition_cache_hits += 1
                    next_support, gain = transition
                    if next_support is None:
                        break
                    next_record = self.fit(next_support, current)
                else:
                    one_exchange = list(one_exchange_neighbors(current.support, self.dictionary))
                    joint = self.splice_proposals(current)
                    next_record, gain = self.best_exact_neighbor(current, one_exchange + joint)
                    transition_cache[current.support] = (
                        None if next_record is None else next_record.support, gain
                    )
                    if next_record is None:
                        break
                self.diagnostics.accepted_moves += 1
                moves.append({
                    "from": support_key(current.support),
                    "to": support_key(next_record.support),
                    "gain": float(gain),
                })
                current = next_record
            # A complete exact-or-dual one-exchange audit was performed by the
            # terminal best-neighbor call above.
            if current.score > 0:
                terminals[current.support] = current
            paths.append({"start": support_key(start), "terminal": support_key(current.support), "moves": moves})
        family_map = {**positive_atoms, **terminals}
        return SearchResult(
            family=tuple(family_map[key] for key in sorted(family_map, key=lambda support: support.rules)),
            terminals=tuple(terminals[key] for key in sorted(terminals, key=lambda support: support.rules)),
            positive_atoms=tuple(positive_atoms[key] for key in sorted(positive_atoms, key=lambda support: support.rules)),
            paths=tuple(paths),
            diagnostics=self.diagnostics,
        )


def support_key(support: Support) -> str:
    if not support.rules:
        return "empty"
    return ";".join(
        f"{','.join(map(str, rule.antecedent))}|W{rule.window}|{'exc' if rule.sign > 0 else 'inh'}"
        for rule in support.rules
    )

