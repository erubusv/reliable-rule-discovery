from __future__ import annotations

import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .config import RunConfig
from .dual import DualCertificate, DualGeometry, dual_certificate, dual_geometry
from .likelihood import loss_rows
from .native import configure_cpu_threads, moments, moments_batch
from .objective import ObjectiveSpec, SupportRecord, support_score
from .response import Context, ModelMatrix, ResponseEngine, SparseBlock
from .rules import (
    EMPTY_SUPPORT,
    Antecedent,
    ClosureTerm,
    RuleIdentity,
    Support,
    hierarchy_closure,
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
    pricing_passes: int = 0
    priced_blocks: int = 0
    bound_evaluations: int = 0
    relaxation_screens: int = 0
    directional_relaxation_screens: int = 0
    dual_screens: int = 0
    dual_fail_open: int = 0
    accepted_moves: int = 0
    states: int = 0
    transition_cache_hits: int = 0
    terminal_audits: int = 0
    total_skeletons: int = 0
    admissible_skeletons: int = 0
    empty_skeletons: int = 0
    equivalent_window_identities: int = 0
    fused_skeleton_pricing_passes: int = 0
    fused_sign_prices: int = 0
    parallel_exact_batches: int = 0
    forced_fit_cache_hits: int = 0
    dual_geometry_cache_hits: int = 0
    block_score_evaluations: int = 0
    block_score_screens: int = 0
    block_score_admissible: int = 0
    block_score_exact_rejections: int = 0
    nonnested_exact_audits: int = 0


@dataclass(frozen=True)
class SearchResult:
    family: tuple[SupportRecord, ...]
    terminals: tuple[SupportRecord, ...]
    positive_atoms: tuple[SupportRecord, ...]
    paths: tuple[dict[str, object], ...]
    diagnostics: SearchDiagnostics


@dataclass(frozen=True)
class _StoredRecord:
    fit: FitResult
    penalty: float
    score: float


def _nonnegative_quadratic_gain(gradient: np.ndarray, hessian: np.ndarray) -> float:
    """Solve the tiny M-dimensional nonnegative Fisher pricing problem exactly."""
    gradient = np.asarray(gradient, dtype=np.float64)
    hessian = 0.5 * (np.asarray(hessian, dtype=np.float64) + np.asarray(hessian).T)
    dimension = len(gradient)
    best = 0.0
    for mask in range(1, 1 << dimension):
        active = np.asarray(
            [index for index in range(dimension) if mask & (1 << index)]
        )
        sub_hessian = hessian[np.ix_(active, active)]
        sub_gradient = gradient[active]
        try:
            delta_active = np.linalg.solve(sub_hessian, -sub_gradient)
        except np.linalg.LinAlgError:
            delta_active = np.linalg.lstsq(sub_hessian, -sub_gradient, rcond=None)[0]
        if np.any(delta_active < -1.0e-10):
            continue
        delta = np.zeros(dimension, dtype=np.float64)
        delta[active] = np.maximum(delta_active, 0.0)
        stationarity = gradient + hessian @ delta
        inactive = np.ones(dimension, dtype=bool)
        inactive[active] = False
        if np.any(stationarity[inactive] < -1.0e-8):
            continue
        gain = -float(gradient @ delta) - 0.5 * float(delta @ hessian @ delta)
        best = max(best, gain)
    return best


class SupportOptimizer:
    """Exact fixed-support fits with hierarchy-aware block-score search.

    Every accepted move is an exact MDL improvement.  Inactive blocks whose
    joint Fisher score cannot pay both the complete hierarchy cost and the
    reported rule's conditional cost are excluded from the local audit.  This
    yields block-score stationarity rather than exact one-exchange stationarity.
    """

    def __init__(self, context: Context, config: RunConfig):
        self.context = context
        self.config = config
        self.engine = ResponseEngine(
            context.dataset,
            lag=config.impact_lag,
            knot_count=config.knot_count,
            cache_bytes=5 * config.cache_bytes // 8,
        )
        all_skeletons = skeletons(context.dataset.n_predicates, config.q_max)
        self.diagnostics = SearchDiagnostics(total_skeletons=len(all_skeletons))
        self.skeletons, self.dictionary = self._structurally_admissible_dictionary(
            all_skeletons
        )
        self.objective = ObjectiveSpec(
            n_entities=len(context.entity_codes),
            skeleton_count=len(self.skeletons),
            knot_count=config.knot_count,
            window_count_by_order=(
                1,
                len(config.formation_windows),
                len(config.formation_windows),
            ),
        )
        self.records: OrderedDict[Support, SupportRecord] = OrderedDict()
        self._stored_records: dict[Support, _StoredRecord] = {}
        self._forced_fits: dict[tuple[Support, tuple[ClosureTerm, ...]], FitResult] = {}
        self._state_lock = threading.RLock()
        self._record_cache_bytes = 0
        self._record_cache_limit = max(1, 2 * config.cache_bytes // 8)
        self._addition_cache: dict[Support, Support | None] = {}
        self._relaxed_upper_cache: OrderedDict[tuple, float] = OrderedDict()
        self._relaxed_upper_limit = max(1, min(200_000, config.cache_bytes // 256))
        self._directional_upper_cache: OrderedDict[Support, float] = OrderedDict()
        self._pricing_state: OrderedDict[Support, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._pricing_cache_bytes = 0
        self._pricing_cache_limit = max(1, config.cache_bytes // 8)
        self._block_price_cache: OrderedDict[
            tuple[Support, Antecedent, int], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._block_price_cache_limit = max(
            1, min(200_000, config.cache_bytes // max(1, 32 * config.knot_count**2))
        )
        self._dual_geometry_cache: OrderedDict[tuple, DualGeometry] = OrderedDict()
        self._dual_geometry_cache_limit = max(
            1, min(50_000, config.cache_bytes // 4096)
        )
        self._skeleton_witnesses: dict[Antecedent, tuple[float, RuleIdentity]] = {}
        self._standalone_screen_cache: dict[Antecedent, tuple[RuleIdentity, ...]] = {}
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
        self.saturated_nll_lower_bound = self._saturated_nll_lower_bound()
        baseline_record = SupportRecord(
            EMPTY_SUPPORT, baseline_matrix, baseline_fit, 0.0, 0.0
        )
        self._stored_records[EMPTY_SUPPORT] = _StoredRecord(baseline_fit, 0.0, 0.0)
        self._retain_record(baseline_record)
        self.diagnostics.exact_fits += 1

    @staticmethod
    def _record_nbytes(record: SupportRecord) -> int:
        return int(record.matrix.nbytes + record.fit.coefficients.nbytes)

    def _retain_record(self, record: SupportRecord) -> SupportRecord:
        previous = self.records.pop(record.support, None)
        if previous is not None:
            self._record_cache_bytes -= self._record_nbytes(previous)
        self.records[record.support] = record
        self._record_cache_bytes += self._record_nbytes(record)
        while self._record_cache_bytes > self._record_cache_limit:
            removable = next(
                (support for support in self.records if support != EMPTY_SUPPORT),
                None,
            )
            if removable is None:
                break
            removed = self.records.pop(removable)
            self._record_cache_bytes -= self._record_nbytes(removed)
        return record

    def release_search_caches(self) -> None:
        self.engine.clear_caches()
        self._addition_cache.clear()
        self._pricing_state.clear()
        self._pricing_cache_bytes = 0
        self._relaxed_upper_cache.clear()
        self._directional_upper_cache.clear()
        self._block_price_cache.clear()
        self._dual_geometry_cache.clear()
        baseline = self.records.get(EMPTY_SUPPORT)
        self.records.clear()
        self._record_cache_bytes = 0
        if baseline is not None:
            self._retain_record(baseline)

    def _structurally_admissible_dictionary(
        self, all_skeletons: tuple[Antecedent, ...]
    ) -> tuple[tuple[Antecedent, ...], tuple[RuleIdentity, ...]]:
        """Remove only empty atoms and exactly response-equivalent W values."""
        admitted_skeletons: list[Antecedent] = []
        admitted_rules: list[RuleIdentity] = []
        for antecedent in all_skeletons:
            windows = (0,) if len(antecedent) == 1 else self.config.formation_windows
            effective_windows = list(
                self.engine.effective_windows(self.context, antecedent, windows)
            )
            self.diagnostics.equivalent_window_identities += 2 * (
                len(windows) - len(effective_windows)
            )
            if not effective_windows:
                self.diagnostics.empty_skeletons += 1
                continue
            admitted_skeletons.append(antecedent)
            for window in effective_windows:
                admitted_rules.extend(
                    (
                        RuleIdentity(antecedent, window, -1),
                        RuleIdentity(antecedent, window, 1),
                    )
                )
        self.diagnostics.admissible_skeletons = len(admitted_skeletons)
        return tuple(admitted_skeletons), tuple(admitted_rules)

    def _saturated_nll_lower_bound(self) -> float:
        if self.context.dataset.likelihood == "first_event_cloglog":
            return 0.0
        counts = self.context.target_counts.astype(np.float64, copy=False)
        positive = counts > 0
        exposure = 1.0 / self.context.dataset.ticks_per_unit
        return float(
            np.sum(counts[positive] * (1.0 - np.log(counts[positive] / exposure)))
        )

    @staticmethod
    def _block_map(
        matrix: ModelMatrix,
    ) -> tuple[dict[object, slice], dict[object, slice]]:
        closure_map: dict[object, slice] = {}
        left = matrix.free_dimension - matrix.closure_dimension
        if matrix.rule_slices:
            knot_count = matrix.rule_slices[0].stop - matrix.rule_slices[0].start
        elif matrix.closure:
            knot_count = matrix.closure_dimension // len(matrix.closure)
        else:
            knot_count = 0
        for index, term in enumerate(matrix.closure):
            closure_map[term] = slice(
                left + index * knot_count, left + (index + 1) * knot_count
            )
        rule_map = {
            rule: block
            for rule, block in zip(
                matrix.support.rules, matrix.rule_slices, strict=True
            )
        }
        return closure_map, rule_map

    def warm_start(self, source: SupportRecord, target: ModelMatrix) -> np.ndarray:
        output = np.zeros(target.dimension, dtype=np.float64)
        baseline = min(
            self.baseline_dimension, len(source.fit.coefficients), target.dimension
        )
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
        output[target.free_dimension :] = np.maximum(
            output[target.free_dimension :], 0.0
        )
        return output

    def fit(
        self, support: Support, source: SupportRecord | None = None
    ) -> SupportRecord:
        with self._state_lock:
            cached = self.records.get(support)
            if cached is not None:
                self.records.move_to_end(support)
                self.diagnostics.fit_cache_hits += 1
                return cached
            stored = self._stored_records.get(support)
        if stored is not None:
            matrix = self.engine.model_matrix(self.context, support)
            record = SupportRecord(
                support, matrix, stored.fit, stored.penalty, stored.score
            )
            with self._state_lock:
                self.diagnostics.fit_cache_hits += 1
                return self._retain_record(record)
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
        score = (
            support_score(
                baseline_nll=self.baseline_nll, fit_nll=fit.nll, penalty=penalty
            )
            if fit.converged
            else -math.inf
        )
        record = SupportRecord(support, matrix, fit, penalty, score)
        with self._state_lock:
            # Parallel proposal waves are unique by construction, but this
            # race check also makes the public method safe for repeated jobs.
            existing = self._stored_records.get(support)
            if existing is not None:
                record = SupportRecord(
                    support, matrix, existing.fit, existing.penalty, existing.score
                )
                self.diagnostics.fit_cache_hits += 1
            else:
                self._stored_records[support] = _StoredRecord(fit, penalty, score)
                self.diagnostics.exact_fits += 1
            return self._retain_record(record)

    def fit_many(
        self,
        supports: list[Support] | tuple[Support, ...],
        source: SupportRecord | None = None,
    ) -> list[SupportRecord]:
        """Fit independent supports concurrently and return input order.

        Each worker invokes the identical float64 fixed-support solver.  The
        deterministic input order is restored before any incumbent decision,
        so parallel scheduling cannot change the selected support.
        """
        ordered = list(supports)
        if len(ordered) <= 1 or self.config.exact_workers == 1:
            return [self.fit(support, source) for support in ordered]
        with self._state_lock:
            self.diagnostics.parallel_exact_batches += 1
        threads_per_fit = max(
            1, self.config.pricing_workers // self.config.exact_workers
        )

        def fit_one(support: Support) -> SupportRecord:
            configure_cpu_threads(threads_per_fit)
            return self.fit(support, source)

        with ThreadPoolExecutor(
            max_workers=min(self.config.exact_workers, len(ordered)),
            thread_name_prefix="crbstpp-exact",
        ) as executor:
            return list(executor.map(fit_one, ordered))

    def fit_fixed(
        self,
        support: Support,
        closure: tuple[ClosureTerm, ...],
    ) -> tuple[ModelMatrix, FitResult]:
        """Fit a forced-closure model with a shared certification cache."""
        key = (support, tuple(closure))
        with self._state_lock:
            cached = self._forced_fits.get(key)
            natural = self._stored_records.get(support)
            if cached is None and tuple(closure) == hierarchy_closure(support):
                cached = None if natural is None else natural.fit
                if cached is not None:
                    self._forced_fits[key] = cached
        matrix = self.engine.model_matrix(
            self.context, support, forced_closure=tuple(closure)
        )
        if cached is not None:
            with self._state_lock:
                self.diagnostics.forced_fit_cache_hits += 1
            return matrix, cached
        fit = fit_model_matrix(
            matrix,
            likelihood=self.context.dataset.likelihood,
            tolerance=self.config.solver_tolerance,
            max_iter=self.config.solver_max_iter,
        )
        with self._state_lock:
            incumbent = self._forced_fits.setdefault(key, fit)
        return matrix, incumbent

    def fit_fixed_many(
        self,
        specifications: list[tuple[Support, tuple[ClosureTerm, ...]]],
    ) -> list[tuple[ModelMatrix, FitResult]]:
        if len(specifications) <= 1 or self.config.exact_workers == 1:
            return [self.fit_fixed(*specification) for specification in specifications]
        with self._state_lock:
            self.diagnostics.parallel_exact_batches += 1
        threads_per_fit = max(
            1, self.config.pricing_workers // self.config.exact_workers
        )

        def fit_one(
            item: tuple[Support, tuple[ClosureTerm, ...]],
        ) -> tuple[ModelMatrix, FitResult]:
            configure_cpu_threads(threads_per_fit)
            return self.fit_fixed(*item)

        with ThreadPoolExecutor(
            max_workers=min(self.config.exact_workers, len(specifications)),
            thread_name_prefix="crbstpp-fixed",
        ) as executor:
            return list(executor.map(fit_one, specifications))

    def saturated_upper_score(self, support: Support) -> float:
        return support_score(
            baseline_nll=self.baseline_nll,
            fit_nll=self.saturated_nll_lower_bound,
            penalty=self.objective.structural_penalty(support),
        )

    def _feature_rows(self, support: Support) -> np.ndarray:
        parts = [
            self.engine.response_rows(self.context, term.antecedent, term.window)
            for term in hierarchy_closure(support)
        ]
        parts.extend(
            self.engine.response_rows(self.context, rule.antecedent, rule.window)
            for rule in support.rules
        )
        nonempty = [rows for rows in parts if len(rows)]
        result = (
            np.unique(np.concatenate(nonempty))
            if nonempty
            else np.zeros(0, dtype=np.int64)
        )
        return result

    def localized_upper_score(self, support: Support) -> float:
        """Safe score upper bound from a localized saturated relaxation.

        Every row touched by any support or closure feature receives its own
        unrestricted predictor; untouched rows retain one common intercept.
        This relaxed model contains the exact support model, so its minimum
        NLL is a rigorous lower bound on the exact minimum NLL.
        """
        key = self._unsigned_geometry_key(support)
        with self._state_lock:
            if key in self._relaxed_upper_cache:
                cached = self._relaxed_upper_cache[key]
                self._relaxed_upper_cache.move_to_end(key)
                return cached
        affected = self._feature_rows(support)
        score = self._localized_score_from_rows(support, affected)
        with self._state_lock:
            self._relaxed_upper_cache[key] = score
            while len(self._relaxed_upper_cache) > self._relaxed_upper_limit:
                self._relaxed_upper_cache.popitem(last=False)
        return score

    def _localized_score_from_rows(
        self, support: Support, affected: np.ndarray
    ) -> float:
        positions = np.searchsorted(self.context.target_rows, affected)
        matched = positions < len(self.context.target_rows)
        if len(self.context.target_rows):
            safe = np.minimum(positions, len(self.context.target_rows) - 1)
            matched &= self.context.target_rows[safe] == affected
        affected_counts = (
            self.context.target_counts[positions[matched]]
            if np.any(matched)
            else np.zeros(0, dtype=np.float64)
        )
        total_events = float(np.sum(self.context.target_counts))
        affected_events = float(np.sum(affected_counts))
        remaining_rows = int(self.context.n_grid - len(affected))
        remaining_events = total_events - affected_events
        if self.context.dataset.likelihood == "poisson":
            exposure = self.engine.tick_exposure
            positive = affected_counts > 0
            affected_lower = float(
                np.sum(
                    affected_counts[positive]
                    * (1.0 - np.log(affected_counts[positive] / exposure))
                )
            )
            remaining_exposure = exposure * remaining_rows
            if remaining_events > 0 and remaining_exposure > 0:
                remaining_lower = remaining_events * (
                    1.0 - math.log(remaining_events / remaining_exposure)
                )
            else:
                remaining_lower = 0.0
        else:
            affected_lower = 0.0
            remaining_noevents = remaining_rows - remaining_events
            if remaining_events > 0 and remaining_noevents > 0:
                total = remaining_events + remaining_noevents
                probability = remaining_events / total
                remaining_lower = -remaining_events * math.log(probability)
                remaining_lower -= remaining_noevents * math.log1p(-probability)
            else:
                remaining_lower = 0.0
        lower_nll = affected_lower + remaining_lower
        score = support_score(
            baseline_nll=self.baseline_nll,
            fit_nll=lower_nll,
            penalty=self.objective.structural_penalty(support),
        )
        return min(score, self.saturated_upper_score(support))

    @staticmethod
    def _union_rows(parts: list[np.ndarray]) -> np.ndarray:
        nonempty = [part for part in parts if len(part)]
        return (
            np.unique(np.concatenate(nonempty))
            if nonempty
            else np.zeros(0, dtype=np.int64)
        )

    def _event_count_at_rows(self, rows: np.ndarray) -> float:
        if not len(rows) or not len(self.context.target_rows):
            return 0.0
        positions = np.searchsorted(self.context.target_rows, rows)
        matched = positions < len(self.context.target_rows)
        safe = np.minimum(positions, len(self.context.target_rows) - 1)
        matched &= self.context.target_rows[safe] == rows
        return float(np.sum(self.context.target_counts[positions[matched]]))

    def directional_upper_score(self, support: Support) -> float:
        """Safe sign-aware saturated-relaxation upper bound for cloglog.

        Closure rows and rows touched by both signs are allowed arbitrary
        predictors.  Excitation-only rows are allowed any predictor above the
        common intercept and inhibition-only rows any predictor below it.
        This strictly contains the rule model while retaining the sign cone,
        and its profiled first-event likelihood has a closed-form aggregate
        optimum.  No candidate capable of improving J is removed.
        """
        if self.context.dataset.likelihood != "first_event_cloglog":
            return self.localized_upper_score(support)
        with self._state_lock:
            cached = self._directional_upper_cache.get(support)
            if cached is not None:
                self._directional_upper_cache.move_to_end(support)
                return cached
        closure_rows = self._union_rows(
            [
                self.engine.response_rows(self.context, term.antecedent, term.window)
                for term in hierarchy_closure(support)
            ]
        )
        excitation = self._union_rows(
            [
                self.engine.response_rows(self.context, rule.antecedent, rule.window)
                for rule in support.rules
                if rule.sign > 0
            ]
        )
        inhibition = self._union_rows(
            [
                self.engine.response_rows(self.context, rule.antecedent, rule.window)
                for rule in support.rules
                if rule.sign < 0
            ]
        )
        score = self._directional_score_from_rows(
            support, closure_rows, excitation, inhibition
        )
        with self._state_lock:
            self._directional_upper_cache[support] = score
            while len(self._directional_upper_cache) > self._relaxed_upper_limit:
                self._directional_upper_cache.popitem(last=False)
        return score

    def _directional_score_from_rows(
        self,
        support: Support,
        closure_rows: np.ndarray,
        excitation: np.ndarray,
        inhibition: np.ndarray,
    ) -> float:
        opposite_overlap = np.intersect1d(excitation, inhibition, assume_unique=True)
        free_rows = np.union1d(closure_rows, opposite_overlap)
        excitation_only = np.setdiff1d(excitation, free_rows, assume_unique=True)
        inhibition_only = np.setdiff1d(inhibition, free_rows, assume_unique=True)
        free_events = self._event_count_at_rows(free_rows)
        excitation_events = self._event_count_at_rows(excitation_only)
        inhibition_events = self._event_count_at_rows(inhibition_only)
        total_events = float(np.sum(self.context.target_counts))
        total_noevents = float(self.context.n_grid) - total_events
        free_noevents = float(len(free_rows)) - free_events
        inhibition_noevents = float(len(inhibition_only)) - inhibition_events
        # Excitation-only event rows and inhibition-only no-event rows achieve
        # their saturated limit.  Their opposite outcomes remain tied to the
        # common intercept and therefore cannot be discarded by the bound.
        effective_events = max(0.0, total_events - free_events - excitation_events)
        effective_noevents = max(
            0.0, total_noevents - free_noevents - inhibition_noevents
        )
        if effective_events > 0 and effective_noevents > 0:
            total = effective_events + effective_noevents
            probability = effective_events / total
            lower_nll = -effective_events * math.log(probability)
            lower_nll -= effective_noevents * math.log1p(-probability)
        else:
            lower_nll = 0.0
        score = support_score(
            baseline_nll=self.baseline_nll,
            fit_nll=lower_nll,
            penalty=self.objective.structural_penalty(support),
        )
        return min(score, self.localized_upper_score(support))

    def safe_upper_score(self, support: Support) -> float:
        return min(
            self.localized_upper_score(support),
            self.directional_upper_score(support),
        )

    def bounds(self, current: SupportRecord, trial: Support) -> ProposalBounds:
        """Return a feasible-primal lower score and verified-dual upper score."""
        with self._state_lock:
            self.diagnostics.bound_evaluations += 1
        matrix = self.engine.model_matrix(self.context, trial)
        warm = self.warm_start(current, matrix)
        eta = matrix.x @ warm
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
        geometry_key = (
            tuple((rule.antecedent, rule.window) for rule in trial.rules),
            matrix.closure,
        )
        with self._state_lock:
            geometry = self._dual_geometry_cache.get(geometry_key)
            if geometry is not None:
                self._dual_geometry_cache.move_to_end(geometry_key)
                self.diagnostics.dual_geometry_cache_hits += 1
        if geometry is None:
            geometry = dual_geometry(matrix)
            with self._state_lock:
                self._dual_geometry_cache[geometry_key] = geometry
                while len(self._dual_geometry_cache) > self._dual_geometry_cache_limit:
                    self._dual_geometry_cache.popitem(last=False)
        certificate = dual_certificate(
            matrix,
            likelihood=self.context.dataset.likelihood,
            beta=warm,
            tolerance=min(1.0e-9, self.config.solver_tolerance * 0.01),
            max_iter=3,
            geometry=geometry,
        )
        if not certificate.feasible:
            # Three damped projected-Newton splices remain a feasible
            # primal point and usually places the score dual inside the new
            # closure/cone geometry.  Retry it after three cheap projection
            # sweeps instead of wasting hundreds of alternating projections
            # at the baseline warm start; it is not an acceptance shortcut.
            splice_fit = fit_model_matrix(
                matrix,
                likelihood=self.context.dataset.likelihood,
                tolerance=self.config.solver_tolerance,
                max_iter=3,
                warm_start=warm,
            )
            one_eta = matrix.x @ splice_fit.coefficients
            one_rows, _, _ = loss_rows(
                one_eta,
                likelihood=self.context.dataset.likelihood,
                exposure_weight=matrix.exposure_weight,
                noevent_weight=matrix.noevent_weight,
                event_weight=matrix.event_weight,
            )
            one_nll = float(np.sum(one_rows))
            if one_nll < feasible_nll:
                warm = splice_fit.coefficients
                feasible_nll = one_nll
                lower_score = support_score(
                    baseline_nll=self.baseline_nll,
                    fit_nll=feasible_nll,
                    penalty=penalty,
                )
            certificate = dual_certificate(
                matrix,
                likelihood=self.context.dataset.likelihood,
                beta=warm,
                tolerance=min(1.0e-9, self.config.solver_tolerance * 0.01),
                geometry=geometry,
            )
        if (
            certificate.feasible
            and certificate.nll_lower_bound
            <= feasible_nll + 1.0e-7 * max(1.0, abs(feasible_nll))
        ):
            upper_score = support_score(
                baseline_nll=self.baseline_nll,
                fit_nll=certificate.nll_lower_bound,
                penalty=penalty,
            )
        else:
            with self._state_lock:
                self.diagnostics.dual_fail_open += 1
            upper_score = self.safe_upper_score(trial)
            certificate = None
        # The analytic saturated likelihood bound is always valid and can
        # tighten a loose numerical Fenchel certificate.
        upper_score = min(upper_score, self.safe_upper_score(trial))
        if upper_score + 1.0e-7 * max(1.0, abs(upper_score)) < lower_score:
            raise AssertionError("proposal primal/dual sandwich is invalid")
        return ProposalBounds(
            trial,
            lower_score,
            upper_score,
            lower_score - current.score,
            upper_score - current.score,
            certificate,
            warm,
        )

    @staticmethod
    def _unsigned_geometry_key(support: Support) -> tuple:
        return tuple((rule.antecedent, rule.window) for rule in support.rules)

    def bounds_many(
        self, current: SupportRecord, trials: list[Support]
    ) -> list[ProposalBounds]:
        """Evaluate bounds in deterministic geometry-reuse order.

        Each bound already parallelizes its tall matrix reductions over the
        physical cores.  Running several memory-bandwidth-bound projections
        concurrently was measured to more than double wall time on the target
        workstation.  Serial sign-paired evaluation also maximizes reuse of
        the cached unsigned dual geometry without changing any certificate.
        """
        configure_cpu_threads(self.config.pricing_workers)
        return [self.bounds(current, trial) for trial in trials]

    def _pricing_components(
        self, current: SupportRecord
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self._pricing_state.get(current.support)
        if cached is not None:
            self._pricing_state.move_to_end(current.support)
            return cached
        eta = current.matrix.x @ current.fit.coefficients
        _, _, second = loss_rows(
            eta,
            likelihood=self.context.dataset.likelihood,
            exposure_weight=current.matrix.exposure_weight,
            noevent_weight=current.matrix.noevent_weight,
            event_weight=current.matrix.event_weight,
        )
        _, hessian = moments(
            current.matrix.x, np.zeros_like(second), second, device="cpu"
        )
        active = np.arange(current.matrix.dimension) < current.matrix.free_dimension
        active |= current.fit.coefficients > 1.0e-10
        indices = np.flatnonzero(active)
        inverse = np.linalg.pinv(
            hessian[np.ix_(indices, indices)],
            rcond=max(1.0e-12, self.config.solver_tolerance),
        )
        result = indices, inverse
        self._pricing_state[current.support] = result
        self._pricing_cache_bytes += indices.nbytes + inverse.nbytes
        while self._pricing_cache_bytes > self._pricing_cache_limit:
            _, removed = self._pricing_state.popitem(last=False)
            self._pricing_cache_bytes -= removed[0].nbytes + removed[1].nbytes
        return result

    def _price_skeleton(
        self,
        current: SupportRecord,
        antecedent: Antecedent,
        windows: tuple[int, ...],
        *,
        device: str,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Compute every requested W and both signs from one derivative pass.

        Blocks are nested in W.  The current-model design and likelihood
        derivatives are therefore evaluated once on the largest footprint;
        each W uses an exactly zero-padded view on that common row set.  The
        unsigned gradient changes sign algebraically while its conditional
        Fisher matrix does not, eliminating the second sign pass exactly.
        """
        requested = tuple(sorted(set(int(window) for window in windows)))
        cached: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        missing: list[int] = []
        with self._state_lock:
            for window in requested:
                key = (current.support, antecedent, window)
                value = self._block_price_cache.get(key)
                if value is None:
                    missing.append(window)
                else:
                    self._block_price_cache.move_to_end(key)
                    cached[window] = value
        if not missing:
            return cached
        blocks = self.engine.blocks_many(self.context, antecedent, requested)
        maximum = blocks[max(requested)]
        if not len(maximum.rows):
            zeros = np.zeros(self.config.knot_count, dtype=np.float64)
            zero_matrix = np.zeros(
                (self.config.knot_count, self.config.knot_count), dtype=np.float64
            )
            computed = {
                window: (zeros.copy(), zero_matrix.copy()) for window in missing
            }
        else:
            indices, inverse = self._pricing_components(current)
            gradients = {
                window: np.zeros(self.config.knot_count, dtype=np.float64)
                for window in missing
            }
            hessians = {
                window: np.zeros(
                    (self.config.knot_count, self.config.knot_count),
                    dtype=np.float64,
                )
                for window in missing
            }
            crosses = {
                window: np.zeros(
                    (len(indices), self.config.knot_count), dtype=np.float64
                )
                for window in missing
            }
            for start in range(0, len(maximum.rows), 131_072):
                end = min(len(maximum.rows), start + 131_072)
                rows = maximum.rows[start:end]
                design = self.engine.design_at_rows_with_context(
                    self.context, current.matrix, rows
                )
                eta = design @ current.fit.coefficients
                positions = np.searchsorted(self.context.target_rows, rows)
                matched = positions < len(self.context.target_rows)
                if len(self.context.target_rows):
                    safe = np.minimum(positions, len(self.context.target_rows) - 1)
                    matched &= self.context.target_rows[safe] == rows
                event = np.zeros(len(rows), dtype=np.float64)
                if np.any(matched):
                    event[matched] = self.context.target_counts[positions[matched]]
                exposure = np.full(
                    len(rows), self.engine.tick_exposure, dtype=np.float64
                )
                noevent = (
                    exposure - event
                    if self.context.dataset.likelihood == "first_event_cloglog"
                    else exposure
                )
                _, first, second = loss_rows(
                    eta,
                    likelihood=self.context.dataset.likelihood,
                    exposure_weight=exposure,
                    noevent_weight=noevent,
                    event_weight=event,
                )
                candidates = []
                for window in missing:
                    block = blocks[window]
                    candidate = np.zeros(
                        (len(rows), self.config.knot_count), dtype=np.float64
                    )
                    left = int(np.searchsorted(block.rows, rows[0], side="left"))
                    right = int(np.searchsorted(block.rows, rows[-1], side="right"))
                    if right > left:
                        block_rows = block.rows[left:right]
                        candidate[np.searchsorted(rows, block_rows)] = block.values[
                            left:right
                        ]
                    candidates.append(candidate)
                candidate_batch = np.asarray(candidates, dtype=np.float64)
                tile_gradients, tile_hessians = moments_batch(
                    candidate_batch, first, second, device=device
                )
                if len(indices):
                    flat_candidates = candidate_batch.transpose(1, 0, 2).reshape(
                        len(rows), -1
                    )
                    flat_cross = design[:, indices].T @ (
                        second[:, None] * flat_candidates
                    )
                    tile_crosses = flat_cross.reshape(
                        len(indices), len(missing), self.config.knot_count
                    ).transpose(1, 0, 2)
                for window_index, window in enumerate(missing):
                    tile_gradient = tile_gradients[window_index]
                    tile_hessian = tile_hessians[window_index]
                    gradients[window] += tile_gradient
                    hessians[window] += tile_hessian
                    if len(indices):
                        crosses[window] += tile_crosses[window_index]
            computed = {}
            for window in missing:
                hessian = hessians[window]
                if len(indices):
                    cross = crosses[window]
                    hessian = hessian - cross.T @ inverse @ cross
                eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
                scale = max(1.0, float(np.max(np.abs(eigenvalues))))
                eigenvalues = np.maximum(eigenvalues, scale * 1.0e-12)
                conditioned = (eigenvectors * eigenvalues) @ eigenvectors.T
                computed[window] = (gradients[window], conditioned)
        with self._state_lock:
            self.diagnostics.fused_skeleton_pricing_passes += 1
            for window, value in computed.items():
                key = (current.support, antecedent, window)
                self._block_price_cache[key] = value
                self._block_price_cache.move_to_end(key)
            while len(self._block_price_cache) > self._block_price_cache_limit:
                self._block_price_cache.popitem(last=False)
        return {**cached, **computed}

    def block_price(
        self, current: SupportRecord, rule: RuleIdentity, *, device: str = "cpu"
    ) -> float:
        """Conditional-Fisher price used solely for deterministic work ordering."""
        gradient, hessian = self._price_skeleton(
            current,
            rule.antecedent,
            (rule.window,),
            device=device,
        )[rule.window]
        return _nonnegative_quadratic_gain(float(rule.sign) * gradient, hessian)

    def _legacy_block_price_components(
        self, current: SupportRecord, rule: RuleIdentity, *, device: str = "cpu"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unfused reference retained for numerical parity tests."""
        block = self.engine.block(self.context, rule.antecedent, rule.window)
        if not len(block.rows):
            return (
                np.zeros(self.config.knot_count, dtype=np.float64),
                np.zeros(
                    (self.config.knot_count, self.config.knot_count),
                    dtype=np.float64,
                ),
            )
        indices, inverse = self._pricing_components(current)
        gradient = np.zeros(self.config.knot_count, dtype=np.float64)
        hessian = np.zeros(
            (self.config.knot_count, self.config.knot_count), dtype=np.float64
        )
        cross = np.zeros((len(indices), self.config.knot_count), dtype=np.float64)
        # Bound host/device working memory independently of the footprint size.
        for start in range(0, len(block.rows), 131_072):
            end = min(len(block.rows), start + 131_072)
            rows = block.rows[start:end]
            design = self.engine.design_at_rows_with_context(
                self.context, current.matrix, rows
            )
            eta = design @ current.fit.coefficients
            positions = np.searchsorted(self.context.target_rows, rows)
            matched = positions < len(self.context.target_rows)
            if len(self.context.target_rows):
                safe = np.minimum(positions, len(self.context.target_rows) - 1)
                matched &= self.context.target_rows[safe] == rows
            event = np.zeros(len(rows), dtype=np.float64)
            if np.any(matched):
                event[matched] = self.context.target_counts[positions[matched]]
            exposure = np.full(len(rows), self.engine.tick_exposure, dtype=np.float64)
            noevent = (
                exposure - event
                if self.context.dataset.likelihood == "first_event_cloglog"
                else exposure
            )
            _, first, second = loss_rows(
                eta,
                likelihood=self.context.dataset.likelihood,
                exposure_weight=exposure,
                noevent_weight=noevent,
                event_weight=event,
            )
            candidate = block.values[start:end]
            tile_gradient, tile_hessian = moments(
                candidate, first, second, device=device
            )
            gradient += tile_gradient
            hessian += tile_hessian
            if len(indices):
                cross += design[:, indices].T @ (second[:, None] * candidate)
        if len(indices):
            hessian -= cross.T @ inverse @ cross
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        eigenvalues = np.maximum(eigenvalues, scale * 1.0e-12)
        conditioned = (eigenvectors * eigenvalues) @ eigenvectors.T
        return gradient, conditioned

    def _rank_identities(
        self, current: SupportRecord, identities: tuple[RuleIdentity, ...]
    ) -> list[tuple[float, RuleIdentity]]:
        self.diagnostics.pricing_passes += 1
        self.diagnostics.priced_blocks += len(identities)
        # Build the common conditional information once before worker launch.
        self._pricing_components(current)
        grouped: dict[Antecedent, set[int]] = {}
        for rule in identities:
            grouped.setdefault(rule.antecedent, set()).add(rule.window)
        groups = [
            (antecedent, tuple(sorted(windows)))
            for antecedent, windows in sorted(grouped.items())
        ]
        devices = self.config.pricing_devices
        if devices and len(groups) > 1:

            def price(
                index_group: tuple[int, tuple[Antecedent, tuple[int, ...]]],
            ) -> tuple[Antecedent, dict[int, tuple[np.ndarray, np.ndarray]]]:
                configure_cpu_threads(1)
                index, (antecedent, windows) = index_group
                device = devices[index % len(devices)]
                return antecedent, self._price_skeleton(
                    current, antecedent, windows, device=device
                )

            with ThreadPoolExecutor(
                max_workers=min(self.config.pricing_workers, len(groups)),
                thread_name_prefix="crbstpp-pricing",
            ) as executor:
                component_items = list(executor.map(price, enumerate(groups)))
        else:
            device = devices[0] if devices else "cpu"
            component_items = [
                (
                    antecedent,
                    self._price_skeleton(current, antecedent, windows, device=device),
                )
                for antecedent, windows in groups
            ]
        components = dict(component_items)
        prices = [
            (
                _nonnegative_quadratic_gain(
                    float(rule.sign) * components[rule.antecedent][rule.window][0],
                    components[rule.antecedent][rule.window][1],
                ),
                rule,
            )
            for rule in identities
        ]
        with self._state_lock:
            self.diagnostics.fused_sign_prices += len(identities) - sum(
                len(windows) for _, windows in groups
            )
        return sorted(prices, key=lambda item: (-item[0], item[1]))

    def _hierarchy_quadratic_gain(
        self,
        gradient: np.ndarray,
        hessian: np.ndarray,
        closure_dimension: int,
        sign: int,
    ) -> tuple[float, float]:
        """Return total and rule-only gains after profiling closure nuisance."""
        gradient = np.asarray(gradient, dtype=np.float64)
        hessian = 0.5 * (
            np.asarray(hessian, dtype=np.float64)
            + np.asarray(hessian, dtype=np.float64).T
        )
        closure_dimension = int(closure_dimension)
        closure_gain = 0.0
        rule_gradient = gradient[closure_dimension:]
        rule_hessian = hessian[closure_dimension:, closure_dimension:]
        if closure_dimension:
            closure_hessian = hessian[:closure_dimension, :closure_dimension]
            eigenvalues, eigenvectors = np.linalg.eigh(closure_hessian)
            scale = max(1.0, float(np.max(np.abs(eigenvalues))))
            keep = eigenvalues > scale * max(1.0e-12, self.config.solver_tolerance)
            inverse = (
                (eigenvectors[:, keep] / eigenvalues[keep]) @ eigenvectors[:, keep].T
                if np.any(keep)
                else np.zeros_like(closure_hessian)
            )
            closure_gradient = gradient[:closure_dimension]
            projected = closure_hessian @ (inverse @ closure_gradient)
            residual = closure_gradient - projected
            if np.linalg.norm(residual, ord=np.inf) > math.sqrt(
                self.config.solver_tolerance
            ) * max(1.0, np.linalg.norm(closure_gradient, ord=np.inf)):
                return math.inf, math.inf
            closure_gain = max(
                0.0,
                0.5 * float(closure_gradient @ inverse @ closure_gradient),
            )
            cross = hessian[:closure_dimension, closure_dimension:]
            rule_gradient = rule_gradient - cross.T @ inverse @ closure_gradient
            rule_hessian = rule_hessian - cross.T @ inverse @ cross
        rule_gain = _nonnegative_quadratic_gain(
            float(sign) * rule_gradient, rule_hessian
        )
        return closure_gain + rule_gain, rule_gain

    def _price_hierarchy_skeleton(
        self,
        current: SupportRecord,
        antecedent: Antecedent,
        windows: tuple[int, ...],
        *,
        device: str,
    ) -> dict[int, tuple[float, float, float, float, int, bool]]:
        """Joint closure+rule prices for every W and both signs in one pass."""
        requested = tuple(sorted(set(map(int, windows))))
        old_closure = set(current.matrix.closure)
        specifications: dict[int, tuple[tuple[ClosureTerm, ...], bool]] = {}
        for window in requested:
            trial = current.support.add(RuleIdentity(antecedent, window, 1))
            trial_closure = set(hierarchy_closure(trial))
            nested = old_closure.issubset(trial_closure)
            additions = tuple(sorted(trial_closure - old_closure)) if nested else ()
            specifications[window] = additions, nested
        nested_windows = tuple(
            window for window in requested if specifications[window][1]
        )
        output = {
            window: (math.inf, math.inf, math.inf, math.inf, 0, False)
            for window in requested
            if not specifications[window][1]
        }
        if not nested_windows:
            return output

        block_requests: dict[Antecedent, set[int]] = {antecedent: set(nested_windows)}
        for window in nested_windows:
            for term in specifications[window][0]:
                block_requests.setdefault(term.antecedent, set()).add(term.window)
        blocks: dict[tuple[Antecedent, int], SparseBlock] = {}
        for block_antecedent, block_windows in sorted(block_requests.items()):
            built = self.engine.blocks_many(
                self.context, block_antecedent, tuple(sorted(block_windows))
            )
            blocks.update(
                ((block_antecedent, window), block) for window, block in built.items()
            )
        maximal_blocks = [
            blocks[(block_antecedent, max(block_windows))]
            for block_antecedent, block_windows in sorted(block_requests.items())
        ]
        nonempty_rows = [block.rows for block in maximal_blocks if len(block.rows)]
        maximum_rows = (
            np.unique(np.concatenate(nonempty_rows))
            if nonempty_rows
            else np.zeros(0, dtype=np.int64)
        )
        dimensions = {
            window: (len(specifications[window][0]) + 1) * self.config.knot_count
            for window in nested_windows
        }
        maximum_dimension = max(dimensions.values())
        gradients = {
            window: np.zeros(maximum_dimension, dtype=np.float64)
            for window in nested_windows
        }
        hessians = {
            window: np.zeros((maximum_dimension, maximum_dimension), dtype=np.float64)
            for window in nested_windows
        }
        indices, inverse = self._pricing_components(current)
        crosses = {
            window: np.zeros((len(indices), maximum_dimension), dtype=np.float64)
            for window in nested_windows
        }
        batch_size = len(nested_windows)
        tile_rows = max(
            1_024,
            min(
                65_536,
                128 * 1024**2 // max(8, 8 * batch_size * maximum_dimension),
            ),
        )

        def insert(
            destination: np.ndarray,
            rows: np.ndarray,
            block: SparseBlock,
            column: int,
        ) -> None:
            if not len(block.rows) or not len(rows):
                return
            left = int(np.searchsorted(block.rows, rows[0], side="left"))
            right = int(np.searchsorted(block.rows, rows[-1], side="right"))
            if right <= left:
                return
            block_rows = block.rows[left:right]
            destination[
                np.searchsorted(rows, block_rows),
                column : column + self.config.knot_count,
            ] = block.values[left:right]

        for start in range(0, len(maximum_rows), tile_rows):
            rows = maximum_rows[start : start + tile_rows]
            design = self.engine.design_at_rows_with_context(
                self.context, current.matrix, rows
            )
            eta = design @ current.fit.coefficients
            positions = np.searchsorted(self.context.target_rows, rows)
            matched = positions < len(self.context.target_rows)
            if len(self.context.target_rows):
                safe = np.minimum(positions, len(self.context.target_rows) - 1)
                matched &= self.context.target_rows[safe] == rows
            event = np.zeros(len(rows), dtype=np.float64)
            if np.any(matched):
                event[matched] = self.context.target_counts[positions[matched]]
            exposure = np.full(len(rows), self.engine.tick_exposure, dtype=np.float64)
            noevent = (
                exposure - event
                if self.context.dataset.likelihood == "first_event_cloglog"
                else exposure
            )
            _, first, second = loss_rows(
                eta,
                likelihood=self.context.dataset.likelihood,
                exposure_weight=exposure,
                noevent_weight=noevent,
                event_weight=event,
            )
            candidate_batch = np.zeros(
                (batch_size, len(rows), maximum_dimension), dtype=np.float64
            )
            for batch_index, window in enumerate(nested_windows):
                closure = specifications[window][0]
                for term_index, term in enumerate(closure):
                    insert(
                        candidate_batch[batch_index],
                        rows,
                        blocks[(term.antecedent, term.window)],
                        term_index * self.config.knot_count,
                    )
                insert(
                    candidate_batch[batch_index],
                    rows,
                    blocks[(antecedent, window)],
                    len(closure) * self.config.knot_count,
                )
            tile_gradients, tile_hessians = moments_batch(
                candidate_batch, first, second, device=device
            )
            if len(indices):
                flat = candidate_batch.transpose(1, 0, 2).reshape(len(rows), -1)
                flat_cross = design[:, indices].T @ (second[:, None] * flat)
                tile_crosses = flat_cross.reshape(
                    len(indices), batch_size, maximum_dimension
                ).transpose(1, 0, 2)
            for batch_index, window in enumerate(nested_windows):
                gradients[window] += tile_gradients[batch_index]
                hessians[window] += tile_hessians[batch_index]
                if len(indices):
                    crosses[window] += tile_crosses[batch_index]

        for window in nested_windows:
            dimension = dimensions[window]
            gradient = gradients[window][:dimension]
            hessian = hessians[window][:dimension, :dimension]
            if len(indices):
                cross = crosses[window][:, :dimension]
                hessian = hessian - cross.T @ inverse @ cross
            closure_dimension = dimension - self.config.knot_count
            negative_total, negative_rule = self._hierarchy_quadratic_gain(
                gradient, hessian, closure_dimension, -1
            )
            positive_total, positive_rule = self._hierarchy_quadratic_gain(
                gradient, hessian, closure_dimension, 1
            )
            output[window] = (
                negative_total,
                positive_total,
                negative_rule,
                positive_rule,
                closure_dimension,
                True,
            )
        with self._state_lock:
            self.diagnostics.fused_skeleton_pricing_passes += 1
        return output

    def _rank_block_identities(
        self, current: SupportRecord, identities: tuple[RuleIdentity, ...]
    ) -> list[tuple[float, float, RuleIdentity, bool]]:
        """Rank every identity by hierarchy-aware MDL-adjusted block score."""
        if not identities:
            return []
        with self._state_lock:
            self.diagnostics.pricing_passes += 1
            self.diagnostics.priced_blocks += len(identities)
            self.diagnostics.block_score_evaluations += len(identities)
        self._pricing_components(current)
        grouped: dict[Antecedent, set[int]] = {}
        for rule in identities:
            grouped.setdefault(rule.antecedent, set()).add(rule.window)
        groups = [
            (antecedent, tuple(sorted(windows)))
            for antecedent, windows in sorted(grouped.items())
        ]
        devices = self.config.pricing_devices

        def price_group(
            indexed: tuple[int, tuple[Antecedent, tuple[int, ...]]],
        ) -> tuple[
            Antecedent,
            dict[int, tuple[float, float, float, float, int, bool]],
        ]:
            configure_cpu_threads(1)
            index, (antecedent, windows) = indexed
            device = devices[index % len(devices)] if devices else "cpu"
            return antecedent, self._price_hierarchy_skeleton(
                current, antecedent, windows, device=device
            )

        if len(groups) > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.config.pricing_workers, len(groups)),
                thread_name_prefix="crbstpp-block-score",
            ) as executor:
                component_items = list(executor.map(price_group, enumerate(groups)))
        else:
            component_items = [price_group(item) for item in enumerate(groups)]
        components = dict(component_items)
        ranked: list[tuple[float, float, RuleIdentity, bool]] = []
        for rule in identities:
            (
                negative,
                positive,
                negative_rule,
                positive_rule,
                closure_dimension,
                nested,
            ) = components[rule.antecedent][rule.window]
            gain = positive if rule.sign > 0 else negative
            rule_gain = positive_rule if rule.sign > 0 else negative_rule
            trial = current.support.add(rule)
            penalty_delta = self.objective.structural_penalty(trial) - current.penalty
            closure_parameter_code = closure_dimension * math.log(
                max(2, self.objective.n_entities)
            )
            rule_penalty = penalty_delta - closure_parameter_code
            total_net = 2.0 * gain - penalty_delta
            rule_net = 2.0 * rule_gain - rule_penalty
            net = math.inf if not nested else min(total_net, rule_net)
            ranked.append((net, gain, rule, nested))
        with self._state_lock:
            self.diagnostics.fused_sign_prices += len(identities) - sum(
                len(windows) for _, windows in groups
            )
            self.diagnostics.block_score_screens += sum(
                nested and net <= self.config.search_tolerance
                for net, _, _, nested in ranked
            )
            self.diagnostics.block_score_admissible += sum(
                (not nested) or net > self.config.search_tolerance
                for net, _, _, nested in ranked
            )
            self.diagnostics.nonnested_exact_audits += sum(
                not nested for _, _, _, nested in ranked
            )
        for antecedent in sorted(grouped):
            candidates = [item for item in ranked if item[2].antecedent == antecedent]
            best = sorted(candidates, key=lambda item: (-item[0], item[2]))[0]
            self._skeleton_witnesses[antecedent] = (best[0], best[2])
        return sorted(ranked, key=lambda item: (-item[0], item[2]))

    def _inactive_identities(
        self, support: Support, antecedents: set[Antecedent] | None = None
    ) -> tuple[RuleIdentity, ...]:
        existing = set(support.antecedents)
        return tuple(
            rule
            for rule in self.dictionary
            if rule.antecedent not in existing
            and (antecedents is None or rule.antecedent in antecedents)
        )

    def _safe_identity_survivors(
        self,
        current: SupportRecord,
        identities: tuple[RuleIdentity, ...],
        threshold: float,
    ) -> tuple[RuleIdentity, ...]:
        survivors: list[RuleIdentity] = []
        scored: list[tuple[float, RuleIdentity]] = []
        # Every W of one skeleton has a nested response.  Expand the maximum
        # requested W once, then obtain all smaller exact footprints by a
        # threshold lookup.  Include closure requests up front so concurrent
        # standalone workers share their lower-order single-flight entries.
        row_requests: dict[Antecedent, set[int]] = {}

        def request(antecedent: Antecedent, window: int) -> None:
            row_requests.setdefault(antecedent, set()).add(int(window))

        for rule in current.support.rules:
            request(rule.antecedent, rule.window)
        for rule in identities:
            request(rule.antecedent, rule.window)
            for term in hierarchy_closure(current.support.add(rule)):
                request(term.antecedent, term.window)
        row_thresholds: dict[Antecedent, tuple[np.ndarray, np.ndarray]] = {}
        for antecedent, windows in sorted(row_requests.items()):
            row_thresholds[antecedent] = self.engine.response_row_thresholds(
                self.context, antecedent, max(windows)
            )

        def response_rows(antecedent: Antecedent, window: int) -> np.ndarray:
            rows, minimum_spans = row_thresholds[antecedent]
            return rows[
                minimum_spans <= int(window) * self.context.dataset.ticks_per_unit
            ]

        common_excitation = self._union_rows(
            [
                response_rows(rule.antecedent, rule.window)
                for rule in current.support.rules
                if rule.sign > 0
            ]
        )
        common_inhibition = self._union_rows(
            [
                response_rows(rule.antecedent, rule.window)
                for rule in current.support.rules
                if rule.sign < 0
            ]
        )
        grouped: dict[tuple, list[RuleIdentity]] = {}
        for rule in identities:
            trial = current.support.add(rule)
            grouped.setdefault(self._unsigned_geometry_key(trial), []).append(rule)
        previous_closure: tuple[ClosureTerm, ...] | None = None
        closure_rows = np.zeros(0, dtype=np.int64)
        for rules in grouped.values():
            representative = current.support.add(rules[0])
            closure = hierarchy_closure(representative)
            if closure != previous_closure:
                closure_rows = self._union_rows(
                    [response_rows(term.antecedent, term.window) for term in closure]
                )
                previous_closure = closure
            new_rows = response_rows(rules[0].antecedent, rules[0].window)
            affected = self._union_rows(
                [closure_rows, common_excitation, common_inhibition, new_rows]
            )
            localized = self._localized_score_from_rows(representative, affected)
            unsigned_key = self._unsigned_geometry_key(representative)
            with self._state_lock:
                self._relaxed_upper_cache[unsigned_key] = localized
            for rule in rules:
                trial = current.support.add(rule)
                excitation = (
                    np.union1d(common_excitation, new_rows)
                    if rule.sign > 0
                    else common_excitation
                )
                inhibition = (
                    np.union1d(common_inhibition, new_rows)
                    if rule.sign < 0
                    else common_inhibition
                )
                directional = self._directional_score_from_rows(
                    trial, closure_rows, excitation, inhibition
                )
                with self._state_lock:
                    self._directional_upper_cache[trial] = directional
                upper = min(localized, directional)
                scored.append((upper, rule))
                if upper <= threshold + self.config.search_tolerance:
                    with self._state_lock:
                        if upper < localized - 1.0e-12:
                            self.diagnostics.directional_relaxation_screens += 1
                        else:
                            self.diagnostics.relaxation_screens += 1
                else:
                    survivors.append(rule)
        with self._state_lock:
            while len(self._relaxed_upper_cache) > self._relaxed_upper_limit:
                self._relaxed_upper_cache.popitem(last=False)
            while len(self._directional_upper_cache) > self._relaxed_upper_limit:
                self._directional_upper_cache.popitem(last=False)
        if current.support == EMPTY_SUPPORT:
            for antecedent in sorted({rule.antecedent for rule in identities}):
                candidates = [
                    item for item in scored if item[1].antecedent == antecedent
                ]
                if candidates:
                    best = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
                    with self._state_lock:
                        self._skeleton_witnesses[antecedent] = best
        return tuple(survivors)

    def _admission_statistics(
        self,
        admission: np.ndarray,
        maximum_window: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative row/event counts for every W without materializing masks."""
        admitted = admission <= maximum_window
        row_histogram = np.bincount(
            admission[admitted].astype(np.int64, copy=False),
            minlength=maximum_window + 1,
        ).astype(np.float64)
        event_histogram = np.zeros(maximum_window + 1, dtype=np.float64)
        if len(self.context.target_rows):
            target_admission = admission[self.context.target_rows]
            matched = target_admission <= maximum_window
            if np.any(matched):
                event_histogram += np.bincount(
                    target_admission[matched].astype(np.int64, copy=False),
                    weights=self.context.target_counts[matched],
                    minlength=maximum_window + 1,
                )
        return np.cumsum(row_histogram), np.cumsum(event_histogram)

    @staticmethod
    def _bernoulli_aggregate_nll(events: float, noevents: float) -> float:
        if events <= 0.0 or noevents <= 0.0:
            return 0.0
        total = events + noevents
        probability = events / total
        return -events * math.log(probability) - noevents * math.log1p(-probability)

    def _safe_standalone_survivors(
        self,
        empty: SupportRecord,
        antecedent: Antecedent,
        identities: tuple[RuleIdentity, ...],
    ) -> tuple[RuleIdentity, ...]:
        """Exact empty-support safe bounds from one minimum-W row histogram.

        For a standalone atom, every higher-order closure footprint and the
        rule footprint are nested in W.  Therefore each row has a single first
        admitting W.  Cumulative histograms reproduce the localized and
        directional relaxation values exactly, without rebuilding 13 unions.
        """
        if self.context.dataset.likelihood != "first_event_cloglog":
            return self._safe_identity_survivors(empty, identities, 0.0)
        if not identities:
            return ()
        maximum_window = max(rule.window for rule in identities)
        ticks = self.context.dataset.ticks_per_unit

        def thresholds(
            term_antecedent: Antecedent, maximum: int
        ) -> tuple[np.ndarray, np.ndarray]:
            rows, minimum_spans = self.engine.response_row_thresholds(
                self.context, term_antecedent, maximum
            )
            admission = ((minimum_spans + ticks - 1) // ticks).astype(
                np.int64, copy=False
            )
            return rows, admission

        rule_rows, rule_admission = thresholds(antecedent, maximum_window)
        closure_parts: list[tuple[np.ndarray, np.ndarray]] = []
        for order in range(1, len(antecedent)):
            for subset in combinations(antecedent, order):
                term_maximum = 0 if order == 1 else maximum_window
                closure_parts.append(thresholds(subset, term_maximum))
        admission_dtype = np.min_scalar_type(maximum_window + 1)
        sentinel = maximum_window + 1
        closure_admission = np.full(
            self.context.n_grid, sentinel, dtype=admission_dtype
        )
        for rows, admission in closure_parts:
            closure_admission[rows] = np.minimum(closure_admission[rows], admission)
        union_admission = closure_admission.copy()
        union_admission[rule_rows] = np.minimum(
            union_admission[rule_rows], rule_admission
        )
        closure_count, closure_events = self._admission_statistics(
            closure_admission, maximum_window
        )
        union_count, union_events = self._admission_statistics(
            union_admission, maximum_window
        )
        total_events = float(np.sum(self.context.target_counts))
        total_noevents = float(self.context.n_grid) - total_events
        by_window: dict[int, tuple[float, float]] = {}
        for window in sorted({rule.window for rule in identities}):
            affected_events = float(union_events[window])
            affected_rows = float(union_count[window])
            closure_event = float(closure_events[window])
            closure_noevent = float(closure_count[window] - closure_event)
            excitation_nll = self._bernoulli_aggregate_nll(
                total_events - affected_events,
                total_noevents - closure_noevent,
            )
            inhibition_nll = self._bernoulli_aggregate_nll(
                total_events - closure_event,
                total_noevents - (affected_rows - affected_events),
            )
            by_window[window] = (excitation_nll, inhibition_nll)

        survivors: list[RuleIdentity] = []
        scored: list[tuple[float, RuleIdentity]] = []
        for rule in identities:
            trial = Support.of((rule,))
            excitation_nll, inhibition_nll = by_window[rule.window]
            # The localized relaxation is weaker than either directional
            # expression and is represented by saturating the entire union.
            affected_events = float(union_events[rule.window])
            affected_rows = float(union_count[rule.window])
            localized_nll = self._bernoulli_aggregate_nll(
                total_events - affected_events,
                total_noevents - (affected_rows - affected_events),
            )
            penalty = self.objective.structural_penalty(trial)
            localized = min(
                support_score(
                    baseline_nll=self.baseline_nll,
                    fit_nll=localized_nll,
                    penalty=penalty,
                ),
                self.saturated_upper_score(trial),
            )
            directional_nll = excitation_nll if rule.sign > 0 else inhibition_nll
            directional = min(
                localized,
                support_score(
                    baseline_nll=self.baseline_nll,
                    fit_nll=directional_nll,
                    penalty=penalty,
                ),
            )
            with self._state_lock:
                self._relaxed_upper_cache[self._unsigned_geometry_key(trial)] = (
                    localized
                )
                self._directional_upper_cache[trial] = directional
            upper = min(localized, directional)
            scored.append((upper, rule))
            if upper > self.config.search_tolerance:
                survivors.append(rule)
            else:
                with self._state_lock:
                    if upper < localized - 1.0e-12:
                        self.diagnostics.directional_relaxation_screens += 1
                    else:
                        self.diagnostics.relaxation_screens += 1
        best = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
        with self._state_lock:
            self._skeleton_witnesses[antecedent] = best
            while len(self._relaxed_upper_cache) > self._relaxed_upper_limit:
                self._relaxed_upper_cache.popitem(last=False)
            while len(self._directional_upper_cache) > self._relaxed_upper_limit:
                self._directional_upper_cache.popitem(last=False)
        return tuple(survivors)

    def _screen_standalones(self, empty: SupportRecord) -> None:
        """Screen independent skeletons concurrently before any exact fit."""
        if len(self._standalone_screen_cache) == len(self.skeletons):
            return

        def screen(
            antecedent: Antecedent,
        ) -> tuple[Antecedent, tuple[RuleIdentity, ...]]:
            configure_cpu_threads(1)
            identities = tuple(
                rule for rule in self.dictionary if rule.antecedent == antecedent
            )
            return antecedent, self._safe_standalone_survivors(
                empty, antecedent, identities
            )

        workers = min(self.config.pricing_workers, len(self.skeletons))
        if workers <= 1:
            screened = [screen(antecedent) for antecedent in self.skeletons]
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="crbstpp-screen",
            ) as executor:
                screened = list(executor.map(screen, self.skeletons))
        self._standalone_screen_cache.update(screened)

    def _best_addition(
        self,
        current: SupportRecord,
        *,
        antecedents: set[Antecedent] | None = None,
    ) -> SupportRecord | None:
        if antecedents is None and current.support in self._addition_cache:
            cached_support = self._addition_cache[current.support]
            return None if cached_support is None else self.fit(cached_support, current)
        identities = self._inactive_identities(current.support, antecedents)
        if not identities:
            if antecedents is None:
                self._addition_cache[current.support] = None
            return None
        identities = self._safe_identity_survivors(current, identities, current.score)
        if not identities:
            if antecedents is None:
                self._addition_cache[current.support] = None
            return None
        ranked = self._rank_identities(current, identities)
        survivors: list[tuple[float, Support]] = []
        for price, rule in ranked:
            trial = current.support.add(rule)
            survivors.append((price, trial))
        evaluated = self.bounds_many(current, [trial for _, trial in survivors])
        bounds = [
            (price, item) for (price, _), item in zip(survivors, evaluated, strict=True)
        ]
        bounds.sort(
            key=lambda item: (-item[1].upper_score, -item[0], item[1].support.rules)
        )
        incumbent: SupportRecord | None = None
        incumbent_score = current.score
        position = 0
        while position < len(bounds):
            if (
                bounds[position][1].upper_score
                <= incumbent_score + self.config.search_tolerance
            ):
                self.diagnostics.dual_screens += len(bounds) - position
                break
            wave: list[ProposalBounds] = []
            while position < len(bounds) and len(wave) < self.config.exact_workers:
                item = bounds[position][1]
                position += 1
                if item.upper_score > incumbent_score + self.config.search_tolerance:
                    wave.append(item)
                else:
                    self.diagnostics.dual_screens += 1
            records = self.fit_many([item.support for item in wave], current)
            for record in records:
                if record.score > incumbent_score + self.config.search_tolerance:
                    incumbent, incumbent_score = record, record.score
        if antecedents is None:
            self._addition_cache[current.support] = (
                None if incumbent is None else incumbent.support
            )
        return incumbent

    def _best_block_addition(
        self,
        current: SupportRecord,
        *,
        minimum_score: float | None = None,
        antecedents: set[Antecedent] | None = None,
    ) -> SupportRecord | None:
        """Exact-refit only MDL-positive joint block-score directions."""
        acceptance = current.score if minimum_score is None else float(minimum_score)
        cacheable = antecedents is None and minimum_score is None
        if cacheable and current.support in self._addition_cache:
            cached_support = self._addition_cache[current.support]
            return None if cached_support is None else self.fit(cached_support, current)
        identities = self._inactive_identities(current.support, antecedents)
        ranked = self._rank_block_identities(current, identities)
        admissible = [
            item
            for item in ranked
            if (not item[3]) or item[0] > self.config.search_tolerance
        ]
        for start in range(0, len(admissible), self.config.exact_workers):
            wave = admissible[start : start + self.config.exact_workers]
            records = self.fit_many(
                [current.support.add(item[2]) for item in wave], current
            )
            improving = [
                record
                for record in records
                if record.score > acceptance + self.config.search_tolerance
            ]
            if improving:
                best = sorted(
                    improving, key=lambda record: (-record.score, record.support.rules)
                )[0]
                if cacheable:
                    self._addition_cache[current.support] = best.support
                return best
            with self._state_lock:
                self.diagnostics.block_score_exact_rejections += len(records)
        if cacheable:
            self._addition_cache[current.support] = None
        return None

    def _standalone_block_atoms(
        self, empty: SupportRecord
    ) -> tuple[SupportRecord, ...]:
        """Freeze every exact-positive atom whose joint block score is MDL-positive."""
        ranked = self._rank_block_identities(empty, self.dictionary)
        admissible = [
            item
            for item in ranked
            if (not item[3]) or item[0] > self.config.search_tolerance
        ]
        positives: dict[Support, SupportRecord] = {}
        for start in range(0, len(admissible), self.config.exact_workers):
            wave = admissible[start : start + self.config.exact_workers]
            records = self.fit_many([Support.of((item[2],)) for item in wave], empty)
            for record in records:
                if record.score > self.config.search_tolerance:
                    positives[record.support] = record
                else:
                    with self._state_lock:
                        self.diagnostics.block_score_exact_rejections += 1
        return tuple(
            positives[support]
            for support in sorted(positives, key=lambda item: item.rules)
        )

    def _standalone_skeleton(
        self, empty: SupportRecord, antecedent: Antecedent
    ) -> tuple[SupportRecord, ...]:
        identities = self._standalone_screen_cache.get(antecedent)
        if identities is None:
            all_identities = tuple(
                rule for rule in self.dictionary if rule.antecedent == antecedent
            )
            identities = self._safe_identity_survivors(empty, all_identities, 0.0)
        if not identities:
            return ()
        ranked = self._rank_identities(empty, identities)
        positives: dict[Support, SupportRecord] = {}
        survivors: list[tuple[float, Support]] = []
        for price, rule in ranked:
            trial = Support.of((rule,))
            survivors.append((price, trial))
        evaluated = self.bounds_many(empty, [trial for _, trial in survivors])
        pending = [
            (price, item) for (price, _), item in zip(survivors, evaluated, strict=True)
        ]
        pending.sort(
            key=lambda item: (-item[1].upper_score, -item[0], item[1].support.rules)
        )
        admissible = []
        for _, item in pending:
            if item.upper_score <= self.config.search_tolerance:
                self.diagnostics.dual_screens += 1
            else:
                admissible.append(item)
        for start in range(0, len(admissible), self.config.exact_workers):
            wave = admissible[start : start + self.config.exact_workers]
            records = self.fit_many([item.support for item in wave], empty)
            for record in records:
                if record.score > 0:
                    positives[record.support] = record
        return tuple(positives.values())

    def _best_exact_proposal(
        self, current: SupportRecord, proposals: list[Support]
    ) -> SupportRecord | None:
        unique = sorted(set(proposals), key=lambda support: support.rules)
        survivors: list[Support] = []
        for support in unique:
            if support == current.support:
                continue
            upper = self.safe_upper_score(support)
            if upper <= current.score + self.config.search_tolerance:
                if upper < self.localized_upper_score(support) - 1.0e-12:
                    self.diagnostics.directional_relaxation_screens += 1
                else:
                    self.diagnostics.relaxation_screens += 1
                continue
            survivors.append(support)
        bounded = self.bounds_many(current, survivors)
        bounded.sort(key=lambda item: (-item.upper_score, item.support.rules))
        incumbent = None
        incumbent_score = current.score
        position = 0
        while position < len(bounded):
            if (
                bounded[position].upper_score
                <= incumbent_score + self.config.search_tolerance
            ):
                self.diagnostics.dual_screens += len(bounded) - position
                break
            wave: list[ProposalBounds] = []
            while position < len(bounded) and len(wave) < self.config.exact_workers:
                item = bounded[position]
                position += 1
                if item.upper_score > incumbent_score + self.config.search_tolerance:
                    wave.append(item)
                else:
                    self.diagnostics.dual_screens += 1
            records = self.fit_many([item.support for item in wave], current)
            for record in records:
                if record.score > incumbent_score + self.config.search_tolerance:
                    incumbent, incumbent_score = record, record.score
        return incumbent

    def _splice_proposals(
        self, current: SupportRecord, drops: list[tuple[RuleIdentity, SupportRecord]]
    ) -> list[Support]:
        existing = set(current.support.antecedents)
        additions = [
            item[1]
            for item in sorted(
                (
                    item
                    for antecedent, item in self._skeleton_witnesses.items()
                    if antecedent not in existing
                ),
                key=lambda item: (-item[0], item[1]),
            )
        ]
        if len(additions) < 2:
            return []
        proposals: set[Support] = set()
        base = list(current.support.rules)
        # Every prefix size is evaluated; there is no top-k or support-size cap.
        for size in range(2, len(additions) + 1):
            proposals.add(Support.of((*base, *additions[:size])))
        ordered_drops = sorted(
            drops,
            key=lambda item: (current.score - item[1].score, item[0]),
        )
        for size in range(1, max(len(ordered_drops), len(additions)) + 1):
            removed = {item[0] for item in ordered_drops[:size]}
            added = additions[:size]
            proposals.add(
                Support.of(
                    [rule for rule in current.support.rules if rule not in removed]
                    + added
                )
            )
        proposals.discard(current.support)
        return sorted(proposals, key=lambda support: support.rules)

    def _best_one_exchange(self, current: SupportRecord) -> SupportRecord | None:
        incumbent: SupportRecord | None = None
        incumbent_score = current.score
        direct = self._best_addition(current)
        if (
            direct is not None
            and direct.score > incumbent_score + self.config.search_tolerance
        ):
            incumbent, incumbent_score = direct, direct.score
        reduced_records = self.fit_many(
            [current.support.drop(removed) for removed in current.support.rules],
            current,
        )
        drops: list[tuple[RuleIdentity, SupportRecord]] = []
        for removed, reduced in zip(
            current.support.rules, reduced_records, strict=True
        ):
            drops.append((removed, reduced))
            if reduced.score > incumbent_score + self.config.search_tolerance:
                incumbent, incumbent_score = reduced, reduced.score
            # Adding any inactive identity to S\{r} exactly covers every
            # identity replacement and every one-drop/one-add swap.
            replacement = self._best_addition(reduced)
            if (
                replacement is not None
                and replacement.support != current.support
                and replacement.score > incumbent_score + self.config.search_tolerance
            ):
                incumbent, incumbent_score = replacement, replacement.score
        if incumbent is not None:
            return incumbent
        # Joint splicing is an escape step only after the complete one-exchange
        # audit found no move.  Any accepted splice is still an exact J increase.
        joint = self._best_exact_proposal(
            current, self._splice_proposals(current, drops)
        )
        if joint is None:
            self.diagnostics.terminal_audits += 1
        return joint

    def _best_block_exchange(self, current: SupportRecord) -> SupportRecord | None:
        """Block-score add/swap audit with exact drops and exact acceptance."""
        direct = self._best_block_addition(current)
        if direct is not None:
            return direct
        reduced_records = self.fit_many(
            [current.support.drop(removed) for removed in current.support.rules],
            current,
        )
        improving_drops = [
            record
            for record in reduced_records
            if record.score > current.score + self.config.search_tolerance
        ]
        if improving_drops:
            return sorted(
                improving_drops,
                key=lambda record: (-record.score, record.support.rules),
            )[0]
        # Scoring every reduced support covers W/sign replacement and
        # one-drop/one-add swaps.  Promotions of an existing closure nuisance
        # are nonnested and therefore always exact-audited.
        for reduced in reduced_records:
            replacement = self._best_block_addition(
                reduced, minimum_score=current.score
            )
            if replacement is not None and replacement.support != current.support:
                return replacement
        self.diagnostics.terminal_audits += 1
        return None

    def search(self) -> SearchResult:
        positive_atoms: dict[Support, SupportRecord] = {}
        starts: list[Support] = [EMPTY_SUPPORT]
        empty = self.records[EMPTY_SUPPORT]
        # One fused hierarchy-aware score pass replaces candidate-wise
        # standalone dual/refit enumeration.  Every score-positive atom is
        # still exact-fitted before it can seed a discovery basin.
        for record in self._standalone_block_atoms(empty):
            positive_atoms[record.support] = record
            starts.append(record.support)
        starts = sorted(set(starts), key=lambda support: support.rules)
        transition_cache: dict[Support, Support | None] = {}
        terminals: dict[Support, SupportRecord] = {}
        paths: list[dict[str, object]] = []
        for start in starts:
            current = self.fit(start, empty) if start.rules else empty
            moves: list[dict[str, object]] = []
            while True:
                self.diagnostics.states += 1
                if current.support in transition_cache:
                    self.diagnostics.transition_cache_hits += 1
                    next_support = transition_cache[current.support]
                    if next_support is None:
                        break
                    next_record = self.fit(next_support, current)
                else:
                    next_record = self._best_block_exchange(current)
                    transition_cache[current.support] = (
                        None if next_record is None else next_record.support
                    )
                    if next_record is None:
                        break
                gain = next_record.score - current.score
                if gain <= self.config.search_tolerance:
                    raise AssertionError(
                        "accepted support move does not strictly increase J"
                    )
                self.diagnostics.accepted_moves += 1
                moves.append(
                    {
                        "from": support_key(current.support),
                        "to": support_key(next_record.support),
                        "gain": float(gain),
                    }
                )
                current = next_record
            if current.score > 0:
                terminals[current.support] = current
            paths.append(
                {
                    "start": support_key(start),
                    "terminal": support_key(current.support),
                    "moves": moves,
                }
            )
        family_map = {**positive_atoms, **terminals}
        return SearchResult(
            family=tuple(
                family_map[key]
                for key in sorted(family_map, key=lambda support: support.rules)
            ),
            terminals=tuple(
                terminals[key]
                for key in sorted(terminals, key=lambda support: support.rules)
            ),
            positive_atoms=tuple(
                positive_atoms[key]
                for key in sorted(positive_atoms, key=lambda support: support.rules)
            ),
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
