from __future__ import annotations

import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .config import RunConfig
from .dual import (
    DualCertificate,
    DualGeometry,
    dual_certificate,
    dual_geometry,
    offset_dual_certificate,
)
from .likelihood import loss_grid_sparse_event_derivatives, loss_rows
from .native import (
    aggregate_design_rows,
    aggregate_design_rows_with_groups,
    configure_cpu_threads,
    moments,
    moments_batch,
    nonnegative_quadratic_gains,
)
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
from .solver import FitResult, fit_model_matrix, fit_offset_design


@dataclass(frozen=True)
class ProposalBounds:
    support: Support
    lower_score: float
    upper_score: float
    lower_gain: float
    upper_gain: float
    dual: DualCertificate | None
    warm_coefficients: np.ndarray


@dataclass(frozen=True)
class _RestrictedAddBounds:
    rule: RuleIdentity
    lower_score: float
    upper_score: float
    lower_gain: float
    upper_gain: float
    dual: DualCertificate | None


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
    profiled_skeletons: int = 0
    score_basin_nodes: int = 0
    score_basin_seeds: int = 0
    positive_primitive_roots: int = 0
    nonnested_exact_audits: int = 0
    full_dictionary_audits: int = 0
    working_set_expansions: int = 0
    working_set_skeletons: int = 0
    restricted_block_fits: int = 0
    restricted_block_screens: int = 0
    restricted_add_audits: int = 0
    restricted_drop_audits: int = 0
    restricted_block_terminals: int = 0
    restricted_problem_builds: int = 0
    restricted_problem_hits: int = 0
    standalone_branch_audits: int = 0
    standalone_branch_rejections: int = 0
    multi_source_roots: int = 0
    path_compression_hits: int = 0
    restricted_bound_evaluations: int = 0
    restricted_dual_screens: int = 0
    restricted_dual_fail_open: int = 0
    restricted_relaxation_screens: int = 0
    lazy_exact_refits_avoided: int = 0
    lazy_bound_stops: int = 0
    restricted_geometry_builds: int = 0
    restricted_geometry_hits: int = 0
    restricted_geometry_evictions: int = 0


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


@dataclass(frozen=True)
class _RestrictedAddProblem:
    offset: np.ndarray
    unsigned_design: np.ndarray
    exposure: np.ndarray
    noevent: np.ndarray
    event: np.ndarray
    old_nll: float
    free_dimension: int


@dataclass(frozen=True)
class _RestrictedGeometry:
    rows: np.ndarray
    design_patterns: np.ndarray
    row_patterns: np.ndarray
    event: np.ndarray
    free_dimension: int

    @property
    def nbytes(self) -> int:
        return int(
            self.rows.nbytes
            + self.design_patterns.nbytes
            + self.row_patterns.nbytes
            + self.event.nbytes
        )


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
    """Exact fixed-support fits with profiled rule-block score search.

    Each antecedent skeleton owns exactly one W/sign identity profiled once at
    the D_fit baseline.  Empty and every objective-admissible single-skeleton
    root then ascend a shared support DAG over that frozen dictionary.  Every
    score-positive inactive identity is either removed by a verified upper bound
    or receives a restricted exact audit; every active drop is audited exactly.
    Every accepted move improves the fully refitted MDL objective.  Positive
    atoms and all unique terminals reach certification.  The terminal claim is
    frozen-dictionary block-score restricted add/drop stationarity, not
    all-identity one-exchange stationarity or a global optimum.
    """

    def __init__(self, context: Context, config: RunConfig):
        configure_cpu_threads(config.pricing_workers)
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
        self._relaxed_upper_cache: OrderedDict[tuple, float] = OrderedDict()
        self._relaxed_upper_limit = max(1, min(200_000, config.cache_bytes // 256))
        self._directional_upper_cache: OrderedDict[Support, float] = OrderedDict()
        self._pricing_state: OrderedDict[Support, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._pricing_cache_bytes = 0
        self._pricing_cache_limit = max(1, config.cache_bytes // 8)
        self._raw_pricing_state: OrderedDict[
            Support, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._raw_pricing_cache_bytes = 0
        self._raw_pricing_cache_limit = max(1, config.cache_bytes // 8)
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
        self._profiled_dictionary: tuple[RuleIdentity, ...] | None = None
        self._working_antecedents: set[Antecedent] = set()
        self._restricted_add_scores: dict[Support, dict[RuleIdentity, float]] = {}
        self._restricted_add_fits: dict[Support, dict[RuleIdentity, FitResult]] = {}
        self._restricted_add_bounds: dict[
            Support, dict[RuleIdentity, _RestrictedAddBounds]
        ] = {}
        self._restricted_drop_scores: dict[Support, dict[Support, float]] = {}
        self._restricted_add_problems: dict[
            Support,
            dict[tuple[Antecedent, int], _RestrictedAddProblem | None],
        ] = {}
        self._restricted_add_events: dict[
            tuple[Support, Antecedent, int], threading.Event
        ] = {}
        self._restricted_geometries: OrderedDict[
            tuple[tuple[ClosureTerm, ...], Antecedent, int],
            _RestrictedGeometry | None,
        ] = OrderedDict()
        self._restricted_geometry_events: dict[
            tuple[tuple[ClosureTerm, ...], Antecedent, int], threading.Event
        ] = {}
        self._restricted_geometry_bytes = 0
        self._restricted_geometry_limit = max(1, config.cache_bytes // 16)
        baseline_matrix = self.engine.model_matrix(context, EMPTY_SUPPORT)
        baseline_fit = fit_model_matrix(
            baseline_matrix,
            likelihood=context.dataset.likelihood,
            tolerance=config.solver_tolerance,
            max_iter=config.solver_max_iter,
            device=(config.pricing_devices or ("cpu",))[0],
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
        self._pricing_state.clear()
        self._pricing_cache_bytes = 0
        self._raw_pricing_state.clear()
        self._raw_pricing_cache_bytes = 0
        self._relaxed_upper_cache.clear()
        self._directional_upper_cache.clear()
        self._block_price_cache.clear()
        self._restricted_add_scores.clear()
        self._restricted_add_fits.clear()
        self._restricted_add_bounds.clear()
        self._restricted_drop_scores.clear()
        self._restricted_add_problems.clear()
        self._restricted_add_events.clear()
        self._restricted_geometries.clear()
        self._restricted_geometry_events.clear()
        self._restricted_geometry_bytes = 0
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
        self,
        support: Support,
        source: SupportRecord | None = None,
        *,
        device: str | None = None,
        warm_start_override: np.ndarray | None = None,
    ) -> SupportRecord:
        with self._state_lock:
            cached = self.records.get(support)
            if cached is not None and (
                cached.fit.converged or warm_start_override is None
            ):
                self.records.move_to_end(support)
                self.diagnostics.fit_cache_hits += 1
                return cached
            stored = self._stored_records.get(support)
        if stored is not None and (stored.fit.converged or warm_start_override is None):
            matrix = self.engine.model_matrix(self.context, support)
            record = SupportRecord(
                support, matrix, stored.fit, stored.penalty, stored.score
            )
            with self._state_lock:
                self.diagnostics.fit_cache_hits += 1
                return self._retain_record(record)
        matrix = (
            self.engine.extend_model_matrix(self.context, support, source.matrix)
            if source is not None and set(source.support.rules).issubset(support.rules)
            else self.engine.model_matrix(self.context, support)
        )
        warm = (
            np.asarray(warm_start_override, dtype=np.float64)
            if warm_start_override is not None
            else None
            if source is None
            else self.warm_start(source, matrix)
        )
        if warm is not None and warm.shape != (matrix.dimension,):
            raise ValueError("warm-start override dimension does not match model")
        fit = fit_model_matrix(
            matrix,
            likelihood=self.context.dataset.likelihood,
            tolerance=self.config.solver_tolerance,
            max_iter=self.config.solver_max_iter,
            warm_start=warm,
            device=device or (self.config.pricing_devices or ("cpu",))[0],
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
            replace_existing = (
                existing is not None
                and fit.converged
                and (
                    not existing.fit.converged
                    or fit.nll
                    < existing.fit.nll
                    - self.config.solver_tolerance * max(1.0, abs(existing.fit.nll))
                )
            )
            if existing is not None and not replace_existing:
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

        devices = self.config.pricing_devices or ("cpu",)

        def fit_one(item: tuple[int, Support]) -> SupportRecord:
            configure_cpu_threads(threads_per_fit)
            index, support = item
            return self.fit(support, source, device=devices[index % len(devices)])

        with ThreadPoolExecutor(
            max_workers=min(self.config.exact_workers, len(ordered)),
            thread_name_prefix="crbstpp-exact",
        ) as executor:
            return list(executor.map(fit_one, enumerate(ordered)))

    def fit_fixed(
        self,
        support: Support,
        closure: tuple[ClosureTerm, ...],
        *,
        device: str | None = None,
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
            device=device or (self.config.pricing_devices or ("cpu",))[0],
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

        devices = self.config.pricing_devices or ("cpu",)

        def fit_one(
            indexed: tuple[int, tuple[Support, tuple[ClosureTerm, ...]]],
        ) -> tuple[ModelMatrix, FitResult]:
            configure_cpu_threads(threads_per_fit)
            index, item = indexed
            return self.fit_fixed(*item, device=devices[index % len(devices)])

        with ThreadPoolExecutor(
            max_workers=min(self.config.exact_workers, len(specifications)),
            thread_name_prefix="crbstpp-fixed",
        ) as executor:
            return list(executor.map(fit_one, enumerate(specifications)))

    def _fit_embedded_closure_null(
        self,
        record: SupportRecord,
        *,
        device: str,
    ) -> FitResult:
        """Fit a closure null by losslessly projecting its fitted full matrix."""
        closure = tuple(record.matrix.closure)
        key = (EMPTY_SUPPORT, closure)
        with self._state_lock:
            cached = self._forced_fits.get(key)
            if cached is not None:
                self.diagnostics.forced_fit_cache_hits += 1
                return cached
        if not closure:
            return self._stored_records[EMPTY_SUPPORT].fit
        dimension = record.matrix.free_dimension
        projected = np.ascontiguousarray(record.matrix.x[:, :dimension])
        projected, exposure, noevent, event = aggregate_design_rows(
            projected,
            record.matrix.exposure_weight,
            record.matrix.noevent_weight,
            record.matrix.event_weight,
            copy_input=True,
        )
        empty_i64 = np.zeros(0, dtype=np.int64)
        matrix = ModelMatrix(
            x=projected,
            exposure_weight=exposure,
            noevent_weight=noevent,
            event_weight=event,
            free_dimension=dimension,
            closure_dimension=len(closure) * self.config.knot_count,
            rule_slices=(),
            support=EMPTY_SUPPORT,
            closure=closure,
            active_rows=empty_i64,
            active_design_groups=empty_i64,
            active_age_bins=empty_i64,
            aggregate_bins=empty_i64,
        )
        fit = fit_model_matrix(
            matrix,
            likelihood=self.context.dataset.likelihood,
            tolerance=self.config.solver_tolerance,
            max_iter=self.config.solver_max_iter,
            device=device,
        )
        with self._state_lock:
            return self._forced_fits.setdefault(key, fit)

    def _fit_embedded_closure_nulls(
        self, records: list[SupportRecord]
    ) -> list[FitResult]:
        if not records:
            return []
        devices = self.config.pricing_devices or ("cpu",)
        threads_per_fit = max(
            1, self.config.pricing_workers // self.config.exact_workers
        )

        def solve(indexed: tuple[int, SupportRecord]) -> FitResult:
            configure_cpu_threads(threads_per_fit)
            index, record = indexed
            device = devices[index % len(devices)]
            return self._fit_embedded_closure_null(record, device=device)

        if len(records) == 1 or self.config.exact_workers == 1:
            return [solve((index, record)) for index, record in enumerate(records)]
        with self._state_lock:
            self.diagnostics.parallel_exact_batches += 1
        with ThreadPoolExecutor(
            max_workers=min(self.config.exact_workers, len(records)),
            thread_name_prefix="crbstpp-closure-null",
        ) as executor:
            return list(executor.map(solve, enumerate(records)))

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
                device=(self.config.pricing_devices or ("cpu",))[0],
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
            current.matrix.x,
            np.zeros_like(second),
            second,
            device=(self.config.pricing_devices or ("cpu",))[0],
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

    def _raw_pricing_components(
        self, current: SupportRecord
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cache eta and exact row derivatives once per fitted support state."""
        with self._state_lock:
            cached = self._raw_pricing_state.get(current.support)
            if cached is not None:
                self._raw_pricing_state.move_to_end(current.support)
                return cached
        eta = self.engine.linear_predictor(
            self.context, current.matrix, current.fit.coefficients
        )
        first, second = loss_grid_sparse_event_derivatives(
            eta,
            likelihood=self.context.dataset.likelihood,
            exposure=self.engine.tick_exposure,
            event_rows=self.context.target_rows,
            event_counts=self.context.target_counts,
        )
        result = (
            np.ascontiguousarray(eta, dtype=np.float64),
            np.ascontiguousarray(first, dtype=np.float64),
            np.ascontiguousarray(second, dtype=np.float64),
        )
        size = sum(value.nbytes for value in result)
        with self._state_lock:
            existing = self._raw_pricing_state.get(current.support)
            if existing is not None:
                return existing
            self._raw_pricing_state[current.support] = result
            self._raw_pricing_cache_bytes += size
            while self._raw_pricing_cache_bytes > self._raw_pricing_cache_limit:
                _, removed = self._raw_pricing_state.popitem(last=False)
                self._raw_pricing_cache_bytes -= sum(value.nbytes for value in removed)
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
            _, raw_first, raw_second = self._raw_pricing_components(current)
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
                first = raw_first[rows]
                second = raw_second[rows]
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
                    weighted_candidates = second[:, None] * flat_candidates
                    if (
                        current.matrix.dimension == 1
                        and len(indices) == 1
                        and int(indices[0]) == 0
                    ):
                        flat_cross = np.sum(weighted_candidates, axis=0, keepdims=True)
                    else:
                        design = self.engine.design_at_rows_with_context(
                            self.context, current.matrix, rows
                        )
                        flat_cross = design[:, indices].T @ weighted_candidates
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
        self._raw_pricing_components(current)
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
        gradients = np.asarray(
            [
                float(rule.sign) * components[rule.antecedent][rule.window][0]
                for rule in identities
            ],
            dtype=np.float64,
        )
        hessians = np.asarray(
            [components[rule.antecedent][rule.window][1] for rule in identities],
            dtype=np.float64,
        )
        compiled_gains = nonnegative_quadratic_gains(gradients, hessians)
        gains = (
            compiled_gains
            if compiled_gains is not None
            else np.asarray(
                [
                    _nonnegative_quadratic_gain(gradient, hessian)
                    for gradient, hessian in zip(gradients, hessians, strict=True)
                ]
            )
        )
        prices = list(zip(gains.tolist(), identities, strict=True))
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
            tile_gradients = np.zeros((batch_size, maximum_dimension), dtype=np.float64)
            tile_hessians = np.zeros(
                (batch_size, maximum_dimension, maximum_dimension),
                dtype=np.float64,
            )
            tile_crosses = np.zeros(
                (batch_size, len(indices), maximum_dimension), dtype=np.float64
            )
            active_design = design[:, indices]
            for batch_index in range(batch_size):
                candidate = candidate_batch[batch_index]
                joint = (
                    np.concatenate((active_design, candidate), axis=1)
                    if len(indices)
                    else candidate
                )
                joint, reduced_first, reduced_second, _ = aggregate_design_rows(
                    joint,
                    first,
                    second,
                    np.zeros_like(first),
                    copy_input=True,
                )
                reduced_candidate = joint[:, len(indices) :]
                tile_gradients[batch_index], tile_hessians[batch_index] = moments(
                    reduced_candidate,
                    reduced_first,
                    reduced_second,
                    device=device,
                )
                if len(indices):
                    tile_crosses[batch_index] = joint[:, : len(indices)].T @ (
                        reduced_second[:, None] * reduced_candidate
                    )
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

    def _rank_mdl_identities(
        self, current: SupportRecord, identities: tuple[RuleIdentity, ...]
    ) -> list[tuple[float, float, RuleIdentity]]:
        """Return every W/sign identity's conditional-Fisher MDL score.

        The rule block is conditioned on the currently fitted model, but new
        hierarchy nuisance is deliberately excluded from pricing.  It enters
        the exact restricted audit.  Consequently this score orders and admits
        work; it never by itself accepts a support move.
        """
        if not identities:
            return []
        raw = self._rank_identities(current, identities)
        scored: list[tuple[float, float, RuleIdentity]] = []
        current_closure_count = len(current.matrix.closure)
        log_n = math.log(max(2, self.objective.n_entities))
        for gain, rule in raw:
            trial = current.support.add(rule)
            penalty_delta = self.objective.structural_penalty(trial) - current.penalty
            closure_delta = (
                len(hierarchy_closure(trial)) - current_closure_count
            ) * self.config.knot_count
            rule_penalty = penalty_delta - closure_delta * log_n
            scored.append((2.0 * gain - rule_penalty, gain, rule))
        positive = sum(item[0] > self.config.search_tolerance for item in scored)
        with self._state_lock:
            self.diagnostics.block_score_evaluations += len(identities)
            self.diagnostics.block_score_admissible += positive
            self.diagnostics.block_score_screens += len(identities) - positive
        return sorted(scored, key=lambda item: (-item[0], item[2]))

    def _rank_profiled_identities(
        self, current: SupportRecord, identities: tuple[RuleIdentity, ...]
    ) -> list[tuple[float, float, RuleIdentity]]:
        """Profile the best W/sign score once per antecedent skeleton."""
        scored = self._rank_mdl_identities(current, identities)
        profiled: list[tuple[float, float, RuleIdentity]] = []
        for antecedent in sorted({item[2].antecedent for item in scored}):
            group = [item for item in scored if item[2].antecedent == antecedent]
            best = sorted(group, key=lambda item: (-item[0], item[2]))[0]
            profiled.append(best)
            self._skeleton_witnesses[antecedent] = (best[0], best[2])
        with self._state_lock:
            self.diagnostics.profiled_skeletons += len(profiled)
        return sorted(profiled, key=lambda item: (-item[0], item[2]))

    def _local_score_maxima(
        self, profiled: list[tuple[float, float, RuleIdentity]]
    ) -> list[tuple[float, float, RuleIdentity]]:
        """Return all positive local maxima of the skeleton add/drop DAG."""
        by_antecedent = {item[2].antecedent: item for item in profiled}

        def dominates(
            candidate: tuple[float, float, RuleIdentity],
            current: tuple[float, float, RuleIdentity],
        ) -> bool:
            if candidate[0] > current[0] + self.config.search_tolerance:
                return True
            return (
                abs(candidate[0] - current[0]) <= self.config.search_tolerance
                and candidate[2] < current[2]
            )

        positive = [item for item in profiled if item[0] > self.config.search_tolerance]
        local_maxima: list[tuple[float, float, RuleIdentity]] = []
        for item in positive:
            antecedent = item[2].antecedent
            members = set(antecedent)
            adjacent = []
            for other_antecedent, other in by_antecedent.items():
                if abs(len(other_antecedent) - len(antecedent)) != 1:
                    continue
                other_members = set(other_antecedent)
                if members < other_members or other_members < members:
                    adjacent.append(other)
            local_maximum = not any(dominates(other, item) for other in adjacent)
            if local_maximum:
                local_maxima.append(item)
        return sorted(local_maxima, key=lambda item: (-item[0], item[2]))

    def _objective_root_candidates(
        self, profiled: list[tuple[float, float, RuleIdentity]]
    ) -> list[tuple[float, float, RuleIdentity]]:
        """Return all objective-defined score-basin roots without a top-k.

        Singletons remain primitive roots.  Higher-order roots are the local
        maxima of both the inclusion DAG and the same-order one-predicate
        exchange graph.  Non-root positive skeletons remain available to the
        delayed full-dictionary add audit; only redundant launches are removed.
        """
        positive = [item for item in profiled if item[0] > self.config.search_tolerance]
        local_maxima = self._local_score_maxima(profiled)
        by_order: dict[int, list[tuple[float, float, RuleIdentity]]] = {}
        for item in positive:
            by_order.setdefault(item[2].order, []).append(item)

        def dominates(
            candidate: tuple[float, float, RuleIdentity],
            current: tuple[float, float, RuleIdentity],
        ) -> bool:
            if candidate[0] > current[0] + self.config.search_tolerance:
                return True
            return (
                abs(candidate[0] - current[0]) <= self.config.search_tolerance
                and candidate[2] < current[2]
            )

        exchange_maxima: list[tuple[float, float, RuleIdentity]] = []
        for order, items in by_order.items():
            if order == 1:
                continue
            for item in items:
                members = set(item[2].antecedent)
                adjacent = (
                    other
                    for other in items
                    if len(members.intersection(other[2].antecedent)) == order - 1
                )
                if not any(dominates(other, item) for other in adjacent):
                    exchange_maxima.append(item)
        candidates = {
            item[2].antecedent: item for item in positive if item[2].order == 1
        }
        candidates.update({item[2].antecedent: item for item in local_maxima})
        candidates.update({item[2].antecedent: item for item in exchange_maxima})
        self._working_antecedents.update(item[2].antecedent for item in positive)
        with self._state_lock:
            self.diagnostics.score_basin_nodes += len(positive)
            self.diagnostics.score_basin_seeds += sum(
                item[2].order > 1 for item in candidates.values()
            )
            self.diagnostics.positive_primitive_roots += sum(
                len(item[2].antecedent) == 1 for item in positive
            )
        return sorted(candidates.values(), key=lambda item: (-item[0], item[2]))

    def _expand_working_set(self, current: SupportRecord) -> bool:
        """Run one complete delayed-column audit and activate every violation."""
        existing = set(current.support.antecedents)
        dictionary = self._profiled_dictionary or self.dictionary
        outside = tuple(
            rule
            for rule in dictionary
            if rule.antecedent not in existing
            and rule.antecedent not in self._working_antecedents
        )
        if not outside:
            return False
        with self._state_lock:
            self.diagnostics.full_dictionary_audits += 1
        profiled = self._rank_profiled_identities(current, outside)
        additions = {
            item[2].antecedent
            for item in profiled
            if item[0] > self.config.search_tolerance
        }
        if not additions:
            return False
        self._working_antecedents.update(additions)
        with self._state_lock:
            self.diagnostics.working_set_expansions += 1
            self.diagnostics.working_set_skeletons += len(additions)
        return True

    def _inactive_identities(
        self, support: Support, antecedents: set[Antecedent] | None = None
    ) -> tuple[RuleIdentity, ...]:
        existing = set(support.antecedents)
        dictionary = self._profiled_dictionary or self.dictionary
        return tuple(
            rule
            for rule in dictionary
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

    def _best_restricted_addition(
        self,
        current: SupportRecord,
        *,
        antecedents: set[Antecedent],
    ) -> SupportRecord | None:
        """Return the certified best score-admissible restricted add block.

        A hierarchy-aware saturated relaxation first removes whole skeletons
        whose *fully reoptimized* support cannot beat the incumbent.  A
        support-conditioned Fenchel certificate then upper-bounds each remaining
        frozen-current block optimum.  Exact block fits are performed lazily in
        decreasing upper-bound order and stop as soon as the exact incumbent is
        no worse than every unvisited bound.  Certificate failure always leaves
        the candidate alive.
        """
        acceptance = current.score
        identities = self._inactive_identities(current.support, antecedents)
        ranked = self._rank_profiled_identities(current, identities)
        admissible = [item for item in ranked if item[0] > self.config.search_tolerance]
        if not admissible:
            return None

        rules = tuple(item[2] for item in admissible)
        safe = set(self._safe_identity_survivors(current, rules, acceptance))
        with self._state_lock:
            self.diagnostics.restricted_relaxation_screens += len(rules) - len(safe)
        admissible = [item for item in admissible if item[2] in safe]
        if not admissible:
            return None

        # The analytic relaxation is already cached by the safe survivor pass.
        # It orders work without constructing a candidate problem.  Conditional
        # dual geometry is created only when the current exact incumbent cannot
        # yet dominate this cheaper upper bound.
        viable = [
            (self.safe_upper_score(current.support.add(rule)), net, gain, rule)
            for net, gain, rule in admissible
        ]
        viable.sort(key=lambda item: (-item[0], -item[1], item[3]))
        devices = self.config.pricing_devices or ("cpu",)
        best_score = acceptance
        best_rule: RuleIdentity | None = None
        audited = 0
        for index, (analytic_upper, _, _, rule) in enumerate(viable):
            if analytic_upper <= best_score + self.config.search_tolerance:
                with self._state_lock:
                    self.diagnostics.lazy_exact_refits_avoided += len(viable) - index
                    self.diagnostics.lazy_bound_stops += 1
                break
            # The cloglog conjugate requires scalar root solves on mixed
            # aggregate rows.  On this model its certified dual solve is more
            # expensive than the exact M-dimensional primal block solve.  Skip
            # that redundant certificate deterministically; the preceding
            # analytic bound remains safe and the exact solve preserves the
            # identical accepted move.  Poisson retains the cheap vectorized
            # dual certificate.
            if self.context.dataset.likelihood != "first_event_cloglog":
                bound = self._restricted_add_bound(current, rule)
                if bound.upper_score <= best_score + self.config.search_tolerance:
                    with self._state_lock:
                        if bound.dual is None:
                            self.diagnostics.restricted_relaxation_screens += 1
                        else:
                            self.diagnostics.restricted_dual_screens += 1
                        self.diagnostics.lazy_exact_refits_avoided += 1
                    continue
            score = self._restricted_add_score(
                current, rule, device=devices[audited % len(devices)]
            )
            audited += 1
            if score > best_score + self.config.search_tolerance or (
                best_rule is not None
                and abs(score - best_score) <= self.config.search_tolerance
                and rule < best_rule
            ):
                best_score = score
                best_rule = rule
        with self._state_lock:
            self.diagnostics.restricted_add_audits += audited
        if best_rule is None:
            return None
        record = self._full_refit_after_restricted_add(current, best_rule)
        if record.score <= acceptance + self.config.search_tolerance:
            raise AssertionError(
                "full refit degraded an improving restricted add block: "
                f"current={current.support!r}, rule={best_rule!r}, "
                f"restricted_score={best_score:.17g}, "
                f"full_score={record.score:.17g}, fit={record.fit!r}"
            )
        return record

    def _full_refit_after_restricted_add(
        self, current: SupportRecord, rule: RuleIdentity
    ) -> SupportRecord:
        """Warm the exact full model with its improving block solution.

        The restricted solve optimizes every newly introduced hierarchy block
        and the candidate rule while freezing existing coefficients.  Embedding
        that feasible point into the full model prevents a failed cold retry
        from turning a proven improving add into a false non-convergence.
        """
        trial = current.support.add(rule)
        with self._state_lock:
            restricted = self._restricted_add_fits.get(current.support, {}).get(rule)
        if restricted is None or not restricted.converged:
            return self.fit(trial, current)

        matrix = self.engine.extend_model_matrix(self.context, trial, current.matrix)
        warm = self.warm_start(current, matrix)
        target_closure, target_rules = self._block_map(matrix)
        new_closure = tuple(sorted(set(matrix.closure) - set(current.matrix.closure)))
        width = self.config.knot_count
        expected = (len(new_closure) + 1) * width
        if len(restricted.coefficients) != expected:
            raise AssertionError("restricted add coefficient layout is inconsistent")
        for index, term in enumerate(new_closure):
            warm[target_closure[term]] = restricted.coefficients[
                index * width : (index + 1) * width
            ]
        warm[target_rules[rule]] = restricted.coefficients[-width:]
        return self.fit(
            trial,
            current,
            warm_start_override=warm,
        )

    def _restricted_add_bound(
        self, current: SupportRecord, rule: RuleIdentity
    ) -> _RestrictedAddBounds:
        """Certified score sandwich for one frozen-current add problem."""
        with self._state_lock:
            cached = self._restricted_add_bounds.get(current.support, {}).get(rule)
            if cached is not None:
                return cached
            self.diagnostics.restricted_bound_evaluations += 1

        trial = current.support.add(rule)
        penalty = self.objective.structural_penalty(trial)
        lower_score = support_score(
            baseline_nll=self.baseline_nll,
            fit_nll=current.fit.nll,
            penalty=penalty,
        )
        problem = self._restricted_add_problem(current, rule)
        certificate: DualCertificate | None = None
        upper_score = self.safe_upper_score(trial)
        if problem is not None:
            design = problem.unsigned_design
            if rule.sign < 0:
                design = design.copy()
                design[:, -self.config.knot_count :] *= -1.0
            certificate = offset_dual_certificate(
                design,
                problem.offset,
                problem.exposure,
                problem.noevent,
                problem.event,
                likelihood=self.context.dataset.likelihood,
                beta=np.zeros(design.shape[1], dtype=np.float64),
                free_dimension=problem.free_dimension,
                tolerance=min(1.0e-9, self.config.solver_tolerance * 0.01),
                max_iter=min(64, self.config.solver_max_iter),
            )
            valid = (
                certificate.feasible
                and certificate.nll_lower_bound
                <= problem.old_nll + 1.0e-7 * max(1.0, abs(problem.old_nll))
            )
            if valid:
                restricted_lower_nll = (
                    current.fit.nll + certificate.nll_lower_bound - problem.old_nll
                )
                dual_upper = support_score(
                    baseline_nll=self.baseline_nll,
                    fit_nll=restricted_lower_nll,
                    penalty=penalty,
                )
                upper_score = min(upper_score, dual_upper)
            else:
                certificate = None
                with self._state_lock:
                    self.diagnostics.restricted_dual_fail_open += 1
        if upper_score + 1.0e-7 * max(1.0, abs(upper_score)) < lower_score:
            # A numerical certificate is never allowed to remove work if its
            # primal/dual sandwich is inconsistent.  The analytic relaxation is
            # independent and remains the fail-open bound.
            certificate = None
            upper_score = self.safe_upper_score(trial)
            with self._state_lock:
                self.diagnostics.restricted_dual_fail_open += 1
        result = _RestrictedAddBounds(
            rule=rule,
            lower_score=lower_score,
            upper_score=upper_score,
            lower_gain=lower_score - current.score,
            upper_gain=upper_score - current.score,
            dual=certificate,
        )
        with self._state_lock:
            incumbent = self._restricted_add_bounds.setdefault(
                current.support, {}
            ).setdefault(rule, result)
        return incumbent

    def _restricted_add_problem(
        self, current: SupportRecord, rule: RuleIdentity
    ) -> _RestrictedAddProblem | None:
        """Materialize a compact offset problem from cached sparse geometry."""
        identity = (rule.antecedent, rule.window)
        event_key = (current.support, rule.antecedent, rule.window)
        while True:
            with self._state_lock:
                state = self._restricted_add_problems.setdefault(current.support, {})
                if identity in state:
                    self.diagnostics.restricted_problem_hits += 1
                    return state[identity]
                pending = self._restricted_add_events.get(event_key)
                if pending is None:
                    pending = threading.Event()
                    self._restricted_add_events[event_key] = pending
                    owner = True
                else:
                    owner = False
            if owner:
                break
            pending.wait()

        problem: _RestrictedAddProblem | None = None
        succeeded = False
        try:
            geometry = self._restricted_geometry(current, rule)
            if geometry is not None:
                raw_offset = self._raw_pricing_components(current)[0][geometry.rows]
                # ``row_patterns`` exactly identifies the unsigned sparse design
                # row.  Aggregating the two-column (offset, pattern-id) key is
                # therefore identical to aggregating (offset, full design), but
                # avoids materializing N x d for every support state.
                keys = np.empty((len(geometry.rows), 2), dtype=np.float64)
                keys[:, 0] = raw_offset
                keys[:, 1] = geometry.row_patterns
                exposure = np.full(
                    len(geometry.rows), self.engine.tick_exposure, dtype=np.float64
                )
                event = geometry.event.copy()
                noevent = (
                    exposure - event
                    if self.context.dataset.likelihood == "first_event_cloglog"
                    else exposure.copy()
                )
                keys, exposure, noevent, event = aggregate_design_rows(
                    keys,
                    exposure,
                    noevent,
                    event,
                    copy_input=False,
                )
                pattern_ids = keys[:, 1].astype(np.int64)
                if np.any(pattern_ids < 0) or np.any(
                    pattern_ids >= len(geometry.design_patterns)
                ):
                    raise AssertionError("restricted pattern id is out of range")
                offset = np.ascontiguousarray(keys[:, 0])
                design = np.ascontiguousarray(geometry.design_patterns[pattern_ids])
                old_values, _, _ = loss_rows(
                    offset,
                    likelihood=self.context.dataset.likelihood,
                    exposure_weight=exposure,
                    noevent_weight=noevent,
                    event_weight=event,
                )
                problem = _RestrictedAddProblem(
                    offset=offset,
                    unsigned_design=design,
                    exposure=exposure,
                    noevent=noevent,
                    event=event,
                    old_nll=float(np.sum(old_values)),
                    free_dimension=geometry.free_dimension,
                )
            succeeded = True
        finally:
            with self._state_lock:
                if succeeded:
                    self._restricted_add_problems.setdefault(current.support, {})[
                        identity
                    ] = problem
                    self.diagnostics.restricted_problem_builds += 1
                completed = self._restricted_add_events.pop(event_key)
                completed.set()
        return problem

    def _restricted_geometry(
        self, current: SupportRecord, rule: RuleIdentity
    ) -> _RestrictedGeometry | None:
        """Return reusable exact row patterns for one hierarchy-aware skeleton."""
        unsigned = RuleIdentity(rule.antecedent, rule.window, 1)
        trial = current.support.add(unsigned)
        old_closure = set(current.matrix.closure)
        new_closure = tuple(sorted(set(hierarchy_closure(trial)) - old_closure))
        key = (new_closure, rule.antecedent, rule.window)
        while True:
            with self._state_lock:
                if key in self._restricted_geometries:
                    geometry = self._restricted_geometries[key]
                    self._restricted_geometries.move_to_end(key)
                    self.diagnostics.restricted_geometry_hits += 1
                    return geometry
                pending = self._restricted_geometry_events.get(key)
                if pending is None:
                    pending = threading.Event()
                    self._restricted_geometry_events[key] = pending
                    owner = True
                else:
                    owner = False
            if owner:
                break
            pending.wait()

        geometry: _RestrictedGeometry | None = None
        succeeded = False
        try:
            specifications = [(term.antecedent, term.window) for term in new_closure]
            specifications.append((rule.antecedent, rule.window))
            blocks = [
                self.engine.block(self.context, antecedent, window)
                for antecedent, window in specifications
            ]
            nonempty = [block.rows for block in blocks if len(block.rows)]
            if nonempty:
                rows = np.unique(np.concatenate(nonempty))
                dimension = len(specifications) * self.config.knot_count
                design = np.zeros((len(rows), dimension), dtype=np.float64)
                for index, block in enumerate(blocks):
                    if not len(block.rows):
                        continue
                    positions = np.searchsorted(rows, block.rows)
                    left = index * self.config.knot_count
                    design[positions, left : left + self.config.knot_count] = (
                        block.values
                    )
                dummy = np.zeros(len(rows), dtype=np.float64)
                patterns, _, _, _, row_patterns = aggregate_design_rows_with_groups(
                    design,
                    dummy,
                    dummy,
                    dummy,
                    copy_input=False,
                )
                event = np.zeros(len(rows), dtype=np.float64)
                if len(self.context.target_rows):
                    positions = np.searchsorted(self.context.target_rows, rows)
                    matched = positions < len(self.context.target_rows)
                    safe = np.minimum(positions, len(self.context.target_rows) - 1)
                    matched &= self.context.target_rows[safe] == rows
                    event[matched] = self.context.target_counts[positions[matched]]
                geometry = _RestrictedGeometry(
                    rows=np.ascontiguousarray(rows),
                    design_patterns=np.ascontiguousarray(patterns),
                    row_patterns=np.ascontiguousarray(row_patterns),
                    event=np.ascontiguousarray(event),
                    free_dimension=len(new_closure) * self.config.knot_count,
                )
            succeeded = True
        finally:
            with self._state_lock:
                if succeeded:
                    self._restricted_geometries[key] = geometry
                    self._restricted_geometries.move_to_end(key)
                    if geometry is not None:
                        self._restricted_geometry_bytes += geometry.nbytes
                    self.diagnostics.restricted_geometry_builds += 1
                    while (
                        self._restricted_geometry_bytes
                        > self._restricted_geometry_limit
                        and self._restricted_geometries
                    ):
                        _, removed = self._restricted_geometries.popitem(last=False)
                        if removed is not None:
                            self._restricted_geometry_bytes -= removed.nbytes
                        self.diagnostics.restricted_geometry_evictions += 1
                completed = self._restricted_geometry_events.pop(key)
                completed.set()
        return geometry

    def _restricted_add_score(
        self,
        current: SupportRecord,
        rule: RuleIdentity,
        *,
        device: str | None = None,
    ) -> float:
        """Exact fixed-current block-coordinate objective for one new rule."""
        with self._state_lock:
            cached = self._restricted_add_scores.get(current.support, {}).get(rule)
        if cached is not None:
            return cached
        trial = current.support.add(rule)
        problem = self._restricted_add_problem(current, rule)
        if problem is None:
            with self._state_lock:
                self._restricted_add_scores.setdefault(current.support, {})[
                    rule
                ] = -math.inf
            return -math.inf
        design = problem.unsigned_design
        if rule.sign < 0:
            design = design.copy()
            design[:, -self.config.knot_count :] *= -1.0
        fit = fit_offset_design(
            design,
            problem.offset,
            problem.exposure,
            problem.noevent,
            problem.event,
            likelihood=self.context.dataset.likelihood,
            free_dimension=problem.free_dimension,
            tolerance=self.config.solver_tolerance,
            max_iter=self.config.solver_max_iter,
            device=device or (self.config.pricing_devices or ("cpu",))[0],
        )
        with self._state_lock:
            self.diagnostics.restricted_block_fits += 1
            self._restricted_add_fits.setdefault(current.support, {})[rule] = fit
        score = -math.inf
        if fit.converged:
            restricted_nll = current.fit.nll + fit.nll - problem.old_nll
            score = support_score(
                baseline_nll=self.baseline_nll,
                fit_nll=restricted_nll,
                penalty=self.objective.structural_penalty(trial),
            )
        with self._state_lock:
            self._restricted_add_scores.setdefault(current.support, {})[rule] = score
        return score

    def _best_restricted_drop(self, current: SupportRecord) -> SupportRecord | None:
        if not current.support.rules:
            return None
        raw_eta = self._raw_pricing_components(current)[0]
        candidates: list[tuple[float, Support]] = []
        current_baseline = (
            current.matrix.free_dimension - current.matrix.closure_dimension
        )
        width = self.config.knot_count
        current_rule_slices = dict(
            zip(current.support.rules, current.matrix.rule_slices, strict=True)
        )

        def score_trial(trial: Support) -> float:
            with self._state_lock:
                cached = self._restricted_drop_scores.get(current.support, {}).get(
                    trial
                )
            if cached is not None:
                return cached
            trial_closure = set(hierarchy_closure(trial))
            effects: list[tuple[SparseBlock, np.ndarray]] = []
            for index, term in enumerate(current.matrix.closure):
                if term in trial_closure:
                    continue
                left = current_baseline + index * width
                effects.append(
                    (
                        self.engine.block(self.context, term.antecedent, term.window),
                        current.fit.coefficients[left : left + width],
                    )
                )
            trial_rules = set(trial.rules)
            for rule in current.support.rules:
                if rule in trial_rules:
                    continue
                promoted = ClosureTerm(rule.antecedent, rule.window) in trial_closure
                if not promoted:
                    effects.append(
                        (
                            self.engine.block(
                                self.context, rule.antecedent, rule.window
                            ),
                            float(rule.sign)
                            * current.fit.coefficients[current_rule_slices[rule]],
                        )
                    )
            nonempty = [block.rows for block, _ in effects if len(block.rows)]
            if nonempty:
                rows = np.unique(np.concatenate(nonempty))
                removed = np.zeros(len(rows), dtype=np.float64)
                for block, coefficient in effects:
                    if len(block.rows):
                        removed[np.searchsorted(rows, block.rows)] += (
                            block.values @ coefficient
                        )
                old_eta = raw_eta[rows]
                new_eta = old_eta - removed
                event = np.zeros(len(rows), dtype=np.float64)
                if len(self.context.target_rows):
                    positions = np.searchsorted(self.context.target_rows, rows)
                    matched = positions < len(self.context.target_rows)
                    safe = np.minimum(positions, len(self.context.target_rows) - 1)
                    matched &= self.context.target_rows[safe] == rows
                    event[matched] = self.context.target_counts[positions[matched]]
                exposure = np.full(
                    len(rows), self.engine.tick_exposure, dtype=np.float64
                )
                noevent = (
                    exposure - event
                    if self.context.dataset.likelihood == "first_event_cloglog"
                    else exposure
                )
                old_loss, _, _ = loss_rows(
                    old_eta,
                    likelihood=self.context.dataset.likelihood,
                    exposure_weight=exposure,
                    noevent_weight=noevent,
                    event_weight=event,
                )
                new_loss, _, _ = loss_rows(
                    new_eta,
                    likelihood=self.context.dataset.likelihood,
                    exposure_weight=exposure,
                    noevent_weight=noevent,
                    event_weight=event,
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = float(np.sum(new_loss - old_loss, dtype=np.longdouble))
                nll = current.fit.nll + delta
            else:
                nll = current.fit.nll
            score = support_score(
                baseline_nll=self.baseline_nll,
                fit_nll=nll,
                penalty=self.objective.structural_penalty(trial),
            )
            with self._state_lock:
                self._restricted_drop_scores.setdefault(current.support, {})[trial] = (
                    score
                )
            return score

        for rule in current.support.rules:
            trial = current.support.drop(rule)
            candidates.append((score_trial(trial), trial))
        with self._state_lock:
            self.diagnostics.restricted_drop_audits += len(candidates)
        improving = sorted(
            [
                (score, trial)
                for score, trial in candidates
                if score > current.score + self.config.search_tolerance
            ],
            key=lambda item: (-item[0], item[1].rules),
        )
        selected = [
            next(iter(set(current.support.rules) - set(trial.rules)))
            for _, trial in improving
        ]
        while selected:
            removed = set(selected)
            trial = Support.of(
                rule for rule in current.support.rules if rule not in removed
            )
            score = score_trial(trial)
            with self._state_lock:
                self.diagnostics.restricted_drop_audits += 1
            if score <= current.score + self.config.search_tolerance:
                selected.pop()
                continue
            record = self.fit(trial, current)
            if record.fit.converged and (
                record.score > current.score + self.config.search_tolerance
            ):
                return record
            selected.pop()
        return None

    def _standalone_profiled_atoms(
        self, empty: SupportRecord
    ) -> tuple[SupportRecord, ...]:
        """Return objective-admissible single-skeleton discovery roots.

        W/sign are profiled once within every score-basin skeleton, so a root
        contains exactly one identity.  A higher-order root must improve both
        the baseline MDL and its exact closure-only null; otherwise automatic
        hierarchy nuisance, rather than the reported branch, created the gain.
        """
        ranked = self._rank_profiled_identities(empty, self.dictionary)
        self._profiled_dictionary = tuple(
            self._skeleton_witnesses[antecedent][1]
            for antecedent in self.skeletons
            if antecedent in self._skeleton_witnesses
        )
        if len(self._profiled_dictionary) != len(self.skeletons):
            raise AssertionError("baseline W/sign profile did not cover every skeleton")
        admissible = self._objective_root_candidates(ranked)
        if admissible:
            safe_rules = set(
                self._safe_identity_survivors(
                    empty, tuple(item[2] for item in admissible), 0.0
                )
            )
            admissible = [item for item in admissible if item[2] in safe_rules]
        standalone: dict[Support, SupportRecord] = {}
        for start in range(0, len(admissible), self.config.exact_workers):
            wave = admissible[start : start + self.config.exact_workers]
            records = self.fit_many([Support.of((item[2],)) for item in wave], empty)
            for record in records:
                if record.score > self.config.search_tolerance:
                    standalone[record.support] = record
                else:
                    with self._state_lock:
                        self.diagnostics.block_score_exact_rejections += 1

        ordered = [
            standalone[support]
            for support in sorted(standalone, key=lambda item: item.rules)
        ]
        null_fits = self._fit_embedded_closure_nulls(ordered)
        positives: dict[Support, SupportRecord] = {}
        log_n = math.log(max(2, self.objective.n_entities))
        for record, null_fit in zip(ordered, null_fits, strict=True):
            closure_code = len(record.matrix.closure) * self.config.knot_count * log_n
            branch_code = record.penalty - closure_code
            if branch_code < -1.0e-8:
                raise AssertionError("standalone branch MDL code is negative")
            branch_score = (
                2.0 * (null_fit.nll - record.fit.nll) - max(0.0, branch_code)
                if null_fit.converged
                else -math.inf
            )
            with self._state_lock:
                self.diagnostics.standalone_branch_audits += 1
            if branch_score > self.config.search_tolerance:
                positives[record.support] = record
            else:
                with self._state_lock:
                    self.diagnostics.standalone_branch_rejections += 1

        # Rejected closure-driven seeds must not inflate every multi-source
        # working set.  A complete delayed-dictionary audit remains mandatory
        # at a stalled state, so this changes cost but not the terminal audit.
        self._working_antecedents = {
            record.support.rules[0].antecedent for record in positives.values()
        }
        return tuple(
            positives[support]
            for support in sorted(positives, key=lambda item: item.rules)
        )

    def _best_profiled_move(self, current: SupportRecord) -> SupportRecord | None:
        """Audit score-admissible adds and every active drop.

        Each inactive skeleton selects one W/sign under the conditional-Fisher
        block score.  Every positive profiled identity is then either safely
        bounded below the incumbent or audited with common effects frozen; every
        active drop is audited.  Terminal states are profiled block-score
        restricted add/drop fixed points, not exact one-exchange optima: an
        unprofiled identity or later W/sign replacement is not certified.
        """
        # Delayed column generation prices only the finite basin working set
        # during coordinate moves.  A complete outside-dictionary audit is
        # compulsory whenever that restricted problem stalls, and every
        # violating skeleton is activated at once.  Hence terminal states have
        # still passed the complete dictionary audit without paying for it at
        # every accepted move.
        direct = self._best_restricted_addition(
            current, antecedents=set(self._working_antecedents)
        )
        if direct is not None:
            return direct
        dropped = self._best_restricted_drop(current)
        if dropped is not None:
            return dropped
        if self._expand_working_set(current):
            direct = self._best_restricted_addition(
                current, antecedents=set(self._working_antecedents)
            )
            if direct is not None:
                return direct
        self.diagnostics.terminal_audits += 1
        self.diagnostics.restricted_block_terminals += 1
        return None

    def search(self) -> SearchResult:
        positive_atoms: dict[Support, SupportRecord] = {}
        empty = self.records[EMPTY_SUPPORT]
        for record in self._standalone_profiled_atoms(empty):
            positive_atoms[record.support] = record

        # Empty plus every objective-admissible singleton/pair/triplet root
        # defines the multi-source search.  There is no top-k or restart budget.
        # High-score roots run first so their exact transition chains are most
        # likely to be reused by later roots.
        root_records = sorted(
            positive_atoms.values(),
            key=lambda record: (-record.score, record.support.rules),
        )
        starts = [empty, *root_records]
        self.diagnostics.multi_source_roots = len(starts)
        transition_cache: dict[Support, Support | None] = {}
        resolved_terminal: dict[Support, Support] = {}
        terminals: dict[Support, SupportRecord] = {}
        paths: list[dict[str, object]] = []
        for start_record in starts:
            current = start_record
            moves: list[dict[str, object]] = []
            visited: list[Support] = []
            while True:
                self.diagnostics.states += 1
                compressed = resolved_terminal.get(current.support)
                if compressed is not None:
                    self.diagnostics.path_compression_hits += 1
                    terminal_record = terminals[compressed]
                    if terminal_record.support != current.support:
                        moves.append(
                            {
                                "from": support_key(current.support),
                                "to": support_key(terminal_record.support),
                                "gain": float(terminal_record.score - current.score),
                                "cached_path": True,
                            }
                        )
                    current = terminal_record
                    break
                visited.append(current.support)
                if current.support in transition_cache:
                    self.diagnostics.transition_cache_hits += 1
                    next_support = transition_cache[current.support]
                    if next_support is None:
                        break
                    next_record = self.fit(next_support, current)
                else:
                    next_record = self._best_profiled_move(current)
                    transition_cache[current.support] = (
                        None if next_record is None else next_record.support
                    )
                    # This exact transition is now immutable in the shared DAG;
                    # per-identity audit values for the state can be released.
                    self._restricted_add_scores.pop(current.support, None)
                    self._restricted_add_fits.pop(current.support, None)
                    self._restricted_add_bounds.pop(current.support, None)
                    self._restricted_drop_scores.pop(current.support, None)
                    self._restricted_add_problems.pop(current.support, None)
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
                for support in visited:
                    resolved_terminal[support] = current.support
            paths.append(
                {
                    "start": support_key(start_record.support),
                    "terminal": support_key(current.support),
                    "moves": moves,
                }
            )
        # Every exact-positive atom and every terminal support is frozen for
        # F0--F2.  In particular, higher-order atoms cannot be reported merely
        # because their automatic closure fitted well: F2 must validate their
        # hierarchy-preserving branch contribution on D_cert.
        family_map = dict(positive_atoms)
        family_map.update(terminals)
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
