from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .dual import DualCertificate, dual_certificate
from .likelihood import loss_rows
from .native import moments
from .objective import ObjectiveSpec, SupportRecord, support_score
from .response import Context, ModelMatrix, ResponseEngine
from .rules import (
    EMPTY_SUPPORT,
    Antecedent,
    RuleIdentity,
    Support,
    hierarchy_closure,
    skeletons,
)
from .solver import fit_model_matrix


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


@dataclass(frozen=True)
class SearchResult:
    family: tuple[SupportRecord, ...]
    terminals: tuple[SupportRecord, ...]
    positive_atoms: tuple[SupportRecord, ...]
    paths: tuple[dict[str, object], ...]
    diagnostics: SearchDiagnostics


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
    """Exact fixed-support fits with safely screened block-splicing search.

    Fisher prices only order work.  Every rejection that affects the terminal
    claim is justified by a support-independent saturated-loss bound, a
    numerically rechecked Fenchel certificate, or an exact refit.
    """

    def __init__(self, context: Context, config: RunConfig):
        self.context = context
        self.config = config
        self.engine = ResponseEngine(
            context.dataset,
            lag=config.impact_lag,
            knot_count=config.knot_count,
            cache_bytes=config.cache_bytes,
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
        self.records: dict[Support, SupportRecord] = {}
        self._addition_cache: dict[Support, SupportRecord | None] = {}
        self._feature_rows_cache: dict[Support, np.ndarray] = {}
        self._relaxed_upper_cache: dict[Support, float] = {}
        self._pricing_state: dict[
            Support, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
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
        self.records[EMPTY_SUPPORT] = SupportRecord(
            EMPTY_SUPPORT, baseline_matrix, baseline_fit, 0.0, 0.0
        )
        self.diagnostics.exact_fits += 1

    def _structurally_admissible_dictionary(
        self, all_skeletons: tuple[Antecedent, ...]
    ) -> tuple[tuple[Antecedent, ...], tuple[RuleIdentity, ...]]:
        """Remove only empty atoms and exactly response-equivalent W values."""
        admitted_skeletons: list[Antecedent] = []
        admitted_rules: list[RuleIdentity] = []
        for antecedent in all_skeletons:
            windows = (0,) if len(antecedent) == 1 else self.config.formation_windows
            distinct: list[tuple[np.ndarray, np.ndarray]] = []
            effective_windows: list[int] = []
            for window in windows:
                block = self.engine.block(self.context, antecedent, int(window))
                if not len(block.rows):
                    continue
                equivalent = any(
                    np.array_equal(block.rows, rows)
                    and np.array_equal(block.values, values)
                    for rows, values in distinct
                )
                if equivalent:
                    self.diagnostics.equivalent_window_identities += 2
                    continue
                distinct.append((block.rows, block.values))
                effective_windows.append(int(window))
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
        score = (
            support_score(
                baseline_nll=self.baseline_nll, fit_nll=fit.nll, penalty=penalty
            )
            if fit.converged
            else -math.inf
        )
        record = SupportRecord(support, matrix, fit, penalty, score)
        self.records[support] = record
        self.diagnostics.exact_fits += 1
        return record

    def saturated_upper_score(self, support: Support) -> float:
        return support_score(
            baseline_nll=self.baseline_nll,
            fit_nll=self.saturated_nll_lower_bound,
            penalty=self.objective.structural_penalty(support),
        )

    def _feature_rows(self, support: Support) -> np.ndarray:
        cached = self._feature_rows_cache.get(support)
        if cached is not None:
            return cached
        parts = [
            self.engine.block(self.context, term.antecedent, term.window).rows
            for term in hierarchy_closure(support)
        ]
        parts.extend(
            self.engine.block(self.context, rule.antecedent, rule.window).rows
            for rule in support.rules
        )
        nonempty = [rows for rows in parts if len(rows)]
        result = (
            np.unique(np.concatenate(nonempty))
            if nonempty
            else np.zeros(0, dtype=np.int64)
        )
        self._feature_rows_cache[support] = result
        return result

    def localized_upper_score(self, support: Support) -> float:
        """Safe score upper bound from a localized saturated relaxation.

        Every row touched by any support or closure feature receives its own
        unrestricted predictor; untouched rows retain one common intercept.
        This relaxed model contains the exact support model, so its minimum
        NLL is a rigorous lower bound on the exact minimum NLL.
        """
        cached = self._relaxed_upper_cache.get(support)
        if cached is not None:
            return cached
        affected = self._feature_rows(support)
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
        score = min(score, self.saturated_upper_score(support))
        self._relaxed_upper_cache[support] = score
        return score

    def bounds(self, current: SupportRecord, trial: Support) -> ProposalBounds:
        """Return a feasible-primal lower score and verified-dual upper score."""
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
        certificate = dual_certificate(
            matrix,
            likelihood=self.context.dataset.likelihood,
            beta=warm,
            tolerance=min(1.0e-9, self.config.solver_tolerance * 0.01),
        )
        if not certificate.feasible:
            # A single damped projected-Newton splice remains a feasible
            # primal point and usually places the score dual inside the new
            # closure/cone geometry.  This is invoked only after the cheap
            # warm certificate fails; it is not an acceptance shortcut.
            one_step = fit_model_matrix(
                matrix,
                likelihood=self.context.dataset.likelihood,
                tolerance=self.config.solver_tolerance,
                max_iter=1,
                warm_start=warm,
            )
            one_eta = matrix.x @ one_step.coefficients
            one_rows, _, _ = loss_rows(
                one_eta,
                likelihood=self.context.dataset.likelihood,
                exposure_weight=matrix.exposure_weight,
                noevent_weight=matrix.noevent_weight,
                event_weight=matrix.event_weight,
            )
            one_nll = float(np.sum(one_rows))
            if one_nll < feasible_nll:
                warm = one_step.coefficients
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
            self.diagnostics.dual_fail_open += 1
            upper_score = self.localized_upper_score(trial)
            certificate = None
        # The analytic saturated likelihood bound is always valid and can
        # tighten a loose numerical Fenchel certificate.
        upper_score = min(upper_score, self.localized_upper_score(trial))
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

    def _pricing_components(
        self, current: SupportRecord
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._pricing_state.get(current.support)
        if cached is not None:
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
        result = indices, inverse, eta
        self._pricing_state[current.support] = result
        return result

    def block_price(
        self, current: SupportRecord, rule: RuleIdentity, *, device: str = "cpu"
    ) -> float:
        """Conditional-Fisher price used solely for deterministic work ordering."""
        block = self.engine.block(self.context, rule.antecedent, rule.window)
        if not len(block.rows):
            return 0.0
        indices, inverse, _ = self._pricing_components(current)
        design = self.engine.design_at_rows_with_context(
            self.context, current.matrix, block.rows
        )
        eta = design @ current.fit.coefficients
        positions = np.searchsorted(self.context.target_rows, block.rows)
        matched = positions < len(self.context.target_rows)
        if len(self.context.target_rows):
            safe = np.minimum(positions, len(self.context.target_rows) - 1)
            matched &= self.context.target_rows[safe] == block.rows
        event = np.zeros(len(block.rows), dtype=np.float64)
        if np.any(matched):
            event[matched] = self.context.target_counts[positions[matched]]
        exposure = np.full(len(block.rows), self.engine.tick_exposure, dtype=np.float64)
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
        candidate = float(rule.sign) * block.values
        gradient, hessian = moments(candidate, first, second, device=device)
        if len(indices):
            cross = design[:, indices].T @ (second[:, None] * candidate)
            hessian = hessian - cross.T @ inverse @ cross
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        eigenvalues = np.maximum(eigenvalues, scale * 1.0e-12)
        conditioned = (eigenvectors * eigenvalues) @ eigenvectors.T
        return _nonnegative_quadratic_gain(gradient, conditioned)

    def _rank_identities(
        self, current: SupportRecord, identities: tuple[RuleIdentity, ...]
    ) -> list[tuple[float, RuleIdentity]]:
        self.diagnostics.pricing_passes += 1
        self.diagnostics.priced_blocks += len(identities)
        # Build the common conditional information once before worker launch.
        self._pricing_components(current)
        devices = self.config.pricing_devices
        if devices and len(identities) > 1:

            def price(
                index_rule: tuple[int, RuleIdentity],
            ) -> tuple[float, RuleIdentity]:
                index, rule = index_rule
                device = devices[index % len(devices)]
                return self.block_price(current, rule, device=device), rule

            with ThreadPoolExecutor(
                max_workers=min(self.config.pricing_workers, len(identities)),
                thread_name_prefix="crbstpp-pricing",
            ) as executor:
                prices = list(executor.map(price, enumerate(identities)))
        else:
            prices = [(self.block_price(current, rule), rule) for rule in identities]
        return sorted(prices, key=lambda item: (-item[0], item[1]))

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

    def _best_addition(
        self,
        current: SupportRecord,
        *,
        antecedents: set[Antecedent] | None = None,
    ) -> SupportRecord | None:
        if antecedents is None and current.support in self._addition_cache:
            return self._addition_cache[current.support]
        identities = self._inactive_identities(current.support, antecedents)
        if not identities:
            if antecedents is None:
                self._addition_cache[current.support] = None
            return None
        ranked = self._rank_identities(current, identities)
        bounds: list[tuple[float, ProposalBounds]] = []
        for price, rule in ranked:
            trial = current.support.add(rule)
            if (
                self.localized_upper_score(trial)
                <= current.score + self.config.search_tolerance
            ):
                self.diagnostics.relaxation_screens += 1
                continue
            bounds.append((price, self.bounds(current, trial)))
        bounds.sort(
            key=lambda item: (-item[1].upper_score, -item[0], item[1].support.rules)
        )
        incumbent: SupportRecord | None = None
        incumbent_score = current.score
        for _, item in bounds:
            if item.upper_score <= incumbent_score + self.config.search_tolerance:
                self.diagnostics.dual_screens += 1
                continue
            record = self.fit(item.support, current)
            if record.score > incumbent_score + self.config.search_tolerance:
                incumbent, incumbent_score = record, record.score
        if antecedents is None:
            self._addition_cache[current.support] = incumbent
        return incumbent

    def _standalone_skeleton(
        self, empty: SupportRecord, antecedent: Antecedent
    ) -> tuple[SupportRecord, ...]:
        identities = tuple(
            rule for rule in self.dictionary if rule.antecedent == antecedent
        )
        ranked = self._rank_identities(empty, identities)
        positives: dict[Support, SupportRecord] = {}
        pending: list[tuple[float, ProposalBounds]] = []
        for price, rule in ranked:
            trial = Support.of((rule,))
            saturated = self.localized_upper_score(trial)
            if saturated <= self.config.search_tolerance:
                self.diagnostics.relaxation_screens += 1
                continue
            pending.append((price, self.bounds(empty, trial)))
        pending.sort(
            key=lambda item: (-item[1].upper_score, -item[0], item[1].support.rules)
        )
        for _, item in pending:
            may_be_positive = item.upper_score > self.config.search_tolerance
            if not may_be_positive:
                self.diagnostics.dual_screens += 1
                continue
            record = self.fit(item.support, empty)
            if record.score > 0:
                positives[record.support] = record
        return tuple(positives.values())

    def _best_exact_proposal(
        self, current: SupportRecord, proposals: list[Support]
    ) -> SupportRecord | None:
        unique = sorted(set(proposals), key=lambda support: support.rules)
        bounded: list[ProposalBounds] = []
        for support in unique:
            if support == current.support:
                continue
            if (
                self.localized_upper_score(support)
                <= current.score + self.config.search_tolerance
            ):
                self.diagnostics.relaxation_screens += 1
                continue
            bounded.append(self.bounds(current, support))
        bounded.sort(key=lambda item: (-item.upper_score, item.support.rules))
        incumbent = None
        incumbent_score = current.score
        for item in bounded:
            if item.upper_score <= incumbent_score + self.config.search_tolerance:
                self.diagnostics.dual_screens += 1
                continue
            record = self.fit(item.support, current)
            if record.score > incumbent_score + self.config.search_tolerance:
                incumbent, incumbent_score = record, record.score
        return incumbent

    def _splice_proposals(
        self, current: SupportRecord, drops: list[tuple[RuleIdentity, SupportRecord]]
    ) -> list[Support]:
        identities = self._inactive_identities(current.support)
        if len(identities) < 2:
            return []
        ranked = self._rank_identities(current, identities)
        best_by_antecedent: dict[Antecedent, tuple[float, RuleIdentity]] = {}
        for price, rule in ranked:
            previous = best_by_antecedent.get(rule.antecedent)
            if previous is None or (price, rule) > previous:
                best_by_antecedent[rule.antecedent] = (price, rule)
        additions = [
            item[1]
            for item in sorted(
                best_by_antecedent.values(), key=lambda item: (-item[0], item[1])
            )
        ]
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
        drops: list[tuple[RuleIdentity, SupportRecord]] = []
        for removed in current.support.rules:
            reduced = self.fit(current.support.drop(removed), current)
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

    def search(self) -> SearchResult:
        positive_atoms: dict[Support, SupportRecord] = {}
        starts: list[Support] = [EMPTY_SUPPORT]
        empty = self.records[EMPTY_SUPPORT]
        # The DAG frontier prices every W/sign atom.  One basin witness per
        # skeleton plus every positive standalone atom is retained; joint
        # splicing from empty preserves access to suppressor combinations.
        for antecedent in self.skeletons:
            positives = self._standalone_skeleton(empty, antecedent)
            for record in positives:
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
                    next_record = self._best_one_exchange(current)
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
