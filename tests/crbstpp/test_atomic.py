from __future__ import annotations

import itertools
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from crbstpp.atomic import (
    AtomicSignatureRelaxation,
    CertifiedSidetrackQueue,
    minimum_descendant_penalty,
)
from crbstpp.config import RunConfig
from crbstpp.objective import ObjectiveSpec
from crbstpp.response import Context
from crbstpp.rules import EMPTY_SUPPORT, RuleIdentity, Support
from crbstpp.search import SearchDiagnostics, SupportOptimizer
from tests.crbstpp.test_higher_order import cell_dataset, optimizer


class AtomicSubtreeTests(unittest.TestCase):
    def test_empty_route_geometry_uses_baseline_batch(self) -> None:
        fitted = object.__new__(SupportOptimizer)
        fitted._baseline_batched_component_items = mock.Mock(return_value=[])
        fitted._support_batched_component_items = mock.Mock(
            side_effect=AssertionError("empty state reached conditional batch")
        )
        current = SimpleNamespace(support=EMPTY_SUPPORT)
        groups = [(('atomic', (0,)), (0,))]
        self.assertEqual(
            fitted._route_batched_component_items(current, groups, ('cuda:0',)),
            [],
        )
        fitted._baseline_batched_component_items.assert_called_once_with(
            current, groups, ('cuda:0',)
        )
        fitted._support_batched_component_items.assert_not_called()

    @staticmethod
    def _three_predicate_data(root: Path):
        return cell_dataset(
            root,
            3,
            {
                (0, 0, 0): 0.10,
                (0, 0, 1): 0.14,
                (0, 1, 0): 0.18,
                (0, 1, 1): 0.34,
                (1, 0, 0): 0.20,
                (1, 0, 1): 0.38,
                (1, 1, 0): 0.46,
                (1, 1, 1): 0.72,
            },
            per_cell=50,
        )

    def test_descendant_penalty_matches_brute_force(self) -> None:
        antecedents = ((0,), (1,), (0, 1), (0, 1, 2))
        spec = ObjectiveSpec(
            n_entities=240,
            skeleton_count=9,
            knot_count=3,
            window_count_by_order=(1, 4, 2),
            window_count_by_antecedent=tuple(
                (antecedent, count)
                for antecedent, count in zip(
                    antecedents, (1, 3, 2, 4), strict=True
                )
            ),
        )
        required = Support.of((RuleIdentity((0,), 0, 1),))
        optional = antecedents[1:]
        observed = minimum_descendant_penalty(spec, required, optional)
        brute = math.inf
        for selected in itertools.chain.from_iterable(
            itertools.combinations(optional, size)
            for size in range(len(optional) + 1)
        ):
            trial = required
            for antecedent in selected:
                trial = trial.add(RuleIdentity(antecedent, 0, -1))
            brute = min(brute, spec.structural_penalty(trial))
        self.assertAlmostEqual(observed, brute, places=12)

    def test_block_score_terminal_mode_skips_global_exact_add_frontier(self) -> None:
        """Fast discovery retains exact terminals without replaying exact Adds."""

        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fitted = optimizer(
                data,
                3,
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
                terminal_add_audit="block_score",
            )
            fitted.config = replace(
                fitted.config,
                conditional_basin_branching=True,
                fisher_separated_rashomon=True,
                online_predicate_pareto_frontier=True,
            )
            try:
                with mock.patch.object(
                    fitted,
                    "_global_exact_family_frontier",
                    side_effect=AssertionError("global exact Add frontier was called"),
                ):
                    result = fitted.search()
                self.assertTrue(result.terminals)
                self.assertTrue(all(item.fit.converged for item in result.terminals))
                self.assertEqual(
                    result.diagnostics.block_score_terminal_add_audits,
                    len(result.terminals),
                )
                self.assertEqual(
                    result.diagnostics.duplicate_exact_family_frontiers_avoided,
                    1,
                )
                self.assertTrue(
                    all(
                        path.get("terminal_add_audit")
                        == "complete_dictionary_block_score"
                        for path in result.paths
                    )
                )
            finally:
                fitted.close()

    def test_sparse_fisher_distance_uses_common_union_geometry(self) -> None:
        left = (
            np.asarray((1, 3), dtype=np.int64),
            np.asarray((2.0, 1.0), dtype=np.float64),
        )
        right = (
            np.asarray((2, 3), dtype=np.int64),
            np.asarray((4.0, -1.0), dtype=np.float64),
        )
        self.assertAlmostEqual(
            SupportOptimizer._sparse_fisher_distance(left, right),
            24.0,
            places=12,
        )

    def test_online_predicate_pareto_orders_without_pruning(self) -> None:
        fitted = object.__new__(SupportOptimizer)
        fitted.config = SimpleNamespace(
            online_predicate_pareto_frontier=True,
            search_tolerance=1.0e-8,
        )
        fitted._online_pareto_reference_predicates = frozenset()
        fitted._state_lock = mock.MagicMock()
        fitted._state_lock.__enter__.return_value = None
        fitted._state_lock.__exit__.return_value = False
        fitted.diagnostics = SearchDiagnostics()
        current = SimpleNamespace(
            support=Support.of((RuleIdentity((0,), 0, 1),))
        )
        overlapping = (10.0, 10.0, RuleIdentity((0, 1), 0, 1))
        novel = (8.0, 8.0, RuleIdentity((2,), 0, 1))
        dominated = (7.0, 7.0, RuleIdentity((0, 3), 0, 1))
        original = [overlapping, novel, dominated]

        ordered = fitted._online_predicate_pareto_order(current, original)

        self.assertEqual(
            [item[2] for item in ordered],
            [novel[2], overlapping[2], dominated[2]],
        )
        self.assertEqual(set(map(id, ordered)), set(map(id, original)))
        self.assertEqual(fitted.diagnostics.predicate_pareto_rankings, 1)
        self.assertEqual(fitted.diagnostics.predicate_pareto_candidates, 3)
        self.assertEqual(
            fitted.diagnostics.predicate_pareto_first_front_candidates,
            2,
        )

    def test_online_predicate_pareto_reference_changes_novelty(self) -> None:
        fitted = object.__new__(SupportOptimizer)
        fitted.config = SimpleNamespace(
            online_predicate_pareto_frontier=True,
            search_tolerance=1.0e-8,
        )
        fitted._online_pareto_reference_predicates = frozenset((2,))
        fitted._state_lock = mock.MagicMock()
        fitted._state_lock.__enter__.return_value = None
        fitted._state_lock.__exit__.return_value = False
        fitted.diagnostics = SearchDiagnostics()
        current = SimpleNamespace(
            support=Support.of((RuleIdentity((0,), 0, 1),))
        )
        formerly_novel = (8.0, 8.0, RuleIdentity((2,), 0, 1))
        new_direction = (7.0, 7.0, RuleIdentity((3,), 0, 1))

        ordered = fitted._online_predicate_pareto_order(
            current, [formerly_novel, new_direction]
        )

        self.assertEqual(ordered[0][2], new_direction[2])

    def test_predicate_coverage_selects_one_admissible_root_per_primitive(self) -> None:
        fitted = object.__new__(SupportOptimizer)
        fitted._state_lock = mock.MagicMock()
        fitted._state_lock.__enter__.return_value = None
        fitted._state_lock.__exit__.return_value = False
        fitted.diagnostics = SearchDiagnostics()
        fitted._predicate_coverage_root_alternatives = {}
        dominant = (10.0, 10.0, RuleIdentity((0, 1), 0, 1))
        second = (8.0, 8.0, RuleIdentity((1, 2), 0, 1))
        third = (6.0, 6.0, RuleIdentity((3,), 0, -1))

        selected = fitted._predicate_coverage_route_patterns(
            (dominant, second, third), {dominant[2].pattern_key}
        )

        self.assertEqual(
            selected,
            {
                dominant[2].pattern_key,
                second[2].pattern_key,
                third[2].pattern_key,
            },
        )
        self.assertEqual(
            set(fitted._predicate_coverage_root_alternatives), {0, 1, 2, 3}
        )
        self.assertEqual(fitted.diagnostics.predicate_coverage_primitives, 4)

    def test_predicate_coverage_releases_anchor_before_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fitted = optimizer(
                data,
                3,
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
                terminal_add_audit="block_score",
            )
            fitted.config = replace(
                fitted.config,
                conditional_basin_branching=True,
                fisher_separated_rashomon=True,
                online_predicate_pareto_frontier=False,
                predicate_coverage_rashomon=True,
            )
            try:
                result = fitted.search()
                self.assertTrue(result.paths)
                self.assertTrue(
                    all(path.get("anchor_antecedents") for path in result.paths)
                )
                self.assertTrue(
                    all(path.get("anchor_released") for path in result.paths)
                )
                self.assertEqual(
                    result.diagnostics.predicate_coverage_anchor_releases,
                    len(result.paths),
                )
                self.assertGreaterEqual(
                    result.diagnostics.predicate_coverage_exact_positive_roots,
                    1,
                )
            finally:
                fitted.close()

    def test_block_score_terminal_reopens_complete_w_sign_dictionary(self) -> None:
        """Terminal cleanup must not reuse only standalone-profiled identities."""

        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fitted = optimizer(
                data,
                3,
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
                terminal_add_audit="block_score",
            )
            try:
                current = fitted.records[EMPTY_SUPPORT]
                observed: list[tuple[bool, bool, bool, bool]] = []

                def no_add(
                    _current: object,
                    *,
                    antecedents: object,
                    frozen_identities: bool = False,
                    composite_cleanup: bool = False,
                ) -> None:
                    del antecedents
                    observed.append(
                        (
                            frozen_identities,
                            fitted._force_exact_candidate_validation,
                            fitted._terminal_add_audit_active,
                            composite_cleanup,
                        )
                    )
                    return None

                with (
                    mock.patch.object(
                        fitted,
                        "_best_exact_nonpositive_branch_drop",
                        return_value=None,
                    ),
                    mock.patch.object(
                        fitted, "_best_exact_rule_objective_drop", return_value=None
                    ),
                    mock.patch.object(
                        fitted, "_best_final_identity_change", return_value=None
                    ),
                    mock.patch.object(
                        fitted, "_first_conditional_addition", side_effect=no_add
                    ),
                ):
                    decision = fitted._best_terminal_cleanup_decision(
                        current,
                        audit_add=True,
                    )
                self.assertIsNone(decision.record)
                self.assertEqual(observed, [(False, False, False, True)])
            finally:
                fitted.close()

    def test_terminal_reliability_cleanup_drops_only_nonrobust_rule(self) -> None:
        """A pooled-MDL-positive branch cannot survive a failed D_fit guard."""

        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fitted = optimizer(
                data,
                3,
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
                terminal_add_audit="block_score",
            )
            try:
                fitted.config = replace(
                    fitted.config,
                    reliability_aware_search=True,
                )
                empty = fitted.records[EMPTY_SUPPORT]
                rule_a = RuleIdentity((0,), 0, 1)
                rule_b = RuleIdentity((1,), 0, 1)
                current = fitted.fit(Support.of((rule_a, rule_b)), empty)
                self.assertTrue(current.fit.converged, current.fit.message)

                def robust(
                    _parent: object,
                    _child: object,
                    rule: RuleIdentity,
                ) -> bool:
                    return rule != rule_b

                with mock.patch.object(
                    fitted, "_fit_add_is_material", side_effect=robust
                ):
                    dropped = fitted._best_exact_nonrobust_rule_drop(
                        current,
                        protected_antecedents=frozenset(),
                    )
                self.assertIsNotNone(dropped)
                assert dropped is not None
                self.assertEqual(dropped.support, Support.of((rule_a,)))
                self.assertIn(
                    (current.support, dropped.support),
                    fitted._terminal_forced_reliability_drops,
                )
                self.assertIn(
                    (dropped.support, current.support),
                    fitted._conditional_parent_forbidden,
                )
            finally:
                fitted.close()

    def test_composite_add_rejects_child_whose_cleanup_returns_parent(self) -> None:
        """An Add-cleanup corridor back to S is not accepted as a route move."""

        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fitted = optimizer(
                data,
                3,
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
                terminal_add_audit="block_score",
            )
            try:
                current = fitted.records[EMPTY_SUPPORT]
                rule = RuleIdentity((0,), 0, 1)
                proposal = fitted.fit(Support.of((rule,)), current)
                self.assertTrue(proposal.fit.converged, proposal.fit.message)
                with (
                    mock.patch.object(
                        fitted,
                        "_exact_add_branch_validation",
                        return_value=(proposal, 1.0),
                    ),
                    mock.patch.object(
                        fitted, "_fit_add_is_material", return_value=True
                    ),
                    mock.patch.object(
                        fitted, "_add_is_indecomposable", return_value=True
                    ),
                    mock.patch.object(
                        fitted, "_frequency_evidence_for_move", return_value=None
                    ),
                    mock.patch.object(
                        fitted, "_exact_nonadd_normal_form", return_value=current
                    ),
                ):
                    selected = fitted._normalize_composite_addition(
                        current, proposal, rule
                    )
                self.assertIsNone(selected)
                self.assertIn(
                    (current.support, proposal.support),
                    fitted._conditional_parent_forbidden,
                )
                self.assertEqual(fitted.diagnostics.composite_add_rejections, 1)
            finally:
                fitted.close()

    def test_terminal_stabilizes_identity_before_complete_add_audit(self) -> None:
        """A changed active identity invalidates pricing, so it must move first."""

        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fitted = optimizer(
                data,
                3,
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
                terminal_add_audit="block_score",
            )
            try:
                current = fitted.fit(
                    Support.of((RuleIdentity((0,), 0, 1),)),
                    fitted.records[EMPTY_SUPPORT],
                )
                replacement = fitted.fit(
                    Support.of((RuleIdentity((0,), 0, -1),)),
                    fitted.records[EMPTY_SUPPORT],
                )
                replacement = replace(
                    replacement,
                    score=current.score + 1.0,
                )
                with (
                    mock.patch.object(
                        fitted,
                        "_best_exact_nonpositive_branch_drop",
                        return_value=None,
                    ),
                    mock.patch.object(
                        fitted, "_best_exact_nonrobust_rule_drop", return_value=None
                    ),
                    mock.patch.object(
                        fitted, "_best_exact_rule_objective_drop", return_value=None
                    ),
                    mock.patch.object(
                        fitted,
                        "_best_final_identity_change",
                        return_value=replacement,
                    ),
                    mock.patch.object(
                        fitted,
                        "_best_composite_fast_addition",
                        side_effect=AssertionError("Add audit ran before identity"),
                    ),
                ):
                    decision = fitted._best_terminal_cleanup_decision(
                        current,
                        audit_add=True,
                    )
                self.assertEqual(decision.record.support, replacement.support)
            finally:
                fitted.close()

    def test_atomic_bound_contains_every_signed_higher_order_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fitted = optimizer(data, 3, search_mode="safe_column_generation")
            try:
                empty = fitted.records[EMPTY_SUPPORT]
                required = fitted.fit(
                    Support.of((RuleIdentity((0,), 0, 1),)), empty
                )
                self.assertTrue(required.fit.converged, required.fit.message)
                optional = ((1,), (0, 1), (0, 1, 2))
                certificate = fitted.atomic_subtree_upper_bound(required, optional)
                exact_maximum = -math.inf
                exact_argmax = EMPTY_SUPPORT
                observed_orders: set[int] = set()
                for choices in itertools.product((0, -1, 1), repeat=len(optional)):
                    support = required.support
                    for antecedent, sign in zip(optional, choices, strict=True):
                        if sign:
                            support = support.add(RuleIdentity(antecedent, 0, sign))
                            observed_orders.add(len(antecedent))
                    record = fitted.fit(support, required)
                    if record.fit.converged and record.score > exact_maximum:
                        exact_maximum = record.score
                        exact_argmax = support
                    if record.fit.converged:
                        self.assertGreaterEqual(
                            certificate.score_upper_bound + 2.0e-6,
                            record.score,
                            msg=f"unsafe for descendant {support.rules}",
                        )
                self.assertEqual(observed_orders, {1, 2, 3})
                self.assertNotEqual(exact_argmax, EMPTY_SUPPORT)
                self.assertLessEqual(
                    certificate.score_upper_bound,
                    certificate.coarse_score_upper_bound + 1.0e-10,
                )
            finally:
                fitted.close()

    def test_atomic_bound_contains_multi_window_multi_rule_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 13)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0, 1),
                solver_tolerance=1.0e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                reliability_aware_search=False,
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
            )
            fitted = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                empty = fitted.records[Support(())]
                optional = ((0,), (1,), (0, 1))
                certificate = fitted.atomic_subtree_upper_bound(empty, optional)
                choices = tuple(
                    (None,)
                    + tuple(
                        (window, sign)
                        for window in ((0,) if len(antecedent) == 1 else (0, 1))
                        for sign in (-1, 1)
                    )
                    for antecedent in optional
                )
                for selected in itertools.product(*choices):
                    support = Support(())
                    for antecedent, identity in zip(optional, selected, strict=True):
                        if identity is not None:
                            support = support.add(
                                RuleIdentity(antecedent, identity[0], identity[1])
                            )
                    exact = fitted.fit(support, empty)
                    if exact.fit.converged:
                        self.assertGreaterEqual(
                            certificate.score_upper_bound + 2.0e-6,
                            exact.score,
                            msg=f"unsafe W/sign descendant: {support.rules}",
                        )
            finally:
                fitted.close()

    def test_memory_fallback_remains_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fitted = optimizer(data, 3)
            try:
                empty = fitted.records[EMPTY_SUPPORT]
                required = fitted.fit(
                    Support.of((RuleIdentity((0,), 0, 1),)), empty
                )
                relaxation = AtomicSignatureRelaxation(
                    fitted.context,
                    fitted.engine,
                    fitted.objective,
                    baseline_nll=fitted.baseline_nll,
                    window_dictionary=fitted.window_dictionary,
                    workspace_bytes=1,
                    global_nll_lower_bound=fitted.saturated_nll_lower_bound,
                )
                optional = ((1,), (0, 1), (0, 1, 2))
                certificate = relaxation.bound(
                    required.support, required.matrix, optional
                )
                self.assertGreater(certificate.saturated_components, 0)
                for choices in itertools.product((0, -1, 1), repeat=len(optional)):
                    support = required.support
                    for antecedent, sign in zip(optional, choices, strict=True):
                        if sign:
                            support = support.add(RuleIdentity(antecedent, 0, sign))
                    record = fitted.fit(support, required)
                    if record.fit.converged:
                        self.assertGreaterEqual(
                            certificate.score_upper_bound + 2.0e-6,
                            record.score,
                        )
            finally:
                fitted.close()

    def test_primitive_cover_coarse_map_contains_all_descendants(self) -> None:
        """The O(P) production pre-bound safely covers q=1,2,3 effects."""

        with tempfile.TemporaryDirectory() as directory:
            data = self._three_predicate_data(Path(directory) / "data")
            fitted = optimizer(data, 3)
            try:
                empty = fitted.records[EMPTY_SUPPORT]
                relaxation = AtomicSignatureRelaxation(
                    fitted.context,
                    fitted.engine,
                    fitted.objective,
                    baseline_nll=fitted.baseline_nll,
                    window_dictionary=fitted.window_dictionary,
                    # Force the decision-level primitive cover to fail open
                    # before the full optional incidence map is built.
                    workspace_bytes=1,
                    global_nll_lower_bound=fitted.saturated_nll_lower_bound,
                )
                compiled_cover = relaxation._primitive_cover((0, 1, 2))
                reference_cover = np.unique(
                    np.concatenate(
                        [
                            relaxation._footprint(("atomic", (predicate,)))
                            for predicate in (0, 1, 2)
                        ]
                    )
                )
                np.testing.assert_array_equal(compiled_cover, reference_cover)
                optional = ((0,), (1,), (0, 1), (0, 1, 2))
                certificate = relaxation.bound(
                    empty.support,
                    empty.matrix,
                    optional,
                    prune_threshold=-1.0e100,
                )
                self.assertEqual(certificate.refined_components, 0)
                for choices in itertools.product((0, -1, 1), repeat=len(optional)):
                    support = EMPTY_SUPPORT
                    for antecedent, sign in zip(optional, choices, strict=True):
                        if sign:
                            support = support.add(
                                RuleIdentity(antecedent, 0, sign)
                            )
                    record = fitted.fit(support, empty)
                    if record.fit.converged:
                        self.assertGreaterEqual(
                            certificate.score_upper_bound + 2.0e-6,
                            record.score,
                        )
            finally:
                fitted.close()

    def test_poisson_localized_bound_contains_exact_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = self._three_predicate_data(Path(directory) / "data")
            data = replace(original, likelihood="poisson", ticks_per_unit=60)
            fitted = optimizer(data, 2)
            try:
                support = Support.of((RuleIdentity((0, 1), 0, 1),))
                exact = fitted.fit(support, fitted.records[EMPTY_SUPPORT])
                self.assertTrue(exact.fit.converged, exact.fit.message)
                self.assertGreaterEqual(
                    fitted.localized_upper_score(support) + 2.0e-6,
                    exact.score,
                )
                affected = fitted._feature_rows(support)
                self.assertLess(fitted._row_saturated_nll(affected), 0.0)
            finally:
                fitted.close()

    def test_sidetrack_queue_is_deterministic_deduplicated_and_fail_open(self) -> None:
        queue = CertifiedSidetrackQueue(1.0)
        self.assertTrue(
            queue.push("a", upper_score=math.inf, lower_score=2.0, payload="first")
        )
        self.assertFalse(
            queue.push("a", upper_score=3.0, lower_score=2.0, payload="duplicate")
        )
        self.assertTrue(
            queue.push("b", upper_score=4.0, lower_score=1.5, payload="second")
        )
        self.assertFalse(
            queue.push("c", upper_score=1.0, lower_score=1.0, payload="pruned")
        )
        self.assertEqual(queue.pop().payload, "first")
        self.assertEqual(queue.pop().payload, "second")
        self.assertIsNone(queue.pop())
        self.assertEqual(queue.merged, 1)
        self.assertEqual(queue.pruned, 1)

    def test_atomic_frontier_prunes_a_complete_redundant_descendant_region(
        self,
    ) -> None:
        """A=B leaves no distinct child state, so the whole Add region closes."""

        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data",
                2,
                {(0, 0): 0.10, (1, 1): 0.70},
                per_cell=500,
            )
            fitted = optimizer(
                data,
                2,
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
            )
            try:
                result = fitted.search()
                self.assertGreater(
                    result.diagnostics.atomic_frontier_node_screens,
                    0,
                )
                self.assertGreater(
                    result.diagnostics.atomic_frontier_remaining_candidates_avoided,
                    0,
                )
                # The redundant B and AB descendants are certified away as a
                # region; the representative A mechanism is unchanged.
                self.assertEqual(
                    [
                        tuple(rule.antecedent for rule in record.support.rules)
                        for record in result.family
                    ],
                    [((0,),)],
                )
            finally:
                fitted.close()

    def test_atomic_frontier_keeps_signed_pair_and_triplet_regions(self) -> None:
        cases = (
            (
                2,
                {
                    (0, 0): 0.10,
                    (0, 1): 0.10,
                    (1, 0): 0.80,
                    (1, 1): 0.01,
                },
                (0, 1),
            ),
            (
                3,
                {
                    cell: (0.75 if sum(cell) % 2 == 1 else 0.05)
                    for cell in itertools.product((0, 1), repeat=3)
                },
                (0, 1, 2),
            ),
        )
        for predicates, probabilities, expected in cases:
            with self.subTest(order=predicates), tempfile.TemporaryDirectory() as directory:
                data = cell_dataset(
                    Path(directory) / "data",
                    predicates,
                    probabilities,
                    per_cell=1_000 if predicates == 2 else 180,
                )
                fitted = optimizer(
                    data,
                    predicates,
                    search_mode="atomic_rashomon_frontier",
                    adaptive_gradient_racing=True,
                )
                try:
                    result = fitted.search()
                    self.assertTrue(
                        any(
                            any(
                                rule.antecedent == expected
                                for rule in record.support.rules
                            )
                            for record in result.family
                        )
                    )
                    self.assertTrue(
                        all(record.fit.converged for record in result.terminals)
                    )
                    self.assertGreater(
                        result.diagnostics.atomic_frontier_regions,
                        0,
                    )
                finally:
                    fitted.close()

    def test_atomic_terminal_reopens_all_columns_with_the_exact_safe_oracle(
        self,
    ) -> None:
        """Route-only feasible screens cannot contaminate terminal stationarity."""

        with tempfile.TemporaryDirectory() as directory:
            data = cell_dataset(
                Path(directory) / "data",
                2,
                {
                    (0, 0): 0.05,
                    (0, 1): 0.05,
                    (1, 0): 0.75,
                    (1, 1): 0.80,
                },
                per_cell=500,
            )
            fitted = optimizer(
                data,
                2,
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
            )
            try:
                empty = fitted.records[EMPTY_SUPPORT]
                fitted._force_exact_candidate_validation = True
                # This routine is a feasible lower-endpoint route screen.  An
                # exact atomic terminal must never call it or inherit its
                # rejection table.
                with mock.patch.object(
                    fitted,
                    "_batched_parent_frozen_survivors",
                    side_effect=AssertionError("route screen reached terminal"),
                ):
                    added = fitted._best_conditional_addition(
                        empty,
                        antecedents=set(fitted.skeletons),
                        frozen_identities=False,
                    )
                self.assertIsNotNone(added)
                assert added is not None
                self.assertTrue(added.fit.converged, added.fit.message)
                self.assertGreater(added.score, empty.score)
                self.assertGreater(fitted.diagnostics.safe_column_candidates, 0)
            finally:
                fitted._force_exact_candidate_validation = False
                fitted.close()


if __name__ == "__main__":
    unittest.main()
