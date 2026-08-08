from __future__ import annotations

import math
import tempfile
import unittest
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from crbstpp.config import RunConfig
from crbstpp.certification import _entity_losses_frozen, certify_family
from crbstpp.data import Dataset, write_dataset
from crbstpp.dependency import model_dependency_complexity
from crbstpp.dual import dual_certificate
from crbstpp.ensemble import (
    _components_from_profiles,
    _fit_frozen_model_with_retry,
    _fit_sparse_profiles,
    _fixed_model_mixture,
    _fit_simplex_statistics,
    _mixture_nll_gradient,
    _mixture_statistics_nll_gradient,
    _mixture_sufficient_statistics,
    _select_intensity_family,
    _SparseComponents,
    _MixtureSufficientStatistics,
    _SparseProfile,
    _sparse_profile,
)
from crbstpp.likelihood import cloglog_event_terms, loss_rows, loss_value_rows
from crbstpp.native import (
    accumulate_cluster_scores,
    cuda_available,
    fill_pricing_values,
    implicit_moments_batch,
    implicit_objective_batch,
    implicit_poisson_objective_batch,
)
from crbstpp.pipeline import _discovery_context, _support_from_payload, _support_payload
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import (
    ClosureTerm,
    EMPTY_SUPPORT,
    RuleIdentity,
    Support,
    hierarchy_branch_drop,
    hierarchy_branch_null_closure,
    hierarchy_closure,
    skeletons,
)
from crbstpp.objective import (
    ObjectiveSpec,
    SupportRecord,
    block_mdl_delta,
    family_block_mdl_delta,
    freeze_support_record,
    relaxed_family_block_mdl_delta,
    support_score,
)
from crbstpp.search import (
    _HierarchyRuleGeometry,
    _MoveDecision,
    SearchDiagnostics,
    SupportOptimizer,
    _gradient_bundle_lower_bound,
    _StoredRecord,
    _mdl_nondominated_records,
    _mechanism_mdl_representatives,
    _nested_mdl_representatives,
    _positive_deletion_minimal_records,
    _TerminalPointerCache,
    support_from_key,
    support_key,
)
from crbstpp.solver import (
    FitResult,
    ProjectedDesignEvaluator,
    fit_model_matrix,
    fit_model_matrix_continued,
    fit_projected_model_matrix,
    fit_sparse_grid_model,
    one_step_model_matrix,
)


class FamilyBlockMdlObjectiveTests(unittest.TestCase):
    def test_block_delta_uses_likelihood_and_code_change(self) -> None:
        self.assertEqual(
            block_mdl_delta(
                likelihood_gain=8.0,
                parent_penalty=10.0,
                child_penalty=14.0,
            ),
            12.0,
        )

    def test_separate_family_is_the_add_incumbent(self) -> None:
        self.assertEqual(
            family_block_mdl_delta(
                parent_score=20.0,
                child_score=27.0,
                separate_score=25.0,
            ),
            2.0,
        )

    def test_branch_and_quadratic_use_the_same_family_delta(self) -> None:
        self.assertEqual(
            family_block_mdl_delta(
                parent_score=20.0,
                child_score=27.0,
                separate_score=25.0,
                branch_delta=-1.0,
            ),
            -1.0,
        )
        self.assertEqual(
            relaxed_family_block_mdl_delta(
                parent_score=20.0, block_delta=7.0, separate_score=25.0
            ),
            2.0,
        )


@unittest.skipUnless(cuda_available(), "CUDA pricing extension is unavailable")
class SparseCompletionCudaTests(unittest.TestCase):
    def test_sparse_entity_span_lookup_matches_dense_offsets_exactly(self) -> None:
        starts = np.asarray([0, 0], dtype=np.int64)
        ends = np.asarray([3, 3], dtype=np.int64)
        grid_offsets = np.asarray([0, 4, 8], dtype=np.int64)
        times = np.asarray([1, 2], dtype=np.int64)
        spans = np.asarray([0, 1], dtype=np.int64)
        dense_offsets = np.asarray([[0, 1, 2]], dtype=np.int64)
        sparse_offsets = np.asarray([[0, 2]], dtype=np.int64)
        packed_entity_spans = np.asarray([(0 << 32) | 0, (1 << 32) | 1], dtype=np.int64)
        basis = np.asarray([[1.0, 0.5]], dtype=np.float64)
        predicates = np.asarray([[[0, -1, -1]]], dtype=np.int32)
        orders = np.asarray([[1]], dtype=np.int32)
        windows = np.asarray([[1]], dtype=np.int64)
        counts = np.asarray([1], dtype=np.int32)
        active_offsets = np.asarray([0, 2], dtype=np.int64)
        active_entities = np.asarray([0, 1], dtype=np.int32)
        first = np.linspace(-0.3, 0.4, 8)
        second = np.linspace(0.5, 1.2, 8)
        groups = np.zeros(8, dtype=np.int32)
        current_x = np.ones((1, 1), dtype=np.float64)
        common = dict(
            starts=starts,
            ends=ends,
            grid_offsets=grid_offsets,
            basis=basis,
            block_predicates=predicates,
            block_orders=orders,
            block_windows=windows,
            block_counts=counts,
            candidate_entity_offsets=active_offsets,
            candidate_entities=active_entities,
            first=first,
            second=second,
            group_by_row=groups,
            current_x=current_x,
            derivative_token=902,
            device="cuda:0",
            current_columns=np.asarray([0], dtype=np.int32),
        )
        dense = implicit_moments_batch(
            dense_offsets,
            times,
            spans,
            source_token=900,
            completion_mode=1,
            **common,
        )
        sparse = implicit_moments_batch(
            sparse_offsets,
            times,
            packed_entity_spans,
            source_token=901,
            completion_mode=2,
            **common,
        )
        self.assertIsNotNone(dense)
        self.assertIsNotNone(sparse)
        assert dense is not None and sparse is not None
        for expected, observed in zip(dense, sparse, strict=True):
            np.testing.assert_array_equal(observed, expected)

    def test_preconvolved_gradient_matches_full_sparse_moments(self) -> None:
        """The fast terminal KKT oracle preserves the exact raw score."""
        starts = np.asarray([0, 0], dtype=np.int64)
        ends = np.asarray([3, 3], dtype=np.int64)
        grid_offsets = np.asarray([0, 4, 8], dtype=np.int64)
        times = np.asarray([1, 2], dtype=np.int64)
        packed_entity_spans = np.asarray([(0 << 32) | 0, (1 << 32) | 1], dtype=np.int64)
        sparse_offsets = np.asarray([[0, 2]], dtype=np.int64)
        basis = np.asarray([[1.0, 0.5], [0.25, 0.75]], dtype=np.float64)
        predicates = np.asarray([[[0, -1, -1]], [[0, -1, -1]]], dtype=np.int32)
        orders = np.ones((2, 1), dtype=np.int32)
        windows = np.asarray([[0], [1]], dtype=np.int64)
        counts = np.ones(2, dtype=np.int32)
        active_offsets = np.asarray([0, 2, 4], dtype=np.int64)
        active_entities = np.asarray([0, 1, 0, 1], dtype=np.int32)
        group_by_row = np.zeros(8, dtype=np.int32)
        current_x = np.ones((1, 1), dtype=np.float64)
        derivative = np.asarray([0.6], dtype=np.float64)
        events = np.asarray([0, 0, 1, 0, 0, 0, 0, 1], dtype=np.uint8)
        common = dict(
            source_offsets=sparse_offsets,
            source_times=times,
            source_spans=packed_entity_spans,
            starts=starts,
            ends=ends,
            grid_offsets=grid_offsets,
            basis=basis,
            block_predicates=predicates,
            block_orders=orders,
            block_windows=windows,
            block_counts=counts,
            first=derivative,
            second=derivative,
            group_by_row=group_by_row,
            current_x=current_x,
            compact_poisson_events=events,
            completion_mode=2,
            device="cuda:0",
        )
        full = implicit_moments_batch(
            candidate_entity_offsets=active_offsets,
            candidate_entities=active_entities,
            source_token=910,
            derivative_token=911,
            current_columns=np.asarray([0], dtype=np.int32),
            **common,
        )
        direct = implicit_moments_batch(
            candidate_entity_offsets=np.zeros(3, dtype=np.int64),
            candidate_entities=np.zeros(0, dtype=np.int32),
            source_token=912,
            derivative_token=913,
            gradient_only=True,
            **common,
        )
        self.assertIsNotNone(full)
        self.assertIsNotNone(direct)
        assert full is not None and direct is not None
        np.testing.assert_allclose(direct[0], full[0], rtol=2.0e-15, atol=2.0e-15)
        self.assertEqual(direct[1].shape, (2, 0, 0))
        self.assertEqual(direct[2].shape, (2, 0, 2))


class GradientBundleBoundTests(unittest.TestCase):
    def test_convex_cone_bundle_is_a_global_lower_bound(self) -> None:
        # f(x, y) = ((x - 1)^2 + (y + 1)^2) / 2, with x free and y >= 0.
        # The exact constrained optimum is (1, 0), f*=1/2.  The two cuts
        # bracket the free gradient and have a nonnegative aggregate cone
        # gradient, so they form a feasible dual bundle certificate.
        points = [np.asarray([0.0, 0.0]), np.asarray([2.0, 0.0])]
        values = [
            0.5 * float((point[0] - 1.0) ** 2 + (point[1] + 1.0) ** 2)
            for point in points
        ]
        gradients = [np.asarray([point[0] - 1.0, point[1] + 1.0]) for point in points]
        lower = _gradient_bundle_lower_bound(
            values,
            points,
            gradients,
            free_dimension=1,
        )
        self.assertTrue(np.isfinite(lower))
        self.assertLessEqual(lower, 0.5 + 1.0e-12)
        self.assertGreater(lower, -1.0e-10)

    def test_infeasible_gradient_bundle_fails_open(self) -> None:
        points = [np.asarray([0.0, 0.0]), np.asarray([0.5, 0.0])]
        values = [1.0, 0.625]
        gradients = [np.asarray([-1.0, -1.0]), np.asarray([-0.5, -0.5])]
        lower = _gradient_bundle_lower_bound(
            values,
            points,
            gradients,
            free_dimension=1,
        )
        self.assertEqual(lower, -np.inf)


class DependencyAwareSelectionTests(unittest.TestCase):
    def test_cluster_accumulator_accepts_unobserved_trailing_clusters(self) -> None:
        design = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        groups = np.asarray([0, 1, 0], dtype=np.int32)
        clusters = np.asarray([0, 1, 1], dtype=np.int32)
        multipliers = np.asarray([1.0, 2.0, -1.0], dtype=np.float64)
        output = np.zeros((4, 2), dtype=np.float64)
        accumulate_cluster_scores(design, groups, clusters, multipliers, output)
        np.testing.assert_allclose(
            output,
            np.asarray([[1.0, 2.0], [5.0, 6.0], [0.0, 0.0], [0.0, 0.0]]),
        )

    def test_two_way_code_preserves_tpp_mle_and_never_cheapens_bic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            base_config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                romano_wolf_resamples=1_000,
                solver_tolerance=1.0e-8,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                pricing_workers=1,
                exact_workers=1,
            )
            fit_codes, _, _ = data.split(
                base_config.split_fractions, base_config.split_seed
            )
            support = Support.of((RuleIdentity((0,), 0, 1),))
            ordinary = SupportOptimizer(Context.make(data, fit_codes), base_config)
            dependency = SupportOptimizer(
                Context.make(data, fit_codes),
                replace(base_config, dependency_aware_mdl=True),
            )
            try:
                ordinary_record = ordinary.fit(support, ordinary.records[EMPTY_SUPPORT])
                dependency_record = dependency.fit(
                    support, dependency.records[EMPTY_SUPPORT]
                )
                np.testing.assert_allclose(
                    dependency_record.fit.coefficients,
                    ordinary_record.fit.coefficients,
                    rtol=1.0e-10,
                    atol=1.0e-10,
                )
                self.assertAlmostEqual(
                    dependency_record.fit.nll, ordinary_record.fit.nll, places=9
                )
                self.assertGreaterEqual(
                    dependency_record.penalty,
                    dependency.objective.structural_penalty(support) - 1.0e-12,
                )
                self.assertIsNotNone(dependency_record.dependency_effective_dimension)
                self.assertLess(
                    dependency_record.dependency_diagnostics["score_residual"],
                    1.0e-8,
                )
                unaggregated = dependency.engine.model_matrix(
                    dependency.context,
                    support,
                    _allow_extension=False,
                    _aggregate_rows=False,
                )
                aggregated_complexity = model_dependency_complexity(
                    dependency.engine,
                    dependency.context,
                    dependency_record.matrix,
                    dependency_record.fit,
                    dependence_horizon_ticks=3,
                )
                unaggregated_complexity = model_dependency_complexity(
                    dependency.engine,
                    dependency.context,
                    unaggregated,
                    dependency_record.fit,
                    dependence_horizon_ticks=3,
                )
                self.assertAlmostEqual(
                    unaggregated_complexity.effective_dimension,
                    aggregated_complexity.effective_dimension,
                    places=9,
                )
            finally:
                ordinary.close()
                dependency.close()

    def test_route_dependency_code_changes_only_provisional_mdl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                dependency_aware_mdl=True,
                romano_wolf_resamples=1_000,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                pricing_workers=1,
                exact_workers=1,
            )
            fit_codes, _, _ = data.split(config.split_fractions, config.split_seed)
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                support = Support.of((RuleIdentity((0,), 0, 1),))
                exact = optimizer.fit(support, optimizer.records[EMPTY_SUPPORT])
                structural_penalty = optimizer.objective.structural_penalty(support)
                provisional = replace(
                    exact,
                    fit=replace(exact.fit, converged=False, message="route point"),
                    penalty=structural_penalty,
                    score=support_score(
                        baseline_nll=optimizer.baseline_nll,
                        fit_nll=exact.fit.nll,
                        penalty=structural_penalty,
                    ),
                    dependency_effective_dimension=None,
                    dependency_diagnostics=None,
                )
                rescored = optimizer._dependency_route_rescore(provisional)
                np.testing.assert_array_equal(
                    rescored.fit.coefficients, provisional.fit.coefficients
                )
                self.assertEqual(rescored.fit.nll, provisional.fit.nll)
                self.assertFalse(rescored.fit.converged)
                self.assertGreaterEqual(rescored.penalty, structural_penalty)
                self.assertIsNotNone(rescored.dependency_effective_dimension)
                self.assertTrue(rescored.dependency_diagnostics["route_only"])
            finally:
                optimizer.close()

    def test_nested_route_owner_is_exactified_before_next_dag_node(self) -> None:
        """A masking-changing route point cannot own noncanonical geometry."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0, 1),
                romano_wolf_resamples=1_000,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                pricing_workers=1,
                exact_workers=1,
            )
            fit_codes, _, _ = data.split(config.split_fractions, config.split_seed)
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                higher = RuleIdentity((0, 1), 1, 1)
                lower = RuleIdentity((0,), 0, 1)
                parent = optimizer.fit(
                    Support.of((higher,)), optimizer.records[EMPTY_SUPPORT]
                )
                child = optimizer.fit(parent.support.add(lower), parent)
                provisional = replace(
                    child,
                    fit=replace(child.fit, converged=False, message="route point"),
                )
                with (
                    patch.object(
                        optimizer,
                        "_exactify_path_state",
                        return_value=child,
                    ) as exactify,
                    patch.object(
                        optimizer,
                        "_attach_rule_score",
                        return_value=child,
                    ) as attach,
                ):
                    canonical = optimizer._canonicalize_route_geometry(
                        parent, provisional, reason="dag"
                    )
                self.assertIs(canonical, child)
                exactify.assert_called_once_with(provisional, reason="dag")
                attach.assert_called_once_with(child)

                ordinary_parent = optimizer.records[EMPTY_SUPPORT]
                ordinary_child = optimizer.fit(Support.of((lower,)), ordinary_parent)
                with patch.object(
                    optimizer, "_exactify_path_state"
                ) as exactify_ordinary:
                    unchanged = optimizer._canonicalize_route_geometry(
                        ordinary_parent, ordinary_child, reason="dag"
                    )
                self.assertIs(unchanged, ordinary_child)
                exactify_ordinary.assert_not_called()
            finally:
                optimizer.close()

    def test_forced_cleanup_can_terminate_at_empty_without_repeating(self) -> None:
        """A failed singleton guard must be allowed to reach the empty node."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                romano_wolf_resamples=1_000,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            fit_codes, _, _ = data.split(config.split_fractions, config.split_seed)
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                empty = optimizer.records[EMPTY_SUPPORT]
                singleton = optimizer.fit(
                    Support.of((RuleIdentity((0,), 0, 1),)), empty
                )
                self.assertTrue(
                    optimizer._route_score_move_allowed(
                        singleton, empty, forced_cleanup_drop=True
                    )
                )
                self.assertFalse(
                    optimizer._route_score_move_allowed(
                        singleton, empty, forced_cleanup_drop=False
                    )
                )
            finally:
                optimizer.close()

    def test_terminal_add_applies_bound_before_sparse_exact_fail_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                romano_wolf_resamples=1_000,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            fit_codes, _, _ = data.split(config.split_fractions, config.split_seed)
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                optimizer._terminal_add_audit_active = True
                self.assertFalse(
                    optimizer._use_sparse_add_prefetch(
                        stop_at_first_improving=True,
                        has_finite_endpoint=True,
                    )
                )
                optimizer._terminal_add_audit_active = False
                self.assertTrue(
                    optimizer._use_sparse_add_prefetch(
                        stop_at_first_improving=True,
                        has_finite_endpoint=True,
                    )
                )
                self.assertFalse(
                    optimizer._use_sparse_add_prefetch(
                        stop_at_first_improving=False,
                        has_finite_endpoint=True,
                    )
                )
            finally:
                optimizer.close()

    def test_precert_compaction_removes_nested_mdl_loser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                precert_family_compaction=True,
                romano_wolf_resamples=1_000,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                pricing_workers=1,
                exact_workers=1,
            )
            fit_codes, _, _ = data.split(config.split_fractions, config.split_seed)
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                first = Support.of((RuleIdentity((0,), 0, 1),))
                larger = Support.of(
                    (RuleIdentity((0,), 0, 1), RuleIdentity((1,), 0, 1))
                )
                records = []
                for support, score in ((first, 10.0), (larger, 9.0)):
                    metadata = optimizer.engine.model_metadata(support)
                    fit = FitResult(
                        np.zeros(metadata.dimension),
                        1.0,
                        True,
                        1,
                        0.0,
                        metadata.dimension,
                        False,
                        "test",
                    )
                    records.append(
                        SupportRecord(
                            support,
                            metadata,
                            fit,
                            1.0,
                            score,
                            rule_score=score,
                            closure_null_nll=optimizer.baseline_nll,
                            rule_score_upper=score,
                        )
                    )
                compact = optimizer.compact_before_certification(tuple(records))
                self.assertEqual(tuple(item.support for item in compact), (first,))
                self.assertEqual(optimizer.diagnostics.precert_family_input, 2)
                self.assertEqual(optimizer.diagnostics.precert_family_output, 1)
            finally:
                optimizer.close()


class EnsembleRefitContractTests(unittest.TestCase):
    @staticmethod
    def _fit(converged: bool, message: str) -> FitResult:
        return FitResult(
            np.zeros(1),
            1.0,
            converged,
            1,
            0.0,
            1,
            False,
            message,
        )

    def test_warm_rank_failure_retries_cold_without_changing_problem(self) -> None:
        failed = self._fit(False, "rank-deficient fixed-support information")
        converged = self._fit(True, "converged")
        with patch(
            "crbstpp.ensemble.fit_model_matrix_continued",
            side_effect=(failed, converged),
        ) as solver:
            result = _fit_frozen_model_with_retry(
                SimpleNamespace(),
                likelihood="poisson",
                tolerance=1.0e-8,
                max_iter=20,
                warm_start=np.ones(1),
                device="cpu",
            )
        self.assertTrue(result.converged)
        self.assertEqual(solver.call_count, 2)
        self.assertIsNotNone(solver.call_args_list[0].kwargs["warm_start"])
        self.assertIsNone(solver.call_args_list[1].kwargs["warm_start"])

    def test_all_exact_refit_failures_are_reported_not_silently_dropped(self) -> None:
        failed = self._fit(False, "rank-deficient fixed-support information")
        with patch(
            "crbstpp.ensemble.fit_model_matrix_continued",
            side_effect=(failed, failed, failed),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "failed exact refit after fail-open retries"
            ):
                _fit_frozen_model_with_retry(
                    SimpleNamespace(),
                    likelihood="poisson",
                    tolerance=1.0e-8,
                    max_iter=20,
                    warm_start=np.ones(1),
                    device="cuda:0",
                )

    def test_parallel_worker_can_defer_cpu_fallback_until_wave_joins(self) -> None:
        failed = self._fit(False, "numerical stagnation")
        with patch(
            "crbstpp.ensemble.fit_model_matrix_continued",
            side_effect=(failed, failed),
        ) as solver:
            with self.assertRaisesRegex(
                RuntimeError, "failed exact refit after fail-open retries"
            ):
                _fit_frozen_model_with_retry(
                    SimpleNamespace(),
                    likelihood="poisson",
                    tolerance=1.0e-8,
                    max_iter=20,
                    warm_start=np.ones(1),
                    device="cuda:0",
                    allow_cpu_fallback=False,
                )
        self.assertEqual(solver.call_count, 2)
        self.assertTrue(
            all(call.kwargs["device"] == "cuda:0" for call in solver.call_args_list)
        )


def synthetic_dataset(
    root: Path,
    n_entities: int = 90,
    *,
    explicit_partition: bool = False,
    likelihood: str = "first_event_cloglog",
) -> Dataset:
    entities = pd.DataFrame(
        {
            "entity_id": [f"e{index:04d}" for index in range(n_entities)],
            "start_time": np.zeros(n_entities, dtype=np.int64),
            "end_time": np.full(n_entities, 8, dtype=np.int64),
            "baseline_origin": np.zeros(n_entities, dtype=np.int64),
            "split_group": np.zeros(n_entities, dtype=np.int64),
        }
    )
    if explicit_partition:
        entities["partition"] = np.arange(n_entities, dtype=np.int64) * 3 // n_entities
    events = []
    targets = []
    for entity in range(n_entities):
        if entity % 3 != 0:
            events.append((entity, 1, 0))
        if entity % 5 != 0:
            events.append((entity, 2, 1))
        # Strong A excitation, with some negative controls retained.
        if entity % 3 != 0 and entity % 7 != 0:
            targets.append((entity, 3, 1))
        elif entity % 17 == 0:
            targets.append((entity, 6, 1))
    event_frame = pd.DataFrame(
        events, columns=["entity_code", "time", "predicate_code"]
    )
    target_frame = pd.DataFrame(
        targets, columns=["entity_code", "time", "multiplicity"]
    )
    write_dataset(
        root,
        entities=entities,
        events=event_frame,
        targets=target_frame,
        predicate_names=("A", "B"),
        likelihood=likelihood,
        time_unit="month",
        adverse_event_name="synthetic adverse event",
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded_from_reported_dictionary": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
            "primitive_event_provenance": True,
            "independent_certification_units": True,
        },
        provenance={"generator": "test"},
    )
    return Dataset.load(root)


def controlled_synthetic_dataset(root: Path, n_entities: int = 600) -> Dataset:
    """Dataset whose fitted baseline contains a nonzero dynamic control.

    Candidate response rows deliberately include control-inactive observations
    at ages beyond one impact lag.  This detects accidental treatment of the
    control coefficient as an age-bin intercept in sparse objectives.
    """
    rng = np.random.default_rng(947)
    entities = pd.DataFrame(
        {
            "entity_id": [f"c{index:04d}" for index in range(n_entities)],
            "start_time": np.zeros(n_entities, dtype=np.int64),
            "end_time": np.full(n_entities, 6, dtype=np.int64),
            "baseline_origin": np.zeros(n_entities, dtype=np.int64),
            "split_group": np.zeros(n_entities, dtype=np.int64),
        }
    )
    events: list[tuple[int, int, int]] = []
    targets: list[tuple[int, int, int]] = []
    for entity in range(n_entities):
        a = rng.random() < 0.58
        b = rng.random() < (0.72 if a else 0.34)
        control = rng.random() < (0.28 if a else 0.10)
        if a:
            events.append((entity, 1, 0))
        if b:
            events.append((entity, 2, 1))
        if control:
            events.append((entity, 1, 2))
        probability = 0.015 + 0.10 * a + 0.07 * b + 0.42 * control
        if rng.random() < probability:
            targets.append((entity, 3, 1))
    event_frame = pd.DataFrame(
        events, columns=["entity_code", "time", "predicate_code"]
    ).sort_values(["entity_code", "time", "predicate_code"], ignore_index=True)
    write_dataset(
        root,
        entities=entities,
        events=event_frame,
        targets=pd.DataFrame(targets, columns=["entity_code", "time", "multiplicity"]),
        predicate_names=("A", "B", "prior_state"),
        predicate_roles=("reported", "reported", "baseline_control"),
        likelihood="first_event_cloglog",
        time_unit="month",
        adverse_event_name="synthetic adverse event",
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded_from_reported_dictionary": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
            "primitive_event_provenance": True,
            "independent_certification_units": True,
        },
        provenance={"generator": "controlled-test"},
    )
    return Dataset.load(root)


class DataRuleTests(unittest.TestCase):
    def test_scalar_kernel_is_exact_submodel_with_smaller_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 45)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int64))
            engine = ResponseEngine(data, lag=3, knot_count=4, cache_bytes=8 * 1024**2)
            full_rule = RuleIdentity((0,), 0, 1)
            scalar_rule = full_rule.with_kernel_rank(1)
            full = engine.model_matrix(context, Support.of((full_rule,)))
            scalar = engine.model_matrix(context, Support.of((scalar_rule,)))
            self.assertEqual(full.rule_slices[0].stop - full.rule_slices[0].start, 4)
            self.assertEqual(
                scalar.rule_slices[0].stop - scalar.rule_slices[0].start, 1
            )
            scalar_beta = np.zeros(scalar.dimension, dtype=np.float64)
            scalar_beta[: scalar.baseline_dimension] = -1.5
            scalar_beta[scalar.rule_slices[0]] = 2.0
            full_beta = np.zeros(full.dimension, dtype=np.float64)
            full_beta[: full.baseline_dimension] = -1.5
            full_beta[full.rule_slices[0]] = 2.0 / 4.0
            np.testing.assert_allclose(
                engine.linear_predictor(context, scalar, scalar_beta),
                engine.linear_predictor(context, full, full_beta),
                rtol=0.0,
                atol=1.0e-12,
            )

            objective = ObjectiveSpec(
                n_entities=data.n_entities,
                skeleton_count=2,
                knot_count=4,
                window_count_by_order=(1, 1, 1),
                kernel_family_count=2,
            )
            self.assertAlmostEqual(
                objective.structural_penalty(Support.of((full_rule,)))
                - objective.structural_penalty(Support.of((scalar_rule,))),
                3.0 * np.log(data.n_entities),
            )

    def test_support_key_preserves_kernel_family_and_reads_legacy_key(self) -> None:
        scalar = Support.of((RuleIdentity((0, 1), 3, -1, 1),))
        self.assertEqual(support_from_key(support_key(scalar)), scalar)
        self.assertEqual(
            support_from_key("0,1|W3|inh"),
            Support.of((RuleIdentity((0, 1), 3, -1),)),
        )

    def test_adaptive_kernel_audit_keeps_or_improves_exact_mdl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=4,
                adaptive_kernel_mdl=True,
                formation_windows=(0,),
                early_warning_horizon=3,
                exact_workers=1,
                pricing_workers=1,
                pricing_devices=(),
                cache_bytes=16 * 1024**2,
            )
            optimizer = SupportOptimizer(
                Context.make(data, np.arange(data.n_entities, dtype=np.int64)),
                config,
            )
            empty = optimizer.fit(EMPTY_SUPPORT)
            full = optimizer._attach_rule_score(
                optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)), empty)
            )
            selected = optimizer._adaptive_kernel_mdl_family((full,))
            self.assertEqual(len(selected), 1)
            self.assertGreaterEqual(
                selected[0].score + config.search_tolerance, full.score
            )
            self.assertIn(
                selected[0].support.rules[0].kernel_dimension(config.knot_count),
                {1, config.knot_count},
            )
            self.assertEqual(
                selected[0].matrix.dimension,
                len(selected[0].fit.coefficients),
            )

    def test_scalar_kernel_identity_survives_frozen_certification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 75)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=4,
                adaptive_kernel_mdl=True,
                formation_windows=(0,),
                early_warning_horizon=3,
                romano_wolf_resamples=1_000,
                exact_workers=1,
                pricing_workers=1,
                pricing_devices=(),
                cache_bytes=16 * 1024**2,
            )
            fit_codes = np.arange(50, dtype=np.int64)
            cert_codes = np.arange(50, 75, dtype=np.int64)
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.fit(EMPTY_SUPPORT)
            scalar_support = Support.of((RuleIdentity((0,), 0, 1, 1),))
            scalar = optimizer._attach_rule_score(optimizer.fit(scalar_support, empty))
            frozen = freeze_support_record(scalar)
            result = certify_family(
                optimizer,
                Context.make(data, cert_codes),
                (frozen,),
                config,
            )
            self.assertEqual(result.models[0].record.support, scalar_support)
            self.assertEqual(
                result.models[0].certificate.multiplicity_method,
                "romano_wolf_stepdown_max_t",
            )

    def test_route_family_contract_uses_proven_joint_lower_bound(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer.config = SimpleNamespace(search_tolerance=1.0e-8)
        parent_rule = RuleIdentity((0,), 0, 1)
        added_rule = RuleIdentity((1,), 0, 1)
        parent = SimpleNamespace(support=Support.of((parent_rule,)))
        child = SimpleNamespace(
            support=Support.of((parent_rule, added_rule)),
            score=8.0,
            discovery_upper_score=9.0,
            fit=SimpleNamespace(converged=False),
        )
        optimizer._positive_atom_by_antecedent = {added_rule.antecedent: object()}
        optimizer._support_contract_add_decisions = {}
        optimizer._separate_family_scores = lambda *_: {added_rule.antecedent: 7.0}
        optimizer._exactify_path_state = lambda *_args, **_kwargs: self.fail(
            "a certified family lower bound must not be exact-refit"
        )
        self.assertIs(
            optimizer._family_route_add_contract(parent, child, added_rule), child
        )

    def test_separate_family_threshold_covers_every_requested_skeleton(self) -> None:
        """An unresolved standalone atom must still use the parent alternative."""

        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer._positive_atom_by_antecedent = {}
        optimizer._separate_family_score_cache = {}
        parent_rule = RuleIdentity((0,), 0, 1)
        parent = SimpleNamespace(
            support=Support.of((parent_rule,)),
            score=13.5,
            fit=SimpleNamespace(converged=True),
        )
        singleton = RuleIdentity((1,), 0, 1).pattern_key
        pair = RuleIdentity((1, 2), 1, -1).pattern_key
        thresholds = optimizer._separate_family_scores(parent, (singleton, pair))
        self.assertEqual(set(thresholds), {singleton, pair})
        self.assertEqual(thresholds[singleton], parent.score)
        self.assertEqual(thresholds[pair], parent.score)

    def test_reliability_contract_has_no_sign_or_order_quota(self) -> None:
        fields = set(RunConfig.__dataclass_fields__)
        self.assertFalse(
            fields.intersection(
                {
                    "minimum_excitation",
                    "minimum_inhibition",
                    "minimum_pair",
                    "minimum_triplet",
                    "order_quota",
                    "sign_quota",
                }
            )
        )

    def test_exact_branch_decision_is_reused_across_finalization_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                pricing_devices=(),
                solver_tolerance=1.0e-7,
                solver_max_iter=120,
            )
            context = Context.make(data, np.arange(60, dtype=np.int32))
            optimizer = SupportOptimizer(context, config)
            empty = optimizer.records[EMPTY_SUPPORT]
            support = Support.of(
                (
                    RuleIdentity((0,), 0, 1),
                    RuleIdentity((1,), 0, 1),
                )
            )
            record = optimizer._attach_rule_score(optimizer.fit(support, empty))
            first = optimizer._best_exact_nonpositive_branch_drop(
                record, protected_antecedents=frozenset()
            )
            second = optimizer._best_exact_nonpositive_branch_drop(
                record, protected_antecedents=frozenset()
            )
            self.assertEqual(
                None if first is None else first.support,
                None if second is None else second.support,
            )
            self.assertEqual(
                optimizer.diagnostics.exact_branch_decision_cache_misses, 1
            )
            self.assertEqual(optimizer.diagnostics.exact_branch_decision_cache_hits, 1)
            optimizer.close()

    def test_global_family_frontier_rejects_add_cleanup_cycle(self) -> None:
        """An invalid Add->minimality->parent edge is skipped, not replayed."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                search_mode="atomic_rashomon_frontier",
                pricing_devices=(),
                solver_tolerance=1.0e-7,
                solver_max_iter=120,
            )
            context = Context.make(data, np.arange(60, dtype=np.int32))
            optimizer = SupportOptimizer(context, config)
            empty = optimizer.records[EMPTY_SUPPORT]
            parent = optimizer._attach_rule_score(
                optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)), empty)
            )
            child = optimizer._attach_rule_score(
                optimizer.fit(parent.support.add(RuleIdentity((1,), 0, 1)), parent)
            )
            # The mocked exact Add is improving before minimality, but its
            # exact child cleanup deletes the new branch and returns parent.
            child = child.__class__(
                child.support,
                child.matrix,
                child.fit,
                child.penalty,
                parent.score + 1.0,
                child.rule_score,
                child.closure_null_nll,
                child.rule_score_upper,
            )

            def cleanup(current, **_):
                if current.support == child.support:
                    return _MoveDecision(parent, False)
                return _MoveDecision(None, True)

            def add(current, **_):
                edge = (current.support, child.support)
                return (
                    None if edge in optimizer._conditional_parent_forbidden else child
                )

            with (
                patch.object(
                    optimizer, "_best_terminal_cleanup_decision", side_effect=cleanup
                ),
                patch.object(optimizer, "_first_conditional_addition", side_effect=add),
            ):
                terminals, _, targets, _ = optimizer._global_exact_family_frontier(
                    (parent,)
                )
            self.assertEqual(
                tuple(item.support for item in terminals), (parent.support,)
            )
            self.assertEqual(targets[parent.support], parent.support)
            self.assertEqual(
                optimizer.diagnostics.global_family_frontier_composite_rejections,
                1,
            )
            optimizer.close()

    def test_global_family_frontier_follows_one_certified_edge_per_state(self) -> None:
        """Root diversity must not enumerate every identity at one state."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                search_mode="atomic_rashomon_frontier",
                terminal_add_audit="exact",
                rashomon_branching=True,
                pricing_devices=(),
                solver_tolerance=1.0e-7,
                solver_max_iter=120,
            )
            context = Context.make(data, np.arange(60, dtype=np.int32))
            optimizer = SupportOptimizer(context, config)
            empty = optimizer.records[EMPTY_SUPPORT]
            parent = optimizer._attach_rule_score(
                optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)), empty)
            )
            raw_children = tuple(
                optimizer._attach_rule_score(
                    optimizer.fit(parent.support.add(rule), parent)
                )
                for rule in (
                    RuleIdentity((1,), 0, 1),
                    RuleIdentity((0, 1), 0, 1),
                )
            )
            parent = parent.__class__(
                parent.support,
                parent.matrix,
                parent.fit,
                parent.penalty,
                1.0,
                1.0,
                parent.closure_null_nll,
                1.0,
            )
            children = tuple(
                child.__class__(
                    child.support,
                    child.matrix,
                    replace(child.fit, converged=True, recession=False),
                    child.penalty,
                    3.0 - index,
                    1.0,
                    child.closure_null_nll,
                    1.0,
                )
                for index, child in enumerate(raw_children)
            )
            records = {
                EMPTY_SUPPORT: empty,
                parent.support: parent,
                **{child.support: child for child in children},
            }

            def fit(support, *_args, **_kwargs):
                return records[support]

            def cleanup(_current, **_):
                return _MoveDecision(None, True)

            def add(current, **_):
                if current.support != parent.support:
                    return None
                for child in children:
                    edge = (parent.support, child.support)
                    if edge not in optimizer._conditional_parent_forbidden:
                        return child
                return None

            with (
                patch.object(optimizer, "fit", side_effect=fit),
                patch.object(
                    optimizer, "_best_terminal_cleanup_decision", side_effect=cleanup
                ),
                patch.object(optimizer, "_first_conditional_addition", side_effect=add),
            ):
                terminals, _, _, _ = optimizer._global_exact_family_frontier((parent,))
            self.assertEqual(
                {record.support for record in terminals},
                {children[0].support},
            )
            self.assertEqual(optimizer.diagnostics.global_family_frontier_add_moves, 1)
            optimizer.close()

    def test_conditional_basin_frontier_branches_and_reprices_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                search_mode="atomic_rashomon_frontier",
                terminal_add_audit="exact",
                conditional_basin_branching=True,
                ensemble_residual_search=True,
                ensemble_residual_dictionary_repricing=True,
                pricing_devices=(),
                solver_tolerance=1.0e-7,
                solver_max_iter=120,
            )
            context = Context.make(data, np.arange(60, dtype=np.int32))
            optimizer = SupportOptimizer(context, config)
            empty = optimizer.records[EMPTY_SUPPORT]
            parent = optimizer._attach_rule_score(
                optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)), empty)
            )
            raw_children = tuple(
                optimizer._attach_rule_score(
                    optimizer.fit(parent.support.add(rule), parent)
                )
                for rule in (
                    RuleIdentity((1,), 0, 1),
                    RuleIdentity((0, 1), 0, 1),
                )
            )
            parent = replace(parent, score=1.0, rule_score=1.0)
            children = tuple(
                replace(
                    child,
                    fit=replace(child.fit, converged=True, recession=False),
                    score=3.0 - index,
                    rule_score=1.0,
                )
                for index, child in enumerate(raw_children)
            )
            records = {
                EMPTY_SUPPORT: empty,
                parent.support: parent,
                **{child.support: child for child in children},
            }

            def fit(support, *_args, **_kwargs):
                return records[support]

            def cleanup(_current, **_):
                return _MoveDecision(None, True)

            def branch(current):
                return children if current.support == parent.support else ()

            repricing_calls = []

            def reprice(items):
                repricing_calls.append(tuple(item.support for item in items))
                return items

            with (
                patch.object(optimizer, "fit", side_effect=fit),
                patch.object(
                    optimizer, "_best_terminal_cleanup_decision", side_effect=cleanup
                ),
                patch.object(
                    optimizer, "_conditional_basin_branch_additions", side_effect=branch
                ),
                patch.object(
                    optimizer, "_ensemble_residual_route_order", side_effect=reprice
                ),
            ):
                terminals, _, _, _ = optimizer._global_exact_family_frontier((parent,))
            self.assertEqual(
                {record.support for record in terminals},
                {child.support for child in children},
            )
            self.assertEqual(optimizer.diagnostics.global_family_frontier_add_moves, 2)
            self.assertEqual(
                optimizer.diagnostics.ensemble_dictionary_repricing_rounds, 1
            )
            self.assertEqual(len(repricing_calls), 1)
            optimizer.close()

    def test_block_score_conditional_frontier_follows_one_score_successor(self) -> None:
        """Conditional branching must not promote block-score mode to exact."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                search_mode="atomic_rashomon_frontier",
                terminal_add_audit="block_score",
                conditional_basin_branching=True,
                pricing_devices=(),
                solver_tolerance=1.0e-7,
                solver_max_iter=120,
            )
            context = Context.make(data, np.arange(60, dtype=np.int32))
            optimizer = SupportOptimizer(context, config)
            empty = optimizer.records[EMPTY_SUPPORT]
            parent = optimizer._attach_rule_score(
                optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)), empty)
            )
            child = optimizer._attach_rule_score(
                optimizer.fit(
                    parent.support.add(RuleIdentity((1,), 0, 1)), parent
                )
            )
            parent = replace(parent, score=1.0, rule_score=1.0)
            child = replace(
                child,
                fit=replace(child.fit, converged=True, recession=False),
                score=2.0,
                rule_score=1.0,
            )
            records = {
                EMPTY_SUPPORT: empty,
                parent.support: parent,
                child.support: child,
            }

            def fit(support, *_args, **_kwargs):
                return records[support]

            def cleanup(_current, **_kwargs):
                return _MoveDecision(None, True)

            def block_add(current, **_kwargs):
                return child if current.support == parent.support else None

            with (
                patch.object(optimizer, "fit", side_effect=fit),
                patch.object(
                    optimizer, "_best_terminal_cleanup_decision", side_effect=cleanup
                ),
                patch.object(
                    optimizer,
                    "_conditional_basin_branch_additions",
                    side_effect=AssertionError("exact basin branching was called"),
                ),
                patch.object(
                    optimizer, "_best_composite_fast_addition", side_effect=block_add
                ),
            ):
                terminals, _, _, _ = optimizer._global_exact_family_frontier((parent,))
            self.assertEqual(
                tuple(record.support for record in terminals), (child.support,)
            )
            self.assertEqual(optimizer.diagnostics.global_family_frontier_add_moves, 1)
            optimizer.close()

    def test_conditional_basin_search_reopens_intermediate_supports(self) -> None:
        """Positive roots/prefixes, not only greedy terminals, seed the DAG."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                search_mode="atomic_rashomon_frontier",
                terminal_add_audit="exact",
                conditional_basin_branching=True,
                ensemble_residual_search=True,
                ensemble_residual_dictionary_repricing=True,
                rashomon_branching=True,
                pricing_devices=(),
                pricing_workers=2,
                solver_tolerance=1.0e-7,
                solver_max_iter=120,
            )
            context = Context.make(data, np.arange(80, dtype=np.int32))
            optimizer = SupportOptimizer(context, config)
            try:
                result = optimizer.search()
                diagnostics = result.diagnostics
                self.assertGreater(
                    diagnostics.conditional_basin_frontier_seed_states,
                    len(result.terminals),
                )
                self.assertGreater(
                    diagnostics.conditional_basin_frontier_intermediate_seeds, 0
                )
                self.assertGreater(diagnostics.conditional_basin_frontier_children, 0)
                self.assertTrue(
                    any(len(record.support.rules) > 1 for record in result.family)
                )
            finally:
                optimizer.close()

    def test_mixture_target_sufficient_statistics_are_lossless(self) -> None:
        generator = np.random.default_rng(7391)
        intensity = generator.lognormal(mean=-0.4, sigma=0.8, size=(4, 80))
        baseline = generator.lognormal(mean=-0.2, sigma=0.4, size=(4, 1))
        event = generator.poisson(0.12, size=80).astype(np.float64)
        active_weight = generator.lognormal(mean=0.1, sigma=0.3, size=80)
        weights = np.asarray([0.11, 0.24, 0.29, 0.36], dtype=np.float64)
        for likelihood in ("poisson", "first_event_cloglog"):
            components = _SparseComponents(
                active_intensity=intensity,
                baseline_intensity=baseline,
                active_event=event,
                active_noevent=(
                    active_weight
                    if likelihood == "poisson"
                    else np.ones(80, dtype=np.float64) - event
                ),
                inactive_baseline_groups=np.asarray([413.0]),
                likelihood=likelihood,
                tick_exposure=0.25 if likelihood == "poisson" else 1.0,
            )
            expected_nll, expected_gradient = _mixture_nll_gradient(components, weights)
            actual_nll, actual_gradient = _mixture_statistics_nll_gradient(
                _mixture_sufficient_statistics(components), weights
            )
            self.assertAlmostEqual(actual_nll, expected_nll, places=11)
            np.testing.assert_allclose(
                actual_gradient,
                expected_gradient,
                rtol=2e-13,
                atol=2e-13,
            )

    def test_frozen_sparse_predictor_matches_dense_model_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = controlled_synthetic_dataset(Path(directory) / "data")
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=64 * 1024**2)
            rule = RuleIdentity((0, 1), 2, -1)
            support = Support.of((rule,))
            for term in hierarchy_closure(support):
                engine.set_closure_sign(term, 1 if term.antecedent == (0,) else -1)
            dense = engine.model_matrix(context, support)
            coefficients = np.linspace(-0.35, 0.45, dense.dimension)
            coefficients[dense.free_dimension :] = np.abs(
                coefficients[dense.free_dimension :]
            )
            expected = engine.linear_predictor(context, dense, coefficients)
            metadata = engine.model_metadata(support, forced_closure=dense.closure)
            rows, actual = engine.frozen_active_predictor(
                context, metadata, coefficients
            )
            np.testing.assert_array_equal(rows, dense.active_rows)
            np.testing.assert_allclose(actual, expected[rows], rtol=2e-13, atol=2e-13)
            subset = rows[::3]
            np.testing.assert_allclose(
                engine.frozen_linear_predictor_at_rows(
                    context, metadata, coefficients, subset
                ),
                expected[subset],
                rtol=2e-13,
                atol=2e-13,
            )

    def test_contextual_rule_contribution_uses_exclusive_total_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            context = Context.make(data, np.arange(60, dtype=np.int32))
            engine = ResponseEngine(
                data,
                lag=3,
                knot_count=2,
                cache_bytes=8 * 1024**2,
            )
            ancestor = RuleIdentity((0,), 0, 1)
            root = RuleIdentity((0, 1), 2, -1)
            support = Support.of((ancestor, root))
            self.assertEqual(hierarchy_closure(support), ())
            matrix = engine.model_matrix(context, support)
            coefficients = np.zeros(matrix.dimension, dtype=np.float64)
            coefficients[matrix.rule_slices[0]] = (0.4, 0.15)
            coefficients[matrix.rule_slices[1]] = (0.3, 0.25)
            full_eta = engine.linear_predictor(context, matrix, coefficients)
            np.testing.assert_allclose(
                full_eta[matrix.active_rows],
                engine.design_at_rows_with_context(context, matrix, matrix.active_rows)
                @ coefficients,
                rtol=2e-13,
                atol=2e-13,
            )
            rows = engine.footprint_rows(context, root, horizon=3)

            expected = np.zeros(len(rows), dtype=np.float64)
            blocks = engine.total_state_rule_blocks(context, support)
            self.assertEqual(np.intersect1d(blocks[0].rows, blocks[1].rows).size, 0)
            positions = np.searchsorted(blocks[1].rows, rows)
            matched = positions < len(blocks[1].rows)
            safe = np.minimum(positions, len(blocks[1].rows) - 1)
            matched &= blocks[1].rows[safe] == rows
            expected[matched] = -(
                blocks[1].values[positions[matched]]
                @ coefficients[matrix.rule_slices[1]]
            )
            actual = engine.frozen_contextual_rule_contribution_at_rows(
                context,
                matrix,
                coefficients,
                rule_index=1,
                rows=rows,
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

            # The singleton keeps its own interpretation even when a
            # high-order descendant is present in the same support.
            singleton_rows = engine.footprint_rows(context, ancestor, horizon=3)
            singleton_expected = np.zeros(len(singleton_rows), dtype=np.float64)
            block = blocks[0]
            positions = np.searchsorted(block.rows, singleton_rows)
            matched = positions < len(block.rows)
            safe = np.minimum(positions, len(block.rows) - 1)
            matched &= block.rows[safe] == singleton_rows
            singleton_expected[matched] = (
                block.values[positions[matched]] @ coefficients[matrix.rule_slices[0]]
            )
            singleton_actual = engine.frozen_contextual_rule_contribution_at_rows(
                context,
                matrix,
                coefficients,
                rule_index=0,
                rows=singleton_rows,
            )
            np.testing.assert_allclose(
                singleton_actual,
                singleton_expected,
                rtol=2e-13,
                atol=2e-13,
            )

    def test_intensity_mixture_keeps_complementary_models(self) -> None:
        intensities = np.asarray(
            [
                [2.0, 2.0, 0.01, 0.01],
                [0.01, 0.01, 2.0, 2.0],
            ],
            dtype=np.float64,
        )
        components = _SparseComponents(
            active_intensity=intensities,
            baseline_intensity=np.ones((2, 1), dtype=np.float64),
            active_event=np.ones(4, dtype=np.float64),
            active_noevent=np.ones(4, dtype=np.float64),
            inactive_baseline_groups=np.zeros(1, dtype=np.float64),
            likelihood="poisson",
            tick_exposure=1.0,
        )
        baseline_nll = float(4.0 * (0.2 - np.log(0.2)))
        selected = _select_intensity_family(
            components,
            np.zeros(2, dtype=np.float64),
            baseline_nll=baseline_nll,
            n_entities=4,
            tolerance=1.0e-10,
            search_tolerance=1.0e-10,
        )
        self.assertEqual(selected.indices, (0, 1))
        np.testing.assert_allclose(selected.weights, (0.5, 0.5), atol=1.0e-6)

    def test_family_objective_removes_duplicate_but_keeps_complement(self) -> None:
        intensities = np.asarray(
            [
                [2.0, 2.0, 0.01, 0.01],
                [0.01, 0.01, 2.0, 2.0],
                [2.0, 2.0, 0.01, 0.01],
            ],
            dtype=np.float64,
        )
        components = _SparseComponents(
            active_intensity=intensities,
            baseline_intensity=np.ones((3, 1), dtype=np.float64),
            active_event=np.ones(4, dtype=np.float64),
            active_noevent=np.ones(4, dtype=np.float64),
            inactive_baseline_groups=np.zeros(1, dtype=np.float64),
            likelihood="poisson",
            tick_exposure=1.0,
        )
        selected = _select_intensity_family(
            components,
            np.ones(3, dtype=np.float64),
            baseline_nll=float(4.0 * (0.2 - np.log(0.2))),
            n_entities=4,
            tolerance=1.0e-10,
            search_tolerance=1.0e-10,
        )
        self.assertEqual(len(selected.indices), 2)
        self.assertIn(1, selected.indices)
        self.assertEqual(len(set(selected.indices).intersection({0, 2})), 1)
        self.assertGreater(selected.score, 0.0)
        self.assertGreaterEqual(selected.moves, 1)

    def test_exact_fit_plan_pipelines_cpu_builders_onto_cuda_devices(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer.config = SimpleNamespace(
            pricing_devices=("cuda:0", "cuda:1"),
            exact_workers=6,
            pricing_workers=12,
        )
        workers, devices, threads = optimizer._exact_fit_plan()
        self.assertEqual(workers, 6)
        self.assertEqual(
            devices,
            ("cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0", "cuda:1"),
        )
        self.assertEqual(threads, 2)

        for configured in ((), ("cpu",)):
            optimizer.config = SimpleNamespace(
                pricing_devices=configured,
                exact_workers=3,
                pricing_workers=12,
            )
            workers, devices, threads = optimizer._exact_fit_plan()
            self.assertEqual(workers, 3)
            self.assertEqual(devices, ("cpu", "cpu", "cpu"))
            self.assertEqual(threads, 4)

    def test_fast_derivative_grid_uses_exact_group_scatter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 13)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                current = optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)))
                self.assertTrue(current.fit.converged)
                rows = np.arange(optimizer.context.n_grid, dtype=np.int64)
                _, expected_first, expected_second = optimizer._pricing_rows(
                    current, rows
                )
                optimizer._build_fast_derivative_grid(current)
                assert optimizer._fast_derivative_state is not None
                actual_first, actual_second = optimizer._fast_derivative_state[1:]
                np.testing.assert_allclose(
                    actual_first, expected_first, rtol=1e-14, atol=1e-14
                )
                np.testing.assert_allclose(
                    actual_second, expected_second, rtol=1e-14, atol=1e-14
                )
                self.assertEqual(optimizer.diagnostics.fast_derivative_group_builds, 1)
                self.assertEqual(optimizer.diagnostics.fast_derivative_tiled_builds, 0)
            finally:
                optimizer.close()

    def test_structural_lazy_bound_is_safe_and_fails_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 13)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                cache_bytes=8 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                empty = optimizer.records[Support(())]
                current = SupportRecord(
                    empty.support,
                    empty.matrix,
                    empty.fit,
                    empty.penalty,
                    1.0,
                )
                rules = (
                    RuleIdentity((0,), 0, 1),
                    RuleIdentity((1,), 0, -1),
                )
                with patch.object(
                    optimizer,
                    "saturated_upper_score",
                    side_effect=(0.0, float("inf")),
                ):
                    survivors = optimizer._structural_upper_survivors(
                        current, rules, threshold=0.0
                    )
                self.assertEqual(survivors, (rules[1],))
                self.assertEqual(optimizer.diagnostics.structural_lazy_bound_audits, 2)
                self.assertEqual(optimizer.diagnostics.structural_lazy_bound_screens, 1)
            finally:
                optimizer.close()

    def test_sparse_target_gather_and_uniform_weight_fast_paths_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            codes = np.arange(data.n_entities, dtype=np.int32)
            context = Context.make(data, codes)
            rows = np.arange(0, context.n_grid, 2, dtype=np.int64)
            expected = np.zeros(len(rows), dtype=np.float64)
            positions = np.searchsorted(context.target_rows, rows)
            matched = positions < len(context.target_rows)
            safe = np.minimum(positions, len(context.target_rows) - 1)
            matched &= context.target_rows[safe] == rows
            expected[matched] = context.target_counts[positions[matched]]
            np.testing.assert_array_equal(
                context.target_counts_at_sorted_rows(rows), expected
            )
            np.testing.assert_array_equal(context.weights_at_rows(rows), 1.0)
            np.testing.assert_array_equal(context.all_row_weights(), 1.0)
            self.assertEqual(context.weighted_n_grid, context.n_grid)

            weighted = Context.make(
                data,
                codes,
                entity_weights=np.full(len(codes), 2.0),
                population_entities=len(codes),
            )
            np.testing.assert_array_equal(weighted.weights_at_rows(rows), 2.0)
            self.assertEqual(weighted.weighted_n_grid, 2.0 * weighted.n_grid)

    def test_temporal_baseline_cells_are_frozen_and_cover_each_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit = np.arange(0, 60, dtype=np.int32)
            cert = np.arange(60, 90, dtype=np.int32)
            engine = ResponseEngine(
                data,
                lag=3,
                knot_count=2,
                cache_bytes=1024**2,
                baseline_time_bins=4,
            )
            for codes in (fit, cert):
                context = Context.make(data, codes)
                counts = context.temporal_baseline_counts(time_bins=4)
                self.assertEqual(counts.shape[1], engine.free_baseline_dimension)
                np.testing.assert_array_equal(
                    counts.sum(axis=1), context.ends - context.starts + 1
                )
                totals = context.weighted_baseline_totals(
                    engine.free_baseline_dimension, time_bins=4
                )
                np.testing.assert_array_equal(totals, counts.sum(axis=0))
                rows = np.arange(context.n_grid, dtype=np.int64)
                groups = context.temporal_baseline_groups_at_rows(rows, time_bins=4)
                self.assertTrue(np.all(groups >= 0))
                self.assertTrue(np.all(groups < engine.free_baseline_dimension))

            context = Context.make(data, fit)
            support = Support.of((RuleIdentity((0,), 0, 1),))
            matrix = engine.model_matrix(context, support)
            blocks = engine.total_state_rule_blocks(context, support)
            sparse = fit_sparse_grid_model(
                context,
                blocks,
                (1,),
                likelihood=data.likelihood,
                tick_exposure=engine.tick_exposure,
                tolerance=1.0e-8,
                max_iter=120,
                baseline_group_count=engine.free_baseline_dimension,
                baseline_time_bins=4,
            )
            # This synthetic split can have a genuine recession cell; the
            # regression contract is that sparse temporal grouping reaches
            # the solver with the exact dense model dimension instead of
            # failing at the former static-baseline mismatch.
            self.assertEqual(sparse.coefficients.shape, (matrix.dimension,))

    def test_zero_exposure_dynamic_baseline_sparse_fit_matches_dense_exactly(self) -> None:
        """Observation masks must not reappear as sparse no-event rows."""

        with tempfile.TemporaryDirectory() as directory:
            base = synthetic_dataset(
                Path(directory) / "data",
                600,
                likelihood="first_event_cloglog",
            )
            lengths = base.end_times - base.start_times + 1
            local_time = np.concatenate(
                [np.arange(int(length), dtype=np.int64) for length in lengths]
            )
            entity_code = np.repeat(np.arange(base.n_entities), lengths)
            strata = np.ascontiguousarray(
                (local_time + entity_code) % 2, dtype=np.int16
            )
            exposure = np.ones(len(local_time), dtype=np.float64)
            exposure[(local_time % 4) == 0] = 0.0
            offsets = np.empty(base.n_entities + 1, dtype=np.int64)
            offsets[0] = 0
            offsets[1:] = np.cumsum(lengths, dtype=np.int64)
            target_rows = (
                offsets[base.target_entities]
                + base.target_times
                - base.start_times[base.target_entities]
            )
            exposure[target_rows] = 1.0
            data = replace(
                base,
                baseline_cell_strata=strata,
                baseline_cell_exposure=np.ascontiguousarray(exposure),
            )
            context = Context.make(
                data, np.arange(data.n_entities, dtype=np.int32)
            )
            engine = ResponseEngine(
                data,
                lag=3,
                knot_count=2,
                cache_bytes=64 * 1024**2,
                baseline_time_bins=1,
            )
            support = Support.of((RuleIdentity((0,), 0, 1),))
            matrix = engine.model_matrix(context, support)
            dense = fit_model_matrix(
                matrix,
                likelihood="first_event_cloglog",
                tolerance=1.0e-9,
                max_iter=200,
            )
            sparse = fit_sparse_grid_model(
                context,
                engine.total_state_rule_blocks(context, support),
                (1,),
                likelihood="first_event_cloglog",
                tick_exposure=engine.tick_exposure,
                tolerance=1.0e-9,
                max_iter=200,
                baseline_group_count=engine.free_baseline_dimension,
                baseline_time_bins=1,
            )
            self.assertTrue(dense.converged, dense.message)
            self.assertTrue(sparse.converged, sparse.message)
            np.testing.assert_allclose(sparse.nll, dense.nll, rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(
                sparse.coefficients,
                dense.coefficients,
                rtol=2e-8,
                atol=2e-8,
            )
            entity_loss = _entity_losses_frozen(engine, context, matrix, dense)
            np.testing.assert_allclose(
                np.sum(entity_loss), dense.nll, rtol=2e-11, atol=2e-11
            )
            profile = _sparse_profile(engine, context, matrix, dense)
            components = _components_from_profiles(engine, context, [profile])
            mixture_nll, _ = _mixture_nll_gradient(
                components, np.ones(1, dtype=np.float64)
            )
            np.testing.assert_allclose(
                mixture_nll, dense.nll, rtol=2e-11, atol=2e-11
            )
            holder = SimpleNamespace(
                context=context,
                engine=engine,
                config=SimpleNamespace(pricing_workers=1),
                _unit_row_exposure=False,
            )
            with patch(
                "crbstpp.search.entity_loss_contrast",
                side_effect=AssertionError("masked rows must bypass native contrast"),
            ):
                contrast = SupportOptimizer._entity_loss_difference_same_matrix(
                    holder, matrix, dense, dense
                )
            np.testing.assert_allclose(contrast, 0.0, rtol=0.0, atol=2e-13)
            with patch(
                "crbstpp.dependency.dependency_row_derivatives",
                side_effect=AssertionError("masked rows must bypass native derivatives"),
            ):
                dependency = model_dependency_complexity(
                    engine,
                    context,
                    matrix,
                    dense,
                    dependence_horizon_ticks=3,
                )
            self.assertTrue(np.isfinite(dependency.effective_dimension))

    @unittest.skipUnless(cuda_available(), "CUDA implicit pricing is unavailable")
    def test_dynamic_baseline_exposure_fused_moments_match_dense_reference(self) -> None:
        """Fused ranking must use every time-varying baseline and exposure mask."""

        with tempfile.TemporaryDirectory() as directory:
            base = synthetic_dataset(
                Path(directory) / "data",
                180,
                likelihood="first_event_cloglog",
            )
            lengths = base.end_times - base.start_times + 1
            local_time = np.concatenate(
                [np.arange(int(length), dtype=np.int64) for length in lengths]
            )
            entity_code = np.repeat(np.arange(base.n_entities), lengths)
            strata = np.ascontiguousarray(
                (local_time + entity_code) % 2, dtype=np.int16
            )
            exposure = np.ones(len(local_time), dtype=np.float64)
            exposure[(local_time % 4) == 0] = 0.0
            offsets = np.empty(base.n_entities + 1, dtype=np.int64)
            offsets[0] = 0
            offsets[1:] = np.cumsum(lengths, dtype=np.int64)
            target_rows = (
                offsets[base.target_entities]
                + base.target_times
                - base.start_times[base.target_entities]
            )
            exposure[target_rows] = 1.0
            data = replace(
                base,
                baseline_cell_strata=strata,
                baseline_cell_exposure=np.ascontiguousarray(exposure),
            )
            context = Context.make(
                data, np.arange(data.n_entities, dtype=np.int32)
            )
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                solver_tolerance=1e-9,
                solver_max_iter=160,
                cache_bytes=64 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=("cuda:0",),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(context, config)
            try:
                current = optimizer.records[EMPTY_SUPPORT]
                self.assertFalse(optimizer._unit_row_exposure)
                compact = optimizer._compact_cloglog_derivatives(current)
                self.assertIsNotNone(compact)
                assert compact is not None

                groups, current_x = optimizer._implicit_current_groups(current)
                group_eta = current_x @ current.fit.coefficients
                row_baseline = context.temporal_baseline_groups_at_rows(
                    np.arange(context.n_grid, dtype=np.int64), time_bins=1
                )
                observed = exposure > 0.0
                np.testing.assert_allclose(
                    group_eta[groups[observed]],
                    current.fit.coefficients[row_baseline[observed]],
                    rtol=0.0,
                    atol=1e-14,
                )
                np.testing.assert_allclose(
                    current_x[groups[~observed]],
                    0.0,
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    compact[0][groups[~observed]],
                    0.0,
                    rtol=0.0,
                    atol=0.0,
                )
                baseline_eta = current.matrix.x @ current.fit.coefficients
                np.testing.assert_allclose(
                    baseline_eta[optimizer._baseline_group_by_row],
                    current.fit.coefficients[row_baseline],
                    rtol=0.0,
                    atol=1e-14,
                )

                pattern = ("atomic", (0,))
                specifications = (((0,), 0, (), ((pattern, 0),)),)
                fused = optimizer._implicit_hierarchy_moments(
                    current,
                    specifications,
                    None,
                    None,
                    device="cuda:0",
                )
                self.assertIsNotNone(fused)
                assert fused is not None
                block = optimizer.engine.block(
                    context, (0,), 0, relation="atomic"
                )
                reference = optimizer._joint_hierarchy_moments(
                    current, (block,), device="cpu"
                )
                for actual, expected in zip(
                    fused,
                    (
                        reference[0][None, :],
                        reference[1][None, :, :],
                        reference[2][None, :, :],
                    ),
                    strict=True,
                ):
                    np.testing.assert_allclose(
                        actual, expected, rtol=3e-11, atol=3e-11
                    )
            finally:
                optimizer.close()

    def test_sparse_ensemble_uses_the_same_temporal_baseline_as_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(
                data,
                lag=3,
                knot_count=2,
                cache_bytes=1024**2,
                baseline_time_bins=4,
            )
            rows = np.arange(context.n_grid, dtype=np.int64)
            profile = _SparseProfile(
                rows=rows,
                intensity=np.ones(len(rows), dtype=np.float64),
                baseline_intensity=np.ones(
                    engine.free_baseline_dimension, dtype=np.float64
                ),
            )
            components = _components_from_profiles(
                engine,
                context,
                [profile],
            )
            np.testing.assert_allclose(components.inactive_baseline_groups, 0.0)

    def test_direct_record_mixture_statistics_match_sparse_reference(self) -> None:
        for likelihood in ("first_event_cloglog", "poisson"):
            with (
                self.subTest(likelihood=likelihood),
                tempfile.TemporaryDirectory() as directory,
            ):
                data = synthetic_dataset(
                    Path(directory) / "data", 90, likelihood=likelihood
                )
                context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
                config = RunConfig(
                    dataset=str(data.root),
                    q_max=1,
                    impact_lag=3,
                    knot_count=2,
                    formation_windows=(0,),
                    solver_tolerance=1.0e-8,
                    solver_max_iter=120,
                    cache_bytes=16 * 1024**2,
                    early_warning_horizon=3,
                    pricing_devices=("cpu",),
                )
                optimizer = SupportOptimizer(context, config)
                try:
                    record = optimizer.fit(
                        Support.of((RuleIdentity((0,), 0, 1),)),
                        optimizer.records[EMPTY_SUPPORT],
                    )
                    self.assertTrue(record.fit.converged, record.fit.message)
                    direct = optimizer._record_mixture_statistics(record)
                    metadata_direct = optimizer._record_mixture_statistics(
                        freeze_support_record(record)
                    )
                    reference = _mixture_sufficient_statistics(
                        _components_from_profiles(
                            optimizer.engine,
                            context,
                            [
                                _sparse_profile(
                                    optimizer.engine,
                                    context,
                                    record.matrix,
                                    record.fit,
                                )
                            ],
                        )
                    )
                    np.testing.assert_allclose(
                        direct.linear, reference.linear, rtol=2e-11, atol=2e-11
                    )
                    np.testing.assert_allclose(
                        metadata_direct.linear,
                        reference.linear,
                        rtol=2e-11,
                        atol=2e-11,
                    )
                    np.testing.assert_allclose(
                        metadata_direct.target_intensity,
                        reference.target_intensity,
                        rtol=2e-11,
                        atol=2e-11,
                    )
                    np.testing.assert_allclose(
                        direct.target_intensity,
                        reference.target_intensity,
                        rtol=2e-11,
                        atol=2e-11,
                    )
                    np.testing.assert_array_equal(
                        direct.target_weight, reference.target_weight
                    )
                    atom_record = optimizer.fit(
                        Support.of((RuleIdentity((1,), 0, 1),)),
                        optimizer.records[EMPTY_SUPPORT],
                    )
                    self.assertTrue(atom_record.fit.converged, atom_record.fit.message)
                    atom = optimizer._record_mixture_statistics(atom_record)
                    optimizer._positive_atom_by_antecedent = {
                        RuleIdentity((0,), 0, 1).pattern_key: record,
                        RuleIdentity((1,), 0, 1).pattern_key: atom_record,
                    }
                    optimizer._standalone_mixture_statistics = direct
                    optimizer._standalone_mixture_index = {
                        RuleIdentity((0,), 0, 1).pattern_key: 0
                    }
                    optimizer._separate_family_score_cache = {
                        (record.support, RuleIdentity((0,), 0, 1).pattern_key): 1.0,
                        (record.support, RuleIdentity((1,), 0, 1).pattern_key): 2.0,
                    }
                    optimizer._update_standalone_mixture_statistics(
                        RuleIdentity((1,), 0, 1).pattern_key,
                        atom_record,
                    )
                    updated = optimizer._standalone_mixture_statistics
                    self.assertIsNotNone(updated)
                    if updated is None:
                        self.fail("incremental mixture update disappeared")
                    np.testing.assert_allclose(
                        updated.linear,
                        np.asarray([direct.linear[0], atom.linear[0]]),
                    )
                    np.testing.assert_allclose(
                        updated.target_intensity,
                        np.vstack(
                            (direct.target_intensity[0], atom.target_intensity[0])
                        ),
                    )
                    self.assertIn(
                        (record.support, RuleIdentity((0,), 0, 1).pattern_key),
                        optimizer._separate_family_score_cache,
                    )
                    self.assertNotIn(
                        (record.support, RuleIdentity((1,), 0, 1).pattern_key),
                        optimizer._separate_family_score_cache,
                    )
                    batched_nll, batched_converged = (
                        optimizer._binary_parent_atom_mixture_nlls(direct, atom, (0,))
                    )
                    paired = _MixtureSufficientStatistics(
                        np.asarray([direct.linear[0], atom.linear[0]]),
                        np.vstack(
                            (direct.target_intensity[0], atom.target_intensity[0])
                        ),
                        direct.target_weight,
                        direct.likelihood,
                    )
                    generic = _fit_simplex_statistics(
                        paired, tolerance=config.solver_tolerance
                    )
                    self.assertTrue(generic.converged)
                    self.assertTrue(batched_converged[0])
                    self.assertAlmostEqual(batched_nll[0], generic.nll, delta=2e-9)
                finally:
                    optimizer.close()

    def test_full_discovery_reuses_prebuilt_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            codes = np.arange(data.n_entities, dtype=np.int32)
            context = Context.make(data, codes)
            config = RunConfig(dataset=str(data.root), discovery_sampling="full")
            observed, _ = _discovery_context(data, codes, config, full_context=context)
            self.assertIs(observed, context)

    def test_discovery_mode_cannot_hide_reference_cohort_under_full(self) -> None:
        with self.assertRaisesRegex(ValueError, "full always means complete D_fit"):
            RunConfig(
                dataset="full",
                discovery_sampling="full",
                discovery_reference_dataset="reference",
            ).validate()
        with self.assertRaisesRegex(ValueError, "requires discovery_reference_dataset"):
            RunConfig(
                dataset="full",
                discovery_sampling="reference_cohort",
            ).validate()
        RunConfig(
            dataset="full",
            discovery_sampling="reference_cohort",
            discovery_reference_dataset="reference",
        ).validate()

    @unittest.skipUnless(cuda_available(), "CUDA implicit pricing is unavailable")
    def test_implicit_completion_moments_match_sparse_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(
                Path(directory) / "data", 120, likelihood="poisson"
            )
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=100,
                cache_bytes=256 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=("cuda:0",),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(context, config)
            try:
                current = optimizer.fit(
                    Support.of((RuleIdentity((0,), 0, 1),)),
                    device="cuda:0",
                )
                optimizer._build_fast_derivative_grid(current)
                assert optimizer._fast_derivative_state is not None
                first, second = optimizer._fast_derivative_state[1:]
                antecedent = (0, 1)
                window = 2
                trial = current.support.add(RuleIdentity(antecedent, window, 1))
                additions = tuple(
                    sorted(set(hierarchy_closure(trial)) - set(current.matrix.closure))
                )
                keys = tuple(
                    (term.antecedent, int(term.window)) for term in additions
                ) + ((antecedent, window),)
                specifications = ((antecedent, window, additions, keys),)
                implicit = optimizer._implicit_hierarchy_moments(
                    current,
                    specifications,
                    first,
                    second,
                    device="cuda:0",
                )
                self.assertIsNotNone(implicit)
                assert implicit is not None
                blocks = tuple(
                    optimizer.engine.block(context, term.antecedent, term.window)
                    for term in additions
                ) + (optimizer.engine.block(context, antecedent, window),)
                reference = optimizer._joint_hierarchy_moments(
                    current, blocks, device="cuda:0"
                )
                for actual, expected in zip(
                    implicit,
                    (
                        reference[0][None, :],
                        reference[1][None, :, :],
                        reference[2][None, :, :],
                    ),
                    strict=True,
                ):
                    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
                selected_columns = np.asarray(
                    [0, current.matrix.dimension - 1], dtype=np.int64
                )
                reduced = optimizer._implicit_hierarchy_moments(
                    current,
                    specifications,
                    first,
                    second,
                    device="cuda:0",
                    current_columns=selected_columns,
                )
                self.assertIsNotNone(reduced)
                assert reduced is not None
                np.testing.assert_allclose(
                    reduced[0], implicit[0], rtol=2e-11, atol=2e-11
                )
                np.testing.assert_allclose(
                    reduced[1], implicit[1], rtol=2e-11, atol=2e-11
                )
                np.testing.assert_allclose(
                    reduced[2],
                    implicit[2][:, selected_columns],
                    rtol=2e-11,
                    atol=2e-11,
                )
                gradient_only = optimizer._implicit_hierarchy_moments(
                    current,
                    specifications,
                    first,
                    second,
                    device="cuda:0",
                    gradient_only=True,
                )
                self.assertIsNotNone(gradient_only)
                assert gradient_only is not None
                np.testing.assert_allclose(
                    gradient_only[0], implicit[0], rtol=2e-11, atol=2e-11
                )
                self.assertEqual(gradient_only[1].shape, (1, 0, 0))
                self.assertEqual(
                    gradient_only[2].shape,
                    (1, 0, implicit[0].shape[-1]),
                )
                with_footprint = optimizer._implicit_hierarchy_moments(
                    current,
                    specifications,
                    first,
                    second,
                    device="cuda:0",
                    gradient_only=True,
                    collect_footprint_stats=True,
                )
                self.assertIsNotNone(with_footprint)
                assert with_footprint is not None
                self.assertEqual(len(with_footprint), 4)
                candidate_rows = np.unique(
                    np.concatenate([block.rows for block in blocks])
                )
                current_groups, current_x = optimizer._implicit_current_groups(current)
                signed_state = np.zeros(len(current_x), dtype=np.uint8)
                for rule, block in zip(
                    current.support.rules, current.matrix.rule_slices, strict=True
                ):
                    signed_state[np.any(np.abs(current_x[:, block]) > 0.0, axis=1)] |= (
                        1 if rule.sign > 0 else 2
                    )
                categories = signed_state[current_groups[candidate_rows]]
                for category in range(4):
                    rows = candidate_rows[categories == category]
                    groups_for_rows = optimizer._baseline_group_by_row[rows]
                    expected_rows = np.bincount(
                        groups_for_rows,
                        minlength=len(optimizer._baseline_group_exposure),
                    ).astype(np.float64)
                    expected_events = np.bincount(
                        groups_for_rows,
                        weights=context.target_counts_at_sorted_rows(rows),
                        minlength=len(optimizer._baseline_group_exposure),
                    ).astype(np.float64)
                    np.testing.assert_allclose(
                        with_footprint[3][0, category, :, 0],
                        expected_rows,
                        atol=0.0,
                        rtol=0.0,
                    )
                    np.testing.assert_allclose(
                        with_footprint[3][0, category, :, 1],
                        expected_events,
                        atol=0.0,
                        rtol=0.0,
                    )

                with_group_footprint = optimizer._implicit_hierarchy_moments(
                    current,
                    specifications,
                    first,
                    second,
                    device="cuda:0",
                    gradient_only=True,
                    collect_footprint_stats=True,
                    collect_group_footprint_stats=True,
                )
                self.assertIsNotNone(with_group_footprint)
                assert with_group_footprint is not None
                self.assertEqual(len(with_group_footprint), 5)
                relaxation = optimizer._informative_group_relaxation(current)
                mapped = relaxation.group_map[current_groups[candidate_rows]]
                keep = mapped >= 0
                expected_rows = np.bincount(
                    mapped[keep], minlength=len(relaxation.exposure)
                ).astype(np.float64)
                expected_events = np.bincount(
                    mapped[keep],
                    weights=context.target_counts_at_sorted_rows(candidate_rows[keep]),
                    minlength=len(relaxation.exposure),
                ).astype(np.float64)
                np.testing.assert_allclose(
                    with_group_footprint[4][0, :, 0],
                    expected_rows,
                    atol=0.0,
                    rtol=0.0,
                )
                np.testing.assert_allclose(
                    with_group_footprint[4][0, :, 1],
                    expected_events,
                    atol=0.0,
                    rtol=0.0,
                )

                predicates = np.full((1, len(keys), 3), -1, dtype=np.int32)
                orders = np.empty((1, len(keys)), dtype=np.int32)
                windows = np.empty((1, len(keys)), dtype=np.int64)
                (
                    _,
                    _,
                    _,
                    completion_index,
                    completion_source_token,
                ) = optimizer._implicit_completion_batch(
                    tuple(block_antecedent for block_antecedent, _ in keys)
                )
                for block_index, (block_antecedent, block_window) in enumerate(keys):
                    # Provenance-aware pricing stores each exact completion
                    # stream as one resident source.  The objective must use
                    # the same packed source identity as the moment pass.
                    predicates[0, block_index, 0] = completion_index[block_antecedent]
                    orders[0, block_index] = 1
                    windows[0, block_index] = block_window * data.ticks_per_unit
                coefficient = np.linspace(
                    -0.03,
                    0.04,
                    len(keys) * config.knot_count,
                    dtype=np.float64,
                )[None, :]
                groups, current_x = optimizer._implicit_current_groups(current)
                group_eta = current_x @ current.fit.coefficients
                source_offsets, source_times = optimizer._implicit_sources()
                active_offsets, active_entities = (
                    optimizer._implicit_candidate_entity_index(
                        specifications, source_offsets, source_times
                    )
                )
                observed = implicit_poisson_objective_batch(
                    predicates,
                    orders,
                    windows,
                    np.asarray([len(keys)], dtype=np.int32),
                    active_offsets,
                    active_entities,
                    coefficient,
                    source_token=completion_source_token,
                    derivative_token=optimizer._compact_poisson_derivatives(current)[2],
                    entity_count=len(context.entity_codes),
                    current_groups=len(current_x),
                    knot_count=config.knot_count,
                    lag=config.impact_lag * data.ticks_per_unit,
                    maximum_entity_rows=int(np.max(context.ends - context.starts + 1)),
                    device="cuda:0",
                )
                self.assertIsNotNone(observed)
                effect = np.zeros(context.n_grid, dtype=np.float64)
                for block_index, block_response in enumerate(blocks):
                    left = block_index * config.knot_count
                    effect[block_response.rows] += (
                        block_response.values
                        @ coefficient[0, left : left + config.knot_count]
                    )
                eta = group_eta[groups]
                expected_delta = (
                    optimizer.engine.tick_exposure
                    * np.sum(np.exp(eta + effect) - np.exp(eta))
                    - context.target_counts @ effect[context.target_rows]
                )
                np.testing.assert_allclose(
                    observed[0], expected_delta, rtol=2e-11, atol=2e-11
                )
            finally:
                optimizer.close()

    @unittest.skipUnless(cuda_available(), "CUDA implicit pricing is unavailable")
    def test_fused_directional_footprint_bound_contains_exact_children(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=256 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=("cuda:0",),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(context, config)
            try:
                empty = optimizer.records[EMPTY_SUPPORT]
                parent = optimizer.fit(
                    Support.of((RuleIdentity((0,), 0, 1),)),
                    empty,
                    device="cuda:0",
                )
                self.assertTrue(parent.fit.converged, parent.fit.message)
                identities = tuple(
                    rule
                    for rule in optimizer.dictionary
                    if rule.pattern_key not in parent.support.patterns
                )[:8]
                uppers = optimizer._fused_localized_identity_upper_scores(
                    parent, identities
                )
                self.assertTrue(any(np.isfinite(tuple(uppers.values()))))
                checked = 0
                for rule in identities:
                    child = optimizer.fit(parent.support.add(rule), parent)
                    if not child.fit.converged or not np.isfinite(uppers[rule]):
                        continue
                    slack = 1.0e-8 * max(1.0, abs(child.score))
                    self.assertGreaterEqual(uppers[rule] + slack, child.score)
                    checked += 1
                self.assertGreater(checked, 0)
            finally:
                optimizer.close()

    @unittest.skipUnless(cuda_available(), "CUDA implicit pricing is unavailable")
    def test_compact_cloglog_completion_moments_match_sparse_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(
                Path(directory) / "data",
                120,
                likelihood="first_event_cloglog",
            )
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=100,
                cache_bytes=256 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=("cuda:0",),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(context, config)
            try:
                current = optimizer.fit(
                    Support.of((RuleIdentity((0,), 0, 1),)),
                    device="cuda:0",
                )
                antecedent = (0, 1)
                window = 2
                trial = current.support.add(RuleIdentity(antecedent, window, 1))
                additions = tuple(
                    sorted(set(hierarchy_closure(trial)) - set(current.matrix.closure))
                )
                keys = tuple(
                    (term.antecedent, int(term.window)) for term in additions
                ) + ((antecedent, window),)
                specifications = ((antecedent, window, additions, keys),)
                implicit = optimizer._implicit_hierarchy_moments(
                    current,
                    specifications,
                    None,
                    None,
                    device="cuda:0",
                )
                self.assertIsNotNone(implicit)
                assert implicit is not None
                blocks = tuple(
                    optimizer.engine.block(context, term.antecedent, term.window)
                    for term in additions
                ) + (optimizer.engine.block(context, antecedent, window),)
                reference = optimizer._joint_hierarchy_moments(
                    current, blocks, device="cpu"
                )
                for actual, expected in zip(
                    implicit,
                    (
                        reference[0][None, :],
                        reference[1][None, :, :],
                        reference[2][None, :, :],
                    ),
                    strict=True,
                ):
                    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
                compact = optimizer._compact_cloglog_derivatives(current)
                self.assertIsNotNone(compact)
                assert compact is not None
                _, current_x = optimizer._implicit_current_groups(current)
                self.assertEqual(compact[0].shape, (len(current_x),))
                self.assertEqual(compact[2].shape, (len(current_x),))
                self.assertEqual(compact[3].shape, (len(current_x),))
                self.assertEqual(compact[1].shape, (context.n_grid,))

                (
                    completion_offsets,
                    completion_times,
                    completion_spans,
                    completion_index,
                    source_token,
                ) = optimizer._implicit_completion_batch(tuple(key[0] for key in keys))
                predicates = np.full((1, len(keys), 3), -1, dtype=np.int32)
                orders = np.ones((1, len(keys)), dtype=np.int32)
                windows = np.empty((1, len(keys)), dtype=np.int64)
                for block_index, (block_antecedent, block_window) in enumerate(keys):
                    predicates[0, block_index, 0] = completion_index[block_antecedent]
                    windows[0, block_index] = block_window * data.ticks_per_unit
                source_offsets, source_times = optimizer._implicit_sources()
                active_offsets, active_entities = (
                    optimizer._implicit_candidate_entity_index(
                        specifications, source_offsets, source_times
                    )
                )
                groups, current_x = optimizer._implicit_current_groups(current)
                coefficient = np.linspace(
                    -0.03,
                    0.04,
                    len(keys) * config.knot_count,
                    dtype=np.float64,
                )[None, :]
                uploaded = implicit_moments_batch(
                    completion_offsets,
                    completion_times,
                    completion_spans,
                    context.starts,
                    context.ends,
                    context.offsets,
                    optimizer.engine.basis,
                    predicates,
                    orders,
                    windows,
                    np.asarray([len(keys)], dtype=np.int32),
                    active_offsets,
                    active_entities,
                    compact[0],
                    compact[0],
                    groups,
                    current_x,
                    source_token=source_token,
                    derivative_token=compact[4],
                    device="cuda:0",
                    compact_poisson_events=compact[1],
                    compact_cloglog_event_deltas=(compact[2], compact[3]),
                    completion_mode=True,
                    current_columns=np.asarray([0], dtype=np.int32),
                )
                self.assertIsNotNone(uploaded)
                group_eta = current_x @ current.fit.coefficients
                observed = implicit_objective_batch(
                    predicates,
                    orders,
                    windows,
                    np.asarray([len(keys)], dtype=np.int32),
                    active_offsets,
                    active_entities,
                    coefficient,
                    group_eta,
                    likelihood="first_event_cloglog",
                    source_token=source_token,
                    derivative_token=compact[4],
                    entity_count=len(context.entity_codes),
                    current_groups=len(current_x),
                    knot_count=config.knot_count,
                    lag=config.impact_lag * data.ticks_per_unit,
                    maximum_entity_rows=int(np.max(context.ends - context.starts + 1)),
                    device="cuda:0",
                )
                self.assertIsNotNone(observed)
                effect = np.zeros(context.n_grid, dtype=np.float64)
                for block_index, block_response in enumerate(blocks):
                    left = block_index * config.knot_count
                    effect[block_response.rows] += (
                        block_response.values
                        @ coefficient[0, left : left + config.knot_count]
                    )
                eta = group_eta[groups]
                event = np.zeros(context.n_grid, dtype=np.float64)
                event[context.target_rows] = context.target_counts
                noevent = 1.0 - event
                expected = np.sum(
                    loss_value_rows(
                        eta + effect,
                        likelihood="first_event_cloglog",
                        exposure_weight=np.ones(context.n_grid),
                        noevent_weight=noevent,
                        event_weight=event,
                    )
                    - loss_value_rows(
                        eta,
                        likelihood="first_event_cloglog",
                        exposure_weight=np.ones(context.n_grid),
                        noevent_weight=noevent,
                        event_weight=event,
                    )
                )
                np.testing.assert_allclose(
                    observed[0], expected, rtol=2e-11, atol=2e-11
                )
            finally:
                optimizer.close()

    def test_persistent_completion_store_seeds_exact_response_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90)
            fit_codes = np.arange(data.n_entities, dtype=np.int32)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=100,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            cache = root / "completion-cache"
            cache.mkdir()
            antecedent = (0, 1)

            first = SupportOptimizer(Context.make(data, fit_codes), config)
            first._persistent_completion_dir = cache
            expected = tuple(
                np.asarray(array).copy()
                for array in first._completion_for_antecedent(antecedent)
            )
            first.close()

            second = SupportOptimizer(Context.make(data, fit_codes), config)
            second._persistent_completion_dir = cache
            try:
                with patch.object(
                    second.engine,
                    "_compute_completions",
                    side_effect=AssertionError("persistent completion was rebuilt"),
                ):
                    restored = second._completion_for_antecedent(antecedent)
                    exact_fit_view = second.engine.completions(
                        second.context, antecedent
                    )
                for actual, wanted in zip(restored, expected, strict=True):
                    np.testing.assert_array_equal(actual, wanted)
                for actual, wanted in zip(exact_fit_view, expected, strict=True):
                    np.testing.assert_array_equal(actual, wanted)
                self.assertEqual(second.diagnostics.completion_batch_persistent_hits, 1)
            finally:
                second.close()

    def test_completion_batch_builds_only_requested_antecedents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=3,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=100,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(context, config)
            singleton = next(item for item in optimizer.skeletons if len(item) == 1)
            pair = next(item for item in optimizer.skeletons if len(item) == 2)
            requested = (singleton, pair)
            try:
                offsets, times, spans, index, _ = optimizer._implicit_completion_batch(
                    requested
                )
                self.assertEqual(set(index), set(requested))
                self.assertEqual(offsets.shape[0], len(requested))
                self.assertEqual(times.shape, spans.shape)
                self.assertEqual(optimizer.diagnostics.completion_batch_builds, 1)
                self.assertEqual(
                    optimizer.diagnostics.completion_batch_antecedents,
                    len(requested),
                )
                self.assertLess(
                    optimizer.diagnostics.completion_batch_antecedents,
                    len(optimizer.skeletons),
                )
            finally:
                optimizer.close()

    def test_global_completion_pack_is_an_exact_resident_superset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=3,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=100,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(context, config)
            singleton = next(item for item in optimizer.skeletons if len(item) == 1)
            pair = next(item for item in optimizer.skeletons if len(item) == 2)
            requested = (singleton, pair)
            try:
                local = optimizer._implicit_completion_batch(
                    requested, _allow_global=False
                )
                with optimizer._state_lock:
                    optimizer._completion_pack_cache.clear()
                    optimizer._completion_pack_cache_bytes = 0
                persistent = root / "completion-cache"
                persistent.mkdir()
                optimizer._persistent_completion_dir = persistent
                global_pack = optimizer._implicit_completion_batch(requested)
                self.assertEqual(set(global_pack[3]), set(optimizer.skeletons))
                other = next(
                    item for item in optimizer.skeletons if item not in requested
                )
                self.assertIs(
                    global_pack,
                    optimizer._implicit_completion_batch((other,)),
                )
                for antecedent in requested:
                    local_index = local[3][antecedent]
                    global_index = global_pack[3][antecedent]
                    local_offsets = local[0][local_index]
                    global_offsets = global_pack[0][global_index]
                    local_start, local_end = (
                        int(local_offsets[0]),
                        int(local_offsets[-1]),
                    )
                    global_start, global_end = (
                        int(global_offsets[0]),
                        int(global_offsets[-1]),
                    )
                    np.testing.assert_array_equal(
                        local_offsets - local_start,
                        global_offsets - global_start,
                    )
                    np.testing.assert_array_equal(
                        local[1][local_start:local_end],
                        global_pack[1][global_start:global_end],
                    )
                    np.testing.assert_array_equal(
                        local[2][local_start:local_end],
                        global_pack[2][global_start:global_end],
                    )
            finally:
                optimizer.close()

    def test_case_cohort_context_uses_entity_level_ht_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes = np.arange(90, dtype=np.int32)
            config = RunConfig(
                dataset=str(data.root),
                discovery_sampling="case_cohort_ipw",
                discovery_noncase_fraction=0.25,
                discovery_sampling_seed=7,
                pricing_devices=("cpu",),
                cache_bytes=1024**2,
            )
            context, metadata = _discovery_context(data, fit_codes, config)
            target_entities = set(data.target_entities.tolist())
            sampled = context.entity_codes.tolist()
            self.assertTrue(target_entities.issubset(sampled))
            self.assertEqual(context.population_entities, 90)
            self.assertEqual(metadata["unit"], "entity_complete_history")
            for code, weight in zip(
                context.entity_codes, context.entity_weights, strict=True
            ):
                if int(code) in target_entities:
                    self.assertEqual(weight, 1.0)
                else:
                    self.assertAlmostEqual(
                        weight,
                        1.0 / float(metadata["noncase_inclusion_probability"]),
                    )
            lengths = context.ends - context.starts + 1
            expected_exposure = float(context.entity_weights @ lengths)
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=1024**2)
            matrix = engine.model_matrix(context, Support(()))
            self.assertAlmostEqual(
                float(matrix.exposure_weight.sum()), expected_exposure
            )
            target_local, _ = context.rows_to_entity_time(context.target_rows)
            expected_events = float(
                np.sum(
                    data.target_multiplicity[
                        np.isin(data.target_entities, context.entity_codes)
                    ]
                    * context.entity_weights[target_local]
                )
            )
            self.assertAlmostEqual(float(matrix.event_weight.sum()), expected_events)

    def test_support_key_roundtrip(self) -> None:
        support = Support.of(
            (
                RuleIdentity((0,), 0, 1),
                RuleIdentity((1, 2), 3, -1),
            )
        )
        self.assertEqual(support_from_key(support_key(support)), support)
        self.assertEqual(support_from_key("empty"), Support(()))

    def test_mdl_compaction_defers_symmetric_nested_representative(self) -> None:
        singleton = Support.of((RuleIdentity((0,), 0, 1),))
        pair = Support.of((RuleIdentity((0,), 0, 1), RuleIdentity((1,), 0, 1)))

        def record(support: Support, score: float) -> SupportRecord:
            return SupportRecord(support, None, None, 0.0, score)  # type: ignore[arg-type]

        kept, dominated = _mdl_nondominated_records(
            (record(singleton, 10.0), record(pair, 9.0)), tolerance=1.0e-8
        )
        self.assertEqual(tuple(item.support for item in kept), (singleton,))
        self.assertEqual(dominated, 1)
        kept, dominated = _mdl_nondominated_records(
            (record(singleton, 10.0), record(pair, 11.0)), tolerance=1.0e-8
        )
        self.assertEqual({item.support for item in kept}, {singleton, pair})
        self.assertEqual(dominated, 0)
        kept, dominated = _nested_mdl_representatives(
            (record(singleton, 11.0), record(pair, 11.0)), tolerance=1.0e-8
        )
        self.assertEqual(tuple(item.support for item in kept), (singleton,))
        self.assertEqual(dominated, 1)
        smaller = SupportRecord(singleton, None, None, 0.0, 10.0, rule_score=100.0)  # type: ignore[arg-type]
        larger = SupportRecord(pair, None, None, 0.0, 11.0, rule_score=1.0)  # type: ignore[arg-type]
        kept, dominated = _nested_mdl_representatives(
            (smaller, larger), tolerance=1.0e-8
        )
        self.assertEqual(tuple(item.support for item in kept), (pair,))
        self.assertEqual(dominated, 1)
        kept, dominated = _nested_mdl_representatives(
            (record(singleton, 10.0), record(pair, 11.0)), tolerance=1.0e-8
        )
        self.assertEqual(tuple(item.support for item in kept), (pair,))
        self.assertEqual(dominated, 1)

        kept, dominated = _nested_mdl_representatives(
            (record(singleton, 10.0), record(pair, 11.0)),
            tolerance=1.0e-8,
            protected_supports=frozenset((singleton,)),
        )
        self.assertEqual({item.support for item in kept}, {singleton, pair})
        self.assertEqual(dominated, 0)

    def test_positive_deletion_minimal_family_preserves_nested_mechanisms(self) -> None:
        a = RuleIdentity((0,), 0, 1)
        b = RuleIdentity((1,), 0, 1)
        c = RuleIdentity((2,), 0, 1)

        def record(rules: tuple[RuleIdentity, ...], score: float) -> SupportRecord:
            support = Support.of(rules)
            fit = FitResult(
                np.zeros(1, dtype=np.float64),
                0.0,
                True,
                1,
                0.0,
                1,
                False,
                "test",
            )
            return SupportRecord(support, None, fit, 0.0, score, rule_score=score)  # type: ignore[arg-type]

        lattice = (
            record((a,), 2.0),
            record((b,), 2.0),
            record((c,), 2.0),
            record((a, b), 5.0),
            record((a, c), 1.0),
            record((b, c), 1.0),
            record((a, b, c), 6.0),
        )
        kept = _positive_deletion_minimal_records(lattice, tolerance=1.0e-8)
        self.assertEqual(
            {item.support for item in kept},
            {
                Support.of((a,)),
                Support.of((b,)),
                Support.of((c,)),
                Support.of((a, b)),
                Support.of((a, b, c)),
            },
        )

    def test_total_rule_representation_quotient_uses_common_mdl(self) -> None:
        additive = Support.of((RuleIdentity((0,), 0, 1), RuleIdentity((1,), 0, 1)))
        total_pair = Support.of((RuleIdentity((0, 1), 3, -1),))
        other = Support.of((RuleIdentity((2,), 0, 1),))

        def record(support: Support, score: float) -> SupportRecord:
            return SupportRecord(support, None, None, 0.0, score)  # type: ignore[arg-type]

        kept, removed = _mechanism_mdl_representatives(
            (
                record(additive, 8.0),
                record(total_pair, 9.0),
                record(other, 7.0),
            ),
            tolerance=1.0e-8,
        )
        self.assertEqual({item.support for item in kept}, {total_pair, other})
        self.assertEqual(removed, 1)

        kept, removed = _mechanism_mdl_representatives(
            (
                record(additive, 9.0),
                record(total_pair, 8.0),
                record(other, 7.0),
            ),
            tolerance=1.0e-8,
            protected_supports=frozenset((total_pair,)),
        )
        self.assertEqual({item.support for item in kept}, {additive, total_pair, other})
        self.assertEqual(removed, 0)

    def test_signed_exposure_controls_are_fixed_nuisance_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            entities = pd.DataFrame(
                {
                    "entity_id": ["e0", "e1"],
                    "start_time": [0, 0],
                    "end_time": [3, 3],
                    "baseline_origin": [0, 0],
                    "split_group": [0, 0],
                }
            )
            events = pd.DataFrame(
                [(0, 1, 0), (0, 1, 1), (1, 1, 2)],
                columns=["entity_code", "time", "predicate_code"],
            )
            targets = pd.DataFrame(
                [(0, 2, 1)], columns=["entity_code", "time", "multiplicity"]
            )
            write_dataset(
                root,
                entities=entities,
                events=events,
                targets=targets,
                predicate_names=("A", "exposure_up", "exposure_down"),
                predicate_roles=(
                    "reported",
                    "exposure_increase_control",
                    "exposure_decrease_control",
                ),
                likelihood="first_event_cloglog",
                time_unit="month",
                adverse_event_name="synthetic adverse event",
                f0_contract={
                    "dynamic_predicates": True,
                    "outcome_blind_predicate_construction": True,
                    "direct_target_proxy_excluded_from_reported_dictionary": True,
                    "strict_future_effect_required": True,
                    "atomic_predicates": True,
                    "primitive_event_provenance": True,
                },
                provenance={"generator": "test"},
            )
            data = Dataset.load(root)
            self.assertEqual(data.baseline_control_signs, (1, -1))
            context = Context.make(data, np.arange(2, dtype=np.int32))
            engine = ResponseEngine(data, lag=2, knot_count=2, cache_bytes=1024**2)
            matrix = engine.model_matrix(context, Support(()))
            increase = matrix.x[:, 1:3]
            decrease = matrix.x[:, 3:5]
            self.assertGreater(float(increase.max(initial=0.0)), 0.0)
            self.assertLess(float(decrease.min(initial=0.0)), 0.0)

    def test_total_mdl_bound_is_only_an_admissibility_gate_for_q(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer.config = SimpleNamespace(search_tolerance=1.0e-8)
        optimizer.saturated_upper_score = lambda _support: 5.0
        current = SupportRecord(
            Support.of((RuleIdentity((0,), 0, 1),)),
            None,
            None,
            0.0,
            10.0,
            rule_score=1.0,
            rule_score_upper=1.0,
        )  # type: ignore[arg-type]
        candidate = RuleIdentity((1,), 0, 1)
        self.assertEqual(
            optimizer._structural_upper_survivors(
                current,
                (candidate,),
                threshold=0.0,
            ),
            (candidate,),
        )

    def test_compact_move_uses_exact_q_drop_before_add(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer._working_antecedents = set()
        current = object()
        dropped = object()
        optimizer._best_exact_rule_objective_drop = lambda _, **__: dropped
        optimizer._best_conditional_drop = lambda *_args, **_kwargs: self.fail(
            "total-MDL Drop must not repeat the exact Q Drop audit"
        )
        add_called = False

        def addition(*args: object, **kwargs: object) -> object:
            nonlocal add_called
            add_called = True
            return object()

        optimizer._best_conditional_addition = addition
        decision = optimizer._best_profiled_decision(current)  # type: ignore[arg-type]
        self.assertIs(decision.record, dropped)
        self.assertFalse(decision.drop_stable)
        self.assertFalse(add_called)
        self.assertEqual(optimizer.diagnostics.backward_reduction_moves, 1)

    def test_compact_move_audits_identity_before_add(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer._working_antecedents = set()
        current = object()
        replaced = object()
        optimizer._best_exact_rule_objective_drop = lambda _, **__: None
        optimizer._best_conditional_drop = lambda _, **__: None
        optimizer._best_conditional_identity_change = lambda _, **__: replaced
        add_called = False

        def addition(*args: object, **kwargs: object) -> object:
            nonlocal add_called
            add_called = True
            return object()

        optimizer._first_conditional_addition = addition
        decision = optimizer._best_profiled_decision(current)  # type: ignore[arg-type]
        self.assertIs(decision.record, replaced)
        self.assertTrue(decision.drop_stable)
        self.assertFalse(add_called)

    def test_fast_route_takes_interleaved_drop_before_add(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        current = SimpleNamespace(fit=SimpleNamespace(converged=False))
        dropped = object()
        optimizer._best_interleaved_fast_drop = lambda *_args, **_kwargs: dropped
        optimizer._first_conditional_addition = lambda *_args, **_kwargs: self.fail(
            "an improving intermediate Drop must precede Add pricing"
        )
        decision = optimizer._best_fast_route_decision(current)  # type: ignore[arg-type]
        self.assertIs(decision.record, dropped)
        self.assertFalse(decision.drop_stable)

    def test_exact_drop_uses_cached_score_before_matrix_projection(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer.config = SimpleNamespace(
            search_tolerance=1.0e-8,
            pricing_devices=(),
        )
        first = RuleIdentity((0,), 0, 1)
        second = RuleIdentity((1,), 0, 1)
        current_support = Support.of((first, second))
        cached_support = current_support.drop(second)
        coefficients = np.zeros(1, dtype=np.float64)
        cached_fit = FitResult(
            coefficients,
            1.0,
            True,
            1,
            0.0,
            1,
            False,
            "cached exact fit",
        )
        optimizer._stored_records = {
            cached_support: _StoredRecord(cached_fit, 1.0, 5.0)
        }
        optimizer._conditional_parent_forbidden = set()
        optimizer._project_factorized_support = lambda *_: self.fail(
            "non-improving cached Drop must not materialize a matrix"
        )
        current = SimpleNamespace(
            support=current_support,
            score=10.0,
        )
        result = optimizer._best_exact_rule_objective_drop(
            current,  # type: ignore[arg-type]
            protected_antecedents=frozenset({first.antecedent}),
        )
        self.assertIsNone(result)
        self.assertEqual(optimizer.diagnostics.cached_drop_score_audits, 1)
        self.assertEqual(optimizer.diagnostics.cached_drop_matrix_builds_avoided, 1)

    def test_fast_route_uses_exact_composite_add_contract(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._working_antecedents = {(0,), (1,)}
        optimizer.skeletons = {(0,): object(), (1,): object(), (0, 1): object()}
        optimizer._exact_forward_transitions = {}
        optimizer._conditional_parent_forbidden = set()
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        rule = RuleIdentity((0,), 0, 1)
        candidate = SimpleNamespace(
            fit=SimpleNamespace(converged=True),
            support=Support.of((rule,)),
        )
        observed: dict[str, object] = {}

        def addition(*_: object, **kwargs: object) -> object:
            observed.update(kwargs)
            return candidate

        optimizer._best_composite_fast_addition = addition
        optimizer._best_terminal_cleanup_decision = lambda *_args, **_kwargs: (
            SimpleNamespace(record=None)
        )
        optimizer._attach_rule_score = lambda value: value
        current = SimpleNamespace(
            support=EMPTY_SUPPORT,
            fit=SimpleNamespace(converged=True),
        )
        decision = optimizer._best_fast_route_decision(
            current  # type: ignore[arg-type]
        )
        self.assertIs(decision.record, candidate)
        self.assertFalse(decision.drop_stable)
        self.assertEqual(observed["antecedents"], set(optimizer.skeletons))
        self.assertNotIn("exact_normal_form", observed)

    def test_composite_route_prices_conditional_rule_before_standalone_atom(
        self,
    ) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._force_exact_candidate_validation = False
        optimizer._terminal_add_audit_active = False
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer.config = SimpleNamespace(
            search_tolerance=1.0e-8,
            search_mode="atomic_rashomon_frontier",
            adaptive_gradient_racing=True,
            dependency_aware_mdl=False,
        )
        conditional_rule = RuleIdentity((1,), 0, 1)
        standalone_rule = RuleIdentity((2,), 0, 1)
        parent_rule = RuleIdentity((0,), 0, 1)
        current = SimpleNamespace(
            support=Support.of((parent_rule,)),
            fit=SimpleNamespace(converged=True),
            score=0.0,
        )
        optimizer._null_matched_parent_score = lambda *_: 0.0
        optimizer._inactive_identities = lambda *_args, **_kwargs: (
            conditional_rule,
            standalone_rule,
        )
        optimizer._structural_upper_survivors = lambda _c, identities, **_kw: identities
        optimizer._rank_profiled_identities = lambda *_: [
            (2.0, 2.0, standalone_rule),
            (1.0, 1.0, conditional_rule),
        ]
        optimizer._positive_atom_by_antecedent = {standalone_rule.pattern_key: object()}
        optimizer._separate_family_scores = lambda *_args, **_kwargs: {
            conditional_rule.pattern_key: 0.0,
            standalone_rule.pattern_key: 2.0,
        }
        optimizer._stored_records = {}
        optimizer._conditional_forbidden = set()
        optimizer._conditional_parent_forbidden = set()
        sentinel = object()
        observed: list[RuleIdentity] = []

        def validate(_current: object, viable: list[tuple], **_: object) -> object:
            observed.extend(item[2] for item in viable)
            return sentinel

        optimizer._first_validated_block_score_add = validate
        result = optimizer._best_conditional_addition(
            current,  # type: ignore[arg-type]
            antecedents={conditional_rule.pattern_key, standalone_rule.pattern_key},
            composite_cleanup=True,
        )
        self.assertIs(result, sentinel)
        self.assertEqual(observed, [conditional_rule, standalone_rule])

    def test_objective_roots_preserve_both_signs_without_extra_route_basins(
        self,
    ) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.config = SimpleNamespace(
            search_tolerance=1.0e-8,
            search_mode="atomic_rashomon_frontier",
        )
        optimizer.diagnostics = SearchDiagnostics()
        optimizer._working_antecedents = set()
        optimizer._history_family_profile_representatives = lambda values, **_kwargs: (
            list(values)
        )
        excitation = RuleIdentity((0,), 0, 1)
        inhibition = RuleIdentity((0,), 0, -1)
        selected = optimizer._objective_root_candidates(
            [
                (2.0, 1.0, excitation),
                (1.0, 0.5, inhibition),
            ]
        )
        self.assertEqual({item[2].sign for item in selected}, {-1, 1})
        self.assertEqual(
            set(optimizer._route_root_antecedents),
            {excitation.pattern_key},
        )

    def test_support_add_keeps_unrelated_positive_atom_separate(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer.config = SimpleNamespace(
            search_tolerance=1.0e-8,
            solver_tolerance=1.0e-7,
            effect_model="support_additive",
        )
        optimizer.objective = SimpleNamespace(n_entities=100)
        optimizer.baseline_nll = 100.0
        optimizer.engine = object()
        optimizer.context = object()
        optimizer.records = {EMPTY_SUPPORT: object()}
        optimizer._attach_rule_score = lambda value: value
        optimizer._support_contract_add_decisions = {}
        parent_rule = RuleIdentity((0,), 0, 1)
        added_rule = RuleIdentity((1,), 0, 1)
        parent = SimpleNamespace(
            support=Support.of((parent_rule,)),
            fit=SimpleNamespace(converged=True),
            matrix=SimpleNamespace(x=np.ones((1, 1))),
            penalty=10.0,
        )
        child = SimpleNamespace(
            support=Support.of((parent_rule, added_rule)),
            fit=SimpleNamespace(converged=True),
            matrix=SimpleNamespace(x=np.ones((1, 1))),
            penalty=20.0,
            score=22.0,
        )
        atom = SupportRecord(
            support=Support.of((added_rule,)),
            fit=FitResult(np.zeros(1), 80.0, True, 1, 0.0, 1, False, "test"),
            matrix=SimpleNamespace(x=np.ones((1, 1))),  # type: ignore[arg-type]
            penalty=8.0,
            score=12.0,
            rule_score=12.0,
        )
        optimizer.fit = lambda support, *_args, **_kwargs: (
            atom if support == atom.support else parent
        )
        optimizer._positive_atom_by_antecedent = {}
        optimizer._standalone_total_direction_aligned = lambda _record: True
        optimizer._separate_family_scores = lambda *_args, **_kwargs: {
            added_rule.pattern_key: 32.0
        }
        # 2*(100-70) - (10+8+2*log(100)) > child score 22, so the
        # exact separate family is preferred.
        with patch("crbstpp.search.freeze_support_record", side_effect=lambda x: x):
            self.assertFalse(
                optimizer._add_is_indecomposable(parent, child, added_rule)
            )
        self.assertEqual(optimizer.diagnostics.support_contract_add_audits, 1)
        self.assertEqual(optimizer.diagnostics.support_contract_add_rejections, 1)
        self.assertEqual(optimizer.diagnostics.on_demand_standalone_audits, 1)
        self.assertEqual(optimizer.diagnostics.on_demand_standalone_positive, 1)

    def test_support_add_allows_hierarchy_and_conditional_high_order(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer.config = SimpleNamespace(
            search_tolerance=1.0e-8,
            solver_tolerance=1.0e-7,
        )
        optimizer.objective = SimpleNamespace(n_entities=100)
        optimizer.baseline_nll = 100.0
        optimizer.engine = object()
        optimizer.context = object()
        optimizer.records = {EMPTY_SUPPORT: object()}
        optimizer._attach_rule_score = lambda value: value
        optimizer._support_contract_add_decisions = {}
        parent_rule = RuleIdentity((0,), 0, 1)
        parent = SimpleNamespace(
            support=Support.of((parent_rule,)),
            fit=SimpleNamespace(converged=True),
            matrix=SimpleNamespace(x=np.ones((1, 1))),
            penalty=10.0,
        )

        pair = RuleIdentity((0, 1), 1, -1)
        pair_atom = SimpleNamespace(
            support=Support.of((pair,)),
            discovery_score=5.0,
            fit=SimpleNamespace(converged=True),
            matrix=SimpleNamespace(x=np.ones((1, 1))),
            penalty=8.0,
        )
        optimizer._positive_atom_by_antecedent = {(0, 1): pair_atom}
        hierarchical_child = SimpleNamespace(
            support=parent.support.add(pair),
            fit=SimpleNamespace(converged=True),
            matrix=SimpleNamespace(x=np.ones((1, 1))),
            penalty=20.0,
            score=22.0,
        )
        optimizer.fit = lambda support, *_args, **_kwargs: (
            pair_atom if support == pair_atom.support else parent
        )
        # Here the exact separate family score is below 22, so the joint
        # A+AB representation is retained.
        with patch("crbstpp.search._sparse_profile", side_effect=(object(), object())):
            with patch(
                "crbstpp.search._fit_sparse_profiles",
                return_value=SimpleNamespace(converged=True, nll=95.0),
            ):
                self.assertTrue(
                    optimizer._add_is_indecomposable(parent, hierarchical_child, pair)
                )

        optimizer._support_contract_add_decisions.clear()
        conditional_triplet = RuleIdentity((0, 1, 2), 2, 1)
        optimizer._positive_atom_by_antecedent = {}
        conditional_child = SimpleNamespace(
            support=parent.support.add(conditional_triplet)
        )
        self.assertTrue(
            optimizer._add_is_indecomposable(
                parent, conditional_child, conditional_triplet
            )
        )

        optimizer._support_contract_add_decisions.clear()
        crossing_pair = RuleIdentity((1, 2), 1, 1)
        crossing_child = SimpleNamespace(support=parent.support.add(crossing_pair))
        # A high-order atom that is invalid standalone may become a genuine
        # suppressor/contextual rule conditional on the parent.
        self.assertTrue(
            optimizer._add_is_indecomposable(parent, crossing_child, crossing_pair)
        )

        optimizer._support_contract_add_decisions.clear()
        # Standalone-positive status alone no longer rejects the Add before
        # the exact joint-vs-separate family comparison.
        optimizer._positive_atom_by_antecedent = {
            crossing_pair.antecedent: SimpleNamespace(discovery_score=4.0)
        }
        self.assertTrue(
            optimizer._add_respects_support_contract(parent.support, crossing_pair)
        )

        optimizer._support_contract_add_decisions.clear()
        conditional_singleton = RuleIdentity((2,), 0, -1)
        singleton_child = SimpleNamespace(
            support=parent.support.add(conditional_singleton)
        )
        self.assertFalse(
            optimizer._add_is_indecomposable(
                parent, singleton_child, conditional_singleton
            )
        )

    def test_rejected_cached_forward_edge_cannot_replay_forever(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer.config = SimpleNamespace(
            max_rules_per_support=None,
            search_mode="fast_block_score",
            exact_workers=1,
            pricing_devices=(),
        )
        rule = RuleIdentity((0,), 0, 1)
        child = EMPTY_SUPPORT.add(rule)
        key = (EMPTY_SUPPORT, (), False)
        optimizer._exact_forward_transitions = OrderedDict(((key, child),))
        optimizer._exact_forward_transition_limit = 8
        optimizer._conditional_forbidden = set()
        optimizer._conditional_parent_forbidden = {(EMPTY_SUPPORT, child)}
        optimizer._profiled_by_antecedent = {}
        optimizer._baseline_identity_priority = {}
        current = SimpleNamespace(
            support=EMPTY_SUPPORT,
            fit=SimpleNamespace(converged=True),
        )

        self.assertIsNone(
            optimizer._first_conditional_addition(
                current,  # type: ignore[arg-type]
                antecedents=set(),
                frozen_identities=False,
            )
        )
        self.assertIsNone(optimizer._exact_forward_transitions[key])

    def test_natural_forced_fit_reuses_sparse_evaluation_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            rule = optimizer.dictionary[0]
            record = optimizer.fit(
                Support.of((rule,)),
                optimizer.records[EMPTY_SUPPORT],
            )
            self.assertTrue(record.fit.converged, record.fit.message)
            with patch.object(
                optimizer,
                "_project_nested_matrix",
                side_effect=AssertionError("natural cache must not project"),
            ):
                matrix, fit = optimizer.fit_fixed(
                    record.support,
                    hierarchy_closure(record.support),
                    source=record,
                )
            self.assertIs(matrix, record.matrix)
            self.assertIs(fit, record.fit)

    def test_entity_materiality_losses_restore_projected_entity_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(context, config)
            pair = optimizer.fit(
                Support.of((RuleIdentity((0, 1), 2, 1),)),
                optimizer.records[EMPTY_SUPPORT],
            )
            self.assertTrue(pair.fit.converged, pair.fit.message)
            projected, null_fit = optimizer.fit_fixed(
                EMPTY_SUPPORT,
                tuple(pair.matrix.closure),
                source=pair,
            )
            self.assertEqual(projected.dimension, projected.free_dimension)
            self.assertGreater(len(projected.active_rows), 0)

            restored = optimizer._entity_losses_for_model(projected, null_fit)
            optimizer._entity_loss_cache.clear()
            optimizer._entity_loss_cache_bytes = 0
            evaluated = optimizer.engine.model_matrix(
                context,
                EMPTY_SUPPORT,
                forced_closure=projected.closure,
            )
            expected = optimizer._entity_losses_for_model(evaluated, null_fit)
        # Sparse fixed-coefficient accumulation changes only the order of
        # IEEE-754 additions relative to dense grouped GEMV.
        np.testing.assert_allclose(
            restored,
            expected,
            rtol=4 * np.finfo(np.float64).eps,
            atol=4 * np.finfo(np.float64).eps,
        )

    def test_hidden_closure_is_not_admitted_by_total_state_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(context, config)
            pair_support = Support.of((RuleIdentity((0, 1), 2, 1),))
            fitted = optimizer.fit(
                pair_support,
                optimizer.records[EMPTY_SUPPORT],
            )
            self.assertTrue(fitted.fit.converged, fitted.fit.message)
            self.assertEqual(fitted.matrix.closure, ())
            with self.assertRaisesRegex(ValueError, "hidden closure"):
                optimizer.engine.model_matrix(
                    context,
                    pair_support,
                    forced_closure=(ClosureTerm((0,), 0),),
                )
            optimizer.close()

    def test_terminal_pointer_cache_has_hard_memory_bound(self) -> None:
        cache = _TerminalPointerCache(4_096)
        terminal = Support.of((RuleIdentity((99,), 0, 1),))
        sources = [Support.of((RuleIdentity((index,), 0, 1),)) for index in range(40)]
        for source in sources:
            cache.remember(source, terminal)
            self.assertLessEqual(cache.nbytes, cache.limit)
        self.assertGreater(cache.evictions, 0)
        self.assertIsNone(cache.get(sources[0]))
        self.assertEqual(cache.get(sources[-1]), terminal)

        oversized = Support.of(RuleIdentity((index,), 0, 1) for index in range(100))
        cache.remember(oversized, terminal)
        self.assertIsNone(cache.get(oversized))
        self.assertLessEqual(cache.nbytes, cache.limit)

    def test_roundtrip_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data")
            self.assertEqual(data.n_entities, 90)
            split = data.split((0.6, 0.2, 0.2), 111)
            self.assertEqual(sum(map(len, split)), 90)
            self.assertEqual(len(set(np.concatenate(split).tolist())), 90)

    def test_explicit_partition_overrides_fraction_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", explicit_partition=True)
            first = data.split((0.98, 0.01, 0.01), 1)
            second = data.split((0.1, 0.1, 0.8), 999)
            for code, (left, right) in enumerate(zip(first, second, strict=True)):
                np.testing.assert_array_equal(left, right)
                self.assertTrue(np.all(data.partitions[left] == code))

    def test_total_state_has_no_automatic_hierarchy_closure(self) -> None:
        a = RuleIdentity((0,), 0, 1)
        ab = RuleIdentity((0, 1), 2, -1)
        abc = RuleIdentity((0, 1, 2), 3, 1)
        self.assertEqual(hierarchy_closure(Support.of((ab,))), ())
        self.assertEqual(hierarchy_closure(Support.of((a, ab))), ())
        self.assertEqual(hierarchy_closure(Support.of((abc,))), ())
        ab_short = RuleIdentity((0, 1), 1, -1)
        self.assertEqual(hierarchy_closure(Support.of((ab_short, abc))), ())

    def test_completion_is_nonnegative_and_strictly_future(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 12)
            context = Context.make(data, np.arange(12))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=1024**2)
            block = engine.block(context, (0, 1), 2)
            self.assertTrue(np.all(block.values >= 0))
            local, times = context.rows_to_entity_time(block.rows)
            self.assertTrue(np.all(times >= 3))
            # A completion at B's t=2 cannot affect t=2 itself.
            self.assertFalse(np.any(times == 2))

    def test_paired_sign_matrix_reuse_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            context = Context.make(data, np.arange(60, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            positive = Support.of((RuleIdentity((0,), 0, 1),))
            negative = Support.of((RuleIdentity((0,), 0, -1),))
            positive_matrix = engine.model_matrix(context, positive)
            reused = SupportOptimizer._flip_single_rule_matrix(
                positive_matrix, negative
            )
            reference = engine.model_matrix(context, negative)
            np.testing.assert_array_equal(reused.x, reference.x)
            np.testing.assert_array_equal(
                reused.exposure_weight, reference.exposure_weight
            )
            np.testing.assert_array_equal(
                reused.noevent_weight, reference.noevent_weight
            )
            np.testing.assert_array_equal(reused.event_weight, reference.event_weight)
            np.testing.assert_array_equal(
                reused.active_design_groups, reference.active_design_groups
            )

    def test_shared_window_family_projection_matches_individual_exact_fits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(90, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            baseline = engine.model_matrix(context, EMPTY_SUPPORT)
            family = engine.standalone_window_family_matrix(
                context, (0, 1), (1, 2), baseline
            )
            baseline_fit = fit_model_matrix_continued(
                baseline,
                likelihood=data.likelihood,
                tolerance=1e-8,
                max_iter=120,
            )
            self.assertTrue(baseline_fit.converged, baseline_fit.message)
            for window_index, window in enumerate((1, 2)):
                for sign in (-1, 1):
                    columns = np.concatenate(
                        [
                            np.arange(baseline.dimension, dtype=np.int64),
                            np.arange(
                                baseline.dimension + 2 * window_index,
                                baseline.dimension + 2 * (window_index + 1),
                                dtype=np.int64,
                            ),
                        ]
                    )
                    scales = np.ones(len(columns), dtype=np.float64)
                    scales[baseline.dimension :] = sign
                    warm = np.zeros(len(columns), dtype=np.float64)
                    warm[: baseline.dimension] = baseline_fit.coefficients
                    projected = fit_projected_model_matrix(
                        family,
                        columns,
                        scales,
                        likelihood=data.likelihood,
                        free_dimension=baseline.free_dimension,
                        tolerance=1e-8,
                        max_iter=120,
                        warm_start=warm,
                    )
                    reference = engine.model_matrix(
                        context,
                        Support.of((RuleIdentity((0, 1), window, sign),)),
                    )
                    exact = fit_model_matrix_continued(
                        reference,
                        likelihood=data.likelihood,
                        tolerance=1e-8,
                        max_iter=120,
                        warm_start=warm,
                    )
                    self.assertEqual(projected.converged, exact.converged)
                    self.assertEqual(projected.recession, exact.recession)
                    if not exact.converged:
                        continue
                    self.assertAlmostEqual(projected.nll, exact.nll, places=9)
                    np.testing.assert_allclose(
                        projected.coefficients,
                        exact.coefficients,
                        rtol=0,
                        atol=2e-7,
                    )

    def test_hierarchical_window_family_projection_matches_exact_fits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(90, dtype=np.int32))
            engine = ResponseEngine(
                data,
                lag=3,
                knot_count=2,
                cache_bytes=8 * 1024**2,
                effect_model="additive_hierarchy",
            )
            baseline = engine.model_matrix(context, EMPTY_SUPPORT)
            baseline_fit = fit_model_matrix_continued(
                baseline,
                likelihood=data.likelihood,
                tolerance=1e-8,
                max_iter=120,
            )
            self.assertTrue(baseline_fit.converged, baseline_fit.message)
            supports = tuple(
                Support.of(
                    (
                        RuleIdentity(
                            (0, 1),
                            window,
                            sign,
                            hierarchical=True,
                        ),
                    )
                )
                for window in (1, 2)
                for sign in (-1, 1)
            )
            family, projections = engine.hierarchical_standalone_family_matrix(
                context, supports, baseline
            )
            for support in supports:
                selected = projections[support]
                columns = np.concatenate(
                    [
                        np.arange(baseline.dimension, dtype=np.int64),
                        *(
                            np.arange(
                                baseline.dimension + index * engine.knot_count,
                                baseline.dimension + (index + 1) * engine.knot_count,
                                dtype=np.int64,
                            )
                            for index, _ in selected
                        ),
                    ]
                )
                scales = np.ones(len(columns), dtype=np.float64)
                left = baseline.dimension
                for _, sign in selected:
                    right = left + engine.knot_count
                    scales[left:right] = sign
                    left = right
                warm = np.zeros(len(columns), dtype=np.float64)
                warm[: baseline.dimension] = baseline_fit.coefficients
                projected = fit_projected_model_matrix(
                    family,
                    columns,
                    scales,
                    likelihood=data.likelihood,
                    free_dimension=baseline.free_dimension,
                    tolerance=1e-8,
                    max_iter=120,
                    warm_start=warm,
                )
                reference = engine.model_matrix(context, support)
                exact = fit_model_matrix_continued(
                    reference,
                    likelihood=data.likelihood,
                    tolerance=1e-8,
                    max_iter=120,
                    warm_start=warm,
                )
                self.assertEqual(projected.converged, exact.converged)
                self.assertEqual(projected.recession, exact.recession)
                if not exact.converged:
                    continue
                self.assertEqual(reference.dimension, len(projected.coefficients))
                self.assertAlmostEqual(projected.nll, exact.nll, places=9)
                np.testing.assert_allclose(
                    projected.coefficients,
                    exact.coefficients,
                    rtol=0,
                    atol=2e-7,
                )

    def test_nested_add_window_family_preserves_existing_rule_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(90, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            current_support = Support.of((RuleIdentity((0,), 0, 1),))
            current = engine.model_matrix(context, current_support)
            current_fit = fit_model_matrix_continued(
                current,
                likelihood=data.likelihood,
                tolerance=1e-8,
                max_iter=120,
            )
            self.assertTrue(current_fit.converged, current_fit.message)
            family = engine.nested_add_window_family_matrix(
                context, (1,), (0,), current
            )
            self.assertEqual(len(family.rule_slices), 1)
            for sign in (-1, 1):
                added = RuleIdentity((1,), 0, sign)
                trial = current_support.add(added)
                canonical = engine.model_matrix(context, trial)
                columns = list(range(current.baseline_dimension))
                scales = [1.0] * current.baseline_dimension
                for rule in trial.rules:
                    if rule == added:
                        columns.extend(
                            range(
                                current.baseline_dimension,
                                current.baseline_dimension + 2,
                            )
                        )
                        scales.extend([float(sign), float(sign)])
                    else:
                        block = family.rule_slices[0]
                        columns.extend(range(block.start, block.stop))
                        scales.extend([1.0, 1.0])
                warm = np.zeros(canonical.dimension, dtype=np.float64)
                current_rule = trial.rules.index(current_support.rules[0])
                destination = canonical.rule_slices[current_rule]
                warm[: current.baseline_dimension] = current_fit.coefficients[
                    : current.baseline_dimension
                ]
                warm[destination] = current_fit.coefficients[current.rule_slices[0]]
                projected = fit_projected_model_matrix(
                    family,
                    np.asarray(columns, dtype=np.int64),
                    np.asarray(scales, dtype=np.float64),
                    likelihood=data.likelihood,
                    free_dimension=family.free_dimension,
                    tolerance=1e-8,
                    max_iter=120,
                    warm_start=warm,
                )
                exact = fit_model_matrix_continued(
                    canonical,
                    likelihood=data.likelihood,
                    tolerance=1e-8,
                    max_iter=120,
                    warm_start=warm,
                )
                self.assertEqual(projected.converged, exact.converged)
                self.assertEqual(projected.recession, exact.recession)
                if exact.converged:
                    self.assertAlmostEqual(projected.nll, exact.nll, places=9)
                    np.testing.assert_allclose(
                        projected.coefficients,
                        exact.coefficients,
                        rtol=0,
                        atol=2e-7,
                    )

    def test_sparse_exact_total_state_fit_matches_dense_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            context = Context.make(data, np.arange(120, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=16 * 1024**2)
            supports = (
                Support.of((RuleIdentity((0,), 0, 1),)),
                Support.of((RuleIdentity((0, 1), 2, -1),)),
                Support.of(
                    (
                        RuleIdentity((0,), 0, 1),
                        RuleIdentity((0, 1), 2, -1),
                    )
                ),
            )
            for support in supports:
                blocks = engine.total_state_rule_blocks(context, support)
                sparse = fit_sparse_grid_model(
                    context,
                    blocks,
                    tuple(rule.sign for rule in support.rules),
                    likelihood=data.likelihood,
                    tick_exposure=engine.tick_exposure,
                    tolerance=1e-8,
                    max_iter=120,
                )
                dense_matrix = engine.model_matrix(context, support)
                dense = fit_model_matrix_continued(
                    dense_matrix,
                    likelihood=data.likelihood,
                    tolerance=1e-8,
                    max_iter=120,
                )
                self.assertEqual(sparse.converged, dense.converged)
                self.assertEqual(sparse.recession, dense.recession)
                if not dense.converged:
                    continue
                self.assertAlmostEqual(sparse.nll, dense.nll, places=8)
                np.testing.assert_allclose(
                    sparse.coefficients,
                    dense.coefficients,
                    rtol=0,
                    atol=3e-7,
                )

    def test_exact_matrix_free_add_matches_dense_for_plain_and_nested_rules(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 240)
            context = Context.make(data, np.arange(160, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(context, config)
            try:
                empty = optimizer.records[EMPTY_SUPPORT]
                current = optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)), empty)
                self.assertTrue(current.fit.converged, current.fit.message)
                node = optimizer._conditional_node_state(current, device="cpu")
                group_by_row, _ = optimizer._implicit_current_groups(current)
                with ProjectedDesignEvaluator(
                    current.matrix,
                    likelihood=data.likelihood,
                    devices=("cpu",),
                ) as evaluator:
                    for rule in (
                        RuleIdentity((1,), 0, 1),
                        RuleIdentity((0, 1), 2, 1),
                    ):
                        trial = current.support.add(rule)
                        state = optimizer._matrix_free_state_splice(
                            current,
                            trial,
                            rule,
                            node,
                            current.fit.nll,
                            "cpu",
                            group_by_row,
                        )
                        self.assertIsNotNone(state)
                        assert state is not None
                        matrix_free = state.fit_exact(
                            source_evaluator=evaluator,
                            tolerance=1e-8,
                            max_iter=120,
                        )
                        dense_matrix = optimizer.engine.model_matrix(context, trial)
                        dense = fit_model_matrix_continued(
                            dense_matrix,
                            likelihood=data.likelihood,
                            tolerance=1e-8,
                            max_iter=120,
                            warm_start=optimizer.warm_start(current, dense_matrix),
                        )
                        self.assertEqual(matrix_free.converged, dense.converged)
                        self.assertEqual(matrix_free.recession, dense.recession)
                        if dense.converged:
                            self.assertAlmostEqual(matrix_free.nll, dense.nll, places=8)
                            np.testing.assert_allclose(
                                matrix_free.coefficients,
                                dense.coefficients,
                                rtol=0,
                                atol=5e-7,
                            )
                    # Opposite W/sign identities reuse exactly the same large
                    # touched-row geometry.  The algebraic sign transform must
                    # remain identical to rebuilding that identity afresh.
                    positive = RuleIdentity((1,), 0, 1)
                    negative = RuleIdentity((1,), 0, -1)
                    positive_state = optimizer._matrix_free_state_splice(
                        current,
                        current.support.add(positive),
                        positive,
                        node,
                        current.fit.nll,
                        "cpu",
                        group_by_row,
                    )
                    negative_state = optimizer._matrix_free_state_splice(
                        current,
                        current.support.add(negative),
                        negative,
                        node,
                        current.fit.nll,
                        "cpu",
                        group_by_row,
                    )
                    self.assertIsNotNone(positive_state)
                    self.assertIsNotNone(negative_state)
                    assert positive_state is not None
                    assert negative_state is not None
                    reused = positive_state.with_added_sign(-1)
                    np.testing.assert_array_equal(
                        reused.coordinate_signs,
                        negative_state.coordinate_signs,
                    )
                    np.testing.assert_allclose(
                        reused.gradient, negative_state.gradient, rtol=0, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        reused.hessian, negative_state.hessian, rtol=0, atol=1e-12
                    )
                    reused_fit = reused.fit_exact(
                        source_evaluator=evaluator,
                        tolerance=1e-8,
                        max_iter=120,
                    )
                    fresh_fit = negative_state.fit_exact(
                        source_evaluator=evaluator,
                        tolerance=1e-8,
                        max_iter=120,
                    )
                    self.assertEqual(reused_fit.converged, fresh_fit.converged)
                    self.assertAlmostEqual(reused_fit.nll, fresh_fit.nll, places=10)
                    np.testing.assert_allclose(
                        reused_fit.coefficients,
                        fresh_fit.coefficients,
                        rtol=0,
                        atol=1e-10,
                    )
            finally:
                optimizer.close()

    def test_matrix_free_gradient_bundle_safely_bounds_signed_high_order_adds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 240)
            context = Context.make(data, np.arange(160, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(context, config)
            try:
                empty = optimizer.records[EMPTY_SUPPORT]
                current = optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)), empty)
                node = optimizer._conditional_node_state(current, device="cpu")
                group_by_row, _ = optimizer._implicit_current_groups(current)
                with ProjectedDesignEvaluator(
                    current.matrix,
                    likelihood=data.likelihood,
                    devices=("cpu",),
                ) as evaluator:
                    checked: list[RuleIdentity] = []
                    for rule in (
                        RuleIdentity((1,), 0, 1),
                        RuleIdentity((1,), 0, -1),
                        RuleIdentity((0, 1), 2, 1),
                        RuleIdentity((0, 1), 2, -1),
                    ):
                        trial = current.support.add(rule)
                        state = optimizer._matrix_free_state_splice(
                            current,
                            trial,
                            rule,
                            node,
                            current.fit.nll,
                            "cpu",
                            group_by_row,
                        )
                        self.assertIsNotNone(state)
                        assert state is not None
                        dense_matrix = optimizer.engine.model_matrix(context, trial)
                        dense = fit_model_matrix_continued(
                            dense_matrix,
                            likelihood=data.likelihood,
                            tolerance=1e-8,
                            max_iter=120,
                            warm_start=optimizer.warm_start(current, dense_matrix),
                        )
                        if not dense.converged:
                            continue
                        bounded = state.fit_bounded(
                            source_evaluator=evaluator,
                            tolerance=1e-8,
                            max_iter=120,
                            nll_screen_threshold=dense.nll - 1.0e-5,
                        )
                        self.assertLessEqual(
                            bounded.nll_lower_bound,
                            dense.nll + 2.0e-7,
                        )
                        if bounded.certified_non_improving:
                            self.assertGreaterEqual(
                                bounded.nll_lower_bound,
                                dense.nll - 1.0e-5,
                            )
                        else:
                            # A bundle can remain dual-infeasible, especially
                            # when a high-order splice adds many closure
                            # coordinates.  Fail-open must retain the same
                            # exact optimum rather than force a screen.
                            self.assertTrue(bounded.fit.converged)
                            self.assertAlmostEqual(bounded.fit.nll, dense.nll, places=6)
                        checked.append(rule)
                    self.assertTrue(checked)
                    self.assertTrue(any(rule.sign > 0 for rule in checked))
                    self.assertTrue(any(rule.order > 1 for rule in checked))
            finally:
                optimizer.close()

    def test_nested_directional_bound_matches_naive_row_set_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(60, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(context, config)
            current = optimizer.fit(
                Support.of((RuleIdentity((0,), 0, 1),)),
                optimizer.records[EMPTY_SUPPORT],
            )
            self.assertTrue(current.fit.converged, current.fit.message)
            identities = tuple(
                rule
                for rule in optimizer.dictionary
                if rule.antecedent in {(1,), (0, 1)}
            )
            optimizer._safe_identity_survivors(current, identities, -np.inf)
            common_excitation = optimizer.engine.response_rows(context, (0,), 0)
            empty_rows = np.zeros(0, dtype=np.int64)
            for rule in identities:
                trial = current.support.add(rule)
                new_rows = optimizer.engine.response_rows(
                    context, rule.antecedent, rule.window
                )
                affected = np.union1d(common_excitation, new_rows)
                expected_localized = optimizer._localized_score_from_rows(
                    trial, affected
                )
                expected_directional = optimizer._directional_score_from_rows(
                    trial,
                    empty_rows,
                    (affected if rule.sign > 0 else common_excitation),
                    (new_rows if rule.sign < 0 else empty_rows),
                )
                observed_localized = optimizer._relaxed_upper_cache[
                    optimizer._unsigned_geometry_key(trial)
                ]
                observed_directional = optimizer._directional_upper_cache[trial]
                self.assertAlmostEqual(
                    observed_localized, expected_localized, places=10
                )
                self.assertAlmostEqual(
                    observed_directional, expected_directional, places=10
                )
            optimizer.close()

    def test_baseline_control_is_fixed_and_strictly_future(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            entities = pd.DataFrame(
                {
                    "entity_id": ["e0", "e1"],
                    "start_time": [0, 0],
                    "end_time": [3, 3],
                    "baseline_origin": [0, 0],
                    "split_group": [0, 0],
                }
            )
            events = pd.DataFrame(
                [(0, 1, 0), (0, 1, 1)],
                columns=["entity_code", "time", "predicate_code"],
            )
            targets = pd.DataFrame(
                [(0, 2, 1)],
                columns=["entity_code", "time", "multiplicity"],
            )
            write_dataset(
                root,
                entities=entities,
                events=events,
                targets=targets,
                predicate_names=("A", "prior_30dpd"),
                predicate_roles=("reported", "baseline_control"),
                likelihood="first_event_cloglog",
                time_unit="month",
                adverse_event_name="synthetic adverse event",
                f0_contract={
                    "dynamic_predicates": True,
                    "outcome_blind_predicate_construction": True,
                    "direct_target_proxy_excluded_from_reported_dictionary": True,
                    "strict_future_effect_required": True,
                    "atomic_predicates": True,
                    "primitive_event_provenance": True,
                },
                provenance={"generator": "test"},
            )
            data = Dataset.load(root)
            context = Context.make(data, np.arange(2, dtype=np.int32))
            engine = ResponseEngine(data, lag=2, knot_count=2, cache_bytes=1024**2)
            matrix = engine.model_matrix(context, Support(()))
            # One intercept plus one two-knot fixed control block.
            self.assertEqual(engine.baseline_dimension, 3)
            self.assertEqual(matrix.dimension, 3)
            self.assertEqual(matrix.free_dimension, 1)
            self.assertEqual(matrix.control_dimension, 2)
            control = engine.control_block(context, 1)
            _, times = context.rows_to_entity_time(control.rows)
            np.testing.assert_array_equal(times, np.asarray([2, 3]))
            self.assertNotIn(1, skeletons(data.n_reported_predicates, 1)[0])

    def test_feasible_branch_null_is_a_safe_exact_contribution_upper_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                search_mode="fast_block_score",
                solver_tolerance=1.0e-8,
                solver_max_iter=120,
                pricing_devices=(),
                exact_workers=1,
                pricing_workers=1,
                cache_bytes=8 * 1024**2,
                early_warning_horizon=3,
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            support = Support.of(
                (
                    RuleIdentity((0,), 0, 1),
                    RuleIdentity((1,), 0, 1),
                )
            )
            current = optimizer.fit(support, optimizer.records[EMPTY_SUPPORT])
            self.assertTrue(current.fit.converged, current.fit.message)
            full_code = optimizer.objective.reported_branch_penalty(
                support,
                len(support.rules) * config.knot_count,
            )
            for root in support.rules:
                dropped = hierarchy_branch_drop(support, root)
                null_closure = hierarchy_branch_null_closure(
                    current.matrix.closure,
                    dropped,
                    root,
                )
                projection = optimizer._factorized_drop_projection(
                    current,
                    dropped,
                    forced_closure=null_closure,
                )
                self.assertIsNotNone(projection)
                columns, scales = projection
                beta = current.fit.coefficients[columns] * scales
                beta[current.matrix.free_dimension :] = np.maximum(
                    beta[current.matrix.free_dimension :], 0.0
                )
                source_beta = np.zeros(current.matrix.dimension, dtype=np.float64)
                source_beta[columns] = scales * beta
                feasible_nll = optimizer._matrix_nll(current.matrix, source_beta)
                projected = optimizer._fit_projected_fixed_view(
                    dropped,
                    null_closure,
                    source=current,
                    device="cpu",
                )
                self.assertIsNotNone(projected)
                exact_null = projected[0]
                self.assertTrue(exact_null.converged, exact_null.message)
                drop_code = optimizer.objective.reported_branch_penalty(
                    dropped,
                    len(dropped.rules) * config.knot_count,
                )
                exact_net = 2.0 * (exact_null.nll - current.fit.nll) - (
                    full_code - drop_code
                )
                safe_upper = 2.0 * (feasible_nll - current.fit.nll) - (
                    full_code - drop_code
                )
                self.assertLessEqual(exact_net, safe_upper + 1.0e-10)

    def test_terminal_cleanup_safely_drops_balanced_redundant_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            n_entities = 1600
            entities = pd.DataFrame(
                {
                    "entity_id": [f"e{index:04d}" for index in range(n_entities)],
                    "start_time": np.zeros(n_entities, dtype=np.int64),
                    "end_time": np.full(n_entities, 5, dtype=np.int64),
                    "baseline_origin": np.zeros(n_entities, dtype=np.int64),
                    "split_group": np.zeros(n_entities, dtype=np.int64),
                }
            )
            events: list[tuple[int, int, int]] = []
            targets: list[tuple[int, int, int]] = []
            for entity in range(n_entities):
                cell = entity % 8
                replicate = entity // 8
                has_a = cell >= 4
                has_b = cell % 2 == 1
                has_c = (cell // 2) % 2 == 1
                if has_a:
                    events.append((entity, 1, 0))
                if has_b:
                    events.append((entity, 1, 1))
                if has_c:
                    events.append((entity, 1, 2))
                # The target rate depends on A and is exactly balanced over
                # both B and C.
                target = replicate % 10 < (8 if has_a else 1)
                if target:
                    targets.append((entity, 3, 1))
            write_dataset(
                root,
                entities=entities,
                events=pd.DataFrame(
                    events,
                    columns=["entity_code", "time", "predicate_code"],
                ),
                targets=pd.DataFrame(
                    targets,
                    columns=["entity_code", "time", "multiplicity"],
                ),
                predicate_names=("A", "balanced_B", "balanced_C"),
                likelihood="first_event_cloglog",
                time_unit="month",
                adverse_event_name="synthetic adverse event",
                f0_contract={
                    "dynamic_predicates": True,
                    "outcome_blind_predicate_construction": True,
                    "direct_target_proxy_excluded_from_reported_dictionary": True,
                    "strict_future_effect_required": True,
                    "atomic_predicates": True,
                    "primitive_event_provenance": True,
                },
                provenance={"generator": "balanced redundant rule test"},
            )
            data = Dataset.load(root)
            config = RunConfig(
                dataset=str(root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                search_mode="fast_block_score",
                solver_tolerance=1.0e-8,
                solver_max_iter=120,
                pricing_devices=(),
                exact_workers=1,
                pricing_workers=1,
                cache_bytes=8 * 1024**2,
                early_warning_horizon=3,
            )
            optimizer = SupportOptimizer(
                Context.make(data, np.arange(n_entities, dtype=np.int32)),
                config,
            )
            support = Support.of(
                (
                    RuleIdentity((0,), 0, 1),
                    RuleIdentity((1,), 0, 1),
                    RuleIdentity((2,), 0, 1),
                )
            )
            current = optimizer._attach_rule_score(
                optimizer.fit(support, optimizer.records[EMPTY_SUPPORT])
            )
            self.assertTrue(current.fit.converged, current.fit.message)
            dropped = optimizer._best_exact_nonpositive_branch_drop(
                current,
                protected_antecedents=frozenset(),
            )
            self.assertIsNotNone(dropped)
            self.assertEqual(
                dropped.support,
                Support.of((RuleIdentity((0,), 0, 1),)),
            )
            self.assertGreater(dropped.score, current.score)
            self.assertEqual(optimizer.diagnostics.terminal_branch_bundle_moves, 1)
            self.assertEqual(optimizer.diagnostics.terminal_branch_bundle_rules, 2)
            # Both redundant branches are removed by one guarded natural fit,
            # irrespective of whether the feasible bound or exact null
            # comparison supplied their individual certificates.
            self.assertGreaterEqual(
                optimizer.diagnostics.terminal_branch_safe_bound_drops
                + optimizer.diagnostics.terminal_branch_drop_audits,
                1,
            )
            self.assertIsNone(
                optimizer._best_exact_nonpositive_branch_drop(
                    dropped,
                    protected_antecedents=frozenset(),
                )
            )

    def test_cached_implicit_row_predictor_matches_sparse_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 3),
                search_mode="fast_block_score",
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            current = optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)))
            rows = np.arange(optimizer.context.n_grid, dtype=np.int64)
            expected = optimizer._predict_rows(current, rows)
            optimizer._implicit_current_groups(current)
            observed = optimizer._predict_rows(current, rows)
            np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


class LikelihoodSolverTests(unittest.TestCase):
    def test_streamed_mixture_matches_simultaneous_matrix_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            supports = (
                EMPTY_SUPPORT,
                Support.of((RuleIdentity((0,), 0, 1),)),
            )
            matrices = [engine.model_matrix(context, support) for support in supports]
            fits = [
                fit_model_matrix_continued(
                    matrix,
                    likelihood=data.likelihood,
                    tolerance=1.0e-8,
                    max_iter=120,
                    device="cpu",
                )
                for matrix in matrices
            ]
            simultaneous = _fixed_model_mixture(
                engine,
                context,
                matrices,
                fits,
                tolerance=1.0e-8,
            )
            profiles = [
                _sparse_profile(engine, context, matrix, fit)
                for matrix, fit in zip(matrices, fits, strict=True)
            ]
            streamed = _fit_sparse_profiles(
                engine,
                context,
                profiles,
                tolerance=1.0e-8,
            )
            self.assertTrue(simultaneous.converged)
            self.assertTrue(streamed.converged)
            self.assertAlmostEqual(streamed.nll, simultaneous.nll, places=10)
            np.testing.assert_allclose(
                streamed.weights,
                simultaneous.weights,
                rtol=0,
                atol=1.0e-9,
            )

    def test_compiled_pricing_gather_matches_exact_rows_and_lookup(self) -> None:
        source_rows = np.asarray([1, 4, 7, 11], dtype=np.int64)
        source_values = np.arange(12, dtype=np.float64).reshape(4, 3)
        query = np.asarray([0, 1, 3, 7, 8, 11], dtype=np.int64)
        expected = np.zeros((len(query), 3), dtype=np.float64)
        expected[[1, 3, 5]] = source_values[[0, 2, 3]]

        ordered = np.empty_like(expected)
        if fill_pricing_values(query, source_rows, source_values, ordered):
            np.testing.assert_array_equal(ordered, expected)

        lookup = np.full(12, -1, dtype=np.int32)
        lookup[source_rows] = np.arange(len(source_rows), dtype=np.int32)
        indexed = np.empty_like(expected)
        if fill_pricing_values(
            query, source_rows, source_values, indexed, lookup=lookup
        ):
            np.testing.assert_array_equal(indexed, expected)

    def test_response_row_lookup_is_exact_and_shared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data")
            context = Context.make(data, np.arange(60, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            rows, _ = engine.response_row_thresholds(context, (0, 1), 2)
            first = engine.response_row_lookup(context, (0, 1), 2)
            second = engine.response_row_lookup(context, (0, 1), 2)
            self.assertIsNotNone(first)
            self.assertIs(first, second)
            np.testing.assert_array_equal(
                first[rows], np.arange(len(rows), dtype=np.int32)
            )

    def test_quantile_windows_match_productive_completion_spans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(90, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            quantiles = (0.25, 0.50, 0.75, 1.0)
            entities, times, spans = engine.completions(context, (0, 1))
            productive = times < context.ends[entities]
            positive = np.sort(spans[productive & (spans > 0)])
            zero_count = int(np.count_nonzero(productive & (spans == 0)))
            expected = [0] if zero_count else []
            for quantile in quantiles:
                rank = max(1, int(np.ceil(quantile * len(positive))))
                expected.append(int(np.ceil(positive[rank - 1] / data.ticks_per_unit)))
            self.assertEqual(
                engine.quantile_windows(context, (0, 1), quantiles),
                tuple(sorted(set(expected))),
            )
            windows, provenance = engine.quantile_windows_with_provenance(
                context, (0, 1), quantiles
            )
            self.assertEqual(windows, tuple(sorted(set(expected))))
            self.assertEqual(set(windows), set(provenance))
            self.assertEqual(provenance.get(0), () if zero_count else None)
            self.assertEqual(
                sorted(
                    quantile
                    for window, labels in provenance.items()
                    if window > 0
                    for quantile in labels
                ),
                list(quantiles),
            )
            bounded = engine.quantile_windows(
                context,
                (0, 1),
                quantiles,
                maximum_window=0,
            )
            self.assertTrue(all(window == 0 for window in bounded))

    def test_quantile_window_dictionary_is_frozen_and_penalized_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2, 3),
                formation_window_mode="fit_quantile",
                formation_window_quantiles=(0.25, 0.5, 0.75, 1.0),
                solver_tolerance=1.0e-8,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                pair_windows = optimizer.window_dictionary[(0, 1)]
                self.assertTrue(pair_windows)
                self.assertEqual(
                    optimizer.objective.window_count((0, 1)),
                    len(pair_windows),
                )
                self.assertEqual(
                    set(optimizer.window_quantile_dictionary[(0, 1)]),
                    set(pair_windows),
                )
                fit_only = SupportOptimizer(
                    Context.make(data, fit_codes),
                    config,
                    fit_only=True,
                    window_dictionary=optimizer.window_dictionary,
                )
                try:
                    self.assertEqual(
                        fit_only.window_dictionary,
                        optimizer.window_dictionary,
                    )
                    self.assertEqual(
                        fit_only.objective.window_count((0, 1)),
                        len(pair_windows),
                    )
                finally:
                    fit_only.close()
            finally:
                optimizer.close()

    def test_incremental_window_pricing_matches_snapshot_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2, 3),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=128 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            reference_optimizer = SupportOptimizer(
                Context.make(data, fit_codes), config
            )
            reference_empty = reference_optimizer.records[Support(())]
            reference = reference_optimizer._price_hierarchy_skeleton(
                reference_empty,
                (0, 1),
                config.formation_windows,
                device="cpu",
                allow_incremental=False,
            )
            incremental_optimizer = SupportOptimizer(
                Context.make(data, fit_codes), config
            )
            incremental_empty = incremental_optimizer.records[Support(())]
            incremental = incremental_optimizer._price_hierarchy_incremental(
                incremental_empty,
                (0, 1),
                config.formation_windows,
                device="cpu",
            )
            self.assertIsNotNone(incremental)
            assert incremental is not None
            for window in config.formation_windows:
                np.testing.assert_allclose(
                    incremental[window][:4],
                    reference[window][:4],
                    rtol=2e-11,
                    atol=2e-11,
                )
                self.assertEqual(incremental[window][4:], reference[window][4:])
            self.assertEqual(
                incremental_optimizer.diagnostics.incremental_window_pricing_passes,
                1,
            )
            self.assertEqual(
                incremental_optimizer.diagnostics.incremental_window_pricing_fallbacks,
                0,
            )
            singleton = RuleIdentity((0,), 0, 1)
            nonempty = incremental_optimizer.fit(
                Support.of((singleton,)), incremental_empty
            )
            nonempty_reference = incremental_optimizer._price_hierarchy_skeleton(
                nonempty,
                (0, 1),
                config.formation_windows,
                device="cpu",
                allow_incremental=False,
            )
            nonempty_incremental = incremental_optimizer._price_hierarchy_incremental(
                nonempty,
                (0, 1),
                config.formation_windows,
                device="cpu",
            )
            self.assertIsNotNone(nonempty_incremental)
            assert nonempty_incremental is not None
            for window in config.formation_windows:
                np.testing.assert_allclose(
                    nonempty_incremental[window][:4],
                    nonempty_reference[window][:4],
                    rtol=2e-11,
                    atol=2e-11,
                )

    def test_factorized_add_matches_full_matrix_one_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            singleton = RuleIdentity((0,), 0, 1)
            current = optimizer.fit(Support.of((singleton,)), empty)
            candidates = (
                RuleIdentity((1,), 0, 1),
                RuleIdentity((0, 1), 1, -1),
            )
            for rule in candidates:
                optimizer._rank_profiled_identities(current, (rule,))
                factorized = optimizer._factorized_add_one_step(
                    current, rule, device="cpu"
                )
                if set(singleton.antecedent) < set(rule.antecedent):
                    # A higher-order total state masks the existing singleton;
                    # this is a sparse state splice, not a column-only Add.
                    self.assertIsNone(factorized)
                    full = optimizer._conditional_one_step(
                        current, current.support.add(rule), device="cpu"
                    )
                    self.assertTrue(np.isfinite(full.record.fit.nll))
                    effective = optimizer.engine.total_state_rule_blocks(
                        optimizer.context, full.record.support
                    )
                    self.assertEqual(
                        np.intersect1d(effective[0].rows, effective[1].rows).size,
                        0,
                    )
                    continue
                self.assertIsNotNone(factorized)
                assert factorized is not None
                deferred = optimizer._factorized_add_one_step(
                    current,
                    rule,
                    device="cpu",
                    finalize_derivatives=False,
                    record_diagnostics=False,
                )
                self.assertIsNotNone(deferred)
                assert deferred is not None
                self.assertIsNotNone(deferred.deferred_state)
                finalized = optimizer._finalize_factorized_add_state(
                    current,
                    deferred,
                    device="cpu",
                    record_diagnostics=False,
                )
                self.assertIsNone(finalized.deferred_state)
                np.testing.assert_allclose(
                    finalized.fit.coefficients,
                    factorized.fit.coefficients,
                    rtol=0,
                    atol=0,
                )
                self.assertAlmostEqual(finalized.fit.nll, factorized.fit.nll, places=12)
                self.assertAlmostEqual(
                    finalized.fit.projected_kkt,
                    factorized.fit.projected_kkt,
                    places=12,
                )
                self.assertEqual(finalized.fit.rank, factorized.fit.rank)
                self.assertAlmostEqual(
                    finalized.lower_score, factorized.lower_score, places=12
                )
                full = optimizer._conditional_one_step(
                    current, current.support.add(rule), device="cpu"
                )
                np.testing.assert_allclose(
                    factorized.fit.coefficients,
                    full.record.fit.coefficients,
                    rtol=0,
                    atol=1e-11,
                )
                self.assertAlmostEqual(
                    factorized.fit.nll, full.record.fit.nll, places=10
                )
                self.assertAlmostEqual(
                    factorized.fit.projected_kkt,
                    full.record.fit.projected_kkt,
                    places=9,
                )
                self.assertEqual(factorized.fit.rank, full.record.fit.rank)
                self.assertAlmostEqual(
                    factorized.lower_score, full.lower_score, places=9
                )
            second_singleton = RuleIdentity((1,), 0, 1)
            pair_support = Support.of((singleton, second_singleton))
            pair_record = optimizer.fit(pair_support, current)
            drop_trial = pair_support.drop(singleton)
            factorized_drop = optimizer._factorized_drop_one_step(
                pair_record, drop_trial, device="cpu"
            )
            self.assertIsNotNone(factorized_drop)
            assert factorized_drop is not None
            full_drop = optimizer._conditional_one_step(
                pair_record, drop_trial, device="cpu"
            )
            np.testing.assert_allclose(
                factorized_drop.fit.coefficients,
                full_drop.record.fit.coefficients,
                rtol=0,
                atol=1e-11,
            )
            self.assertAlmostEqual(
                factorized_drop.fit.nll, full_drop.record.fit.nll, places=10
            )
            self.assertAlmostEqual(
                factorized_drop.fit.projected_kkt,
                full_drop.record.fit.projected_kkt,
                places=9,
            )
            self.assertEqual(factorized_drop.fit.rank, full_drop.record.fit.rank)

            redundant_beta = pair_record.fit.coefficients.copy()
            for block in pair_record.matrix.rule_slices:
                redundant_beta[block] = 0.0
            redundant_nll = optimizer._matrix_nll(pair_record.matrix, redundant_beta)
            redundant_penalty = optimizer.objective.structural_penalty(pair_support)
            redundant = SupportRecord(
                support=pair_support,
                matrix=pair_record.matrix,
                fit=FitResult(
                    coefficients=redundant_beta,
                    nll=redundant_nll,
                    converged=False,
                    iterations=0,
                    projected_kkt=math.inf,
                    rank=pair_record.matrix.dimension,
                    recession=False,
                    message="test redundant route point",
                ),
                penalty=redundant_penalty,
                score=support_score(
                    baseline_nll=optimizer.baseline_nll,
                    fit_nll=redundant_nll,
                    penalty=redundant_penalty,
                ),
            )
            interleaved = optimizer._best_interleaved_fast_drop(
                redundant, protected_antecedents=frozenset()
            )
            # The deliberately unoptimized parent improves by updating its
            # existing coefficients alone.  A Drop must not claim that shared
            # refinement as its own contribution, so the matched parent path
            # correctly rejects this provisional deletion.
            self.assertIsNone(interleaved)
            self.assertGreater(
                optimizer._null_matched_parent_score(redundant),
                redundant.score,
            )

            # A routed support remains a feasible one-step state until its
            # terminal exact correction.  Matrix-free algebra must remain
            # identical there; otherwise every later candidate falls back to
            # a full matrix and large supports can exhaust memory.
            provisional = optimizer._conditional_one_step(
                empty, Support.of((singleton,)), device="cpu"
            ).record
            self.assertFalse(provisional.fit.converged)
            next_rule = RuleIdentity((1,), 0, 1)
            optimizer._rank_profiled_identities(provisional, (next_rule,))
            provisional_factorized = optimizer._factorized_add_one_step(
                provisional, next_rule, device="cpu"
            )
            self.assertIsNotNone(provisional_factorized)
            assert provisional_factorized is not None
            provisional_full = optimizer._conditional_one_step(
                provisional, provisional.support.add(next_rule), device="cpu"
            )
            np.testing.assert_allclose(
                provisional_factorized.fit.coefficients,
                provisional_full.record.fit.coefficients,
                rtol=0,
                atol=1e-11,
            )
            self.assertAlmostEqual(
                provisional_factorized.fit.nll,
                provisional_full.record.fit.nll,
                places=10,
            )

            provisional_pair = optimizer._materialize_factorized_add(
                provisional,
                provisional_factorized,
            )
            provisional_drop = optimizer._factorized_drop_one_step(
                provisional_pair,
                provisional_pair.support.drop(next_rule),
                device="cpu",
            )
            self.assertIsNotNone(provisional_drop)
            assert provisional_drop is not None
            materialized_drop = optimizer._materialize_factorized_drop(
                provisional_pair, provisional_drop
            )
            unaggregated_drop = optimizer._materialize_factorized_drop(
                provisional_pair,
                provisional_drop,
                aggregate_rows=False,
            )
            # Keeping duplicate projected sufficient-statistic rows is a
            # storage/scheduling rewrite only.  It must preserve the exact
            # feasible objective used to rank every W/sign replacement.
            self.assertAlmostEqual(
                optimizer._matrix_nll(
                    materialized_drop.matrix,
                    provisional_drop.fit.coefficients,
                ),
                optimizer._matrix_nll(
                    unaggregated_drop.matrix,
                    provisional_drop.fit.coefficients,
                ),
                places=10,
            )
            self.assertEqual(
                unaggregated_drop.matrix.x.shape[0],
                provisional_pair.matrix.x.shape[0],
            )
            canonical_drop = optimizer.engine.model_matrix(
                optimizer.context, materialized_drop.support
            )
            rows = np.arange(optimizer.context.n_grid, dtype=np.int64)
            np.testing.assert_allclose(
                optimizer.engine.design_at_rows_with_context(
                    optimizer.context, materialized_drop.matrix, rows
                ),
                optimizer.engine.design_at_rows_with_context(
                    optimizer.context, canonical_drop, rows
                ),
                rtol=0,
                atol=0,
            )

            # A high-order Drop unmasks selected lower-order states.  The
            # interleaved batch audit must zero those changed blocks so its
            # source-coordinate NLL is an actually feasible target-model NLL,
            # not a quadratic score or an invalid masked predictor.
            higher = RuleIdentity((0, 1), 1, -1)
            nested_support = Support.of((singleton, second_singleton, higher))
            nested = optimizer.fit(nested_support, empty)
            nested_trial = nested_support.drop(higher)
            zeroed = optimizer._interleaved_drop_zero_rules(
                nested, higher, nested_trial
            )
            self.assertIn(singleton, zeroed)
            self.assertIn(second_singleton, zeroed)
            batched_nll, source_coefficients = optimizer._batched_interleaved_drop_nlls(
                nested, (zeroed,), device="cpu"
            )
            batched_nll = batched_nll[0]
            target = optimizer.engine.model_matrix(optimizer.context, nested_trial)
            target_beta = optimizer.warm_start(nested, target)
            target_beta[: target.baseline_dimension] = source_coefficients[
                0, : nested.matrix.baseline_dimension
            ]
            for index, rule in enumerate(nested_trial.rules):
                source_index = nested_support.rules.index(rule)
                target_beta[target.rule_slices[index]] = source_coefficients[
                    0, nested.matrix.rule_slices[source_index]
                ]
            self.assertAlmostEqual(
                batched_nll,
                optimizer._matrix_nll(target, target_beta),
                places=9,
            )

            higher = RuleIdentity((0, 1), 1, -1)
            higher_record = optimizer.fit(Support.of((higher,)), empty)
            promoted_support = higher_record.support.add(singleton)
            self.assertTrue(
                optimizer.engine.total_state_geometry_changed(
                    higher_record.support, promoted_support
                )
            )
            self.assertFalse(optimizer._closure_dominated_add(higher_record, singleton))
            promoted = optimizer._project_factorized_support(
                higher_record, promoted_support
            )
            self.assertIsNone(promoted)
            effective = optimizer.engine.total_state_rule_blocks(
                optimizer.context, promoted_support
            )
            self.assertEqual(
                np.intersect1d(effective[0].rows, effective[1].rows).size,
                0,
            )
            self.assertLessEqual(
                optimizer._hierarchy_rule_geometry_bytes,
                optimizer._hierarchy_rule_geometry_limit,
            )

    def test_nonnested_transition_matrix_matches_full_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            source = Support.of((RuleIdentity((0, 1), 1, 1),))
            target = Support.of((RuleIdentity((0, 1), 2, -1),))
            source_matrix = optimizer.engine.model_matrix(optimizer.context, source)
            source_coefficients = np.zeros(source_matrix.dimension, dtype=np.float64)
            source_fit = FitResult(
                source_coefficients,
                optimizer._matrix_nll(source_matrix, source_coefficients),
                False,
                0,
                np.inf,
                source_matrix.dimension,
                False,
                "matrix-transition fixture",
            )
            current = SupportRecord(
                source,
                source_matrix,
                source_fit,
                optimizer.objective.penalty(
                    source, source_matrix, optimizer.baseline_dimension
                ),
                0.0,
            )
            self.assertIsNone(optimizer._project_factorized_support(current, target))

            transition, warm = optimizer._conditional_matrix_and_warm(current, target)
            canonical = optimizer.engine.model_matrix(optimizer.context, target)
            canonical_warm = optimizer.warm_start(current, canonical)
            rows = np.arange(optimizer.context.n_grid, dtype=np.int64)
            np.testing.assert_allclose(
                optimizer.engine.design_at_rows_with_context(
                    optimizer.context, transition, rows
                ),
                optimizer.engine.design_at_rows_with_context(
                    optimizer.context, canonical, rows
                ),
                rtol=0,
                atol=0,
            )
            np.testing.assert_allclose(warm, canonical_warm, rtol=0, atol=0)
            actual = one_step_model_matrix(
                transition,
                likelihood=data.likelihood,
                warm_start=warm,
                tolerance=config.solver_tolerance,
                device="cpu",
            )
            expected = one_step_model_matrix(
                canonical,
                likelihood=data.likelihood,
                warm_start=canonical_warm,
                tolerance=config.solver_tolerance,
                device="cpu",
            )
            np.testing.assert_allclose(
                actual.fit.coefficients,
                expected.fit.coefficients,
                rtol=0,
                atol=1e-11,
            )
            self.assertAlmostEqual(actual.fit.nll, expected.fit.nll, places=10)
            optimizer.close()

    def test_iteration_window_continuation_reaches_same_exact_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data")
            context = Context.make(data, np.arange(60, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            matrix = engine.model_matrix(
                context, Support.of((RuleIdentity((0,), 0, 1),))
            )
            reference = fit_model_matrix(
                matrix, likelihood=data.likelihood, tolerance=1e-8, max_iter=150
            )
            continued = fit_model_matrix_continued(
                matrix, likelihood=data.likelihood, tolerance=1e-8, max_iter=1
            )
            self.assertTrue(reference.converged, reference.message)
            self.assertTrue(continued.converged, continued.message)
            self.assertGreater(continued.iterations, 1)
            self.assertAlmostEqual(continued.nll, reference.nll, places=10)

    def test_cloglog_derivatives(self) -> None:
        eta = np.linspace(-5, 3, 21)
        value, first, second = cloglog_event_terms(eta)
        epsilon = 1.0e-5
        plus = cloglog_event_terms(eta + epsilon)[0]
        minus = cloglog_event_terms(eta - epsilon)[0]
        numeric_first = (plus - minus) / (2 * epsilon)
        numeric_second = (plus - 2 * value + minus) / epsilon**2
        np.testing.assert_allclose(first, numeric_first, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(second, numeric_second, rtol=2e-3, atol=2e-5)

    def test_value_only_likelihood_matches_full_path(self) -> None:
        eta = np.linspace(-20.0, 5.0, 101)
        event = (np.arange(len(eta)) % 7 == 0).astype(np.float64)
        exposure = np.ones_like(eta)
        for likelihood in ("poisson", "first_event_cloglog"):
            noevent = (
                exposure - event if likelihood == "first_event_cloglog" else exposure
            )
            expected = loss_rows(
                eta,
                likelihood=likelihood,
                exposure_weight=exposure,
                noevent_weight=noevent,
                event_weight=event,
            )[0]
            actual = loss_value_rows(
                eta,
                likelihood=likelihood,
                exposure_weight=exposure,
                noevent_weight=noevent,
                event_weight=event,
            )
            np.testing.assert_array_equal(actual, expected)

    def test_dual_bound_contains_exact_optimum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data")
            context = Context.make(data, np.arange(60, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            support = Support.of((RuleIdentity((0,), 0, 1),))
            matrix = engine.model_matrix(context, support)
            fit = fit_model_matrix(
                matrix, likelihood=data.likelihood, tolerance=1e-8, max_iter=150
            )
            self.assertTrue(fit.converged, fit.message)
            certificate = dual_certificate(
                matrix,
                likelihood=data.likelihood,
                beta=fit.coefficients,
                tolerance=1e-8,
                max_iter=1000,
            )
            self.assertTrue(certificate.feasible)
            self.assertLessEqual(certificate.nll_lower_bound, fit.nll + 1e-6)

    def test_search_resume_restores_exact_active_state_and_working_set(self) -> None:
        class Interrupted(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            captured: dict[str, object] = {}
            interrupted = SupportOptimizer(Context.make(data, fit_codes), config)

            def stop_after_first_move(completed, active) -> None:
                if active is None:
                    return
                captured["completed"] = completed
                captured["start"] = active[0]
                captured["support"] = active[1].support
                captured["moves"] = active[2]
                captured["fit"] = active[1].fit.to_dict()
                captured["search_state"] = interrupted.checkpoint_search_state()
                raise Interrupted

            with self.assertRaises(Interrupted):
                interrupted.search(progress_callback=stop_after_first_move)
            interrupted.close()
            self.assertIn("fit", captured)
            self.assertIn("profiled_roots", captured["search_state"])

            resumed = SupportOptimizer(Context.make(data, fit_codes), config)
            resumed.restore_search_state(captured["search_state"])
            with patch.object(
                resumed,
                "_standalone_profiled_atoms",
                side_effect=AssertionError("resume replayed standalone profiling"),
            ):
                resumed_result = resumed.search(
                    completed_paths=captured["completed"],
                    active_path=(
                        captured["start"],
                        captured["support"],
                        captured["moves"],
                        captured["fit"],
                    ),
                )
            reference = SupportOptimizer(Context.make(data, fit_codes), config)
            reference_result = reference.search()
            self.assertEqual(
                tuple(record.support for record in resumed_result.family),
                tuple(record.support for record in reference_result.family),
            )
            self.assertEqual(
                tuple(record.support for record in resumed_result.terminals),
                tuple(record.support for record in reference_result.terminals),
            )
            resumed.close()
            reference.close()

    def test_fit_materializes_restored_metadata_without_refitting(self) -> None:
        """A checkpoint cache hit must restore rows, not return empty metadata."""
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            support = Support.of((RuleIdentity((0,), 0, 1),))
            live = optimizer.fit(support, optimizer.records[EMPTY_SUPPORT])
            self.assertTrue(live.fit.converged, live.fit.message)
            frozen = freeze_support_record(live)
            self.assertEqual(frozen.matrix.x.shape[0], 0)
            optimizer._retain_record(frozen)
            exact_fits = optimizer.diagnostics.exact_fits

            # A metadata-only record has no design groups from which a safe
            # state quotient can be computed. It must fail open and, crucially,
            # must not cache an empty group state under the live support key.
            addition = RuleIdentity((1,), 0, 1)
            self.assertTrue(
                math.isinf(
                    optimizer._state_splice_group_upper_score(frozen, addition)
                )
            )

            restored = optimizer.fit(support, optimizer.records[EMPTY_SUPPORT])

            self.assertGreater(restored.matrix.x.shape[0], 0)
            self.assertEqual(optimizer.diagnostics.exact_fits, exact_fits)
            np.testing.assert_array_equal(
                restored.fit.coefficients,
                live.fit.coefficients,
            )
            self.assertEqual(restored.fit.nll, live.fit.nll)
            self.assertTrue(
                math.isfinite(
                    optimizer._state_splice_group_upper_score(restored, addition)
                )
            )

            checkpoint_state = optimizer.checkpoint_search_state()
            resumed = SupportOptimizer(Context.make(data, fit_codes), config)
            resumed.restore_search_state(checkpoint_state)
            resumed_exact_fits = resumed.diagnostics.exact_fits
            resumed_record = resumed.fit(
                support, resumed.records[EMPTY_SUPPORT]
            )
            self.assertGreater(resumed_record.matrix.x.shape[0], 0)
            self.assertEqual(resumed.diagnostics.exact_fits, resumed_exact_fits)
            np.testing.assert_array_equal(
                resumed_record.fit.coefficients,
                live.fit.coefficients,
            )
            resumed.close()
            optimizer.close()

    def test_factorized_additive_hierarchy_add_matches_full_one_step(self) -> None:
        """A closure-only append must not be misclassified as a state splice."""
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                effect_model="additive_hierarchy",
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[EMPTY_SUPPORT]
            current = optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)), empty)
            rule = RuleIdentity((0, 1), 1, 1, hierarchical=True)
            self.assertEqual(
                _support_from_payload(_support_payload(Support.of((rule,)))),
                Support.of((rule,)),
            )
            optimizer._rank_profiled_identities(current, (rule,))
            factorized = optimizer._factorized_add_one_step(current, rule, device="cpu")
            self.assertIsNotNone(factorized)
            assert factorized is not None
            full = optimizer._conditional_one_step(
                current, current.support.add(rule), device="cpu"
            )
            np.testing.assert_allclose(
                factorized.fit.coefficients,
                full.record.fit.coefficients,
                rtol=0,
                atol=2e-10,
            )
            self.assertAlmostEqual(factorized.fit.nll, full.record.fit.nll, places=9)
            self.assertAlmostEqual(factorized.lower_score, full.lower_score, places=8)
            optimizer.close()

    def test_sparse_additive_hierarchy_fit_matches_dense_exact(self) -> None:
        """Automatic closure blocks are ordinary exact sparse columns."""
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=("cpu",),
                effect_model="additive_hierarchy",
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[EMPTY_SUPPORT]
            support = Support.of((RuleIdentity((0, 1), 1, 1, hierarchical=True),))
            optimizer._ensure_closure_signs(hierarchy_closure(support))
            dense_matrix = optimizer.engine.model_matrix(optimizer.context, support)
            dense = fit_model_matrix_continued(
                dense_matrix,
                likelihood=data.likelihood,
                tolerance=config.solver_tolerance,
                max_iter=config.solver_max_iter,
                warm_start=optimizer.warm_start(empty, dense_matrix),
            )
            self.assertTrue(dense.converged, dense.message)
            sparse = optimizer._fit_sparse_supports_exact_many((support,), empty)[0]
            self.assertIsNotNone(sparse)
            assert sparse is not None
            self.assertTrue(sparse.fit.converged, sparse.fit.message)
            self.assertAlmostEqual(sparse.fit.nll, dense.nll, places=8)
            np.testing.assert_allclose(
                sparse.fit.coefficients,
                dense.coefficients,
                rtol=0,
                atol=2e-7,
            )
            null_coefficients = dense.coefficients.copy()
            null_coefficients[dense_matrix.rule_slices[0]] = 0.0
            null_fit = FitResult(
                coefficients=null_coefficients,
                nll=optimizer._matrix_nll(dense_matrix, null_coefficients),
                converged=True,
                iterations=0,
                projected_kkt=0.0,
                rank=dense.rank,
                recession=False,
                message="test embedded null",
            )
            direct_difference = optimizer._entity_loss_difference_same_matrix(
                dense_matrix,
                null_fit,
                dense,
            )
            reference_difference = optimizer._entity_losses_for_model(
                dense_matrix,
                null_fit,
            ) - optimizer._entity_losses_for_model(dense_matrix, dense)
            np.testing.assert_allclose(
                direct_difference,
                reference_difference,
                rtol=2e-11,
                atol=2e-11,
            )
            optimizer.close()

    def test_disjoint_route_shards_reproduce_serial_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                search_mode="fast_block_score",
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            context = Context.make(data, fit_codes)
            serial = SupportOptimizer(context, config)
            serial_result = serial.search()

            profiler = SupportOptimizer(context, config)
            base = profiler.search(
                allowed_starts=frozenset((EMPTY_SUPPORT,)),
                finalize_family=False,
            )
            state = profiler.checkpoint_search_state()
            roots = tuple(record.support for record in base.positive_atoms)
            base_completed = tuple(
                (
                    support_from_key(str(path["start"])),
                    support_from_key(str(path["terminal"])),
                    path,
                )
                for path in base.paths
            )
            merged = {
                start: (start, terminal, path)
                for start, terminal, path in base_completed
            }
            for shard in (roots[::2], roots[1::2]):
                worker = SupportOptimizer(context, config)
                worker.restore_search_state(state)
                partial = worker.search(
                    completed_paths=base_completed,
                    allowed_starts=frozenset(shard),
                    finalize_family=False,
                )
                for path in partial.paths:
                    start = support_from_key(str(path["start"]))
                    if start not in shard:
                        continue
                    merged[start] = (
                        start,
                        support_from_key(str(path["terminal"])),
                        path,
                    )
                worker.close()

            combined = SupportOptimizer(context, config)
            combined.restore_search_state(state)
            combined_result = combined.search(completed_paths=tuple(merged.values()))
            self.assertEqual(
                tuple(record.support for record in combined_result.family),
                tuple(record.support for record in serial_result.family),
            )
            self.assertEqual(
                tuple(record.support for record in combined_result.terminals),
                tuple(record.support for record in serial_result.terminals),
            )
            self.assertEqual(
                {path["start"]: path["terminal"] for path in combined_result.paths},
                {path["start"]: path["terminal"] for path in serial_result.paths},
            )
            profiler.close()
            combined.close()
            serial.close()

    def test_fast_search_resume_restores_provisional_route(self) -> None:
        class Interrupted(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                search_mode="fast_block_score",
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            captured: dict[str, object] = {}
            interrupted = SupportOptimizer(Context.make(data, fit_codes), config)

            def stop_after_first_move(completed, active) -> None:
                if active is None:
                    return
                captured["completed"] = completed
                captured["start"] = active[0]
                captured["support"] = active[1].support
                captured["moves"] = active[2]
                captured["fit"] = active[1].fit.to_dict()
                captured["search_state"] = interrupted.checkpoint_search_state()
                raise Interrupted

            with self.assertRaises(Interrupted):
                interrupted.search(progress_callback=stop_after_first_move)
            interrupted.close()
            # Accepted Adds are exactified before entering a route because
            # their reoptimized shared-closure contrast is now the gate.
            self.assertTrue(bool(captured["fit"]["converged"]))

            resumed = SupportOptimizer(Context.make(data, fit_codes), config)
            resumed.restore_search_state(captured["search_state"])
            resumed_result = resumed.search(
                completed_paths=captured["completed"],
                active_path=(
                    captured["start"],
                    captured["support"],
                    captured["moves"],
                    captured["fit"],
                ),
            )
            reference = SupportOptimizer(Context.make(data, fit_codes), config)
            reference_result = reference.search()
            self.assertEqual(
                tuple(record.support for record in resumed_result.family),
                tuple(record.support for record in reference_result.family),
            )
            self.assertEqual(
                tuple(record.support for record in resumed_result.terminals),
                tuple(record.support for record in reference_result.terminals),
            )
            resumed.close()
            reference.close()

    def test_factorized_add_respects_dynamic_baseline_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = controlled_synthetic_dataset(Path(directory) / "data")
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=2,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=2,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(context, config)
            empty = optimizer.records[Support(())]
            current = optimizer.fit(Support.of((RuleIdentity((0,), 0, 1),)), empty)
            candidates = (
                RuleIdentity((1,), 0, 1),
                RuleIdentity((0, 1), 1, -1),
            )
            for rule in candidates:
                optimizer._rank_profiled_identities(current, (rule,))
                factorized = optimizer._factorized_add_one_step(
                    current, rule, device="cpu"
                )
                if set((0,)) < set(rule.antecedent):
                    self.assertIsNone(factorized)
                    full = optimizer._conditional_one_step(
                        current, current.support.add(rule), device="cpu"
                    )
                    self.assertEqual(
                        full.record.matrix.control_dimension,
                        config.knot_count,
                    )
                    self.assertTrue(np.isfinite(full.record.fit.nll))
                    continue
                self.assertIsNotNone(factorized)
                assert factorized is not None
                full = optimizer._conditional_one_step(
                    current, current.support.add(rule), device="cpu"
                )
                # Parity includes the full multi-knot control slice, which was
                # the part formerly omitted from the factorized branch.
                self.assertEqual(
                    full.record.matrix.control_dimension,
                    config.knot_count,
                )
                np.testing.assert_allclose(
                    factorized.fit.coefficients,
                    full.record.fit.coefficients,
                    rtol=0,
                    atol=2e-10,
                )
                self.assertAlmostEqual(
                    factorized.fit.nll, full.record.fit.nll, places=9
                )
                self.assertLessEqual(
                    factorized.lower_score,
                    optimizer.saturated_upper_score(factorized.support)
                    + config.search_tolerance,
                )

    def test_terminal_exactification_rejects_a_worse_converged_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(context, config)
            support = Support.of((RuleIdentity((0,), 0, 1),))
            exact = optimizer.fit(support, optimizer.records[Support(())])
            self.assertTrue(exact.fit.converged, exact.fit.message)

            good_nll = optimizer._matrix_nll(exact.matrix, exact.fit.coefficients)
            provisional_fit = FitResult(
                exact.fit.coefficients.copy(),
                good_nll,
                False,
                1,
                10.0 * config.solver_tolerance,
                exact.fit.rank,
                False,
                "better feasible route point",
            )
            provisional = SupportRecord(
                support,
                exact.matrix,
                provisional_fit,
                exact.penalty,
                support_score(
                    baseline_nll=optimizer.baseline_nll,
                    fit_nll=good_nll,
                    penalty=exact.penalty,
                ),
            )

            worse_coefficients = exact.fit.coefficients.copy()
            worse_coefficients[0] += 0.5
            worse_nll = optimizer._matrix_nll(exact.matrix, worse_coefficients)
            self.assertGreater(worse_nll, good_nll)
            worse_fit = FitResult(
                worse_coefficients,
                worse_nll,
                True,
                exact.fit.iterations,
                exact.fit.projected_kkt,
                exact.fit.rank,
                False,
                "stale converged cache",
            )
            optimizer._stored_records[support] = _StoredRecord(
                worse_fit,
                exact.penalty,
                support_score(
                    baseline_nll=optimizer.baseline_nll,
                    fit_nll=worse_nll,
                    penalty=exact.penalty,
                ),
            )

            corrected = optimizer._exactify_path_state(provisional, reason="terminal")
            self.assertTrue(corrected.fit.converged, corrected.fit.message)
            self.assertLessEqual(corrected.fit.nll, good_nll + 1e-10)
            self.assertLess(corrected.fit.nll, worse_nll)
            self.assertAlmostEqual(
                corrected.fit.nll,
                optimizer._stored_records[support].fit.nll,
                places=12,
            )

    def test_terminal_cache_has_no_conflicting_scaled_nll_slack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(context, config)
            support = Support.of((RuleIdentity((0,), 0, 1),))
            exact = optimizer.fit(support, optimizer.records[Support(())])
            provisional_fit = FitResult(
                exact.fit.coefficients.copy(),
                1.0e9,
                False,
                1,
                1.0,
                exact.fit.rank,
                False,
                "large-scale feasible point",
            )
            provisional = SupportRecord(
                support,
                exact.matrix,
                provisional_fit,
                exact.penalty,
                -1.0,
            )
            stale_fit = FitResult(
                exact.fit.coefficients.copy(),
                1.0e9 + 1.0e-6,
                True,
                1,
                exact.fit.projected_kkt,
                exact.fit.rank,
                False,
                "numerically worse cache",
            )
            optimizer._stored_records[support] = _StoredRecord(
                stale_fit, exact.penalty, -2.0
            )
            refined = FitResult(
                provisional_fit.coefficients,
                1.0e9,
                True,
                1,
                0.0,
                exact.fit.rank,
                False,
                "refined",
            )
            with (
                patch.object(
                    optimizer,
                    "_matrix_nll",
                    side_effect=(1.0e9, 1.0e9 + 1.0e-6, 1.0e9),
                ),
                patch(
                    "crbstpp.search.fit_model_matrix_continued",
                    return_value=refined,
                ) as solve,
            ):
                corrected = optimizer._exactify_path_state(
                    provisional, reason="terminal"
                )
            solve.assert_called_once()
            self.assertEqual(corrected.fit.nll, 1.0e9)

    def test_parallel_search_matches_serial_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            common = dict(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            families = []
            for workers in (1, 3):
                result = SupportOptimizer(
                    Context.make(data, fit_codes),
                    RunConfig(**common, exact_workers=workers),
                ).search()
                families.append(
                    [(record.support, record.score) for record in result.family]
                )
            self.assertEqual(
                [item[0] for item in families[0]],
                [item[0] for item in families[1]],
            )
            np.testing.assert_allclose(
                [item[1] for item in families[0]],
                [item[1] for item in families[1]],
                rtol=0,
                atol=1e-10,
            )

    def test_fast_block_score_search_uses_common_total_state_adds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 3),
                search_mode="fast_block_score",
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            result = optimizer.search()
            self.assertGreater(len(result.terminals), 0)
            self.assertEqual(
                result.diagnostics.multi_source_roots,
                1 + result.diagnostics.route_family_active_roots,
            )
            self.assertLessEqual(
                result.diagnostics.route_family_active_roots,
                len(result.positive_atoms),
            )
            self.assertTrue(
                {
                    record.support.rules[0].pattern_key
                    for record in result.positive_atoms
                }.issubset(optimizer._standalone_report_antecedents)
            )
            self.assertEqual(
                optimizer._working_antecedents,
                set(optimizer.skeletons),
            )
            self.assertGreater(result.diagnostics.exact_branch_add_audits, 0)
            # Every unresolved terminal Add reaches an exact-safe audit.  A
            # tiny dictionary may contain only state-splice candidates, so it
            # need not invoke the closure-free sparse sibling backend.
            self.assertGreater(
                result.diagnostics.safe_column_exact_audits,
                0,
            )
            self.assertEqual(result.diagnostics.provisional_route_moves, 0)
            self.assertGreater(
                result.diagnostics.support_contract_add_audits
                + result.diagnostics.indecomposable_add_screens,
                0,
            )
            self.assertGreater(result.diagnostics.rule_objective_drop_audits, 0)
            self.assertGreater(result.diagnostics.matched_null_workspace_reuses, 0)
            self.assertGreater(result.diagnostics.reverse_drop_screens, 0)
            self.assertLess(
                result.diagnostics.terminal_exact_corrections,
                result.diagnostics.accepted_moves,
            )
            for antecedent in optimizer._profiled_by_antecedent:
                directional = {
                    sign
                    for candidate, sign in optimizer._profiled_by_antecedent_sign
                    if candidate == antecedent
                }
                self.assertEqual(directional, {-1, 1})
            for term, sign in optimizer._closure_signs.items():
                lower = optimizer._profiled_by_antecedent.get(term.antecedent)
                if lower is not None:
                    self.assertEqual(sign, lower.sign)
            frozen = optimizer._inactive_identities(
                optimizer.records[Support(())],
                set(optimizer._profiled_by_antecedent),
                frozen=True,
            )
            by_antecedent: dict[tuple[int, ...], set[int]] = {}
            for rule in frozen:
                by_antecedent.setdefault(rule.antecedent, set()).add(rule.sign)
            self.assertTrue(all(len(signs) == 1 for signs in by_antecedent.values()))
            self.assertEqual(
                {rule for rule in frozen},
                set(optimizer._profiled_by_antecedent.values()),
            )
            for record in (*result.family, *result.terminals):
                self.assertTrue(record.fit.converged, record.fit.message)
                # Frozen search outputs must not retain observation-sized
                # design rows through certification.
                self.assertEqual(record.matrix.x.shape[0], 0)
                self.assertEqual(record.matrix.x.shape[1], len(record.fit.coefficients))
                if config.max_rules_per_support is not None:
                    self.assertLessEqual(
                        len(record.support.rules), config.max_rules_per_support
                    )
            for path in result.paths:
                terminal = support_from_key(str(path["terminal"]))
                anchors = {
                    tuple(int(value) for value in antecedent)
                    for antecedent in path["anchor_antecedents"]
                }
                # Positive atoms initialize routes but are never mandatory
                # members of a reported terminal.
                self.assertEqual(anchors, set())
                self.assertEqual(
                    path["max_rules_per_support"], config.max_rules_per_support
                )
                self.assertEqual(
                    path["route_policy"],
                    "common_total_state_add_terminal_drop_identity_audit",
                )
                for move in path["moves"]:
                    if move.get("phase") == "forward":
                        self.assertIn(move["move"], {"add", "drop", "splice"})
                        if move["move"] == "add":
                            self.assertNotIn("matched_branch_deferred", move)
                            self.assertGreater(move["attributable_gain"], 0.0)
                            self.assertGreaterEqual(move["parent_refinement_gain"], 0.0)
                            self.assertGreater(move["matched_branch_net"], 0.0)
                        else:
                            self.assertTrue(move["added_rules"])
                            self.assertTrue(move["dropped_rules"])
                    elif move.get("phase") == "cleanup":
                        self.assertIn(move["move"], {"add", "drop", "identity"})
                terminal_record = next(
                    record for record in result.terminals if record.support == terminal
                )
                self.assertIsNone(
                    optimizer._best_terminal_cleanup_decision(
                        terminal_record,
                        protected_antecedents=frozenset(),
                    ).record
                )
            self.assertGreater(result.diagnostics.null_matched_parent_steps, 0)
            self.assertEqual(
                result.diagnostics.standalone_branch_audits,
                len(result.positive_atoms)
                + result.diagnostics.standalone_branch_rejections,
            )

    def test_fast_add_continues_after_higher_ranked_exact_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                search_mode="fast_block_score",
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=8 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            current = optimizer.records[EMPTY_SUPPORT]
            first_rule = RuleIdentity((0,), 0, 1)
            second_rule = RuleIdentity((1,), 0, 1)
            first_support = current.support.add(first_rule)
            second_support = current.support.add(second_rule)
            first = SupportRecord(
                first_support,
                current.matrix,
                current.fit,
                current.penalty,
                current.score,
            )
            second = SupportRecord(
                second_support,
                current.matrix,
                current.fit,
                current.penalty,
                current.score,
            )
            optimizer._stored_records[first_support] = _StoredRecord(
                current.fit, current.penalty, current.score
            )
            optimizer._stored_records[second_support] = _StoredRecord(
                current.fit, current.penalty, current.score
            )
            viable = [
                (2.0, 2.0, first_rule),
                (1.0, 1.0, second_rule),
            ]
            with (
                patch.object(
                    SupportOptimizer,
                    "fit",
                    side_effect=(first, second),
                ),
                patch.object(
                    SupportOptimizer,
                    "_exact_add_branch_validation",
                    side_effect=((None, -1.0), (second, 1.0)),
                ) as validate,
                patch.object(
                    SupportOptimizer,
                    "_fit_add_is_material",
                    return_value=True,
                ),
            ):
                selected = optimizer._first_validated_block_score_add(current, viable)
            self.assertIs(selected, second)
            self.assertEqual(validate.call_count, 2)

    def test_exact_terminal_routes_through_safe_column_oracle(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer.config = SimpleNamespace(
            search_tolerance=1.0e-8,
            pricing_devices=("cpu",),
            effect_model="support_additive",
        )
        optimizer._terminal_add_audit_active = True
        optimizer._block_score_terminal_audit_active = False
        optimizer._conditional_forbidden = set()
        optimizer._conditional_parent_forbidden = set()
        optimizer._relaxed_upper_cache = {}
        optimizer._directional_upper_cache = {}
        parent_rule = RuleIdentity((0,), 0, 1)
        added_rule = RuleIdentity((1,), 0, -1)
        current = SimpleNamespace(
            support=Support.of((parent_rule,)),
            fit=SimpleNamespace(converged=True),
            score=3.0,
        )
        optimizer._add_respects_support_contract = lambda *_args: True
        optimizer._separate_family_scores = lambda *_args: {added_rule.pattern_key: 4.0}
        optimizer._safe_identity_survivors = lambda _current, identities, _threshold: (
            identities
        )
        sentinel = object()
        observed: dict[str, object] = {}

        def safe_add(_current, viable, **kwargs):
            observed["viable"] = viable
            observed["thresholds"] = kwargs["score_thresholds"]
            return sentinel

        optimizer._best_safe_column_addition = safe_add
        selected = optimizer._first_validated_block_score_add(
            current,
            [(2.0, 2.0, added_rule)],
        )
        self.assertIs(selected, sentinel)
        self.assertEqual(observed["viable"][0][2], added_rule)
        self.assertEqual(
            observed["thresholds"][current.support.add(added_rule)],
            4.0,
        )

    def test_terminal_family_decomposition_checks_only_active_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                search_mode="fast_block_score",
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=8 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[EMPTY_SUPPORT]
            support = Support.of(
                (RuleIdentity((0,), 0, 1), RuleIdentity((1,), 0, 1))
            )
            current = optimizer._attach_rule_score(optimizer.fit(support, empty))
            parents = {
                current.support.drop(rule): optimizer._attach_rule_score(
                    optimizer.fit(current.support.drop(rule), current)
                )
                for rule in current.support.rules
            }
            with (
                patch.object(
                    optimizer,
                    "_stored_metadata_record",
                    side_effect=lambda item: parents.get(item),
                ),
                patch.object(
                    optimizer,
                    "_fit_sparse_supports_exact_many",
                    side_effect=AssertionError("inactive candidates were fitted"),
                ),
                patch.object(optimizer, "_add_is_indecomposable", return_value=False)
                as indecomposable,
            ):
                dropped = optimizer._best_exact_family_decomposition_drop(
                    current, protected_antecedents=frozenset()
                )
            self.assertIsNotNone(dropped)
            assert dropped is not None
            self.assertEqual(len(dropped.support.rules), 1)
            self.assertEqual(indecomposable.call_count, 2)
            optimizer.close()

    def test_terminal_block_add_uses_one_rank_table_across_exact_rejections(self) -> None:
        """Rejected exact children advance without repricing the parent."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                search_mode="fast_block_score",
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=8 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            optimizer._block_score_terminal_audit_active = True
            optimizer._terminal_add_audit_active = False
            current = optimizer.records[EMPTY_SUPPORT]
            first_rule = RuleIdentity((0,), 0, 1)
            second_rule = RuleIdentity((1,), 0, 1)
            first_support = current.support.add(first_rule)
            second_support = current.support.add(second_rule)
            first = SupportRecord(
                first_support,
                current.matrix,
                current.fit,
                current.penalty,
                current.score,
            )
            second = SupportRecord(
                second_support,
                current.matrix,
                current.fit,
                current.penalty,
                current.score,
            )
            optimizer._stored_records[first_support] = _StoredRecord(
                current.fit, current.penalty, current.score
            )
            optimizer._stored_records[second_support] = _StoredRecord(
                current.fit, current.penalty, current.score
            )
            viable = [
                (2.0, 2.0, first_rule),
                (1.0, 1.0, second_rule),
            ]
            with (
                patch.object(
                    SupportOptimizer,
                    "fit",
                    side_effect=(first, second),
                ),
                patch.object(
                    SupportOptimizer,
                    "_normalize_composite_addition",
                    side_effect=(None, second),
                ) as normalize,
            ):
                selected = optimizer._first_validated_block_score_add(
                    current,
                    viable,
                    composite_cleanup=True,
                )
            self.assertIs(selected, second)
            self.assertEqual(normalize.call_count, 2)

    def test_w_profile_groups_only_windows_and_preserves_sign_history(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._conditional_forbidden = set()
        optimizer._conditional_parent_forbidden = set()
        current = SimpleNamespace(support=EMPTY_SUPPORT)
        excitation_w2 = RuleIdentity((0, 1), 2, 1, relation="unordered")
        excitation_w4 = RuleIdentity((0, 1), 4, 1, relation="unordered")
        inhibition_w2 = RuleIdentity((0, 1), 2, -1, relation="unordered")
        repeated_w2 = RuleIdentity(
            (0, 1),
            2,
            1,
            relation="unordered",
            history_marks=((30, 2), (0, 0)),
        )
        ranked = [
            (10.0, 10.0, excitation_w2),
            (9.0, 9.0, excitation_w4),
            (8.0, 8.0, inhibition_w2),
            (7.0, 7.0, repeated_w2),
        ]

        first = optimizer._w_profile_representatives(current, ranked)
        self.assertEqual(
            {item[2] for item in first},
            {excitation_w2, inhibition_w2, repeated_w2},
        )

        optimizer._conditional_parent_forbidden.add(
            (EMPTY_SUPPORT, EMPTY_SUPPORT.add(excitation_w2))
        )
        second = optimizer._w_profile_representatives(current, ranked)
        self.assertEqual(
            {item[2] for item in second},
            {excitation_w4, inhibition_w2, repeated_w2},
        )

    def test_exact_parent_rank_cache_survives_equivalent_matrix_rebuild(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer.config = SimpleNamespace(
            knot_count=0,
            search_mode="atomic_rashomon_frontier",
            frequency_effect_separation=False,
        )
        optimizer._skeleton_witnesses = {}
        optimizer._add_rank_tables = OrderedDict()
        optimizer._add_rank_table_limit = 8
        optimizer._hierarchy_rule_geometries = {}
        optimizer._conditional_forbidden = set()
        rule = RuleIdentity((0, 1), 2, 1, relation="unordered")
        calls = 0

        def rank(*_args):
            nonlocal calls
            calls += 1
            return [(3.0, 3.0, rule, False)]

        optimizer._rank_block_identities = rank
        fit_one = SimpleNamespace(
            converged=True, coefficients=np.array([0.25], dtype=np.float64)
        )
        fit_two = SimpleNamespace(
            converged=True, coefficients=np.array([0.25], dtype=np.float64)
        )
        first = SimpleNamespace(
            support=EMPTY_SUPPORT, matrix=SimpleNamespace(), fit=fit_one
        )
        rebuilt = SimpleNamespace(
            support=EMPTY_SUPPORT, matrix=SimpleNamespace(), fit=fit_two
        )

        self.assertEqual(optimizer._rank_profiled_identities(first, (rule,))[0][2], rule)
        self.assertEqual(
            optimizer._rank_profiled_identities(rebuilt, (rule,))[0][2], rule
        )
        self.assertEqual(calls, 1)
        self.assertEqual(optimizer.diagnostics.add_rank_table_hits, 1)

    def test_profiled_rank_screen_rejects_candidate_collinear_with_parent(self) -> None:
        """Raw candidate rank cannot hide zero conditional information."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                search_mode="fast_block_score",
                cache_bytes=8 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            current = optimizer.records[EMPTY_SUPPORT]
            rule = RuleIdentity((1,), 0, 1)
            # The unprofiled candidate block is full rank.  After projecting
            # out the active model it has exactly zero information, which is
            # the rank condition the fixed-support solver would encounter.
            optimizer._hierarchy_rule_geometries[
                (current.support, rule.pattern_key, rule.window)
            ] = _HierarchyRuleGeometry(
                gradient=np.zeros(2, dtype=np.float64),
                hessian=np.zeros((2, 2), dtype=np.float64),
                nested=True,
                joint_gradient=np.zeros(2, dtype=np.float64),
                joint_hessian=np.eye(2, dtype=np.float64),
            )
            with patch.object(
                optimizer,
                "_rank_block_identities",
                return_value=[(10.0, 10.0, rule, True)],
            ):
                ranked = optimizer._rank_profiled_identities(current, (rule,))
            self.assertEqual(ranked, [])
            self.assertIn(current.support.add(rule), optimizer._conditional_forbidden)
            self.assertEqual(optimizer.diagnostics.factorized_add_rank_screens, 1)
            optimizer.close()

    def test_matched_branch_rejects_old_coefficient_hitchhiker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 180)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 3),
                search_mode="fast_block_score",
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            exact_empty = optimizer.records[Support(())]
            bad_coefficients = exact_empty.fit.coefficients.copy()
            bad_coefficients[0] += 4.0
            bad_nll = optimizer._matrix_nll(exact_empty.matrix, bad_coefficients)
            provisional = SupportRecord(
                exact_empty.support,
                exact_empty.matrix,
                FitResult(
                    bad_coefficients,
                    bad_nll,
                    False,
                    0,
                    1.0,
                    exact_empty.matrix.dimension,
                    False,
                    "deliberately unrefined parent",
                ),
                exact_empty.penalty,
                support_score(
                    baseline_nll=optimizer.baseline_nll,
                    fit_nll=bad_nll,
                    penalty=exact_empty.penalty,
                ),
            )
            rule = RuleIdentity((0,), 0, 1)
            support = Support.of((rule,))
            matrix = optimizer.engine.model_matrix(optimizer.context, support)
            child_coefficients = optimizer.warm_start(exact_empty, matrix)
            child_nll = optimizer._matrix_nll(matrix, child_coefficients)
            penalty = optimizer.objective.penalty(
                support, matrix, optimizer.baseline_dimension
            )
            child = SupportRecord(
                support,
                matrix,
                FitResult(
                    child_coefficients,
                    child_nll,
                    False,
                    0,
                    1.0,
                    matrix.dimension,
                    False,
                    "old coefficients repaired but new rule remains zero",
                ),
                penalty,
                support_score(
                    baseline_nll=optimizer.baseline_nll,
                    fit_nll=child_nll,
                    penalty=penalty,
                ),
            )
            self.assertGreater(child.score, provisional.score)
            bad_parent_net = optimizer._matched_add_branch_net(provisional, child, rule)
            exact_child = optimizer.fit(support, exact_empty)
            exact_parent_net = optimizer._matched_add_branch_net(
                exact_empty, exact_child, rule
            )
            self.assertAlmostEqual(bad_parent_net, exact_parent_net, places=8)


if __name__ == "__main__":
    unittest.main()
