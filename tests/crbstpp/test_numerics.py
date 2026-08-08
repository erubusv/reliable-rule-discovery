from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from crbstpp.likelihood import (
    cloglog_conjugate,
    conjugate_sum,
    loss_grid_sparse_event_derivatives,
    loss_rows,
)
from crbstpp.native import (
    aggregate_design_rows,
    aggregate_design_rows_with_groups,
    aggregate_quotient_rows,
    bounded_span_order,
    completion_events,
    cpu_available,
    cuda_available,
    design_column_cross,
    fused_likelihood_value_eta_gradient,
    moments,
    moments_batch,
    new_derivative_token,
    nonnegative_quadratic_gains,
    resident_cloglog_objective,
    resident_eta,
    resident_poisson_objective,
    response_min_spans,
    sparse_moments_batch,
    sparse_moments_indexed_batch,
    sorted_unique_union,
    subtract_group_weights,
)
from crbstpp.objective import support_score
from crbstpp.response import Context, ResponseEngine
from crbstpp.config import RunConfig
from crbstpp.data import Dataset, write_dataset
from crbstpp.dual import natural_dual_certificate, offset_dual_certificate
from crbstpp.rules import RuleIdentity, Support, hierarchy_closure
from crbstpp.search import (
    SearchDiagnostics,
    SupportOptimizer,
    _batched_nonnegative_quadratic_gain,
    _nonnegative_quadratic_gain,
    _nonnegative_quadratic_solution,
)
from crbstpp.solver import (
    FitResult,
    ProjectedDesignEvaluator,
    _axis_recession,
    _critical_cone_rank,
    _general_recession_design,
    _objective,
    fit_model_matrix,
    fit_offset_design,
    fit_offset_design_continued,
    fit_projected_model_matrix,
    one_step_model_matrix,
)

from tests.crbstpp.test_core import synthetic_dataset


class NumericalParityTests(unittest.TestCase):
    @unittest.skipUnless(cpu_available(), "compiled CPU operators are unavailable")
    def test_fused_likelihood_gradient_and_column_cross_match_numpy(self) -> None:
        rng = np.random.default_rng(1907)
        x = np.ascontiguousarray(rng.normal(size=(4096, 9)))
        beta = rng.normal(size=9)
        primary = rng.uniform(0.2, 1.4, size=len(x))
        event = (rng.random(len(x)) < 0.07).astype(np.float64)
        for likelihood in ("poisson", "first_event_cloglog"):
            actual = fused_likelihood_value_eta_gradient(
                x,
                beta,
                primary,
                event,
                likelihood=likelihood,
            )
            self.assertIsNotNone(actual)
            nll, eta, gradient = actual
            expected_eta = x @ beta
            rows, first, _ = loss_rows(
                expected_eta,
                likelihood=likelihood,
                exposure_weight=primary,
                noevent_weight=primary,
                event_weight=event,
            )
            self.assertAlmostEqual(nll, float(np.sum(rows)), places=8)
            np.testing.assert_allclose(eta, expected_eta, rtol=1.0e-13, atol=1.0e-13)
            np.testing.assert_allclose(
                gradient, x.T @ first, rtol=1.0e-12, atol=1.0e-9
            )
        np.testing.assert_allclose(
            design_column_cross(x, 4),
            x.T @ x[:, 4],
            rtol=1.0e-12,
            atol=1.0e-10,
        )

    def test_constrained_rank_uses_the_critical_face(self) -> None:
        beta = np.asarray([0.2, 0.0])
        hessian = np.asarray([[2.0, 0.0], [0.0, 0.0]])
        rank, dimension = _critical_cone_rank(
            beta,
            np.asarray([0.0, 0.5]),
            hessian,
            free_dimension=1,
            tolerance=1.0e-8,
        )
        self.assertEqual((rank, dimension), (1, 1))
        rank, dimension = _critical_cone_rank(
            beta,
            np.asarray([0.0, 0.0]),
            hessian,
            free_dimension=1,
            tolerance=1.0e-8,
        )
        self.assertEqual((rank, dimension), (1, 2))

    def test_canonical_fit_fails_open_to_route_warm_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 80)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=50,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            canonical_failed = replace(
                empty.fit,
                converged=False,
                recession=False,
                iterations=2,
                message="rank-deficient fixed-support information",
            )
            with mock.patch(
                "crbstpp.search.fit_model_matrix",
                side_effect=(canonical_failed, empty.fit),
            ) as solver:
                recovered = optimizer._fit_support_matrix(
                    empty.matrix,
                    warm_start=empty.fit.coefficients,
                    device="cpu",
                )
            self.assertTrue(recovered.converged)
            self.assertEqual(solver.call_count, 2)
            self.assertIsNone(solver.call_args_list[0].kwargs["warm_start"])
            self.assertIsNotNone(solver.call_args_list[1].kwargs["warm_start"])
            self.assertEqual(optimizer.diagnostics.canonical_first_fits, 1)
            self.assertEqual(
                optimizer.diagnostics.canonical_first_warm_fallbacks,
                1,
            )
            self.assertEqual(
                optimizer.diagnostics.canonical_first_warm_recoveries,
                1,
            )
            optimizer.close()

    def test_converged_canonical_fit_skips_route_warm_solve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 80)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=50,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            with mock.patch(
                "crbstpp.search.fit_model_matrix",
                return_value=empty.fit,
            ) as solver:
                recovered = optimizer._fit_support_matrix(
                    empty.matrix,
                    warm_start=empty.fit.coefficients,
                    device="cpu",
                )
            self.assertTrue(recovered.converged)
            self.assertEqual(solver.call_count, 1)
            self.assertIsNone(solver.call_args.kwargs["warm_start"])
            self.assertEqual(optimizer.diagnostics.canonical_first_fits, 1)
            self.assertEqual(
                optimizer.diagnostics.canonical_first_warm_fallbacks,
                0,
            )
            optimizer.close()

    @unittest.skipUnless(cuda_available(), "CUDA objective solver is unavailable")
    def test_resident_poisson_objective_matches_host_reference(self) -> None:
        rng = np.random.default_rng(20260724)
        x = np.ascontiguousarray(rng.normal(size=(5003, 7)))
        beta = rng.normal(scale=0.2, size=7)
        exposure = rng.uniform(0.01, 2.0, size=len(x))
        event = rng.poisson(0.15, size=len(x)).astype(np.float64)
        eta = x @ beta
        values, first, second = loss_rows(
            eta,
            likelihood="poisson",
            exposure_weight=exposure,
            noevent_weight=exposure,
            event_weight=event,
        )
        expected_gradient, expected_hessian = moments(x, first, second, device="cpu")
        actual = resident_poisson_objective(
            x,
            beta,
            exposure,
            event,
            device="cuda:0",
            matrix_token=new_derivative_token(),
            compute_moments=True,
            return_eta=True,
        )
        self.assertIsNotNone(actual)
        assert actual is not None
        nll, actual_eta, gradient, hessian = actual
        self.assertIsNotNone(actual_eta)
        self.assertIsNotNone(gradient)
        self.assertIsNotNone(hessian)
        np.testing.assert_allclose(actual_eta, eta, rtol=2e-13, atol=2e-13)
        self.assertAlmostEqual(nll, float(np.sum(values)), places=10)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(hessian, expected_hessian, rtol=2e-12, atol=2e-12)

        value_only = resident_poisson_objective(
            x,
            beta,
            exposure,
            event,
            device="cuda:0",
            matrix_token=new_derivative_token(),
            compute_moments=False,
            return_eta=False,
        )
        self.assertIsNotNone(value_only)
        assert value_only is not None
        self.assertAlmostEqual(value_only[0], float(np.sum(values)), places=10)
        self.assertIsNone(value_only[1])
        self.assertIsNone(value_only[2])
        self.assertIsNone(value_only[3])

    @unittest.skipUnless(cuda_available(), "CUDA objective solver is unavailable")
    def test_resident_cloglog_objective_matches_host_reference(self) -> None:
        rng = np.random.default_rng(20260729)
        x = np.ascontiguousarray(rng.normal(size=(5003, 7)))
        beta = rng.normal(scale=0.2, size=7)
        noevent = rng.uniform(0.01, 2.0, size=len(x))
        event = rng.binomial(1, 0.12, size=len(x)).astype(np.float64)
        eta = x @ beta
        values, first, second = loss_rows(
            eta,
            likelihood="first_event_cloglog",
            exposure_weight=noevent,
            noevent_weight=noevent,
            event_weight=event,
        )
        expected_gradient, expected_hessian = moments(x, first, second, device="cpu")
        actual = resident_cloglog_objective(
            x,
            beta,
            noevent,
            event,
            device="cuda:0",
            matrix_token=new_derivative_token(),
            compute_moments=True,
            return_eta=True,
        )
        self.assertIsNotNone(actual)
        assert actual is not None
        nll, actual_eta, gradient, hessian = actual
        np.testing.assert_allclose(actual_eta, eta, rtol=2e-13, atol=2e-13)
        self.assertAlmostEqual(nll, float(np.sum(values)), places=10)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(hessian, expected_hessian, rtol=2e-12, atol=2e-12)

    @unittest.skipUnless(cuda_available(), "CUDA objective solver is unavailable")
    def test_sharded_cloglog_evaluator_matches_host_reference(self) -> None:
        rng = np.random.default_rng(20260730)
        x = np.ascontiguousarray(rng.normal(size=(10003, 5)))
        beta = rng.normal(scale=0.15, size=5)
        noevent = rng.uniform(0.2, 1.8, size=len(x))
        event = rng.binomial(1, 0.08, size=len(x)).astype(np.float64)
        matrix = SimpleNamespace(
            x=x,
            exposure_weight=noevent,
            noevent_weight=noevent,
            event_weight=event,
            dimension=x.shape[1],
        )
        values, first, second = loss_rows(
            x @ beta,
            likelihood="first_event_cloglog",
            exposure_weight=noevent,
            noevent_weight=noevent,
            event_weight=event,
        )
        expected_gradient, expected_hessian = moments(x, first, second, device="cpu")
        with ProjectedDesignEvaluator(
            matrix,
            likelihood="first_event_cloglog",
            devices=("cuda:0", "cuda:1"),
        ) as evaluator:
            self.assertEqual(evaluator.shard_count, 2)
            nll, gradient, hessian = evaluator.objective(beta)
            self.assertAlmostEqual(evaluator.value(beta), float(np.sum(values)), places=10)
        self.assertAlmostEqual(nll, float(np.sum(values)), places=10)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(hessian, expected_hessian, rtol=2e-12, atol=2e-12)

    def test_lossless_unique_design_bypass_preserves_rows_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data")
            engine = ResponseEngine(
                data,
                lag=3,
                knot_count=2,
                cache_bytes=1024**3,
            )
            rows = 70_000
            index = np.arange(rows, dtype=np.float64)
            design = np.column_stack((np.ones(rows), index / rows))
            exposure = 1.0 + (index % 3.0)
            noevent = exposure.copy()
            event = (index % 17.0 == 0.0).astype(np.float64)
            x, e, n, y, groups = engine._aggregate_or_keep_design_rows(
                design.copy(),
                exposure.copy(),
                noevent.copy(),
                event.copy(),
            )
            np.testing.assert_array_equal(x, design)
            np.testing.assert_array_equal(e, exposure)
            np.testing.assert_array_equal(n, noevent)
            np.testing.assert_array_equal(y, event)
            np.testing.assert_array_equal(groups, np.arange(rows))

    def test_constraint_generated_recession_matches_complete_dense_lp(self) -> None:
        rng = np.random.default_rng(20260723)
        for likelihood in ("poisson", "first_event_cloglog"):
            for _ in range(12):
                x = rng.normal(size=(24, 5))
                exposure = np.ones(24, dtype=np.float64)
                event = (rng.random(24) < 0.30).astype(np.float64)
                noevent = (
                    np.ones(24, dtype=np.float64)
                    if likelihood == "poisson"
                    else (rng.random(24) < 0.70).astype(np.float64)
                )
                if likelihood == "poisson":
                    constraints = [x, -x[event > 0.0]]
                    improving = event == 0.0
                    objective = x[improving].T @ exposure[improving]
                else:
                    constraints = [
                        x[noevent > 0.0],
                        -x[event > 0.0],
                    ]
                    noevent_only = (noevent > 0.0) & (event == 0.0)
                    event_only = (event > 0.0) & (noevent == 0.0)
                    objective = np.zeros(x.shape[1], dtype=np.float64)
                    objective += x[noevent_only].T @ noevent[noevent_only]
                    objective -= x[event_only].T @ event[event_only]
                dense = np.vstack([value for value in constraints if len(value)])
                reference = linprog(
                    objective,
                    A_ub=dense,
                    b_ub=np.zeros(len(dense), dtype=np.float64),
                    bounds=[(-1.0, 1.0)] * 2 + [(0.0, 1.0)] * 3,
                    method="highs",
                    options={
                        "presolve": True,
                        "primal_feasibility_tolerance": 1.0e-9,
                        "dual_feasibility_tolerance": 1.0e-9,
                    },
                )
                self.assertTrue(reference.success)
                scale = max(1.0, float(np.linalg.norm(objective, ord=1)))
                expected = bool(reference.fun < -1.0e-10 * scale)
                actual = _general_recession_design(
                    x,
                    exposure,
                    noevent,
                    event,
                    2,
                    likelihood,
                )
                self.assertEqual(actual, expected)

    def test_discovery_score_uses_common_exact_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            pair = next(
                rule
                for rule in optimizer.dictionary
                if rule.antecedent == (0, 1) and rule.sign > 0
            )
            record = optimizer.fit(Support.of((pair,)), empty)
            self.assertTrue(record.fit.converged, record.fit.message)
            scored = optimizer._attach_rule_score(record)
            self.assertIsNotNone(scored.closure_null_nll)
            reported_penalty = optimizer.objective.reported_branch_penalty(
                scored.support,
                len(scored.support.rules) * config.knot_count,
            )
            expected = support_score(
                baseline_nll=float(scored.closure_null_nll),
                fit_nll=scored.fit.nll,
                penalty=reported_penalty,
            )
            self.assertAlmostEqual(scored.discovery_score, expected, places=10)
            null_matrix, null_fit = optimizer.fit_fixed(
                Support(()),
                tuple(scored.matrix.closure),
                source=scored,
            )
            self.assertEqual(null_matrix.support, Support(()))
            self.assertEqual(null_matrix.closure, scored.matrix.closure)
            self.assertTrue(null_fit.converged, null_fit.message)
            self.assertAlmostEqual(
                float(scored.closure_null_nll), null_fit.nll, places=10
            )

    def test_certified_inexact_add_advances_without_child_refit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 500)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=64 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                search_mode="fast_block_score",
                pricing_workers=2,
                exact_workers=1,
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            candidates = []
            for rule in optimizer.dictionary:
                record = optimizer.fit(Support.of((rule,)), empty)
                if record.fit.converged:
                    candidates.append((optimizer._attach_rule_score(record), rule))
            exact, rule = max(
                candidates,
                key=lambda item: item[0].discovery_score,
            )
            self.assertGreater(exact.discovery_score, 0.0)
            provisional = replace(
                exact,
                fit=replace(exact.fit, converged=False),
                rule_score=None,
                closure_null_nll=None,
                rule_score_upper=None,
            )
            optimizer.records.pop(exact.support, None)
            optimizer._stored_records.pop(exact.support, None)
            accepted, branch = optimizer._exact_add_branch_validation(
                empty, provisional, rule
            )
            self.assertIsNotNone(accepted)
            assert accepted is not None
            self.assertFalse(accepted.fit.converged)
            self.assertGreater(branch, 0.0)
            self.assertGreater(
                accepted.discovery_score,
                empty.discovery_upper_score,
            )
            self.assertGreaterEqual(
                accepted.discovery_upper_score,
                accepted.discovery_score,
            )

    def test_natural_dual_reuses_exact_kkt_derivatives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            matrix = engine.model_matrix(
                context, Support.of((RuleIdentity((0,), 0, 1),))
            )
            fit = fit_model_matrix(
                matrix,
                likelihood=data.likelihood,
                tolerance=1e-9,
                max_iter=150,
            )
            self.assertTrue(fit.converged, fit.message)
            nll, gradient, _, eta = _objective(
                matrix, data.likelihood, fit.coefficients
            )
            _, dual, _ = loss_rows(
                eta,
                likelihood=data.likelihood,
                exposure_weight=matrix.exposure_weight,
                noevent_weight=matrix.noevent_weight,
                event_weight=matrix.event_weight,
            )
            certificate = natural_dual_certificate(
                dual,
                gradient,
                free_dimension=matrix.free_dimension,
                likelihood=data.likelihood,
                exposure_weight=matrix.exposure_weight,
                noevent_weight=matrix.noevent_weight,
                event_weight=matrix.event_weight,
                tolerance=1e-8,
            )
            self.assertTrue(certificate.feasible)
            self.assertLessEqual(certificate.nll_lower_bound, nll + 1e-7)
            self.assertAlmostEqual(certificate.nll_lower_bound, nll, places=6)

    def test_conditional_one_step_intervals_contain_exact_add_and_drop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            a = RuleIdentity((0,), 0, 1)
            b = RuleIdentity((1,), 0, 1)
            current = optimizer.fit(Support.of((a,)), empty)

            add_support = current.support.add(b)
            add = optimizer._conditional_one_step(current, add_support)
            self.assertFalse(add.record.fit.converged)
            self.assertNotIn(add_support, optimizer._stored_records)
            exact_add = optimizer.fit(add_support, current)
            self.assertTrue(exact_add.fit.converged, exact_add.fit.message)
            slack = 1e-6 * max(1.0, abs(exact_add.score))
            self.assertLessEqual(add.lower_score, exact_add.score + slack)
            self.assertGreaterEqual(add.upper_score + slack, exact_add.score)

            drop_support = exact_add.support.drop(a)
            drop = optimizer._conditional_one_step(exact_add, drop_support)
            self.assertFalse(drop.record.fit.converged)
            self.assertNotIn(drop_support, optimizer._stored_records)
            exact_drop = optimizer.fit(drop_support, exact_add)
            self.assertTrue(exact_drop.fit.converged, exact_drop.fit.message)
            slack = 1e-6 * max(1.0, abs(exact_drop.score))
            self.assertLessEqual(drop.lower_score, exact_drop.score + slack)
            self.assertGreaterEqual(drop.upper_score + slack, exact_drop.score)

    def test_search_caches_only_exact_terminal_corrections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            result = optimizer.search()
            self.assertGreater(result.diagnostics.conditional_one_steps, 0)
            self.assertGreater(result.diagnostics.conditional_full_refits_avoided, 0)
            self.assertTrue(all(item.fit.converged for item in result.terminals))
            self.assertTrue(
                all(item.fit.converged for item in result.family),
                "a conditional path state leaked into the certification family",
            )
            self.assertTrue(
                all("terminal_block_audit" in path for path in result.paths)
            )
            for path in result.paths:
                for item in path["terminal_block_audit"]:
                    self.assertIn(item["move"], {"add", "drop"})
                    self.assertFalse(item["total_mdl_and_branch_improve"])

    def test_pricing_cache_is_owned_by_matrix_and_coefficient_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            support = Support.of((RuleIdentity((0,), 0, 1),))
            record = optimizer.fit(support, optimizer.records[Support(())])
            self.assertTrue(record.fit.converged)
            optimizer._group_eta(record)
            fresh = replace(
                record,
                matrix=optimizer.engine.model_matrix(optimizer.context, support),
            )
            optimizer._row_eta_state[support] = np.full(
                len(record.matrix.x), np.nan, dtype=np.float64
            )
            optimizer._row_eta_cache_bytes = optimizer._row_eta_state[support].nbytes
            eta = optimizer._group_eta(fresh)
            self.assertEqual(eta.shape, (len(fresh.matrix.x),))
            self.assertTrue(np.all(np.isfinite(eta)))

    def test_vectorized_axis_recession_matches_coordinate_reference(self) -> None:
        rng = np.random.default_rng(781)

        def reference(matrix: object, likelihood: str) -> bool:
            for index in range(matrix.dimension):
                signs = (-1.0, 1.0) if index < matrix.free_dimension else (1.0,)
                for sign in signs:
                    direction = sign * matrix.x[:, index]
                    if not np.any(direction):
                        continue
                    event = matrix.event_weight > 0
                    if likelihood == "poisson":
                        valid = np.all(direction <= 1.0e-14) and np.all(
                            np.abs(direction[event]) <= 1.0e-14
                        )
                        strict = np.any(direction < -1.0e-14)
                    else:
                        noevent = matrix.noevent_weight > 0
                        valid = np.all(direction[noevent] <= 1.0e-14) and np.all(
                            direction[event] >= -1.0e-14
                        )
                        strict = np.any(direction[noevent] < -1.0e-14) or np.any(
                            direction[event] > 1.0e-14
                        )
                    if valid and strict:
                        return True
            return False

        for likelihood in ("poisson", "first_event_cloglog"):
            for _ in range(40):
                x = rng.choice((-1.0, 0.0, 1.0), size=(113, 9), p=(0.1, 0.8, 0.1))
                event = rng.integers(0, 2, size=113).astype(np.float64)
                noevent = rng.integers(0, 2, size=113).astype(np.float64)
                matrix = SimpleNamespace(
                    x=x,
                    dimension=x.shape[1],
                    free_dimension=int(rng.integers(0, x.shape[1] + 1)),
                    event_weight=event,
                    noevent_weight=noevent,
                )
                self.assertEqual(
                    _axis_recession(matrix, likelihood),
                    reference(matrix, likelihood),
                )

    def test_native_bounded_span_order_is_stable_and_exact(self) -> None:
        spans = np.array([3, 1, 5, 1, 0, 3, 2], dtype=np.int64)
        actual = bounded_span_order(spans, 3)
        if actual is None:
            self.skipTest("compiled CPU operators unavailable")
        admitted = np.flatnonzero(spans <= 3)
        expected = admitted[np.argsort(spans[admitted], kind="stable")]
        np.testing.assert_array_equal(actual, expected)

    def test_native_sorted_unique_union_matches_numpy(self) -> None:
        parts = [
            np.array([1, 3, 7, 11], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([0, 3, 4, 11], dtype=np.int64),
            np.array([2, 9], dtype=np.int64),
        ]
        actual = sorted_unique_union(parts)
        if actual is None:
            self.skipTest("compiled CPU operators unavailable")
        expected = np.unique(np.concatenate(parts))
        np.testing.assert_array_equal(actual, expected)

    def test_cuda_ragged_sparse_moments_match_dense_reference(self) -> None:
        if not cuda_available():
            self.skipTest("CUDA pricing operators unavailable")
        rng = np.random.default_rng(19)
        candidates = []
        references = []
        for _ in range(3):
            first_grid = rng.normal(size=80)
            second_grid = rng.uniform(0.1, 2.0, size=80)
            blocks = []
            dense = np.zeros((80, 12), dtype=np.float64)
            for block_index in range(3):
                rows = np.sort(
                    rng.choice(80, size=18 + block_index * 3, replace=False)
                ).astype(np.int64)
                values = rng.normal(size=(len(rows), 4))
                dense[rows, block_index * 4 : (block_index + 1) * 4] = values
                blocks.append(
                    (
                        rows,
                        values,
                        first_grid[rows],
                        second_grid[rows],
                    )
                )
            candidates.append(tuple(blocks))
            references.append(
                (
                    dense.T @ first_grid,
                    dense.T @ (second_grid[:, None] * dense),
                    dense.T @ second_grid,
                )
            )
        actual = sparse_moments_batch(candidates, device="cuda:0")
        self.assertIsNotNone(actual)
        for index, reference in enumerate(references):
            np.testing.assert_allclose(actual[0][index], reference[0], atol=1e-12)
            np.testing.assert_allclose(actual[1][index], reference[1], atol=1e-12)
            np.testing.assert_allclose(actual[2][index], reference[2], atol=1e-12)

    def test_cuda_indexed_sparse_moments_match_dense_reference(self) -> None:
        if not cuda_available():
            self.skipTest("CUDA pricing operators unavailable")
        rng = np.random.default_rng(191)
        first = rng.normal(size=80)
        second = rng.uniform(0.1, 2.0, size=80)
        candidates = []
        references = []
        for _ in range(3):
            blocks = []
            dense = np.zeros((80, 12), dtype=np.float64)
            for block_index in range(3):
                rows = np.sort(
                    rng.choice(80, size=18 + block_index * 3, replace=False)
                ).astype(np.int64)
                values = rng.normal(size=(len(rows), 4))
                dense[rows, block_index * 4 : (block_index + 1) * 4] = values
                blocks.append((rows, values))
            candidates.append(tuple(blocks))
            references.append(
                (
                    dense.T @ first,
                    dense.T @ (second[:, None] * dense),
                    dense.T @ second,
                )
            )
        derivative_token = new_derivative_token()
        actual = sparse_moments_indexed_batch(
            candidates,
            first,
            second,
            device="cuda:0",
            derivative_token=derivative_token,
        )
        self.assertIsNotNone(actual)
        cached = sparse_moments_indexed_batch(
            candidates,
            first,
            second,
            device="cuda:0",
            derivative_token=derivative_token,
        )
        self.assertIsNotNone(cached)
        uncached = sparse_moments_indexed_batch(
            candidates, first + 0.25, second + 0.5, device="cuda:0"
        )
        self.assertIsNotNone(uncached)
        restored = sparse_moments_indexed_batch(
            candidates,
            first,
            second,
            device="cuda:0",
            derivative_token=derivative_token,
        )
        self.assertIsNotNone(restored)
        for index, reference in enumerate(references):
            np.testing.assert_allclose(actual[0][index], reference[0], atol=1e-12)
            np.testing.assert_allclose(actual[1][index], reference[1], atol=1e-12)
            np.testing.assert_allclose(actual[2][index], reference[2], atol=1e-12)
        for first_value, cached_value in zip(actual, cached, strict=True):
            np.testing.assert_array_equal(first_value, cached_value)
        for first_value, restored_value in zip(actual, restored, strict=True):
            np.testing.assert_array_equal(first_value, restored_value)

    def test_sparse_grid_derivatives_match_dense_weights(self) -> None:
        eta = np.linspace(-4.0, 2.0, 31)
        rows = np.asarray([1, 7, 19, 30], dtype=np.int64)
        counts = np.asarray([1.0, 1.0, 1.0, 1.0])
        for likelihood, exposure in (("poisson", 0.25), ("first_event_cloglog", 1.0)):
            event = np.zeros(len(eta), dtype=np.float64)
            event[rows] = counts
            exposure_weight = np.full(len(eta), exposure, dtype=np.float64)
            noevent = (
                exposure_weight - event
                if likelihood == "first_event_cloglog"
                else exposure_weight
            )
            dense = loss_rows(
                eta,
                likelihood=likelihood,
                exposure_weight=exposure_weight,
                noevent_weight=noevent,
                event_weight=event,
            )
            derivatives = loss_grid_sparse_event_derivatives(
                eta,
                likelihood=likelihood,
                exposure=exposure,
                event_rows=rows,
                event_counts=counts,
            )
            for expected, actual in zip(dense[1:], derivatives, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)

    def test_poisson_offset_dual_certificate_matches_exact_optimum(self) -> None:
        rows = 60
        index = np.arange(rows)
        design = np.column_stack(
            (
                np.where(index % 2 == 0, 1.0, -1.0),
                0.2 + (index % 5) / 5.0,
            )
        )
        offset = -2.5 + 0.01 * index
        event = np.zeros(rows, dtype=np.float64)
        event[[1, 5, 8, 13, 21, 34, 55]] = 1.0
        exposure = np.ones(rows, dtype=np.float64)
        exact = fit_offset_design(
            design,
            offset,
            exposure,
            exposure,
            event,
            likelihood="poisson",
            free_dimension=1,
            tolerance=1e-8,
            max_iter=200,
        )
        self.assertTrue(exact.converged, exact.message)
        certificate = offset_dual_certificate(
            design,
            offset,
            exposure,
            exposure,
            event,
            likelihood="poisson",
            beta=exact.coefficients,
            free_dimension=1,
            tolerance=1e-7,
            max_iter=500,
        )
        self.assertTrue(certificate.feasible)
        self.assertLessEqual(certificate.nll_lower_bound, exact.nll + 1e-7)
        self.assertAlmostEqual(certificate.nll_lower_bound, exact.nll, places=6)

    def test_incremental_support_matrix_matches_fresh_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            for likelihood in ("first_event_cloglog", "poisson"):
                variant = replace(data, likelihood=likelihood)
                context = Context.make(
                    variant, np.arange(variant.n_entities, dtype=np.int32)
                )
                engine = ResponseEngine(
                    variant, lag=3, knot_count=2, cache_bytes=32 * 1024**2
                )
                source_support = Support.of((RuleIdentity((0,), 0, 1),))
                target_support = source_support.add(RuleIdentity((0, 1), 2, -1))
                source = engine.model_matrix(context, source_support)
                incremental = engine.extend_model_matrix(
                    context, target_support, source
                )
                fresh = engine.model_matrix(context, target_support)
                np.testing.assert_array_equal(
                    incremental.active_rows, fresh.active_rows
                )
                np.testing.assert_allclose(
                    incremental.x[incremental.active_design_groups],
                    fresh.x[fresh.active_design_groups],
                    rtol=0.0,
                    atol=0.0,
                )
                incremental_canonical = aggregate_design_rows(
                    incremental.x,
                    incremental.exposure_weight,
                    incremental.noevent_weight,
                    incremental.event_weight,
                )
                fresh_canonical = aggregate_design_rows(
                    fresh.x,
                    fresh.exposure_weight,
                    fresh.noevent_weight,
                    fresh.event_weight,
                )
                # Sufficient-statistic groups are unordered.  The touched-only
                # incremental builder preserves their exact values and weights
                # without reproducing the native hash table's insertion order.
                incremental_order = np.lexsort(np.flipud(incremental_canonical[0].T))
                fresh_order = np.lexsort(np.flipud(fresh_canonical[0].T))
                for left, right in zip(
                    (array[incremental_order] for array in incremental_canonical),
                    (array[fresh_order] for array in fresh_canonical),
                    strict=True,
                ):
                    np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)

    def test_incremental_negative_residual_fails_open_to_exact_rebuild(self) -> None:
        """An invalid touched quotient must not abort or clip the likelihood."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            context = Context.make(
                data, np.arange(data.n_entities, dtype=np.int32)
            )
            engine = ResponseEngine(
                data, lag=3, knot_count=2, cache_bytes=32 * 1024**2
            )
            source_support = Support.of((RuleIdentity((0,), 0, 1),))
            target_support = source_support.add(RuleIdentity((1,), 0, -1))
            source = engine.model_matrix(context, source_support)

            def invalid_touched_quotient(*args: object):
                residuals = [
                    np.array(value, dtype=np.float64, copy=True)
                    for value in subtract_group_weights(*args)
                ]
                residuals[0][0] = -1.0
                return tuple(residuals)

            with mock.patch(
                "crbstpp.response.subtract_group_weights",
                side_effect=invalid_touched_quotient,
            ):
                incremental = engine.extend_model_matrix(
                    context, target_support, source
                )
            fresh = engine.model_matrix(context, target_support)
            incremental_canonical = aggregate_design_rows(
                incremental.x,
                incremental.exposure_weight,
                incremental.noevent_weight,
                incremental.event_weight,
            )
            fresh_canonical = aggregate_design_rows(
                fresh.x,
                fresh.exposure_weight,
                fresh.noevent_weight,
                fresh.event_weight,
            )
            incremental_order = np.lexsort(np.flipud(incremental_canonical[0].T))
            fresh_order = np.lexsort(np.flipud(fresh_canonical[0].T))
            for left, right in zip(
                (array[incremental_order] for array in incremental_canonical),
                (array[fresh_order] for array in fresh_canonical),
                strict=True,
            ):
                np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)

    def test_incremental_unaggregated_parent_uses_baseline_mass_group(self) -> None:
        """Duplicate intercept rows must not force a full-union Add rebuild."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            context = Context.make(
                data, np.arange(data.n_entities, dtype=np.int32)
            )
            engine = ResponseEngine(
                data, lag=3, knot_count=2, cache_bytes=32 * 1024**2
            )
            source_support = Support.of((RuleIdentity((0,), 0, 1),))
            target_support = source_support.add(RuleIdentity((1,), 0, -1))
            with mock.patch.object(
                engine, "_keep_unaggregated_design", return_value=True
            ):
                source = engine.model_matrix(context, source_support)
                original = engine._finalize_touched_extended_model_matrix
                incremental_results: list[bool] = []

                def capture_incremental(**kwargs: object):
                    result = original(**kwargs)
                    incremental_results.append(result is not None)
                    return result

                with mock.patch.object(
                    engine,
                    "_finalize_touched_extended_model_matrix",
                    side_effect=capture_incremental,
                ):
                    incremental = engine.extend_model_matrix(
                        context, target_support, source
                    )
                fresh = engine.model_matrix(context, target_support)

            self.assertEqual(incremental_results, [True])
            incremental_canonical = aggregate_design_rows(
                incremental.x,
                incremental.exposure_weight,
                incremental.noevent_weight,
                incremental.event_weight,
            )
            fresh_canonical = aggregate_design_rows(
                fresh.x,
                fresh.exposure_weight,
                fresh.noevent_weight,
                fresh.event_weight,
            )
            incremental_order = np.lexsort(np.flipud(incremental_canonical[0].T))
            fresh_order = np.lexsort(np.flipud(fresh_canonical[0].T))
            for left, right in zip(
                (array[incremental_order] for array in incremental_canonical),
                (array[fresh_order] for array in fresh_canonical),
                strict=True,
            ):
                np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)

    def test_incremental_retains_zero_exposure_active_parent_group(self) -> None:
        """Masked untouched rows must retain their zero-mass parent metadata."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            n_entities = 12
            entities = pd.DataFrame(
                {
                    "entity_id": [f"e{index}" for index in range(n_entities)],
                    "start_time": np.zeros(n_entities, dtype=np.int64),
                    "end_time": np.full(n_entities, 5, dtype=np.int64),
                    "baseline_origin": np.zeros(n_entities, dtype=np.int64),
                    "split_group": np.zeros(n_entities, dtype=np.int64),
                }
            )
            events = pd.DataFrame(
                [(entity, 0, 0) for entity in range(8)]
                + [(entity, 1, 1) for entity in range(4)],
                columns=["entity_code", "time", "predicate_code"],
            )
            targets = pd.DataFrame(
                [(entity, 4, 1) for entity in range(4)],
                columns=["entity_code", "time", "multiplicity"],
            )
            baseline_cells = pd.DataFrame(
                [
                    (
                        entity,
                        time,
                        0,
                        0.0
                        if entity in range(4, 8) and time in (2, 3)
                        else 1.0,
                    )
                    for entity in range(n_entities)
                    for time in range(6)
                ],
                columns=["entity_code", "time", "baseline_stratum", "exposure"],
            )
            write_dataset(
                root,
                entities=entities,
                events=events,
                targets=targets,
                baseline_cells=baseline_cells,
                predicate_names=("A", "B"),
                likelihood="first_event_cloglog",
                time_unit="month",
                adverse_event_name="target",
                f0_contract={
                    "dynamic_predicates": True,
                    "outcome_blind_predicate_construction": True,
                    "direct_target_proxy_excluded_from_reported_dictionary": True,
                    "strict_future_effect_required": True,
                    "atomic_predicates": True,
                    "primitive_event_provenance": True,
                    "independent_certification_units": True,
                },
                provenance={"generator": "zero-exposure-parent-test"},
            )
            data = Dataset.load(root)
            context = Context.make(
                data, np.arange(data.n_entities, dtype=np.int32)
            )
            engine = ResponseEngine(
                data, lag=3, knot_count=2, cache_bytes=32 * 1024**2
            )
            source_support = Support.of((RuleIdentity((0,), 0, 1),))
            target_support = source_support.add(RuleIdentity((1,), 0, 1))
            source = engine.model_matrix(context, source_support)
            incremental = engine.extend_model_matrix(
                context, target_support, source
            )
            fresh = engine.model_matrix(context, target_support)

            np.testing.assert_array_equal(
                incremental.active_rows, fresh.active_rows
            )
            np.testing.assert_allclose(
                incremental.x[incremental.active_design_groups],
                fresh.x[fresh.active_design_groups],
                rtol=0.0,
                atol=0.0,
            )
            incremental_canonical = aggregate_design_rows(
                incremental.x,
                incremental.exposure_weight,
                incremental.noevent_weight,
                incremental.event_weight,
            )
            fresh_canonical = aggregate_design_rows(
                fresh.x,
                fresh.exposure_weight,
                fresh.noevent_weight,
                fresh.event_weight,
            )
            incremental_order = np.lexsort(
                np.flipud(incremental_canonical[0].T)
            )
            fresh_order = np.lexsort(np.flipud(fresh_canonical[0].T))
            for left, right in zip(
                (array[incremental_order] for array in incremental_canonical),
                (array[fresh_order] for array in fresh_canonical),
                strict=True,
            ):
                np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)

    def test_compiled_nonnegative_quadratic_batch_matches_reference(self) -> None:
        rng = np.random.default_rng(314)
        gradients = rng.normal(size=(257, 4))
        factors = rng.normal(size=(257, 4, 4))
        hessians = np.einsum("bki,bkj->bij", factors, factors)
        hessians += 0.1 * np.eye(4)[None, :, :]
        compiled = nonnegative_quadratic_gains(gradients, hessians)
        self.assertIsNotNone(compiled)
        reference = np.asarray(
            [
                _nonnegative_quadratic_gain(gradient, hessian)
                for gradient, hessian in zip(gradients, hessians, strict=True)
            ]
        )
        np.testing.assert_allclose(compiled, reference, rtol=1e-12, atol=1e-12)

    def test_exact_batched_quadratic_gain_preserves_singular_fallback(self) -> None:
        rng = np.random.default_rng(2718)
        gradients = rng.normal(size=(65, 4))
        factors = rng.normal(size=(65, 4, 4))
        hessians = np.einsum("bki,bkj->bij", factors, factors)
        hessians[:5, -1] = hessians[:5, 0]
        hessians[:5, :, -1] = hessians[:5, :, 0]
        reference = np.asarray(
            [
                _nonnegative_quadratic_gain(gradient, hessian)
                for gradient, hessian in zip(gradients, hessians, strict=True)
            ]
        )
        actual = _batched_nonnegative_quadratic_gain(gradients, hessians)
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)

    def test_nonnegative_quadratic_solution_attains_reported_gain(self) -> None:
        gradient = np.asarray([-2.0, 0.5, -1.0, 0.2])
        hessian = np.asarray(
            [
                [2.0, 0.2, 0.1, 0.0],
                [0.2, 1.5, 0.0, 0.1],
                [0.1, 0.0, 1.0, 0.1],
                [0.0, 0.1, 0.1, 1.2],
            ]
        )
        gain, direction = _nonnegative_quadratic_solution(gradient, hessian)
        self.assertTrue(np.all(direction >= 0.0))
        attained = -gradient @ direction - 0.5 * direction @ hessian @ direction
        self.assertAlmostEqual(gain, attained)
        self.assertAlmostEqual(gain, _nonnegative_quadratic_gain(gradient, hessian))

    def test_offset_warm_start_preserves_exact_optimum(self) -> None:
        index = np.arange(80)
        design = np.column_stack((np.ones(80), 0.2 + (index % 7) / 7.0))
        offset = -2.0 + 0.005 * index
        event = np.zeros(80, dtype=np.float64)
        event[[2, 7, 11, 19, 31, 47, 63, 71]] = 1.0
        exposure = np.ones(80, dtype=np.float64)
        cold = fit_offset_design(
            design,
            offset,
            exposure,
            exposure,
            event,
            likelihood="poisson",
            free_dimension=1,
            tolerance=1e-9,
            max_iter=200,
        )
        warm = fit_offset_design(
            design,
            offset,
            exposure,
            exposure,
            event,
            likelihood="poisson",
            free_dimension=1,
            tolerance=1e-9,
            max_iter=200,
            warm_start=cold.coefficients,
        )
        self.assertTrue(cold.converged, cold.message)
        self.assertTrue(warm.converged, warm.message)
        np.testing.assert_allclose(warm.coefficients, cold.coefficients, atol=1e-9)
        self.assertAlmostEqual(warm.nll, cold.nll, places=10)

    def test_offset_iteration_window_continuation_is_exact(self) -> None:
        index = np.arange(80)
        design = np.column_stack((np.ones(80), 0.2 + (index % 7) / 7.0))
        offset = -2.0 + 0.005 * index
        event = np.zeros(80, dtype=np.float64)
        event[[2, 7, 11, 19, 31, 47, 63, 71]] = 1.0
        exposure = np.ones(80, dtype=np.float64)
        reference = fit_offset_design(
            design,
            offset,
            exposure,
            exposure,
            event,
            likelihood="poisson",
            free_dimension=1,
            tolerance=1e-9,
            max_iter=200,
        )
        continued = fit_offset_design_continued(
            design,
            offset,
            exposure,
            exposure,
            event,
            likelihood="poisson",
            free_dimension=1,
            tolerance=1e-9,
            max_iter=1,
        )
        self.assertTrue(reference.converged, reference.message)
        self.assertTrue(continued.converged, continued.message)
        self.assertGreater(continued.iterations, 1)
        np.testing.assert_allclose(
            continued.coefficients, reference.coefficients, atol=1e-9
        )
        self.assertAlmostEqual(continued.nll, reference.nll, places=10)

    def test_lossless_design_aggregation_preserves_objective_and_moments(self) -> None:
        rng = np.random.default_rng(82)
        prototypes = rng.normal(size=(17, 6))
        prototypes[0, 0] = -0.0
        assignment = rng.integers(0, len(prototypes), size=400)
        assignment[:2] = 0
        x = prototypes[assignment]
        x[0, 0] = 0.0
        exposure = rng.uniform(0.1, 2.0, size=len(x))
        noevent = rng.uniform(0.1, 2.0, size=len(x))
        event = rng.integers(0, 2, size=len(x)).astype(np.float64)
        beta = rng.normal(size=x.shape[1])
        aggregated = aggregate_design_rows(x, exposure, noevent, event)
        grouped = aggregate_design_rows_with_groups(x, exposure, noevent, event)
        self.assertLessEqual(len(aggregated[0]), len(prototypes))
        np.testing.assert_array_equal(grouped[0][grouped[4]], x)
        for left, right in zip(aggregated, grouped[:4], strict=True):
            np.testing.assert_array_equal(left, right)
        for likelihood in ("poisson", "first_event_cloglog"):
            original_rows, original_first, original_second = loss_rows(
                x @ beta,
                likelihood=likelihood,
                exposure_weight=exposure,
                noevent_weight=noevent,
                event_weight=event,
            )
            reduced_rows, reduced_first, reduced_second = loss_rows(
                aggregated[0] @ beta,
                likelihood=likelihood,
                exposure_weight=aggregated[1],
                noevent_weight=aggregated[2],
                event_weight=aggregated[3],
            )
            original_moments = moments(x, original_first, original_second)
            reduced_moments = moments(aggregated[0], reduced_first, reduced_second)
            self.assertAlmostEqual(
                float(np.sum(original_rows)), float(np.sum(reduced_rows)), places=11
            )
            np.testing.assert_allclose(
                reduced_moments[0], original_moments[0], rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                reduced_moments[1], original_moments[1], rtol=1e-12, atol=1e-12
            )

    def test_parallel_quotient_aggregation_matches_generic_signature(self) -> None:
        rng = np.random.default_rng(1908)
        prototypes = rng.normal(size=(29, 4))
        prototypes[0, 0] = -0.0
        assignment = rng.integers(0, len(prototypes), size=5000)
        values = np.ascontiguousarray(prototypes[assignment])
        values[0, 0] = 0.0
        old_group_prototypes = rng.integers(0, 13, size=len(prototypes))
        old_groups = np.ascontiguousarray(old_group_prototypes[assignment])
        exposure = rng.uniform(0.1, 2.0, size=len(values))
        noevent = rng.uniform(0.1, 2.0, size=len(values))
        event = rng.integers(0, 2, size=len(values)).astype(np.float64)

        signatures = np.column_stack((old_groups, values))
        expected = aggregate_design_rows(
            signatures, exposure, noevent, event, copy_input=True
        )[1:]
        actual = aggregate_quotient_rows(
            old_groups, values, exposure, noevent, event
        )
        bounded_workers = aggregate_quotient_rows(
            old_groups,
            values,
            exposure,
            noevent,
            event,
            worker_count=1,
        )

        def ordered(weights: tuple[np.ndarray, ...]) -> np.ndarray:
            table = np.column_stack(weights)
            order = np.lexsort(tuple(table[:, column] for column in reversed(range(3))))
            return table[order]

        self.assertEqual(len(actual[0]), len(expected[0]))
        np.testing.assert_allclose(
            ordered(actual[:3]), ordered(expected), rtol=2e-15, atol=2e-15
        )
        np.testing.assert_allclose(
            ordered(bounded_workers[:3]),
            ordered(expected),
            rtol=2e-15,
            atol=2e-15,
        )
        group_count = int(old_groups.max(initial=-1)) + 1
        for observed, weights in zip(
            actual[3:], (exposure, noevent, event), strict=True
        ):
            np.testing.assert_allclose(
                observed,
                np.bincount(old_groups, weights=weights, minlength=group_count),
                rtol=2e-15,
                atol=2e-15,
            )

    def test_safe_standalone_window_race_matches_exhaustive_exact_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                search_mode="safe_column_generation",
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                exact_workers=1,
                pricing_workers=2,
            )
            raced = SupportOptimizer(Context.make(data, fit_codes), config)
            reference = SupportOptimizer(Context.make(data, fit_codes), config)
            try:
                raced_empty = raced.records[Support(())]
                raced_records = raced._standalone_profiled_atoms(raced_empty)
                raced_by_antecedent = {
                    record.support.rules[0].antecedent: record
                    for record in raced_records
                }

                reference_empty = reference.records[Support(())]
                exact = reference.fit_many(
                    [Support.of((rule,)) for rule in reference.dictionary],
                    reference_empty,
                )
                positive = reference._branch_positive_standalone_records(
                    [
                        record
                        for record in exact
                        if record.fit.converged
                        and record.score > config.search_tolerance
                    ]
                )
                expected = {}
                for record in positive.values():
                    antecedent = record.support.rules[0].antecedent
                    incumbent = expected.get(antecedent)
                    if incumbent is None or (
                        record.score > incumbent.score + config.search_tolerance
                        or (
                            abs(record.score - incumbent.score)
                            <= config.search_tolerance
                            and record.support.rules < incumbent.support.rules
                        )
                    ):
                        expected[antecedent] = record
                self.assertEqual(set(raced_by_antecedent), set(expected))
                for antecedent, record in raced_by_antecedent.items():
                    exhaustive = expected[antecedent]
                    self.assertEqual(record.support, exhaustive.support)
                    self.assertAlmostEqual(record.score, exhaustive.score, places=8)
                # A valid safe bound need not screen on every finite sample.
                # Its non-negotiable property is exact-profile parity; when it
                # does not separate candidates it must fail open to the same
                # exact fits instead of changing the selected identity.
                self.assertLessEqual(
                    raced.diagnostics.exact_fits,
                    reference.diagnostics.exact_fits,
                )
            finally:
                raced.close()
                reference.close()

    def test_poisson_value_gradient_hessian_and_conjugate(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.normal(size=(31, 5))
        beta = rng.normal(size=5)
        exposure = rng.uniform(0.1, 2.0, size=31)
        event = rng.poisson(0.7, size=31).astype(float)
        eta = x @ beta
        value, first, second = loss_rows(
            eta,
            likelihood="poisson",
            exposure_weight=exposure,
            noevent_weight=exposure,
            event_weight=event,
        )
        epsilon = 2.0e-6
        direction = rng.normal(size=5)

        def objective(coefficient: np.ndarray) -> float:
            return float(
                np.sum(
                    loss_rows(
                        x @ coefficient,
                        likelihood="poisson",
                        exposure_weight=exposure,
                        noevent_weight=exposure,
                        event_weight=event,
                    )[0]
                )
            )

        numeric_first = (
            objective(beta + epsilon * direction)
            - objective(beta - epsilon * direction)
        ) / (2 * epsilon)
        analytic_first = float(direction @ (x.T @ first))
        self.assertAlmostEqual(numeric_first, analytic_first, places=5)
        numeric_second = (
            objective(beta + epsilon * direction)
            - 2 * float(np.sum(value))
            + objective(beta - epsilon * direction)
        ) / epsilon**2
        analytic_second = float(direction @ (x.T @ (second[:, None] * x)) @ direction)
        self.assertAlmostEqual(numeric_second, analytic_second, delta=2.0e-2)
        dual = first
        conjugate = conjugate_sum(
            dual,
            likelihood="poisson",
            exposure_weight=exposure,
            noevent_weight=exposure,
            event_weight=event,
        )
        self.assertAlmostEqual(
            float(np.sum(value)), float(dual @ eta - conjugate), places=9
        )

    def test_compiled_mixed_cloglog_conjugate_matches_reference(self) -> None:
        rng = np.random.default_rng(901)
        eta = rng.uniform(-8.0, 4.0, size=257)
        noevent = rng.uniform(0.01, 25.0, size=len(eta))
        event = rng.uniform(0.01, 4.0, size=len(eta))
        _, dual, _ = loss_rows(
            eta,
            likelihood="first_event_cloglog",
            exposure_weight=noevent + event,
            noevent_weight=noevent,
            event_weight=event,
        )
        compiled = cloglog_conjugate(dual, noevent, event)
        with mock.patch(
            "crbstpp.likelihood.fill_cloglog_mixed_conjugate",
            return_value=False,
        ):
            reference = cloglog_conjugate(dual, noevent, event)
        np.testing.assert_allclose(compiled, reference, rtol=2e-13, atol=2e-13)

    def test_compiled_cpu_and_cuda_moments_match_reference(self) -> None:
        self.assertTrue(cpu_available())
        rng = np.random.default_rng(10)
        x = rng.normal(size=(257, 11))
        first = rng.normal(size=257)
        second = rng.uniform(0.01, 3.0, size=257)
        reference_gradient = x.T @ first
        reference_hessian = x.T @ (second[:, None] * x)
        cpu_gradient, cpu_hessian = moments(x, first, second, device="cpu")
        np.testing.assert_allclose(
            cpu_gradient, reference_gradient, rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            cpu_hessian, reference_hessian, rtol=1e-13, atol=1e-13
        )
        if cuda_available():
            for device in ("cuda:0", "cuda:1"):
                eta = resident_eta(
                    x,
                    np.arange(x.shape[1], dtype=np.float64),
                    device=device,
                    matrix_token=id(x),
                )
                self.assertIsNotNone(eta)
                np.testing.assert_allclose(
                    eta,
                    x @ np.arange(x.shape[1], dtype=np.float64),
                    rtol=1e-13,
                    atol=1e-13,
                )
                gradient, hessian = moments(
                    x,
                    first,
                    second,
                    device=device,
                    matrix_token=id(x),
                )
                np.testing.assert_allclose(
                    gradient, reference_gradient, rtol=1e-13, atol=1e-13
                )
                np.testing.assert_allclose(
                    hessian, reference_hessian, rtol=1e-13, atol=1e-13
                )

    def test_batched_cpu_and_cuda_moments_match_scalar_reference(self) -> None:
        rng = np.random.default_rng(101)
        x = rng.normal(size=(5, 193, 4))
        first = rng.normal(size=193)
        second = rng.uniform(0.01, 2.0, size=193)
        reference = [moments(block, first, second, device="cpu") for block in x]
        reference_cross = np.einsum("brd,r->bd", x, second)
        for device in ("cpu", "cuda:0", "cuda:1") if cuda_available() else ("cpu",):
            gradient, hessian = moments_batch(x, first, second, device=device)
            np.testing.assert_allclose(
                gradient,
                np.asarray([item[0] for item in reference]),
                rtol=1e-13,
                atol=1e-13,
            )
            gradient, hessian, cross = moments_batch(
                x, first, second, device=device, return_second_gradient=True
            )
            np.testing.assert_allclose(cross, reference_cross, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(
                hessian,
                np.asarray([item[1] for item in reference]),
                rtol=1e-13,
                atol=1e-13,
            )

    def test_compiled_completion_matches_python_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 30)
            context = Context.make(data, np.arange(30, dtype=np.int32))
            compiled = ResponseEngine(data, lag=3, knot_count=3, cache_bytes=1024**2)
            compiled_block = compiled.block(context, (0, 1), 2)
            with (
                mock.patch("crbstpp.response.completion_events", return_value=None),
                mock.patch("crbstpp.response.kernel_contributions", return_value=None),
            ):
                reference = ResponseEngine(
                    data, lag=3, knot_count=3, cache_bytes=1024**2
                )
                reference_block = reference.block(context, (0, 1), 2)
            np.testing.assert_array_equal(compiled_block.rows, reference_block.rows)
            np.testing.assert_allclose(
                compiled_block.values, reference_block.values, rtol=0, atol=0
            )

    @unittest.skipUnless(cpu_available(), "compiled CPU operators are unavailable")
    def test_compact_thirty_tick_accumulator_matches_direct_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 80)
            context = Context.make(data, np.arange(80, dtype=np.int32))
            compiled = ResponseEngine(data, lag=30, knot_count=4, cache_bytes=64 * 1024**2)
            actual = compiled.block(context, (0, 1), 3)
            with (
                mock.patch("crbstpp.response.accumulate_kernel", return_value=False),
                mock.patch("crbstpp.response.kernel_contributions", return_value=None),
            ):
                reference = ResponseEngine(
                    data, lag=30, knot_count=4, cache_bytes=64 * 1024**2
                ).block(context, (0, 1), 3)
            np.testing.assert_array_equal(actual.rows, reference.rows)
            np.testing.assert_allclose(
                actual.values, reference.values, rtol=2.0e-14, atol=2.0e-14
            )

    @unittest.skipUnless(cpu_available(), "compiled CPU operators are unavailable")
    def test_large_response_min_span_sweep_is_exact(self) -> None:
        # Cross the native large-input threshold so this exercises the
        # entity-local monotone-deque implementation rather than its tiny
        # direct-loop reference path.
        n_entities = 50_000
        entities = np.repeat(np.arange(n_entities, dtype=np.int64), 2)
        times = np.tile(np.asarray((0, 1), dtype=np.int64), n_entities)
        first = np.arange(n_entities, dtype=np.int64) % 11
        second = (3 * np.arange(n_entities, dtype=np.int64) + 1) % 13
        spans = np.column_stack((first, second)).reshape(-1)
        starts = np.zeros(n_entities, dtype=np.int64)
        ends = np.full(n_entities, 3, dtype=np.int64)
        offsets = np.arange(n_entities + 1, dtype=np.int64) * 4

        result = response_min_spans(
            entities,
            times,
            spans,
            starts,
            ends,
            offsets,
            horizon=3,
            n_grid=4 * n_entities,
        )
        self.assertIsNotNone(result)
        assert result is not None
        rows, actual = result
        expected_rows = np.column_stack(
            (
                offsets[:-1] + 1,
                offsets[:-1] + 2,
                offsets[:-1] + 3,
            )
        ).reshape(-1)
        minimum = np.minimum(first, second)
        expected = np.column_stack((first, minimum, minimum)).reshape(-1)
        np.testing.assert_array_equal(rows, expected_rows)
        np.testing.assert_array_equal(actual, expected)

    def test_compiled_footprint_matches_python_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 30)
            context = Context.make(data, np.arange(30, dtype=np.int32))
            rule = RuleIdentity((0, 1), 2, 1)
            compiled = ResponseEngine(data, lag=3, knot_count=3, cache_bytes=1024**2)
            compiled_rows = compiled.footprint_rows(context, rule, 2)
            with (
                mock.patch("crbstpp.response.response_min_spans", return_value=None),
                mock.patch("crbstpp.response.future_rows", return_value=None),
            ):
                reference = ResponseEngine(
                    data, lag=3, knot_count=3, cache_bytes=1024**2
                )
                reference_rows = reference.footprint_rows(context, rule, 2)
            np.testing.assert_array_equal(compiled_rows, reference_rows)

    def test_effective_windows_equal_exact_response_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 45)
            context = Context.make(data, np.arange(45, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            windows = (0, 1, 2, 3)
            exact = []
            distinct = []
            for window in windows:
                block = engine.block(context, (0, 1), window)
                np.testing.assert_array_equal(
                    engine.response_rows(context, (0, 1), window), block.rows
                )
                if not len(block.rows):
                    continue
                if any(
                    np.array_equal(block.rows, previous.rows)
                    and np.array_equal(block.values, previous.values)
                    for previous in distinct
                ):
                    continue
                exact.append(window)
                distinct.append(block)
            self.assertEqual(
                engine.effective_windows(context, (0, 1), windows), tuple(exact)
            )

    def test_nested_response_thresholds_equal_every_exact_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 45)
            context = Context.make(data, np.arange(45, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            windows = (0, 1, 2, 3)
            rows, minimum_spans = engine.response_row_thresholds(
                context, (0, 1), max(windows)
            )
            batched = engine.response_rows_many(context, (0, 1), windows)
            reference = {
                window: engine.block(context, (0, 1), window) for window in windows
            }
            engine.clear_caches()
            blocks = engine.blocks_many(context, (0, 1), windows)
            engine.clear_caches()
            streamed = dict(
                engine.iter_blocks_many(context, (0, 1), windows, retain=False)
            )
            for window in windows:
                expected = rows[minimum_spans <= window * data.ticks_per_unit]
                np.testing.assert_array_equal(batched[window], expected)
                np.testing.assert_array_equal(batched[window], reference[window].rows)
                np.testing.assert_array_equal(
                    blocks[window].rows, reference[window].rows
                )
                np.testing.assert_allclose(
                    blocks[window].values,
                    reference[window].values,
                    rtol=0,
                    atol=0,
                )
                np.testing.assert_array_equal(
                    streamed[window].rows, reference[window].rows
                )
                np.testing.assert_allclose(
                    streamed[window].values,
                    reference[window].values,
                    rtol=0,
                    atol=0,
                )

    def test_latest_witness_completion_updates_all_three_sources(self) -> None:
        sources = [
            (np.asarray([0, 0]), np.asarray([1, 5])),
            (np.asarray([0, 0]), np.asarray([2, 4])),
            (np.asarray([0, 0]), np.asarray([3, 5])),
        ]
        result = completion_events(sources)
        self.assertIsNotNone(result)
        entities, times, spans = result
        np.testing.assert_array_equal(entities, [0, 0, 0])
        np.testing.assert_array_equal(times, [3, 4, 5])
        np.testing.assert_array_equal(spans, [2, 3, 1])

    def test_same_primitive_attributes_are_not_temporal_completions(self) -> None:
        sources = [
            (
                np.asarray([0, 0]),
                np.asarray([1, 3]),
                np.asarray([100, 101]),
            ),
            (
                np.asarray([0, 0]),
                np.asarray([1, 4]),
                np.asarray([100, 102]),
            ),
        ]
        result = completion_events(sources)
        self.assertIsNotNone(result)
        entities, times, spans = result
        # A and B at t=1 are two attributes of primitive event 100 and cannot
        # form a temporal pair.  A later distinct B event can complete it.
        np.testing.assert_array_equal(entities, [0, 0])
        np.testing.assert_array_equal(times, [3, 4])
        np.testing.assert_array_equal(spans, [2, 1])

    def test_same_time_alternative_primitive_preserves_valid_pair(self) -> None:
        sources = [
            (
                np.asarray([0, 0]),
                np.asarray([1, 1]),
                np.asarray([100, 101]),
            ),
            (
                np.asarray([0]),
                np.asarray([1]),
                np.asarray([101]),
            ),
        ]
        result = completion_events(sources)
        self.assertIsNotNone(result)
        entities, times, spans = result
        # The last A row shares primitive 101 with B, but A=100 remains a
        # valid distinct witness at the same timestamp.
        np.testing.assert_array_equal(entities, [0])
        np.testing.assert_array_equal(times, [1])
        np.testing.assert_array_equal(spans, [0])

    def test_lru_does_not_retain_oversized_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 30)
            context = Context.make(data, np.arange(30, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=3, cache_bytes=1)
            block = engine.block(context, (0,), 0)
            self.assertGreater(block.nbytes, 1)
            self.assertLessEqual(engine._cache_size, engine.cache_bytes)

    def test_support_matrix_cache_is_bounded_without_refitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                solver_tolerance=1e-8,
                solver_max_iter=120,
                cache_bytes=4096,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            fitted = []
            for rule in optimizer.dictionary:
                fitted.append(optimizer.fit(Support.of((rule,)), empty))
            exact_before = optimizer.diagnostics.exact_fits
            for record in fitted:
                recovered = optimizer.fit(record.support, empty)
                self.assertEqual(recovered.fit.nll, record.fit.nll)
            self.assertEqual(optimizer.diagnostics.exact_fits, exact_before)
            baseline_bytes = optimizer.records[Support(())].matrix.nbytes
            self.assertLessEqual(
                optimizer._record_cache_bytes,
                max(optimizer._record_cache_limit, baseline_bytes),
            )

    def test_ragged_baseline_hierarchy_prices_match_cpu_reference(self) -> None:
        if not cuda_available():
            self.skipTest("CUDA pricing operators unavailable")
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=("cuda:0", "cuda:1"),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            windows = tuple(
                sorted(
                    {
                        rule.window
                        for rule in optimizer.dictionary
                        if rule.antecedent == (0, 1)
                    }
                )
            )
            expected = optimizer._price_hierarchy_skeleton(
                empty, (0, 1), windows, device="cpu"
            )
            actual = dict(
                optimizer._price_baseline_hierarchy_batch(
                    empty, (((0, 1), windows),), device="cuda:0"
                )
            )[(0, 1)]
            for window in windows:
                np.testing.assert_allclose(
                    actual[window][:4], expected[window][:4], rtol=1e-10, atol=1e-10
                )
                self.assertEqual(actual[window][4:], expected[window][4:])

    def test_embedded_closure_null_matches_direct_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            rule = RuleIdentity((0, 1), 2, 1)
            full = optimizer.fit(Support.of((rule,)), empty)
            direct_matrix = optimizer.engine.model_matrix(
                optimizer.context,
                Support(()),
                forced_closure=full.matrix.closure,
            )
            direct = fit_model_matrix(
                direct_matrix,
                likelihood=data.likelihood,
                tolerance=config.solver_tolerance,
                max_iter=config.solver_max_iter,
            )
            original = tuple(
                value.copy()
                for value in (
                    full.matrix.x,
                    full.matrix.exposure_weight,
                    full.matrix.noevent_weight,
                    full.matrix.event_weight,
                )
            )
            embedded = optimizer._fit_embedded_closure_null(full, device="cpu")
            for expected, actual in zip(
                original,
                (
                    full.matrix.x,
                    full.matrix.exposure_weight,
                    full.matrix.noevent_weight,
                    full.matrix.event_weight,
                ),
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(embedded.converged, direct.converged)
            self.assertAlmostEqual(embedded.nll, direct.nll, places=10)
            if direct.converged:
                np.testing.assert_allclose(
                    embedded.coefficients,
                    direct.coefficients,
                    rtol=1e-8,
                    atol=1e-8,
                )
            else:
                self.assertEqual(embedded.recession, direct.recession)

    def test_projected_drop_one_matches_direct_response_refit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            retained = RuleIdentity((0,), 0, 1)
            removed = RuleIdentity((1,), 0, 1)
            full = optimizer.fit(Support.of((retained, removed)), empty)
            target = Support.of((retained,))
            projection = optimizer._factorized_drop_projection(full, target)
            self.assertIsNotNone(projection)
            columns, scales = projection
            view_warm = full.fit.coefficients[columns] * scales
            view_warm[full.matrix.free_dimension :] = np.maximum(
                view_warm[full.matrix.free_dimension :], 0.0
            )
            view = fit_projected_model_matrix(
                full.matrix,
                columns,
                scales,
                likelihood=data.likelihood,
                free_dimension=full.matrix.free_dimension,
                tolerance=config.solver_tolerance,
                max_iter=config.solver_max_iter,
                warm_start=view_warm,
            )
            # An exact but non-improving Drop is decided from the sparse-grid
            # solver (or its projected-view fail-open) without ever
            # constructing its observation-sized target matrix.
            optimizer._stored_records.pop(target, None)
            optimizer.records.pop(target, None)
            with mock.patch.object(
                optimizer,
                "_project_factorized_support",
                side_effect=AssertionError(
                    "non-improving projected Drop materialized a matrix"
                ),
            ):
                rejected = optimizer._best_exact_rule_objective_drop(
                    replace(full, score=1.0e100),
                    protected_antecedents=frozenset({retained.antecedent}),
                )
            self.assertIsNone(rejected)
            self.assertEqual(
                optimizer.diagnostics.safe_column_sparse_exact_fits
                + optimizer.diagnostics.projected_view_drop_fits,
                1,
            )
            self.assertEqual(
                optimizer.diagnostics.projected_view_drop_matrix_builds_avoided,
                1,
            )
            projected_matrix, projected = optimizer.fit_fixed(
                target, (), source=full, device="cpu"
            )
            direct_matrix = optimizer.engine.model_matrix(
                optimizer.context, target, forced_closure=()
            )
            direct = fit_model_matrix(
                direct_matrix,
                likelihood=data.likelihood,
                tolerance=config.solver_tolerance,
                max_iter=config.solver_max_iter,
            )

            def resident_reference(
                x: np.ndarray,
                beta: np.ndarray,
                exposure_weight: np.ndarray,
                event_weight: np.ndarray,
                **kwargs: object,
            ) -> tuple[
                float,
                None,
                np.ndarray | None,
                np.ndarray | None,
            ]:
                eta = x @ beta
                intensity = exposure_weight * np.exp(eta)
                nll = float(np.sum(intensity - event_weight * eta))
                if not bool(kwargs["compute_moments"]):
                    return nll, None, None, None
                first = intensity - event_weight
                gradient = x.T @ first
                hessian = x.T @ (intensity[:, None] * x)
                return nll, None, gradient, hessian

            with (
                mock.patch(
                    "crbstpp.solver.release_cuda_workspaces",
                    return_value=True,
                ),
                mock.patch(
                    "crbstpp.solver.resident_poisson_objective",
                    side_effect=resident_reference,
                ),
            ):
                evaluator = ProjectedDesignEvaluator(
                    full.matrix,
                    likelihood="poisson",
                    devices=("cuda:0", "cuda:1"),
                )
                try:
                    self.assertEqual(evaluator.shard_count, 2)
                    self.assertTrue(
                        all(
                            np.shares_memory(shard.x, full.matrix.x)
                            for shard in evaluator._shards
                        )
                    )
                    shard_nll, shard_gradient, shard_hessian = evaluator.objective(
                        full.fit.coefficients
                    )
                finally:
                    evaluator.close()
            direct_nll, direct_gradient, direct_hessian, _ = _objective(
                full.matrix,
                "poisson",
                full.fit.coefficients,
            )
            self.assertAlmostEqual(shard_nll, direct_nll, places=10)
            np.testing.assert_allclose(
                shard_gradient, direct_gradient, rtol=2e-12, atol=1e-10
            )
            np.testing.assert_allclose(
                shard_hessian, direct_hessian, rtol=2e-12, atol=1e-10
            )
            self.assertTrue(projected.converged, projected.message)
            self.assertTrue(direct.converged, direct.message)
            self.assertTrue(view.converged, view.message)
            self.assertLessEqual(len(projected_matrix.x), len(direct_matrix.x))
            self.assertAlmostEqual(projected.nll, direct.nll, places=10)
            self.assertAlmostEqual(view.nll, direct.nll, places=10)
            np.testing.assert_allclose(
                projected.coefficients, direct.coefficients, rtol=2e-9, atol=1e-8
            )
            np.testing.assert_allclose(
                view.coefficients, direct.coefficients, rtol=2e-9, atol=1e-8
            )

    def test_pair_pricing_has_no_hidden_closure_gain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            rule = next(
                identity
                for identity in optimizer.dictionary
                if identity.antecedent == (0, 1) and identity.sign > 0
            )
            priced = optimizer._price_hierarchy_skeleton(
                empty, rule.antecedent, (rule.window,), device="cpu"
            )
            self.assertEqual(priced[rule.window][4], 0)
            ranked = optimizer._rank_block_identities(empty, (rule,))
            self.assertTrue(np.isfinite(ranked[0][0]))

    def test_triplet_pricing_uses_ragged_gpu_batches_not_static_cpu_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 45)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=3,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_workers=6,
                pricing_devices=("cuda:0", "cuda:1"),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            identities = tuple(
                RuleIdentity(antecedent, 0, 1)
                for antecedent in (
                    (0,),
                    (0, 1),
                    (0, 1, 2),
                    (0, 1, 3),
                    (0, 2, 3),
                    (1, 2, 3),
                    (0, 1, 4),
                    (0, 2, 4),
                    (1, 2, 4),
                    (0, 3, 4),
                    (1, 3, 4),
                )
            )
            calls: list[tuple[tuple[int, ...], str]] = []

            def fake_price(current, groups, *, device, implicit_only=False):
                del current
                output = []
                for antecedent, windows in groups:
                    calls.append((antecedent, device))
                    output.append(
                        (
                            antecedent,
                            {
                                window: (0.0, 0.0, 0.0, 0.0, 0, True)
                                for window in windows
                            },
                        )
                    )
                return output

            with mock.patch.object(
                optimizer, "_price_baseline_hierarchy_batch", side_effect=fake_price
            ):
                optimizer._rank_block_identities(empty, identities)
            triplet_devices = {
                device for antecedent, device in calls if len(antecedent) == 3
            }
            self.assertEqual(triplet_devices, {"cuda:0", "cuda:1"})
            self.assertFalse(
                any(
                    device == "cpu"
                    for antecedent, device in calls
                    if len(antecedent) == 3
                )
            )

    def test_nonempty_poisson_pricing_uses_compact_batch_for_small_wave(self) -> None:
        """The 1--7 candidate tail must not fall back to scalar CPU pricing."""
        with tempfile.TemporaryDirectory() as directory:
            original = synthetic_dataset(Path(directory) / "data", 60)
            data = replace(original, likelihood="poisson")
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_workers=2,
                pricing_devices=("cuda:0",),
                search_mode="fast_block_score",
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            active = optimizer.fit(
                Support.of((RuleIdentity((0,), 0, 1),)),
                empty,
                device="cpu",
            )
            identities = (RuleIdentity((0, 1), 1, 1),)
            expected = [
                (
                    (0, 1),
                    {
                        1: (0.0, 0.0, 0.0, 0.0, config.knot_count, True),
                    },
                )
            ]
            with (
                mock.patch.object(
                    optimizer,
                    "_support_batched_component_items",
                    return_value=expected,
                ) as batched,
                mock.patch.object(
                    optimizer,
                    "_price_hierarchy_skeleton",
                    side_effect=AssertionError("scalar pricing path was used"),
                ),
            ):
                ranked = optimizer._rank_block_identities(active, identities)
            self.assertEqual(len(ranked), 1)
            batched.assert_called_once()

    def test_safe_nonempty_cloglog_pricing_uses_compact_gpu_batch(self) -> None:
        """Safe-column pricing must use the exact GPU batch, not scalar CPU."""
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_workers=2,
                pricing_devices=("cuda:0",),
                search_mode="safe_column_generation",
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            active = optimizer.fit(
                Support.of((RuleIdentity((0,), 0, 1),)),
                empty,
                device="cpu",
            )
            identities = (RuleIdentity((0, 1), 1, 1),)
            expected = [
                (
                    (0, 1),
                    {1: (0.0, 0.0, 0.0, 0.0, config.knot_count, True)},
                )
            ]
            with (
                mock.patch.object(
                    optimizer,
                    "_support_batched_component_items",
                    return_value=expected,
                ) as batched,
                mock.patch.object(
                    optimizer,
                    "_price_hierarchy_skeleton",
                    side_effect=AssertionError("scalar pricing path was used"),
                ),
            ):
                ranked = optimizer._rank_block_identities(active, identities)
            self.assertEqual(len(ranked), 1)
            batched.assert_called_once()

    def test_lazy_baseline_profile_selects_rule_level_priority_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            eager = SupportOptimizer(Context.make(data, fit_codes), config)
            eager_ranked = eager._rank_mdl_identities(
                eager.records[Support(())], eager.dictionary
            )
            expected = {}
            for antecedent in eager.skeletons:
                group = [
                    item for item in eager_ranked if item[2].antecedent == antecedent
                ]
                if group:
                    expected[antecedent] = sorted(
                        group, key=lambda item: (-item[0], item[2])
                    )[0][2]

            lazy = SupportOptimizer(Context.make(data, fit_codes), config)
            first = {lazy.skeletons[0]}
            lazy._baseline_profile_skeletons(first)
            self.assertEqual(set(lazy._profiled_by_antecedent), first)
            lazy._baseline_profile_skeletons(set(lazy.skeletons) - first)
            self.assertEqual(lazy._profiled_by_antecedent, expected)

    def test_inactive_dictionary_uses_all_support_conditioned_identities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            optimizer._baseline_profile_skeletons(set(optimizer.skeletons))
            pair = (0, 1)
            expected = tuple(
                rule for rule in optimizer.dictionary if rule.antecedent == pair
            )
            self.assertGreater(len(expected), 1)
            # The baseline profile keeps one route-priority identity, whereas
            # a support state must expose the complete finite identity envelope.
            self.assertEqual(
                sum(
                    rule.antecedent == pair
                    for rule in (optimizer._profiled_dictionary or ())
                ),
                1,
            )
            actual = optimizer._inactive_identities(
                optimizer.records[Support(())], {pair}
            )
            self.assertEqual(actual, expected)

    def test_profiled_identity_is_conditioned_on_the_current_support(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer._skeleton_witnesses = {}
        antecedent = (0, 1)
        baseline = RuleIdentity(antecedent, 0, 1)
        conditional = RuleIdentity(antecedent, 2, -1)
        optimizer._rank_block_identities = lambda current, identities: [
            (1.0, 1.0, baseline, True),
            (5.0, 3.0, conditional, True),
        ]
        ranked = optimizer._rank_profiled_identities(  # type: ignore[arg-type]
            object(), (baseline, conditional)
        )
        self.assertEqual(ranked, [(5.0, 3.0, conditional)])
        self.assertEqual(optimizer._skeleton_witnesses[antecedent], (5.0, conditional))

    def test_nonnested_identity_profile_fails_open(self) -> None:
        optimizer = object.__new__(SupportOptimizer)
        optimizer._state_lock = __import__("threading").RLock()
        optimizer.diagnostics = SearchDiagnostics()
        optimizer._skeleton_witnesses = {}
        antecedent = (0, 1)
        nested = RuleIdentity(antecedent, 1, 1)
        promoted_negative = RuleIdentity(antecedent, 2, -1)
        promoted_positive = RuleIdentity(antecedent, 2, 1)
        optimizer._rank_block_identities = lambda current, identities: [
            (2.0, 1.0, nested, True),
            (float("inf"), float("inf"), promoted_negative, False),
            (float("inf"), float("inf"), promoted_positive, False),
        ]
        ranked = optimizer._rank_profiled_identities(  # type: ignore[arg-type]
            object(), (nested, promoted_negative, promoted_positive)
        )
        self.assertEqual(
            {item[2] for item in ranked},
            {nested, promoted_negative, promoted_positive},
        )

    def test_active_identity_envelope_can_replace_baseline_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            old = RuleIdentity((0,), 0, -1)
            alternative = RuleIdentity((0,), 0, 1)
            current = optimizer.fit(Support.of((old,)), empty)
            # The envelope implementation is being tested rather than the
            # synthetic objective's preferred sign, so expose an incumbent
            # score below every feasible candidate and prescribe the
            # support-conditioned identity ranking.
            current = replace(current, score=-1.0e12)
            with (
                mock.patch.object(optimizer, "_identity_drop_base", return_value=empty),
                mock.patch.object(
                    optimizer,
                    "_rank_profiled_identities",
                    return_value=[(float("inf"), float("inf"), alternative)],
                ),
            ):
                selected = optimizer._best_conditional_identity_change(
                    current, drop_proposals={}
                )
            self.assertIsNotNone(selected)
            self.assertEqual(selected.support, Support.of((alternative,)))
            self.assertEqual(optimizer.diagnostics.conditional_identity_moves, 1)

    def test_directional_relaxation_contains_exact_support_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            supports = [Support.of((rule,)) for rule in optimizer.dictionary]
            a = RuleIdentity((0,), 0, 1)
            b = RuleIdentity((1,), 0, -1)
            ab = next(
                rule
                for rule in optimizer.dictionary
                if rule.antecedent == (0, 1) and rule.sign > 0
            )
            supports.extend((Support.of((a, b)), Support.of((a, ab))))
            for support in supports:
                exact = optimizer.fit(support, empty)
                if exact.fit.converged:
                    self.assertGreaterEqual(
                        optimizer.directional_upper_score(support) + 1e-7,
                        exact.score,
                    )

    def test_state_quotient_relaxation_contains_exact_add_scores(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            a = RuleIdentity((0,), 0, 1)
            ab_positive = next(
                rule
                for rule in optimizer.dictionary
                if rule.antecedent == (0, 1) and rule.sign > 0
            )
            ab_negative = next(
                rule
                for rule in optimizer.dictionary
                if rule.antecedent == (0, 1)
                and rule.window == ab_positive.window
                and rule.sign < 0
            )
            checked = 0
            for current, addition in (
                (optimizer.fit(Support.of((a,)), empty), ab_positive),
                (optimizer.fit(Support.of((a,)), empty), ab_negative),
                (optimizer.fit(Support.of((ab_positive,)), empty), a),
            ):
                self.assertTrue(current.fit.converged, current.fit.message)
                trial = current.support.add(addition)
                raw = optimizer.engine.block(
                    optimizer.context,
                    addition.antecedent,
                    addition.window,
                    addition.relation,
                )
                streamed = optimizer.engine.mask_total_state_added_block(
                    optimizer.context,
                    trial,
                    addition,
                    raw,
                )
                scalar = optimizer.engine.total_state_added_block(
                    optimizer.context,
                    trial,
                    addition,
                )
                np.testing.assert_array_equal(streamed.rows, scalar.rows)
                np.testing.assert_array_equal(streamed.values, scalar.values)
                self.assertTrue(
                    optimizer.engine.total_state_geometry_changed(
                        current.support, trial
                    )
                )
                exact = optimizer.fit(trial, current)
                if not exact.fit.converged:
                    continue
                upper = optimizer._state_splice_group_upper_score(
                    current, addition
                )
                slack = 1.0e-7 * max(1.0, abs(exact.score))
                self.assertGreaterEqual(upper + slack, exact.score)
                checked += 1
                # A repeated query must be scalar-cache only.
                self.assertEqual(
                    upper,
                    optimizer._state_splice_group_upper_score(current, addition),
                )
                with mock.patch.object(
                    optimizer,
                    "safe_upper_score",
                    side_effect=AssertionError(
                        "finite conditional bound must skip global row unions"
                    ),
                ):
                    self.assertEqual(
                        upper,
                        optimizer._state_splice_safe_upper_score(
                            current, addition
                        ),
                    )
            # The same quotient is valid for an ordinary Add whose existing
            # total-state columns do not change.
            b = RuleIdentity((1,), 0, 1)
            current = optimizer.fit(Support.of((a,)), empty)
            trial = current.support.add(b)
            self.assertFalse(
                optimizer.engine.total_state_geometry_changed(
                    current.support,
                    trial,
                )
            )
            exact = optimizer.fit(trial, current)
            self.assertTrue(exact.fit.converged, exact.fit.message)
            upper = optimizer._state_splice_group_upper_score(current, b)
            slack = 1.0e-7 * max(1.0, abs(exact.score))
            self.assertGreaterEqual(upper + slack, exact.score)
            checked += 1
            self.assertGreater(checked, 0)
            self.assertGreater(
                optimizer.diagnostics.state_splice_group_bound_cache_hits, 0
            )
            self.assertGreater(optimizer.diagnostics.state_quotient_rows, 0)
            self.assertGreater(optimizer.diagnostics.state_quotient_groups, 0)
            self.assertLessEqual(
                optimizer.diagnostics.state_quotient_groups,
                optimizer.diagnostics.state_quotient_rows,
            )
            self.assertGreater(
                optimizer.diagnostics.state_quotient_strict_tightenings,
                0,
            )
            optimizer.close()

    def test_additive_hierarchy_quotient_contains_exact_add_scores(self) -> None:
        """The joint rule/closure quotient must contain every exact child."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 160)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=3,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                effect_model="additive_hierarchy",
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            a = next(
                rule
                for rule in optimizer.dictionary
                if rule.antecedent == (0,) and rule.sign > 0
            )
            current = optimizer.fit(Support.of((a,)), empty)
            self.assertTrue(current.fit.converged, current.fit.message)
            checked = 0
            for addition in optimizer.dictionary:
                if addition.order <= 1 or 0 not in addition.antecedent:
                    continue
                trial = current.support.add(addition)
                if optimizer._ensure_closure_signs(
                    hierarchy_closure(trial)
                ).intersection(hierarchy_closure(trial)):
                    continue
                exact = optimizer.fit(trial, current)
                if not exact.fit.converged:
                    continue
                upper = optimizer._additive_state_splice_group_upper_score(
                    current, addition
                )
                slack = 2.0e-7 * max(1.0, abs(exact.score))
                self.assertGreaterEqual(upper + slack, exact.score)
                self.assertLessEqual(
                    upper,
                    optimizer.localized_upper_score(trial) + slack,
                )
                checked += 1
                if checked >= 6:
                    break
            self.assertGreaterEqual(checked, 1)
            optimizer.close()

    def test_sparse_total_state_splice_matches_full_matrix_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            a = RuleIdentity((0,), 0, 1)
            ab = next(
                rule
                for rule in optimizer.dictionary
                if rule.antecedent == (0, 1) and rule.sign > 0
            )
            ab_negative = RuleIdentity(ab.antecedent, ab.window, -1)
            rng = np.random.default_rng(9127)
            for current, addition in (
                (optimizer.fit(Support.of((a,)), empty), ab),
                (optimizer.fit(Support.of((a,)), empty), ab_negative),
                (optimizer.fit(Support.of((ab,)), empty), a),
            ):
                self.assertTrue(current.fit.converged, current.fit.message)
                trial = current.support.add(addition)
                sparse = optimizer.engine.splice_total_state_add(
                    optimizer.context,
                    trial,
                    current.matrix,
                    addition,
                )
                compact = optimizer.engine.splice_total_state_add(
                    optimizer.context,
                    trial,
                    current.matrix,
                    addition,
                    include_active_metadata=False,
                )
                reference = optimizer.engine.model_matrix(
                    optimizer.context, trial, forced_closure=()
                )
                self.assertEqual(len(compact.active_rows), 0)
                extension = RuleIdentity((1,), 0, 1)
                extended_support = trial.add(extension)
                extended = optimizer.engine.extend_model_matrix(
                    optimizer.context,
                    extended_support,
                    compact,
                )
                extended_reference = optimizer.engine.model_matrix(
                    optimizer.context,
                    extended_support,
                )
                extended_beta = rng.normal(size=extended_reference.dimension)
                extended_beta[extended_reference.free_dimension :] = np.abs(
                    extended_beta[extended_reference.free_dimension :]
                )
                self.assertAlmostEqual(
                    _objective(extended, data.likelihood, extended_beta)[0],
                    _objective(
                        extended_reference,
                        data.likelihood,
                        extended_beta,
                    )[0],
                    places=9,
                )
                beta = rng.normal(size=reference.dimension)
                beta[reference.free_dimension :] = np.abs(
                    beta[reference.free_dimension :]
                )
                sparse_nll = _objective(sparse, data.likelihood, beta)[0]
                compact_nll = _objective(compact, data.likelihood, beta)[0]
                reference_nll = _objective(reference, data.likelihood, beta)[0]
                self.assertAlmostEqual(sparse_nll, reference_nll, places=9)
                self.assertAlmostEqual(compact_nll, reference_nll, places=9)
                node = optimizer._conditional_node_state(
                    current, device="cpu"
                )
                matrix_free = optimizer._matrix_free_state_splice(
                    current, trial, addition, node
                )
                self.assertIsNotNone(matrix_free)
                assert matrix_free is not None
                matrix_free_nll = matrix_free.one_step_nll(
                    config.solver_tolerance
                )
                compact_step = one_step_model_matrix(
                    compact,
                    likelihood=data.likelihood,
                    warm_start=optimizer.warm_start(current, compact),
                    tolerance=config.solver_tolerance,
                    device="cpu",
                )
                self.assertIsNotNone(matrix_free_nll)
                assert matrix_free_nll is not None
                self.assertAlmostEqual(
                    matrix_free_nll, compact_step.fit.nll, places=9
                )
                sparse_fit = fit_model_matrix(
                    sparse,
                    likelihood=data.likelihood,
                    tolerance=1e-9,
                    max_iter=150,
                )
                reference_fit = fit_model_matrix(
                    reference,
                    likelihood=data.likelihood,
                    tolerance=1e-9,
                    max_iter=150,
                )
                if addition.sign < 0:
                    self.assertEqual(
                        sparse_fit.recession, reference_fit.recession
                    )
                    continue
                self.assertTrue(sparse_fit.converged, sparse_fit.message)
                self.assertTrue(reference_fit.converged, reference_fit.message)
                self.assertAlmostEqual(
                    sparse_fit.nll, reference_fit.nll, places=8
                )
            optimizer.close()

    def test_single_sparse_recession_is_a_conclusive_exact_rejection(self) -> None:
        """A proven nonattained child must not enter another exact backend."""

        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 80)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                solver_tolerance=1e-8,
                solver_max_iter=50,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            trial = Support.of((RuleIdentity((0,), 0, -1),))
            metadata = optimizer.engine.model_metadata(trial)
            recession = FitResult(
                coefficients=np.zeros(metadata.dimension, dtype=np.float64),
                nll=np.inf,
                converged=False,
                iterations=0,
                projected_kkt=np.inf,
                rank=0,
                recession=True,
                message="nonattained sparse coordinate recession direction",
            )
            with mock.patch(
                "crbstpp.search.fit_sparse_grid_model", return_value=recession
            ) as solver:
                record = optimizer._fit_sparse_support_exact(
                    trial, empty, device="cpu"
                )
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(solver.call_count, 1)
            self.assertTrue(record.fit.recession)
            self.assertEqual(record.score, -np.inf)
            self.assertIn(trial, optimizer._conditional_forbidden)
            self.assertEqual(
                optimizer.diagnostics.unattainable_support_rejections, 1
            )
            self.assertEqual(
                optimizer.diagnostics.safe_column_sparse_exact_fallbacks, 0
            )
            optimizer.close()

    def test_reused_hierarchy_row_lookup_preserves_exact_moments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1, 2),
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            blocks = (
                optimizer.engine.block(optimizer.context, (0,), 0),
                optimizer.engine.block(optimizer.context, (1,), 0),
                optimizer.engine.block(optimizer.context, (0, 1), 2),
            )
            expected = optimizer._joint_hierarchy_moments(empty, blocks, device="cpu")
            lookup = np.full(optimizer.context.n_grid, 1_234_567, dtype=np.int32)
            actual = optimizer._joint_hierarchy_moments(
                empty, blocks, device="cpu", row_lookup=lookup
            )
            np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
            np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=0)

    def test_parallel_exact_batch_matches_serial_solver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            common = dict(
                dataset=str(data.root),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            serial = SupportOptimizer(
                Context.make(data, fit_codes), RunConfig(**common, exact_workers=1)
            )
            parallel = SupportOptimizer(
                Context.make(data, fit_codes), RunConfig(**common, exact_workers=3)
            )
            supports = [Support.of((rule,)) for rule in serial.dictionary[:6]]
            serial_records = serial.fit_many(supports, serial.records[Support(())])
            parallel_records = parallel.fit_many(
                supports, parallel.records[Support(())]
            )
            for left, right in zip(serial_records, parallel_records, strict=True):
                self.assertEqual(left.fit.converged, right.fit.converged)
                self.assertAlmostEqual(left.fit.nll, right.fit.nll, places=11)
                np.testing.assert_allclose(
                    left.fit.coefficients,
                    right.fit.coefficients,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_continuous_poisson_uses_exact_tick_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            write_dataset(
                root,
                entities=pd.DataFrame(
                    {
                        "entity_id": ["a", "b", "c"],
                        "start_time": [0, 0, 0],
                        "end_time": [59, 59, 59],
                        "baseline_origin": [0, 0, 0],
                        "split_group": [0, 0, 0],
                    }
                ),
                events=pd.DataFrame(
                    [(0, 1, 0), (1, 1, 0), (2, 1, 0)],
                    columns=["entity_code", "time", "predicate_code"],
                ),
                targets=pd.DataFrame(
                    [(0, 20, 1), (1, 30, 1), (2, 40, 1)],
                    columns=["entity_code", "time", "multiplicity"],
                ),
                predicate_names=("A",),
                likelihood="poisson",
                time_unit="hour",
                ticks_per_unit=60,
                adverse_event_name="synthetic recurrent event",
                f0_contract={
                    "dynamic_predicates": True,
                    "outcome_blind_predicate_construction": True,
                    "direct_target_proxy_excluded_from_reported_dictionary": True,
                    "strict_future_effect_required": True,
                    "atomic_predicates": True,
                    "primitive_event_provenance": True,
                },
                provenance={"generator": "continuous-poisson-test"},
            )
            data = Dataset.load(root)
            context = Context.make(data, np.arange(3, dtype=np.int32))
            engine = ResponseEngine(data, lag=1, knot_count=2, cache_bytes=1024**2)
            matrix = engine.model_matrix(context, Support(()))
            fit = fit_model_matrix(
                matrix, likelihood="poisson", tolerance=1e-10, max_iter=100
            )
            self.assertTrue(fit.converged, fit.message)
            total_exposure_hours = 3 * 60 / 60
            self.assertAlmostEqual(
                np.exp(fit.coefficients[0]), 3 / total_exposure_hours, places=8
            )
            self.assertEqual(engine.lag, 60)


if __name__ == "__main__":
    unittest.main()
