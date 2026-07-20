from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from crbstpp.likelihood import (
    conjugate_sum,
    loss_grid_sparse_event_derivatives,
    loss_rows,
)
from crbstpp.native import (
    aggregate_design_rows,
    aggregate_design_rows_with_groups,
    completion_events,
    cpu_available,
    cuda_available,
    moments,
    moments_batch,
    nonnegative_quadratic_gains,
)
from crbstpp.response import Context, ResponseEngine
from crbstpp.config import RunConfig
from crbstpp.data import Dataset, write_dataset
from crbstpp.dual import offset_dual_certificate
from crbstpp.rules import RuleIdentity, Support, hierarchy_closure
from crbstpp.search import (
    SupportOptimizer,
    _RestrictedAddBounds,
    _nonnegative_quadratic_gain,
    _nonnegative_quadratic_solution,
)
from crbstpp.solver import _objective, fit_model_matrix, fit_offset_design

from tests.crbstpp.test_core import synthetic_dataset


class NumericalParityTests(unittest.TestCase):
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

    def test_restricted_drop_scores_are_feasible_lower_bounds(self) -> None:
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
            support = Support.of(
                (
                    RuleIdentity((0,), 0, 1),
                    RuleIdentity((1,), 0, 1),
                )
            )
            current = optimizer.fit(support, empty)
            self.assertTrue(current.fit.converged)
            self.assertIsNone(optimizer._best_restricted_drop(current))
            for rule in support.rules:
                trial = support.drop(rule)
                restricted = optimizer._restricted_drop_scores[support][trial]
                full = optimizer.fit(trial, current)
                if full.fit.converged:
                    self.assertGreaterEqual(full.score + 1e-8, restricted)

    def test_restricted_add_score_is_a_feasible_full_support_lower_bound(self) -> None:
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
            compared = 0
            for rule in optimizer.dictionary:
                restricted = optimizer._restricted_add_score(empty, rule)
                if not np.isfinite(restricted):
                    continue
                full = optimizer.fit(Support.of((rule,)), empty)
                self.assertGreaterEqual(full.score + 1e-8, restricted)
                compared += 1
            self.assertGreater(compared, 0)
            unsigned = {(rule.antecedent, rule.window) for rule in optimizer.dictionary}
            self.assertEqual(
                optimizer.diagnostics.restricted_problem_builds, len(unsigned)
            )
            self.assertEqual(
                optimizer.diagnostics.restricted_problem_hits,
                len(optimizer.dictionary) - len(unsigned),
            )

    def test_offset_dual_certificate_contains_exact_restricted_optimum(self) -> None:
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
            certified = 0
            for rule in optimizer.dictionary:
                problem = optimizer._restricted_add_problem(empty, rule)
                if problem is None:
                    continue
                design = problem.unsigned_design.copy()
                if rule.sign < 0:
                    design[:, -config.knot_count :] *= -1.0
                exact = fit_offset_design(
                    design,
                    problem.offset,
                    problem.exposure,
                    problem.noevent,
                    problem.event,
                    likelihood=data.likelihood,
                    free_dimension=problem.free_dimension,
                    tolerance=config.solver_tolerance,
                    max_iter=config.solver_max_iter,
                )
                if not exact.converged:
                    continue
                certificate = offset_dual_certificate(
                    design,
                    problem.offset,
                    problem.exposure,
                    problem.noevent,
                    problem.event,
                    likelihood=data.likelihood,
                    beta=exact.coefficients,
                    free_dimension=problem.free_dimension,
                    tolerance=1e-7,
                    max_iter=500,
                )
                if not certificate.feasible:
                    continue
                self.assertLessEqual(
                    certificate.nll_lower_bound,
                    exact.nll + 1e-7 * max(1.0, abs(exact.nll)),
                )
                self.assertAlmostEqual(certificate.nll_lower_bound, exact.nll, places=5)
                certified += 1
                if certified == 4:
                    break
            self.assertGreater(certified, 0)

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

    def test_restricted_add_bound_contains_exact_block_score(self) -> None:
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
            checked = 0
            for rule in optimizer.dictionary:
                bound = optimizer._restricted_add_bound(empty, rule)
                exact = optimizer._restricted_add_score(empty, rule)
                if not np.isfinite(exact):
                    continue
                slack = 1e-7 * max(1.0, abs(exact))
                self.assertLessEqual(bound.lower_score, exact + slack)
                self.assertLessEqual(exact, bound.upper_score + slack)
                checked += 1
            self.assertGreater(checked, 0)

    def test_pattern_compressed_restricted_problem_matches_dense_reference(
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
                cache_bytes=32 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            rule = next(rule for rule in optimizer.dictionary if rule.order == 2)
            compact = optimizer._restricted_add_problem(empty, rule)
            self.assertIsNotNone(compact)

            unsigned = RuleIdentity(rule.antecedent, rule.window, 1)
            trial = empty.support.add(unsigned)
            new_closure = tuple(
                sorted(set(hierarchy_closure(trial)) - set(empty.matrix.closure))
            )
            specifications = [(term.antecedent, term.window) for term in new_closure]
            specifications.append((rule.antecedent, rule.window))
            blocks = [
                optimizer.engine.block(optimizer.context, antecedent, window)
                for antecedent, window in specifications
            ]
            rows = np.unique(
                np.concatenate([block.rows for block in blocks if len(block.rows)])
            )
            design = np.zeros(
                (len(rows), len(specifications) * config.knot_count),
                dtype=np.float64,
            )
            for block_index, block in enumerate(blocks):
                positions = np.searchsorted(rows, block.rows)
                left = block_index * config.knot_count
                design[positions, left : left + config.knot_count] = block.values
            offset = optimizer._pricing_rows(empty, rows)[0]
            event = np.zeros(len(rows), dtype=np.float64)
            positions = np.searchsorted(optimizer.context.target_rows, rows)
            matched = positions < len(optimizer.context.target_rows)
            safe = np.minimum(positions, len(optimizer.context.target_rows) - 1)
            matched &= optimizer.context.target_rows[safe] == rows
            event[matched] = optimizer.context.target_counts[positions[matched]]
            exposure = np.full(len(rows), optimizer.engine.tick_exposure)
            noevent = exposure - event
            joint, exposure, noevent, event = aggregate_design_rows(
                np.concatenate((offset[:, None], design), axis=1),
                exposure,
                noevent,
                event,
                copy_input=False,
            )
            np.testing.assert_array_equal(compact.offset, joint[:, 0])
            np.testing.assert_array_equal(compact.unsigned_design, joint[:, 1:])
            np.testing.assert_array_equal(compact.exposure, exposure)
            np.testing.assert_array_equal(compact.noevent, noevent)
            np.testing.assert_array_equal(compact.event, event)
            self.assertEqual(
                compact.free_dimension, len(new_closure) * config.knot_count
            )

            geometry = optimizer._restricted_geometry(empty, rule)
            self.assertIsNotNone(geometry)
            self.assertGreaterEqual(optimizer.diagnostics.restricted_geometry_hits, 1)

    def test_lazy_add_stops_after_incumbent_dominates_remaining_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            rules = optimizer.dictionary[:2]
            ranked = [(9.0, 5.0, rules[0]), (8.0, 4.0, rules[1])]
            bounds = {
                rules[0]: _RestrictedAddBounds(rules[0], -1.0, 7.0, -1.0, 7.0, None),
                rules[1]: _RestrictedAddBounds(rules[1], -1.0, 5.0, -1.0, 5.0, None),
            }

            def fitted(support: Support, source: object) -> object:
                return type(empty)(support, empty.matrix, empty.fit, empty.penalty, 6.0)

            with (
                mock.patch.object(
                    optimizer, "_rank_profiled_identities", return_value=ranked
                ),
                mock.patch.object(
                    optimizer, "_safe_identity_survivors", return_value=rules
                ),
                mock.patch.object(
                    optimizer,
                    "_restricted_add_bound",
                    side_effect=lambda current, rule: bounds[rule],
                ),
                mock.patch.object(
                    optimizer,
                    "safe_upper_score",
                    side_effect=lambda support: bounds[support.rules[-1]].upper_score,
                ),
                mock.patch.object(
                    optimizer,
                    "_restricted_add_score",
                    side_effect=lambda current, rule, device=None: (
                        6.0 if rule == rules[0] else 4.0
                    ),
                ) as exact,
                mock.patch.object(optimizer, "fit", side_effect=fitted),
            ):
                result = optimizer._best_restricted_addition(
                    empty, antecedents=set(optimizer.skeletons)
                )
            self.assertIsNotNone(result)
            self.assertEqual(result.support, empty.support.add(rules[0]))
            self.assertEqual(exact.call_count, 1)
            self.assertEqual(optimizer.diagnostics.lazy_exact_refits_avoided, 1)

    def test_restricted_add_audits_one_identity_per_skeleton(self) -> None:
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
            blocked = type(empty)(
                empty.support,
                empty.matrix,
                empty.fit,
                empty.penalty,
                float("inf"),
            )
            antecedents = set(optimizer.skeletons)
            self.assertIsNone(
                optimizer._best_restricted_addition(blocked, antecedents=antecedents)
            )
            self.assertLessEqual(
                optimizer.diagnostics.restricted_add_audits,
                len(antecedents),
            )

    def test_incremental_support_matrix_matches_fresh_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=32 * 1024**2)
            source_support = Support.of((RuleIdentity((0,), 0, 1),))
            target_support = source_support.add(RuleIdentity((0, 1), 2, -1))
            source = engine.model_matrix(context, source_support)
            incremental = engine.extend_model_matrix(context, target_support, source)
            fresh = engine.model_matrix(context, target_support)
            for left, right in (
                (incremental.x, fresh.x),
                (incremental.exposure_weight, fresh.exposure_weight),
                (incremental.noevent_weight, fresh.noevent_weight),
                (incremental.event_weight, fresh.event_weight),
                (incremental.active_rows, fresh.active_rows),
                (incremental.active_design_groups, fresh.active_design_groups),
            ):
                np.testing.assert_array_equal(left, right)

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

    def test_reliability_price_fails_open_when_f3_is_untestable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 90)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            rule = RuleIdentity((0,), 0, 1)
            price = optimizer._reliability_price(empty, rule)
            self.assertFalse(price.testable)
            self.assertTrue(price.admissible)
            self.assertEqual(
                optimizer.diagnostics.reliability_untestable_fail_open, 1
            )

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
                gradient, hessian = moments(x, first, second, device=device)
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

    def test_every_primal_dual_sandwich_contains_exact_support_score(self) -> None:
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
                solver_max_iter=150,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            for rule in optimizer.dictionary:
                support = Support.of((rule,))
                bounds = optimizer.bounds(empty, support)
                exact = optimizer.fit(support, empty)
                if exact.fit.converged:
                    self.assertLessEqual(bounds.lower_score, exact.score + 1e-7)
                    self.assertGreaterEqual(bounds.upper_score + 1e-7, exact.score)

    def test_fused_w_and_sign_pricing_matches_unfused_reference(self) -> None:
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
            rules = tuple(
                rule for rule in optimizer.dictionary if rule.antecedent == (0, 1)
            )
            fused = optimizer._price_skeleton(
                empty,
                (0, 1),
                tuple(rule.window for rule in rules),
                device="cpu",
            )
            for rule in rules:
                gradient, hessian = optimizer._legacy_block_price_components(
                    empty, rule, device="cpu"
                )
                np.testing.assert_allclose(
                    fused[rule.window][0], gradient, rtol=1e-13, atol=1e-13
                )
                np.testing.assert_allclose(
                    fused[rule.window][1], hessian, rtol=1e-13, atol=1e-13
                )

    def test_hierarchy_joint_block_score_matches_direct_trial_quadratic(self) -> None:
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
            empty = optimizer.records[Support(())]
            prices = optimizer._price_hierarchy_skeleton(
                empty, (0, 1), (0, 1, 2), device="cpu"
            )
            for window in (0, 1, 2):
                positive = RuleIdentity((0, 1), window, 1)
                matrix = optimizer.engine.model_matrix(
                    optimizer.context, Support.of((positive,))
                )
                warm = optimizer.warm_start(empty, matrix)
                _, gradient, hessian, _ = _objective(matrix, data.likelihood, warm)
                old = empty.matrix.dimension
                old_inverse = np.linalg.pinv(hessian[:old, :old], rcond=1e-12)
                cross = hessian[:old, old:]
                conditional_gradient = (
                    gradient[old:] - cross.T @ old_inverse @ gradient[:old]
                )
                conditional_hessian = (
                    hessian[old:, old:] - cross.T @ old_inverse @ cross
                )
                expected = tuple(
                    optimizer._hierarchy_quadratic_gain(
                        conditional_gradient,
                        conditional_hessian,
                        matrix.closure_dimension,
                        sign,
                    )
                    for sign in (-1, 1)
                )
                self.assertTrue(prices[window][5])
                np.testing.assert_allclose(
                    prices[window][:2],
                    (expected[0][0], expected[1][0]),
                    rtol=1e-11,
                    atol=1e-11,
                )
                np.testing.assert_allclose(
                    prices[window][2:4],
                    (expected[0][1], expected[1][1]),
                    rtol=1e-11,
                    atol=1e-11,
                )

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
            np.testing.assert_allclose(
                embedded.coefficients,
                direct.coefficients,
                rtol=1e-11,
                atol=1e-11,
            )

    def test_closure_only_gain_does_not_admit_reported_pair(self) -> None:
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
            empty = optimizer.records[Support(())]
            rule = next(
                identity
                for identity in optimizer.dictionary
                if identity.antecedent == (0, 1) and identity.sign > 0
            )
            closure_dimension = config.knot_count * 2
            artificial_price = {
                rule.window: (
                    1.0e6,
                    1.0e6,
                    0.0,
                    0.0,
                    closure_dimension,
                    True,
                )
            }
            with mock.patch.object(
                optimizer,
                "_price_hierarchy_skeleton",
                return_value=artificial_price,
            ):
                ranked = optimizer._rank_block_identities(empty, (rule,))
            self.assertLess(ranked[0][0], 0.0)

    def test_dual_geometry_is_shared_across_signs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 75)
            fit_codes, _, _ = data.split((0.6, 0.2, 0.2), 111)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                solver_tolerance=1e-8,
                solver_max_iter=150,
                cache_bytes=8 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            optimizer = SupportOptimizer(Context.make(data, fit_codes), config)
            empty = optimizer.records[Support(())]
            for sign in (-1, 1):
                rule = RuleIdentity((0,), 0, sign)
                optimizer.bounds(empty, Support.of((rule,)))
            self.assertGreaterEqual(optimizer.diagnostics.dual_geometry_cache_hits, 1)

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

    def test_histogram_standalone_screen_equals_generic_safe_bounds(self) -> None:
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
            identities = tuple(
                rule for rule in optimizer.dictionary if rule.antecedent == (0, 1)
            )
            generic = optimizer._safe_identity_survivors(empty, identities, 0.0)
            generic_upper = {
                rule: min(
                    optimizer.localized_upper_score(Support.of((rule,))),
                    optimizer.directional_upper_score(Support.of((rule,))),
                )
                for rule in identities
            }
            optimizer._relaxed_upper_cache.clear()
            optimizer._directional_upper_cache.clear()
            histogram = optimizer._safe_standalone_survivors(empty, (0, 1), identities)
            histogram_upper = {
                rule: min(
                    optimizer.localized_upper_score(Support.of((rule,))),
                    optimizer.directional_upper_score(Support.of((rule,))),
                )
                for rule in identities
            }
            self.assertEqual(generic, histogram)
            for rule in identities:
                self.assertAlmostEqual(
                    generic_upper[rule], histogram_upper[rule], places=10
                )

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
                    "direct_target_proxy_excluded": True,
                    "strict_future_effect_required": True,
                    "atomic_predicates": True,
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
