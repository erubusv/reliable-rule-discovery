from __future__ import annotations

import ast
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from crbstpp.certification import (
    _BootstrapComponent,
    _branch_null_closure,
    _calendar_block_losses_sparse,
    _entity_losses_sparse,
    _frozen_reported_direction,
    _multinomial_l1_radius,
    _romano_wolf_adjust,
    _scalar_direction_score,
    _worst_case_total_variation_mean,
    one_sided_mean_test,
)
from crbstpp.cli import _supervised_fit
from crbstpp.config import RunConfig
from crbstpp.data import Dataset
from crbstpp.evidence import (
    frequency_channel_evidence,
    overlapping_block_mean_test,
    prepare_risk_set_derivatives,
)
from crbstpp.ensemble import _fit_rule_effect_stack
from crbstpp.checkpoint import RESULT_SCHEMA, load_checkpoint
from crbstpp.pipeline import inspect_run, run
from crbstpp.reliability import (
    density_ratio_robust_test,
    EnvironmentSpec,
    environment_spec,
    environment_robust_lcb,
    environment_robust_pvalue,
    multinomial_l1_radius,
)
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import EMPTY_SUPPORT, RuleIdentity, Support
from crbstpp.search import _nonnegative_quadratic_minimum
from crbstpp.solver import fit_model_matrix

from tests.crbstpp.test_core import synthetic_dataset


class PipelineContractTests(unittest.TestCase):
    def test_intermediate_family_quadratic_solves_correlated_cone(self) -> None:
        linear = np.asarray([-2.0, -1.0, 0.5], dtype=np.float64)
        hessian = np.asarray(
            [[3.0, 1.0, 0.0], [1.0, 2.0, 0.25], [0.0, 0.25, 1.0]],
            dtype=np.float64,
        )
        value, weights, converged = _nonnegative_quadratic_minimum(
            linear,
            hessian,
            np.zeros(3, dtype=np.float64),
            tolerance=1.0e-10,
        )
        gradient = linear + hessian @ weights
        active = weights > 1.0e-8
        self.assertTrue(converged)
        self.assertLess(value, 0.0)
        np.testing.assert_allclose(gradient[active], 0.0, atol=1.0e-7)
        self.assertTrue(np.all(gradient[~active] >= -1.0e-7))

    def test_rule_effect_stack_reaches_nonnegative_cone_kkt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 240)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(
                data, lag=3, knot_count=2, cache_bytes=16 * 1024**2
            )
            baseline_matrix = engine.model_matrix(context, EMPTY_SUPPORT)
            baseline_fit = fit_model_matrix(
                baseline_matrix,
                likelihood=data.likelihood,
                tolerance=1.0e-9,
                max_iter=150,
            )
            supports = [
                Support.of((RuleIdentity((0,), 0, 1),)),
                Support.of((RuleIdentity((1,), 0, 1),)),
            ]
            matrices = [engine.model_matrix(context, support) for support in supports]
            fits = [
                fit_model_matrix(
                    matrix,
                    likelihood=data.likelihood,
                    tolerance=1.0e-9,
                    max_iter=150,
                )
                for matrix in matrices
            ]
            stack = _fit_rule_effect_stack(
                engine,
                context,
                baseline_matrix,
                baseline_fit,
                matrices,
                fits,
                tolerance=1.0e-9,
            )
            self.assertTrue(stack.converged)
            self.assertTrue(np.all(stack.weights >= 0.0))
            self.assertLessEqual(stack.nll, baseline_fit.nll + 1.0e-9)

    def test_unified_rule_stack_keeps_support_conditioned_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 240)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(
                data, lag=3, knot_count=2, cache_bytes=16 * 1024**2
            )
            baseline_matrix = engine.model_matrix(context, EMPTY_SUPPORT)
            baseline_fit = fit_model_matrix(
                baseline_matrix,
                likelihood=data.likelihood,
                tolerance=1.0e-9,
                max_iter=150,
            )
            first = RuleIdentity((0,), 0, 1)
            second = RuleIdentity((1,), 0, 1)
            supports = [
                Support.of((first,)),
                Support.of((first, second)),
            ]
            matrices = [engine.model_matrix(context, support) for support in supports]
            fits = [
                fit_model_matrix(
                    matrix,
                    likelihood=data.likelihood,
                    tolerance=1.0e-9,
                    max_iter=150,
                )
                for matrix in matrices
            ]
            stack = _fit_rule_effect_stack(
                engine,
                context,
                baseline_matrix,
                baseline_fit,
                matrices,
                fits,
                tolerance=1.0e-9,
                deduplicate_rules=False,
            )
            self.assertTrue(stack.converged)
            self.assertEqual(stack.rules.count(first), 2)
            self.assertEqual(
                stack.source_indices,
                ((0, 0), (1, 0), (1, 1)),
            )

    def test_overlapping_blocks_have_no_arbitrary_boundary(self) -> None:
        values = np.zeros(40, dtype=np.float64)
        values[19:22] = 1.0
        left = overlapping_block_mean_test(values, 10)
        shifted = overlapping_block_mean_test(np.roll(values, 1), 10)
        self.assertTrue(left.testable)
        self.assertTrue(shifted.testable)
        # A one-tick translation can change edge windows slightly, but unlike
        # disjoint blocks it cannot make the signal disappear at a boundary.
        self.assertGreater(left.statistic, 0.0)
        self.assertGreater(shifted.statistic, 0.0)
        self.assertLess(abs(left.statistic - shifted.statistic), 0.25)

    def test_frequency_channels_exactly_decompose_frozen_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 120)
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(
                data,
                lag=3,
                knot_count=2,
                cache_bytes=16 * 1024**2,
                baseline_time_bins=2,
            )
            rule = RuleIdentity((0,), 0, 1)
            null_matrix = engine.model_matrix(context, EMPTY_SUPPORT)
            full_support = Support.of((rule,))
            full_matrix = engine.model_matrix(context, full_support)
            null_fit = fit_model_matrix(
                null_matrix,
                likelihood=data.likelihood,
                tolerance=1e-9,
                max_iter=150,
            )
            full_fit = fit_model_matrix(
                full_matrix,
                likelihood=data.likelihood,
                tolerance=1e-9,
                max_iter=150,
            )
            self.assertTrue(null_fit.converged, null_fit.message)
            self.assertTrue(full_fit.converged, full_fit.message)
            rows = engine.footprint_rows(context, rule, 3)
            evidence = frequency_channel_evidence(
                engine,
                context,
                engine.model_metadata(EMPTY_SUPPORT, forced_closure=()),
                null_fit,
                engine.model_metadata(full_support),
                full_fit,
                rows=rows,
                dependence_horizon_ticks=6,
                prepared=prepare_risk_set_derivatives(
                    engine,
                    context,
                    engine.model_metadata(EMPTY_SUPPORT, forced_closure=()),
                    null_fit,
                ),
            )
            np.testing.assert_allclose(
                evidence.raw_entity_score,
                evidence.systemic_entity_score + evidence.relative_entity_score,
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertAlmostEqual(
                evidence.raw_information,
                evidence.systemic_information + evidence.relative_information,
                places=9,
            )
            self.assertIn(evidence.selected_channel, {"systemic", "relative"})

    def test_frequency_separation_is_connected_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90)
            config = RunConfig(
                dataset=str(data.root),
                run_root=str(root / "runs"),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                frequency_effect_separation=True,
                romano_wolf_resamples=1_000,
                solver_tolerance=1e-7,
                solver_max_iter=80,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                pricing_workers=1,
                exact_workers=1,
            )
            report = run(config, run_dir=root / "run-frequency")
            self.assertTrue(
                report.result["search"]["frequency_effect_separation"]["enabled"]
            )
            for item in report.result["certification"]["all"]:
                self.assertEqual(
                    item["certificate"]["multiplicity_method"],
                    "romano_wolf_stepdown_max_t_plus_holm_frequency_channel",
                )
                for rule in item["diagnostics"]["rules"]:
                    separation = rule["scalar_direction"]["frequency_effect_separation"]
                    self.assertIn(
                        separation["selected_channel"], {"systemic", "relative"}
                    )
                    self.assertEqual(
                        separation["selected_channel"],
                        rule["fit_frequency_effect_separation"]["selected_channel"],
                    )

    def test_reported_total_direction_is_frozen_from_fit_only(self) -> None:
        self.assertEqual(
            _frozen_reported_direction(np.asarray([0.0, 0.1, 0.2])),
            1,
        )
        self.assertEqual(
            _frozen_reported_direction(np.asarray([0.0, -0.1, -0.2])),
            -1,
        )
        self.assertEqual(_frozen_reported_direction(np.zeros(3)), 0)

    def test_romano_wolf_is_deterministic_and_uses_dependence(self) -> None:
        rng = np.random.default_rng(991)
        influence = rng.normal(size=4_000)
        influence = np.asarray(
            (influence - influence.mean()) / influence.std(ddof=1),
            dtype=np.float32,
        )
        component = _BootstrapComponent(2.2, influence)
        adjusted = _romano_wolf_adjust(
            [(component,), (component,)],
            resamples=20_000,
            seed=17,
        )
        repeated = _romano_wolf_adjust(
            [(component,), (component,)],
            resamples=20_000,
            seed=17,
        )
        np.testing.assert_array_equal(adjusted, repeated)
        self.assertAlmostEqual(adjusted[0], adjusted[1])
        # Perfectly overlapping hypotheses have one effective max-T test, not
        # a Bonferroni factor of two.
        self.assertLess(adjusted[0], 0.025)

    def test_supervisor_records_oom_sigkill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            with (
                mock.patch("crbstpp.cli._oom_kills", side_effect=(21, 22)),
                mock.patch(
                    "crbstpp.cli.subprocess.run",
                    return_value=SimpleNamespace(returncode=-9),
                ),
            ):
                code = _supervised_fit(root / "config.yaml", run_dir)
            self.assertEqual(code, -9)
            failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["reason"], "oom_kill")
            self.assertEqual(failure["signal"], 9)
            self.assertEqual(failure["oom_kill_delta"], 1)
            self.assertTrue((run_dir / "stderr.log").is_file())
            inspected = inspect_run(run_dir)
            self.assertFalse(inspected["complete"])
            self.assertIsNone(inspected["manifest"])
            self.assertEqual(inspected["failure"], failure)

    def test_f3_total_variation_worst_case_is_exact(self) -> None:
        values = np.asarray([1.0, 3.0])
        probabilities = np.asarray([0.5, 0.5])
        self.assertAlmostEqual(
            _worst_case_total_variation_mean(values, probabilities, 0.2),
            1.8,
        )
        self.assertAlmostEqual(
            _worst_case_total_variation_mean(values, probabilities, 2.0),
            1.0,
        )
        self.assertEqual(_multinomial_l1_radius(100, 1, 0.05), 0.0)
        self.assertGreater(_multinomial_l1_radius(100, 2, 0.05), 0.0)

    def test_f3_robust_pvalue_inverts_the_preregistered_lcb(self) -> None:
        rng = np.random.default_rng(20260723)
        inverse = np.repeat(np.arange(3, dtype=np.int64), 160)
        values = 0.04 + 0.01 * inverse + rng.normal(0.0, 0.08, len(inverse))
        counts = np.bincount(inverse).astype(np.float64)
        environments = EnvironmentSpec(
            inverse=inverse,
            labels=np.arange(3),
            counts=counts,
            probabilities=counts / counts.sum(),
            l1_radius=multinomial_l1_radius(len(inverse), len(counts), 0.05 / 2.0),
            source="test",
        )
        threshold = 0.005
        lower, _, _ = environment_robust_lcb(values, environments, alpha=0.05)
        pvalue, testable = environment_robust_pvalue(
            values, environments, threshold=threshold
        )
        self.assertTrue(testable)
        self.assertEqual(lower > threshold, pvalue <= 0.05)

    def test_f3_entity_density_ratio_test_separates_robust_gain(self) -> None:
        rng = np.random.default_rng(20260724)
        positive = rng.normal(0.08, 0.12, 2_000)
        negative = rng.normal(-0.02, 0.12, 2_000)
        accepted = density_ratio_robust_test(positive, alpha=0.05)
        rejected = density_ratio_robust_test(negative, alpha=0.05)
        self.assertTrue(accepted.testable)
        self.assertGreater(accepted.robust_gain, 0.0)
        self.assertLessEqual(accepted.pvalue, 0.05)
        self.assertTrue(rejected.testable)
        self.assertGreater(rejected.pvalue, 0.05)

    def test_recurrent_fixed_panel_uses_calendar_f3_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = synthetic_dataset(Path(directory) / "data", 60)
            data = replace(original, likelihood="poisson")
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0, 1),
                early_warning_horizon=3,
                pricing_devices=(),
            )
            environments = environment_spec(context, config)
            self.assertEqual(
                environments.source,
                "calendar_time_blocks/max_formation_plus_impact_horizon",
            )
            self.assertEqual(len(environments.labels), 2)
            self.assertEqual(environments.calibration_observations, data.n_entities)
            np.testing.assert_array_equal(environments.calendar_edges, [0, 5, 9])
            self.assertEqual(
                environments.inverse.shape,
                (data.n_entities * len(environments.labels),),
            )

    def test_calendar_block_sparse_loss_matches_dense_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = synthetic_dataset(Path(directory) / "data", 60)
            data = replace(original, likelihood="poisson")
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            support = Support.of((RuleIdentity((0,), 0, 1),))
            matrix = engine.model_matrix(context, support)
            fit = fit_model_matrix(
                matrix, likelihood="poisson", tolerance=1e-8, max_iter=150
            )
            self.assertTrue(fit.converged, fit.message)
            config = RunConfig(
                dataset=str(data.root),
                q_max=1,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0,),
                early_warning_horizon=3,
                pricing_devices=(),
            )
            environments = environment_spec(context, config)
            sparse = _calendar_block_losses_sparse(
                engine, context, matrix, fit, environments
            ).reshape(data.n_entities, -1)
            eta = engine.linear_predictor(context, matrix, fit.coefficients)
            event = np.zeros(context.n_grid)
            event[context.target_rows] = context.target_counts
            from crbstpp.likelihood import loss_rows

            dense_rows, _, _ = loss_rows(
                eta,
                likelihood="poisson",
                exposure_weight=np.ones(context.n_grid),
                noevent_weight=np.ones(context.n_grid),
                event_weight=event,
            )
            dense_grid = dense_rows.reshape(data.n_entities, 9)
            edges = np.asarray(environments.calendar_edges, dtype=np.int64)
            dense = np.column_stack(
                [
                    dense_grid[:, left:right].sum(axis=1) * (9.0 / float(right - left))
                    for left, right in zip(edges[:-1], edges[1:], strict=True)
                ]
            )
            np.testing.assert_allclose(sparse, dense, rtol=1e-12, atol=1e-12)

    def test_manifest_provenance_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            synthetic_dataset(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["provenance"]["generator"] = "tampered"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance digest"):
                Dataset.load(root)

    def test_branch_null_preserves_strict_lower_order_nuisance(self) -> None:
        pair = RuleIdentity((0, 1), 2, 1)
        support = Support.of((pair,))
        from crbstpp.rules import hierarchy_closure

        closure = hierarchy_closure(support)
        retained = _branch_null_closure(closure, EMPTY_SUPPORT, pair)
        self.assertEqual(retained, closure)

    def test_sparse_entity_loss_matches_dense_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_dataset(Path(directory) / "data", 60)
            context = Context.make(data, np.arange(60, dtype=np.int32))
            engine = ResponseEngine(data, lag=3, knot_count=2, cache_bytes=8 * 1024**2)
            support = Support.of((RuleIdentity((0,), 0, 1),))
            matrix = engine.model_matrix(context, support)
            fit = fit_model_matrix(
                matrix, likelihood=data.likelihood, tolerance=1e-8, max_iter=150
            )
            self.assertTrue(fit.converged, fit.message)
            sparse = _entity_losses_sparse(engine, context, matrix, fit)
            eta = engine.linear_predictor(context, matrix, fit.coefficients)
            event = np.zeros(context.n_grid)
            event[context.target_rows] = context.target_counts
            from crbstpp.likelihood import loss_rows

            dense_rows, _, _ = loss_rows(
                eta,
                likelihood=data.likelihood,
                exposure_weight=np.ones(context.n_grid),
                noevent_weight=1.0 - event,
                event_weight=event,
            )
            dense = np.add.reduceat(dense_rows, context.offsets[:-1])
            np.testing.assert_allclose(sparse, dense, rtol=1e-12, atol=1e-12)

    def test_frozen_shape_scalar_score_matches_dense_reference(self) -> None:
        for likelihood in ("poisson", "first_event_cloglog"):
            with (
                self.subTest(likelihood=likelihood),
                tempfile.TemporaryDirectory() as directory,
            ):
                data = synthetic_dataset(
                    Path(directory) / "data",
                    180,
                    likelihood=likelihood,
                )
                context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
                engine = ResponseEngine(
                    data, lag=3, knot_count=2, cache_bytes=16 * 1024**2
                )
                null_matrix = engine.model_matrix(context, EMPTY_SUPPORT)
                full_matrix = engine.model_matrix(
                    context,
                    Support.of((RuleIdentity((0,), 0, 1),)),
                )
                null_fit = fit_model_matrix(
                    null_matrix,
                    likelihood=likelihood,
                    tolerance=1e-9,
                    max_iter=150,
                )
                full_fit = fit_model_matrix(
                    full_matrix,
                    likelihood=likelihood,
                    tolerance=1e-9,
                    max_iter=150,
                )
                self.assertTrue(null_fit.converged, null_fit.message)
                self.assertTrue(full_fit.converged, full_fit.message)
                test, component, diagnostics = _scalar_direction_score(
                    engine,
                    context,
                    engine.model_metadata(EMPTY_SUPPORT, forced_closure=()),
                    null_fit,
                    engine.model_metadata(full_matrix.support),
                    full_fit,
                )
                null_eta = engine.linear_predictor(
                    context, null_matrix, null_fit.coefficients
                )
                full_eta = engine.linear_predictor(
                    context, full_matrix, full_fit.coefficients
                )
                event = np.zeros(context.n_grid, dtype=np.float64)
                event[context.target_rows] = context.target_counts
                exposure = np.full(
                    context.n_grid,
                    1.0 / data.ticks_per_unit if likelihood == "poisson" else 1.0,
                    dtype=np.float64,
                )
                from crbstpp.likelihood import loss_rows

                _, first, second = loss_rows(
                    null_eta,
                    likelihood=likelihood,
                    exposure_weight=exposure,
                    noevent_weight=(
                        exposure - event
                        if likelihood == "first_event_cloglog"
                        else exposure
                    ),
                    event_weight=event,
                )
                direction = full_eta - null_eta
                dense_entity_score = np.add.reduceat(
                    -first * direction, context.offsets[:-1]
                )
                expected = one_sided_mean_test(dense_entity_score)
                self.assertAlmostEqual(test.mean, expected.mean, places=11)
                self.assertAlmostEqual(
                    test.standard_error, expected.standard_error, places=11
                )
                self.assertAlmostEqual(test.statistic, expected.statistic, places=10)
                self.assertAlmostEqual(
                    float(diagnostics["conditional_information"]),
                    float(np.sum(second * direction * direction)),
                    places=9,
                )
                self.assertEqual(diagnostics["degrees_of_freedom"], 1)
                self.assertFalse(diagnostics["amplitude_fitted_on_D_cert"])
                self.assertIsNotNone(component)

    def test_run_resume_and_existing_directory_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 60)
            config = RunConfig(
                dataset=str(data.root),
                run_root=str(root / "runs"),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                solver_tolerance=1e-7,
                # A single iteration is deliberately only a continuation
                # checkpoint; exact search/certification fits must still
                # reach their KKT-certified optima.
                solver_max_iter=1,
                cache_bytes=8 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
            )
            run_dir = root / "run"
            report = run(config, run_dir=run_dir)
            self.assertEqual(report.schema, RESULT_SCHEMA)
            self.assertIn("f0", report.result["certification"]["all"][0]["certificate"])
            self.assertIn("f3", report.result["certification"]["all"][0]["certificate"])
            for item in report.result["certification"]["all"]:
                certificate = item["certificate"]
                diagnostics = item["diagnostics"]
                self.assertEqual(
                    certificate["f1_pvalue"],
                    diagnostics["f1_fixed_predictor_nll"]["pvalue"],
                )
                self.assertEqual(
                    diagnostics["f3"]["certification_materiality_threshold"],
                    0.0,
                )
                self.assertGreater(
                    diagnostics["f3"]["discovery_mdl_materiality_reference_per_entity"],
                    0.0,
                )
                self.assertGreaterEqual(
                    certificate["family_adjusted_pvalue"],
                    certificate["family_pvalue"],
                )
                for pvalue, rule in zip(
                    certificate["f2_pvalues"],
                    diagnostics["rules"],
                    strict=True,
                ):
                    self.assertIn(rule["reported_total_sign"], {-1, 0, 1})
                    self.assertIn(
                        rule["reported_total_direction"],
                        {"excitation", "inhibition", "unidentified"},
                    )
                    for field in (
                        "mean",
                        "standard_error",
                        "statistic",
                        "pvalue",
                        "testable",
                    ):
                        self.assertEqual(
                            rule["probability"][field],
                            rule["total_contextual_probability"][field],
                        )
                    self.assertEqual(
                        pvalue,
                        max(
                            rule["scalar_direction"]["pvalue"],
                            rule["probability"]["pvalue"],
                        ),
                    )
                    self.assertEqual(rule["scalar_direction"]["degrees_of_freedom"], 1)
                if certificate["certified"]:
                    self.assertLessEqual(certificate["f1_pvalue"], config.alpha)
                    self.assertTrue(certificate["f3"])
                    self.assertTrue(
                        all(
                            pvalue <= config.alpha
                            for pvalue in certificate["f2_pvalues"]
                        )
                    )
            self.assertIn(
                "exact_branch_add_audits",
                report.result["search"]["diagnostics"],
            )
            resumed = run(config, run_dir=run_dir, resume=True)
            self.assertEqual(resumed.result, report.result)
            repeated = run(config, run_dir=root / "run-repeat")
            self.assertEqual(repeated.result, report.result)
            with self.assertRaises(FileExistsError):
                run(config, run_dir=run_dir)

    def test_successor_route_process_shards_match_serial_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 120)
            common = dict(
                dataset=str(data.root),
                run_root=str(root / "runs"),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                search_mode="successor_rashomon_path",
                adaptive_gradient_racing=True,
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                romano_wolf_resamples=1000,
            )
            serial = run(
                RunConfig(
                    **common,
                    exact_workers=1,
                    pricing_workers=2,
                    pricing_devices=("cpu",),
                    route_workers=1,
                ),
                run_dir=root / "serial",
            )
            parallel = run(
                RunConfig(
                    **common,
                    exact_workers=2,
                    pricing_workers=2,
                    pricing_devices=("cpu", "cpu"),
                    route_workers=2,
                ),
                run_dir=root / "parallel",
            )
            self.assertEqual(
                [item["key"] for item in serial.result["family"]],
                [item["key"] for item in parallel.result["family"]],
            )
            self.assertEqual(
                {
                    path["start"]: path["terminal"]
                    for path in serial.result["search"]["paths"]
                },
                {
                    path["start"]: path["terminal"]
                    for path in parallel.result["search"]["paths"]
                },
            )
            self.assertEqual(
                serial.result["certification"]["selected_supports"],
                parallel.result["certification"]["selected_supports"],
            )
            for name in (
                "score_basin_nodes",
                "adaptive_gradient_root_exact_fits",
                "adaptive_gradient_root_exact_fits_avoided",
                "multi_source_roots",
                "route_family_active_roots",
            ):
                self.assertEqual(
                    serial.result["search"]["diagnostics"][name],
                    parallel.result["search"]["diagnostics"][name],
                )
            self.assertGreater(
                parallel.result["search"]["diagnostics"]["block_score_evaluations"],
                0,
            )

    def test_atomic_frontier_parallel_routes_preserve_global_family_frontier(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 120)
            common = dict(
                dataset=str(data.root),
                run_root=str(root / "runs"),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                search_mode="atomic_rashomon_frontier",
                adaptive_gradient_racing=True,
                solver_tolerance=1e-7,
                solver_max_iter=120,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                romano_wolf_resamples=1000,
            )
            serial = run(
                RunConfig(
                    **common,
                    exact_workers=1,
                    pricing_workers=2,
                    pricing_devices=("cpu",),
                    route_workers=1,
                ),
                run_dir=root / "atomic-serial",
            )
            config = RunConfig(
                **common,
                exact_workers=2,
                pricing_workers=2,
                pricing_devices=("cpu", "cpu"),
                route_workers=2,
            )
            run_dir = root / "atomic-frontier"
            report = run(config, run_dir=run_dir)
            self.assertEqual(
                report.result["search"]["route_policy"],
                "atomic_descendant_safe_shared_rashomon_frontier",
            )
            self.assertGreater(
                report.result["search"]["diagnostics"]["atomic_frontier_regions"],
                0,
            )
            self.assertGreater(
                report.result["search"]["diagnostics"]["global_family_frontier_rounds"],
                0,
            )
            self.assertGreater(
                report.result["search"]["diagnostics"]["global_family_frontier_states"],
                0,
            )
            self.assertTrue((run_dir / "_route_shards").is_dir())
            self.assertEqual(
                [item["key"] for item in serial.result["family"]],
                [item["key"] for item in report.result["family"]],
            )
            self.assertEqual(
                {
                    path["start"]: path["terminal"]
                    for path in serial.result["search"]["paths"]
                },
                {
                    path["start"]: path["terminal"]
                    for path in report.result["search"]["paths"]
                },
            )
            self.assertEqual(
                serial.result["certification"]["selected_supports"],
                report.result["certification"]["selected_supports"],
            )
            repeated = run(config, run_dir=root / "atomic-frontier-repeat")
            self.assertEqual(report.result, repeated.result)

    def test_safe_column_route_process_shards_match_serial_search(self) -> None:
        """Route sharding changes scheduling, not safe-column terminals."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90)
            common = dict(
                dataset=str(data.root),
                run_root=str(root / "runs"),
                q_max=2,
                impact_lag=3,
                knot_count=2,
                formation_windows=(0, 1),
                search_mode="safe_column_generation",
                adaptive_gradient_racing=False,
                solver_tolerance=1e-7,
                solver_max_iter=100,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                romano_wolf_resamples=1000,
            )
            serial = run(
                RunConfig(
                    **common,
                    exact_workers=1,
                    pricing_workers=2,
                    pricing_devices=("cpu",),
                    route_workers=1,
                ),
                run_dir=root / "safe-serial",
            )
            parallel = run(
                RunConfig(
                    **common,
                    exact_workers=2,
                    pricing_workers=2,
                    pricing_devices=("cpu", "cpu"),
                    route_workers=2,
                ),
                run_dir=root / "safe-parallel",
            )
            self.assertEqual(
                [item["key"] for item in serial.result["family"]],
                [item["key"] for item in parallel.result["family"]],
            )
            self.assertEqual(
                {
                    path["start"]: path["terminal"]
                    for path in serial.result["search"]["paths"]
                },
                {
                    path["start"]: path["terminal"]
                    for path in parallel.result["search"]["paths"]
                },
            )
            self.assertEqual(
                serial.result["certification"]["selected_supports"],
                parallel.result["certification"]["selected_supports"],
            )

    def test_quantile_w_and_romano_wolf_are_connected_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90)
            config = RunConfig(
                dataset=str(data.root),
                run_root=str(root / "runs"),
                q_max=2,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0, 1, 2, 3),
                formation_window_mode="fit_quantile",
                formation_window_quantiles=(0.25, 0.5, 0.75, 0.9),
                romano_wolf_resamples=1_000,
                solver_tolerance=1e-7,
                solver_max_iter=80,
                cache_bytes=16 * 1024**2,
                early_warning_horizon=3,
                pricing_devices=(),
                pricing_workers=1,
                exact_workers=1,
            )
            report = run(config, run_dir=root / "run-quantile")
            search = report.result["search"]
            self.assertEqual(search["formation_window_mode"], "fit_quantile")
            self.assertEqual(
                search["formation_window_quantiles"], [0.25, 0.5, 0.75, 0.9]
            )
            self.assertIn("0,1", search["frozen_window_dictionary"])
            self.assertIn("0,1", search["frozen_window_quantile_labels"])
            self.assertEqual(
                set(search["frozen_window_quantile_labels"]["0,1"]),
                {str(window) for window in search["frozen_window_dictionary"]["0,1"]},
            )
            self.assertEqual(
                search["objective"],
                "common_baseline_total_state_rule_MDL_with_local_"
                "representation_audit_and_family_intensity_mixture_MDL_selection",
            )
            self.assertGreater(
                search["diagnostics"]["family_ensemble_objective_audits"],
                0,
            )
            self.assertGreaterEqual(
                search["diagnostics"]["route_family_candidates"],
                search["diagnostics"]["route_family_active_roots"],
            )
            for item in report.result["certification"]["all"]:
                certificate = item["certificate"]
                self.assertEqual(
                    certificate["multiplicity_method"],
                    "romano_wolf_stepdown_max_t",
                )
                self.assertEqual(certificate["romano_wolf_resamples"], 1_000)

    def test_ipw_discovery_is_refit_on_complete_fit_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 120, explicit_partition=True)
            config = RunConfig(
                dataset=str(data.root),
                run_root=str(root / "runs"),
                q_max=1,
                impact_lag=3,
                knot_count=1,
                formation_windows=(0,),
                early_warning_horizon=3,
                discovery_sampling="case_cohort_ipw",
                discovery_noncase_fraction=0.25,
                search_mode="fast_block_score",
                pricing_devices=("cpu",),
                pricing_workers=1,
                exact_workers=1,
                cache_bytes=8 * 1024**2,
            )
            run_dir = root / "run-ipw"
            report = run(config, run_dir=run_dir)
            sampling = report.result["search"]["discovery_sampling"]
            self.assertEqual(sampling["method"], "case_cohort_ipw")
            self.assertLess(
                sampling["sample_entities"], sampling["population_entities"]
            )
            self.assertGreaterEqual(
                report.result["search"]["full_verification_rejections"], 0
            )
            for record in report.result["family"]:
                self.assertGreater(record["score"], config.search_tolerance)
                self.assertLessEqual(record["projected_kkt"], 5e-6)
            timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
            self.assertIn("full_verification", timing)

    def test_corrupted_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "unsupported.checkpoint",
                        "config_digest": "a",
                        "dataset_digest": "b",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_checkpoint(path, config_digest="a", dataset_digest="b")

    def test_package_has_no_external_script_import(self) -> None:
        package = Path(__file__).parents[2] / "src" / "crbstpp"
        forbidden = {"scripts"}
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""]
                    )
                    self.assertTrue(
                        forbidden.isdisjoint(name.split(".")[0] for name in names),
                        str(path),
                    )


if __name__ == "__main__":
    unittest.main()
