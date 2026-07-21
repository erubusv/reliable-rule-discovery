from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from crbstpp.certification import (
    _branch_null_closure,
    _entity_losses_sparse,
    _multinomial_l1_radius,
    _worst_case_total_variation_mean,
)
from crbstpp.cli import _supervised_fit
from crbstpp.config import RunConfig
from crbstpp.data import Dataset
from crbstpp.checkpoint import RESULT_SCHEMA, load_checkpoint
from crbstpp.pipeline import inspect_run, run
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import EMPTY_SUPPORT, RuleIdentity, Support
from crbstpp.solver import fit_model_matrix

from tests.crbstpp.test_core import synthetic_dataset


class PipelineContractTests(unittest.TestCase):
    def test_supervisor_records_oom_sigkill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            with (
                mock.patch(
                    "crbstpp.cli._oom_kills", side_effect=(21, 22)
                ),
                mock.patch(
                    "crbstpp.cli.subprocess.run",
                    return_value=SimpleNamespace(returncode=-9),
                ),
            ):
                code = _supervised_fit(root / "config.yaml", run_dir)
            self.assertEqual(code, -9)
            failure = json.loads(
                (run_dir / "failure.json").read_text(encoding="utf-8")
            )
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
            self.assertGreater(
                report.result["search"]["diagnostics"][
                    "nonattained_exact_rejections"
                ],
                0,
            )
            resumed = run(config, run_dir=run_dir, resume=True)
            self.assertEqual(resumed.result, report.result)
            repeated = run(config, run_dir=root / "run-repeat")
            self.assertEqual(repeated.result, report.result)
            with self.assertRaises(FileExistsError):
                run(config, run_dir=run_dir)

    def test_corrupted_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "certscr.checkpoint",
                        "config_digest": "a",
                        "dataset_digest": "b",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "legacy or unsupported"):
                load_checkpoint(path, config_digest="a", dataset_digest="b")

    def test_new_package_has_no_legacy_import(self) -> None:
        package = Path(__file__).parents[2] / "src" / "crbstpp"
        forbidden = {"certscr", "scripts", "legacy"}
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
