from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from crbstpp.baselines.config import BaselineConfig
from crbstpp.baselines.data import (
    build_easytpp_payload,
    build_landmarks,
    load_landmarks,
    prepare_baseline_data,
)
from crbstpp.baselines.easytpp import easytpp_yaml
from crbstpp.baselines.logical import fit_branch_price
from crbstpp.baselines.metrics import classification_metrics, roc_auc
from crbstpp.baselines.statistical import fit_point_process_baseline
from tests.crbstpp.test_core import synthetic_dataset


class BaselineContractTests(unittest.TestCase):
    def config(self, data: Path, run_root: Path) -> BaselineConfig:
        return BaselineConfig(
            dataset=data,
            dataset_id="synthetic",
            run_root=run_root,
            models=("baseline_tpp", "hawkes", "branch_price"),
            warning_horizon=2,
            history_horizon=3,
            effect_horizon=3,
            baseline_time_bins=1,
            device="cpu",
            num_workers=1,
            max_sequence_length=16,
            sequence_context_length=4,
            logical_max_rules=3,
            logical_time_limit_seconds=10.0,
            hawkes_half_lives=(1.0, 3.0),
        )

    def test_landmarks_are_frozen_split_and_strict_future(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90, explicit_partition=True)
            config = self.config(data.root, root / "runs")
            landmarks = build_landmarks(data, config)
            for split in range(3):
                expected = set(np.flatnonzero(data.partitions == split).tolist())
                observed = set(landmarks[split].entity_codes.tolist())
                self.assertTrue(observed.issubset(expected))
                self.assertTrue(expected.issubset(observed))
            # Target at t=3 is positive from query t=1 or t=2, but never from
            # a query at t=3 because labels use the strict interval (t, t+h].
            for split in range(3):
                rows = landmarks[split]
                at_three = rows.times == 3
                self.assertFalse(np.any(rows.outcomes[at_three]))

    def test_prepared_cache_and_gatech_target_order_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90, explicit_partition=True)
            config = self.config(data.root, root / "runs")
            first = prepare_baseline_data(config)
            second = prepare_baseline_data(config)
            self.assertEqual(first.root, second.root)
            landmarks = load_landmarks(first.landmarks_path)
            self.assertEqual(landmarks[0].features.shape[1], data.n_reported_predicates)
            payload = build_easytpp_payload(data, config)
            self.assertEqual(payload["dim_process"], data.n_reported_predicates + 1)
            target_type = data.n_reported_predicates
            for sequence in payload["train"]:
                times = [item["time_since_start"] for item in sequence]
                self.assertEqual(times, sorted(times))
                target_positions = [
                    position
                    for position, item in enumerate(sequence)
                    if item["type_event"] == target_type
                ]
                self.assertLessEqual(len(target_positions), 1)
                if target_positions:
                    self.assertEqual(target_positions[0], len(sequence) - 1)

    def test_point_process_baselines_fit_both_null_and_hawkes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90, explicit_partition=True)
            config = self.config(data.root, root / "runs")
            null = fit_point_process_baseline(
                data, config, source_dimension=0, output_dir=root / "null"
            )
            hawkes = fit_point_process_baseline(
                data,
                config,
                source_dimension=data.n_reported_predicates,
                output_dir=root / "hawkes",
            )
            self.assertTrue(null["converged"])
            self.assertTrue(hawkes["converged"])
            self.assertTrue(np.isfinite(null["test_nll_per_entity"]))
            self.assertTrue(np.isfinite(hawkes["test_nll_per_entity"]))

    def test_branch_price_never_exceeds_declared_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90, explicit_partition=True)
            config = self.config(data.root, root / "runs")
            result = fit_branch_price(
                build_landmarks(data, config),
                data,
                config,
                seed=17,
                output_dir=root / "logic",
            )
            self.assertTrue(
                all(
                    len(rule["antecedent"]) <= config.logical_max_order
                    for rule in result["rules"]
                )
            )

    def test_easytpp_yaml_receives_the_single_cli_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = synthetic_dataset(root / "data", 90, explicit_partition=True)
            config = self.config(data.root, root / "runs")
            prepared = prepare_baseline_data(config)
            payload = easytpp_yaml(
                "rmtpp",
                config,
                prepared,
                data,
                seed=918273,
                output_dir=root / "rmtpp",
            )
            experiment = payload["RMTPP_train"]
            self.assertEqual(experiment["trainer_config"]["seed"], 918273)

    def test_metric_reference_values(self) -> None:
        y = np.asarray([0, 0, 1, 1], dtype=np.int8)
        p = np.asarray([0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(roc_auc(y, p), 0.75)
        metrics = classification_metrics(y, p)
        self.assertEqual(metrics["targets"], 2)
        self.assertGreater(metrics["binary_nll"], 0.0)


if __name__ == "__main__":
    unittest.main()
