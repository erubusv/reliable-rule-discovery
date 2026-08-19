from __future__ import annotations

import json
from pathlib import Path

import yaml
import numpy as np

from crbstpp.metric_report import collect_metric_report
from crbstpp.response import Context
from crbstpp.rule_prediction import _landmark_next_rows


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_metric_report_keeps_incomparable_likelihoods_separate(tmp_path: Path) -> None:
    baseline = tmp_path / "baselines" / "seed-111"
    _write_json(
        baseline / "logistic" / "result.json",
        {
            "model": "logistic",
            "seed": 111,
            "dataset_digest": "data",
            "elapsed_seconds": 2.0,
            "details": {
                "test": {
                    "binary_nll": 0.2,
                    "auroc": 0.7,
                    "brier": 0.1,
                    "n": 10,
                    "targets": 2,
                }
            },
        },
    )
    _write_json(
        baseline / "rmtpp" / "result.json",
        {
            "model": "rmtpp",
            "seed": 111,
            "dataset_digest": "data",
            "elapsed_seconds": 3.0,
            "details": {
                "test": {"loglike": -1.5, "acc": 0.4, "rmse": 2.5, "num_events": 20}
            },
        },
    )
    _write_json(
        baseline / "rmtpp" / "target_metrics.json",
        {
            "target": {
                "target_nll_per_entity": 1.25,
                "target_nll_total": 2.5,
            },
            "test": {
                "binary_nll": 0.3,
                "brier": 0.12,
                "n": 10,
                "targets": 2,
            },
        },
    )
    ours = tmp_path / "ours"
    _write_json(
        ours / "manifest.json",
        {"created_at_utc": "2026-01-01T00:00:00+00:00"},
    )
    _write_json(
        ours / "result.json",
        {
            "dataset_digest": "data",
            "split_sizes": {"fit": 5, "cert": 3, "test": 2},
            "ensemble": {
                "test_nll": 6.0,
                "baseline_test_nll": 8.0,
                "active_rule_effect_count": 1,
                "active_support_count": 1,
            },
            "certification": {
                "family_size": 1,
                "certified_count": 1,
                "selected_count": 1,
                "selected_supports": ["rule"],
                "all": [
                    {
                        "certificate": {
                            "f0": True,
                            "f3": True,
                            "family_adjusted_pvalue": 0.01,
                            "reasons": [],
                        }
                    }
                ],
            },
            "family": [
                {
                    "key": "rule",
                    "rules": [
                        {
                            "antecedent": [0, 1],
                            "window": 3,
                            "sign": -1,
                            "relation": "unordered",
                            "direction": "inhibition",
                        }
                    ],
                }
            ],
            "search": {"terminal_count": 1, "positive_atom_count": 2, "diagnostics": {}},
        },
    )
    tex = tmp_path / "rows.tex"
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "metrics"),
                "tex_output": str(tex),
                "datasets": [
                    {
                        "key": "aave",
                        "baseline_seed_dir": str(baseline),
                        "ours_run_dir": str(ours),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = collect_metric_report(spec)
    models = report["datasets"][0]["models"]
    assert models["logistic"]["binary_nll_per_landmark"] == 0.2
    assert models["logistic"]["target_nll_per_entity"] is None
    assert models["rmtpp"]["joint_nll"] == 1.5
    assert models["rmtpp"]["target_nll_per_entity"] == 1.25
    assert models["rmtpp"]["binary_nll_per_landmark"] == 0.3
    assert models["rmtpp"]["brier"] == 0.12
    assert models["ours"]["target_nll_per_entity"] == 3.0
    assert models["ours"]["target_nll_gain_per_entity"] == 1.0
    assert models["ours"]["relative_target_nll_reduction"] == 0.25
    rules = report["datasets"][0]["rule_metrics"]["ours"]
    assert rules["direction_counts"] == {"excitation": 0, "inhibition": 1}
    assert rules["order_counts"]["pair"] == 1
    assert "Logistic regression & -- & 0.2000 & 0.100000" in tex.read_text(
        encoding="utf-8"
    )


def test_landmarks_use_the_first_strictly_future_row() -> None:
    context = Context(
        dataset=None,  # type: ignore[arg-type]
        entity_codes=np.asarray([2], dtype=np.int32),
        entity_lookup=np.asarray([-1, -1, 0], dtype=np.int32),
        starts=np.asarray([100], dtype=np.int64),
        ends=np.asarray([140], dtype=np.int64),
        baseline_origins=np.asarray([100], dtype=np.int64),
        baseline_strata=np.asarray([0], dtype=np.int16),
        offsets=np.asarray([0, 4], dtype=np.int64),
        row_times=np.asarray([100, 110, 120, 130], dtype=np.int64),
        baseline_row_strata=np.zeros(4, dtype=np.int16),
        baseline_row_exposure=np.ones(4, dtype=np.float64),
        n_grid=4,
        target_rows=np.zeros(0, dtype=np.int64),
        target_counts=np.zeros(0, dtype=np.float64),
        entity_weights=np.ones(1, dtype=np.float64),
        uniform_entity_weight=1.0,
        population_entities=1,
    )
    rows, valid = _landmark_next_rows(
        context,
        np.asarray([2, 2, 2], dtype=np.int32),
        np.asarray([100, 119, 130], dtype=np.int64),
    )
    assert rows.tolist() == [1, 2, -1]
    assert valid.tolist() == [True, True, False]
