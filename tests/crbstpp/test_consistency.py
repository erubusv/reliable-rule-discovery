from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from crbstpp.consistency import compare_runs


class ConsistencyTests(unittest.TestCase):
    def test_report_compares_labelled_supports_ensemble_and_nll(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for seed, window, test_nll in ((11, 2, 8.0), (12, 3, 7.5)):
                run = root / f"seed{seed}"
                run.mkdir()
                (run / "config.yaml").write_text(
                    yaml.safe_dump({"split_seed": seed}), encoding="utf-8"
                )
                (run / "timing.json").write_text(
                    json.dumps({"search": 1.0, "total": 2.0}),
                    encoding="utf-8",
                )
                key = f"0,1|W{window}|exc"
                result = {
                    "search": {
                        "frozen_window_quantile_labels": {"0,1": {str(window): ["Q50"]}}
                    },
                    "certification": {
                        "all": [
                            {
                                "certificate": {
                                    "support_key": key,
                                    "certified": True,
                                },
                                "diagnostics": {
                                    "rules": [
                                        {
                                            "reported_total_sign": (
                                                -1 if seed == 11 else 1
                                            )
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                    "family": [
                        {
                            "key": key,
                            "rules": [
                                {
                                    "antecedent": [0, 1],
                                    "window": window,
                                    "sign": 1,
                                    "lag_profile": [0.5, 0.25],
                                }
                            ],
                        }
                    ],
                    "ensemble": {
                        "supports": [
                            [
                                {
                                    "antecedent": [0, 1],
                                    "window": window,
                                    "sign": 1,
                                }
                            ]
                        ],
                        "weights": [1.0],
                        "baseline_weight": 0.0,
                        "train_nll": 6.0,
                        "test_nll": test_nll,
                        "baseline_test_nll": 10.0,
                    },
                }
                (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
                runs.append(run)

            report = compare_runs(tuple(runs), output=root / "report.json")
            exact = report["stability"]["exact_certified_frequency"]
            self.assertEqual(exact[0]["identity"], "0,1|Q50|exc")
            self.assertEqual(exact[0]["count"], 2)
            self.assertEqual(
                report["stability"]["mean_pairwise_exact_certified_jaccard"],
                1.0,
            )
            self.assertTrue(report["prediction"]["all_runs_improve_baseline"])
            self.assertEqual(report["prediction"]["test_nll_improvement_mean"], 2.25)
            reported = report["stability"]["reported_total_certified_frequency"]
            self.assertEqual(len(reported), 2)
            self.assertEqual({item["count"] for item in reported}, {1})
            self.assertEqual(
                report["stability"]["mean_pairwise_reported_total_certified_jaccard"],
                0.0,
            )
            self.assertTrue((root / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
