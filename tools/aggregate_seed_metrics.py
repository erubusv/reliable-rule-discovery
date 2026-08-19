from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "target_nll_per_entity",
    "binary_nll_per_landmark",
    "brier",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate comparable prediction metrics across experiment seeds."
    )
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    by_dataset: dict[str, dict[str, dict[str, list[float]]]] = {}
    labels: dict[str, str] = {}
    for report in reports:
        for dataset in report["datasets"]:
            key = str(dataset["key"])
            labels[key] = str(dataset.get("label", key))
            models = by_dataset.setdefault(key, {})
            for model, result in dataset["models"].items():
                values = models.setdefault(str(model), {metric: [] for metric in METRICS})
                for metric in METRICS:
                    value = result.get(metric)
                    if value is not None and math.isfinite(float(value)):
                        values[metric].append(float(value))

    datasets = []
    for key in sorted(by_dataset):
        models = {}
        for model in sorted(by_dataset[key]):
            metrics = {}
            for metric, raw_values in by_dataset[key][model].items():
                values = np.asarray(raw_values, dtype=np.float64)
                metrics[metric] = {
                    "mean": None if not len(values) else float(values.mean()),
                    "std": None if len(values) < 2 else float(values.std(ddof=1)),
                    "n": int(len(values)),
                }
            models[model] = metrics
        datasets.append({"key": key, "label": labels[key], "models": models})

    output = {
        "schema": "crbstpp.seed-metric-aggregate.v1",
        "input_reports": [str(path) for path in args.input],
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
