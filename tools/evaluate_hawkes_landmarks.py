from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from crbstpp.baselines.config import BaselineConfig
from crbstpp.baselines.data import load_landmarks
from crbstpp.baselines.metrics import classification_metrics
from crbstpp.baselines.runner import prepare_baselines
from crbstpp.data import Dataset
from crbstpp.response import Context
from tools.integrated_rule_metrics import _test_landmarks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    config = BaselineConfig.from_yaml(args.config)
    dataset = Dataset.load(config.dataset)
    run_dir = config.run_root / config.dataset_id / f"seed-{args.seed}" / "hawkes"
    result = json.loads((run_dir / "result.json").read_text())
    model = np.load(run_dir / "model.npz")
    parameters = np.asarray(model["parameters"], dtype=np.float64)
    half_life = float(np.asarray(model["half_life"]).ravel()[0])
    beta = math.log(2.0) / half_life
    dimension = dataset.n_reported_predicates
    baseline = parameters[:-dimension]
    coefficient = parameters[-dimension:]
    test_codes = np.flatnonzero(dataset.partitions == 2).astype(np.int32)
    context = Context.make(dataset, test_codes)
    if args.baseline_config is not None:
        landmarks = load_landmarks(
            prepare_baselines(
                BaselineConfig.from_yaml(args.baseline_config), seed=args.seed
            ).landmarks_path
        )[2]
    else:
        landmarks = _test_landmarks(dataset, 0, config.warning_horizon)

    event_count = np.bincount(dataset.event_entities, minlength=dataset.n_entities)
    event_offset = np.r_[0, np.cumsum(event_count, dtype=np.int64)]
    state_score = np.zeros(len(landmarks.outcomes), dtype=np.float64)
    scale = float(dataset.ticks_per_unit)
    for entity in np.unique(landmarks.entity_codes):
        selected = np.flatnonzero(landmarks.entity_codes == entity)
        query = landmarks.times[selected]
        e0, e1 = int(event_offset[entity]), int(event_offset[entity + 1])
        times = dataset.event_times[e0:e1]
        predicates = dataset.event_predicates[e0:e1]
        keep = predicates < dimension
        times, predicates = times[keep], predicates[keep]
        state = np.zeros(dimension, dtype=np.float64)
        cursor = 0
        previous = float(dataset.start_times[entity])
        for destination, current_query in zip(selected, query, strict=True):
            while cursor < len(times) and times[cursor] <= current_query:
                current = float(times[cursor])
                state *= math.exp(-beta * (current - previous) / scale)
                same = times[cursor]
                while cursor < len(times) and times[cursor] == same:
                    state[int(predicates[cursor])] += (
                        beta if dataset.likelihood == "continuous_poisson" else 1.0
                    )
                    cursor += 1
                previous = current
            state *= math.exp(-beta * (float(current_query) - previous) / scale)
            previous = float(current_query)
            state_score[destination] = float(state @ coefficient)

    available = np.maximum(
        0,
        dataset.end_times[landmarks.entity_codes].astype(np.int64)
        - landmarks.times.astype(np.int64),
    )
    horizon_ticks = np.minimum(
        available,
        int(config.warning_horizon) * int(dataset.ticks_per_unit),
    )
    if dataset.likelihood == "continuous_poisson":
        horizon_units = horizon_ticks.astype(np.float64) / scale
        kernel_hazard = state_score * (-np.expm1(-beta * horizon_units)) / beta
        all_rows = np.arange(context.n_grid, dtype=np.int64)
        groups = context.temporal_baseline_groups_at_rows(
            all_rows, time_bins=config.baseline_time_bins
        )
        row_hazard = baseline[groups] * context.baseline_row_exposure
        baseline_hazard = np.zeros(len(landmarks.outcomes), dtype=np.float64)
        lookup = {int(code): local for local, code in enumerate(context.entity_codes)}
        for entity in np.unique(landmarks.entity_codes):
            selected = np.flatnonzero(landmarks.entity_codes == entity)
            local = lookup[int(entity)]
            left, right = int(context.offsets[local]), int(context.offsets[local + 1])
            times = context.row_times[left:right]
            prefix = np.r_[0.0, np.cumsum(row_hazard[left:right])]
            lo = np.searchsorted(times, landmarks.times[selected] + 1, side="left")
            end = landmarks.times[selected] + horizon_ticks[selected] + 1
            hi = np.searchsorted(times, end, side="left")
            baseline_hazard[selected] = prefix[hi] - prefix[lo]
    else:
        steps = horizon_ticks.astype(np.int64)
        ratio = math.exp(-beta)
        kernel_hazard = state_score * ratio * (1.0 - np.power(ratio, steps)) / (
            1.0 - ratio
        )
        baseline_hazard = np.zeros(len(landmarks.outcomes), dtype=np.float64)
        local = context.entity_lookup[landmarks.entity_codes]
        for step in range(1, int(config.warning_horizon) + 1):
            valid = step <= steps
            rows = (
                context.offsets[local[valid]]
                + landmarks.times[valid]
                + step
                - context.starts[local[valid]]
            )
            groups = context.temporal_baseline_groups_at_rows(
                rows, time_bins=config.baseline_time_bins
            )
            baseline_hazard[valid] += baseline[groups]
    probability = -np.expm1(-np.maximum(0.0, baseline_hazard + kernel_hazard))
    metrics = classification_metrics(landmarks.outcomes, probability)
    payload = {
        "schema": "crbstpp.hawkes-landmark-metrics.v1",
        "dataset_digest": result["dataset_digest"],
        "readout": "integrated_exponential_hawkes_with_frozen_history",
        "future_predicates_used": False,
        "test": metrics,
    }
    (run_dir / "hawkes_landmark_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
