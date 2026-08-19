from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from crbstpp.baselines.config import BaselineConfig
from crbstpp.baselines.data import LandmarkSplit, load_landmarks
from crbstpp.baselines.metrics import classification_metrics
from crbstpp.baselines.runner import prepare_baselines
from crbstpp.data import Dataset


MODEL_IDS = {
    "rmtpp": "RMTPP",
    "nhp": "NHP",
    "thp": "THP",
    "attnhp": "AttNHP",
}


@dataclass
class Chunk:
    times: np.ndarray
    types: np.ndarray
    score: np.ndarray
    boundary: float


def _offsets(values: np.ndarray, size: int) -> np.ndarray:
    return np.r_[0, np.cumsum(np.bincount(values, minlength=size), dtype=np.int64)]


def _load_runner(result: dict[str, object], *, device: str):
    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner

    details = dict(result["details"])
    source = Path(str(details["cert_trials"][0]["config"]))
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    experiment = f"{details['model_id']}_train"
    payload[experiment]["trainer_config"]["gpu"] = (
        -1 if device == "cpu" else int(device.split(":", 1)[1])
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        handle.flush()
        config = Config.build_from_yaml_file(handle.name, experiment_id=experiment)
    runner = Runner.build_from_config(config)
    runner._load_model(str(details["checkpoint"]))
    runner.model.eval()
    return runner


def _model_samples(
    model,
    model_name: str,
    times: torch.Tensor,
    deltas: torch.Tensor,
    types: torch.Tensor,
    sample_deltas: torch.Tensor,
    *,
    last_only: bool,
) -> torch.Tensor:
    samples = sample_deltas
    # EasyTPP 0.3.0's public NHP sampler calls ``forward`` and fails for a
    # one-event history because that method also constructs an empty list of
    # left limits.  The right state after the first event is nevertheless
    # fully defined.  Evaluate that exact recurrence directly so the first
    # valid prediction interval is not discarded.
    if model_name == "nhp" and times.shape[1] == 1:
        batch = times.shape[0]
        c_t, c_bar_t, delta_t, o_t = model.get_init_state(batch)
        c_decay, h_decay = model.rnn_cell.decay(
            c_t,
            c_bar_t,
            delta_t,
            o_t,
            deltas[:, 0, None],
        )
        embedding = model.layer_type_emb(types[:, 0])
        c_t, c_bar_t, delta_t, o_t = model.rnn_cell(
            x_i=embedding,
            hidden_ti_minus=h_decay,
            ct_ti_minus=c_decay,
            c_bar_im1=c_bar_t,
        )
        _, hidden = model.rnn_cell.decay(
            c_t[:, None, None, :],
            c_bar_t[:, None, None, :],
            delta_t[:, None, None, :],
            o_t[:, None, None, :],
            samples[:, -1:, :, None],
        )
        return model.layer_intensity(hidden)
    if model_name == "attnhp" and samples.shape[1] == times.shape[1]:
        samples = times[:, :, None] + samples
    return model.compute_intensities_at_sample_times(
        times,
        deltas,
        types,
        samples,
        compute_last_step_only=last_only,
    )


def _scaled_history(
    event_times: np.ndarray,
    event_types: np.ndarray,
    *,
    ticks_per_unit: int,
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    scale = float(ticks_per_unit) * float(time_scale)
    times = (event_times.astype(np.float64) - float(event_times[0])) / scale
    return times, event_types.astype(np.int64, copy=False)


def _chunks(
    dataset: Dataset,
    *,
    split: int,
    maximum: int,
    context: int,
    time_scale: float,
) -> tuple[dict[int, list[Chunk]], float, int]:
    event_offsets = _offsets(dataset.event_entities, dataset.n_entities)
    target_offsets = _offsets(dataset.target_entities, dataset.n_entities)
    target_type = dataset.n_reported_predicates
    queues: dict[int, list[Chunk]] = defaultdict(list)
    empty_exposure = 0.0
    empty_targets = 0
    stride = maximum - context
    scale = float(dataset.ticks_per_unit)
    for entity in np.flatnonzero(dataset.partitions == split):
        e0, e1 = int(event_offsets[entity]), int(event_offsets[entity + 1])
        keep = dataset.event_predicates[e0:e1] < target_type
        source_times = dataset.event_times[e0:e1][keep]
        source_types = dataset.event_predicates[e0:e1][keep]
        t0, t1 = int(target_offsets[entity]), int(target_offsets[entity + 1])
        target_times = dataset.target_times[t0:t1]
        target_counts = dataset.target_multiplicity[t0:t1]
        merged = sorted(
            [(int(t), 1, int(p), 1) for t, p in zip(source_times, source_types)]
            + [(int(t), 0, target_type, int(c)) for t, c in zip(target_times, target_counts)]
        )
        segment_start = int(dataset.start_times[entity])
        segment: list[tuple[int, int, int]] = []

        def flush(end: int, target_end: bool) -> None:
            nonlocal segment, empty_exposure, empty_targets
            if not segment:
                empty_exposure += max(0.0, (end - segment_start) / scale)
                segment = []
                return
            first_time = segment[0][0]
            empty_exposure += max(0.0, (first_time - segment_start) / scale)
            if segment[0][1] == target_type:
                empty_targets += int(segment[0][2])
                segment = []
                return
            raw_times = np.asarray([item[0] for item in segment], dtype=np.int64)
            raw_types = np.asarray([item[1] for item in segment], dtype=np.int64)
            times, types = _scaled_history(
                raw_times,
                raw_types,
                ticks_per_unit=dataset.ticks_per_unit,
                time_scale=time_scale,
            )
            scored_right = 1
            for left in range(0, len(segment), stride):
                right = min(len(segment), left + maximum)
                if right - left < 1:
                    break
                local_times = times[left:right] - times[left]
                local_types = types[left:right]
                score = np.zeros(max(0, right - left - 1), dtype=bool)
                begin = max(scored_right, left + 1)
                if begin < right:
                    score[begin - left - 1 : right - left - 1] = True
                boundary = 0.0
                if right == len(segment) and not target_end:
                    boundary = max(
                        0.0,
                        (float(end) - float(raw_times[-1]))
                        / (float(dataset.ticks_per_unit) * time_scale),
                    )
                queues[len(local_times)].append(
                    Chunk(local_times, local_types, score, boundary)
                )
                scored_right = max(scored_right, right)
                if right == len(segment):
                    break
            segment = []

        for time, is_source, event_type, count in merged:
            segment.append((time, event_type, count))
            if not is_source:
                flush(time, True)
                segment_start = time
        flush(int(dataset.end_times[entity]), False)
    return queues, empty_exposure, empty_targets


def _target_process_metrics(
    runner,
    model_name: str,
    dataset: Dataset,
    config: BaselineConfig,
    *,
    split: int,
    null_rate: float,
    quadrature: int,
) -> dict[str, float | int]:
    time_scale = float(runner._data_loader.time_scale)
    queues, empty_exposure, empty_targets = _chunks(
        dataset,
        split=split,
        maximum=int(config.max_sequence_length),
        context=int(config.sequence_context_length),
        time_scale=time_scale,
    )
    model = runner.model
    device = model.device
    target_type = dataset.n_reported_predicates
    total_hazard = null_rate * empty_exposure
    continuous_event_log = empty_targets * math.log(max(null_rate, 1.0e-12))
    target_count = int(empty_targets)
    fractions = (np.arange(quadrature, dtype=np.float32) + 0.5) / quadrature
    batch_size = max(1, min(int(config.batch_size), 128))
    with torch.no_grad():
        for length, chunks in sorted(queues.items()):
            for left in range(0, len(chunks), batch_size):
                batch = chunks[left : left + batch_size]
                times = torch.as_tensor(
                    np.stack([item.times for item in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                types = torch.as_tensor(
                    np.stack([item.types for item in batch]),
                    dtype=torch.long,
                    device=device,
                )
                deltas = torch.zeros_like(times)
                if length > 1:
                    deltas[:, 1:] = times[:, 1:] - times[:, :-1]
                    interval = deltas[:, 1:]
                    sample = interval[:, :, None] * torch.as_tensor(
                        fractions, device=device
                    )[None, None, :]
                    endpoint = interval[:, :, None]
                    query = torch.cat((sample, endpoint), dim=-1)
                    intensity = _model_samples(
                        model,
                        model_name,
                        times[:, :-1],
                        deltas[:, :-1],
                        types[:, :-1],
                        query,
                        last_only=False,
                    )[..., target_type]
                    masks = torch.as_tensor(
                        np.stack([item.score for item in batch]),
                        dtype=torch.bool,
                        device=device,
                    )
                    hazard = intensity[..., :quadrature].mean(-1) * interval
                    total_hazard += float(hazard[masks].sum().cpu())
                    if dataset.likelihood == "continuous_poisson":
                        next_types = types[:, 1:]
                        target = masks & (next_types == target_type)
                        if torch.any(target):
                            at_event = intensity[..., -1]
                            continuous_event_log += float(
                                torch.log(torch.clamp(at_event[target], min=1.0e-12)).sum().cpu()
                            ) - int(target.sum()) * math.log(time_scale)
                            target_count += int(target.sum())
                boundary = torch.as_tensor(
                    [item.boundary for item in batch],
                    dtype=torch.float32,
                    device=device,
                )
                active = boundary > 0
                if torch.any(active):
                    sample = boundary[:, None, None] * torch.as_tensor(
                        fractions, device=device
                    )[None, None, :]
                    intensity = _model_samples(
                        model,
                        model_name,
                        times,
                        deltas,
                        types,
                        sample,
                        last_only=True,
                    )[:, -1, :, target_type]
                    total_hazard += float(
                        (intensity.mean(-1)[active] * boundary[active]).sum().cpu()
                    )
    if dataset.likelihood == "continuous_poisson":
        nll = total_hazard - continuous_event_log
    else:
        target_hazard = _target_interval_hazards(
            runner,
            model_name,
            dataset,
            config,
            split=split,
            null_rate=null_rate,
            quadrature=quadrature,
        )
        target_count = len(target_hazard)
        nll = (
            total_hazard
            - float(np.sum(target_hazard))
            - float(np.sum(np.log(-np.expm1(-np.maximum(target_hazard, 1.0e-12)))))
        )
    test_entities = int(np.count_nonzero(dataset.partitions == split))
    return {
        "target_nll_total": float(nll),
        "target_nll_per_entity": float(nll / test_entities),
        "target_count": int(target_count),
        "unconditioned_exposure": float(empty_exposure),
        "base_total_hazard": float(total_hazard),
        "base_continuous_event_log": float(continuous_event_log),
        "base_target_interval_hazards": (
            [] if dataset.likelihood == "continuous_poisson" else target_hazard.tolist()
        ),
    }


def _history_batches(
    dataset: Dataset,
    rows: LandmarkSplit,
    *,
    maximum: int,
    time_scale: float,
    horizon_units: float,
) -> tuple[dict[int, list[tuple[int, np.ndarray, np.ndarray, float]]], list[int]]:
    event_offsets = _offsets(dataset.event_entities, dataset.n_entities)
    target_offsets = _offsets(dataset.target_entities, dataset.n_entities)
    target_type = dataset.n_reported_predicates
    queues: dict[int, list[tuple[int, np.ndarray, np.ndarray, float]]] = defaultdict(list)
    empty: list[int] = []
    horizon_ticks = int(round(float(horizon_units) * int(dataset.ticks_per_unit)))
    for position, (entity, query) in enumerate(zip(rows.entity_codes, rows.times)):
        entity = int(entity)
        query = int(query)
        e0, e1 = int(event_offsets[entity]), int(event_offsets[entity + 1])
        predicates = dataset.event_predicates[e0:e1]
        keep = predicates < target_type
        times = dataset.event_times[e0:e1][keep]
        types = predicates[keep]
        t0, t1 = int(target_offsets[entity]), int(target_offsets[entity + 1])
        targets = dataset.target_times[t0:t1]
        prior_target = int(np.searchsorted(targets, query, side="right")) - 1
        start = (
            int(np.searchsorted(times, targets[prior_target], side="left"))
            if prior_target >= 0
            else 0
        )
        end = int(np.searchsorted(times, query, side="right"))
        if end <= start:
            empty.append(position)
            continue
        start = max(start, end - maximum)
        history_times, history_types = _scaled_history(
            times[start:end],
            types[start:end],
            ticks_per_unit=dataset.ticks_per_unit,
            time_scale=time_scale,
        )
        available = max(0, int(dataset.end_times[entity]) - query)
        horizon = min(available, horizon_ticks) / (
            float(dataset.ticks_per_unit) * time_scale
        )
        queues[len(history_times)].append((position, history_times, history_types, horizon))
    return queues, empty


def _hazards_at_rows(
    runner,
    model_name: str,
    dataset: Dataset,
    config: BaselineConfig,
    rows: LandmarkSplit,
    *,
    null_rate: float,
    quadrature: int,
    horizon_units: float,
) -> np.ndarray:
    time_scale = float(runner._data_loader.time_scale)
    queues, empty = _history_batches(
        dataset,
        rows,
        maximum=int(config.max_sequence_length),
        time_scale=time_scale,
        horizon_units=horizon_units,
    )
    hazards = np.full(len(rows.outcomes), null_rate * horizon_units, dtype=np.float64)
    model = runner.model
    device = model.device
    target_type = dataset.n_reported_predicates
    fractions = torch.as_tensor(
        (np.arange(quadrature, dtype=np.float32) + 0.5) / quadrature,
        device=device,
    )
    batch_size = max(1, min(int(config.batch_size), 128))
    with torch.no_grad():
        for length, items in sorted(queues.items()):
            for left in range(0, len(items), batch_size):
                batch = items[left : left + batch_size]
                times = torch.as_tensor(
                    np.stack([item[1] for item in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                types = torch.as_tensor(
                    np.stack([item[2] for item in batch]),
                    dtype=torch.long,
                    device=device,
                )
                deltas = torch.zeros_like(times)
                if length > 1:
                    deltas[:, 1:] = times[:, 1:] - times[:, :-1]
                horizon = torch.as_tensor(
                    [item[3] for item in batch], dtype=torch.float32, device=device
                )
                sample = horizon[:, None, None] * fractions[None, None, :]
                intensity = _model_samples(
                    model,
                    model_name,
                    times,
                    deltas,
                    types,
                    sample,
                    last_only=True,
                )[:, -1, :, target_type]
                value = (intensity.mean(-1) * horizon).cpu().numpy()
                hazards[[item[0] for item in batch]] = value
    return hazards


def _target_interval_hazards(
    runner,
    model_name: str,
    dataset: Dataset,
    config: BaselineConfig,
    *,
    split: int,
    null_rate: float,
    quadrature: int,
) -> np.ndarray:
    target_offsets = _offsets(dataset.target_entities, dataset.n_entities)
    entities: list[int] = []
    times: list[int] = []
    for entity in np.flatnonzero(dataset.partitions == split):
        t0, t1 = int(target_offsets[entity]), int(target_offsets[entity + 1])
        for target, count in zip(
            dataset.target_times[t0:t1], dataset.target_multiplicity[t0:t1]
        ):
            entities.extend([int(entity)] * int(count))
            times.extend([int(target) - int(dataset.ticks_per_unit)] * int(count))
    rows = LandmarkSplit(
        features=np.zeros((len(times), 0), dtype=np.float32),
        outcomes=np.ones(len(times), dtype=np.int8),
        entity_codes=np.asarray(entities, dtype=np.int32),
        times=np.asarray(times, dtype=np.int64),
    )
    return _hazards_at_rows(
        runner,
        model_name,
        dataset,
        config,
        rows,
        null_rate=null_rate,
        quadrature=quadrature,
        horizon_units=1.0,
    )


def _scaled_target_nll(
    statistics: dict[str, float | int | list[float]],
    likelihood: str,
    scale: float,
) -> float:
    scale = max(float(scale), 1.0e-12)
    total = scale * float(statistics["base_total_hazard"])
    count = int(statistics["target_count"])
    if likelihood == "continuous_poisson":
        return (
            total
            - float(statistics["base_continuous_event_log"])
            - count * math.log(scale)
        )
    target = scale * np.asarray(
        statistics["base_target_interval_hazards"], dtype=np.float64
    )
    return float(
        total
        - np.sum(target)
        - np.sum(np.log(-np.expm1(-np.maximum(target, 1.0e-12))))
    )


def _calibration_scale(
    statistics: dict[str, float | int | list[float]], likelihood: str
) -> float:
    if likelihood == "continuous_poisson":
        return max(
            1.0e-8,
            int(statistics["target_count"])
            / max(float(statistics["base_total_hazard"]), 1.0e-12),
        )
    from scipy.optimize import minimize_scalar

    optimum = minimize_scalar(
        lambda log_scale: _scaled_target_nll(
            statistics, likelihood, math.exp(float(log_scale))
        ),
        bounds=(-18.0, 18.0),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    if not optimum.success:
        raise RuntimeError(f"target intensity calibration failed: {optimum.message}")
    return float(math.exp(float(optimum.x)))


def _binary_calibration_scale(outcomes: np.ndarray, hazards: np.ndarray) -> float:
    """Select one positive warning-probability scale without using D_test."""
    from scipy.optimize import minimize_scalar

    labels = np.asarray(outcomes, dtype=np.float64)
    base_hazard = np.maximum(np.asarray(hazards, dtype=np.float64), 0.0)

    def objective(log_scale: float) -> float:
        probability = -np.expm1(-math.exp(float(log_scale)) * base_hazard)
        probability = np.clip(probability, 1.0e-12, 1.0 - 1.0e-12)
        return float(
            -np.mean(
                labels * np.log(probability)
                + (1.0 - labels) * np.log1p(-probability)
            )
        )

    optimum = minimize_scalar(
        objective,
        bounds=(-18.0, 18.0),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    if not optimum.success:
        raise RuntimeError(f"binary calibration failed: {optimum.message}")
    return float(math.exp(float(optimum.x)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quadrature", type=int, default=20)
    args = parser.parse_args()
    config = BaselineConfig.from_yaml(args.config)
    dataset = Dataset.load(config.dataset)
    run_dir = config.run_root / config.dataset_id / f"seed-{args.seed}" / args.model
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    prepared = prepare_baselines(config, seed=args.seed)
    landmarks = load_landmarks(prepared.landmarks_path)
    fit_probability = float(np.mean(landmarks[0].outcomes))
    warning_units = float(config.warning_horizon)
    null_rate = -math.log1p(-min(fit_probability, 1.0 - 1.0e-12)) / warning_units
    runner = _load_runner(result, device=args.device)
    cert_target = _target_process_metrics(
        runner,
        args.model,
        dataset,
        config,
        split=1,
        null_rate=null_rate,
        quadrature=args.quadrature,
    )
    calibration_scale = _calibration_scale(cert_target, dataset.likelihood)
    target = _target_process_metrics(
        runner,
        args.model,
        dataset,
        config,
        split=2,
        null_rate=null_rate,
        quadrature=args.quadrature,
    )
    target["target_nll_total"] = _scaled_target_nll(
        target, dataset.likelihood, calibration_scale
    )
    target["target_nll_per_entity"] = float(target["target_nll_total"]) / int(
        np.count_nonzero(dataset.partitions == 2)
    )
    cert_rows = landmarks[1]
    cert_hazards = _hazards_at_rows(
        runner,
        args.model,
        dataset,
        config,
        cert_rows,
        null_rate=null_rate,
        quadrature=args.quadrature,
        horizon_units=warning_units,
    )
    binary_scale = _binary_calibration_scale(cert_rows.outcomes, cert_hazards)
    test_rows = landmarks[2]
    hazards = _hazards_at_rows(
        runner,
        args.model,
        dataset,
        config,
        test_rows,
        null_rate=null_rate,
        quadrature=args.quadrature,
        horizon_units=warning_units,
    )
    probability = -np.expm1(-binary_scale * np.maximum(hazards, 0.0))
    binary = classification_metrics(test_rows.outcomes, probability)
    payload = {
        "schema": "crbstpp.easytpp-target-metrics.v1",
        "dataset_digest": dataset.digest,
        "model": args.model,
        "seed": args.seed,
        "readout": "target_type_intensity_with_frozen_landmark_history",
        "quadrature_points": args.quadrature,
        "empty_history_rate": null_rate,
        "target_intensity_scale_selected_on_cert": calibration_scale,
        "binary_hazard_scale_selected_on_cert": binary_scale,
        "cert_target_nll_per_entity": _scaled_target_nll(
            cert_target, dataset.likelihood, calibration_scale
        )
        / int(np.count_nonzero(dataset.partitions == 1)),
        "target": target,
        "test": binary,
    }
    (run_dir / "target_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
