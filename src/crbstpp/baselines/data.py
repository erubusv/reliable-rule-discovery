from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import Dataset
from .config import BaselineConfig


PREPARED_SCHEMA = "crbstpp.baselines.prepared.v2"


@dataclass(frozen=True)
class LandmarkSplit:
    features: np.ndarray
    outcomes: np.ndarray
    entity_codes: np.ndarray
    times: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.outcomes)
        if (
            self.features.ndim != 2
            or self.features.shape[0] != n
            or self.entity_codes.shape != (n,)
            or self.times.shape != (n,)
        ):
            raise ValueError("landmark arrays are not aligned")


@dataclass(frozen=True)
class PreparedBaselines:
    root: Path
    manifest: dict[str, object]

    @property
    def landmarks_path(self) -> Path:
        return self.root / "landmarks.npz"

    @property
    def easytpp_path(self) -> Path:
        return self.root / "easytpp.pkl"


def _digest_payload(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prepared_root(config: BaselineConfig, dataset: Dataset) -> Path:
    contract = {
        "schema": PREPARED_SCHEMA,
        "dataset_digest": dataset.digest,
        "history_horizon": int(config.history_horizon),
        "warning_horizon": int(config.warning_horizon),
        "max_sequence_length": int(config.max_sequence_length),
        "sequence_context_length": int(config.sequence_context_length),
    }
    return config.run_root / "prepared" / f"{config.dataset_id}-{_digest_payload(contract)[:16]}"


def _event_bounds(dataset: Dataset) -> np.ndarray:
    counts = np.bincount(dataset.event_entities, minlength=dataset.n_entities)
    offsets = np.zeros(dataset.n_entities + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def _target_bounds(dataset: Dataset) -> np.ndarray:
    counts = np.bincount(dataset.target_entities, minlength=dataset.n_entities)
    offsets = np.zeros(dataset.n_entities + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def build_landmarks(dataset: Dataset, config: BaselineConfig) -> dict[int, LandmarkSplit]:
    """Build identical rolling-count prediction rows for Logistic and XGBoost.

    A row is created at the entity start and immediately after every distinct
    predicate-event time.  Features use only predicates at or before that
    time.  Its label is one exactly when a target lies in ``(t, t+h]``.  This
    strict interval prevents a source and target with the same timestamp from
    leaking the outcome into its own predictor.
    """

    if dataset.partitions is None:
        raise ValueError("baseline experiments require a frozen dataset partition")
    event_offsets = _event_bounds(dataset)
    target_offsets = _target_bounds(dataset)
    history = int(config.history_horizon) * int(dataset.ticks_per_unit)
    warning = int(config.warning_horizon) * int(dataset.ticks_per_unit)
    split_rows: dict[int, list[tuple[np.ndarray, int, int, int]]] = {
        0: [],
        1: [],
        2: [],
    }
    dimension = dataset.n_reported_predicates
    for entity in range(dataset.n_entities):
        split = int(dataset.partitions[entity])
        e0, e1 = int(event_offsets[entity]), int(event_offsets[entity + 1])
        event_times = dataset.event_times[e0:e1]
        event_predicates = dataset.event_predicates[e0:e1]
        reported = event_predicates < dimension
        event_times = event_times[reported]
        event_predicates = event_predicates[reported]
        query_times = np.unique(
            np.r_[np.int64(dataset.start_times[entity]), event_times]
        )
        query_times = query_times[query_times <= dataset.end_times[entity]]
        t0, t1 = int(target_offsets[entity]), int(target_offsets[entity + 1])
        targets = dataset.target_times[t0:t1]
        left = 0
        right = 0
        counts = np.zeros(dimension, dtype=np.float32)
        for query in query_times:
            while right < len(event_times) and event_times[right] <= query:
                counts[int(event_predicates[right])] += 1.0
                right += 1
            cutoff = int(query) - history
            while left < right and event_times[left] < cutoff:
                counts[int(event_predicates[left])] -= 1.0
                left += 1
            lo = int(np.searchsorted(targets, query, side="right"))
            hi = int(np.searchsorted(targets, int(query) + warning, side="right"))
            outcome = int(hi > lo)
            split_rows[split].append((counts.copy(), outcome, entity, int(query)))
    output: dict[int, LandmarkSplit] = {}
    for split, rows in split_rows.items():
        if not rows:
            raise ValueError(f"partition {split} has no landmark rows")
        output[split] = LandmarkSplit(
            features=np.vstack([row[0] for row in rows]).astype(np.float32),
            outcomes=np.asarray([row[1] for row in rows], dtype=np.int8),
            entity_codes=np.asarray([row[2] for row in rows], dtype=np.int32),
            times=np.asarray([row[3] for row in rows], dtype=np.int64),
        )
    return output


def _segment_sequence(
    sequence: list[dict[str, float | int]],
    *,
    maximum: int,
    context: int,
) -> list[list[dict[str, float | int]]]:
    if len(sequence) <= maximum:
        return [sequence]
    stride = maximum - context
    output = []
    for left in range(0, len(sequence), stride):
        right = min(len(sequence), left + maximum)
        segment = sequence[left:right]
        if len(segment) >= 2:
            origin = float(segment[0]["time_since_start"])
            shifted = []
            previous = 0.0
            for item in segment:
                current = float(item["time_since_start"]) - origin
                shifted.append(
                    {
                        "time_since_start": current,
                        "time_since_last_event": max(0.0, current - previous),
                        "type_event": int(item["type_event"]),
                    }
                )
                previous = current
            output.append(shifted)
        if right == len(sequence):
            break
    return output


def build_easytpp_payload(dataset: Dataset, config: BaselineConfig) -> dict[str, object]:
    """Convert the frozen partitions to EasyTPP's official Gatech format.

    A target terminates a sequence and is never carried into the encoder
    history of a later target.  This keeps the neural baselines on the same
    target-blind source history used by the other comparison models.  Sources
    at the target timestamp start the following sequence, so they cannot
    explain a simultaneous target.
    """

    if dataset.partitions is None:
        raise ValueError("EasyTPP conversion requires frozen partitions")
    event_offsets = _event_bounds(dataset)
    target_offsets = _target_bounds(dataset)
    target_type = dataset.n_reported_predicates
    split_names = {0: "train", 1: "dev", 2: "test"}
    output: dict[str, object] = {"dim_process": target_type + 1}
    sequences: dict[str, list[list[dict[str, float | int]]]] = {
        name: [] for name in split_names.values()
    }
    scale = float(dataset.ticks_per_unit)
    for entity in range(dataset.n_entities):
        start = int(dataset.start_times[entity])
        end = int(dataset.end_times[entity])
        e0, e1 = int(event_offsets[entity]), int(event_offsets[entity + 1])
        source = [
            (int(time), 1, int(predicate))
            for time, predicate in zip(
                dataset.event_times[e0:e1],
                dataset.event_predicates[e0:e1],
                strict=True,
            )
            if predicate < target_type and start <= time <= end
        ]
        t0, t1 = int(target_offsets[entity]), int(target_offsets[entity + 1])
        # Target sorts before a source at the same timestamp.  A simultaneous
        # predicate can therefore never explain the target at that time.
        target = [
            (int(time), 0, target_type)
            for time in dataset.target_times[t0:t1]
            if start <= time <= end
        ]
        merged = sorted((*source, *target))
        split_name = split_names[int(dataset.partitions[entity])]
        segment_start = start
        segment: list[tuple[int, int]] = []

        def flush() -> None:
            nonlocal segment
            if len(segment) < 2:
                segment = []
                return
            sequence: list[dict[str, float | int]] = []
            previous = float(segment_start)
            for event_time, event_type in segment:
                current = (float(event_time) - float(segment_start)) / scale
                delta = max(0.0, (float(event_time) - previous) / scale)
                sequence.append(
                    {
                        "time_since_start": current,
                        "time_since_last_event": delta,
                        "type_event": event_type,
                    }
                )
                previous = float(event_time)
            sequences[split_name].extend(
                _segment_sequence(
                    sequence,
                    maximum=config.max_sequence_length,
                    context=config.sequence_context_length,
                )
            )
            segment = []

        for time, is_source, event_type in merged:
            segment.append((time, event_type))
            if not is_source:
                flush()
                segment_start = time
        flush()
    output.update(sequences)
    return output


def _write_npz(path: Path, splits: dict[int, LandmarkSplit]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    arrays: dict[str, np.ndarray] = {}
    for split, label in ((0, "fit"), (1, "cert"), (2, "test")):
        value = splits[split]
        arrays[f"x_{label}"] = value.features
        arrays[f"y_{label}"] = value.outcomes
        arrays[f"entity_{label}"] = value.entity_codes
        arrays[f"time_{label}"] = value.times
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_landmarks(path: Path) -> dict[int, LandmarkSplit]:
    with np.load(path, allow_pickle=False) as arrays:
        return {
            split: LandmarkSplit(
                features=arrays[f"x_{label}"],
                outcomes=arrays[f"y_{label}"],
                entity_codes=arrays[f"entity_{label}"],
                times=arrays[f"time_{label}"],
            )
            for split, label in ((0, "fit"), (1, "cert"), (2, "test"))
        }


def prepare_baseline_data(config: BaselineConfig) -> PreparedBaselines:
    dataset = Dataset.load(config.dataset)
    root = prepared_root(config, dataset)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") == PREPARED_SCHEMA
            and manifest.get("dataset_digest") == dataset.digest
            and (root / "landmarks.npz").is_file()
            and (root / "easytpp.pkl").is_file()
        ):
            return PreparedBaselines(root, manifest)
        raise ValueError("prepared baseline cache exists with a different contract")
    root.mkdir(parents=True, exist_ok=False)
    landmarks = build_landmarks(dataset, config)
    _write_npz(root / "landmarks.npz", landmarks)
    payload = build_easytpp_payload(dataset, config)
    temporary_pickle = root / "easytpp.tmp.pkl"
    with temporary_pickle.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_pickle, root / "easytpp.pkl")
    manifest = {
        "schema": PREPARED_SCHEMA,
        "dataset": str(config.dataset),
        "dataset_id": config.dataset_id,
        "dataset_digest": dataset.digest,
        "history_horizon": config.history_horizon,
        "warning_horizon": config.warning_horizon,
        "landmark_rows": {
            label: int(len(landmarks[split].outcomes))
            for split, label in ((0, "fit"), (1, "cert"), (2, "test"))
        },
        "landmark_targets": {
            label: int(np.sum(landmarks[split].outcomes))
            for split, label in ((0, "fit"), (1, "cert"), (2, "test"))
        },
        "easytpp_sequences": {
            name: len(payload[name]) for name in ("train", "dev", "test")
        },
        "easytpp_target_type": dataset.n_reported_predicates,
        "same_time_contract": "target_before_source",
        "source_history_contract": (
            "reported target-blind predicates; target terminates each sequence"
        ),
    }
    temporary_manifest = root / "manifest.tmp.json"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return PreparedBaselines(root, manifest)
