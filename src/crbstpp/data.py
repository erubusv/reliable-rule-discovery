from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATASET_SCHEMA = "crbstpp.dataset.v1"
SUPPORTED_LIKELIHOODS = frozenset({"poisson", "first_event_cloglog"})
REQUIRED_F0 = (
    "dynamic_predicates",
    "outcome_blind_predicate_construction",
    "direct_target_proxy_excluded",
    "strict_future_effect_required",
    "atomic_predicates",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Dataset:
    """Immutable sparse event-sequence dataset.

    Predicate events are unique by ``(entity, time, predicate)``.  Target
    multiplicity is preserved separately, which is necessary for recurrent
    counting processes.
    """

    root: Path
    entity_ids: np.ndarray
    start_times: np.ndarray
    end_times: np.ndarray
    baseline_origins: np.ndarray
    split_groups: np.ndarray
    predicate_names: tuple[str, ...]
    event_entities: np.ndarray
    event_times: np.ndarray
    event_predicates: np.ndarray
    target_entities: np.ndarray
    target_times: np.ndarray
    target_multiplicity: np.ndarray
    likelihood: str
    time_unit: str
    adverse_event_name: str
    f0_contract: dict[str, object]
    digest: str

    @property
    def n_entities(self) -> int:
        return int(self.entity_ids.size)

    @property
    def n_predicates(self) -> int:
        return len(self.predicate_names)

    @classmethod
    def load(cls, root: str | Path) -> "Dataset":
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing CRBS dataset manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != DATASET_SCHEMA:
            raise ValueError("legacy or unsupported dataset schema")
        likelihood = str(manifest.get("likelihood", ""))
        if likelihood not in SUPPORTED_LIKELIHOODS:
            raise ValueError(f"unsupported likelihood: {likelihood}")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != {"entities", "events", "targets"}:
            raise ValueError("manifest must name entities/events/targets files")
        for name, metadata in files.items():
            if not isinstance(metadata, dict) or "path" not in metadata or "sha256" not in metadata:
                raise ValueError(f"invalid file manifest entry: {name}")
            path = root / str(metadata["path"])
            if not path.is_file() or _sha256_file(path) != metadata["sha256"]:
                raise ValueError(f"dataset file digest mismatch: {path}")
        entities = pd.read_parquet(root / files["entities"]["path"])
        events = pd.read_parquet(root / files["events"]["path"])
        targets = pd.read_parquet(root / files["targets"]["path"])
        expected_entities = [
            "entity_id", "start_time", "end_time", "baseline_origin", "split_group"
        ]
        expected_events = ["entity_code", "time", "predicate_code"]
        expected_targets = ["entity_code", "time", "multiplicity"]
        if entities.columns.tolist() != expected_entities:
            raise ValueError("invalid entities schema")
        if events.columns.tolist() != expected_events:
            raise ValueError("invalid events schema")
        if targets.columns.tolist() != expected_targets:
            raise ValueError("invalid targets schema")
        predicate_names = tuple(str(value) for value in manifest.get("predicate_names", ()))
        if not predicate_names or len(set(predicate_names)) != len(predicate_names):
            raise ValueError("predicate_names must be nonempty and unique")
        result = cls(
            root=root,
            entity_ids=entities["entity_id"].astype(str).to_numpy(),
            start_times=entities["start_time"].to_numpy(dtype=np.int64),
            end_times=entities["end_time"].to_numpy(dtype=np.int64),
            baseline_origins=entities["baseline_origin"].to_numpy(dtype=np.int64),
            split_groups=entities["split_group"].to_numpy(dtype=np.int64),
            predicate_names=predicate_names,
            event_entities=events["entity_code"].to_numpy(dtype=np.int32),
            event_times=events["time"].to_numpy(dtype=np.int64),
            event_predicates=events["predicate_code"].to_numpy(dtype=np.int16),
            target_entities=targets["entity_code"].to_numpy(dtype=np.int32),
            target_times=targets["time"].to_numpy(dtype=np.int64),
            target_multiplicity=targets["multiplicity"].to_numpy(dtype=np.int32),
            likelihood=likelihood,
            time_unit=str(manifest["time_unit"]),
            adverse_event_name=str(manifest["adverse_event_name"]),
            f0_contract=dict(manifest.get("f0_contract", {})),
            digest=str(manifest["dataset_digest"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        n = self.n_entities
        if (
            n == 0
            or self.start_times.shape != (n,)
            or self.end_times.shape != (n,)
            or self.baseline_origins.shape != (n,)
        ):
            raise ValueError("invalid entity arrays")
        if self.split_groups.shape != (n,) or np.any(self.end_times < self.start_times):
            raise ValueError("invalid observation bounds")
        if len(set(self.entity_ids.tolist())) != n:
            raise ValueError("entity IDs must be unique")
        if not self.adverse_event_name:
            raise ValueError("adverse_event_name must be pre-registered")
        if not all(self.f0_contract.get(name) is True for name in REQUIRED_F0):
            raise ValueError("dataset does not satisfy the F0 preprocessing contract")
        for entities, times, label in (
            (self.event_entities, self.event_times, "predicate"),
            (self.target_entities, self.target_times, "target"),
        ):
            if entities.shape != times.shape or np.any(entities < 0) or np.any(entities >= n):
                raise ValueError(f"invalid {label} arrays")
            if len(times):
                if np.any(times < self.start_times[entities]) or np.any(times > self.end_times[entities]):
                    raise ValueError(f"{label} outside observation bounds")
        if self.event_predicates.shape != self.event_times.shape:
            raise ValueError("invalid predicate code array")
        if np.any(self.event_predicates < 0) or np.any(self.event_predicates >= self.n_predicates):
            raise ValueError("predicate code out of range")
        if self.target_multiplicity.shape != self.target_times.shape or np.any(self.target_multiplicity < 1):
            raise ValueError("invalid target multiplicity")
        if len(self.event_times) > 1:
            previous_entity, entity = self.event_entities[:-1], self.event_entities[1:]
            previous_time, time = self.event_times[:-1], self.event_times[1:]
            previous_predicate, predicate = self.event_predicates[:-1], self.event_predicates[1:]
            bad = (entity < previous_entity) | (
                (entity == previous_entity)
                & (
                    (time < previous_time)
                    | ((time == previous_time) & (predicate <= previous_predicate))
                )
            )
            if np.any(bad):
                raise ValueError("predicate events must be strictly sorted and unique")
        if len(self.target_times) > 1:
            previous_entity, entity = self.target_entities[:-1], self.target_entities[1:]
            previous_time, time = self.target_times[:-1], self.target_times[1:]
            if np.any(
                (entity < previous_entity)
                | ((entity == previous_entity) & (time <= previous_time))
            ):
                raise ValueError("target events must be strictly sorted and grouped")
        if self.likelihood == "first_event_cloglog":
            counts = np.bincount(self.target_entities, minlength=n)
            if np.any(counts > 1) or np.any(self.target_multiplicity != 1):
                raise ValueError("first-event cloglog requires at most one target per entity")

    def predicate_stream(self, predicate: int, entities: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        mask = self.event_predicates == int(predicate)
        stream_entities = self.event_entities[mask]
        stream_times = self.event_times[mask]
        if entities is None:
            return stream_entities, stream_times
        selected = np.zeros(self.n_entities, dtype=bool)
        selected[np.asarray(entities, dtype=np.int64)] = True
        keep = selected[stream_entities]
        return stream_entities[keep], stream_times[keep]

    def split(self, fractions: tuple[float, float, float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split complete entities; ordered groups are used when nonconstant."""
        if len(np.unique(self.split_groups)) > 1:
            groups, counts = np.unique(self.split_groups, return_counts=True)
            cumulative = np.cumsum(counts)
            first_target = fractions[0] * self.n_entities
            second_target = (fractions[0] + fractions[1]) * self.n_entities
            first = int(np.argmin(np.abs(cumulative - first_target)))
            second = int(np.argmin(np.abs(cumulative - second_target)))
            if second <= first:
                second = min(len(groups) - 1, first + 1)
            return tuple(
                np.flatnonzero(mask).astype(np.int32)
                for mask in (
                    self.split_groups <= groups[first],
                    (self.split_groups > groups[first]) & (self.split_groups <= groups[second]),
                    self.split_groups > groups[second],
                )
            )  # type: ignore[return-value]
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(self.n_entities)
        first = int(round(fractions[0] * self.n_entities))
        second = int(round((fractions[0] + fractions[1]) * self.n_entities))
        return tuple(np.sort(part).astype(np.int32) for part in np.split(order, [first, second]))  # type: ignore[return-value]


def write_dataset(
    root: str | Path,
    *,
    entities: pd.DataFrame,
    events: pd.DataFrame,
    targets: pd.DataFrame,
    predicate_names: Iterable[str],
    likelihood: str,
    time_unit: str,
    adverse_event_name: str,
    f0_contract: dict[str, object],
    provenance: dict[str, object],
) -> Path:
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    entities = entities[
        ["entity_id", "start_time", "end_time", "baseline_origin", "split_group"]
    ].copy()
    events = events[["entity_code", "time", "predicate_code"]].copy()
    targets = targets[["entity_code", "time", "multiplicity"]].copy()
    entities.to_parquet(root / "entities.parquet", index=False)
    events.to_parquet(root / "events.parquet", index=False)
    targets.to_parquet(root / "targets.parquet", index=False)
    files = {}
    for name in ("entities", "events", "targets"):
        path = root / f"{name}.parquet"
        files[name] = {"path": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
    identity = {
        "schema": DATASET_SCHEMA,
        "predicate_names": list(predicate_names),
        "likelihood": likelihood,
        "time_unit": time_unit,
        "adverse_event_name": adverse_event_name,
        "f0_contract": f0_contract,
        "files": files,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {**identity, "dataset_digest": digest, "provenance": provenance}
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    Dataset.load(root)
    return root
