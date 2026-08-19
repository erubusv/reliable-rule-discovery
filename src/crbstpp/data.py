from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATASET_SCHEMA = "crbstpp.dataset.v7"
LEGACY_DATASET_SCHEMAS = frozenset(
    {
        "crbstpp.dataset.v2",
        "crbstpp.dataset.v3",
        "crbstpp.dataset.v4",
        "crbstpp.dataset.v5",
        "crbstpp.dataset.v6",
    }
)
SUPPORTED_LIKELIHOODS = frozenset(
    {"poisson", "continuous_poisson", "first_event_cloglog"}
)
REQUIRED_F0 = (
    "dynamic_predicates",
    "outcome_blind_predicate_construction",
    "direct_target_proxy_excluded_from_reported_dictionary",
    "strict_future_effect_required",
    "atomic_predicates",
    "primitive_event_provenance",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_identity(manifest: dict[str, object]) -> dict[str, object]:
    """Return exactly the provenance-bearing payload protected by the digest."""
    names = (
        "schema",
        "predicate_names",
        "predicate_roles",
        "likelihood",
        "time_unit",
        "ticks_per_unit",
        "adverse_event_name",
        "f0_contract",
        "files",
        "provenance",
    )
    identity = {name: manifest[name] for name in names}
    if str(manifest.get("schema")) in {
        "crbstpp.dataset.v5",
        "crbstpp.dataset.v6",
        DATASET_SCHEMA,
    }:
        identity["predicate_definitions"] = manifest["predicate_definitions"]
    return identity


def _identity_digest(identity: dict[str, object]) -> str:
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
    partitions: np.ndarray | None
    predicate_names: tuple[str, ...]
    predicate_roles: tuple[str, ...]
    event_entities: np.ndarray
    event_times: np.ndarray
    event_predicates: np.ndarray
    event_primitive_ids: np.ndarray
    target_entities: np.ndarray
    target_times: np.ndarray
    target_multiplicity: np.ndarray
    likelihood: str
    time_unit: str
    ticks_per_unit: int
    adverse_event_name: str
    f0_contract: dict[str, object]
    digest: str
    end_reasons: np.ndarray | None = None
    # Optional, pre-registered entity-level nuisance stratum.  Reported rules
    # share coefficients across strata, while the null and every support fit
    # receive one unrestricted baseline intercept per stratum.
    baseline_strata: np.ndarray | None = None
    # Sampling-unit identity for dependency-robust model selection.  Several
    # point-process episodes may belong to one financial entity (for example,
    # one Aave address across V2/V3 and debt episodes).  Legacy datasets fall
    # back to one cluster per entity and therefore never merge observations by
    # an inferred, dataset-specific naming convention.
    dependency_groups: np.ndarray | None = None
    # v5 distinguishes primitive event atoms from predictable history-state
    # atoms. Legacy datasets resolve every predicate to ``kind=event``.
    predicate_definitions: tuple[dict[str, object], ...] | None = None

    # Optional target-blind, time-varying structural baseline stratum. Values
    # are stored in exact entity/time grid order and replace the static entity
    # stratum at likelihood rows. Reported rule coefficients remain shared
    # across these nuisance strata.
    baseline_cell_strata: np.ndarray | None = None
    baseline_cell_exposure: np.ndarray | None = None
    baseline_cell_times: np.ndarray | None = None
    baseline_cell_entities: np.ndarray | None = None

    @property
    def n_entities(self) -> int:
        return int(self.entity_ids.size)

    @property
    def n_predicates(self) -> int:
        return len(self.predicate_names)

    @property
    def n_reported_predicates(self) -> int:
        return sum(role == "reported" for role in self.predicate_roles)

    @property
    def n_baseline_strata(self) -> int:
        if self.baseline_cell_strata is not None:
            return int(np.max(self.baseline_cell_strata, initial=0)) + 1
        if self.baseline_strata is not None:
            return int(np.max(self.baseline_strata, initial=0)) + 1
        return 1

    def predicate_definition(self, predicate: int) -> dict[str, object]:
        predicate = int(predicate)
        if not 0 <= predicate < self.n_predicates:
            raise IndexError("predicate is outside the dataset dictionary")
        if self.predicate_definitions is None:
            return {"kind": "event"}
        return self.predicate_definitions[predicate]

    def is_state_predicate(self, predicate: int) -> bool:
        return self.predicate_definition(predicate).get("kind") in {
            "history_state",
            "transition_state",
        }

    @property
    def reported_state_predicates(self) -> tuple[int, ...]:
        return tuple(
            predicate
            for predicate in range(self.n_reported_predicates)
            if self.is_state_predicate(predicate)
        )

    @property
    def reported_event_predicates(self) -> tuple[int, ...]:
        return tuple(
            predicate
            for predicate in range(self.n_reported_predicates)
            if not self.is_state_predicate(predicate)
        )

    @property
    def baseline_control_predicates(self) -> tuple[int, ...]:
        control_roles = {
            "baseline_control",
            "exposure_increase_control",
            "exposure_decrease_control",
        }
        return tuple(
            index
            for index, role in enumerate(self.predicate_roles)
            if role in control_roles
        )

    @property
    def baseline_control_signs(self) -> tuple[int, ...]:
        """Frozen directions of non-reportable predictable nuisance events."""

        signs = {
            "baseline_control": 1,
            "exposure_increase_control": 1,
            "exposure_decrease_control": -1,
        }
        return tuple(signs[self.predicate_roles[index]] for index in self.baseline_control_predicates)

    @classmethod
    def load(cls, root: str | Path) -> "Dataset":
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing CRBS dataset manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = str(manifest.get("schema", ""))
        if schema not in {DATASET_SCHEMA, *LEGACY_DATASET_SCHEMAS}:
            raise ValueError("unsupported dataset schema")
        try:
            expected_digest = _identity_digest(_dataset_identity(manifest))
        except KeyError as error:
            raise ValueError(f"incomplete dataset manifest: {error.args[0]}") from error
        if manifest.get("dataset_digest") != expected_digest:
            raise ValueError("dataset manifest or predicate provenance digest mismatch")
        likelihood = str(manifest.get("likelihood", ""))
        if likelihood not in SUPPORTED_LIKELIHOODS:
            raise ValueError(f"unsupported likelihood: {likelihood}")
        files = manifest.get("files")
        required_files = {"entities", "events", "targets"}
        allowed_files = required_files | {"baseline_cells"}
        if (
            not isinstance(files, dict)
            or not required_files.issubset(files)
            or not set(files).issubset(allowed_files)
        ):
            raise ValueError("manifest must name required dataset files")
        for name, metadata in files.items():
            if (
                not isinstance(metadata, dict)
                or "path" not in metadata
                or "sha256" not in metadata
            ):
                raise ValueError(f"invalid file manifest entry: {name}")
            path = root / str(metadata["path"])
            if not path.is_file() or _sha256_file(path) != metadata["sha256"]:
                raise ValueError(f"dataset file digest mismatch: {path}")
        entities = pd.read_parquet(root / files["entities"]["path"])
        events = pd.read_parquet(root / files["events"]["path"])
        targets = pd.read_parquet(root / files["targets"]["path"])
        baseline_cells = (
            pd.read_parquet(root / files["baseline_cells"]["path"])
            if "baseline_cells" in files
            else None
        )
        valid_baseline_columns = (
            ["entity_code", "time", "baseline_stratum"],
            ["entity_code", "time", "baseline_stratum", "exposure"],
        )
        if (
            baseline_cells is not None
            and baseline_cells.columns.tolist() not in valid_baseline_columns
        ):
            raise ValueError("invalid baseline_cells schema")
        expected_entities = [
            "entity_id",
            "start_time",
            "end_time",
            "baseline_origin",
            "split_group",
        ]
        event_core_columns = ["entity_code", "time", "predicate_code"]
        # v2/v3 originally predated primitive-event provenance, but datasets
        # regenerated by the current preprocessors can preserve that optional
        # column while retaining the legacy manifest contract.  Accept both
        # exact legacy layouts; current schemas still require provenance.
        valid_event_columns = (
            [event_core_columns, [*event_core_columns, "primitive_event_id"]]
            if schema in {"crbstpp.dataset.v2", "crbstpp.dataset.v3"}
            else [[*event_core_columns, "primitive_event_id"]]
        )
        expected_targets = ["entity_code", "time", "multiplicity"]
        optional_entity_columns = [
            name
            for name in (
                "baseline_stratum",
                "partition",
                "end_reason",
                "dependency_group",
            )
            if name in entities.columns
        ]
        if entities.columns.tolist() != [*expected_entities, *optional_entity_columns]:
            raise ValueError("invalid entities schema")
        if events.columns.tolist() not in valid_event_columns:
            raise ValueError("invalid events schema")
        if targets.columns.tolist() != expected_targets:
            raise ValueError("invalid targets schema")
        if baseline_cells is not None:
            cell_entities = baseline_cells["entity_code"].to_numpy(dtype=np.int64)
            cell_times = baseline_cells["time"].to_numpy(dtype=np.int64)
            n_entities = len(entities)
            if (
                len(cell_entities) == 0
                or np.any(cell_entities < 0)
                or np.any(cell_entities >= n_entities)
                or np.any(cell_entities[1:] < cell_entities[:-1])
            ):
                raise ValueError("baseline cells must be sorted by valid entity code")
            counts = np.bincount(cell_entities, minlength=n_entities)
            starts = entities["start_time"].to_numpy(dtype=np.int64)
            ends = entities["end_time"].to_numpy(dtype=np.int64)
            offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
            first = offsets[:-1]
            last = offsets[1:] - 1
            same_entity = cell_entities[1:] == cell_entities[:-1]
            if likelihood == "continuous_poisson":
                if "exposure" not in baseline_cells.columns:
                    raise ValueError(
                        "continuous Poisson baseline cells require interval exposure"
                    )
                if np.any(counts == 0) or not np.array_equal(
                    cell_times[first], starts
                ):
                    raise ValueError(
                        "continuous risk intervals must start at every entity bound"
                    )
                if np.any(np.diff(cell_times)[same_entity] <= 0):
                    raise ValueError(
                        "continuous risk-interval times must be strictly increasing"
                    )
                exposure = baseline_cells["exposure"].to_numpy(dtype=np.float64)
                right_ticks = cell_times + np.rint(
                    exposure * int(manifest["ticks_per_unit"])
                ).astype(np.int64)
                if np.any(
                    right_ticks[:-1][same_entity] != cell_times[1:][same_entity]
                ):
                    raise ValueError("continuous risk intervals must be contiguous")
                if not np.array_equal(right_ticks[last], ends + 1):
                    raise ValueError(
                        "continuous risk intervals must end at every entity bound"
                    )
            else:
                expected_counts = ends - starts + 1
                if not np.array_equal(counts, expected_counts):
                    raise ValueError(
                        "baseline cells must cover every risk-grid row exactly"
                    )
                if (
                    not np.array_equal(cell_times[first], starts)
                    or not np.array_equal(cell_times[last], ends)
                ):
                    raise ValueError("baseline cell bounds do not match entity bounds")
                if np.any(np.diff(cell_times)[same_entity] != 1):
                    raise ValueError(
                        "baseline cells must contain consecutive time ticks"
                    )
            strata = baseline_cells["baseline_stratum"].to_numpy(dtype=np.int64)
            if np.any(strata < 0):
                raise ValueError("baseline cell strata must be nonnegative")
            observed_strata = np.unique(strata)
            if not np.array_equal(observed_strata, np.arange(len(observed_strata))):
                raise ValueError("baseline cell strata must be contiguous from zero")
        predicate_names = tuple(
            str(value) for value in manifest.get("predicate_names", ())
        )
        predicate_roles = tuple(
            str(value) for value in manifest.get("predicate_roles", ())
        )
        if not predicate_names or len(set(predicate_names)) != len(predicate_names):
            raise ValueError("predicate_names must be nonempty and unique")
        valid_roles = {
            "reported",
            "baseline_control",
            "exposure_increase_control",
            "exposure_decrease_control",
            "state_source",
        }
        if len(predicate_roles) != len(predicate_names) or any(
            role not in valid_roles for role in predicate_roles
        ):
            raise ValueError("every predicate must have a valid role")
        result = cls(
            root=root,
            entity_ids=entities["entity_id"].astype(str).to_numpy(),
            start_times=entities["start_time"].to_numpy(dtype=np.int64),
            end_times=entities["end_time"].to_numpy(dtype=np.int64),
            baseline_origins=entities["baseline_origin"].to_numpy(dtype=np.int64),
            split_groups=entities["split_group"].to_numpy(dtype=np.int64),
            partitions=(
                entities["partition"].to_numpy(dtype=np.int8)
                if "partition" in entities.columns
                else None
            ),
            predicate_names=predicate_names,
            predicate_roles=predicate_roles,
            event_entities=events["entity_code"].to_numpy(dtype=np.int32),
            event_times=events["time"].to_numpy(dtype=np.int64),
            event_predicates=events["predicate_code"].to_numpy(dtype=np.int16),
            event_primitive_ids=(
                events["primitive_event_id"].to_numpy(dtype=np.int64)
                if "primitive_event_id" in events.columns
                else np.arange(len(events), dtype=np.int64)
            ),
            target_entities=targets["entity_code"].to_numpy(dtype=np.int32),
            target_times=targets["time"].to_numpy(dtype=np.int64),
            target_multiplicity=targets["multiplicity"].to_numpy(dtype=np.int32),
            likelihood=likelihood,
            time_unit=str(manifest["time_unit"]),
            ticks_per_unit=int(manifest["ticks_per_unit"]),
            adverse_event_name=str(manifest["adverse_event_name"]),
            f0_contract=dict(manifest.get("f0_contract", {})),
            digest=str(manifest["dataset_digest"]),
            end_reasons=(
                entities["end_reason"].astype(str).to_numpy()
                if "end_reason" in entities.columns
                else None
            ),
            baseline_strata=(
                entities["baseline_stratum"].to_numpy(dtype=np.int16)
                if "baseline_stratum" in entities.columns
                else None
            ),
            dependency_groups=(
                pd.factorize(
                    entities["dependency_group"].astype(str), sort=True
                )[0].astype(np.int32)
                if "dependency_group" in entities.columns
                else None
            ),
            predicate_definitions=(
                tuple(dict(value) for value in manifest["predicate_definitions"])
                if schema
                in {"crbstpp.dataset.v5", "crbstpp.dataset.v6", DATASET_SCHEMA}
                else None
            ),
            baseline_cell_strata=(
                baseline_cells["baseline_stratum"].to_numpy(dtype=np.int16)
                if baseline_cells is not None
                else None
            ),
            baseline_cell_exposure=(
                baseline_cells["exposure"].to_numpy(dtype=np.float64)
                if baseline_cells is not None and "exposure" in baseline_cells.columns
                else (np.ones(len(baseline_cells)) if baseline_cells is not None else None)
            ),
            baseline_cell_times=(
                baseline_cells["time"].to_numpy(dtype=np.int64)
                if baseline_cells is not None and likelihood == "continuous_poisson"
                else None
            ),
            baseline_cell_entities=(
                baseline_cells["entity_code"].to_numpy(dtype=np.int32)
                if baseline_cells is not None and likelihood == "continuous_poisson"
                else None
            ),
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
        if self.partitions is not None:
            if self.partitions.shape != (n,) or not np.all(
                np.isin(self.partitions, (0, 1, 2))
            ):
                raise ValueError(
                    "explicit partition must contain only fit/cert/test codes"
                )
            if not np.array_equal(np.unique(self.partitions), np.array([0, 1, 2])):
                raise ValueError(
                    "explicit partition must contain fit, cert and test entities"
                )
        if self.end_reasons is not None:
            if self.end_reasons.shape != (n,):
                raise ValueError("end reasons must align with entities")
            if any(not str(value).strip() for value in self.end_reasons.tolist()):
                raise ValueError("end reasons must be nonempty strings")
        if self.baseline_strata is not None:
            if self.baseline_strata.shape != (n,) or np.any(
                self.baseline_strata < 0
            ):
                raise ValueError("baseline strata must be nonnegative and align with entities")
            observed = np.unique(self.baseline_strata)
            if not np.array_equal(observed, np.arange(len(observed))):
                raise ValueError("baseline strata must be contiguous from zero")
        if self.baseline_cell_strata is not None:
            expected_cells = (
                len(self.baseline_cell_times)
                if self.likelihood == "continuous_poisson"
                and self.baseline_cell_times is not None
                else int(
                    np.sum(self.end_times - self.start_times + 1, dtype=np.int64)
                )
            )
            if self.baseline_cell_strata.shape != (expected_cells,) or np.any(
                self.baseline_cell_strata < 0
            ):
                raise ValueError("baseline cell strata must align with the risk grid")
            observed = np.unique(self.baseline_cell_strata)
            if not np.array_equal(observed, np.arange(len(observed))):
                raise ValueError("baseline cell strata must be contiguous from zero")
        if self.baseline_cell_exposure is not None:
            if (
                self.baseline_cell_strata is None
                or self.baseline_cell_exposure.shape != self.baseline_cell_strata.shape
                or np.any(~np.isfinite(self.baseline_cell_exposure))
                or np.any(self.baseline_cell_exposure < 0.0)
                or (
                    self.likelihood == "continuous_poisson"
                    and np.any(self.baseline_cell_exposure <= 0.0)
                )
                or (
                    self.likelihood != "continuous_poisson"
                    and np.any(self.baseline_cell_exposure > 1.0)
                )
            ):
                raise ValueError("invalid baseline cell exposure")
        if self.likelihood == "continuous_poisson":
            if (
                self.baseline_cell_times is None
                or self.baseline_cell_entities is None
                or self.baseline_cell_exposure is None
                or self.baseline_cell_strata is None
            ):
                raise ValueError(
                    "continuous Poisson requires explicit irregular risk intervals"
                )
            if not (
                self.baseline_cell_times.shape
                == self.baseline_cell_entities.shape
                == self.baseline_cell_exposure.shape
            ):
                raise ValueError("continuous risk-interval arrays are misaligned")
        if self.dependency_groups is not None:
            if self.dependency_groups.shape != (n,) or np.any(
                self.dependency_groups < 0
            ):
                raise ValueError(
                    "dependency groups must be nonnegative and align with entities"
                )
            # A dependency cluster is a sampling unit and may not straddle the
            # immutable fit/cert/test split.
            if self.partitions is not None:
                order = np.argsort(self.dependency_groups, kind="stable")
                groups = self.dependency_groups[order]
                partitions = self.partitions[order]
                boundaries = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
                minimum = np.minimum.reduceat(partitions, boundaries)
                maximum = np.maximum.reduceat(partitions, boundaries)
                if np.any(minimum != maximum):
                    raise ValueError(
                        "one dependency group cannot cross data partitions"
                    )
        if len(set(self.entity_ids.tolist())) != n:
            raise ValueError("entity IDs must be unique")
        if self.predicate_definitions is not None:
            if len(self.predicate_definitions) != self.n_predicates:
                raise ValueError("predicate definitions must align with names")
            for predicate, definition in enumerate(self.predicate_definitions):
                kind = str(definition.get("kind", ""))
                if kind == "event":
                    continue
                if kind not in {"history_state", "transition_state"}:
                    raise ValueError("unsupported predicate definition kind")
                if self.predicate_roles[predicate] != "reported":
                    raise ValueError("history states must be reportable predicates")
                if kind == "history_state":
                    source = definition.get("source_predicate")
                    transform = str(definition.get("transform", ""))
                    horizon = definition.get("horizon")
                    if (
                        not isinstance(source, int)
                        or not 0 <= source < self.n_reported_predicates
                        or self.predicate_definitions[source].get("kind") != "event"
                    ):
                        raise ValueError(
                            "history state requires a primitive event source"
                        )
                    if transform not in {
                        "recent",
                        "recurrent",
                        "accelerating",
                        "decelerating",
                    }:
                        raise ValueError("invalid history-state transform")
                    if (
                        isinstance(horizon, bool)
                        or not isinstance(horizon, int)
                        or horizon < 1
                    ):
                        raise ValueError("history-state horizon must be positive")
                else:
                    entry = definition.get("entry_predicate")
                    exit_ = definition.get("exit_predicate")
                    if any(
                        not isinstance(value, int)
                        or not 0 <= value < self.n_predicates
                        or self.predicate_definitions[value].get("kind") != "event"
                        or self.predicate_roles[value] != "state_source"
                        for value in (entry, exit_)
                    ):
                        raise ValueError(
                            "transition state requires hidden event entry/exit sources"
                        )
        if not self.adverse_event_name:
            raise ValueError("adverse_event_name must be pre-registered")
        if self.ticks_per_unit < 1:
            raise ValueError("ticks_per_unit must be positive")
        if not all(self.f0_contract.get(name) is True for name in REQUIRED_F0):
            raise ValueError("dataset does not satisfy the F0 preprocessing contract")
        reported = tuple(
            index
            for index, role in enumerate(self.predicate_roles)
            if role == "reported"
        )
        controls = self.baseline_control_predicates
        if not reported or reported != tuple(range(len(reported))):
            raise ValueError("reported predicates must form a nonempty leading block")
        if any(index < len(reported) for index in controls):
            raise ValueError("baseline controls must follow reported predicates")
        if any(
            role == "state_source" and index < len(reported)
            for index, role in enumerate(self.predicate_roles)
        ):
            raise ValueError("state sources must follow reported predicates")
        for entities, times, label in (
            (self.event_entities, self.event_times, "predicate"),
            (self.target_entities, self.target_times, "target"),
        ):
            if (
                entities.shape != times.shape
                or np.any(entities < 0)
                or np.any(entities >= n)
            ):
                raise ValueError(f"invalid {label} arrays")
            if len(times):
                if np.any(times < self.start_times[entities]) or np.any(
                    times > self.end_times[entities]
                ):
                    raise ValueError(f"{label} outside observation bounds")
        if self.event_predicates.shape != self.event_times.shape:
            raise ValueError("invalid predicate code array")
        if self.event_primitive_ids.shape != self.event_times.shape:
            raise ValueError("invalid primitive event ID array")
        if np.any(self.event_predicates < 0) or np.any(
            self.event_predicates >= self.n_predicates
        ):
            raise ValueError("predicate code out of range")
        if self.target_multiplicity.shape != self.target_times.shape or np.any(
            self.target_multiplicity < 1
        ):
            raise ValueError("invalid target multiplicity")
        if len(self.event_times) > 1:
            previous_entity, entity = self.event_entities[:-1], self.event_entities[1:]
            previous_time, time = self.event_times[:-1], self.event_times[1:]
            previous_predicate, predicate = (
                self.event_predicates[:-1],
                self.event_predicates[1:],
            )
            previous_primitive, primitive = (
                self.event_primitive_ids[:-1],
                self.event_primitive_ids[1:],
            )
            bad = (entity < previous_entity) | (
                (entity == previous_entity)
                & (
                    (time < previous_time)
                    | (
                        (time == previous_time)
                        & (
                            (predicate < previous_predicate)
                            | (
                                (predicate == previous_predicate)
                                & (primitive <= previous_primitive)
                            )
                        )
                    )
                )
            )
            if np.any(bad):
                raise ValueError("predicate events must be strictly sorted and unique")
            # One raw transaction/state update may emit several predicate
            # attributes, but it cannot occupy two ticks for one entity.  The
            # ordered/unordered motif scanner relies on this provenance
            # invariant when it prevents one primitive from witnessing a
            # high-order rule twice.
            primitive_order = np.lexsort(
                (self.event_times, self.event_primitive_ids, self.event_entities)
            )
            primitive_entity = self.event_entities[primitive_order]
            primitive_id = self.event_primitive_ids[primitive_order]
            primitive_time = self.event_times[primitive_order]
            same_primitive = (
                (primitive_entity[1:] == primitive_entity[:-1])
                & (primitive_id[1:] == primitive_id[:-1])
            )
            if np.any(
                same_primitive
                & (primitive_time[1:] != primitive_time[:-1])
            ):
                raise ValueError(
                    "one primitive event ID cannot span ticks within an entity"
                )
        if len(self.target_times) > 1:
            previous_entity, entity = (
                self.target_entities[:-1],
                self.target_entities[1:],
            )
            previous_time, time = self.target_times[:-1], self.target_times[1:]
            if np.any(
                (entity < previous_entity)
                | ((entity == previous_entity) & (time <= previous_time))
            ):
                raise ValueError("target events must be strictly sorted and grouped")
        if (
            self.baseline_cell_exposure is not None
            and len(self.target_times)
            and self.likelihood != "continuous_poisson"
        ):
            lengths = self.end_times - self.start_times + 1
            offsets = np.zeros(n + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(lengths, dtype=np.int64)
            target_rows = (
                offsets[self.target_entities]
                + self.target_times
                - self.start_times[self.target_entities]
            )
            if np.any(self.baseline_cell_exposure[target_rows] <= 0.0):
                raise ValueError("target event lies outside observation opportunity")
        if self.likelihood == "first_event_cloglog":
            counts = np.bincount(self.target_entities, minlength=n)
            recurrent = bool(self.f0_contract.get("recurrent_target_process", False))
            if (
                not recurrent
                and (np.any(counts > 1) or np.any(self.target_multiplicity != 1))
            ):
                raise ValueError(
                    "first-event cloglog requires at most one target per entity"
                )
            if recurrent and np.any(self.target_multiplicity != 1):
                raise ValueError("recurrent cloglog requires binary target rows")

    def predicate_stream(
        self, predicate: int, entities: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = self.event_predicates == int(predicate)
        stream_entities = self.event_entities[mask]
        stream_times = self.event_times[mask]
        if entities is None:
            return stream_entities, stream_times
        selected = np.zeros(self.n_entities, dtype=bool)
        selected[np.asarray(entities, dtype=np.int64)] = True
        keep = selected[stream_entities]
        return stream_entities[keep], stream_times[keep]

    def predicate_stream_with_ids(
        self, predicate: int, entities: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one predicate stream together with raw-event provenance.

        ``primitive_event_id`` identifies the originating transaction or
        account-state record.  Different predicate attributes emitted by the
        same raw event therefore share an ID, while genuinely distinct events
        remain distinguishable even when coarse timestamps coincide.
        """

        mask = self.event_predicates == int(predicate)
        stream_entities = self.event_entities[mask]
        stream_times = self.event_times[mask]
        stream_ids = self.event_primitive_ids[mask]
        if entities is None:
            return stream_entities, stream_times, stream_ids
        selected = np.zeros(self.n_entities, dtype=bool)
        selected[np.asarray(entities, dtype=np.int64)] = True
        keep = selected[stream_entities]
        return stream_entities[keep], stream_times[keep], stream_ids[keep]

    def split(
        self, fractions: tuple[float, float, float], seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a disjoint entity split, preferring a pre-registered partition."""
        if self.partitions is not None:
            parts = tuple(
                np.flatnonzero(self.partitions == code).astype(np.int32)
                for code in (0, 1, 2)
            )
        elif len(np.unique(self.split_groups)) > 1:
            groups, counts = np.unique(self.split_groups, return_counts=True)
            if len(groups) < 3:
                raise ValueError("ordered entity split requires at least three cohorts")
            cumulative = np.cumsum(counts)
            first_target = fractions[0] * self.n_entities
            second_target = (fractions[0] + fractions[1]) * self.n_entities
            first = int(np.argmin(np.abs(cumulative - first_target)))
            second = int(np.argmin(np.abs(cumulative - second_target)))
            if second <= first:
                second = min(len(groups) - 1, first + 1)
            parts = tuple(
                np.flatnonzero(mask).astype(np.int32)
                for mask in (
                    self.split_groups <= groups[first],
                    (self.split_groups > groups[first])
                    & (self.split_groups <= groups[second]),
                    self.split_groups > groups[second],
                )
            )  # type: ignore[assignment]
        else:
            rng = np.random.default_rng(int(seed))
            order = rng.permutation(self.n_entities)
            first = int(round(fractions[0] * self.n_entities))
            second = int(round((fractions[0] + fractions[1]) * self.n_entities))
            parts = tuple(
                np.sort(part).astype(np.int32)
                for part in np.split(order, [first, second])
            )
        if any(len(part) == 0 for part in parts):
            raise ValueError("entity split produced an empty fit/cert/test partition")
        combined = np.concatenate(parts)
        if (
            len(combined) != self.n_entities
            or len(np.unique(combined)) != self.n_entities
        ):
            raise AssertionError("entity split is not a disjoint partition")
        return parts  # type: ignore[return-value]


def write_dataset(
    root: str | Path,
    *,
    entities: pd.DataFrame,
    events: pd.DataFrame,
    targets: pd.DataFrame,
    baseline_cells: pd.DataFrame | None = None,
    predicate_names: Iterable[str],
    predicate_roles: Iterable[str] | None = None,
    predicate_definitions: Iterable[dict[str, object]] | None = None,
    likelihood: str,
    time_unit: str,
    ticks_per_unit: int = 1,
    adverse_event_name: str,
    f0_contract: dict[str, object],
    provenance: dict[str, object],
) -> Path:
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    entity_columns = [
        "entity_id",
        "start_time",
        "end_time",
        "baseline_origin",
        "split_group",
    ]
    if "baseline_stratum" in entities.columns:
        entity_columns.append("baseline_stratum")
    if "partition" in entities.columns:
        entity_columns.append("partition")
    if "end_reason" in entities.columns:
        entity_columns.append("end_reason")
    if "dependency_group" in entities.columns:
        entity_columns.append("dependency_group")
    entities = entities[entity_columns].copy()
    if "primitive_event_id" not in events.columns:
        # Synthetic/unit-test datasets commonly construct one row per primitive
        # event.  Assigning deterministic row IDs preserves that exact meaning.
        # Production preprocessors must provide shared IDs when one raw record
        # emits several predicate attributes.
        events = events.copy()
        events["primitive_event_id"] = np.arange(len(events), dtype=np.int64)
    events = events[
        ["entity_code", "time", "predicate_code", "primitive_event_id"]
    ].copy()
    events = events.sort_values(
        ["entity_code", "time", "predicate_code", "primitive_event_id"],
        kind="stable",
    ).reset_index(drop=True)
    targets = targets[["entity_code", "time", "multiplicity"]].copy()
    if baseline_cells is not None:
        baseline_columns = ["entity_code", "time", "baseline_stratum"]
        if "exposure" in baseline_cells.columns:
            baseline_columns.append("exposure")
        baseline_cells = baseline_cells[baseline_columns].copy()
        baseline_cells = baseline_cells.sort_values(
            ["entity_code", "time"], kind="stable"
        ).reset_index(drop=True)
    entities.to_parquet(root / "entities.parquet", index=False)
    events.to_parquet(root / "events.parquet", index=False)
    targets.to_parquet(root / "targets.parquet", index=False)
    if baseline_cells is not None:
        baseline_cells.to_parquet(root / "baseline_cells.parquet", index=False)
    files = {}
    file_names = ["entities", "events", "targets"]
    if baseline_cells is not None:
        file_names.append("baseline_cells")
    for name in file_names:
        path = root / f"{name}.parquet"
        files[name] = {
            "path": path.name,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    predicate_names = tuple(str(value) for value in predicate_names)
    predicate_roles = (
        tuple("reported" for _ in predicate_names)
        if predicate_roles is None
        else tuple(str(value) for value in predicate_roles)
    )
    if len(predicate_roles) != len(predicate_names):
        raise ValueError("predicate_roles must align with predicate_names")
    predicate_definitions = (
        tuple({"kind": "event"} for _ in predicate_names)
        if predicate_definitions is None
        else tuple(dict(value) for value in predicate_definitions)
    )
    if len(predicate_definitions) != len(predicate_names):
        raise ValueError("predicate_definitions must align with predicate_names")
    identity: dict[str, object] = {
        "schema": DATASET_SCHEMA,
        "predicate_names": list(predicate_names),
        "predicate_roles": list(predicate_roles),
        "predicate_definitions": list(predicate_definitions),
        "likelihood": likelihood,
        "time_unit": time_unit,
        "ticks_per_unit": int(ticks_per_unit),
        "adverse_event_name": adverse_event_name,
        "f0_contract": f0_contract,
        "files": files,
        "provenance": provenance,
    }
    digest = _identity_digest(identity)
    manifest = {**identity, "dataset_digest": digest}
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    Dataset.load(root)
    return root
