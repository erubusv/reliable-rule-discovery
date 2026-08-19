from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


HISTORY_STATE_TRANSFORMS = (
    "recent",
    "recurrent",
    "accelerating",
    "decelerating",
)


@dataclass(frozen=True)
class StateIntervals:
    """Half-open-at-entry predictable state intervals.

    A state is active immediately *after* ``start`` through ``end``.  Hence a
    rule completed at tick ``t`` sees the state iff ``start < t <= end``.  The
    strict inequality prevents the current primitive from creating its own
    context.
    """

    entities: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    entry_primitive_ids: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.entities)
        if any(
            array.shape != (length,)
            for array in (self.starts, self.ends, self.entry_primitive_ids)
        ):
            raise ValueError("state interval arrays must align")
        if length and np.any(self.ends < self.starts):
            raise ValueError("state interval end precedes its entry")


def _merge_intervals(
    candidates: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    output: list[tuple[int, int, int]] = []
    for start, end, primitive in candidates:
        if end <= start:
            continue
        if output and start <= output[-1][1]:
            left, old_end, old_primitive = output[-1]
            output[-1] = (left, max(old_end, end), old_primitive)
        else:
            output.append((start, end, primitive))
    return output


def history_state_intervals(
    entities: np.ndarray,
    times: np.ndarray,
    primitive_ids: np.ndarray,
    entity_ends: np.ndarray,
    *,
    transform: str,
    horizon_ticks: int,
) -> StateIntervals:
    """Build one target-blind history state from a primitive event stream."""

    entities = np.asarray(entities, dtype=np.int32)
    times = np.asarray(times, dtype=np.int64)
    primitive_ids = np.asarray(primitive_ids, dtype=np.int64)
    ends = np.asarray(entity_ends, dtype=np.int64)
    if transform not in HISTORY_STATE_TRANSFORMS:
        raise ValueError(f"unsupported history-state transform: {transform}")
    if horizon_ticks < 1:
        raise ValueError("history-state horizon must be positive")
    if not (entities.shape == times.shape == primitive_ids.shape):
        raise ValueError("source event arrays must align")
    if len(entities) and (np.any(entities < 0) or np.any(entities >= len(ends))):
        raise ValueError("state source entity is outside the observation table")

    out_entities: list[int] = []
    out_starts: list[int] = []
    out_ends: list[int] = []
    out_primitives: list[int] = []
    if not len(entities):
        return StateIntervals(
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    order = np.lexsort((primitive_ids, times, entities))
    entities, times, primitive_ids = (
        entities[order],
        times[order],
        primitive_ids[order],
    )
    unique, first = np.unique(entities, return_index=True)
    for index, entity_value in enumerate(unique.tolist()):
        right = first[index + 1] if index + 1 < len(first) else len(entities)
        local_times = times[first[index] : right]
        local_ids = primitive_ids[first[index] : right]
        entity_end = int(ends[entity_value])
        candidates: list[tuple[int, int, int]] = []
        if transform == "recent":
            candidates = [
                (int(tick), min(entity_end, int(tick) + horizon_ticks), int(pid))
                for tick, pid in zip(local_times, local_ids, strict=True)
            ]
        elif transform == "recurrent":
            for position in range(1, len(local_times)):
                previous = int(local_times[position - 1])
                current = int(local_times[position])
                if current - previous <= horizon_ticks:
                    candidates.append(
                        (
                            current,
                            min(entity_end, previous + horizon_ticks),
                            int(local_ids[position]),
                        )
                    )
        else:
            for position in range(2, len(local_times)):
                previous_gap = int(local_times[position - 1] - local_times[position - 2])
                current_gap = int(local_times[position] - local_times[position - 1])
                selected = (
                    current_gap < previous_gap
                    if transform == "accelerating"
                    else current_gap > previous_gap
                )
                if not selected:
                    continue
                current = int(local_times[position])
                next_tick = (
                    int(local_times[position + 1])
                    if position + 1 < len(local_times)
                    else current + horizon_ticks
                )
                candidates.append(
                    (
                        current,
                        min(entity_end, current + horizon_ticks, next_tick),
                        int(local_ids[position]),
                    )
                )
        for start, end, primitive in _merge_intervals(candidates):
            out_entities.append(int(entity_value))
            out_starts.append(start)
            out_ends.append(end)
            out_primitives.append(primitive)
    return StateIntervals(
        np.asarray(out_entities, dtype=np.int32),
        np.asarray(out_starts, dtype=np.int64),
        np.asarray(out_ends, dtype=np.int64),
        np.asarray(out_primitives, dtype=np.int64),
    )


def transition_state_intervals(
    entry_entities: np.ndarray,
    entry_times: np.ndarray,
    entry_primitive_ids: np.ndarray,
    exit_entities: np.ndarray,
    exit_times: np.ndarray,
    entity_ends: np.ndarray,
) -> StateIntervals:
    """Build predictable intervals from target-blind state transitions.

    Entry at ``s`` activates the state strictly after ``s``.  Exit at ``e``
    leaves it active through ``e`` so an action that caused the observed exit
    is still evaluated under its immediately preceding state.  Redundant
    entries and exits are ignored deterministically.
    """

    entry_entities = np.asarray(entry_entities, dtype=np.int32)
    entry_times = np.asarray(entry_times, dtype=np.int64)
    entry_primitive_ids = np.asarray(entry_primitive_ids, dtype=np.int64)
    exit_entities = np.asarray(exit_entities, dtype=np.int32)
    exit_times = np.asarray(exit_times, dtype=np.int64)
    entity_ends = np.asarray(entity_ends, dtype=np.int64)
    if not (
        entry_entities.shape == entry_times.shape == entry_primitive_ids.shape
    ):
        raise ValueError("transition-state entry arrays must align")
    if exit_entities.shape != exit_times.shape:
        raise ValueError("transition-state exit arrays must align")

    output: list[tuple[int, int, int, int]] = []
    for entity in np.union1d(entry_entities, exit_entities).tolist():
        entries = np.flatnonzero(entry_entities == entity)
        exits = np.flatnonzero(exit_entities == entity)
        transitions = [
            *( (int(entry_times[i]), 1, int(entry_primitive_ids[i])) for i in entries ),
            *( (int(exit_times[i]), 0, -1) for i in exits ),
        ]
        transitions.sort(key=lambda item: (item[0], -item[1], item[2]))
        active_start: int | None = None
        active_primitive = -1
        for tick, entering, primitive in transitions:
            if entering:
                if active_start is None:
                    active_start = tick
                    active_primitive = primitive
                continue
            if active_start is None:
                continue
            output.append((int(entity), active_start, tick, active_primitive))
            active_start = None
            active_primitive = -1
        if active_start is not None:
            output.append(
                (
                    int(entity),
                    active_start,
                    int(entity_ends[int(entity)]),
                    active_primitive,
                )
            )
    return StateIntervals(
        np.asarray([item[0] for item in output], dtype=np.int32),
        np.asarray([item[1] for item in output], dtype=np.int64),
        np.asarray([item[2] for item in output], dtype=np.int64),
        np.asarray([item[3] for item in output], dtype=np.int64),
    )


def active_at(
    intervals: StateIntervals,
    entities: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Return whether each completion observes the state immediately before it."""

    entities = np.asarray(entities, dtype=np.int32)
    times = np.asarray(times, dtype=np.int64)
    if entities.shape != times.shape:
        raise ValueError("state queries must align")
    output = np.zeros(len(entities), dtype=bool)
    if not len(entities) or not len(intervals.entities):
        return output
    # One lexicographic vector search replaces a Python entity loop for every
    # state x action pair.  ``side=left`` implements start < t exactly.
    dtype = np.dtype([("entity", "<i4"), ("time", "<i8")])
    interval_keys = np.empty(len(intervals.entities), dtype=dtype)
    interval_keys["entity"] = intervals.entities
    interval_keys["time"] = intervals.starts
    query_keys = np.empty(len(entities), dtype=dtype)
    query_keys["entity"] = entities
    query_keys["time"] = times
    chosen = np.searchsorted(interval_keys, query_keys, side="left") - 1
    valid = chosen >= 0
    if np.any(valid):
        positions = np.flatnonzero(valid)
        selected = chosen[positions]
        valid[positions] &= (
            (intervals.entities[selected] == entities[positions])
            & (intervals.ends[selected] >= times[positions])
        )
    output[:] = valid
    return output


def fit_q90_horizon(
    events: pd.DataFrame,
    fit_entities: np.ndarray,
    predicate: int,
) -> int | None:
    """Return the past-only D_fit Q90 positive inter-arrival horizon."""

    selected = events.loc[
        events["predicate_code"].eq(int(predicate))
        & events["entity_code"].isin(np.asarray(fit_entities, dtype=np.int32)),
        ["entity_code", "time", "primitive_event_id"],
    ].sort_values(["entity_code", "time", "primitive_event_id"], kind="stable")
    if selected.empty:
        return None
    gaps = selected.groupby("entity_code", sort=False)["time"].diff().dropna()
    positive = gaps.loc[gaps > 0].to_numpy(dtype=np.int64)
    if not len(positive):
        return None
    return max(1, int(np.quantile(positive, 0.90, method="higher")))


def state_definition(
    *, source_predicate: int, transform: str, horizon: int
) -> dict[str, object]:
    if transform not in HISTORY_STATE_TRANSFORMS:
        raise ValueError(transform)
    return {
        "kind": "history_state",
        "source_predicate": int(source_predicate),
        "transform": str(transform),
        "horizon": int(horizon),
        "clock": "dataset_time_unit",
        "activation": "strictly_after_entry_through_state_exit",
    }


def event_definitions(count: int) -> tuple[dict[str, object], ...]:
    return tuple({"kind": "event"} for _ in range(int(count)))


def history_state_name(source_name: str, transform: str) -> str:
    stem = source_name[5:] if source_name.startswith("pred_") else source_name
    return f"state_{stem}_{transform}_q90"


def augment_history_state_dictionary(
    entities: pd.DataFrame,
    events: pd.DataFrame,
    *,
    predicate_names: tuple[str, ...],
    predicate_roles: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    """Append a common, fit-frozen state grammar without reading outcomes.

    Only reportable primitive events are lifted.  Structurally empty or exact
    duplicate state-entry streams are removed using D_fit alone.  Baseline
    control codes are shifted after the resulting leading reported block.
    """

    names = tuple(map(str, predicate_names))
    roles = tuple(map(str, predicate_roles))
    reported_count = sum(role == "reported" for role in roles)
    if tuple(roles[:reported_count]) != ("reported",) * reported_count:
        raise ValueError("reported predicates must be a leading block")
    if "partition" not in entities.columns:
        raise ValueError("history-state construction requires a frozen data split")
    fit_entities = np.flatnonzero(
        entities["partition"].to_numpy(dtype=np.int8) == 0
    ).astype(np.int32)
    entity_ends = entities["end_time"].to_numpy(dtype=np.int64)
    definitions: list[dict[str, object]] = [
        {"kind": "event"} for _ in range(reported_count)
    ]
    state_names: list[str] = []
    state_audit: list[dict[str, object]] = []
    seen_fit_signatures: set[tuple[tuple[int, int], ...]] = set()
    for source in range(reported_count):
        horizon = fit_q90_horizon(events, fit_entities, source)
        if horizon is None:
            continue
        source_rows = events.loc[
            events["predicate_code"].eq(source),
            ["entity_code", "time", "primitive_event_id"],
        ]
        stream = (
            source_rows["entity_code"].to_numpy(dtype=np.int32),
            source_rows["time"].to_numpy(dtype=np.int64),
            source_rows["primitive_event_id"].to_numpy(dtype=np.int64),
        )
        source_fit_signature = tuple(
            sorted(
                set(
                    zip(
                        source_rows.loc[
                            source_rows["entity_code"].isin(fit_entities),
                            "entity_code",
                        ].astype(int),
                        source_rows.loc[
                            source_rows["entity_code"].isin(fit_entities),
                            "time",
                        ].astype(int),
                        strict=True,
                    )
                )
            )
        )
        for transform in HISTORY_STATE_TRANSFORMS:
            intervals = history_state_intervals(
                *stream,
                entity_ends,
                transform=transform,
                horizon_ticks=int(horizon),
            )
            fit_mask = np.isin(intervals.entities, fit_entities)
            signature = tuple(
                zip(
                    intervals.entities[fit_mask].astype(int).tolist(),
                    intervals.starts[fit_mask].astype(int).tolist(),
                    strict=True,
                )
            )
            if not signature or signature == source_fit_signature:
                continue
            if signature in seen_fit_signatures:
                continue
            seen_fit_signatures.add(signature)
            state_names.append(history_state_name(names[source], transform))
            definitions.append(
                state_definition(
                    source_predicate=source,
                    transform=transform,
                    horizon=int(horizon),
                )
            )
            state_audit.append(
                {
                    "name": state_names[-1],
                    "source_predicate": source,
                    "source_name": names[source],
                    "transform": transform,
                    "q90_horizon": int(horizon),
                    "fit_entries": int(np.count_nonzero(fit_mask)),
                    "all_entries": int(len(intervals.entities)),
                }
            )
    state_count = len(state_names)
    output_events = events.copy()
    controls = output_events["predicate_code"].to_numpy(dtype=np.int64) >= reported_count
    if state_count and np.any(controls):
        output_events.loc[controls, "predicate_code"] = (
            output_events.loc[controls, "predicate_code"].to_numpy(dtype=np.int64)
            + state_count
        )
    control_names = names[reported_count:]
    control_roles = roles[reported_count:]
    definitions.extend({"kind": "event"} for _ in control_names)
    output_names = (*names[:reported_count], *state_names, *control_names)
    output_roles = (
        *("reported" for _ in range(reported_count + state_count)),
        *control_roles,
    )
    audit: dict[str, object] = {
        "state_grammar": list(HISTORY_STATE_TRANSFORMS),
        "state_predicates": state_audit,
        "state_predicate_count": state_count,
        "construction_split": "D_fit only",
        "horizon": "per-source positive inter-arrival Q90 on D_fit",
        "activation": "state entry for singleton; state active at t- for context",
        "structural_pruning": "empty, source-identical, and entry-identical only",
    }
    return (
        output_events,
        tuple(output_names),
        tuple(output_roles),
        tuple(definitions),
        audit,
    )


def augment_dataset_with_history_states(
    source_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Create a provenance-linked v5 dataset from an existing event dataset."""

    from .data import Dataset, write_dataset

    source_root, output_root = Path(source_root), Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    dataset = Dataset.load(source_root)
    if dataset.reported_state_predicates:
        raise ValueError("source dataset already contains history states")
    entities = pd.read_parquet(source_root / "entities.parquet")
    events = pd.read_parquet(source_root / "events.parquet")
    targets = pd.read_parquet(source_root / "targets.parquet")
    (
        events,
        names,
        roles,
        definitions,
        audit,
    ) = augment_history_state_dictionary(
        entities,
        events,
        predicate_names=dataset.predicate_names,
        predicate_roles=dataset.predicate_roles,
    )
    output = write_dataset(
        output_root,
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=names,
        predicate_roles=roles,
        predicate_definitions=definitions,
        likelihood=dataset.likelihood,
        time_unit=dataset.time_unit,
        ticks_per_unit=dataset.ticks_per_unit,
        adverse_event_name=dataset.adverse_event_name,
        f0_contract={
            **dataset.f0_contract,
            "history_states_fit_frozen": True,
            "history_states_target_blind": True,
        },
        provenance={
            "preprocessor": "crbstpp.history_state_augmentation.v1",
            "source_dataset_digest": dataset.digest,
            "source_dataset": str(source_root),
            "history_state_contract": audit,
        },
    )
    (output / "state_predicate_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
