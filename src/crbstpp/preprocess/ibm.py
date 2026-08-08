from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import write_dataset


# Reported atoms are target-blind, semantically defined transaction transitions.
# In particular, no target-calibrated amount, velocity, dormancy or cadence
# threshold is allowed in this dictionary.
PREDICATES = (
    "pred_out_new_receiver",
    "pred_in_new_sender",
    "pred_out_receiver_revisit_after_alternative",
    "pred_in_sender_revisit_after_alternative",
    "pred_out_currency_conversion",
    "pred_out_currency_switch",
    "pred_in_currency_switch",
    "pred_out_payment_format_switch_to_cash",
    "pred_out_payment_format_switch_to_wire",
    "pred_in_payment_format_switch_to_cash",
    "pred_in_payment_format_switch_to_wire",
    "pred_in_to_out_transition",
    "pred_out_to_in_transition",
    "pred_out_reciprocal_edge_onset",
)

# The target is a marked subset of outgoing transactions.  These two histories
# absorb generic account activity so reported rules are not rewarded merely for
# identifying accounts that transact frequently.
BASELINE_CONTROLS = (
    "control_outgoing_transaction_history",
    "control_incoming_transaction_history",
)

ALL_PREDICATES = (*PREDICATES, *BASELINE_CONTROLS)
INCOMING_PREDICATES = frozenset({1, 3, 6, 9, 10, 12})
PARTITION_NAMES = ("fit", "cert", "test")


def _fixed_entity_partition(
    count: int,
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> np.ndarray:
    if count < 3:
        raise ValueError("IBM preprocessing requires at least three accounts")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("entity partition fractions must sum to one")
    order = np.random.default_rng(int(seed)).permutation(count)
    first = int(round(fractions[0] * count))
    second = int(round((fractions[0] + fractions[1]) * count))
    partition = np.empty(count, dtype=np.int8)
    partition[order[:first]] = 0
    partition[order[first:second]] = 1
    partition[order[second:]] = 2
    if not np.array_equal(np.unique(partition), np.array([0, 1, 2])):
        raise ValueError("entity partition produced an empty split")
    return partition


def _update_two_latest(
    first_counterparty: np.ndarray,
    first_time: np.ndarray,
    second_counterparty: np.ndarray,
    second_time: np.ndarray,
    *,
    entity: int,
    counterparty: int,
    time: int,
) -> None:
    """Update the two most recent distinct counterparties for one entity."""

    if first_counterparty[entity] == counterparty:
        first_time[entity] = time
        return
    if second_counterparty[entity] == counterparty:
        second_time[entity] = time
        if second_time[entity] > first_time[entity]:
            (
                first_counterparty[entity],
                second_counterparty[entity],
            ) = (
                second_counterparty[entity],
                first_counterparty[entity],
            )
            first_time[entity], second_time[entity] = (
                second_time[entity],
                first_time[entity],
            )
        return
    if time >= first_time[entity]:
        second_counterparty[entity] = first_counterparty[entity]
        second_time[entity] = first_time[entity]
        first_counterparty[entity] = counterparty
        first_time[entity] = time
    elif time > second_time[entity]:
        second_counterparty[entity] = counterparty
        second_time[entity] = time


def _strict_flow_events(
    from_code: np.ndarray,
    to_code: np.ndarray,
    time: np.ndarray,
    n_accounts: int,
) -> tuple[np.ndarray, ...]:
    """Compute target-blind flow predicates from strictly earlier timestamps.

    Raw IBM timestamps have one-minute resolution and can contain ties.  Events
    at the same minute are evaluated against one frozen pre-minute state and
    committed only after the complete tie block.  Therefore CSV row order
    cannot manufacture a counterparty, revisit, direction or reciprocal event.
    """

    out_new = np.zeros(len(time), dtype=bool)
    in_new = np.zeros(len(time), dtype=bool)
    out_revisit = np.zeros(len(time), dtype=bool)
    in_revisit = np.zeros(len(time), dtype=bool)
    in_to_out = np.zeros(len(time), dtype=bool)
    out_to_in = np.zeros(len(time), dtype=bool)
    reciprocal = np.zeros(len(time), dtype=bool)
    last_in = np.full(n_accounts, -1, dtype=np.int64)
    last_out = np.full(n_accounts, -1, dtype=np.int64)
    edge_last_time: dict[int, int] = {}
    out_first_cp = np.full(n_accounts, -1, dtype=np.int32)
    out_first_time = np.full(n_accounts, -1, dtype=np.int64)
    out_second_cp = np.full(n_accounts, -1, dtype=np.int32)
    out_second_time = np.full(n_accounts, -1, dtype=np.int64)
    in_first_cp = np.full(n_accounts, -1, dtype=np.int32)
    in_first_time = np.full(n_accounts, -1, dtype=np.int64)
    in_second_cp = np.full(n_accounts, -1, dtype=np.int32)
    in_second_time = np.full(n_accounts, -1, dtype=np.int64)
    base = int(n_accounts)

    left = 0
    while left < len(time):
        current = int(time[left])
        right = left + 1
        while right < len(time) and int(time[right]) == current:
            right += 1
        for index in range(left, right):
            sender = int(from_code[index])
            receiver = int(to_code[index])
            edge = sender * base + receiver
            reverse = receiver * base + sender
            previous_edge = edge_last_time.get(edge)

            out_new[index] = last_out[sender] >= 0 and previous_edge is None
            in_new[index] = last_in[receiver] >= 0 and previous_edge is None
            if previous_edge is not None:
                latest_other_out = (
                    out_first_time[sender]
                    if out_first_cp[sender] != receiver
                    else out_second_time[sender]
                )
                latest_other_in = (
                    in_first_time[receiver]
                    if in_first_cp[receiver] != sender
                    else in_second_time[receiver]
                )
                out_revisit[index] = latest_other_out > previous_edge
                in_revisit[index] = latest_other_in > previous_edge
            if last_in[sender] > last_out[sender] and last_in[sender] < current:
                in_to_out[index] = True
            if last_out[receiver] > last_in[receiver] and last_out[receiver] < current:
                out_to_in[index] = True

            reverse_time = edge_last_time.get(reverse)
            reciprocal[index] = previous_edge is None and reverse_time is not None

        # Commit one update per directed edge only after every event at the
        # current timestamp has been evaluated against the frozen past.
        batch_edges = np.unique(
            from_code[left:right].astype(np.int64) * base
            + to_code[left:right].astype(np.int64)
        )
        for edge in batch_edges.tolist():
            sender, receiver = divmod(int(edge), base)
            edge_last_time[int(edge)] = current
            _update_two_latest(
                out_first_cp,
                out_first_time,
                out_second_cp,
                out_second_time,
                entity=sender,
                counterparty=receiver,
                time=current,
            )
            _update_two_latest(
                in_first_cp,
                in_first_time,
                in_second_cp,
                in_second_time,
                entity=receiver,
                counterparty=sender,
                time=current,
            )
            last_out[sender] = current
            last_in[receiver] = current
        left = right

    return (
        out_new,
        in_new,
        out_revisit,
        in_revisit,
        in_to_out,
        out_to_in,
        reciprocal,
    )


def _strict_state_switch(
    frame: pd.DataFrame,
    *,
    owner: str,
    value: str,
) -> np.ndarray:
    """Return switches from one unambiguous strictly-prior timestamp state."""

    keys = [owner, "_time_minute"]
    states = (
        frame.groupby(keys, sort=False)[value]
        .agg(state="first", state_count="nunique")
        .reset_index()
    )
    previous = states.groupby(owner, sort=False)[["state", "state_count"]].shift()
    states["_previous_state"] = previous["state"]
    states["_previous_count"] = previous["state_count"]
    aligned = frame[keys + [value]].merge(states, on=keys, how="left", sort=False)
    return (
        aligned["_previous_count"].eq(1) & aligned[value].ne(aligned["_previous_state"])
    ).to_numpy(dtype=bool)


def _predicate_masks(
    frame: pd.DataFrame,
    *,
    n_accounts: int,
) -> tuple[np.ndarray, ...]:
    from_code = frame["from_code"].to_numpy(dtype=np.int32)
    to_code = frame["to_code"].to_numpy(dtype=np.int32)
    # Predicate semantics use the finest timestamp available in the raw data,
    # even though the point-process risk intervals below are one hour wide.
    # This preserves strict chronological witnesses within an hour without
    # allowing a source to explain a target in that same model interval.
    time = frame["_time_minute"].to_numpy(dtype=np.int64)
    nonself = from_code != to_code

    payment_format = frame["payment_format"]

    (
        out_new,
        in_new,
        out_revisit,
        in_revisit,
        in_to_out,
        out_to_in,
        reciprocal,
    ) = _strict_flow_events(from_code, to_code, time, n_accounts)
    out_currency_switch = _strict_state_switch(
        frame, owner="from_code", value="payment_currency"
    )
    in_currency_switch = _strict_state_switch(
        frame, owner="to_code", value="receiving_currency"
    )
    out_format_switch = _strict_state_switch(
        frame, owner="from_code", value="payment_format"
    )
    in_format_switch = _strict_state_switch(
        frame, owner="to_code", value="payment_format"
    )
    conversion = (
        frame["payment_currency"].ne("")
        & frame["receiving_currency"].ne("")
        & frame["payment_currency"].ne(frame["receiving_currency"])
    ).to_numpy()

    masks = (
        out_new & nonself,
        in_new & nonself,
        out_revisit & nonself,
        in_revisit & nonself,
        conversion & nonself,
        out_currency_switch & nonself,
        in_currency_switch & nonself,
        out_format_switch & payment_format.eq("CASH").to_numpy() & nonself,
        out_format_switch & payment_format.eq("WIRE").to_numpy() & nonself,
        in_format_switch & payment_format.eq("CASH").to_numpy() & nonself,
        in_format_switch & payment_format.eq("WIRE").to_numpy() & nonself,
        in_to_out & nonself,
        out_to_in & nonself,
        reciprocal & nonself,
    )
    if len(masks) != len(PREDICATES):
        raise AssertionError("IBM predicate definitions are misaligned")
    return tuple(np.asarray(mask, dtype=bool) for mask in masks)


def _predicate_audit(
    *,
    output_root: Path,
    events: pd.DataFrame,
    targets: pd.DataFrame,
    partition: np.ndarray,
    start_time: int,
    end_time: int,
    ticks_per_hour: int,
) -> None:
    """Write a deterministic, target-blind structural dictionary audit.

    The audit deliberately reports counts rather than imposing a minimum
    frequency threshold.  Structural testability belongs to the downstream
    fit/cert split actually used by the estimator; preprocessing must not tune
    the rule dictionary to target outcomes.
    """

    reported = events.loc[
        events["predicate_code"].lt(len(PREDICATES)),
        ["entity_code", "time", "predicate_code"],
    ]
    split_summaries: dict[str, dict[str, int]] = {}
    target_codes = targets["entity_code"].to_numpy(dtype=np.int32)
    target_multiplicity = targets["multiplicity"].to_numpy(dtype=np.int64)
    for split_code, name in enumerate(PARTITION_NAMES):
        target_keep = partition[target_codes] == split_code
        split_summaries[name] = {
            "entities": int(np.sum(partition == split_code)),
            "target_entities": int(np.unique(target_codes[target_keep]).size),
            "target_events": int(target_multiplicity[target_keep].sum()),
        }

    predicate_summaries: list[dict[str, object]] = []
    entity_streams: list[np.ndarray] = []
    time_streams: list[np.ndarray] = []
    time_base = int(end_time - start_time + 1)
    for code, name in enumerate(PREDICATES):
        stream = reported.loc[
            reported["predicate_code"].eq(code), ["entity_code", "time"]
        ]
        stream_entities = stream["entity_code"].to_numpy(dtype=np.int32)
        unique_entities = np.unique(stream_entities)
        entity_streams.append(unique_entities)
        # A collision-free int64 key for exact same-entity/same-time overlap.
        time_keys = (
            stream_entities.astype(np.int64) * time_base
            + stream["time"].to_numpy(dtype=np.int64)
            - int(start_time)
        )
        time_streams.append(np.unique(time_keys))
        split_entities: list[int] = []
        split_events: list[int] = []
        for split_code in range(3):
            keep = partition[stream_entities] == split_code
            split_events.append(int(np.sum(keep)))
            split_entities.append(int(np.unique(stream_entities[keep]).size))
        entity_count = int(unique_entities.size)
        predicate_summaries.append(
            {
                "code": code,
                "name": name,
                "events": int(len(stream)),
                "entities": entity_count,
                "coverage_pct": round(100.0 * entity_count / len(partition), 6),
                "events_per_active_entity": round(
                    len(stream) / max(1, entity_count), 6
                ),
                "split_events": split_events,
                "split_entities": split_entities,
            }
        )

    entity_pairs: list[dict[str, object]] = []
    time_pairs: list[dict[str, object]] = []
    for left in range(len(PREDICATES)):
        for right in range(left + 1, len(PREDICATES)):
            entity_intersection = int(
                np.intersect1d(
                    entity_streams[left],
                    entity_streams[right],
                    assume_unique=True,
                ).size
            )
            entity_union = (
                len(entity_streams[left])
                + len(entity_streams[right])
                - entity_intersection
            )
            entity_pairs.append(
                {
                    "left": PREDICATES[left],
                    "right": PREDICATES[right],
                    "entities": entity_intersection,
                    "jaccard": round(entity_intersection / max(1, entity_union), 6),
                }
            )
            time_intersection = int(
                np.intersect1d(
                    time_streams[left],
                    time_streams[right],
                    assume_unique=True,
                ).size
            )
            time_union = (
                len(time_streams[left]) + len(time_streams[right]) - time_intersection
            )
            time_pairs.append(
                {
                    "left": PREDICATES[left],
                    "right": PREDICATES[right],
                    "intersection": time_intersection,
                    "jaccard": round(time_intersection / max(1, time_union), 6),
                }
            )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    audit = {
        "schema": "crbstpp.ibm_predicate_audit.v1",
        "dataset_digest": manifest["dataset_digest"],
        "entities": int(len(partition)),
        "hours": round((end_time - start_time) / float(ticks_per_hour), 6),
        "all_event_rows": int(len(events)),
        "reported_event_rows": int(len(reported)),
        "target_rows": int(len(targets)),
        "target_events": int(target_multiplicity.sum()),
        "splits": split_summaries,
        "predicates": predicate_summaries,
        "top_same_time_jaccard": sorted(
            time_pairs, key=lambda value: (-float(value["jaccard"]), value["left"])
        )[:12],
        "lowest_entity_pair_counts": sorted(
            entity_pairs, key=lambda value: (int(value["entities"]), value["left"])
        )[:12],
    }
    (output_root / "predicate_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def preprocess_ibm(
    raw_zip: str | Path,
    output_root: str | Path,
    *,
    partition_seed: int = 111,
    overwrite: bool = False,
) -> Path:
    raw_zip, output_root = Path(raw_zip), Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    usecols = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Receiving Currency",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    ]
    frame = pd.read_csv(raw_zip, usecols=usecols, dtype="string")
    frame = frame.rename(
        columns={
            "From Bank": "from_bank",
            "Account": "from_account",
            "To Bank": "to_bank",
            "Account.1": "to_account",
            "Receiving Currency": "receiving_currency",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "target",
        }
    )
    for name in (
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "receiving_currency",
        "payment_currency",
        "payment_format",
    ):
        frame[name] = frame[name].fillna("").str.strip()
    for name in ("receiving_currency", "payment_currency", "payment_format"):
        frame[name] = frame[name].str.upper()
    frame["_raw_order"] = np.arange(len(frame), dtype=np.int64)
    frame["timestamp"] = pd.to_datetime(
        frame.pop("Timestamp"), format="%Y/%m/%d %H:%M", errors="raise"
    )
    frame = frame.sort_values(
        ["timestamp", "_raw_order"],
        kind="stable",
    ).reset_index(drop=True)
    origin = frame["timestamp"].min()
    frame["_time_minute"] = (
        (frame["timestamp"] - origin).dt.total_seconds() // 60
    ).astype(np.int64)
    # The fitted estimator is an exact recurrent Poisson model on
    # pre-registered one-hour risk intervals.  Raw minute order is retained
    # above for predicate construction, while coarsening the likelihood grid
    # prevents an otherwise 60x expansion of every M-knot response.
    frame["time"] = (frame["_time_minute"] // 60).astype(np.int64)
    from_key = frame["from_bank"] + ":" + frame["from_account"]
    to_key = frame["to_bank"] + ":" + frame["to_account"]
    account_ids = np.unique(np.concatenate([from_key.to_numpy(), to_key.to_numpy()]))
    account_map = pd.Series(
        np.arange(len(account_ids), dtype=np.int32), index=account_ids
    )
    frame["from_code"] = from_key.map(account_map).astype(np.int32)
    frame["to_code"] = to_key.map(account_map).astype(np.int32)

    # Reinvestment/self-transfer rows are bookkeeping activity, not an
    # external counterparty or flow-direction witness.  They remain in the
    # generic activity controls and can remain targets, but cannot create or
    # alter a reported predicate.
    reported_frame = frame.loc[frame["from_code"].ne(frame["to_code"])].copy()
    masks = _predicate_masks(reported_frame, n_accounts=len(account_ids))
    event_parts: list[pd.DataFrame] = []
    reported_from = reported_frame["from_code"].to_numpy(dtype=np.int32)
    reported_to = reported_frame["to_code"].to_numpy(dtype=np.int32)
    reported_time = reported_frame["time"].to_numpy(dtype=np.int64)
    reported_primitive = reported_frame["_raw_order"].to_numpy(dtype=np.int64)
    for code, mask in enumerate(masks):
        owners = reported_to if code in INCOMING_PREDICATES else reported_from
        event_parts.append(
            pd.DataFrame(
                {
                    "entity_code": owners[mask],
                    "time": reported_time[mask],
                    "predicate_code": np.full(int(np.sum(mask)), code, dtype=np.int16),
                    "primitive_event_id": reported_primitive[mask],
                }
            )
        )

    # Generic transaction histories are fixed nuisance controls, not reported
    # discoveries.  Minute-level duplicates are collapsed only within an
    # entity/predicate stream; target event multiplicity remains intact.
    from_code = frame["from_code"].to_numpy(dtype=np.int32)
    to_code = frame["to_code"].to_numpy(dtype=np.int32)
    time = frame["time"].to_numpy(dtype=np.int64)
    primitive = frame["_raw_order"].to_numpy(dtype=np.int64)
    for code, owners in (
        (len(PREDICATES), from_code),
        (len(PREDICATES) + 1, to_code),
    ):
        event_parts.append(
            pd.DataFrame(
                {
                    "entity_code": owners,
                    "time": time,
                    "predicate_code": np.full(len(frame), code, dtype=np.int16),
                    "primitive_event_id": primitive,
                }
            )
        )
    events = (
        pd.concat(event_parts, ignore_index=True)
        .sort_values(
            ["entity_code", "time", "predicate_code", "primitive_event_id"],
            kind="stable",
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    target_mask = pd.to_numeric(frame["target"], errors="coerce").fillna(0).eq(1)
    target_rows = frame.loc[target_mask]
    targets = (
        target_rows.groupby(["from_code", "time"], as_index=False)
        .size()
        .rename(columns={"from_code": "entity_code", "size": "multiplicity"})
        .sort_values(["entity_code", "time"], kind="stable")
        .reset_index(drop=True)
    )

    start_time = int(frame["time"].min())
    end_time = int(frame["time"].max())
    partition = _fixed_entity_partition(
        len(account_ids),
        seed=partition_seed,
    )
    entities = pd.DataFrame(
        {
            "entity_id": account_ids,
            "dependency_group": account_ids,
            # AMLworld simulates a fixed account population.  Inactivity is
            # observed zero-event exposure, not a reason to truncate risk time
            # at each account's first/last transaction.
            "start_time": np.full(len(account_ids), start_time, dtype=np.int64),
            "end_time": np.full(len(account_ids), end_time, dtype=np.int64),
            "baseline_origin": np.full(len(account_ids), start_time, dtype=np.int64),
            "split_group": np.zeros(len(account_ids), dtype=np.int64),
            "partition": partition,
        }
    )

    reported = events[events["predicate_code"] < len(PREDICATES)]
    event_counts = (
        reported.groupby("predicate_code", sort=True)
        .size()
        .reindex(range(len(PREDICATES)), fill_value=0)
        .astype(int)
    )
    entity_counts = (
        reported.groupby("predicate_code", sort=True)["entity_code"]
        .nunique()
        .reindex(range(len(PREDICATES)), fill_value=0)
        .astype(int)
    )
    target_events_by_partition = np.bincount(
        partition[targets["entity_code"].to_numpy(dtype=np.int32)],
        weights=targets["multiplicity"].to_numpy(dtype=np.int32),
        minlength=3,
    ).astype(np.int64)
    output = write_dataset(
        output_root,
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=ALL_PREDICATES,
        predicate_roles=(
            *("reported" for _ in PREDICATES),
            *("baseline_control" for _ in BASELINE_CONTROLS),
        ),
        likelihood="poisson",
        time_unit="hour",
        ticks_per_unit=1,
        adverse_event_name="laundering-labelled outgoing transaction",
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded_from_reported_dictionary": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
            "primitive_event_provenance": True,
            "independent_certification_units": True,
        },
        provenance={
            "preprocessor": "crbstpp.preprocess.ibm.entity_semantic_hourly.v4",
            "source": str(raw_zip),
            "source_sha256": _sha256(raw_zip),
            "timestamp_resolution": (
                "raw one-minute order for predicate semantics; fixed one-hour "
                "risk intervals for recurrent Poisson likelihood"
            ),
            "predicate_definition": (
                "target-blind counterparty, currency, domain-defined channel and "
                "strictly-past flow transitions; no empirical quantile thresholds"
            ),
            "baseline_control_definition": {
                "names": list(BASELINE_CONTROLS),
                "construction": (
                    "strictly-future outgoing and incoming transaction histories "
                    "with the same M-knot lag basis as reported rules and a "
                    "pre-registered positive direction; fixed, non-reportable "
                    "nuisance identities in every null and support model"
                ),
            },
            "amount_policy": (
                "amount is not discretized in the unmarked v1 model; complete "
                "pass-through/split/merge motifs are reserved for external audit"
            ),
            "entity_definition": "bank-account pair",
            "entity_partition": {
                "method": "fixed label-blind random account partition",
                "seed": int(partition_seed),
                "fractions": [0.60, 0.20, 0.20],
            },
            "observation_bounds": (
                "all observed accounts share the complete AMLworld simulation window"
            ),
            "environment_definition": (
                "outcome-blind equal calendar-time blocks whose minimum width is "
                "max(formation window)+impact lag; F3 uses entity-clustered "
                "block loss contributions"
            ),
            "network_dependence_limitation": (
                "transactions link account entities across partitions; no target "
                "labels or fitted parameters are shared across partitions"
            ),
            "raw_transaction_count": int(len(frame)),
            "entity_count": int(len(account_ids)),
            "target_event_count": int(targets["multiplicity"].sum()),
            "target_events_by_partition": target_events_by_partition.tolist(),
            "reported_predicate_event_counts": {
                PREDICATES[index]: int(event_counts.iloc[index])
                for index in range(len(PREDICATES))
            },
            "reported_predicate_entity_counts": {
                PREDICATES[index]: int(entity_counts.iloc[index])
                for index in range(len(PREDICATES))
            },
        },
    )
    _predicate_audit(
        output_root=output,
        events=events,
        targets=targets,
        partition=partition,
        start_time=start_time,
        end_time=end_time,
        ticks_per_hour=1,
    )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
