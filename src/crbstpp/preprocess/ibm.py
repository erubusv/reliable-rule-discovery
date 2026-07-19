from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import write_dataset


PREDICATES = (
    "pred_out_amount_spike_rel_mean",
    "pred_out_receiver_novelty_after_history",
    "pred_in_sender_novelty_after_history",
    "pred_out_burst_hour_starts",
    "pred_in_burst_hour_starts",
    "pred_out_currency_switch_after_history",
    "pred_out_format_switch_after_history",
    "pred_out_bank_route_switch_after_history",
    "pred_out_dormancy_reactivation",
    "pred_in_to_out_turnaround_starts",
    "pred_out_amount_drop_rel_mean",
    "pred_in_amount_spike_rel_mean",
    "pred_out_receiver_revisit_after_alternative",
    "pred_in_sender_revisit_after_alternative",
    "pred_out_cadence_acceleration",
)


def _quantile(values: np.ndarray, probability: float, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if finite.size else float(default)


def preprocess_ibm(
    raw_zip: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
    max_rows: int | None = None,
) -> Path:
    raw_zip, output_root = Path(raw_zip), Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    usecols = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
        "Payment Format", "Is Laundering",
    ]
    frame = pd.read_csv(raw_zip, usecols=usecols, nrows=max_rows)
    frame = frame.rename(columns={
        "From Bank": "from_bank", "Account": "from_account", "To Bank": "to_bank",
        "Account.1": "to_account", "Amount Received": "received",
        "Receiving Currency": "receiving_currency", "Amount Paid": "paid",
        "Payment Currency": "payment_currency", "Payment Format": "payment_format",
        "Is Laundering": "target",
    })
    frame["timestamp"] = pd.to_datetime(frame.pop("Timestamp"), format="%Y/%m/%d %H:%M", errors="raise")
    frame = frame.sort_values(
        ["timestamp", "from_bank", "from_account", "to_bank", "to_account"], kind="stable"
    ).reset_index(drop=True)
    origin = frame["timestamp"].min()
    frame["time"] = ((frame["timestamp"] - origin).dt.total_seconds() // 3600).astype(np.int64)
    from_key = frame["from_bank"].astype(str) + ":" + frame["from_account"].astype(str)
    to_key = frame["to_bank"].astype(str) + ":" + frame["to_account"].astype(str)
    account_ids = np.unique(np.concatenate([from_key.to_numpy(), to_key.to_numpy()]))
    account_map = pd.Series(np.arange(len(account_ids), dtype=np.int32), index=account_ids)
    frame["from_code"] = from_key.map(account_map).astype(np.int32)
    frame["to_code"] = to_key.map(account_map).astype(np.int32)
    calibration_end = frame["timestamp"].quantile(0.20)
    calibration = frame.loc[frame["timestamp"] <= calibration_end]
    hourly_out = calibration.groupby(["from_code", "time"], sort=False).size().to_numpy()
    hourly_in = calibration.groupby(["to_code", "time"], sort=False).size().to_numpy()
    out_group = frame.groupby("from_code", sort=False)
    in_group = frame.groupby("to_code", sort=False)
    out_count = out_group.cumcount().to_numpy(dtype=np.int64)
    in_count = in_group.cumcount().to_numpy(dtype=np.int64)
    paid = pd.to_numeric(frame["paid"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    received = pd.to_numeric(frame["received"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    prior_out_sum = out_group["paid"].cumsum().to_numpy(dtype=np.float64) - paid
    prior_in_sum = in_group["received"].cumsum().to_numpy(dtype=np.float64) - received
    prior_out_mean = np.divide(prior_out_sum, out_count, out=np.zeros_like(paid), where=out_count > 0)
    prior_in_mean = np.divide(prior_in_sum, in_count, out=np.zeros_like(received), where=in_count > 0)
    out_pair_count = frame.groupby(["from_code", "to_code"], sort=False).cumcount().to_numpy()
    in_pair_count = frame.groupby(["to_code", "from_code"], sort=False).cumcount().to_numpy()
    previous_receiver = out_group["to_code"].shift().to_numpy(dtype=np.float64)
    previous_sender = in_group["from_code"].shift().to_numpy(dtype=np.float64)
    previous_pair_out_time = frame.groupby(["from_code", "to_code"], sort=False)["time"].shift().to_numpy(dtype=np.float64)
    previous_pair_in_time = frame.groupby(["to_code", "from_code"], sort=False)["time"].shift().to_numpy(dtype=np.float64)
    previous_out_time = out_group["time"].shift().to_numpy(dtype=np.float64)
    previous_out_gap = pd.Series(frame["time"].to_numpy() - previous_out_time).groupby(frame["from_code"], sort=False).shift().to_numpy(dtype=np.float64)
    out_gap = frame["time"].to_numpy(dtype=np.float64) - previous_out_time
    last_incoming_by_account: dict[int, int] = {}
    turnaround = np.zeros(len(frame), dtype=bool)
    for index, row in enumerate(frame[["from_code", "to_code", "time"]].itertuples(index=False)):
        last_in = last_incoming_by_account.get(int(row.from_code))
        if last_in is not None and (not np.isfinite(previous_out_time[index]) or previous_out_time[index] <= last_in):
            turnaround[index] = True
        last_incoming_by_account[int(row.to_code)] = int(row.time)
    out_hour_pos = frame.groupby(["from_code", "time"], sort=False).cumcount().to_numpy() + 1
    in_hour_pos = frame.groupby(["to_code", "time"], sort=False).cumcount().to_numpy() + 1
    dormancy = _quantile(
        (calibration["time"] - calibration.groupby("from_code", sort=False)["time"].shift()).to_numpy(dtype=np.float64),
        0.95, 1.0,
    )
    ratio_out = np.divide(paid, np.maximum(prior_out_mean, np.finfo(float).eps))
    ratio_in = np.divide(received, np.maximum(prior_in_mean, np.finfo(float).eps))
    q95_out = _quantile(ratio_out[(out_count >= 3) & (frame["timestamp"] <= calibration_end)], 0.95, 5.0)
    q05_out = _quantile(ratio_out[(out_count >= 3) & (frame["timestamp"] <= calibration_end)], 0.05, 0.2)
    q95_in = _quantile(ratio_in[(in_count >= 3) & (frame["timestamp"] <= calibration_end)], 0.95, 5.0)
    cadence = np.divide(out_gap, previous_out_gap, out=np.full(len(frame), np.nan), where=previous_out_gap > 0)
    cadence_q05 = _quantile(cadence[frame["timestamp"] <= calibration_end], 0.05, 0.25)
    return_out_gap = _quantile(
        (frame["time"].to_numpy() - previous_pair_out_time)[frame["timestamp"] <= calibration_end], 0.95, 1.0
    )
    return_in_gap = _quantile(
        (frame["time"].to_numpy() - previous_pair_in_time)[frame["timestamp"] <= calibration_end], 0.95, 1.0
    )
    out_hour_threshold = max(2, int(np.ceil(_quantile(hourly_out.astype(float), 0.95, 3.0))))
    in_hour_threshold = max(2, int(np.ceil(_quantile(hourly_in.astype(float), 0.95, 3.0))))
    masks = (
        (out_count >= 3) & (ratio_out >= q95_out),
        (out_pair_count == 0) & (out_count >= 1) & (frame["from_code"].to_numpy() != frame["to_code"].to_numpy()),
        (in_pair_count == 0) & (in_count >= 1) & (frame["from_code"].to_numpy() != frame["to_code"].to_numpy()),
        out_hour_pos == out_hour_threshold,
        in_hour_pos == in_hour_threshold,
        (out_count >= 1) & frame["payment_currency"].ne(out_group["payment_currency"].shift()).fillna(False).to_numpy(),
        (out_count >= 1) & frame["payment_format"].ne(out_group["payment_format"].shift()).fillna(False).to_numpy(),
        (out_count >= 1) & frame["to_bank"].ne(out_group["to_bank"].shift()).fillna(False).to_numpy(),
        np.isfinite(out_gap) & (out_gap >= dormancy),
        turnaround,
        (out_count >= 3) & (ratio_out <= q05_out),
        (in_count >= 3) & (ratio_in >= q95_in),
        (out_pair_count > 0) & np.isfinite(previous_receiver) & (previous_receiver != frame["to_code"].to_numpy()) & ((frame["time"].to_numpy() - previous_pair_out_time) >= return_out_gap),
        (in_pair_count > 0) & np.isfinite(previous_sender) & (previous_sender != frame["from_code"].to_numpy()) & ((frame["time"].to_numpy() - previous_pair_in_time) >= return_in_gap),
        (out_gap > 0) & (previous_out_gap > 0) & (cadence <= cadence_q05),
    )
    event_parts = []
    for code, mask in enumerate(masks):
        owners = frame["to_code"].to_numpy() if code in {2, 4, 11, 13} else frame["from_code"].to_numpy()
        event_parts.append(pd.DataFrame({
            "entity_code": owners[mask].astype(np.int32),
            "time": frame.loc[mask, "time"].to_numpy(dtype=np.int64),
            "predicate_code": np.full(int(np.sum(mask)), code, dtype=np.int16),
        }))
    events = pd.concat(event_parts, ignore_index=True).sort_values(
        ["entity_code", "time", "predicate_code"], kind="stable"
    ).drop_duplicates().reset_index(drop=True)
    target_rows = frame.loc[frame["target"].astype(bool)]
    targets = (
        target_rows.groupby(["from_code", "time"], as_index=False).size()
        .rename(columns={"from_code": "entity_code", "size": "multiplicity"})
        .sort_values(["entity_code", "time"], kind="stable").reset_index(drop=True)
    )
    participation = pd.concat([
        frame[["from_code", "time"]].rename(columns={"from_code": "entity_code"}),
        frame[["to_code", "time"]].rename(columns={"to_code": "entity_code"}),
    ], ignore_index=True)
    bounds = participation.groupby("entity_code", sort=True)["time"].agg(["min", "max"])
    entities = pd.DataFrame({
        "entity_id": account_ids,
        "start_time": bounds.reindex(np.arange(len(account_ids)))["min"].to_numpy(dtype=np.int64),
        "end_time": bounds.reindex(np.arange(len(account_ids)))["max"].to_numpy(dtype=np.int64),
        "baseline_origin": np.zeros(len(account_ids), dtype=np.int64),
        "split_group": np.zeros(len(account_ids), dtype=np.int64),
    })
    return write_dataset(
        output_root,
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=PREDICATES,
        likelihood="poisson",
        time_unit="hour",
        adverse_event_name="laundering-labelled outgoing transaction",
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
        },
        provenance={
            "preprocessor": "crbstpp.preprocess.ibm.v1",
            "source": str(raw_zip),
            "calibration": "chronological first 20% of transactions; target labels unused",
            "predicate_count": len(PREDICATES),
        },
    )
