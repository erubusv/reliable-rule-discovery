#!/usr/bin/env python3
"""Build account-level sparse TPP event sequences from IBM AMLworld data.

The raw AMLworld transaction table is an edge stream:

    source account -> destination account at timestamp

This script converts the edge stream into account-level event sequences.  The
target is a laundering-labeled outgoing transaction for the source account.
Predicates are observable transaction/flow events and never use the laundering
label or any pattern/typology labels.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RAW_ZIP = Path("data/ibm_aml/raw/HI-Small_Trans.csv.zip")
DEFAULT_OUTPUT_ROOT = Path("data/ibm_aml/processed/hi_small_tpp_v1")
TARGET_EVENT_NAME = "T_AML_LAUNDERING_OUT"


@dataclass(frozen=True)
class PredicateInfo:
    name: str
    family: str
    role: str
    tier: str
    description: str
    literature_basis: str


PREDICATE_CATALOG: list[PredicateInfo] = [
    PredicateInfo(
        "pred_out_cash_or_cheque",
        "payment_channel",
        "source",
        "candidate",
        "Outgoing transaction used cash or cheque.",
        "Payment-format features are basic AMLworld/GFP transaction features.",
    ),
    PredicateInfo(
        "pred_out_wire_or_ach",
        "payment_channel",
        "source",
        "candidate",
        "Outgoing transaction used wire or ACH.",
        "Payment-format features are basic AMLworld/GFP transaction features.",
    ),
    PredicateInfo(
        "pred_out_credit_card",
        "payment_channel",
        "source",
        "candidate",
        "Outgoing transaction used credit-card payment format.",
        "Payment-format features are basic AMLworld/GFP transaction features.",
    ),
    PredicateInfo(
        "pred_out_bitcoin",
        "payment_channel",
        "source",
        "candidate",
        "Outgoing transaction used Bitcoin format or currency.",
        "AMLworld includes cryptocurrency-like payment channels.",
    ),
    PredicateInfo(
        "pred_out_reinvestment",
        "payment_channel",
        "source",
        "candidate",
        "Outgoing transaction used reinvestment payment format.",
        "AMLworld models placement/layering/integration including reinvestment-like activity.",
    ),
    PredicateInfo(
        "pred_currency_conversion_out",
        "currency",
        "source",
        "core",
        "Outgoing transaction paid and received in different currencies.",
        "AMLworld and SAML-D include multiple currencies; conversion can support layering.",
    ),
    PredicateInfo(
        "pred_cross_bank_out",
        "institution",
        "source",
        "core",
        "Outgoing transaction moved funds across banks.",
        "AMLworld models multiple banks; interbank movement is a basic flow feature.",
    ),
    PredicateInfo(
        "pred_cross_bank_in",
        "institution",
        "receiver",
        "core",
        "Incoming transaction arrived from a different bank.",
        "AMLworld models multiple banks; interbank movement is a basic flow feature.",
    ),
    PredicateInfo(
        "pred_out_large_q95",
        "amount",
        "source",
        "candidate",
        "Outgoing amount exceeded the calibration-period 95th percentile.",
        "Transaction amount is a basic AMLworld/GFP transaction feature.",
    ),
    PredicateInfo(
        "pred_out_very_large_q99",
        "amount",
        "source",
        "core",
        "Outgoing amount exceeded the calibration-period 99th percentile.",
        "Large transfers are standard AML transaction-monitoring risk factors.",
    ),
    PredicateInfo(
        "pred_out_amount_spike_rel_mean",
        "amount",
        "source",
        "core",
        "Outgoing amount was at least 5x the source account's prior outgoing mean and globally non-small.",
        "Behavioral deviation features are common in AML transaction monitoring.",
    ),
    PredicateInfo(
        "pred_out_round_amount",
        "amount",
        "source",
        "core",
        "Outgoing amount was a large round-number transfer.",
        "Round-amount and threshold-like behavior are common rule-monitoring signals.",
    ),
    PredicateInfo(
        "pred_out_structured_small_repeats_day",
        "structuring",
        "source",
        "core",
        "The source first reached three qualifying small outgoing transfers in the day.",
        "Structuring/smurfing splits funds into multiple smaller transfers.",
    ),
    PredicateInfo(
        "pred_new_receiver_out",
        "counterparty",
        "source",
        "core",
        "Source account sent to a receiver not previously observed for that source.",
        "Counterparty novelty is a behavioral AML signal.",
    ),
    PredicateInfo(
        "pred_many_new_receivers_day",
        "fan_out",
        "source",
        "core",
        "Source account first reached three distinct receivers in the day.",
        "AMLworld/GFP use fan-out patterns as laundering motifs.",
    ),
    PredicateInfo(
        "pred_outgoing_burst_hour",
        "velocity",
        "source",
        "candidate",
        "Source account reached its third outgoing transaction in the hour.",
        "Velocity and burst behavior are common AML monitoring signals.",
    ),
    PredicateInfo(
        "pred_new_sender_in",
        "counterparty",
        "receiver",
        "candidate",
        "Receiver account received funds from a sender not previously observed for that receiver.",
        "Counterparty novelty is a behavioral AML signal.",
    ),
    PredicateInfo(
        "pred_many_unique_senders_day",
        "fan_in",
        "receiver",
        "core",
        "Receiver account first reached three distinct senders in the day.",
        "AMLworld/GFP use fan-in patterns as laundering motifs.",
    ),
    PredicateInfo(
        "pred_incoming_burst_hour",
        "velocity",
        "receiver",
        "candidate",
        "Receiver account reached its third incoming transaction in the hour.",
        "Velocity and burst behavior are common AML monitoring signals.",
    ),
    PredicateInfo(
        "pred_rapid_in_to_out_day",
        "pass_through",
        "source",
        "core",
        "The first outgoing transition after incoming activity in the day transferred material funds.",
        "Pass-through and layering behavior are core AML flow motifs.",
    ),
    PredicateInfo(
        "pred_fan_in_then_out_day",
        "gather_scatter",
        "source",
        "core",
        "The first outgoing transition after a same-day fan-in transferred material funds.",
        "AMLworld/GFP describe gather-scatter as fan-in followed by fan-out.",
    ),
    PredicateInfo(
        "pred_cycle_return_72h",
        "cycle",
        "source",
        "core",
        "Outgoing transaction reversed a prior opposite-direction transfer within 72 hours.",
        "GFP and AMLworld use cycle patterns as suspicious financial-crime motifs.",
    ),
    PredicateInfo(
        "pred_self_transfer",
        "self_flow",
        "source",
        "candidate",
        "Source and receiver account endpoints were identical.",
        "Self/reinvestment flows are present in AMLworld and should be audited for frequency.",
    ),
    PredicateInfo(
        "pred_out_receiver_novelty_after_history",
        "counterparty_change",
        "source",
        "dynamic_nonproxy",
        "A new outgoing receiver appeared after at least one prior outgoing transaction.",
        "History-conditioned counterparty novelty is an observable behavioral change.",
    ),
    PredicateInfo(
        "pred_in_sender_novelty_after_history",
        "counterparty_change",
        "receiver",
        "dynamic_nonproxy",
        "A new incoming sender appeared after at least one prior incoming transaction.",
        "History-conditioned counterparty novelty is an observable behavioral change.",
    ),
    PredicateInfo(
        "pred_out_burst_hour_starts",
        "velocity_change",
        "source",
        "dynamic_nonproxy",
        "The calibration-upper-tail outgoing count in an hour was first reached.",
        "Onset encoding removes future-within-bin look-ahead and persistent burst-state overlap.",
    ),
    PredicateInfo(
        "pred_in_burst_hour_starts",
        "velocity_change",
        "receiver",
        "dynamic_nonproxy",
        "The calibration-upper-tail incoming count in an hour was first reached.",
        "Onset encoding removes future-within-bin look-ahead and persistent burst-state overlap.",
    ),
    PredicateInfo(
        "pred_out_currency_switch_after_history",
        "currency_change",
        "source",
        "dynamic_nonproxy",
        "Outgoing payment currency changed after at least one prior outgoing transaction.",
        "A within-account currency transition is dynamic and does not use the laundering label.",
    ),
    PredicateInfo(
        "pred_out_format_switch_after_history",
        "format_change",
        "source",
        "dynamic_nonproxy",
        "Outgoing payment format changed after at least one prior outgoing transaction.",
        "A within-account payment-format transition is dynamic and label-free.",
    ),
    PredicateInfo(
        "pred_out_bank_route_switch_after_history",
        "route_change",
        "source",
        "dynamic_nonproxy",
        "Destination bank changed after at least one prior outgoing transaction.",
        "A within-account routing transition is dynamic and label-free.",
    ),
    PredicateInfo(
        "pred_out_dormancy_reactivation",
        "timing_change",
        "source",
        "dynamic_nonproxy",
        "Outgoing activity resumed after an interarrival above the calibration-period 95th percentile.",
        "A calibration-frozen interarrival threshold captures dynamic reactivation without target labels.",
    ),
    PredicateInfo(
        "pred_in_to_out_turnaround_starts",
        "direction_change",
        "source",
        "dynamic_nonproxy",
        "The first outgoing transaction after a strictly earlier incoming transaction occurred.",
        "A direction change is observable behavior; onset encoding avoids persistent pass-through states.",
    ),
    PredicateInfo(
        "pred_out_amount_drop_rel_mean",
        "amount_change",
        "source",
        "dynamic_nonproxy",
        "Outgoing amount fell into the calibration lower tail relative to the account's prior mean.",
        "A history-relative amount contraction is dynamic and outcome-blind.",
    ),
    PredicateInfo(
        "pred_in_amount_spike_rel_mean",
        "amount_change",
        "receiver",
        "dynamic_nonproxy",
        "Incoming amount entered the calibration upper tail relative to the account's prior incoming mean.",
        "A history-relative incoming amount expansion is dynamic and outcome-blind.",
    ),
    PredicateInfo(
        "pred_out_receiver_revisit_after_alternative",
        "counterparty_return",
        "source",
        "dynamic_nonproxy",
        "An earlier receiver was revisited after the immediately preceding outgoing receiver differed.",
        "Counterparty return after an alternative is a dynamic transition rather than a static identity.",
    ),
    PredicateInfo(
        "pred_in_sender_revisit_after_alternative",
        "counterparty_return",
        "receiver",
        "dynamic_nonproxy",
        "An earlier sender returned after the immediately preceding incoming sender differed.",
        "Sender return after an alternative is a dynamic transition rather than a static identity.",
    ),
    PredicateInfo(
        "pred_out_cadence_acceleration",
        "timing_change",
        "source",
        "dynamic_nonproxy",
        "A positive outgoing interarrival contracted into the calibration lower tail versus the previous gap.",
        "Interarrival contraction is a label-free dynamic timing change and excludes simultaneous-hour bursts.",
    ),
]


LOW_OVERLAP_PREDICATES = [
    "pred_currency_conversion_out",
    "pred_out_very_large_q99",
    "pred_out_amount_spike_rel_mean",
    "pred_out_structured_small_repeats_day",
    "pred_many_new_receivers_day",
    "pred_outgoing_burst_hour",
    "pred_many_unique_senders_day",
    "pred_incoming_burst_hour",
    "pred_rapid_in_to_out_day",
    "pred_fan_in_then_out_day",
    "pred_cycle_return_72h",
]

DYNAMIC_NONPROXY_CANDIDATES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IBM AMLworld as sparse TPP sequences.")
    parser.add_argument("--raw-zip", type=Path, default=DEFAULT_RAW_ZIP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row cap.")
    parser.add_argument("--part-size", type=int, default=100_000)
    parser.add_argument(
        "--time-unit",
        choices=["hour", "two_hour"],
        default="hour",
        help="Integer time grid used by the TPP algorithm.",
    )
    parser.add_argument(
        "--calibration-frac",
        type=float,
        default=0.6,
        help="Earliest temporal fraction used for unsupervised amount thresholds.",
    )
    parser.add_argument(
        "--predicate-tier",
        choices=["low_overlap", "core", "candidate", "dynamic_nonproxy_candidate"],
        default="candidate",
        help=(
            "candidate keeps all audited predicates; core keeps domain-core predicates; "
            "low_overlap keeps a compact first experiment set after frequency/overlap screening."
        ),
    )
    parser.add_argument(
        "--include-laundering-transaction-predicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build observable predicates from every transaction regardless of its target label "
            "(default: true). The negative form is retained only for legacy ablations; using it "
            "makes covariate construction outcome-dependent."
        ),
    )
    parser.add_argument(
        "--analysis-start-frac",
        type=float,
        help=(
            "When supplied, use the earliest fraction only for label-free calibration/history, "
            "start the TPP observation window strictly afterward, and attach pre-window USD flow exposure."
        ),
    )
    return parser.parse_args()


def selected_predicates(tier: str) -> list[str]:
    if tier == "candidate":
        return [p.name for p in PREDICATE_CATALOG]
    if tier == "low_overlap":
        catalog_names = {p.name for p in PREDICATE_CATALOG}
        missing = [p for p in LOW_OVERLAP_PREDICATES if p not in catalog_names]
        if missing:
            raise ValueError(f"low-overlap predicate list has unknown predicates: {missing}")
        return list(LOW_OVERLAP_PREDICATES)
    if tier == "dynamic_nonproxy_candidate":
        return list(DYNAMIC_NONPROXY_CANDIDATES)
    return [p.name for p in PREDICATE_CATALOG if p.tier == "core"]


def read_transactions(path: Path, max_rows: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"raw AML zip not found: {path}")
    usecols = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    ]
    df = pd.read_csv(
        path,
        usecols=usecols,
        nrows=max_rows,
        dtype={
            "Timestamp": "string",
            "From Bank": "string",
            "Account": "string",
            "To Bank": "string",
            "Account.1": "string",
            "Amount Received": "float64",
            "Receiving Currency": "category",
            "Amount Paid": "float64",
            "Payment Currency": "category",
            "Payment Format": "category",
            "Is Laundering": "int8",
        },
    )
    df = df.rename(
        columns={
            "Timestamp": "timestamp_raw",
            "From Bank": "from_bank",
            "Account": "from_account",
            "To Bank": "to_bank",
            "Account.1": "to_account",
            "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency",
            "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "is_laundering",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp_raw"], format="%Y/%m/%d %H:%M")
    df = df.drop(columns=["timestamp_raw"])
    df = df.sort_values(["timestamp", "from_bank", "from_account", "to_bank", "to_account"], kind="mergesort")
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return df


def assign_time_index(df: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    start = df["timestamp"].min()
    delta_minutes = (df["timestamp"] - start).dt.total_seconds().floordiv(60).astype("int64")
    if unit == "hour":
        df["time_index"] = np.floor_divide(delta_minutes, 60).astype("int32")
    elif unit == "two_hour":
        df["time_index"] = np.floor_divide(delta_minutes, 120).astype("int32")
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(unit)
    # Keep the generic preprocessor safe beyond the short AMLworld horizon;
    # int16 silently wraps after roughly 89 years of daily bins.
    df["day_index"] = np.floor_divide(delta_minutes, 60 * 24).astype("int32")
    return df, start


def factorize_accounts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from_pairs = pd.MultiIndex.from_arrays(
        [df["from_bank"], df["from_account"]], names=["bank", "account"]
    )
    to_pairs = pd.MultiIndex.from_arrays(
        [df["to_bank"], df["to_account"]], names=["bank", "account"]
    )
    combined = from_pairs.append(to_pairs)
    codes, uniques = pd.factorize(combined, sort=False)
    n = len(df)
    df["from_code"] = codes[:n].astype("int32")
    df["to_code"] = codes[n:].astype("int32")
    accounts = pd.DataFrame(
        {
            "account_code": np.arange(len(uniques), dtype=np.int32),
            "account_key": pd.Series(
                [f"B{bank}:{account}" for bank, account in uniques], dtype="string"
            ),
        }
    )
    accounts["sequence_id"] = "AML" + accounts["account_code"].astype(str).str.zfill(8)
    return df, accounts


def calibration_thresholds(df: pd.DataFrame, calibration_frac: float) -> dict[str, float]:
    if not (0.0 < calibration_frac <= 1.0):
        raise ValueError("--calibration-frac must be in (0, 1]")
    max_time = int(df["time_index"].max())
    cutoff = int(np.floor(max_time * calibration_frac))
    # Calibration is outcome-blind: selecting only non-laundering rows would use
    # the target label while constructing the covariates.
    calib_rows = df.loc[df["time_index"].le(cutoff)].copy()
    if "amount_paid_usd" not in df or "amount_received_usd" not in df:
        raise ValueError("USD-normalized amounts must be computed before calibration thresholds")
    calib = calib_rows["amount_paid_usd"].astype("float64")
    if calib.empty:
        calib = df["amount_paid_usd"].astype("float64")
    qs = calib.quantile([0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
    thresholds = {f"amount_q{int(q * 100):02d}": float(v) for q, v in qs.items()}
    calib_outgoing = calib_rows.groupby("from_code", sort=False)
    calib_incoming = calib_rows.groupby("to_code", sort=False)
    calib_prev_count = calib_outgoing.cumcount().to_numpy(dtype=np.int32)
    calib_cum_amount = calib_outgoing["amount_paid_usd"].cumsum().to_numpy(dtype=np.float64)
    calib_amount = calib_rows["amount_paid_usd"].to_numpy(dtype=np.float64)
    calib_prior_mean = np.divide(
        calib_cum_amount - calib_amount,
        calib_prev_count,
        out=np.full(len(calib_rows), np.nan, dtype=np.float64),
        where=calib_prev_count > 0,
    )
    stable = calib_prev_count >= 3
    ratios = calib_amount[stable] / np.maximum(calib_prior_mean[stable], np.finfo(np.float64).eps)
    ratios = ratios[np.isfinite(ratios)]
    if not ratios.size:
        raise ValueError("calibration interval cannot identify relative outgoing-amount thresholds")
    thresholds["relative_amount_ratio_q95"] = float(np.quantile(ratios, 0.95))
    thresholds["relative_amount_ratio_q05"] = float(np.quantile(ratios, 0.05))

    calib_in_count = calib_incoming.cumcount().to_numpy(dtype=np.int32)
    calib_in_amount = calib_rows["amount_received_usd"].to_numpy(dtype=np.float64)
    calib_cum_in = calib_incoming["amount_received_usd"].cumsum().to_numpy(dtype=np.float64)
    calib_prior_in_mean = np.divide(
        calib_cum_in - calib_in_amount,
        calib_in_count,
        out=np.full(len(calib_rows), np.nan, dtype=np.float64),
        where=calib_in_count > 0,
    )
    stable_in = calib_in_count >= 3
    in_ratios = calib_in_amount[stable_in] / np.maximum(
        calib_prior_in_mean[stable_in], np.finfo(np.float64).eps
    )
    in_ratios = in_ratios[np.isfinite(in_ratios)]
    if not in_ratios.size:
        raise ValueError("calibration interval cannot identify relative incoming-amount thresholds")
    thresholds["relative_in_amount_ratio_q95"] = float(np.quantile(in_ratios, 0.95))

    out_hour_sizes = calib_rows.groupby(["from_code", "time_index"], sort=False).size().to_numpy()
    in_hour_sizes = calib_rows.groupby(["to_code", "time_index"], sort=False).size().to_numpy()
    thresholds["out_hour_count_q95"] = float(np.quantile(out_hour_sizes, 0.95, method="higher"))
    thresholds["in_hour_count_q95"] = float(np.quantile(in_hour_sizes, 0.95, method="higher"))
    positive_gaps = (
        calib_outgoing["time_index"].diff().dropna().astype("float64")
    )
    positive_gaps = positive_gaps.loc[positive_gaps.gt(0)]
    if positive_gaps.empty:
        raise ValueError("calibration interval cannot identify outgoing interarrival thresholds")
    thresholds["out_interarrival_q95"] = float(positive_gaps.quantile(0.95))
    calib_gap = calib_outgoing["time_index"].diff().astype("float64")
    calib_prev_gap = calib_gap.groupby(calib_rows["from_code"], sort=False).shift()
    cadence_ratio = (calib_gap / calib_prev_gap).loc[calib_gap.gt(0) & calib_prev_gap.gt(0)]
    cadence_ratio = cadence_ratio.loc[np.isfinite(cadence_ratio)]
    if cadence_ratio.empty:
        raise ValueError("calibration interval cannot identify outgoing cadence thresholds")
    thresholds["out_cadence_ratio_q05"] = float(cadence_ratio.quantile(0.05))
    out_pair_gap = calib_rows.groupby(["from_code", "to_code"], sort=False)["time_index"].diff()
    in_pair_gap = calib_rows.groupby(["to_code", "from_code"], sort=False)["time_index"].diff()
    out_pair_gap = out_pair_gap.loc[out_pair_gap.gt(0)]
    in_pair_gap = in_pair_gap.loc[in_pair_gap.gt(0)]
    if out_pair_gap.empty or in_pair_gap.empty:
        raise ValueError("calibration interval cannot identify counterparty-return thresholds")
    thresholds["out_counterparty_return_gap_q95"] = float(out_pair_gap.quantile(0.95))
    thresholds["in_counterparty_return_gap_q95"] = float(in_pair_gap.quantile(0.95))
    thresholds["calibration_cutoff_time"] = float(cutoff)
    return thresholds


def infer_currency_usd_rates(
    df: pd.DataFrame,
    calibration_frac: float,
) -> dict[str, float]:
    """Infer USD-per-unit rates from paired transaction amounts without labels.

    Each transaction supplies
    log r_pay - log r_recv = log amount_recv - log amount_paid.
    Pairwise means and counts are exact sufficient statistics for the
    transaction-level least-squares normal equations.  The US dollar is fixed
    to one and only currencies graph-connected to USD are identified.
    """
    max_time = int(df["time_index"].max())
    cutoff = int(np.floor(max_time * calibration_frac))
    calibration = df.loc[df["time_index"].le(cutoff)].copy()
    calibration = calibration.loc[
        calibration["amount_paid"].gt(0) & calibration["amount_received"].gt(0)
    ].copy()
    calibration["pay_currency"] = calibration["payment_currency"].astype(str)
    calibration["recv_currency"] = calibration["receiving_currency"].astype(str)
    calibration["log_ratio"] = np.log(
        calibration["amount_received"].to_numpy(dtype=np.float64)
        / calibration["amount_paid"].to_numpy(dtype=np.float64)
    )
    pairs = (
        calibration.groupby(["pay_currency", "recv_currency"], as_index=False)
        .agg(log_ratio=("log_ratio", "mean"), count=("log_ratio", "size"))
    )
    anchor = "US Dollar"
    graph: dict[str, set[str]] = {}
    for pay, recv in pairs[["pay_currency", "recv_currency"]].itertuples(index=False):
        graph.setdefault(str(pay), set()).add(str(recv))
        graph.setdefault(str(recv), set()).add(str(pay))
    connected = {anchor}
    frontier = [anchor]
    while frontier:
        current = frontier.pop()
        for neighbor in graph.get(current, ()):
            if neighbor not in connected:
                connected.add(neighbor)
                frontier.append(neighbor)
    currencies = sorted(connected - {anchor})
    index = {currency: idx for idx, currency in enumerate(currencies)}
    rows: list[np.ndarray] = []
    values: list[float] = []
    for pay, recv, log_ratio, count in pairs.itertuples(index=False):
        pay = str(pay)
        recv = str(recv)
        if pay not in connected or recv not in connected or pay == recv:
            continue
        weight = math.sqrt(float(count))
        row = np.zeros(len(currencies), dtype=np.float64)
        if pay != anchor:
            row[index[pay]] += weight
        if recv != anchor:
            row[index[recv]] -= weight
        rows.append(row)
        values.append(weight * float(log_ratio))
    if currencies and not rows:
        raise ValueError("no calibration equations connect transaction currencies to USD")
    beta = (
        np.linalg.lstsq(np.stack(rows), np.asarray(values), rcond=None)[0]
        if currencies
        else np.zeros(0, dtype=np.float64)
    )
    rates = {anchor: 1.0}
    rates.update({currency: float(np.exp(beta[idx])) for currency, idx in index.items()})
    return rates


def add_usd_amounts(df: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    """Attach a common, outcome-blind financial scale to every transaction."""
    paid_rate = df["payment_currency"].astype(str).map(rates).to_numpy(dtype=np.float64)
    received_rate = df["receiving_currency"].astype(str).map(rates).to_numpy(dtype=np.float64)
    paid = df["amount_paid"].to_numpy(dtype=np.float64) * paid_rate
    received = df["amount_received"].to_numpy(dtype=np.float64) * received_rate
    valid = np.isfinite(paid) & np.isfinite(received) & (paid > 0) & (received > 0)
    if not np.all(valid):
        missing = sorted(
            set(df.loc[~np.isfinite(paid_rate), "payment_currency"].astype(str))
            | set(df.loc[~np.isfinite(received_rate), "receiving_currency"].astype(str))
        )
        raise ValueError(f"transaction currencies are not identifiable relative to USD: {missing}")
    df["amount_paid_usd"] = paid
    df["amount_received_usd"] = received
    # The geometric mean is symmetric between the paid and received records
    # and remains positive in the presence of spreads or rounding.
    df["transaction_amount_usd"] = np.sqrt(paid * received)
    return df


def compute_source_last_prior_incoming(df: pd.DataFrame) -> np.ndarray:
    """Last incoming grid time strictly before each outgoing row."""
    incoming = (
        df[["to_code", "time_index"]]
        .drop_duplicates()
        .rename(columns={"to_code": "account_code", "time_index": "last_in_time"})
        .sort_values("last_in_time", kind="mergesort")
    )
    queries = (
        df[["row_id", "from_code", "time_index"]]
        .rename(columns={"from_code": "account_code"})
        .sort_values("time_index", kind="mergesort")
    )
    merged = pd.merge_asof(
        queries,
        incoming,
        left_on="time_index",
        right_on="last_in_time",
        by="account_code",
        direction="backward",
        allow_exact_matches=False,
    )
    out = np.full(len(df), np.nan, dtype=np.float64)
    out[merged["row_id"].to_numpy(dtype=np.int64)] = merged["last_in_time"].to_numpy(dtype=np.float64)
    return out


def add_frame(
    frames: list[pd.DataFrame],
    account_codes: np.ndarray,
    times: np.ndarray,
    mask: np.ndarray,
    token: str,
) -> None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    frames.append(
        pd.DataFrame(
            {
                "account_code": account_codes[idx].astype("int32", copy=False),
                "time_index": times[idx].astype("int32", copy=False),
                "token": token,
            }
        )
    )


def compute_cycle_return(df: pd.DataFrame, window_bins: int) -> np.ndarray:
    if window_bins < 1:
        raise ValueError("cycle-return window must contain at least one time bin")
    cur = df[["row_id", "from_code", "to_code", "time_index"]].copy()
    rev = pd.DataFrame(
        {
            "from_code": df["to_code"].to_numpy(dtype=np.int32, copy=False),
            "to_code": df["from_code"].to_numpy(dtype=np.int32, copy=False),
            "rev_time": df["time_index"].to_numpy(dtype=np.int32, copy=False),
        }
    )
    cur = cur.sort_values("time_index", kind="mergesort")
    rev = rev.sort_values("rev_time", kind="mergesort")
    out = np.zeros(len(df), dtype=bool)
    merged = pd.merge_asof(
        cur,
        rev,
        left_on="time_index",
        right_on="rev_time",
        by=["from_code", "to_code"],
        direction="backward",
        allow_exact_matches=False,
    )
    has_rev = merged["rev_time"].notna()
    within = has_rev & (merged["time_index"] - merged["rev_time"]).le(window_bins)
    out[merged.loc[within, "row_id"].to_numpy(dtype=np.int64)] = True
    return out


def compute_source_prior_incoming(
    df: pd.DataFrame,
    *,
    amount_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amount_column = amount_column or (
        "amount_received_usd" if "amount_received_usd" in df else "amount_received"
    )
    n = len(df)
    row_id = df["row_id"].to_numpy(dtype=np.int64, copy=False)
    first_sender_to_receiver_day = (
        df.groupby(["to_code", "day_index", "from_code"], sort=False)
        .cumcount()
        .eq(0)
        .to_numpy(dtype=np.int8)
    )
    order = np.concatenate([row_id * 2, row_id * 2 + 1])
    combined = pd.DataFrame(
        {
            "order": order,
            "row_id": np.concatenate([row_id, row_id]),
            "account_code": np.concatenate(
                [
                    df["from_code"].to_numpy(dtype=np.int32, copy=False),
                    df["to_code"].to_numpy(dtype=np.int32, copy=False),
                ]
            ),
            "day_index": np.concatenate(
                [
                    df["day_index"].to_numpy(dtype=np.int16, copy=False),
                    df["day_index"].to_numpy(dtype=np.int16, copy=False),
                ]
            ),
            "is_in": np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)]),
            "amount_in": np.concatenate(
                [
                    np.zeros(n, dtype=np.float64),
                    df[amount_column].to_numpy(dtype=np.float64, copy=False),
                ]
            ),
            "unique_sender_increment": np.concatenate(
                [np.zeros(n, dtype=np.int8), first_sender_to_receiver_day]
            ),
        }
    )
    combined = combined.sort_values(["account_code", "day_index", "order"], kind="mergesort")
    group_cols = [combined["account_code"], combined["day_index"]]
    combined["cum_in_count"] = combined["is_in"].groupby(group_cols, sort=False).cumsum()
    combined["cum_in_amount"] = combined["amount_in"].groupby(group_cols, sort=False).cumsum()
    combined["cum_unique_in_senders"] = combined["unique_sender_increment"].groupby(
        group_cols, sort=False
    ).cumsum()
    source_rows = combined.loc[
        combined["is_in"].eq(0),
        ["row_id", "cum_in_count", "cum_in_amount", "cum_unique_in_senders"],
    ]
    source_rows = source_rows.sort_values("row_id", kind="mergesort")
    prior_count = source_rows["cum_in_count"].to_numpy(dtype=np.int32, copy=False)
    prior_amount = source_rows["cum_in_amount"].to_numpy(dtype=np.float64, copy=False)
    prior_unique_senders = source_rows["cum_unique_in_senders"].to_numpy(
        dtype=np.int32, copy=False
    )
    return prior_count, prior_amount, prior_unique_senders


def build_predicate_events(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    include_laundering_transaction_predicates: bool,
    cycle_return_window_bins: int,
    selected_predicate_names: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    n = len(df)
    selected_names = (
        None
        if selected_predicate_names is None
        else {str(name) for name in selected_predicate_names}
    )

    def wanted(token: str) -> bool:
        return selected_names is None or token in selected_names

    fmt = df["payment_format"].astype("string")
    paid_currency = df["payment_currency"].astype("string")
    recv_currency = df["receiving_currency"].astype("string")
    amount = df["amount_paid_usd"].to_numpy(dtype=np.float64, copy=False)
    time_index = df["time_index"].to_numpy(dtype=np.int32, copy=False)
    from_code = df["from_code"].to_numpy(dtype=np.int32, copy=False)
    to_code = df["to_code"].to_numpy(dtype=np.int32, copy=False)
    day_index = df["day_index"].to_numpy(dtype=np.int32, copy=False)
    laundering = df["is_laundering"].to_numpy(dtype=np.int8, copy=False).astype(bool)
    # Covariate construction must be outcome-blind.  Observable attributes of a
    # laundering-labelled transaction are available once that transaction has
    # happened, just like attributes of every other transaction.  The TPP uses
    # strictly positive response lags, so retaining the row cannot explain its
    # own contemporaneous target.  Label-masking is kept only for an explicit
    # legacy ablation.
    usable_tx = np.ones(n, dtype=bool) if include_laundering_transaction_predicates else ~laundering
    non_self = from_code != to_code

    work = df[
        [
            "from_code",
            "to_code",
            "day_index",
            "time_index",
            "amount_paid_usd",
            "amount_received_usd",
        ]
    ].copy()
    outgoing = work.groupby("from_code", sort=False)
    incoming = work.groupby("to_code", sort=False)
    outgoing_pair = work.groupby(["from_code", "to_code"], sort=False)
    incoming_pair = work.groupby(["to_code", "from_code"], sort=False)

    # Positions are causal.  transform("size") leaks later transactions in the
    # same grid cell into the first transaction and must not be used for onset
    # predicates.
    out_hour_position = (
        work.groupby(["from_code", "time_index"], sort=False).cumcount().to_numpy(dtype=np.int32) + 1
    )
    in_hour_position = (
        work.groupby(["to_code", "time_index"], sort=False).cumcount().to_numpy(dtype=np.int32) + 1
    )
    prev_out_count = outgoing.cumcount().to_numpy(dtype=np.int32)
    prev_in_count = incoming.cumcount().to_numpy(dtype=np.int32)
    cum_out_sum = outgoing["amount_paid_usd"].cumsum().to_numpy(dtype=np.float64)
    prior_out_mean = np.divide(
        cum_out_sum - amount,
        prev_out_count,
        out=np.zeros(n, dtype=np.float64),
        where=prev_out_count > 0,
    )
    received_amount = df["amount_received_usd"].to_numpy(dtype=np.float64, copy=False)
    cum_in_sum = incoming["amount_received_usd"].cumsum().to_numpy(dtype=np.float64)
    prior_in_mean_all = np.divide(
        cum_in_sum - received_amount,
        prev_in_count,
        out=np.zeros(n, dtype=np.float64),
        where=prev_in_count > 0,
    )

    new_receiver = outgoing_pair.cumcount().eq(0).to_numpy()
    new_sender = incoming_pair.cumcount().eq(0).to_numpy()

    if wanted("pred_many_new_receivers_day"):
        first_receiver_day = (
            work.groupby(["from_code", "day_index", "to_code"], sort=False)
            .cumcount()
            .eq(0)
            .astype("int8")
        )
        unique_receivers_day = first_receiver_day.groupby(
            [work["from_code"], work["day_index"]], sort=False
        ).cumsum().to_numpy(dtype=np.int16)
    else:
        first_receiver_day = pd.Series(np.zeros(n, dtype=np.int8), index=work.index)
        unique_receivers_day = np.zeros(n, dtype=np.int16)

    if wanted("pred_many_unique_senders_day"):
        first_sender_day = (
            work.groupby(["to_code", "day_index", "from_code"], sort=False)
            .cumcount()
            .eq(0)
            .astype("int8")
        )
        unique_senders_day = first_sender_day.groupby(
            [work["to_code"], work["day_index"]], sort=False
        ).cumsum().to_numpy(dtype=np.int16)
    else:
        first_sender_day = pd.Series(np.zeros(n, dtype=np.int8), index=work.index)
        unique_senders_day = np.zeros(n, dtype=np.int16)

    if wanted("pred_out_structured_small_repeats_day"):
        small_out = amount <= thresholds["amount_q60"]
        small_out_count = pd.Series(small_out.astype("int8")).groupby(
            [work["from_code"], work["day_index"]], sort=False
        ).cumsum().to_numpy(dtype=np.int16)
        small_out_amount = pd.Series(np.where(small_out, amount, 0.0)).groupby(
            [work["from_code"], work["day_index"]], sort=False
        ).cumsum().to_numpy(dtype=np.float64)
    else:
        small_out = np.zeros(n, dtype=bool)
        small_out_count = np.zeros(n, dtype=np.int16)
        small_out_amount = np.zeros(n, dtype=np.float64)

    need_prior_incoming = wanted("pred_rapid_in_to_out_day") or wanted(
        "pred_fan_in_then_out_day"
    )
    if need_prior_incoming:
        prior_in_count, prior_in_amount, prior_unique_in_senders = (
            compute_source_prior_incoming(df)
        )
    else:
        prior_in_count = np.zeros(n, dtype=np.int32)
        prior_in_amount = np.zeros(n, dtype=np.float64)
        prior_unique_in_senders = np.zeros(n, dtype=np.int32)
    need_turnaround = need_prior_incoming or wanted("pred_in_to_out_turnaround_starts")
    last_prior_in_time = (
        compute_source_last_prior_incoming(df)
        if need_turnaround
        else np.full(n, np.nan, dtype=np.float64)
    )
    cycle_return = (
        compute_cycle_return(df, window_bins=cycle_return_window_bins)
        if wanted("pred_cycle_return_72h")
        else np.zeros(n, dtype=bool)
    )

    prev_out_time = outgoing["time_index"].shift().to_numpy(dtype=np.float64)
    out_interarrival = time_index.astype(np.float64) - prev_out_time
    previous_out_interarrival = (
        pd.Series(out_interarrival).groupby(work["from_code"], sort=False).shift().to_numpy(dtype=np.float64)
    )
    cadence_ratio = np.divide(
        out_interarrival,
        previous_out_interarrival,
        out=np.full(n, np.nan, dtype=np.float64),
        where=previous_out_interarrival > 0,
    )
    previous_receiver = outgoing["to_code"].shift().to_numpy(dtype=np.float64)
    previous_sender = incoming["from_code"].shift().to_numpy(dtype=np.float64)
    previous_receiver_time = outgoing_pair["time_index"].shift().to_numpy(
        dtype=np.float64
    )
    previous_sender_time = incoming_pair["time_index"].shift().to_numpy(
        dtype=np.float64
    )
    prev_currency = df.groupby("from_code", sort=False)["payment_currency"].shift()
    prev_format = df.groupby("from_code", sort=False)["payment_format"].shift()
    prev_to_bank = df.groupby("from_code", sort=False)["to_bank"].shift()
    has_prior_in = np.isfinite(last_prior_in_time)
    turnaround_start = has_prior_in & (np.isnan(prev_out_time) | (prev_out_time <= last_prior_in_time))

    amount_cents = np.rint(df["amount_paid"].to_numpy(dtype=np.float64) * 100.0).astype(np.int64)
    round_amount = (amount >= thresholds["amount_q75"]) & (np.mod(amount_cents, 100_000) == 0)

    frames: list[pd.DataFrame] = []

    def add_source(mask: np.ndarray, token: str) -> None:
        if selected_names is None or token in selected_names:
            add_frame(frames, from_code, time_index, usable_tx & mask, token)

    def add_receiver(mask: np.ndarray, token: str) -> None:
        if selected_names is None or token in selected_names:
            add_frame(frames, to_code, time_index, usable_tx & mask, token)

    add_source(fmt.isin(["Cash", "Cheque"]).to_numpy(), "pred_out_cash_or_cheque")
    add_source(fmt.isin(["Wire", "ACH"]).to_numpy(), "pred_out_wire_or_ach")
    add_source(fmt.eq("Credit Card").to_numpy(), "pred_out_credit_card")
    add_source(
        fmt.eq("Bitcoin").to_numpy() | paid_currency.eq("Bitcoin").to_numpy() | recv_currency.eq("Bitcoin").to_numpy(),
        "pred_out_bitcoin",
    )
    add_source(fmt.eq("Reinvestment").to_numpy(), "pred_out_reinvestment")
    add_source(paid_currency.ne(recv_currency).to_numpy(), "pred_currency_conversion_out")
    add_source(df["from_bank"].to_numpy() != df["to_bank"].to_numpy(), "pred_cross_bank_out")
    add_receiver(df["from_bank"].to_numpy() != df["to_bank"].to_numpy(), "pred_cross_bank_in")
    add_source(amount >= thresholds["amount_q95"], "pred_out_large_q95")
    add_source(amount >= thresholds["amount_q99"], "pred_out_very_large_q99")
    add_source(
        (prev_out_count >= 3)
        & (amount >= thresholds["relative_amount_ratio_q95"] * np.maximum(prior_out_mean, np.finfo(np.float64).eps)),
        "pred_out_amount_spike_rel_mean",
    )
    add_source(round_amount, "pred_out_round_amount")
    structured_candidate = (
        small_out
        & (small_out_count >= 3)
        & (small_out_amount >= thresholds["amount_q75"])
    )
    structured_onset_number = (
        pd.Series(structured_candidate.astype("int8"))
        .groupby([work["from_code"], work["day_index"]], sort=False)
        .cumsum()
        .to_numpy(dtype=np.int16)
        if wanted("pred_out_structured_small_repeats_day")
        else np.zeros(n, dtype=np.int16)
    )
    add_source(
        structured_candidate & (structured_onset_number == 1),
        "pred_out_structured_small_repeats_day",
    )
    add_source(new_receiver & non_self, "pred_new_receiver_out")
    add_source(
        first_receiver_day.to_numpy(dtype=bool) & (unique_receivers_day == 3) & non_self,
        "pred_many_new_receivers_day",
    )
    add_source(out_hour_position == 3, "pred_outgoing_burst_hour")
    add_receiver(new_sender & non_self, "pred_new_sender_in")
    add_receiver(
        first_sender_day.to_numpy(dtype=bool) & (unique_senders_day == 3) & non_self,
        "pred_many_unique_senders_day",
    )
    add_receiver(in_hour_position == 3, "pred_incoming_burst_hour")
    add_source(
        turnaround_start
        & (prior_in_count > 0)
        & (amount >= 0.25 * np.maximum(prior_in_amount, 1.0)),
        "pred_rapid_in_to_out_day",
    )
    add_source(
        turnaround_start
        & (prior_unique_in_senders >= 3)
        & (amount >= 0.10 * np.maximum(prior_in_amount, 1.0)),
        "pred_fan_in_then_out_day",
    )
    add_source(cycle_return & non_self, "pred_cycle_return_72h")
    add_source(from_code == to_code, "pred_self_transfer")

    # Dynamic, non-target-indicating candidates.  Every predicate is a current
    # transition/onset conditioned only on information available beforehand.
    add_source(new_receiver & non_self & (prev_out_count >= 1), "pred_out_receiver_novelty_after_history")
    add_receiver(new_sender & non_self & (prev_in_count >= 1), "pred_in_sender_novelty_after_history")
    add_source(out_hour_position == int(thresholds["out_hour_count_q95"]), "pred_out_burst_hour_starts")
    add_receiver(in_hour_position == int(thresholds["in_hour_count_q95"]), "pred_in_burst_hour_starts")
    add_source(
        (prev_out_count >= 1) & paid_currency.ne(prev_currency).fillna(False).to_numpy(),
        "pred_out_currency_switch_after_history",
    )
    add_source(
        (prev_out_count >= 1) & fmt.ne(prev_format).fillna(False).to_numpy(),
        "pred_out_format_switch_after_history",
    )
    add_source(
        (prev_out_count >= 1)
        & pd.Series(df["to_bank"]).ne(prev_to_bank).fillna(False).to_numpy(),
        "pred_out_bank_route_switch_after_history",
    )
    add_source(
        np.isfinite(out_interarrival)
        & (out_interarrival >= thresholds["out_interarrival_q95"]),
        "pred_out_dormancy_reactivation",
    )
    add_source(turnaround_start, "pred_in_to_out_turnaround_starts")
    add_source(
        (prev_out_count >= 3)
        & (amount <= thresholds["relative_amount_ratio_q05"] * np.maximum(prior_out_mean, np.finfo(np.float64).eps)),
        "pred_out_amount_drop_rel_mean",
    )
    add_receiver(
        (prev_in_count >= 3)
        & (
            received_amount
            >= thresholds["relative_in_amount_ratio_q95"]
            * np.maximum(prior_in_mean_all, np.finfo(np.float64).eps)
        ),
        "pred_in_amount_spike_rel_mean",
    )
    add_source(
        (~new_receiver)
        & np.isfinite(previous_receiver)
        & (previous_receiver != to_code)
        & ((time_index - previous_receiver_time) >= thresholds["out_counterparty_return_gap_q95"]),
        "pred_out_receiver_revisit_after_alternative",
    )
    add_receiver(
        (~new_sender)
        & np.isfinite(previous_sender)
        & (previous_sender != from_code)
        & ((time_index - previous_sender_time) >= thresholds["in_counterparty_return_gap_q95"]),
        "pred_in_sender_revisit_after_alternative",
    )
    add_source(
        (out_interarrival > 0)
        & (previous_out_interarrival > 0)
        & (cadence_ratio <= thresholds["out_cadence_ratio_q05"]),
        "pred_out_cadence_acceleration",
    )

    target_mark = df["transaction_amount_usd"].to_numpy(dtype=np.float64, copy=False)
    invalid_mark = laundering & (~np.isfinite(target_mark) | (target_mark <= 0))
    if np.any(invalid_mark):
        missing = sorted(
            set(df.loc[invalid_mark, "payment_currency"].astype(str))
            | set(df.loc[invalid_mark, "receiving_currency"].astype(str))
        )
        raise ValueError(f"laundering target currencies are not USD-connected: {missing}")
    target = pd.DataFrame(
        {
            "account_code": from_code[laundering].astype("int32", copy=False),
            "time_index": time_index[laundering].astype("int32", copy=False),
            "target_mark": target_mark[laundering].astype("float64", copy=False),
        }
    )
    target = target.groupby(["account_code", "time_index"], as_index=False).agg(
        target_token=("target_mark", "size"),
        target_mark_values=("target_mark", list),
    )
    target["target_token"] = target["target_token"].astype("int32")

    if frames:
        predicate_long = pd.concat(frames, ignore_index=True)
        predicate_long = predicate_long.drop_duplicates(["account_code", "time_index", "token"])
    else:
        predicate_long = pd.DataFrame(columns=["account_code", "time_index", "token"])
    counts = predicate_long["token"].value_counts().to_dict()
    return predicate_long, target, {str(k): int(v) for k, v in counts.items()}


def build_sequence_bounds(df: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    from_bounds = df.groupby("from_code", sort=False)["time_index"].agg(["min", "max"]).reset_index()
    from_bounds = from_bounds.rename(columns={"from_code": "account_code", "min": "from_min", "max": "from_max"})
    to_bounds = df.groupby("to_code", sort=False)["time_index"].agg(["min", "max"]).reset_index()
    to_bounds = to_bounds.rename(columns={"to_code": "account_code", "min": "to_min", "max": "to_max"})
    bounds = accounts.merge(from_bounds, on="account_code", how="left").merge(to_bounds, on="account_code", how="left")
    bounds["start_month"] = bounds[["from_min", "to_min"]].min(axis=1)
    bounds["end_month"] = bounds[["from_max", "to_max"]].max(axis=1)
    bounds = bounds.dropna(subset=["start_month", "end_month"]).copy()
    bounds["start_month"] = bounds["start_month"].astype("int32")
    bounds["end_month"] = bounds["end_month"].astype("int32")
    bounds["sequence_length"] = bounds["end_month"] - bounds["start_month"] + 1
    return bounds[["account_code", "account_key", "sequence_id", "start_month", "end_month", "sequence_length"]]


def calibration_usd_exposure(df: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Fixed pre-analysis financial scale for reliability/effect materiality."""
    pre = df.loc[df["time_index"].le(cutoff)].copy()
    usd = pre["transaction_amount_usd"].to_numpy(dtype=np.float64)
    source = pd.DataFrame({"account_code": pre["from_code"], "value": usd})
    receiver = pd.DataFrame({"account_code": pre["to_code"], "value": usd})
    exposure = pd.concat([source, receiver], ignore_index=True).groupby("account_code", as_index=False)["value"].sum()
    return exposure.rename(columns={"value": "financial_exposure_calibration_usd"})


def make_wide_events(
    predicate_long: pd.DataFrame,
    target: pd.DataFrame,
    seq_bounds: pd.DataFrame,
    predicate_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predicate_long.empty:
        pred_wide = pd.DataFrame(columns=["account_code", "time_index", *predicate_names])
    else:
        pred_wide = (
            predicate_long.assign(value=1)
            .pivot_table(
                index=["account_code", "time_index"],
                columns="token",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )
        for name in predicate_names:
            if name not in pred_wide.columns:
                pred_wide[name] = 0
        pred_wide = pred_wide[["account_code", "time_index", *predicate_names]]
    event_rows = pred_wide.merge(target, on=["account_code", "time_index"], how="outer")
    if "target_token" not in event_rows.columns:
        event_rows["target_token"] = 0
        event_rows["target_mark_values"] = [[] for _ in range(len(event_rows))]
    event_rows["target_token"] = event_rows["target_token"].fillna(0).astype("int32")
    if "target_mark_values" not in event_rows.columns:
        event_rows["target_mark_values"] = [[] for _ in range(len(event_rows))]
    else:
        event_rows["target_mark_values"] = event_rows["target_mark_values"].apply(
            lambda value: value if isinstance(value, list) else []
        )
    for name in predicate_names:
        event_rows[name] = event_rows[name].fillna(0).astype("int8")
    has_any_pred = event_rows[predicate_names].sum(axis=1).gt(0) if predicate_names else False
    event_rows = event_rows.loc[has_any_pred | event_rows["target_token"].gt(0)].copy()
    event_rows = event_rows.merge(seq_bounds[["account_code", "account_key", "sequence_id"]], on="account_code", how="inner")
    event_rows = event_rows.sort_values(["sequence_id", "time_index"], kind="mergesort")
    event_rows["position"] = event_rows.groupby("sequence_id", sort=False).cumcount().astype("int32")
    event_rows["month_index"] = event_rows["time_index"].astype("int32")
    event_rows["relative_time"] = event_rows["time_index"].astype("int32")
    cols = [
        "sequence_id",
        "account_key",
        "position",
        "month_index",
        "time_index",
        "relative_time",
        "target_token",
        "target_mark_values",
        *predicate_names,
    ]
    event_rows = event_rows[cols]

    seq_event_stats = (
        event_rows.groupby("sequence_id")
        .agg(n_sequence_rows=("position", "size"), n_target_events=("target_token", "sum"))
        .reset_index()
    )
    return event_rows, seq_event_stats


def write_parts(df: pd.DataFrame, out_dir: Path, part_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, start in enumerate(range(0, len(df), part_size)):
        part = df.iloc[start : start + part_size].copy()
        part.to_parquet(out_dir / f"part-{idx:04d}.parquet", index=False)


def write_diagnostics(
    output_root: Path,
    event_rows: pd.DataFrame,
    predicate_names: list[str],
    lag: int = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diag_rows = []
    n_rows = len(event_rows)
    sequence_codes, unique_sequences = pd.factorize(event_rows["sequence_id"], sort=False)
    sequence_codes = sequence_codes.astype(np.int64, copy=False)
    times = event_rows["time_index"].to_numpy(dtype=np.int64, copy=False)
    target_mask = event_rows["target_token"].to_numpy(dtype=np.int32, copy=False) > 0
    if n_rows:
        time_origin = int(np.min(times))
        stride = int(np.max(times) - time_origin + 2)
        row_keys = sequence_codes * stride + times - time_origin
        target_keys = row_keys[target_mask]
        positions = np.searchsorted(target_keys, row_keys, side="right")
        safe = np.minimum(positions, max(0, len(target_keys) - 1))
        future_target = positions < len(target_keys)
        if len(target_keys):
            next_keys = target_keys[safe]
            next_sequence = next_keys // stride
            next_time = next_keys % stride + time_origin
            future_target &= next_sequence == sequence_codes
            future_target &= (next_time - times > 0) & (next_time - times <= lag)
        else:
            future_target[:] = False
    else:
        future_target = np.zeros(0, dtype=bool)
    x = event_rows[predicate_names].to_numpy(dtype=np.uint8)
    pred_counts_array = np.sum(x, axis=0, dtype=np.int64)
    follow_counts_array = np.sum(x[future_target], axis=0, dtype=np.int64)
    baseline_follow = int(np.sum(future_target))

    seq_count = len(unique_sequences)
    for column_index, name in enumerate(predicate_names):
        active = x[:, column_index] == 1
        active_sequence_count = int(np.unique(sequence_codes[active]).size) if np.any(active) else 0
        diag_rows.append(
            {
                "predicate": name,
                "event_rows": int(np.sum(active)),
                "event_row_rate": float(np.mean(active)) if n_rows else 0.0,
                "sequence_count": active_sequence_count,
                "sequence_rate": (
                    float(active_sequence_count / seq_count) if seq_count else 0.0
                ),
                f"target_within_{lag}h_after_predicate": int(follow_counts_array[column_index]),
                f"target_follow_rate_{lag}h": (
                    float(follow_counts_array[column_index] / pred_counts_array[column_index])
                    if pred_counts_array[column_index]
                    else 0.0
                ),
            }
        )
    diag = pd.DataFrame(diag_rows).sort_values(["event_rows", "predicate"], ascending=[False, True])
    baseline_rate = float(baseline_follow / n_rows) if n_rows else 0.0
    diag[f"event_row_baseline_follow_rate_{lag}h"] = baseline_rate
    diag[f"lift_vs_eventrow_baseline_{lag}h"] = np.where(
        baseline_rate > 0.0,
        diag[f"target_follow_rate_{lag}h"].to_numpy(dtype=np.float64) / baseline_rate,
        0.0,
    )
    diag.to_csv(output_root / "metadata" / "predicate_diagnostics.csv", index=False)

    co = np.zeros((len(predicate_names), len(predicate_names)), dtype=np.int64)
    for left in range(0, n_rows, 1_000_000):
        block = x[left : left + 1_000_000].astype(np.int64, copy=False)
        co += block.T @ block
    counts = np.diag(co)
    overlap_rows = []
    for i, a in enumerate(predicate_names):
        for j in range(i + 1, len(predicate_names)):
            b = predicate_names[j]
            inter = int(co[i, j])
            union = int(counts[i] + counts[j] - inter)
            overlap_rows.append(
                {
                    "predicate_a": a,
                    "predicate_b": b,
                    "intersection": inter,
                    "union": union,
                    "jaccard": float(inter / union) if union else 0.0,
                    "contain_a_in_b": float(inter / counts[i]) if counts[i] else 0.0,
                    "contain_b_in_a": float(inter / counts[j]) if counts[j] else 0.0,
                }
            )
    overlap = pd.DataFrame(overlap_rows).sort_values("jaccard", ascending=False)
    overlap.to_csv(output_root / "metadata" / "predicate_overlap.csv", index=False)
    return diag, overlap


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {args.output_root}")
        shutil.rmtree(args.output_root)
    (args.output_root / "metadata").mkdir(parents=True, exist_ok=True)

    df = read_transactions(args.raw_zip, args.max_rows)
    df, start_ts = assign_time_index(df, args.time_unit)
    df, accounts = factorize_accounts(df)
    threshold_frac = args.analysis_start_frac if args.analysis_start_frac is not None else args.calibration_frac
    currency_usd_rates = infer_currency_usd_rates(df, threshold_frac)
    df = add_usd_amounts(df, currency_usd_rates)
    thresholds = calibration_thresholds(df, threshold_frac)
    predicate_names = selected_predicates(args.predicate_tier)

    predicate_long, target, predicate_counts = build_predicate_events(
        df=df,
        thresholds=thresholds,
        include_laundering_transaction_predicates=args.include_laundering_transaction_predicates,
        cycle_return_window_bins=(72 if args.time_unit == "hour" else 36),
        selected_predicate_names=set(predicate_names),
    )

    seq_bounds = build_sequence_bounds(df, accounts)
    analysis_start = None
    if args.analysis_start_frac is not None:
        if not (0.0 < args.analysis_start_frac < 1.0):
            raise ValueError("--analysis-start-frac must be in (0, 1)")
        cutoff = int(np.floor(int(df["time_index"].max()) * args.analysis_start_frac))
        analysis_start = cutoff + 1
        predicate_long = predicate_long.loc[predicate_long["time_index"].ge(analysis_start)].copy()
        target = target.loc[target["time_index"].ge(analysis_start)].copy()
        exposure = calibration_usd_exposure(df, cutoff)
        seq_bounds = seq_bounds.merge(exposure, on="account_code", how="left")
        seq_bounds["financial_exposure_calibration_usd"] = (
            seq_bounds["financial_exposure_calibration_usd"].fillna(0.0).astype("float64")
        )
        seq_bounds["start_month"] = np.maximum(seq_bounds["start_month"], analysis_start).astype("int32")
        seq_bounds["sequence_length"] = seq_bounds["end_month"] - seq_bounds["start_month"] + 1
        # Do not condition the discovery cohort on having pre-window USD flow:
        # AMLworld introduces many accounts after the cutoff, and dropping them
        # creates severe survivorship/established-account selection.  Positive
        # exposure is available for a separate financially weighted robustness
        # cohort; the primary discovery run remains outcome-blind and inclusive.
        seq_bounds = seq_bounds.loc[seq_bounds["sequence_length"].gt(0)].copy()
    event_rows, seq_event_stats = make_wide_events(predicate_long, target, seq_bounds, predicate_names)
    seq_bounds = seq_bounds.merge(seq_event_stats, on="sequence_id", how="left")
    seq_bounds["n_sequence_rows"] = seq_bounds["n_sequence_rows"].fillna(0).astype("int32")
    seq_bounds["n_target_events"] = seq_bounds["n_target_events"].fillna(0).astype("int32")
    seq_bounds["has_target"] = seq_bounds["n_target_events"].gt(0)

    write_parts(event_rows, args.output_root / "sequence_months", args.part_size)
    write_parts(seq_bounds, args.output_root / "sequences", args.part_size)
    token_long = predicate_long.rename(columns={"token": "event_token"})
    if not token_long.empty:
        token_long = token_long.merge(seq_bounds[["account_code", "sequence_id", "account_key"]], on="account_code", how="inner")
        token_long = token_long[["sequence_id", "account_key", "time_index", "event_token"]]
        write_parts(token_long, args.output_root / "sequence_tokens", args.part_size)

    catalog = pd.DataFrame([asdict(p) for p in PREDICATE_CATALOG])
    catalog = catalog.loc[catalog["name"].isin(predicate_names)].copy()
    catalog.to_json(args.output_root / "metadata" / "predicate_catalog.json", orient="records", indent=2)
    diag, overlap = write_diagnostics(args.output_root, event_rows, predicate_names)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_zip": str(args.raw_zip.resolve()),
        "output_root": str(args.output_root.resolve()),
        "dataset": "IBM AMLworld HI-Small transaction table",
        "target_event_name": TARGET_EVENT_NAME,
        "target_process": "recurrent",
        "target_definition": "source account has an outgoing transaction with Is Laundering = 1",
        "target_mark": {
            "column": "target_mark_values",
            "definition": "USD-equivalent outgoing laundering transaction amount",
            "conversion": (
                "outcome-blind count-weighted least squares on calibration-period paired currency amounts; "
                "USD fixed to one"
            ),
            "multiple_same_bin_events": "preserved as one list entry per transaction",
        },
        "leakage_policy": {
            "is_laundering": "used only as target_token",
            "pattern_or_typology_labels": "not used",
            "laundering_transaction_predicates": bool(args.include_laundering_transaction_predicates),
            "amount_and_timing_thresholds": "computed label-free on the pre-analysis/calibration interval",
        },
        "f0_contract": {
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": bool(
                args.include_laundering_transaction_predicates
            ),
            "direct_target_proxy_excluded": True,
            "predicate_history_includes_target_labeled_observations": bool(
                args.include_laundering_transaction_predicates
            ),
            "strict_future_effect_required": True,
        },
        "time_unit": args.time_unit,
        "time_origin": str(start_ts),
        "n_raw_rows": int(len(df)),
        "time_index_min": int(df["time_index"].min()),
        "time_index_max": int(df["time_index"].max()),
        "n_accounts_observed": int(len(accounts)),
        "n_sequences": int(len(seq_bounds)),
        "n_sequence_rows": int(len(event_rows)),
        "n_target_rows": int(event_rows["target_token"].sum()),
        "n_target_sequences": int(seq_bounds["has_target"].sum()),
        "predicate_tier": args.predicate_tier,
        "analysis_start_frac": args.analysis_start_frac,
        "analysis_start_time": analysis_start,
        "predicate_names": predicate_names,
        "n_predicates": int(len(predicate_names)),
        "predicate_counts_long": predicate_counts,
        "amount_thresholds": thresholds,
        "payment_format_counts": {str(k): int(v) for k, v in df["payment_format"].value_counts().items()},
        "currency_counts": {str(k): int(v) for k, v in df["payment_currency"].value_counts().head(20).items()},
        "currency_usd_rates": currency_usd_rates,
        "top_predicate_diagnostics": diag.head(12).to_dict(orient="records"),
        "top_predicate_overlaps": overlap.head(12).to_dict(orient="records"),
        "sources": [
            "Altman et al., NeurIPS 2023 AMLworld: fan-in, fan-out, gather-scatter, scatter-gather, cycle, random, bipartite, stack.",
            "Blanusa et al., ICAIF 2024 Graph Feature Preprocessor: fan-in/fan-out, cycles, scatter-gather graph features.",
            "Han et al., AAAI 2026 D-EMAML: sequential dual-edge motifs for AML edge anomaly detection.",
            "Starnini et al., ECML PKDD 2021 smurf-based AML: structuring/smurfing velocity motifs.",
        ],
    }
    (args.output_root / "metadata" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
