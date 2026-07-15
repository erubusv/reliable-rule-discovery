#!/usr/bin/env python3
"""Build Home Credit client-level TPP sequences from historical tables.

The original Kaggle target is a future binary screening label without an event
timestamp.  This script therefore builds a pure first-event TPP target from the
historical account/payment tables:

    first serious delinquency before the current application date.

The current application row is used only as the t=0 anchor and optional
sequence metadata.  It is never converted into a predicate event.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from certscr.predicate_policy import (
    HOME_CREDIT_BEHAVIORAL_NONPROXY,
    HOME_CREDIT_BEHAVIORAL_NONPROXY_EXPANDED,
)


TARGET_EVENT_NAME = "T_HC_FIRST_SERIOUS_DELINQUENCY"
DEFAULT_RAW_ROOT = Path(
    "data/home_credit_default_risk/kagglehub_cache/competitions/home-credit-default-risk"
)
DEFAULT_OUTPUT_ROOT = Path("data/home_credit_default_risk/processed/tpp_v3")


@dataclass(frozen=True)
class PredicateInfo:
    name: str
    family: str
    tier: str
    raw_table: str
    raw_fields: tuple[str, ...]
    description: str


PREDICATE_CATALOG: list[PredicateInfo] = [
    PredicateInfo(
        "pred_prev_application_refused",
        "previous_application",
        "core",
        "previous_application.csv",
        ("DAYS_DECISION", "NAME_CONTRACT_STATUS"),
        "A previous Home Credit application was refused.",
    ),
    PredicateInfo(
        "pred_prev_application_canceled",
        "previous_application",
        "core",
        "previous_application.csv",
        ("DAYS_DECISION", "NAME_CONTRACT_STATUS"),
        "A previous Home Credit application was canceled or became an unused offer.",
    ),
    PredicateInfo(
        "pred_prev_cash_application",
        "previous_application",
        "core",
        "previous_application.csv",
        ("DAYS_DECISION", "NAME_CONTRACT_TYPE"),
        "A previous application was for a cash loan.",
    ),
    PredicateInfo(
        "pred_prev_consumer_application",
        "previous_application",
        "core",
        "previous_application.csv",
        ("DAYS_DECISION", "NAME_CONTRACT_TYPE"),
        "A previous application was for a consumer loan.",
    ),
    PredicateInfo(
        "pred_prev_revolving_application",
        "previous_application",
        "core",
        "previous_application.csv",
        ("DAYS_DECISION", "NAME_CONTRACT_TYPE"),
        "A previous application was for a revolving loan.",
    ),
    PredicateInfo(
        "pred_prev_credit_exceeds_application_20pct",
        "previous_application",
        "core",
        "previous_application.csv",
        ("DAYS_DECISION", "AMT_APPLICATION", "AMT_CREDIT"),
        "The credited amount exceeded the requested application amount by at least 20%.",
    ),
    PredicateInfo(
        "pred_prev_multi_application_month",
        "previous_application",
        "core",
        "previous_application.csv",
        ("DAYS_DECISION",),
        "At least two previous Home Credit applications occurred in the same month bin.",
    ),
    PredicateInfo(
        "pred_bureau_credit_opened",
        "bureau",
        "core",
        "bureau.csv",
        ("DAYS_CREDIT",),
        "A credit account was opened/reported by an external bureau.",
    ),
    PredicateInfo(
        "pred_bureau_credit_closed",
        "bureau",
        "core",
        "bureau.csv",
        ("DAYS_ENDDATE_FACT",),
        "An external bureau credit account actually closed before the current application.",
    ),
    PredicateInfo(
        "pred_bureau_consumer_credit_opened",
        "bureau",
        "core",
        "bureau.csv",
        ("DAYS_CREDIT", "CREDIT_TYPE"),
        "An external consumer-credit account was opened.",
    ),
    PredicateInfo(
        "pred_bureau_credit_card_opened",
        "bureau",
        "core",
        "bureau.csv",
        ("DAYS_CREDIT", "CREDIT_TYPE"),
        "An external credit-card account was opened.",
    ),
    PredicateInfo(
        "pred_bureau_microloan_opened",
        "bureau",
        "core",
        "bureau.csv",
        ("DAYS_CREDIT", "CREDIT_TYPE"),
        "An external microloan was opened.",
    ),
    PredicateInfo(
        "pred_bureau_large_credit_opened",
        "bureau",
        "core",
        "bureau.csv",
        ("DAYS_CREDIT", "AMT_CREDIT_SUM"),
        "An external credit account opened with amount at or above the dataset 80th percentile.",
    ),
    PredicateInfo(
        "pred_bureau_status_1_starts",
        "bureau_balance",
        "precursor",
        "bureau_balance.csv",
        ("MONTHS_BALANCE", "STATUS"),
        "A bureau monthly status entered code 1, a mild delinquency band.",
    ),
    PredicateInfo(
        "pred_bureau_unknown_status_starts",
        "bureau_balance",
        "core",
        "bureau_balance.csv",
        ("MONTHS_BALANCE", "STATUS"),
        "A bureau monthly status entered X/unknown.",
    ),
    PredicateInfo(
        "pred_bureau_unknown_status_3m_starts",
        "bureau_balance",
        "core",
        "bureau_balance.csv",
        ("MONTHS_BALANCE", "STATUS"),
        "A bureau account started a three-month consecutive X/unknown-status run.",
    ),
    PredicateInfo(
        "pred_bureau_closed_status_starts",
        "bureau_balance",
        "core",
        "bureau_balance.csv",
        ("MONTHS_BALANCE", "STATUS"),
        "A bureau monthly status entered closed.",
    ),
    PredicateInfo(
        "pred_bureau_recovers_to_current",
        "bureau_balance",
        "precursor",
        "bureau_balance.csv",
        ("MONTHS_BALANCE", "STATUS"),
        "A bureau account moved from non-current/missing to current status 0.",
    ),
    PredicateInfo(
        "pred_pos_mild_dpd_starts",
        "pos_cash",
        "precursor",
        "POS_CASH_balance.csv",
        ("MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF"),
        "A previous POS/cash loan entered a mild positive DPD/DPD_DEF below 30 days.",
    ),
    PredicateInfo(
        "pred_pos_contract_completed",
        "pos_cash",
        "core",
        "POS_CASH_balance.csv",
        ("MONTHS_BALANCE", "NAME_CONTRACT_STATUS"),
        "A previous POS/cash loan entered completed status.",
    ),
    PredicateInfo(
        "pred_pos_future_installments_not_decreasing_2m",
        "pos_cash",
        "core",
        "POS_CASH_balance.csv",
        ("MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE"),
        "Remaining POS/cash installments failed to decrease for two consecutive monthly updates.",
    ),
    PredicateInfo(
        "pred_pos_installment_count_increases",
        "pos_cash",
        "core",
        "POS_CASH_balance.csv",
        ("MONTHS_BALANCE", "CNT_INSTALMENT"),
        "The POS/cash total installment count increased versus the previous account month.",
    ),
    PredicateInfo(
        "pred_pos_contract_demand",
        "pos_cash",
        "high_directness",
        "POS_CASH_balance.csv",
        ("MONTHS_BALANCE", "NAME_CONTRACT_STATUS"),
        "A previous POS/cash loan entered demand status.",
    ),
    PredicateInfo(
        "pred_card_mild_dpd_starts",
        "credit_card",
        "precursor",
        "credit_card_balance.csv",
        ("MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF"),
        "A previous credit card entered a mild positive DPD/DPD_DEF below 30 days.",
    ),
    PredicateInfo(
        "pred_card_utilization_cross_80",
        "credit_card",
        "core",
        "credit_card_balance.csv",
        ("MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"),
        "Credit-card utilization crossed upward through 80%.",
    ),
    PredicateInfo(
        "pred_card_utilization_cross_95",
        "credit_card",
        "core",
        "credit_card_balance.csv",
        ("MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"),
        "Credit-card utilization crossed upward through 95%.",
    ),
    PredicateInfo(
        "pred_card_utilization_jump_20pp",
        "credit_card",
        "core",
        "credit_card_balance.csv",
        ("MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"),
        "Credit-card utilization increased by at least 20 percentage points in one account month.",
    ),
    PredicateInfo(
        "pred_card_balance_jump_50pct",
        "credit_card",
        "core",
        "credit_card_balance.csv",
        ("MONTHS_BALANCE", "AMT_BALANCE"),
        "Credit-card balance rose by at least 50% and by a positive amount versus the previous account month.",
    ),
    PredicateInfo(
        "pred_card_payment_shortfall_min_due",
        "credit_card",
        "precursor",
        "credit_card_balance.csv",
        ("MONTHS_BALANCE", "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT"),
        "Current credit-card payment was below the regular minimum installment.",
    ),
    PredicateInfo(
        "pred_card_cash_withdrawal",
        "credit_card",
        "core",
        "credit_card_balance.csv",
        ("MONTHS_BALANCE", "AMT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_ATM_CURRENT"),
        "ATM cash withdrawal activity appeared on the previous credit card.",
    ),
    PredicateInfo(
        "pred_card_limit_cut_20pct",
        "credit_card",
        "core",
        "credit_card_balance.csv",
        ("MONTHS_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"),
        "Credit-card limit dropped by at least 20% versus the previous account month.",
    ),
    PredicateInfo(
        "pred_inst_late_1_15d",
        "installments",
        "precursor",
        "installments_payments.csv",
        ("DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"),
        "An installment was paid 1-15 days after its scheduled due date.",
    ),
    PredicateInfo(
        "pred_inst_late_16_29d",
        "installments",
        "precursor",
        "installments_payments.csv",
        ("DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"),
        "An installment was paid 16-29 days after its scheduled due date.",
    ),
    PredicateInfo(
        "pred_inst_underpaid_5pct",
        "installments",
        "precursor",
        "installments_payments.csv",
        ("AMT_INSTALMENT", "AMT_PAYMENT", "DAYS_ENTRY_PAYMENT"),
        "An installment payment was at least 5% below the scheduled amount.",
    ),
    PredicateInfo(
        "pred_inst_underpaid_50pct",
        "installments",
        "high_directness",
        "installments_payments.csv",
        ("AMT_INSTALMENT", "AMT_PAYMENT", "DAYS_ENTRY_PAYMENT"),
        "An installment payment was less than half of the scheduled amount.",
    ),
    PredicateInfo(
        "pred_inst_paid_early_7d",
        "installments",
        "core",
        "installments_payments.csv",
        ("DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"),
        "An installment was paid at least seven days before its scheduled due date.",
    ),
]


LOW_OVERLAP_PREDICATES = [
    "pred_prev_application_refused",
    "pred_prev_application_canceled",
    "pred_prev_revolving_application",
    "pred_prev_credit_exceeds_application_20pct",
    "pred_prev_multi_application_month",
    "pred_bureau_credit_card_opened",
    "pred_bureau_large_credit_opened",
    "pred_bureau_unknown_status_3m_starts",
    "pred_bureau_closed_status_starts",
    "pred_bureau_recovers_to_current",
    "pred_pos_installment_count_increases",
    "pred_card_utilization_cross_95",
    "pred_card_utilization_jump_20pp",
    "pred_card_balance_jump_50pct",
    "pred_card_cash_withdrawal",
    "pred_card_limit_cut_20pct",
]

# Customer actions or account-use changes that unfold over historical time.
# Lender decisions (refused/canceled/limit cut), reporting artifacts, recovery
# states, and near-target stress thresholds such as utilization >95% are
# deliberately excluded.  This tier is outcome-blind: no target rate is used
# to choose its members.
BEHAVIORAL_NONPROXY_PREDICATES = list(HOME_CREDIT_BEHAVIORAL_NONPROXY)
BEHAVIORAL_NONPROXY_EXPANDED_PREDICATES = list(HOME_CREDIT_BEHAVIORAL_NONPROXY_EXPANDED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Home Credit as TPP event sequences.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include application_test clients as unlabeled anchors. Default: train clients only.",
    )
    parser.add_argument(
        "--current-contract-type",
        choices=["Cash loans", "Revolving loans"],
        default=None,
        help="Optional current application product filter before building sequences.",
    )
    parser.add_argument(
        "--predicate-tier",
        choices=[
            "strict",
            "balanced",
            "all",
            "low_overlap",
            "behavioral_nonproxy",
            "behavioral_nonproxy_expanded",
        ],
        default="balanced",
        help=(
            "strict=core only, balanced=core+precursor, all=core+precursor+high_directness, "
            "low_overlap=hand-pruned low-overlap discovery set, "
            "behavioral_nonproxy=dynamic customer/account-use actions without direct target proxies. "
            "behavioral_nonproxy_expanded=12 dynamic nonproxy actions across five source families. "
            "Target-proxy severe delinquency is never included as a predicate."
        ),
    )
    parser.add_argument(
        "--sparse-events",
        action="store_true",
        help="Write only predicate/target event rows instead of one row per relative month.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["first_truncated", "event_stream"],
        default="first_truncated",
        help=(
            "first_truncated keeps the legacy first-event hazard view; "
            "event_stream keeps target events inside the full historical sequence "
            "without using target as a predicate."
        ),
    )
    parser.add_argument(
        "--financial-mark-contract",
        choices=["none", "installment_shortfall"],
        default="none",
        help=(
            "installment_shortfall uses only scheduled installments with positive unpaid amount by "
            "due+30 as targets and stores that amount as the event mark"
        ),
    )
    parser.add_argument("--max-lookback-months", type=int, default=120)
    parser.add_argument("--min-sequence-months", type=int, default=2)
    parser.add_argument("--part-size", type=int, default=50000)
    parser.add_argument(
        "--max-clients",
        type=int,
        default=None,
        help="Optional deterministic client cap for smoke tests. Default: use all selected application clients.",
    )
    return parser.parse_args()


def selected_predicates(tier: str) -> list[str]:
    if tier in {"low_overlap", "behavioral_nonproxy", "behavioral_nonproxy_expanded"}:
        selected = {
            "low_overlap": LOW_OVERLAP_PREDICATES,
            "behavioral_nonproxy": BEHAVIORAL_NONPROXY_PREDICATES,
            "behavioral_nonproxy_expanded": BEHAVIORAL_NONPROXY_EXPANDED_PREDICATES,
        }[tier]
        catalog_names = {p.name for p in PREDICATE_CATALOG}
        missing = [p for p in selected if p not in catalog_names]
        if missing:
            raise ValueError(f"{tier} predicate list has unknown predicates: {missing}")
        return list(selected)
    allowed = {"core"} if tier == "strict" else {"core", "precursor"}
    if tier == "all":
        allowed.add("high_directness")
    return [p.name for p in PREDICATE_CATALOG if p.tier in allowed]


def require_files(raw_root: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (raw_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing Home Credit raw files under {raw_root}: {missing}")


def read_csv(raw_root: Path, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(raw_root / name, **kwargs)


def rel_month_from_days(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = np.floor_divide(values.to_numpy(dtype=np.float64), 30.0)
    return pd.Series(out, index=series.index).astype("float64")


def rel_month_from_numeric_days(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = np.floor_divide(values.to_numpy(dtype=np.float64), 30.0)
    return pd.Series(out, index=series.index).astype("float64")


def valid_past_month(df: pd.DataFrame, rel_col: str, max_lookback_months: int) -> pd.Series:
    return df[rel_col].notna() & df[rel_col].lt(0) & df[rel_col].ge(-max_lookback_months)


def add_events(
    frames: list[pd.DataFrame],
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    token: str,
    source: str,
    token_type: str = "predicate",
    mark_col: str | None = None,
) -> None:
    if isinstance(mask, np.ndarray):
        mask = pd.Series(mask, index=df.index)
    if not bool(mask.any()):
        return
    columns = ["SK_ID_CURR", "rel_month", *((mark_col,) if mark_col else ())]
    frame = df.loc[mask, columns].copy()
    if mark_col is not None:
        frame = frame.rename(columns={mark_col: "financial_mark"})
    frame["token"] = token
    frame["token_type"] = token_type
    frame["source"] = source
    frames.append(frame)


def add_observed(observed_frames: list[pd.DataFrame], df: pd.DataFrame, mask: pd.Series | np.ndarray) -> None:
    if isinstance(mask, np.ndarray):
        mask = pd.Series(mask, index=df.index)
    if not bool(mask.any()):
        return
    observed_frames.append(df.loc[mask, ["SK_ID_CURR", "rel_month"]].copy())


def entered(current: pd.Series, previous: pd.Series, value: object) -> pd.Series:
    return current.eq(value) & ~previous.eq(value)


def starts_positive_under(series: pd.Series, previous: pd.Series, upper: float) -> pd.Series:
    current = series.gt(0) & series.lt(upper)
    prev_positive = previous.gt(0) & previous.lt(upper)
    return current & ~prev_positive


def to_int_month(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def load_applications(raw_root: Path, include_test: bool) -> pd.DataFrame:
    train = read_csv(
        raw_root,
        "application_train.csv",
        usecols=["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE"],
    )
    train["sample"] = "train"
    frames = [train]
    if include_test:
        test = read_csv(
            raw_root,
            "application_test.csv",
            usecols=["SK_ID_CURR", "NAME_CONTRACT_TYPE"],
        )
        test["TARGET"] = pd.NA
        test["sample"] = "test"
        frames.append(test)
    apps = pd.concat(frames, ignore_index=True)
    apps["SK_ID_CURR"] = apps["SK_ID_CURR"].astype("int64")
    return apps.drop_duplicates("SK_ID_CURR")


def preprocess_previous_application(
    raw_root: Path,
    client_ids: set[int],
    max_lookback_months: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict]:
    usecols = [
        "SK_ID_CURR",
        "DAYS_DECISION",
        "NAME_CONTRACT_STATUS",
        "NAME_CONTRACT_TYPE",
        "AMT_APPLICATION",
        "AMT_CREDIT",
    ]
    df = read_csv(raw_root, "previous_application.csv", usecols=usecols)
    df = df[df["SK_ID_CURR"].isin(client_ids)].copy()
    df["rel_month"] = rel_month_from_days(df["DAYS_DECISION"])
    valid = valid_past_month(df, "rel_month", max_lookback_months)
    df = df.loc[valid].copy()
    df["rel_month"] = df["rel_month"].astype("int16")

    events: list[pd.DataFrame] = []
    observed: list[pd.DataFrame] = []
    add_observed(observed, df, np.ones(len(df), dtype=bool))

    status = df["NAME_CONTRACT_STATUS"].astype("string")
    contract = df["NAME_CONTRACT_TYPE"].astype("string")
    add_events(events, df, status.eq("Refused"), "pred_prev_application_refused", "previous_application")
    add_events(
        events,
        df,
        status.isin(["Canceled", "Unused offer"]),
        "pred_prev_application_canceled",
        "previous_application",
    )
    add_events(events, df, contract.eq("Cash loans"), "pred_prev_cash_application", "previous_application")
    add_events(
        events,
        df,
        contract.eq("Consumer loans"),
        "pred_prev_consumer_application",
        "previous_application",
    )
    add_events(
        events,
        df,
        contract.eq("Revolving loans"),
        "pred_prev_revolving_application",
        "previous_application",
    )

    app_amt = pd.to_numeric(df["AMT_APPLICATION"], errors="coerce")
    credit_amt = pd.to_numeric(df["AMT_CREDIT"], errors="coerce")
    add_events(
        events,
        df,
        app_amt.gt(0) & credit_amt.gt(app_amt * 1.2),
        "pred_prev_credit_exceeds_application_20pct",
        "previous_application",
    )

    counts = df.groupby(["SK_ID_CURR", "rel_month"], observed=True).size().rename("n").reset_index()
    multi = counts.loc[counts["n"].ge(2), ["SK_ID_CURR", "rel_month"]].copy()
    if not multi.empty:
        multi["token"] = "pred_prev_multi_application_month"
        multi["token_type"] = "predicate"
        multi["source"] = "previous_application"
        events.append(multi)

    meta = {
        "rows_after_client_filter": int(len(df)),
        "valid_month_rows": int(len(df)),
        "status_counts": status.value_counts(dropna=False).head(20).to_dict(),
    }
    return events, observed, meta


def preprocess_bureau(
    raw_root: Path,
    client_ids: set[int],
    max_lookback_months: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame], dict]:
    usecols = [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "DAYS_CREDIT",
        "DAYS_ENDDATE_FACT",
        "CREDIT_TYPE",
        "AMT_CREDIT_SUM",
    ]
    bureau = read_csv(raw_root, "bureau.csv", usecols=usecols)
    bureau = bureau[bureau["SK_ID_CURR"].isin(client_ids)].copy()
    bureau["SK_ID_CURR"] = bureau["SK_ID_CURR"].astype("int64")
    bureau["SK_ID_BUREAU"] = bureau["SK_ID_BUREAU"].astype("int64")

    amount = pd.to_numeric(bureau["AMT_CREDIT_SUM"], errors="coerce")
    large_credit_threshold = float(amount.dropna().quantile(0.80)) if amount.notna().any() else float("nan")

    events: list[pd.DataFrame] = []
    observed: list[pd.DataFrame] = []

    opened = bureau[["SK_ID_CURR", "DAYS_CREDIT", "CREDIT_TYPE", "AMT_CREDIT_SUM"]].copy()
    opened["rel_month"] = rel_month_from_days(opened["DAYS_CREDIT"])
    valid_open = valid_past_month(opened, "rel_month", max_lookback_months)
    opened = opened.loc[valid_open].copy()
    opened["rel_month"] = opened["rel_month"].astype("int16")
    add_observed(observed, opened, np.ones(len(opened), dtype=bool))
    add_events(events, opened, np.ones(len(opened), dtype=bool), "pred_bureau_credit_opened", "bureau")

    credit_type = opened["CREDIT_TYPE"].astype("string")
    add_events(
        events,
        opened,
        credit_type.eq("Consumer credit"),
        "pred_bureau_consumer_credit_opened",
        "bureau",
    )
    add_events(events, opened, credit_type.eq("Credit card"), "pred_bureau_credit_card_opened", "bureau")
    add_events(events, opened, credit_type.eq("Microloan"), "pred_bureau_microloan_opened", "bureau")
    add_events(
        events,
        opened,
        pd.to_numeric(opened["AMT_CREDIT_SUM"], errors="coerce").ge(large_credit_threshold),
        "pred_bureau_large_credit_opened",
        "bureau",
    )

    closed = bureau[["SK_ID_CURR", "DAYS_ENDDATE_FACT"]].copy()
    closed["rel_month"] = rel_month_from_days(closed["DAYS_ENDDATE_FACT"])
    valid_closed = valid_past_month(closed, "rel_month", max_lookback_months)
    closed = closed.loc[valid_closed].copy()
    closed["rel_month"] = closed["rel_month"].astype("int16")
    add_observed(observed, closed, np.ones(len(closed), dtype=bool))
    add_events(events, closed, np.ones(len(closed), dtype=bool), "pred_bureau_credit_closed", "bureau")

    mapping = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates("SK_ID_BUREAU")
    meta = {
        "rows_after_client_filter": int(len(bureau)),
        "large_credit_opened_threshold_p80": large_credit_threshold,
        "credit_type_counts": bureau["CREDIT_TYPE"].astype("string").value_counts(dropna=False).to_dict(),
    }
    return mapping, events, observed, meta


def preprocess_bureau_balance(
    raw_root: Path,
    bureau_map: pd.DataFrame,
    max_lookback_months: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], dict]:
    df = read_csv(raw_root, "bureau_balance.csv", usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"])
    df = df.merge(bureau_map, on="SK_ID_BUREAU", how="inner")
    df["rel_month"] = to_int_month(df["MONTHS_BALANCE"])
    valid = valid_past_month(df, "rel_month", max_lookback_months)
    df = df.loc[valid].copy()
    df["rel_month"] = df["rel_month"].astype("int16")
    df = df.sort_values(["SK_ID_BUREAU", "rel_month"], kind="mergesort").reset_index(drop=True)

    events: list[pd.DataFrame] = []
    observed: list[pd.DataFrame] = []
    target_events: list[pd.DataFrame] = []
    add_observed(observed, df, np.ones(len(df), dtype=bool))

    status = df["STATUS"].astype("string")
    prev_status = status.shift(1).where(df["SK_ID_BUREAU"].eq(df["SK_ID_BUREAU"].shift(1)))
    add_events(events, df, entered(status, prev_status, "1"), "pred_bureau_status_1_starts", "bureau_balance")
    add_events(events, df, entered(status, prev_status, "X"), "pred_bureau_unknown_status_starts", "bureau_balance")
    add_events(events, df, entered(status, prev_status, "C"), "pred_bureau_closed_status_starts", "bureau_balance")
    add_events(
        events,
        df,
        status.eq("0") & prev_status.isin(["1", "X"]),
        "pred_bureau_recovers_to_current",
        "bureau_balance",
    )

    prev1 = status.shift(1).where(
        df["SK_ID_BUREAU"].eq(df["SK_ID_BUREAU"].shift(1))
        & df["rel_month"].eq(df["rel_month"].shift(1) + 1)
    )
    prev2 = status.shift(2).where(
        df["SK_ID_BUREAU"].eq(df["SK_ID_BUREAU"].shift(2))
        & df["rel_month"].eq(df["rel_month"].shift(2) + 2)
    )
    prev3 = status.shift(3).where(
        df["SK_ID_BUREAU"].eq(df["SK_ID_BUREAU"].shift(3))
        & df["rel_month"].eq(df["rel_month"].shift(3) + 3)
    )
    add_events(
        events,
        df,
        status.eq("X") & prev1.eq("X") & prev2.eq("X") & ~prev3.eq("X"),
        "pred_bureau_unknown_status_3m_starts",
        "bureau_balance",
    )

    serious = status.isin(["2", "3", "4", "5"])
    same_bureau = df["SK_ID_BUREAU"].eq(df["SK_ID_BUREAU"].shift(1))
    consecutive_bureau = same_bureau & df["rel_month"].eq(df["rel_month"].shift(1) + 1)
    prev_serious = (serious.shift(1, fill_value=False) & consecutive_bureau).astype(bool)
    serious_start = serious & ~prev_serious
    add_events(
        target_events,
        df,
        serious_start,
        "target_bureau_status_2plus",
        "bureau_balance",
        token_type="target_candidate",
    )

    meta = {
        "rows_after_map_and_time_filter": int(len(df)),
        "status_counts": status.value_counts(dropna=False).to_dict(),
        "serious_target_rows": int(serious.sum()),
        "serious_target_start_rows": int(serious_start.sum()),
    }
    return events, observed, target_events, meta


def preprocess_pos_cash(
    raw_root: Path,
    client_ids: set[int],
    max_lookback_months: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], dict]:
    usecols = [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "NAME_CONTRACT_STATUS",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    df = read_csv(raw_root, "POS_CASH_balance.csv", usecols=usecols)
    df = df[df["SK_ID_CURR"].isin(client_ids)].copy()
    df["rel_month"] = to_int_month(df["MONTHS_BALANCE"])
    valid = valid_past_month(df, "rel_month", max_lookback_months)
    df = df.loc[valid].copy()
    df["rel_month"] = df["rel_month"].astype("int16")
    df = df.sort_values(["SK_ID_PREV", "rel_month"], kind="mergesort").reset_index(drop=True)

    events: list[pd.DataFrame] = []
    observed: list[pd.DataFrame] = []
    target_events: list[pd.DataFrame] = []
    add_observed(observed, df, np.ones(len(df), dtype=bool))

    status = df["NAME_CONTRACT_STATUS"].astype("string")
    prev_status = status.shift(1).where(df["SK_ID_PREV"].eq(df["SK_ID_PREV"].shift(1)))
    add_events(events, df, entered(status, prev_status, "Completed"), "pred_pos_contract_completed", "pos_cash")
    add_events(events, df, entered(status, prev_status, "Demand"), "pred_pos_contract_demand", "pos_cash")

    dpd = pd.to_numeric(df["SK_DPD"], errors="coerce").fillna(0)
    dpd_def = pd.to_numeric(df["SK_DPD_DEF"], errors="coerce").fillna(0)
    mild_dpd = ((dpd.gt(0) & dpd.lt(30)) | (dpd_def.gt(0) & dpd_def.lt(30))).astype("int8")
    prev_mild_dpd = mild_dpd.shift(1).where(df["SK_ID_PREV"].eq(df["SK_ID_PREV"].shift(1))).fillna(0)
    add_events(
        events,
        df,
        mild_dpd.eq(1) & ~prev_mild_dpd.eq(1),
        "pred_pos_mild_dpd_starts",
        "pos_cash",
    )

    cnt = pd.to_numeric(df["CNT_INSTALMENT"], errors="coerce")
    fut = pd.to_numeric(df["CNT_INSTALMENT_FUTURE"], errors="coerce")
    same_account = df["SK_ID_PREV"].eq(df["SK_ID_PREV"].shift(1))
    consecutive = same_account & df["rel_month"].eq(df["rel_month"].shift(1) + 1)
    prev_cnt = cnt.shift(1).where(consecutive)
    prev_fut = fut.shift(1).where(consecutive)
    prev2_fut = fut.shift(2).where(
        df["SK_ID_PREV"].eq(df["SK_ID_PREV"].shift(2))
        & df["rel_month"].eq(df["rel_month"].shift(2) + 2)
    )
    prev_not_decreasing = fut.ge(prev_fut) & fut.gt(0) & prev_fut.gt(0)
    prev_prev_not_decreasing = prev_fut.ge(prev2_fut) & prev_fut.gt(0) & prev2_fut.gt(0)
    add_events(
        events,
        df,
        prev_not_decreasing & prev_prev_not_decreasing,
        "pred_pos_future_installments_not_decreasing_2m",
        "pos_cash",
    )
    add_events(
        events,
        df,
        cnt.gt(prev_cnt) & prev_cnt.gt(0),
        "pred_pos_installment_count_increases",
        "pos_cash",
    )

    serious = dpd.ge(30) | dpd_def.ge(30)
    prev_serious = (serious.shift(1, fill_value=False) & consecutive).astype(bool)
    serious_start = serious & ~prev_serious
    add_events(
        target_events,
        df,
        serious_start,
        "target_pos_dpd_30plus",
        "pos_cash",
        token_type="target_candidate",
    )

    meta = {
        "rows_after_client_and_time_filter": int(len(df)),
        "status_counts": status.value_counts(dropna=False).to_dict(),
        "dpd_30plus_rows": int(serious.sum()),
        "dpd_30plus_start_rows": int(serious_start.sum()),
    }
    return events, observed, target_events, meta


def preprocess_credit_card(
    raw_root: Path,
    client_ids: set[int],
    max_lookback_months: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], dict]:
    usecols = [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    df = read_csv(raw_root, "credit_card_balance.csv", usecols=usecols)
    df = df[df["SK_ID_CURR"].isin(client_ids)].copy()
    df["rel_month"] = to_int_month(df["MONTHS_BALANCE"])
    valid = valid_past_month(df, "rel_month", max_lookback_months)
    df = df.loc[valid].copy()
    df["rel_month"] = df["rel_month"].astype("int16")
    df = df.sort_values(["SK_ID_PREV", "rel_month"], kind="mergesort").reset_index(drop=True)

    events: list[pd.DataFrame] = []
    observed: list[pd.DataFrame] = []
    target_events: list[pd.DataFrame] = []
    add_observed(observed, df, np.ones(len(df), dtype=bool))

    bal = pd.to_numeric(df["AMT_BALANCE"], errors="coerce")
    limit = pd.to_numeric(df["AMT_CREDIT_LIMIT_ACTUAL"], errors="coerce")
    util = (bal / limit.where(limit.gt(0))).replace([np.inf, -np.inf], np.nan)

    same_account = df["SK_ID_PREV"].eq(df["SK_ID_PREV"].shift(1))
    consecutive = same_account & df["rel_month"].eq(df["rel_month"].shift(1) + 1)
    prev_util = util.shift(1).where(consecutive)
    prev_bal = bal.shift(1).where(consecutive)
    prev_limit = limit.shift(1).where(consecutive)

    add_events(
        events,
        df,
        util.ge(0.80) & (prev_util.lt(0.80) | prev_util.isna()),
        "pred_card_utilization_cross_80",
        "credit_card",
    )
    add_events(
        events,
        df,
        util.ge(0.95) & (prev_util.lt(0.95) | prev_util.isna()),
        "pred_card_utilization_cross_95",
        "credit_card",
    )
    add_events(
        events,
        df,
        util.sub(prev_util).ge(0.20),
        "pred_card_utilization_jump_20pp",
        "credit_card",
    )
    add_events(
        events,
        df,
        bal.gt(0) & prev_bal.gt(0) & bal.ge(prev_bal * 1.5),
        "pred_card_balance_jump_50pct",
        "credit_card",
    )

    min_due = pd.to_numeric(df["AMT_INST_MIN_REGULARITY"], errors="coerce")
    pay_current = pd.to_numeric(df["AMT_PAYMENT_CURRENT"], errors="coerce").fillna(0)
    add_events(
        events,
        df,
        min_due.gt(0) & pay_current.lt(min_due),
        "pred_card_payment_shortfall_min_due",
        "credit_card",
    )

    atm = pd.to_numeric(df["AMT_DRAWINGS_ATM_CURRENT"], errors="coerce").fillna(0)
    add_events(events, df, atm.gt(0), "pred_card_cash_withdrawal", "credit_card")
    add_events(
        events,
        df,
        prev_limit.gt(0) & limit.le(prev_limit * 0.8),
        "pred_card_limit_cut_20pct",
        "credit_card",
    )

    dpd = pd.to_numeric(df["SK_DPD"], errors="coerce").fillna(0)
    dpd_def = pd.to_numeric(df["SK_DPD_DEF"], errors="coerce").fillna(0)
    mild_dpd = ((dpd.gt(0) & dpd.lt(30)) | (dpd_def.gt(0) & dpd_def.lt(30))).astype("int8")
    prev_mild_dpd = mild_dpd.shift(1).where(same_account).fillna(0)
    add_events(
        events,
        df,
        mild_dpd.eq(1) & ~prev_mild_dpd.eq(1),
        "pred_card_mild_dpd_starts",
        "credit_card",
    )

    serious = dpd.ge(30) | dpd_def.ge(30)
    prev_serious = (serious.shift(1, fill_value=False) & consecutive).astype(bool)
    serious_start = serious & ~prev_serious
    add_events(
        target_events,
        df,
        serious_start,
        "target_card_dpd_30plus",
        "credit_card",
        token_type="target_candidate",
    )

    meta = {
        "rows_after_client_and_time_filter": int(len(df)),
        "dpd_30plus_rows": int(serious.sum()),
        "dpd_30plus_start_rows": int(serious_start.sum()),
        "utilization_nonnull_rows": int(util.notna().sum()),
    }
    return events, observed, target_events, meta


def preprocess_installments(
    raw_root: Path,
    client_ids: set[int],
    max_lookback_months: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], dict]:
    usecols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    df = read_csv(raw_root, "installments_payments.csv", usecols=usecols)
    df = df[df["SK_ID_CURR"].isin(client_ids)].copy()
    for col in ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["due_month"] = rel_month_from_days(df["DAYS_INSTALMENT"])
    valid = valid_past_month(df, "due_month", max_lookback_months)
    df = df.loc[valid].copy()
    df["due_month"] = df["due_month"].astype("int16")

    events: list[pd.DataFrame] = []
    observed: list[pd.DataFrame] = []
    target_events: list[pd.DataFrame] = []

    observed_due = df[["SK_ID_CURR", "due_month"]].rename(columns={"due_month": "rel_month"})
    add_observed(observed, observed_due, np.ones(len(observed_due), dtype=bool))

    keys = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
    ]
    base = (
        df.groupby(keys, observed=True)
        .agg(
            due_amt=("AMT_INSTALMENT", "max"),
            due_month=("due_month", "first"),
            n_payment_rows=("AMT_PAYMENT", "size"),
            total_paid=("AMT_PAYMENT", "sum"),
            first_entry=("DAYS_ENTRY_PAYMENT", "min"),
            last_entry=("DAYS_ENTRY_PAYMENT", "max"),
        )
        .reset_index()
    )
    base["due_plus_30"] = base["DAYS_INSTALMENT"] + 30
    base["target_month"] = rel_month_from_numeric_days(base["due_plus_30"])
    base = base[base["target_month"].notna() & base["target_month"].lt(0)].copy()
    base["target_month"] = base["target_month"].astype("int16")
    base["due_month"] = base["due_month"].astype("int16")

    paid_by_due = (
        df.loc[df["DAYS_ENTRY_PAYMENT"].notna() & df["DAYS_ENTRY_PAYMENT"].le(df["DAYS_INSTALMENT"])]
        .groupby(keys, observed=True)["AMT_PAYMENT"]
        .sum()
        .rename("paid_by_due")
        .reset_index()
    )
    paid_by_30 = (
        # DAYS_INSTALMENT is already part of the installment key, so joining
        # the full payment table back to `base` merely to recover due+30
        # duplicated the largest intermediate.  The row-local expression is
        # exactly the same cutoff.
        df.loc[
            df["DAYS_ENTRY_PAYMENT"].notna()
            & df["DAYS_ENTRY_PAYMENT"].le(df["DAYS_INSTALMENT"] + 30)
        ]
        .groupby(keys, observed=True)["AMT_PAYMENT"]
        .sum()
        .rename("paid_by_30")
        .reset_index()
    )
    base = base.merge(paid_by_due, on=keys, how="left")
    base = base.merge(paid_by_30, on=keys, how="left")
    base["paid_by_due"] = base["paid_by_due"].fillna(0)
    base["paid_by_30"] = base["paid_by_30"].fillna(0)

    pay = df.merge(base[keys + ["due_amt"]], on=keys, how="inner")
    pay = pay[pay["DAYS_ENTRY_PAYMENT"].notna() & pay["DAYS_ENTRY_PAYMENT"].lt(0)].copy()
    pay = pay.sort_values([*keys, "DAYS_ENTRY_PAYMENT"], kind="mergesort")
    pay["cum_paid"] = pay.groupby(keys, observed=True)["AMT_PAYMENT"].cumsum()
    pay["full_threshold"] = pay["due_amt"] * 0.95
    full = (
        pay.loc[pay["due_amt"].gt(0) & pay["cum_paid"].ge(pay["full_threshold"])]
        .groupby(keys, observed=True)["DAYS_ENTRY_PAYMENT"]
        .first()
        .rename("full_payment_day")
        .reset_index()
    )
    base = base.merge(full, on=keys, how="left")
    base["delay_to_full"] = base["full_payment_day"] - base["DAYS_INSTALMENT"]
    base["full_payment_month"] = rel_month_from_numeric_days(base["full_payment_day"])

    full_df = base[base["full_payment_month"].notna() & base["full_payment_month"].lt(0)].copy()
    full_df["rel_month"] = full_df["full_payment_month"].astype("int16")
    add_events(
        events,
        full_df,
        full_df["delay_to_full"].between(1, 15, inclusive="both"),
        "pred_inst_late_1_15d",
        "installments",
    )
    add_events(
        events,
        full_df,
        full_df["delay_to_full"].between(16, 29, inclusive="both"),
        "pred_inst_late_16_29d",
        "installments",
    )
    add_events(
        events,
        full_df,
        full_df["delay_to_full"].le(-7),
        "pred_inst_paid_early_7d",
        "installments",
    )

    due_df = base.copy()
    due_df["rel_month"] = due_df["due_month"]
    add_events(
        events,
        due_df,
        due_df["due_amt"].gt(0) & due_df["paid_by_due"].lt(due_df["due_amt"] * 0.95),
        "pred_inst_underpaid_5pct",
        "installments",
    )
    add_events(
        events,
        due_df,
        due_df["due_amt"].gt(0) & due_df["paid_by_due"].lt(due_df["due_amt"] * 0.50),
        "pred_inst_underpaid_50pct",
        "installments",
    )

    target_df = base.copy()
    target_df["rel_month"] = target_df["target_month"]
    target_df["financial_mark"] = (
        target_df["due_amt"] - target_df["paid_by_30"]
    ).clip(lower=0.0)
    serious = (
        target_df["due_amt"].gt(0)
        & target_df["paid_by_30"].lt(target_df["due_amt"] * 0.95)
        & target_df["financial_mark"].gt(0)
    )
    add_events(
        target_events,
        target_df,
        serious,
        "target_installment_paid_less_than_95pct_by_30d",
        "installments",
        token_type="target_candidate",
        mark_col="financial_mark",
    )

    meta = {
        "rows_after_client_and_time_filter": int(len(df)),
        "scheduled_installments_after_filter": int(len(base)),
        "paid_less_than_95pct_by_30d_targets": int(serious.sum()),
        "paid_by_30_ratio_summary": {
            "nonnull": int((base["due_amt"].gt(0)).sum()),
            "mean": float((base.loc[base["due_amt"].gt(0), "paid_by_30"] / base.loc[base["due_amt"].gt(0), "due_amt"]).mean())
            if base["due_amt"].gt(0).any()
            else None,
            "p05": float((base.loc[base["due_amt"].gt(0), "paid_by_30"] / base.loc[base["due_amt"].gt(0), "due_amt"]).quantile(0.05))
            if base["due_amt"].gt(0).any()
            else None,
        },
    }
    return events, observed, target_events, meta


def build_schema_summary(raw_root: Path) -> dict:
    files = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "previous_application.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
    ]
    summary = {}
    for name in files:
        path = raw_root / name
        header = pd.read_csv(path, nrows=0)
        with path.open("rb") as handle:
            row_count = sum(1 for _ in handle) - 1
        summary[name] = {
            "rows": int(row_count),
            "columns": list(header.columns),
            "n_columns": int(len(header.columns)),
            "size_bytes": int(path.stat().st_size),
        }

    desc_path = raw_root / "HomeCredit_columns_description.csv"
    if desc_path.exists():
        desc = pd.read_csv(desc_path, encoding="ISO-8859-1")
        selected = desc[
            desc["Table"].isin(
                [
                    "application_{train|test}.csv",
                    "previous_application.csv",
                    "bureau.csv",
                    "bureau_balance.csv",
                    "POS_CASH_balance.csv",
                    "credit_card_balance.csv",
                    "installments_payments.csv",
                ]
            )
        ]
        summary["column_description_counts"] = selected["Table"].value_counts().to_dict()
    return summary


def collapse_events(
    frames: list[pd.DataFrame],
    selected: set[str] | None = None,
    *,
    preserve_multiplicity: bool = False,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["SK_ID_CURR", "rel_month", "token", "token_type", "source"])
    df = pd.concat(frames, ignore_index=True)
    if selected is not None:
        df = df[df["token"].isin(selected)].copy()
    if df.empty:
        return pd.DataFrame(columns=["SK_ID_CURR", "rel_month", "token", "token_type", "source"])
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype("int64")
    df["rel_month"] = df["rel_month"].astype("int16")
    if preserve_multiplicity:
        return df.reset_index(drop=True)
    return df.drop_duplicates(["SK_ID_CURR", "rel_month", "token", "token_type", "source"])


def collapse_observed(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["SK_ID_CURR", "rel_month"])
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["SK_ID_CURR", "rel_month"])
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype("int64")
    df["rel_month"] = df["rel_month"].astype("int16")
    return df[["SK_ID_CURR", "rel_month"]].drop_duplicates()


def first_targets(target_events: pd.DataFrame) -> pd.DataFrame:
    if target_events.empty:
        return pd.DataFrame(columns=["SK_ID_CURR", "target_month", "target_source", "target_candidate"])
    target_events = target_events.sort_values(
        ["SK_ID_CURR", "rel_month", "source", "token"], kind="mergesort"
    )
    first = target_events.groupby("SK_ID_CURR", as_index=False).first()
    return first.rename(
        columns={
            "rel_month": "target_month",
            "source": "target_source",
            "token": "target_candidate",
        }
    )[["SK_ID_CURR", "target_month", "target_source", "target_candidate"]]


def build_sequence_summary(
    apps: pd.DataFrame,
    observed_months: pd.DataFrame,
    predicate_events: pd.DataFrame,
    first_target: pd.DataFrame,
    min_sequence_months: int,
    truncate_at_first_target: bool = True,
) -> pd.DataFrame:
    obs = (
        observed_months.groupby("SK_ID_CURR")["rel_month"]
        .agg(first_observed_month="min", last_observed_month="max")
        .reset_index()
    )
    pred_bounds = (
        predicate_events.groupby("SK_ID_CURR")["rel_month"]
        .agg(first_predicate_month="min", last_predicate_month="max")
        .reset_index()
        if not predicate_events.empty
        else pd.DataFrame(columns=["SK_ID_CURR", "first_predicate_month", "last_predicate_month"])
    )
    seq = obs.merge(pred_bounds, on="SK_ID_CURR", how="left")
    seq = seq.merge(first_target, on="SK_ID_CURR", how="left")
    seq = seq.merge(apps, on="SK_ID_CURR", how="left")
    seq["has_target"] = seq["target_month"].notna()
    seq["end_month"] = np.where(
        truncate_at_first_target & seq["has_target"],
        seq["target_month"],
        seq["last_observed_month"],
    )
    seq["start_month"] = seq["first_observed_month"]
    seq = seq[seq["start_month"].notna() & seq["end_month"].notna()].copy()
    seq["start_month"] = seq["start_month"].astype("int16")
    seq["end_month"] = seq["end_month"].astype("int16")
    seq = seq[seq["start_month"].le(seq["end_month"])].copy()
    seq["sequence_length_months"] = (seq["end_month"].astype("int32") - seq["start_month"].astype("int32") + 1)
    seq = seq[seq["sequence_length_months"].ge(min_sequence_months)].copy()
    seq["sequence_id"] = "HC" + seq["SK_ID_CURR"].astype(str)
    seq["target_position"] = np.where(
        seq["has_target"],
        seq["target_month"].astype("float64") - seq["start_month"].astype("float64"),
        -1,
    ).astype("int32")
    seq["antecedent_months"] = np.where(seq["has_target"], seq["target_position"], seq["sequence_length_months"])
    if truncate_at_first_target:
        seq["end_reason"] = np.where(seq["has_target"], "first_serious_delinquency", "right_censored_last_observed")
    else:
        seq["end_reason"] = "last_observed_event_stream"
    return seq.sort_values("SK_ID_CURR", kind="mergesort").reset_index(drop=True)


def valid_predicate_events_for_sequences(
    predicate_events: pd.DataFrame,
    seq_summary: pd.DataFrame,
    truncate_at_first_target: bool = True,
) -> pd.DataFrame:
    if predicate_events.empty or seq_summary.empty:
        return predicate_events.iloc[0:0].copy()
    events = predicate_events.merge(
        seq_summary[["SK_ID_CURR", "end_month", "target_month", "has_target"]],
        on="SK_ID_CURR",
        how="inner",
    )
    before_end = events["rel_month"].le(events["end_month"])
    before_target = True
    if truncate_at_first_target:
        before_target = ~events["has_target"] | events["rel_month"].lt(events["target_month"])
    events = events.loc[before_end & before_target, ["SK_ID_CURR", "rel_month", "token", "token_type", "source"]]
    return events.drop_duplicates(["SK_ID_CURR", "rel_month", "token", "token_type", "source"])


def valid_target_events_for_sequences(
    target_events: pd.DataFrame,
    seq_summary: pd.DataFrame,
    *,
    preserve_multiplicity: bool = False,
) -> pd.DataFrame:
    if target_events.empty or seq_summary.empty:
        return target_events.iloc[0:0].copy()
    events = target_events.merge(
        seq_summary[["SK_ID_CURR", "end_month"]],
        on="SK_ID_CURR",
        how="inner",
    )
    columns = ["SK_ID_CURR", "rel_month", "token", "token_type", "source"]
    if "financial_mark" in events.columns:
        columns.append("financial_mark")
    events = events.loc[events["rel_month"].le(events["end_month"]), columns]
    if preserve_multiplicity:
        return events.reset_index(drop=True)
    return events.drop_duplicates(["SK_ID_CURR", "rel_month", "token", "token_type", "source"])


def write_sequence_parts(
    output_root: Path,
    seq_summary: pd.DataFrame,
    predicate_events: pd.DataFrame,
    target_events: pd.DataFrame,
    predicate_names: list[str],
    part_size: int,
    sparse_events: bool = False,
    truncate_at_first_target: bool = True,
    marked_financial: bool = False,
) -> tuple[int, dict[str, int]]:
    months_dir = output_root / "sequence_months"
    tokens_dir = output_root / "sequence_tokens"
    months_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    predicate_counts = {name: 0 for name in predicate_names}
    total_rows = 0
    seq_summary = seq_summary.reset_index(drop=True)
    sequence_part = pd.Series(
        np.arange(len(seq_summary), dtype=np.int64) // int(part_size),
        index=seq_summary["SK_ID_CURR"].to_numpy(dtype=np.int64),
    )
    predicate_work = predicate_events.copy()
    predicate_work["_write_part"] = predicate_work["SK_ID_CURR"].map(sequence_part)
    predicate_work = predicate_work.loc[predicate_work["_write_part"].notna()].copy()
    predicate_work["_write_part"] = predicate_work["_write_part"].astype(np.int32)
    predicate_groups = predicate_work.groupby("_write_part", sort=False)
    target_work = target_events.copy()
    target_work["_write_part"] = target_work["SK_ID_CURR"].map(sequence_part)
    target_work = target_work.loc[target_work["_write_part"].notna()].copy()
    target_work["_write_part"] = target_work["_write_part"].astype(np.int32)
    target_groups = target_work.groupby("_write_part", sort=False)

    def rows_for_part(grouped: object, part_index: int, template: pd.DataFrame) -> pd.DataFrame:
        if part_index not in grouped.indices:
            return template.iloc[0:0].copy()
        return grouped.get_group(part_index).drop(columns=["_write_part"]).copy()

    for part_idx, start in enumerate(range(0, len(seq_summary), part_size)):
        part_seq = seq_summary.iloc[start : start + part_size].copy()
        if part_seq.empty:
            continue
        part_token_frames: list[pd.DataFrame] = []

        if sparse_events:
            part_events = rows_for_part(predicate_groups, part_idx, predicate_events)
            part_events = valid_predicate_events_for_sequences(
                part_events,
                part_seq,
                truncate_at_first_target=truncate_at_first_target,
            )

            sparse_frames: list[pd.DataFrame] = []
            if not part_events.empty:
                wide = (
                    part_events.assign(value=1)
                    .pivot_table(
                        index=["SK_ID_CURR", "rel_month"],
                        columns="token",
                        values="value",
                        aggfunc="max",
                        fill_value=0,
                    )
                    .reset_index()
                    .rename(columns={"rel_month": "relative_month"})
                )
                for name in predicate_names:
                    if name not in wide.columns:
                        wide[name] = 0
                wide = wide[["SK_ID_CURR", "relative_month", *predicate_names]]
                wide["target_token"] = 0
                if marked_financial:
                    wide["target_mark_values"] = [[] for _ in range(len(wide))]
                wide["event_order"] = 0
                sparse_frames.append(
                    wide[
                        [
                            "SK_ID_CURR",
                            "relative_month",
                            "target_token",
                            *(("target_mark_values",) if marked_financial else ()),
                            "event_order",
                            *predicate_names,
                        ]
                    ]
                )

            if truncate_at_first_target:
                part_targets = part_seq.loc[
                    part_seq["has_target"],
                    ["SK_ID_CURR", "target_month", "target_source"],
                ].copy()
                target_event_rows = part_targets.rename(columns={"target_month": "relative_month"})
                target_token_rows = target_event_rows[["SK_ID_CURR", "relative_month", "target_source"]].copy()
                target_token_rows["source"] = target_token_rows["target_source"]
                target_token_rows = target_token_rows.drop(columns=["target_source"])
            else:
                part_targets = rows_for_part(target_groups, part_idx, target_events)
                part_targets = valid_target_events_for_sequences(
                    part_targets,
                    part_seq,
                    preserve_multiplicity=marked_financial,
                )
                if marked_financial:
                    if "financial_mark" not in part_targets.columns:
                        raise ValueError("installment financial targets are missing financial_mark")
                    invalid_mark = (
                        ~np.isfinite(part_targets["financial_mark"])
                        | part_targets["financial_mark"].le(0)
                    )
                    if invalid_mark.any():
                        raise ValueError("installment target marks must be finite and positive")
                    target_event_rows = (
                        part_targets.groupby(["SK_ID_CURR", "rel_month"], as_index=False)
                        .agg(
                            target_token=("financial_mark", "size"),
                            target_mark_values=("financial_mark", list),
                        )
                        .rename(columns={"rel_month": "relative_month"})
                    )
                else:
                    target_event_rows = (
                        part_targets[["SK_ID_CURR", "rel_month"]]
                        .drop_duplicates()
                        .rename(columns={"rel_month": "relative_month"})
                    )
                target_token_rows = (
                    part_targets[["SK_ID_CURR", "rel_month", "source"]]
                    .drop_duplicates()
                    .rename(columns={"rel_month": "relative_month"})
                )

            if not target_event_rows.empty:
                target_columns = [
                    "SK_ID_CURR",
                    "relative_month",
                    *(("target_token", "target_mark_values") if marked_financial else ()),
                ]
                target_rows = target_event_rows[target_columns].drop_duplicates(
                    ["SK_ID_CURR", "relative_month"]
                ).copy()
                target_rows["relative_month"] = target_rows["relative_month"].astype("int16")
                if not marked_financial:
                    target_rows["target_token"] = 1
                target_rows["event_order"] = 1
                for name in predicate_names:
                    target_rows[name] = 0
                sparse_frames.append(
                    target_rows[
                        [
                            "SK_ID_CURR",
                            "relative_month",
                            "target_token",
                            *(("target_mark_values",) if marked_financial else ()),
                            "event_order",
                            *predicate_names,
                        ]
                    ]
                )

            if not sparse_frames:
                continue

            rows = pd.concat(sparse_frames, ignore_index=True)
            rows = rows.merge(part_seq[["SK_ID_CURR", "sequence_id"]], on="SK_ID_CURR", how="inner")
            rows["relative_month"] = rows["relative_month"].astype("int16")
            rows["month_index"] = rows["relative_month"]
            rows = rows.sort_values(
                ["sequence_id", "relative_month", "event_order"],
                kind="mergesort",
            ).reset_index(drop=True)
            rows["position"] = rows.groupby("sequence_id", sort=False).cumcount().astype("int32")
            rows = rows[
                [
                    "sequence_id",
                    "SK_ID_CURR",
                    "position",
                    "month_index",
                    "relative_month",
                    "target_token",
                    *(("target_mark_values",) if marked_financial else ()),
                    *predicate_names,
                ]
            ]
            rows["target_token"] = rows["target_token"].astype(np.int32)
            for name in predicate_names:
                rows[name] = rows[name].fillna(0).astype(np.int8)
                predicate_counts[name] += int(rows[name].sum())

            if not part_events.empty:
                token_rows = part_events[["SK_ID_CURR", "rel_month", "token", "source"]].drop_duplicates().copy()
                token_rows = token_rows.merge(part_seq[["SK_ID_CURR", "sequence_id"]], on="SK_ID_CURR")
                token_rows = token_rows.rename(columns={"rel_month": "relative_month"})
                token_rows["month_index"] = token_rows["relative_month"].astype("int16")
                token_rows = token_rows.merge(
                    rows.loc[rows["target_token"].eq(0), ["sequence_id", "relative_month", "position"]],
                    on=["sequence_id", "relative_month"],
                    how="left",
                )
                token_rows["token_type"] = "predicate"
                token_rows["is_target_token"] = 0
                part_token_frames.append(
                    token_rows[
                        [
                            "sequence_id",
                            "SK_ID_CURR",
                            "position",
                            "month_index",
                            "relative_month",
                            "token",
                            "token_type",
                            "source",
                            "is_target_token",
                        ]
                    ]
                )

            if not target_token_rows.empty:
                target_rows = target_token_rows.merge(part_seq[["SK_ID_CURR", "sequence_id"]], on="SK_ID_CURR")
                target_rows["relative_month"] = target_rows["relative_month"].astype("int16")
                target_rows["month_index"] = target_rows["relative_month"]
                target_rows = target_rows.merge(
                    rows.loc[rows["target_token"].gt(0), ["sequence_id", "relative_month", "position"]],
                    on=["sequence_id", "relative_month"],
                    how="left",
                )
                target_rows["token"] = TARGET_EVENT_NAME
                target_rows["token_type"] = "target"
                target_rows["is_target_token"] = 1
                part_token_frames.append(
                    target_rows[
                        [
                            "sequence_id",
                            "SK_ID_CURR",
                            "position",
                            "month_index",
                            "relative_month",
                            "token",
                            "token_type",
                            "source",
                            "is_target_token",
                        ]
                    ]
                )

            rows.to_parquet(months_dir / f"part-{part_idx:04d}.parquet", index=False)
            if part_token_frames:
                pd.concat(part_token_frames, ignore_index=True).sort_values(
                    ["sequence_id", "position", "token"], kind="mergesort"
                ).to_parquet(tokens_dir / f"part-{part_idx:04d}.parquet", index=False)
            total_rows += len(rows)
            continue

        lengths = part_seq["sequence_length_months"].to_numpy(dtype=np.int64)
        starts = part_seq["start_month"].to_numpy(dtype=np.int64)
        total = int(lengths.sum())
        client_values = np.repeat(part_seq["SK_ID_CURR"].to_numpy(dtype=np.int64), lengths)
        sequence_values = np.repeat(part_seq["sequence_id"].astype(str).to_numpy(), lengths)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths, dtype=np.int64)
        position = (
            np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1], lengths)
        ).astype(np.int32, copy=False)
        rel_month = (
            np.repeat(starts, lengths) + position.astype(np.int64, copy=False)
        ).astype(np.int16, copy=False)
        rows = pd.DataFrame(
            {
                "sequence_id": sequence_values,
                "SK_ID_CURR": client_values,
                "position": position,
                "month_index": rel_month.astype(np.int16),
                "relative_month": rel_month.astype(np.int16),
            }
        )
        rows["target_token"] = 0

        target_part = part_seq[part_seq["has_target"]][
            ["SK_ID_CURR", "target_month", "target_source", "target_candidate"]
        ].copy()
        if not target_part.empty:
            key = pd.MultiIndex.from_frame(
                target_part.rename(columns={"target_month": "relative_month"})[
                    ["SK_ID_CURR", "relative_month"]
                ].astype({"SK_ID_CURR": "int64", "relative_month": "int16"})
            )
            row_key = pd.MultiIndex.from_frame(rows[["SK_ID_CURR", "relative_month"]])
            rows.loc[row_key.isin(key), "target_token"] = 1

        part_events = rows_for_part(predicate_groups, part_idx, predicate_events)
        part_events = part_events.merge(
            part_seq[["SK_ID_CURR", "start_month", "end_month", "target_month", "has_target"]],
            on="SK_ID_CURR",
            how="inner",
        )
        before_end = part_events["rel_month"].le(part_events["end_month"])
        before_target = ~part_events["has_target"] | part_events["rel_month"].lt(part_events["target_month"])
        part_events = part_events.loc[before_end & before_target].copy()

        for name in predicate_names:
            rows[name] = np.zeros(len(rows), dtype=np.int8)

        if not part_events.empty:
            wide = (
                part_events.assign(value=1)
                .pivot_table(
                    index=["SK_ID_CURR", "rel_month"],
                    columns="token",
                    values="value",
                    aggfunc="max",
                    fill_value=0,
                )
                .reset_index()
                .rename(columns={"rel_month": "relative_month"})
            )
            for name in predicate_names:
                if name not in wide.columns:
                    wide[name] = 0
            wide = wide[["SK_ID_CURR", "relative_month", *predicate_names]]
            rows = rows.merge(wide, on=["SK_ID_CURR", "relative_month"], how="left", suffixes=("", "_event"))
            for name in predicate_names:
                event_col = f"{name}_event"
                if event_col in rows.columns:
                    rows[name] = rows[event_col].fillna(0).astype(np.int8)
                    rows = rows.drop(columns=[event_col])
                rows[name] = rows[name].fillna(0).astype(np.int8)
                predicate_counts[name] += int(rows[name].sum())

            token_rows = part_events[["SK_ID_CURR", "rel_month", "token", "source"]].drop_duplicates().copy()
            token_rows = token_rows.merge(part_seq[["SK_ID_CURR", "sequence_id", "start_month"]], on="SK_ID_CURR")
            token_rows["position"] = token_rows["rel_month"].astype("int32") - token_rows["start_month"].astype("int32")
            token_rows = token_rows.rename(columns={"rel_month": "relative_month"})
            token_rows["month_index"] = token_rows["relative_month"]
            token_rows["token_type"] = "predicate"
            token_rows["is_target_token"] = 0
            part_token_frames.append(
                token_rows[
                    [
                        "sequence_id",
                        "SK_ID_CURR",
                        "position",
                        "month_index",
                        "relative_month",
                        "token",
                        "token_type",
                        "source",
                        "is_target_token",
                    ]
                ]
            )

        if not target_part.empty:
            target_rows = target_part.merge(part_seq[["SK_ID_CURR", "sequence_id", "start_month"]], on="SK_ID_CURR")
            target_rows["position"] = (
                target_rows["target_month"].astype("int32") - target_rows["start_month"].astype("int32")
            )
            target_rows["month_index"] = target_rows["target_month"].astype("int16")
            target_rows["relative_month"] = target_rows["target_month"].astype("int16")
            target_rows["token"] = TARGET_EVENT_NAME
            target_rows["token_type"] = "target"
            target_rows["source"] = target_rows["target_source"]
            target_rows["is_target_token"] = 1
            part_token_frames.append(
                target_rows[
                    [
                        "sequence_id",
                        "SK_ID_CURR",
                        "position",
                        "month_index",
                        "relative_month",
                        "token",
                        "token_type",
                        "source",
                        "is_target_token",
                    ]
                ]
            )

        rows.to_parquet(months_dir / f"part-{part_idx:04d}.parquet", index=False)
        if part_token_frames:
            pd.concat(part_token_frames, ignore_index=True).sort_values(
                ["sequence_id", "position", "token"], kind="mergesort"
            ).to_parquet(tokens_dir / f"part-{part_idx:04d}.parquet", index=False)
        total_rows += total

    return total_rows, predicate_counts


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root
    output_root = args.output_root
    truncate_at_first_target = args.target_mode == "first_truncated"
    marked_financial = args.financial_mark_contract == "installment_shortfall"
    if not truncate_at_first_target and not args.sparse_events:
        raise ValueError("--target-mode event_stream requires --sparse-events")
    if marked_financial and (truncate_at_first_target or not args.sparse_events):
        raise ValueError(
            "--financial-mark-contract installment_shortfall requires "
            "--target-mode event_stream --sparse-events"
        )
    require_files(
        raw_root,
        [
            "application_train.csv",
            "application_test.csv",
            "bureau.csv",
            "bureau_balance.csv",
            "previous_application.csv",
            "POS_CASH_balance.csv",
            "credit_card_balance.csv",
            "installments_payments.csv",
            "HomeCredit_columns_description.csv",
        ],
    )

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    (output_root / "metadata").mkdir(parents=True)
    (output_root / "sequences").mkdir(parents=True)

    apps = load_applications(raw_root, include_test=args.include_test)
    if args.current_contract_type is not None:
        apps = apps[apps["NAME_CONTRACT_TYPE"].eq(args.current_contract_type)].copy()
    if args.max_clients is not None:
        apps = apps.sort_values("SK_ID_CURR", kind="mergesort").head(args.max_clients).copy()
    client_ids = set(apps["SK_ID_CURR"].astype(int))
    predicate_names = selected_predicates(args.predicate_tier)
    predicate_set = set(predicate_names)

    raw_schema = build_schema_summary(raw_root)

    all_events: list[pd.DataFrame] = []
    all_observed: list[pd.DataFrame] = []
    all_targets: list[pd.DataFrame] = []
    source_meta: dict[str, dict] = {}

    events, observed, meta = preprocess_previous_application(raw_root, client_ids, args.max_lookback_months)
    all_events.extend(events)
    all_observed.extend(observed)
    source_meta["previous_application"] = meta

    bureau_map, events, observed, meta = preprocess_bureau(raw_root, client_ids, args.max_lookback_months)
    all_events.extend(events)
    all_observed.extend(observed)
    source_meta["bureau"] = meta

    events, observed, targets, meta = preprocess_bureau_balance(raw_root, bureau_map, args.max_lookback_months)
    all_events.extend(events)
    all_observed.extend(observed)
    if not marked_financial:
        all_targets.extend(targets)
    source_meta["bureau_balance"] = meta

    events, observed, targets, meta = preprocess_pos_cash(raw_root, client_ids, args.max_lookback_months)
    all_events.extend(events)
    all_observed.extend(observed)
    if not marked_financial:
        all_targets.extend(targets)
    source_meta["pos_cash"] = meta

    events, observed, targets, meta = preprocess_credit_card(raw_root, client_ids, args.max_lookback_months)
    all_events.extend(events)
    all_observed.extend(observed)
    if not marked_financial:
        all_targets.extend(targets)
    source_meta["credit_card"] = meta

    events, observed, targets, meta = preprocess_installments(raw_root, client_ids, args.max_lookback_months)
    all_events.extend(events)
    all_observed.extend(observed)
    all_targets.extend(targets)
    source_meta["installments"] = meta

    predicate_events = collapse_events(all_events, selected=predicate_set)
    observed_months = collapse_observed(all_observed)
    target_events = collapse_events(
        all_targets,
        selected=None,
        preserve_multiplicity=marked_financial,
    )

    first_target = first_targets(target_events)
    seq_summary = build_sequence_summary(
        apps=apps,
        observed_months=observed_months,
        predicate_events=predicate_events,
        first_target=first_target,
        min_sequence_months=args.min_sequence_months,
        truncate_at_first_target=truncate_at_first_target,
    )
    # Sparse storage changes the row representation, not the statistical
    # cohort.  Sequences with no predicate/target event still contribute
    # exposure and are retained in the authoritative sequence table.

    keep_clients = set(seq_summary["SK_ID_CURR"].astype("int64"))
    predicate_events = predicate_events[predicate_events["SK_ID_CURR"].isin(keep_clients)].copy()
    target_events = target_events[target_events["SK_ID_CURR"].isin(keep_clients)].copy()
    if not truncate_at_first_target:
        target_events = valid_target_events_for_sequences(
            target_events,
            seq_summary,
            preserve_multiplicity=marked_financial,
        )

    total_rows, predicate_counts = write_sequence_parts(
        output_root=output_root,
        seq_summary=seq_summary,
        predicate_events=predicate_events,
        target_events=target_events,
        predicate_names=predicate_names,
        part_size=args.part_size,
        sparse_events=args.sparse_events,
        truncate_at_first_target=truncate_at_first_target,
        marked_financial=marked_financial,
    )

    seq_summary.to_parquet(output_root / "sequences" / "part-0000.parquet", index=False)

    if truncate_at_first_target:
        target_source_counts = (
            seq_summary.loc[seq_summary["has_target"], "target_source"].value_counts(dropna=False).to_dict()
        )
        n_target_event_rows = int(seq_summary["has_target"].sum())
    else:
        target_source_counts = target_events["source"].value_counts(dropna=False).to_dict()
        n_target_event_rows = (
            int(len(target_events))
            if marked_financial
            else int(target_events[["SK_ID_CURR", "rel_month"]].drop_duplicates().shape[0])
        )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "target_event_name": TARGET_EVENT_NAME,
        "financial_mark_contract": (
            {
                "name": "installment_shortfall",
                "column": "target_mark_values",
                "target_scope": "installments_payments only",
                "mark": "max(AMT_INSTALMENT - cumulative AMT_PAYMENT by DAYS_INSTALMENT+30, 0)",
                "currency": "Home Credit amount units",
                "multiple_same_month_installments": "preserved as separate marks",
            }
            if marked_financial
            else None
        ),
        "target_definition": (
            {
                "installments_payments": (
                    "scheduled-installment level: positive shortfall remains and cumulative "
                    "AMT_PAYMENT by DAYS_INSTALMENT + 30 is less than 95% of AMT_INSTALMENT"
                )
            }
            if marked_financial
            else {
                "bureau_balance": "entry into STATUS in {2,3,4,5}",
                "POS_CASH_balance": "entry into SK_DPD >= 30 or SK_DPD_DEF >= 30",
                "credit_card_balance": "entry into SK_DPD >= 30 or SK_DPD_DEF >= 30",
                "installments_payments": (
                    "scheduled-installment level: cumulative AMT_PAYMENT by "
                    "DAYS_INSTALMENT + 30 is less than 95% of AMT_INSTALMENT"
                ),
            }
        ),
        "censoring": {
            "anchor": "current application date is t=0",
            "kept_time": "strictly before current application: relative month < 0",
            "target_handling": (
                "truncated at first serious delinquency; predicates in target month are removed"
                if truncate_at_first_target
                else "target events remain in the full historical sequence; target is never a predicate"
            ),
            "negative_sequences": "right-censored at each client's last observed historical month",
        },
        "f0_contract": {
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded": True,
            "predicate_history_includes_target_labeled_observations": (
                "not applicable: first-event process is censored at its first target"
                if truncate_at_first_target
                else True
            ),
            "strict_future_effect_required": True,
        },
        "predicate_tier": args.predicate_tier,
        "sparse_events": bool(args.sparse_events),
        "target_mode": args.target_mode,
        "target_process": (
            "first_event" if truncate_at_first_target else "recurrent"
        ),
        "current_contract_type_filter": args.current_contract_type,
        "predicate_names": predicate_names,
        "n_predicates": int(len(predicate_names)),
        "excluded_from_predicates": [
            "application_train/test snapshot fields",
            "Kaggle TARGET",
            "severe delinquency states used to define the TPP target",
            (
                "all rows at or after a client's first target event"
                if truncate_at_first_target
                else "target event tokens; target_token is a response column only"
            ),
        ],
        "n_application_clients": int(len(apps)),
        "n_sequences": int(len(seq_summary)),
        "n_sequence_rows": int(total_rows),
        "n_target_sequences": int(seq_summary["has_target"].sum()),
        "n_target_event_rows": n_target_event_rows,
        "n_censored_sequences": int((~seq_summary["has_target"]).sum()),
        "target_source_counts": target_source_counts,
        "sequence_length_summary": {
            "mean": float(seq_summary["sequence_length_months"].mean()) if len(seq_summary) else None,
            "median": float(seq_summary["sequence_length_months"].median()) if len(seq_summary) else None,
            "max": int(seq_summary["sequence_length_months"].max()) if len(seq_summary) else 0,
        },
        "antecedent_months_summary": {
            "mean": float(seq_summary["antecedent_months"].mean()) if len(seq_summary) else None,
            "median": float(seq_summary["antecedent_months"].median()) if len(seq_summary) else None,
            "max": int(seq_summary["antecedent_months"].max()) if len(seq_summary) else 0,
        },
        "predicate_counts": predicate_counts,
        "source_meta": source_meta,
    }
    (output_root / "metadata" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "metadata" / "raw_schema_summary.json").write_text(
        json.dumps(raw_schema, indent=2), encoding="utf-8"
    )
    catalog = [
        {
            "name": p.name,
            "family": p.family,
            "tier": p.tier,
            "selected": p.name in predicate_set,
            "raw_table": p.raw_table,
            "raw_fields": list(p.raw_fields),
            "description": p.description,
        }
        for p in PREDICATE_CATALOG
    ]
    (output_root / "metadata" / "predicate_catalog.json").write_text(
        json.dumps(catalog, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
