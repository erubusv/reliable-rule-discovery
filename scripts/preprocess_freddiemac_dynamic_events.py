#!/usr/bin/env python3
"""Build loan-level continuous event sequences from Freddie Mac SFLLD.

Output unit:
    One mortgage loan is one sequence.

Sequence rule:
    * If the first 90+ delinquency / REO acquisition target event T occurs,
      the observable analysis prefix ends at that first T month.
    * A gap inside the observable analysis prefix right-censors the sequence at
      the last consecutive month before the gap.  Rows after T cannot decide
      whether a loan is admitted, and no sequence is joined across a gap.
    * Predicate tokens at the T month are zeroed, so T is consequent-only.
    * If no T occurs, the sequence is kept through its last observed month and
      is marked by an end reason: active cutoff or zero-balance termination.

Feature time rule:
    Predicate columns use only month t and, for transition onsets, consecutive
    months t-1 and t-2. They never use t+1 or later information.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from predicate_policy import CURRENT_PREDICATE_GROUPS, predicate_policy_metadata


PERF_NAMES = [
    "loan_id",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_deferred_upb",
    "ddlpi",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance_preservation_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "modification_cost",
    "step_modification_flag",
    "deferred_payment_plan",
    "eltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
    "current_month_modification_cost",
    "interest_bearing_upb",
]

PERF_USECOLS = [
    "loan_id",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "modification_flag",
    "zero_balance_code",
    "current_interest_rate",
    "current_deferred_upb",
    "deferred_payment_plan",
    "eltv",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
]

PREDICATE_COLUMNS = [
    "pred_eltv_enters_high_ltv",
    "pred_eltv_exits_high_ltv",
    "pred_eltv_enters_negative_equity",
    "pred_eltv_exits_negative_equity",
    "pred_eltv_rises_within_band",
    "pred_eltv_falls_within_band",
    "pred_upb_increase_starts",
    "pred_upb_increase_continues",
    "pred_upb_flat_starts",
    "pred_upb_paydown_resumes",
    "pred_upb_paydown_accelerates",
    "pred_upb_paydown_decelerates",
    "pred_upb_paydown_steady",
]

TARGET_TOKEN = "T_SDQ3"
PREPROCESSING_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class QuarterFile:
    year: int
    quarter: int
    perf_path: Path

    @property
    def label(self) -> str:
        return f"{self.year}Q{self.quarter}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build continuous Freddie Mac loan-level event sequences."
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/freddiemac"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/freddiemac/processed/sdq3_primitive_v5_sparse"),
    )
    parser.add_argument(
        "--vintage",
        action="append",
        help="Process one vintage such as 2023Q1. May be repeated. Default: all found.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output root before writing.",
    )
    parser.add_argument(
        "--emit-target-mark",
        action="store_true",
        help=(
            "Emit current UPB at T as an optional exposure mark. The default "
            "first-event occurrence experiment is unmarked and does not "
            "validate or write this unused column."
        ),
    )
    parser.add_argument(
        "--skip-sequence-tokens",
        action="store_true",
        help=(
            "Skip the redundant long token table when training reads the wide "
            "sequence_months table directly."
        ),
    )
    return parser.parse_args()


def discover_quarters(input_root: Path, requested: Iterable[str] | None) -> list[QuarterFile]:
    requested_set = {x.upper() for x in requested or []}
    pattern = re.compile(r"historical_data_time_(\d{4})Q([1-4])\.txt$")
    quarters: list[QuarterFile] = []
    for perf_path in sorted(
        input_root.glob("[0-9][0-9][0-9][0-9]/Q[1-4]/historical_data_time_*.txt")
    ):
        match = pattern.match(perf_path.name)
        if not match:
            continue
        year = int(match.group(1))
        quarter = int(match.group(2))
        label = f"{year}Q{quarter}"
        if requested_set and label not in requested_set:
            continue
        quarters.append(QuarterFile(year=year, quarter=quarter, perf_path=perf_path))

    found = {q.label for q in quarters}
    missing = requested_set - found
    if missing:
        raise FileNotFoundError(f"Requested vintages not found: {sorted(missing)}")
    if not quarters:
        raise FileNotFoundError(f"No monthly performance files found below {input_root}")
    return quarters


def clean_str(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(clean_str(series).replace({"": pd.NA}), errors="coerce")


def read_performance(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=PERF_NAMES,
        usecols=PERF_USECOLS,
        dtype="string",
        keep_default_na=False,
        na_values=[],
        engine="c",
    )
    for col in PERF_USECOLS:
        df[col] = clean_str(df[col])

    yyyymm = to_num(df["monthly_reporting_period"])
    year = (yyyymm // 100).astype("Int64")
    month = (yyyymm % 100).astype("Int64")
    valid_month = year.notna() & month.between(1, 12)
    df["month_index"] = (year * 12 + month).astype("Int64").where(valid_month)
    df["reporting_year"] = year.where(valid_month)
    df["reporting_month"] = month.where(valid_month)
    invalid_months = int(df["month_index"].isna().sum())
    if invalid_months:
        raise ValueError(f"{path} has invalid Monthly Reporting Period rows: {invalid_months}")

    for col in [
        "current_actual_upb",
        "loan_age",
        "current_interest_rate",
        "current_deferred_upb",
        "eltv",
    ]:
        df[col] = to_num(df[col])

    delq = clean_str(df["current_loan_delinquency_status"]).str.upper()
    df["delq_status"] = delq
    df["delq_numeric"] = pd.to_numeric(delq.where(delq.ne("RA")), errors="coerce")
    df["is_target_month"] = (df["delq_numeric"].ge(3) | delq.eq("RA")).fillna(False)
    df["is_terminated"] = clean_str(df["zero_balance_code"]).ne("")

    df = df.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)
    duplicate_count = int(df.duplicated(["loan_id", "month_index"]).sum())
    if duplicate_count:
        raise ValueError(f"{path} has duplicated loan-month rows: {duplicate_count}")
    return df


def as_i8(mask: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(mask, pd.Series):
        return mask.fillna(False).astype("int8")
    return pd.Series(mask).fillna(False).astype("int8")


def consecutive_previous(df: pd.DataFrame) -> pd.Series:
    return (
        df["loan_id"].eq(df["loan_id"].shift(1))
        & df["month_index"].eq(df["month_index"].shift(1) + 1)
    ).fillna(False)


def previous(series: pd.Series, has_prev: pd.Series) -> pd.Series:
    return series.shift(1).where(has_prev)


def previous_n(df: pd.DataFrame, series: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    has_lag = (
        df["loan_id"].eq(df["loan_id"].shift(n))
        & df["month_index"].eq(df["month_index"].shift(n) + n)
    ).fillna(False)
    return series.shift(n).where(has_lag), has_lag


def add_legacy_dynamic_predicates(df: pd.DataFrame) -> pd.DataFrame:
    has_prev = consecutive_previous(df)

    deferral = clean_str(df["deferred_payment_plan"]).str.upper()
    assistance = clean_str(df["borrower_assistance_status_code"]).str.upper()
    disaster = clean_str(df["delinquency_due_to_disaster"]).str.upper()

    prev_deferred_upb = previous(df["current_deferred_upb"], has_prev)
    prev_assistance = previous(assistance, has_prev)
    prev_disaster = previous(disaster, has_prev)
    prev_eltv = previous(df["eltv"], has_prev)
    prev_upb = previous(df["current_actual_upb"], has_prev)
    prev_loan_age = previous(df["loan_age"], has_prev)
    prev2_eltv, has_prev2 = previous_n(df, df["eltv"], 2)
    prev3_eltv, has_prev3 = previous_n(df, df["eltv"], 3)
    prev4_eltv, has_prev4 = previous_n(df, df["eltv"], 4)
    prev5_eltv, has_prev5 = previous_n(df, df["eltv"], 5)
    prev6_eltv, has_prev6 = previous_n(df, df["eltv"], 6)
    prev2_upb, has_prev2_upb = previous_n(df, df["current_actual_upb"], 2)
    prev3_upb, has_prev3_upb = previous_n(df, df["current_actual_upb"], 3)
    prev6_upb, has_prev6_upb = previous_n(df, df["current_actual_upb"], 6)
    prev7_upb, has_prev7_upb = previous_n(df, df["current_actual_upb"], 7)

    eltv = df["eltv"]
    eltv_valid = eltv.notna() & eltv.ne(999)
    prev_eltv_valid = prev_eltv.notna() & prev_eltv.ne(999)
    prev2_eltv_valid = prev2_eltv.notna() & prev2_eltv.ne(999)
    prev3_eltv_valid = prev3_eltv.notna() & prev3_eltv.ne(999)
    prev4_eltv_valid = prev4_eltv.notna() & prev4_eltv.ne(999)
    prev5_eltv_valid = prev5_eltv.notna() & prev5_eltv.ne(999)
    prev6_eltv_valid = prev6_eltv.notna() & prev6_eltv.ne(999)
    upb = df["current_actual_upb"]
    loan_age = df["loan_age"]
    upb_valid = upb.notna() & upb.gt(0)
    prev_upb_valid = prev_upb.notna() & prev_upb.gt(0)
    prev2_upb_valid = prev2_upb.notna() & prev2_upb.gt(0)
    prev3_upb_valid = prev3_upb.notna() & prev3_upb.gt(0)
    prev6_upb_valid = prev6_upb.notna() & prev6_upb.gt(0)
    prev7_upb_valid = prev7_upb.notna() & prev7_upb.gt(0)
    eltv_for_rank = eltv.where(eltv_valid)
    eltv_month_p90 = eltv_for_rank.groupby(df["month_index"], sort=False).transform(
        "quantile", q=0.90
    )
    prev_eltv_month_p90 = previous(eltv_month_p90, has_prev)
    eltv_month_rank = eltv_for_rank.groupby(df["month_index"], sort=False).rank(pct=True)
    prev3_eltv_month_rank, has_prev3_rank = previous_n(df, eltv_month_rank, 3)
    # pandas GroupBy.first skips nulls.  That would replace a missing first-row
    # balance with a later balance and leak future information into an
    # "initial UPB" predicate.  Repeat the literal first observed row instead.
    loan_values = df["loan_id"].to_numpy(copy=False)
    loan_starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(loan_values[1:] != loan_values[:-1]).astype(np.int64) + 1,
        )
    )
    loan_ends = np.concatenate((loan_starts[1:], np.asarray([len(df)], dtype=np.int64)))
    initial_upb = pd.Series(
        np.repeat(upb.to_numpy(dtype=np.float64, copy=False)[loan_starts], loan_ends - loan_starts),
        index=df.index,
        dtype="float64",
    )
    initial_upb_valid = initial_upb.notna() & initial_upb.gt(0)

    predicates = pd.DataFrame(index=df.index)
    predicates["pred_payment_deferral_current"] = as_i8(deferral.eq("Y"))
    predicates["pred_deferred_upb_starts"] = as_i8(
        has_prev & df["current_deferred_upb"].fillna(0).gt(0) & prev_deferred_upb.fillna(0).le(0)
    )
    predicates["pred_deferred_upb_clears"] = as_i8(
        has_prev & df["current_deferred_upb"].fillna(0).le(0) & prev_deferred_upb.fillna(0).gt(0)
    )
    predicates["pred_forbearance_starts"] = as_i8(has_prev & assistance.eq("F") & prev_assistance.ne("F"))
    predicates["pred_forbearance_ends"] = as_i8(has_prev & prev_assistance.eq("F") & assistance.ne("F"))
    predicates["pred_repayment_plan_starts"] = as_i8(
        has_prev & assistance.eq("R") & prev_assistance.ne("R")
    )
    predicates["pred_repayment_plan_ends"] = as_i8(has_prev & prev_assistance.eq("R") & assistance.ne("R"))
    predicates["pred_trial_plan_starts"] = as_i8(has_prev & assistance.eq("T") & prev_assistance.ne("T"))
    predicates["pred_trial_plan_ends"] = as_i8(has_prev & prev_assistance.eq("T") & assistance.ne("T"))
    predicates["pred_disaster_hardship_starts"] = as_i8(has_prev & disaster.eq("Y") & prev_disaster.ne("Y"))
    predicates["pred_disaster_hardship_ends"] = as_i8(has_prev & prev_disaster.eq("Y") & disaster.ne("Y"))
    predicates["pred_eltv_cross_90"] = as_i8(
        has_prev & eltv_valid & prev_eltv_valid & eltv.ge(90) & prev_eltv.lt(90)
    )
    predicates["pred_eltv_cross_up_90"] = predicates["pred_eltv_cross_90"]
    predicates["pred_eltv_cross_down_90"] = as_i8(
        has_prev & eltv_valid & prev_eltv_valid & prev_eltv.ge(90) & eltv.lt(90)
    )
    predicates["pred_eltv_jump_10pp"] = as_i8(
        has_prev & eltv_valid & prev_eltv_valid & (eltv - prev_eltv).ge(10)
    )
    predicates["pred_eltv_drop_10pp"] = as_i8(
        has_prev & eltv_valid & prev_eltv_valid & (prev_eltv - eltv).ge(10)
    )
    predicates["pred_upb_increase_1pct"] = as_i8(
        has_prev
        & upb_valid
        & prev_upb_valid
        & loan_age.notna()
        & prev_loan_age.notna()
        & loan_age.gt(6)
        & prev_loan_age.gt(5)
        & upb.ge(prev_upb * 1.01)
        & (upb - prev_upb).ge(1000)
    )
    predicates["pred_eltv_cross_up_80"] = as_i8(
        has_prev & eltv_valid & prev_eltv_valid & eltv.ge(80) & prev_eltv.lt(80)
    )
    predicates["pred_eltv_cross_up_100"] = as_i8(
        has_prev & eltv_valid & prev_eltv_valid & eltv.ge(100) & prev_eltv.lt(100)
    )
    predicates["pred_eltv_jump_5pp_1m"] = as_i8(
        has_prev & eltv_valid & prev_eltv_valid & (eltv - prev_eltv).ge(5)
    )
    predicates["pred_eltv_jump_10pp_3m"] = as_i8(
        has_prev3 & eltv_valid & prev3_eltv_valid & (eltv - prev3_eltv).ge(10)
    )
    predicates["pred_eltv_drop_10pp_3m"] = as_i8(
        has_prev3 & eltv_valid & prev3_eltv_valid & (prev3_eltv - eltv).ge(10)
    )

    sustained90 = (
        has_prev2
        & eltv_valid
        & prev_eltv_valid
        & prev2_eltv_valid
        & eltv.ge(90)
        & prev_eltv.ge(90)
        & prev2_eltv.ge(90)
    )
    prior_sustained90 = (
        has_prev3
        & prev_eltv_valid
        & prev2_eltv_valid
        & prev3_eltv_valid
        & prev_eltv.ge(90)
        & prev2_eltv.ge(90)
        & prev3_eltv.ge(90)
    )
    predicates["pred_eltv_sustained_ge_90_3m_starts"] = as_i8(
        sustained90 & ~prior_sustained90
    )

    sustained100 = (
        has_prev2
        & eltv_valid
        & prev_eltv_valid
        & prev2_eltv_valid
        & eltv.ge(100)
        & prev_eltv.ge(100)
        & prev2_eltv.ge(100)
    )
    prior_sustained100 = (
        has_prev3
        & prev_eltv_valid
        & prev2_eltv_valid
        & prev3_eltv_valid
        & prev_eltv.ge(100)
        & prev2_eltv.ge(100)
        & prev3_eltv.ge(100)
    )
    predicates["pred_eltv_sustained_ge_100_3m_starts"] = as_i8(
        sustained100 & ~prior_sustained100
    )

    eltv_window6_valid = (
        has_prev5
        & eltv_valid
        & prev_eltv_valid
        & prev2_eltv_valid
        & prev3_eltv_valid
        & prev4_eltv_valid
        & prev5_eltv_valid
    )
    eltv_current_window = [
        values.to_numpy(dtype=np.float64, copy=False)
        for values in (eltv, prev_eltv, prev2_eltv, prev3_eltv, prev4_eltv, prev5_eltv)
    ]
    eltv_window6_max = pd.Series(np.maximum.reduce(eltv_current_window), index=df.index)
    eltv_window6_min = pd.Series(np.minimum.reduce(eltv_current_window), index=df.index)
    eltv_prior_window6_valid = (
        has_prev6
        & prev_eltv_valid
        & prev2_eltv_valid
        & prev3_eltv_valid
        & prev4_eltv_valid
        & prev5_eltv_valid
        & prev6_eltv_valid
    )
    eltv_prior_window = [
        values.to_numpy(dtype=np.float64, copy=False)
        for values in (
            prev_eltv,
            prev2_eltv,
            prev3_eltv,
            prev4_eltv,
            prev5_eltv,
            prev6_eltv,
        )
    ]
    eltv_prior_window6_max = pd.Series(np.maximum.reduce(eltv_prior_window), index=df.index)
    eltv_prior_window6_min = pd.Series(np.minimum.reduce(eltv_prior_window), index=df.index)
    eltv_range6_high = eltv_window6_valid & (eltv_window6_max - eltv_window6_min).ge(15)
    eltv_prior_range6_high = (
        eltv_prior_window6_valid
        & (eltv_prior_window6_max - eltv_prior_window6_min).ge(15)
    )
    predicates["pred_eltv_range_ge_15pp_6m_starts"] = as_i8(
        eltv_range6_high & ~eltv_prior_range6_high
    )

    eltv_missing = eltv.isna() | eltv.eq(999)
    predicates["pred_eltv_missing_999_starts"] = as_i8(
        has_prev & eltv_missing & prev_eltv_valid
    )
    predicates["pred_eltv_enters_monthly_p90"] = as_i8(
        has_prev
        & eltv_valid
        & prev_eltv_valid
        & eltv_month_p90.notna()
        & prev_eltv_month_p90.notna()
        & eltv.ge(eltv_month_p90)
        & prev_eltv.lt(prev_eltv_month_p90)
    )
    predicates["pred_eltv_rank_worsens_20pctile_3m"] = as_i8(
        has_prev3_rank
        & eltv_month_rank.notna()
        & prev3_eltv_month_rank.notna()
        & (eltv_month_rank - prev3_eltv_month_rank).ge(0.20)
    )

    upb_increase_0p5pct_1m_after_6m = (
        has_prev
        & upb_valid
        & prev_upb_valid
        & loan_age.notna()
        & prev_loan_age.notna()
        & loan_age.gt(6)
        & prev_loan_age.gt(5)
        & upb.ge(prev_upb * 1.005)
        & (upb - prev_upb).ge(500)
    )
    predicates["pred_upb_increase_0p5pct"] = as_i8(upb_increase_0p5pct_1m_after_6m)
    predicates["pred_upb_increase_0p5pct_1m_after_6m"] = predicates[
        "pred_upb_increase_0p5pct"
    ]
    predicates["pred_upb_jump_1pct_3m_after_6m"] = as_i8(
        has_prev3_upb
        & upb_valid
        & prev3_upb_valid
        & loan_age.notna()
        & loan_age.gt(6)
        & upb.ge(prev3_upb * 1.01)
        & (upb - prev3_upb).ge(1000)
    )

    flat3 = (
        has_prev2_upb
        & upb_valid
        & prev2_upb_valid
        & loan_age.notna()
        & loan_age.gt(6)
        & (prev2_upb - upb).le(prev2_upb * 0.001)
        & (prev2_upb - upb).le(250)
    )
    prior_flat3 = (
        has_prev3_upb
        & prev_upb_valid
        & prev3_upb_valid
        & (prev3_upb - prev_upb).le(prev3_upb * 0.001)
        & (prev3_upb - prev_upb).le(250)
    )
    predicates["pred_upb_flat_3m_after_6m_starts"] = as_i8(flat3 & ~prior_flat3)

    slow_amort6 = (
        has_prev6_upb
        & upb_valid
        & prev6_upb_valid
        & loan_age.notna()
        & loan_age.gt(12)
        & (prev6_upb - upb).lt(prev6_upb * 0.01)
    )
    prior_slow_amort6 = (
        has_prev7_upb
        & prev_upb_valid
        & prev7_upb_valid
        & (prev7_upb - prev_upb).lt(prev7_upb * 0.01)
    )
    predicates["pred_upb_reduction_lt_1pct_6m_after_12m_starts"] = as_i8(
        slow_amort6 & ~prior_slow_amort6
    )

    upb_reduction6 = ((prev6_upb - upb) / prev6_upb).astype("float64").where(
        has_prev6_upb
        & upb_valid
        & prev6_upb_valid
        & loan_age.notna()
        & loan_age.gt(12)
    )
    upb_reduction6_age_q10 = upb_reduction6.groupby(loan_age, sort=False).transform(
        "quantile", q=0.10
    ).astype("float64")
    bottom_reduction6 = (
        upb_reduction6.notna()
        & upb_reduction6_age_q10.notna()
        & (upb_reduction6 <= upb_reduction6_age_q10)
    )
    prior_bottom_reduction6 = previous(
        bottom_reduction6.astype("float64"), has_prev
    ).fillna(0.0).gt(0.5)
    predicates["pred_upb_reduction_bottom_decile_by_age_6m_starts"] = as_i8(
        bottom_reduction6 & ~prior_bottom_reduction6
    )

    upb_to_initial_ratio = upb / initial_upb
    prev_upb_to_initial_ratio = prev_upb / initial_upb
    high_initial_ratio = (
        upb_valid
        & initial_upb_valid
        & loan_age.notna()
        & loan_age.ge(12)
        & upb_to_initial_ratio.ge(0.98)
    )
    prior_high_initial_ratio = (
        has_prev
        & prev_upb_valid
        & initial_upb_valid
        & prev_loan_age.notna()
        & prev_loan_age.ge(12)
        & prev_upb_to_initial_ratio.ge(0.98)
    )
    predicates["pred_upb_to_initial_upb_ratio_ge_0p98_after_12m"] = as_i8(
        high_initial_ratio & ~prior_high_initial_ratio
    )
    predicates["pred_loan_age_reaches_12"] = as_i8(
        has_prev & loan_age.notna() & prev_loan_age.notna() & loan_age.ge(12) & prev_loan_age.lt(12)
    )
    predicates["pred_loan_age_reaches_24"] = as_i8(
        has_prev & loan_age.notna() & prev_loan_age.notna() & loan_age.ge(24) & prev_loan_age.lt(24)
    )
    return predicates


def add_dynamic_predicates(df: pd.DataFrame) -> pd.DataFrame:
    """Create the frozen primitive-v3 outcome-blind transition dictionary.

    Every token is a one-step state transition or change-direction onset. No
    entry contains delinquency, modification, forbearance, termination, a
    rolling motif, or a future value. The six ELTV transitions partition the
    same-band and boundary-crossing cases; the UPB onsets partition distinct
    local payment-direction changes. This prevents a "primitive" from already
    encoding the pair/triplet interaction the rule grammar is meant to find.
    """
    has_prev = consecutive_previous(df)
    prev2_contiguous = (
        df["loan_id"].eq(df["loan_id"].shift(2))
        & df["month_index"].eq(df["month_index"].shift(2) + 2)
    ).fillna(False)

    eltv = df["eltv"]
    prev_eltv = previous(eltv, has_prev)
    valid_eltv = eltv.notna() & eltv.ne(999)
    valid_prev_eltv = prev_eltv.notna() & prev_eltv.ne(999)
    eltv_pair = has_prev & valid_eltv & valid_prev_eltv
    current_band = pd.Series(
        np.select(
            [
                eltv.lt(80).fillna(False).to_numpy(dtype=bool),
                eltv.lt(100).fillna(False).to_numpy(dtype=bool),
            ],
            [0, 1],
            default=2,
        ),
        index=df.index,
    )
    previous_band = pd.Series(
        np.select(
            [
                prev_eltv.lt(80).fillna(False).to_numpy(dtype=bool),
                prev_eltv.lt(100).fillna(False).to_numpy(dtype=bool),
            ],
            [0, 1],
            default=2,
        ),
        index=df.index,
    )

    upb = df["current_actual_upb"]
    prev_upb = previous(upb, has_prev)
    prev2_upb = upb.shift(2).where(prev2_contiguous)
    valid_upb_pair = has_prev & upb.notna() & prev_upb.notna() & upb.gt(0) & prev_upb.gt(0)
    valid_upb_triple = (
        valid_upb_pair
        & prev2_contiguous
        & prev2_upb.notna()
        & prev2_upb.gt(0)
    )
    current_paydown = prev_upb - upb
    previous_paydown = prev2_upb - prev_upb

    predicates = pd.DataFrame(index=df.index)
    predicates["pred_eltv_enters_high_ltv"] = as_i8(
        eltv_pair & previous_band.eq(0) & current_band.eq(1)
    )
    predicates["pred_eltv_exits_high_ltv"] = as_i8(
        eltv_pair & previous_band.eq(1) & current_band.eq(0)
    )
    predicates["pred_eltv_enters_negative_equity"] = as_i8(
        eltv_pair & previous_band.lt(2) & current_band.eq(2)
    )
    predicates["pred_eltv_exits_negative_equity"] = as_i8(
        eltv_pair & previous_band.eq(2) & current_band.lt(2)
    )
    predicates["pred_eltv_rises_within_band"] = as_i8(
        eltv_pair & current_band.eq(previous_band) & eltv.gt(prev_eltv)
    )
    predicates["pred_eltv_falls_within_band"] = as_i8(
        eltv_pair & current_band.eq(previous_band) & eltv.lt(prev_eltv)
    )
    predicates["pred_upb_increase_starts"] = as_i8(
        valid_upb_triple & upb.gt(prev_upb) & prev_upb.le(prev2_upb)
    )
    predicates["pred_upb_increase_continues"] = as_i8(
        valid_upb_triple & upb.gt(prev_upb) & prev_upb.gt(prev2_upb)
    )
    predicates["pred_upb_flat_starts"] = as_i8(
        valid_upb_triple & upb.eq(prev_upb) & prev_upb.ne(prev2_upb)
    )
    predicates["pred_upb_paydown_resumes"] = as_i8(
        valid_upb_triple & upb.lt(prev_upb) & prev_upb.ge(prev2_upb)
    )
    predicates["pred_upb_paydown_accelerates"] = as_i8(
        valid_upb_triple
        & current_paydown.gt(0)
        & previous_paydown.gt(0)
        & current_paydown.gt(previous_paydown)
    )
    predicates["pred_upb_paydown_decelerates"] = as_i8(
        valid_upb_triple
        & current_paydown.gt(0)
        & previous_paydown.gt(0)
        & current_paydown.lt(previous_paydown)
    )
    predicates["pred_upb_paydown_steady"] = as_i8(
        valid_upb_triple
        & current_paydown.gt(0)
        & previous_paydown.gt(0)
        & current_paydown.eq(previous_paydown)
    )
    return predicates[PREDICATE_COLUMNS]


def observable_sequence_prefix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Index]:
    """Return first-event risk histories without post-outcome selection.

    The input must be sorted by ``loan_id, month_index`` and contain unique
    loan-month rows.  First T is applied before continuity is assessed, so a
    gap after T is irrelevant.  A gap before T is an observation endpoint: the
    gap row and all later rows are excluded rather than silently bridged.
    """
    working = df.copy()
    working["raw_position"] = (
        working.groupby("loan_id", sort=False).cumcount().astype("int32")
    )
    target_positions = (
        working.loc[working["is_target_month"], ["loan_id", "raw_position"]]
        .groupby("loan_id", sort=False)["raw_position"]
        .min()
    )
    first_target_position = working["loan_id"].map(target_positions)
    through_first_target = first_target_position.isna() | working[
        "raw_position"
    ].le(first_target_position.astype("float64"))
    analysis = working.loc[through_first_target].copy()

    same_loan_as_previous = analysis["loan_id"].eq(analysis["loan_id"].shift(1))
    gap_starts = same_loan_as_previous & ~analysis["month_index"].eq(
        analysis["month_index"].shift(1) + 1
    )
    first_gap_positions = (
        analysis.loc[gap_starts, ["loan_id", "raw_position"]]
        .groupby("loan_id", sort=False)["raw_position"]
        .min()
    )
    first_gap_position = analysis["loan_id"].map(first_gap_positions)
    before_first_gap = first_gap_position.isna() | analysis["raw_position"].lt(
        first_gap_position.astype("float64")
    )
    prefix = analysis.loc[before_first_gap].copy()
    target_counts = prefix.groupby("loan_id", sort=False)["is_target_month"].sum()
    if bool(target_counts.gt(1).any()):
        raise AssertionError("first-event preprocessing retained multiple targets for a loan")
    return prefix, first_gap_positions.index


def build_sequence_outputs(
    q: QuarterFile,
    *,
    emit_target_mark: bool = False,
    emit_sequence_tokens: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = read_performance(q.perf_path)
    raw_rows = int(len(raw))
    raw_loans = int(raw["loan_id"].nunique())

    months, gap_censored_loan_ids = observable_sequence_prefix(raw)
    gap_censored_loans = int(len(gap_censored_loan_ids))
    predicates = add_dynamic_predicates(months)
    months = pd.concat(
        [months.reset_index(drop=True), predicates.reset_index(drop=True)], axis=1
    )
    months["target_token"] = months["is_target_month"].astype("int8")

    # T is consequent-only. Any same-month predicates are not allowed to become antecedents.
    target_row_mask = months["target_token"].eq(1)
    months.loc[target_row_mask, PREDICATE_COLUMNS] = 0

    months["sequence_id"] = months["loan_id"].astype("string")
    months["position"] = months.groupby("sequence_id", sort=False).cumcount().astype("int32")

    last_raw = raw.groupby("loan_id", sort=False).tail(1).set_index("loan_id")
    seq_group = months.groupby("sequence_id", sort=False)
    sequences = seq_group.agg(
        loan_id=("loan_id", "first"),
        start_month=("monthly_reporting_period", "first"),
        end_month=("monthly_reporting_period", "last"),
        start_month_index=("month_index", "first"),
        end_month_index=("month_index", "last"),
        sequence_length_months=("month_index", "size"),
        has_target=("target_token", "max"),
        target_token_count=("target_token", "sum"),
    ).reset_index()
    sequences["vintage_year"] = np.int16(q.year)
    sequences["vintage_quarter"] = np.int8(q.quarter)

    target_meta = months.loc[months["target_token"].eq(1), ["sequence_id", "position", "monthly_reporting_period", "month_index"]]
    target_meta = target_meta.rename(
        columns={
            "position": "target_position",
            "monthly_reporting_period": "target_month",
            "month_index": "target_month_index",
        }
    )
    sequences = sequences.merge(target_meta, on="sequence_id", how="left", validate="one_to_one")
    sequences["target_position"] = sequences["target_position"].fillna(-1).astype("int32")
    sequences["target_month"] = sequences["target_month"].fillna("").astype("string")
    sequences["target_month_index"] = sequences["target_month_index"].fillna(-1).astype("int32")
    sequences["antecedent_months"] = np.where(
        sequences["has_target"].eq(1),
        sequences["target_position"],
        sequences["sequence_length_months"],
    ).astype("int32")

    last_raw_zbc = sequences["loan_id"].map(clean_str(last_raw["zero_balance_code"]))
    sequences["last_raw_zero_balance_code"] = last_raw_zbc.fillna("").astype("string")
    gap_censored = sequences["loan_id"].isin(gap_censored_loan_ids)
    sequences["end_reason"] = np.select(
        [
            sequences["has_target"].eq(1).to_numpy(),
            gap_censored.to_numpy(),
            sequences["last_raw_zero_balance_code"].ne("").to_numpy(),
        ],
        [
            "target",
            "observation_gap",
            "termination_" + sequences["last_raw_zero_balance_code"].astype(str),
        ],
        default="active_cutoff",
    )
    sequences["is_continuous_analysis_sequence"] = np.int8(1)
    sequences["is_continuous_raw_sequence"] = (~gap_censored).astype("int8")

    month_cols = [
        "sequence_id",
        "loan_id",
        "position",
        "monthly_reporting_period",
        "reporting_year",
        "reporting_month",
        "month_index",
        "target_token",
        "raw_delq_status_t",
        "raw_delq_numeric_t",
        "raw_current_actual_upb_t",
        "raw_current_interest_rate_t",
        "raw_current_deferred_upb_t",
        "raw_loan_age_t",
        "raw_eltv_t",
        "raw_modification_flag_t",
        "raw_deferred_payment_plan_t",
        "raw_assistance_status_t",
        "raw_disaster_flag_t",
        "raw_zero_balance_code_t",
    ]
    months["raw_delq_status_t"] = months["delq_status"].astype("string")
    months["raw_delq_numeric_t"] = months["delq_numeric"].fillna(-1).astype("int16")
    months["raw_current_actual_upb_t"] = months["current_actual_upb"].astype("float64")
    optional_mark_columns: list[str] = []
    if emit_target_mark:
        invalid_target_mark = target_row_mask & (
            ~np.isfinite(months["raw_current_actual_upb_t"])
            | months["raw_current_actual_upb_t"].le(0)
        )
        if invalid_target_mark.any():
            raise ValueError(
                "serious-delinquency target has missing or nonpositive current UPB mark"
            )
        months["target_mark_values"] = [
            [float(value)] if is_target else []
            for value, is_target in zip(
                months["raw_current_actual_upb_t"].to_numpy(dtype=np.float64),
                target_row_mask.to_numpy(dtype=bool),
                strict=True,
            )
        ]
        optional_mark_columns.append("target_mark_values")
    months["raw_current_interest_rate_t"] = months["current_interest_rate"].astype("float32")
    months["raw_current_deferred_upb_t"] = months["current_deferred_upb"].fillna(0).astype("float64")
    months["raw_loan_age_t"] = months["loan_age"].fillna(-1).astype("int16")
    # ELTV is not necessarily integral; int16 silently truncated the raw audit
    # field (and could overflow malformed values).
    months["raw_eltv_t"] = months["eltv"].astype("float32")
    months["raw_modification_flag_t"] = months["modification_flag"].astype("string")
    months["raw_deferred_payment_plan_t"] = months["deferred_payment_plan"].astype("string")
    months["raw_assistance_status_t"] = months["borrower_assistance_status_code"].astype("string")
    months["raw_disaster_flag_t"] = months["delinquency_due_to_disaster"].astype("string")
    months["raw_zero_balance_code_t"] = months["zero_balance_code"].astype("string")
    sequence_months = months[
        month_cols + optional_mark_columns + PREDICATE_COLUMNS
    ].copy()
    sequence_months["reporting_year"] = sequence_months["reporting_year"].astype("int16")
    sequence_months["reporting_month"] = sequence_months["reporting_month"].astype("int8")
    sequence_months["month_index"] = sequence_months["month_index"].astype("int32")

    # The likelihood grid is reconstructed exactly from the authoritative
    # sequence start/end bounds.  A loan-month on which neither a predicate nor
    # the target occurs therefore carries no event information and need not be
    # serialized.  Keeping only event rows is an algebraic sparse
    # representation of the same point process: omitted months remain in the
    # risk set through ``start_month_index``/``end_month_index`` and retain
    # unit exposure in ``QueryContext``.  Re-number positions after filtering
    # because EventData positions index serialized observations, not elapsed
    # risk time; ``month_index`` preserves the actual temporal gaps.
    risk_month_rows = int(len(sequence_months))
    event_row_mask = sequence_months["target_token"].gt(0)
    if PREDICATE_COLUMNS:
        event_row_mask |= sequence_months[PREDICATE_COLUMNS].any(axis=1)
    sequence_months = sequence_months.loc[event_row_mask].copy()
    sequence_months["position"] = (
        sequence_months.groupby("sequence_id", sort=False)
        .cumcount()
        .astype("int32")
    )

    base_cols = [
        "sequence_id",
        "loan_id",
        "position",
        "monthly_reporting_period",
        "month_index",
    ]
    token_frames: list[pd.DataFrame] = []
    if emit_sequence_tokens:
        for pred_col in PREDICATE_COLUMNS:
            rows = sequence_months.loc[
                sequence_months[pred_col].eq(1), base_cols
            ]
            if rows.empty:
                continue
            token_frame = rows.copy()
            token_frame["token"] = pred_col
            token_frame["token_type"] = "predicate"
            token_frame["is_target_token"] = np.int8(0)
            token_frames.append(token_frame)

        t_rows = sequence_months.loc[
            sequence_months["target_token"].eq(1), base_cols
        ]
        if not t_rows.empty:
            t_frame = t_rows.copy()
            t_frame["token"] = TARGET_TOKEN
            t_frame["token_type"] = "target"
            t_frame["is_target_token"] = np.int8(1)
            token_frames.append(t_frame)

    if not token_frames:
        sequence_tokens = pd.DataFrame(
            columns=base_cols + ["token", "token_type", "is_target_token"]
        )
    else:
        sequence_tokens = pd.concat(token_frames, ignore_index=True)
        sequence_tokens = sequence_tokens.sort_values(
            ["sequence_id", "position", "is_target_token", "token"],
            kind="mergesort",
        ).reset_index(drop=True)
        sequence_tokens["month_index"] = sequence_tokens["month_index"].astype("int32")
        sequence_tokens["position"] = sequence_tokens["position"].astype("int32")

    predicate_sums = {col: int(sequence_months[col].sum()) for col in PREDICATE_COLUMNS}
    min_len_counts = {}
    for min_len in [1, 3, 6, 9, 12, 15, 18, 24]:
        ok = sequences["antecedent_months"].ge(min_len)
        min_len_counts[str(min_len)] = {
            "sequences": int(ok.sum()),
            "target_sequences": int((ok & sequences["has_target"].eq(1)).sum()),
            "target_rate": float((ok & sequences["has_target"].eq(1)).sum() / ok.sum())
            if int(ok.sum())
            else None,
        }

    metadata = {
        "vintage": q.label,
        "performance_file": str(q.perf_path),
        "raw_rows": raw_rows,
        "raw_loans": raw_loans,
        # Kept for consumers of the v3 metadata. Version 4 censors at the last
        # observable consecutive month instead of deleting an entire loan.
        "excluded_gap_loans": 0,
        "gap_censored_loans": gap_censored_loans,
        "sequence_count": int(len(sequences)),
        "target_sequence_count": int(sequences["has_target"].sum()),
        "target_sequence_rate": float(sequences["has_target"].mean()) if len(sequences) else None,
        "sequence_month_rows": int(len(sequence_months)),
        "risk_month_rows": risk_month_rows,
        "event_stream_storage": "predicate_or_target_rows_with_implicit_unit_risk_grid",
        "sequence_token_rows": int(len(sequence_tokens)),
        "sequence_tokens_emitted": bool(emit_sequence_tokens),
        "target_token_rows": int(sequence_months["target_token"].sum()),
        "predicate_sums": predicate_sums,
        "min_antecedent_month_counts": min_len_counts,
        "sequence_length": {
            "min": int(sequences["sequence_length_months"].min()) if len(sequences) else None,
            "median": float(sequences["sequence_length_months"].median()) if len(sequences) else None,
            "mean": float(sequences["sequence_length_months"].mean()) if len(sequences) else None,
            "max": int(sequences["sequence_length_months"].max()) if len(sequences) else None,
        },
        "antecedent_months": {
            "min": int(sequences["antecedent_months"].min()) if len(sequences) else None,
            "median": float(sequences["antecedent_months"].median()) if len(sequences) else None,
            "mean": float(sequences["antecedent_months"].mean()) if len(sequences) else None,
            "max": int(sequences["antecedent_months"].max()) if len(sequences) else None,
        },
        "end_reason_counts": {
            str(k): int(v) for k, v in sequences["end_reason"].value_counts().sort_index().items()
        },
        "predicate_names": PREDICATE_COLUMNS,
        "predicate_groups": {
            col: CURRENT_PREDICATE_GROUPS[col] for col in PREDICATE_COLUMNS
        },
        "predicate_policy": predicate_policy_metadata(),
        "predicate_dictionary": "freddie_primitive_dynamic_v3",
        "atomic_predicates": True,
        "occurrence_likelihood": "first_event_cloglog",
        "target_token": TARGET_TOKEN,
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        "target_mark": (
            {
                "column": "target_mark_values",
                "definition": "current_actual_upb at the first 90+ delinquency/RA target month",
                "interpretation": "outstanding balance exposed at serious delinquency, not realized credit loss",
            }
            if emit_target_mark
            else None
        ),
    }
    return sequences, sequence_months, sequence_tokens, metadata


def write_part(
    output_root: Path,
    q: QuarterFile,
    sequences: pd.DataFrame,
    sequence_months: pd.DataFrame,
    sequence_tokens: pd.DataFrame,
    metadata: dict,
) -> None:
    for subdir in ["sequences", "sequence_months", "sequence_tokens", "metadata"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)
    sequences.to_parquet(output_root / "sequences" / f"part-{q.label}.parquet", index=False)
    sequence_months.to_parquet(output_root / "sequence_months" / f"part-{q.label}.parquet", index=False)
    sequence_tokens.to_parquet(output_root / "sequence_tokens" / f"part-{q.label}.parquet", index=False)
    (output_root / "metadata" / f"metadata-{q.label}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )


def build_summary(output_root: Path, vintages: list[dict]) -> dict:
    total_sequences = sum(v["sequence_count"] for v in vintages)
    total_targets = sum(v["target_sequence_count"] for v in vintages)
    total_months = sum(v["sequence_month_rows"] for v in vintages)
    total_risk_months = sum(v["risk_month_rows"] for v in vintages)
    total_tokens = sum(v["sequence_token_rows"] for v in vintages)

    min_len_counts = {}
    for min_len in [1, 3, 6, 9, 12, 15, 18, 24]:
        key = str(min_len)
        n = sum(v["min_antecedent_month_counts"][key]["sequences"] for v in vintages)
        p = sum(v["min_antecedent_month_counts"][key]["target_sequences"] for v in vintages)
        min_len_counts[key] = {
            "sequences": int(n),
            "target_sequences": int(p),
            "target_rate": float(p / n) if n else None,
        }

    predicate_sums = {
        col: int(sum(v["predicate_sums"][col] for v in vintages)) for col in PREDICATE_COLUMNS
    }
    end_reason_counts: dict[str, int] = {}
    for v in vintages:
        for reason, count in v["end_reason_counts"].items():
            end_reason_counts[reason] = end_reason_counts.get(reason, 0) + int(count)

    return {
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "sequences_path": str(output_root / "sequences"),
        "sequence_months_path": str(output_root / "sequence_months"),
        "sequence_tokens_path": str(output_root / "sequence_tokens"),
        "source_documents": [
            "data/freddiemac/file_layout.xlsx",
            "data/freddiemac/user_guide.pdf",
        ],
        "unit": "one consecutive observable mortgage-loan risk-history prefix is one sequence",
        "target_definition": "T_SDQ3 is the first month whose delinquency status is 3+ or RA.",
        "target_process": "first_event",
        "forward_bias_rule": (
            "Predicate tokens use only t and consecutive t-1/t-2. First T is applied before "
            "continuity; a pre-T observation gap censors at its preceding month and post-T rows "
            "cannot affect admission. Predicate columns on T are 0 so T is consequent-only."
        ),
        "f0_contract": {
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded": True,
            "predicate_history_includes_target_labeled_observations": (
                "not applicable: first-event process is censored at its first target"
            ),
            "strict_future_effect_required": True,
            "atomic_predicates": True,
        },
        "predicate_names": PREDICATE_COLUMNS,
        "predicate_groups": {
            col: CURRENT_PREDICATE_GROUPS[col] for col in PREDICATE_COLUMNS
        },
        "predicate_policy": predicate_policy_metadata(),
        "predicate_dictionary": "freddie_primitive_dynamic_v3",
        "occurrence_likelihood": "first_event_cloglog",
        "target_token": TARGET_TOKEN,
        "target_mark": vintages[0].get("target_mark") if vintages else None,
        "vintages": vintages,
        "total_sequences": int(total_sequences),
        "total_target_sequences": int(total_targets),
        "total_target_sequence_rate": float(total_targets / total_sequences)
        if total_sequences
        else None,
        "total_sequence_month_rows": int(total_months),
        "total_risk_month_rows": int(total_risk_months),
        "event_stream_storage": "predicate_or_target_rows_with_implicit_unit_risk_grid",
        "total_sequence_token_rows": int(total_tokens),
        "total_predicate_sums": predicate_sums,
        "total_end_reason_counts": end_reason_counts,
        "min_antecedent_month_counts": min_len_counts,
    }


def main() -> None:
    args = parse_args()
    quarters = discover_quarters(args.input_root, args.vintage)
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {args.output_root}. Use --overwrite.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    vintages: list[dict] = []
    for q in quarters:
        print(f"[{q.label}] building continuous loan sequences from {q.perf_path}", flush=True)
        sequences, sequence_months, sequence_tokens, metadata = build_sequence_outputs(
            q,
            emit_target_mark=bool(args.emit_target_mark),
            emit_sequence_tokens=not bool(args.skip_sequence_tokens),
        )
        print(
            f"[{q.label}] sequences={metadata['sequence_count']:,} "
            f"targets={metadata['target_sequence_count']:,} "
            f"rate={metadata['target_sequence_rate']:.6f} "
            f"event_rows={metadata['sequence_month_rows']:,} "
            f"risk_months={metadata['risk_month_rows']:,} "
            f"tokens={metadata['sequence_token_rows']:,} "
            f"gap_censored_loans={metadata['gap_censored_loans']:,}",
            flush=True,
        )
        write_part(args.output_root, q, sequences, sequence_months, sequence_tokens, metadata)
        vintages.append(metadata)

    summary = build_summary(args.output_root, vintages)
    summary_path = args.output_root / "metadata" / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
