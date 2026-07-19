from __future__ import annotations

import hashlib
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import write_dataset


PERFORMANCE_COLUMNS = (
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
)

PREDICATES = (
    "pred_eltv_enters_high_ltv",
    "pred_eltv_exits_high_ltv",
    "pred_eltv_enters_negative_equity",
    "pred_eltv_exits_negative_equity",
    "pred_eltv_deterioration_starts_within_band",
    "pred_eltv_improvement_starts_within_band",
    "pred_upb_increase_starts",
    "pred_upb_flat_starts",
    "pred_upb_paydown_resumes",
    "pred_upb_paydown_acceleration_starts",
    "pred_upb_paydown_deceleration_starts",
    "pred_upb_paydown_steady_starts",
)


def _read(path: Path) -> pd.DataFrame:
    use = [
        "loan_id",
        "monthly_reporting_period",
        "current_actual_upb",
        "current_loan_delinquency_status",
        "loan_age",
        "zero_balance_code",
        "eltv",
    ]
    frame = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=PERFORMANCE_COLUMNS,
        usecols=use,
        dtype="string",
        keep_default_na=False,
        engine="c",
    )
    for name in use:
        frame[name] = frame[name].str.strip()
    period = pd.to_numeric(frame["monthly_reporting_period"], errors="coerce")
    year, month = period // 100, period % 100
    if bool(period.isna().any() or (~month.between(1, 12)).any()):
        raise ValueError(f"invalid monthly reporting period in {path}")
    frame["time"] = (year * 12 + month).astype(np.int64)
    frame["upb"] = pd.to_numeric(frame["current_actual_upb"], errors="coerce")
    frame["loan_age_num"] = pd.to_numeric(frame["loan_age"], errors="coerce")
    frame["eltv_num"] = pd.to_numeric(frame["eltv"], errors="coerce")
    status = frame["current_loan_delinquency_status"].str.upper()
    numeric = pd.to_numeric(status.where(status.ne("RA")), errors="coerce")
    frame["target"] = (numeric.ge(3) | status.eq("RA")).fillna(False)
    frame = frame.sort_values(["loan_id", "time"], kind="stable").reset_index(drop=True)
    if bool(frame.duplicated(["loan_id", "time"]).any()):
        raise ValueError(f"duplicated loan-month in {path}")
    return frame[
        ["loan_id", "time", "upb", "loan_age_num", "eltv_num", "target"]
    ].copy()


def _contiguous(frame: pd.DataFrame, lag: int) -> pd.Series:
    return (
        frame["loan_id"].eq(frame["loan_id"].shift(lag))
        & frame["time"].eq(frame["time"].shift(lag) + lag)
    ).fillna(False)


def _prefix(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["position"] = frame.groupby("loan_id", sort=False).cumcount()
    first_target = (
        frame.loc[frame["target"], ["loan_id", "position"]]
        .groupby("loan_id", sort=False)["position"]
        .min()
    )
    stop = frame["loan_id"].map(first_target)
    frame = frame.loc[stop.isna() | frame["position"].le(stop)].copy()
    gap = frame["loan_id"].eq(frame["loan_id"].shift()) & ~frame["time"].eq(
        frame["time"].shift() + 1
    )
    first_gap = (
        frame.loc[gap, ["loan_id", "position"]]
        .groupby("loan_id", sort=False)["position"]
        .min()
    )
    gap_stop = frame["loan_id"].map(first_gap)
    frame = frame.loc[gap_stop.isna() | frame["position"].lt(gap_stop)].copy()
    if bool(frame.groupby("loan_id", sort=False)["target"].sum().gt(1).any()):
        raise AssertionError("first-event prefix retained multiple targets")
    return frame


def _predicate_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    prev1, prev2, prev3 = (_contiguous(frame, lag) for lag in (1, 2, 3))
    eltv = frame["eltv_num"]
    e1, e2 = eltv.shift(), eltv.shift(2)
    valid = eltv.notna() & eltv.ne(999)
    valid1, valid2 = e1.notna() & e1.ne(999), e2.notna() & e2.ne(999)

    def band(values: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [values.lt(80).fillna(False), values.lt(100).fillna(False)],
                [0, 1],
                default=2,
            ),
            index=frame.index,
        )

    b0, b1, b2 = band(eltv), band(e1), band(e2)
    epair = prev1 & valid & valid1
    prior_rise = prev2 & valid1 & valid2 & b1.eq(b2) & e1.gt(e2)
    prior_fall = prev2 & valid1 & valid2 & b1.eq(b2) & e1.lt(e2)
    upb, u1, u2, u3 = (
        frame["upb"],
        frame["upb"].shift(),
        frame["upb"].shift(2),
        frame["upb"].shift(3),
    )
    pair = prev1 & upb.gt(0) & u1.gt(0)
    triple = pair & prev2 & u2.gt(0)
    quad = triple & prev3 & u3.gt(0)
    d0, d1, d2 = u1 - upb, u2 - u1, u3 - u2
    out = pd.DataFrame(index=frame.index)
    out[PREDICATES[0]] = epair & b1.eq(0) & b0.eq(1)
    out[PREDICATES[1]] = epair & b1.eq(1) & b0.eq(0)
    out[PREDICATES[2]] = epair & b1.lt(2) & b0.eq(2)
    out[PREDICATES[3]] = epair & b1.eq(2) & b0.lt(2)
    out[PREDICATES[4]] = epair & b0.eq(b1) & eltv.gt(e1) & ~prior_rise
    out[PREDICATES[5]] = epair & b0.eq(b1) & eltv.lt(e1) & ~prior_fall
    out[PREDICATES[6]] = triple & upb.gt(u1) & u1.le(u2)
    out[PREDICATES[7]] = triple & upb.eq(u1) & u1.ne(u2)
    out[PREDICATES[8]] = triple & upb.lt(u1) & u1.ge(u2)
    accelerating = quad & d0.gt(0) & d1.gt(0) & d0.gt(d1)
    prior_accelerating = quad & d1.gt(0) & d2.gt(0) & d1.gt(d2)
    decelerating = quad & d0.gt(0) & d1.gt(0) & d0.lt(d1)
    prior_decelerating = quad & d1.gt(0) & d2.gt(0) & d1.lt(d2)
    steady = quad & d0.gt(0) & d1.gt(0) & d0.eq(d1)
    prior_steady = quad & d1.gt(0) & d2.gt(0) & d1.eq(d2)
    out[PREDICATES[9]] = accelerating & ~prior_accelerating
    out[PREDICATES[10]] = decelerating & ~prior_decelerating
    out[PREDICATES[11]] = steady & ~prior_steady
    out.loc[frame["target"], :] = False
    return out.fillna(False).astype(np.uint8)


def _load_vintage(
    item: tuple[str, Path],
) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    """Read, hash and derive one independent vintage deterministically."""
    vintage, path = item
    raw = _read(path)
    frame = _prefix(raw)
    del raw
    matrix = _predicate_matrix(frame)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return vintage, frame, matrix, digest.hexdigest()


def preprocess_freddie(
    input_root: str | Path,
    output_root: str | Path,
    *,
    vintages: tuple[str, ...],
    overwrite: bool = False,
) -> Path:
    input_root, output_root = Path(input_root), Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    pattern = re.compile(r"historical_data_time_(\d{4}Q[1-4])\.txt$")
    paths: list[tuple[str, Path]] = []
    requested = set(vintages)
    for path in input_root.glob(
        "[0-9][0-9][0-9][0-9]/Q[1-4]/historical_data_time_*.txt"
    ):
        match = pattern.match(path.name)
        if match and (not requested or match.group(1) in requested):
            paths.append((match.group(1), path))
    paths.sort()
    if requested - {name for name, _ in paths}:
        raise FileNotFoundError(
            f"missing vintages: {sorted(requested - {name for name, _ in paths})}"
        )
    if not paths:
        raise FileNotFoundError("no Freddie performance files found")
    entity_parts, event_parts, target_parts = [], [], []
    entity_offset = 0
    source_digests: dict[str, str] = {}
    # Vintages are independent preprocessing units.  executor.map preserves
    # the sorted vintage order, so parallel I/O and predicate construction do
    # not alter entity codes, file digests or output bytes.
    with ThreadPoolExecutor(
        max_workers=min(3, len(paths)), thread_name_prefix="crbstpp-freddie"
    ) as executor:
        derived = executor.map(_load_vintage, paths)
        for vintage, frame, matrix, source_digest in derived:
            groups = frame.groupby("loan_id", sort=False)
            entities = (
                groups.agg(
                    start_time=("time", "first"),
                    end_time=("time", "last"),
                    start_age=("loan_age_num", "first"),
                )
                .reset_index()
                .rename(columns={"loan_id": "entity_id"})
            )
            if bool(
                entities["start_age"].isna().any() or (entities["start_age"] < 0).any()
            ):
                raise ValueError("invalid Freddie start loan age")
            entities["split_group"] = entities["start_time"].astype(
                np.int64
            ) - entities["start_age"].astype(np.int64)
            entities["baseline_origin"] = entities["start_age"].astype(np.int64)
            entity_codes = pd.Series(
                np.arange(len(entities), dtype=np.int32) + entity_offset,
                index=entities["entity_id"],
            )
            active = matrix.to_numpy(dtype=bool)
            row_index, predicate_code = np.nonzero(active)
            events = pd.DataFrame(
                {
                    "entity_code": frame.iloc[row_index]["loan_id"]
                    .map(entity_codes)
                    .to_numpy(dtype=np.int32),
                    "time": frame.iloc[row_index]["time"].to_numpy(dtype=np.int64),
                    "predicate_code": predicate_code.astype(np.int16),
                }
            )
            target_rows = frame.loc[frame["target"]]
            targets = pd.DataFrame(
                {
                    "entity_code": target_rows["loan_id"]
                    .map(entity_codes)
                    .to_numpy(dtype=np.int32),
                    "time": target_rows["time"].to_numpy(dtype=np.int64),
                    "multiplicity": np.ones(len(target_rows), dtype=np.int32),
                }
            )
            entity_parts.append(
                entities[
                    [
                        "entity_id",
                        "start_time",
                        "end_time",
                        "baseline_origin",
                        "split_group",
                    ]
                ]
            )
            event_parts.append(events)
            target_parts.append(targets)
            entity_offset += len(entities)
            source_digests[vintage] = source_digest
    entities = pd.concat(entity_parts, ignore_index=True)
    events = (
        pd.concat(event_parts, ignore_index=True)
        .sort_values(["entity_code", "time", "predicate_code"], kind="stable")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    targets = (
        pd.concat(target_parts, ignore_index=True)
        .sort_values(["entity_code", "time"], kind="stable")
        .reset_index(drop=True)
    )
    return write_dataset(
        output_root,
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=PREDICATES,
        likelihood="first_event_cloglog",
        time_unit="month",
        ticks_per_unit=1,
        adverse_event_name="first serious mortgage delinquency (90+ DPD or REO acquisition)",
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
        },
        provenance={
            "preprocessor": "crbstpp.preprocess.freddie.v1",
            "vintages": [name for name, _ in paths],
            "source_sha256": source_digests,
            "predicate_definition": "primitive ELTV-band and UPB-direction transition onsets",
        },
    )
