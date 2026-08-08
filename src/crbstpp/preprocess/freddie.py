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
    "pred_eltv_enters_80_100_band",
    "pred_eltv_returns_below_80",
    "pred_eltv_enters_negative_equity",
    "pred_eltv_exits_negative_equity",
    "pred_eltv_deterioration_starts_within_band",
    "pred_eltv_improvement_starts_within_band",
    "pred_upb_increase_starts",
    "pred_upb_paydown_resumes",
    "pred_upb_paydown_acceleration_starts",
    "pred_upb_paydown_deceleration_starts",
    "pred_modification_starts",
    "pred_deferred_payment_plan_starts",
    "pred_deferred_upb_starts",
)

BASELINE_CONTROLS = (
    "control_borrower_forbearance_history",
    "control_disaster_delinquency_history",
    "control_30dpd_history",
    "control_60dpd_history",
)

ALL_PREDICATES = (*PREDICATES, *BASELINE_CONTROLS)


def _read(path: Path) -> pd.DataFrame:
    use = [
        "loan_id",
        "monthly_reporting_period",
        "current_actual_upb",
        "current_loan_delinquency_status",
        "loan_age",
        "zero_balance_code",
        "eltv",
        "modification_flag",
        "deferred_payment_plan",
        "current_deferred_upb",
        "delinquency_due_to_disaster",
        "borrower_assistance_status_code",
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
    frame["delinquency_level"] = numeric.fillna(-1).astype(np.int16)
    frame["modification"] = frame["modification_flag"].str.upper().eq("Y")
    frame["deferred_plan"] = frame["deferred_payment_plan"].str.upper().eq("Y")
    frame["deferred_upb"] = pd.to_numeric(
        frame["current_deferred_upb"], errors="coerce"
    )
    frame["disaster_delinquency"] = (
        frame["delinquency_due_to_disaster"].str.upper().eq("Y")
    )
    frame["borrower_forbearance"] = (
        frame["borrower_assistance_status_code"].str.upper().eq("F")
    )
    frame["target"] = (numeric.ge(3) | status.eq("RA")).fillna(False)
    frame["termination"] = frame["zero_balance_code"].ne("")
    frame = frame.sort_values(["loan_id", "time"], kind="stable").reset_index(drop=True)
    if bool(frame.duplicated(["loan_id", "time"]).any()):
        raise ValueError(f"duplicated loan-month in {path}")
    return frame[
        [
            "loan_id",
            "time",
            "upb",
            "loan_age_num",
            "eltv_num",
            "delinquency_level",
            "modification",
            "deferred_plan",
            "deferred_upb",
            "disaster_delinquency",
            "borrower_forbearance",
            "target",
            "termination",
        ]
    ].copy()


def _contiguous(frame: pd.DataFrame, lag: int) -> pd.Series:
    return (
        frame["loan_id"].eq(frame["loan_id"].shift(lag))
        & frame["time"].eq(frame["time"].shift(lag) + lag)
    ).fillna(False)


def _prefix(frame: pd.DataFrame, *, max_observation_months: int = 36) -> pd.DataFrame:
    if max_observation_months < 1:
        raise ValueError("max_observation_months must be positive")
    frame = frame.copy()
    frame["position"] = frame.groupby("loan_id", sort=False).cumcount()
    first_absorbing = (
        frame.loc[frame["target"] | frame["termination"], ["loan_id", "position"]]
        .groupby("loan_id", sort=False)["position"]
        .min()
    )
    stop = frame["loan_id"].map(first_absorbing)
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
    frame = frame.loc[frame["position"].lt(max_observation_months)].copy()
    if bool(frame.groupby("loan_id", sort=False)["target"].sum().gt(1).any()):
        raise AssertionError("first-event prefix retained multiple targets")
    return frame


def _predicate_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    prev1, prev2, prev3 = (_contiguous(frame, lag) for lag in (1, 2, 3))
    eltv = frame["eltv_num"]
    e1 = eltv.shift()
    e2 = eltv.shift(2)
    valid = eltv.notna() & eltv.ne(999)
    valid1 = e1.notna() & e1.ne(999)
    valid2 = e2.notna() & e2.ne(999)

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
    previous_rise_same_band = prev2 & valid1 & valid2 & b1.eq(b2) & e1.gt(e2)
    previous_fall_same_band = prev2 & valid1 & valid2 & b1.eq(b2) & e1.lt(e2)
    upb, u1, u2, u3 = (
        frame["upb"],
        frame["upb"].shift(),
        frame["upb"].shift(2),
        frame["upb"].shift(3),
    )
    upb_valid = prev2 & upb.gt(0) & u1.gt(0) & u2.gt(0)
    delta, previous_delta = upb - u1, u1 - u2
    upb_quad = upb_valid & prev3 & u3.gt(0)
    # Dollar paydown is the primitive observable balance change.  Ratios would
    # introduce a new balance-scale normalization and make the exact steady
    # state almost unobservable even under a stable payment schedule.
    current_paydown = u1 - upb
    previous_paydown = u2 - u1
    prior_paydown = u3 - u2
    accelerating = (
        upb_quad
        & current_paydown.gt(0)
        & previous_paydown.gt(0)
        & current_paydown.gt(previous_paydown)
    )
    prior_accelerating = (
        upb_quad
        & previous_paydown.gt(0)
        & prior_paydown.gt(0)
        & previous_paydown.gt(prior_paydown)
    )
    decelerating = (
        upb_quad
        & current_paydown.gt(0)
        & previous_paydown.gt(0)
        & current_paydown.lt(previous_paydown)
    )
    prior_decelerating = (
        upb_quad
        & previous_paydown.gt(0)
        & prior_paydown.gt(0)
        & previous_paydown.lt(prior_paydown)
    )

    def onset(values: pd.Series) -> pd.Series:
        values = values.fillna(False).astype(bool)
        previous = values.shift(fill_value=False)
        return prev1 & values & ~previous

    out = pd.DataFrame(index=frame.index)
    out[PREDICATES[0]] = epair & b1.eq(0) & b0.eq(1)
    out[PREDICATES[1]] = epair & b1.eq(1) & b0.eq(0)
    out[PREDICATES[2]] = epair & b1.lt(2) & b0.eq(2)
    out[PREDICATES[3]] = epair & b1.eq(2) & b0.lt(2)
    out[PREDICATES[4]] = epair & b0.eq(b1) & eltv.gt(e1) & ~previous_rise_same_band
    out[PREDICATES[5]] = epair & b0.eq(b1) & eltv.lt(e1) & ~previous_fall_same_band
    out[PREDICATES[6]] = upb_valid & delta.gt(0) & previous_delta.le(0)
    out[PREDICATES[7]] = upb_valid & delta.lt(0) & previous_delta.ge(0)
    out[PREDICATES[8]] = accelerating & ~prior_accelerating
    out[PREDICATES[9]] = decelerating & ~prior_decelerating
    out[PREDICATES[10]] = onset(frame["modification"])
    out[PREDICATES[11]] = onset(frame["deferred_plan"])
    deferred_positive = frame["deferred_upb"].fillna(0.0).gt(0.0)
    out[PREDICATES[12]] = onset(deferred_positive)
    return out.fillna(False).astype(np.uint8)


def _baseline_control_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly delinquency histories used only as fixed nuisance controls.

    A status event at month t is convolved with a strictly-future M-knot
    kernel, so it controls both prior state and duration when predicting t+1,
    while the contemporaneous target status can never enter its own hazard.
    """
    out = pd.DataFrame(index=frame.index)
    out[BASELINE_CONTROLS[0]] = frame["borrower_forbearance"]
    out[BASELINE_CONTROLS[1]] = frame["disaster_delinquency"]
    out[BASELINE_CONTROLS[2]] = frame["delinquency_level"].eq(1)
    out[BASELINE_CONTROLS[3]] = frame["delinquency_level"].eq(2)
    return out.fillna(False).astype(np.uint8)


def _read_first_payment(path: Path) -> tuple[pd.Series, str]:
    """Read the contract first-payment cohort from Freddie origination data."""
    frame = pd.read_csv(
        path,
        sep="|",
        header=None,
        usecols=[1, 19],
        dtype="string",
        keep_default_na=False,
        engine="c",
    ).rename(columns={1: "first_payment_date", 19: "loan_id"})
    frame["loan_id"] = frame["loan_id"].str.strip()
    period = pd.to_numeric(frame["first_payment_date"].str.strip(), errors="coerce")
    year, month = period // 100, period % 100
    if bool(
        frame["loan_id"].eq("").any()
        or frame["loan_id"].duplicated().any()
        or period.isna().any()
        or (~month.between(1, 12)).any()
    ):
        raise ValueError(f"invalid Freddie origination cohort data in {path}")
    first_payment = pd.Series(
        (year * 12 + month).to_numpy(dtype=np.int64),
        index=frame["loan_id"],
        name="first_payment_time",
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return first_payment, digest.hexdigest()


def _assert_distinct_predicate_streams(events: pd.DataFrame) -> None:
    """Reject observationally identical event types before rule generation."""
    signatures: dict[tuple[int, bytes], list[int]] = {}
    for code, group in events.groupby("predicate_code", sort=True):
        coordinates = np.ascontiguousarray(
            group[["entity_code", "time"]].to_numpy(dtype=np.int64)
        )
        signature = (len(coordinates), hashlib.sha256(coordinates).digest())
        for previous in signatures.get(signature, []):
            prior = events.loc[
                events["predicate_code"].eq(previous), ["entity_code", "time"]
            ].to_numpy(dtype=np.int64)
            if np.array_equal(prior, coordinates):
                raise ValueError(
                    "observationally identical Freddie predicates: "
                    f"{ALL_PREDICATES[previous]} and {ALL_PREDICATES[int(code)]}"
                )
        signatures.setdefault(signature, []).append(int(code))


def _load_vintage(
    item: tuple[str, Path, Path],
    *,
    max_observation_months: int,
) -> tuple[
    str, pd.DataFrame, pd.DataFrame, pd.Series, str, str, np.ndarray, np.ndarray
]:
    """Read, hash and derive one independent vintage deterministically."""
    vintage, performance_path, origination_path = item
    raw = _read(performance_path)
    frame = _prefix(raw, max_observation_months=max_observation_months)
    del raw
    reported = _predicate_matrix(frame)
    controls = _baseline_control_matrix(frame)
    matrix = pd.concat([reported, controls], axis=1)
    report_values = reported.to_numpy(dtype=np.int64, copy=False)
    event_counts = report_values.sum(axis=0, dtype=np.int64)
    same_month_overlap = report_values.T @ report_values
    first_payment, origination_digest = _read_first_payment(origination_path)
    digest = hashlib.sha256()
    with performance_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return (
        vintage,
        frame,
        matrix,
        first_payment,
        digest.hexdigest(),
        origination_digest,
        event_counts,
        same_month_overlap,
    )


def preprocess_freddie(
    input_root: str | Path,
    output_root: str | Path,
    *,
    vintages: tuple[str, ...],
    test_vintages: tuple[str, ...] = (),
    development_fit_fraction: float = 0.75,
    partition_seed: int = 111,
    max_observation_months: int = 36,
    overwrite: bool = False,
) -> Path:
    if not 0.0 < development_fit_fraction < 1.0:
        raise ValueError("development_fit_fraction must lie strictly between 0 and 1")
    if max_observation_months < 1:
        raise ValueError("max_observation_months must be positive")
    input_root, output_root = Path(input_root), Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    pattern = re.compile(r"historical_data_time_(\d{4}Q[1-4])\.txt$")
    paths: list[tuple[str, Path, Path]] = []
    requested = set(vintages)
    for path in input_root.glob(
        "[0-9][0-9][0-9][0-9]/Q[1-4]/historical_data_time_*.txt"
    ):
        match = pattern.match(path.name)
        if match and (not requested or match.group(1) in requested):
            origination = path.with_name(f"historical_data_{match.group(1)}.txt")
            if not origination.is_file():
                raise FileNotFoundError(
                    f"missing Freddie origination file for {match.group(1)}: "
                    f"{origination}"
                )
            paths.append((match.group(1), path, origination))
    paths.sort()
    if requested - {name for name, _, _ in paths}:
        raise FileNotFoundError(
            f"missing vintages: {sorted(requested - {name for name, _, _ in paths})}"
        )
    if not paths:
        raise FileNotFoundError("no Freddie performance files found")
    available = {name for name, _, _ in paths}
    held_out = set(test_vintages)
    if held_out - available:
        raise ValueError(
            f"test vintages are not among requested vintages: {sorted(held_out - available)}"
        )
    if held_out and held_out == available:
        raise ValueError("at least one development vintage is required")
    entity_parts, event_parts, target_parts = [], [], []
    entity_offset = 0
    source_digests: dict[str, str] = {}
    predicate_event_counts = np.zeros(len(PREDICATES), dtype=np.int64)
    predicate_same_month_overlap = np.zeros(
        (len(PREDICATES), len(PREDICATES)), dtype=np.int64
    )
    # Vintages are independent preprocessing units.  executor.map preserves
    # the sorted vintage order, so parallel I/O and predicate construction do
    # not alter entity codes, file digests or output bytes.
    with ThreadPoolExecutor(
        max_workers=min(3, len(paths)), thread_name_prefix="crbstpp-freddie"
    ) as executor:
        derived = executor.map(
            lambda item: _load_vintage(
                item, max_observation_months=max_observation_months
            ),
            paths,
        )
        for (
            vintage,
            frame,
            matrix,
            first_payment,
            performance_digest,
            origination_digest,
            event_counts,
            same_month_overlap,
        ) in derived:
            predicate_event_counts += event_counts
            predicate_same_month_overlap += same_month_overlap
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
            first_payment_time = entities["entity_id"].map(first_payment)
            if bool(first_payment_time.isna().any()):
                missing = entities.loc[first_payment_time.isna(), "entity_id"].head(5)
                raise ValueError(
                    "performance loans are absent from Freddie origination data: "
                    f"{missing.tolist()}"
                )
            match = re.fullmatch(r"(\d{4})Q([1-4])", vintage)
            if match is None:
                raise ValueError(f"invalid Freddie vintage: {vintage}")
            entities["split_group"] = int(match.group(1)) * 4 + int(match.group(2)) - 1
            if held_out:
                if vintage in held_out:
                    entities["partition"] = np.int8(2)
                else:
                    threshold = int(development_fit_fraction * (1 << 64))
                    partition = np.fromiter(
                        (
                            0
                            if int.from_bytes(
                                hashlib.sha256(
                                    f"{partition_seed}:{entity_id}".encode("utf-8")
                                ).digest()[:8],
                                byteorder="big",
                            )
                            < threshold
                            else 1
                            for entity_id in entities["entity_id"]
                        ),
                        dtype=np.int8,
                        count=len(entities),
                    )
                    entities["partition"] = partition
            entities["baseline_origin"] = entities["start_age"].astype(np.int64)
            entities["dependency_group"] = entities["entity_id"].astype(str)
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
                    # All predicate attributes emitted by one loan-month
                    # performance record share a primitive identity.
                    "primitive_event_id": (
                        frame.iloc[row_index]["loan_id"]
                        .map(entity_codes)
                        .to_numpy(dtype=np.int64)
                        * np.int64(1 << 32)
                        + frame.iloc[row_index]["time"].to_numpy(dtype=np.int64)
                    ),
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
            entity_columns = [
                "entity_id",
                "start_time",
                "end_time",
                "baseline_origin",
                "split_group",
                "dependency_group",
            ]
            if held_out:
                entity_columns.append("partition")
            entity_parts.append(entities[entity_columns])
            event_parts.append(events)
            target_parts.append(targets)
            entity_offset += len(entities)
            source_digests[f"{vintage}:performance"] = performance_digest
            source_digests[f"{vintage}:origination"] = origination_digest
    entities = pd.concat(entity_parts, ignore_index=True)
    events = (
        pd.concat(event_parts, ignore_index=True)
        .sort_values(
            ["entity_code", "time", "predicate_code", "primitive_event_id"],
            kind="stable",
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    targets = (
        pd.concat(target_parts, ignore_index=True)
        .sort_values(["entity_code", "time"], kind="stable")
        .reset_index(drop=True)
    )
    _assert_distinct_predicate_streams(events)
    return write_dataset(
        output_root,
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=ALL_PREDICATES,
        predicate_roles=(
            *("reported" for _ in PREDICATES),
            *("baseline_control" for _ in BASELINE_CONTROLS),
        ),
        likelihood="first_event_cloglog",
        time_unit="month",
        ticks_per_unit=1,
        adverse_event_name="first serious mortgage delinquency (90+ DPD or REO acquisition)",
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
            "preprocessor": "crbstpp.preprocess.freddie.v11",
            "vintages": [name for name, _, _ in paths],
            "source_sha256": source_digests,
            "predicate_definition": (
                "13 reported dynamic primitive events: six ELTV transitions, "
                "four UPB/payment transitions, and three contract onsets. "
                "Forbearance, disaster-delinquency and exact 30/60-DPD monthly "
                "states are excluded from the search dictionary and enter only "
                "as strictly-future rolling-history baseline controls; no "
                "calendar, static, target-status or empirically tuned magnitude "
                "predicate is reportable"
            ),
            "baseline_control_definition": {
                "names": list(BASELINE_CONTROLS),
                "construction": (
                    "raw monthly forbearance, disaster-delinquency, 30-DPD and "
                    "60-DPD states; fixed identities, strictly-future rolling "
                    "histories with the same M-knot lag basis as reported rules "
                    "and a pre-registered positive direction; jointly refit in "
                    "every null and support model"
                ),
            },
            "reported_predicate_event_counts": {
                name: int(count)
                for name, count in zip(
                    PREDICATES, predicate_event_counts.tolist(), strict=True
                )
            },
            "reported_same_month_overlap": {
                PREDICATES[left]: {
                    PREDICATES[right]: int(predicate_same_month_overlap[left, right])
                    for right in range(left + 1, len(PREDICATES))
                    if predicate_same_month_overlap[left, right] > 0
                }
                for left in range(len(PREDICATES) - 1)
            },
            "environment_definition": "Freddie acquisition quarter",
            "partition_definition": (
                {
                    "development_vintages": sorted(available - held_out),
                    "development_assignment": "SHA-256(seed:loan_id) threshold",
                    "development_fit_fraction": development_fit_fraction,
                    "partition_seed": int(partition_seed),
                    "test_vintages": sorted(held_out),
                }
                if held_out
                else "not pre-assigned; Dataset.split fallback"
            ),
            "observation_definition": {
                "maximum_months": int(max_observation_months),
                "right_censoring": "last contiguous performance month",
                "absorbing_stops": ["first adverse target", "zero-balance termination"],
                "absorbing_row_included": True,
                "post_gap_rows_excluded": True,
            },
        },
    )
