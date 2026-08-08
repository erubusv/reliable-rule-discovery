from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..data import write_dataset


# Reported predicates are pre-registered, dynamic financial events. None uses
# application TARGET, delinquency state, overdue amount, payment lateness, or
# payment shortfall. Each directional behavioural channel is represented by a
# target-blind onset/recovery pair.  The dictionary deliberately spans loan
# demand, credit lifecycle, card usage, balance/payment state, limit pressure,
# and contract structure instead of multiplying near-duplicate state labels.
# The catalog spans six target-blind financial axes: credit demand, credit
# supply, debt burden, liquidity use, repayment, and contract lifecycle.
# Broad bureau events remain split into mutually exclusive product channels so
# one generic event cannot absorb every external-credit mechanism.
APPLICATION_PREDICATES = (
    "pred_cash_loan_approved",
    "pred_cash_loan_refused",
    "pred_consumer_loan_approved",
    "pred_consumer_loan_refused",
    "pred_revolving_credit_approved",
    "pred_revolving_credit_refused",
    "pred_requested_amount_increases",
    "pred_requested_amount_decreases",
    "pred_approved_term_lengthens",
    "pred_approved_term_shortens",
)
BUREAU_PREDICATES = (
    "pred_external_consumer_credit_opened",
    "pred_external_consumer_credit_closed",
    "pred_external_revolving_credit_opened",
    "pred_external_revolving_credit_closed",
    "pred_external_microloan_opened",
    "pred_external_microloan_closed",
)
CARD_PREDICATES = (
    "pred_card_credit_limit_increases",
    "pred_card_credit_limit_decreases",
    "pred_card_cash_withdrawal_starts",
    "pred_card_cash_withdrawal_stops",
    "pred_card_pos_purchase_starts",
    "pred_card_pos_purchase_stops",
    "pred_card_revolving_balance_starts",
    "pred_card_revolving_balance_clears",
    "pred_card_sustained_utilization_rise_starts",
    "pred_card_sustained_utilization_fall_starts",
    "pred_card_sustained_payment_rate_decline_starts",
    "pred_card_sustained_payment_rate_recovery_starts",
)
POS_PREDICATES = (
    "pred_pos_schedule_extended",
    "pred_pos_accelerated_amortization",
    "pred_pos_contract_completed",
    "pred_pos_active_contract_count_increases",
)
PREDICATES = (
    *APPLICATION_PREDICATES,
    *BUREAU_PREDICATES,
    *CARD_PREDICATES,
    *POS_PREDICATES,
)
PREDICATE_CODES = {name: code for code, name in enumerate(PREDICATES)}

PREDICATE_SOURCES = {
    **{name: "previous_application" for name in APPLICATION_PREDICATES},
    **{name: "bureau" for name in BUREAU_PREDICATES},
    **{name: "credit_card" for name in CARD_PREDICATES},
    **{name: "pos_cash" for name in POS_PREDICATES},
}

PREDICATE_DESCRIPTIONS = {
    "pred_cash_loan_approved": "A cash-loan application was approved.",
    "pred_cash_loan_refused": "A cash-loan application was refused.",
    "pred_consumer_loan_approved": "A consumer-loan application was approved.",
    "pred_consumer_loan_refused": "A consumer-loan application was refused.",
    "pred_revolving_credit_approved": (
        "A revolving-credit application was approved."
    ),
    "pred_revolving_credit_refused": (
        "A revolving-credit application was refused."
    ),
    "pred_requested_amount_increases": (
        "Requested credit amount increased from the preceding application "
        "of the same product type."
    ),
    "pred_requested_amount_decreases": (
        "Requested credit amount decreased from the preceding application "
        "of the same product type."
    ),
    "pred_approved_term_lengthens": (
        "The installment term of an approved application lengthened relative "
        "to the client's preceding approved application of the same product."
    ),
    "pred_approved_term_shortens": (
        "The installment term of an approved application shortened relative "
        "to the client's preceding approved application of the same product."
    ),
    "pred_external_consumer_credit_opened": (
        "An external consumer-credit account opened."
    ),
    "pred_external_consumer_credit_closed": (
        "An external consumer-credit account closed."
    ),
    "pred_external_revolving_credit_opened": (
        "An external revolving credit-card account opened."
    ),
    "pred_external_revolving_credit_closed": (
        "An external revolving credit-card account closed."
    ),
    "pred_external_microloan_opened": "An external microloan opened.",
    "pred_external_microloan_closed": "An external microloan closed.",
    "pred_card_credit_limit_increases": (
        "The observed credit-card limit increased from the preceding month."
    ),
    "pred_card_credit_limit_decreases": (
        "The observed credit-card limit decreased from the preceding month."
    ),
    "pred_card_cash_withdrawal_starts": (
        "A credit card switched from no cash drawing to cash drawing."
    ),
    "pred_card_cash_withdrawal_stops": (
        "A credit card switched from cash drawing to no cash drawing."
    ),
    "pred_card_pos_purchase_starts": (
        "A credit card switched from no POS drawing to POS drawing."
    ),
    "pred_card_pos_purchase_stops": (
        "A credit card switched from POS drawing to no POS drawing."
    ),
    "pred_card_revolving_balance_starts": (
        "A card moved from no revolving balance to a positive balance."
    ),
    "pred_card_revolving_balance_clears": (
        "A positive revolving card balance was cleared."
    ),
    "pred_card_sustained_utilization_rise_starts": (
        "Card utilization began a sustained two-month rise."
    ),
    "pred_card_sustained_utilization_fall_starts": (
        "Card utilization began a sustained two-month fall."
    ),
    "pred_card_sustained_payment_rate_decline_starts": (
        "Payment relative to the card limit began a sustained two-month decline."
    ),
    "pred_card_sustained_payment_rate_recovery_starts": (
        "Payment relative to the card limit began a sustained two-month recovery."
    ),
    "pred_pos_schedule_extended": (
        "The total or remaining POS/cash installment schedule increased."
    ),
    "pred_pos_accelerated_amortization": (
        "Remaining POS/cash installments fell by more than one in one month."
    ),
    "pred_pos_contract_completed": "A POS/cash contract entered completed status.",
    "pred_pos_active_contract_count_increases": (
        "The client's number of observed active POS/cash contracts increased."
    ),
}

BASELINE_CONTROLS: tuple[str, ...] = ()
BASELINE_CONTROL_ROLES: tuple[str, ...] = ()
ALL_PREDICATES = PREDICATES
ALL_PREDICATE_CODES = {name: code for code, name in enumerate(ALL_PREDICATES)}
PARTITION_NAMES = ("fit", "cert", "test")
RECURRENT_TARGET_SOURCES = ("bureau", "credit_card", "pos_cash")

RAW_FILES = (
    "application_train.csv",
    "application_test.csv",
    "previous_application.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
)
TARGET_COLUMNS = {
    "application_train.csv": ("TARGET",),
    "bureau_balance.csv": ("STATUS",),
    "POS_CASH_balance.csv": ("SK_DPD", "SK_DPD_DEF"),
    "credit_card_balance.csv": ("SK_DPD", "SK_DPD_DEF"),
}
FORBIDDEN_PREDICATE_FIELDS = (
    "application TARGET",
    "bureau_balance STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
    "CREDIT_DAY_OVERDUE",
    "AMT_CREDIT_MAX_OVERDUE",
    "AMT_CREDIT_SUM_OVERDUE",
    "DAYS_ENTRY_PAYMENT-DAYS_INSTALMENT",
    "AMT_INSTALMENT-AMT_PAYMENT",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_raw_root(root: Path) -> None:
    missing = [name for name in RAW_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing Home Credit raw files under {root}: {missing}"
        )


def _relative_month_from_days(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    month = np.floor_divide(numeric, 30.0)
    return pd.Series(month, index=values.index, dtype="float64")


def _valid_month(values: pd.Series, max_observation_months: int) -> pd.Series:
    return values.notna() & values.lt(0) & values.ge(-int(max_observation_months))


def _filter_clients(
    frame: pd.DataFrame, selected_clients: frozenset[int] | None
) -> pd.DataFrame:
    if selected_clients is None:
        return frame
    return frame.loc[frame["SK_ID_CURR"].isin(selected_clients)].copy()


def _application_clients(root: Path) -> np.ndarray:
    frames = [
        pd.read_csv(
            root / name,
            usecols=["SK_ID_CURR"],
            dtype={"SK_ID_CURR": np.int64},
            engine="c",
        )
        for name in ("application_train.csv", "application_test.csv")
    ]
    return np.unique(
        pd.concat(frames, ignore_index=True)["SK_ID_CURR"].to_numpy(dtype=np.int64)
    )


def _diagnostic_clients(
    clients: np.ndarray, *, limit: int | None, partition_seed: int
) -> frozenset[int] | None:
    if limit is None or limit >= len(clients):
        return None
    if limit < 3:
        raise ValueError("diagnostic_max_clients must be at least three")
    keyed = sorted(
        (
            hashlib.sha256(
                f"diagnostic:{int(partition_seed)}:{int(client)}".encode("utf-8")
            ).digest(),
            int(client),
        )
        for client in clients.tolist()
    )
    return frozenset(client for _, client in keyed[:limit])


def _event_frame(
    frame: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    predicate_code: int,
    *,
    month_column: str = "month",
) -> pd.DataFrame:
    if isinstance(mask, np.ndarray):
        mask = pd.Series(mask, index=frame.index)
    if not bool(mask.any()):
        return pd.DataFrame(
            {
                "client_id": pd.Series(dtype=np.int64),
                "month": pd.Series(dtype=np.int16),
                "predicate_code": pd.Series(dtype=np.int16),
                "primitive_event_id": pd.Series(dtype=np.int64),
            }
        )
    if "_primitive_event_id" not in frame.columns:
        raise ValueError("event frame is missing primitive-event provenance")
    result = frame.loc[mask, ["SK_ID_CURR", month_column, "_primitive_event_id"]].copy()
    result.columns = ["client_id", "month", "primitive_event_id"]
    result["client_id"] = result["client_id"].astype(np.int64)
    result["month"] = result["month"].astype(np.int16)
    result["predicate_code"] = np.int16(predicate_code)
    return result


def _assign_primitive_event_ids(
    frame: pd.DataFrame,
    *,
    namespace: str,
    key_columns: tuple[str, ...],
) -> None:
    """Attach a fast, audited raw-record identity to one event source."""

    if not key_columns or any(column not in frame.columns for column in key_columns):
        raise ValueError(f"invalid primitive key for {namespace}: {key_columns}")
    keys = frame.loc[:, list(key_columns)]
    hashed = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    salt = np.uint64(
        int.from_bytes(
            hashlib.sha256(namespace.encode("utf-8")).digest()[:8],
            byteorder="little",
        )
    )
    primitive = np.bitwise_xor(hashed, salt).view(np.int64)
    # A collision would incorrectly turn distinct financial events into one
    # compound event.  Fail closed instead of accepting that ambiguity.
    if len(primitive):
        audit = keys.copy()
        audit["_primitive_event_id"] = primitive
        distinct = audit.drop_duplicates()
        if bool(distinct.duplicated("_primitive_event_id", keep=False).any()):
            raise RuntimeError(f"primitive-event hash collision in {namespace}")
    frame["_primitive_event_id"] = primitive


def _observed_clients(frame: pd.DataFrame) -> np.ndarray:
    return np.unique(frame["SK_ID_CURR"].to_numpy(dtype=np.int64))


def _account_month_counts(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Count observable contracts without using delinquency or future values."""

    result = (
        frame.groupby(["SK_ID_CURR", "month"], sort=True)["SK_ID_PREV"]
        .nunique()
        .rename("account_count")
        .reset_index()
    )
    result.columns = ["client_id", "month", "account_count"]
    result["client_id"] = result["client_id"].astype(np.int64)
    result["month"] = result["month"].astype(np.int16)
    result["source"] = str(source)
    return result


def _recurrent_30dpd_state(
    frame: pd.DataFrame,
    *,
    account_column: str,
    client_column: str,
    month_column: str,
    serious: pd.Series | np.ndarray,
    eligible: pd.Series | np.ndarray,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build account-level recurrent 30+ DPD onsets and client risk cells.

    An account is at risk while its current observed state is non-serious.  A
    contiguous transition from an eligible non-serious month to a serious
    month contributes one onset and keeps that transition row exposed.  A
    persistent serious episode contributes neither additional targets nor
    exposure; after an observed recovery the account re-enters the risk set.
    Left-boundary/gap serious states are treated as prevalent, never incident.
    """

    required = {account_column, client_column, month_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing recurrent-state columns: {missing}")
    work = frame[[account_column, client_column, month_column]].copy()
    work["_serious"] = np.asarray(serious, dtype=bool)
    work["_eligible"] = np.asarray(eligible, dtype=bool)
    work = work.sort_values([account_column, month_column], kind="stable").reset_index(
        drop=True
    )
    if bool(work.duplicated([account_column, month_column]).any()):
        raise ValueError(f"duplicated {source} account-month in recurrent target state")

    same_account = work[account_column].eq(work[account_column].shift())
    contiguous = same_account & work[month_column].eq(
        work[month_column].shift() + 1
    )
    prior_eligible = work["_eligible"].shift(fill_value=False).astype(bool)
    prior_serious = work["_serious"].shift(fill_value=False).astype(bool)
    onset = (
        work["_eligible"]
        & work["_serious"]
        & contiguous
        & prior_eligible
        & ~prior_serious
    )
    # The transition month is an opportunity row even though its end-of-month
    # state is serious.  Persistent serious months are not new-onset chances.
    at_risk = work["_eligible"] & (~work["_serious"] | onset)

    onset_rows = work.loc[onset, [client_column, month_column]].copy()
    onset_rows.columns = ["client_id", "month"]
    if len(onset_rows):
        targets = (
            onset_rows.groupby(["client_id", "month"], sort=True)
            .size()
            .rename("multiplicity")
            .reset_index()
        )
    else:
        targets = pd.DataFrame(
            {
                "client_id": pd.Series(dtype=np.int64),
                "month": pd.Series(dtype=np.int16),
                "multiplicity": pd.Series(dtype=np.int32),
            }
        )
    targets = targets.astype(
        {"client_id": np.int64, "month": np.int16, "multiplicity": np.int32}
    )

    risk_rows = work.loc[at_risk, [client_column, month_column, account_column]].copy()
    risk_rows.columns = ["client_id", "month", "account_id"]
    if len(risk_rows):
        risk = (
            risk_rows.groupby(["client_id", "month"], sort=True)["account_id"]
            .nunique()
            .rename("account_count")
            .reset_index()
        )
    else:
        risk = pd.DataFrame(
            {
                "client_id": pd.Series(dtype=np.int64),
                "month": pd.Series(dtype=np.int16),
                "account_count": pd.Series(dtype=np.int16),
            }
        )
    risk = risk.astype(
        {"client_id": np.int64, "month": np.int16, "account_count": np.int32}
    )
    risk["source"] = str(source)
    return targets, risk, {
        "onset_events": int(targets["multiplicity"].sum()),
        "onset_client_months": int(len(targets)),
        "onset_clients": int(targets["client_id"].nunique()),
        "risk_client_months": int(len(risk)),
        "persistent_serious_account_months_excluded": int(
            np.count_nonzero(work["_eligible"] & work["_serious"] & ~onset)
        ),
        "prevalent_or_gap_serious_account_months": int(
            np.count_nonzero(
                work["_eligible"]
                & work["_serious"]
                & ~(contiguous & prior_eligible)
            )
        ),
    }


def _unified_client_recurrent_state(
    serious_frames: Iterable[pd.DataFrame],
    risk_frames: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Collapse all credit sources into one recurrent client distress state."""

    serious_parts = [
        frame[["client_id", "month"]] for frame in serious_frames if len(frame)
    ]
    risk_parts = [frame[["client_id", "month"]] for frame in risk_frames if len(frame)]
    if not serious_parts or not risk_parts:
        raise ValueError("unified recurrent target requires observed risk and 30+ DPD")
    serious = pd.concat(serious_parts, ignore_index=True).drop_duplicates()
    serious["_serious"] = True
    eligible = pd.concat([*risk_parts, serious[["client_id", "month"]]], ignore_index=True)
    eligible = eligible.drop_duplicates().merge(
        serious,
        on=["client_id", "month"],
        how="left",
        validate="one_to_one",
    )
    eligible["_serious"] = eligible["_serious"].eq(True)
    eligible = eligible.sort_values(["client_id", "month"], kind="stable").reset_index(
        drop=True
    )
    contiguous = eligible["client_id"].eq(eligible["client_id"].shift()) & eligible[
        "month"
    ].eq(eligible["month"].shift() + 1)
    prior_serious = eligible["_serious"].shift(fill_value=False).astype(bool)
    onset = eligible["_serious"] & contiguous & ~prior_serious
    # A recovery row immediately re-opens follow-up. A serious state observed
    # after an unobserved gap is prevalent rather than a fabricated onset.
    at_risk = ~eligible["_serious"] | onset

    targets = eligible.loc[onset, ["client_id", "month"]].copy()
    targets["multiplicity"] = np.int32(1)
    targets = targets.astype(
        {"client_id": np.int64, "month": np.int16, "multiplicity": np.int32}
    )
    risk = eligible.loc[at_risk, ["client_id", "month"]].copy()
    risk["account_count"] = np.int32(1)
    risk["source"] = "unified"
    risk = risk.astype(
        {"client_id": np.int64, "month": np.int16, "account_count": np.int32}
    )
    return targets, risk, {
        "onset_events": int(len(targets)),
        "onset_clients": int(targets["client_id"].nunique()),
        "risk_client_months": int(len(risk)),
        "persistent_distress_client_months_excluded": int(
            np.count_nonzero(eligible["_serious"] & ~onset)
        ),
        "recovery_client_months": int(
            np.count_nonzero(contiguous & prior_serious & ~eligible["_serious"])
        ),
    }


def _read_previous_application(
    root: Path,
    selected_clients: frozenset[int] | None,
    max_observation_months: int,
) -> tuple[list[pd.DataFrame], np.ndarray, dict[str, object]]:
    frame = pd.read_csv(
        root / "previous_application.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "DAYS_DECISION",
            "NAME_CONTRACT_TYPE",
            "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION",
            "CNT_PAYMENT",
        ],
        dtype={
            "SK_ID_CURR": np.int64,
            "NAME_CONTRACT_TYPE": "string",
            "NAME_CONTRACT_STATUS": "string",
        },
        engine="c",
    )
    frame = _filter_clients(frame, selected_clients)
    frame["month"] = _relative_month_from_days(frame["DAYS_DECISION"])
    frame = frame.loc[_valid_month(frame["month"], max_observation_months)].copy()
    frame = frame.sort_values(
        ["SK_ID_CURR", "NAME_CONTRACT_TYPE", "DAYS_DECISION", "SK_ID_PREV"],
        kind="stable",
    ).reset_index(drop=True)
    _assign_primitive_event_ids(
        frame,
        namespace="home_credit.previous_application",
        key_columns=("SK_ID_PREV",),
    )
    contract = frame["NAME_CONTRACT_TYPE"].str.strip()
    status = frame["NAME_CONTRACT_STATUS"].str.strip()
    contract_predicates = (
        ("Cash loans", "Approved", "pred_cash_loan_approved"),
        ("Cash loans", "Refused", "pred_cash_loan_refused"),
        ("Consumer loans", "Approved", "pred_consumer_loan_approved"),
        ("Consumer loans", "Refused", "pred_consumer_loan_refused"),
        ("Revolving loans", "Approved", "pred_revolving_credit_approved"),
        ("Revolving loans", "Refused", "pred_revolving_credit_refused"),
    )
    events = [
        _event_frame(
            frame,
            contract.eq(contract_type) & status.eq(outcome),
            PREDICATE_CODES[name],
        )
        for contract_type, outcome, name in contract_predicates
    ]
    amount = pd.to_numeric(frame["AMT_APPLICATION"], errors="coerce")
    preceding_amount = amount.shift()
    preceding_client = frame["SK_ID_CURR"].shift()
    preceding_contract = contract.shift()
    preceding_day = pd.to_numeric(frame["DAYS_DECISION"], errors="coerce").shift()
    comparable = (
        frame["SK_ID_CURR"].eq(preceding_client)
        & contract.eq(preceding_contract)
        & pd.to_numeric(frame["DAYS_DECISION"], errors="coerce").gt(preceding_day)
        & amount.gt(0.0)
        & preceding_amount.gt(0.0)
    )
    events.extend(
        (
            _event_frame(
                frame,
                comparable & amount.gt(preceding_amount),
                PREDICATE_CODES["pred_requested_amount_increases"],
            ),
            _event_frame(
                frame,
                comparable & amount.lt(preceding_amount),
                PREDICATE_CODES["pred_requested_amount_decreases"],
            ),
        )
    )
    term = pd.to_numeric(frame["CNT_PAYMENT"], errors="coerce")
    approved = status.eq("Approved") & term.gt(0.0)
    product_keys = [frame["SK_ID_CURR"], contract]
    last_approved_term = term.where(approved).groupby(product_keys).ffill()
    preceding_approved_term = last_approved_term.groupby(product_keys).shift()
    events.extend(
        (
            _event_frame(
                frame,
                approved
                & preceding_approved_term.gt(0.0)
                & term.gt(preceding_approved_term),
                PREDICATE_CODES["pred_approved_term_lengthens"],
            ),
            _event_frame(
                frame,
                approved
                & preceding_approved_term.gt(0.0)
                & term.lt(preceding_approved_term),
                PREDICATE_CODES["pred_approved_term_shortens"],
            ),
        )
    )
    audit = {
        "rows_in_window": int(len(frame)),
        "clients_in_window": int(frame["SK_ID_CURR"].nunique()),
        "contract_type_counts": {
            str(key): int(value)
            for key, value in contract.value_counts(dropna=False).items()
        },
        "contract_status_counts": {
            str(key): int(value)
            for key, value in status.value_counts(dropna=False).items()
        },
    }
    return events, _observed_clients(frame), audit


def _read_bureau(
    root: Path,
    selected_clients: frozenset[int] | None,
    max_observation_months: int,
) -> tuple[
    list[pd.DataFrame],
    np.ndarray,
    pd.Series,
    dict[str, object],
]:
    frame = pd.read_csv(
        root / "bureau.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_BUREAU",
            "DAYS_CREDIT",
            "DAYS_ENDDATE_FACT",
            "CREDIT_TYPE",
        ],
        dtype={
            "SK_ID_CURR": np.int64,
            "SK_ID_BUREAU": np.int64,
            "CREDIT_TYPE": "string",
        },
        engine="c",
    )
    frame = _filter_clients(frame, selected_clients)
    bureau_to_client = (
        frame[["SK_ID_BUREAU", "SK_ID_CURR"]]
        .drop_duplicates("SK_ID_BUREAU")
        .set_index("SK_ID_BUREAU")["SK_ID_CURR"]
        .astype(np.int64)
    )
    if len(bureau_to_client) != frame["SK_ID_BUREAU"].nunique():
        raise ValueError("one bureau account maps to multiple Home Credit clients")

    opened = frame[["SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT", "CREDIT_TYPE"]].copy()
    opened["month"] = _relative_month_from_days(opened["DAYS_CREDIT"])
    opened = opened.loc[_valid_month(opened["month"], max_observation_months)].copy()
    _assign_primitive_event_ids(
        opened,
        namespace="home_credit.bureau.open",
        key_columns=("SK_ID_BUREAU",),
    )
    opened_type = opened["CREDIT_TYPE"].str.strip()
    events = [
        _event_frame(
            opened,
            opened_type.eq("Consumer credit"),
            PREDICATE_CODES["pred_external_consumer_credit_opened"],
        ),
        _event_frame(
            opened,
            opened_type.eq("Credit card"),
            PREDICATE_CODES["pred_external_revolving_credit_opened"],
        ),
        _event_frame(
            opened,
            opened_type.eq("Microloan"),
            PREDICATE_CODES["pred_external_microloan_opened"],
        ),
    ]

    closed = frame[
        ["SK_ID_CURR", "SK_ID_BUREAU", "DAYS_ENDDATE_FACT", "CREDIT_TYPE"]
    ].copy()
    closed["month"] = _relative_month_from_days(closed["DAYS_ENDDATE_FACT"])
    closed = closed.loc[_valid_month(closed["month"], max_observation_months)].copy()
    _assign_primitive_event_ids(
        closed,
        namespace="home_credit.bureau.close",
        key_columns=("SK_ID_BUREAU", "DAYS_ENDDATE_FACT"),
    )
    closed_type = closed["CREDIT_TYPE"].str.strip()
    events.extend(
        (
            _event_frame(
                closed,
                closed_type.eq("Consumer credit"),
                PREDICATE_CODES["pred_external_consumer_credit_closed"],
            ),
            _event_frame(
                closed,
                closed_type.eq("Credit card"),
                PREDICATE_CODES["pred_external_revolving_credit_closed"],
            ),
            _event_frame(
                closed,
                closed_type.eq("Microloan"),
                PREDICATE_CODES["pred_external_microloan_closed"],
            ),
        )
    )
    # Bureau monthly histories can contain a target even when account opening
    # predates the window and no closure occurs inside it.  Those clients still
    # belong to the client-level risk set.
    observed = _observed_clients(frame)
    audit = {
        "accounts": int(len(frame)),
        "clients_with_bureau_history": int(len(observed)),
        "opened_credit_type_counts": {
            str(key): int(value)
            for key, value in opened_type.value_counts(dropna=False).items()
        },
        "actual_closures_in_window": int(len(closed)),
    }
    return events, observed, bureau_to_client, audit


def _read_bureau_targets(
    root: Path,
    bureau_to_client: pd.Series,
    max_observation_months: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    pieces: list[pd.DataFrame] = []
    observation_pieces: list[pd.DataFrame] = []
    state_pieces: list[pd.DataFrame] = []
    rows_in_window = 0
    serious_rows = 0
    for chunk in pd.read_csv(
        root / "bureau_balance.csv",
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
        dtype={
            "SK_ID_BUREAU": np.int64,
            "MONTHS_BALANCE": np.int16,
            "STATUS": "string",
        },
        chunksize=2_000_000,
        engine="c",
    ):
        chunk["SK_ID_CURR"] = chunk["SK_ID_BUREAU"].map(bureau_to_client)
        keep = chunk["SK_ID_CURR"].notna() & _valid_month(
            chunk["MONTHS_BALANCE"], max_observation_months
        )
        chunk = chunk.loc[keep]
        rows_in_window += len(chunk)
        state = chunk[
            ["SK_ID_BUREAU", "SK_ID_CURR", "MONTHS_BALANCE", "STATUS"]
        ].copy()
        state.columns = ["account_id", "client_id", "month", "status"]
        state_pieces.append(state)
        observed = (
            chunk.groupby(["SK_ID_CURR", "MONTHS_BALANCE"], sort=True)["SK_ID_BUREAU"]
            .nunique()
            .rename("account_count")
            .reset_index()
        )
        observed.columns = ["client_id", "month", "account_count"]
        observation_pieces.append(observed)
        serious = chunk["STATUS"].isin(("2", "3", "4", "5"))
        serious_rows += int(serious.sum())
        if bool(serious.any()):
            part = chunk.loc[serious, ["SK_ID_CURR", "MONTHS_BALANCE"]].copy()
            part.columns = ["client_id", "month"]
            pieces.append(part)
    if pieces:
        targets = pd.concat(pieces, ignore_index=True)
        targets["client_id"] = targets["client_id"].astype(np.int64)
        targets["month"] = targets["month"].astype(np.int16)
    else:
        targets = pd.DataFrame(
            {
                "client_id": pd.Series(dtype=np.int64),
                "month": pd.Series(dtype=np.int16),
            }
        )
    observations = pd.concat(observation_pieces, ignore_index=True)
    observations = (
        observations.groupby(["client_id", "month"], sort=True)["account_count"]
        .sum()
        .reset_index()
    )
    observations["client_id"] = observations["client_id"].astype(np.int64)
    observations["month"] = observations["month"].astype(np.int16)
    observations["source"] = "bureau"
    states = pd.concat(state_pieces, ignore_index=True)
    status = states["status"].astype("string").str.strip()
    recurrent_targets, recurrent_risk, recurrent_audit = _recurrent_30dpd_state(
        states,
        account_column="account_id",
        client_column="client_id",
        month_column="month",
        serious=status.isin(("2", "3", "4", "5")),
        # C is closed and X is unknown; neither is a delinquency opportunity.
        eligible=status.isin(("0", "1", "2", "3", "4", "5")),
        source="bureau",
    )
    return (
        targets,
        observations,
        recurrent_targets,
        recurrent_risk,
        {
            "mapped_rows_in_window": int(rows_in_window),
            "serious_30plus_rows": int(serious_rows),
            "target_clients": int(targets["client_id"].nunique()),
            "recurrent_onset": recurrent_audit,
        },
    )


def _account_contiguous(frame: pd.DataFrame, lag: int = 1) -> pd.Series:
    return (
        frame["SK_ID_PREV"].eq(frame["SK_ID_PREV"].shift(lag))
        & frame["month"].eq(frame["month"].shift(lag) + lag)
    ).fillna(False)


def _sustained_direction_onset(
    frame: pd.DataFrame, values: pd.Series, *, positive: bool
) -> pd.Series:
    """Detect the first month of a two-step directional run.

    Requiring two consecutive moves and a preceding non-move prevents monthly
    measurement noise from emitting a new event at every sign flip.  The rule
    is symmetric for rises and falls and uses only the current and past rows.
    """

    previous = values.shift()
    previous_two = values.shift(2)
    previous_three = values.shift(3)
    current_delta = values - previous
    previous_delta = previous - previous_two
    preceding_delta = previous_two - previous_three
    valid = (
        _account_contiguous(frame, 3)
        & values.notna()
        & previous.notna()
        & previous_two.notna()
        & previous_three.notna()
    )
    if positive:
        return (
            valid
            & current_delta.gt(0.0)
            & previous_delta.gt(0.0)
            & preceding_delta.le(0.0)
        )
    return (
        valid
        & current_delta.lt(0.0)
        & previous_delta.lt(0.0)
        & preceding_delta.ge(0.0)
    )


def _credit_card_predicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Build card predicates without reading either delinquency column."""
    required = {
        "SK_ID_PREV",
        "SK_ID_CURR",
        "month",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_OTHER_CURRENT",
        "AMT_DRAWINGS_POS_CURRENT",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing card predicate columns: {missing}")
    previous = _account_contiguous(frame)
    cash = (
        pd.to_numeric(frame["AMT_DRAWINGS_ATM_CURRENT"], errors="coerce")
        .fillna(0.0)
        .add(
            pd.to_numeric(frame["AMT_DRAWINGS_OTHER_CURRENT"], errors="coerce").fillna(
                0.0
            )
        )
    )
    previous_cash = cash.shift().where(previous)
    pos = pd.to_numeric(frame["AMT_DRAWINGS_POS_CURRENT"], errors="coerce").fillna(0.0)
    previous_pos = pos.shift().where(previous)
    limit = pd.to_numeric(frame["AMT_CREDIT_LIMIT_ACTUAL"], errors="coerce")
    balance = pd.to_numeric(frame["AMT_BALANCE"], errors="coerce")
    payment = pd.to_numeric(frame["AMT_PAYMENT_TOTAL_CURRENT"], errors="coerce")
    utilization = balance.div(limit.where(limit.gt(0.0)))
    nonnegative_payment = payment.where(payment.ge(0.0))
    # Normalize payment effort by the contemporaneous contractual limit, not
    # by balance.  payment/balance is mechanically the inverse of utilization
    # and made the two reported channels observationally redundant.
    payment_rate = nonnegative_payment.div(limit.where(limit.gt(0.0)))

    out = pd.DataFrame(index=frame.index)
    previous_limit = limit.shift().where(previous)
    out["pred_card_credit_limit_increases"] = (
        previous & limit.notna() & previous_limit.notna() & limit.gt(previous_limit)
    )
    out["pred_card_credit_limit_decreases"] = (
        previous & limit.notna() & previous_limit.notna() & limit.lt(previous_limit)
    )
    out["pred_card_cash_withdrawal_starts"] = (
        previous & cash.gt(0.0) & previous_cash.le(0.0)
    )
    out["pred_card_cash_withdrawal_stops"] = (
        previous & previous_cash.gt(0.0) & cash.le(0.0)
    )
    out["pred_card_pos_purchase_starts"] = previous & pos.gt(0.0) & previous_pos.le(0.0)
    out["pred_card_pos_purchase_stops"] = previous & pos.le(0.0) & previous_pos.gt(0.0)
    out["pred_card_revolving_balance_starts"] = (
        previous & balance.gt(0.0) & balance.shift().where(previous).le(0.0)
    )
    out["pred_card_revolving_balance_clears"] = (
        previous & balance.le(0.0) & balance.shift().where(previous).gt(0.0)
    )
    out["pred_card_sustained_utilization_rise_starts"] = _sustained_direction_onset(
        frame, utilization, positive=True
    )
    out["pred_card_sustained_utilization_fall_starts"] = _sustained_direction_onset(
        frame, utilization, positive=False
    )
    out["pred_card_sustained_payment_rate_decline_starts"] = (
        _sustained_direction_onset(
        frame, payment_rate, positive=False
        )
    )
    out["pred_card_sustained_payment_rate_recovery_starts"] = (
        _sustained_direction_onset(
        frame, payment_rate, positive=True
        )
    )
    return out.fillna(False).astype(np.uint8)


def _read_credit_card(
    root: Path,
    selected_clients: frozenset[int] | None,
    max_observation_months: int,
) -> tuple[
    list[pd.DataFrame],
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    frame = pd.read_csv(
        root / "credit_card_balance.csv",
        usecols=[
            "SK_ID_PREV",
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_PAYMENT_TOTAL_CURRENT",
            "AMT_DRAWINGS_ATM_CURRENT",
            "AMT_DRAWINGS_OTHER_CURRENT",
            "AMT_DRAWINGS_POS_CURRENT",
            "SK_DPD",
        ],
        dtype={
            "SK_ID_PREV": np.int64,
            "SK_ID_CURR": np.int64,
            "MONTHS_BALANCE": np.int16,
            "SK_DPD": np.int32,
        },
        engine="c",
    )
    frame = _filter_clients(frame, selected_clients)
    frame["month"] = frame["MONTHS_BALANCE"].astype(np.int16)
    frame = frame.loc[_valid_month(frame["month"], max_observation_months)].copy()
    frame = frame.sort_values(["SK_ID_PREV", "month"], kind="stable").reset_index(
        drop=True
    )
    if bool(frame.duplicated(["SK_ID_PREV", "month"]).any()):
        raise ValueError("duplicated Home Credit credit-card account-month")
    _assign_primitive_event_ids(
        frame,
        namespace="home_credit.credit_card.account_month",
        key_columns=("SK_ID_PREV", "month"),
    )

    matrix = _credit_card_predicates(frame)
    events = [
        _event_frame(
            frame,
            matrix[name].astype(bool),
            PREDICATE_CODES[name],
        )
        for name in CARD_PREDICATES
    ]
    target_mask = frame["SK_DPD"].ge(30)
    targets = frame.loc[target_mask, ["SK_ID_CURR", "month"]].copy()
    targets.columns = ["client_id", "month"]
    recurrent_targets, recurrent_risk, recurrent_audit = _recurrent_30dpd_state(
        frame,
        account_column="SK_ID_PREV",
        client_column="SK_ID_CURR",
        month_column="month",
        serious=target_mask,
        eligible=np.ones(len(frame), dtype=bool),
        source="credit_card",
    )
    audit = {
        "rows_in_window": int(len(frame)),
        "clients_in_window": int(frame["SK_ID_CURR"].nunique()),
        "serious_30plus_rows": int(target_mask.sum()),
        "target_clients": int(targets["client_id"].nunique()),
        "recurrent_onset": recurrent_audit,
    }
    observations = _account_month_counts(frame, source="credit_card")
    return (
        events,
        _observed_clients(frame),
        targets,
        observations,
        recurrent_targets,
        recurrent_risk,
        audit,
    )


def _pos_predicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Build POS predicates without reading either delinquency column."""
    remaining = pd.to_numeric(frame["CNT_INSTALMENT_FUTURE"], errors="coerce")
    total = pd.to_numeric(frame["CNT_INSTALMENT"], errors="coerce")
    contiguous = _account_contiguous(frame)
    previous_remaining = remaining.shift().where(contiguous)
    previous_total = total.shift().where(contiguous)
    status = frame["NAME_CONTRACT_STATUS"].str.strip()
    previous_status = status.shift().where(contiguous)
    out = pd.DataFrame(index=frame.index)
    out["pred_pos_schedule_extended"] = (
        contiguous
        & (
            (total.notna() & previous_total.notna() & total.gt(previous_total))
            | (
                remaining.notna()
                & previous_remaining.notna()
                & remaining.gt(previous_remaining)
            )
        )
    )
    out["pred_pos_accelerated_amortization"] = (
        contiguous
        & status.eq("Active")
        & previous_status.eq("Active")
        & remaining.notna()
        & previous_remaining.notna()
        & previous_remaining.sub(remaining).gt(1.0)
    )
    out["pred_pos_contract_completed"] = (
        status.eq("Completed")
        & previous_status.notna()
        & previous_status.ne("Completed")
    )
    return out.fillna(False).astype(np.uint8)


def _pos_active_count_events(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Create client-month contract-count transitions with aggregate provenance."""

    active = frame["NAME_CONTRACT_STATUS"].str.strip().eq("Active")
    observed = frame[["SK_ID_CURR", "month"]].drop_duplicates()
    active_counts = (
        frame.loc[active]
        .groupby(["SK_ID_CURR", "month"], sort=True)["SK_ID_PREV"]
        .nunique()
        .rename("active_count")
        .reset_index()
    )
    counts = (
        observed.merge(
            active_counts,
            on=["SK_ID_CURR", "month"],
            how="left",
            validate="one_to_one",
        )
        .fillna({"active_count": 0})
        .sort_values(["SK_ID_CURR", "month"], kind="stable")
        .reset_index(drop=True)
    )
    preceding = counts["active_count"].shift()
    contiguous = counts["SK_ID_CURR"].eq(counts["SK_ID_CURR"].shift()) & counts[
        "month"
    ].eq(counts["month"].shift() + 1)
    counts["_primitive_event_id"] = np.int64(0)
    _assign_primitive_event_ids(
        counts,
        namespace="home_credit.pos_cash.client_active_count_month",
        key_columns=("SK_ID_CURR", "month"),
    )
    return [
        _event_frame(
            counts,
            contiguous & counts["active_count"].gt(preceding),
            PREDICATE_CODES["pred_pos_active_contract_count_increases"],
        )
    ]


def _read_pos_cash(
    root: Path,
    selected_clients: frozenset[int] | None,
    max_observation_months: int,
) -> tuple[
    list[pd.DataFrame],
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    frame = pd.read_csv(
        root / "POS_CASH_balance.csv",
        usecols=[
            "SK_ID_PREV",
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "CNT_INSTALMENT",
            "CNT_INSTALMENT_FUTURE",
            "NAME_CONTRACT_STATUS",
            "SK_DPD",
        ],
        dtype={
            "SK_ID_PREV": np.int64,
            "SK_ID_CURR": np.int64,
            "MONTHS_BALANCE": np.int16,
            "SK_DPD": np.int32,
            "NAME_CONTRACT_STATUS": "string",
        },
        engine="c",
    )
    frame = _filter_clients(frame, selected_clients)
    frame["month"] = frame["MONTHS_BALANCE"].astype(np.int16)
    frame = frame.loc[_valid_month(frame["month"], max_observation_months)].copy()
    frame = frame.sort_values(["SK_ID_PREV", "month"], kind="stable").reset_index(
        drop=True
    )
    if bool(frame.duplicated(["SK_ID_PREV", "month"]).any()):
        raise ValueError("duplicated Home Credit POS/cash account-month")
    _assign_primitive_event_ids(
        frame,
        namespace="home_credit.pos_cash.account_month",
        key_columns=("SK_ID_PREV", "month"),
    )

    matrix = _pos_predicates(frame)
    events = [
        _event_frame(
            frame,
            matrix[name].astype(bool),
            PREDICATE_CODES[name],
        )
        for name in POS_PREDICATES
        if name
        not in {
            "pred_pos_active_contract_count_increases",
            "pred_pos_active_contract_count_decreases",
        }
    ]
    events.extend(_pos_active_count_events(frame))
    target_mask = frame["SK_DPD"].ge(30)
    targets = frame.loc[target_mask, ["SK_ID_CURR", "month"]].copy()
    targets.columns = ["client_id", "month"]
    recurrent_targets, recurrent_risk, recurrent_audit = _recurrent_30dpd_state(
        frame,
        account_column="SK_ID_PREV",
        client_column="SK_ID_CURR",
        month_column="month",
        serious=target_mask,
        # The transition row remains eligible even if the servicing status
        # changes at the same month (for example Active -> Demand).  Closed
        # non-serious rows do not re-enter the risk set.
        eligible=(
            frame["NAME_CONTRACT_STATUS"].str.strip().eq("Active") | target_mask
        ),
        source="pos_cash",
    )
    audit = {
        "rows_in_window": int(len(frame)),
        "clients_in_window": int(frame["SK_ID_CURR"].nunique()),
        "serious_30plus_rows": int(target_mask.sum()),
        "target_clients": int(targets["client_id"].nunique()),
        "recurrent_onset": recurrent_audit,
    }
    observations = _account_month_counts(frame, source="pos_cash")
    return (
        events,
        _observed_clients(frame),
        targets,
        observations,
        recurrent_targets,
        recurrent_risk,
        audit,
    )


def _fixed_client_partition(
    client_ids: np.ndarray,
    *,
    seed: int,
    fractions: tuple[float, float, float],
) -> dict[int, int]:
    clients = np.unique(np.asarray(client_ids, dtype=np.int64))
    values = np.asarray(fractions, dtype=np.float64)
    if (
        values.shape != (3,)
        or np.any(values <= 0.0)
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError(
            "partition fractions must be three positive values summing to one"
        )
    first = int(float(values[0]) * (1 << 64))
    second = int(float(values[:2].sum()) * (1 << 64))
    result: dict[int, int] = {}
    for client in clients.tolist():
        value = int.from_bytes(
            hashlib.sha256(f"{int(seed)}:{int(client)}".encode("utf-8")).digest()[:8],
            byteorder="big",
        )
        result[int(client)] = 0 if value < first else (1 if value < second else 2)
    if set(result.values()) != {0, 1, 2}:
        raise ValueError("client partition produced an empty fit/cert/test split")
    return result


def _build_baseline_cells(
    entities: pd.DataFrame,
    entity_codes: pd.Series,
    observation_frames: Iterable[pd.DataFrame],
    *,
    max_observation_months: int,
    target_risk_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build target-blind age x source-opportunity nuisance states."""

    sources = ("bureau", "credit_card", "pos_cash")
    source_index = {name: index for index, name in enumerate(sources)}
    lengths = entities["end_time"].to_numpy(dtype=np.int64) + 1
    offsets = np.zeros(len(entities) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    total = int(offsets[-1])
    cell_entities = np.repeat(np.arange(len(entities), dtype=np.int32), lengths)
    cell_times = np.arange(total, dtype=np.int64) - offsets[cell_entities]
    counts = np.zeros((len(sources), total), dtype=np.uint8)
    retained_clients = frozenset(entity_codes.index.to_numpy(dtype=np.int64).tolist())
    for frame in observation_frames:
        if len(frame) == 0:
            continue
        source_values = frame["source"].astype(str).unique()
        if len(source_values) != 1 or source_values[0] not in source_index:
            raise ValueError("invalid Home Credit observation source")
        current = frame.loc[frame["client_id"].isin(retained_clients)].copy()
        if len(current) == 0:
            continue
        local = current["client_id"].map(entity_codes).to_numpy(dtype=np.int64)
        ticks = current["month"].to_numpy(dtype=np.int64) + int(max_observation_months)
        end = entities["end_time"].to_numpy(dtype=np.int64)[local]
        keep = (ticks >= 0) & (ticks <= end)
        flat = offsets[local[keep]] + ticks[keep]
        category = np.minimum(
            current["account_count"].to_numpy(dtype=np.int64)[keep], 2
        ).astype(np.uint8)
        np.maximum.at(counts[source_index[source_values[0]]], flat, category)
    age = np.minimum(cell_times // 12, 2).astype(np.int16)
    structural_exposure = np.any(counts > 0, axis=0)
    if target_risk_frame is None:
        exposure = structural_exposure
    else:
        exposure = np.zeros(total, dtype=bool)
        current = target_risk_frame.loc[
            target_risk_frame["client_id"].isin(retained_clients)
        ]
        local = current["client_id"].map(entity_codes).to_numpy(dtype=np.int64)
        ticks = (
            current["month"].to_numpy(dtype=np.int64)
            + int(max_observation_months)
        )
        keep = (ticks >= 0) & (ticks < lengths[local])
        flat = offsets[local[keep]] + ticks[keep]
        exposure[flat] = True
        if np.any(exposure & ~structural_exposure):
            raise AssertionError("target-source risk lies outside observed source history")
    source_mask = (
        (counts[0] > 0).astype(np.int16)
        + 2 * (counts[1] > 0).astype(np.int16)
        + 4 * (counts[2] > 0).astype(np.int16)
    )
    raw_state = 8 * age + source_mask
    if not bool(np.any(exposure)):
        raise ValueError("no observable Home Credit target opportunities")
    raw_values = np.unique(raw_state[exposure])
    lookup = np.full(24, -1, dtype=np.int16)
    lookup[raw_values] = np.arange(len(raw_values), dtype=np.int16)
    compact = lookup[raw_state]
    compact[~exposure] = 0
    if np.any(compact < 0):
        raise AssertionError("exposed baseline state was not compacted")
    baseline_cells = pd.DataFrame(
        {
            "entity_code": cell_entities,
            "time": cell_times,
            "baseline_stratum": compact,
            "exposure": exposure.astype(np.uint8),
        }
    )
    state_map = []
    for compact_code, raw in enumerate(raw_values.tolist()):
        source_state = int(raw) % 8
        state_map.append(
            {
                "baseline_stratum": compact_code,
                "history_age_bin": int(raw) // 8,
                "bureau_observed": bool(source_state & 1),
                "credit_card_observed": bool(source_state & 2),
                "pos_cash_observed": bool(source_state & 4),
            }
        )
    return baseline_cells, {
        "exposed_cells": int(np.count_nonzero(exposure)),
        "unexposed_cells": int(np.count_nonzero(~exposure)),
        "structurally_observed_cells": int(np.count_nonzero(structural_exposure)),
        "target_source_risk_conditioned": target_risk_frame is not None,
        "sources": list(sources),
        "source_state": "three-bit bureau/credit-card/POS observation mask",
        "history_age_bins_months": [[0, 12], [12, 24], [24, 36]],
        "observed_strata": state_map,
    }


def _assert_distinct_predicate_streams(events: pd.DataFrame) -> None:
    reported = events.loc[
        events["predicate_code"].lt(len(PREDICATES)),
        ["entity_code", "time", "predicate_code"],
    ]
    signatures: dict[tuple[int, bytes], list[int]] = {}
    for code, group in reported.groupby("predicate_code", sort=True):
        coordinates = np.ascontiguousarray(
            group[["entity_code", "time"]].to_numpy(dtype=np.int64)
        )
        signature = (len(coordinates), hashlib.sha256(coordinates).digest())
        for previous in signatures.get(signature, []):
            prior = reported.loc[
                reported["predicate_code"].eq(previous),
                ["entity_code", "time"],
            ].to_numpy(dtype=np.int64)
            if np.array_equal(prior, coordinates):
                raise ValueError(
                    "observationally identical Home Credit predicates: "
                    f"{PREDICATES[previous]} and {PREDICATES[int(code)]}"
                )
        signatures.setdefault(signature, []).append(int(code))


def _combine_client_histories(
    event_frames: Iterable[pd.DataFrame],
    target_frames: dict[str, pd.DataFrame],
    observed_client_arrays: Iterable[np.ndarray],
    observation_frames: Iterable[pd.DataFrame],
    *,
    max_observation_months: int,
    partition_seed: int,
    partition_fractions: tuple[float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    observed_arrays = [
        np.asarray(values, dtype=np.int64)
        for values in observed_client_arrays
        if len(values)
    ]
    if not observed_arrays:
        raise ValueError("Home Credit preprocessing found no observed clients")
    observed_clients = np.unique(np.concatenate(observed_arrays))

    nonempty_targets = [
        frame[["client_id", "month"]] for frame in target_frames.values() if len(frame)
    ]
    if nonempty_targets:
        target_candidates = pd.concat(nonempty_targets, ignore_index=True)
        target_candidates["client_id"] = target_candidates["client_id"].astype(np.int64)
        target_candidates["month"] = target_candidates["month"].astype(np.int16)
        first_target = target_candidates.groupby("client_id", sort=True)["month"].min()
    else:
        target_candidates = pd.DataFrame(
            {
                "client_id": pd.Series(dtype=np.int64),
                "month": pd.Series(dtype=np.int16),
            }
        )
        first_target = pd.Series(dtype=np.int16)

    first_month = -int(max_observation_months)
    prevalent = first_target.index[first_target.le(first_month)].to_numpy(
        dtype=np.int64
    )
    clients = np.setdiff1d(observed_clients, prevalent, assume_unique=True)
    if len(clients) < 3:
        raise ValueError("too few incident-risk Home Credit clients")
    entity_codes = pd.Series(np.arange(len(clients), dtype=np.int32), index=clients)
    client_target_month = first_target.reindex(clients)
    target_time = client_target_month + int(max_observation_months)
    has_target = target_time.notna()

    entities = pd.DataFrame({"client_id": clients})
    entities["entity_id"] = "client:" + entities["client_id"].astype(str)
    entities["dependency_group"] = entities["client_id"].astype(str)
    entities["start_time"] = np.int64(0)
    entities["end_time"] = (
        entities["client_id"]
        .map(target_time)
        .fillna(max_observation_months - 1)
        .astype(np.int64)
    )
    entities["baseline_origin"] = np.int64(0)
    entities["split_group"] = np.int64(0)
    partition = _fixed_client_partition(
        clients, seed=partition_seed, fractions=partition_fractions
    )
    entities["partition"] = entities["client_id"].map(partition).astype(np.int8)
    entities["end_reason"] = np.where(
        entities["client_id"].isin(target_time.index[has_target]),
        "target",
        "current_application_censor",
    )

    baseline_cells, baseline_audit = _build_baseline_cells(
        entities,
        entity_codes,
        observation_frames,
        max_observation_months=max_observation_months,
    )
    nonempty_events = [frame for frame in event_frames if len(frame)]
    if nonempty_events:
        raw_events = pd.concat(nonempty_events, ignore_index=True)
        raw_events = raw_events.loc[raw_events["client_id"].isin(clients)].copy()
        raw_events["time"] = (
            raw_events["month"].astype(np.int64) + max_observation_months
        )
        raw_events["target_time"] = raw_events["client_id"].map(target_time)
        before_target = raw_events["target_time"].isna() | raw_events["time"].lt(
            raw_events["target_time"]
        )
        removed_at_or_after_target = int((~before_target).sum())
        raw_events = raw_events.loc[before_target]
        events = pd.DataFrame(
            {
                "entity_code": raw_events["client_id"]
                .map(entity_codes)
                .to_numpy(dtype=np.int32),
                "time": raw_events["time"].to_numpy(dtype=np.int64),
                "predicate_code": raw_events["predicate_code"].to_numpy(dtype=np.int16),
                "primitive_event_id": raw_events["primitive_event_id"].to_numpy(
                    dtype=np.int64
                ),
            }
        )
    else:
        removed_at_or_after_target = 0
        events = pd.DataFrame(
            {
                "entity_code": pd.Series(dtype=np.int32),
                "time": pd.Series(dtype=np.int64),
                "predicate_code": pd.Series(dtype=np.int16),
                "primitive_event_id": pd.Series(dtype=np.int64),
            }
        )

    events = (
        events.drop_duplicates()
        .sort_values(
            ["entity_code", "time", "predicate_code", "primitive_event_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    target_clients = target_time.index[has_target].to_numpy(dtype=np.int64)
    targets = pd.DataFrame(
        {
            "entity_code": entity_codes.reindex(target_clients).to_numpy(
                dtype=np.int32
            ),
            "time": target_time.loc[target_clients].to_numpy(dtype=np.int64),
            "multiplicity": np.ones(len(target_clients), dtype=np.int32),
        }
    ).sort_values(["entity_code", "time"], kind="stable")
    targets = targets.reset_index(drop=True)
    if bool(
        events.loc[events["predicate_code"].lt(len(PREDICATES))]
        .merge(targets, on="entity_code", suffixes=("_event", "_target"))
        .eval("time_event >= time_target")
        .any()
    ):
        raise AssertionError("reported Home Credit event is not strictly pre-target")
    opportunity = baseline_cells[["entity_code", "time", "exposure"]]
    target_opportunity = targets.merge(
        opportunity,
        on=["entity_code", "time"],
        how="left",
        validate="one_to_one",
    )
    if target_opportunity["exposure"].isna().any() or not bool(
        target_opportunity["exposure"].eq(1).all()
    ):
        raise AssertionError("Home Credit target lies outside its observation risk set")

    target_source_counts = {
        source: {
            "rows": int(len(frame)),
            "clients": int(frame["client_id"].nunique()),
        }
        for source, frame in target_frames.items()
    }
    prefix_audit = {
        "observed_clients": int(len(observed_clients)),
        "prevalent_target_clients_excluded": int(len(prevalent)),
        "retained_clients": int(len(clients)),
        "incident_target_clients": int(len(targets)),
        "target_source_candidates": target_source_counts,
        "events_removed_at_or_after_first_target": removed_at_or_after_target,
        "censored_clients": int(len(clients) - len(targets)),
        "baseline": baseline_audit,
    }
    return entities, events, targets, baseline_cells, prefix_audit


def _combine_recurrent_source_histories(
    event_frames: Iterable[pd.DataFrame],
    target_onsets: pd.DataFrame,
    target_risk: pd.DataFrame,
    observation_frames: Iterable[pd.DataFrame],
    *,
    target_source: str,
    max_observation_months: int,
    partition_seed: int,
    partition_fractions: tuple[float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Combine all predictor histories with one source-specific recurrent target."""

    if target_source not in {*RECURRENT_TARGET_SOURCES, "unified"}:
        raise ValueError(f"invalid recurrent Home Credit target source: {target_source}")
    clients = np.unique(target_risk["client_id"].to_numpy(dtype=np.int64))
    if len(clients) < 3:
        raise ValueError(f"too few {target_source} recurrent-risk clients")
    entity_codes = pd.Series(np.arange(len(clients), dtype=np.int32), index=clients)
    entities = pd.DataFrame({"client_id": clients})
    entities["entity_id"] = "client:" + entities["client_id"].astype(str)
    entities["dependency_group"] = entities["client_id"].astype(str)
    entities["start_time"] = np.int64(0)
    entities["end_time"] = np.int64(max_observation_months - 1)
    entities["baseline_origin"] = np.int64(0)
    entities["split_group"] = np.int64(0)
    partition = _fixed_client_partition(
        clients, seed=partition_seed, fractions=partition_fractions
    )
    entities["partition"] = entities["client_id"].map(partition).astype(np.int8)
    entities["end_reason"] = "current_application_censor"

    baseline_cells, baseline_audit = _build_baseline_cells(
        entities,
        entity_codes,
        observation_frames,
        max_observation_months=max_observation_months,
        target_risk_frame=target_risk,
    )

    nonempty_events = [frame for frame in event_frames if len(frame)]
    if nonempty_events:
        raw_events = pd.concat(nonempty_events, ignore_index=True)
        raw_events = raw_events.loc[raw_events["client_id"].isin(clients)].copy()
        raw_events["time"] = (
            raw_events["month"].astype(np.int64) + max_observation_months
        )
        events = pd.DataFrame(
            {
                "entity_code": raw_events["client_id"]
                .map(entity_codes)
                .to_numpy(dtype=np.int32),
                "time": raw_events["time"].to_numpy(dtype=np.int64),
                "predicate_code": raw_events["predicate_code"].to_numpy(
                    dtype=np.int16
                ),
                "primitive_event_id": raw_events["primitive_event_id"].to_numpy(
                    dtype=np.int64
                ),
            }
        )
    else:
        events = pd.DataFrame(
            {
                "entity_code": pd.Series(dtype=np.int32),
                "time": pd.Series(dtype=np.int64),
                "predicate_code": pd.Series(dtype=np.int16),
                "primitive_event_id": pd.Series(dtype=np.int64),
            }
        )
    events = (
        events.drop_duplicates()
        .sort_values(
            ["entity_code", "time", "predicate_code", "primitive_event_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    current_targets = target_onsets.loc[
        target_onsets["client_id"].isin(clients)
    ].copy()
    current_targets["time"] = (
        current_targets["month"].astype(np.int64) + max_observation_months
    )
    targets = pd.DataFrame(
        {
            "entity_code": current_targets["client_id"]
            .map(entity_codes)
            .to_numpy(dtype=np.int32),
            "time": current_targets["time"].to_numpy(dtype=np.int64),
            "multiplicity": current_targets["multiplicity"].to_numpy(dtype=np.int32),
        }
    ).sort_values(["entity_code", "time"], kind="stable")
    targets = targets.reset_index(drop=True)
    if bool(targets.duplicated(["entity_code", "time"]).any()):
        raise AssertionError("recurrent Home Credit targets were not aggregated")

    target_opportunity = targets.merge(
        baseline_cells[["entity_code", "time", "exposure"]],
        on=["entity_code", "time"],
        how="left",
        validate="one_to_one",
    )
    if target_opportunity["exposure"].isna().any() or not bool(
        target_opportunity["exposure"].eq(1).all()
    ):
        raise AssertionError("recurrent target lies outside its source risk set")

    prefix_audit = {
        "target_mode": "source_specific_recurrent_30plus_onset",
        "target_source": target_source,
        "retained_clients": int(len(clients)),
        "target_client_months": int(len(targets)),
        "target_events": int(targets["multiplicity"].sum()),
        "target_clients": int(targets["entity_code"].nunique()),
        "events_retained_after_target": True,
        "persistent_30plus_months_excluded_from_onset_exposure": True,
        "recovery_reenters_risk_set": True,
        "baseline": baseline_audit,
    }
    return entities, events, targets, baseline_cells, prefix_audit


def _write_audit(
    output_root: Path,
    *,
    events: pd.DataFrame,
    targets: pd.DataFrame,
    entities: pd.DataFrame,
    prefix_audit: dict[str, object],
    source_audit: dict[str, object],
) -> None:
    reported = events.loc[events["predicate_code"].lt(len(PREDICATES))]
    counts = np.bincount(
        reported["predicate_code"].to_numpy(dtype=np.int16),
        minlength=len(PREDICATES),
    )
    entity_counts = np.bincount(
        reported["predicate_code"].to_numpy(dtype=np.int16),
        weights=np.ones(len(reported), dtype=np.float64),
        minlength=len(PREDICATES),
    )
    if len(reported):
        unique_coordinates = reported.drop_duplicates(["predicate_code", "entity_code"])
        entity_counts = np.bincount(
            unique_coordinates["predicate_code"].to_numpy(dtype=np.int16),
            minlength=len(PREDICATES),
        )
        incidence = (
            reported.assign(value=np.uint8(1))
            .pivot_table(
                index=["entity_code", "time"],
                columns="predicate_code",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            .reindex(columns=range(len(PREDICATES)), fill_value=0)
            .to_numpy(dtype=np.int64)
        )
        overlap = incidence.T @ incidence
    else:
        overlap = np.zeros((len(PREDICATES), len(PREDICATES)), dtype=np.int64)
    partition = entities["partition"].to_numpy(dtype=np.int8)
    target_codes = targets["entity_code"].to_numpy(dtype=np.int32)
    target_multiplicity = targets["multiplicity"].to_numpy(dtype=np.int64)
    split = {}
    for code, name in enumerate(PARTITION_NAMES):
        split_targets = partition[target_codes] == code
        split[name] = {
            "clients": int(np.sum(partition == code)),
            "target_clients": int(np.unique(target_codes[split_targets]).size),
            "target_client_months": int(np.count_nonzero(split_targets)),
            "target_events": int(target_multiplicity[split_targets].sum()),
        }
    same_month = {}
    for left in range(len(PREDICATES) - 1):
        entries = {}
        for right in range(left + 1, len(PREDICATES)):
            shared = int(overlap[left, right])
            if not shared:
                continue
            union = int(overlap[left, left] + overlap[right, right] - shared)
            entries[PREDICATES[right]] = {
                "count": shared,
                "jaccard": 0.0 if union == 0 else shared / union,
            }
        same_month[PREDICATES[left]] = entries
    payload = {
        "schema": "crbstpp.home_credit_client_multisource_audit.v4",
        "prefix": prefix_audit,
        "sources": source_audit,
        "splits": split,
        "forbidden_predicate_fields": list(FORBIDDEN_PREDICATE_FIELDS),
        "reported_predicates": [
            {
                "code": code,
                "name": name,
                "source": PREDICATE_SOURCES[name],
                "description": PREDICATE_DESCRIPTIONS[name],
                "events": int(counts[code]),
                "clients": int(entity_counts[code]),
            }
            for code, name in enumerate(PREDICATES)
        ],
        "same_month_overlap": same_month,
    }
    (output_root / "predicate_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_home_credit_dataset(
    output_root: Path,
    *,
    source_manifest: dict[str, object],
    entities: pd.DataFrame,
    events: pd.DataFrame,
    targets: pd.DataFrame,
    baseline_cells: pd.DataFrame,
    prefix_audit: dict[str, object],
    source_audit: dict[str, object],
    target_source: str,
    partition_seed: int,
    partition_fractions: tuple[float, float, float],
    max_observation_months: int,
    diagnostic_max_clients: int | None,
) -> Path:
    recurrent = target_source in {*RECURRENT_TARGET_SOURCES, "unified"}
    unified = target_source == "unified"
    target_definition = (
        (
            "recurrent client-level transition from no observed 30+ DPD account "
            "to at least one observed 30+ DPD account across all sources"
            if unified
            else (
                f"recurrent account-level 30+ DPD onset in {target_source}; a target is "
                "a contiguous eligible <30 to >=30 transition"
            )
        )
        if recurrent
        else (
            "earliest monthly 30+ DPD evidence across bureau STATUS in "
            "{2,3,4,5}, POS SK_DPD>=30, or credit-card SK_DPD>=30; "
            "prevalent targets at the left boundary are excluded"
        )
    )
    target_handling = (
        (
            "persistent client distress months have zero onset exposure; when no "
            "observed account remains at 30+ DPD the client re-enters the binary "
            "risk set; account counts and target-source marks are not modeled"
            if unified
            else (
                "persistent 30+ DPD account-months have zero onset exposure; an "
                "observed recovery below 30 DPD re-enters the account risk set; "
                "all target-blind financial behavior remains in predicate history; "
                "simultaneous account onsets are preserved as target multiplicity"
            )
        )
        if recurrent
        else (
            "one incident target per client; reported events at or after the "
            "target are removed; target is never an input predicate"
        )
    )
    output = write_dataset(
        output_root,
        entities=entities[
            [
                "entity_id",
                "start_time",
                "end_time",
                "baseline_origin",
                "split_group",
                "dependency_group",
                "partition",
                "end_reason",
            ]
        ],
        events=events,
        baseline_cells=baseline_cells,
        targets=targets,
        predicate_names=ALL_PREDICATES,
        predicate_roles=(
            *("reported" for _ in PREDICATES),
            *BASELINE_CONTROL_ROLES,
        ),
        # The implementation name is retained for binary compatibility; the
        # recurrent_target_process contract removes first-event censoring.
        likelihood=(
            "first_event_cloglog"
            if unified or not recurrent
            else "poisson"
        ),
        time_unit="month",
        ticks_per_unit=1,
        adverse_event_name=(
            (
                "recurrent client transition into any observed 30-or-more DPD state"
                if unified
                else f"recurrent {target_source} account transition into 30-or-more DPD"
            )
            if recurrent
            else "first observed client-level 30-or-more days-past-due month"
        ),
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded_from_reported_dictionary": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
            "primitive_event_provenance": True,
            "independent_certification_units": True,
            "observation_opportunity_controlled": True,
            "required_impact_lag": 12,
            "recurrent_target_process": bool(recurrent),
        },
        provenance={
            "preprocessor": (
                (
                    "crbstpp.preprocess.home_credit.unified_recurrent_cloglog.v14"
                    if unified
                    else "crbstpp.preprocess.home_credit.recurrent_30dpd.v13"
                )
                if recurrent
                else "crbstpp.preprocess.home_credit.financial_mechanisms.v12"
            ),
            "sources": source_manifest,
            "diagnostic_max_clients": diagnostic_max_clients,
            "entity": "SK_ID_CURR client across all historical credit products",
            "independent_split_unit": "SK_ID_CURR client",
            "partition": {
                "method": "SHA-256(seed:SK_ID_CURR) thresholds",
                "fractions": [float(value) for value in partition_fractions],
                "seed": int(partition_seed),
            },
            "time_axis": {
                "origin": f"{max_observation_months} months before current application",
                "last_observed_month": "month immediately before current application",
                "maximum_months": int(max_observation_months),
                "days_to_month": "floor(relative_days / 30)",
            },
            "target_source": target_source,
            "target_definition": target_definition,
            "target_handling": target_handling,
            "forbidden_predicate_fields": list(FORBIDDEN_PREDICATE_FIELDS),
            "predicate_catalog": [
                {
                    "name": name,
                    "source": PREDICATE_SOURCES[name],
                    "description": PREDICATE_DESCRIPTIONS[name],
                }
                for name in PREDICATES
            ],
            "predicate_design": {
                "application_outcomes": (
                    "mutually exclusive product-by-approval/refusal events"
                ),
                "directional_symmetry": (
                    "every retained deterioration channel has a target-blind "
                    "recovery or contraction counterpart"
                ),
                "card_trend_onset": (
                    "first month completing two consecutive same-direction "
                    "moves after a non-move; current and past rows only"
                ),
                "normal_schedule_ticks_excluded": True,
                "same_primitive_attributes_cannot_form_high_order_witnesses": True,
            },
            "structural_baseline": {
                "reported": False,
                "behavior_predicates_included": False,
                "table": "baseline_cells.parquet",
                "history_age_bins_months": [[0, 12], [12, 24], [24, 36]],
                "source_state": "three-bit bureau/credit-card/POS observation mask",
                "target_source_risk_conditioned": recurrent,
                "binary_client_risk_set": unified,
                "sources": ["bureau", "credit_card", "pos_cash"],
                "description": (
                    "one unrestricted nuisance intercept for each target-blind "
                    "history-age and source-observation-presence state; target "
                    "exposure is additionally restricted to eligible account-level "
                    "onset opportunities in the selected source"
                    if recurrent
                    else (
                        "one unrestricted nuisance intercept for each target-blind "
                        "history-age and source-observation-presence state; the same "
                        "baseline is shared by null and every support"
                    )
                ),
            },
            "observation_definition": {
                "right_censoring": "current application boundary",
                "account_closure": (
                    "closes only that account and is an event; it never terminates "
                    "the client history"
                ),
                "end_reason_column": "entities.parquet:end_reason",
            },
        },
    )
    _write_audit(
        output,
        events=events,
        targets=targets,
        entities=entities,
        prefix_audit=prefix_audit,
        source_audit=source_audit,
    )
    return output


def preprocess_home_credit(
    input_root: str | Path,
    output_root: str | Path,
    *,
    partition_seed: int = 111,
    partition_fractions: tuple[float, float, float] = (0.50, 0.30, 0.20),
    max_observation_months: int = 36,
    diagnostic_max_clients: int | None = None,
    target_source: str = "pooled_first",
    overwrite: bool = False,
) -> Path:
    """Create pooled-first or source-specific recurrent Home Credit data."""

    input_root, output_root = Path(input_root), Path(output_root)
    if input_root.is_file():
        raise ValueError(
            "client-level Home Credit preprocessing requires the raw dataset directory"
        )
    _require_raw_root(input_root)
    if not 12 <= max_observation_months <= 120:
        raise ValueError("max_observation_months must lie in [12, 120]")
    valid_target_sources = {
        "pooled_first",
        "unified",
        "all_recurrent",
        *RECURRENT_TARGET_SOURCES,
    }
    if target_source not in valid_target_sources:
        raise ValueError(
            f"target_source must be one of {sorted(valid_target_sources)}"
        )
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)

    application_clients = _application_clients(input_root)
    selected_clients = _diagnostic_clients(
        application_clients,
        limit=diagnostic_max_clients,
        partition_seed=partition_seed,
    )

    event_frames: list[pd.DataFrame] = []
    observation_frames: list[pd.DataFrame] = []
    observed_clients: list[np.ndarray] = []
    source_audit: dict[str, object] = {}

    events, observed, audit = _read_previous_application(
        input_root, selected_clients, max_observation_months
    )
    event_frames.extend(events)
    observed_clients.append(observed)
    source_audit["previous_application"] = audit

    events, observed, bureau_to_client, audit = _read_bureau(
        input_root, selected_clients, max_observation_months
    )
    event_frames.extend(events)
    observed_clients.append(observed)
    source_audit["bureau"] = audit
    (
        bureau_targets,
        bureau_observations,
        bureau_recurrent_targets,
        bureau_recurrent_risk,
        audit,
    ) = _read_bureau_targets(
        input_root, bureau_to_client, max_observation_months
    )
    observation_frames.append(bureau_observations)
    source_audit["bureau_balance"] = audit

    (
        events,
        observed,
        card_targets,
        card_observations,
        card_recurrent_targets,
        card_recurrent_risk,
        audit,
    ) = _read_credit_card(input_root, selected_clients, max_observation_months)
    observation_frames.append(card_observations)
    event_frames.extend(events)
    observed_clients.append(observed)
    source_audit["credit_card"] = audit

    (
        events,
        observed,
        pos_targets,
        pos_observations,
        pos_recurrent_targets,
        pos_recurrent_risk,
        audit,
    ) = _read_pos_cash(input_root, selected_clients, max_observation_months)
    observation_frames.append(pos_observations)
    event_frames.extend(events)
    observed_clients.append(observed)
    source_audit["pos_cash"] = audit

    pooled_targets = {
        "bureau_30plus": bureau_targets,
        "credit_card_30plus": card_targets,
        "pos_cash_30plus": pos_targets,
    }
    recurrent_targets = {
        "bureau": bureau_recurrent_targets,
        "credit_card": card_recurrent_targets,
        "pos_cash": pos_recurrent_targets,
    }
    recurrent_risk = {
        "bureau": bureau_recurrent_risk,
        "credit_card": card_recurrent_risk,
        "pos_cash": pos_recurrent_risk,
    }
    unified_targets, unified_risk, unified_audit = _unified_client_recurrent_state(
        pooled_targets.values(), recurrent_risk.values()
    )
    recurrent_targets["unified"] = unified_targets
    recurrent_risk["unified"] = unified_risk
    source_audit["unified_recurrent_target"] = unified_audit
    source_manifest: dict[str, object] = {
        name: {
            "sha256": _sha256_file(input_root / name),
            "target_columns": list(TARGET_COLUMNS.get(name, ())),
        }
        for name in RAW_FILES
    }

    requested_sources = (
        RECURRENT_TARGET_SOURCES
        if target_source == "all_recurrent"
        else (target_source,)
    )
    outputs: list[Path] = []
    for current_source in requested_sources:
        if current_source == "pooled_first":
            entities, combined_events, targets, baseline_cells, prefix_audit = (
                _combine_client_histories(
                    event_frames,
                    pooled_targets,
                    observed_clients,
                    observation_frames,
                    max_observation_months=max_observation_months,
                    partition_seed=partition_seed,
                    partition_fractions=partition_fractions,
                )
            )
            current_output = output_root
        else:
            entities, combined_events, targets, baseline_cells, prefix_audit = (
                _combine_recurrent_source_histories(
                    event_frames,
                    recurrent_targets[current_source],
                    recurrent_risk[current_source],
                    observation_frames,
                    target_source=current_source,
                    max_observation_months=max_observation_months,
                    partition_seed=partition_seed,
                    partition_fractions=partition_fractions,
                )
            )
            current_output = (
                output_root / current_source
                if target_source == "all_recurrent"
                else output_root
            )
        _assert_distinct_predicate_streams(combined_events)
        outputs.append(
            _write_home_credit_dataset(
                current_output,
                source_manifest=source_manifest,
                entities=entities,
                events=combined_events,
                targets=targets,
                baseline_cells=baseline_cells,
                prefix_audit=prefix_audit,
                source_audit=source_audit,
                target_source=current_source,
                partition_seed=partition_seed,
                partition_fractions=partition_fractions,
                max_observation_months=max_observation_months,
                diagnostic_max_clients=diagnostic_max_clients,
            )
        )
    return output_root if target_source == "all_recurrent" else outputs[0]
