from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crbstpp.data import Dataset
from crbstpp.response import Context
from crbstpp.preprocess.freddie import (
    BASELINE_CONTROLS,
    PERFORMANCE_COLUMNS,
    PREDICATES,
    _assert_distinct_predicate_streams,
    _baseline_control_matrix,
    _predicate_matrix,
    _prefix,
    preprocess_freddie,
)
from crbstpp.preprocess.ibm import (
    BASELINE_CONTROLS as IBM_BASELINE_CONTROLS,
    PREDICATES as IBM_PREDICATES,
    preprocess_ibm,
)
from crbstpp.preprocess.home_credit import (
    BASELINE_CONTROLS as HOME_CREDIT_BASELINE_CONTROLS,
    PREDICATES as HOME_CREDIT_PREDICATES,
    _credit_card_predicates,
    _pos_predicates,
    _recurrent_30dpd_state,
    _unified_client_recurrent_state,
    preprocess_home_credit,
)


class PreprocessingTests(unittest.TestCase):
    def test_home_credit_recurrent_onset_risk_reenters_after_recovery(self) -> None:
        frame = pd.DataFrame(
            {
                "account": [10] * 7,
                "client": [1] * 7,
                "month": np.arange(-7, 0),
                "dpd": [0, 10, 40, 60, 20, 0, 35],
            }
        )
        targets, risk, audit = _recurrent_30dpd_state(
            frame,
            account_column="account",
            client_column="client",
            month_column="month",
            serious=frame["dpd"].ge(30),
            eligible=np.ones(len(frame), dtype=bool),
            source="test",
        )
        self.assertEqual(targets["month"].tolist(), [-5, -1])
        self.assertEqual(targets["multiplicity"].tolist(), [1, 1])
        self.assertEqual(risk["month"].tolist(), [-7, -6, -5, -3, -2, -1])
        self.assertEqual(audit["persistent_serious_account_months_excluded"], 1)

    def test_home_credit_unified_recurrent_state_is_binary_and_repeatable(self) -> None:
        serious = pd.DataFrame({"client_id": [1, 1], "month": [-5, -2]})
        risk = pd.DataFrame(
            {
                "client_id": [1] * 6,
                "month": np.arange(-6, 0),
                "account_count": np.ones(6, dtype=np.int32),
                "source": ["test"] * 6,
            }
        )
        targets, exposure, audit = _unified_client_recurrent_state(
            (serious,), (risk,)
        )
        self.assertEqual(targets["month"].tolist(), [-5, -2])
        self.assertEqual(targets["multiplicity"].tolist(), [1, 1])
        self.assertEqual(exposure["account_count"].unique().tolist(), [1])
        self.assertEqual(audit["onset_events"], 2)

    @staticmethod
    def _write_home_credit_raw(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        clients = np.arange(200_000, 200_060, dtype=np.int64)
        pd.DataFrame(
            {"SK_ID_CURR": clients[:45], "TARGET": np.zeros(45, dtype=np.int8)}
        ).to_csv(raw / "application_train.csv", index=False)
        pd.DataFrame({"SK_ID_CURR": clients[45:]}).to_csv(
            raw / "application_test.csv", index=False
        )

        previous_rows = []
        for index, client in enumerate(clients):
            contract = ("Cash loans", "Consumer loans", "Revolving loans")[index % 3]
            for occurrence, (days, amount) in enumerate(
                ((-330, 100.0), (-210, 130.0), (-90, 110.0))
            ):
                previous_rows.append(
                    {
                        "SK_ID_PREV": 700_000 + 10 * index + occurrence,
                        "SK_ID_CURR": client,
                        "DAYS_DECISION": days,
                        "NAME_CONTRACT_TYPE": contract,
                        "NAME_CONTRACT_STATUS": (
                            "Refused" if (occurrence + index) % 3 == 1 else "Approved"
                        ),
                        "AMT_APPLICATION": amount,
                        "CNT_PAYMENT": (12, 18, 9)[occurrence],
                    }
                )
        previous = pd.DataFrame(previous_rows)
        previous.to_csv(raw / "previous_application.csv", index=False)

        bureau = pd.DataFrame(
            {
                "SK_ID_CURR": clients,
                "SK_ID_BUREAU": 800_000 + np.arange(len(clients)),
                "DAYS_CREDIT": -30 * (3 + np.arange(len(clients)) % 7),
                "DAYS_ENDDATE_FACT": np.where(
                    np.arange(len(clients)) % 4 == 0,
                    -30 * (1 + np.arange(len(clients)) % 5),
                    np.nan,
                ),
                "CREDIT_TYPE": np.asarray(
                    ["Consumer credit", "Credit card", "Microloan"]
                )[np.arange(len(clients)) % 3],
            }
        )
        bureau.to_csv(raw / "bureau.csv", index=False)
        bureau_balance_rows = []
        for index, bureau_id in enumerate(bureau["SK_ID_BUREAU"]):
            for month in range(-12, 0):
                status = "2" if index % 11 == 0 and month == -3 else "0"
                bureau_balance_rows.append(
                    {
                        "SK_ID_BUREAU": bureau_id,
                        "MONTHS_BALANCE": month,
                        "STATUS": status,
                    }
                )
        pd.DataFrame(bureau_balance_rows).to_csv(
            raw / "bureau_balance.csv", index=False
        )

        card_rows: list[dict[str, object]] = []
        pos_rows: list[dict[str, object]] = []
        installment_rows: list[dict[str, object]] = []
        for index, client in enumerate(clients):
            card_account = 900_000 + index
            pos_account = 950_000 + index
            for step, month in enumerate(range(-12, 0)):
                cycle = (step + index) % 4
                balance = float(20 + 5 * cycle + 7 * (step % 3))
                if index % 10 == 0:
                    balance = 0.0 if step < 3 or step >= 8 else 40.0
                elif index % 10 == 1:
                    balance = float(
                        (20, 20, 25, 30, 25, 20, 20, 20, 20, 20, 20, 20)[
                            step
                        ]
                    )
                limit = float(100 - (20 if 7 <= step < 9 and index % 5 == 0 else 0))
                payment = float(30 - 4 * cycle + 3 * (step % 2))
                if index % 10 == 2:
                    payment = float(
                        (20, 20, 18, 16, 18, 20, 20, 20, 20, 20, 20, 20)[
                            step
                        ]
                    )
                cash = (
                    float(5 * (step - 4)) if 5 <= step < 8 and index % 6 == 0 else 0.0
                )
                card_rows.append(
                    {
                        "SK_ID_PREV": card_account,
                        "SK_ID_CURR": client,
                        "MONTHS_BALANCE": month,
                        "AMT_BALANCE": balance
                        + (100 if step == 8 and index % 7 == 0 else 0),
                        "AMT_CREDIT_LIMIT_ACTUAL": limit,
                        "AMT_PAYMENT_TOTAL_CURRENT": payment,
                        "AMT_DRAWINGS_ATM_CURRENT": cash,
                        "AMT_DRAWINGS_OTHER_CURRENT": 0.0,
                        "AMT_DRAWINGS_POS_CURRENT": (
                            12.0 if 4 <= step < 8 and index % 5 == 2 else 0.0
                        ),
                        "SK_DPD": 30 if index % 7 == 0 and month == -2 else 0,
                    }
                )
                pos_rows.append(
                    {
                        "SK_ID_PREV": pos_account,
                        "SK_ID_CURR": client,
                        "MONTHS_BALANCE": month,
                        "CNT_INSTALMENT": 12
                        + (3 if 6 <= step < 9 and index % 5 == 1 else 0),
                        "CNT_INSTALMENT_FUTURE": 12
                        + (3 if 6 <= step < 9 and index % 5 == 1 else 0),
                        "NAME_CONTRACT_STATUS": (
                            ("Completed" if index % 2 else "Signed")
                            if month == -2
                            else "Active"
                        ),
                        "SK_DPD": 30 if index % 13 == 0 and month == -4 else 0,
                    }
                )
                installment_rows.append(
                    {
                        "SK_ID_PREV": pos_account,
                        "SK_ID_CURR": client,
                        "NUM_INSTALMENT_VERSION": 1,
                        "NUM_INSTALMENT_NUMBER": step + 1,
                        "DAYS_INSTALMENT": 30 * month,
                        "AMT_INSTALMENT": float(
                            100 + 20 * (5 <= step < 9 and index % 3 == 0)
                        ),
                    }
                )
        card = pd.DataFrame(card_rows)
        pos = pd.DataFrame(pos_rows)
        card.to_csv(raw / "credit_card_balance.csv", index=False)
        pos.to_csv(raw / "POS_CASH_balance.csv", index=False)
        pd.DataFrame(installment_rows).to_csv(
            raw / "installments_payments.csv", index=False
        )
        return card, pos

    def test_home_credit_first_30dpd_contract_and_client_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            self._write_home_credit_raw(raw)
            output = preprocess_home_credit(
                raw,
                root / "processed",
                max_observation_months=12,
            )
            data = Dataset.load(output)
            self.assertEqual(data.likelihood, "first_event_cloglog")
            self.assertEqual(data.time_unit, "month")
            self.assertEqual(
                data.predicate_names[: len(HOME_CREDIT_PREDICATES)],
                HOME_CREDIT_PREDICATES,
            )
            self.assertEqual(
                data.predicate_names[len(HOME_CREDIT_PREDICATES) :],
                HOME_CREDIT_BASELINE_CONTROLS,
            )
            self.assertNotIn("dpd", " ".join(HOME_CREDIT_PREDICATES).lower())
            for left, right in (
                ("cash_loan_approved", "cash_loan_refused"),
                ("consumer_loan_approved", "consumer_loan_refused"),
                ("revolving_credit_approved", "revolving_credit_refused"),
                ("amount_increases", "amount_decreases"),
                ("term_lengthens", "term_shortens"),
                ("consumer_credit_opened", "consumer_credit_closed"),
                ("revolving_credit_opened", "revolving_credit_closed"),
                ("microloan_opened", "microloan_closed"),
                ("cash_withdrawal_starts", "cash_withdrawal_stops"),
                ("pos_purchase_starts", "pos_purchase_stops"),
                ("revolving_balance_starts", "revolving_balance_clears"),
                ("utilization_rise", "utilization_fall"),
                ("payment_rate_decline", "payment_rate_recovery"),
                ("schedule_extended", "accelerated_amortization"),
            ):
                self.assertTrue(any(left in name for name in HOME_CREDIT_PREDICATES))
                self.assertTrue(any(right in name for name in HOME_CREDIT_PREDICATES))
            self.assertIsNotNone(data.partitions)
            self.assertEqual(HOME_CREDIT_BASELINE_CONTROLS, ())
            self.assertIsNotNone(data.baseline_cell_strata)
            self.assertEqual(
                len(data.baseline_cell_strata), int(np.sum(data.end_times + 1))
            )
            context = Context.make(data, np.arange(data.n_entities, dtype=np.int32))
            rows = np.arange(context.n_grid, dtype=np.int64)
            groups = context.temporal_baseline_groups_at_rows(rows, time_bins=1)
            np.testing.assert_array_equal(groups, data.baseline_cell_strata)
            counts = context.temporal_baseline_counts(time_bins=1)
            np.testing.assert_array_equal(
                counts.sum(axis=1), context.entity_exposure_totals()
            )
            self.assertIsNotNone(context.baseline_row_exposure)
            assert context.baseline_row_exposure is not None
            np.testing.assert_array_equal(
                context.baseline_row_exposure[context.target_rows], 1.0
            )
            self.assertIsNotNone(data.end_reasons)
            self.assertEqual(set(data.partitions.tolist()), {0, 1, 2})
            entity_by_client = {
                int(entity.removeprefix("client:")): index
                for index, entity in enumerate(data.entity_ids.tolist())
            }
            for entity, time in zip(
                data.target_entities, data.target_times, strict=True
            ):
                self.assertEqual(data.end_times[entity], time)
                self.assertGreater(time, data.start_times[entity])
                self.assertEqual(data.end_reasons[entity], "target")
            completed_client = 200_000 + 1
            completed_code = entity_by_client[completed_client]
            self.assertEqual(
                data.end_reasons[completed_code], "current_application_censor"
            )
            self.assertEqual(data.end_times[completed_code], 11)
            audit = json.loads((output / "predicate_audit.json").read_text())
            self.assertGreater(audit["prefix"]["incident_target_clients"], 0)
            self.assertGreater(
                audit["prefix"]["events_removed_at_or_after_first_target"], 0
            )
            self.assertEqual(
                len(audit["reported_predicates"]), len(HOME_CREDIT_PREDICATES)
            )
            self.assertTrue(
                all(row["events"] > 0 for row in audit["reported_predicates"])
            )

    def test_home_credit_source_recurrent_poisson_risk_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            self._write_home_credit_raw(raw)
            output = preprocess_home_credit(
                raw,
                root / "processed",
                max_observation_months=12,
                target_source="all_recurrent",
            )
            for source in ("bureau", "credit_card", "pos_cash"):
                data = Dataset.load(output / source)
                self.assertEqual(data.likelihood, "poisson")
                self.assertGreater(int(data.target_multiplicity.sum()), 0)
                self.assertTrue(np.all(data.end_times == 11))
                self.assertTrue(np.all(data.end_reasons == "current_application_censor"))
                assert data.baseline_cell_exposure is not None
                self.assertTrue(np.all(data.baseline_cell_exposure >= 0.0))
                self.assertTrue(np.all(data.baseline_cell_exposure <= 1.0))
                manifest = json.loads(
                    ((output / source) / "manifest.json").read_text()
                )
                self.assertEqual(manifest["provenance"]["target_source"], source)
                self.assertIn("recovery", manifest["provenance"]["target_handling"])

    def test_home_credit_predicates_do_not_read_target_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            card, pos = self._write_home_credit_raw(raw)
            for frame, builder in (
                (card, _credit_card_predicates),
                (pos, _pos_predicates),
            ):
                frame = frame.rename(columns={"MONTHS_BALANCE": "month"}).sort_values(
                    ["SK_ID_PREV", "month"], kind="stable"
                )
                first = builder(frame)
                changed = frame.copy()
                changed["SK_DPD"] = np.arange(len(changed)) % 91
                changed["SK_DPD_DEF"] = np.arange(len(changed))[::-1] % 67
                second = builder(changed)
                pd.testing.assert_frame_equal(first, second)

    def test_freddie_rejects_identical_predicate_streams(self) -> None:
        events = pd.DataFrame(
            {
                "entity_code": [0, 0],
                "time": [1, 1],
                "predicate_code": [0, 1],
            }
        )
        with self.assertRaisesRegex(ValueError, "observationally identical"):
            _assert_distinct_predicate_streams(events)

    def test_freddie_predicates_are_explicit_transition_events(self) -> None:
        self.assertNotIn("pred_borrower_forbearance_starts", PREDICATES)
        self.assertNotIn("pred_disaster_delinquency_starts", PREDICATES)
        self.assertFalse(any("interest_rate" in name for name in PREDICATES))
        self.assertFalse(any("remaining_term" in name for name in PREDICATES))
        self.assertIn("control_borrower_forbearance_history", BASELINE_CONTROLS)
        self.assertIn("control_disaster_delinquency_history", BASELINE_CONTROLS)
        rows = []

        def add(
            loan: str,
            eltv: list[float],
            upb: list[float],
            *,
            rate: list[float] | None = None,
            modification: list[bool] | None = None,
            deferred: list[float] | None = None,
            deferred_plan: list[bool] | None = None,
            forbearance: list[bool] | None = None,
            disaster: list[bool] | None = None,
        ) -> None:
            count = len(eltv)
            rate = [5.0] * count if rate is None else rate
            modification = [False] * count if modification is None else modification
            deferred = [0.0] * count if deferred is None else deferred
            deferred_plan = [False] * count if deferred_plan is None else deferred_plan
            forbearance = [False] * count if forbearance is None else forbearance
            disaster = [False] * count if disaster is None else disaster
            for index, values in enumerate(
                zip(
                    eltv,
                    upb,
                    rate,
                    modification,
                    deferred,
                    deferred_plan,
                    forbearance,
                    disaster,
                    strict=True,
                ),
                start=1,
            ):
                (
                    ratio,
                    balance,
                    note_rate,
                    modified,
                    deferred_upb,
                    in_deferred_plan,
                    in_forbearance,
                    disaster_related,
                ) = values
                rows.append(
                    {
                        "loan_id": loan,
                        "time": index,
                        "eltv_num": ratio,
                        "upb": balance,
                        "interest_rate": note_rate,
                        "modification_settles": modified,
                        "modification": modified,
                        "deferred_upb": deferred_upb,
                        "deferred_plan": in_deferred_plan,
                        "borrower_forbearance": in_forbearance,
                        "disaster_delinquency": disaster_related,
                        "delinquency_level": 0,
                    }
                )

        add("cross", [70, 85, 105, 95, 75], [100] * 5)
        add("upb", [70] * 5, [100, 100, 101, 101, 100])
        add("eltv_direction", [70, 70, 75, 75, 70], [100] * 5)
        add("accelerate", [70] * 4, [10_000, 9_900, 9_800, 9_600])
        add("decelerate", [70] * 4, [10_000, 9_800, 9_600, 9_500])
        add("steady", [70] * 4, [10_000, 9_800, 9_700, 9_600])
        add("mod", [70, 70], [100, 99], modification=[False, True])
        add("plan", [70, 70], [100, 99], deferred_plan=[False, True])
        add("forbear", [70, 70], [100, 99], forbearance=[False, True])
        add("disaster", [70, 70], [100, 99], disaster=[False, True])
        add("deferred", [70, 70], [100, 99], deferred=[0.0, 5.0])
        frame = pd.DataFrame(rows)
        matrix = _predicate_matrix(frame)
        self.assertEqual(tuple(matrix.columns), PREDICATES)
        np.testing.assert_array_equal(matrix.sum(axis=0).to_numpy(), np.ones(13))
        controls = _baseline_control_matrix(frame)
        self.assertEqual(tuple(controls.columns), BASELINE_CONTROLS)
        self.assertEqual(int(controls["control_borrower_forbearance_history"].sum()), 1)
        self.assertEqual(int(controls["control_disaster_delinquency_history"].sum()), 1)

    def test_freddie_prefix_obeys_absorbing_gap_and_horizon_stops(self) -> None:
        rows: list[dict[str, object]] = []
        for loan, times, target, termination in (
            ("target", range(1, 6), 3, None),
            ("termination", range(1, 6), 4, 2),
            ("gap", (1, 2, 4, 5), None, None),
            ("horizon", range(1, 41), None, None),
        ):
            for time in times:
                rows.append(
                    {
                        "loan_id": loan,
                        "time": time,
                        "target": time == target,
                        "termination": time == termination,
                    }
                )
        prefixed = _prefix(pd.DataFrame(rows), max_observation_months=36)
        observed = {
            loan: group["time"].tolist()
            for loan, group in prefixed.groupby("loan_id", sort=False)
        }
        self.assertEqual(observed["target"], [1, 2, 3])
        self.assertEqual(observed["termination"], [1, 2])
        self.assertEqual(observed["gap"], [1, 2])
        self.assertEqual(observed["horizon"], list(range(1, 37)))

    def test_freddie_first_event_risk_set_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "2023" / "Q1"
            raw.mkdir(parents=True)
            rows = []
            for loan in range(6):
                for month, status in (
                    (1, "0"),
                    (2, "0"),
                    (3, "3" if loan % 2 == 0 else "0"),
                    (4, "0"),
                ):
                    values = {name: "" for name in PERFORMANCE_COLUMNS}
                    values.update(
                        {
                            "loan_id": f"L{loan}",
                            "monthly_reporting_period": f"2023{month:02d}",
                            "current_actual_upb": str(100_000 - 1_000 * month),
                            "current_loan_delinquency_status": status,
                            "loan_age": str(month),
                            "zero_balance_code": "",
                            "eltv": str(70 + 5 * month),
                        }
                    )
                    rows.append([values[name] for name in PERFORMANCE_COLUMNS])
            pd.DataFrame(rows).to_csv(
                raw / "historical_data_time_2023Q1.txt",
                sep="|",
                header=False,
                index=False,
            )
            origination = []
            for loan in range(6):
                values = [""] * 20
                values[1] = f"2023{loan + 1:02d}"
                values[19] = f"L{loan}"
                origination.append(values)
            pd.DataFrame(origination).to_csv(
                raw / "historical_data_2023Q1.txt",
                sep="|",
                header=False,
                index=False,
            )
            output = preprocess_freddie(
                root / "raw", root / "processed", vintages=("2023Q1",)
            )
            data = Dataset.load(output)
            self.assertEqual(data.likelihood, "first_event_cloglog")
            self.assertEqual(data.ticks_per_unit, 1)
            self.assertEqual(data.predicate_names[-4:], BASELINE_CONTROLS)
            self.assertEqual(data.n_reported_predicates, len(PREDICATES))
            for entity, time in zip(
                data.target_entities, data.target_times, strict=True
            ):
                self.assertEqual(data.end_times[entity], time)

    def test_ibm_uses_minute_order_and_hourly_poisson_risk_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "transactions.csv"
            rows = []
            for index in range(30):
                rows.append(
                    {
                        "Timestamp": (
                            pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=5 * index)
                        ).strftime("%Y/%m/%d %H:%M"),
                        "From Bank": 1,
                        "Account": f"A{index % 3}",
                        "To Bank": 2,
                        "Account.1": f"B{index % 5}",
                        "Amount Received": 100 + index,
                        "Receiving Currency": "USD",
                        "Amount Paid": 100 + index,
                        "Payment Currency": "USD" if index < 20 else "EUR",
                        "Payment Format": "ACH" if index % 4 else "WIRE",
                        "Is Laundering": 1 if index in {10, 20} else 0,
                    }
                )
            pd.DataFrame(rows).to_csv(raw, index=False)
            output = preprocess_ibm(raw, root / "processed")
            data = Dataset.load(output)
            self.assertEqual(data.likelihood, "poisson")
            self.assertEqual(data.time_unit, "hour")
            self.assertEqual(data.ticks_per_unit, 1)
            self.assertGreaterEqual(int(data.event_times.max(initial=0)), 2)
            self.assertEqual(int(data.target_multiplicity.sum()), 2)
            self.assertEqual(
                data.predicate_names[: len(IBM_PREDICATES)], IBM_PREDICATES
            )
            self.assertEqual(
                data.predicate_names[len(IBM_PREDICATES) :],
                IBM_BASELINE_CONTROLS,
            )
            self.assertIsNotNone(data.partitions)
            self.assertTrue(data.f0_contract["independent_certification_units"])
            audit = json.loads((output / "predicate_audit.json").read_text())
            self.assertEqual(audit["dataset_digest"], data.digest)
            self.assertEqual(len(audit["predicates"]), len(IBM_PREDICATES))
            self.assertEqual(audit["target_events"], 2)

    def test_ibm_dictionary_is_semantic_and_target_blind(self) -> None:
        forbidden = ("amount_spike", "amount_drop", "burst", "dormancy", "cadence")
        self.assertFalse(
            any(token in name for token in forbidden for name in IBM_PREDICATES)
        )
        self.assertIn("pred_out_new_receiver", IBM_PREDICATES)
        self.assertIn("pred_out_payment_format_switch_to_wire", IBM_PREDICATES)
        self.assertIn("pred_in_to_out_transition", IBM_PREDICATES)
        self.assertIn("pred_out_reciprocal_edge_onset", IBM_PREDICATES)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "Timestamp": f"2023/01/01 00:{minute:02d}",
                    "From Bank": from_bank,
                    "Account": sender,
                    "To Bank": to_bank,
                    "Account.1": receiver,
                    "Amount Received": 100,
                    "Receiving Currency": receiving_currency,
                    "Amount Paid": 100,
                    "Payment Currency": payment_currency,
                    "Payment Format": payment_format,
                    "Is Laundering": 0,
                }
                for minute, (
                    from_bank,
                    sender,
                    to_bank,
                    receiver,
                    payment_currency,
                    receiving_currency,
                    payment_format,
                ) in enumerate(
                    (
                        (1, "A", 2, "B", "USD", "USD", "ACH"),
                        (2, "B", 1, "A", "USD", "USD", "WIRE"),
                        (1, "A", 3, "C", "EUR", "USD", "WIRE"),
                        (4, "D", 1, "A", "USD", "USD", "CHEQUE"),
                        (1, "A", 4, "D", "USD", "USD", "ACH"),
                    )
                )
            ]
            rows.append(
                {
                    "Timestamp": "2023/01/01 01:05",
                    "From Bank": 1,
                    "Account": "A",
                    "To Bank": 1,
                    "Account.1": "A",
                    "Amount Received": 100,
                    "Receiving Currency": "USD",
                    "Amount Paid": 100,
                    "Payment Currency": "USD",
                    "Payment Format": "Reinvestment",
                    "Is Laundering": 0,
                }
            )
            first_raw = root / "first.csv"
            second_raw = root / "second.csv"
            pd.DataFrame(rows).to_csv(first_raw, index=False)
            changed = pd.DataFrame(rows)
            changed["Is Laundering"] = [1, 0, 1, 0, 1, 0]
            changed.to_csv(second_raw, index=False)
            first = Dataset.load(preprocess_ibm(first_raw, root / "first"))
            second = Dataset.load(preprocess_ibm(second_raw, root / "second"))
            np.testing.assert_array_equal(first.event_entities, second.event_entities)
            np.testing.assert_array_equal(first.event_times, second.event_times)
            np.testing.assert_array_equal(
                first.event_predicates, second.event_predicates
            )
            account = int(np.flatnonzero(first.entity_ids == "1:A")[0])
            self_time = 1
            at_self_transaction = (first.event_entities == account) & (
                first.event_times == self_time
            )
            self.assertTrue(np.any(at_self_transaction))
            self.assertTrue(
                np.all(
                    first.event_predicates[at_self_transaction] >= len(IBM_PREDICATES)
                )
            )
            self.assertNotEqual(
                int(first.target_multiplicity.sum()),
                int(second.target_multiplicity.sum()),
            )

    def test_ibm_tied_transactions_are_independent_of_csv_row_order(self) -> None:
        rows = [
            {
                "Timestamp": timestamp,
                "From Bank": from_bank,
                "Account": sender,
                "To Bank": to_bank,
                "Account.1": receiver,
                "Amount Received": 100,
                "Receiving Currency": receiving_currency,
                "Amount Paid": 100,
                "Payment Currency": payment_currency,
                "Payment Format": payment_format,
                "Is Laundering": target,
            }
            for timestamp, (
                from_bank,
                sender,
                to_bank,
                receiver,
                payment_currency,
                receiving_currency,
                payment_format,
                target,
            ) in (
                ("2023/01/01 00:00", (1, "A", 2, "B", "USD", "USD", "ACH", 0)),
                ("2023/01/01 00:00", (1, "A", 3, "C", "EUR", "EUR", "WIRE", 0)),
                ("2023/01/01 00:05", (2, "B", 1, "A", "USD", "USD", "CASH", 0)),
                ("2023/01/01 00:05", (4, "D", 1, "A", "EUR", "EUR", "WIRE", 0)),
                ("2023/01/01 01:10", (1, "A", 2, "B", "USD", "USD", "ACH", 1)),
                ("2023/01/01 01:10", (1, "A", 4, "D", "EUR", "EUR", "CASH", 0)),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_raw = root / "first.csv"
            second_raw = root / "second.csv"
            frame = pd.DataFrame(rows)
            frame.to_csv(first_raw, index=False)
            frame.iloc[[1, 0, 3, 2, 5, 4]].to_csv(second_raw, index=False)
            first = Dataset.load(preprocess_ibm(first_raw, root / "first"))
            second = Dataset.load(preprocess_ibm(second_raw, root / "second"))
            np.testing.assert_array_equal(first.entity_ids, second.entity_ids)
            np.testing.assert_array_equal(first.event_entities, second.event_entities)
            np.testing.assert_array_equal(first.event_times, second.event_times)
            np.testing.assert_array_equal(
                first.event_predicates, second.event_predicates
            )
            np.testing.assert_array_equal(first.target_entities, second.target_entities)
            np.testing.assert_array_equal(first.target_times, second.target_times)
            np.testing.assert_array_equal(
                first.target_multiplicity, second.target_multiplicity
            )


if __name__ == "__main__":
    unittest.main()
