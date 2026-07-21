from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crbstpp.data import Dataset
from crbstpp.preprocess.freddie import (
    PERFORMANCE_COLUMNS,
    PREDICATES,
    _assert_distinct_predicate_streams,
    _predicate_matrix,
    _prefix,
    preprocess_freddie,
)
from crbstpp.preprocess.ibm import preprocess_ibm


class PreprocessingTests(unittest.TestCase):
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
        rows = []

        def add(
            loan: str,
            eltv: list[float],
            upb: list[float],
            *,
            rate: list[float] | None = None,
            modification: list[bool] | None = None,
            deferred: list[float] | None = None,
        ) -> None:
            count = len(eltv)
            rate = [5.0] * count if rate is None else rate
            modification = [False] * count if modification is None else modification
            deferred = [0.0] * count if deferred is None else deferred
            for index, values in enumerate(
                zip(
                    eltv,
                    upb,
                    rate,
                    modification,
                    deferred,
                    strict=True,
                ),
                start=1,
            ):
                ratio, balance, note_rate, modified, deferred_upb = values
                rows.append(
                    {
                        "loan_id": loan,
                        "time": index,
                        "eltv_num": ratio,
                        "upb": balance,
                        "interest_rate": note_rate,
                        "modification_settles": modified,
                        "deferred_upb": deferred_upb,
                    }
                )

        add("cross", [70, 85, 105, 95, 75], [100] * 5)
        add("upb", [70] * 5, [100, 100, 101, 101, 100])
        add("accelerate", [70] * 4, [10_000, 9_900, 9_801, 9_605])
        add("decelerate", [70] * 4, [10_000, 9_800, 9_604, 9_508])
        add("rate", [70] * 4, [100] * 4, rate=[5, 5, 6, 5])
        add(
            "modification",
            [70] * 3,
            [100] * 3,
            modification=[False, True, False],
        )
        add("deferred", [70] * 3, [100] * 3, deferred=[0, 10, 0])
        frame = pd.DataFrame(rows)
        matrix = _predicate_matrix(frame)
        self.assertEqual(tuple(matrix.columns), PREDICATES)
        np.testing.assert_array_equal(matrix.sum(axis=0).to_numpy(), np.ones(14))

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
            for entity, time in zip(
                data.target_entities, data.target_times, strict=True
            ):
                self.assertEqual(data.end_times[entity], time)

    def test_ibm_preserves_minute_order_for_continuous_poisson(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "transactions.csv"
            rows = []
            for index in range(30):
                rows.append(
                    {
                        "Timestamp": f"2023/01/01 00:{index:02d}",
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
            self.assertEqual(data.ticks_per_unit, 60)
            self.assertGreaterEqual(int(data.event_times.max(initial=0)), 20)
            self.assertEqual(int(data.target_multiplicity.sum()), 2)


if __name__ == "__main__":
    unittest.main()
