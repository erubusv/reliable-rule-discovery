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
    _predicate_matrix,
    _prefix,
    preprocess_freddie,
)
from crbstpp.preprocess.ibm import preprocess_ibm


class PreprocessingTests(unittest.TestCase):
    def test_freddie_predicates_are_explicit_transition_events(self) -> None:
        rows = []

        def add(loan: str, eltv: list[int], upb: list[int]) -> None:
            for index, (ratio, balance) in enumerate(
                zip(eltv, upb, strict=True), start=1
            ):
                rows.append(
                    {
                        "loan_id": loan,
                        "time": index,
                        "eltv_num": ratio,
                        "upb": balance,
                    }
                )

        add("cross", [70, 85, 105, 95, 75], [100] * 5)
        add("low", [70, 70, 75, 72], [100] * 4)
        add("high", [85, 85, 90, 87], [100] * 4)
        add("negative", [105, 105, 110, 108], [100] * 4)
        add("upb", [70] * 5, [100, 100, 101, 101, 100])
        frame = pd.DataFrame(rows)
        matrix = _predicate_matrix(frame)
        self.assertEqual(tuple(matrix.columns), PREDICATES)
        np.testing.assert_array_equal(matrix.sum(axis=0).to_numpy(), np.ones(13))

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
