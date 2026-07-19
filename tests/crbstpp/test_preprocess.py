from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from crbstpp.data import Dataset
from crbstpp.preprocess.freddie import PERFORMANCE_COLUMNS, preprocess_freddie
from crbstpp.preprocess.ibm import preprocess_ibm


class PreprocessingTests(unittest.TestCase):
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
