#!/usr/bin/env python3
"""Attach decision-time Home Credit exposure to an existing TPP dataset.

The event stream is copied with hard links (and therefore remains byte-identical),
while the sequence table is rewritten with the current application's AMT_CREDIT.
AMT_CREDIT is known at the application decision and is used only as a nonnegative
cluster weight; neither Kaggle TARGET nor any future outcome is attached.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPOSURE_COLUMN = "financial_exposure_amt_credit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _hardlink_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    shutil.copytree(source, destination, copy_function=os.link)


def _read_applications(raw_root: Path, samples: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if "train" in samples:
        train = pd.read_csv(
            raw_root / "application_train.csv",
            usecols=["SK_ID_CURR", "AMT_CREDIT"],
        )
        train["sample"] = "train"
        frames.append(train)
    if "test" in samples:
        test = pd.read_csv(
            raw_root / "application_test.csv",
            usecols=["SK_ID_CURR", "AMT_CREDIT"],
        )
        test["sample"] = "test"
        frames.append(test)
    if not frames:
        raise ValueError("sequence table has neither train nor test samples")
    apps = pd.concat(frames, ignore_index=True)
    apps["SK_ID_CURR"] = apps["SK_ID_CURR"].astype("int64")
    if apps.duplicated(["SK_ID_CURR", "sample"]).any():
        raise ValueError("application exposure key is not unique")
    return apps


def main() -> None:
    args = parse_args()
    source_sequences = args.input_root / "sequences" / "part-0000.parquet"
    if not source_sequences.is_file():
        raise FileNotFoundError(source_sequences)
    if args.output_root.resolve() == args.input_root.resolve():
        raise ValueError("output-root must differ from input-root")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_root)

    sequences = pd.read_parquet(source_sequences)
    required = {"SK_ID_CURR", "sequence_id", "sample"}
    missing = sorted(required - set(sequences.columns))
    if missing:
        raise ValueError(f"sequence table is missing keys: {missing}")
    if sequences["sequence_id"].duplicated().any():
        raise ValueError("sequence_id is not unique")

    samples = set(sequences["sample"].astype(str).unique())
    applications = _read_applications(args.raw_root, samples)
    applications = applications.rename(columns={"AMT_CREDIT": EXPOSURE_COLUMN})
    merged = sequences.merge(
        applications,
        on=["SK_ID_CURR", "sample"],
        how="left",
        validate="one_to_one",
    )
    exposure = pd.to_numeric(merged[EXPOSURE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(exposure)):
        raise ValueError(f"{int(np.sum(~np.isfinite(exposure)))} sequences have missing/nonfinite exposure")
    if np.any(exposure < 0) or not np.any(exposure > 0):
        raise ValueError("financial exposure must be nonnegative with positive mass")

    args.output_root.mkdir(parents=True)
    _hardlink_tree(args.input_root / "sequence_months", args.output_root / "sequence_months")
    _hardlink_tree(args.input_root / "sequence_tokens", args.output_root / "sequence_tokens")
    shutil.copytree(args.input_root / "metadata", args.output_root / "metadata")
    (args.output_root / "sequences").mkdir()
    merged.to_parquet(args.output_root / "sequences" / "part-0000.parquet", index=False)

    contract = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(args.input_root.resolve()),
        "event_stream_identity": "hard-linked byte-identical event parquet",
        "financial_weight_column": EXPOSURE_COLUMN,
        "source_table": "application_train.csv/application_test.csv",
        "source_field": "AMT_CREDIT",
        "availability": "known at the current application decision time",
        "semantics": (
            "current requested credit exposure used as a cluster weight for the TPP proper scoring loss; "
            "not realized monetary loss or profit"
        ),
        "n_sequences": int(len(merged)),
        "missing_count": int(np.sum(~np.isfinite(exposure))),
        "negative_count": int(np.sum(exposure < 0)),
        "summary": {
            "mean": float(np.mean(exposure)),
            "median": float(np.median(exposure)),
            "p01": float(np.quantile(exposure, 0.01)),
            "p99": float(np.quantile(exposure, 0.99)),
            "min": float(np.min(exposure)),
            "max": float(np.max(exposure)),
        },
        "outcome_fields_present_in_inherited_sequence_metadata": [
            name for name in ("TARGET",) if name in merged.columns
        ],
        "outcome_fields_used_to_construct_financial_weight": [],
        "loader_columns_for_financial_experiment": [
            "sequence_id",
            "start_month",
            "end_month",
            EXPOSURE_COLUMN,
        ],
        "leakage_note": (
            "inherited Kaggle TARGET may remain in the sequence parquet for provenance, "
            "but load_event_data reads neither TARGET nor any other application outcome"
        ),
    }
    (args.output_root / "metadata" / "financial_loss_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
