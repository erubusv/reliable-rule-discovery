#!/usr/bin/env python3
"""Attach pre-outcome Freddie Mac UPB exposure to a dynamic TPP dataset.

The fixed cluster weight is the current actual unpaid principal balance in the
first observed performance month of each continuous loan sequence. Event and
token parquet files are hard-linked and remain byte-identical.
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


EXPOSURE_COLUMN = "financial_exposure_initial_upb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _hardlink_tree(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, copy_function=os.link)


def main() -> None:
    args = parse_args()
    if args.output_root.resolve() == args.input_root.resolve():
        raise ValueError("output-root must differ from input-root")
    sequence_dir = args.input_root / "sequences"
    month_dir = args.input_root / "sequence_months"
    sequence_files = sorted(sequence_dir.glob("*.parquet"))
    if not sequence_files:
        raise FileNotFoundError(f"no sequence parquet files in {sequence_dir}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_root)

    args.output_root.mkdir(parents=True)
    _hardlink_tree(month_dir, args.output_root / "sequence_months")
    _hardlink_tree(args.input_root / "sequence_tokens", args.output_root / "sequence_tokens")
    if (args.input_root / "metadata").is_dir():
        shutil.copytree(args.input_root / "metadata", args.output_root / "metadata")
    else:
        (args.output_root / "metadata").mkdir()
    output_sequences = args.output_root / "sequences"
    output_sequences.mkdir()

    summaries: list[np.ndarray] = []
    sequence_count = 0
    for sequence_file in sequence_files:
        month_file = month_dir / sequence_file.name
        if not month_file.is_file():
            raise FileNotFoundError(month_file)
        sequences = pd.read_parquet(sequence_file)
        months = pd.read_parquet(
            month_file,
            columns=["sequence_id", "position", "raw_current_actual_upb_t"],
        )
        if sequences["sequence_id"].duplicated().any():
            raise ValueError(f"duplicate sequence_id in {sequence_file}")
        # The preprocessing contract writes each partition in
        # (sequence_id, position) order; avoid a redundant 40M-row global sort.
        if np.any(months["position"].to_numpy(dtype=np.int64) < 0):
            raise ValueError(f"{month_file.name}: negative sequence position")
        first = (
            months.drop_duplicates("sequence_id", keep="first")
            .loc[:, ["sequence_id", "raw_current_actual_upb_t"]]
            .rename(columns={"raw_current_actual_upb_t": EXPOSURE_COLUMN})
        )
        merged = sequences.merge(first, on="sequence_id", how="left", validate="one_to_one")
        exposure = pd.to_numeric(merged[EXPOSURE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(exposure)):
            raise ValueError(
                f"{sequence_file.name}: {int(np.sum(~np.isfinite(exposure)))} missing/nonfinite initial UPB"
            )
        if np.any(exposure < 0):
            raise ValueError(f"{sequence_file.name}: negative initial UPB")
        merged.to_parquet(output_sequences / sequence_file.name, index=False)
        summaries.append(exposure)
        sequence_count += len(merged)

    exposure = np.concatenate(summaries)
    if not np.any(exposure > 0):
        raise ValueError("initial UPB exposure has no positive mass")
    contract = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(args.input_root.resolve()),
        "event_stream_identity": "hard-linked byte-identical sequence_months and sequence_tokens parquet",
        "financial_weight_column": EXPOSURE_COLUMN,
        "source_field": "raw_current_actual_upb_t at minimum observed position",
        "availability": "observed in the first performance month, before any future target",
        "semantics": (
            "initial outstanding mortgage principal used as a fixed cluster exposure weight "
            "for the TPP proper scoring loss; not realized loss or profit"
        ),
        "n_sequences": int(sequence_count),
        "zero_count": int(np.sum(exposure == 0)),
        "summary": {
            "mean": float(np.mean(exposure)),
            "median": float(np.median(exposure)),
            "p01": float(np.quantile(exposure, 0.01)),
            "p99": float(np.quantile(exposure, 0.99)),
            "min": float(np.min(exposure)),
            "max": float(np.max(exposure)),
        },
        "outcome_fields_used_to_construct_financial_weight": [],
        "loader_columns_for_financial_experiment": [
            "sequence_id",
            "start_month_index",
            "end_month_index",
            EXPOSURE_COLUMN,
        ],
    }
    (args.output_root / "metadata" / "financial_loss_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
