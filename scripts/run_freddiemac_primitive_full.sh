#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/freddiemac/processed/sdq3_primitive_v5_sparse}"
OUTPUT="${OUTPUT:-results/freddiemac_primitive_v5_sparse_full.json}"
EXPECTED_PREPROCESSING_SCHEMA=5

# Twelve independent small convex fits should occupy the physical cores.
# Nested BLAS pools only contend for those same cores and can otherwise turn
# 12 exact workers into 144 runnable threads.  These settings alter scheduling
# only; every objective, candidate and KKT tolerance is unchanged.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -f "$ROOT/metadata/summary.json" ]]; then
  python scripts/preprocess_freddiemac_dynamic_events.py \
    --input-root data/freddiemac \
    --output-root "$ROOT" \
    --skip-sequence-tokens
fi

python -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
metadata = json.loads(path.read_text())
actual = metadata.get("preprocessing_schema_version")
if actual != expected:
    raise SystemExit(
        f"incompatible preprocessing schema at {path}: expected {expected}, got {actual}; "
        "rebuild with --overwrite"
    )
if metadata.get("predicate_dictionary") != "freddie_primitive_dynamic_v3":
    raise SystemExit("incompatible Freddie predicate dictionary; rebuild preprocessing")
if metadata.get("occurrence_likelihood") != "first_event_cloglog":
    raise SystemExit("incompatible target-process likelihood metadata")
if metadata.get("event_stream_storage") != "predicate_or_target_rows_with_implicit_unit_risk_grid":
    raise SystemExit("Freddie full run requires the exact sparse event-stream artifact")
if metadata.get("target_mark") is not None:
    raise SystemExit("the unmarked runner refuses a marked preprocessing artifact")
root = path.parent.parent
labels = [str(item["vintage"]) for item in metadata.get("vintages", [])]
if not labels:
    raise SystemExit("preprocessing summary contains no vintages")
missing = [
    str(candidate)
    for label in labels
    for candidate in (
        root / "sequence_months" / f"part-{label}.parquet",
        root / "sequences" / f"part-{label}.parquet",
    )
    if not candidate.is_file()
]
if missing:
    raise SystemExit("incomplete preprocessing artifact: " + ", ".join(missing))
' "$ROOT/metadata/summary.json" "$EXPECTED_PREPROCESSING_SCHEMA"

python scripts/certscr_tpp.py \
  --data "$ROOT/sequence_months" \
  --output "$OUTPUT" \
  --predicate-policy freddie_primitive_dynamic_v3 \
  --q-max 3 \
  --impact-lag 12 \
  --knots 4 \
  --max-window 12 \
  --fit-fraction 0.60 \
  --cert-fraction 0.20 \
  --test-fraction 0.20 \
  --stratify-target-sequences \
  --identity-profile dictionary_mdl \
  --triplet-generation all \
  --support-search active_set \
  --active-start-policy all_atoms \
  --support-family terminal_atoms \
  --no-safe-mdl-screen \
  --no-target-history-control \
  --occurrence-likelihood first_event_cloglog \
  --certification-mode early_warning \
  --adverse-event-name "first serious mortgage delinquency (90+ DPD or REO acquisition)" \
  --early-warning-horizon 12 \
  --device cpu \
  --solver-dtype float64 \
  --solver-workers 12 \
  --response-workers 12 \
  --feature-cache-gb 32 \
  --loss-summary-cache-gb 2 \
  --fit-summary-cache-gb 16
