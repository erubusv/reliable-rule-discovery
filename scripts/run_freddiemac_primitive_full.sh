#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/freddiemac/processed/sdq3_primitive_v7_financial}"
OUTPUT="${OUTPUT:-results/freddiemac_primitive_v7_mdl_working_set_full.json}"
LOG="${LOG:-${OUTPUT%.json}.log}"
EXPECTED_PREPROCESSING_SCHEMA=7

# Use the registered budget-free MDL working-set estimator while bounding
# concurrent resident memory.  Worker/cache values control only execution;
# ACTIVE_NEIGHBOR_STRATEGY selects the explicitly reported estimator.
# One independent native float64 solve per physical core.  This host has 12
# physical cores / 24 SMT threads; using 24 solver workers only duplicates
# memory and competes for the same execution units, while six left half the
# physical cores idle during the dominant profile/support waves.
SOLVER_WORKERS="${SOLVER_WORKERS:-12}"
RESPONSE_WORKERS="${RESPONSE_WORKERS:-12}"
# Exact fitting materializes substantially larger closure/design state than
# response pricing. Keep pricing on all cores but admit only three simultaneous
# exact designs, which bounds peak RSS without removing or approximating a fit.
MAX_CONCURRENT_EXACT_FITS="${MAX_CONCURRENT_EXACT_FITS:-3}"
FEATURE_CACHE_GB="${FEATURE_CACHE_GB:-8}"
LOSS_SUMMARY_CACHE_GB="${LOSS_SUMMARY_CACHE_GB:-1}"
FIT_SUMMARY_CACHE_GB="${FIT_SUMMARY_CACHE_GB:-4}"
PERSISTENT_RESPONSE_GB="${PERSISTENT_RESPONSE_GB:-32}"
PERSISTENT_RESPONSE_STORE="${PERSISTENT_RESPONSE_STORE:-results/cache/freddiemac_primitive_v7_responses}"
ACTIVE_NEIGHBOR_STRATEGY="${ACTIVE_NEIGHBOR_STRATEGY:-mdl_score_working_set}"

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"
# Preserve the complete, unbuffered tmux run rather than relying on the
# terminal scrollback.  A new invocation owns a new full-run log.
: > "$LOG"
exec > >(tee "$LOG") 2>&1
echo "[run] started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) root=$ROOT output=$OUTPUT"
on_exit() {
  status=$?
  echo "[run] finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_status=$status"
}
trap on_exit EXIT

# Long-lived shared-memory workers call single-threaded native/MKL kernels.
# This avoids both nested BLAS oversubscription and the copy-on-write cache/
# allocator replication that exhausted memory in the former rolling-process
# path.  These settings alter execution only; skeleton restriction is performed
# separately by the registered sampled/IPW block-MDL screen below.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Bound glibc arena multiplication and return released large NumPy/native
# buffers promptly.  These are allocator controls only and cannot change a
# candidate, objective value, or optimizer solution.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"
export CERTSCR_MALLOC_TRIM="${CERTSCR_MALLOC_TRIM:-1}"
export CERTSCR_MALLOC_TRIM_INTERVAL_SECONDS="${CERTSCR_MALLOC_TRIM_INTERVAL_SECONDS:-30}"
# The former opt-in fork backends duplicate allocator/cache pages and caused
# both historical Freddie full runs to terminate before returning worker
# results.  This runner intentionally uses the exact shared-memory thread
# backends; make that invariant independent of the parent tmux environment.
export CERTSCR_PROCESS_PROFILE=0
export CERTSCR_PROCESS_FITS=0
export CERTSCR_PROCESS_REFIT=0
export CERTSCR_MAX_CONCURRENT_EXACT_FITS="$MAX_CONCURRENT_EXACT_FITS"
echo "[run] strategy=$ACTIVE_NEIGHBOR_STRATEGY solver_workers=$SOLVER_WORKERS response_workers=$RESPONSE_WORKERS max_concurrent_exact_fits=$MAX_CONCURRENT_EXACT_FITS malloc_arena_max=$MALLOC_ARENA_MAX"

if [[ ! -f "$ROOT/metadata/summary.json" ]]; then
  python scripts/preprocess_freddiemac_dynamic_events.py \
    --input-root data/freddiemac \
    --output-root "$ROOT" \
    --vintage 2023Q1 \
    --vintage 2023Q2 \
    --vintage 2023Q3 \
    --vintage 2023Q4 \
    --vintage 2024Q1 \
    --vintage 2024Q2 \
    --vintage 2024Q3 \
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
if metadata.get("predicate_dictionary") != "freddie_primitive_dynamic_v4":
    raise SystemExit("incompatible Freddie predicate dictionary; rebuild preprocessing")
expected_predicates = [
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
]
if metadata.get("predicate_names") != expected_predicates:
    raise SystemExit("incompatible frozen predicate list or order; rebuild preprocessing")
f0 = metadata.get("f0_contract", {})
required_f0 = (
    "dynamic_predicates",
    "outcome_blind_predicate_construction",
    "direct_target_proxy_excluded",
    "strict_future_effect_required",
    "atomic_predicates",
)
if not all(f0.get(key) is True for key in required_f0):
    raise SystemExit("preprocessing artifact does not satisfy the registered F0 contract")
split_group = metadata.get("ordered_split_group", {})
if split_group != {
    "column": "first_payment_month_index",
    "definition": "start_month_index - start_loan_age",
    "outcome_blind": True,
}:
    raise SystemExit("preprocessing artifact lacks the registered monthly cohort split")
if metadata.get("target_process") != "first_event":
    raise SystemExit("Freddie full run requires a first-event target process")
if metadata.get("occurrence_likelihood") != "first_event_cloglog":
    raise SystemExit("incompatible target-process likelihood metadata")
if metadata.get("event_stream_storage") != "predicate_or_target_rows_with_implicit_unit_risk_grid":
    raise SystemExit("Freddie full run requires the exact sparse event-stream artifact")
if metadata.get("target_mark") is not None:
    raise SystemExit("the unmarked runner refuses a marked preprocessing artifact")
root = path.parent.parent
labels = [str(item["vintage"]) for item in metadata.get("vintages", [])]
expected_labels = [
    "2023Q1", "2023Q2", "2023Q3", "2023Q4",
    "2024Q1", "2024Q2", "2024Q3",
]
if labels != expected_labels:
    raise SystemExit(
        f"incompatible administrative-followup vintages: expected {expected_labels}, got {labels}"
    )
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
if sum(int(item["sequence_count"]) for item in metadata["vintages"]) != int(metadata["total_sequences"]):
    raise SystemExit("inconsistent aggregate sequence count in preprocessing metadata")
if sum(int(item["target_sequence_count"]) for item in metadata["vintages"]) != int(metadata["total_target_sequences"]):
    raise SystemExit("inconsistent aggregate target count in preprocessing metadata")
' "$ROOT/metadata/summary.json" "$EXPECTED_PREPROCESSING_SCHEMA"

# The rigorous saturated bound screened 0/111 score-admitted IBM Adds and
# added 0.72 s. Keep it available as an ablation, but skip this measured no-op
# in the fastest runner; disabling a safe screen cannot change an accepted move.
python scripts/certscr_tpp.py \
  --data "$ROOT/sequence_months" \
  --output "$OUTPUT" \
  --predicate-policy freddie_primitive_dynamic_v4 \
  --q-max 3 \
  --impact-lag 12 \
  --knots 4 \
  --max-window 12 \
  --fit-fraction 0.60 \
  --cert-fraction 0.20 \
  --test-fraction 0.20 \
  --split-strategy ordered_group \
  --split-group-column first_payment_month_index \
  --start-age-column start_loan_age \
  --fit-negative-sample-size 30000 \
  --multifidelity-skeleton-screen \
  --resume \
  --identity-profile dictionary_mdl \
  --triplet-generation all \
  --support-search active_set \
  --active-start-policy all_atoms \
  --active-neighbor-strategy "$ACTIVE_NEIGHBOR_STRATEGY" \
  --no-conditional-safe-mdl-screen \
  --support-family terminal_atoms \
  --no-safe-mdl-screen \
  --no-target-history-control \
  --loan-age-baseline \
  --occurrence-likelihood first_event_cloglog \
  --certification-mode early_warning \
  --adverse-event-name "first serious mortgage delinquency (90+ DPD or REO acquisition)" \
  --early-warning-horizon 12 \
  --device cpu \
  --solver-dtype float64 \
  --solver-workers "$SOLVER_WORKERS" \
  --response-workers "$RESPONSE_WORKERS" \
  --feature-cache-gb "$FEATURE_CACHE_GB" \
  --persistent-response-store "$PERSISTENT_RESPONSE_STORE" \
  --persistent-response-gb "$PERSISTENT_RESPONSE_GB" \
  --loss-summary-cache-gb "$LOSS_SUMMARY_CACHE_GB" \
  --fit-summary-cache-gb "$FIT_SUMMARY_CACHE_GB"
