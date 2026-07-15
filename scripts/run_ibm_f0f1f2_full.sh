#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export PYTHONPATH=scripts
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
# Twelve independent exact-fit workers occupy the physical cores.  Keeping
# each BLAS call single-threaded prevents nested oversubscription; this changes
# scheduling only, never the model, dictionary, or acceptance rule.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RAW_ZIP="data/ibm_aml/raw/HI-Small_Trans.csv.zip"
export DATA_ROOT="data/ibm_aml/processed/hi_small_tpp_dynamic_nonproxy_v4_f0_clean"
OUTPUT="results/certscr_ibm_dynamic_nonproxy_v4_f0_clean_full_schema11.json"
LOG="results/certscr_ibm_dynamic_nonproxy_v4_f0_clean_full_schema11.log"

mkdir -p results
exec > >(tee "${LOG}") 2>&1

echo "[$(date -u +%FT%TZ)] preprocessing clean IBM recurrent event stream"
python scripts/preprocess_ibm_aml_tpp.py \
  --raw-zip "${RAW_ZIP}" \
  --output-root "${DATA_ROOT}" \
  --overwrite \
  --time-unit hour \
  --predicate-tier dynamic_nonproxy_candidate \
  --analysis-start-frac 0.2 \
  --include-laundering-transaction-predicates

# Fail before any expensive fit unless the serialized artifact itself proves
# the current target-blind recurrent-history contract.
python - <<'PY'
import json
import os
from pathlib import Path

from certscr.predicate_policy import resolve_predicate_policy

root = Path(os.environ["DATA_ROOT"])
summary = json.loads((root / "metadata" / "summary.json").read_text())
f0 = summary.get("f0_contract", {})
leakage = summary.get("leakage_policy", {})
required = set(resolve_predicate_policy("ibm_aml_dynamic_nonproxy_v3"))
available = set(summary.get("predicate_names", []))
assert summary.get("target_process") == "recurrent"
assert leakage.get("laundering_transaction_predicates") is True
assert leakage.get("is_laundering") == "used only as target_token"
assert f0.get("dynamic_predicates") is True
assert f0.get("outcome_blind_predicate_construction") is True
assert f0.get("direct_target_proxy_excluded") is True
assert f0.get("predicate_history_includes_target_labeled_observations") is True
assert f0.get("strict_future_effect_required") is True
assert required <= available
print({
    "preprocessing_contract": "passed",
    "target_process": summary["target_process"],
    "sequences": summary["n_sequences"],
    "target_events": summary["n_target_rows"],
    "available_predicates": len(available),
    "registered_rule_predicates": len(required),
})
PY

echo "[$(date -u +%FT%TZ)] starting IBM recurrent F0/F1/F2 full run"
# The completed IBM search audit screened 0/3017 supports with the global
# bound. Direct exact fitting keeps the same feasible family and optimizer.
# The frozen D_fit graph has about 3,377 reusable model keys and 10,016
# repeated references.  The summary byte caps fit the audited 125-GiB host and
# avoid most reconstruction; they affect memoization only.
python scripts/certscr_tpp.py \
  --data "${DATA_ROOT}/sequence_months" \
  --output "${OUTPUT}" \
  --predicate-policy ibm_aml_dynamic_nonproxy_v3 \
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
  --target-history-control \
  --certification-mode early_warning \
  --adverse-event-name "outgoing money-laundering transaction" \
  --early-warning-horizon 12 \
  --early-warning-threshold 0 \
  --solver-workers 12 \
  --response-workers 8 \
  --device cpu \
  --solver-dtype float64 \
  --feature-cache-gb 32 \
  --loss-summary-cache-gb 2 \
  --fit-summary-cache-gb 16 \
  --alpha-fit-screen 0.05 \
  --alpha-family 0.05

echo "[$(date -u +%FT%TZ)] completed: ${OUTPUT}"
