#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export PYTHONPATH=scripts
export PYTHONHASHSEED=0
# Physical-core workers independently consume exact skeleton fits.
# Prevent each worker from recursively creating a full BLAS thread pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RAW_ROOT="data/home_credit_default_risk/kagglehub_cache/competitions/home-credit-default-risk"
DATA_ROOT="data/home_credit_default_risk/processed/tpp_v9_cash_behavioral_nonproxy_expanded_event_stream_marked"
OUTPUT="results/certscr_home_credit_marked_capfree_full.json"
LOG="results/certscr_home_credit_marked_capfree_full.log"

mkdir -p results
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -u +%FT%TZ)] preprocessing marked Home Credit data"
python scripts/preprocess_home_credit_tpp.py \
  --raw-root "${RAW_ROOT}" \
  --output-root "${DATA_ROOT}" \
  --overwrite \
  --current-contract-type "Cash loans" \
  --predicate-tier behavioral_nonproxy_expanded \
  --sparse-events \
  --target-mode event_stream \
  --financial-mark-contract installment_shortfall

echo "[$(date -u +%FT%TZ)] starting cap-free marked CertSCR-TPP full run"
python scripts/certscr_tpp.py \
  --data "${DATA_ROOT}/sequence_months" \
  --output "${OUTPUT}" \
  --predicate-policy home_credit_behavioral_nonproxy_expanded \
  --mark-column target_mark_values \
  --q-max 3 \
  --impact-lag 12 \
  --knots 4 \
  --max-window 12 \
  --fit-fraction 0.40 \
  --cert-fraction 0.40 \
  --test-fraction 0.20 \
  --stratify-target-sequences \
  --fit-negative-sample-size 15000 \
  --hybrid-full-acceptance \
  --identity-profile dictionary_mdl \
  --triplet-generation all \
  --support-search active_set \
  --active-restarts 8 \
  --response-workers 8 \
  --device cpu \
  --solver-dtype float64 \
  --feature-cache-gb 32 \
  --alpha-fit-screen 0.05 \
  --alpha-family 0.05

echo "[$(date -u +%FT%TZ)] completed: ${OUTPUT}"
