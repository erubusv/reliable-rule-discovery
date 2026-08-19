#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ACTION="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

SEEDS=(111 222 333)
if [[ "${1:-}" == "--seed" ]]; then
    [[ $# -eq 2 ]] || { echo "--seed requires one integer" >&2; exit 2; }
    SEEDS=("$2")
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [download|prepare|configure|rules|baselines|metrics|all] [--seed INTEGER]" >&2
    exit 2
fi

RAW_AAVE="data/aave/raw/ethereum_v2_v3"
RAW_WSELOB="data/wselob_2017/raw"

aave_dataset() {
    printf 'data/crbstpp/reproduction/aave_seed%s' "$1"
}

wselob_dataset() {
    printf 'data/crbstpp/reproduction/wselob_seed%s' "$1"
}

seed_root() {
    printf 'runs/reproduction/seed-%s' "$1"
}

rpc_args=()
if [[ -n "${AAVE_RPC_URL:-}" ]]; then
    rpc_args=(--rpc-url "${AAVE_RPC_URL}")
fi

download_raw() {
    crbstpp preprocess aave \
        --download \
        --raw-root "${RAW_AAVE}" \
        --end-block 25660939 \
        --chunk-size 10000 \
        --workers 4 \
        "${rpc_args[@]}"
    for stock in PEKAO KGHM PKNORLEN; do
        crbstpp preprocess wselob \
            --download \
            --raw-root "${RAW_WSELOB}" \
            --stock "${stock}"
    done
}

prepare_seed() {
    local seed="$1"
    local aave_output wselob_output stock lower part
    local -a parts
    aave_output="$(aave_dataset "${seed}")"
    wselob_output="$(wselob_dataset "${seed}")"

    if [[ ! -f "${aave_output}/manifest.json" ]]; then
        crbstpp preprocess aave \
            --build-dataset \
            --raw-root "${RAW_AAVE}" \
            --output-root "${aave_output}" \
            --partition-seed "${seed}" \
            --partition-fractions 0.5 0.3 0.2 \
            --workers 4 \
            "${rpc_args[@]}"
    fi

    parts=()
    for stock in PEKAO KGHM PKNORLEN; do
        lower="$(printf '%s' "${stock}" | tr '[:upper:]' '[:lower:]')"
        part="data/crbstpp/reproduction/wselob_${lower}_seed${seed}"
        parts+=("${part}")
        if [[ ! -f "${part}/manifest.json" ]]; then
            crbstpp preprocess wselob \
                --build-dataset \
                --raw-root "${RAW_WSELOB}" \
                --output-root "${part}" \
                --stock "${stock}" \
                --impact-seconds 30 \
                --continuous-time-unit millisecond \
                --kernel-knots 4 \
                --baseline-bins 4 \
                --partition-fractions 0.5 0.3 0.2 \
                --partition-method month_stratified \
                --partition-seed "${seed}" \
                --diagnostic-max-days 30 \
                --target-horizon-seconds 30 \
                --target-quantile 0.90 \
                --target-rearm-fraction 0.50
        fi
    done
    if [[ ! -f "${wselob_output}/manifest.json" ]]; then
        crbstpp preprocess wselob \
            --merge-input-root "${parts[@]}" \
            --output-root "${wselob_output}"
    fi
}

materialize_configs() {
    local seed="$1"
    local root config_root
    root="$(seed_root "${seed}")"
    config_root="${root}/configs"
    mkdir -p "${config_root}"
    python - "${seed}" "${config_root}" "$(aave_dataset "${seed}")" \
        "$(wselob_dataset "${seed}")" "${root}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

seed = int(sys.argv[1])
output = Path(sys.argv[2])
aave_data = sys.argv[3]
wselob_data = sys.argv[4]
root = Path(sys.argv[5])


def read(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write(name: str, payload: dict) -> None:
    (output / name).write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


for dataset, template, data in (
    ("aave", "configs/experiments/aave.yaml", aave_data),
    ("wselob", "configs/experiments/wselob.yaml", wselob_data),
):
    payload = read(template)
    payload.update(
        dataset=data,
        run_root=str(root / "rules"),
        run_id=dataset,
        split_seed=seed,
        discovery_sampling_seed=seed,
        romano_wolf_seed=seed,
    )
    write(f"{dataset}.yaml", payload)

for name, data, dataset_id in (
    ("aave", aave_data, "aave"),
    ("aave_nhp", aave_data, "aave"),
    ("aave_attnhp", aave_data, "aave"),
    ("wselob", wselob_data, "wselob"),
    ("wselob_attnhp", wselob_data, "wselob"),
):
    payload = read(f"configs/baselines/{name}.yaml")
    payload.update(
        dataset=data,
        dataset_id=dataset_id,
        run_root=str(root / "baselines"),
    )
    write(f"baseline_{name}.yaml", payload)

write(
    "metrics.yaml",
    {
        "output_dir": str(root / "metrics"),
        "datasets": [
            {
                "key": "aave",
                "label": "Aave",
                "baseline_seed_dir": str(root / "baselines" / "aave" / f"seed-{seed}"),
                "ours_run_dir": str(root / "rules" / "aave"),
            },
            {
                "key": "wselob",
                "label": "WSELOB",
                "baseline_seed_dir": str(root / "baselines" / "wselob" / f"seed-{seed}"),
                "ours_run_dir": str(root / "rules" / "wselob"),
            },
        ],
    },
)
PY
}

fit_models() {
    local seed="$1" config="$2"
    shift 2
    for model in "$@"; do
        crbstpp baseline fit --config "${config}" --model "${model}" --seed "${seed}"
    done
}

fit_rules() {
    local seed="$1" config_root
    config_root="$(seed_root "${seed}")/configs"
    crbstpp fit --config "${config_root}/aave.yaml"
    crbstpp fit --config "${config_root}/wselob.yaml"
}

fit_baselines() {
    local seed="$1" config_root
    config_root="$(seed_root "${seed}")/configs"
    fit_models "${seed}" "${config_root}/baseline_aave.yaml" \
        logistic xgboost hawkes rmtpp thp branch_price neurosymbolic_tpp
    fit_models "${seed}" "${config_root}/baseline_aave_nhp.yaml" nhp
    fit_models "${seed}" "${config_root}/baseline_aave_attnhp.yaml" attnhp

    fit_models "${seed}" "${config_root}/baseline_wselob.yaml" \
        logistic xgboost hawkes rmtpp nhp thp branch_price neurosymbolic_tpp
    fit_models "${seed}" "${config_root}/baseline_wselob_attnhp.yaml" attnhp
}

evaluate_seed() {
    local seed="$1" root config_root aave_run wselob_run temporary
    root="$(seed_root "${seed}")"
    config_root="${root}/configs"
    aave_run="${root}/rules/aave"
    wselob_run="${root}/rules/wselob"

    temporary="${aave_run}/integrated_landmark_metrics.json.tmp"
    python tools/integrated_rule_metrics.py \
        --run-dir "${aave_run}" \
        --baseline-config "${config_root}/baseline_aave.yaml" \
        > "${temporary}"
    mv "${temporary}" "${aave_run}/integrated_landmark_metrics.json"

    temporary="${wselob_run}/integrated_landmark_metrics.json.tmp"
    python tools/integrated_rule_metrics.py \
        --run-dir "${wselob_run}" \
        --baseline-config "${config_root}/baseline_wselob.yaml" \
        > "${temporary}"
    mv "${temporary}" "${wselob_run}/integrated_landmark_metrics.json"

    python tools/evaluate_hawkes_landmarks.py \
        --config "${config_root}/baseline_aave.yaml" --seed "${seed}" >/dev/null
    python tools/evaluate_hawkes_landmarks.py \
        --config "${config_root}/baseline_wselob.yaml" --seed "${seed}" >/dev/null

    python tools/evaluate_easytpp_target.py \
        --config "${config_root}/baseline_aave.yaml" --model rmtpp --seed "${seed}" --device cuda:0 >/dev/null
    python tools/evaluate_easytpp_target.py \
        --config "${config_root}/baseline_aave_nhp.yaml" --model nhp --seed "${seed}" --device cuda:0 >/dev/null
    python tools/evaluate_easytpp_target.py \
        --config "${config_root}/baseline_aave.yaml" --model thp --seed "${seed}" --device cuda:0 >/dev/null
    python tools/evaluate_easytpp_target.py \
        --config "${config_root}/baseline_aave_attnhp.yaml" --model attnhp --seed "${seed}" --device cuda:1 >/dev/null

    python tools/evaluate_easytpp_target.py \
        --config "${config_root}/baseline_wselob.yaml" --model rmtpp --seed "${seed}" --device cuda:0 >/dev/null
    python tools/evaluate_easytpp_target.py \
        --config "${config_root}/baseline_wselob.yaml" --model nhp --seed "${seed}" --device cuda:0 >/dev/null
    python tools/evaluate_easytpp_target.py \
        --config "${config_root}/baseline_wselob.yaml" --model thp --seed "${seed}" --device cuda:0 >/dev/null
    python tools/evaluate_easytpp_target.py \
        --config "${config_root}/baseline_wselob_attnhp.yaml" --model attnhp --seed "${seed}" --device cuda:1 >/dev/null

    crbstpp metrics --spec "${config_root}/metrics.yaml" >/dev/null
}

case "${ACTION}" in
    download)
        download_raw
        ;;
    prepare|configure|rules|baselines|metrics|all)
        for seed in "${SEEDS[@]}"; do
            if [[ "${ACTION}" == "prepare" || "${ACTION}" == "all" ]]; then
                prepare_seed "${seed}"
            fi
            materialize_configs "${seed}"
            if [[ "${ACTION}" == "rules" || "${ACTION}" == "all" ]]; then
                fit_rules "${seed}"
            fi
            if [[ "${ACTION}" == "baselines" || "${ACTION}" == "all" ]]; then
                fit_baselines "${seed}"
            fi
            if [[ "${ACTION}" == "metrics" || "${ACTION}" == "all" ]]; then
                evaluate_seed "${seed}"
            fi
        done
        ;;
    *)
        echo "usage: $0 [download|prepare|configure|rules|baselines|metrics|all] [--seed INTEGER]" >&2
        exit 2
        ;;
esac

if [[ ("${ACTION}" == "metrics" || "${ACTION}" == "all") && ${#SEEDS[@]} -gt 1 ]]; then
    metric_inputs=()
    for seed in "${SEEDS[@]}"; do
        metric_inputs+=("$(seed_root "${seed}")/metrics/metrics.json")
    done
    python tools/aggregate_seed_metrics.py \
        --input "${metric_inputs[@]}" \
        --output runs/reproduction/metrics_three_seeds.json
fi
