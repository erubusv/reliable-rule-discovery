from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

from ..data import Dataset
from .config import BASELINE_NAMES, EASYTPP_NAMES, BaselineConfig
from .data import PreparedBaselines, load_landmarks, prepare_baseline_data
from .easytpp import run_easytpp
from .logical import fit_branch_price, fit_neurosymbolic_tpp
from .seed import set_reproducible_seed, validate_seed
from .statistical import (
    fit_logistic,
    fit_point_process_baseline,
    fit_xgboost,
)


RESULT_SCHEMA = "crbstpp.baseline.result.v1"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prepare_baselines(config: BaselineConfig, *, seed: int) -> PreparedBaselines:
    # Data construction itself is label-blind and deterministic.  Setting the
    # declared seed here ensures any dependency invoked by a future converter
    # inherits the same run-wide RNG state.
    set_reproducible_seed(validate_seed(seed))
    return prepare_baseline_data(config)


def _run_directory(config: BaselineConfig, model: str, seed: int) -> Path:
    return config.run_root / config.dataset_id / f"seed-{seed}" / model


def run_baseline(
    config: BaselineConfig,
    model: str,
    *,
    seed: int,
) -> dict[str, object]:
    model = str(model).lower()
    if model not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline: {model}")
    if model not in config.models:
        raise ValueError(f"baseline {model} is disabled by the suite config")
    seed = set_reproducible_seed(validate_seed(seed))
    prepared = prepare_baseline_data(config)
    dataset = Dataset.load(config.dataset)
    output_dir = _run_directory(config, model, seed)
    result_path = output_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"incomplete baseline run exists; inspect before retrying: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "config.json", config.to_dict())
    _atomic_json(
        output_dir / "manifest.json",
        {
            "schema": "crbstpp.baseline.run.v1",
            "model": model,
            "seed": seed,
            "dataset": str(config.dataset),
            "dataset_id": config.dataset_id,
            "dataset_digest": dataset.digest,
            "prepared_manifest": prepared.manifest,
            "python": sys.version,
            "platform": platform.platform(),
        },
    )
    started = time.perf_counter()
    if model in {"logistic", "xgboost", "branch_price", "neurosymbolic_tpp"}:
        landmarks = load_landmarks(prepared.landmarks_path)
    if model == "logistic":
        details = fit_logistic(
            landmarks, config, seed=seed, output_dir=output_dir
        )
    elif model == "xgboost":
        details = fit_xgboost(
            landmarks, config, seed=seed, output_dir=output_dir
        )
    elif model == "baseline_tpp":
        details = fit_point_process_baseline(
            dataset,
            config,
            source_dimension=0,
            output_dir=output_dir,
        )
    elif model == "hawkes":
        details = fit_point_process_baseline(
            dataset,
            config,
            source_dimension=dataset.n_reported_predicates,
            output_dir=output_dir,
        )
    elif model in EASYTPP_NAMES:
        details = run_easytpp(
            model,
            config,
            prepared,
            dataset,
            seed=seed,
            output_dir=output_dir,
        )
    elif model == "branch_price":
        details = fit_branch_price(
            landmarks,
            dataset,
            config,
            seed=seed,
            output_dir=output_dir,
        )
    else:
        details = fit_neurosymbolic_tpp(
            landmarks,
            dataset,
            config,
            seed=seed,
            output_dir=output_dir,
        )
    result = {
        "schema": RESULT_SCHEMA,
        "model": model,
        "seed": seed,
        "dataset_id": config.dataset_id,
        "dataset_digest": dataset.digest,
        "elapsed_seconds": time.perf_counter() - started,
        "details": details,
    }
    _atomic_json(result_path, result)
    return result


def run_suite(config: BaselineConfig, *, seed: int) -> dict[str, object]:
    seed = validate_seed(seed)
    prepare_baselines(config, seed=seed)
    started = time.perf_counter()
    results = {
        model: run_baseline(config, model, seed=seed) for model in config.models
    }
    payload = {
        "schema": "crbstpp.baseline.suite.v1",
        "dataset_id": config.dataset_id,
        "seed": seed,
        "models": list(config.models),
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    root = config.run_root / config.dataset_id / f"seed-{seed}"
    _atomic_json(root / "suite.json", payload)
    return payload


def inspect_baseline(path: Path) -> dict[str, object]:
    path = Path(path)
    if (path / "suite.json").is_file():
        target = path / "suite.json"
    elif (path / "result.json").is_file():
        target = path / "result.json"
    elif (path / "manifest.json").is_file():
        target = path / "manifest.json"
    else:
        raise FileNotFoundError(f"no baseline result in {path}")
    return json.loads(target.read_text(encoding="utf-8"))
