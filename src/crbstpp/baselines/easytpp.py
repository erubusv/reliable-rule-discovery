from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import yaml

from ..data import Dataset
from .config import BaselineConfig, EASYTPP_NAMES
from .data import PreparedBaselines
from .seed import set_reproducible_seed


MODEL_IDS = {
    "rmtpp": "RMTPP",
    "nhp": "NHP",
    "thp": "THP",
    "attnhp": "AttNHP",
}


def _gpu_index(device: str) -> int:
    return -1 if device == "cpu" else int(device.split(":", 1)[1])


def _model_config(model: str, hidden_size: int) -> dict[str, object]:
    common: dict[str, object] = {
        "hidden_size": int(hidden_size),
        "loss_integral_num_sample_per_step": 20,
        "use_ln": False,
        "thinning": {
            "num_seq": 10,
            "num_sample": 1,
            "num_exp": 500,
            "look_ahead_time": 10,
            "patience_counter": 5,
            "over_sample_rate": 5,
            "num_samples_boundary": 5,
            "dtime_max": 5,
            "num_step_gen": 1,
        },
    }
    if model == "rmtpp":
        common.update(
            {
                "time_emb_size": 16,
                "num_layers": 2,
                "num_heads": 2,
                "mc_num_sample_per_step": 20,
                "sharing_param_layer": False,
                "dropout": 0.0,
            }
        )
    elif model == "nhp":
        pass
    elif model == "thp":
        common.update(
            {
                "time_emb_size": 16,
                "num_layers": 2,
                "num_heads": 2,
                "mc_num_sample_per_step": 20,
            }
        )
    elif model == "attnhp":
        common.update({"time_emb_size": 4, "num_layers": 2, "num_heads": 2})
        common["loss_integral_num_sample_per_step"] = 10
    else:
        raise ValueError(f"not an EasyTPP model: {model}")
    return common


def easytpp_yaml(
    model: str,
    config: BaselineConfig,
    prepared: PreparedBaselines,
    dataset: Dataset,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    if model not in EASYTPP_NAMES:
        raise ValueError(f"not an EasyTPP model: {model}")
    experiment_id = f"{MODEL_IDS[model]}_train"
    data_id = config.dataset_id
    return {
        "pipeline_config_id": "runner_config",
        "data": {
            data_id: {
                "data_format": "pkl",
                "train_dir": str(prepared.easytpp_path.resolve()),
                "valid_dir": str(prepared.easytpp_path.resolve()),
                "test_dir": str(prepared.easytpp_path.resolve()),
                "data_specs": {
                    "num_event_types": dataset.n_reported_predicates + 1,
                    "pad_token_id": dataset.n_reported_predicates + 1,
                    "padding_side": "right",
                    "truncation_side": "right",
                # Sequences are already segmented at ``max_len``.  Padding to
                # the longest sequence in each batch is therefore exactly the
                # same masked likelihood with substantially less wasted work
                # than padding every short sequence to ``max_len``.
                "padding_strategy": "longest",
                    "max_len": config.max_sequence_length,
                    "rescale_time": True,
                },
            }
        },
        experiment_id: {
            "base_config": {
                "stage": "train",
                "backend": "torch",
                "dataset_id": data_id,
                "runner_id": "std_tpp",
                "model_id": MODEL_IDS[model],
                "base_dir": str(output_dir.resolve()),
            },
            "trainer_config": {
                "batch_size": config.batch_size,
                "max_epoch": config.max_epochs,
                "shuffle": False,
                "optimizer": "adam",
                "learning_rate": config.learning_rate,
                "valid_freq": 1,
                "use_tfb": False,
                "metrics": ["acc", "rmse"],
                "seed": seed,
                "gpu": _gpu_index(config.device),
            },
            "model_config": _model_config(model, config.hidden_sizes[0]),
        },
    }


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_easytpp(
    model: str,
    config: BaselineConfig,
    prepared: PreparedBaselines,
    dataset: Dataset,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Run EasyTPP while evaluating D_test only after cert selection.

    EasyTPP's public ``Runner.train`` evaluates its configured test loader at
    every validation epoch.  We call the same official training and evaluation
    routines separately so D_test is touched once, after the best D_cert
    checkpoint has been restored.
    """

    try:
        installed = importlib.metadata.version("easy-tpp")
        from easy_tpp.config_factory import Config
        from easy_tpp.runner import Runner
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            "neural TPP baselines require `pip install -e '.[baselines]'`"
        ) from error
    if installed != config.easytpp_version:
        raise RuntimeError(
            f"EasyTPP version mismatch: expected {config.easytpp_version}, got {installed}"
        )
    set_reproducible_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"{MODEL_IDS[model]}_train"
    trials: list[dict[str, object]] = []
    for hidden_size in config.hidden_sizes:
        # Every architecture receives the identical CLI seed.  D_cert chooses
        # hidden width, and D_test is not constructed until that choice is
        # frozen.
        set_reproducible_seed(seed)
        trial_dir = output_dir / "trials" / f"hidden-{hidden_size}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_config = BaselineConfig(
            **{
                **config.__dict__,
                "hidden_sizes": (int(hidden_size),),
            }
        )
        payload = easytpp_yaml(
            model,
            trial_config,
            prepared,
            dataset,
            seed=seed,
            output_dir=trial_dir,
        )
        config_path = trial_dir / "easytpp.yaml"
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        runner_config = Config.build_from_yaml_file(
            str(config_path), experiment_id=experiment_id
        )
        runner = Runner.build_from_config(runner_config)
        train_loader = runner._data_loader.train_loader(shuffle=False)
        cert_loader = runner._data_loader.valid_loader(shuffle=False)
        runner._train_model(train_loader, cert_loader, test_loader=None)
        model_dir = runner.get_model_dir()
        runner._load_model(model_dir)
        cert_metrics = runner._evaluate_model(cert_loader)
        trials.append(
            {
                "hidden_size": int(hidden_size),
                "cert": _jsonable(cert_metrics),
                "checkpoint": str(model_dir),
                "config": str(config_path),
            }
        )
        del runner
    selected = max(
        trials,
        key=lambda trial: (
            float(dict(trial["cert"])["loglike"]),
            -int(trial["hidden_size"]),
        ),
    )
    runner_config = Config.build_from_yaml_file(
        str(selected["config"]), experiment_id=experiment_id
    )
    runner = Runner.build_from_config(runner_config)
    model_dir = str(selected["checkpoint"])
    runner._load_model(model_dir)
    test_loader = runner._data_loader.test_loader(shuffle=False)
    test_metrics = runner._evaluate_model(test_loader)
    result = {
        "implementation": "EasyTPP official PyTorch implementation",
        "easy_tpp_version": installed,
        "easy_tpp_commit": config.easytpp_commit,
        "model_id": MODEL_IDS[model],
        "selected_hidden_size": int(selected["hidden_size"]),
        "cert_trials": trials,
        "seed": seed,
        "test": _jsonable(test_metrics),
        "metric_scope": (
            "official EasyTPP joint multitype event log-likelihood; "
            "reported separately from target-only CRBS-TPP NLL"
        ),
        "checkpoint": model_dir,
        "test_evaluation_count": 1,
        "input_contract": (
            "reported predicate types plus target type in frozen Gatech sequences"
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
