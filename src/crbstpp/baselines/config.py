from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


BASELINE_NAMES = (
    "logistic",
    "xgboost",
    "baseline_tpp",
    "hawkes",
    "rmtpp",
    "nhp",
    "thp",
    "attnhp",
    "branch_price",
    "neurosymbolic_tpp",
)
EASYTPP_NAMES = frozenset(("rmtpp", "nhp", "thp", "attnhp"))


def _tuple_of_floats(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    output = tuple(float(item) for item in value)
    if any(item <= 0.0 for item in output):
        raise ValueError(f"{name} must contain positive values")
    return output


@dataclass(frozen=True)
class BaselineConfig:
    """Dataset-independent contract for every comparison model.

    A random seed is deliberately not stored here.  It is supplied by the
    command line for each run, recorded in the result manifest, and forwarded
    unchanged to every library involved in that run.
    """

    dataset: Path
    dataset_id: str
    run_root: Path = Path("runs/baselines")
    models: tuple[str, ...] = BASELINE_NAMES
    warning_horizon: int = 1
    history_horizon: int = 1
    effect_horizon: int = 1
    baseline_time_bins: int = 1
    cache_bytes: int = 2 * 1024**3
    device: str = "cuda:0"
    num_workers: int = 8
    max_sequence_length: int = 512
    sequence_context_length: int = 128
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 5
    learning_rate: float = 1.0e-3
    hidden_sizes: tuple[int, ...] = (32, 64)
    logistic_c: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    xgboost_depths: tuple[int, ...] = (4, 6)
    xgboost_learning_rates: tuple[float, ...] = (0.05, 0.1)
    hawkes_half_lives: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0)
    logical_max_rules: int = 20
    logical_max_order: int = 3
    logical_time_limit_seconds: float = 3600.0
    easytpp_version: str = "0.3.0"
    easytpp_commit: str = "a3466985e76357b3dd235da845ec69e7a9fee0f8"
    branch_price_commit: str = "4160b27bb4606e87129bd36985fbba2ccdb9c925"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.models) - set(BASELINE_NAMES)
        if unknown:
            raise ValueError(f"unknown baselines: {sorted(unknown)}")
        if len(set(self.models)) != len(self.models):
            raise ValueError("baseline model names must be unique")
        if not self.dataset_id or any(char.isspace() for char in self.dataset_id):
            raise ValueError("dataset_id must be a nonempty path-safe label")
        for name in (
            "warning_horizon",
            "history_horizon",
            "effect_horizon",
            "baseline_time_bins",
            "num_workers",
            "max_sequence_length",
            "batch_size",
            "max_epochs",
            "patience",
            "logical_max_rules",
            "logical_max_order",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.logical_max_order <= 3:
            raise ValueError("logical_max_order must lie in [1, 3]")
        if not 0 <= self.sequence_context_length < self.max_sequence_length:
            raise ValueError("sequence context must be smaller than sequence length")
        if self.device != "cpu" and not self.device.startswith("cuda:"):
            raise ValueError("device must be cpu or cuda:N")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaselineConfig":
        path = Path(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("baseline config must be a YAML mapping")
        if "seed" in payload:
            raise ValueError("seed belongs on the CLI, not in a baseline config")
        root = path.parent
        dataset = Path(str(payload.pop("dataset")))
        run_root = Path(str(payload.pop("run_root", "runs/baselines")))
        if not dataset.is_absolute():
            candidate = (root / dataset).resolve()
            dataset = candidate if candidate.exists() else dataset
        conversions = {
            "models": lambda value: tuple(str(item).lower() for item in value),
            "hidden_sizes": lambda value: tuple(int(item) for item in value),
            "logistic_c": lambda value: _tuple_of_floats(value, "logistic_c"),
            "xgboost_depths": lambda value: tuple(int(item) for item in value),
            "xgboost_learning_rates": lambda value: _tuple_of_floats(
                value, "xgboost_learning_rates"
            ),
            "hawkes_half_lives": lambda value: _tuple_of_floats(
                value, "hawkes_half_lives"
            ),
        }
        for name, convert in conversions.items():
            if name in payload:
                payload[name] = convert(payload[name])
        payload["dataset"] = dataset
        payload["run_root"] = run_root
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.__dict__)
        payload["dataset"] = str(self.dataset)
        payload["run_root"] = str(self.run_root)
        for name in (
            "models",
            "hidden_sizes",
            "logistic_c",
            "xgboost_depths",
            "xgboost_learning_rates",
            "hawkes_half_lives",
        ):
            payload[name] = list(payload[name])
        return payload
