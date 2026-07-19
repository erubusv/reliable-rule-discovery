from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Complete, serializable configuration for one CRBS-TPP run."""

    dataset: str
    run_root: str = "runs/crbstpp"
    run_id: str | None = None
    q_max: int = 3
    impact_lag: int = 12
    knot_count: int = 4
    formation_windows: tuple[int, ...] = tuple(range(13))
    split_fractions: tuple[float, float, float] = (0.60, 0.20, 0.20)
    split_seed: int = 111
    alpha: float = 0.05
    early_warning_horizon: int = 12
    probability_materiality: float = 0.0
    solver_tolerance: float = 2.0e-7
    solver_max_iter: int = 100
    search_tolerance: float = 1.0e-8
    exact_workers: int = 3
    pricing_workers: int = 12
    pricing_devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    cache_bytes: int = 8 * 1024**3
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("run config must be a YAML mapping")
        for name in ("formation_windows", "split_fractions", "pricing_devices"):
            if name in payload:
                payload[name] = tuple(payload[name])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.dataset:
            raise ValueError("dataset is required")
        if not 1 <= self.q_max <= 3:
            raise ValueError("q_max must be in [1, 3]")
        if self.impact_lag < 1 or self.knot_count < 1:
            raise ValueError("impact_lag and knot_count must be positive")
        windows = tuple(int(value) for value in self.formation_windows)
        if not windows or windows != tuple(sorted(set(windows))) or windows[0] < 0:
            raise ValueError("formation_windows must be sorted, unique and nonnegative")
        if len(self.split_fractions) != 3 or any(
            value <= 0 for value in self.split_fractions
        ):
            raise ValueError("three positive split fractions are required")
        if not math.isclose(sum(self.split_fractions), 1.0, abs_tol=1.0e-12):
            raise ValueError("split fractions must sum to one")
        if not 0 < self.alpha < 0.5:
            raise ValueError("alpha must lie in (0, 0.5)")
        if not 1 <= self.early_warning_horizon <= self.impact_lag:
            raise ValueError("early_warning_horizon must lie in [1, impact_lag]")
        if not 0 <= self.probability_materiality < 1:
            raise ValueError("probability_materiality must lie in [0, 1)")
        if self.solver_tolerance <= 0 or self.solver_max_iter < 1:
            raise ValueError("invalid solver controls")
        if self.search_tolerance < 0:
            raise ValueError("search_tolerance must be nonnegative")
        if self.exact_workers < 1 or self.pricing_workers < 1:
            raise ValueError("worker counts must be positive")
        if self.exact_workers > 3:
            raise ValueError("exact fit concurrency is capped at three")
        for device in self.pricing_devices:
            if device != "cpu" and not (
                device.startswith("cuda:") and device[5:].isdigit()
            ):
                raise ValueError(f"invalid pricing device: {device}")
        if self.cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
