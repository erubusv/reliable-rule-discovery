from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Certificate:
    support_key: str
    f0: bool
    f1_pvalue: float | None
    f2_pvalues: tuple[float, ...]
    f3: bool
    family_pvalue: float | None
    holm_adjusted_pvalue: float | None
    certified: bool
    reasons: tuple[str, ...] = ()
    family_adjusted_pvalue: float | None = None
    multiplicity_method: str = "romano_wolf_stepdown_max_t"
    romano_wolf_resamples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunReport:
    schema: str
    algorithm: str
    config_digest: str
    dataset_digest: str
    support_count: int
    certified_count: int
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
