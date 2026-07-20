from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA = "crbstpp.checkpoint.v1"
RESULT_SCHEMA = "crbstpp.result.v2"


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def atomic_text(path: str | Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def load_checkpoint(
    path: str | Path, *, config_digest: str, dataset_digest: str
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("legacy or unsupported checkpoint schema")
    if payload.get("config_digest") != config_digest:
        raise ValueError("checkpoint config digest mismatch")
    if payload.get("dataset_digest") != dataset_digest:
        raise ValueError("checkpoint dataset digest mismatch")
    return payload
