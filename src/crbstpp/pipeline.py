from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from .certification import certify_family
from .checkpoint import (
    CHECKPOINT_SCHEMA,
    RESULT_SCHEMA,
    atomic_json,
    atomic_text,
    load_checkpoint,
)
from .config import RunConfig
from .data import Dataset
from .ensemble import fit_ensemble
from .objective import SupportRecord
from .report import RunReport
from .response import Context
from .rules import RuleIdentity, Support
from .search import SupportOptimizer, support_key


def _rule_payload(rule: RuleIdentity) -> dict[str, object]:
    return {
        "antecedent": list(rule.antecedent),
        "window": rule.window,
        "sign": rule.sign,
    }


def _support_payload(support: Support) -> list[dict[str, object]]:
    return [_rule_payload(rule) for rule in support.rules]


def _support_from_payload(payload: list[dict[str, object]]) -> Support:
    return Support.of(
        RuleIdentity(
            tuple(int(value) for value in item["antecedent"]),
            int(item["window"]),
            int(item["sign"]),
        )
        for item in payload
    )


def _record_payload(
    record: SupportRecord, predicate_names: tuple[str, ...]
) -> dict[str, object]:
    return {
        "key": support_key(record.support),
        "rules": [
            {
                **_rule_payload(rule),
                "antecedent_names": [
                    predicate_names[index] for index in rule.antecedent
                ],
                "kernel": record.fit.coefficients[block].tolist(),
                "reported": True,
            }
            for rule, block in zip(
                record.support.rules, record.matrix.rule_slices, strict=True
            )
        ],
        "closure": [
            {
                "antecedent": list(term.antecedent),
                "window": term.window,
                "kernel": record.fit.coefficients[
                    record.matrix.free_dimension
                    - record.matrix.closure_dimension
                    + index
                    * (
                        record.matrix.closure_dimension // len(record.matrix.closure)
                    ) : record.matrix.free_dimension
                    - record.matrix.closure_dimension
                    + (index + 1)
                    * (record.matrix.closure_dimension // len(record.matrix.closure))
                ].tolist(),
                "reported": False,
            }
            for index, term in enumerate(record.matrix.closure)
        ],
        "score": record.score,
        "penalty": record.penalty,
        "nll": record.fit.nll,
        "projected_kkt": record.fit.projected_kkt,
    }


def _default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run(
    config: RunConfig, *, run_dir: str | Path | None = None, resume: bool = False
) -> RunReport:
    run_started = time.perf_counter()
    config.validate()
    dataset = Dataset.load(config.dataset)
    if run_dir is None:
        run_dir = Path(config.run_root) / (config.run_id or _default_run_id())
    run_dir = Path(run_dir)
    existed = run_dir.exists() and any(run_dir.iterdir())
    if existed and not resume:
        raise FileExistsError(
            f"refusing to overwrite existing run directory: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logger = logging.getLogger(f"crbstpp.{run_dir.name}")
    for existing_handler in logger.handlers:
        existing_handler.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        log_path, mode="a" if resume else "w", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    manifest_path = run_dir / "manifest.json"
    if existed:
        if not manifest_path.is_file():
            raise ValueError("existing run directory has no CRBS manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "crbstpp.run.v1":
            raise ValueError("legacy or unsupported run schema")
        if manifest.get("config_digest") != config.digest:
            raise ValueError("run config digest mismatch")
        if manifest.get("dataset_digest") != dataset.digest:
            raise ValueError("run dataset digest mismatch")
    else:
        atomic_text(
            run_dir / "config.yaml", yaml.safe_dump(config.to_dict(), sort_keys=True)
        )
        manifest = {
            "schema": "crbstpp.run.v1",
            "algorithm": "CRBS-TPP",
            "config_digest": config.digest,
            "dataset_digest": dataset.digest,
            "dataset": str(dataset.root),
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_json(manifest_path, manifest)
    completed_path = run_dir / "result.json"
    if resume and completed_path.is_file():
        payload = json.loads(completed_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != RESULT_SCHEMA
            or payload.get("config_digest") != config.digest
            or payload.get("dataset_digest") != dataset.digest
        ):
            raise ValueError("completed result does not match this run")
        handler.close()
        logger.removeHandler(handler)
        return RunReport(
            schema=RESULT_SCHEMA,
            algorithm="CRBS-TPP",
            config_digest=config.digest,
            dataset_digest=dataset.digest,
            support_count=int(payload.get("search", {}).get("family_size", 0)),
            certified_count=int(
                payload.get("certification", {}).get("certified_count", 0)
            ),
            result=payload,
        )
    fit_codes, cert_codes, test_codes = dataset.split(
        config.split_fractions, config.split_seed
    )
    fit_context = Context.make(dataset, fit_codes)
    cert_context = Context.make(dataset, cert_codes)
    test_context = Context.make(dataset, test_codes)
    optimizer = SupportOptimizer(fit_context, config)
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint: dict[str, Any] | None = None
    if resume and checkpoint_path.is_file():
        checkpoint = load_checkpoint(
            checkpoint_path,
            config_digest=config.digest,
            dataset_digest=dataset.digest,
        )
        logger.info("resuming stage=%s", checkpoint.get("stage"))
    if checkpoint and checkpoint.get("stage") in {
        "search_complete",
        "certification_complete",
    }:
        resumed_supports = [
            _support_from_payload(payload) for payload in checkpoint["family"]
        ]
        family = tuple(
            optimizer.fit_many(resumed_supports, optimizer.records[Support(())])
        )
        search_result = None
        search_seconds = 0.0
    else:
        logger.info("starting support search")
        search_started = time.perf_counter()
        search_result = optimizer.search()
        search_seconds = time.perf_counter() - search_started
        family = search_result.family
        atomic_json(
            checkpoint_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "stage": "search_complete",
                "config_digest": config.digest,
                "dataset_digest": dataset.digest,
                "family": [_support_payload(record.support) for record in family],
                "diagnostics": asdict(search_result.diagnostics),
            },
        )
    logger.info("certifying family_size=%d", len(family))
    certification_started = time.perf_counter()
    certification = certify_family(optimizer, cert_context, family, config)
    certification_seconds = time.perf_counter() - certification_started
    atomic_json(
        checkpoint_path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "certification_complete",
            "config_digest": config.digest,
            "dataset_digest": dataset.digest,
            "family": [_support_payload(record.support) for record in family],
            "certificates": [
                model.certificate.to_dict() for model in certification.models
            ],
        },
    )
    combined_codes = np.sort(np.concatenate([fit_codes, cert_codes])).astype(np.int32)
    combined_context = Context.make(dataset, combined_codes)
    certified_supports = tuple(
        model.record.support for model in certification.certified
    )
    optimizer.release_search_caches()
    ensemble_started = time.perf_counter()
    ensemble = fit_ensemble(combined_context, test_context, certified_supports, config)
    ensemble_seconds = time.perf_counter() - ensemble_started
    search_payload: dict[str, object]
    if search_result is None:
        search_payload = {
            "resumed": True,
            "family_size": len(family),
            "diagnostics": asdict(optimizer.diagnostics),
        }
    else:
        search_payload = {
            "resumed": False,
            "family_size": len(search_result.family),
            "terminal_count": len(search_result.terminals),
            "positive_atom_count": len(search_result.positive_atoms),
            "paths": search_result.paths,
            "diagnostics": asdict(search_result.diagnostics),
        }
    result_payload = {
        "schema": RESULT_SCHEMA,
        "algorithm": "CRBS-TPP",
        "claim": "certified predictive early-warning rules; non-causal",
        "config_digest": config.digest,
        "dataset_digest": dataset.digest,
        "split_sizes": {
            "fit": len(fit_codes),
            "cert": len(cert_codes),
            "test": len(test_codes),
        },
        "search": search_payload,
        "family": [
            _record_payload(record, dataset.predicate_names) for record in family
        ],
        "certification": {
            "family_size": certification.family_size,
            "certified_count": len(certification.certified),
            "all": [
                {
                    "certificate": model.certificate.to_dict(),
                    "diagnostics": model.diagnostics,
                }
                for model in certification.models
            ],
        },
        "ensemble": ensemble.to_dict(),
    }
    atomic_json(
        run_dir / "timing.json",
        {
            "search": search_seconds,
            "certification": certification_seconds,
            "ensemble": ensemble_seconds,
            "total": time.perf_counter() - run_started,
        },
    )
    atomic_json(run_dir / "result.json", result_payload)
    checkpoint_path.unlink(missing_ok=True)
    logger.info("completed certified=%d", len(certification.certified))
    canonical_result = json.loads(json.dumps(result_payload))
    handler.close()
    logger.removeHandler(handler)
    return RunReport(
        schema=RESULT_SCHEMA,
        algorithm="CRBS-TPP",
        config_digest=config.digest,
        dataset_digest=dataset.digest,
        support_count=len(family),
        certified_count=len(certification.certified),
        result=canonical_result,
    )


def inspect_run(run_dir: str | Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "checkpoint.json"
    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "complete": result_path.is_file(),
        "result": json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else None,
        "checkpoint": json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else None,
    }
