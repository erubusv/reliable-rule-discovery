from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .certification import CertificationResult, certify_family
from .checkpoint import CHECKPOINT_SCHEMA, RESULT_SCHEMA, atomic_json, load_checkpoint
from .config import RunConfig
from .data import Dataset
from .ensemble import fit_ensemble
from .objective import SupportRecord
from .report import RunReport
from .response import Context
from .rules import RuleIdentity, Support
from .search import SearchResult, SupportOptimizer, support_key


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


def _record_payload(record: SupportRecord, predicate_names: tuple[str, ...]) -> dict[str, object]:
    return {
        "key": support_key(record.support),
        "rules": [
            {
                **_rule_payload(rule),
                "antecedent_names": [predicate_names[index] for index in rule.antecedent],
                "kernel": record.fit.coefficients[block].tolist(),
            }
            for rule, block in zip(record.support.rules, record.matrix.rule_slices, strict=True)
        ],
        "closure": [
            {"antecedent": list(term.antecedent), "window": term.window}
            for term in record.matrix.closure
        ],
        "score": record.score,
        "penalty": record.penalty,
        "nll": record.fit.nll,
        "projected_kkt": record.fit.projected_kkt,
    }


def _default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run(config: RunConfig, *, run_dir: str | Path | None = None, resume: bool = False) -> RunReport:
    config.validate()
    dataset = Dataset.load(config.dataset)
    if run_dir is None:
        run_dir = Path(config.run_root) / (config.run_id or _default_run_id())
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logger = logging.getLogger(f"crbstpp.{run_dir.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema": "crbstpp.run.v1",
        "algorithm": "CRBS-TPP",
        "config_digest": config.digest,
        "dataset_digest": dataset.digest,
        "dataset": str(dataset.root),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(run_dir / "manifest.json", manifest)
    fit_codes, cert_codes, test_codes = dataset.split(config.split_fractions, config.split_seed)
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
    if checkpoint and checkpoint.get("stage") in {"search_complete", "certification_complete"}:
        family = tuple(
            optimizer.fit(_support_from_payload(payload))
            for payload in checkpoint["family"]
        )
        search_result = None
    else:
        logger.info("starting support search")
        search_result = optimizer.search()
        family = search_result.family
        atomic_json(checkpoint_path, {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "search_complete",
            "config_digest": config.digest,
            "dataset_digest": dataset.digest,
            "family": [_support_payload(record.support) for record in family],
            "diagnostics": asdict(search_result.diagnostics),
        })
    logger.info("certifying family_size=%d", len(family))
    certification = certify_family(optimizer, cert_context, family, config)
    atomic_json(checkpoint_path, {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "certification_complete",
        "config_digest": config.digest,
        "dataset_digest": dataset.digest,
        "family": [_support_payload(record.support) for record in family],
        "certificates": [model.certificate.to_dict() for model in certification.models],
    })
    combined_codes = __import__("numpy").sort(
        __import__("numpy").concatenate([fit_codes, cert_codes])
    ).astype("int32")
    combined_context = Context.make(dataset, combined_codes)
    certified_supports = tuple(model.record.support for model in certification.certified)
    ensemble = fit_ensemble(combined_context, test_context, certified_supports, config)
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
            "fit": len(fit_codes), "cert": len(cert_codes), "test": len(test_codes)
        },
        "search": search_payload,
        "family": [_record_payload(record, dataset.predicate_names) for record in family],
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
    atomic_json(run_dir / "result.json", result_payload)
    checkpoint_path.unlink(missing_ok=True)
    logger.info("completed certified=%d", len(certification.certified))
    return RunReport(
        schema=RESULT_SCHEMA,
        algorithm="CRBS-TPP",
        config_digest=config.digest,
        dataset_digest=dataset.digest,
        support_count=len(family),
        certified_count=len(certification.certified),
        result=result_payload,
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
        "result": json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None,
        "checkpoint": json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.is_file() else None,
    }

