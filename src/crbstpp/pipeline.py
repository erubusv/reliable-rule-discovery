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

from .certification import (
    CertificationResult,
    CertifiedModel,
    certify_family,
)
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
from .report import Certificate, RunReport
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


def _certificate_from_payload(payload: dict[str, object]) -> Certificate:
    return Certificate(
        support_key=str(payload["support_key"]),
        f0=bool(payload["f0"]),
        f1_pvalue=(
            None if payload.get("f1_pvalue") is None else float(payload["f1_pvalue"])
        ),
        f2_pvalues=tuple(float(value) for value in payload.get("f2_pvalues", [])),
        f3=bool(payload["f3"]),
        family_pvalue=(
            None
            if payload.get("family_pvalue") is None
            else float(payload["family_pvalue"])
        ),
        holm_adjusted_pvalue=(
            None
            if payload.get("holm_adjusted_pvalue") is None
            else float(payload["holm_adjusted_pvalue"])
        ),
        certified=bool(payload["certified"]),
        reasons=tuple(str(value) for value in payload.get("reasons", [])),
    )


def _record_payload(
    record: SupportRecord,
    predicate_names: tuple[str, ...],
    basis: np.ndarray,
) -> dict[str, object]:
    def direction(profile: np.ndarray) -> str:
        tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(profile), initial=0.0)))
        if np.all(profile >= -tolerance) and np.any(profile > tolerance):
            return "excitation"
        if np.all(profile <= tolerance) and np.any(profile < -tolerance):
            return "inhibition"
        if np.all(np.abs(profile) <= tolerance):
            return "zero"
        return "mixed"

    rules = []
    for rule, block in zip(
        record.support.rules, record.matrix.rule_slices, strict=True
    ):
        coefficients = record.fit.coefficients[block]
        profile = float(rule.sign) * (coefficients @ basis)
        rules.append(
            {
                **_rule_payload(rule),
                "antecedent_names": [
                    predicate_names[index] for index in rule.antecedent
                ],
                "kernel": coefficients.tolist(),
                "lag_profile": profile.tolist(),
                "direction": direction(profile),
                "role": "reported_rule",
                "reported": True,
            }
        )
    closure = []
    closure_width = (
        record.matrix.closure_dimension // len(record.matrix.closure)
        if record.matrix.closure
        else 0
    )
    closure_left = record.matrix.free_dimension - record.matrix.closure_dimension
    for index, term in enumerate(record.matrix.closure):
        coefficients = record.fit.coefficients[
            closure_left + index * closure_width : closure_left
            + (index + 1) * closure_width
        ]
        profile = coefficients @ basis
        closure.append(
            {
                "antecedent": list(term.antecedent),
                "antecedent_names": [
                    predicate_names[item] for item in term.antecedent
                ],
                "window": term.window,
                "kernel": coefficients.tolist(),
                "lag_profile": profile.tolist(),
                "direction": direction(profile),
                "role": "hierarchy_nuisance",
                "reported": False,
                "certified_separately": False,
            }
        )
    return {
        "key": support_key(record.support),
        "rules": rules,
        "closure": closure,
        "interpretation": (
            "Only rules are certified identities. Closure kernels are shared "
            "hierarchy nuisance and are shown solely to make a higher-order "
            "rule's lower-order decomposition explicit."
        ),
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
        resumed_paths: tuple[
            tuple[Support, Support, dict[str, object]], ...
        ] = ()
        resumed_active: tuple[
            Support, Support, tuple[dict[str, object], ...]
        ] | None = None
        if checkpoint and checkpoint.get("stage") == "search_progress":
            resumed_paths = tuple(
                (
                    _support_from_payload(item["start"]),
                    _support_from_payload(item["terminal"]),
                    item["path"],
                )
                for item in checkpoint.get("completed_paths", [])
            )
            active = checkpoint.get("active_path")
            if isinstance(active, dict):
                resumed_active = (
                    _support_from_payload(active["start"]),
                    _support_from_payload(active["current"]),
                    tuple(active.get("moves", [])),
                )

        def save_search_progress(
            completed: tuple[
                tuple[Support, Support, dict[str, object]], ...
            ],
            active: tuple[
                Support, Support, tuple[dict[str, object], ...]
            ]
            | None,
        ) -> None:
            atomic_json(
                checkpoint_path,
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "stage": "search_progress",
                    "config_digest": config.digest,
                    "dataset_digest": dataset.digest,
                    "completed_paths": [
                        {
                            "start": _support_payload(start),
                            "terminal": _support_payload(terminal),
                            "path": path,
                        }
                        for start, terminal, path in completed
                    ],
                    "active_path": (
                        {
                            "start": _support_payload(active[0]),
                            "current": _support_payload(active[1]),
                            "moves": list(active[2]),
                        }
                        if active is not None
                        else None
                    ),
                },
            )

        # Persist a valid restart boundary before the expensive W/sign profile.
        # The profile itself is deterministic and will be replayed on resume,
        # while accepted support paths recorded later remain reusable.
        if checkpoint is None:
            save_search_progress((), None)
        search_result = optimizer.search(
            completed_paths=resumed_paths,
            active_path=resumed_active,
            progress_callback=save_search_progress,
        )
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
    optimizer.prepare_certification(family)
    if checkpoint and checkpoint.get("stage") == "certification_complete":
        by_key = {support_key(record.support): record for record in family}
        restored_models = []
        for item in checkpoint.get("certification", []):
            certificate = _certificate_from_payload(item["certificate"])
            record = by_key.get(certificate.support_key)
            if record is None:
                raise ValueError("certification checkpoint references unknown support")
            restored_models.append(
                CertifiedModel(record, certificate, item.get("diagnostics", {}))
            )
        certification = CertificationResult(
            tuple(restored_models),
            tuple(model for model in restored_models if model.certificate.certified),
            len(restored_models),
        )
        certification_seconds = 0.0
    else:
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
                "certification": [
                    {
                        "certificate": model.certificate.to_dict(),
                        "diagnostics": model.diagnostics,
                    }
                    for model in certification.models
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
        "claim": (
            "cohort-mixture distributionally robust predictive early-warning "
            "rules; non-causal"
        ),
        "config_digest": config.digest,
        "dataset_digest": dataset.digest,
        "split_sizes": {
            "fit": len(fit_codes),
            "cert": len(cert_codes),
            "test": len(test_codes),
        },
        "search": search_payload,
        "family": [
            _record_payload(record, dataset.predicate_names, optimizer.engine.basis)
            for record in family
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
    manifest_path = run_dir / "manifest.json"
    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "checkpoint.json"
    failure_path = run_dir / "failure.json"
    return {
        "run_dir": str(run_dir),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None,
        "complete": result_path.is_file(),
        "result": json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else None,
        "checkpoint": json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else None,
        "failure": json.loads(failure_path.read_text(encoding="utf-8"))
        if failure_path.is_file()
        else None,
    }
