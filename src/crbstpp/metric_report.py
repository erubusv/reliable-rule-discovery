from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


MODEL_ORDER = (
    "logistic",
    "xgboost",
    "hawkes",
    "rmtpp",
    "nhp",
    "thp",
    "attnhp",
    "branch_price",
    "neurosymbolic_tpp",
    "ours",
)

DISPLAY_NAMES = {
    "logistic": "Logistic regression",
    "xgboost": "XGBoost",
    "hawkes": "Exponential Hawkes",
    "rmtpp": "RMTPP",
    "nhp": "Neural Hawkes",
    "thp": "Transformer Hawkes",
    "attnhp": "Attentive Neural Hawkes",
    "branch_price": "Branch-and-price temporal rules",
    "neurosymbolic_tpp": "Neuro-Symbolic TPP",
    "ours": "Ours",
}

PREDICTION_FIELDS = (
    "target_nll_per_entity",
    "target_nll_total",
    "joint_nll",
    "joint_nll_total",
    "binary_nll_per_landmark",
    "binary_nll_total",
    "brier",
    "event_type_accuracy",
    "time_rmse",
    "elapsed_seconds",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _finite(value: object) -> float | None:
    if value is None:
        return None
    output = float(value)
    return output if math.isfinite(output) else None


def _pending_record(model: str, root: Path) -> dict[str, Any]:
    failure = root / "failure.json"
    return {
        "model": model,
        "display_name": DISPLAY_NAMES[model],
        "status": "failed" if failure.is_file() else "pending",
        "source": str(root / "result.json"),
        **{field: None for field in PREDICTION_FIELDS},
    }


def _baseline_record(model: str, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = root / "result.json"
    if not result_path.is_file():
        return _pending_record(model, root), {}
    payload = _read_json(result_path)
    details = dict(payload.get("details", {}))
    record = _pending_record(model, root)
    record.update(
        {
            "status": "complete",
            "elapsed_seconds": _finite(payload.get("elapsed_seconds")),
            "dataset_digest": payload.get("dataset_digest"),
            "seed": payload.get("seed"),
        }
    )

    test = dict(details.get("test", {}))
    if model in {"logistic", "xgboost", "branch_price", "neurosymbolic_tpp"}:
        record["binary_nll_per_landmark"] = _finite(test.get("binary_nll"))
        record["brier"] = _finite(test.get("brier"))
        record["landmarks"] = test.get("n")
        record["landmark_targets"] = test.get("targets")
        if record["binary_nll_per_landmark"] is not None and test.get("n") is not None:
            record["binary_nll_total"] = (
                record["binary_nll_per_landmark"] * int(test["n"])
            )
    if model in {"hawkes", "baseline_tpp"}:
        record["target_nll_per_entity"] = _finite(
            details.get("test_nll_per_entity")
        )
        landmark_path = root / "hawkes_landmark_metrics.json"
        if landmark_path.is_file():
            landmark = dict(_read_json(landmark_path).get("test", {}))
            record["binary_nll_per_landmark"] = _finite(
                landmark.get("binary_nll")
            )
            record["brier"] = _finite(landmark.get("brier"))
            record["landmarks"] = landmark.get("n")
            record["landmark_targets"] = landmark.get("targets")
    if model in {"branch_price", "neurosymbolic_tpp"}:
        point_process = dict(details.get("point_process", {}))
        record["target_nll_per_entity"] = _finite(
            point_process.get("test_nll_per_entity")
        )
    if model in {"rmtpp", "nhp", "thp", "attnhp"}:
        loglike = _finite(test.get("loglike"))
        record["joint_nll"] = None if loglike is None else -loglike
        record["event_type_accuracy"] = _finite(test.get("acc"))
        record["time_rmse"] = _finite(test.get("rmse"))
        record["joint_event_count"] = test.get("num_events")
        if record["joint_nll"] is not None and test.get("num_events") is not None:
            record["joint_nll_total"] = record["joint_nll"] * int(
                test["num_events"]
            )
        target_path = root / "target_metrics.json"
        if target_path.is_file():
            target_metrics = _read_json(target_path)
            target = dict(target_metrics.get("target", {}))
            landmark = dict(target_metrics.get("test", {}))
            record["target_nll_per_entity"] = _finite(
                target.get("target_nll_per_entity")
            )
            record["target_nll_total"] = _finite(target.get("target_nll_total"))
            record["binary_nll_per_landmark"] = _finite(
                landmark.get("binary_nll")
            )
            record["brier"] = _finite(landmark.get("brier"))
            record["landmarks"] = landmark.get("n")
            record["landmark_targets"] = landmark.get("targets")
            if record["binary_nll_per_landmark"] is not None and landmark.get("n"):
                record["binary_nll_total"] = (
                    record["binary_nll_per_landmark"] * int(landmark["n"])
                )

    rules = list(details.get("rules", []))
    rule_metrics: dict[str, Any] = {}
    if rules:
        order = {"singleton": 0, "pair": 0, "triplet": 0}
        signs = {"excitation": 0, "inhibition": 0, "zero": 0}
        predicates: set[int] = set()
        for rule in rules:
            antecedent = tuple(int(value) for value in rule.get("antecedent", []))
            if 1 <= len(antecedent) <= 3:
                order[("singleton", "pair", "triplet")[len(antecedent) - 1]] += 1
            predicates.update(antecedent)
            coefficient = float(rule.get("coefficient", 0.0))
            signs[
                "excitation" if coefficient > 0 else "inhibition" if coefficient < 0 else "zero"
            ] += 1
        rule_metrics = {
            "reported_rule_count": len(rules),
            "order_counts": order,
            "direction_counts": signs,
            "unique_predicate_count": len(predicates),
        }
    return record, rule_metrics


def _run_elapsed_seconds(run_root: Path, result_path: Path) -> float | None:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    created = _read_json(manifest_path).get("created_at_utc")
    if not created:
        return None
    try:
        start = datetime.fromisoformat(str(created)).timestamp()
    except ValueError:
        return None
    return max(0.0, result_path.stat().st_mtime - start)


def _ours_record(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = root / "result.json"
    if not result_path.is_file():
        return _pending_record("ours", root), {}
    payload = _read_json(result_path)
    ensemble = dict(payload.get("ensemble", {}))
    stable_path = root / "stable_stacking_metrics.json"
    if stable_path.is_file():
        stable = _read_json(stable_path)
        if stable.get("test_nll_total") is not None:
            ensemble["test_nll"] = stable["test_nll_total"]
        if stable.get("baseline_test_nll") is not None:
            ensemble["baseline_test_nll"] = stable["baseline_test_nll"]
    certification = dict(payload.get("certification", {}))
    search = dict(payload.get("search", {}))
    split_sizes = dict(payload.get("split_sizes", {}))
    test_entities = int(split_sizes.get("test", 0))
    test_nll = _finite(ensemble.get("test_nll"))
    baseline_nll = _finite(ensemble.get("baseline_test_nll"))
    target_nll = (
        None if test_nll is None or test_entities <= 0 else test_nll / test_entities
    )
    baseline_target_nll = (
        None
        if baseline_nll is None or test_entities <= 0
        else baseline_nll / test_entities
    )
    gain = (
        None
        if test_nll is None or baseline_nll is None
        else baseline_nll - test_nll
    )
    record = _pending_record("ours", root)
    record.update(
        {
            "status": "complete",
            "source": str(result_path),
            "dataset_digest": payload.get("dataset_digest"),
            "target_nll_per_entity": target_nll,
            "target_nll_total": test_nll,
            "baseline_target_nll_per_entity": baseline_target_nll,
            "baseline_target_nll_total": baseline_nll,
            "target_nll_gain_total": gain,
            "target_nll_gain_per_entity": (
                None if gain is None or test_entities <= 0 else gain / test_entities
            ),
            "relative_target_nll_reduction": (
                None
                if gain is None or baseline_nll is None or baseline_nll == 0.0
                else gain / baseline_nll
            ),
            "deviance_gain": None if gain is None else 2.0 * gain,
            "test_entities": test_entities,
            "elapsed_seconds": _run_elapsed_seconds(root, result_path),
        }
    )
    integrated_path = root / "integrated_landmark_metrics.json"
    landmark_path = (
        integrated_path if integrated_path.is_file() else root / "landmark_metrics.json"
    )
    if landmark_path.is_file():
        landmark = _read_json(landmark_path)
        landmark_test = dict(landmark.get("test", {}))
        record["binary_nll_per_landmark"] = _finite(
            landmark_test.get("binary_nll")
        )
        record["brier"] = _finite(landmark_test.get("brier"))
        record["landmarks"] = landmark_test.get("n")
        record["landmark_targets"] = landmark_test.get("targets")
        if record["binary_nll_per_landmark"] is not None and record["landmarks"]:
            record["binary_nll_total"] = (
                record["binary_nll_per_landmark"] * int(record["landmarks"])
            )

    selected_keys = set(str(value) for value in certification.get("selected_supports", []))
    selected_families = [
        item for item in payload.get("family", []) if str(item.get("key")) in selected_keys
    ]
    order = {"singleton": 0, "pair": 0, "triplet": 0}
    relation = {"atomic": 0, "unordered": 0, "ordered": 0}
    directions = {"excitation": 0, "inhibition": 0}
    predicates: set[int] = set()
    unique_rules: set[str] = set()
    rule_set_sizes: list[int] = []
    for family in selected_families:
        rules = list(family.get("rules", []))
        rule_set_sizes.append(len(rules))
        for rule in rules:
            antecedent = tuple(int(value) for value in rule.get("antecedent", []))
            identity = json.dumps(
                {
                    "antecedent": antecedent,
                    "window": rule.get("window"),
                    "sign": rule.get("sign"),
                    "relation": rule.get("relation"),
                    "history_marks": rule.get("history_marks", []),
                },
                sort_keys=True,
            )
            unique_rules.add(identity)
            predicates.update(antecedent)
            if 1 <= len(antecedent) <= 3:
                order[("singleton", "pair", "triplet")[len(antecedent) - 1]] += 1
            relation[str(rule.get("relation", "unordered"))] += 1
            direction = str(rule.get("direction", rule.get("rule_direction", "")))
            if direction in directions:
                directions[direction] += 1

    gate_counts = {"f0": 0, "f1": 0, "f2": 0, "f3": 0, "romano_wolf": 0}
    for item in certification.get("all", []):
        certificate = dict(item.get("certificate", {}))
        reasons = tuple(str(value) for value in certificate.get("reasons", []))
        gate_counts["f0"] += int(bool(certificate.get("f0")))
        gate_counts["f1"] += int(not any(value.startswith("f1_") for value in reasons))
        gate_counts["f2"] += int(not any(value.startswith("f2_") for value in reasons))
        gate_counts["f3"] += int(bool(certificate.get("f3")))
        adjusted = _finite(certificate.get("family_adjusted_pvalue"))
        gate_counts["romano_wolf"] += int(adjusted is not None and adjusted <= 0.05)

    diagnostics = dict(search.get("diagnostics", {}))
    rule_metrics = {
        "candidate_rule_set_count": certification.get("family_size"),
        "certified_rule_set_count": certification.get("certified_count"),
        "selected_rule_set_count": certification.get("selected_count"),
        "selected_rule_set_sizes": rule_set_sizes,
        "unique_certified_rule_count": len(unique_rules),
        "order_counts": order,
        "relation_counts": relation,
        "direction_counts": directions,
        "unique_predicate_count": len(predicates),
        "gate_pass_counts": gate_counts,
        "active_stacking_rule_count": ensemble.get("active_rule_effect_count"),
        "active_stacking_rule_set_count": ensemble.get("active_support_count"),
        "search_terminal_count": search.get("terminal_count"),
        "search_positive_atom_count": search.get("positive_atom_count"),
        "search_exact_fit_count": diagnostics.get("exact_fits"),
        "search_bound_candidate_count": diagnostics.get("implicit_gpu_candidates"),
    }
    return record, rule_metrics


def _latex_number(value: object, digits: int = 4) -> str:
    output = _finite(value)
    return "--" if output is None else f"{output:.{digits}f}"


def _latex_rows(datasets: list[dict[str, Any]]) -> str:
    by_key = {str(item["key"]): item for item in datasets}
    ordered = [by_key.get("aave"), by_key.get("wselob")]
    lines = ["% Generated by `crbstpp metrics`; do not edit by hand."]
    for model in MODEL_ORDER:
        cells: list[str] = []
        for dataset in ordered:
            record = {} if dataset is None else dataset["models"].get(model, {})
            cells.extend(
                (
                    _latex_number(record.get("target_nll_per_entity")),
                    _latex_number(record.get("binary_nll_per_landmark")),
                    _latex_number(record.get("brier"), digits=6),
                )
            )
        lines.append(f"{DISPLAY_NAMES[model]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def collect_metric_report(spec_path: str | Path) -> dict[str, Any]:
    spec_path = Path(spec_path)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("datasets"), list):
        raise ValueError("metric report spec requires a datasets list")
    datasets: list[dict[str, Any]] = []
    for item in spec["datasets"]:
        key = str(item["key"])
        baseline_root = Path(item["baseline_seed_dir"])
        models: dict[str, Any] = {}
        rule_metrics: dict[str, Any] = {}
        for model in MODEL_ORDER[:-1]:
            record, rules = _baseline_record(model, baseline_root / model)
            models[model] = record
            if rules:
                rule_metrics[model] = rules
        ours, ours_rules = _ours_record(Path(item["ours_run_dir"]))
        models["ours"] = ours
        expected_digest = ours.get("dataset_digest")
        for model, record in models.items():
            observed_digest = record.get("dataset_digest")
            if (
                model != "ours"
                and record.get("status") == "complete"
                and expected_digest is not None
                and observed_digest != expected_digest
            ):
                raise ValueError(
                    f"{key}/{model} uses dataset digest {observed_digest}, "
                    f"not the rule-model digest {expected_digest}"
                )
        test_entities = ours.get("test_entities")
        if test_entities:
            for record in models.values():
                if (
                    record.get("target_nll_per_entity") is not None
                    and record.get("target_nll_total") is None
                ):
                    record["target_nll_total"] = (
                        record["target_nll_per_entity"] * int(test_entities)
                    )
        if ours_rules:
            rule_metrics["ours"] = ours_rules
        datasets.append(
            {
                "key": key,
                "label": str(item.get("label", key)),
                "models": models,
                "rule_metrics": rule_metrics,
            }
        )

    payload = {
        "schema": "crbstpp.metric-report.v1",
        "metric_contract": {
            "target_nll_per_entity": "target-process NLL divided by test entities",
            "joint_nll": "negative of the official EasyTPP joint loglike; not target-only NLL",
            "joint_nll_total": "joint_nll multiplied by the official EasyTPP event count",
            "binary_nll_per_landmark": "fixed-horizon binary log loss; not event-process NLL",
            "binary_nll_total": "binary_nll_per_landmark multiplied by landmark count",
            "brier": "fixed-horizon landmark Brier score",
            "event_type_accuracy": "official EasyTPP next-event type accuracy",
            "time_rmse": "official EasyTPP next-event time RMSE",
            "relative_target_nll_reduction": "(baseline target NLL - model target NLL) / baseline target NLL",
            "deviance_gain": "2 * (baseline target NLL - model target NLL)",
        },
        "datasets": datasets,
    }

    output_dir = Path(spec.get("output_dir", "runs/metrics"))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, json_path)

    csv_path = output_dir / "metrics.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    fieldnames = [
        "dataset",
        "model",
        "status",
        *PREDICTION_FIELDS,
        "binary_nll_per_landmark",
        "target_nll_gain_per_entity",
        "relative_target_nll_reduction",
        "deviance_gain",
    ]
    # Preserve field order while removing the repeated binary-NLL entry.
    fieldnames = list(dict.fromkeys(fieldnames))
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for dataset in datasets:
            for model in MODEL_ORDER:
                writer.writerow(
                    {
                        "dataset": dataset["key"],
                        "model": model,
                        **dataset["models"][model],
                    }
                )
    os.replace(temporary_csv, csv_path)

    tex_path = Path(spec.get("tex_output", output_dir / "prediction_rows.tex"))
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_tex = tex_path.with_suffix(tex_path.suffix + ".tmp")
    temporary_tex.write_text(_latex_rows(datasets), encoding="utf-8")
    os.replace(temporary_tex, tex_path)
    return payload
