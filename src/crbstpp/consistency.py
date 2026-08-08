from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .checkpoint import atomic_json
from .rules import RuleIdentity, Support
from .search import support_from_key, support_key


def _quantile_label_map(result: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    payload = result.get("search", {}).get("frozen_window_quantile_labels", {})
    return {
        str(antecedent): {
            str(window): [str(label) for label in labels]
            for window, labels in windows.items()
        }
        for antecedent, windows in payload.items()
    }


def _label_for_rule(
    rule: RuleIdentity,
    labels: dict[str, dict[str, list[str]]],
) -> str:
    if rule.window == 0:
        return "W0"
    antecedent = ",".join(map(str, rule.antecedent))
    quantiles = labels.get(antecedent, {}).get(str(rule.window), [])
    return "+".join(quantiles) if quantiles else f"W{rule.window}"


def _labelled_rule_key(
    rule: RuleIdentity,
    labels: dict[str, dict[str, list[str]]],
) -> str:
    antecedent = ",".join(map(str, rule.antecedent))
    direction = "exc" if rule.sign > 0 else "inh"
    kernel = "|K1" if rule.kernel_rank == 1 else ""
    return f"{antecedent}|{_label_for_rule(rule, labels)}|{direction}{kernel}"


def _labelled_support_key(
    support: Support,
    labels: dict[str, dict[str, list[str]]],
) -> str:
    if not support.rules:
        return "empty"
    return ";".join(_labelled_rule_key(rule, labels) for rule in support.rules)


def _structural_support_key(support: Support) -> str:
    if not support.rules:
        return "empty"
    return ";".join(
        f"{','.join(map(str, rule.antecedent))}|{'exc' if rule.sign > 0 else 'inh'}"
        for rule in support.rules
    )


def _reported_total_support_key(
    support: Support,
    diagnostics: dict[str, Any],
    labels: dict[str, dict[str, list[str]]],
    *,
    structural: bool = False,
) -> str:
    """Serialize the externally certified direction, not the internal cone."""

    if not support.rules:
        return "empty"
    rule_diagnostics = diagnostics.get("rules", [])
    if not isinstance(rule_diagnostics, list):
        rule_diagnostics = []
    output = []
    for index, rule in enumerate(support.rules):
        item = (
            rule_diagnostics[index]
            if index < len(rule_diagnostics)
            and isinstance(rule_diagnostics[index], dict)
            else {}
        )
        sign = int(item.get("reported_total_sign", 0))
        direction = "exc" if sign > 0 else "inh" if sign < 0 else "unidentified"
        antecedent = ",".join(map(str, rule.antecedent))
        if structural:
            output.append(f"{antecedent}|{direction}")
        else:
            output.append(f"{antecedent}|{_label_for_rule(rule, labels)}|{direction}")
    return ";".join(output)


def _payload_support(payload: list[dict[str, Any]]) -> Support:
    return Support.of(
        RuleIdentity(
            tuple(int(value) for value in rule["antecedent"]),
            int(rule["window"]),
            int(rule["sign"]),
            int(rule.get("kernel_rank", 0)),
            str(rule.get("relation", "auto")),
            bool(rule.get("hierarchical", False)),
            tuple(
                (int(mark[0]), int(mark[1]))
                for mark in rule.get("history_marks", [])
            ),
            bool(rule.get("support_additive", False)),
        )
        for rule in payload
    )


def _mean_pairwise_jaccard(families: list[set[str]]) -> float:
    values = []
    for left, right in itertools.combinations(families, 2):
        union = left | right
        values.append(1.0 if not union else len(left & right) / len(union))
    return float(np.mean(values)) if values else 1.0


def _frequency(families: dict[int, set[str]]) -> list[dict[str, Any]]:
    universe = sorted(set().union(*families.values())) if families else []
    return [
        {
            "identity": identity,
            "count": sum(identity in family for family in families.values()),
            "fraction": (
                sum(identity in family for family in families.values()) / len(families)
            ),
            "seeds": [
                seed for seed, family in sorted(families.items()) if identity in family
            ],
        }
        for identity in universe
    ]


def _kernel_summary(
    profiles: dict[str, list[tuple[int, np.ndarray]]],
) -> list[dict[str, Any]]:
    output = []
    for identity, observations in sorted(profiles.items()):
        arrays = [values for _, values in observations]
        cosine = []
        for left, right in itertools.combinations(arrays, 2):
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator > 0.0:
                cosine.append(float(np.dot(left, right) / denominator))
        output.append(
            {
                "rule_identity": identity,
                "appearances": len(observations),
                "seeds": sorted({seed for seed, _ in observations}),
                "mean_pairwise_cosine": (float(np.mean(cosine)) if cosine else None),
                "peak_abs_lags": [
                    int(np.argmax(np.abs(values)) + 1) for values in arrays
                ],
                "signed_areas": [float(np.sum(values)) for values in arrays],
                "lag_profiles": [
                    {"seed": seed, "values": values.tolist()}
                    for seed, values in observations
                ],
            }
        )
    return output


def compare_runs(
    run_dirs: tuple[Path, ...],
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("consistency comparison requires at least two runs")
    exact_candidates: dict[int, set[str]] = {}
    exact_certified: dict[int, set[str]] = {}
    structural_certified: dict[int, set[str]] = {}
    active_ensemble: dict[int, set[str]] = {}
    reported_candidates: dict[int, set[str]] = {}
    reported_certified: dict[int, set[str]] = {}
    reported_structural_certified: dict[int, set[str]] = {}
    reported_active_ensemble: dict[int, set[str]] = {}
    kernel_profiles: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    runs: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
        seed = int(config["split_seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate split seed: {seed}")
        seen_seeds.add(seed)
        labels = _quantile_label_map(result)
        certification = result["certification"]["all"]
        diagnostic_by_support: dict[str, dict[str, Any]] = {}
        for item in certification:
            raw_key = str(item["certificate"]["support_key"])
            diagnostics = item.get("diagnostics", {})
            diagnostic_by_support[raw_key] = diagnostics
            # v1--v14 keys omit the explicit full-kernel tag.  Index both the
            # persisted and canonical forms so stability reports remain
            # comparable across that schema boundary.
            diagnostic_by_support[support_key(support_from_key(raw_key))] = diagnostics
        candidate_supports = [
            support_from_key(item["certificate"]["support_key"])
            for item in certification
        ]
        certified_supports = [
            support_from_key(item["certificate"]["support_key"])
            for item in certification
            if item["certificate"]["certified"]
        ]
        exact_candidates[seed] = {
            _labelled_support_key(support, labels) for support in candidate_supports
        }
        exact_certified[seed] = {
            _labelled_support_key(support, labels) for support in certified_supports
        }
        structural_certified[seed] = {
            _structural_support_key(support) for support in certified_supports
        }
        reported_candidates[seed] = {
            _reported_total_support_key(
                support,
                diagnostic_by_support.get(support_key(support), {}),
                labels,
            )
            for support in candidate_supports
        }
        reported_certified[seed] = {
            _reported_total_support_key(
                support,
                diagnostic_by_support.get(support_key(support), {}),
                labels,
            )
            for support in certified_supports
        }
        reported_structural_certified[seed] = {
            _reported_total_support_key(
                support,
                diagnostic_by_support.get(support_key(support), {}),
                labels,
                structural=True,
            )
            for support in certified_supports
        }

        certified_raw = {
            key
            for support in certified_supports
            for key in (
                support_key(support),
                support_key(support).replace("|KM", ""),
            )
        }
        for record in result.get("family", []):
            if record["key"] not in certified_raw:
                continue
            for rule in record.get("rules", []):
                identity = RuleIdentity(
                    tuple(int(value) for value in rule["antecedent"]),
                    int(rule["window"]),
                    int(rule["sign"]),
                    int(rule.get("kernel_rank", 0)),
                    str(rule.get("relation", "auto")),
                    bool(rule.get("hierarchical", False)),
                    tuple(
                        (int(mark[0]), int(mark[1]))
                        for mark in rule.get("history_marks", [])
                    ),
                    bool(rule.get("support_additive", False)),
                )
                kernel_profiles[_labelled_rule_key(identity, labels)].append(
                    (seed, np.asarray(rule["lag_profile"], dtype=np.float64))
                )

        ensemble = result["ensemble"]
        ensemble_items = []
        active: set[str] = set()
        reported_active: set[str] = set()
        for payload, weight in zip(
            ensemble["supports"], ensemble["weights"], strict=True
        ):
            support = _payload_support(payload)
            labelled = _labelled_support_key(support, labels)
            numeric = support_key(support)
            value = float(weight)
            reported = _reported_total_support_key(
                support,
                diagnostic_by_support.get(numeric, {}),
                labels,
            )
            if value > 1.0e-10:
                active.add(labelled)
                reported_active.add(reported)
            ensemble_items.append(
                {
                    "support_key": numeric,
                    "labelled_support_key": labelled,
                    "structural_support_key": _structural_support_key(support),
                    "reported_total_support_key": reported,
                    "weight": value,
                }
            )
        active_ensemble[seed] = active
        reported_active_ensemble[seed] = reported_active
        baseline_nll = float(ensemble["baseline_test_nll"])
        test_nll = float(ensemble["test_nll"])
        runs.append(
            {
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "candidate_count": len(candidate_supports),
                "certified_count": len(certified_supports),
                "certified_supports": sorted(exact_certified[seed]),
                "reported_total_certified_supports": sorted(reported_certified[seed]),
                "structural_certified_supports": sorted(structural_certified[seed]),
                "ensemble": ensemble_items,
                "baseline_weight": float(ensemble["baseline_weight"]),
                "train_nll": float(ensemble["train_nll"]),
                "test_nll": test_nll,
                "baseline_test_nll": baseline_nll,
                "test_nll_improvement": baseline_nll - test_nll,
                "timing_seconds": {
                    name: float(value) for name, value in timing.items()
                },
            }
        )

    runs.sort(key=lambda item: item["seed"])
    improvements = np.asarray(
        [run["test_nll_improvement"] for run in runs], dtype=np.float64
    )
    test_nll = np.asarray([run["test_nll"] for run in runs], dtype=np.float64)
    payload = {
        "schema": "crbstpp.consistency.v1",
        "run_count": len(runs),
        "seeds": [run["seed"] for run in runs],
        "protocol": {
            "independent_full_pipeline_per_split": True,
            "cross_seed_candidate_sharing": False,
            "selection_frequency_used_as_certificate": False,
            "window_identity": "D_fit quantile label plus realized integer window",
            "reported_direction": (
                "D_fit-frozen total contextual probability contribution; "
                "D_cert-tested without refitting"
            ),
            "interaction_sign_role": "internal additive decomposition only",
            "test_role": "comparison only; never used by search or certification",
        },
        "runs": runs,
        "stability": {
            "exact_candidate_frequency": _frequency(exact_candidates),
            "exact_certified_frequency": _frequency(exact_certified),
            "structural_certified_frequency": _frequency(structural_certified),
            "active_ensemble_frequency": _frequency(active_ensemble),
            "reported_total_candidate_frequency": _frequency(reported_candidates),
            "reported_total_certified_frequency": _frequency(reported_certified),
            "reported_total_structural_certified_frequency": _frequency(
                reported_structural_certified
            ),
            "reported_total_active_ensemble_frequency": _frequency(
                reported_active_ensemble
            ),
            "mean_pairwise_exact_certified_jaccard": _mean_pairwise_jaccard(
                list(exact_certified.values())
            ),
            "mean_pairwise_structural_certified_jaccard": _mean_pairwise_jaccard(
                list(structural_certified.values())
            ),
            "mean_pairwise_reported_total_certified_jaccard": (
                _mean_pairwise_jaccard(list(reported_certified.values()))
            ),
            "mean_pairwise_reported_total_structural_certified_jaccard": (
                _mean_pairwise_jaccard(list(reported_structural_certified.values()))
            ),
            "kernel_effects": _kernel_summary(kernel_profiles),
        },
        "prediction": {
            "test_nll_mean": float(np.mean(test_nll)),
            "test_nll_sample_std": (
                float(np.std(test_nll, ddof=1)) if len(test_nll) > 1 else 0.0
            ),
            "test_nll_improvement_mean": float(np.mean(improvements)),
            "test_nll_improvement_sample_std": (
                float(np.std(improvements, ddof=1)) if len(improvements) > 1 else 0.0
            ),
            "all_runs_improve_baseline": bool(np.all(improvements > 0.0)),
        },
    }
    if not all(math.isfinite(float(value)) for value in improvements):
        raise ValueError("non-finite ensemble NLL comparison")
    if output is not None:
        atomic_json(Path(output), payload)
    return payload
