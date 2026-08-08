from __future__ import annotations

import datetime as dt
import json
import logging
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from .certification import (
    CertificationResult,
    CertifiedModel,
    certify_family,
    compact_certified_models,
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
from .objective import ObjectiveSpec, SupportRecord, freeze_support_record
from .report import Certificate, RunReport
from .response import Context
from .rules import RuleIdentity, Support
from .search import SearchDiagnostics, SupportOptimizer, support_from_key, support_key


_DIAGNOSTIC_MAX_FIELDS = {
    "total_skeletons",
    "admissible_skeletons",
    "empty_skeletons",
    "theoretical_skeletons",
    "observed_motif_skeletons",
    "score_basin_nodes",
    "score_basin_seeds",
    "positive_primitive_roots",
    "route_family_candidates",
    "route_family_active_roots",
    "multi_source_roots",
    "adaptive_gradient_root_exact_fits",
    "adaptive_gradient_root_exact_fits_avoided",
    "family_ensemble_initial_score",
    "family_ensemble_final_score",
    "family_ensemble_active_supports",
}


def _pattern_label(pattern: tuple[str, tuple[int, ...]]) -> str:
    relation, antecedent = pattern
    separator = ">" if relation == "ordered" else ","
    return f"{relation}:{separator.join(map(str, antecedent))}"


def _rule_structure_counts(records: object) -> dict[str, dict[str, int]]:
    order = {"singleton": 0, "pair": 0, "triplet": 0}
    relation = {"atomic": 0, "unordered": 0, "ordered": 0}
    for record in records:
        for rule in record.support.rules:
            order[("singleton", "pair", "triplet")[rule.order - 1]] += 1
            relation[rule.relation] += 1
    return {"order": order, "relation": relation}


def _merge_route_diagnostics(
    target: SearchDiagnostics,
    payloads: list[dict[str, object]],
) -> None:
    """Merge deterministic process-shard execution counters.

    Route workers own disjoint start supports but each reconstructs the same
    immutable dictionary metadata.  Operational counters are additive;
    dictionary sizes, objective snapshots, byte budgets and peaks are gauges
    and therefore take their maximum.  Without this merge a parallel full run
    reported zeros for the very gradient/refit counters used to audit its
    speed and optimization policy.
    """

    for descriptor in fields(SearchDiagnostics):
        name = descriptor.name
        values = [payload.get(name, 0) for payload in payloads]
        if not values:
            continue
        current = getattr(target, name)
        gauge = bool(
            name in _DIAGNOSTIC_MAX_FIELDS
            or "maximum" in name
            or "peak" in name
            or name.endswith("_bytes")
        )
        if gauge:
            setattr(target, name, max([current, *values]))
        else:
            setattr(target, name, current + sum(values))


def _discovery_context(
    dataset: Dataset,
    fit_codes: np.ndarray,
    config: RunConfig,
    *,
    full_context: Context | None = None,
) -> tuple[Context, dict[str, object]]:
    """Construct an entity-level case-cohort HT discovery sample.

    Sampling an entire entity preserves every antecedent history and every
    strictly-future response row.  All entities with at least one D_fit target
    are included; noncases are a simple random sample without replacement.
    The inverse first-order inclusion probability is consequently exact.
    """
    fit_codes = np.sort(np.unique(np.asarray(fit_codes, dtype=np.int32)))
    if config.discovery_sampling == "full":
        context = (
            full_context
            if full_context is not None
            else Context.make(dataset, fit_codes)
        )
        if not np.array_equal(context.entity_codes, fit_codes):
            raise ValueError("prebuilt full discovery context does not match D_fit")
        return context, {
            "method": "full",
            "population_entities": len(fit_codes),
            "sample_entities": len(fit_codes),
            "case_entities": int(
                np.count_nonzero(np.isin(fit_codes, dataset.target_entities))
            ),
            "noncase_inclusion_probability": 1.0,
        }
    target_entity = np.zeros(dataset.n_entities, dtype=bool)
    target_entity[np.unique(dataset.target_entities)] = True
    cases = fit_codes[target_entity[fit_codes]]
    noncases = fit_codes[~target_entity[fit_codes]]
    requested = int(np.ceil(config.discovery_noncase_fraction * len(noncases)))
    sample_count = min(len(noncases), max(1, requested)) if len(noncases) else 0
    if sample_count == len(noncases):
        sampled_noncases = noncases
        inclusion_probability = 1.0
    else:
        generator = np.random.default_rng(config.discovery_sampling_seed)
        positions = np.sort(
            generator.choice(len(noncases), size=sample_count, replace=False)
        )
        sampled_noncases = noncases[positions]
        inclusion_probability = sample_count / len(noncases)
    codes = np.concatenate([cases, sampled_noncases]).astype(np.int32, copy=False)
    weights = np.concatenate(
        [
            np.ones(len(cases), dtype=np.float64),
            np.full(
                len(sampled_noncases),
                1.0 / inclusion_probability if sample_count else 1.0,
                dtype=np.float64,
            ),
        ]
    )
    context = Context.make(
        dataset,
        codes,
        entity_weights=weights,
        population_entities=len(fit_codes),
    )
    return context, {
        "method": "case_cohort_ipw",
        "population_entities": len(fit_codes),
        "sample_entities": len(codes),
        "case_entities": len(cases),
        "sampled_noncase_entities": len(sampled_noncases),
        "population_noncase_entities": len(noncases),
        "noncase_inclusion_probability": inclusion_probability,
        "sampling_seed": config.discovery_sampling_seed,
        "unit": "entity_complete_history",
    }


def _reference_discovery_context(
    full_dataset: Dataset,
    reference_dataset: Dataset,
    config: RunConfig,
) -> tuple[Context, dict[str, object]]:
    """Build a target-blind reference-cohort route context.

    Discovery may use a pre-registered deterministic entity subset, but every
    resulting support is subsequently refitted and rechecked on complete
    D_fit.  Predicate semantics and split ownership must be identical so this
    cannot be used to smuggle an outcome-dependent dictionary into search.
    """

    comparable = (
        "predicate_names",
        "predicate_roles",
        "likelihood",
        "time_unit",
        "ticks_per_unit",
        "adverse_event_name",
        "f0_contract",
    )
    for name in comparable:
        if getattr(reference_dataset, name) != getattr(full_dataset, name):
            raise ValueError(
                f"discovery reference differs from full dataset field: {name}"
            )
    if reference_dataset.n_entities >= full_dataset.n_entities:
        raise ValueError("discovery reference must be a strict entity subset")
    # ``np.isin`` on variable-width Unicode arrays performs an expensive
    # global sort/temporary allocation at this scale.  Exact hash membership
    # is linear, uses bounded object references, and checks the identical set
    # relation without touching event rows.
    full_entity_ids = set(full_dataset.entity_ids.tolist())
    if any(value not in full_entity_ids for value in reference_dataset.entity_ids):
        raise ValueError("discovery reference contains entities absent from full data")
    reference_fit, _, _ = reference_dataset.split(
        config.split_fractions, config.split_seed
    )
    context = Context.make(reference_dataset, reference_fit)
    return context, {
        "method": "target_blind_reference_cohort",
        "reference_dataset": str(reference_dataset.root),
        "reference_dataset_digest": reference_dataset.digest,
        "population_entities": full_dataset.n_entities,
        "reference_entities": reference_dataset.n_entities,
        "sample_entities": len(reference_fit),
        "unit": "entity_complete_history",
        "final_fit_verification": "complete_D_fit",
    }


def _rule_payload(rule: RuleIdentity) -> dict[str, object]:
    return {
        "antecedent": list(rule.antecedent),
        "window": rule.window,
        "sign": rule.sign,
        "kernel_rank": rule.kernel_rank,
        "relation": rule.relation,
        "hierarchical": rule.hierarchical,
        "support_additive": rule.support_additive,
        "history_marks": [list(mark) for mark in rule.history_marks],
    }


def _support_payload(support: Support) -> list[dict[str, object]]:
    return [_rule_payload(rule) for rule in support.rules]


def _support_from_payload(payload: list[dict[str, object]]) -> Support:
    return Support.of(
        RuleIdentity(
            tuple(int(value) for value in item["antecedent"]),
            int(item["window"]),
            int(item["sign"]),
            int(item.get("kernel_rank", 0)),
            str(item.get("relation", "auto")),
            bool(item.get("hierarchical", False)),
            tuple(
                (int(mark[0]), int(mark[1])) for mark in item.get("history_marks", [])
            ),
            bool(item.get("support_additive", False)),
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
        family_adjusted_pvalue=(
            None
            if payload.get("family_adjusted_pvalue") is None
            else float(payload["family_adjusted_pvalue"])
        ),
        multiplicity_method=str(
            payload.get(
                "multiplicity_method",
                "romano_wolf_stepdown_max_t",
            )
        ),
        romano_wolf_resamples=(
            None
            if payload.get("romano_wolf_resamples") is None
            else int(payload["romano_wolf_resamples"])
        ),
    )


def _record_payload(
    record: SupportRecord,
    predicate_names: tuple[str, ...],
    predicate_roles: tuple[str, ...],
    basis: np.ndarray,
    objective: ObjectiveSpec,
    certification_diagnostics: dict[str, object] | None = None,
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

    baseline_controls = []
    control_indices = [
        index for index, role in enumerate(predicate_roles) if role != "reported"
    ]
    role_signs = {
        "baseline_control": 1,
        "exposure_increase_control": 1,
        "exposure_decrease_control": -1,
    }
    for position, predicate in enumerate(control_indices):
        left = record.matrix.free_dimension + position * basis.shape[0]
        coefficients = record.fit.coefficients[left : left + basis.shape[0]]
        role = predicate_roles[predicate]
        profile = role_signs[role] * (coefficients @ basis)
        baseline_controls.append(
            {
                "predicate": predicate_names[predicate],
                "kernel": coefficients.tolist(),
                "lag_profile": profile.tolist(),
                "direction": direction(profile),
                "role": role,
                "reported": False,
                "certified_separately": False,
            }
        )
    rules = []
    certified_rule_diagnostics = (
        certification_diagnostics.get("rules", [])
        if isinstance(certification_diagnostics, dict)
        else []
    )
    if not isinstance(certified_rule_diagnostics, list):
        certified_rule_diagnostics = []
    for rule_index, (rule, block) in enumerate(
        zip(record.support.rules, record.matrix.rule_slices, strict=True)
    ):
        coefficients = record.fit.coefficients[block]
        rule_basis = (
            basis if rule.kernel_rank == 0 else np.mean(basis, axis=0, keepdims=True)
        )
        profile = float(rule.sign) * (coefficients @ rule_basis)
        diagnostic = (
            certified_rule_diagnostics[rule_index]
            if rule_index < len(certified_rule_diagnostics)
            and isinstance(certified_rule_diagnostics[rule_index], dict)
            else {}
        )
        total = diagnostic.get("total_contextual_probability", {})
        if not isinstance(total, dict):
            total = {}
        reported_total_sign = int(diagnostic.get("reported_total_sign", 0))
        reported_total_direction = str(
            diagnostic.get("reported_total_direction", "unidentified")
        )
        semantic_type = str(diagnostic.get("semantic_type", "total_state_rule"))
        rule_direction = direction(profile)
        reported_direction = reported_total_direction
        rules.append(
            {
                **_rule_payload(rule),
                "antecedent_names": [
                    predicate_names[index] for index in rule.antecedent
                ],
                "history_conditions": [
                    {
                        "antecedent_position": index,
                        "predicate": predicate_names[predicate],
                        "lookback": int(mark[0]),
                        "minimum_prior_count": int(mark[1]),
                        "interval": "[t-lookback,t)",
                    }
                    for index, (predicate, mark) in enumerate(
                        zip(
                            rule.antecedent,
                            (rule.history_marks or ((0, 0),) * len(rule.antecedent)),
                            strict=True,
                        )
                    )
                    if mark != (0, 0)
                ],
                "kernel": coefficients.tolist(),
                "kernel_dimension": int(len(coefficients)),
                "kernel_family": (
                    "full_m_knot" if rule.kernel_rank == 0 else "scalar_normalized"
                ),
                "lag_profile": profile.tolist(),
                "direction": reported_direction,
                "direction_semantics": (
                    "conditional_additive_modifier_given_lower_order_effects"
                    if rule.hierarchical
                    else "support_conditional_additive_component"
                    if rule.support_additive
                    else "direct_total_state_component_with_nested_state_masking"
                ),
                "semantic_type": semantic_type,
                "rule_sign": int(rule.sign),
                "rule_direction": rule_direction,
                "reported_total_sign": reported_total_sign,
                "reported_total_direction": reported_total_direction,
                "reported_direction_semantics": (
                    (
                        "fixed-predictor total contextual contribution including "
                        "the fitted strict lower-order hierarchy effects"
                    )
                    if rule.hierarchical
                    else (
                        "fixed-predictor additive contribution conditional on "
                        "the other selected reported support rules"
                    )
                    if rule.support_additive
                    else (
                        "fixed-predictor contribution of the reported total-state "
                        "block after strict nested-state masking"
                    )
                ),
                "fit_total_probability_mean": diagnostic.get(
                    "fit_total_probability_mean"
                ),
                "cert_total_probability_mean": total.get("raw_cert_mean"),
                "role": "reported_rule",
                "reported": True,
            }
        )
    closure = []
    closure_left = record.matrix.baseline_dimension
    for index, (term, sign) in enumerate(
        zip(record.matrix.closure, record.matrix.closure_signs, strict=True)
    ):
        left = closure_left + index * basis.shape[0]
        coefficients = record.fit.coefficients[left : left + basis.shape[0]]
        profile = float(sign) * (coefficients @ basis)
        closure.append(
            {
                "antecedent": list(term.antecedent),
                "antecedent_names": [
                    predicate_names[predicate] for predicate in term.antecedent
                ],
                "history_marks": [list(mark) for mark in term.history_marks],
                "window": int(term.window),
                "sign": int(sign),
                "kernel": coefficients.tolist(),
                "lag_profile": profile.tolist(),
                "direction": direction(profile),
                "role": "hierarchy_nuisance",
                "reported": False,
                "certified_separately": False,
            }
        )
    additive = any(rule.hierarchical for rule in record.support.rules)
    support_additive = any(rule.support_additive for rule in record.support.rules)
    if (record.matrix.closure or record.matrix.closure_dimension) and not additive:
        raise AssertionError("a reportable total-state model contains hidden closure")
    return {
        "key": support_key(record.support),
        "rules": rules,
        "baseline_controls": baseline_controls,
        "closure": closure,
        "interpretation": (
            (
                "Higher-order reported rules are signed additive modifiers "
                "estimated after their shared strict lower-order effects. "
                "Those lower-order hierarchy terms are nuisance components "
                "and are listed separately; total contextual direction and "
                "conditional modifier direction are both reported."
            )
            if additive
            else (
                "Every certified identity is an ordinary signed additive "
                "component with no automatic lower-order closure and no "
                "nested-state masking. Its direction is conditional only on "
                "the other reported rules explicitly present in this support."
            )
            if support_additive
            else (
                "Every certified identity is a directly signed total-state "
                "component. A selected strict superset masks selected lower-order "
                "states on its active footprint; incomparable rules remain "
                "additive. Excitation/inhibition is therefore read from the "
                "reported component rather than from a hidden interaction. There "
                "is no automatic hierarchy nuisance; only fixed baseline controls "
                "remain non-reportable."
            )
        ),
        "score": record.discovery_score,
        "rule_score": record.discovery_score,
        "total_mdl_score": record.score,
        "total_penalty": record.penalty,
        "dependency_effective_dimension": record.dependency_effective_dimension,
        "dependency_diagnostics": record.dependency_diagnostics,
        "reported_penalty": (
            None
            if not record.support.rules
            else objective.reported_branch_penalty(
                record.support,
                sum(
                    rule.kernel_dimension(objective.knot_count)
                    for rule in record.support.rules
                ),
            )
        ),
        "common_baseline_nll": record.closure_null_nll,
        "nll": record.fit.nll,
        "projected_kkt": record.fit.projected_kkt,
    }


def _default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _completed_path_tuples(
    paths: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> tuple[tuple[Support, Support, dict[str, object]], ...]:
    """Convert checkpoint path dictionaries to the search resume contract."""
    converted = []
    for path in paths:
        start = path.get("start")
        terminal = path.get("terminal")
        if not isinstance(start, str) or not isinstance(terminal, str):
            raise ValueError("route path is missing a start or terminal support")
        converted.append(
            (
                support_from_key(start),
                support_from_key(terminal),
                path,
            )
        )
    return tuple(converted)


def _validate_preassigned_partition_contract(
    dataset: Dataset,
    fractions: tuple[float, float, float],
    seed: int | None = None,
) -> None:
    """Reject a config whose split contract differs from immutable data.

    Explicit partitions are already frozen in ``entities.parquet``.  Silently
    accepting another ratio in YAML would change only the reported config,
    not the entities used by fitting/certification/testing.
    """

    if dataset.partitions is None:
        return
    manifest = json.loads((dataset.root / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest.get("provenance", {}).get("partition", {}).get("fractions")
    if declared is None:
        # Legacy synthetic/pre-v12 datasets froze entity partition codes before
        # the ratio was added to provenance.  They remain loadable; every v12
        # Aave dataset declares the ratio and is checked below.
        return
    declared_array = np.asarray(declared, dtype=np.float64)
    requested = np.asarray(fractions, dtype=np.float64)
    if declared_array.shape != (3,) or not np.allclose(
        declared_array,
        requested,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "config split_fractions do not match the preassigned dataset partition"
        )
    declared_seed = manifest.get("provenance", {}).get("partition", {}).get("seed")
    if (
        seed is not None
        and declared_seed is not None
        and int(declared_seed) != int(seed)
    ):
        raise ValueError(
            "config split_seed does not match the preassigned dataset partition"
        )


def _route_shard_worker(
    config: RunConfig,
    *,
    device: str,
    parent_config_digest: str,
    search_dataset_path: str,
    dataset_digest: str,
    search_state: dict[str, object],
    window_dictionary: dict[tuple[int, ...], tuple[int, ...]],
    base_paths: tuple[dict[str, object], ...],
    assigned_starts: tuple[list[dict[str, object]], ...],
    checkpoint_path: str,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Explore one root shard with identical unrestricted search rules.

    Each positive atom is only an initialization: its rules may be dropped by
    the exact terminal audit.  A process shard may recompute a suffix also
    reached by another shard, but that changes scheduling only; every route
    uses the same objective, candidate dictionary, and terminal audit.
    """
    worker_config = replace(
        config,
        dataset=search_dataset_path,
        discovery_reference_dataset=None,
        route_workers=1,
        # Each shard owns one GPU, but matrix-free exact evaluation has a
        # CPU-heavy touched-state construction phase.  Give the shard its
        # proportional share of exact producers so three bounded host builds
        # can feed that GPU while another Newton objective is resident.  The
        # previous hard-coded value of one silently disabled the accelerated
        # exact scheduler in every real multi-root full run.
        exact_workers=max(1, math.ceil(config.exact_workers / config.route_workers)),
        # Shards run concurrently on disjoint GPUs.  Divide physical CPU
        # producers between them instead of giving every process the complete
        # pool and oversubscribing the 12-core host.
        pricing_workers=max(
            1, math.ceil(config.pricing_workers / config.route_workers)
        ),
        pricing_devices=(device,),
        cache_bytes=max(512 * 1024**2, config.cache_bytes // config.route_workers),
    )
    dataset = Dataset.load(worker_config.dataset)
    if dataset.digest != dataset_digest:
        raise ValueError("route shard dataset digest mismatch")
    fit_codes, _, _ = dataset.split(
        worker_config.split_fractions, worker_config.split_seed
    )
    discovery_context = Context.make(dataset, fit_codes)
    optimizer = SupportOptimizer(
        discovery_context,
        worker_config,
        window_dictionary=window_dictionary,
    )
    optimizer.restore_search_state(search_state)
    shard_path = Path(checkpoint_path)
    completed = _completed_path_tuples(base_paths)
    active = None
    if shard_path.is_file():
        restored = json.loads(shard_path.read_text(encoding="utf-8"))
        if (
            restored.get("schema") == CHECKPOINT_SCHEMA
            and restored.get("config_digest") == parent_config_digest
            and restored.get("dataset_digest") == dataset_digest
        ):
            raw_shard_state = restored.get("search_state")
            if isinstance(raw_shard_state, dict):
                optimizer.restore_search_state(raw_shard_state)
            completed = tuple(
                (
                    _support_from_payload(item["start"]),
                    _support_from_payload(item["terminal"]),
                    item["path"],
                )
                for item in restored.get("completed_paths", [])
            )
            raw_active = restored.get("active_path")
            if isinstance(raw_active, dict) and isinstance(
                raw_active.get("record"), dict
            ):
                active = (
                    _support_from_payload(raw_active["start"]),
                    _support_from_payload(raw_active["current"]),
                    tuple(raw_active.get("moves", [])),
                    raw_active.get("record"),
                )

    allowed = frozenset(_support_from_payload(item) for item in assigned_starts)

    def save(
        finished: tuple[tuple[Support, Support, dict[str, object]], ...],
        running: tuple[Support, SupportRecord, tuple[dict[str, object], ...]] | None,
    ) -> None:
        atomic_json(
            shard_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "stage": "route_shard_progress",
                "config_digest": parent_config_digest,
                "dataset_digest": dataset_digest,
                "completed_paths": [
                    {
                        "start": _support_payload(start),
                        "terminal": _support_payload(terminal),
                        "path": path,
                    }
                    for start, terminal, path in finished
                ],
                "active_path": (
                    {
                        "start": _support_payload(running[0]),
                        "current": _support_payload(running[1].support),
                        "moves": list(running[2]),
                        "record": optimizer._checkpoint_record_payload(running[1]),
                    }
                    if running is not None
                    else None
                ),
                "diagnostics": asdict(optimizer.diagnostics),
                "search_state": optimizer.checkpoint_search_state(),
            },
        )

    try:
        result = optimizer.search(
            completed_paths=completed,
            active_path=active,
            progress_callback=save,
            allowed_starts=allowed,
            finalize_family=False,
        )
        assigned = {_support_from_payload(payload) for payload in assigned_starts}
        inherited = {support_from_key(str(path["start"])) for path in base_paths}
        paths = tuple(
            path
            for path in result.paths
            if (
                support_from_key(str(path["start"])) in assigned
                or support_from_key(str(path["start"])) not in inherited
            )
        )
        return paths, asdict(result.diagnostics)
    finally:
        optimizer.release_search_caches()
        optimizer.close()


def run(
    config: RunConfig, *, run_dir: str | Path | None = None, resume: bool = False
) -> RunReport:
    run_started = time.perf_counter()
    config.validate()
    dataset = Dataset.load(config.dataset)
    if dataset.likelihood == "continuous_poisson" and (
        config.dependency_aware_mdl or config.frequency_effect_separation
    ):
        raise ValueError(
            "continuous_poisson currently requires dependency_aware_mdl=false "
            "and frequency_effect_separation=false; raw nanosecond timestamps "
            "cannot be treated as a dense regular calendar grid"
        )
    _validate_preassigned_partition_contract(
        dataset, config.split_fractions, config.split_seed
    )
    required_impact_lag = dataset.f0_contract.get("required_impact_lag")
    if required_impact_lag is not None and config.impact_lag != int(
        required_impact_lag
    ):
        raise ValueError(
            "dataset baseline contract requires impact_lag="
            f"{int(required_impact_lag)}, got {config.impact_lag}"
        )
    required_kernel_knots = dataset.f0_contract.get("required_kernel_knots")
    if required_kernel_knots is not None and config.knot_count != int(
        required_kernel_knots
    ):
        raise ValueError(
            "dataset response-grid contract requires knot_count="
            f"{int(required_kernel_knots)}, got {config.knot_count}"
        )
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
    # Search owns the expensive exact frontier and historically logged only
    # to an unconfigured module logger.  Sharing this run-scoped handler makes
    # every long audit stage observable without polling or instrumentation.
    search_logger = logging.getLogger("crbstpp.search")
    search_logger.setLevel(logging.INFO)
    search_logger.addHandler(handler)
    manifest_path = run_dir / "manifest.json"
    if existed:
        if not manifest_path.is_file():
            raise ValueError("existing run directory has no CRBS manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "crbstpp.run.v1":
            raise ValueError("unsupported run schema")
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
        search_logger.removeHandler(handler)
        return RunReport(
            schema=RESULT_SCHEMA,
            algorithm="CRBS-TPP",
            config_digest=config.digest,
            dataset_digest=dataset.digest,
            support_count=int(payload.get("search", {}).get("family_size", 0)),
            certified_count=int(
                payload.get("certification", {}).get(
                    "selected_count",
                    payload.get("certification", {}).get("certified_count", 0),
                )
            ),
            result=payload,
        )
    fit_codes, cert_codes, test_codes = dataset.split(
        config.split_fractions, config.split_seed
    )
    fit_context = Context.make(dataset, fit_codes)
    discovery_dataset = dataset
    if config.discovery_sampling == "reference_cohort":
        if config.discovery_reference_dataset is None:
            raise AssertionError("validated reference discovery has no dataset")
        discovery_dataset = Dataset.load(config.discovery_reference_dataset)
        discovery_context, discovery_sampling = _reference_discovery_context(
            dataset, discovery_dataset, config
        )
    else:
        discovery_context, discovery_sampling = _discovery_context(
            dataset, fit_codes, config, full_context=fit_context
        )
    search_optimizer = SupportOptimizer(discovery_context, config)
    optimizer = search_optimizer
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
        discovery_family = tuple(
            search_optimizer._attach_rule_score(record)
            for record in search_optimizer.fit_many(
                resumed_supports,
                search_optimizer.records[Support(())],
            )
        )
        search_optimizer.family_ensemble_weights = {
            _support_from_payload(item["support"]): float(item["weight"])
            for item in checkpoint.get("family_discovery_weights", [])
        }
        search_result = None
        search_seconds = 0.0
    else:
        logger.info("starting support search")
        search_started = time.perf_counter()
        resumed_paths: tuple[tuple[Support, Support, dict[str, object]], ...] = ()
        resumed_active: (
            tuple[
                Support,
                Support,
                tuple[dict[str, object], ...],
                dict[str, object] | None,
            ]
            | None
        ) = None
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
            # An interrupted v13 write can end before the active SupportRecord
            # payload is installed.  Deterministically replay that unfinished
            # route from its restored exact root; completed routes remain
            # reusable.  Older checkpoint schemas are rejected before this
            # branch by ``load_checkpoint``.
            if isinstance(active, dict) and isinstance(active.get("record"), dict):
                restored_current = _support_from_payload(active["current"])
                # Early v14 checkpoint payloads omitted the additive
                # ``hierarchical`` flag even though the canonical record key
                # already contained it.  Prefer that lossless identity and
                # reject any unrelated disagreement instead of silently
                # resuming an additive support as a total-state support.
                record_key = active["record"].get("support")
                if isinstance(record_key, str):
                    keyed_current = support_from_key(record_key)
                    if tuple(
                        (rule.pattern_key, rule.window, rule.sign, rule.kernel_rank)
                        for rule in keyed_current.rules
                    ) != tuple(
                        (rule.pattern_key, rule.window, rule.sign, rule.kernel_rank)
                        for rule in restored_current.rules
                    ):
                        raise ValueError(
                            "active checkpoint support identity disagrees with record"
                        )
                    restored_current = keyed_current
                resumed_active = (
                    _support_from_payload(active["start"]),
                    restored_current,
                    tuple(active.get("moves", [])),
                    active.get("record"),
                )
            raw_search_state = checkpoint.get("search_state")
            if isinstance(raw_search_state, dict):
                search_optimizer.restore_search_state(raw_search_state)
            raw_diagnostics = checkpoint.get("diagnostics")
            if isinstance(raw_diagnostics, dict):
                # Execution counters are part of the resumable audit trail.
                # Restoring only fitted supports made an interrupted run's
                # final report look as if its completed route work never
                # happened, even though the exact paths were reused.
                _merge_route_diagnostics(
                    search_optimizer.diagnostics,
                    [raw_diagnostics],
                )

        def save_search_progress(
            completed: tuple[tuple[Support, Support, dict[str, object]], ...],
            active: tuple[Support, SupportRecord, tuple[dict[str, object], ...]] | None,
        ) -> None:
            checkpoint_started = time.perf_counter()
            search_state = search_optimizer.checkpoint_search_state()
            state_finished = time.perf_counter()
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
                            "current": _support_payload(active[1].support),
                            "moves": list(active[2]),
                            "fit": active[1].fit.to_dict(),
                            "record": search_optimizer._checkpoint_record_payload(
                                active[1]
                            ),
                        }
                        if active is not None
                        else None
                    ),
                    "search_state": search_state,
                    "diagnostics": asdict(search_optimizer.diagnostics),
                },
            )
            elapsed = time.perf_counter() - checkpoint_started
            if elapsed >= 1.0:
                logger.info(
                    "search checkpoint timing state=%.3f write=%.3f total=%.3f",
                    state_finished - checkpoint_started,
                    elapsed - (state_finished - checkpoint_started),
                    elapsed,
                )

        # Persist a valid restart boundary before the expensive W/sign profile.
        # The profile itself is deterministic and will be replayed on resume,
        # while accepted support paths recorded later remain reusable.
        if checkpoint is None:
            save_search_progress((), None)
        parallel_routes = (
            config.route_workers > 1
            and config.search_mode
            in {
                "fast_block_score",
                "safe_column_generation",
                "gap_safe_rashomon_path",
                "successor_rashomon_path",
                "atomic_rashomon_frontier",
            }
            and len(config.pricing_devices) >= config.route_workers
        )
        shard_paths: list[Path] = []
        if parallel_routes:
            # Profile the finite dictionary once and complete the empty route
            # first.  Root shards are an execution schedule only: every seed is
            # subsequently audited by the same unrestricted Add/Drop oracle.
            empty_support = Support(())
            base_result = search_optimizer.search(
                completed_paths=resumed_paths,
                active_path=(
                    resumed_active
                    if resumed_active is not None and resumed_active[0] == empty_support
                    else None
                ),
                progress_callback=save_search_progress,
                allowed_starts=frozenset((empty_support,)),
                finalize_family=False,
            )
            base_paths = tuple(base_result.paths)
            search_state = search_optimizer.checkpoint_search_state()
            base_execution_diagnostics = asdict(search_optimizer.diagnostics)
            # Atomic discovery starts from the exact-positive profiled roots.
            # ``route_seed_records`` also contains negative gap-path seeds for
            # other search modes. Scheduling those 23 records while the atomic
            # optimizer itself admitted only five roots produced misleading
            # progress and unnecessary shard setup. This selects the exact
            # same starts as SupportOptimizer.search().
            frozen_ensemble_order = search_state.get("ensemble_residual_route_order")
            raw_roots = (
                search_state.get("profiled_roots", [])
                if config.search_mode == "atomic_rashomon_frontier"
                else search_state.get(
                    "route_seed_records",
                    search_state.get("profiled_roots", []),
                )
            )
            if not isinstance(raw_roots, list):
                raise ValueError("profiled route roots are missing")
            raw_route_antecedents = search_state.get("route_root_antecedents", [])
            if not isinstance(raw_route_antecedents, list):
                raise ValueError("profiled route antecedents are missing")
            route_antecedents = {
                tuple(int(value) for value in item)
                for item in raw_route_antecedents
                if isinstance(item, list)
            }
            if (
                config.ensemble_residual_search or config.rule_effect_stacking_search
            ) and isinstance(frozen_ensemble_order, list):
                all_roots = tuple(
                    support_from_key(str(value)) for value in frozen_ensemble_order
                )
            else:
                all_roots = tuple(
                    support_from_key(str(item["support"]))
                    for item in raw_roots
                    if isinstance(item, dict)
                    and (
                        not route_antecedents
                        or support_from_key(str(item["support"])).rules[0].antecedent
                        in route_antecedents
                    )
                )
            completed_starts = {
                support_from_key(str(path["start"])) for path in base_paths
            }
            remaining = sorted(
                (support for support in all_roots if support not in completed_starts),
                key=lambda support: (
                    -len(support.rules[0].antecedent),
                    support.rules,
                ),
            )
            # Keep every root, but co-locate roots in the same deterministic
            # standalone basin.  They then share one process-local canonical
            # support cache and terminal-pointer DAG.  Splitting a basin
            # across GPUs would recompute any convergent suffix independently.
            # Greedy whole-group bin packing balances the two physical GPUs
            # without changing a single search decision.
            raw_schedule = search_state.get("rashomon_root_schedule", [])
            if not isinstance(raw_schedule, list):
                raise ValueError("profiled Rashomon root schedule is missing")
            root_schedule: dict[Support, Support] = {}
            for item in raw_schedule:
                if not isinstance(item, dict):
                    continue
                root_schedule[support_from_key(str(item["root"]))] = support_from_key(
                    str(item["representative"])
                )
            grouped: dict[Support, list[Support]] = {}
            for support in remaining:
                representative = root_schedule.get(support, support)
                grouped.setdefault(representative, []).append(support)
            assignments: list[list[Support]] = [[] for _ in range(config.route_workers)]
            for representative, group in sorted(
                grouped.items(),
                key=lambda item: (-len(item[1]), item[0].rules),
            ):
                worker = min(
                    range(config.route_workers),
                    key=lambda index: (len(assignments[index]), index),
                )
                assignments[worker].extend(
                    sorted(
                        group,
                        key=lambda support: (
                            support != representative,
                            support.rules,
                        ),
                    )
                )
            active_assignments = [
                (index, supports)
                for index, supports in enumerate(assignments)
                if supports
            ]
            if active_assignments:
                frozen_window_dictionary = {
                    antecedent: tuple(windows)
                    for antecedent, windows in search_optimizer.window_dictionary.items()
                }
                frozen_window_quantile_dictionary = {
                    antecedent: {
                        int(window): tuple(quantiles)
                        for window, quantiles in labels.items()
                    }
                    for antecedent, labels in (
                        search_optimizer.window_quantile_dictionary.items()
                    )
                }
                # ``spawn`` workers do not inherit the parent's Python heap.
                # Keep its immutable completion/profile mmap and exact-root
                # metadata for the canonical post-shard audit; dropping all
                # 6+ GiB of host cache here and reconstructing the same state
                # afterwards was a multi-minute no-op on Aave.  Only
                # process-global CUDA workspaces can contend with workers.
                search_optimizer.release_accelerator_resources()
                shard_root = run_dir / "_route_shards"
                shard_root.mkdir(parents=True, exist_ok=True)
                worker_started = time.perf_counter()
                merged = {
                    support_from_key(str(path["start"])): path for path in base_paths
                }
                worker_diagnostics: list[dict[str, object]] = []
                context = mp.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=len(active_assignments),
                    mp_context=context,
                ) as executor:
                    futures = []
                    for index, supports in active_assignments:
                        shard_path = shard_root / f"shard_{index}.json"
                        shard_paths.append(shard_path)
                        futures.append(
                            executor.submit(
                                _route_shard_worker,
                                config,
                                device=config.pricing_devices[index],
                                parent_config_digest=config.digest,
                                search_dataset_path=str(discovery_dataset.root),
                                dataset_digest=discovery_dataset.digest,
                                search_state=search_state,
                                window_dictionary=frozen_window_dictionary,
                                base_paths=base_paths,
                                assigned_starts=tuple(
                                    _support_payload(support) for support in supports
                                ),
                                checkpoint_path=str(shard_path),
                            )
                        )
                    for future in futures:
                        paths, diagnostics = future.result()
                        worker_diagnostics.append(diagnostics)
                        for path in paths:
                            merged[support_from_key(str(path["start"]))] = path
                logger.info(
                    "completed route shards workers=%d roots=%d seconds=%.3f",
                    len(active_assignments),
                    len(remaining),
                    time.perf_counter() - worker_started,
                )
                merged_paths = _completed_path_tuples(
                    list(
                        sorted(
                            merged.values(),
                            key=lambda path: support_from_key(str(path["start"])).rules,
                        )
                    )
                )
                # Reuse the canonical parent optimizer and restore every
                # frozen exact path.  No route or immutable dictionary state
                # is recomputed; family compaction and D_cert see the same
                # union as serial traversal.
                search_optimizer.window_quantile_dictionary = (
                    frozen_window_quantile_dictionary
                )
                optimizer = search_optimizer
                search_optimizer.restore_search_state(search_state)
                _merge_route_diagnostics(
                    search_optimizer.diagnostics,
                    [base_execution_diagnostics, *worker_diagnostics],
                )
                # Persist the merged worker paths before the parent begins the
                # expensive global family audit.  An OOM or interruption after
                # this boundary now resumes from the completed route DAG
                # instead of replaying every shard.
                save_search_progress(merged_paths, None)
                search_result = search_optimizer.search(
                    completed_paths=merged_paths,
                    progress_callback=save_search_progress,
                )
            else:
                search_result = search_optimizer.search(
                    completed_paths=_completed_path_tuples(list(base_paths)),
                    progress_callback=save_search_progress,
                )
        else:
            search_result = search_optimizer.search(
                completed_paths=resumed_paths,
                active_path=resumed_active,
                progress_callback=save_search_progress,
            )
        search_seconds = time.perf_counter() - search_started
        discovery_family = search_result.family

    discovery_family_size = len(discovery_family)
    full_verification_started = time.perf_counter()
    uses_reduced_discovery = bool(
        config.discovery_sampling == "case_cohort_ipw"
        or config.discovery_reference_dataset is not None
    )
    if uses_reduced_discovery:
        optimizer = SupportOptimizer(
            fit_context,
            config,
            fit_only=True,
            window_dictionary=search_optimizer.window_dictionary,
        )
        full_candidates = optimizer.fit_many(
            tuple(record.support for record in discovery_family),
            optimizer.records[Support(())],
        )
        full_candidates = [
            optimizer._attach_rule_score(record) for record in full_candidates
        ]
        family = tuple(
            record
            for record in full_candidates
            if record.fit.converged
            and record.score > config.search_tolerance
            and record.discovery_score > config.search_tolerance
        )
    else:
        family = discovery_family
    precert_family_input_size = len(family)
    if config.precert_family_compaction:
        # This is deliberately before checkpoint freezing, optimizer
        # certification preparation, every D_cert statistic, and
        # Romano--Wolf. Thus D_cert cannot influence the multiplicity family.
        family = optimizer.compact_before_certification(tuple(family))
    precert_compacted_size = len(family)
    if config.ensemble_irreducible_family:
        # Freeze one coherent D_fit family objective before D_cert. The exact
        # fixed-intensity simplex Add/Drop audit removes supports whose best
        # ensemble weight is zero or whose removal improves coded family MDL.
        family = optimizer._family_ensemble_objective(tuple(family))
    # Checkpoints resumed from older builds and full-data verification both
    # return observation-sized records.  Certification consumes only frozen
    # identities and coefficients, so enforce metadata-only ownership at this
    # stage regardless of how the family was produced.
    family = tuple(freeze_support_record(record) for record in family)
    discovery_family = ()
    if uses_reduced_discovery:
        full_candidates = []
    full_verification_seconds = time.perf_counter() - full_verification_started
    if not (checkpoint and checkpoint.get("stage") == "certification_complete"):
        atomic_json(
            checkpoint_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "stage": "search_complete",
                "config_digest": config.digest,
                "dataset_digest": dataset.digest,
                "family": [_support_payload(record.support) for record in family],
                "family_discovery_weights": [
                    {
                        "support": _support_payload(support),
                        "weight": weight,
                    }
                    for support, weight in sorted(
                        search_optimizer.family_ensemble_weights.items(),
                        key=lambda item: item[0].rules,
                    )
                ],
                "diagnostics": asdict(search_optimizer.diagnostics),
                "discovery_sampling": discovery_sampling,
                "sample_family_size": discovery_family_size,
                "full_verified_family_size": len(family),
                "precert_family_input_size": precert_family_input_size,
                "precert_family_output_size": len(family),
            },
        )
    if search_optimizer is not optimizer:
        search_optimizer.release_search_caches()
        search_optimizer.close()
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
        restored_models_tuple = tuple(restored_models)
        restored_certified = tuple(
            model for model in restored_models_tuple if model.certificate.certified
        )
        certification = CertificationResult(
            restored_models_tuple,
            restored_certified,
            compact_certified_models(restored_models_tuple),
            len(restored_models_tuple),
        )
        certification_seconds = 0.0
    else:
        logger.info("certifying family_size=%d", len(family))
        certification_started = time.perf_counter()
        cert_context = Context.make(dataset, cert_codes)
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
                "family_discovery_weights": [
                    {
                        "support": _support_payload(support),
                        "weight": weight,
                    }
                    for support, weight in sorted(
                        search_optimizer.family_ensemble_weights.items(),
                        key=lambda item: item[0].rules,
                    )
                ],
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
    # Every independently certified Rashomon alternative enters the frozen
    # simplex.  A component may receive an exact zero optimum weight, but its
    # certified identity remains in the result instead of being mistaken for
    # a failed rule.  ``selected`` is still reported as the compact
    # non-dominated presentation family; it does not censor ensemble columns.
    certified_supports = tuple(
        model.record.support for model in certification.certified
    )
    optimizer.release_search_caches()
    ensemble_started = time.perf_counter()
    test_context = Context.make(dataset, test_codes)
    ensemble = fit_ensemble(
        combined_context,
        test_context,
        certified_supports,
        config,
        closure_signs=dict(optimizer._closure_signs),
        baseline_warm_start=optimizer.records[Support(())].fit.coefficients,
        support_warm_starts={
            model.record.support: model.record.fit.coefficients
            for model in certification.certified
        },
        mixture_warm_start=search_optimizer.family_ensemble_weights,
    )
    ensemble_seconds = time.perf_counter() - ensemble_started
    search_payload: dict[str, object]
    if search_result is None:
        search_payload = {
            "resumed": True,
            "family_size": len(family),
            "diagnostics": asdict(search_optimizer.diagnostics),
        }
    else:
        search_payload = {
            "resumed": False,
            "family_size": len(family),
            "sample_family_size": len(search_result.family),
            "terminal_count": len(search_result.terminals),
            "positive_atom_count": len(search_result.positive_atoms),
            "compact_candidate_count": len(search_result.compact_candidates),
            "paths": search_result.paths,
            "diagnostics": asdict(search_result.diagnostics),
        }
    search_payload["mode"] = config.search_mode
    search_payload["terminal_add_audit"] = config.terminal_add_audit
    search_payload["route_policy"] = (
        (
            "adaptive_profiled_gradient_racing_terminal_exact"
            if config.adaptive_gradient_racing
            else "complete_score_cascade_terminal_exact"
        )
        if config.search_mode == "fast_block_score"
        else (
            (
                "successor_equivalence_rashomon_terminal_exact"
                if config.search_mode == "successor_rashomon_path"
                else (
                    (
                        (
                            (
                                "predictive_family_and_fisher_mdl_rashomon_"
                                "basin_block_score_terminal_exact"
                                if config.predictive_basin_rashomon_search
                                else "score_basin_representative_ensemble_"
                                "residual_block_score_terminal_exact"
                            )
                            if (
                                config.ensemble_residual_search
                                or config.rule_effect_stacking_search
                            )
                            else "monotone_block_score_rashomon_terminal_exact"
                        )
                        if config.terminal_add_audit == "block_score"
                        else "atomic_descendant_safe_shared_rashomon_frontier"
                    )
                    if config.search_mode == "atomic_rashomon_frontier"
                    else (
                        "gap_safe_gradient_continuation_rashomon_terminal_exact"
                        if config.search_mode == "gap_safe_rashomon_path"
                        else "safe_column_generation_representative_rashomon"
                    )
                )
            )
            if config.search_mode
            in {
                "safe_column_generation",
                "gap_safe_rashomon_path",
                "successor_rashomon_path",
                "atomic_rashomon_frontier",
            }
            else "exact_add_drop_identity_stationary"
        )
    )
    search_payload["formation_windows"] = list(config.formation_windows)
    search_payload["formation_window_mode"] = config.formation_window_mode
    search_payload["formation_window_quantiles"] = list(
        config.formation_window_quantiles
    )
    search_payload["frozen_window_dictionary"] = {
        (
            ",".join(map(str, pattern[1]))
            if config.temporal_relations == ("unordered",)
            else _pattern_label(pattern)
        ): list(windows)
        for pattern, windows in sorted(optimizer.window_dictionary.items())
    }
    search_payload["frozen_window_quantile_labels"] = {
        (
            ",".join(map(str, pattern[1]))
            if config.temporal_relations == ("unordered",)
            else _pattern_label(pattern)
        ): {
            str(window): (
                ["W0"]
                if window == 0
                else [f"Q{int(round(100.0 * quantile)):02d}" for quantile in quantiles]
            )
            for window, quantiles in sorted(labels.items())
        }
        for pattern, labels in sorted(optimizer.window_quantile_dictionary.items())
    }
    search_payload["family_discovery_weights"] = {
        support_key(support): weight
        for support, weight in sorted(
            search_optimizer.family_ensemble_weights.items(),
            key=lambda item: item[0].rules,
        )
    }
    predictive_supports = frozenset(
        support
        for support, weight in search_optimizer.family_ensemble_weights.items()
        if weight > max(1.0e-12, 10.0 * config.solver_tolerance)
    )
    search_payload["family_active_set"] = [
        support_key(support)
        for support in sorted(predictive_supports, key=lambda item: item.rules)
    ]
    basin_members: dict[Support, list[Support]] = {}
    basin_map = getattr(search_optimizer, "_rashomon_basin_by_support", {})
    frozen_family_supports = frozenset(record.support for record in family)
    for record in family:
        representative = basin_map.get(record.support, record.support)
        if representative not in frozen_family_supports:
            representative = record.support
        basin_members.setdefault(representative, []).append(record.support)
    search_payload["route_equivalence_classes"] = [
        {
            "representative": support_key(representative),
            "predictive": any(member in predictive_supports for member in members),
            "members": [
                support_key(member)
                for member in sorted(members, key=lambda item: item.rules)
            ],
        }
        for representative, members in sorted(
            basin_members.items(), key=lambda item: item[0].rules
        )
    ]
    search_payload["ensemble_reduced_costs"] = {
        support_key(support): value
        for support, value in sorted(
            getattr(search_optimizer, "ensemble_reduced_costs", {}).items(),
            key=lambda item: item[0].rules,
        )
    }
    search_payload["frequency_effect_separation"] = {
        "enabled": bool(config.frequency_effect_separation),
        "moves": {
            support_key(support): diagnostics
            for support, diagnostics in sorted(
                getattr(search_optimizer, "frequency_evidence", {}).items(),
                key=lambda item: item[0].rules,
            )
        },
    }
    search_payload["candidate_pattern_counts"] = {
        "order": {
            "singleton": sum(len(pattern[1]) == 1 for pattern in optimizer.patterns),
            "pair": sum(len(pattern[1]) == 2 for pattern in optimizer.patterns),
            "triplet": sum(len(pattern[1]) == 3 for pattern in optimizer.patterns),
        },
        "relation": {
            name: sum(pattern[0] == name for pattern in optimizer.patterns)
            for name in ("atomic", "unordered", "ordered")
        },
    }
    search_payload["terminal_rule_counts"] = _rule_structure_counts(
        () if search_result is None else search_result.terminals
    )
    search_payload["family_rule_counts"] = _rule_structure_counts(family)
    search_payload["max_rules_per_support"] = config.max_rules_per_support
    search_payload["discovery_sampling"] = discovery_sampling
    search_payload["full_verification_rejections"] = discovery_family_size - len(family)
    search_payload["precert_family_compaction"] = {
        "enabled": bool(config.precert_family_compaction),
        "input": precert_family_input_size,
        "compacted": precert_compacted_size,
        "output": len(family),
        "fit_only": True,
        "ensemble_irreducible": bool(config.ensemble_irreducible_family),
        "before_certification_and_romano_wolf": True,
    }
    search_payload["objective"] = (
        (
            "two_way_wallet_calendar_dependency_CLBIC_"
            if config.dependency_aware_mdl
            else "common_baseline_"
        )
        + (
            "closure_matched_additive_hierarchy_rule_MDL_"
            if config.effect_model == "additive_hierarchy"
            else "support_relative_additive_rule_MDL_"
            if config.effect_model == "support_additive"
            else "total_state_rule_MDL_"
        )
        + "with_local_representation_audit_and_family_intensity_mixture_MDL_selection"
    )
    proposal_role = (
        "which is the approximate intermediate-route admission rule; it may "
        "discard an exact discrete basin before terminal refinement"
        if config.adaptive_gradient_racing
        else "which orders work but never accepts or rejects a support"
    )
    representation_role = "with exact W/sign coordinate reoptimization"
    terminal_kernel_role = (
        "choose one normalized amplitude or the full M-knot kernel by exact common MDL"
        if config.adaptive_kernel_mdl
        else "retain the configured full M-knot kernel"
    )
    terminal_add_role = (
        "exact-safe finite-dictionary family-aware Add"
        if config.terminal_add_audit == "exact"
        else "certified block-score finite-dictionary family-aware Add"
    )
    representation_contract = (
        "support-additive representations retain exact W/sign auditing, while "
        "structural representation pruning removes only prediction-equivalent "
        "or strictly fit-MDL-dominated supports"
        if config.effect_model == "support_additive"
        else (
            "every selected pair/triplet receives an exact lower/high-order "
            "pair-lattice or Add/Drop/Swap one-exchange representation audit "
            f"{representation_role}; globally MDL-positive high-order "
            "representations are additionally retained only when exact joint-state "
            "footprint MDL makes them Pareto-distinct from the lower-order winner"
        )
    )
    add_contract = (
        "each Add is compared only with the current selected support and every "
        "Drop only with its one-rule-deleted support"
        if config.effect_model == "support_additive"
        else (
            "at route creation a standalone-positive rule joins another support "
            "only when a valid joint lower bound or exact joint fit beats the "
            "exact separately fitted two-component simplex family; unresolved "
            "comparisons fail open to exact fitting"
        )
    )
    effect_contract = (
        "all higher-order rules use shared, complexity-counted strict lower-order "
        "main effects and one signed additive modifier; discovery credits the "
        "modifier only relative to its matched hierarchy null; certification "
        "reports and tests the direct modifier direction separately from the "
        "complete contextual direction"
        if config.effect_model == "additive_hierarchy"
        else (
            "reported rules are ordinary additive components with no automatic "
            "hierarchy closure and no nested-state masking; every Add/Drop is "
            "evaluated relative to the currently selected support; a high-order "
            "rule alone is a conjunction-state association, and it is a "
            "conditional modifier only when its lower-order rules are explicitly "
            "selected in the same support"
        )
        if config.effect_model == "support_additive"
        else (
            "no hidden hierarchy coefficient is fitted and a selected strict "
            "superset masks selected lower-order states on its response footprint"
        )
    )
    search_payload["guarantee"] = (
        "all reported models use the same preregistered baseline; "
        f"{effect_contract}; every accepted route move strictly improves a "
        "certified feasible full-M-knot route endpoint; final rule identities "
        f"{terminal_kernel_role}; singleton, pair and "
        "triplet proposals share the same normalized one-amplitude relaxation, "
        f"{proposal_role}; exact terminal correction is followed by "
        f"{terminal_add_role}, exact Drop and exact W/sign "
        f"auditing; {add_contract}; "
        "standalone routes are the deterministic same-order exchange-graph "
        "score-basin maxima, while every screened atom remains an Add candidate; "
        f"{representation_contract}; "
        "prediction-equivalent and nested MDL-dominated supports alone are "
        "compacted before independent D_cert testing; F1 tests held-out "
        "support gain, F2 tests every rule's necessity and frozen signed "
        "direct contribution, and F3 tests support-level entity-history "
        "distributional robustness; exact nonlinear global optimality and "
        "enumeration of every local optimum are not claimed"
    )
    if config.dependency_aware_mdl:
        search_payload["guarantee"] += (
            "; fixed-support coefficients remain exact TPP maximum-likelihood "
            "estimates, while standalone, exact Add/Drop/W-sign decisions and "
            "D_fit family compaction use one wallet-by-calendar two-way "
            "Godambe effective-dimension code with the structural BIC "
            "dimension as a lower floor"
        )
    if config.adaptive_gradient_racing:
        terminal_stationarity = (
            "exact family-aware Add/Drop/W-sign stationary"
            if config.terminal_add_audit == "exact"
            else (
                "complete-finite-dictionary Add block-score stationary and "
                + (
                    "exact Drop/W-sign stationary"
                    if config.effect_model == "support_additive"
                    else "exact Drop/W-sign/representation stationary"
                )
            )
        )
        search_payload["guarantee"] += (
            "; intermediate Add routes use feasible support-profiled projected "
            "Newton steps with adaptive KKT/objective stopping and may miss an "
            "exact discrete Add basin; every reportable terminal is nevertheless "
            f"exactly refitted, {terminal_stationarity}, "
            "direct/total-direction audited and independently certified"
        )
    if config.max_rules_per_support is not None:
        search_payload["guarantee"] += (
            "; stationarity is over the pre-registered interpretable support "
            f"class |S|<={config.max_rules_per_support}, and each nonempty "
            "route is conditional on its objective-basin anchor antecedent"
        )
    if uses_reduced_discovery:
        search_payload["guarantee"] += (
            "; every reported support is exactly refitted and MDL-positive on "
            "complete D_fit, while reduced-cohort discovery can still miss supports"
        )
    certification_diagnostics_by_key = {
        support_key(model.record.support): model.diagnostics
        for model in certification.models
    }
    ensemble_payload = ensemble.to_dict()
    ensemble_zero_tolerance = max(1.0e-12, 10.0 * config.solver_tolerance)
    ensemble_active = [
        support_key(support)
        for support, weight in zip(ensemble.supports, ensemble.weights, strict=True)
        if float(weight) > ensemble_zero_tolerance
    ]
    ensemble_alternatives = [
        support_key(support)
        for support, weight in zip(ensemble.supports, ensemble.weights, strict=True)
        if float(weight) <= ensemble_zero_tolerance
    ]
    ensemble_payload["active_support_count"] = len(ensemble_active)
    ensemble_payload["active_ensemble_supports"] = ensemble_active
    ensemble_payload["reliability_certified_but_ensemble_redundant_supports"] = (
        ensemble_alternatives
    )
    result_payload = {
        "schema": RESULT_SCHEMA,
        "algorithm": "CRBS-TPP",
        "claim": (
            "entity-density-ratio distributionally robust predictive "
            "early-warning rules with held-out total-contextual direction; "
            "non-causal"
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
            _record_payload(
                record,
                dataset.predicate_names,
                dataset.predicate_roles,
                optimizer.engine.basis,
                optimizer.objective,
                certification_diagnostics_by_key.get(support_key(record.support)),
            )
            for record in family
        ],
        "certification": {
            "family_size": certification.family_size,
            "certified_count": len(certification.certified),
            "selected_count": len(certification.selected),
            "selected_supports": [
                support_key(model.record.support) for model in certification.selected
            ],
            "ensemble_candidate_supports": [
                support_key(model.record.support) for model in certification.certified
            ],
            "certified_rule_counts": _rule_structure_counts(
                model.record for model in certification.certified
            ),
            "selected_rule_counts": _rule_structure_counts(
                model.record for model in certification.selected
            ),
            "empirically_dominated_certified_count": (
                len(certification.certified) - len(certification.selected)
            ),
            "all": [
                {
                    "certificate": model.certificate.to_dict(),
                    "diagnostics": model.diagnostics,
                }
                for model in certification.models
            ],
        },
        "ensemble": ensemble_payload,
    }
    atomic_json(
        run_dir / "timing.json",
        {
            "search": search_seconds,
            "full_verification": full_verification_seconds,
            "certification": certification_seconds,
            "ensemble": ensemble_seconds,
            "total": time.perf_counter() - run_started,
        },
    )
    atomic_json(run_dir / "result.json", result_payload)
    checkpoint_path.unlink(missing_ok=True)
    # A successful resume supersedes the supervised worker's prior failure
    # marker.  Leaving both result.json and failure.json makes `inspect`
    # report a completed run as failed even though the immutable result was
    # written successfully.  Keep stderr.log as historical diagnostics, but
    # clear the state marker atomically after result publication.
    (run_dir / "failure.json").unlink(missing_ok=True)
    logger.info(
        "completed certified=%d selected=%d",
        len(certification.certified),
        len(certification.selected),
    )
    canonical_result = json.loads(json.dumps(result_payload))
    optimizer.close()
    handler.close()
    logger.removeHandler(handler)
    search_logger.removeHandler(handler)
    return RunReport(
        schema=RESULT_SCHEMA,
        algorithm="CRBS-TPP",
        config_digest=config.digest,
        dataset_digest=dataset.digest,
        support_count=len(family),
        certified_count=len(certification.selected),
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
