from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .baselines.config import BaselineConfig
from .baselines.data import load_landmarks
from .baselines.metrics import classification_metrics
from .baselines.runner import prepare_baselines
from .checkpoint import atomic_json
from .config import RunConfig
from .data import Dataset
from .ensemble import fit_ensemble
from .response import Context, ResponseEngine
from .rules import EMPTY_SUPPORT, RuleIdentity, Support


def _rule(payload: dict[str, object]) -> RuleIdentity:
    return RuleIdentity(
        antecedent=tuple(int(value) for value in payload["antecedent"]),
        window=int(payload["window"]),
        sign=int(payload["sign"]),
        kernel_rank=int(payload.get("kernel_rank", 0)),
        relation=str(payload.get("relation", "auto")),
        hierarchical=bool(payload.get("hierarchical", False)),
        history_marks=tuple(
            (int(mark[0]), int(mark[1]))
            for mark in payload.get("history_marks", [])
        ),
        support_additive=bool(payload.get("support_additive", False)),
    )


def _supports(payload: dict[str, object]) -> tuple[Support, ...]:
    ensemble = dict(payload.get("ensemble", {}))
    return tuple(
        Support.of(_rule(dict(rule)) for rule in support)
        for support in ensemble.get("supports", [])
    )


def _landmark_next_rows(
    context: Context,
    entity_codes: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each landmark to its first strictly-future model row."""

    entity_codes = np.asarray(entity_codes, dtype=np.int32)
    times = np.asarray(times, dtype=np.int64)
    local = context.entity_lookup[entity_codes]
    valid = local >= 0
    rows = np.full(len(times), -1, dtype=np.int64)
    if context.row_times is None:
        candidate_time = times + np.int64(1)
        valid &= candidate_time <= context.ends[np.maximum(local, 0)]
        selected = np.flatnonzero(valid)
        rows[selected] = (
            context.offsets[local[selected]]
            + candidate_time[selected]
            - context.starts[local[selected]]
        )
        return rows, valid

    for local_entity in np.unique(local[valid]):
        positions = np.flatnonzero(local == local_entity)
        left = int(context.offsets[local_entity])
        right = int(context.offsets[local_entity + 1])
        row_times = context.row_times[left:right]
        offsets = np.searchsorted(row_times, times[positions], side="right")
        present = offsets < len(row_times)
        rows[positions[present]] = left + offsets[present]
        valid[positions[~present]] = False
    return rows, valid


def evaluate_rule_model_landmarks(
    run_dir: str | Path,
    baseline_config_path: str | Path,
) -> dict[str, object]:
    """Score a frozen discovered model on the common landmark task.

    The search and certification result is immutable.  We refit only its
    already-certified identities on D_fit+D_cert, reproduce the final
    nonnegative rule stack, and use the instantaneous strictly-future
    intensity as a constant-hazard forecast over the registered horizon.
    No future predicate event enters this readout.
    """

    run_dir = Path(run_dir)
    output_path = run_dir / "landmark_metrics.json"
    if output_path.is_file():
        return json.loads(output_path.read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    run_config = RunConfig.from_yaml(run_dir / "config.yaml")
    baseline_config = BaselineConfig.from_yaml(baseline_config_path)
    dataset = Dataset.load(run_config.dataset)
    if dataset.digest != result.get("dataset_digest"):
        raise ValueError("rule result and dataset digests differ")
    baseline_dataset = Dataset.load(baseline_config.dataset)
    if baseline_dataset.digest != dataset.digest:
        raise ValueError("baseline landmarks and rule model use different datasets")
    if int(baseline_config.warning_horizon) != int(run_config.early_warning_horizon):
        raise ValueError("baseline and rule-model warning horizons differ")

    prepared = prepare_baselines(baseline_config, seed=int(result.get("seed", 111)))
    landmarks = load_landmarks(prepared.landmarks_path)[2]
    if dataset.partitions is None:
        raise ValueError("landmark evaluation requires frozen partitions")
    fit_codes = np.flatnonzero(dataset.partitions == 0).astype(np.int32)
    cert_codes = np.flatnonzero(dataset.partitions == 1).astype(np.int32)
    test_codes = np.flatnonzero(dataset.partitions == 2).astype(np.int32)
    combined = Context.make(
        dataset, np.sort(np.concatenate((fit_codes, cert_codes))).astype(np.int32)
    )
    test = Context.make(dataset, test_codes)
    supports = _supports(result)
    frozen = fit_ensemble(combined, None, supports, run_config)

    engine = ResponseEngine(
        dataset,
        lag=run_config.impact_lag,
        knot_count=run_config.knot_count,
        baseline_time_bins=run_config.baseline_time_bins,
        effect_model=run_config.effect_model,
        cache_bytes=max(64 * 1024**2, run_config.cache_bytes // 8),
    )
    baseline_matrix = engine.model_metadata(EMPTY_SUPPORT)
    support_matrices = [engine.model_metadata(support) for support in frozen.supports]
    landmark_rows, valid = _landmark_next_rows(
        test, landmarks.entity_codes, landmarks.times
    )
    unique_rows, inverse = np.unique(landmark_rows[valid], return_inverse=True)
    eta = engine.frozen_linear_predictor_at_rows(
        test,
        baseline_matrix,
        frozen.baseline_fit.coefficients,
        unique_rows,
    )
    support_index = {support: index for index, support in enumerate(frozen.supports)}
    for rule, source, weight in zip(
        frozen.rule_effects,
        frozen.rule_effect_sources,
        frozen.rule_effect_weights,
        strict=True,
    ):
        if float(weight) <= 0.0:
            continue
        source_position = support_index[source]
        rule_position = source.rules.index(rule)
        eta += float(weight) * engine.frozen_contextual_rule_contribution_at_rows(
            test,
            support_matrices[source_position],
            frozen.fits[source_position].coefficients,
            rule_index=rule_position,
            rows=unique_rows,
        )

    horizon = np.minimum(
        float(run_config.early_warning_horizon),
        np.maximum(
            0.0,
            (
                dataset.end_times[landmarks.entity_codes[valid]].astype(np.float64)
                - landmarks.times[valid].astype(np.float64)
            )
            / float(dataset.ticks_per_unit),
        ),
    )
    probability = np.zeros(len(landmarks.outcomes), dtype=np.float64)
    probability[valid] = -np.expm1(
        -horizon * np.exp(np.clip(eta[inverse], -745.0, 700.0))
    )
    metrics = classification_metrics(landmarks.outcomes, probability)
    payload: dict[str, object] = {
        "schema": "crbstpp.rule-landmark-metrics.v1",
        "dataset_digest": dataset.digest,
        "run_dir": str(run_dir),
        "warning_horizon": int(run_config.early_warning_horizon),
        "readout": "strictly_future_instantaneous_intensity_constant_hazard",
        "future_predicates_used": False,
        "fit_or_selection_changed": False,
        "valid_landmarks": int(np.count_nonzero(valid)),
        "terminal_landmarks": int(np.count_nonzero(~valid)),
        "test": metrics,
    }
    atomic_json(output_path, payload)
    engine.clear_caches()
    return payload
