from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crbstpp.baselines.config import BaselineConfig
from crbstpp.baselines.data import LandmarkSplit, load_landmarks
from crbstpp.baselines.metrics import classification_metrics
from crbstpp.baselines.runner import prepare_baselines
from crbstpp.config import RunConfig
from crbstpp.data import Dataset
from crbstpp.ensemble import _fit_frozen_model_with_retry
from crbstpp.response import Context, ResponseEngine
from crbstpp.rule_prediction import _rule, _supports
from crbstpp.rules import EMPTY_SUPPORT, RuleIdentity, Support
from crbstpp.search import support_key


def _test_landmarks(dataset: Dataset, history: int, warning: int) -> LandmarkSplit:
    """Build only the frozen test landmarks without materializing train/cert."""
    split = np.flatnonzero(dataset.partitions == 2).astype(np.int32)
    event_count = np.bincount(dataset.event_entities, minlength=dataset.n_entities)
    event_offset = np.r_[0, np.cumsum(event_count, dtype=np.int64)]
    target_count = np.bincount(dataset.target_entities, minlength=dataset.n_entities)
    target_offset = np.r_[0, np.cumsum(target_count, dtype=np.int64)]
    entities: list[np.ndarray] = []
    times: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    scale = int(dataset.ticks_per_unit)
    horizon = int(warning) * scale
    for entity in split:
        e0, e1 = int(event_offset[entity]), int(event_offset[entity + 1])
        keep = dataset.event_predicates[e0:e1] < dataset.n_reported_predicates
        query = np.unique(
            np.r_[
                np.int64(dataset.start_times[entity]),
                dataset.event_times[e0:e1][keep],
            ]
        )
        query = query[query <= dataset.end_times[entity]]
        t0, t1 = int(target_offset[entity]), int(target_offset[entity + 1])
        target = dataset.target_times[t0:t1]
        lo = np.searchsorted(target, query, side="right")
        hi = np.searchsorted(target, query + horizon, side="right")
        entities.append(np.full(len(query), entity, dtype=np.int32))
        times.append(np.asarray(query, dtype=np.int64))
        outcomes.append(np.asarray(hi > lo, dtype=np.int8))
    n = sum(map(len, times))
    return LandmarkSplit(
        features=np.zeros((n, 0), dtype=np.float32),
        outcomes=np.concatenate(outcomes),
        entity_codes=np.concatenate(entities),
        times=np.concatenate(times),
    )


def _source_support(payload: dict[str, object]) -> Support:
    return Support.of(_rule(dict(item)) for item in payload["source_support"])


def _fit_frozen(
    run_dir: Path,
) -> tuple[
    Dataset,
    RunConfig,
    Context,
    ResponseEngine,
    object,
    tuple[Support, ...],
    list[object],
    list[object],
    tuple[RuleIdentity, ...],
    tuple[Support, ...],
    np.ndarray,
]:
    result = json.loads((run_dir / "result.json").read_text())
    config = RunConfig.from_yaml(run_dir / "config.yaml")
    dataset = Dataset.load(config.dataset)
    fit = np.flatnonzero(dataset.partitions == 0).astype(np.int32)
    cert = np.flatnonzero(dataset.partitions == 1).astype(np.int32)
    combined = Context.make(dataset, np.sort(np.r_[fit, cert]).astype(np.int32))
    test = Context.make(
        dataset, np.flatnonzero(dataset.partitions == 2).astype(np.int32)
    )
    engine = ResponseEngine(
        dataset,
        lag=config.impact_lag,
        knot_count=config.knot_count,
        baseline_time_bins=config.baseline_time_bins,
        effect_model=config.effect_model,
        cache_bytes=max(64 * 1024**2, config.cache_bytes // 8),
    )
    baseline_matrix = engine.model_matrix(combined, EMPTY_SUPPORT)
    device = (config.pricing_devices or ("cpu",))[0]
    baseline_fit = _fit_frozen_model_with_retry(
        baseline_matrix,
        likelihood=dataset.likelihood,
        tolerance=config.solver_tolerance,
        max_iter=config.solver_max_iter,
        warm_start=None,
        device=device,
    )
    supports = _supports(result)
    family = {str(item["key"]): item for item in result["family"]}
    matrices: list[object] = []
    fits: list[object] = []
    for support in supports:
        matrix = engine.model_matrix(combined, support)
        warm = np.zeros(matrix.dimension, dtype=np.float64)
        warm[: engine.baseline_dimension] = baseline_fit.coefficients
        record = family[support_key(support)]
        kernels = {
            _rule(dict(item)): np.asarray(item["kernel"], dtype=np.float64)
            for item in record["rules"]
        }
        for rule, destination in zip(support.rules, matrix.rule_slices, strict=True):
            warm[destination] = kernels[rule]
        fitted = _fit_frozen_model_with_retry(
            matrix,
            likelihood=dataset.likelihood,
            tolerance=config.solver_tolerance,
            max_iter=config.solver_max_iter,
            warm_start=warm,
            device=device,
        )
        matrices.append(matrix)
        fits.append(fitted)
    stable = json.loads((run_dir / "stable_stacking_metrics.json").read_text())
    effects_payload = result["ensemble"]["rule_effects"]
    effects = tuple(_rule(dict(item)) for item in effects_payload)
    sources = tuple(_source_support(dict(item)) for item in effects_payload)
    weights = np.asarray(stable["weights"], dtype=np.float64)
    if len(weights) != len(effects):
        raise ValueError("stored constrained weights do not match rule effects")
    return (
        dataset,
        config,
        test,
        engine,
        baseline_fit,
        supports,
        matrices,
        fits,
        effects,
        sources,
        weights,
    )


def _actual_eta_rows(
    engine: ResponseEngine,
    context: Context,
    baseline_fit: object,
    supports: tuple[Support, ...],
    matrices: list[object],
    fits: list[object],
    effects: tuple[RuleIdentity, ...],
    sources: tuple[Support, ...],
    weights: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    baseline = engine.model_metadata(EMPTY_SUPPORT)
    eta = engine.frozen_linear_predictor_at_rows(
        context, baseline, baseline_fit.coefficients, rows
    )
    support_position = {support: index for index, support in enumerate(supports)}
    for rule, source, weight in zip(effects, sources, weights, strict=True):
        if weight <= 1e-14:
            continue
        index = support_position[source]
        eta += weight * engine.frozen_contextual_rule_contribution_at_rows(
            context,
            engine.model_metadata(source),
            fits[index].coefficients,
            rule_index=source.rules.index(rule),
            rows=rows,
        )
    return eta


def _completion_counts(
    engine: ResponseEngine, context: Context, rule: RuleIdentity
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entity, time, span = engine.rule_completions(context, rule)
    keep = span <= rule.window
    entity = context.entity_codes[np.asarray(entity[keep], dtype=np.int32)]
    time = np.asarray(time[keep], dtype=np.int64)
    if not len(time):
        return entity, time, np.zeros(0, dtype=np.int64)
    order = np.lexsort((time, entity))
    entity, time = entity[order], time[order]
    boundary = np.r_[True, (entity[1:] != entity[:-1]) | (time[1:] != time[:-1])]
    first = np.flatnonzero(boundary)
    return entity[first], time[first], np.diff(np.r_[first, len(time)]).astype(np.int64)


def _discrete_metrics(
    dataset: Dataset,
    config: RunConfig,
    context: Context,
    engine: ResponseEngine,
    baseline_fit: object,
    supports: tuple[Support, ...],
    matrices: list[object],
    fits: list[object],
    effects: tuple[RuleIdentity, ...],
    sources: tuple[Support, ...],
    weights: np.ndarray,
    landmarks: LandmarkSplit,
) -> tuple[dict[str, float | int], np.ndarray]:
    n = len(landmarks.outcomes)
    cumulative = np.zeros(n, dtype=np.float64)
    local = context.entity_lookup[landmarks.entity_codes]
    support_position = {support: index for index, support in enumerate(supports)}
    completion = [_completion_counts(engine, context, rule) for rule in effects]
    for step in range(1, int(config.early_warning_horizon) + 1):
        query = landmarks.times + step
        valid = (local >= 0) & (query <= dataset.end_times[landmarks.entity_codes])
        selected = np.flatnonzero(valid)
        rows = context.offsets[local[selected]] + query[selected] - context.starts[local[selected]]
        unique, inverse = np.unique(rows, return_inverse=True)
        eta = _actual_eta_rows(
            engine, context, baseline_fit, supports, matrices, fits,
            effects, sources, weights, unique,
        )[inverse]
        # Remove completions after the landmark.  They are present in the
        # ordinary response block at t+step but are unavailable at forecast time t.
        for effect_index, (rule, source, weight) in enumerate(
            zip(effects, sources, weights, strict=True)
        ):
            if weight <= 1e-14:
                continue
            source_index = support_position[source]
            beta = fits[source_index].coefficients[
                matrices[source_index].rule_slices[source.rules.index(rule)]
            ]
            entity, time, count = completion[effect_index]
            if not len(time):
                continue
            composite = entity.astype(np.int64) * np.int64(10**12) + time
            for delta in range(1, step):
                age = step - delta
                value = float(rule.sign) * float(
                    engine.basis[:, age - 1] @ beta
                )
                key = (
                    landmarks.entity_codes[selected].astype(np.int64) * np.int64(10**12)
                    + landmarks.times[selected]
                    + delta
                )
                pos = np.searchsorted(composite, key)
                matched = pos < len(composite)
                safe = np.minimum(pos, len(composite) - 1)
                matched &= composite[safe] == key
                eta[matched] -= weight * value * count[pos[matched]]
        cumulative[selected] += np.exp(np.clip(eta, -745.0, 700.0))
    probability = -np.expm1(-cumulative)
    return classification_metrics(landmarks.outcomes, probability), probability


def _continuous_metrics(
    dataset: Dataset,
    config: RunConfig,
    context: Context,
    engine: ResponseEngine,
    baseline_fit: object,
    supports: tuple[Support, ...],
    matrices: list[object],
    fits: list[object],
    effects: tuple[RuleIdentity, ...],
    sources: tuple[Support, ...],
    weights: np.ndarray,
    landmarks: LandmarkSplit,
) -> tuple[dict[str, float | int], np.ndarray]:
    rows = np.arange(context.n_grid, dtype=np.int64)
    eta = _actual_eta_rows(
        engine, context, baseline_fit, supports, matrices, fits,
        effects, sources, weights, rows,
    )
    intensity = np.exp(np.clip(eta, -745.0, 700.0))
    exposure = np.asarray(context.baseline_row_exposure, dtype=np.float64)
    hazard = np.zeros(len(landmarks.outcomes), dtype=np.float64)
    horizon = int(config.early_warning_horizon) * int(dataset.ticks_per_unit)
    support_position = {support: index for index, support in enumerate(supports)}
    completion = [_completion_counts(engine, context, rule) for rule in effects]
    by_global = {int(code): local for local, code in enumerate(context.entity_codes)}
    for global_entity in np.unique(landmarks.entity_codes):
        positions = np.flatnonzero(landmarks.entity_codes == global_entity)
        local_entity = by_global[int(global_entity)]
        left, right = int(context.offsets[local_entity]), int(context.offsets[local_entity + 1])
        times = context.row_times[left:right]
        local_hazard = intensity[left:right] * exposure[left:right]
        prefix = np.r_[0.0, np.cumsum(local_hazard)]
        start = np.searchsorted(times, landmarks.times[positions] + 1, side="left")
        terminal = min(int(context.ends[local_entity]) + 1, 2**63 - 1)
        end_time = np.minimum(landmarks.times[positions] + horizon + 1, terminal)
        stop = np.searchsorted(times, end_time, side="left")
        hazard[positions] = prefix[stop] - prefix[start]

        # Correct only landmarks followed by an active selected-rule completion.
        for effect_index, (rule, source, weight) in enumerate(
            zip(effects, sources, weights, strict=True)
        ):
            if weight <= 1e-14:
                continue
            ce, ct, cc = completion[effect_index]
            selected_completion = ce == global_entity
            future = ct[selected_completion]
            counts = cc[selected_completion]
            if not len(future):
                continue
            lo_c = np.searchsorted(future, landmarks.times[positions], side="right")
            hi_c = np.searchsorted(future, end_time, side="left")
            affected = np.flatnonzero(hi_c > lo_c)
            if not len(affected):
                continue
            source_index = support_position[source]
            beta = fits[source_index].coefficients[
                matrices[source_index].rule_slices[source.rules.index(rule)]
            ]
            widths = np.diff(engine.continuous_edges) / float(dataset.ticks_per_unit)
            profile = float(rule.sign) * beta / widths
            for item in affected:
                destination = positions[item]
                row_lo, row_hi = int(start[item]), int(stop[item])
                adjusted = eta[left + row_lo : left + row_hi].copy()
                row_time = times[row_lo:row_hi]
                for completion_index in range(int(lo_c[item]), int(hi_c[item])):
                    age = row_time - (int(future[completion_index]) + 1)
                    active = (age >= 0) & (age < engine.lag)
                    if not np.any(active):
                        continue
                    knot = np.searchsorted(engine.continuous_edges[1:], age[active], side="right")
                    adjusted[active] -= (
                        weight * int(counts[completion_index]) * profile[knot]
                    )
                hazard[destination] = float(
                    np.sum(np.exp(np.clip(adjusted, -745.0, 700.0)) * exposure[left + row_lo : left + row_hi])
                )
    probability = -np.expm1(-hazard)
    return classification_metrics(landmarks.outcomes, probability), probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--predictions-out", type=Path)
    args = parser.parse_args()
    (
        dataset, config, test, engine, baseline_fit, supports, matrices, fits,
        effects, sources, weights,
    ) = _fit_frozen(args.run_dir)
    if args.baseline_config is not None:
        baseline_config = BaselineConfig.from_yaml(args.baseline_config)
        landmarks = load_landmarks(
            prepare_baselines(baseline_config, seed=111).landmarks_path
        )[2]
    else:
        landmarks = _test_landmarks(
            dataset, 0, int(config.early_warning_horizon)
        )
    if dataset.likelihood == "continuous_poisson":
        metrics, probability = _continuous_metrics(
            dataset, config, test, engine, baseline_fit, supports, matrices, fits,
            effects, sources, weights, landmarks,
        )
    else:
        metrics, probability = _discrete_metrics(
            dataset, config, test, engine, baseline_fit, supports, matrices, fits,
            effects, sources, weights, landmarks,
        )
    if args.predictions_out is not None:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_out,
            outcomes=landmarks.outcomes,
            probability=probability,
            entity_codes=landmarks.entity_codes,
            times=landmarks.times,
        )
    print(json.dumps({
        "run_dir": str(args.run_dir),
        "readout": "integrated_hazard_with_landmark_frozen_rule_history",
        "future_predicates_used": False,
        "test": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
