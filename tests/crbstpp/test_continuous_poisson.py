from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crbstpp.data import Dataset, write_dataset
from crbstpp.config import RunConfig
from crbstpp.likelihood import loss_rows
from crbstpp.native import (
    continuous_additive_support_profiles,
    continuous_single_block_moments,
    continuous_single_block_profile_distances,
    dense_increment_sparse_distances,
    continuous_single_block_profiles,
    sparse_squared_distances,
)
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import RuleIdentity, Support
from crbstpp.solver import (
    fit_model_matrix,
    fit_sparse_grid_model,
    fit_sparse_grid_models_shared,
)
from crbstpp.search import SupportOptimizer


def _continuous_dataset(root: Path) -> Dataset:
    entities = pd.DataFrame(
        {
            "entity_id": ["day-0"],
            "start_time": np.asarray([0], dtype=np.int64),
            "end_time": np.asarray([39], dtype=np.int64),
            "baseline_origin": np.asarray([0], dtype=np.int64),
            "split_group": np.asarray([0], dtype=np.int64),
            "baseline_stratum": np.asarray([0], dtype=np.int16),
        }
    )
    events = pd.DataFrame(
        {
            "entity_code": np.asarray([0, 0], dtype=np.int32),
            "time": np.asarray([5, 10], dtype=np.int64),
            "predicate_code": np.asarray([0, 1], dtype=np.int16),
            "primitive_event_id": np.asarray([6, 7], dtype=np.int64),
        }
    )
    targets = pd.DataFrame(
        {
            "entity_code": np.asarray([0, 0], dtype=np.int32),
            "time": np.asarray([10, 12], dtype=np.int64),
            "multiplicity": np.asarray([1, 1], dtype=np.int32),
        }
    )
    boundaries = np.asarray(
        [0, 6, 10, 11, 12, 16, 21, 26, 31, 40], dtype=np.int64
    )
    baseline_cells = pd.DataFrame(
        {
            "entity_code": np.zeros(len(boundaries) - 1, dtype=np.int32),
            "time": boundaries[:-1],
            "baseline_stratum": np.zeros(len(boundaries) - 1, dtype=np.int16),
            "exposure": np.diff(boundaries).astype(np.float64) / 10.0,
        }
    )
    write_dataset(
        root,
        entities=entities,
        events=events,
        targets=targets,
        baseline_cells=baseline_cells,
        predicate_names=("source_a", "source_b"),
        likelihood="continuous_poisson",
        time_unit="second",
        ticks_per_unit=10,
        adverse_event_name="target",
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded_from_reported_dictionary": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
            "primitive_event_provenance": True,
        },
        provenance={"test": "exact continuous risk intervals"},
    )
    return Dataset.load(root)


def test_continuous_poisson_uses_exact_strict_future_intervals(tmp_path: Path) -> None:
    dataset = _continuous_dataset(tmp_path / "continuous")
    context = Context.make(dataset, np.asarray([0], dtype=np.int32))
    engine = ResponseEngine(
        dataset,
        lag=2,
        knot_count=2,
        cache_bytes=1 << 20,
        baseline_time_bins=1,
    )
    rule = RuleIdentity((1,), 0, 1)
    block = engine.rule_block(context, rule)
    np.testing.assert_array_equal(
        context.row_times[block.rows], [11, 12, 16, 21, 26]
    )
    np.testing.assert_allclose(
        block.values,
        np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
    )
    assert 10 not in context.row_times[block.rows]

    pair = RuleIdentity((0, 1), 1, 1, relation="unordered")
    pair_block = engine.rule_block(context, pair)
    np.testing.assert_array_equal(pair_block.rows, block.rows)
    np.testing.assert_allclose(pair_block.values, block.values)

    matrix = engine.model_matrix(context, Support((rule,)))
    fit = fit_model_matrix(
        matrix,
        likelihood=dataset.likelihood,
        tolerance=1.0e-9,
        max_iter=100,
    )
    assert fit.converged
    assert np.isfinite(fit.nll)

    sparse = fit_sparse_grid_model(
        context,
        (block,),
        (1,),
        likelihood=dataset.likelihood,
        tick_exposure=engine.tick_exposure,
        tolerance=1.0e-9,
        max_iter=100,
        baseline_group_count=engine.free_baseline_dimension,
        baseline_time_bins=1,
    )
    assert sparse.converged, sparse.message
    np.testing.assert_allclose(sparse.nll, fit.nll, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        sparse.coefficients, fit.coefficients, rtol=2e-9, atol=2e-9
    )

    shared = fit_sparse_grid_models_shared(
        context,
        ((block,), (pair_block,)),
        ((1,), (1,)),
        likelihood=dataset.likelihood,
        tick_exposure=engine.tick_exposure,
        tolerance=1.0e-9,
        max_iter=100,
        baseline_group_count=engine.free_baseline_dimension,
        baseline_time_bins=1,
        devices=("cpu",),
        max_workers=2,
    )
    for shared_fit in shared:
        assert shared_fit.converged, shared_fit.message
        np.testing.assert_allclose(shared_fit.nll, fit.nll, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(
            shared_fit.coefficients, fit.coefficients, rtol=2e-9, atol=2e-9
        )


def test_continuous_context_accepts_exact_terminal_boundary(tmp_path: Path) -> None:
    dataset = _continuous_dataset(tmp_path / "continuous-terminal")
    context = Context.make(dataset, np.asarray([0], dtype=np.int32))
    position = context.interval_positions(
        np.asarray([0], dtype=np.int32),
        np.asarray([dataset.end_times[0] + 1], dtype=np.int64),
        allow_right_endpoint=True,
    )
    np.testing.assert_array_equal(position, [context.n_grid])


def test_continuous_response_thresholds_drop_terminal_completions(
    tmp_path: Path,
) -> None:
    dataset = _continuous_dataset(tmp_path / "continuous-terminal-completion")
    context = Context.make(dataset, np.asarray([0], dtype=np.int32))
    engine = ResponseEngine(
        dataset,
        lag=2,
        knot_count=2,
        cache_bytes=1 << 20,
        baseline_time_bins=1,
    )
    rows, minimum_spans = engine._continuous_response_thresholds(
        context,
        np.asarray([0, 0], dtype=np.int32),
        np.asarray([10, dataset.end_times[0]], dtype=np.int64),
        np.asarray([2, 7], dtype=np.int64),
        maximum_span=10,
        horizon_ticks=20,
    )
    expected = engine._continuous_footprint_rows(
        context,
        np.asarray([0], dtype=np.int32),
        np.asarray([10], dtype=np.int64),
        horizon_ticks=20,
    )
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_array_equal(
        minimum_spans,
        np.full(len(expected), 2, dtype=np.int64),
    )


def test_interval_native_moments_match_materialized_block(tmp_path: Path) -> None:
    dataset = _continuous_dataset(tmp_path / "continuous-native")
    context = Context.make(dataset, np.asarray([0], dtype=np.int32))
    engine = ResponseEngine(
        dataset,
        lag=2,
        knot_count=2,
        cache_bytes=1 << 20,
        baseline_time_bins=1,
    )
    block = engine.rule_block(context, RuleIdentity((1,), 0, 1))
    first = np.linspace(-0.7, 0.9, context.n_grid, dtype=np.float64)
    second = np.linspace(0.2, 1.1, context.n_grid, dtype=np.float64)
    groups = np.asarray([0, 0, 1, 1, 0, 1, 1, 0, 0], dtype=np.int32)
    current_x = np.asarray([[1.0, 0.0, 0.4], [0.0, 1.0, -0.2]])
    columns = np.asarray([0, 2], dtype=np.int32)
    result = continuous_single_block_moments(
        np.asarray([0], dtype=np.int32),
        np.asarray([10], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        context.ends,
        context.offsets,
        context.row_times,
        engine.continuous_edges,
        dataset.ticks_per_unit / np.diff(engine.continuous_edges),
        first,
        second,
        groups,
        current_x,
        columns,
        workers=1,
    )
    assert result is not None
    gradient, hessian, cross = result
    reference_gradient = block.values.T @ first[block.rows]
    reference_hessian = block.values.T @ (
        second[block.rows, None] * block.values
    )
    reference_cross = current_x[groups[block.rows]][:, columns].T @ (
        second[block.rows, None] * block.values
    )
    np.testing.assert_allclose(gradient[0], reference_gradient, atol=1.0e-13)
    np.testing.assert_allclose(hessian[0], reference_hessian, atol=1.0e-13)
    np.testing.assert_allclose(cross[0], reference_cross, atol=1.0e-13)

    coefficients = np.asarray([[0.3, -0.2]], dtype=np.float64)
    profiles = continuous_single_block_profiles(
        np.asarray([0], dtype=np.int32),
        np.asarray([10], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        context.ends,
        context.offsets,
        context.row_times,
        engine.continuous_edges,
        dataset.ticks_per_unit / np.diff(engine.continuous_edges),
        coefficients,
        workers=1,
    )
    assert profiles is not None
    expected_profile = np.zeros(context.n_grid, dtype=np.float64)
    expected_profile[block.rows] = block.values @ coefficients[0]
    np.testing.assert_allclose(profiles[0], expected_profile, atol=1.0e-13)

    component_coefficients = np.asarray(
        [[0.3, -0.2], [-0.1, 0.4], [0.25, 0.15]], dtype=np.float64
    )
    component_profiles = continuous_single_block_profiles(
        np.asarray([0], dtype=np.int32),
        np.asarray([10], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.ones(3, dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        context.ends,
        context.offsets,
        context.row_times,
        engine.continuous_edges,
        dataset.ticks_per_unit / np.diff(engine.continuous_edges),
        component_coefficients,
        workers=1,
    )
    additive_profiles = continuous_additive_support_profiles(
        np.asarray([0], dtype=np.int32),
        np.asarray([10], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.ones(3, dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.asarray([0, 2, 3], dtype=np.int64),
        context.ends,
        context.offsets,
        context.row_times,
        engine.continuous_edges,
        dataset.ticks_per_unit / np.diff(engine.continuous_edges),
        component_coefficients,
        workers=1,
    )
    assert component_profiles is not None and additive_profiles is not None
    np.testing.assert_array_equal(
        additive_profiles,
        np.vstack(
            (
                component_profiles[0] + component_profiles[1],
                component_profiles[2],
            )
        ),
    )


def test_continuous_poisson_likelihood_matches_poisson() -> None:
    eta = np.asarray([-1.0, 0.2, 1.3])
    exposure = np.asarray([0.1, 2.5, 0.7])
    event = np.asarray([0.0, 2.0, 1.0])
    expected = loss_rows(
        eta,
        likelihood="poisson",
        exposure_weight=exposure,
        noevent_weight=exposure,
        event_weight=event,
    )
    actual = loss_rows(
        eta,
        likelihood="continuous_poisson",
        exposure_weight=exposure,
        noevent_weight=exposure,
        event_weight=event,
    )
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right)


def test_continuous_dictionary_keeps_pair_search_separate(tmp_path: Path) -> None:
    dataset = _continuous_dataset(tmp_path / "continuous-search")
    context = Context.make(dataset, np.asarray([0], dtype=np.int32))
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        knot_count=2,
        formation_windows=(0, 1),
        effect_model="support_additive",
        early_warning_horizon=1,
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=1 << 20,
        history_marked_events=False,
        dependency_aware_mdl=False,
        frequency_effect_separation=False,
        search_mode="atomic_rashomon_frontier",
        terminal_add_audit="block_score",
    )
    optimizer = SupportOptimizer(context, config)
    try:
        assert optimizer.records[Support(())].fit.converged
        assert any(rule.antecedent == (0, 1) for rule in optimizer.dictionary)
    finally:
        optimizer.close()


def test_interval_native_candidate_novelty_matches_scalar_profile(
    tmp_path: Path,
) -> None:
    dataset = _continuous_dataset(tmp_path / "continuous-novelty")
    context = Context.make(dataset, np.asarray([0], dtype=np.int32))
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        knot_count=2,
        formation_windows=(0, 1),
        effect_model="support_additive",
        early_warning_horizon=1,
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=1 << 20,
        history_marked_events=False,
        dependency_aware_mdl=False,
        frequency_effect_separation=False,
        search_mode="atomic_rashomon_frontier",
        terminal_add_audit="block_score",
        predictive_basin_rashomon_search=True,
    )
    optimizer = SupportOptimizer(context, config)
    try:
        # Production contexts mmap the immutable v3 completion pack.  Tiny
        # tests deliberately disable filesystem caches, so assemble the same
        # canonical slices in memory to exercise the batched path.
        parts = tuple(
            optimizer._completion_for_antecedent(pattern)
            for pattern in optimizer.patterns
        )
        lengths = np.asarray([len(part[0]) for part in parts], dtype=np.int64)
        offsets = np.empty(len(parts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        optimizer._compact_completion_store = (
            offsets,
            np.ascontiguousarray(
                np.concatenate([part[0] for part in parts]), dtype=np.int32
            ),
            np.ascontiguousarray(
                np.concatenate([part[1] for part in parts]), dtype=np.int64
            ),
            np.ascontiguousarray(
                np.concatenate([part[2] for part in parts]), dtype=np.int64
            ),
        )
        current = optimizer.records[Support(())]
        assert current.fit.converged
        ranked = optimizer._rank_profiled_identities(
            current, optimizer.dictionary
        )
        items = [(net, gain, rule) for net, gain, rule in ranked[:4]]
        reference = optimizer._fisher_effect_profile(current)
        observed = optimizer._continuous_candidate_fisher_novelties(
            current, items, [reference]
        )
        assert observed is not None
        uppers = optimizer._continuous_candidate_novelty_uppers(current, items)
        scalar_novelties = {}
        for _, _, rule in items:
            scalar = optimizer._candidate_fisher_profile(current, rule)
            if scalar is None:
                assert observed.get(rule) is None
            elif observed.get(rule) is None:
                # A numerically empty native wave deliberately takes the
                # ordinary scalar fail-open path in production.
                continue
            else:
                expected = optimizer._sparse_fisher_distance(scalar, reference)
                scalar_novelties[rule] = expected
                if np.isfinite(uppers[rule]):
                    assert expected <= uppers[rule] + 1.0e-12
                np.testing.assert_allclose(
                    observed[rule], expected, rtol=2.0e-12, atol=2.0e-12
                )

        tolerance = config.search_tolerance
        expected_pareto = []
        for item in items:
            novelty = scalar_novelties.get(item[2])
            if novelty is None:
                expected_pareto.append(item)
                continue
            dominated = False
            for other in items:
                if other[2] == item[2]:
                    continue
                other_novelty = scalar_novelties.get(other[2])
                if other_novelty is None:
                    continue
                if (
                    other[0] + tolerance >= item[0]
                    and other_novelty + tolerance >= novelty
                    and (
                        other[0] > item[0] + tolerance
                        or other_novelty > novelty + tolerance
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                expected_pareto.append(item)
        expected_pareto.sort(key=lambda item: (-item[0], item[2]))
        actual_pareto = optimizer._intermediate_pareto_representatives(
            current, items
        )
        assert [item[2] for item in actual_pareto] == [
            item[2] for item in expected_pareto
        ]

        root_records_list = []
        for _, _, rule in ranked:
            record = optimizer._attach_rule_score(
                optimizer.fit(Support((rule,)))
            )
            if record.fit.converged:
                root_records_list.append(record)
            if len(root_records_list) == 2:
                break
        root_records = tuple(root_records_list)
        assert root_records
        root_expected = {
            record.support: optimizer._fisher_effect_profile(record)
            for record in root_records
        }
        root_observed = optimizer._continuous_basin_profiles_for_supports(
            tuple(record.support for record in root_records)
        )
        assert root_observed is not None
        for support, (expected_rows, expected_values) in root_expected.items():
            observed_rows, observed_values = root_observed[support]
            np.testing.assert_array_equal(observed_rows, expected_rows)
            np.testing.assert_allclose(
                observed_values,
                expected_values,
                rtol=2.0e-12,
                atol=2.0e-12,
            )

        if len(root_records) == 2:
            combined_support = Support.of(
                rule
                for record in root_records
                for rule in record.support.rules
            )
            combined = optimizer._attach_rule_score(
                optimizer.fit(combined_support)
            )
            assert combined.fit.converged
            combined_expected = optimizer._fisher_effect_profile(combined)
            # Remove the scalar cache entry so the compiled additive path must
            # reconstruct this multi-rule support from its fitted components.
            optimizer._rashomon_basin_profiles.pop(combined_support, None)
            combined_observed = optimizer._continuous_basin_profiles_for_supports(
                (combined_support,)
            )
            assert combined_observed is not None
            observed_rows, observed_values = combined_observed[combined_support]
            np.testing.assert_array_equal(observed_rows, combined_expected[0])
            np.testing.assert_allclose(
                observed_values,
                combined_expected[1],
                rtol=2.0e-12,
                atol=2.0e-12,
            )
    finally:
        optimizer.close()



def test_native_sparse_squared_distances_match_union_reference() -> None:
    left = (
        (np.asarray([0, 3, 8]), np.asarray([1.0, -2.0, 0.5])),
        (np.asarray([2, 3]), np.asarray([4.0, 1.5])),
    )
    right = (
        (np.asarray([0, 2, 8]), np.asarray([-1.0, 3.0, 0.25])),
        (np.asarray([3, 9]), np.asarray([-0.5, 2.0])),
    )
    observed = sparse_squared_distances(left, right, workers=4)
    assert observed is not None
    expected = np.asarray(
        [
            [SupportOptimizer._sparse_fisher_distance(a, b) for b in right]
            for a in left
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=1.0e-14, atol=1.0e-14)



def test_native_dense_increment_sparse_distances_match_materialized_profiles() -> None:
    rng = np.random.default_rng(41)
    row_count = 20_000
    tolerance = 1.0e-8
    dense = np.zeros((12, row_count), dtype=np.float64)
    for candidate in range(len(dense)):
        rows = np.sort(rng.choice(row_count, size=1_200, replace=False))
        dense[candidate, rows] = rng.normal(size=len(rows))
    current_rows = np.sort(
        rng.choice(row_count, size=2_600, replace=False)
    ).astype(np.int64)
    current_values = rng.normal(size=len(current_rows))
    references = []
    for _ in range(7):
        rows = np.sort(
            rng.choice(row_count, size=3_000, replace=False)
        ).astype(np.int64)
        references.append((rows, rng.normal(size=len(rows))))
    observed = dense_increment_sparse_distances(
        dense,
        (current_rows, current_values),
        tuple(references),
        tolerance=tolerance,
        workers=12,
    )
    assert observed is not None

    materialized = []
    for increment in dense:
        active = np.isfinite(increment) & (np.abs(increment) > tolerance)
        increment_rows = np.flatnonzero(active).astype(np.int64)
        rows = np.union1d(current_rows, increment_rows)
        values = np.zeros(len(rows), dtype=np.float64)
        values[np.searchsorted(rows, current_rows)] += current_values
        values[np.searchsorted(rows, increment_rows)] += increment[active]
        keep = np.abs(values) > tolerance
        materialized.append(
            (
                np.ascontiguousarray(rows[keep], dtype=np.int64),
                np.ascontiguousarray(values[keep], dtype=np.float64),
            )
        )
    expected = sparse_squared_distances(
        tuple(materialized), tuple(references), workers=12
    )
    assert expected is not None
    np.testing.assert_array_equal(observed, expected)



def test_native_fused_continuous_profile_distances_match_reference() -> None:
    entities = np.asarray([0, 0, 1], dtype=np.int32)
    times = np.asarray([0, 2, 1], dtype=np.int64)
    spans = np.zeros(3, dtype=np.int64)
    candidate_starts = np.asarray([0, 1], dtype=np.int64)
    candidate_ends = np.asarray([3, 3], dtype=np.int64)
    candidate_windows = np.asarray([10, 10], dtype=np.int64)
    entity_ends = np.asarray([6, 5], dtype=np.int64)
    grid_offsets = np.asarray([0, 7, 13], dtype=np.int64)
    row_times = np.concatenate([np.arange(7), np.arange(6)]).astype(np.int64)
    knot_edges = np.asarray([0, 2, 4], dtype=np.int64)
    knot_scales = np.asarray([0.5, 0.5], dtype=np.float64)
    coefficients = np.asarray([[1.2, 0.3], [-0.7, 0.8]], dtype=np.float64)
    sqrt_fisher = np.linspace(0.5, 1.5, len(row_times))
    current = (
        np.asarray([1, 3, 8, 11], dtype=np.int64),
        np.asarray([0.2, -0.1, 0.5, 0.3], dtype=np.float64),
    )
    references = (
        current,
        (
            np.asarray([0, 3, 5, 9], dtype=np.int64),
            np.asarray([0.4, 0.2, -0.3, 0.1], dtype=np.float64),
        ),
        (
            np.asarray([1, 8, 12], dtype=np.int64),
            np.asarray([0.2, 0.7, -0.4], dtype=np.float64),
        ),
    )
    profiles = continuous_single_block_profiles(
        entities,
        times,
        spans,
        candidate_starts,
        candidate_ends,
        candidate_windows,
        entity_ends,
        grid_offsets,
        row_times,
        knot_edges,
        knot_scales,
        coefficients,
        workers=2,
    )
    assert profiles is not None
    profiles *= sqrt_fisher[None, :]
    expected = dense_increment_sparse_distances(
        profiles, current, references, tolerance=1.0e-12, workers=2
    )
    observed = continuous_single_block_profile_distances(
        entities,
        times,
        spans,
        candidate_starts,
        candidate_ends,
        candidate_windows,
        entity_ends,
        grid_offsets,
        row_times,
        knot_edges,
        knot_scales,
        coefficients,
        sqrt_fisher,
        current,
        references,
        tolerance=1.0e-12,
        workers=2,
    )
    assert expected is not None
    assert observed is not None
    np.testing.assert_array_equal(observed[:, 0], np.min(expected, axis=1))
