from __future__ import annotations

import json
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import crbstpp.native as native_module
from crbstpp.config import RunConfig
from crbstpp.data import Dataset, write_dataset
from crbstpp.native import (
    completion_events,
    completion_window_profile,
    observed_temporal_motifs,
)
from crbstpp.objective import support_score
from crbstpp.pipeline import _validate_preassigned_partition_contract
from crbstpp.preprocess.aave import (
    AAVE_ORACLE_ADDRESSES,
    _oracle_cache_record,
    _oracle_cache_records,
    _wallet_partition,
)
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import EMPTY_SUPPORT, RuleIdentity, Support, temporal_patterns
from crbstpp.search import SupportOptimizer, support_from_key, support_key


F0 = {
    "dynamic_predicates": True,
    "outcome_blind_predicate_construction": True,
    "direct_target_proxy_excluded_from_reported_dictionary": True,
    "strict_future_effect_required": True,
    "atomic_predicates": True,
    "primitive_event_provenance": True,
}


def _tiny_temporal_dataset(root: Path, *, partitions: bool = False) -> Dataset:
    entities = pd.DataFrame(
        {
            "entity_id": ["e0", "e1", "e2"],
            "start_time": [0, 0, 0],
            "end_time": [5, 5, 5],
            "baseline_origin": [0, 0, 0],
            "split_group": [0, 0, 0],
        }
    )
    if partitions:
        entities["partition"] = [0, 1, 2]
    # e0: A->B, e1: B->A, e2: distinct primitives observed at one tick.
    events = pd.DataFrame(
        {
            "entity_code": [0, 0, 1, 1, 2, 2],
            "time": [1, 2, 1, 2, 1, 1],
            "predicate_code": [0, 1, 1, 0, 0, 1],
            "primitive_event_id": [10, 20, 30, 40, 50, 51],
        }
    )
    targets = pd.DataFrame(
        {
            "entity_code": [0, 1],
            "time": [3, 4],
            "multiplicity": [1, 1],
        }
    )
    write_dataset(
        root,
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=("A", "B", "C_never_observed"),
        likelihood="poisson",
        time_unit="tick",
        adverse_event_name="T",
        f0_contract=F0,
        provenance=(
            {"partition": {"fractions": [0.5, 0.3, 0.2], "seed": 111}}
            if partitions
            else {"fixture": "temporal-v13"}
        ),
    )
    return Dataset.load(root)


def _ordered_recovery_dataset(root: Path) -> Dataset:
    """Factorial histories in which only the strict A->B->C order is risky."""

    histories = (
        ((0, 1, 2), (1, 2, 3), 0.70),
        ((0, 1), (1, 2), 0.10),
        ((0, 2), (1, 3), 0.10),
        ((1, 2), (2, 3), 0.10),
        ((2, 1, 0), (1, 2, 3), 0.10),
        ((0, 1, 2), (1, 3, 2), 0.10),
        ((0, 1, 2), (2, 1, 3), 0.10),
        ((), (), 0.10),
    )
    entities: list[tuple[object, ...]] = []
    events: list[tuple[int, int, int, int]] = []
    targets: list[tuple[int, int, int]] = []
    per_history = 120
    code = 0
    for predicates, times, probability in histories:
        target_count = int(round(per_history * probability))
        for replicate in range(per_history):
            entities.append((f"ordered-{code:06d}", 0, 5, 0, 0))
            for position, (predicate, tick) in enumerate(
                zip(predicates, times, strict=True)
            ):
                events.append((code, tick, predicate, code * 4 + position))
            # Deterministic balanced allocation avoids a random recovery test.
            if (replicate * 7_919) % per_history < target_count:
                targets.append((code, 5, 1))
            code += 1
    write_dataset(
        root,
        entities=pd.DataFrame(
            entities,
            columns=(
                "entity_id",
                "start_time",
                "end_time",
                "baseline_origin",
                "split_group",
            ),
        ),
        events=pd.DataFrame(
            events,
            columns=(
                "entity_code",
                "time",
                "predicate_code",
                "primitive_event_id",
            ),
        ),
        targets=pd.DataFrame(
            targets, columns=("entity_code", "time", "multiplicity")
        ),
        predicate_names=("A", "B", "C"),
        likelihood="first_event_cloglog",
        time_unit="tick",
        adverse_event_name="T",
        f0_contract=F0,
        provenance={"fixture": "strict-ordered-recovery"},
    )
    return Dataset.load(root)


def test_relation_identity_and_checkpoint_key_are_unambiguous() -> None:
    unordered = RuleIdentity((0, 1), 3, 1, relation="unordered")
    ordered = RuleIdentity((1, 0), 3, -1, relation="ordered")
    support = Support.of((unordered, ordered))
    encoded = support_key(support)
    assert "|RU|" in encoded
    assert "|RO|" in encoded
    assert support_from_key(encoded) == support
    with pytest.raises(ValueError, match="positive window"):
        RuleIdentity((0, 1), 0, 1, relation="ordered")


def test_search_checkpoint_preserves_descending_ordered_patterns(
    tmp_path: Path,
) -> None:
    """A v13 checkpoint must never reinterpret B->A as unordered A AND B."""
    dataset = _tiny_temporal_dataset(tmp_path / "checkpoint-relations")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        knot_count=1,
        formation_windows=(0, 2),
        temporal_relations=("unordered", "ordered"),
        split_fractions=(0.5, 0.3, 0.2),
        early_warning_horizon=2,
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=4 * 1024**2,
        romano_wolf_resamples=1_000,
    )
    context = Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32))
    source = SupportOptimizer(context, config)
    target = SupportOptimizer(context, config)
    descending = ("ordered", (1, 0))
    try:
        source._working_antecedents.add(descending)
        source._route_root_antecedents.add(descending)
        source._standalone_report_antecedents.add(descending)
        source._standalone_identity_audited.add(descending)
        descending_rule = next(
            rule for rule in source.dictionary if rule.pattern_key == descending
        )
        rejected_child = Support.of((descending_rule,))
        source._conditional_forbidden.add(rejected_child)
        source._conditional_parent_forbidden.add(
            (EMPTY_SUPPORT, rejected_child)
        )
        source._fit_add_robust_margins[
            (EMPTY_SUPPORT, rejected_child)
        ] = -0.125
        payload = source.checkpoint_search_state()

        encoded = payload["standalone_report_antecedents"]
        assert encoded == [{"relation": "ordered", "antecedent": [1, 0]}]
        target.restore_search_state(payload)
        assert descending in target._working_antecedents
        assert descending in target._route_root_antecedents
        assert descending in target._standalone_report_antecedents
        assert descending in target._standalone_identity_audited
        assert rejected_child in target._conditional_forbidden
        assert (
            EMPTY_SUPPORT,
            rejected_child,
        ) in target._conditional_parent_forbidden
        assert target._fit_add_robust_margins[
            (EMPTY_SUPPORT, rejected_child)
        ] == pytest.approx(-0.125)
    finally:
        source.close()
        target.close()


def test_search_checkpoint_reuses_exact_ensemble_residual_route_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process shards must not refit every root to reconstruct route order."""
    dataset = _tiny_temporal_dataset(tmp_path / "checkpoint-route-order")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=1,
        impact_lag=2,
        knot_count=1,
        formation_windows=(0, 2),
        temporal_relations=("unordered",),
        split_fractions=(0.5, 0.3, 0.2),
        early_warning_horizon=2,
        ensemble_residual_search=True,
        search_mode="atomic_rashomon_frontier",
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=4 * 1024**2,
        romano_wolf_resamples=1_000,
    )
    context = Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32))
    source = SupportOptimizer(context, config)
    target = SupportOptimizer(context, config)
    try:
        rule = next(rule for rule in source.dictionary if rule.antecedent == (0,))
        support = Support.of((rule,))
        record = source._attach_rule_score(source.fit(support))
        assert record.fit.converged
        source._checkpoint_profiled_roots = (record,)
        source._checkpoint_ensemble_route_order = (support,)
        source._checkpoint_route_seeds = (record,)
        source._route_predictive_records = {support: record}
        source._rashomon_basin_by_support[support] = support
        source.ensemble_reduced_costs[support] = 1.25
        payload = source.checkpoint_search_state()
        assert "route_seed_records" not in payload

        target.restore_search_state(payload)
        assert target._checkpoint_ensemble_route_order == (support,)
        assert target.ensemble_reduced_costs == {support: 1.25}
        assert set(target._checkpoint_route_predictive_family) == {support}
        assert target._checkpoint_rashomon_basin_map == {support: support}
        monkeypatch.setattr(
            target,
            "_ensemble_residual_route_order",
            lambda _records: pytest.fail("recomputed frozen route order"),
        )
        target.search(allowed_starts=frozenset(), finalize_family=False)
    finally:
        source.close()
        target.close()


def test_ensemble_residual_orders_but_never_discards_exact_roots(
    tmp_path: Path,
) -> None:
    dataset = _tiny_temporal_dataset(tmp_path / "residual-route-family")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=1,
        impact_lag=2,
        knot_count=1,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        split_fractions=(0.5, 0.3, 0.2),
        early_warning_horizon=2,
        ensemble_residual_search=True,
        search_mode="atomic_rashomon_frontier",
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=4 * 1024**2,
        romano_wolf_resamples=1_000,
    )
    context = Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32))
    optimizer = SupportOptimizer(context, config)
    try:
        roots = tuple(
            optimizer._attach_rule_score(
                optimizer.fit(Support.of((RuleIdentity((predicate,), 0, 1),)))
            )
            for predicate in (0, 1)
        )
        assert all(record.fit.converged for record in roots)
        ordered = optimizer._ensemble_residual_route_order(roots)
        assert len(ordered) == len(roots)
        assert {record.support for record in ordered} == {
            record.support for record in roots
        }
    finally:
        optimizer.close()


def test_predictive_basin_routes_keep_predictive_and_merge_equivalent_alternative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _tiny_temporal_dataset(tmp_path / "predictive-basin-roots")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=1,
        impact_lag=2,
        knot_count=1,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        split_fractions=(0.5, 0.3, 0.2),
        early_warning_horizon=2,
        ensemble_residual_search=True,
        predictive_basin_rashomon_search=True,
        fisher_separated_rashomon=True,
        search_mode="atomic_rashomon_frontier",
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=4 * 1024**2,
        romano_wolf_resamples=1_000,
    )
    context = Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32))
    optimizer = SupportOptimizer(context, config)
    try:
        roots = tuple(
            optimizer._attach_rule_score(
                optimizer.fit(Support.of((RuleIdentity((predicate,), 0, 1),)))
            )
            for predicate in (0, 1)
        )

        def residual_order(records: tuple) -> tuple:
            optimizer._predictive_root_supports = frozenset({roots[0].support})
            optimizer._route_predictive_records = {
                roots[0].support: roots[0]
            }
            return records

        monkeypatch.setattr(optimizer, "_ensemble_residual_route_order", residual_order)
        monkeypatch.setattr(
            optimizer,
            "_fisher_effect_profile",
            lambda _record: (
                np.asarray([0], dtype=np.int64),
                np.asarray([1.0], dtype=np.float64),
            ),
        )
        routed = optimizer._predictive_basin_route_roots(roots)
        assert tuple(record.support for record in routed) == (roots[0].support,)
        assert optimizer._rashomon_basin_by_support[roots[1].support] == roots[0].support
        assert optimizer.diagnostics.predictive_basin_root_routes_avoided == 1
    finally:
        optimizer.close()


def test_intermediate_frontier_keeps_predictive_and_distinct_rashomon_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _tiny_temporal_dataset(tmp_path / "intermediate-two-frontier")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=1,
        impact_lag=2,
        knot_count=1,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        split_fractions=(0.5, 0.3, 0.2),
        early_warning_horizon=2,
        ensemble_residual_search=True,
        predictive_basin_rashomon_search=True,
        fisher_separated_rashomon=True,
        search_mode="atomic_rashomon_frontier",
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=4 * 1024**2,
        romano_wolf_resamples=1_000,
    )
    context = Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32))
    optimizer = SupportOptimizer(context, config)
    try:
        parent = optimizer.records[EMPTY_SUPPORT]
        children = tuple(
            optimizer._attach_rule_score(
                optimizer.fit(Support.of((RuleIdentity((predicate,), 0, 1),)))
            )
            for predicate in (0, 1)
        )
        monkeypatch.setattr(
            optimizer,
            "_conditional_basin_branch_additions",
            lambda _current: children,
        )

        def family_selection(records):
            family = {record.support: record for record in records}
            predictive = children[0]
            if predictive.support in family:
                return {predictive.support: predictive}, 10.0
            return {}, 0.0

        monkeypatch.setattr(optimizer, "_route_family_selection", family_selection)
        profiles = {
            parent.support: (
                np.asarray([0], dtype=np.int64),
                np.asarray([0.0], dtype=np.float64),
            ),
            children[0].support: (
                np.asarray([0], dtype=np.int64),
                np.asarray([1.0], dtype=np.float64),
            ),
            children[1].support: (
                np.asarray([0], dtype=np.int64),
                np.asarray([10.0], dtype=np.float64),
            ),
        }
        monkeypatch.setattr(
            optimizer,
            "_fisher_effect_profile",
            lambda record: profiles[record.support],
        )
        successors = optimizer._intermediate_frontier_successors(parent)
        assert tuple(record.support for record in successors) == (
            children[0].support,
            children[1].support,
        )
        assert optimizer._intermediate_frontier_labels[
            (parent.support, children[0].support)
        ][0] == "predictive_complement"
        assert optimizer._intermediate_frontier_labels[
            (parent.support, children[1].support)
        ] == ("rashomon_alternative", children[1].support)
        # Pricing a frontier must not mutate the live family before the route
        # accepts the primary edge.
        assert optimizer._route_predictive_records == {}
        assert (
            parent.support,
            children[0].support,
        ) in optimizer._intermediate_frontier_family_updates
    finally:
        optimizer.close()


def test_intermediate_pareto_prunes_before_exact_fit_and_lru_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _tiny_temporal_dataset(tmp_path / "intermediate-pareto")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=1,
        impact_lag=2,
        knot_count=1,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        split_fractions=(0.5, 0.3, 0.2),
        early_warning_horizon=2,
        predictive_basin_rashomon_search=True,
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=4 * 1024**2,
        romano_wolf_resamples=1_000,
    )
    context = Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32))
    optimizer = SupportOptimizer(context, config)
    try:
        parent = optimizer.records[EMPTY_SUPPORT]
        first = RuleIdentity((0,), 0, 1)
        second = RuleIdentity((1,), 0, 1)
        empty_profile = (
            np.asarray([0], dtype=np.int64),
            np.asarray([0.0], dtype=np.float64),
        )
        profiles = {
            first: (
                np.asarray([0], dtype=np.int64),
                np.asarray([2.0], dtype=np.float64),
            ),
            second: (
                np.asarray([0], dtype=np.int64),
                np.asarray([1.0], dtype=np.float64),
            ),
        }
        monkeypatch.setattr(
            optimizer, "_fisher_effect_profile", lambda _record: empty_profile
        )
        monkeypatch.setattr(
            optimizer,
            "_candidate_fisher_profile",
            lambda _current, rule: profiles[rule],
        )
        kept = optimizer._intermediate_pareto_representatives(
            parent,
            [(5.0, 5.0, first), (4.0, 4.0, second)],
        )
        assert [item[2] for item in kept] == [first]

        optimizer._fisher_profile_limit = 24
        support_first = Support.of((first,))
        support_second = Support.of((second,))
        optimizer._retain_effect_profile(support_first, profiles[first])
        optimizer._retain_effect_profile(support_second, profiles[second])
        assert optimizer._fisher_effect_profile_bytes <= 24
        assert tuple(optimizer._fisher_effect_profiles) == (support_second,)
    finally:
        optimizer.close()


def test_temporal_dictionary_contains_all_observable_relations() -> None:
    patterns = set(temporal_patterns(3, 3, ("unordered", "ordered")))
    assert ("unordered", (0, 1)) in patterns
    assert ("ordered", (0, 1)) in patterns
    assert ("ordered", (1, 0)) in patterns
    assert ("unordered", (0, 1, 2)) in patterns
    assert sum(pattern[0] == "ordered" and len(pattern[1]) == 3 for pattern in patterns) == 6


def test_compiled_observed_motif_frontier_is_exact_and_lazy(tmp_path: Path) -> None:
    dataset = _tiny_temporal_dataset(tmp_path / "motifs")
    masks = observed_temporal_motifs(
        dataset.event_entities,
        dataset.event_times,
        dataset.event_predicates,
        dataset.event_primitive_ids,
        predicate_count=2,
        q_max=2,
        maximum_span=2,
        allow_unordered=True,
        allow_ordered=True,
    )
    assert masks is not None
    atomic, unordered_pair, ordered_pair, _, _ = masks
    assert np.all(atomic > 0)
    assert unordered_pair[1] > 0  # canonical (0, 1)
    assert ordered_pair[1] > 0  # 0 -> 1
    assert ordered_pair[2] > 0  # 1 -> 0

    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        knot_count=1,
        formation_windows=(0, 2),
        temporal_relations=("unordered", "ordered"),
        split_fractions=(0.5, 0.3, 0.2),
        early_warning_horizon=2,
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=4 * 1024**2,
        romano_wolf_resamples=1_000,
    )
    optimizer = SupportOptimizer(
        Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32)),
        config,
    )
    try:
        assert set(optimizer.patterns) == {
            ("atomic", (0,)),
            ("atomic", (1,)),
            ("unordered", (0, 1)),
            ("ordered", (0, 1)),
            ("ordered", (1, 0)),
        }
        assert optimizer.diagnostics.theoretical_skeletons == 12
        assert optimizer.diagnostics.observed_motif_skeletons == 5
        for pattern in optimizer.patterns:
            count = optimizer._observed_completion_upper_counts[pattern]
            rows = optimizer.engine.response_rows(
                optimizer.context,
                pattern[1],
                max(optimizer.window_dictionary[pattern]),
                relation=pattern[0],
            )
            assert len(rows) <= count * config.impact_lag
            probe = Support.of(
                (
                    RuleIdentity(
                        pattern[1],
                        optimizer.window_dictionary[pattern][0],
                        1,
                        relation=pattern[0],
                    ),
                )
            )
            row_upper = min(optimizer.context.n_grid, count * config.impact_lag)
            cardinality_upper = support_score(
                baseline_nll=optimizer.baseline_nll,
                fit_nll=optimizer._cardinality_saturated_lower_nll(row_upper),
                penalty=optimizer.objective.structural_penalty(probe),
            )
            exact_upper = optimizer._representation_structure_upper_score(probe)
            assert cardinality_upper + 1.0e-10 >= exact_upper

        # The terminal Add version includes the parent's exact active-row
        # union in the cardinality budget.  Its relaxation must contain every
        # signed atomic/unordered/ordered child after the parent coefficients
        # are fully reoptimized, not merely standalone rules.
        empty = optimizer.records[Support(())]
        parent_rule = RuleIdentity((0,), 0, 1, relation="atomic")
        parent = optimizer.fit(Support.of((parent_rule,)), empty)
        assert parent.fit.converged, parent.fit.message
        for pattern in optimizer.patterns:
            if pattern == parent_rule.pattern_key:
                continue
            count = optimizer._observed_completion_upper_counts[pattern]
            for sign in (-1, 1):
                child_rule = RuleIdentity(
                    pattern[1],
                    optimizer.window_dictionary[pattern][-1],
                    sign,
                    relation=pattern[0],
                )
                child_support = parent.support.add(child_rule)
                child = optimizer.fit(child_support, parent)
                if not child.fit.converged:
                    continue
                row_upper = min(
                    optimizer.context.n_grid,
                    len(parent.matrix.active_rows) + count * config.impact_lag,
                )
                cardinality_upper = support_score(
                    baseline_nll=optimizer.baseline_nll,
                    fit_nll=optimizer._cardinality_saturated_lower_nll(row_upper),
                    penalty=optimizer.objective.structural_penalty(child_support),
                )
                slack = 1.0e-8 * max(1.0, abs(child.score))
                assert cardinality_upper + slack >= child.score
    finally:
        optimizer.close()


def test_compiled_observed_motif_scanner_matches_random_brute_force() -> None:
    generator = np.random.default_rng(173)
    predicate_count = 5
    maximum_span = 2
    rows: list[tuple[int, int, int, int]] = []
    grouped: dict[tuple[int, int], tuple[int, set[int]]] = {}
    primitive = 0
    for entity in range(8):
        for _ in range(int(generator.integers(2, 9))):
            tick = int(generator.integers(0, 6))
            attributes = set(
                map(
                    int,
                    generator.choice(
                        predicate_count,
                        size=int(generator.integers(1, 4)),
                        replace=False,
                    ),
                )
            )
            grouped[(entity, primitive)] = (tick, attributes)
            rows.extend((entity, tick, predicate, primitive) for predicate in attributes)
            primitive += 1
    rows.sort()
    entities, times, predicates, primitives = (
        np.asarray(value, dtype=np.int64) for value in zip(*rows, strict=True)
    )
    actual = observed_temporal_motifs(
        entities,
        times,
        predicates,
        primitives,
        predicate_count=predicate_count,
        q_max=3,
        maximum_span=maximum_span,
        allow_unordered=True,
        allow_ordered=True,
    )
    assert actual is not None
    expected = tuple(np.zeros_like(value) for value in actual)
    atomic, unordered_pair, ordered_pair, unordered_triplet, ordered_triplet = expected
    for (_, _), (_, attributes) in grouped.items():
        for predicate in attributes:
            atomic[predicate] = 1
    for entity in range(8):
        groups = [
            (tick, primitive_id, attributes)
            for (candidate, primitive_id), (tick, attributes) in grouped.items()
            if candidate == entity
        ]
        groups.sort()
        for selected_count in (2, 3):
            for selected in itertools.combinations(groups, selected_count):
                selected_times = [value[0] for value in selected]
                if max(selected_times) - min(selected_times) > maximum_span:
                    continue
                for values in itertools.product(*(value[2] for value in selected)):
                    if len(set(values)) != selected_count:
                        continue
                    if selected_count == 2:
                        low, high = sorted(values)
                        unordered_pair[low * predicate_count + high] = 1
                        if selected_times[0] < selected_times[1]:
                            ordered_pair[values[0] * predicate_count + values[1]] = 1
                    else:
                        canonical = sorted(values)
                        unordered_triplet[
                            (canonical[0] * predicate_count + canonical[1])
                            * predicate_count
                            + canonical[2]
                        ] = 1
                        if selected_times[0] < selected_times[1] < selected_times[2]:
                            ordered_triplet[
                                (values[0] * predicate_count + values[1])
                                * predicate_count
                                + values[2]
                            ] = 1
    for output, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(output > 0, reference > 0)


def test_ordered_completion_is_strict_and_primitive_safe() -> None:
    a = (
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([1, 2, 1], dtype=np.int64),
        np.asarray([10, 40, 50], dtype=np.int64),
    )
    b = (
        np.asarray([0, 1, 2, 2], dtype=np.int64),
        np.asarray([2, 1, 1, 3], dtype=np.int64),
        np.asarray([20, 30, 50, 60], dtype=np.int64),
    )
    result = completion_events([a, b], relation="ordered")
    assert result is not None
    entities, times, spans = result
    # e0 has A->B; e1 has only B->A; e2's coincident attributes are one
    # primitive and cannot witness a pair, while the later distinct B can.
    np.testing.assert_array_equal(entities, np.asarray([0, 2]))
    np.testing.assert_array_equal(times, np.asarray([2, 3]))
    np.testing.assert_array_equal(spans, np.asarray([1, 2]))

    c = (
        np.asarray([0], dtype=np.int64),
        np.asarray([4], dtype=np.int64),
        np.asarray([70], dtype=np.int64),
    )
    triple = completion_events([a, b, c], relation="ordered")
    assert triple is not None
    np.testing.assert_array_equal(triple[0], np.asarray([0]))
    np.testing.assert_array_equal(triple[1], np.asarray([4]))
    np.testing.assert_array_equal(triple[2], np.asarray([3]))

    compiled = tuple(value.copy() for value in triple)
    original = native_module._cpu_native
    native_module._cpu_native = None
    try:
        reference = completion_events([a, b, c], relation="ordered")
    finally:
        native_module._cpu_native = original
    assert reference is not None
    for actual, expected in zip(compiled, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_dataset_rejects_one_primitive_spanning_multiple_ticks(tmp_path: Path) -> None:
    dataset = _tiny_temporal_dataset(tmp_path / "primitive-contract")
    primitive = dataset.event_primitive_ids.copy()
    primitive[1] = primitive[0]
    broken = replace(dataset, event_primitive_ids=primitive)
    with pytest.raises(ValueError, match="cannot span ticks"):
        broken.validate()


def test_exact_fit_recovers_ordered_pair_and_triplet_direction(tmp_path: Path) -> None:
    dataset = _ordered_recovery_dataset(tmp_path / "ordered-recovery")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=3,
        impact_lag=4,
        knot_count=1,
        formation_windows=(0, 3),
        temporal_relations=("unordered", "ordered"),
        early_warning_horizon=4,
        reliability_aware_search=False,
        pricing_devices=(),
        pricing_workers=2,
        exact_workers=1,
        cache_bytes=16 * 1024**2,
        romano_wolf_resamples=1_000,
        solver_tolerance=1.0e-8,
        solver_max_iter=150,
    )
    optimizer = SupportOptimizer(
        Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32)),
        config,
    )
    try:
        empty = optimizer.records[Support(())]
        candidates = {
            "ordered_pair_excitation": RuleIdentity(
                (0, 1), 3, 1, relation="ordered"
            ),
            "ordered_triplet_excitation": RuleIdentity(
                (0, 1, 2), 3, 1, relation="ordered"
            ),
            "ordered_triplet_inhibition": RuleIdentity(
                (0, 1, 2), 3, -1, relation="ordered"
            ),
            "unordered_triplet_excitation": RuleIdentity(
                (0, 1, 2), 3, 1, relation="unordered"
            ),
        }
        fitted = {
            name: optimizer.fit(Support.of((rule,)), empty)
            for name, rule in candidates.items()
        }
        assert all(record.fit.converged for record in fitted.values())
        assert fitted["ordered_pair_excitation"].score > 0.0
        assert fitted["ordered_triplet_excitation"].score > 0.0
        assert (
            fitted["ordered_triplet_excitation"].score
            > fitted["ordered_triplet_inhibition"].score
        )
        assert (
            fitted["ordered_triplet_excitation"].score
            > fitted["unordered_triplet_excitation"].score
        )
    finally:
        optimizer.close()


def test_descending_ordered_rule_warm_start_and_sign_reuse_preserve_relation(
    tmp_path: Path,
) -> None:
    """Ordered permutations must never pass through unordered closure state."""
    dataset = _ordered_recovery_dataset(tmp_path / "ordered-warm-start")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=4,
        knot_count=1,
        formation_windows=(0, 3),
        temporal_relations=("unordered", "ordered"),
        early_warning_horizon=4,
        reliability_aware_search=False,
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=16 * 1024**2,
        romano_wolf_resamples=1_000,
    )
    optimizer = SupportOptimizer(
        Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32)),
        config,
    )
    try:
        empty = optimizer.records[Support(())]
        rules = (
            RuleIdentity((1, 0), 3, -1, relation="ordered"),
            RuleIdentity((1, 0), 3, 1, relation="ordered"),
        )
        target = optimizer.engine.model_matrix(
            optimizer.context, Support.of((rules[0],))
        )
        warm = optimizer.warm_start(empty, target)
        assert warm.shape == (target.dimension,)

        records = optimizer._fit_standalone_rules_with_sign_reuse(  # noqa: SLF001
            list(rules), empty
        )
        assert len(records) == 2
        assert all(record.support.rules[0].relation == "ordered" for record in records)
        assert all(record.support.rules[0].antecedent == (1, 0) for record in records)
    finally:
        optimizer.close()


def test_compact_completion_store_is_byte_identical_to_atomic_v2_entries(
    tmp_path: Path,
) -> None:
    dataset = _tiny_temporal_dataset(tmp_path / "compact-dataset")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        knot_count=1,
        formation_windows=(0, 2),
        temporal_relations=("unordered", "ordered"),
        reliability_aware_search=False,
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=16 * 1024**2,
        romano_wolf_resamples=1_000,
    )
    optimizer = SupportOptimizer(
        Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32)),
        config,
    )
    try:
        persistent = tmp_path / "completion-v2"
        persistent.mkdir()
        optimizer._persistent_completion_dir = persistent  # noqa: SLF001
        reference = {}
        for pattern in optimizer.patterns:
            value = optimizer._completion_for_antecedent(pattern)  # noqa: SLF001
            reference[pattern] = tuple(np.asarray(array).copy() for array in value)
        source_offsets, source_times = optimizer._implicit_sources()  # noqa: SLF001
        active_reference = {
            pattern: optimizer._implicit_entities_for_antecedent(  # noqa: SLF001
                pattern, 2, source_offsets, source_times
            ).copy()
            for pattern in optimizer.patterns
        }
        optimizer._completion_value_cache.clear()  # noqa: SLF001
        optimizer._completion_value_cache_bytes = 0  # noqa: SLF001
        optimizer.engine.clear_completion_cache()

        assert optimizer.prepare_compact_completion_store()
        assert optimizer.prepare_compact_completion_profiles()
        for pattern in optimizer.patterns:
            paths = optimizer._completion_cache_paths(pattern)  # noqa: SLF001
            assert paths is not None
            for path in paths:
                path.unlink()
        optimizer._completion_value_cache.clear()  # noqa: SLF001
        optimizer._completion_value_cache_bytes = 0  # noqa: SLF001
        optimizer._implicit_antecedent_profiles.clear()  # noqa: SLF001
        optimizer._implicit_antecedent_entities.clear()  # noqa: SLF001
        optimizer.engine.clear_completion_cache()
        for pattern in optimizer.patterns:
            actual = optimizer._completion_for_antecedent(pattern)  # noqa: SLF001
            for observed, expected in zip(actual, reference[pattern], strict=True):
                np.testing.assert_array_equal(observed, expected)
            active = optimizer._implicit_entities_for_antecedent(  # noqa: SLF001
                pattern, 2, source_offsets, source_times
            )
            np.testing.assert_array_equal(active, active_reference[pattern])

        # Distinct waves share immutable global values/token but carry their
        # own absolute entity offsets.  Reconstructing each stream must still
        # recover the exact v2 arrays byte for byte.
        first_pattern, last_pattern = optimizer.patterns[0], optimizer.patterns[-1]
        first_wave = optimizer._implicit_completion_batch((first_pattern,))  # noqa: SLF001
        last_wave = optimizer._implicit_completion_batch((last_pattern,))  # noqa: SLF001
        assert first_wave[4] == last_wave[4]
        assert np.shares_memory(first_wave[1], last_wave[1])
        for pattern, wave in ((first_pattern, first_wave), (last_pattern, last_wave)):
            row = wave[0][wave[3][pattern]]
            pieces = [
                wave[1][int(row[entity]) : int(row[entity + 1])]
                for entity in range(dataset.n_entities)
            ]
            reconstructed = np.concatenate(pieces) if pieces else np.zeros(0)
            np.testing.assert_array_equal(reconstructed, reference[pattern][1])

        specifications = tuple(
            (pattern[1], 2, (), ((pattern, 2),))
            for pattern in optimizer.patterns
        )
        packed_offsets, packed_entities = optimizer._implicit_candidate_entity_index(  # noqa: SLF001
            specifications, source_offsets, source_times
        )
        for index, pattern in enumerate(optimizer.patterns):
            np.testing.assert_array_equal(
                packed_entities[packed_offsets[index] : packed_offsets[index + 1]],
                active_reference[pattern],
            )
    finally:
        optimizer.close()


def test_atomic_subtree_bound_contains_ordered_unordered_signed_descendants(
    tmp_path: Path,
) -> None:
    dataset = _ordered_recovery_dataset(tmp_path / "ordered-bound")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=3,
        impact_lag=4,
        knot_count=1,
        formation_windows=(0, 3),
        temporal_relations=("unordered", "ordered"),
        early_warning_horizon=4,
        reliability_aware_search=False,
        pricing_devices=(),
        pricing_workers=2,
        exact_workers=1,
        cache_bytes=16 * 1024**2,
        romano_wolf_resamples=1_000,
        solver_tolerance=1.0e-8,
        solver_max_iter=150,
    )
    optimizer = SupportOptimizer(
        Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32)),
        config,
    )
    try:
        empty = optimizer.records[Support(())]
        optional = (
            ("unordered", (0, 1)),
            ("ordered", (0, 1)),
            ("ordered", (0, 1, 2)),
        )
        certificate = optimizer.atomic_subtree_upper_bound(empty, optional)
        for signs in itertools.product((0, -1, 1), repeat=len(optional)):
            support = Support(())
            for pattern, sign in zip(optional, signs, strict=True):
                if sign:
                    support = support.add(
                        RuleIdentity(
                            pattern[1], 3, sign, relation=pattern[0]
                        )
                    )
            exact = optimizer.fit(support, empty)
            if exact.fit.converged:
                assert certificate.score_upper_bound + 2.0e-6 >= exact.score
    finally:
        optimizer.close()


def test_relation_specific_window_profile_and_total_state_masking(tmp_path: Path) -> None:
    dataset = _tiny_temporal_dataset(tmp_path / "dataset")
    sources = [dataset.predicate_stream_with_ids(index) for index in (0, 1)]
    windows = np.asarray([0, 1, 2], dtype=np.int64)
    ordered = completion_window_profile(
        sources,
        dataset.end_times,
        windows,
        relation="ordered",
    )
    unordered = completion_window_profile(
        sources,
        dataset.end_times,
        windows,
        relation="unordered",
    )
    assert ordered is not None and unordered is not None
    # W0 exists only for distinct primitives coincident in the unordered
    # state; ordered completion requires strictly increasing ticks.
    assert int(ordered[0][0]) == 0
    assert int(unordered[0][0]) == 1

    context = Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32))
    engine = ResponseEngine(dataset, lag=2, knot_count=2, cache_bytes=4 * 1024**2)
    unordered_rule = RuleIdentity((0, 1), 2, 1, relation="unordered")
    ordered_rule = RuleIdentity((0, 1), 2, 1, relation="ordered")
    raw_ordered = engine.block(context, (0, 1), 2, "ordered")
    support = Support.of((unordered_rule, ordered_rule))
    by_rule = dict(
        zip(support.rules, engine.total_state_rule_blocks(context, support), strict=True)
    )
    masked_unordered, retained_ordered = by_rule[unordered_rule], by_rule[ordered_rule]
    assert len(raw_ordered.rows) > 0
    np.testing.assert_array_equal(retained_ordered.rows, raw_ordered.rows)
    assert not np.intersect1d(masked_unordered.rows, raw_ordered.rows).size
    # A completion at tick 2 affects only strictly future response ticks.
    local, response_times = context.rows_to_entity_time(raw_ordered.rows)
    assert np.all(response_times[local == 0] > 2)


def test_preassigned_split_contract_is_enforced(tmp_path: Path) -> None:
    dataset = _tiny_temporal_dataset(tmp_path / "partitioned", partitions=True)
    _validate_preassigned_partition_contract(dataset, (0.5, 0.3, 0.2), 111)
    with pytest.raises(ValueError, match="split_fractions"):
        _validate_preassigned_partition_contract(dataset, (0.6, 0.2, 0.2))
    with pytest.raises(ValueError, match="split_seed"):
        _validate_preassigned_partition_contract(dataset, (0.5, 0.3, 0.2), 112)


def test_wallet_hash_partition_uses_requested_fractions() -> None:
    values = np.asarray(
        [_wallet_partition(f"0x{index:040x}", 111, (0.5, 0.3, 0.2)) for index in range(20_000)]
    )
    shares = np.bincount(values, minlength=3) / len(values)
    np.testing.assert_allclose(shares, (0.5, 0.3, 0.2), atol=0.015)


def test_oracle_cache_is_immutable_and_digest_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_rpc(urls, method, params, **kwargs):
        calls.append((urls, method, params, kwargs))
        return "0x2a"

    monkeypatch.setattr("crbstpp.preprocess.aave._rpc", fake_rpc)
    reserve = "0x" + "12" * 20
    kwargs = {
        "version": "v3",
        "reserve": reserve,
        "tick": 3_000,
        "end_block": 30_000_000,
        "rpc_urls": ("https://rpc.invalid",),
    }
    first = _oracle_cache_record(tmp_path, **kwargs)
    second = _oracle_cache_record(tmp_path, **kwargs)
    assert first == second
    assert len(calls) == 1
    assert first["oracle_address"] == AAVE_ORACLE_ADDRESSES["v3"]
    assert first["value"] == 42
    assert first["rpc_response_digest"]

    path = tmp_path / "oracle_cache" / "v12" / "v3" / reserve / "3000.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rpc_response"] = "0x2b"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        _oracle_cache_record(tmp_path, **kwargs)


def test_oracle_batch_preserves_job_order_and_immutable_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_batch(urls, requests, **kwargs):
        requests = list(requests)
        calls.append(len(requests))
        return [hex(index + 101) for index in range(len(requests))]

    monkeypatch.setattr("crbstpp.preprocess.aave._rpc_batch", fake_batch)
    jobs = [
        ("v2", "0x" + "10" * 20, 1_600),
        ("v3", "0x" + "20" * 20, 2_400),
        ("v3", "0x" + "30" * 20, 2_401),
    ]
    first = _oracle_cache_records(
        tmp_path,
        jobs,
        rpc_urls=("https://rpc.invalid",),
        workers=2,
        end_block=30_000_000,
    )
    second = _oracle_cache_records(
        tmp_path,
        jobs,
        rpc_urls=("https://rpc.invalid",),
        workers=2,
        end_block=30_000_000,
    )
    assert calls == [3]
    assert first == second
    assert [value for *_, value in first] == [101, 102, 103]


def test_oracle_unavailable_history_is_cached_but_not_used_as_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def unavailable_batch(urls, requests, **kwargs):
        nonlocal calls
        requests = list(requests)
        calls += 1
        return [RuntimeError("execution reverted") for _ in requests]

    monkeypatch.setattr("crbstpp.preprocess.aave._rpc_batch", unavailable_batch)
    reserve = "0x" + "40" * 20
    jobs = [("v2", reserve, 1_700)]
    first = _oracle_cache_records(
        tmp_path,
        jobs,
        rpc_urls=("https://rpc.invalid",),
        workers=1,
        end_block=30_000_000,
    )
    second = _oracle_cache_records(
        tmp_path,
        jobs,
        rpc_urls=("https://rpc.invalid",),
        workers=1,
        end_block=30_000_000,
    )
    assert first == second == []
    assert calls == 1
    path = tmp_path / "oracle_cache" / "v12" / "v2" / reserve / "1700.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "unavailable"
    assert record["value"] is None
    assert record["rpc_response_digest"]


def test_current_aave_config_keeps_fixed_support_and_block_score_guarantees() -> None:
    config = RunConfig.from_yaml(
        "configs/crbstpp/aave_v37_rule_effect_search_full.yaml"
    )
    assert config.split_fractions == (0.5, 0.3, 0.2)
    assert config.temporal_relations == ("unordered",)
    assert config.formation_window_quantiles == (0.25, 0.5, 0.75, 0.9)
    assert config.terminal_add_audit == "block_score"
    assert config.adaptive_gradient_racing
    assert config.route_workers == 1
    assert not config.adaptive_kernel_mdl
    assert not config.ensemble_residual_search
    assert config.rule_effect_stacking_search
