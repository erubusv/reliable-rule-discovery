from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crbstpp.data import Dataset, write_dataset
from crbstpp.config import RunConfig
from crbstpp.objective import ObjectiveSpec
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import (
    RuleIdentity,
    Support,
    hierarchy_closure,
    state_aware_temporal_patterns,
)
from crbstpp.search import SupportOptimizer, support_from_key, support_key
from crbstpp.state import active_at, transition_state_intervals


F0 = {
    "dynamic_predicates": True,
    "outcome_blind_predicate_construction": True,
    "direct_target_proxy_excluded_from_reported_dictionary": True,
    "strict_future_effect_required": True,
    "atomic_predicates": True,
    "primitive_event_provenance": True,
}


def test_transition_state_is_active_after_entry_through_exit() -> None:
    intervals = transition_state_intervals(
        np.asarray([0], dtype=np.int32),
        np.asarray([2], dtype=np.int64),
        np.asarray([20], dtype=np.int64),
        np.asarray([0], dtype=np.int32),
        np.asarray([5], dtype=np.int64),
        np.asarray([8], dtype=np.int64),
    )
    observed = active_at(
        intervals,
        np.zeros(5, dtype=np.int32),
        np.asarray([2, 3, 5, 6, 8], dtype=np.int64),
    )
    np.testing.assert_array_equal(observed, [False, True, True, False, False])


def _state_dataset(root: Path) -> Dataset:
    entities = pd.DataFrame(
        {
            "entity_id": ["fit", "cert", "test"],
            "start_time": [0, 0, 0],
            "end_time": [6, 6, 6],
            "baseline_origin": [0, 0, 0],
            "split_group": [0, 0, 0],
            "partition": [0, 1, 2],
        }
    )
    events = pd.DataFrame(
        {
            "entity_code": [0, 0, 0, 1, 1, 2, 2],
            "time": [1, 1, 2, 1, 2, 1, 2],
            "predicate_code": [0, 1, 1, 0, 1, 0, 1],
            "primitive_event_id": [10, 11, 12, 20, 21, 30, 31],
        }
    )
    targets = pd.DataFrame(
        {
            "entity_code": [0],
            "time": [4],
            "multiplicity": [1],
        }
    )
    write_dataset(
        root,
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=("source", "action", "state_source_recent_q90"),
        predicate_roles=("reported", "reported", "reported"),
        predicate_definitions=(
            {"kind": "event"},
            {"kind": "event"},
            {
                "kind": "history_state",
                "source_predicate": 0,
                "transform": "recent",
                "horizon": 3,
                "clock": "dataset_time_unit",
                "activation": "strictly_after_entry_through_state_exit",
            },
        ),
        likelihood="first_event_cloglog",
        time_unit="day",
        adverse_event_name="target",
        f0_contract=F0,
        provenance={"test": True},
    )
    return Dataset.load(root)


def _state_family_dataset(root: Path) -> Dataset:
    entities = pd.DataFrame(
        {
            "entity_id": ["fit", "cert", "test"],
            "start_time": [0, 0, 0],
            "end_time": [8, 8, 8],
            "baseline_origin": [0, 0, 0],
            "split_group": [0, 0, 0],
            "partition": [0, 1, 2],
        }
    )
    events = pd.DataFrame(
        {
            "entity_code": [0, 0, 0, 1, 1, 2, 2],
            "time": [1, 2, 4, 1, 3, 1, 3],
            "predicate_code": [0, 1, 0, 0, 0, 0, 0],
            "primitive_event_id": [10, 11, 12, 20, 21, 30, 31],
        }
    )
    targets = pd.DataFrame({"entity_code": [0], "time": [6], "multiplicity": [1]})
    definitions = (
        {"kind": "event"},
        {"kind": "event"},
        {
            "kind": "history_state",
            "source_predicate": 0,
            "transform": "recent",
            "horizon": 4,
            "clock": "dataset_time_unit",
            "activation": "strictly_after_entry_through_state_exit",
        },
        {
            "kind": "history_state",
            "source_predicate": 0,
            "transform": "recurrent",
            "horizon": 4,
            "clock": "dataset_time_unit",
            "activation": "strictly_after_entry_through_state_exit",
        },
    )
    write_dataset(
        root,
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=("source", "action", "source_recent", "source_recurrent"),
        predicate_roles=("reported",) * 4,
        predicate_definitions=definitions,
        likelihood="first_event_cloglog",
        time_unit="day",
        adverse_event_name="target",
        f0_contract=F0,
        provenance={"test": True},
    )
    return Dataset.load(root)


def test_state_is_context_at_t_minus_and_not_a_repeated_tick(tmp_path: Path) -> None:
    dataset = _state_dataset(tmp_path / "dataset")
    context = Context.make(dataset, np.array([0], dtype=np.int32))
    engine = ResponseEngine(
        dataset,
        lag=2,
        knot_count=2,
        cache_bytes=1024**2,
        effect_model="additive_hierarchy",
    )
    state_entities, state_entries, _ = engine._source(2, context)
    assert state_entities.tolist() == [0]
    assert state_entries.tolist() == [1]
    entities, times, spans = engine.completions(context, (1, 2), "unordered")
    assert entities.tolist() == [0]
    assert times.tolist() == [2]
    assert spans.tolist() == [0]


def test_additive_pair_has_shared_main_effects_and_no_masking(tmp_path: Path) -> None:
    dataset = _state_dataset(tmp_path / "dataset")
    context = Context.make(dataset, np.array([0], dtype=np.int32))
    engine = ResponseEngine(
        dataset,
        lag=2,
        knot_count=2,
        cache_bytes=1024**2,
        effect_model="additive_hierarchy",
    )
    pair = RuleIdentity((0, 1), 0, -1, hierarchical=True)
    support = Support.of((pair,))
    closure = hierarchy_closure(support)
    assert tuple(term.antecedent for term in closure) == ((0,), (1,))
    matrix = engine.model_matrix(context, support)
    assert matrix.closure_dimension == 4
    assert matrix.dimension - matrix.baseline_dimension == 6
    objective = ObjectiveSpec(
        n_entities=3,
        skeleton_count=3,
        knot_count=2,
        window_count_by_order=(1, 1, 1),
    )
    assert objective.parameter_dimension(support) == 6
    objective.penalty(support, matrix, matrix.baseline_dimension)


def test_state_grammar_is_finite_and_excludes_state_state_rules() -> None:
    patterns = state_aware_temporal_patterns((0, 1), (2, 3), 3, ("unordered",))
    assert ("unordered", (0, 1)) in patterns
    assert ("unordered", (0, 2)) in patterns
    assert ("unordered", (2, 3)) not in patterns
    assert not any(
        len(antecedent) == 3 and 2 in antecedent for _, antecedent in patterns
    )


def test_history_state_family_compresses_routes_but_reopens_exact_identities(
    tmp_path: Path,
) -> None:
    dataset = _state_family_dataset(tmp_path / "dataset")
    context = Context.make(dataset, np.array([0], dtype=np.int32))
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        early_warning_horizon=1,
        knot_count=2,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        effect_model="additive_hierarchy",
        search_mode="atomic_rashomon_frontier",
        terminal_add_audit="block_score",
        adaptive_gradient_racing=True,
        history_state_family_search=True,
        romano_wolf_resamples=1_000,
        pricing_devices=("cpu",),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=1024**2,
    )
    optimizer = SupportOptimizer(context, config)
    try:
        recent = next(
            rule
            for rule in optimizer.dictionary
            if rule.pattern_key == ("atomic", (2,)) and rule.sign == 1
        )
        recurrent = next(
            rule
            for rule in optimizer.dictionary
            if rule.pattern_key == ("atomic", (3,)) and rule.sign == 1
        )
        assert optimizer._history_pattern_family_key(recent.pattern_key) == (
            optimizer._history_pattern_family_key(recurrent.pattern_key)
        )
        representatives = optimizer._history_family_profile_representatives(
            [(1.0, 1.0, recent), (2.0, 2.0, recurrent)]
        )
        assert [item[2] for item in representatives] == [recurrent]
        reopened = optimizer._identity_coordinate_rules(recurrent)
        assert recent in reopened
        assert recurrent in reopened
        assert {rule.pattern_key for rule in reopened} == {
            ("atomic", (2,)),
            ("atomic", (3,)),
        }
    finally:
        optimizer.close()


def test_optimizer_fits_closure_matched_additive_state_rule(tmp_path: Path) -> None:
    dataset = _state_dataset(tmp_path / "dataset")
    context = Context.make(dataset, np.array([0], dtype=np.int32))
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        early_warning_horizon=1,
        knot_count=2,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        effect_model="additive_hierarchy",
        romano_wolf_resamples=1_000,
        pricing_devices=("cpu",),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=1024**2,
    )
    optimizer = SupportOptimizer(context, config)
    try:
        rule = RuleIdentity((1, 2), 0, 1, hierarchical=True)
        support = Support.of((rule,))
        empty = optimizer.fit(Support.of(()))
        conditional, warm = optimizer._conditional_matrix_and_warm(empty, support)
        assert conditional.closure == hierarchy_closure(support)
        assert warm.shape == (conditional.dimension,)
        record = optimizer.fit(support)
        assert record.matrix.closure == hierarchy_closure(record.support)
        assert record.matrix.closure_dimension == 4
        scored = optimizer._attach_rule_score(record)
        assert scored.closure_null_nll is not None
        assert scored.rule_score is not None
    finally:
        optimizer.close()


def test_support_relative_additive_has_no_automatic_closure_or_masking(
    tmp_path: Path,
) -> None:
    dataset = _state_dataset(tmp_path / "dataset_support_additive")
    context = Context.make(dataset, np.array([0], dtype=np.int32))
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        early_warning_horizon=1,
        knot_count=2,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        effect_model="support_additive",
        romano_wolf_resamples=1_000,
        pricing_devices=("cpu",),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=1024**2,
    )
    optimizer = SupportOptimizer(context, config)
    try:
        a = optimizer._rule_identity((0,), 0, 1)
        ab = optimizer._rule_identity((0, 1), 0, -1)
        support_a = Support.of((a,))
        support_ab = Support.of((a, ab))
        assert a.support_additive and ab.support_additive
        assert not a.hierarchical and not ab.hierarchical
        assert hierarchy_closure(support_ab) == ()
        assert support_from_key(support_key(support_ab)) == support_ab
        assert "|SA" in support_key(support_ab)
        assert not optimizer.engine.total_state_geometry_changed(support_a, support_ab)
        b = optimizer._rule_identity((1,), 0, 1)
        assert optimizer._add_respects_support_contract(support_a, b)

        raw_a = optimizer.engine.rule_block(context, a)
        effective_a, effective_ab = optimizer.engine.total_state_rule_blocks(
            context, support_ab
        )
        np.testing.assert_array_equal(effective_a.rows, raw_a.rows)
        np.testing.assert_allclose(effective_a.values, raw_a.values)
        assert len(effective_ab.rows) > 0

        matrix = optimizer.engine.model_matrix(context, support_ab)
        assert matrix.closure == ()
        assert matrix.closure_dimension == 0
        assert len(matrix.rule_slices) == 2
    finally:
        optimizer.close()


def test_adaptive_route_uses_common_hierarchy_objective(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = _state_dataset(tmp_path / "dataset")
    context = Context.make(dataset, np.array([0], dtype=np.int32))
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        early_warning_horizon=1,
        knot_count=2,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        effect_model="additive_hierarchy",
        search_mode="atomic_rashomon_frontier",
        adaptive_gradient_racing=True,
        romano_wolf_resamples=1_000,
        pricing_devices=("cpu",),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=1024**2,
    )
    optimizer = SupportOptimizer(context, config)
    try:
        pair = next(rule for rule in optimizer.dictionary if rule.order == 2)
        empty = optimizer.records[Support.of(())]
        original = optimizer._rank_block_identities
        calls = 0

        def counted(current, identities):
            nonlocal calls
            calls += 1
            return original(current, identities)

        monkeypatch.setattr(optimizer, "_rank_block_identities", counted)
        optimizer._force_exact_candidate_validation = False
        optimizer._rank_profiled_identities(empty, (pair,))
        assert calls == 1

        optimizer._add_rank_tables.clear()
        optimizer._force_exact_candidate_validation = True
        optimizer._rank_profiled_identities(empty, (pair,))
        assert calls == 2
    finally:
        optimizer.close()


def test_additive_safe_identity_bound_includes_hidden_closure(tmp_path: Path) -> None:
    dataset = _state_dataset(tmp_path / "dataset")
    context = Context.make(dataset, np.array([0], dtype=np.int32))
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        early_warning_horizon=1,
        knot_count=2,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        effect_model="additive_hierarchy",
        romano_wolf_resamples=1_000,
        pricing_devices=("cpu",),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=1024**2,
    )
    optimizer = SupportOptimizer(context, config)
    try:
        pair = next(rule for rule in optimizer.dictionary if rule.order == 2)
        empty = optimizer.records[Support.of(())]
        survivors = optimizer._safe_identity_survivors(empty, (pair,), 0.0)
        assert isinstance(survivors, tuple)
        trial = empty.support.add(pair)
        assert trial in optimizer._directional_upper_cache
        assert np.isfinite(optimizer._directional_upper_cache[trial])
    finally:
        optimizer.close()


def test_additive_appended_closure_is_eligible_for_exact_zero_kkt(
    tmp_path: Path,
) -> None:
    dataset = _state_dataset(tmp_path / "dataset")
    context = Context.make(dataset, np.array([0], dtype=np.int32))
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=2,
        impact_lag=2,
        early_warning_horizon=1,
        knot_count=2,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        effect_model="additive_hierarchy",
        romano_wolf_resamples=1_000,
        pricing_devices=("cpu",),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=1024**2,
    )
    optimizer = SupportOptimizer(context, config)
    try:
        pair = next(
            rule for rule in optimizer.dictionary if rule.order == 2 and rule.sign == 1
        )
        current = optimizer.records[Support.of(())]
        trial = current.support.add(pair)
        # Additive hierarchy appends the lower-order closure; it does not mask
        # or rewrite the empty parent's columns.  The appended closure+rule
        # block is therefore eligible for the exact joint zero-KKT test.
        assert not optimizer.engine.total_state_geometry_changed(current.support, trial)
        closure = hierarchy_closure(trial)
        expected = (len(closure) + 1) * config.knot_count
        gradient = np.ones(expected, dtype=np.float64)
        for index, term in enumerate(closure):
            left = index * config.knot_count
            gradient[left : left + config.knot_count] = optimizer.engine.closure_sign(
                term
            )
        gradient[-config.knot_count :] = pair.sign
        optimizer._hierarchy_kkt_gradients[
            (current.support, pair.pattern_key, pair.window)
        ] = gradient
        assert optimizer._nested_zero_block_is_kkt(current, pair)
    finally:
        optimizer.close()
