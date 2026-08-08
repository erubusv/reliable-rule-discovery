from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from crbstpp.config import RunConfig
from crbstpp.data import Dataset, write_dataset
from crbstpp.objective import ObjectiveSpec
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import RuleIdentity, Support, hierarchy_closure
from crbstpp.search import (
    SearchDiagnostics,
    SupportOptimizer,
    support_from_key,
    support_key,
)


F0 = {
    "dynamic_predicates": True,
    "outcome_blind_predicate_construction": True,
    "direct_target_proxy_excluded_from_reported_dictionary": True,
    "strict_future_effect_required": True,
    "atomic_predicates": True,
    "primitive_event_provenance": True,
}


def _dataset(root: Path) -> Dataset:
    write_dataset(
        root,
        entities=pd.DataFrame(
            {
                "entity_id": ["fit", "cert", "test"],
                "start_time": [0, 0, 0],
                "end_time": [7, 7, 7],
                "baseline_origin": [0, 0, 0],
                "split_group": [0, 0, 0],
                "partition": [0, 1, 2],
            }
        ),
        events=pd.DataFrame(
            {
                "entity_code": [0, 0, 0, 0, 0],
                "time": [1, 1, 2, 3, 4],
                "predicate_code": [0, 0, 0, 1, 0],
                "primitive_event_id": [10, 11, 12, 13, 14],
            }
        ),
        targets=pd.DataFrame(
            {"entity_code": [0], "time": [6], "multiplicity": [1]}
        ),
        predicate_names=("A", "B"),
        likelihood="poisson",
        time_unit="tick",
        adverse_event_name="T",
        f0_contract=F0,
        provenance={"fixture": "history-marked-event"},
    )
    return Dataset.load(root)


def test_history_mark_uses_strict_t_minus_and_is_an_event_refinement(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "data")
    context = Context.make(dataset, np.asarray([0], dtype=np.int32))
    engine = ResponseEngine(dataset, lag=2, knot_count=2, cache_bytes=2**20)

    # The two A events at t=1 do not count one another.  A(t=2) sees both;
    # A(t=4) sees the three A events in [1,4).
    source = engine._history_marked_source(0, 3, 2, context)
    assert source[1].tolist() == [2, 4]
    assert engine._history_marked_source(0, 3, 3, context)[1].tolist() == [4]
    assert engine.history_count_levels(context, 0, 3) == (2, 3)

    repeat_a = RuleIdentity((0,), 0, -1, history_marks=((3, 2),))
    entities, times, spans = engine.rule_completions(context, repeat_a)
    assert entities.tolist() == [0, 0]
    assert times.tolist() == [2, 4]
    assert spans.tolist() == [0, 0]
    assert min(engine.rule_block(context, repeat_a).rows.tolist()) > 2


def test_marked_atom_participates_in_pair_and_additive_closure(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "data")
    context = Context.make(dataset, np.asarray([0], dtype=np.int32))
    engine = ResponseEngine(
        dataset,
        lag=2,
        knot_count=2,
        cache_bytes=2**20,
        effect_model="additive_hierarchy",
    )
    pair = RuleIdentity(
        (0, 1),
        1,
        1,
        hierarchical=True,
        history_marks=((3, 2), (0, 0)),
    )
    entities, times, spans = engine.rule_completions(context, pair)
    assert entities.tolist() == [0, 0]
    assert times.tolist() == [3, 4]
    assert spans.tolist() == [1, 1]

    closure = hierarchy_closure(Support.of((pair,)))
    assert closure[0].antecedent == (0,)
    assert closure[0].history_marks == ((3, 2),)
    assert closure[1].antecedent == (1,)
    assert closure[1].history_marks == ()
    matrix = engine.model_matrix(context, Support.of((pair,)))
    assert matrix.closure_dimension == 4


def test_mark_identity_roundtrip_and_mdl_code() -> None:
    plain = RuleIdentity((0, 1), 3, 1, hierarchical=True)
    marked = RuleIdentity(
        (0, 1),
        3,
        -1,
        hierarchical=True,
        history_marks=((7, 2), (0, 0)),
    )
    encoded = support_key(Support.of((marked,)))
    assert support_from_key(encoded) == Support.of((marked,))

    ordinary = ObjectiveSpec(10, 3, 2, (1, 2, 2))
    expanded = ObjectiveSpec(
        10,
        3,
        2,
        (1, 2, 2),
        history_identity_count_by_pattern=((plain.pattern_key, 5),),
    )
    assert expanded.structural_penalty(Support.of((plain,))) > ordinary.structural_penalty(
        Support.of((plain,))
    )


def test_support_conditioned_basin_keeps_singletons_and_local_high_order_maxima() -> None:
    optimizer = object.__new__(SupportOptimizer)
    optimizer.config = SimpleNamespace(search_tolerance=1.0e-9)
    optimizer._state_lock = threading.RLock()
    optimizer.diagnostics = SearchDiagnostics()
    a = RuleIdentity((0,), 0, 1)
    b = RuleIdentity((1,), 0, 1)
    ab = RuleIdentity((0, 1), 1, 1, hierarchical=True)
    ac = RuleIdentity((0, 2), 1, 1, hierarchical=True)
    abc = RuleIdentity((0, 1, 2), 1, 1, hierarchical=True)
    representatives = optimizer._conditional_score_basin_representatives(
        [(1.0, 1.0, a), (0.5, 0.5, b), (4.0, 4.0, ab), (3.0, 3.0, ac),
         (2.0, 2.0, abc)]
    )
    rules = {item[2] for item in representatives}
    assert {a, b, ab, abc}.issubset(rules)
    assert ac not in rules


def test_repeat_marks_enter_initial_route_before_root_selection() -> None:
    optimizer = object.__new__(SupportOptimizer)
    optimizer.config = SimpleNamespace(
        search_tolerance=1.0e-9,
        history_marked_events=True,
    )
    optimizer._state_lock = threading.RLock()
    optimizer.diagnostics = SearchDiagnostics()
    plain_a = RuleIdentity((0,), 0, 1)
    plain_ab = RuleIdentity((0, 1), 1, 1, hierarchical=True)
    plain_ac = RuleIdentity((0, 2), 1, 1, hierarchical=True)

    def coordinates(rule: RuleIdentity) -> tuple[RuleIdentity, ...]:
        marks = tuple((30, 2) if index == 0 else (0, 0) for index in range(rule.order))
        return (
            rule,
            RuleIdentity(
                rule.antecedent,
                rule.window,
                rule.sign,
                relation=rule.relation,
                hierarchical=rule.hierarchical,
                history_marks=marks,
            ),
        )

    optimizer._identity_coordinate_rules = coordinates
    optimizer._rank_history_marked_rule_only_proposals = (
        lambda current, identities: [
            (10.0, 10.0, rule) for rule in identities
        ]
    )
    current = SimpleNamespace(support=Support(()))
    augmented = optimizer._augment_route_with_history_marks(
        current,
        [(-1.0, 0.0, plain_a), (4.0, 4.0, plain_ab), (3.0, 3.0, plain_ac)],
        initial=True,
    )
    marked_patterns = {
        item[2].pattern_key for item in augmented if item[2].history_marks
    }
    # A repeat singleton is opened even though its plain score is negative;
    # among adjacent pair skeletons only the residual-basin maximum is opened.
    assert plain_a.pattern_key in marked_patterns
    assert plain_ab.pattern_key in marked_patterns
    assert plain_ac.pattern_key not in marked_patterns


def test_checkpoint_restores_lazy_history_identity(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "checkpoint-data")
    config = RunConfig(
        dataset=str(dataset.root),
        q_max=1,
        impact_lag=2,
        knot_count=1,
        formation_windows=(0,),
        temporal_relations=("unordered",),
        history_marked_events=True,
        history_lookback_windows=(3,),
        history_count_quantiles=(0.5,),
        pricing_devices=(),
        pricing_workers=1,
        exact_workers=1,
        cache_bytes=2**20,
        romano_wolf_resamples=1_000,
        early_warning_horizon=2,
    )
    context = Context.make(dataset, np.asarray([0], dtype=np.int32))
    source = SupportOptimizer(context, config)
    target = SupportOptimizer(context, config)
    try:
        plain = next(
            rule
            for rule in source.dictionary
            if rule.antecedent == (0,) and rule.sign == 1
        )
        marked = next(
            rule
            for rule in source._identity_coordinate_rules(plain)
            if rule.history_marks
        )
        support = Support.of((marked,))
        empty = Support(())
        source._conditional_forbidden.add(support)
        source._conditional_parent_forbidden.add((empty, support))
        source._profiled_by_antecedent[marked.pattern_key] = marked
        source._baseline_identity_priority[marked.pattern_key] = 1.0

        payload = source.checkpoint_search_state()
        profiled = payload["frozen_profiled_identities"]
        assert profiled[0]["history_marks"] == [list(marked.history_marks[0])]

        target.restore_search_state(payload)
        assert support in target._conditional_forbidden
        assert (empty, support) in target._conditional_parent_forbidden
        assert target._profiled_by_antecedent[marked.pattern_key] == marked
    finally:
        source.close()
        target.close()
