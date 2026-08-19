from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from crbstpp.data import Dataset, write_dataset
from crbstpp.preprocess.wselob import (
    _month_stratified_partition,
    ASK_DOMINANCE_STARTS,
    BEST_BID_ADD,
    BID_DOMINANCE_STARTS,
    BID_QUEUE_CLEARED,
    PASSIVE_BID_FILL,
    PREDICATES,
    PREDICATE_MEANINGS,
    SPREAD_WIDENS,
    MECHANISM_PREDICATES,
    MECHANISM_PREDICATE_MEANINGS,
    MECH_BEST_BID_ADD,
    MECH_BID_FILL_NONCLEAR,
    MECH_CANCEL_BID_CLEAR,
    MECH_BID_REPLENISH,
    MECH_TRADE_BID_CLEAR,
    BALANCED_MECHANISM_PREDICATES,
    BALANCED_MECHANISM_PREDICATE_MEANINGS,
    REGIME_MECHANISM_PREDICATES,
    REGIME_MECHANISM_PREDICATE_MEANINGS,
    BAL_LIQUIDITY_ADD_WORSEN,
    BAL_CANCEL_CLEAR,
    BAL_QUEUE_REPLENISH,
    _adverse_excursion_targets,
    _duration_weighted_quantile,
    _continuous_risk_frame,
    _lob_regime_transition_events,
    _lob_market_state_transition_events,
    _market_stress_scores,
    _ordered_partition,
    _process_day,
    _process_day_mechanisms,
    _realized_volatility,
    _volatility_burst_targets,
)
from crbstpp.response import Context, ResponseEngine


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Reconstruction-only book snapshot.
            (-1, 99, 10, 1, "2", "Y"),
            (0, 100, 10, 1, "2", "Y"),
            (1, 102, 10, 2, "2", "Y"),
            # Best bid replenishment starts bid dominance.
            (10, 100, 20, 1, "2", "A"),
            # A partial fill reverses depth dominance.
            (20, 100, 5, 1, "2", "M"),
            # Removing the remaining best bid retreats the quote and target.
            (30, 100, 0, 1, "2", "D"),
        ],
        columns=["time", "price", "agg_volume", "side", "order_type", "action_type"],
    )


def test_wselob_day_preserves_primitive_identity_and_strict_target_time() -> None:
    events, targets, _ = _process_day(
        _orders(),
        entity_code=3,
        session_start_ns=10,
        session_end_ns=39,
        bin_nanoseconds=10,
    )
    by_tick = {(tick, predicate): primitive for _, tick, predicate, primitive in events}
    assert (0, BEST_BID_ADD) in by_tick
    assert (0, BID_DOMINANCE_STARTS) in by_tick
    assert by_tick[(0, BEST_BID_ADD)] == by_tick[(0, BID_DOMINANCE_STARTS)]
    assert (1, PASSIVE_BID_FILL) in by_tick
    assert (1, ASK_DOMINANCE_STARTS) in by_tick
    assert (2, BID_QUEUE_CLEARED) in by_tick
    assert (2, SPREAD_WIDENS) in by_tick
    assert targets == [2]


def test_wselob_partition_is_ordered_and_complete() -> None:
    partition = _ordered_partition(10, (0.5, 0.3, 0.2))
    np.testing.assert_array_equal(partition, [0, 0, 0, 0, 0, 1, 1, 1, 2, 2])


def test_wselob_month_stratified_partition_is_seeded_and_month_complete() -> None:
    dates = tuple(pd.date_range("2017-01-02", periods=10, freq="B")) + tuple(
        pd.date_range("2017-02-01", periods=10, freq="B")
    )
    first = _month_stratified_partition(dates, (0.5, 0.3, 0.2), seed=111)
    second = _month_stratified_partition(dates, (0.5, 0.3, 0.2), seed=111)
    np.testing.assert_array_equal(first, second)
    for indices in (np.arange(10), np.arange(10, 20)):
        np.testing.assert_array_equal(
            np.bincount(first[indices], minlength=3),
            np.asarray([5, 3, 2]),
        )


def test_wselob_month_stratified_partition_preserves_global_fractions() -> None:
    dates = tuple(pd.date_range("2017-01-02", periods=19, freq="B")) + tuple(
        pd.date_range("2017-02-01", periods=17, freq="B")
    )
    partition = _month_stratified_partition(dates, (0.5, 0.3, 0.2), seed=111)
    expected = np.bincount(_ordered_partition(len(dates), (0.5, 0.3, 0.2)))
    np.testing.assert_array_equal(np.bincount(partition), expected)
    for indices in (np.arange(19), np.arange(19, 36)):
        assert set(partition[indices].tolist()) == {0, 1, 2}


def test_wselob_continuous_day_keeps_raw_exchange_timestamps() -> None:
    events, targets, _ = _process_day(
        _orders(),
        entity_code=3,
        session_start_ns=10,
        session_end_ns=39,
        bin_nanoseconds=10,
        continuous=True,
    )
    assert {time for _, time, _, _ in events} == {10, 20, 30}
    assert targets == [30]


def test_wselob_predicate_catalog_is_complete() -> None:
    assert len(PREDICATES) == len(PREDICATE_MEANINGS) == 16
    assert len(set(PREDICATES)) == len(PREDICATES)


def test_wselob_mechanism_schema_is_exclusive_and_restores_liquidity() -> None:
    orders = pd.concat(
        [
            _orders(),
            pd.DataFrame(
                [(35, 100, 12, 1, "2", "A")],
                columns=_orders().columns,
            ),
        ],
        ignore_index=True,
    ).sort_values("time", kind="stable")
    events, _, _, _ = _process_day_mechanisms(
        orders,
        entity_code=3,
        session_start_ns=10,
        session_end_ns=39,
        bin_nanoseconds=10,
        continuous=True,
        replenishment_lookback_ns=10,
    )
    predicates = {tick: predicate for _, tick, predicate, _ in events}
    assert predicates[10] == MECH_BEST_BID_ADD
    assert predicates[20] == MECH_BID_FILL_NONCLEAR
    assert predicates[30] == MECH_CANCEL_BID_CLEAR
    assert predicates[35] == MECH_BID_REPLENISH
    assert len(events) == 4
    assert len(MECHANISM_PREDICATES) == len(MECHANISM_PREDICATE_MEANINGS) == 16
    trade_events, _, _, _ = _process_day_mechanisms(
        orders,
        entity_code=3,
        session_start_ns=10,
        session_end_ns=39,
        bin_nanoseconds=10,
        continuous=True,
        replenishment_lookback_ns=10,
        trade_times=frozenset((30,)),
    )
    trade_predicates = {tick: predicate for _, tick, predicate, _ in trade_events}
    assert trade_predicates[30] == MECH_TRADE_BID_CLEAR


def test_wselob_balanced_schema_is_direction_free_and_mechanically_symmetric() -> None:
    orders = pd.concat(
        [
            _orders(),
            pd.DataFrame(
                [(35, 100, 12, 1, "2", "A")],
                columns=_orders().columns,
            ),
        ],
        ignore_index=True,
    ).sort_values("time", kind="stable")
    events, _, _, _ = _process_day_mechanisms(
        orders,
        entity_code=3,
        session_start_ns=10,
        session_end_ns=39,
        bin_nanoseconds=10,
        continuous=True,
        replenishment_lookback_ns=10,
        balanced_mechanisms=True,
    )
    predicates = {tick: predicate for _, tick, predicate, _ in events}
    assert predicates[10] == BAL_LIQUIDITY_ADD_WORSEN
    assert 20 not in predicates
    assert predicates[30] == BAL_CANCEL_CLEAR
    assert predicates[35] == BAL_QUEUE_REPLENISH
    assert len(events) == 3
    assert len(BALANCED_MECHANISM_PREDICATES) == len(
        BALANCED_MECHANISM_PREDICATE_MEANINGS
    ) == 11


def test_wselob_regime_transitions_are_symmetric_and_nonduplicated() -> None:
    payload = {
        "entity_code": 2,
        "context_rows": [
            (10, 101, 0.10, 0.10, 0.0, 0.0, 1.0),
            (20, 102, 0.90, 0.90, 0.0, 0.0, 1.0),
            (30, 103, 0.40, 0.90, 0.0, 0.0, 1.0),
        ],
    }
    states = [
        {
            "column": 2,
            "direction": "high",
            "entry_threshold": 0.75,
            "exit_threshold": 0.50,
        },
        {
            "column": 3,
            "transform": "absolute",
            "direction": "high",
            "entry_threshold": 0.75,
            "exit_threshold": 0.50,
        },
    ]
    events = _lob_regime_transition_events(
        payload,
        states,
        first_transition_predicate=len(BALANCED_MECHANISM_PREDICATES),
    )
    assert events == [
        (2, 20, 11, 102),
        (2, 30, 12, 103),
    ]
    assert len({event[3] for event in events}) == len(events)
    assert len(REGIME_MECHANISM_PREDICATES) == len(
        REGIME_MECHANISM_PREDICATE_MEANINGS
    ) == 13


def test_wselob_aggregate_market_states_are_mutually_exclusive() -> None:
    rows = [
        (10, 101, 0.10, 0.10, 0.10, 0.10, 0.10),
        (20, 102, 0.90, 0.90, 0.90, 0.90, 0.90),
        (30, 103, 0.10, 0.10, 0.10, 0.10, 0.10),
    ]
    profile = {
        channel: {
            "transform": transform,
            "values": [0.0, 1.0],
            "probabilities": [0.0, 1.0],
        }
        for channel, _, transform in (
            ("relative_spread", 2, "identity"),
            ("abs_depth_imbalance", 3, "absolute"),
            ("cancel_fraction_30s", 4, "identity"),
            ("trade_fraction_30s", 5, "identity"),
            ("message_rate_30s", 6, "identity"),
        )
    }
    np.testing.assert_allclose(_market_stress_scores(rows, profile), [0.1, 0.9, 0.1])
    states = [
        {
            "direction": "high",
            "entry_threshold": 0.75,
            "exit_threshold": 0.50,
        },
        {
            "direction": "low",
            "entry_threshold": 0.25,
            "exit_threshold": 0.50,
        },
    ]
    events = _lob_market_state_transition_events(
        {"entity_code": 2, "context_rows": rows},
        states,
        profile,
        first_source_predicate=13,
    )
    assert events == [
        (2, 10, 15, 101),
        (2, 20, 13, 102),
        (2, 20, 16, 102),
        (2, 30, 14, 103),
        (2, 30, 15, 103),
    ]


def test_wselob_context_trace_is_pre_event_and_target_blind() -> None:
    context_rows: list[tuple[int, int, float, float, float, float, float]] = []
    _process_day_mechanisms(
        _orders(),
        entity_code=3,
        session_start_ns=10,
        session_end_ns=39,
        bin_nanoseconds=10,
        continuous=True,
        replenishment_lookback_ns=10,
        context_rows=context_rows,
    )
    assert [row[0] for row in context_rows] == [10, 20, 30]
    # The first context has no earlier in-session message.  The second sees
    # exactly the first message and never counts its current primitive.
    assert context_rows[0][6] == 0.0
    assert context_rows[1][6] == 1.0 / 30.0


def test_continuous_risk_exposure_uses_declared_millisecond_unit() -> None:
    frame = _continuous_risk_frame(
        entity_code=0,
        session_start=0,
        session_end=1_999_999,
        weekday=0,
        events=[],
        target_ticks=[],
        baseline_bins=1,
        impact_edges=np.asarray([0, 1_000_000], dtype=np.int64),
        ticks_per_unit=1_000_000,
    )
    assert np.isclose(frame["exposure"].sum(), 2.0)


def test_wselob_adverse_excursion_target_is_declustered_and_rearmed() -> None:
    ticks = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)
    returns = np.asarray([-0.01, -0.03, -0.04, -0.005, -0.03])
    assert _adverse_excursion_targets(
        ticks,
        returns,
        threshold=0.02,
        rearm_fraction=0.5,
    ) == [20, 50]


def test_wselob_realized_volatility_is_past_only_and_declustered() -> None:
    ticks = np.asarray([0, 10, 20, 30, 40], dtype=np.int64)
    mid = 200.0 * np.exp(np.asarray([0.0, 0.01, 0.01, -0.01, -0.01]))
    observed_ticks, volatility = _realized_volatility(
        ticks,
        mid,
        session_start_ns=0,
        session_end_ns=40,
        horizon_ns=20,
    )
    np.testing.assert_array_equal(observed_ticks, [0, 10, 20, 30, 40])
    np.testing.assert_allclose(volatility, [0.0, 0.01, 0.01, 0.02, 0.02], atol=1e-14)
    assert _volatility_burst_targets(
        observed_ticks,
        volatility,
        threshold=0.015,
        rearm_fraction=0.5,
    ) == [30]


def test_duration_weighted_quantile_uses_time_not_change_count() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    weights = np.asarray([8.0, 1.0, 1.0])
    assert _duration_weighted_quantile(values, weights, 0.5) == 1.0
    assert _duration_weighted_quantile(values, weights, 0.9) == 2.0


def test_quantile_lag_bands_partition_cumulative_rule_response() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "data"
        write_dataset(
            root,
            entities=pd.DataFrame(
                {
                    "entity_id": ["day"],
                    "start_time": [0],
                    "end_time": [8],
                    "baseline_origin": [0],
                    "split_group": [0],
                }
            ),
            events=pd.DataFrame(
                [
                    (0, 0, 0, 1),
                    (0, 0, 1, 2),
                    (0, 2, 1, 3),
                    (0, 3, 0, 4),
                    (0, 4, 1, 5),
                ],
                columns=[
                    "entity_code",
                    "time",
                    "predicate_code",
                    "primitive_event_id",
                ],
            ),
            targets=pd.DataFrame(
                [(0, 7, 1)],
                columns=["entity_code", "time", "multiplicity"],
            ),
            predicate_names=("A", "B"),
            likelihood="poisson",
            time_unit="second",
            adverse_event_name="movement",
            f0_contract={
                "dynamic_predicates": True,
                "outcome_blind_predicate_construction": True,
                "direct_target_proxy_excluded_from_reported_dictionary": True,
                "strict_future_effect_required": True,
                "atomic_predicates": True,
                "primitive_event_provenance": True,
                "independent_certification_units": True,
            },
            provenance={"test": True},
        )
        data = Dataset.load(root)
        context = Context.make(data, np.asarray([0], dtype=np.int32))
        cumulative_engine = ResponseEngine(
            data, lag=2, knot_count=2, cache_bytes=1024**2
        )
        cumulative = cumulative_engine.blocks_many(context, (0, 1), (0, 1, 2))
        band_engine = ResponseEngine(data, lag=2, knot_count=2, cache_bytes=1024**2)
        band_engine.configure_window_bands({("unordered", (0, 1)): (0, 1, 2)})
        bands = band_engine.blocks_many(context, (0, 1), (0, 1, 2))

        def dense(block) -> np.ndarray:
            result = np.zeros((context.n_grid, 2), dtype=np.float64)
            result[block.rows] = block.values
            return result

        np.testing.assert_allclose(dense(bands[0]), dense(cumulative[0]))
        np.testing.assert_allclose(
            dense(bands[1]), dense(cumulative[1]) - dense(cumulative[0])
        )
        np.testing.assert_allclose(
            dense(bands[2]), dense(cumulative[2]) - dense(cumulative[1])
        )
