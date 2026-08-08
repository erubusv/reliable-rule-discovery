from __future__ import annotations

import numpy as np
import pandas as pd

from crbstpp.preprocess.wselob import (
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
    _adverse_excursion_targets,
    _ordered_partition,
    _process_day,
    _process_day_mechanisms,
)


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


def test_wselob_adverse_excursion_target_is_declustered_and_rearmed() -> None:
    ticks = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)
    returns = np.asarray([-0.01, -0.03, -0.04, -0.005, -0.03])
    assert _adverse_excursion_targets(
        ticks,
        returns,
        threshold=0.02,
        rearm_fraction=0.5,
    ) == [20, 50]
