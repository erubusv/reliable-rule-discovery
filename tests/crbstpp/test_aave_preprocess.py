from __future__ import annotations

import json
import gzip

import pandas as pd
import numpy as np

from crbstpp.data import Dataset, write_dataset
from crbstpp.preprocess.aave import (
    BLOCKS_PER_TICK,
    DEPLOYMENTS,
    PREDICATES,
    PREDICATE_CONTRAST_FAMILIES,
    PREDICATE_INDEX,
    RESERVE_DATA_UPDATED_TOPIC,
    STAGING_PREDICATES,
    _decode_position_log,
    _market_exposure_events,
    _entry_mechanism_predicate,
    _episode_tables,
    _mechanism_predicate,
    _transaction_mechanism_predicate,
    _write_parquet_atomic,
    stage_finsurvival_sample,
)
from crbstpp.response import Context, ResponseEngine
from crbstpp.rules import Support
from crbstpp.solver import fit_model_matrix


def test_reported_predicates_cover_preregistered_two_sided_financial_actions():
    catalog = set(PREDICATES)
    for sides in PREDICATE_CONTRAST_FAMILIES.values():
        assert sides["side_a"]
        assert sides["side_b"]
        assert set((*sides["side_a"], *sides["side_b"])) <= catalog


def test_aave_staging_preserves_transaction_identity_and_atomicity(tmp_path):
    columns = [
        "id",
        "type",
        "timestamp",
        "user",
        "onBehalfOf",
        "reserve",
        "borrowRateMode",
        "fromState",
        "toState",
        "borrowRateModeFrom",
        "borrowRateModeTo",
        "version",
        "deployment",
    ]
    alice = "0x" + "11" * 20
    bob = "0x" + "22" * 20
    records = [
        [
            "0xaaa",
            "deposit",
            10,
            alice,
            alice,
            "USDC",
            None,
            None,
            None,
            None,
            None,
            "V2",
            "Mainnet",
        ],
        [
            "0xbbb",
            "deposit",
            20,
            alice,
            alice,
            "USDC",
            None,
            None,
            None,
            None,
            None,
            "V2",
            "Mainnet",
        ],
        [
            "0xccc",
            "borrow",
            30,
            bob,
            alice,
            "USDC",
            "Variable",
            None,
            None,
            None,
            None,
            "V2",
            "Mainnet",
        ],
        [
            "0xddd",
            "repay",
            40,
            alice,
            bob,
            "USDC",
            None,
            None,
            None,
            None,
            None,
            "V2",
            "Mainnet",
        ],
        [
            "0xeee",
            "collateral",
            50,
            alice,
            None,
            "USDC",
            None,
            "FALSE",
            "TRUE",
            None,
            None,
            "V2",
            "Mainnet",
        ],
        [
            "0xeee",
            "liquidation",
            50,
            alice,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "V2",
            "Mainnet",
        ],
    ]
    input_csv = tmp_path / "sample.csv"
    pd.DataFrame(records, columns=columns).to_csv(input_csv, index=False)

    output = stage_finsurvival_sample(input_csv, tmp_path / "staging")
    manifest = json.loads((output / "manifest.json").read_text())
    events = pd.read_parquet(output / "predicate_events.parquet")
    targets = pd.read_parquet(output / "target_events.parquet")

    assert manifest["fit_ready"] is False
    assert tuple(manifest["semantics"]["predicate_names"]) == STAGING_PREDICATES
    assert events.groupby("primitive_event_id")["predicate_code"].nunique().max() == 1
    assert events["predicate_name"].tolist()[:2] == [
        "pred_first_self_supply_reserve",
        "pred_repeat_self_supply_reserve",
    ]
    assert "pred_delegated_borrow" in set(events["predicate_name"])
    assert "pred_third_party_repay" in set(events["predicate_name"])
    collateral_id = int(
        events.loc[events["transaction_hash"].eq("0xeee"), "primitive_event_id"].iloc[0]
    )
    assert (
        int(
            targets.loc[
                targets["transaction_hash"].eq("0xeee"), "primitive_event_id"
            ].iloc[0]
        )
        == collateral_id
    )


def test_aave_debt_episodes_use_wallet_split_and_first_liquidation_end(tmp_path):
    raw = tmp_path / "raw"
    chunk = raw / "v3" / "pool_logs" / "logs_17000000_18000000.jsonl.gz"
    chunk.parent.mkdir(parents=True)
    reserve = "0x" + "aa" * 20

    def topic(address):
        return "0x" + "0" * 24 + address[2:]

    def word(value):
        if isinstance(value, str):
            value = int(value, 16)
        return f"{int(value):064x}"

    logs = []
    zero_rows = []
    for index in range(48):
        owner = "0x" + f"{index + 1:040x}"
        first = 17_000_000 + index * 21_600
        common = {
            "version": "v3",
            "address": DEPLOYMENTS["v3"].pool_address,
            "transaction_hash": "0x" + f"{index + 1:064x}",
            "removed": False,
        }
        logs.extend(
            [
                {
                    **common,
                    "event_type": "supply",
                    "block_number": first,
                    "transaction_index": 0,
                    "log_index": 0,
                    "topics": [
                        DEPLOYMENTS["v3"].topics[0],
                        topic(reserve),
                        topic(owner),
                        "0x" + "0" * 64,
                    ],
                    "data": "0x" + word(owner) + word(100),
                },
                {
                    **common,
                    "event_type": "borrow",
                    "block_number": first,
                    "transaction_index": 1,
                    "log_index": 1,
                    "topics": [
                        DEPLOYMENTS["v3"].topics[2],
                        topic(reserve),
                        topic(owner),
                        "0x" + "0" * 64,
                    ],
                    "data": "0x" + word(owner) + word(50) + word(2) + word(1),
                },
                {
                    **common,
                    "event_type": "borrow",
                    "block_number": first + 100,
                    "transaction_index": 0,
                    "log_index": 2,
                    "topics": [
                        DEPLOYMENTS["v3"].topics[2],
                        topic(reserve),
                        topic(owner),
                        "0x" + "0" * 64,
                    ],
                    "data": "0x" + word(owner) + word(5) + word(2) + word(1),
                },
                {
                    **common,
                    "event_type": "collateral_enable",
                    "block_number": first + 200,
                    "transaction_index": 0,
                    "log_index": 3,
                    "topics": [
                        DEPLOYMENTS["v3"].topics[5],
                        topic(reserve),
                        topic(owner),
                    ],
                    "data": "0x",
                },
                {
                    **common,
                    "event_type": "supply",
                    "block_number": first + 300,
                    "transaction_index": 0,
                    "log_index": 4,
                    "topics": [
                        DEPLOYMENTS["v3"].topics[0],
                        topic(reserve),
                        topic(owner),
                        "0x" + "0" * 64,
                    ],
                    "data": "0x" + word(owner) + word(10),
                },
                {
                    **common,
                    "event_type": "repay",
                    "block_number": first + 400,
                    "transaction_index": 0,
                    "log_index": 5,
                    "topics": [
                        DEPLOYMENTS["v3"].topics[3],
                        topic(reserve),
                        topic(owner),
                        topic(owner),
                    ],
                    "data": "0x" + word(1) + word(0),
                },
                {
                    **common,
                    "event_type": "liquidation",
                    "block_number": first + 7_200,
                    "transaction_index": 2,
                    "log_index": 6,
                    "topics": [
                        DEPLOYMENTS["v3"].topics[7],
                        topic(reserve),
                        topic(reserve),
                        topic(owner),
                    ],
                    "data": "0x" + word(10) + word(10) + word(owner) + word(0),
                },
                {
                    **common,
                    "event_type": "repay",
                    "block_number": first + 14_400,
                    "transaction_index": 3,
                    "log_index": 7,
                    "topics": [
                        DEPLOYMENTS["v3"].topics[3],
                        topic(reserve),
                        topic(owner),
                        topic(owner),
                    ],
                    "data": "0x" + word(100) + word(0),
                },
            ]
        )
        zero_rows.append(("v3", owner, first + 14_400, True))
    logs.sort(key=lambda row: (row["block_number"], row["transaction_index"]))
    with gzip.open(chunk, "wt", encoding="utf-8") as handle:
        for row in logs:
            handle.write(json.dumps(row) + "\n")
    (raw / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "crbstpp.aave_raw_pool_logs.v1",
                "end_block": logs[-1]["block_number"],
                "chunks": [
                    {
                        "version": "v3",
                        "first_block": 17_000_000,
                        "path": str(chunk.relative_to(raw)),
                    }
                ],
            }
        )
    )
    zero = pd.DataFrame(zero_rows, columns=["version", "owner", "block", "debt_zero"])
    entities, events, targets, audit = _episode_tables(
        raw, zero_state=zero, partition_seed=111
    )
    assert len(entities) == 48
    assert set(entities["partition"]) == {0, 1, 2}
    assert entities["end_reason"].eq("first_liquidation").all()
    assert len(targets) == 48 and targets["multiplicity"].eq(1).all()
    # The generic risk-set-opening Borrow occurrence is not a rule, but its
    # target-blind debt configuration is retained as exactly one marked entry
    # event.  Later actions remain separate strictly-future mechanisms.
    assert set(events["predicate_code"]) == {
        PREDICATE_INDEX["pred_entry_self_variable_volatile_debt"],
        PREDICATE_INDEX["pred_self_variable_borrow_previously_used_debt_reserve"],
        PREDICATE_INDEX["pred_self_repay_while_debt_remains"],
        PREDICATE_INDEX[
            "pred_self_collateral_top_up_without_borrow"
        ],
        PREDICATE_INDEX["pred_collateral_reserve_enabled_without_borrow"],
    }
    assert audit["baseline_control_counts"] == []
    assert audit["opening_transactions_marked"] == 48
    target_by_entity = targets.set_index("entity_code")["time"]
    assert all(
        row.time < target_by_entity.loc[row.entity_code]
        for row in events.itertuples(index=False)
    )
    assert entities["baseline_stratum"].eq(1).all()
    assert audit["liquidations_outside_reconstructed_episode"] == 0


def test_aave_post_entry_mechanisms_use_transaction_frozen_portfolio_state():
    owner = "0x" + "11" * 20
    actor = "0x" + "22" * 20
    reserve = "0x" + "aa" * 20
    other_reserve = "0x" + "bb" * 20
    base = {
        "version": "v3",
        "owner": owner,
        "actor": owner,
        "reserve": reserve,
        "rate_mode": None,
    }
    def classify(record, active=(), seen=(), history=()):
        return _mechanism_predicate(
            record,
            active_collateral_reserves=frozenset(active),
            seen_debt_reserves=frozenset(seen),
        )

    def code(name):
        return PREDICATE_INDEX[name]

    assert classify({**base, "event_type": "borrow"}, seen={reserve}) == code(
        "pred_self_variable_borrow_previously_used_debt_reserve"
    )
    assert classify({**base, "event_type": "borrow"}) == code(
        "pred_self_variable_borrow_new_debt_reserve"
    )
    assert classify(
        {**base, "rate_mode": 1, "event_type": "borrow"}, seen={reserve}
    ) == code("pred_self_stable_rate_borrow")
    assert classify(
        {**base, "actor": actor, "event_type": "borrow"}, seen={reserve}
    ) == code("pred_delegated_borrow_previously_used_debt_reserve")
    assert classify({**base, "actor": actor, "event_type": "borrow"}) == code(
        "pred_delegated_borrow_new_debt_reserve"
    )
    assert classify({**base, "event_type": "repay"}) == code(
        "pred_self_repay_while_debt_remains"
    )
    assert classify({**base, "event_type": "repay", "use_a_tokens": True}) == code(
        "pred_self_repay_while_debt_remains"
    )
    assert classify(
        {**base, "event_type": "repay"}, history={"self_repay"}
    ) == code("pred_self_repay_while_debt_remains")
    assert classify({**base, "actor": actor, "event_type": "repay"}) == code(
        "pred_third_party_repay_while_debt_remains"
    )
    assert classify(
        {**base, "actor": actor, "event_type": "repay"},
        history={"third_party_repay"},
    ) == code(
        "pred_third_party_repay_while_debt_remains"
    )
    assert classify({**base, "event_type": "supply"}, active={reserve}) == code(
        "pred_self_collateral_top_up_without_borrow"
    )
    assert classify(
        {**base, "event_type": "supply"},
        active={reserve},
        history={"self_collateral_top_up"},
    ) == code(
        "pred_self_collateral_top_up_without_borrow"
    )
    assert classify(
        {**base, "actor": actor, "event_type": "supply"}, active={reserve}
    ) == code("pred_third_party_collateral_top_up_without_borrow")
    assert classify(
        {**base, "actor": actor, "event_type": "supply"},
        active={reserve},
        history={"third_party_collateral_top_up"},
    ) == code("pred_third_party_collateral_top_up_without_borrow")
    assert classify({**base, "event_type": "withdraw"}, active={reserve}) == code(
        "pred_last_collateral_withdrawal_while_debt_remains"
    )
    assert classify(
        {**base, "actor": actor, "event_type": "withdraw"},
        active={reserve, other_reserve},
    ) == code("pred_collateral_withdrawal_with_alternative_remaining")
    assert classify({**base, "event_type": "collateral_enable"}) == code(
        "pred_collateral_reserve_enabled_without_borrow"
    )
    assert classify(
        {**base, "reserve": other_reserve, "event_type": "collateral_enable"},
        active={reserve},
    ) == code("pred_collateral_reserve_enabled_without_borrow")
    assert classify(
        {**base, "event_type": "collateral_disable"},
        active={reserve, other_reserve},
    ) == code("pred_collateral_reserve_disabled")
    assert classify(
        {**base, "event_type": "collateral_disable"}, active={reserve}
    ) == code("pred_collateral_reserve_disabled")
    assert classify({**base, "event_type": "rate_mode_swap", "rate_mode": 1}) is None
    assert classify({**base, "event_type": "rate_mode_swap", "rate_mode": 2}) is None
    assert classify({**base, "event_type": "supply"}) is None


def test_aave_transaction_classifier_emits_one_compound_financial_action():
    owner = "0x" + "11" * 20
    reserve = "0x" + "aa" * 20
    collateral = "0x" + "bb" * 20
    base = {"version": "v3", "owner": owner, "actor": owner, "rate_mode": 2}
    predicate = _transaction_mechanism_predicate(
        [
            {**base, "event_type": "borrow", "reserve": reserve},
            {**base, "event_type": "supply", "reserve": collateral},
        ],
        active_collateral_reserves={collateral},
        seen_debt_reserves={reserve},
    )
    assert predicate == PREDICATE_INDEX["pred_leveraged_position_expansion_transaction"]

    predicate = _transaction_mechanism_predicate(
        [
            {**base, "event_type": "repay", "reserve": reserve},
            {**base, "event_type": "supply", "reserve": collateral},
        ],
        active_collateral_reserves={collateral},
        seen_debt_reserves={reserve},
    )
    assert predicate == PREDICATE_INDEX[
        "pred_self_position_deleveraging_transaction"
    ]


def test_aave_entry_classifier_retains_one_target_blind_debt_mark():
    owner = "0x" + "11" * 20
    actor = "0x" + "22" * 20
    stable = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    volatile = "0x" + "aa" * 20

    def borrow(reserve, *, who=owner, mode=2, log_index=0):
        return {
            "event_type": "borrow",
            "owner": owner,
            "actor": who,
            "reserve": reserve,
            "rate_mode": mode,
            "log_index": log_index,
        }

    def classify(records, active=()):
        return _entry_mechanism_predicate(
            records, active_collateral_reserves=frozenset(active)
        )
    assert (
        classify([borrow(stable)])
        == PREDICATE_INDEX["pred_entry_self_variable_stable_value_debt"]
    )
    assert (
        classify([borrow(volatile)])
        == PREDICATE_INDEX["pred_entry_self_variable_volatile_debt"]
    )
    assert (
        classify([borrow(stable, mode=1)])
        == PREDICATE_INDEX["pred_entry_self_stable_rate_debt"]
    )
    assert (
        classify([borrow(stable, who=actor)])
        == PREDICATE_INDEX["pred_entry_delegated_debt"]
    )
    assert (
        classify(
            [
                borrow(stable),
                {
                    "event_type": "collateral_enable",
                    "owner": owner,
                    "actor": owner,
                    "reserve": volatile,
                    "log_index": 1,
                },
            ]
        )
        == PREDICATE_INDEX["pred_entry_leveraged_position"]
    )


def test_aave_zero_address_system_log_is_excluded():
    reserve = "0x" + "aa" * 20
    record = {
        "version": "v3",
        "event_type": "collateral_enable",
        "block_number": 18_841_878,
        "transaction_index": 1,
        "log_index": 1,
        "topics": [
            DEPLOYMENTS["v3"].topics[5],
            "0x" + "0" * 24 + reserve[2:],
            "0x" + "0" * 64,
        ],
        "data": "0x",
    }
    assert _decode_position_log(record) is None


def test_aave_uint256_debt_is_losslessly_persisted(tmp_path):
    value = str(2**255 + 123)
    frame = pd.DataFrame(
        {
            "version": ["v2"],
            "owner": ["0x" + "11" * 20],
            "block": [12_000_000],
            "total_debt_base": [value],
            "debt_zero": [False],
            "state_source": ["historical_eth_call"],
        }
    )
    path = tmp_path / "debt.parquet"
    _write_parquet_atomic(frame, path)
    assert pd.read_parquet(path).loc[0, "total_debt_base"] == value


def test_aave_market_rate_predicates_are_symmetric_and_strictly_post_exposure(
    tmp_path,
):
    raw = tmp_path / "raw"
    chunk = raw / "v3" / "market_logs" / "logs.jsonl.gz"
    chunk.parent.mkdir(parents=True)
    reserve = "0x" + "aa" * 20
    topic_reserve = "0x" + "0" * 24 + reserve[2:]

    def word(value):
        return f"{int(value):064x}"

    # One hundred ordinary one-unit moves establish the preceding reference.
    # The two shocks then have conservative rank p-values 1/101 and 1/102.
    rates = [2000]
    rates.extend(2000 + offset for offset in range(1, 101))
    rates.extend((4100, 1100))
    with gzip.open(chunk, "wt", encoding="utf-8") as handle:
        for offset, rate in enumerate(rates):
            tick = 100 + offset
            row = {
                "version": "v3",
                "event_type": "reserve_data_updated",
                "address": DEPLOYMENTS["v3"].pool_address,
                "block_number": tick * BLOCKS_PER_TICK,
                "transaction_hash": "0x" + f"{offset + 1:064x}",
                "transaction_index": 0,
                "log_index": offset,
                "topics": [RESERVE_DATA_UPDATED_TOPIC, topic_reserve],
                "data": "0x" + word(0) + word(0) + word(rate) + word(0) + word(0),
                "removed": False,
            }
            handle.write(json.dumps(row) + "\n")
    (raw / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "crbstpp.aave_raw_pool_logs.v2",
                "end_block": 202 * BLOCKS_PER_TICK,
                "chunks": [],
                "market_chunks": [
                    {
                        "version": "v3",
                        "first_block": 100 * BLOCKS_PER_TICK,
                        "path": str(chunk.relative_to(raw)),
                    }
                ],
            }
        )
    )
    events, audit = _market_exposure_events(
        raw,
        [(7, "v3", reserve, 99, 203)],
    )
    observed = {(time, code) for _, time, code, _ in events}
    assert observed == {
        (
            201,
            PREDICATE_INDEX["pred_active_debt_reserve_variable_rate_upward_shock"],
        ),
        (
            202,
            PREDICATE_INDEX["pred_active_debt_reserve_variable_rate_downward_shock"],
        ),
    }
    assert audit["market_predicate_events"] == 2


def test_version_strata_are_distinct_free_baselines(tmp_path):
    entities = pd.DataFrame(
        {
            "entity_id": [f"e{i}" for i in range(8)],
            "start_time": np.zeros(8, dtype=np.int64),
            "end_time": np.ones(8, dtype=np.int64),
            "baseline_origin": np.zeros(8, dtype=np.int64),
            "split_group": np.arange(8, dtype=np.int64),
            "baseline_stratum": [0] * 4 + [1] * 4,
        }
    )
    events = pd.DataFrame(
        {
            "entity_code": pd.Series(dtype="int32"),
            "time": pd.Series(dtype="int64"),
            "predicate_code": pd.Series(dtype="int16"),
            "primitive_event_id": pd.Series(dtype="int64"),
        }
    )
    targets = pd.DataFrame(
        {
            "entity_code": [0, 4, 5],
            "time": [1, 1, 1],
            "multiplicity": [1, 1, 1],
        }
    )
    root = write_dataset(
        tmp_path / "strata",
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=("unused_dynamic_source",),
        likelihood="first_event_cloglog",
        time_unit="tick",
        adverse_event_name="first event",
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded_from_reported_dictionary": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
            "primitive_event_provenance": True,
        },
        provenance={"test": True},
    )
    dataset = Dataset.load(root)
    context = Context.make(dataset, np.arange(dataset.n_entities, dtype=np.int32))
    engine = ResponseEngine(dataset, lag=1, knot_count=1, cache_bytes=0)
    matrix = engine.model_matrix(context, Support(()))
    fit = fit_model_matrix(
        matrix,
        likelihood=dataset.likelihood,
        tolerance=1.0e-9,
        max_iter=100,
    )
    assert fit.converged
    assert matrix.free_dimension == matrix.dimension == 2
    assert fit.coefficients[1] > fit.coefficients[0]
