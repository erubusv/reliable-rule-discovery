from __future__ import annotations

import bisect
import concurrent.futures
import gzip
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from ..data import write_dataset
from ..state import augment_history_state_dictionary, event_definitions


STAGING_SCHEMA = "crbstpp.aave_staging.v1"
# Public archival providers impose different result/range limits.  Queries are
# split recursively when the primary endpoint rejects a range; the remaining
# endpoints are deterministic failovers, not alternative data samples.
DEFAULT_RPC_URLS = (
    "https://eth.drpc.org",
    "https://0xrpc.io/eth",
    "https://rpc.mevblocker.io",
    "https://eth-pokt.nodies.app",
)
DEFAULT_RPC_URL = DEFAULT_RPC_URLS[0]


@dataclass(frozen=True)
class AaveDeployment:
    version: str
    pool_address: str
    start_block: int
    topics: tuple[str, ...]


V2_TOPICS = (
    # Deposit, Withdraw, Borrow, Repay, Swap, collateral on/off, liquidation.
    "0xde6857219544bb5b7746f48ed30be6386fefc61b2f864cacf559893bf50fd951",
    "0x3115d1449a7b732c986cba18244e897a450f61e1bb8d589cd2e69e6c8924f9f7",
    "0xc6a898309e823ee50bac64e45ca8adba6690e99e7841c45d754e2a38e9019d9b",
    "0x4cdde6e09bb755c9a5589ebaec640bbfedff1362d4b255ebf8339782b9942faa",
    "0xea368a40e9570069bb8e6511d668293ad2e1f03b0d982431fd223de9f3b70ca6",
    "0x00058a56ea94653cdf4f152d227ace22d4c00ad99e2a43f58cb7d9e3feb295f2",
    "0x44c58d81365b66dd4b1a7f36c25aa97b8c71c361ee4937adc1a00000227db5dd",
    "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286",
)

V3_TOPICS = (
    # Supply, Withdraw, Borrow, Repay, rate-mode swap, collateral on/off,
    # liquidation.  ReserveDataUpdated is intentionally not a user predicate.
    "0x2b627736bca15cd5381dcf80b0bf11fd197d01a037c52b927a881a10fb73ba61",
    "0x3115d1449a7b732c986cba18244e897a450f61e1bb8d589cd2e69e6c8924f9f7",
    "0xb3d084820fb1a9decffb176436bd02558d15fac9b0ddfed8c465bc7359d7dce0",
    "0xa534c8dbe71f871f9f3530e97a74601fea17b426cae02e1c5aee42c96c784051",
    "0x7962b394d85a534033ba2efcf43cd36de57b7ebeb3de0ca4428965d9b3ddc481",
    "0x00058a56ea94653cdf4f152d227ace22d4c00ad99e2a43f58cb7d9e3feb295f2",
    "0x44c58d81365b66dd4b1a7f36c25aa97b8c71c361ee4937adc1a00000227db5dd",
    "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286",
)

# Emitted by both Ethereum V2 and V3 Pool contracts.  It is downloaded to a
# separate immutable stream because it is protocol state, not a wallet action.
# Keccak256("ReserveDataUpdated(address,uint256,uint256,uint256,uint256,uint256)").
RESERVE_DATA_UPDATED_TOPIC = (
    "0x804c9b842b2748a22bb64b345453a3de7ca54a6ca45ce00d415894979e22897a"
)

DEPLOYMENTS = {
    "v2": AaveDeployment(
        version="v2",
        # Preserve EIP-55 case: at least one public archival provider applies
        # an incorrect case-sensitive address filter.
        pool_address="0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
        start_block=11_360_920,
        topics=V2_TOPICS,
    ),
    "v3": AaveDeployment(
        version="v3",
        pool_address="0x87870Bca3F3fD6335C3F4ce8392D69350B4fa4E2",
        start_block=16_291_006,
        topics=V3_TOPICS,
    ),
}

EVENT_NAMES = {
    V2_TOPICS[0]: "supply",
    V3_TOPICS[0]: "supply",
    V2_TOPICS[1]: "withdraw",
    V2_TOPICS[2]: "borrow",
    V3_TOPICS[2]: "borrow",
    V2_TOPICS[3]: "repay",
    V3_TOPICS[3]: "repay",
    V2_TOPICS[4]: "rate_mode_swap",
    V3_TOPICS[4]: "rate_mode_swap",
    V2_TOPICS[5]: "collateral_enable",
    V2_TOPICS[6]: "collateral_disable",
    V2_TOPICS[7]: "liquidation",
    RESERVE_DATA_UPDATED_TOPIC: "reserve_data_updated",
}

# Reported predicates are target-blind financial mechanisms observed strictly
# after entry into an active debt episode.  One wallet/transaction emits at
# most one reported predicate: multi-leg transactions are represented by one
# compound economic action instead of several same-transaction attributes.
# Identities use only ABI action, actor/owner relations, rate mode and
# transaction-frozen portfolio state. Repetition is deliberately *not* a
# second primitive predicate: the search layer represents it with the common
# target-blind history mark ``c``.
PREDICATES = (
    "pred_debt_restructuring_transaction",
    "pred_collateral_rotation_transaction",
    "pred_leveraged_position_expansion_transaction",
    "pred_self_position_deleveraging_transaction",
    "pred_third_party_position_deleveraging_transaction",
    "pred_self_variable_borrow_previously_used_debt_reserve",
    "pred_self_variable_borrow_new_debt_reserve",
    "pred_self_stable_rate_borrow",
    "pred_delegated_borrow_previously_used_debt_reserve",
    "pred_delegated_borrow_new_debt_reserve",
    "pred_self_repay_while_debt_remains",
    "pred_third_party_repay_while_debt_remains",
    "pred_self_collateral_top_up_without_borrow",
    "pred_third_party_collateral_top_up_without_borrow",
    "pred_collateral_reserve_enabled_without_borrow",
    "pred_collateral_withdrawal_with_alternative_remaining",
    "pred_last_collateral_withdrawal_while_debt_remains",
    "pred_collateral_reserve_disabled",
    "pred_entry_self_variable_stable_value_debt",
    "pred_entry_self_variable_volatile_debt",
    "pred_entry_self_stable_rate_debt",
    "pred_entry_delegated_debt",
    "pred_entry_leveraged_position",
    "pred_active_debt_reserve_variable_rate_upward_shock",
    "pred_active_debt_reserve_variable_rate_downward_shock",
    "pred_active_debt_reserve_liquidity_rate_upward_shock",
    "pred_active_debt_reserve_liquidity_rate_downward_shock",
    "pred_active_collateral_price_upward_shock",
    "pred_active_collateral_price_downward_shock",
    "pred_active_debt_asset_price_upward_shock",
    "pred_active_debt_asset_price_downward_shock",
)
PREDICATE_INDEX = {name: code for code, name in enumerate(PREDICATES)}

# Economic reserve classes are fixed from public token identities, never from
# liquidation outcomes or fitted target statistics.  They are used only to
# mark the debt asset in the episode-opening transaction.  Every address not
# explicitly identified as stable-value remains in the volatile/other class,
# which is recorded in dataset provenance.
STABLE_VALUE_RESERVES = frozenset(
    {
        # USDC, USDT, DAI, USDe, GHO, EURC, sUSDe, crvUSD, FRAX, LUSD,
        # USDP, TUSD, USDS, RLUSD and USDtb on Ethereum mainnet.
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "0xdac17f958d2ee523a2206206994597c13d831ec7",
        "0x6b175474e89094c44da98b954eedeac495271d0f",
        "0x4c9edd5852cd905f086c759e8383e09bff1e68b3",
        "0x40d16fc0246ad3160ccc09b8d0d3a2cd28ae6c2f",
        "0x1abaea1f7c830bd89acc67ec4af516284b1bc33c",
        "0x9d39a5de30e57443bff2a8307a4256c8797a3497",
        "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e",
        "0x853d955acef822db058eb8505911ed77f175b99e",
        "0x5f98805a4e8be255a32880fdec7f6728c6568ba0",
        "0x1456688345527be1f37e9e627da0837d6f08c925",
        "0x0000000000085d4780b73119b644ae5ecd22b376",
        "0xdc035d45d973e3ec169d2276ddab16f1e407384f",
        "0x8292bb45bf1ee4d140127049757c2e0ff06317ed",
        "0xc139190f447e929f090edeb554d95abb8b18ac1c",
    }
)
# Pre-registered two-sided financial-action contrasts.  The sides deliberately
# do not encode an expected liquidation direction: discovery remains free to
# estimate excitation or inhibition for either side.  This contract prevents
# the reported dictionary from containing only risk-taking opportunities while
# retaining neutral restructuring/rotation predicates as separate mechanisms.
PREDICATE_CONTRAST_FAMILIES = {
    "position_scale": {
        "side_a": ("pred_leveraged_position_expansion_transaction",),
        "side_b": (
            "pred_self_position_deleveraging_transaction",
            "pred_third_party_position_deleveraging_transaction",
        ),
    },
    "self_debt_flow": {
        "side_a": (
            "pred_self_variable_borrow_previously_used_debt_reserve",
            "pred_self_variable_borrow_new_debt_reserve",
            "pred_self_stable_rate_borrow",
        ),
        "side_b": ("pred_self_repay_while_debt_remains",),
    },
    "third_party_debt_flow": {
        "side_a": (
            "pred_delegated_borrow_previously_used_debt_reserve",
            "pred_delegated_borrow_new_debt_reserve",
        ),
        "side_b": ("pred_third_party_repay_while_debt_remains",),
    },
    "collateral_balance": {
        "side_a": (
            "pred_self_collateral_top_up_without_borrow",
            "pred_third_party_collateral_top_up_without_borrow",
        ),
        "side_b": (
            "pred_collateral_withdrawal_with_alternative_remaining",
            "pred_last_collateral_withdrawal_while_debt_remains",
        ),
    },
    "collateral_eligibility": {
        "side_a": ("pred_collateral_reserve_enabled_without_borrow",),
        "side_b": ("pred_collateral_reserve_disabled",),
    },
    "entry_debt_asset": {
        "side_a": ("pred_entry_self_variable_stable_value_debt",),
        "side_b": ("pred_entry_self_variable_volatile_debt",),
    },
    "debt_market_rate": {
        "side_a": ("pred_active_debt_reserve_variable_rate_upward_shock",),
        "side_b": ("pred_active_debt_reserve_variable_rate_downward_shock",),
    },
    "reserve_liquidity_rate": {
        "side_a": ("pred_active_debt_reserve_liquidity_rate_upward_shock",),
        "side_b": ("pred_active_debt_reserve_liquidity_rate_downward_shock",),
    },
    "collateral_market_price": {
        "side_a": ("pred_active_collateral_price_upward_shock",),
        "side_b": ("pred_active_collateral_price_downward_shock",),
    },
    "debt_asset_market_price": {
        "side_a": ("pred_active_debt_asset_price_upward_shock",),
        "side_b": ("pred_active_debt_asset_price_downward_shock",),
    },
}
for _family, _sides in PREDICATE_CONTRAST_FAMILIES.items():
    if not _sides["side_a"] or not _sides["side_b"]:
        raise AssertionError(f"predicate contrast {_family!r} is one-sided")
    _unknown = {
        name
        for names in _sides.values()
        for name in names
        if name not in PREDICATE_INDEX
    }
    if _unknown:
        raise AssertionError(f"unknown predicates in {_family!r}: {sorted(_unknown)}")
# Aave exposes the complete debt-episode risk interval, so a transaction-history
# opportunity control is neither required as an offset nor innocuous: it would
# condition away a discoverable financial behaviour.  The null therefore uses
# only the protocol-version intercepts and all post-entry actions compete in the
# reported rule dictionary.
BASELINE_CONTROLS: tuple[str, ...] = ()
POSITION_ACTION_TYPES = frozenset(
    {
        "supply",
        "withdraw",
        "borrow",
        "repay",
        "rate_mode_swap",
        "collateral_enable",
        "collateral_disable",
    }
)

# The public FinSurvival sample cannot reconstruct debt-positive episodes and
# is staging-only.  Preserve its coarse action catalog under an explicit name
# rather than silently presenting it as the fit-ready predicate contract.
STAGING_PREDICATES = (
    "pred_first_self_supply_reserve",
    "pred_repeat_self_supply_reserve",
    "pred_supply_for_other",
    "pred_first_self_withdraw_reserve",
    "pred_repeat_self_withdraw_reserve",
    "pred_withdraw_to_other",
    "pred_first_self_stable_borrow_reserve",
    "pred_repeat_self_stable_borrow_reserve",
    "pred_first_self_variable_borrow_reserve",
    "pred_repeat_self_variable_borrow_reserve",
    "pred_delegated_borrow",
    "pred_self_repay",
    "pred_third_party_repay",
    "pred_collateral_enable",
    "pred_collateral_disable",
    "pred_rate_mode_stable_to_variable",
    "pred_rate_mode_variable_to_stable",
)

F0_CONTRACT = {
    "dynamic_predicates": True,
    "outcome_blind_predicate_construction": True,
    "direct_target_proxy_excluded_from_reported_dictionary": True,
    "strict_future_effect_required": True,
    "atomic_predicates": True,
    "primitive_event_provenance": True,
    "same_primitive_attributes_share_provenance": True,
    "same_primitive_cannot_repeat_as_high_order_witness": True,
    "two_sided_financial_action_contrasts": True,
    "outcome_blind_external_market_state": True,
}

EVENT_CODE = {
    "supply": 0,
    "withdraw": 1,
    "borrow": 2,
    "repay": 3,
    "rate_mode_swap": 4,
    "collateral_enable": 5,
    "collateral_disable": 6,
    "liquidation": 7,
}

STATE_RPC_URLS = (
    "https://eth.drpc.org",
    "https://rpc.mevblocker.io",
    "https://gateway.tenderly.co/public/mainnet",
)
GET_USER_ACCOUNT_DATA_SELECTOR = "bf92857c"
PARTITION_NAMES = ("fit", "cert", "test")
# The current sparse solver represents risk time on an integer grid.  A raw
# block grid would create years of empty rows per wallet; deterministic
# 7,200-block epochs retain chain ordering while avoiding that exact, redundant
# expansion.  No target or fitted statistic determines this resolution.
BLOCKS_PER_TICK = 7_200
# Pre-registered, target-free level for the sequential reserve-rate rank test.
MARKET_SHOCK_ALPHA = 0.01
# Official Ethereum Aave price oracles used by the corresponding deployments.
# Historical ``eth_call`` is made against the deployment-specific interface;
# every response is content-digested before it becomes a predicate input.
AAVE_ORACLE_ADDRESSES = {
    "v2": "0xa50ba011c48153de246e5192c8f9258a2ba79ca9",
    "v3": "0x54586be62e3c3580375ae3723c145253060ca0c2",
}
GET_ASSET_PRICE_SELECTOR = "b3596f07"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rpc(
    urls: str | Iterable[str],
    method: str,
    params: list[object],
    *,
    retries: int = 7,
    timeout: float = 90.0,
) -> object:
    endpoints = (urls,) if isinstance(urls, str) else tuple(urls)
    if not endpoints:
        raise ValueError("at least one RPC endpoint is required")
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    for attempt in range(retries):
        last_error: Exception | None = None
        for url in endpoints:
            try:
                response = requests.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                body = response.json()
                if "error" in body:
                    raise RuntimeError(str(body["error"]))
                return body["result"]
            except (requests.RequestException, ValueError, RuntimeError) as error:
                last_error = error
        if attempt + 1 == retries:
            assert last_error is not None
            raise last_error
        time.sleep(min(16.0, 0.5 * (2**attempt)))
    raise AssertionError("unreachable")


def _rpc_batch(
    urls: str | Iterable[str],
    calls: Iterable[tuple[str, list[object]]],
    *,
    retries: int = 5,
    timeout: float = 90.0,
) -> list[object | Exception]:
    """Execute one ordered JSON-RPC batch, failing open at the caller.

    Historical oracle prices are independent immutable calls.  Batching only
    removes HTTP round trips; request parameters, returned values and cache
    digests are exactly the same as for :func:`_rpc`.
    """

    endpoints = (urls,) if isinstance(urls, str) else tuple(urls)
    if not endpoints:
        raise ValueError("at least one RPC endpoint is required")
    requests_payload = [
        {"jsonrpc": "2.0", "id": index, "method": method, "params": params}
        for index, (method, params) in enumerate(calls, start=1)
    ]
    if not requests_payload:
        return []
    for attempt in range(retries):
        last_error: Exception | None = None
        for url in endpoints:
            try:
                response = requests.post(url, json=requests_payload, timeout=timeout)
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, list):
                    raise ValueError("JSON-RPC batch returned a non-list response")
                by_id: dict[int, object | Exception] = {}
                for item in body:
                    if not isinstance(item, dict) or "id" not in item:
                        raise ValueError("malformed JSON-RPC batch item")
                    identifier = int(item["id"])
                    if identifier in by_id:
                        raise ValueError("duplicate JSON-RPC batch item")
                    if "error" in item:
                        by_id[identifier] = RuntimeError(str(item["error"]))
                        continue
                    if "result" not in item:
                        raise ValueError("incomplete JSON-RPC batch item")
                    by_id[identifier] = item["result"]
                expected = set(range(1, len(requests_payload) + 1))
                if set(by_id) != expected:
                    raise ValueError("incomplete JSON-RPC batch response")
                return [by_id[index] for index in range(1, len(expected) + 1)]
            except (requests.RequestException, ValueError, RuntimeError) as error:
                last_error = error
        if attempt + 1 == retries:
            assert last_error is not None
            raise last_error
        time.sleep(min(16.0, 0.5 * (2**attempt)))
    raise AssertionError("unreachable")


def _fetch_log_chunk(
    *,
    deployment: AaveDeployment,
    first: int,
    last: int,
    rpc_urls: tuple[str, ...],
    output: Path,
    topics: tuple[str, ...] | None = None,
) -> dict[str, object]:
    selected_topics = tuple(topics or deployment.topics)
    allowed_topics = frozenset(selected_topics)

    def fetch_range(left: int, right: int) -> list[dict[str, object]]:
        params: list[object] = [
            {
                "fromBlock": hex(left),
                "toBlock": hex(right),
                "address": deployment.pool_address,
                # Ethereum JSON-RPC interprets the nested list as topic0 OR.
                "topics": [list(selected_topics)],
            }
        ]
        try:
            # The primary source is fastest.  Large-result failures are split
            # instead of falling through to a provider that silently truncates.
            result = _rpc(rpc_urls[:1], "eth_getLogs", params, retries=2)
        except (requests.RequestException, ValueError, RuntimeError):
            if right > left:
                midpoint = (left + right) // 2
                return fetch_range(left, midpoint) + fetch_range(midpoint + 1, right)
            result = _rpc(rpc_urls[1:], "eth_getLogs", params, retries=4)
        if not isinstance(result, list):
            raise ValueError("eth_getLogs returned a non-list result")
        return result

    logs_by_key: dict[tuple[int, int, int], dict[str, object]] = {}
    for raw in fetch_range(first, last):
        if not isinstance(raw, dict):
            raise ValueError("invalid Ethereum log record")
        block = int(str(raw["blockNumber"]), 16)
        transaction_index = int(str(raw["transactionIndex"]), 16)
        log_index = int(str(raw["logIndex"]), 16)
        if block < first or block > last:
            raise ValueError("RPC returned a log outside the requested range")
        if str(raw["address"]).lower() != deployment.pool_address.lower():
            raise ValueError("RPC returned a log from another contract")
        if str(raw["topics"][0]).lower() not in allowed_topics:
            raise ValueError("RPC returned a log with another event signature")
        logs_by_key[(block, transaction_index, log_index)] = raw
    logs = [logs_by_key[key] for key in sorted(logs_by_key)]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
        for log in logs:
            if not isinstance(log, dict):
                raise ValueError("invalid Ethereum log record")
            record = {
                "version": deployment.version,
                "event_type": EVENT_NAMES[str(log["topics"][0]).lower()],
                "address": str(log["address"]).lower(),
                "block_number": int(str(log["blockNumber"]), 16),
                "transaction_hash": str(log["transactionHash"]).lower(),
                "transaction_index": int(str(log["transactionIndex"]), 16),
                "log_index": int(str(log["logIndex"]), 16),
                "topics": [str(value).lower() for value in log["topics"]],
                "data": str(log["data"]).lower(),
                "removed": bool(log.get("removed", False)),
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, output)
    return {
        "first_block": first,
        "last_block": last,
        "events": len(logs),
        "path": output.name,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
    }


def download_aave_pool_logs(
    raw_root: str | Path,
    *,
    rpc_urls: Iterable[str] = DEFAULT_RPC_URLS,
    versions: Iterable[str] = ("v2", "v3"),
    end_block: int | None = None,
    chunk_size: int = 10_000,
    workers: int = 4,
    include_market_state: bool = True,
) -> Path:
    """Resume-safe download of Aave user actions and reserve-rate state.

    Files are immutable block-range chunks.  Existing chunks are accepted only
    after their digest is recorded in the final provenance manifest, so a
    stopped process can safely be rerun without repeating completed RPC calls.
    """

    raw_root = Path(raw_root)
    rpc_urls = tuple(str(value) for value in rpc_urls)
    if not rpc_urls:
        raise ValueError("at least one RPC endpoint is required")
    if chunk_size < 1 or chunk_size > 10_000:
        raise ValueError("public Ethereum RPC chunks must be in [1, 10000]")
    selected = tuple(str(value).lower() for value in versions)
    if not selected or any(value not in DEPLOYMENTS for value in selected):
        raise ValueError("versions must be a nonempty subset of v2/v3")
    if end_block is None:
        latest = _rpc(rpc_urls, "eth_blockNumber", [])
        end_block = int(str(latest), 16)
    end_block = int(end_block)

    jobs: list[tuple[AaveDeployment, int, int, Path, str]] = []
    complete: list[dict[str, object]] = []
    complete_market: list[dict[str, object]] = []
    for version in selected:
        deployment = DEPLOYMENTS[version]
        if end_block < deployment.start_block:
            continue
        directory = raw_root / version / "pool_logs"
        for first in range(deployment.start_block, end_block + 1, chunk_size):
            last = min(end_block, first + chunk_size - 1)
            output = directory / f"logs_{first:08d}_{last:08d}.jsonl.gz"
            if output.is_file():
                complete.append(
                    {
                        "first_block": first,
                        "last_block": last,
                        "path": str(output.relative_to(raw_root)),
                        "bytes": output.stat().st_size,
                        "sha256": _sha256(output),
                        "events": None,
                        "version": version,
                    }
                )
            else:
                jobs.append((deployment, first, last, output, "action"))
            if include_market_state:
                market_output = (
                    raw_root
                    / version
                    / "market_logs"
                    / f"logs_{first:08d}_{last:08d}.jsonl.gz"
                )
                if market_output.is_file():
                    complete_market.append(
                        {
                            "first_block": first,
                            "last_block": last,
                            "path": str(market_output.relative_to(raw_root)),
                            "bytes": market_output.stat().st_size,
                            "sha256": _sha256(market_output),
                            "events": None,
                            "version": version,
                        }
                    )
                else:
                    jobs.append((deployment, first, last, market_output, "market"))

    def execute(
        job: tuple[AaveDeployment, int, int, Path, str],
    ) -> tuple[str, dict[str, object]]:
        deployment, first, last, output, kind = job
        result = _fetch_log_chunk(
            deployment=deployment,
            first=first,
            last=last,
            rpc_urls=rpc_urls,
            output=output,
            topics=(RESERVE_DATA_UPDATED_TOPIC,) if kind == "market" else None,
        )
        result["path"] = str(output.relative_to(raw_root))
        result["version"] = deployment.version
        return kind, result

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(workers))
    ) as pool:
        for kind, result in pool.map(execute, jobs):
            (complete_market if kind == "market" else complete).append(result)

    complete.sort(key=lambda value: (str(value["version"]), int(value["first_block"])))
    complete_market.sort(
        key=lambda value: (str(value["version"]), int(value["first_block"]))
    )
    manifest = {
        "schema": "crbstpp.aave_raw_pool_logs.v2",
        "chain": "ethereum-mainnet",
        "chain_id": 1,
        "end_block": end_block,
        "chunk_size": chunk_size,
        "deployments": {
            version: {
                "pool_address": DEPLOYMENTS[version].pool_address,
                "start_block": DEPLOYMENTS[version].start_block,
                "topics": list(DEPLOYMENTS[version].topics),
            }
            for version in selected
        },
        "source": {
            "kind": "ethereum_json_rpc",
            # Do not persist API keys embedded in a user URL.
            "endpoint_hosts": [
                url.split("//", 1)[-1].split("/", 1)[0] for url in rpc_urls
            ],
            "aave_subgraph_reference_commit": "23c5374a809ca7223644326ee6bc05040c603e7b",
        },
        "chunks": complete,
        "market_chunks": complete_market,
        "market_topics": [RESERVE_DATA_UPDATED_TOPIC],
    }
    raw_root.mkdir(parents=True, exist_ok=True)
    temporary = raw_root / f"manifest.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, raw_root / "manifest.json")
    return raw_root


def _clean_address(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.where(values.str.fullmatch(r"0x[0-9a-f]{40}", na=False))


def _strict_first_mask(
    frame: pd.DataFrame,
    *,
    owner: pd.Series,
    family: pd.Series,
) -> np.ndarray:
    keys = pd.DataFrame(
        {
            "owner": owner,
            "reserve": frame["reserve"],
            "family": family,
            "timestamp": frame["timestamp"],
        }
    )
    first_time = keys.groupby(["owner", "reserve", "family"], sort=False)[
        "timestamp"
    ].transform("min")
    # Every log tied at the first block timestamp sees the same frozen past.
    return keys["timestamp"].eq(first_time).to_numpy(dtype=bool)


def stage_finsurvival_sample(
    input_csv: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Normalize the public Aave raw sample without inventing risk exposure.

    This creates an auditable staging table and predicate dictionary.  It does
    *not* emit a fit-ready :class:`Dataset`: exact debt-positive intervals need
    debt-token state events that the public sample omits.
    """

    input_csv, output_root = Path(input_csv), Path(output_root)
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    usecols = [
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
    frame = pd.read_csv(input_csv, usecols=usecols, dtype="string")
    frame["event_type"] = frame.pop("type").str.strip().str.lower()
    frame["transaction_hash"] = frame.pop("id").str.strip().str.lower()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype(
        np.int64
    )
    frame["user"] = _clean_address(frame["user"])
    frame["counterparty"] = _clean_address(frame.pop("onBehalfOf"))
    frame["reserve"] = frame["reserve"].fillna("").str.strip().str.upper()
    frame["rate_mode"] = frame.pop("borrowRateMode").fillna("").str.strip().str.lower()
    frame["rate_mode_from"] = (
        frame.pop("borrowRateModeFrom").fillna("").str.strip().str.lower()
    )
    frame["rate_mode_to"] = (
        frame.pop("borrowRateModeTo").fillna("").str.strip().str.lower()
    )
    frame["from_state"] = frame.pop("fromState").fillna("").str.strip().str.lower()
    frame["to_state"] = frame.pop("toState").fillna("").str.strip().str.lower()
    frame["version"] = frame["version"].str.strip().str.lower()
    frame["deployment"] = frame["deployment"].str.strip().str.lower()
    frame = frame.sort_values(
        ["timestamp", "transaction_hash", "event_type"], kind="stable"
    ).reset_index(drop=True)
    hashes = np.sort(frame["transaction_hash"].unique())
    primitive_map = pd.Series(np.arange(len(hashes), dtype=np.int64), index=hashes)
    frame["primitive_event_id"] = (
        frame["transaction_hash"].map(primitive_map).astype(np.int64)
    )

    event_type = frame["event_type"]
    # Owner is the account whose Aave position changes; actor is the account
    # initiating or receiving the operation.
    frame["owner"] = frame["user"]
    frame["actor"] = frame["user"]
    supply = event_type.eq("deposit")
    borrow = event_type.eq("borrow")
    repay = event_type.eq("repay")
    frame.loc[supply | borrow, "owner"] = frame.loc[
        supply | borrow, "counterparty"
    ].fillna(frame.loc[supply | borrow, "user"])
    frame.loc[repay, "actor"] = frame.loc[repay, "counterparty"].fillna(
        frame.loc[repay, "user"]
    )
    # In the FinSurvival export, onBehalfOf stores Withdraw.to.
    withdraw = event_type.eq("withdraw")
    frame.loc[withdraw, "actor"] = frame.loc[withdraw, "counterparty"].fillna(
        frame.loc[withdraw, "user"]
    )

    valid_source = frame["owner"].notna() & frame["reserve"].ne("")
    first_supply = _strict_first_mask(
        frame, owner=frame["owner"], family=frame["event_type"]
    )
    first_withdraw = _strict_first_mask(
        frame, owner=frame["owner"], family=frame["event_type"]
    )
    borrow_family = "borrow_" + frame["rate_mode"]
    first_borrow = _strict_first_mask(frame, owner=frame["owner"], family=borrow_family)

    owner = frame["owner"]
    actor = frame["actor"]
    self_action = owner.eq(actor).fillna(False).to_numpy(dtype=bool)
    masks = (
        supply.to_numpy() & valid_source.to_numpy() & self_action & first_supply,
        supply.to_numpy() & valid_source.to_numpy() & self_action & ~first_supply,
        supply.to_numpy() & valid_source.to_numpy() & ~self_action,
        withdraw.to_numpy() & valid_source.to_numpy() & self_action & first_withdraw,
        withdraw.to_numpy() & valid_source.to_numpy() & self_action & ~first_withdraw,
        withdraw.to_numpy() & valid_source.to_numpy() & ~self_action,
        borrow.to_numpy()
        & valid_source.to_numpy()
        & self_action
        & frame["rate_mode"].eq("stable").to_numpy()
        & first_borrow,
        borrow.to_numpy()
        & valid_source.to_numpy()
        & self_action
        & frame["rate_mode"].eq("stable").to_numpy()
        & ~first_borrow,
        borrow.to_numpy()
        & valid_source.to_numpy()
        & self_action
        & frame["rate_mode"].eq("variable").to_numpy()
        & first_borrow,
        borrow.to_numpy()
        & valid_source.to_numpy()
        & self_action
        & frame["rate_mode"].eq("variable").to_numpy()
        & ~first_borrow,
        borrow.to_numpy() & valid_source.to_numpy() & ~self_action,
        repay.to_numpy() & valid_source.to_numpy() & self_action,
        repay.to_numpy() & valid_source.to_numpy() & ~self_action,
        event_type.eq("collateral").to_numpy()
        & frame["to_state"].eq("true").to_numpy(),
        event_type.eq("collateral").to_numpy()
        & frame["to_state"].eq("false").to_numpy(),
        event_type.eq("swap").to_numpy()
        & frame["rate_mode_from"].eq("stable").to_numpy()
        & frame["rate_mode_to"].eq("variable").to_numpy(),
        event_type.eq("swap").to_numpy()
        & frame["rate_mode_from"].eq("variable").to_numpy()
        & frame["rate_mode_to"].eq("stable").to_numpy(),
    )
    if len(masks) != len(STAGING_PREDICATES):
        raise AssertionError("Aave predicate dictionary is misaligned")
    membership = np.sum(np.stack(masks, axis=1), axis=1)
    if np.any(membership > 1):
        raise AssertionError("Aave atomic predicates overlap within one raw log")

    event_parts = []
    for code, mask in enumerate(masks):
        selected = frame.loc[
            mask,
            ["owner", "timestamp", "primitive_event_id", "transaction_hash"],
        ].copy()
        selected["predicate_code"] = np.int16(code)
        selected["predicate_name"] = STAGING_PREDICATES[code]
        event_parts.append(selected)
    predicate_events = pd.concat(event_parts, ignore_index=True).sort_values(
        ["owner", "timestamp", "predicate_code", "primitive_event_id"],
        kind="stable",
    )
    targets = frame.loc[
        event_type.eq("liquidation") & frame["owner"].notna(),
        ["owner", "timestamp", "primitive_event_id", "transaction_hash"],
    ].copy()

    normalized_columns = [
        "transaction_hash",
        "primitive_event_id",
        "timestamp",
        "version",
        "deployment",
        "event_type",
        "owner",
        "actor",
        "reserve",
        "rate_mode",
        "rate_mode_from",
        "rate_mode_to",
        "from_state",
        "to_state",
    ]
    frame[normalized_columns].to_parquet(
        output_root / "normalized_events.parquet", index=False
    )
    predicate_events.to_parquet(output_root / "predicate_events.parquet", index=False)
    targets.to_parquet(output_root / "target_events.parquet", index=False)
    counts = predicate_events.groupby(
        ["predicate_code", "predicate_name"], sort=True
    ).agg(events=("owner", "size"), entities=("owner", "nunique"))
    counts = counts.reset_index().to_dict(orient="records")
    files = {}
    for name in ("normalized_events", "predicate_events", "target_events"):
        path = output_root / f"{name}.parquet"
        files[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "schema": STAGING_SCHEMA,
        "fit_ready": False,
        "fit_blocker": (
            "exact debt-positive borrower episodes require stable/variable "
            "debt-token state events absent from this public sample"
        ),
        "source": {
            "project": "FinSurvival public raw transaction sample",
            "repository": "Large-Transaction-Models/DMLR_DeFi_Survival_Benchmark",
            "commit": "b0d487ffca4eedf445502a01d342d2470bb55426",
            "input_path": str(input_csv),
            "input_sha256": _sha256(input_csv),
        },
        "semantics": {
            "target": "Aave borrower liquidation",
            "entity": "borrower position owner",
            "primitive_event_id": "Ethereum transaction hash",
            "strict_time": "Unix-second timestamp; tied transactions use frozen past",
            "predicate_names": list(STAGING_PREDICATES),
            "predicate_contract": "staging_only_coarse_actions_v1",
            "amount_or_target_tuned_predicates": False,
        },
        "rows": {
            "raw_logs": len(frame),
            "transactions": int(frame["transaction_hash"].nunique()),
            "owners": int(frame["owner"].nunique()),
            "predicate_events": len(predicate_events),
            "target_events": len(targets),
            "target_owners": int(targets["owner"].nunique()),
        },
        "predicate_counts": counts,
        "files": files,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_root


ZERO_ADDRESS = "0x" + "0" * 40


def _topic_address(value: str, *, allow_zero: bool = False) -> str:
    value = str(value).lower()
    if not value.startswith("0x") or len(value) != 66:
        raise ValueError(f"invalid indexed address word: {value!r}")
    address = "0x" + value[-40:]
    if address == ZERO_ADDRESS and not allow_zero:
        raise ValueError("zero address cannot identify an Aave position owner")
    return address


def _data_words(value: str) -> tuple[str, ...]:
    value = str(value).lower()
    if not value.startswith("0x"):
        raise ValueError("invalid Ethereum event data")
    payload = value[2:]
    if len(payload) % 64:
        raise ValueError("Ethereum event data is not word aligned")
    return tuple(payload[index : index + 64] for index in range(0, len(payload), 64))


def _word_address(value: str, *, allow_zero: bool = False) -> str:
    if len(value) != 64:
        raise ValueError("invalid ABI word")
    address = "0x" + value[-40:].lower()
    if address == ZERO_ADDRESS and not allow_zero:
        raise ValueError("zero address cannot identify an Aave actor")
    return address


def _decode_position_log(record: dict[str, object]) -> dict[str, object] | None:
    """Decode only fields needed by the pre-registered event dictionary.

    V2 and V3 deliberately share this canonical representation.  The two Pool
    ABIs differ in event names and a few trailing fields, but the indexed
    reserve/position-owner fields used here are aligned.
    """

    event_type = str(record["event_type"])
    version = str(record["version"])
    topics = tuple(str(value) for value in record["topics"])
    words = _data_words(str(record["data"]))
    if event_type not in EVENT_CODE or version not in DEPLOYMENTS:
        raise ValueError("unknown Aave log identity")
    if len(topics) < 3:
        raise ValueError(f"incomplete {event_type} topics")
    reserve = _topic_address(topics[1], allow_zero=True)
    owner = _topic_address(topics[2], allow_zero=True)
    actor = owner
    amount: int | None = None
    rate_mode: int | None = None
    use_a_tokens = False
    if event_type == "supply":
        if len(words) < 2:
            raise ValueError("incomplete Supply/Deposit data")
        actor = _word_address(words[0], allow_zero=True)
        amount = int(words[1], 16)
    elif event_type == "withdraw":
        if len(topics) < 4 or not words:
            raise ValueError("incomplete Withdraw data")
        actor = _topic_address(topics[3], allow_zero=True)
        amount = int(words[0], 16)
    elif event_type == "borrow":
        if len(words) < 3:
            raise ValueError("incomplete Borrow data")
        actor = _word_address(words[0], allow_zero=True)
        amount = int(words[1], 16)
        rate_mode = int(words[2], 16)
    elif event_type == "repay":
        if len(topics) < 4 or not words:
            raise ValueError("incomplete Repay data")
        actor = _topic_address(topics[3], allow_zero=True)
        amount = int(words[0], 16)
        # V3 appends ``useATokens`` to the canonical Repay event.  V2 has no
        # such field and therefore always represents external-liquidity
        # repayment here.
        use_a_tokens = version == "v3" and len(words) >= 2 and bool(int(words[1], 16))
    elif event_type == "rate_mode_swap":
        if not words:
            raise ValueError("incomplete SwapBorrowRateMode data")
        rate_mode = int(words[0], 16)
    elif event_type == "liquidation":
        if len(topics) < 4 or not words:
            raise ValueError("incomplete LiquidationCall data")
        # The position owner and repaid debt reserve differ from the generic
        # user-event layout: topics are collateral, debt, user.
        reserve = _topic_address(topics[2], allow_zero=True)
        owner = _topic_address(topics[3], allow_zero=True)
        actor = owner
        amount = int(words[0], 16)
    # A zero reserve/owner cannot define a user position, and a zero actor
    # cannot define one of the reported user actions. Such Pool-emitted system
    # records are excluded target-blindly instead of being assigned to a fake
    # wallet or crashing a long preprocessing run.
    if ZERO_ADDRESS in {reserve, owner, actor}:
        return None
    block = int(record["block_number"])
    transaction_index = int(record["transaction_index"])
    if transaction_index < 0 or transaction_index >= 1 << 20:
        raise ValueError("transaction index exceeds primitive-ID encoding")
    return {
        "version": version,
        "event_type": event_type,
        "block": block,
        "transaction_index": transaction_index,
        "log_index": int(record["log_index"]),
        "primitive_event_id": (block << 20) | transaction_index,
        "reserve": reserve,
        "owner": owner,
        "actor": actor,
        "amount": amount,
        "rate_mode": rate_mode,
        "use_a_tokens": use_a_tokens,
    }


def _decode_market_log(record: dict[str, object]) -> dict[str, object]:
    """Decode the outcome-blind reserve-rate state used by market predicates."""

    if str(record.get("event_type")) != "reserve_data_updated":
        raise ValueError("unknown Aave market log identity")
    version = str(record["version"])
    topics = tuple(str(value) for value in record["topics"])
    words = _data_words(str(record["data"]))
    if version not in DEPLOYMENTS or len(topics) < 2 or len(words) < 5:
        raise ValueError("incomplete ReserveDataUpdated log")
    reserve = _topic_address(topics[1])
    block = int(record["block_number"])
    transaction_index = int(record["transaction_index"])
    return {
        "version": version,
        "reserve": reserve,
        "block": block,
        "tick": block // BLOCKS_PER_TICK,
        "transaction_index": transaction_index,
        "log_index": int(record["log_index"]),
        "liquidity_rate": int(words[0], 16),
        "variable_borrow_rate": int(words[2], 16),
    }


def _raw_chunk_paths(raw_root: Path) -> tuple[Path, ...]:
    manifest_path = raw_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {
        "crbstpp.aave_raw_pool_logs.v1",
        "crbstpp.aave_raw_pool_logs.v2",
    }:
        raise ValueError("unsupported Aave raw-log manifest")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Aave raw manifest has no chunks")
    rows = sorted(
        chunks,
        key=lambda value: (str(value["version"]), int(value["first_block"])),
    )
    paths = tuple(raw_root / str(row["path"]) for row in rows)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("an Aave raw chunk named by the manifest is missing")
    return paths


def _market_chunk_paths(raw_root: Path) -> tuple[Path, ...]:
    manifest_path = raw_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("market_chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Aave market_chunks must be a list")
    rows = sorted(
        chunks,
        key=lambda value: (str(value["version"]), int(value["first_block"])),
    )
    paths = tuple(raw_root / str(row["path"]) for row in rows)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("an Aave market chunk named by the manifest is missing")
    return paths


def _iter_position_logs(
    raw_root: Path, *, audit: dict[str, int] | None = None
) -> Iterable[dict[str, object]]:
    for path in _raw_chunk_paths(raw_root):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if bool(record.get("removed", False)):
                    continue
                decoded = _decode_position_log(record)
                if decoded is None:
                    if audit is not None:
                        audit["excluded_zero_address_logs"] = (
                            audit.get("excluded_zero_address_logs", 0) + 1
                        )
                    continue
                yield decoded


def _iter_market_logs(raw_root: Path) -> Iterable[dict[str, object]]:
    for path in _market_chunk_paths(raw_root):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if bool(record.get("removed", False)):
                    continue
                yield _decode_market_log(record)


def _online_rank_shocks(
    values: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Prequential two-sided shock directions without future information."""
    values.sort()
    ticks: list[int] = []
    directions: list[int] = []
    previous: int | None = None
    previous_absolute_changes: list[int] = []
    for tick, value in values:
        if previous is not None and value != previous:
            change = value - previous
            absolute = abs(change)
            left = bisect.bisect_left(previous_absolute_changes, absolute)
            greater_or_equal = len(previous_absolute_changes) - left
            pvalue = (1 + greater_or_equal) / (len(previous_absolute_changes) + 1)
            if pvalue <= MARKET_SHOCK_ALPHA:
                ticks.append(int(tick))
                directions.append(1 if change > 0 else -1)
            bisect.insort(previous_absolute_changes, absolute)
        previous = value
    return np.asarray(ticks, dtype=np.int64), np.asarray(directions, dtype=np.int8)


def _online_rank_shocks_with_ids(
    values: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prequential shocks retaining the raw update's primitive provenance."""
    values.sort()
    ticks: list[int] = []
    directions: list[int] = []
    primitive_ids: list[int] = []
    previous: int | None = None
    previous_absolute_changes: list[int] = []
    for tick, value, primitive in values:
        if previous is not None and value != previous:
            change = value - previous
            absolute = abs(change)
            left = bisect.bisect_left(previous_absolute_changes, absolute)
            greater_or_equal = len(previous_absolute_changes) - left
            pvalue = (1 + greater_or_equal) / (len(previous_absolute_changes) + 1)
            if pvalue <= MARKET_SHOCK_ALPHA:
                ticks.append(int(tick))
                directions.append(1 if change > 0 else -1)
                primitive_ids.append(int(primitive))
            bisect.insort(previous_absolute_changes, absolute)
        previous = value
    return (
        np.asarray(ticks, dtype=np.int64),
        np.asarray(directions, dtype=np.int8),
        np.asarray(primitive_ids, dtype=np.int64),
    )


def _market_tick_directions(
    raw_root: Path,
) -> tuple[
    dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[tuple[str, str], int],
]:
    """Return rare daily reserve-rate shocks using only preceding rate data.

    All updates in one deterministic chain-time tick are collapsed to the last
    on-chain state.  For absolute change score ``a_t``, the conservative online
    rank p-value is ``(1 + #{a_s >= a_t, s<t}) / (t + 1)``.  An event is emitted
    only when it is at most ``MARKET_SHOCK_ALPHA``; ties can only make the test
    more conservative.  No future rate, target, or fitted rule is used.
    """

    final_by_tick: dict[
        tuple[str, str, str, int], tuple[int, int, int, int]
    ] = {}
    for row in _iter_market_logs(raw_root):
        order = (int(row["block"]), int(row["log_index"]))
        primitive = (int(row["block"]) << 20) | int(row["transaction_index"])
        for signal in ("variable_borrow_rate", "liquidity_rate"):
            key = (
                str(row["version"]),
                str(row["reserve"]),
                signal,
                int(row["tick"]),
            )
            previous = final_by_tick.get(key)
            if previous is None or order > previous[:2]:
                final_by_tick[key] = (*order, int(row[signal]), primitive)
    grouped: dict[tuple[str, str, str], list[tuple[int, int, int]]] = {}
    for (version, reserve, signal, tick), (_, _, rate, primitive) in final_by_tick.items():
        grouped.setdefault((version, reserve, signal), []).append(
            (tick, rate, primitive)
        )
    output: dict[
        tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for key, values in grouped.items():
        output[key] = _online_rank_shocks_with_ids(values)
    reserve_starts: dict[tuple[str, str], int] = {}
    for version, reserve, _, tick in final_by_tick:
        key = (version, reserve)
        reserve_starts[key] = min(reserve_starts.get(key, tick), tick)
    return output, reserve_starts


def _market_exposure_events(
    raw_root: Path,
    debt_intervals: Iterable[tuple[int, str, str, int, int]],
    *,
    include_reserve_starts: bool = False,
) -> tuple[set[tuple[int, int, int, int]], dict[str, int]] | tuple[
    set[tuple[int, int, int, int]], dict[str, int], dict[tuple[str, str], int]
]:
    """Map rate changes only to conservatively confirmed debt exposure.

    ``debt_intervals`` are half-open and use nominal on-chain flow as a lower
    bound: a positive balance proves exposure; an uncertain nonpositive balance
    emits no market predicate.  A change at the entry tick is excluded so that
    the exposure must predate the market event.
    """

    directions, reserve_starts = _market_tick_directions(raw_root)
    predicate_by_signal = {
        "variable_borrow_rate": (
            PREDICATE_INDEX["pred_active_debt_reserve_variable_rate_upward_shock"],
            PREDICATE_INDEX["pred_active_debt_reserve_variable_rate_downward_shock"],
        ),
        "liquidity_rate": (
            PREDICATE_INDEX["pred_active_debt_reserve_liquidity_rate_upward_shock"],
            PREDICATE_INDEX["pred_active_debt_reserve_liquidity_rate_downward_shock"],
        ),
    }
    # Collapse multiple active debt reserves moving in the same direction on
    # the same day to one interpretable wallet-level market event.
    rows: dict[tuple[int, int, int], int] = {}
    intervals = 0
    for entity, version, reserve, start, end in debt_intervals:
        intervals += 1
        for signal, (increase, decrease) in predicate_by_signal.items():
            tick_direction = directions.get((version, reserve, signal))
            if tick_direction is None or end <= start:
                continue
            ticks, signs, primitives = tick_direction
            left = int(np.searchsorted(ticks, start, side="right"))
            right = int(np.searchsorted(ticks, end, side="left"))
            for tick, sign, primitive in zip(
                ticks[left:right],
                signs[left:right],
                primitives[left:right],
                strict=True,
            ):
                code = increase if int(sign) > 0 else decrease
                key = (int(entity), int(tick), code)
                old = rows.get(key)
                rows[key] = int(primitive) if old is None else min(old, int(primitive))
    events = {
        (entity, tick, code, primitive)
        for (entity, tick, code), primitive in rows.items()
    }
    audit = {
        "market_reserve_series": len(directions),
        "confirmed_debt_intervals": intervals,
        "market_predicate_events": len(events),
    }
    return (events, audit, reserve_starts) if include_reserve_starts else (events, audit)


def _oracle_cache_metadata(
    raw_root: Path,
    *,
    version: str,
    reserve: str,
    tick: int,
    end_block: int,
) -> tuple[Path, str, int]:
    oracle = AAVE_ORACLE_ADDRESSES[version]
    block = min(int(end_block), (int(tick) + 1) * BLOCKS_PER_TICK - 1)
    path = raw_root / "oracle_cache" / "v12" / version / reserve / f"{tick}.json"
    return path, oracle, block


def _read_oracle_cache_record(
    raw_root: Path,
    *,
    version: str,
    reserve: str,
    tick: int,
    end_block: int,
) -> dict[str, object] | None:
    path, oracle, block = _oracle_cache_metadata(
        raw_root,
        version=version,
        reserve=reserve,
        tick=tick,
        end_block=end_block,
    )
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "crbstpp.aave_oracle_price.v1",
        "version": version,
        "tick": int(tick),
        "block": block,
        "oracle_address": oracle,
        "asset": reserve,
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError(f"immutable oracle cache metadata mismatch: {path}")
    response = str(record.get("rpc_response", ""))
    if hashlib.sha256(response.encode("utf-8")).hexdigest() != record.get(
        "rpc_response_digest"
    ):
        raise ValueError(f"immutable oracle cache digest mismatch: {path}")
    status = str(record.get("status", "ok"))
    if status == "ok":
        if int(response, 16) != int(record.get("value", -1)):
            raise ValueError(f"immutable oracle cache value mismatch: {path}")
    elif status == "unavailable":
        if record.get("value") is not None:
            raise ValueError(f"unavailable oracle cache has a value: {path}")
    else:
        raise ValueError(f"unknown immutable oracle cache status: {path}")
    return record


def _write_oracle_cache_record(
    raw_root: Path,
    *,
    version: str,
    reserve: str,
    tick: int,
    end_block: int,
    result: object,
) -> dict[str, object]:
    path, oracle, block = _oracle_cache_metadata(
        raw_root,
        version=version,
        reserve=reserve,
        tick=tick,
        end_block=end_block,
    )
    if isinstance(result, Exception):
        response = json.dumps(
            {"error": str(result)}, sort_keys=True, separators=(",", ":")
        )
        status = "unavailable"
        value: int | None = None
    else:
        response = str(result).lower()
        if not response.startswith("0x"):
            raise ValueError("Aave Oracle returned a non-hex response")
        status = "ok"
        value = int(response, 16)
    record = {
        "schema": "crbstpp.aave_oracle_price.v1",
        "version": version,
        "tick": int(tick),
        "block": block,
        "oracle_address": oracle,
        "asset": reserve,
        "status": status,
        "value": value,
        "rpc_response": response,
        "rpc_response_digest": hashlib.sha256(response.encode("utf-8")).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)
    loaded = _read_oracle_cache_record(
        raw_root,
        version=version,
        reserve=reserve,
        tick=tick,
        end_block=end_block,
    )
    assert loaded is not None
    return loaded


def _oracle_cache_record(
    raw_root: Path,
    *,
    version: str,
    reserve: str,
    tick: int,
    end_block: int,
    rpc_urls: tuple[str, ...],
) -> dict[str, object]:
    """Return one immutable historical Aave Oracle response."""

    cached = _read_oracle_cache_record(
        raw_root,
        version=version,
        reserve=reserve,
        tick=tick,
        end_block=end_block,
    )
    if cached is not None:
        return cached
    _, oracle, block = _oracle_cache_metadata(
        raw_root,
        version=version,
        reserve=reserve,
        tick=tick,
        end_block=end_block,
    )
    data = "0x" + GET_ASSET_PRICE_SELECTOR + reserve[2:].rjust(64, "0")
    try:
        result = _rpc(
            rpc_urls,
            "eth_call",
            [{"to": oracle, "data": data}, hex(block)],
            retries=5,
        )
    except RuntimeError as error:
        if "execution reverted" not in str(error).lower():
            raise
        result = error
    return _write_oracle_cache_record(
        raw_root,
        version=version,
        reserve=reserve,
        tick=tick,
        end_block=end_block,
        result=result,
    )


def _oracle_cache_records(
    raw_root: Path,
    jobs: list[tuple[str, str, int]],
    *,
    rpc_urls: tuple[str, ...],
    workers: int,
    end_block: int,
    batch_size: int = 100,
) -> list[tuple[str, str, int, int]]:
    """Read/fetch oracle jobs in deterministic order with batch fail-open."""

    records: list[dict[str, object] | None] = [None] * len(jobs)
    missing: list[int] = []
    for index, (version, reserve, tick) in enumerate(jobs):
        cached = _read_oracle_cache_record(
            raw_root,
            version=version,
            reserve=reserve,
            tick=tick,
            end_block=end_block,
        )
        records[index] = cached
        if cached is None:
            missing.append(index)

    chunks = [
        tuple(missing[left : left + max(1, int(batch_size))])
        for left in range(0, len(missing), max(1, int(batch_size)))
    ]

    def fetch_chunk(indices: tuple[int, ...]) -> list[tuple[int, dict[str, object]]]:
        calls: list[tuple[str, list[object]]] = []
        for index in indices:
            version, reserve, tick = jobs[index]
            _, oracle, block = _oracle_cache_metadata(
                raw_root,
                version=version,
                reserve=reserve,
                tick=tick,
                end_block=end_block,
            )
            data = "0x" + GET_ASSET_PRICE_SELECTOR + reserve[2:].rjust(64, "0")
            calls.append(("eth_call", [{"to": oracle, "data": data}, hex(block)]))
        try:
            responses = _rpc_batch(rpc_urls, calls, retries=5)
        except (requests.RequestException, ValueError, RuntimeError):
            # Public providers may disable batch JSON-RPC.  Falling back to the
            # exact single-call path changes performance only, never data.
            return [
                (
                    index,
                    _oracle_cache_record(
                        raw_root,
                        version=jobs[index][0],
                        reserve=jobs[index][1],
                        tick=jobs[index][2],
                        end_block=end_block,
                        rpc_urls=rpc_urls,
                    ),
                )
                for index in indices
            ]
        output: list[tuple[int, dict[str, object]]] = []
        for index, response in zip(indices, responses, strict=True):
            if isinstance(response, Exception):
                if "execution reverted" in str(response).lower():
                    record = _write_oracle_cache_record(
                        raw_root,
                        version=jobs[index][0],
                        reserve=jobs[index][1],
                        tick=jobs[index][2],
                        end_block=end_block,
                        result=response,
                    )
                else:
                    record = _oracle_cache_record(
                        raw_root,
                        version=jobs[index][0],
                        reserve=jobs[index][1],
                        tick=jobs[index][2],
                        end_block=end_block,
                        rpc_urls=rpc_urls,
                    )
            else:
                record = _write_oracle_cache_record(
                    raw_root,
                    version=jobs[index][0],
                    reserve=jobs[index][1],
                    tick=jobs[index][2],
                    end_block=end_block,
                    result=response,
                )
            output.append((index, record))
        return output

    if chunks:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max(1, int(workers)), len(chunks))
        ) as pool:
            for result in pool.map(fetch_chunk, chunks):
                for index, record in result:
                    records[index] = record
    if any(record is None for record in records):
        raise AssertionError("oracle batch left an unresolved job")
    return [
        (version, reserve, tick, int(record["value"]))
        for (version, reserve, tick), record in zip(jobs, records, strict=True)
        if record is not None and record.get("value") is not None
    ]


def _oracle_tick_directions(
    raw_root: Path,
    intervals: Iterable[tuple[int, str, str, int, int]],
    *,
    rpc_urls: tuple[str, ...],
    workers: int,
    end_block: int,
    reserve_start_ticks: dict[tuple[str, str], int] | None = None,
) -> tuple[
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    dict[str, int],
]:
    """Build target-blind price shocks over every exposed asset's full span."""
    ranges: dict[tuple[str, str], tuple[int, int]] = {}
    for _, version, reserve, start, end in intervals:
        if end <= start + 1:
            continue
        key = (version, reserve)
        old = ranges.get(key)
        # The prequential rank at an exposure date must use all price changes
        # observable since that protocol version entered the registered raw
        # range, not merely changes since this wallet first became exposed.
        # No target or future observation enters this history.
        history_start = (
            (reserve_start_ticks or {}).get((version, reserve))
            if reserve_start_ticks is not None
            else None
        )
        if history_start is None:
            history_start = int(start)
        bounds = (min(history_start, int(start)), int(end))
        ranges[key] = bounds if old is None else (
            min(old[0], bounds[0]),
            max(old[1], bounds[1]),
        )
    jobs = [
        (version, reserve, tick)
        for (version, reserve), (start, end) in sorted(ranges.items())
        for tick in range(start, end)
    ]

    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for version, reserve, tick, value in _oracle_cache_records(
        raw_root,
        jobs,
        rpc_urls=rpc_urls,
        workers=workers,
        end_block=end_block,
    ):
        if value > 0:
            grouped.setdefault((version, reserve), []).append((tick, value))
    directions = {
        key: _online_rank_shocks(values) for key, values in grouped.items()
    }
    return directions, {
        "oracle_assets": len(grouped),
        "oracle_historical_calls_or_cache_hits": len(jobs),
        "oracle_price_shocks": int(sum(len(value[0]) for value in directions.values())),
    }


def _price_primitive_id(version: str, reserve: str, tick: int) -> int:
    digest = hashlib.sha256(f"oracle:{version}:{reserve}:{tick}".encode()).digest()
    return (3 << 61) | (int.from_bytes(digest[:8], "big") & ((1 << 61) - 1))


def _price_exposure_events(
    raw_root: Path,
    debt_intervals: Iterable[tuple[int, str, str, int, int]],
    collateral_intervals: Iterable[tuple[int, str, str, int, int]],
    *,
    rpc_urls: tuple[str, ...],
    workers: int,
    end_block: int,
    reserve_start_ticks: dict[tuple[str, str], int] | None = None,
) -> tuple[set[tuple[int, int, int, int]], dict[str, int]]:
    debt = tuple(debt_intervals)
    collateral = tuple(collateral_intervals)
    directions, audit = _oracle_tick_directions(
        raw_root,
        (*debt, *collateral),
        rpc_urls=rpc_urls,
        workers=workers,
        end_block=end_block,
        reserve_start_ticks=reserve_start_ticks,
    )
    roles = (
        (
            collateral,
            PREDICATE_INDEX["pred_active_collateral_price_upward_shock"],
            PREDICATE_INDEX["pred_active_collateral_price_downward_shock"],
        ),
        (
            debt,
            PREDICATE_INDEX["pred_active_debt_asset_price_upward_shock"],
            PREDICATE_INDEX["pred_active_debt_asset_price_downward_shock"],
        ),
    )
    rows: dict[tuple[int, int, int], int] = {}
    for intervals, increase, decrease in roles:
        for entity, version, reserve, start, end in intervals:
            value = directions.get((version, reserve))
            if value is None:
                continue
            ticks, signs = value
            left = int(np.searchsorted(ticks, start, side="right"))
            right = int(np.searchsorted(ticks, end, side="left"))
            for tick, sign in zip(ticks[left:right], signs[left:right], strict=True):
                code = increase if int(sign) > 0 else decrease
                key = (int(entity), int(tick), code)
                primitive = _price_primitive_id(version, reserve, int(tick))
                old = rows.get(key)
                rows[key] = primitive if old is None else min(old, primitive)
    events = {
        (entity, tick, code, primitive)
        for (entity, tick, code), primitive in rows.items()
    }
    audit["oracle_price_predicate_events"] = len(events)
    audit["confirmed_collateral_intervals"] = len(collateral)
    return events, audit


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _debt_zero_candidates(raw_root: Path, cache_root: Path) -> pd.DataFrame:
    """Return a safe superset of blocks at which an account can become debt-free.

    Nominal borrow flow excludes accrued interest and is therefore a lower
    bound on actual debt.  A positive lower bound proves that debt remains;
    nonpositive points are resolved exactly with a historical state call.
    """

    output = cache_root / "debt_zero_candidates.parquet"
    if output.is_file():
        return pd.read_parquet(output)
    balance: dict[tuple[str, str, str], int] = {}
    positive_reserves: dict[tuple[str, str], int] = {}
    segment: dict[tuple[str, str], int] = {}
    borrowed: set[tuple[str, str]] = set()
    candidate_keys: set[tuple[str, str, int, int]] = set()
    for record in _iter_position_logs(raw_root):
        event_type = str(record["event_type"])
        if event_type not in {"borrow", "repay", "liquidation"}:
            continue
        version, owner, reserve = (
            str(record["version"]),
            str(record["owner"]),
            str(record["reserve"]),
        )
        owner_key = (version, owner)
        key = (version, owner, reserve)
        amount = int(record["amount"] or 0)
        previous = balance.get(key, 0)
        if event_type == "borrow":
            segment[owner_key] = segment.get(owner_key, -1) + 1
            updated = previous + amount
            balance[key] = updated
            if previous <= 0 < updated:
                positive_reserves[owner_key] = positive_reserves.get(owner_key, 0) + 1
            borrowed.add(owner_key)
            continue
        updated = previous - amount
        balance[key] = updated
        if previous > 0 >= updated:
            positive_reserves[owner_key] = positive_reserves.get(owner_key, 0) - 1
        if owner_key in borrowed and positive_reserves.get(owner_key, 0) == 0:
            candidate_keys.add(
                (version, owner, segment[owner_key], int(record["block"]))
            )
    candidates = pd.DataFrame(
        sorted(candidate_keys), columns=["version", "owner", "segment", "block"]
    )
    _write_parquet_atomic(candidates, output)
    return candidates


def _account_debt_at_block(
    version: str,
    owner: str,
    block: int,
    rpc_urls: tuple[str, ...],
) -> int:
    calldata = "0x" + GET_USER_ACCOUNT_DATA_SELECTOR + "0" * 24 + owner[2:]
    result = _rpc(
        rpc_urls,
        "eth_call",
        [{"to": DEPLOYMENTS[version].pool_address, "data": calldata}, hex(block)],
        retries=8,
        timeout=45.0,
    )
    payload = str(result)
    if not payload.startswith("0x") or len(payload) < 2 + 6 * 64:
        raise ValueError("getUserAccountData returned an incomplete response")
    # (totalCollateralBase, totalDebtBase, availableBorrowsBase, ...)
    return int(payload[2 + 64 : 2 + 2 * 64], 16)


def _resolve_debt_zero_points(
    candidates: pd.DataFrame,
    cache_root: Path,
    *,
    rpc_urls: tuple[str, ...],
    workers: int,
) -> pd.DataFrame:
    output = cache_root / "debt_state.parquet"
    parts_root = cache_root / "debt_state_parts"
    columns = [
        "version",
        "owner",
        "block",
        "total_debt_base",
        "debt_zero",
        "state_source",
    ]
    cached_parts: list[pd.DataFrame] = []
    if output.is_file():
        cached_parts.append(pd.read_parquet(output))
    if parts_root.is_dir():
        cached_parts.extend(
            pd.read_parquet(path) for path in sorted(parts_root.glob("part_*.parquet"))
        )
    cached = (
        pd.concat(cached_parts, ignore_index=True)
        if cached_parts
        else pd.DataFrame(columns=columns)
    )
    # A valid V2 debt balance is a uint256-denominated wei amount and may
    # exceed signed int64. Decimal text preserves it losslessly; only the exact
    # zero indicator is consumed by episode construction.
    cached["total_debt_base"] = cached["total_debt_base"].astype(str)
    cached = cached.drop_duplicates(["version", "owner", "block"], keep="last")
    known = {
        (str(row.version), str(row.owner), int(row.block)): (
            str(row.total_debt_base),
            bool(row.debt_zero),
            str(row.state_source),
        )
        for row in cached.itertuples(index=False)
    }
    groups = [
        (
            (str(version), str(owner), int(segment)),
            np.sort(group["block"].to_numpy(dtype=np.int64)),
        )
        for (version, owner, segment), group in candidates.groupby(
            ["version", "owner", "segment"], sort=True
        )
    ]
    batch_urls = tuple(
        dict.fromkeys(
            (
                *(
                    url
                    for url in rpc_urls
                    if "mevblocker.io" in url or "nodies.app" in url
                ),
                "https://rpc.mevblocker.io",
                "https://eth-pokt.nodies.app",
            )
        )
    )
    next_part = 0
    if parts_root.is_dir():
        indices = [
            int(path.stem.rsplit("_", 1)[1])
            for path in parts_root.glob("part_*.parquet")
        ]
        next_part = max(indices, default=-1) + 1

    def query_batch(
        keys: list[tuple[str, str, int]],
    ) -> list[tuple[str, str, int, str, bool, str]]:
        payload = []
        for identity, (version, owner, block) in enumerate(keys):
            calldata = "0x" + GET_USER_ACCOUNT_DATA_SELECTOR + "0" * 24 + owner[2:]
            payload.append(
                {
                    "jsonrpc": "2.0",
                    "id": identity,
                    "method": "eth_call",
                    "params": [
                        {"to": DEPLOYMENTS[version].pool_address, "data": calldata},
                        hex(block),
                    ],
                }
            )
        last_error: Exception | None = None
        for attempt in range(5):
            for url in batch_urls:
                try:
                    response = requests.post(url, json=payload, timeout=90.0)
                    response.raise_for_status()
                    body = response.json()
                    if not isinstance(body, list) or len(body) != len(keys):
                        raise ValueError("incomplete JSON-RPC batch response")
                    by_id = {int(item["id"]): item for item in body}
                    rows = []
                    for identity, key in enumerate(keys):
                        item = by_id.get(identity)
                        if item is None or "result" not in item:
                            raise RuntimeError(str(item))
                        result = str(item["result"])
                        if not result.startswith("0x") or len(result) < 2 + 6 * 64:
                            raise ValueError("incomplete getUserAccountData result")
                        debt = int(result[2 + 64 : 2 + 2 * 64], 16)
                        rows.append((*key, str(debt), debt == 0, "historical_eth_call"))
                    return rows
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    last_error = error
            if attempt + 1 < 5:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        assert last_error is not None
        # Fail open to the already validated exact singleton RPC path. This is
        # slower, never an approximation, and preserves resumability.
        rows = []
        for version, owner, block in keys:
            debt = _account_debt_at_block(version, owner, block, rpc_urls)
            rows.append(
                (version, owner, block, str(debt), debt == 0, "historical_eth_call")
            )
        return rows

    def persist(rows: list[tuple[str, str, int, str, bool, str]]) -> None:
        nonlocal next_part
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=columns).drop_duplicates(
            ["version", "owner", "block"], keep="last"
        )
        path = parts_root / f"part_{next_part:06d}.parquet"
        _write_parquet_atomic(frame, path)
        next_part += 1
        for row in frame.itertuples(index=False):
            known[(str(row.version), str(row.owner), int(row.block))] = (
                str(row.total_debt_base),
                bool(row.debt_zero),
                str(row.state_source),
            )

    def fetch_missing(keys: Iterable[tuple[str, str, int]]) -> None:
        missing = sorted({key for key in keys if key not in known})
        request_width = 200
        checkpoint_width = 3_200
        request_workers = min(4, max(1, int(workers)))
        for left in range(0, len(missing), checkpoint_width):
            checkpoint = missing[left : left + checkpoint_width]
            batches = [
                checkpoint[index : index + request_width]
                for index in range(0, len(checkpoint), request_width)
            ]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=request_workers
            ) as pool:
                persist(
                    [row for rows in pool.map(query_batch, batches) for row in rows]
                )

    # Round 1 queries the last possible close in every Borrow-delimited segment.
    # If it is nonzero, no earlier candidate in that segment can be zero.
    fetch_missing(
        (version, owner, int(blocks[-1]))
        for (version, owner, _), blocks in groups
        if len(blocks)
    )
    active_groups: list[tuple[str, str, np.ndarray, int, int]] = []
    for (version, owner, _), blocks in groups:
        if len(blocks) and known[(version, owner, int(blocks[-1]))][1]:
            active_groups.append((version, owner, blocks, 0, len(blocks) - 1))

    # Batched binary-search rounds locate the first exact zero in each segment.
    while any(left < right for _, _, _, left, right in active_groups):
        fetch_missing(
            (version, owner, int(blocks[(left + right) // 2]))
            for version, owner, blocks, left, right in active_groups
            if left < right
        )
        updated = []
        for version, owner, blocks, left, right in active_groups:
            if left < right:
                middle = (left + right) // 2
                if known[(version, owner, int(blocks[middle]))][1]:
                    right = middle
                else:
                    left = middle + 1
            updated.append((version, owner, blocks, left, right))
        active_groups = updated

    inferred = []
    for version, owner, blocks, first_zero, _ in active_groups:
        for block_value in blocks[first_zero:]:
            key = (version, owner, int(block_value))
            if key not in known:
                inferred.append((*key, "inferred_zero", True, "monotone_after_zero"))
    persist(inferred)
    final = pd.DataFrame(
        [(*key, *value) for key, value in known.items()], columns=columns
    ).sort_values(["version", "owner", "block"], kind="stable")
    _write_parquet_atomic(final, output)
    for path in parts_root.glob("part_*.parquet"):
        path.unlink()
    if parts_root.is_dir():
        parts_root.rmdir()
    return final


def _wallet_partition(
    owner: str,
    seed: int,
    fractions: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> int:
    fractions = tuple(map(float, fractions))
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("partition fractions must contain three positive values")
    if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("partition fractions must sum to one")
    value = int.from_bytes(
        hashlib.sha256(f"{int(seed)}:{owner}".encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    first = int(fractions[0] * (1 << 64))
    second = int((fractions[0] + fractions[1]) * (1 << 64))
    return 0 if value < first else (1 if value < second else 2)


def _mechanism_predicate(
    record: dict[str, object],
    *,
    active_collateral_reserves: frozenset[str] | set[str],
    seen_debt_reserves: frozenset[str] | set[str],
) -> int | None:
    """Classify one atomic action from frozen, target-blind state.

    This helper does not decide whether several logs form a compound economic
    action.  :func:`_transaction_mechanism_predicate` performs that aggregation
    and is the only classifier used by the fit-ready preprocessor.
    """

    event_type = str(record["event_type"])
    owner = str(record["owner"])
    actor = str(record["actor"])
    reserve = str(record["reserve"])
    self_action = owner == actor

    if event_type == "borrow":
        existing = reserve in seen_debt_reserves
        if self_action:
            if int(record.get("rate_mode") or 0) == 1:
                return PREDICATE_INDEX["pred_self_stable_rate_borrow"]
            name = (
                "pred_self_variable_borrow_previously_used_debt_reserve"
                if existing
                else "pred_self_variable_borrow_new_debt_reserve"
            )
            return PREDICATE_INDEX[name]
        name = (
            "pred_delegated_borrow_previously_used_debt_reserve"
            if existing
            else "pred_delegated_borrow_new_debt_reserve"
        )
        return PREDICATE_INDEX[name]
    if event_type == "repay":
        actor = "self" if self_action else "third_party"
        return PREDICATE_INDEX[f"pred_{actor}_repay_while_debt_remains"]
    if event_type == "supply":
        if reserve not in active_collateral_reserves:
            return None
        actor = "self" if self_action else "third_party"
        return PREDICATE_INDEX[f"pred_{actor}_collateral_top_up_without_borrow"]
    if event_type == "withdraw":
        if reserve not in active_collateral_reserves:
            return None
        return PREDICATE_INDEX[
            (
                "pred_collateral_withdrawal_with_alternative_remaining"
                if len(active_collateral_reserves) > 1
                else "pred_last_collateral_withdrawal_while_debt_remains"
            )
        ]
    if event_type == "collateral_enable":
        if reserve in active_collateral_reserves:
            return None
        return PREDICATE_INDEX["pred_collateral_reserve_enabled_without_borrow"]
    if event_type == "collateral_disable":
        if reserve not in active_collateral_reserves:
            return None
        return PREDICATE_INDEX["pred_collateral_reserve_disabled"]
    # Rate-mode swaps are execution-method changes, not distinct balance-sheet
    # actions.  Their two sparse identities previously expanded the dictionary
    # without surviving held-out reliability, so they remain provenance events
    # but are not reported rules.
    if event_type == "rate_mode_swap":
        return None
    return None


def _transaction_mechanism_predicate(
    records: list[dict[str, object]],
    *,
    active_collateral_reserves: frozenset[str] | set[str],
    seen_debt_reserves: frozenset[str] | set[str],
) -> int | None:
    """Return one non-overlapping financial mechanism for a wallet transaction.

    Same-transaction Pool logs are attributes of one primitive financial
    decision, not temporal witnesses.  Economically recognisable multi-leg
    decisions receive one compound identity.  Ambiguous multi-action patterns
    are excluded rather than pooled into a heterogeneous catch-all predicate.
    """

    actions = [
        record
        for record in records
        if str(record["event_type"]) in POSITION_ACTION_TYPES
    ]
    if not actions:
        return None

    borrows = [record for record in actions if record["event_type"] == "borrow"]
    repays = [record for record in actions if record["event_type"] == "repay"]
    additions = [
        record
        for record in actions
        if record["event_type"] == "collateral_enable"
        or (
            record["event_type"] == "supply"
            and str(record["reserve"]) in active_collateral_reserves
        )
    ]
    removals = [
        record
        for record in actions
        if record["event_type"] == "collateral_disable"
        or (
            record["event_type"] == "withdraw"
            and str(record["reserve"]) in active_collateral_reserves
        )
    ]
    addition_reserves = {str(record["reserve"]) for record in additions}
    removal_reserves = {str(record["reserve"]) for record in removals}
    compound: list[str] = []
    if borrows and repays:
        compound.append("pred_debt_restructuring_transaction")
    if (
        addition_reserves
        and removal_reserves
        and (
            addition_reserves != removal_reserves
            or len(addition_reserves | removal_reserves) > 1
        )
    ):
        compound.append("pred_collateral_rotation_transaction")
    if borrows and additions:
        compound.append("pred_leveraged_position_expansion_transaction")
    if repays and additions:
        owner = str(actions[0]["owner"])
        externally_assisted = any(
            str(record["actor"]) != owner for record in (*repays, *additions)
        )
        compound.append(
            "pred_third_party_position_deleveraging_transaction"
            if externally_assisted
            else "pred_self_position_deleveraging_transaction"
        )
    if len(compound) == 1:
        return PREDICATE_INDEX[compound[0]]
    if len(compound) > 1:
        return None

    # A Supply immediately followed by enablement, or a Withdraw immediately
    # followed by disablement, describes one collateral transition.  Let the
    # enable/disable log carry that identity and suppress the redundant flow
    # attribute before checking whether the transaction is otherwise atomic.
    enabled = {
        str(record["reserve"])
        for record in actions
        if record["event_type"] == "collateral_enable"
    }
    disabled = {
        str(record["reserve"])
        for record in actions
        if record["event_type"] == "collateral_disable"
    }
    codes: set[int] = set()
    for record in actions:
        if record["event_type"] == "supply" and str(record["reserve"]) in enabled:
            continue
        if record["event_type"] == "withdraw" and str(record["reserve"]) in disabled:
            continue
        code = _mechanism_predicate(
            record,
            active_collateral_reserves=active_collateral_reserves,
            seen_debt_reserves=seen_debt_reserves,
        )
        if code is not None:
            codes.add(code)
    if not codes:
        return None
    if len(codes) == 1:
        return next(iter(codes))
    return None


def _entry_mechanism_predicate(
    records: list[dict[str, object]],
    *,
    active_collateral_reserves: frozenset[str] | set[str],
) -> int | None:
    """Mark one target-blind financial configuration at risk-set entry.

    Borrow occurrence itself defines the debt-episode risk set and is therefore
    not a reportable rule.  Its observed mark is not constant, however: actor,
    rate mode, debt-asset class and same-transaction collateral expansion are
    legitimate baseline-time financial events.  Exactly one identity is
    emitted and its kernel remains strictly future, so the opening event can
    never explain a liquidation in the same chain-time tick.
    """

    borrows = [record for record in records if record["event_type"] == "borrow"]
    if not borrows:
        return None
    additions = [
        record
        for record in records
        if record["event_type"] == "collateral_enable"
        or (
            record["event_type"] == "supply"
            and str(record["reserve"]) in active_collateral_reserves
        )
    ]
    if additions:
        return PREDICATE_INDEX["pred_entry_leveraged_position"]

    # Multiple Borrow logs for one owner/transaction are one primitive entry.
    # Prefer the more structurally distinctive categories deterministically;
    # otherwise classify the first log in canonical log order.
    if any(str(record["actor"]) != str(record["owner"]) for record in borrows):
        return PREDICATE_INDEX["pred_entry_delegated_debt"]
    if any(int(record.get("rate_mode") or 0) == 1 for record in borrows):
        return PREDICATE_INDEX["pred_entry_self_stable_rate_debt"]
    first = min(borrows, key=lambda record: int(record["log_index"]))
    name = (
        "pred_entry_self_variable_stable_value_debt"
        if str(first["reserve"]) in STABLE_VALUE_RESERVES
        else "pred_entry_self_variable_volatile_debt"
    )
    return PREDICATE_INDEX[name]


def _episode_tables(
    raw_root: Path,
    *,
    zero_state: pd.DataFrame,
    partition_seed: int,
    partition_fractions: tuple[float, float, float] = (0.5, 0.3, 0.2),
    rpc_urls: tuple[str, ...] = STATE_RPC_URLS,
    workers: int = 8,
    include_oracle_prices: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_manifest = json.loads((raw_root / "manifest.json").read_text())
    protocol_end = int(raw_manifest["end_block"])
    close_by_block: dict[tuple[str, int], list[str]] = {}
    for row in zero_state.loc[zero_state["debt_zero"].astype(bool)].itertuples(
        index=False
    ):
        close_by_block.setdefault((str(row.version), int(row.block)), []).append(
            str(row.owner)
        )
    active: dict[tuple[str, str], int] = {}
    episode_number: dict[tuple[str, str], int] = {}
    entity_rows: list[dict[str, object]] = []
    open_rows: dict[int, dict[str, object]] = {}
    reported_event_rows: set[tuple[int, int, int, int]] = set()
    first_target: dict[int, int] = {}
    entry_primitive: dict[tuple[str, str], int] = {}
    active_collateral_by_owner: dict[tuple[str, str], set[str]] = {}
    debt_reserves_seen_by_owner: dict[tuple[str, str], set[str]] = {}
    confirmed_debt_balance: dict[tuple[str, str, str], int] = {}
    confirmed_debt_start: dict[tuple[str, str, str], tuple[int, int]] = {}
    confirmed_debt_keys_by_owner: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    debt_intervals: list[tuple[int, str, str, int, int]] = []
    collateral_start: dict[tuple[str, str, str], tuple[int, int]] = {}
    collateral_keys_by_owner: dict[
        tuple[str, str], set[tuple[str, str, str]]
    ] = {}
    collateral_intervals: list[tuple[int, str, str, int, int]] = []
    decoded_count = 0
    skipped_liquidations = 0
    observed_liquidation_calls = 0
    excluded_pre_entry_logs = 0
    marked_entry_transactions = 0
    excluded_terminal_block_logs = 0
    excluded_post_target_logs = 0
    raw_audit: dict[str, int] = {}
    next_entity = 0

    def close_confirmed_debt(owner_key: tuple[str, str], end_tick: int) -> None:
        for key in tuple(confirmed_debt_keys_by_owner.pop(owner_key, ())):
            opened = confirmed_debt_start.pop(key, None)
            if opened is not None and int(end_tick) > opened[1]:
                debt_intervals.append(
                    (opened[0], key[0], key[2], opened[1], int(end_tick))
                )
            confirmed_debt_balance.pop(key, None)

    def close_collateral(owner_key: tuple[str, str], end_tick: int) -> None:
        for key in tuple(collateral_keys_by_owner.pop(owner_key, ())):
            opened = collateral_start.pop(key, None)
            if opened is not None and int(end_tick) > opened[1]:
                collateral_intervals.append(
                    (opened[0], key[0], key[2], opened[1], int(end_tick))
                )

    def process_block(records: list[dict[str, object]]) -> None:
        nonlocal next_entity, decoded_count, skipped_liquidations
        nonlocal observed_liquidation_calls
        nonlocal excluded_pre_entry_logs, marked_entry_transactions
        nonlocal excluded_terminal_block_logs, excluded_post_target_logs
        if not records:
            return
        records = sorted(
            records,
            key=lambda value: (
                int(value["primitive_event_id"]),
                int(value["log_index"]),
            ),
        )
        block = int(records[0]["block"])
        tick = block // BLOCKS_PER_TICK
        closing_owners = frozenset(
            close_by_block.get((str(records[0]["version"]), block), ())
        )

        left = 0
        while left < len(records):
            primitive = int(records[left]["primitive_event_id"])
            right = left + 1
            while (
                right < len(records)
                and int(records[right]["primitive_event_id"]) == primitive
            ):
                right += 1
            transaction = records[left:right]

            # Open at the first Borrow transaction.  The full opening
            # transaction defines entry into the risk set and is not a rule.
            for record in transaction:
                if record["event_type"] != "borrow":
                    continue
                owner_key = (str(record["version"]), str(record["owner"]))
                if owner_key in active:
                    continue
                episode = episode_number.get(owner_key, 0)
                episode_number[owner_key] = episode + 1
                code = next_entity
                next_entity += 1
                active[owner_key] = code
                entry_primitive[owner_key] = primitive
                open_rows[code] = {
                    "_entity_code": code,
                    "entity_id": f"{owner_key[0]}:{owner_key[1]}:{episode}",
                    # V2/V3 and repeated debt episodes of one on-chain wallet
                    # are one dependency cluster, while their baseline strata
                    # and point-process risk intervals remain distinct.
                    "dependency_group": owner_key[1],
                    "start_time": tick,
                    "end_time": protocol_end // BLOCKS_PER_TICK,
                    "baseline_origin": tick,
                    "split_group": 0,
                    "baseline_stratum": 0 if owner_key[0] == "v2" else 1,
                    "partition": _wallet_partition(
                        owner_key[1], partition_seed, partition_fractions
                    ),
                    "end_reason": "protocol_observation_end",
                }
                for reserve in active_collateral_by_owner.get(owner_key, ()):
                    collateral_key = (owner_key[0], owner_key[1], reserve)
                    collateral_start[collateral_key] = (code, tick)
                    collateral_keys_by_owner.setdefault(owner_key, set()).add(
                        collateral_key
                    )

            decoded_count += len(transaction)
            for record in transaction:
                if record["event_type"] != "liquidation":
                    continue
                owner_key = (str(record["version"]), str(record["owner"]))
                code = active.get(owner_key)
                if code is None:
                    skipped_liquidations += 1
                else:
                    observed_liquidation_calls += 1
                    if code not in first_target:
                        first_target[code] = tick
                        close_confirmed_debt(owner_key, tick)
                        close_collateral(owner_key, tick)

            actions_by_owner: dict[tuple[str, str], list[dict[str, object]]] = {}
            for record in transaction:
                if str(record["event_type"]) not in POSITION_ACTION_TYPES:
                    continue
                owner_key = (str(record["version"]), str(record["owner"]))
                actions_by_owner.setdefault(owner_key, []).append(record)
            for owner_key, owner_actions in actions_by_owner.items():
                code = active.get(owner_key)
                action_count = len(owner_actions)
                if code is None:
                    excluded_pre_entry_logs += action_count
                    continue
                if code in first_target:
                    excluded_post_target_logs += action_count
                    continue
                if primitive == entry_primitive[owner_key]:
                    predicate = _entry_mechanism_predicate(
                        owner_actions,
                        active_collateral_reserves=frozenset(
                            active_collateral_by_owner.get(owner_key, ())
                        ),
                    )
                    if predicate is not None:
                        reported_event_rows.add((code, tick, predicate, primitive))
                        marked_entry_transactions += 1
                    continue
                if owner_key[1] in closing_owners:
                    # No strictly-future risk row exists after this block.
                    excluded_terminal_block_logs += action_count
                    continue
                predicate = _transaction_mechanism_predicate(
                    owner_actions,
                    active_collateral_reserves=frozenset(
                        active_collateral_by_owner.get(owner_key, ())
                    ),
                    seen_debt_reserves=frozenset(
                        debt_reserves_seen_by_owner.get(owner_key, ())
                    ),
                )
                if predicate is not None:
                    reported_event_rows.add((code, tick, predicate, primitive))

            # Freeze state within a transaction and update it only afterward.
            for record in transaction:
                owner_key = (str(record["version"]), str(record["owner"]))
                reserve = str(record["reserve"])
                event_type = str(record["event_type"])
                code = active.get(owner_key)
                active_collateral = active_collateral_by_owner.setdefault(
                    owner_key, set()
                )
                collateral_key = (owner_key[0], owner_key[1], reserve)
                if event_type == "collateral_enable":
                    was_active = reserve in active_collateral
                    active_collateral.add(reserve)
                    if (
                        not was_active
                        and code is not None
                        and code not in first_target
                    ):
                        collateral_start[collateral_key] = (code, tick)
                        collateral_keys_by_owner.setdefault(owner_key, set()).add(
                            collateral_key
                        )
                elif event_type == "collateral_disable":
                    active_collateral.discard(reserve)
                    opened = collateral_start.pop(collateral_key, None)
                    collateral_keys_by_owner.get(owner_key, set()).discard(
                        collateral_key
                    )
                    if opened is not None and tick > opened[1]:
                        collateral_intervals.append(
                            (opened[0], owner_key[0], reserve, opened[1], tick)
                        )
                if code is None or code in first_target:
                    continue
                amount = int(record["amount"] or 0)
                debt_key = (owner_key[0], owner_key[1], reserve)
                if event_type == "borrow" and amount > 0:
                    debt_reserves_seen_by_owner.setdefault(owner_key, set()).add(
                        reserve
                    )
                    previous = confirmed_debt_balance.get(debt_key, 0)
                    confirmed_debt_balance[debt_key] = previous + amount
                    if previous <= 0:
                        confirmed_debt_start[debt_key] = (code, tick)
                        confirmed_debt_keys_by_owner.setdefault(owner_key, set()).add(
                            debt_key
                        )
                elif event_type in {"repay", "liquidation"} and amount > 0:
                    previous = confirmed_debt_balance.get(debt_key, 0)
                    if previous > 0:
                        updated = previous - amount
                        confirmed_debt_balance[debt_key] = updated
                        if updated <= 0:
                            opened = confirmed_debt_start.pop(debt_key, None)
                            if opened is not None and tick > opened[1]:
                                debt_intervals.append(
                                    (opened[0], debt_key[0], reserve, opened[1], tick)
                                )
                            confirmed_debt_balance.pop(debt_key, None)
                            confirmed_debt_keys_by_owner.get(owner_key, set()).discard(
                                debt_key
                            )
            left = right

        version = str(records[0]["version"])
        for owner in close_by_block.get((version, block), ()):
            owner_key = (version, owner)
            code = active.get(owner_key)
            if code is None:
                continue
            row = open_rows.pop(code)
            row["end_time"] = tick
            row["end_reason"] = "verified_zero_debt"
            entity_rows.append(row)
            close_confirmed_debt(owner_key, tick)
            close_collateral(owner_key, tick)
            del active[owner_key]
            entry_primitive.pop(owner_key, None)
            debt_reserves_seen_by_owner.pop(owner_key, None)

    block_records: list[dict[str, object]] = []
    block_key: tuple[str, int] | None = None
    for record in _iter_position_logs(raw_root, audit=raw_audit):
        key = (str(record["version"]), int(record["block"]))
        if block_key is not None and key != block_key:
            process_block(block_records)
            block_records = []
        block_key = key
        block_records.append(record)
    process_block(block_records)
    for code, row in sorted(open_rows.items()):
        owner_key = tuple(str(row["entity_id"]).split(":", 2)[:2])
        close_confirmed_debt((owner_key[0], owner_key[1]), int(row["end_time"]))
        close_collateral((owner_key[0], owner_key[1]), int(row["end_time"]))
        entity_rows.append(row)

    entities = pd.DataFrame(entity_rows)
    if entities.empty:
        raise ValueError("Aave preprocessing produced no debt episodes")
    entities = entities.sort_values("_entity_code", kind="stable").reset_index(
        drop=True
    )
    if not np.array_equal(
        entities.pop("_entity_code").to_numpy(dtype=np.int64),
        np.arange(next_entity, dtype=np.int64),
    ):
        raise AssertionError("Aave entity allocation is not contiguous")
    partitions = entities["partition"].to_numpy(dtype=np.int8)
    if set(partitions.tolist()) != {0, 1, 2}:
        raise ValueError("Aave wallet partition produced an empty split")
    market_rows, market_audit, reserve_start_ticks = _market_exposure_events(
        raw_root, debt_intervals, include_reserve_starts=True
    )
    if include_oracle_prices:
        price_rows, price_audit = _price_exposure_events(
            raw_root,
            debt_intervals,
            collateral_intervals,
            rpc_urls=rpc_urls,
            workers=workers,
            end_block=protocol_end,
            reserve_start_ticks=reserve_start_ticks,
        )
    else:
        price_rows, price_audit = set(), {
            "oracle_assets": 0,
            "oracle_historical_calls_or_cache_hits": 0,
            "oracle_price_shocks": 0,
            "oracle_price_predicate_events": 0,
            "confirmed_collateral_intervals": len(collateral_intervals),
        }
    event_rows = set(reported_event_rows)
    event_rows.update(market_rows)
    event_rows.update(price_rows)
    events = pd.DataFrame(
        sorted(event_rows),
        columns=["entity_code", "time", "predicate_code", "primitive_event_id"],
    )
    maximum_per_primitive = 0
    if len(events):
        reported = events.loc[events["predicate_code"] < len(PREDICATES)]
        primitive_ticks = reported.groupby(
            ["entity_code", "primitive_event_id"], sort=False
        )["time"].nunique()
        if int(primitive_ticks.max()) > 1:
            raise AssertionError(
                "one primitive event cannot occur at multiple temporal ticks"
            )
        maximum = reported.groupby(
            ["entity_code", "primitive_event_id"], sort=False
        )["predicate_code"].nunique().max()
        maximum_per_primitive = 0 if pd.isna(maximum) else int(maximum)
    # First-event survival semantics require sources to be strictly earlier
    # than the outcome.  This also removes same-block predicate logs regardless
    # of their log order, which the integer block clock cannot distinguish.
    if len(events) and first_target:
        event_codes = events["entity_code"].to_numpy(dtype=np.int32)
        event_times = events["time"].to_numpy(dtype=np.int64)
        cutoffs = np.asarray(
            [
                first_target.get(int(code), np.iinfo(np.int64).max)
                for code in event_codes
            ],
            dtype=np.int64,
        )
        events = events.loc[event_times < cutoffs].reset_index(drop=True)
    for code, tick in first_target.items():
        entities.loc[code, "end_time"] = int(tick)
        entities.loc[code, "end_reason"] = "first_liquidation"
    targets = pd.DataFrame(
        [(code, tick, 1) for code, tick in sorted(first_target.items())],
        columns=["entity_code", "time", "multiplicity"],
    )
    target_codes = targets["entity_code"].to_numpy(dtype=np.int32)
    predicate_counts = np.bincount(
        events["predicate_code"].to_numpy(dtype=np.int16),
        minlength=len(PREDICATES) + len(BASELINE_CONTROLS),
    )
    reported_events = events.loc[events["predicate_code"] < len(PREDICATES)]
    predicate_entity_counts = (
        reported_events.groupby("predicate_code", sort=False)["entity_code"]
        .nunique()
        .to_dict()
    )
    split_summary: dict[str, dict[str, int]] = {}
    for code, name in enumerate(PARTITION_NAMES):
        target_keep = (
            partitions[target_codes] == code
            if len(targets)
            else np.zeros(0, dtype=bool)
        )
        split_summary[name] = {
            "episodes": int(np.sum(partitions == code)),
            "target_episodes": int(np.unique(target_codes[target_keep]).size),
            "target_events": int(
                targets.loc[target_keep, "multiplicity"].sum() if len(targets) else 0
            ),
        }
    audit: dict[str, object] = {
        "raw_logs": decoded_count,
        "excluded_zero_address_logs": raw_audit.get("excluded_zero_address_logs", 0),
        "debt_episodes": len(entities),
        "wallets": int(entities["entity_id"].str.split(":", expand=True)[1].nunique()),
        "predicate_events": len(events),
        "target_events": int(targets["multiplicity"].sum()),
        "target_episodes": int(targets["entity_code"].nunique()),
        "liquidation_calls_inside_reconstructed_episode": observed_liquidation_calls,
        "repeated_liquidation_calls_collapsed": int(
            observed_liquidation_calls - len(targets)
        ),
        "liquidations_outside_reconstructed_episode": skipped_liquidations,
        "opening_transactions_marked": int(marked_entry_transactions),
        "pre_entry_logs_excluded_from_rules": int(excluded_pre_entry_logs),
        "terminal_block_logs_excluded_from_rules": int(excluded_terminal_block_logs),
        "post_target_logs_excluded_from_rules": int(excluded_post_target_logs),
        **market_audit,
        **price_audit,
        "maximum_predicate_attributes_per_primitive": maximum_per_primitive,
        "predicate_definition": (
            "one non-overlapping atomic or compound financial action per "
            "wallet/primitive transaction, using ABI action, actor-owner "
            "relation, rate mode, fixed reserve class, transaction-frozen "
            "prior portfolio state, and target-blind episode-history initiation "
            "versus continuation of repay/top-up interventions, plus daily "
            "market directions; no amount or outcome threshold"
        ),
        "predicate_contrast_families": PREDICATE_CONTRAST_FAMILIES,
        "verified_zero_debt_episodes": int(
            entities["end_reason"].eq("verified_zero_debt").sum()
        ),
        "partition_summary": split_summary,
        "predicate_counts": [
            {
                "predicate_code": code,
                "name": name,
                "events": int(predicate_counts[code]),
                "entities": int(predicate_entity_counts.get(code, 0)),
            }
            for code, name in enumerate(PREDICATES)
        ],
        "baseline_control_counts": [
            {
                "predicate_code": len(PREDICATES) + offset,
                "name": name,
                "events": int(predicate_counts[len(PREDICATES) + offset]),
            }
            for offset, name in enumerate(BASELINE_CONTROLS)
        ],
    }
    return entities, events, targets, audit


def preprocess_aave_full(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    partition_seed: int = 111,
    partition_fractions: tuple[float, float, float] = (0.5, 0.3, 0.2),
    rpc_urls: Iterable[str] = STATE_RPC_URLS,
    workers: int = 8,
    include_history_states: bool = False,
    overwrite: bool = False,
) -> Path:
    """Build the Aave temporal market-mechanism liquidation dataset."""

    raw_root, output_root = Path(raw_root), Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    if not _market_chunk_paths(raw_root):
        raise ValueError(
            "Aave v12 requires ReserveDataUpdated market chunks; rerun "
            "`crbstpp preprocess aave --download` without --skip-market-state"
        )
    cache_root = raw_root / "state_cache"
    candidates = _debt_zero_candidates(raw_root, cache_root)
    states = _resolve_debt_zero_points(
        candidates,
        cache_root,
        rpc_urls=tuple(str(value) for value in rpc_urls),
        workers=workers,
    )
    entities, events, targets, audit = _episode_tables(
        raw_root,
        zero_state=states,
        partition_seed=partition_seed,
        partition_fractions=partition_fractions,
        rpc_urls=tuple(str(value) for value in rpc_urls),
        workers=workers,
        include_oracle_prices=True,
    )
    audit.update(
        {
            "debt_zero_candidate_blocks": int(len(candidates)),
            "historical_state_calls": int(
                states["state_source"].eq("historical_eth_call").sum()
            ),
            "monotone_inferred_zero_states": int(
                states["state_source"].eq("monotone_after_zero").sum()
            ),
        }
    )
    predicate_names = (*PREDICATES, *BASELINE_CONTROLS)
    predicate_roles = (
        *(("reported",) * len(PREDICATES)),
        *(("baseline_control",) * len(BASELINE_CONTROLS)),
    )
    predicate_definitions = event_definitions(len(predicate_names))
    if include_history_states:
        (
            events,
            predicate_names,
            predicate_roles,
            predicate_definitions,
            state_audit,
        ) = augment_history_state_dictionary(
            entities,
            events,
            predicate_names=tuple(predicate_names),
            predicate_roles=tuple(predicate_roles),
        )
        audit["history_state_predicates"] = state_audit
    output = write_dataset(
        output_root,
        entities=entities,
        events=events,
        targets=targets,
        predicate_names=predicate_names,
        predicate_roles=predicate_roles,
        predicate_definitions=predicate_definitions,
        likelihood="first_event_cloglog",
        time_unit="7200_ethereum_block_epoch",
        ticks_per_unit=1,
        adverse_event_name="first Aave borrower liquidation in a debt episode",
        f0_contract={
            **F0_CONTRACT,
            "independent_certification_units": True,
            "complete_episode_risk_interval_observed": True,
            "behavioral_history_excluded_from_baseline": True,
        },
        provenance={
            "preprocessor": (
                "crbstpp.preprocess.aave.market_mechanism_first_liquidation_episode."
                + ("v17" if include_history_states else "v16")
            ),
            "raw_manifest_sha256": _sha256(raw_root / "manifest.json"),
            "entity": "wallet by Aave version and verified debt-positive episode",
            "independent_split_unit": "Ethereum wallet",
            "partition": {
                "method": "SHA-256(seed:wallet) thresholds",
                "fractions": list(map(float, partition_fractions)),
                "seed": int(partition_seed),
            },
            "clock": (
                "floor(Ethereum block number / 7200); deterministic chain-time "
                "epochs, approximately one day post-Merge"
            ),
            "episode_end": (
                "minimum of first liquidation and historical Pool.getUserAccountData "
                "totalDebtBase == 0 after block"
            ),
            "target_handling": (
                "one first liquidation target per debt episode; all later calls are "
                "collapsed and observation is censored at the first target tick"
            ),
            "risk_entry": (
                "first positive-debt Borrow transaction; generic Borrow occurrence "
                "defines the risk set, while exactly one outcome-blind opening "
                "configuration mark is retained with strictly-future impact"
            ),
            "risk_exit": (
                "minimum of first liquidation, verified zero total debt, and "
                "protocol observation end"
            ),
            "predicate_state": (
                "transaction-frozen reserve history, active collateral set, ABI "
                "actor-owner relation, "
                "a fixed public stable-value reserve map, "
                "target-blind ReserveDataUpdated rate/liquidity shocks, and "
                "historical Aave Oracle collateral/debt-asset price shocks"
            ),
            "history_state_grammar": (
                audit.get("history_state_predicates")
                if include_history_states
                else None
            ),
            "predicate_identity": (
                "one atomic or compound financial action per wallet transaction; "
                "the opening configuration may use debt-asset class, actor and rate "
                "mode; repeated interventions are represented only by the search-layer "
                "history-count mark; external market predicates "
                "require conservatively confirmed "
                "pre-existing debt exposure; amount, health factor, liquidation "
                "outcome, and cert/test data are unused"
            ),
            "stable_value_reserve_addresses": sorted(STABLE_VALUE_RESERVES),
            "unknown_reserve_class": "volatile_or_other",
            "compound_transactions": (
                "same-transaction debt restructuring, collateral rotation, "
                "leveraged expansion, or actor-specific deleveraging replace their "
                "overlapping component predicates; ambiguous compounds are not reported"
            ),
            "terminal_actions": (
                "logs in a verified zero-debt block have no strictly-future risk "
                "row and are excluded from the reported dictionary"
            ),
            "baseline_strata": {
                "0": "Aave Ethereum V2",
                "1": "Aave Ethereum V3",
                "rule_kernels_shared_across_strata": True,
            },
            "baseline_controls": {},
            "baseline_model": (
                "Aave V2/V3 protocol stratum crossed with target-blind "
                "episode-age and calendar-time cells at fit time; no financial "
                "behaviour or market predicate is conditioned away before rule "
                "discovery"
            ),
            "predicate_catalog": list(PREDICATES),
            "predicate_contrast_families": PREDICATE_CONTRAST_FAMILIES,
            "amount_or_target_tuned_predicates": {
                "amount_used": False,
                "target_used": False,
                "threshold_source": "none",
            },
            "market_state": {
                "source_event": "Aave Pool ReserveDataUpdated",
                "aggregation": "last on-chain variableBorrowRate per 7200-block tick",
                "comparison": (
                    "sequential conservative absolute-change rank p <= 0.01; "
                    "direction reported as upward/downward"
                ),
                "exposure": "strictly pre-existing positive nominal debt lower bound",
                "oracle_interface": "AaveOracle.getAssetPrice(address)",
                "oracle_addresses": AAVE_ORACLE_ADDRESSES,
                "oracle_cache": (
                    "immutable block/oracle/asset/value/RPC-response-digest records"
                ),
            },
        },
    )
    (output / "predicate_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
