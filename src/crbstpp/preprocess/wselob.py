from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import shutil
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests

from ..data import write_dataset


DATASET_DOI = "10.17632/3g4mhdp899.1"
DATASET_PAGE = "https://data.mendeley.com/datasets/3g4mhdp899/1"
PARTITION_NAMES = ("fit", "cert", "test")


@dataclass(frozen=True)
class WSELOBFile:
    file_id: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return (
            "https://data.mendeley.com/public-files/datasets/3g4mhdp899/files/"
            f"{self.file_id}/file_downloaded"
        )


# Immutable Mendeley v1 file identities.  The first experiment deliberately
# uses PEKAO, the smallest stock, so a complete primary-estimator run can stay
# within the requested wall-time budget without subsampling messages.
WSELOB_FILES: dict[str, dict[str, WSELOBFile]] = {
    "PEKAO": {
        "orders": WSELOBFile(
            "63e3f3ab-f562-4389-a740-357446266a6c",
            "3c418a55a492ebe2e39c8513cd7fc7e3e6827dc1a176af09fdcaad9f3485bae6",
            152_759_953,
        ),
        "trades": WSELOBFile(
            "4f19d1b9-5c0a-4e25-a5f8-67dba0613ccb",
            "7101a929f1882842f857e1467faf56b192c7f0df763027995b06a3fb7b9f1d53",
            7_684_603,
        ),
    },
    "KGHM": {
        "orders": WSELOBFile(
            "d30e1966-5bc7-492e-993d-2a1e4c5e6da0",
            "0246f994b42af04c65deb38f28f64988b7b6fa71d0fb83e41bda8487afff4c5c",
            325_838_174,
        ),
        "trades": WSELOBFile(
            "2c61561f-e683-412d-bd45-ec11f957a2d3",
            "c062f1b8d12b149af6c129a1400a94ec2a67cad9100ec508eb6009823798e5c4",
            11_144_901,
        ),
    },
    "PKNORLEN": {
        "orders": WSELOBFile(
            "337abfb0-285e-4f9b-a4ab-992417991e1d",
            "5b8f394144b8a3dace3e08bb2b4228d63231a12ae21f5f27aa662a8d09f3fa40",
            290_678_577,
        ),
        "trades": WSELOBFile(
            "bd6e42d7-9b88-41bc-babc-5b0ccb1aef6f",
            "817a5428b0a98aa352dc0ad18b328651117ef720b94a778d50c7dc3eb61ca938",
            9_406_549,
        ),
    },
    "PKOBP": {
        "orders": WSELOBFile(
            "ba14a4f2-865d-4d8b-8068-4af9a2efdda6",
            "ecf94a905b04836442c9124c7f7d7edc5b7d98e7236ca4026f0ec37b9407407e",
            281_092_890,
        ),
        "trades": WSELOBFile(
            "d9d9d93a-3b7e-492a-a572-3c33a190b192",
            "5154320d33c781bd9dac66e662cdf379b79039e279f68ea80ffdf6ee1e0380cb",
            9_453_680,
        ),
    },
    "PZU": {
        "orders": WSELOBFile(
            "e1acaa20-c404-4cbb-bfdf-a3eab7593ca3",
            "4ca8f7c136c940d71b271cc427d641c3881857e01b43555792d484d9eb2f4b8c",
            262_873_333,
        ),
        "trades": WSELOBFile(
            "66e9ad51-8803-4a5c-ad48-80232cb51a97",
            "c87cc034e6a0d104b4b945365aa58123ac95bc683d4eda61ce0017018bb37187",
            9_431_581,
        ),
    },
}


# Canonical primitive mechanisms.  A message can emit one action predicate and
# one state-transition predicate; both retain the same primitive ID, so one raw
# order-book update can never witness two antecedents of a high-order rule.
PREDICATES = (
    "pred_best_bid_limit_add",
    "pred_best_ask_limit_add",
    "pred_buy_limit_inside_spread",
    "pred_sell_limit_inside_spread",
    "pred_best_bid_order_removal",
    "pred_best_ask_order_removal",
    "pred_passive_bid_partial_fill",
    "pred_passive_ask_partial_fill",
    "pred_buy_market_order_submission",
    "pred_sell_market_order_submission",
    "pred_bid_queue_cleared",
    "pred_ask_queue_cleared",
    "pred_spread_widens",
    "pred_spread_narrows",
    "pred_bid_depth_dominance_starts",
    "pred_ask_depth_dominance_starts",
)

PREDICATE_MEANINGS = (
    "a new limit buy joins the current best bid",
    "a new limit sell joins the current best ask",
    "a new limit buy improves the best bid without using future price information",
    "a new limit sell improves the best ask without using future price information",
    "an order is removed from the current best bid",
    "an order is removed from the current best ask",
    "a partial fill reduces displayed volume at the current best bid",
    "a partial fill reduces displayed volume at the current best ask",
    "a buy-side market order is submitted",
    "a sell-side market order is submitted",
    "removal or fill exhausts the best-bid price level",
    "removal or fill exhausts the best-ask price level",
    "the quoted bid-ask spread becomes wider",
    "the quoted bid-ask spread becomes narrower",
    "best-bid displayed depth becomes larger than best-ask depth",
    "best-ask displayed depth becomes larger than best-bid depth",
)


# Mechanism-oriented schema used by the primary price-risk experiment.  Each
# raw order message emits at most one predicate.  In particular, derived
# spread/depth transitions are not duplicated beside their causal order
# action.  Queue exhaustion is separated by cancellation versus execution,
# and replenishment is defined using past book state only.
MECHANISM_PREDICATES = (
    "pred_best_bid_liquidity_add",
    "pred_best_ask_liquidity_add",
    "pred_buy_limit_inside_spread",
    "pred_sell_limit_inside_spread",
    "pred_best_bid_cancel_nonclear",
    "pred_best_ask_cancel_nonclear",
    "pred_passive_bid_fill_nonclear",
    "pred_passive_ask_fill_nonclear",
    "pred_buy_market_order_submission",
    "pred_sell_market_order_submission",
    "pred_cancel_bid_queue_cleared",
    "pred_cancel_ask_queue_cleared",
    "pred_trade_bid_queue_cleared",
    "pred_trade_ask_queue_cleared",
    "pred_bid_queue_replenished_after_clear",
    "pred_ask_queue_replenished_after_clear",
)

MECHANISM_PREDICATE_MEANINGS = (
    "a new limit buy supplies liquidity at the unchanged best bid",
    "a new limit sell supplies liquidity at the unchanged best ask",
    "a new limit buy improves the best bid",
    "a new limit sell improves the best ask",
    "cancellation reduces best-bid liquidity without clearing the price level",
    "cancellation reduces best-ask liquidity without clearing the price level",
    "execution reduces best-bid liquidity without clearing the price level",
    "execution reduces best-ask liquidity without clearing the price level",
    "a buy-side market order is submitted",
    "a sell-side market order is submitted",
    "cancellation exhausts the best-bid price level",
    "cancellation exhausts the best-ask price level",
    "execution exhausts the best-bid price level",
    "execution exhausts the best-ask price level",
    "bid liquidity is restored to the pre-clear price within the past-only lookback",
    "ask liquidity is restored to the pre-clear price within the past-only lookback",
)

# Direction-free mechanisms for a direction-free volatility target.  Bid and
# ask events are pooled only when their financial role is the same.  Passive
# additions and removals are then separated by whether the current message
# moves absolute top-of-book depth imbalance toward or away from zero.  This
# preserves the stabilising/destabilising distinction without using the
# future volatility target or a manually chosen size threshold.
BALANCED_MECHANISM_PREDICATES = (
    "pred_passive_liquidity_add_balance_restoring",
    "pred_passive_liquidity_add_balance_worsening",
    "pred_inside_spread_liquidity_add",
    "pred_cancel_balance_restoring",
    "pred_cancel_balance_worsening",
    "pred_execution_balance_restoring",
    "pred_execution_balance_worsening",
    "pred_market_order_submission",
    "pred_cancel_queue_cleared",
    "pred_trade_queue_cleared",
    "pred_queue_replenished_after_clear",
)

BALANCED_MECHANISM_PREDICATE_MEANINGS = (
    "passive best-quote liquidity is added and top-of-book depth becomes more balanced",
    "passive best-quote liquidity is added and top-of-book depth becomes less balanced",
    "a new limit order improves the quote and narrows the bid-ask spread",
    "a cancellation reduces top-of-book depth imbalance without clearing the quote",
    "a cancellation increases top-of-book depth imbalance without clearing the quote",
    "an execution reduces top-of-book depth imbalance without clearing the quote",
    "an execution increases top-of-book depth imbalance without clearing the quote",
    "a market order is submitted",
    "a cancellation exhausts the best quote",
    "an execution exhausts the best quote",
    "liquidity is restored to a recently cleared best quote",
)

# A non-overlapping, direction-free regime layer.  A raw message keeps its
# balanced action label unless the pre-event market context changes at that
# message.  In that case the action is replaced by exactly one regime label.
# This preserves one reportable predicate per primitive event while exposing
# both deterioration and recovery with the same target-blind construction.
REGIME_TRANSITION_PREDICATES = (
    "pred_market_stress_transition",
    "pred_market_recovery_completion",
)

REGIME_TRANSITION_PREDICATE_MEANINGS = (
    "the number of active pre-event market stress conditions increases",
    "the number of active pre-event market stress conditions decreases",
)

REGIME_MECHANISM_PREDICATES = (
    *BALANCED_MECHANISM_PREDICATES,
    *REGIME_TRANSITION_PREDICATES,
)

REGIME_MECHANISM_PREDICATE_MEANINGS = (
    *BALANCED_MECHANISM_PREDICATE_MEANINGS,
    *REGIME_TRANSITION_PREDICATE_MEANINGS,
)

# One target-blind categorical market-state axis for mechanism_v5.  Neutral
# is the reference level and therefore needs no reportable predicate.  The two
# reported states are mutually exclusive and coexist with, rather than replace,
# the primitive action occurring at the same clock time.
AGGREGATE_MARKET_STATE_SPECS = (
    ("state_market_stressed", "high", 0.75),
    ("state_market_calm", "low", 0.25),
)

AGGREGATE_MARKET_STATE_MEANINGS = {
    "state_market_stressed": (
        "the target-blind pre-event market stress score is in its high regime"
    ),
    "state_market_calm": (
        "the target-blind pre-event market stress score is in its low regime"
    ),
}

MARKET_STRESS_CHANNELS = (
    ("relative_spread", 2, "identity"),
    ("abs_depth_imbalance", 3, "absolute"),
    ("cancel_fraction_30s", 4, "identity"),
    ("trade_fraction_30s", 5, "identity"),
    ("message_rate_30s", 6, "identity"),
)

(
    BAL_LIQUIDITY_ADD_RESTORE,
    BAL_LIQUIDITY_ADD_WORSEN,
    BAL_INSIDE_SPREAD_ADD,
    BAL_CANCEL_RESTORE,
    BAL_CANCEL_WORSEN,
    BAL_EXECUTION_RESTORE,
    BAL_EXECUTION_WORSEN,
    BAL_MARKET_ORDER,
    BAL_CANCEL_CLEAR,
    BAL_TRADE_CLEAR,
    BAL_QUEUE_REPLENISH,
) = range(len(BALANCED_MECHANISM_PREDICATES))

# Predictable market contexts.  These are not duplicated action predicates.
# D_fit freezes one entry/exit threshold per channel, and the search engine
# combines an active state with a later primitive action lazily.
LOB_CONTEXT_STATE_SPECS = (
    ("state_wide_spread", "relative_spread", "high", 0.75),
    ("state_tight_spread", "relative_spread", "low", 0.25),
    ("state_bid_depth_dominant", "depth_imbalance", "high", 0.75),
    ("state_ask_depth_dominant", "depth_imbalance", "low", 0.25),
    ("state_cancellation_pressure_high", "cancel_fraction_30s", "high", 0.75),
    ("state_execution_pressure_high", "trade_fraction_30s", "high", 0.75),
    ("state_order_flow_activity_high", "message_rate_30s", "high", 0.75),
)

LOB_CONTEXT_STATE_MEANINGS = {
    "state_wide_spread": "the pre-event relative bid-ask spread is high",
    "state_tight_spread": "the pre-event relative bid-ask spread is low",
    "state_bid_depth_dominant": "pre-event best-bid depth dominates best-ask depth",
    "state_ask_depth_dominant": "pre-event best-ask depth dominates best-bid depth",
    "state_cancellation_pressure_high": "the past 30 seconds have high cancellation pressure",
    "state_execution_pressure_high": "the past 30 seconds have high execution pressure",
    "state_order_flow_activity_high": "the past 30 seconds have high message activity",
}

# Symmetric states used with the balanced mechanism schema.  Unlike the v2
# profile, every pressure channel has both a high and a low state, while depth
# uses direction-free imbalance/balance states.  Entry and exit thresholds
# are frozen from D_fit and use only information available before each event.
BALANCED_LOB_CONTEXT_STATE_SPECS = (
    ("state_wide_spread", "relative_spread", "high", 0.75),
    ("state_tight_spread", "relative_spread", "low", 0.25),
    ("state_depth_imbalanced", "abs_depth_imbalance", "high", 0.75),
    ("state_depth_balanced", "abs_depth_imbalance", "low", 0.25),
    ("state_cancellation_pressure_high", "cancel_fraction_30s", "high", 0.75),
    ("state_cancellation_pressure_low", "cancel_fraction_30s", "low", 0.25),
    ("state_execution_pressure_high", "trade_fraction_30s", "high", 0.75),
    ("state_execution_pressure_low", "trade_fraction_30s", "low", 0.25),
    ("state_order_flow_activity_high", "message_rate_30s", "high", 0.75),
    ("state_order_flow_activity_low", "message_rate_30s", "low", 0.25),
)

BALANCED_LOB_CONTEXT_STATE_MEANINGS = {
    "state_wide_spread": "the pre-event relative bid-ask spread is high",
    "state_tight_spread": "the pre-event relative bid-ask spread is low",
    "state_depth_imbalanced": "pre-event top-of-book depth is strongly imbalanced",
    "state_depth_balanced": "pre-event top-of-book depth is balanced",
    "state_cancellation_pressure_high": "the past 30 seconds have high cancellation pressure",
    "state_cancellation_pressure_low": "the past 30 seconds have low cancellation pressure",
    "state_execution_pressure_high": "the past 30 seconds have high execution pressure",
    "state_execution_pressure_low": "the past 30 seconds have low execution pressure",
    "state_order_flow_activity_high": "the past 30 seconds have high message activity",
    "state_order_flow_activity_low": "the past 30 seconds have low message activity",
}

(
    MECH_BEST_BID_ADD,
    MECH_BEST_ASK_ADD,
    MECH_BUY_INSIDE_SPREAD,
    MECH_SELL_INSIDE_SPREAD,
    MECH_BID_CANCEL_NONCLEAR,
    MECH_ASK_CANCEL_NONCLEAR,
    MECH_BID_FILL_NONCLEAR,
    MECH_ASK_FILL_NONCLEAR,
    MECH_BUY_MARKET,
    MECH_SELL_MARKET,
    MECH_CANCEL_BID_CLEAR,
    MECH_CANCEL_ASK_CLEAR,
    MECH_TRADE_BID_CLEAR,
    MECH_TRADE_ASK_CLEAR,
    MECH_BID_REPLENISH,
    MECH_ASK_REPLENISH,
) = range(len(MECHANISM_PREDICATES))

(
    BEST_BID_ADD,
    BEST_ASK_ADD,
    BUY_INSIDE_SPREAD,
    SELL_INSIDE_SPREAD,
    BEST_BID_REMOVE,
    BEST_ASK_REMOVE,
    PASSIVE_BID_FILL,
    PASSIVE_ASK_FILL,
    BUY_MARKET,
    SELL_MARKET,
    BID_QUEUE_CLEARED,
    ASK_QUEUE_CLEARED,
    SPREAD_WIDENS,
    SPREAD_NARROWS,
    BID_DOMINANCE_STARTS,
    ASK_DOMINANCE_STARTS,
) = range(len(PREDICATES))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_paths(raw_root: Path, stock: str) -> tuple[Path, Path]:
    return (
        raw_root / f"{stock}_lob_2017_zlib.h5",
        raw_root / f"{stock}_trades_2017_zlib.h5",
    )


def download_wselob(raw_root: str | Path, *, stock: str = "PEKAO") -> Path:
    """Download and digest-check one immutable WSELOB-2017 stock."""

    stock = str(stock).upper()
    if stock not in WSELOB_FILES:
        raise ValueError(f"unsupported WSELOB stock: {stock}")
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    paths = _raw_paths(raw_root, stock)
    for kind, output in zip(("orders", "trades"), paths, strict=True):
        identity = WSELOB_FILES[stock][kind]
        if output.is_file() and _sha256(output) == identity.sha256:
            continue
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with requests.get(identity.url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if temporary.stat().st_size != identity.size:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"WSELOB download size mismatch: {output.name}")
        if _sha256(temporary) != identity.sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"WSELOB download digest mismatch: {output.name}")
        os.replace(temporary, output)
    manifest = {
        "schema": "crbstpp.raw.wselob.v1",
        "dataset": "WSELOB-2017",
        "doi": DATASET_DOI,
        "license": "CC BY 4.0",
        "stock": stock,
        "files": {
            kind: {
                "path": path.name,
                "bytes": WSELOB_FILES[stock][kind].size,
                "sha256": WSELOB_FILES[stock][kind].sha256,
                "source_url": WSELOB_FILES[stock][kind].url,
            }
            for kind, path in zip(("orders", "trades"), paths, strict=True)
        },
    }
    temporary_manifest = raw_root / f".manifest.{os.getpid()}.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, raw_root / f"manifest_{stock}.json")
    return raw_root


def merge_wselob_datasets(
    input_roots: Sequence[str | Path],
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Combine independently preprocessed stocks without changing their data.

    Each input owns its fit-frozen target and context-state thresholds.  The
    merge therefore concatenates already constructed stock-day histories
    rather than recomputing a pooled threshold.  Entity, dependency, baseline
    and primitive-event identities are made stock-specific before the common
    immutable dataset manifest is written.
    """
    roots = tuple(Path(value) for value in input_roots)
    if len(roots) < 2:
        raise ValueError("multi-stock WSELOB merge requires at least two inputs")
    if len(set(roots)) != len(roots):
        raise ValueError("multi-stock WSELOB inputs must be distinct")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_root}")
        shutil.rmtree(output_root)

    manifests = [
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for root in roots
    ]
    stocks = [str(item.get("provenance", {}).get("stock", "")) for item in manifests]
    if any(not stock for stock in stocks) or len(set(stocks)) != len(stocks):
        raise ValueError("every WSELOB input must identify one distinct stock")

    identity_fields = (
        "predicate_names",
        "predicate_roles",
        "likelihood",
        "time_unit",
        "ticks_per_unit",
        "adverse_event_name",
        "f0_contract",
    )
    reference = manifests[0]
    for manifest in manifests[1:]:
        for field in identity_fields:
            if manifest.get(field) != reference.get(field):
                raise ValueError(f"incompatible WSELOB merge field: {field}")
        left = reference.get("predicate_definitions", [])
        right = manifest.get("predicate_definitions", [])
        if len(left) != len(right):
            raise ValueError("WSELOB predicate definitions do not align")
        for left_definition, right_definition in zip(left, right, strict=True):
            for key in (
                "kind",
                "meaning",
                "construction",
                "entry_predicate",
                "exit_predicate",
                "channel",
                "transform",
                "direction",
                "threshold_partition",
                "activation",
            ):
                if left_definition.get(key) != right_definition.get(key):
                    raise ValueError(
                        f"incompatible WSELOB predicate definition field: {key}"
                    )

    entity_sources: list[pd.DataFrame] = []
    event_sources: list[pd.DataFrame] = []
    target_sources: list[pd.DataFrame] = []
    baseline_sources: list[pd.DataFrame] = []
    primitive_cursor = 0
    baseline_cursor = 0
    entity_baseline_cursor = 0
    source_summaries: list[dict[str, object]] = []

    for stock_index, (stock, root, manifest) in enumerate(
        zip(stocks, roots, manifests, strict=True)
    ):
        entities = pd.read_parquet(root / manifest["files"]["entities"]["path"])
        events = pd.read_parquet(root / manifest["files"]["events"]["path"])
        targets = pd.read_parquet(root / manifest["files"]["targets"]["path"])
        cells_specification = manifest["files"].get("baseline_cells")
        if cells_specification is None:
            raise ValueError("continuous WSELOB merge requires baseline cells")
        cells = pd.read_parquet(root / cells_specification["path"])
        if "partition" not in entities or "dependency_group" not in entities:
            raise ValueError("WSELOB merge requires frozen partitions and dependency groups")

        local_entity_count = len(entities)
        local_codes = np.arange(local_entity_count, dtype=np.int32)
        entities = entities.copy()
        entities["_stock_index"] = np.int16(stock_index)
        entities["_old_code"] = local_codes
        entities["entity_id"] = [
            value if str(value).startswith(f"{stock}:") else f"{stock}:{value}"
            for value in entities["entity_id"]
        ]
        entities["dependency_group"] = [
            f"{stock}:{value}" for value in entities["dependency_group"]
        ]
        entities["split_group"] = (
            np.int64(stock_index) * np.int64(100_000_000)
            + entities["split_group"].to_numpy(dtype=np.int64)
        )

        # A continuous dataset has one nuisance label per entity (weekday)
        # and a finer label per exposure cell (weekday x intraday bin).  They
        # are validated independently, so each needs its own stock offset.
        entity_labels = np.sort(entities["baseline_stratum"].unique())
        entity_label_map = {
            int(value): entity_baseline_cursor + index
            for index, value in enumerate(entity_labels.tolist())
        }
        entities["baseline_stratum"] = (
            entities["baseline_stratum"].map(entity_label_map).astype(np.int16)
        )
        entity_baseline_cursor += len(entity_labels)

        cell_labels = np.sort(cells["baseline_stratum"].unique())
        cell_label_map = {
            int(value): baseline_cursor + index
            for index, value in enumerate(cell_labels.tolist())
        }
        cells = cells.copy()
        cells["baseline_stratum"] = (
            cells["baseline_stratum"].map(cell_label_map).astype(np.int16)
        )
        baseline_cursor += len(cell_labels)

        events = events.copy()
        primitive = events["primitive_event_id"].to_numpy(dtype=np.int64)
        if len(primitive):
            primitive_min = int(primitive.min())
            primitive_span = int(primitive.max()) - primitive_min + 1
            events["primitive_event_id"] = (
                primitive - primitive_min + primitive_cursor
            )
            primitive_cursor += primitive_span
        events["_stock_index"] = np.int16(stock_index)
        targets = targets.copy()
        targets["_stock_index"] = np.int16(stock_index)
        cells["_stock_index"] = np.int16(stock_index)

        entity_sources.append(entities)
        event_sources.append(events)
        target_sources.append(targets)
        baseline_sources.append(cells)
        source_summaries.append(
            {
                "stock": stock,
                "dataset_digest": manifest["dataset_digest"],
                "root": str(root),
                "entities": int(local_entity_count),
                "targets": int(targets["multiplicity"].sum()),
                "target_threshold": manifest.get("provenance", {})
                .get("target_volatility", {})
                .get("frozen_threshold"),
            }
        )

    entities = pd.concat(entity_sources, ignore_index=True)
    entities.sort_values(
        ["partition", "_stock_index", "_old_code"], kind="stable", inplace=True
    )
    entities.reset_index(drop=True, inplace=True)
    remaps: dict[int, np.ndarray] = {}
    for stock_index in range(len(stocks)):
        selected = entities["_stock_index"].to_numpy() == stock_index
        old = entities.loc[selected, "_old_code"].to_numpy(dtype=np.int32)
        new = np.flatnonzero(selected).astype(np.int32)
        mapping = np.empty(len(old), dtype=np.int32)
        mapping[old] = new
        remaps[stock_index] = mapping

    def remap_rows(parts: list[pd.DataFrame]) -> pd.DataFrame:
        remapped: list[pd.DataFrame] = []
        for stock_index, part in enumerate(parts):
            part = part.copy()
            part["entity_code"] = remaps[stock_index][
                part["entity_code"].to_numpy(dtype=np.int32)
            ]
            part.drop(columns=["_stock_index"], inplace=True)
            remapped.append(part)
        output = pd.concat(remapped, ignore_index=True)
        output.sort_values(["entity_code", "time"], kind="stable", inplace=True)
        output.reset_index(drop=True, inplace=True)
        return output

    events = remap_rows(event_sources)
    targets = remap_rows(target_sources)
    baseline_cells = remap_rows(baseline_sources)
    entities.drop(columns=["_stock_index", "_old_code"], inplace=True)

    definitions = deepcopy(reference.get("predicate_definitions", []))
    for predicate, definition in enumerate(definitions):
        if definition.get("kind") != "transition_state":
            continue
        definition["entry_threshold_by_stock"] = {
            stock: manifests[index]["predicate_definitions"][predicate].get(
                "entry_threshold"
            )
            for index, stock in enumerate(stocks)
        }
        definition["exit_threshold_by_stock"] = {
            stock: manifests[index]["predicate_definitions"][predicate].get(
                "exit_threshold"
            )
            for index, stock in enumerate(stocks)
        }
        definition.pop("entry_threshold", None)
        definition.pop("exit_threshold", None)

    target_counts = np.bincount(
        entities["partition"].to_numpy(dtype=np.int8)[
            targets["entity_code"].to_numpy(dtype=np.int32)
        ],
        weights=targets["multiplicity"].to_numpy(dtype=np.int64),
        minlength=3,
    ).astype(np.int64)
    return write_dataset(
        output_root,
        entities=entities,
        events=events,
        targets=targets,
        baseline_cells=baseline_cells,
        predicate_names=reference["predicate_names"],
        predicate_roles=reference["predicate_roles"],
        predicate_definitions=definitions,
        likelihood=reference["likelihood"],
        time_unit=reference["time_unit"],
        ticks_per_unit=int(reference["ticks_per_unit"]),
        adverse_event_name=reference["adverse_event_name"],
        f0_contract=reference["f0_contract"],
        provenance={
            "preprocessor": "crbstpp.preprocess.wselob.multi_stock_merge.v1",
            "dataset": "WSELOB-2017",
            "doi": DATASET_DOI,
            "dataset_page": DATASET_PAGE,
            "license": "CC BY 4.0",
            "stocks": stocks,
            "source_datasets": source_summaries,
            "entity_definition": "stock by continuous-session trading day",
            "target_definition": (
                "stock-specific duration-weighted D_fit Q90 30-second realized-"
                "volatility burst onset"
            ),
            "predicate_definition": reference["provenance"].get(
                "predicate_definition"
            ),
            "predicate_schema": reference["provenance"].get("predicate_schema"),
            "baseline": (
                "stock crossed with weekday and four fixed intraday bins; "
                "no order behavior or target history"
            ),
            "partition": {
                "method": "within-stock calendar-month-stratified split",
                "fractions": [0.5, 0.3, 0.2],
                "seed": 111,
                "counts": [
                    int(np.sum(entities["partition"].to_numpy() == code))
                    for code in range(3)
                ],
            },
            "dependency_unit": "stock trading day",
            "target_events": int(targets["multiplicity"].sum()),
            "target_events_by_partition": target_counts.tolist(),
            "trading_days": int(len(entities)),
            "risk_ticks": int(len(baseline_cells)),
        },
    )


class _BookSide:
    def __init__(self, *, bid: bool) -> None:
        self.bid = bool(bid)
        self.volume: dict[int, int] = {}
        self.heap: list[int] = []

    def clear(self) -> None:
        self.volume.clear()
        self.heap.clear()

    def set(self, price: int, volume: int) -> None:
        if price <= 0:
            return
        if volume > 0:
            self.volume[price] = volume
            heapq.heappush(self.heap, -price if self.bid else price)
        else:
            self.volume.pop(price, None)

    def best(self) -> tuple[int | None, int]:
        while self.heap:
            encoded = self.heap[0]
            price = -encoded if self.bid else encoded
            volume = int(self.volume.get(price, 0))
            if volume > 0:
                return price, volume
            heapq.heappop(self.heap)
        return None, 0


def _dominance(bid_depth: int, ask_depth: int) -> int:
    return int(bid_depth > ask_depth) - int(ask_depth > bid_depth)


def _process_day(
    orders: pd.DataFrame,
    *,
    entity_code: int,
    session_start_ns: int,
    session_end_ns: int,
    bin_nanoseconds: int,
    continuous: bool = False,
) -> tuple[list[tuple[int, int, int, int]], list[int], dict[str, int]]:
    """Reconstruct one book and emit strictly timestamped primitive events."""

    bids, asks = _BookSide(bid=True), _BookSide(bid=False)
    events: list[tuple[int, int, int, int]] = []
    target_ticks: list[int] = []
    action_counts: dict[str, int] = {}
    if not orders["time"].is_monotonic_increasing:
        orders = orders.sort_values("time", kind="stable")

    columns = [
        "time",
        "price",
        "agg_volume",
        "side",
        "order_type",
        "action_type",
    ]
    for row_number, row in enumerate(
        orders[columns].itertuples(index=False, name=None)
    ):
        time_ns, price, aggregate, side, order_type, action = row
        time_ns = int(time_ns)
        price = int(price)
        aggregate = int(aggregate)
        side = int(side)
        order_type = str(order_type)
        action = str(action)
        action_counts[action] = action_counts.get(action, 0) + 1
        before_bid, before_bid_depth = bids.best()
        before_ask, before_ask_depth = asks.best()
        before_spread = (
            before_ask - before_bid
            if before_bid is not None and before_ask is not None
            else None
        )
        before_dominance = _dominance(before_bid_depth, before_ask_depth)
        action_predicate: int | None = None

        in_session = session_start_ns <= time_ns <= session_end_ns
        is_bid = side == 1
        is_ask = side in (2, 5)
        if in_session:
            if action == "A" and order_type == "1":
                action_predicate = (
                    BUY_MARKET if is_bid else SELL_MARKET if is_ask else None
                )
            elif action == "A" and order_type == "2":
                if is_bid and before_bid is not None:
                    if price > before_bid:
                        action_predicate = BUY_INSIDE_SPREAD
                    elif price == before_bid:
                        action_predicate = BEST_BID_ADD
                elif is_ask and before_ask is not None:
                    if price < before_ask:
                        action_predicate = SELL_INSIDE_SPREAD
                    elif price == before_ask:
                        action_predicate = BEST_ASK_ADD
            elif action == "M":
                if is_bid and price == before_bid:
                    action_predicate = PASSIVE_BID_FILL
                elif is_ask and price == before_ask:
                    action_predicate = PASSIVE_ASK_FILL
            elif action == "D":
                if is_bid and price == before_bid:
                    action_predicate = BEST_BID_REMOVE
                elif is_ask and price == before_ask:
                    action_predicate = BEST_ASK_REMOVE

        if action == "F":
            bids.clear()
            asks.clear()
        elif action in {"A", "M", "D", "Y"} and price > 0:
            if is_bid:
                bids.set(price, aggregate)
            elif is_ask:
                asks.set(price, aggregate)

        after_bid, after_bid_depth = bids.best()
        after_ask, after_ask_depth = asks.best()
        if not in_session:
            continue
        tick = (
            time_ns
            if continuous
            else int((time_ns - session_start_ns) // bin_nanoseconds)
        )
        primitive_id = (int(entity_code) << 32) + int(row_number)
        emitted: list[int] = []
        if action_predicate is not None:
            if (
                action_predicate == BEST_BID_REMOVE
                and before_bid is not None
                and (after_bid is None or after_bid < before_bid)
            ):
                action_predicate = BID_QUEUE_CLEARED
            elif (
                action_predicate == BEST_ASK_REMOVE
                and before_ask is not None
                and (after_ask is None or after_ask > before_ask)
            ):
                action_predicate = ASK_QUEUE_CLEARED
            emitted.append(action_predicate)

        if after_bid is not None and after_ask is not None:
            after_spread = after_ask - after_bid
            if before_spread is not None:
                if after_spread > before_spread:
                    emitted.append(SPREAD_WIDENS)
                elif after_spread < before_spread:
                    emitted.append(SPREAD_NARROWS)
                before_midprice_twice = before_bid + before_ask
                after_midprice_twice = after_bid + after_ask
                if after_midprice_twice < before_midprice_twice:
                    target_ticks.append(tick)
            after_dominance = _dominance(after_bid_depth, after_ask_depth)
            if after_dominance == 1 and before_dominance != 1:
                emitted.append(BID_DOMINANCE_STARTS)
            elif after_dominance == -1 and before_dominance != -1:
                emitted.append(ASK_DOMINANCE_STARTS)

        for predicate in dict.fromkeys(emitted):
            events.append((entity_code, tick, int(predicate), primitive_id))
    return events, target_ticks, action_counts


def _process_day_mechanisms(
    orders: pd.DataFrame,
    *,
    entity_code: int,
    session_start_ns: int,
    session_end_ns: int,
    continuous: bool,
    bin_nanoseconds: int,
    replenishment_lookback_ns: int,
    trade_times: frozenset[int] = frozenset(),
    balanced_mechanisms: bool = False,
    context_rows: list[tuple[int, int, float, float, float, float, float]]
    | None = None,
) -> tuple[
    list[tuple[int, int, int, int]],
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    """Emit one mutually exclusive financial mechanism per raw message.

    The returned mid-price trace contains only changes and is used later to
    freeze the adverse-excursion threshold on D_fit.  It is not consulted by
    predicate construction.
    """

    bids, asks = _BookSide(bid=True), _BookSide(bid=False)
    events: list[tuple[int, int, int, int]] = []
    action_counts: dict[str, int] = {}
    mid_ticks: list[int] = []
    mid_twice: list[int] = []
    last_bid_clear: tuple[int, int] | None = None
    last_ask_clear: tuple[int, int] | None = None
    recent_actions: deque[tuple[int, str]] = deque()
    recent_counts = {"cancel": 0, "trade": 0, "other": 0}
    context_horizon_ns = 30 * 1_000_000_000
    if not orders["time"].is_monotonic_increasing:
        orders = orders.sort_values("time", kind="stable")

    columns = [
        "time",
        "price",
        "agg_volume",
        "side",
        "order_type",
        "action_type",
    ]
    for row_number, row in enumerate(
        orders[columns].itertuples(index=False, name=None)
    ):
        time_ns, price, aggregate, side, order_type, action = row
        time_ns = int(time_ns)
        price = int(price)
        aggregate = int(aggregate)
        side = int(side)
        order_type = str(order_type)
        action = str(action)
        action_counts[action] = action_counts.get(action, 0) + 1
        before_bid, before_bid_depth = bids.best()
        before_ask, before_ask_depth = asks.best()
        in_session = session_start_ns <= time_ns <= session_end_ns
        is_bid = side == 1
        is_ask = side in (2, 5)
        execution = action == "M" or time_ns in trade_times

        while recent_actions and recent_actions[0][0] < time_ns - context_horizon_ns:
            _, expired_kind = recent_actions.popleft()
            recent_counts[expired_kind] -= 1
        if in_session and context_rows is not None:
            valid_book = (
                before_bid is not None
                and before_ask is not None
                and before_ask > before_bid
                and before_bid_depth + before_ask_depth > 0
            )
            relative_spread = (
                2.0 * float(before_ask - before_bid) / float(before_ask + before_bid)
                if valid_book and before_ask + before_bid > 0
                else math.nan
            )
            depth_imbalance = (
                float(before_bid_depth - before_ask_depth)
                / float(before_bid_depth + before_ask_depth)
                if valid_book
                else math.nan
            )
            recent_total = len(recent_actions)
            denominator = float(max(1, recent_total))
            context_rows.append(
                (
                    time_ns,
                    (int(entity_code) << 32) + int(row_number),
                    relative_spread,
                    depth_imbalance,
                    float(recent_counts["cancel"]) / denominator,
                    float(recent_counts["trade"]) / denominator,
                    float(recent_total) / 30.0,
                )
            )

        if action == "F":
            bids.clear()
            asks.clear()
        elif action in {"A", "M", "D", "Y"} and price > 0:
            if is_bid:
                bids.set(price, aggregate)
            elif is_ask:
                asks.set(price, aggregate)
        after_bid, after_bid_depth = bids.best()
        after_ask, after_ask_depth = asks.best()
        if not in_session:
            continue

        predicate: int | None = None
        if action == "A" and order_type == "1":
            predicate = (
                MECH_BUY_MARKET if is_bid else MECH_SELL_MARKET if is_ask else None
            )
        elif action == "A" and order_type == "2":
            if is_bid:
                if (
                    last_bid_clear is not None
                    and 0 <= time_ns - last_bid_clear[0] <= replenishment_lookback_ns
                    and price >= last_bid_clear[1]
                ):
                    predicate = MECH_BID_REPLENISH
                    last_bid_clear = None
                elif before_bid is not None and price > before_bid:
                    predicate = MECH_BUY_INSIDE_SPREAD
                elif before_bid is not None and price == before_bid:
                    predicate = MECH_BEST_BID_ADD
            elif is_ask:
                if (
                    last_ask_clear is not None
                    and 0 <= time_ns - last_ask_clear[0] <= replenishment_lookback_ns
                    and price <= last_ask_clear[1]
                ):
                    predicate = MECH_ASK_REPLENISH
                    last_ask_clear = None
                elif before_ask is not None and price < before_ask:
                    predicate = MECH_SELL_INSIDE_SPREAD
                elif before_ask is not None and price == before_ask:
                    predicate = MECH_BEST_ASK_ADD
        elif action in {"D", "M"}:
            if is_bid and before_bid is not None and price == before_bid:
                cleared = after_bid is None or after_bid < before_bid
                if cleared:
                    predicate = (
                        MECH_TRADE_BID_CLEAR if execution else MECH_CANCEL_BID_CLEAR
                    )
                    last_bid_clear = (time_ns, before_bid)
                else:
                    predicate = (
                        MECH_BID_FILL_NONCLEAR
                        if execution
                        else MECH_BID_CANCEL_NONCLEAR
                    )
            elif is_ask and before_ask is not None and price == before_ask:
                cleared = after_ask is None or after_ask > before_ask
                if cleared:
                    predicate = (
                        MECH_TRADE_ASK_CLEAR if execution else MECH_CANCEL_ASK_CLEAR
                    )
                    last_ask_clear = (time_ns, before_ask)
                else:
                    predicate = (
                        MECH_ASK_FILL_NONCLEAR
                        if execution
                        else MECH_ASK_CANCEL_NONCLEAR
                    )

        if balanced_mechanisms and predicate is not None:
            before_total = int(before_bid_depth) + int(before_ask_depth)
            after_total = int(after_bid_depth) + int(after_ask_depth)
            if before_total > 0 and after_total > 0:
                before_abs_imbalance = abs(
                    float(before_bid_depth - before_ask_depth) / float(before_total)
                )
                after_abs_imbalance = abs(
                    float(after_bid_depth - after_ask_depth) / float(after_total)
                )
                tolerance = 1.0e-15
                balance_effect = (
                    -1
                    if after_abs_imbalance < before_abs_imbalance - tolerance
                    else 1
                    if after_abs_imbalance > before_abs_imbalance + tolerance
                    else 0
                )
            else:
                balance_effect = 0

            if predicate in {MECH_BEST_BID_ADD, MECH_BEST_ASK_ADD}:
                predicate = (
                    BAL_LIQUIDITY_ADD_RESTORE
                    if balance_effect < 0
                    else BAL_LIQUIDITY_ADD_WORSEN
                    if balance_effect > 0
                    else None
                )
            elif predicate in {MECH_BUY_INSIDE_SPREAD, MECH_SELL_INSIDE_SPREAD}:
                predicate = BAL_INSIDE_SPREAD_ADD
            elif predicate in {MECH_BID_CANCEL_NONCLEAR, MECH_ASK_CANCEL_NONCLEAR}:
                predicate = (
                    BAL_CANCEL_RESTORE
                    if balance_effect < 0
                    else BAL_CANCEL_WORSEN
                    if balance_effect > 0
                    else None
                )
            elif predicate in {MECH_BID_FILL_NONCLEAR, MECH_ASK_FILL_NONCLEAR}:
                predicate = (
                    BAL_EXECUTION_RESTORE
                    if balance_effect < 0
                    else BAL_EXECUTION_WORSEN
                    if balance_effect > 0
                    else None
                )
            elif predicate in {MECH_BUY_MARKET, MECH_SELL_MARKET}:
                predicate = BAL_MARKET_ORDER
            elif predicate in {MECH_CANCEL_BID_CLEAR, MECH_CANCEL_ASK_CLEAR}:
                predicate = BAL_CANCEL_CLEAR
            elif predicate in {MECH_TRADE_BID_CLEAR, MECH_TRADE_ASK_CLEAR}:
                predicate = BAL_TRADE_CLEAR
            elif predicate in {MECH_BID_REPLENISH, MECH_ASK_REPLENISH}:
                predicate = BAL_QUEUE_REPLENISH
            else:  # pragma: no cover - the exhaustive mapping is an invariant
                raise AssertionError(f"unmapped balanced mechanism: {predicate}")

        tick = (
            time_ns
            if continuous
            else int((time_ns - session_start_ns) // bin_nanoseconds)
        )
        if predicate is not None:
            primitive_id = (int(entity_code) << 32) + int(row_number)
            events.append((entity_code, tick, int(predicate), primitive_id))

        if after_bid is not None and after_ask is not None:
            value = int(after_bid + after_ask)
            if mid_ticks and mid_ticks[-1] == time_ns:
                mid_twice[-1] = value
            elif not mid_twice or value != mid_twice[-1]:
                mid_ticks.append(time_ns)
                mid_twice.append(value)

        action_kind = (
            "trade"
            if execution
            else "cancel"
            if action == "D"
            else "other"
        )
        recent_actions.append((time_ns, action_kind))
        recent_counts[action_kind] += 1

    return (
        events,
        np.asarray(mid_ticks, dtype=np.int64),
        np.asarray(mid_twice, dtype=np.float64),
        action_counts,
    )


def _adverse_excursion_returns(
    ticks: np.ndarray,
    mid_twice: np.ndarray,
    *,
    session_start_ns: int,
    horizon_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return change timestamps and past-only horizon log returns."""

    if len(ticks) != len(mid_twice):
        raise ValueError("mid-price ticks and values have different lengths")
    if len(ticks) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    reference = np.searchsorted(ticks, ticks - int(horizon_ns), side="right") - 1
    valid = (reference >= 0) & (ticks >= int(session_start_ns) + int(horizon_ns))
    indices = np.flatnonzero(valid)
    if not len(indices):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    values = np.log(mid_twice[indices] / mid_twice[reference[indices]])
    return ticks[indices], np.ascontiguousarray(values, dtype=np.float64)


def _adverse_excursion_targets(
    ticks: np.ndarray,
    returns: np.ndarray,
    *,
    threshold: float,
    rearm_fraction: float,
) -> list[int]:
    """Decluster downward threshold crossings with deterministic hysteresis."""

    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("adverse excursion threshold must be positive")
    if not 0.0 < rearm_fraction < 1.0:
        raise ValueError("target rearm fraction must lie strictly between zero and one")
    armed = True
    targets: list[int] = []
    rearm_level = -float(rearm_fraction) * float(threshold)
    trigger_level = -float(threshold)
    for tick, value in zip(ticks.tolist(), returns.tolist(), strict=True):
        if not armed and value > rearm_level:
            armed = True
        if armed and value <= trigger_level:
            targets.append(int(tick))
            armed = False
    return targets


def _realized_volatility(
    ticks: np.ndarray,
    mid_twice: np.ndarray,
    *,
    session_start_ns: int,
    session_end_ns: int,
    horizon_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact piecewise-constant past-only realized volatility.

    A price jump at time ``s`` contributes to the value at ``t`` exactly when
    ``t-horizon < s <= t``.  Consequently the process changes both at a new
    jump and when an old jump leaves the window at ``s+horizon``.  Including
    both boundaries is required for a genuine continuous-time weighted
    quantile; otherwise a quiet period after a burst is incorrectly assigned
    the stale high value.  The first reconstructed quote has no preceding
    within-session price and therefore contributes a zero jump.
    """

    ticks = np.asarray(ticks, dtype=np.int64)
    mid_twice = np.asarray(mid_twice, dtype=np.float64)
    if len(ticks) != len(mid_twice):
        raise ValueError("mid-price ticks and values have different lengths")
    if len(ticks) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    if np.any(mid_twice <= 0.0):
        raise ValueError("mid-price must be positive")
    jumps = np.zeros(len(ticks), dtype=np.float64)
    if len(ticks) > 1:
        jumps[1:] = np.diff(np.log(mid_twice))
    squared_prefix = np.empty(len(ticks) + 1, dtype=np.float64)
    squared_prefix[0] = 0.0
    np.cumsum(jumps * jumps, out=squared_prefix[1:])
    first = int(session_start_ns)
    last = int(session_end_ns)
    if first > last:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    boundaries = np.unique(
        np.concatenate(
            (
                np.asarray([first], dtype=np.int64),
                ticks[(ticks >= first) & (ticks <= last)],
                (ticks + int(horizon_ns))[
                    (ticks + int(horizon_ns) >= first)
                    & (ticks + int(horizon_ns) <= last)
                ],
            )
        )
    )
    left = np.searchsorted(ticks, boundaries - int(horizon_ns), side="right")
    right = np.searchsorted(ticks, boundaries, side="right")
    variance = squared_prefix[right] - squared_prefix[left]
    return (
        np.ascontiguousarray(boundaries, dtype=np.int64),
        np.sqrt(np.maximum(variance, 0.0), dtype=np.float64),
    )


def _duration_weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Return the left-continuous weighted empirical quantile."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("weighted quantile values and weights must align")
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(keep):
        raise ValueError("weighted quantile has no positive finite weight")
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values, kind="stable")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    cutoff = float(quantile) * float(cumulative[-1])
    index = min(len(values) - 1, int(np.searchsorted(cumulative, cutoff, side="left")))
    return float(values[index])


def _context_duration_weights(
    payload: dict[str, object],
    rows: list[tuple[int, int, float, float, float, float, float]],
) -> np.ndarray:
    times = np.fromiter((int(row[0]) for row in rows), dtype=np.int64, count=len(rows))
    next_times = np.minimum(
        np.r_[times[1:], int(payload["session_end"]) + 1],
        int(payload["session_end"]) + 1,
    )
    return np.maximum(0, next_times - times).astype(np.float64)


def _weighted_percentile_profile(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, list[float]]:
    """Freeze a compact duration-weighted empirical CDF."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(keep):
        raise ValueError("percentile profile has no positive finite weight")
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values, kind="stable")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    probabilities = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    indices = np.searchsorted(
        cumulative,
        probabilities * float(cumulative[-1]),
        side="left",
    )
    indices = np.minimum(indices, len(values) - 1)
    knots = values[indices]
    # Quantile knots can repeat for discrete channels.  Keep the largest CDF
    # value at each knot so interpolation remains monotone and deterministic.
    unique = np.unique(knots)
    cdf = np.asarray(
        [float(np.max(probabilities[knots == value])) for value in unique],
        dtype=np.float64,
    )
    return {"values": unique.tolist(), "probabilities": cdf.tolist()}


def _market_stress_scores(
    rows: list[tuple[int, int, float, float, float, float, float]],
    profile: dict[str, dict[str, object]],
) -> np.ndarray:
    """Return the equal-weight percentile score of the five context channels."""

    if not rows:
        return np.zeros(0, dtype=np.float64)
    scores = np.zeros(len(rows), dtype=np.float64)
    used = 0
    for channel, column, transform in MARKET_STRESS_CHANNELS:
        values = np.fromiter(
            (float(row[column]) for row in rows),
            dtype=np.float64,
            count=len(rows),
        )
        if transform == "absolute":
            values = np.abs(values)
        frozen = profile[channel]
        knots = np.asarray(frozen["values"], dtype=np.float64)
        probabilities = np.asarray(frozen["probabilities"], dtype=np.float64)
        if not len(knots):
            raise ValueError(f"empty market-state percentile profile: {channel}")
        scores += np.interp(
            values,
            knots,
            probabilities,
            left=0.0,
            right=1.0,
        )
        used += 1
    return scores / float(used)


def _fit_lob_market_states(
    days: list[dict[str, object]],
    partition: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Freeze one mutually exclusive calm/neutral/stressed axis on D_fit."""

    values_by_channel: dict[str, list[np.ndarray]] = {
        channel: [] for channel, _, _ in MARKET_STRESS_CHANNELS
    }
    weights_by_channel: dict[str, list[np.ndarray]] = {
        channel: [] for channel, _, _ in MARKET_STRESS_CHANNELS
    }
    fit_payloads: list[
        tuple[
            dict[str, object],
            list[tuple[int, int, float, float, float, float, float]],
            np.ndarray,
        ]
    ] = []
    for payload in days:
        if int(partition[int(payload["entity_code"])]) != 0:
            continue
        rows = payload.get("context_rows", [])
        if not isinstance(rows, list) or not rows:
            continue
        weights = _context_duration_weights(payload, rows)
        fit_payloads.append((payload, rows, weights))
        for channel, column, transform in MARKET_STRESS_CHANNELS:
            values = np.fromiter(
                (float(row[column]) for row in rows),
                dtype=np.float64,
                count=len(rows),
            )
            if transform == "absolute":
                values = np.abs(values)
            keep = np.isfinite(values) & (weights > 0.0)
            if np.any(keep):
                values_by_channel[channel].append(values[keep])
                weights_by_channel[channel].append(weights[keep])
    profile: dict[str, dict[str, object]] = {}
    for channel, _, transform in MARKET_STRESS_CHANNELS:
        if not values_by_channel[channel]:
            raise ValueError(f"D_fit has no market-state values for {channel}")
        frozen = _weighted_percentile_profile(
            np.concatenate(values_by_channel[channel]),
            np.concatenate(weights_by_channel[channel]),
        )
        profile[channel] = {"transform": transform, **frozen}

    score_parts: list[np.ndarray] = []
    score_weights: list[np.ndarray] = []
    for _, rows, weights in fit_payloads:
        scores = _market_stress_scores(rows, profile)
        keep = np.isfinite(scores) & (weights > 0.0)
        if np.any(keep):
            score_parts.append(scores[keep])
            score_weights.append(weights[keep])
    if not score_parts:
        raise ValueError("D_fit produced no market stress scores")
    scores = np.concatenate(score_parts)
    weights = np.concatenate(score_weights)
    q25 = _duration_weighted_quantile(scores, weights, 0.25)
    q50 = _duration_weighted_quantile(scores, weights, 0.50)
    q75 = _duration_weighted_quantile(scores, weights, 0.75)
    if not q25 < q50 < q75:
        raise ValueError("D_fit market-state quantiles are not separated")
    states = [
        {
            "name": name,
            "meaning": AGGREGATE_MARKET_STATE_MEANINGS[name],
            "channel": "equal_weight_market_stress_percentile",
            "direction": direction,
            "entry_threshold": float(q75 if direction == "high" else q25),
            "exit_threshold": float(q50),
            "entry_quantile": float(quantile),
            "exit_quantile": 0.50,
            "mutual_exclusion_axis": "market_state",
        }
        for name, direction, quantile in AGGREGATE_MARKET_STATE_SPECS
    ]
    return states, profile


def _fit_lob_context_states(
    days: list[dict[str, object]],
    partition: np.ndarray,
    *,
    balanced: bool = False,
    high_only: bool = False,
) -> list[dict[str, object]]:
    """Freeze nondegenerate pre-event context states using D_fit only."""

    channels: dict[str, tuple[int, str]] = {
        "relative_spread": (2, "identity"),
        "depth_imbalance": (3, "identity"),
        "abs_depth_imbalance": (3, "absolute"),
        "cancel_fraction_30s": (4, "identity"),
        "trade_fraction_30s": (5, "identity"),
        "message_rate_30s": (6, "identity"),
    }
    fit_values: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    fit_weights: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    for payload in days:
        entity = int(payload["entity_code"])
        if int(partition[entity]) != 0:
            continue
        rows = payload.get("context_rows", [])
        if not isinstance(rows, list) or not rows:
            continue
        times = np.fromiter(
            (int(row[0]) for row in rows), dtype=np.int64, count=len(rows)
        )
        next_times = np.minimum(
            np.r_[times[1:], int(payload["session_end"]) + 1],
            int(payload["session_end"]) + 1,
        )
        weights = np.maximum(0, next_times - times).astype(np.float64)
        for channel, (column, transform) in channels.items():
            values = np.fromiter(
                (float(row[column]) for row in rows),
                dtype=np.float64,
                count=len(rows),
            )
            if transform == "absolute":
                values = np.abs(values)
            keep = np.isfinite(values) & (weights > 0.0)
            if np.any(keep):
                fit_values[channel].append(values[keep])
                fit_weights[channel].append(weights[keep])

    output: list[dict[str, object]] = []
    state_specs = (
        BALANCED_LOB_CONTEXT_STATE_SPECS if balanced else LOB_CONTEXT_STATE_SPECS
    )
    if high_only:
        state_specs = tuple(spec for spec in state_specs if spec[2] == "high")
    state_meanings = (
        BALANCED_LOB_CONTEXT_STATE_MEANINGS
        if balanced
        else LOB_CONTEXT_STATE_MEANINGS
    )
    for name, channel, direction, quantile in state_specs:
        if not fit_values[channel]:
            continue
        values = np.concatenate(fit_values[channel])
        weights = np.concatenate(fit_weights[channel])
        entry = _duration_weighted_quantile(values, weights, float(quantile))
        exit_ = _duration_weighted_quantile(values, weights, 0.50)
        scale = max(1.0, abs(entry), abs(exit_))
        separated = (
            entry > exit_ + 1.0e-12 * scale
            if direction == "high"
            else entry < exit_ - 1.0e-12 * scale
        )
        if not separated:
            continue
        output.append(
            {
                "name": name,
                "meaning": state_meanings[name],
                "channel": channel,
                "column": channels[channel][0],
                "transform": channels[channel][1],
                "direction": direction,
                "entry_threshold": float(entry),
                "exit_threshold": float(exit_),
                "entry_quantile": float(quantile),
                "exit_quantile": 0.50,
            }
        )
    return output


def _lob_context_transition_events(
    payload: dict[str, object],
    states: list[dict[str, object]],
    *,
    first_source_predicate: int,
) -> list[tuple[int, int, int, int]]:
    """Materialize hidden state entry/exit sources for one trading day."""

    rows = payload.get("context_rows", [])
    if not isinstance(rows, list) or not rows or not states:
        return []
    entity = int(payload["entity_code"])
    output: list[tuple[int, int, int, int]] = []
    for state_index, state in enumerate(states):
        active = False
        column = int(state["column"])
        high = state["direction"] == "high"
        entry = float(state["entry_threshold"])
        exit_ = float(state["exit_threshold"])
        entry_code = int(first_source_predicate) + 2 * state_index
        exit_code = entry_code + 1
        for row in rows:
            value = float(row[column])
            if state.get("transform") == "absolute":
                value = abs(value)
            if not math.isfinite(value):
                continue
            enter = value >= entry if high else value <= entry
            leave = value < exit_ if high else value > exit_
            if not active and enter:
                output.append((entity, int(row[0]), entry_code, int(row[1])))
                active = True
            elif active and leave:
                output.append((entity, int(row[0]), exit_code, int(row[1])))
                active = False
    return output


def _lob_market_state_transition_events(
    payload: dict[str, object],
    states: list[dict[str, object]],
    profile: dict[str, dict[str, object]],
    *,
    first_source_predicate: int,
) -> list[tuple[int, int, int, int]]:
    """Materialize hidden entry/exit sources for one categorical state axis."""

    rows = payload.get("context_rows", [])
    if not isinstance(rows, list) or not rows or len(states) != 2:
        return []
    scores = _market_stress_scores(rows, profile)
    entity = int(payload["entity_code"])
    stressed = False
    calm = False
    initialized = False
    output: list[tuple[int, int, int, int]] = []
    for row, score in zip(rows, scores.tolist(), strict=True):
        if not math.isfinite(score):
            continue
        stressed_entry = float(states[0]["entry_threshold"])
        stressed_exit = float(states[0]["exit_threshold"])
        calm_entry = float(states[1]["entry_threshold"])
        calm_exit = float(states[1]["exit_threshold"])
        next_stressed = stressed
        next_calm = calm
        if not initialized:
            next_stressed = score >= stressed_entry
            next_calm = score <= calm_entry
            initialized = True
        else:
            if stressed and score < stressed_exit:
                next_stressed = False
            if calm and score > calm_exit:
                next_calm = False
            if not next_stressed and not next_calm:
                if score >= stressed_entry:
                    next_stressed = True
                elif score <= calm_entry:
                    next_calm = True
        if next_stressed and next_calm:
            raise AssertionError("market states are not mutually exclusive")
        primitive = int(row[1])
        tick = int(row[0])
        if next_stressed != stressed:
            output.append(
                (
                    entity,
                    tick,
                    int(first_source_predicate) + (0 if next_stressed else 1),
                    primitive,
                )
            )
        if next_calm != calm:
            output.append(
                (
                    entity,
                    tick,
                    int(first_source_predicate) + (2 if next_calm else 3),
                    primitive,
                )
            )
        stressed, calm = next_stressed, next_calm
    return output


def _lob_regime_transition_events(
    payload: dict[str, object],
    states: list[dict[str, object]],
    *,
    first_transition_predicate: int,
) -> list[tuple[int, int, int, int]]:
    """Collapse symmetric context changes into one stress/recovery event.

    Each state uses the D_fit-frozen hysteresis thresholds.  The first
    observation only initializes the state, because the earlier session state
    is unknown.  Later rows compare the number of active high-stress channels
    before and after the row.  A net increase emits stress, a net decrease
    emits recovery, and a tie emits no transition.
    """

    rows = payload.get("context_rows", [])
    if not isinstance(rows, list) or not rows or not states:
        return []
    if any(state["direction"] != "high" for state in states):
        raise ValueError("regime transitions require high-direction states")
    entity = int(payload["entity_code"])
    active: list[bool | None] = [None] * len(states)
    output: list[tuple[int, int, int, int]] = []
    for row in rows:
        net_change = 0
        for state_index, state in enumerate(states):
            value = float(row[int(state["column"])])
            if state.get("transform") == "absolute":
                value = abs(value)
            if not math.isfinite(value):
                continue
            entry = float(state["entry_threshold"])
            exit_ = float(state["exit_threshold"])
            if active[state_index] is None:
                active[state_index] = value >= entry
            elif not active[state_index] and value >= entry:
                active[state_index] = True
                net_change += 1
            elif active[state_index] and value < exit_:
                active[state_index] = False
                net_change -= 1
        if net_change == 0:
            continue
        predicate = int(first_transition_predicate) + (0 if net_change > 0 else 1)
        output.append((entity, int(row[0]), predicate, int(row[1])))
    return output


def _volatility_burst_targets(
    ticks: np.ndarray,
    volatility: np.ndarray,
    *,
    threshold: float,
    rearm_fraction: float,
) -> list[int]:
    """Decluster realized-volatility threshold crossings by hysteresis."""

    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("realized-volatility threshold must be positive")
    if not 0.0 < rearm_fraction < 1.0:
        raise ValueError("target rearm fraction must lie strictly between zero and one")
    rearm_level = float(rearm_fraction) * float(threshold)
    targets: list[int] = []
    armed = True
    for tick, value in zip(ticks.tolist(), volatility.tolist(), strict=True):
        if not armed and value < rearm_level:
            armed = True
        if armed and value > threshold:
            targets.append(int(tick))
            armed = False
    return targets


def _continuous_risk_frame(
    *,
    entity_code: int,
    session_start: int,
    session_end: int,
    weekday: int,
    events: list[tuple[int, int, int, int]],
    target_ticks: list[int],
    baseline_bins: int,
    impact_edges: np.ndarray,
    ticks_per_unit: int,
) -> pd.DataFrame:
    """Build the exact piecewise-constant continuous-time risk intervals."""

    terminal = np.int64(session_end) + np.int64(1)
    event_times = (
        np.unique(
            np.fromiter(
                (value[1] for value in events),
                dtype=np.int64,
                count=len(events),
            )
        )
        if events
        else np.zeros(0, dtype=np.int64)
    )
    target_boundaries = (
        np.unique(np.asarray(target_ticks, dtype=np.int64))
        if target_ticks
        else np.zeros(0, dtype=np.int64)
    )
    duration = int(terminal - session_start)
    baseline_boundaries = session_start + (
        np.arange(baseline_bins + 1, dtype=np.int64) * duration
    ) // int(baseline_bins)
    shifted = (
        (event_times[:, None] + np.int64(1) + impact_edges[None, :]).reshape(-1)
        if len(event_times)
        else np.zeros(0, dtype=np.int64)
    )
    boundaries = np.unique(
        np.concatenate(
            (
                np.asarray([session_start, terminal], dtype=np.int64),
                target_boundaries,
                baseline_boundaries,
                shifted,
            )
        )
    )
    boundaries = boundaries[(boundaries >= session_start) & (boundaries <= terminal)]
    if boundaries[0] != session_start or boundaries[-1] != terminal:
        raise AssertionError("continuous risk boundaries lost session bounds")
    left = boundaries[:-1]
    if int(ticks_per_unit) < 1:
        raise ValueError("ticks_per_unit must be positive")
    exposure = np.diff(boundaries).astype(np.float64) / float(ticks_per_unit)
    intraday = np.minimum(
        baseline_bins - 1,
        ((left - session_start) * baseline_bins) // max(1, duration),
    ).astype(np.int16)
    return pd.DataFrame(
        {
            "entity_code": np.full(len(left), entity_code, dtype=np.int32),
            "time": left,
            "baseline_stratum": (int(weekday) * baseline_bins + intraday).astype(
                np.int16
            ),
            "exposure": exposure,
        }
    )


def _ordered_partition(count: int, fractions: tuple[float, float, float]) -> np.ndarray:
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("partition fractions must sum to one")
    if count < 5:
        raise ValueError("WSELOB preprocessing requires at least five trading days")
    first = max(1, min(count - 2, int(round(fractions[0] * count))))
    second = max(first + 1, min(count - 1, int(round(sum(fractions[:2]) * count))))
    partition = np.full(count, 2, dtype=np.int8)
    partition[:first] = 0
    partition[first:second] = 1
    return partition


def _month_stratified_partition(
    dates: Sequence[pd.Timestamp],
    fractions: tuple[float, float, float],
    *,
    seed: int,
) -> np.ndarray:
    """Return a deterministic, label-blind split within each calendar month."""

    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("partition fractions must sum to one")
    if len(dates) < 5:
        raise ValueError("WSELOB preprocessing requires at least five trading days")
    months: dict[tuple[int, int], list[int]] = {}
    for index, raw_date in enumerate(dates):
        date = pd.Timestamp(raw_date)
        months.setdefault((int(date.year), int(date.month)), []).append(index)
    month_keys = tuple(sorted(months))
    sizes = np.asarray([len(months[key]) for key in month_keys], dtype=np.int32)
    if np.any(sizes < 3):
        raise ValueError(
            "month-stratified partition requires at least three trading days per month"
        )
    global_counts = np.bincount(
        _ordered_partition(len(dates), fractions), minlength=3
    ).astype(np.int32)

    def apportion(
        fraction: float,
        total: int,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        ideal = fraction * sizes.astype(np.float64)
        values = np.clip(np.floor(ideal).astype(np.int32), lower, upper)
        while int(values.sum()) < int(total):
            eligible = np.flatnonzero(values < upper)
            if not len(eligible):
                raise ValueError("month-stratified partition cannot reach split total")
            index = min(
                eligible.tolist(),
                key=lambda item: (-(ideal[item] - values[item]), month_keys[item]),
            )
            values[index] += 1
        while int(values.sum()) > int(total):
            eligible = np.flatnonzero(values > lower)
            if not len(eligible):
                raise ValueError("month-stratified partition exceeds split total")
            index = min(
                eligible.tolist(),
                key=lambda item: (-(values[item] - ideal[item]), month_keys[item]),
            )
            values[index] -= 1
        return values

    ones = np.ones(len(month_keys), dtype=np.int32)
    test_counts = apportion(
        float(fractions[2]),
        int(global_counts[2]),
        ones,
        sizes - 2,
    )
    cert_counts = apportion(
        float(fractions[1]),
        int(global_counts[1]),
        ones,
        sizes - test_counts - 1,
    )
    fit_counts = sizes - cert_counts - test_counts
    if not np.array_equal(
        np.asarray(
            [fit_counts.sum(), cert_counts.sum(), test_counts.sum()], dtype=np.int32
        ),
        global_counts,
    ):
        raise AssertionError("month-stratified global split totals changed")

    partition = np.empty(len(dates), dtype=np.int8)
    for month_index, key in enumerate(month_keys):
        indices = months[key]
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{int(seed)}:{pd.Timestamp(dates[index]).date().isoformat()}".encode(
                    "utf-8"
                )
            ).digest(),
        )
        fit_right = int(fit_counts[month_index])
        cert_right = fit_right + int(cert_counts[month_index])
        partition[np.asarray(ordered[:fit_right], dtype=np.int32)] = 0
        partition[np.asarray(ordered[fit_right:cert_right], dtype=np.int32)] = 1
        partition[np.asarray(ordered[cert_right:], dtype=np.int32)] = 2
    if not np.array_equal(np.unique(partition), np.asarray([0, 1, 2])):
        raise ValueError("month-stratified trading-day partition produced an empty split")
    return partition


def preprocess_wselob(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    stock: str = "PEKAO",
    bin_seconds: int = 5,
    continuous: bool = False,
    continuous_impact_seconds: int = 60,
    continuous_knot_count: int = 4,
    continuous_baseline_bins: int = 8,
    continuous_time_unit: str = "second",
    partition_fractions: tuple[float, float, float] = (0.5, 0.3, 0.2),
    partition_method: str = "ordered",
    partition_seed: int = 111,
    diagnostic_max_days: int | None = None,
    predicate_schema: str = "legacy",
    target_mode: str = "down_tick",
    target_horizon_seconds: int = 30,
    target_quantile: float = 0.90,
    target_rearm_fraction: float = 0.50,
    context_states: bool = False,
    overwrite: bool = False,
) -> Path:
    """Build stock-day recurrent down-move histories from WSELOB-2017."""

    stock = str(stock).upper()
    raw_root, output_root = Path(raw_root), Path(output_root)
    if stock not in WSELOB_FILES:
        raise ValueError(f"unsupported WSELOB stock: {stock}")
    if bin_seconds < 1:
        raise ValueError("bin_seconds must be positive")
    if continuous_impact_seconds < 1 or continuous_knot_count < 1:
        raise ValueError("continuous impact horizon and knot count must be positive")
    if continuous_impact_seconds < continuous_knot_count:
        raise ValueError("continuous impact horizon must span every kernel interval")
    if continuous_baseline_bins < 1:
        raise ValueError("continuous baseline bin count must be positive")
    if continuous_time_unit not in {"second", "millisecond"}:
        raise ValueError(
            "continuous_time_unit must be 'second' or 'millisecond'"
        )
    continuous_ticks_per_unit = (
        1_000_000 if continuous_time_unit == "millisecond" else 1_000_000_000
    )
    if predicate_schema not in {
        "legacy",
        "mechanism_v2",
        "mechanism_v3",
        "mechanism_v4",
        "mechanism_v5",
    }:
        raise ValueError(
            "predicate_schema must be 'legacy', 'mechanism_v2', "
            "'mechanism_v3', 'mechanism_v4' or 'mechanism_v5'"
        )
    mechanism_schema = predicate_schema in {
        "mechanism_v2",
        "mechanism_v3",
        "mechanism_v4",
        "mechanism_v5",
    }
    balanced_mechanisms = predicate_schema in {
        "mechanism_v3",
        "mechanism_v4",
        "mechanism_v5",
    }
    regime_transition_schema = predicate_schema == "mechanism_v4"
    aggregate_market_state_schema = predicate_schema == "mechanism_v5"
    base_predicate_names = (
        REGIME_MECHANISM_PREDICATES
        if regime_transition_schema
        else BALANCED_MECHANISM_PREDICATES
        if balanced_mechanisms
        else MECHANISM_PREDICATES
        if mechanism_schema
        else PREDICATES
    )
    base_predicate_meanings = (
        REGIME_MECHANISM_PREDICATE_MEANINGS
        if regime_transition_schema
        else BALANCED_MECHANISM_PREDICATE_MEANINGS
        if balanced_mechanisms
        else MECHANISM_PREDICATE_MEANINGS
        if mechanism_schema
        else PREDICATE_MEANINGS
    )
    if target_mode not in {"down_tick", "adverse_excursion", "volatility_burst"}:
        raise ValueError(
            "target_mode must be 'down_tick', 'adverse_excursion' or 'volatility_burst'"
        )
    if target_mode in {"adverse_excursion", "volatility_burst"} and (
        not continuous or not mechanism_schema
    ):
        raise ValueError(
            "window target modes require continuous mode and mechanism predicates"
        )
    if context_states and (
        not continuous
        or not mechanism_schema
        or target_mode not in {"adverse_excursion", "volatility_burst"}
    ):
        raise ValueError(
            "LOB context states require a continuous mechanism window-target dataset"
        )
    if target_horizon_seconds < 1:
        raise ValueError("target_horizon_seconds must be positive")
    if not 0.0 < target_quantile < 1.0:
        raise ValueError("target_quantile must lie strictly between zero and one")
    if not 0.0 < target_rearm_fraction < 1.0:
        raise ValueError("target_rearm_fraction must lie strictly between zero and one")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    order_path, trade_path = _raw_paths(raw_root, stock)
    for kind, path in zip(("orders", "trades"), (order_path, trade_path), strict=True):
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != WSELOB_FILES[stock][kind].sha256:
            raise ValueError(f"raw WSELOB digest mismatch: {path}")

    with (
        pd.HDFStore(order_path, mode="r") as order_store,
        pd.HDFStore(trade_path, mode="r") as trade_store,
    ):
        keys = sorted(set(order_store.keys()) & set(trade_store.keys()))
        if diagnostic_max_days is not None:
            if diagnostic_max_days < 5:
                raise ValueError("diagnostic_max_days must be at least five")
            keys = keys[: int(diagnostic_max_days)]
        entities_rows: list[dict[str, object]] = []
        all_events: list[tuple[int, int, int, int]] = []
        target_parts: list[pd.DataFrame] = []
        risk_parts: list[pd.DataFrame] = []
        daily_audit: list[dict[str, object]] = []
        excursion_days: list[dict[str, object]] = []
        bin_ns = int(bin_seconds) * 1_000_000_000
        continuous_edges = (
            np.arange(continuous_knot_count + 1, dtype=np.int64)
            * int(continuous_impact_seconds)
            * 1_000_000_000
        ) // int(continuous_knot_count)
        continuous_edges[-1] = int(continuous_impact_seconds) * 1_000_000_000
        for key in keys:
            trades = trade_store.select(
                key,
                columns=["time", "opening_trade_indicator", "trade_origin"],
            )
            core_trades = trades.loc[
                trades["opening_trade_indicator"].eq("S")
                & trades["trade_origin"].eq("B"),
                "time",
            ]
            if core_trades.empty:
                continue
            session_start = int(core_trades.min())
            session_end = int(core_trades.max())
            orders = order_store.select(key)
            entity_code = len(entities_rows)
            if mechanism_schema:
                context_rows: list[
                    tuple[int, int, float, float, float, float, float]
                ] = []
                events, mid_ticks, mid_twice, action_counts = _process_day_mechanisms(
                    orders,
                    entity_code=entity_code,
                    session_start_ns=session_start,
                    session_end_ns=session_end,
                    bin_nanoseconds=bin_ns,
                    continuous=continuous,
                    replenishment_lookback_ns=(
                        int(target_horizon_seconds) * 1_000_000_000
                    ),
                    trade_times=frozenset(
                        core_trades.to_numpy(dtype=np.int64).tolist()
                    ),
                    balanced_mechanisms=balanced_mechanisms,
                    context_rows=(
                        context_rows
                        if context_states
                        or regime_transition_schema
                        or aggregate_market_state_schema
                        else None
                    ),
                )
                target_ticks: list[int] = []
            else:
                events, target_ticks, action_counts = _process_day(
                    orders,
                    entity_code=entity_code,
                    session_start_ns=session_start,
                    session_end_ns=session_end,
                    bin_nanoseconds=bin_ns,
                    continuous=continuous,
                )
            end_time = (
                session_end
                if continuous
                else int((session_end - session_start) // bin_ns)
            )
            start_time = session_start if continuous else 0
            if end_time <= start_time:
                continue
            day = str(key).removeprefix("/d")
            date = pd.Timestamp(day)
            entities_rows.append(
                {
                    "entity_id": f"{stock}:{day}",
                    "start_time": start_time,
                    "end_time": end_time,
                    "baseline_origin": start_time,
                    "split_group": int(day),
                    "baseline_stratum": int(date.dayofweek),
                    "dependency_group": day,
                }
            )
            all_events.extend(events)
            if target_mode in {"adverse_excursion", "volatility_burst"}:
                excursion_days.append(
                    {
                        "entity_code": entity_code,
                        "session_start": session_start,
                        "session_end": session_end,
                        "date": date,
                        "events": events,
                        "mid_ticks": mid_ticks,
                        "mid_twice": mid_twice,
                        "context_rows": (
                            context_rows
                            if context_states
                            or regime_transition_schema
                            or aggregate_market_state_schema
                            else []
                        ),
                    }
                )
            elif target_ticks:
                values, counts = np.unique(
                    np.asarray(target_ticks, dtype=np.int64), return_counts=True
                )
                target_parts.append(
                    pd.DataFrame(
                        {
                            "entity_code": np.full(
                                len(values), entity_code, dtype=np.int32
                            ),
                            "time": values,
                            "multiplicity": counts.astype(np.int32),
                        }
                    )
                )
            if continuous and target_mode == "down_tick":
                terminal = np.int64(session_end) + np.int64(1)
                event_times = (
                    np.unique(
                        np.fromiter(
                            (value[1] for value in events),
                            dtype=np.int64,
                            count=len(events),
                        )
                    )
                    if events
                    else np.zeros(0, dtype=np.int64)
                )
                target_boundaries = (
                    np.unique(np.asarray(target_ticks, dtype=np.int64))
                    if target_ticks
                    else np.zeros(0, dtype=np.int64)
                )
                duration = int(terminal - session_start)
                baseline_boundaries = session_start + (
                    np.arange(continuous_baseline_bins + 1, dtype=np.int64) * duration
                ) // int(continuous_baseline_bins)
                shifted = (
                    (
                        event_times[:, None] + np.int64(1) + continuous_edges[None, :]
                    ).reshape(-1)
                    if len(event_times)
                    else np.zeros(0, dtype=np.int64)
                )
                boundaries = np.unique(
                    np.concatenate(
                        (
                            np.asarray([session_start, terminal], dtype=np.int64),
                            target_boundaries,
                            baseline_boundaries,
                            shifted,
                        )
                    )
                )
                boundaries = boundaries[
                    (boundaries >= session_start) & (boundaries <= terminal)
                ]
                if boundaries[0] != session_start or boundaries[-1] != terminal:
                    raise AssertionError(
                        "continuous risk boundaries lost session bounds"
                    )
                left = boundaries[:-1]
                exposure = np.diff(boundaries).astype(np.float64) / float(
                    continuous_ticks_per_unit
                )
                intraday = np.minimum(
                    continuous_baseline_bins - 1,
                    ((left - session_start) * continuous_baseline_bins)
                    // max(1, duration),
                ).astype(np.int16)
                risk_parts.append(
                    pd.DataFrame(
                        {
                            "entity_code": np.full(
                                len(left), entity_code, dtype=np.int32
                            ),
                            "time": left,
                            "baseline_stratum": (
                                int(date.dayofweek) * continuous_baseline_bins
                                + intraday
                            ).astype(np.int16),
                            "exposure": exposure,
                        }
                    )
                )
            daily_audit.append(
                {
                    "date": day,
                    "order_messages": int(len(orders)),
                    "core_book_trades": int(len(core_trades)),
                    "risk_ticks": (
                        int(len(risk_parts[-1]))
                        if continuous and target_mode == "down_tick"
                        else end_time + 1
                        if not continuous
                        else 0
                    ),
                    "target_events": int(len(target_ticks)),
                    "predicate_events": int(len(events)),
                    "action_counts": action_counts,
                }
            )

    entities = pd.DataFrame(entities_rows)
    if entities.empty:
        raise ValueError("WSELOB input contains no continuous-session trading days")
    if partition_method == "ordered":
        partition = _ordered_partition(len(entities), partition_fractions)
    elif partition_method == "month_stratified":
        partition = _month_stratified_partition(
            tuple(pd.Timestamp(value.split(":", 1)[1]) for value in entities["entity_id"]),
            partition_fractions,
            seed=partition_seed,
        )
    else:
        raise ValueError(f"unsupported WSELOB partition method: {partition_method}")
    entities["partition"] = partition
    context_state_specs: list[dict[str, object]] = []
    regime_state_specs: list[dict[str, object]] = []
    market_state_profile: dict[str, dict[str, object]] = {}
    if aggregate_market_state_schema:
        context_state_specs, market_state_profile = _fit_lob_market_states(
            excursion_days,
            partition,
        )
        first_source_predicate = len(base_predicate_names) + len(context_state_specs)
        for payload in excursion_days:
            transitions = _lob_market_state_transition_events(
                payload,
                context_state_specs,
                market_state_profile,
                first_source_predicate=first_source_predicate,
            )
            if transitions:
                payload["events"].extend(transitions)
                all_events.extend(transitions)
                entity_code = int(payload["entity_code"])
                daily_audit[entity_code]["predicate_events"] += len(transitions)
    elif regime_transition_schema:
        regime_state_specs = _fit_lob_context_states(
            excursion_days,
            partition,
            balanced=True,
            high_only=True,
        )
        if not regime_state_specs:
            raise ValueError("D_fit produced no nondegenerate LOB regime channel")
        first_transition_predicate = len(BALANCED_MECHANISM_PREDICATES)
        for payload in excursion_days:
            transitions = _lob_regime_transition_events(
                payload,
                regime_state_specs,
                first_transition_predicate=first_transition_predicate,
            )
            transition_by_primitive = {event[3]: event for event in transitions}
            original_events = payload["events"]
            payload["events"] = sorted(
                (
                    event
                    for event in original_events
                    if event[3] not in transition_by_primitive
                ),
                key=lambda event: (event[1], event[2], event[3]),
            ) + list(transitions)
            payload["events"].sort(key=lambda event: (event[1], event[2], event[3]))
            entity_code = int(payload["entity_code"])
            daily_audit[entity_code]["predicate_events"] = len(payload["events"])
        # The original action events were appended before D_fit thresholds
        # were available.  Rebuild once so replaced primitives occur exactly
        # once in the final event table.
        all_events = [
            event for payload in excursion_days for event in payload["events"]
        ]
    elif context_states:
        context_state_specs = _fit_lob_context_states(
            excursion_days,
            partition,
            balanced=balanced_mechanisms,
        )
        if not context_state_specs:
            raise ValueError("D_fit produced no nondegenerate LOB context state")
        first_source_predicate = len(base_predicate_names) + len(context_state_specs)
        for payload in excursion_days:
            transitions = _lob_context_transition_events(
                payload,
                context_state_specs,
                first_source_predicate=first_source_predicate,
            )
            if transitions:
                payload["events"].extend(transitions)
                all_events.extend(transitions)
                entity_code = int(payload["entity_code"])
                daily_audit[entity_code]["predicate_events"] += len(transitions)
    excursion_threshold: float | None = None
    if target_mode in {"adverse_excursion", "volatility_burst"}:
        horizon_ns = int(target_horizon_seconds) * 1_000_000_000
        fit_values: list[np.ndarray] = []
        fit_weights: list[np.ndarray] = []
        for payload in excursion_days:
            if target_mode == "adverse_excursion":
                process_ticks, process_values = _adverse_excursion_returns(
                    payload["mid_ticks"],
                    payload["mid_twice"],
                    session_start_ns=int(payload["session_start"]),
                    horizon_ns=horizon_ns,
                )
            else:
                process_ticks, process_values = _realized_volatility(
                    payload["mid_ticks"],
                    payload["mid_twice"],
                    session_start_ns=int(payload["session_start"]),
                    session_end_ns=int(payload["session_end"]),
                    horizon_ns=horizon_ns,
                )
            payload["process_ticks"] = process_ticks
            payload["process_values"] = process_values
            if partition[int(payload["entity_code"])] == 0:
                complete = process_ticks >= (
                    int(payload["session_start"]) + int(horizon_ns)
                )
                values = (
                    -process_values[complete & (process_values < 0.0)]
                    if target_mode == "adverse_excursion"
                    else process_values[complete]
                )
                if len(values):
                    fit_values.append(values)
                    if target_mode == "volatility_burst":
                        next_ticks = np.minimum(
                            np.r_[process_ticks[1:], int(payload["session_end"]) + 1],
                            int(payload["session_end"]) + 1,
                        )
                        durations = np.maximum(0, next_ticks - process_ticks).astype(
                            np.float64
                        )
                        fit_weights.append(durations[complete])
        if not fit_values:
            raise ValueError("D_fit contains no eligible target-process values")
        excursion_threshold = (
            _duration_weighted_quantile(
                np.concatenate(fit_values),
                np.concatenate(fit_weights),
                target_quantile,
            )
            if target_mode == "volatility_burst"
            else float(np.quantile(np.concatenate(fit_values), target_quantile))
        )
        if not math.isfinite(excursion_threshold) or excursion_threshold <= 0.0:
            raise ValueError("D_fit target-process threshold is not positive")
        for payload in excursion_days:
            target_ticks = (
                _volatility_burst_targets(
                    payload["process_ticks"],
                    payload["process_values"],
                    threshold=excursion_threshold,
                    rearm_fraction=target_rearm_fraction,
                )
                if target_mode == "volatility_burst"
                else _adverse_excursion_targets(
                    payload["process_ticks"],
                    payload["process_values"],
                    threshold=excursion_threshold,
                    rearm_fraction=target_rearm_fraction,
                )
            )
            entity_code = int(payload["entity_code"])
            if target_ticks:
                target_parts.append(
                    pd.DataFrame(
                        {
                            "entity_code": np.full(
                                len(target_ticks), entity_code, dtype=np.int32
                            ),
                            "time": np.asarray(target_ticks, dtype=np.int64),
                            "multiplicity": np.ones(len(target_ticks), dtype=np.int32),
                        }
                    )
                )
            frame = _continuous_risk_frame(
                entity_code=entity_code,
                session_start=int(payload["session_start"]),
                session_end=int(payload["session_end"]),
                weekday=int(payload["date"].dayofweek),
                events=payload["events"],
                target_ticks=target_ticks,
                baseline_bins=continuous_baseline_bins,
                impact_edges=continuous_edges,
                ticks_per_unit=continuous_ticks_per_unit,
            )
            risk_parts.append(frame)
            daily_audit[entity_code]["risk_ticks"] = int(len(frame))
            daily_audit[entity_code]["target_events"] = int(len(target_ticks))
    events = pd.DataFrame.from_records(
        all_events,
        columns=["entity_code", "time", "predicate_code", "primitive_event_id"],
    )
    events = events.astype(
        {
            "entity_code": "int32",
            "time": "int64",
            "predicate_code": "int16",
            "primitive_event_id": "int64",
        }
    )
    targets = (
        pd.concat(target_parts, ignore_index=True)
        if target_parts
        else pd.DataFrame(
            {
                "entity_code": pd.Series(dtype="int32"),
                "time": pd.Series(dtype="int64"),
                "multiplicity": pd.Series(dtype="int32"),
            }
        )
    )
    baseline_cells = None
    if continuous:
        baseline_cells = pd.concat(risk_parts, ignore_index=True)
        # Some diagnostic subsets may not contain every weekday.  Compress
        # only the nuisance labels; their weekday/intraday meaning is already
        # immutable in provenance and no observation or target enters here.
        observed = np.sort(baseline_cells["baseline_stratum"].unique())
        remap = {int(value): index for index, value in enumerate(observed.tolist())}
        baseline_cells["baseline_stratum"] = (
            baseline_cells["baseline_stratum"].map(remap).astype(np.int16)
        )
    predicate_names = base_predicate_names
    predicate_meanings = base_predicate_meanings
    predicate_roles: tuple[str, ...] | None = None
    predicate_definitions: tuple[dict[str, object], ...] | None = None
    if context_state_specs:
        state_names = tuple(str(item["name"]) for item in context_state_specs)
        source_names = tuple(
            name
            for state in state_names
            for name in (f"_{state}_entry", f"_{state}_exit")
        )
        first_state = len(base_predicate_names)
        first_source = first_state + len(state_names)
        predicate_names = (*base_predicate_names, *state_names, *source_names)
        predicate_meanings = (
            *base_predicate_meanings,
            *(str(item["meaning"]) for item in context_state_specs),
            *(
                "internal target-blind context-state transition source"
                for _ in source_names
            ),
        )
        predicate_roles = (
            *("reported" for _ in range(first_source)),
            *("state_source" for _ in source_names),
        )
        definitions: list[dict[str, object]] = [
            {
                "kind": "event",
                "meaning": meaning,
                "construction": (
                    "online order-book state using current and past messages only"
                ),
            }
            for meaning in base_predicate_meanings
        ]
        for index, state in enumerate(context_state_specs):
            definitions.append(
                {
                    "kind": "transition_state",
                    "meaning": state["meaning"],
                    "entry_predicate": first_source + 2 * index,
                    "exit_predicate": first_source + 2 * index + 1,
                    "channel": state["channel"],
                    "transform": state.get("transform", "identity"),
                    "direction": state["direction"],
                    "entry_threshold": state["entry_threshold"],
                    "exit_threshold": state["exit_threshold"],
                    "threshold_partition": "fit",
                    "activation": "strictly_after_entry_through_exit",
                }
            )
        definitions.extend(
            {
                "kind": "event",
                "meaning": "internal target-blind context-state transition source",
            }
            for _ in source_names
        )
        predicate_definitions = tuple(definitions)
    predicate_counts = (
        events.groupby("predicate_code")
        .size()
        .reindex(range(len(predicate_names)), fill_value=0)
    )
    target_events_by_split = np.bincount(
        partition[targets["entity_code"].to_numpy(dtype=np.int32)],
        weights=targets["multiplicity"].to_numpy(dtype=np.int64),
        minlength=3,
    ).astype(np.int64)
    # The continuous fused evaluator slices one contiguous entity range per
    # split.  A month-stratified assignment is intentionally nonchronological,
    # so store its unchanged membership in fit/cert/test order and remap only
    # the internal entity codes.  Dates, dependency groups, events, targets,
    # exposures and every statistical decision remain byte-for-byte attached
    # to the same trading day.
    if partition_method == "month_stratified":
        storage_order = np.argsort(partition, kind="stable")
        inverse = np.empty(len(storage_order), dtype=np.int32)
        inverse[storage_order] = np.arange(len(storage_order), dtype=np.int32)
        entities = entities.iloc[storage_order].reset_index(drop=True)
        partition = partition[storage_order]
        entities["partition"] = partition
        for frame in (events, targets, baseline_cells):
            if frame is None or frame.empty:
                continue
            old_codes = frame["entity_code"].to_numpy(dtype=np.int32)
            frame["entity_code"] = inverse[old_codes]
            frame.sort_values(["entity_code", "time"], kind="stable", inplace=True)
            frame.reset_index(drop=True, inplace=True)
        daily_audit = [daily_audit[int(index)] for index in storage_order]
    output = write_dataset(
        output_root,
        entities=entities,
        events=events,
        targets=targets,
        baseline_cells=baseline_cells,
        predicate_names=predicate_names,
        predicate_roles=predicate_roles,
        predicate_definitions=(
            predicate_definitions
            if predicate_definitions is not None
            else (
                {
                    "kind": "event",
                    "meaning": meaning,
                    "construction": (
                        "online order-book state using current and past messages only"
                    ),
                }
                for meaning in predicate_meanings
            )
        ),
        likelihood="continuous_poisson" if continuous else "poisson",
        time_unit=(
            continuous_time_unit
            if continuous
            else f"{bin_seconds}-second interval"
        ),
        ticks_per_unit=continuous_ticks_per_unit if continuous else 1,
        adverse_event_name=(
            "strictly-future 30-second realized-volatility burst onset"
            if target_mode == "volatility_burst"
            else "strictly-future 30-second adverse mid-price excursion onset"
            if target_mode == "adverse_excursion"
            else "strictly-future mid-price down-tick"
        ),
        f0_contract={
            "dynamic_predicates": True,
            "outcome_blind_predicate_construction": True,
            "direct_target_proxy_excluded_from_reported_dictionary": True,
            "strict_future_effect_required": True,
            "atomic_predicates": True,
            "primitive_event_provenance": True,
            "independent_certification_units": True,
            "target_threshold_fit_only": target_mode
            in {"adverse_excursion", "volatility_burst"},
            **(
                {
                    "required_impact_lag": int(continuous_impact_seconds)
                    * (1000 if continuous_time_unit == "millisecond" else 1),
                    "required_kernel_knots": int(continuous_knot_count),
                }
                if continuous
                else {}
            ),
        },
        provenance={
            "preprocessor": (
                "crbstpp.preprocess.wselob.continuous_volatility_market_state_mechanism.v6"
                if target_mode == "volatility_burst"
                and aggregate_market_state_schema
                else
                "crbstpp.preprocess.wselob.continuous_volatility_regime_mechanism.v5"
                if target_mode == "volatility_burst" and regime_transition_schema
                else
                "crbstpp.preprocess.wselob.continuous_volatility_balanced_mechanism.v4"
                if target_mode == "volatility_burst" and balanced_mechanisms
                else
                "crbstpp.preprocess.wselob.continuous_volatility_mechanism.v3"
                if target_mode == "volatility_burst"
                else "crbstpp.preprocess.wselob.continuous_adverse_excursion_mechanism.v2"
                if target_mode == "adverse_excursion"
                else "crbstpp.preprocess.wselob.continuous_stock_day_microstructure.v1"
                if continuous
                else "crbstpp.preprocess.wselob.stock_day_microstructure.v1"
            ),
            "dataset": "WSELOB-2017",
            "doi": DATASET_DOI,
            "dataset_page": DATASET_PAGE,
            "license": "CC BY 4.0",
            "stock": stock,
            "raw_files": {
                "orders": {
                    "path": str(order_path),
                    "sha256": WSELOB_FILES[stock]["orders"].sha256,
                },
                "trades": {
                    "path": str(trade_path),
                    "sha256": WSELOB_FILES[stock]["trades"].sha256,
                },
            },
            "entity_definition": "stock by continuous-session trading day",
            "target_definition": (
                "one recurrent onset when past-only 30-second realized volatility "
                "crosses above its duration-weighted D_fit quantile; the process "
                "rearms only after falling below half the threshold; source effects "
                "start one nanosecond later"
                if target_mode == "volatility_burst"
                else "one recurrent onset when the past-only 30-second log mid-price return "
                "crosses below the D_fit-frozen adverse-return quantile; the process "
                "rearms only after half-threshold recovery; source effects start one "
                "nanosecond later"
                if target_mode == "adverse_excursion"
                else "one recurrent event at its raw exchange timestamp whenever reconstructed "
                "best-bid plus best-ask strictly decreases; source effects start one "
                "nanosecond later"
                if continuous
                else "one recurrent event whenever reconstructed best-bid plus best-ask "
                "strictly decreases; same-bin predicate effects are forbidden by the "
                "strict-future model"
            ),
            "target_excursion": (
                {
                    "horizon_seconds": int(target_horizon_seconds),
                    "fit_quantile": float(target_quantile),
                    "frozen_log_return_threshold": excursion_threshold,
                    "rearm_fraction": float(target_rearm_fraction),
                    "threshold_partition": "fit",
                }
                if target_mode == "adverse_excursion"
                else None
            ),
            "target_volatility": (
                {
                    "horizon_seconds": int(target_horizon_seconds),
                    "estimator": "square root of rolling squared log-mid jumps",
                    "fit_quantile": float(target_quantile),
                    "quantile_weighting": "continuous-time duration",
                    "frozen_threshold": excursion_threshold,
                    "rearm_fraction": float(target_rearm_fraction),
                    "threshold_partition": "fit",
                }
                if target_mode == "volatility_burst"
                else None
            ),
            "predicate_definition": (
                "one target-blind direction-free action per raw message plus one "
                "mutually exclusive D_fit-frozen calm/neutral/stressed state axis"
                if aggregate_market_state_schema
                else
                "one target-blind direction-free action or aggregate stress/recovery "
                "transition per raw message; a transition replaces the coincident action"
                if regime_transition_schema
                else
                "one target-blind direction-free liquidity mechanism per raw message, "
                "with passive additions and removals separated by whether they restore "
                "or worsen top-of-book depth balance"
                if balanced_mechanisms
                else
                "one mutually exclusive target-blind order-flow, liquidity-withdrawal, "
                "queue-exhaustion, or replenishment mechanism per raw message"
                if mechanism_schema
                else "target-blind order action and liquidity-state transitions; past "
                "mid-price direction is excluded from the reported dictionary"
            ),
            "predicate_schema": predicate_schema,
            "context_states": {
                "enabled": bool(context_state_specs),
                "threshold_partition": "fit" if context_state_specs else None,
                "states": context_state_specs,
                "construction": (
                    "past-only pre-event state with D_fit-frozen quantile entry and median exit"
                    if context_state_specs
                    else None
                ),
            },
            "regime_transitions": {
                "enabled": bool(regime_transition_schema),
                "threshold_partition": "fit" if regime_transition_schema else None,
                "channels": regime_state_specs,
                "construction": (
                    "net change in the number of active high-stress channels with "
                    "D_fit-frozen quantile entry and median exit thresholds"
                    if regime_transition_schema
                    else None
                ),
                "primitive_contract": (
                    "a transition replaces the action at the same primitive event"
                    if regime_transition_schema
                    else None
                ),
            },
            "aggregate_market_state": {
                "enabled": bool(aggregate_market_state_schema),
                "threshold_partition": (
                    "fit" if aggregate_market_state_schema else None
                ),
                "channel_profiles": (
                    market_state_profile if aggregate_market_state_schema else {}
                ),
                "states": (
                    context_state_specs if aggregate_market_state_schema else []
                ),
                "construction": (
                    "equal-weight mean of five duration-weighted D_fit percentile "
                    "channels; Q25/Q75 entry and median exit"
                    if aggregate_market_state_schema
                    else None
                ),
            },
            "session_definition": (
                "from first to last core continuous book trade (indicator S, origin B); "
                "opening auction and book retransmission are reconstruction-only"
            ),
            "time_discretization": {
                "raw": "nanosecond exchange timestamps",
                "mode": "continuous piecewise-constant Poisson"
                if continuous
                else "fixed-bin recurrent Poisson",
                "likelihood_bin_seconds": None if continuous else int(bin_seconds),
                "ticks_per_second": 1_000_000_000 if continuous else None,
                "model_ticks_per_unit": (
                    continuous_ticks_per_unit if continuous else None
                ),
                "strict_future_offset_nanoseconds": 1 if continuous else None,
                "impact_horizon_seconds": (
                    int(continuous_impact_seconds) if continuous else None
                ),
                "kernel_edges_nanoseconds": (
                    continuous_edges.tolist() if continuous else None
                ),
            },
            "partition": {
                "method": (
                    "ordered trading-day split"
                    if partition_method == "ordered"
                    else "calendar-month-stratified deterministic trading-day split"
                ),
                "fractions": list(map(float, partition_fractions)),
                "seed": int(partition_seed),
                "counts": [int(np.sum(partition == code)) for code in range(3)],
                "storage_order": "partition_contiguous_then_original_trading_day",
            },
            "baseline": (
                f"weekday crossed with {continuous_baseline_bins} fixed intraday bins; "
                "no order behavior or target history"
                if continuous
                else "weekday stratum crossed downstream with pre-registered intraday "
                "time bins; no order behavior or target history"
            ),
            "diagnostic_max_days": diagnostic_max_days,
            "trading_days": int(len(entities)),
            "risk_ticks": (
                int(len(baseline_cells))
                if continuous and baseline_cells is not None
                else int(np.sum(entities["end_time"].to_numpy() + 1))
            ),
            "target_events": int(targets["multiplicity"].sum()),
            "target_events_by_partition": target_events_by_split.tolist(),
            "predicate_event_counts": {
                name: int(predicate_counts.iloc[code])
                for code, name in enumerate(predicate_names)
            },
        },
    )
    predicate_rows: list[dict[str, object]] = []
    event_entity = events["entity_code"].to_numpy(dtype=np.int32)
    for code, name in enumerate(predicate_names):
        keep = events["predicate_code"].eq(code).to_numpy()
        codes = event_entity[keep]
        predicate_rows.append(
            {
                "code": code,
                "name": name,
                "meaning": predicate_meanings[code],
                "events": int(np.sum(keep)),
                "trading_days": int(np.unique(codes).size),
                "events_by_partition": [
                    int(np.sum(partition[codes] == split)) for split in range(3)
                ],
            }
        )
    audit = {
        "schema": "crbstpp.wselob.predicate_audit.v1",
        "stock": stock,
        "bin_seconds": int(bin_seconds),
        "continuous": bool(continuous),
        "continuous_impact_seconds": (
            int(continuous_impact_seconds) if continuous else None
        ),
        "continuous_knot_count": int(continuous_knot_count) if continuous else None,
        "trading_days": int(len(entities)),
        "target_events_by_partition": {
            name: int(target_events_by_split[code])
            for code, name in enumerate(PARTITION_NAMES)
        },
        "predicates": predicate_rows,
        "daily": daily_audit,
    }
    (output / "predicate_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
