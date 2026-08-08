from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

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
    for row_number, row in enumerate(orders[columns].itertuples(index=False, name=None)):
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
                action_predicate = BUY_MARKET if is_bid else SELL_MARKET if is_ask else None
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
    for row_number, row in enumerate(orders[columns].itertuples(index=False, name=None)):
        time_ns, price, aggregate, side, order_type, action = row
        time_ns = int(time_ns)
        price = int(price)
        aggregate = int(aggregate)
        side = int(side)
        order_type = str(order_type)
        action = str(action)
        action_counts[action] = action_counts.get(action, 0) + 1
        before_bid, _ = bids.best()
        before_ask, _ = asks.best()
        in_session = session_start_ns <= time_ns <= session_end_ns
        is_bid = side == 1
        is_ask = side in (2, 5)

        if action == "F":
            bids.clear()
            asks.clear()
        elif action in {"A", "M", "D", "Y"} and price > 0:
            if is_bid:
                bids.set(price, aggregate)
            elif is_ask:
                asks.set(price, aggregate)
        after_bid, _ = bids.best()
        after_ask, _ = asks.best()
        if not in_session:
            continue

        predicate: int | None = None
        if action == "A" and order_type == "1":
            predicate = (
                MECH_BUY_MARKET
                if is_bid
                else MECH_SELL_MARKET
                if is_ask
                else None
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
            execution = action == "M" or time_ns in trade_times
            if is_bid and before_bid is not None and price == before_bid:
                cleared = after_bid is None or after_bid < before_bid
                if cleared:
                    predicate = (
                        MECH_TRADE_BID_CLEAR
                        if execution
                        else MECH_CANCEL_BID_CLEAR
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
                        MECH_TRADE_ASK_CLEAR
                        if execution
                        else MECH_CANCEL_ASK_CLEAR
                    )
                    last_ask_clear = (time_ns, before_ask)
                else:
                    predicate = (
                        MECH_ASK_FILL_NONCLEAR
                        if execution
                        else MECH_ASK_CANCEL_NONCLEAR
                    )

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
    boundaries = boundaries[
        (boundaries >= session_start) & (boundaries <= terminal)
    ]
    if boundaries[0] != session_start or boundaries[-1] != terminal:
        raise AssertionError("continuous risk boundaries lost session bounds")
    left = boundaries[:-1]
    exposure = np.diff(boundaries).astype(np.float64) / 1_000_000_000.0
    intraday = np.minimum(
        baseline_bins - 1,
        ((left - session_start) * baseline_bins) // max(1, duration),
    ).astype(np.int16)
    return pd.DataFrame(
        {
            "entity_code": np.full(len(left), entity_code, dtype=np.int32),
            "time": left,
            "baseline_stratum": (
                int(weekday) * baseline_bins + intraday
            ).astype(np.int16),
            "exposure": exposure,
        }
    )


def _ordered_partition(
    count: int, fractions: tuple[float, float, float]
) -> np.ndarray:
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
    partition_fractions: tuple[float, float, float] = (0.5, 0.3, 0.2),
    diagnostic_max_days: int | None = None,
    predicate_schema: str = "legacy",
    target_mode: str = "down_tick",
    target_horizon_seconds: int = 30,
    target_quantile: float = 0.90,
    target_rearm_fraction: float = 0.50,
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
    if predicate_schema not in {"legacy", "mechanism_v2"}:
        raise ValueError("predicate_schema must be 'legacy' or 'mechanism_v2'")
    if target_mode not in {"down_tick", "adverse_excursion"}:
        raise ValueError("target_mode must be 'down_tick' or 'adverse_excursion'")
    if target_mode == "adverse_excursion" and (
        not continuous or predicate_schema != "mechanism_v2"
    ):
        raise ValueError(
            "adverse_excursion requires continuous mode and mechanism_v2 predicates"
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

    with pd.HDFStore(order_path, mode="r") as order_store, pd.HDFStore(
        trade_path, mode="r"
    ) as trade_store:
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
            if predicate_schema == "mechanism_v2":
                events, mid_ticks, mid_twice, action_counts = (
                    _process_day_mechanisms(
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
                    )
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
            if target_mode == "adverse_excursion":
                excursion_days.append(
                    {
                        "entity_code": entity_code,
                        "session_start": session_start,
                        "session_end": session_end,
                        "date": date,
                        "events": events,
                        "mid_ticks": mid_ticks,
                        "mid_twice": mid_twice,
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
                    np.arange(continuous_baseline_bins + 1, dtype=np.int64)
                    * duration
                ) // int(continuous_baseline_bins)
                shifted = (
                    (
                        event_times[:, None]
                        + np.int64(1)
                        + continuous_edges[None, :]
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
                    raise AssertionError("continuous risk boundaries lost session bounds")
                left = boundaries[:-1]
                exposure = np.diff(boundaries).astype(np.float64) / 1_000_000_000.0
                intraday = np.minimum(
                    continuous_baseline_bins - 1,
                    (
                        (left - session_start) * continuous_baseline_bins
                    ) // max(1, duration),
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
    partition = _ordered_partition(len(entities), partition_fractions)
    entities["partition"] = partition
    excursion_threshold: float | None = None
    if target_mode == "adverse_excursion":
        horizon_ns = int(target_horizon_seconds) * 1_000_000_000
        fit_magnitudes: list[np.ndarray] = []
        for payload in excursion_days:
            return_ticks, return_values = _adverse_excursion_returns(
                payload["mid_ticks"],
                payload["mid_twice"],
                session_start_ns=int(payload["session_start"]),
                horizon_ns=horizon_ns,
            )
            payload["return_ticks"] = return_ticks
            payload["return_values"] = return_values
            if partition[int(payload["entity_code"])] == 0:
                negative = -return_values[return_values < 0.0]
                if len(negative):
                    fit_magnitudes.append(negative)
        if not fit_magnitudes:
            raise ValueError("D_fit contains no negative horizon mid-price returns")
        excursion_threshold = float(
            np.quantile(np.concatenate(fit_magnitudes), target_quantile)
        )
        if not math.isfinite(excursion_threshold) or excursion_threshold <= 0.0:
            raise ValueError("D_fit adverse-excursion threshold is not positive")
        for payload in excursion_days:
            target_ticks = _adverse_excursion_targets(
                payload["return_ticks"],
                payload["return_values"],
                threshold=excursion_threshold,
                rearm_fraction=target_rearm_fraction,
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
    predicate_names = (
        MECHANISM_PREDICATES if predicate_schema == "mechanism_v2" else PREDICATES
    )
    predicate_meanings = (
        MECHANISM_PREDICATE_MEANINGS
        if predicate_schema == "mechanism_v2"
        else PREDICATE_MEANINGS
    )
    predicate_counts = events.groupby("predicate_code").size().reindex(
        range(len(predicate_names)), fill_value=0
    )
    target_events_by_split = np.bincount(
        partition[targets["entity_code"].to_numpy(dtype=np.int32)],
        weights=targets["multiplicity"].to_numpy(dtype=np.int64),
        minlength=3,
    ).astype(np.int64)
    output = write_dataset(
        output_root,
        entities=entities,
        events=events,
        targets=targets,
        baseline_cells=baseline_cells,
        predicate_names=predicate_names,
        predicate_definitions=(
            {
                "kind": "event",
                "meaning": meaning,
                "construction": "online order-book state using current and past messages only",
            }
            for meaning in predicate_meanings
        ),
        likelihood="continuous_poisson" if continuous else "poisson",
        time_unit="second" if continuous else f"{bin_seconds}-second interval",
        ticks_per_unit=1_000_000_000 if continuous else 1,
        adverse_event_name=(
            "strictly-future 30-second adverse mid-price excursion onset"
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
            "target_threshold_fit_only": target_mode == "adverse_excursion",
            **(
                {
                    "required_impact_lag": int(continuous_impact_seconds),
                    "required_kernel_knots": int(continuous_knot_count),
                }
                if continuous
                else {}
            ),
        },
        provenance={
            "preprocessor": (
                "crbstpp.preprocess.wselob.continuous_adverse_excursion_mechanism.v2"
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
                "one recurrent onset when the past-only 30-second log mid-price return "
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
            "predicate_definition": (
                "one mutually exclusive target-blind order-flow, liquidity-withdrawal, "
                "queue-exhaustion, or replenishment mechanism per raw message"
                if predicate_schema == "mechanism_v2"
                else "target-blind order action and liquidity-state transitions; past "
                "mid-price direction is excluded from the reported dictionary"
            ),
            "predicate_schema": predicate_schema,
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
                "strict_future_offset_nanoseconds": 1 if continuous else None,
                "impact_horizon_seconds": (
                    int(continuous_impact_seconds) if continuous else None
                ),
                "kernel_edges_nanoseconds": (
                    continuous_edges.tolist() if continuous else None
                ),
            },
            "partition": {
                "method": "ordered trading-day split",
                "fractions": list(map(float, partition_fractions)),
                "counts": [int(np.sum(partition == code)) for code in range(3)],
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
