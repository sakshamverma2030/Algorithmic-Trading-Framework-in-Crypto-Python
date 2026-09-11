"""
live_trader.py - Asynchronous real-time execution engine (PAPER_TRADING / LIVE_TRADING).

Architecture (one asyncio event loop, cooperative tasks)
-------------------------------------------------------

    MarketDataFeed --- BarEvent (closed candles; lossless FIFO queue) ------> strategy task --+
    (ccxt.pro WS |                                                                            |
     REST poll   |                                                                            +--> ExecutionHandler
     replay)     --- OrderBookEvent / TickEvent (latest-value mailbox) -----> risk monitor ---+    (Paper | Live)
                                                                                                         |
                     heartbeat / watchdog task (event-loop lag, stale feed)          TradeJournal (SQLite)

How the engine avoids event-loop starvation
-------------------------------------------
1. Closed bars go through a FIFO ``asyncio.Queue`` (never dropped). Order-book and tick
   updates go through a single-slot *mailbox* where the newest value overwrites the
   old one. A burst of 1,000 book updates per second therefore costs O(1) memory, and
   consumers always act on the freshest state instead of working through a backlog.
2. CPU-bound work (recomputing the Ichimoku frame with pandas) runs in a worker
   thread via ``asyncio.to_thread``, so WebSocket reads keep being serviced.
3. Blocking disk I/O (SQLite journal) also runs through ``asyncio.to_thread``.
4. Every wait has a timeout, and a heartbeat task measures loop lag
   (actual sleep - requested sleep). A persistent lag means something is blocking.
5. Stream coroutines are supervised: on error they reconnect with exponential
   back-off and jitter instead of crashing the engine.

Research/production parity: the engine reuses the selected ``IchimokuStrategy`` or
``MomentumStrategy`` and the ``risk`` module (sizing, stops, circuit breaker) exactly
as the backtester does.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import sqlite3
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd

from config import AppConfig, ConfigError, FeedType, StrategyKind, TradingMode, timeframe_to_minutes
from data_loader import OHLCV_COLUMNS, ohlcv_frame
from momentum import MomentumStrategy
from risk import (
    LONG,
    SHORT,
    CircuitBreaker,
    CostModel,
    ExitReason,
    StopManager,
    build_position_sizer,
)
from strategy import IchimokuStrategy, SignalSnapshot

logger = logging.getLogger(__name__)
T = TypeVar("T")


class LiveTradingError(RuntimeError):
    """Unrecoverable engine error (configuration, connectivity, exchange rejection)."""


class ExecutionError(LiveTradingError):
    """An order could not be executed."""


# ======================================================================================
# Events & channels
# ======================================================================================
@dataclass(frozen=True, slots=True)
class BarEvent:
    timestamp: pd.Timestamp  # candle open time (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class OrderBookEvent:
    timestamp: pd.Timestamp
    bids: tuple[tuple[float, float], ...]  # (price, amount), best first
    asks: tuple[tuple[float, float], ...]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else math.nan

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else math.nan

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread_bps(self) -> float:
        return (self.best_ask - self.best_bid) / self.mid * 1e4


@dataclass(frozen=True, slots=True)
class TickEvent:
    timestamp: pd.Timestamp
    price: float
    amount: float
    side: str


class Mailbox(Generic[T]):
    """Single-slot, latest-value-wins channel (a "conflating" queue).

    ``put`` never blocks and never grows memory. ``get`` waits until a value newer than
    the last one read arrives. ``dropped`` counts updates that were overwritten before
    a consumer saw them, which is useful to monitor consumer health.
    """

    def __init__(self) -> None:
        self._value: T | None = None
        self._event = asyncio.Event()
        self._consumed = asyncio.Event()
        self._consumed.set()
        self.updates = 0
        self.dropped = 0

    def put(self, value: T) -> None:
        if self._event.is_set():
            self.dropped += 1
        self._value = value
        self.updates += 1
        self._consumed.clear()
        self._event.set()

    def peek(self) -> T | None:
        return self._value

    async def get(self, timeout: float | None = None) -> T | None:
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        self._event.clear()
        self._consumed.set()
        return self._value

    async def wait_consumed(self, timeout: float) -> bool:
        """Back-pressure for replays only: wait until a consumer has read the latest value.
        Live feeds never call this, since a WebSocket reader must not block on its consumers."""
        return await _wait_or_timeout(self._consumed, timeout)


class EventBus:
    """Channels between the feed and the engine's consumer tasks."""

    def __init__(self, bar_queue_size: int = 1_000) -> None:
        self.bars: asyncio.Queue[BarEvent] = asyncio.Queue(maxsize=bar_queue_size)
        self.order_book: Mailbox[OrderBookEvent] = Mailbox()
        self.ticks: Mailbox[TickEvent] = Mailbox()
        self.last_event_monotonic = time.monotonic()

    def _touch(self) -> None:
        self.last_event_monotonic = time.monotonic()

    async def publish_bar(self, bar: BarEvent) -> None:
        await self.bars.put(bar)
        self._touch()

    def publish_book(self, book: OrderBookEvent) -> None:
        self.order_book.put(book)
        self._touch()

    def publish_tick(self, tick: TickEvent) -> None:
        self.ticks.put(tick)
        self._touch()


async def _wait_or_timeout(event: asyncio.Event, timeout: float) -> bool:
    """Sleep up to ``timeout`` seconds but wake immediately when ``event`` is set."""
    try:
        await asyncio.wait_for(event.wait(), timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def supervise(name: str, factory: Callable[[], Awaitable[None]], stop: asyncio.Event,
                    base_delay: float = 1.0, max_delay: float = 60.0) -> None:
    """Run a streaming coroutine forever, reconnecting with exponential back-off + jitter."""
    attempt = 0
    while not stop.is_set():
        try:
            await factory()
            attempt = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any stream failure triggers a reconnect
            attempt += 1
            delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            logger.warning("%s stream error (%s: %s); reconnecting in %.1fs", name, type(exc).__name__, exc, delay,
                           extra={"event": "stream_error", "stream": name, "attempt": attempt})
            await _wait_or_timeout(stop, delay)


# ======================================================================================
# Market-data feeds
# ======================================================================================
class MarketDataFeed(ABC):
    """Source of closed bars and order-book / tick updates."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.symbol = cfg.data.symbol
        self.timeframe = cfg.data.timeframe
        self.tf_ms = timeframe_to_minutes(cfg.data.timeframe) * 60_000

    @abstractmethod
    async def warmup(self, n_bars: int) -> pd.DataFrame:
        """Historical closed bars used to initialise the indicators."""

    @abstractmethod
    async def run(self, bus: EventBus, stop: asyncio.Event) -> None:
        """Publish events until ``stop`` is set (a replay may return earlier)."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""


class _CCXTFeedBase(MarketDataFeed):
    """Shared logic for CCXT-based feeds: warm-up download and closed-bar detection."""

    exchange: Any

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg)
        self._last_bar_ts: int | None = None

    async def warmup(self, n_bars: int) -> pd.DataFrame:
        await self.exchange.load_markets()
        if self.symbol not in self.exchange.markets:
            raise LiveTradingError(f"{self.symbol} is not listed on {self.cfg.exchange.exchange_id}")
        now_ms = self.exchange.milliseconds()
        since = now_ms - (n_bars + 2) * self.tf_ms
        rows: list[list[float]] = []
        while since < now_ms:
            batch = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            since = int(batch[-1][0]) + self.tf_ms
            if len(batch) < 2:
                break
        df = ohlcv_frame(rows)
        df = df[~df.index.duplicated(keep="last")]
        closed = df.index + pd.Timedelta(milliseconds=self.tf_ms) <= pd.Timestamp.now(tz="UTC")
        df = df[closed].tail(n_bars)
        if not df.empty:
            self._last_bar_ts = int(df.index[-1].timestamp() * 1000)
        logger.info("Warm-up: %d closed %s bars loaded from %s", len(df), self.timeframe,
                    self.cfg.exchange.exchange_id)
        return df

    async def _emit_closed(self, candles: list[list[float]], bus: EventBus) -> None:
        """A candle is final once a candle with a later open time exists."""
        if not candles:
            return
        forming_ts = int(candles[-1][0])
        cached_closed = [c for c in candles if int(c[0]) < forming_ts]
        if self._last_bar_ts is not None:
            expected_next = self._last_bar_ts + self.tf_ms
            earliest_cached = int(cached_closed[0][0]) if cached_closed else forming_ts
            if earliest_cached > expected_next:
                # The stream cache does not cover bars we missed (e.g. after a reconnect): backfill via REST.
                missed = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=expected_next, limit=1000)
                cached_closed = [c for c in missed if int(c[0]) < forming_ts] + cached_closed
        for c in sorted({int(c[0]): c for c in cached_closed}.values(), key=lambda c: c[0]):
            ts = int(c[0])
            if self._last_bar_ts is not None and ts <= self._last_bar_ts:
                continue
            self._last_bar_ts = ts
            await bus.publish_bar(BarEvent(pd.Timestamp(ts, unit="ms", tz="UTC"), *map(float, c[1:6])))

    def _book_event(self, ob: dict[str, Any]) -> OrderBookEvent:
        depth = self.cfg.live.order_book_depth
        ts = ob.get("timestamp") or self.exchange.milliseconds()
        return OrderBookEvent(
            timestamp=pd.Timestamp(ts, unit="ms", tz="UTC"),
            bids=tuple((float(p), float(a)) for p, a, *_ in ob.get("bids", [])[:depth]),
            asks=tuple((float(p), float(a)) for p, a, *_ in ob.get("asks", [])[:depth]),
        )

    async def close(self) -> None:
        try:
            await self.exchange.close()
        except Exception as exc:  # noqa: BLE001 - best effort during shutdown
            logger.debug("feed close: %s", exc)


class CCXTProFeed(_CCXTFeedBase):
    """WebSocket streams through ccxt.pro (bundled with ccxt >= 4): klines, depth, trades."""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg)
        try:
            import ccxt.pro as ccxtpro
        except ImportError as exc:
            raise LiveTradingError("ccxt.pro is unavailable; install `ccxt>=4` or use the REST feed") from exc
        if not hasattr(ccxtpro, cfg.exchange.exchange_id):
            raise LiveTradingError(f"ccxt.pro has no WebSocket support for {cfg.exchange.exchange_id}")
        self.exchange = getattr(ccxtpro, cfg.exchange.exchange_id)(cfg.exchange.ccxt_options())

    async def run(self, bus: EventBus, stop: asyncio.Event) -> None:
        await asyncio.gather(
            supervise("ws-ohlcv", lambda: self._watch_ohlcv(bus, stop), stop),
            supervise("ws-orderbook", lambda: self._watch_book(bus, stop), stop),
            supervise("ws-trades", lambda: self._watch_trades(bus, stop), stop),
        )

    async def _watch_ohlcv(self, bus: EventBus, stop: asyncio.Event) -> None:
        while not stop.is_set():
            candles = await self.exchange.watch_ohlcv(self.symbol, self.timeframe)
            await self._emit_closed(candles, bus)

    async def _watch_book(self, bus: EventBus, stop: asyncio.Event) -> None:
        while not stop.is_set():
            ob = await self.exchange.watch_order_book(self.symbol, limit=self.cfg.live.order_book_depth)
            bus.publish_book(self._book_event(ob))

    async def _watch_trades(self, bus: EventBus, stop: asyncio.Event) -> None:
        while not stop.is_set():
            trades = await self.exchange.watch_trades(self.symbol)
            if trades:
                t = trades[-1]
                bus.publish_tick(TickEvent(pd.Timestamp(t["timestamp"], unit="ms", tz="UTC"), float(t["price"]),
                                           float(t["amount"]), str(t.get("side") or "")))


class RestPollingFeed(_CCXTFeedBase):
    """REST polling fallback (ccxt.async_support). Latency is bounded by the poll interval."""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg)
        try:
            import ccxt.async_support as ccxt_async
        except ImportError as exc:
            raise LiveTradingError("ccxt is not installed (`pip install ccxt`)") from exc
        self.exchange = getattr(ccxt_async, cfg.exchange.exchange_id)(cfg.exchange.ccxt_options())

    async def run(self, bus: EventBus, stop: asyncio.Event) -> None:
        await asyncio.gather(
            supervise("rest-ohlcv", lambda: self._poll_ohlcv(bus, stop), stop),
            supervise("rest-orderbook", lambda: self._poll_book(bus, stop), stop),
        )

    async def _poll_ohlcv(self, bus: EventBus, stop: asyncio.Event) -> None:
        while not stop.is_set():
            candles = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=5)
            await self._emit_closed(candles, bus)
            await _wait_or_timeout(stop, self.cfg.live.poll_interval_sec)

    async def _poll_book(self, bus: EventBus, stop: asyncio.Event) -> None:
        while not stop.is_set():
            ob = await self.exchange.fetch_order_book(self.symbol, limit=self.cfg.live.order_book_depth)
            bus.publish_book(self._book_event(ob))
            await _wait_or_timeout(stop, max(1.0, self.cfg.live.poll_interval_sec / 2))


class ReplayFeed(MarketDataFeed):
    """Replays historical bars through the *real* engine: an offline paper-trading demo
    and integration test that needs no network.

    Each bar is expanded into a plausible intrabar path (O -> L -> H -> C for an up bar,
    O -> H -> L -> C for a down bar), linearly interpolated into ``steps_per_leg`` ticks
    per leg, and published as synthetic order-book updates. Protective stops therefore
    trigger from "ticks" close to their level, as they would live. Only then is the closed
    bar published. Back-pressure (each tick is read by the risk monitor, each bar is
    processed via ``bars.join()``) makes a replay deterministic: it never outruns the engine.
    """

    def __init__(self, cfg: AppConfig, data: pd.DataFrame, warmup_count: int, speed: float = 0.0,
                 levels: int = 5, steps_per_leg: int = 4) -> None:
        super().__init__(cfg)
        if len(data) <= warmup_count:
            raise LiveTradingError(f"Replay needs more than {warmup_count} bars (got {len(data)})")
        self.data = data[OHLCV_COLUMNS]
        self.warmup_count = warmup_count
        self.speed = speed
        self.levels = levels
        self.steps_per_leg = steps_per_leg
        self.half_spread = cfg.costs.half_spread_bps / 1e4

    async def warmup(self, n_bars: int) -> pd.DataFrame:
        return self.data.iloc[: self.warmup_count]

    def _book(self, ts: pd.Timestamp, price: float, depth_unit: float) -> OrderBookEvent:
        half = price * self.half_spread
        tick = price * 0.5e-4
        bids = tuple((price - half - k * tick, depth_unit * (k + 1)) for k in range(self.levels))
        asks = tuple((price + half + k * tick, depth_unit * (k + 1)) for k in range(self.levels))
        return OrderBookEvent(ts, bids, asks)

    async def run(self, bus: EventBus, stop: asyncio.Event) -> None:
        replay = self.data.iloc[self.warmup_count:]
        logger.info("Replay feed: streaming %d bars (%s -> %s)", len(replay), replay.index[0], replay.index[-1])
        for ts, o, h, l, c, v in replay.itertuples(name=None):
            if stop.is_set():
                break
            path = (o, l, h, c) if c >= o else (o, h, l, c)
            ticks = [o] + [a + (b - a) * k / self.steps_per_leg
                           for a, b in zip(path, path[1:]) for k in range(1, self.steps_per_leg + 1)]
            depth_unit = max(v / 200.0, 1e-6)  # displayed depth scales with the bar's volume
            for price in ticks:
                bus.publish_book(self._book(ts, price, depth_unit))
                bus.publish_tick(TickEvent(ts, price, 0.0, ""))
                await bus.order_book.wait_consumed(timeout=0.05)  # let the risk monitor see every tick
            await bus.publish_bar(BarEvent(ts, o, h, l, c, v))
            await bus.bars.join()
            if self.speed:
                await asyncio.sleep(self.speed)
        logger.info("Replay feed finished")


# ======================================================================================
# Execution
# ======================================================================================
@dataclass(frozen=True)
class Fill:
    order_id: str
    client_order_id: str
    timestamp: pd.Timestamp
    side: str               # "buy" | "sell"
    quantity: float
    price: float
    fee: float
    reference_price: float
    mode: str

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def slippage_bps(self) -> float:
        sign = 1.0 if self.side == "buy" else -1.0
        return sign * (self.price / self.reference_price - 1.0) * 1e4 if self.reference_price else math.nan


def _client_order_id() -> str:
    """Unique client order id: lets us reconcile an order whose response was lost."""
    return f"ichi-{uuid.uuid4().hex[:20]}"


class ExecutionHandler(ABC):
    mode: TradingMode

    async def initialize(self) -> None:  # noqa: B027 - optional hook
        """Connect, load markets, check balances."""

    @abstractmethod
    async def market_order(self, side: str, quantity: float, reference_price: float) -> Fill:
        """Execute a market order and return the fill."""

    @abstractmethod
    async def equity(self, mark_price: float) -> float:
        """Account value in quote currency, marked at ``mark_price``."""

    @abstractmethod
    async def free_balance(self) -> tuple[float, float]:
        """(free base, free quote) that new orders may use."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


class PaperExecutionHandler(ExecutionHandler):
    """Simulated execution against the live (or replayed) order book.

    A market order walks the book: the fill price is the volume-weighted average
    over the levels it consumes,  P_fill = sum(p_i * q_i) / sum(q_i).  That is real
    depth-dependent slippage rather than a fixed guess. A latency slippage component
    is added on top, and any size beyond the visible depth fills 10 bps past the last level.
    """

    mode = TradingMode.PAPER_TRADING

    def __init__(self, cfg: AppConfig, bus: EventBus) -> None:
        self.cfg = cfg
        self.bus = bus
        self.cost_model = CostModel(cfg.costs)
        self.cash = cfg.risk.initial_capital
        self.base = 0.0  # signed base-asset position (negative = short)

    @staticmethod
    def walk_book(levels: tuple[tuple[float, float], ...], quantity: float) -> float | None:
        if not levels or quantity <= 0:
            return None
        remaining, cost = quantity, 0.0
        for price, amount in levels:
            take = min(remaining, amount)
            cost += take * price
            remaining -= take
            if remaining <= 1e-12:
                return cost / quantity
        cost += remaining * levels[-1][0] * (1.0 + (1e-3 if levels[0][0] <= levels[-1][0] else -1e-3))
        return cost / quantity

    async def market_order(self, side: str, quantity: float, reference_price: float) -> Fill:
        if quantity <= 0:
            raise ExecutionError(f"invalid order quantity {quantity}")
        sign = 1 if side == "buy" else -1
        book = self.bus.order_book.peek()
        vwap = self.walk_book(book.asks if sign > 0 else book.bids, quantity) if book else None
        if vwap is None:
            price = self.cost_model.fill_price(reference_price, sign, quantity)
        else:
            price = vwap * (1.0 + sign * self.cfg.costs.slippage_bps / 1e4)
        fee = self.cost_model.fee(quantity * price)
        self.cash -= sign * quantity * price + fee
        self.base += sign * quantity
        return Fill(f"paper-{uuid.uuid4().hex[:12]}", _client_order_id(), pd.Timestamp.now(tz="UTC"), side,
                    quantity, price, fee, reference_price, self.mode.value)

    async def equity(self, mark_price: float) -> float:
        return self.cash + self.base * mark_price

    async def free_balance(self) -> tuple[float, float]:
        return max(self.base, 0.0), self.cash


class LiveExecutionHandler(ExecutionHandler):
    """Real orders through CCXT's unified async API.

    Safety features: testnet/sandbox by default, exchange precision and minimum
    checks, a hard per-order notional cap, client order ids for idempotent retry,
    and reconciliation when an order response is lost to a network error.
    """

    mode = TradingMode.LIVE_TRADING

    def __init__(self, cfg: AppConfig) -> None:
        if not cfg.exchange.has_credentials:
            raise ConfigError("LIVE_TRADING requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET")
        try:
            import ccxt
            import ccxt.async_support as ccxt_async
        except ImportError as exc:
            raise LiveTradingError("ccxt is not installed (`pip install ccxt`)") from exc
        self._ccxt = ccxt
        self.cfg = cfg
        self.exchange = getattr(ccxt_async, cfg.exchange.exchange_id)(cfg.exchange.ccxt_options(authenticated=True))
        if cfg.exchange.use_testnet:
            self.exchange.set_sandbox_mode(True)
        self.market: dict[str, Any] = {}
        self.cost_model = CostModel(cfg.costs)

    async def initialize(self) -> None:
        await self.exchange.load_markets()
        symbol = self.cfg.data.symbol
        if symbol not in self.exchange.markets:
            raise LiveTradingError(f"{symbol} not available on {self.cfg.exchange.exchange_id}")
        self.market = self.exchange.market(symbol)
        balance = await self.exchange.fetch_balance()
        quote = self.market["quote"]
        logger.warning("LIVE TRADING on %s%s | free %s: %.2f", self.cfg.exchange.exchange_id,
                       " (TESTNET)" if self.cfg.exchange.use_testnet else " (MAINNET - REAL FUNDS)",
                       quote, float(balance.get(quote, {}).get("free") or 0.0))

    def _validate(self, quantity: float, reference_price: float) -> float:
        symbol = self.cfg.data.symbol
        amount = float(self.exchange.amount_to_precision(symbol, quantity))
        limits = self.market.get("limits", {})
        min_amount = (limits.get("amount") or {}).get("min") or 0.0
        min_cost = (limits.get("cost") or {}).get("min") or self.cfg.risk.min_order_notional
        notional = amount * reference_price
        if amount <= 0 or amount < min_amount or notional < min_cost:
            raise ExecutionError(f"order too small: amount={amount} notional={notional:.2f} "
                                 f"(min amount {min_amount}, min cost {min_cost})")
        if notional > self.cfg.live.max_order_notional:
            raise ExecutionError(f"order notional {notional:.2f} exceeds the hard cap "
                                 f"{self.cfg.live.max_order_notional:.2f}")
        return amount

    async def market_order(self, side: str, quantity: float, reference_price: float) -> Fill:
        ccxt = self._ccxt
        symbol = self.cfg.data.symbol
        amount = self._validate(quantity, reference_price)
        client_id = _client_order_id()
        params = {"clientOrderId": client_id}
        order: dict[str, Any] | None = None
        for attempt in range(1, 4):
            try:
                order = await self.exchange.create_order(symbol, "market", side, amount, None, params)
                break
            except ccxt.NetworkError as exc:
                # The request may have reached the exchange: look the order up before retrying.
                logger.warning("create_order network error (%s), reconciling %s", exc, client_id)
                order = await self._find_order(client_id)
                if order is not None:
                    break
                await asyncio.sleep(min(2 ** attempt, 10))
            except ccxt.InsufficientFunds as exc:
                raise ExecutionError(f"insufficient funds: {exc}") from exc
            except ccxt.InvalidOrder as exc:
                raise ExecutionError(f"order rejected: {exc}") from exc
            except ccxt.ExchangeError as exc:
                raise ExecutionError(f"exchange error: {exc}") from exc
        if order is None:
            raise ExecutionError(f"order {client_id} could not be placed after retries")

        if order.get("status") != "closed" or not order.get("average"):
            try:
                order = await self.exchange.fetch_order(order["id"], symbol)
            except Exception as exc:  # noqa: BLE001 - fall back to the create_order response
                logger.warning("fetch_order failed (%s); using the create_order response", exc)
        filled = float(order.get("filled") or amount)
        price = float(order.get("average") or order.get("price") or reference_price)
        quantity, fee = self._net_of_fees(order, side, filled, price)
        return Fill(str(order.get("id")), client_id, pd.Timestamp.now(tz="UTC"), side, quantity, price, fee,
                    reference_price, self.mode.value)

    def _net_of_fees(self, order: dict[str, Any], side: str, filled: float, price: float) -> tuple[float, float]:
        """Return (quantity actually received or delivered, fee in quote currency).

        Spot exchanges such as Binance charge the fee in the asset you *receive*: buying Q BTC
        credits Q - fee BTC. The position must record the net amount, or the exit would try to
        sell BTC the account does not hold. Fees paid in a third asset (e.g. BNB) are valued at
        the configured taker rate.
        """
        base, quote = self.market["base"], self.market["quote"]
        fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
        quantity, fee_quote, unpriced = filled, 0.0, not fees
        for f in fees:
            cost = float(f.get("cost") or 0.0)
            if f.get("currency") == quote:
                fee_quote += cost
            elif f.get("currency") == base:
                fee_quote += cost * price
                if side == "buy":
                    quantity -= cost
            elif cost:
                unpriced = True
        if unpriced:
            fee_quote += self.cost_model.fee(filled * price)
        return quantity, fee_quote

    async def free_balance(self) -> tuple[float, float]:
        try:
            balance = await self.exchange.fetch_balance()
        except Exception as exc:  # noqa: BLE001 - unknown balance: let the exchange be the judge
            logger.warning("fetch_balance failed (%s); order size not capped by balance", exc)
            return math.inf, math.inf
        free = balance.get("free") or {}
        return float(free.get(self.market["base"]) or 0.0), float(free.get(self.market["quote"]) or 0.0)

    async def _find_order(self, client_id: str) -> dict[str, Any] | None:
        try:
            return await self.exchange.fetch_order(None, self.cfg.data.symbol, {"clientOrderId": client_id})
        except Exception:  # noqa: BLE001 - not found / not supported
            return None

    async def equity(self, mark_price: float) -> float:
        balance = await self.exchange.fetch_balance()
        base, quote = self.market["base"], self.market["quote"]
        base_total = float((balance.get(base) or {}).get("total") or 0.0)
        quote_total = float((balance.get(quote) or {}).get("total") or 0.0)
        return quote_total + base_total * mark_price

    async def close(self) -> None:
        await self.exchange.close()


# ======================================================================================
# Journal
# ======================================================================================
class TradeJournal:
    """Append-only SQLite audit trail. Writes run in a worker thread (``asyncio.to_thread``)."""

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS fills (run_id TEXT, ts TEXT, order_id TEXT, client_order_id TEXT, side TEXT, "
        "quantity REAL, price REAL, fee REAL, reference_price REAL, slippage_bps REAL, mode TEXT, reason TEXT)",
        "CREATE TABLE IF NOT EXISTS trades (run_id TEXT, side INTEGER, entry_time TEXT, exit_time TEXT, "
        "entry_price REAL, exit_price REAL, quantity REAL, net_pnl REAL, return_pct REAL, r_multiple REAL, "
        "exit_reason TEXT)",
        "CREATE TABLE IF NOT EXISTS equity (run_id TEXT, ts TEXT, equity REAL, position INTEGER, price REAL)",
        "CREATE TABLE IF NOT EXISTS events (run_id TEXT, ts TEXT, kind TEXT, message TEXT)",
    )

    def __init__(self, db_path: Path, run_id: str) -> None:
        self.db_path = Path(db_path)
        self.run_id = run_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._equity_buffer: list[tuple[Any, ...]] = []
        # One connection shared by worker threads; the lock serialises access. WAL mode with
        # synchronous=NORMAL makes each commit an append, not a full fsync.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._conn:
            for statement in self._SCHEMA:
                self._conn.execute(statement)

    def _write_many(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(sql, rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    async def _write(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        try:
            await asyncio.to_thread(self._write_many, sql, rows)
        except Exception:  # noqa: BLE001 - journaling must never take the engine down
            logger.exception("journal write failed")

    async def fill(self, fill: Fill, reason: str) -> None:
        await self._write("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [(
            self.run_id, fill.timestamp.isoformat(), fill.order_id, fill.client_order_id, fill.side, fill.quantity,
            fill.price, fill.fee, fill.reference_price, fill.slippage_bps, fill.mode, reason)])

    async def trade(self, t: dict[str, Any]) -> None:
        await self._write("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", [(
            self.run_id, t["side"], str(t["entry_time"]), str(t["exit_time"]), t["entry_price"], t["exit_price"],
            t["quantity"], t["net_pnl"], t["return_pct"], t["r_multiple"], t["exit_reason"])])

    async def event(self, kind: str, message: str) -> None:
        await self._write("INSERT INTO events VALUES (?,?,?,?)",
                          [(self.run_id, pd.Timestamp.now(tz="UTC").isoformat(), kind, message)])

    async def equity(self, ts: pd.Timestamp, equity: float, position: int, price: float, flush_every: int = 50) -> None:
        self._equity_buffer.append((self.run_id, ts.isoformat(), equity, position, price))
        if len(self._equity_buffer) >= flush_every:
            await self.flush()

    async def flush(self) -> None:
        if self._equity_buffer:
            rows, self._equity_buffer = self._equity_buffer, []
            await self._write("INSERT INTO equity VALUES (?,?,?,?,?)", rows)


# ======================================================================================
# Engine
# ======================================================================================
@dataclass
class LivePosition:
    side: int
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    stop: float
    initial_stop: float
    take_profit: float | None
    entry_fee: float
    initial_risk: float
    extreme: float


@dataclass
class LiveSessionSummary:
    run_id: str
    mode: str
    feed: str
    bars_processed: int
    fills: int
    trades: list[dict[str, Any]] = field(default_factory=list)
    start_equity: float = math.nan
    final_equity: float = math.nan
    open_position: dict[str, Any] | None = None
    mailbox_dropped: int = 0
    max_loop_lag_ms: float = 0.0

    @property
    def total_return(self) -> float:
        return self.final_equity / self.start_equity - 1.0 if self.start_equity else math.nan

    def format(self) -> str:
        wins = sum(1 for t in self.trades if t["net_pnl"] > 0)
        lines = [
            "=" * 66,
            f" LIVE SESSION SUMMARY ({self.mode}, feed={self.feed}, run={self.run_id})",
            "=" * 66,
            f" Bars processed          {self.bars_processed:>12,}",
            f" Fills                   {self.fills:>12,}",
            f" Round-trip trades       {len(self.trades):>12,}   (winners {wins})",
            f" Start equity            {self.start_equity:>12,.2f}",
            f" Final equity            {self.final_equity:>12,.2f}   ({self.total_return:+.2%})",
            f" Book updates coalesced  {self.mailbox_dropped:>12,}",
            f" Max event-loop lag      {self.max_loop_lag_ms:>12,.1f} ms",
            f" Open position           {self.open_position or 'flat'}",
            "=" * 66,
        ]
        return "\n".join(lines)


class LiveTradingEngine:
    """Coordinates feed, strategy, risk monitor, execution and journal."""

    def __init__(self, cfg: AppConfig, feed: MarketDataFeed, bus: EventBus | None = None,
                 executor: ExecutionHandler | None = None) -> None:
        self.cfg = cfg
        self.feed = feed
        self.bus = bus or EventBus()
        self.executor = executor or build_execution_handler(cfg, self.bus)
        self.strategy = build_strategy(cfg)
        self.sizer = build_position_sizer(cfg.risk)
        self.stops = StopManager(cfg.risk)
        self.breaker = CircuitBreaker.from_config(cfg.risk, cfg.data.timeframe)
        self.run_id = f"{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.journal = TradeJournal(cfg.live.journal_db_path, self.run_id)

        self.position: LivePosition | None = None
        self.bars = pd.DataFrame(columns=OHLCV_COLUMNS, dtype=float)
        self.trades: list[dict[str, Any]] = []
        self.fills = 0
        self.bars_processed = 0
        self.last_price = math.nan
        # Market clock: exchange time of the latest event (book timestamp, or bar close time).
        # Trades are stamped with it, so a replay is directly comparable with the backtester.
        self.bar_delta = pd.Timedelta(minutes=timeframe_to_minutes(cfg.data.timeframe))
        self.market_time = pd.Timestamp.now(tz="UTC")
        self.max_loop_lag = 0.0
        self._stop = asyncio.Event()
        self._trade_lock = asyncio.Lock()  # serialises orders from the strategy and risk-monitor tasks
        self._bar_log_level = logging.DEBUG if isinstance(feed, ReplayFeed) else logging.INFO

    # ------------------------------------------------------------------ lifecycle
    def stop(self) -> None:
        self._stop.set()

    def _validate_mode(self) -> None:
        mode = self.cfg.live.mode
        if mode is TradingMode.LIVE_TRADING:
            if isinstance(self.feed, ReplayFeed):
                raise ConfigError("The replay feed cannot be combined with LIVE_TRADING")
            if self.strategy.allow_short and self.cfg.exchange.market_type == "spot":
                raise ConfigError("Short selling needs a derivatives/margin market (exchange.market_type)")
        if self.executor.mode is not mode:
            raise ConfigError(f"Executor mode {self.executor.mode.value} does not match configured mode {mode.value}")

    async def run(self) -> LiveSessionSummary:
        self._validate_mode()
        await self.executor.initialize()
        warm = await self.feed.warmup(warmup_bars_needed(self.cfg))
        if len(warm) < self.strategy.warmup_bars:
            logger.warning("Only %d warm-up bars (strategy needs %d); signals stay inactive until enough bars arrive",
                           len(warm), self.strategy.warmup_bars)
        self.bars = warm[OHLCV_COLUMNS].astype(float)
        self.last_price = float(self.bars["close"].iloc[-1]) if len(self.bars) else math.nan
        start_equity = await self.executor.equity(self.last_price if math.isfinite(self.last_price) else 0.0)
        logger.info("Engine %s started | mode=%s | %s | %s %s | equity %.2f", self.run_id, self.cfg.live.mode.value,
                    self.strategy.name, self.cfg.data.symbol, self.cfg.data.timeframe, start_equity)
        await self.journal.event("start", f"{self.cfg.live.mode.value} {self.strategy.name}")

        tasks = [
            asyncio.create_task(self.feed.run(self.bus, self._stop), name="feed"),
            asyncio.create_task(self._strategy_loop(), name="strategy"),
            asyncio.create_task(self._risk_monitor_loop(), name="risk-monitor"),
            asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
            asyncio.create_task(self._stop.wait(), name="stop-signal"),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    logger.critical("Task %s crashed: %r", task.get_name(), task.exception())
                elif task.get_name() == "feed":
                    logger.info("Feed completed; shutting down")
        finally:
            self._stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            summary = await self._shutdown(start_equity)
        return summary

    async def _shutdown(self, start_equity: float) -> LiveSessionSummary:
        if self.position is not None and self.cfg.live.flatten_on_exit:
            async with self._trade_lock:
                await self._exit_position(ExitReason.SHUTDOWN, self.last_price)
        final_equity = await self.executor.equity(self.last_price)
        await self.journal.flush()
        await self.journal.event("stop", f"final equity {final_equity:.2f}")
        self.journal.close()
        await self.feed.close()
        await self.executor.close()
        summary = LiveSessionSummary(
            run_id=self.run_id, mode=self.cfg.live.mode.value, feed=type(self.feed).__name__,
            bars_processed=self.bars_processed, fills=self.fills, trades=list(self.trades),
            start_equity=start_equity, final_equity=final_equity,
            open_position=asdict(self.position) if self.position else None,
            mailbox_dropped=self.bus.order_book.dropped, max_loop_lag_ms=self.max_loop_lag * 1000,
        )
        logger.info("Engine stopped: %d bars, %d trades, equity %.2f -> %.2f", summary.bars_processed,
                    len(summary.trades), start_equity, final_equity)
        return summary

    # ------------------------------------------------------------------ tasks
    async def _strategy_loop(self) -> None:
        while not self._stop.is_set():
            bar = await self.bus.bars.get()
            try:
                await self._on_bar(bar)
            except Exception:  # noqa: BLE001 - one bad bar must not kill the engine
                logger.exception("Error while processing bar %s", bar.timestamp)
            finally:
                self.bus.bars.task_done()

    async def _risk_monitor_loop(self) -> None:
        """React to every order-book update: stop-loss / take-profit between bar closes."""
        while not self._stop.is_set():
            book = await self.bus.order_book.get(timeout=1.0)
            if book is None or not book.bids or not book.asks:
                continue
            self.last_price = book.mid
            self.market_time = book.timestamp
            pos = self.position
            if pos is None:
                continue
            # A long exits by selling at the bid; a short exits by buying at the ask.
            exit_price = book.best_bid if pos.side == LONG else book.best_ask
            reason = self._touched(pos, exit_price)
            if reason is not None:
                async with self._trade_lock:
                    if self.position is pos:  # re-check: the strategy task may have closed it meanwhile
                        await self._exit_position(reason, exit_price)

    async def _heartbeat_loop(self) -> None:
        loop = asyncio.get_running_loop()
        interval = self.cfg.live.heartbeat_sec
        while not self._stop.is_set():
            started = loop.time()
            if await _wait_or_timeout(self._stop, interval):
                break
            lag = loop.time() - started - interval  # > 0: the loop was blocked by someone
            self.max_loop_lag = max(self.max_loop_lag, lag)
            stale = time.monotonic() - self.bus.last_event_monotonic
            equity = await self.executor.equity(self.last_price)
            logger.info("heartbeat | price=%.2f equity=%.2f position=%s loop_lag=%.1fms feed_age=%.0fs queue=%d",
                        self.last_price, equity, self.position.side if self.position else 0, lag * 1000, stale,
                        self.bus.bars.qsize(), extra={"event": "heartbeat", "equity": equity, "loop_lag_ms": lag * 1000})
            if lag > 0.5:
                logger.warning("Event-loop lag %.0f ms: something is blocking the loop", lag * 1000)
            if stale > self.cfg.live.stale_feed_timeout_sec:
                logger.warning("No market data for %.0fs: feed may be stalled", stale)

    # ------------------------------------------------------------------ bar handling
    async def _on_bar(self, bar: BarEvent) -> None:
        row = pd.DataFrame([[bar.open, bar.high, bar.low, bar.close, bar.volume]], columns=OHLCV_COLUMNS,
                           index=pd.DatetimeIndex([bar.timestamp], name="timestamp"))
        bars = pd.concat([self.bars, row]) if len(self.bars) else row
        keep = max(self.cfg.live.bar_buffer, self.strategy.history_bars)
        self.bars = bars[~bars.index.duplicated(keep="last")].iloc[-keep:]
        self.last_price = bar.close
        self.market_time = bar.timestamp + self.bar_delta  # the bar's close time
        self.bars_processed += 1

        # 1) Safety net: a level crossed inside the bar that the tick monitor did not catch
        #    (e.g. the book stream was down during a spike). Exit at the current price.
        async with self._trade_lock:
            pos = self.position
            if pos is not None:
                hit = StopManager.check_intrabar_exit(pos.side, pos.stop, pos.take_profit, bar.open, bar.high, bar.low)
                if hit is not None:
                    await self._exit_position(self._label(pos, hit[0]), bar.close)

        # 2) Signals: CPU-bound pandas work runs in a worker thread, on just enough history
        #    to reproduce a full-history computation (see IchimokuStrategy.history_bars).
        window = self.bars.iloc[-self.strategy.history_bars:].copy()
        snap: SignalSnapshot = await asyncio.to_thread(self.strategy.latest_signal, window)

        # 3) Portfolio circuit breaker on marked-to-market equity.
        equity = await self.executor.equity(bar.close)
        event = self.breaker.update(bar.timestamp, equity)
        halted = self.breaker.is_halted(bar.timestamp)

        async with self._trade_lock:
            if event is not None:
                await self.journal.event("circuit_breaker", f"{event.reason} dd={event.drawdown:.4f}")
                if self.position is not None:
                    await self._exit_position(ExitReason.CIRCUIT_BREAKER, bar.close)
            # 4) Trailing stop.
            pos = self.position
            if pos is not None:
                pos.extreme = max(pos.extreme, bar.high) if pos.side == LONG else min(pos.extreme, bar.low)
                pos.stop = self.stops.trail(pos.side, pos.stop, pos.extreme, snap.atr, snap.kijun)
            # 5) Signal-driven decisions: the same state machine as the event-driven backtester.
            if snap.is_ready:
                await self._act(snap, halted)

        await self.journal.equity(bar.timestamp, equity, self.position.side if self.position else 0, bar.close)
        logger.log(self._bar_log_level, "bar | %s | pos=%s equity=%.2f", snap.describe(),
                   self.position.side if self.position else 0, equity)

    async def _act(self, snap: SignalSnapshot, halted: bool) -> None:
        allow_short = self.strategy.allow_short
        pos = self.position
        if pos is None:
            if halted:
                return
            if snap.long_entry:
                await self._enter_position(LONG, snap)
            elif allow_short and snap.short_entry:
                await self._enter_position(SHORT, snap)
        elif pos.side == LONG and snap.exit_long:
            reverse = allow_short and snap.short_entry and not halted
            await self._exit_position(ExitReason.REVERSAL if reverse else ExitReason.SIGNAL, snap.close)
            if reverse and self.position is None:
                await self._enter_position(SHORT, snap)
        elif pos.side == SHORT and snap.exit_short:
            reverse = snap.long_entry and not halted
            await self._exit_position(ExitReason.REVERSAL if reverse else ExitReason.SIGNAL, snap.close)
            if reverse and self.position is None:
                await self._enter_position(LONG, snap)

    @staticmethod
    def _touched(pos: LivePosition, price: float) -> ExitReason | None:
        if pos.side == LONG:
            if price <= pos.stop:
                return LiveTradingEngine._label(pos, ExitReason.STOP_LOSS)
            if pos.take_profit is not None and price >= pos.take_profit:
                return ExitReason.TAKE_PROFIT
        else:
            if price >= pos.stop:
                return LiveTradingEngine._label(pos, ExitReason.STOP_LOSS)
            if pos.take_profit is not None and price <= pos.take_profit:
                return ExitReason.TAKE_PROFIT
        return None

    @staticmethod
    def _label(pos: LivePosition, reason: ExitReason) -> ExitReason:
        if reason is ExitReason.STOP_LOSS and pos.stop != pos.initial_stop:
            return ExitReason.TRAILING_STOP
        return reason

    def _reference_price(self, side: int, fallback: float) -> float:
        book = self.bus.order_book.peek()
        if book is not None and book.bids and book.asks:
            return book.best_ask if side > 0 else book.best_bid
        return fallback

    # ------------------------------------------------------------------ orders (call with _trade_lock held)
    async def _enter_position(self, side: int, snap: SignalSnapshot) -> None:
        ref = self._reference_price(side, snap.close)
        levels = self.stops.initial_levels(side, ref, snap.atr, snap.cloud_top, snap.cloud_bottom)
        if levels is None:
            return
        equity = await self.executor.equity(ref)
        qty = self.sizer.size(equity, ref, levels.stop_loss)
        if side == LONG:  # spot: a buy can only spend the free quote balance
            _, free_quote = await self.executor.free_balance()
            qty = min(qty, free_quote / (ref * (1.0 + self.cfg.costs.taker_fee)))
        if self.cfg.live.mode is TradingMode.LIVE_TRADING:
            qty = min(qty, 0.999 * self.cfg.live.max_order_notional / ref)
        if qty * ref < self.cfg.risk.min_order_notional:
            logger.info("Entry skipped: size %.6f below the minimum order value", qty)
            return
        try:
            fill = await self.executor.market_order("buy" if side == LONG else "sell", qty, ref)
        except ExecutionError as exc:
            logger.error("Entry failed: %s", exc, extra={"event": "order_error"})
            await self.journal.event("order_error", str(exc))
            return
        levels = self.stops.initial_levels(side, fill.price, snap.atr, snap.cloud_top, snap.cloud_bottom)
        assert levels is not None
        self.position = LivePosition(side, fill.quantity, fill.price, self.market_time, levels.stop_loss,
                                     levels.stop_loss, levels.take_profit, fill.fee,
                                     fill.quantity * abs(fill.price - levels.stop_loss), fill.price)
        self.fills += 1
        await self.journal.fill(fill, "entry")
        logger.info("ENTER %s %.6f @ %.2f | SL %.2f TP %s | slippage %.1f bps", "LONG" if side == LONG else "SHORT",
                    fill.quantity, fill.price, levels.stop_loss,
                    f"{levels.take_profit:.2f}" if levels.take_profit else "-", fill.slippage_bps,
                    extra={"event": "entry", "side": side, "price": fill.price, "qty": fill.quantity})

    async def _exit_position(self, reason: ExitReason, ref_price: float) -> None:
        pos = self.position
        if pos is None:
            return
        side = "sell" if pos.side == LONG else "buy"
        ref = self._reference_price(-pos.side, ref_price)
        qty = pos.quantity
        if pos.side == LONG:  # never try to sell more than the account holds (fees, dust)
            free_base, _ = await self.executor.free_balance()
            qty = min(qty, free_base)
        try:
            fill = await self.executor.market_order(side, qty, ref)
        except ExecutionError as exc:
            logger.critical("EXIT FAILED (%s): %s - position still open, will retry", reason.value, exc,
                            extra={"event": "exit_error"})
            await self.journal.event("exit_error", str(exc))
            return
        gross = pos.side * (fill.price - pos.entry_price) * pos.quantity
        net = gross - pos.entry_fee - fill.fee
        trade = {
            "side": pos.side, "entry_time": pos.entry_time, "exit_time": self.market_time,
            "entry_price": pos.entry_price, "exit_price": fill.price, "quantity": pos.quantity,
            "net_pnl": net, "return_pct": net / (pos.entry_price * pos.quantity),
            "r_multiple": net / pos.initial_risk if pos.initial_risk > 0 else math.nan,
            "exit_reason": reason.value,
        }
        self.trades.append(trade)
        self.sizer.record_trade(trade["r_multiple"])
        self.position = None
        self.fills += 1
        await self.journal.fill(fill, reason.value)
        await self.journal.trade(trade)
        logger.info("EXIT %s @ %.2f (%s) | net PnL %.2f (%.2f%%)", "LONG" if pos.side == LONG else "SHORT",
                    fill.price, reason.value, net, 100 * trade["return_pct"],
                    extra={"event": "exit", "reason": reason.value, "net_pnl": net})


# ======================================================================================
# Factories & runner
# ======================================================================================
def build_execution_handler(cfg: AppConfig, bus: EventBus) -> ExecutionHandler:
    if cfg.live.mode is TradingMode.LIVE_TRADING:
        return LiveExecutionHandler(cfg)
    return PaperExecutionHandler(cfg, bus)


def build_strategy(cfg: AppConfig) -> IchimokuStrategy | MomentumStrategy:
    """The signal generator selected by ``cfg.live.live_strategy`` (shared with the backtester)."""
    if cfg.live.live_strategy is StrategyKind.MOMENTUM:
        return MomentumStrategy(cfg.momentum)
    return IchimokuStrategy(cfg.strategy)


def warmup_bars_needed(cfg: AppConfig) -> int:
    """Closed bars loaded before the engine starts (enough for fully converged indicators)."""
    strategy = build_strategy(cfg)
    return max(cfg.live.warmup_bars, strategy.history_bars)


def build_feed(cfg: AppConfig, replay_data: pd.DataFrame | None = None) -> MarketDataFeed:
    feed_type = cfg.live.feed
    if feed_type is FeedType.REPLAY:
        if replay_data is None:
            raise ConfigError("The replay feed needs historical data")
        return ReplayFeed(cfg, replay_data, warmup_bars_needed(cfg), cfg.live.replay_speed)
    if feed_type is FeedType.CCXT_PRO:
        try:
            return CCXTProFeed(cfg)
        except LiveTradingError as exc:
            logger.warning("%s; falling back to REST polling", exc)
    return RestPollingFeed(cfg)


async def run_live_session(cfg: AppConfig, replay_data: pd.DataFrame | None = None) -> LiveSessionSummary:
    bus = EventBus()
    feed = build_feed(cfg, replay_data)
    engine = LiveTradingEngine(cfg, feed, bus=bus, executor=build_execution_handler(cfg, bus))
    return await engine.run()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run`` with the selector loop on Windows (needed by aiohttp/aiodns in CCXT)."""
    if sys.platform == "win32":
        if sys.version_info >= (3, 12):
            return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # pragma: no cover
    return asyncio.run(coro)
