"""
live_paper_trader.py - real-time WebSocket ingestion, paper trading and CCXT live templates.

Spec-facing facade around ``live_trader.py`` (the asynchronous trading engine) so the
pipeline structure (WebSockets + paper simulator + CCXT live order module)
maps one-to-one onto the codebase.
"""

from __future__ import annotations

from live_trader import (  # noqa: F401 - public API of the real-time engine
    BarEvent,
    CCXTProFeed,
    EventBus,
    ExecutionHandler,
    FeedType,
    Fill,
    LiveExecutionHandler,
    LiveTradingEngine,
    MarketDataFeed,
    OrderBookEvent,
    PaperExecutionHandler,
    ReplayFeed,
    RestPollingFeed,
    TradeJournal,
    TradingMode,
    build_execution_handler,
    build_feed,
    build_strategy,
    run_async,
    warmup_bars_needed,
)

__all__ = [
    "BarEvent", "CCXTProFeed", "EventBus", "ExecutionHandler", "Fill",
    "LiveExecutionHandler", "LiveTradingEngine", "MarketDataFeed", "OrderBookEvent",
    "PaperExecutionHandler", "ReplayFeed", "RestPollingFeed", "TradeJournal",
    "build_execution_handler", "build_feed", "build_strategy", "run_async",
    "warmup_bars_needed",
]