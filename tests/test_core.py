"""Unit and integration tests for the Ichimoku trading system.  Run:  python -m pytest"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from analytics import (
    PerformanceAnalyzer,
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    trade_statistics,
)
from backtester import EventDrivenBacktester, VectorizedBacktester
from config import (
    AppConfig,
    CostConfig,
    DataConfig,
    DataSource,
    FeedType,
    IchimokuParams,
    LiveConfig,
    RiskConfig,
    StrategyConfig,
    TrailingMethod,
)
from data_loader import OHLCVStore, SyntheticDataGenerator, load_csv, save_csv, to_epoch_ms, validate_and_clean
from indicators import IchimokuCloud, average_true_range
from live_trader import (
    EventBus,
    LiveExecutionHandler,
    LiveTradingEngine,
    PaperExecutionHandler,
    ReplayFeed,
    run_async,
    warmup_bars_needed,
)
from risk import (
    LONG,
    SHORT,
    CircuitBreaker,
    CostModel,
    ExitReason,
    FixedFractionalSizer,
    KellySizer,
    StopManager,
)
from strategy import IchimokuStrategy


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return SyntheticDataGenerator(seed=7).generate(1500, "1h", start=pd.Timestamp("2025-01-01", tz="UTC"))


# ----------------------------------------------------------------------------- indicators
def test_ichimoku_lines_match_textbook_definitions(ohlcv):
    ind = IchimokuCloud(IchimokuParams(9, 26, 52, 26)).compute(ohlcv)
    h, l, c = (ohlcv[k].to_numpy() for k in ("high", "low", "close"))
    mid = lambda end, n: (h[end - n + 1: end + 1].max() + l[end - n + 1: end + 1].min()) / 2  # noqa: E731

    i = 400
    assert ind["tenkan"].iloc[i] == pytest.approx(mid(i, 9))
    assert ind["kijun"].iloc[i] == pytest.approx(mid(i, 26))
    j = i - 26  # the Kumo visible at bar i was computed 26 bars earlier
    assert ind["senkou_a"].iloc[i] == pytest.approx((mid(j, 9) + mid(j, 26)) / 2)
    assert ind["senkou_b"].iloc[i] == pytest.approx(mid(j, 52))
    assert ind["chikou_span"].iloc[i] == c[i + 26]       # plot-only (future data)
    assert ind["chikou_reference"].iloc[i] == c[i - 26]  # causal comparison price
    first_b = 52 + 26 - 1
    assert ind["senkou_b"].iloc[:first_b].isna().all() and not math.isnan(ind["senkou_b"].iloc[first_b])


def test_atr_matches_wilder_reference(ohlcv):
    n = 14
    h, l, c = (ohlcv[k].to_numpy() for k in ("high", "low", "close"))
    prev = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    ref = np.full(len(c), np.nan)
    ref[n] = tr[1: n + 1].mean()
    for t in range(n + 1, len(c)):
        ref[t] = (ref[t - 1] * (n - 1) + tr[t]) / n
    atr = average_true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"], n).to_numpy()
    assert np.isnan(atr[:n]).all()
    np.testing.assert_allclose(atr[n:], ref[n:], rtol=1e-9)


@pytest.mark.parametrize("dynamic", [False, True])
def test_signals_never_use_future_data(ohlcv, dynamic):
    """Signals computed on a truncated history must equal those computed on the full history."""
    strategy = IchimokuStrategy(StrategyConfig(dynamic=dynamic, regime_lookback=200, allow_short=True,
                                               cross_lookback=3))
    full = strategy.generate_signals(ohlcv)
    cut = 900
    part = strategy.generate_signals(ohlcv.iloc[:cut])
    cols = [c for c in full.columns if c != "chikou_span"]  # the only plot-only, forward-looking column
    pd.testing.assert_frame_equal(part[cols], full[cols].iloc[:cut], check_dtype=False)


def test_position_state_machine():
    s = lambda *v: pd.Series(np.array(v, dtype=bool))  # noqa: E731
    pos = IchimokuStrategy.signals_to_position(
        long_entry=s(0, 1, 0, 0, 0, 0, 0, 0),
        short_entry=s(0, 0, 0, 0, 1, 0, 0, 0),
        exit_long=s(0, 0, 0, 1, 1, 0, 0, 0),
        exit_short=s(0, 0, 0, 0, 0, 0, 1, 0),
    )
    assert pos.tolist() == [0, 1, 1, 0, -1, -1, 0, 0]


# ----------------------------------------------------------------------------- risk
def test_cost_model_is_adverse_in_both_directions():
    cm = CostModel(CostConfig(half_spread_bps=1, slippage_bps=2, use_market_impact=False))
    assert cm.fill_price(100.0, LONG) == pytest.approx(100.03)
    assert cm.fill_price(100.0, SHORT) == pytest.approx(99.97)
    assert cm.fee(10_000.0) == pytest.approx(10.0)
    impact = CostModel(CostConfig(half_spread_bps=1, slippage_bps=2, impact_coefficient=0.5))
    expected = 100 * (1 + 3e-4 + 0.5 * 0.01 * math.sqrt(4 / 100))  # square-root law
    assert impact.fill_price(100.0, LONG, qty=4, bar_volume=100, bar_sigma=0.01) == pytest.approx(expected)


def test_position_sizers():
    ff = FixedFractionalSizer(0.01, max_position_pct=1.0)
    assert ff.size(10_000, 100, 98) == pytest.approx(50)     # 100 risk / 2 per unit
    assert ff.size(10_000, 100, 99.9) == pytest.approx(100)  # capped at 100 % notional

    kelly = KellySizer(0.01, multiplier=0.5, lookback=50, min_trades=10, max_fraction=0.5, max_position_pct=1)
    assert kelly.risk_fraction() == 0.01  # fallback before enough trades
    for r in [2.0] * 4 + [-1.0] * 6:     # W = 0.4, R = 2  ->  f* = 0.4 - 0.6 / 2 = 0.1
        kelly.record_trade(r)
    assert kelly.kelly_fraction() == pytest.approx(0.1)
    assert kelly.risk_fraction() == pytest.approx(0.05)  # half-Kelly


def test_intrabar_exit_rules():
    check = StopManager.check_intrabar_exit
    assert check(LONG, 95, 110, 100, 105, 94) == (ExitReason.STOP_LOSS, 95)
    assert check(LONG, 95, 110, 93, 96, 90) == (ExitReason.STOP_LOSS, 93)    # gap through the stop: fill at open
    assert check(LONG, 95, 110, 100, 111, 94) == (ExitReason.STOP_LOSS, 95)  # both hit: assume the worst
    assert check(LONG, 95, 110, 100, 111, 96) == (ExitReason.TAKE_PROFIT, 110)
    assert check(SHORT, 105, 90, 100, 104, 89) == (ExitReason.TAKE_PROFIT, 90)
    assert check(SHORT, 105, 90, 100, 106, 95) == (ExitReason.STOP_LOSS, 105)
    assert check(LONG, 95, 110, 100, 104, 96) is None


def test_trailing_stop_only_ratchets():
    sm = StopManager(RiskConfig(trailing_method=TrailingMethod.ATR, trailing_atr_multiplier=3))
    assert sm.trail(LONG, 90.0, extreme_price=110, atr=5, kijun=math.nan) == 95.0
    assert sm.trail(LONG, 95.0, extreme_price=100, atr=5, kijun=math.nan) == 95.0  # never loosens
    assert sm.trail(SHORT, 110.0, extreme_price=90, atr=5, kijun=math.nan) == 105.0


def test_circuit_breaker_trips_cools_down_and_resets():
    hour = pd.Timedelta(hours=1)
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    cb = CircuitBreaker(max_drawdown=0.2, cooldown=2 * hour, daily_loss_limit=0.0)
    assert cb.update(t0, 100) is None
    assert cb.update(t0 + hour, 120) is None
    event = cb.update(t0 + 2 * hour, 95)  # 95 / 120 - 1 = -20.8 %
    assert event is not None and event.reason == "max_drawdown"
    assert cb.is_halted(t0 + 3 * hour) and not cb.is_halted(t0 + 4 * hour)
    assert cb.update(t0 + 4 * hour, 95) is None  # released; high-water mark reset to 95
    assert cb.update(t0 + 5 * hour, 90) is None  # -5 % from the new peak

    daily = CircuitBreaker(max_drawdown=0.0, cooldown=hour, daily_loss_limit=0.05)
    daily.update(t0, 100)
    assert daily.update(t0 + hour, 94).reason == "daily_loss_limit"
    assert daily.is_halted(t0 + 5 * hour) and not daily.is_halted(t0 + pd.Timedelta(days=1))


# ----------------------------------------------------------------------------- backtesters
def test_event_driven_accounting_identities(ohlcv):
    cfg = AppConfig(strategy=StrategyConfig(allow_short=True, cross_lookback=3),
                    risk=RiskConfig(trailing_method=TrailingMethod.ATR))
    res = EventDrivenBacktester(cfg).run(ohlcv)
    ec, trades = res.equity_curve, res.trades
    assert len(trades) >= 5 and (trades["side"] == SHORT).any()
    # Mark-to-market identity on every bar: E = cash + side * qty * close
    np.testing.assert_allclose(ec["equity"], ec["cash"] + ec["position"] * ec["quantity"] * ec["close"], rtol=1e-12)
    # Flat at the end, so the round trips explain the whole PnL (fees and financing included)
    assert ec["position"].iloc[-1] == 0
    assert trades["net_pnl"].sum() == pytest.approx(res.final_equity - res.initial_capital, rel=1e-9)
    assert trades["fees"].sum() == pytest.approx(res.metadata["total_fees"], rel=1e-9)
    # One position at a time
    assert (trades["entry_time"].to_numpy()[1:] >= trades["exit_time"].to_numpy()[:-1]).all()
    assert set(trades["exit_reason"]) <= {r.value for r in ExitReason}


def test_vectorised_engine_trades_on_the_next_bar(ohlcv):
    cfg = AppConfig()
    strategy = IchimokuStrategy(cfg.strategy)
    sig = strategy.generate_signals(ohlcv)
    res = VectorizedBacktester(cfg, strategy).run(ohlcv, signals=sig)
    held = res.equity_curve["position"].to_numpy()
    assert (held[1:] == sig["position"].to_numpy()[:-1]).all()  # position(t) = signal(t-1)
    assert res.equity_curve["equity"].iloc[0] == cfg.risk.initial_capital


def test_performance_report_runs(ohlcv):
    res = EventDrivenBacktester(AppConfig()).run(ohlcv)
    report = PerformanceAnalyzer().analyze(res)
    assert report.strategy["max_drawdown"] <= 0
    assert "Sharpe ratio" in report.format()


# ----------------------------------------------------------------------------- analytics
def test_metric_formulas():
    idx = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    equity = pd.Series([100, 120, 90, 95, 130], index=idx, dtype=float)
    assert max_drawdown(equity) == pytest.approx(90 / 120 - 1)
    assert annualized_return(pd.Series([100.0, 110.0, 121.0]), ppy=1) == pytest.approx(0.10)

    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    assert sharpe_ratio(r, 252) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    downside = math.sqrt(0.02**2 / 4)
    assert sortino_ratio(r, 252) == pytest.approx(r.mean() / downside * math.sqrt(252))


def test_trade_statistics():
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    trades = pd.DataFrame({
        "side": [1, 1, -1, 1], "net_pnl": [100.0, -50.0, 30.0, -20.0], "return_pct": [0.02, -0.01, 0.006, -0.004],
        "r_multiple": [2.0, -1.0, 0.6, -0.4], "bars_held": [5, 3, 2, 1], "exit_reason": ["take_profit"] * 4,
        "entry_time": [t0] * 4, "exit_time": [t0 + pd.Timedelta(hours=h) for h in (5, 3, 2, 1)],
    })
    stats = trade_statistics(trades)
    assert stats["win_rate"] == 0.5
    assert stats["profit_factor"] == pytest.approx(130 / 70)
    assert stats["max_consecutive_losses"] == 1
    assert stats["avg_trade_duration"] == pd.Timedelta(hours=2.75)


# ----------------------------------------------------------------------------- data
def test_validation_cleans_and_reports(ohlcv):
    raw = ohlcv.iloc[:50].copy()
    raw = pd.concat([raw, raw.iloc[[10]]])           # duplicate timestamp
    raw.iloc[20, raw.columns.get_loc("high")] = 1.0  # impossible candle (high below the body)
    raw = raw.drop(raw.index[30:33])                 # 3-bar gap
    clean, report = validate_and_clean(raw, "1h")
    assert report.duplicates_removed == 1 and report.invalid_rows_removed == 1
    assert report.missing_bars == 4                  # the gap plus the removed invalid bar
    assert clean.index.is_monotonic_increasing and not clean.index.has_duplicates


def test_sqlite_store_roundtrip_is_idempotent(tmp_path, ohlcv):
    store = OHLCVStore(tmp_path / "market.sqlite")
    df = ohlcv.iloc[:300]
    store.upsert(df, "binance", "BTC/USDT", "1h")
    store.upsert(df.iloc[-50:], "binance", "BTC/USDT", "1h")  # overlapping re-download
    loaded = store.load("binance", "BTC/USDT", "1h")
    assert len(loaded) == len(df)
    np.testing.assert_allclose(loaded.to_numpy(), df.to_numpy())
    assert (to_epoch_ms(loaded.index) == to_epoch_ms(df.index)).all()
    assert store.bounds("binance", "BTC/USDT", "1h") == (df.index[0], df.index[-1])


def test_csv_roundtrip(tmp_path, ohlcv):
    path = save_csv(ohlcv.iloc[:100], tmp_path / "btc.csv")
    back = load_csv(path)
    np.testing.assert_allclose(back.to_numpy(), ohlcv.iloc[:100].to_numpy())


# ----------------------------------------------------------------------------- live engine
def test_paper_fill_walks_the_order_book():
    asks = ((100.0, 1.0), (101.0, 2.0))
    assert PaperExecutionHandler.walk_book(asks, 0.5) == 100.0
    assert PaperExecutionHandler.walk_book(asks, 2.0) == pytest.approx(100.5)
    assert PaperExecutionHandler.walk_book(asks, 4.0) == pytest.approx((100 + 202 + 101 * 1.001) / 4)


def test_live_fills_record_the_quantity_net_of_exchange_fees():
    """Binance charges spot fees in the received asset: a 0.01 BTC buy credits 0.00999 BTC."""
    handler = object.__new__(LiveExecutionHandler)  # no network / ccxt needed for the fee maths
    handler.market = {"base": "BTC", "quote": "USDT"}
    handler.cost_model = CostModel(CostConfig())
    fee_in_base = {"fees": [{"cost": 0.00001, "currency": "BTC"}]}
    assert handler._net_of_fees(fee_in_base, "buy", 0.01, 60_000) == pytest.approx((0.00999, 0.6))
    fee_in_quote = {"fee": {"cost": 0.6, "currency": "USDT"}}
    assert handler._net_of_fees(fee_in_quote, "sell", 0.01, 60_000) == pytest.approx((0.01, 0.6))
    fee_in_bnb = {"fees": [{"cost": 0.001, "currency": "BNB"}]}  # valued at the taker rate
    assert handler._net_of_fees(fee_in_bnb, "buy", 0.01, 60_000) == pytest.approx((0.01, 0.6))


class _EntriesFrom(IchimokuStrategy):
    """Backtest helper: ignore entry signals before the live engine starts trading."""

    def __init__(self, cfg: StrategyConfig, start: pd.Timestamp) -> None:
        super().__init__(cfg)
        self.start = start

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        sig = super().generate_signals(df)
        sig.loc[sig.index < self.start, ["long_entry", "short_entry"]] = False
        return sig


def test_live_replay_reproduces_the_event_driven_backtest(tmp_path):
    """Research/production parity: same bars in, same trades out (timing, side, exit reason)."""
    data = SyntheticDataGenerator(seed=11).generate(1000, "1h", start=pd.Timestamp("2025-06-01", tz="UTC"))
    cfg = AppConfig(
        data=DataConfig(source=DataSource.SYNTHETIC),
        strategy=StrategyConfig(allow_short=True, cross_lookback=3),  # paper mode may short
        risk=RiskConfig(max_drawdown_limit=0.0, daily_loss_limit=0.0),
        live=LiveConfig(feed=FeedType.REPLAY, heartbeat_sec=60, journal_db_path=tmp_path / "journal.sqlite"),
    )
    warmup = warmup_bars_needed(cfg)
    bus = EventBus()
    engine = LiveTradingEngine(cfg, ReplayFeed(cfg, data, warmup), bus=bus, executor=PaperExecutionHandler(cfg, bus))
    summary = run_async(engine.run())
    assert summary.bars_processed == len(data) - warmup

    backtest = EventDrivenBacktester(cfg, _EntriesFrom(cfg.strategy, data.index[warmup])).run(data)
    bt = backtest.trades[backtest.trades["exit_reason"] != "end_of_data"]
    live = pd.DataFrame(summary.trades)
    assert len(bt) >= 3
    key = ["side", "entry_time", "exit_time", "exit_reason"]
    assert live[key].to_dict("records") == bt[key].to_dict("records")
    np.testing.assert_allclose(live["entry_price"], bt["entry_price"], rtol=5e-4)


def test_cli_builds_config_from_arguments():
    from main import build_config, parse_args

    args = parse_args(["backtest", "--source", "synthetic", "--preset", "crypto", "--sizing", "kelly",
                       "--stop", "cloud", "--allow-short", "--fee", "0.0004", "--timeframe", "15m"])
    cfg = build_config(args)
    assert cfg.strategy.params == IchimokuParams(10, 30, 60, 30) and cfg.strategy.allow_short
    assert cfg.risk.sizing_method.value == "kelly" and cfg.risk.stop_method.value == "cloud"
    assert cfg.costs.taker_fee == 0.0004 and cfg.data.timeframe == "15m"
    assert cfg.live.feed is FeedType.REPLAY  # offline data is replayed, never streamed
    assert replace(cfg, risk=cfg.risk).to_dict()["exchange"].get("api_key") is None  # secrets never serialised
