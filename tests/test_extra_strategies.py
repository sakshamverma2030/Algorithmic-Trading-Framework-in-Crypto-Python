"""Tests for the RSI overlay, K-Means regimes, momentum and pairs-trading modules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtester import BacktestError, EventDrivenBacktester, VectorizedBacktester
from config import (
    AppConfig,
    MomentumConfig,
    PairsConfig,
    RegimeMethod,
    StrategyConfig,
)
from data_loader import SyntheticDataGenerator
from indicators import (
    AdaptiveIchimoku,
    relative_strength_index,
    rsi_divergence,
)
from momentum import MomentumStrategy
from pairs import CointegrationAnalyzer, PairsBacktester
from risk import ExitReason
from strategy import IchimokuStrategy


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return SyntheticDataGenerator(seed=9).generate(1200, "1h", start=pd.Timestamp("2025-01-01", tz="UTC"))


# ----------------------------------------------------------------------------- RSI
def test_rsi_matches_wilder_reference(ohlcv):
    n = 14
    close = ohlcv["close"].to_numpy()
    delta = np.diff(close)
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    ag = np.full(len(close), np.nan)
    al = np.full(len(close), np.nan)
    ag[n] = gain[:n].mean()
    al[n] = loss[:n].mean()
    for t in range(n + 1, len(close)):
        ag[t] = (ag[t - 1] * (n - 1) + gain[t - 1]) / n
        al[t] = (al[t - 1] * (n - 1) + loss[t - 1]) / n
    ref = 100.0 - 100.0 / (1.0 + ag / al)
    rsi = relative_strength_index(ohlcv["close"], n).to_numpy()
    assert np.isnan(rsi[:n]).all()
    np.testing.assert_allclose(rsi[n:], ref[n:], rtol=1e-9)


def test_rsi_divergence_detects_constructed_patterns():
    def frame(rsi_vals, close_vals):
        idx = pd.date_range("2026-01-01", periods=len(rsi_vals), freq="h")
        return pd.Series(rsi_vals, index=idx), pd.Series(close_vals, index=idx)

    # Bullish: recent lower-low in price with a higher RSI low, then a turn up.
    rsi_b, close_b = frame(
        [45, 42, 40, 43, 41, 38, 42, 44, 41, 39, 36, 39, 44, 46, 45, 43],
        [102, 100, 101, 99, 98, 99, 97, 95, 96, 94, 93, 90, 92, 94, 95, 96],
    )
    out = rsi_divergence(rsi_b, close_b, lookback=6)
    assert out["rsi_bull_div"].any() and not out["rsi_bear_div"].any()

    # Bearish: three rising price peaks (higher highs) but RSI peaks keep falling
    # (lower RSI highs) - exhaustion of the uptrend, then a turn down.
    rsi_s, close_s = frame(
        [60, 64, 68, 70, 68, 65, 64, 66, 68, 66, 63, 60, 62, 61, 59, 58],
        [100, 102, 104, 106, 105, 103, 107, 108, 110, 109, 107, 105, 112, 111, 110, 109],
    )
    out = rsi_divergence(rsi_s, close_s, lookback=6)
    assert out["rsi_bear_div"].any() and not out["rsi_bull_div"].any()


def test_rsi_and_divergence_are_causal(ohlcv):
    cfg = StrategyConfig(use_rsi_filter=True, use_rsi_divergence=True, divergence_lookback=8)
    strategy = IchimokuStrategy(cfg)
    full = strategy.generate_signals(ohlcv)
    cut = 800
    part = strategy.generate_signals(ohlcv.iloc[:cut])
    for col in ("rsi", "rsi_bull_div", "rsi_bear_div", "long_entry", "exit_long"):
        pd.testing.assert_series_equal(part[col], full[col].iloc[:cut], check_dtype=False)


def test_rsi_filter_blocks_overbought_entries(ohlcv):
    plain = IchimokuStrategy(StrategyConfig()).generate_signals(ohlcv)
    filtered = IchimokuStrategy(StrategyConfig(use_rsi_filter=True)).generate_signals(ohlcv)
    assert int(filtered["long_entry"].sum()) <= int(plain["long_entry"].sum())
    harness = filtered.loc[filtered["long_entry"]]
    if len(harness):
        assert (harness["rsi"] < 70).all()


# ----------------------------------------------------------------------------- K-Means regime
def test_kmeans_regime_labels_and_causality(ohlcv):
    cfg = StrategyConfig(dynamic=True, regime_lookback=120, regime_method=RegimeMethod.KMEANS,
                         kmeans_clusters=3, regime_presets=("standard", "crypto", "crypto_slow"))
    strategy = IchimokuStrategy(cfg)
    full = strategy.generate_signals(ohlcv)
    assert set(np.unique(full["regime"].dropna())) <= {AdaptiveIchimoku.LOW,
                                                       AdaptiveIchimoku.NORMAL, AdaptiveIchimoku.HIGH}
    part = strategy.generate_signals(ohlcv.iloc[:700])
    pd.testing.assert_series_equal(part["regime"], full["regime"].iloc[:700], check_dtype=False)


def test_kmeans_requires_three_clusters():
    with pytest.raises(ValueError):
        StrategyConfig(dynamic=True, regime_method=RegimeMethod.KMEANS, kmeans_clusters=2)


# ----------------------------------------------------------------------------- momentum
def test_momentum_signals_never_use_future_data(ohlcv):
    strategy = MomentumStrategy(MomentumConfig(lookback=20, sma_period=50, require_above_sma=True, allow_short=True))
    full = strategy.generate_signals(ohlcv)
    cut = 800
    part = strategy.generate_signals(ohlcv.iloc[:cut])
    for col in ("momentum", "long_entry", "short_entry", "exit_long", "position"):
        pd.testing.assert_series_equal(part[col], full[col].iloc[:cut], check_dtype=False)


def test_momentum_on_a_clean_trend(ohlcv):
    strategy = MomentumStrategy(MomentumConfig(lookback=20, entry_threshold=0.0, allow_short=True))
    sig = strategy.generate_signals(ohlcv)
    assert (sig["momentum"].dropna() == (sig["close"] / sig["close"].shift(20) - 1).dropna()).all()
    # A long-only momentum run plus a short-enabled run both complete through the engines.
    for side_cfg in (MomentumConfig(entry_threshold=0.0), MomentumConfig(entry_threshold=0.0, allow_short=True)):
        cfg = AppConfig(strategy=StrategyConfig(allow_short=side_cfg.allow_short), momentum=side_cfg)
        res = EventDrivenBacktester(cfg, MomentumStrategy(side_cfg)).run(ohlcv)
        assert res.equity_curve["equity"].iloc[-1] > 0
        assert set(res.trades["exit_reason"]) <= {r.value for r in ExitReason}


def test_momentum_latest_signal(ohlcv):
    strategy = MomentumStrategy(MomentumConfig())
    snap = strategy.latest_signal(ohlcv)
    assert snap.is_ready and snap.target_position in (-1, 0, 1)


# ----------------------------------------------------------------------------- pairs
def make_cointegrated_pair(n=1500, beta=1.2, phi=0.9):
    rng = np.random.default_rng(7)
    logx = np.cumsum(rng.normal(0, 0.008, n))
    spread = np.empty(n)
    spread[0] = 0.0
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + rng.normal(0, 0.002)
    logy = beta * logx + spread
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    one = lambda arr: pd.DataFrame({"open": np.exp(arr), "high": np.exp(arr) * 1.001,
                                    "low": np.exp(arr) * 0.999, "close": np.exp(arr),
                                    "volume": 100.0}, index=idx)  # noqa: E731
    return one(logx), one(logy)


def test_cointegration_analyzer_finds_the_hedge():
    base, quote = make_cointegrated_pair()
    diag = CointegrationAnalyzer.diagnostics(base, quote, window=120)
    assert diag["return_correlation"] > 0.9
    assert diag["hedge_beta_mean"] == pytest.approx(1.2, rel=0.15)
    assert math.isfinite(diag["half_life_bars"]) and 0 < diag["half_life_bars"] < 500


def test_pairs_beta_and_zscore_are_causal(base_quote=None):
    base, quote = base_quote if base_quote else make_cointegrated_pair(n=900)
    x, y = np.log(base["close"]), np.log(quote["close"])
    window = 60
    full_beta = CointegrationAnalyzer.beta_series(x, y, window)
    cut = 600
    part_beta = CointegrationAnalyzer.beta_series(x.iloc[:cut], y.iloc[:cut], window)
    np.testing.assert_allclose(part_beta.dropna().to_numpy(), full_beta.iloc[:cut].dropna().to_numpy())
    full_z = CointegrationAnalyzer.zscore(CointegrationAnalyzer.spread_series(x, y, full_beta), window)
    part_spread = CointegrationAnalyzer.spread_series(x.iloc[:cut], y.iloc[:cut], part_beta)
    part_z = CointegrationAnalyzer.zscore(part_spread, window)
    np.testing.assert_allclose(part_z.dropna().to_numpy(), full_z.iloc[:cut].dropna().to_numpy())


def test_pairs_backtest_accounting_and_trades():
    base, quote = make_cointegrated_pair()
    cfg = AppConfig(pairs=PairsConfig(lookback=60, entry_zscore=1.5, exit_zscore=0.3, max_hold_bars=200))
    result = PairsBacktester(cfg).run(base, quote)
    assert len(result.trades) >= 3
    assert result.equity_curve["position"].iloc[-1] == 0
    assert result.trades["net_pnl"].sum() == pytest.approx(
        result.final_equity - result.initial_capital, rel=1e-6)
    assert set(result.trades["exit_reason"]) <= {r.value for r in ExitReason}
    assert result.metadata["pairs"]["annualised_correlation"] > 0.5


def test_pairs_need_aligned_data():
    base, _ = make_cointegrated_pair(n=300)
    quote_small, _ = make_cointegrated_pair(n=60)
    with pytest.raises(BacktestError):
        PairsBacktester(AppConfig()).run(base, quote_small)


# ----------------------------------------------------------------------------- CLI wiring
def test_cli_parses_new_strategy_flags():
    from main import build_config, parse_args

    cfg = build_config(parse_args(["backtest", "--rsi", "--rsi-divergence", "--rsi-period", "9",
                                   "--regime", "kmeans", "--preset", "dynamic",
                                   "--divergence-lookback", "8"]))
    assert cfg.strategy.use_rsi_filter and cfg.strategy.use_rsi_divergence
    assert cfg.strategy.rsi_period == 9 and cfg.strategy.divergence_lookback == 8
    assert cfg.strategy.regime_method is RegimeMethod.KMEANS and cfg.strategy.dynamic

    mom = build_config(parse_args(["momentum", "--mom-lookback", "30", "--mom-require-sma", "--mom-short"]))
    assert mom.momentum.lookback == 30 and mom.momentum.require_above_sma and mom.momentum.allow_short

    prs = build_config(parse_args(["pairs", "--pair", "LTC/USDT", "--entry-z", "1.5", "--raw-prices"]))
    assert prs.pairs.quote_symbol == "LTC/USDT" and prs.pairs.entry_zscore == 1.5
    assert not prs.pairs.use_log_prices


# ----------------------------------------------------------------------------- live strategy selection
def test_live_strategy_factory_selects_momentum():
    from main import build_config, parse_args
    from config import StrategyKind
    from live_trader import build_strategy, warmup_bars_needed
    from momentum import MomentumStrategy
    from strategy import IchimokuStrategy

    default = build_config(parse_args(["live"]))
    assert default.live.live_strategy is StrategyKind.ICHIMOKU
    assert isinstance(build_strategy(default), IchimokuStrategy)
    assert warmup_bars_needed(default) == max(default.live.warmup_bars, IchimokuStrategy(default.strategy).history_bars)

    mom = build_config(parse_args(["live", "--live-strategy", "momentum"]))
    assert mom.live.live_strategy is StrategyKind.MOMENTUM
    assert isinstance(build_strategy(mom), MomentumStrategy)
    assert warmup_bars_needed(mom) == max(mom.live.warmup_bars, MomentumStrategy(mom.momentum).history_bars)

    strategy = build_strategy(mom)
    assert strategy.name == mom.momentum.label
    assert strategy.allow_short == mom.momentum.allow_short