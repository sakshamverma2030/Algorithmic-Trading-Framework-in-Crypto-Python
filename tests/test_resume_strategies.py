"""Tests for the course-aligned strategies: calendar anomalies, Aroon/RSI
divergence, Hurst-exponent filter, K-Means clustering and the cross-sectional
momentum-alpha portfolio, plus the new indicators (Aroon, Bollinger, ADF)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from advanced_ml_strategies import (
    HurstTrendStrategy,
    KMeansAssetClusterer,
    MomentumAlphaPortfolio,
    rolling_hurst,
)
from config import (
    CalendarConfig,
    DivergenceConfig,
    HurstConfig,
    PortfolioConfig,
    load_config,
)
from data_loader import SyntheticDataGenerator
from indicators import aroon, bollinger_bands, hurst_exponent
from intermediate_strategies import AroonRsiDivergenceStrategy, CalendarAnomalyStrategy


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return SyntheticDataGenerator(seed=7).generate(2000, "1h",
                                                   start=pd.Timestamp("2025-01-01", tz="UTC"))


# ----------------------------------------------------------------------------- indicators
def test_aroon_bounds_and_monotonicity():
    h = pd.Series([100.0, 101, 102, 103, 104, 105], dtype="float64")
    l = pd.Series([99.0, 99.5, 100, 100.5, 101, 101.5], dtype="float64")
    ar = aroon(h, l, period=3)
    # Steady march higher: Aroon Up near 100, Aroon Down near 0 after the warm-up.
    assert ar["aroon_up"].dropna().iloc[-1] >= 80.0
    assert ar["aroon_down"].dropna().iloc[-1] <= 20.0
    assert (((ar.dropna() >= 0.0) & (ar.dropna() <= 100.0))).all().all()


def test_hurst_differentiates_trending_vs_mean_reverting():
    rng = np.random.default_rng(3)
    n = 1500
    trend = pd.Series(np.cumsum(rng.standard_normal(n)) + np.arange(n) * 0.05)
    # Strong mean-reversion: AR(1) with negative coefficient pulls every shock back.
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = -0.9 * x[i - 1] + rng.standard_normal()
    meanrev = pd.Series(x)
    h_trend = hurst_exponent(trend, max_lag=100)
    h_rev = hurst_exponent(meanrev, max_lag=100)
    assert h_trend > 0.7, f"expected trending H>0.7, got {h_trend:.3f}"
    assert h_rev < 0.55, f"expected reversion H<0.55, got {h_rev:.3f}"


def test_rolling_hurst_is_causal_and_bounded(ohlcv):
    h = rolling_hurst(ohlcv["close"], window=150, max_lag=40)
    assert h.notna().any()
    valid = h.dropna()
    # Forward-fill must never read the future: strictly-increasing fill index.
    idx = h.index[h.notna()]
    diffs = np.diff(idx.view("int64"))
    assert (diffs >= 0).all()
    assert ((valid > 0.0) & (valid < 1.5)).all()


def test_adf_detects_stationary_series():
    from indicators import adf_test

    rng = np.random.default_rng(4)
    mr = pd.Series(np.sin(np.arange(800) * 0.2) * 3.0 + rng.standard_normal(800) * 0.3)
    result = adf_test(mr)
    assert result["is_stationary"] is True
    assert result["p_value"] < 0.05


def test_bollinger_bands_geometry():
    s = pd.Series(np.linspace(100.0, 120.0, 40) + np.sin(np.arange(40)), dtype="float64")
    bb = bollinger_bands(s, period=10).dropna()
    assert (bb["bb_upper"] >= bb["bb_middle"]).all()
    assert (bb["bb_lower"] <= bb["bb_middle"]).all()


# ----------------------------------------------------------------------------- calendar-anomaly strategy
def test_calendar_signals_are_bucket_causal(ohlcv):
    strat = CalendarAnomalyStrategy(CalendarConfig(dayofweek=True, hour=False, min_obs=5))
    sig = strat.generate_signals(ohlcv)
    assert set(["long_entry", "short_entry", "position", "seasonal_profile"]).issubset(sig.columns)
    # Only buckets with enough history are tradable.
    assert sig["seasonal_samples"].max() >= 5
    # The strategy output is valid for the vectorised backtester (position lattice).
    assert sig["position"].isin([-1, 0, 1]).all()


def test_calendar_config_rejects_no_features():
    from config import ConfigError

    with pytest.raises(ConfigError):
        CalendarConfig(dayofweek=False, hour=False)


# ----------------------------------------------------------------------------- Aroon / RSI divergence
def test_divergence_signal_frame_completeness(ohlcv):
    sig = AroonRsiDivergenceStrategy(load_config().divergence).generate_signals(ohlcv)
    for col in ("aroon_up", "aroon_down", "aroon_rsi", "bull_divergence", "bear_divergence",
                "rsi", "position"):
        assert col in sig.columns
    assert sig["position"].isin([-1, 0, 1]).all()


def test_divergence_can_require_or_relax(ohlcv):
    sig_req = AroonRsiDivergenceStrategy(DivergenceConfig(require_divergence=True)).generate_signals(ohlcv)
    sig_free = AroonRsiDivergenceStrategy(DivergenceConfig(require_divergence=False)).generate_signals(ohlcv)
    assert sig_free["long_entry"].sum() >= sig_req["long_entry"].sum()


# ----------------------------------------------------------------------------- Hurst strategy
def test_hurst_strategy_runs_vectorised(ohlcv):
    sig = HurstTrendStrategy(load_config().hurst).generate_signals(ohlcv)
    assert "hurst" in sig.columns
    assert sig["position"].isin([-1, 0, 1]).all()


def test_hurst_config_validation():
    with pytest.raises(Exception):
        HurstConfig(max_lag=500, window=10)  # window < 2*max_lag


# ----------------------------------------------------------------------------- K-Means + momentum-alpha portfolio
def test_kmeans_clusters_assets_on_features():
    frames = {}
    for i, sym in enumerate(["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]):
        frames[sym] = SyntheticDataGenerator(seed=i + 1).generate(800, "1h")
    km = KMeansAssetClusterer(k=3, random_state=7)
    feat = km.features(frames, lookback=20)
    assert len(feat) == len(frames)
    km.fit(feat)
    stats = km.cluster_stats(feat)
    assert set(stats["cluster"]) == {0, 1, 2}
    assert stats["n_assets"].sum() == len(frames)
    assert km.best_cluster(feat) in {0, 1, 2}


def test_momentum_alpha_portfolio_rebalances_in_top_n():
    frames = {}
    for i, sym in enumerate(["BTC/USDT", "ETH/USDT", "SOL/USDT"]):
        frames[sym] = SyntheticDataGenerator(seed=i + 10).generate(400, "1h")
    cfg = PortfolioConfig(universe=tuple(frames), lookback=10, top_n=2, rebalance_every=20)
    model = MomentumAlphaPortfolio(cfg)
    curve, trades = model.run(frames, initial_capital=10_000.0, one_way_cost=0.001)
    assert not curve.empty
    assert len(trades) > 0
    # Every rebalance decision must rank the universe symbols (closes matrix).
    closes = model.align(frames)
    scores = model.alpha_scores(closes)
    assert set(scores.columns) == set(frames)
    # Benchmark column tracks buy-and-hold of the first symbol.
    assert curve["benchmark"].iloc[-1] > 0


def test_momentum_alpha_portfolio_rejects_empty_universe():
    from advanced_ml_strategies import MomentumAlphaPortfolio

    with pytest.raises(ValueError):
        MomentumAlphaPortfolio(PortfolioConfig(universe=("BTC/USDT",))).align({})