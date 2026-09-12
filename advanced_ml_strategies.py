"""
advanced_ml_strategies.py - Advanced / ML strategy modules.

    1. KMeansAssetClusterer      - unsupervised clustering of assets in an
                                   (volatility, momentum, volume) feature space.
    2. HurstTrendStrategy        - R/S Hurst-exponent regime filter combined with RSI.
    3. MomentumAlphaPortfolio    - cross-sectional momentum: 2-day return rankings,
                                   combined alpha scores, top-N dynamic rebalancing.
    4. PairsBacktester           - cointegrated pairs trading (re-exported from
                                   ``pairs.py``: ADF/cointegration, OLS hedge ratio,
                                   spread Bollinger z-score execution).

All single-symbol strategies obey the shared signal-frame contract so the existing
``VectorizedBacktester`` / ``EventDrivenBacktester`` run them unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import HurstConfig, PortfolioConfig
from indicators import (
    HAS_SKLEARN,
    KMeans,
    adf_test,
    average_true_range,
    hurst_exponent,
    relative_strength_index,
)
from pairs import CointegrationAnalyzer, PairsBacktester, check_pair
from strategy import IchimokuStrategy, SignalSnapshot

__all__ = [
    "KMeansAssetClusterer",
    "HurstTrendStrategy",
    "MomentumAlphaPortfolio",
    "PairsBacktester",
    "CointegrationAnalyzer",
    "check_pair",
    "adf_test",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# 1. K-Means asset clustering (unsupervised ML)
# --------------------------------------------------------------------------------------
class KMeansAssetClusterer:
    """Cluster a universe of crypto assets on risk/momentum/liquidity features.

    Features (per asset, over ``lookback`` bars):
        volatility : annualised stdev of log returns
        momentum   : total log return over the window
        volume     : mean quote-volume (log-scaled)

    ``KMeans(n_clusters, n_init=10, random_state)`` (scikit-learn when installed,
    scipy ``kmeans2`` otherwise) partitions the assets; ``cluster_stats`` returns the
    per-cluster centroid, so the "best" cluster (highest momentum, lowest risk) can
    be traded automatically.
    """

    def __init__(self, cfg: PortfolioConfig | None = None, k: int | None = None,
                 random_state: int = 42) -> None:
        if cfg is not None:
            self.k = cfg.kmeans_clusters
            self.random_state = cfg.kmeans_random_state
        else:
            self.k = k or 3
            self.random_state = random_state
        if self.k < 2:
            raise ValueError("K must be >= 2")
        self.labels_: dict[str, int] = {}
        self.centers_: np.ndarray | None = None

    def features(self, frames: dict[str, pd.DataFrame], lookback: int = 20) -> pd.DataFrame:
        """Build the (assets x 3) feature matrix from a dict of symbol -> OHLCV frames."""
        rows: dict[str, list[float]] = {}
        for sym, df in frames.items():
            close, vol = df["close"], df["volume"]
            if len(close) < lookback + 2:
                continue
            log_ret = np.log(close / close.shift(1)).dropna()
            last = close.iloc[-lookback:]
            feat = [
                float(log_ret.iloc[-lookback:].std() * np.sqrt(365.0 * 24.0 * 60.0)) if len(log_ret) >= lookback else float("nan"),
                float(np.log(last.iloc[-1] / last.iloc[0])) if last.iloc[0] > 0 else float("nan"),
                float(np.log1p(vol.iloc[-lookback:].mean())) if len(vol) >= lookback else float("nan"),
            ]
            rows[sym] = feat
        out = pd.DataFrame.from_dict(rows, orient="index", columns=["volatility", "momentum", "log_volume"])
        out.dropna(inplace=True)
        return out

    def fit(self, features: pd.DataFrame) -> "KMeansAssetClusterer":
        """Fit K-Means on the feature matrix; returns self for chaining."""
        if len(features) < 2:
            raise ValueError("K-Means needs at least 2 assets with data")
        x = features.to_numpy(dtype="float64")
        # Standardise features so volatility/momentum/volume are comparable.
        mu, sd = x.mean(axis=0), x.std(axis=0)
        xz = (x - mu) / np.where(sd < 1e-12, 1.0, sd)
        if HAS_SKLEARN:
            model = KMeans(n_clusters=self.k, n_init=10, random_state=self.random_state)
            labels = model.fit_predict(xz)
            centers = model.cluster_centers_
        else:
            from scipy.cluster.vq import kmeans2
            centers, labels = kmeans2(xz, self.k, minit="points", seed=self.random_state)
        self.labels_ = dict(zip(features.index, [int(l) for l in labels]))
        self.centers_ = centers
        return self

    def best_cluster(self, features: pd.DataFrame) -> int:
        """Cluster with the highest momentum centroid (highest growth, tradable)."""
        if self.centers_ is None:
            raise RuntimeError("fit() must be called before best_cluster()")
        centers_z = (self.centers_ - features.to_numpy().mean(axis=0)) / np.where(
            features.to_numpy().std(axis=0) < 1e-12, 1.0, features.to_numpy().std(axis=0)
        )
        return int(np.argmax(centers_z[:, 1]))  # momentum feature

    def cluster_stats(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per-cluster table: size, mean momentum, mean volatility, members."""
        if self.labels_:
            labels = pd.Series(self.labels_, dtype=int)
        else:
            self.fit(features)
            labels = pd.Series(self.labels_, dtype=int)
        stats = []
        for cluster in sorted(set(labels)):
            members = list(labels[labels == cluster].index)
            sub = features.loc[members]
            stats.append({"cluster": cluster, "n_assets": len(members),
                          "mean_volatility": sub["volatility"].mean(),
                          "mean_momentum": sub["momentum"].mean(),
                          "members": ", ".join(members)})
        return pd.DataFrame(stats)


# --------------------------------------------------------------------------------------
# 2. Hurst-exponent trend regime + RSI filter strategy
# --------------------------------------------------------------------------------------
def rolling_hurst(close: pd.Series, window: int, max_lag: int) -> pd.Series:
    """Rolling rescaled-range Hurst exponent (causal, stride-sampled).

    R/S estimation is O(n * max_lag) per window, so recomputing it on *every* bar is
    wasteful. We estimate it once per ``window // 5`` bars (``stride``) and forward-fill
    the stale value in between - the fill only uses data strictly <= its own bar, so
    the signal remains look-ahead free. For a runtime trade-off use a smaller window.
    """
    stride = max(1, window // 5)
    n = len(close)
    windows = range(window - 1, n, stride)
    values: list[tuple[int, float]] = []
    for i in windows:
        w = close.iloc[i - window + 1: i + 1]
        values.append((i, hurst_exponent(w, max_lag=max_lag)))
    out = pd.Series(np.nan, index=close.index, dtype="float64")
    if values:
        for i, h in values:
            out.iloc[i] = h
        out = out.ffill()
    return out


class HurstTrendStrategy:
    """Trade the Hurst exponent H with an RSI trigger.

    Regime classification from H (R/S analysis, Peters 1994):
        H > 0.5          persistent / trending      ->  follow momentum
        H < 0.5          anti-persistent / reverting ->  fade extremes
        H ~ 0.5          random walk                ->  stand aside

    Entry combines the H regime with Wilder RSI:
        trending regime      long  when RSI re-crosses 50 (trend continuation)
        reversion regime     long  when RSI < oversold threshold (buy the dip)
    A trailing regime (H ~ 0.5) forces the position flat.
    """

    def __init__(self, cfg: HurstConfig) -> None:
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.label

    @property
    def allow_short(self) -> bool:
        return True

    @property
    def warmup_bars(self) -> int:
        c = self.cfg
        return max(c.window, c.min_hurst_obs) + c.rsi_period + 5

    @property
    def history_bars(self) -> int:
        return self.warmup_bars * 2

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        close = df["close"]
        atr = average_true_range(df["high"], df["low"], close, 14)
        rsi = relative_strength_index(close, c.rsi_period)
        h = rolling_hurst(close, c.window, c.max_lag)

        trend_regime = h > c.trend_threshold
        revert_regime = h < c.reversion_threshold
        neutral = ~trend_regime & ~revert_regime

        rsi_prev = rsi.shift(1)
        cross_up_50 = (rsi_prev <= 50.0) & (rsi > 50.0)
        cross_down_50 = (rsi_prev >= 50.0) & (rsi < 50.0)
        oversold = rsi < c.rsi_oversold

        long_entry = ((trend_regime & cross_up_50) | (revert_regime & oversold)).fillna(False)
        short_entry = (trend_regime & cross_down_50).fillna(False)  # symmetric momentum fade
        exit_long = exit_from_neutral(neutral, trend_regime)
        exit_short = exit_long

        out = df[["open", "high", "low", "close", "volume"]].copy()
        out["atr"] = atr
        out["natr"] = atr / close
        out["tenkan"] = close.rolling(9).mean()
        out["kijun"] = close.rolling(26).mean()
        out["cloud_top"] = close.rolling(20).max().rolling(20).mean()
        out["cloud_bottom"] = close.rolling(20).min().rolling(20).mean()
        out["hurst"] = h
        out["rsi"] = rsi
        out["long_entry"] = long_entry
        out["short_entry"] = short_entry
        out["exit_long"] = exit_long.fillna(False)
        out["exit_short"] = exit_short.fillna(False)
        out["position"] = IchimokuStrategy.signals_to_position(
            out["long_entry"], out["short_entry"], out["exit_long"], out["exit_short"]
        )
        return out

    def latest_signal(self, df: pd.DataFrame) -> SignalSnapshot:
        if df.empty:
            raise ValueError("latest_signal() needs at least one bar")
        sig = self.generate_signals(df)
        row = sig.iloc[-1]
        return SignalSnapshot(
            timestamp=sig.index[-1],
            close=float(row["close"]),
            long_entry=bool(row["long_entry"]),
            short_entry=bool(row["short_entry"]),
            exit_long=bool(row["exit_long"]),
            exit_short=bool(row["exit_short"]),
            target_position=int(row["position"]),
            tenkan=float(row["tenkan"]) if np.isfinite(row["tenkan"]) else float("nan"),
            kijun=float(row["kijun"]) if np.isfinite(row["kijun"]) else float("nan"),
            cloud_top=float(row["cloud_top"]) if np.isfinite(row["cloud_top"]) else float("nan"),
            cloud_bottom=float(row["cloud_bottom"]) if np.isfinite(row["cloud_bottom"]) else float("nan"),
            atr=float(row["atr"]) if np.isfinite(row["atr"]) else float("nan"),
            regime=None,
            rsi=float(row["rsi"]) if np.isfinite(row["rsi"]) else None,
        )


def exit_from_neutral(neutral: pd.Series, trend_regime: pd.Series) -> pd.Series:
    """Exit a position once the regime stops being decisively trending."""
    return (neutral | trend_regime.shift(1).fillna(False) | trend_regime.shift(2).fillna(False))


# --------------------------------------------------------------------------------------
# 3. Long-only momentum & quant framework (cross-sectional)
# --------------------------------------------------------------------------------------
class MomentumAlphaPortfolio:
    """Cross-sectional momentum over a universe with top-N rebalancing.

    Score per symbol ``s`` at rebalance ``t``:
        ret2    = Close_t / Close_{t-2} - 1          (2-day return ranking)
        retL    = Close_t / Close_{t-lookback} - 1   (medium-term return)
        alpha   = rank(ret2) + rank(retL) - 2*rank(vol)   (combined alpha score)
        vol     = std of log returns over lookback

    The ``top_n`` highest-scoring assets are held with equal weight until the next
    rebalance (every ``rebalance_every`` bars). Returns are computed causally: the
    ranking of bar t decides the portfolio held during bar t+1.
    """

    def __init__(self, cfg: PortfolioConfig) -> None:
        self.cfg = cfg
        self.last_alpha_: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "alpha momentum " + self.cfg.label

    @property
    def warmup_bars(self) -> int:
        return max(self.cfg.lookback, 3) + 2

    def align(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Inner-join every universe symbol into a single close matrix."""
        closes = []
        for sym in self.cfg.universe:
            if sym in frames and len(frames[sym]):
                closes.append(frames[sym]["close"].rename(sym))
        if not closes:
            raise ValueError("No universe symbols have data")
        joined = pd.concat(closes, axis=1).dropna()
        common = sorted(set(joined.columns))
        return joined[common]

    def alpha_scores(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Combined alpha score per symbol (rank of short & medium momentum minus vol)."""
        c = self.cfg
        ret2 = closes / closes.shift(2) - 1.0
        retL = closes / closes.shift(c.lookback) - 1.0
        logret = np.log(closes / closes.shift(1))
        vol = logret.rolling(c.lookback).std()
        scores = (
            ret2.rank(axis=1) + retL.rank(axis=1) - 2.0 * vol.rank(axis=1)
        )
        self.last_alpha_ = scores
        return scores

    def run(self, frames: dict[str, pd.DataFrame],
            initial_capital: float = 10_000.0, one_way_cost: float = 0.001
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Equal-weight top-N portfolio backtest.

        Returns ``(equity_curve, trades)``. ``trades`` summarises every weight change:
        symbol, rebalance time, weight before/after.
        """
        closes = self.align(frames)
        scores = self.alpha_scores(closes)
        c = self.cfg
        n = len(closes)
        equity = np.full(n, np.nan)
        equity[:self.warmup_bars] = 1.0
        trades: list[dict[str, Any]] = []
        weights: dict[str, float] = {s: 0.0 for s in closes.columns}
        new_w: dict[str, float] = weights

        for i in range(self.warmup_bars, n):
            if i % c.rebalance_every == 0:
                row = scores.iloc[i - 1]  # ranking of the previous close (causal)
                top = row.dropna().sort_values(ascending=False).head(c.top_n).index
                new_w = {s: 0.0 for s in closes.columns}
                if len(top):
                    fill = 1.0 / len(top)
                    for s in top:
                        new_w[s] = fill
                turnover = sum(abs(new_w[s] - weights[s]) for s in closes.columns)
                if turnover > 1e-9:
                    for s in closes.columns:
                        delta = new_w[s] - weights[s]
                        if abs(delta) > 1e-9:
                            trades.append({"time": closes.index[i], "symbol": s,
                                           "weight_before": weights[s],
                                           "weight_after": new_w[s],
                                           "turnover": abs(delta)})
                weights = new_w
            else:
                turnover = 0.0
            # Portfolio return uses weights set at i and the next-day returns.
            ret_t = sum(weights[s] * (closes[s].iloc[i] / closes[s].iloc[i - 1] - 1.0)
                        for s in closes.columns)
            ret_t -= turnover * one_way_cost
            equity[i] = equity[i - 1] * (1.0 + ret_t)

        curve = pd.DataFrame({"equity": equity * initial_capital}, index=closes.index)
        first = next((s for s in self.cfg.universe if s in closes.columns), closes.columns[0])
        curve["benchmark"] = initial_capital * closes[first] / closes[first].iloc[0]
        trades_df = pd.DataFrame(trades)
        return curve, trades_df

    def rebalance(self, frames: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Latest (causal) target weights for the real-time engine."""
        closes = self.align(frames)
        scores = self.alpha_scores(closes)
        row = scores.iloc[-1].dropna().sort_values(ascending=False)
        top = row.head(self.cfg.top_n).index
        return {s: (1.0 / len(top) if len(top) else 0.0) for s in top}