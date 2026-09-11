"""
momentum.py - Long / short time-series momentum strategy.

A second, independent strategy for the framework, sharing the *exact* same
signal-frame contract as ``IchimokuStrategy`` so that ``VectorizedBacktester``
and ``EventDrivenBacktester`` run it unchanged (identical risk model, stops,
costs and journaling).

Trading rules (decided on the close of bar t, executed at the open of t+1):

    momentum_t      = Close_t / Close_{t-lookback} - 1
    Long entry      momentum_t >= entry_threshold  [and Close_t > SMA when require_above_sma]
    Long exit       momentum_t <  exit_threshold   [or  Close_t < SMA  when require_above_sma]
    Short entry     momentum_t <= -entry_threshold (only when allow_short)
    Short exit      momentum_t >  -exit_threshold

The "cloud" the risk engine needs is the strategy's own Donchian channel of
``high``/``low`` over the lookback window, and the "Kijun" line is the SMA, so
ATR stops, channel-style stops and the Kijun trailing stop all work unchanged.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import MomentumConfig
from indicators import average_true_range, rolling_max, rolling_min
from strategy import IchimokuStrategy, SignalSnapshot

logger = logging.getLogger(__name__)


class MomentumStrategy:
    """Stateless, vectorised momentum signal generator (mirrors IchimokuStrategy)."""

    def __init__(self, cfg: MomentumConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ strategy API
    @property
    def name(self) -> str:
        return self.cfg.label

    @property
    def allow_short(self) -> bool:
        """Whether the strategy config permits short entries."""
        return self.cfg.allow_short

    @property
    def warmup_bars(self) -> int:
        """Bars needed before every indicator is defined."""
        c = self.cfg
        return max(c.lookback, c.sma_period, c.atr_period + 1) + 1

    @property
    def history_bars(self) -> int:
        """History for a latest signal to equal a full-history computation (ATR seed decay)."""
        return self.warmup_bars + 20 * self.cfg.atr_period

    # ------------------------------------------------------------------ signals
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        close = df["close"]
        atr = average_true_range(df["high"], df["low"], close, c.atr_period)

        cloud_top = rolling_max(df["high"], c.lookback)
        cloud_bottom = rolling_min(df["low"], c.lookback)
        sma = close.rolling(c.sma_period, min_periods=c.sma_period).mean()
        momentum = close / close.shift(c.lookback) - 1.0

        long_entry = momentum >= c.entry_threshold
        exit_long = momentum < c.exit_threshold
        if c.require_above_sma:
            above_sma = close > sma
            long_entry &= above_sma
            exit_long |= close < sma

        if c.allow_short:
            short_entry = momentum <= -c.entry_threshold
            exit_short = momentum > -c.exit_threshold
        else:
            short_entry = pd.Series(False, index=df.index)
            exit_short = pd.Series(False, index=df.index)

        out = df[["open", "high", "low", "close", "volume"]].copy()
        out["atr"] = atr
        out["natr"] = atr / close
        out["tenkan"] = sma
        out["kijun"] = sma
        out["cloud_top"] = cloud_top
        out["cloud_bottom"] = cloud_bottom
        out["momentum"] = momentum
        out["long_entry"] = long_entry.fillna(False)
        out["short_entry"] = short_entry.fillna(False)
        out["exit_long"] = exit_long.fillna(False)
        out["exit_short"] = exit_short.fillna(False)
        out["position"] = IchimokuStrategy.signals_to_position(
            out["long_entry"], out["short_entry"], out["exit_long"], out["exit_short"]
        )
        return out

    def latest_signal(self, df: pd.DataFrame) -> SignalSnapshot:
        """Signal state on the last row of ``df`` (the most recent closed bar)."""
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
            rsi=None,
        )