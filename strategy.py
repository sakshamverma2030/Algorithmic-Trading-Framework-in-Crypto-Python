"""
strategy.py - Ichimoku Cloud signal generation.

Trading rules (evaluated on the CLOSED bar t, executed at the next bar's open):

    Long entry   Close_t > Kumo top_t                        (price above the cloud)
                 AND Tenkan crosses above Kijun              (bullish TK cross, within
                     `cross_lookback` bars, and still Tenkan > Kijun)
                 AND Close_t > Close_{t-d}                   (Chikou above the price it
                                                              is plotted against)
                 [AND Senkou A lead > Senkou B lead]         (optional Kumo-twist filter)

    Long exit    Close_t < Kumo bottom_t  OR  Tenkan crosses below Kijun

    Short entry  exact mirror of the long entry (only when ``allow_short``); using the full
                 confluence avoids shorting on a lone TK cross while price is above the Kumo
    Short exit   Close_t > Kumo top_t     OR  Tenkan crosses above Kijun

A TK cross is a sign change of D_t = Tenkan_t - Kijun_t:
    bullish cross:  D_t > 0  and  D_{t-1} <= 0
    bearish cross:  D_t < 0  and  D_{t-1} >= 0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from config import ICHIMOKU_PRESETS, StrategyConfig
from indicators import build_ichimoku, compute_indicators

logger = logging.getLogger(__name__)


class Signal(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass(frozen=True)
class SignalSnapshot:
    """Strategy state on the most recent closed bar (consumed by the live engine)."""

    timestamp: pd.Timestamp
    close: float
    long_entry: bool
    short_entry: bool
    exit_long: bool
    exit_short: bool
    target_position: int
    tenkan: float
    kijun: float
    cloud_top: float
    cloud_bottom: float
    atr: float
    regime: int | None = None
    rsi: float | None = None

    @property
    def is_ready(self) -> bool:
        """False while indicators are still warming up (NaN)."""
        return bool(np.isfinite([self.tenkan, self.kijun, self.cloud_top, self.cloud_bottom, self.atr]).all())

    def describe(self) -> str:
        flags = [name for name, on in (("LONG_ENTRY", self.long_entry), ("SHORT_ENTRY", self.short_entry),
                                       ("EXIT_LONG", self.exit_long), ("EXIT_SHORT", self.exit_short)) if on]
        return (f"{self.timestamp:%Y-%m-%d %H:%M} close={self.close:,.2f} tenkan={self.tenkan:,.2f} "
                f"kijun={self.kijun:,.2f} kumo=[{self.cloud_bottom:,.2f}, {self.cloud_top:,.2f}] "
                f"atr={self.atr:,.2f} signals={','.join(flags) or 'none'}")


class IchimokuStrategy:
    """Stateless, vectorised signal generator shared by the backtesters and the live engine."""

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.indicator_engine = build_ichimoku(cfg)

    @property
    def name(self) -> str:
        return self.cfg.label

    @property
    def warmup_bars(self) -> int:
        """Bars needed before every indicator is defined."""
        if self.cfg.dynamic:
            params = [ICHIMOKU_PRESETS[p] for p in self.cfg.regime_presets]
            extra = self.cfg.regime_lookback // 4  # min_periods of the volatility-rank window
        else:
            params, extra = [self.cfg.params], 0
        return max(p.warmup for p in params) + max(self.cfg.atr_period, extra) + 1

    @property
    def history_bars(self) -> int:
        """History needed for the latest signal to equal a full-history computation.

        The Ichimoku lines have finite memory, but Wilder's ATR is recursive: its seed decays
        like (1 - 1/n)^k, i.e. below 1e-9 after 20 * n bars. The dynamic mode also needs a
        full volatility-rank window. The live engine computes signals on exactly this many bars.
        """
        extra = self.cfg.regime_lookback if self.cfg.dynamic else 0
        return self.warmup_bars + 20 * self.cfg.atr_period + extra

    # ----------------------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicators plus boolean signal columns and the vectorised target position."""
        cfg = self.cfg
        ind = compute_indicators(df, cfg)
        close = ind["close"]

        above_cloud = close > ind["cloud_top"]
        below_cloud = close < ind["cloud_bottom"]

        tk_diff = ind["tenkan"] - ind["kijun"]
        prev_diff = tk_diff.shift(1)
        bull_cross = (tk_diff > 0) & (prev_diff <= 0)
        bear_cross = (tk_diff < 0) & (prev_diff >= 0)
        if cfg.dynamic:  # a regime switch swaps the line definitions: not a genuine cross
            bull_cross &= ind["regime_stable"]
            bear_cross &= ind["regime_stable"]

        if cfg.cross_lookback > 1:
            recent_bull = bull_cross.astype("float64").rolling(cfg.cross_lookback, min_periods=1).max() > 0
            recent_bear = bear_cross.astype("float64").rolling(cfg.cross_lookback, min_periods=1).max() > 0
        else:
            recent_bull, recent_bear = bull_cross, bear_cross

        # Causal Chikou rule: today's close vs the close d bars ago (see indicators.py).
        chikou_bull = close > ind["chikou_reference"]
        chikou_bear = close < ind["chikou_reference"]

        long_entry = above_cloud & recent_bull & (tk_diff > 0)
        short_entry = below_cloud & recent_bear & (tk_diff < 0)
        if cfg.require_chikou:
            long_entry &= chikou_bull
            short_entry &= chikou_bear
        if cfg.require_kumo_twist:
            long_entry &= ind["senkou_a_lead"] > ind["senkou_b_lead"]
            short_entry &= ind["senkou_a_lead"] < ind["senkou_b_lead"]
        if not cfg.allow_short:
            short_entry = pd.Series(False, index=ind.index)

        exit_long = below_cloud | bear_cross
        exit_short = above_cloud | bull_cross

        # RSI overlay: avoid chasing overbought rallies, exit when the trend exhausts,
        # and (optionally) only take entries that form an RSI/price divergence.
        rsi = ind["rsi"]
        rsi_x_overbought = (rsi.shift(1) < cfg.rsi_overbought) & (rsi >= cfg.rsi_overbought)
        rsi_x_oversold = (rsi.shift(1) > cfg.rsi_oversold) & (rsi <= cfg.rsi_oversold)
        if cfg.use_rsi_filter:
            not_overbought = (rsi < cfg.rsi_overbought).fillna(False)
            not_oversold = (rsi > cfg.rsi_oversold).fillna(False)
            long_entry &= not_overbought
            short_entry &= not_oversold
            exit_long = exit_long | rsi_x_overbought.fillna(False)
            exit_short = exit_short | rsi_x_oversold.fillna(False)
        if cfg.use_rsi_divergence:
            recent_bull_div = ind["rsi_bull_div"].rolling(cfg.divergence_lookback, min_periods=1).max() > 0
            recent_bear_div = ind["rsi_bear_div"].rolling(cfg.divergence_lookback, min_periods=1).max() > 0
            long_entry &= recent_bull_div
            short_entry &= recent_bear_div

        ind["above_cloud"] = above_cloud
        ind["below_cloud"] = below_cloud
        ind["tk_bull_cross"] = bull_cross
        ind["tk_bear_cross"] = bear_cross
        ind["chikou_bull"] = chikou_bull
        ind["chikou_bear"] = chikou_bear
        ind["rsi_x_overbought"] = rsi_x_overbought
        ind["rsi_x_oversold"] = rsi_x_oversold
        ind["long_entry"] = long_entry
        ind["short_entry"] = short_entry
        ind["exit_long"] = exit_long
        ind["exit_short"] = exit_short
        ind["position"] = self.signals_to_position(long_entry, short_entry, exit_long, exit_short)
        return ind

    @staticmethod
    def signals_to_position(
        long_entry: pd.Series, short_entry: pd.Series, exit_long: pd.Series, exit_short: pd.Series
    ) -> pd.Series:
        """Vectorised position state machine (no Python loop).

        Two independent "latches" are built with forward-fill:
            long_state  := 1 on long_entry, 0 on exit_long or short_entry, else carry forward
            short_state := 1 on short_entry, 0 on exit_short or long_entry, else carry forward
            position    =  long_state - short_state   in {-1, 0, +1}
        Entries are written after exits, so an entry wins on the same bar. A long and a
        short entry can never fire together (price cannot be above and below the cloud).
        """
        le, se = long_entry.to_numpy(bool), short_entry.to_numpy(bool)
        xl, xs = exit_long.to_numpy(bool), exit_short.to_numpy(bool)

        long_events = np.full(le.shape, np.nan)
        long_events[xl | se] = 0.0
        long_events[le] = 1.0
        short_events = np.full(se.shape, np.nan)
        short_events[xs | le] = 0.0
        short_events[se] = 1.0

        long_state = pd.Series(long_events).ffill().fillna(0.0).to_numpy()
        short_state = pd.Series(short_events).ffill().fillna(0.0).to_numpy()
        return pd.Series((long_state - short_state).astype("int8"), index=long_entry.index, name="position")

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
            tenkan=float(row["tenkan"]),
            kijun=float(row["kijun"]),
            cloud_top=float(row["cloud_top"]),
            cloud_bottom=float(row["cloud_bottom"]),
            atr=float(row["atr"]),
            regime=int(row["regime"]) if "regime" in sig.columns else None,
            rsi=float(row["rsi"]) if "rsi" in sig.columns and np.isfinite(row["rsi"]) else None,
        )
