"""
intermediate_strategies.py - "Intermediate" module.

Implements the three intermediate strategy families on top of the framework's shared
signal-frame contract (``long_entry`` / ``short_entry`` / ``exit_long`` / ``exit_short`` /
``position`` + ``atr`` / ``tenkan`` / ``kijun`` / ``cloud_top`` / ``cloud_bottom``), so
``VectorizedBacktester`` and ``EventDrivenBacktester`` run them unchanged.

    1. IchimokuStrategy          - full five-line Ichimoku system (re-exported from
                                   ``strategy.py``: Kumo breakout, TK cross, trend dir).
    2. CalendarAnomalyStrategy   - hourly / day-of-week seasonal return profiles.
    3. AroonRsiDivergenceStrategy- Aroon Up/Down + Aroon(RSI) divergence setups.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import CalendarConfig, DivergenceConfig
from indicators import aroon, relative_strength_index
from strategy import IchimokuStrategy, SignalSnapshot

__all__ = ["IchimokuStrategy", "CalendarAnomalyStrategy", "AroonRsiDivergenceStrategy"]

logger = logging.getLogger(__name__)


class CalendarAnomalyStrategy:
    """Trade the recurring seasonality of crypto returns.

    Crypto markets are open 24/7/365, which produces measurable time-of-week and
    hour-of-day return regularities (protocol settlement cycles, retail trading
    hours, US/EU/Asia time-zone asymmetries, weekend liquidity dry-ups).

    For every bar the *causal* average return of its calendar bucket (UTC weekday
    and/or UTC hour) is computed over all earlier bars with at least ``min_obs``
    samples. The strategy is long when the bucket profile is positive and flat (or
    short, when ``allow_short``) otherwise. All features derive from the candle's
    open timestamp, so there is no look-ahead bias.
    """

    def __init__(self, cfg: CalendarConfig) -> None:
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.label

    @property
    def allow_short(self) -> bool:
        return self.cfg.allow_short

    @property
    def warmup_bars(self) -> int:
        """Bars needed before the first bucket profile becomes statistically valid."""
        return self.cfg.min_obs * 2 + 20

    @property
    def history_bars(self) -> int:
        return self.warmup_bars * 4

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        close = df["close"]
        atr = _rolling_atr(df)

        # --- causal calendar profiles ------------------------------------------
        # profile[bucket] = expanding mean of all earlier bars in the same bucket.
        ret = close.pct_change()
        profile = pd.Series(0.0, index=df.index)
        counts = pd.Series(0, index=df.index)
        for _, buckets in self._buckets(df.index):
            for value in buckets["keys"]:
                mask = (buckets["series"] == value).to_numpy()
                cum_mean = pd.Series(
                    np.where(mask, ret.where(mask).expanding(min_periods=c.min_obs).mean().fillna(0.0), 0.0),
                    index=df.index,
                )
                cum_count = pd.Series(
                    np.where(mask, np.cumsum(mask), 0),
                    index=df.index,
                )
                profile = profile.where(~mask, cum_mean)
                counts = counts.where(~mask, cum_count)
        profile_valid = counts >= c.min_obs

        long_entry = profile_valid & (profile > 0.0)
        exit_long = profile_valid & (profile <= 0.0)
        if c.allow_short:
            short_entry = profile_valid & (profile < 0.0)
            exit_short = profile_valid & (profile >= 0.0)
        else:
            short_entry = pd.Series(False, index=df.index)
            exit_short = pd.Series(False, index=df.index)

        out = df[["open", "high", "low", "close", "volume"]].copy()
        out["atr"] = atr
        out["natr"] = atr / close
        out["tenkan"] = close.rolling(9).mean()
        out["kijun"] = close.rolling(26).mean()
        out["cloud_top"] = close.rolling(26).max().rolling(26).mean().fillna(out["kijun"])
        out["cloud_bottom"] = close.rolling(26).min().rolling(26).mean().fillna(out["kijun"])
        out["seasonal_profile"] = profile
        out["seasonal_samples"] = counts
        out["long_entry"] = long_entry.fillna(False)
        out["short_entry"] = short_entry.fillna(False)
        out["exit_long"] = exit_long.fillna(False)
        out["exit_short"] = exit_short.fillna(False)
        out["position"] = IchimokuStrategy.signals_to_position(
            out["long_entry"], out["short_entry"], out["exit_long"], out["exit_short"]
        )
        return out

    def _buckets(self, index: pd.DatetimeIndex) -> list[tuple[str, dict[str, pd.Series]]]:
        c = self.cfg
        out: list[tuple[str, dict[str, pd.Series]]] = []
        if c.dayofweek:
            keys = pd.Series(index.dayofweek, index=index)
            out.append(("dayofweek", {"keys": pd.Index(keys.unique()).sort_values(), "series": keys}))
        if c.hour:
            keys = pd.Series(index.hour, index=index)
            out.append(("hour", {"keys": pd.Index(keys.unique()).sort_values(), "series": keys}))
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
            rsi=None,
        )


def _rolling_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Internal ATR fallback (period-14 Wilder) so calendar/hurst modules stay
    self-contained even before the rest of the indicator suite is imported."""
    from indicators import average_true_range

    return average_true_range(df["high"], df["low"], df["close"], period)


class AroonRsiDivergenceStrategy:
    """Aroon trend-strength + RSI divergence strategy.

    * ``Aroon Up/Down`` (Chande, 1995) quantify how recently the rolling high / low
      was set - a strong trend shows one line pinned near 100 and the other near 0.
    * ``Aroon(RSI)`` applies the same Aroon logic to the RSI oscillator itself.
      Applying an oscillator to an oscillator *detrends* the momentum signal and
      exposes its turning points.

    Divergence setup: in a trailing window the price makes a lower low while the
    Aroon-RSI makes a higher low (bullish) - momentum refuses to confirm the drop -
    gated by a minimum Aroon-Up/Aroon-Down spread so only directional, liquid moves
    are traded. All inputs at bar t use bars <= t.
    """

    def __init__(self, cfg: DivergenceConfig) -> None:
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.label

    @property
    def allow_short(self) -> bool:
        return self.cfg.allow_short

    @property
    def warmup_bars(self) -> int:
        c = self.cfg
        return max(c.aroon_period, c.rsi_period) + c.divergence_lookback + 2

    @property
    def history_bars(self) -> int:
        return self.warmup_bars + 20 * self.cfg.rsi_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        close, high, low = df["close"], df["high"], df["low"]
        atr = _rolling_atr(df, 14)
        rsi = relative_strength_index(close, c.rsi_period)
        ar = aroon(high, low, c.aroon_period)
        # Aroon of the RSI oscillator: re-apply the Aroon window to RSI itself.
        ar_rsi = aroon(rsi.fillna(0.0), rsi.fillna(0.0), c.aroon_period)
        spread_ar = ar["aroon_up"] - ar["aroon_down"]

        L = max(c.divergence_lookback // 2, 2)

        def med(series: pd.Series, fn: str, window: int) -> pd.Series:
            return getattr(series.rolling(window, min_periods=window), fn)()

        # Recent-half vs older-half price lows and momentum lows (causal).
        price_low_a, price_low_b = med(close, "min", L).shift(L), med(close, "min", L)
        mom_low_a, mom_low_b = med(ar_rsi["aroon_up"], "min", L).shift(L), med(ar_rsi["aroon_up"], "min", L)
        bull = (price_low_b < price_low_a) & (mom_low_b > mom_low_a) & (spread_ar > -c.min_aroon_spread)

        price_high_a, price_high_b = med(close, "max", L).shift(L), med(close, "max", L)
        mom_high_a, mom_high_b = med(ar_rsi["aroon_down"], "max", L).shift(L), med(ar_rsi["aroon_down"], "max", L)
        bear = (price_high_b > price_high_a) & (mom_high_b < mom_high_a) & (spread_ar < c.min_aroon_spread)

        # Entry requires an explicit divergence (unless disabled) and RSI agreement.
        rsi_ok_bull = rsi < 50.0
        rsi_ok_bear = rsi > 50.0
        long_entry = (bull if c.require_divergence else spread_ar > 0) & rsi_ok_bull.fillna(False)
        if c.allow_short:
            short_entry = (bear if c.require_divergence else spread_ar < 0) & rsi_ok_bear.fillna(False)
            exit_short = med(ar_rsi["aroon_down"], "min", L).fillna(0.0) <= 20.0
        else:
            short_entry = pd.Series(False, index=df.index)
            exit_short = pd.Series(False, index=df.index)
        exit_long = (ar_rsi["aroon_up"] < 50.0).fillna(True)  # momentum oscillator weakens

        out = df[["open", "high", "low", "close", "volume"]].copy()
        out["atr"] = atr
        out["natr"] = atr / close
        out["tenkan"] = close.rolling(c.aroon_period).mean()
        out["kijun"] = close.rolling(c.aroon_period * 2).mean()
        out["cloud_top"] = high.rolling(c.aroon_period).max()
        out["cloud_bottom"] = low.rolling(c.aroon_period).min()
        out["rsi"] = rsi
        out["aroon_up"] = ar["aroon_up"]
        out["aroon_down"] = ar["aroon_down"]
        out["aroon_rsi"] = ar_rsi["aroon_up"]
        out["aroon_spread"] = spread_ar
        out["bull_divergence"] = bull.fillna(False)
        out["bear_divergence"] = bear.fillna(False)
        out["long_entry"] = long_entry.fillna(False)
        out["short_entry"] = short_entry.fillna(False)
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