"""
indicators.py - Vectorised technical indicators for the Ichimoku strategy.

Implements the complete five-line Ichimoku Kinko Hyo ("one-glance equilibrium chart")
plus the Kumo, Wilder's Average True Range, and a volatility-regime-adaptive Ichimoku
for crypto-tuned dynamic parameters.

Ichimoku definitions (HH_n / LL_n = highest high / lowest low of the last n bars):

    Tenkan-sen  (conversion line)   T_t  = (HH_9  + LL_9 ) / 2
    Kijun-sen   (base line)         K_t  = (HH_26 + LL_26) / 2
    Senkou A    (leading span A)    A*_t = (T_t + K_t) / 2           plotted 26 bars ahead
    Senkou B    (leading span B)    B*_t = (HH_52 + LL_52) / 2       plotted 26 bars ahead
    Chikou      (lagging span)      Close_t                          plotted 26 bars behind

Each line is a Donchian-channel midpoint, i.e. the "equilibrium" price of its
look-back window, rather than an average of closes.

Look-ahead bias (the main trap when backtesting Ichimoku)
--------------------------------------------------------
* The Kumo visible on the chart at bar t was computed 26 bars earlier:
      A_t = A*_{t-26},  B_t = B*_{t-26}   ->   ``senkou_a = senkou_a_lead.shift(26)``
  Comparing price with the *un-shifted* spans would test against a cloud that
  does not exist yet on the trader's screen.
* The Chikou span drawn at bar t is ``Close_{t+26}``, which is future data. We keep it
  (``chikou_span``) for plotting only. The causal equivalent of "Chikou above price"
  compares today's close with the close 26 bars ago: ``Close_t > Close_{t-26}``.
  That reference price is exposed as ``chikou_reference``.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

from config import ICHIMOKU_PRESETS, IchimokuParams, StrategyConfig

logger = logging.getLogger(__name__)

try:  # TA-Lib is optional: C-speed rolling extremes and ATR; identical results otherwise.
    import talib  # type: ignore[import-not-found]

    HAS_TALIB = True
except Exception:  # noqa: BLE001 - ImportError, or OSError when the C library is missing
    talib = None
    HAS_TALIB = False


# --------------------------------------------------------------------------------------
# Primitive building blocks
# --------------------------------------------------------------------------------------
def rolling_max(series: pd.Series, window: int) -> pd.Series:
    """Highest value of the trailing ``window`` bars (NaN until the window is full)."""
    if HAS_TALIB:
        return pd.Series(talib.MAX(series.to_numpy(dtype="float64"), timeperiod=window), index=series.index)
    return series.rolling(window, min_periods=window).max()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    """Lowest value of the trailing ``window`` bars (NaN until the window is full)."""
    if HAS_TALIB:
        return pd.Series(talib.MIN(series.to_numpy(dtype="float64"), timeperiod=window), index=series.index)
    return series.rolling(window, min_periods=window).min()


def donchian_midpoint(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Equilibrium price of the last ``window`` bars: (HH_n + LL_n) / 2."""
    return (rolling_max(high, window) + rolling_min(low, window)) / 2.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|). Undefined on the first bar."""
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1, skipna=False)


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ATR, the volatility unit used for stops, sizing and regime detection.

        ATR_n     = mean(TR_1 .. TR_n)                          (SMA seed)
        ATR_t     = ATR_{t-1} + (TR_t - ATR_{t-1}) / n          (Wilder smoothing)

    Wilder smoothing is an EMA with alpha = 1/n, so after seeding we can delegate the
    recursion to pandas' vectorised ``ewm(adjust=False)``. The output matches TA-Lib.
    """
    if HAS_TALIB:
        return pd.Series(
            talib.ATR(high.to_numpy("float64"), low.to_numpy("float64"), close.to_numpy("float64"), timeperiod=period),
            index=close.index,
        )
    tr = true_range(high, low, close)
    if len(tr) <= period:
        return pd.Series(np.nan, index=close.index)
    seeded = tr.copy()
    seeded.iloc[: period + 1] = np.nan
    seeded.iloc[period] = tr.iloc[1: period + 1].mean()
    return seeded.ewm(alpha=1.0 / period, adjust=False, ignore_na=True).mean()


# --------------------------------------------------------------------------------------
# Ichimoku Kinko Hyo
# --------------------------------------------------------------------------------------
class IchimokuCloud:
    """Vectorised Ichimoku calculator for one parameter set."""

    def __init__(self, params: IchimokuParams) -> None:
        self.params = params

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return every Ichimoku component aligned to ``df.index``.

        Columns
        -------
        tenkan, kijun            conversion / base lines
        senkou_a_lead, _b_lead   spans as computed at t (they belong to bar t + d)
        senkou_a, senkou_b       the Kumo that is visible at bar t (lead shifted by +d)
        cloud_top, cloud_bottom  max / min of the visible spans
        cloud_bullish            Span A above Span B (green Kumo)
        chikou_span              Close_{t+d}: plot-only, contains future data
        chikou_reference         Close_{t-d}: causal comparison price for the Chikou rule
        displacement             d, kept per row so mixed-parameter frames stay plottable
        """
        p = self.params
        high, low, close = df["high"], df["low"], df["close"]

        tenkan = donchian_midpoint(high, low, p.tenkan)
        kijun = donchian_midpoint(high, low, p.kijun)
        span_a_lead = (tenkan + kijun) / 2.0
        span_b_lead = donchian_midpoint(high, low, p.senkou_b)
        span_a = span_a_lead.shift(p.displacement)   # the cloud plotted d bars ahead ...
        span_b = span_b_lead.shift(p.displacement)   # ... is what the chart shows at t

        cloud_top = np.maximum(span_a, span_b)       # NaN-propagating on purpose
        cloud_bottom = np.minimum(span_a, span_b)
        out = pd.DataFrame(
            {
                "tenkan": tenkan,
                "kijun": kijun,
                "senkou_a_lead": span_a_lead,
                "senkou_b_lead": span_b_lead,
                "senkou_a": span_a,
                "senkou_b": span_b,
                "cloud_top": cloud_top,
                "cloud_bottom": cloud_bottom,
                "chikou_span": close.shift(-p.displacement),
                "chikou_reference": close.shift(p.displacement),
                "displacement": p.displacement,
            },
            index=df.index,
        )
        out["cloud_bullish"] = out["senkou_a"] > out["senkou_b"]
        out["cloud_thickness"] = (out["cloud_top"] - out["cloud_bottom"]) / close
        return out


class AdaptiveIchimoku:
    """Volatility-regime-switching Ichimoku ("crypto-tuned dynamic parameters").

    Normalised ATR  NATR_t = ATR_t / Close_t  measures volatility per unit of price.
    Its rolling percentile rank over the last ``lookback`` bars (strictly trailing,
    so causal) classifies each bar into a regime:

        rank <= q_low   LOW  volatility  ->  fast lines (e.g. 9/26/52): react early to
                                             breakouts from volatility compression
        rank >= q_high  HIGH volatility  ->  slow lines (e.g. 20/60/120): filter the
                                             whipsaws of noisy, fast markets
        otherwise       NORMAL           ->  24/7-calendar lines (10/30/60)

    All three Ichimoku frames are computed once (vectorised), and each output column is
    then chosen per bar with ``np.select``. When the regime changes, the lines jump
    between parameter sets, so ``regime_stable`` marks bars whose regime equals the
    previous bar's. The strategy ignores TK "crosses" created by such a switch.
    """

    LOW, NORMAL, HIGH = 0, 1, 2

    def __init__(
        self,
        regime_params: Mapping[int, IchimokuParams],
        atr_period: int = 14,
        lookback: int = 500,
        q_low: float = 0.33,
        q_high: float = 0.67,
    ) -> None:
        missing = {self.LOW, self.NORMAL, self.HIGH} - set(regime_params)
        if missing:
            raise ValueError(f"regime_params is missing regimes {missing}")
        self.regime_params = dict(regime_params)
        self.atr_period = atr_period
        self.lookback = lookback
        self.q_low = q_low
        self.q_high = q_high

    def volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        natr = average_true_range(df["high"], df["low"], df["close"], self.atr_period) / df["close"]
        rank = natr.rolling(self.lookback, min_periods=max(20, self.lookback // 4)).rank(pct=True)
        regime = np.select([rank <= self.q_low, rank >= self.q_high], [self.LOW, self.HIGH], default=self.NORMAL)
        return pd.Series(regime.astype("int8"), index=df.index, name="regime")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        regime = self.volatility_regime(df)
        frames = {r: IchimokuCloud(p).compute(df) for r, p in self.regime_params.items()}
        base = frames[self.NORMAL]
        r = regime.to_numpy()
        conditions = [r == self.LOW, r == self.HIGH]
        out = pd.DataFrame(
            {
                col: np.select(
                    conditions,
                    [frames[self.LOW][col].to_numpy(dtype="float64"), frames[self.HIGH][col].to_numpy(dtype="float64")],
                    default=base[col].to_numpy(dtype="float64"),
                )
                for col in base.columns
                if col != "cloud_bullish"
            },
            index=df.index,
        )
        out["displacement"] = out["displacement"].astype("int64")
        out["cloud_bullish"] = out["senkou_a"] > out["senkou_b"]
        out["regime"] = regime
        out["regime_stable"] = regime.eq(regime.shift(1))
        return out


# --------------------------------------------------------------------------------------
# Convenience API
# --------------------------------------------------------------------------------------
def build_ichimoku(cfg: StrategyConfig) -> IchimokuCloud | AdaptiveIchimoku:
    if cfg.dynamic:
        low, normal, high = (ICHIMOKU_PRESETS[name] for name in cfg.regime_presets)
        return AdaptiveIchimoku(
            {AdaptiveIchimoku.LOW: low, AdaptiveIchimoku.NORMAL: normal, AdaptiveIchimoku.HIGH: high},
            atr_period=cfg.atr_period,
            lookback=cfg.regime_lookback,
            q_low=cfg.regime_low_quantile,
            q_high=cfg.regime_high_quantile,
        )
    return IchimokuCloud(cfg.params)


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """OHLCV + Ichimoku + ATR / NATR in one frame (the strategy's feature set)."""
    ichimoku = build_ichimoku(cfg).compute(df)
    atr = average_true_range(df["high"], df["low"], df["close"], cfg.atr_period)
    out = df[["open", "high", "low", "close", "volume"]].join(ichimoku)
    out["atr"] = atr
    out["natr"] = atr / df["close"]
    return out


def projected_cloud(indicators: pd.DataFrame) -> pd.DataFrame:
    """The Kumo drawn *beyond* the last bar (its next ``displacement`` future bars).

    The lead value computed at bar n-d+k-1 belongs to future bar n+k (k = 1..d), so the
    last d lead values, re-indexed onto future timestamps, form the projected cloud.
    """
    if len(indicators) < 2:
        return pd.DataFrame(columns=["senkou_a", "senkou_b"])
    d = int(indicators["displacement"].iloc[-1])
    step = indicators.index[-1] - indicators.index[-2]
    future = pd.date_range(indicators.index[-1] + step, periods=d, freq=step)
    return pd.DataFrame(
        {
            "senkou_a": indicators["senkou_a_lead"].iloc[-d:].to_numpy(),
            "senkou_b": indicators["senkou_b_lead"].iloc[-d:].to_numpy(),
        },
        index=future,
    )
