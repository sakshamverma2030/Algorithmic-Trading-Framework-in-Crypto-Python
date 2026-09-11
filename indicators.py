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
from scipy.cluster.vq import kmeans2

from config import ICHIMOKU_PRESETS, IchimokuParams, RegimeMethod, StrategyConfig

try:  # scikit-learn KMeans is used when available for the ML clustering strategy.
    from sklearn.cluster import KMeans  # type: ignore[import-not-found]

    HAS_SKLEARN = True
except Exception:  # noqa: BLE001
    KMeans = None
    HAS_SKLEARN = False

try:  # statsmodels ADF / cointegration tables are used when available.
    from statsmodels.tsa.stattools import adfuller  # type: ignore[import-not-found]

    HAS_STATSMODELS = True
except Exception:  # noqa: BLE001
    adfuller = None
    HAS_STATSMODELS = False

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
# RSI (Wilder) and causal RSI / price divergence
# --------------------------------------------------------------------------------------
def relative_strength_index(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

        RS_t = avg_gain_t / avg_loss_t       (Wilder smoothing of up/down moves)
        RSI_t = 100 - 100 / (1 + RS_t)

    The first valid value appears at index ``period`` (SMA-seeded like Wilder's ATR,
    so the output matches TA-Lib's ``RSI``). Output is NaN until then.
    """
    if HAS_TALIB:
        return pd.Series(
            talib.RSI(close.to_numpy("float64"), timeperiod=period), index=close.index
        )
    delta = close.diff()
    gain, loss = delta.clip(lower=0.0), (-delta).clip(lower=0.0)

    def wilder(window: pd.Series) -> pd.Series:
        if len(window) <= period:
            return pd.Series(np.nan, index=window.index)
        seeded = window.copy()
        seeded.iloc[: period + 1] = np.nan
        seeded.iloc[period] = window.iloc[1: period + 1].mean()
        return seeded.ewm(alpha=1.0 / period, adjust=False, ignore_na=True).mean()

    avg_gain, avg_loss = wilder(gain), wilder(loss)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def rsi_divergence(rsi: pd.Series, close: pd.Series, lookback: int = 10) -> pd.DataFrame:
    """Causal (no look-ahead) RSI / price divergence flags.

    The trailing ``lookback`` window is split into an older half and a recent half
    (each ``L = max(lookback // 2, 2)`` bars). A divergence forms when the recent half
    breaks to a new price extreme but the RSI extreme does not confirm it:

        Bullish  recent price  low   < older price  low     AND  recent RSI  low  > older RSI  low
                (price makes a lower low, RSI makes a higher low = fading the downtrend)
        Bearish  recent price  high  > older price  high    AND  recent RSI  high < older RSI  high

    Every input at bar ``t`` uses only bars ``<= t``, so the flags are safe to feed a
    backtest signal. Returns columns ``rsi_bull_div`` / ``rsi_bear_div``.
    """
    L = max(lookback // 2, 2)

    def med(series: pd.Series, fn: str, window: int) -> pd.Series:
        return getattr(series.rolling(window, min_periods=window), fn)()

    price_low_a, price_low_b = med(close, "min", L).shift(L), med(close, "min", L)
    rsi_low_a, rsi_low_b = med(rsi, "min", L).shift(L), med(rsi, "min", L)
    bull = (price_low_b < price_low_a) & (rsi_low_b > rsi_low_a) & (rsi > rsi_low_b)

    price_high_a, price_high_b = med(close, "max", L).shift(L), med(close, "max", L)
    rsi_high_a, rsi_high_b = med(rsi, "max", L).shift(L), med(rsi, "max", L)
    bear = (price_high_b > price_high_a) & (rsi_high_b < rsi_high_a) & (rsi < rsi_high_b)

    return pd.DataFrame(
        {"rsi_bull_div": bull.fillna(False), "rsi_bear_div": bear.fillna(False)},
        index=rsi.index,
    )


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    """Aroon Up / Aroon Down (Tushar Chande, 1995) - trend-strength oscillator.

        Aroon Up   = 100 * (period - bars since the period-length high) / period
        Aroon Down = 100 * (period - bars since the period-length low)  / period

    Values near 100 signal a persistent new extreme (strong trend); values near 0
    signal prolonged consolidation. ``period`` counts the look-back window plus the
    current bar, so ``period=25`` matches the classic 25-bar Aroon.
    """
    window = period + 1
    high_idx = high.rolling(window=window).apply(lambda w: int(np.argmax(w)), raw=True)
    low_idx = low.rolling(window=window).apply(lambda w: int(np.argmin(w)), raw=True)
    # argmax position ``p`` (0 = oldest, period = newest); bars since the extreme is
    # ``period - p``, so Aroon = 100 * (period - (period - p)) / period = 100 * p / period.
    aroon_up = 100.0 * high_idx.dropna() / period
    aroon_down = 100.0 * low_idx.dropna() / period
    return pd.DataFrame(
        {"aroon_up": aroon_up.reindex(high.index),
         "aroon_down": aroon_down.reindex(low.index)},
        index=high.index,
    )


def hurst_exponent(close: pd.Series, max_lag: int = 100) -> float:
    """Rescaled-range (R/S) Hurst exponent H of a price series.

    For a time series with n points the statistic R/S(n) = (max - min) of the
    cumulative-deviation series (rescaled by its standard deviation) grows as a power
    law R/S ~ (n/2)^H. A least-squares regression of log(R/S) on log(lag) yields H:

        H > 0.5   persistent / trending    (momentum strategies work)
        H = 0.5   random walk
        H < 0.5   mean-reverting           (pairs / reversal strategies work)

    Returns NaN when fewer than 3 usable lags exist.
    """
    x = close.to_numpy(dtype="float64")
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return float("nan")
    lags = range(2, min(max_lag + 1, len(x) // 2))
    tau: list[float] = []
    for lag in lags:
        segments = len(x) // lag
        if segments < 1:
            continue
        s = [x[i * lag:(i + 1) * lag] for i in range(segments)]
        rs = []
        for seg in s:
            mean = seg.mean()
            if mean == 0:
                continue
            dev = seg - mean
            rs.append((np.max(np.cumsum(dev)) - np.min(np.cumsum(dev))) / np.std(seg))
        if rs:
            tau.append(np.mean(rs))
    if len(tau) < 3:
        return float("nan")
    pts = np.log(np.array(list(lags), dtype=float)[:len(tau)]), np.log(np.array(tau))
    slope, _ = np.polyfit(pts[0], pts[1], 1)
    return float(slope)


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0
                    ) -> pd.DataFrame:
    """Bollinger Bands: 2 std-dev channel around a moving average.

        middle = SMA(period)
        upper  = middle + num_std * rolling_std(close, period)
        lower  = middle - num_std * rolling_std(close, period)

    Used by the pairs-trading spread engine (z-score equivalent) and as a
    mean-reversion filter in the momentum-alpha framework.
    """
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"bb_middle": middle, "bb_upper": middle + num_std * std,
         "bb_lower": middle - num_std * std},
        index=close.index,
    )


def adf_test(series: pd.Series, max_lag: int | None = None) -> dict[str, float | int]:
    """Augmented Dickey-Fuller stationarity test (statsmodels backed).

    Tests the null hypothesis that ``series`` has a unit root (is non-stationary).
    A very negative ADF statistic with p-value < 0.05 rejects the null, meaning the
    series is stationary and therefore suitable for mean-reversion trading.

    Returns a dict with keys: adf_stat, p_value, used_lag, n_obs, critical_1/5/10,
    is_stationary. Without statsmodels the function returns NaN statistics.
    """
    out: dict[str, float | int] = {}
    x = series.dropna()
    if len(x) < 8 or not HAS_STATSMODELS:
        out = {"adf_stat": float("nan"), "p_value": float("nan"), "used_lag": -1,
               "n_obs": len(x), "critical_1": float("nan"), "critical_5": float("nan"),
               "critical_10": float("nan"), "is_stationary": False}
        return out
    try:
        stat = adfuller(x, maxlag=max_lag, autolag="AIC", result_object=False)
    except TypeError:  # older statsmodels without the return-type switch
        stat = adfuller(x, maxlag=max_lag, autolag="AIC")
    out = {"adf_stat": float(stat[0]), "p_value": float(stat[1]),
           "used_lag": int(stat[2]), "n_obs": int(stat[3]),
           "critical_1": float(stat[4]["1%"]), "critical_5": float(stat[4]["5%"]),
           "critical_10": float(stat[4]["10%"]),
           "is_stationary": bool(stat[1] < 0.05)}
    return out


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
    Each bar is classified into a regime by one of two strictly trailing methods:

        quantile (default)   rolling percentile rank of NATR over the last ``lookback``
                             bars: LOW (bottom third), NORMAL, HIGH (top third).
        kmeans               causal K-Means clustering (``scipy.cluster.vq.kmeans2``).
                             For every bar ``t`` the last ``lookback`` NATR values are
                             standardised with their own mean/std and split into three
                             clusters (deterministic seed); cluster centres are sorted
                             by volatility, so the lowest-vol cluster becomes LOW etc.

    All three Ichimoku frames are computed once (vectorised), and each output column is
    then chosen per bar with ``np.select``. Regime detection is a pure function of bars
    ``<= t``, so a truncated history reproduces the same labels as the full history.
    When the regime changes, the lines jump between parameter sets, so ``regime_stable``
    marks bars whose regime equals the previous bar's. The strategy ignores TK "crosses"
    created by such a switch.
    """

    LOW, NORMAL, HIGH = 0, 1, 2

    def __init__(
        self,
        regime_params: Mapping[int, IchimokuParams],
        atr_period: int = 14,
        lookback: int = 500,
        q_low: float = 0.33,
        q_high: float = 0.67,
        regime_method: RegimeMethod | str = RegimeMethod.QUANTILE,
        kmeans_clusters: int = 3,
        kmeans_fit_iters: int = 20,
        kmeans_seed: int = 42,
    ) -> None:
        missing = {self.LOW, self.NORMAL, self.HIGH} - set(regime_params)
        if missing:
            raise ValueError(f"regime_params is missing regimes {missing}")
        self.regime_params = dict(regime_params)
        self.atr_period = atr_period
        self.lookback = lookback
        self.q_low = q_low
        self.q_high = q_high
        self.regime_method = (
            regime_method if isinstance(regime_method, RegimeMethod) else RegimeMethod(regime_method)
        )
        self.kmeans_clusters = kmeans_clusters
        self.kmeans_fit_iters = kmeans_fit_iters
        self.kmeans_seed = kmeans_seed

    def volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        natr = average_true_range(df["high"], df["low"], df["close"], self.atr_period) / df["close"]
        return self.classify(natr)

    def classify(self, natr: pd.Series) -> pd.Series:
        """Regime label per bar (0/1/2 = LOW/NORMAL/HIGH). Causal in both modes."""
        if self.regime_method is RegimeMethod.KMEANS:
            return self._kmeans_regime(natr)
        rank = natr.rolling(self.lookback, min_periods=max(20, self.lookback // 4)).rank(pct=True)
        regime = np.select([rank <= self.q_low, rank >= self.q_high], [self.LOW, self.HIGH], default=self.NORMAL)
        return pd.Series(regime.astype("int8"), index=natr.index, name="regime")

    def _kmeans_regime(self, natr: pd.Series) -> pd.Series:
        """Causal per-bar K-Means label: cluster the trailing window, then classify the last bar.

        Deterministic and refit on every bar using only data up to that bar. The cluster
        labels returned by ``kmeans2`` are arbitrary, so the centres are sorted by mean
        NATR and mapped to LOW/NORMAL/HIGH. Bars whose window is still warming up default
        to NORMAL.
        """
        k, lookback = self.kmeans_clusters, self.lookback
        values = natr.to_numpy(dtype="float64")
        out = np.full(values.shape, self.NORMAL, dtype=np.int8)
        start = k * 20  # enough bars for the clusters to be meaningful
        for t in range(start, len(values)):
            lo = max(0, t - lookback + 1)
            window = values[lo: t + 1]
            if not np.all(np.isfinite(window)) or window.std(ddof=0) < 1e-12:
                continue
            x = (window - window.mean()) / window.std(ddof=0)
            code, labels = kmeans2(np.column_stack([x]), k, minit="points", seed=self.kmeans_seed,
                                   iter=self.kmeans_fit_iters)
            order = np.argsort(code[:, 0])  # clusters sorted by volatility centre
            rank = {c: r for r, c in enumerate(order)}
            out[t] = rank[int(labels[-1])]
        return pd.Series(out, index=natr.index, name="regime")

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
            regime_method=cfg.regime_method,
            kmeans_clusters=cfg.kmeans_clusters,
            kmeans_fit_iters=cfg.kmeans_fit_iters,
        )
    return IchimokuCloud(cfg.params)


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """OHLCV + Ichimoku + ATR/NATR + RSI & divergence in one frame (the strategy's feature set)."""
    ichimoku = build_ichimoku(cfg).compute(df)
    atr = average_true_range(df["high"], df["low"], df["close"], cfg.atr_period)
    out = df[["open", "high", "low", "close", "volume"]].join(ichimoku)
    out["atr"] = atr
    out["natr"] = atr / df["close"]
    out["rsi"] = relative_strength_index(df["close"], cfg.rsi_period)
    divergence = rsi_divergence(out["rsi"], df["close"], cfg.divergence_lookback)
    out["rsi_bull_div"] = divergence["rsi_bull_div"]
    out["rsi_bear_div"] = divergence["rsi_bear_div"]
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
