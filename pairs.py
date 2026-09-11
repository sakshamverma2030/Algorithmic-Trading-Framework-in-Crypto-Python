"""
pairs.py - Statistical arbitrage between two cointegrated crypto assets.

The pair is modelled as a single mean-reverting "spread" portfolio:

    x_t = log(base_t)          y_t = log(quote_t)
    beta_t = rolling cov(x, y) / rolling var(x)      (OLS hedge, bar t uses bars <= t)
    spread_t = y_t - beta_t * x_t
    z_t      = (spread_t - rolling_mean) / rolling_std

Trading rule (decided on the close of bar t, executed at t+1):

    z_t < -entry  ->  long the spread   (+1: long x, short beta units of y)
    z_t >  +entry ->  short the spread  (-1)
    |z_t| < exit  ->  close (mean reversion completed)
    z beyond -/+stop -> protective stop; max_hold -> forced close

Accounting (documented simplification - standard for residuals-based models):
the strategy return on bar t uses the hedge ratio known at t-1,

    return_t = w_t * (dx_t - beta_{t-1} * dy_t)   w_t in {-1, 0, +1}

compounded into the equity curve, with a two-leg transaction cost applied to
every change of w_t. The result is a normal ``BacktestResult``, so analytics,
charts and report plumbing work unchanged.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from backtester import (
    TRADE_COLUMNS,
    BacktestError,
    BacktestResult,
    _finalise_curve,
    _metadata,
)
from config import AppConfig, PairsConfig
from risk import CostModel, ExitReason

logger = logging.getLogger(__name__)


class CointegrationAnalyzer:
    """Rolling-OLS cointegration summary statistics for a ``(base, quote)`` pair."""

    @staticmethod
    def aligned(base: pd.DataFrame, quote: pd.DataFrame) -> pd.DataFrame:
        """Inner join of the two symbol frames on their datetime index."""
        joined = base.join(quote, lsuffix="_base", rsuffix="_quote", how="inner")
        return joined.dropna(subset=["close_base", "close_quote"], how="any")

    @staticmethod
    def beta_series(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Rolling OLS slope of ``y`` on ``x``  = cov(x, y) / var(x)."""
        return x.rolling(window, min_periods=window).cov(y) / x.rolling(window, min_periods=window).var()

    @staticmethod
    def spread_series(x: pd.Series, y: pd.Series, beta: pd.Series) -> pd.Series:
        return y - beta * x

    @staticmethod
    def zscore(spread: pd.Series, window: int) -> pd.Series:
        return (spread - spread.rolling(window, min_periods=window).mean()) / spread.rolling(
            window, min_periods=window
        ).std()

    @staticmethod
    def half_life(spread: pd.Series) -> float:
        """OLS half-life of the (full-sample) stationary residual, in bars.

        Fits dS_t = c + gamma * S_{t-1}; if gamma < 0 the residual is mean-reverting
        with half-life ln(2) / -gamma (diagnostic only - never used as a signal).
        """
        s = spread.dropna()
        if len(s) < 10:
            return float("nan")
        y = s.diff().to_numpy()[1:]
        x_front = s.to_numpy()[:-1]
        denom = float(np.sum((x_front - x_front.mean()) ** 2))
        if denom <= 0:
            return math.inf
        gamma = float(np.sum((x_front - x_front.mean()) * (y - y.mean())) / denom)
        return math.log(2.0) / -gamma if gamma < 0 else math.inf

    @staticmethod
    def diagnostics(base: pd.DataFrame, quote: pd.DataFrame, window: int) -> dict[str, Any]:
        data = CointegrationAnalyzer.aligned(base, quote)
        ret = data["close_base"].pct_change().dropna(), data["close_quote"].pct_change().dropna()
        x, y = np.log(data["close_base"]), np.log(data["close_quote"])
        beta = CointegrationAnalyzer.beta_series(x, y, window)
        spread = CointegrationAnalyzer.spread_series(x, y, beta)
        local_beta = beta.dropna()
        return {
            "bars": len(data),
            "return_correlation": float(pd.concat([ret[0], ret[1]], axis=1).corr().iloc[0, 1]),
            "hedge_beta_mean": float(local_beta.mean()) if len(local_beta) else float("nan"),
            "hedge_beta_std": float(local_beta.std()) if len(local_beta) else float("nan"),
            "spread_mean": float(spread.mean()),
            "spread_std": float(spread.std()),
            "half_life_bars": CointegrationAnalyzer.half_life(spread),
        }


@dataclass
class _PairsRun:
    """Pre-computed series for one pairs backtest (all causal)."""

    index: pd.DatetimeIndex
    x: np.ndarray          # log / raw base price
    y: np.ndarray          # log / raw quote price
    beta: np.ndarray       # rolling hedge ratio (bar t uses bars <= t)
    spread: np.ndarray     # residual y - beta * x
    z: np.ndarray          # rolling z-score of the residual
    corr: np.ndarray       # rolling log-return correlation (for the cointegration filter)
    sr: np.ndarray         # spread-portfolio return: dx_t - beta_{t-1} * dy_t


class PairsBacktester:
    """Bar-by-bar statistical-arbitrage backtest returning a standard BacktestResult."""

    def __init__(self, cfg: AppConfig, pairs: PairsConfig | None = None) -> None:
        self.cfg = cfg
        self.pairs = replace(cfg.pairs) if pairs is None else replace(cfg.pairs)
        self.cost_model = CostModel(cfg.costs)

    @property
    def name(self) -> str:
        return self.pairs.label

    @property
    def warmup_bars(self) -> int:
        return max(self.pairs.lookback, self.pairs.atr_period) + 2

    def prepare(self, base: pd.DataFrame, quote: pd.DataFrame) -> _PairsRun:
        """Align both symbols and compute the causal spread / z-score / returns."""
        p = self.pairs
        data = CointegrationAnalyzer.aligned(base, quote)
        if len(data) < p.lookback + 20:
            raise BacktestError(f"Pairs backtest needs at least {p.lookback + 20} aligned bars, got {len(data)}")
        c1, c2 = data["close_base"], data["close_quote"]
        if p.use_log_prices:
            x, y = np.log(c1), np.log(c2)
        else:
            x, y = c1.astype(float), c2.astype(float)

        beta = CointegrationAnalyzer.beta_series(x, y, p.lookback)
        spread = CointegrationAnalyzer.spread_series(x, y, beta)
        z = CointegrationAnalyzer.zscore(spread, p.lookback)
        dx, dy = x.diff(), y.diff()
        corr = dx.rolling(p.lookback, min_periods=p.lookback).corr(dy)
        # Weights used at bar t are the ratios known at t-1.
        sr = (dx - beta.shift(1) * dy).to_numpy("float64")

        return _PairsRun(data.index, x.to_numpy("float64"), y.to_numpy("float64"),
                         beta.to_numpy("float64"), spread.to_numpy("float64"),
                         z.to_numpy("float64"), corr.to_numpy("float64"), sr)

    def run(self, base: pd.DataFrame, quote: pd.DataFrame, name: str | None = None) -> BacktestResult:
        run = self.prepare(base, quote)
        p, idx = self.pairs, run.index
        n, capital = len(idx), self.cfg.risk.initial_capital
        leg_cost = 2.0 * self.cost_model.one_way_cost  # entering/leaving a spread pays two legs

        equity = np.empty(n)
        pos = np.zeros(n, dtype=np.int8)
        equity[0] = capital
        w, entry_bar, eq_entry = 0, None, capital
        block_until = 0  # bars after a losing exit during which new entries are skipped
        trades: list[dict[str, Any]] = []

        def close_trade(bar: int, side: int, reason: str) -> None:
            pnl = equity[bar] - eq_entry
            trades.append({
                "trade_id": len(trades) + 1, "side": side,
                "entry_time": idx[entry_bar], "exit_time": idx[bar],
                "entry_price": run.spread[max(entry_bar - 1, 0)], "exit_price": run.spread[max(bar - 1, 0)],
                "quantity": float("nan"), "gross_pnl": pnl, "fees": float("nan"), "financing": 0.0,
                "net_pnl": pnl, "return_pct": pnl / eq_entry if eq_entry > 0 else float("nan"),
                "r_multiple": float("nan"), "bars_held": bar - entry_bar, "exit_reason": reason,
            })

        for i in range(1, n):
            z_prev = run.z[i - 1]          # signal from the previous close (no look-ahead)
            new_w, reason = w, None
            if w != 0:                     # manage the open position first
                if abs(z_prev) < p.exit_zscore:
                    reason = ExitReason.SIGNAL.value
                elif (w > 0 and z_prev < -p.stop_zscore) or (w < 0 and z_prev > p.stop_zscore):
                    reason = ExitReason.STOP_LOSS.value
                elif i - entry_bar >= p.max_hold_bars:
                    reason = ExitReason.TIME_STOP.value
                if reason is not None:
                    new_w = 0
            elif np.isfinite(z_prev) and i >= block_until:
                corr_ok = np.isfinite(run.corr[i - 1]) and run.corr[i - 1] >= p.min_correlation
                if z_prev <= -p.entry_zscore and corr_ok:
                    new_w, reason = 1, None
                elif z_prev >= p.entry_zscore and corr_ok:
                    new_w, reason = -1, None

            sr = run.sr[i] if np.isfinite(run.sr[i]) else 0.0
            ret = new_w * sr - leg_cost * abs(new_w - w)
            equity[i] = equity[i - 1] * (1.0 + ret)
            pos[i] = new_w

            if reason is not None:
                close_trade(i, int(w), reason)   # w is the side being closed
                if reason in (ExitReason.STOP_LOSS.value, ExitReason.TIME_STOP.value):
                    block_until = i + max(p.lookback // 2, 1)
            elif new_w != w:
                entry_bar, eq_entry = i, equity[i - 1]
            w = new_w

        if w != 0:  # an open spread at the end of the sample is liquidated at the final close
            equity[-1] *= 1.0 - leg_cost
            close_trade(n - 1, int(w), ExitReason.END_OF_DATA.value)
            pos[-1] = 0

        curve = pd.DataFrame(
            {"equity": equity, "cash": np.nan, "position": pos, "quantity": np.nan,
             "close": run.y, "halted": False},
            index=idx,
        )
        curve = _finalise_curve(curve, capital, self.cost_model.one_way_cost)
        trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS) if trades else pd.DataFrame(columns=TRADE_COLUMNS)

        diag = CointegrationAnalyzer.diagnostics(base, quote, p.lookback)
        metadata = _metadata(self.cfg, name or self.name, "pairs", self.name, 0.0, 0.0, 0.0, 0)
        metadata["pairs"] = {
            "base": p.base_symbol, "quote": p.quote_symbol, "lookback": p.lookback,
            "entry_zscore": p.entry_zscore, "exit_zscore": p.exit_zscore, "stop_zscore": p.stop_zscore,
            "max_hold_bars": p.max_hold_bars, "use_log_prices": p.use_log_prices,
            "annualised_correlation": diag["return_correlation"],
            "hedge_beta_mean": diag["hedge_beta_mean"], "hedge_beta_std": diag["hedge_beta_std"],
            "spread_half_life_bars": diag["half_life_bars"],
        }

        signals = pd.DataFrame(
            {"close": run.y, "base": run.x, "quote": run.y, "hedge": run.beta, "spread": run.spread,
             "z": run.z, "position": pos},
            index=idx,
        )
        logger.info("Pairs '%s': %d trades, final equity %.2f", metadata["name"], len(trades_df), equity[-1])
        return BacktestResult(metadata["name"], curve, trades_df, signals, metadata)


def check_pair(base: pd.DataFrame, quote: pd.DataFrame, pairs: PairsConfig) -> dict[str, Any]:
    """Quick pair fitness snapshot (correlation, hedge ratio, half-life) without a backtest."""
    return CointegrationAnalyzer.diagnostics(base, quote, pairs.lookback)