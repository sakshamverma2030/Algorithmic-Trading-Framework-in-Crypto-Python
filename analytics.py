"""
analytics.py - Performance metrics and Matplotlib visualisation.

Metric definitions (r_t = per-bar simple return, N = bars per year, E = equity):

    Total return            E_T / E_0 - 1
    CAGR                    (E_T / E_0)^(N / T) - 1
    Annualised volatility   sigma(r) * sqrt(N)
    Sharpe ratio            mean(r - r_f) / sigma(r - r_f) * sqrt(N)
    Sortino ratio           mean(r - r_f) / sqrt(mean(min(r - r_f, 0)^2)) * sqrt(N)
                            (penalises downside deviation only)
    Drawdown                DD_t = E_t / max_{s<=t} E_s - 1 ;  MaxDD = min_t DD_t
    Calmar ratio            CAGR / |MaxDD|
    Probabilistic Sharpe    PSR = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )
                            Bailey & Lopez de Prado (2012): the probability that the true Sharpe
                            exceeds SR*, corrected for skewness g3 and kurtosis g4 (fat tails)
    Historical VaR / CVaR   -q_5%(r)  /  -E[r | r <= q_5%(r)]
    Alpha / beta            OLS  r^s_t = alpha + beta * r^b_t + e_t  (alpha annualised as alpha * N)
    Information ratio       mean(r^s - r^b) / sigma(r^s - r^b) * sqrt(N)
    Profit factor           gross profit / gross loss     (> 1 means the strategy makes money)
    Payoff ratio            average win / |average loss|
    Expectancy              average net PnL per trade

Charts follow a restrained, colour-blind-safe palette: blue = strategy / bullish,
orange = benchmark, red = bearish, neutral greys for price candles. Buy and sell
markers differ by shape as well as colour.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats

from indicators import projected_cloud

if TYPE_CHECKING:  # imported for type hints only, which keeps the dependency graph acyclic
    from backtester import BacktestResult, OptimizationResult

logger = logging.getLogger(__name__)


# ======================================================================================
# Pure metric functions
# ======================================================================================
def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()


def _rf_per_period(rf_annual: float, ppy: float) -> float:
    """Convert an annual risk-free rate into the equivalent per-bar rate."""
    return (1.0 + rf_annual) ** (1.0 / ppy) - 1.0


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else float("nan")


def annualized_return(equity: pd.Series, ppy: float) -> float:
    """Compound annual growth rate (geometric, so volatility drag is included)."""
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (ppy / n) - 1.0) if growth > 0 else -1.0


def annualized_volatility(returns: pd.Series, ppy: float) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * math.sqrt(ppy)) if len(r) > 1 else float("nan")


def sharpe_ratio(returns: pd.Series, ppy: float, rf: float = 0.0) -> float:
    r = _clean(returns) - _rf_per_period(rf, ppy)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(ppy)) if len(r) > 1 and sd > 0 else float("nan")


def sortino_ratio(returns: pd.Series, ppy: float, rf: float = 0.0) -> float:
    r = _clean(returns) - _rf_per_period(rf, ppy)
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)) if len(r) else 0.0
    return float(r.mean() / downside * math.sqrt(ppy)) if downside > 0 else float("nan")


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min()) if len(equity) else float("nan")


def calmar_ratio(equity: pd.Series, ppy: float) -> float:
    mdd = max_drawdown(equity)
    return annualized_return(equity, ppy) / abs(mdd) if mdd < 0 else float("nan")


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float
    peak: pd.Timestamp | None
    trough: pd.Timestamp | None
    recovery: pd.Timestamp | None          # None = not recovered by the end of the sample
    longest_duration: pd.Timedelta          # longest peak-to-recovery spell
    longest_duration_bars: int
    current_duration: pd.Timedelta          # time underwater at the end of the sample


def drawdown_stats(equity: pd.Series) -> DrawdownStats:
    """Depth and duration of drawdowns (the duration is often more painful than the depth)."""
    zero = pd.Timedelta(0)
    if len(equity) < 2:
        return DrawdownStats(0.0, None, None, None, zero, 0, zero)
    dd = drawdown_series(equity)
    idx = equity.index
    mdd = float(dd.min())
    trough = dd.idxmin() if mdd < 0 else None
    peak = equity.loc[:trough].idxmax() if trough is not None else None
    recovery = None
    if trough is not None:
        after = equity.loc[trough:]
        recovered = after[after >= equity.loc[peak]]
        recovery = recovered.index[0] if len(recovered) else None

    underwater = dd.to_numpy() < 0
    if not underwater.any():
        return DrawdownStats(mdd, peak, trough, recovery, zero, 0, zero)
    starts = underwater & ~np.r_[False, underwater[:-1]]
    run_id = np.cumsum(starts)
    run_id[~underwater] = 0
    counts = np.bincount(run_id)[1:]
    k = int(np.argmax(counts)) + 1
    positions = np.flatnonzero(run_id == k)
    first, last = positions[0], positions[-1]
    spell_start = idx[max(first - 1, 0)]                          # the peak before the spell
    spell_end = idx[last + 1] if last + 1 < len(idx) else idx[last]  # recovery bar (or sample end)
    current = zero
    if underwater[-1]:
        last_run = np.flatnonzero(run_id == run_id[-1])
        current = idx[-1] - idx[max(last_run[0] - 1, 0)]
    return DrawdownStats(mdd, peak, trough, recovery, spell_end - spell_start, len(positions), current)


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-bar Sharpe > sr_benchmark) with skew and kurtosis correction."""
    r = _clean(returns)
    if len(r) < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)  # raw kurtosis (normal = 3)
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(len(r) - 1) / math.sqrt(denom)))


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    r = _clean(returns)
    return float(-np.quantile(r, 1.0 - level)) if len(r) else float("nan")


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    r = _clean(returns)
    if not len(r):
        return float("nan")
    q = np.quantile(r, 1.0 - level)
    tail = r[r <= q]
    return float(-tail.mean()) if len(tail) else float("nan")


def series_metrics(equity: pd.Series, returns: pd.Series, ppy: float, rf: float = 0.0) -> dict[str, Any]:
    """Return / risk metrics of one equity curve."""
    active = _clean(returns)
    dd = drawdown_stats(equity)
    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return(equity),
        "cagr": annualized_return(equity, ppy),
        "volatility": annualized_volatility(returns, ppy),
        "sharpe": sharpe_ratio(returns, ppy, rf),
        "sortino": sortino_ratio(returns, ppy, rf),
        "calmar": calmar_ratio(equity, ppy),
        "max_drawdown": dd.max_drawdown,
        "max_dd_duration": dd.longest_duration,
        "psr": probabilistic_sharpe_ratio(returns),
        "skew": float(stats.skew(active)) if len(active) > 2 else float("nan"),
        "kurtosis": float(stats.kurtosis(active)) if len(active) > 3 else float("nan"),  # excess
        "var_95": value_at_risk(returns),
        "cvar_95": conditional_value_at_risk(returns),
    }


def relative_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series, ppy: float) -> dict[str, float]:
    """CAPM-style regression of strategy on benchmark returns."""
    joined = pd.concat([strategy_returns, benchmark_returns], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    nan = float("nan")
    if len(joined) < 10 or joined.iloc[:, 1].std() == 0:
        return {"alpha": nan, "beta": nan, "correlation": nan, "information_ratio": nan, "tracking_error": nan}
    s, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    fit = stats.linregress(b, s)
    active = s - b
    te = active.std(ddof=1)
    return {
        "alpha": float(fit.intercept * ppy),
        "beta": float(fit.slope),
        "correlation": float(fit.rvalue),
        "information_ratio": float(active.mean() / te * math.sqrt(ppy)) if te > 0 else nan,
        "tracking_error": float(te * math.sqrt(ppy)),
    }


def trade_statistics(trades: pd.DataFrame) -> dict[str, Any]:
    """Round-trip statistics: win rate, profit factor, payoff, expectancy, durations, streaks."""
    nan = float("nan")
    n = len(trades)
    if n == 0:
        return {"total_trades": 0, "long_trades": 0, "short_trades": 0, "win_rate": nan, "profit_factor": nan,
                "avg_win_pct": nan, "avg_loss_pct": nan, "payoff_ratio": nan, "expectancy": nan,
                "expectancy_pct": nan, "avg_r_multiple": nan, "largest_win": nan, "largest_loss": nan,
                "avg_trade_duration": pd.Timedelta(0), "median_trade_duration": pd.Timedelta(0),
                "avg_bars_held": nan, "max_consecutive_losses": 0, "exit_reasons": {}}
    pnl = trades["net_pnl"].astype(float)
    ret = trades["return_pct"].astype(float)
    wins, losses = pnl > 0, pnl < 0
    gross_profit, gross_loss = pnl[wins].sum(), -pnl[losses].sum()
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = math.inf if gross_profit > 0 else nan
    avg_win = ret[wins].mean() if wins.any() else nan
    avg_loss = ret[losses].mean() if losses.any() else nan
    payoff = avg_win / abs(avg_loss) if wins.any() and losses.any() and avg_loss != 0 else nan
    durations = pd.to_datetime(trades["exit_time"], utc=True) - pd.to_datetime(trades["entry_time"], utc=True)

    streak = worst = 0
    for loss in losses.to_numpy():
        streak = streak + 1 if loss else 0
        worst = max(worst, streak)

    return {
        "total_trades": n,
        "long_trades": int((trades["side"] > 0).sum()),
        "short_trades": int((trades["side"] < 0).sum()),
        "win_rate": float(wins.mean()),
        "profit_factor": float(profit_factor),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "payoff_ratio": float(payoff),
        "expectancy": float(pnl.mean()),
        "expectancy_pct": float(ret.mean()),
        "avg_r_multiple": float(trades["r_multiple"].astype(float).mean()),
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "avg_trade_duration": durations.mean(),
        "median_trade_duration": durations.median(),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "max_consecutive_losses": int(worst),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


def monthly_returns_table(equity: pd.Series) -> pd.DataFrame:
    """Year x month grid of compounded monthly returns."""
    idx = pd.DatetimeIndex(equity.index)
    month_end = equity.groupby([idx.year, idx.month]).last()
    previous = month_end.shift(1)
    previous.iloc[0] = equity.iloc[0]
    table = (month_end / previous - 1.0).unstack(level=1).reindex(columns=range(1, 13))
    table.index.name, table.columns.name = "year", "month"
    return table


# ======================================================================================
# Report
# ======================================================================================
def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, pd.Timedelta):
        if pd.isna(value):
            return "n/a"
        hours = int(value.total_seconds() // 3600)
        return f"{hours // 24}d {hours % 24:02d}h"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "inf"
    if kind == "pct":
        return f"{value * 100:+.2f}%"
    if kind == "share":
        return f"{value * 100:.1f}%"
    if kind == "money":
        return f"{value:,.2f}"
    if kind == "int":
        return f"{int(value):,}"
    return f"{value:.3f}"


_SERIES_ROWS: list[tuple[str, str, str]] = [
    ("Final equity", "final_equity", "money"),
    ("Total return", "total_return", "pct"),
    ("Annualised return (CAGR)", "cagr", "pct"),
    ("Annualised volatility", "volatility", "share"),
    ("Sharpe ratio", "sharpe", "ratio"),
    ("Sortino ratio", "sortino", "ratio"),
    ("Calmar ratio", "calmar", "ratio"),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Max drawdown duration", "max_dd_duration", "duration"),
    ("Probabilistic Sharpe (SR>0)", "psr", "share"),
    ("Skewness (per bar)", "skew", "ratio"),
    ("Excess kurtosis (per bar)", "kurtosis", "ratio"),
    ("VaR 95% (per bar)", "var_95", "share"),
    ("CVaR 95% (per bar)", "cvar_95", "share"),
    ("Time in market", "exposure", "share"),
]
_TRADE_ROWS: list[tuple[str, str, str]] = [
    ("Total trades", "total_trades", "int"),
    ("Long / short trades", "long_short", "text"),
    ("Win rate", "win_rate", "share"),
    ("Profit factor", "profit_factor", "ratio"),
    ("Average win", "avg_win_pct", "pct"),
    ("Average loss", "avg_loss_pct", "pct"),
    ("Payoff ratio (avg win / avg loss)", "payoff_ratio", "ratio"),
    ("Expectancy per trade", "expectancy", "money"),
    ("Average R-multiple", "avg_r_multiple", "ratio"),
    ("Largest win", "largest_win", "money"),
    ("Largest loss", "largest_loss", "money"),
    ("Average trade duration", "avg_trade_duration", "duration"),
    ("Median trade duration", "median_trade_duration", "duration"),
    ("Max consecutive losses", "max_consecutive_losses", "int"),
]
_RELATIVE_ROWS: list[tuple[str, str, str]] = [
    ("Alpha (annualised)", "alpha", "pct"),
    ("Beta", "beta", "ratio"),
    ("Correlation", "correlation", "ratio"),
    ("Information ratio", "information_ratio", "ratio"),
    ("Tracking error", "tracking_error", "share"),
]
_COST_ROWS: list[tuple[str, str, str]] = [
    ("Fees paid", "total_fees", "money"),
    ("Slippage & impact paid", "total_slippage", "money"),
    ("Short financing paid", "total_financing", "money"),
    ("Entries skipped (size < minimum)", "skipped_entries", "int"),
    ("Circuit-breaker trips", "breaker_trips", "int"),
]


@dataclass
class PerformanceReport:
    name: str
    period: dict[str, Any]
    strategy: dict[str, Any]
    benchmark: dict[str, Any]
    relative: dict[str, float]
    trades: dict[str, Any]
    costs: dict[str, Any]
    benchmark_label: str = "Buy & Hold"
    extra: dict[str, Any] = field(default_factory=dict)

    def headline(self) -> str:
        s, t = self.strategy, self.trades
        return (f"{self.name}: return {_fmt(s['total_return'], 'pct')} | CAGR {_fmt(s['cagr'], 'pct')} | "
                f"Sharpe {_fmt(s['sharpe'], 'ratio')} | MaxDD {_fmt(s['max_drawdown'], 'pct')} | "
                f"trades {t['total_trades']} | win rate {_fmt(t['win_rate'], 'share')} | "
                f"PF {_fmt(t['profit_factor'], 'ratio')}")

    def to_frame(self) -> pd.DataFrame:
        rows = [(label, _fmt(self.strategy.get(key), kind), _fmt(self.benchmark.get(key), kind))
                for label, key, kind in _SERIES_ROWS]
        return pd.DataFrame(rows, columns=["Metric", "Strategy", self.benchmark_label]).set_index("Metric")

    def format(self) -> str:
        width = 78
        p = self.period
        trades = {**self.trades, "long_short": f"{self.trades['long_trades']} / {self.trades['short_trades']}"}
        lines = [
            "=" * width,
            f" PERFORMANCE REPORT - {self.name}",
            f" {p['symbol']} {p['timeframe']} | {p['start']:%Y-%m-%d %H:%M} -> {p['end']:%Y-%m-%d %H:%M} UTC "
            f"({p['days']:.0f} days, {p['bars']:,} bars) | engine: {p['engine']}",
            "=" * width,
            f" {'Metric':<40}{'Strategy':>18}{self.benchmark_label:>18}",
            "-" * width,
        ]
        lines += [f" {label:<40}{_fmt(self.strategy.get(k), kind):>18}{_fmt(self.benchmark.get(k), kind):>18}"
                  for label, k, kind in _SERIES_ROWS]
        for title, rows, source in (("TRADE STATISTICS", _TRADE_ROWS, trades),
                                    ("RELATIVE TO BENCHMARK", _RELATIVE_ROWS, self.relative),
                                    ("EXECUTION COSTS & RISK CONTROLS", _COST_ROWS, self.costs)):
            lines += ["-" * width, f" {title}"]
            for label, key, kind in rows:
                value = source.get(key)
                text = value if kind == "text" else _fmt(value, kind)
                lines.append(f" {label:<40}{text:>18}")
        reasons = self.trades.get("exit_reasons") or {}
        if reasons:
            lines.append(f" {'Exit reasons':<40}" + ", ".join(f"{k}={v}" for k, v in reasons.items()))
        lines.append("=" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "period": self.period, "strategy": self.strategy, "benchmark": self.benchmark,
                "relative": self.relative, "trades": self.trades, "costs": self.costs, **self.extra}

    def save(self, directory: Path, stem: str) -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {"json": directory / f"{stem}_report.json", "txt": directory / f"{stem}_report.txt"}
        paths["json"].write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        paths["txt"].write_text(self.format(), encoding="utf-8")
        return paths


class PerformanceAnalyzer:
    """Turns a ``BacktestResult`` into a ``PerformanceReport``."""

    def analyze(self, result: BacktestResult) -> PerformanceReport:
        ec, meta = result.equity_curve, result.metadata
        ppy = float(meta["periods_per_year"])
        rf = float(meta.get("risk_free_rate", 0.0))

        strategy = series_metrics(ec["equity"], ec["returns"], ppy, rf)
        strategy["exposure"] = float((ec["position"] != 0).mean())
        benchmark = series_metrics(ec["benchmark_equity"], ec["benchmark_returns"], ppy, rf)
        benchmark["exposure"] = 1.0

        period = {
            "symbol": meta.get("symbol", ""), "timeframe": meta.get("timeframe", ""),
            "start": ec.index[0], "end": ec.index[-1],
            "days": (ec.index[-1] - ec.index[0]) / pd.Timedelta(days=1), "bars": len(ec),
            "engine": meta.get("engine", ""),
        }
        costs = {
            "total_fees": meta.get("total_fees"), "total_slippage": meta.get("total_slippage"),
            "total_financing": meta.get("total_financing"), "skipped_entries": meta.get("skipped_entries", 0),
            "breaker_trips": len(result.breaker_events),
        }
        return PerformanceReport(
            name=result.name, period=period, strategy=strategy, benchmark=benchmark,
            relative=relative_metrics(ec["returns"], ec["benchmark_returns"], ppy),
            trades=trade_statistics(result.trades), costs=costs,
            benchmark_label=meta.get("benchmark_label", "Buy & Hold"),
        )

    def compare(self, results: Mapping[str, BacktestResult]) -> pd.DataFrame:
        """One row per backtest with the headline metrics (numeric, for tables and charts)."""
        rows = []
        for key, res in results.items():
            rep = self.analyze(res)
            rows.append({
                "preset": key, "strategy": res.metadata.get("strategy", res.name),
                "total_return": rep.strategy["total_return"], "cagr": rep.strategy["cagr"],
                "volatility": rep.strategy["volatility"], "sharpe": rep.strategy["sharpe"],
                "sortino": rep.strategy["sortino"], "calmar": rep.strategy["calmar"],
                "max_drawdown": rep.strategy["max_drawdown"], "win_rate": rep.trades["win_rate"],
                "profit_factor": rep.trades["profit_factor"], "trades": rep.trades["total_trades"],
                "exposure": rep.strategy["exposure"],
            })
        return pd.DataFrame(rows).set_index("preset")


# ======================================================================================
# Visualisation
# ======================================================================================
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STRATEGY_C, BENCHMARK_C = "#2a78d6", "#eb6834"
BULL_C, BEAR_C = "#2a78d6", "#e34948"
TENKAN_C, KIJUN_C, CHIKOU_C = "#4a3aa7", "#eda100", "#1baf7a"
CRITICAL_C = "#d03b3b"
DIVERGING_CMAP = LinearSegmentedColormap.from_list("bear_neutral_bull", [BEAR_C, "#f0efec", BULL_C])
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _font_stack() -> list[str]:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    return [f for f in preferred if f in available] or ["DejaVu Sans"]


def _style_rc() -> dict[str, Any]:
    return {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelsize": 10, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "font.family": "sans-serif", "font.sans-serif": _font_stack(), "font.size": 10, "text.color": INK,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    }


def _to_num(index: pd.Index) -> np.ndarray:
    """DatetimeIndex (tz-aware or naive) -> Matplotlib date numbers."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return mdates.date2num(idx.to_numpy())


def _date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _money(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def _end_label(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    """Direct label at the end of a line: a series-coloured dot plus ink-coloured text."""
    ax.scatter([x], [y], s=36, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=5)
    ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK, fontweight="semibold")


def _zero_aligned_bins(values: np.ndarray, target: int) -> np.ndarray:
    """Histogram edges with an edge exactly at 0, so winners and losers never share a bin."""
    lo, hi = float(np.min(values)), float(np.max(values))
    width = (hi - lo) / max(target, 1) or 1e-9
    return np.arange(math.floor(lo / width) * width, hi + width, width)


def _heatmap(ax: plt.Axes, data: np.ndarray, vmax: float, fmt: str) -> Any:
    """Diverging (bear red <-> neutral grey <-> bull blue) heatmap with 2px surface gaps."""
    vmax = max(vmax, 1e-9)
    cmap = DIVERGING_CMAP.with_extremes(bad=SURFACE)  # missing cells stay blank, unlike a 0 % cell
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (r, c), v in np.ndenumerate(data):
        if np.isfinite(v):
            text = fmt.format(v)
            if text.lstrip("-").strip("0.") == "":  # avoid "-0.0"
                text = text.lstrip("-")
            ax.text(c, r, text, ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 * vmax else INK)
    return im


class ChartVisualizer:
    """Publication-quality charts saved as PNG files (and shown on screen with ``show=True``)."""

    def __init__(self, output_dir: Path, show: bool = False, dpi: int = 150) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show = show
        self.dpi = dpi
        if not show and matplotlib.get_backend().lower() != "agg":
            plt.switch_backend("Agg")  # head-less rendering (servers, CI, SSH sessions)
        self._rc = _style_rc()

    def _finish(self, fig: plt.Figure, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(fig)
        logger.info("Chart saved: %s", path)
        return path

    @staticmethod
    def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
        h = fig.get_figheight()  # offsets in inches, so spacing is identical on short and tall figures
        fig.text(0.01, 1 - 0.05 / h, title, ha="left", va="top", fontsize=15, fontweight="semibold", color=INK)
        fig.text(0.01, 1 - 0.40 / h, subtitle, ha="left", va="top", fontsize=10, color=INK2)

    # ------------------------------------------------------------------ Ichimoku charts
    def plot_ichimoku(self, result: BacktestResult, last_n: int = 300, filename: str = "ichimoku_chart.png") -> Path:
        """Candlesticks with all five Ichimoku lines, the Kumo (incl. its projection) and trades."""
        sig = result.signals
        window = sig.iloc[-last_n:]
        future = projected_cloud(sig)
        x, xf = _to_num(window.index), _to_num(future.index)
        bar_days = float(np.median(np.diff(x))) if len(x) > 1 else 1 / 24

        with plt.rc_context(self._rc):
            fig, (ax, axv) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                          gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.04,
                                                       "top": 0.86, "bottom": 0.07, "left": 0.06, "right": 0.97})
            a, b = window["senkou_a"].to_numpy(), window["senkou_b"].to_numpy()
            ax.fill_between(x, a, b, where=a >= b, color=BULL_C, alpha=0.13, linewidth=0, interpolate=True,
                            label="Kumo, bullish (A > B)")
            ax.fill_between(x, a, b, where=a < b, color=BEAR_C, alpha=0.13, linewidth=0, interpolate=True,
                            label="Kumo, bearish (B > A)")
            if len(future):
                fa, fb = future["senkou_a"].to_numpy(), future["senkou_b"].to_numpy()
                ax.fill_between(xf, fa, fb, where=fa >= fb, color=BULL_C, alpha=0.07, linewidth=0, interpolate=True)
                ax.fill_between(xf, fa, fb, where=fa < fb, color=BEAR_C, alpha=0.07, linewidth=0, interpolate=True)
                ax.plot(xf, fa, color=BULL_C, lw=0.8, alpha=0.6)
                ax.plot(xf, fb, color=BEAR_C, lw=0.8, alpha=0.6)
                ax.axvline(x[-1], color=AXIS, lw=0.8)
                ax.annotate("projected Kumo", xy=(xf[len(xf) // 2], np.nanmax([fa.max(), fb.max()])),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color=MUTED)

            self._candles(ax, x, window, width=bar_days * 0.65)
            ax.plot(x, window["tenkan"], color=TENKAN_C, lw=1.3, label="Tenkan-sen (conversion)")
            ax.plot(x, window["kijun"], color=KIJUN_C, lw=1.8, label="Kijun-sen (base)")
            ax.plot(x, window["senkou_a"], color=BULL_C, lw=0.9, label="Senkou Span A")
            ax.plot(x, window["senkou_b"], color=BEAR_C, lw=0.9, label="Senkou Span B")
            ax.plot(x, window["chikou_span"], color=CHIKOU_C, lw=1.1, label="Chikou Span (lagging)")
            self._trade_markers(ax, result.trades, window.index[0], window.index[-1], size=90)

            up = window["close"].to_numpy() >= window["open"].to_numpy()
            axv.bar(x, window["volume"], width=bar_days * 0.65, color=np.where(up, AXIS, MUTED), linewidth=0)
            axv.set_ylabel("Volume")
            axv.grid(axis="x", visible=False)
            ax.set_ylabel("Price")
            _money(ax)
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=5, handlelength=1.6)
            end = xf[-1] if len(xf) else x[-1]
            ax.set_xlim(x[0] - bar_days, end + bar_days)
            _date_axis(axv)
            meta = result.metadata
            self._title(fig, f"{meta.get('symbol', '')} {meta.get('timeframe', '')} - Ichimoku Kinko Hyo",
                        f"{meta.get('strategy', result.name)} | last {len(window)} bars | "
                        "^ buy  v sell | hollow candle = up bar, filled = down bar")
            return self._finish(fig, filename)

    def plot_ichimoku_overview(self, result: BacktestResult, filename: str = "ichimoku_overview.png") -> Path:
        """Whole backtest: close, Kumo, Kijun and every trade."""
        sig = result.signals
        x = _to_num(sig.index)
        with plt.rc_context(self._rc):
            fig, ax = plt.subplots(figsize=(16, 7), gridspec_kw={"top": 0.84, "bottom": 0.08, "left": 0.06,
                                                                  "right": 0.97})
            a, b = sig["senkou_a"].to_numpy(), sig["senkou_b"].to_numpy()
            ax.fill_between(x, a, b, where=a >= b, color=BULL_C, alpha=0.15, linewidth=0, interpolate=True,
                            label="Kumo, bullish")
            ax.fill_between(x, a, b, where=a < b, color=BEAR_C, alpha=0.15, linewidth=0, interpolate=True,
                            label="Kumo, bearish")
            ax.plot(x, sig["close"], color=INK2, lw=1.0, label="Close")
            ax.plot(x, sig["kijun"], color=KIJUN_C, lw=1.2, label="Kijun-sen")
            self._trade_markers(ax, result.trades, sig.index[0], sig.index[-1], size=40)
            ax.set_ylabel("Price")
            _money(ax)
            _date_axis(ax)
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=6)
            meta = result.metadata
            self._title(fig, f"{meta.get('symbol', '')} {meta.get('timeframe', '')} - trades over the full backtest",
                        f"{meta.get('strategy', result.name)} | {len(result.trades)} round trips")
            return self._finish(fig, filename)

    # ------------------------------------------------------------------ secondary strategies
    def plot_simple(self, result: BacktestResult, filename: str = "strategy_chart.png",
                    last_n: int = 400) -> Path:
        """Candles + trend line + trade markers + volume for non-Ichimoku strategies (momentum)."""
        sig = result.signals
        window = sig.iloc[-last_n:]
        x = _to_num(window.index)
        bar_days = float(np.median(np.diff(x))) if len(x) > 1 else 1 / 24

        with plt.rc_context(self._rc):
            fig, (ax, axv) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                          gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.04,
                                                       "top": 0.86, "bottom": 0.07, "left": 0.06, "right": 0.97})
            self._candles(ax, x, window, width=bar_days * 0.65)
            if "kijun" in window.columns and window["kijun"].notna().any():
                ax.plot(x, window["kijun"], color=KIJUN_C, lw=1.4, label="SMA (trend line)")
            self._trade_markers(ax, result.trades, window.index[0], window.index[-1], size=70)
            ax.set_ylabel("Price")
            _money(ax)
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=5)

            up = window["close"].to_numpy() >= window["open"].to_numpy()
            axv.bar(x, window["volume"], width=bar_days * 0.65, color=np.where(up, AXIS, MUTED), linewidth=0)
            axv.set_ylabel("Volume")
            axv.grid(axis="x", visible=False)
            _date_axis(axv)
            meta = result.metadata
            self._title(fig, f"{meta.get('symbol', '')} {meta.get('timeframe', '')} - {meta.get('strategy', result.name)}",
                        f"last {len(window)} bars | ^ buy  v sell")
            return self._finish(fig, filename)

    def plot_pairs(self, result: BacktestResult, filename: str = "pairs_chart.png", last_n: int = 400) -> Path:
        """Normalised legs, spread z-score (with entry/exit/stop levels) and position shade."""
        sig = result.signals
        window = sig.iloc[-last_n:]
        x = _to_num(window.index)
        pairs = result.metadata.get("pairs", {})

        with plt.rc_context(self._rc):
            fig, (ax, az) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                         gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.05,
                                                      "top": 0.86, "bottom": 0.07, "left": 0.06, "right": 0.97})
            base_n = window["base"].to_numpy() / window["base"].iloc[0]
            quote_n = window["quote"].to_numpy() / window["quote"].iloc[0]
            ax.plot(x, base_n, color=BULL_C, lw=1.5, label=pairs.get("base", "base"))
            ax.plot(x, quote_n, color=BENCHMARK_C, lw=1.5, label=pairs.get("quote", "quote"))
            ax.axhline(1.0, color=AXIS, lw=0.9)
            ax.set_ylabel("Normalised price")
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=4)

            z = window["z"].to_numpy()
            az.fill_between(x, 0, 1, where=(window["position"].to_numpy() != 0),
                            transform=az.get_xaxis_transform(), color=STRATEGY_C, alpha=0.10, linewidth=0,
                            step="mid", label="Spread position")
            az.plot(x, z, color=STRATEGY_C, lw=1.1, label="z-score")
            for level, color, style in ((pairs.get("entry_zscore", 2.0), BULL_C, "--"),
                                        (-pairs.get("entry_zscore", 2.0), BEAR_C, "--"),
                                        (pairs.get("exit_zscore", 0.5), MUTED, ":"),
                                        (-pairs.get("exit_zscore", 0.5), MUTED, ":")):
                az.axhline(level, color=color, lw=0.9, ls=style)
            az.axhline(0, color=AXIS, lw=0.9)
            az.set_ylabel("z-score")
            if np.isfinite(z).any():
                bound = float(np.nanpercentile(np.abs(z), 99)) + 0.5
                az.set_ylim(-bound, bound)
            az.legend(loc="lower left", ncol=5)
            _date_axis(az)

            self._title(fig, f"Pairs trading: {pairs.get('base', '')} x {pairs.get('quote', '')}",
                        f"corr {pairs.get('annualised_correlation', float('nan')):.2f} | "
                        f"beta {pairs.get('hedge_beta_mean', float('nan')):.2f} | "
                        f"half-life {pairs.get('spread_half_life_bars', float('nan')):.0f} bars | "
                        f"z entry {pairs.get('entry_zscore', 2.0):.1f} / exit {pairs.get('exit_zscore', 0.5):.1f} / "
                        f"stop {pairs.get('stop_zscore', 3.5):.1f}")
            return self._finish(fig, filename)

    @staticmethod
    def _candles(ax: plt.Axes, x: np.ndarray, frame: pd.DataFrame, width: float) -> None:
        o, h, l, c = (frame[k].to_numpy() for k in ("open", "high", "low", "close"))
        up = c >= o
        ax.vlines(x, l, h, colors=INK2, linewidth=0.7, zorder=2)
        body_low = np.minimum(o, c)
        body_h = np.maximum(np.abs(c - o), (h - l) * 0.02 + 1e-12)
        ax.bar(x[up], body_h[up], width=width, bottom=body_low[up], facecolor=SURFACE, edgecolor=INK2,
               linewidth=0.7, zorder=3)
        ax.bar(x[~up], body_h[~up], width=width, bottom=body_low[~up], color=INK2, linewidth=0, zorder=3)

    @staticmethod
    def _trade_markers(ax: plt.Axes, trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                       size: float) -> None:
        """Buys (long entries, short covers) as blue ^; sells (long exits, short entries) as red v."""
        if trades.empty:
            return
        buys: list[tuple[pd.Timestamp, float]] = []
        sells: list[tuple[pd.Timestamp, float]] = []
        for t in trades.itertuples(index=False):
            entry = (t.entry_time, t.entry_price)
            exit_ = (t.exit_time, t.exit_price)
            opening, closing_ = (buys, sells) if t.side > 0 else (sells, buys)
            if start <= t.entry_time <= end:
                opening.append(entry)
            if start <= t.exit_time <= end:
                closing_.append(exit_)
        for points, marker, color, label in ((buys, "^", BULL_C, "Buy"), (sells, "v", BEAR_C, "Sell")):
            if points:
                ts, px = zip(*points)
                ax.scatter(_to_num(pd.DatetimeIndex(ts)), px, marker=marker, s=size, color=color,
                           edgecolors=SURFACE, linewidths=1.5, zorder=6, label=label)

    # ------------------------------------------------------------------ performance
    def plot_performance_dashboard(self, result: BacktestResult, report: PerformanceReport,
                                   filename: str = "performance_dashboard.png") -> Path:
        """Cumulative return vs benchmark, equity curve, drawdowns, monthly returns, trade PnL."""
        ec = result.equity_curve
        x = _to_num(ec.index)
        capital = result.initial_capital
        bench_label = report.benchmark_label

        with plt.rc_context(self._rc):
            fig = plt.figure(figsize=(16, 15))
            gs = fig.add_gridspec(4, 2, height_ratios=[3.0, 2.1, 1.6, 2.5], hspace=0.42, wspace=0.16,
                                  top=0.925, bottom=0.05, left=0.06, right=0.94)
            ax_cum = fig.add_subplot(gs[0, :])
            ax_eq = fig.add_subplot(gs[1, :], sharex=ax_cum)
            ax_dd = fig.add_subplot(gs[2, :], sharex=ax_cum)
            ax_heat = fig.add_subplot(gs[3, 0])
            ax_hist = fig.add_subplot(gs[3, 1])

            # 1) Cumulative performance vs buy & hold
            strat_cum = ec["equity"] / capital - 1.0
            bench_cum = ec["benchmark_equity"] / capital - 1.0
            ax_cum.plot(x, strat_cum, color=STRATEGY_C, lw=1.9, label="Ichimoku strategy")
            ax_cum.plot(x, bench_cum, color=BENCHMARK_C, lw=1.5, label=bench_label)
            ax_cum.axhline(0, color=AXIS, lw=0.9)
            _end_label(ax_cum, x[-1], strat_cum.iloc[-1], f"{strat_cum.iloc[-1]:+.1%}", STRATEGY_C)
            _end_label(ax_cum, x[-1], bench_cum.iloc[-1], f"{bench_cum.iloc[-1]:+.1%}", BENCHMARK_C)
            ax_cum.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax_cum.set_title("Cumulative return vs benchmark")
            ax_cum.legend(loc="upper left")

            # 2) Equity curve, shaded while a position is open
            ax_eq.plot(x, ec["equity"], color=STRATEGY_C, lw=1.6)
            ax_eq.fill_between(x, 0, 1, where=(ec["position"] != 0).to_numpy(), transform=ax_eq.get_xaxis_transform(),
                               color=STRATEGY_C, alpha=0.07, linewidth=0, step="mid", label="Position open")
            for event in result.breaker_events:
                ex = _to_num(pd.DatetimeIndex([event.timestamp]))[0]
                ax_eq.axvline(ex, color=CRITICAL_C, lw=1.0)
                ax_eq.annotate(f"(!) breaker: {event.reason}", xy=(ex, 1), xycoords=("data", "axes fraction"),
                               xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK2)
            _money(ax_eq)
            ax_eq.set_title(f"Equity curve (start {capital:,.0f})")
            ax_eq.legend(loc="upper left")

            # 3) Underwater (drawdown) chart
            dd_s = ec["drawdown"]
            dd_b = drawdown_series(ec["benchmark_equity"])
            ax_dd.fill_between(x, dd_s, 0, color=STRATEGY_C, alpha=0.12, linewidth=0)
            ax_dd.plot(x, dd_s, color=STRATEGY_C, lw=1.3, label="Ichimoku strategy")
            ax_dd.plot(x, dd_b, color=BENCHMARK_C, lw=1.0, label=bench_label)
            trough = int(np.argmin(dd_s.to_numpy()))
            ax_dd.scatter([x[trough]], [dd_s.iloc[trough]], s=36, color=STRATEGY_C, edgecolors=SURFACE,
                          linewidths=1.5, zorder=5)
            ax_dd.annotate(f"max DD {dd_s.iloc[trough]:.1%}", xy=(x[trough], dd_s.iloc[trough]), xytext=(8, -2),
                           textcoords="offset points", fontsize=9, color=INK, va="top")
            ax_dd.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax_dd.set_title("Drawdown from high-water mark")
            ax_dd.legend(loc="lower left", ncol=2)
            _date_axis(ax_dd)
            plt.setp(ax_cum.get_xticklabels(), visible=False)
            plt.setp(ax_eq.get_xticklabels(), visible=False)

            # 4) Monthly returns heatmap
            table = monthly_returns_table(ec["equity"])
            data = table.to_numpy(dtype=float) * 100
            finite = data[np.isfinite(data)]
            _heatmap(ax_heat, data, float(np.abs(finite).max()) if finite.size else 1.0, "{:.1f}")
            ax_heat.set_xticks(range(12))
            ax_heat.set_xticklabels(MONTHS)
            ax_heat.set_yticks(range(len(table)))
            ax_heat.set_yticklabels([str(y) for y in table.index])
            ax_heat.set_title("Monthly returns (%)")

            # 5) Trade return distribution
            self._trade_histogram(ax_hist, result.trades)

            s, t = report.strategy, report.trades
            self._title(
                fig,
                f"{result.metadata.get('symbol', '')} {result.metadata.get('timeframe', '')} - {result.name}",
                f"CAGR {_fmt(s['cagr'], 'pct')}  |  Sharpe {_fmt(s['sharpe'], 'ratio')}  |  "
                f"Sortino {_fmt(s['sortino'], 'ratio')}  |  Max DD {_fmt(s['max_drawdown'], 'pct')}  |  "
                f"Win rate {_fmt(t['win_rate'], 'share')}  |  Profit factor {_fmt(t['profit_factor'], 'ratio')}  |  "
                f"{t['total_trades']} trades",
            )
            return self._finish(fig, filename)

    @staticmethod
    def _trade_histogram(ax: plt.Axes, trades: pd.DataFrame) -> None:
        ax.set_title("Net return per trade")
        if trades.empty:
            ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes, color=MUTED)
            return
        r = trades["return_pct"].to_numpy(dtype=float) * 100
        bins = _zero_aligned_bins(r, int(np.clip(len(r) // 2, 8, 40)))
        wins, losses = r[r > 0], r[r <= 0]
        ax.hist(wins, bins=bins, color=BULL_C, rwidth=0.85, label=f"Winners ({len(wins)})")
        ax.hist(losses, bins=bins, color=BEAR_C, rwidth=0.85, label=f"Losers ({len(losses)})")
        ax.axvline(0, color=AXIS, lw=1.0)
        mean = float(r.mean())
        ax.axvline(mean, color=INK, lw=1.2)
        ax.annotate(f"mean {mean:+.2f}%", xy=(mean, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=9, color=INK)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_ylabel("Trades")
        ax.legend(loc="upper right")

    def plot_trade_analysis(self, result: BacktestResult, filename: str = "trade_analysis.png") -> Path:
        """Rolling Sharpe, cumulative PnL per trade, exit reasons and the R-multiple distribution."""
        ec, trades = result.equity_curve, result.trades
        ppy = float(result.metadata["periods_per_year"])
        window = max(int(ppy * 30 / 365), 20)  # 30-day rolling window
        x = _to_num(ec.index)

        with plt.rc_context(self._rc):
            fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                                     gridspec_kw={"hspace": 0.38, "wspace": 0.18, "top": 0.89, "bottom": 0.07,
                                                  "left": 0.06, "right": 0.97})
            ax_rs, ax_cum, ax_exit, ax_r = axes.ravel()

            for col, color, label in (("returns", STRATEGY_C, "Ichimoku strategy"),
                                      ("benchmark_returns", BENCHMARK_C, result.metadata.get("benchmark_label"))):
                r = ec[col]
                rolling = r.rolling(window).mean() / r.rolling(window).std() * math.sqrt(ppy)
                ax_rs.plot(x, rolling.replace([np.inf, -np.inf], np.nan), color=color, lw=1.4, label=label)
            ax_rs.axhline(0, color=AXIS, lw=0.9)
            ax_rs.set_title("Rolling 30-day Sharpe ratio (annualised)")
            ax_rs.legend(loc="upper left")
            _date_axis(ax_rs)

            if trades.empty:
                for ax in (ax_cum, ax_exit, ax_r):
                    ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes, color=MUTED)
            else:
                cum = trades["net_pnl"].cumsum().to_numpy()
                n = np.arange(1, len(cum) + 1)
                ax_cum.step(n, cum, where="post", color=STRATEGY_C, lw=1.6)
                ax_cum.fill_between(n, cum, 0, step="post", color=STRATEGY_C, alpha=0.08, linewidth=0)
                ax_cum.axhline(0, color=AXIS, lw=0.9)
                _end_label(ax_cum, n[-1], cum[-1], f"{cum[-1]:+,.0f}", STRATEGY_C)
                ax_cum.set_xlabel("Trade number")
                _money(ax_cum)

                reasons = trades["exit_reason"].value_counts().sort_values()
                ypos = np.arange(len(reasons))
                ax_exit.barh(ypos, reasons.to_numpy(), height=0.5, color=STRATEGY_C)
                ax_exit.set_yticks(ypos)
                ax_exit.set_yticklabels([r.replace("_", " ") for r in reasons.index])
                for yy, v in zip(ypos, reasons.to_numpy()):
                    ax_exit.annotate(f"{v}", xy=(v, yy), xytext=(4, 0), textcoords="offset points", va="center",
                                     fontsize=9, color=INK)
                ax_exit.grid(axis="y", visible=False)
                ax_exit.set_xlabel("Trades")

                rm = trades["r_multiple"].to_numpy(dtype=float)
                rm = rm[np.isfinite(rm)]
                if rm.size:
                    bins = _zero_aligned_bins(rm, int(np.clip(rm.size // 2, 8, 40)))
                    ax_r.hist(rm[rm > 0], bins=bins, color=BULL_C, rwidth=0.85, label="Winners")
                    ax_r.hist(rm[rm <= 0], bins=bins, color=BEAR_C, rwidth=0.85, label="Losers")
                    ax_r.axvline(0, color=AXIS, lw=1.0)
                    ax_r.axvline(-1, color=MUTED, lw=1.0)
                    ax_r.annotate("-1R = initial stop", xy=(-1, 1), xycoords=("data", "axes fraction"),
                                  xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK2)
                    ax_r.set_xlabel("R-multiple (net PnL / initial risk)")
                    ax_r.set_ylabel("Trades")
                    ax_r.legend(loc="upper right")
                else:
                    ax_r.text(0.5, 0.5, "R-multiples need the event-driven engine", ha="center", va="center",
                              transform=ax_r.transAxes, color=MUTED)
            ax_cum.set_title("Cumulative net PnL by trade")
            ax_exit.set_title("Exit reasons")
            ax_r.set_title("R-multiple distribution")
            self._title(fig, f"Trade analysis - {result.name}",
                        f"{result.metadata.get('symbol', '')} {result.metadata.get('timeframe', '')} | "
                        f"{len(trades)} round trips")
            return self._finish(fig, filename)

    # ------------------------------------------------------------------ research charts
    def plot_optimization_heatmap(self, opt: OptimizationResult, filename: str = "optimization_heatmap.png") -> Path:
        """In-sample vs out-of-sample score over the Tenkan x Kijun grid (overfitting check)."""
        metric = opt.metric
        pivots = {p: opt.table.pivot(index="tenkan", columns="kijun", values=f"{p}_{metric}") for p in ("is", "oos")}
        values = np.concatenate([p.to_numpy(dtype=float).ravel() for p in pivots.values()])
        values = values[np.isfinite(values)]
        vmax = float(np.abs(values).max()) if values.size else 1.0

        with plt.rc_context(self._rc):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6.5),
                                     gridspec_kw={"wspace": 0.12, "top": 0.82, "bottom": 0.1, "left": 0.05,
                                                  "right": 0.93})
            im = None
            for ax, (prefix, pivot), title in zip(axes, pivots.items(),
                                                  ("In-sample (fitted)", "Out-of-sample (unseen data)")):
                im = _heatmap(ax, pivot.to_numpy(dtype=float), vmax, "{:.2f}")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([str(c) for c in pivot.columns])
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([str(i) for i in pivot.index])
                ax.set_xlabel("Kijun period (Senkou B = 2 x Kijun, displacement = Kijun)")
                ax.set_ylabel("Tenkan period")
                ax.set_title(title)
                best = opt.best_params
                if best.tenkan in pivot.index and best.kijun in pivot.columns:
                    r, c = pivot.index.get_loc(best.tenkan), pivot.columns.get_loc(best.kijun)
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
            cax = fig.add_axes((0.945, 0.1, 0.012, 0.72))
            fig.colorbar(im, cax=cax).set_label(f"{metric.title()} ratio", color=INK2)
            self._title(fig, f"Parameter optimisation - {metric.title()} over the Tenkan x Kijun grid",
                        f"Split at {opt.split_time:%Y-%m-%d}. Black outline = best in-sample "
                        f"{opt.best_params.label}. Compare both panels to judge overfitting.")
            return self._finish(fig, filename)

    def plot_preset_comparison(self, results: Mapping[str, BacktestResult], table: pd.DataFrame,
                               filename: str = "preset_comparison.png") -> Path:
        """Small multiples (one panel per preset vs buy & hold) plus headline-metric bars."""
        n = len(results)
        with plt.rc_context(self._rc):
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.4], hspace=0.35, top=0.89, bottom=0.08,
                                  left=0.06, right=0.97)
            top = gs[0].subgridspec(1, n, wspace=0.08)
            first_ax = None
            for k, (key, res) in enumerate(results.items()):
                ax = fig.add_subplot(top[0, k], sharey=first_ax)
                first_ax = first_ax or ax
                ec = res.equity_curve
                x = _to_num(ec.index)
                growth = ec["equity"] / res.initial_capital
                bench = ec["benchmark_equity"] / res.initial_capital
                ax.plot(x, bench, color=BENCHMARK_C, lw=1.1, label=res.metadata.get("benchmark_label"))
                ax.plot(x, growth, color=STRATEGY_C, lw=1.7, label="Strategy")
                ax.axhline(1.0, color=AXIS, lw=0.9)
                _end_label(ax, x[-1], growth.iloc[-1], f"{growth.iloc[-1] - 1:+.0%}", STRATEGY_C)
                ax.set_title(key.replace("_", " "))
                _date_axis(ax)
                if k:
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    ax.set_ylabel("Growth of 1 unit")
                    ax.legend(loc="upper left")

            bottom = gs[1].subgridspec(1, 3, wspace=0.25)
            for k, (col, title, fmt) in enumerate((("sharpe", "Sharpe ratio", "{:.2f}"),
                                                   ("cagr", "CAGR", "{:+.1%}"),
                                                   ("max_drawdown", "Max drawdown", "{:.1%}"))):
                ax = fig.add_subplot(bottom[0, k])
                vals = table[col].to_numpy(dtype=float)
                ypos = np.arange(len(vals))
                ax.barh(ypos, np.nan_to_num(vals), height=0.5, color=STRATEGY_C)
                ax.axvline(0, color=AXIS, lw=0.9)
                ax.set_yticks(ypos)
                if k == 0:
                    ax.set_yticklabels([s.replace("_", " ") for s in table.index])
                else:
                    ax.tick_params(labelleft=False)
                ax.invert_yaxis()
                ax.grid(axis="y", visible=False)
                for yy, v in zip(ypos, vals):
                    if np.isfinite(v):
                        ax.annotate(fmt.format(v), xy=(v, yy), xytext=(4 if v >= 0 else -4, 0),
                                    textcoords="offset points", va="center", ha="left" if v >= 0 else "right",
                                    fontsize=9, color=INK)
                ax.set_title(title)
                ax.margins(x=0.25)
                if col != "sharpe":
                    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            self._title(fig, "Ichimoku presets compared (event-driven backtest, identical risk settings)",
                        "standard 9/26/52 | crypto 10/30/60 (24/7 calendar) | crypto slow 20/60/120 | "
                        "dynamic = volatility-regime switching")
            return self._finish(fig, filename)

    def create_all(self, result: BacktestResult, report: PerformanceReport) -> list[Path]:
        return [
            self.plot_ichimoku(result),
            self.plot_ichimoku_overview(result),
            self.plot_performance_dashboard(result, report),
            self.plot_trade_analysis(result),
        ]
