"""
backtester.py - Vectorised and event-driven backtesting engines.

VectorizedBacktester
    Research engine in the vectorised style: whole-array arithmetic, full notional,
    proportional costs, no stops. Thousands of parameter sets per minute, so it drives
    the parameter optimiser.

EventDrivenBacktester
    Bar-by-bar simulation of what the live engine will do:
      * signals on the close of bar t are executed at the open of bar t+1 (no look-ahead)
      * risk-based position sizing (fixed-fractional / Kelly)
      * ATR or Kumo stop-loss & take-profit, optional trailing stop, conservative
        intrabar fill rules for gaps and "both levels inside one bar"
      * taker fees, bid/ask spread, slippage and square-root market impact
      * financing cost on shorts
      * max-drawdown / daily-loss circuit breaker

Both engines return a ``BacktestResult`` that ``analytics.py`` turns into metrics and charts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from config import ICHIMOKU_PRESETS, AppConfig, IchimokuParams, periods_per_year
from risk import (
    FLAT,
    LONG,
    SHORT,
    BreakerEvent,
    CircuitBreaker,
    CostModel,
    ExitReason,
    StopManager,
    build_position_sizer,
)
from strategy import IchimokuStrategy

logger = logging.getLogger(__name__)

TRADE_COLUMNS: list[str] = [
    "trade_id", "side", "entry_time", "exit_time", "entry_price", "exit_price", "quantity",
    "gross_pnl", "fees", "financing", "net_pnl", "return_pct", "r_multiple", "bars_held", "exit_reason",
]


class BacktestError(RuntimeError):
    """Raised when a backtest cannot run (e.g. not enough data for the indicators)."""


# --------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------
@dataclass
class Trade:
    trade_id: int
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    financing: float
    net_pnl: float
    return_pct: float
    r_multiple: float
    bars_held: int
    exit_reason: str


def trades_to_frame(trades: Iterable[Trade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades], columns=TRADE_COLUMNS)


@dataclass
class BacktestResult:
    name: str
    equity_curve: pd.DataFrame   # equity, cash, position, quantity, close, halted, returns, benchmark_*, drawdown
    trades: pd.DataFrame         # one row per round trip (TRADE_COLUMNS)
    signals: pd.DataFrame        # indicators + signals used, for charts and audit
    metadata: dict[str, Any]
    breaker_events: list[BreakerEvent] = field(default_factory=list)

    @property
    def initial_capital(self) -> float:
        return float(self.metadata["initial_capital"])

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve["equity"].iloc[-1])

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_capital - 1.0

    def save(self, directory: Path) -> dict[str, Path]:
        """Persist equity curve, trade list and metadata (the 'table view' of every chart)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        slug = slugify(self.name)
        paths = {
            "equity_curve": directory / f"{slug}_equity_curve.csv",
            "trades": directory / f"{slug}_trades.csv",
            "metadata": directory / f"{slug}_metadata.json",
        }
        self.equity_curve.to_csv(paths["equity_curve"])
        self.trades.to_csv(paths["trades"], index=False)
        meta = {**self.metadata, "breaker_events": [asdict(e) for e in self.breaker_events]}
        paths["metadata"].write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return paths


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "backtest"


def _finalise_curve(curve: pd.DataFrame, capital: float, one_way_cost: float) -> pd.DataFrame:
    """Add strategy returns, the buy-and-hold benchmark and the drawdown series.

    Benchmark: buy at the first close (paying one market-order cost) and hold, so
    E^BH_t = E_0 * (1 - c) * C_t / C_0.
    """
    curve["returns"] = curve["equity"].pct_change().fillna(0.0)
    curve["benchmark_equity"] = capital * (1.0 - one_way_cost) * curve["close"] / curve["close"].iloc[0]
    curve["benchmark_returns"] = curve["benchmark_equity"].pct_change().fillna(0.0)
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
    return curve


# --------------------------------------------------------------------------------------
# Vectorised engine
# --------------------------------------------------------------------------------------
class VectorizedBacktester:
    """Fully vectorised backtest.

        r_t      = C_t / C_{t-1} - 1                      bar return
        w_t      = position_{t-1}                         decided on the previous close
        cost_t   = |w_t - w_{t-1}| * c                    c = fee + half-spread + slippage
        r^s_t    = w_t * r_t - cost_t                     strategy return
        E_t      = E_0 * prod_{u<=t} (1 + r^s_u)          equity (compounded)

    Using shift(1) is what removes look-ahead: the position used for bar t's return
    is known before bar t starts. Crypto trades 24/7, so the next bar opens where the
    last one closed (Open_{t+1} ~ Close_t), and filling at the signal bar's close is
    practically the same as filling at the next open.
    """

    def __init__(self, cfg: AppConfig, strategy: IchimokuStrategy | None = None) -> None:
        self.cfg = cfg
        self.strategy = strategy or IchimokuStrategy(cfg.strategy)
        self.cost_model = CostModel(cfg.costs)

    def run(self, df: pd.DataFrame, name: str | None = None, signals: pd.DataFrame | None = None) -> BacktestResult:
        sig = signals if signals is not None else self.strategy.generate_signals(df)
        if len(sig) < self.strategy.warmup_bars + 2:
            raise BacktestError(f"Need at least {self.strategy.warmup_bars + 2} bars, got {len(sig)}")
        capital = self.cfg.risk.initial_capital
        cost = self.cost_model.one_way_cost

        close = sig["close"]
        bar_ret = close.pct_change().fillna(0.0)
        held = sig["position"].shift(1).fillna(0).astype("float64")
        turnover = held.diff().abs().fillna(held.abs())
        strat_ret = held * bar_ret - turnover * cost
        equity = capital * (1.0 + strat_ret).cumprod()

        curve = pd.DataFrame(
            {
                "equity": equity,
                "cash": np.nan,
                "position": held.astype("int8"),
                "quantity": np.nan,
                "close": close,
                "halted": False,
            },
            index=sig.index,
        )
        curve = _finalise_curve(curve, capital, cost)
        prev_equity = equity.shift(1).fillna(capital)
        fees = float((turnover * self.cfg.costs.taker_fee * prev_equity).sum())
        slippage = float((turnover * (cost - self.cfg.costs.taker_fee) * prev_equity).sum())
        trades = self._extract_trades(held, close, equity, cost)

        metadata = _metadata(self.cfg, name or f"{self.strategy.name} [vectorised]", "vectorised",
                             self.strategy.name, fees, slippage, 0.0, 0)
        return BacktestResult(metadata["name"], curve, trades, sig, metadata)

    @staticmethod
    def _extract_trades(held: pd.Series, close: pd.Series, equity: pd.Series, cost: float) -> pd.DataFrame:
        """Round trips = maximal runs of a constant non-zero position.

        A run starting at bar s was opened at the close of bar s-1 and closed at the
        close of its last bar e. Net return = side * (C_e / C_{s-1} - 1) - 2c.
        """
        h, c, eq, idx = held.to_numpy(), close.to_numpy(), equity.to_numpy(), held.index
        n, i, trades = len(h), 1, []
        while i < n:
            if h[i] != 0 and h[i] != h[i - 1]:
                side, start, j = int(h[i]), i, i
                while j + 1 < n and h[j + 1] == side:
                    j += 1
                entry_px, exit_px, eq_entry = c[start - 1], c[j], eq[start - 1]
                gross_ret = side * (exit_px / entry_px - 1.0)
                net_ret = gross_ret - 2.0 * cost
                trades.append(Trade(
                    trade_id=len(trades) + 1, side=side, entry_time=idx[start - 1], exit_time=idx[j],
                    entry_price=entry_px, exit_price=exit_px, quantity=float("nan"),
                    gross_pnl=gross_ret * eq_entry, fees=2.0 * cost * eq_entry, financing=0.0,
                    net_pnl=net_ret * eq_entry, return_pct=net_ret, r_multiple=float("nan"),
                    bars_held=j - start + 1,
                    exit_reason=(ExitReason.END_OF_DATA if j == n - 1 else ExitReason.SIGNAL).value,
                ))
                i = j + 1
            else:
                i += 1
        return trades_to_frame(trades)


# --------------------------------------------------------------------------------------
# Event-driven engine
# --------------------------------------------------------------------------------------
@dataclass
class _Position:
    side: int
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_index: int
    stop: float
    initial_stop: float
    take_profit: float | None
    entry_fee: float
    initial_risk: float    # currency lost if the initial stop is hit: qty * |entry - stop|
    extreme: float         # highest high (long) / lowest low (short) since entry
    financing: float = 0.0


@dataclass(frozen=True)
class _Order:
    action: str            # "enter" | "exit" | "reverse"
    side: int              # side of the position to open (enter / reverse)
    reason: ExitReason     # why the current position is being closed (exit / reverse)


class _Simulation:
    """Mutable state of one event-driven run. Uses NumPy arrays in the hot loop."""

    def __init__(self, sig: pd.DataFrame, cfg: AppConfig, cost_model: CostModel, stops: StopManager) -> None:
        self.cfg = cfg
        self.cost_model = cost_model
        self.stops = stops
        self.idx = sig.index
        self.o, self.h, self.l, self.c, self.v = (sig[k].to_numpy("float64") for k in ("open", "high", "low", "close", "volume"))
        self.atr = sig["atr"].to_numpy("float64")
        self.bar_sigma = sig["natr"].to_numpy("float64")   # ATR/Close as the per-bar volatility for impact
        self.kijun = sig["kijun"].to_numpy("float64")
        self.cloud_top = sig["cloud_top"].to_numpy("float64")
        self.cloud_bottom = sig["cloud_bottom"].to_numpy("float64")
        self.long_entry = sig["long_entry"].to_numpy(bool)
        self.short_entry = sig["short_entry"].to_numpy(bool)
        self.exit_long = sig["exit_long"].to_numpy(bool)
        self.exit_short = sig["exit_short"].to_numpy(bool)

        self.cash = cfg.risk.initial_capital
        self.pos: _Position | None = None
        self.trades: list[Trade] = []
        self.fees = self.slippage = self.financing = 0.0
        self.skipped_entries = 0
        self.sizer = build_position_sizer(cfg.risk)
        self.breaker = CircuitBreaker.from_config(cfg.risk, cfg.data.timeframe)
        self.ppy = periods_per_year(cfg.data.timeframe)

    def equity(self, price: float) -> float:
        """Mark-to-market equity  E = cash + side * qty * price  (valid for longs and shorts)."""
        return self.cash + (self.pos.side * self.pos.qty * price if self.pos else 0.0)

    # --------------------------------------------------------------------- execution
    def open_position(self, side: int, i: int, s: int) -> None:
        """Open at bar i's open using information from signal bar s = i - 1."""
        ref = self.o[i]
        atr, top, bottom = self.atr[s], self.cloud_top[s], self.cloud_bottom[s]
        levels = self.stops.initial_levels(side, ref, atr, top, bottom)
        if levels is None:
            self.skipped_entries += 1
            return
        equity = self.cash  # flat, so equity == cash
        qty = self.sizer.size(equity, ref, levels.stop_loss)
        # Liquidity inputs come from the last *completed* bar: bar i's volume is unknown at its open.
        fill = self.cost_model.fill_price(ref, side, qty, self.v[s], self.bar_sigma[s])
        levels = self.stops.initial_levels(side, fill, atr, top, bottom)   # anchor levels to the real fill
        assert levels is not None  # same ATR as above, so this cannot fail
        qty = self.sizer.size(equity, fill, levels.stop_loss)
        if side == LONG:  # spot: cannot spend more than the cash balance (fee included)
            qty = min(qty, self.cash / (fill * (1.0 + self.cfg.costs.taker_fee)))
        notional = qty * fill
        if notional < self.cfg.risk.min_order_notional:
            self.skipped_entries += 1
            return
        fee = self.cost_model.fee(notional)
        self.cash -= side * notional + fee          # long: pay notional; short: receive proceeds
        self.fees += fee
        self.slippage += qty * abs(fill - ref)
        self.pos = _Position(side, qty, fill, self.idx[i], i, levels.stop_loss, levels.stop_loss,
                             levels.take_profit, fee, qty * abs(fill - levels.stop_loss), fill)

    def close_position(self, i: int, ref_price: float, reason: ExitReason) -> None:
        p = self.pos
        assert p is not None
        s = max(i - 1, 0)
        fill = self.cost_model.fill_price(ref_price, -p.side, p.qty, self.v[s], self.bar_sigma[s])
        notional = p.qty * fill
        fee = self.cost_model.fee(notional)
        self.cash += p.side * notional - fee        # long: receive proceeds; short: buy back
        self.fees += fee
        self.slippage += p.qty * abs(fill - ref_price)

        gross = p.side * (fill - p.entry_price) * p.qty
        net = gross - p.entry_fee - fee - p.financing
        r_multiple = net / p.initial_risk if p.initial_risk > 0 else float("nan")
        self.trades.append(Trade(
            trade_id=len(self.trades) + 1, side=p.side, entry_time=p.entry_time, exit_time=self.idx[i],
            entry_price=p.entry_price, exit_price=fill, quantity=p.qty, gross_pnl=gross,
            fees=p.entry_fee + fee, financing=p.financing, net_pnl=net,
            return_pct=net / (p.entry_price * p.qty), r_multiple=r_multiple,
            bars_held=i - p.entry_index + 1, exit_reason=reason.value,
        ))
        self.sizer.record_trade(r_multiple)
        self.pos = None

    # --------------------------------------------------------------------- decisions
    def decide(self, i: int, halted: bool) -> _Order | None:
        """Position state machine, identical to the live engine's."""
        allow_short = self.cfg.strategy.allow_short
        p = self.pos
        if p is None:
            if halted:
                return None
            if self.long_entry[i]:
                return _Order("enter", LONG, ExitReason.SIGNAL)
            if allow_short and self.short_entry[i]:
                return _Order("enter", SHORT, ExitReason.SIGNAL)
            return None
        if p.side == LONG and self.exit_long[i]:
            if allow_short and self.short_entry[i] and not halted:
                return _Order("reverse", SHORT, ExitReason.REVERSAL)
            return _Order("exit", FLAT, ExitReason.SIGNAL)
        if p.side == SHORT and self.exit_short[i]:
            if self.long_entry[i] and not halted:
                return _Order("reverse", LONG, ExitReason.REVERSAL)
            return _Order("exit", FLAT, ExitReason.SIGNAL)
        return None

    # --------------------------------------------------------------------- main loop
    def run(self) -> dict[str, np.ndarray]:
        n = len(self.idx)
        out = {
            "equity": np.empty(n), "cash": np.empty(n), "quantity": np.zeros(n),
            "position": np.zeros(n, dtype=np.int8), "halted": np.zeros(n, dtype=bool),
        }
        borrow_rate = self.cfg.costs.borrow_rate_annual
        pending: _Order | None = None

        for i in range(n):
            ts = self.idx[i]

            # 1) Fill the order decided at the previous close, at this bar's open.
            if pending is not None:
                if pending.action in ("exit", "reverse") and self.pos is not None:
                    self.close_position(i, self.o[i], pending.reason)
                if pending.action in ("enter", "reverse") and self.pos is None and not self.breaker.is_halted(ts):
                    self.open_position(pending.side, i, i - 1)
                pending = None

            # 2) Protective stop / target inside this bar.
            if self.pos is not None:
                hit = StopManager.check_intrabar_exit(self.pos.side, self.pos.stop, self.pos.take_profit,
                                                      self.o[i], self.h[i], self.l[i])
                if hit is not None:
                    reason, level = hit
                    if reason is ExitReason.STOP_LOSS and self.pos.stop != self.pos.initial_stop:
                        reason = ExitReason.TRAILING_STOP
                    self.close_position(i, level, reason)

            # 3) Financing on shorts: notional * annual rate / bars per year.
            if self.pos is not None and self.pos.side == SHORT and borrow_rate > 0:
                carry = self.pos.qty * self.c[i] * borrow_rate / self.ppy
                self.cash -= carry
                self.pos.financing += carry
                self.financing += carry

            # 4) Mark to market and feed the circuit breaker.
            equity = self.equity(self.c[i])
            if self.breaker.update(ts, equity) is not None and self.pos is not None:
                pending = _Order("exit", FLAT, ExitReason.CIRCUIT_BREAKER)
            halted = self.breaker.is_halted(ts)

            # 5) Trailing stop, using information up to this close (effective from next bar).
            if self.pos is not None:
                p = self.pos
                p.extreme = max(p.extreme, self.h[i]) if p.side == LONG else min(p.extreme, self.l[i])
                p.stop = self.stops.trail(p.side, p.stop, p.extreme, self.atr[i], self.kijun[i])

            # 6) Strategy decision on this close -> order for the next open.
            if pending is None and i < n - 1:
                pending = self.decide(i, halted)

            out["equity"][i] = equity
            out["cash"][i] = self.cash
            out["quantity"][i] = self.pos.qty if self.pos else 0.0
            out["position"][i] = self.pos.side if self.pos else 0
            out["halted"][i] = halted

        # 7) Liquidate any open position at the final close.
        if self.pos is not None:
            self.close_position(n - 1, self.c[n - 1], ExitReason.END_OF_DATA)
            out["equity"][-1] = self.equity(self.c[-1])
            out["cash"][-1] = self.cash
            out["quantity"][-1] = 0.0
            out["position"][-1] = 0
        return out


class EventDrivenBacktester:
    """Bar-by-bar backtest with the full risk and execution-cost model."""

    def __init__(self, cfg: AppConfig, strategy: IchimokuStrategy | None = None) -> None:
        self.cfg = cfg
        self.strategy = strategy or IchimokuStrategy(cfg.strategy)
        self.cost_model = CostModel(cfg.costs)
        self.stops = StopManager(cfg.risk)

    def run(self, df: pd.DataFrame, name: str | None = None) -> BacktestResult:
        sig = self.strategy.generate_signals(df)
        if len(sig) < self.strategy.warmup_bars + 2:
            raise BacktestError(f"Need at least {self.strategy.warmup_bars + 2} bars, got {len(sig)}")
        sim = _Simulation(sig, self.cfg, self.cost_model, self.stops)
        arrays = sim.run()

        curve = pd.DataFrame({**arrays, "close": sig["close"].to_numpy()}, index=sig.index)
        curve = _finalise_curve(curve, self.cfg.risk.initial_capital, self.cost_model.one_way_cost)
        metadata = _metadata(self.cfg, name or self.strategy.name, "event-driven", self.strategy.name,
                             sim.fees, sim.slippage, sim.financing, sim.skipped_entries)
        logger.info("Backtest '%s': %d trades, final equity %.2f (fees %.2f, slippage %.2f)",
                    metadata["name"], len(sim.trades), curve["equity"].iloc[-1], sim.fees, sim.slippage)
        return BacktestResult(metadata["name"], curve, trades_to_frame(sim.trades), sig, metadata,
                              list(sim.breaker.events))


def _metadata(cfg: AppConfig, name: str, engine: str, strategy_label: str, fees: float, slippage: float,
              financing: float, skipped: int) -> dict[str, Any]:
    return {
        "name": name,
        "engine": engine,
        "strategy": strategy_label,
        "symbol": cfg.data.symbol,
        "timeframe": cfg.data.timeframe,
        "periods_per_year": periods_per_year(cfg.data.timeframe),
        "initial_capital": cfg.risk.initial_capital,
        "total_fees": fees,
        "total_slippage": slippage,
        "total_financing": financing,
        "skipped_entries": skipped,
        "risk_free_rate": cfg.backtest.risk_free_rate,
        "benchmark_label": cfg.backtest.benchmark_label,
        "config": cfg.to_dict(),
    }


# --------------------------------------------------------------------------------------
# Research utilities
# --------------------------------------------------------------------------------------
def compare_presets(
    df: pd.DataFrame,
    cfg: AppConfig,
    presets: Sequence[str] = ("standard", "crypto", "crypto_slow", "dynamic"),
) -> dict[str, BacktestResult]:
    """Run the event-driven backtest once per Ichimoku preset (other settings unchanged)."""
    results: dict[str, BacktestResult] = {}
    for preset in presets:
        if preset == "dynamic":
            strat_cfg = replace(cfg.strategy, dynamic=True)
        else:
            strat_cfg = replace(cfg.strategy, params=ICHIMOKU_PRESETS[preset], dynamic=False)
        run_cfg = replace(cfg, strategy=strat_cfg)
        results[preset] = EventDrivenBacktester(run_cfg).run(df, name=f"{preset}: {strat_cfg.label}")
    return results


@dataclass
class OptimizationResult:
    table: pd.DataFrame
    best_params: IchimokuParams
    split_time: pd.Timestamp
    metric: str


class ParameterOptimizer:
    """Grid search with a chronological in-sample / out-of-sample split.

    Parameters are ranked on the in-sample (IS) segment only. The out-of-sample
    (OOS) columns show how the winners hold up on data they were not fitted to.
    A large IS -> OOS decay is the signature of overfitting (data snooping), which is
    why an honest capstone report shows both columns.

    The grid keeps Hosoda's structure: Senkou B = 2 x Kijun and displacement = Kijun.
    Indicators are computed on the full history (they are causal), so the OOS segment
    is simply evaluated on returns after the split time.
    """

    DEFAULT_TENKAN: tuple[int, ...] = (5, 7, 9, 10, 12, 15, 20)
    DEFAULT_KIJUN: tuple[int, ...] = (20, 26, 30, 40, 52, 60)

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def grid_search(
        self,
        df: pd.DataFrame,
        tenkan_grid: Sequence[int] | None = None,
        kijun_grid: Sequence[int] | None = None,
        metric: str = "sharpe",
    ) -> OptimizationResult:
        from analytics import annualized_return, max_drawdown, sharpe_ratio, sortino_ratio

        scorers = {"sharpe": sharpe_ratio, "sortino": sortino_ratio}
        if metric not in scorers and metric != "cagr":
            raise ValueError("metric must be one of: sharpe, sortino, cagr")
        ppy = self.cfg.periods_per_year
        split = int(len(df) * self.cfg.backtest.train_fraction)
        if split < 200 or len(df) - split < 100:
            raise BacktestError("Not enough data for an in-sample / out-of-sample split")
        split_time = df.index[split]

        def score(returns: pd.Series, equity: pd.Series) -> float:
            if metric == "cagr":
                return annualized_return(equity, ppy)
            return scorers[metric](returns, ppy, self.cfg.backtest.risk_free_rate)

        rows = []
        for tenkan in tenkan_grid or self.DEFAULT_TENKAN:
            for kijun in kijun_grid or self.DEFAULT_KIJUN:
                if tenkan >= kijun:
                    continue
                params = IchimokuParams(tenkan, kijun, 2 * kijun, kijun)
                strategy = IchimokuStrategy(replace(self.cfg.strategy, params=params, dynamic=False))
                res = VectorizedBacktester(self.cfg, strategy).run(df)
                ret, eq = res.equity_curve["returns"], res.equity_curve["equity"]
                is_ret, oos_ret = ret.iloc[:split], ret.iloc[split:]
                is_eq, oos_eq = eq.iloc[:split], eq.iloc[split - 1:]
                n_is = int((res.trades["entry_time"] < split_time).sum()) if not res.trades.empty else 0
                rows.append({
                    "tenkan": tenkan, "kijun": kijun, "senkou_b": 2 * kijun, "displacement": kijun,
                    f"is_{metric}": score(is_ret, is_eq),
                    "is_cagr": annualized_return(is_eq, ppy),
                    "is_max_dd": max_drawdown(is_eq),
                    "is_trades": n_is,
                    f"oos_{metric}": score(oos_ret, oos_eq),
                    "oos_cagr": annualized_return(oos_eq, ppy),
                    "oos_max_dd": max_drawdown(oos_eq),
                    "oos_trades": len(res.trades) - n_is,
                })
        table = pd.DataFrame(rows).sort_values(f"is_{metric}", ascending=False, na_position="last")
        table = table.reset_index(drop=True)
        best = table.iloc[0]
        best_params = IchimokuParams(int(best.tenkan), int(best.kijun), int(best.senkou_b), int(best.displacement))
        logger.info("Optimiser: %d combinations; best IS %s = %.3f with %s (OOS %.3f)", len(table), metric,
                    best[f"is_{metric}"], best_params.label, best[f"oos_{metric}"])
        return OptimizationResult(table, best_params, split_time, metric)
