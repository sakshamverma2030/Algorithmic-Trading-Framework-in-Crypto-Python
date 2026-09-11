"""
risk.py - Execution-cost model, position sizing, protective stops and circuit breakers.

This module is shared by ``backtester.py`` and ``live_trader.py``. Keeping the risk
logic in one place guarantees that paper/live trading applies exactly the rules that
were backtested (research/production parity).

Components
----------
CostModel        exchange fees + bid/ask spread + slippage + square-root market impact
PositionSizer    fixed-fractional and (fractional) Kelly-criterion risk sizing
StopManager      ATR- or Kumo-based stop-loss / take-profit, trailing stops, and
                 conservative intrabar fill rules for OHLC data
CircuitBreaker   max-drawdown kill switch with cool-down, plus a daily loss limit
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from config import CostConfig, RiskConfig, SizingMethod, StopMethod, TrailingMethod, timeframe_to_minutes

logger = logging.getLogger(__name__)

LONG, FLAT, SHORT = 1, 0, -1


class ExitReason(str, Enum):
    SIGNAL = "signal"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    REVERSAL = "reversal"
    CIRCUIT_BREAKER = "circuit_breaker"
    END_OF_DATA = "end_of_data"
    SHUTDOWN = "shutdown"


class Liquidity(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


# --------------------------------------------------------------------------------------
# Transaction costs & market microstructure
# --------------------------------------------------------------------------------------
class CostModel:
    """Effective execution price and fees.

    For an order of size Q (base units) at reference price P (the mid / open):

        P_fill = P * (1 + s * (h + delta + I(Q)))       s = +1 buy, -1 sell

        h     half the quoted bid/ask spread: a market buy lifts the ask, a sell hits the bid
        delta fixed slippage from latency and queue position
        I(Q)  square-root market impact (Almgren et al. 2005; Toth et al. 2011):
                  I(Q) = Y * sigma_bar * sqrt(Q / V_bar)
              sigma_bar = per-bar volatility, V_bar = bar volume, Y ~ O(1) is empirical.
              Impact grows with the square root of participation, so doubling the order
              raises the cost per unit by ~41 %, not 100 %.

        Fee = |Q| * P_fill * f,   f = taker fee for market orders (maker for resting limits)
    """

    def __init__(self, cfg: CostConfig) -> None:
        self.cfg = cfg

    def slippage_fraction(self, qty: float = 0.0, bar_volume: float | None = None,
                          bar_sigma: float | None = None) -> float:
        frac = (self.cfg.half_spread_bps + self.cfg.slippage_bps) / 1e4
        if (self.cfg.use_market_impact and qty > 0 and bar_volume and bar_volume > 0
                and bar_sigma is not None and math.isfinite(bar_sigma)):
            participation = min(qty / bar_volume, 1.0)
            frac += self.cfg.impact_coefficient * bar_sigma * math.sqrt(participation)
        return frac

    def fill_price(self, ref_price: float, side: int, qty: float = 0.0,
                   bar_volume: float | None = None, bar_sigma: float | None = None) -> float:
        """Price actually paid (buy, side=+1) or received (sell, side=-1)."""
        return ref_price * (1.0 + side * self.slippage_fraction(qty, bar_volume, bar_sigma))

    def fee(self, notional: float, liquidity: Liquidity = Liquidity.TAKER) -> float:
        rate = self.cfg.taker_fee if liquidity is Liquidity.TAKER else self.cfg.maker_fee
        return abs(notional) * rate

    @property
    def one_way_cost(self) -> float:
        """Proportional cost of one market order, excluding impact (vectorised backtests)."""
        return self.cfg.taker_fee + (self.cfg.half_spread_bps + self.cfg.slippage_bps) / 1e4


# --------------------------------------------------------------------------------------
# Position sizing
# --------------------------------------------------------------------------------------
class PositionSizer(ABC):
    """Risk-based sizing: choose the quantity so that hitting the stop loses f * equity.

        Q = f * E / |P_entry - P_stop|              (units of the base asset)
        subject to  Q * P_entry <= L_max * E        (leverage / notional cap)

    Because the stop distance comes from ATR, volatile markets automatically get
    smaller positions: volatility-normalised risk parity at the trade level.
    """

    def __init__(self, max_position_pct: float) -> None:
        self.max_position_pct = max_position_pct

    @abstractmethod
    def risk_fraction(self) -> float:
        """Fraction f of equity to risk on the next trade."""

    def record_trade(self, r_multiple: float) -> None:  # noqa: B027 - optional hook
        """Feed back the realised R-multiple of a closed trade (used by Kelly)."""

    def size(self, equity: float, entry_price: float, stop_price: float) -> float:
        if equity <= 0 or entry_price <= 0 or not math.isfinite(stop_price):
            return 0.0
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return 0.0
        qty = self.risk_fraction() * equity / risk_per_unit
        max_qty = self.max_position_pct * equity / entry_price
        return max(0.0, min(qty, max_qty))


class FixedFractionalSizer(PositionSizer):
    def __init__(self, fraction: float, max_position_pct: float) -> None:
        super().__init__(max_position_pct)
        self.fraction = fraction

    def risk_fraction(self) -> float:
        return self.fraction


class KellySizer(PositionSizer):
    """Fractional Kelly criterion estimated from the strategy's own recent trades.

    Kelly (1956) maximises the expected log-growth of capital. For a bet that wins
    R units per unit risked with probability W, and otherwise loses 1 unit:

        f* = W - (1 - W) / R

    Trades are measured in R-multiples (PnL / initial risk), so f* is directly the
    fraction of equity to *risk*. W and R come from the last N closed trades.
    Estimation error makes full Kelly dangerous (over-betting is punished far more
    than under-betting), so we use a fraction c (default 1/2): half-Kelly keeps
    about 75 % of the growth rate with half the volatility. The result is capped at
    ``max_fraction``. With no measurable edge (f* <= 0) we keep trading at a small
    "probe" size so that statistics can keep updating.
    """

    def __init__(self, fallback_fraction: float, multiplier: float, lookback: int, min_trades: int,
                 max_fraction: float, max_position_pct: float) -> None:
        super().__init__(max_position_pct)
        self.fallback_fraction = fallback_fraction
        self.multiplier = multiplier
        self.min_trades = min_trades
        self.max_fraction = max_fraction
        self.history: deque[float] = deque(maxlen=lookback)

    def record_trade(self, r_multiple: float) -> None:
        if math.isfinite(r_multiple):
            self.history.append(r_multiple)

    def kelly_fraction(self) -> float | None:
        """Raw (full) Kelly fraction, or None until enough trades exist."""
        if len(self.history) < self.min_trades:
            return None
        r = np.asarray(self.history)
        wins, losses = r[r > 0], r[r < 0]
        if len(wins) == 0:
            return 0.0
        if len(losses) == 0:
            return self.max_fraction
        win_rate = len(wins) / len(r)
        payoff = wins.mean() / abs(losses.mean())
        return win_rate - (1.0 - win_rate) / payoff

    def risk_fraction(self) -> float:
        f_star = self.kelly_fraction()
        if f_star is None:
            return self.fallback_fraction
        if f_star <= 0:
            return self.fallback_fraction * 0.25
        return float(min(self.max_fraction, self.multiplier * f_star))


def build_position_sizer(cfg: RiskConfig) -> PositionSizer:
    if cfg.sizing_method is SizingMethod.KELLY:
        return KellySizer(cfg.risk_per_trade, cfg.kelly_multiplier, cfg.kelly_lookback_trades,
                          cfg.kelly_min_trades, cfg.max_risk_per_trade, cfg.max_position_pct)
    return FixedFractionalSizer(cfg.risk_per_trade, cfg.max_position_pct)


# --------------------------------------------------------------------------------------
# Stops & targets
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProtectiveLevels:
    stop_loss: float
    take_profit: float | None


class StopManager:
    """Initial stop-loss / take-profit placement, trailing and intrabar exit detection."""

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def initial_levels(self, side: int, entry_price: float, atr: float,
                       cloud_top: float = float("nan"), cloud_bottom: float = float("nan")) -> ProtectiveLevels | None:
        """Place protective levels. Returns None if volatility is undefined (skip the trade).

        ATR method:    SL = P -/+ k_sl * ATR           TP = P +/- k_tp * ATR
        Cloud method:  SL = Kumo bottom - b * ATR (long) / Kumo top + b * ATR (short).
                       A trade that breaks back through the cloud has failed, so the
                       far edge of the Kumo is a natural invalidation level.
                       TP = P +/- RR * |P - SL|
        The cloud stop falls back to the ATR stop when it is on the wrong side or
        tighter than 0.5 ATR, and is capped at ``cloud_max_stop_atr`` ATRs.
        """
        c = self.cfg
        if not (math.isfinite(atr) and atr > 0 and math.isfinite(entry_price)):
            return None

        if c.stop_method is StopMethod.CLOUD and math.isfinite(cloud_top) and math.isfinite(cloud_bottom):
            raw_stop = cloud_bottom - c.cloud_buffer_atr * atr if side == LONG else cloud_top + c.cloud_buffer_atr * atr
            distance = side * (entry_price - raw_stop)
            if distance < 0.5 * atr:
                distance = c.atr_sl_multiplier * atr
            distance = min(distance, c.cloud_max_stop_atr * atr)
            stop = entry_price - side * distance
            target = entry_price + side * c.reward_risk_ratio * distance
        else:
            stop = entry_price - side * c.atr_sl_multiplier * atr
            target = entry_price + side * c.atr_tp_multiplier * atr

        if side == LONG:
            stop = max(stop, entry_price * 1e-6)  # a long stop can never be negative
        return ProtectiveLevels(stop_loss=stop, take_profit=target if c.use_take_profit else None)

    def trail(self, side: int, current_stop: float, extreme_price: float, atr: float, kijun: float) -> float:
        """Ratchet the stop in the trade's favour (it never loosens).

        ATR  (Chandelier exit): candidate = extreme since entry -/+ m * ATR
        KIJUN:                  candidate = Kijun-sen (a close through the base line
                                means the medium-term equilibrium has turned)
        """
        method = self.cfg.trailing_method
        if method is TrailingMethod.NONE:
            return current_stop
        if method is TrailingMethod.ATR:
            if not math.isfinite(atr):
                return current_stop
            candidate = extreme_price - side * self.cfg.trailing_atr_multiplier * atr
        else:
            if not math.isfinite(kijun):
                return current_stop
            candidate = kijun
        return max(current_stop, candidate) if side == LONG else min(current_stop, candidate)

    @staticmethod
    def check_intrabar_exit(side: int, stop: float, take_profit: float | None,
                            bar_open: float, bar_high: float, bar_low: float) -> tuple[ExitReason, float] | None:
        """Detect a stop / target hit inside an OHLC bar and return (reason, trigger price).

        OHLC bars hide the path taken inside the bar, so we use conservative rules:
          1. A gap through a level at the open fills at the open (worse than the stop).
          2. If both the stop and the target lie inside the bar's range, assume the
             stop was hit first (pessimistic: it never flatters the backtest).
        """
        if side == LONG:
            if bar_open <= stop:
                return ExitReason.STOP_LOSS, bar_open
            if take_profit is not None and bar_open >= take_profit:
                return ExitReason.TAKE_PROFIT, bar_open
            if bar_low <= stop:
                return ExitReason.STOP_LOSS, stop
            if take_profit is not None and bar_high >= take_profit:
                return ExitReason.TAKE_PROFIT, take_profit
        elif side == SHORT:
            if bar_open >= stop:
                return ExitReason.STOP_LOSS, bar_open
            if take_profit is not None and bar_open <= take_profit:
                return ExitReason.TAKE_PROFIT, bar_open
            if bar_high >= stop:
                return ExitReason.STOP_LOSS, stop
            if take_profit is not None and bar_low <= take_profit:
                return ExitReason.TAKE_PROFIT, take_profit
        return None


# --------------------------------------------------------------------------------------
# Portfolio-level circuit breaker
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BreakerEvent:
    timestamp: pd.Timestamp
    reason: str
    equity: float
    drawdown: float
    resume_at: pd.Timestamp


class CircuitBreaker:
    """Kill switch that flattens the book and blocks new entries after large losses.

    Drawdown from the high-water mark:   DD_t = E_t / max_{s<=t} E_s - 1
        DD_t <= -DD_max        -> halt for ``cooldown``; on resume the high-water mark
                                  resets to current equity (otherwise the breaker would
                                  re-trip immediately and never let the system trade).
    Daily loss:                 E_t / E_{UTC day open} - 1 <= -L_daily -> halt until the next UTC day.
    """

    def __init__(self, max_drawdown: float, cooldown: pd.Timedelta, daily_loss_limit: float) -> None:
        self.max_drawdown = max_drawdown
        self.cooldown = cooldown
        self.daily_loss_limit = daily_loss_limit
        self.events: list[BreakerEvent] = []
        self._peak: float | None = None
        self._day: pd.Timestamp | None = None
        self._day_open_equity: float | None = None
        self._halted_until: pd.Timestamp | None = None

    @classmethod
    def from_config(cls, cfg: RiskConfig, timeframe: str) -> CircuitBreaker:
        cooldown = pd.Timedelta(minutes=timeframe_to_minutes(timeframe) * cfg.circuit_breaker_cooldown_bars)
        return cls(cfg.max_drawdown_limit, cooldown, cfg.daily_loss_limit)

    def is_halted(self, ts: pd.Timestamp) -> bool:
        return self._halted_until is not None and ts < self._halted_until

    def update(self, ts: pd.Timestamp, equity: float) -> BreakerEvent | None:
        """Feed the latest marked-to-market equity; returns an event if the breaker trips now."""
        if self._halted_until is not None and ts >= self._halted_until:
            logger.info("Circuit breaker released at %s; high-water mark reset to %.2f", ts, equity)
            self._halted_until = None
            self._peak = equity

        day = ts.normalize()
        if day != self._day:
            self._day, self._day_open_equity = day, equity

        self._peak = equity if self._peak is None else max(self._peak, equity)
        if self._halted_until is not None:
            return None

        drawdown = equity / self._peak - 1.0 if self._peak > 0 else 0.0
        if self.max_drawdown > 0 and drawdown <= -self.max_drawdown:
            return self._trip(ts, "max_drawdown", equity, drawdown, ts + self.cooldown)
        if self.daily_loss_limit > 0 and self._day_open_equity:
            daily = equity / self._day_open_equity - 1.0
            if daily <= -self.daily_loss_limit:
                return self._trip(ts, "daily_loss_limit", equity, daily, day + pd.Timedelta(days=1))
        return None

    def _trip(self, ts: pd.Timestamp, reason: str, equity: float, loss: float,
              resume_at: pd.Timestamp) -> BreakerEvent:
        self._halted_until = resume_at
        event = BreakerEvent(ts, reason, equity, loss, resume_at)
        self.events.append(event)
        logger.warning("CIRCUIT BREAKER (%s) at %s: equity=%.2f loss=%.2f%% - trading halted until %s",
                       reason, ts, equity, 100 * loss, resume_at,
                       extra={"event": "circuit_breaker", "reason": reason, "equity": equity})
        return event
