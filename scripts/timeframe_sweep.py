"""Timeframe / settings selection sweep - the evidence behind the defaults in config.py.

For one symbol it resamples a cached 1h history into 1h..1d candles and backtests every
combination of:

    timeframe   x  Ichimoku preset  x  TK-cross window  x  exit style  x  stop width

Each run is split chronologically into in-sample (first 70 %) and out-of-sample (last 30 %)
and both halves are reported. Settings are chosen on *out-of-sample* behaviour and on being
positive in both halves - picking the single best in-sample cell is curve fitting, and the
table shows exactly how badly those cells fall apart on unseen data.

    python scripts/timeframe_sweep.py                 # BTC/USDT, writes reports/timeframe_sweep.csv
    python scripts/timeframe_sweep.py ETH/USDT
"""

from __future__ import annotations

import itertools
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics import max_drawdown, sharpe_ratio  # noqa: E402
from backtester import BacktestError, EventDrivenBacktester  # noqa: E402
from config import (ICHIMOKU_PRESETS, MARKET_DB_PATH, REPORT_DIR, AppConfig, DataConfig,  # noqa: E402
                    RiskConfig, StrategyConfig, TrailingMethod, periods_per_year)
from data_loader import OHLCVStore  # noqa: E402

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
RESAMPLE_RULE = {"1h": None, "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D"}
EXIT_STYLES = {
    "tp + no trail": dict(use_take_profit=True, trailing_method=TrailingMethod.NONE),
    "no tp + kijun": dict(use_take_profit=False, trailing_method=TrailingMethod.KIJUN),
    "no tp + atr": dict(use_take_profit=False, trailing_method=TrailingMethod.ATR),
    "tp + kijun": dict(use_take_profit=True, trailing_method=TrailingMethod.KIJUN),
}
PRESETS = ["standard", "crypto", "crypto_slow", "dynamic"]
CROSS_WINDOWS = [1, 3]
STOP_WIDTHS = [2.0, 4.0]
TRAIN_FRACTION = 0.7


def run(symbol: str = "BTC/USDT", exchange: str = "binance") -> pd.DataFrame:
    base = OHLCVStore(MARKET_DB_PATH).load(exchange, symbol, "1h")
    if base.empty:
        raise SystemExit(f"No cached 1h data for {symbol}. Run: python main.py fetch --symbol {symbol} "
                         "--timeframe 1h --days 1100")
    print(f"{symbol}: {len(base)} 1h bars {base.index[0]:%Y-%m-%d} -> {base.index[-1]:%Y-%m-%d}", flush=True)

    rows, started = [], time.perf_counter()
    for timeframe, rule in RESAMPLE_RULE.items():
        df = base if rule is None else base.resample(rule, label="left", closed="left").agg(AGG).dropna()
        split = int(len(df) * TRAIN_FRACTION)
        ppy = periods_per_year(timeframe)
        for preset, cross, (exit_name, exit_kw), stop in itertools.product(
                PRESETS, CROSS_WINDOWS, EXIT_STYLES.items(), STOP_WIDTHS):
            strategy = replace(StrategyConfig(), cross_lookback=cross, dynamic=(preset == "dynamic"),
                               **({} if preset == "dynamic" else {"params": ICHIMOKU_PRESETS[preset]}))
            risk = replace(RiskConfig(), atr_sl_multiplier=stop, atr_tp_multiplier=2 * stop, **exit_kw)
            cfg = AppConfig(data=replace(DataConfig(), symbol=symbol, timeframe=timeframe),
                            strategy=strategy, risk=risk)
            try:
                result = EventDrivenBacktester(cfg).run(df)
            except BacktestError:
                continue  # not enough bars for this preset on this timeframe
            curve, trades = result.equity_curve, result.trades

            def segment(start: int, end: int) -> tuple[float, float, float, float]:
                equity = curve["equity"].iloc[start:end]
                bench = curve["benchmark_equity"].iloc[start:end]
                return (equity.iloc[-1] / equity.iloc[0] - 1, sharpe_ratio(curve["returns"].iloc[start:end], ppy),
                        max_drawdown(equity), bench.iloc[-1] / bench.iloc[0] - 1)

            is_ret, is_sharpe, _, is_bh = segment(0, split)
            oos_ret, oos_sharpe, oos_dd, oos_bh = segment(split, len(curve))
            full_ret, full_sharpe, full_dd, full_bh = segment(0, len(curve))
            entries = pd.to_datetime(trades["entry_time"], utc=True) if len(trades) else pd.Series(dtype="datetime64[ns, UTC]")
            rows.append({
                "timeframe": timeframe, "preset": preset, "cross_lb": cross, "exit": exit_name, "sl_atr": stop,
                "trades": len(trades), "win_rate": (trades["net_pnl"] > 0).mean() if len(trades) else float("nan"),
                "exposure": float((curve["position"] != 0).mean()),
                "is_ret": is_ret, "is_sharpe": is_sharpe, "is_bh": is_bh,
                "oos_ret": oos_ret, "oos_sharpe": oos_sharpe, "oos_dd": oos_dd, "oos_bh": oos_bh,
                "oos_trades": int((entries >= curve.index[split]).sum()),
                "oos_excess": oos_ret - oos_bh,
                "full_ret": full_ret, "full_sharpe": full_sharpe, "full_dd": full_dd, "full_bh": full_bh,
            })
        print(f"  {timeframe}: {len(df)} bars ({time.perf_counter() - started:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    out = REPORT_DIR / f"timeframe_sweep_{symbol.replace('/', '')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n{len(table)} runs -> {out}")
    return table


def summarise(table: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pct = lambda s: (s * 100).round(1)  # noqa: E731
    print("\n=== Median out-of-sample return (%) by timeframe ===")
    print(pct(table.groupby("timeframe")["oos_ret"].median()).to_string())
    print("\n=== Median out-of-sample return (%) by exit style ===")
    print(pct(table.groupby("exit")["oos_ret"].median()).sort_values(ascending=False).to_string())
    keep = ["timeframe", "preset", "cross_lb", "exit", "sl_atr", "trades", "is_ret", "oos_ret", "full_ret",
            "oos_sharpe", "full_dd"]
    both = table[(table["is_ret"] > 0) & (table["oos_ret"] > 0) & (table["oos_trades"] >= 5)].copy()
    for col in ("is_ret", "oos_ret", "full_ret", "full_dd"):
        both[col] = pct(both[col])
    print("\n=== Positive in BOTH halves (the only candidates worth trusting), best full-period first ===")
    print(both.sort_values("full_ret", ascending=False)[keep].head(12).to_string(index=False))


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    summarise(run(sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"))
