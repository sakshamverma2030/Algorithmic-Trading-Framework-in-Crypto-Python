"""Cross-strategy comparison on a single real dataset.

Runs every candidate single-symbol strategy through the event-driven
backtester on the same OHLCV frame and prints a scorecard, writes
reports/<SYMBOL>_<TF>_<source>/strategy_comparison.csv and equity-chart png.

Usage:
    python scripts/compare_all.py --symbol BTC/USDT --timeframe 1h --days 180
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig
from data_loader import DataLoader
from strategy import IchimokuStrategy
from momentum import MomentumStrategy
from intermediate_strategies import CalendarAnomalyStrategy, AroonRsiDivergenceStrategy
from advanced_ml_strategies import HurstTrendStrategy
from backtester import EventDrivenBacktester
from analytics import PerformanceAnalyzer
from main import build_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=180)
    args = p.parse_args()

    cfg: AppConfig = build_config(["backtest", "--source", "exchange", "--symbol", args.symbol,
                                   "--timeframe", args.timeframe, "--days", str(args.days),
                                   "--no-plots"])
    df = DataLoader(cfg).load(refresh=False)
    if df.empty:
        print(f"no data for {args.symbol} {args.timeframe}; run fetch first")
        return

    strategies = {
        "ichimoku": IchimokuStrategy(cfg.strategy),
        "momentum": MomentumStrategy(cfg.momentum),
        "calendar": CalendarAnomalyStrategy(cfg.calendar),
        "divergence": AroonRsiDivergenceStrategy(cfg.divergence),
        "hurst": HurstTrendStrategy(cfg.hurst),
    }

    analyzer = PerformanceAnalyzer()
    rows: list[dict] = []
    curves: dict[str, pd.Series] = {}
    for key, strat in strategies.items():
        result = EventDrivenBacktester(cfg, strat).run(df)
        report = analyzer.analyze(result)
        s, t = report.strategy, report.trades
        rows.append({
            "strategy": key,
            "label": report.name,
            "final_equity": s["final_equity"],
            "total_return": s["total_return"],
            "cagr": s["cagr"],
            "sharpe": s["sharpe"],
            "max_drawdown": s["max_drawdown"],
            "trades": t["total_trades"],
            "win_rate": t["win_rate"],
            "profit_factor": t["profit_factor"],
        })
        curves[key] = result.equity_curve["equity"]

    score = pd.DataFrame(rows).set_index("strategy")
    print(score[["label", "final_equity", "total_return", "sharpe", "max_drawdown",
                 "trades", "win_rate", "profit_factor"]].to_string())

    out_dir = cfg.backtest.report_dir / f"{args.symbol.replace('/', '')}_{args.timeframe}_exchange"
    out_dir.mkdir(parents=True, exist_ok=True)
    score.to_csv(out_dir / "strategy_comparison.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    for key, curve in curves.items():
        ax.plot(curve.index, curve / curve.iloc[0], label=key, linewidth=1.2)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(f"Strategy equity (normalised) - {args.symbol} {args.timeframe}")
    ax.set_ylabel("growth of 1.00")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out_dir / "strategy_comparison_equity.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"\nSaved {score.shape[0]} rows -> {out_dir / 'strategy_comparison.csv'}")
    print(f"Chart -> {png}")


if __name__ == "__main__":
    main()