"""
main.py - Interactive CLI and pipeline orchestrator for the Ichimoku crypto trading system.

    Data fetch -> vectorised backtest -> event-driven backtest -> analytics & charts -> paper trader

Examples
--------
    python main.py                                    interactive menu
    python main.py pipeline --source synthetic        full pipeline offline (no exchange needed)
    python main.py fetch --symbol ETH/USDT --timeframe 15m --days 180
    python main.py backtest --preset crypto --sizing kelly --stop cloud --trailing kijun
    python main.py compare                            standard vs crypto vs crypto_slow vs dynamic
    python main.py optimize --metric sharpe           in-sample / out-of-sample grid search
    python main.py live --feed ccxtpro                paper trading on live Binance WebSocket data
    python main.py live --live-strategy momentum       paper trade the momentum strategy instead
    python main.py live --feed replay --source synthetic   offline paper-trading replay
    python main.py live --mode LIVE_TRADING           real orders (testnet unless EXCHANGE_USE_TESTNET=false)
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence


def _preflight() -> None:
    """Fail fast, with an actionable message, when a core dependency cannot be imported."""
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:
        details = traceback.format_exc()
        hint = ""
        if "DLL load failed" in details or "Application Control" in details:
            hint = (
                "\nHint: Windows (Smart App Control) blocked a package's native library. This is not a bug in\n"
                "the project, and it usually hits freshly downloaded copies of a package.\n"
                "  * Inside a virtual environment? Run `deactivate` and use the Python installation\n"
                "    whose packages already load.\n"
            )
            if "pyarrow" in details:
                hint += "  * pyarrow is optional here:  python -m pip uninstall -y pyarrow\n"
            hint += "  * Still blocked? Run the project under WSL (Ubuntu on Windows) or on a Linux server.\n"
        sys.stderr.write(f"Failed to import a core dependency: {exc}\n{hint}\n"
                         "Install the requirements with:  python -m pip install -r requirements.txt\n")
        raise SystemExit(1) from exc


_preflight()

import pandas as pd  # noqa: E402

from analytics import ChartVisualizer, PerformanceAnalyzer, PerformanceReport  # noqa: E402
from backtester import (  # noqa: E402
    BacktestError,
    BacktestResult,
    EventDrivenBacktester,
    ParameterOptimizer,
    VectorizedBacktester,
    compare_presets,
    slugify,
)
from config import (  # noqa: E402
    ICHIMOKU_PRESETS,
    TIMEFRAME_MINUTES,
    AppConfig,
    ConfigError,
    DataSource,
    FeedType,
    IchimokuParams,
    RegimeMethod,
    SizingMethod,
    StopMethod,
    StrategyKind,
    TradingMode,
    TrailingMethod,
    load_config,
    setup_logging,
)
from data_loader import DataLoader, DataLoaderError  # noqa: E402
from live_trader import LiveTradingError, build_strategy, run_async, run_live_session, warmup_bars_needed  # noqa: E402
from momentum import MomentumStrategy  # noqa: E402
from pairs import CointegrationAnalyzer, PairsBacktester  # noqa: E402
from strategy import IchimokuStrategy  # noqa: E402

logger = logging.getLogger("main")

COMMANDS = ("menu", "fetch", "backtest", "compare", "optimize", "momentum", "pairs", "live", "pipeline")


# ======================================================================================
# CLI arguments -> AppConfig
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("market data")
    g.add_argument("--symbol", help="trading pair, e.g. BTC/USDT")
    g.add_argument("--timeframe", choices=list(TIMEFRAME_MINUTES), help="candle size")
    g.add_argument("--days", type=int, help="history length in days")
    g.add_argument("--source", choices=[s.value for s in DataSource], help="data source")
    g.add_argument("--csv-path", type=Path, help="OHLCV CSV file for --source csv")
    g.add_argument("--exchange", help="CCXT exchange id (default: EXCHANGE_ID or binance)")
    g.add_argument("--seed", type=int, help="random seed of the synthetic generator")

    s = common.add_argument_group("strategy")
    s.add_argument("--preset", choices=[*ICHIMOKU_PRESETS, "dynamic"], help="Ichimoku parameter preset")
    s.add_argument("--tenkan", type=int, help="custom Tenkan period")
    s.add_argument("--kijun", type=int, help="custom Kijun period")
    s.add_argument("--senkou-b", type=int, help="custom Senkou Span B period")
    s.add_argument("--displacement", type=int, help="custom Kumo displacement")
    s.add_argument("--allow-short", action="store_true", help="enable short entries (perps / margin only)")
    s.add_argument("--cross-lookback", type=int, help="bars a TK cross stays valid (default 1)")
    s.add_argument("--no-chikou", action="store_true", help="disable the Chikou confirmation rule")
    s.add_argument("--kumo-twist", action="store_true", help="require the leading cloud to match the trade")
    s.add_argument("--rsi", action="store_true", help="RSI overlay: block overbought entries, exit on RSI exhaustion")
    s.add_argument("--rsi-period", type=int, help="RSI smoothing period (default 14)")
    s.add_argument("--rsi-overbought", type=float, help="RSI overbought threshold (default 70)")
    s.add_argument("--rsi-oversold", type=float, help="RSI oversold threshold (default 30)")
    s.add_argument("--rsi-divergence", action="store_true", help="only enter on a causal RSI/price divergence")
    s.add_argument("--divergence-lookback", type=int, help="divergence detection window (default 10)")
    s.add_argument("--regime", choices=[m.value for m in RegimeMethod], help="dynamic regime detector")

    r = common.add_argument_group("risk & execution costs")
    r.add_argument("--capital", type=float, help="initial capital in quote currency")
    r.add_argument("--sizing", choices=[m.value for m in SizingMethod])
    r.add_argument("--risk-per-trade", type=float, help="equity fraction risked per trade, e.g. 0.01")
    r.add_argument("--stop", choices=[m.value for m in StopMethod], help="stop-loss method")
    r.add_argument("--trailing", choices=[m.value for m in TrailingMethod], help="trailing-stop method")
    r.add_argument("--no-take-profit", action="store_true")
    r.add_argument("--max-dd", type=float, help="circuit-breaker drawdown limit, e.g. 0.2")
    r.add_argument("--fee", type=float, help="maker and taker fee, e.g. 0.001 = 0.1%%")
    r.add_argument("--slippage-bps", type=float, help="fixed slippage in basis points")

    o = common.add_argument_group("output")
    o.add_argument("--no-plots", action="store_true", help="skip chart generation")
    o.add_argument("--show", action="store_true", help="open chart windows as well as saving PNGs")
    o.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Ichimoku Cloud crypto trading system: data, backtests, analytics and paper/live trading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[1] if __doc__ else None,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.add_parser("menu", parents=[common], help="interactive menu (default)")
    sub.add_parser("fetch", parents=[common], help="download / update OHLCV data")
    bt = sub.add_parser("backtest", parents=[common], help="run backtests and produce the report and charts")
    bt.add_argument("--engine", choices=["event", "vectorised", "both"], default="both")
    sub.add_parser("compare", parents=[common], help="compare Ichimoku presets")
    opt = sub.add_parser("optimize", parents=[common], help="in-sample / out-of-sample parameter grid search")
    opt.add_argument("--metric", choices=["sharpe", "sortino", "cagr"], default="sharpe")
    mom = sub.add_parser("momentum", parents=[common], help="backtest the long/short momentum strategy")
    mom.add_argument("--engine", choices=["event", "vectorised", "both"], default="both")
    mom.add_argument("--mom-lookback", type=int, help="momentum window in bars (default 20)")
    mom.add_argument("--mom-entry", type=float, help="lookback return to enter, e.g. 0.02 = 2%%")
    mom.add_argument("--mom-exit", type=float, help="momentum below which the long closes")
    mom.add_argument("--mom-sma", type=int, help="SMA trend-filter period (default 50)")
    mom.add_argument("--mom-require-sma", action="store_true", help="long entries need Close > SMA")
    mom.add_argument("--mom-short", action="store_true", help="allow symmetric short entries")
    pr = sub.add_parser("pairs", parents=[common], help="statistical-arbitrage backtest of two symbols")
    pr.add_argument("--pair", help="second symbol, e.g. ETH/USDT")
    pr.add_argument("--pairs-lookback", type=int, help="rolling cointegration window (default 60)")
    pr.add_argument("--entry-z", type=float, help="z-score to open a spread position (default 2.0)")
    pr.add_argument("--exit-z", type=float, help="z-score to close (mean reversion completed)")
    pr.add_argument("--stop-z", type=float, help="z-score stop-loss (default 3.5)")
    pr.add_argument("--max-hold", type=int, help="bars after which an open spread is force-closed")
    pr.add_argument("--min-corr", type=float, help="minimum rolling correlation to open")
    pr.add_argument("--raw-prices", action="store_true", help="cointegrate raw prices instead of log prices")
    live = sub.add_parser("live", parents=[common], help="launch the real-time paper / live trader")
    _add_live_args(live)
    pipe = sub.add_parser("pipeline", parents=[common], help="fetch -> backtest -> charts -> paper trader")
    _add_live_args(pipe)
    pipe.add_argument("--launch-live", choices=["ask", "yes", "no"], default="ask",
                      help="start the paper trader at the end of the pipeline")
    return parser


def _add_live_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=[m.value for m in TradingMode], help="PAPER_TRADING (default) or LIVE_TRADING")
    p.add_argument("--feed", choices=[f.value for f in FeedType], help="market-data feed")
    p.add_argument("--replay-bars", type=int, help="bars streamed by the replay feed")
    p.add_argument("--speed", type=float, help="seconds between replayed bars")
    p.add_argument("--flatten-on-exit", action="store_true", help="close open positions on shutdown")
    p.add_argument("--confirm-live", action="store_true", help="skip the interactive LIVE_TRADING confirmation")
    p.add_argument("--live-strategy", choices=[s.value for s in StrategyKind], help="strategy for the live engine")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    argv = list(argv)
    if not any(a in COMMANDS for a in argv) and not any(a in ("-h", "--help") for a in argv):
        argv.insert(0, "menu")
    return build_parser().parse_args(argv)


def build_config(args: argparse.Namespace) -> AppConfig:
    """Apply command-line overrides on top of the defaults in config.py."""
    cfg = load_config()
    opt = lambda name: getattr(args, name, None)  # noqa: E731 - some flags exist on some sub-commands only

    exchange = replace(cfg.exchange, exchange_id=opt("exchange")) if opt("exchange") else cfg.exchange
    data = cfg.data
    for field_name, value in (("symbol", opt("symbol") and opt("symbol").upper()), ("timeframe", opt("timeframe")),
                              ("history_days", opt("days")), ("csv_path", opt("csv_path")),
                              ("synthetic_seed", opt("seed"))):
        if value is not None:
            data = replace(data, **{field_name: value})
    if opt("source"):
        data = replace(data, source=DataSource(opt("source")))
    elif opt("csv_path"):
        data = replace(data, source=DataSource.CSV)

    strategy = cfg.strategy
    if opt("preset") == "dynamic":
        strategy = replace(strategy, dynamic=True)
    elif opt("preset"):
        strategy = replace(strategy, params=ICHIMOKU_PRESETS[opt("preset")], dynamic=False)
    if any(opt(k) for k in ("tenkan", "kijun", "senkou_b", "displacement")):
        p = strategy.params
        strategy = replace(strategy, dynamic=False, params=IchimokuParams(
            opt("tenkan") or p.tenkan, opt("kijun") or p.kijun, opt("senkou_b") or p.senkou_b,
            opt("displacement") or p.displacement))
    strategy = replace(
        strategy,
        allow_short=bool(opt("allow_short")) or strategy.allow_short,
        cross_lookback=opt("cross_lookback") or strategy.cross_lookback,
        require_chikou=strategy.require_chikou and not opt("no_chikou"),
        require_kumo_twist=bool(opt("kumo_twist")) or strategy.require_kumo_twist,
    )
    strategy = replace(
        strategy,
        use_rsi_filter=bool(opt("rsi")) or strategy.use_rsi_filter,
        rsi_period=opt("rsi_period") or strategy.rsi_period,
        rsi_overbought=opt("rsi_overbought") or strategy.rsi_overbought,
        rsi_oversold=opt("rsi_oversold") or strategy.rsi_oversold,
        use_rsi_divergence=bool(opt("rsi_divergence")) or strategy.use_rsi_divergence,
        divergence_lookback=opt("divergence_lookback") or strategy.divergence_lookback,
    )
    if opt("regime"):
        strategy = replace(strategy, regime_method=RegimeMethod(opt("regime")))

    risk = cfg.risk
    if opt("capital"):
        risk = replace(risk, initial_capital=opt("capital"))
    if opt("sizing"):
        risk = replace(risk, sizing_method=SizingMethod(opt("sizing")))
    if opt("risk_per_trade"):
        risk = replace(risk, risk_per_trade=opt("risk_per_trade"))
    if opt("stop"):
        risk = replace(risk, stop_method=StopMethod(opt("stop")))
    if opt("trailing"):
        risk = replace(risk, trailing_method=TrailingMethod(opt("trailing")))
    if opt("no_take_profit"):
        risk = replace(risk, use_take_profit=False)
    if opt("max_dd") is not None:
        risk = replace(risk, max_drawdown_limit=opt("max_dd"))

    costs = cfg.costs
    if opt("fee") is not None:
        costs = replace(costs, maker_fee=opt("fee"), taker_fee=opt("fee"))
    if opt("slippage_bps") is not None:
        costs = replace(costs, slippage_bps=opt("slippage_bps"))

    live = cfg.live
    if opt("mode"):
        live = replace(live, mode=TradingMode(opt("mode")))
    if opt("feed"):
        live = replace(live, feed=FeedType(opt("feed")))
    elif data.source is not DataSource.EXCHANGE:
        live = replace(live, feed=FeedType.REPLAY)  # offline data can only be replayed
    if opt("replay_bars"):
        live = replace(live, replay_bars=opt("replay_bars"))
    if opt("speed") is not None:
        live = replace(live, replay_speed=opt("speed"))
    if opt("flatten_on_exit"):
        live = replace(live, flatten_on_exit=True)
    if opt("live_strategy"):
        live = replace(live, live_strategy=StrategyKind(opt("live_strategy")))

    momentum = replace(
        cfg.momentum,
        lookback=opt("mom_lookback") or cfg.momentum.lookback,
        entry_threshold=opt("mom_entry") if opt("mom_entry") is not None else cfg.momentum.entry_threshold,
        exit_threshold=opt("mom_exit") if opt("mom_exit") is not None else cfg.momentum.exit_threshold,
        sma_period=opt("mom_sma") or cfg.momentum.sma_period,
        require_above_sma=bool(opt("mom_require_sma")) or cfg.momentum.require_above_sma,
        allow_short=bool(opt("mom_short")) or cfg.momentum.allow_short,
    )
    pairs = replace(
        cfg.pairs,
        base_symbol=data.symbol,
        quote_symbol=opt("pair") or cfg.pairs.quote_symbol,
        lookback=opt("pairs_lookback") or cfg.pairs.lookback,
        entry_zscore=opt("entry_z") if opt("entry_z") is not None else cfg.pairs.entry_zscore,
        exit_zscore=opt("exit_z") if opt("exit_z") is not None else cfg.pairs.exit_zscore,
        stop_zscore=opt("stop_z") if opt("stop_z") is not None else cfg.pairs.stop_zscore,
        max_hold_bars=opt("max_hold") or cfg.pairs.max_hold_bars,
        min_correlation=opt("min_corr") if opt("min_corr") is not None else cfg.pairs.min_correlation,
        use_log_prices=cfg.pairs.use_log_prices and not bool(opt("raw_prices")),
    )

    return AppConfig(exchange=exchange, data=data, strategy=strategy, costs=costs, risk=risk,
                     backtest=cfg.backtest, live=live, momentum=momentum, pairs=pairs)


# ======================================================================================
# Application
# ======================================================================================
def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n {text}\n" + "=" * 78)


class TradingBotCLI:
    """Runs each stage of the pipeline and keeps the loaded dataset between menu actions."""

    def __init__(self, cfg: AppConfig, plots: bool = True, show: bool = False) -> None:
        self.cfg = cfg
        self.plots = plots
        self.show = show
        self.analyzer = PerformanceAnalyzer()
        self._data: pd.DataFrame | None = None
        self._data_key: tuple[Any, ...] | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def report_dir(self) -> Path:
        d = self.cfg.data
        return self.cfg.backtest.report_dir / f"{d.symbol.replace('/', '')}_{d.timeframe}_{d.source.value}"

    def _dataset_key(self) -> tuple[Any, ...]:
        d = self.cfg.data
        return (d.symbol, d.timeframe, d.history_days, d.source, d.csv_path, d.synthetic_seed,
                self.cfg.exchange.exchange_id)

    def data(self) -> pd.DataFrame:
        if self._data is None or self._data_key != self._dataset_key():
            self.fetch()
        assert self._data is not None
        return self._data

    def _visualizer(self) -> ChartVisualizer:
        return ChartVisualizer(self.report_dir, show=self.show)

    # ------------------------------------------------------------------ stages
    def fetch(self, refresh: bool = True) -> pd.DataFrame:
        d = self.cfg.data
        banner(f"DATA: {d.symbol} {d.timeframe} | {d.history_days} days | source={d.source.value}")
        loader = DataLoader(self.cfg)
        df = loader.load(refresh=refresh)
        self._data, self._data_key = df, self._dataset_key()
        print(f" {len(df):,} bars from {df.index[0]:%Y-%m-%d %H:%M} to {df.index[-1]:%Y-%m-%d %H:%M} UTC")
        print(f" last close {df['close'].iloc[-1]:,.2f} | period return "
              f"{df['close'].iloc[-1] / df['close'].iloc[0] - 1:+.2%}")
        if d.source is DataSource.EXCHANGE:
            paths = loader.export(df)
            print(f" cached in SQLite: {loader.store.db_path} | CSV export: {paths['csv']}")
        if d.source is DataSource.SYNTHETIC:
            print(" NOTE: synthetic regime-switching GBM data - use it to test the system, not to judge the strategy.")
        return df

    def backtest(self, engine: str = "both") -> tuple[BacktestResult, PerformanceReport]:
        result, report = self._run_backtest(IchimokuStrategy(self.cfg.strategy), engine)
        if self.plots:
            charts = self._visualizer().create_all(result, report)
            print(" Charts:")
            for path in charts:
                print(f"   {path}")
        return result, report

    def _run_backtest(self, strategy: Any, engine: str = "both") -> tuple[BacktestResult, PerformanceReport]:
        df = self.data()
        banner(f"BACKTEST: {strategy.name} | sizing={self.cfg.risk.sizing_method.value} "
               f"stop={self.cfg.risk.stop_method.value} trailing={self.cfg.risk.trailing_method.value}")
        result: BacktestResult | None = None
        report: PerformanceReport | None = None
        if engine in ("vectorised", "both"):
            v_result = VectorizedBacktester(self.cfg, strategy).run(df)
            v_report = self.analyzer.analyze(v_result)
            print(f" [vectorised, full notional, no stops] {v_report.headline()}")
            result, report = v_result, v_report
        if engine in ("event", "both"):
            result = EventDrivenBacktester(self.cfg, strategy).run(df)
            report = self.analyzer.analyze(result)
        assert result is not None and report is not None
        print(report.format())

        out = self.report_dir
        saved = result.save(out)
        saved.update(report.save(out, slugify(result.name)))
        print(f"\n Tables saved to {out}:")
        for label, path in saved.items():
            print(f"   {label:<13} {path.name}")
        return result, report

    def momentum(self, engine: str = "both") -> tuple[BacktestResult, PerformanceReport]:
        result, report = self._run_backtest(MomentumStrategy(self.cfg.momentum), engine)
        if self.plots:
            viz = self._visualizer()
            print(" Charts:")
            for path in (viz.plot_performance_dashboard(result, report),
                         viz.plot_simple(result, "momentum_chart.png")):
                print(f"   {path}")
        return result, report

    def pairs_backtest(self) -> BacktestResult:
        cfg = self.cfg
        base = self.data()
        banner(f"PAIRS BACKTEST: {cfg.pairs.label} | capital={cfg.risk.initial_capital:,.0f}")
        quote_data = DataLoader(replace(cfg, data=replace(cfg.data, symbol=cfg.pairs.quote_symbol))).load(refresh=True)
        diag = CointegrationAnalyzer.diagnostics(base, quote_data, cfg.pairs.lookback)
        print(f" Pair fitness: {diag['bars']:,} aligned bars | return corr {diag['return_correlation']:.2f} | "
              f"beta {diag['hedge_beta_mean']:.3f} (+/-{diag['hedge_beta_std']:.3f}) | "
              f"half-life {diag['half_life_bars']:.0f} bars")

        result = PairsBacktester(cfg).run(base, quote_data)
        report = self.analyzer.analyze(result)
        print(report.format())

        out = self.report_dir
        saved = result.save(out)
        saved.update(report.save(out, slugify(result.name)))
        print(f"\n Tables saved to {out}:")
        for label, path in saved.items():
            print(f"   {label:<13} {path.name}")
        if self.plots:
            viz = self._visualizer()
            print(" Charts:")
            for path in (viz.plot_performance_dashboard(result, report),
                         viz.plot_pairs(result, "pairs_chart.png")):
                print(f"   {path}")
        return result

    def compare(self) -> pd.DataFrame:
        df = self.data()
        banner("PRESET COMPARISON (event-driven, identical risk & cost settings)")
        results = compare_presets(df, self.cfg)
        table = self.analyzer.compare(results)
        shown = table.drop(columns=["strategy"]).copy()
        for col in ("total_return", "cagr", "max_drawdown"):
            shown[col] = shown[col].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "n/a")
        for col in ("volatility", "win_rate", "exposure"):
            shown[col] = shown[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "n/a")
        for col in ("sharpe", "sortino", "calmar", "profit_factor"):
            shown[col] = shown[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(shown.to_string())
        self.report_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.report_dir / "preset_comparison.csv")
        if self.plots:
            print(f" Chart: {self._visualizer().plot_preset_comparison(results, table)}")
        return table

    def optimize(self, metric: str = "sharpe") -> None:
        df = self.data()
        banner(f"PARAMETER OPTIMISATION ({metric}, vectorised engine, "
               f"{self.cfg.backtest.train_fraction:.0%} in-sample / {1 - self.cfg.backtest.train_fraction:.0%} "
               "out-of-sample)")
        result = ParameterOptimizer(self.cfg).grid_search(df, metric=metric)
        cols = ["tenkan", "kijun", "senkou_b", f"is_{metric}", "is_cagr", "is_max_dd", "is_trades",
                f"oos_{metric}", "oos_cagr", "oos_max_dd", "oos_trades"]
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
            print(result.table[cols].head(10).to_string(index=False))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        result.table.to_csv(self.report_dir / f"optimization_{metric}.csv", index=False)
        best = result.table.iloc[0]
        print(f"\n Best in-sample parameters: {result.best_params.label} -> IS {metric} {best[f'is_{metric}']:.3f}, "
              f"OOS {metric} {best[f'oos_{metric}']:.3f}")
        print(" A large drop from IS to OOS indicates overfitting; prefer robust regions of the heatmap over the "
              "single best cell.")
        if self.plots:
            print(f" Chart: {self._visualizer().plot_optimization_heatmap(result)}")

    def live(self, confirm_live: bool = False) -> None:
        cfg = self.cfg
        if cfg.live.mode is TradingMode.LIVE_TRADING and not self._confirm_live(confirm_live):
            print(" LIVE_TRADING cancelled.")
            return
        replay = None
        if cfg.live.feed is FeedType.REPLAY:
            replay = self.data().tail(warmup_bars_needed(cfg) + cfg.live.replay_bars)
        banner(f"REAL-TIME ENGINE: {cfg.live.mode.value} | feed={cfg.live.feed.value} | "
               f"{build_strategy(cfg).name} | {cfg.data.symbol} {cfg.data.timeframe}  (Ctrl+C to stop)")
        if cfg.live.feed is not FeedType.REPLAY:
            print(f" Signals are evaluated on closed {cfg.data.timeframe} candles; the first decision arrives when "
                  "the current candle closes. Stops are monitored on every order-book update.")
        try:
            summary = run_async(run_live_session(cfg, replay))
        except KeyboardInterrupt:
            print("\n Stopped by user (see logs/trading_bot.jsonl and the trade journal).")
            return
        print(summary.format())
        print(f" Journal: {cfg.live.journal_db_path} (run_id={summary.run_id})")

    def _confirm_live(self, pre_confirmed: bool) -> bool:
        ex = self.cfg.exchange
        if not ex.has_credentials:
            raise ConfigError("LIVE_TRADING needs EXCHANGE_API_KEY / EXCHANGE_API_SECRET in the environment or .env")
        print("\n" + "!" * 78)
        print(f" LIVE TRADING on {ex.exchange_id} {'TESTNET' if ex.use_testnet else 'MAINNET - REAL MONEY'}")
        print(f" Per-order hard cap: {self.cfg.live.max_order_notional:,.2f} quote currency.")
        print(" This software is an educational project with no warranty. You are responsible for every order.")
        print("!" * 78)
        if pre_confirmed:
            return True
        if not sys.stdin.isatty():
            return False
        return input(" Type 'I UNDERSTAND' to continue: ").strip() == "I UNDERSTAND"

    def pipeline(self, launch_live: str = "ask") -> None:
        steps: list[tuple[str, Callable[[], Any]]] = [
            ("1/4 Data fetch", self.fetch),
            ("2/4 Vectorised + event-driven backtest", self.backtest),
            ("3/4 Preset comparison", self.compare),
        ]
        for title, step in steps:
            print(f"\n>>> PIPELINE STEP {title}")
            step()
        print("\n>>> PIPELINE STEP 4/4 Real-time paper trader")
        if launch_live == "no" or (launch_live == "ask" and not _ask_yes_no("Launch the paper trader now?")):
            print(" Skipped. Run `python main.py live` whenever you are ready.")
            return
        # The pipeline never places real orders, whatever TRADING_MODE says.
        paper_cfg = replace(self.cfg, live=replace(self.cfg.live, mode=TradingMode.PAPER_TRADING))
        TradingBotCLI(paper_cfg, self.plots, self.show)._with_data(self._data, self._data_key).live()

    def _with_data(self, df: pd.DataFrame | None, key: tuple[Any, ...] | None) -> TradingBotCLI:
        self._data, self._data_key = df, key
        return self

    # ------------------------------------------------------------------ interactive menu
    def menu(self) -> None:
        actions: dict[str, tuple[str, Callable[[], Any]]] = {
            "1": ("Fetch / update market data", self.fetch),
            "2": ("Backtest Ichimoku (vectorised + event-driven, report & charts)", self.backtest),
            "3": ("Compare Ichimoku presets (standard / crypto / dynamic)", self.compare),
            "4": ("Optimise parameters (in-sample vs out-of-sample)", self.optimize),
            "5": ("Backtest momentum strategy", self.momentum),
            "6": ("Backtest pairs trading (two symbols)", self.pairs_backtest),
            "7": ("Launch real-time trader", self.live),
            "8": ("Run the full pipeline", self.pipeline),
            "9": ("Settings", self.settings),
        }
        while True:
            c = self.cfg
            banner(f"CRYPTO ALGO TRADING SYSTEM | {c.data.symbol} {c.data.timeframe} | "
                   f"{build_strategy(c).name} | {c.live.mode.value}")
            print(f" data: {c.data.source.value}, {c.data.history_days} days | sizing: {c.risk.sizing_method.value} "
                  f"| stop: {c.risk.stop_method.value} | trailing: {c.risk.trailing_method.value} | "
                  f"feed: {c.live.feed.value}")
            for key, (label, _) in actions.items():
                print(f"  {key}) {label}")
            print("  0) Exit")
            choice = _input(" Select an option: ")
            if choice in ("0", "q", "quit", "exit", None):
                print(" Bye.")
                return
            if choice not in actions:
                print(" Unknown option.")
                continue
            try:
                actions[choice][1]()
            except DataLoaderError as exc:
                print(f"\n DATA ERROR: {exc}")
                if self.cfg.data.source is DataSource.EXCHANGE and _ask_yes_no("Switch to synthetic data instead?"):
                    self._update(data=replace(self.cfg.data, source=DataSource.SYNTHETIC),
                                 live=replace(self.cfg.live, feed=FeedType.REPLAY))
            except (BacktestError, ConfigError, LiveTradingError, ValueError) as exc:
                print(f"\n ERROR: {exc}")

    def _update(self, **sections: Any) -> None:
        self.cfg = self.cfg.with_updates(**sections)

    def settings(self) -> None:
        c = self.cfg
        print("\n Press Enter to keep the current value.")
        symbol = _prompt("Symbol", c.data.symbol, str.upper)
        timeframe = _prompt("Timeframe", c.data.timeframe, str, list(TIMEFRAME_MINUTES))
        days = _prompt("History (days)", c.data.history_days, int)
        source = _prompt("Data source", c.data.source.value, str, [s.value for s in DataSource])
        csv_path = c.data.csv_path
        if source == DataSource.CSV.value:
            csv_path = Path(_prompt("CSV path", str(c.data.csv_path or ""), str))
        preset_now = "dynamic" if c.strategy.dynamic else next(
            (k for k, v in ICHIMOKU_PRESETS.items() if v == c.strategy.params), "custom")
        preset = _prompt("Ichimoku preset", preset_now, str, [*ICHIMOKU_PRESETS, "dynamic", "custom"])
        allow_short = _prompt("Allow shorts (y/n)", "y" if c.strategy.allow_short else "n", str, ["y", "n"]) == "y"
        capital = _prompt("Initial capital", c.risk.initial_capital, float)
        sizing = _prompt("Position sizing", c.risk.sizing_method.value, str, [m.value for m in SizingMethod])
        risk_pt = _prompt("Risk per trade (fraction)", c.risk.risk_per_trade, float)
        stop = _prompt("Stop method", c.risk.stop_method.value, str, [m.value for m in StopMethod])
        trailing = _prompt("Trailing stop", c.risk.trailing_method.value, str, [m.value for m in TrailingMethod])
        feed = _prompt("Live feed", c.live.feed.value, str, [f.value for f in FeedType])
        live_strategy = _prompt("Live strategy", c.live.live_strategy.value, str,
                                [s.value for s in StrategyKind])

        strategy = replace(c.strategy, allow_short=allow_short)
        if preset == "dynamic":
            strategy = replace(strategy, dynamic=True)
        elif preset in ICHIMOKU_PRESETS:
            strategy = replace(strategy, params=ICHIMOKU_PRESETS[preset], dynamic=False)
        try:
            self._update(
                data=replace(c.data, symbol=symbol, timeframe=timeframe, history_days=days,
                             source=DataSource(source), csv_path=csv_path),
                strategy=strategy,
                risk=replace(c.risk, initial_capital=capital, sizing_method=SizingMethod(sizing),
                             risk_per_trade=risk_pt, stop_method=StopMethod(stop),
                             trailing_method=TrailingMethod(trailing)),
                live=replace(c.live, feed=FeedType(feed), live_strategy=StrategyKind(live_strategy)),
            )
            print(" Settings updated.")
        except (ConfigError, ValueError) as exc:
            print(f" Invalid settings, nothing changed: {exc}")


def _input(prompt: str) -> str | None:
    try:
        return input(prompt).strip().lstrip("﻿")  # Windows shells may prepend a BOM to piped input
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _ask_yes_no(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    return (_input(f" {question} [y/N]: ") or "").lower() in ("y", "yes")


def _prompt(label: str, current: Any, cast: Callable[[str], Any], choices: Sequence[str] | None = None) -> Any:
    hint = f" ({'/'.join(choices)})" if choices else ""
    while True:
        raw = _input(f"  {label}{hint} [{current}]: ")
        if not raw:
            return current
        try:
            value = cast(raw)
        except ValueError:
            print("  invalid value")
            continue
        if choices and value not in choices:
            print(f"  choose one of: {', '.join(choices)}")
            continue
        return value


# ======================================================================================
# Entry point
# ======================================================================================
def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    args = parse_args(sys.argv[1:] if argv is None else argv)
    setup_logging(args.log_level)
    try:
        cfg = build_config(args)
    except (ConfigError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    app = TradingBotCLI(cfg, plots=not args.no_plots, show=args.show)
    try:
        if args.command == "menu":
            app.menu()
        elif args.command == "fetch":
            app.fetch()
        elif args.command == "backtest":
            app.backtest(args.engine)
        elif args.command == "compare":
            app.compare()
        elif args.command == "optimize":
            app.optimize(args.metric)
        elif args.command == "momentum":
            app.momentum(args.engine)
        elif args.command == "pairs":
            app.pairs_backtest()
        elif args.command == "live":
            app.live(confirm_live=args.confirm_live)
        elif args.command == "pipeline":
            app.pipeline(args.launch_live)
    except (DataLoaderError, BacktestError, ConfigError, LiveTradingError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
