# Capstone Project — Algorithmic Trading Framework for Crypto (B.Tech.)

An end-to-end systematic trading framework that implements a curated set of *intermediate
and advanced* cryptocurrency trading strategies (technical, statistical-arbitrage and ML)
as a production-grade Python stack:
data → indicators → strategies → vectorised/event-driven backtesting → risk execution →
analytics → real-time paper/live trading.

> Scope note: the framework targets crypto markets. (A Yahoo-Finance forex data fetcher
> exists in the codebase, but the docs and demos intentionally stay crypto-only.)

## 1. Problem statement

- Retail crypto markets are full of high-frequency noise, fat tails and fees/slippage;
  naive Ichimoku/momentum signals alone do not provide a consistent edge.
- Research question: within *event-driven backtesting + realistic cost/risk modelling*, can
  a multi-signal ensemble of indicator/ML strategies form a workable trading pipeline — and
  how much of a strategy's apparent loss is the signal itself versus implementation choices
  (bar size, exit rule, turnover) that are rarely reported?
- Deliverable: a repeatable pipeline (data → signal → simulate → mitigate → report) that
  offers both a vectorised side (fast research) and an event-driven side (realistic fills).

## 2. Methodology

### 2.1 Data
- Binance (CCXT) OHLCV — paginated REST download, retry/back-off, gap report, SQLite cache
  (idempotent upsert, incremental refresh), CSV/Parquet export.
- Synthetic regime-switching GBM generator (Markov chain hidden regimes) for offline
  research and unit tests.

### 2.2 Indicators (`indicators.py`)
- Ichimoku 5-line: Tenkan/Kijun/Senkou A/Senkou B/Chikou + Kumo + projection.
- Wilder ATR, RSI (+ causal RSI divergence detection), Aroon, Bollinger Bands.
- Rescaled-range **Hurst exponent** (R/S regression), **ADF** stationarity test (statsmodels).
- Volatility-regime adaptive Ichimoku: quantile ya **K-Means** regime labels (causal,
  trailing), 3 preset sets Low/Normal/High.

### 2.3 Strategies
| Module | Strategy | Course level |
|---|---|---|
| `strategy.py` | Ichimoku TK-cross / Kumo-breakout with optional overlays (RSI, divergence) | Intermediate |
| `momentum.py` | Long/short time-series momentum (lookback return + SMA filter) | Intermediate |
| `intermediate_strategies.py` | **Calendar anomalies** (hour / day-of-week seasonal profiles) | Intermediate |
| `intermediate_strategies.py` | **Aroon / RSI divergence** (Aroon trend gate + RSI divergence entry) | Intermediate |
| `advanced_ml_strategies.py` | **K-Means asset clustering** (volatility/momentum/volume features, scikit-learn or scipy fallback) | Advanced |
| `advanced_ml_strategies.py` | **Pairs trading** — cointegration (ADF), rolling OLS hedge, spread z-score-Bollinger entries (re-exported) | Advanced |
| `advanced_ml_strategies.py` | **Hurst-exponent regime filter + RSI** (trend → momentum fade; reversion → RSI extremes) | Advanced |
| `advanced_ml_strategies.py` | **Long-only momentum / alpha portfolio** — cross-sectional `rank(ret2)+rank(retL)-2*rank(vol)`, equal-weight top-N, periodic rebalance | Advanced |

### 2.4 Execution & risk model
- CostModel: taker/maker fees, half-spread, slippage bps, square-root market impact
  `I = Y·σ·sqrt(Q/V)`, financing on shorts.
- Position sizing: fixed-fractional and half-Kelly, ATR/Kumo initial + trailing stops,
  intrabar fill rules (gaps, both-levels-in-bar), max-drawdown & daily-loss circuit breaker.
- 0.1% baseline per-side costs — included in every result reported below.

### 2.5 Backtesting
- `VectorizedBacktester`: full-notional, proportional costs, no stops → fast research/optimiser sweeps.
- `EventDrivenBacktester`: bar-by-bar, signals close-of-t → fill next open (no look-ahead),
  realistic fills, breaker. Return `BacktestResult` → `PerformanceReport` (Sharpe, Sortino,
  Calmar, MaxDD + duration, PSR, VaR/CVaR, alpha/beta, trade stats) + charts.

### 2.6 Live / paper
- asyncio engine: ccxt.pro WebSocket / REST / replay feeds; paper execution (order-book
  walking) and live (CCXT, testnet-first) modes; SQLite trade journal (fills, trades,
  equity, events).

## 3. Results (real Binance data)

**Headline first.** Results are given in two states. *Original defaults* (1h bars, fixed
4×ATR take-profit, no trailing stop) — Section 3.1 — leave every strategy in a loss.
*Repaired defaults* (4h bars, no take-profit, Kijun trailing stop), selected out-of-sample in
Section 3.2 — Section 3.3 — turn **two strategies positive** over three years (Ichimoku
+19.3%, momentum +36.2%) and shrink every strategy's drawdown. Neither state beats
buy-and-hold (+197.3%) over a tripling market, and the positive results are thin; their
limits are stated in Section 3.2.

### 3.1 Single-symbol event-driven, original defaults (BTC/USDT 1h, 180d, 2026-03 → 2026-09, $10k start)
Buy & hold benchmark: period return **+8.95%**, final **10,880.95** (annual vol 39.3%, Sharpe 0.64).

| Strategy | Final equity | Return | Sharpe | MaxDD | Trades | Win% | PF |
|---|---|---|---|---|---|---|---|
| Ichimoku 10/30/60/30 | 9,637.03 | -3.63% | -0.81 | -5.48% | 28 | 32.1% | 0.74 |
| Long-only momentum 20b | 9,345.11 | -6.55% | -1.07 | -11.45% | 58 | 36.2% | 0.74 |
| Divergence (Aroon/RSI) | 9,052.62 | -9.47% | -2.56 | -10.96% | 63 | 33.3% | 0.56 |
| Pairs (BTC×ETH, 310d window) | 7,233.03 | -27.7% | -0.77 | -35.03% | 121 | 31.4% | 0.71 |
| Hurst + RSI | 5,840.83 | -41.59% | -14.3 | -41.59% | 226 | 18.1% | 0.14 |
| Calendar anomalies | 1,475.42 | -85.25% | -21.1 | -85.25% | 798 | 21.3% | 0.21 |

### 3.2 Timeframe & exit-rule selection (3 years BTC/USDT, out-of-sample)

Section 3.1 runs every strategy at its *original* defaults (1h bars, fixed 4×ATR
take-profit, no trailing stop). To separate the strategy logic from those implementation
choices, `scripts/timeframe_sweep.py` backtests **384 configurations** — timeframe ×
Ichimoku preset × TK-cross window × exit rule × stop width — over three years of BTC/USDT
(Oct 2023 → Oct 2026; buy-and-hold +197.3% with a −53.5% drawdown). Every run is split 70/30
in-sample / out-of-sample, and the sweep is repeated on an independent two-year window.

**Finding 1 — the fixed take-profit was the main leak** (median out-of-sample return):

| Exit rule | Median OOS return |
|---|---|
| no take-profit + Kijun trailing stop | **+2.9%** |
| no take-profit + Chandelier (ATR) trailing | 0.0% |
| take-profit + Kijun trailing | 0.0% |
| take-profit, no trailing (original default) | −1.2% |

**Finding 2 — hourly bars lose to fees and whipsaw** (with the exit rule above):

| Timeframe | Median trades (3y) | Median OOS return | Median OOS Sharpe |
|---|---|---|---|
| 1h | 224 | −3.4% | −0.24 |
| 2h | 101 | +8.8% | 1.42 |
| **4h** | **52** | **+3.9%** | **0.77** |
| 6h | 40 | −1.9% | −0.63 |
| 12h | 15 | +3.8% | 1.41 |
| 1d | 9 | +1.4% | 0.95 |

**Adopted defaults:** 4h, preset 10/30/60/30, 3-bar cross window, no take-profit, Kijun
trailing stop, 2×ATR stop — the only combination positive in both halves of both data
windows with a usable trade count.

| Same 3 years of BTC/USDT | Original (1h, TP, no trail) | Adopted (4h, Kijun trail) |
|---|---|---|
| Total return | **−33.9%** | **+19.3%** |
| In-sample / out-of-sample | −17.8% / −19.6% | +9.5% / +9.0% |
| Max drawdown | −37.2% | **−13.0%** |
| Profit factor | 0.4–0.7 | 1.57 |
| Trades / win rate | 240 / 32.1% | 67 / 26.9% |
| Time in market | — | 13.3% |

Year by year (strategy vs buy-and-hold): 2023 (Oct–Dec) −3.0% vs +53.2%; 2024 +14.9% vs
+121.1%; 2025 −5.0% vs −6.6%; 2026 (Jan–Oct) **+12.7% vs −6.4%**.

**Three caveats stated up front.** (i) The profit is concentrated: of 67 trades, 18 win and
49 lose, and the single best trade (+1,937) equals the entire three-year net profit (1,934) —
remove it and the period is flat. (ii) Cross-asset decay: ETH/USDT gives +4.6% vs +52.4%
buy-and-hold, though the drawdown benefit transfers (−13.1% vs −68.0%). (iii) A Chandelier
(ATR) trail scored far better on BTC (+34.8%, PF 1.90) but **lost on ETH** (−3.4%), so it was
rejected — picking it would have been curve fitting.

### 3.3 Repaired defaults applied to the whole strategy set (BTC/USDT 4h, 3 years)

The exit rule and bar size are not Ichimoku-specific — every strategy shares `risk.py`. The
full scorecard on the repaired defaults
(`python scripts/compare_all.py --symbol BTC/USDT --timeframe 4h --days 1100`;
buy-and-hold +197.3% with a −53.5% drawdown):

| Strategy | Final | Return | CAGR | Sharpe | MaxDD | Trades | Win% | PF |
|---|---|---|---|---|---|---|---|---|
| Momentum 20b | 13,623 | **+36.2%** | +10.8% | 0.94 | −20.6% | 248 | 28.2 | 1.36 |
| Ichimoku 10/30/60/30 | 11,934 | **+19.3%** | +6.0% | 0.77 | **−13.0%** | 67 | 26.9 | 1.57 |
| Aroon/RSI divergence | 9,031 | −9.7% | −3.3% | −1.42 | −9.9% | 108 | 36.1 | 0.48 |
| Hurst + RSI | 7,547 | −24.5% | −8.9% | −2.10 | −26.4% | 307 | 33.2 | 0.54 |
| Calendar anomalies | 1,916 | −80.8% | −42.2% | −4.42 | −82.1% | 1,839 | 32.5 | 0.58 |

Against Section 3.1 (same strategies, original defaults, 180-day 1h sample): momentum moves
from −6.6% to +36.2% and Ichimoku from −3.6% to +19.3%, while the high-turnover strategies
stay deeply negative — calendar anomalies trade 1,839 times in three years and pay their
entire capital away in costs. Turnover and cost discipline, not signal sophistication,
separate the two groups.

Two caveats: the samples differ in length (180 days vs three years), so the comparison is
directional rather than like-for-like; and momentum's +36.2% carries a −20.6% drawdown
against Ichimoku's −13.0%, so risk-adjusted the two are closer than the headline returns
suggest (Sharpe 0.94 vs 0.77).

**Should momentum be the default?** Its +36.2% is the largest number in the table, so the
same discipline was applied to it: a 72-configuration sweep (timeframe x lookback x entry
threshold) with the 70/30 split, plus the ETH cross-check. The verdict is mixed and it was
*not* promoted to the default.

| | Ichimoku 4h | Momentum 4h |
|---|---|---|
| BTC full 3y | +19.3% | **+36.2%** |
| BTC in-sample / out-of-sample | +9.5% / **+9.0%** | +28.1% / +6.4% |
| BTC max drawdown | **−13.0%** | −20.6% |
| BTC profit factor | **1.57** | 1.36 |
| Best trade as share of profit | 100% | **65%** |
| ETH in-sample / out-of-sample | +13.6% / −7.9% | +14.7% / −6.4% |
| Median OOS return across its own grid | positive on 2h-1d | **negative on 1h-12h** |

Momentum earns most of its three-year figure in the in-sample bull phase, carries half again
the drawdown, and the median configuration of its own parameter grid loses out of sample on
every timeframe except 1d - the opposite of the "robust region" criterion of Section 2.4. It is
better than Ichimoku on exactly one axis, concentration: its profit survives the removal of
its best trade (+1,255), while Ichimoku's does not. Both lose out of sample on ETH. Momentum
is therefore selectable (`python main.py backtest --strategy momentum`) and reported, but
Ichimoku - more balanced across halves, lower drawdown - remains the default.

**Tooling note.** `scripts/compare_all.py` passed its argument list to `build_config()`
instead of the parsed namespace, so `--symbol/--timeframe/--days` were silently ignored and
every scorecard ran on the default window. Fixed; the table above is the first scorecard
whose window matches its caption.

### 3.4 Portfolio (K-Means + momentum-alpha, BTC/ETH/SOL/ADA/XRP/USDT, 120d)

*Run at the original exit defaults, before the Section 3.2 change; it has not been re-run on
the repaired settings, so it is not directly comparable with Section 3.3.*
- K-Means clusters real data, ranking each cluster by its characteristic mean-return/vol
  profile; the best-performing cluster's top-N names are held equal-weight (weights in CSV)
  across 348 rebalances.
- Portfolio final 8,083.26 vs buy-and-hold BTC/USDT 9,574.27 same window → **portfolio
  underperformed the BTC hold (−15.6%)**; it still beat its two weakest members (ADA
  7,800.60, XRP 9,255.46). (A draft "+3.4% beat buy-and-hold" figure was traced to a
  benchmark bug that compared against ADA/USDT; corrected.)

### 3.5 Live/paper demo
- Replay 2,500 BTC/USDT 1h bars: paper engine ENTER/EXIT with ATR stops, 2bp slippage,
  net PnL per trade, SQLite journal (`fills`/`trades`/`equity`/`events`).
- Live ccxt.pro WebSocket paper: heartbeat lag ~15ms, feed_age 0s, live price streaming.

### 3.6 Honest limitations
- **Sample negative at the original defaults.** 180-day single-symbol returns are negative
  for every strategy in that window. Section 3.2 shows that for Ichimoku most of the loss
  came from the bar size and the take-profit rather than the signal, but the repaired
  configuration still is not a proven edge: its three-year profit sits in one trade and does
  not transfer in magnitude to ETH.
- At the original defaults every profit factor was below 1. At the repaired defaults
  (Section 3.3) Ichimoku reaches PF 1.57 and momentum 1.36, while divergence (0.48),
  Hurst (0.54) and calendar (0.58) stay below 1: their turnover — up to 1,839 trades in
  three years — hands the capital to fees. Educationally useful, commercially unproven.
- Parameter sensitivity is now covered for Ichimoku (384 configurations, 1h→1d, two data
  windows, two assets, out-of-sample selection); the other strategies still run at their
  original settings and have had no equivalent sweep.
- Walk-forward re-fitting (rolling re-selection rather than one 70/30 split) is still
  pending, as is validation on a second exchange.
- sklearn's KMeans was DLL-blocked by Windows Smart App Control on the development machine,
  so the code falls back to scipy `kmeans2` with identical results; it prefers sklearn when importable.

## 4. Reproduce

```bash
pip install -r requirements.txt
python main.py fetch --symbol BTC/USDT --timeframe 1h --days 180
# Section 3.1 figures predate the Section 3.2 change: to reproduce them, restore the original
# exits per strategy, e.g.
#   python main.py backtest --timeframe 1h --days 180 --preset crypto --take-profit --trailing none
python scripts/compare_all.py --symbol BTC/USDT --timeframe 1h --days 180
python main.py calendar|divergence|hurst --source exchange --symbol BTC/USDT
python main.py portfolio --source exchange --timeframe 1h --days 120
python main.py pairs --symbol BTC/USDT --pair ETH/USDT
python main.py live --feed replay --source exchange --replay-bars 2500 --speed 0.01
python main.py live --feed ccxtpro --symbol BTC/USDT --timeframe 1m

# Section 3.2 — selection sweep and the adopted configuration
python main.py fetch --symbol BTC/USDT --timeframe 1h --days 1100
python scripts/timeframe_sweep.py          # 384 runs → reports/timeframe_sweep_BTCUSDT.csv
python main.py backtest --days 1100        # adopted defaults (4h, Kijun trail, no TP)
python main.py backtest --symbol ETH/USDT --days 1100    # cross-asset check
```

## 5. Tests
53 unit tests (`pytest`) — causality/no-look-ahead checks, synthetic regime-switching data,
cointegration/Hurst/KMeans/calendar/divergence/portfolio behaviour, and a research/production
parity test in which replaying bars through the live engine reproduces the event-driven
backtest trade for trade. All 53 pass on the development machine. The ADF stationarity test
needs `statsmodels` (listed in `requirements.txt`); without it `adf_test` returns NaN by
design and that single test fails.

## 6. Coverage vs syllabus
- Intermediate topics covered: Ichimoku, calendar anomalies, Aroon/RSI divergence,
  long/short momentum.
- Advanced topics covered: ML K-Means clustering, pairs/statistical arbitrage, Hurst
  regime filtering, long-only momentum/alpha portfolio, risk & execution framework,
  ML pipeline.
- Not covered (future work): ML return prediction (XGBoost/LSTM), portfolio optimisation
  beyond top-N equal-weight (mean-variance/HRP), backtesting via `backtesting.py`-style
  native integration.