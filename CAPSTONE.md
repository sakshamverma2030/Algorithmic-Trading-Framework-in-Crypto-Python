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
  a multi-signal ensemble of indicator/ML strategies form a workable trading pipeline?
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

### 3.1 Single-symbol event-driven (BTC/USDT 1h, 180d, 2026-03 → 2026-09, $10k start)
Buy & hold benchmark: period return **+8.95%**, final **10,880.95** (annual vol 39.3%, Sharpe 0.64).

| Strategy | Final equity | Return | Sharpe | MaxDD | Trades | Win% | PF |
|---|---|---|---|---|---|---|---|
| Ichimoku 10/30/60/30 | 9,637.03 | -3.63% | -0.81 | -5.48% | 28 | 32.1% | 0.74 |
| Long-only momentum 20b | 9,345.11 | -6.55% | -1.07 | -11.45% | 58 | 36.2% | 0.74 |
| Divergence (Aroon/RSI) | 9,052.62 | -9.47% | -2.56 | -10.96% | 63 | 33.3% | 0.56 |
| Pairs (BTC×ETH, 310d window) | 7,233.03 | -27.7% | -0.77 | -35.03% | 121 | 31.4% | 0.71 |
| Hurst + RSI | 5,840.83 | -41.59% | -14.3 | -41.59% | 226 | 18.1% | 0.14 |
| Calendar anomalies | 1,475.42 | -85.25% | -21.1 | -85.25% | 798 | 21.3% | 0.21 |

### 3.2 Portfolio (K-Means + momentum-alpha, BTC/ETH/SOL/ADA/XRP/USDT, 120d)
- K-Means clusters real data, ranking each cluster by its characteristic mean-return/vol
  profile; the best-performing cluster's top-N names are held equal-weight (weights in CSV)
  across 348 rebalances.
- Portfolio final 8,083.26 vs buy-and-hold BTC/USDT 9,574.27 same window → **portfolio
  underperformed the BTC hold (−15.6%)**; it still beat its two weakest members (ADA
  7,800.60, XRP 9,255.46). (A draft "+3.4% beat buy-and-hold" figure was traced to a
  benchmark bug that compared against ADA/USDT; corrected.)

### 3.3 Live/paper demo
- Replay 2,500 BTC/USDT 1h bars: paper engine ENTER/EXIT with ATR stops, 2bp slippage,
  net PnL per trade, SQLite journal (`fills`/`trades`/`equity`/`events`).
- Live ccxt.pro WebSocket paper: heartbeat lag ~15ms, feed_age 0s, live price streaming.

### 3.4 Honest limitations
- **Sample negative.** 180-day single-symbol returns are negative for every strategy in the
  tested window; the ensemble is validated plumbing, not a proven edge.
- Ichimoku/momentum/divergence PF < 1; calendar/hurst share costs-heavy high-frequency
  turnover (fees dominated) — educationally useful, commercially not proven.
- Parameter sensitivity: only the 1h/synthetic grid was swept — 1d/multi-timeframe
  validation is pending.
- sklearn's KMeans was DLL-blocked by Windows Smart App Control on the development machine,
  so the code falls back to scipy `kmeans2` with identical results; it prefers sklearn when importable.

## 4. Reproduce

```bash
pip install -r requirements.txt
python main.py fetch --symbol BTC/USDT --timeframe 1h --days 180
python scripts/compare_all.py --symbol BTC/USDT --timeframe 1h --days 180
python main.py calendar|divergence|hurst --source exchange --symbol BTC/USDT
python main.py portfolio --source exchange --timeframe 1h --days 120
python main.py pairs --symbol BTC/USDT --pair ETH/USDT
python main.py live --feed replay --source exchange --replay-bars 2500 --speed 0.01
python main.py live --feed ccxtpro --symbol BTC/USDT --timeframe 1m
```

## 5. Tests
53 unit tests pass (`pytest`) — causality/no-look-ahead checks, synthetic regime-switching
data, cointegration/Hurst/KMeans/calendar/divergence/portfolio behaviour.

## 6. Coverage vs syllabus
- Intermediate topics covered: Ichimoku, calendar anomalies, Aroon/RSI divergence,
  long/short momentum.
- Advanced topics covered: ML K-Means clustering, pairs/statistical arbitrage, Hurst
  regime filtering, long-only momentum/alpha portfolio, risk & execution framework,
  ML pipeline.
- Not covered (future work): ML return prediction (XGBoost/LSTM), portfolio optimisation
  beyond top-N equal-weight (mean-variance/HRP), backtesting via `backtesting.py`-style
  native integration.