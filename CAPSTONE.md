# Capstone Project — Algorithmic Trading Framework for Crypto (B.Tech.)

End-to-end systematic trading framework implement karta hua our *Crypto Trading
Strategies: Intermediate* and *Advanced* , ek production-grade Python stack ke
roles hai: data → indicators → strategies → vectorised/event-driven backtesting → risk
execution → analytics → real-time paper/live trading.

> Scope note: framework crypto markets ke liye hai. (A Yahoo-Finance Forex data fetcher
> codebase mein maujood hai lekin docs/demo crypto-only rakhe gaye.)

## 1. Problem statement

- Retail crypto markets high-frequency noise + fat tails + fees/slippage se bharе hain;
  naive Ichimoku/momentum signals akele consistently edge nahi de Paate.
- Research question: kya *event-driven backtesting + realistic cost/risk modelling* ke
  andar, indicators/ML strategies ka multi-signal ensemble ek workable trading pipeline
  bana sakta hai?
- Deliverable: repeatable pipeline (data → signal → simulate → mitigate → report) jo vectorised
  side (ya tez research) + event-driven side (ya realistic fills) dono offer kare.

## 2. Methodology

### 2.1 Data
- Binance (CCXT) OHLCV — paginated REST download, retry/back-off, gap report, SQLite cache
  (idempotent upsert, incremental refresh), CSV/Parquet export.
- Synthetic regime-switching GBM generator (Markov chain hidden regimes) offline research
  aur unittests ke liye.

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
- 0.1% baseline per-side costs — har D=ke results mein included.

### 2.5 Backtesting
- `VectorizedBacktester`: full-notional, proportional costs, no stops → tez research/optimizer.
- `EventDrivenBacktester`: bar-by-bar, signals close-of-t → fill next open (no look-ahead),
  realistic fills, breaker. Return `BacktestResult` → `PerformanceReport` (Sharpe, Sortino,
  Calmar, MaxDD + duration, PSR, VaR/CVaR, alpha/beta, trade stats) + charts.

### 2.6 Live / paper
- asyncio engine: ccxt.pro WebSocket / REST / replay feeds; paper execution (order-book
  walking) aur live (CCXT, testnet-first) modes; SQLite trade journal (fills, trades,
  equity, events).

## 3. Results (real Binance data)

### 3.1 Single-symbol event-driven (BTC/USDT 1h, 180d, 2026-03 → 2026-09, ₹/$10k start)
Buy & hold benchmark: -~1x (period return ≈ -31.8% from 2026-03-15 high to 2026-09-11).

| Strategy | Final equity | Return | Sharpe | MaxDD | Trades | Win% | PF |
|---|---|---|---|---|---|---|---|
| Divergence (Aroon/RSI) | 7,928 | -20.7% | -3.09 | -22.9% | 128 | 28.9% | 0.48 |
| Ichimoku (crypto preset) | 7,848 | -21.5% | -2.74 | -23.7% | 67 | 25.4% | 0.43 |
| Momentum (20b +2%) | 7,780 | -22.2% | -1.96 | -27.3% | 127 | 30.7% | 0.62 |
| Hurst + RSI | 4,129 | -58.7% | -10.84 | -58.7% | 445 | 20.9% | 0.25 |
| Calendar anomalies | 279 | -97.2% | -21.4 | -97.2% | 1580 | 20.6% | 0.23 |

### 3.2 Portfolio (K-Means + momentum-alpha, BTC/ETH/SOL/ADA/XRP/USDT, 120d)
- K-Means clusters real data pe, each cluster's characteristic mean-return/vol ranked;
  best-performing cluster se top-N equal-weight (weights CSV mein) 348 rebalances.
- Portfolio final 8,001.93 vs buy-and-hold BTC/USDT 7,735.85 → **portfolio added value
  (+3.4%) vs single-asset hold** in the same window.

### 3.3 Live/paper demo
- Replay 2,500 BTC/USDT 1h bars: paper engine ENTER/EXIT with ATR stops, 2bp slippage,
  net PnL per trade, SQLite journal (`fills`/`trades`/`equity`/`events`).
- Live ccxt.pro WebSocket paper: heartbeat lag ~15ms, feed_age 0s, live price streaming.

### 3.4 Honest limitations
- **Sample negative.** 180-day single-symbol returns are negative for every strategy in the
  tested window; the ensemble is validated plumbing, not a proven edge.
- Ichimoku/momentum/divergence PF < 1; calendar/hurst share costs-heavy high-frequency
  turnover (fees dominated) — educationally useful, commercially not proven.
- Parameter sensitivity: 1h/synthetic grid hi atrasi hui — 1d/multi-timeframe validation
  pending.
- sklearn's KMeans this machine pe DLL-blocked (Windows Smart App Control) → scipy
  `kmeans2` fallback; results identical, code prefers sklearn when importable.

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

## 6. Credits / gap vs syllabus
- : "Intermediate" → Ichimoku, calendar anomalies,
  divergence, momentum (covered).
- : "Advanced" → ML K-Means, pairs/stat-arb, Hurst,
  long-only momentum/alpha framework, risk/execution, ML-flow (covered).
- Not covered (future work): ML return prediction (XGBoost/LSTM), portfolio optimisation
  beyond top-N equal-weight (mean-variance/HRP), backtesting via `backtesting.py`-style
  native integration.