# Capstone Abstract & Presentation Outline

## Abstract

Cryptocurrency markets trade continuously, with high volatility, fat-tailed returns and
non-trivial transaction costs. This project presents an end-to-end Python framework for
designing, backtesting, risk-managing and paper/live-trading systematic cryptocurrency
strategies, implementing a full *intermediate and advanced* crypto strategy set
(technical, statistical-arbitrage and ML) on a single, reproducible codebase.

The system ingests OHLCV data from Binance (CCXT REST/WebSocket) with a cached SQLite
store and a synthetic regime-switching GBM generator for offline research, and computes a
library of indicators (Ichimoku Kinko Hyo, Wilder ATR, RSI with causal divergence,
Aroon, Bollinger Bands, rescaled-range Hurst exponent and the ADF stationarity test).
Strategies span technical (Ichimoku, calendar anomalies, Aroon/RSI divergence,
time-series and cross-sectional momentum), statistical-arbitrage (cointegrated pairs with
rolling OLS hedge and spread z-score) and machine-learning (K-Means market-regime
classification and asset clustering, Hurst-based regime filtering) families, together with
a long-only momentum/alpha portfolio that rebalances periodically into the top-ranked
assets.

Backtesting is provided in two engines: a vectorised research engine for fast parameter
sweeps and an event-driven engine that executes signals at the next bar's open (no
look-ahead), applies 0.1 % per-side fees, spread, slippage and square-root market impact,
uses fixed-fractional or half-Kelly position sizing with ATR/Kumo stops, and enforces
drawdown/daily-loss circuit breakers. Analytics cover Sharpe, Sortino, Calmar, drawdown
depth and duration, probabilistic Sharpe, VaR/CVaR, alpha/beta and full trade statistics,
with Matplotlib dashboards. A real-time asyncio engine supports paper trading by walking
the live order book and CCXT-based testnet/live execution.

The framework is validated by 53 unit tests (causality/no-look-ahead, synthetic regime
data, cointegration, clustering and portfolio behaviour) and demonstrated on real Binance
data. In the tested 180-day window every single-symbol strategy was unprofitable, and the
K-Means + momentum-alpha portfolio underperformed a BTC buy-and-hold (8,083.26 vs 9,574.27,
−15.6%) although it beat its two weakest members — illustrating that realistic cost/risk
modelling and portfolio construction — rather than any single signal — are what the
pipeline reliably delivers. All results are reported honestly as a validated
research/execution pipeline with candidate (not yet proven) signals.

**Keywords:** algorithmic trading, cryptocurrency, Ichimoku, pairs trading, K-Means,
Hurst exponent, backtesting, event-driven simulation, transaction costs, risk management.

---

## Presentation outline (12 slides)

1. **Title** — "An Algorithmic Trading & Backtesting Framework for Cryptocurrency";
   name, guide, institution, date.
2. **Motivation & problem statement** — volatile 24/7 markets, cost sensitivity, why a
   systematic + reproducible pipeline is needed.
3. **Objectives** — implement Intermediate + Advanced ; realistic backtesting;
   risk management; live/paper execution; honest evaluation.
4. **System architecture** — data → indicators → strategies → backtest engines →
   risk/execution → analytics → live engine (one flow diagram).
5. **Data & indicators** — CCXT/SQLite/synthetic; Ichimoku, ATR, RSI(+divergence), Aroon,
   Bollinger, Hurst, ADF.
6. **Strategies implemented** — table: technical / statistical-arbitrage / ML /
   portfolio, with the course-level mapping.
7. **Backtesting & execution realism** — vectorised vs event-driven; next-open fills,
   fees 0.1 %, spread, slippage, square-root impact, ATR/Kumo stops, sizing, breakers.
8. **Live & paper trading** — asyncio engine, order-book-walk paper fills, ccxt.pro
   WebSocket, CCXT testnet, SQLite journal (demo screenshot/log).
9. **Results — single-symbol** — comparison table (BTC/USDT 1h 180d) + equity chart;
   note the falling-market sample.
10. **Results — portfolio & robustness** — K-Means clusters, momentum-alpha vs
    buy-and-hold (−15.6% vs BTC; still above ADA/XRP); tests/causality.
11. **Discussion & limitations** — sample negative for singles, turnover/cost drag,
    sklearn DLL block → scipy fallback, forward-testing required.
12. **Conclusion & future work** — ML return prediction (XGBoost/LSTM), mean-variance/HRP
    portfolio optimisation, multi-timeframe validation; Q&A.

## Demo script (viva)
```bash
python main.py fetch --symbol BTC/USDT --timeframe 1h --days 180
python scripts/compare_all.py --symbol BTC/USDT --timeframe 1h --days 180
python main.py portfolio --source exchange --timeframe 1h --days 120
python main.py live --feed ccxtpro --symbol BTC/USDT --timeframe 1m     # live paper
```