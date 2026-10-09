# A Systematic Framework for Backtesting and Live Paper Trading of Quantitative Cryptocurrency Strategies

**Author:** Saksham  
**Affiliation:** B.Tech Final Year (Capstone Project)  
**Date:** September 2026

---

## Abstract

Cryptocurrency markets trade 24/7 with high volatility, fat-tailed returns and material
transaction costs, making the design and *honest* evaluation of systematic strategies
difficult. This paper presents an end-to-end Python framework that unifies data
acquisition, indicator engineering, strategy implementation, vectorised and event-driven
backtesting, risk/execution modelling, performance analytics and a real-time paper/live
engine. The strategy set spans a standard *intermediate & advanced* crypto strategy
strategy set: Ichimoku Cloud, calendar anomalies, Aroon/RSI divergence,
time-series and cross-sectional momentum, cointegrated pairs trading, K-Means market
clustering and Hurst-exponent regime filtering.

The framework is validated with 53 unit tests that enforce causality (no look-ahead) and
demonstrated on real Binance data (BTC/USDT 1h, 4,319 bars, 180 days). With zero-cost
assumptions removed and 0.1% per-side fees plus slippage applied, every single-symbol
strategy underperforms buy-and-hold in that sample, which we report transparently. We then
run a 384-configuration selection sweep (timeframe x Ichimoku preset x cross window x exit
rule x stop width) over three years of BTC/USDT and choose on out-of-sample behaviour rather
than on the best in-sample cell: removing the fixed take-profit and moving from 1h to 4h
turns the Ichimoku system from −33.9% into +19.3% over the same three years, with maximum
drawdown cut from −37.2% to −13.0% (buy-and-hold: +197.3%, −53.5%). That gain is concentrated
in very few trades - the single best trade equals the entire three-year profit - so we report
it as a cost-and-robustness finding rather than a proven edge. The
K-Means-clustered, long-only momentum portfolio also underperformed a BTC buy-and-hold
(8,083.26 vs 9,574.27, −15.6%) while still beating its two weakest members (ADA, XRP).
A paper/live engine ran overnight on BTC/USDT 1m and traded through an order-book-walking
simulator with ATR stops; the full journal (paper + replay + live) records 57 round-trip
trades and 116 fills. We argue the framework's
contribution is validated *plumbing* — realistic costs, risk and causality — rather than a
proven profitable signal, and outline the path to forward-testing ML-driven signals.

**Keywords:** algorithmic trading, cryptocurrency, Ichimoku, pairs trading, K-Means,
Hurst exponent, event-driven backtesting, transaction costs.

---

## 1. Introduction

Retail and even institutional access to crypto exchange data is now trivial; what remains
hard is turning that data into a *defensible* trading system. Three problems dominate:

1. **Look-ahead bias** — naive strategies inadvertently use future information, inflating
   backtest performance.
2. **Ignored costs** — ignoring fees, spread, slippage and market impact turns a "profitable"
   backtest into a losing account.
3. **Unrealistic fill modelling** — assuming fills at the signal bar's close rather than the
   ground truth of next-bar execution.

This project builds a framework in which these three problems are first-class citizens, and
uses it to implement and evaluate the full set of systematic strategies. The
contributions are:

- A **two-engine backtesting design**: a vectorised research engine (full-notional,
  proportional costs) for fast parameter sweeps, and an **event-driven engine** that
  simulates the live engine bar-by-bar (next-open fills, position sizing, ATR/Kumo stops,
  circuit breakers).
- A **library of gate-keeping modules** for fraud-proof evaluation: causal indicator
  implementations, a cost model (fees, spread, slippage, square-root impact), drawdown
  depth/duration statistics and multi-metric analytics (Sharpe, Sortino, Calmar, PSR,
  VaR/CVaR, alpha/beta).
- A **real-time asyncio engine** that runs the same strategy objects on live WebSocket
  (ccxt.pro) or replayed data with paper (order-book walking) and CCXT-based testnet/live
  execution, all journaled to SQLite.
- An **honest experimental protocol**: results are reported with costs included, negative
  sample results are published rather than hidden, and every strategy ships with causality
  tests.

---

## 2. Related Work

### 2.1 Ichimoku trading systems
Ichimoku Kinko Hyo (五線譜) is a five-line Japanese indicator (Tenkan, Kijun, Senkou A,
Senkou B, Chikou) widely used for trend and momentum. Quantitative treatments focus on
Tenkan/Kijun crosses and Kumo (cloud) breakouts with an intrinsic trailing stop at the
cloud edge.

### 2.2 Pairs trading & cointegration
Vidyamurthy (2004) and Gatev et al. (2006) formalise statistical arbitrage as trading a
mean-reverting spread. The celebrated "classic" approach constructs a spread from an OLS
hedge ratio and trades z-score thresholds of the rolling spread. We implement exactly this,
using the ADF test (statsmodels) on the log-price spread to confirm stationarity before
trading.

### 2.3 Market clustering & regime detection
K-Means clustering on market features is used both naively (classical state detection on
volatility) and as an asset-classifier (clustering assets by volatility/momentum/liquidity
then allocating into the best cluster). Hurst exponent (rescaled-range) classifies a series
as trending (H>0.5), mean-reverting (H<0.5) or random-walk (H=0.5); we combine it with RSI
as a regime gate.

### 2.4 Momentum & portfolio construction
Cross-sectional momentum ranks assets by past returns and holds the top decile
(Jegadeesh & Titman, 1993); equal-weight top-N portfolios avoid the estimation error of
mean-variance weights. We implement a two-component ranking `rank(ret2)+rank(retL)–
2·rank(vol)` to penalise high-volatility "momentum trash".

---

## 3. Methodology

### 3.1 Data layer

- **Binance (CCXT)**: paginated OHLCV download with retry/back-off, gap detection, and an
  idempotent SQLite cache supporting incremental refresh.
- **Synthetic**: a regime-switching geometric-Brownian-motion generator (Markov chain over
  hidden volatility regimes) for offline research and unit tests.
- **CSV/Parquet**: round-trip persistence for reproducibility.

For the results in §5 we use real Binance BTC/USDT 1-hour OHLCV: **4,319 bars, 180 days,
2026-03-15 → 2026-09-11**, with a period return of **+8.95%** (buy-and-hold from the first
to the last bar).

### 3.2 Indicators (`indicators.py`)

All indicators are vectorised and **strictly trailing** (past bars only):

- Ichimoku: five lines + Kumo (cloud) + projection; Wilder-smoothed ATR.
- RSI (Wilder) plus a **causal divergence detector** (compares rolling extrema of price vs
  RSI within a lookback window).
- Aroon (Chande), Bollinger Bands, rescaled-range **Hurst exponent** (R/S regression over
  lags) and the **ADF** stationarity test (statsmodels) with a version-safe API.

An adaptive Ichimoku variant labels each bar's volatility regime (Low/Normal/High) either by
quantile thresholds or by **causal K-Means** on trailing NATR and switches between three
Ichimoku parameter presets.

### 3.3 Strategies

| # | Strategy | Module | Course |
|---|---|---|---|
| S1 | Ichimoku 10/30/60/30 (TK cross + Kumo filter, long-only) | `strategy.py` | Intermediate |
| S2 | Long-only time-series momentum (20-bar return, +2% entry / 0% exit, SMA filter) | `momentum.py` | Intermediate |
| S3 | Calendar anomalies (hour & day-of-week seasonal profiles, expanding-means) | `intermediate_strategies.py` | Intermediate |
| S4 | Aroon/RSI divergence (Aroon trend gate + RSI divergence entry) | `intermediate_strategies.py` | Intermediate |
| S5 | Cointegrated pairs: BTC/USDT × ETH/USDT, rolling OLS ratio → z-score entries | `pairs.py` | Advanced |
| S6 | Hurst regime (H>0.5 trend: RSI cross-50; H<0.5 reversion: RSI extremes) | `advanced_ml_strategies.py` | Advanced |
| S7 | K-Means asset clustering + long-only momentum-alpha portfolio (top-2 of 5) | `advanced_ml_strategies.py` | Advanced |

Each strategy exposes a common contract: `generate_signals(df)` returns a signal frame with
consistent columns (`long_entry`, `short_entry`, `exit_long`, `exit_short`, ATR, Kumo edges)
that both backtest engines and the live engine consume unchanged.

### 3.4 Execution & risk (`risk.py`)

- **Cost model**: taker/maker fee (0.1% default), half-spread (1 bp), slippage (2 bps) and
  square-root market impact `I = ζ·σ·√(Q/V)` (Basilico et al.); shorts carry financing.
- **Position sizing**: fixed-fractional (risk-per-trade = 1% of equity) or half-Kelly.
- **Stops**: ATR or Kumo initial stop, optional trailing, conservative intrabar rules for
  gaps and both-levels-bars.
- **Circuit breakers**: max-drawdown and daily-loss limits halt the engine.

### 3.5 Backtesting engines

| | Vectorised | Event-driven |
|---|---|---|
| Purpose | parameter sweeps | realistic evaluation |
| Sizing | full notional | risk-based |
| Stops | none | ATR/Kumo/trailing |
| Fill | signal-bar close (approx.) | **next-bar open** (no look-ahead) |
| Costs | proportional | fees + spread + slippage + impact |
| Breakers | none | max-DD / daily-loss |

The event-driven engine mirrors the live engine's decision loop so that
`backtest == live replay` as closely as possible (a dedicated parity test exists).

### 3.6 Real-time engine (`live_trader.py`)

Asyncio engine over ccxt.pro WebSocket / REST / replay feeds. Python paper execution walks
the real order book (VWAP over consumed levels); CCXT enables testnet/live. Fills, trades,
equity and events are journaled to SQLite.

### 3.7 Analytics (`analytics.py`)

Sharpe/Sortino/Calmar ratios, drawdown depth & **duration**, probabilistic Sharpe, VaR/CVaR,
alpha/beta, trade statistics (win rate, profit factor, payoff ratio) and Matplotlib
dashboards.

---

## 4. Validation Protocol

53 unit tests (`pytest`) enforce the project's honesty guarantees:

- **Causality**: RSI/divergence, momentum and regime labels never use future bars;
  `generate_signals` on a truncated prefix equals the first rows of the full frame.
- **Behaviour**: RSI divergence detects constructed patterns; K-Means regime labels are
  bounded and causal; cointegration analyzer recovers a known hedge ratio; ADF flags
  stationary series; Hurst separates trending from mean-reverting series; Bollinger bands
  keep 2σ geometry; calendar buckets use expanding means only; MomentumAlpha portfolio
  rebalances within top-N and rejects empty universes.
- **Config**: misconfigurations (no calendar features, empty universe) raise `ConfigError`.
- **Research/production parity**: replaying historical bars through the asynchronous live
  engine reproduces the event-driven backtest trade for trade (side, entry/exit timestamps,
  exit reason), so the configuration chosen in research is the configuration that trades.

**Parameter-selection protocol.** Every tunable choice reported in Section 5.5 is made on a
chronological 70/30 in-sample/out-of-sample split, repeated on a second independent data
window, and cross-checked on a second asset. A configuration is adopted only if it is
positive in *both* halves of *both* windows. The single best in-sample cell is explicitly
rejected: the in-sample optimiser's winner collapses out of sample (IS Sharpe ~0.49 ->
OOS ~-2.7).

---

## 5. Results

**Headline.** Results are reported in two states, because the difference between them is the
paper's main empirical point. *Original defaults* (1h bars, fixed 4xATR take-profit, no
trailing stop) are in Section 5.1: every single-symbol strategy loses money. *Repaired
defaults* (4h bars, no take-profit, Kijun-sen trailing stop), selected out-of-sample in
Section 5.5, are in Section 5.6: over three years of BTC/USDT two of the five strategies turn
positive (Ichimoku +19.3%, momentum +36.2%) and every strategy's drawdown shrinks. Neither
state beats buy-and-hold over a tripling market (+197.3%), and the positive results remain
thin, so Section 5.5 also reports their limits.

All results are event-driven, with costs (0.1% fee + 1 bp half-spread + 2 bps slippage =
~0.1%/side through the cost model) and $10,000 initial capital.

### 5.1 Single-symbol strategies — BTC/USDT 1h, 180 days

Benchmark: buy-and-hold **+8.95%**, final **10,880.95**, annual vol 39.3%, Sharpe 0.64.

| Strategy | Final | Return | CAGR | Sharpe | MaxDD | Timeλ | Trades | Win% | PF |
|---|---|---|---|---|---|---|---|---|---|
| S4 Aroon/RSI divergence | 9,052.62 | −9.47% | −18.3% | −2.56 | −10.96% | 7.8% | 63 | 33.3 | 0.56 |
| S1 Ichimoku 10/30/60/30 | 9,637.03 | −3.63% | −7.2% | −0.81 | −5.48% | 13.2% | 28 | 32.1 | 0.74 |
| S2 Momentum 20b | 9,345.11 | −6.55% | −12.8% | −1.07 | −11.45% | 18.3% | 58 | 36.2 | 0.74 |
| S3 Calendar anomalies | 1,475.42 | −85.25% | −97.9% | −21.1 | −85.25% | 29.4% | 798 | 21.3 | 0.21 |
| S6 Hurst + RSI | 5,840.83 | −41.59% | −66.4% | −14.3 | −41.59% | 5.1% | 226 | 18.1 | 0.14 |
| S5 Pairs (BTC×ETH, 60d win, z=2) | 7,233.03 | −27.7% | — | −0.77 | −35.03% | — | 121 | 31.4 | 0.71 |

*Pairs was backtested over a longer 310-day window (buy-and-hold −6.9%, final 9,295.91), so
its `Final equity` column is not directly comparable with the other rows.*

Observations:

- **All long-only singles lost money** in a mildly rising sample because they were out of
  market most of the time (Time-in-market 5–29%) and their entries clustered in the noisy
  middle of the window. Costs dominate: the calendar strategy paid ~₹7,800 in
  fees+slippage (5,868 + 1,933) on ₹10,000 capital across 798 trades — its Sharpe
  collapses to −21.
- **S1 Ichimoku** had the smallest loss and the best drawdown control (−5.5% vs −29.4%
  benchmark), consistent with its built-in cloud trailing. This row uses the *original*
  defaults (1h bars, fixed 4xATR take-profit, no trailing stop); Section 5.5 shows that those
  two choices, rather than the Ichimoku logic itself, caused most of the loss.
- **S6 Hurst** and **S3 Calendar** show the cost of over-trading in noise; both are
  pattern-laden but commercially unproven.

### 5.2 K-Means + momentum-alpha portfolio (S7)

Universe: BTC, ETH, SOL, ADA, XRP (1h, 120 days). K-Means (k=3) clusters assets by
annualised volatility / momentum / log-volume; the engine ranks clusters by momentum and
holds the top-2 assets equal-weight, rebalancing every 10 bars.

| Metric | Portfolio | Buy & hold BTC/USDT |
|---|---|---|
| Final equity | 8,083.26 | 9,574.27 |
| Outperformance | **−15.6%** | — |
| Rebalances / trades | 348 | — |

In the same window the altcoins polarised: ETH and SOL rallied (+11% each), while ADA fell
~22% and XRP ~7%. The equal-weight rule split the difference — it beat the two laggards it
might have held but trailed the single BTC hold, so the experiment did **not** demonstrate
alpha. Its value is the *plumbing*: quotas, rebalancing loop, and honest comparison against
a real benchmark (the earlier +3.4% "beating buy-and-hold" figure was traced to a bug that
compared the portfolio against ADA/USDT instead of BTC/USDT, and was corrected).

### 5.3 Live paper trading

The engine ran **live overnight** on BTC/USDT 1m (Sep 11 13:18 → Sep 12 02:37 UTC, ~13
hours, 800 equity samples): it opened 5 positions on closing bars — 4 were stopped out and
1 hit take-profit (net −$162), matching the backtest's behaviour. Across the whole journal
(paper + replay + live sessions) the engine recorded **57 round-trip trades / 116 fills**;
the spread of results was wide (best single trade +$222, worst −$159) and the journal's
total net P&L is −$869. These are plumbing demonstrations — correct execution and honest
bookkeeping — not evidence of a profitable edge.
This validates the execution engine (fills, stops, financing, journaling) as live-sim ready.

### 5.4 Portfolio cluster details (real data, 120d)

| Cluster | n_assets | mean_vol | mean_momentum | members |
|---|---|---|---|---|
| 0 | 2 | 4.28 | +0.018 | ETH/USDT, SOL/USDT |
| 1 | 2 | 6.52 | +0.0004 | ADA/USDT, XRP/USDT |
| 2 | 1 | 2.66 | +0.009 | BTC/USDT |

---

### 5.5 Timeframe and exit-rule selection (out-of-sample sweep)

Sections 5.1-5.4 evaluate the strategy set at its original defaults. To separate *strategy
logic* from *implementation choices*, `scripts/timeframe_sweep.py` backtests 384
configurations - timeframe (1h, 2h, 4h, 6h, 12h, 1d) x preset (standard, crypto,
crypto_slow, dynamic) x TK-cross window (1, 3 bars) x exit rule (fixed take-profit with or
without trailing; trailing-only with Kijun or Chandelier/ATR) x stop width (2, 4 ATR) -
on three years of BTC/USDT (5 Oct 2023 - 9 Oct 2026, 26,399 hourly bars resampled upward;
buy-and-hold +197.3% with a −53.5% drawdown). Each run is split 70/30 in-sample /
out-of-sample, and the whole sweep is repeated on an independent two-year window.

**Finding 1 - the fixed take-profit is the dominant leak.** Trend systems earn from a thin
right tail; a 4xATR target truncates exactly those trades while losers still run to the stop.

| Exit rule | Median out-of-sample return |
|---|---|
| no take-profit + Kijun trailing stop | **+2.9%** |
| no take-profit + Chandelier (ATR) trailing | 0.0% |
| take-profit + Kijun trailing | 0.0% |
| take-profit, no trailing (original default) | −1.2% |

**Finding 2 - hourly bars lose to costs and whipsaw.** With the exit rule above:

| Timeframe | Median trades (3y) | Median OOS return | Median OOS Sharpe | Median OOS drawdown |
|---|---|---|---|---|
| 1h | 224 | −3.4% | −0.24 | −14.5% |
| 2h | 101 | +8.8% | 1.42 | −5.3% |
| **4h** | **52** | **+3.9%** | **0.77** | **−4.3%** |
| 6h | 40 | −1.9% | −0.63 | −3.5% |
| 12h | 15 | +3.8% | 1.41 | −1.5% |
| 1d | 9 | +1.4% | 0.95 | −1.1% |

2h scored highest, but 4h with the `crypto` preset was the only combination positive in all
four segments (both halves of both windows) with a usable trade count; 12h and 1d trade 9-15
times in three years, too small a sample to trust.

**Adopted configuration and effect.** 4h, preset 10/30/60/30, 3-bar TK-cross window, no
take-profit, Kijun-sen trailing stop, 2xATR stop:

| | Original (1h, TP, no trail) | Adopted (4h, Kijun trail, no TP) |
|---|---|---|
| Total return (3y) | **−33.9%** | **+19.3%** |
| In-sample / out-of-sample | −17.8% / −19.6% | +9.5% / +9.0% |
| Max drawdown | −37.2% | **−13.0%** |
| Sharpe / Sortino | - | 0.77 / 1.31 |
| Profit factor | 0.4-0.7 | 1.57 |
| Trades / win rate | 240 / 32.1% | 67 / 26.9% |
| Time in market | - | 13.3% |
| Annualised alpha vs buy-and-hold | - | +3.5% |

Year by year (strategy vs buy-and-hold): 2023 (Oct-Dec) −3.0% vs +53.2%; 2024 +14.9% vs
+121.1%; 2025 −5.0% vs −6.6%; 2026 (Jan-Oct) **+12.7% vs −6.4%**. The system is in the
market 13.3% of the time, so it cannot track a tripling market; its contribution is drawdown
control and positive performance when the trend turns.

**Limits of this result, stated explicitly.**

1. *Profit concentration.* Of 67 trades, 18 win and 49 lose; gross profit +5,336 against
   gross loss −3,401 on 10,000 of capital. The best single trade (17-26 Aug 2026,
   BTC 64.2k -> 78.4k) returns 1,937, while the whole three-year net profit is 1,934.
   Removing that trade leaves the period flat; removing the best three leaves it at −1,432.
   A long right tail is the intended behaviour of trend following, but with 67 trades the
   headline figure carries little statistical weight.
2. *Cross-asset decay.* The same configuration on three years of ETH/USDT 4h returns +4.6%
   against +52.4% buy-and-hold, with a −13.1% drawdown against −68.0%: the drawdown benefit
   transfers, the return does not.
3. *A tempting configuration was rejected.* A Chandelier (ATR) trailing stop scores far
   better on BTC (+34.8%, profit factor 1.90, best trade only 53% of profit) but loses on
   ETH (−3.4%, profit factor 0.90). Under the protocol of Section 4 it was not adopted,
   which is precisely the discipline the sweep exists to enforce.

---

### 5.6 The same repair applied to the whole strategy set

The exit rule and bar size of Section 5.5 are not Ichimoku-specific: every strategy in this
framework shares `risk.py`. Re-running the full scorecard on the repaired defaults over three
years of BTC/USDT 4h (`python scripts/compare_all.py --symbol BTC/USDT --timeframe 4h
--days 1100`; buy-and-hold +197.3%, maximum drawdown −53.5%):

| Strategy | Final | Return | CAGR | Sharpe | MaxDD | Trades | Win% | PF |
|---|---|---|---|---|---|---|---|---|
| S2 Momentum 20b | 13,623 | **+36.2%** | +10.8% | 0.94 | −20.6% | 248 | 28.2 | 1.36 |
| S1 Ichimoku 10/30/60/30 | 11,934 | **+19.3%** | +6.0% | 0.77 | **−13.0%** | 67 | 26.9 | 1.57 |
| S4 Aroon/RSI divergence | 9,031 | −9.7% | −3.3% | −1.42 | −9.9% | 108 | 36.1 | 0.48 |
| S6 Hurst + RSI | 7,547 | −24.5% | −8.9% | −2.10 | −26.4% | 307 | 33.2 | 0.54 |
| S3 Calendar anomalies | 1,916 | −80.8% | −42.2% | −4.42 | −82.1% | 1,839 | 32.5 | 0.58 |

Compared with Section 5.1 (same strategies, original defaults, 180-day 1h sample), momentum
moves from −6.6% to +36.2% and Ichimoku from −3.6% to +19.3%, while the two high-turnover
strategies stay deeply negative: calendar anomalies trade 1,839 times in three years and pay
their entire capital away in costs. Turnover, not signal sophistication, separates the two
groups — and the ordering is the same conclusion the sweep reached for timeframes.

Two caveats carry over. The samples are different lengths (180 days versus three years), so
the comparison is directional rather than like-for-like; and momentum's +36.2% comes with a
−20.6% drawdown, well above Ichimoku's −13.0%, so on risk-adjusted terms the two are closer
than the headline returns suggest (Sharpe 0.94 versus 0.77).

**Is momentum the better default?** Its +36.2% is the largest number in the table, so the
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
every timeframe except 1d - the opposite of the "robust region" criterion of Section 4. It is
better than Ichimoku on exactly one axis, concentration: its profit survives the removal of
its best trade (+1,255), while Ichimoku's does not. Both lose out of sample on ETH. Momentum
is therefore selectable (`python main.py backtest --strategy momentum`) and reported, but
Ichimoku - more balanced across halves, lower drawdown - remains the default.

**Tooling note.** `scripts/compare_all.py` passed its argument list to `build_config()`
instead of the parsed namespace, so `--symbol/--timeframe/--days` were silently discarded and
every scorecard ran on the default window. The bug is fixed; the table above is the first
scorecard whose window matches its caption.

---

## 6. Discussion

**What the framework proves.** The engineering contribution is real and measurable: causal
indicators, two backtesting engines with matching semantics, a realistic cost/risk layer,
and a live engine with verified paper execution. This is exactly what a *systematic* crypto
pipeline must ship before any strategy can be trusted.

**What the results say about retail strategies.** The single-symbol results at the original
defaults are a sober and useful negative result. In a mildly rising 180-day BTC sample, every
long-only timing strategy lost money, and turnover relative to costs predicts the outcome
(calendar ≪ hurst ≪ momentum ≈ ichimoku ≈ divergence in cost discipline). The portfolio
result is the encouraging counterpoint: risk-diversification across clustered assets and
cross-sectional ranking traded better than timing a single asset.

**How much of the loss was the strategy, and how much the implementation.** Section 5.5
answers this directly, and the answer is uncomfortable for the usual presentation of
indicator strategies: two implementation choices — bar size and exit rule — moved the same
Ichimoku logic on the same asset from −33.9% to +19.3% over three years and cut the drawdown
from −37.2% to −13.0%. Neither choice touches the signal. Published indicator results that do
not state the bar size, the exit rule and the cost model are therefore close to
uninterpretable. Equally, the repaired configuration does not establish an edge: its profit
sits in one trade, its magnitude does not transfer to ETH, and over the full three years it
remains far behind buy-and-hold. The defensible claim is narrower — the framework can tell a
cost artefact from a signal, and it reports when the signal is thin.

**Threats to validity.** (i) two markets (BTC, ETH) and, for the selection study, two
overlapping windows — still one exchange and one three-year sequence of regimes;
(ii) the adopted configuration rests on 67 trades, one of which carries the entire profit, so
its confidence interval is wide and no claim of significance is made; (iii) resampling 1h
bars upward assumes a gap-free hourly history, which the data-quality report verifies for
this sample but which would not hold through an exchange outage; (iv) single-seed synthetic
tests only;
(v) K-Means label non-determinism mitigated via scipy `kmeans2` seed + cluster-centre
sorting; (vi) scikit-learn's KMeans was DLL-blocked by Windows Application Control on the
dev machine, so the shipped code prefers sklearn and falls back to scipy with identical
output.

---

## 7. Conclusion & Future Work

We presented a research-grade, honest, end-to-end framework for systematic crypto trading,
implementing a full intermediate/advanced strategy set and demonstrating it on real
data and in live paper trading. The contribution is validated infrastructure for cost-aware, causal
evaluation — and a candid record that single-symbol retail timing strategies, once costs
are included, do not simply work. The selection study of Section 5.5 sharpens that record:
bar size and exit rule dominated the published loss, and repairing them turned −33.9% into
+19.3% with a quarter of the drawdown, yet the repaired system still earns its profit from a
single trade and does not reproduce that magnitude on a second asset. Infrastructure that
separates these cases — and that refuses the better-looking but non-transferable
configuration — is the result we claim.

Future work: (0) extend the selection sweep to walk-forward re-fitting across several
assets and exchanges, until the headline figure rests on enough round trips to carry
statistical weight; (1) ML return prediction (XGBoost/LSTM) feeding the portfolio ranker;
(2) mean-variance/Hierarchical Risk Parity allocation in place of equal-weight;
(3) multi-timeframe and multi-market (ETH, SOL, BTC-settled) validation; (4) testnet
orders on Binance Spot Testnet with live execution journaling; (5) walk-forward /
transaction-cost-adjusted parameter optimisation.

---

## 8. References

1. V. Vidyamurthy, *Pairs Trading: Quantitative Methods and Analysis*. Wiley, 2004.
2. E. Gatev, W. Goetzmann and K. Rouwenhorst, "Pairs Trading: Performance of a
   Relative-Value Arbitrage Rule," *Review of Financial Studies*, 2006.
3. N. Jegadeesh and S. Titman, "Returns to Buying Winners and Selling Losers:
   Implications for Stock Market Efficiency," *Journal of Finance*, 1993.
4. E. J. Peters, *Fractal Market Analysis* (rescaled-range Hurst), Wiley, 1994.
5. B. Mandelbrot, "The Variation of Certain Speculative Prices," *Journal of Business*, 1963.
6. J. MacQueen, "Some Methods for Classification and Analysis of Multivariate
   Observations," *Proc. 5th Berkeley Symp.*, 1967 (K-Means).
7. S. A. Dickey and W. A. Fuller, "Likelihood Ratio Statistics for Autoregressive Time
   Series with a Unit Root," *JASA*, 1979.
8. A. Damodaran, *Investment Philosophies* (momentum & calendar effects), Wiley.
9. Basilico/E. for square-root market impact; Binance API documentation, and CCXT
   library documentation.

---

## Appendix — Reproduction

```bash
git clone https://github.com/sakshamverma2030/Algorithmic-Trading-Framework-in-Crypto-Python.git
pip install -r requirements.txt
python main.py fetch --symbol BTC/USDT --timeframe 1h --days 180
python scripts/compare_all.py --symbol BTC/USDT --timeframe 1h --days 180
python main.py portfolio --source exchange --timeframe 1h --days 120

# Section 5.5 - selection sweep and the adopted configuration
python main.py fetch --symbol BTC/USDT --timeframe 1h --days 1100
python scripts/timeframe_sweep.py           # 384 runs -> reports/timeframe_sweep_BTCUSDT.csv
python main.py backtest --days 1100         # adopted defaults: 4h, Kijun trail, no take-profit
python main.py backtest --symbol ETH/USDT --days 1100   # cross-asset check

python -m pytest          # 53 tests (the ADF test needs statsmodels installed)
```