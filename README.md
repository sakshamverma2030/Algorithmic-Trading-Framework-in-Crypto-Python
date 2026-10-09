# Crypto Algo Trading & Backtesting System

An end-to-end quantitative trading system for **cryptocurrency markets** built around the
**Ichimoku Kinko Hyo** strategy, extended with a full intermediate & advanced strategy
set (calendar anomalies, Aroon/RSI
divergence, K-Means asset clustering, cointegrated pairs trading, Hurst-exponent regime
filtering, long-only momentum / alpha portfolios). It covers data ingestion, vectorised
and event-driven backtesting with realistic execution costs, risk management,
performance analytics and charts, and an asynchronous real-time engine that trades
automatically in **paper**, **demo (exchange testnet)** or **real** mode.

Crypto data comes live from Binance (CCXT); backtests run on exchange, synthetic or CSV
data offline.

```
Data fetch (CCXT / SQLite) -> Signals (Ichimoku + overlays + ML strategies) -> Backtests (vectorised + event-driven)
        -> Analytics & charts -> Real-time engine (paper / demo / real)
```

> Educational software. Nothing here is investment advice, and past or simulated
> performance says nothing about future results. Paper-trade first; if you ever use real
> money, start with a small amount you can afford to lose.

---

## 1. Project layout

### 1.1 Source modules

| File | Responsibility |
|---|---|
| `config.py` | Frozen dataclass configuration: exchange, data, strategy, costs, risk, backtest, live settings; Ichimoku presets (standard/crypto/crypto_slow/dynamic); intermediate & advanced strategy params — `CalendarConfig`, `DivergenceConfig`, `HurstConfig`, `PortfolioConfig`, `MomentumConfig`, `PairsConfig`; env/`.env` secrets; JSON-lines structured logging setup. All W=validation runs in dataclass `__post_init__` (bad config → `ConfigError`). |
| `data_loader.py` | One `DataLoader(cfg)` class: CCXT paginated OHLCV download (retry/back-off, gap report), SQLite cache (idempotent upsert, incremental refresh), CSV import/export, optional Parquet, and a synthetic regime-switching GBM generator (`MarketRegime`, hidden Markov chain over volatility regimes) for offline research and tests. |
| `indicators.py` | Pure-vectorised indicator library: Ichimoku all five lines + Kumo + projection, Wilder ATR (TA-Lib fallback), RSI + **causal** divergence detector, Aroon, Bollinger Bands, rescaled-range **Hurst exponent**, **ADF** stationarity test (statsmodels), and an adaptive volatility-regime Ichimoku (quantile or K-Means labels via scipy `kmeans2`, sklearn optional). Exposes `HAS_SKLEARN` / `HAS_STATSMODELS` feature flags. |
| `strategy.py` | `IchimokuStrategy` — the core long/short/exit radar rules evaluated on close-of-bar t, a vectorised position state machine, and a latest-bar `SignalSnapshot` for the live engine. Defines the **signal-frame contract** (`long_entry`, `short_entry`, `exit_long`, `exit_short`, `atr`, `cloud_top`, `cloud_bottom`, ...) that every strategy and backtest engine shares. |
| `momentum.py` | `MomentumStrategy` — long/short time-series momentum (`Close_t/Close_{t-lookback}-1` vs entry/exit thresholds, optional SMA filter), reuse of the exact same signal-frame contract + risk engine. |
| `pairs.py` | `CointegrationAnalyzer`, `check_pair` and the pairs backtest machinery — rolling OLS hedge ratio on log prices, stationary spread, z-score entry/exit/stop bands, forced close on max-hold; returns a standard `BacktestResult`. |
| `intermediate_strategies.py` | "Intermediate" strategy layer: re-exports `IchimokuStrategy`/`SignalSnapshot`, adds `CalendarAnomalyStrategy` (causal expanding-mean hourly / day-of-week seasonal profiles) and `AroonRsiDivergenceStrategy` (Aroon trend gate + RSI divergence entries). |
| `advanced_ml_strategies.py` | "Advanced / ML" strategy layer: `KMeansAssetClusterer` (volatility/momentum/volume features — sklearn `KMeans`, scipy `kmeans2` fallback), `rolling_hurst` + `HurstTrendStrategy` (H>0.5 trend / H<0.5 reversion regime gate combined with RSI), `MomentumAlphaPortfolio` (cross-sectional `rank(ret2)+rank(retL)-2·rank(vol)`, equal-weight top-N, periodic rebalance), plus re-exports of the pairs modules. |
| `risk.py` | Cost model (taker/maker fees, half-spread, slippage, square-root market impact `I=Y·σ·√(Q/V)`, financing), fixed-fractional & half-Kelly position sizers, ATR/Kumo stops, Chandelier/Kijun trailing stops, conservative intrabar fill rules, and the drawdown/daily-loss circuit breaker. Shared verbatim by backtester and live engine. |
| `backtester.py` | `VectorizedBacktester` (full-notional, proportional costs, ms-fast, drives the optimiser), `EventDrivenBacktester` (bar-by-bar mirror of the live engine: signals on close → fills at next open, risk sizing, stops, breakers), `compare` preset runner, and in-sample/out-of-sample `ParameterOptimizer`. Returns `BacktestResult` (equity, trades, signals, metadata). |
| `analytics.py` | `PerformanceAnalyzer` + `ChartVisualizer`: Sharpe/Sortino/Calmar/CAGR, drawdown depth & duration, probabilistic Sharpe, VaR/CVaR, alpha/beta, full trade statistics, and all Matplotlib charts (Ichimoku candlestick, dashboard, trade analysis, optimisation heatmaps, preset comparison). |
| `live_trader.py` | asyncio real-time engine: ccxt.pro WebSocket / REST / replay feeds (lossless bar queue + latest-value order-book mailbox, no event-loop starvation), warm-up, closed-candle signal evaluation, order-book-walking paper fills, CCXT testnet/live execution with precision/notional checks, net-of-fee accounting, heartbeats and watchdog, SQLite trade journal (`fills`, `trades`, `equity`, `events`). |
| `live_paper_trader.py` | Spec-facing facade re-exporting the real-time engine (WebSockets + paper simulator + CCXT live templates) so the project exposes the expected `live_paper_trader` module name. |
| `main.py` | `TradingBotCLI` and everything a user types: interactive menu and CLI subcommands (`fetch`, `backtest`, `compare`, `optimize`, `momentum`, `pairs`, `calendar`, `divergence`, `hurst`, `portfolio`, `live`, `pipeline`), config building from CLI flags, report/chart orchestration. |

### 1.2 Tests & tooling

| File | Responsibility |
|---|---|
| `tests/test_core.py` | Core engine tests: textbook Ichimoku values, TA-Lib-compatible Wilder ATR, no-look-ahead (static + dynamic), position FSM, cost model, sizers, intrabar stop/target rules, trailing ratchet, circuit breaker, accounting identities, vectorised timing, metrics, data validation, SQLite/CSV round trips, order-book walking, fee-adjusted fills, CLI parsing, **backtest-vs-live-replay parity** (22 tests). |
| `tests/test_extra_strategies.py` | RSI divergence overlay, K-Means regime labels (causality + bounds), momentum strategy (causality, trend behaviour, live selection) and cointegrated-pairs behaviour on synthetic data. |
| `tests/test_resume_strategies.py` | Strategy behaviour tests: Aroon bounds/monotonicity, Hurst trend-vs-revert separation, rolling-Hurst causality, ADF stationarity detection, Bollinger geometry, calendar bucket causality, divergence signal completeness + require/relax modes, Hurst strategy runs clean, K-Means clusters assets on features, momentum-alpha portfolio rebalances into top-N and rejects empty universes (14 tests). |
| `scripts/compare_all.py` | Cross-strategy scorecard: runs every single-symbol strategy on one dataset through the event-driven engine, writes `strategy_comparison.csv` + a normalised equity chart into `reports/<SYMBOL>_<TF>_<source>/`. |
| `scripts/timeframe_sweep.py` | Timeframe/settings selection sweep: 384 backtests (timeframe x preset x cross window x exit style x stop width) with in-sample/out-of-sample split; writes `reports/timeframe_sweep_<SYMBOL>.csv` and prints the robust candidates. This is the evidence behind the defaults in `config.py` (section 6.1). |
| `scripts/paper_to_pdf.py` | Renders `RESEARCH_PAPER.md` → `RESEARCH_PAPER.pdf` (markdown → HTML → xhtml2pdf) with inline styling. |
| `requirements.txt` | Core deps (numpy, pandas, scipy, matplotlib, ccxt, pytest) + ML stack (`scikit-learn` K-Means, `statsmodels` ADF); optional TA-Lib / pyarrow commented. |

### 1.3 Documents & evidence

| File | Responsibility |
|---|---|
| `README.md` | This file — setup, quick start, trading modes, strategy maths, backtester design, test suite, honest status/limitations. |
| `CAPSTONE.md` | Capstone report: problem statement, methodology, per-course strategy mapping, real-data results tables, live-demo evidence and honest limitations. |
| `CAPSTONE_ABSTRACT.md` | 250-word abstract + 12-slide presentation outline + viva demo script. |
| `RESEARCH_PAPER.md` / `.pdf` | IEEE-style research paper: abstract, intro, related work, methodology, validation protocol, real results, discussion, conclusion, references. |
| `reports/BTCUSDT_1h_exchange/` | Committed real Binance backtest evidence: per-strategy report `.txt`/`.json`, equity curves CSV, trades CSV, metadata, chart PNGs, `strategy_comparison.*`, portfolio equity/rebalances, preset comparison, optimisation heatmap. |
| `data/trade_journal_snapshot.sqlite` | Snapshot of the live engine's journal committed as evidence: 57 round-trip trades / 116 fills across paper + replay sessions, incl. a ~13 h overnight live run (5 trades); the live DB `data/trade_journal.sqlite` stays git-ignored. |

### 1.4 Runtime outputs (git-ignored unless committed as evidence)

| Path | Contents |
|---|---|
| `reports/<SYMBOL>_<TF>_<source>/` | Backtest reports/charts per run (curated evidence is force-committed). |
| `data/trade_journal.sqlite` | Live/paper fills, trades, equity and events (WAL checkpoint available). |
| `data/market_data.sqlite` | OHLCV cache; `data/<SYMBOL>_<TF>.csv` exports. |
| `logs/trading_bot.jsonl` | JSON-lines structured logging; `logs/live_paper.*` engine console output. |

```mermaid
flowchart LR
    A[Exchange / CSV / Synthetic] --> B[data_loader<br/>validate + SQLite]
    B --> C[indicators<br/>Ichimoku + ATR]
    C --> D[strategy<br/>signals + FSM]
    D --> E[backtester<br/>vectorised / event-driven]
    R[risk<br/>costs, sizing, stops, breaker] --> E
    E --> F[analytics<br/>metrics + charts]
    D --> G[live_trader<br/>async engine]
    R --> G
    G --> H[(trade journal<br/>SQLite)]
    F --> I[reports<br/>CSV + PNG evidence]
    C -.-> S[intermediate & advanced strategies]
    S --> D
```

## 2. Setup

```bash
python -m pip install -r requirements.txt
```

A virtual environment (`python -m venv .venv`) is optional; see the Windows note below before
using one.

* `ccxt` is required for anything that talks to an exchange: downloading real data, and the paper,
  demo and real trading modes. Backtests on `--source synthetic` / `--source csv` and the offline
  replay work without it.
* ML strategy stack: `scikit-learn` (K-Means asset clustering — code falls back to scipy
  `cluster.vq.kmeans2` when import fails) and `statsmodels` (ADF stationarity test for pairs
  trading). Both are listed in `requirements.txt`.
* Optional extras: `TA-Lib` (C-speed rolling extremes and ATR; the NumPy/pandas fallback gives
  identical numbers) and `pyarrow` (Parquet export).
* **Windows troubleshooting.** *"DLL load failed ... An Application Control policy has blocked
  this file"* means Windows Smart App Control rejected a newly downloaded native library (numpy,
  pyarrow, ...). It is not a bug in this project.
  * Inside a virtual environment: leave it (`deactivate`) and use the Python installation whose
    packages already load. Fresh copies in a new venv are often the ones that get blocked.
  * If only pyarrow is blocked: `python -m pip uninstall -y pyarrow` (it is optional here).
  * If a package cannot be loaded at all, run the project under WSL (Ubuntu on Windows) or on a
    Linux server, where Smart App Control does not apply.

  `main.py` detects this error and prints these hints.

## 3. Quick start

```bash
python main.py                                         # interactive menu
python main.py pipeline --source synthetic             # full pipeline offline, no exchange needed
python main.py pipeline                                # same on real Binance BTC/USDT 4h data
python main.py fetch --symbol ETH/USDT --timeframe 15m --days 180
python main.py backtest --preset crypto --sizing kelly --stop cloud --trailing kijun
python main.py backtest --strategy momentum           # same engine, momentum signals
python main.py backtest --strategy both               # run and report both
python main.py compare                                 # standard vs crypto vs crypto_slow vs dynamic
python main.py optimize --metric sharpe                # IS/OOS grid search + heatmap
python main.py live                                    # paper trading on live market data
python main.py live --source synthetic --replay-bars 800   # offline paper-trading replay
python main.py momentum --source synthetic                      # momentum backtest
python main.py calendar --source synthetic                      # calendar-anomaly backtest
python main.py divergence --source synthetic                    # Aroon / RSI divergence backtest
python main.py hurst --source synthetic                         # Hurst-exponent regime filter + RSI
python main.py portfolio --source synthetic                     # K-Means clustering + momentum-alpha portfolio
python scripts/compare_all.py --symbol BTC/USDT                 # all-strategy scorecard + equity chart
```

Useful flags: `--allow-short` (derivatives / margin only), `--cross-lookback N`, `--no-chikou`,
`--kumo-twist`, `--capital`, `--risk-per-trade`, `--max-dd`, `--fee`, `--slippage-bps`,
`--timeframe`, `--flatten-on-exit`, `--no-plots`, `--show`. Run `python main.py <command> -h` for
the full list.

Outputs go to `reports/<SYMBOL>_<TF>_<source>/`: the text/JSON report, equity-curve and trade
CSVs (the table view behind every chart), configuration metadata and PNG charts. Logs are
JSON lines in `logs/trading_bot.jsonl`; live fills, trades and equity go to
`data/trade_journal.sqlite`.

## 4. Trading modes

**"Live" means real-time, not real money.** The real-time engine runs in one of three modes, and
real money is only used when you explicitly configure it.

| Mode | What happens | Program must keep running | Account / API keys | Real money |
|---|---|---|---|---|
| Backtest | Replays history, results in seconds | No | None | No |
| **Paper** (default) | Real market prices, simulated fills and balance | Yes | None | No |
| **Demo** | Real orders on Binance Spot Testnet, with testnet funds | Yes | Testnet keys | No |
| **Real** | Real orders on your Binance account | Yes | Real keys | **Yes** |

### 4.1 Choosing the mode

The mode is set by `--mode` (or `TRADING_MODE` in `.env`), and demo vs real by
`EXCHANGE_USE_TESTNET` in `.env`:

| Mode | `.env` | Command |
|---|---|---|
| Paper | nothing needed | `python main.py live` |
| Demo | testnet keys, `EXCHANGE_USE_TESTNET=true` | `python main.py live --mode LIVE_TRADING` |
| Real | real keys, `EXCHANGE_USE_TESTNET=false` | `python main.py live --mode LIVE_TRADING` |

Demo and real use the same command; only the keys and `EXCHANGE_USE_TESTNET` differ. At start-up
the console shows **TESTNET** or **MAINNET - REAL MONEY**, and you must type `I UNDERSTAND` to
continue. The interactive menu (option 5) uses `TRADING_MODE` from `.env`, and runs paper trading
if it is not set.

Real money needs all three of: real API keys in `.env`, `EXCHANGE_USE_TESTNET=false`, and the
typed confirmation. Remove any one of them and no real order can be sent.

### 4.2 What the bot does automatically

Once started, no manual action is needed. On every closed candle the bot computes the Ichimoku
signals and buys or sells by itself, sizing each position from your risk settings. It watches
stop-loss and take-profit on every order-book update, enforces the drawdown circuit breaker, and
records every fill and trade in the journal.

It trades only while the program is running. If the computer sleeps or the program stops, trading
stops, and so do the stop-losses, because stops are managed by the bot rather than placed on the
exchange. For long unattended runs, use a cloud server (VPS). Stop with `Ctrl+C`; add
`--flatten-on-exit` to close any open position on shutdown.

Signals use closed candles, so on the default 4h timeframe the first decision arrives when the
current 4-hour candle closes. For a quicker demo use `--timeframe 5m` or `15m`, keeping in mind
that the timeframe changes the strategy's behaviour (see section 6.1).

### 4.3 Paper trading (no account)

```bash
python main.py live
```

Uses live Binance market data and simulates each market order by walking the real order book.

### 4.4 Demo account: Binance Spot Testnet

1. Open https://testnet.binance.vision, choose **Log In with GitHub**, then **Generate
   HMAC_SHA256 Key**. Copy the API key and secret; the secret is shown only once. The testnet
   account comes with test balances.
2. Copy `.env.example` to `.env` and fill in:
   ```ini
   EXCHANGE_API_KEY=your_testnet_api_key
   EXCHANGE_API_SECRET=your_testnet_secret
   EXCHANGE_USE_TESTNET=true
   ```
3. Run `python main.py live --mode LIVE_TRADING --flatten-on-exit` and type `I UNDERSTAND`.

Testnet prices and liquidity differ from the real market, so fill prices there can look odd. Use
the demo to confirm that orders are placed, filled and journaled correctly, and use paper mode
to judge execution quality (section 5).

### 4.5 Real account

Do this only after paper and demo have run correctly for a while. Demo is not technically
required, but it catches problems with test funds instead of real ones.

1. Complete KYC on your Binance account.
2. **Profile -> API Management -> Create API**:
   * **Enable Spot & Margin Trading:** on.
   * **Enable Withdrawals:** always off. A leaked key then cannot move funds out.
   * Restrict the key to your IP address if you have a fixed IP.
3. Put the real keys in `.env` and set `EXCHANGE_USE_TESTNET=false`.
4. Use a dedicated sub-account holding only the trading capital, and start small (for example
   50-100 USDT). The bot sizes positions from the account's USDT + BTC balance, buys only with
   free USDT, and caps every order at `max_order_notional` (1,000 USDT by default, `config.py`).
5. Run `python main.py live --mode LIVE_TRADING --flatten-on-exit`. The banner must say
   **MAINNET - REAL MONEY**.

Keep API keys only in `.env` (git-ignored), never share or paste them anywhere, and check your
local tax rules before trading real funds (in India, for example, crypto gains are taxed at 30 %
with 1 % TDS).

## 5. Validating the results

Backtests and the demo account answer different questions.

**Is the backtest computed correctly?** This needs historical data only, no account:
* The test suite checks accounting identities, the absence of look-ahead, textbook Ichimoku values
  and the cost model (section 9).
* Manual cross-check for a report or viva: take 5-6 trades from the trades CSV, open the same
  symbol and timeframe on TradingView with Ichimoku (9, 26, 52, 26), and confirm each entry had
  price above the Kumo, a fresh Tenkan/Kijun cross and Close above the close 26 bars earlier.

**Does the same strategy behave the same in real time?** This is *forward testing* with paper or
demo mode:
1. Run the bot in paper (or demo) mode for a few weeks; every trade is saved in the journal.
2. Run the backtest over the same period.
3. Compare trade by trade. Entry and exit times should match; small price differences are slippage.

The same comparison is automated offline: `test_live_replay_reproduces_the_event_driven_backtest`
streams bars through the real engine and gets exactly the backtester's trades. With 1h candles
the strategy trades roughly once a week, so collecting 20-30 live trades takes one to two months.

## 6. Strategy

With HH_n / LL_n the highest high / lowest low of the last n bars:

| Line | Formula | Plotted |
|---|---|---|
| Tenkan-sen | (HH_9 + LL_9) / 2 | at t |
| Kijun-sen | (HH_26 + LL_26) / 2 | at t |
| Senkou Span A | (Tenkan + Kijun) / 2 | 26 bars ahead |
| Senkou Span B | (HH_52 + LL_52) / 2 | 26 bars ahead |
| Chikou Span | Close_t | 26 bars behind |

**Rules** (evaluated on the close of bar t, executed at the open of t+1):

* **Long entry:** Close > Kumo top **and** Tenkan crosses above Kijun **and** Close_t > Close_{t-26} (Chikou confirmation). An optional Kumo-twist filter is available.
* **Long exit:** Close < Kumo bottom **or** Tenkan crosses below Kijun.
* **Short entry / exit:** exact mirror (enabled with `--allow-short`, derivatives / margin only).

**Look-ahead bias.** The cloud visible at bar t is the span computed at t-26
(`senkou_a = senkou_a_lead.shift(26)`). The Chikou line drawn at t is *future* data
(Close_{t+26}), so it is used for plotting only. "Chikou above price" is evaluated causally
as Close_t > Close_{t-26}. A unit test proves that signals computed on truncated history equal
signals computed on the full history.

**Presets.** `standard` 9/26/52/26 (Hosoda's six-day trading week), `crypto` 10/30/60/30
(the same calendar logic on a 24/7 market), `crypto_slow` 20/60/120/30 (doubled, fewer
whipsaws). **`dynamic`** classifies each bar by the rolling percentile of NATR = ATR/Close:
low volatility uses fast lines, normal uses crypto lines, high volatility uses slow lines.
TK "crosses" caused purely by a regime switch are ignored.

### 6.1 Choosing the timeframe and settings

The defaults in `config.py` are not hand-picked. They come from `scripts/timeframe_sweep.py`,
which backtests **384 combinations** of timeframe x preset x TK-cross window x exit style x stop
width on three years of real BTC/USDT data (Oct 2023 - Oct 2026: a +197 % bull run followed by a
-20 % drawdown). Every run is split into in-sample (first 70 %) and out-of-sample (last 30 %), and
the whole sweep was repeated on an independent two-year window as a cross-check.

Two findings dominate everything else.

**1. A fixed take-profit was the main leak.** A trend system pays for many small losers with a few
large winners; capping winners at 4 x ATR while losers run to the stop inverts that maths.

| Exit style | Median out-of-sample return |
|---|---|
| no take-profit + Kijun trailing stop | **+2.9 %** |
| no take-profit + ATR (Chandelier) trailing | 0.0 % |
| take-profit + Kijun trailing | 0.0 % |
| take-profit, no trailing (the old default) | -1.2 % |

**2. Hourly candles lose to costs and whipsaw.** With the exit style above:

| Timeframe | Median trades (3y) | Median OOS return | Median OOS Sharpe | Median OOS drawdown |
|---|---|---|---|---|
| 1h | 224 | -3.4 % | -0.24 | -14.5 % |
| **4h** | **52** | **+3.9 %** | **0.77** | **-4.3 %** |
| 2h | 101 | +8.8 % | 1.42 | -5.3 % |
| 6h | 40 | -1.9 % | -0.63 | -3.5 % |
| 12h | 15 | +3.8 % | 1.41 | -1.5 % |
| 1d | 9 | +1.4 % | 0.95 | -1.1 % |

2h scored highest, but 4h + the `crypto` preset was the only combination that stayed **positive in
all four segments** (both halves of both data windows) with a usable number of trades, so that is
the default. 12h and 1d look fine but trade 9-15 times in three years, which is too small a sample
to trust.

**What changed**

| Setting | Before | Now |
|---|---|---|
| Timeframe | 1h | **4h** |
| Preset | standard 9/26/52/26 | **crypto 10/30/60/30** |
| TK-cross window | 1 bar | **3 bars** |
| Take-profit | on (4 x ATR) | **off** |
| Trailing stop | none | **Kijun-sen** |

**Result on three years of real BTC/USDT 4h data** (`python main.py backtest --days 1100`):

| | Strategy | Buy & hold |
|---|---|---|
| Total return | +19.3 % | +197.3 % |
| Max drawdown | **-13.0 %** | -53.5 % |
| Sharpe / Sortino | 0.77 / 1.31 | 1.02 / 1.46 |
| Profit factor | 1.57 | - |
| Trades / win rate | 67 / 26.9 % | - |
| Time in market | 13.3 % | 100 % |
| Annualised alpha | +3.5 % | - |

Year by year, strategy vs buy & hold: 2023 (Oct-Dec) -3.0 % vs +53.2 %; 2024 +14.9 % vs +121.1 %;
2025 -5.0 % vs -6.6 %; 2026 (Jan-Oct) **+12.7 % vs -6.4 %**.

**Old vs new defaults on exactly the same three years of BTC/USDT:**

| | Old (1h, standard, take-profit, no trailing) | New (4h, crypto, Kijun trailing, no take-profit) |
|---|---|---|
| Total return | **-33.9 %** | **+19.3 %** |
| In-sample / out-of-sample | -17.8 % / -19.6 % | +9.5 % / +9.0 % |
| Max drawdown | -37.2 % | -13.0 % |
| Trades | 240 | 67 |

**The profit is concentrated in very few trades - say this out loud in any report.** Of 67 trades,
18 win and 49 lose, and the median trade is -46 USD. Total net profit is 1,934 USD, while the single
best trade (17-26 Aug 2026, BTC 64.2k -> 78.4k) made 1,937 USD. **Remove that one trade and the
three years are flat (-2 USD); remove the best three and they are negative (-1,432 USD).** A long
right tail is how trend following is supposed to work, but with only 67 trades it also means the
headline number is not statistically reliable.

**Cross-asset check, and why the default is Kijun and not the higher-scoring ATR trail.** On three
years of BTC/USDT 4h, a Chandelier (ATR) trailing stop looks clearly better than the Kijun trail:
+34.8 % vs +19.3 %, profit factor 1.90 vs 1.57, and much less concentration (the best trade is 53 %
of the profit instead of 100 %). On ETH/USDT over the same three years it flips: the ATR trail loses
(-3.4 %, profit factor 0.90) while the Kijun trail is slightly positive (+4.6 %, profit factor 1.16,
max drawdown -13.1 % vs -68.0 % for buy-and-hold). One asset is not evidence, so the default stays
with the exit that is positive on both and best by median across all 384 runs. The honest reading is
that the edge is small and BTC-heavy: on ETH the profit is even more concentrated (the best trade is
about twice the total profit).

**Read this honestly.** A long-only trend filter cannot beat a market that triples - it is in the
market only 13 % of the time. What it does is lose far less when the trend turns: a quarter of
buy & hold's drawdown, and a positive 2026 while BTC fell. Judge it on drawdown and
risk-adjusted return, not on the headline number. The previous default (1h, take-profit, no
trailing) lost money on the same data, which is what the sweep was built to detect.

Re-run the selection yourself (writes `reports/timeframe_sweep_BTCUSDT.csv`):

```bash
python scripts/timeframe_sweep.py
```

## 7. Backtesting

* **Vectorised:** `r_strat = position.shift(1) * r - |Δposition| * c`. Full notional, proportional
  costs, no stops. It runs in milliseconds and drives the optimiser. Because crypto trades 24/7
  (next open ≈ last close), trading at the signal bar's close is equivalent to the next open.
* **Event-driven:** a bar-by-bar loop that mirrors the live engine. Orders fill at the next open,
  positions are risk-sized, stops and targets are checked inside each bar, trailing stops ratchet,
  shorts pay financing, and the circuit breaker is enforced. Two conservative OHLC rules apply:
  a gap through a stop fills at the open, and if a stop and a target are both inside one bar the
  stop is assumed to have been hit first.

**Execution costs**: `P_fill = P * (1 ± (half_spread + slippage + Y·σ_bar·√(Q/V_bar)))`, where the
last term is the square-root market-impact law. Fees are 0.1 % taker / maker by default.

**Risk management**

* Fixed-fractional sizing: `Q = f·E / |P_entry − P_stop|`, capped at `max_position_pct·E / P`.
* Kelly: `f* = W − (1 − W)/R` from the last N trades' R-multiples, half-Kelly, capped. Before
  enough trades exist it falls back to fixed fractional.
* Stops: ATR (`entry ∓ k·ATR`) or Kumo (far cloud edge ∓ buffer), with take-profit at an ATR multiple
  or a reward:risk multiple. Trailing: Chandelier (ATR) or Kijun-sen.
* Circuit breaker: flatten and halt at a 20 % drawdown from the high-water mark (cool-down, then
  the mark resets) and at a 5 % daily loss.

**Analytics**: total and annualised return, volatility, Sharpe, Sortino, Calmar, max drawdown and its
duration, Probabilistic Sharpe Ratio, skew/kurtosis, VaR/CVaR, alpha/beta/information ratio vs buy &
hold, win rate, profit factor, payoff, expectancy, R-multiples, trade durations, exit reasons, fees and
slippage paid. Annualisation uses 365 × 24 × 60 / bar-minutes periods, because crypto never closes.

**Charts**: Ichimoku candlestick chart with projected Kumo and buy/sell markers, full-period
overview, performance dashboard (cumulative return vs benchmark, equity curve, underwater chart,
monthly heatmap, trade distribution), trade analysis (rolling Sharpe, cumulative PnL, exit reasons,
R-multiples), optimisation heatmaps (in-sample vs out-of-sample), and preset comparison. The
palette was checked for colour-blind separation, and buy/sell markers differ by shape as well as
colour.

## 8. Real-time engine

```
feed (ccxt.pro WS | REST | replay) --bars (FIFO queue)--------> strategy task --+
                                   --order book (mailbox)-----> risk monitor ---+--> Paper | Live execution
heartbeat / watchdog (loop lag, stale feed)                           trade journal (SQLite)
```

* **No event-loop starvation.** Closed bars use a lossless queue. Order-book and tick bursts go
  through a latest-value mailbox (O(1) memory, never a backlog). pandas work runs in
  `asyncio.to_thread`, as do SQLite writes. Every wait has a timeout. A heartbeat measures
  loop lag, and streams reconnect with exponential back-off and jitter.
* **Signals use closed candles only** (no repainting), computed with exactly the strategy code
  the backtests use. Stops are monitored on every order-book update, with a bar-level safety net.
* **Paper mode** fills market orders by walking the live order book (VWAP over the consumed levels).
* **Live mode (demo and real)** uses CCXT with exchange precision and minimum checks, a hard
  per-order notional cap, client order ids with reconciliation after network errors, and a typed
  confirmation before start. Positions are recorded **net of exchange fees**: Binance charges
  spot fees in the asset received, so a 0.01 BTC buy credits 0.00999 BTC, and the bot never tries
  to sell more than the account holds. Buys are capped by the free quote balance.
* **Replay feed** streams historical bars (with an interpolated intrabar path) through the
  *real* engine; the parity test uses it.

## 9. Tests

```bash
python -m pytest
```

The **53 tests** are split across three files:

* `test_core.py` (22): textbook Ichimoku values, the TA-Lib-compatible Wilder ATR, the
  absence of look-ahead (static and dynamic modes), the position state machine, the cost
  model and its direction, both position sizers, intrabar stop/target rules, trailing-stop
  ratcheting, the circuit breaker, event-driven accounting identities (`equity = cash +
  side·qty·close`, Σ trade PnL = equity change), vectorised timing, metric formulas, trade
  statistics, data validation, the SQLite and CSV round trips, order-book walking,
  fee-adjusted live fills, CLI config parsing, and **live-replay vs backtest parity**.
* `test_extra_strategies.py`: RSI divergence (constructed patterns + causality), K-Means
  regime labels (bounded + causal), momentum signal causality/trend/live selection, and
  cointegration recovery on a generated pair.
* `test_resume_strategies.py` (14): Aroon bounds/monotonicity, Hurst trend-vs-mean-reversion
  separation, rolling-Hurst causality & bounds, ADF stationary-spotting, Bollinger 2σ
  geometry, calendar bucket causality + config validation, divergence signal completeness
  and require/relax modes, Hurst strategy vectorised run, K-Means asset clustering, and the
  momentum-alpha portfolio rebalancing and empty-universe rejection.

## 10. Status, limitations and next steps

* **Exchange connectivity has been validated against Binance.** Real 365/730-day OHLCV history
  downloads (REST), the live WebSocket feed, warm-up, and the real-time engine all run. Paper
  trading (order-book walking) and Binance Spot Testnet orders were exercised end-to-end, including
  fee-adjusted fills and the SQLite trade journal. A live overnight paper run (BTC/USDT 1m,
  Sep 11 → 12 2026) executed 5 trades with a ~13 h session; the full journal (paper + replay +
  live) holds 57 round-trip trades/116 fills and is committed at
  `data/trade_journal_snapshot.sqlite`. Real mainnet orders have not been run; only testnet funds
  should be used until the strategy itself is proven.
* **Backtest results on real data, after the timeframe selection (section 6.1).** The default
  Ichimoku configuration (4h, crypto preset, Kijun trailing stop, no take-profit) returns +19.3 %
  over three years of BTC/USDT with a -13.0 % max drawdown and profit factor 1.57, versus +197.3 %
  and -53.5 % for buy-and-hold: lower return, a quarter of the drawdown, and positive in the 2026
  downtrend (+12.7 % while BTC fell 6.4 %). It does **not** beat buy-and-hold over a bull market and
  is not meant to. The earlier defaults (1h, fixed take-profit, no trailing stop) lost money on
  BTC/USDT and ETH/USDT, which is what `scripts/timeframe_sweep.py` was written to detect; the
  in-sample parameter optimiser still overfits badly (IS Sharpe ~0.49 -> OOS ~-2.7), which is why
  settings are chosen on out-of-sample behaviour. The other strategies remain unvalidated: pairs
  trading lost over its 310-day window, and the K-Means + momentum-alpha portfolio underperformed a
  BTC buy-and-hold (-15.6 % on BTC/ETH/SOL/ADA/XRP over 120 days). Forward-test in paper/demo before
  any real funds.
* Documentation: `CAPSTONE.md` (capstone report), `CAPSTONE_ABSTRACT.md` (abstract + slide outline),
  `RESEARCH_PAPER.md`/`.pdf` (IEEE-style paper with real results).
* Synthetic data exists to test the software. Judge the strategy only on real exchange data,
  and look at the out-of-sample columns rather than the best in-sample cell.
* A correctly working bot is not a profitable one; profitability comes only from real-data
  backtests and forward testing.
* OHLC bars hide the intrabar path; the backtest fill rules are deliberately pessimistic.
* Stops are client-side (they stop if the bot stops). Perpetual-futures funding is approximated
  by a constant borrow rate.
* Possible extensions: a `reconcile` command comparing journal trades with a backtest of the same
  period, walk-forward optimisation, multi-asset portfolios, exchange-native OCO/stop orders, and a
  C++20 kernel (monotonic-deque rolling extremes) exposed through pybind11 for high-frequency
  timeframes.
