"""Simplify wording of PROJECT_PRESENTATION.pptx in place.
- Does NOT regenerate the deck -> manual edits are preserved.
- Each rule: (slide#, shape_idx, para_idx, exact_current_text, new_text).
- If current text differs from exact_current_text (user edited it), the rule is skipped safely.
Run: python scripts/simplify_ppt.py
"""
from pptx import Presentation

FILE = "PROJECT_PRESENTATION.pptx"

E = []  # (slide, shape, para, old, new)

def rule(slide, shape, para, old, new):
    E.append((slide, shape, para, old, new))

# ---------------- Slide 2: Overview ----------------
rule(2, 5, 1,
     "▸  A Python framework that BULLDS and TESTS crypto trading strategies — then runs them in live/paper mode",
     "▸  Software that builds and tests crypto trading strategies, then runs them automatically on live prices (fake money)")

rule(2, 5, 3,
     "▸  Realistic backtesting: real fees, spread, slippage — so results can be trusted",
     "▸  Simulated trading includes real-world costs (fees, spread, slippage) — so the numbers are trustworthy")

rule(2, 5, 4,
     "▸  No look-ahead bias — every signal uses only PAST data (verified by 53 tests)",
     "▸  No cheating: every decision uses data from BEFORE the trade, never the future (verified by 53 tests)")

rule(2, 5, 6,
     "▸  8 strategies across the  (technical + ML + statistical arbitrage)",
     "▸  8 strategies covering technical analysis, machine learning, and statistical arbitrage")

rule(2, 5, 8,
     "▸  An honest report: single-symbol timing was negative in the tested window, the K-Means portfolio beat buy-and-hold by +3.4%",
     "▸  Honest result: trading a single coin lost money in the test window, while the K-Means portfolio beat buy-and-hold by +3.4%")

rule(2, 6, 0,
     "Talk the walk: data ➜ signals ➜ backtest ➜ risk ➜ live paper — one codebase, no hand-waving.",
     "Data → signals → test → risk → live: everything works together, and nothing is faked.")

# ---------------- Slide 3: Architecture banner ----------------
rule(3, 35, 0,
     "The SAME strategy code runs in backtest and in the live engine — so backtest results match real-time behaviour.",
     "The same strategy code runs in both testing and live trading — so tested results match reality.")

# ---------------- Slide 4: Data ----------------
rule(4, 5, 1,
     "▸  REST paginated OHLCV download — retry/back-off, gap detection, incremental updates",
     "▸  Downloads candle prices (open/high/low/close/volume) from Binance, with retries and gap detection")

rule(4, 5, 2,
     "▸  SQLite cache (data/market_data.sqlite) + CSV export per symbol / timeframe",
     "▸  Saves data in a local database (SQLite) and CSV files — per coin and timeframe")

rule(4, 5, 3,
     "▸  Any timeframe 1m → 1d; any symbol Binance lists (BTC, ETH, SOL, ADA, XRP …)",
     "▸  Any timeframe (1 minute to 1 day) and any coin on Binance (BTC, ETH, SOL, ADA, XRP …)")

rule(4, 5, 5,
     "▸  Synthetic regime-switching generator — used by the 53 unit tests",
     "▸  A generator that creates fake market data — used by the 53 tests")

rule(4, 5, 7,
     "▸  BTC/USDT 1h — 4,319 bars, 180 days (2026-03-15 → 2026-09-11), period return +8.95%",
     "▸  BTC/USDT hourly — 4,319 candles over 180 days (Mar→Sep 2026); BTC itself rose +8.95% in that period")

rule(4, 5, 8,
     "▸  Multi-asset universe for the K-Means portfolio: BTC, ETH, SOL, ADA, XRP (1h, 120 days)",
     "▸  5 coins for the portfolio experiment: BTC, ETH, SOL, ADA, XRP (hourly, 120 days)")

# ---------------- Slide 5: Signals ----------------
rule(5, 5, 1,
     "▸  Ichimoku 5-line cloud — Tenkan/Kijun/Senkou A/B/Chikou + Kumo, the trend backbone",
     "▸  Ichimoku Cloud — 5 lines that show the trend's direction and strength")

rule(5, 5, 2,
     "▸  RSI + divergence detector — momentum not confirming price = early warning",
     "▸  RSI + divergence — an early 'danger' flag when momentum no longer matches price")

rule(5, 5, 3,
     "▸  Aroon, Bollinger Bands, Wilder ATR (used for stops & position sizing)",
     "▸  Aroon, Bollinger Bands, and ATR (used for stop-losses and how much to bet)")

rule(5, 5, 5,
     "▸  Hurst exponent (R/S): H>0.5 = trending, H<0.5 = mean-reverting",
     "▸  Hurst exponent — decides if a market trends (keep betting) or mean-reverts (prices snap back)")

rule(5, 5, 6,
     "▸  ADF test (statsmodels): is the pairs-spread stationary? (gate for pairs trading)",
     "▸  ADF test — checks if two coins stay linked over time (needed for pairs trading)")

rule(5, 5, 7,
     "▸  K-Means (sklearn, scipy fallback): clusters assets by volatility / momentum / volume",
     "▸  K-Means — groups similar coins together, using a simple machine-learning model")

rule(5, 5, 9,
     "▸  Every indicator uses only PAST bars — vectorised, strictly trailing, no look-ahead",
     "▸  Fairness rule: indicators only use PAST prices — the future is never used")

# ---------------- Slide 6: Strategies ----------------
rule(6, 6, 0, "ADVANCED  (ML + stat-arb)", "ADVANCED  (machine learning + statistical arbitrage)")

rule(6, 7, 0,
     "▸  Ichimoku — TK cross + Kumo filter + Chikou confirm (long/short)",
     "▸  Ichimoku — buy/sell when the trend lines cross, confirmed by the cloud and price")

rule(6, 7, 1,
     "▸  Calendar anomalies — hour / day-of-week seasonality",
     "▸  Calendar effect — some hours or days of the week rise or fall more often")

rule(6, 7, 2,
     "▸  Aroon / RSI divergence — trend gate + divergence entry",
     "▸  Aroon + RSI divergence — trade only when trend and momentum agree")

rule(6, 7, 3,
     "▸  Time-series momentum — return threshold + SMA filter",
     "▸  Momentum — buy coins that recently went up, skip the ones that fell")

rule(6, 8, 0,
     "▸  Pairs trading — ADF cointegration, rolling OLS hedge, z-score entries",
     "▸  Pairs trading — when two similar coins drift apart, buy the cheap one and sell the expensive one")

rule(6, 8, 1,
     "▸  Hurst regime + RSI — trend → momentum; reversion → RSI extremes",
     "▸  Hurst + RSI — switches style automatically between trending and snapping-back markets")

rule(6, 8, 2,
     "▸  K-Means asset clustering — rank clusters, trade the best",
     "▸  K-Means clustering — group the coins, then only trade the best group")

rule(6, 8, 3,
     "▸  Momentum-alpha portfolio — top-N equal weight, periodic rebalance",
     "▸  Portfolio momentum — hold the top few winners, equal weight, rebalanced regularly")

rule(6, 10, 0,
     "All strategies share ONE signal format (long_entry / exit / ATR / Kumo) — every engine can run every strategy.",
     "One signal format for every strategy — so a strategy tested in the lab runs live identically.")

# ---------------- Slide 7: Backtest ----------------
rule(7, 8, 0, "For FAST research & parameter sweeps", "For quick research and 'what if' tests")
rule(7, 9, 0,
     "milliseconds  •  full notional  •  proportional costs  •  no stops",
     "very fast to run  •  ideal for testing many settings")
rule(7, 13, 0,
     "For REALISTIC evaluation — mirrors the live bot",
     "For final, trustworthy results — works exactly like the live bot")
rule(7, 14, 0,
     "signals on close → fills at NEXT open  •  risk-sized  •  ATR stops  •  costs + impact  •  circuit breaker",
     "fills only on the NEXT candle  •  sized by risk  •  stop-losses  •  all costs included")
rule(7, 15, 0,
     "▸  Fill honesty: a gap through the stop fills at the open; if stop and target both lie inside one bar, the stop is assumed hit first",
     "▸  Fair fills: if price gaps through the stop level we fill at the open — when stop and target hit in one candle, the stop wins")
rule(7, 15, 1,
     "▸  Cost model per trade: fee 0.1% + half-spread + slippage + square-root impact  I = Y·σ·√(Q/V)",
     "▸  Costs per trade: 0.1% fee + spread + slippage + market impact (big orders move the price a little)")

# ---------------- Slide 8: Risk ----------------
rule(8, 5, 1,
     "▸  Fixed-fractional — risk 1% of equity per trade (Q = f·E / distance to stop)",
     "▸  Fixed-fractional — risk only 1% of the account on any single trade")
rule(8, 5, 2,
     "▸  Half-Kelly — sizes from rolling R-multiples, capped, with fixed-fractional fallback",
     "▸  Half-Kelly — sizes bets from past win/loss results, with a safety cap")
rule(8, 5, 4,
     "▸  ATR-stop (entry ∓ k·ATR) or Kumo-edge stop; take-profit at ATR/r:m multiple",
     "▸  Stop-loss set by volatility (ATR) or the cloud edge; take-profit locks in winners")
rule(8, 5, 5,
     "▸  Trailing: Chandelier (ATR) or Kijun-sen — locks in winners",
     "▸  Trailing stop — lets winners run but protects profit along the way")
rule(8, 5, 7,
     "▸  Circuit breaker: flatten & halt at 20% drawdown from high-water mark; also at 5% daily loss",
     "▸  Circuit breaker — stop everything if the account falls 20% from its best, or 5% in a day")
rule(8, 5, 8,
     "▸  Stops are monitored on EVERY order-book tick (not only at bar close)",
     "▸  Stops are checked constantly — not just once per candle")
rule(8, 6, 0,
     "Risk logic is SHARED between backtest and live engine — what you simulate is what you trade.",
     "The same safety rules run in tests and in live trading — no surprises later.")

# ---------------- Slide 9: Live engine ----------------
rule(9, 8, 0, "Real Binance prices, simulated fills", "Real prices, fake money")
rule(9, 9, 0,
     "Fill by walking the live order book (VWAP over levels). No account needed.",
     "Orders fill against the live order book. No account or API keys needed.")
rule(9, 14, 0,
     "Testnet keys + EXCHANGE_USE_TESTNET=true. Test funds only.",
     "Test-only money. Uses Binance's test server with your own testnet keys.")
rule(9, 19, 0,
     "Real keys + typed 'I UNDERSTAND' + smaller sub-account. Real money!",
     "Real money! Needs a typed confirmation and a small sub-account.")
rule(9, 20, 0,
     "▸  asyncio engine: ccxt.pro WebSocket / REST / replay feeds — lossless bar queue + order-book mailbox (no event-loop starvation)",
     "▸  Fast streaming engine that never misses a price update — live websocket feed, REST, or recorded replay")
rule(9, 20, 1,
     "▸  Closed-candle signals only (no repainting); warm-up then stream; journal ▶ SQLite (fills / trades / equity / events)",
     "▸  Signals only after a candle CLOSES (no peeking); every fill and trade is saved to a journal")

# ---------------- Slide 10: Evidence ----------------
rule(10, 5, 0,
     "57 round-trip trades • 116 fills • order-book-walking fills with 2 bps slippage • exits: take_profit / stop_loss",
     "57 trades • 116 fills • realistic order-book fills (2 bps slippage) • exits: take-profit or stop-loss")

# ---------------- Slide 11: Results 1 ----------------
rule(11, 8, 0, "best defence: MaxDD −5.5%", "smallest loss — worst dip only −5.5%")
rule(11, 17, 0, "second-best PF 0.74", "second-best profile (profit factor 0.74)")
rule(11, 26, 0, "7.8% time in market", "traded only 7.8% of the time")
rule(11, 35, 0, "over-trading → cost-heavy", "traded too often — costs ate the profits")
rule(11, 44, 0, "paid ~$7,800 in fees+slippage", "lost ~$7,800 to fees + slippage alone")
rule(11, 50, 0,
     "HONEST NEGATIVE RESULT: in a mildly rising market every long-only timing strategy lost — turnover vs costs is what decides.",
     "Honest result: in a gently rising market, every single-coin timing strategy still lost money — it was trading costs, not the market, that decided the outcome.")

# ---------------- Slide 12: Portfolio ----------------
rule(12, 5, 1,
     "▸  Universe: BTC, ETH, SOL, ADA, XRP (1h, 120 days) — K-Means (k=3) clusters by volatility / momentum / log-volume",
     "▸  Watched 5 coins (BTC, ETH, SOL, ADA, XRP) over 120 days; K-Means grouped them into 3 'personality' groups")
rule(12, 5, 2,
     "▸  Cluster ranking → hold top-2 assets equal-weight, rebalance every 10 bars → 348 rebalances",
     "▸  Then held the best 2 coins of the best group, rebalanced every 10 candles (348 times)")
rule(12, 5, 5,
     "▸  MULTI-ASSET + CROSS-SECTIONAL RANKING added +3.4% — while single-asset hold lost money",
     "▸  Picking the best group of coins beat holding one coin by +3.4%")
rule(12, 5, 7,
     "▸  Diversification across clustered assets beats timing one asset in choppy markets",
     "▸  Spreading bets across groups of coins beats trying to time a single coin in a choppy market")

# ---------------- Slide 13: Validation ----------------
rule(13, 5, 1,
     "▸  No look-ahead: signals on truncated history == first rows of full-history signals (static + dynamic)",
     "▸  No cheating: a short history gives the SAME signals as the full history")
rule(13, 5, 2,
     "▸  Backtest ↔ live parity: replay bars through the real engine reproduce the backtester's trades exactly",
     "▸  Backtest = live: replaying old data through the real bot reproduces the same trades")
rule(13, 5, 3,
     "▸  Accounting identities: equity = cash + position·close, Σ trade PnL = equity change",
     "▸  Math checks out: account = cash + holdings, and trade profits match the balance change")
rule(13, 5, 4,
     "▸  Indicator correctness: textbook Ichimoku values, Wilder ATR, Bollinger 2σ geometry",
     "▸  Indicators matched against known textbook answers (Ichimoku, ATR, Bollinger)")
rule(13, 5, 5,
     "▸  Strategy behaviour: momentum causality, K-Means regimes bounded + causal, cointegration recovers a known beta",
     "▸  Strategy logic verified — including the machine-learning and pairs-trading parts")
rule(13, 5, 6,
     "▸  Config safety: bad settings → ConfigError (calendar without features, empty universe, bad Hurst thresholds)",
     "▸  Bad settings raise clear errors instead of silently misbehaving")

# ---------------- Slide 15: Limitations ----------------
rule(15, 5, 1,
     "▸  Sample negative: every single-symbol strategy lost in the tested 180-day window once costs are included",
     "▸  Result was negative: every single-coin strategy lost money once costs were included")
rule(15, 5, 3,
     "▸  scikit-learn DLL blocked on the dev machine (Windows App Control) → scipy kmeans2 fallback, identical output",
     "▸  A Windows security setting blocked one ML library; a backup library gave identical results")
rule(15, 5, 4,
     "▸  Stops are client-side (halt if the bot halts); perpetual-futures funding approximated by a constant rate",
     "▸  Stops live in our software (if the bot stops, the stops stop); one futures fee is approximated")

# ---------------- Apply ----------------
prs = Presentation(FILE)
done, skipped = 0, 0
for slide_no, shape_no, para_no, old, new in E:
    slide = prs.slides[slide_no - 1]
    try:
        para = slide.shapes[shape_no].text_frame.paragraphs[para_no]
    except (IndexError, AttributeError):
        print(f"skip: no shape/para S{slide_no} sh{shape_no} p{para_no}")
        skipped += 1
        continue
    cur = "".join(r.text for r in para.runs)
    if cur != old:
        print(f"skip: text differs S{slide_no} sh{shape_no} p{para_no}")
        print(f"   expect: {old[:50]!r}  vs  found: {cur[:50]!r}")
        skipped += 1
        continue
    para.runs[0].text = new
    for extra in para.runs[1:]:
        extra._r.getparent().remove(extra._r)
    done += 1

prs.save(FILE)
print(f"\nApplied: {done}   Skipped: {skipped}   Total rules: {len(E)}")
print("Saved:", FILE)