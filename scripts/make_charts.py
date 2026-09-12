"""Generate clean, human-friendly charts from ACTUAL project data for the PPT."""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("charts_ppt")
OUT.mkdir(exist_ok=True)

NAVY, BLUE, SKY, ORANGE, GREEN, RED, GREY = (
    "#0F2B52", "#1E5FB0", "#4FA8E6", "#E8742E", "#2EA543", "#C0392B", "#667788")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#CCD5E0",
                     "axes.linewidth": 0.8, "figure.dpi": 150})

# ---------------------------------------------------------------- 1. LIVE TRADES
tr = pd.read_csv("data/trades_snapshot.csv", parse_dates=["exit_time"])
tr = tr.sort_values("exit_time").reset_index(drop=True)
tr["cum"] = tr["net_pnl"].cumsum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
colors = [GREEN if v >= 0 else RED for v in tr["net_pnl"]]
ax1.bar(range(len(tr)), tr["net_pnl"], color=colors, width=0.85)
ax1.axhline(0, color="#33415C", lw=1)
ax1.set_title("Each real trade's profit (USD)", fontsize=13, fontweight="bold", color=NAVY, pad=10)
ax1.set_xlabel("Trade # (57 total)")
ax1.set_ylabel("Net P&L (USD)")
ax1.set_xticks([])
ax1.spines[["top", "right"]].set_visible(False)
ax1.text(0.02, 0.98, "winners: green  •  losers: red",
         transform=ax1.transAxes, fontsize=9, color=GREY, va="top")

ax2.plot(tr["cum"], color=NAVY, lw=2)
ax2.fill_between(range(len(tr)), tr["cum"], 0, where=tr["cum"] >= 0,
                 color=GREEN, alpha=0.15)
ax2.axhline(0, color="#33415C", lw=1)
ax2.set_title("Running total (equity curve)", fontsize=13, fontweight="bold", color=NAVY, pad=10)
ax2.set_xlabel("Trade #")
ax2.set_ylabel("Cumulative P&L (USD)")
ax2.set_xticks([])
ax2.spines[["top", "right"]].set_visible(False)
total = tr["net_pnl"].sum(); wins = (tr["net_pnl"] > 0).sum()
ax2.text(0.03, 0.95, f"final: {total:+,.2f} USD   •   {wins} winners / {len(tr)-wins} losers",
         transform=ax2.transAxes, fontsize=10, fontweight="bold", color=RED if total < 0 else GREEN, va="top")
fig.suptitle("EVERY TRADE FROM THE BOT'S JOURNAL — 57 trades, 16 winners  (paper mode + one overnight live run)",
             fontsize=14, fontweight="bold", color=NAVY, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "live_trades.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 2. STRATEGIES
rows = [
    ("Buy & Hold BTC", 10880.95, SKY, True),
    ("Ichimoku 10/30/60/30", 9637.03, BLUE, False),
    ("Long-Only Momentum 20b", 9345.11, BLUE, False),
    ("Aroon/RSI Divergence", 9052.62, BLUE, False),
    ("Hurst + RSI", 5840.83, BLUE, False),
    ("Calendar anomalies", 1475.42, BLUE, False),
]
df = pd.DataFrame(rows, columns=["label", "equity", "color", "bench"])
y = range(len(df))
fig, ax = plt.subplots(figsize=(10, 4.8))
for i, (lab, eq, col, bench) in enumerate(df.itertuples(index=False, name=None)):
    c = GREEN if eq >= 10000 else col
    ax.barh(i, eq, color=c, height=0.6, alpha=0.9)
    ax.text(eq + 90, i, f"{eq:,.0f}", va="center", fontsize=10, fontweight="bold", color=NAVY)
ax.axvline(10000, color=GREY, lw=1.2, ls="--")
ax.text(10000, len(df) - 0.35, "  start: $10,000", color=GREY, fontsize=9, va="center")
ax.set_yticks(list(y))
ax.set_yticklabels(df["label"], fontsize=11)
ax.set_xlim(0, 12000)
ax.set_xlabel("Final equity after 180 days (USD)")
ax.set_title("BACKTESTS ON BTC/USDT 1h — 180 DAYS, ALL COSTS INCLUDED\n"
             "market rose +8.95% in the same period",
             fontsize=13, fontweight="bold", color=NAVY)
ax.spines[["top", "right"]].set_visible(False)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "strategies.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 3. ETH/USDT pairs spread
c = sqlite3.connect("data/market_data.sqlite")
raw = pd.read_sql_query("select symbol, ts, close from ohlcv where timeframe='1h'", c)
raw["ts"] = pd.to_datetime(raw["ts"], unit="ms", utc=True)
btc = raw[raw.symbol == "BTC/USDT"].set_index("ts")["close"].sort_index()
eth = raw[raw.symbol == "ETH/USDT"].set_index("ts")["close"].sort_index()
px = pd.concat([btc, eth], axis=1, keys=["BTC", "ETH"]).dropna()
px = px.loc["2026-05-01":"2026-09-12"]

log = (px["BTC"].apply(__import__("numpy").log) - px["ETH"].apply(__import__("numpy").log))
z = (log - log.rolling(60).mean()) / log.rolling(60).std()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                               gridspec_kw={"height_ratios": [1.4, 1]})
ax1.plot(px.index, px["BTC"], color=NAVY, lw=1.1, label="BTC/USDT")
ax1.plot(px.index, px["ETH"], color=ORANGE, lw=1.1, label="ETH/USDT")
ax1.set_ylabel("Price (USD)")
ax1.set_title("PAIRS TRADING — BTC & ETH moved together (cointegrated)", fontsize=13,
              fontweight="bold", color=NAVY, pad=10)
ax1.legend(frameon=False, fontsize=9, ncol=2)
ax2.plot(z.index, z, color=BLUE, lw=1.1)
ax2.axhline(2, color=RED, ls="--", lw=1)
ax2.axhline(-2, color=GREEN, ls="--", lw=1)
ax2.axhline(0, color="#CCD5E0", lw=1)
ax2.set_ylabel("z-score of the spread")
ax2.set_title("Trade signal: enter when the two coins drift apart, exit when they return",
              fontsize=11, color=GREY)
ax2.text(px.index[int(len(px) * 0.98)], 2.05, "+2 = sell spread", color=RED, fontsize=9, ha="right")
ax2.text(px.index[int(len(px) * 0.98)], -2.9, "−2 = buy spread", color=GREEN, fontsize=9, ha="right")
for axx in (ax1, ax2):
    axx.spines[["top", "right"]].set_visible(False)
    axx.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    axx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
fig.tight_layout()
fig.savefig(OUT / "pairs.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 4. PORTFOLIO vs BTC
pe = pd.read_csv("reports/BTCUSDT_1h_exchange/portfolio_equity.csv", parse_dates=["timestamp"])
btc2 = px  # same window approx; recompute exactly over portfolio window below
raw = pd.read_sql_query("select symbol, ts, close from ohlcv where timeframe='1h'", c)
raw["ts"] = pd.to_datetime(raw["ts"], unit="ms", utc=True)
bt = raw[raw.symbol == "BTC/USDT"].set_index("ts")["close"].sort_index()
a, b = pe.timestamp.iloc[0], pe.timestamp.iloc[-1]
bt_win = bt[(bt.index >= a) & (bt.index <= b)]
bench = 10000.0 * bt_win / bt_win.iloc[0]

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(pe["timestamp"], pe["equity"], color=NAVY, lw=2.2, label="K-Means portfolio (top-2)")
ax.plot(bench.index, bench, color=ORANGE, lw=1.6, ls="--", label="Buy & hold BTC/USDT")
ax.fill_between(pe["timestamp"], pe["equity"], 0, alpha=0.06, color=NAVY)
ax.axhline(10000, color=GREY, lw=1, ls=":")
ax.text(pe["timestamp"].iloc[0], 10080, "start: $10,000", fontsize=9, color=GREY)
ax.legend(frameon=False, fontsize=11, loc="upper right")
ax.set_title("K-MEANS PORTFOLIO vs BUY & HOLD  (BTC/ETH/SOL/ADA/XRP, 120 days)\n"
             "clustered coins + equal-weight top-2 + rebalance every 10 hours",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_ylabel("Equity (USD)")
for axx in (ax,):
    axx.spines[["top", "right"]].set_visible(False)
    axx.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    axx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
ax.text(pe["timestamp"].iloc[-1], pe["equity"].iloc[-1] - 250, f"portfolio: {pe['equity'].iloc[-1]:,.0f}",
        fontsize=10, fontweight="bold", color=NAVY, ha="right")
ax.text(bench.index[-1], bench.iloc[-1] + 250, f"BTC hold: {bench.iloc[-1]:,.0f}",
        fontsize=10, fontweight="bold", color=ORANGE, ha="right")
fig.tight_layout()
fig.savefig(OUT / "portfolio.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("charts ->", OUT)
for p in sorted(OUT.glob("*.png")):
    print("  ", p.name, f"{p.stat().st_size//1024}KB")