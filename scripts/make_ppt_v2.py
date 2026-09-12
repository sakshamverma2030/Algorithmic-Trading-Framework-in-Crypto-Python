"""Generate a warm, human-friendly, graph-rich capstone deck.
Real charts come from scripts/make_charts.py (actual project data).
Local only — never committed."""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ----------------------------------------------------------------- palette
CREAM  = RGBColor(0xFA, 0xF7, 0xF2)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
INK    = RGBColor(0x26, 0x2E, 0x38)
NAVY   = RGBColor(0x16, 0x3A, 0x5C)
BLUE   = RGBColor(0x24, 0x6B, 0xB8)
SKY    = RGBColor(0x7F, 0xBF, 0xEB)
TEAL   = RGBColor(0x1E, 0x8E, 0x84)
ORANGE = RGBColor(0xE0, 0x7A, 0x2F)
GREEN  = RGBColor(0x2E, 0x9E, 0x53)
RED    = RGBColor(0xC5, 0x3B, 0x2E)
GREY   = RGBColor(0x6B, 0x74, 0x7F)
CARD   = RGBColor(0xEF, 0xEA, 0xE1)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
FONT = "Segoe UI"


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=0.08):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1.2)
    sp.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sp.adjustments[0] = radius
        except Exception:
            pass
    return sp


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: list of (t, sz, bold, color, italic=False)."""
    b = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, r in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(3)
        txt, sz, bold, col = r[0], r[1], r[2], r[3]
        it = r[4] if len(r) > 4 else False
        run = p.add_run(); run.text = txt
        run.font.size = Pt(sz); run.font.bold = bold; run.font.color.rgb = col
        run.font.name = FONT; run.font.italic = it
    return b


def header(slide, kicker, title, num):
    rect(slide, 0, 0, 13.333, 1.15, NAVY)
    rect(slide, 0, 1.15, 13.333, 0.07, ORANGE)
    text(slide, 0.6, 0.12, 11.5, 0.35, [(kicker.upper(), 12, True, SKY)])
    text(slide, 0.6, 0.44, 11.5, 0.65, [(title, 25, True, WHITE)])
    chip = rect(slide, 12.05, 0.3, 0.95, 0.6, ORANGE, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = chip.text_frame; tf.word_wrap = False
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = num; r.font.size = Pt(18); r.font.bold = True
    r.font.color.rgb = WHITE; r.font.name = FONT


def card(slide, x, y, w, h, title=None, title_color=INK, accent=None):
    c = rect(slide, x, y, w, h, WHITE, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    if accent is not None:
        rect(slide, x, y, w, 0.09, accent)
    if title:
        text(slide, x + 0.22, y + 0.22, w - 0.44, 0.5, [(title, 17, True, title_color)])
    return c


def bullets(slide, x, y, w, h, items, size=15, gap=8, color=GREY):
    b = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = b.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        if item.startswith("#"):
            r = p.add_run(); r.text = item[1:]
            r.font.size = Pt(size + 3); r.font.bold = True
            r.font.color.rgb = NAVY; r.font.name = FONT
        else:
            r = p.add_run(); r.text = "▸  " + item
            r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT


def banner(slide, y, h, msg, fill=NAVY, fg=WHITE, size=15):
    b = rect(slide, 0, y, 13.333, h, fill, shape=MSO_SHAPE.RECTANGLE)
    rect(slide, 0, y, 0.10, h, ORANGE)
    tf = b.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = msg
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = fg; r.font.name = FONT
    return b


def pill(slide, x, y, w, h, label, fill=BLUE, fg=WHITE, size=12):
    c = rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    tf = c.text_frame; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = fg; r.font.name = FONT
    return c


def picture(slide, path, x, y, w=None, h=None, border=RGBColor(0xE0, 0xD8, 0xCA)):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    p = slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)
    p.line.color.rgb = border; p.line.width = Pt(1)
    return p


ICONS = ["📊", "🔍", "🧠", "🤖", "🛡️", "📈"]


def bg(slide):
    rect(slide, 0, 0, 13.333, 7.5, CREAM)


# ============================================================== 1. COVER
s = prs.slides.add_slide(BLANK)
bg(s)
rect(s, 0, 0, 13.333, 7.5, NAVY)
rect(s, 0, 4.95, 13.333, 0.09, ORANGE)
text(s, 0.9, 0.85, 11.5, 0.5, [("B.TECH FINAL YEAR CAPSTONE  •  SEPTEMBER 2026  •  SAKSHAM", 14, True, SKY)])
text(s, 0.9, 1.55, 11.6, 2.1, [
    ("Algorithmic Trading Framework", 48, True, WHITE),
    ("for Cryptocurrency Markets", 48, True, ORANGE),
])
text(s, 0.9, 3.65, 11.5, 1.1, [
    ("A khaamdiver? No — a careful builder.", 17, False, RGBColor(0xDF, 0xE8, 0xF2)),
    ("Every strategy is tested on old data with real costs BEFORE it ever sees live money.", 14, False, RGBColor(0xB9, 0xC8, 0xD8)),
])
tools = [("Ichimoku", "#246BB8"), ("Pairs Trading", "#1E8E84"), ("K-Means ML", "#E07A2F"),
         ("Hurst", "#7FBFEB"), ("Momentum", "#2E9E53"), ("Live Engine", "#C53B2E")]
for i, (t, col) in enumerate(tools):
    w = 1.85; x = 0.9 + i * (w + 0.12)
    pill(s, x, 5.25, w, 0.55, t, fill=RGBColor.from_string(col.lstrip("#")))
text(s, 0.9, 6.55, 11.5, 0.5, [
    ("Data  →  Signals  →  Backtest  →  Risk  →  Live Paper  —  one codebase, no shortcuts", 13, False, RGBColor(0x9F, 0xB2, 0xC6))])

# ============================================================== 2. IN PLAIN WORDS
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Overview", "What is this, in plain words?", "2")
cards = [
    ("❓ WHAT", "A Python system that BUILDS and TESTS crypto strategies, then runs them automatically on live prices — with fake money first.", BLUE),
    ("✅ WHY IT WORKS", "Backtests include real-world costs (fees, spread, slippage) and never peek into the future. So the results can be trusted.", TEAL),
    ("🤔 THE HONEST TRUTH", "Trading a single coin on short signals lost money in our 180-day test. The K-Means portfolio did better than its weakest coins. We show both.", ORANGE),
]
for i, (t, d, col) in enumerate(cards):
    x = 0.6 + i * 4.2
    card(s, x, 1.6, 4.0, 5.6, accent=col)
    text(s, x + 0.25, 2.0, 3.5, 0.7, [(t, 17, True, col)])
    text(s, x + 0.25, 2.75, 3.5, 4.2, [(d, 14, False, INK)])

# ============================================================== 3. PIPELINE
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Architecture", "The 6-step pipeline", "3")
steps = [
    ("1 • DATA", "Live Binance prices + offline fake data. No account needed.", BLUE),
    ("2 • SIGNALS", "Indicators + strategy rules decide: buy, sell or stay out.", TEAL),
    ("3 • BACKTEST", "Two engines: fast (research) and realistic (final answer).", ORANGE),
    ("4 • RISK", "Position size, stop-loss, safety breaker — on every trade.", RED),
    ("5 • REPORT", "Equity curve, Sharpe, drawdown, charts + CSV evidence.", NAVY),
    ("6 • LIVE", "Same code runs on paper / testnet / real account.", GREEN),
]
for i, (t, d, col) in enumerate(steps):
    col_id, row_id = i % 3, i // 3
    x = 0.6 + col_id * 4.2
    y = 1.8 + row_id * 2.85
    card(s, x, y, 4.0, 2.4, accent=col)
    text(s, x + 0.22, y + 0.35, 3.6, 0.5, [(t, 16, True, col)],)
    text(s, x + 0.22, y + 1.05, 3.6, 1.2, [(d, 13, False, GREY)])
banner(s, 6.95, 0.5, "The same strategy code runs in backtest AND live — so tested results match real behaviour.")

# ============================================================== 4. DATA
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Step 1", "Where the data comes from", "4")
card(s, 0.6, 1.6, 6.1, 5.3, title="LIVE EXCHANGE (Binance)", accent=BLUE)
bullets(s, 0.85, 2.4, 5.6, 4.2, [
    "Downloads candle prices (open/high/low/close/volume)",
    "Retries + gap detection, saves in a local database",
    "Any timeframe: 1 minute → 1 day; any coin — BTC, ETH, SOL, ADA, XRP…",
])
card(s, 7.0, 1.6, 5.7, 5.3, title="WHAT WE USED", accent=ORANGE)
bullets(s, 7.25, 2.4, 5.2, 4.2, [
    "BTC/USDT 1h — 4,319 candles over 180 days (Mar → Sep 2026)",
    "That window ended +8.95% for simply holding BTC",
    "Plus 4 more coins for the K-Means portfolio test (120 days)",
])
banner(s, 6.95, 0.5, "Fake data generator (offline) powers 53 unit tests — so we can test the tests, for free.", GREY, WHITE, 13)

# ============================================================== 5. SIGNALS TOOLKIT
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Step 2", "The indicator toolkit", "5")
card(s, 0.6, 1.6, 6.1, 5.3, title="TECHNICAL FRIENDS", accent=TEAL)
bullets(s, 0.85, 2.4, 5.6, 4.2, [
    "Ichimoku Cloud — 5 lines showing trend strength",
    "RSI + divergence — early 'danger' flag",
    "Aroon, Bollinger Bands, ATR — for stops and sizing",
])
card(s, 7.0, 1.6, 5.7, 5.3, title="SMART / ML FRIENDS", accent=ORANGE)
bullets(s, 7.25, 2.4, 5.2, 4.2, [
    "Hurst exponent — trending or 'snap-back' market?",
    "ADF test — do two coins move together? (pairs trading)",
    "K-Means — groups similar coins (machine learning)",
])
banner(s, 6.95, 0.5, "Fairness rule: indicators only use PAST prices — the future is never used. (Verified by tests.)")

# ============================================================== 6. STRATEGIES
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Step 2b", "Strategies across the  — the full set", "6")
card(s, 0.6, 1.6, 6.1, 5.3, title="INTERMEDIATE", accent=BLUE)
bullets(s, 0.85, 2.4, 5.6, 4.2, [
    "Ichimoku — buy/sell on trend-line crosses",
    "Calendar effect — some hours/days rise more often",
    "Aroon + RSI divergence — trend & momentum in sync",
    "Momentum — buy coins that recently went up",
])
card(s, 7.0, 1.6, 5.7, 5.3, title="ADVANCED (ML + stat-arb)", accent=ORANGE)
bullets(s, 7.25, 2.4, 5.2, 4.2, [
    "Pairs — buy cheap coin, sell expensive twin, wait",
    "Hurst + RSI — switch style automatically",
    "K-Means clustering — group, then trade the best group",
    "Top-N momentum portfolio — hold winners equally",
])
banner(s, 6.95, 0.5, "All strategies share one signal format — easy to test, easy to go live.")

# ============================================================== 7. BACKTEST ENGINES
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Step 3", "Two backtest engines", "7")
card(s, 0.6, 1.6, 6.1, 5.3, title="FAST ENGINE", accent=SKY)
bullets(s, 0.85, 2.4, 5.6, 4.2, [
    "Milliseconds per run — great for 'what if' tests",
    "Simpler cost model, no stop-losses",
    "Use it to scan many parameter settings",
])
card(s, 7.0, 1.6, 5.7, 5.3, title="REALISTIC ENGINE", accent=ORANGE)
bullets(s, 7.25, 2.4, 5.2, 4.2, [
    "Final, trustworthy answers",
    "Fills only on the NEXT candle — no peeking",
    "Stops + risk sizing + ALL costs + circuit breaker",
    "Behaves exactly like the live bot",
])
banner(s, 6.95, 0.5, "Fair fills: if price gaps past a stop we fill at the open; when stop & target hit together, the stop wins.")

# ============================================================== 8. RISK
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Step 4", "Staying alive: risk management", "8")
card(s, 0.6, 1.6, 6.1, 5.3, title="PER TRADE", accent=GREEN)
bullets(s, 0.85, 2.4, 5.6, 4.2, [
    "Risk only 1% of the account on any single trade",
    "Stop-loss placed by volatility (ATR) or cloud edge",
    "Trailing stop — let winners run, protect profit",
    "Kelly sizing — size bets from past results, capped",
])
card(s, 7.0, 1.6, 5.7, 5.3, title="WHOLE ACCOUNT", accent=RED)
bullets(s, 7.25, 2.4, 5.2, 4.2, [
    "Circuit breaker — halt if down 20% from best, or 5% in a day",
    "Stops checked constantly, not just at candle close",
    "Same safety rules in backtest and live — no surprises",
])
banner(s, 6.95, 0.5, "Safety rules run in tests AND live trading — what you simulate is what you trade.")

# ============================================================== 9. LIVE ENGINE
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Step 5 → 6", "The live / paper trading engine", "9")
modes = [
    ("PAPER", "Real Prices, Fake Money", "Orders fill against the real order book. No account or keys needed.", GREEN),
    ("DEMO", "Binance Test Server", "Binance's official test playground. Test-only money.", BLUE),
    ("REAL", "Your Real Account", "Real money! Requires typing 'I UNDERSTAND' + a small sub-account.", RED),
]
for i, (t, d, desc, col) in enumerate(modes):
    x = 0.6 + i * 4.2
    card(s, x, 1.6, 4.0, 3.6, title=t, title_color=col, accent=col)
    text(s, x + 0.22, 2.35, 3.6, 0.6, [(d, 14, True, INK)])
    text(s, x + 0.22, 3.05, 3.6, 1.8, [(desc, 13, False, GREY)])
card(s, 0.6, 5.45, 12.1, 1.6, title="WORKS 24/7", accent=NAVY)
bullets(s, 0.85, 6.0, 11.6, 0.9, [
    "Streaming engine never misses a price update — live websocket, REST, or recorded replay",
    "Signals only after a candle CLOSES (no repainting); every fill & trade is saved to a journal",
], size=13, gap=5)

# ============================================================== 10. REAL: LIVE SESSION
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Evidence 1", "Every trade the bot ever made — 57 trades, all shown", "10")
text(s, 0.6, 1.3, 12.0, 0.5, [
    ("57 trades • 116 fills • 16 winners • 10 test sessions (paper + replay + live overnight)", 14, True, ORANGE)])
picture(s, "charts_ppt/live_trades.png", 0.6, 1.85, w=12.1)
banner(s, 6.85, 0.55, "One session ran LIVE overnight — Sep 11 13:18 → Sep 12 02:37 UTC on BTC/USDT 1m: 5 real signals, 4 stop-losses + 1 take-profit (−$162). Live behaviour matched the backtest. Whole journal sum: −$869.", NAVY, WHITE, 13)

# ============================================================== 11. REAL: STRATEGY RESULTS
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Evidence 2", "Backtests on 180 real days of BTC (all costs included)", "11")
picture(s, "charts_ppt/strategies.png", 0.6, 1.5, w=12.1)
banner(s, 6.85, 0.55, "Honest negative result: all cost-inclusive timing strategies fell short of just holding BTC. The framework proved judgement — costs matter.", RED, WHITE, 13)

# ============================================================== 12. REAL: PAIRS
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Evidence 3", "Pairs trading — the statistical arbitrage module", "12")
picture(s, "charts_ppt/pairs.png", 0.6, 1.5, w=12.1)
banner(s, 6.85, 0.55, "When two linked coins drift too far apart (red band), sell the spread; when they return, close. Mean reversion — the market's rubber band.", NAVY, WHITE, 13)

# ============================================================== 13. REAL: PORTFOLIO
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Evidence 4", "K-Means portfolio vs buy-and-hold (120 real days)", "13")
picture(s, "charts_ppt/portfolio.png", 0.6, 1.5, w=12.1)
banner(s, 6.85, 0.55, "The clustered portfolio beat its two weakest members — but still fell short of simply holding BTC. Diversification softened the crash without beating it.", ORANGE, WHITE, 13)

# ============================================================== 14. VALIDATION
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Trust", "53 tests — the no-cutting-corners guarantee", "14")
stats = [("53", "tests pass"), ("22", "core-engine tests"), ("31", "strategy tests"), ("0", "look-ahead allowed")]
for i, (num, lab) in enumerate(stats):
    x = 0.6 + i * 3.1
    card(s, x, 1.6, 2.9, 1.7, accent=GREEN if i < 3 else RED)
    text(s, x + 0.2, 1.85, 2.5, 0.7, [(num, 30, True, NAVY)])
    text(s, x + 0.2, 2.6, 2.5, 0.5, [(lab, 12, False, GREY)])
bullets(s, 0.8, 3.8, 11.8, 3.2, [
    "No cheating: a short history gives the SAME signals as the full history",
    "Backtest = live: replaying old data through the real bot reproduces the same trades",
    "Math checks out: account = cash + holdings; trade profits match balance changes",
    "Indicators matched against known textbook answers (Ichimoku, ATR, Bollinger)",
    "Bad settings raise clear errors instead of silently misbehaving",
], size=14, gap=7)

# ============================================================== 15. RESULTS TABLE (real)
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Evidence 5", "Real backtest scorecard (BTC/USDT 1h, 180 days, $10k start)", "15")
rows = [
    ("Ichimoku 10/30/60/30", "9,637", "−3.6%", "28", "smallest loss window"),
    ("Momentum (long-only 20b)", "9,345", "−6.6%", "58", "second-best score"),
    ("Aroon/RSI divergence", "9,053", "−9.5%", "63", "only 7.8% time in market"),
    ("Pairs BTC × ETH", "7,233", "−27.7%", "121", "longer 310-day window"),
    ("Hurst + RSI", "5,841", "−41.6%", "226", "traded too often → fees ate profits"),
    ("Calendar anomalies", "1,475", "−85.3%", "798", "fees + slippage alone ~$7,800"),
]
header_row = ["Strategy", "Final", "Return", "Trades", "In plain words"]
y_start = 1.55
rect(s, 0.6, y_start, 12.13, 0.55, NAVY)
for j, col in enumerate(header_row):
    text(s, 0.85 + j * 2.42, y_start + 0.1, 2.4, 0.4, [(col, 12.5, True, WHITE)])
for i, (name, fin, ret, tr, note) in enumerate(rows):
    y = y_start + 0.65 + i * 0.78
    fill = WHITE if i % 2 == 0 else CARD
    rect(s, 0.6, y, 12.13, 0.7, fill)
    text(s, 0.85, y + 0.13, 2.4, 0.5, [(name, 12, True, INK)])
    text(s, 3.27, y + 0.15, 1.3, 0.5, [(fin, 12.5, True, NAVY)])
    text(s, 5.69, y + 0.15, 1.3, 0.5, [(ret, 12.5, True, RED)])
    text(s, 8.11, y + 0.15, 1.3, 0.5, [(tr, 12, False, GREY)])
    text(s, 10.53, y + 0.08, 2.2, 0.6, [(note, 10.5, False, GREY)])
banner(s, 6.9, 0.52, "Benchmark buy-and-hold would have turned $10k into $10,881. Honest spreadsheets beat rosy assertions.", NAVY, WHITE, 13)

# ============================================================== 16. DOCS & LIMITS
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Wrap-up", "Docs, honest limits & what's next", "16")
card(s, 0.6, 1.6, 6.1, 3.2, title="DOCUMENTATION", accent=BLUE)
bullets(s, 0.85, 2.3, 5.6, 2.3, [
    "README.md — install, run, trade modes",
    "CAPSTONE.md / CAPSTONE_ABSTRACT.md — full report",
    "RESEARCH_PAPER.md + .pdf — academic style",
    "reports/ — saved evidence + charts; data/ — journal CSV",
], size=13, gap=5)
card(s, 0.6, 5.0, 6.1, 2.2, title="WHAT WE DID NOT PROVE", accent=RED)
bullets(s, 0.85, 5.6, 5.6, 1.5, [
    "Every single-coin strategy lost after costs (180 days)",
    "One market, one period — not a general verdict",
], size=13, gap=5)
card(s, 7.0, 1.6, 5.7, 5.6, title="NEXT STEPS", accent=ORANGE)
bullets(s, 7.25, 2.3, 5.2, 4.7, [
    "AI return prediction (XGBoost / LSTM) for coin ranking",
    "Smarter portfolio balancing (risk parity) not just equal weight",
    "Test more coins & timeframes; walk-forward validation",
    "End-to-end live (testnet) orders",
], size=14, gap=8)

# ============================================================== 17. QUICK START
s = prs.slides.add_slide(BLANK)
bg(s)
header(s, "Try it", "Reproduce everything", "17")
box = rect(s, 0.7, 1.6, 11.9, 4.7, RGBColor(0x14, 0x1B, 0x24), shape=MSO_SHAPE.ROUNDED_RECTANGLE)
code = [
    ("pip install -r requirements.txt", False),
    ("python main.py fetch --symbol BTC/USDT --timeframe 1h --days 180", False),
    ("python main.py backtest --preset crypto          # single strategy", False),
    ("python main.py compare                           # presets vs each other", False),
    ("python scripts/compare_all.py --symbol BTC/USDT  # all-strategy scorecard", False),
    ("python main.py portfolio --source exchange       # K-Means + momentum", False),
    ("python main.py live --feed ccxtpro --timeframe 1m      # LIVE paper", True),
    ("python main.py live --feed replay --replay-bars 2500    # offline demo", False),
    ("python -m pytest                                 # 53 tests", True),
]
tf = box.text_frame; tf.word_wrap = True
for i, (line, hot) in enumerate(code):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(5)
    r = p.add_run(); r.text = line
    r.font.name = "Consolas"; r.font.size = Pt(15)
    r.font.color.rgb = ORANGE if hot else RGBColor(0xE4, 0xEE, 0xF6)
banner(s, 6.55, 0.55, "Every command above was actually run on this machine — the results are saved in reports/ and the journal.", ORANGE, WHITE, 13)

OUT = Path("PROJECT_PRESENTATION.pptx")
prs.save(OUT)
print(f"OK -> {OUT}  ({len(prs.slides._sldIdLst)} slides, {OUT.stat().st_size:,} bytes)")