import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

SRC = Path("RESEARCH_PAPER.md")
PDF = Path("RESEARCH_PAPER.pdf")

html_body = markdown.markdown(
    SRC.read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code"],
)

css = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #111; }
h1 { font-size: 15pt; color: #1a3a6b; border-bottom: 2px solid #1a3a6b; padding-bottom: 4px; }
h2 { font-size: 12.5pt; color: #1a3a6b; margin-top: 18px; border-bottom: 1px solid #bbb; padding-bottom: 2px; }
h3 { font-size: 11pt; color: #333; margin-top: 12px; }
p { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.5pt; }
th { background: #1a3a6b; color: white; padding: 4px 6px; text-align: left; }
td { border: 1px solid #ccc; padding: 3px 6px; }
tr:nth-child(even) td { background: #f2f5fa; }
code { font-family: Consolas, monospace; font-size: 8.5pt; background: #f0f0f0; padding: 1px 3px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 8px; font-size: 8.5pt; overflow-wrap: break-word; }
blockquote { color: #555; border-left: 3px solid #999; margin-left: 0; padding-left: 10px; }
hr { border: 0; border-top: 1px solid #ccc; margin: 10px 0; }
a { color: #1a3a6b; }
strong em, em strong { font-style: italic; }
"""

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head>
<body>{html_body}</body></html>"""

with open(PDF, "w+b") as f:
    status = pisa.CreatePDF(html, dest=f)
    ok = not status.err
    print(f"{'OK' if ok else 'FAILED'} -> {PDF} ({PDF.stat().st_size:,} bytes)")