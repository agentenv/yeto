"""Static single-file export (design D3).

The page is ``static/index.html`` with the full reducer view inlined as
JSON; it makes no network request when opened (no fonts, no CDN, the Ray
iframe is a placeholder). ``://`` inside the data is JSON-escaped so the
file carries no literal http(s) URL.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

STATIC = Path(__file__).with_name("static")
PAGE = STATIC / "index.html"
APP = STATIC / "app.js"
MARKER = "<!--YETO-DATA-->"
APP_MARKER = "<!--YETO-APP-->"


def inline_json(data: dict) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":"))
    # JSON escapes that keep the parsed value identical but stop the HTML
    # parser (</script>, <!--) and keep URL schemes out of the raw file.
    return text.replace("</", "<\\/").replace("<!--", "<\\u0021--").replace("://", ":\\/\\/")


def render_page(data: dict | None) -> str:
    html = PAGE.read_text(encoding="utf-8").replace(
        APP_MARKER, "<script>\n" + APP.read_text(encoding="utf-8") + "\n</script>")
    if data is None:
        return html.replace(MARKER, "")
    block = f'<script type="application/json" id="yeto-data">{inline_json(data)}</script>'
    return html.replace(MARKER, block)


def export_view(reducer, *, generated_at: float | None = None) -> dict:
    view = reducer.full_view(live=False)
    view["generated_at"] = generated_at if generated_at is not None else time.time()
    return view


def export_html(reducer, out: str | Path, *, generated_at: float | None = None) -> Path:
    out = Path(out)
    out.write_text(render_page(export_view(reducer, generated_at=generated_at)), encoding="utf-8")
    return out
