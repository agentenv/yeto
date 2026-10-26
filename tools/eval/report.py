"""Markdown and HTML reports over a set of run verdicts (one table, then one
section per run with every criterion, its evidence, and the metrics)."""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

from .cells import BACKENDS, CAPABILITIES
from .verdict import load

MARK = {"pass": "PASS", "fail": "FAIL", "expected_fail": "EXPECTED FAIL (unsupported)",
        "unexpected_pass": "UNEXPECTED PASS", "error": "ERROR"}


def _latest(dirs: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """The last verdict per cell (dirs in the given order)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for d in dirs:
        doc = load(d)
        out[(doc["capability"], doc["backend"])] = doc
    return out


def _short(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 3] + "..."


def markdown(dirs: list[Path], *, title: str) -> str:
    cells = _latest(dirs)
    lines = [f"# {title}", "", "| Capability | " + " | ".join(BACKENDS) + " |", "|---|" + "---|" * len(BACKENDS)]
    for cap, text in CAPABILITIES.items():
        row = [f"{cap} {text}"]
        for backend in BACKENDS:
            doc = cells.get((cap, backend))
            if doc is None:
                row.append("not run")
                continue
            r = doc.get("run") or {}
            row.append(f"{MARK[doc['status']]}<br>`{doc['run_id']}` {r.get('card_seconds', '?')} card-s, "
                       f"~${r.get('usd_estimate', '?')}, wall {r.get('wall_s', '?')} s")
        lines.append("| " + " | ".join(row) + " |")
    total = sum(float((d.get("run") or {}).get("usd_estimate") or 0) for d in cells.values())
    lines += ["", f"Estimated GPU cost of the cells above: ${total:.2f} (list-price estimate, not a bill).", ""]
    for (cap, backend), doc in sorted(cells.items()):
        lines += [f"## {cap} {backend}: {MARK[doc['status']]}", "",
                  f"run `{doc['run_id']}`, code `{(doc.get('run') or {}).get('code_sha', '')[:10]}`"
                  + (f", error: {doc['error']}" if doc.get("error") else ""), "",
                  "| Criterion | Result | Evidence |", "|---|---|---|"]
        for c in doc["criteria"]:
            lines.append(f"| {c['id']}: {c['text']} | {'pass' if c['passed'] else '**FAIL**'} | "
                         f"`{_short(c['evidence'])}` |")
        if doc.get("metrics"):
            lines += ["", "Metrics: `" + _short(doc["metrics"], 900) + "`"]
        lines.append("")
    return "\n".join(lines) + "\n"


def html(dirs: list[Path], *, title: str) -> str:
    cells = _latest(dirs)
    esc = _html.escape
    color = {"pass": "#1a7f37", "fail": "#cf222e", "expected_fail": "#9a6700", "unexpected_pass": "#0969da",
             "error": "#cf222e"}
    head = "".join(f"<th>{esc(b)}</th>" for b in BACKENDS)
    rows = []
    for cap, text in CAPABILITIES.items():
        tds = []
        for backend in BACKENDS:
            doc = cells.get((cap, backend))
            if doc is None:
                tds.append("<td>not run</td>")
                continue
            r = doc.get("run") or {}
            tds.append(f"<td style='color:{color[doc['status']]}'><b>{esc(MARK[doc['status']])}</b><br>"
                       f"<code>{esc(doc['run_id'])}</code><br>{r.get('card_seconds', '?')} card-s, "
                       f"~${r.get('usd_estimate', '?')}</td>")
        rows.append(f"<tr><td>{esc(cap)} {esc(text)}</td>{''.join(tds)}</tr>")
    sections = []
    for (cap, backend), doc in sorted(cells.items()):
        crit = "".join(f"<tr><td>{esc(c['id'])}: {esc(c['text'])}</td><td>{'pass' if c['passed'] else '<b>FAIL</b>'}"
                       f"</td><td><code>{esc(_short(c['evidence'], 400))}</code></td></tr>" for c in doc["criteria"])
        sections.append(f"<h2>{esc(cap)} {esc(backend)}: {esc(MARK[doc['status']])}</h2>"
                        f"<table><tr><th>Criterion</th><th>Result</th><th>Evidence</th></tr>{crit}</table>"
                        f"<p><code>{esc(_short(doc.get('metrics') or {}, 1500))}</code></p>")
    style = ("body{font-family:sans-serif;margin:16px;background:#fff;color:#1f2328}"
             "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #d0d7de;padding:4px 8px;"
             "vertical-align:top}code{font-size:12px;word-break:break-all}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title><style>{style}</style>"
            f"</head><body><h1>{esc(title)}</h1><table><tr><th>Capability</th>{head}</tr>{''.join(rows)}</table>"
            f"{''.join(sections)}</body></html>\n")
