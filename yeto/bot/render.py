"""Channel-neutral messages built from the dashboard's ``/api/view`` JSON.

A :class:`Message` is a title, body lines, optional buttons (a command, or a
URL when it starts with ``http``), optional charts and a flag to @ the owners.
Bold spans are wrapped by :func:`b` in two control characters; each channel
turns them into its own markup (Telegram ``*x*``, Lark ``**x**``, plain text
drops them) and escapes everything else itself.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from yeto.dashboard.sft import MERGE_COLS, STEP_COLS

from .charts import Chart

BOLD_OPEN, BOLD_CLOSE = "\x02", "\x03"

# sft.islands[].rows and sft.syncer.merges are positional; their column names come
# with the view (sft.cols, sft.syncer.merge_cols) and are resolved per view with
# step_cols() / merge_cols(). A column the view does not name resolves to None and
# reads as missing data. syncer.outer rows carry no names:
# [ts, step, fragment, step_seconds, timed_out, arrived].
PROBLEMS = ("rejected", "timeout", "not_carried")
STATUS_DOT = {"ok": "\U0001f7e2", "warn": "\U0001f7e1", "stale": "\U0001f534", "done": "\U0001f3c1",
              "unknown": "⚪"}
LEVEL_ICON = {"info": "ℹ️", "ok": "✅", "warn": "⚠️", "p0": "\U0001f6a8", "p1": "\U0001f7e0",
              "p2": "⚪", "report": "\U0001f4ca", "decision": "\U0001f9ed"}


@dataclass
class Message:
    title: str
    lines: list[str] = field(default_factory=list)
    level: str = "info"  # info | ok | warn | p0 | p1 | p2 | report | decision
    buttons: list[tuple[str, str]] = field(default_factory=list)  # (label, command or URL)
    charts: list[Chart] = field(default_factory=list)
    mention: bool = False  # @ the configured owners

    def plain(self) -> str:
        return strip_bold("\n".join([self.title, *self.lines]))


def b(text: Any) -> str:
    return f"{BOLD_OPEN}{text}{BOLD_CLOSE}"


def strip_bold(text: str) -> str:
    return text.replace(BOLD_OPEN, "").replace(BOLD_CLOSE, "")


def is_url(target: str) -> bool:
    return target.startswith(("http://", "https://"))


# -- number formatting ------------------------------------------------------------

def num(v: Any) -> float | None:
    """A finite float, or None (bools, strings and NaN/Inf are None)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def nonfinite(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("nan", "inf", "-inf", "infinity", "-infinity")
    return isinstance(v, float) and not math.isfinite(v)


def f3(v: Any) -> str:
    v = num(v)
    if v is None:
        return "-"
    return f"{v:.4f}" if abs(v) < 10 else f"{v:.3g}"


def fsig(v: Any, digits: int = 3) -> str:
    v = num(v)
    return "-" if v is None else f"{v:.{digits}g}"


def fsigned(v: Any) -> str:
    v = num(v)
    return "-" if v is None else f"{v:+.4f}"


def _g1(v: float) -> str:
    """One decimal, without a trailing ``.0``."""
    out = f"{v:.1f}"
    return out[:-2] if out.endswith(".0") else out


def fdur(s: Any) -> str:
    s = num(s)
    if s is None:
        return "-"
    if s < 120:
        return f"{s:.0f}s"
    if s < 7200:
        return f"{_g1(s / 60)}min"
    if s < 172800:
        return f"{_g1(s / 3600)}h"
    return f"{_g1(s / 86400)}d"


def fbytes(v: Any) -> str:
    v = num(v)
    if v is None:
        return "-"
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if v >= scale:
            return f"{v / scale:.2f} {unit}"
    return f"{v:.0f} B"


def fclock(ts: Any) -> str:
    ts = num(ts)
    return "-" if ts is None else time.strftime("%m-%d %H:%M", time.localtime(ts))


# -- view helpers ---------------------------------------------------------------

def col_index(names: Any, fallback: tuple[str, ...]) -> dict[str, int]:
    """Column name -> position; ``fallback`` when the view names no columns (an older
    dashboard, whose layout was the fallback)."""
    if not isinstance(names, (list, tuple)) or not names:
        names = fallback
    return {str(n): i for i, n in enumerate(names)}


def step_cols(sft: dict) -> dict[str, int]:
    """Positions in ``sft.islands[].rows`` by name (``sft.cols``)."""
    return col_index((sft or {}).get("cols"), STEP_COLS)


def merge_cols(sft: dict) -> dict[str, int]:
    """Positions in ``sft.syncer.merges`` rows by name (``sft.syncer.merge_cols``)."""
    return col_index(((sft or {}).get("syncer") or {}).get("merge_cols"), MERGE_COLS)


def cell(r: list, col: int | None) -> Any:
    """``r[col]``, or None when the column is unknown or the row is short."""
    return r[col] if col is not None and 0 <= col < len(r) else None


def last_value(rows: list[list], col: int | None) -> Any:
    """Newest non-None value of a column (the page does the same: the newest step
    often has bench numbers before ms-swift logs its loss)."""
    for r in reversed(rows):
        v = cell(r, col)
        if v is not None:
            return v
    return None


def column(rows: list[list], col: int | None) -> list[float]:
    return [v for v in (num(cell(r, col)) for r in rows) if v is not None]


def step_p50(rows: list[list], ix: dict[str, int]) -> float | None:
    vals = sorted(column(rows[-20:], ix.get("step_s")))
    return vals[max(0, math.ceil(0.5 * len(vals)) - 1)] if vals else None


def refs(view: dict) -> dict:
    return ((view or {}).get("sft") or {}).get("refs") or {}


def problem_counts(sft: dict) -> dict[str, int]:
    out = {c: 0 for c in PROBLEMS}
    for e in (sft.get("syncer") or {}).get("events") or []:
        if len(e) > 1 and e[1] in out:
            out[e[1]] += 1
    return out


def last_round_ts(sft: dict) -> float | None:
    syn = sft.get("syncer") or {}
    m_ts = merge_cols(sft).get("ts")
    ts = [num(r[0]) for r in (syn.get("outer") or []) if r] + [num(cell(r, m_ts)) for r in (syn.get("merges") or [])]
    ts = [t for t in ts if t is not None]
    return max(ts) if ts else None


def round_intervals(sft: dict, n: int = 16) -> list[float]:
    """Seconds between consecutive outer steps (newest ``n``)."""
    syn = sft.get("syncer") or {}
    pts = sorted({(num(r[0]), r[1]) for r in (syn.get("outer") or []) if r and num(r[0]) is not None
                  and isinstance(r[1], int)})
    if len(pts) < 2:
        mx = merge_cols(sft)
        m_ts, m_step = mx.get("ts"), mx.get("step")
        pts = sorted({(num(cell(r, m_ts)), cell(r, m_step)) for r in (syn.get("merges") or [])
                      if num(cell(r, m_ts)) is not None and isinstance(cell(r, m_step), int)})
    gaps = [q[0] - p[0] for p, q in zip(pts, pts[1:], strict=False) if q[1] > p[1] and q[0] > p[0]]
    return gaps[-n:]


def find_island(sft: dict, key: str) -> dict | None:
    k = key.strip().lower()
    if k.startswith("island"):
        k = k[len("island"):].strip(" -_")
    for isl in sft.get("islands") or []:
        if str(isl.get("id")).lower() == k or str(isl.get("name") or "").lower() in (key.strip().lower(), k):
            return isl
    return None


def latest_rss(sft: dict) -> float | None:
    rss = (sft.get("syncer") or {}).get("rss") or []
    return num(rss[-1][1]) if rss and len(rss[-1]) > 1 else None


def island_label(isl: dict) -> str:
    return isl.get("name") or f"island {isl.get('id')}"


def window(rows: list[list], now: float | None, seconds: float, ts_col: int | None) -> list[list]:
    if now is None:
        return rows
    return [r for r in rows if num(cell(r, ts_col)) is not None and r[ts_col] >= now - seconds]


def loss_chart(sft: dict, seconds: float, title: str) -> Chart:
    now = num(sft.get("now"))
    ix = step_cols(sft)
    ts, loss = ix.get("ts"), ix.get("loss")
    series = []
    for isl in sft.get("islands") or []:
        rows = window(isl.get("rows") or [], now, seconds, ts)
        series.append((island_label(isl), [cell(r, ts) for r in rows], [cell(r, loss) for r in rows]))
    return Chart(title, series, time_axis=True)


def merge_health(view: dict, n: int = 20) -> dict:
    """Direct-in-round ratio (rounds merged with every expected push and no resend
    or timeout), missed counts per island, round timing medians."""
    sft = (view or {}).get("sft") or {}
    syn = sft.get("syncer") or {}
    rounds = ((view or {}).get("rounds_syncer") or [])[-n:]
    out: dict[str, Any] = {"rounds": len(rounds), "direct": None, "missed": {}, "weights": None}
    if rounds:
        out["direct"] = sum(1 for r in rounds if not r.get("missed") and not r.get("resend"))
        for r in rounds:
            for iid in r.get("missed") or []:
                out["missed"][str(iid)] = out["missed"].get(str(iid), 0) + 1
        w = [r.get("weights") for r in rounds if isinstance(r.get("weights"), dict)]
        out["weights"] = w[-1] if w else None  # hook: per-island merge weight once exposed
    else:
        outer = (syn.get("outer") or [])[-n:]
        out["rounds"] = len(outer)
        out["direct"] = sum(1 for r in outer if not (len(r) > 4 and r[4])) if outer else None
    merges = (syn.get("merges") or [])[-n:]
    mx = merge_cols(sft)
    for name in ("round_s", "quorum_s", "grace_s", "merge_s"):
        vals = column(merges, mx.get(name))
        out[name] = statistics.median(vals) if vals else None
    return out


def merge_line(view: dict) -> str:
    mh = merge_health(view)
    if not mh["rounds"]:
        return f"{b('Merge')} no rounds yet"
    parts = [f"{b('Merge')} direct-in-round {mh['direct']}/{mh['rounds']}"
             f" ({100 * mh['direct'] / mh['rounds']:.0f}%)"]
    if mh["round_s"] is not None:
        parts.append(f"median round {fdur(mh['round_s'])} (quorum {fdur(mh['quorum_s'])}, grace {fdur(mh['grace_s'])},"
                     f" merge {fdur(mh['merge_s'])})")
    if mh["missed"]:
        parts.append("missed " + ", ".join(f"{k} x{v}" for k, v in sorted(mh["missed"].items())))
    return " | ".join(parts)


# -- commands -------------------------------------------------------------------

NAV_BUTTONS = [("Status", "/status"), ("Merge", "/merge"), ("Eval", "/eval"), ("Alerts", "/alerts")]

HELP_LINES = [
    "/status - progress: outer step, ETA, islands, merge health, 6 h loss chart",
    "/island <name|id> - one island: last steps, loss / grad_norm / step-time charts",
    "/eval - eval table, dloss vs references, chart",
    "/merge - syncer rounds: direct-in-round ratio, missed islands, timing, problem events",
    "/res - resources: HBM per island, host memory, syncer RSS",
    "/ckpt - checkpoints: syncer writes and eval checkpoints",
    "/alerts - active alerts and the last 24 h",
    "/ack [cN|all] - mute the subjects of alert card N (or all) for 2 h",
    "/mute <30m|2h|1d> [all] - mute P1/P2 (all: P0 too); /unmute",
    "/help - this help; /start - register this chat for alerts",
]


def help_message(prefix: str | None = None) -> Message:
    return Message("yeto training bot", ([prefix, ""] if prefix else []) + HELP_LINES, "info", NAV_BUTTONS)


def island_row(isl: dict, ix: dict[str, int]) -> str:
    rows = isl.get("rows") or []
    st = isl.get("status") or "unknown"
    step = last_value(rows, ix.get("step"))
    return (f"{STATUS_DOT.get(st, '')} {b(island_label(isl))} {step if step is not None else '-'}/"
            f"{isl.get('steps_total') or '?'} | loss {f3(last_value(rows, ix.get('loss')))}"
            f" | {fdur(step_p50(rows, ix))}/step" + (f" | {st}" if st != "ok" else ""))


def syncer_line(sft: dict) -> str:
    syn = sft.get("syncer") or {}
    cur, total = syn.get("outer_step"), syn.get("total_steps")
    pct = f" ({100 * cur / total:.1f}%)" if isinstance(cur, int) and isinstance(total, int) and total else ""
    lrt, now = last_round_ts(sft), num(sft.get("now"))
    line = f"{b('Outer')} {cur if cur is not None else '-'}/{total or '?'}{pct}"
    if num(syn.get("eta_ts")):
        line += f" | ETA {fclock(syn['eta_ts'])}"
    if lrt is not None and now is not None:
        line += f" | last round {fdur(now - lrt)} ago"
    return line


def eval_refs(rows: list[dict]) -> list[str]:
    """Reference names of eval rows: every ``vs_<name>`` key is a comparison against
    reference ``<name>`` (``vs_base``, a fixed checkpoint, ...), in first-seen order;
    ``prev`` (the previous eval point) goes last."""
    names: list[str] = []
    for r in rows:
        for k in r if isinstance(r, dict) else ():
            if isinstance(k, str) and k.startswith("vs_") and len(k) > 3 and k[3:] not in names:
                names.append(k[3:])
    return [n for n in names if n != "prev"] + (["prev"] if "prev" in names else [])


def ref_cmp(row: dict, name: str) -> dict:
    c = row.get(f"vs_{name}")
    return c if isinstance(c, dict) else {}


def eval_line(row: dict | None) -> str:
    if not row:
        return f"{b('Eval')} none yet"
    parts = [f"{b('Eval')} step {row.get('step')} loss {f3(row.get('loss'))}"]
    for name in eval_refs([row]):
        d = num(ref_cmp(row, name).get("dloss"))
        if d is not None:
            parts.append(f"vs {name} {fsigned(d)}")
    return " | ".join(parts)


def status_message(view: dict, *, alerts_line: str | None = None, chart_hours: float = 6.0) -> Message:
    """The progress card (/status and the scheduled one)."""
    ov = (view or {}).get("overview") or {}
    sft = (view or {}).get("sft")
    title = f"yeto progress: {ov.get('run') or 'run'}"
    if not sft:
        return rl_status_message(view, title)
    lines = [syncer_line(sft)]
    ix = step_cols(sft)
    lines += [island_row(isl, ix) for isl in sft.get("islands") or []] or ["no SFT islands yet"]
    lines += [merge_line(view), eval_line(((sft.get("eval") or {}).get("rows") or [None])[-1])]
    if alerts_line:
        lines.append(alerts_line)
    chart = loss_chart(sft, chart_hours * 3600, f"loss, last {chart_hours:g} h")
    return Message(title, lines, "report", NAV_BUTTONS, [chart] if not chart.empty() else [])


def rl_status_message(view: dict, title: str) -> Message:
    ov = (view or {}).get("overview") or {}
    gs = ov.get("global_status") or {}
    lines = [f"{b('Fleet')} {gs.get('text') or '-'}"]
    for c in ov.get("islands") or []:
        lines.append(f"{STATUS_DOT.get(c.get('status'), '')} {b('island ' + str(c.get('id')))} {c.get('status')}"
                     f" | round {c.get('round', '-')} | last event {fdur(c.get('last_event_age_s'))} ago")
    alerts = ov.get("alerts") or []
    if alerts:
        lines.append(f"{len(alerts)} dashboard alert(s); first: {alerts[0].get('title')}")
    return Message(title, lines, "report", NAV_BUTTONS)


def steps_table(rows: list[list], ix: dict[str, int], n: int = 10) -> list[str]:
    out = [b("step | loss | grad_norm | step time | HBM GiB | tokens/s")]
    for r in rows[-n:]:
        c = {k: cell(r, ix.get(k)) for k in ("step", "loss", "grad_norm", "step_s", "hbm_gib", "tps")}
        out.append(f"{c['step'] if c['step'] is not None else '-'} | {f3(c['loss'])} | {fsig(c['grad_norm'])}"
                   f" | {fdur(c['step_s'])} | {fsig(c['hbm_gib'])} | {fsig(c['tps'], 4)}")
    return out


def unknown_island(sft: dict, key: str) -> Message:
    names = ", ".join(f"{i.get('id')} ({i.get('name')})" for i in sft.get("islands") or []) or "none"
    return Message("unknown island", [f"no island {key!r}; islands: {names}", "usage: /island <name|id>"], "warn")


def island_message(view: dict, key: str, *, hbm_limit: float | None = None, chart_steps: int = 200) -> Message:
    sft = (view or {}).get("sft") or {}
    isl = find_island(sft, key) if key else None
    if isl is None:
        return unknown_island(sft, key)
    rows = isl.get("rows") or []
    ix = step_cols(sft)
    meta = [f"{k} {isl[k]}" for k in ("nodes", "layout", "cards", "dataset", "gbs") if isl.get(k) not in (None, "")]
    lines = [island_row(isl, ix)]
    if meta:
        lines.append(" | ".join(meta))
    lines.append(f"HBM {fsig(last_value(rows, ix.get('hbm_gib')))}/{fsig(hbm_limit)} GiB"
                 f" | tokens/s {fsig(last_value(rows, ix.get('tps')), 4)}"
                 f" | MFU {fsig(last_value(rows, ix.get('mfu')))}%"
                 f" | lr {fsig(last_value(rows, ix.get('lr')))}"
                 f" | last step {fdur(isl.get('age_s'))} ago")
    if isl.get("last_push"):
        p = isl["last_push"]
        lines.append(f"last push step {p.get('step')} at {fclock(p.get('ts'))} (export {fdur(p.get('export_s'))})")
    if isl.get("last_apply"):
        a = isl["last_apply"]
        lines.append(f"last apply step {a.get('step')} at {fclock(a.get('ts'))} (apply {fdur(a.get('apply_s'))})")
    lines += ["", *steps_table(rows, ix, 5)]
    tail = rows[-chart_steps:]
    xs = [cell(r, ix.get("step")) for r in tail]
    charts = [Chart(f"{island_label(isl)} {name}", [(name, xs, [cell(r, ix.get(col)) for r in tail])],
                    xlabel="step")
              for name, col in (("loss", "loss"), ("grad_norm", "grad_norm"), ("step time s", "step_s"))]
    return Message(f"island {isl.get('id')}: {island_label(isl)}", lines, "report",
                   [("Status", "/status"), ("Resources", "/res")], [c for c in charts if not c.empty()])


def merge_message(view: dict, limit: int = 8) -> Message:
    sft = (view or {}).get("sft") or {}
    syn = sft.get("syncer") or {}
    mh = merge_health(view)
    lines = [syncer_line(sft) if sft else "no syncer data", merge_line(view)]
    if mh["weights"]:
        lines.append("merge weight " + ", ".join(f"{k} {fsig(v)}" for k, v in sorted(mh["weights"].items())))
    timed_out = [r for r in syn.get("outer") or [] if len(r) > 4 and r[4]]
    mx = merge_cols(sft)

    def mc(m: list, name: str) -> Any:
        return cell(m, mx.get(name))

    merges = [m for m in syn.get("merges") or [] if num(mc(m, "round_s")) is not None]
    med = statistics.median([mc(m, "round_s") for m in merges]) if merges else None
    slow = [m for m in merges if med and mc(m, "round_s") > 2 * med]
    if timed_out:
        lines.append(b(f"Timed-out rounds ({len(timed_out)})"))
        lines += [f"outer {r[1]} frag {r[2]} at {fclock(r[0])}" for r in timed_out[-limit:]]
    if slow:
        lines.append(b(f"Slow rounds (> 2x median {fdur(med)}): {len(slow)}"))
        lines += [f"outer {mc(m, 'step')} frag {mc(m, 'fragment')} round {fdur(mc(m, 'round_s'))}"
                  f" (quorum {fdur(mc(m, 'quorum_s'))}, grace {fdur(mc(m, 'grace_s'))},"
                  f" merge {fdur(mc(m, 'merge_s'))}) at {fclock(mc(m, 'ts'))}"
                  for m in slow[-limit:]]
    events = [e for e in syn.get("events") or [] if len(e) > 1 and e[1] in PROBLEMS]
    pc = problem_counts(sft)
    lines.append(f"rejected {pc['rejected']} | timeout {pc['timeout']} | not carried {pc['not_carried']}")
    if events:
        lines += [f"{fclock(e[0])} {e[1]}" + (f" learner {e[2]}" if e[2] is not None else "") for e in events[-limit:]]
    bad = [r for r in (view or {}).get("rounds_syncer") or [] if r.get("bad")]
    for r in bad[-limit:]:
        lines.append(f"round {r.get('round')}: missed {', '.join(map(str, r.get('missed') or [])) or '-'}"
                     f" | resend {r.get('resend', 0)}")
    walls = [(mc(m, "ts"), mc(m, "round_s")) for m in merges[-200:]]
    chart = Chart("syncer round time s", [("round", [w[0] for w in walls], [w[1] for w in walls])], time_axis=True)
    return Message("yeto merge", lines, "report", [("Status", "/status")], [] if chart.empty() else [chart])


def res_message(view: dict, *, hbm_limit: float | None, hbm_warn: float | None, rss_warn: float | None,
                host_min: float) -> Message:
    sft = (view or {}).get("sft") or {}
    lines = [b(f"HBM peak (warn > {fsig(hbm_warn)}, limit {fsig(hbm_limit)} GiB)")]
    c_hbm = step_cols(sft).get("hbm_gib")
    for isl in sft.get("islands") or []:
        rows = isl.get("rows") or []
        recent = column(rows[-20:], c_hbm)
        lines.append(f"{island_label(isl)}: last {fsig(last_value(rows, c_hbm))} | max of 20 steps"
                     f" {fsig(max(recent) if recent else None)} GiB")
    hosts = [(island_label(isl), node, ser[-1]) for isl in sft.get("islands") or []
             for node, ser in (isl.get("host") or {}).items() if ser]
    lines.append(b(f"Host memory available (warn < {fbytes(host_min)})"))
    lines += [f"{who} {node}: {fbytes(s[1])}" + (f" of {fbytes(s[2])}" if len(s) > 2 else "")
              for who, node, s in hosts] or ["no host samples"]
    lines.append(f"{b('Syncer RSS')} {fbytes(latest_rss(sft))} (warn > {fbytes(rss_warn)})")
    rss = (sft.get("syncer") or {}).get("rss") or []
    chart = Chart("syncer RSS GB", [("rss", [r[0] for r in rss[-500:]], [r[1] / 1e9 for r in rss[-500:]])],
                  time_axis=True)
    return Message("yeto resources", lines, "report", [("Status", "/status")], [] if chart.empty() else [chart])


def ckpt_message(view: dict, limit: int = 8) -> Message:
    sft = (view or {}).get("sft") or {}
    syn = sft.get("syncer") or {}
    writes = [e for e in syn.get("events") or [] if len(e) > 1 and e[1] in ("checkpoint", "checkpoint_failed")]
    lines = [b(f"Syncer checkpoint writes ({len(writes)})")]
    lines += [f"{fclock(e[0])} {'FAILED' if e[1] == 'checkpoint_failed' else 'written'}"
              + (f" in {fdur(e[4])}" if len(e) > 4 and num(e[4]) is not None else "") for e in writes[-limit:]] \
        or ["none seen in syncer.log"]
    ev = [r for r in (sft.get("eval") or {}).get("rows") or [] if r.get("ckpt_written")]
    if ev:
        lines.append(b("Evaluated checkpoints"))
        lines += [f"step {r.get('step')} written {r.get('ckpt_written')} loss {f3(r.get('loss'))}" for r in ev[-limit:]]
    isl_ck = [e for e in syn.get("events") or [] if len(e) > 1 and e[1] == "island_checkpoint"]
    if isl_ck:  # hook: island checkpoint events once the dashboard reports them
        lines.append(b("Island checkpoints"))
        lines += [f"{fclock(e[0])} learner {e[2]}" for e in isl_ck[-limit:]]
    return Message("yeto checkpoints", lines, "report", [("Status", "/status")])


def eval_message(view: dict, limit: int = 8) -> Message:
    rows = (((view or {}).get("sft") or {}).get("eval") or {}).get("rows") or []
    if not rows:
        return Message("yeto eval", ["no eval points yet"], "info")
    last = rows[-1]
    lines = eval_detail(last)
    names = eval_refs(rows[-limit:])
    lines += ["", b(" | ".join(["step", "loss", *(f"vs {n}" for n in names)]))]
    for r in rows[-limit:]:
        d = [fsigned(ref_cmp(r, n).get("dloss")) for n in names]
        lines.append(" | ".join([str(r.get("step")), f3(r.get("loss")), *d]))
    return Message(f"yeto eval: step {last.get('step')}", lines, "report", [("Status", "/status")],
                   [c for c in [eval_chart(rows)] if not c.empty()])


def eval_chart(rows: list[dict]) -> Chart:
    xs = [r.get("step") for r in rows]
    series = [("eval loss", xs, [r.get("loss") for r in rows])]
    for name in eval_refs(rows):
        if name != "prev":  # the previous point's loss is the eval line itself, shifted
            series.append((f"{name} ref", xs, [ref_cmp(r, name).get("ref_loss") for r in rows]))
    return Chart("held-out eval loss", series, xlabel="syncer outer step")


def eval_detail(row: dict) -> list[str]:
    lines = [f"loss {f3(row.get('loss'))} | ppl {fsig(row.get('ppl'), 4)}"]
    for label in eval_refs([row]):
        c = ref_cmp(row, label)
        if num(c.get("dloss")) is None:
            continue
        better = num(c.get("better"))
        se = num(c.get("se"))
        lines.append(f"vs {label}: dloss {fsigned(c['dloss'])} (ref {f3(c.get('ref_loss'))})"
                     + (f" | paired se {fsig(se, 2)}" if se is not None else "")
                     + (f" | better on {100 * better:.0f}%" if better is not None else ""))
    return lines


def details_message(view: dict, notices: list, *, ref: str) -> Message:
    """Context for one alert card: its notices, the involved islands' last 10 steps,
    the last syncer rounds when a syncer subject is involved."""
    sft = (view or {}).get("sft") or {}
    ix, mx = step_cols(sft), merge_cols(sft)
    lines = []
    for n in notices:
        lines.append(f"{b(n.title)} ({n.kind})")
        lines += [f"    {x}" for x in n.lines if x]
    islands = []
    for n in notices:
        if n.island is not None and n.island not in islands:
            islands.append(n.island)
    for iid in islands:
        isl = find_island(sft, str(iid))
        if isl:
            head = f"{island_label(isl)} (status {isl.get('status')}, last step {fdur(isl.get('age_s'))} ago)"
            lines += ["", b(head), *steps_table(isl.get("rows") or [], ix, 10)]
    if any(n.key and n.key[0] in ("syncer", "dashboard") for n in notices):
        merges = ((sft.get("syncer") or {}).get("merges") or [])[-10:]
        lines += ["", syncer_line(sft) if sft else "", b("outer | frag | round | quorum | grace | merge | at")]
        lines += [" | ".join([str(cell(m, mx.get("step"))), str(cell(m, mx.get("fragment")))]
                             + [fdur(cell(m, mx.get(k))) for k in ("round_s", "quorum_s", "grace_s", "merge_s")]
                             + [fclock(cell(m, mx.get("ts")))]) for m in merges]
    return Message(f"details {ref}", lines or ["nothing recorded"], "report", [("Status", "/status")])


def digest_message(view: dict, *, alert_counts: dict[str, int], top_alerts: list[str]) -> Message:
    """Daily digest: 24 h progress, per-island throughput / MFU, eval, alerts."""
    sft = (view or {}).get("sft") or {}
    ov = (view or {}).get("overview") or {}
    now = num(sft.get("now")) or time.time()
    lines = [syncer_line(sft) if sft else "no SFT data"]
    syn = sft.get("syncer") or {}
    outer_24 = [r for r in syn.get("outer") or [] if num(r[0]) is not None and r[0] >= now - 86400]
    steps = [r[1] for r in outer_24 if isinstance(r[1], int)]
    if steps:
        lines.append(f"outer steps in 24 h: {max(steps) - min(steps) + 1}")
    lines.append(b("Islands, last 24 h"))
    ix = step_cols(sft)
    for isl in sft.get("islands") or []:
        rows = window(isl.get("rows") or [], now, 86400, ix.get("ts"))
        loss = column(rows, ix.get("loss"))
        lines.append(f"{STATUS_DOT.get(isl.get('status') or 'unknown', '')} {island_label(isl)}: {len(rows)} steps"
                     f" | loss {f3(loss[0] if loss else None)} -> {f3(loss[-1] if loss else None)}"
                     f" | {fdur(step_p50(rows, ix))}/step | tokens/s {fsig(_mean(column(rows, ix.get('tps'))), 4)}"
                     f" | MFU {fsig(_mean(column(rows, ix.get('mfu'))))}%")
    rows = (sft.get("eval") or {}).get("rows") or []
    lines.append(eval_line(rows[-1] if rows else None))
    lines.append(merge_line(view))
    lines.append(f"{b('Alerts, last 24 h')} P0 {alert_counts.get('P0', 0)} | P1 {alert_counts.get('P1', 0)}"
                 f" | P2 {alert_counts.get('P2', 0)}")
    lines += [f"    {t}" for t in top_alerts[:5]]
    charts = [loss_chart(sft, 86400, "loss, last 24 h"), eval_chart(rows)]
    return Message(f"yeto daily digest: {ov.get('run') or 'run'} {time.strftime('%Y-%m-%d', time.localtime(now))}",
                   lines, "report", NAV_BUTTONS, [c for c in charts if not c.empty()])


def _mean(vals: list[float]) -> float | None:
    return statistics.fmean(vals) if vals else None
