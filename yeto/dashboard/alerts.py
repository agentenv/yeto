"""Alert rule table (design D6). Pure functions over the reducer state.

Severity: 0 严重 / 1 警告 / 2 提示. Every alert carries a locate target
(``island``, ``round``, ``metric``) the page uses to focus the chart, the
round table and the island drill-down. Thresholds are configurable through
``merged_thresholds`` (CLI ``--thresholds file.json``).
"""

from __future__ import annotations

import math
import statistics
from typing import Any

DEFAULT_THRESHOLDS: dict[str, float] = {
    "heartbeat_warn_s": 60.0,
    "heartbeat_severe_s": 300.0,
    "missed_consecutive_warn": 2,
    "missed_consecutive_severe": 5,
    "quorum_recent_rounds": 7,
    "quorum_min_rounds": 10,
    "quorum_factor": 3.0,
    "grad_norm_window": 8,
    "grad_norm_spike_factor": 5.0,
    "clipfrac_info": 0.2,
    "kl_warn": 0.1,
    "budget_warn_pct": 60.0,
    "budget_severe_pct": 80.0,
}


def merged_thresholds(overrides: dict | None) -> dict:
    out = dict(DEFAULT_THRESHOLDS)
    for k, v in (overrides or {}).items():
        if k not in DEFAULT_THRESHOLDS:
            raise ValueError(f"unknown alert threshold {k!r}")
        out[k] = float(v)
    return out


def _alert(sev: int, rule: str, title: str, detail: str, *, island: str | None = None,
           round: Any = None, metric: str | None = None) -> dict:
    return {"sev": sev, "rule": rule, "title": title, "detail": detail,
            "island": island, "round": round, "metric": metric}


def heartbeat(cards: list[dict], th: dict) -> list[dict]:
    out = []
    for c in cards:
        age = c["last_event_age_s"]
        if c["status"] == "done" or c.get("finalized") or age is None:
            continue
        if age > th["heartbeat_severe_s"]:
            sev = 0
        elif age > th["heartbeat_warn_s"]:
            sev = 1
        else:
            continue
        what = "心跳" if c["heartbeat_seen"] else "事件（无心跳事件，按最后事件计）"
        out.append(_alert(sev, "heartbeat", f"岛 {c['id']} 心跳超时",
                          f"最后{what} {age:.0f}s 前（阈值 {th['heartbeat_warn_s']:.0f}s/"
                          f"{th['heartbeat_severe_s']:.0f}s）", island=c["id"], round=c["round"],
                          metric="reward"))
    return out


def consecutive_missed(rounds: list[dict], th: dict) -> list[dict]:
    run: dict[str, list] = {}
    best: dict[str, list] = {}
    for r in rounds:
        missed = set(r["missed"])
        for iid in list(run):
            if iid not in missed:
                run.pop(iid)
        for iid in missed:
            run.setdefault(iid, []).append(r["round"])
            if len(run[iid]) > len(best.get(iid, [])):
                best[iid] = list(run[iid])
    out = []
    for iid, rs in best.items():
        n = len(rs)
        if n >= th["missed_consecutive_severe"]:
            sev = 0
        elif n >= th["missed_consecutive_warn"]:
            sev = 1
        else:
            continue
        out.append(_alert(sev, "missed", f"岛 {iid} 连续 missed",
                          f"round {rs[0]}–{rs[-1]} 连续 {n} 轮未响应", island=iid,
                          round=rs[0], metric="reward"))
    return out


def quorum(rounds: list[dict], th: dict) -> list[dict]:
    qs = [r["quorum_ms"] for r in rounds if r["quorum_ms"] is not None]
    if len(qs) < th["quorum_min_rounds"]:
        return []
    k = int(th["quorum_recent_rounds"])
    base = statistics.median(qs[:-k]) if len(qs) > k else statistics.median(qs)
    recent = statistics.median(qs[-k:])
    if base <= 0 or recent < th["quorum_factor"] * base:
        return []
    resends = sum(r["resend"] for r in rounds[-k:])
    return [_alert(1, "quorum", "quorum 时延异常" + ("/重发" if resends else ""),
                   f"近 {k} 轮 quorum 中位 {recent / 1000:.1f}s（基线 {base / 1000:.1f}s）"
                   + (f"，重发 {resends} 次" if resends else ""),
                   round=rounds[-1]["round"], metric="reward")]


def grad_norm(reducer, th: dict) -> list[dict]:
    out = []
    w = int(th["grad_norm_window"])
    for iid in reducer.island_ids():
        isl = reducer.islands[iid]
        for x in isl["nonfinite"]:
            out.append(_alert(0, "grad_norm_nan", f"岛 {iid} grad_norm/loss 非有限",
                              f"round {x} 出现 NaN/Inf", island=iid, round=x, metric="grad_norm"))
        pts = [(x, isl["points"][x]["grad_norm"]) for x in sorted(isl["points"])
               if isl["points"][x].get("grad_norm") is not None]
        for i in range(w, len(pts)):
            base = statistics.median(v for _, v in pts[i - w:i])
            x, v = pts[i]
            if base > 0 and v > th["grad_norm_spike_factor"] * base:
                out.append(_alert(1, "grad_norm_spike", f"岛 {iid} grad_norm 尖峰",
                                  f"round {x} grad_norm {v:.3g}（基线 ~{base:.3g}）",
                                  island=iid, round=x, metric="grad_norm"))
    return out


def clip_kl(reducer, th: dict) -> list[dict]:
    out = []
    for iid in reducer.island_ids():
        pts = reducer.islands[iid]["points"]
        if not pts:
            continue
        x = max(pts)
        clip = pts[x].get("clip")
        kl = pts[x].get("kl")
        if clip is not None and clip > th["clipfrac_info"]:
            out.append(_alert(2, "clipfrac", f"岛 {iid} clipfrac 偏高",
                              f"clip {clip:.3f}（阈值 {th['clipfrac_info']}），仅提示",
                              island=iid, round=x, metric="clip"))
        if kl is not None and kl > th["kl_warn"]:
            out.append(_alert(1, "kl", f"岛 {iid} KL 超阈值",
                              f"KL {kl:.4g}（阈值 {th['kl_warn']}）", island=iid, round=x, metric="kl"))
    return out


def budget(cost: dict, th: dict) -> list[dict]:
    pct = cost.get("budget_pct")
    if pct is None:
        return []
    if pct >= th["budget_severe_pct"]:
        sev = 0
    elif pct >= th["budget_warn_pct"]:
        sev = 1
    else:
        return []
    eta = cost.get("hours_to_cap")
    detail = f"${cost['total_usd']:.0f} / ${cost['budget_usd']:.0f}（估算，非账单）"
    if eta is not None and math.isfinite(eta):
        detail += f"，按 ${cost['burn_usd_h']:.1f}/h 约 {eta:.1f} h 后触达"
    return [_alert(sev, "budget", f"预算逼近 {pct:.0f}%", detail)]


def recovery(reducer) -> list[dict]:
    out = []
    for iid in reducer.island_ids():
        rr = reducer.islands[iid]["recovery_required"]
        if not rr:
            continue
        last = rr[-1]
        tx = f"，E1 {last['tx_id']}" if last.get("tx_id") else ""
        out.append(_alert(0, "recovery_required", f"岛 {iid} RECOVERY_REQUIRED",
                          f"{(last.get('error') or '')[:160]}{tx}（共 {len(rr)} 条记录）",
                          island=iid, round=last.get("round"), metric="reward"))
    return out


def source_lost(cards: list[dict]) -> list[dict]:
    return [_alert(1, "source_lost", f"岛 {c['id']} 磁带来源丢失",
                   f"ssh 跟随断开：{(c['source_lost'].get('error') or '')[:120]}", island=c["id"])
            for c in cards if c["source_lost"] and not c["source_lost"].get("recovered")]


def evaluate(reducer, *, now: float, rounds: list[dict], cost: dict, cards: list[dict]) -> list[dict]:
    th = reducer.thresholds
    alerts = (recovery(reducer) + heartbeat(cards, th) + consecutive_missed(rounds, th)
              + quorum(rounds, th) + grad_norm(reducer, th) + clip_kl(reducer, th)
              + budget(cost, th) + source_lost(cards))
    alerts.sort(key=lambda a: a["sev"])
    for i, a in enumerate(alerts):
        a["id"] = i
    return alerts
