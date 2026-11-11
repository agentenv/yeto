"""Alert rule table. Pure functions over the reducer state.

Severity: 0 critical / 1 warning / 2 info. Every alert carries a locate target
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
    # Island still starting (no rl_driver_start yet: weight load, Ray, engine init;
    # FN 2x8 measured 10-15 min). Tasks 8.1.
    "startup_warn_s": 1200.0,
    "startup_severe_s": 1800.0,
    # the actor init stage of a trainer start (model build + weight load: trainer_start_phase
    # actor_import_done -> actor_init_done, or rl_startup_step before weights_loaded) writes
    # no event for its whole length; K3.1 mbridge load measured ~22 min
    "startup_init_warn_s": 2700.0,
    "startup_init_severe_s": 3600.0,
    # 8.4: one startup sub-step (Ray up / engines ready / weights loaded) taking
    # longer than this while heartbeats still arrive (process alive, step stuck).
    "startup_step_warn_s": 900.0,
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
    # SFT page reference lines (not alerts), 0 = none drawn: the expected step time
    # of the recipe, the accelerator's HBM limit, the syncer host's RSS cap;
    # "slow step" = step time above the reference x factor
    "sft_ref_step_s": 0.0,
    "sft_hbm_limit_gib": 0.0,
    "sft_syncer_rss_limit_bytes": 0.0,
    "sft_slow_factor": 1.1,
    # SFT island status: "stopped" after no step for factor x step time p50 + slack
    "sft_stop_factor": 1.5,
    "sft_stop_slack_s": 300.0,
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


# keep in sync with yeto.rl.engine.telemetry.STARTUP_STEPS (dashboard does not import the engine)
STARTUP_STEPS = ("ray_connected", "engine_ready", "weights_loaded")


def _next_startup_step(done: str | None) -> str:
    if done is None or done not in STARTUP_STEPS:
        return STARTUP_STEPS[0]
    i = STARTUP_STEPS.index(done)
    return STARTUP_STEPS[i + 1] if i + 1 < len(STARTUP_STEPS) else "rl_driver_start"


ENDED = ("done", "stopped", "left", "lost", "unknown")
QUIET_OK = ("done", "stopped", "left", "unknown")  # no heartbeat alert (a lost island keeps its alert)


def heartbeat(cards: list[dict], th: dict) -> list[dict]:
    out = []
    for c in cards:
        age = c["last_event_age_s"]
        if c["status"] in QUIET_OK or c.get("finalized") or age is None:
            continue
        if c.get("starting"):
            stage = c.get("startup_stage") or {}
            warn, severe = stage.get("warn_s", th["startup_warn_s"]), stage.get("severe_s", th["startup_severe_s"])
            step_warn = max(th["startup_step_warn_s"], warn) if stage.get("init") else th["startup_step_warn_s"]
            step_age = c.get("startup_step_age_s")
            if step_age is not None and step_age > step_warn and age <= warn:
                pending = _next_startup_step(c.get("startup_step"))
                out.append(_alert(1, "startup_step", f"island {c['id']} startup step slow",
                                  f"waiting for {pending} for {step_age:.0f}s (previous step {c.get('startup_step') or 'start'}; "
                                  f"threshold {step_warn:.0f}s)", island=c["id"]))
            if age > severe:
                sev = 0
            elif age > warn:
                sev = 1
            else:
                continue
            out.append(_alert(sev, "startup", f"island {c['id']} no events during startup",
                              f"still starting (no rl_driver_start yet; stage: {stage.get('label') or 'n/a'}"
                              + (f" for {stage['age_s']:.0f}s" if stage.get("age_s") is not None else "")
                              + f"), last event {age:.0f}s ago (startup threshold {warn:.0f}s/{severe:.0f}s)",
                              island=c["id"], metric="reward"))
            continue
        if c.get("waiting_backlog"):
            continue  # a sampler idle at its backlog cap, by design (its trainers are behind)
        # without heartbeats an island is quiet for as long as its own cadence says
        # (rounds / generations / policy loads; the card's stale_after_s)
        warn = max(th["heartbeat_warn_s"], c.get("stale_after_s") or 0.0)
        severe = th["heartbeat_severe_s"] if warn <= th["heartbeat_warn_s"] else max(th["heartbeat_severe_s"], 2 * warn)
        if age > severe:
            sev = 0
        elif age > warn:
            sev = 1
        else:
            continue
        what = "heartbeat" if c["heartbeat_seen"] else "event (no heartbeat events, using the last event)"
        out.append(_alert(sev, "heartbeat", f"island {c['id']} heartbeat timeout",
                          f"last {what} {age:.0f}s ago (threshold {warn:.0f}s/"
                          f"{severe:.0f}s)", island=c["id"], round=c["round"],
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
        out.append(_alert(sev, "missed", f"island {iid} missed consecutive rounds",
                          f"rounds {rs[0]}–{rs[-1]}: no response for {n} rounds in a row", island=iid,
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
    return [_alert(1, "quorum", "quorum latency abnormal" + ("/resends" if resends else ""),
                   f"median quorum over the last {k} rounds {recent / 1000:.1f}s (baseline {base / 1000:.1f}s)"
                   + (f", {resends} resends" if resends else ""),
                   round=rounds[-1]["round"], metric="reward")]


def grad_norm(reducer, th: dict) -> list[dict]:
    out = []
    w = int(th["grad_norm_window"])
    for iid in reducer.island_ids():
        isl = reducer.islands[iid]
        for x in isl["nonfinite"]:
            out.append(_alert(0, "grad_norm_nan", f"island {iid} grad_norm/loss not finite",
                              f"NaN/Inf in round {x}", island=iid, round=x, metric="grad_norm"))
        pts = [(x, isl["points"][x]["grad_norm"]) for x in sorted(isl["points"])
               if isl["points"][x].get("grad_norm") is not None]
        for i in range(w, len(pts)):
            base = statistics.median(v for _, v in pts[i - w:i])
            x, v = pts[i]
            if base > 0 and v > th["grad_norm_spike_factor"] * base:
                out.append(_alert(1, "grad_norm_spike", f"island {iid} grad_norm spike",
                                  f"round {x} grad_norm {v:.3g} (baseline ~{base:.3g})",
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
            out.append(_alert(2, "clipfrac", f"island {iid} clipfrac high",
                              f"clip {clip:.3f} (threshold {th['clipfrac_info']}), info only",
                              island=iid, round=x, metric="clip"))
        if kl is not None and kl > th["kl_warn"]:
            out.append(_alert(1, "kl", f"island {iid} KL above threshold",
                              f"KL {kl:.4g} (threshold {th['kl_warn']})", island=iid, round=x, metric="kl"))
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
    detail = f"${cost['total_usd']:.0f} / ${cost['budget_usd']:.0f}(estimate, not billing data)"
    if eta is not None and math.isfinite(eta):
        detail += f", reached in about {eta:.1f} h at ${cost['burn_usd_h']:.1f}/h"
    return [_alert(sev, "budget", f"budget at {pct:.0f}%", detail)]


def recovery(reducer) -> list[dict]:
    out = []
    for iid in reducer.island_ids():
        isl = reducer.islands[iid]
        rr = isl["recovery_required"]
        if not rr or reducer.stopped_by_us(isl):  # failure caused by our own stop of the app
            continue
        last = rr[-1]
        tx = f", transaction {last['tx_id']}" if last.get("tx_id") else ""
        out.append(_alert(0, "recovery_required", f"island {iid} RECOVERY_REQUIRED",
                          f"{(last.get('error') or '')[:160]}{tx}({len(rr)} records)",
                          island=iid, round=last.get("round"), metric="reward"))
    return out


def source_lost(cards: list[dict], torn_down: bool = False) -> list[dict]:
    return [_alert(1, "source_lost", f"island {c['id']} tape source lost",
                   f"ssh follow disconnected: {(c['source_lost'].get('error') or '')[:120]}", island=c["id"])
            for c in cards if c["source_lost"] and not c["source_lost"].get("recovered")
            and not torn_down and c["status"] not in ("done", "stopped", "left")]
    # an island that ended (or a run torn down): its tape follow ending is expected


def evaluate(reducer, *, now: float, rounds: list[dict], cost: dict, cards: list[dict]) -> list[dict]:
    th = reducer.thresholds
    alerts = (recovery(reducer) + heartbeat(cards, th) + consecutive_missed(rounds, th)
              + quorum(rounds, th) + grad_norm(reducer, th) + clip_kl(reducer, th)
              + budget(cost, th)
              + source_lost(cards, torn_down=(reducer.operator_stop() or {}).get("cause") == "teardown"))
    alerts.sort(key=lambda a: a["sev"])
    for i, a in enumerate(alerts):
        a["id"] = i
    return alerts
