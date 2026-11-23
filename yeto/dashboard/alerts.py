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
    # CompactionRL (sampler rounds' agentic.compaction), over each sampler's last
    # compaction_window_rounds rounds that carry it, once it has compaction_min_trials
    # summaries / compacted trials: summary_ok share below warn / severe; share of the
    # compacted trials truncated after compacting above warn; share of trials that
    # compacted at all above info (the context budget is small for the tasks)
    "compaction_window_rounds": 5,
    "compaction_min_trials": 5,
    "compaction_summary_ok_warn": 0.9,
    "compaction_summary_ok_severe": 0.5,
    "compaction_truncated_warn": 0.2,
    "compaction_frac_info": 0.5,
    # share of multi-segment trials not written (lost: drain timeout / other version /
    # aborted; dropped: group not written) minus the share of one-segment trials
    "compaction_long_loss_gap": 0.2,
    # trials the gateway invalidated for a fork / history rewrite of a compacted session
    # (compaction_fork_lost), as a share of the compacted trials (written + fork-lost)
    "compaction_fork_lost_warn": 0.05,
    # agentic samplers: trials that ended over the last trial_fail_window_rounds rounds of the
    # current incarnation in which any trial ended, once trial_fail_min_trials ended: none
    # with a verdict (ok / policy_limit) is critical, an aborted (infra) share above warn
    # a warning. Mirrors the run breaker's "every group INFRA" rule on the page.
    "trial_fail_window_rounds": 10,
    "trial_fail_min_trials": 8,
    "trial_fail_warn": 0.5,
    # an async run whose samplers are up while no trainer island is on the tapes (its
    # launch failed, or it was torn down): seconds since the first sampler event
    "trainer_missing_warn_s": 1200.0,
    "trainer_missing_severe_s": 1800.0,
    # trainer launches that ended with a failure leave (island_leave not STOPPED / DONE ...)
    # in the last trainer_fail_window_s: one is a warning, trainer_fail_loop or more critical
    "trainer_fail_window_s": 3600.0,
    "trainer_fail_loop": 2,
    # a sampler waiting on a full backlog (as it counts it) while the trainers' read
    # position leaves less than its cap: it reported within this many seconds
    "stale_read_recent_s": 300.0,
    # a trainer skipping releases because the previous upload is still in flight, with no
    # release since the first skip of the streak, for this long
    "release_stall_warn_s": 600.0,
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
            # a step whose groups all had equal rewards (zero advantage) has grad_norm exactly 0:
            # no gradient, not a small one; it would pull the baseline down and flag normal steps
            prev = [v for _, v in pts[i - w:i] if v != 0]
            if len(prev) < (w + 1) // 2:
                continue
            base = statistics.median(prev)
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


def compaction(reducer, th: dict) -> list[dict]:
    from .async_rl import _row_order, compaction_total

    out = []
    k, least = int(th["compaction_window_rounds"]), th["compaction_min_trials"]
    state = getattr(reducer, "async_", None)
    for iid, s in sorted((state.sampler if state is not None else {}).items()):
        rows = [r for r in sorted(s["rounds"].values(), key=_row_order) if r.get("compaction")][-k:]
        c = compaction_total(rows)
        if c is None:
            continue
        last = rows[-1].get("rollout_id")
        ok, n = c.get("summary_ok_frac"), c.get("summaries") or 0
        if ok is not None and n >= least and ok < th["compaction_summary_ok_warn"]:
            sev = 0 if ok < th["compaction_summary_ok_severe"] else 1
            cut = c.get("summary_truncated_frac")
            why = ("the policy's summaries miss the <summary> block, the next segment resumes from raw text"
                   if not cut else
                   f"{cut:.0%} of the summaries hit the summary turn's max_tokens (summary_truncated): raise "
                   "agentic.compaction.summary_max_tokens if t_comp has room; the rest miss the <summary> block")
            out.append(_alert(sev, "compaction_summary", f"sampler {iid} compaction summaries failing",
                              f"summary_ok {ok:.0%} of {n:.0f} summaries in the last {len(rows)} rounds "
                              f"(threshold {th['compaction_summary_ok_warn']:.0%}): {why}", island=iid,
                              round=last))
        tr, nc = c.get("truncated_compacted_frac"), c.get("compacted_trials") or 0
        if tr is not None and nc >= least and tr > th["compaction_truncated_warn"]:
            out.append(_alert(1, "compaction_truncated", f"sampler {iid} trials truncated after compaction",
                              f"{tr:.0%} of {nc:.0f} compacted trials in the last {len(rows)} rounds ended truncated "
                              f"(threshold {th['compaction_truncated_warn']:.0%}; compactions used up or "
                              "max_response_len reached)", island=iid, round=last))
        fate = c.get("by_segments") or {}
        long_ = [v for b, v in fate.items() if b in ("2", "3", "4+")]
        n_long = sum(v["written"] + sum(v["lost"].values()) + sum(v["dropped"].values()) for v in long_)
        if n_long >= least:
            bad = sum(sum(v["lost"].values()) + sum(v["dropped"].values()) for v in long_)
            short = (fate.get("1") or {}).get("not_written_frac") or 0.0
            gap = bad / n_long - short
            if gap > th["compaction_long_loss_gap"]:
                out.append(_alert(1, "compaction_long_lost", f"sampler {iid} loses long (compacted) trials",
                                  f"{bad / n_long:.0%} of {n_long:.0f} multi-segment trials not written vs "
                                  f"{short:.0%} of one-segment trials in the last {len(rows)} rounds "
                                  f"(gap threshold {th['compaction_long_loss_gap']:.0%}): the GRPO baseline leans "
                                  "to short trials; raise drain_timeout_s / reload_max_defer_s / max_policy_age "
                                  "or use a live engine", island=iid, round=last))
        fl, ne = c.get("fork_lost_frac"), c.get("compaction_exposed_trials") or 0
        if fl is not None and ne >= least and fl > th["compaction_fork_lost_warn"]:
            out.append(_alert(1, "compaction_fork_lost", f"sampler {iid} loses trials to forks of compacted sessions",
                              f"{fl:.0%} of {ne:.0f} compacted trials in the last {len(rows)} rounds were invalidated "
                              f"(compaction_fork_lost; threshold {th['compaction_fork_lost_warn']:.0%}): the harness "
                              "retried or rewrote history after a compaction; check its request timeout and "
                              "retry_replays", island=iid, round=last))
        cf, nt = c.get("compacted_frac"), c.get("trials") or 0
        if cf is not None and nt >= least and cf > th["compaction_frac_info"]:
            out.append(_alert(2, "compaction_frac", f"sampler {iid} most trials compact",
                              f"{cf:.0%} of {nt:.0f} trials compacted in the last {len(rows)} rounds, "
                              f"{c.get('compactions_mean') or 0:.2f} compactions per trial "
                              f"(threshold {th['compaction_frac_info']:.0%}), info only", island=iid, round=last))
    return out


def store_unreadable(store: dict | None) -> list[dict]:
    """``--store`` reads failing since the last clean poll: without this the page shows
    "no data", the same as a run that has not started."""
    if not store or not store.get("failing"):
        return []
    where = store.get("uri") or "the run store"
    return [_alert(1, "store_unreadable", "run store unreadable",
                   f"{store['failing']} failed read(s) of {where} since the last clean poll: "
                   f"{(store.get('last_error') or '')[:200]}")]


def trial_failures(reducer, th: dict) -> list[dict]:
    from .async_rl import _row_order

    out = []
    state = getattr(reducer, "async_", None)
    for iid, s in sorted((state.sampler if state is not None else {}).items()):
        inc = s.get("incarnation")
        rows = [r for r in sorted(s["rounds"].values(), key=_row_order)
                if r.get("incarnation") == inc and any(r.get(f"trials_{k}") for k in ("ok", "policy_limit", "aborted"))]
        rows = rows[-int(th["trial_fail_window_rounds"]):]
        tot = {k: sum(r.get(f"trials_{k}") or 0 for r in rows) for k in ("ok", "policy_limit", "aborted")}
        ended = sum(tot.values())
        if not rows or ended < th["trial_fail_min_trials"]:
            continue
        last = rows[-1]
        why = ", ".join(f"{k} {v:g}" for k, v in sorted((last.get("infra_retries_by") or {}).items()))
        lost = ", ".join(f"{k} {v:g}" for k, v in sorted((last.get("lost_by") or {}).items()))
        detail = (f"{tot['aborted']:g} of {ended:g} trials ended aborted (infrastructure) in the last {len(rows)} "
                  f"round(s), ok {tot['ok']:g}, policy limit {tot['policy_limit']:g}"
                  + (f"; infra retries so far: {why}" if why else "") + (f"; lost: {lost}" if lost else ""))
        if tot["ok"] + tot["policy_limit"] == 0:
            out.append(_alert(0, "trial_fail", f"sampler {iid}: no trial is finishing", detail, island=iid))
        elif tot["aborted"] / ended > th["trial_fail_warn"]:
            out.append(_alert(1, "trial_fail", f"sampler {iid}: most trials fail on infrastructure", detail,
                              island=iid))
    return out


def trainer_missing(reducer, th: dict, now: float, cards: list[dict]) -> list[dict]:
    state = getattr(reducer, "async_", None)
    if state is None:
        return []
    roles = {c["id"]: state.role(c["id"]) for c in cards}
    live = [c for c in cards if roles[c["id"]] == "sampler" and c["status"] in ("ok", "waiting", "starting")]
    if not live or any(r == "trainer" for r in roles.values()):
        return []
    firsts = [reducer.islands[c["id"]].get("first_ts") for c in live if c["id"] in reducer.islands]
    firsts = [t for t in firsts if t is not None]
    if not firsts:
        return []
    age = now - min(firsts)
    if age <= th["trainer_missing_warn_s"]:
        return []
    stop = reducer.operator_stop() or {}
    torn = (f"; the launcher tore the run's app down (operator stop: {stop.get('cause')}, {stop.get('marker')})"
            if stop.get("cause") else "")
    sev = 0 if age > th["trainer_missing_severe_s"] else 1
    names = ", ".join(c["id"] for c in live)
    return [_alert(sev, "trainer_missing", "samplers running without a trainer",
                   f"{len(live)} sampler(s) up for {age / 60:.0f} min ({names}) and no trainer island on the "
                   f"tapes: their groups pile up in the buffer and nothing trains them{torn}",
                   island=live[0]["id"])]


def trainer_failed(reducer, th: dict, now: float) -> list[dict]:
    from .reducer import CLEAN_LEAVE

    out = []
    state = getattr(reducer, "async_", None)
    for iid, t in sorted((state.trainer if state is not None else {}).items()):
        bad = [x for x in t.get("leaves") or []
               if str(x.get("reason") or "").upper() not in CLEAN_LEAVE
               and x.get("ts") is not None and now - x["ts"] <= th["trainer_fail_window_s"]]
        if not bad:
            continue
        last = bad[-1]
        why = last.get("cause") or last.get("error_type") or "no error text"
        detail = (f"{len(bad)} failed launch(es) in the last {th['trainer_fail_window_s'] / 60:.0f} min, "
                  f"last {last.get('reason')}: {why}")
        isl = reducer.islands.get(iid) or {}
        relaunched = isl.get("launch_ts") is not None and isl["launch_ts"] > last["ts"]
        if relaunched:  # the current launch started after the last failure: say so, not critical
            out.append(_alert(1, "trainer_failed", f"trainer {iid} failed earlier, relaunched",
                              detail + "; the current launch started after it", island=iid))
        elif len(bad) >= th["trainer_fail_loop"]:
            out.append(_alert(0, "trainer_failed", f"trainer {iid} keeps failing", detail, island=iid))
        else:
            out.append(_alert(1, "trainer_failed", f"trainer {iid} failed", detail, island=iid))
    return out


def stale_read_position(reducer, th: dict, now: float) -> list[dict]:
    state = getattr(reducer, "async_", None)
    if state is None or not state.sampler:
        return []
    series = state.backlog_series() or {}
    pts = [p for p in series.get("points") or [] if p[3] is not None]
    if not pts:
        return []
    out = []
    for iid, s in sorted(state.sampler.items()):
        cap = s.get("max_backlog_groups")
        rep = s["backlog"][-1] if s["backlog"] else None
        if s.get("ended") or cap is None or rep is None or not rep[2] or now - rep[0] > th["stale_read_recent_s"]:
            continue
        seen = [p for p in pts if p[0] <= rep[0]]  # what the trainers had read when the sampler reported
        if not seen or seen[-1][3] >= float(cap) or rep[1] < float(cap):
            continue
        p = seen[-1]
        out.append(_alert(1, "stale_read_position", f"sampler {iid} waits on a stale read position",
                          f"it counts {rep[1]:g} groups unread (cap {cap}) and stops generating, but the trainers "
                          f"have read {p[2]:g} of the {p[1]:g} groups written ({p[3]:g} left): a sampler learns the "
                          "read position from a policy release or the trainer's read heartbeat "
                          "(sample-buffer/read-position.json), and none came since", island=iid))
    return out


def release_stalled(reducer, th: dict, now: float, stale: list[dict]) -> list[dict]:
    state = getattr(reducer, "async_", None)
    if state is None:
        return []
    out = []
    for iid, t in sorted(state.trainer.items()):
        last_rel = max((r["ts"] for r in state.releases if r.get("trainer") == iid and r.get("ts") is not None),
                       default=None)
        streak = [x for x in t.get("upload_skips") or [] if last_rel is None or x["ts"] > last_rel]
        if not streak or now - streak[0]["ts"] <= th["release_stall_warn_s"]:
            continue
        versions = ", ".join(f"v{x['policy_version']}" for x in streak[-5:])
        detail = (f"a policy release has been uploading for over {(now - streak[0]['ts']) / 60:.0f} min: the "
                  f"trainer skipped {len(streak)} release(s) since ({versions}) and wrote no rl_policy_released")
        if stale:  # samplers wait for this release to learn the read position: the loop stands still
            detail += "; samplers wait for it (stale read position), so nothing is generated or trained"
        out.append(_alert(0 if stale else 1, "release_stalled", f"trainer {iid}: policy release stuck uploading",
                          detail, island=iid))
    return out


def _with_release(reducer, th: dict, now: float) -> list[dict]:
    stale = stale_read_position(reducer, th, now)
    return stale + release_stalled(reducer, th, now, stale)


def evaluate(reducer, *, now: float, rounds: list[dict], cost: dict, cards: list[dict],
             store: dict | None = None) -> list[dict]:
    th = reducer.thresholds
    alerts = (store_unreadable(store) + recovery(reducer) + heartbeat(cards, th) + consecutive_missed(rounds, th)
              + quorum(rounds, th) + grad_norm(reducer, th) + clip_kl(reducer, th)
              + budget(cost, th) + compaction(reducer, th) + trial_failures(reducer, th)
              + trainer_missing(reducer, th, now, cards) + trainer_failed(reducer, th, now)
              + _with_release(reducer, th, now)
              + source_lost(cards, torn_down=(reducer.operator_stop() or {}).get("cause") == "teardown"))
    alerts.sort(key=lambda a: a["sev"])
    for i, a in enumerate(alerts):
        a["id"] = i
    return alerts
