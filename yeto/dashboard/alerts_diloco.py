"""Alert rules of a run with a syncer (decoupled DiLoCo RL, synced groups).

Pure functions over ``topology_diloco.current``: the same numbers the topology panel
draws, so an alert always has its node, lane or fragment cell on the page. The numbers
below are this module's defaults (the run's syncer settings are not on the tapes);
the generic rules of ``alerts.py`` (heartbeat, missed rounds, trainer failed) still apply.
"""

from __future__ import annotations

from typing import Any

from . import topology_diloco
from .alerts import _alert

BCAST_WARN_S = 90.0  # a merged fragment not applied by a live island for this long
BCAST_SEVERE_S = 300.0
ROLLOUT_STALL_FACTOR = 3.0  # generating for longer than this x the island's median rollout
ROLLOUT_STALL_MIN_S = 60.0
REOPEN_RECENT_S = 600.0  # a merge after a quorum-timeout reopen within this long


def _dur(s: float) -> str:
    s = max(0, round(s))
    return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"


def roster(d: dict, now: float) -> list[dict]:
    if not d["running"] or d["roster"] >= d["M"]:
        return []
    gone = [t for t in d["trainers"] if not t["live"]]
    names = ", ".join(f"{t['id']} ({t['status']}{': ' + t['reason'] if t['reason'] else ''})" for t in gone)
    return [_alert(0, "diloco_roster", f"syncer roster {d['roster']}/{d['M']}",
                   f"trainer(s) not live: {names or 'n/a'}; a round needs quorum {d['quorum'] or 'n/a'} of "
                   f"{d['M']}, the missing islands' fragments wait for them", island=gone[0]["id"] if gone else None)]


def trainer_lost(d: dict, now: float) -> list[dict]:
    """A trainer the fleet or its own leave reports lost while the others still run."""
    if not d["running"]:
        return []
    out = []
    for t in d["trainers"]:
        if t["status"] != "lost" and not (t["status"] == "left" and t["reason"]):
            continue
        age = now - t["last_event_ts"] if t["last_event_ts"] else None
        out.append(_alert(0, "diloco_trainer_lost", f"trainer {t['id']} lost",
                          f"{t['status']}{': ' + t['reason'] if t['reason'] else ''}"
                          + (f", no event for {_dur(age)}" if age is not None else "")
                          + "; its PUSH and BCAST lanes are down", island=t["id"]))
    return out


def rounds_open(d: dict, now: float) -> list[dict]:
    out = []
    for r in d["rounds"]:
        if not r["blocked"]:
            continue
        miss = ", ".join(r["missing"]) or "none"
        out.append(_alert(0 if r["attempt"] > 1 else 1, "diloco_round_open",
                          f"step {r['step']} (f{r['frag']}) open {_dur(r['age_s'])}"
                          + (f", attempt {r['attempt']}" if r["attempt"] > 1 else ""),
                          f"pushed {len(r['pushed'])}/{d['M']}; no PUSH from trainer {miss}", round=r["step"]))
    for s in d["shards"]:
        last = s["last"] or {}
        if (last.get("attempt") or 1) > 1 and last.get("ts") and now - last["ts"] < REOPEN_RECENT_S:
            out.append(_alert(1, "diloco_round_open", f"step {last['step']} merged after a quorum timeout",
                              f"attempt {last['attempt']}: the syncer reopened the round; responded "
                              f"{', '.join(last['responded']) or 'none'} of {', '.join(last['expected']) or 'n/a'}",
                              round=last["step"]))
    return out


def bcast_backlog(d: dict, now: float) -> list[dict]:
    out = []
    for t in d["trainers"]:
        if not t["live"] or not t["pending_oldest_ts"]:
            continue
        age = now - t["pending_oldest_ts"]
        if age <= BCAST_WARN_S:
            continue
        frags = ", ".join(f"f{f} ×{n}" for f, n in sorted(t["pending"].items()))
        out.append(_alert(0 if age > BCAST_SEVERE_S else 1, "diloco_bcast_backlog",
                          f"trainer {t['id']}: {t['pending_n']} merged fragment(s) not applied, oldest {_dur(age)}",
                          f"{frags}; the island applies BCASTs only at its safe boundary "
                          f"({t['phase'] or 'phase n/a'}); the next PUSH of those fragments waits too",
                          island=t["id"]))
    return out


def strict_failures(d: dict, now: float) -> list[dict]:
    out = []
    for t in d["trainers"]:
        f = t["failure"]
        metric = str((f or {}).get("metric") or "")
        if f and "hash" in metric:
            out.append(_alert(0, "diloco_hash_mismatch", f"trainer {t['id']} policy hash mismatch after apply",
                              f"rl_strict_failure {metric}: {f['error']}", island=t["id"]))
        elif (f and "lr" in metric) or (t["live"] and t["lr_zero"]):
            out.append(_alert(0, "diloco_zero_lr", f"trainer {t['id']} learning rate 0",
                              f"rl_strict_failure {metric}: {f['error']}" if f else
                              "the last rl_local_round applied lr 0 before the final round: its deltas are empty",
                              island=t["id"]))
        elif f:
            out.append(_alert(0, "diloco_strict_failure", f"trainer {t['id']} strict failure {metric}",
                              f["error"], island=t["id"]))
    return out


def stuck(d: dict, now: float) -> list[dict]:
    if not d["stuck"]:
        return []
    frags = ", ".join(f"f{f} v{d['versions'][f]}" for f in d["stuck"])
    return [_alert(1, "diloco_fragment_stuck", "fragment version not advancing: " + frags,
                   f"no merge for more than 3 × the fragment period ({_dur(d['period_s'] or 0)}) or "
                   f"{_dur(topology_diloco.STUCK_MIN_S)}")]


def syncer_idle(d: dict, now: float) -> list[dict]:
    last = d["last_merge_ts_any"]
    limit = max(topology_diloco.IDLE_MIN_S, 4 * (d["merge_gap_s"] or 0))
    if not d["running"] or d["behind"] or last is None or now - last <= limit:
        return []
    return [_alert(1, "diloco_syncer_idle", f"syncer: no merge for {_dur(now - last)}",
                   f"recent merge interval {_dur(d['merge_gap_s'] or 0)} (threshold {_dur(limit)})")]


def rejected(d: dict, now: float) -> list[dict]:
    out = []
    for s in d["shards"]:
        last = s["last"] or {}
        if (last.get("rejected") or 0) > 0:
            out.append(_alert(1, "diloco_rejected_push", f"syncer rejected {last['rejected']} stale PUSH(es)",
                              f"step {last['step']} (f{last['frag']}): sync/rejected_stale_updates", round=last["step"]))
    return out


def rollout_stalled(d: dict, now: float) -> list[dict]:
    out = []
    for t in d["trainers"]:
        roll = t["island"]
        if not t["live"] or not roll or t["phase_raw"] != "generate" or not roll["rollout_p50_s"]:
            continue
        age = now - t["phase_since"]
        limit = max(ROLLOUT_STALL_MIN_S, ROLLOUT_STALL_FACTOR * roll["rollout_p50_s"])
        if age > limit:
            out.append(_alert(1, "diloco_rollout_stalled", f"island {t['id']} rollout stalled",
                              f"generating for {_dur(age)}, median rollout {_dur(roll['rollout_p50_s'])} "
                              f"(threshold {_dur(limit)})", island=t["id"]))
    return out


RULES = (roster, trainer_lost, rounds_open, bcast_backlog, strict_failures, stuck, syncer_idle, rejected,
         rollout_stalled)


def evaluate(reducer, now: float, cards: list[dict]) -> list[dict[str, Any]]:
    d = topology_diloco.current(reducer, cards, now)
    if d is None:
        return []
    return [a for rule in RULES for a in rule(d, now)]
