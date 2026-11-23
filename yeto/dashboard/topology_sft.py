"""Live topology of a decoupled DiLoCo SFT run: learner islands <-> syncer and the fragment
matrix, in the shape of ``topology_diloco.view`` (kind ``sft``) so the page draws it with the
same component as an RL run with a syncer.

Sources, all of them on the tapes today:

* islands: the SFT adapter's view (``sft.SftState``): steps from ms-swift ``logging.jsonl``,
  the yeto plugin's ``yeto_steps.jsonl`` (``export_s`` on a step that pushed a fragment,
  ``applies`` / ``apply_s`` on a step that applied broadcasts), ``bench_metrics.jsonl``,
  ``island.json`` (machines) and ``args.json`` (parallel layout, cards);
* syncer: the merge records (``responders[]``: ``base_version``, ``staleness``,
  ``c_tokens``, ``arrived_unix`` / ``recv_seconds``) and ``delta_accepted`` records of the
  event tape, ``cmdline.txt`` (``--sync-interval-steps`` = H, ``--total-steps``).

Known gaps: an SFT learner writes no fragment-version vector and does not say which fragment a
broadcast it applied belonged to. A fragment cell therefore shows the island's last
contribution (the merge step and the base version its delta was computed on); "BCAST pending"
counts the merges after the island's last applying step, and a BCAST window ends at that next
applying step (approximate). Which round a ``delta_accepted`` belongs to is not on the tape:
the deltas accepted since the last merge are drawn as the next step's round.
"""

from __future__ import annotations

from .sft import STEP_COLS
from .topology import RATE_S, WINDOW_S
from .topology_diloco import (
    MAX_FLOWS,
    _f,
    _int,
    _members,
    merge_rows,
    push_flows,
    shard_count,
    shard_rows,
    stuck_fragments,
    syncer_state,
)

COL = {c: k for k, c in enumerate(STEP_COLS)}


def _last(rows: list[list], col: str) -> list | None:
    k = COL[col]
    return next((r for r in reversed(rows) if r[k] is not None), None)


def _contributions(merges: list[dict]) -> dict[str, dict[int, dict]]:
    """Each island's last contribution to each fragment, from the merge records' responders."""
    out: dict[str, dict[int, dict]] = {}
    for m in merges:
        for r in m["rec"].get("responders") or []:
            if isinstance(r, dict) and r.get("id") is not None:
                out.setdefault(str(r["id"]), {})[m["frag"]] = {
                    "step": m["step"], "frag": m["frag"], "base": _int(r.get("base_version")),
                    "staleness": _int(r.get("staleness")), "tokens": _f(r.get("c_tokens")),
                    "recv_s": _f(r.get("recv_seconds")), "ts": _f(r.get("arrived_unix")) or m["ts"]}
    return out


def _applies(isl: dict) -> list[tuple[float, int, float | None]]:
    """(time, step, apply_s) of the island's steps that applied broadcasts."""
    a, t = COL["applies"], COL["apply_s"]
    return [(r[1], r[0], r[t]) for r in isl["rows"] if r[1] is not None and (r[a] or (r[t] or 0) > 0)]


def _nodes(isl: dict, launches: int) -> list[dict]:
    """One label per machine (``island.json`` nodes): its cards; every card is a learner card."""
    names = [str(x) for x in isl.get("nodes") or []]
    cards = _int(isl.get("cards"))
    machines = max(1, len(names))
    per = cards // machines if cards and cards % machines == 0 else None
    return [{"id": f"n{k}", "machine": names[k] if k < len(names) else None,
             "machine_full": names[k] if k < len(names) else None, "gpu": None, "gpus": per, "roles": ["T"],
             "inferred": False, "incarnation": launches, "util": None, "mem_pct": None} for k in range(machines)]


def _learner(isl: dict, contrib: dict[int, dict], merges: list[dict], pushed: dict, launches: int, now: float) -> dict:
    iid, rows, status = isl["id"], isl["rows"], isl["status"]
    last = rows[-1] if rows else None
    applies = _applies(isl)
    la = applies[-1] if applies else None
    # merged since the island's last applying step: not applied yet (approximate, see module doc)
    pending: dict[int, list[float]] = {}
    for m in merges:
        members = _members(m["rec"])
        if (iid in members or not members) and m["ts"] is not None and (la is None or m["ts"] > la[0]):
            pending.setdefault(m["frag"], []).append(m["ts"])
    lp = max(contrib.values(), key=lambda c: c["step"]) if contrib else None
    pick = lambda col: (_last(rows, col) or [None] * len(STEP_COLS))[COL[col]]  # noqa: E731
    return {
        "id": iid, "name": isl.get("name"), "gpu": None, "gpus": _int(isl.get("cards")), "arch": None,
        "status": status, "live": status == "ok", "reason": None, "finalized": status in ("done", "stopped"),
        "last_event_ts": isl.get("last_ts"), "step": last[0] if last else None, "steps_total": isl.get("steps_total"),
        "outer": None, "versions": None, "since": None,
        "contrib": [contrib.get(f) for f in range(max(contrib) + 1)] if contrib else [],
        "pushed": pushed, "pending": {str(f): len(ts) for f, ts in pending.items()},
        "pending_oldest_ts": min((t for ts in pending.values() for t in ts), default=None),
        "pending_n": sum(len(ts) for ts in pending.values()),
        "hash": None, "lr": pick("lr"), "lr_zero": False, "phase": None, "phase_raw": None, "phase_since": None,
        "last_push": dict(lp, export_s=(isl.get("last_push") or {}).get("export_s")) if lp else None,
        "last_bcast": None, "last_apply": {"ts": la[0], "step": la[1], "apply_s": la[2]} if la else None,
        "apply_s": la[2] if la else None, "apply_ts": la[0] if la else None, "full_push": None, "failure": None,
        "gpu_util": None, "mem_pct": None, "trained_groups": 0, "truncated": None, "island": None,
        "learner": {"layout": isl.get("layout"), "dataset": isl.get("dataset"), "loss": pick("loss"),
                    "tps": pick("tps"), "step_s": pick("step_s")},
        "nodes": _nodes(isl, launches),
    }


def _bcast_flows(merges: list[dict], learners: list[dict], isls: dict[str, dict], lo: float) -> list[dict]:
    """Merge -> the member island's next applying step (approximate: ``a``)."""
    out = []
    for m in merges:
        if m["ts"] is None or m["ts"] < lo:
            continue
        for t in learners:
            members = _members(m["rec"])
            if members and t["id"] not in members:
                continue
            nxt = next((a for a in _applies(isls[t["id"]]) if a[0] > m["ts"]), None)
            if nxt is not None:
                out.append({"k": "b", "i": t["id"], "sh": m["shard"], "f": m["frag"], "t0": m["ts"],
                            "t1": max(m["ts"] + 0.3, nxt[0] - (nxt[2] or 0)), "b": None, "a": True})
    return out


def view(reducer, now: float) -> dict | None:
    """The ``diloco`` part of the topology view for an SFT run (kind ``sft``)."""
    sv = reducer.sft_view(now)
    if sv is None:
        return None
    n = shard_count(reducer)
    merges = merge_rows(reducer, now, n)
    p = max([m["frag"] + 1 for m in merges] + [0]) or None
    sy = syncer_state(merges, p, n)
    isls = {i["id"]: i for i in sv["islands"]}
    ids = list(isls)
    live = [i for i in ids if isls[i]["status"] == "ok"]
    running, done = bool(live), bool(ids) and not live
    expected = sy["expected"]
    m_size = len(expected) or len(ids)

    # deltas the syncer accepted after the last merge: the next step's round
    last_any = sy["last_merge_ts_any"]
    acc = [(ts, str(i)) for ts, i in reducer.sft.accepted
           if ts is not None and ts <= now and i is not None and (last_any is None or ts > last_any)]
    rounds, pushed = [], {}
    if acc and p and running:
        step = max(sy["merged_steps"], default=0) + 1
        frag = (step - 1) % p
        since = min(ts for ts, _ in acc)
        who = sorted({i for _, i in acc}, key=lambda i: (0, int(i)) if i.isdigit() else (1, i))
        state = "merging" if len(who) >= m_size else "gathering"
        rounds.append({"step": step, "frag": frag, "shard": (step - 1) % n, "since": since, "attempt": 1,
                       "pushed": who, "age_s": now - since, "state": state,
                       "blocked": state != "merging" and now - since > sy["open_limit_s"],
                       "missing": [i for i in (expected or ids) if i not in who]})
        pushed = {i: {str(frag): step} for i in who}

    contrib = _contributions(merges)
    learners = [_learner(isls[i], contrib.get(i, {}), merges, pushed.get(i, {}),
                         (reducer.sft.islands.get(i) or {}).get("sync_launch", 0) + 1, now) for i in ids]
    lo = now - WINDOW_S
    flows = push_flows(merges, lo, {}) + _bcast_flows(merges, learners, isls, lo)
    flows.sort(key=lambda f: f["t0"])
    meta = sv["syncer"]["meta"]
    arrived5 = [r for m in merges for r in m["rec"].get("responders") or []
                if isinstance(r, dict) and (_f(r.get("arrived_unix")) or 0) > now - RATE_S]
    return {
        "kind": "sft", "P": p, "M": m_size, "N": n, "H": _int(meta.get("sync_interval")), "H_exact": True,
        "total_steps": _int(meta.get("total_steps")),
        "quorum": sy["quorum"], "versions": sy["versions"], "last_merge_ts": sy["last_merge_ts"], "behind": False,
        "stuck": stuck_fragments(sy, running=running, behind=False, any_merge=bool(merges), now=now),
        "period_s": sy["period_s"], "merge_gap_s": sy["merge_gap_s"], "quorum_p50_s": sy["quorum_p50_s"],
        "roster": len(live), "expected": expected, "shards": shard_rows(p, n, merges, rounds, sy["committed"], done),
        "rounds": rounds, "last_merge_ts_any": last_any, "running": running, "done": done,
        "traffic": {"push": len(arrived5), "push_bytes": None,
                    "bcast": sum(1 for i in ids for a in _applies(isls[i]) if a[0] > now - RATE_S)},
        "trainers": learners, "flows": flows[-MAX_FLOWS:],
    }
