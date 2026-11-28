"""Live topology of a run with a syncer: trainers <-> syncer, the fragment matrix and the
in-island rollout (decoupled DiLoCo RL, synced groups).

Like ``topology.py`` a read-only view over what the reducer already folded; it adds no
event and no request. Which drawing the page uses follows from the run (``kind``):

* ``async``: no syncer (no merge record, no fragment-version vector on a trainer):
  ``topology.py`` alone, the V1 drawing;
* ``synced_group``: a syncer and cross-island sampler islands: samplers -> store ->
  trainers <-> syncer;
* ``island``: a syncer and no sampler island: every island runs its own rollout
  (colocated or fixed-partition), trainers <-> syncer;
* ``sft``: decoupled DiLoCo SFT, learner islands (no rollout) <-> syncer (``topology_sft``);
  ``none``: an SFT run without a syncer, no drawing.

Sources, all of them on the tapes today:

* syncer: the merge records of ``yeto-tape.jsonl`` and ``yeto-tape.shard<i>.jsonl``
  (``step``, ``fragment``, ``attempt``, ``expected``, ``quorum_ms``,
  ``sync/merge_seconds``, ``checkpoint_ms``, ``responders[].arrived_unix /
  recv_seconds``); shard ``i`` of N owns fragments ``[i*P//N, (i+1)*P//N)`` and steps
  ``t = i+1 (mod N)`` (docs/PROTOCOL.md);
* trainers: ``rl_local_round`` (decoupled) or ``rl_full_parameter_round`` (full
  parameter): local step, fragment-version vector, policy hash, applied lr;
  ``rl_fragment_push`` / ``rl_fragment_bcast`` / ``rl_policy_apply``; ``rl_strict_failure``;
* in-island rollout: ``rl_driver_start.placement / execution_mode``,
  ``rl_driver_phase``, ``rl_local_round`` rollout stats, ``rl_policy_snapshot``,
  ``rl_publication`` (its members name the rollout engines).

Known gaps (no record on the tapes yet): a round whose PULL went out but got no PUSH yet is
invisible; a BCAST is seen only once applied, so "pending" means merged on the syncer
and not applied by the island (on the wire or queued for the safe boundary); T, tau
and H are not on the tapes (H is bounded by the smallest ``realized_h`` of a PUSH);
the trainer / rollout GPU split of a fixed-partition island is not on the tapes.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any

from . import fleet_islands
from .topology import DRIVER_PHASES, RATE_S, WINDOW_S, _engine, arch_of

# shard i > 0 of the syncer: its local tape yeto-tape.shard<i>.jsonl, or its store upload
# tapes/syncer-shard<i>/<incarnation>/ (source "store:syncer-shard<i>/<incarnation>")
SHARD_TAPE = re.compile(r"\.shard(\d+)\.jsonl$|(?:^|[:/])syncer-shard(\d+)/")
VECTOR_KEYS = ("rl/fragment_versions", "sync/fragment_versions")
MOVE_QUIET_S = 1800.0  # an island moving / relaunching quiets the stuck and blocked alerts this long at most
ROUNDS_STALE_S = 1800.0  # no trainer up and none wrote for this long: its open rounds are stale
ROUND_OPEN_WARN_S = 120.0  # a round open longer than max(this, 3 x median round wall time): waiting for quorum
STUCK_MIN_S = 240.0  # a fragment not merged for max(this, 3 x the fragment period): not advancing
IDLE_MIN_S = 180.0  # no merge for max(this, 4 x the recent merge interval): syncer idle
MERGE_HISTORY = 12  # recent merges that set the merge interval
MAX_FLOWS = 3000


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _vector(rec: dict) -> list[int] | None:
    for k in VECTOR_KEYS:
        v = rec.get(k)
        if isinstance(v, list) and v and all(_int(x) is not None for x in v):
            return list(v)
    return None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def shard_count(reducer) -> int:
    """Shards of the syncer: one tape per shard, ``yeto-tape.shard<i>.jsonl`` (or its store
    upload ``syncer-shard<i>``) for i > 0."""
    idx = [int(m.group(1) or m.group(2)) for s in reducer.sources_seen for m in [SHARD_TAPE.search(s)] if m]
    return max(idx) + 1 if idx else 1


def _outer_sync(reducer) -> dict[str, bool]:
    """Each trainer's own word on outer sync: ``rl_engine_selected.rl/outer_sync``, written when
    the engine is chosen, before the first round and before any syncer record."""
    out = {}
    for e in reducer.events:
        if e.get("type") == "rl_engine_selected" and e.get("island") is not None:
            v = (e.get("record") or {}).get("rl/outer_sync")
            if isinstance(v, bool):
                out[e["island"]] = v
    return out


def layout(reducer) -> dict:
    """Which drawing the run gets and why. Evidence, strongest first: the trainers' declared
    outer sync (``rl_engine_selected.rl/outer_sync``; the run manifest records no sync preset),
    then the tape kinds (syncer merge records, fragment-version vectors on trainer events).
    Sampler islands (``rl_sampler_*`` events) split a syncer run into ``synced_group`` /
    ``island``. Once the trainers declared it, a late (or stray) syncer tape changes nothing:
    the drawing does not flip during a run. A run with no RL trainer island and records only
    an SFT island writes is ``sft`` with a syncer (``topology_sft``), else ``none``; the shared
    syncer tape and syncer.log never count as SFT records, so an RL run is never ``sft``."""
    declared = _outer_sync(reducer)
    roles = reducer.async_.roles
    samplers = sum(1 for i in reducer.islands if roles.get(i) == "sampler")
    merges = sum(1 for m in reducer.merges if "fragment" in m)
    if not any(roles.get(i) == "trainer" for i in reducer.islands):
        if reducer.sft.sft_records:  # records only an SFT island writes (ms-swift logs, yeto plugin steps)
            n = len(reducer.sft.islands)
            if merges or reducer.sft.syncer_meta or reducer.sft.outer:
                return {"kind": "sft", "reason": f"SFT islands {n} (ms-swift logs), no RL trainer island, "
                                                 f"{merges} syncer merge record(s)"}
            return {"kind": "none", "reason": f"SFT islands {n}, no syncer: nothing to draw"}
        # a syncer tape alone, or a run whose trainers have not written yet: nothing to draw
        return {"kind": "async", "reason": f"no RL trainer island, sampler islands {samplers}"}
    if declared:
        syncer = any(declared.values())
        why = (f"trainer(s) {', '.join(sorted(i for i, v in declared.items() if v))} declare outer sync"
               if syncer else "trainers declare no outer sync (rl_engine_selected)")
    elif merges:
        syncer, why = True, f"{merges} syncer merge record(s)"
    elif any(e.get("type") in ("rl_local_round", "rl_full_parameter_round", "rl_policy_snapshot")
             and _vector(e.get("record") or {}) is not None for e in reducer.events):
        syncer, why = True, "trainer events carry a fragment-version vector"
    else:
        syncer, why = False, "no syncer tape"
    if not syncer:
        return {"kind": "async", "reason": f"{why}, sampler islands {samplers}"}
    plan = getattr(reducer, "fleet_plan", None)
    if plan:  # a fleet of async islands: drawn as one before its samplers write
        miles = sum(len(isl.get("miles_samplers") or ()) for isl in plan["islands"])
        return {"kind": "synced_group", "reason": f"{why}, fleet plan: {len(plan['islands'])} island(s) with "
                                                  f"external sampler pools"
                                                  + (f" and {miles} Miles sampler island(s)" if miles else "")
                                                  + f", {samplers} sampler(s) on the tapes"}
    k = "synced_group" if samplers else "island"
    return {"kind": k, "reason": f"{why}, " + (f"sampler islands {samplers}" if samplers
                                               else "no sampler island: rollout runs inside the islands")}


def kind(reducer) -> str:
    """``async`` / ``synced_group`` / ``island`` / ``sft`` / ``none`` (module docstring, ``layout``)."""
    return layout(reducer)["kind"]


class _Island:
    """What one trainer island's tape says about its DiLoCo state."""

    def __init__(self) -> None:
        self.placement = self.exec_mode = None
        self.rounds: list[dict] = []  # rl_local_round
        self.full: dict | None = None  # last rl_full_parameter_round
        self.snapshot: dict | None = None
        self.publication: dict | None = None
        self.pushes: list[dict] = []
        self.bcasts: list[dict] = []
        self.apply: dict | None = None
        self.failure: dict | None = None
        self.phases: list[tuple[float, str]] = []
        self.trained_groups = 0
        self.truncated: list[float] = []
        self.readies: list[float] = []  # fleet island_ready: one per machine (re)launch
        self.losses: list[dict] = []
        self.walls: list[dict] = []  # rl_round_wall: the in-island rollout's time per round
        self.trained: list[tuple[float, int]] = []  # (time, groups) of each rl_round_trained
        self.full_pushes = 0  # full parameter: PUSHes over the run (sync/push_count per round)
        self.terminal = False  # full parameter: a terminal round (the run is finalizing)

    def feed(self, ev: str, rec: dict, ts: float) -> None:
        if ev in ("rl_driver_start", "rl_engine_selected", "rl_round_wall") and rec.get("placement"):
            self.placement = rec["placement"]
            self.exec_mode = rec.get("execution_mode") or self.exec_mode
        if ev == "rl_round_wall":
            self.walls.append(dict(rec, _ts=ts))
        elif ev == "rl_local_round":
            self.rounds.append(dict(rec, _ts=ts))
        elif ev == "rl_full_parameter_round":
            self.full = dict(rec, _ts=ts)
            self.full_pushes += _int(rec.get("sync/push_count")) or 0
            self.terminal = self.terminal or rec.get("terminal") is True
        elif ev == "rl_policy_snapshot":
            self.snapshot = dict(rec, _ts=ts)
        elif ev == "rl_publication":
            self.publication = dict(rec, _ts=ts)
        elif ev == "rl_fragment_push":
            self.pushes.append(dict(rec, _ts=ts))
        elif ev == "rl_fragment_bcast":
            self.bcasts.append(dict(rec, _ts=ts))
        elif ev == "rl_policy_apply":
            self.apply = dict(rec, _ts=ts)
        elif ev == "rl_strict_failure":
            self.failure = {"ts": ts, "metric": rec.get("metric"), "error": str(rec.get("error") or "")[:200]}
        elif ev == "rl_driver_phase" and rec.get("phase"):
            self.phases.append((ts, str(rec["phase"])))
        elif ev == "island_ready":
            self.readies.append(ts)
        elif ev == "island_lost":
            self.losses.append({"ts": ts, "reason": rec.get("reason")})
        elif ev == "rl_round_trained":
            self.trained_groups += _int(rec.get("trained_groups")) or 0
            self.trained.append((ts, _int(rec.get("trained_groups")) or 0))
            if _f(rec.get("truncated_frac")) is not None:
                self.truncated.append(_f(rec["truncated_frac"]))

    def vector(self) -> tuple[list[int] | None, float | None]:
        """The newest fragment-version vector and its time."""
        best = None
        for rec in [self.snapshot, self.full] + self.rounds[-1:]:
            v = _vector(rec or {})
            if v is not None and (best is None or rec["_ts"] >= best[1]):
                best = (v, rec["_ts"])
        return best or (None, None)


def _launch_rank(reducer, iid: str):
    """For an island with more than one trainer launch (a move, a restart): a sort key of its
    events by the launch they belong to, in the order the launches started (the stamp in
    their tape folder names, else their first record; a tie: by name); an event of no launch
    (a launcher-side record) goes with the newest launch started by its time. None for an
    island with one launch or none: its events keep the order they were read in."""
    from .reducer import _store_launch  # noqa: PLC0415
    from .store_tapes import launch_started  # noqa: PLC0415

    launches = (reducer.islands.get(iid) or {}).get("launches") or {}
    if len(launches) < 2:
        return None
    start = {}
    for name, row in launches.items():
        t = launch_started(name)
        start[name] = (t if t is not None else (row.get("first_ts") or 0.0), name)
    ordered = sorted(start.values())

    def rank(e: dict) -> tuple:
        launch = _store_launch(str(e.get("source") or ""))
        if launch in start:
            return start[launch]
        ts = e.get("ts") or 0.0
        return max((s for s in ordered if s[0] <= ts), default=ordered[0])
    return rank


def _fold(reducer, ids: set[str], now: float) -> dict[str, _Island]:
    out = {i: _Island() for i in ids}
    per: dict[str, list[dict]] = {i: [] for i in ids}
    for e in reducer.events:
        ts, iid = e.get("ts"), e.get("island")
        if ts is None or ts > now or iid not in out:
            continue
        per[iid].append(e)
    for iid, evs in per.items():
        # a moved or restarted trainer: its launches' records in the order the launches ran,
        # whatever order their tapes were read in (polls, folder names), so the newest
        # launch's step, outer sweep and fragment versions are the island's (an older
        # launch's late tail never overwrites them)
        rank = _launch_rank(reducer, iid)
        for e in sorted(evs, key=rank) if rank is not None else evs:
            out[iid].feed(e.get("type"), e.get("record") or {}, e["ts"])
    for isl in out.values():
        isl.phases.sort()
    return out


def merge_rows(reducer, now: float, n: int) -> list[dict]:
    """The syncer's merge records up to ``now``, by step, with the shard that merged each."""
    rows = []
    for m in reducer.merges:
        step, frag, ts = _int(m.get("step")), _int(m.get("fragment")), _f(m.get("time_unix"))
        if step is None or frag is None or (ts is not None and ts > now):
            continue
        rows.append({"step": step, "frag": frag, "ts": ts, "shard": (step - 1) % n, "rec": m})
    rows.sort(key=lambda r: r["step"])
    return rows


def _members(rec: dict) -> list[str]:
    return [str(x) for x in rec.get("expected") or []]


def current(reducer, cards: list[dict], now: float) -> dict | None:
    """``view`` of a run with a syncer, else None; computed once per reducer state and time
    (the alert rules and the topology panel of one page view share it)."""
    key = (reducer.event_seq, len(reducer.merges), now, getattr(reducer, "fleet_plan_seq", 0))
    hit = getattr(reducer, "_diloco_view", None)
    if hit is not None and hit[0] == key:
        return hit[1]
    lay = layout(reducer)
    if lay["kind"] == "sft":
        from . import topology_sft  # imports this module's helpers

        out = topology_sft.view(reducer, now)
    else:
        out = view(reducer, cards, now, lay["kind"]) if lay["kind"] in ("island", "synced_group") else None
    if out is not None:
        out["layout_reason"] = lay["reason"]
    reducer._diloco_view = (key, out)
    return out


def syncer_state(merges: list[dict], p: int | None, n: int) -> dict:
    """What the merge records alone say: each fragment's committed version (the global step of
    its last merge) and time, each shard's committed step, the roster the merges expected (the
    widest: a later merge without a lost island expects fewer) and the merge cadence (the
    recent interval, and each fragment's period: the time between its merges)."""
    versions, last_ts = [0] * (p or 0), [None] * (p or 0)
    for m in merges:
        if m["frag"] < (p or 0):
            versions[m["frag"]] = max(versions[m["frag"]], m["step"])
            last_ts[m["frag"]] = m["ts"]
    mts = [m["ts"] for m in merges if m["ts"] is not None]
    gaps = [b - a for a, b in zip(mts[-MERGE_HISTORY - 1:], mts[-MERGE_HISTORY:])]
    merge_gap = _median(gaps)
    prev: dict[int, float] = {}
    fgaps: list[float] = []
    for m in merges:
        if m["ts"] is None:
            continue
        if m["frag"] in prev:
            fgaps.append(m["ts"] - prev[m["frag"]])
        prev[m["frag"]] = m["ts"]
    walls = [_f(m["rec"].get("ms")) / 1000 for m in merges[-MERGE_HISTORY:] if _f(m["rec"].get("ms")) is not None]
    qwait = [_f(m["rec"].get("quorum_ms")) / 1000 for m in merges[-MERGE_HISTORY:] if _f(m["rec"].get("quorum_ms")) is not None]
    return {
        "versions": versions, "last_merge_ts": last_ts,
        "committed": {i: max((m["step"] for m in merges if m["shard"] == i), default=0) for i in range(n)} if p else {},
        "merged_steps": {m["step"] for m in merges},
        "expected": max((_members(m["rec"]) for m in merges), key=len, default=[]),
        "merge_gap_s": merge_gap, "period_s": _median(fgaps[-2 * (p or 1):]) or ((merge_gap or 0) * (p or 0) or None),
        "open_limit_s": max(ROUND_OPEN_WARN_S, 3 * (_median(walls) or 0)), "quorum_p50_s": _median(qwait),
        "quorum": _int(merges[-1]["rec"].get("quorum")) if merges else None,
        "last_merge_ts_any": mts[-1] if mts else None,
    }


def syncer_records(reducer) -> dict:
    """The syncer's own records (P1 tapes; empty on older tapes): its config per shard, the
    attempts it opened (PULL sent) per global step, its quorum timeouts per step, the attempts
    it relaunched at once because every island of the round left (``members_left``, not a
    timeout) and each island's last connection record (by learner id)."""
    configs: dict[int, dict] = {}
    opens: dict[int, dict] = {}
    timeouts: dict[int, list[dict]] = {}
    relaunched: dict[int, list[dict]] = {}
    members: dict[str, dict] = {}
    for r in getattr(reducer, "syncer_records", ()):
        k = r.get("kind")
        if k == "syncer_config":
            configs[_int(r.get("shard_index")) or 0] = r
        elif k == "round_open" and _int(r.get("global_step")) is not None:
            step = _int(r["global_step"])
            if (_int(r.get("attempt")) or 1) >= (_int((opens.get(step) or {}).get("attempt")) or 0):
                opens[step] = r
        elif k == "round_attempt_timeout" and _int(r.get("global_step")) is not None:
            into = relaunched if r.get("reason") == "members_left" else timeouts
            into.setdefault(_int(r["global_step"]), []).append(r)
        elif k in ("member_connected", "member_disconnected") and r.get("learner_id") is not None:
            members[str(r["learner_id"])] = r
    return {"configs": configs, "opens": opens, "timeouts": timeouts, "relaunched": relaunched, "members": members}


def config_summary(configs: dict[int, dict]) -> dict | None:
    """``syncer_config`` as the panel shows it (shard 0's, or the first one on the tapes)."""
    if not configs:
        return None
    c = configs.get(0) or configs[min(configs)]
    return {"P": _int(c.get("fragments")), "tau": _int(c.get("pipeline")), "quorum": _int(c.get("quorum")),
            "learners": _int(c.get("learners")), "T": _int(c.get("total_steps")), "last_step": _int(c.get("last_step")),
            "lr": _f(c.get("outer_lr")), "momentum": _f(c.get("outer_momentum")),
            "H": _f(c.get("sync_interval_steps")) or None, "quorum_timeout_s": _f(c.get("quorum_timeout_s")),
            "shards": _int(c.get("shard_count")), "resumed": bool(c.get("resumed")),
            # the step the syncer (re)started at: past 0 after a resume (paces the finish)
            "step0": _int(c.get("global_step")), "ts": c.get("_ts")}


def _replacing(card: dict | None) -> bool:
    """A resident ssh: island that lost a node: waiting for a spare node or starting on one."""
    sp = (card or {}).get("spares") or {}
    return bool(sp.get("waiting") or sp.get("lost"))


def _overdue(card: dict | None) -> bool:
    """A wait for a spare node the page does not believe (past the longest wait, or unreported)."""
    return ((((card or {}).get("spares") or {}).get("wait_state") or {}).get("state")) in ("expired", "frozen")


def _quiet_move(inow: dict | None, now: float) -> bool:
    """Whether a trainer island's record says it is moving or relaunching, for no longer than
    MOVE_QUIET_S (one stuck in ordering or starting longer than that raises the stuck and
    blocked alerts again); one with no time on its record never quiets them."""
    if not inow or inow.get("state") not in ("moving", "starting"):
        return False
    since = inow.get("state_since")
    return since is not None and now - float(since) <= MOVE_QUIET_S


def stuck_fragments(sy: dict, *, running: bool, behind: bool, any_merge: bool, now: float) -> list[int]:
    """Fragments with no merge for max(STUCK_MIN_S, 3 x the fragment period) while the run runs."""
    return [f for f, ts in enumerate(sy["last_merge_ts"]) if running and any_merge and not behind and ts is not None
            and now - ts > max(STUCK_MIN_S, 3 * (sy["period_s"] or 0))]


def shard_rows(p: int | None, n: int, merges: list[dict], rounds: list[dict], committed: dict[int, int],
               done: bool) -> list[dict]:
    """One row per syncer shard: its fragments, committed step, merges, state and last merge."""
    out = []
    for i in range(n if p else 0):
        mine = [m for m in merges if m["shard"] == i]
        own = [r for r in rounds if r["shard"] == i]
        state = ("done" if done else "stale" if own and all(r.get("ended") for r in own)
                 else "waiting for quorum" if any(r["blocked"] for r in own)
                 else "merging" if any(r["state"] == "merging" for r in own) else "gathering" if own else "idle")
        out.append({"index": i, "lo": i * p // n, "hi": (i + 1) * p // n, "step": committed[i], "merges": len(mine),
                    "state": state, "attempt_max": max((_int(r["attempt"]) or 1 for r in own), default=1),
                    "last": _merge_row(mine[-1]) if mine else None})
    return out


def push_flows(merges: list[dict], lo: float, push_bytes: dict) -> list[dict]:
    """PUSH windows: each responder's wire window on the merge record (arrived - recv_seconds)."""
    out = []
    for m in merges:
        if m["ts"] is None or m["ts"] < lo - 120:
            continue
        for r in m["rec"].get("responders") or []:
            if not isinstance(r, dict):
                continue
            arr, rs = _f(r.get("arrived_unix")), _f(r.get("recv_seconds"))
            if arr is None or arr < lo:
                continue
            iid = str(r.get("id"))
            out.append({"k": "p", "i": iid, "sh": m["shard"], "f": m["frag"], "t0": arr - (rs or 0.5), "t1": arr,
                        "b": push_bytes.get((iid, m["step"]))})
    return out


def view(reducer, cards: list[dict], now: float, kind_: str) -> dict:
    """The ``diloco`` part of the topology view (trainers, syncer shards, fragments, flows)."""
    states = reducer.island_states(now)
    roles = reducer.async_.roles
    trainer_ids = sorted((i for i in reducer.islands if roles.get(i) == "trainer"
                          or (roles.get(i) != "sampler" and (states.get(i) or {}).get("role") == "trainer")),
                         key=lambda i: (0, int(i)) if i.isdigit() else (1, i))
    by_id = {c["id"]: c for c in cards}
    isl = _fold(reducer, set(trainer_ids), now)
    n = shard_count(reducer)
    merges = merge_rows(reducer, now, n)
    srec = syncer_records(reducer)
    config = config_summary(srec["configs"])
    p = max([len(v) for v, _ in (x.vector() for x in isl.values()) if v] + [m["frag"] + 1 for m in merges]
            + [(config or {}).get("P") or 0]) or None
    sy = syncer_state(merges, p, n)
    if sy["quorum"] is None and config:
        sy["quorum"] = config["quorum"]
    versions, committed, merged_steps, expected = sy["versions"], sy["committed"], sy["merged_steps"], sy["expected"]
    m_size = len(expected) or len(trainer_ids)
    live = [i for i in trainer_ids if (states.get(i) or {}).get("alive")]
    # a resident ssh: island that lost a node (stopped, waiting for a spare or starting on one) is out of the
    # roster: the syncer merges without it until it is back
    roster = [i for i in live if not _replacing(by_id.get(i))]
    # trainers out of it only while a spare takes a lost node's place (a warning, not a loss)
    roster_spare = sum(1 for i in trainer_ids if _replacing(by_id.get(i)) and not _overdue(by_id.get(i)))

    # rounds in flight: PUSHes of steps the owner shard has not committed yet (a PULL without a
    # PUSH is not on the tapes)
    inflight: dict[int, dict] = {}
    for iid, x in isl.items():
        for ps in x.pushes:
            step = _int(ps.get("global_step"))
            if step is None or step in merged_steps or step <= committed.get((step - 1) % n, 0):
                continue
            r = inflight.setdefault(step, {"step": step, "frag": _int(ps.get("fragment_id")), "shard": (step - 1) % n,
                                           "since": ps["_ts"], "attempt": 1, "pushed": []})
            r["since"] = min(r["since"], ps["_ts"])
            r["attempt"] = max(r["attempt"], _int(ps.get("round_attempt")) or 1)
            r["pushed"].append(iid)
    # the syncer's round_open: a PULL sent and not merged yet, also before any PUSH ("PULL out")
    for step, op in srec["opens"].items():
        if step in merged_steps or step <= committed.get((step - 1) % n, 0) or op.get("_ts") is None:
            continue
        r = inflight.setdefault(step, {"step": step, "frag": _int(op.get("fragment")), "shard": (step - 1) % n,
                                       "since": op["_ts"], "attempt": 1, "pushed": []})
        r["since"] = min(r["since"], op["_ts"])
        r["attempt"] = max(r["attempt"], _int(op.get("attempt")) or 1)
        r["opened_ts"] = op["_ts"]
        r["pulled"] = sorted(str(x.get("id")) for x in op.get("members") or [] if isinstance(x, dict))
    rounds = []
    for r in sorted(inflight.values(), key=lambda r: r["step"]):
        r["pushed"] = sorted(set(r["pushed"]))
        r["age_s"] = now - r["since"]
        r["state"] = "merging" if len(r["pushed"]) >= m_size else "gathering" if r["pushed"] else "pull"
        r["timeouts"] = len(srec["timeouts"].get(r["step"], ()))
        r["members_left"] = bool(srec["relaunched"].get(r["step"]))
        r["blocked"] = r["state"] != "merging" and r["age_s"] > sy["open_limit_s"]
        r["missing"] = [i for i in (expected or trainer_ids) if i not in r["pushed"]]
        rounds.append(r)

    # a terminal full-parameter round: the islands wait for each other and the syncer for its
    # final cut (terminal grace), no fragment is expected to advance
    running = (bool(live) and not all(by_id.get(i, {}).get("finalized") for i in trainer_ids)
               and not any(x.terminal for x in isl.values()))
    trainers = [_trainer(iid, isl[iid], by_id.get(iid) or {}, states.get(iid) or {}, merges, versions, committed,
                         merged_steps, p, n, now, kind_) for iid in trainer_ids]
    # a trainer anchored past the syncer's committed version: the syncer tape lags (it is copied
    # from the syncer's host); no negative lag, and no "not advancing" from a stale copy
    behind = any(t["versions"] and any(a > b for a, b in zip(t["versions"], versions)) for t in trainers)
    # no syncer record here at all (a per-island fleet's syncer on its own VM, its tape not uploaded):
    # each fragment's version is at least what the islands applied (their BCASTs), drawn as such
    tape_seen = bool(merges) or bool(srec["configs"])
    inferred = None
    if not tape_seen and p:
        vs = [t["versions"] for t in trainers if t.get("versions")]
        if vs:
            inferred = [max((v[f] for v in vs if f < len(v) and v[f] is not None), default=0) for f in range(p)]
    stuck = stuck_fragments(sy, running=running, behind=behind, any_merge=bool(merges), now=now)
    if any(_quiet_move(t.get("island_now"), now) for t in trainers):
        # an island moving or relaunching (for at most MOVE_QUIET_S): no merge is expected until its
        # new launch joins, so no fragment is stuck and no round is blocked yet
        stuck = []
        for r in rounds:
            r["blocked"] = False
    for t in trainers:
        t["outer"] = t["outer"] if t["outer"] is not None else _outer(t["versions"], p)

    # the run ended: no trainer is up and one of them finished (others may have failed at the end)
    done = bool(trainer_ids) and not live and any((states.get(i) or {}).get("status") in ("done", "stopped")
                                                  for i in trainer_ids)
    if done:  # PULLs nobody answered before the end are not rounds in flight
        rounds = [r for r in rounds if r["pushed"]]
    if inferred is not None:  # no syncer tape: a PUSH of a step an island already applied was merged
        rounds = [r for r in rounds if r["frag"] is None or r["frag"] >= len(inferred) or r["step"] > inferred[r["frag"]]]
    ends = [(reducer.islands.get(i) or {}).get("last_event_ts") for i in trainer_ids]
    end = max((t for t in ends if t is not None), default=now)
    # the run ended, or no trainer has written for ROUNDS_STALE_S (a restart of every island
    # shorter than that keeps its rounds in flight and their quorum warning): a round still
    # open is not in flight; its clock stops at the trainers' last event
    if done or (not live and now - end > max(ROUNDS_STALE_S, sy["open_limit_s"])):
        for r in rounds:
            r.update(ended="run ended" if done else "no trainer up", ended_at=end,
                     age_s=max(0.0, min(end, now) - r["since"]), blocked=False)
    for t in trainers:  # the syncer's word on the island's connection (P1 tapes)
        mrec = srec["members"].get(t["id"])
        t["syncer_link"] = {"state": "disconnected" if mrec["kind"] == "member_disconnected" else "connected",
                            "since": mrec.get("_ts"), "generation": _int(mrec.get("generation")),
                            "phase": mrec.get("phase")} if mrec else None
    pushes5 = [ps for x in isl.values() for ps in x.pushes if ps["_ts"] > now - RATE_S]
    bcasts5 = [b for x in isl.values() for b in x.bcasts if b["_ts"] > now - RATE_S]
    hs = [_int(ps.get("realized_h")) for x in isl.values() for ps in x.pushes if _int(ps.get("realized_h"))]
    return {
        "kind": kind_, "P": p, "M": m_size, "N": n, "H": min(hs) if hs else (config or {}).get("H"),
        "config": config,
        "quorum": sy["quorum"], "versions": versions, "last_merge_ts": sy["last_merge_ts"], "behind": behind,
        "syncer_tape": tape_seen, "inferred_versions": inferred,
        "stuck": stuck, "period_s": sy["period_s"], "merge_gap_s": sy["merge_gap_s"], "quorum_p50_s": sy["quorum_p50_s"],
        "roster": len(roster), **({"roster_spare": roster_spare} if roster_spare else {}), "expected": expected, "shards": shard_rows(p, n, merges, rounds, committed, done),
        "rounds": rounds, "last_merge_ts_any": sy["last_merge_ts_any"], "running": running, "done": done,
        "traffic": {"push": len(pushes5), "push_bytes": sum(_push_bytes(x) or 0 for x in pushes5),
                    "bcast": len(bcasts5)},
        "trainers": trainers,
        "flows": _flows(isl, merges, now, {t["id"] for t in trainers if t["island"]}),
        "foreign_tape": _foreign_tape(reducer) if not merges else None,
        **(_samplers(reducer, trainer_ids, by_id, states) if kind_ == "synced_group" else {}),
    }


def _foreign_tape(reducer) -> dict | None:
    """The host-wide syncer tape's records from before this run started (``sources.HOST_TAPE_SINCE``):
    another run's syncer, not drawn. None when it had none."""
    tapes = [t for t in (getattr(reducer, "foreign_tapes", None) or {}).values() if t.get("records")]
    if not tapes:
        return None
    last = [t["last_ts"] for t in tapes if t.get("last_ts") is not None]
    return {"records": sum(t["records"] for t in tapes), "last_ts": max(last) if last else None,
            "since": min(t["since"] for t in tapes if t.get("since") is not None)}


SUBSTORE = re.compile(r"/trainer(\d+)/?$")


def _samplers(reducer, trainer_ids: list[str], by_id: dict, states: dict) -> dict:
    """The sampler side of a synced group. With ``--rl-sampler-follow trainer`` every trainer
    and its samplers share the store sub-directory ``trainer<k>`` (``sample_roles.trainer_substore``,
    k = the trainer's learner id): a sampler's ``rl_sampler_start.store`` ends in it, and each
    pair is drawn as one island with its own lane of the store. A fleet of async islands
    (``fleet_islands``) binds external pools the same way, by ``rl_sampler_start.island`` or its
    plan first. Otherwise all samplers share one store and load the publisher's releases
    (``rl_policy_release_role.publisher``)."""
    asy = reducer.async_
    store: dict[str, str] = {}
    publisher = None
    for e in reducer.events:
        rec, iid = e.get("record") or {}, e.get("island")
        if e.get("type") == "rl_sampler_start" and iid is not None and rec.get("store"):
            store[iid] = str(rec["store"])
        elif e.get("type") == "rl_policy_release_role" and rec.get("publisher") is True and iid is not None:
            publisher = iid
    sids = sorted((i for i in reducer.islands if asy.roles.get(i) == "sampler"),
                  key=lambda i: (0, int(i)) if i.isdigit() else (1, i))
    fleet = fleet_islands.resolve(reducer, trainer_ids)
    learners = fleet_islands.learner_trainers(trainer_ids, getattr(reducer, "fleet_plan", None))
    follows = {}
    for sid in sids:
        if fleet is not None:
            k = fleet["island_of"].get(sid)
        else:  # the sub-directory names the learner id, not the trainer's island id
            m = SUBSTORE.search(store.get(sid, ""))
            k = int(m.group(1)) if m else None
        if k is not None and learners.get(k) is not None:
            follows[sid] = learners[k]
    follow = bool(sids) and len(follows) == len(sids)
    if fleet is not None:  # a pool whose trainer has not written yet does not unbind the others
        follow = True
    if follow:
        publisher = None  # every trainer publishes to its own lane (its release origin counts within the lane)
    lanes = {}
    if follow:
        for tid in trainer_ids:
            mine = [s for s, t in follows.items() if t == tid]
            written = sum(_f(r.get("groups")) or 0 for s in mine for r in (asy.sampler.get(s) or {}).get("rounds", {}).values())
            rel = [x for x in asy.releases if x.get("trainer") == tid and _int(x.get("policy_version")) is not None]
            read = max((_f(x.get("buffer_selected")) for x in rel if _f(x.get("buffer_selected")) is not None), default=None)
            k = next((k for k, t in learners.items() if t == tid), None)
            lanes[tid] = {"substore": f"trainer{tid if k is None else k}", "samplers": mine, "written": written,
                          "read": read,
                          "backlog": max(0.0, written - read) if read is not None else None,
                          "latest": max((_int(x["policy_version"]) for x in rel), default=None)}
    nodes = []
    for sid in sids:
        c = by_id.get(sid) or {}
        nodes.append({"id": sid, "trainer": follows.get(sid), "gpu": c.get("gpu"), "gpus": _int(c.get("gpus")),
                      "cloud": c.get("cloud"), "machine_full": c.get("name"), "live": bool((states.get(sid) or {}).get("alive")),
                      "status": (states.get(sid) or {}).get("status")})
        if fleet is not None:
            nodes[-1].update(island=fleet["island_of"].get(sid), pool=fleet["pool_of"].get(sid))
    out = {"follow": follow, "publisher": publisher, "lanes": lanes, "sampler_nodes": nodes}
    if fleet is not None:
        out["fleet"] = _fleet_part(fleet, asy)
    return out


def _pool_machines(p: dict) -> dict:
    """What a fleet pool's machines are, from its pool state (``members``, as run up bills them):
    ``released`` when the pool ended (stopped or removed) and every machine is gone (billed no
    more, as the cost panel counts it); a pool still running that lost a machine reads
    ``replacing`` while none is held yet, ``waiting for capacity`` while its replacement is
    ordered but not granted. Nothing without members (older stores) or for a pool that never
    lost one (its samplers' tapes say how it starts)."""
    members = p.get("members") or []
    if not members:
        return {}
    held = [m for m in members if m.get("state") != fleet_islands.GONE]
    if p.get("stopped") or p.get("removed"):
        return {} if held else {"released": True}
    if len(held) == len(members):
        return {}
    if not held:
        return {"machines": "replacing"}
    return {"machines": "waiting for capacity"} if all(m.get("granted") is None for m in held) else {}


def _fleet_part(fleet: dict, asy) -> dict:
    """The islands and pools the page draws in each island box: the pool's engine and
    rollout precision (plan, else the samplers' ``rl_sampler_start.provenance``), its provider,
    card and the samplers it runs. A Miles sampler island of the launch is a row too
    (``source: miles``, its launch island and shard; the external pools then say
    ``source: external``)."""
    islands = []
    for isl in fleet["islands"]:
        pools = []
        for p in isl["pools"]:
            prov = next(((asy.sampler.get(sid) or {}).get("provenance") for sid in p["samplers"]
                         if (asy.sampler.get(sid) or {}).get("provenance")), None) or {}
            quant = p.get("rollout_quant") or prov.get("precision")
            card = prov.get("card_model") or p.get("card")  # a Miles row: its --gpu cluster's card
            row = {"name": p["name"], "source_name": p.get("source_name"), "provider": p.get("provider"),
                   "engine": _engine(p.get("engine") or prov.get("engine")),
                   "precision": str(quant).upper() if quant else None,
                   "card": card, "arch": arch_of(card),
                   "topology": prov.get("topology"),
                   "samplers": list(p["samplers"])}
            if p.get("removed"):  # fleet apply removed the pool: drawn as ended
                row["removed"] = p["removed"]
            if p.get("stopped"):  # stopped (pool-stop, its island given up): drawn as ended
                row["stopped"] = p["stopped"]
            row.update(_pool_machines(p))
            if "source" in p:  # a fleet with Miles sampler islands: where each sampler row runs
                row["source"] = p["source"]
            if p.get("source") == "miles":
                row.update(launch_island=p.get("launch_island"), shard=p.get("shard"), gpus=p.get("gpus"))
            pools.append(row)
        # index: the island's place in the fleet (the plan's; bound from the events, its learner id)
        index = isl.get("index") if isl.get("index") is not None else isl["learner"]
        islands.append({"island": isl["learner"], "index": index, "trainer": isl.get("trainer"), "name": isl.get("name"),
                        "gpu": isl.get("gpu"), "substore": isl.get("substore"), "pools": pools,
                        # a fleet launched island by island: its desired-state row (gpu, moves, removal) and its
                        # record's state (fleet_islands.island_state)
                        **({"desired": isl["desired"]} if isl.get("desired") else {}),
                        **({"now": isl["now"]} if isl.get("now") else {}),
                        **({"added": True} if isl.get("added") else {})})
    return {"source": fleet["source"], "islands": islands, "unbound": list(fleet["unbound"])}


def _merge_row(m: dict) -> dict:
    rec = m["rec"]
    ms = lambda k: (_f(rec.get(k)) / 1000) if _f(rec.get(k)) is not None else None  # noqa: E731
    return {"step": m["step"], "frag": m["frag"], "ts": m["ts"], "attempt": _int(rec.get("attempt")),
            "quorum_s": ms("quorum_ms"), "merge_s": _f(rec.get("sync/merge_seconds")) or ms("sync_ms"),
            "checkpoint_s": ms("checkpoint_ms"), "wall_s": ms("ms"), "base": _int(rec.get("sync/base_version")),
            "responded": [str(x) for x in rec.get("responded") or []], "expected": _members(rec),
            "missed": [str(x) for x in rec.get("missed_grace") or []], "quorum": _int(rec.get("quorum")),
            "rejected": _int(rec.get("sync/rejected_stale_updates"))}


def _outer(vec: list[int] | None, p: int | None) -> int | None:
    """Outer version from the vector: the sweep its oldest fragment belongs to."""
    if not vec or not p:
        return None
    lo = min(vec)
    return (lo - 1) // p + 1 if lo > 0 else 0


def _trainer(iid: str, x: _Island, card: dict, st: dict, merges: list[dict], versions: list[int],
             committed: dict[int, int], merged_steps: set[int], p: int | None, n: int, now: float,
             kind_: str) -> dict:
    vec, _ = x.vector()
    last = x.rounds[-1] if x.rounds else {}
    full = x.full or {}
    snap = x.snapshot or {}
    step = _int(last.get("local_round_id")) if last else None
    if step is None and full:
        step = _int(full.get("policy_version"))
    if step is None:
        step = card.get("round")
    # local steps since each fragment's anchor (the round that applied its BCAST); decoupled only
    since = None
    if x.rounds and p:
        since = [0] * p
        for r in x.rounds:
            applied = set(_int(a) for a in r.get("sync/applied_fragments") or [])
            since = [0 if f in applied else since[f] + 1 for f in range(p)]
    # BCAST pending: merges this island is a member of that its vector has not applied yet
    pending: dict[int, list[float]] = {}
    if vec and p:
        for m in merges:
            if m["frag"] < len(vec) and m["step"] > vec[m["frag"]] and (iid in _members(m["rec"]) or not _members(m["rec"])):
                pending.setdefault(m["frag"], []).append(m["ts"] if m["ts"] is not None else now)
    pushed = {}
    for ps in x.pushes:
        s, f = _int(ps.get("global_step")), _int(ps.get("fragment_id"))
        if s is not None and f is not None and s not in merged_steps and s > committed.get((s - 1) % n, 0):
            pushed[f] = s
    lp = x.pushes[-1] if x.pushes else None
    lb = x.bcasts[-1] if x.bcasts else None
    lr = _f(last.get("applied_lr")) if last else None
    if lr is None and isinstance(last.get("applied_lrs"), list) and last["applied_lrs"]:
        lr = _f(last["applied_lrs"][0])
    ph = x.phases[-1] if x.phases else None
    # a synced group's rollout runs on its sampler islands: its trainers are buffer trainers (their
    # placement field still reads "colocated", and their rounds carry buffer read stats), never
    # drawn with an in-island rollout or a shared GPU
    roll = _rollout(x, now) if kind_ == "island" else None
    oldest = min((t for ts in pending.values() for t in ts), default=None)
    return {
        "id": iid, "name": card.get("name"), "gpu": card.get("gpu"), "gpus": card.get("gpus"),
        "backend": card.get("backend_label"), "unreported": card.get("unreported") or {},
        "arch": arch_of(card.get("gpu")), "status": st.get("status"), "live": bool(st.get("alive")),
        "reason": (card.get("left") or {}).get("reason"), "finalized": bool(card.get("finalized")),
        "island_now": card.get("island_now"),  # its trainer island record (a fleet launched island by island)
        "straggler": card.get("straggler"),  # its last island_straggler event, None once cleared
        "last_event_ts": st.get("last_event_ts"), "step": step,
        "outer": _int(full.get("sync/outer_version")) if full else None,
        "versions": vec, "since": since, "pushed": {str(k): v for k, v in pushed.items()},
        "pending": {str(f): len(ts) for f, ts in pending.items()},
        "pending_oldest_ts": oldest, "pending_n": sum(len(ts) for ts in pending.values()),
        "hash": str(snap.get("rl/policy_hash") or last.get("rl/policy_hash") or full.get("sync/global_policy_hash")
                    or "")[:8] or None,
        "lr": lr, "lr_zero": lr == 0.0,
        "phase": DRIVER_PHASES.get(ph[1], ph[1]) if ph else None, "phase_raw": ph[1] if ph else None,
        "phase_since": ph[0] if ph else None,
        "last_push": {"frag": _int(lp.get("fragment_id")), "step": _int(lp.get("global_step")),
                      "base": _int(lp.get("base_version")), "bytes": _push_bytes(lp),
                      "wire": _f(lp.get("wire_bytes")) is not None, "payload_bytes": _f(lp.get("payload_bytes")),
                      "p2p_s": _f(lp.get("pull_to_push_seconds")), "h": _int(lp.get("realized_h")),
                      "ts": lp["_ts"]} if lp else None,
        # a full-parameter island marks a BCAST of a round that closed without its PUSH
        # (partial quorum: ``overwrite``) or one it fetched to catch up (``catch_up``)
        "last_bcast": {"frag": _int(lb.get("fragment_id")), "v": _int(lb.get("version")),
                       "bytes": _f(lb.get("payload_bytes")), "queue_s": _f(lb.get("queue_seconds")),
                       "received_ts": _f(lb.get("received_unix")), "applied_ts": _f(lb.get("applied_unix")),
                       "overwrite": lb.get("overwrite") is True, "catch_up": lb.get("catch_up") is True,
                       "ts": lb["_ts"]} if lb else None,
        "bcast_overwrites": sum(1 for b in x.bcasts if b.get("overwrite") is True),
        "bcast_catch_ups": sum(1 for b in x.bcasts if b.get("catch_up") is True),
        "apply_s": _f((x.apply or {}).get("sync/apply_seconds")), "apply_ts": (x.apply or {}).get("_ts"),
        "full_push": {"count": _int(full.get("sync/push_count")), "bytes": _f(full.get("sync/upload_bytes")),
                      "inflight": _int(full.get("sync/inflight_pairs")), "total": x.full_pushes,
                      "overwritten": _int(full.get("sync/overwritten_rounds"))} if full else None,
        "failure": x.failure, "gpu_util": card.get("gpu_util_pct"), "mem_pct": card.get("mem_pct"),
        "trained_groups": x.trained_groups,
        "truncated": x.truncated[-1] if x.truncated else None,
        "island": roll,
        "nodes": _nodes(x, card, roll, now),
        # a resident ssh: island: its spare nodes and its unhealthy nodes (no key without them)
        **{k: card[k] for k in ("spares", "node_health") if k in card},
    }


def _nodes(x: _Island, card: dict, roll: dict | None, now: float) -> list[dict]:
    """One label per machine of the island: its cards and the role of each card (T trainer,
    R rollout, T+R both, colocated). The trainer / rollout split of a fixed-partition
    machine is not on the tapes: with 2 cards it can only be one each (``inferred``)."""
    gpus = _int(card.get("gpus"))
    probes = card.get("nodes") or []
    machines = max(1, len(probes))
    per = gpus // machines if gpus and gpus % machines == 0 else None
    roles, inferred = None, False
    if roll and roll["colocated"]:
        roles = ["T+R"] * (per or 1)
    elif roll and roll["placement"] and machines == 1 and per == 2:
        roles, inferred = ["T", "R"], True
    elif not roll:
        roles = ["T"] * (per or 1)  # a trainer whose rollout runs on sampler islands
    name = card.get("name") or ""
    inc = max(1, len(x.readies))
    out = []
    for k in range(machines):
        probe = probes[k] if k < len(probes) else {}
        out.append({"id": f"n{k}", "machine": name.rsplit("-", 2)[-2] + "-" + name.rsplit("-", 1)[-1]
                    if name.count("-") >= 2 else name or None, "machine_full": name or None,
                    "gpu": card.get("gpu"), "gpus": per, "roles": roles, "inferred": inferred,
                    "incarnation": inc, "util": _f(probe.get("gpu_util_pct")) if probe else _f(card.get("gpu_util_pct")),
                    "mem_pct": _f(card.get("mem_pct"))})
    last_loss = x.losses[-1] if x.losses else None
    if inc > 1 and last_loss and last_loss["ts"] is not None and now - last_loss["ts"] < 600:
        out.append({"id": "n0", "previous": True, "incarnation": inc - 1, "machine": out[0]["machine"],
                    "machine_full": name or None, "gpu": card.get("gpu"), "gpus": per, "roles": roles,
                    "inferred": inferred, "reason": last_loss["reason"], "lost_ts": last_loss["ts"]})
    return out


def _publish_s(x: _Island, at: float) -> float | None:
    """Length of the publish phase that ended with the publication at ``at``."""
    marks = x.phases
    for k in range(len(marks) - 1, -1, -1):
        if marks[k][1] == "publish" and marks[k][0] <= at + 1:
            return marks[k + 1][0] - marks[k][0] if k + 1 < len(marks) else None
    return None


def _rollout(x: _Island, now: float) -> dict:
    """The island's own rollout: placement, engines, last round, served snapshot. Decoupled
    islands report the rollout per round (``rl_local_round`` rl/*); a full-parameter island
    does not: its round and time come from ``rl_round_wall`` and its groups per round are the
    ones the trainer trained (``groups_src`` "trained"; the run total is not reported)."""
    if not x.rounds and (x.walls or x.trained):
        return _rollout_from_trainer(x, now)
    last = x.rounds[-1] if x.rounds else {}
    get = lambda k: last.get("rl/" + k, last.get(k))  # noqa: E731
    # group latency 0.0: the rollout did not time its groups (a group never takes 0 s)
    lat = lambda k: _f(get(k)) or None  # noqa: E731
    pub = x.publication or {}
    members = [str(m) for m in pub.get("sync/publication_members") or []]
    token = str(pub.get("rl/policy_token") or (x.snapshot or {}).get("rl/policy_token") or "")
    secs = [_f(r.get("rl/rollout_seconds", r.get("rollout_seconds"))) for r in x.rounds[-8:]]
    secs = [s for s in secs if s is not None]
    groups = sum(_int(r.get("rl/completed_groups", r.get("completed_groups"))) or 0 for r in x.rounds)
    zv = [_f(r.get("rl/zero_variance_group_ratio", r.get("zero_variance_group_ratio"))) for r in x.rounds[-8:]]
    zv = [z for z in zv if z is not None]
    return {
        "placement": x.placement, "exec_mode": x.exec_mode,
        "colocated": (x.placement or "").startswith("coloc") or (x.exec_mode or "").startswith("colocated"),
        "engines": sum(1 for m in members if m.startswith("engine:")) or None,
        "round": _int(get("rollout_id")), "groups": _int(get("completed_groups")),
        "traj": _int(get("completed_trajectories")), "tokens": _int(get("action_tokens")),
        "rollout_s": _f(get("rollout_seconds")), "p50_s": lat("group_p50_seconds"),
        "p95_s": lat("group_p95_seconds"), "reward": _f(get("reward_mean")),
        "zero_var": _f(get("zero_variance_group_ratio")), "zero_var_recent": _median(zv),
        "rollout_p50_s": _median(secs), "groups_total": groups, "groups_src": "rollout",
        "served": {"v": _int(pub.get("policy_version")), "hash": token.rsplit(":", 1)[-1][:8] if token else None,
                   "token": token or None, "ts": pub.get("_ts"), "bytes": _f(pub.get("sync/publication_payload_bytes")),
                   "publish_s": _publish_s(x, pub["_ts"]) if pub else None} if pub else None,
        "groups_5m": sum(_int(r.get("rl/completed_groups", r.get("completed_groups"))) or 0
                         for r in x.rounds if r["_ts"] > now - RATE_S),
    }


def _rollout_from_trainer(x: _Island, now: float) -> dict:
    pub = x.publication or {}
    token = str(pub.get("rl/policy_token") or "")
    members = [str(m) for m in pub.get("sync/publication_members") or []]
    w = x.walls[-1] if x.walls else {}
    secs = [s for s in (_f(r.get("rollout_seconds")) for r in x.walls[-8:]) if s is not None]
    return {
        "placement": x.placement, "exec_mode": x.exec_mode,
        "colocated": (x.placement or "").startswith("coloc") or (x.exec_mode or "").startswith("colocated"),
        "engines": sum(1 for m in members if m.startswith("engine:")) or None,
        "round": _int(w.get("rollout_id")), "groups": x.trained[-1][1] if x.trained else None, "traj": None,
        "tokens": None, "rollout_s": _f(w.get("rollout_seconds")), "p50_s": None, "p95_s": None, "reward": None,
        "zero_var": None, "zero_var_recent": None, "rollout_p50_s": _median(secs), "groups_total": None,
        "groups_src": "trained",
        "served": {"v": _int(pub.get("policy_version")), "hash": token.rsplit(":", 1)[-1][:8] if token else None,
                   "token": token or None, "ts": pub.get("_ts"), "bytes": _f(pub.get("sync/publication_payload_bytes")),
                   "publish_s": _publish_s(x, pub["_ts"]) if pub else None} if pub else None,
        "groups_5m": sum(g for ts, g in x.trained if ts > now - RATE_S),
    }


def _push_bytes(rec: dict) -> float | None:
    """A PUSH's bytes on the wire: ``wire_bytes`` when the island encoded it ahead (a
    full-parameter background push / prefetch, e.g. int4), else ``payload_bytes``."""
    wire = _f(rec.get("wire_bytes"))
    return wire if wire is not None else _f(rec.get("payload_bytes"))


def _flows(isl: dict[str, _Island], merges: list[dict], now: float, rollout: set[str]) -> list[dict]:
    """Timed windows the page replays: ``p`` PUSH (the responder's wire window on the merge
    record), ``b`` BCAST (merge -> received = applied record time - queue_seconds; ``a`` marks
    an approximate window when the island wrote no rl_fragment_bcast), ``q`` rollout groups
    handed to the trainer (generate -> train), ``s`` snapshot publish (publish phase)."""
    lo = now - WINDOW_S
    push_bytes = {(iid, _int(ps.get("global_step"))): _push_bytes(ps) for iid, x in isl.items() for ps in x.pushes}
    out = push_flows(merges, lo, push_bytes)
    bc = {(iid, _int(b.get("fragment_id")), _int(b.get("version"))): b for iid, x in isl.items() for b in x.bcasts}
    for m in merges:
        if m["ts"] is None or m["ts"] < lo:
            continue
        for iid in (_members(m["rec"]) or list(isl)):
            if iid not in isl:
                continue
            b = bc.get((iid, m["frag"], m["step"]))
            # P1 stamps: the syncer queued the frames at bcast_unix, the island received them at
            # received_unix; older tapes: the merge time and the apply record minus its queue time
            t0 = _f(m["rec"].get("bcast_unix")) or m["ts"]
            t1 = (_f(b.get("received_unix")) or b["_ts"] - (_f(b.get("queue_seconds")) or 0)) if b else t0 + 1.5
            out.append({"k": "b", "i": iid, "sh": m["shard"], "f": m["frag"], "t0": t0, "t1": max(t0 + 0.3, t1),
                        "b": _f(b.get("payload_bytes")) if b else _f(m["rec"].get("bcast_bytes")), "a": b is None})
    for iid, x in isl.items():
        if iid not in rollout:
            continue
        for (t0, a), (t1, _b) in zip(x.phases, x.phases[1:]):
            if t1 < lo:
                continue
            if a == "generate":  # the groups of the round that follows (its rl_local_round)
                g = next((_int(r.get("rl/completed_groups", r.get("completed_groups"))) for r in x.rounds
                          if r["_ts"] >= t1), None)
                if g is None:  # full parameter: the groups the trainer trained from it
                    g = next((n for ts, n in x.trained if ts >= t1), None)
                out.append({"k": "q", "i": iid, "t0": t1, "t1": t1 + 1.2, "m": g})
            elif a == "publish":
                out.append({"k": "s", "i": iid, "t0": t0, "t1": max(t1, t0 + 0.5)})
    out.sort(key=lambda f: f["t0"])
    return out[-MAX_FLOWS:]
