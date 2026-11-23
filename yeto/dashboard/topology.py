"""Live topology panel: samplers -> run store (R2 / S3 bucket) -> trainers.

A read-only view over what the reducer already folded from the island tapes; it adds
no event, no request and no write anywhere (the tapes are the ones the islands upload
anyway, read by ``StoreTapes`` / the tape followers):

* nodes: every sampler and trainer island, with its state from
  ``Reducer.island_states`` (status / alive: the one rule the overview cards and the
  "Islands now" table use); a
  sampler's card / precision / engine from ``rl_sampler_start.provenance``;
* the buffer: groups written minus the trainers' read position
  (``AsyncState.backlog_series``); the newest release (``rl_policy_released``);
* flows: the recent tape events the page animates, replayed at their own
  ``time_unix``: groups written (``rl_agentic_group`` with ``written``, or
  ``rl_sampler_round.groups`` for samplers without agentic groups), groups read
  (``rl_group_policies``), releases with their upload window
  (``release_timings.upload_s``), drains (``rl_agentic_drain``) and loads with
  their fetch window (``rl_sampler_policy_loaded.seen_at`` +
  ``follower_timings.resolve_s``).
"""

from __future__ import annotations

import math
import re
from typing import Any

WINDOW_S = 600.0  # flow events sent to the page: the animation replays the last 10 minutes
RATE_S = 300.0  # groups / min per sampler over the last 5 minutes
MAX_FLOWS = 4000
DRIVER_PHASES = {"generate": "waiting for data", "train": "training", "sync": "syncing", "publish": "publishing",
                 "finish": "finished"}
ENGINES = {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TRT-LLM", "trt-llm": "TRT-LLM", "tensorrt-llm": "TRT-LLM",
           "lmdeploy": "LMDeploy", "mindie": "MindIE"}
# accelerator model -> architecture (the panel colors nodes by it); first match wins
ARCHS = (
    (r"\bMI3[5-9]\d", "CDNA4"), (r"\bMI3\d\d", "CDNA3"), (r"\bMI2\d\d", "CDNA2"),
    (r"\b(GB|B)[12]\d\d\b|\bB300|\bB200|\bRTX\s*PRO\s*6000", "Blackwell"),
    (r"\b(GH|H)[12]\d\d\b|\bH20\b|\bH800", "Hopper"),
    (r"\bL40|\bL4\b|\bRTX\s*6000\s*Ada|\bRTX\s*40", "Ada"),
    (r"\bA[1-9]\d{1,2}(G|\b)|\bA800|\bRTX\s*A\d|\bRTX\s*30", "Ampere"),
    (r"910|Ascend", "Ascend"),
)


def arch_of(card: Any) -> str | None:
    text = str(card or "")
    for pat, arch in ARCHS:
        if re.search(pat, text, re.IGNORECASE):
            return arch
    return None


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _int(v: Any) -> int | None:
    x = _f(v)
    return int(x) if x is not None else None


def _engine(name: Any) -> str | None:
    if not name:
        return None
    s = str(name)
    return ENGINES.get(s.lower(), s)


def _release_kind(rec: dict) -> str:
    """``delta`` when the release shipped a delta (``delta_ratio`` set), else ``full``."""
    return "delta" if _f(rec.get("delta_ratio")) is not None else "full"


def view(reducer, cards: list[dict], now: float) -> dict | None:
    """The topology panel's data; None for a run without samplers. ``cards``: the island
    cards of the same page view (hardware, step, policy version, leave reason); status and
    liveness come from ``reducer.island_states``."""
    asy = reducer.async_
    states = reducer.island_states(now)
    roles = {iid: role for iid, role in asy.roles.items() if iid in reducer.islands}
    sampler_ids = [i for i, r in roles.items() if r == "sampler"]
    trainer_ids = [i for i, r in roles.items() if r == "trainer"]
    if not sampler_ids:
        return None
    by_id = {c["id"]: c for c in cards}
    lo, rate_lo = now - WINDOW_S, now - RATE_S
    flows: list[dict] = []
    agentic: set[str] = set()  # samplers whose groups come as rl_agentic_group
    rounds_flow: list[dict] = []
    groups_recent: dict[str, int] = {}
    last_group: dict[str, float] = {}
    phase: dict[str, dict] = {}
    last_release: dict | None = None
    drains: dict[str, float] = {}
    for e in reducer.events:
        ts, ev, iid, rec = e.get("ts"), e.get("type"), e.get("island"), e.get("record") or {}
        if ts is None or ts > now:  # a view at an earlier time (tests, replays) sees only its past
            continue
        if ev == "rl_policy_released" and (last_release is None or ts >= last_release["ts"]):
            last_release = {"ts": ts, "v": _int(rec.get("policy_version")), "bytes": _f(rec.get("release_bytes")),
                            "kind": _release_kind(rec), "delta_ratio": _f(rec.get("delta_ratio")),
                            "upload_s": _f((rec.get("release_timings") or {}).get("upload_s"))
                            if isinstance(rec.get("release_timings"), dict) else None}
        if ev == "rl_driver_phase" and iid is not None and ts >= (phase.get(iid) or {}).get("since", float("-inf")):
            phase[iid] = {"phase": rec.get("phase"), "since": ts}
        if ev == "rl_agentic_drain" and iid is not None:
            drains[iid] = max(drains.get(iid, ts), ts)
        if ev == "rl_agentic_group" and rec.get("written") and iid is not None:
            agentic.add(iid)
            last_group[iid] = max(last_group.get(iid, ts), ts)
            if ts > rate_lo:
                groups_recent[iid] = groups_recent.get(iid, 0) + 1
        if ts < lo:
            continue
        if ev == "rl_agentic_group" and rec.get("written") and iid is not None:
            flows.append({"t": ts, "k": "g", "s": iid, "n": _int(rec.get("trials")) or _int(rec.get("kept")), "m": 1})
        elif ev == "rl_sampler_round" and iid is not None and (_int(rec.get("groups")) or 0) > 0:
            rounds_flow.append({"t": ts, "k": "g", "s": iid, "n": None, "m": _int(rec.get("groups"))})
        elif ev == "rl_group_policies" and iid is not None:
            n = len(rec["groups"]) if isinstance(rec.get("groups"), list) else 0
            flows.append({"t": ts, "k": "r", "i": iid, "m": n})
        elif ev == "rl_policy_released" and iid is not None:
            up = last_release["upload_s"] if last_release and last_release["ts"] == ts else None
            flows.append({"t": ts, "k": "u", "i": iid, "v": _int(rec.get("policy_version")),
                          "b": _f(rec.get("release_bytes")), "kind": _release_kind(rec), "u0": ts - up if up else None})
        elif ev == "rl_agentic_drain" and iid is not None:
            flows.append({"t": ts, "k": "d", "s": iid, "w": _f(rec.get("waited_s"))})
        elif ev == "rl_sampler_policy_loaded" and iid is not None:
            ft = rec.get("follower_timings") if isinstance(rec.get("follower_timings"), dict) else {}
            flows.append({"t": ts, "k": "l", "s": iid, "v": _int(rec.get("policy_version")),
                          "seen": _f(rec.get("seen_at")), "rs": _f(ft.get("resolve_s")), "ls": _f(ft.get("load_s"))})
    # samplers without agentic groups: their rounds carry the group counts
    for fl in rounds_flow:
        if fl["s"] not in agentic:
            flows.append(fl)
    for iid in sampler_ids:
        if iid in agentic:
            continue
        rows = (asy.sampler.get(iid) or {}).get("rounds", {}).values()
        for r in rows:
            ts, g = r.get("ts"), _int(r.get("groups")) or 0
            if ts is None or g <= 0:
                continue
            last_group[iid] = max(last_group.get(iid, ts), ts)
            if ts > rate_lo:
                groups_recent[iid] = groups_recent.get(iid, 0) + g
    flows.sort(key=lambda x: x["t"])
    del flows[:-MAX_FLOWS]

    releases = [x for x in asy.releases if _int(x.get("policy_version")) is not None and (x.get("ts") or 0) <= now]
    latest = max((_int(x["policy_version"]) for x in releases), default=None)
    latest_rel = max(releases, key=lambda x: (x.get("ts") or 0, _int(x["policy_version"]))) if releases else None
    if last_release is None and latest_rel is not None:  # the release record left the event window
        last_release = {"ts": latest_rel.get("ts"), "v": _int(latest_rel["policy_version"]),
                        "bytes": _f(latest_rel.get("bytes")), "kind": None, "delta_ratio": None,
                        "upload_s": _f(latest_rel.get("upload_s"))}
    loaded: dict[str, int] = {}
    for ld in asy.loads:
        v = _int(ld.get("policy_version"))
        if v is not None and ld.get("loaded_at") is not None and ld["loaded_at"] <= now:
            loaded[ld["sampler"]] = max(loaded.get(ld["sampler"], v), v)

    samplers = []
    for iid in sampler_ids:
        s = asy.sampler.get(iid) or {}
        c = by_id.get(iid) or {}
        prov = s.get("provenance") or {}
        card = prov.get("card_model") or c.get("gpu")
        rows = sorted((r for r in s.get("rounds", {}).values() if r.get("ts") is not None), key=lambda r: r["ts"])
        rv = _int(rows[-1].get("policy_version")) if rows else None
        ver = max([v for v in (loaded.get(iid), rv) if v is not None], default=None)
        st = states.get(iid) or {}
        status, live = st.get("status"), bool(st.get("alive"))
        state = "sampling"
        reload = None
        if not live:
            state = "offline"
        elif status == "starting":
            state = "starting"
        elif status == "waiting":
            state = "waiting"
        if live and latest is not None and ver is not None and latest > ver and last_release is not None:
            # a release this sampler has not loaded yet: fetching + draining until its drain
            # event, then loading (the fetch window is only on the load event, once it is done)
            rel_ts = last_release["ts"] or 0
            drained = drains.get(iid) is not None and drains[iid] >= rel_ts
            state = "loading" if drained else "reloading"
            reload = {"target": latest, "since": drains[iid] if drained else rel_ts}
        topo = str(prov.get("topology") or "")
        m = re.search(r"tp(\d+)", topo)
        samplers.append({
            "id": iid, "card": card, "precision": (prov.get("precision") or "").upper() or None,
            "engine": _engine(prov.get("engine")), "arch": arch_of(card),
            "gpus": c.get("gpus") or (int(m.group(1)) if m else None),
            "status": status, "live": live, "state": state, "reload": reload,
            "reason": (c.get("left") or {}).get("reason") or c.get("waiting_reason"),
            "policy_version": ver, "lag": (latest - ver) if latest is not None and ver is not None else None,
            "rate_gpm": round(groups_recent.get(iid, 0) / (RATE_S / 60.0), 3),
            "last_group_ts": last_group.get(iid), "since_ts": s.get("start"),
        })
    samplers.sort(key=lambda x: (not x["live"], str(x["card"] or "~"), x["id"]))

    trainers = []
    for iid in trainer_ids:
        c = by_id.get(iid) or {}
        st = states.get(iid) or {}
        status = st.get("status")
        ph = phase.get(iid) or {}
        trainers.append({
            "id": iid, "name": c.get("name"), "gpu": c.get("gpu"), "gpus": c.get("gpus"), "arch": arch_of(c.get("gpu")),
            "status": status, "live": bool(st.get("alive")), "starting": bool(st.get("starting")),
            "step": c.get("round") if c.get("round") is not None else c.get("rollout_id"),
            "policy_version": c.get("policy_version"),
            "phase": DRIVER_PHASES.get(ph.get("phase"), ph.get("phase")), "phase_since": ph.get("since"),
            "reason": (c.get("left") or {}).get("reason"),
        })
    trainers.sort(key=lambda x: (not x["live"], x["id"]))

    bl = asy.backlog_series() or {}
    pts = bl.get("points") or []
    last_pt = pts[-1] if pts else None
    return {
        "now": now, "window_s": WINDOW_S, "rate_s": RATE_S,
        "samplers": samplers, "trainers": trainers,
        "buffer": {"backlog": last_pt[3] if last_pt else None, "written": bl.get("written"), "read": bl.get("read"),
                   "read_source": bl.get("read_source")},
        "latest": latest, "last_release": last_release,
        "store": (reducer.store_status() or {}).get("uri") if callable(reducer.store_status) else None,
        "flows": flows,
    }
