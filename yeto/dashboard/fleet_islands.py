"""A fleet of async GRPO islands: K trainers of one synced group, each with its own external
sampler pools on its store lane ``trainer<k>/``, and a syncer outside the islands.

Read-only, like the rest of the dashboard. Two sources, both optional:

* ``<store>/fleet/plan.json`` (written by ``yeto run up`` for a run.json with ``islands``):
  island names, their trainer GPU, and the pools of each island (engine, rollout precision,
  provider, shard base). Read like ``breakers.resolved.json`` (:class:`PlanFile`): a local
  copy next to a tape or under a local store directory, else the run store's object at most
  every ``PLAN_POLL_S``;
* ``rl_sampler_start.island`` / ``.store``: the island (learner id) and the store each
  external sampler runs on.

Which island a sampler belongs to, strongest evidence first: its own ``island`` field; the
plan pool whose shard base names its shard; the ``trainer<k>`` its store ends in. ``k`` is
the learner id everywhere (``sample_roles.trainer_substore``); the trainer of learner ``k``
is the ``k``-th trainer island in island order (``sample_roles.synced_group_learners``),
which is island ``k`` itself in a fleet (no sampler island in its launch).

A run without a plan and without a sampler that names its island is no fleet: :func:`resolve`
returns None and every view keeps its single-trainer form.

Per-pool mismatch (:func:`pool_mismatch`): a Miles trainer reports the train-rollout
log-prob difference (``train_metrics.train_rollout_logprob_abs_diff``) and the TIS clip
fraction (``train_metrics.tis_clipfrac`` / ``mismatch.tis_clipfrac``) once per step, over
groups of every pool of its island. Each step's composition comes from
``rl_xhw_mismatch.sources[*].share`` (token shares, with ``YETO_RL_XHW_GUARD``) or from the
step's ``rl_trajectory_reward`` ids (sample counts; ``Reducer.sampler_shares``). A pool's
value is the mean of the steps it fed alone; else a least-squares split of the mixed steps
when the shares vary enough to separate the pools; else the share-weighted mean (marked).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

PLAN_KEY = "fleet/plan.json"
PLAN_FILE = Path(PLAN_KEY)
PLAN_POLL_S = 60.0  # the plan is written once by run up: a slow re-read is enough
SUBSTORE = re.compile(r"/trainer(\d+)/?$")
SHARD_N = re.compile(r"-\d+$")
MISMATCH_KEYS = ("logprob_diff", "tis_clip")
MISMATCH_WINDOW = 50  # trained rounds the per-pool split looks at (the newest)
COND_MAX = 1e6  # a normal matrix worse than this cannot separate the pools


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _f(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _str(v: Any) -> str | None:
    return str(v) if isinstance(v, str) and v else None


# -- the plan ---------------------------------------------------------------------------
def load_plan(doc: Any) -> dict | None:
    """The plan in the form the views read, or None when ``doc`` is not a fleet plan.
    Unknown keys are ignored (a later run up may add some)."""
    if not isinstance(doc, dict) or not isinstance(doc.get("islands"), list) or _int(doc.get("version")) is None:
        return None
    islands = []
    for pos, isl in enumerate(doc["islands"]):
        if not isinstance(isl, dict):
            return None
        index = _int(isl.get("index"))
        index = pos if index is None else index
        learner = _int(isl.get("learner"))
        learner = index if learner is None else learner
        pools = []
        for p in isl.get("pools") or ():
            if not isinstance(p, dict) or not _str(p.get("name")):
                continue
            pools.append({"name": p["name"], "source_name": _str(p.get("source_name")),
                          "provider": _str(p.get("provider")), "shard_base": _str(p.get("shard_base")),
                          "engine": _str(p.get("engine")), "rollout_quant": _str(p.get("rollout_quant"))})
        shards = [str(s) for s in isl.get("buffer_shards") or () if isinstance(s, (str, int))]
        islands.append({"index": index, "learner": learner, "name": _str(isl.get("name")),
                        "substore": _str(isl.get("substore")) or f"trainer{learner}", "gpu": _str(isl.get("gpu")),
                        "usd_per_h": _f(isl.get("usd_per_h")), "release_every": _int(isl.get("release_every")),
                        "max_policy_age": _int(isl.get("max_policy_age")), "buffer_shards": shards, "pools": pools})
    islands.sort(key=lambda x: x["learner"])
    return {"version": doc["version"], "run": _str(doc.get("run")), "store": _str(doc.get("store")),
            "dispense": _str(doc.get("dispense")), "islands": islands}


def plan_candidates(paths: list[str], max_up: int = 3) -> list[str]:
    """Where a local copy of the plan may be: ``fleet/plan.json`` under a tape's directory
    (a local store directory passed as a tape path) or one of its parents."""
    out: list[str] = []
    for p in paths:
        base = Path(os.path.expanduser(p))
        for d in [base, *list(base.parents)[:max_up]]:
            c = str(d / PLAN_FILE)
            if c not in out:
                out.append(c)
    return out


class PlanFile:
    """Feeds ``fleet/plan.json`` to a reducer (``Reducer.load_fleet_plan``): the first local
    candidate that exists (read again when its mtime changes), else the run store's copy
    (read at most every ``PLAN_POLL_S``). Once a plan names its islands, :meth:`tapes` reads
    the tapes the external samplers upload under their island's store lane
    (``<store>/trainer<k>/tapes/``)."""

    def __init__(self, candidates: list[str | os.PathLike], objects: Any = None, *,
                 clock: Callable[[], float] | None = None, interval_s: float = PLAN_POLL_S,
                 tape_interval_s: float = 0.0) -> None:
        import time  # noqa: PLC0415

        self.candidates = [Path(os.path.expanduser(str(c))) for c in candidates]
        self.objects = objects
        self.clock = clock or time.monotonic
        self.interval_s = float(interval_s)
        self.tape_interval_s = float(tape_interval_s)
        self._mtime: tuple[str, float] | None = None
        self._next_store: float | None = None
        self._store_bytes: bytes | None = None
        self._tapes: dict[str, Any] = {}
        self.substores: list[str] = []

    def pump(self, reducer) -> bool:
        for p in self.candidates:
            try:
                st = p.stat()
            except OSError:
                continue
            key = (str(p), st.st_mtime)
            if key == self._mtime:
                return False
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeDecodeError):
                return False
            self._mtime = key
            return self._load(reducer, doc, str(p))
        if self.objects is None:
            return False
        now = self.clock()
        if self._next_store is not None and now < self._next_store:
            return False
        self._next_store = now + self.interval_s
        try:
            data = self.objects.read(PLAN_KEY)
        except Exception:  # noqa: BLE001 - the store's own status reports read failures
            return False
        if not data or data == self._store_bytes:
            return False
        try:
            doc = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return False
        self._store_bytes = data
        return self._load(reducer, doc, "store " + PLAN_KEY)

    def _load(self, reducer, doc: Any, source: str) -> bool:
        if not reducer.load_fleet_plan(doc, source):
            return False
        self.substores = [isl["substore"] for isl in reducer.fleet_plan["islands"]]
        return True

    def tapes(self) -> list:
        """One ``StoreTapes`` per island store lane of the plan (the run store's objects)."""
        if self.objects is None:
            return []
        from .store_tapes import PREFIX, StoreTapes  # noqa: PLC0415

        for sub in self.substores:
            if sub not in self._tapes:
                self._tapes[sub] = StoreTapes(self.objects, label=f"store/{sub}", prefix=f"{sub}/{PREFIX}",
                                              interval_s=self.tape_interval_s)
        return [self._tapes[s] for s in self.substores]

    def pump_tapes(self, reducer) -> int:
        return sum(t.pump(reducer) for t in self.tapes())


# -- which island, which pool -------------------------------------------------------------
def _order(i: str) -> tuple:
    return (0, int(i)) if i.isdigit() else (1, i)


def trainer_ids(reducer) -> list[str]:
    roles = reducer.async_.roles
    return sorted((i for i in reducer.islands if roles.get(i) == "trainer"), key=_order)


def learner_trainers(tids: list[str], plan: dict | None = None) -> dict[int, str]:
    """{learner id: trainer island id}: the trainers in island order are learners 0..K-1
    (``sample_roles.synced_group_learners``). A fleet's launch has no sampler island, so
    there trainer island ``k`` is learner ``k`` (the plan's ``learner``)."""
    out = {k: tid for k, tid in enumerate(tids)}
    for isl in (plan or {}).get("islands") or ():
        if str(isl["learner"]) in tids:
            out[isl["learner"]] = str(isl["learner"])
    return out


def sampler_starts(reducer) -> dict[str, dict]:
    """Each sampler's newest ``rl_sampler_start`` record (its island and store)."""
    out: dict[str, tuple[float, dict]] = {}
    for e in reducer.events:
        if e.get("type") != "rl_sampler_start" or e.get("island") is None:
            continue
        ts = e.get("ts")
        old = out.get(e["island"])
        if old is None or ts is None or old[0] is None or ts >= old[0]:
            out[e["island"]] = (ts, e.get("record") or {})
    return {k: v[1] for k, v in out.items()}


def _plan_pool(plan: dict | None, sid: str) -> tuple[int, dict] | None:
    """(learner, pool) of the plan pool whose shards include ``sid`` (longest shard base)."""
    best = None
    for isl in (plan or {}).get("islands") or ():
        for p in isl["pools"]:
            base = p.get("shard_base")
            if base and (sid == base or (sid.startswith(base + "-") and sid[len(base) + 1:].isdigit())):
                if best is None or len(base) > len(best[1]["shard_base"]):
                    best = (isl["learner"], p)
        if best is None and sid in isl["buffer_shards"] and len(isl["pools"]) == 1:
            best = (isl["learner"], isl["pools"][0])
    return best


def sampler_learner(sid: str, start: dict, plan: dict | None) -> int | None:
    """The learner id (store lane ``trainer<k>``) sampler ``sid`` feeds (module doc)."""
    k = _int(start.get("island"))
    if k is not None:
        return k
    hit = _plan_pool(plan, sid)
    if hit is not None:
        return hit[0]
    m = SUBSTORE.search(str(start.get("store") or ""))
    return int(m.group(1)) if m else None


def is_fleet(reducer) -> bool:
    if getattr(reducer, "fleet_plan", None):
        return True
    return any(_int(r.get("island")) is not None for r in sampler_starts(reducer).values())


def resolve(reducer, tids: list[str] | None = None) -> dict | None:
    """The fleet as the views draw it (module doc); None for a run that is no fleet."""
    tids = trainer_ids(reducer) if tids is None else list(tids)
    plan = getattr(reducer, "fleet_plan", None)
    key = (reducer.event_seq, getattr(reducer, "fleet_plan_seq", 0), tuple(tids))
    hit = getattr(reducer, "_fleet_view", None)
    if hit is not None and hit[0] == key:
        return hit[1]
    out = _resolve(reducer, tids, plan)
    reducer._fleet_view = (key, out)
    return out


def _resolve(reducer, tids: list[str], plan: dict | None) -> dict | None:
    starts = sampler_starts(reducer)
    if not plan and not any(_int(r.get("island")) is not None for r in starts.values()):
        return None
    asy = reducer.async_
    sids = sorted((i for i in reducer.islands if asy.roles.get(i) == "sampler"), key=_order)
    learners = learner_trainers(tids, plan)
    islands: dict[int, dict] = {}
    for isl in (plan or {}).get("islands") or ():
        islands[isl["learner"]] = {**{k: isl[k] for k in ("index", "learner", "name", "substore", "gpu", "usd_per_h",
                                                         "release_every", "max_policy_age")},
                                   "pools": [dict(p, samplers=[]) for p in isl["pools"]]}
    island_of: dict[str, int] = {}
    pool_of: dict[str, str] = {}
    unbound: list[str] = []
    for sid in sids:
        start = starts.get(sid) or {}
        k = sampler_learner(sid, start, plan)
        if k is None:
            unbound.append(sid)
            continue
        island_of[sid] = k
        isl = islands.setdefault(k, {"index": k, "learner": k, "name": None, "substore": f"trainer{k}", "gpu": None,
                                     "usd_per_h": None, "release_every": None, "max_policy_age": None, "pools": []})
        hit = _plan_pool(plan, sid)
        name = hit[1]["name"] if hit is not None and hit[0] == k else SHARD_N.sub("", sid)
        pool = next((p for p in isl["pools"] if p["name"] == name), None)
        if pool is None:  # no plan pool names it: the samplers sharing a shard base are one pool
            prov = (asy.sampler.get(sid) or {}).get("provenance") or {}
            pool = {"name": name, "source_name": None, "provider": None, "shard_base": name,
                    "engine": _str(prov.get("engine")), "rollout_quant": None, "samplers": []}
            isl["pools"].append(pool)
        pool["samplers"].append(sid)
        pool_of[sid] = name
    for k, isl in islands.items():
        isl["trainer"] = learners.get(k)
        for tid in [isl["trainer"]] if isl["trainer"] is not None else []:
            island_of[tid] = k
    return {"source": "plan" if plan else "events", "run": (plan or {}).get("run"),
            "dispense": (plan or {}).get("dispense"),
            "islands": [islands[k] for k in sorted(islands)], "island_of": island_of, "pool_of": pool_of,
            "unbound": unbound}


# -- per-pool mismatch ---------------------------------------------------------------------
def xhw_shares(reducer, tid: str) -> dict[int, dict[str, float]]:
    """{rollout_id: {sampler id: token share}} of trainer ``tid``'s ``rl_xhw_mismatch``
    records (source keys ``sampler|engine|version|card``)."""
    out: dict[int, dict[str, float]] = {}
    for e in reducer.events:
        if e.get("type") != "rl_xhw_mismatch" or e.get("island") != tid:
            continue
        rec = e.get("record") or {}
        rid, src = _int(rec.get("rollout_id")), rec.get("sources")
        if rid is None or not isinstance(src, dict):
            continue
        shares: dict[str, float] = {}
        for key, row in src.items():
            share = _f((row or {}).get("share")) if isinstance(row, dict) else None
            if share:
                sid = str(key).split("|", 1)[0]
                shares[sid] = shares.get(sid, 0.0) + share
        if shares:
            out[rid] = shares
    return out


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Least squares a x = b by the normal equations; None when they do not separate x."""
    n = len(a[0])
    m = [[sum(r[i] * r[j] for r in a) for j in range(n)] + [sum(r[i] * y for r, y in zip(a, b))] for i in range(n)]
    diag = [m[i][i] for i in range(n)]
    if min(diag) <= 0:
        return None
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) * COND_MAX < max(diag):
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def pool_mismatch(rounds: list[dict], pool_of: dict[str, str], xhw: dict[int, dict[str, float]] | None = None
                  ) -> dict[str, dict]:
    """{pool: {metric: {"value", "method", "rounds", "last"}}} over a trainer's round records
    (``Reducer.round_records``: ``round``, ``logprob_diff``, ``tis_clip``, ``by_sampler``)."""
    comp: list[tuple[dict[str, float], dict]] = []
    for r in rounds:
        rid = r.get("round")
        src = (xhw or {}).get(rid)
        if src is None:
            src = {sid: _f(v.get("samples")) or 0.0 for sid, v in (r.get("by_sampler") or {}).items()}
        shares: dict[str, float] = {}
        for sid, w in src.items():
            if sid in pool_of and w > 0:
                shares[pool_of[sid]] = shares.get(pool_of[sid], 0.0) + w
        total = sum(shares.values())
        if total > 0:
            comp.append(({p: w / total for p, w in shares.items()}, r))
    comp = comp[-MISMATCH_WINDOW:]
    out: dict[str, dict] = {}
    for key in MISMATCH_KEYS:
        rows = [(s, _f(r.get(key)), r.get("round")) for s, r in comp if _f(r.get(key)) is not None]
        pools = sorted({p for s, _v, _r in rows for p in s})
        solved = None
        if len(pools) > 1 and len(rows) > len(pools):
            solved = _solve([[s.get(p, 0.0) for p in pools] for s, _v, _r in rows], [v for _s, v, _r in rows])
        for j, p in enumerate(pools):
            fed = [(s[p], v, rid) for s, v, rid in rows if p in s]
            alone = [v for w, v, _rid in fed if w >= 0.999]
            if alone:
                value, method = sum(alone) / len(alone), "single-pool rounds"
            elif solved is not None:
                value, method = max(0.0, solved[j]), "least squares over mixed rounds"
            else:
                value = sum(w * v for w, v, _rid in fed) / sum(w for w, _v, _rid in fed)
                method = "share-weighted (pools not separable)"
            last = fed[-1]
            out.setdefault(p, {})[key] = {"value": round(value, 6), "method": method, "rounds": len(fed),
                                          "alone": len(alone), "last": {"round": last[2], "value": last[1],
                                                                        "share": round(last[0], 4)}}
    return out


def attach_mismatch(reducer, asy: dict, rounds_by_island: dict[str, list[dict]], fleet: dict) -> None:
    """Per-pool train-rollout mismatch into the async view's island blocks (``pools[*].mismatch``)."""
    for block in asy.get("islands") or ():
        tid = block.get("trainer")
        if tid is None:
            continue
        per = pool_mismatch(rounds_by_island.get(tid) or [], fleet["pool_of"], xhw_shares(reducer, tid))
        for p in block["pools"]:
            if p["name"] in per:
                p["mismatch"] = per[p["name"]]
