"""Fold JSONL tapes into the dashboard views (design D1/D5).

The reducer is the single source of the numbers shown by both the live
server and the static export. It is incremental: ``feed`` takes one record
together with its (source, byte offset); a record at or before an offset
already consumed from that source is ignored, so re-reading a tape from an
older offset never double counts.

Streams understood (classified per record, not per file):

* learner tapes: ``{"event": "rl_*", "island_id": N, ...}``
* syncer tape: merge records (``"step"`` present, see
  ``wandb_tape._is_merge_record``) and ``policy_sweep_ledger``
* controller journal: ``{"kind": ..., "seq": ...}`` (``journal.Journal.append``)
* head ``fleet.jsonl``: ``island_launch|island_ready|island_lost|island_stop``
  and ``cost_tick``

Anything else is passed through to the events view. Missing values stay
``None`` (rendered as "无数据"), never 0.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Iterable

from . import alerts as alerts_mod
from . import cost as cost_mod

FLEET_EVENTS = frozenset({"island_launch", "island_ready", "island_lost", "island_stop", "cost_tick"})
POLICY_SWEEP_LEDGER_EVENT = "policy_sweep_ledger"
SYNCER_ISLAND_EVENTS = frozenset({"rl_fragment_push", "rl_policy_apply", "rl_pull_resend",
                                  "rl_member_publication"})
TERMINAL_TX_PHASES = ("COMMITTED", "SUCCEEDED", "CANCELLED", "REBUILT_OLD", "RECOVERY_REQUIRED",
                      "FAILED")
MAX_EVENTS = 20000
NO_DATA = None

# Overlay-chart metrics (v7 tabs): series key -> (label, extractor field paths).
# A path "a.b" reads record["a"]["b"]. The first finite value wins.
METRICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("reward", "reward mean", ("reward_mean", "rl/reward_mean")),
    ("reward_p10", "reward p10", ("reward_p10",)),
    ("reward_p90", "reward p90", ("reward_p90",)),
    ("pg_loss", "pg_loss", ("pg_loss", "train_metrics.pg_loss", "train/pg_loss")),
    ("kl", "KL", ("mean_kl", "train_metrics.ppo_kl", "train_metrics.kl_loss", "train/train_rollout_kl")),
    ("entropy", "entropy", ("entropy", "train_metrics.entropy", "train_metrics.entropy_loss")),
    ("grad_norm", "grad_norm", ("grad_norm", "train_metrics.grad_norm", "train/grad_norm")),
    ("clip", "clip 总", ("clip_fraction", "train_metrics.pg_clipfrac", "train/pg_clipfrac")),
    ("clip_lo", "clip 低侧", ("train_metrics.pg_clipfrac_lower", "train_metrics.pg_clipfrac_low")),
    ("clip_hi", "clip 高侧", ("train_metrics.pg_clipfrac_upper", "train_metrics.pg_clipfrac_high")),
    ("resp_len", "response len", ("resp_len_mean",)),
    ("tok_s", "tok/s", ("tok_per_s",)),
)
EXTRA_SERIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loss", ("loss", "train_metrics.loss", "train/loss")),
    ("reward_std", ("reward_std",)),
    ("lr", ("lr", "applied_lr")),
    ("ess", ("ess_ratio", "train_metrics.ess_ratio")),
    ("trunc", ("truncated_frac",)),
    ("reward_p50", ("reward_p50",)),
    ("resp_p95", ("resp_len_p95",)),
    ("logprob_diff", ("train_metrics.train_rollout_logprob_abs_diff", "train/train_rollout_logprob_abs_diff")),
    ("train_step", ("train_step",)),
)


def _get(record: dict, path: str) -> Any:
    cur: Any = record
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def first_finite(record: dict, paths: Iterable[str]) -> float | None:
    for p in paths:
        v = finite(_get(record, p))
        if v is not None:
            return v
    return None


def nonfinite_seen(record: dict, paths: Iterable[str]) -> bool:
    for p in paths:
        v = _get(record, p)
        if isinstance(v, float) and not math.isfinite(v):
            return True
        if isinstance(v, str) and v.lower() in ("nan", "inf", "-inf"):
            return True
    return False


def record_ts(record: dict) -> float | None:
    for key in ("time_unix", "ts", "wall_time", "time"):
        v = finite(record.get(key))
        if v is not None:
            return v
    return None


def classify(record: dict) -> str:
    event = record.get("event")
    if event in FLEET_EVENTS:
        return "fleet"
    if "kind" in record and "seq" in record and event is None:
        return "journal"
    if event == POLICY_SWEEP_LEDGER_EVENT or ("step" in record and event is None
                                              and ("responders" in record or "expected" in record
                                                   or "sync/responders" in record)):
        return "syncer"
    if isinstance(event, str):
        return "learner"
    return "other"


HOST_SAMPLE_EVENT = "modal_host_sample"
STOP_SLACK_S = 5.0
NODE_SERIES_MAX = 20000
NODE_SERIES_POINTS = 300
# rl_timeline_span task -> page phase: R 推理生成 / T 训练 / S 训练后同步 / P 发布
SPAN_PHASES = {"generate": "R", "train": "T", "outer_sync": "S", "publish": "P"}
# Events only written once driver.run() is going (older tapes may lack rl_driver_start).
DRIVER_RUNNING_EVENTS = frozenset({"rl_driver_phase", "rl_heartbeat", "rl_local_round", "rl_round_trained",
                                   "rl_resource_sample", "rl_publication"})


def _iid(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _new_island(iid: str) -> dict:
    return {
        "id": iid, "name": None, "cloud": None, "region": None, "gpu": None, "gpus": None,
        "price_key": None, "cpus": None, "memory_gib": None, "driver_started": False,
        "nodes": {}, "node_series": {}, "spans": [], "pubs": {}, "first_ts": None, "last_event_ts": None, "last_heartbeat_ts": None,
        "heartbeat_seen": False, "round": None, "rollout_id": None, "policy_version": None,
        "phase": None, "finalized": False, "fleet_state": None, "ready_ts": None, "stop_ts": None,
        "lost_ts": None, "open_ts": None, "closed_s": 0.0, "points": {}, "nonfinite": [], "resource": None, "host": None, "staleness": None,
        "contribution": None, "reconfig": [], "cells": None, "cells_source": None,
        "transactions": {}, "tx_order": [], "recovery_required": [], "source_lost": None,
        "recent": deque(maxlen=50), "events_by_type": {},
        # rl-resume-from-checkpoint 4.3: one entry per resume (launch boundary), the
        # rounds each launch trained (rid -> [ts...]) and the cuts saved
        "resumes": [], "trained_ts": {}, "cuts": [],

        # fleet-dashboard 8.4: startup sub-steps {step: {"seconds", "step_s", "ts"}}
        "startup_steps": {}, "startup_step": None, "startup_step_ts": None,
    }


class Reducer:
    """Incremental tape -> view reducer. Not thread-safe; callers lock."""

    def __init__(self, *, run: str | None = None, prices: dict | None = None,
                 budget_usd: float | None = None, thresholds: dict | None = None):
        self.run = run
        self.prices = prices if prices is not None else cost_mod.load_prices(None)
        self.budget_usd = budget_usd
        self.thresholds = alerts_mod.merged_thresholds(thresholds)
        self.offsets: dict[str, int] = {}
        self.islands: dict[str, dict] = {}
        self.merges: list[dict] = []
        self.ledger: dict | None = None
        self.pushes: dict[int, set[str]] = {}
        self.resends: dict[int, int] = {}
        self.applies: dict[int, set[str]] = {}
        self.publications: dict[int, set[str]] = {}
        self.fleet_records: list[dict] = []
        self.cost_ticks: list[dict] = []
        self.events: deque = deque(maxlen=MAX_EVENTS)
        self.event_seq = 0
        self.max_ts: float | None = None
        self.min_ts: float | None = None
        self.counts = {"learner": 0, "syncer": 0, "journal": 0, "fleet": 0, "other": 0}
        self.sources_seen: list[str] = []
        self.operator_stops: list[dict] = []  # our own stop of the run (sources.operator_stops_near)
        self._run_meta_cache: tuple | None = None

    # -- feeding ----------------------------------------------------------------
    def feed(self, record: Any, *, source: str = "-", offset: int | None = None,
             island: str | None = None, node: str | None = None) -> bool:
        """Fold one record. Returns False when skipped as already consumed."""
        if source not in self.sources_seen and source != "-":
            self.sources_seen.append(source)
            self._run_meta_cache = None
        if offset is not None:
            if offset <= self.offsets.get(source, -1):
                return False
            self.offsets[source] = offset
        if not isinstance(record, dict):
            return False
        if record.get("event") == HOST_SAMPLE_EVENT:
            return self._feed_host(record, island, node)
        kind = classify(record)
        self.counts[kind] += 1
        ts = record_ts(record)
        if ts is not None:
            self.max_ts = ts if self.max_ts is None else max(self.max_ts, ts)
            self.min_ts = ts if self.min_ts is None else min(self.min_ts, ts)
        iid = _iid(record.get("island_id", record.get("learner_id"))) or island
        getattr(self, "_feed_" + kind)(record, iid, ts)
        self.event_seq += 1
        etype = record.get("event") or (("journal:" + str(record.get("kind"))) if kind == "journal"
                                        else ("syncer_merge" if kind == "syncer" else "unknown"))
        self.events.append({"seq": self.event_seq, "ts": ts, "island": iid, "type": etype,
                            "source": source, "stream": kind, "record": record})
        return True

    def _feed_host(self, r: dict, island: str | None, node: str | None = None) -> bool:
        """``modal_host_sample`` has no island_id: attach it to the island the
        source resolved (path ``l<N>/rank<R>``); otherwise ignore (never a "?" island)."""
        self.counts["other"] += 1
        iid = _iid(r.get("island_id")) or island
        if iid is None:
            return True
        isl = self.island(iid)
        ts = record_ts(r)
        used, total = finite(r.get("meminfo_used")), finite(r.get("meminfo_total"))
        cur = finite(r.get("cgroup_current"))
        cur = cur if cur is not None else used
        host = isl["host"] or {"peak_bytes": None, "samples": 0}
        if cur is not None:
            host["peak_bytes"] = cur if host["peak_bytes"] is None else max(host["peak_bytes"], cur)
        peak = finite(r.get("cgroup_peak"))
        if peak is not None:
            host["peak_bytes"] = peak if host["peak_bytes"] is None else max(host["peak_bytes"], peak)
        gm = [finite(x) for x in r.get("gpu_mem_used_mib") or []]
        prev = host.get("gpu_mem_used_mib_peak") or []
        peaks = [max(a or 0, b or 0) for a, b in zip(gm + [0] * (len(prev) - len(gm)),
                                                    prev + [0] * (len(gm) - len(prev)))]
        host.update({"samples": host["samples"] + 1, "ts": ts, "current_bytes": cur, "total_bytes": total,
                     "gpu_mem_used_mib": gm, "gpu_mem_used_mib_peak": peaks})
        isl["host"] = host
        rank = _iid(r.get("node_rank")) or node
        if rank is not None:
            # Per-node view: a disaggregated island's rollout node is only
            # visible here (rl_resource_sample covers the driver's node only).
            utils = [finite(x) for x in r.get("gpu_util_pct") or []]
            utils = [u for u in utils if u is not None]
            nd = isl["nodes"].get(rank) or {"samples": 0, "gpu_mem_used_mib_peak": [], "gpu_util_pct_max": None}
            npk = nd["gpu_mem_used_mib_peak"]
            nd["gpu_mem_used_mib_peak"] = [max(a or 0, b or 0) for a, b in zip(
                gm + [0] * (len(npk) - len(gm)), npk + [0] * (len(gm) - len(npk)))]
            util = round(sum(utils) / len(utils), 1) if utils else None
            if util is not None:
                nd["gpu_util_pct_max"] = util if nd["gpu_util_pct_max"] is None else max(nd["gpu_util_pct_max"], util)
            nd.update({"node": rank, "samples": nd["samples"] + 1, "ts": ts, "gpu_mem_used_mib": gm,
                       "gpu_util_pct": util, "host_mem_bytes": cur, "host_mem_total_bytes": total})
            isl["nodes"][rank] = nd
            if ts is not None and gm:
                ser = isl["node_series"].setdefault(rank, deque(maxlen=NODE_SERIES_MAX))
                ser.append([ts, round(sum(x or 0 for x in gm) / len(gm) / 1024, 2), util])
        self._touch(isl, ts)
        return True

    def feed_operator_stop(self, rec: dict) -> bool:
        """Record that WE stopped the run's cloud app (gate / early / watchdog).
        Idempotent per (path, time). The earliest stop wins."""
        key = (rec.get("path"), rec.get("time_unix"), rec.get("cause"))
        if any((x.get("path"), x.get("time_unix"), x.get("cause")) == key for x in self.operator_stops):
            return False
        self.operator_stops.append(dict(rec))
        self.operator_stops.sort(key=lambda x: x.get("time_unix") or 0)
        ts = finite(rec.get("time_unix"))
        self.event_seq += 1
        self.events.append({"seq": self.event_seq, "ts": ts, "island": None, "type": "operator_stop",
                            "source": rec.get("path") or "-", "stream": "other", "record": rec})
        return True

    def operator_stop(self) -> dict | None:
        return self.operator_stops[0] if self.operator_stops else None

    def stopped_by_us(self, isl: dict) -> bool:
        """True when we issued a stop and the island's first failure (RECOVERY_REQUIRED
        or fleet island_lost) is not earlier than it (5 s slack for clock skew between
        this machine and the container): the failure is a consequence of our stop."""
        stop = self.operator_stop()
        if stop is None or stop.get("time_unix") is None:
            return False
        fails = [x["ts"] for x in isl["recovery_required"] if x.get("ts") is not None]
        if isl["lost_ts"] is not None:
            fails.append(isl["lost_ts"])
        return not fails or min(fails) >= stop["time_unix"] - STOP_SLACK_S

    def feed_many(self, records: Iterable[dict], **kw: Any) -> None:
        for r in records:
            self.feed(r, **kw)

    def island(self, iid: str) -> dict:
        isl = self.islands.get(iid)
        if isl is None:
            isl = self.islands[iid] = _new_island(iid)
        return isl

    def _touch(self, isl: dict, ts: float | None) -> None:
        if ts is None:
            return
        isl["last_event_ts"] = ts if isl["last_event_ts"] is None else max(isl["last_event_ts"], ts)
        isl["first_ts"] = ts if isl["first_ts"] is None else min(isl["first_ts"], ts)

    def _point(self, isl: dict, x: int, values: dict) -> None:
        pt = isl["points"].setdefault(x, {})
        for k, v in values.items():
            if v is not None:
                pt[k] = v

    def _feed_learner(self, r: dict, iid: str | None, ts: float | None) -> None:
        event = r["event"]
        if iid is None:
            return  # no island identity: never fabricate a "?" island
        isl = self.island(iid)
        self._touch(isl, ts)
        isl["events_by_type"][event] = isl["events_by_type"].get(event, 0) + 1
        isl["recent"].append({"ts": ts, "type": event, "summary": _summary(r)})
        if r.get("policy_version") is not None and event in (
                "rl_policy_apply", "rl_publication", "rl_driver_phase", "rl_heartbeat",
                "rl_round_cut", "rl_member_publication", "rl_resume"):
            isl["policy_version"] = r.get("policy_version")
        if event == "rl_driver_start" and isl.get("driver_start_ts") is None:
            isl["driver_start_ts"] = ts
        if event == "rl_driver_start":
            isl.setdefault("driver_starts", []).append(ts)
        if event == "rl_driver_start" or (event in DRIVER_RUNNING_EVENTS and r.get("phase") != "startup"):
            isl["driver_started"] = True
        if event == "rl_driver_phase":
            isl["phase"] = r.get("phase")
        elif event == "rl_startup_step":
            step = r.get("step")
            if step is not None:
                isl["startup_steps"][str(step)] = {"seconds": r.get("seconds"), "step_s": r.get("step_s"), "ts": ts}
                isl["startup_step"] = str(step)
                if ts is not None:
                    isl["startup_step_ts"] = max(ts, isl["startup_step_ts"] or ts)
        elif event == "rl_heartbeat":
            if r.get("phase") == "startup" and r.get("startup_step") and isl["startup_step"] is None:
                isl["startup_step"] = r.get("startup_step") if r.get("startup_step") != "begin" else None
            isl["heartbeat_seen"] = True
            if ts is not None:
                isl["last_heartbeat_ts"] = max(ts, isl["last_heartbeat_ts"] or ts)
            isl["phase"] = r.get("phase", isl["phase"])
            if r.get("rollout_id") is not None:
                isl["rollout_id"] = r["rollout_id"]
        elif event == "rl_timeline_span":
            self._span(isl, r)
        elif event == "rl_publication":
            pv = r.get("policy_version")
            if isinstance(pv, int) and ts is not None:
                isl["pubs"][pv] = ts
        elif event == "rl_resource_sample":
            isl["resource"] = _resource(r, ts)
        elif event in ("rl_local_round", "rl_round_trained"):
            self._round_metrics(isl, r, event)
            if event == "rl_round_trained" and isinstance(r.get("rollout_id"), int):
                isl["trained_ts"].setdefault(r["rollout_id"], []).append(ts)
        elif event == "rl_resume":
            isl["resumes"].append({
                "ts": ts, "rollout_id": r.get("rollout_id"), "incarnation": r.get("incarnation"),
                "cut_id": r.get("cut_id"), "restore_s": r.get("restore_s"),
                "cut_bytes": r.get("cut_bytes"), "config_diff": r.get("config_diff"),
                "driver_start_ts": isl.get("driver_start_ts")})
        elif event == "rl_cut_saved":
            isl["cuts"].append({"ts": ts, "rollout_id": r.get("rollout_id"), "cut_id": r.get("cut_id"),
                                "bytes": r.get("cut_bytes"), "save_s": r.get("save_s"),
                                "total_s": r.get("total_s")})
        elif event == "rl_learner_finalized":
            isl["finalized"] = True
        elif event == "rl_reconfiguration":
            isl["reconfig"].append({"ts": ts, "result": r.get("result"), "rollout_id": r.get("rollout_id"),
                                    "cause": r.get("cause"), "error": r.get("error")})
            if r.get("result") == "RECOVERY_REQUIRED":
                isl["recovery_required"].append({"ts": ts, "source": "tape", "tx_id": None,
                                                 "round": r.get("rollout_id"),
                                                 "error": r.get("error") or r.get("cause")})
        elif event == "rl_cell_snapshot":
            self._cell_snapshot(isl, r)
        elif event == "rl_reconfig_phase":
            self._reconfig_phase(isl, r, ts)
        elif event == "dashboard_source_lost":
            isl["source_lost"] = {"ts": ts, "error": r.get("error"), "recovered": False}
        elif event == "dashboard_source_restored":
            if isl["source_lost"]:
                isl["source_lost"]["recovered"] = True
        if event in SYNCER_ISLAND_EVENTS:
            step = r.get("global_step", r.get("step", r.get("policy_version")))
            if isinstance(step, int):
                target = {"rl_fragment_push": self.pushes, "rl_policy_apply": self.applies,
                          "rl_member_publication": self.publications}.get(event)
                if target is not None:
                    target.setdefault(step, set()).add(iid)
                elif event == "rl_pull_resend":
                    self.resends[step] = self.resends.get(step, 0) + 1

    def _span(self, isl: dict, r: dict) -> None:
        """Keep phase spans on the wall clock. ``start``/``end`` are the driver's
        monotonic seconds; the record's ``time_unix`` is written at ``end``."""
        task = SPAN_PHASES.get(r.get("task"))
        a, b, t = finite(r.get("start")), finite(r.get("end")), finite(r.get("time_unix"))
        if task is None or a is None or b is None or t is None:
            return
        off = t - b
        isl["spans"].append({"phase": task, "rollout_id": r.get("rollout_id"), "start": a + off, "end": b + off})

    def round_records(self, isl: dict) -> list[dict]:
        """Per-round record for the page (tasks 9.4): metrics + phase spans + time split.
        A publish span has no rollout_id: it belongs to the round whose sync it follows
        (it publishes policy rollout_id+1). Missing values stay None."""
        rounds: dict[int, dict] = {}
        pubs = sorted((s for s in isl["spans"] if s["phase"] == "P"), key=lambda s: s["start"])
        for sp in isl["spans"]:
            rid = sp["rollout_id"]
            if sp["phase"] != "P" and isinstance(rid, int):
                rounds.setdefault(rid, {})[sp["phase"]] = [sp["start"], sp["end"]]
        for rid, ph in rounds.items():
            after = max(v[1] for v in ph.values())
            nxt = [p for p in pubs if p["start"] >= after - 1]
            if nxt:
                ph["P"] = [nxt[0]["start"], nxt[0]["end"]]
        pts = isl["points"]
        ids = sorted(set(rounds) | {x - 1 for x in pts})
        out = []
        for rid in ids:
            pt, ph = pts.get(rid + 1, {}), rounds.get(rid, {})
            out.append({
                "round": rid, "policy_version": rid,
                "published_version": rid + 1 if (rid + 1) in isl["pubs"] else None,
                "reward": pt.get("reward"), "reward_std": pt.get("reward_std"),
                "p10": pt.get("reward_p10"), "p50": pt.get("reward_p50"), "p90": pt.get("reward_p90"),
                "trunc": pt.get("trunc"), "logprob_diff": pt.get("logprob_diff"),
                "resp_mean": pt.get("resp_len"), "resp_p95": pt.get("resp_p95"), "tok_s": pt.get("tok_s"),
                "grad_norm": pt.get("grad_norm"), "kl": pt.get("kl"),
                "phases": ph, "dur": {k: round(v[1] - v[0], 1) for k, v in ph.items()},
                "discarded_trainings": self.discarded_trainings(isl, rid),
            })
        return out

    @staticmethod
    def discarded_trainings(isl: dict, rid: int) -> int:
        """rl-resume-from-checkpoint 4.3: how many times round ``rid`` was trained by a
        launch whose result a later resume dropped (trained before a resume that went
        back to a cut at or below ``rid``). The page greys such trainings out."""
        n = 0
        for ts in isl["trained_ts"].get(rid, []):
            if ts is None:
                continue
            if any(x["ts"] is not None and x["ts"] > ts and isinstance(x["rollout_id"], int)
                   and x["rollout_id"] <= rid for x in isl["resumes"]):
                n += 1
        return n

    def resume_segments(self, isl: dict) -> list[dict]:
        """One segment per launch: start (driver start), first round after it, the
        startup overhead (start -> first trained round) and the resume it began with."""
        starts = [t for t in isl.get("driver_starts", []) if t is not None]
        trained = sorted(t for v in isl["trained_ts"].values() for t in v if t is not None)
        out = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else None
            first = next((t for t in trained if t >= start and (end is None or t < end)), None)
            resume = next((x for x in isl["resumes"] if x["ts"] is not None and x["ts"] >= start
                           and (end is None or x["ts"] < end)), None)
            out.append({"incarnation": i, "start_ts": start, "first_round_ts": first,
                        "startup_s": None if first is None else _r(first - start),
                        "resumed_at": None if resume is None else resume["rollout_id"],
                        "cut_id": None if resume is None else resume["cut_id"]})
        return out

    def node_series(self, isl: dict) -> dict:
        out = {}
        for rank, ser in isl["node_series"].items():
            rows = list(ser)
            step = max(1, len(rows) // NODE_SERIES_POINTS)
            out[rank] = rows[::step]
        return out

    def page_view(self, *, live: bool = False, now: float | None = None) -> dict:
        """Everything the redesigned page (section 9) draws, in one object."""
        now = self.now(live) if now is None else now
        ov = self.overview(live=live, now=now)
        isl_extra = {}
        for iid in self.island_ids():
            isl = self.islands[iid]
            starts = [e["ts"] for e in isl["recent"] if e["type"] == "rl_driver_start"]
            isl_extra[iid] = {
                "rounds": self.round_records(isl), "node_series": self.node_series(isl),
                "first_ts": isl["first_ts"], "ready_ts": isl["ready_ts"],
                "driver_start_ts": isl.get("driver_start_ts") or (starts[0] if starts else None),
                "cells": len(isl["cells"] or []),
                "transactions": len(isl["tx_order"]), "ray_embed": self.ray_embed(isl, self.island_ids().index(iid)),
            }
        usage = {"syncer": self.counts["syncer"] > 0, "journal": self.counts["journal"] > 0,
                 "cells": any(v["cells"] for v in isl_extra.values()),
                 "transactions": any(v["transactions"] for v in isl_extra.values())}
        return {"overview": ov, "islands": isl_extra, "rounds_syncer": self.rounds(), "usage": usage}

    def _round_metrics(self, isl: dict, r: dict, event: str) -> None:
        if event == "rl_local_round":
            x = r.get("local_round_id")
            if not isinstance(x, int):
                x = (r.get("rollout_id") + 1) if isinstance(r.get("rollout_id"), int) else None
        else:
            x = (r["rollout_id"] + 1) if isinstance(r.get("rollout_id"), int) else None
            if isinstance(r.get("rollout_id"), int):
                isl["rollout_id"] = r["rollout_id"]
        if x is None:
            return
        isl["round"] = x if isl["round"] is None else max(isl["round"], x)
        values = {key: first_finite(r, paths) for key, _label, paths in METRICS}
        values.update({key: first_finite(r, paths) for key, paths in EXTRA_SERIES})
        if values.get("tok_s") is None and isl["points"].get(x, {}).get("tok_s_derived") is not False \
                and isl["points"].get(x, {}).get("tok_s") is None:
            toks = finite(r.get("action_tokens"))
            secs = sum(v for v in (finite(r.get("rollout_seconds")), finite(r.get("train_seconds")))
                       if v is not None)
            if toks is not None and secs > 0:
                values["tok_s"] = toks / secs
                values["tok_s_derived"] = True
        elif values.get("tok_s") is not None:
            values["tok_s_derived"] = False  # measured tok_per_s wins over the derived estimate
        if nonfinite_seen(r, ("grad_norm", "train_metrics.grad_norm", "loss", "pg_loss")):
            isl["nonfinite"].append(x)
        self._point(isl, x, values)

    def _feed_syncer(self, r: dict, iid: str | None, ts: float | None) -> None:
        if r.get("event") == POLICY_SWEEP_LEDGER_EVENT:
            self.ledger = r
            return
        self.merges.append(r)
        for resp in r.get("responders") or []:
            if isinstance(resp, dict) and "id" in resp:
                isl = self.island(_iid(resp["id"]))
                if resp.get("staleness") is not None:
                    isl["staleness"] = resp.get("staleness")
                if resp.get("contribution") is not None:
                    isl["contribution"] = resp.get("contribution")
        for m in r.get("expected") or []:
            self.island(_iid(m))

    def _feed_journal(self, r: dict, iid: str | None, ts: float | None) -> None:
        if iid is None:
            return
        isl = self.island(iid)
        ts = finite(r.get("wall_time")) or ts
        self._touch(isl, None)
        kind = r.get("kind")
        tx = r.get("tx_id")
        if kind == "rl_cell_snapshot":  # 5.1 controller event journaled (preferred source)
            self._cell_snapshot(isl, r)
        elif kind == "rl_reconfig_phase":  # 5.2
            self._reconfig_phase(isl, r, ts)
        elif kind == "request":
            body = r.get("body") if isinstance(r.get("body"), dict) else {}
            t = self._tx(isl, tx)
            t.update({"request_id": r.get("request_id"), "kind": body.get("kind", "rollout-only"),
                      "requested_ts": ts})
        elif kind == "phase":
            phase = r.get("phase")
            if tx is None and r.get("scope") == "island":
                tx = "island-%s" % r.get("seq")
            if tx is not None and isl["transactions"].get(str(tx), {}).get("source") == "rl_reconfig_phase":
                return  # rl_reconfig_phase events own this transaction
            self._tx_phase(isl, tx, phase, ts, error=r.get("error"), scope=r.get("scope"),
                           request_id=r.get("request_id"))
            if phase == "RECOVERY_REQUIRED":
                isl["recovery_required"].append({"ts": ts, "source": "journal", "tx_id": tx,
                                                 "round": None, "error": r.get("error")})
        elif kind == "gpu_pool" and isinstance(r.get("roles"), dict):
            if isl["cells_source"] != "rl_cell_snapshot":
                isl["cells"] = [{"cell": uuid, "role": role, "gpus": 1,
                                 "state": "accepted" if r.get("accepted") else "rejected"}
                                for uuid, role in sorted(r["roles"].items())]
                isl["cells_source"] = "journal gpu_pool（派生）"
        elif kind == "node_lost" and isl["cells"]:
            for c in isl["cells"]:
                c.setdefault("note", "node_lost: " + str(r.get("error") or ""))
        elif tx is not None and kind in ("fork_op", "recovery", "watchdog_action", "drain_timeout"):
            t = self._tx(isl, tx)
            t["notes"].append({"ts": ts, "kind": kind})

    def _cell_snapshot(self, isl: dict, r: dict) -> None:
        cells = r.get("cells")
        if not isinstance(cells, list):
            return
        isl["cells"] = [{"cell": c.get("cell_id", c.get("id")), "role": c.get("role"),
                         "gpus": c.get("gpus"), "state": c.get("state", c.get("status")),
                         **{k: c[k] for k in ("node", "gpu_uuid", "config", "epoch")
                            if c.get(k) is not None}}
                        for c in cells if isinstance(c, dict)]
        if r.get("txn_id") is not None:
            isl["cells_txn"] = r.get("txn_id")
        isl["cells_source"] = "rl_cell_snapshot"

    def _reconfig_phase(self, isl: dict, r: dict, ts: float | None) -> None:
        tx = r.get("txn_id", r.get("tx_id"))
        if tx is None:
            return
        ts = finite(r.get("t")) or ts
        t = self._tx(isl, tx)
        if t.get("source") != "rl_reconfig_phase":  # drop journal-derived phases once
            t.update({"source": "rl_reconfig_phase", "phases": [], "result": None})
        t["source_config"], t["target_config"] = r.get("source"), r.get("target")
        t["expected_epoch"] = r.get("expected_epoch")
        reason = r.get("reason") or r.get("error")
        self._tx_phase(isl, tx, r.get("phase"), ts, result=r.get("result"), error=reason,
                       request_id=r.get("request_id"))
        if (r.get("result") or r.get("phase")) == "RECOVERY_REQUIRED" and not any(
                a.get("tx_id") == str(tx) for a in isl["recovery_required"]):
            isl["recovery_required"].append({"ts": ts, "source": "rl_reconfig_phase",
                                             "tx_id": str(tx), "round": None, "error": reason})

    def _tx(self, isl: dict, tx: Any) -> dict:
        key = str(tx)
        t = isl["transactions"].get(key)
        if t is None:
            t = isl["transactions"][key] = {"tx_id": key, "request_id": None, "kind": None,
                                            "phases": [], "result": None, "error": None,
                                            "requested_ts": None, "last_ts": None, "notes": [],
                                            "scope": None}
            isl["tx_order"].append(key)
        return t

    def _tx_phase(self, isl: dict, tx: Any, phase: Any, ts: float | None, **fields: Any) -> None:
        if tx is None or phase is None:
            return
        t = self._tx(isl, tx)
        t["phases"].append(str(phase))
        t["last_ts"] = ts
        for k in ("request_id", "scope"):
            if fields.get(k) is not None:
                t[k] = fields[k]
        if fields.get("error"):
            t["error"] = fields["error"]
        result = fields.get("result")
        if result in TERMINAL_TX_PHASES:
            t["result"] = result
        elif phase in TERMINAL_TX_PHASES:
            # COMMITTED is followed by SUCCEEDED; RECOVERY_REQUIRED overrides both.
            if t["result"] != "RECOVERY_REQUIRED":
                t["result"] = phase

    def _feed_fleet(self, r: dict, iid: str | None, ts: float | None) -> None:
        event = r["event"]
        self.fleet_records.append(r)
        if r.get("budget_usd") is not None and self.budget_usd is None:
            self.budget_usd = finite(r.get("budget_usd"))
        if event == "cost_tick":
            self.cost_ticks.append(r)
            return
        iid = iid or _iid(r.get("island"))
        if iid is None:
            return
        isl = self.island(iid)
        for key in ("cloud", "region", "gpu", "gpus", "price_key", "cpus", "memory_gib"):
            if r.get(key) is not None:
                isl[key] = r[key]
        if r.get("island") is not None:
            isl["name"] = r["island"]
        isl["fleet_state"] = event[len("island_"):]
        # wall clock accrues over ready -> lost/stop intervals (relaunches re-open one)
        if event == "island_ready" and ts is not None:
            if isl["ready_ts"] is None:
                isl["ready_ts"] = ts
            if isl["open_ts"] is None:
                isl["open_ts"] = ts
            isl["stop_ts"] = None
        elif event in ("island_lost", "island_stop") and ts is not None:
            if event == "island_lost":
                isl["lost_ts"] = ts
            else:
                isl["stop_ts"] = ts
            if isl["open_ts"] is not None:
                isl["closed_s"] += max(0.0, ts - isl["open_ts"])
                isl["open_ts"] = None

    def _feed_other(self, r: dict, iid: str | None, ts: float | None) -> None:
        if iid is not None:
            self._touch(self.island(iid), ts)

    # -- views ------------------------------------------------------------------
    def now(self, live: bool) -> float:
        if live:
            return time.time()
        return self.max_ts if self.max_ts is not None else time.time()

    def rounds(self) -> list[dict]:
        rows = []
        for rec in self.merges:
            step = rec.get("step")
            expected = [str(_iid(x)) for x in rec.get("expected") or []]
            responded = rec.get("responded")
            if responded is None:
                responded = [x.get("id") for x in rec.get("responders") or [] if isinstance(x, dict)]
            responded = [str(_iid(x)) for x in responded]
            pushed = sorted(self.pushes.get(step, set())) if isinstance(step, int) else []
            if not expected and pushed:
                expected = sorted(set(pushed) | set(self.islands))
            if not responded and pushed:
                responded = pushed
            if rec.get("missed_grace") is not None:
                missed = [str(_iid(x)) for x in rec["missed_grace"]]
            else:
                missed = sorted(set(expected) - set(responded))
            merge_s = finite(rec.get("sync/merge_seconds"))
            row = {
                "round": step, "fragment": rec.get("fragment"), "attempt": rec.get("attempt"),
                "expected": len(expected) if expected else finite(rec.get("sync/quorum")),
                "responded": len(responded) if responded or expected
                else finite(rec.get("sync/responders")),
                "expected_ids": expected, "responded_ids": responded, "missed": missed,
                "quorum_ms": finite(rec.get("quorum_ms")), "grace_ms": finite(rec.get("grace_ms")),
                "sync_ms": finite(rec.get("sync_ms")),
                "merge_ms": round(merge_s * 1000, 1) if merge_s is not None else None,
                "wall_ms": finite(rec.get("ms")), "gnorm": finite(rec.get("gnorm")),
                "resend": self.resends.get(step, 0) if isinstance(step, int) else 0,
                "pushed": pushed,
                "applied": sorted(self.applies.get(step, set())) if isinstance(step, int) else [],
                "published": sorted(self.publications.get(step, set())) if isinstance(step, int) else [],
                "derived": True,
            }
            row["bad"] = bool(row["missed"] or row["resend"])
            rows.append(row)
        return rows

    def _island_card(self, isl: dict, now: float) -> dict:
        pts = isl["points"]
        last = pts[max(pts)] if pts else {}
        last_any = max([t for t in (isl["last_event_ts"], isl["last_heartbeat_ts"]) if t is not None],
                       default=None)
        age = (now - last_any) if last_any is not None else None
        hb_age = (now - isl["last_heartbeat_ts"]) if isl["last_heartbeat_ts"] is not None else None
        res = isl["resource"] or {}
        starting = not isl["driver_started"] and not isl["finalized"] and isl["fleet_state"] not in ("lost", "stop")
        if starting and isl["ready_ts"] is not None:
            # A container that is still loading weights may have written nothing yet:
            # count the startup age from island_ready as well.
            ref = max(t for t in (last_any, isl["ready_ts"]) if t is not None)
            age = now - ref
        startup_s = (now - min(t for t in (isl["ready_ts"], isl["first_ts"]) if t is not None)
                     if starting and (isl["ready_ts"] is not None or isl["first_ts"] is not None) else None)
        stopped = self.stopped_by_us(isl)
        if stopped and not isl["finalized"]:
            status = "stopped"
            starting = False
        elif isl["recovery_required"] and not _recovered_after(isl):
            status = "recovery"
        elif isl["fleet_state"] == "lost":
            status = "lost"
        elif isl["finalized"] or isl["fleet_state"] == "stop":
            status = "done"
        elif starting and age is not None:
            status = "stale" if age > self.thresholds["startup_warn_s"] else "starting"
        elif age is not None and age > self.thresholds["heartbeat_warn_s"]:
            status = "stale"
        elif age is None:
            status = "unknown"
        else:
            status = "ok"
        return {
            "id": isl["id"], "name": isl["name"], "cloud": isl["cloud"], "region": isl["region"],
            "gpu": isl["gpu"], "gpus": isl["gpus"], "status": status, "finalized": bool(isl["finalized"]),
            "stopped_by_us": stopped, "starting": starting, "startup_s": _r(startup_s), "driver_started": isl["driver_started"],
            "nodes": [isl["nodes"][k] for k in sorted(isl["nodes"], key=lambda x: int(x) if x.isdigit() else 0)],
            "last_event_age_s": _r(age), "heartbeat_age_s": _r(hb_age),
            "heartbeat_seen": isl["heartbeat_seen"], "round": isl["round"],
            "startup_steps": {k: {"seconds": v["seconds"], "step_s": v["step_s"]}
                              for k, v in isl["startup_steps"].items()},
            "startup_step": isl["startup_step"],
            "startup_step_age_s": _r(now - (isl["startup_step_ts"] or isl["first_ts"]))
            if starting and (isl["startup_step_ts"] or isl["first_ts"]) is not None else None,
            "rollout_id": isl["rollout_id"], "policy_version": isl["policy_version"],
            "phase": isl["phase"], "staleness": isl["staleness"], "contribution": isl["contribution"],
            "gpu_util_pct": res.get("util_pct"), "mem_pct": res.get("mem_pct"),
            "resource_available": res.get("available") if res else None,
            "host_mem_peak_bytes": (isl["host"] or {}).get("peak_bytes"),
            "host_mem_current_bytes": (isl["host"] or {}).get("current_bytes"),
            "reward": last.get("reward"), "tok_s": last.get("tok_s"),
            "source_lost": isl["source_lost"],
            "resumes": list(isl["resumes"]), "cuts_saved": len(isl["cuts"]),
            "segments": self.resume_segments(isl),
        }

    def series(self, isl: dict) -> dict:
        out: dict[str, list] = {}
        xs = sorted(isl["points"])
        keys = [m[0] for m in METRICS] + [m[0] for m in EXTRA_SERIES]
        for key in keys:
            pts = [[x, isl["points"][x][key]] for x in xs if isl["points"][x].get(key) is not None]
            out[key] = pts
        return out

    def ray_embed(self, isl: dict, index: int) -> dict:
        cloud = (isl["cloud"] or "").lower()
        if not cloud:
            return {"mode": "unknown", "note": "无数据：fleet.jsonl 未记录该岛 cloud"}
        if cloud == "modal":
            return {"mode": "none", "note": "不可嵌入：Modal 容器无稳定入站端口"}
        if cloud in ("local", "本地"):
            return {"mode": "direct", "port": 8265, "command": None}
        port = 18265 + index
        host = isl["name"] or isl["id"]
        return {"mode": "tunnel", "port": port,
                "command": f"ssh -L {port}:localhost:8265 {host}"}

    def cost(self, now: float) -> dict:
        self.apply_run_host_shape()
        return cost_mod.cost_view(self, now)

    def overview(self, *, live: bool = False, now: float | None = None) -> dict:
        now = self.now(live) if now is None else now
        rounds = self.rounds()
        cost = self.cost(now)
        cards = [self._island_card(self.islands[k], now) for k in self.island_ids()]
        alerts = alerts_mod.evaluate(self, now=now, rounds=rounds, cost=cost, cards=cards)
        n_bad = sum(1 for c in cards if c["status"] in ("stale", "lost", "recovery"))
        sev0 = sum(1 for a in alerts if a["sev"] == 0)
        stop = self.operator_stop()
        if not cards:
            global_status = {"level": "muted", "text": "无数据"}
        elif stop and all(c["status"] in ("stopped", "done") for c in cards):
            global_status = {"level": "muted", "text": f"已停止（我方停机：{stop.get('cause')}）"}
        elif n_bad:
            global_status = {"level": "bad", "text": f"{n_bad} 岛异常"}
        elif sev0:
            global_status = {"level": "bad", "text": f"{sev0} 条严重告警"}
        elif alerts:
            global_status = {"level": "warn", "text": f"{len(alerts)} 条告警"}
        else:
            global_status = {"level": "ok", "text": "全部健康"}
        run, inferred = self.run_name()
        return {
            "run": run, "run_inferred": inferred, "operator_stop": stop, "run_kind": self.run_kind(),
            "mode": "live" if live else "offline", "now": now,
            "data_ts": self.max_ts, "first_ts": self.min_ts, "global_status": global_status,
            "alerts": alerts, "cost": cost, "islands": cards,
            "metrics": [[k, label] for k, label, _ in METRICS],
            "series": {c["id"]: self.series(self.islands[c["id"]]) for c in cards},
            "round_marks": [{"round": r["round"], "missed": bool(r["missed"]), "resend": r["resend"]}
                            for r in rounds],
            "counts": dict(self.counts), "events_total": self.event_seq,
        }

    def _run_meta(self) -> tuple:
        if self._run_meta_cache is None:
            from .sources import infer_run_name, run_meta_near

            meta = next((m for m in (run_meta_near(s) for s in self.sources_seen) if m), None)
            names = [isl["name"] for isl in self.islands.values() if isl["name"]]
            self._run_meta_cache = (meta, infer_run_name(self.sources_seen, names))
        return self._run_meta_cache

    def run_name(self) -> tuple[str | None, bool]:
        """(name, inferred): ``--run`` wins; otherwise inferred from the tapes (8.3)."""
        if self.run:
            return self.run, False
        name = self._run_meta()[1]
        return name, name is not None

    def run_kind(self) -> str:
        """single_island: no syncer tape and at most one island; else multi_island (D-UI2)."""
        return "single_island" if self.counts["syncer"] == 0 and len(self.islands) <= 1 else "multi_island"

    def apply_run_host_shape(self) -> None:
        """Fill Modal islands' missing cpus/memory_gib from the run's meta.json
        args (``--gpu modal:NxGxTYPE``, ``--modal-cpu``, ``--modal-memory-gib``;
        defaults per GPU as in modal_runner) for tapes written before
        island_ready carried them."""
        meta = self._run_meta()[0]
        if not meta:
            return
        shape = modal_host_shape(meta.get("args") or {})
        if not shape:
            return
        for isl in self.islands.values():
            if (isl["cloud"] or "").lower() == "modal" and isl["cpus"] is None and isl["memory_gib"] is None:
                isl["cpus"], isl["memory_gib"] = shape["cpus"], shape["memory_gib"]
                isl["host_shape_source"] = "meta.json"

    def island_ids(self) -> list[str]:
        def key(i: str):
            return (0, int(i), "") if i.lstrip("-").isdigit() else (1, 0, i)
        return sorted(self.islands, key=key)

    def island_view(self, iid: str, *, live: bool = False, now: float | None = None) -> dict | None:
        isl = self.islands.get(iid)
        if isl is None:
            return None
        now = self.now(live) if now is None else now
        index = self.island_ids().index(iid)
        txs = [isl["transactions"][k] for k in isl["tx_order"]]
        return {
            "card": self._island_card(isl, now), "series": self.series(isl),
            "resource": isl["resource"], "host": isl["host"], "cells": isl["cells"], "cells_source": isl["cells_source"],
            "transactions": txs, "reconfigurations": isl["reconfig"],
            "recovery_required": isl["recovery_required"],
            "recent_events": list(isl["recent"])[-20:], "events_by_type": isl["events_by_type"],
            "ray_embed": self.ray_embed(isl, index),
            "nonfinite_rounds": isl["nonfinite"],
        }

    def fleet_view(self, *, live: bool = False, now: float | None = None) -> dict:
        now = self.now(live) if now is None else now
        return {"records": self.fleet_records[-500:], "cost_ticks": self.cost_ticks[-500:],
                "prices_source": self.prices.get("_source"), "prices_note": self.prices.get("_note"),
                "cost": self.cost(now)}

    def events_view(self, *, island: str | None = None, type: str | None = None,
                    after: int = 0, limit: int = 200) -> dict:
        out = []
        for e in self.events:
            if e["seq"] <= after:
                continue
            if island is not None and e["island"] != island:
                continue
            if type is not None and e["type"] != type:
                continue
            out.append(e)
            if len(out) >= limit:
                break
        cursor = out[-1]["seq"] if out else after
        return {"events": out, "cursor": cursor, "more": bool(out) and len(out) >= limit}

    def full_view(self, *, live: bool = False, now: float | None = None, events_limit: int = 500) -> dict:
        now = self.now(live) if now is None else now
        evs = list(self.events)[-events_limit:]
        return {
            "overview": self.overview(live=live, now=now),
            "islands": {i: self.island_view(i, live=live, now=now) for i in self.island_ids()},
            "rounds": self.rounds(),
            "fleet": self.fleet_view(live=live, now=now),
            "page": self.page_view(live=live, now=now),
            "events": {"events": evs, "cursor": evs[-1]["seq"] if evs else 0, "more": False},
        }


def modal_host_shape(args: dict) -> dict | None:
    """Requested host cpus/memory for a ``--gpu modal:[N x]GxTYPE`` run."""
    import re

    m = re.fullmatch(r"modal:(?:(\d+)x)?(\d+)x([A-Za-z0-9!]+)", str(args.get("gpu") or ""))
    if not m:
        return None
    nodes, per = int(m.group(1) or 1), int(m.group(2))
    from ..modal_runner import MODAL_CPU_CORES_PER_GPU, MODAL_MEMORY_GIB_PER_GPU

    cpu = args.get("modal_cpu") or MODAL_CPU_CORES_PER_GPU * per
    mem = args.get("modal_memory_gib") or MODAL_MEMORY_GIB_PER_GPU * per
    return {"cpus": cpu * nodes, "memory_gib": mem * nodes}


def _recovered_after(isl: dict) -> bool:
    last = isl["recovery_required"][-1]["ts"] or 0
    return any(r.get("result") == "RECOVERED" and (r.get("ts") or 0) > last for r in isl["reconfig"])


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


def _resource(r: dict, ts: float | None) -> dict:
    if r.get("available") is False:
        return {"available": False, "ts": ts, "util_pct": None, "mem_pct": None, "gpus": []}
    gpus = [g for g in r.get("gpus") or [] if isinstance(g, dict)]
    utils = [finite(g.get("util_pct")) for g in gpus]
    utils = [u for u in utils if u is not None]
    used = sum(finite(g.get("mem_used_mb")) or 0 for g in gpus)
    total = sum(finite(g.get("mem_total_mb")) or 0 for g in gpus)
    return {"available": True, "ts": ts, "gpus": gpus,
            "util_pct": round(sum(utils) / len(utils), 1) if utils else None,
            "mem_pct": round(used / total * 100, 1) if total else None}


_SUMMARY_KEYS = ("phase", "rollout_id", "local_round_id", "policy_version", "reward_mean",
                 "grad_norm", "result", "global_step", "fragment_id", "error")


def _summary(r: dict) -> str:
    parts = []
    for k in _SUMMARY_KEYS:
        v = r.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            v = f"{v:.4g}"
        s = str(v)
        parts.append(f"{k}={s[:80]}")
    return " ".join(parts)
