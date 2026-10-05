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
        "price_key": None, "first_ts": None, "last_event_ts": None, "last_heartbeat_ts": None,
        "heartbeat_seen": False, "round": None, "rollout_id": None, "policy_version": None,
        "phase": None, "finalized": False, "fleet_state": None, "ready_ts": None, "stop_ts": None,
        "lost_ts": None, "points": {}, "nonfinite": [], "resource": None, "staleness": None,
        "contribution": None, "reconfig": [], "cells": None, "cells_source": None,
        "transactions": {}, "tx_order": [], "recovery_required": [], "source_lost": None,
        "recent": deque(maxlen=50), "events_by_type": {},
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

    # -- feeding ----------------------------------------------------------------
    def feed(self, record: Any, *, source: str = "-", offset: int | None = None,
             island: str | None = None) -> bool:
        """Fold one record. Returns False when skipped as already consumed."""
        if offset is not None:
            if offset <= self.offsets.get(source, -1):
                return False
            self.offsets[source] = offset
        if not isinstance(record, dict):
            return False
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
            iid = "?"
        isl = self.island(iid)
        self._touch(isl, ts)
        isl["events_by_type"][event] = isl["events_by_type"].get(event, 0) + 1
        isl["recent"].append({"ts": ts, "type": event, "summary": _summary(r)})
        if r.get("policy_version") is not None and event in (
                "rl_policy_apply", "rl_publication", "rl_driver_phase", "rl_heartbeat",
                "rl_round_cut", "rl_member_publication"):
            isl["policy_version"] = r.get("policy_version")
        if event == "rl_driver_phase":
            isl["phase"] = r.get("phase")
        elif event == "rl_heartbeat":
            isl["heartbeat_seen"] = True
            if ts is not None:
                isl["last_heartbeat_ts"] = max(ts, isl["last_heartbeat_ts"] or ts)
            isl["phase"] = r.get("phase", isl["phase"])
            if r.get("rollout_id") is not None:
                isl["rollout_id"] = r["rollout_id"]
        elif event == "rl_resource_sample":
            isl["resource"] = _resource(r, ts)
        elif event in ("rl_local_round", "rl_round_trained"):
            self._round_metrics(isl, r, event)
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
            cells = r.get("cells")
            if isinstance(cells, list):
                isl["cells"] = [{"cell": c.get("cell_id", c.get("id")), "role": c.get("role"),
                                 "gpus": c.get("gpus"), "state": c.get("state", c.get("status"))}
                                for c in cells if isinstance(c, dict)]
                isl["cells_source"] = "rl_cell_snapshot"
        elif event == "rl_reconfig_phase":
            self._tx_phase(isl, r.get("txn_id", r.get("tx_id")), r.get("phase"), ts,
                           result=r.get("result"), error=r.get("error"))
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
        if values.get("tok_s") is None:
            toks = finite(r.get("action_tokens"))
            secs = sum(v for v in (finite(r.get("rollout_seconds")), finite(r.get("train_seconds")))
                       if v is not None)
            if toks is not None and secs > 0:
                values["tok_s"] = toks / secs
                values["tok_s_derived"] = True
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
        iid = iid or "?"
        isl = self.island(iid)
        ts = finite(r.get("wall_time")) or ts
        self._touch(isl, None)
        kind = r.get("kind")
        tx = r.get("tx_id")
        if kind == "request":
            body = r.get("body") if isinstance(r.get("body"), dict) else {}
            t = self._tx(isl, tx)
            t.update({"request_id": r.get("request_id"), "kind": body.get("kind", "rollout-only"),
                      "requested_ts": ts})
        elif kind == "phase":
            phase = r.get("phase")
            if tx is None and r.get("scope") == "island":
                tx = "island-%s" % r.get("seq")
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
        iid = iid or _iid(r.get("island")) or "?"
        isl = self.island(iid)
        for key in ("cloud", "region", "gpu", "gpus", "price_key"):
            if r.get(key) is not None:
                isl[key] = r[key]
        if r.get("island") is not None:
            isl["name"] = r["island"]
        isl["fleet_state"] = event[len("island_"):]
        if event in ("island_ready", "island_launch") and ts is not None and isl["ready_ts"] is None:
            if event == "island_ready" or isl["ready_ts"] is None:
                isl["ready_ts"] = ts
        if event == "island_ready" and ts is not None:
            isl["stop_ts"] = None
        if event == "island_lost":
            isl["lost_ts"] = ts
        if event == "island_stop":
            isl["stop_ts"] = ts

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
        if isl["recovery_required"] and not _recovered_after(isl):
            status = "recovery"
        elif isl["fleet_state"] == "lost":
            status = "lost"
        elif isl["finalized"] or isl["fleet_state"] == "stop":
            status = "done"
        elif age is not None and age > self.thresholds["heartbeat_warn_s"]:
            status = "stale"
        elif age is None:
            status = "unknown"
        else:
            status = "ok"
        return {
            "id": isl["id"], "name": isl["name"], "cloud": isl["cloud"], "region": isl["region"],
            "gpu": isl["gpu"], "gpus": isl["gpus"], "status": status,
            "last_event_age_s": _r(age), "heartbeat_age_s": _r(hb_age),
            "heartbeat_seen": isl["heartbeat_seen"], "round": isl["round"],
            "rollout_id": isl["rollout_id"], "policy_version": isl["policy_version"],
            "phase": isl["phase"], "staleness": isl["staleness"], "contribution": isl["contribution"],
            "gpu_util_pct": res.get("util_pct"), "mem_pct": res.get("mem_pct"),
            "resource_available": res.get("available") if res else None,
            "reward": last.get("reward"), "tok_s": last.get("tok_s"),
            "source_lost": isl["source_lost"],
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
        return cost_mod.cost_view(self, now)

    def overview(self, *, live: bool = False, now: float | None = None) -> dict:
        now = self.now(live) if now is None else now
        rounds = self.rounds()
        cost = self.cost(now)
        cards = [self._island_card(self.islands[k], now) for k in self.island_ids()]
        alerts = alerts_mod.evaluate(self, now=now, rounds=rounds, cost=cost, cards=cards)
        n_bad = sum(1 for c in cards if c["status"] in ("stale", "lost", "recovery"))
        sev0 = sum(1 for a in alerts if a["sev"] == 0)
        if not cards:
            global_status = {"level": "muted", "text": "无数据"}
        elif n_bad:
            global_status = {"level": "bad", "text": f"{n_bad} 岛异常"}
        elif sev0:
            global_status = {"level": "bad", "text": f"{sev0} 条严重告警"}
        elif alerts:
            global_status = {"level": "warn", "text": f"{len(alerts)} 条告警"}
        else:
            global_status = {"level": "ok", "text": "全部健康"}
        return {
            "run": self.run, "mode": "live" if live else "offline", "now": now,
            "data_ts": self.max_ts, "first_ts": self.min_ts, "global_status": global_status,
            "alerts": alerts, "cost": cost, "islands": cards,
            "metrics": [[k, label] for k, label, _ in METRICS],
            "series": {c["id"]: self.series(self.islands[c["id"]]) for c in cards},
            "round_marks": [{"round": r["round"], "missed": bool(r["missed"]), "resend": r["resend"]}
                            for r in rounds],
            "counts": dict(self.counts), "events_total": self.event_seq,
        }

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
            "resource": isl["resource"], "cells": isl["cells"], "cells_source": isl["cells_source"],
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
            "events": {"events": evs, "cursor": evs[-1]["seq"] if evs else 0, "more": False},
        }


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
