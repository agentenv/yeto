"""Fold JSONL tapes into the dashboard views.

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
* breakers (any tape): ``rl_breaker``, ``rl_breaker_status`` and the registry's
  legacy trip events, folded by ``breakers.BreakerState`` (the Breakers panel)

Anything else is passed through to the events view. Missing values stay
``None`` (rendered as "no data"), never 0.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
import time
from collections import deque
from typing import Any, Iterable

from . import alerts as alerts_mod
from . import async_rl
from . import breakers as breakers_mod
from . import cost as cost_mod
from . import fleet_islands
from . import sft as sft_mod
from . import topology as topology_mod
from .sources import COST_EVENTS_FILE, up_state_stops

FLEET_EVENTS = frozenset({"island_launch", "island_ready", "island_lost", "island_stop", "island_done",
                          "cost_tick"})
POLICY_SWEEP_LEDGER_EVENT = "policy_sweep_ledger"
# the syncer's own records (``kind``, no ``step``): its config, round opens / timeouts and
# the island connections it saw; a ``learner_id`` there is not a sign of life of that island
SYNCER_RECORD_KINDS = frozenset({"syncer_config", "round_open", "round_attempt_timeout",
                                 "member_connected", "member_disconnected"})
SYNCER_RECORDS_MAX = 200_000  # a few per merge: a long run's rounds fit
# a DiLoCo trainer without heartbeats in the syncer's terminal phase: the last rounds wait up to
# the syncer's terminal grace for every island (when its config does not say how long)
TERMINAL_GRACE_DEFAULT_S = 120.0
SYNCER_ISLAND_EVENTS = frozenset({"rl_fragment_push", "rl_policy_apply", "rl_pull_resend",
                                  "rl_member_publication"})
TERMINAL_TX_PHASES = ("COMMITTED", "SUCCEEDED", "CANCELLED", "REBUILT_OLD", "RECOVERY_REQUIRED",
                      "FAILED")
MAX_EVENTS = 20000
# rl-spot-cost-saving 2.5: reclaim events and replacement advice shown as one table
SPOT_EVENTS = frozenset({"spot_reclaim", "spot_replace_advice", "spot_replace_launch", "spot_budget_cap"})
SPOT_MAX = 500
FLEET_VIEW_MAX = 500  # fleet records / cost ticks the fleet view shows (the last ones)
NO_DATA = None

# Overlay-chart metrics (v7 tabs): series key -> (label, extractor field paths).
# A path "a.b" reads record["a"]["b"]. The first finite value wins.
METRICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("reward", "reward mean", ("reward_mean", "rl/reward_mean")),
    ("reward_p10", "reward p10", ("reward_p10",)),
    ("reward_p90", "reward p90", ("reward_p90",)),
    ("pg_loss", "pg_loss", ("pg_loss", "train_metrics.pg_loss", "train/pg_loss")),
    # PPO KL (trainer policy vs the batch's old logprobs). ``mean_kl`` is ambiguous on
    # the tapes: rl_local_round carries the trainer's loss-dict KL there (ppo_kl), while
    # an async buffer round's rl_round_trained overwrites it with train_rollout_kl; the
    # explicit keys win and _round_metrics drops a ``mean_kl`` that is the train-rollout KL.
    ("kl", "PPO KL", ("train_metrics.ppo_kl", "train/ppo_kl", "ppo_kl", "train_metrics.kl_loss",
                      "train/kl_loss", "mean_kl")),
    # train-vs-rollout KL (trainer logprobs vs the sampler's rollout logprobs): explicit keys
    # only, so a record without them (rl_local_round) never overwrites it with ``mean_kl``
    ("kl_tr", "train-rollout KL", ("train_metrics.train_rollout_kl", "mismatch.train_rollout_kl",
                                   "train/train_rollout_kl")),
    ("tis_clip", "TIS clipfrac", ("mismatch.tis_clipfrac", "train_metrics.tis_clipfrac", "train/tis_clipfrac")),
    ("entropy", "entropy", ("entropy", "train_metrics.entropy", "train_metrics.entropy_loss")),
    ("grad_norm", "grad_norm", ("grad_norm", "train_metrics.grad_norm", "train/grad_norm")),
    ("clip", "clip total", ("clip_fraction", "train_metrics.pg_clipfrac", "train/pg_clipfrac")),
    ("clip_lo", "clip low side", ("train_metrics.pg_clipfrac_lower", "train_metrics.pg_clipfrac_low")),
    ("clip_hi", "clip high side", ("train_metrics.pg_clipfrac_upper", "train_metrics.pg_clipfrac_high")),
    ("resp_len", "response len", ("resp_len_mean",)),
    ("tok_s", "tok/s", ("tok_per_s",)),
    ("policy_age", "policy age mean", ("policy_age_mean",)),
    ("policy_age_max", "policy age max", ("policy_age_max",)),
    # score centering: the off-policy guard is the IS weight, not a PPO ratio clip
    ("is_clip", "IS clipped tokens", ("train_metrics.sc_is_clipped", "train/sc_is_clipped")),
    ("is_weight", "IS weight mean", ("train_metrics.sc_importance_weight", "train/sc_importance_weight")),
    ("tail_ratio", "tail ratio", ("train_metrics.sc_tail_ratio", "train/sc_tail_ratio")),
    # YETO_RL_NAN_SKIP: cumulative rounds skipped for a non-finite grad_norm (rl_round_trained)
    ("nan_skipped", "NaN-skipped rounds", ("nan_skipped",)),
)
EXTRA_SERIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loss", ("loss", "train_metrics.loss", "train/loss")),
    ("reward_std", ("reward_std",)),
    ("lr", ("lr", "applied_lr")),
    ("ess", ("ess_ratio", "train_metrics.ess_ratio")),
    ("trunc", ("truncated_frac",)),
    ("reward_p50", ("reward_p50",)),
    ("resp_p95", ("resp_len_p95",)),
    # full response length (generated + tool / env tokens); resp_len_mean counts loss-mask tokens
    ("resp_full", ("resp_full_len_mean",)),
    ("logprob_diff", ("train_metrics.train_rollout_logprob_abs_diff", "train/train_rollout_logprob_abs_diff")),
    ("train_step", ("train_step",)),
    ("step_s", ("step_seconds",)),
    # share of the round's groups whose rewards were all equal (zero advantage; rl_local_round)
    ("zero_var_ratio", ("zero_variance_group_ratio",)),
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


def unreported_metrics(backend: str | None) -> dict[str, str]:
    """``{metric: "not reported by <backend>"}`` for each round metric the island's backend
    does not report (``yeto.rl.engine.backend_identity.reported_metrics``); empty for a
    backend that reports them all or is not known."""
    if not backend:
        return {}
    from yeto.rl.engine.backend_identity import ROUND_METRICS, reported_metrics  # noqa: PLC0415

    return {m: f"not reported by {backend}" for m in sorted(ROUND_METRICS - reported_metrics(backend))}


def backend_label(backend: str | None) -> str | None:
    """The backend named on the island's card: one other than the default training
    backend (``yeto.rl.engine.backends.DEFAULT_BACKEND``); None for the default or unknown."""
    from yeto.rl.engine.backends import DEFAULT_BACKEND  # noqa: PLC0415

    return backend if backend and backend != DEFAULT_BACKEND else None


def ppo_kl(record: dict) -> float | None:
    """PPO KL of a trainer round record. ``mean_kl`` counts only when the record has no
    explicit KL key and it is not the record's train-rollout KL (the async driver writes
    train_rollout_kl into ``mean_kl`` of rl_round_trained)."""
    paths = dict((key, p) for key, _label, p in METRICS)
    explicit = first_finite(record, [p for p in paths["kl"] if p != "mean_kl"])
    if explicit is not None:
        return explicit
    mean = finite(record.get("mean_kl"))
    if mean is None or mean == first_finite(record, paths["kl_tr"]):
        return None
    return mean


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


def pool_record(record: dict) -> dict:
    """A pool event under its neutral name (``rl_pool_<x>``, the unified sampler pool's) read
    as ``nc_elastic_<x>``: the views match the legacy names, records of either kind fold alike."""
    event = record.get("event")
    if isinstance(event, str) and event.startswith(NEUTRAL_POOL_PREFIX):
        return {**record, "event": async_rl.POOL_PREFIX + event[len(NEUTRAL_POOL_PREFIX):]}
    return record


def classify(record: dict) -> str:
    event = record.get("event")
    if isinstance(event, str) and event.startswith("sft_"):
        return "sft"
    if isinstance(event, str) and event.startswith(async_rl.POOL_PREFIX):
        return "pool"
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
NEUTRAL_POOL_PREFIX = "rl_pool_"  # yeto.launch.pool.status.NEUTRAL_PREFIX (no import: the dashboard stands alone)
STOP_SLACK_S = 5.0
# a pool's release time (its host's clock) against a sampler's events (the sampler's host's clock)
RELEASE_SKEW_S = 60.0
NODE_SERIES_MAX = 20000
NODE_SERIES_POINTS = 300
# rl_timeline_span task -> page phase: R rollout / T training / S post-train sync / P publish
SPAN_PHASES = {"generate": "R", "train": "T", "outer_sync": "S", "publish": "P"}
# rl_driver_phase name -> rl_timeline_span task (derived spans)
DRIVER_PHASE_TASK = {"sync": "outer_sync"}
# rl_rollout_cutoff fields shown per round
CUTOFF_KEYS = ("submitted_groups", "target_groups", "filtered_groups", "discarded_groups",
               "discarded_trajectories", "discarded_tokens", "discarded_unknown_groups", "mechanism")
# rl_trajectory_reward phase fields -> page names (generation / tool / judge / sandbox start)
TRAJ_PHASES = (("gen", "generation_seconds"), ("tool", "tool_seconds"), ("judge", "evaluate_time"),
               ("sandbox", "sandbox_start_seconds"))
LOAD_KEYS = ("kv_used_tokens", "kv_capacity_tokens", "queued_requests", "running_requests",
             "tool_wait_trajectories")
UTIL_TRAJ_MAX = 400  # per-trajectory rows sent to the page (most recent)
# rows of a list the page draws only as a table (syncer merges, releases, per-round cutoff tallies,
# carried-over trajectories) that the page view sends: the newest ones; the run's totals are in
# ``table_totals`` (utilization: ``rounds_total`` / ``carry_total``). The page shows the newest 10 rows of a table by default; the chart
# sources (per-round records of each island) are not cut. ``/api/view?limit=`` overrides it.
VIEW_TABLE_MAX = 500
# Events only written once driver.run() is going (older tapes may lack rl_driver_start).
DRIVER_RUNNING_EVENTS = frozenset({"rl_driver_phase", "rl_heartbeat", "rl_local_round", "rl_round_trained",
                                   "rl_resource_sample", "rl_publication",
                                   # a sampler island runs no driver: its loop's events
                                   "rl_sampler_start", "rl_sampler_round"})


def _fmt_override_value(value: Any) -> str:
    return "n/a" if value is None else str(value)


def negative_test_label(overrides: Any) -> dict:
    """``rl_island_override`` -> the island badge: "Negative-test island: learning-rate schedule linear→constant"."""
    rows = [o for o in (overrides or []) if isinstance(o, dict)]
    parts = [f"{o.get('name') or o.get('key')} {_fmt_override_value(o.get('old'))}→"
             f"{_fmt_override_value(o.get('new'))}" for o in rows]
    return {"label": "Negative-test island: " + ", ".join(parts) if parts else "Negative-test island",
            "overrides": [{k: o.get(k) for k in ("key", "name", "old", "new")} for o in rows]}


# a tape a launch island uploads under a trainer lane (``store/trainer<k>:trainer/l<i>-…``)
_SUBSTORE_TRAINER_TAPE = re.compile(r"^store/trainer(\d+):trainer/")


def _iid(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _spot_row(r: dict) -> dict:
    """One row of the reclaim / replacement table (no derived numbers)."""
    cands = r.get("candidates") or []
    top = cands[0] if cands else {}
    return {"ts": record_ts(r), "event": r.get("event"), "island": r.get("island"),
            "cloud": r.get("cloud") or top.get("cloud"), "region": r.get("region") or top.get("region"),
            "source": r.get("source"), "remaining_s": r.get("remaining_s"), "saved": r.get("saved"),
            "outcome": r.get("outcome"), "candidates": len(cands) if "candidates" in r else None,
            "rejected": len(r.get("rejected") or []) if "rejected" in r else None,
            "auto_launch": r.get("auto_launch")}


def _new_island(iid: str) -> dict:
    return {
        "id": iid, "name": None, "cloud": None, "region": None, "gpu": None, "gpus": None,
        "price_key": None, "cpus": None, "memory_gib": None, "driver_started": False,
        "nodes": {}, "node_series": {}, "spans": [], "pubs": {}, "first_ts": None, "last_event_ts": None, "last_heartbeat_ts": None,
        "heartbeat_seen": False, "round": None, "rollout_id": None, "policy_version": None,
        "phase": None, "finalized": False, "run_end": None, "fleet_state": None, "ready_ts": None, "last_ready_ts": None, "stop_ts": None,
        "lost_ts": None, "open_ts": None, "closed_s": 0.0, "points": {}, "nonfinite": [], "resource": None, "host": None, "staleness": None,
        "contribution": None, "reconfig": [], "cells": None, "cells_source": None,
        "transactions": {}, "tx_order": [], "recovery_required": [], "source_lost": None,
        "recent": deque(maxlen=50), "events_by_type": {},
        # rl-resume-from-checkpoint 4.3: one entry per resume (launch boundary), the
        # rounds each launch trained (rid -> [ts...]) and the cuts saved
        "resumes": [], "trained_ts": {}, "cuts": [],

        # startup sub-steps {step: {"seconds", "step_s", "ts"}}
        "startup_steps": {}, "startup_step": None, "startup_step_ts": None,
        # cutoff per round, per-trajectory timing, load samples
        "cutoffs": {}, "trajs": [], "load": deque(maxlen=NODE_SERIES_MAX),
        # rl_trajectory_reward per round on the island's own tape: rid -> [reward sum, count]
        "traj_reward": {},
        # rl_driver_phase marks (wall clock): phase spans when the tape has no rl_timeline_span
        "phase_marks": [],
        # launch-preflight-guards 3.5: rl_island_override (negative-test island)
        "negative_test": None,
        # store tape launches (tapes/<island>/<launch>/: one per trainer restart) and
        # trainer_start_phase marks of the current launch
        "launches": {}, "launch": None, "launch_ts": None, "start_phase": None, "start_phase_ts": None,
    }


class Reducer:
    """Incremental tape -> view reducer. Not thread-safe; callers lock."""

    def __init__(self, *, run: str | None = None, prices: dict | None = None,
                 budget_usd: float | None = None, thresholds: dict | None = None):
        self.run = run
        self.prices = prices if prices is not None else cost_mod.load_prices(None)
        self.budget_usd = budget_usd
        self.budget_from_cli = budget_usd is not None
        self.thresholds = alerts_mod.merged_thresholds(thresholds)
        self.threshold_overrides = frozenset(thresholds or ())  # --thresholds keys: never replaced
        self.offsets: dict[str, int] = {}
        self.islands: dict[str, dict] = {}
        self.merges: list[dict] = []
        # the syncer's own records (SYNCER_RECORD_KINDS) with ``_ts`` / ``_source``
        self.syncer_records: list[dict] = []
        # each trainer's rl_engine_selected rl/outer_sync (a DiLoCo island, not a no-sync one)
        self.outer_sync: dict[str, bool] = {}
        self.ledger: dict | None = None
        self.pushes: dict[int, set[str]] = {}
        self.resends: dict[int, int] = {}
        self.applies: dict[int, set[str]] = {}
        self.publications: dict[int, set[str]] = {}
        self.fleet_records: deque = deque(maxlen=FLEET_VIEW_MAX)  # the view shows the last ones
        self.spot: deque = deque(maxlen=SPOT_MAX)
        self.cost_ticks: deque = deque(maxlen=FLEET_VIEW_MAX)
        # island -> its timed fleet.jsonl lifecycle records [(time, seq, event)], in time order:
        # an older record that arrives later (another host's store copy) is replayed in place
        self.fleet_lifecycle: dict[str, list[tuple[float, int, str]]] = {}
        self.fleet_shape_ts: dict[str, dict[str, float]] = {}  # island -> shape field -> its record's time
        self.events: deque = deque(maxlen=MAX_EVENTS)
        self.event_seq = 0
        self.max_ts: float | None = None
        self.min_ts: float | None = None
        self.counts = {"learner": 0, "syncer": 0, "journal": 0, "fleet": 0, "other": 0}
        self.sources_seen: list[str] = []
        self.operator_stops: list[dict] = []  # our own stop of the run (sources.operator_stops_near)
        self._run_meta_cache: tuple | None = None
        self.label: str | None = None  # display label when compared (7.6)
        self.compare: list["Reducer"] = []  #
        self.sft = sft_mod.SftState()  # ms-swift SFT islands + syncer side streams (sft.py)
        self._source_island: dict[str, str] = {}  # tape path -> the island its records name
        self.async_ = async_rl.AsyncState()  # samplers / buffer / releases / NC pool (async_rl.py)
        self.breakers = breakers_mod.BreakerState()  # resolved settings, status snapshots, trips (breakers.py)
        self._superseded = False  # the record being fed is from a superseded trainer launch
        # set by serve / export when --store is given: () -> StoreTapes.status() (store_tapes.py)
        self.store_status: Any = None
        self._seen_digests: set[bytes] = set()  # timestamped events folded so far (_digest)
        self._launch_owner: dict[bytes, str] = {}  # store record -> the trainer launch that carried it first
        self.duplicates = 0  # records dropped as already folded from another source
        self._source: str = "-"  # the source of the record being fed
        self._source_inc: dict[str, str] = {}  # source -> sampler incarnation its last join named
        self._source_pre: dict[str, float] = {}  # source -> earliest record time before its first join
        # the dashboard's own command line, for the page's "Commands" box (cli.page_commands)
        self.commands: dict[str, str] | None = None
        # <store>/fleet/plan.json of a fleet of async islands (fleet_islands.PlanFile); None otherwise
        self.fleet_plan: dict | None = None
        self.fleet_plan_source: str | None = None
        self.fleet_plan_seq = 0  # bumped by every plan change (the cached views key on it)
        # the fleet's live membership: pools fleet apply added / removed, retired shards and their
        # samplers' leaves (fleet_islands.read_fleet_state); None until the store has one
        self.fleet_state: dict | None = None
        # the host-wide syncer tape's records of another run, counted by sources.TapeSource: {path: ...}
        self.foreign_tapes: dict[str, dict] = {}

    # -- feeding ----------------------------------------------------------------
    def feed(self, record: Any, *, source: str = "-", offset: int | None = None,
             island: str | None = None, node: str | None = None, island_fixed: bool = False) -> bool:
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
        record = pool_record(record)
        kind = classify(record)
        iid = (_iid(island) if island_fixed and island is not None and kind == "learner" else None) \
            or _iid(record.get("island_id", record.get("learner_id"))) or island \
            or (_iid(record.get("sampler_id")) if kind == "learner" else None)
        if kind == "learner" and self.fleet_plan:
            sub = _SUBSTORE_TRAINER_TAPE.search(source)
            if sub is not None:  # a Miles sampler island's tape uploaded under its trainer's lane: its launch island
                iid = self._plan_miles_island(int(sub.group(1))) or iid
        digest = self._digest(record, iid, kind)
        duplicate = digest is not None and digest in self._seen_digests
        launch = _store_launch(source) if kind == "learner" and iid is not None else None
        if launch is not None and digest is not None:
            # a store record dates the launch that carried it first; the same record in a later
            # launch is history a restarted trainer uploaded again and must not date it
            owner = self._launch_owner.setdefault(digest, launch)
            if duplicate:  # folded already (often from the echoed tape): count once, still date
                if owner == launch:
                    self._launch(self.island(iid), launch,
                                 record_ts(record) if kind != "pool" else async_rl.pool_ts(record))
                else:  # the launch exists, so an older leave is attributed to the older launch
                    self.island(iid)["launches"].setdefault(launch, {"launch": launch, "first_ts": None,
                                                                     "last_ts": None})
        if duplicate:
            self.duplicates += 1
            return False
        if digest is not None:
            self._seen_digests.add(digest)
        if record.get("event") in SPOT_EVENTS:
            self.spot.append(_spot_row(record))
        self.counts[kind] = self.counts.get(kind, 0) + 1
        ts = record_ts(record) if kind != "pool" else async_rl.pool_ts(record)
        self.breakers.fold(record, iid, ts)
        if ts is not None:
            self.max_ts = ts if self.max_ts is None else max(self.max_ts, ts)
            self.min_ts = ts if self.min_ts is None else min(self.min_ts, ts)
        if record.get("event") in ("dashboard_source_lost", "dashboard_source_restored") and source.endswith(
                ".source.jsonl"):
            # a mirror's status file names the island by the mirror's --island label; the
            # records of the mirrored tape (the same path without .source.jsonl) name it
            iid = self._source_island.get(source[: -len(".source.jsonl")], iid)
        elif kind == "learner" and iid is not None and source != "-":
            self._source_island[source] = iid
        self._superseded = False
        self._source = source
        launch = _store_launch(source)
        if kind == "learner" and iid is not None and launch is not None:
            self._superseded = self._launch(self.island(iid), launch, ts)
        if kind != "sft" and sft_mod.is_syncer_tape(record):
            # The syncer writes the same tape for RL and SFT runs: the SFT page
            # reads it too, but it alone does not make a run an SFT run.
            self.sft.fold(record, ts)
        if record.get("event") == "rl_engine_selected" and iid is not None \
                and isinstance(record.get("rl/outer_sync"), bool):
            self.outer_sync[iid] = record["rl/outer_sync"]
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

    def _released_version(self, iid: str) -> int | None:
        """The newest version this trainer released (``rl_policy_released``): with a release
        every N rounds it trails the trainer's own (local) ``policy_version``."""
        rel = [x for x in self.async_.releases if str(x.get("trainer")) == str(iid)
               and isinstance(x.get("policy_version"), int)]
        return max(rel, key=lambda x: (x.get("ts") or 0, x["policy_version"]))["policy_version"] if rel else None

    def run_finish(self) -> dict | None:
        """The run ended on its own: every trainer (every island when no role is known) that
        wrote ``rl_run_end`` -- its process exits 0 at ``data_ended`` / ``total_steps`` /
        ``stop_after_rounds`` / ``stopped`` (the sync ended it). None while one has not."""
        ids = [i for i, role in self.async_.roles.items() if role == "trainer" and i in self.islands] \
            or [i for i, isl in self.islands.items() if isl.get("run_end")]
        ends = [self.islands[i].get("run_end") for i in ids]
        if not ends or any(e is None for e in ends):
            return None
        last = max(ends, key=lambda e: e.get("ts") or 0)
        reasons = sorted({str(e.get("reason") or "ended") for e in ends})
        return {"reason": " / ".join(reasons), "source": last.get("source"), "ts": last.get("ts")}

    def active_stops(self) -> list[dict]:
        """Our stops that end a run on purpose: the stop markers (gate / early / watchdog),
        a ``run up`` breaker's stop (``stop_reason`` in its state file, or an ``rl_breaker``
        stop trip). ``teardown.json`` is not one: the launcher writes it after a normal finish
        as well (see :meth:`run_end_status`)."""
        out = [dict(x, kind="marker") for x in self.operator_stops if x.get("cause") != "teardown"]
        for t in self.breakers.trips:
            if t.get("state") == "trip" and t.get("stop_reason"):
                out.append({"cause": t["stop_reason"], "time_unix": t.get("ts"), "marker": "rl_breaker " + str(t.get("name")),
                            "kind": "breaker"})
        out += up_state_stops(getattr(self, "up_state", None))
        return sorted(out, key=lambda x: x.get("time_unix") or 0)

    def run_end_status(self) -> dict | None:
        """How the run ended, once it has: ``{"kind": "stopped", "stop": ...}`` when we stopped
        it (an active stop not later than the trainers' normal end, or a teardown without one),
        ``{"kind": "finished", "finish": ...}`` when the trainers ended on their own (a teardown
        after that is the launcher cleaning up), else None."""
        finish = self.run_finish()
        stops = self.active_stops()
        if finish is not None:
            end = finish.get("ts")
            early = [x for x in stops if end is None or x.get("time_unix") is None
                     or x["time_unix"] <= end + STOP_SLACK_S]
            return {"kind": "stopped", "stop": early[0]} if early else {"kind": "finished", "finish": finish}
        if stops:
            return {"kind": "stopped", "stop": stops[0]}
        stop = self.operator_stop()
        return {"kind": "stopped", "stop": stop} if stop else None

    def stopped_by_us(self, isl: dict) -> bool:
        """True when we issued a stop and the island's first failure (RECOVERY_REQUIRED
        or fleet island_lost) is not earlier than it (5 s slack for clock skew between
        this machine and the container): the failure is a consequence of our stop."""
        stop = self.operator_stop()
        if stop is None or stop.get("time_unix") is None:
            return False
        if stop.get("cause") == "teardown" and (self.run_end_status() or {}).get("kind") == "finished":
            return False  # the launcher's cleanup after the trainers ended on their own
        if stop.get("cause") == "teardown" and (isl["last_event_ts"] or 0) > stop["time_unix"] + 60:
            return False  # an older launch's teardown.json: this island ran after it
        if stop.get("cause") == "teardown" and (isl["last_ready_ts"] or 0) > stop["time_unix"] + STOP_SLACK_S:
            return False  # relaunched after that teardown: the new container is up, its tapes may not be yet
        fails = [x["ts"] for x in isl["recovery_required"] if x.get("ts") is not None]
        if isl["lost_ts"] is not None:
            fails.append(isl["lost_ts"])
        return not fails or min(fails) >= stop["time_unix"] - STOP_SLACK_S

    def feed_breaker(self, record: Any, *, source: str, offset: int | None = None) -> bool:
        """Fold a record of a tape that feeds only the Breakers panel (the cost guard's
        ``cost-events.jsonl``, the ``run up`` monitor's ``events.jsonl``: their other
        records, e.g. the guard's own ``cost_tick``, are not the fleet's)."""
        if offset is not None:
            if offset <= self.offsets.get(source, -1):
                return False
            self.offsets[source] = offset
        if not isinstance(record, dict):
            return False
        ts = record_ts(record)
        if not self.breakers.fold(record, None, ts, cost_guard=source.endswith(COST_EVENTS_FILE)):
            return False
        self.event_seq += 1
        self.events.append({"seq": self.event_seq, "ts": ts, "island": None, "type": record.get("event"),
                            "source": source, "stream": "other", "record": record})
        return True

    def load_resolved(self, doc: Any, source: str) -> bool:
        """Take a run's ``breakers.resolved.json`` (``breakers.ResolvedFile``). When the
        run config's ``breakers:`` section sets ``run.budget.usd`` / ``warn_ratio``, they
        are the page's budget (unless ``--budget``) and its budget warn line (unless
        ``--thresholds`` sets ``budget_warn_pct``); a ``--budget`` that differs is shown as
        a mismatch. Without a section nothing changes."""
        if not self.breakers.load_resolved(doc, source):
            return False
        usd = finite(self.breakers.section_param("run.budget", "usd"))
        if usd is not None and not self.budget_from_cli:
            self.budget_usd = usd
        warn = finite(self.breakers.section_param("run.budget", "warn_ratio"))
        if warn is not None and "budget_warn_pct" not in self.threshold_overrides:
            self.thresholds["budget_warn_pct"] = round(warn * 100.0, 6)
        return True

    def _plan_miles_island(self, learner: int) -> str | None:
        """The launch island of the Miles sampler that feeds trainer ``learner`` (the plan's
        first; a lane's own tapes name its launch-local island 0)."""
        for isl in (self.fleet_plan or {}).get("islands") or ():
            if isl["learner"] == learner and isl.get("miles_samplers"):
                return str(isl["miles_samplers"][0]["launch_island"])
        return None

    def load_fleet_plan(self, doc: Any, source: str) -> bool:
        """Take a fleet's ``fleet/plan.json`` (``fleet_islands.PlanFile``); False when ``doc``
        is not a fleet plan or did not change."""
        plan = fleet_islands.load_plan(doc)
        if plan is None or plan == self.fleet_plan:
            return False
        self.fleet_plan, self.fleet_plan_source = plan, source
        self.fleet_plan_seq += 1
        return True

    def load_fleet_state(self, state: Any) -> bool:
        """Take the fleet's live membership (``fleet_islands.read_fleet_state``); False when it
        did not change."""
        if not isinstance(state, dict) or state == self.fleet_state:
            return False
        self.fleet_state = state
        self.fleet_plan_seq += 1
        return True

    def _trainer_island_now(self, isl: dict) -> dict | None:
        """The record of trainer island ``isl`` (``fleet_islands.island_state`` of its records under
        ``fleet/trainers/<k>/``: learner k is the card's id), with its gpu, price and moves."""
        trainers = (self.fleet_state or {}).get("trainers") or {}
        k = int(isl["id"]) if str(isl["id"]).isdigit() else None
        row = trainers.get(k) if k is not None else None
        if not row or not row.get("now"):
            return None
        rec = row.get("record") or {}
        stop = rec.get("stop") or {}
        # since when it has been in this state: its lifecycle state, its launch's start or the STOP
        # that began a move, whichever is newest
        marks = [t for t in (rec.get("since"), rec.get("started_at"), stop.get("at")) if t is not None]
        return {**row["now"], "since": rec.get("since"), "state_since": max(marks) if marks else None,
                "gpu": rec.get("gpu") or row.get("gpu"),
                "usd_per_h": rec.get("usd_per_h") or row.get("usd_per_h"), "incarnation": rec.get("incarnation"),
                "relaunches": rec.get("relaunches"), "moves": row.get("moves") or [], "name": row.get("name"),
                "prefix": rec.get("prefix") or row.get("prefix")}

    def _membership_left(self, isl: dict, now: float | None = None) -> dict | None:
        """Sampler ``isl`` ended by the fleet's membership (``fleet_state``): the leave its member
        record names (written by the sampler, or by its pool for a removed pool's sampler that
        wrote none), else its shard retired by its pool's member table (the pool was removed or
        scaled it down). None when the membership does not end it.

        As the tape's pool exits (:meth:`AsyncRL.pool_exit`), a leave is believed only when the
        sampler wrote nothing after it (``STOP_SLACK_S``) or its pool released its machine
        (``member_state`` gone): a sampler still writing after its shard retired, or whose
        machine the pool still holds (draining, or a release that failed: still billed), runs.
        Such a leave comes back with ``draining`` set (the card shows it; not left)."""
        rec = ((self.fleet_state or {}).get("shards") or {}).get(isl["id"])
        if not rec:
            return None
        cur = (self.async_.sampler.get(isl["id"]) or {}).get("incarnation")
        out = None
        if rec.get("left_at") is not None and (rec.get("incarnation") is None or cur is None
                                               or rec["incarnation"] == cur):
            reason = rec.get("leave_reason") or "left"
            by = "its pool" if rec.get("left_by") == "pool" else "the sampler"
            out = {"ts": rec["left_at"], "reason": reason, "source": "member record",
                   "detail": f"leave {reason} written by {by} in its member record (pool {rec.get('pool')})",
                   "done": reason in async_rl.SAMPLER_DONE_REASONS, "pool": rec.get("pool")}
        elif rec.get("stopped") and rec.get("member_state") == fleet_islands.GONE:
            pool = rec.get("pool")
            out = {"ts": rec.get("released_at"), "reason": "pool stopped", "source": "pool state", "done": False,
                   "pool": pool, "detail": f"pool {pool} stopped ({rec['stopped']})"}
        elif rec.get("retired"):
            pool = rec.get("pool")
            row = ((self.fleet_state or {}).get("pools") or {}).get(pool) or {}
            removed = row.get("removed_at") is not None
            out = {"ts": rec.get("retired_at"), "reason": "pool removed" if removed else "retired",
                   "source": "pool member table", "done": False, "pool": pool,
                   "detail": f"shard retired by pool {pool}'s member table"
                             + (" (fleet apply removed the pool)" if removed else " (scaled down)")}
        if out is None:
            return None
        member = rec.get("member_state")
        last = isl["last_event_ts"]
        wrote_after = last is not None and out["ts"] is not None and last > out["ts"] + STOP_SLACK_S
        held = member is not None and member != fleet_islands.GONE  # the pool still holds (bills) its machine
        released = member == fleet_islands.GONE
        # released, yet the sampler still writes now, after the release: a copy runs on, or the release
        # did not take; it may be billed. The release time is the pool's clock and the sampler's last
        # event its own: compared with RELEASE_SKEW_S; and only a sampler writing now counts, so a
        # skewed last round of a sampler that did stop clears once it is quiet (then: left)
        rel = rec.get("released_at")
        writing_now = now is not None and now - last <= self.thresholds["heartbeat_warn_s"] if last is not None else False
        active = released and wrote_after and writing_now and (rel is None or last > rel + RELEASE_SKEW_S)
        if not (wrote_after or held) or (released and not active):
            if released:
                out["detail"] += "; its pool released the machine"
            return out
        out["draining"] = True
        out["member_state"] = member
        out["wrote_after"] = wrote_after
        out["released"] = bool(active)
        if active and rel is not None:
            out["ts"] = rel  # alerted from the release on
        out["detail"] += ("; its pool released the machine, but the sampler still writes (pool gone but "
                          "sampler still active)" if active else ("; the sampler still writes after it"
                                                                 if wrote_after else "")
                          + (f"; its pool still holds the machine ({member}, billed)" if held else ""))
        return out

    def guard_drop_rates(self) -> dict[str, dict]:
        """Data-dropping guards' share of the data dropped over their recent rounds
        (``BreakerState.drop_rates``, window and minimum from the monitor thresholds)."""
        th = self.thresholds
        return self.breakers.drop_rates(int(th["guard_drop_window_rounds"]), th["guard_drop_min_groups"])

    def breakers_view(self, cost: dict | None = None) -> dict | None:
        """The Breakers panel; ``cost``: the page's cost view, whose estimate is the
        ``run.budget`` gauge when no process reports one."""
        gauges = {}
        total, cap = finite((cost or {}).get("total_usd")), finite((cost or {}).get("budget_usd"))
        if total is not None and cap:
            gauges["run.budget"] = {"value": total, "limit": cap, "unit": "usd", "ratio": round(total / cap, 4),
                                    "gauge": "level", "detail": "dashboard estimate, not billing data",
                                    "scope": None, "scopes": 1, "ts": None}
        th = self.thresholds
        view = self.breakers.view(gauges, self.guard_drop_rates(), th["guard_drop_alert_ratio"],
                                  int(th["guard_drop_min_rounds"]))
        if view is None:
            return None
        usd = finite(self.breakers.section_param("run.budget", "usd"))
        if self.budget_from_cli and usd is not None and self.budget_usd != usd:
            view["budget_mismatch"] = {"cli": self.budget_usd, "resolved": usd}
        return view

    def feed_many(self, records: Iterable[dict], **kw: Any) -> None:
        for r in records:
            self.feed(r, **kw)

    def _launch(self, isl: dict, launch: str, ts: float | None) -> bool:
        """Note a record of store tape ``launch`` (``tapes/<island>/<launch>/``). A launch
        that starts after the current one is a restart: the island's startup state starts
        over. Returns True for a record of a superseded (older) launch, which must not
        move the startup state of the current one."""
        row = isl["launches"].get(launch)
        if row is None:
            row = isl["launches"][launch] = {"launch": launch, "first_ts": ts, "last_ts": ts}
        elif ts is not None:
            row["first_ts"] = ts if row["first_ts"] is None else min(row["first_ts"], ts)
            row["last_ts"] = ts if row["last_ts"] is None else max(row["last_ts"], ts)
        cur = isl["launches"].get(isl["launch"]) if isl["launch"] is not None else None
        if cur is None:
            isl["launch"], isl["launch_ts"] = launch, row["first_ts"]
            if any(k != launch for k in isl["launches"]):
                # an earlier launch is known only from records the echoed tape folded first
                # (feed: duplicates): this one is its restart
                _restart(isl, row["first_ts"])
            return False
        if launch == isl["launch"]:
            return False
        if row["first_ts"] is not None and (cur["first_ts"] is None or row["first_ts"] > cur["first_ts"]):
            isl["launch"] = launch  # a restart: a new launch of the same island
            _restart(isl, row["first_ts"])
            return False
        return True

    def _digest(self, record: dict, iid: str | None, kind: str | None = None) -> bytes | None:
        """Identity of a timestamped event of island ``iid``, None for any other record.
        One record reaches the dashboard more than once: the launcher's echoed tape
        (``~/.yeto/runs/<run>/events``) and the trainer's store upload carry the same
        records, and a restarted trainer uploads its whole local tape again under its new
        launch; ``feed`` folds it once. Records without ``time_unix`` (which may repeat
        legitimately) have no identity; equal records of two islands (whose tapes name the
        island by file name only) are two records. The syncer's merge and pool records (no
        ``event``) have one too: the local syncer tape and its store upload carry the same ones."""
        syncer = kind == "syncer" or sft_mod.is_syncer_tape(record)
        if not (isinstance(record.get("event"), str) or syncer) or not isinstance(record.get("time_unix"), (int, float)):
            return None
        try:
            text = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return None
        return hashlib.blake2b(f"{iid}\x00{text}".encode("utf-8", "surrogatepass"), digest_size=16).digest()

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
        if values.get("reward") is not None:
            pt.pop("reward_src", None)  # a reported reward_mean wins over the trajectory mean
        for k, v in values.items():
            if v is not None:
                pt[k] = v
        if pt.get("kl") is not None and pt.get("kl") == pt.get("kl_tr"):
            # the round's other record (rl_local_round) carries the train-rollout KL in
            # ``mean_kl`` without naming it: it is not the PPO KL (either arrival order)
            pt.pop("kl")

    def _feed_learner(self, r: dict, iid: str | None, ts: float | None) -> None:
        event = r["event"]
        if iid is None:
            return  # no island identity: never fabricate a "?" island
        isl = self.island(iid)
        if not event.startswith("dashboard_"):  # the mirror's own status is not island activity
            self._touch(isl, ts)
        isl["events_by_type"][event] = isl["events_by_type"].get(event, 0) + 1
        isl["recent"].append({"ts": ts, "type": event, "summary": _summary(r)})
        if event in ("rl_sampler_join", "rl_sampler_start") and r.get("incarnation") is not None \
                and str(r["incarnation"]) != (self.async_.sampler.get(iid) or {}).get("incarnation"):
            isl["finalized"] = False  # a new sampler incarnation: an earlier one's end is not this one's
        # a sampler tape names its incarnation in rl_sampler_join; the records before it (the
        # engine start, the model load) belong to that incarnation too: its boot time
        src, boot = self._source, None
        if src != "-":
            if event in ("rl_sampler_join", "rl_sampler_start") and r.get("incarnation") is not None:
                self._source_inc[src] = str(r["incarnation"])
                boot = self._source_pre.pop(src, None)
            elif src not in self._source_inc and ts is not None and r.get("incarnation") is None:
                self._source_pre[src] = min(ts, self._source_pre.get(src, ts))
        self.async_.fold(r, iid, ts, self._source_inc.get(src, "" if src != "-" else None), boot)
        if r.get("policy_version") is not None and event in (
                "rl_policy_apply", "rl_publication", "rl_driver_phase", "rl_heartbeat",
                "rl_round_cut", "rl_member_publication", "rl_resume",
                # a sampler's policy is the version it serves
                "rl_sampler_round", "rl_sampler_policy_loaded", "rl_sampler_policy_adopted"):
            # the newest record wins, whatever order the tapes are read in (a run store lists a
            # trainer's launch folders and an echo directory its rotated files by name)
            if ts is None or isl.get("policy_version_ts") is None or ts >= isl["policy_version_ts"]:
                isl["policy_version"] = r.get("policy_version")
                isl["policy_version_ts"] = ts
        if event == "rl_island_hardware":  # decoupling 7.7: recorded, never a refusal reason
            isl["hardware"] = {k: r.get(k) for k in ("compat_group", "driver_version", "cuda_version")}
        if getattr(self, "_superseded", False) and (
                event in ("rl_driver_start", "rl_startup_step", "trainer_start_phase", "rl_heartbeat")
                or event in DRIVER_RUNNING_EVENTS):
            return  # a stopped launch's late records: not this island's current state
        if event == "remote_code_prewarm" and isinstance(r.get("nodes"), list):
            isl["nodes_seen"] = max(len(r["nodes"]), isl.get("nodes_seen") or 0)
        if event == "trainer_start_phase" and r.get("phase") is not None:
            isl["start_phase"] = str(r.get("phase"))
            if ts is not None:
                isl["start_phase_ts"] = ts
        if event == "rl_driver_start" and isl.get("driver_start_ts") is None:
            isl["driver_start_ts"] = ts
        if event == "rl_driver_start":
            isl.setdefault("driver_starts", []).append(ts)
        if event == "rl_driver_start" or (event in DRIVER_RUNNING_EVENTS and r.get("phase") != "startup"):
            isl["driver_started"] = True
        if event == "rl_driver_phase" and (ts is None or isl.get("phase_ts") is None or ts >= isl["phase_ts"]):
            isl["phase"], isl["phase_ts"] = r.get("phase"), ts
            if ts is not None and len(isl["phase_marks"]) < NODE_SERIES_MAX:
                isl["phase_marks"].append((ts, r.get("phase"), r.get("rollout_id")))
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
            gl = [g for g in (r.get("gpus") or []) if isinstance(g, dict)]
            if gl:
                isl["gpus_per_node"] = max(len(gl), isl.get("gpus_per_node") or 0)
                isl["gpu_mem_total_mb"] = finite(gl[0].get("mem_total_mb")) or isl.get("gpu_mem_total_mb")
            gm = [finite(g.get("mem_used_mb")) for g in isl["resource"]["gpus"]]
            gm = [g for g in gm if g is not None]
            if ts is not None and gm:  # NVML samples of the driver's node (no host probe needed)
                ser = isl["node_series"].setdefault("nvml", deque(maxlen=NODE_SERIES_MAX))
                ser.append([ts, round(sum(gm) / len(gm) / 1024, 2), isl["resource"]["util_pct"]])
        elif event == "rl_rollout_carry_over":
            self._derived_resp_len(isl, r)
        elif event == "rl_round_phases" and isinstance(r.get("rollout_id"), int):
            self._point(isl, r["rollout_id"] + 1, {"round_s": finite(r.get("wall_seconds")),
                                                   "tokens": finite(r.get("trained_tokens"))})
        elif event == "rl_eval":
            v = next((finite(r[k]) for k in sorted(r) if k.startswith("eval/") and finite(r.get(k)) is not None
                      and any(w in k for w in ("reward", "acc", "pass", "score"))), None)
            if v is not None and ts is not None:
                isl.setdefault("evals", []).append([ts, v, next(k for k in sorted(r) if finite(r.get(k)) == v
                                                                 and k.startswith("eval/"))])
        elif event == "rl_rollout_cutoff":
            rid = r.get("rollout_id")
            if isinstance(rid, int):
                isl["cutoffs"][rid] = {k: r.get(k) for k in CUTOFF_KEYS}
        elif event == "rl_trajectory_reward":
            t = _traj(r)
            isl["trajs"].append(t)
            if t["rid"] is not None and t["reward"] is not None and t["aborted"] is not True:
                acc = isl["traj_reward"].setdefault(t["rid"], [0.0, 0])
                acc[0] += t["reward"]
                acc[1] += 1
        elif event == "rl_load_sample":
            if ts is not None:
                isl["load"].append([ts] + [finite(r.get(k)) for k in LOAD_KEYS])
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
        elif event == "rl_run_end":
            # the island ends normally (its process exits 0): why (driver._emit_run_end)
            if ts is None or isl.get("launch_ts") is None or ts >= isl["launch_ts"]:
                isl["run_end"] = {"ts": ts, "reason": r.get("reason"), "source": r.get("source"),
                                  "rollout_id": r.get("rollout_id")}
        elif event == "rl_learner_finalized":
            # a launch's finalize read after a newer launch began belongs to the older launch
            if ts is None or isl.get("launch_ts") is None or ts >= isl["launch_ts"]:
                isl["finalized"] = True
                isl["finalized_ts"] = ts
        elif event == "rl_island_override":
            isl["negative_test"] = negative_test_label(r.get("overrides"))
        elif event == "rl_verl_island_start":
            isl["backend"] = "verl"
        elif event == "rl_engine_selected" and (r.get("rl_backend") or (r.get("miles_commit") and not isl.get("backend"))):
            # the island's training backend (its rl_backend; a Miles island writes only its miles_commit,
            # which never overrides a backend the island named): what its rounds report is the
            # backend's declaration (backend_identity.reported_metrics)
            isl["backend"] = str(r.get("rl_backend") or "miles")
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
            # Elastic islands carry the syncer's version explicitly; a re-JOINed
            # island's local policy_version runs ahead of it, so it must not key the merge.
            step = r.get("sync/outer_version", r.get("global_step", r.get("step", r.get("policy_version"))))
            if isinstance(step, int):
                target = {"rl_fragment_push": self.pushes, "rl_policy_apply": self.applies,
                          "rl_member_publication": self.publications}.get(event)
                if target is not None:
                    target.setdefault(step, set()).add(iid)
                elif event == "rl_pull_resend":
                    self.resends[step] = self.resends.get(step, 0) + 1

    def _derived_resp_len(self, isl: dict, r: dict) -> None:
        """Mean response length of a round whose rl_round_trained has no batch
        summary: the carry-over record's trained response tokens over the round's
        trained samples (both counted on the trained batch)."""
        rid = r.get("rollout_id")
        if not isinstance(rid, int):
            return
        row = (self.async_.trainer.get(isl["id"]) or {}).get("rounds", {}).get(rid) or {}
        toks, n = finite(r.get("trained_response_tokens")), finite(row.get("trained_samples"))
        pt = isl["points"].get(rid + 1, {})
        if toks is not None and n and pt.get("resp_len") is None:
            self._point(isl, rid + 1, {"resp_len": toks / n, "resp_len_derived": True})

    def phase_spans(self, isl: dict) -> list[dict]:
        """rl_timeline_span spans; without them (observation off) the wall-clock
        spans between consecutive rl_driver_phase marks: a phase lasts until the next
        mark. A publish mark has no rollout_id (round_records attaches it by time)."""
        if isl["spans"]:
            return isl["spans"]
        out = []
        marks = isl["phase_marks"]
        for (t, phase, rid), nxt in zip(marks, marks[1:]):
            task = SPAN_PHASES.get(DRIVER_PHASE_TASK.get(phase, phase))
            if task is None or nxt[0] < t:
                continue
            out.append({"phase": task, "rollout_id": rid if isinstance(rid, int) else None,
                        "start": t, "end": nxt[0], "derived": True})
        return out

    def _span(self, isl: dict, r: dict) -> None:
        """Keep phase spans on the wall clock. ``start``/``end`` are the driver's
        monotonic seconds; the record's ``time_unix`` is written at ``end``."""
        task = SPAN_PHASES.get(r.get("task"))
        a, b, t = finite(r.get("start")), finite(r.get("end")), finite(r.get("time_unix"))
        if task is None or a is None or b is None or t is None:
            return
        off = t - b
        isl["spans"].append({"phase": task, "rollout_id": r.get("rollout_id"), "start": a + off, "end": b + off})

    @staticmethod
    def _fill_traj_reward(isl: dict) -> None:
        """A trained round whose rl_round_trained carries no reward_mean (a full-parameter island
        logs per-trajectory rewards; the batch summary needs the buffer reader) reads the mean
        of the island's own rl_trajectory_reward records of that round instead."""
        for rid, (total, n) in isl["traj_reward"].items():
            pt = isl["points"].get(rid + 1)
            if n and pt is not None and rid in isl["trained_ts"] \
                    and (pt.get("reward") is None or pt.get("reward_src") == "trajectories"):
                pt["reward"] = total / n
                pt["reward_src"] = "trajectories"

    def round_records(self, isl: dict) -> list[dict]:
        """Per-round record for the page: metrics + phase spans + time split.
        A publish span has no rollout_id: it belongs to the round whose sync it follows
        (it publishes policy rollout_id+1). Missing values stay None."""
        self._fill_traj_reward(isl)
        rounds: dict[int, dict] = {}
        spans = self.phase_spans(isl)
        pubs = sorted((s for s in spans if s["phase"] == "P"), key=lambda s: s["start"])
        for sp in spans:
            rid = sp["rollout_id"]
            if sp["phase"] != "P" and isinstance(rid, int):
                rounds.setdefault(rid, {})[sp["phase"]] = [sp["start"], sp["end"]]
        pub_starts = [p["start"] for p in pubs]
        for rid, ph in rounds.items():
            after = max(v[1] for v in ph.values())
            j = bisect.bisect_left(pub_starts, after - 1)  # the first publish starting at/after the round
            if j < len(pubs):
                ph["P"] = [pubs[j]["start"], pubs[j]["end"]]
        pts = isl["points"]
        shares = self.sampler_shares(isl)
        ids = sorted(set(rounds) | {x - 1 for x in pts})
        out = []
        for rid in ids:
            pt, ph = pts.get(rid + 1, {}), rounds.get(rid, {})
            out.append({
                "round": rid, "policy_version": rid,
                "published_version": rid + 1 if (rid + 1) in isl["pubs"] else None,
                "reward": pt.get("reward"), "reward_src": pt.get("reward_src"), "reward_std": pt.get("reward_std"),
                "p10": pt.get("reward_p10"), "p50": pt.get("reward_p50"), "p90": pt.get("reward_p90"),
                "trunc": pt.get("trunc"), "logprob_diff": pt.get("logprob_diff"),
                "resp_mean": pt.get("resp_len"), "resp_p95": pt.get("resp_p95"), "tok_s": pt.get("tok_s"),
                "resp_full_mean": pt.get("resp_full"),
                "grad_norm": pt.get("grad_norm"), "kl": pt.get("kl"), "kl_tr": pt.get("kl_tr"),
                "tis_clip": pt.get("tis_clip"), "is_clip": pt.get("is_clip"),
                "resp_derived": bool(pt.get("resp_len_derived")),
                "ts": pt.get("t"), "lr": pt.get("lr"), "step_s": pt.get("step_s"), "round_s": pt.get("round_s"),
                "tokens": pt.get("tokens"), "zero_var_ratio": pt.get("zero_var_ratio"),
                "phases": ph, "dur": {k: round(v[1] - v[0], 1) for k, v in ph.items()},
                "discarded_trainings": self.discarded_trainings(isl, rid),
                "by_sampler": shares.get(rid),
            })
        return out

    def sampler_shares(self, isl: dict) -> dict[int, dict]:
        """Async buffer trainer: per trained round, what each sampler contributed: groups, samples,
        the policy versions they were generated with, their mean age and mean reward. A sampler
        names its trajectories ``<incarnation>-r<round>-row<row>-<sample>``, so the incarnation
        prefix of rl_trajectory_reward.trajectory_id gives the sampler."""
        owner = {x["incarnation"]: iid for iid, s in self.async_.sampler.items() for x in s.get("incarnations") or ()}
        if not owner or not isl["trajs"]:
            return {}
        acc: dict[int, dict] = {}
        for t in isl["trajs"]:
            m = TRAJ_ID_RE.match(str(t.get("id") or ""))
            sid = owner.get(m.group(1)) if m else None
            if sid is None or t.get("rid") is None:
                continue
            a = acc.setdefault(t["rid"], {}).setdefault(sid, {"groups": set(), "samples": 0, "versions": set(),
                                                             "rewards": [], "ages": []})
            a["samples"] += 1
            if t.get("group") is not None:
                a["groups"].add(t["group"])
            # generated with: the version segments of the trajectory ([version, from, to] per segment);
            # rl_trajectory_reward.policy_version is the trainer's own version
            gen = [x[0] if isinstance(x, list) and x else x for x in (t.get("versions") or ())]
            gen = [v for v in gen if isinstance(v, int) and not isinstance(v, bool)]
            if gen:
                a["versions"].update(gen)
                a["ages"].append(t["rid"] - min(gen))
            if t.get("reward") is not None:
                a["rewards"].append(t["reward"])
        return {rid: {sid: {"groups": len(a["groups"]) or None, "samples": a["samples"],
                            "versions": sorted(a["versions"]),
                            "age_mean": round(sum(a["ages"]) / len(a["ages"]), 3) if a["ages"] else None,
                            "reward_mean": round(sum(a["rewards"]) / len(a["rewards"]), 4) if a["rewards"] else None}
                      for sid, a in by.items()} for rid, by in acc.items()}

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

    def utilization(self, isl: dict, rounds: list[dict] | None = None) -> dict | None:
        """per-round cutoff tally, completion times inside the
        generation span, four-phase time split, load (KV / queue) series and stage-2
        carry-over. None when the tape has none of these events (older tapes)."""
        if not (isl["cutoffs"] or isl["trajs"] or isl["load"]):
            return None
        rounds = self.round_records(isl) if rounds is None else rounds
        gen = {r["round"]: r["phases"].get("R") for r in rounds}
        by_rid: dict[int, list[dict]] = {}
        for t in isl["trajs"]:
            if t["rid"] is not None:
                by_rid.setdefault(t["rid"], []).append(t)
        load_ts = sorted(isl["load"], key=lambda x: x[0])  # once: each round bisects its window
        load_keys = [x[0] for x in load_ts]
        out_rounds = []
        for rid in sorted(set(gen) | set(isl["cutoffs"]) | set(by_rid)):
            span, trs = gen.get(rid), by_rid.get(rid, [])
            t0 = span[0] if span else None
            done = sorted(round(t["end"] - t0, 2) for t in trs if t0 is not None and t["end"] is not None)
            phases = {k: _sum(t[k] for t in trs) for k, _ in TRAJ_PHASES}
            known = [v for v in phases.values() if v is not None]
            tot = sum(known) if known else None
            peaks = {"kv": None, "queued": None}
            if span:
                win = load_ts[bisect.bisect_left(load_keys, span[0]):bisect.bisect_right(load_keys, span[1])]
                kv = [x[1] / x[2] for x in win if x[1] is not None and x[2]]
                q = [x[3] for x in win if x[3] is not None]
                peaks = {"kv": round(max(kv), 4) if kv else None, "queued": max(q) if q else None}
            out_rounds.append({
                "round": rid, "gen": span, "gen_s": round(span[1] - span[0], 1) if span else None,
                "cutoff": isl["cutoffs"].get(rid), "done": done, "trajectories": len(trs),
                "phases": phases, "tool_share": round(phases["tool"] / tot, 4) if tot and phases["tool"] is not None else None,
                "peaks": peaks})
        trajs = [{k: t[k] for k in ("rid", "id", "task", "reward", "gen", "tool", "judge", "sandbox")} |
                 {"off": round(t["start"] - gen[t["rid"]][0], 2) if gen.get(t["rid"]) and t["start"] is not None else None,
                  "dur": round(t["end"] - t["start"], 2) if t["start"] is not None and t["end"] is not None else None}
                 for t in isl["trajs"][-UTIL_TRAJ_MAX:]]
        carry = [{"id": t["id"], "task": t["task"], "from": t["started_rid"], "to": t["rid"], "versions": t["versions"]}
                 for t in isl["trajs"]
                 if (t["started_rid"] is not None and t["rid"] is not None and t["started_rid"] != t["rid"])
                 or (t["versions"] and len({_seg_version(v) for v in t["versions"]}) > 1)]
        load = list(isl["load"])
        step = max(1, len(load) // NODE_SERIES_POINTS)
        has_kv = any(x[1] is not None and x[2] for x in load)
        return {"rounds": out_rounds, "trajectories": trajs, "carry": carry or None,
                "load": [[x[0], round(x[1] / x[2], 4) if x[1] is not None and x[2] else None, x[3], x[4], x[5]]
                         for x in load[::step]] if load else None,
                "load_has_kv": has_kv}

    def node_series(self, isl: dict) -> dict:
        out = {}
        for rank, ser in isl["node_series"].items():
            rows = sorted(ser, key=lambda p: p[0])  # tapes of several launches / incarnations interleave
            step = max(1, len(rows) // NODE_SERIES_POINTS)
            out[rank] = rows[::step]
        return out

    def page_view(self, *, live: bool = False, now: float | None = None,
                  table_limit: int | None = VIEW_TABLE_MAX) -> dict:
        """Everything the redesigned page (section 9) draws, in one object. Lists drawn only as
        tables keep their newest ``table_limit`` rows (None: all); ``table_totals`` names their full
        length (the async section keeps the keys of ``AsyncState.view``)."""
        now = self.now(live) if now is None else now
        ov = self.overview(live=live, now=now)
        isl_extra = {}
        for iid in self.island_ids():
            isl = self.islands[iid]
            starts = [e["ts"] for e in isl["recent"] if e["type"] == "rl_driver_start"]
            rounds = self.round_records(isl)
            isl_extra[iid] = {
                "rounds": rounds, "node_series": self.node_series(isl),
                "util": self.utilization(isl, rounds),
                "first_ts": isl["first_ts"], "ready_ts": isl["ready_ts"],
                "driver_start_ts": isl.get("driver_start_ts") or (starts[0] if starts else None),
                "cells": len(isl["cells"] or []), "hardware": isl.get("hardware"),
                "transactions": len(isl["tx_order"]), "ray_embed": self.ray_embed(isl, self.island_ids().index(iid)),
                "role": self.async_.role(iid), "gpu_mem_note": gpu_mem_note(isl),
                "evals": list(isl.get("evals") or [])[-500:],
                "hbm_total_gib": _hbm_total_gib(isl), "last_ts": isl["last_event_ts"],
            }
        usage = {"syncer": self.counts["syncer"] > 0, "journal": self.counts["journal"] > 0,
                 "cells": any(v["cells"] for v in isl_extra.values()),
                 "transactions": any(v["transactions"] for v in isl_extra.values())}
        rounds_syncer = self.rounds()
        totals: dict = {"rounds_syncer": len(rounds_syncer)}
        view = {"overview": ov, "islands": isl_extra, "rounds_syncer": _tail(rounds_syncer, table_limit),
                "usage": usage, "label": self.label, "spot": list(self.spot),
                "table_limit": table_limit, "table_totals": totals}
        for x in isl_extra.values():
            _cap_util(x["util"], table_limit)
        fleet = fleet_islands.resolve(self)  # None unless a fleet of async islands
        asy = self.async_.view(fleet=fleet)
        if asy is not None:
            if fleet is not None:  # per-pool train-rollout mismatch (fleet_islands.pool_mismatch)
                fleet_islands.attach_mismatch(self, asy, {i: x["rounds"] for i, x in isl_extra.items()}, fleet)
            totals.update(_cap_releases(asy, table_limit))
            view["async"] = asy
        topo = topology_mod.view(self, ov["islands"], now)
        if topo is not None:
            view["topology"] = topo
        view["layout"] = topology_mod.layout(self)  # which drawing and why (topology_diloco.layout)
        brk = self.breakers_view(ov.get("cost"))
        if brk is not None:
            view["breakers"] = brk
        sft = self.sft_view(now)
        if sft is not None:
            view["sft"] = sft
        if self.compare:
            # other runs, utilization only (side by side / overlay)
            view["compare"] = [{"label": c.label, "run": c.run_name()[0],
                                "util": {i: _cap_util(c.utilization(c.islands[i]), table_limit)
                                         for i in c.island_ids()}}
                               for c in self.compare]
        return view

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
        values["kl"] = ppo_kl(r)
        if event == "rl_local_round" and values.get("kl_tr") is not None \
                and values["kl_tr"] == finite(r.get("mean_kl")):
            # the ports engine writes its loss-dict KL (``mean_kl``) under train/train_rollout_kl too;
            # the round's train-rollout KL is on its rl_round_trained (0.0 here is not a measurement)
            values["kl_tr"] = None
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
        if values.get("lr") is None and isinstance(r.get("applied_lrs"), list) and r["applied_lrs"]:
            values["lr"] = finite(r["applied_lrs"][0])
        values["t"] = record_ts(r)  # wall clock of the round (the metrics page plots over time)
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
                isl["cells_source"] = "journal gpu_pool (derived)"
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
        life = self.fleet_lifecycle.setdefault(iid, [])
        older = ts is not None and bool(life) and ts < life[-1][0]
        shape_ts = self.fleet_shape_ts.setdefault(iid, {})
        for key, field in (*((k, k) for k in ("cloud", "region", "gpu", "gpus", "price_key", "cpus",
                                                 "memory_gib")), ("name", "island")):
            # each shape field: the newest record that names it (an older one arriving later does not)
            if r.get(field) is not None and (ts is None or ts >= shape_ts.get(key, ts)):
                isl[key] = r[field]
                if ts is not None:
                    shape_ts[key] = ts
        if ts is not None:
            bisect.insort(life, (ts, self.event_seq, event))
            if older:  # folded in time order again: the newest record still decides the state
                self._replay_fleet_lifecycle(isl, life)
                return
        self._fold_fleet_lifecycle(isl, event, ts)

    def _replay_fleet_lifecycle(self, isl: dict, life: list[tuple[float, int, str]]) -> None:
        """The island's lifecycle fields from its timed records alone, in time order (the
        fold of :meth:`_fold_fleet_lifecycle`, from a new island's values)."""
        fresh = _new_island(isl["id"])
        for key in ("fleet_state", "ready_ts", "last_ready_ts", "stop_ts", "lost_ts", "open_ts", "closed_s"):
            isl[key] = fresh[key]
        for t, _seq, event in life:
            self._fold_fleet_lifecycle(isl, event, t)

    @staticmethod
    def _fold_fleet_lifecycle(isl: dict, event: str, ts: float | None) -> None:
        isl["fleet_state"] = event[len("island_"):]
        # wall clock accrues over ready -> lost/stop intervals (relaunches re-open one)
        if event == "island_ready" and ts is not None:
            if isl["ready_ts"] is None:
                isl["ready_ts"] = ts
            isl["last_ready_ts"] = max(ts, isl["last_ready_ts"] or ts)
            if isl["open_ts"] is None:
                isl["open_ts"] = ts
            isl["stop_ts"] = None
        elif event in ("island_lost", "island_stop", "island_done") and ts is not None:
            if event == "island_lost":
                isl["lost_ts"] = ts
            else:
                isl["stop_ts"] = ts
            if isl["open_ts"] is not None:
                isl["closed_s"] += max(0.0, ts - isl["open_ts"])
                isl["open_ts"] = None

    def _feed_pool(self, r: dict, iid: str | None, ts: float | None) -> None:
        self.async_.fold_pool(r)

    def _feed_sft(self, r: dict, iid: str | None, ts: float | None) -> None:
        self.sft.fold(r, ts)

    def sft_view(self, now: float) -> dict | None:
        """SFT islands (ms-swift) and the syncer as the page draws them; None for RL runs."""
        if not self.sft.islands and not self.sft.sft_records:
            return None  # RL run (its syncer tape / syncer.log alone do not make an SFT run)
        key = (self.sft.records, len(self.merges), now)  # the page view and its topology (topology_sft) share it
        hit = getattr(self, "_sft_view", None)
        if hit is not None and hit[0] == key:
            return hit[1]
        refs = {k: self.thresholds.get(k, v) for k, v in sft_mod.DEFAULT_REFS.items()}
        refs = {k: v for k, v in refs.items() if v > 0 or k not in sft_mod.REF_LINES}  # unset lines are not drawn
        out = self.sft.view(now=now, merges=self.merges, refs=refs)
        self._sft_view = (key, out)
        return out

    def _syncer_terminal_grace_s(self) -> float | None:
        """The syncer's terminal grace once its committed steps reach the last P steps of the run
        (its ``syncer_config``: last_step, fragments, terminal_grace_ms), else None."""
        cfg = next((r for r in reversed(self.syncer_records) if r.get("kind") == "syncer_config"), None)
        if cfg is None or not self.merges:
            return None
        last, p = cfg.get("last_step"), cfg.get("fragments")
        if not isinstance(last, int) or not isinstance(p, int) or p < 1:
            return None
        top = max((m.get("step") for m in self.merges if isinstance(m.get("step"), int)), default=0)
        if top <= last - p:
            return None
        grace = cfg.get("terminal_grace_ms")
        return max(TERMINAL_GRACE_DEFAULT_S, grace / 1000.0) if isinstance(grace, (int, float)) and grace > 0 \
            else TERMINAL_GRACE_DEFAULT_S

    def _feed_other(self, r: dict, iid: str | None, ts: float | None) -> None:
        if r.get("kind") in SYNCER_RECORD_KINDS and r.get("event") is None:
            if len(self.syncer_records) < SYNCER_RECORDS_MAX:
                self.syncer_records.append(dict(r, _ts=ts, _source=self._source))
            return
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
                "quorum": finite(rec.get("quorum")),
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
        self._fill_traj_reward(isl)
        pts = isl["points"]
        last = pts[max(pts)] if pts else {}
        last_any = max([t for t in (isl["last_event_ts"], isl["last_heartbeat_ts"]) if t is not None],
                       default=None)
        age = (now - last_any) if last_any is not None else None
        hb_age = (now - isl["last_heartbeat_ts"]) if isl["last_heartbeat_ts"] is not None else None
        res = isl["resource"] or {}
        role = self.async_.role(isl["id"])
        ended = isl["fleet_state"] in ("lost", "stop", "done")
        starting = not isl["driver_started"] and not isl["finalized"] and not ended
        samp = self.async_.sampler_state(isl["id"]) if role == "sampler" else None
        if samp is not None:
            # a sampler is starting until its first round (engine load + checks + the
            # first generation write nothing for minutes)
            starting = not samp["started_rounds"] and not samp["ended"] and not ended
        if starting and isl["ready_ts"] is not None:
            # A container that is still loading weights may have written nothing yet:
            # count the startup age from island_ready as well.
            ref = max(t for t in (last_any, isl["ready_ts"]) if t is not None)
            age = now - ref
        startup_s = (now - min(t for t in (isl["ready_ts"], isl["first_ts"]) if t is not None)
                     if starting and (isl["ready_ts"] is not None or isl["first_ts"] is not None) else None)
        if starting and len(isl["launches"]) > 1 and isl["launch_ts"] is not None:
            startup_s = now - isl["launch_ts"]  # a restarted trainer: since its current launch
        stage = self._startup_stage(isl, now) if starting else None
        # the current step began at the last startup step / start phase mark (of this launch)
        step_ref = max([t for t in (isl["startup_step_ts"], isl["start_phase_ts"]) if t is not None], default=None) \
            or (isl["launch_ts"] if len(isl["launches"]) > 1 else None) or isl["first_ts"]
        stopped = self.stopped_by_us(isl)
        waiting_why = samp["waiting"] if samp else None
        waiting = bool(waiting_why) or (bool(isl["recent"]) and isl["recent"][-1]["type"] == "rl_sampler_backlog_full")
        left = self._trainer_left(isl)
        pool = self.async_.pool_exit(isl["id"], after=isl["last_event_ts"]) if role == "sampler" else None
        if left is None and pool is not None and pool["status"] != "done":
            left = {"ts": pool.get("cause_ts") or pool["ts"], "reason": pool["reason"], "event": pool["event"],
                    "ticket": pool.get("ticket"), "replaced_by": pool.get("replaced_by"),
                    "source": "nc pool", "lost": pool["status"] == "lost"}
        pool_removed = None
        if role == "sampler" and left is None:
            # the sampler's own leave on its tape, else the fleet's membership (its member record,
            # its shard retired): a removed pool's sampler is gone, not stale
            own = (samp or {}).get("left")
            if own and own.get("ts") is not None and (isl["last_event_ts"] or 0) > own["ts"] + STOP_SLACK_S:
                own = None  # it wrote after its leave: not ended by it
            gone = ({"ts": own["ts"], "reason": own.get("reason") or "left", "source": "sampler leave",
                     "done": (own.get("reason") or "") in async_rl.SAMPLER_DONE_REASONS}
                    if own else self._membership_left(isl, now))
            if gone is not None and gone.get("draining"):
                pool_removed = {k: gone.get(k) for k in ("ts", "reason", "pool", "member_state", "wrote_after",
                                                         "released", "detail")}  # still runs: draining, or not released
            elif gone is not None and gone.pop("done"):
                samp = {**(samp or {}), "ended": True}  # left after finishing its work: done
            elif gone is not None:
                left = {**gone, "lost": False}
        # a trainer island of a fleet launched island by island: its records in the run store say
        # what it is (yeto.rl.trainer_worker), over what its tape last said
        inow = self._trainer_island_now(isl) if role != "sampler" else None
        rec_stopped = bool(inow) and inow["state"] in ("stopped", "done", "down")
        if inow and inow["state"] in ("moving", "starting"):
            left = None  # the leave on its tape is the superseded launch's: its next launch is starting
        if left is None and inow and inow["state"] in ("removed", "failed", "abandoned"):
            left = {"ts": inow.get("since"), "reason": inow["state"] + (f": {inow['reason']}" if inow.get("reason")
                                                                        and inow["state"] != "removed" else ""),
                    "detail": inow.get("detail"), "source": "trainer island record",
                    "lost": inow["state"] in ("failed", "abandoned")}
        sl = isl["source_lost"]
        if left is None and sl and not sl.get("recovered") and sl.get("ts") is not None \
                and (isl["last_event_ts"] or 0) <= sl["ts"] + STOP_SLACK_S and not isl["finalized"]:
            # the follow of its tape broke and nothing arrived since: the node is gone for this page
            left = {"ts": sl["ts"], "reason": "tape source lost: " + str(sl.get("error") or "")[:120],
                    "source": "mirror", "lost": True}
        role = role or self._guess_role(isl)
        # no heartbeats: the island's own cadence (round / generation / load times) sets
        # how long a quiet island is still healthy
        gap = self.async_.expected_gap_s(isl["id"])
        warn_s = self.thresholds["heartbeat_warn_s"]
        if not isl["heartbeat_seen"] and gap is not None:
            warn_s = max(warn_s, gap)
        if not isl["heartbeat_seen"] and self.outer_sync.get(isl["id"]):
            # a DiLoCo trainer without heartbeats (older islands) writes nothing while it waits for
            # the others: its first round (warm-up, the first generation) and, in the syncer's
            # terminal phase, a last round held open for the slowest island
            if isl["driver_started"] and not (self.async_.trainer.get(isl["id"]) or {}).get("walls"):
                warn_s = max(warn_s, self.thresholds["startup_warn_s"])
            terminal = self._syncer_terminal_grace_s()
            if terminal is not None:
                warn_s = max(warn_s, (gap or 0.0) + terminal)
        # a sampler "runs until its trainers end": done once they ended and it stopped writing groups;
        # one still writing groups after the trainers finished is running (and billed), not done
        trainers_done = role == "sampler" and self._trainers_done() and (
            (self.async_.last_write_ts(isl["id"]) or 0) <= (self._trainers_end_ts() or float("inf"))
            + SAMPLER_AFTER_TRAINERS_S)
        # an island we stopped stays "stopped" whatever it wrote on its way out (island_leave
        # STOPPED, RECOVERY_REQUIRED); only its fleet record (removed / failed / abandoned) is above it
        rec_left = left is not None and left.get("source") == "trainer island record"
        if (stopped or rec_stopped) and not isl["finalized"] and not rec_left:
            status = "stopped"
            starting = False
        elif isl["recovery_required"] and not _recovered_after(isl):
            status = "recovery"
        elif isl["fleet_state"] == "lost":
            status = "lost"
        elif left is not None and not isl["finalized"]:
            # left the run before finishing (island_leave, or the NC pool released / lost its node)
            status = "lost" if left.get("lost") else "left"
            starting = False
        elif pool is not None and pool["status"] == "done":
            status = "done"
            starting = False
        elif role == "unrecognized":
            status = "unknown"  # a tape whose records name no trainer / sampler event
            starting = False
        elif isl["finalized"] or isl["fleet_state"] in ("stop", "done") or (samp and samp["ended"]):
            status = "done"
        elif trainers_done:
            status = "done"  # a sampler that runs until its trainers end
            starting = False
        elif waiting:
            status = "waiting"
            starting = False
        elif starting and age is not None:
            # loading (weights, Ray, engine, first generation) is not stale: the
            # startup alerts report a slow start instead
            status = "starting"
        elif age is not None and not isl["heartbeat_seen"] and gap is not None \
                and age > async_rl.LOST_FACTOR * warn_s:
            status = "lost"  # silent for 3x its own cadence: counted as gone
        elif age is not None and age > warn_s:
            status = "stale"
        elif age is None:
            status = "unknown"
        else:
            status = "ok"
        if inow and inow["state"] in ("starting", "moving") and status in ("stale", "lost", "left", "waiting",
                                                                          "unknown", "ok", "recovery"):
            status, starting = "starting", True  # relaunching / moving: its new incarnation is starting
            waiting_why = inow.get("reason")
        return {
            "id": isl["id"], "name": isl["name"], "cloud": isl["cloud"], "region": isl["region"],
            "gpu": isl["gpu"], "gpus": isl["gpus"], "status": status, "finalized": bool(isl["finalized"]),
            "stopped_by_us": stopped, "starting": starting, "startup_s": _r(startup_s), "driver_started": isl["driver_started"],
            # a sampler waiting at its backlog cap writes nothing until the trainers read
            "waiting_backlog": waiting, "waiting_reason": waiting_why or ("backlog full" if waiting else None),
            "role": role, "left": left, "stale_after_s": _r(warn_s), "island_now": inow,
            # a sampler whose pool removed or retired it but that still runs (draining, or its
            # machine not released): shown running, alerted past alerts.POOL_REMOVED_GRACE_S
            "pool_removed": ({**pool_removed, "since_s": _r(now - pool_removed["ts"])
                              if pool_removed.get("ts") is not None else None}
                             if pool_removed and left is None else None),
            "startup_slow": bool(starting and age is not None and age > (stage or {}).get(
                "warn_s", self.thresholds["startup_warn_s"])),
            # the current startup stage (trainer_start_phase / rl_startup_step) and its thresholds
            "startup_stage": stage,
            "launch": isl["launch"],
            "launches": [{**v, "state": "current" if k == isl["launch"] else "superseded"}
                         for k, v in sorted(isl["launches"].items(), key=lambda kv: kv[1]["first_ts"] or 0)],
            "nodes": [isl["nodes"][k] for k in sorted(isl["nodes"], key=lambda x: int(x) if x.isdigit() else 0)],
            "last_event_age_s": _r(age), "heartbeat_age_s": _r(hb_age),
            "heartbeat_seen": isl["heartbeat_seen"], "round": isl["round"],
            "startup_steps": {k: {"seconds": v["seconds"], "step_s": v["step_s"]}
                              for k, v in isl["startup_steps"].items()},
            "startup_step": isl["startup_step"],
            "startup_step_age_s": _r(now - step_ref) if starting and step_ref is not None else None,
            "rollout_id": isl["rollout_id"], "policy_version": isl["policy_version"],
            "released_version": self._released_version(isl["id"]),
            "phase": isl["phase"], "staleness": isl["staleness"], "contribution": isl["contribution"],
            "gpu_util_pct": res.get("util_pct"), "mem_pct": res.get("mem_pct"),
            "resource_available": res.get("available") if res else None,
            "host_mem_peak_bytes": (isl["host"] or {}).get("peak_bytes"),
            "host_mem_current_bytes": (isl["host"] or {}).get("current_bytes"),
            "reward": last.get("reward"), "tok_s": last.get("tok_s"),
            "source_lost": isl["source_lost"],
            "resumes": list(isl["resumes"]), "cuts_saved": len(isl["cuts"]),
            "segments": self.resume_segments(isl),
            "negative_test": isl["negative_test"],
            "backend": isl.get("backend"),
            # round metrics the island's backend does not report: shown as "not reported by <backend>"
            "unreported": unreported_metrics(isl.get("backend")),
            "backend_label": backend_label(isl.get("backend")),
        }

    def infer_hardware(self) -> None:
        """Cloud / accelerator / count of an island that no fleet.jsonl record names (a
        store-only run): a sampler from its rl_sampler_start provenance (card model,
        tpN) and the NC pool naming its shard; a trainer from its tape launch dir
        (``l0-modal-...``), its NVML samples (per-GPU memory -> model, GPUs per node)
        and the node count of remote_code_prewarm. Marked ``hw_inferred``."""
        pool_shards = self.async_.pool_shards
        for iid, isl in self.islands.items():
            if isl["cloud"] is not None and isl["gpu"] is not None and not isl.get("hw_inferred"):
                continue
            if self.async_.role(iid) == "sampler":
                prov = (self.async_.sampler.get(iid) or {}).get("provenance") or {}
                m = re.search(r"tp(\d+)", str(prov.get("topology") or ""))
                gpus = int(m.group(1)) if m else None
                nc = iid in pool_shards
                if nc:  # an NC node is billed whole, whatever the engine's TP uses of it
                    seen = len((isl.get("resource") or {}).get("gpus") or ())
                    gpus = max(gpus or 0, seen or NC_NODE_GPUS)
                hw = {"cloud": "nc" if nc else None, "gpu": prov.get("card_model"), "gpus": gpus}
            elif isl["launch"] is not None or isl.get("gpu_mem_total_mb"):
                per_node, mem = isl.get("gpus_per_node"), isl.get("gpu_mem_total_mb")
                hw = {"cloud": "modal" if "-modal-" in f"-{isl['launch'] or ''}" else None,
                      "gpu": _gpu_from_mem(mem),
                      "gpus": per_node * max(1, isl.get("nodes_seen") or 1) if per_node else None}
            else:
                continue
            for k, v in hw.items():
                if v is not None and (isl[k] is None or isl.get("hw_inferred")):
                    isl[k] = v
                    isl["hw_inferred"] = True

    def _startup_stage(self, isl: dict, now: float) -> dict:
        """What a starting island is doing now and how long it may stay quiet: the actor
        init stage (model build + weight load, K3.1 mbridge ~22 min with no log line)
        gets ``startup_init_*`` thresholds instead of ``startup_*``."""
        th = self.thresholds
        phase, step = isl["start_phase"], isl["startup_step"]
        ts = max([t for t in (isl["start_phase_ts"], isl["startup_step_ts"]) if t is not None], default=None)
        init = phase in INIT_STAGE_PHASES or step in INIT_STAGE_STEPS
        if phase is not None and (isl["start_phase_ts"] or 0) >= (isl["startup_step_ts"] or 0):
            label = START_STAGE.get(phase, f"after {phase}")
        elif step is not None:
            from .alerts import _next_startup_step  # noqa: PLC0415
            label = f"waiting for {_next_startup_step(step)}"
        else:
            label = "starting"
        return {"phase": phase, "step": step, "label": label, "init": init,
                "since_ts": ts, "age_s": _r(now - ts) if ts is not None else None,
                "warn_s": th["startup_init_warn_s"] if init else th["startup_warn_s"],
                "severe_s": th["startup_init_severe_s"] if init else th["startup_severe_s"]}

    TRAINER_HINTS = ("rl_driver_", "trainer_start_", "rl_round_", "rl_local_round", "rl_publication",
                     "rl_heartbeat", "rl_policy_release", "rl_engine_selected", "rl_learner_")

    def _guess_role(self, isl: dict) -> str | None:
        """Role from the tape's records, never from a file name: trainer events make a
        trainer; a fleet.jsonl island with no tape yet stays None (a launcher island);
        anything else (only dashboard / unknown events) is ``unrecognized``."""
        types = isl["events_by_type"]
        if any(t.startswith(self.TRAINER_HINTS) for t in types):
            return "trainer"
        if isl["fleet_state"] is not None or not types:
            return None
        return "unrecognized"

    def _trainers_end_ts(self) -> float | None:
        """When the last trainer ended (its finalize, leave or last event); None if unknown."""
        ends = []
        for i, role in self.async_.roles.items():
            if role != "trainer" or i not in self.islands:
                continue
            isl = self.islands[i]
            left = self._trainer_left(isl)
            ends.append(isl.get("finalized_ts") or (left or {}).get("ts") or isl["last_event_ts"])
        ends = [t for t in ends if t is not None]
        return max(ends) if ends else None

    def _trainers_done(self) -> bool:
        ids = [i for i, role in self.async_.roles.items() if role == "trainer" and i in self.islands]
        return bool(ids) and all(
            self.islands[i]["finalized"] or self.islands[i]["fleet_state"] in ("stop", "done", "lost")
            or _clean_leave(self._trainer_left(self.islands[i])) for i in ids)

    def _trainer_left(self, isl: dict) -> dict | None:
        """The island_leave of the island's current launch: a leave older than the
        current launch belongs to a launch that a restart replaced."""
        left = self.async_.trainer_left(isl["id"])
        if left and left.get("ts") is not None and isl.get("launch_ts") is not None \
                and len(isl["launches"]) > 1 and left["ts"] < isl["launch_ts"]:
            return None
        return left

    def series(self, isl: dict) -> dict:
        self._fill_traj_reward(isl)
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
            return {"mode": "unknown", "note": "no data: fleet.jsonl has no cloud for this island"}
        if cloud == "modal":
            return {"mode": "none", "note": "cannot embed: Modal containers have no stable inbound port"}
        if cloud in ("local", "\u672c\u5730"):
            return {"mode": "direct", "port": 8265, "command": None}
        port = 18265 + index
        host = isl["name"] or isl["id"]
        return {"mode": "tunnel", "port": port,
                "command": f"ssh -L {port}:localhost:8265 {host}"}

    def cost(self, now: float) -> dict:
        self.apply_run_host_shape()
        self.infer_hardware()
        view = cost_mod.cost_view(self, now)
        view["run_up"] = cost_mod.run_up_view(getattr(self, "up_state", None), now)
        return view

    def overview(self, *, live: bool = False, now: float | None = None) -> dict:
        now = self.now(live) if now is None else now
        rounds = self.rounds()
        cost = self.cost(now)
        cards = [self._island_card(self.islands[k], now) for k in self.island_ids()]
        store = self.store_status() if callable(self.store_status) else None
        alerts = alerts_mod.evaluate(self, now=now, rounds=rounds, cost=cost, cards=cards, store=store)
        torn_down = (self.operator_stop() or {}).get("cause") == "teardown"
        n_bad = sum(1 for c in cards if c["status"] in ("stale", "recovery")
                    or (c["status"] == "lost" and not torn_down))  # after teardown a lost node is expected
        sev0 = sum(1 for a in alerts if a["sev"] == 0)
        stop = self.operator_stop()
        # every island ended (a trainer leaving on a STOP flag reads "left"): how the run ended
        ended = self.run_end_status() if cards and all(c["status"] in ("stopped", "done", "left") for c in cards) else None
        if not cards and store and store.get("failing"):
            global_status = {"level": "bad", "text": "run store unreadable"}
        elif not cards:
            global_status = {"level": "muted", "text": "no data"}
        elif ended and ended["kind"] == "finished":
            fin_ = ended["finish"]
            global_status = {"level": "ok", "text": f"finished ({fin_['reason']}"
                             + (f", {fin_['source']}" if fin_.get("source") else "") + ")"}
        elif ended and ended["kind"] == "stopped":
            global_status = {"level": "muted", "text": f"stopped (stopped by us: {ended['stop'].get('cause')})"}
        elif n_bad:
            global_status = {"level": "bad", "text": f"{n_bad} islands unhealthy"}
        elif sev0:
            global_status = {"level": "bad", "text": f"{sev0} critical alerts"}
        elif alerts:
            global_status = {"level": "warn", "text": f"{len(alerts)} alerts"}
        else:
            global_status = {"level": "ok", "text": "all healthy"}
        run, inferred = self.run_name()
        return {
            "run": run, "run_inferred": inferred, "operator_stop": stop, "run_kind": self.run_kind(),
            "run_end": ended,
            "mode": "live" if live else "offline", "now": now,
            "data_ts": self.max_ts, "first_ts": self.min_ts, "global_status": global_status,
            "alerts": alerts, "cost": cost, "islands": cards,
            "metrics": [[k, label] for k, label, _ in METRICS],
            "series": {c["id"]: self.series(self.islands[c["id"]]) for c in cards},
            "round_marks": [{"round": r["round"], "missed": bool(r["missed"]), "resend": r["resend"]}
                            for r in rounds],
            "counts": dict(self.counts), "events_total": self.event_seq,
            "store": store, "commands": self.commands, "duplicates": self.duplicates,
        }

    def run_meta_fresh(self) -> dict | None:
        """The run's ``meta.json`` read now (its state / ``finished_at`` change when the run
        ends; :meth:`_run_meta` caches the first read)."""
        from .sources import run_meta_near

        return next((m for m in (run_meta_near(s) for s in self.sources_seen) if m), None)

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

    def island_states(self, now: float | None = None, *, live: bool = False) -> dict[str, dict]:
        """Each island's state, one rule for every view (overview cards, Islands now, topology):
        ``status`` (ok / waiting / starting / stale / stopped / left / lost / done / ...), ``alive``
        (status in ALIVE_STATUSES), role, current launch and sampler incarnation, last event time.
        A relaunched trainer is its new launch's state; a sampler is done only once its trainers
        ended and it stopped writing groups (_island_card)."""
        now = self.now(live) if now is None else now
        out = {}
        for iid in self.island_ids():
            isl = self.islands[iid]
            card = self._island_card(isl, now)
            samp = self.async_.sampler.get(iid) or {}
            out[iid] = {"status": card["status"], "alive": card["status"] in ALIVE_STATUSES,
                        "role": card.get("role"), "starting": bool(card.get("starting")),
                        "launch": isl.get("launch"), "incarnation": samp.get("incarnation"),
                        "last_event_ts": isl.get("last_event_ts")}
        return out

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
        return {"records": list(self.fleet_records), "cost_ticks": list(self.cost_ticks),
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


def _tail(rows: list, limit: int | None) -> list:
    """The newest ``limit`` rows of a list kept oldest first (all of them for None)."""
    return rows if limit is None or len(rows) <= limit else rows[-limit:]


def _cap_util(util: dict | None, limit: int | None) -> dict | None:
    """Per-round cutoff tallies and carried-over trajectories of a utilization view: the newest rows."""
    if util is None:
        return None
    util["rounds_total"] = len(util["rounds"])
    util["rounds"] = _tail(util["rounds"], limit)
    if util.get("carry"):
        util["carry_total"] = len(util["carry"])
        util["carry"] = _tail(util["carry"], limit)
    return util


def _cap_releases(asy: dict, limit: int | None) -> dict:
    """Release rows of the async view (and of each fleet island): the newest versions. Returns the
    totals: every row, the rows a trainer released (a sampler's base-weights v0 row is none) and
    each fleet island's rows."""
    rows = asy.get("releases") or []
    out: dict = {"releases": len(rows), "releases_trainer": sum(1 for r in rows if r.get("trainer") is not None
                                                                 or r.get("released_at") is not None)}
    if rows:
        asy["releases"] = _tail(rows, limit)
    if asy.get("islands"):
        out["island_releases"] = {}
        for x in asy["islands"]:
            out["island_releases"][str(x["island"])] = len(x.get("releases") or [])
            if x.get("releases"):
                x["releases"] = _tail(x["releases"], limit)
    return out


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


GPU_MEM_HOW = ("launch with --rl-resource-sample-interval <seconds> (trainer NVML, rl_resource_sample) "
               "or set YETO_MODAL_HOSTMEM_SAMPLE_S=<seconds> (Modal per-node probe, modal_host_sample)")


def _hbm_total_gib(isl: dict) -> float | None:
    gpus = (isl["resource"] or {}).get("gpus") or []
    tot = [finite(g.get("mem_total_mb")) for g in gpus]
    tot = [t for t in tot if t]
    return round(max(tot) / 1024, 1) if tot else None


def gpu_mem_note(isl: dict) -> str | None:
    """Why the GPU memory chart of an island is empty (None when it has data)."""
    if isl["node_series"]:
        return None
    res = isl["resource"]
    if res and res.get("available") is False:
        return ("rl_resource_sample reported unavailable" + (f" ({res['reason']})" if res.get("reason") else "")
                + ": the island's image lacks the card telemetry library (NVML: pynvml; ROCm: amdsmi)")
    return "not sampled: " + GPU_MEM_HOW


def _recovered_after(isl: dict) -> bool:
    last = isl["recovery_required"][-1]["ts"] or 0
    return any(r.get("result") == "RECOVERED" and (r.get("ts") or 0) > last for r in isl["reconfig"])


# per-GPU memory (NVML mem_total_mb) -> accelerator model, largest first
NC_NODE_GPUS = 8  # a National Compute sampler node (yeto.launch.nc.client.hourly_cost_ucr)
_GPU_BY_MEM = ((250_000, "B300"), (170_000, "B200"), (130_000, "H200"), (75_000, "H100"))


def _gpu_from_mem(mb: float | None) -> str | None:
    if not mb:
        return None
    return next((name for floor, name in _GPU_BY_MEM if mb >= floor), None)


# island_leave reasons that end a run on purpose; a failed trainer (INFRA_FAILED ...)
# is relaunched, so its samplers keep running and are not "done"
CLEAN_LEAVE = frozenset({"STOPPED", "DONE", "FINISHED", "COMPLETED", "DRAINED"})
# island statuses that mean the island is up (island_states / topology); everything else ended or is unknown
ALIVE_STATUSES = frozenset({"ok", "waiting", "starting", "stale"})
# a sampler that still writes groups this long after its trainers ended is running, not done
SAMPLER_AFTER_TRAINERS_S = 300.0  # its last round in flight may finish a few minutes after


def _restart(isl: dict, since: float | None) -> None:
    """A new launch of ``isl`` began at ``since``: its startup state starts over. State its
    own records set already (the echoed tape can deliver them before the store upload
    that dates the launch) is kept: only what is older than ``since`` is cleared."""
    def old(ts: Any) -> bool:
        return since is None or ts is None or ts < since

    starts = [t for t in isl.get("driver_starts") or [] if t is not None and not old(t)]
    if not starts:
        isl["driver_started"] = False
    isl["startup_steps"] = {k: v for k, v in isl["startup_steps"].items() if not old(v.get("ts"))}
    if old(isl.get("startup_step_ts")):
        isl["startup_step"], isl["startup_step_ts"] = None, None
    if old(isl.get("start_phase_ts")):
        isl["start_phase"], isl["start_phase_ts"] = None, None
    # an earlier launch that finished (rl_learner_finalized) does not finish this one, and its
    # policy version is not this launch's until this launch reports one
    if isl.get("finalized") and old(isl.get("finalized_ts")):
        isl["finalized"], isl["finalized_ts"] = False, None
    if isl.get("policy_version") is not None and old(isl.get("policy_version_ts")):
        isl["policy_version"], isl["policy_version_ts"] = None, since
    if old(isl.get("phase_ts")):
        isl["phase"] = None
    isl["launch_ts"] = since


def _clean_leave(left: dict | None) -> bool:
    return bool(left) and str(left.get("reason") or "").upper() in CLEAN_LEAVE


def _store_launch(source: str) -> str | None:
    """``<label>:<island>/<launch>`` (store_tapes source) -> ``<launch>``; None otherwise.
    Only the trainer's tape directory is one per launch (``l0-<host>-<utc>``); a sampler's
    folder outlives its in-place restarts (rl_sampler_join carries those)."""
    head, sep, rest = source.partition(":")
    if not sep or "/" not in rest or rest.startswith("/"):
        return None
    island, _, launch = rest.partition("/")
    return launch if island == "trainer" and launch and "/" not in launch else None


# startup stages: (label, trainer_start_phase / rl_startup_step that ends it)
START_STAGE = {"submitted": "container start", "container_ready": "island setup", "setup_done": "Ray start",
               "ray_ready": "actor import", "actor_import_done": "actor init (model + weights load)",
               "actor_init_done": "first step"}
INIT_STAGE_PHASES = frozenset({"actor_import_done"})
INIT_STAGE_STEPS = frozenset({"engine_ready"})  # waiting for weights_loaded


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


def _resource(r: dict, ts: float | None) -> dict:
    if r.get("available") is False:
        return {"available": False, "ts": ts, "util_pct": None, "mem_pct": None, "gpus": [],
                "reason": r.get("reason")}
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


TRAJ_ID_RE = re.compile(r"^(.+)-r\d+-row\d+-\d+$")  # <sampler incarnation>-r<round>-row<row>-<sample>


def _traj(r: dict) -> dict:
    """Compact per-trajectory record. Missing = None."""
    out = {"rid": r.get("rollout_id") if isinstance(r.get("rollout_id"), int) else None,
           "id": r.get("trajectory_id"), "task": r.get("task_id"),
           "start": finite(r.get("trajectory_started_at")), "end": finite(r.get("trajectory_ended_at")),
           "reward": finite(r.get("reward")), "aborted": r.get("aborted"),
           "turns": finite(r.get("turns")) if not isinstance(r.get("turns"), list) else len(r["turns"])}
    for k, src in TRAJ_PHASES:
        out[k] = finite(r.get(src))
    # stage 2 (reserved): the round a carried-over trajectory started in, and its version segments
    sr = r.get("started_rollout_id")
    out["started_rid"] = sr if isinstance(sr, int) and not isinstance(sr, bool) else None
    pv = r.get("policy_versions")
    out["versions"] = pv if isinstance(pv, list) and pv else None
    gi = r.get("group_index")
    out["group"] = gi if isinstance(gi, int) and not isinstance(gi, bool) else None
    return out


def _seg_version(seg: Any) -> Any:
    """A version segment is ``[version, start, end)`` (version_segments) or a bare version."""
    return seg[0] if isinstance(seg, (list, tuple)) and seg else seg


def _sum(vals: Iterable[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return round(sum(xs), 3) if xs else None


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
