"""Async GRPO (sample buffer) view: samplers, buffer backlog, releases, NC pool.

The buffer trainers and the samplers of an async run write their own event
types (``rl_sampler_*``, ``rl_group_policies``, ``rl_buffer_stale_skipped``,
``rl_policy_released`` ...); this module folds them into the tables and
series of the page's "Async GRPO" section. Pure observation of the tapes:
every number is read from a record, missing stays None.

Island roles are inferred from the events an island writes: ``sampler`` (any
``rl_sampler_*`` event), ``trainer`` (``rl_round_trained`` / ``rl_driver_start``).
Health of each role is judged by its own cadence (see ``expected_gap_s``).
"""

from __future__ import annotations

import math
import re
from typing import Any

POOL_PREFIX = "nc_elastic_"
POOL_MAX = 2000
ROUNDS_MAX = 5000
CADENCE_WINDOW = 5  # recent rounds whose duration sets the expected gap between events
CADENCE_FACTOR = 2.0
CADENCE_SLACK_S = 60.0
# no event for LOST_FACTOR x the expected gap: the island is counted lost (not active)
LOST_FACTOR = 3.0
SAMPLER_FLOOR_S = 300.0  # the shortest quiet time a sampler is allowed (3 x p50 round below it)
# events after which a sampler writes nothing by design until something else happens
SAMPLER_WAIT_EVENTS = {"rl_sampler_backlog_full": "backlog full (trainers behind)",
                       "rl_sampler_waiting_policy": "waiting for a policy release"}
SAMPLER_END_EVENTS = frozenset({"rl_sampler_stopped", "rl_sampler_data_end"})
TRAINER_EVENTS = frozenset({"rl_round_trained", "rl_driver_start", "rl_group_policies",
                            "rl_buffer_stale_skipped", "rl_policy_released", "rl_policy_release_skipped",
                            "rl_buffer_unidentified_policy_skipped"})
# sampler-reported reward/length summary keys (rl_sampler_engine_round)
# the sample-buffer reader's short-group handling per round (rl_buffer_stale_skipped)
FILL_KEYS = ("incomplete_groups", "dropped_small_groups", "carried_over_groups", "padded_samples")
SAMPLER_STAT_KEYS = ("reward_mean", "reward_std", "reward_p10", "reward_p50", "reward_p90",
                     "resp_len_mean", "resp_len_p95", "truncated", "zero_variance_seen")
# CompactionRL summary (sampler ``rl_sampler_engine_round.agentic.compaction``, trainer
# ``rl_round_trained.compaction``; yeto.rl.engine.timeline.CompactionTally): counts summed
# across rounds, ratios derived again from the sums. Keep in sync with
# timeline.COMPACTION_COUNT_KEYS (the dashboard does not import the engine).
COMPACTION_COUNT_KEYS = ("trials", "samples", "compacted_trials", "compactions", "segment_samples", "summaries",
                         "summary_ok", "truncated_trials", "truncated_compacted_trials", "exhausted_trials",
                         "exhausted_truncated_trials", "context_full_trials", "trainable_tokens", "summary_tokens",
                         "retry_replays", "retry_replay_trials", "fork_lost_trials", "compaction_exposed_trials")
COMPACTION_RATIOS = {"compacted_frac": ("compacted_trials", "trials"),
                     "compactions_mean": ("compactions", "trials"),
                     "samples_per_trial": ("samples", "trials"),
                     "segment_sample_frac": ("segment_samples", "samples"),
                     "summary_ok_frac": ("summary_ok", "summaries"),
                     "truncated_compacted_frac": ("truncated_compacted_trials", "compacted_trials"),
                     "summary_token_frac": ("summary_tokens", "trainable_tokens"),
                     "fork_lost_frac": ("fork_lost_trials", "compaction_exposed_trials")}
# per-round scalars shown as is (not summed): per-trial vs per-segment means, xhw_guard
COMPACTION_ROUND_KEYS = ("reward_mean_trial", "reward_mean_segment", "policy_tokens_mean_trial",
                         "policy_tokens_mean_segment", "xhw_downweighted_groups", "xhw_downweighted_trials",
                         "xhw_downweighted_samples", "xhw_trial_mean_shift")
SEGMENT_BUCKETS = ("0", "1", "2", "3", "4+", "unknown")  # "0": given up before it started
# pool events that end a shard's sampler (left: drained / released / fenced; lost: the node went away)
POOL_LEFT = {"nc_elastic_released": "left", "nc_elastic_released_externally": "left",
             "nc_elastic_shard_fenced": "left", "nc_elastic_sampler_finished": "done",
             "nc_elastic_sampler_crash_loop": "left", "nc_elastic_drain_failed": "left",
             "nc_elastic_scaled_down": "left", "nc_elastic_released_by_earlier_pool": "left",
             "nc_elastic_manual_release_detected": "left", "nc_elastic_sampler_config_error": "left",
             "nc_elastic_start_timeout": "left", "nc_elastic_ended_for_balance": "lost",
             "nc_elastic_evicted": "lost", "nc_elastic_expired": "lost", "nc_elastic_node_lost": "lost"}
POOL_UP = {"nc_elastic_member_ready", "nc_elastic_node_reachable_again", "nc_elastic_control_plane_recovered",
           "nc_elastic_scale_drain_cancelled", "nc_elastic_sampler_retry"}
# the follow-up events of a retirement (fence, release, drain): they never hide the cause before them
POOL_FOLLOW_UP = frozenset({"nc_elastic_shard_fenced", "nc_elastic_released", "nc_elastic_drain_failed"})
POOL_TARGET = "nc_elastic_target_changed"
# a scale-down victim finishing its trials in flight: still sampling, not left
POOL_DRAINING = "nc_elastic_scale_draining"
POOL_STOPPED = "nc_elastic_pool_stopped"
POOL_KEYS = ("ticket", "account", "slot", "slot_name", "shard", "why", "reason", "detail", "error", "state", "after_s",
             "waited_s", "restarts", "attempt", "data_slot", "old", "new", "target", "outcome", "source",
             "by", "usd_per_h", "end_reason", "replaces", "continues_shard", "max_frac", "alerts",
             "retry_in_s", "successor", "level", "action", "expected_ticket", "usable", "halted")
# a person's operation (pool-scale / nc down / console release / pool-pause): shown as manual
POOL_MANUAL_PREFIX = "nc_elastic_manual_"


def compaction_row(c: Any) -> dict | None:
    """The scalar counts, ratios and segment histogram of one round's compaction summary."""
    if not isinstance(c, dict):
        return None
    out: dict[str, Any] = {k: _f(c.get(k)) for k in (*COMPACTION_COUNT_KEYS, *COMPACTION_RATIOS, "compactions_max",
                                                    "max_compactions", *COMPACTION_ROUND_KEYS)}
    hist = c.get("segments_hist")
    out["segments_hist"] = {str(k): v for k, v in hist.items() if _f(v) is not None} if isinstance(hist, dict) else {}
    out["by_segments"] = _bucket_fate(c.get("by_segments"))
    return out


def _reasons(v: Any) -> dict[str, float]:
    return {str(k): n for k, n in v.items() if _f(n) is not None} if isinstance(v, dict) else {}


def _bucket_fate(by: Any) -> dict[str, dict]:
    """``by_segments`` of one round: per segment-count bucket, trials written and
    trials lost / dropped by reason."""
    if not isinstance(by, dict):
        return {}
    return {str(b): {"written": _f(v.get("written")) or 0.0, "lost": _reasons(v.get("lost")),
                     "dropped": _reasons(v.get("dropped"))}
            for b, v in by.items() if isinstance(v, dict)}


def segment_fate(rows: list[dict], stale: list[dict] = ()) -> dict[str, dict]:
    """Per bucket over rounds: written, lost and dropped (by reason), the share not
    written, plus the trainers' stale drops (``stale_by_segments``) when reported."""
    out: dict[str, dict] = {}
    for r in rows:
        for b, v in r.items():
            o = out.setdefault(b, {"written": 0.0, "lost": {}, "dropped": {}, "stale": None})
            o["written"] += v["written"]
            for kind in ("lost", "dropped"):
                for k, n in v[kind].items():
                    o[kind][k] = o[kind].get(k, 0) + n
    for r in stale:
        for b, n in r.items():
            o = out.setdefault(b, {"written": 0.0, "lost": {}, "dropped": {}, "stale": None})
            o["stale"] = (o["stale"] or 0) + n
    for o in out.values():
        bad = sum(o["lost"].values()) + sum(o["dropped"].values())
        total = o["written"] + bad
        o["not_written_frac"] = round(bad / total, 4) if total else None
    return {b: out[b] for b in sorted(out, key=lambda b: SEGMENT_BUCKETS.index(b) if b in SEGMENT_BUCKETS else 99)}


def compaction_total(rows: list[dict]) -> dict | None:
    """Rounds' compaction rows summed (None when no round carried one)."""
    rows = [r["compaction"] for r in rows if isinstance(r.get("compaction"), dict)]
    if not rows:
        return None
    out: dict[str, Any] = {"rounds": len(rows)}
    for k in COMPACTION_COUNT_KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = sum(vals) if vals else None
    for k, (num, den) in COMPACTION_RATIOS.items():
        out[k] = round(out[num] / out[den], 4) if out[num] is not None and out[den] else None
    maxes = [r["compactions_max"] for r in rows if r.get("compactions_max") is not None]
    out["compactions_max"] = max(maxes) if maxes else None
    out["max_compactions"] = next((r["max_compactions"] for r in reversed(rows) if r.get("max_compactions") is not None),
                                  None)
    hist: dict[str, float] = {}
    for r in rows:
        for k, v in r["segments_hist"].items():
            hist[k] = hist.get(k, 0) + v
    out["segments_hist"] = dict(sorted(hist.items(), key=lambda kv: _age_key(kv[0])))
    out["by_segments"] = segment_fate([r.get("by_segments") or {} for r in rows])
    xhw = [r for r in rows if r.get("xhw_trial_mean_shift") is not None]
    out["xhw_downweighted_groups"] = sum(r.get("xhw_downweighted_groups") or 0 for r in xhw) if xhw else None
    out["xhw_trial_mean_shift_max"] = max(r["xhw_trial_mean_shift"] for r in xhw) if xhw else None
    last = rows[-1]
    out.update({k: last.get(k) for k in ("reward_mean_trial", "reward_mean_segment",
                                         "policy_tokens_mean_trial", "policy_tokens_mean_segment")})
    return out


_ERROR_LINE = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception)): (.+)$")


def _latest_incarnation(incs: list[dict]) -> str | None:
    """The incarnation that started last (rows without a start time count as oldest)."""
    if not incs:
        return None
    return max(incs, key=lambda x: (x["first_ts"] is not None, x["first_ts"] or 0))["incarnation"]


def root_cause(error: Any) -> str | None:
    """The first ``SomeError: message`` line of a leave's error text: with chained
    exceptions ("The above exception was the direct cause ...") that is the root cause
    (a CUDA OOM), not the wrapper raised last (miles' NonRetryableError)."""
    for line in str(error or "").splitlines():
        m = _ERROR_LINE.match(line)
        if m:
            return f"{m.group(1)}: {m.group(2)}"[:240]
    return None


def trial_outcomes(agentic: dict) -> dict:
    """One ``rl_sampler_engine_round.agentic``: trials that ended in the round by outcome
    (``outcomes.harness_outcome_<o>_total`` are counts since the previous round event,
    yeto.rl.engine.agentic_round._outcomes: ok / policy_limit carry a verdict, aborted is
    an infrastructure failure), plus the incarnation's running infra-retry and lost totals
    by reason. The per-round ``lost_trials`` only counts when a group ends, so a sampler
    whose every trial fails shows it here first (the trial-failure alert)."""
    out: dict = {}
    oc = agentic.get("outcomes") if isinstance(agentic.get("outcomes"), dict) else {}
    for key in ("ok", "policy_limit", "aborted"):
        if _f(oc.get(f"harness_outcome_{key}_total")) is not None:
            out[f"trials_{key}"] = _f(oc.get(f"harness_outcome_{key}_total"))
    for key, src in (("infra_retries_by", "infra_retries_total"), ("lost_by", "lost_total")):
        if isinstance(agentic.get(src), dict):
            out[key] = _reasons(agentic[src])
    return out


def _f(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def quantile(vals: list[float], q: float) -> float | None:
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pool_ts(r: dict) -> float | None:
    """``nc_elastic_*`` records carry the wall clock in ``t`` (EventLog)."""
    return _f(r.get("t"))


def pool_state_event(p: dict) -> bool:
    """A pool event that :meth:`AsyncState.pool_scale` or :meth:`AsyncState.pool_exit`
    folds into state: a target change, a shard coming up / draining / leaving, the pool
    stop, a person's operation, an order (``replaced_by``) or any event tying a ticket to
    a shard. The rest (retries, refusals, warnings) only fill the event table."""
    ev = str(p.get("event"))
    return (ev in POOL_LEFT or ev in POOL_UP or ev in (POOL_TARGET, POOL_DRAINING, POOL_STOPPED,
                                                       "nc_elastic_ordered")
            or ev.startswith(POOL_MANUAL_PREFIX) or (p.get("shard") is not None and bool(p.get("ticket"))))


# rl_dispense_* kinds that end one attempt without settling the slot, and the kinds listed
DISPENSE_ENDS = ("expired", "release", "requeue", "lost")
DISPENSE_NOTED = DISPENSE_ENDS + ("scan_failed", "recover", "drain")


class AsyncState:
    def __init__(self) -> None:
        self.roles: dict[str, str] = {}
        self.sampler: dict[str, dict] = {}
        self.trainer: dict[str, dict] = {}
        self.releases: list[dict] = []
        self.loads: list[dict] = []
        self.drains: list[dict] = []  # rl_agentic_drain: the weight switch's wait for in-flight trials
        # task dispense ledger as the samplers saw it (rl_dispense_*): slot -> attempts / settle;
        # tapes of several samplers arrive out of time order, so state is keyed by attempt
        self.dispense: dict[int, dict] = {}
        self.dispense_events: list[dict] = []  # expiries, releases, requeues, losses, scan failures
        self.pool: list[dict] = []
        self.pool_shards: set[str] = set()  # every shard a pool event named (trimmed events too)
        self._pool_kept = 0  # self.pool[:_pool_kept] are all state events: the trim scan starts here

    # -- folding ----------------------------------------------------------------
    def _s(self, iid: str) -> dict:
        s = self.sampler.get(iid)
        if s is None:
            s = self.sampler[iid] = {"rounds": {}, "start": None, "last_wait": None, "ended": False,
                                     "backlog": [], "max_backlog_groups": None, "resource": None,
                                     "first_round_ts": None, "last_type": None, "last_ts": None,
                                     "vram": None,
                                     # one entry per sampler incarnation (an in-place restart is a new one)
                                     "incarnations": []}
        return s

    def _t(self, iid: str) -> dict:
        t = self.trainer.get(iid)
        if t is None:
            t = self.trainer[iid] = {"rounds": {}, "left": None, "walls": []}
        return t

    def fold_pool(self, r: dict) -> None:
        """Every pool event is kept until there are ``POOL_MAX``; past it the oldest event
        that no view folds into state (:func:`pool_state_event`) is dropped, so a long run's
        later member_ready / exit / target change still reaches the sampler count."""
        p = {"ts": pool_ts(r), "event": r.get("event"), **{k: r.get(k) for k in POOL_KEYS if r.get(k) is not None}}
        self.pool.append(p)
        if p.get("shard") is not None:
            self.pool_shards.add(str(p["shard"]))
        if len(self.pool) > POOL_MAX:
            i = self._pool_kept
            while i < len(self.pool) and pool_state_event(self.pool[i]):
                i += 1
            self._pool_kept = i
            if i < len(self.pool):
                del self.pool[i]

    def fold(self, r: dict, iid: str, ts: float | None, inc: str | None = None,
             boot_ts: float | None = None) -> None:
        """``inc``: the sampler incarnation the record's tape last announced (rl_sampler_join /
        rl_sampler_start earlier in the same source; "" = a known source that has announced none
        yet, None = no source). Tapes of several incarnations arrive interleaved (a run store
        lists their folders by name, not by time), so the latest join is not necessarily the
        incarnation a round belongs to. ``boot_ts``: on a join, the earliest record of its tape
        before it (engine start, model load), which starts the incarnation."""
        ev = r.get("event")
        if not isinstance(ev, str):
            return
        if ev.startswith("rl_sampler_"):
            self.roles[iid] = "sampler"
            self._fold_sampler(r, iid, ts, ev, inc, boot_ts)
        elif ev in TRAINER_EVENTS or ev in ("rl_rollout_carry_over", "offpolicy_gate", "rl_round_phases",
                                            "rl_round_wall", "island_leave",
                                            "rl_buffer_unidentified_policy_starved"):
            if ev in TRAINER_EVENTS and self.roles.get(iid) != "sampler":
                self.roles[iid] = "trainer"
            self._fold_trainer(r, iid, ts, ev)
        elif ev == "rl_agentic_drain":
            if ts is not None:  # matched to the policy load it preceded (releases view)
                self.drains.append({"sampler": iid, "ts": ts, "waited_s": _f(r.get("waited_s")),
                                    "in_flight": r.get("in_flight"), "timed_out": bool(r.get("timed_out")),
                                    "cut": sum(_f(r.get(k)) or 0 for k in ("abandoned_trials", "detached_trials"))})
                del self.drains[:-2000]
            if isinstance(r.get("drained_by_segments"), dict):
                # compaction on: the trials each drain cut, by segment count (agentic_round)
                d = self._s(iid).setdefault("drained", {})
                for b, n in _reasons(r["drained_by_segments"]).items():
                    d[b] = d.get(b, 0) + n
        elif ev.startswith("rl_dispense_"):
            self._fold_dispense(r, iid, ts, ev)
        elif ev == "rl_resource_sample" and iid in self.sampler:
            self.sampler[iid]["resource"] = {"available": r.get("available"), "reason": r.get("reason"),
                                             "source": r.get("source")}

    def _fold_dispense(self, r: dict, iid: str, ts: float | None, ev: str) -> None:
        kind = ev[len("rl_dispense_"):]
        slot, attempt = _int(r.get("slot")), _int(r.get("attempt"))
        if kind in DISPENSE_NOTED:
            self.dispense_events.append({"ts": ts, "kind": kind, "by": iid, "slot": slot, "attempt": attempt,
                                         **{k: r.get(k) for k in ("reason", "state", "seen", "task_id", "superseded_by")
                                            if r.get(k) is not None}})
            del self.dispense_events[:-500]
        if slot is None:
            return
        d = self.dispense.setdefault(slot, {"attempts": {}, "done": None, "done_ts": None})
        if kind == "claim" and attempt is not None:
            a = d["attempts"].setdefault(attempt, {"end": None, "end_ts": None})
            a.update({"by": iid, "claim_ts": ts, "task_id": r.get("task_id")})
        elif kind == "settle":
            d["done"], d["done_ts"] = r.get("state") or "settled", ts
            if attempt is not None:
                d["attempts"].setdefault(attempt, {}).update({"end": r.get("state") or "settled", "end_ts": ts})
        elif kind in DISPENSE_ENDS and attempt is not None:
            a = d["attempts"].setdefault(attempt, {})
            if a.get("end") is None:
                a.update({"end": kind, "end_ts": ts, "end_by": iid})

    def dispense_view(self) -> dict | None:
        """Slots by state (settled trained / dropped / exhausted, held now) and the claims
        still held, with who holds them: a sampler that crashed keeps its claims until their
        lease expires and another sampler re-claims them (attempt + 1)."""
        if not self.dispense and not self.dispense_events:
            return None
        settled: dict[str, int] = {}
        held: list[dict] = []
        reclaimed = 0
        for slot, d in self.dispense.items():
            if d["done"] is not None:
                settled[d["done"]] = settled.get(d["done"], 0) + 1
            if any(a > 0 for a in d["attempts"]):
                reclaimed += 1
            if d["done"] is None and d["attempts"]:
                top = max(d["attempts"])
                a = d["attempts"][top]
                if a.get("end") is None and a.get("by") is not None:
                    held.append({"slot": slot, "attempt": top, "by": a["by"], "claim_ts": a.get("claim_ts"),
                                 "task_id": a.get("task_id")})
        ends: dict[str, int] = {}
        for d in self.dispense.values():
            for a in d["attempts"].values():
                if a.get("end") in DISPENSE_ENDS:
                    ends[a["end"]] = ends.get(a["end"], 0) + 1
        return {"slots": len(self.dispense), "settled": dict(sorted(settled.items())), "ends": dict(sorted(ends.items())),
                "reclaimed_slots": reclaimed, "held": sorted(held, key=lambda x: (x["by"], x["slot"])),
                "events": self.dispense_events[-100:]}

    def _fold_sampler(self, r: dict, iid: str, ts: float | None, ev: str, hint: str | None = None,
                      boot_ts: float | None = None) -> None:
        s = self._s(iid)
        if ts is None or s["last_ts"] is None or ts >= s["last_ts"]:  # the newest record, not the last read
            s["last_type"], s["last_ts"] = ev, ts
        rid = _int(r.get("rollout_id"))
        inc = r.get("incarnation")
        if inc is not None and (ev in ("rl_sampler_join", "rl_sampler_start")
                                or not any(x["incarnation"] == str(inc) for x in s["incarnations"])):
            # a join / start, or any record naming an incarnation not seen yet (its join unread)
            inc = str(inc)
            row = next((x for x in s["incarnations"] if x["incarnation"] == inc), None)
            if row is None:
                row = {"incarnation": inc, "first_ts": ts, "last_ts": ts}
                s["incarnations"].append(row)
            for t in (ts, boot_ts):
                if t is not None:
                    row["first_ts"] = t if row["first_ts"] is None else min(row["first_ts"], t)
            latest = _latest_incarnation(s["incarnations"])
            if latest != s.get("incarnation"):
                s["ended"] = False  # a restart: the earlier incarnation's stop is not this one's
            s["incarnation"] = latest  # the newest by start time, whatever order the tapes arrive in
        # the incarnation this record belongs to: its own, else the one its tape announced last,
        # else (an old tape without the announcement in the same source) the newest one
        # ("" = its tape has not announced one yet: the record is not attributed)
        own = str(inc) if inc is not None else (hint or None) if hint is not None else s.get("incarnation")
        if s["incarnations"] and ts is not None and own is not None:
            cur = next((x for x in s["incarnations"] if x["incarnation"] == own), None)
            if cur is not None:
                cur["last_ts"] = ts if cur["last_ts"] is None else max(cur["last_ts"], ts)
        if ev == "rl_sampler_join":
            s["shard"] = r.get("sampler")
        # the newest engine start sets the sampler's settings: tapes of older incarnations may be read later
        if ev == "rl_sampler_start" and (ts is None or s.get("start") is None or ts >= s["start"]):
            s["start"] = ts
            s["max_backlog_groups"] = r.get("max_backlog_groups")
            s["cap_ts"] = ts
            s["sampling"] = {k: (r.get("sampling") or {}).get(k) for k in ("n", "max_tokens", "temperature")}
            s["provenance"] = {k: (r.get("provenance") or {}).get(k)
                               for k in ("card_model", "engine", "precision", "topology", "device_family")}
        if ev == "rl_sampler_round" and s["first_round_ts"] is None:
            s["first_round_ts"] = ts if ts is not None else 0.0
        rinc = own if own is not None else s.get("incarnation")
        key = (rinc, rid)  # a restarted sampler may count its rounds from 0 again
        if ev == "rl_sampler_round" and rid is not None:
            row = s["rounds"].setdefault(key, {"rollout_id": rid, "incarnation": rinc})
            row.update({"ts": ts, "policy_version": r.get("policy_version"), "version_lag": r.get("version_lag"),
                        "groups": r.get("groups"), "buffer_end": r.get("buffer_end"),
                        "generate_s": _f(r.get("generate_s"))})
            if s["first_round_ts"] is None:
                s["first_round_ts"] = ts
        elif ev == "rl_sampler_engine_round" and rid is not None:
            row = s["rounds"].setdefault(key, {"rollout_id": rid, "incarnation": rinc})
            finish = r.get("finish") if isinstance(r.get("finish"), dict) else {}
            if row.get("ts") is None:
                row["ts"] = ts  # rl_sampler_round (when it follows) sets the round's own time
            samples = _f(r.get("samples"))
            trunc = _f(finish.get("length"))
            row.update({"samples": samples, "response_tokens": _f(r.get("response_tokens")),
                        "dropped_groups": r.get("dropped_groups"), "dropped_reason": r.get("dropped_reason"),
                        "zero_variance_dropped": r.get("zero_variance_groups"),
                        "truncated": trunc if trunc is not None else (0.0 if finish else None),
                        "truncated_frac": (trunc or 0.0) / samples if samples and finish else None})
            for k in SAMPLER_STAT_KEYS:
                if r.get(k) is not None:
                    row[k] = r.get(k)
            if samples and _f(r.get("response_tokens")) is not None and row.get("resp_len_mean") is None:
                row["resp_len_mean"] = _f(r.get("response_tokens")) / samples
            agentic = r.get("agentic") if isinstance(r.get("agentic"), dict) else None
            for k in ("retried_trials", "lost_trials"):
                if agentic and _f(agentic.get(k)) is not None:
                    row[k] = _f(agentic.get(k))
            if agentic:  # trials ended this round by outcome; infra retries / lost so far
                row.update(trial_outcomes(agentic))
            if agentic and row.get("reward_mean") is None and _f(agentic.get("reward_mean")) is not None:
                row["reward_mean"] = _f(agentic.get("reward_mean"))
            comp = compaction_row((agentic or {}).get("compaction"))
            if comp is not None:  # absent without compaction (old row keys)
                row["compaction"] = comp
        elif ev in ("rl_sampler_vram_high", "rl_sampler_vram_ok"):
            # vram_guard.VramWatch: a card above the alert threshold, or every card back under
            v = s["vram"] or {"alerts": 0, "max_frac": None}
            high = ev == "rl_sampler_vram_high"
            frac = _f(r.get("max_frac"))
            v.update({"high": high, "ts": ts, "threshold": _f(r.get("threshold")), "last_frac": frac})
            if high:
                v["alerts"] += 1
                v["max_frac"] = frac if v["max_frac"] is None or frac is None else max(v["max_frac"], frac)
                v["cards"] = r.get("cards") if isinstance(r.get("cards"), list) else None
            s["vram"] = v
        elif ev == "rl_sampler_policy_loaded":
            ft = r.get("follower_timings") if isinstance(r.get("follower_timings"), dict) else {}
            seen = _f(r.get("seen_at"))
            self.loads.append({"sampler": iid, "policy_version": r.get("policy_version"), "seq": r.get("seq"),
                               "released_at": _f(r.get("released_at")), "seen_at": seen, "loaded_at": ts,
                               "resolve_s": _f(ft.get("resolve_s")), "load_s": _f(ft.get("load_s")),
                               "pause_s": _f(r.get("pause_s")), "buffer_selected": r.get("buffer_selected"),
                               "load_wall_s": round(ts - seen, 3) if ts is not None and seen is not None else None,
                               "memory_guard": (r.get("memory_guard") or {}).get("mode")
                               if isinstance(r.get("memory_guard"), dict) else None})
        elif ev == "rl_sampler_start_on_latest" and r.get("base_version") is not None:
            # a new or replacement island starts on the newest release: a load of that
            # version (at start, not a hot reload), so the release row shows the island
            self.loads.append({"sampler": iid, "policy_version": _int(r.get("base_version")), "seq": None,
                               "released_at": None, "seen_at": None, "loaded_at": ts, "resolve_s": None,
                               "load_s": None, "pause_s": None, "buffer_selected": None, "load_wall_s": None,
                               "at_start": True})
        elif ev in ("rl_sampler_backlog_full", "rl_sampler_admitted") or (
                ev == "rl_sampler_waiting" and r.get("reason") == "backlog_full"):
            # rl_sampler_waiting repeats the backlog the sampler sees while it waits
            b = _f(r.get("backlog_groups"))
            if ts is not None and b is not None:
                s["backlog"].append([ts, b, ev != "rl_sampler_admitted"])
                del s["backlog"][:-2000]
            if r.get("max_backlog_groups") is not None and (ts is None or s.get("cap_ts") is None or ts >= s["cap_ts"]):
                s["max_backlog_groups"] = r.get("max_backlog_groups")
                s["cap_ts"] = ts
        if ev in SAMPLER_END_EVENTS and rinc == s.get("incarnation"):
            s["ended"] = True  # an older incarnation's stop read late does not end the newest one
        if ev == "rl_sampler_waiting":
            reason = str(r.get("reason") or "")
            s["last_wait"] = SAMPLER_WAIT_EVENTS.get(f"rl_sampler_{reason}", f"waiting ({reason or 'no reason'})")
        else:
            s["last_wait"] = SAMPLER_WAIT_EVENTS.get(ev)

    def _fold_trainer(self, r: dict, iid: str, ts: float | None, ev: str) -> None:
        t = self._t(iid)
        rid = _int(r.get("rollout_id"))
        if ev == "island_leave":
            t["left"] = {"ts": ts, "reason": r.get("reason_name") or r.get("reason"),
                         "error_type": r.get("error_type"), "cause": root_cause(r.get("error"))}
            t.setdefault("leaves", []).append(t["left"])
            del t["leaves"][:-50]
            return
        if ev == "rl_buffer_unidentified_policy_starved":
            # the reader reads groups and keeps none (groups of versions no record names)
            t["unidentified_starved"] = {"ts": ts, "selections": r.get("selections"),
                                         "groups": r.get("unidentified_policy_groups_total")}
            return
        if ev == "rl_policy_release_skipped":
            # a release skipped because the previous one is still uploading (--rl-release-skip-busy)
            if r.get("reason") == "upload_in_flight" and ts is not None:
                t.setdefault("upload_skips", []).append({"ts": ts, "policy_version": r.get("policy_version")})
                del t["upload_skips"][:-200]
            return
        if ev == "rl_policy_released":
            origin = r.get("origin") if isinstance(r.get("origin"), dict) else {}
            timings = r.get("release_timings") if isinstance(r.get("release_timings"), dict) else {}
            self.releases.append({"trainer": iid, "policy_version": r.get("policy_version"),
                                  "local_version": r.get("local_version"), "seq": r.get("seq"),
                                  "released_at": _f(r.get("released_at")) or ts, "ts": ts,
                                  "bytes": r.get("release_bytes"), "upload_s": _f(timings.get("upload_s")),
                                  "origin": origin.get("kind"), "buffer_selected": r.get("buffer_selected")})
            return
        if ev in ("rl_round_phases", "rl_round_wall"):
            w = _f(r.get("wall_seconds")) if ev == "rl_round_phases" else _f(r.get("round_seconds"))
            if w is not None:
                t["walls"] = (t["walls"] + [w])[-CADENCE_WINDOW:]
            return
        if rid is None or len(t["rounds"]) >= ROUNDS_MAX and rid not in t["rounds"]:
            return
        row = t["rounds"].setdefault(rid, {"rollout_id": rid})
        if ev == "rl_round_trained":
            row.update({"ts": ts, "trained_groups": r.get("trained_groups"),
                        "trained_samples": r.get("trained_samples"),
                        "policy_age_mean": r.get("policy_age_mean"), "policy_age_max": r.get("policy_age_max")})
            pag = r.get("policy_age_groups")
            if isinstance(pag, dict) and pag:  # the trained groups' ages ({age: groups}); rl_group_policies
                # names the groups read, which a filter may still drop, so the trained ones win
                row["ages"] = {str(k): v for k, v in sorted(pag.items(), key=lambda kv: _age_key(kv[0]))}
                row["ages_trained"] = True
            comp = compaction_row(r.get("compaction"))
            if comp is not None:  # absent without compaction (old row keys)
                row["compaction"] = comp
        elif ev == "rl_buffer_unidentified_policy_skipped":
            row["unidentified_policy_groups"] = r.get("unidentified_policy_groups")
        elif ev == "rl_buffer_stale_skipped":
            row["stale_groups"] = r.get("stale_groups")
            row["max_policy_age"] = r.get("max_policy_age")
            if isinstance(r.get("stale_by_segments"), dict):  # compaction: stale trials per segment bucket
                row["stale_by_segments"] = _reasons(r["stale_by_segments"])
            for k in FILL_KEYS:
                if r.get(k) is not None:
                    row[k] = r.get(k)
        elif ev == "rl_group_policies":
            ages = r.get("ages") if isinstance(r.get("ages"), dict) else {}
            row["groups_read"] = len(r["groups"]) if isinstance(r.get("groups"), list) else None
            if not row.get("ages_trained"):
                row["ages"] = {str(k): v for k, v in sorted(ages.items(), key=lambda kv: _age_key(kv[0]))}
            row["merged_version"] = r.get("merged_version")
        elif ev == "offpolicy_gate":
            c = r.get("counts") if isinstance(r.get("counts"), dict) else {}
            row["gate_dropped"] = c.get("group_dropped")
            row["unknown_version_dropped"] = c.get("unknown_version_dropped")
        elif ev == "rl_rollout_carry_over":
            row["over_age_discarded_groups"] = r.get("over_age_discarded_groups")
            row["trained_response_tokens"] = r.get("trained_response_tokens")
        if ev == "rl_buffer_stale_skipped" and row.get("ts") is None:
            row["read_ts"] = ts  # the read happens before the step; rl_round_trained's ts wins

    # -- derived per-island facts used by the reducer's island cards ----------
    def role(self, iid: str) -> str | None:
        return self.roles.get(iid)

    def expected_gap_s(self, iid: str) -> float | None:
        """Longest quiet time this island's own cadence explains: a sampler writes
        one round event per generation and a load event per release (the load can
        take minutes); a trainer one round per step. ``CADENCE_FACTOR`` times the
        longest recent duration plus slack; None before the first round."""
        if self.roles.get(iid) == "sampler":
            s = self.sampler.get(iid) or {}
            rows = sorted((v for v in s.get("rounds", {}).values() if v.get("generate_s") is not None),
                          key=_row_order)[-CADENCE_WINDOW:]
            if not rows:
                return None
            loads = [x["load_wall_s"] for x in self.loads if x["sampler"] == iid and x["load_wall_s"] is not None]
            gens = sorted(v["generate_s"] for v in rows)
            return max(SAMPLER_FLOOR_S, 3.0 * gens[len(gens) // 2], CADENCE_FACTOR * gens[-1]) \
                + (max(loads) if loads else 0.0) + CADENCE_SLACK_S
        t = self.trainer.get(iid) or {}
        if t.get("walls"):
            return CADENCE_FACTOR * max(t["walls"]) + CADENCE_SLACK_S
        return None

    def sampler_state(self, iid: str) -> dict:
        s = self.sampler.get(iid) or {}
        return {"started_rounds": s.get("first_round_ts") is not None,
                "waiting": s.get("last_wait"), "ended": bool(s.get("ended"))}

    def pool_exit(self, iid: str, after: float | None = None) -> dict | None:
        """The NC pool's word on sampler ``iid`` (its shard): the last exit event of
        the shard, or of a ticket that served it, with no member_ready after it; a
        pool_stopped ends every shard. None when the pool has not ended it (or the
        exit is older than ``after``, the sampler's last event: a restart since).

        ``reason`` names the cause: the first exit event of the ticket (evicted,
        crash loop, scale down, ...), not the fence / release that follow it;
        ``replaced_by`` is the ticket ordered to replace the one that left."""
        tickets = {p.get("ticket") for p in self.pool if p.get("shard") is not None
                   and str(p.get("shard")) == str(iid) and p.get("ticket")}
        state = None
        events = sorted(self.pool, key=lambda x: x.get("ts") or 0)
        for p in events:
            ev = p.get("event")
            mine = (p.get("shard") is not None and str(p.get("shard")) == str(iid)) or (
                p.get("shard") is None and p.get("ticket") in tickets)
            if ev == POOL_STOPPED:
                cause = state if state and state["ticket"] is not None else None
                state = {"status": "left", "event": ev, "ts": p.get("ts"), "why": p.get("why"), "ticket": None,
                         "reason": cause["reason"] if cause else exit_reason(p),
                         "cause": cause["cause"] if cause else ev,
                         "cause_ts": cause["cause_ts"] if cause else p.get("ts")}
            elif mine and ev in POOL_LEFT:
                keep = (state is not None and ev in POOL_FOLLOW_UP and state["ticket"] == p.get("ticket")
                        and state["cause"] not in POOL_FOLLOW_UP)
                state = {"status": POOL_LEFT[ev], "event": ev, "ts": p.get("ts"),
                         "why": p.get("why") or p.get("detail"), "ticket": p.get("ticket"),
                         "reason": state["reason"] if keep else exit_reason(p),
                         "cause": state["cause"] if keep else ev,
                         "cause_ts": state.get("cause_ts") if keep else p.get("ts")}
            elif mine and ev in POOL_UP:
                state = None
        if state and after is not None and state["ts"] is not None and state["ts"] < after - 5:
            return None
        if state and state.get("ticket"):
            state["replaced_by"] = next(
                (p.get("ticket") for p in events if p.get("event") == "nc_elastic_ordered"
                 and p.get("replaces") == state["ticket"]), None)
        return state

    def trainer_left(self, iid: str) -> dict | None:
        return (self.trainer.get(iid) or {}).get("left")

    # -- view ---------------------------------------------------------------------
    def active(self) -> bool:
        return bool(self.sampler or self.releases or self.pool or any(
            t["rounds"] and any("ages" in r or "stale_groups" in r for r in t["rounds"].values())
            for t in self.trainer.values()))

    def last_write_ts(self, iid: str) -> float | None:
        """When the sampler last wrote groups (a round with groups > 0)."""
        ts = [r.get("ts") for r in (self.sampler.get(iid) or {}).get("rounds", {}).values()
              if r.get("ts") is not None and (_f(r.get("groups")) or 0) > 0]
        return max(ts) if ts else None

    def view(self) -> dict | None:
        if not self.active():
            return None
        samplers = {}
        for iid, s in self.sampler.items():
            rows = sorted(s["rounds"].values(), key=_row_order)
            # empty rounds (no group written: backlog / policy waits) would push every
            # reward-bearing round out of the 500-row window of the charts
            full = [r for r in rows if not _empty_round(r)]
            lags = [_f(r.get("version_lag")) for r in rows if _f(r.get("version_lag")) is not None]
            hist: dict[str, int] = {}
            for lag in lags:
                hist[str(int(lag))] = hist.get(str(int(lag)), 0) + 1
            gens = [r["generate_s"] for r in rows if r.get("generate_s") is not None]

            def tot(key: str, _rows: list = rows) -> float | None:
                vals = [_f(r.get(key)) for r in _rows if _f(r.get(key)) is not None]
                return sum(vals) if vals else None

            last = rows[-1] if rows else {}
            rew = next((r for r in reversed(rows) if r.get("reward_mean") is not None), None)
            incs = sorted(s["incarnations"], key=lambda x: (x["first_ts"] is None, x["first_ts"] or 0))
            samplers[iid] = {
                "rounds": full[-500:], "n_rounds": len(rows), "empty_rounds": len(rows) - len(full),
                "last_round": last or None,
                "policy_version": last.get("policy_version"),
                "lag_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
                "lag_p50": quantile(lags, 0.5), "lag_max": max(lags) if lags else None,
                "generate_p50_s": quantile(gens, 0.5), "generate_max_s": max(gens) if gens else None,
                "groups": tot("groups"), "samples": tot("samples"), "truncated": tot("truncated"),
                "zero_variance_dropped": tot("zero_variance_dropped"), "zero_variance_seen": tot("zero_variance_seen"),
                "dropped_groups": tot("dropped_groups"),
                "retried_trials": tot("retried_trials"), "lost_trials": tot("lost_trials"),
                # trials ended by outcome over every round (trial_outcomes)
                "trials_ok": tot("trials_ok"), "trials_policy_limit": tot("trials_policy_limit"),
                "trials_aborted": tot("trials_aborted"),
                # the last round that carried a reward (empty rounds carry none)
                "reward_mean_last": rew.get("reward_mean") if rew else None,
                "reward_last_ts": rew.get("ts") if rew else None,
                "reward_last_rollout_id": rew.get("rollout_id") if rew else None,
                # incarnations in start order; every one after the first is a restart
                "incarnations": incs,
                "restarts": [{"ts": x["first_ts"], "incarnation": x["incarnation"]} for x in incs[1:]],
                "backlog": s["backlog"][-500:], "max_backlog_groups": s["max_backlog_groups"],
                "waiting": s["last_wait"], "ended": s["ended"], "resource": s["resource"],
                "vram": s.get("vram"),
                "sampling": s.get("sampling"), "provenance": s.get("provenance"),
                "incarnation": incs[-1]["incarnation"] if incs else s.get("incarnation"), "shard": s.get("shard"),
                "first_ts": s.get("start"), "last_ts": s.get("last_ts"), "last_type": s.get("last_type"),
            }
            comp = compaction_total(rows)
            if comp is None and s.get("drained"):
                comp = {"rounds": 0}
            if comp is not None:  # absent without compaction (old view keys)
                if s.get("drained"):  # trials cut by drains (rl_agentic_drain), by segment count
                    comp["drained_by_segments"] = dict(sorted(
                        s["drained"].items(), key=lambda kv: SEGMENT_BUCKETS.index(kv[0])
                        if kv[0] in SEGMENT_BUCKETS else 99))
                samplers[iid]["compaction"] = comp
        trainers = {}
        for iid, t in self.trainer.items():
            rows = [t["rounds"][k] for k in sorted(t["rounds"])]
            if not rows:
                continue
            ages: dict[str, float] = {}
            for r in rows:
                for k, v in (r.get("ages") or {}).items():
                    if _f(v) is not None:
                        ages[k] = ages.get(k, 0) + v

            def tot(key: str, _rows: list = rows) -> float | None:
                vals = [_f(r.get(key)) for r in _rows if _f(r.get(key)) is not None]
                return sum(vals) if vals else None

            trainers[iid] = {"rounds": rows[-500:], "age_hist": dict(sorted(ages.items(), key=lambda kv: _age_key(kv[0]))),
                             "stale_groups": tot("stale_groups"), "gate_dropped": tot("gate_dropped"),
                             "unidentified_policy_groups": tot("unidentified_policy_groups"),
                             "unidentified_starved": t.get("unidentified_starved"),
                             "over_age_discarded_groups": tot("over_age_discarded_groups"),
                             "trained_groups": tot("trained_groups"), "left": t["left"],
                             **{k: tot(k) for k in FILL_KEYS}}
            comp = compaction_total(rows)
            stale = [r["stale_by_segments"] for r in rows if r.get("stale_by_segments")]
            if comp is None and stale:
                comp = {"rounds": 0}
            if comp is not None:  # absent without compaction (old view keys)
                if stale:  # trained (written) and stale-dropped trials per bucket
                    comp["by_segments"] = segment_fate([r["compaction"].get("by_segments") or {} for r in rows
                                                        if r.get("compaction")], stale)
                trainers[iid]["compaction"] = comp
        return {"samplers": samplers, "trainers": trainers, "backlog": self.backlog_series(),
                "releases": self.release_rows(), "pool": self.pool[-500:], "pool_scale": self.pool_scale(),
                "dispense": self.dispense_view(),
                "roles": dict(self.roles)}

    def pool_scale(self) -> dict | None:
        """The NC pool's sampler count over time: the target (``target_changed``, the
        scale commands) and the shards it runs (ready, not yet left): one point per
        pool event that changes either. None without pool events."""
        if not self.pool:
            return None
        live: set[str] = set()
        draining: set[str] = set()
        paused: set[str] = set()
        target = source = None
        points = []
        manual = []
        for p in sorted(self.pool, key=lambda x: x.get("ts") or 0):
            ev, shard = p.get("event"), p.get("shard")
            before = (target, len(live), len(draining))
            if str(ev).startswith(POOL_MANUAL_PREFIX):
                manual.append(p)
            if ev in (POOL_TARGET, "nc_elastic_manual_release_detected") and _int(p.get("new")) is not None:
                target = p["new"]
                source = p.get("source") or ("manual_release" if ev != POOL_TARGET else "manual")
            elif ev == POOL_STOPPED:
                live.clear()
                draining.clear()
                paused.clear()
            elif shard is not None and ev == "nc_elastic_manual_pause":
                paused.add(str(shard))
            elif shard is not None and ev in ("nc_elastic_manual_resume", "nc_elastic_manual_pause_ended"):
                paused.discard(str(shard))
            elif shard is not None and ev in POOL_UP:
                live.add(str(shard))
                draining.discard(str(shard))
            elif shard is not None and ev == POOL_DRAINING:
                draining.add(str(shard))
            elif shard is not None and ev in POOL_LEFT:
                live.discard(str(shard))
                draining.discard(str(shard))
                paused.discard(str(shard))
            if (target, len(live), len(draining)) != before:
                points.append({"ts": p.get("ts"), "target": target, "ready": len(live), "draining": len(draining),
                               "source": source})
        return {"target": target, "target_source": source, "manual": source in ("manual", "manual_release"),
                "ready": sorted(live), "draining": sorted(draining), "paused": sorted(paused),
                "manual_events": manual[-50:], "points": points[-500:]}

    def backlog_series(self) -> dict | None:
        """Groups in the buffer not yet read by the trainers: groups written by all
        samplers (``rl_sampler_round.groups``) minus the trainers' read position
        (``buffer_selected`` of ``rl_policy_released`` / ``rl_sampler_policy_loaded``,
        the position the samplers' own backlog rule starts from). Points at every
        write and every read-position update; None without sampler rounds."""
        flow = [(r["ts"], 0, _f(r.get("groups")))
                for s in self.sampler.values() for r in s["rounds"].values()
                if r.get("ts") is not None and _f(r.get("groups"))]
        if not flow:
            return None
        reads = [(x["ts"], 1, _f(x["buffer_selected"])) for x in self.releases
                 if x["ts"] is not None and _f(x.get("buffer_selected")) is not None]
        reads += [(x["loaded_at"], 1, _f(x["buffer_selected"])) for x in self.loads
                  if x["loaded_at"] is not None and _f(x.get("buffer_selected")) is not None]
        source = "buffer_selected" if reads else None
        if not reads:
            # no buffer_selected on the tapes (e.g. no release yet): the trainers' read
            # position is the groups they consumed (rl_group_policies + stale skipped)
            done = 0.0
            for ts, n in sorted((r.get("ts") or r.get("read_ts"), (r.get("groups_read") or 0) + (_f(r.get("stale_groups")) or 0))
                                for t in self.trainer.values() for r in t["rounds"].values()
                                if (r.get("ts") or r.get("read_ts")) is not None and r.get("groups_read") is not None):
                done += n
                reads.append((ts, 1, done))
            source = "trainer groups consumed" if reads else None
        flow.sort(key=lambda x: x[0])
        written, read, pts = 0.0, None, []
        for ts, kind, v in sorted(flow + reads, key=lambda x: (x[0], x[1])):
            if kind == 0:
                written += v
            else:
                read = v if read is None else max(read, v)
            pts.append([ts, written, read, None if read is None else max(0.0, written - read)])
        step = max(1, len(pts) // 400)
        out = pts[::step]
        if out[-1] is not pts[-1]:
            out.append(pts[-1])
        reported = {iid: s["backlog"] for iid, s in self.sampler.items() if s["backlog"]}
        return {"points": out, "written": written, "read": read, "has_reads": read is not None, "read_source": source,
                "reported": reported}

    def release_rows(self) -> list[dict]:
        """One row per released policy version: when the trainer released it and
        when each sampler finished loading it (loaded_at - released_at = latency)."""
        by_v: dict[Any, dict] = {}
        for rel in self.releases:
            v = rel["policy_version"]
            row = by_v.setdefault(v, {"policy_version": v, "loads": {}})
            row.update({k: rel[k] for k in ("trainer", "released_at", "upload_s", "bytes", "origin", "local_version")})
        for ld in self.loads:
            v = ld["policy_version"]
            row = by_v.setdefault(v, {"policy_version": v, "loads": {}, "released_at": ld["released_at"]})
            if row.get("released_at") is None:
                row["released_at"] = ld["released_at"]
            rel = row.get("released_at")
            # the drain this load waited for: the sampler's drain between seeing the release and loading it
            dr = [d for d in self.drains if d["sampler"] == ld["sampler"] and ld["seen_at"] is not None
                  and ld["loaded_at"] is not None and ld["seen_at"] <= d["ts"] <= ld["loaded_at"]]
            row["loads"][ld["sampler"]] = {
                "drain_s": dr[-1]["waited_s"] if dr else None, "drain_in_flight": dr[-1]["in_flight"] if dr else None,
                "drain_cut": dr[-1]["cut"] if dr else None, "drain_timed_out": dr[-1]["timed_out"] if dr else None,
                "pause_s": ld["pause_s"],
                "loaded_at": ld["loaded_at"], "seen_at": ld["seen_at"], "load_wall_s": ld["load_wall_s"],
                "resolve_s": ld["resolve_s"], "load_s": ld["load_s"], "at_start": bool(ld.get("at_start")),
                "latency_s": (round(ld["loaded_at"] - rel, 3) if ld["loaded_at"] is not None and rel is not None
                              and not ld.get("at_start") else None)}
        return [by_v[k] for k in sorted(by_v, key=lambda v: (v is None, v if isinstance(v, int) else 0))]


def exit_reason(p: dict) -> str:
    """One line naming why the pool ended a sampler, from its pool event."""
    ev = str(p.get("event") or "").removeprefix(POOL_PREFIX)
    end = p.get("end_reason")
    detail = str(p.get("detail") or p.get("why") or p.get("error") or "")[:160]
    if ev == "evicted":
        return "evicted" + (f" ({end})" if end else "")
    if ev == "expired":
        return "lease expired" + (f" ({end})" if end else "")
    if ev == "released_externally":
        return "released outside the pool" + (f" ({end})" if end else "")
    if ev == "released_by_earlier_pool":
        return "released by an earlier pool process"
    if ev == "manual_release_detected":
        return "released by hand" + (f" ({p['by']})" if p.get("by") else "")
    if ev == "scaled_down":
        return "scaled down" + (f" ({p['outcome']})" if p.get("outcome") else "")
    if ev == "sampler_crash_loop":
        n = p.get("restarts")
        return "sampler crash loop" + (f" after {n} restarts" if n is not None else "") + (
            f": {detail}" if detail else "")
    if ev == "sampler_config_error":
        return "configuration error" + (f": {detail}" if detail else "")
    if ev == "ended_for_balance":
        return "ended for the account balance"
    return ev.replace("_", " ") + (f": {detail}" if detail else "")


def _row_order(r: dict) -> tuple:
    ts = r.get("ts", r.get("read_ts"))
    return (ts is None, ts if ts is not None else 0.0, r.get("rollout_id") or 0)


def _empty_round(r: dict) -> bool:
    return r.get("reward_mean") is None and not _f(r.get("groups")) and not _f(r.get("samples"))


def _age_key(k: Any) -> int:
    try:
        return int(k)
    except (TypeError, ValueError):
        return 1 << 30
