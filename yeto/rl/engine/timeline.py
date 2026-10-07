"""Profile/epoch-tagged execution timeline accounting (rl-infra-spec task 1.7, design D9).

Pure accounting. ``IslandDriver(observe=True)`` emits the tagged events
(``rl_timeline_span``/``rl_readiness``/``rl_load_sample``/``rl_round_labels``);
``load_windows`` turns them into ``LoadSummary`` rows. observe=False emits none.

Two rules from the acceptance text:

* overlapping spans are never billed twice: wall time and per-role busy time
  are interval unions, so a partitioned-overlap run and a serial run are
  compared on the same clock;
* tool waiting is not GPU saturation: a load sample is classified from
  queued/active/tool-wait counts, never from total rollout time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

SPAN_KINDS = ("compute", "wait", "tool-wait", "transfer")


@dataclass(frozen=True)
class Span:
    task: str  # execution_profile.TASKS entry or a reconfiguration phase
    role: str  # trainer | rollout | cpu | trainer+rollout
    kind: str
    start: float
    end: float
    profile_hash: str
    epoch: int
    rollout_id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in SPAN_KINDS:
            raise ValueError(f"span kind must be one of {SPAN_KINDS}")
        if self.end < self.start:
            raise ValueError("span ends before it starts")


def _union(intervals: Iterable[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def summarize(spans: Iterable[Span]) -> dict[str, object]:
    """Union-based totals per role/kind/task. Mixed profiles or epochs are refused."""
    spans = list(spans)
    if not spans:
        return {"wall_s": 0.0, "by_role": {}, "by_kind": {}, "by_task": {}, "overlap_s": 0.0}
    keys = {(s.profile_hash, s.epoch) for s in spans}
    if len(keys) != 1:
        raise ValueError(f"one summary covers one (profile, epoch); got {sorted(keys)}")
    groups: dict[str, dict[str, list[tuple[float, float]]]] = {
        "by_role": defaultdict(list), "by_kind": defaultdict(list), "by_task": defaultdict(list)
    }
    for s in spans:
        for role in s.role.split("+"):
            groups["by_role"][role].append((s.start, s.end))
        groups["by_kind"][s.kind].append((s.start, s.end))
        groups["by_task"][s.task].append((s.start, s.end))
    wall = _union((s.start, s.end) for s in spans)
    summed = sum(s.end - s.start for s in spans)
    return {
        "profile_hash": spans[0].profile_hash,
        "epoch": spans[0].epoch,
        "wall_s": wall,
        "overlap_s": summed - wall,  # time that naive summing would double count
        **{g: {k: _union(v) for k, v in d.items()} for g, d in groups.items()},
    }


@dataclass(frozen=True)
class LoadSample:
    queued_requests: int
    active_requests: int
    tool_wait_trajectories: int
    ready_groups: int
    engine_capacity: int  # max concurrent requests the active engines accept
    # IR-2 (codex-harness R-IR): harness sessions open / sandbox leases alive
    # (idle included). Neither is a GPU gap; classify_load ignores both.
    harness_in_flight: int = 0
    env_live: int = 0


# -- IR-4: 1.7 ``rl_load_sample`` metric schema -------------------------------
# Every key the driver's load sampler may emit (``rl_load_sample`` events carry
# the ``profile_hash``/``epoch`` labels; observe=False emits nothing, so the
# old path is untouched). ``kind``: gauge = instantaneous, counter = cumulative
# since island start. ``tito_chain_breaks`` is a {reason: count} map with
# reasons restricted to TITO_CHAIN_BREAK_REASONS.
LOAD_SAMPLE_LABELS = ("profile_hash", "epoch")
TITO_CHAIN_BREAK_REASONS = (
    "retry_fork", "history_rewrite", "template_drops_reasoning", "compaction_window",
)
LOAD_SAMPLE_SCHEMA: dict[str, tuple[str, type]] = {
    "active_requests": ("gauge", int),
    "workers": ("gauge", int),
    "cordoned": ("gauge", int),
    "running_requests": ("gauge", int),
    "queued_requests": ("gauge", int),
    "engine_capacity": ("gauge", int),
    "tool_wait_trajectories": ("gauge", int),
    "ready_groups": ("gauge", int),
    "load_class": ("label", str),
    # IR-4 harness metrics
    "harness_in_flight": ("gauge", int),
    "env_live": ("gauge", int),
    "tito_session_mismatch": ("counter", int),
    "tito_chain_breaks": ("counter", dict),
    "policy_age_violation": ("counter", int),
    # 1.7 resource peaks (gauge since last sample; None when not probed)
    "peak_gpu_mem_bytes": ("gauge", int),
    "peak_cpu_rss_bytes": ("gauge", int),
}
HARNESS_METRIC_KEYS = (
    "harness_in_flight", "env_live", "tito_session_mismatch", "tito_chain_breaks",
    "policy_age_violation",
)

# -- S14-M1: per-record TITO session mismatches (``rl_harness_mismatch``) -----
# Observe-only (``--rl-observe-timeline``): one event per mismatch record that
# upstream Miles' session server stored in ``sample.metadata["tito_session_mismatch"]``
# (list[dict] from ``TokenSeqComparator.compare_sequences``), capped per round and
# with the texts truncated, so the tape can tell *which* kind/segment/text
# mismatched instead of only how many. observe=False emits nothing.
HARNESS_MISMATCH_EVENT = "rl_harness_mismatch"
HARNESS_MISMATCH_MAX_PER_ROUND = 64
HARNESS_MISMATCH_TEXT_MAX = 512
HARNESS_MISMATCH_KINDS = (
    "assistant_text", "non_assistant_text", "special_token_count", "special_token_type",
)
HARNESS_MISMATCH_SCHEMA: dict[str, type] = {
    "rollout_id": int,          # training round (driver rollout_id)
    "policy_version": int,      # batch policy version
    "sample_index": int,        # Miles Sample.index (-1 when the sample has none)
    "group_index": int,         # Miles Sample.group_index (-1 when absent)
    "record_index": int,        # position inside the sample's mismatch list
    "kind": str,                # upstream ``type`` (HARNESS_MISMATCH_KINDS or other)
    "segment_index": int,       # upstream segment_index (-1 = structural)
    "expected_text": str,       # truncated to HARNESS_MISMATCH_TEXT_MAX chars
    "actual_text": str,
    "detail": str,
    "truncated": bool,          # any text was cut
}


# -- rl-fn-codex-rollout 1.0: per-trajectory rewards (``rl_trajectory_reward``) --
# Observe-only: one event per trained sample with its Terminal-Bench task_id and
# the reward the trainer saw, so a tape can show which task scored (the judge's
# ``--known-scorable``). ``reward`` None = non-finite; ``success`` None = no signed verdict.
TRAJECTORY_REWARD_EVENT = "rl_trajectory_reward"
TRAJECTORY_REWARD_MAX_PER_ROUND = 256
TRAJECTORY_REWARD_SCHEMA: dict[str, tuple[type, ...]] = {
    "rollout_id": (int,),
    "policy_version": (int,),
    "sample_index": (int,),
    "group_index": (int,),
    "task_id": (str,),
    "trajectory_id": (str,),
    "reward": (float, type(None)),
    "success": (bool, type(None)),
    "aborted": (bool,),
}


# S15 stage-2 follow-up (observe only): why a trajectory scored what it did.
# Present only when the harness reported them (old key set kept otherwise):
# Codex exit status (completed / max_turns / max_seq_len / timeout), agent
# counters and the signed test.sh exit code.
TRAJECTORY_REWARD_OPTIONAL: dict[str, tuple[type, ...]] = {
    "exit_status": (str,),
    "turns": (int,),
    "terminal_calls": (int,),
    "submit_calls": (int,),
    "parse_failures": (int,),
    "max_seq_len_hit": (int,),
    "timed_out": (int,),
    "testsh_rc": (int, type(None)),
}


def validate_trajectory_reward(record: Mapping[str, object]) -> list[str]:
    """Schema check of one ``rl_trajectory_reward`` payload (labels excluded)."""
    problems = []
    for key, types in TRAJECTORY_REWARD_OPTIONAL.items():
        if key in record:
            value = record[key]
            if (int in types and isinstance(value, bool)) or not isinstance(value, types):
                problems.append(f"trajectory reward key {key!r} is {type(value).__name__}")
    for key, types in TRAJECTORY_REWARD_SCHEMA.items():
        if key not in record:
            problems.append(f"missing trajectory reward key {key!r}")
            continue
        value = record[key]
        if (int in types and isinstance(value, bool)) or not isinstance(value, types):
            problems.append(f"trajectory reward key {key!r} is {type(value).__name__}")
    return problems


def validate_harness_mismatch(record: Mapping[str, object]) -> list[str]:
    """Schema check of one ``rl_harness_mismatch`` payload (labels excluded)."""
    problems = []
    for key, typ in HARNESS_MISMATCH_SCHEMA.items():
        if key not in record:
            problems.append(f"missing mismatch key {key!r}")
            continue
        value = record[key]
        if typ is int and isinstance(value, bool) or not isinstance(value, typ):
            problems.append(f"mismatch key {key!r} is {type(value).__name__}, expected {typ.__name__}")
    for key in ("expected_text", "actual_text", "detail"):
        if isinstance(record.get(key), str) and len(record[key]) > HARNESS_MISMATCH_TEXT_MAX:
            problems.append(f"mismatch key {key!r} longer than {HARNESS_MISMATCH_TEXT_MAX}")
    return problems


def validate_load_sample(sample: Mapping[str, object]) -> list[str]:
    """Schema check of one load payload: unknown keys, wrong types and unknown
    chain-break reasons are reported (None = unknown is accepted everywhere)."""
    problems = []
    for key, value in sample.items():
        spec = LOAD_SAMPLE_SCHEMA.get(key)
        if spec is None:
            problems.append(f"unknown load key {key!r}")
            continue
        if value is None:
            continue
        kind, typ = spec
        if not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
            problems.append(f"{key}: expected {typ.__name__}, got {type(value).__name__}")
        elif key == "tito_chain_breaks":
            for reason, n in value.items():  # type: ignore[union-attr]
                if reason not in TITO_CHAIN_BREAK_REASONS:
                    problems.append(f"tito_chain_breaks: unknown reason {reason!r}")
                if not isinstance(n, int) or isinstance(n, bool):
                    problems.append(f"tito_chain_breaks[{reason!r}]: expected int")
    return problems


def classify_load(sample: LoadSample) -> str:
    """D9 attribution of one sample; never treats tool waiting as a GPU gap."""
    if sample.active_requests == 0 and sample.tool_wait_trajectories > 0:
        return "tool-wait"
    if sample.queued_requests > 0 and sample.active_requests >= sample.engine_capacity:
        return "rollout-saturated"
    if sample.queued_requests == 0 and 0 < sample.active_requests < sample.engine_capacity:
        return "long-tail"
    if sample.active_requests == 0 and sample.queued_requests == 0:
        return "rollout-idle"
    return "rollout-busy"


# -- task 5.1: reconfiguration cost distribution and bottleneck choice ---------
# Phases of one source->target transition (design D6/D10 vocabulary). "blocking"
# phases stop training; "background" ones overlap the next rounds.
TRANSITION_PHASES = (
    "wait_safe_point", "drain", "export", "init", "restore", "publish", "first_step",
)
BACKGROUND_PHASES = ("background_restore",)
# Pre-declared selection rule (local-gpu-plan.md 5.1; not tuned after the fact).
MIN_SAMPLES_PER_EDGE = 3


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def transition_cost_distribution(
    samples: Iterable[tuple[str, str, dict[str, float]]],
) -> dict[tuple[str, str], dict[str, object]]:
    """Per (source, target) edge: n and p50/p90/max seconds per phase and blocking total.

    ``samples`` are ``(source, target, {phase: seconds})`` of individual
    transitions; phase keys outside the vocabulary are refused (no silent
    buckets), a missing phase counts as 0 s.
    """
    by_edge: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    known = set(TRANSITION_PHASES) | set(BACKGROUND_PHASES)
    for source, target, phases in samples:
        unknown = set(phases) - known
        if unknown:
            raise ValueError(f"unknown transition phases {sorted(unknown)}")
        if any(v < 0 for v in phases.values()):
            raise ValueError("negative phase duration")
        by_edge[(source, target)].append(dict(phases))
    out: dict[tuple[str, str], dict[str, object]] = {}
    for edge, rows in by_edge.items():
        dist: dict[str, dict[str, float]] = {}
        for phase in (*TRANSITION_PHASES, *BACKGROUND_PHASES, "blocking_total"):
            if phase == "blocking_total":
                vals = [sum(r.get(p, 0.0) for p in TRANSITION_PHASES) for r in rows]
            else:
                vals = [r.get(phase, 0.0) for r in rows]
            dist[phase] = {"p50": _quantile(vals, 0.5), "p90": _quantile(vals, 0.9),
                           "max": max(vals)}
        out[edge] = {"n": len(rows), "phases": dist}
    return out


def select_bottleneck(distribution: dict[tuple[str, str], dict[str, object]]) -> dict[str, object]:
    """The ONE blocking phase to optimize first (5.1), by the pre-declared rule.

    Rule: every edge needs >= MIN_SAMPLES_PER_EDGE transitions; for each blocking
    phase take its p50 share of the edge's p50 blocking total, averaged over
    edges with equal weight; the largest mean share wins. Ties (within 1e-9)
    are reported, not broken, so a human records the choice.
    """
    if not distribution:
        return {"status": "insufficient", "reason": "no transitions measured"}
    thin = sorted(e for e, d in distribution.items() if d["n"] < MIN_SAMPLES_PER_EDGE)
    if thin:
        return {"status": "insufficient",
                "reason": f"edges with < {MIN_SAMPLES_PER_EDGE} samples: {thin}"}
    shares: dict[str, list[float]] = defaultdict(list)
    for d in distribution.values():
        phases = d["phases"]
        total = phases["blocking_total"]["p50"]
        for p in TRANSITION_PHASES:
            shares[p].append(phases[p]["p50"] / total if total > 0 else 0.0)
    mean = {p: sum(v) / len(v) for p, v in shares.items()}
    best = max(mean.values())
    winners = sorted(p for p, v in mean.items() if abs(v - best) <= 1e-9)
    if best <= 0:
        return {"status": "no-cost", "mean_share": mean}
    if len(winners) > 1:
        return {"status": "tie", "phases": winners, "mean_share": mean}
    return {"status": "selected", "phase": winners[0], "mean_share": mean}


# -- task 1.7: stable read-only load summary (consumed by 6.1 shadow attribution) --
GPU_ROLES = ("trainer", "rollout")


@dataclass(frozen=True)
class LoadSummary:
    """One fixed window of observed load for one (profile_hash, epoch).

    STABLE INTERFACE (rl-infra-spec 1.7 -> 6.1): field names do not change.

    * ``window_start``/``window_end``: driver clock seconds, ``[start, end)``.
    * ``gpu_busy_fraction``: union of ``compute`` spans on trainer/rollout
      roles, clipped to the window, / window length (overlap never billed twice).
    * ``publish_block_fraction``: union of ``transfer`` spans of task
      ``publish`` / window length.
    * ``tool_wait_fraction`` / ``tail_wait_fraction``: share of
      ``rl_load_sample`` events in the window classified ``tool-wait`` /
      ``long-tail`` by :func:`classify_load`; None when no classifiable sample.
    * ``queued`` / ``active`` / ``ready_groups``: mean over samples (ready
      groups from ``rl_readiness`` too); None when unobserved.
    * ``consume_rate``: trained groups (``rl_round_labels`` ``rl/groups``) per
      second in the window.
    * ``policy_age``: max ``rl_readiness.policy_age`` in the window, else None.
    * ``weight_transport``: transport label of the window's events (None if absent).
    * ``train_fraction`` (added for 6.1/6.4, appended so positional use is
      unchanged): union of ``compute`` spans on the trainer role / window length.
    * ``rollout_busy_fraction`` (appended, d2-wire): union of ``compute`` spans on
      the rollout role / window length -- the only part a rollout resize scales
      (``gpu_busy_fraction`` also contains trainer compute).
    """

    window_start: float
    window_end: float
    profile_hash: str | None
    epoch: int
    gpu_busy_fraction: float
    tool_wait_fraction: float | None
    tail_wait_fraction: float | None
    publish_block_fraction: float
    queued: float | None
    active: float | None
    ready_groups: float | None
    consume_rate: float
    policy_age: int | None
    weight_transport: str | None = None
    train_fraction: float = 0.0
    rollout_busy_fraction: float = 0.0


def _clip_union(intervals: list[tuple[float, float]], lo: float, hi: float) -> float:
    return _union((max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_windows(events: Iterable[Mapping[str, object]], window_s: float) -> list[LoadSummary]:
    """Bucket driver ``observe=True`` events into fixed windows per (profile, epoch).

    Reads ``rl_timeline_span`` (start/end), and ``rl_load_sample`` /
    ``rl_readiness`` / ``rl_round_labels`` (field ``t``); other events and
    untimed ones are ignored, so observe=False (no such events) yields ``[]``.
    Windows are aligned to the first timed event of each (profile, epoch).
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    by_key: dict[tuple[object, int], list[Mapping[str, object]]] = defaultdict(list)
    for ev in events:
        name = ev.get("event")
        if name == "rl_timeline_span" or (
            name in ("rl_load_sample", "rl_readiness") and ev.get("t") is not None
        ):
            by_key[(ev.get("profile_hash"), int(ev.get("epoch") or 0))].append(ev)
        elif name == "rl_round_labels" and ev.get("t") is not None:
            by_key[(ev.get("profile_hash"), int(ev.get("config_epoch") or 0))].append(ev)
    out: list[LoadSummary] = []
    for (profile, epoch), evs in by_key.items():
        def times(ev):
            return (ev["start"], ev["end"]) if ev["event"] == "rl_timeline_span" else (ev["t"], ev["t"])
        t0 = min(times(e)[0] for e in evs)
        t1 = max(times(e)[1] for e in evs)
        gpu = [(e["start"], e["end"]) for e in evs if e["event"] == "rl_timeline_span"
               and e.get("kind") == "compute"
               and any(r in GPU_ROLES for r in str(e.get("role")).split("+"))]
        train = [(e["start"], e["end"]) for e in evs if e["event"] == "rl_timeline_span"
                 and e.get("kind") == "compute" and "trainer" in str(e.get("role")).split("+")]
        roll = [(e["start"], e["end"]) for e in evs if e["event"] == "rl_timeline_span"
                and e.get("kind") == "compute" and "rollout" in str(e.get("role")).split("+")
                and e.get("task") != "eval"]  # eval generation is not training rollout load
        pub = [(e["start"], e["end"]) for e in evs if e["event"] == "rl_timeline_span"
               and e.get("kind") == "transfer" and e.get("task") == "publish"]
        transports = {e.get("weight_transport") for e in evs} - {None}
        n = max(1, int(-(-(t1 - t0) // window_s)))
        for i in range(n):
            lo, hi = t0 + i * window_s, t0 + (i + 1) * window_s
            last = i == n - 1
            inside = [e for e in evs if e["event"] != "rl_timeline_span"
                      and (lo <= e["t"] < hi or (last and e["t"] == hi))]
            samples = [e for e in inside if e["event"] == "rl_load_sample"]
            classes = []
            for s in samples:
                q, a, c = s.get("queued_requests"), s.get("running_requests",
                                                          s.get("active_requests")), s.get("engine_capacity")
                tw = s.get("tool_wait_trajectories")
                if None in (q, a, c, tw):
                    continue
                classes.append(classify_load(LoadSample(q, a, tw, s.get("ready_groups") or 0, c)))
            ready = [float(e["ready_groups"]) for e in inside
                     if e["event"] in ("rl_readiness", "rl_load_sample")
                     and e.get("ready_groups") is not None]
            ages = [int(e["policy_age"]) for e in inside
                    if e["event"] == "rl_readiness" and e.get("policy_age") is not None]
            groups = sum(int(e.get("rl/groups") or 0) for e in inside
                         if e["event"] == "rl_round_labels")
            out.append(LoadSummary(
                window_start=lo, window_end=hi, profile_hash=profile, epoch=epoch,
                gpu_busy_fraction=_clip_union(gpu, lo, hi) / window_s,
                tool_wait_fraction=(classes.count("tool-wait") / len(classes)) if classes else None,
                tail_wait_fraction=(classes.count("long-tail") / len(classes)) if classes else None,
                publish_block_fraction=_clip_union(pub, lo, hi) / window_s,
                queued=_mean([float(s["queued_requests"]) for s in samples
                              if s.get("queued_requests") is not None]),
                active=_mean([float(v) for s in samples if (v := s.get(
                    "running_requests", s.get("active_requests"))) is not None]),
                ready_groups=_mean(ready),
                consume_rate=groups / window_s,
                policy_age=max(ages) if ages else None,
                weight_transport=next(iter(transports)) if len(transports) == 1 else None,
                train_fraction=_clip_union(train, lo, hi) / window_s,
                rollout_busy_fraction=_clip_union(roll, lo, hi) / window_s,
            ))
    return sorted(out, key=lambda w: (str(w.profile_hash), w.epoch, w.window_start))
