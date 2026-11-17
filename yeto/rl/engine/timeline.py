"""Profile/epoch-tagged execution timeline accounting.

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

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

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
    # agentic-rollout-utilization 1.3: engine KV tokens in use / KV capacity
    "kv_used_tokens": ("gauge", int),
    "kv_capacity_tokens": ("gauge", int),
}


def load_peaks(events: Iterable[Mapping[str, object]], start: float, end: float) -> dict[str, float | None]:
    """agentic-rollout-utilization 1.3: queue and KV peaks of the ``rl_load_sample``
    events in ``[start, end]`` (e.g. one round's generation span). None = no sample
    reported the field."""
    queued: list[float] = []
    kv: list[float] = []
    for ev in events:
        t = ev.get("t")
        if ev.get("event") != "rl_load_sample" or not isinstance(t, (int, float)) or not start <= t <= end:
            continue
        if isinstance(ev.get("queued_requests"), int):
            queued.append(float(ev["queued_requests"]))
        used, cap = ev.get("kv_used_tokens"), ev.get("kv_capacity_tokens")
        if isinstance(used, int) and isinstance(cap, int) and cap > 0:
            kv.append(used / cap)
    return {"peak_queued_requests": max(queued) if queued else None,
            "peak_kv_fraction": max(kv) if kv else None}
HARNESS_METRIC_KEYS = (
    "harness_in_flight", "env_live", "tito_session_mismatch", "tito_chain_breaks",
    "policy_age_violation",
)

# -- per-record TITO session mismatches (``rl_harness_mismatch``) -----
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


# The --data rows one round trained, all of
# them (``rl_trajectory_reward`` stops at TRAJECTORY_REWARD_MAX_PER_ROUND samples).
# ``rows``: distinct row numbers as arithmetic runs [first, last, step]
# (``data_split.row_runs``); ``samples``: trained samples that carried a row.
ROUND_DATA_ROWS_EVENT = "rl_round_data_rows"
ROUND_DATA_ROWS_SCHEMA: dict[str, tuple[type, ...]] = {
    "rollout_id": (int,),
    "policy_version": (int,),
    "samples": (int,),
    "row_count": (int,),
    "rows": (list,),
}


def validate_round_data_rows(record: Mapping[str, object]) -> list[str]:
    """Schema check of one ``rl_round_data_rows`` payload (labels excluded)."""
    problems = []
    for key, types in ROUND_DATA_ROWS_SCHEMA.items():
        value = record.get(key)
        if key not in record or (int in types and isinstance(value, bool)) or not isinstance(value, types):
            problems.append(f"round data rows key {key!r} is missing or {type(value).__name__}")
    runs = record.get("rows")
    if isinstance(runs, list) and not all(
            isinstance(run, list) and len(run) == 3 and all(type(v) is int for v in run) for run in runs):
        problems.append("round data rows 'rows' is not a list of [first, last, step] runs")
    return problems


# Observe only: why a trajectory scored what it did.
# Present only when the harness reported them (old key set kept otherwise):
# Codex exit status (completed / max_turns / max_seq_len / timeout), agent
# counters and the signed test.sh exit code.
TRAJECTORY_REWARD_OPTIONAL: dict[str, tuple[type, ...]] = {
    # The sample's row number in the unsplit --data, on an
    # island that reads a row slot (``data_split.DATA_ROW_KEY``)
    "data_row": (int,),
    "exit_status": (str,),
    "turns": (int,),
    "terminal_calls": (int,),
    "submit_calls": (int,),
    "parse_failures": (int,),
    "max_seq_len_hit": (int,),
    "timed_out": (int,),
    "testsh_rc": (int, type(None)),
    # Why Codex stopped and the shape of its last reply.
    "end_reason": (str,),
    "last_finish_reason": (str,),
    "last_content_chars": (int,),
    "last_reasoning_chars": (int,),
    "last_tool_calls": (int,),
    "last_tool_names": (str,),
    "last_completion_tokens": (int, type(None)),
    "last_content_head": (str,),
    "last_reasoning_tail": (str,),
    "verifier_log": (str,),
    # Finer end reason than ``exit_status`` (submit / completed_no_submit
    # / turn_limit / protocol_error / response_truncated / context_limit /
    # timeout) and per-turn lengths: model reply tokens, context tokens after
    # each reply, tool-result bytes fed back.  Lists hold ints (None = no usage).
    "end_kind": (str,),
    "turn_completion_tokens": (list,),
    "turn_context_tokens": (list,),
    "turn_tool_output_bytes": (list,),
    # Seconds the trusted verifier spent grading this trajectory
    # (absent when grading was skipped, e.g. timeout).
    "evaluate_time": (float,),
    # agentic-rollout-utilization 1.3: wall start/end, sandbox cold start and
    # per-turn model-generation / tool seconds (engine.trajectory_timing).
    "trajectory_started_at": (float,),
    "trajectory_ended_at": (float,),
    "sandbox_start_seconds": (float,),
    "worker_seconds": (float,),
    "turn_generation_seconds": (list,),
    "turn_tool_seconds": (list,),
    "generation_seconds": (float,),
    "tool_seconds": (float,),
    # agentic-rollout-utilization 6.2: oldest/newest generating version of a
    # carried-over trajectory (absent at limit 0: every token is current)
    "policy_versions": (list,),
    # agentic-rollout-utilization 4.1: the round a carried-over trajectory
    # started in (Miles ``start_rollout_id``); absent at limit 0
    "started_rollout_id": (int,),
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
    """Time attribution of one sample; never treats tool waiting as a GPU gap."""
    if sample.active_requests == 0 and sample.tool_wait_trajectories > 0:
        return "tool-wait"
    if sample.queued_requests > 0 and sample.active_requests >= sample.engine_capacity:
        return "rollout-saturated"
    if sample.queued_requests == 0 and 0 < sample.active_requests < sample.engine_capacity:
        return "long-tail"
    if sample.active_requests == 0 and sample.queued_requests == 0:
        return "rollout-idle"
    return "rollout-busy"


# -- reconfiguration cost distribution and bottleneck choice ---------
# Phases of one source->target transition. "blocking"
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

    STABLE INTERFACE: field names do not change.

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


# -- CompactionRL segment metrics (``metadata.compaction``) --------------------
# With context compaction on, one trial (trajectory) becomes ``num_segments``
# buffer samples that share its reward; each carries ``metadata.compaction``
# (schema 1: trajectory_id, segment_index, num_segments, context_budget,
# summary_ok, kept_steps, segment_tokens, tokens_after; optional summary_tokens,
# compactions, context_full). A sample without that key is one whole trial.
# The same summary is read on the sampler (one round's written samples, under
# ``rl_sampler_engine_round.agentic.compaction``) and on the trainer (one step's
# trained Miles samples). Without any compaction data it returns None, so the
# events and the payloads keep their field set.
COMPACTION_KEY = "compaction"
COMPACTION_STATS_SCHEMA = 1
# Counts (summed across rounds by readers) and the ratios derived from them.
COMPACTION_COUNT_KEYS = (
    "trials", "samples", "compacted_trials", "compactions", "segment_samples", "summaries",
    "summary_ok", "truncated_trials", "truncated_compacted_trials", "exhausted_trials",
    "exhausted_truncated_trials", "context_full_trials", "trainable_tokens", "summary_tokens",
    # gateway (records ``gateway`` snapshot): lost replies replayed; trials invalidated for a
    # fork / history rewrite of a compacted session; written compacted + fork-lost trials
    "retry_replays", "retry_replay_trials", "fork_lost_trials", "compaction_exposed_trials",
)
COMPACTION_RATIO_KEYS = (
    "compacted_frac", "compactions_mean", "samples_per_trial", "segment_sample_frac",
    "summary_ok_frac", "truncated_compacted_frac", "summary_token_frac", "fork_lost_frac",
)
FORK_LOST_MARK = "compaction_fork_lost"  # gateway invalidation reason prefix


def _int_field(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _field(sample: Any, key: str) -> Any:
    return sample.get(key) if isinstance(sample, Mapping) else getattr(sample, key, None)


def _status_text(sample: Any) -> str:
    value = _field(sample, "status")
    value = getattr(value, "value", value)  # Miles Sample.Status
    return str(value or "").lower()


def _ratio(num: float | None, den: float | None) -> float | None:
    return None if num is None or not den else round(num / den, 4)


def derive_compaction_ratios(counts: Mapping[str, Any]) -> dict[str, float | None]:
    """The ratios of :data:`COMPACTION_RATIO_KEYS` from the summed counts."""
    return {
        "compacted_frac": _ratio(counts.get("compacted_trials"), counts.get("trials")),
        "compactions_mean": _ratio(counts.get("compactions"), counts.get("trials")),
        "samples_per_trial": _ratio(counts.get("samples"), counts.get("trials")),
        "segment_sample_frac": _ratio(counts.get("segment_samples"), counts.get("samples")),
        "summary_ok_frac": _ratio(counts.get("summary_ok"), counts.get("summaries")),
        "truncated_compacted_frac": _ratio(counts.get("truncated_compacted_trials"),
                                           counts.get("compacted_trials")),
        "summary_token_frac": _ratio(counts.get("summary_tokens"), counts.get("trainable_tokens")),
        "fork_lost_frac": _ratio(counts.get("fork_lost_trials"), counts.get("compaction_exposed_trials")),
    }


# "0": a trial given up before it started (a drain's abandoned slots)
SEGMENT_BUCKETS = ("0", "1", "2", "3", "4+", "unknown")


def segment_bucket(num_segments: int | None) -> str:
    """The trial-length bucket of a segment count (None: not known)."""
    n = _int_field(num_segments)
    if n is None or n < 0:
        return "unknown"
    return "4+" if n >= 4 else str(n)


def live_segments(recorder: Any, trajectory_id: str) -> int | None:
    """Segments the records proxy holds now for a trial still running (the most of
    its backend sessions); None without a records backend or an open session.
    Read on the recorder's loop thread, before the trial's session is closed."""
    backend = getattr(getattr(recorder, "proxy", None), "backend", None)
    sessions = getattr(backend, "sessions", None)
    if not isinstance(sessions, Mapping):
        return None
    counts = [len(getattr(s, "segments", None) or ()) for s in list(sessions.values())
              if getattr(s, "trajectory_id", None) == trajectory_id]
    return max(counts) if counts else None


def drained_fields(by_bucket: Mapping[str, int]) -> dict[str, Any]:
    """``rl_agentic_drain`` fields of the trials a drain cut, by segment count."""
    by = {b: int(by_bucket[b]) for b in SEGMENT_BUCKETS if by_bucket.get(b)}
    return {"drained_by_segments": by,
            "drained_compacted_trials": sum(n for b, n in by.items() if b in ("2", "3", "4+")),
            "drained_uncompacted_trials": by.get("1", 0)}


def trial_segments(result: Any) -> int | None:
    """Segment count of a settled agentic trial (finished or lost): its
    ``num_segments`` attribute, else its compaction ``parts`` (``FinishedSample``:
    one per segment, empty for one stream), else the length of a list of
    ``segments``, else 1 for one joined stream; None when unknown (a trial
    cancelled without one)."""
    n = _int_field(getattr(result, "num_segments", None))
    if n is not None:
        return n
    parts = getattr(result, "parts", None)
    if isinstance(parts, (list, tuple)) and parts:
        return len(parts)
    segments = getattr(result, "segments", None)
    if isinstance(segments, (list, tuple)):
        return len(segments) or None
    return 1 if segments is not None else None


def _reward(sample: Any) -> float | None:
    value = _field(sample, "reward")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _mean4(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


class CompactionTally:
    """Accumulates the light per-sample facts of compaction metrics (no token arrays kept).

    ``add(samples, counters)``: written / trained samples (buffer sample dicts or
    Miles ``Sample`` objects); ``counters``: optional records-proxy counters by
    trajectory id (``RecordingBackend.counters``: ``compactions``, ``context_full``).
    ``lose(result, reason)`` / ``drop(results, reason)``: trials lost before their
    group was written, and kept trials of a group that was not written, bucketed by
    segment count (``by_segments``: are long trials dropped more often?)."""

    def __init__(self) -> None:
        self._trials: dict[Any, dict[str, Any]] = {}
        self._samples = 0
        self._seen = False  # any sample carried metadata.compaction
        self._fate: dict[str, dict[str, Counter]] = {}
        self._rewards: list[float] = []  # per sample
        self._xhw: dict[Any, list[tuple[Any, float, float]]] = {}  # group -> (trial, before, after)
        self._fork_lost = 0

    def _bucket(self, bucket: str) -> dict[str, Counter]:
        return self._fate.setdefault(bucket, {"lost": Counter(), "dropped": Counter()})

    def lose(self, result: Any, reason: str, num_segments: int | None = None) -> None:
        n = trial_segments(result) if num_segments is None else num_segments
        self._seen = self._seen or (n or 0) > 1
        trajectory = getattr(result, "trajectory", None)
        loss = getattr(result, "loss", None)
        why = f"{getattr(trajectory, 'reason', '') or ''} {(loss or {}).get('error', '') if isinstance(loss, Mapping) else ''}"
        if FORK_LOST_MARK in why:  # the gateway invalidated a fork of a compacted session
            self._fork_lost += 1
            self._seen = True
            reason = FORK_LOST_MARK
        self._bucket(segment_bucket(n))["lost"][str(reason)] += 1

    def drop(self, results: Iterable[Any], reason: str) -> None:
        for result in results:
            n = trial_segments(result)
            self._seen = self._seen or (n or 0) > 1
            self._bucket(segment_bucket(n))["dropped"][str(reason)] += 1

    def add(self, samples: Iterable[Any], counters: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        for sample in samples:
            meta = _field(sample, "metadata")
            meta = meta if isinstance(meta, Mapping) else {}
            comp = meta.get(COMPACTION_KEY)
            comp = comp if isinstance(comp, Mapping) else None
            self._seen = self._seen or comp is not None
            tid = (comp or {}).get("trajectory_id") or meta.get("trajectory_id")
            key = ("t", str(tid)) if tid else ("i", _field(sample, "group_index"), _field(sample, "index"))
            reward = _reward(sample)
            trial = self._trials.setdefault(key, {
                "segments": 0, "num_segments": 1, "last": -1, "status": "", "compactions": 0,
                "summaries": 0, "summary_ok": 0, "trainable": 0, "summary_tokens": None,
                "context_full": None, "reward": reward, "replays": None})
            self._samples += 1
            if reward is not None:
                self._rewards.append(reward)
            down = meta.get("yeto_xhw_downweight")
            if isinstance(down, Mapping) and reward is not None and _reward(down) is not None:
                self._xhw.setdefault(_field(sample, "group_index"), []).append((key, _reward(down), reward))
            trial["segments"] += 1
            mask = _field(sample, "loss_mask")
            tokens = sum(1 for m in mask if m) if mask is not None else _int_field((comp or {}).get("segment_tokens"))
            trial["trainable"] += tokens or 0
            index = 0
            if comp is not None:
                index = _int_field(comp.get("segment_index")) or 0
                trial["num_segments"] = max(trial["num_segments"], _int_field(comp.get("num_segments")) or 1)
                if isinstance(comp.get("summary_ok"), bool):
                    trial["summaries"] += 1
                    trial["summary_ok"] += int(comp["summary_ok"])
                n = _int_field(comp.get("summary_tokens"))
                if n is not None:
                    trial["summary_tokens"] = (trial["summary_tokens"] or 0) + n
                n = _int_field(comp.get("compactions"))
                if n is not None:
                    trial["compactions"] = max(trial["compactions"], n)
                if isinstance(comp.get("context_full"), (bool, int)):
                    trial["context_full"] = bool(trial["context_full"]) or bool(comp["context_full"])
                n = _int_field(comp.get("retry_replays"))
                if n is not None:  # one trial's gateway count, carried by each of its segments
                    trial["replays"] = max(trial["replays"] or 0, n)
            if index >= trial["last"]:  # the last segment decides the trial's status
                trial["last"], trial["status"] = index, _status_text(sample)
            counted = (counters or {}).get(str(tid)) if tid else None
            if counted:
                trial["compactions"] = max(trial["compactions"], _int_field(counted.get("compactions")) or 0)
                if counted.get("context_full") is not None:
                    trial["context_full"] = bool(trial["context_full"]) or bool(counted.get("context_full"))
                if _int_field(counted.get("retry_replays")) is not None:
                    trial["replays"] = max(trial["replays"] or 0, counted["retry_replays"])
                self._seen = self._seen or bool(counted.get("compactions"))

    def _xhw_fields(self) -> dict[str, Any]:
        """xhw_guard downweights rewards around the group's *sample* mean: with
        segments the group's trial mean moves; ``xhw_trial_mean_shift`` is the mean
        |trial mean after - before| over the downweighted groups (0 without segments)."""
        if not self._xhw:
            return {}
        shifts = []
        for rows in self._xhw.values():
            before: dict[Any, float] = {}
            after: dict[Any, float] = {}
            for trial, b, a in rows:
                before.setdefault(trial, b)
                after.setdefault(trial, a)
            shifts.append(abs(sum(after.values()) / len(after) - sum(before.values()) / len(before)))
        return {"xhw_downweighted_groups": len(self._xhw),
                "xhw_downweighted_samples": sum(len(rows) for rows in self._xhw.values()),
                "xhw_downweighted_trials": sum(len({t for t, _, _ in rows}) for rows in self._xhw.values()),
                "xhw_trial_mean_shift": round(sum(shifts) / len(shifts), 6)}

    def _by_segments(self, written: Mapping[str, int]) -> dict[str, dict[str, Any]]:
        out = {}
        for bucket in SEGMENT_BUCKETS:
            fate = self._fate.get(bucket, {"lost": Counter(), "dropped": Counter()})
            n_written, n_lost, n_dropped = written.get(bucket, 0), sum(fate["lost"].values()), \
                sum(fate["dropped"].values())
            total = n_written + n_lost + n_dropped
            if total:
                out[bucket] = {"written": n_written, "lost": dict(sorted(fate["lost"].items())),
                               "dropped": dict(sorted(fate["dropped"].items())),
                               "not_written_frac": round((n_lost + n_dropped) / total, 4)}
        return out

    def summary(self, *, enabled: bool = False, max_compactions: int | None = None) -> dict[str, Any] | None:
        """The metrics (counts, ratios, ``segments_hist``, ``by_segments``); None
        without any trial, or when compaction is off (``enabled`` False) and no
        sample or settled trial carried compaction data."""
        if not (self._trials or self._fate) or not (self._seen or enabled):
            return None
        counts = dict.fromkeys(COMPACTION_COUNT_KEYS, 0)
        hist: dict[int, int] = {}
        written: dict[str, int] = {}
        summary_tokens = full = replays = None
        trial_rewards: list[float] = []
        trial_tokens: list[float] = []
        for trial in self._trials.values():
            compactions = max(trial["compactions"], trial["num_segments"] - 1)
            compacted = compactions > 0
            truncated = trial["status"] == "truncated"
            exhausted = max_compactions is not None and compactions >= int(max_compactions)
            hist[trial["num_segments"]] = hist.get(trial["num_segments"], 0) + 1
            bucket = segment_bucket(trial["num_segments"])
            written[bucket] = written.get(bucket, 0) + 1
            counts["trials"] += 1
            counts["compacted_trials"] += int(compacted)
            counts["compactions"] += compactions
            counts["segment_samples"] += trial["segments"] if trial["num_segments"] > 1 else 0
            counts["summaries"] += trial["summaries"]
            counts["summary_ok"] += trial["summary_ok"]
            counts["truncated_trials"] += int(truncated)
            counts["truncated_compacted_trials"] += int(truncated and compacted)
            counts["exhausted_trials"] += int(exhausted)
            counts["exhausted_truncated_trials"] += int(exhausted and truncated)
            counts["trainable_tokens"] += trial["trainable"]
            trial_tokens.append(float(trial["trainable"]))
            if trial["reward"] is not None:
                trial_rewards.append(trial["reward"])
            if trial["summary_tokens"] is not None:
                summary_tokens = (summary_tokens or 0) + trial["summary_tokens"]
            if trial["context_full"] is not None:
                full = (full or 0) + int(trial["context_full"])
            if trial["replays"] is not None:
                replays = replays or [0, 0]
                replays[0] += trial["replays"]
                replays[1] += int(trial["replays"] > 0)
        counts["samples"] = self._samples
        counts["summary_tokens"] = summary_tokens  # None: no sample reported its summary length
        counts["context_full_trials"] = full
        counts["retry_replays"], counts["retry_replay_trials"] = replays or (None, None)  # None: not reported
        counts["fork_lost_trials"] = self._fork_lost
        counts["compaction_exposed_trials"] = counts["compacted_trials"] + self._fork_lost
        if max_compactions is None:
            counts["exhausted_trials"] = counts["exhausted_truncated_trials"] = None
        maxes = [max(t["compactions"], t["num_segments"] - 1) for t in self._trials.values()]
        return {"schema": COMPACTION_STATS_SCHEMA, **counts, **derive_compaction_ratios(counts),
                "compactions_max": max(maxes) if maxes else None,
                "segments_hist": {str(k): hist[k] for k in sorted(hist)},
                "max_compactions": None if max_compactions is None else int(max_compactions),
                # per trial (the GRPO unit) and per segment sample: they differ once trials split
                "reward_mean_trial": _mean4(trial_rewards), "reward_mean_segment": _mean4(self._rewards),
                "policy_tokens_mean_trial": _mean4(trial_tokens),
                "policy_tokens_mean_segment": _ratio(counts["trainable_tokens"], self._samples),
                "by_segments": self._by_segments(written), **self._xhw_fields()}


def compaction_stats(samples: Iterable[Any], *, counters: Mapping[str, Mapping[str, Any]] | None = None,
                     enabled: bool = False, max_compactions: int | None = None) -> dict[str, Any] | None:
    """:class:`CompactionTally` over ``samples`` in one call (the trainer's step batch)."""
    tally = CompactionTally()
    tally.add(samples, counters)
    return tally.summary(enabled=enabled, max_compactions=max_compactions)


def compaction_config(recorder: Any) -> Any:
    """The records proxy's ``CompactionConfig`` behind an agentic recorder (None: off)."""
    backend = getattr(getattr(recorder, "proxy", None), "backend", None)
    return getattr(backend, "compaction", None)


def compaction_log_line(stats: Mapping[str, Any]) -> str:
    """One log line of :func:`compaction_stats` output."""
    def pct(key: str) -> str:
        value = stats.get(key)
        return "n/a" if value is None else f"{100 * value:.1f}%"

    return (f"compaction: trials={stats.get('trials')} samples={stats.get('samples')} "
            f"compacted={stats.get('compacted_trials')} ({pct('compacted_frac')}) "
            f"compactions/trial={stats.get('compactions_mean')} segments={stats.get('segments_hist')} "
            f"summary_ok={pct('summary_ok_frac')} truncated_after_compaction={stats.get('truncated_compacted_trials')} "
            f"({pct('truncated_compacted_frac')}) summary_tokens={pct('summary_token_frac')}")
