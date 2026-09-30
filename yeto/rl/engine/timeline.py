"""Profile/epoch-tagged execution timeline accounting (rl-infra-spec task 1.7, design D9).

Pure accounting only. Emission from ``IslandDriver`` / the Miles adapter is not
wired here (driver.py and miles_adapter are frozen until the lr-fix merge); a
driver that never records spans keeps its old behaviour.

Two rules from the acceptance text:

* overlapping spans are never billed twice: wall time and per-role busy time
  are interval unions, so a partitioned-overlap run and a serial run are
  compared on the same clock;
* tool waiting is not GPU saturation: a load sample is classified from
  queued/active/tool-wait counts, never from total rollout time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
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
