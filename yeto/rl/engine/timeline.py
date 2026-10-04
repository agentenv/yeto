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
