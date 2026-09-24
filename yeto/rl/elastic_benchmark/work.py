"""Paired work: data splits, logical seeds, phase schedules and scenarios.

Splits are drawn once from a seeded permutation so every arm sees the same
train rows and the held-out rows are never handed to training. Phases bind to
logical update indexes, not wall time, so a slow arm and a fast arm observe the
same phase sequence. Scenario recipes only shape the prompt mix and external
waits; they never touch GBS, group size, optimizer steps or truncation rules.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from yeto.rl.elastic_benchmark.manifest import ManifestError

SPLIT_NAMES = ("train", "calibration", "test", "held_out")
LENGTH_BUCKETS = ("short", "long")


@dataclass(frozen=True)
class Splits:
    train: tuple[int, ...]
    calibration: tuple[int, ...]
    test: tuple[int, ...]
    held_out: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {name: list(getattr(self, name)) for name in SPLIT_NAMES}


def split_rows(row_count: int, sizes: dict[str, int], *, seed: int) -> Splits:
    """Deterministic disjoint splits by row index. Held-out rows never reach train."""
    total = sum(sizes[name] for name in SPLIT_NAMES)
    if total > row_count:
        raise ManifestError(f"splits need {total} rows but only {row_count} are available")
    order = list(range(row_count))
    random.Random(f"elastic-split:{seed}").shuffle(order)
    cursor = 0
    parts = {}
    for name in SPLIT_NAMES:
        parts[name] = tuple(order[cursor : cursor + sizes[name]])
        cursor += sizes[name]
    return Splits(**parts)


def logical_sample_seed(*, study_seed: int, group_index: int, sample_index: int) -> int:
    """Stable per-sample seed so paired arms can reconcile their sampling inputs."""
    material = f"{study_seed}:{group_index}:{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def group_seeds(*, study_seed: int, groups: int, samples_per_group: int) -> list[list[int]]:
    return [
        [logical_sample_seed(study_seed=study_seed, group_index=g, sample_index=s) for s in range(samples_per_group)]
        for g in range(groups)
    ]


@dataclass(frozen=True)
class PhaseSlot:
    update: int  # 1-based logical update index
    phase: str
    mix: dict[str, float]


def phase_schedule(phases: list[dict[str, Any]]) -> list[PhaseSlot]:
    """Expand phases into one slot per logical update. Independent of wall time."""
    slots = []
    update = 1
    for phase in phases:
        mix = _validated_mix(phase.get("mix"), phase["name"])
        for _ in range(int(phase["updates"])):
            slots.append(PhaseSlot(update, str(phase["name"]), mix))
            update += 1
    return slots


def _validated_mix(mix: Any, phase: str) -> dict[str, float]:
    if mix is None:
        return {"short": 1.0}
    if not isinstance(mix, dict) or not mix:
        raise ManifestError(f"phase {phase!r} mix must be a non-empty object")
    total = 0.0
    for bucket, share in mix.items():
        if bucket not in LENGTH_BUCKETS:
            raise ManifestError(f"phase {phase!r} uses unknown length bucket {bucket!r}")
        if not isinstance(share, (int, float)) or share < 0:
            raise ManifestError(f"phase {phase!r} bucket {bucket!r} share must be non-negative")
        total += float(share)
    if abs(total - 1.0) > 1e-6:
        raise ManifestError(f"phase {phase!r} mix shares must sum to 1")
    return {bucket: float(share) for bucket, share in mix.items()}


def bucket_counts(mix: dict[str, float], groups: int) -> dict[str, int]:
    """Largest-remainder rounding so every update draws exactly ``groups`` groups."""
    raw = {bucket: share * groups for bucket, share in mix.items()}
    counts = {bucket: int(value) for bucket, value in raw.items()}
    remainder = groups - sum(counts.values())
    for bucket in sorted(raw, key=lambda b: (raw[b] - counts[b], b), reverse=True)[:remainder]:
        counts[bucket] += 1
    return counts


def assign_groups(
    schedule: list[PhaseSlot],
    *,
    pools: dict[str, list[int]],
    groups_per_update: int,
    seed: int,
) -> list[list[int]]:
    """Prompt ids per update, drawn round-robin from each bucket's pool.

    Pools are the calibration-measured length buckets over train rows. The
    same seed and pools give the same assignment for every arm.
    """
    for bucket in LENGTH_BUCKETS:
        if any(slot.mix.get(bucket, 0) > 0 for slot in schedule) and not pools.get(bucket):
            raise ManifestError(f"schedule needs bucket {bucket!r} but its pool is empty")
    cursors = {bucket: 0 for bucket in pools}
    rng = random.Random(f"elastic-assign:{seed}")
    shuffled = {bucket: rng.sample(ids, len(ids)) for bucket, ids in pools.items()}
    assignment = []
    for slot in schedule:
        chosen = []
        for bucket, count in bucket_counts(slot.mix, groups_per_update).items():
            for _ in range(count):
                pool = shuffled[bucket]
                chosen.append(pool[cursors[bucket] % len(pool)])
                cursors[bucket] += 1
        assignment.append(chosen)
    return assignment


def scenario_phases(
    scenario: str,
    *,
    updates: int,
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    """Phase list for a scenario from calibration-measured facts only.

    ``calibration`` carries ``mixes`` (per-scenario bucket shares measured from
    real outputs), ``payback_updates`` (measured amortization window) and, for
    tool-wait, the frozen environment contract. Nothing here decides GBS or
    optimizer steps.
    """
    if updates < 1:
        raise ManifestError("scenario needs at least one update")
    builders = {
        "stable": _stable,
        "phased": _phased,
        "tail": _tail,
        "tool-wait": _tool_wait,
        "oscillating": _oscillating,
    }
    if scenario not in builders:
        raise ManifestError(f"unknown scenario {scenario!r}")
    return builders[scenario](updates, calibration)


def _mix_for(calibration: dict[str, Any], name: str) -> dict[str, float]:
    mixes = calibration.get("mixes") or {}
    mix = mixes.get(name)
    if mix is None:
        raise ManifestError(f"calibration provides no measured mix {name!r}")
    return _validated_mix(mix, name)


def _stable(updates: int, calibration: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": "stable", "updates": updates, "mix": _mix_for(calibration, "stable")}]


def _phased(updates: int, calibration: dict[str, Any]) -> list[dict[str, Any]]:
    if updates < 3:
        raise ManifestError("phased scenario needs at least three updates for A->B->A")
    a, b = _mix_for(calibration, "short-heavy"), _mix_for(calibration, "long-heavy")
    third = updates // 3
    return [
        {"name": "A", "updates": third, "mix": a},
        {"name": "B", "updates": updates - 2 * third, "mix": b},
        {"name": "A2", "updates": third, "mix": a},
    ]


def _tail(updates: int, calibration: dict[str, Any]) -> list[dict[str, Any]]:
    mix = _mix_for(calibration, "tail")
    if mix.get("long", 0) <= 0 or mix.get("long", 0) >= 0.5:
        raise ManifestError("tail scenario needs a measured minority of long prompts")
    return [{"name": "tail", "updates": updates, "mix": mix}]


def _tool_wait(updates: int, calibration: dict[str, Any]) -> list[dict[str, Any]]:
    environment = calibration.get("environment")
    required = ("task_pack", "tool_concurrency", "response_contract")
    if not isinstance(environment, dict) or any(not environment.get(k) for k in required):
        raise ManifestError(f"tool-wait scenario needs calibration.environment with {required}")
    return [
        {
            "name": "tool-wait",
            "updates": updates,
            "mix": _mix_for(calibration, "tool-wait"),
            "environment": dict(environment),
        }
    ]


def _oscillating(updates: int, calibration: dict[str, Any]) -> list[dict[str, Any]]:
    payback = calibration.get("payback_updates")
    if not isinstance(payback, int) or payback < 1:
        raise ManifestError("oscillating scenario needs a measured calibration.payback_updates")
    length = max(1, payback // 2)
    if length >= payback:
        raise ManifestError("oscillating phase length must be shorter than the payback window")
    if updates < 2 * length:
        raise ManifestError("oscillating scenario needs at least two phases")
    a, b = _mix_for(calibration, "short-heavy"), _mix_for(calibration, "long-heavy")
    phases = []
    remaining, index = updates, 0
    while remaining > 0:
        count = min(length, remaining)
        phases.append({"name": f"{'A' if index % 2 == 0 else 'B'}{index}", "updates": count, "mix": a if index % 2 == 0 else b})
        remaining -= count
        index += 1
    return phases


def length_diagnostics(samples: list[dict[str, Any]], *, max_response_len: int) -> dict[str, Any]:
    """Real length, cap-hit and zero-advantage facts from captured samples."""
    lengths = [int(s["response_length"]) for s in samples]
    capped = sum(1 for s in samples if s.get("status") == "truncated" or int(s["response_length"]) >= max_response_len)
    groups: dict[Any, list[float]] = {}
    for sample in samples:
        groups.setdefault(sample.get("group"), []).append(float(sample.get("reward", 0.0)))
    zero_advantage = sum(1 for rewards in groups.values() if len(set(rewards)) <= 1)
    return {
        "samples": len(samples),
        "mean_response_length": (sum(lengths) / len(lengths)) if lengths else None,
        "max_response_length": max(lengths) if lengths else None,
        "cap_hit_ratio": (capped / len(samples)) if samples else None,
        "groups": len(groups),
        "zero_advantage_ratio": (zero_advantage / len(groups)) if groups else None,
    }
