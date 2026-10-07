"""Reward and group filter for Terminal-Bench Codex samples on the ports path.

- ``reward_func``: only HMAC-authenticated ``tbench_outcome`` rewards (first of
  the three verification points, design D7).  An unsigned infrastructure marker
  marks the sample ABORTED and returns the numeric 0.0 Miles requires; the
  group filter never lets it into training.  A verified outcome also sets
  ``sample.metadata["success"]`` to the signed ``passed`` bit (all
  Terminal-Bench tests passed); aborted samples get no flag.
- ``check_group``: authenticates every sample again, drops groups with aborted
  members, and enforces R-D5a: samples of one trajectory (siblings) share one
  reward and are counted once (``rollout_id``/``trajectory_id``).
"""

from __future__ import annotations

from typing import Any

from yeto.rl.tbench_outcome import MAC_KEY, OUTCOME_KEY, UntrustedTBenchOutcome, verified_outcome

INFRASTRUCTURE_KEY = "tbench_infrastructure_error"


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise UntrustedTBenchOutcome("sample has no metadata")
    return metadata


def _mark_aborted(sample: Any) -> None:
    try:
        from miles.utils.types import Sample

        sample.status = Sample.Status.ABORTED
    except ImportError:  # CPU tests without the Miles package
        sample.status = "ABORTED"


def is_infrastructure(sample: Any) -> bool:
    metadata = _metadata(sample)
    return INFRASTRUCTURE_KEY in metadata and OUTCOME_KEY not in metadata and MAC_KEY not in metadata


def _sample_reward(sample: Any) -> float:
    if is_infrastructure(sample):
        _mark_aborted(sample)
        _metadata(sample).pop("success", None)  # no verdict: never a positive
        return 0.0
    metadata = _metadata(sample)
    outcome, value = verified_outcome(metadata)
    # Full success = the signed pass bit (every test passed), not reward > 0.
    metadata["success"] = outcome["passed"] is True
    return value


async def reward_func(args: Any, samples: Any, **_kwargs: Any) -> float | list[float]:
    del args
    if isinstance(samples, list):
        return [_sample_reward(sample) for sample in samples]
    return _sample_reward(samples)


def iter_samples(group: list[Any]) -> list[Any]:
    flat: list[Any] = []
    for item in group:
        flat.extend(item if isinstance(item, list) else [item])
    return flat


def sibling_key(sample: Any) -> Any:
    rollout_id = getattr(sample, "rollout_id", None)
    if rollout_id is not None:
        return ("rollout", rollout_id)
    trajectory_id = _metadata(sample).get("trajectory_id")
    if trajectory_id is not None:
        return ("trajectory", trajectory_id)
    return ("row", id(sample))


def check_siblings(samples: list[Any]) -> dict[Any, float]:
    """R-D5a: siblings share one signed reward; returns one reward per trajectory."""
    per_trajectory: dict[Any, float] = {}
    outcomes: dict[Any, dict[str, Any]] = {}
    for sample in samples:
        outcome, value = verified_outcome(_metadata(sample))
        key = sibling_key(sample)
        if key in per_trajectory:
            if per_trajectory[key] != value or outcomes[key] != outcome:
                raise UntrustedTBenchOutcome(
                    f"sibling samples of {key!r} carry different outcomes"
                )
        else:
            per_trajectory[key] = value
            outcomes[key] = outcome
    return per_trajectory


def check_group(args: Any, samples: list[Any], **_kwargs: Any):
    """Dynamic group filter: authenticate, drop aborted groups, enforce sibling sharing."""
    try:
        from miles.rollout.filter_hub.base_types import DynamicFilterOutput
    except ImportError:  # CPU tests without the Miles package
        from types import SimpleNamespace as DynamicFilterOutput  # type: ignore[assignment]
    flat = iter_samples(samples)
    if any(is_infrastructure(sample) for sample in flat):
        for sample in flat:
            if is_infrastructure(sample):
                _mark_aborted(sample)
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    rewards = check_siblings(flat)  # raises on tampering / mismatch
    distinct = set(rewards.values())
    if len(distinct) < 2:
        return DynamicFilterOutput(keep=False, reason=f"zero_std_{next(iter(distinct)) if distinct else 'empty'}")
    return DynamicFilterOutput(keep=True, reason=None)
