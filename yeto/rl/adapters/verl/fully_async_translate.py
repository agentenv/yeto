"""verl fully_async <-> yeto policy-age translation (agentic-rollout-utilization 6.4a).

Pure functions only: the verl adapter still declares stage 1 (``policy_age.SUPPORT``)
and a limit > 0 is refused before launch. Wiring these into a fully_async
adapter path (rollouter/trainer actors driven by the yeto ``IslandDriver``) is
task 6.4b, designed separately (design decision 9, "6.4b").

The four sub-requirements of 6.4 (6.1 code reading, verl fork acad9875):

(a) verl continues an aborted request inside ``FullyAsyncLLMServerClient.generate``
    (``verl/workers/rollout/llm_server.py:243-332``), invisible to the agent loop,
    so the adapter must record the version segments itself:
    :func:`provenance_from_calls` builds the per-token versions and generation
    log-probabilities from the per-call records (one record per resumed call:
    the server's version and the log-probs of the tokens that call added).
(b) verl only keeps ``min_global_steps`` / ``max_global_steps`` per trajectory
    (``llm_server.py:307-311, 334-338``): :func:`versions_from_global_steps`
    turns them into yeto version segments (every outer version in between,
    judged by the oldest).
(c) ``async_training.staleness_threshold`` only throttles by sample count
    (``fully_async_rollouter.py:493-497, 1155``) and never compares versions or
    drops anything, so the over-limit discard is yeto's
    (:func:`judge_trajectory`); :func:`fully_async_overrides` derives the
    throttle from the limit (:func:`staleness_threshold_for`) and
    :func:`queue_version_lag` states the lag bound the throttle alone allows
    (s = N, so up to N + 1 versions: the extra one is yeto's to discard).
(d) the trainer's local ``current_param_version``
    (``fully_async_trainer.py:143, 687, 841``; also checkpoint directory and log
    step) is not yeto's outer version: :class:`VersionMap` records the pairing at
    every publication and translates both ways; checkpoints and logs must be
    keyed by the outer version (:meth:`VersionMap.checkpoint_step`).

Assumption to verify in 6.4b: the ``global_steps`` a server reports in
``extra_fields`` is the trainer ``current_param_version`` pushed with the
weights it serves (``checkpoint_engine/base.py:510+``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.rl.engine.policy_age import PolicyAgeError, validate_limit
from yeto.rl.engine.version_segments import TokenProvenance


class VerlTranslationError(ValueError):
    """verl's async bookkeeping cannot be expressed in yeto terms."""


# ------------------------------------------------------------------ (c) throttle


def staleness_threshold_for(limit: int) -> float:
    """``async_training.staleness_threshold`` for a yeto limit N > 0.

    verl's rollouter may hold at most ``required * (1 + s) * trigger`` samples
    generated since the last parameter sync (``fully_async_rollouter.py:493-497``);
    with ``trigger_parameter_sync_step = 1`` (one verl version per yeto round) a
    queued sample can wait for at most ``floor(s) + 1`` versions
    (:func:`queue_version_lag`).

    Mapping s = N (S19 decision 10-10, main agent; was s = N - 1).  With s = N - 1
    = 0 the rollouter generated exactly one round and then idled until the next
    sync, so nothing was in flight at a push and partial rollout never resumed
    (GPU run s19-verl64b-async8-20261010a: "staleness_samples 32 >=
    max_required_samples 32" every round, 0 resumed) -- fully_async degenerated
    to sync alternation.  s = N keeps the rollouter generating while the trainer
    trains; the limit is yeto's: groups whose oldest version is more than N
    behind are discarded by version segment (:func:`judge_trajectory`,
    ``RoundCollector``), not by verl's throttle."""
    if validate_limit(limit) == 0:
        raise PolicyAgeError("limit 0 does not use verl fully_async (the sync trainer stays)")
    return float(limit)


def queue_version_lag(staleness_threshold: float, trigger_parameter_sync_step: int = 1) -> int:
    """Upper bound on how many versions a *queued* sample waits under verl's throttle."""
    if staleness_threshold < 0 or trigger_parameter_sync_step < 1:
        raise VerlTranslationError("need staleness_threshold >= 0 and trigger_parameter_sync_step >= 1")
    if trigger_parameter_sync_step != 1:
        raise VerlTranslationError(
            "trigger_parameter_sync_step != 1 makes one verl version span several yeto rounds")
    return int(math.floor(staleness_threshold)) + 1


def fully_async_overrides(limit: int, *, samples_per_round: int,
                          ppo_mini_batch_size: int) -> dict[str, Any]:
    """Hydra ``async_training`` overrides for a limit N > 0 (6.4b passes them on).

    One yeto round = one verl parameter version: ``trigger_parameter_sync_step=1``
    and ``require_batches * ppo_mini_batch_size == samples_per_round``."""
    threshold = staleness_threshold_for(limit)
    if samples_per_round <= 0 or ppo_mini_batch_size <= 0 or samples_per_round % ppo_mini_batch_size:
        raise VerlTranslationError(
            f"samples per round ({samples_per_round}) must be a positive multiple of "
            f"ppo_mini_batch_size ({ppo_mini_batch_size})")
    return {
        "async_training.staleness_threshold": threshold,
        "async_training.partial_rollout": True,
        "async_training.trigger_parameter_sync_step": 1,
        "async_training.require_batches": samples_per_round // ppo_mini_batch_size,
    }


# --------------------------------------------------------------- (d) versions


@dataclass
class VersionMap:
    """Pairs verl's trainer-local ``current_param_version`` with yeto's outer version.

    Recorded at every publication (both strictly increase); a restart (re-loaded
    checkpoint) records a new pairing for the same outer version only if it is the
    one already recorded for that param version."""

    _outer: dict[int, int] = field(default_factory=dict)
    _local: dict[int, int] = field(default_factory=dict)

    def record(self, param_version: int, outer_version: int) -> None:
        for name, value in (("param_version", param_version), ("outer_version", outer_version)):
            if type(value) is not int or value < 0:
                raise VerlTranslationError(f"{name} must be a non-negative int, got {value!r}")
        if self._outer.get(param_version, outer_version) != outer_version \
                or self._local.get(outer_version, param_version) != param_version:
            raise VerlTranslationError(
                f"param v{param_version} <-> outer v{outer_version} contradicts the recorded "
                f"pairing ({self._outer.get(param_version)} / {self._local.get(outer_version)})")
        if param_version not in self._outer and self._outer:
            last_param, last_outer = max(self._outer), self._outer[max(self._outer)]
            if param_version < last_param or outer_version <= last_outer:
                raise VerlTranslationError(
                    f"versions must increase: param v{param_version} after v{last_param}, "
                    f"outer v{outer_version} after v{last_outer}")
        self._outer[param_version] = outer_version
        self._local[outer_version] = param_version

    def outer(self, param_version: int) -> int:
        try:
            return self._outer[int(param_version)]
        except KeyError:
            raise VerlTranslationError(
                f"verl param version {param_version} was never published by yeto") from None

    def local(self, outer_version: int) -> int:
        try:
            return self._local[int(outer_version)]
        except KeyError:
            raise VerlTranslationError(f"outer version {outer_version} has no verl param version") from None

    def checkpoint_step(self, param_version: int) -> int:
        """Checkpoint directory / log step: always the outer version."""
        return self.outer(param_version)

    def to_dict(self) -> dict[str, int]:
        return {str(k): v for k, v in sorted(self._outer.items())}

    @classmethod
    def from_dict(cls, raw: Mapping[str, int]) -> "VersionMap":
        vm = cls()
        for k, v in sorted(raw.items(), key=lambda kv: int(kv[0])):
            vm.record(int(k), int(v))
        return vm


# ------------------------------------------------------------- (b) segments


def versions_from_global_steps(min_global_steps: Any, max_global_steps: Any,
                               vmap: VersionMap) -> tuple[int, ...]:
    """Outer versions a trajectory may span, from verl's per-trajectory interval.

    Every published param version in ``[min, max]`` is included (verl does not
    say where the boundaries fall); the oldest decides the age."""
    if min_global_steps is None or max_global_steps is None:
        raise VerlTranslationError("trajectory without min/max_global_steps (version unknown)")
    lo, hi = int(min_global_steps), int(max_global_steps)
    if lo > hi:
        raise VerlTranslationError(f"min_global_steps {lo} > max_global_steps {hi}")
    published = [p for p in vmap._outer if lo <= p <= hi]
    if lo not in vmap._outer or hi not in vmap._outer:
        raise VerlTranslationError(f"global steps [{lo}, {hi}] include an unpublished version")
    return tuple(sorted({vmap.outer(p) for p in published}))


# ------------------------------------------------------------ (a) provenance


@dataclass(frozen=True)
class ResumedCall:
    """One request ``FullyAsyncLLMServerClient.generate`` sent for a trajectory
    (the first or a resume after an abort): the server's param version and the
    log-probs of the tokens this call added."""

    param_version: int
    logprobs: tuple[float, ...]


def provenance_from_calls(calls: Sequence[ResumedCall], vmap: VersionMap) -> TokenProvenance:
    """Per-token outer versions and generation log-probs of one model turn."""
    provenance = TokenProvenance((), ())
    for call in calls:
        provenance = provenance.extend(vmap.outer(call.param_version),
                                       [min(float(p), 0.0) for p in call.logprobs])
    return provenance


# ---------------------------------------------------------- (c) yeto discard


def judge_trajectory(versions: Iterable[int], current_outer: int, limit: int) -> str | None:
    """None when the trajectory may train at ``current_outer`` under ``limit``; else
    the discard reason (verl's throttle never drops anything itself)."""
    validate_limit(limit)
    versions = tuple(versions)
    if not versions:
        return "unknown version"
    if max(versions) > current_outer:
        return f"token from future version v{max(versions)} > v{current_outer}"
    age = current_outer - min(versions)
    return None if age <= limit else f"policy age {age} exceeds max_policy_age={limit}"
