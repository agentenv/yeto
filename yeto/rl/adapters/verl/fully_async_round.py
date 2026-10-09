"""verl fully_async adapter path, pure part (agentic-rollout-utilization 6.4b).

The image-side runner (:mod:`.fully_async_runner`) keeps verl's
``FullyAsyncRollouter`` as the resident generator and lets the yeto
``IslandDriver`` drive the trainer actor round by round.  Everything that
decides *what* happens -- hydra overrides, the execution profile, which queued
samples a round trains and what is reported -- lives here, without verl or
Ray, so it is unit tested on the CPU:

* :func:`fully_async_run_overrides` -- the sync-path overrides turned into the
  fully_async ones; ``--rl-max-policy-age N`` drives
  ``async_training.staleness_threshold = N - 1`` and ``partial_rollout=True``
  through 6.4a's :func:`~.fully_async_translate.fully_async_overrides`.
* :func:`sample_meta` -- one queued ``RolloutSample`` (one prompt, ``n``
  responses) -> plain metadata (versions, per-call token counts, lengths).
* :class:`RoundCollector` -- takes queued samples until a round is full:
  groups older than the limit are discarded by yeto (verl's throttle never
  drops anything, 6.4a (c)); the kept groups carry their version segments; the
  round's carry-over / discard tally is reported with the Miles field names.
* :func:`execution_profile_for` -- the driver profile of a limit > 0 run
  (bounded-staleness contract, partitioned-serial: rollouter and trainer on
  separate GPUs, the driver still runs generate -> train -> publish in order).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .fully_async_translate import (
    VerlTranslationError,
    VersionMap,
    fully_async_overrides,
    judge_trajectory,
    versions_from_global_steps,
)

# Field the build-time patch (patch_verl.py, llm_server.py) adds to every
# trajectory: one [param_version, new_token_count] pair per generate call
# (the first call and each partial-rollout resume).
RESUME_CALLS_KEY = "yeto_resume_calls"
FULLY_ASYNC_MODE = "partitioned-serial"
FULLY_ASYNC_PLACEMENT = "fixed-partition"
DISCARD_MECHANISM = "yeto policy-age discard (verl fully_async queue)"

# Sync-path overrides that do not apply to the fully_async (legacy) trainer.
# ``data.train_batch_size`` is replaced: FullyAsyncRollouter asserts it is 0
# (prompts are pulled one at a time, ``gen_batch_size=1``); the batch size of a
# round is ``ppo_mini_batch_size * require_batches``.
_SYNC_ONLY_PREFIXES = ("trainer.use_v1=", "trainer.v1.trainer_mode=", "trainer.n_gpus_per_node=",
                       "data.train_batch_size=")

# Overrides the island re-reads from the resolved config (initialisation assertions).
FULLY_ASYNC_ASSERTED_KEYS = (
    "async_training.staleness_threshold",
    "async_training.partial_rollout",
    "async_training.trigger_parameter_sync_step",
    "async_training.require_batches",
    "async_training.use_trainer_do_validate",
    "async_training.use_dynamic_resource_scheduling",
    "rollout.n_gpus_per_node",
    "trainer.n_gpus_per_node",
    "data.gen_batch_size",
    "data.train_batch_size",
    "actor_rollout_ref.hybrid_engine",
)


def fully_async_startup_problems(lookup) -> list[str]:
    """Every config assertion the fork (acad9875) runs while the fully_async
    island starts, evaluated on the composed config (``lookup(dotted_key)``).

    Sources (fork ``verl/experimental/fully_async_policy``):
    ``fully_async_rollouter.py`` 350-364 (``__init__``), 733 (``_validate_config``),
    842 (``_init_async_rollout_manager``); ``fully_async_trainer.py`` 76;
    ``fully_async_main.py`` 228.  Found one by one on GPU before (dbg1:
    hybrid_engine; async3: train_batch_size), so the set is checked as a whole
    here and in the Modal CPU dry run.  Returns the failing checks."""
    def num(key):
        v = lookup(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def at_least(key, low):
        v = num(key)
        return v is not None and v >= low

    def truthy(key):
        return str(lookup(key)).strip().lower() == "true"

    checks = [
        ("actor_rollout_ref.hybrid_engine is False", not truthy("actor_rollout_ref.hybrid_engine")),
        ("data.train_batch_size == 0", num("data.train_batch_size") == 0),
        ("data.gen_batch_size == 1", num("data.gen_batch_size") == 1),
        ("async_training.staleness_threshold >= 0",
         at_least("async_training.staleness_threshold", 0)),
        ("async_training.trigger_parameter_sync_step >= 1",
         at_least("async_training.trigger_parameter_sync_step", 1)),
        ("reward model off or enable_resource_pool",
         not truthy("reward.reward_model.enable") or truthy("reward.reward_model.enable_resource_pool")),
        ("actor_rollout_ref.rollout.calculate_log_probs", truthy("actor_rollout_ref.rollout.calculate_log_probs")),
        ("actor_rollout_ref.rollout.mode == async", str(lookup("actor_rollout_ref.rollout.mode")) == "async"),
        ("async_training present", lookup("async_training") is not None),
        ("async_training.require_batches >= 1", at_least("async_training.require_batches", 1)),
        ("actor_rollout_ref.actor.ppo_mini_batch_size >= 1",
         at_least("actor_rollout_ref.actor.ppo_mini_batch_size", 1)),
    ]
    return [name for name, ok in checks if not ok]


# Assertion lines in the fork's fully_async startup path that
# :func:`fully_async_startup_problems` mirrors (whitespace-stripped prefixes).
# The dry run greps the image's sources; a line not covered here fails it.
FORK_STARTUP_ASSERTS = (
    "assert not self.hybrid_engine",
    "assert self.config.data.train_batch_size == 0",
    "assert self.config.data.gen_batch_size == 1",
    "assert self.config.async_training.staleness_threshold >= 0",
    "assert self.config.async_training.trigger_parameter_sync_step >= 1",
    "assert self.config.reward.reward_model.enable_resource_pool",
    "assert self.config.actor_rollout_ref.rollout.calculate_log_probs",
    "assert self.config.actor_rollout_ref.rollout.mode == \"async\"",
)


def total_rollout_steps(limit: int, *, groups_per_round: int, rounds: int) -> int:
    """Prompts the rollouter may feed: two rounds' worth per round (room for yeto's
    over-age discards) plus the ``limit + 1`` rounds that can be queued ahead."""
    return groups_per_round * (2 * rounds + limit + 1)


def fully_async_run_overrides(sync_overrides: Sequence[str], limit: int, *, groups_per_round: int,
                              rounds: int, rollout_gpus: int = 1, trainer_gpus: int = 1) -> list[str]:
    """Hydra overrides for ``fully_async_ppo_trainer.yaml`` (fork acad9875).

    One yeto round = one verl parameter version (prompt-level samples:
    ``ppo_mini_batch_size = groups_per_round``, ``require_batches = 1``)."""
    if rollout_gpus < 1 or trainer_gpus < 1:
        raise VerlTranslationError("fully_async needs at least one rollout GPU and one trainer GPU")
    async_keys = fully_async_overrides(limit, samples_per_round=groups_per_round,
                                       ppo_mini_batch_size=groups_per_round)
    out = [o for o in sync_overrides if not o.startswith(_SYNC_ONLY_PREFIXES)]
    out += [f"{k}={v}" for k, v in async_keys.items()]
    out += [
        "async_training.use_trainer_do_validate=False",
        "async_training.use_dynamic_resource_scheduling=False",
        "data.gen_batch_size=1",
        "rollout.nnodes=1",
        f"rollout.n_gpus_per_node={rollout_gpus}",
        f"rollout.total_rollout_steps={total_rollout_steps(limit, groups_per_round=groups_per_round, rounds=rounds)}",
        f"trainer.n_gpus_per_node={trainer_gpus}",
        "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
        # FullyAsyncTrainer/FullyAsyncRollouter assert not hybrid_engine
        # (fork acad9875, fully_async_trainer.py:76); the sync path never sets
        # this key, so without this override the config default True kills the
        # island at startup (s19-verl64b-dbg1-20261009a, exit 1 after 86 s).
        "actor_rollout_ref.hybrid_engine=False",
        # FullyAsyncRollouter asserts train_batch_size == 0 (fork
        # fully_async_rollouter.py:351; s19-verl64b-async3-20261009a died here).
        "data.train_batch_size=0",
    ]
    return out


# ------------------------------------------------------------------ samples


@dataclass(frozen=True)
class SampleMeta:
    """One queued prompt (``n`` responses) as plain data."""

    uid: str
    min_steps: tuple[Any, ...]
    max_steps: tuple[Any, ...]
    calls: tuple[tuple[tuple[int, int], ...] | None, ...]  # per response; None = patch absent
    lengths: tuple[int, ...]
    rewards: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.lengths)


def _list(value: Any, n: int) -> list:
    if value is None:
        return [None] * n
    items = list(value)
    if len(items) != n:
        raise VerlTranslationError(f"per-response field has {len(items)} entries for {n} responses")
    return items


def sample_meta(non_tensor: Mapping[str, Any], lengths: Sequence[int],
                rewards: Sequence[float] | None = None) -> SampleMeta:
    """Metadata of one ``RolloutSample.full_batch`` (``non_tensor_batch`` + the
    response-mask lengths and reward sums the runner reads from its tensors)."""
    n = len(lengths)
    uids = {str(u) for u in _list(non_tensor.get("uid"), n)}
    if len(uids) != 1:
        raise VerlTranslationError(f"one queued sample must carry one uid, got {sorted(uids)}")
    calls = []
    for raw in _list(non_tensor.get(RESUME_CALLS_KEY), n):
        calls.append(None if raw is None else tuple((int(v), int(c)) for v, c in raw))
    rew = [float("nan")] * n if rewards is None else [float(r) for r in rewards]
    return SampleMeta(uid=uids.pop(), min_steps=tuple(_list(non_tensor.get("min_global_steps"), n)),
                      max_steps=tuple(_list(non_tensor.get("max_global_steps"), n)),
                      calls=tuple(calls), lengths=tuple(int(x) for x in lengths), rewards=tuple(rew))


@dataclass(frozen=True)
class GroupVerdict:
    meta: SampleMeta
    versions: tuple[int, ...]          # outer versions of all tokens of the group
    reason: str | None                 # None = kept
    unknown: bool                      # version could not be determined
    older_tokens: int                  # tokens generated by a version < current
    resumed: int                       # responses continued after an abort (> 1 call)
    segment_mismatches: int            # responses whose call token counts != length


def judge_group(meta: SampleMeta, vmap: VersionMap, current_outer: int, limit: int) -> GroupVerdict:
    """A GRPO group is kept or discarded whole; its oldest token decides (6.4a (b)/(c))."""
    versions: set[int] = set()
    older = resumed = mismatched = 0
    try:
        for i in range(meta.n):
            span = versions_from_global_steps(meta.min_steps[i], meta.max_steps[i], vmap)
            versions.update(span)
            calls = meta.calls[i]
            if calls is None:  # no per-call record: every token counted at the oldest version
                if min(span) < current_outer:
                    older += meta.lengths[i]
                continue
            resumed += len(calls) > 1
            if sum(c for _, c in calls) != meta.lengths[i]:
                mismatched += 1
            for version, count in calls:
                outer = vmap.outer(version)
                if outer not in span:
                    raise VerlTranslationError(
                        f"call version v{outer} outside the trajectory's span {span}")
                if outer < current_outer:
                    older += count
    except VerlTranslationError as exc:
        return GroupVerdict(meta, (), f"unknown version: {exc}", True, 0, 0, 0)
    ordered = tuple(sorted(versions))
    reason = judge_trajectory(ordered, current_outer, limit)
    return GroupVerdict(meta, ordered, reason, False, older, resumed, mismatched)


@dataclass
class RoundCollector:
    """Queued samples -> one round of ``required`` kept groups + its report."""

    vmap: VersionMap
    current_outer: int
    limit: int
    required: int
    kept: list[GroupVerdict] = field(default_factory=list)
    discarded: list[GroupVerdict] = field(default_factory=list)

    @property
    def full(self) -> bool:
        return len(self.kept) >= self.required

    def offer(self, meta: SampleMeta) -> bool:
        """True when the sample is trained this round."""
        if self.full:
            raise VerlTranslationError("round already full")
        verdict = judge_group(meta, self.vmap, self.current_outer, self.limit)
        (self.discarded if verdict.reason else self.kept).append(verdict)
        return verdict.reason is None

    def discard_tally(self) -> dict[str, int]:
        """``{groups, samples, response_tokens, unknown_groups}`` (rollout_events.cutoff_fields)."""
        return {"groups": len(self.discarded),
                "samples": sum(v.meta.n for v in self.discarded),
                "response_tokens": sum(sum(v.meta.lengths) for v in self.discarded),
                "unknown_groups": sum(1 for v in self.discarded if v.unknown)}

    def carry_over_fields(self) -> dict[str, Any]:
        """The round's ``rl_rollout_carry_over`` fields (Miles names, carry_over.CARRY_FIELDS)."""
        carried = [v for v in self.kept if v.versions and min(v.versions) < self.current_outer]
        over_age = [v for v in self.discarded if not v.unknown]
        return {
            "max_policy_age": self.limit,
            "carried_in_groups": len(carried),
            "carried_in_trajectories": sum(v.meta.n for v in carried),
            "carried_in_tokens": sum(v.older_tokens for v in carried),
            "over_age_discarded_groups": len(over_age),
            "over_age_discarded_trajectories": sum(v.meta.n for v in over_age),
            "over_age_discarded_tokens": sum(sum(v.meta.lengths) for v in over_age),
            "unknown_version_discarded_groups": sum(1 for v in self.discarded if v.unknown),
            "cross_version_tokens": sum(v.older_tokens for v in self.kept),
            "trained_response_tokens": sum(sum(v.meta.lengths) for v in self.kept),
            "resumed_trajectories": sum(v.resumed for v in self.kept),
            "segment_token_mismatches": sum(v.segment_mismatches for v in self.kept),
            "discard_reasons": sorted({v.reason for v in self.discarded if v.reason})[:5],
        }


def group_metadata(collector: RoundCollector, rollout_id: int, published: Mapping[int, str]):
    """Kept groups as :class:`GroupMetadata`; the policy token names the oldest version
    (the driver checks it against the hashes it published within the window)."""
    from yeto.rl.engine.ports import GroupMetadata

    out = []
    for verdict in collector.kept:
        meta, oldest = verdict.meta, min(verdict.versions)
        rewards = [r for r in meta.rewards if not math.isnan(r)]
        mean = sum(rewards) / len(rewards) if rewards else float("nan")
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards)) if rewards else float("nan")
        out.append(GroupMetadata(
            group_id=f"r{rollout_id}-{meta.uid}",
            sample_ids=tuple(f"r{rollout_id}-{meta.uid}-{k}" for k in range(meta.n)),
            policy_token=f"yeto:{oldest}:{published[oldest]}",
            reward_mean=mean, reward_std=std, token_count=sum(meta.lengths),
            policy_versions=None if verdict.versions == (rollout_id,) else verdict.versions))
    return tuple(out)


# ------------------------------------------------------------------ profile


def execution_profile_for(spec: Any, limit: int, *, groups_per_round: int, samples_per_group: int,
                          sync: str):
    """Driver profile of a limit > 0 verl fully_async island."""
    from yeto.rl.engine.execution_profile import ExecutionProfile
    from yeto.rl.engine.policy_age import BOUNDED_STALENESS_CONTRACT

    protocol = {"none": "none", "strict": "strict-avg", "elastic": "strict-avg"}[sync]
    return ExecutionProfile(
        name="verl-fully-async", execution_mode=FULLY_ASYNC_MODE, outer_protocol=protocol,
        algorithm_contract=BOUNDED_STALENESS_CONTRACT, max_policy_age=limit,
        groups_per_batch=groups_per_round, samples_per_group=samples_per_group,
        algorithm_spec_sha256=spec.sha256(), extra={"engine": "verl-fully-async"})
