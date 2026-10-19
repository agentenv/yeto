"""The engine sampler island: ``SamplerLoop`` over a ``SamplerEngine`` adapter
(rl-hetero-sampler-engines D1, S25 N2.3).

The Miles sampler island (``adapters.miles.sampler_island``) lets Miles drive
SGLang. This island drives any engine behind the port instead, so the same
loop, buffer writer, reward call and release follower serve SGLang on PPU and
vllm-ascend on 910B. Per round (:class:`EngineRound`):

1. take the round's prompts (``prompts(rollout_id)``);
2. ``engine.generate`` with the run's :class:`SamplingParams` (one value for
   every source of the run, checked at start);
3. cut looping completions (:class:`~.loop_guard.LoopGuard`, off by default);
4. score each completion (``reward(item, completion)``);
5. write one group per prompt to the buffer (:class:`~.buffer_payload.GroupWriter`)
   with the behavior version the engine serves and the source record
   (:class:`~.sampler_engine.Provenance`) in the index entry;
6. write ``rl_sampler_engine_round`` (finish reasons, tokens, provenance key).

:func:`build_engine_sampler` wires the island: the start checks, the engine
start on the base weights with the post-load checks (:mod:`.sampler_checks`;
a failure there stops the island), the follower and the loop. The caller
starts the resource sampler of its device family
(``telemetry.ResourceSampler`` with ``card_telemetry.collector_for``) and
calls ``island.run()``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .buffer_payload import GroupWriter
from .loop_guard import LoopGuard
from .policy_release import PolicyFollower, PolicyReleaseStore, ReleaseNotLoaded
from .policy_versions import engine_weight_version
from .sampler_checks import PortReleaseLoader, PostLoadChecks, SamplerLoadFatal, check_sampling_fits
from .sampler_engine import (
    FINISH_STOP,
    Completion,
    Provenance,
    SamplerEngine,
    SamplerEngineError,
    SamplingParams,
)
from .sampler_loop import SamplerLoop

ROUND_EVENT = "rl_sampler_engine_round"
START_EVENT = "rl_sampler_start"


@dataclass(frozen=True)
class PromptItem:
    """One prompt of a round; ``label`` and ``metadata`` go to the reward and payload."""

    prompt_id: str
    prompt: str
    label: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


Reward = Callable[[PromptItem, Completion], float]


def sample_record(item: PromptItem, completion: Completion, *, reward: float, version: int,
                  provenance: Mapping[str, str], group_index: int, index: int) -> dict[str, Any]:
    """One sample in the buffer payload (the ``Sample.to_dict()`` keys the
    trainer readers use; ``tokens`` is prompt + response when the engine gave
    the prompt tokens, else the response only and ``prompt_tokens_known`` False)."""
    response = list(completion.token_ids)
    prompt_ids = list(completion.prompt_token_ids)
    return {
        "group_index": group_index, "index": index, "prompt": item.prompt,
        "response": completion.text, "label": item.label,
        "tokens": prompt_ids + response, "response_length": len(response),
        "rollout_log_probs": list(completion.logprobs), "reward": float(reward),
        "status": "completed" if completion.finish_reason == FINISH_STOP else "truncated",
        "metadata": {**dict(item.metadata), "prompt_id": item.prompt_id,
                     "weight_version": engine_weight_version(int(version)),
                     "finish_reason": completion.finish_reason,
                     "prompt_tokens_known": bool(prompt_ids), "provenance": dict(provenance)},
    }


class EngineRound:
    """``generate(rollout_id)`` of the :class:`SamplerLoop` (module doc)."""

    def __init__(self, engine: SamplerEngine, writer: GroupWriter, *,
                 prompts: Callable[[int], Sequence[PromptItem]], reward: Reward,
                 sampling: SamplingParams, provenance: Provenance, emit: Callable[..., None],
                 loop_guard: LoopGuard | None = None) -> None:
        self.engine = engine
        self.writer = writer
        self.prompts = prompts
        self.reward = reward
        self.sampling = sampling.check()
        self.provenance = provenance
        self.emit = emit
        self.loop_guard = loop_guard

    def __call__(self, rollout_id: int) -> None:
        items = list(self.prompts(int(rollout_id)))
        version = self.engine.served_version()
        if version is None:
            raise SamplerEngineError("the engine serves no version; it cannot generate")
        groups = self.engine.generate([item.prompt for item in items], self.sampling)
        if len(groups) != len(items):
            raise SamplerEngineError(f"engine returned {len(groups)} groups for {len(items)} prompts")
        if self.engine.served_version() != version:  # a reload during the round would mix versions
            raise SamplerEngineError(f"served version changed from v{version} during round {rollout_id}")
        record = self.provenance.to_record()
        finish: Counter[str] = Counter()
        tokens = 0
        for position, (item, group) in enumerate(zip(items, groups)):
            if len(group) != self.sampling.n:
                raise SamplerEngineError(f"prompt {item.prompt_id}: {len(group)} completions, "
                                         f"expected n={self.sampling.n}")
            samples = []
            for j, completion in enumerate(group):
                if self.loop_guard is not None:
                    completion = self.loop_guard.apply(completion)
                finish[completion.finish_reason] += 1
                tokens += len(completion.token_ids)
                samples.append(sample_record(
                    item, completion, reward=self.reward(item, completion), version=version,
                    provenance=record, group_index=position,
                    index=position * self.sampling.n + j))
            self.writer.write(samples, group_key=f"r{int(rollout_id)}-{item.prompt_id}",
                              provenance=record)
        self.emit(ROUND_EVENT, rollout_id=int(rollout_id), policy_version=version,
                  groups=len(items), samples=sum(finish.values()), response_tokens=tokens,
                  finish=dict(sorted(finish.items())), source=self.provenance.key())


@dataclass
class EngineSamplerConfig:
    """What the run config gives an engine sampler island."""

    sampler_id: str
    num_rollouts: int
    max_backlog_groups: int
    sampling: SamplingParams
    max_prompt_tokens: int
    base_version: int = 0
    base_dir: Path | None = None  # None: the engine's configured base weights
    min_round_s: float | None = None
    poll_s: float = 5.0
    loop_guard: LoopGuard | None = None
    max_consecutive_load_failures: int = 3


@dataclass
class EngineSamplerIsland:
    loop: SamplerLoop
    loader: PortReleaseLoader
    follower: PolicyFollower
    round: EngineRound

    def run(self, start: int = 0) -> int:
        return self.loop.run(start)


def build_engine_sampler(config: EngineSamplerConfig, *, engine: SamplerEngine,
                         releases: PolicyReleaseStore, buffer: Any, emit: Callable[..., None],
                         prompts: Callable[[int], Sequence[PromptItem]], reward: Reward,
                         checks: PostLoadChecks | None = None,
                         should_stop: Callable[[], bool] = lambda: False,
                         sleep: Callable[[float], None] | None = None) -> EngineSamplerIsland:
    """Check the sampling against the engine, start it on the base weights,
    run the post-load checks, and return the wired island (module doc)."""
    sampling = config.sampling.check()
    check_sampling_fits(sampling, engine, max_prompt_tokens=config.max_prompt_tokens)
    provenance = Provenance.of(engine.identity, sampler_id=config.sampler_id, sampling=sampling)
    emit(START_EVENT, sampler_id=config.sampler_id, num_rollouts=int(config.num_rollouts),
         max_backlog_groups=int(config.max_backlog_groups), sampling=sampling.to_dict(),
         sampling_digest=sampling.digest(), allow_penalties=sampling.allow_penalties,
         max_prompt_tokens=int(config.max_prompt_tokens), provenance=provenance.to_record(),
         loop_guard=None if config.loop_guard is None else vars(config.loop_guard).copy())
    loader = PortReleaseLoader(engine, checks or PostLoadChecks(), emit,
                               max_consecutive_failures=config.max_consecutive_load_failures)
    try:
        loader.start(config.base_dir, int(config.base_version))
    except ReleaseNotLoaded as error:  # no served version to fall back to
        raise SamplerLoadFatal(f"the engine failed its start checks: {error}") from error
    follower = PolicyFollower(releases, loader, version=int(config.base_version))
    engine_round = EngineRound(engine, GroupWriter(buffer, config.sampler_id), prompts=prompts,
                               reward=reward, sampling=sampling, provenance=provenance, emit=emit,
                               loop_guard=config.loop_guard)
    extra = {} if sleep is None else {"sleep": sleep}
    loop = SamplerLoop(follower=follower, buffer=buffer, generate=engine_round,
                       num_rollouts=int(config.num_rollouts),
                       max_backlog_groups=int(config.max_backlog_groups), emit=emit,
                       poll_s=config.poll_s, should_stop=should_stop,
                       min_round_s=config.min_round_s, loaded_fields=loader.loaded_fields,
                       ready=lambda: loader.ready, **extra)
    return EngineSamplerIsland(loop=loop, loader=loader, follower=follower, round=engine_round)
