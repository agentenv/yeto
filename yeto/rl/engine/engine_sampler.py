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

Replacement (S25 G1C, the engine-sampler side of C1's :mod:`.sampler_resume`):
each group's sample id carries the incarnation, the round and the prompt row,
so a new incarnation never writes an earlier one's payload name; each index
entry carries a :class:`~.sampler_resume.SamplerRound` record; a round already
in the buffer is never generated or written again (:class:`~.sampler_resume.RoundGuard`);
after the reclaim marker no group is written.

:func:`build_engine_sampler` wires the island: the start checks, the engine
start on the base weights with the post-load checks (:mod:`.sampler_checks`;
a failure there stops the island), the follower and the loop. The caller
starts the resource sampler of its device family
(``telemetry.ResourceSampler`` with ``card_telemetry.collector_for``) and
calls ``island.run()``. The process entry (``python -m yeto.rl.engine.engine_sampler
--spec SPEC.json``, :func:`main`) does both from a spec file (:mod:`.engine_sampler_entry`).
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
    updates_during_generation,
)
from .sampler_loop import SamplerLoop
from .sampler_resume import RoundGuard, SamplerResume, SamplerRound
from .task_dispense import DISPENSE_POSITION_KEY

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
                     "prompt_tokens_known": bool(prompt_ids), "provenance": dict(provenance),
                     **({} if completion.final_weight_version is None else
                        {"final_weight_version": engine_weight_version(int(completion.final_weight_version))})},
    }


def _completion_version(completion: Completion, served: int, *, live: bool) -> int:
    """The behavior version of one completion: the round's served version, or the
    completion's own on an engine that updates during a generation (it must give
    one)."""
    if not live:
        if completion.weight_version not in (None, served):
            raise SamplerEngineError(f"completion of v{completion.weight_version} in a round served "
                                     f"at v{served}; the engine does not update during a generation")
        return served
    if completion.weight_version is None:
        raise SamplerEngineError("an engine that updates during a generation must give each "
                                 "completion its weight_version")
    return int(completion.weight_version)


class EngineRound:
    """``generate(rollout_id)`` of the :class:`SamplerLoop` (module doc)."""

    def __init__(self, engine: SamplerEngine, writer: GroupWriter, *,
                 prompts: Callable[[int], Sequence[PromptItem]], reward: Reward,
                 sampling: SamplingParams, provenance: Provenance, emit: Callable[..., None],
                 loop_guard: LoopGuard | None = None, incarnation: str | None = None,
                 resume: SamplerResume | None = None,
                 data_cursor: Callable[[int], Mapping[str, int]] | None = None,
                 reclaimed: Callable[[], bool] = lambda: False,
                 unchecked_in_flight: Callable[[], bool] = lambda: False) -> None:
        """``incarnation``: this process's id (``sampler_resume.new_incarnation``);
        with it, sample ids are ``<sampler>-<incarnation>-r<round>-<prompt id>`` and
        every entry carries its round record. ``resume``: where the earlier
        incarnations got to (its rounds are refused). ``data_cursor(rollout_id)``:
        the prompt source's position after the round drew its prompts.
        ``unchecked_in_flight()``: True when a load during this round's generation
        left weights that did not pass the post-load checks (an engine that
        updates during a generation); the round's groups are then not written."""
        self.engine = engine
        self.writer = writer
        self.prompts = prompts
        self.reward = reward
        self.sampling = sampling.check()
        self.provenance = provenance
        self.emit = emit
        self.loop_guard = loop_guard
        self.incarnation = None if incarnation is None else str(incarnation)
        self.guard = None if resume is None else RoundGuard(resume)
        self.data_cursor = data_cursor
        self.reclaimed = reclaimed
        self.unchecked_in_flight = unchecked_in_flight
        self.dropped_groups = 0  # groups not written because the island is leaving

    def _round_record(self, rollout_id: int, last: bool,
                      position: int | None = None) -> dict[str, Any] | None:
        if self.incarnation is None:
            return None
        cursor = None if self.data_cursor is None else self.data_cursor(rollout_id)
        return SamplerRound(self.writer.sampler_id, self.incarnation, rollout_id,
                            data_cursor=cursor, last=last, dispense_position=position).record()

    def _written(self, rollout_id: int, positions: list[int]) -> None:
        """S27 P4: prompts from the task dispenser learn which groups are in the buffer."""
        written = getattr(self.prompts, "written", None)
        if callable(written):
            result = dict(written(rollout_id, positions, unwritten="release") or {})
            result.pop("rollout_id", None)
            self.emit("rl_dispense_written", rollout_id=rollout_id, **result)

    def __call__(self, rollout_id: int) -> None:
        rollout_id = int(rollout_id)
        if self.guard is not None:
            self.guard.check(rollout_id)
        items = list(self.prompts(rollout_id))
        version = self.engine.served_version()
        if version is None:
            raise SamplerEngineError("the engine serves no version; it cannot generate")
        groups = self.engine.generate([item.prompt for item in items], self.sampling)
        if len(groups) != len(items):
            raise SamplerEngineError(f"engine returned {len(groups)} groups for {len(items)} prompts")
        live = updates_during_generation(self.engine)
        if not live and self.engine.served_version() != version:  # a reload would mix versions
            raise SamplerEngineError(f"served version changed from v{version} during round {rollout_id}")
        if live and self.unchecked_in_flight():  # some tokens may come from weights that failed a check
            self.dropped_groups += len(items)
            self.emit(ROUND_EVENT, rollout_id=rollout_id, policy_version=version, groups=0,
                      samples=0, response_tokens=0, finish={}, source=self.provenance.key(),
                      dropped_groups=len(items), dropped_reason="unchecked_weights_in_flight")
            self._written(rollout_id, [])
            return
        record = self.provenance.to_record()
        finish: Counter[str] = Counter()
        tokens = 0
        scored = []
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
                    item, completion, reward=self.reward(item, completion),
                    version=_completion_version(completion, version, live=live),
                    provenance=record, group_index=position,
                    index=position * self.sampling.n + j))
            scored.append((item, samples))
        self._write_scored(rollout_id, version, scored, finish=finish, tokens=tokens, record=record)

    def _write_scored(self, rollout_id: int, version: int, scored: Sequence[tuple[PromptItem, list[dict]]], *,
                      finish: Mapping[str, int], tokens: int, record: Mapping[str, str],
                      samples: int | None = None, extra: Mapping[str, Any] | None = None) -> int:
        """Write one round's scored groups (after the reclaim marker none) and its
        round event; returns the groups written. ``samples``: the round's sample
        count (default: the finish counts); ``extra``: more round-event fields."""
        rollout_id = int(rollout_id)
        prefix = "" if self.incarnation is None else f"{self.incarnation}-"
        written = 0
        dispensed: list[int] = []
        for position, (item, group) in enumerate(scored):
            if self.reclaimed():  # leaving: the replacement owns the shard from now on
                self.dropped_groups += len(scored) - position
                break
            spot = item.metadata.get(DISPENSE_POSITION_KEY)
            spot = spot if isinstance(spot, int) and not isinstance(spot, bool) else None
            self.writer.write(group, group_key=f"{prefix}r{rollout_id}-{item.prompt_id}",
                              provenance=record,
                              sampler_round=self._round_record(rollout_id, position == len(scored) - 1,
                                                               spot))
            written += 1
            if spot is not None:
                dispensed.append(spot)
        self._written(rollout_id, dispensed)
        if written and self.guard is not None:
            self.guard.written(rollout_id)
        self.emit(ROUND_EVENT, rollout_id=rollout_id, policy_version=version,
                  groups=written, samples=sum(finish.values()) if samples is None else int(samples),
                  response_tokens=tokens, finish=dict(sorted(finish.items())), source=self.provenance.key(),
                  dropped_groups=len(scored) - written, **dict(extra or {}))
        return written


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
    live_poll_s: float | None = None  # store poll during a round (engines that update live); None: poll_s
    loop_guard: LoopGuard | None = None
    max_consecutive_load_failures: int = 3


@dataclass
class EngineSamplerIsland:
    loop: SamplerLoop
    loader: PortReleaseLoader
    follower: PolicyFollower
    round: EngineRound

    resume: SamplerResume | None = None

    def run(self, start: int | None = None) -> int:
        """Rounds from ``start`` (default: the resume point, else 0)."""
        if start is None:
            start = 0 if self.resume is None else self.resume.next_rollout_id
        return self.loop.run(start)


def build_engine_sampler(config: EngineSamplerConfig, *, engine: SamplerEngine,
                         releases: PolicyReleaseStore, buffer: Any, emit: Callable[..., None],
                         prompts: Callable[[int], Sequence[PromptItem]], reward: Reward,
                         checks: PostLoadChecks | None = None,
                         should_stop: Callable[[], bool] = lambda: False,
                         sleep: Callable[[float], None] | None = None,
                         incarnation: str | None = None, resume: SamplerResume | None = None,
                         data_cursor: Callable[[int], Mapping[str, int]] | None = None,
                         reclaimed: Callable[[], bool] | None = None,
                         make_round: Callable[..., EngineRound] | None = None,
                         ) -> EngineSamplerIsland:
    """Check the sampling against the engine, start it on the base weights,
    run the post-load checks, and return the wired island (module doc).
    ``should_stop``: the loop starts no new round. ``reclaimed``: also stop
    writing the current round's groups (default: ``should_stop``). ``make_round``:
    builds the round instead of :class:`EngineRound` with the same arguments
    (S28 G-120: :class:`~.agentic_round.AgenticRound`, bound to its own settings)."""
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
    follower = PolicyFollower(releases, loader, version=int(config.base_version),
                              live_update=updates_during_generation(engine))
    engine_round = (make_round or EngineRound)(engine, GroupWriter(buffer, config.sampler_id), prompts=prompts,
                               reward=reward, sampling=sampling, provenance=provenance, emit=emit,
                               loop_guard=config.loop_guard, incarnation=incarnation,
                               resume=resume, data_cursor=data_cursor,
                               reclaimed=should_stop if reclaimed is None else reclaimed)
    extra = {} if sleep is None else {"sleep": sleep}
    loop: SamplerLoop
    engine_round.unchecked_in_flight = lambda: loop.unchecked_in_flight
    loop = SamplerLoop(follower=follower, buffer=buffer, generate=engine_round,
                       num_rollouts=int(config.num_rollouts),
                       max_backlog_groups=int(config.max_backlog_groups), emit=emit,
                       poll_s=config.poll_s, should_stop=should_stop,
                       min_round_s=config.min_round_s, loaded_fields=loader.loaded_fields,
                       ready=lambda: loader.ready, live_poll_s=config.live_poll_s, **extra)
    return EngineSamplerIsland(loop=loop, loader=loader, follower=follower, round=engine_round,
                               resume=resume)


def main(argv: Sequence[str] | None = None) -> int:
    """Process entry: build the island from a spec file and run it
    (:mod:`.engine_sampler_entry`)."""
    from .engine_sampler_entry import main as entry_main  # noqa: PLC0415 - keeps this module's imports lean

    return entry_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
