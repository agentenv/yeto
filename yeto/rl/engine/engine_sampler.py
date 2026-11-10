"""The engine sampler island: ``SamplerLoop`` over a ``SamplerEngine`` adapter
(rl-hetero-sampler-engines).

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

Replacement (the engine-sampler side of C1's :mod:`.sampler_resume`):
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

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .buffer_payload import (
    ENGINE_META,
    ROLLOUT_ROUTED_EXPERTS,
    ROLLOUT_TOP_LOGPROBS,
    ROLLOUT_TOP_LOGPROBS_IDS,
    TOP_LOGPROBS_DTYPE,
    TOP_LOGPROBS_IDS_DTYPE,
    GroupWriter,
    routed_experts_uint16,
)
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
    completion_version,
    returns_routed_experts,
    top_logprobs_num,
    updates_during_generation,
    version_fields,
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


def unique_prompt_ids(items: Sequence[PromptItem]) -> list[PromptItem]:
    """``items`` with a prompt id that repeats in the round made unique.

    A dispenser can hand out the same row twice in one round (two dispense
    positions of one row across an epoch wrap of a small dataset, or a requeued
    row next to its fresh copy). The round's names -- the buffer group key and
    the agentic recorder sessions ``<incarnation>-r<rollout>-<prompt>-<j>`` --
    are built from the prompt id, so a repeat would collide; the second copy
    gets its dispense position (else its position in the round) appended.
    """
    seen: set[str] = set()
    out = []
    for position, item in enumerate(items):
        if item.prompt_id in seen:
            spot = dispense_position(item)
            tag = f"p{spot}" if spot is not None else f"i{position}"
            prompt_id = f"{item.prompt_id}-{tag}"
            while prompt_id in seen:
                prompt_id = f"{prompt_id}-i{position}"
            item = PromptItem(prompt_id=prompt_id, prompt=item.prompt, label=item.label, metadata=item.metadata)
        seen.add(item.prompt_id)
        out.append(item)
    return out


def _accepted_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The keyword arguments of ``kwargs`` that ``fn`` accepts (all with ``**kwargs``)."""
    import inspect  # noqa: PLC0415

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def dispense_position(item: PromptItem) -> int | None:
    """The task-dispenser position of a prompt's row, None without one."""
    spot = item.metadata.get(DISPENSE_POSITION_KEY)
    return spot if isinstance(spot, int) and not isinstance(spot, bool) else None


def sample_record(item: PromptItem, completion: Completion, *, reward: float, version: int,
                  provenance: Mapping[str, str], group_index: int, index: int,
                  engine_meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One sample in the buffer payload (the ``Sample.to_dict()`` keys the
    trainer readers use; ``tokens`` is prompt + response when the engine gave
    the prompt tokens, else the response only and ``prompt_tokens_known`` False).
    The rollout arrays of the completion (R3 routed experts as ``uint16``, the
    top-K log-probs as ``int32`` / ``float32``) are written under the
    :mod:`.buffer_payload` field names when the engine returned them;
    ``engine_meta`` (``version_fields`` ...) goes to ``metadata.engine``."""
    import numpy

    response = list(completion.token_ids)
    prompt_ids = list(completion.prompt_token_ids)
    record = {
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
                        {"final_weight_version": engine_weight_version(int(completion.final_weight_version))}),
                     **({} if engine_meta is None else {ENGINE_META: dict(engine_meta)})},
    }
    if completion.routed_experts is not None:
        record[ROLLOUT_ROUTED_EXPERTS] = routed_experts_uint16(completion.routed_experts,
                                                               where=f"prompt {item.prompt_id}")
    if completion.top_logprobs is not None:
        record[ROLLOUT_TOP_LOGPROBS_IDS] = numpy.asarray(completion.top_logprobs_ids).astype(TOP_LOGPROBS_IDS_DTYPE)
        record[ROLLOUT_TOP_LOGPROBS] = numpy.asarray(completion.top_logprobs).astype(TOP_LOGPROBS_DTYPE)
    return record


def _quantile(xs: Sequence[float], q: float) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def round_summary(scored: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    """Reward / response-length / truncation summary of one round's scored groups
    for ``rl_sampler_engine_round`` (the dashboard's sampler table). ``truncated``
    counts samples whose status is ``truncated``; ``zero_variance_seen`` the groups
    whose samples all scored the same (whether or not they are dropped)."""
    rewards = [float(x.get("reward", 0.0)) for _, group in scored for x in group]
    lengths = [float(x.get("response_length", 0)) for _, group in scored for x in group]
    if not rewards:
        return {}
    mean = sum(rewards) / len(rewards)
    return {
        "reward_mean": mean,
        "reward_std": math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards)),
        "reward_p10": _quantile(rewards, 0.1), "reward_p50": _quantile(rewards, 0.5),
        "reward_p90": _quantile(rewards, 0.9),
        "resp_len_mean": sum(lengths) / len(lengths), "resp_len_p95": _quantile(lengths, 0.95),
        "truncated": sum(x.get("status") == "truncated" for _, group in scored for x in group),
        "zero_variance_seen": sum(zero_variance(group) for _, group in scored),
    }


def zero_variance(samples: Sequence[Mapping[str, Any]]) -> bool:
    """Whether every sample of a group scored the same reward (no GRPO advantage)."""
    rewards = {float(sample.get("reward", 0.0)) for sample in samples}
    return len(rewards) <= 1


def engine_meta(engine: SamplerEngine, sampling: SamplingParams) -> dict[str, Any]:
    """``metadata.engine`` of every sample of a round (:mod:`.buffer_payload`)."""
    return {"version_fields": version_fields(engine, sampling),
            "routed_experts": returns_routed_experts(engine),
            "top_logprobs_num": top_logprobs_num(engine)}


_completion_version = completion_version  # moved to sampler_engine (the records proxy uses it)


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
        self.engine_meta = engine_meta(engine, self.sampling)
        self.dropped_groups = 0  # groups not written because the island is leaving
        # Read right before every group write; True = another incarnation
        # holds the island's lease (incarnation_lease), so nothing more is written
        self.write_fenced: Callable[[], bool] = lambda: False
        # spec drop_zero_variance_groups: a group whose samples all score the same
        # reward carries no GRPO advantage; it is scored, counted and not written
        self.drop_zero_variance = False

    def _round_record(self, rollout_id: int, last: bool, position: int | None = None,
                      cursor: Mapping[str, int] | None = None) -> dict[str, Any] | None:
        """``cursor``: the data cursor taken when the round drew its prompts
        (default: read now; a streaming round reads it at the draw)."""
        if self.incarnation is None:
            return None
        if cursor is None and self.data_cursor is not None:
            cursor = self.data_cursor(rollout_id)
        return SamplerRound(self.writer.sampler_id, self.incarnation, rollout_id,
                            data_cursor=cursor, last=last, dispense_position=position).record()

    def _written(self, rollout_id: int, positions: list[int], keep: Sequence[int] = (),
                 requeue: Sequence[int] = (), outcome: Mapping[int, Mapping[str, Any]] | None = None) -> None:
        """Prompts from the task dispenser learn which groups are in the buffer.
        ``keep``: positions still generating (a streaming round), not released;
        ``requeue``: groups lost to infrastructure (given out again, counted
        toward the redo limit); ``outcome``: per-position detail for the log."""
        written = getattr(self.prompts, "written", None)
        if callable(written):
            extra: dict[str, Any] = {"keep": sorted(keep)} if keep else {}
            if requeue:
                extra["requeue"] = sorted(requeue)
            if outcome:
                extra["outcome"] = dict(outcome)
            extra = _accepted_kwargs(written, extra)  # an older written() takes neither requeue nor outcome
            result = dict(written(rollout_id, positions, unwritten="release", **extra) or {})
            result.pop("rollout_id", None)
            self.emit("rl_dispense_written", rollout_id=rollout_id, **result)

    def __call__(self, rollout_id: int) -> None:
        rollout_id = int(rollout_id)
        if self.guard is not None:
            self.guard.check(rollout_id)
        items = unique_prompt_ids(list(self.prompts(rollout_id)))
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
                    index=position * self.sampling.n + j, engine_meta=self.engine_meta))
            scored.append((item, samples))
        # dashboard: the round's reward / length / truncation summary over every
        # scored group (observation only; what is written is unchanged)
        extra: dict[str, Any] = round_summary(scored)
        if self.drop_zero_variance:
            kept = [(item, samples) for item, samples in scored if not zero_variance(samples)]
            extra["zero_variance_groups"] = len(scored) - len(kept)
            scored = kept
        self._write_scored(rollout_id, version, scored, finish=finish, tokens=tokens, record=record,
                           extra=extra)

    def _write_group(self, rollout_id: int, item: PromptItem, group: Sequence[Mapping[str, Any]], *,
                     record: Mapping[str, str], last: bool,
                     cursor: Mapping[str, int] | None = None) -> int | None:
        """Write one scored group (key ``<incarnation>-r<round>-<prompt id>``, its
        round record) and mark its round written; returns its dispenser position."""
        prefix = "" if self.incarnation is None else f"{self.incarnation}-"
        spot = dispense_position(item)
        self.writer.write(group, group_key=f"{prefix}r{int(rollout_id)}-{item.prompt_id}", provenance=record,
                          sampler_round=self._round_record(int(rollout_id), last, spot, cursor))
        if self.guard is not None:
            self.guard.written(int(rollout_id))
        return spot

    def _write_scored(self, rollout_id: int, version: int, scored: Sequence[tuple[PromptItem, list[dict]]], *,
                      finish: Mapping[str, int], tokens: int, record: Mapping[str, str],
                      samples: int | None = None, extra: Mapping[str, Any] | None = None) -> int:
        """Write one round's scored groups (after the reclaim marker none) and its
        round event; returns the groups written. ``samples``: the round's sample
        count (default: the finish counts); ``extra``: more round-event fields."""
        rollout_id = int(rollout_id)
        written = 0
        dispensed: list[int] = []
        for position, (item, group) in enumerate(scored):
            if self.reclaimed() or self.write_fenced():  # leaving: the replacement owns the shard
                self.dropped_groups += len(scored) - position
                break
            held = getattr(self.prompts, "still_holds", None)
            spot = dispense_position(item)
            if spot is not None and callable(held) and not held(spot):  # a store ledger gave it out again
                self.dropped_groups += 1
                continue
            spot = self._write_group(rollout_id, item, group, record=record, last=position == len(scored) - 1)
            written += 1
            if spot is not None:
                dispensed.append(spot)
        self._written(rollout_id, dispensed)
        self.emit(ROUND_EVENT, rollout_id=rollout_id, policy_version=version,
                  groups=written, samples=sum(finish.values()) if samples is None else int(samples),
                  response_tokens=tokens, finish=dict(sorted(finish.items())), source=self.provenance.key(),
                  dropped_groups=len(scored) - written, **dict(extra or {}))
        return written


DEFAULT_RELOAD_MAX_DEFER_S = 600.0  # spec reload_max_defer_s (EngineSamplerConfig)


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
    # the longest a ready release waits on a streaming round's running trials before it is
    # loaded anyway (PolicyFollower.max_defer_s); None: no bound
    reload_max_defer_s: float | None = DEFAULT_RELOAD_MAX_DEFER_S
    loop_guard: LoopGuard | None = None
    max_consecutive_load_failures: int = 3
    # the partial-experts selection the run trains (Kimi-K3.1): releases of another
    # selection are refused (sampler_checks.PortReleaseLoader); None: no check
    expert_selection_sha256: str | None = None
    # the trainable expert sets only (manifest.partial_experts.trainable_sha256): any EP layout of
    # the same experts is accepted; when set it replaces expert_selection_sha256's check
    expert_trainable_sha256: str | None = None
    # skip groups whose samples all score the same reward (DAPO dynamic sampling)
    drop_zero_variance_groups: bool = False


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


def started_on_release(releases: PolicyReleaseStore, base_version: int):
    """The newest release when the engine started on it (a sampler after a trainer phase,
    ``role_switch.sampler_spec_on_release``), so the follower serves it and never loads it
    again; None otherwise (the base weights, or a newer release the follower loads)."""
    latest = releases.latest() if int(base_version) > 0 else None
    return latest if latest is not None and int(latest.version) == int(base_version) else None


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
    (:class:`~.agentic_round.AgenticRound`, bound to its own settings).
    A round with ``streams`` True (trials run across the loop's rounds)
    sets the follower's ``live_update`` (its ``live``), holds loads while its
    trials run (``holds_weights``) and reads pending releases (``release_pending``)."""
    sampling = config.sampling.check()
    check_sampling_fits(sampling, engine, max_prompt_tokens=config.max_prompt_tokens)
    provenance = Provenance.of(engine.identity, sampler_id=config.sampler_id, sampling=sampling,
                               routed_experts=returns_routed_experts(engine))
    emit(START_EVENT, sampler_id=config.sampler_id, num_rollouts=int(config.num_rollouts),
         max_backlog_groups=int(config.max_backlog_groups), sampling=sampling.to_dict(),
         sampling_digest=sampling.digest(), allow_penalties=sampling.allow_penalties,
         max_prompt_tokens=int(config.max_prompt_tokens), provenance=provenance.to_record(),
         loop_guard=None if config.loop_guard is None else vars(config.loop_guard).copy())
    loader = PortReleaseLoader(engine, checks or PostLoadChecks(), emit,
                               max_consecutive_failures=config.max_consecutive_load_failures,
                               expected_selection_sha256=config.expert_selection_sha256,
                               expected_trainable_sha256=config.expert_trainable_sha256)
    try:
        loader.start(config.base_dir, int(config.base_version))
    except ReleaseNotLoaded as error:  # no served version to fall back to
        raise SamplerLoadFatal(f"the engine failed its start checks: {error}") from error
    # background download: a 207 GB Kimi-K3.1 release took longer to fetch than a
    # trainer step, and the engines idled through it
    follower = PolicyFollower(releases, loader, version=int(config.base_version),
                              live_update=updates_during_generation(engine), background_download=True,
                              max_defer_s=config.reload_max_defer_s)
    follower.release = started_on_release(releases, int(config.base_version))
    engine_round = (make_round or EngineRound)(engine, GroupWriter(buffer, config.sampler_id), prompts=prompts,
                               reward=reward, sampling=sampling, provenance=provenance, emit=emit,
                               loop_guard=config.loop_guard, incarnation=incarnation,
                               resume=resume, data_cursor=data_cursor,
                               reclaimed=should_stop if reclaimed is None else reclaimed)
    if config.drop_zero_variance_groups:
        engine_round.drop_zero_variance = True
    if getattr(engine_round, "streams", False):  # a streaming agentic round
        follower.live_update = engine_round.live
        follower.hold = engine_round.holds_weights
        engine_round.release_pending = follower.pending_seq
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
