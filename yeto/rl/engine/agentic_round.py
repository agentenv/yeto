"""Agentic rounds of the engine sampler island (S28 G-120).

The engine sampler (:mod:`.engine_sampler`) writes one buffer group per prompt.
An agentic round writes one group per *task*: ``sampling.n`` agent trials of
the task (Harbor, e.g. Codex in a boat sandbox), each calling the island's own
engine through a token recorder (the records proxy,
:mod:`yeto.rl.harness.gateway.recorder`), so the group's samples carry the
exact ids the engine saw and sampled and their log-probs.

Nothing here is new agent logic; per trial the round runs the neutral layer:

1. ``recorder.open`` -> the OpenAI base URL the agent calls;
2. :func:`yeto.rl.agentic.run_agent_task` (admission, harness, wall-clock cap,
   verdict -> outcome; the reward is the Harbor verifier's);
3. :func:`yeto.rl.agentic.finish.finish_sample` (close, join the calls into one
   token stream, ABORTED on misalignment).

Lost trajectories: an ABORTED trial (infrastructure failure, Harbor timeout,
misaligned tokens) and, with ``weight_switch`` ``drain``, a trial whose calls
were served by another weight version than its group's are left out of their
group (never trained, never in a group baseline, as Miles' ``check_no_aborted``
and verl's ``isolate_rows``). A group with fewer than ``min_samples`` kept
trials is not written. The round event carries the counts (``agentic``:
trials, kept, lost by reason, loss rate, reward and response-length means).

Streaming: the round is no barrier. The island keeps up to
``max_in_flight`` trials running; a call ``round(rollout_id)`` registers the
round's tasks (each task's ``n`` trials together, one group) and returns as
soon as every registered trial has started and a slot is free, so the sampler
loop draws the next round while the earlier ones still run. A group is
settled (filtered, scored, written) as soon as its ``n`` trials ended, by the
calling thread inside a later call; a slow trial holds back its own group
only. The sampler loop calls :meth:`AgenticRound.flush` after its last round.
The rollout id stays the numbering and resume unit:

* group key ``<incarnation>-r<round>-<prompt id>``, and the
  ``sampler_round`` record of the group's own round, whose data cursor is the
  one taken when that round drew its prompts (``resume_point`` refuses two
  cursors per round); ``last`` is on the group whose write settled the round's
  final group (a round whose final group is not written reads as partial);
* :class:`~.sampler_resume.RoundGuard` refuses a round already in the buffer
  at the draw; a new incarnation resumes after the newest round with an entry,
  so a round of the old incarnation still running then is never written by
  anyone (its dispenser rows are released when the old one closes or is fenced;
  its rows of a JSONL slot are skipped);
* after the reclaim marker no new group starts and no group is written;
* dispenser rows (``written``): each settled group reports its own row (trained
  when written, released when not) and ``keep`` names the rows still in the
  stream, so they are neither released nor dropped.

Weight switches (``weight_switch``; the default ``auto`` is ``live`` on an
engine with ``updates_during_generation``, else ``drain``):

* ``drain``: a pending release (read every ``release_poll_s``) stops
  new groups; the started groups finish (``drain_timeout_s``: then their trials
  are cancelled and lost as ``drain_timeout``), the call returns and the loop
  loads the release before the next call. While a group runs the follower loads
  nothing (``holds_weights``), so every group is of one version.
* ``live`` (an engine with ``updates_during_generation`` only): the loop loads
  each release into the running engine; no trial is lost for its version. A
  sample's version is the oldest one its turns were served by (the trainer's
  ``max_policy_age`` gate and TIS on ``rollout_log_probs`` handle the lag);
  ``metadata.final_weight_version`` is the newest (the key the live engine
  round uses); the records proxy takes each turn's version from the engine's
  completion (first token) instead of refusing a turn a reload fell inside.

Events: ``rl_agentic_group`` per settled group (group seconds from its first
trial's start, each trial's seconds, behavior version and lag, in-flight
utilization over its life, drain wait during it); ``rl_agentic_drain`` per
drain; the round event per call (the groups it settled) with the ``agentic``
fields above plus ``in_flight``, ``in_flight_utilization`` and ``drain_wait_s``.

Buffer sample (``engine_sampler.sample_record`` keys, so the trainer readers
are unchanged): ``tokens`` = prompt ids + response ids of the joined stream,
``response_length``, ``rollout_log_probs`` (0 at mask-0 positions),
``loss_mask`` (the response mask: 1 = sampled by the policy, 0 = tool output
and template tokens), ``reward``; ``metadata`` adds the trial outcome fields.

The trials run on one event loop thread kept for the island's life (the
recorder's HTTP server lives on that loop); the engine calls run in its
executor, sized to the in-flight limit. Groups are written on the calling
(sampler loop) thread only.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import re
import statistics
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .engine_sampler import ROUND_EVENT, EngineRound, PromptItem, dispense_position
from .policy_versions import engine_weight_version
from .sampler_engine import FINISH_LENGTH, FINISH_STOP, SamplerEngineError, updates_during_generation

AGENT_TASK_KEY = "agent_task"
LOST_VERSION = "served_by_other_version"
def resolve_weight_switch(switch: str, engine: Any) -> str:
    """``auto`` -> ``live`` on an engine that updates during a generation, else ``drain``."""
    if switch != "auto":
        return switch
    return "live" if updates_during_generation(engine) else "drain"


LOST_DRAIN = "drain_timeout"  # cancelled when a drain reached its timeout
WEIGHT_SWITCHES = ("auto", "drain", "live")
GROUP_EVENT = "rl_agentic_group"
RETRY_EVENT = "rl_agentic_trial_retry"
DEFAULT_INFRA_RETRIES = 2  # per trial slot (spec agentic.infra_retries)
# ABORTED reasons that are never an infrastructure failure before the policy ran: the outer
# wall-clock cap (a timeout; retrying it would favour short solutions), a closing gate, and the
# token checks of finish_sample
_NOT_INFRA = frozenset({"wall_clock_cap", "admission_draining", "token_alignment",
                        "prompt_exceeds_prompt_length", "policy_output_error"})
DRAIN_EVENT = "rl_agentic_drain"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class LoopThread:
    """One asyncio loop on a daemon thread; :meth:`run` blocks for a coroutine's result."""

    def __init__(self, *, workers: int) -> None:
        self.loop = asyncio.new_event_loop()
        self.loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=max(4, int(workers))))
        self._thread = threading.Thread(target=self.loop.run_forever, name="agentic-round", daemon=True)
        self._thread.start()

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)


def session_id(*parts: Any) -> str:
    """A recorder session id (a URL path segment) from the round's names."""
    return _UNSAFE.sub("_", "-".join(str(p) for p in parts))[:180]


def agentic_sample(item: PromptItem, finished: Any, *, version: int, provenance: Mapping[str, str],
                   group_index: int, index: int, versions: tuple[int, int] | None = None) -> dict[str, Any]:
    """One kept trial as a buffer sample (module doc). ``versions``: the oldest and
    newest weight version of its turns (``live``): ``weight_version`` is the
    oldest and ``final_weight_version`` (as :func:`~.engine_sampler.sample_record`'s)
    the newest."""
    trajectory, seg = finished.trajectory, finished.segments
    response = list(seg.response_ids)
    meta = trajectory.to_metadata()
    last = finished.turns[-1].finish_reason if finished.turns else None
    # A last turn the engine cut at its length limit ends the trajectory as
    # surely as a segment over max_response_len: both are truncated.
    truncated = seg.truncated or last == FINISH_LENGTH
    return {
        "group_index": group_index, "index": index, "prompt": item.prompt,
        "response": "", "label": item.label,
        "tokens": list(seg.prompt_ids) + response, "response_length": len(response),
        "rollout_log_probs": [float(x) for x in seg.response_logprobs],
        "loss_mask": [int(m) for m in seg.response_mask],
        "reward": float(trajectory.reward),
        "status": "truncated" if truncated else "completed",
        "metadata": {**{k: v for k, v in item.metadata.items() if k != AGENT_TASK_KEY},
                     "prompt_id": item.prompt_id, "task_id": trajectory.task.task_id,
                     "trajectory_id": trajectory.task.trajectory_id,
                     "weight_version": engine_weight_version(int(version)),
                     "finish_reason": FINISH_LENGTH if truncated else (last or FINISH_STOP),
                     "prompt_tokens_known": True, "provenance": dict(provenance),
                     "agent_turns": len(finished.turns),
                     "agent": {k: meta[k] for k in ("yeto_agent_outcome", "outcome_reason", "eval_report",
                                                    "agent_metrics", "trial_dir")},
                     **({} if versions is None else
                        {"final_weight_version": engine_weight_version(int(versions[1]))})},
    }


def infra_failure(result: Any) -> str | None:
    """The failure name when ``result`` (a finished trial) is an
    infrastructure failure *before the policy produced anything* -- ABORTED with no
    model call recorded (sandbox start, image pull, agent install, a harness error)
    -- else None. Such a trial is unbiased to run again; a trial that called the
    model, a timeout and a policy-limit outcome are never retried."""
    from yeto.rl.agentic import Outcome
    from yeto.rl.agentic.outcome import POLICY_LIMIT_EXCEPTIONS
    from yeto.rl.harness.harbor.trial import NOT_RETRIED_EXCEPTIONS

    trajectory = getattr(result, "trajectory", None)
    if trajectory is None or trajectory.outcome is not Outcome.ABORTED or getattr(result, "turns", ()):
        return None
    name = str(trajectory.reason or "").split(":", 1)[0].strip() or "aborted"
    verdict = trajectory.verdict
    exc = None if verdict is None else verdict.exception_type
    if name in _NOT_INFRA or name in NOT_RETRIED_EXCEPTIONS or name in POLICY_LIMIT_EXCEPTIONS \
            or exc in NOT_RETRIED_EXCEPTIONS or exc in POLICY_LIMIT_EXCEPTIONS:
        return None
    return name


class _Lost:
    """A trial that ended without a result (cancelled at a drain's timeout or at close)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


@dataclasses.dataclass(eq=False)
class _Group:
    """One task's ``n`` trials in the stream (registered together, written together)."""

    rollout_id: int
    position: int  # in its round
    item: PromptItem
    sessions: list[str]
    started: int = 0  # trials started
    done: int = 0  # trials ended
    results: list[Any] = dataclasses.field(default_factory=list)
    trial_s: list[float | None] = dataclasses.field(default_factory=list)
    start_versions: list[int | None] = dataclasses.field(default_factory=list)
    version: int | None = None  # the served version when its first trial started
    started_at: float | None = None
    busy_at_start: float = 0.0
    drain_at_start: float = 0.0
    utilization: float | None = None  # mean trials in flight / max_in_flight over its life
    drain_s: float = 0.0  # drain waits during its life
    retry_slots: deque = dataclasses.field(default_factory=deque)  # slots to run again
    retries: list[int] = dataclasses.field(default_factory=list)  # infra retries per slot
    retry_reasons: list[str] = dataclasses.field(default_factory=list)
    abandoned: bool = False  # a drain timed out on it: no slot of it runs again

    def __post_init__(self) -> None:
        n = len(self.sessions)
        self.results, self.trial_s, self.start_versions = [None] * n, [None] * n, [None] * n
        self.retries = [0] * n
        self.base_sessions = list(self.sessions)

    @property
    def pending(self) -> bool:
        """Trials still to start: unstarted slots or slots to run again."""
        return self.started < len(self.sessions) or bool(self.retry_slots)

    @property
    def spot(self) -> int | None:
        return dispense_position(self.item)


@dataclasses.dataclass
class _RoundState:
    """A round (rollout id) with groups not settled yet: its data cursor (taken when
    it drew its prompts) and how many of its groups are still open."""

    cursor: Mapping[str, int] | None
    open: int


@dataclasses.dataclass
class _Tally:
    """What one call (or the flush) settled: the round event's fields."""

    trials: int = 0
    written: int = 0
    dropped: int = 0
    small: int = 0
    tokens: int = 0
    samples: int = 0
    lost: Counter[str] = dataclasses.field(default_factory=Counter)
    finish: Counter[str] = dataclasses.field(default_factory=Counter)
    rewards: list[float] = dataclasses.field(default_factory=list)
    lengths: list[int] = dataclasses.field(default_factory=list)
    retried: Counter[str] = dataclasses.field(default_factory=Counter)  # infra retries
    refilled: int = 0  # groups written whole only thanks to a retry


class AgenticRound(EngineRound):
    """``generate(rollout_id)`` of the sampler loop for agentic tasks (module doc).

    ``recorder``: a :class:`~yeto.rl.agentic.TokenRecorder` in front of this
    island's engine; ``harness``: an :class:`~yeto.rl.agentic.AgentHarness`;
    ``task_defaults``: :meth:`AgentTask.from_mapping` defaults (provider, agent
    version, model name, budget); ``sampling_params``: what each recorder
    session is opened with (checked against the proxy's signed sampling).

    Streaming (module doc): ``weight_switch`` ``auto`` (default: ``live``
    when the engine updates during a generation, else ``drain``), ``drain`` or
    ``live``; ``drain_timeout_s``: the longest drain before the trials still
    running are cancelled (None: wait for them; each is capped by its
    ``AgentBudget``); ``release_poll_s``: how often a drain-mode call reads the
    pending release; ``wait_s``: the calling thread's wake-up period.

    ``infra_retries``: a trial that failed on infrastructure before
    the policy produced anything (:func:`infra_failure`) runs again in its slot
    with a new session, at most this many times per slot, so its group is still
    written whole instead of short (dropped by a trainer that trains whole
    groups). No trial gets a wall-clock cap beyond its own budget."""

    streams = True  # build_engine_sampler wires the follower to the stream

    def __init__(self, engine: Any, writer: Any, *, recorder: Any, harness: Any, max_in_flight: int,
                 max_response_len: int, max_prompt_len: int | None = None, min_samples: int = 2,
                 task_defaults: Mapping[str, Any] | None = None,
                 sampling_params: Mapping[str, Any] | None = None,
                 loop: LoopThread | None = None, weight_switch: str = "auto",
                 drain_timeout_s: float | None = None, release_poll_s: float = 5.0, wait_s: float = 0.5,
                 infra_retries: int = DEFAULT_INFRA_RETRIES,
                 clock: Callable[[], float] = time.monotonic, **kwargs: Any) -> None:
        kwargs.setdefault("reward", lambda item, completion: 0.0)  # the verifier's reward is used
        super().__init__(engine, writer, **kwargs)
        if int(min_samples) < 1 or int(min_samples) > int(self.sampling.n):
            raise ValueError(f"min_samples must be in [1, n={self.sampling.n}], got {min_samples}")
        if weight_switch not in WEIGHT_SWITCHES:
            raise ValueError(f"weight_switch must be one of {WEIGHT_SWITCHES}, got {weight_switch!r}")
        weight_switch = resolve_weight_switch(weight_switch, engine)
        if weight_switch == "live" and not updates_during_generation(engine):
            raise ValueError("weight_switch live needs an engine that updates its weights during a "
                             "generation (updates_during_generation)")
        self.recorder = recorder
        self.harness = harness
        self.max_in_flight = int(max_in_flight)
        self.max_response_len = int(max_response_len)
        self.max_prompt_len = None if max_prompt_len is None else int(max_prompt_len)
        self.min_samples = int(min_samples)
        self.task_defaults = dict(task_defaults or {})
        self.sampling_params = dict(sampling_params or {"temperature": self.sampling.temperature,
                                                        "top_p": self.sampling.top_p})
        self.loop = loop or LoopThread(workers=self.max_in_flight + 8)
        self.lost: Counter[str] = Counter()  # over the island's life
        self.trials = 0
        self.weight_switch = weight_switch
        self.live = weight_switch == "live"
        self.drain_timeout_s = None if drain_timeout_s is None else float(drain_timeout_s)
        self.release_poll_s = float(release_poll_s)
        self.wait_s = float(wait_s)
        self.clock = clock
        if int(infra_retries) < 0:
            raise ValueError(f"infra_retries must be >= 0, got {infra_retries}")
        self.infra_retries = int(infra_retries)
        self.retries_total: Counter[str] = Counter()  # over the island's life (loop thread)
        self.retries_exhausted = 0
        self._retry_events: deque[dict[str, Any]] = deque()
        # the seq of a release newer than the served one (build_engine_sampler: the follower's)
        self.release_pending: Callable[[], int | None] = lambda: None
        # the stream: the loop thread starts and ends trials, the calling thread settles groups
        self._gate: Any = None
        self._queue: deque[_Group] = deque()  # groups with trials not started yet
        self._running: dict[asyncio.Task, tuple[_Group, int]] = {}
        self._finished: deque[_Group] = deque()  # groups whose trials all ended, not settled
        self._unsettled: list[_Group] = []
        self._rounds: dict[int, _RoundState] = {}
        self._wake = threading.Event()
        self._error: BaseException | None = None
        self._open_started = 0  # groups with a trial started and one not ended
        self._paused = False  # draining: no new group starts
        self._cancel_reason = LOST_DRAIN
        self._busy, self._busy_mark = 0.0, clock()  # trial-seconds in flight
        self._drain_total, self._drain_since = 0.0, None
        self._drain_seq: int | None = None
        self._drain_in_flight = 0
        self._drained_seq: int | None = None
        self._pending_seq: int | None = None
        self._release_read = -float("inf")
        self._outcomes_mark: dict[str, int] = {}
        self._last_rollout: int | None = None

    def holds_weights(self) -> bool:
        """True while a group's trials run: the follower loads nothing then (drain)."""
        return self._open_started > 0

    def close(self) -> int:
        """End of the island: cancel the trials still running, delete the pooled
        boat sandboxes (the account returns to the sandboxes other runs hold) and
        stop the loop; returns the VMs deleted."""
        deleted = 0
        try:
            try:
                self._on_loop(self._cancel_all, "closed")
            except Exception:  # noqa: BLE001 - the loop may be gone already
                pass
            from yeto.rl.harness.harbor.boat_pool import close_pools_now

            deleted = int(close_pools_now() or 0)
        finally:
            self.loop.close()
        self.emit("rl_agentic_close", boat_vms_deleted=deleted, lost_total=dict(sorted(self.lost.items())),
                  trials_total=self.trials)
        return deleted

    # ------------------------------------------------------------------ trials
    def _task(self, item: PromptItem, session: str) -> Any:
        from yeto.rl.agentic import AgentTask, HarnessConfigError

        row = item.metadata.get(AGENT_TASK_KEY)
        if not isinstance(row, Mapping):
            raise HarnessConfigError(f"prompt {item.prompt_id}: no {AGENT_TASK_KEY} mapping")
        try:
            task = AgentTask.from_mapping(row, defaults=self.task_defaults)
        except ValueError as exc:
            raise HarnessConfigError(f"prompt {item.prompt_id}: {exc}") from exc
        return dataclasses.replace(task, trajectory_id=session)

    async def _trial(self, item: PromptItem, session: str, gate: Any) -> Any:
        from yeto.rl.agentic import run_agent_task
        from yeto.rl.agentic.finish import finish_sample

        task = self._task(item, session)
        endpoint = await self.recorder.open(session, self.sampling_params)
        try:
            trajectory = await run_agent_task(task, self.harness, endpoint, gate)
        except BaseException:
            try:
                await self.recorder.close(session)
            except Exception:  # noqa: BLE001 - the trial's error is the one to report
                pass
            raise
        return await finish_sample(trajectory, self.recorder, session, gate.metrics,
                                   max_response_len=self.max_response_len, max_prompt_len=self.max_prompt_len)

    # ------------------------------------------------------------------ the stream (loop thread)
    def _on_loop(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn`` on the loop thread and return its result."""
        async def call() -> Any:
            return fn(*args)

        return self.loop.run(call())

    def _busy_now(self) -> float:
        return self._busy + len(self._running) * (self.clock() - self._busy_mark)

    def _tick(self) -> None:
        """Close the in-flight integral before the running count changes."""
        now = self.clock()
        self._busy += len(self._running) * (now - self._busy_mark)
        self._busy_mark = now

    def _drain_now(self) -> float:
        return self._drain_total + (0.0 if self._drain_since is None else self.clock() - self._drain_since)

    def _enqueue(self, groups: Sequence[_Group]) -> None:
        self._queue.extend(groups)
        self._fill()

    def _fill(self) -> None:
        """Start queued trials up to ``max_in_flight``: the rest of a started group
        always, a new group unless draining or reclaimed."""
        while len(self._running) < self.max_in_flight and self._error is None:
            fresh = not self._paused and not self.reclaimed()
            group = next((g for g in self._queue if g.started_at is not None or fresh), None)
            if group is None:
                return
            self._start(group)

    def _start(self, group: _Group) -> None:
        if self._gate is None:
            from yeto.rl.agentic import AdmissionGate

            self._gate = AdmissionGate(self.max_in_flight, policy_version=self.engine.served_version)
        if group.retry_slots:
            j = group.retry_slots.popleft()
        else:
            j = group.started
            group.started += 1
        if not group.pending:
            self._queue.remove(group)
        now = self.clock()
        if group.started_at is None:
            group.started_at, group.busy_at_start, group.drain_at_start = now, self._busy_now(), self._drain_now()
            self._open_started += 1
        version = self.engine.served_version()
        group.start_versions[j] = version
        if group.version is None:
            group.version = version
        self._tick()
        task = self.loop.loop.create_task(self._timed(group, j, now))
        self._running[task] = (group, j)
        task.add_done_callback(self._trial_done)

    async def _timed(self, group: _Group, j: int, started: float) -> Any:
        try:
            return await self._trial(group.item, group.sessions[j], self._gate)
        finally:
            group.trial_s[j] = round(self.clock() - started, 3)

    def _trial_done(self, task: asyncio.Task) -> None:
        self._tick()
        group, j = self._running.pop(task)
        if task.cancelled():
            result: Any = _Lost(self._cancel_reason)
        elif task.exception() is not None:
            error = task.exception()
            if self._error is None:  # raised on the calling thread: the island stops
                self._error = error
            result = _Lost(type(error).__name__)
        else:
            result = task.result()
        if self._retry(group, j, result):
            self._fill()
            self._wake.set()
            return
        group.results[j] = result
        self._ended(group, 1)
        self._fill()
        self._wake.set()

    def _retry(self, group: _Group, j: int, result: Any) -> bool:
        """Put slot ``j`` back in the stream when ``result`` is an infrastructure
        failure before the policy ran and the slot has retries left (loop thread)."""
        name = infra_failure(result)
        if name is None or self._error is not None or self.reclaimed():
            return False
        # a trial that finished just before a drain gave up on it (its cancel came too
        # late) or while draining must not start again: the drain waits for it to end
        if group.abandoned or self._paused:
            return False
        if group.retries[j] >= self.infra_retries:
            self.retries_exhausted += 1
            return False
        group.retries[j] += 1
        group.retry_reasons.append(name)
        self.retries_total[name] += 1
        group.sessions[j] = session_id(group.base_sessions[j], f"a{group.retries[j]}")
        group.retry_slots.append(j)
        if group not in self._queue:
            self._queue.appendleft(group)  # a started group's trials go first
        self._retry_events.append(dict(rollout_id=group.rollout_id, prompt_id=group.item.prompt_id, slot=j,
                  attempt=group.retries[j], reason=name, trial_s=group.trial_s[j],
                  retries_total=sum(self.retries_total.values())))  # emitted by the calling thread
        return True

    def _ended(self, group: _Group, trials: int) -> None:
        group.done += trials
        if group.done < len(group.sessions):
            return
        life = self.clock() - (group.started_at or 0.0)
        busy = self._busy_now() - group.busy_at_start
        group.utilization = round(busy / (life * self.max_in_flight), 4) if life > 0 else None
        group.drain_s = round(self._drain_now() - group.drain_at_start, 3)
        self._open_started -= 1
        self._finished.append(group)

    def _abandon(self) -> int:
        """A drain reached its timeout: cancel the running trials and give up the
        unstarted trials of started groups (lost as ``drain_timeout``); returns
        how many trials."""
        self._cancel_reason = LOST_DRAIN
        abandoned = 0
        for group in [g for g in self._queue if g.started_at is not None]:
            self._queue.remove(group)
            slots = [*range(group.started, len(group.sessions)), *group.retry_slots]
            rest = len(slots)
            for j in slots:
                group.results[j] = _Lost(LOST_DRAIN)
            group.started = len(group.sessions)
            group.retry_slots.clear()
            group.abandoned = True
            abandoned += rest
            self._ended(group, rest)
        for task, (group, _j) in list(self._running.items()):
            group.abandoned = True  # cancel() is a no-op on a task whose callback is pending
            task.cancel()
            abandoned += 1
        self._wake.set()
        return abandoned

    def _cancel_all(self, reason: str) -> None:
        self._cancel_reason = reason
        self._queue.clear()
        for task in list(self._running):
            task.cancel()

    # ------------------------------------------------------------------ the stream (calling thread)
    def _raise(self) -> None:
        if self._error is not None:
            raise self._error

    def _release_due(self) -> int | None:
        """A pending release this island has not drained for yet (read at most
        every ``release_poll_s``), else None."""
        now = self.clock()
        if now - self._release_read >= self.release_poll_s:
            self._release_read = now
            self._pending_seq = self.release_pending()
        seq = self._pending_seq
        return None if seq is None or seq == self._drained_seq else int(seq)

    def _wants_work(self) -> bool:
        return self.reclaimed() or (not self._queue and len(self._running) < self.max_in_flight)

    def _stream(self, tally: _Tally, *, final: bool) -> None:
        """Settle the groups that end until the stream wants the next round's
        tasks, or until a drain ends (``final``: until every group is settled)."""
        drain_started = None
        abandoned = 0
        while True:
            self._wake.clear()
            while self._retry_events:
                self.emit(RETRY_EVENT, **self._retry_events.popleft())
            while self._finished:
                self._settle(self._finished.popleft(), tally)
            self._raise()
            if self._paused:
                if drain_started is None:
                    drain_started = self.clock()
                if self._open_started == 0 or self.reclaimed():
                    self._end_drain(self.clock() - drain_started, abandoned)
                    return
                if (not abandoned and self.drain_timeout_s is not None
                        and self.clock() - drain_started >= self.drain_timeout_s):
                    abandoned = max(1, self._on_loop(self._abandon))
                    continue
            elif final:
                if not self._unsettled or self.reclaimed():
                    return
            else:
                seq = None if self.live else self._release_due()
                if seq is not None:
                    self._begin_drain(seq)
                    continue
                if self._wants_work():
                    return
            self._wake.wait(self.wait_s)

    def _begin_drain(self, seq: int) -> None:
        self._drain_seq = seq
        self._drain_since = self.clock()
        self._drain_in_flight = len(self._running)
        self._paused = True

    def _end_drain(self, waited: float, abandoned: int) -> None:
        """No trial runs: the loop may load the pending release. New groups stay
        paused until the next call (after the loop's load)."""
        if self._drain_since is not None:
            self._drain_total += self.clock() - self._drain_since
        self._drain_since = None
        self._drained_seq = self._drain_seq
        self.emit(DRAIN_EVENT, seq=self._drain_seq, policy_version=self.engine.served_version(),
                  waited_s=round(waited, 3), in_flight=self._drain_in_flight,
                  timed_out=bool(abandoned), abandoned_trials=int(abandoned))

    def _settle(self, group: _Group, tally: _Tally) -> None:
        """Filter, score and write one group whose trials all ended (module doc)."""
        from yeto.rl.agentic import Outcome

        n = len(group.sessions)
        lost: Counter[str] = Counter()
        kept: list[tuple[Any, int, int]] = []  # (finished, behavior version, newest version)
        for j, finished in enumerate(group.results):
            if isinstance(finished, _Lost):
                lost[finished.reason] += 1
                continue
            if finished.segments is None or finished.trajectory.outcome is Outcome.ABORTED:
                lost[str(finished.trajectory.reason).split(":", 1)[0] or "aborted"] += 1
                continue
            turns = [int(t.policy_version) for t in finished.turns if t.policy_version is not None]
            start = group.start_versions[j]
            if self.live:  # the oldest version of any turn: the trainer's age gate and TIS see it
                seen = [v for v in (start, *turns) if v is not None]
                if not seen:  # no version is known for any of its tokens
                    lost[LOST_VERSION] += 1
                    continue
                kept.append((finished, min(seen), max(seen)))
            elif start != group.version or any(v != group.version for v in turns):
                lost[LOST_VERSION] += 1
            else:
                kept.append((finished, int(group.version), int(group.version)))
        self._unsettled.remove(group)
        rid = group.rollout_id
        state = self._rounds[rid]
        state.open -= 1
        if state.open == 0:
            del self._rounds[rid]
        tally.trials += n
        tally.lost.update(lost)
        tally.retried.update(group.retry_reasons)
        reason = None
        if len(kept) < self.min_samples:
            reason = "below_min_samples"
            tally.small += 1
        elif self.live and self.unchecked_in_flight():  # tokens of weights that failed a check
            reason = "unchecked_weights_in_flight"
        elif self.reclaimed():  # leaving: the replacement owns the shard from now on
            reason = "reclaimed"
        elif self.write_fenced():  # a newer incarnation holds the lease
            reason = "fenced"
        if reason is not None and reason != "below_min_samples":
            self.dropped_groups += 1
            tally.dropped += 1
        record = self.provenance.to_record()
        behavior = min((v for _, v, _ in kept), default=None)
        newest = max((v for _, _, v in kept), default=None)
        if reason is None:
            samples = []
            for j, (finished, version, top) in enumerate(kept):
                sample = agentic_sample(group.item, finished, version=version, provenance=record,
                                        group_index=group.position, index=group.position * n + j,
                                        versions=(version, top) if self.live else None)
                tally.finish[sample["metadata"]["finish_reason"]] += 1
                tally.tokens += sample["response_length"]
                tally.rewards.append(sample["reward"])
                tally.lengths.append(int(sum(sample["loss_mask"])))
                samples.append(sample)
            self._write_group(rid, group.item, samples, record=record, last=state.open == 0, cursor=state.cursor)
            tally.written += 1
            if group.retry_reasons and len(kept) == n:
                tally.refilled += 1
            tally.samples += len(samples)
        written = reason is None and group.spot is not None
        self._written(rid, [group.spot] if written else [],
                      keep=[g.spot for g in self._unsettled if g.spot is not None])
        served = self.engine.served_version()
        self.emit(GROUP_EVENT, rollout_id=rid, prompt_id=group.item.prompt_id, written=reason is None,
                  dropped_reason=reason, trials=n, kept=len(kept), lost=dict(sorted(lost.items())),
                  behavior_version=behavior, versions=None if behavior is None else [behavior, newest],
                  version_lag=None if behavior is None or served is None else int(served) - behavior,
                  group_s=round(self.clock() - (group.started_at or self.clock()), 3),
                  trial_s=list(group.trial_s), in_flight_utilization=group.utilization,
                  drain_wait_s=group.drain_s, weight_switch=self.weight_switch,
                  infra_retries=list(group.retries), retry_reasons=list(group.retry_reasons))

    def _outcomes(self) -> dict[str, int]:
        """The admission counters: totals since the last round event, in flight now."""
        if self._gate is None:
            return {}
        now = self._gate.metrics.snapshot()
        out = {k: v - self._outcomes_mark.get(k, 0) if k.endswith("_total") else v for k, v in now.items()}
        self._outcomes_mark = now
        return out

    def _emit_round(self, rollout_id: int, tally: _Tally, *, started: float, busy: float, drain: float,
                    **extra: Any) -> None:
        self.lost.update(tally.lost)
        self.trials += tally.trials
        n_lost = sum(tally.lost.values())
        span = self.clock() - started
        agentic = {"trials": tally.trials, "kept": tally.trials - n_lost, "lost": dict(sorted(tally.lost.items())),
                   "lost_rate": round(n_lost / tally.trials, 4) if tally.trials else 0.0,
                   "groups_below_min": tally.small, "outcomes": self._outcomes(),
                   "reward_mean": round(statistics.fmean(tally.rewards), 4) if tally.rewards else None,
                   "policy_tokens_mean": round(statistics.fmean(tally.lengths), 1) if tally.lengths else None,
                   "lost_total": dict(sorted(self.lost.items())), "trials_total": self.trials,
                   "weight_switch": self.weight_switch, "in_flight": len(self._running),
                   "in_flight_utilization": round((self._busy_now() - busy) / (span * self.max_in_flight), 4)
                   if span > 0 else None,
                   "drain_wait_s": round(self._drain_now() - drain, 3),
                   "infra_retries": dict(sorted(tally.retried.items())), "groups_refilled": tally.refilled,
                   "infra_retries_total": dict(sorted(self.retries_total.items())),
                   "infra_retries_exhausted_total": self.retries_exhausted}
        self.emit(ROUND_EVENT, rollout_id=int(rollout_id), policy_version=self.engine.served_version(),
                  groups=tally.written, samples=tally.samples, response_tokens=tally.tokens,
                  finish=dict(sorted(tally.finish.items())), source=self.provenance.key(),
                  dropped_groups=tally.dropped, agentic=agentic, **extra)

    # ------------------------------------------------------------------ round
    def __call__(self, rollout_id: int) -> None:
        """Register round ``rollout_id``'s tasks in the stream, then settle the
        groups that end until the stream wants more tasks or a drain ends."""
        rollout_id = int(rollout_id)
        self._raise()
        if self.guard is not None:
            self.guard.check(rollout_id)
        items = list(self.prompts(rollout_id))
        if self.engine.served_version() is None:
            raise SamplerEngineError("the engine serves no version; it cannot generate")
        prefix = self.incarnation or "s"
        groups = []
        for position, item in enumerate(items):
            sessions = [session_id(prefix, f"r{rollout_id}", item.prompt_id, j) for j in range(self.sampling.n)]
            self._task(item, sessions[0])  # a row without a valid task fails before any trial starts
            groups.append(_Group(rollout_id, position, item, sessions))
        started, busy, drain = self.clock(), self._busy_now(), self._drain_now()
        self._last_rollout = rollout_id
        if groups:
            cursor = None if self.data_cursor is None else self.data_cursor(rollout_id)
            self._rounds[rollout_id] = _RoundState(cursor=cursor, open=len(groups))
            self._unsettled.extend(groups)
        self._paused = False  # a drain ended at the previous call; the loop loaded what it could
        seq = None if self.live else self._release_due()
        if seq is not None:  # before any of this round's groups start
            self._begin_drain(seq)
        self._on_loop(self._enqueue, groups)
        tally = _Tally()
        self._stream(tally, final=False)
        self._emit_round(rollout_id, tally, started=started, busy=busy, drain=drain)

    def flush(self) -> None:
        """After the last round (the sampler loop): settle every group still in the stream."""
        if not self._unsettled:
            return
        started, busy, drain = self.clock(), self._busy_now(), self._drain_now()
        self._paused = False
        self._on_loop(self._fill)
        tally = _Tally()
        self._stream(tally, final=True)
        self._emit_round(-1 if self._last_rollout is None else self._last_rollout, tally,
                         started=started, busy=busy, drain=drain, flush=True)


# ----------------------------------------------------------------------------- spec
def round_factory(spec: Mapping[str, Any], engine: Any, environ: Mapping[str, str]) -> Callable[..., AgenticRound]:
    """``make_round`` of :func:`~.engine_sampler.build_engine_sampler` from the
    spec's ``agentic`` section (``engine_sampler_entry``):

    * ``template`` (a :mod:`yeto.rl.harness.chat_templates` name), ``tokenizer``
      (default ``base_dir``), ``model`` (the model name the agent sends);
    * ``turn_max_tokens``: the signed ``max_tokens`` of each model call;
    * ``max_response_len`` / ``max_prompt_len``: the trainer's row widths (a
      longer stream is cut / ABORTED);
    * ``max_in_flight``: concurrent trials; ``min_samples`` (default 2);
    * ``weight_switch``: ``auto`` (default: ``live`` on a live-update engine,
      else ``drain``), ``drain`` or ``live``; ``drain_timeout_s``
      (default None: no cap beyond each trial's budget);
    * ``proxy``: ``{"host", "port"}`` of the records proxy; the public base and
      the path secret come from ``YETO_RECORDS_PROXY_PUBLIC_BASE`` and
      ``YETO_RECORDS_PROXY_SECRET_FILE`` (the Modal tunnel, written in the container);
    * ``infra_retries``: runs again of a trial that failed on infrastructure before
      the policy ran, per slot (default 2; 0 = off);
    * ``task_defaults``: :meth:`AgentTask.from_mapping` defaults; ``provider``
      falls back to ``HARBOR_ENV_TYPE``.

    Harbor's settings (task directories, trials, provider kwargs) are the
    environment variables both backends read (``HarborSettings.from_env``)."""
    from functools import partial
    from pathlib import Path

    from yeto.rl.agentic import HarnessConfigError
    from yeto.rl.harness.chat_templates import ChatTemplate
    from yeto.rl.harness.gateway import records_env
    from yeto.rl.harness.gateway.proxy import RecordsProxy
    from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder
    from yeto.rl.harness.harbor import HarborHarness, HarborSettings, build_trial_config
    from yeto.rl.harness.harbor.boat_client import check_in_flight_cap

    section = dict(spec["agentic"])
    try:
        sampling = dict(spec["sampling"])
        signed = {"max_tokens": int(section["turn_max_tokens"]),
                  "temperature": float(sampling.get("temperature", 1.0)), "top_p": float(sampling.get("top_p", 1.0))}
        tokenizer = section.get("tokenizer") or spec.get("base_dir")
        template = ChatTemplate.load(str(section["template"]), tokenizer_path=tokenizer)
        model = str(section.get("model", "model"))
        max_in_flight = int(section["max_in_flight"])
        max_response_len = int(section["max_response_len"])
        switch = str(section.get("weight_switch", "auto"))
        drain_timeout_s = None if section.get("drain_timeout_s") is None else float(section["drain_timeout_s"])
        infra_retries = int(section.get("infra_retries", DEFAULT_INFRA_RETRIES))
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessConfigError(f"spec agentic section: bad or missing {exc}") from exc
    proxy_cfg = dict(section.get("proxy") or {})
    secret_file = str(environ.get(records_env.PROXY_SECRET_FILE_ENV, "") or "").strip()
    public_base = str(environ.get(records_env.PROXY_PUBLIC_BASE_ENV, "") or "").strip() or None
    if public_base and not secret_file:
        raise HarnessConfigError("a public records-proxy base needs a path secret (the tunnel is public)")
    proxy = RecordsProxy(engine, template, model=model, signed_sampling=signed,
                         sampler_id=str(spec.get("sampler_id", "records-proxy")))
    recorder = ProxyTokenRecorder(proxy, host=str(proxy_cfg.get("host", "127.0.0.1")),
                                  port=int(proxy_cfg.get("port", 0)), public_base=public_base,
                                  path_secret=Path(secret_file).read_text().strip() if secret_file else None)
    settings = HarborSettings.from_env()
    harness = HarborHarness(partial(build_trial_config, settings=settings), setup_retries=settings.setup_retries)
    defaults = {"model_name": model, **dict(section.get("task_defaults") or {})}
    provider = str(environ.get("HARBOR_ENV_TYPE", "") or "").strip().lower()
    if provider and "provider" not in defaults:
        defaults["provider"] = provider
    if switch not in WEIGHT_SWITCHES:
        raise HarnessConfigError(f"spec agentic.weight_switch must be one of {WEIGHT_SWITCHES}, got {switch!r}")
    switch = resolve_weight_switch(switch, engine)
    if switch == "live" and not updates_during_generation(engine):
        raise HarnessConfigError("spec agentic.weight_switch live needs an engine with live_update")
    if defaults.get("provider") == "boat":
        check_in_flight_cap(max_in_flight)
    loop = LoopThread(workers=max_in_flight + 8)

    def make(engine_: Any, writer: Any, **kwargs: Any) -> AgenticRound:
        return AgenticRound(engine_, writer, recorder=recorder, harness=harness, max_in_flight=max_in_flight,
                            max_response_len=max_response_len,
                            max_prompt_len=None if section.get("max_prompt_len") is None
                            else int(section["max_prompt_len"]),
                            min_samples=int(section.get("min_samples", 2)), task_defaults=defaults,
                            sampling_params={"temperature": signed["temperature"], "top_p": signed["top_p"]},
                            weight_switch=switch, drain_timeout_s=drain_timeout_s, loop=loop,
                            infra_retries=infra_retries, **kwargs)

    return make
