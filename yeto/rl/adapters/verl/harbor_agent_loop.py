"""verl backend of the Harbor agentic path: ``HarborAgentLoop(AgentLoopBase)`` (S25 N10.3).

verl's agent loop schedules samples and assembles the batch; Harbor runs the
trial; the yeto outer layer admits and classifies.  Neither verl nor Harbor is
modified: this class registers itself with verl's agent-loop registry under
``AGENT_LOOP_NAME`` and a data row selects it with ``agent_name``.

Per sample, ``run``:

1. reads the :class:`~yeto.rl.agentic.AgentTask` from the row
   (``extra_info.agent_task``);
2. opens a token-recorder session (N10.5) with verl's sampling parameters and
   hands its OpenAI URL to the agent;
3. runs one Harbor ``Trial`` through ``run_agent_task``;
4. joins the recorded calls into prompt / response / mask / logprobs
   (``join_turns``: a call that does not extend the stream is ABORTED, so the
   trained tokens are always the generated ones).

ABORTED on verl: the batch has no per-sample drop, so an ABORTED sample is a
one-token placeholder with mask 0 and reward 0, flagged in ``extra_fields``.
It adds no policy-gradient tokens, but it still counts in its GRPO group's
reward mean; a group-level drop on verl is not implemented (not verified).

Configuration (worker environment; pass it to the Ray workers, e.g. through
``ray_kwargs.ray_init.runtime_env.env_vars``):
  YETO_AGENTIC_RECORDER     ``module:callable`` returning a TokenRecorder (required).
                            verl builds the agent loop once per sample, so the
                            callable is called per sample; return a shared object.
  HARBOR_TASKS_DIR          Harbor task directories (required)
  HARBOR_TRIALS_DIR         trial output (default /tmp/harbor_trials)
  HARBOR_ENV_TYPE           sandbox provider of rows without ``agent_task.provider``
                            (``boat``, ``modal``, ``docker``, ...; same name as Miles)
  HARBOR_ENV_KWARGS         JSON provider kwargs (default {}); boat: ``machine_type``
                            (default: smallest that fits the task), ``ttl_s``
  HARBOR_TIMEOUT_MULTIPLIER Harbor timeout multiplier (default 1.0)
  YETO_HARBOR_MAX_IN_FLIGHT sandboxes per worker process (default 64)
  YETO_BOAT_MAX_ACTIVE, YETO_BOAT_STARTS_PER_MINUTE
                            boat caps per worker process (``boat_client``)

Concurrency: verl runs ``rollout.agent.num_workers`` worker processes, so the
account holds up to ``num_workers * YETO_HARBOR_MAX_IN_FLIGHT`` sandboxes.  For
boat, a worker refuses a ``YETO_HARBOR_MAX_IN_FLIGHT`` above
``YETO_BOAT_MAX_ACTIVE``; keeping ``num_workers * YETO_BOAT_MAX_ACTIVE`` within
the plan (box_20: 100 active, 12 starts per minute) is the launcher's job.

The admission gate and the boat quota hold asyncio primitives and are kept
per event loop (:class:`~yeto.rl.agentic.loop_local.LoopLocal`).  verl runs
``AgentLoopWorker`` as a Ray async actor (one loop per process), so in a
worker they are per process.
"""

from __future__ import annotations

import importlib
import os
import uuid
from functools import partial
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register

from yeto.rl.agentic import (
    OUTCOME_KEY,
    AdmissionGate,
    AgentHarness,
    AgentTask,
    AgentTrajectory,
    HarnessConfigError,
    Outcome,
    TokenAlignmentError,
    TokenRecorder,
    TokenSegments,
    join_turns,
    run_agent_task,
)
from yeto.rl.agentic.loop_local import LoopLocal
from yeto.rl.harness.harbor import HarborHarness, HarborSettings, build_trial_config
from yeto.rl.harness.harbor.boat_client import check_in_flight_cap

AGENT_LOOP_NAME = "yeto_harbor"
RECORDER_ENV = "YETO_AGENTIC_RECORDER"
PROVIDER_ENV = "HARBOR_ENV_TYPE"
MAX_IN_FLIGHT_ENV = "YETO_HARBOR_MAX_IN_FLIGHT"
_DEFAULT_MAX_IN_FLIGHT = 64


def _max_in_flight() -> int:
    return int(os.environ.get(MAX_IN_FLIGHT_ENV, _DEFAULT_MAX_IN_FLIGHT))


_gates: LoopLocal[AdmissionGate] = LoopLocal(lambda: AdmissionGate(_max_in_flight()))


def _gate() -> AdmissionGate:
    """The admission gate of this worker's event loop (one sandbox per admitted sample)."""
    return _gates.get()


def _recorder_from_env() -> TokenRecorder:
    spec = os.environ.get(RECORDER_ENV, "")
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise HarnessConfigError(f"set {RECORDER_ENV}=module:callable (the N10.5 token recorder)")
    return getattr(importlib.import_module(module), attr)()


def _harness_from_env() -> AgentHarness:
    return HarborHarness(partial(build_trial_config, settings=HarborSettings.from_env()))


def task_from_row(kwargs: dict[str, Any]) -> AgentTask:
    """The sample's :class:`AgentTask`; ``HARBOR_ENV_TYPE`` fills a missing provider.

    A boat task is refused (configuration error, fails the batch) when this
    worker's gate admits more sandboxes than its boat quota holds.
    """
    row = agent_task_row(kwargs)
    provider = os.environ.get(PROVIDER_ENV, "").strip().lower()
    try:
        task = AgentTask.from_mapping(row, defaults={"provider": provider or None})
        if task.provider == "boat":
            check_in_flight_cap(_max_in_flight())
    except ValueError as exc:
        raise HarnessConfigError(str(exc)) from exc
    return task


def agent_task_row(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The ``agent_task`` mapping of a verl dataset row (``extra_info.agent_task``)."""
    extra = kwargs.get("extra_info") or {}
    row = extra.get("agent_task") if isinstance(extra, dict) else None
    if not isinstance(row, dict):
        raise HarnessConfigError("verl row has no extra_info.agent_task mapping")
    return row


@register(AGENT_LOOP_NAME)
class HarborAgentLoop(AgentLoopBase):
    """One Harbor trial per sample; tokens from the recorder, verdict from Harbor."""

    # Factories are class attributes so a test (or another wiring) can replace them.
    recorder_factory = staticmethod(_recorder_from_env)
    harness_factory = staticmethod(_harness_from_env)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_length = int(self.rollout_config.prompt_length)
        self.response_length = int(self.rollout_config.response_length)
        self._recorder = self.recorder_factory()
        self._harness = self.harness_factory()

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        task = task_from_row(kwargs)
        trajectory_id = task.trajectory_id or uuid.uuid4().hex
        endpoint = await self._recorder.open(trajectory_id, sampling_params)
        try:
            trajectory = await run_agent_task(task, self._harness, endpoint, _gate())
        finally:
            turns = await self._recorder.close(trajectory_id)

        segments: TokenSegments | None = None
        if trajectory.outcome is not Outcome.ABORTED:
            try:
                segments = join_turns(turns, max_response_len=self.response_length)
            except TokenAlignmentError as exc:
                trajectory = _abort(trajectory, f"token_alignment: {exc}")
        if segments is not None and len(segments.prompt_ids) > self.prompt_length:
            trajectory, segments = _abort(trajectory, "prompt_exceeds_prompt_length"), None
        return self._output(trajectory, segments, n_calls=len(turns))

    def _output(self, trajectory: AgentTrajectory, segments: TokenSegments | None, *, n_calls: int) -> AgentLoopOutput:
        if segments is None:
            segments = self._placeholder()
        metadata = trajectory.to_metadata()
        return AgentLoopOutput(
            prompt_ids=list(segments.prompt_ids),
            response_ids=list(segments.response_ids),
            response_mask=list(segments.response_mask),
            response_logprobs=list(segments.response_logprobs),
            reward_score=trajectory.reward,
            num_turns=2 * n_calls,
            metrics=AgentLoopMetrics(),
            extra_fields={
                OUTCOME_KEY: metadata[OUTCOME_KEY],
                "outcome_reason": trajectory.reason,
                "policy_version": trajectory.policy_version,
                "truncated": segments.truncated,
                "agent_metrics": metadata["agent_metrics"],
            },
        )

    def _placeholder(self) -> TokenSegments:
        pad = self.tokenizer.pad_token_id
        pad = self.tokenizer.eos_token_id if pad is None else pad
        return TokenSegments(prompt_ids=(pad,), response_ids=(pad,), response_mask=(0,), response_logprobs=(0.0,))


def _abort(trajectory: AgentTrajectory, reason: str) -> AgentTrajectory:
    _gate().metrics.reclassify(trajectory.outcome, Outcome.ABORTED)
    return AgentTrajectory(task=trajectory.task, outcome=Outcome.ABORTED, reward=0.0, reason=reason,
                           verdict=trajectory.verdict, policy_version=trajectory.policy_version)
