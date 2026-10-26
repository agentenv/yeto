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

Configuration (worker environment):
  YETO_AGENTIC_RECORDER     ``module:callable`` returning a TokenRecorder (required)
  HARBOR_TASKS_DIR          Harbor task directories (required)
  HARBOR_TRIALS_DIR         trial output (default /tmp/harbor_trials)
  HARBOR_ENV_KWARGS         JSON provider kwargs (default {})
  YETO_HARBOR_MAX_IN_FLIGHT sandboxes per worker process (default 64)
"""

from __future__ import annotations

import importlib
import json
import os
import uuid
from functools import lru_cache, partial
from pathlib import Path
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
from yeto.rl.harness.harbor import HarborHarness, HarborSettings, build_trial_config

AGENT_LOOP_NAME = "yeto_harbor"
RECORDER_ENV = "YETO_AGENTIC_RECORDER"


@lru_cache(maxsize=1)
def _gate() -> AdmissionGate:
    return AdmissionGate(int(os.environ.get("YETO_HARBOR_MAX_IN_FLIGHT", 64)))


def _recorder_from_env() -> TokenRecorder:
    spec = os.environ.get(RECORDER_ENV, "")
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise HarnessConfigError(f"set {RECORDER_ENV}=module:callable (the N10.5 token recorder)")
    return getattr(importlib.import_module(module), attr)()


def _harness_from_env() -> AgentHarness:
    if not os.environ.get("HARBOR_TASKS_DIR"):
        raise HarnessConfigError("set HARBOR_TASKS_DIR to the Harbor task directories")
    settings = HarborSettings(
        tasks_dir=Path(os.environ["HARBOR_TASKS_DIR"]),
        trials_dir=Path(os.environ.get("HARBOR_TRIALS_DIR", "/tmp/harbor_trials")),
        provider_kwargs=json.loads(os.environ.get("HARBOR_ENV_KWARGS", "{}") or "{}"),
    )
    return HarborHarness(partial(build_trial_config, settings=settings))


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
        task = AgentTask.from_mapping(agent_task_row(kwargs))
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
