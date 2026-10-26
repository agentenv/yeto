"""verl backend of the Harbor agentic path: ``HarborAgentLoop(AgentLoopBase)`` (S25 N10.3).

verl's agent loop schedules samples and assembles the batch; Harbor runs the
trial; the yeto outer layer admits and classifies.  Neither verl nor Harbor is
modified: this class registers itself with verl's agent-loop registry under
``AGENT_LOOP_NAME`` and a data row selects it with ``agent_name``.

Per sample, ``run``:

1. reads the :class:`~yeto.rl.agentic.AgentTask` from the row
   (``extra_info.agent_task``);
2. opens a token-recorder session (N10.5) with verl's sampling parameters and
   hands its OpenAI URL to the agent (the session id is the task's
   ``trajectory_id`` plus a per-sample suffix: verl repeats a row ``n`` times);
3. runs one Harbor ``Trial`` through ``run_agent_task``;
4. joins the recorded calls into prompt / response / mask / logprobs
   (``join_turns``: a call that does not extend the stream is ABORTED, so the
   trained tokens are always the generated ones).

Calls aborted by verl's fully-async rollouter at a weight update are sent
again after a back-off (``rollout_engine`` module doc, S26 E9.5); the trial
keeps running, and a time limit hit after such a wait is ABORTED.

ABORTED on verl: the batch has no per-sample drop, so an ABORTED sample is a
one-token placeholder with mask 0 and reward 0, flagged in ``extra_fields``.
It adds no policy-gradient tokens.  The ``yeto_grpo`` estimator
(:mod:`.advantage`) keeps it out of its GRPO group's mean and std (N10.3b);
the loop refuses to start under another estimator.

Configuration (worker environment; pass it to the Ray workers, e.g. through
``ray_kwargs.ray_init.runtime_env.env_vars``):
  YETO_AGENTIC_RECORDER     ``module:callable`` returning a TokenRecorder (required),
                            e.g. ``yeto.rl.harness.gateway.recorder:recorder_from_env``.
                            verl builds the agent loop once per sample; the loop
                            calls the callable once per event loop and shares
                            the recorder (``LoopLocal``).
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

The admission gate, the recorder and the boat quota hold asyncio state and are kept
per event loop (:class:`~yeto.rl.agentic.loop_local.LoopLocal`).  verl runs
``AgentLoopWorker`` as a Ray async actor (one loop per process), so in a
worker they are per process.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
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
from yeto.rl.agentic.outcome import policy_output_failure
from yeto.rl.agentic.suspend import SuspendLedger, after_suspension
from yeto.rl.agentic.types import TokenTurn
from yeto.rl.engine.sampler_engine import FINISH_LENGTH
from yeto.rl.harness.harbor import HarborHarness, HarborSettings, build_trial_config
from yeto.rl.harness.harbor.boat_client import check_in_flight_cap

from .rollout_engine import bind_rollout_client, track_aborts, untrack_aborts, weight_steps
from .run_config_rules import AGENTIC_ADV_ESTIMATOR as YETO_GRPO

logger = logging.getLogger(__name__)

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


_recorders: LoopLocal[TokenRecorder] = LoopLocal(_recorder_from_env)


def _shared_recorder() -> TokenRecorder:
    """The recorder of this worker's event loop, built once from ``YETO_AGENTIC_RECORDER``."""
    return _recorders.get()


def recorder_session_id(task: AgentTask) -> str:
    """One recorder session per sample: verl repeats a row ``n`` times with the same task."""
    suffix = uuid.uuid4().hex[:12]
    return f"{task.trajectory_id}-{suffix}" if task.trajectory_id else uuid.uuid4().hex


def check_estimator(config: Any) -> None:
    """Refuse an advantage estimator that would count ABORTED placeholders (N10.3b).

    A config without ``algorithm.adv_estimator`` (a test stub) is not checked.
    """
    algorithm = getattr(config, "algorithm", None)
    estimator = getattr(algorithm, "adv_estimator", None)
    if estimator is not None and str(getattr(estimator, "value", estimator)) != YETO_GRPO:
        raise HarnessConfigError(
            f"HarborAgentLoop needs algorithm.adv_estimator={YETO_GRPO} (got {estimator}): another "
            "estimator counts ABORTED placeholder samples in their group's baseline")


def _harness_from_env() -> AgentHarness:
    settings = HarborSettings.from_env()
    return HarborHarness(partial(build_trial_config, settings=settings), setup_retries=settings.setup_retries)


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
    recorder_factory = staticmethod(_shared_recorder)
    harness_factory = staticmethod(_harness_from_env)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        check_estimator(self.config)
        self.prompt_length = int(self.rollout_config.prompt_length)
        self.response_length = int(self.rollout_config.response_length)
        self._harness = self.harness_factory()

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        task = task_from_row(kwargs)
        recorder = self.recorder_factory()
        client = getattr(self, "server_manager", None)
        if client is not None:  # a recorder over verl's own rollout needs verl's client (N10.9)
            bind_rollout_client(recorder, client)
        session = recorder_session_id(task)
        endpoint = await recorder.open(session, sampling_params)
        ledger = SuspendLedger(session, task_id=task.task_id)
        tracked = track_aborts(recorder, session, ledger)
        try:
            trajectory = await run_agent_task(task, self._harness, endpoint, _gate(),
                                              ledger=ledger if tracked else None)
        except BaseException:
            await _close_after_error(recorder, session)
            raise
        finally:
            if tracked:
                untrack_aborts(recorder, session)
        trajectory = _after_suspension(trajectory, ledger)
        turns: Sequence[TokenTurn] = ()
        try:
            turns = await recorder.close(session)
        except TokenAlignmentError as exc:
            if trajectory.outcome is not Outcome.ABORTED:
                trajectory = _abort(trajectory, f"token_alignment: {exc}")
        else:
            trajectory = _policy_output_failure(trajectory, turns)

        segments: TokenSegments | None = None
        if trajectory.outcome is not Outcome.ABORTED:
            try:
                segments = join_turns(turns, max_response_len=self.response_length)
            except TokenAlignmentError as exc:
                trajectory = _abort(trajectory, f"token_alignment: {exc}")
        if segments is not None and len(segments.prompt_ids) > self.prompt_length:
            trajectory, segments = _abort(trajectory, "prompt_exceeds_prompt_length"), None
        return self._output(trajectory, segments, turns, steps=weight_steps(recorder, session))

    def _output(self, trajectory: AgentTrajectory, segments: TokenSegments | None,
                turns: Sequence[TokenTurn], *, steps: tuple[int, int] | None = None) -> AgentLoopOutput:
        if segments is None:
            segments = self._placeholder()
        metadata = trajectory.to_metadata()
        # One line per trajectory: the only per-sample record of the outcome, the
        # suspension counters and the weight versions outside verl's batch.
        logger.warning("yeto harbor trajectory %s: outcome=%s reason=%s agent_metrics=%s global_steps=%s",
                       trajectory.task.trajectory_id, metadata[OUTCOME_KEY], trajectory.reason,
                       json.dumps(metadata["agent_metrics"], default=str, sort_keys=True), steps)
        return AgentLoopOutput(
            prompt_ids=list(segments.prompt_ids),
            response_ids=list(segments.response_ids),
            response_mask=list(segments.response_mask),
            response_logprobs=list(segments.response_logprobs),
            reward_score=trajectory.reward,
            num_turns=2 * len(turns),
            metrics=AgentLoopMetrics(),
            extra_fields={
                OUTCOME_KEY: metadata[OUTCOME_KEY],
                "outcome_reason": trajectory.reason,
                "policy_version": trajectory.policy_version,
                "truncated": segments.truncated,
                # Model calls that hit their token cap (the agent then saw a cut reply).
                "length_capped_turns": sum(t.finish_reason == FINISH_LENGTH for t in turns),
                "agent_metrics": metadata["agent_metrics"],
                "provenance": turn_provenance(turns),
                # verl V1 reads both for its staleness metrics; recorders without verl's rollout leave them out
                **({"min_global_steps": steps[0], "max_global_steps": steps[1]} if steps else {}),
            },
        )

    def _placeholder(self) -> TokenSegments:
        pad = self.tokenizer.pad_token_id
        pad = self.tokenizer.eos_token_id if pad is None else pad
        return TokenSegments(prompt_ids=(pad,), response_ids=(pad,), response_mask=(0,), response_logprobs=(0.0,))


def _policy_output_failure(trajectory: AgentTrajectory, turns: Sequence[TokenTurn]) -> AgentTrajectory:
    """A malformed model reply that ended the agent is a 0 reward, not ABORTED (N10.9)."""
    updated = policy_output_failure(trajectory, [t.policy_error for t in turns if t.policy_error])
    if updated.outcome is not trajectory.outcome:
        _gate().metrics.reclassify(trajectory.outcome, updated.outcome)
    return updated


def _after_suspension(trajectory: AgentTrajectory, ledger: SuspendLedger) -> AgentTrajectory:
    """Suspension metrics; a time limit after an aborted-and-retried call is ABORTED (S26 E9.5)."""
    updated = after_suspension(trajectory, ledger)
    if updated.outcome is not trajectory.outcome:
        _gate().metrics.reclassify(trajectory.outcome, updated.outcome)
    return updated


def _abort(trajectory: AgentTrajectory, reason: str) -> AgentTrajectory:
    _gate().metrics.reclassify(trajectory.outcome, Outcome.ABORTED)
    return AgentTrajectory(task=trajectory.task, outcome=Outcome.ABORTED, reward=0.0, reason=reason,
                           verdict=trajectory.verdict, policy_version=trajectory.policy_version,
                           extra_metrics=trajectory.extra_metrics)


def turn_provenance(turns: Sequence[TokenTurn]) -> Mapping[str, str] | None:
    """The sampler source of the sample: the one every call shares, else None.

    Per-source metrics (``offpolicy_source``) group by it; a sample whose calls
    came from different sources is counted under ``unknown``.
    """
    records = {tuple(sorted(t.provenance.items())) for t in turns if t.provenance}
    if len(records) != 1 or any(not t.provenance for t in turns):
        return None
    return dict(next(iter(records)))


async def _close_after_error(recorder: TokenRecorder, session: str) -> None:
    """Close the recorder session after the trial raised; the trial's error is the one to report."""
    try:
        await recorder.close(session)
    except Exception:  # the trial's error propagates
        logger.warning("recorder close failed for %s after a trial error", session, exc_info=True)
