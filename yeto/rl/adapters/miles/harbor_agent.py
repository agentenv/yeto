"""Miles backend of the Harbor agentic path (S25 N10.1/N10.2).

Miles' own Harbor example (``examples/experimental/harbor``) is the backend
entry: its launcher wiring, its ``TrialConfig`` builder
(``harbor_agent_function.build_trial_config``: harness bindings, provider
credentials, resource and timeout knobs) and Miles' session server (token
recording, TITO) are used as they are.  yeto replaces only two hooks:

* ``--custom-agent-function-path yeto.rl.adapters.miles.harbor_agent.run``:
  the example's ``run`` with the yeto outer layer around ``Trial.run()``
  (admission, OK / POLICY_LIMIT / ABORTED, trust checks, metrics).  The
  example scores an infrastructure failure 0 (``AgentError``); here it is
  ABORTED and never trained on.
* ``--custom-rm-path yeto.rl.adapters.miles.harbor_agent.reward_func``: reads
  the yeto outcome and marks ABORTED samples.

Providers Harbor does not ship (``trial.CUSTOM_PROVIDERS``, e.g. ``boat``):
the example maps ``HARBOR_ENV_TYPE`` to Harbor's ``EnvironmentType`` and
rejects them.  For these the ``TrialConfig`` comes from yeto's
``build_trial_config`` (the verl builder) with ``HarborSettings.from_env()``;
the example's harness bindings are not applied, so only agents that
``trial.agent_env`` wires (OpenAI base URL: codex, oracle, nop, ...) run there.

Suspension between rounds (S26 E9.5, Miles stage 3,
``--agentic-suspend-between-turns``): Miles calls this module's
:func:`suspend` at a round's cut-off and :func:`resume` at the next round's
start.  They close and open the loop's
:class:`~yeto.rl.agentic.suspend.TurnGate`.  The agent dials a
:class:`~yeto.rl.harness.gateway.gate_relay.GateRelay` in this process instead
of the session server, so its next model call waits there while the trial
(sandbox, agent process, session-server tokens) stays alive.  The relay is
switched on by ``YETO_HARBOR_GATE_RELAY_PORT`` (:func:`relay_settings`); a
suspend hook without it fails closed.  A trajectory suspended across a process
restart is lost: the cut holds only its reference
(:func:`yeto.rl.agentic.suspend.suspended_references`) and the restore
discards it as ABORTED.

The example module is imported by name (``harbor_agent_function``): the Miles
launcher puts the example directory on the workers' PYTHONPATH
(``launch_common.agentic_pythonpath_dirs``).
"""

from __future__ import annotations

import importlib
import os
import uuid
import weakref
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from yeto.rl.agentic import (
    OUTCOME_KEY,
    AdmissionGate,
    AgentTask,
    HarnessConfigError,
    Outcome,
    run_agent_task,
)
from yeto.rl.agentic.loop_local import LoopLocal
from yeto.rl.agentic.suspend import TurnGate, after_suspension
from yeto.rl.agentic.trust import check_agent_env
from yeto.rl.harness.gateway.gate_relay import GateRelay
from yeto.rl.harness.harbor import CODEX_VERSION, HarborHarness, HarborSettings, build_trial_config
from yeto.rl.harness.harbor.trial import CODEX_KWARGS, CUSTOM_PROVIDERS, TrialConfigBuilder

EXAMPLE_MODULE = "harbor_agent_function"
AGENT_FUNCTION = "yeto.rl.adapters.miles.harbor_agent.run"
REWARD_FUNCTION = "yeto.rl.adapters.miles.harbor_agent.reward_func"
DEFAULT_AGENT = "codex"  # N10.7: Harbor's built-in codex agent
MAX_IN_FLIGHT_ENV = "YETO_HARBOR_MAX_IN_FLIGHT"
_DEFAULT_MAX_IN_FLIGHT = 64


class UntrustedAgentOutcome(RuntimeError):
    """A sample reached the reward hook without a yeto outcome (the agent function raised)."""


_gates: LoopLocal[AdmissionGate] = LoopLocal(
    lambda: AdmissionGate(int(os.environ.get(MAX_IN_FLIGHT_ENV, _DEFAULT_MAX_IN_FLIGHT))))


def gate() -> AdmissionGate:
    """The gate of the rollout loop (one sandbox per admitted trajectory).

    Miles runs its rollout on one event loop, so this is one gate per worker
    process; it is kept per loop because it holds an ``asyncio.Semaphore``.
    """
    return _gates.get()


# --------------------------------------------------------------- suspension

RELAY_PORT_ENV = "YETO_HARBOR_GATE_RELAY_PORT"
RELAY_HOST_ENV = "YETO_HARBOR_GATE_RELAY_HOST"
RELAY_PUBLIC_HOST_ENV = "YETO_HARBOR_GATE_RELAY_PUBLIC_HOST"
RELAY_PUBLIC_BASE_ENV = "YETO_HARBOR_GATE_RELAY_PUBLIC_BASE"
RELAY_SECRET_FILE_ENV = "YETO_HARBOR_GATE_RELAY_SECRET_FILE"


@dataclass(frozen=True)
class RelaySettings:
    """Where the gate relay listens and the address the sandbox dials.

    ``public_host`` None: the host of the session-server URL the agent would
    have dialed (the relay runs in the same process, so the same machine).
    """

    port: int
    host: str = "0.0.0.0"
    public_host: str | None = None
    public_base: str | None = None
    path_secret: str | None = None


def relay_settings(environ: Any = None) -> RelaySettings | None:
    """The relay settings, or None when ``YETO_HARBOR_GATE_RELAY_PORT`` is unset."""
    environ = os.environ if environ is None else environ
    raw = str(environ.get(RELAY_PORT_ENV, "") or "").strip()
    if not raw:
        return None
    if not raw.isdigit() or int(raw) > 65535:
        raise HarnessConfigError(f"{RELAY_PORT_ENV} must be a TCP port, got {raw!r}")
    secret_file = str(environ.get(RELAY_SECRET_FILE_ENV, "") or "").strip()
    return RelaySettings(
        port=int(raw), host=str(environ.get(RELAY_HOST_ENV) or "0.0.0.0"),
        public_host=environ.get(RELAY_PUBLIC_HOST_ENV) or None,
        public_base=environ.get(RELAY_PUBLIC_BASE_ENV) or None,
        path_secret=Path(secret_file).read_text().strip() if secret_file else None)


def check_suspend_config(environ: Any = None) -> RelaySettings:
    """Island start under ``--agentic-suspend-between-turns``: the relay must be configured."""
    settings = relay_settings(environ)
    if settings is None:
        raise HarnessConfigError(f"suspension between turns needs the gate relay: set {RELAY_PORT_ENV} "
                                 "(a port the sandboxes can reach on this machine)")
    return settings


_turn_gates: LoopLocal[TurnGate] = LoopLocal(TurnGate)
_relays: weakref.WeakKeyDictionary[TurnGate, GateRelay] = weakref.WeakKeyDictionary()


def turn_gate() -> TurnGate:
    """The model-turn gate of the rollout loop."""
    return _turn_gates.get()


def _relay(settings: RelaySettings, session_url: str) -> GateRelay:
    gate = turn_gate()
    relay = _relays.get(gate)
    if relay is None:
        public_host = settings.public_host or urlsplit(session_url).hostname or "127.0.0.1"
        relay = _relays[gate] = GateRelay(gate, host=settings.host, port=settings.port,
                                              public_host=public_host, public_base=settings.public_base,
                                              path_secret=settings.path_secret)
    return relay


async def suspend(args: Any) -> None:
    """Miles hook at the cut-off: close the gate (fail closed without the relay)."""
    del args
    check_suspend_config()
    turn_gate().close()


async def resume(args: Any) -> None:
    """Miles hook at the start of the next round: open the gate."""
    del args
    turn_gate().open()


def _example() -> Any:
    return importlib.import_module(EXAMPLE_MODULE)


def task_from_metadata(metadata: dict[str, Any]) -> AgentTask:
    """Miles sample metadata (``instance_id``, ``agent_name``) -> :class:`AgentTask`."""
    # Miles names the agent ``agent_name``; the default must use the same key or
    # it would shadow the sample's value in ``AgentTask.from_mapping``.
    defaults = {
        "agent_name": DEFAULT_AGENT,
        "provider": os.environ.get("HARBOR_ENV_TYPE", "").strip().lower(),
        "wall_clock_s": float(os.environ.get("AGENT_TRIAL_TIMEOUT", "7200")),
        "model_name": os.environ.get("AGENT_MODEL_NAME", "model"),
        "agent_timeout_s": os.environ.get("AGENT_TIMEOUT") or None,  # as the example reads it
    }
    try:
        return AgentTask.from_mapping(metadata, defaults=defaults)
    except (TypeError, ValueError) as exc:
        raise HarnessConfigError(f"sample metadata is not an agent task: {exc}") from exc


def example_config_builder(example: Any, metadata: dict[str, Any], request_kwargs: dict[str, Any]):
    """The Miles example's ``build_trial_config``, adapted to the yeto builder signature."""

    def build(task: AgentTask, session_url: str) -> Any:
        try:
            config = example.build_trial_config({**metadata, "agent_name": task.agent}, session_url, request_kwargs)
        except (FileNotFoundError, ValueError) as exc:
            raise HarnessConfigError(str(exc)) from exc
        check_agent_env(config.agent.env)
        if task.agent == "codex":
            # The example has no codex binding; its default (OPENAI_BASE_URL) is
            # what Harbor's codex agent reads.  Pin the build: Harbor's default is @latest.
            config.agent.kwargs.setdefault("version", task.agent_version or CODEX_VERSION)
            for key, value in CODEX_KWARGS.items():
                config.agent.kwargs.setdefault(key, value)
        return config

    return build


def trial_config_builder(task: AgentTask, example: Any, metadata: dict[str, Any],
                         request_kwargs: dict[str, Any]) -> TrialConfigBuilder:
    """The example's builder for Harbor's providers; yeto's for :data:`CUSTOM_PROVIDERS`."""
    if task.provider in CUSTOM_PROVIDERS:
        return partial(build_trial_config, settings=HarborSettings.from_env())
    return example_config_builder(example, metadata, request_kwargs)


async def run(base_url: str, prompt: Any = None, request_kwargs: dict[str, Any] | None = None,
              metadata: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    """Miles ``--custom-agent-function-path`` entry: one Harbor trial under the yeto outer layer."""
    from miles.rollout.agentic.session import openai_session_url

    del prompt  # Harbor reads the instruction from the task directory
    metadata = metadata or {}
    example = _example()
    example._ensure_provider_key()  # the example's credential contract (key file -> SDK env var); no-op for boat
    task = task_from_metadata(metadata)
    harness = HarborHarness(trial_config_builder(task, example, metadata, request_kwargs or {}))
    session_url = openai_session_url(base_url)
    settings = relay_settings()
    if settings is None:
        trajectory = await run_agent_task(task, harness, session_url, gate())
        return trajectory.to_metadata()
    turns = turn_gate()
    # Miles repeats a row n_samples_per_prompt times with the same metadata.
    suffix = uuid.uuid4().hex[:12]
    ledger = turns.register(f"{task.trajectory_id}-{suffix}" if task.trajectory_id else uuid.uuid4().hex,
                            task_id=task.task_id)
    relay = _relay(settings, session_url)
    endpoint = await relay.open(session_url, ledger)
    try:
        trajectory = await run_agent_task(task, harness, endpoint, gate(), ledger=ledger)
    finally:
        relay.close(endpoint)
        turns.unregister(ledger)
    updated = after_suspension(trajectory, ledger)
    if updated.outcome is not trajectory.outcome:
        gate().metrics.reclassify(trajectory.outcome, updated.outcome)
    return updated.to_metadata()


def _mark_aborted(sample: Any) -> None:
    try:
        from miles.utils.types import Sample

        sample.status = Sample.Status.ABORTED
    except ImportError:  # CPU tests without the Miles package
        sample.status = "ABORTED"


def sample_reward(sample: Any) -> float:
    metadata = getattr(sample, "metadata", None) or {}
    if OUTCOME_KEY not in metadata:
        # Miles still collects tokens when the agent function raised; the only
        # raises are configuration and trust errors, which must stop the run.
        raise UntrustedAgentOutcome("sample has no yeto agent outcome; the agent function failed")
    if Outcome(metadata[OUTCOME_KEY]) is Outcome.ABORTED:
        _mark_aborted(sample)
        return 0.0
    return float(metadata["reward"])


async def reward_func(args: Any, samples: Any, **_: Any) -> float | list[float]:
    del args
    if isinstance(samples, list):
        return [sample_reward(sample) for sample in samples]
    return sample_reward(samples)


def train_args(*, tito_model: str, session_server_workers: int, session_server_port: int = 30000,
               session_server_external_host: str = "") -> str:
    """Miles train.py flags: the example's ``agentic_train_args`` with the two yeto hooks.

    Same flags as ``launch_common.agentic_train_args`` (checked by a CPU test
    against the pinned Miles source) except the agent function and reward hook.
    ``check_no_aborted`` drops a group with an ABORTED sample before training.
    """
    external = f"--session-server-external-host {session_server_external_host} " if session_server_external_host else ""
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        f"--custom-agent-function-path {AGENT_FUNCTION} "
        f"--custom-rm-path {REWARD_FUNCTION} "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--tito-model {tito_model} "
        "--use-session-server "
        f"--session-server-port {session_server_port} "
        f"--session-server-workers {session_server_workers} "
        f"{external}"
    )
