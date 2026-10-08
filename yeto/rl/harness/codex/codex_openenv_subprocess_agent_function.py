"""Trusted-side entry for the Terminal-Bench Codex adapter (REWRITE).

``run`` is the ``--custom-agent-function-path`` target under upstream
``agentic_tool_call.generate``.  It owns the environment lease, spawns the
untrusted worker with the reward key scrubbed from its environment, maps the
worker's tool-wait events onto the ToolWaitBoard (3.1), runs the verifier and
signs the outcome (D7), and guarantees cleanup on cancellation/timeouts:
worker process group, environment lease, tool-wait scope.

Environment provisioning is injected (``EnvironmentProvider``) so the sandbox
backend (D9) and the CPU fake share one code path.  The boards are installed
by the IR-1 preflight hook (``preflight.harness_preflight``): the island's
``ToolWaitBoard`` (3.1) and ``HarnessBoard`` (IR-2) actors.  IR-2 wiring here:
``allow_new_session`` before any environment is acquired (drain closes
admission; fail closed -> infrastructure ABORTED, never a 0 reward),
``enter_session``/``exit_session`` around the worker, ``lease_acquired``
(with the hard deadline) / ``lease_released`` around the environment lease so
``env_live`` counts idle sandboxes too.  IR-3: the target policy token comes
from ``rollout_meta_hook.expected_policy_version`` (prompt metadata first,
else the driver's token in the metadata sink); a missing token refuses the
trajectory before any environment is acquired.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Protocol

from yeto.rl.adapters.miles import rollout_meta_hook
from yeto.rl.engine.tool_wait import _call as _board_call
from yeto.rl.engine.tool_wait import _resolve as _board_resolve

from . import codex_openenv_agent_function as adapter
from .environment import TerminalEnvironment, TrustedVerifier

WORKER_MODULE = "yeto.rl.harness.codex.codex_openenv_agent_worker"
WORKER_GRACE_SECONDS = 5.0


@dataclass
class EnvironmentLease:
    """One environment bound to one trajectory (R-D9: deadline is a hard cap)."""

    env_url: str
    env_token: str
    verifier: TrustedVerifier
    destroy: Callable[[], Awaitable[None]]
    describe: Callable[[], Awaitable[str]]  # "live" | "gone"
    deadline_seconds: float
    # Extra environment for the worker process (e.g. a per-trajectory
    # ``SECRLENV_MAX_TURNS`` turn budget injected by the provider); never a
    # reward key (the worker refuses to start with one).
    worker_env: dict[str, str] | None = None


class EnvironmentProvider(Protocol):
    async def acquire(self, task_id: str, trajectory_id: str) -> EnvironmentLease: ...


_provider: EnvironmentProvider | None = None
_tool_wait_board: Any = None
_harness_board: Any = None
_member: str | None = None

ADMISSION_CLOSED = "harness_admission_closed"
# Consecutive non-injected acquire failures before the island is torn down.
# Each failure is reported as INFRASTRUCTURE (sample ABORTED); upstream
# Miles then draws the next group forever (``generate_rollout`` has no
# abort cap: codex-smoke-20261003-7 looped 17k samples in six minutes on a
# broken provider), so after this many the worker raises
# ``EnvironmentProviderOutage`` instead.
PROVIDER_OUTAGE_THRESHOLD_ENV = "YETO_HARNESS_PROVIDER_OUTAGE_THRESHOLD"
DEFAULT_PROVIDER_OUTAGE_THRESHOLD = 4
_acquire_failures = 0


class EnvironmentProviderOutage(BaseException):
    """The environment provider failed repeatedly: stop the rollout, do not resample.

    A ``BaseException`` on purpose: upstream Miles wraps the agent function and
    each rollout task in ``except Exception`` and keeps generating, so only a
    BaseException escapes ``generate_rollout`` and fails the island.
    """


def _provider_outage_threshold() -> int:
    try:
        return max(1, int(os.environ.get(PROVIDER_OUTAGE_THRESHOLD_ENV) or DEFAULT_PROVIDER_OUTAGE_THRESHOLD))
    except ValueError:
        return DEFAULT_PROVIDER_OUTAGE_THRESHOLD


def _note_acquire_failure(exc: BaseException, trajectory_id: str) -> None:
    """Count real provisioning failures; injected ones (fault tests) do not count."""
    global _acquire_failures
    if getattr(exc, "injected_fault", False):
        return
    _acquire_failures += 1
    threshold = _provider_outage_threshold()
    if _acquire_failures >= threshold:
        raise EnvironmentProviderOutage(
            f"environment provider failed {_acquire_failures} consecutive acquires "
            f"(threshold {threshold}); last for {trajectory_id!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _note_acquire_success() -> None:
    global _acquire_failures
    _acquire_failures = 0
POLICY_VERSION_MISSING = "expected_policy_version missing"


class PolicyVersionMissing(RuntimeError):
    """IR-3: the driver published no target policy token for this rollout (refuse)."""


def configure(
    *,
    provider: EnvironmentProvider | None,
    tool_wait_board: Any = None,
    harness_board: Any = None,
    member: str | None = None,
) -> None:
    """Install the environment provider and the island boards.

    ``tool_wait_board``: ``ToolWaitBoard`` (local or actor handle, 3.1).
    ``harness_board``: ``HarnessBoard`` (local or actor handle, IR-2); None
    means the island reports "harness counts unknown" and stays undrainable.
    ``member``: the rollout member this process targets (admission key).
    """
    global _provider, _tool_wait_board, _harness_board, _member, _acquire_failures
    _acquire_failures = 0
    _provider = provider
    _tool_wait_board = tool_wait_board
    _harness_board = harness_board
    _member = member


def configured() -> dict[str, Any]:
    return {"provider": _provider, "tool_wait_board": _tool_wait_board, "harness_board": _harness_board,
            "member": _member}


async def _board_await(target: Any, method: str, *args: Any) -> None:
    """Call a board method and wait for it, so consecutive calls stay ordered.

    The island boards are Ray actors with ``max_concurrency=64``: back-to-back
    fire-and-forget ``exit``/``enter`` of one trajectory's successive tool
    calls ran out of order ("already in a tool call" / "is not in a tool call",
    A-T3-7, codex-smoke-20261003-11).  Board bookkeeping is telemetry, so its
    errors are logged and never fail the trajectory.
    """
    try:
        value = _board_call(target, method, *args)
        if type(value).__name__ == "ObjectRef":
            await asyncio.to_thread(_board_resolve, value)
    except Exception as exc:  # noqa: BLE001 - never fail a trajectory on telemetry
        print(f"[codex-harness] tool-wait board {method} failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def _board_kwcall(target: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Like ``tool_wait._call`` but forwards keyword arguments (``lease_acquired(deadline=)``)."""
    fn = getattr(target, method)
    remote = getattr(fn, "remote", None)
    return _board_resolve(remote(*args, **kwargs) if callable(remote) else fn(*args, **kwargs))


def resolve_expected_policy_version(metadata: dict[str, Any]) -> str | None:
    """IR-3 target token: prompt metadata first, else the driver token in the sink."""
    try:
        return rollout_meta_hook.expected_policy_version(SimpleNamespace(metadata=metadata))
    except Exception:  # noqa: BLE001 - no sink reachable == no token published
        return None


def scrubbed_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for name in adapter.hmac_key_env_names():
        env.pop(name, None)
    return env


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    for sig, wait in ((signal.SIGTERM, WORKER_GRACE_SECONDS), (signal.SIGKILL, WORKER_GRACE_SECONDS)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        # asyncio.wait (not wait_for): must work inside a cancelled task's finally.
        waiter = asyncio.ensure_future(process.wait())
        done, _pending = await asyncio.wait({waiter}, timeout=wait)
        if done:
            return
        waiter.cancel()


async def _drive_worker(
    job: dict[str, Any], trajectory_id: str, board: Any, worker_env: dict[str, str] | None = None
) -> dict[str, Any]:
    """Spawn the worker, relay tool-wait events, return its result event."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        WORKER_MODULE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL if os.getenv("YETO_CODEX_WORKER_STDERR") != "1" else None,
        env=scrubbed_environment({**os.environ, **{str(k): str(v) for k, v in (worker_env or {}).items()}}),
        start_new_session=True,
    )
    in_tool = False
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write((json.dumps(job) + "\n").encode())
        await process.stdin.drain()
        process.stdin.close()
        while True:
            line = await process.stdout.readline()
            if not line:
                raise adapter.harness.CodexHarnessError("worker exited without a result")
            event = json.loads(line)
            kind = event.get("event")
            if kind == "tool_wait":
                if event.get("phase") == "enter" and not in_tool:
                    in_tool = True
                    if board is not None:
                        await _board_await(board, "enter", trajectory_id)
                elif event.get("phase") == "exit" and in_tool:
                    in_tool = False
                    if board is not None:
                        await _board_await(board, "exit", trajectory_id)
            elif kind == "result":
                return event
            elif kind == "error":
                error = adapter.harness.CodexHarnessError(str(event.get("reason")))
                if isinstance(event.get("metrics"), dict):
                    error.metrics = event["metrics"]  # type: ignore[attr-defined]
                raise error
    finally:
        if in_tool and board is not None:
            await _board_await(board, "exit", trajectory_id)
        await _terminate(process)


class TaskPromptMissing(ValueError):
    """No task statement could be resolved for the episode (fail closed)."""


def _last_user_text(prompt: Any) -> str | None:
    if not isinstance(prompt, list):
        return None
    for message in reversed(prompt):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def task_prompt(metadata: dict[str, Any], prompt: Any, lease: Any) -> str:
    """The task statement Codex receives as its first user message.

    Precedence: explicit ``metadata["prompt"]`` string; the provider task's
    ``instruction.md``; a plain-string Miles prompt; the last user message of a
    chat-format prompt.  A chat prompt without a user message (e.g. only a
    generic system message) is refused instead of being stringified -- S15
    stage 2 sent ``str([{'role': 'system', ...}])`` to every trajectory, so the
    model never saw the task and all 48 rewards were 0.
    """
    explicit = metadata.get("prompt")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    task = getattr(lease, "task", None)
    instruction = getattr(task, "instruction", None)
    if isinstance(instruction, str) and instruction.strip():
        return instruction
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    user = _last_user_text(prompt)
    if user is not None:
        return user
    raise TaskPromptMissing(
        f"no task statement for task {metadata.get('task_id')!r}: metadata has no prompt, "
        "the environment task has no instruction.md and the Miles prompt has no user message"
    )


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any] | None:
    metadata = dict(metadata or {})
    if _provider is None:
        # Rollout workers are separate Ray actors: the driver's configure()
        # never ran here, so build the same wiring from the forwarded env.
        from .preflight import configure_rollout_worker

        configure_rollout_worker()
    if _provider is None:
        raise RuntimeError("codex_openenv_subprocess_agent_function.configure(provider=...) was not called")
    task_id, sample_id = adapter.task_identity(metadata)
    trajectory_id = str(metadata.get("trajectory_id") or sample_id)
    episode_id = adapter.new_episode_id()
    fields = adapter.trajectory_fields(trajectory_id)
    expected_version = resolve_expected_policy_version(metadata)
    if expected_version is None:  # IR-3: refuse before any environment exists
        raise PolicyVersionMissing(f"{POLICY_VERSION_MISSING} for trajectory {trajectory_id!r}")
    board = _harness_board
    if board is not None and not _board_kwcall(board, "allow_new_session", _member):
        # IR-2: admission closed (drain of this member) -> infrastructure, not a reward.
        return {**adapter.infrastructure_metadata(ADMISSION_CLOSED, episode_id=None), **fields}
    lease: EnvironmentLease | None = None
    session_open = lease_open = handed_off = False
    try:
        if board is not None:
            _board_kwcall(board, "enter_session", trajectory_id, _member)
            session_open = True
        try:
            lease = await _provider.acquire(task_id, trajectory_id)
        except Exception as exc:  # noqa: BLE001 - provisioning failures are infrastructure
            _note_acquire_failure(exc, trajectory_id)  # raises EnvironmentProviderOutage past the threshold
            return {**adapter.infrastructure_metadata(f"acquire: {type(exc).__name__}: {exc}", episode_id=None), **fields}
        _note_acquire_success()
        if board is not None:
            _board_kwcall(board, "lease_acquired", trajectory_id, deadline=time.monotonic() + lease.deadline_seconds)
            lease_open = True
        job = {
            "base_url": base_url,
            "prompt": task_prompt(metadata, prompt, lease),
            "request_kwargs": dict(request_kwargs or {}),
            "episode_id": episode_id,
            "max_seq_len": metadata.get("max_seq_len"),
            "env_url": lease.env_url,
            "env_token": lease.env_token,
            "driver": metadata.get("codex_openenv_driver", "stock"),
            **{k: metadata[k] for k in ("script", "final_status", "hang_seconds") if k in metadata},
        }
        try:
            segments = await adapter.prepare_segment_sessions(job)
        except Exception as exc:  # noqa: BLE001 - session-server failures are infrastructure
            return {**adapter.infrastructure_metadata(f"segment sessions: {type(exc).__name__}: {exc}", episode_id=episode_id), **fields}
        try:
            try:
                untrusted = await asyncio.wait_for(
                    _drive_worker(job, trajectory_id, _tool_wait_board, getattr(lease, "worker_env", None)),
                    timeout=lease.deadline_seconds,
                )
            except asyncio.TimeoutError:
                untrusted = {"status": "timeout", "metrics": {"timed_out": 1}, "episode_id": episode_id}
            except adapter.harness.CodexHarnessError as exc:
                metrics = getattr(exc, "metrics", None)
                tito = adapter.mirror_tito_counters(metrics, board)
                handed_off = True
                return {**adapter.infrastructure_metadata(str(exc), episode_id=episode_id, metrics=metrics), **fields, **tito, **segments}
            tito = adapter.mirror_tito_counters(untrusted.get("metrics"), board)
            signed = await adapter.finish_trusted(untrusted, lease.verifier, task_id=task_id, sample_id=sample_id)
            signed["expected_policy_version"] = expected_version
            handed_off = True
            return {**signed, **fields, **tito, **segments}
        finally:
            # Worker crash (any non-harness exception), cancellation or a
            # finish_trusted error: the metadata never reaches
            # codex_openenv_generate, so delete the pre-created sessions here.
            if segments and not handed_off:
                await adapter.release_unreturned_segments(base_url, segments)
    finally:
        try:
            if lease is not None:
                await lease.destroy()
                if await lease.describe() != "gone":
                    # Not confirmed gone: keep the lease on the board; it is
                    # force-released at its hard deadline (leases_expired_total).
                    raise RuntimeError(f"environment for {trajectory_id} was not confirmed destroyed")
                if lease_open:
                    _board_kwcall(board, "lease_released", trajectory_id)
        finally:
            if session_open:
                _board_kwcall(board, "exit_session", trajectory_id)
