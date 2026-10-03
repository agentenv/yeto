"""Terminal-Bench (OpenEnv) Codex adapter for the ports path.

REWRITE, not a move: the legacy ``codex_openenv_*`` modules lived in a private
``agentenv/miles`` ``examples/experimental/openenv`` copy and were never found
(CODEX-PROGRESS §1.3).  The interface below is the one the yeto consumers
expect (``yeto/rl/learner.py``, ``yeto/rl/tbench_direct_preflight.py``,
``tests/test_rl_codex_schema.py``): ``stock``, ``_OPENENV_IDENTITY_ENV``,
``codex_openenv_harness_identity()`` and a module-level ``run``.

Trust split (design D7):
- ``drive_untrusted`` runs Codex against a terminal environment and never sees
  the reward HMAC key (the worker subprocess asserts that);
- ``finish_trusted`` runs the verifier and signs the ``tbench_outcome``.

The stock driver and Responses bridge are reused unchanged from
``codex_harness_agent`` (moved from ``yeto_miles_secrlenv@5bfc011``).
"""

from __future__ import annotations

import asyncio
import os
import secrets
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from yeto.rl.codex_backend import QWEN35_08B_MODEL, QWEN35_08B_REVISION, stock_codex_backend_profile
from yeto.rl.tbench_outcome import (
    NATIVE_VERIFIER,
    TEST_SH_VERIFIER,
    TIMEOUT_VERIFIER,
    build_signed_metadata,
)

from . import agent as legacy
from . import codex_harness_agent as harness
from .environment import TerminalEnvironment, TrustedVerifier
from .pins import OPENENV_BACKEND_PROFILE

BACKEND_PROFILE_NAME = OPENENV_BACKEND_PROFILE  # "qwen35_08b"
INFRASTRUCTURE_STATUS = "infrastructure"
POLICY_STATUSES = frozenset({"completed", "timeout", "max_turns", "max_seq_len"})
REWARD_SCOPE = "trajectory"

stock = SimpleNamespace(
    _BACKEND_PROFILE=stock_codex_backend_profile(BACKEND_PROFILE_NAME),
    BACKEND_MODEL=stock_codex_backend_profile(BACKEND_PROFILE_NAME)["model"],
    _attest_runtime=harness._attest_runtime,
)
if (
    stock._BACKEND_PROFILE.get("model_identifier") != QWEN35_08B_MODEL
    or stock._BACKEND_PROFILE.get("model_revision") != QWEN35_08B_REVISION
):
    raise RuntimeError("the Codex OpenEnv qwen35_08b profile drifted")


def codex_openenv_harness_identity() -> dict[str, str]:
    """Live identity of the Codex surface this adapter drives (same tools as the stock harness)."""
    live = harness.codex_harness_identity()
    return {key: value for key, value in live.items() if key.endswith("_sha256")}


_OPENENV_IDENTITY_ENV: dict[str, str] = {
    "YETO_CODEX_OPENENV_BACKEND_PROFILE": BACKEND_PROFILE_NAME,
    "YETO_CODEX_OPENENV_MODEL_ID": QWEN35_08B_MODEL,
    "YETO_CODEX_OPENENV_MODEL_REVISION": QWEN35_08B_REVISION,
    **{
        f"YETO_CODEX_OPENENV_{name.upper()}": value
        for name, value in codex_openenv_harness_identity().items()
    },
}


class ToolWaitEmitter:
    """Per-tool-call tool-wait hooks (3.1). ``enter``/``exit`` are sync callables."""

    def __init__(self, enter: Callable[[], None], exit: Callable[[], None]) -> None:
        self.enter = enter
        self.exit = exit


class _ToolWaitEnvironment:
    """Wrap an environment so every tool execution is one tool-wait interval."""

    def __init__(self, env: TerminalEnvironment, emitter: ToolWaitEmitter | None) -> None:
        self._env = env
        self._emitter = emitter

    async def _scoped(self, coro: Awaitable[dict[str, Any]]) -> dict[str, Any]:
        if self._emitter is None:
            return await coro
        self._emitter.enter()
        try:
            return await coro
        finally:
            self._emitter.exit()

    async def execute(self, episode_id: str, command: str, timeout_seconds: float, output_bytes: int):
        return await self._scoped(
            self._env.execute(episode_id, command, timeout_seconds=timeout_seconds, output_bytes=output_bytes)
        )

    async def submit(self, episode_id: str, submission: dict[str, Any]):
        return await self._scoped(self._env.submit(episode_id, submission))


def _max_rollout_seconds() -> float:
    return legacy._positive_env("OPENENV_MAX_ROLLOUT_TIME_SECONDS", 1800.0)


async def drive_untrusted(
    job: dict[str, Any],
    env: TerminalEnvironment,
    *,
    tool_wait: ToolWaitEmitter | None = None,
    binary: Path | None = None,
) -> dict[str, Any]:
    """Untrusted layer: run one Codex episode. Returns an UNSIGNED result.

    ``job`` = {base_url, prompt, request_kwargs, episode_id, max_seq_len?}.
    Policy boundaries (turn/seq/timeout) are reported as statuses; harness
    protocol failures raise ``CodexHarnessError`` and are infrastructure errors.
    """
    metrics = legacy.AgentMetrics()
    scoped_env = _ToolWaitEnvironment(env, tool_wait)
    episode = {"episode_id": job["episode_id"], "prompt": job["prompt"]}
    codex = binary if binary is not None else harness._attest_runtime()
    try:
        status = await asyncio.wait_for(
            harness._drive_codex(
                codex,
                job["base_url"],
                scoped_env,
                episode,
                dict(job.get("request_kwargs") or {}),
                metrics,
                max_seq_len=job.get("max_seq_len"),
            ),
            timeout=_max_rollout_seconds(),
        )
    except asyncio.TimeoutError:
        metrics.timed_out = 1
        status = "timeout"
    except harness.CodexHarnessError as exc:
        # The rejection counters (G6a) must survive the failure: the trusted
        # layer mirrors them onto the HarnessBoard before aborting the sample.
        exc.metrics = _metrics_dict(metrics)  # type: ignore[attr-defined]
        raise
    if status not in POLICY_STATUSES:
        raise harness.CodexHarnessError(f"Codex driver returned unknown status {status!r}")
    return {"status": status, "metrics": _metrics_dict(metrics), "episode_id": job["episode_id"]}


def _metrics_dict(metrics: legacy.AgentMetrics) -> dict[str, Any]:
    return {**asdict(metrics), **harness.tito_counters(metrics)}


async def finish_trusted(
    untrusted: dict[str, Any],
    verifier: TrustedVerifier,
    *,
    task_id: str,
    sample_id: str,
    key: str | bytes | None = None,
) -> dict[str, Any]:
    """Trusted layer: verify, build and sign the outcome (reward in {0,1})."""
    status = untrusted["status"]
    if status not in POLICY_STATUSES:
        raise ValueError(f"untrusted result has invalid status {status!r}")
    if status == "timeout":
        passed, testsh_rc, verifier_name = False, None, TIMEOUT_VERIFIER
    else:
        evaluation = await verifier.evaluate(untrusted["episode_id"])
        passed = bool(evaluation.get("passed"))
        testsh_rc = evaluation.get("testsh_rc")
        verifier_name = (
            TEST_SH_VERIFIER if isinstance(testsh_rc, int) else NATIVE_VERIFIER
        )
    metadata = build_signed_metadata(
        task_id=task_id,
        sample_id=sample_id,
        episode_id=untrusted["episode_id"],
        status=status,
        reward=1.0 if passed else 0.0,
        verifier=verifier_name,
        testsh_rc=testsh_rc if isinstance(testsh_rc, int) else None,
        key=key,
    )
    metadata["agent_metrics"] = dict(untrusted.get("metrics") or {})
    metadata["exit_status"] = status
    return metadata


def infrastructure_metadata(reason: str, *, episode_id: str | None, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unsigned marker: no reward, sample must be ABORTED (never a 0 reward)."""
    return {
        "tbench_infrastructure_error": reason,
        "exit_status": INFRASTRUCTURE_STATUS,
        "episode_id": episode_id,
        "agent_metrics": dict(metrics or {}),
    }


TITO_SESSION_MISMATCH_KEY = harness.TITO_SESSION_MISMATCH_KEY
TITO_CHAIN_BREAKS_KEY = harness.TITO_CHAIN_BREAKS_KEY


def mirror_tito_counters(metrics: dict[str, Any] | None, harness_board: Any) -> dict[str, Any]:
    """G6(a): replay the bridge's rejection counters onto the HarnessBoard.

    Same method names and counter names as ``gateway/core.py`` so the 1.7 load
    sample (``tito_session_mismatch`` / ``tito_chain_breaks``) reads the A path
    too.  Returns the trajectory-field overrides: the bridge holds one chain and
    never forks, so ``chains_total`` stays 1 and ``chain_break_reason`` names the
    first break (the sample is infrastructure-aborted, never trained on).
    """
    from yeto.rl.engine.miles_adapter.rollout_meta_hook import counter_value

    metrics = metrics or {}
    mismatches = counter_value(metrics.get(TITO_SESSION_MISMATCH_KEY))
    breaks = dict(metrics.get(TITO_CHAIN_BREAKS_KEY) or {})
    if harness_board is not None:
        if mismatches:
            _board_call(harness_board, "record_session_mismatch", mismatches)
        for reason, count in breaks.items():
            if count:
                _board_call(harness_board, "record_chain_break", reason, int(count))
    first_break = next((reason for reason, count in breaks.items() if count), None)
    return {"chain_break_reason": first_break}


def _board_call(target: Any, method: str, *args: Any) -> Any:
    fn = getattr(target, method)
    remote = getattr(fn, "remote", None)
    return remote(*args) if callable(remote) else fn(*args)


def trajectory_fields(trajectory_id: str) -> dict[str, Any]:
    """D8 / R-D5a bookkeeping carried on every sample of the trajectory."""
    return {
        "trajectory_id": trajectory_id,
        "chain_index": 0,
        "chains_total": 1,
        "chain_break_reason": None,
        "segment_id": 0,
        "segment_boundary_reason": None,
        "reward_scope": REWARD_SCOPE,
    }


def task_identity(metadata: dict[str, Any]) -> tuple[str, str]:
    task_id = metadata.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Terminal-Bench sample metadata must carry task_id")
    sample_id = metadata.get("sample_id") or metadata.get("trajectory_id") or f"tb-{uuid.uuid4().hex}"
    return task_id, str(sample_id)


def new_episode_id() -> str:
    return f"ep-{secrets.token_hex(8)}"


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    environment: TerminalEnvironment | None = None,
    verifier: TrustedVerifier | None = None,
    tool_wait: ToolWaitEmitter | None = None,
    harness_board: Any = None,
    **_kwargs: Any,
) -> dict[str, Any] | None:
    """In-process entry (both layers in one process; use the subprocess entry in training)."""
    metadata = dict(metadata or {})
    if environment is None or verifier is None:
        raise ValueError("in-process run requires an environment and a verifier")
    task_id, sample_id = task_identity(metadata)
    trajectory_id = metadata.get("trajectory_id") or sample_id
    episode_id = new_episode_id()
    job = {
        "base_url": base_url,
        "prompt": metadata.get("prompt") if isinstance(metadata.get("prompt"), str) else str(prompt),
        "request_kwargs": dict(request_kwargs or {}),
        "episode_id": episode_id,
        "max_seq_len": metadata.get("max_seq_len"),
    }
    try:
        untrusted = await drive_untrusted(job, environment, tool_wait=tool_wait)
    except (harness.CodexHarnessError, legacy.EpisodeClientError, OSError) as exc:
        metrics = getattr(exc, "metrics", None)
        tito = mirror_tito_counters(metrics, harness_board)
        return {
            **infrastructure_metadata(f"{type(exc).__name__}: {exc}", episode_id=episode_id, metrics=metrics),
            **trajectory_fields(trajectory_id),
            **tito,
        }
    tito = mirror_tito_counters(untrusted.get("metrics"), harness_board)
    signed = await finish_trusted(untrusted, verifier, task_id=task_id, sample_id=sample_id)
    return {**signed, **trajectory_fields(trajectory_id), **tito}


def hmac_key_env_names() -> tuple[str, ...]:
    return ("TBENCH_REWARD_HMAC_KEY", "TBENCH_REWARD_HMAC_KEY_FILE", "SECRLENV_REWARD_HMAC_KEY")


def assert_no_reward_key(env: dict[str, str] | None = None) -> None:
    present = [name for name in hmac_key_env_names() if (env if env is not None else os.environ).get(name)]
    if present:
        raise RuntimeError(f"reward HMAC key leaked into the untrusted layer: {', '.join(present)}")
