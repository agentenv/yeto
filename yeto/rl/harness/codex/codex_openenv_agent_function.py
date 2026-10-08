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
from typing import Any, Awaitable, Callable, Mapping

from yeto.rl.codex_backend import (
    QWEN35_08B_MODEL,
    QWEN35_08B_REVISION,
    QWEN38_NEXT_4LAYER_MODEL,
    QWEN38_NEXT_4LAYER_REVISION,
    QWEN38_NEXT_MODEL,
    QWEN38_NEXT_REVISION,
    stock_codex_backend_profile,
)
from yeto.rl.tbench_outcome import (
    NATIVE_VERIFIER,
    TEST_SH_VERIFIER,
    TIMEOUT_VERIFIER,
    build_signed_metadata,
)

from . import agent as legacy
from . import codex_harness_agent as harness
from . import compaction_bridge
from .environment import TerminalEnvironment, TrustedVerifier
from .pins import OPENENV_BACKEND_PROFILE, OPENENV_BACKEND_PROFILES

# rl-fn-codex-rollout 1.0: the backend profile is no longer pinned to the
# image default.  Two distinct notions:
# * IMAGE_BACKEND_PROFILE_NAME -- the build-time default recorded in the image
#   env (``YETO_CODEX_OPENENV_BACKEND_PROFILE`` / ``_MODEL_ID`` / ``_MODEL_REVISION``);
# * BACKEND_PROFILE_NAME -- the runtime profile declared by the launch
#   (``--codex-backend-profile``), which the learner publishes to the container
#   as ``YETO_CODEX_CHAT_TEMPLATE`` (``stock_codex_backend_contract(profile)["chat_template"]``,
#   the same declaration ``codex_harness_agent`` reads) and ``validate_stock_codex_fields``
#   already checks against ``--model`` / ``--model-revision``.
# Why no image rebuild: the adapter's tool surface (``*_SHA256`` pins) does not
# depend on the model, so the image record only has to stay a faithful record of
# the build; the model identity of the runtime profile is derived here from the
# ``yeto.rl.codex_backend`` allowlist (``profile_identity``) instead of being
# compared against the image env.  Boundary: the runtime profile must belong to
# ``pins.OPENENV_BACKEND_PROFILES`` and declare the exact HF identity listed in
# ``_PROFILE_IDENTITY``; anything else is refused at import / preflight.
IMAGE_BACKEND_PROFILE_NAME = OPENENV_BACKEND_PROFILE  # "qwen35_08b"
RUNTIME_PROFILE_ENV = "YETO_CODEX_CHAT_TEMPLATE"
_PROFILE_IDENTITY: dict[str, tuple[str, str]] = {
    "qwen35_08b": (QWEN35_08B_MODEL, QWEN35_08B_REVISION),
    "qwen38_next": (QWEN38_NEXT_MODEL, QWEN38_NEXT_REVISION),
    "qwen38_next_4layer": (QWEN38_NEXT_4LAYER_MODEL, QWEN38_NEXT_4LAYER_REVISION),
}
assert set(_PROFILE_IDENTITY) == set(OPENENV_BACKEND_PROFILES)
INFRASTRUCTURE_STATUS = "infrastructure"
POLICY_STATUSES = frozenset({"completed", "timeout", "max_turns", "max_seq_len"})
REWARD_SCOPE = "trajectory"


def profile_identity(profile_name: str) -> tuple[str, str]:
    """``(model_identifier, model_revision)`` the adapter derives for one runtime profile.

    Raises ``ValueError`` ("requires backend profile") for a profile outside
    ``OPENENV_BACKEND_PROFILES`` and ``RuntimeError`` ("profile drifted") when
    the allowlisted ``codex_backend`` profile no longer declares the identity
    pinned here (the legacy module-level qwen35_08b assertion, per profile).
    """
    expected = _PROFILE_IDENTITY.get(profile_name)
    if expected is None:
        raise ValueError(
            "the Codex OpenEnv adapter requires backend profile in "
            + ", ".join(OPENENV_BACKEND_PROFILES) + f" (got {profile_name!r})"
        )
    profile = stock_codex_backend_profile(profile_name)
    if (profile.get("model_identifier"), profile.get("model_revision")) != expected:
        raise RuntimeError(f"the Codex OpenEnv {profile_name} profile drifted")
    return expected


def resolve_backend_profile(name: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """The runtime backend profile: explicit ``name``, else the launch declaration
    ``YETO_CODEX_CHAT_TEMPLATE`` when it names an allowlisted profile (for every
    member of ``OPENENV_BACKEND_PROFILES`` the chat template equals the profile
    name), else the image default.  An explicit name is validated."""
    if name is not None:
        profile_identity(name)
        return name
    env = os.environ if env is None else env
    declared = env.get(RUNTIME_PROFILE_ENV)
    if declared in _PROFILE_IDENTITY:
        return declared
    return IMAGE_BACKEND_PROFILE_NAME


BACKEND_PROFILE_NAME = resolve_backend_profile()
profile_identity(BACKEND_PROFILE_NAME)  # import-time drift check, per selected profile

stock = SimpleNamespace(
    _BACKEND_PROFILE=stock_codex_backend_profile(BACKEND_PROFILE_NAME),
    BACKEND_MODEL=stock_codex_backend_profile(BACKEND_PROFILE_NAME)["model"],
    _attest_runtime=harness._attest_runtime,
)


def codex_openenv_harness_identity() -> dict[str, str]:
    """Live identity of the Codex surface this adapter drives (same tools as the stock harness)."""
    live = harness.codex_harness_identity()
    return {key: value for key, value in live.items() if key.endswith("_sha256")}


# Image record (what the image line bakes into the container env and
# ``yeto.rl.CODEX_OPENENV_IDENTITY_ENV`` mirrors): the *build-time* profile and
# its HF identity plus the profile-independent tool-surface hashes.  It is
# deliberately NOT derived from BACKEND_PROFILE_NAME so the container env drift
# check stays a check of the image, whichever runtime profile the launch declares.
_OPENENV_IDENTITY_ENV: dict[str, str] = {
    "YETO_CODEX_OPENENV_BACKEND_PROFILE": IMAGE_BACKEND_PROFILE_NAME,
    "YETO_CODEX_OPENENV_MODEL_ID": _PROFILE_IDENTITY[IMAGE_BACKEND_PROFILE_NAME][0],
    "YETO_CODEX_OPENENV_MODEL_REVISION": _PROFILE_IDENTITY[IMAGE_BACKEND_PROFILE_NAME][1],
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
    segment_urls = job.get(compaction_bridge.SEGMENT_URLS_KEY)
    if compaction_bridge.compactionrl_enabled():
        if not isinstance(segment_urls, list) or not all(isinstance(u, str) for u in segment_urls):
            raise harness.CodexHarnessError("CompactionRL job has no pre-created segment sessions")
        driving = compaction_bridge.drive_codex_compactionrl(
            codex,
            job["base_url"],
            scoped_env,
            episode,
            dict(job.get("request_kwargs") or {}),
            metrics,
            max_seq_len=job.get("max_seq_len"),
            segment_base_urls=segment_urls,
        )
    elif segment_urls is not None:
        raise harness.CodexHarnessError("segment sessions supplied without CompactionRL")
    else:
        driving = harness._drive_codex(
            codex,
            job["base_url"],
            scoped_env,
            episode,
            dict(job.get("request_kwargs") or {}),
            metrics,
            max_seq_len=job.get("max_seq_len"),
        )
    try:
        status = await asyncio.wait_for(driving, timeout=_max_rollout_seconds())
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
    # compaction_counters is {} unless the CompactionRL bridge ran.
    extra: dict[str, Any] = {}
    # Observe only (S15 r2 follow-up): why the Codex episode ended and the shape
    # of the last model reply; set by the stock driver, absent otherwise.
    for name in ("end_reason", "last_completion"):
        value = getattr(metrics, name, None)
        if value is not None:
            extra[name] = value
    return {**asdict(metrics), **harness.tito_counters(metrics), **compaction_bridge.compaction_counters(metrics), **extra}


# --- CompactionRL segment sessions (trusted side; design D8) ----------------
_SESSION_FIELDS = ("temperature", "top_p", "top_k")


def split_session_url(base_url: str) -> tuple[str, str]:
    """``{router}/sessions/{id}`` -> (router, id)."""
    head, sep, session_id = base_url.rstrip("/").rpartition("/sessions/")
    if not sep or not head or not session_id or "/" in session_id:
        raise ValueError(f"not a session-server session URL: {base_url!r}")
    return head, session_id


def segment_session_body(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    """``CreateSessionRequest`` body with the rollout's own sampling defaults."""
    body: dict[str, Any] = {"evaluation": False}
    for key in _SESSION_FIELDS:
        value = request_kwargs.get(key)
        if value is not None:
            body[key] = int(value) if key == "top_k" else float(value)
    return body


async def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async with session.post(url, json=body) as response:
            if response.status != 200:
                raise RuntimeError(f"session server returned HTTP {response.status}")
            return await response.json()


async def create_segment_sessions(
    base_url: str,
    request_kwargs: dict[str, Any],
    count: int,
    *,
    post: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    delete: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[list[str], list[str]]:
    """Pre-create one session per allowed compaction; returns (ids, agent base URLs).

    Created here, in the trusted layer, so the untrusted worker can only use
    sessions this rollout owns; ``codex_openenv_generate`` collects (and so
    deletes) every listed session, used or not.
    """
    router, _ = split_session_url(base_url)
    body = segment_session_body(request_kwargs)
    ids: list[str] = []
    try:
        for _ in range(count):
            reply = await (post or _post_json)(f"{router}/sessions", dict(body))
            session_id = reply.get("session_id") if isinstance(reply, dict) else None
            if not isinstance(session_id, str) or not session_id or "/" in session_id:
                raise RuntimeError("session server returned no session_id")
            ids.append(session_id)
    except BaseException:
        # Pre-creation failed half-way: nobody will collect the ones already made.
        await delete_segment_sessions(base_url, ids, delete=delete)
        raise
    return ids, [f"{router}/sessions/{session_id}" for session_id in ids]


async def _delete_session(url: str) -> None:
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        async with session.delete(url) as response:
            # Pinned session server (miles/rollout/session/sessions.py): 204 on
            # success; an already collected/deleted session is a 404.
            if response.status not in (200, 204, 404):
                raise RuntimeError(f"session server returned HTTP {response.status}")


async def delete_segment_sessions(
    base_url: str,
    session_ids: Any,
    *,
    delete: Callable[[str], Awaitable[None]] | None = None,
) -> list[str]:
    """Best-effort ``DELETE {router}/sessions/{id}`` for every id; returns the failures.

    Used when pre-created segment sessions will not reach
    ``codex_openenv_generate`` (which otherwise collects and so deletes them).
    Never raises: a cleanup failure must not mask the original error.
    """
    if not isinstance(session_ids, (list, tuple)) or not session_ids:
        return []
    try:
        router, _ = split_session_url(base_url)
    except ValueError:
        return [str(session_id) for session_id in session_ids]
    failed: list[str] = []
    for session_id in session_ids:
        try:
            await (delete or _delete_session)(f"{router}/sessions/{session_id}")
        except Exception:  # noqa: BLE001 - keep deleting the rest
            failed.append(str(session_id))
    return failed


async def release_unreturned_segments(
    base_url: str,
    segments: dict[str, Any],
    *,
    delete: Callable[[str], Awaitable[None]] | None = None,
) -> list[str]:
    """Delete the sessions listed in ``segments`` (the trusted metadata from
    :func:`prepare_segment_sessions`) when it is not being returned."""
    return await delete_segment_sessions(
        base_url, segments.get(compaction_bridge.SESSIONS_METADATA_KEY), delete=delete
    )


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
    evaluation: dict[str, Any] = {}
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
    if status != "timeout" and isinstance(evaluation.get("log"), str):
        metadata["verifier_log"] = evaluation["log"]  # observe only, unsigned
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
    from yeto.rl.adapters.miles.rollout_meta_hook import counter_value

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
    """Call a board method and wait for it (A-T3-7 follow-up, S14/A19).

    The island boards are Ray actors with ``max_concurrency=64``: a
    fire-and-forget ``record_session_mismatch`` / ``record_chain_break`` may
    land after the round's ``snapshot()`` (counter missing from the rollout it
    belongs to) and any error it raises is an unhandled actor error instead of
    this trajectory's. Resolving the ObjectRef here keeps the calls in issue
    order, like the tool-wait board's enter/exit. A failure of the remote
    bookkeeping is logged and never fails the trajectory; a local board's own
    exception (e.g. an unknown chain-break reason) propagates as before."""
    import sys

    from yeto.rl.engine.tool_wait import _resolve

    fn = getattr(target, method)
    remote = getattr(fn, "remote", None)
    if not callable(remote):
        return fn(*args)
    try:
        return _resolve(remote(*args))
    except Exception as exc:  # noqa: BLE001 - never fail a trajectory on telemetry
        print(f"[codex-harness] harness board {method} failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


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
        segments = await prepare_segment_sessions(job)
    except Exception as exc:  # noqa: BLE001 - session-server failures are infrastructure
        return {
            **infrastructure_metadata(f"segment sessions: {type(exc).__name__}: {exc}", episode_id=episode_id),
            **trajectory_fields(trajectory_id),
        }
    handed_off = False  # True once ``segments`` rides in the returned metadata
    try:
        try:
            untrusted = await drive_untrusted(job, environment, tool_wait=tool_wait)
        except (harness.CodexHarnessError, legacy.EpisodeClientError, OSError) as exc:
            metrics = getattr(exc, "metrics", None)
            tito = mirror_tito_counters(metrics, harness_board)
            handed_off = True
            return {
                **infrastructure_metadata(f"{type(exc).__name__}: {exc}", episode_id=episode_id, metrics=metrics),
                **trajectory_fields(trajectory_id),
                **tito,
                **segments,
            }
        tito = mirror_tito_counters(untrusted.get("metrics"), harness_board)
        signed = await finish_trusted(untrusted, verifier, task_id=task_id, sample_id=sample_id)
        handed_off = True
        return {**signed, **trajectory_fields(trajectory_id), **tito, **segments}
    finally:
        if segments and not handed_off:
            await release_unreturned_segments(job["base_url"], segments)


async def prepare_segment_sessions(
    job: dict[str, Any],
    *,
    post: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    delete: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """CompactionRL only: add segment URLs to ``job``; return the trusted metadata.

    Returns ``{}`` (and leaves ``job`` untouched) when CompactionRL is off.
    """
    if not compaction_bridge.compactionrl_enabled():
        return {}
    if job.get("max_seq_len") is None:
        raise harness.CodexHarnessError("CompactionRL requires max_seq_len")
    cfg = compaction_bridge.compaction_config(int(job["max_seq_len"]))
    ids, urls = await create_segment_sessions(
        job["base_url"], dict(job.get("request_kwargs") or {}), cfg.max_compactions, post=post, delete=delete
    )
    job[compaction_bridge.SEGMENT_URLS_KEY] = urls
    return {compaction_bridge.SESSIONS_METADATA_KEY: ids}


def hmac_key_env_names() -> tuple[str, ...]:
    return ("TBENCH_REWARD_HMAC_KEY", "TBENCH_REWARD_HMAC_KEY_FILE", "SECRLENV_REWARD_HMAC_KEY")


def assert_no_reward_key(env: dict[str, str] | None = None) -> None:
    present = [name for name in hmac_key_env_names() if (env if env is not None else os.environ).get(name)]
    if present:
        raise RuntimeError(f"reward HMAC key leaked into the untrusted layer: {', '.join(present)}")
