"""Ports-path preflight for the Codex Terminal-Bench harness (design D3, R-CTX, R-D5a).

``harness_preflight(miles_args, launch)`` is the IR-1 hook (INFRA signature
``entry.HarnessPreflight = Callable[[miles_args, launch], None]``): ports
``entry.preflight_stage`` calls it before ``connect_island_ray`` / placement /
model allocation, resolved from ``miles_args.yeto_harness_preflight`` or
``YETO_HARNESS_PREFLIGHT`` (= ``HARNESS_PREFLIGHT_SPEC``).  It fails closed on:
- ``reward_scope`` other than ``trajectory`` (task 7.1; IR-1
  ``config.check_harness_reward_scope`` on ``miles_args.yeto_harness_reward_scope``);
- an agent function other than this package's ``CODEX_OPENENV_AGENT``;
- no environment provider (``miles_args.yeto_harness_environment_provider`` or
  ``YETO_HARNESS_ENVIRONMENT_PROVIDER`` = ``module:callable`` returning an
  ``EnvironmentProvider``), then installs it together with the island's
  ``ToolWaitBoard`` / ``HarnessBoard`` actors (same names ``entry.harness_source`` uses);
and, through ``preflight_codex_openenv``, on:
- legacy trainable compaction being enabled: the fork pin's session server does
  not know the ``X-Miles-Compaction-*`` headers and would silently roll back
  window 0 (R-D5a), so ``YETO_CODEX_COMPACTION_ENABLED`` must be unset/false;
- the CPU-only scripted worker driver being allowed in a training process;
- Codex binary / version / identity drift (``codex_harness_agent._attest_runtime``:
  signed env, binary sha256 and size, ``codex --version`` == 0.145.0);
- OpenEnv identity env drift against ``codex_openenv_agent_function._OPENENV_IDENTITY_ENV``;
- reward HMAC key source problems (``tbench_outcome.validate_hmac_key_source``).
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from yeto.rl.engine.miles_adapter.config import check_harness_reward_scope
from yeto.rl.tbench_outcome import validate_hmac_key_source

from . import codex_harness_agent as harness
from . import codex_openenv_agent_function as adapter
from .codex_openenv_agent_worker import SCRIPTED_DRIVER_ENV

COMPACTION_ENV = (
    "YETO_CODEX_COMPACTION_ENABLED",
    "YETO_CODEX_COMPACTION_TRIGGER_TOKENS",
    "YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS",
    "YETO_CODEX_MAX_COMPACTIONS",
)
_FALSE = frozenset({"", "0", "false", "no", "off"})


class PreflightError(RuntimeError):
    pass


def assert_compaction_disabled(env: Mapping[str, str]) -> None:
    if env.get("YETO_CODEX_COMPACTION_ENABLED", "").strip().lower() not in _FALSE:
        raise PreflightError(
            "YETO_CODEX_COMPACTION_ENABLED is set: legacy trainable compaction is "
            "incompatible with the fork pin session server (R-D5a) and must stay off"
        )
    extra = [name for name in COMPACTION_ENV[1:] if env.get(name)]
    if extra:
        raise PreflightError("compaction tuning without support: " + ", ".join(extra))


def assert_no_scripted_driver(env: Mapping[str, str]) -> None:
    if env.get(SCRIPTED_DRIVER_ENV):
        raise PreflightError(f"{SCRIPTED_DRIVER_ENV} must not be set in a training process")


def assert_openenv_identity(env: Mapping[str, str]) -> None:
    drift = [name for name, value in adapter._OPENENV_IDENTITY_ENV.items() if env.get(name) != value]
    if drift:
        raise PreflightError("Codex OpenEnv identity drifted: " + ", ".join(drift))


def preflight_codex_openenv(
    env: Mapping[str, str] | None = None,
    *,
    attest_runtime: Callable[[], Path] = harness._attest_runtime,
    validate_key: Callable[[], None] = validate_hmac_key_source,
) -> dict[str, Any]:
    """Return a small report on success; raise ``PreflightError`` otherwise."""
    env = os.environ if env is None else env
    assert_compaction_disabled(env)
    assert_no_scripted_driver(env)
    assert_openenv_identity(env)
    try:
        binary = attest_runtime()
    except harness.CodexHarnessError as exc:
        raise PreflightError(f"Codex runtime attestation failed: {exc}") from exc
    try:
        validate_key()
    except Exception as exc:  # noqa: BLE001 - any key-source problem is fatal
        raise PreflightError(f"reward HMAC key source rejected: {exc}") from exc
    return {
        "codex_binary": str(binary),
        "codex_version": harness.CODEX_CLI_VERSION,
        "identity": adapter.codex_openenv_harness_identity(),
        "compaction": "disabled",
    }


# ---------------------------------------------------------------------------
# IR-1 hook: (miles_args, launch) -> None, run by entry.preflight_stage

HARNESS_PREFLIGHT_SPEC = "yeto.rl.harness.codex.preflight:harness_preflight"
ENVIRONMENT_PROVIDER_ENV = "YETO_HARNESS_ENVIRONMENT_PROVIDER"
EXPECTED_AGENT_FUNCTION = "yeto.rl.harness.codex.codex_openenv_subprocess_agent_function.run"


def _load_dotted(spec: Any, what: str) -> Any:
    if callable(spec):
        return spec
    module, sep, name = str(spec).partition(":")
    if not sep:
        module, _, name = module.rpartition(".")
    if not module or not name:
        raise PreflightError(f"{what} {spec!r} is not module:callable")
    try:
        target = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise PreflightError(f"{what} {spec!r} cannot be imported: {exc}") from exc
    if not callable(target):
        raise PreflightError(f"{what} {spec!r} is not callable")
    return target


def resolve_environment_provider(miles_args: Any, env: Mapping[str, str]) -> Any:
    """The sandbox provider factory for this island (fail closed when absent)."""
    spec = getattr(miles_args, "yeto_harness_environment_provider", None) or env.get(ENVIRONMENT_PROVIDER_ENV)
    if not spec:
        raise PreflightError(
            f"no environment provider: set miles_args.yeto_harness_environment_provider or {ENVIRONMENT_PROVIDER_ENV}"
        )
    factory = _load_dotted(spec, "environment provider")
    provider = factory(miles_args) if not hasattr(factory, "acquire") else factory
    if not hasattr(provider, "acquire"):
        raise PreflightError(f"environment provider {spec!r} has no acquire()")
    return provider


def island_boards(miles_args: Any) -> tuple[Any, Any]:
    """``(tool_wait_board, harness_board)``: the island's named actors, looked up lazily."""
    from yeto.rl.engine.miles_adapter.elastic_wiring import LazyBoardActor
    from yeto.rl.engine.tool_wait import harness_board_actor

    learner_id = int(getattr(miles_args, "yeto_rl_learner_id", 0) or 0)
    return LazyBoardActor(learner_id), LazyBoardActor(learner_id, factory=harness_board_actor)


def harness_preflight(miles_args: Any, launch: Any, *, env: Mapping[str, str] | None = None) -> None:
    """IR-1 hook body. Raises ``PreflightError``/``MilesConfigError`` before any allocation."""
    del launch  # identity / binary / key checks do not depend on the launch args
    env = os.environ if env is None else env
    # 7.1: a per-segment reward scope must not start (IR-1 path, same check as validate_parsed_args).
    check_harness_reward_scope(getattr(miles_args, "yeto_harness_reward_scope", None))
    agent = getattr(miles_args, "custom_agent_function_path", None)
    if agent != EXPECTED_AGENT_FUNCTION:
        raise PreflightError(f"custom_agent_function_path={agent!r}; the Codex harness preflight expects {EXPECTED_AGENT_FUNCTION}")
    preflight_codex_openenv(env)
    provider = resolve_environment_provider(miles_args, env)
    from . import codex_openenv_subprocess_agent_function as subprocess_agent

    tool_wait_board, harness_board = island_boards(miles_args)
    subprocess_agent.configure(
        provider=provider,
        tool_wait_board=tool_wait_board,
        harness_board=harness_board,
        member=resolve_member(miles_args, env),
    )


MEMBER_CELL_ENV = "YETO_RL_CELL_ID"
LEARNER_ID_ENV = "YETO_RL_LEARNER_ID"
# Harness env the island driver forwards into Ray's job runtime_env so the
# rollout workers (which inherit the raylet's environment, not the driver's)
# can self-configure (A-T3-4, codex-smoke-20261003-6).
WORKER_PASSTHROUGH_ENV = (
    ENVIRONMENT_PROVIDER_ENV,
    "TBENCH_REWARD_HMAC_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "OPENENV_RUN_ID",
    "SECRLENV_MAX_TURNS",
    MEMBER_CELL_ENV,
    LEARNER_ID_ENV,
)
WORKER_PASSTHROUGH_ENV_PREFIXES = ("YETO_HARNESS_TB2_", "YETO_CODEX_")


def worker_runtime_env(miles_args: Any, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Env vars every Ray actor of this island must see for the agent function to run."""
    env = os.environ if env is None else env
    forwarded = {
        key: value
        for key, value in env.items()
        if value and (key in WORKER_PASSTHROUGH_ENV or key.startswith(WORKER_PASSTHROUGH_ENV_PREFIXES))
    }
    learner_id = getattr(miles_args, "yeto_rl_learner_id", None)
    if learner_id is not None and str(learner_id) != "":
        forwarded[LEARNER_ID_ENV] = str(int(learner_id))
    cell = getattr(miles_args, "yeto_rl_cell_id", None)
    if cell is not None and str(cell) != "":
        forwarded[MEMBER_CELL_ENV] = str(cell)
    return forwarded


def configure_rollout_worker(env: Mapping[str, str] | None = None) -> bool:
    """Install the provider and boards in a rollout worker from its environment.

    ``harness_preflight`` configures the driver process only; upstream Miles
    calls the agent function inside ``RolloutExecutor`` actors, where the
    module globals start empty.  Returns False (configures nothing) when the
    environment names no provider, so ``run`` keeps failing closed.
    """
    env = os.environ if env is None else env
    if not env.get(ENVIRONMENT_PROVIDER_ENV):
        return False
    miles_args = SimpleNamespace(
        yeto_harness_environment_provider=None,
        yeto_rl_learner_id=int(env.get(LEARNER_ID_ENV) or 0),
        yeto_rl_member_id=None,
        yeto_rl_cell_id=None,
    )
    provider = resolve_environment_provider(miles_args, env)
    from . import codex_openenv_subprocess_agent_function as subprocess_agent

    tool_wait_board, harness_board = island_boards(miles_args)
    subprocess_agent.configure(
        provider=provider,
        tool_wait_board=tool_wait_board,
        harness_board=harness_board,
        member=resolve_member(miles_args, env),
    )
    return True


def resolve_member(miles_args: Any, env: Mapping[str, str] | None = None) -> str | None:
    """The INFRA rollout member key this island's sessions are admitted under.

    INFRA keys members as ``rollout.member_id(cell_id)`` (= ``engine:<cell_id>``,
    ``MilesRolloutPool.members`` / ``close_admission``).  Sources, in order:
    ``miles_args.yeto_rl_member_id`` (a full member key, or a bare cell id),
    ``miles_args.yeto_rl_cell_id`` / ``YETO_RL_CELL_ID`` (a cell id).  None
    (no source) keeps the global admission key, which single-island smoke
    runs rely on.
    """
    from yeto.rl.engine.miles_adapter.rollout import MEMBER_PREFIX, member_id

    env = os.environ if env is None else env
    member = getattr(miles_args, "yeto_rl_member_id", None)
    if member is not None and str(member) != "":
        member = str(member)
        return member if member.startswith(MEMBER_PREFIX) else member_id(member)
    cell = getattr(miles_args, "yeto_rl_cell_id", None)
    if cell is None or str(cell) == "":
        cell = env.get(MEMBER_CELL_ENV) or None
    return member_id(cell) if cell is not None else None


# ---------------------------------------------------------------------------
# Task 2.2: legacy preflight forwarding (injectable; legacy modules are not edited here)
#
# ``yeto/rl/learner.py::_preflight_codex_openenv_adapter`` and
# ``yeto/rl/tbench_direct_preflight.py::_attest_adapter`` currently import the
# adapter from ``<miles_root>/examples/experimental/openenv`` and compare it
# against ``yeto.rl.CODEX_OPENENV_IDENTITY_ENV`` (image-line pins).  Once the
# image line repoints those pins (see ``required_pin_updates``), the legacy
# functions become one-liners calling ``forward_legacy_openenv_preflight``.

def forward_legacy_openenv_preflight(args: Any, profile_name: str, env: Mapping[str, str] | None = None) -> None:
    """Drop-in body for ``learner._preflight_codex_openenv_adapter(args, profile)``.

    Same failure classes/messages as legacy so existing callers keep their
    behaviour: ``ValueError`` with "requires backend profile", "identity drifted",
    "environment drifted".
    """
    del args  # the adapter no longer lives under miles_root
    if profile_name != adapter.BACKEND_PROFILE_NAME:
        raise ValueError("the Codex OpenEnv adapter requires backend profile qwen35_08b")
    live = adapter.codex_openenv_harness_identity()
    expected = {
        name.removeprefix("YETO_CODEX_OPENENV_").lower(): value
        for name, value in adapter._OPENENV_IDENTITY_ENV.items()
        if name.endswith("_SHA256")
    }
    if live != expected:
        raise ValueError("the Codex OpenEnv surface identity drifted")
    env = os.environ if env is None else env
    mismatched = [name for name, value in adapter._OPENENV_IDENTITY_ENV.items() if env.get(name) != value]
    if mismatched:
        raise ValueError("Codex OpenEnv container environment drifted: " + ", ".join(mismatched))


def required_pin_updates() -> dict[str, Any]:
    """What ``yeto/rl/__init__.py`` (image line) must change to point at this package."""
    import hashlib

    here = Path(__file__).resolve().parent
    sha = lambda name: hashlib.sha256((here / name).read_bytes()).hexdigest()  # noqa: E731
    return {
        "SECRLENV_AGENT_PATH": "yeto/rl/harness/codex/agent.py",
        "SECRLENV_AGENT_SHA256": sha("agent.py"),
        "SECRLENV_AGENT": "yeto.rl.harness.codex.agent.run",
        "SECRLENV_REWARD": "yeto.rl.harness.codex.reward:reward_func",
        "SECRLENV_GROUP_FILTER": "yeto.rl.harness.codex.reward.check_group",
        "SECRLENV_GENERATE": "yeto.rl.harness.codex.generate.generate",
        "SECRLENV_GENERATE_SHA256": sha("generate.py"),
        "CODEX_HARNESS_AGENT": "yeto.rl.harness.codex.codex_harness_agent.run",
        "CODEX_HARNESS_AGENT_PATH": "yeto/rl/harness/codex/codex_harness_agent.py",
        "CODEX_HARNESS_AGENT_SHA256": sha("codex_harness_agent.py"),
        "CODEX_OPENENV_AGENT": "yeto.rl.harness.codex.codex_openenv_subprocess_agent_function.run",
        "CODEX_OPENENV_AGENT_MODULES": (
            "codex_openenv_subprocess_agent_function.py", "codex_openenv_agent_worker.py", "codex_openenv_agent_function.py",
        ),
        "CODEX_OPENENV_IDENTITY_ENV": dict(adapter._OPENENV_IDENTITY_ENV),
    }
