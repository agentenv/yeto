"""Ports-path preflight for the Codex Terminal-Bench harness (design D3, R-CTX, R-D5a).

Runs before model allocation (IR-1 hook) and fails closed on:
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

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

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
