"""yeto's own Codex model provider (S26 R3).

The gate relay holds a Codex request while the turn gate is closed and
writes ``: held`` SSE comments meanwhile.  Codex does not count comments as
stream activity: after ``stream_idle_timeout_ms`` (default 300 s) it reports
"idle timeout waiting for SSE" and reconnects.  Harbor's codex agent only
writes ``openai_base_url``, which keeps the built-in ``openai`` provider and
its 300 s idle timeout.

yeto therefore defines a provider ``yeto`` in ``$CODEX_HOME/config.toml``
(same base URL and key variable as before, ``wire_api = "responses"``) with
an idle timeout longer than the longest hold, and selects it with
``model_provider``.  Key names are checked against Codex ``rust-v0.145.0``
(``codex-rs/model-provider-info/src/lib.rs``: ``base_url``, ``env_key``,
``wire_api``, ``stream_idle_timeout_ms``; ``"chat"`` is rejected there).

This module has no Harbor import; :mod:`.codex_agent` holds the agent class.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from yeto.rl.agentic import HarnessConfigError

PROVIDER_ID = "yeto"
STREAM_IDLE_TIMEOUT_ENV = "YETO_CODEX_STREAM_IDLE_TIMEOUT_S"
DEFAULT_STREAM_IDLE_TIMEOUT_S = 1800

# Harbor resolves this instead of its built-in ``codex`` name; the class
# keeps ``name() == "codex"``, so results and stats are unchanged.
CODEX_AGENT_IMPORT_PATH = "yeto.rl.harness.harbor.codex_agent:YetoCodex"
STREAM_IDLE_TIMEOUT_KWARG = "stream_idle_timeout_s"


def stream_idle_timeout_from_env(environ: Mapping[str, str] | None = None) -> int:
    """``YETO_CODEX_STREAM_IDLE_TIMEOUT_S`` (default 1800): Codex SSE idle timeout in seconds."""
    environ = os.environ if environ is None else environ
    raw = environ.get(STREAM_IDLE_TIMEOUT_ENV, "").strip() or str(DEFAULT_STREAM_IDLE_TIMEOUT_S)
    try:
        value = int(raw)
    except ValueError as exc:
        raise HarnessConfigError(f"bad {STREAM_IDLE_TIMEOUT_ENV}: {raw!r}") from exc
    if value <= 0:
        raise HarnessConfigError(f"{STREAM_IDLE_TIMEOUT_ENV} must be > 0")
    return value


def provider_toml(stream_idle_timeout_s: int) -> str:
    """config.toml lines that define and select the ``yeto`` provider.

    ``base_url`` is the shell expansion of ``$OPENAI_BASE_URL``; the caller
    writes these lines through an unquoted heredoc, as Harbor does for
    ``openai_base_url``.  ``model_provider`` is a top-level key, so these
    lines must come before any other table in the file.
    """
    return "\n".join((
        f'model_provider = "{PROVIDER_ID}"',
        "",
        f"[model_providers.{PROVIDER_ID}]",
        f'name = "{PROVIDER_ID}"',
        'base_url = "${OPENAI_BASE_URL}"',
        'env_key = "OPENAI_API_KEY"',
        'wire_api = "responses"',
        f"stream_idle_timeout_ms = {int(stream_idle_timeout_s) * 1000}",
        "",
    ))


def provider_setup_command(stream_idle_timeout_s: int) -> str:
    """Shell command that appends :func:`provider_toml` to ``$CODEX_HOME/config.toml``."""
    return f'cat >>"$CODEX_HOME/config.toml" <<TOML\n{provider_toml(stream_idle_timeout_s)}TOML'


def use_yeto_codex(agent_config: object, environ: Mapping[str, str] | None = None) -> None:
    """Point a Harbor ``AgentConfig`` for ``codex`` at :data:`CODEX_AGENT_IMPORT_PATH`.

    Both backends call it (verl via ``build_trial_config``, Miles via the
    example builder).  The timeout comes from the builder's environment and
    rides in the agent kwargs, so the trial config records it.
    """
    agent_config.name = None
    agent_config.import_path = CODEX_AGENT_IMPORT_PATH
    agent_config.kwargs.setdefault(STREAM_IDLE_TIMEOUT_KWARG, stream_idle_timeout_from_env(environ))
