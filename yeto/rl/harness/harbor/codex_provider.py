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

The provider also sets its retry counts explicitly (S27 G-EF, G-EB section
7.2): ``request_max_retries`` (a failed HTTP request, e.g. a 502 or 503 from
the session server) and ``stream_max_retries`` (a stream cut before it
completed).  Codex's own defaults are small (4 and 5 by its configuration
documentation; not read from the binary), and a model call that fails after them ends the Codex process, so the whole
trial is ABORTED and its group dropped.  Codex backs off exponentially from
200 ms, so 8 retries cover about 50 s of short outages.  Both keys are in the
pinned 0.145.0 binary (``strings``, 10-25).

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

REQUEST_MAX_RETRIES_ENV = "YETO_CODEX_REQUEST_MAX_RETRIES"
STREAM_MAX_RETRIES_ENV = "YETO_CODEX_STREAM_MAX_RETRIES"
DEFAULT_MAX_RETRIES = 8
MAX_RETRIES_CAP = 100  # Codex clamps both counts to 100
REQUEST_MAX_RETRIES_KWARG = "request_max_retries"
STREAM_MAX_RETRIES_KWARG = "stream_max_retries"


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


def max_retries_from_env(name: str, environ: Mapping[str, str] | None = None) -> int:
    """A Codex retry count from ``name`` (``YETO_CODEX_{REQUEST,STREAM}_MAX_RETRIES``,
    default 8): an integer between 0 and 100."""
    environ = os.environ if environ is None else environ
    raw = environ.get(name, "").strip() or str(DEFAULT_MAX_RETRIES)
    try:
        value = int(raw)
    except ValueError as exc:
        raise HarnessConfigError(f"bad {name}: {raw!r}") from exc
    if not 0 <= value <= MAX_RETRIES_CAP:
        raise HarnessConfigError(f"{name} must be between 0 and {MAX_RETRIES_CAP}")
    return value


def provider_toml(stream_idle_timeout_s: int, *, request_max_retries: int = DEFAULT_MAX_RETRIES,
                  stream_max_retries: int = DEFAULT_MAX_RETRIES) -> str:
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
        f"request_max_retries = {int(request_max_retries)}",
        f"stream_max_retries = {int(stream_max_retries)}",
        "",
    ))


def provider_setup_command(stream_idle_timeout_s: int, *, request_max_retries: int = DEFAULT_MAX_RETRIES,
                           stream_max_retries: int = DEFAULT_MAX_RETRIES) -> str:
    """Shell command that appends :func:`provider_toml` to ``$CODEX_HOME/config.toml``."""
    toml = provider_toml(stream_idle_timeout_s, request_max_retries=request_max_retries,
                         stream_max_retries=stream_max_retries)
    return f'cat >>"$CODEX_HOME/config.toml" <<TOML\n{toml}TOML'


def use_yeto_codex(agent_config: object, environ: Mapping[str, str] | None = None) -> None:
    """Point a Harbor ``AgentConfig`` for ``codex`` at :data:`CODEX_AGENT_IMPORT_PATH`.

    Both backends call it (verl via ``build_trial_config``, Miles via the
    example builder).  The timeout and the retry counts come from the
    builder's environment and ride in the agent kwargs, so the trial config
    records them.
    """
    agent_config.name = None
    agent_config.import_path = CODEX_AGENT_IMPORT_PATH
    agent_config.kwargs.setdefault(STREAM_IDLE_TIMEOUT_KWARG, stream_idle_timeout_from_env(environ))
    agent_config.kwargs.setdefault(REQUEST_MAX_RETRIES_KWARG,
                                   max_retries_from_env(REQUEST_MAX_RETRIES_ENV, environ))
    agent_config.kwargs.setdefault(STREAM_MAX_RETRIES_KWARG,
                                   max_retries_from_env(STREAM_MAX_RETRIES_ENV, environ))
