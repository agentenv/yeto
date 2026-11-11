"""yeto's configuration of the pi coding agent (badlogic/pi-mono, npm
``@earendil-works/pi-coding-agent``) behind the records proxy.

Harbor's built-in ``pi`` agent runs ``pi --print --mode json --provider <p>
--model <m>`` and sets no base URL. pi reads custom endpoints from
``<agent-dir>/models.json`` (default agent dir ``~/.pi/agent``), so yeto
writes two files there before Harbor's command runs:

* ``models.json``: provider ``yeto`` (``api = "openai-completions"``, the
  proxy's ``/chat/completions`` entry) with one model, the gateway's model
  name. ``baseUrl`` is the shell expansion of ``$OPENAI_BASE_URL``;
  ``apiKey`` stays the literal ``$OPENAI_API_KEY``, which pi resolves itself.
  The compat flags keep the request to fields the gateway accepts: no
  ``store``, no ``developer`` role, no ``reasoning_effort``, and ``max_tokens``
  instead of ``max_completion_tokens``.
* ``settings.json``: no compaction (a compacted history breaks the token
  chain), a long HTTP idle timeout (the proxy answers a call in one SSE chunk
  at its end, and the gate relay may hold a call), agent-level retries, no
  project trust (a task directory never loads pi extensions) and no install
  telemetry.

``max_tokens``: pi sends ``max_tokens`` on every call, at most the model's
``maxTokens`` and less when its context estimate leaves less room. The gateway
signs ``max_tokens`` (a request must repeat the signed value), so ``maxTokens``
is the signed per-turn value and ``contextWindow`` is set far above any real
window, so pi never lowers it; the engine's own context limit still applies.

Field names are checked against pi-mono ``packages/coding-agent/docs``
(``models-configuration.md``, ``settings.md``) at the pinned version.

This module has no Harbor import; :mod:`.pi_agent` holds the agent class.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from yeto.rl.agentic import HarnessConfigError

PROVIDER_ID = "yeto"
# Harbor resolves this instead of its built-in ``pi``; the class keeps ``name() == "pi"``.
PI_AGENT_IMPORT_PATH = "yeto.rl.harness.harbor.pi_agent:YetoPi"

AGENT_DIR = "${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}"
# Far above any real window: pi's own clamp of max_tokens never fires (module doc).
UNBOUNDED_CONTEXT_WINDOW = 1 << 30
DEFAULT_HTTP_IDLE_TIMEOUT_S = 1800
DEFAULT_MAX_RETRIES = 8

# The pi process: no catalog refresh, no version request, no telemetry.
PI_PROCESS_ENV: Mapping[str, str] = {"PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}

MAX_TOKENS_KWARG = "max_tokens"


def models_json(model: str, max_tokens: int, *, context_window: int = UNBOUNDED_CONTEXT_WINDOW) -> dict[str, Any]:
    """``models.json`` with the ``yeto`` provider; ``baseUrl`` is a shell expansion (module doc)."""
    if int(max_tokens) <= 0:
        raise HarnessConfigError("pi max_tokens must be > 0")
    return {"providers": {PROVIDER_ID: {
        "baseUrl": "${OPENAI_BASE_URL}",
        "api": "openai-completions",
        "apiKey": "$OPENAI_API_KEY",
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False,
                   "maxTokensField": "max_tokens"},
        "models": [{"id": model, "reasoning": False, "input": ["text"], "contextWindow": int(context_window),
                    "maxTokens": int(max_tokens)}],
    }}}


def settings_json(*, http_idle_timeout_s: int = DEFAULT_HTTP_IDLE_TIMEOUT_S,
                  max_retries: int = DEFAULT_MAX_RETRIES) -> dict[str, Any]:
    timeout_ms = int(http_idle_timeout_s) * 1000
    return {"defaultProjectTrust": "never", "compaction": {"enabled": False}, "httpIdleTimeoutMs": timeout_ms,
            "retry": {"enabled": True, "maxRetries": int(max_retries), "provider": {"timeoutMs": timeout_ms}},
            "enableInstallTelemetry": False, "quietStartup": True}


def setup_command(model: str, max_tokens: int, *, context_window: int = UNBOUNDED_CONTEXT_WINDOW,
                  http_idle_timeout_s: int = DEFAULT_HTTP_IDLE_TIMEOUT_S,
                  max_retries: int = DEFAULT_MAX_RETRIES) -> str:
    """Shell command that writes both files into pi's agent directory.

    ``models.json`` goes through an unquoted heredoc (``${OPENAI_BASE_URL}``
    expands; ``$OPENAI_API_KEY`` is escaped and stays literal for pi);
    ``settings.json`` through a quoted one.
    """
    models = json.dumps(models_json(model, max_tokens, context_window=context_window), indent=1)
    if "`" in models or "\\" in models:
        raise HarnessConfigError(f"pi model name {model!r} cannot go into a shell heredoc")
    models = models.replace("$OPENAI_API_KEY", "\\$OPENAI_API_KEY")
    settings = json.dumps(settings_json(http_idle_timeout_s=http_idle_timeout_s, max_retries=max_retries), indent=1)
    return (f'mkdir -p "{AGENT_DIR}" && cat >"{AGENT_DIR}/models.json" <<JSON\n{models}\nJSON\n'
            f"cat >\"{AGENT_DIR}/settings.json\" <<'JSON'\n{settings}\nJSON")


def use_yeto_pi(agent_config: Any, *, max_tokens: int | None) -> None:
    """Point a Harbor ``AgentConfig`` for ``pi`` at :data:`PI_AGENT_IMPORT_PATH`.

    ``max_tokens`` is the records proxy's signed per-turn value; without it
    every pi call would be refused, so a missing value is a configuration error.
    """
    if max_tokens is None:
        raise HarnessConfigError("agent pi needs the signed per-turn max_tokens (HarborSettings.turn_max_tokens)")
    agent_config.name = None
    agent_config.import_path = PI_AGENT_IMPORT_PATH
    agent_config.kwargs.setdefault(MAX_TOKENS_KWARG, int(max_tokens))
