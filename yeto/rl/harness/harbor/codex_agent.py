"""Harbor's codex agent with yeto's model provider (S26 R3); see :mod:`.codex_provider`."""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.codex import Codex

from .codex_provider import DEFAULT_STREAM_IDLE_TIMEOUT_S, provider_setup_command


class YetoCodex(Codex):
    """Codex that talks to ``$OPENAI_BASE_URL`` through the ``yeto`` provider.

    Harbor's ``run`` builds one setup script: auth, ``openai_base_url``
    (top-level), skills, then MCP tables.  The MCP hook is the last point
    that still writes ``config.toml`` and runs before any table, so the
    provider lines go in front of Harbor's MCP lines there.
    """

    def __init__(self, *args: Any, stream_idle_timeout_s: int = DEFAULT_STREAM_IDLE_TIMEOUT_S,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stream_idle_timeout_s = int(stream_idle_timeout_s)

    def _build_register_mcp_servers_command(self) -> str | None:
        mcp = super()._build_register_mcp_servers_command()
        if not self._get_env("OPENAI_BASE_URL"):
            return mcp  # auth.json / default endpoint: keep Harbor's built-in provider
        provider = provider_setup_command(self._stream_idle_timeout_s)
        return provider if mcp is None else f"{provider}\n{mcp}"
