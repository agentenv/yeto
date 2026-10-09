"""SecRLEnv agent-metadata capture (neutral part of the old generate wrapper).

The custom agent (``agent.py`` / ``codex_harness_agent.py``) records its signed
result here; the Miles generate wrapper that restores it after Miles' aborted
fallback moved to ``yeto/rl/adapters/miles/harness_glue/codex_generate.py``
(yeto-framework-decoupling 5.7).
"""

from __future__ import annotations

import copy
from contextvars import ContextVar
from typing import Any

_CAPTURED_AGENT_METADATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "secrlenv_agent_metadata", default=None
)


def capture_agent_metadata(metadata: dict[str, Any]) -> None:
    """Retain one agent result inside its generation task only."""

    _CAPTURED_AGENT_METADATA.set(copy.deepcopy(metadata))
