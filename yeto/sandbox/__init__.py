"""Task environments on sandbox hosts (``yeto sandbox``; ``agentic.envs``)."""

import os
from pathlib import Path

STATE_DIR_ENV = "YETO_SANDBOX_DIR"  # moves the local state root (tests point it at a temporary directory)


def state_root() -> Path:
    """The local state root of ``yeto sandbox`` (``$YETO_SANDBOX_DIR``, default ``~/.yeto/sandbox``)."""
    return Path(os.environ.get(STATE_DIR_ENV) or "~/.yeto/sandbox").expanduser()
