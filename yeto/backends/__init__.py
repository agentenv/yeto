"""Task-backend registry: the plug-in seam for what yeto can fine-tune.

Importing this package registers the built-in backends (``lm``, ``nava``) and
their island engines (``torch``, ``megatron``). A new task or engine is added
by dropping a module here that calls :func:`register_backend` /
:func:`register_engine` and importing it below — no central dispatch to edit.

The public surface is the lookup API; the CLI, launcher, and export modules
route every task-specific decision through it.
"""

from __future__ import annotations

from .base import (
    IslandEngine,
    TaskBackend,
    all_backends,
    backend_names,
    default_task,
    engine_names,
    get_backend,
    get_engine,
    register_backend,
    register_engine,
)

# Import for side effects: each module registers its backend/engines on import.
# NAVA is a self-contained task package (yeto/nava/); its adapter registers here.
from . import lm as _lm  # noqa: E402,F401
from ..nava import backend as _nava  # noqa: E402,F401

__all__ = [
    "IslandEngine",
    "TaskBackend",
    "all_backends",
    "backend_names",
    "default_task",
    "engine_names",
    "get_backend",
    "get_engine",
    "register_backend",
    "register_engine",
]
