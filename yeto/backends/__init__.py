"""Task-backend registry: the plug-in seam for what yeto can fine-tune.

Built-ins register lightweight adapters only. Heavy runtime imports (torch,
transformers, diffusion pipelines) stay lazy inside backend/component hooks so
CLI startup and LM-only runs are not affected by optional diffusion deps.
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

# Import for side effects: each module registers its backend/engines.
from . import lm as _lm  # noqa: E402,F401
from . import diffusion as _diffusion  # noqa: E402,F401

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
