"""Launcher internals, split by responsibility.

``yeto.launcher`` keeps the entry points (``run``, ``dry_run_plan``) and the
orchestration; the modules here build what it launches. Dependencies point one
way: nothing in this package imports ``yeto.launcher``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
