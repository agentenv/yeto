"""Task-backend and island-engine registries.

Yeto's sync core (Rust syncer, fragment layout, wire protocol, fleet
orchestration in :mod:`yeto.launcher`) is task-agnostic: it merges fragments
from learner islands and knows nothing about *what* those islands train. A
**task backend** is the adapter that teaches yeto how to train one kind of
model — the CLI surface it needs, how to build its learner task, how to
export its checkpoint. ``lm`` (text LMs) and ``diffusion`` (component-backed
diffusion models) are built-ins; another backend registers by adding a module
that calls :func:`register_backend`, touching no central ``if task == ...``
dispatch.

Within the ``lm`` task the *island engine* is a second, orthogonal axis: the
intra-island trainer (``torch`` FSDP/DDP vs ``megatron`` expert parallelism).
Both speak the same DiLoCo adapter sync, so they differ only in entrypoint,
extra flags, setup deps, and image. Engines register the same way, via
:func:`register_engine`, so a new engine is also a drop-in.

Both registries are deliberately thin: a backend/engine is a small object with
a ``name`` and a handful of hook methods. Heavy imports (torch, sky,
transformers, task pipelines) stay lazy *inside* the hooks so importing this
package — which the CLI does at parser-build time — never pulls the training
stack.
"""

from __future__ import annotations

import argparse
from typing import Iterable

# ---------------------------------------------------------------------------
# Island engines (intra-island trainer; an lm-task concern today)


class IslandEngine:
    """The per-island trainer for a learner cluster.

    Subclasses describe how one training engine turns shared learner flags
    into a concrete torchrun invocation and its setup dependencies. Hooks
    receive the parsed launch ``args`` and the island's :class:`ClusterSpec`.
    """

    #: value accepted by ``--island-backend``
    name: str = ""

    def entrypoint(self) -> str:
        """``python -m <entrypoint>`` run under torchrun on each island."""
        raise NotImplementedError

    def extra_learner_flags(self, args, spec) -> str:
        """Engine-specific flags appended to the shared learner flags."""
        return ""

    def setup_steps(self, args) -> list[str]:
        """Ordered shell setup commands for a learner node (before prefetch)."""
        raise NotImplementedError

    def image(self, args, spec):
        """Machine image this engine pins, or ``None`` to fall through to the
        explicit ``--learner-image`` / GPU-override table / provider default."""
        return None


_ENGINES: dict[str, IslandEngine] = {}


def register_engine(engine: IslandEngine) -> IslandEngine:
    if not engine.name:
        raise ValueError("island engine must set a non-empty name")
    _ENGINES[engine.name] = engine
    return engine


def get_engine(name: str) -> IslandEngine:
    try:
        return _ENGINES[name]
    except KeyError:
        raise ValueError(
            f"unknown island backend {name!r}; known: {', '.join(engine_names())}"
        ) from None


def engine_names() -> list[str]:
    return list(_ENGINES)


# ---------------------------------------------------------------------------
# Task backends


class TaskBackend:
    """Adapter for one training task (``--task`` value).

    A backend owns its slice of the CLI (a launch arg group and an export arg
    group), the validation of those flags, and the three task-specific pieces
    of orchestration: building the learner task, warning when the fleet can't
    hold the model, and exporting a syncer checkpoint. Everything else — fleet
    planning, provisioning, recovery, delivery — is shared and lives in the
    generic core.
    """

    #: value accepted by ``--task`` (and the default when it is the first
    #: backend registered)
    name: str = ""

    #: whether ``yeto shape`` / ``--budget`` / ``--flops`` auto-fleet planning
    #: applies to this task (LM-only today; other tasks require explicit --gpu)
    supports_auto_fleet: bool = False

    #: directory the learner writes its artifact to (fetched by the head)
    output_dir: str = "yeto-output"

    # -- CLI ----------------------------------------------------------------

    def add_launch_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Register this backend's ``launch`` flags (typically one arg group)."""

    def add_export_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Register this backend's ``yeto.export`` flags."""

    # -- validation ---------------------------------------------------------

    def validate(self, args) -> list[str]:
        """Hard errors for the given launch args (empty list = ok)."""
        return []

    def warnings(self, args) -> list[str]:
        """Non-fatal warnings for the given launch args."""
        return []

    def normalize_args(self, args) -> None:
        """Fill backend defaults after the full command line is parsed."""

    # -- orchestration hooks ------------------------------------------------

    def build_learner_task(self, args, spec, learner_id: int, num_learners: int, syncer_addr: str):
        """Return the ``sky.Task`` for one learner island."""
        raise NotImplementedError

    def image_override(self, args, spec):
        """Image this task/engine pins, or ``None``. Consulted after an
        explicit ``--learner-image`` and before the GPU-override table."""
        return None

    def warn_if_wont_fit(self, args, specs) -> None:
        """Print capacity warnings for the planned fleet (best-effort)."""

    def head_file_mounts(self, args) -> dict:
        """Extra file mounts the head VM needs (e.g. a local task checkout)."""
        return {}

    def rewrite_for_head(self, args) -> None:
        """Rewrite submitter-local paths in ``args`` to their head-VM location.

        Runs on the head before learner tasks are built: a checkout the
        submitting CLI mounted onto the head must be referenced from the
        head's path, not the submitter's.
        """

    # -- export -------------------------------------------------------------

    def export(self, args) -> None:
        """Export a syncer checkpoint into a usable artifact directory."""
        raise NotImplementedError


_BACKENDS: dict[str, TaskBackend] = {}
_DEFAULT: list[str] = []


def register_backend(backend: TaskBackend, *, default: bool = False) -> TaskBackend:
    if not backend.name:
        raise ValueError("task backend must set a non-empty name")
    _BACKENDS[backend.name] = backend
    if default or not _DEFAULT:
        _DEFAULT[:] = [backend.name]
    return backend


def get_backend(name: str | None) -> TaskBackend:
    key = name or default_task()
    try:
        return _BACKENDS[key]
    except KeyError:
        raise ValueError(
            f"unknown task {key!r}; known: {', '.join(backend_names())}"
        ) from None


def backend_names() -> list[str]:
    return list(_BACKENDS)


def all_backends() -> Iterable[TaskBackend]:
    return list(_BACKENDS.values())


def default_task() -> str:
    return _DEFAULT[0] if _DEFAULT else ""
