"""Machine paths of a sampler spec, filled from the node profile.

A spec is written once for every node, in container paths: ``/work/...`` and
``/models/...`` (a container mounts the node's directories there, the National
Compute route since its first run), plus the placeholders ``__YETO_WORK__``,
``__YETO_MODELS__``, ``__YETO_CACHE__`` and ``__RUN_PREFIX__``.

* a container launch keeps the container paths (the mounts make them true) and
  fills the placeholders with them;
* a process launch (no container: a Daytona sandbox, a venv on an SSH node)
  sees the node's own directories, so ``/work`` and ``/models`` paths move under
  the profile's ``work`` and ``models`` (unchanged when they are those paths
  already, as in a Daytona sandbox).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from yeto.launch.node_inventory import NodeProfile

from .profile import CONTAINER_CACHE, CONTAINER_MODELS, CONTAINER_WORK, is_container, node_dirs

PLACEHOLDERS = ("__YETO_WORK__", "__YETO_MODELS__", "__YETO_CACHE__", "__RUN_PREFIX__")


def sampler_dirs(profile: NodeProfile) -> dict[str, str]:
    """``work`` / ``models`` / ``cache`` as the sampler process sees them."""
    if is_container(profile):
        return {"work": CONTAINER_WORK, "models": CONTAINER_MODELS, "cache": CONTAINER_CACHE}
    return node_dirs(profile)


def _move(path: str, table: tuple[tuple[str, str], ...]) -> str:
    for canonical, real in table:
        if path == canonical or path.startswith(canonical + "/"):
            return real + path[len(canonical):]
    return path


def sampler_path(path: str, profile: NodeProfile) -> str:
    """One container path as the sampler process sees it on this node."""
    if is_container(profile):
        return path
    dirs = node_dirs(profile)
    return _move(path, ((CONTAINER_WORK, dirs["work"]), (CONTAINER_MODELS, dirs["models"])))


def host_path(path: str, profile: NodeProfile) -> str:
    """A path the sampler sees as the node runtime (on the host) sees it."""
    dirs = node_dirs(profile)
    return _move(path, ((CONTAINER_WORK, dirs["work"]), (CONTAINER_MODELS, dirs["models"]),
                        (CONTAINER_CACHE, dirs["cache"])))


def fill_value(value: str, profile: NodeProfile, *, run_prefix: str | None = None) -> str:
    dirs = sampler_dirs(profile)
    text = value.replace("__YETO_WORK__", dirs["work"]).replace("__YETO_MODELS__", dirs["models"]) \
        .replace("__YETO_CACHE__", dirs["cache"])
    if run_prefix is not None:
        text = text.replace("__RUN_PREFIX__", run_prefix.strip("/"))
    return sampler_path(text, profile) if text.startswith("/") else text


def fill_spec(spec: Mapping[str, Any], profile: NodeProfile, *, run_prefix: str | None = None) -> dict[str, Any]:
    """A copy of ``spec`` with every machine path filled for ``profile`` (strings in
    nested maps and lists; keys untouched)."""

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return fill_value(value, profile, run_prefix=run_prefix)
        if isinstance(value, Mapping):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return walk(dict(spec))
