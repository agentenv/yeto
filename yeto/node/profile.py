"""The node runtime's view of a node profile (``yeto.launch.node_inventory.NodeProfile``).

The profile itself (device family, card architecture, container kind, node
paths, model source, reach, derived device flags) lives in
:mod:`yeto.launch.node_inventory`. This module adds what the runtime needs on
top of it:

* :func:`profile_doc` / :func:`profile_from_doc`: the profile as the JSON an
  assignment carries, and back (no inventory validation: a rented node has no
  SSH host or host memory figure);
* :func:`node_dirs`: the node's ``work`` / ``models`` / ``cache`` directories, with
  the defaults of the routes (``/work``, ``/models``, ``<work>/cache``);
* the fixed container paths (:data:`CONTAINER_WORK`, :data:`CONTAINER_MODELS`,
  :data:`CONTAINER_CACHE`): a container mounts the node's directories there, so
  one sampler spec written in those paths serves every node.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from yeto.launch.node_inventory import NodePaths, NodeProfile, Reach

CONTAINER_WORK = "/work"
CONTAINER_MODELS = "/models"
CONTAINER_CACHE = "/root/.cache"
CONTAINER_ENGINES = ("docker", "podman")


class ProfileError(ValueError):
    """A profile the node runtime cannot start a sampler on."""


def is_container(profile: NodeProfile) -> bool:
    return profile.container in CONTAINER_ENGINES


def node_dirs(profile: NodeProfile) -> dict[str, str]:
    """``work`` / ``models`` / ``cache``: absolute paths on the node (the host of a container)."""
    work = (profile.paths.work or CONTAINER_WORK).rstrip("/") or "/"
    models = (profile.paths.models or CONTAINER_MODELS).rstrip("/") or "/"
    cache = (profile.paths.cache or f"{work}/cache").rstrip("/")
    for key, value in (("work", work), ("models", models), ("cache", cache)):
        if not value.startswith("/"):
            raise ProfileError(f"node {profile.name}: {key} must be an absolute path, got {value!r}")
    return {"work": work, "models": models, "cache": cache}


def container_mount_list(profile: NodeProfile) -> list[tuple[str, str]]:
    """``(host directory, container path)`` of a container launch, one per container path:
    the node's three directories at ``/work``, ``/models``, ``/root/.cache``, with the
    profile's own ``mounts`` on top (an inventory's shared mounts, a route's per-model
    mounts); a profile mount at one of the three container paths replaces that one. One
    host directory may appear at several container paths."""
    dirs = node_dirs(profile)
    inside_to_host = {CONTAINER_WORK: dirs["work"], CONTAINER_MODELS: dirs["models"], CONTAINER_CACHE: dirs["cache"]}
    for host, inside in (profile.mounts or {}).items():
        inside_to_host[str(inside)] = str(host)
    return sorted(((host, inside) for inside, host in inside_to_host.items()), key=lambda m: m[1])


def container_mounts(profile: NodeProfile) -> dict[str, str]:
    """:func:`container_mount_list` as host directory -> container path, for a launcher
    that takes a map; :class:`ProfileError` when one host directory sits at two container
    paths (a map would drop one of them)."""
    out: dict[str, str] = {}
    for host, inside in container_mount_list(profile):
        if host in out:
            raise ProfileError(f"node {profile.name}: {host} is mounted at both {out[host]} and {inside}: give "
                               "each container path its own host directory")
        out[host] = inside
    return out


# inventory fields the node runtime never reads: left out, so the assignment stays what it
# was (and a node runtime of an older code pack reads it)
_OMIT = frozenset({"fabric_group"})


def profile_doc(profile: NodeProfile) -> dict[str, Any]:
    """The profile as JSON (every field; ``paths`` and ``reach`` as maps; ``node_health``
    only when the inventory declares it)."""
    doc: dict[str, Any] = {}
    for f in fields(profile):
        value = getattr(profile, f.name)
        if f.name in _OMIT or (f.name == "node_health" and value is None):
            continue
        if isinstance(value, NodePaths):
            value = value.record()
        elif isinstance(value, Reach):
            value = {"mode": value.mode, "via": value.via}
        elif isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, Mapping):
            value = dict(value)
        doc[f.name] = value
    return doc


_FIELDS = frozenset(f.name for f in fields(NodeProfile))


def profile_from_doc(doc: Mapping[str, Any], **overrides: Any) -> NodeProfile:
    """A profile from :func:`profile_doc`'s JSON (or a hand-written subset: ``name``,
    ``device_family``, ``card``, ``cards_per_node`` and ``hbm_gib`` are needed);
    ``overrides``: fields that replace the document's (a node's own paths)."""
    values = {**dict(doc), **{k: v for k, v in overrides.items() if v is not None}}
    unknown = sorted(set(values) - _FIELDS)
    if unknown:
        raise ProfileError(f"node profile: unknown field(s) {unknown}")
    missing = [k for k in ("name", "device_family", "card", "cards_per_node", "hbm_gib") if values.get(k) is None]
    if missing:
        raise ProfileError(f"node profile: {missing} required")
    values.setdefault("host", "")
    values.setdefault("cpu_arch", "x86_64")
    values.setdefault("host_memory_gib", 0.0)
    if isinstance(values.get("paths"), Mapping):
        values["paths"] = NodePaths(**dict(values["paths"]))
    if isinstance(values.get("reach"), Mapping):
        values["reach"] = Reach(**dict(values["reach"]))
    elif isinstance(values.get("reach"), str):
        values["reach"] = Reach(values["reach"])
    for key in ("docker_args", "roles"):
        if key in values and values[key] is not None:
            values[key] = tuple(values[key])
    for key in ("mounts", "env"):
        if key in values and values[key] is not None:
            values[key] = dict(values[key])
    try:
        return NodeProfile(**values)
    except (TypeError, ValueError) as error:
        raise ProfileError(f"node profile: {error}") from None
