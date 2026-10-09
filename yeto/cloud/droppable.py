"""Phase 2 of rl-spot-cost-saving: one on-demand anchor island + droppable spot islands.

User decision 2026-10-09 (tasks 5.x, design D8). ``--rl-island-role ISLAND:ROLE``
(repeatable, ROLE = anchor | droppable). An island without a role is an anchor.

* The anchor island runs on-demand. It keeps the latest weights (its contributions
  move the syncer base) and the cut point (its checkpoint store, when one is given).
* A droppable island runs on spot. It never resumes from a checkpoint store. When it
  comes back after a reclaim it starts fresh, JOINs with catch-up and takes the
  current syncer base (``ElasticAvgSync.start``: optimizer reset); the data cursor
  comes from the syncer version (0.21 fresh-machine rule). So it syncs from the
  line of weights the anchor keeps current, never from an archive.
* If every droppable island is gone at once, the anchor alone still closes rounds:
  admission requires ``--rl-q-min`` <= number of anchors.

Admission (task 5.1, spec "推理型岛的定义"): elastic only (legacy refused), at least
one anchor, no anchor on a cloud whose GPUs are always preemptible (Modal), no global
``--spot`` (billing comes from the role), a droppable island only on a cloud whose
spot support is known in the capability table, and an anchor on Verda must not keep its cut on a local
volume (Verda docs do not say what happens to block volumes on eviction).

Imported by the launcher AND by the island; it must not import the launcher.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Callable, Mapping

from yeto.cloud import capabilities

ROLE_ENV = "YETO_ISLAND_ROLE"   # JSON {"role", "cloud", "region"} on every island of a role run
ANCHOR, DROPPABLE = "anchor", "droppable"
ROLES = (ANCHOR, DROPPABLE)
LEGACY_REFUSED = "可丢弃岛只在 elastic 下支持 (droppable islands need --rl-island-scheduling elastic)"


def parse_roles(items, num_islands: int) -> dict[int, str]:
    """``["1:droppable", ...]`` -> ``{0: "anchor", 1: "droppable", ...}`` (every island)."""
    out: dict[int, str] = {}
    for item in items or ():
        island_s, sep, role = str(item).partition(":")
        role = role.strip()
        if not sep or not island_s.strip().isdigit():
            raise ValueError(f"--rl-island-role {item!r}: expected ISLAND:ROLE")
        island = int(island_s)
        if not 0 <= island < num_islands:
            raise ValueError(f"--rl-island-role {item!r}: island {island} out of range "
                             f"(this launch has islands 0..{num_islands - 1})")
        if role not in ROLES:
            raise ValueError(f"--rl-island-role {item!r}: role must be one of {ROLES}")
        if island in out:
            raise ValueError(f"--rl-island-role: island {island} given twice")
        out[island] = role
    return {i: out.get(i, ANCHOR) for i in range(num_islands)}


def _is_local_store(store: str) -> bool:
    return bool(store) and "://" not in store


def admit(args, clouds: list[str]) -> dict[int, str]:
    """Roles of a launch (``{}`` without ``--rl-island-role``). Raises ValueError
    before any cloud work when a rule is broken. ``clouds[i]`` is island i's cloud."""
    items = getattr(args, "rl_island_role", None) or []
    if not items:
        return {}
    if getattr(args, "training_mode", "sft") != "rl":
        raise ValueError("--rl-island-role needs --training-mode rl")
    mode = getattr(args, "rl_island_scheduling", None) or "legacy"
    if mode != "elastic":
        raise ValueError(LEGACY_REFUSED)
    roles = parse_roles(items, len(clouds))
    anchors = [i for i, r in roles.items() if r == ANCHOR]
    if not anchors:
        raise ValueError("--rl-island-role: at least one anchor island (on-demand) is required")
    if getattr(args, "spot", False):
        raise ValueError("--rl-island-role with --spot: billing comes from the role "
                         "(anchor on-demand, droppable spot); drop --spot")
    q_min = getattr(args, "rl_q_min", None)
    q_min = 1 if q_min is None else int(q_min)
    if q_min > len(anchors):
        raise ValueError(f"--rl-q-min {q_min} > {len(anchors)} anchor island(s): rounds would stop "
                         "when every droppable island is reclaimed at once")
    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    for i, role in roles.items():
        cloud = clouds[i]
        spot_value = capabilities.capability(cloud, "spot").value
        if role == ANCHOR:
            if spot_value == "always_preemptible_gpu":
                raise ValueError(f"--rl-island-role: island {i} on {cloud} cannot be the anchor "
                                 "(its GPUs are always preemptible)")
            if cloud == "verda" and _is_local_store(store):
                raise ValueError(f"--rl-island-role: anchor island {i} on Verda keeps its cut on a "
                                 f"local path ({store}); Verda docs do not say what eviction does to "
                                 "block volumes, use a bucket URI")
        elif spot_value is None:
            raise ValueError(f"--rl-island-role: droppable island {i} on {cloud}: spot support "
                             "unknown in yeto/cloud/capabilities.json")
    return roles


def billing(role: str) -> str:
    return "spot" if role == DROPPABLE else "on_demand"


def rejoin_source(role: str) -> str:
    """Where an island's state comes from when it starts again."""
    return "syncer_base_from_anchor" if role == DROPPABLE else "checkpoint_store"


def role_args(args, role: str | None):
    """``args`` itself without a role; else a copy with the role's billing, and a
    droppable island loses the checkpoint store (no resume from an archive)."""
    if role is None:
        return args
    copied = copy.copy(args)
    copied.spot = role == DROPPABLE
    copied._island_role = role  # read by island_keeps_spot_volume
    if role == DROPPABLE:
        copied.rl_checkpoint_store = None
    return copied


def island_keeps_spot_volume(args) -> bool:
    """The legacy ``--spot`` per-island volume (completed groups, re-synced on a
    rebuild) is a resume from an archive; a droppable island must not get it."""
    return bool(getattr(args, "spot", False)) and getattr(args, "_island_role", None) != DROPPABLE


def role_env(role: str | None, cloud: str, region: str | None) -> dict[str, str]:
    if role is None:
        return {}
    return {ROLE_ENV: json.dumps({"role": role, "cloud": cloud, "region": region},
                                 sort_keys=True, separators=(",", ":"))}


def env_role(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    raw = (os.environ if environ is None else environ).get(ROLE_ENV)
    if not raw:
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) and doc.get("role") in ROLES else None


def manifest_fields(roles: dict[int, str]) -> dict[str, Any]:
    if not roles:
        return {}
    return {"island_roles": {str(i): {"role": r, "billing": billing(r), "rejoin": rejoin_source(r)}
                             for i, r in sorted(roles.items())}}


def install_reclaim_listener(island: str, *, leave: Callable[[], Any], emit: Callable[..., Any],
                             environ: Mapping[str, str] | None = None,
                             start: bool = True) -> Any | None:
    """On a droppable island: listen for a reclaim notice and LEAVE. Nothing to save
    (the island loses only its own round increment), so ``save=None``. Modal: signal
    handler; AWS: metadata poll; other clouds: no notice (lease expiry path).
    Returns the handler/poller, or None when not installed."""
    doc = env_role(environ)
    if not doc or doc["role"] != DROPPABLE:
        return None
    from yeto.cloud import preemption as p

    cloud, region = doc.get("cloud"), doc.get("region")
    if cloud == "modal":
        h = p.ModalExitHandler(island, save=None, leave=leave, emit=emit, last_save_s=None,
                               region=region, role=DROPPABLE)
        if start:
            h.install()
        return h
    if cloud == "aws":
        def on_notice(notice):
            return p.handle_notice(notice, save=None, leave=leave, emit=emit, last_save_s=None)

        poller = p.AwsMetadataPoller(island, on_notice, region=region, role=DROPPABLE)
        if start:
            poller.start()
        return poller
    return None
