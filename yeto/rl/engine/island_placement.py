"""Per-island placement (rl-mixed-island-placement).

``--rl-island-placement ISLAND:KEY=VALUE`` (repeatable) gives one island its
own placement values; every other island keeps the run-wide flags. Each island
resolves its values in this order: the per-island value, then the run-wide
flag, then the flag default.

========================  ===============================  =================
key                       run-wide flag (args dest)        values
========================  ===============================  =================
``placement``             ``--rl-placement``               colocated | fixed-partition
``standby-gpus``          ``--rl-standby-gpus``            integer >= 0
``use-gpus-per-node``     ``--rl-island-use-gpus-per-node``  integer >= 1
``tensor-parallel``       ``--tensor-parallel``            integer >= 1
``expert-parallel``       ``--expert-parallel``            integer >= 1
``pipeline-parallel``     ``--pipeline-parallel``          integer >= 1
``max-tokens-per-gpu``    ``--rl-max-tokens-per-gpu``      integer >= 1
========================  ===============================  =================

The four parallel-shape keys need ``--rl-sync-layout shape-free``: only a
shape-free full-parameter layout lets islands of different TP/EP/PP shapes
share one syncer session (``yeto.rl.engine.shape_free_layout``).

Mixed placement (islands with different placements) is allowed only in an
island pool (any island pool option, e.g. ``--rl-sync-island-lease-s``): a fixed
roster is not part of this change and is refused. A
colocated island has no dedicated rollout or standby GPUs, so in a mixed run it
drops the run-wide ``--rl-rollout-gpus`` / ``--rl-standby-gpus`` it would
otherwise inherit; an explicit per-island ``standby-gpus`` above 0 on a
colocated island is refused.

The launcher builds every island-side value (resource shape, learner command,
algorithm capability check) from :func:`island_args`, a copy of the launch
arguments with this island's values. This module is pure: no launcher import,
no cloud work.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

COLOCATED = "colocated"
FIXED_PARTITION = "fixed-partition"
PLACEMENTS = (COLOCATED, FIXED_PARTITION)

OVERRIDES_DEST = "rl_island_placement"


def _placement(value: str) -> str:
    if value not in PLACEMENTS:
        raise ValueError(f"placement must be one of {list(PLACEMENTS)}, got {value!r}")
    return value


def _count(minimum: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise ValueError(f"must be an integer, got {value!r}") from None
        if number < minimum:
            raise ValueError(f"must be >= {minimum}, got {number}")
        return number

    return parse


# override key -> (args dest of the run-wide flag, value parser)
KEYS = {
    "placement": ("rl_placement", _placement),
    "standby-gpus": ("rl_standby_gpus", _count(0)),
    "use-gpus-per-node": ("rl_island_use_gpus_per_node", _count(1)),
    "tensor-parallel": ("tensor_parallel", _count(1)),
    "expert-parallel": ("expert_parallel", _count(1)),
    "pipeline-parallel": ("pipeline_parallel", _count(1)),
    "max-tokens-per-gpu": ("rl_max_tokens_per_gpu", _count(1)),
}
# Parallel-shape keys: the island trains with its own split (shape-free layout only).
PARALLEL_DESTS = ("tensor_parallel", "expert_parallel", "pipeline_parallel", "rl_max_tokens_per_gpu")


@dataclass(frozen=True)
class IslandPlacement:
    """The resolved placement values of one island."""

    island: int
    placement: str
    standby_gpus: int
    use_gpus_per_node: int | None
    # ``(args dest, value)`` of this island's own parallel shape (empty: run-wide).
    parallel: tuple[tuple[str, int], ...] = ()

    def record(self) -> dict[str, Any]:
        record = {"island": self.island, "placement": self.placement,
                  "standby_gpus": self.standby_gpus,
                  "use_gpus_per_node": self.use_gpus_per_node}
        if self.parallel:  # a run without per-island shapes keeps its records
            record["parallel"] = dict(self.parallel)
        return record


def parse_overrides(items: Iterable[str] | None, num_islands: int) -> dict[int, dict[str, Any]]:
    """``ISLAND:KEY=VALUE`` items -> ``{island: {args dest: value}}``.

    Raises ValueError for a bad item, an unknown key, an island outside
    ``0..num_islands-1`` or the same key given twice for one island.
    """

    out: dict[int, dict[str, Any]] = {}
    for item in items or ():
        island_text, sep, assignment = str(item).partition(":")
        key, eq, value = assignment.partition("=")
        if not sep or not eq or not key:
            raise ValueError(f"--rl-island-placement {item!r}: expected ISLAND:KEY=VALUE")
        try:
            island = int(island_text)
        except ValueError:
            raise ValueError(f"--rl-island-placement {item!r}: ISLAND must be an integer") from None
        if not 0 <= island < num_islands:
            raise ValueError(f"--rl-island-placement {item!r}: island {island} is not in "
                             f"0..{num_islands - 1}")
        if key not in KEYS:
            raise ValueError(f"--rl-island-placement {item!r}: unknown key {key!r} "
                             f"(known: {sorted(KEYS)})")
        dest, parse = KEYS[key]
        if dest in out.get(island, {}):
            raise ValueError(f"--rl-island-placement: island {island} gives {key!r} twice")
        try:
            out.setdefault(island, {})[dest] = parse(value)
        except ValueError as error:
            raise ValueError(f"--rl-island-placement {item!r}: {key} {error}") from None
    return out


def _resolved(args: Any, island: int, overrides: dict[int, dict[str, Any]]) -> IslandPlacement:
    own = overrides.get(island, {})

    def value(dest: str, default: Any) -> Any:
        return own[dest] if dest in own else (getattr(args, dest, default) or default)

    use = value("rl_island_use_gpus_per_node", None)
    return IslandPlacement(island=island, placement=value("rl_placement", COLOCATED),
                           standby_gpus=int(value("rl_standby_gpus", 0)),
                           use_gpus_per_node=None if use is None else int(use),
                           parallel=tuple((dest, own[dest]) for dest in PARALLEL_DESTS if dest in own))


def _overrides_and_placements(args: Any, num_islands: int):
    overrides = parse_overrides(getattr(args, OVERRIDES_DEST, None), num_islands)
    return overrides, tuple(_resolved(args, m, overrides) for m in range(num_islands))


def island_placements(args: Any, num_islands: int) -> tuple[IslandPlacement, ...]:
    """Every island's resolved placement values, in island order."""

    return _overrides_and_placements(args, num_islands)[1]


def is_mixed(placements: Sequence[IslandPlacement]) -> bool:
    return len({p.placement for p in placements}) > 1


def island_args(args: Any, island: int, num_islands: int) -> Any:
    """A copy of ``args`` with island ``island``'s placement values (``args``
    itself when the run has no per-island placement)."""

    if not getattr(args, OVERRIDES_DEST, None):
        return args
    placements = island_placements(args, num_islands)
    own = placements[island]
    out = copy.copy(args)
    out.rl_placement = own.placement
    out.rl_standby_gpus = own.standby_gpus
    out.rl_island_use_gpus_per_node = own.use_gpus_per_node
    for dest, value in own.parallel:
        setattr(out, dest, value)
    if own.placement == COLOCATED and is_mixed(placements):
        # A colocated island has no dedicated rollout or standby GPUs.
        out.rollout_num_gpus = None
        out.rl_standby_gpus = 0
    return out


def check(args: Any, num_islands: int, *, island_scheduling: str) -> tuple[IslandPlacement, ...]:
    """Launch checks of per-island placement, before any cloud work.

    Refuses: mixed placement under the legacy (strict) schedule; ``--rl-elastic``
    or ``--rl-overlap-eval`` in a mixed run (both need a fixed partition on every
    island; the error names the colocated islands); a per-island
    ``standby-gpus`` above 0 on a colocated island.
    """

    overrides, placements = _overrides_and_placements(args, num_islands)
    shaped = [p.island for p in placements if p.parallel]
    if shaped and getattr(args, "rl_sync_layout", "shape-bound") != "shape-free":
        raise ValueError(
            f"--rl-island-placement gives island(s) {shaped} their own parallel shape; islands of "
            "different shapes share a syncer session only with --rl-sync-layout shape-free"
        )
    colocated = [p.island for p in placements if p.placement == COLOCATED]
    if is_mixed(placements) and island_scheduling != "elastic":
        kinds = ", ".join(f"island {p.island} {p.placement}" for p in placements)
        raise ValueError(
            f"mixed island placement ({kinds}) needs island pool options: a fixed "
            "roster waits for every island each round and is not supported with "
            "different placements; use one placement for every island or add island "
            "pool options (e.g. --rl-sync-quorum-theta, --rl-sync-island-lease-s)")
    # One placement on every island: the execution-profile checks give the
    # usual error. A mixed run names its colocated islands here.
    for flag, dest in (("--rl-elastic", "rl_elastic"), ("--rl-overlap-eval", "rl_overlap_eval")):
        if getattr(args, dest, False) and colocated and is_mixed(placements):
            raise ValueError(f"{flag} needs --rl-placement fixed-partition on every island; "
                             f"colocated island(s): {colocated}")
    for m in colocated:
        if overrides.get(m, {}).get("rl_standby_gpus", 0) > 0:
            raise ValueError(f"--rl-island-placement: island {m} is colocated and cannot "
                             "reserve standby GPUs (standby-gpus must be 0)")
    return placements
