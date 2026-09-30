"""Read-only placement description and rewrite detection (task 3.6).

R0 supports two placements fixed at startup: ``colocated`` (trainer and
rollout share the same GPUs) and ``fixed-partition`` (disjoint trainer and
rollout GPU sets). Upstream Miles normalizes its arguments (``--colocate``
forces ``rollout_num_gpus``; ``--debug-rollout-only`` turns colocate off;
external engines drop rollout GPUs; ...). The spec forbids running with a
silently rewritten placement, so :func:`check_placement_not_rewritten`
compares the requested placement with the *parsed* namespace and refuses a
mismatch. :class:`MilesPlacement` describes the placement groups that upstream
``create_placement_groups`` actually produced. Pure Python; no ray import.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..ports import PlacementDescription, PlacementKind


class PlacementRewriteError(RuntimeError):
    """Miles' argument normalization changed the requested placement."""


@dataclass(frozen=True)
class PlacementRequest:
    kind: PlacementKind
    trainer_gpus: int
    rollout_gpus: int
    gpus_per_engine: int

    def __post_init__(self) -> None:
        if self.kind not in ("colocated", "fixed-partition"):
            raise ValueError(f"unsupported placement {self.kind!r}")
        for name in ("trainer_gpus", "rollout_gpus", "gpus_per_engine"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"placement {name} must be a positive int")
        if self.kind == "colocated" and self.rollout_gpus != self.trainer_gpus:
            raise ValueError("colocated placement shares one GPU set")
        if self.rollout_gpus % self.gpus_per_engine:
            raise ValueError("rollout GPUs must be a multiple of GPUs per engine")

    def expected_layout(self) -> tuple[int, int]:
        """``(total_bundles, rollout_offset)`` as upstream lays out the PG."""

        if self.kind == "colocated":
            return self.trainer_gpus, 0
        return self.trainer_gpus + self.rollout_gpus, self.trainer_gpus


def _parsed_layout(args: Any) -> tuple[int, int] | None:
    """Mirror of upstream ``_get_placement_group_layout`` for the R0 subset."""

    trainer = int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node)
    rollout = int(getattr(args, "rollout_num_gpus", 0) or 0)
    if getattr(args, "colocate", False):
        return max(trainer, rollout), 0
    return trainer + rollout, trainer


def check_placement_not_rewritten(request: PlacementRequest, args: Any) -> None:
    conflicts: list[str] = []
    for name in ("debug_rollout_only", "debug_train_only", "rollout_external"):
        if getattr(args, name, False):
            conflicts.append(f"{name}=True")
    if int(getattr(args, "eval_num_gpus", 0) or 0):
        conflicts.append(f"eval_num_gpus={args.eval_num_gpus} adds GPUs to the rollout partition")
    colocate = bool(getattr(args, "colocate", False))
    if colocate != (request.kind == "colocated"):
        conflicts.append(f"colocate={colocate} but {request.kind!r} was requested")
    trainer = int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node)
    if trainer != request.trainer_gpus:
        conflicts.append(f"trainer GPUs {trainer} != requested {request.trainer_gpus}")
    rollout = int(getattr(args, "rollout_num_gpus", 0) or 0)
    if rollout != request.rollout_gpus:
        conflicts.append(f"rollout GPUs {rollout} != requested {request.rollout_gpus}")
    engine = getattr(args, "rollout_num_gpus_per_engine", request.gpus_per_engine)
    if int(engine) != request.gpus_per_engine:
        conflicts.append(f"GPUs per engine {engine} != requested {request.gpus_per_engine}")
    if not conflicts and _parsed_layout(args) != request.expected_layout():
        conflicts.append(
            f"placement-group layout {_parsed_layout(args)} != requested {request.expected_layout()}"
        )
    if conflicts:
        raise PlacementRewriteError(
            f"Miles argument normalization rewrote the requested {request.kind!r} "
            "placement: " + "; ".join(conflicts)
        )


def _gpu_ids(info: Any, logical: bool = False) -> tuple[str, ...]:
    """Accepts upstream ``PlacementGroupInfo`` or a ``(pg, bundles, gpus)`` tuple."""

    bundles = list(info[1])
    if logical:
        return tuple(f"bundle{b}" for b in bundles)
    gpus = list(info[2])
    return tuple(f"bundle{b}:gpu{g}" for b, g in zip(bundles, gpus, strict=True))


class MilesPlacement:
    """``Placement`` port over upstream ``create_placement_groups`` output."""

    def __init__(
        self,
        request: PlacementRequest,
        placement_groups: Mapping[str, Any],
        *,
        logical: bool = False,
    ) -> None:
        self._request = request
        actor = _gpu_ids(placement_groups["actor"], logical)
        rollout = _gpu_ids(placement_groups["rollout"], logical)
        total, offset = request.expected_layout()
        if len(actor) != total:
            raise PlacementRewriteError(
                f"placement group has {len(actor)} bundles, requested layout needs {total}"
            )
        trainer_gpus = actor[: request.trainer_gpus]
        if request.kind == "colocated":
            if set(rollout) != set(trainer_gpus):
                raise PlacementRewriteError("colocated rollout GPUs differ from trainer GPUs")
        else:
            if rollout != actor[offset:] or set(rollout) & set(trainer_gpus):
                raise PlacementRewriteError("fixed partition rollout GPUs overlap the trainer")
            if len(rollout) != request.rollout_gpus:
                raise PlacementRewriteError(
                    f"rollout partition has {len(rollout)} GPUs, requested {request.rollout_gpus}"
                )
        self._description = PlacementDescription(
            kind=request.kind,
            trainer_gpus=trainer_gpus,
            rollout_gpus=rollout,
            extra={"gpus_per_engine": request.gpus_per_engine, "logical": logical},
        )

    @classmethod
    def from_parsed_args(cls, request: PlacementRequest, args: Any) -> "MilesPlacement":
        """Describe the layout upstream will build from ``args``, by logical bundle.

        Upstream creates the placement group inside
        ``init_orchestration_script`` -> ``launch_worker_manager``; the
        driver does not get it back, so this describes the logical bundle
        layout (identical ordering to ``create_placement_groups``). Use the
        constructor directly when physical GPU ids are available.
        """

        check_placement_not_rewritten(request, args)
        total, offset = _parsed_layout(args)
        bundles = list(range(total))
        actor = (None, bundles, bundles)
        rollout = (None, bundles[offset:], bundles[offset:])
        return cls(request, {"actor": actor, "rollout": rollout}, logical=True)

    @property
    def request(self) -> PlacementRequest:
        return self._request

    def describe(self) -> PlacementDescription:
        return self._description
