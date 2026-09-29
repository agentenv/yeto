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
    # rl-infra-spec 2.1/2.1a: reserved standby GPUs (fixed partition only) and
    # an optional explicit logical-bundle map {"trainer","rollout","standby"}.
    standby_gpus: int = 0
    bundle_map: Mapping[str, tuple[int, ...]] | None = None

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
        if isinstance(self.standby_gpus, bool) or not isinstance(self.standby_gpus, int) \
                or self.standby_gpus < 0:
            raise ValueError("standby_gpus must be a non-negative int")
        if self.kind == "colocated" and (self.standby_gpus or self.bundle_map is not None):
            raise ValueError("colocated placement has no standby GPUs or bundle map")
        if self.bundle_map is not None:
            object.__setattr__(self, "bundle_map", validate_bundle_map(
                self.bundle_map, trainer=self.trainer_gpus, rollout=self.rollout_gpus,
                standby=self.standby_gpus,
            ))

    @property
    def total_gpus(self) -> int:
        if self.kind == "colocated":
            return self.trainer_gpus
        return self.trainer_gpus + self.rollout_gpus + self.standby_gpus

    @property
    def placement_map(self) -> dict[str, list[int]] | None:
        """``--yeto-placement-map`` payload (fork-M1), or None for the upstream offset layout.

        A standby reservation always needs the map; otherwise only an explicit
        ``bundle_map`` emits one (the offset layout is identical without it).
        """
        if self.kind == "colocated":
            return None
        if self.bundle_map is not None:
            return {k: list(v) for k, v in self.bundle_map.items()}
        if not self.standby_gpus:
            return None
        t, r = self.trainer_gpus, self.rollout_gpus
        return {
            "trainer": list(range(t)),
            "rollout": list(range(t, t + r)),
            "standby": list(range(t + r, t + r + self.standby_gpus)),
        }

    def role_bundles(self) -> dict[str, tuple[int, ...]]:
        pm = self.placement_map
        if pm is not None:
            return {k: tuple(pm.get(k, ())) for k in ("trainer", "rollout", "standby")}
        total, offset = self.expected_layout()
        if self.kind == "colocated":
            return {"trainer": tuple(range(total)), "rollout": tuple(range(total)), "standby": ()}
        return {"trainer": tuple(range(offset)), "rollout": tuple(range(offset, total)),
                "standby": ()}

    def expected_layout(self) -> tuple[int, int]:
        """``(total_bundles, rollout_offset)`` of the upstream args layout (standby excluded)."""

        if self.kind == "colocated":
            return self.trainer_gpus, 0
        return self.trainer_gpus + self.rollout_gpus, self.trainer_gpus


BUNDLE_ROLES = ("trainer", "rollout", "standby")


def validate_bundle_map(
    raw: Mapping[str, Any], *, trainer: int, rollout: int, standby: int
) -> dict[str, tuple[int, ...]]:
    """Same rules as fork-M1 ``validate_placement_map``, checked before launch."""
    unknown = sorted(set(raw) - set(BUNDLE_ROLES))
    if unknown:
        raise ValueError(f"bundle map names unknown roles {unknown}")
    out = {}
    for role in BUNDLE_ROLES:
        values = raw.get(role, ())
        if isinstance(values, (str, bytes)) or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in values
        ):
            raise ValueError(f"bundle map role {role!r} must be a list of ints")
        out[role] = tuple(values)
    n = trainer + rollout + standby
    seen: set[int] = set()
    for role, want in zip(BUNDLE_ROLES, (trainer, rollout, standby), strict=True):
        idx = out[role]
        if len(set(idx)) != len(idx):
            raise ValueError(f"bundle map role {role!r} repeats bundles")
        if any(not 0 <= i < n for i in idx):
            raise ValueError(f"bundle map role {role!r} has bundles outside 0..{n - 1}")
        if seen & set(idx):
            raise ValueError(f"bundle map role {role!r} overlaps another role")
        if len(idx) != want:
            raise ValueError(f"bundle map gives {role} {len(idx)} bundles, requested {want}")
        seen |= set(idx)
    return out


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
    parsed_map = getattr(args, "yeto_placement_map", None)
    if isinstance(parsed_map, str):
        import json

        parsed_map = json.loads(parsed_map)
    if parsed_map != request.placement_map:
        conflicts.append(f"placement map {parsed_map} != requested {request.placement_map}")
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
        standby: tuple[str, ...] = ()
        if request.placement_map is not None:
            # fork-M1 returns each role sliced from the map: "actor" holds the
            # trainer bundles only and "standby" the reserved ones.
            standby = _gpu_ids(placement_groups.get("standby", (None, [], [])), logical)
            everything = actor + rollout + standby
            if len(actor) != request.trainer_gpus or len(standby) != request.standby_gpus \
                    or len(set(everything)) != len(everything):
                raise PlacementRewriteError(
                    f"placement map produced trainer={len(actor)} rollout={len(rollout)} "
                    f"standby={len(standby)} (overlapping or wrong sizes)"
                )
            if len(rollout) != request.rollout_gpus:
                raise PlacementRewriteError(
                    f"rollout partition has {len(rollout)} GPUs, requested {request.rollout_gpus}"
                )
            self._request = request
            self._description = PlacementDescription(
                kind=request.kind,
                trainer_gpus=actor,
                rollout_gpus=rollout,
                extra={
                    "gpus_per_engine": request.gpus_per_engine,
                    "logical": logical,
                    "standby_gpus": standby,
                    "weight_transport": "nccl-broadcast",
                },
            )
            return
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
        if request.placement_map is not None:
            roles = request.role_bundles()
            view = {k: (None, list(v), list(v)) for k, v in roles.items()}
            return cls(request, {"actor": view["trainer"], "rollout": view["rollout"],
                                 "standby": view["standby"]}, logical=True)
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
