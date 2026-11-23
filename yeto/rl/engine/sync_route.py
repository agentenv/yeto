"""Which synchronization stack a requested RL sync preset runs on.

``--rl-sync-preset decoupled`` names asynchronous multi-fragment outer
synchronization (Decoupled DiLoCo) for every trainable form. Two stacks
implement it:

* the LoRA fragment stack (``DecoupledSync`` over the canonical LoRA layout),
  the preset ``decoupled`` itself;
* the full-parameter stack (``FullParameterSync``) with asynchronous fragment
  rounds, the preset ``full-parameter`` with
  ``--rl-sync-full-parameter-rounds async``. It also carries the trainable
  routed experts of a partial-experts run and the DeepSeek V4 expert-full
  LoRA hybrid.

:func:`resolve_sync_route` maps the requested preset and trainable form to the
canonical preset; :func:`apply_sync_route` writes it back into the launch
arguments before any other check reads them, so the islands, the syncer
command and every later rule see only canonical presets. Every combination
the launcher accepted before routing existed maps to itself; only
``decoupled`` with ``--tuning full`` or ``--expert-full-count`` changes, and
the launcher refused both before.

Import-light: no torch, ray or miles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .selection import (
    FULL_PARAMETER_SYNC_PRESET,
    SAO_STREAMING_SYNC_PRESET,
    canonical_sync_preset,
    is_sync_mode_default,
)

DECOUPLED_SYNC_PRESET = "decoupled"
STRICT_AVG_SYNC_PRESET = "strict-avg"

# Transport names recorded in the route (provenance and the dry-run line).
LORA_FRAGMENTS = "lora-fragments"
LORA_STRICT = "lora-strict"
FULL_PARAMETER = "full-parameter"
SAO_STREAMING = "sao-streaming"
NO_SYNC = "no-sync"


class SyncRouteError(ValueError):
    """The requested preset contradicts another option of the run."""


@dataclass(frozen=True)
class SyncRoute:
    requested: str
    preset: str
    transport: str
    full_parameter_rounds: str | None
    reason: str

    @property
    def rerouted(self) -> bool:
        return self.preset != self.requested

    def line(self) -> str:
        """One line for the launch output."""
        rounds = f" ({self.full_parameter_rounds} rounds)" if self.full_parameter_rounds else ""
        return f"[sync-route] {self.requested} -> {self.preset}{rounds}: {self.reason}"


def _full_parameter_form(tuning: str, expert_full_count: int) -> str | None:
    """The trainable form the full-parameter stack carries, or None for plain LoRA."""
    if tuning == "full":
        return "--tuning full"
    if int(expert_full_count or 0) > 0:
        return "--expert-full-count (expert-full LoRA hybrid)"
    return None


def resolve_sync_route(
    *,
    preset: str,
    tuning: str = "lora",
    expert_full_count: int = 0,
    full_parameter_rounds: str | None = None,
    rounds_explicit: bool = False,
    single_island_no_sync: bool = False,
) -> SyncRoute:
    """The canonical preset for ``preset`` and the run's trainable form.

    ``full_parameter_rounds`` / ``rounds_explicit`` describe
    ``--rl-sync-full-parameter-rounds`` as given (explicit or still the
    sync-mode default). The route never validates the target preset's own
    rules; the launcher checks those afterwards on the canonical preset.
    """
    requested = canonical_sync_preset(str(preset))
    if single_island_no_sync:
        return SyncRoute(requested, requested, NO_SYNC, None, "no-sync island: no outer synchronization")
    form = _full_parameter_form(str(tuning), expert_full_count)
    if requested == DECOUPLED_SYNC_PRESET:
        if form is None:
            return SyncRoute(requested, requested, LORA_FRAGMENTS, None, "LoRA fragments")
        if rounds_explicit and full_parameter_rounds not in (None, "async"):
            raise SyncRouteError(
                f"--rl-sync-preset decoupled runs asynchronous fragment rounds; "
                f"--rl-sync-full-parameter-rounds {full_parameter_rounds} is the strict sweep "
                f"(use --rl-sync-preset {FULL_PARAMETER_SYNC_PRESET} for it)")
        return SyncRoute(requested, FULL_PARAMETER_SYNC_PRESET, FULL_PARAMETER, "async", form)
    if requested == FULL_PARAMETER_SYNC_PRESET:
        return SyncRoute(requested, requested, FULL_PARAMETER, full_parameter_rounds, "full-parameter preset")
    if requested == SAO_STREAMING_SYNC_PRESET:
        return SyncRoute(requested, requested, SAO_STREAMING, None, "SAO actor and critic streams")
    return SyncRoute(requested, requested, LORA_STRICT, None, "strict LoRA average")


def apply_sync_route(args: Any) -> SyncRoute | None:
    """Route ``args.rl_sync_preset`` in place (RL launches only).

    Sets ``args.rl_sync_preset_requested`` and ``args.rl_sync_route`` (a
    :class:`SyncRoute`). A rerouted ``decoupled`` run takes
    ``--rl-sync-full-parameter-rounds async`` as an explicit value and, while
    ``--rl-sync-fragment-pipeline`` is unset, the decoupled ``--pipeline``
    bounded by ``--fragments``. Returns None for a non-RL run.
    """
    if getattr(args, "training_mode", "sft") != "rl":
        return None
    rounds = getattr(args, "rl_sync_full_parameter_rounds", None)
    route = resolve_sync_route(
        preset=getattr(args, "rl_sync_preset", STRICT_AVG_SYNC_PRESET),
        tuning=getattr(args, "tuning", "lora"),
        expert_full_count=int(getattr(args, "expert_full_count", 0) or 0),
        full_parameter_rounds=None if rounds is None else str(rounds),
        rounds_explicit=rounds is not None and not is_sync_mode_default(rounds),
        single_island_no_sync=bool(getattr(args, "rl_single_island_no_sync", False)),
    )
    args.rl_sync_preset_requested = route.requested
    args.rl_sync_route = route
    if route.rerouted:
        args.rl_sync_preset = route.preset
        args.rl_sync_full_parameter_rounds = "async"
        pipeline = getattr(args, "rl_sync_fragment_pipeline", None)
        if pipeline is None or is_sync_mode_default(pipeline):
            fragments = max(int(getattr(args, "fragments", 1) or 1), 1)
            args.rl_sync_fragment_pipeline = max(1, min(int(getattr(args, "pipeline", 1) or 1), fragments))
    return route
