"""Trainable LoRA state through Miles' adapter slots (raw bridge mode: Kimi-K3.1).

Miles trains Kimi-K3.1 in raw bridge mode (``--megatron-to-hf-mode raw``), so its
LoRA adapter is not reachable through Megatron-Bridge adapter conversion tasks
(:func:`.state_plugin.adapter_bindings` would ask AutoBridge, which cannot read
``kimi_k31``). Miles keeps the HF layout of the adapter in its adapter slots
(``actor.adapter_slots``: ``NativeAdapterSlots``, a probed HF row mapping) and
checkpoints it through ``DistributedOptimizerMembers``: per rank, the rectangles
(``LocalShardSpec``) of each HF-keyed tensor this rank's DP group leads, with
their fp32 masters and Adam moments. yeto's export/apply reuse exactly that:

* export: every lead rank cuts its fp32 master pieces
  (``members.export_adapter_state(0)``); the primary copy of each piece is
  gathered over Miles' Gloo group and the main rank assembles each HF tensor,
  checking that the pieces tile it exactly once;
* apply: every lead rank cuts its pieces out of the canonical tensors and imports
  them with its current moments (``import_adapter_state`` +
  ``finish_import_adapter_state``: scatter over DP, re-cast to bf16, all-gather the
  parameters). The optimizer mode (``reset``) is applied by the caller afterwards.

Canonical names: ``base_model.model.<HF key>.weight`` (the HF key ends in
``lora_A`` / ``lora_B``). The layout is learned from the first export (the CPU
cannot derive it); every island learns the same one.

The collective and the Miles members are injected (``gather`` / ``members``) so the
assembly and the piece cutting run on the CPU in tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

CANONICAL_PREFIX = "base_model.model."
WEIGHT_SUFFIX = ".weight"

# one primary piece on the wire: (HF key, global shape, offsets, fp32 CPU tensor)
Piece = tuple[str, tuple[int, ...], tuple[int, ...], Any]


class NativeSlotsError(RuntimeError):
    pass


def canonical_name(key: str) -> str:
    return f"{CANONICAL_PREFIX}{key}{WEIGHT_SUFFIX}"


def hf_key(name: str) -> str:
    if not (name.startswith(CANONICAL_PREFIX) and name.endswith(WEIGHT_SUFFIX)):
        raise NativeSlotsError(f"{name!r} is not a canonical native-slots LoRA name")
    return name[len(CANONICAL_PREFIX):-len(WEIGHT_SUFFIX)]


def _box(offsets: Sequence[int], shape: Sequence[int]) -> tuple[slice, ...]:
    return tuple(slice(int(o), int(o) + int(n)) for o, n in zip(offsets, shape, strict=True))


def primary_pieces(state: Any) -> list[Piece]:
    """This rank's primary pieces of an ``AdapterState`` (fp32, CPU)."""
    import torch

    out = []
    for spec, weight in zip(state.shard_specs, state.weights, strict=True):
        if not getattr(spec, "is_primary", True):
            continue
        value = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(value.shape) != tuple(spec.local_shape):
            raise NativeSlotsError(f"{spec.key}: piece {tuple(value.shape)} != {tuple(spec.local_shape)}")
        out.append((str(spec.key), tuple(int(n) for n in spec.global_shape),
                    tuple(int(o) for o in spec.offsets), value))
    return out


def assemble(gathered: Iterable[Sequence[Piece]]) -> dict[str, Any]:
    """Each HF tensor from the primary pieces of every rank, under its canonical name.
    Refuses a tensor whose pieces overlap, leave a hole, or disagree on its shape."""
    import torch

    tensors: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    for rank_pieces in gathered:
        for key, global_shape, offsets, value in rank_pieces or ():
            whole = tensors.get(key)
            if whole is None:
                whole = tensors[key] = torch.zeros(global_shape, dtype=torch.float32)
                counts[key] = torch.zeros(global_shape, dtype=torch.int32)
            elif tuple(whole.shape) != tuple(global_shape):
                raise NativeSlotsError(f"{key}: pieces disagree on the shape "
                                       f"({tuple(whole.shape)} vs {tuple(global_shape)})")
            box = _box(offsets, value.shape)
            whole[box] = value
            counts[key][box] += 1
    if not tensors:
        raise NativeSlotsError("the adapter slots exported no LoRA pieces")
    for key, count in counts.items():
        if not bool((count == 1).all()):
            holes, overlaps = int((count == 0).sum()), int((count > 1).sum())
            raise NativeSlotsError(f"{key}: the primary pieces leave {holes} elements unset "
                                   f"and set {overlaps} more than once")
    return {canonical_name(key): value for key, value in sorted(tensors.items())}


def cut(tensors: Mapping[str, Any], specs: Sequence[Any]) -> list[Any]:
    """The fp32 pieces ``specs`` describe, cut out of canonical tensors."""
    import torch

    out = []
    for spec in specs:
        name = canonical_name(spec.key)
        whole = tensors.get(name)
        if whole is None:
            raise NativeSlotsError(f"apply: no tensor {name!r} for this rank's piece")
        if tuple(whole.shape) != tuple(spec.global_shape):
            raise NativeSlotsError(f"apply: {name!r} has shape {tuple(whole.shape)}, "
                                   f"the slots hold {tuple(spec.global_shape)}")
        out.append(whole[_box(spec.offsets, spec.local_shape)].to(torch.float32).contiguous())
    return out


# -------------------------------------------------------------- Miles defaults


def _precision_aware(args: Any) -> bool:
    backend = getattr(args, "backend", None)
    value = getattr(backend, "use_precision_aware_optimizer", None) if backend is not None else None
    if value is None:
        value = getattr(args, "use_precision_aware_optimizer", False)
    return bool(value)


def miles_members(actor: Any, *, save_moments: bool) -> Any:
    """Miles' single-LoRA checkpoint members of the actor (``DistributedOptimizerMembers``)."""
    from miles.backends.megatron_utils.lora.checkpoint import DistributedOptimizerMembers

    slots = getattr(actor, "adapter_slots", None)
    if slots is None:
        raise NativeSlotsError("the actor has no adapter slots (Miles builds them for a LoRA actor)")
    return DistributedOptimizerMembers(slots, actor.optimizer, actor.model,
                                       precision_aware=_precision_aware(actor.args),
                                       save_moments=save_moments)


def gloo_all_gather(local: list[Piece]) -> list[list[Piece]]:
    """Every rank's pieces, over Miles' Gloo group (one process: just ``local``)."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return [local]
    from miles.backends.training_utils.checkpoint.adapter.adapter_state import gloo_group

    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local, group=gloo_group())
    return gathered


def _adapter_state(specs: Sequence[Any], weights: list[Any], moments: Any) -> Any:
    from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState

    return AdapterState(list(specs), weights, list(moments.exp_avgs), list(moments.exp_avg_sqs),
                        step=moments.step)


# -------------------------------------------------------------- export / apply


def export_tensors(actor: Any, *, is_main: bool,
                   members: Callable[..., Any] | None = None,
                   gather: Callable[[list[Piece]], list[list[Piece]]] | None = None) -> dict[str, Any] | None:
    """Collective: canonical fp32 tensors on the main rank, None elsewhere."""
    state = (members or miles_members)(actor, save_moments=False).export_adapter_state(0)
    gathered = (gather or gloo_all_gather)(primary_pieces(state))
    return assemble(gathered) if is_main else None


def apply_tensors(actor: Any, tensors: Mapping[str, Any], *,
                  members: Callable[..., Any] | None = None,
                  adapter_state: Callable[..., Any] | None = None) -> int:
    """Collective: write canonical tensors into the fp32 masters (moments kept) and the
    bf16 parameters. Every rank receives the whole state. Returns this rank's piece count."""
    held = (members or miles_members)(actor, save_moments=True)
    current = held.export_adapter_state(0)  # this rank's specs and its current moments
    specs = list(current.shard_specs)
    wanted = {canonical_name(spec.key) for spec in specs}
    for name in tensors:
        hf_key(name)  # every rank receives the whole state; each name must be canonical
    missing = sorted(wanted - set(tensors))
    if missing:
        raise NativeSlotsError(f"apply: missing LoRA tensors this rank holds: {missing[:4]}")
    held.import_adapter_state(0, (adapter_state or _adapter_state)(specs, cut(tensors, specs), current))
    held.finish_import_adapter_state(0)
    return len(specs)


__all__ = ["CANONICAL_PREFIX", "NativeSlotsError", "apply_tensors", "assemble", "canonical_name",
           "cut", "export_tensors", "gloo_all_gather", "hf_key", "miles_members", "primary_pieces"]
