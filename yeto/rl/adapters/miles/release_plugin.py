"""LoRA policy release export executed inside every Megatron rank (L4.6 LoRA).

Loaded through ``TrainGroup.run_plugin`` like :mod:`.cut_plugin`. A trainer
island in LoRA mode releases only its adapter: ``write_lora_adapter`` writes
``adapter_config.json`` + ``adapter_model.safetensors`` (HF/PEFT layout) with
the fork's snapshot publisher -- the same writer ``export_hf`` uses for the
``adapter/`` sub-directory of a LoRA HF export, without the merged base model.

The write is collective (the publisher gathers the adapter across TP/PP and
global rank 0 writes the files), so every rank must run it. A colocated
trainer is asleep when it publishes; see :func:`_awake_process_groups`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

WRITE_LORA_ADAPTER = "yeto.rl.adapters.miles.release_plugin.write_lora_adapter"


def write_lora_adapter(actor: Any, *, path: str) -> dict[str, Any]:
    """Write the actor's current LoRA adapter into ``path`` (collective)."""
    publisher = getattr(actor, "snapshot_publisher", None)
    if publisher is None:
        raise RuntimeError("LoRA release export needs the actor's snapshot publisher "
                           "(Miles builds it when --save-hf is set)")
    with _awake_process_groups(actor):
        publisher.write_adapter(None, Path(path))
    return {"path": str(path)}


@contextmanager
def _awake_process_groups(actor: Any) -> Iterator[None]:
    """A colocated trainer publishes asleep: its process groups are destroyed and
    its GPU memory is paused (``actor.sleep``). The adapter gather needs both, so
    this does what the fork's ``update_weights`` does for an asleep actor: reload
    the groups, run with the memory saver disabled, destroy the groups again
    (T-LSMP c: CUDA invalid argument in all_gather_object without it)."""
    args = getattr(actor, "args", None)
    if not (getattr(args, "offload_train", False) and getattr(actor, "_asleep", False)):
        yield
        return
    from miles.utils.reloadable_process_group import destroy_process_groups, reload_process_groups
    from torch_memory_saver import torch_memory_saver

    reload_process_groups()
    try:
        with torch_memory_saver.disable():
            yield
    finally:
        destroy_process_groups()
