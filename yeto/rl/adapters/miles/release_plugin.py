"""LoRA policy release export executed inside every Megatron rank (L4.6 LoRA).

Loaded through ``TrainGroup.run_plugin`` like :mod:`.cut_plugin`. A trainer
island in LoRA mode releases only its adapter: ``write_lora_adapter`` writes
``adapter_config.json`` + ``adapter_model.safetensors`` (HF/PEFT layout) with
the fork's snapshot publisher -- the same writer ``export_hf`` uses for the
``adapter/`` sub-directory of a LoRA HF export, without the merged base model.

The write is collective (the publisher gathers the adapter across TP/PP and
global rank 0 writes the files), so every rank must run it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

WRITE_LORA_ADAPTER = "yeto.rl.adapters.miles.release_plugin.write_lora_adapter"


def write_lora_adapter(actor: Any, *, path: str) -> dict[str, Any]:
    """Write the actor's current LoRA adapter into ``path`` (collective)."""
    publisher = getattr(actor, "snapshot_publisher", None)
    if publisher is None:
        raise RuntimeError("LoRA release export needs the actor's snapshot publisher "
                           "(Miles builds it when --save-hf is set)")
    publisher.write_adapter(None, Path(path))
    return {"path": str(path)}
