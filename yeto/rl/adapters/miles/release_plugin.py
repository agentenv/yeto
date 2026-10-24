"""Policy release exports executed inside every Megatron rank (L4.6).

Full weights (S27 G-AT, :func:`write_full_checkpoint`): the trainer island stages
each release on its local disk; every rank writes its share of the HF
safetensors files (Megatron-Bridge ``distributed_save``) instead of rank 0
writing all of them. The gather and conversion are Miles' ``export_hf`` (same
bridge, same completion marker); only the file writes are spread. A model the
bridge path does not export (``--megatron-to-hf-mode raw``, LoRA) is written by
``export_hf`` itself.

Export timings (S27 G-CJ, measurement only): every rank reports how long its
export took (``export_s``) and how long its safetensors file writes took
(``save_s``, ``save_files``, ``save_bytes``; the writes are timed by wrapping
``safetensors.torch.save_file`` while the export runs). :func:`export_profile`
sums them up for the release event: it tells the conversion from the file writes.

LoRA policy release export (L4.6 LoRA):

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

import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

WRITE_LORA_ADAPTER = "yeto.rl.adapters.miles.release_plugin.write_lora_adapter"
WRITE_FULL_CHECKPOINT = "yeto.rl.adapters.miles.release_plugin.write_full_checkpoint"


def full_checkpoint_writer(actor_model: Any, runner: Any, *, all_ranks: bool
                           ) -> Callable[[int, Path], list[dict[str, Any]] | None]:
    """Driver side: ``write(rollout_id, directory)`` puts the actor's full weights
    into ``directory`` as an HF checkpoint. ``all_ranks``: every rank writes a
    share of the files (:func:`write_full_checkpoint`; ``directory`` must be on a
    disk every trainer rank sees) and ``write`` returns the ranks' reports; else
    Miles' ``export_hf`` (rank 0 writes) and ``write`` returns None."""

    def write(rollout_id: int, directory: Path) -> list[dict[str, Any]] | None:
        if all_ranks:
            return runner.run(actor_model.run_plugin(WRITE_FULL_CHECKPOINT, {
                "path": str(directory), "rollout_id": int(rollout_id)}))
        runner.run(actor_model.export_hf(rollout_id=int(rollout_id), path=str(directory)))
        return None

    return write


def export_profile(reports: Sequence[Any] | None) -> dict[str, float]:
    """The ranks' export timings (module doc) for a release event: the slowest
    rank's export and file-write seconds, the summed write seconds and the GB
    written. Empty when no rank reported them (rank 0 export, LoRA)."""
    ranks = [r for r in (reports or ()) if isinstance(r, Mapping) and r.get("export_s") is not None]
    if not ranks:
        return {}
    saves = [float(r.get("save_s") or 0.0) for r in ranks]
    return {"rank_export_max_s": round(max(float(r["export_s"]) for r in ranks), 3),
            "rank_save_max_s": round(max(saves), 3), "rank_save_sum_s": round(sum(saves), 3),
            "rank_save_gb": round(sum(int(r.get("save_bytes") or 0) for r in ranks) / 1e9, 3)}


@contextmanager
def _timed_saves() -> Iterator[dict[str, float]]:
    """Time the safetensors file writes made while the context is open (module
    doc): ``safetensors.torch.save_file`` is wrapped and restored afterwards. A
    caller that bound the function before the context is not timed (zero)."""
    import safetensors.torch as safetensors_torch

    original = safetensors_torch.save_file
    stats: dict[str, float] = {"save_s": 0.0, "save_files": 0, "save_bytes": 0}

    def save_file(tensors: Any, filename: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(tensors, filename, *args, **kwargs)
        finally:
            stats["save_s"] += time.perf_counter() - started
            stats["save_files"] += 1
            stats["save_bytes"] += os.path.getsize(filename) if os.path.exists(filename) else 0

    safetensors_torch.save_file = save_file
    try:
        yield stats
    finally:
        safetensors_torch.save_file = original


def write_full_checkpoint(actor: Any, *, path: str, rollout_id: int) -> dict[str, Any]:
    """Write the actor's full weights into ``path`` as an HF checkpoint, every rank
    writing a share of the files (collective; module doc). Returns who wrote."""
    from miles.backends.megatron_utils.lora.utils import is_lora_model

    args, model = actor.args, actor.model
    if getattr(args, "megatron_to_hf_mode", None) != "bridge" or is_lora_model(model):
        actor.export_hf(rollout_id=int(rollout_id), path=str(path))
        return {"path": str(path), "writers": "rank0"}
    import torch
    from miles.backends.megatron_utils.hf_export import _get_hf_bridge
    from miles.backends.training_utils.checkpoint_io import write_checkpoint_dir
    from miles.utils.distributed_utils import get_gloo_group
    from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER
    from miles.utils.megatron_bridge_utils import patch_megatron_model

    heartbeat = getattr(actor, "_heartbeat", None)
    if heartbeat is not None:
        heartbeat.bump()
    bridge = _get_hf_bridge(args.hf_checkpoint)

    def write(directory: Path) -> None:
        with patch_megatron_model(model):
            bridge.save_hf_pretrained(model, path=directory, show_progress=False, distributed_save=True)
        torch.distributed.barrier(group=get_gloo_group())
        if not any(Path(directory).glob("*.safetensors")):
            raise RuntimeError(f"HF export to {directory} produced no weight files")

    started = time.perf_counter()
    with _awake_process_groups(actor), _timed_saves() as saves:
        write_checkpoint_dir(Path(path), write, completion_marker=HF_EXPORT_COMPLETE_MARKER)
    return {"path": str(path), "writers": "all", "export_s": round(time.perf_counter() - started, 3),
            "save_s": round(saves["save_s"], 3), "save_files": int(saves["save_files"]),
            "save_bytes": int(saves["save_bytes"])}


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
