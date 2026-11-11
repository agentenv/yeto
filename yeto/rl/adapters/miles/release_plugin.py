"""Policy release exports executed inside every Megatron rank.

Full weights (:func:`write_full_checkpoint`): the trainer island stages
each release on its local disk; every rank writes its share of the HF
safetensors files (Megatron-Bridge ``distributed_save``) instead of rank 0
writing all of them. The gather and conversion are Miles' ``export_hf`` (same
bridge, same completion marker); only the file writes are spread. A model the
bridge path does not export (``--megatron-to-hf-mode raw``, LoRA) is written by
``export_hf`` itself.

LoRA policy release export (LoRA):

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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

WRITE_LORA_ADAPTER = "yeto.rl.adapters.miles.release_plugin.write_lora_adapter"
WRITE_FULL_CHECKPOINT = "yeto.rl.adapters.miles.release_plugin.write_full_checkpoint"


def full_checkpoint_writer(actor_model: Any, runner: Any, *, all_ranks: bool,
                           base_dir: str | Path | None = None) -> Callable[[int, Path], None]:
    """Driver side: ``write(rollout_id, directory)`` puts the actor's full weights
    into ``directory`` as an HF checkpoint. ``all_ranks``: every rank writes a
    share of the files (:func:`write_full_checkpoint`; ``directory`` must be on a
    disk every trainer rank sees); else Miles' ``export_hf`` (rank 0 writes).
    ``base_dir``: the base HF checkpoint; :func:`restore_base_metadata` then runs
    on the export."""

    def write(rollout_id: int, directory: Path) -> None:
        if all_ranks:
            runner.run(actor_model.run_plugin(WRITE_FULL_CHECKPOINT, {
                "path": str(directory), "rollout_id": int(rollout_id)}))
        else:
            runner.run(actor_model.export_hf(rollout_id=int(rollout_id), path=str(directory)))
        if base_dir is not None:
            restore_base_metadata(Path(directory), Path(base_dir))

    return write


# tokenizer files of an HF checkpoint: RL never changes them
TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.json",
                   "merges.txt", "tokenizer.model", "tiktoken.model", "added_tokens.json", "chat_template.jinja")


# the export decides these: a release of a quantized base is not quantized the same way
NOT_RESTORED_CONFIG_KEYS = frozenset({"quantization_config", "compression_config"})


def restore_base_metadata(directory: Path, base_dir: Path) -> dict[str, Any]:
    """Make an export readable by the sampler's transformers, whatever the trainer's.

    The trainer image saves the HF export with transformers 5, which rewrites the
    metadata in its own form: ``config.json`` moves ``rope_theta`` / ``rope_scaling``
    into ``rope_parameters`` and ``torch_dtype`` to ``dtype``; ``tokenizer_config.json``
    writes ``extra_special_tokens`` as a list. A transformers-4 sampler (vLLM 0.11)
    then reads no ``rope_theta`` and serves the release with the default RoPE base
    (found by the Qwen3-0.6B rehearsal: mean KL 0.0007 on base groups, 0.43 on
    groups of every release), or fails to load the tokenizer at all. Here every
    top-level key of the base ``config.json`` the export dropped is put back
    (except :data:`NOT_RESTORED_CONFIG_KEYS`) (keys
    the export wrote are kept), and the tokenizer files are the base's (RL does not
    change them). Returns what was restored."""
    import json
    import shutil

    restored: dict[str, Any] = {"config_keys": [], "tokenizer_files": []}
    base_config, config = base_dir / "config.json", directory / "config.json"
    if base_config.is_file() and config.is_file():
        base = json.loads(base_config.read_text(encoding="utf-8"))
        exported = json.loads(config.read_text(encoding="utf-8"))
        missing = [key for key in base if key not in exported and key not in NOT_RESTORED_CONFIG_KEYS]
        if missing:
            exported.update({key: base[key] for key in missing})
            config.write_text(json.dumps(exported, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            restored["config_keys"] = missing
    for name in TOKENIZER_FILES:
        source = base_dir / name
        if source.is_file():
            shutil.copyfile(source, directory / name)
            restored["tokenizer_files"].append(name)
    return restored


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
    from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
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

    with _awake_process_groups(actor):
        write_checkpoint_dir(Path(path), write, completion_marker=HF_EXPORT_COMPLETE_MARKER)
    return {"path": str(path), "writers": "all"}


def write_lora_adapter(actor: Any, *, path: str) -> dict[str, Any]:
    """Write the actor's current LoRA adapter into ``path`` (collective).

    Upstream's ``SnapshotPublisher`` writes models only; adapters go through
    the adapter library, exactly as ``hf_export.save_hf_model`` writes the
    ``adapter/`` directory next to an HF export: the sampler layout when the
    slots ask for it (``bf16_adapter_key_style == "sampler"``) or the
    parametrization has no plain PEFT form (fused modules with their own A,
    stacked experts), the PEFT layout otherwise. Rank 0 writes, every rank
    takes part in the gather; ``write_checkpoint_dir`` adds the completion
    marker the sampler's loader looks for."""
    slots = getattr(actor, "adapter_slots", None)
    if slots is None:
        raise RuntimeError("LoRA release export needs the actor's adapter slots "
                           "(Miles builds them for a LoRA actor)")
    args = actor.args
    from miles.backends.megatron_utils.lora.checkpoint import single_lora_parametrization
    from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
        write_peft_export,
        write_sampler_export,
    )
    from miles.backends.training_utils.checkpoint.adapter.adapter_exports import has_plain_peft_form
    from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
    from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

    parametrization = single_lora_parametrization(slots, args)
    write_export = (
        write_sampler_export
        if getattr(slots, "bf16_adapter_key_style", "peft") == "sampler"
        or not has_plain_peft_form(parametrization, slots.targets)
        else write_peft_export
    )

    def write(directory: Path) -> None:
        write_export(directory, slots, 0, parametrization, slots.targets, lora_dropout=args.lora_dropout)

    with _awake_process_groups(actor):
        write_checkpoint_dir(Path(path), write, completion_marker=HF_EXPORT_COMPLETE_MARKER)
    return {"path": str(path), "layout": "sampler" if write_export is write_sampler_export else "peft"}


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
