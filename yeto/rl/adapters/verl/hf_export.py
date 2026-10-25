"""HF exports of a verl full-parameter island (``--rl-export-hf-rounds``, S27 G-DS).

The engine-neutral :class:`yeto.rl.engine.hf_export.HfRoundExports` decides
when a round is exported and puts the directory into the run's checkpoint
store; the backend gives it ``write(rollout_id, directory)``.  This module is
that function for a verl FSDP2 island (the Miles one is
``adapters/miles/release_plugin.full_checkpoint_writer``).

What the writer does, all on the island driver:

1. reads the FP32 master parameters from the actor workers in groups of at
   most :data:`SHARD_BYTES` (export dtype) with the full-parameter worker
   function the sync already uses (``VerlFullParameterGroup.export_parameters``:
   ``DTensor.full_tensor`` on every rank, rank 0 returns the tensors);
2. casts them to the model's own dtype (``config.json`` ``torch_dtype`` or
   ``dtype``; bf16 when it names none) and writes one safetensors file per
   group, with ``model.safetensors.index.json`` when there is more than one.
   The groups are the *source checkpoint's own* shard layout when its index
   names exactly the exported parameters (:func:`base_shard_layout`, S27 G-EQ:
   it keeps a later ``release_delta`` against that checkpoint small), otherwise
   groups of at most :data:`SHARD_BYTES` export bytes (:func:`plan_shards`);
3. copies the model's non-weight files (config, generation config, tokenizer,
   chat template) from the checkpoint the island loaded;
4. leaves the completion marker ``.complete`` last (``HfRoundExports`` takes
   it out and puts it back after the upload).

The names are the canonical HF names of :mod:`.full_parameter` (FSDP2 wrapper
segments removed), so ``transformers`` and vLLM load the directory as the
original model; a tied embedding is one parameter and stays one tensor, as
``save_pretrained`` writes it.  A LoRA island has no full export: the island
refuses ``--rl-export-hf-rounds`` before any work.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from yeto.rl.engine.hf_export import COMPLETE_MARKER

HF_EXPORT_STAGING = "yeto-hf-export"  # under the system temp dir (local disk)
SHARD_BYTES = 4 << 30  # export-dtype bytes per safetensors file
INDEX_FILE = "model.safetensors.index.json"
# Weight files of the source checkpoint (never copied: the export writes its own).
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack")
_DTYPES = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16", "fp16": "float16",
           "half": "float16", "float32": "float32", "fp32": "float32", "float": "float32"}


def export_dtype_name(model_path: str | Path) -> str:
    """The dtype the model's ``config.json`` names (``torch_dtype``, or ``dtype``
    in newer configs); ``bfloat16`` when it names none. Refuses a dtype it
    does not know rather than writing the wrong one."""
    config = json.loads((Path(model_path) / "config.json").read_text())
    named = config.get("torch_dtype", config.get("dtype"))
    if named is None:
        return "bfloat16"
    key = str(named).removeprefix("torch.").lower()
    if key not in _DTYPES:
        raise ValueError(f"the model config names dtype {named!r}; an HF export writes bf16, fp16 or fp32")
    return _DTYPES[key]


def base_shard_layout(model_path: str | Path, names: Sequence[str]) -> tuple[list[list[str]], list[str]] | None:
    """The source checkpoint's own shard layout for exactly ``names``, as
    ``(groups, file_names)``; None when it cannot be reused.

    Why (S27 G-EQ): ``release_delta`` only computes a delta for a weight file
    that the base release holds *under the same name with the same size*; every
    other file is copied whole. An export that regroups the tensors into
    :data:`SHARD_BYTES` files therefore turns the first delta against the
    original checkpoint into a full upload (59 GB for Nemotron Nano). Writing the
    same files the source checkpoint has keeps the delta small, which matters when
    the island uploads over a slow link (npu1: 14.6 MB/s to the mirror host).

    None (the caller falls back to :func:`plan_shards`) when there is no
    ``model.safetensors.index.json``, when its ``weight_map`` does not name
    exactly ``names``, or when it maps a name to a file that is not a
    safetensors shard. A single-file checkpoint has no index, so it also gets
    the fallback -- which writes one ``model.safetensors``, the same name.
    """
    index = Path(model_path) / INDEX_FILE
    if not index.is_file():
        return None
    try:
        weight_map = json.loads(index.read_text())["weight_map"]
    except (ValueError, KeyError):
        return None
    if not isinstance(weight_map, dict) or sorted(weight_map) != sorted(names):
        return None
    files = sorted(set(map(str, weight_map.values())))
    if not all(name.endswith(".safetensors") for name in files):
        return None
    groups = [sorted(n for n in names if str(weight_map[n]) == file_name) for file_name in files]
    return groups, files


def plan_shards(shapes: Mapping[str, Sequence[int]], *, itemsize: int,
                shard_bytes: int = SHARD_BYTES) -> list[list[str]]:
    """Names in sorted order, grouped so that each group's export bytes stay at
    most ``shard_bytes`` (a single larger tensor is a group of its own)."""
    if shard_bytes < 1:
        raise ValueError("shard_bytes must be positive")
    groups: list[list[str]] = []
    size = 0
    for name in sorted(shapes):
        numel = 1
        for dimension in shapes[name]:
            numel *= int(dimension)
        nbytes = numel * itemsize
        if groups and size + nbytes <= shard_bytes:
            groups[-1].append(name)
            size += nbytes
        else:
            groups.append([name])
            size = nbytes
    return groups


def copy_model_files(model_path: str | Path, directory: Path) -> list[str]:
    """Copy the top-level non-weight files of the source checkpoint (config,
    tokenizer, chat template, ...) into ``directory``; returns their names."""
    copied = []
    for path in sorted(Path(model_path).iterdir()):
        name = path.name
        if (not path.is_file() or name.startswith(".") or name.endswith(_WEIGHT_SUFFIXES)
                or name.endswith(".index.json")):
            continue
        shutil.copyfile(path, directory / name)
        copied.append(name)
    return copied


class VerlHfWriter:
    """``write(rollout_id, directory)`` of :class:`HfRoundExports` for one verl
    full-parameter island (module doc). ``group`` is the island's
    ``VerlFullParameterGroup``; ``run`` runs one coroutine to completion (the
    island's private loop, so verl's own event loop is untouched)."""

    def __init__(self, group: Any, model_path: str | Path, *, run: Any,
                 shard_bytes: int = SHARD_BYTES, match_base_layout: bool = True) -> None:
        self.group = group
        self.model_path = Path(model_path)
        self.run = run
        self.shard_bytes = int(shard_bytes)
        self.match_base_layout = bool(match_base_layout)
        self.dtype_name = export_dtype_name(self.model_path)

    def __call__(self, rollout_id: int, directory: Path) -> None:
        import torch
        from safetensors.torch import save_file

        from . import full_parameter as fp

        del rollout_id  # the directory names the round; the weights are the trainer's now
        dtype = getattr(torch, self.dtype_name)
        itemsize = torch.empty((), dtype=dtype).element_size()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        shapes = fp.agree_on_shapes(self.run(self.group.manifests()))
        layout = base_shard_layout(self.model_path, sorted(shapes)) if self.match_base_layout else None
        if layout is not None:  # the source checkpoint's own files (delta-friendly)
            groups, files = layout
        else:
            groups = plan_shards(shapes, itemsize=itemsize, shard_bytes=self.shard_bytes)
            files = (["model.safetensors"] if len(groups) == 1 else
                     [f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors" for i in range(len(groups))])
        weight_map: dict[str, str] = {}
        total = 0
        for names, file_name in zip(groups, files, strict=True):
            exported = self.run(self.group.export_parameters(names))
            if sorted(exported) != sorted(names):
                raise RuntimeError(f"verl HF export returned {sorted(exported)[:4]}..., expected {names[:4]}...")
            tensors = {name: exported[name].to(dtype).contiguous() for name in names}
            del exported
            save_file(tensors, str(directory / file_name), metadata={"format": "pt"})
            for name, tensor in tensors.items():
                weight_map[name] = file_name
                total += tensor.numel() * tensor.element_size()
            del tensors
        if len(files) > 1:
            (directory / INDEX_FILE).write_text(json.dumps(
                {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))
        copy_model_files(self.model_path, directory)
        (directory / COMPLETE_MARKER).write_text("")


def hf_round_exports(config: Mapping[str, Any], *, group: Any, model_path: str | Path, run: Any,
                     environ: Mapping[str, str] | None = None):
    """The island's ``HfRoundExports`` from the plan's ``hf_exports`` section
    (``{"rounds", "store"}``, :func:`.full_parameter_stream.full_parameter_extras`)."""
    import tempfile

    from yeto.rl.engine.hf_export import ExportRounds, HfRoundExports
    from yeto.rl.engine.resume import store_for

    return HfRoundExports(ExportRounds.parse(config["rounds"]), store_for(config["store"], environ=environ),
                          VerlHfWriter(group, model_path, run=run),
                          staging=Path(tempfile.gettempdir()) / HF_EXPORT_STAGING)
