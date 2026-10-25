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
2. writes them in the **source checkpoint's own layout** when it can
   (:func:`.hf_checkpoint_keys.base_export`, S27 G-EQ3): the same files, the
   same tensor names, the same dtypes, each file's safetensors header copied
   verbatim (:func:`write_like`), and the source's
   ``model.safetensors.index.json``.  The model's tensors go back through the
   ``transformers`` conversion table the loader applied (``nemotron_h``:
   ``model.`` -> ``backbone.``, a stacked expert tensor -> one tensor per
   expert), so an unchanged model exports byte-identical files.  Why: a
   ``release_delta`` only encodes a file that the base holds under the same
   name with the same size, and a sampler loads a release with the base's
   names (S27 G-EQ2 found the gap: Nemotron Nano loads ``backbone.*`` and
   6243 per-expert tensors into ``model.*`` and 46 stacked ones);
3. otherwise casts them to the model's own dtype (``config.json``
   ``torch_dtype`` or ``dtype``; bf16 when it names none) and writes one
   safetensors file per group of at most :data:`SHARD_BYTES` export bytes
   (:func:`plan_shards`), with ``model.safetensors.index.json`` when there is
   more than one, under the canonical names of :mod:`.full_parameter`; the
   island prints why the source's layout was not used;
4. copies the model's non-weight files (config, generation config, tokenizer,
   chat template) from the checkpoint the island loaded;
5. leaves the completion marker ``.complete`` last (``HfRoundExports`` takes
   it out and puts it back after the upload).

The tensors are the parameters and the persistent buffers (a router's
correction bias is a buffer; :mod:`.full_parameter`).  A tied embedding is one
parameter and stays one tensor, as ``save_pretrained`` writes it.  A LoRA
island has no full export: the island refuses ``--rl-export-hf-rounds``
before any work.
"""

from __future__ import annotations

import json
import shutil
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from yeto.rl.engine.hf_export import COMPLETE_MARKER

from .hf_checkpoint_keys import FLOAT_TAGS, BaseExport, base_export, checkpoint_tensor

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


def write_like(header: bytes, tensors: Mapping[str, Any], path: str | Path) -> int:
    """Write ``tensors`` as a safetensors file whose header is ``header``
    verbatim (a base file's own header bytes): each tensor's bytes go where the
    header's ``data_offsets`` put them, so equal values give a byte-identical
    file.  Refuses a tensor whose dtype or shape is not the header's, a missing
    or extra tensor, and a header whose data has gaps.  Returns the bytes written."""
    import torch

    entries = {name: entry for name, entry in json.loads(header).items() if name != "__metadata__"}
    if sorted(entries) != sorted(tensors):
        raise ValueError(f"{path}: the header names {len(entries)} tensors, the export has {len(tensors)}")
    at = 0
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        for name in sorted(entries, key=lambda n: entries[n]["data_offsets"][0]):
            entry, tensor = entries[name], tensors[name]
            start, stop = (int(v) for v in entry["data_offsets"])
            if start != at:
                raise ValueError(f"{path}: data of {name!r} starts at {start}, expected {at}")
            if str(tensor.dtype) != f"torch.{FLOAT_TAGS.get(entry['dtype'], entry['dtype'])}":
                raise ValueError(f"{path}: {name!r} is {tensor.dtype}, the header says {entry['dtype']}")
            if tuple(tensor.shape) != tuple(int(d) for d in entry["shape"]):
                raise ValueError(f"{path}: {name!r} has shape {tuple(tensor.shape)}, the header says "
                                 f"{entry['shape']}")
            data = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy()
            if data.nbytes != stop - start:
                raise ValueError(f"{path}: {name!r} is {data.nbytes} bytes, the header says {stop - start}")
            handle.write(memoryview(data))
            at = stop
    return 8 + len(header) + at


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
    island's private loop, so verl's own event loop is untouched).
    ``match_base_layout``: write the source checkpoint's own files when it can
    (module doc, step 2); ``last_layout`` says what the last export did."""

    def __init__(self, group: Any, model_path: str | Path, *, run: Any,
                 shard_bytes: int = SHARD_BYTES, match_base_layout: bool = True) -> None:
        self.group = group
        self.model_path = Path(model_path)
        self.run = run
        self.shard_bytes = int(shard_bytes)
        self.match_base_layout = bool(match_base_layout)
        self.dtype_name = export_dtype_name(self.model_path)
        self.last_layout: dict[str, Any] | None = None

    def __call__(self, rollout_id: int, directory: Path) -> None:
        from . import full_parameter as fp

        del rollout_id  # the directory names the round; the weights are the trainer's now
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        manifests = self.run(self.group.manifests())
        shapes = fp.agree_on_shapes(manifests)
        shapes.update(fp.agree_on_shapes(manifests, key="buffers"))
        plan, why = (base_export(self.model_path, shapes) if self.match_base_layout
                     else (None, "match_base_layout is off"))
        if plan is not None:
            total = self._write_base_layout(plan, directory)
        else:
            total = self._write_regrouped(shapes, directory)
        self.last_layout = {"layout": "base" if plan is not None else "regrouped", "why": why,
                            "bytes": total, "freed": self._release_memory()}
        print(f"[yeto hf export] device memory after the export: {self.last_layout['freed']}", flush=True)
        print(f"[yeto hf export] {self.last_layout['layout']} layout: {why}", flush=True)
        copy_model_files(self.model_path, directory)
        (directory / COMPLETE_MARKER).write_text("")

    def _release_memory(self, host: bool = True) -> dict | None:
        """Give the workers' cached device blocks back after the export
        (``full_parameter.remote_release_memory``; the gathers leave large blocks
        cached and the next training step needs the room). Returns the largest
        reserved bytes before and after, or None when the group cannot do it.
        ``host``: with ``YETO_HOST_MEMORY=trim`` also trim the workers' host heap
        (S27 G-EZ3); the per-file calls skip it, the one after the export does it."""
        release = getattr(self.group, "release_memory", None)
        if release is None:
            return None
        try:
            results = [dict(result) for result in (self.run(release(host=host)) or ())]
        except Exception as error:  # noqa: BLE001 -- the export matters, freeing is an optimisation
            print(f"[yeto hf export] could not free device memory: {error!r}", flush=True)
            return None
        if not results:
            return None
        freed = {"reserved_before": max(int(r["before"]["reserved"]) for r in results),
                 "reserved_after": max(int(r["after"]["reserved"]) for r in results),
                 "workers": len(results)}
        hosts = [r["host"] for r in results if r.get("host")]  # YETO_HOST_MEMORY=trim (S27 G-EZ3)
        if hosts:
            freed["host_rss_anon_mib_before"] = sum(int(h.get("rss_anon_mib_before") or 0) for h in hosts)
            freed["host_rss_anon_mib_after"] = sum(int(h.get("rss_anon_mib_after") or 0) for h in hosts)
            freed["host_trim_s_max"] = max(float(h.get("gc_s", 0)) + float(h.get("trim_s", 0)) for h in hosts)
        return freed

    def _write_base_layout(self, plan: BaseExport, directory: Path) -> int:
        """The source checkpoint's own files (module doc, step 2). A stacked model
        tensor that two files share is read once and kept until its last file."""
        import torch

        last_use = plan.last_use()
        cache: dict[str, Any] = {}
        total = 0
        for position, file_name in enumerate(plan.files):
            needed = plan.model_names(file_name)
            fetch = [name for name in needed if name not in cache]
            if fetch:
                exported = self.run(self.group.export_parameters(fetch))
                if sorted(exported) != sorted(fetch):
                    raise RuntimeError(f"verl HF export returned {sorted(exported)[:4]}..., expected {fetch[:4]}...")
                cache.update(exported)
                del exported
            tensors = {}
            for name in plan.names[file_name]:
                key = plan.key_map.keys[name]
                dtype = getattr(torch, FLOAT_TAGS[plan.dtypes[name]])
                tensors[name] = checkpoint_tensor(cache[key.model_name], key).to(dtype).contiguous()
            total += write_like(plan.headers[file_name], tensors, directory / file_name)
            del tensors
            for name in needed:
                if last_use[name] <= position:
                    cache.pop(name, None)
            # Free the device blocks of this file's gathers before the next one: the
            # step after an export has to fit on the same card, and on a 64 GiB card
            # the margin is under one activation group (S27 G-EQ3, npu1).
            self._release_memory(host=False)
        if plan.index:
            shutil.copyfile(plan.model_path / INDEX_FILE, directory / INDEX_FILE)
        return total

    def _write_regrouped(self, shapes: Mapping[str, Sequence[int]], directory: Path) -> int:
        """Groups of at most ``shard_bytes`` under the canonical names (module doc, step 3)."""
        import torch
        from safetensors.torch import save_file

        dtype = getattr(torch, self.dtype_name)
        itemsize = torch.empty((), dtype=dtype).element_size()
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
        return total


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
