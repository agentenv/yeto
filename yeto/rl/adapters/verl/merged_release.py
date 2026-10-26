"""A verl policy as a full HF checkpoint for a sampler island (S25 C2; full parameters S27 G-EK).

The engine sampler island loads full checkpoints only (``SamplerEngine.load``:
the SGLang and vllm-ascend adapters reload weights from a model directory). A
verl buffer trainer trains a LoRA adapter over a frozen base, so each release
merges the adapter into the base: ``W' = W + B @ A`` for every LoRA module
(yeto canonical LoRA has no scaling: ``lora_alpha == rank``,
``config.validate``), summed in float32 and stored in the base tensor's dtype.

The output has the base's file layout (the same safetensors file names, the same
index, the same non-weight files: config, tokenizer, generation config), so the
engine reads it exactly as it reads the base. A module the adapter names that
the base lacks, or an A/B pair with mismatched shapes, is an error: a silently
skipped module would serve the base weights for it.

Rounding: a bf16 base keeps about 8 bits of mantissa, so a LoRA delta far below
``|W| / 256`` does not change the stored weight. The sampler then serves the
merged weights, not the trainer's exact ``W + BA``; the trainer's log-prob
recomputation shows the difference as off-policy mismatch.

Full parameters (S27 G-EK, VERL-PARITY #14): :func:`write_full_checkpoint` puts a
full-parameter trainer's weights into a release directory with the island's HF
writer (``hf_export.VerlHfWriter``: the FP32 masters read from the trainer
workers, cast to the model's dtype, the base's non-weight files copied). Nothing
is merged: the trainer holds the whole model.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .param_names import CANONICAL_PREFIX, ParamNameError

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
LORA_A, LORA_B = ".lora_A.weight", ".lora_B.weight"


class MergeError(ValueError):
    """The adapter does not fit the base checkpoint."""


def lora_pairs(lora: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """{base weight name: (A, B)} from canonical LoRA tensors
    (``base_model.model.<module>.lora_{A,B}.weight``)."""
    pairs: dict[str, dict[str, Any]] = {}
    for name, tensor in lora.items():
        if not name.startswith(CANONICAL_PREFIX) or not name.endswith((LORA_A, LORA_B)):
            raise ParamNameError(f"not a canonical LoRA name: {name!r}")
        side = "A" if name.endswith(LORA_A) else "B"
        module = name[len(CANONICAL_PREFIX):-len(LORA_A)]
        pairs.setdefault(f"{module}.weight", {})[side] = tensor
    out = {}
    for weight, sides in sorted(pairs.items()):
        if set(sides) != {"A", "B"}:
            raise MergeError(f"{weight}: LoRA has only side {sorted(sides)}")
        out[weight] = (sides["A"], sides["B"])
    return out


def weight_files(base_dir: Path) -> dict[str, list[str]]:
    """{safetensors file: [tensor names]} of a HF checkpoint directory."""
    index = base_dir / INDEX_FILE
    if index.is_file():
        files: dict[str, list[str]] = {}
        for name, file in json.loads(index.read_text(encoding="utf-8"))["weight_map"].items():
            files.setdefault(file, []).append(name)
        return files
    if (base_dir / SINGLE_FILE).is_file():
        from safetensors import safe_open

        with safe_open(str(base_dir / SINGLE_FILE), framework="pt") as handle:
            return {SINGLE_FILE: list(handle.keys())}
    raise MergeError(f"{base_dir}: no {INDEX_FILE} and no {SINGLE_FILE}")


def write_merged_checkpoint(base_dir: str | Path, lora: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Write base + LoRA into ``out_dir`` (module doc); returns a summary
    (merged modules, files, the largest relative change of a merged weight)."""
    import torch
    from safetensors.torch import load_file, save_file

    base, out = Path(base_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = lora_pairs(lora)
    files = weight_files(base)
    known = {name for names in files.values() for name in names}
    missing = sorted(set(pairs) - known)
    if missing:
        raise MergeError(f"the base has no weight for {len(missing)} LoRA modules, e.g. {missing[0]}")
    merged, max_rel = 0, 0.0
    for file in sorted(files):
        tensors = load_file(str(base / file))
        for name in files[file]:
            if name not in pairs:
                continue
            a, b = (t.detach().to("cpu", torch.float32) for t in pairs[name])
            weight = tensors[name]
            if a.dim() != 2 or b.dim() != 2 or b.shape[1] != a.shape[0] or \
                    (b.shape[0], a.shape[1]) != tuple(weight.shape):
                raise MergeError(f"{name}: A {tuple(a.shape)}, B {tuple(b.shape)} do not fit "
                                 f"the weight {tuple(weight.shape)}")
            delta = b @ a
            new = (weight.to(torch.float32) + delta).to(weight.dtype)
            norm = float(weight.to(torch.float32).norm())
            if norm > 0:
                max_rel = max(max_rel, float((new.to(torch.float32) - weight.to(torch.float32)).norm()) / norm)
            tensors[name] = new.contiguous()
            merged += 1
        save_file(tensors, str(out / file), metadata={"format": "pt"})
    for item in sorted(base.iterdir()):
        if item.is_file() and item.name not in files and not item.name.startswith("."):
            shutil.copy2(item, out / item.name)
    return {"merged_modules": merged, "files": sorted(files), "max_relative_change": max_rel}


def write_full_checkpoint(writer: Any, out_dir: str | Path) -> dict[str, Any]:
    """Write a full-parameter policy into ``out_dir`` with ``writer(rollout_id, directory)``
    (``hf_export.VerlHfWriter``). The writer creates its directory itself and leaves the
    HF export completion marker, so it writes into a fresh sibling; the files then move
    into ``out_dir`` (the release store may have created it) without the marker.
    Returns a summary (files, bytes)."""
    from yeto.rl.engine.hf_export import COMPLETE_MARKER

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    staged = out.parent / f".{out.name}.full-write"
    if staged.exists():
        shutil.rmtree(staged)
    try:
        writer(0, staged)
        files = {}
        for item in sorted(staged.iterdir()):
            if item.name == COMPLETE_MARKER:
                continue
            if (out / item.name).exists():
                raise MergeError(f"{out}: {item.name} exists already")
            files[item.name] = item.stat().st_size
            item.rename(out / item.name)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    return {"files": sorted(files), "bytes": sum(files.values())}
