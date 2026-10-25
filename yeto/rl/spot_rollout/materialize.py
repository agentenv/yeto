"""Turn a subscriber's weight cut into a Hugging Face directory (S27 G-EA).

The syncer sends each fragment as bf16 values in layout order. This module
splits them by the trainer's layout (``layout_manifest``), joins the row
slices of a split parameter (``<wire name>#rows=a:b``), strips the wire name
``<role>::<shard id>::<Megatron name>`` and converts the Megatron model to HF
tensors with the same code as the offline evaluator
(:func:`yeto.rl.eval.cut_to_hf.convert_model`, Miles' ``megatron_to_hf``).
The engine then loads the directory (``update_weights_from_disk``).

Scope (refused otherwise): one role and one shard id in the layout, i.e. a
trainer with TP = PP = EP = 1 (the whole model in one rank's copy). A
sharded trainer needs the per-shard ``param_layout`` record (global names,
TP split) to merge slices, which the layout record does not carry yet.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from yeto.fragments import FragmentLayout
from yeto.rl.eval.cut_to_hf import (
    WEIGHT_FILE,
    _WEIGHT_SUFFIXES,
    MaterializeError,
    canonical_name,
    convert_model,
    megatron_args,
    miles_converter,
)

from .subscriber import WeightCut, bf16_fragment_tensors

DONE = "YETO-SPOT-WEIGHTS.json"
_ROWS = re.compile(r"^(?P<base>.*)#rows=(?P<start>\d+):(?P<stop>\d+)$")

Converter = Callable[[Any, str, Any], Any]


def _split_wire(name: str) -> tuple[str, str, str]:
    parts = name.split("::", 2)
    if len(parts) != 3:
        raise MaterializeError(f"{name}: not a <role>::<shard>::<name> wire name")
    return parts[0], parts[1], parts[2]


def megatron_model(layout: FragmentLayout, cut: WeightCut) -> dict[str, Any]:
    """``{module.module.<name>: bf16 tensor}`` of the whole model (module doc)."""
    import torch

    if len(cut.fragments) != layout.num_fragments:
        raise MaterializeError(f"cut has {len(cut.fragments)} fragments, layout {layout.num_fragments}")
    pieces: dict[str, list[tuple[int, Any, tuple[int, ...]]]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for fid, data in enumerate(cut.fragments):
        identity = layout.fragments[fid].identity_shapes
        if identity is None:
            raise MaterializeError(f"fragment {fid} has no tensor shapes (identity_shapes) in the layout")
        for name, flat in bf16_fragment_tensors(layout, fid, data):
            shape = tuple(int(d) for d in identity[name])
            match = _ROWS.match(name)
            base, start = (match["base"], int(match["start"])) if match else (name, 0)
            pieces.setdefault(base, []).append((start, flat, shape))
            shapes[base] = shape
    roles = {_split_wire(base)[0] for base in pieces}
    shards = {_split_wire(base)[1] for base in pieces}
    if len(roles) != 1 or len(shards) != 1:
        raise MaterializeError(
            f"the layout holds roles {sorted(roles)} and shards {sorted(shards)[:4]}; only a "
            "TP = PP = EP = 1 trainer (one role, one shard) can be materialized without param_layout")
    model: dict[str, Any] = {}
    for base, parts in pieces.items():
        parts.sort(key=lambda p: p[0])
        tensors = [flat.reshape(shape) for _start, flat, shape in parts]
        tensor = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        model[canonical_name(_split_wire(base)[2])] = tensor
    return model


def materialize(layout: FragmentLayout, cut: WeightCut, *, base_model: str | os.PathLike[str],
                out_root: str | os.PathLike[str], convert: Converter | None = None,
                model_name: str | None = None, keep: int = 2) -> Path:
    """Write ``<out_root>/v<merged>-<digest>`` (config/tokenizer from ``base_model``
    plus ``model.safetensors``) and return it. Keeps the newest ``keep`` such
    directories under ``out_root`` (this module's own outputs only)."""
    import hashlib

    import torch
    from safetensors.torch import save_file

    base = Path(base_model)
    config = json.loads((base / "config.json").read_text())
    if convert is None:
        convert = miles_converter(model_name or str(config["model_type"]))
    tag = hashlib.sha256(json.dumps(list(cut.versions)).encode()).hexdigest()[:12]
    target = Path(out_root) / f"v{cut.merged_version:08d}-{tag}"
    marker = target / DONE
    if marker.is_file():
        return target
    tensors = convert_model(megatron_model(layout, cut), megatron_args(config), convert, dtype=torch.bfloat16)
    if config.get("tie_word_embeddings"):
        tensors.pop("lm_head.weight", None)
    staging = target.with_name(f".{target.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for item in base.iterdir():
        if item.is_file() and not item.name.endswith(_WEIGHT_SUFFIXES) and item.name != "model.safetensors.index.json":
            shutil.copy2(item, staging / item.name)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(staging / WEIGHT_FILE),
              metadata={"format": "pt"})
    (staging / DONE).write_text(json.dumps({"versions": list(cut.versions),
                                            "merged_version": cut.merged_version,
                                            "tensors": len(tensors), "base_model": str(base)}))
    shutil.rmtree(target, ignore_errors=True)
    os.replace(staging, target)
    _prune(Path(out_root), keep=keep, current=target)
    return target


def _prune(root: Path, *, keep: int, current: Path) -> None:
    done = sorted((p for p in root.iterdir() if p.is_dir() and (p / DONE).is_file()),
                  key=lambda p: p.name)
    for old in done[:-max(1, int(keep))]:
        if old != current:
            shutil.rmtree(old, ignore_errors=True)


def identity_converter(_args: Any, name: str, tensor: Any) -> list[tuple[str, Any]]:
    """Tests: keep the Megatron name (minus ``module.module.``) as the HF name."""
    return [(name.removeprefix("module.module."), tensor)]


def converter_from_spec(spec: Mapping[str, Any]) -> Converter | None:
    target = spec.get("converter")
    if not target:
        return None
    import importlib

    module, _, attr = str(target).partition(":")
    return getattr(importlib.import_module(module), attr)
