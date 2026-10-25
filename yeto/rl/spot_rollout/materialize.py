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

S27 G-EV: a verl full-parameter trainer's layout names its parameters by their
HF names under the shard id ``global`` (``adapters/verl/full_parameter.py``: the
FSDP2 layout carries no shard identity). Those names are written as they are,
without the Megatron conversion (:func:`hf_named_layout`).
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
    check_base_names,
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


# The shard id of a layout whose names are the model's HF names (verl's FSDP2 layout).
HF_NAMED_SHARD = "global"


def hf_named_layout(layout: FragmentLayout) -> bool:
    """Every wire name of ``layout`` is ``<role>::global::<HF name>`` (verl, module doc)."""
    shards = {_split_wire(name)[1] for fragment in layout.fragments for name, _numel in fragment.tensors}
    return shards == {HF_NAMED_SHARD}


def megatron_model(layout: FragmentLayout, cut: WeightCut) -> dict[str, Any]:
    """``{module.module.<name>: bf16 tensor}`` of the whole model (module doc)."""
    return {canonical_name(name): tensor for name, tensor in _whole_tensors(layout, cut).items()}


def hf_model(layout: FragmentLayout, cut: WeightCut) -> dict[str, Any]:
    """``{HF name: bf16 tensor}`` of a :func:`hf_named_layout` cut (S27 G-EV)."""
    if not hf_named_layout(layout):
        raise MaterializeError(f"the layout does not name HF parameters (shard {HF_NAMED_SHARD!r})")
    return _whole_tensors(layout, cut)


def _whole_tensors(layout: FragmentLayout, cut: WeightCut) -> dict[str, Any]:
    """``{name: bf16 tensor}``: each parameter's row slices joined, the wire prefix stripped."""
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
        model[_split_wire(base)[2]] = tensor
    return model


def materialize(layout: FragmentLayout, cut: WeightCut, *, base_model: str | os.PathLike[str],
                out_root: str | os.PathLike[str], convert: Converter | None = None,
                model_name: str | None = None, keep: int = 2) -> Path:
    """Write ``<out_root>/v<merged>-<digest>`` (config/tokenizer from ``base_model``
    plus ``model.safetensors``) and return it. Keeps the newest ``keep`` such
    directories under ``out_root`` (this module's own outputs only). The HF names
    must equal the base model's weight names (``cut_to_hf.check_base_names``)."""
    import hashlib

    import torch
    from safetensors.torch import save_file

    base = Path(base_model)
    config = json.loads((base / "config.json").read_text())
    hf_names = convert is None and hf_named_layout(layout)  # S27 G-EV: verl names the HF parameters
    if convert is None and not hf_names:
        convert = miles_converter(model_name or str(config["model_type"]))
    tag = hashlib.sha256(json.dumps(list(cut.versions)).encode()).hexdigest()[:12]
    target = Path(out_root) / f"v{cut.merged_version:08d}-{tag}"
    marker = target / DONE
    if marker.is_file():
        return target
    if hf_names:
        tensors = {name: tensor.to(torch.bfloat16) for name, tensor in hf_model(layout, cut).items()}
    else:
        tensors = convert_model(megatron_model(layout, cut), megatron_args(config), convert, dtype=torch.bfloat16)
    if config.get("tie_word_embeddings"):
        tensors.pop("lm_head.weight", None)
    # every base weight is written (a missing one would keep its old value in the engine
    # after update_weights_from_disk, without an error) and no other name
    check_base_names(tensors, base, config, f"cut v{cut.merged_version}")
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
