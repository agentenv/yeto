"""A servable checkpoint from a base checkpoint and a LoRA release, for an engine
that cannot serve adapters (the vLLM ``kimi-k31`` images: neither the AMD nor the
NVIDIA ``KimiK3ForConditionalGeneration`` implements ``SupportsLoRA``, so
``--enable-lora`` refuses the model).

:func:`merge_lora_release` folds each adapted module into its base weight,
``W' = W + (lora_alpha / r) * B @ A``, summed in float32 and stored in the base
tensor's dtype, writes only the merged tensors (one safetensors file and an index)
into a partial directory, and lays it over the base with
:func:`.release_overlay.overlay_release` (symlinks; nothing else is copied). Every
tensor the adapter does not touch stays byte-identical, including the MXFP4 routed
experts.

Refused, before anything is written (:class:`LoraMergeError`, ``reason``):

* ``quantized_base``: the base stores the module quantized (``weight_packed``, e.g. a
  Kimi-K3.1 MXFP4 routed expert). Merging would re-quantize it, which changes every
  weight of the matrix, not only the adapted direction; serve such an adapter as a
  bypass (SGLang) or train without routed-expert targets;
* ``stacked``: a 3-D (stacked-experts) adapter tensor: the same case, the routed experts;
* ``unknown_module``: a module the base lacks; ``unpaired`` / ``shape``: an A without
  B or factors that do not fit the base weight. A silently skipped module would serve
  the base weights for it.

Rounding: a BF16 base keeps about 8 mantissa bits, so a delta far below ``|W|/256``
does not change the stored weight; the trainer's log-prob recomputation shows the
difference as rollout/train mismatch.

Tensor IO is injected (``read`` / ``write``) so CPU tests run on small files; the
defaults use ``safetensors.torch``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .release_overlay import INDEX_FILE, overlay_release

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
MERGED_FILE = "lora-merged.safetensors"
PEFT_PREFIX = "base_model.model."
MERGE_RECORD = "yeto_lora_merge.json"


class LoraMergeError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


ReadTensors = Callable[[Path, Sequence[str] | None], dict[str, Any]]
WriteTensors = Callable[[dict[str, Any], Path], None]


def _read_safetensors(path: Path, names: Sequence[str] | None = None) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys()) if names is None else list(names)
        return {key: handle.get_tensor(key) for key in keys}


def _write_safetensors(tensors: dict[str, Any], path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))


def adapter_scale(adapter_dir: Path) -> float:
    config = json.loads((Path(adapter_dir) / ADAPTER_CONFIG).read_text())
    rank = int(config.get("r") or 0)
    if rank <= 0:
        raise LoraMergeError("config", f"{ADAPTER_CONFIG} has rank {config.get('r')!r}")
    alpha = config.get("lora_alpha")
    return float(rank if alpha is None else alpha) / rank


def lora_pairs(tensors: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """{module: (A, B)} of an adapter's tensors (``<module>.lora_{A,B}[.weight]``,
    PEFT prefix stripped)."""
    sides: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        key = name.removeprefix(PEFT_PREFIX).removesuffix(".weight")
        for side in ("A", "B"):
            suffix = f".lora_{side}"
            if key.endswith(suffix):
                sides.setdefault(key[: -len(suffix)], {})[side] = tensor
                break
        else:
            raise LoraMergeError("unknown_tensor", f"{name} is not a lora_A / lora_B tensor")
    out = {}
    for module, pair in sorted(sides.items()):
        if set(pair) != {"A", "B"}:
            raise LoraMergeError("unpaired", f"{module} has only lora_{sorted(pair)}")
        out[module] = (pair["A"], pair["B"])
    return out


def _resolve(module: str, weight_map: Mapping[str, str]) -> str:
    """The base tensor name of an adapter module: ``<module>.weight``, also when the
    base nests the text stack one level deeper or shallower (``language_model.``)."""
    candidates = [module, f"language_model.{module}", module.removeprefix("language_model.")]
    for name in dict.fromkeys(candidates):
        if f"{name}.weight" in weight_map:
            return f"{name}.weight"
        if f"{name}.weight_packed" in weight_map:
            raise LoraMergeError("quantized_base", f"{name} is stored quantized (weight_packed); merging "
                                 "would re-quantize it: serve this adapter as a bypass (SGLang)")
    raise LoraMergeError("unknown_module", f"the base has no weight for adapter module {module}")


def plan_merge(base_dir: Path, adapter_tensors: Mapping[str, Any]) -> dict[str, tuple[str, Any, Any]]:
    """{base tensor: (base file, A, B)}; every refusal happens here."""
    weight_map = json.loads((Path(base_dir) / INDEX_FILE).read_text())["weight_map"]
    plan = {}
    for module, (a, b) in lora_pairs(adapter_tensors).items():
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise LoraMergeError("stacked", f"{module}: {len(a.shape)}-D / {len(b.shape)}-D factors are stacked "
                                 "routed experts; merging them re-quantizes the MXFP4 experts")
        if a.shape[0] != b.shape[1]:
            raise LoraMergeError("shape", f"{module}: A {tuple(a.shape)} and B {tuple(b.shape)} ranks differ")
        tensor = _resolve(module, weight_map)
        plan[tensor] = (weight_map[tensor], a, b)
    if not plan:
        raise LoraMergeError("empty", "the adapter has no LoRA tensors")
    return plan


def merge_lora_release(base_dir: str | Path, adapter_dir: str | Path, out_dir: str | Path, *,
                       read: ReadTensors = _read_safetensors,
                       write: WriteTensors = _write_safetensors) -> Path:
    """Build ``out_dir`` = ``base_dir`` with the adapter of ``adapter_dir`` merged
    (module doc). Returns ``out_dir``. ``out_dir`` and ``<out_dir>.partial`` are replaced."""
    import torch

    base, adapter, out = Path(base_dir).resolve(), Path(adapter_dir), Path(out_dir)
    scale = adapter_scale(adapter)
    plan = plan_merge(base, read(adapter / ADAPTER_WEIGHTS, None))
    merged: dict[str, Any] = {}
    by_file: dict[str, list[str]] = {}
    for tensor, (file, _a, _b) in plan.items():
        by_file.setdefault(file, []).append(tensor)
    for file, names in sorted(by_file.items()):
        for name, weight in read(base / file, names).items():
            _file, a, b = plan[name]
            if tuple(weight.shape) != (b.shape[0], a.shape[1]):
                raise LoraMergeError("shape", f"{name}: base {tuple(weight.shape)} != B@A "
                                     f"({b.shape[0]}, {a.shape[1]})")
            delta = b.to(torch.float32) @ a.to(torch.float32)
            merged[name] = (weight.to(torch.float32) + scale * delta).to(weight.dtype)
    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    write(merged, partial / MERGED_FILE)
    (partial / INDEX_FILE).write_text(json.dumps(
        {"metadata": {}, "weight_map": {name: MERGED_FILE for name in sorted(merged)}}, indent=1))
    overlay_release(base, partial, out)
    (out / MERGE_RECORD).write_text(json.dumps(
        {"adapter": str(adapter), "scale": scale, "merged": sorted(merged)}, indent=1))
    return out


__all__ = ["LoraMergeError", "adapter_scale", "lora_pairs", "merge_lora_release", "plan_merge"]
