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

Block-scaled FP8 bases (DeepSeek-V4.1-Flash: attention and shared expert in MXFP8,
``<module>.weight`` float8_e4m3fn with a ``<module>.scale`` float8_e8m0fnu per 32x32
block): the weight is dequantized (``W = q * 2^(code - 127)`` per block), the delta is
added in float32, and every block whose delta is not all zero is re-quantized the way
vLLM quantizes MXFP8 (``_mxfp8_e4m3_quantize_torch``: code = ``ceil(log2(amax / 448))
+ 127``, clamped to [0, 254]); the new scale is written next to the weight. A block
whose delta is zero keeps its stored bytes and scale, so a zero adapter is bit-exact.
Each re-quantized block moves by at most half an FP8 step at its new scale.

Refused, before anything is written (:class:`LoraMergeError`, ``reason``):

* ``quantized_base``: the base stores the module quantized (``weight_packed``, e.g. a
  Kimi-K3.1 MXFP4 routed expert). Merging would re-quantize it, which changes every
  weight of the matrix, not only the adapted direction; serve such an adapter as a
  bypass (SGLang) or train without routed-expert targets;
* ``stacked``: a 3-D (stacked-experts) adapter tensor: the same case, the routed experts;
* ``quant_format``: any other quantized storage of an adapted weight (an FP8 weight
  without an e8m0 block scale, float32 / per-tensor scales, e4m3fnuz, int8, a scale
  that does not tile the weight, NaN scale codes, zero points or global scales):
  merging it as a plain tensor would silently corrupt it;
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


# the names a block scale of ``<module>.weight`` is stored under (DeepSeek-native
# ``.scale``; HF / Quark exports ``.weight_scale`` / ``.weight_scale_inv``)
SCALE_SUFFIXES = (".scale", ".weight_scale", ".weight_scale_inv")
# quantization side tensors this merge does not understand: fail loud, never ignore
UNKNOWN_QUANT_SUFFIXES = (".input_scale", ".weight_zero_point", ".weight_global_scale",
                          ".weight_scale_2", ".weight_shape", ".zeros", ".qweight", ".scales")
FP8_MAX = 448.0


def _resolve(module: str, weight_map: Mapping[str, str]) -> tuple[str, str | None]:
    """(base weight name, its block-scale name or None) of an adapter module:
    ``<module>.weight``, also when the base nests the text stack one level deeper or
    shallower (``language_model.``)."""
    candidates = [module, f"language_model.{module}", module.removeprefix("language_model.")]
    for name in dict.fromkeys(candidates):
        if f"{name}.weight_packed" in weight_map:
            raise LoraMergeError("quantized_base", f"{name} is stored quantized (weight_packed); merging "
                                 "would re-quantize it: serve this adapter as a bypass (SGLang)")
        if f"{name}.weight" in weight_map:
            unknown = [f"{name}{s}" for s in UNKNOWN_QUANT_SUFFIXES if f"{name}{s}" in weight_map]
            if unknown:
                raise LoraMergeError("quant_format", f"{name} carries quantization tensors {unknown} this "
                                     "merge does not handle")
            scales = [f"{name}{s}" for s in SCALE_SUFFIXES if f"{name}{s}" in weight_map]
            if len(scales) > 1:
                raise LoraMergeError("quant_format", f"{name} has several scale tensors {scales}")
            return f"{name}.weight", (scales[0] if scales else None)
    raise LoraMergeError("unknown_module", f"the base has no weight for adapter module {module}")


def _config_block(base_dir: Path) -> tuple[int, int] | None:
    path = Path(base_dir) / "config.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text())
    quant = config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config") or {}
    block = quant.get("weight_block_size")
    return (int(block[0]), int(block[1])) if isinstance(block, list) and len(block) == 2 else None


def _scale_block(name: str, weight_shape: tuple[int, ...], scale_shape: tuple[int, ...],
                 config_block: tuple[int, int] | None) -> tuple[int, int]:
    """The (rows, cols) block one scale entry covers: the config's
    ``weight_block_size`` when the scale has its shape (padded edge blocks
    included), else the block that tiles the weight exactly (e.g. a 1x32 ``wo_a``);
    anything else is refused."""
    if len(weight_shape) != 2 or len(scale_shape) != 2:
        raise LoraMergeError("quant_format", f"{name}: weight {weight_shape} / scale {scale_shape} are not 2-D")
    (rows, cols), (srows, scols) = weight_shape, scale_shape
    if config_block is not None:
        br, bc = config_block
        if -(-rows // br) == srows and -(-cols // bc) == scols:
            return br, bc
    if srows and scols and rows % srows == 0 and cols % scols == 0:
        return rows // srows, cols // scols
    raise LoraMergeError("quant_format", f"{name}: scale {scale_shape} does not tile weight {weight_shape} "
                         f"(config weight_block_size {config_block})")


def _blocks(x: Any, block: tuple[int, int]) -> Any:
    """``x`` (rows, cols) zero-padded to whole blocks, as (srows, scols, br, bc)."""
    import torch.nn.functional as F

    (rows, cols), (br, bc) = x.shape, block
    srows, scols = -(-rows // br), -(-cols // bc)
    padded = F.pad(x, (0, scols * bc - cols, 0, srows * br - rows))
    return padded.view(srows, br, scols, bc).permute(0, 2, 1, 3)


def _unblock(x: Any, shape: tuple[int, int]) -> Any:
    srows, scols, br, bc = x.shape
    return x.permute(0, 2, 1, 3).reshape(srows * br, scols * bc)[: shape[0], : shape[1]]


def _check_mxfp8(name: str, weight: Any, scale: Any) -> Any:
    """The uint8 e8m0 codes of ``scale``; refuses every other storage."""
    import torch

    if weight.dtype != torch.float8_e4m3fn:
        raise LoraMergeError("quant_format", f"{name}: a scaled weight of dtype {weight.dtype} (only "
                             "float8_e4m3fn with an e8m0 block scale is merged)")
    if scale.dtype != torch.float8_e8m0fnu:
        raise LoraMergeError("quant_format", f"{name}: scale dtype {scale.dtype} (only float8_e8m0fnu "
                             "block scales are merged)")
    codes = scale.view(torch.uint8)
    if bool((codes == 255).any()):
        raise LoraMergeError("quant_format", f"{name}: the scale holds NaN e8m0 codes (255)")
    return codes


def dequantize_mxfp8(weight: Any, scale: Any, block: tuple[int, int]) -> Any:
    """float32 ``weight * 2^(code - 127)`` per block (exact)."""
    import torch

    codes = _check_mxfp8("weight", weight, scale)
    factor = torch.exp2(codes.to(torch.float32) - 127.0)[:, :, None, None]
    return _unblock(_blocks(weight.to(torch.float32), block) * factor, tuple(weight.shape))


def quantize_mxfp8(x: Any, block: tuple[int, int]) -> tuple[Any, Any]:
    """(float8_e4m3fn weight, float8_e8m0fnu scale) of float32 ``x``, one scale per
    block, as vLLM's ``_mxfp8_e4m3_quantize_torch``: code = ``ceil(log2(amax/448)) +
    127`` clamped to [0, 254], values ``x / 2^(code-127)`` rounded to E4M3."""
    import torch

    blocked = _blocks(x.to(torch.float32), block)
    amax = blocked.abs().amax(dim=(2, 3)).clamp(min=torch.finfo(torch.float32).tiny)
    codes = (torch.ceil(torch.log2(amax / FP8_MAX)) + 127.0).clamp(0, 254)
    scaled = blocked / torch.exp2(codes - 127.0)[:, :, None, None]
    q = _unblock(scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), tuple(x.shape))
    return q, codes.to(torch.uint8).view(torch.float8_e8m0fnu)


def merge_mxfp8(name: str, weight: Any, scale: Any, delta: Any, block: tuple[int, int]) -> tuple[Any, Any]:
    """(weight, scale) of ``dequant(weight, scale) + delta`` re-quantized block by
    block; a block whose delta is all zero keeps its stored bytes and scale."""
    import torch

    codes = _check_mxfp8(name, weight, scale)
    merged = dequantize_mxfp8(weight, scale, block) + delta.to(torch.float32)
    q, new_scale = quantize_mxfp8(merged, block)
    changed = _blocks(delta.to(torch.float32), block).ne(0).flatten(2).any(dim=2)
    keep = ~changed
    new_codes = new_scale.view(torch.uint8).clone()
    new_codes[keep] = codes[keep]
    q_bytes = _blocks(q.view(torch.uint8), block).clone()
    q_bytes[keep] = _blocks(weight.view(torch.uint8), block)[keep]
    out = _unblock(q_bytes, tuple(weight.shape)).contiguous().view(torch.float8_e4m3fn)
    return out, new_codes.view(torch.float8_e8m0fnu)


Plan = dict[str, tuple[str, Any, Any, str | None, str | None]]


def plan_merge(base_dir: Path, adapter_tensors: Mapping[str, Any]) -> Plan:
    """{base tensor: (base file, A, B, scale tensor | None, scale file | None)};
    every refusal by name happens here, by dtype / layout before anything is written."""
    weight_map = json.loads((Path(base_dir) / INDEX_FILE).read_text())["weight_map"]
    plan: Plan = {}
    for module, (a, b) in lora_pairs(adapter_tensors).items():
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise LoraMergeError("stacked", f"{module}: {len(a.shape)}-D / {len(b.shape)}-D factors are stacked "
                                 "routed experts; merging them re-quantizes the MXFP4 experts")
        if a.shape[0] != b.shape[1]:
            raise LoraMergeError("shape", f"{module}: A {tuple(a.shape)} and B {tuple(b.shape)} ranks differ")
        tensor, scale = _resolve(module, weight_map)
        plan[tensor] = (weight_map[tensor], a, b, scale, weight_map[scale] if scale else None)
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
    for tensor, (file, _a, _b, scale_name, scale_file) in plan.items():
        by_file.setdefault(file, []).append(tensor)
        if scale_name is not None:
            by_file.setdefault(scale_file, []).append(scale_name)
    base_tensors: dict[str, Any] = {}
    for file, names in sorted(by_file.items()):
        base_tensors.update(read(base / file, list(dict.fromkeys(names))))
    config_block = _config_block(base)
    plain = (torch.bfloat16, torch.float16, torch.float32)
    for name, (_file, a, b, scale_name, _scale_file) in sorted(plan.items()):
        weight = base_tensors[name]
        if tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise LoraMergeError("shape", f"{name}: base {tuple(weight.shape)} != B@A "
                                 f"({b.shape[0]}, {a.shape[1]})")
        delta = scale * (b.to(torch.float32) @ a.to(torch.float32))
        if scale_name is None:
            if weight.dtype not in plain:
                raise LoraMergeError("quant_format", f"{name}: dtype {weight.dtype} without a block scale "
                                     "(only bf16 / fp16 / fp32, or fp8 with an e8m0 scale, are merged)")
            merged[name] = (weight.to(torch.float32) + delta).to(weight.dtype)
            continue
        block_scale = base_tensors[scale_name]
        block = _scale_block(name, tuple(weight.shape), tuple(block_scale.shape), config_block)
        merged[name], merged[scale_name] = merge_mxfp8(name, weight, block_scale, delta, block)
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


__all__ = ["LoraMergeError", "adapter_scale", "dequantize_mxfp8", "lora_pairs", "merge_lora_release",
           "merge_mxfp8", "plan_merge", "quantize_mxfp8"]
