"""LoRA merge into a block-scaled FP8 base (DeepSeek-V4.1-Flash: MXFP8 attention and
shared expert, ``<module>.weight`` float8_e4m3fn + ``<module>.scale`` float8_e8m0fnu
per 32x32 block) for an engine without adapter support (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from yeto.rl.engine.lora_release_merge import (
    LoraMergeError,
    dequantize_mxfp8,
    merge_lora_release,
    merge_mxfp8,
    quantize_mxfp8,
)

WQ_A = "layers.0.attn.wq_a"
WO_A = "layers.0.attn.wo_a"
W2 = "layers.0.ffn.shared_experts.w2"
EXPERT = "layers.0.ffn.experts.3.w1"
BLOCK = (32, 32)
SHAPES = {WQ_A: (64, 96), WO_A: (40, 64), W2: (96, 64)}  # wo_a: a padded edge block row


def _vllm_mxfp8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's ``_mxfp8_e4m3_quantize_torch`` (1x32 blocks along the last dim), copied
    as the reference of the quantization convention."""
    m, k = x.shape
    blocked = x.float().view(m, k // 32, 32)
    amax = blocked.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    biased = (torch.ceil(torch.log2(amax / torch.finfo(torch.float8_e4m3fn).max)) + 127.0).clamp(0, 254)
    q = (blocked / torch.exp2(biased - 127.0).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    return q, biased.to(torch.uint8)


def _mxfp8(shape: tuple[int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    # per-block magnitudes over a few binades, as trained weights have
    w = torch.randn(shape) * torch.exp2(torch.randint(-8, -2, (1,)).float())
    return quantize_mxfp8(w, BLOCK)


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    tensors = {}
    for i, (module, shape) in enumerate(SHAPES.items()):
        tensors[f"{module}.weight"], tensors[f"{module}.scale"] = _mxfp8(shape, i)
    experts = {f"{EXPERT}.weight": torch.randint(0, 255, (64, 16), dtype=torch.uint8).view(torch.int8),
               f"{EXPERT}.scale": torch.randint(100, 130, (64, 1), dtype=torch.uint8).view(torch.float8_e8m0fnu)}
    save_file(tensors, str(base / "model-00001.safetensors"))
    save_file(experts, str(base / "model-00002.safetensors"))
    weight_map = {**{k: "model-00001.safetensors" for k in tensors}, **{k: "model-00002.safetensors" for k in experts}}
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (base / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "expert_dtype": "fp4",
                                                  "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                                                                          "scale_fmt": "ue8m0",
                                                                          "weight_block_size": [32, 32]}}))
    return base


def _adapter(path: Path, tensors: dict, *, rank: int = 2, alpha: int = 4) -> Path:
    path.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha, "peft_type": "LORA"}))
    return path


def _lora(modules, *, zero: bool = False, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    out = {}
    for module in modules:
        rows, cols = SHAPES[module]
        out[f"{module}.lora_A.weight"] = torch.randn(2, cols) * 0.05
        out[f"{module}.lora_B.weight"] = torch.zeros(rows, 2) if zero else torch.randn(rows, 2) * 0.05
    return out


def _within_fp8_step(merged: torch.Tensor, want: torch.Tensor, scale: torch.Tensor, block) -> None:
    """Each element within half an E4M3 step at its block's scale (3 mantissa bits:
    relative 2^-4; subnormals: 2^-10 of the scale)."""
    step = dequantize_mxfp8(torch.ones(want.shape).to(torch.float8_e4m3fn), scale, block)
    bound = want.abs() / 16 + step * 2.0 ** -9
    assert bool(((merged - want).abs() <= bound).all()), float(((merged - want).abs() - bound).max())


def test_quantizer_matches_vllm_mxfp8_convention():
    torch.manual_seed(0)
    x = torch.randn(8, 128) * torch.exp2(torch.randint(-12, 4, (8, 1)).float())
    x[3, :32] = 0.0  # an all-zero block
    q, scale = quantize_mxfp8(x, (1, 32))
    q_ref, codes_ref = _vllm_mxfp8_rows(x)
    assert scale.dtype == torch.float8_e8m0fnu and q.dtype == torch.float8_e4m3fn
    assert torch.equal(scale.view(torch.uint8), codes_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    # dequantize is exact: re-quantizing a dequantized weight gives its bytes back
    q2, scale2 = quantize_mxfp8(dequantize_mxfp8(q, scale, (1, 32)), (1, 32))
    assert torch.equal(q2.view(torch.uint8)[x.abs().amax(1) > 0], q.view(torch.uint8)[x.abs().amax(1) > 0])


def test_merged_fp8_dequantizes_to_w_plus_delta_and_writes_its_scale(tmp_path):
    base = _base(tmp_path)
    lora = _lora(SHAPES)
    adapter = _adapter(tmp_path / "release", {f"base_model.model.{k}": v for k, v in lora.items()})
    out = merge_lora_release(base, adapter, tmp_path / "release.merged")
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    for module in SHAPES:
        assert index[f"{module}.weight"] == index[f"{module}.scale"] == "release-lora-merged.safetensors"
    assert index[f"{EXPERT}.weight"] == "model-00002.safetensors"
    assert (out / "model-00002.safetensors").resolve() == (base / "model-00002.safetensors").resolve()
    merged, original = load_file(str(out / "release-lora-merged.safetensors")), load_file(str(base / "model-00001.safetensors"))
    assert set(merged) == {f"{m}.{leaf}" for m in SHAPES for leaf in ("weight", "scale")}
    for module, shape in SHAPES.items():
        weight, scale = merged[f"{module}.weight"], merged[f"{module}.scale"]
        assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu
        assert tuple(weight.shape) == shape and scale.shape == original[f"{module}.scale"].shape
        w = dequantize_mxfp8(original[f"{module}.weight"], original[f"{module}.scale"], BLOCK)
        delta = 2.0 * (lora[f"{module}.lora_B.weight"] @ lora[f"{module}.lora_A.weight"])
        got = dequantize_mxfp8(weight, scale, BLOCK)
        _within_fp8_step(got, w + delta, scale, BLOCK)
        # re-quantized with freshly computed scales, exactly as quantize_mxfp8 of the target
        q_want, s_want = quantize_mxfp8(w + delta, BLOCK)
        assert torch.equal(weight.view(torch.uint8), q_want.view(torch.uint8))
        assert torch.equal(scale.view(torch.uint8), s_want.view(torch.uint8))


def test_zero_delta_is_bit_exact_and_untouched_blocks_keep_their_bytes(tmp_path):
    base = _base(tmp_path)
    adapter = _adapter(tmp_path / "zero", _lora(SHAPES, zero=True))
    out = merge_lora_release(base, adapter, tmp_path / "zero.merged")
    merged, original = load_file(str(out / "release-lora-merged.safetensors")), load_file(str(base / "model-00001.safetensors"))
    for name, tensor in merged.items():
        assert torch.equal(tensor.view(torch.uint8), original[name].view(torch.uint8)), name
    # a delta confined to block row 0 leaves block row 1 (and its scales) byte-identical
    weight, scale = original[f"{WQ_A}.weight"], original[f"{WQ_A}.scale"]
    delta = torch.zeros(SHAPES[WQ_A])
    delta[:32] = torch.randn(32, 96) * 0.01
    new_w, new_s = merge_mxfp8(WQ_A, weight, scale, delta, BLOCK)
    assert torch.equal(new_w.view(torch.uint8)[32:], weight.view(torch.uint8)[32:])
    assert torch.equal(new_s.view(torch.uint8)[1:], scale.view(torch.uint8)[1:])
    assert not torch.equal(new_w.view(torch.uint8)[:32], weight.view(torch.uint8)[:32])


def _rewrite(base: Path, edit) -> None:
    path = base / "model-00001.safetensors"
    tensors = load_file(str(path))
    weight_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    edit(tensors, weight_map)
    save_file(tensors, str(path))
    for name in tensors:
        weight_map.setdefault(name, "model-00001.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def _drop_scale(t, m):
    del t[f"{WQ_A}.scale"], m[f"{WQ_A}.scale"]


def _f32_scale(t, m):
    t[f"{WQ_A}.scale"] = torch.ones(2, 3)


def _fnuz(t, m):
    t[f"{WQ_A}.weight"] = torch.zeros(SHAPES[WQ_A]).to(torch.float8_e4m3fnuz)


def _bad_tiling(t, m):
    t[f"{WQ_A}.scale"] = torch.full((3, 3), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def _nan_code(t, m):
    codes = t[f"{WQ_A}.scale"].view(torch.uint8).clone()
    codes[0, 0] = 255
    t[f"{WQ_A}.scale"] = codes.view(torch.float8_e8m0fnu)


def _input_scale(t, m):
    t[f"{WQ_A}.input_scale"] = torch.ones(1)


def _two_scales(t, m):
    t[f"{WQ_A}.weight_scale_inv"] = t[f"{WQ_A}.scale"].clone()


@pytest.mark.parametrize("edit", [_drop_scale, _f32_scale, _fnuz, _bad_tiling, _nan_code, _input_scale, _two_scales])
def test_unknown_quant_formats_are_refused_before_anything_is_written(tmp_path, edit):
    base = _base(tmp_path)
    _rewrite(base, edit)
    adapter = _adapter(tmp_path / "release", _lora([WQ_A]))
    with pytest.raises(LoraMergeError) as error:
        merge_lora_release(base, adapter, tmp_path / "release.merged")
    assert error.value.reason == "quant_format", error.value
    assert not (tmp_path / "release.merged").exists()
    assert not (tmp_path / "release.merged.partial").exists()
