"""Re-quantization BF16 -> MXFP4 experts, group 32, RTN (S25 N8.5).

Codec rules (E8M0 rounding, FP4 ties, nibble order, sign of zero) are checked
against a literal float re-statement of compressed-tensors 0.18, written
without the package code; the layout tests build a tiny reference artefact in
the format of ``yetalabs/apex-flash-1-mxfp4-experts@70e86a0`` (only routed
experts as ``weight_packed`` + ``weight_scale``, everything else BF16/F32).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from yeto.rl.quant.awq import AwqSmoothing
from yeto.rl.quant.mxfp4 import (
    FP4_VALUES,
    GROUP_SIZE,
    dequantize_mxfp4,
    e8m0_codes,
    e8m0_to_scale,
    fp4_decode,
    fp4_encode,
    pack_fp4,
    quantize_mxfp4,
    unpack_fp4,
)
from yeto.rl.quant.requant import (
    INDEX_FILE,
    MXFP4_G32_RTN,
    CheckpointLayout,
    LayoutError,
    compare_checkpoints,
    main,
    requantize_checkpoint,
    transform_tensor,
)


def _bf16(*xs: float) -> torch.Tensor:
    return torch.tensor(xs, dtype=torch.bfloat16)


def _reference_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """compressed-tensors 0.18 MXFP4 path, re-stated in plain float math (no package code)."""
    rows, cols = w.shape
    out_codes, out_scale = [], []
    for r in range(rows):
        row_codes, row_scale = [], []
        for g in range(0, cols, GROUP_SIZE):
            grp = [float(v) for v in w[r, g:g + GROUP_SIZE].tolist()]
            amax = max(abs(v) for v in grp)
            if amax == 0:
                exp = 0
            else:
                m, e = math.frexp(amax)  # amax = m * 2**e, m in [0.5, 1)
                e -= 1  # amax = (2m) * 2**e, 2m in [1, 2)
                exp = max(0, 127 + e + (1 if 2 * m >= 1.75 else 0) - 2)
            scale = 2.0 ** (exp - 127)
            row_scale.append(exp)
            for v in grp:
                x = v / scale
                mag = min(abs(x), 6.0)
                # nearest E2M1 value, ties to the even mantissa (0, 1, 2, 4 are "even")
                best = min(range(8), key=lambda k: (abs(FP4_VALUES[k] - mag), k % 2))
                row_codes.append(best | (8 if v < 0 else 0))
        out_codes.append(row_codes)
        out_scale.append(row_scale)
    codes = torch.tensor(out_codes, dtype=torch.uint8)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), torch.tensor(out_scale, dtype=torch.uint8)


# -- codec --------------------------------------------------------------------


def test_e8m0_rounds_up_from_mantissa_1_75():
    # amax = 1.0 -> 2**0 -> code 127 - 2; 1.7421875 (largest BF16 below 1.75) stays; 1.75 carries.
    codes = e8m0_codes(_bf16(1.0, 1.7421875, 1.75, 6.0, 0.5, 2.0**-130, 0.0))
    assert codes.tolist() == [125, 125, 126, 127, 124, 0, 0]


def test_e8m0_extremes():
    tiny = torch.tensor([2.0**-126, 2.0**-125, 2.0**-124], dtype=torch.bfloat16)
    assert e8m0_codes(tiny).tolist() == [0, 0, 1]  # 127 - 126 - 2 < 0 clamps to 0
    assert e8m0_codes(_bf16(2.0**127)).item() == 252
    with pytest.raises(ValueError, match="too large"):
        e8m0_codes(torch.tensor([torch.finfo(torch.bfloat16).max], dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="non-finite"):
        e8m0_codes(_bf16(float("inf")))
    with pytest.raises(TypeError):
        e8m0_codes(torch.ones(2))
    assert e8m0_to_scale(torch.tensor([127, 125, 0], dtype=torch.uint8)).tolist() == [1.0, 0.25, 2.0**-127]


def test_fp4_ties_to_even_saturation_and_sign_of_zero():
    x = torch.tensor([0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 100.0, -0.1, -0.0, 0.0, -6.0])
    assert fp4_encode(x).tolist() == [0, 1, 2, 2, 4, 4, 6, 6, 7, 7, 8, 0, 0, 15]
    codes = torch.arange(16, dtype=torch.uint8)
    assert fp4_decode(codes).tolist() == list(FP4_VALUES) + [-v for v in FP4_VALUES]


def test_pack_low_nibble_is_the_even_column_and_round_trips():
    assert pack_fp4(torch.tensor([[1, 2, 15, 0]], dtype=torch.uint8)).tolist() == [[0x21, 0x0F]]
    codes = torch.randint(0, 16, (5, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    assert torch.equal(unpack_fp4(pack_fp4(codes)), codes)
    with pytest.raises(ValueError):
        pack_fp4(torch.zeros(2, 3, dtype=torch.uint8))
    with pytest.raises(ValueError):
        pack_fp4(torch.full((1, 2), 16, dtype=torch.uint8))


def test_quantize_matches_literal_reference_on_mixed_groups():
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(6, 4 * GROUP_SIZE, generator=g) * 0.02).to(torch.bfloat16)
    w[1, :GROUP_SIZE] = 0  # all-zero group
    w[2, GROUP_SIZE:2 * GROUP_SIZE] = -0.0  # all negative zeros
    w[2, 2 * GROUP_SIZE] = -1e-6  # negative, rounds to zero: stored as -0
    w[3, :GROUP_SIZE] = 1e-4
    w[3, 5] = 300.0  # one extreme value: the others round to zero
    w[4, 2 * GROUP_SIZE:3 * GROUP_SIZE] = torch.tensor(FP4_VALUES * 4, dtype=torch.bfloat16) * 2.0**-3
    packed, scale = quantize_mxfp4(w)
    ref_packed, ref_scale = _reference_quantize(w)
    assert torch.equal(scale, ref_scale) and torch.equal(packed, ref_packed)
    assert scale.shape == (6, 4) and packed.shape == (6, 2 * GROUP_SIZE)
    assert scale[1, 0] == 0 and (unpack_fp4(packed)[1, :GROUP_SIZE] == 0).all()
    assert (unpack_fp4(packed)[2, GROUP_SIZE:2 * GROUP_SIZE] == 0).all()  # input -0.0 is stored as +0
    assert int((unpack_fp4(packed)[3, :GROUP_SIZE] & 7).count_nonzero()) == 1
    assert unpack_fp4(packed)[2, 2 * GROUP_SIZE] == 8


def test_representable_values_round_trip_exactly():
    g = torch.Generator().manual_seed(4)
    codes = torch.randint(0, 16, (8, 4 * GROUP_SIZE), dtype=torch.uint8, generator=g)
    codes[:, ::GROUP_SIZE] = 7  # every group reaches 6 * scale, so the scale is recovered
    exps = torch.randint(110, 135, (8, 4), dtype=torch.uint8, generator=g)
    w = dequantize_mxfp4(pack_fp4(codes), exps, torch.bfloat16)
    packed, scale = quantize_mxfp4(w)
    assert torch.equal(scale, exps)
    assert torch.equal(dequantize_mxfp4(packed, scale), w.float())


def test_round_trip_error_is_bounded_by_the_group_step():
    g = torch.Generator().manual_seed(5)
    w = (torch.randn(16, 8 * GROUP_SIZE, generator=g) * 0.03).to(torch.bfloat16)
    packed, scale = quantize_mxfp4(w)
    err = (dequantize_mxfp4(packed, scale) - w.float()).abs().reshape(16, 8, GROUP_SIZE)
    step = e8m0_to_scale(scale).unsqueeze(-1)
    # Largest gap between E2M1 values is 2 (4 -> 6); saturation above 6 loses < 1 step (amax < 7 * scale).
    assert (err <= step).all()


def test_rows_are_independent():
    w = (torch.randn(8, 2 * GROUP_SIZE, generator=torch.Generator().manual_seed(6))).to(torch.bfloat16)
    packed, scale = quantize_mxfp4(w)
    p2, s2 = quantize_mxfp4(w[3:6])
    assert torch.equal(p2, packed[3:6]) and torch.equal(s2, scale[3:6])


def test_quantize_rejects_bad_input():
    with pytest.raises(ValueError, match="multiple of the group size"):
        quantize_mxfp4(torch.zeros(2, 48, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="BF16"):
        quantize_mxfp4(torch.zeros(2, 32))
    with pytest.raises(ValueError, match="2-D"):
        quantize_mxfp4(torch.zeros(32, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="does not fit"):
        dequantize_mxfp4(torch.zeros(2, 16, dtype=torch.uint8), torch.zeros(2, 2, dtype=torch.uint8))


def test_matches_compressed_tensors_if_installed():
    """Cross-check with the library that wrote the reference artefact (skipped where it is not installed)."""
    ct = pytest.importorskip("compressed_tensors.compressors.mxfp4.base")
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from compressed_tensors.quantization.utils.helpers import calculate_qparams

    g = torch.Generator().manual_seed(8)
    w = (torch.randn(4, 4 * GROUP_SIZE, generator=g) * 0.05).to(torch.bfloat16)
    w[0, :GROUP_SIZE] = 0
    w[1, 3] = -0.0
    w[2, 7] = 0.4
    w[3, 9] = -1e-6
    args = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=GROUP_SIZE,
                            symmetric=True, scale_dtype=torch.uint8)
    groups = w.reshape(4, -1, GROUP_SIZE)
    scale, _ = calculate_qparams(groups.amin(-1), groups.amax(-1), args)
    out = ct.MXFP4PackedCompressor.compress({"weight": w, "weight_scale": scale},
                                            QuantizationScheme(targets=["Linear"], weights=args))
    packed, codes = quantize_mxfp4(w)
    assert torch.equal(out["weight_scale"], codes) and torch.equal(out["weight_packed"], packed)


# -- checkpoint layout ----------------------------------------------------------

H, I, E = 64, 32, 2  # hidden, expert width, routed experts
L = "model.language_model.layers.3."
MTP = "model.language_model.layers.4."  # stands for the ignored MTP layer 45

QCONFIG = {
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "config_groups": {
        "MXFP4A16": {
            "format": "mxfp4-pack-quantized",
            "targets": ["re:.*mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$"],
            "weights": {"num_bits": 4, "type": "float", "symmetric": True, "strategy": "group", "group_size": 32,
                        "dynamic": False, "scale_dtype": "torch.uint8", "observer": None},
            "input_activations": None,
        }
    },
    "ignore": ["lm_head", "re:.*layers\\.4\\..*", "re:^(?!.*mlp\\.experts\\.\\d+\\.).*$"],
}


def _source_tensors() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(11)

    def rnd(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    t = {L + "post_attention_layernorm.weight": rnd(H), L + "mlp.gate.weight": rnd(E, H),
         L + "mlp.gate.e_score_correction_bias": torch.randn(E, generator=g),
         L + "mlp.shared_experts.gate_proj.weight": rnd(I, H), "lm_head.weight": rnd(8, H)}
    for layer in (L, MTP):
        for e in range(E):
            t[f"{layer}mlp.experts.{e}.gate_proj.weight"] = rnd(I, H)
            t[f"{layer}mlp.experts.{e}.up_proj.weight"] = rnd(I, H)
            t[f"{layer}mlp.experts.{e}.down_proj.weight"] = rnd(H, I)
    t[L + "mlp.experts.0.down_proj.weight"][2, :GROUP_SIZE] = 0  # an all-zero group
    return t


def _reference_artefact(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, w in src.items():
        if name.startswith(L + "mlp.experts."):
            module = name.removesuffix(".weight")
            out[module + ".weight_packed"], out[module + ".weight_scale"] = _reference_quantize(w)
        else:
            out[name] = w
    return out


def _shard_of(name: str) -> str:
    return "model-00002-of-00002.safetensors" if "experts.1." in name else "model-00001-of-00002.safetensors"


def _write_ckpt(d: Path, tensors: dict[str, torch.Tensor], shard_of) -> None:
    d.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict[str, torch.Tensor]] = {}
    for n, t in tensors.items():
        shards.setdefault(shard_of(n), {})[n] = t.contiguous()
    for f, ts in shards.items():
        save_file(ts, str(d / f), metadata={"format": "pt"})
    (d / INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": {n: shard_of(n) for n in tensors}}))


@pytest.fixture
def world(tmp_path: Path):
    src = _source_tensors()
    original = tmp_path / "mxfp4-experts"
    _write_ckpt(original, _reference_artefact(src), _shard_of)
    (original / "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization_config": QCONFIG}))
    source = tmp_path / "bf16"
    _write_ckpt(source, src, lambda n: "model-00001-of-00001.safetensors")
    template = tmp_path / "template"
    CheckpointLayout.from_checkpoint(original).save_template(original, template)
    return {"original": original, "source": source, "template": template, "src": src, "tmp": tmp_path}


def test_layout_selects_mxfp4_scheme_and_source_names(world):
    layout = CheckpointLayout.load(world["template"])
    assert layout.scheme is MXFP4_G32_RTN and layout.scheme.name == "mxfp4-g32-rtn"
    assert len(layout.quantized_weights()) == 3 * E and all(n.endswith(".weight_packed") for n in layout.quantized_weights())
    assert layout.source_names() == set(world["src"])


def test_untrained_source_reproduces_reference_bit_for_bit(world):
    out = world["tmp"] / "out"
    stats = requantize_checkpoint(world["source"], world["template"], out, AwqSmoothing.none())
    assert stats["quantized"] == 3 * E
    rep = compare_checkpoints(out, world["original"])
    assert rep["fp4_match_share"] == 1.0 and rep["e8m0_match_share"] == 1.0 and rep["unequal_tensors"] == 0
    assert rep["tensors"] == len(_reference_artefact(world["src"]))
    assert (out / "config.json").is_file()


def test_awq_smoothing_is_refused(world, tmp_path):
    torch.save(torch.ones(H, dtype=torch.float64), tmp_path / f"{L}post_attention_layernorm.pt")
    (tmp_path / "recipe.yaml").write_text(
        f"default_stage:\n  default_modifiers:\n    RecordedAWQ:\n      mappings:\n"
        f"      - smooth_layer: {L}post_attention_layernorm\n        balance_layers: ['{L}mlp.gate']\n")
    smoothing = AwqSmoothing.from_artefact(tmp_path / "recipe.yaml", tmp_path)
    with pytest.raises(LayoutError, match="round-to-nearest only"):
        requantize_checkpoint(world["source"], world["template"], world["tmp"] / "x", smoothing)


def test_transform_row_slice_gives_the_same_rows(world):
    layout = CheckpointLayout.load(world["template"])
    name = L + "mlp.experts.1.down_proj.weight"
    full = transform_tensor(name, world["src"][name], layout, AwqSmoothing.none())
    part = transform_tensor(name, world["src"][name][10:20], layout, AwqSmoothing.none())
    for k in full:
        assert torch.equal(part[k], full[k][10:20])
    with pytest.raises(LayoutError, match="not a source tensor"):
        transform_tensor(L + "mlp.experts.1.down_proj.weight_scale", full[L + "mlp.experts.1.down_proj.weight_scale"],
                         layout, AwqSmoothing.none())


@pytest.mark.parametrize("field,value", [("group_size", 16), ("scale_dtype", "torch.bfloat16"), ("type", "int")])
def test_unsupported_mxfp4_config_is_rejected(world, field, value):
    cfg = json.loads((world["original"] / "config.json").read_text())
    cfg["quantization_config"]["config_groups"]["MXFP4A16"]["weights"][field] = value
    (world["original"] / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(LayoutError, match="not supported"):
        CheckpointLayout.from_checkpoint(world["original"])


def test_dense_format_is_rejected(world):
    # Pit 2 (A-QNT): llmcompressor 0.14 wrote format=dense for MXFP4 and readers mis-read the weights.
    cfg = json.loads((world["original"] / "config.json").read_text())
    cfg["quantization_config"]["format"] = "dense"
    (world["original"] / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(LayoutError, match="unsupported quantization format"):
        CheckpointLayout.from_checkpoint(world["original"])


def test_cli_template_convert_compare(world, capsys):
    tpl, out = world["tmp"] / "tpl-cli", world["tmp"] / "out-cli"
    assert main(["template", str(world["original"]), str(tpl)]) == 0
    assert main(["convert", "--source", str(world["source"]), "--template", str(tpl), "--out", str(out), "--no-awq"]) == 0
    capsys.readouterr()
    assert main(["compare", str(out), str(world["original"]), "--match", r"experts\.\d+\."]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["fp4_match_share"] == 1.0 and rep["e8m0_match_share"] == 1.0 and rep["tensors"] == 3 * 3 * E  # packed + scale, plus the BF16 MTP experts
