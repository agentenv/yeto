"""Re-quantization BF16 -> W4A8 (S25 N8.1): int4 round trip, AWQ R2/R1, layout fidelity.

The "original artefact" here is a tiny checkpoint built by an independent,
literal reading of the rules observed on the real artefact (FP64 smoothing,
amax / 7.5, BF16 scales, routers stored F32, a scale stored in another shard).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from yeto.rl.quant.awq import AwqSmoothing
from yeto.rl.quant.int4 import dequantize_per_channel, quantize_per_channel
from yeto.rl.quant.requant import (
    INDEX_FILE,
    CheckpointLayout,
    LayoutError,
    compare_checkpoints,
    main,
    read_header,
    requantize_checkpoint,
    transform_tensor,
)

H, F, E = 16, 8, 2  # hidden, expert width, routed experts
L = "model.language_model.layers.3."
NORM = L + "post_attention_layernorm"

RECIPE = f"""default_stage:
  default_modifiers:
    RecordedAWQ:
      mappings:
      - smooth_layer: {NORM}
        balance_layers: [{L}mlp.gate, 're:model\\.language_model\\.layers\\.3\\.mlp\\.(experts\\.\\d+|shared_experts)\\.(gate|up)_proj$']
      offload_device: !!python/object/apply:torch.device [cpu]
      duo_scaling: true
    QuantizationModifier:
      ignore: [lm_head]
"""

QCONFIG = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*mlp\\.experts\\..*(gate|up|down)_proj$"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "channel",
                        "group_size": None, "dynamic": False, "observer": "memoryless_minmax"},
            "input_activations": {"num_bits": 8, "strategy": "token", "dynamic": True},
        }
    },
}


def _source_tensors(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def r(*shape, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g) * 0.05).to(dtype)

    t = {
        "model.language_model.embed_tokens.weight": r(32, H),
        NORM + ".weight": (1 + r(H, dtype=torch.float32)).to(torch.bfloat16),
        L + "mlp.gate.weight": r(E, H),
        L + "mlp.gate.e_score_correction_bias": r(E, dtype=torch.float32),
        L + "mlp.shared_experts.gate_proj.weight": r(F, H),
        L + "mlp.shared_experts.up_proj.weight": r(F, H),
        L + "mlp.shared_experts.down_proj.weight": r(H, F),
        L + "self_attn.q_convNd.weight": r(4, 1, 3, dtype=torch.float32),
    }
    for e in range(E):
        for p, shape in (("gate", (F, H)), ("up", (F, H)), ("down", (H, F))):
            t[f"{L}mlp.experts.{e}.{p}_proj.weight"] = r(*shape)
    t[f"{L}mlp.experts.0.down_proj.weight"][1].zero_()  # an all-zero row
    return t


def _reference_artefact(src: dict[str, torch.Tensor], s: torch.Tensor) -> dict[str, torch.Tensor]:
    """Literal re-statement of the observed rules, written without the package code."""
    out: dict[str, torch.Tensor] = {}
    for name, w in src.items():
        mod = name.removesuffix(".weight")
        if name == NORM + ".weight":
            out[name] = (w.double() / s).to(torch.bfloat16)
        elif name == L + "mlp.gate.weight":
            out[name] = (w.double() * s).to(torch.float32)
        elif name.endswith("e_score_correction_bias"):
            out[name] = w.to(torch.bfloat16)
        elif ".mlp.experts." in name:
            x = (w.double() * s).to(torch.bfloat16) if ("gate_proj" in name or "up_proj" in name) else w
            scale = (x.float().abs().amax(1, keepdim=True) / 7.5).to(torch.bfloat16)
            scale[scale == 0] = torch.finfo(torch.bfloat16).tiny
            out[name] = torch.round(x.float() / scale.float()).clamp(-8, 7).to(torch.int8)
            out[mod + ".weight_scale"] = scale
        elif "shared_experts" in name and ("gate_proj" in name or "up_proj" in name):
            out[name] = (w.double() * s).to(torch.bfloat16)
        else:
            out[name] = w
    return out


def _write_ckpt(d: Path, tensors: dict[str, torch.Tensor], shard_of) -> None:
    d.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict[str, torch.Tensor]] = {}
    for n, t in tensors.items():
        shards.setdefault(shard_of(n), {})[n] = t.contiguous()
    for f, ts in shards.items():
        save_file(ts, str(d / f), metadata={"format": "pt"})
    (d / INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": {n: shard_of(n) for n in tensors}}))


def _shard_of(name: str) -> str:
    # Like the real artefact: one expert's up_proj scale lives in the other shard.
    if "experts.1" in name or name == f"{L}mlp.experts.0.up_proj.weight_scale":
        return "model-00002-of-00002.safetensors"
    return "model-00001-of-00002.safetensors"


@pytest.fixture
def world(tmp_path: Path):
    s = (0.8 + 0.4 * torch.rand(H, generator=torch.Generator().manual_seed(7))).double()
    scales = tmp_path / "awq-scales"
    scales.mkdir()
    torch.save(s, scales / f"{NORM}.pt")
    original = tmp_path / "checkpoint"
    src = _source_tensors()
    _write_ckpt(original, _reference_artefact(src, s), _shard_of)
    (original / "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization_config": QCONFIG}))
    (original / "recipe.yaml").write_text(RECIPE)
    (original / "tokenizer.json").write_text("{}")
    source = tmp_path / "bf16"
    _write_ckpt(source, src, lambda n: "model-00001-of-00001.safetensors")
    template = tmp_path / "template"
    CheckpointLayout.from_checkpoint(original).save_template(original, template)
    smoothing = AwqSmoothing.from_artefact(original / "recipe.yaml", scales)
    return {"original": original, "source": source, "template": template, "smoothing": smoothing, "src": src, "tmp": tmp_path}


def test_int4_round_trip_error_is_at_most_half_a_step():
    w = (torch.randn(64, 256) * 0.02).to(torch.bfloat16)
    q, scale = quantize_per_channel(w)
    assert q.dtype == torch.int8 and scale.dtype == torch.bfloat16 and scale.shape == (64, 1)
    assert int(q.min()) >= -8 and int(q.max()) <= 7
    err = (dequantize_per_channel(q, scale) - w.float()).abs()
    # Half a step; the BF16 rounding of the scale (relative 2**-9) can push the row max
    # a little past 7.5 steps, where the clamp to 7 adds up to 7.5 * 2**-8 steps.
    assert bool((err <= scale.float() * (0.5 + 7.5 * 2**-8)).all())
    # The largest |w| per row maps to -8 or 7 (that is what amax / 7.5 means).
    assert bool((q.abs().amax(1) >= 7).all())


def test_int4_rows_are_independent():
    w = torch.randn(10, 32).to(torch.bfloat16)
    q, s = quantize_per_channel(w)
    parts = [quantize_per_channel(w[a:b]) for a, b in ((0, 3), (3, 4), (4, 10))]
    assert torch.equal(torch.cat([p[0] for p in parts]), q)
    assert torch.equal(torch.cat([p[1] for p in parts]), s)


def test_int4_rejects_bad_input():
    with pytest.raises(ValueError):
        quantize_per_channel(torch.randn(4))
    with pytest.raises(ValueError):
        quantize_per_channel(torch.tensor([[1.0, float("inf")]]))


def test_recipe_with_python_tags_parses_and_maps_roles(world):
    sm = world["smoothing"]
    assert sm.enabled
    assert sm.role(NORM + ".weight")[0] == "smooth"
    assert sm.role(L + "mlp.gate.weight")[0] == "balance"
    assert sm.role(L + "mlp.experts.1.up_proj.weight")[0] == "balance"
    assert sm.role(L + "mlp.experts.1.down_proj.weight") is None
    assert sm.role(L + "mlp.shared_experts.down_proj.weight") is None
    assert sm.role(L + "mlp.gate.e_score_correction_bias") is None


def test_missing_awq_factor_file_fails(world, tmp_path):
    with pytest.raises(FileNotFoundError):
        AwqSmoothing.from_artefact(world["original"] / "recipe.yaml", tmp_path / "nowhere")


def test_r2_on_untrained_source_reproduces_original_bit_for_bit(world):
    out = world["tmp"] / "requant"
    stats = requantize_checkpoint(world["source"], world["template"], out, world["smoothing"])
    assert stats["shards"] == 2 and stats["quantized"] == 3 * E
    rep = compare_checkpoints(out, world["original"])
    assert rep["int_match_share"] == 1.0 and rep["scale_max_rel_err"] == 0.0 and rep["unequal_tensors"] == 0
    # Same files, same index, same headers (names, dtypes, shapes), same metadata files.
    orig_files = sorted(p.name for p in world["original"].iterdir())
    assert sorted(p.name for p in out.iterdir()) == orig_files
    assert json.loads((out / INDEX_FILE).read_text())["weight_map"] == json.loads((world["original"] / INDEX_FILE).read_text())["weight_map"]
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        ho, hn = read_header(world["original"] / shard), read_header(out / shard)
        assert {n: (v["dtype"], v["shape"]) for n, v in ho.items()} == {n: (v["dtype"], v["shape"]) for n, v in hn.items()}
        with safe_open(str(out / shard), "pt") as f:
            assert f.metadata() == {"format": "pt"}
    assert (out / "recipe.yaml").read_text() == RECIPE


def test_trained_weights_round_trip_within_half_step(world):
    """After a training step the output is still loadable and dequantizes to the smoothed trained weight."""
    trained = {n: (t.float() + 0.01 * torch.randn_like(t.float())).to(t.dtype) for n, t in world["src"].items()}
    src2 = world["tmp"] / "bf16-trained"
    _write_ckpt(src2, trained, lambda n: "model-00001-of-00001.safetensors")
    out = world["tmp"] / "requant-trained"
    requantize_checkpoint(src2, world["template"], out, world["smoothing"])
    s = torch.load(world["tmp"] / "awq-scales" / f"{NORM}.pt")
    name = f"{L}mlp.experts.1.gate_proj"
    with safe_open(str(out / _shard_of(name + ".weight")), "pt") as f:
        q, scale = f.get_tensor(name + ".weight"), f.get_tensor(name + ".weight_scale")
    target = (trained[name + ".weight"].double() * s).to(torch.bfloat16).float()
    assert bool(((dequantize_per_channel(q, scale) - target).abs() <= scale.float() * (0.5 + 7.5 * 2**-8)).all())
    rep = compare_checkpoints(out, world["original"])
    assert rep["int_match_share"] < 1.0  # the weights moved


def test_r1_differs_from_original_only_where_awq_acts(world):
    out = world["tmp"] / "r1"
    requantize_checkpoint(world["source"], world["template"], out, AwqSmoothing.none())
    rep = compare_checkpoints(out, world["original"], names=[n for n in world["src"] if "down_proj" in n and "experts." in n])
    assert rep["int_match_share"] == 1.0  # down_proj is not smoothed
    full = compare_checkpoints(out, world["original"])
    assert any(NORM in n for n in full["unequal_examples"])


def test_row_slices_give_the_same_rows(world):
    layout = CheckpointLayout.load(world["template"])
    name = f"{L}mlp.experts.0.up_proj.weight"
    w = world["src"][name]
    whole = transform_tensor(name, w, layout, world["smoothing"])
    halves = [transform_tensor(name, w[a:b], layout, world["smoothing"]) for a, b in ((0, 5), (5, F))]
    for key in whole:
        assert torch.equal(torch.cat([h[key] for h in halves]), whole[key])


def test_split_by_output_shard_then_finalize_equals_single_run(world):
    single, split = world["tmp"] / "single", world["tmp"] / "split"
    requantize_checkpoint(world["source"], world["template"], single, world["smoothing"])
    for shard in ("model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"):
        requantize_checkpoint(world["source"], world["template"], split, world["smoothing"], shards=[shard])
    assert not (split / INDEX_FILE).exists()
    assert main(["finalize", "--template", str(world["template"]), "--out", str(split)]) == 0
    for p in single.iterdir():
        assert (split / p.name).read_bytes() == p.read_bytes(), p.name


def test_source_with_missing_or_extra_tensors_fails(world):
    src = dict(world["src"])
    src.pop(f"{L}mlp.experts.0.gate_proj.weight")
    src["model.mtp.extra.weight"] = torch.zeros(2, 2)
    bad = world["tmp"] / "bad"
    _write_ckpt(bad, src, lambda n: "model-00001-of-00001.safetensors")
    with pytest.raises(LayoutError, match="1 missing.*1 extra"):
        requantize_checkpoint(bad, world["template"], world["tmp"] / "x", world["smoothing"])


def test_unsupported_scheme_is_rejected(world):
    cfg = json.loads((world["original"] / "config.json").read_text())
    cfg["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = 128
    (world["original"] / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(LayoutError, match="not supported"):
        CheckpointLayout.from_checkpoint(world["original"])


def test_cli_template_convert_compare(world, capsys):
    tpl, out = world["tmp"] / "tpl-cli", world["tmp"] / "out-cli"
    assert main(["template", str(world["original"]), str(tpl)]) == 0
    assert main(["convert", "--source", str(world["source"]), "--template", str(tpl), "--out", str(out),
                 "--recipe", str(world["original"] / "recipe.yaml"), "--awq-scales", str(world["tmp"] / "awq-scales")]) == 0
    capsys.readouterr()
    assert main(["compare", str(out), str(world["original"]), "--match", r"experts\.\d+\."]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["int_match_share"] == 1.0 and rep["tensors"] == 2 * 3 * E  # weights + scales
