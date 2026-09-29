"""yeto.rl.fp32_reference: CPU fp32 LoRA reference gradient (rl-engine-ports tier 2, design D12)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from yeto.rl import fp32_reference as f32  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"


def _tiny_model():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=48, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
        tie_word_embeddings=True)
    return transformers.Qwen3ForCausalLM(cfg).float().eval()


def _initial(model, rank=2, targets=("q_proj", "v_proj")):
    g = torch.Generator().manual_seed(1)
    out = {}
    for name, mod in model.named_modules():
        if name.split(".")[-1] in targets:
            out[f"base_model.model.{name}.lora_A.weight"] = torch.randn(rank, mod.in_features, generator=g) * 0.1
            out[f"base_model.model.{name}.lora_B.weight"] = torch.randn(mod.out_features, rank, generator=g) * 0.1
    return out


def _samples():
    # Two groups; very different response lengths so token-mean != sample-mean.
    s = [f32.ReplaySample(list(range(1, 5)) + [7, 9], 2, [1, 1], 1.0, "g0"),
         f32.ReplaySample(list(range(1, 5)) + [3, 5, 8, 11, 13, 2, 6, 4, 10, 12, 14, 15], 12,
                          [1] * 12, 0.0, "g0"),
         f32.ReplaySample(list(range(2, 6)) + [7, 9, 1], 3, [1, 0, 1], 0.0, "g1"),
         f32.ReplaySample(list(range(2, 6)) + [8, 8, 8, 8], 4, [1] * 4, 1.0, "g1")]
    return f32.group_advantages(s, f32.LossConfig())


def test_group_advantages_match_miles_grpo_by_hand():
    s = [f32.ReplaySample([1, 2], 1, [1], r, g) for r, g in
         ((1.0, "a"), (0.0, "a"), (0.0, "a"), (1.0, "a"), (1.0, "b"), (1.0, "b"))]
    f32.group_advantages(s, f32.LossConfig())
    std = math.sqrt(1 / 3)  # unbiased std of [1, 0, 0, 1]
    assert [x.advantage for x in s[:4]] == pytest.approx([0.5 / (std + 1e-6), -0.5 / (std + 1e-6)] * 1
                                                         + [-0.5 / (std + 1e-6), 0.5 / (std + 1e-6)])
    assert [x.advantage for x in s[4:]] == [0.0, 0.0]  # zero std: mean-centred only
    f32.group_advantages(s, f32.LossConfig(grpo_std_normalization=False))
    assert s[0].advantage == pytest.approx(0.5)


def test_reference_gradient_matches_independent_autograd():
    model = _tiny_model()
    init = _initial(model)
    peft_model, params = f32.attach_lora(model, init, lora_alpha=4)
    samples = _samples()
    got = f32.reference_gradient(peft_model, params, samples, f32.LossConfig(num_samples=4))

    # Independent: at the first on-policy step ratio == 1, so the clipped GRPO
    # surrogate's gradient is that of  sum_i A_i * mean_masked(-logp_i) / N.
    for p in params.values():
        p.grad = None
    total = 0
    for s in samples:
        logits = peft_model(input_ids=torch.tensor([s.tokens])).logits[0]
        logp = torch.log_softmax(logits, -1)
        r = s.response_length
        tok_lp = torch.stack([logp[len(s.tokens) - r - 1 + t, s.tokens[len(s.tokens) - r + t]]
                              for t in range(r)])
        m = torch.tensor(s.loss_mask, dtype=torch.float32)
        total = total + s.advantage * (-(tok_lp * m).sum() / m.sum()) / 4
    names = sorted(params)
    want = torch.autograd.grad(total, [params[n] for n in names])
    for n, w in zip(names, want):
        assert torch.allclose(got["grads"][n], w, rtol=1e-5, atol=1e-7), n
    # ratio == 1: the surrogate's value is sum_i(-A_i)/N (only its gradient carries logp).
    assert got["loss"] == pytest.approx(sum(-s.advantage for s in samples) / 4, abs=1e-6)
    assert any(float(g.abs().sum()) > 0 for g in want)


def test_lora_scaling_and_initial_weights_are_applied():
    model = _tiny_model()
    init = _initial(model)
    peft_model, params = f32.attach_lora(model, init, lora_alpha=4)
    for n, p in params.items():
        assert torch.equal(p.detach(), init[n])
    layer = peft_model.base_model.model.model.layers[0].self_attn.q_proj
    assert layer.scaling["default"] == pytest.approx(2.0)  # alpha / r
    peft_model.unload()


def test_token_mean_aggregation_changes_the_gradient():
    model = _tiny_model()
    peft_model, params = f32.attach_lora(model, _initial(model), lora_alpha=2)
    s = _samples()
    a = f32.reference_gradient(peft_model, params, s, f32.LossConfig())
    for p in params.values():
        p.grad = None
    b = f32.reference_gradient(peft_model, params, s, f32.LossConfig(aggregation=f32.TOKEN_MEAN))
    order = sorted(params)
    ga, gb = f32.flatten(a["grads"], order), f32.flatten(b["grads"], order)
    assert np.linalg.norm(ga - gb) / np.linalg.norm(ga) > 0.1


def test_token_mean_bug_is_caught_by_the_fp32_anchor(tmp_path):
    """A ports path that averages per token (the upstream rollout_mask_sums shape on
    multi-turn batches) fails tier 2 while a legacy path with bf16-like noise passes."""
    import importlib.util
    from pathlib import Path

    from yeto.rl import grad_audit

    spec = importlib.util.spec_from_file_location(
        "rl_engine_equivalence", Path(__file__).resolve().parents[1] / "scripts" / "rl_engine_equivalence.py")
    eq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eq)

    model = _tiny_model()
    peft_model, params = f32.attach_lora(model, _initial(model), lora_alpha=2)
    s = _samples()
    ref = f32.reference_gradient(peft_model, params, s, f32.LossConfig())["grads"]
    for p in params.values():
        p.grad = None
    bug = f32.reference_gradient(peft_model, params, s, f32.LossConfig(aggregation=f32.TOKEN_MEAN))["grads"]
    g = torch.Generator().manual_seed(3)
    noisy = {n: t * (1 + 0.08 * torch.randn(t.shape, generator=g)) for n, t in ref.items()}
    order = sorted(ref)
    fp32 = {"flat": f32.flatten(ref, order), "order": order,
            "shapes": {n: list(ref[n].shape) for n in order}, "grad_norm": 1.0}
    paths = {}
    for label, grads in (("legacy", noisy), ("ports_ok", noisy), ("ports_bug", bug)):
        grad_audit.write(tmp_path / label, "round-00000001", grads, {})
        paths[label] = tmp_path / label / "round-00000001.grad.f32"
    ok = eq.fp32_gate(eq.fp32_anchor_compare(paths["legacy"], paths["ports_ok"], fp32))
    assert ok == {"rel_ok": True, "cos_ok": True}
    a = eq.fp32_anchor_compare(paths["legacy"], paths["ports_bug"], fp32)
    assert a["ports"]["rel_l2"] > a["legacy"]["rel_l2"] + eq.TF_FP32_REL_L2_MARGIN
    assert eq.fp32_gate(a)["rel_ok"] is False


def _write_inputs(tmp_path, model, init, samples):
    snap = tmp_path / "snapshots" / REV
    model.save_pretrained(snap, safe_serialization=True)
    replay = tmp_path / "rollouts" / "island-0" / "0.pt"
    replay.parent.mkdir(parents=True)
    torch.save({"rollout_id": 0, "metadata": {}, "samples": [
        {"tokens": s.tokens, "response_length": s.response_length,
         "loss_mask": s.loss_mask, "reward": s.reward, "group_index": s.group,
         "remove_sample": False} for s in samples]}, replay)
    names = sorted(init)
    flat = np.concatenate([init[n].numpy().astype("<f4").ravel() for n in names])
    base = tmp_path / "audit" / "round-00000001.base.f32"
    base.parent.mkdir()
    flat.tofile(base)
    meta = base.with_name("round-00000001.json")
    meta.write_text(json.dumps({"base_model_revision": REV, "layout_hash": "L", "specs": [
        {"name": n, "shape": list(init[n].shape), "numel": init[n].numel()} for n in names]}))
    return snap, replay, base, meta


def test_compute_island_end_to_end_with_provenance(tmp_path):
    model = _tiny_model()
    init = _initial(model)
    samples = _samples()
    snap, replay, base, meta = _write_inputs(tmp_path, model, init, samples)
    cfg = f32.LossConfig(num_samples=4, lora_alpha=4)
    res = f32.compute_island(replay=str(replay), base_f32=str(base), base_meta=str(meta),
                             model="org/tiny", revision=REV, cfg=cfg, model_path=str(snap))
    peft_model, params = f32.attach_lora(_tiny_model(), init, lora_alpha=4)
    want = f32.reference_gradient(peft_model, params, _samples(), cfg)
    np.testing.assert_allclose(res["flat"], f32.flatten(want["grads"], res["order"]),
                               rtol=1e-5, atol=1e-7)
    prov = res["provenance"]
    assert prov["replay"]["sha256"] == f32.sha256_file(replay)
    assert prov["initial_lora"]["sha256"] == f32.sha256_file(base)
    assert prov["model"]["revision"] == REV and "config.json" in prov["model"]["files_sha256"]
    assert res["num_samples"] == 4 and 0 < res["clip_coefficient"] <= 1.0


def test_missing_or_inconsistent_inputs_raise(tmp_path):
    model = _tiny_model()
    init = _initial(model)
    snap, replay, base, meta = _write_inputs(tmp_path, model, init, _samples())
    kw = dict(replay=str(replay), base_f32=str(base), base_meta=str(meta), model="org/tiny",
              revision=REV, cfg=f32.LossConfig(), model_path=str(snap))
    with pytest.raises(f32.ReferenceInputError, match="replay batch missing"):
        f32.compute_island(**{**kw, "replay": str(tmp_path / "nope.pt")})
    with pytest.raises(f32.ReferenceInputError, match="40-hex"):
        f32.compute_island(**{**kw, "revision": "main"})
    with pytest.raises(f32.ReferenceInputError, match="base_model_revision"):
        f32.compute_island(**{**kw, "revision": "f" * 40})
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "base_f32_sha256": "0" * 64}))
    with pytest.raises(f32.ReferenceInputError, match="sha256"):
        f32.compute_island(**kw)
