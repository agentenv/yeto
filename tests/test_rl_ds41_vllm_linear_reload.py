"""The in-place reload of merged attention projections (DeepSeek-V4.1-Flash on
vLLM ROCm, ``vllm_linear_reload``): the CPU plan (attention-only overlays of the
base, the served release's modules folded in), the registry put-back, and the
adapter path of the ds41-mi355x preset (pause -> worker-extension RPC -> resume;
anything else restarts; a failed RPC kills the server)."""
from __future__ import annotations

import json

import pytest

from yeto.rl.adapters.sampler_engines import vllm_linear_reload as vlr
from yeto.rl.adapters.sampler_engines.vllm_server import MODEL_PRESETS, VllmOptions, VllmServerEngine
from yeto.rl.engine.release_overlay import overlay_release
from yeto.rl.engine.sampler_engine import SamplerEngineError

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

ROCM_CARDS = json.dumps([{"gpu": i} for i in range(8)])
PROJS = ("wq_a", "wkv", "wq_b", "wo_a", "wo_b")


def _attn(layer, proj, kind):
    return f"layers.{layer}.attn.{proj}.{kind}"


def _write_index(directory, weight_map):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / vlr.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    for file in set(weight_map.values()):
        (directory / file).write_bytes(b"")


def _base(directory):
    weight_map = {_attn(layer, proj, kind): f"model-0000{layer + 1}.safetensors"
                  for layer in range(2) for proj in PROJS for kind in vlr.KINDS}
    weight_map.update({"layers.0.ffn.experts.0.w1.weight": "model-00003.safetensors",
                       "layers.0.ffn.experts.0.w1.scale": "model-00003.safetensors",
                       "embed.weight": "model-00003.safetensors"})
    _write_index(directory, weight_map)
    (directory / "config.json").write_text(json.dumps({"architectures": ["DeepseekV41ForCausalLM"]}))
    return weight_map


def _release(tmp_path, base, name, tensors):
    partial = tmp_path / f"{name}.partial"
    _write_index(partial, {t: "lora-merged.safetensors" for t in tensors})
    return overlay_release(base, partial, tmp_path / name)


# ------------------------------------------------------------------ plan
def test_plan_stages_fused_modules_whole_and_refuses_anything_but_attention():
    weight_map = {_attn(3, p, k): "f" for p in PROJS for k in vlr.KINDS}
    plan = vlr.plan_modules([_attn(3, "wkv", "weight"), _attn(3, "wo_b", "scale")], weight_map)
    assert plan == {(3, "fused_wqa_wkv"): [_attn(3, "wq_a", "weight"), _attn(3, "wq_a", "scale"),
                                           _attn(3, "wkv", "weight"), _attn(3, "wkv", "scale")],
                    (3, "wo_b"): [_attn(3, "wo_b", "weight"), _attn(3, "wo_b", "scale")]}
    assert vlr.summarize(plan) == {"modules": 2, "layers": 1, "tensors": 6}
    for name in ("layers.3.ffn.experts.7.w1.weight", "layers.3.ffn.shared_experts.w2.weight",
                 "layers.1.engram.embed.weight", "layers.3.attn.compressor.wkv.weight", "mtp.0.attn.wq_a.weight",
                 "layers.3.attn_norm.weight"):
        with pytest.raises(vlr.LinearReloadError, match="not an attention projection"):
            vlr.plan_modules([name], weight_map)
    with pytest.raises(vlr.LinearReloadError, match="lacks"):
        vlr.plan_modules([_attn(3, "wq_a", "weight")], {_attn(3, "wq_a", "weight"): "f"})


def test_released_tensors_reads_the_overlay_then_the_merge_record(tmp_path):
    base = tmp_path / "base"
    _base(base)
    v1 = _release(tmp_path, base, "v1.merged", [_attn(0, "wq_b", "weight"), _attn(0, "wq_b", "scale")])
    assert vlr.released_tensors(v1) == [_attn(0, "wq_b", "scale"), _attn(0, "wq_b", "weight")]
    record = tmp_path / "plain"
    _write_index(record, {"embed.weight": "a.safetensors"})
    (record / vlr.MERGE_RECORD).write_text(json.dumps({"merged": [_attn(1, "wo_a", "weight")]}))
    assert vlr.released_tensors(record) == [_attn(1, "wo_a", "weight")]


def test_update_plan_folds_in_the_served_release_and_restarts_the_rest(tmp_path):
    base = tmp_path / "base"
    _base(base)
    v1 = _release(tmp_path, base, "v1.merged", [_attn(0, "wq_a", "weight"), _attn(0, "wq_a", "scale")])
    v2 = _release(tmp_path, base, "v2.merged", [_attn(1, "wo_b", "weight")])
    assert vlr.overlay_names(base, base) == []
    assert vlr.update_plan(v1, base, base) == ([_attn(0, "wq_a", "scale"), _attn(0, "wq_a", "weight")],
                                               "attention_overlay")
    names, why = vlr.update_plan(v2, v1, base)  # v1's module goes back to the base
    assert why == "attention_overlay" and set(names) == {_attn(0, "wq_a", "weight"), _attn(0, "wq_a", "scale"),
                                                         _attn(1, "wo_b", "weight")}
    back, _ = vlr.update_plan(base, v2, base)
    assert back == [_attn(1, "wo_b", "weight")]
    experts = _release(tmp_path, base, "ex.merged", ["layers.0.ffn.experts.0.w1.weight"])
    assert vlr.update_plan(experts, base, base) == (None, "not_attention_only")
    assert vlr.update_plan(v1, experts, base) == (None, "not_attention_only")
    plain = tmp_path / "plain"  # a full checkpoint, not an overlay of the base
    _base(plain)
    assert vlr.update_plan(plain, base, base) == (None, "not_an_overlay")
    assert vlr.update_plan(v1, plain, base) == (None, "served_not_an_overlay")
    other = tmp_path / "other"
    _base(other)
    foreign = _release(tmp_path, other, "foreign.merged", [_attn(0, "wq_b", "weight")])
    assert vlr.update_plan(foreign, base, base) == (None, "not_an_overlay")  # an overlay of another base


def test_put_back_restores_none_entries_and_objects():
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(4, 3, bias=False)
    module.register_buffer("scratch", torch.ones(2), persistent=False)
    module.derived = torch.zeros(1)
    weight = module.weight
    saved = vlr._snapshot(module, torch)
    assert "bias" in saved[0] and saved[0]["bias"] is None
    del module._parameters["bias"]
    module._parameters["weight"] = torch.nn.Parameter(torch.zeros(3, 4))
    module._non_persistent_buffers_set.clear()
    module.derived = torch.ones(1)
    vlr._put_back(module, saved)
    assert module.weight is weight and module.bias is None and "bias" in module._parameters
    assert "scratch" in module._non_persistent_buffers_set and module.derived.eq(0).all()
    module(torch.ones(1, 4))  # the forward reads self.bias
    served = torch.zeros(4, dtype=torch.float16)
    vlr._u8(served, torch).copy_(vlr._u8(torch.full((4,), 1.5, dtype=torch.float16), torch))
    assert served.eq(1.5).all()


# ------------------------------------------------------------------ adapter
def _ds41_config(tmp_path, **kw):
    config = _config(tmp_path, in_place_reload=True, quant_scheme="FP8_DENSE_MXFP4_EXPERTS", precision="fp8", **kw)
    _base(config.base_dir)
    return config


def _engine(tmp_path, routes, **options):
    http = _Http({("GET", "/health"): {}, **routes})
    engine = VllmServerEngine(_ds41_config(tmp_path), VllmOptions(platform="rocm", tp=8, model_preset="ds41-mi355x",
                                                                  **options),
                              spawn=lambda a, e: _Process(), http=http, sleep=lambda s: None,
                              run=lambda argv: ROCM_CARDS)
    return engine, http


def test_ds41_preset_reloads_attention_in_place_through_the_extension(tmp_path):
    assert MODEL_PRESETS["ds41-mi355x"]["in_place_reload"] == "linear"
    calls = []

    def rpc(body):
        calls.append(body)
        return {"results": [{"modules": 1, "seconds": 0.72, "staging_peak_bytes": 9953280}] * 8}

    routes = {("POST", "/pause?mode=wait"): {}, ("POST", "/collective_rpc"): rpc, ("POST", "/resume"): {}}
    engine, http = _engine(tmp_path, routes)
    base = engine.config.base_dir
    argv, env = engine.argv(base), engine.env()
    assert argv[argv.index("--worker-extension-cls") + 1] == vlr.EXTENSION and argv.count("--worker-extension-cls") == 1
    assert env["VLLM_SERVER_DEV_MODE"] == "1"
    engine.start(None, 0)
    v1 = _release(tmp_path, base, "v1.merged", [_attn(0, "wq_a", "weight"), _attn(0, "wq_a", "scale")])
    report = engine.load(v1, 1)
    assert report.reload_mode == "in_place" and report.detail["reload"] == "collective_rpc " + vlr.RELOAD_METHOD
    assert report.detail["rank_seconds"] == 0.72 and report.detail["tensors"] == 2
    assert [p for m, p, b in http.calls if m == "POST"] == ["/pause?mode=wait", "/collective_rpc", "/resume"]
    assert calls[0]["method"] == vlr.RELOAD_METHOD and calls[0]["kwargs"]["weights_path"] == str(v1)
    v2 = _release(tmp_path, base, "v2.merged", [_attn(1, "wo_a", "weight")])
    engine.load(v2, 2)
    assert set(json.loads(calls[1]["kwargs"]["names"])) == {_attn(0, "wq_a", "weight"), _attn(0, "wq_a", "scale"),
                                                            _attn(1, "wo_a", "weight")}
    assert all(b is None or b.get("method") != "reload_weights" for m, p, b in http.calls if p == "/collective_rpc")


def test_ds41_release_outside_attention_restarts(tmp_path):
    routes = {("POST", "/pause?mode=wait"): {}, ("POST", "/collective_rpc"): {"results": []}, ("POST", "/resume"): {}}
    engine, http = _engine(tmp_path, routes)
    engine.start(None, 0)
    experts = _release(tmp_path, engine.config.base_dir, "ex.merged", ["layers.0.ffn.experts.0.w1.weight"])
    report = engine.load(experts, 1)
    assert report.detail["reload"] == "restart" and report.detail["linear_reload_off"] == "not_attention_only"
    assert not [p for m, p, b in http.calls if p in ("/pause?mode=wait", "/collective_rpc")]
    assert engine._served_dir == experts


def test_ds41_failed_reload_kills_the_server_so_the_next_load_restarts(tmp_path):
    def broken(body):
        raise RuntimeError("rank 5: shape mismatch")

    routes = {("POST", "/pause?mode=wait"): {}, ("POST", "/collective_rpc"): broken, ("POST", "/resume"): {}}
    engine, http = _engine(tmp_path, routes)
    engine.start(None, 0)
    v1 = _release(tmp_path, engine.config.base_dir, "v1.merged", [_attn(0, "wq_b", "weight")])
    with pytest.raises(Exception, match="shape mismatch") as caught:
        engine.load(v1, 1)
    assert any("failed part way" in note for note in caught.value.__notes__)
    assert [p for m, p, b in http.calls if m == "POST"][-1] == "/resume"
    assert engine._process is None and engine._served_dir is None
    assert engine.load(v1, 1).reload_mode == "restart"


def test_ds41_linear_reload_excludes_the_k31_expert_reload(tmp_path):
    with pytest.raises(SamplerEngineError, match="not both"):
        _engine(tmp_path, {}, expert_reload=True)
