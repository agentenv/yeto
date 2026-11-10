"""K3.1 engine sampler: the score_centering fields (top-K log-probs, fp32 head,
engine.version_fields) and the spec's ``rollout_fields`` check."""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from yeto.rl.adapters.sampler_engines import from_spec
from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.adapters.sampler_engines.server import ServerConfig
from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine, decode_top_logprobs
from yeto.rl.adapters.sampler_engines.vllm_server import VllmOptions, VllmServerEngine
from yeto.rl.adapters.sampler_engines.vllm_server import decode_top_logprobs as vllm_top
from yeto.rl.engine import buffer_payload as bp
from yeto.rl.engine.engine_sampler_entry import SamplerSpecError, build_engine, check_rollout_fields
from yeto.rl.engine.sampler_engine import SamplerEngineError, SamplingParams

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _flat_b64(k, tokens=2, null_prefix=0):
    ids = (np.arange(tokens * k, dtype="<i4").reshape(tokens, k) + 100)[null_prefix:]
    vals = (-np.arange(tokens * k, dtype="<f4").reshape(tokens, k) / 10)[null_prefix:]
    return {"output_top_logprobs_idx_flat_b64": _b64(ids), "output_top_logprobs_val_flat_b64": _b64(vals),
            "output_top_logprobs_idx_flat_b64_dtype": "int32", "output_top_logprobs_val_flat_b64_dtype": "float32",
            "output_top_logprobs_shape": [tokens - null_prefix, k], "output_top_logprobs_null_prefix": null_prefix}


def test_decode_top_logprobs_reads_flat_b64_flat_lists_and_nested_rows():
    ids, vals = decode_top_logprobs(_flat_b64(4), completion_tokens=2, k=4)
    assert ids.dtype == np.int32 and vals.dtype == np.float32 and ids.shape == (2, 4)
    assert ids[1, 0] == 104 and vals[1, 0] == np.float32(-0.4)
    ids, vals = decode_top_logprobs(_flat_b64(4, null_prefix=1), completion_tokens=2, k=4)
    assert ids[0].tolist() == [-1] * 4 and np.isneginf(vals[0]).all() and ids[1, 0] == 104
    flat = {"output_top_logprobs_idx_flat": [1, 2, 3, 4], "output_top_logprobs_val_flat": [-1, -2, -3, -4],
            "output_top_logprobs_shape": [2, 2]}
    ids, vals = decode_top_logprobs(flat, completion_tokens=2, k=2)
    assert ids.tolist() == [[1, 2], [3, 4]]
    nested = {"output_top_logprobs": [[[-0.1, 5, None], [-0.9, 6, None]], [[-0.2, 7, None]]]}
    ids, vals = decode_top_logprobs(nested, completion_tokens=2, k=2)
    assert ids.tolist() == [[5, 6], [7, -1]] and np.isneginf(vals[1, 1])
    assert decode_top_logprobs({}, completion_tokens=2, k=0) is None
    assert decode_top_logprobs({}, completion_tokens=2, k=2) is None
    with pytest.raises(SamplerEngineError, match="positions"):
        decode_top_logprobs(_flat_b64(4), completion_tokens=3, k=4)
    with pytest.raises(SamplerEngineError, match="candidates"):
        decode_top_logprobs(_flat_b64(4), completion_tokens=2, k=8)


def _sglang_generate(k):
    def handler(body):
        out = []
        for _ in body["text"]:
            meta = {"output_token_logprobs": [[-0.1, 7, None], [-0.2, 8, None]], "finish_reason": {"type": "stop"},
                    "prompt_tokens": 3, **(_flat_b64(k) if k else {})}
            out.append({"text": "x", "meta_info": meta})
        return out
    return handler


def test_sglang_top_logprobs_and_fp32_head_flags_requests_and_arrays(tmp_path):
    http = _Http({("GET", "/health"): {}, ("POST", "/generate"): _sglang_generate(128)})
    spawned = []
    engine = SGLangServerEngine(_config(tmp_path), SGLangOptions(top_logprobs_num=128, fp32_lm_head=True),
                                spawn=lambda a, e: spawned.append(a) or _Process(), http=http, sleep=lambda s: None)
    engine.start(None, 0)
    assert "--enable-fp32-lm-head" in spawned[0] and engine.fp32_lm_head and engine.top_logprobs_num == 128
    group = engine.generate(["abc"], SamplingParams(max_tokens=2))[0]
    sent = [b for m, p, b in http.calls if p == "/generate"][0]
    assert sent["top_logprobs_num"] == 128 and sent["return_flat_raw_output_top_logprobs"] is True
    assert sent["return_flat_raw_top_logprobs_b64"] is True
    assert group[0].top_logprobs_ids.shape == (2, 128) and group[0].top_logprobs.dtype == np.float32
    # a server that answers without the field is an error, not a silent empty sample
    http.routes[("POST", "/generate")] = _sglang_generate(0)
    with pytest.raises(SamplerEngineError, match="no top log-probs"):
        engine.generate(["abc"], SamplingParams(max_tokens=2))


def test_sglang_off_by_default_sends_no_top_k_and_no_fp32_flag(tmp_path):
    spawned = []
    engine = SGLangServerEngine(_config(tmp_path), spawn=lambda a, e: spawned.append(a) or _Process(),
                                http=_Http({("GET", "/health"): {}, ("POST", "/generate"): _sglang_generate(0)}),
                                sleep=lambda s: None)
    engine.start(None, 0)
    assert "--enable-fp32-lm-head" not in spawned[0]
    group = engine.generate(["abc"], SamplingParams(max_tokens=2))[0]
    sent = [b for m, p, b in engine._http.calls if p == "/generate"][0]
    assert "top_logprobs_num" not in sent and group[0].top_logprobs is None


def test_vllm_top_logprobs_from_the_openai_shape(tmp_path):
    rows = [{"token_id:7": -0.1, "token_id:9": -2.0}, {"token_id:8": -0.2}]
    ids, vals = vllm_top({"top_logprobs": rows}, k=2)
    assert ids.tolist() == [[7, 9], [8, -1]] and vals.dtype == np.float32 and np.isneginf(vals[1, 1])
    base = tmp_path / "base"
    base.mkdir()
    def completions(body):
        assert body["logprobs"] == 2
        return {"choices": [{"index": 0, "text": "x", "finish_reason": "stop",
                             "logprobs": {"tokens": ["token_id:7", "token_id:8"], "token_logprobs": [-0.1, -0.2],
                                          "top_logprobs": rows}}]}
    engine = VllmServerEngine(ServerConfig(base_dir=base, card_model="910B4", device_family="npu", engine_version="0.30",
                                           precision="bf16", quant_scheme="bf16", topology="tp4", max_model_len=4096),
                              VllmOptions(top_logprobs_num=2), spawn=lambda a, e: _Process(),
                              http=_Http({("GET", "/health"): {}, ("POST", "/v1/completions"): completions}),
                              sleep=lambda s: None)
    engine.start(None, 0)
    group = engine.generate(["abc"], SamplingParams(max_tokens=2))[0]
    assert group[0].top_logprobs_ids.tolist() == [[7, 9], [8, -1]]
    with pytest.raises(SamplerEngineError, match="no top_logprobs"):
        vllm_top({}, k=2)


def _fake_spec(**fields):
    spec = {"engine": {"factory": from_spec.FACTORY, "adapter": "fake", "kwargs": {}},
            "sampling": {"max_tokens": 8, "temperature": 0.9, "top_k": 20}}
    if fields:
        spec["rollout_fields"] = fields
    return spec


def test_rollout_fields_reach_the_fake_engine_and_version_fields(tmp_path):
    engine = build_engine(_fake_spec(routed_experts=True, fp32_lm_head=True))
    assert isinstance(engine, FakeEngine) and engine.returns_routed_experts and engine.top_logprobs_num == 128
    have = check_rollout_fields(_fake_spec(routed_experts=True, fp32_lm_head=True), engine)
    assert have["version_fields"] == {"fp32_lm_head": True, "temperature": 0.9, "top_p": 1.0, "top_k": 20}
    engine.start(None, 0)
    c = engine.generate(["p"], SamplingParams(max_tokens=4))[0][0]
    assert c.top_logprobs_ids.shape == (4, 128) and c.top_logprobs_ids[0, 0] == c.token_ids[0]
    # explicit K and an explicit "no"
    assert build_engine(_fake_spec(top_logprobs_num=16)).top_logprobs_num == 16
    assert build_engine(_fake_spec(top_logprobs_num=0, routed_experts=False)).top_logprobs_num == 0
    assert build_engine(_fake_spec()).top_logprobs_num == 0  # no section: nothing changes


def test_rollout_fields_spec_errors():
    with pytest.raises(SamplerSpecError, match="keys"):
        build_engine(_fake_spec(indexer_topk=True))
    with pytest.raises(SamplerSpecError, match="top_logprobs_num"):
        build_engine(_fake_spec(top_logprobs_num=-1))
    # an engine built without the capability fails the check
    with pytest.raises(SamplerSpecError, match="routed experts"):
        check_rollout_fields(_fake_spec(routed_experts=True), FakeEngine())
    with pytest.raises(SamplerSpecError, match="returns 0"):
        check_rollout_fields(_fake_spec(top_logprobs_num=8), FakeEngine())
    with pytest.raises(SamplerSpecError, match="fp32"):
        check_rollout_fields(_fake_spec(fp32_lm_head=True, top_logprobs_num=0), FakeEngine())


def test_rollout_fields_map_onto_sglang_and_refuse_legacy_vllm_r3(tmp_path):
    base = tmp_path / "b"
    base.mkdir()
    server = {"base_dir": str(base), "card_model": "MI355X", "device_family": "rocm", "engine_version": "x",
              "precision": "mxfp4", "quant_scheme": "MXFP4_EXPERTS_RTN", "topology": "tp8", "max_model_len": 1024}
    section = {"adapter": "sglang-server", "server": server, "options": {"tp": 8},
               "rollout_fields": {"routed_experts": True, "fp32_lm_head": True}}
    engine = from_spec.build(section)
    assert engine.options.return_routed_experts and engine.options.fp32_lm_head and engine.options.top_logprobs_num == 128
    with pytest.raises(ValueError, match="options.top_logprobs_num"):
        from_spec.build({**section, "options": {"tp": 8, "top_logprobs_num": 4}})
    vllm = {"adapter": "vllm-server", "server": {**server, "topology": "tp4", "precision": "bf16", "quant_scheme": "bf16"},
            "options": {"tp": 4}, "rollout_fields": {"routed_experts": True}}
    with pytest.raises(ValueError, match="R3\\) only on upstream vLLM"):
        from_spec.build(vllm)  # the legacy line (no platform: vllm-ascend / the verl image)
    upstream = from_spec.build({**vllm, "platform": "cuda"})
    assert upstream.options.return_routed_experts and upstream.options.top_logprobs_num == 128
    vllm["rollout_fields"] = {"top_logprobs_num": 64}
    assert from_spec.build(vllm).options.top_logprobs_num == 64


def test_island_writes_top_k_fields_and_engine_meta(tmp_path):
    from yeto.rl.engine.engine_sampler import PromptItem, sample_record, engine_meta

    engine = FakeEngine(routed_experts=(896, 2, 4), top_logprobs_num=3, fp32_lm_head=True)
    engine.start(None, 1)
    sampling = SamplingParams(max_tokens=4)
    c = engine.generate(["q"], sampling)[0][0]
    sample = sample_record(PromptItem("p", "q"), c, reward=0.0, version=1, provenance={}, group_index=0, index=0,
                           engine_meta=engine_meta(engine, sampling))
    back = bp.decode_group(bp.encode_group([sample]))[0]
    ids, vals = bp.rollout_top_logprobs(back)
    assert ids.dtype == np.int32 and vals.dtype == np.float32 and ids.shape == (4, 3)
    assert back["metadata"]["engine"] == {"version_fields": {"fp32_lm_head": True, "temperature": 1.0, "top_p": 1.0,
                                                            "top_k": -1}, "routed_experts": True, "top_logprobs_num": 3}
    assert bp.rollout_routed_experts(back).shape == (4, 2, 4)
    assert bp.rollout_top_logprobs({"index": 0}) is None
    with pytest.raises(bp.SampleBufferRolloutError, match="come together"):
        bp.rollout_top_logprobs({"rollout_top_logprobs": vals})
    with pytest.raises(bp.SampleBufferRolloutError, match="response_length"):
        bp.rollout_top_logprobs({"rollout_top_logprobs": vals, "rollout_top_logprobs_ids": ids, "response_length": 3})
