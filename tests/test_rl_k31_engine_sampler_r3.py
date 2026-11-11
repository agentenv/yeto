"""K3.1 engine sampler: R3 routed experts through the port, the SGLang adapter
and the buffer payload (uint16, compatible with uint8 payloads)."""
from __future__ import annotations

import base64
import json
from dataclasses import replace

import numpy as np
import pytest

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.adapters.sampler_engines.server import ServerConfig
from yeto.rl.adapters.sampler_engines.sglang_server import (
    SGLangOptions,
    SGLangServerEngine,
    decode_routed_experts,
    moe_layer_count,
)
from yeto.rl.adapters.sampler_engines.vllm_server import VllmServerEngine
from yeto.rl.engine import buffer_payload as bp
from yeto.rl.engine.engine_sampler import PromptItem, engine_meta, sample_record
from yeto.rl.engine.loop_guard import LoopGuard
from yeto.rl.engine.sampler_engine import (
    FINISH_LOOP_ABORT,
    FINISH_STOP,
    Completion,
    Provenance,
    SamplerEngineError,
    SamplingParams,
    returns_routed_experts,
)


class _Process:
    def __init__(self, code=0):
        self.code, self.polled = code, None
        self.pid = 4242

    def poll(self):
        return self.polled

    def send_signal(self, sig):
        self.polled = self.code

    def kill(self):
        self.polled = -9

    def wait(self, timeout=None):
        return self.polled if self.polled is not None else self.code


class _Http:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, method, url, body, timeout):
        path = url.split(":30000", 1)[1]
        self.calls.append((method, path, body))
        handler = self.routes.get((method, path))
        return (404, None) if handler is None else (200, handler(body) if callable(handler) else handler)


def _config(tmp_path, **kw):
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    (base / "config.json").write_text(json.dumps({
        "architectures": ["KimiK31ForConditionalGeneration"], "num_hidden_layers": 4,
        "first_k_dense_replace": 1, "n_routed_experts": 896}))
    values = dict(base_dir=base, card_model="MI355X", device_family="rocm", engine_version="k31-35708f76",
                  precision="mxfp4", quant_scheme="MXFP4_EXPERTS_RTN", topology="tp8", max_model_len=32768)
    values.update(kw)
    return ServerConfig(**values)


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _generate(experts):
    """A /generate handler returning 2 output tokens and R3 ids for 3 MoE layers."""
    def handler(body):
        prompts = body.get("text") or body.get("input_ids")
        out = []
        for p in prompts:
            prompt_tokens = len(p) if isinstance(p, list) else 3
            rows = prompt_tokens + 2 - 1
            ids = experts[:rows] if experts is not None else None
            meta = {"output_token_logprobs": [[-0.1, 7, None], [-0.2, 8, None]],
                    "finish_reason": {"type": "stop"}, "prompt_tokens": prompt_tokens}
            if ids is not None:
                meta["routed_experts"] = _b64(ids.astype("<i4"))
            out.append({"text": "x", "meta_info": meta})
        return out
    return handler


def test_decode_routed_experts_follows_the_miles_shape_and_refuses_bad_payloads():
    ids = np.arange(4 * 3 * 2, dtype=np.int32).reshape(4, 3, 2) + 700  # ids above uint8
    array = decode_routed_experts({"routed_experts": _b64(ids.astype("<i4"))}, prompt_tokens=3,
                                  completion_tokens=2, moe_layers=3)
    assert array.dtype == np.uint16 and array.shape == (4, 3, 2) and int(array[0, 0, 0]) == 700
    assert decode_routed_experts({}, prompt_tokens=3, completion_tokens=2, moe_layers=3) is None
    with pytest.raises(SamplerEngineError, match="positions"):
        decode_routed_experts({"routed_experts": _b64(ids.astype("<i4"))}, prompt_tokens=9, completion_tokens=2,
                              moe_layers=3)
    with pytest.raises(SamplerEngineError, match="MoE layers"):
        decode_routed_experts({"routed_experts": _b64(ids.astype("<i4"))}, prompt_tokens=3, completion_tokens=2,
                              moe_layers=5)
    with pytest.raises(SamplerEngineError, match="all 0"):
        decode_routed_experts({"routed_experts": _b64(np.zeros(24, dtype="<i4"))}, prompt_tokens=3,
                              completion_tokens=2, moe_layers=3)
    with pytest.raises(SamplerEngineError, match="uint16"):
        decode_routed_experts({"routed_experts": _b64(np.full(24, 70000, dtype="<i4"))}, prompt_tokens=3,
                              completion_tokens=2, moe_layers=3)
    with pytest.raises(SamplerEngineError, match="layer count is unknown"):
        decode_routed_experts({"routed_experts": _b64(ids.astype("<i4"))}, prompt_tokens=3, completion_tokens=2)
    # the server may send the layer count itself
    got = decode_routed_experts({"routed_experts": _b64(ids.astype("<i4")), "routed_experts_num_layers": 3},
                                prompt_tokens=3, completion_tokens=2)
    assert got.shape == (4, 3, 2)


def test_moe_layer_count_reads_the_release_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 61, "first_k_dense_replace": 1,
                                                      "n_routed_experts": 896}))
    assert moe_layer_count(tmp_path) == 60
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 61}))
    assert moe_layer_count(tmp_path) is None  # dense model: no MoE layers
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 8, "num_experts": 4}}))
    assert moe_layer_count(tmp_path) == 8


def test_sglang_r3_flag_request_field_and_uint16_completion(tmp_path):
    experts = np.arange(10 * 3 * 8, dtype=np.int32).reshape(10, 3, 8) + 500
    http = _Http({("GET", "/health"): {}, ("POST", "/generate"): _generate(experts)})
    spawned = []
    engine = SGLangServerEngine(_config(tmp_path), SGLangOptions(return_routed_experts=True),
                                spawn=lambda a, e: spawned.append(a) or _Process(), http=http,
                                sleep=lambda s: None)
    engine.start(None, 0)
    assert "--enable-return-routed-experts" in spawned[0]
    assert returns_routed_experts(engine)
    groups = engine.generate(["abc"], SamplingParams(max_tokens=2))
    sent = [b for m, p, b in http.calls if p == "/generate"][0]
    assert sent["return_routed_experts"] is True and sent["return_logprob"] is True
    completion = groups[0][0]
    # text prompt: meta_info.prompt_tokens (3) + 2 output tokens - 1 rows, L_moe = 4 - 1 from config.json
    assert completion.routed_experts.dtype == np.uint16 and completion.routed_experts.shape == (4, 3, 8)
    # id prompts: the prompt length is the request's
    groups = engine.generate_tokens([[1, 2, 3, 4, 5]], SamplingParams(max_tokens=2))
    assert groups[0][0].routed_experts.shape == (6, 3, 8)


def test_sglang_without_r3_sends_no_flag_and_no_experts(tmp_path):
    http = _Http({("GET", "/health"): {}, ("POST", "/generate"): _generate(None)})
    spawned = []
    engine = SGLangServerEngine(_config(tmp_path), spawn=lambda a, e: spawned.append(a) or _Process(),
                                http=http, sleep=lambda s: None)
    engine.start(None, 0)
    assert "--enable-return-routed-experts" not in spawned[0] and not returns_routed_experts(engine)
    groups = engine.generate(["abc"], SamplingParams(max_tokens=2))
    sent = [b for m, p, b in http.calls if p == "/generate"][0]
    assert "return_routed_experts" not in sent and groups[0][0].routed_experts is None


def test_sglang_r3_needs_a_moe_layer_count(tmp_path):
    config = _config(tmp_path)
    (config.base_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 4}))
    engine = SGLangServerEngine(config, SGLangOptions(return_routed_experts=True), spawn=lambda a, e: _Process(),
                                http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    with pytest.raises(SamplerEngineError, match="MoE layer count"):
        engine.start(None, 0)
    explicit = SGLangServerEngine(config, SGLangOptions(return_routed_experts=True, moe_layers=3),
                                  spawn=lambda a, e: _Process(), http=_Http({("GET", "/health"): {}}),
                                  sleep=lambda s: None)
    explicit.start(None, 0)


def test_vllm_engine_declares_no_routed_experts_and_provenance_says_false(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    engine = VllmServerEngine(ServerConfig(base_dir=base, card_model="910B4", device_family="npu",
                                           engine_version="0.30", precision="bf16", quant_scheme="bf16",
                                           topology="tp4", max_model_len=4096),
                              spawn=lambda a, e: _Process(), http=_Http({}))
    assert not returns_routed_experts(engine)
    provenance = Provenance.of(engine.identity, sampler_id="s1", sampling=SamplingParams(max_tokens=4),
                               routed_experts=returns_routed_experts(engine))
    assert provenance.routed_experts == "false" and provenance.returns_routed_experts() is False
    record = provenance.to_record()
    assert record["routed_experts"] == "false"
    # an index record written before the field reads as unknown
    old = Provenance.from_record({k: v for k, v in record.items() if k != "routed_experts"})
    assert old.routed_experts == "unknown" and old.returns_routed_experts() is None
    assert Provenance.of(engine.identity, sampler_id="s1", sampling=SamplingParams(max_tokens=4),
                         routed_experts=True).routed_experts == "true"


def _completion(experts=None, prompt=(1, 2, 3), **kw):
    return Completion(token_ids=(7, 8, 9), logprobs=(-0.1, -0.2, -0.3), text="x", finish_reason=FINISH_STOP,
                      prompt_token_ids=prompt, routed_experts=experts, **kw)


def test_payload_round_trip_keeps_uint16_and_reads_uint8_payloads_as_uint16():
    experts = (np.arange(5 * 2 * 4, dtype=np.int32).reshape(5, 2, 4) * 7 % 896).astype(np.uint16)
    item = PromptItem(prompt_id="p", prompt="hi")
    sample = sample_record(item, _completion(experts), reward=1.0, version=3, provenance={"engine": "fake"},
                           group_index=0, index=0, engine_meta={"version_fields": {"fp32_lm_head": True}})
    assert sample[bp.ROLLOUT_ROUTED_EXPERTS].dtype == np.uint16
    assert sample["metadata"][bp.ENGINE_META]["version_fields"]["fp32_lm_head"] is True
    back = bp.decode_group(bp.encode_group([sample]))[0]
    got = bp.rollout_routed_experts(back)
    assert got.dtype == np.uint16 and np.array_equal(got, experts)
    # an earlier writer stored uint8 (128 experts) without a cast: still uint16 for the reader
    old = bp.decode_group(json.dumps({"schema": bp.PAYLOAD_SCHEMA, "samples": [{
        "index": 0, bp.ROLLOUT_ROUTED_EXPERTS: {"__ndarray__": _b64(np.arange(8, dtype=np.uint8).reshape(2, 2, 2)),
                                                "dtype": "uint8", "shape": [2, 2, 2]}}]}).encode())[0]
    old_experts = bp.rollout_routed_experts(old)
    assert old_experts.dtype == np.uint16 and old_experts.tolist() == [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    # a Miles-style int32 payload reads the same way; a sample without the field gives None
    assert bp.rollout_routed_experts({bp.ROLLOUT_ROUTED_EXPERTS: np.ones((1, 1, 1), dtype=np.int32)}).dtype == np.uint16
    assert bp.rollout_routed_experts({"index": 1}) is None
    with pytest.raises(bp.SampleBufferRolloutError, match="outside uint16"):
        bp.rollout_routed_experts({bp.ROLLOUT_ROUTED_EXPERTS: np.full((1, 1, 1), 70000, dtype=np.int32)})
    with pytest.raises(bp.SampleBufferRolloutError, match="outside uint16"):
        bp.rollout_routed_experts({bp.ROLLOUT_ROUTED_EXPERTS: np.full((1, 1, 1), -2, dtype=np.int32)})


def test_padding_is_0xffff_and_declared_in_the_field_record():
    padded = np.array([[[3, -1]], [[-1, -1]]], dtype=np.int32)  # Miles-style -1 pads
    sample = sample_record(PromptItem("p", "hi"), _completion(padded), reward=0.0, version=1, provenance={},
                           group_index=0, index=0)
    assert sample[bp.ROLLOUT_ROUTED_EXPERTS].dtype == np.uint16
    assert sample[bp.ROLLOUT_ROUTED_EXPERTS].tolist() == [[[3, 65535]], [[65535, 65535]]]
    raw = json.loads(bp.encode_group([sample]))
    field = raw["samples"][0][bp.ROLLOUT_ROUTED_EXPERTS]
    assert field["pad"] == bp.ROUTED_EXPERTS_PAD == 65535 and field["dtype"] == "uint16"  # not narrowed to uint8
    back = bp.rollout_routed_experts(bp.decode_group(bp.encode_group([sample]))[0])
    assert back.tolist() == [[[3, 65535]], [[65535, 65535]]]
    # the SGLang wire may carry -1 pads too
    wire = np.array([[[700, -1]]], dtype="<i4")
    got = decode_routed_experts({"routed_experts": _b64(wire)}, prompt_tokens=1, completion_tokens=1, moe_layers=1)
    assert got.tolist() == [[[700, 65535]]]
    # other integer fields carry no pad key
    assert "pad" not in raw["samples"][0].get("tokens", {}) if isinstance(raw["samples"][0].get("tokens"), dict) else True
    with pytest.raises(bp.SampleBufferRolloutError, match="shape"):
        bp.rollout_routed_experts({bp.ROLLOUT_ROUTED_EXPERTS: np.ones((2, 2), dtype=np.uint16)})


def test_completion_cut_trims_every_per_token_array():
    experts = np.arange(5 * 2 * 4, dtype=np.uint16).reshape(5, 2, 4)
    top_ids = np.arange(3 * 2, dtype=np.int32).reshape(3, 2)
    top = np.zeros((3, 2), dtype=np.float32)
    c = _completion(experts, top_logprobs_ids=top_ids, top_logprobs=top)
    cut = c.cut(2)
    assert cut.token_ids == (7, 8) and cut.routed_experts.shape == (4, 2, 4)  # 3 prompt + 2 - 1 rows
    assert cut.top_logprobs_ids.shape == (2, 2) and cut.top_logprobs.shape == (2, 2)
    # without the prompt length the routed rows of the prompt are unknown: the array is dropped
    assert _completion(experts, prompt=()).cut(2).routed_experts is None
    guard = LoopGuard(max_period=1, repeats=2)
    looping = replace(c, token_ids=(5, 5, 5), logprobs=(-0.1, -0.1, -0.1))
    kept = guard.apply(looping)
    assert kept.finish_reason == FINISH_LOOP_ABORT and kept.top_logprobs.shape[0] == len(kept.token_ids)
    with pytest.raises(SamplerEngineError, match="top log-probs"):
        _completion(top_logprobs_ids=np.zeros((2, 2), dtype=np.int32), top_logprobs=np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(SamplerEngineError, match="come together"):
        _completion(top_logprobs_ids=np.zeros((3, 2), dtype=np.int32))


def test_fake_engine_returns_r3_arrays_and_meta(tmp_path):
    engine = FakeEngine(routed_experts=(896, 3, 8))
    engine.start(None, 0)
    assert returns_routed_experts(engine)
    group = engine.generate(["abc"], SamplingParams(max_tokens=4))[0]
    assert group[0].routed_experts.shape == (4, 3, 8) and group[0].routed_experts.dtype == np.uint16
    ids = engine.generate_tokens([(1, 2, 3)], SamplingParams(max_tokens=4))[0][0]
    assert ids.routed_experts.shape == (3 + 4 - 1, 3, 8)
    meta = engine_meta(engine, SamplingParams(max_tokens=4, temperature=0.7, top_k=50))
    assert meta == {"version_fields": {"fp32_lm_head": False, "temperature": 0.7, "top_p": 1.0, "top_k": 50},
                    "routed_experts": True, "top_logprobs_num": 0}


def test_decode_routed_experts_drops_the_leading_dense_layers_of_the_wire():
    """sglang-miles-kimi-k31@35708f76 returns one row per model layer (Miles reshapes
    with args.num_layers): Kimi-K3-4layer answers [T, 4, 16] with layer 0 all zeros
    (measured on NVIDIA). The island keeps the INTERFACES shape [T, L_moe, 16]."""
    from yeto.rl.adapters.sampler_engines.sglang_server import dense_layer_count

    rows, k = 5, 4
    full = np.zeros((rows, 4, k), dtype="<i4")
    full[:, 1:] = np.arange(rows * 3 * k, dtype="<i4").reshape(rows, 3, k) + 1
    meta = {"routed_experts": _b64(full)}
    got = decode_routed_experts(meta, prompt_tokens=3, completion_tokens=3, moe_layers=3, dense_layers=1)
    assert got.shape == (rows, 3, k) and got.dtype == np.uint16
    assert int(got[0, 0, 0]) == 1 and int(got[-1, -1, -1]) == rows * 3 * k
    # a wire that already has L_moe layers is taken as is
    moe_only = np.ascontiguousarray(full[:, 1:])
    same = decode_routed_experts({"routed_experts": _b64(moe_only)}, prompt_tokens=3, completion_tokens=3,
                                 moe_layers=3, dense_layers=1)
    assert np.array_equal(same, got)
    # expert ids in the dense rows and no L_moe layout either: not the expected layout
    wide = np.zeros((rows, 4, 5), dtype="<i4")
    wide[:, 1:] = 3
    wide[0, 0, 0] = 7
    with pytest.raises(SamplerEngineError, match="dense layer"):
        decode_routed_experts({"routed_experts": _b64(wide)}, prompt_tokens=3, completion_tokens=3,
                              moe_layers=3, dense_layers=1)
    # neither layer count divides
    with pytest.raises(SamplerEngineError, match="model layers"):
        decode_routed_experts({"routed_experts": _b64(full)}, prompt_tokens=3, completion_tokens=3,
                              moe_layers=5, dense_layers=1)


def test_dense_layer_count_reads_the_release_config(tmp_path):
    from yeto.rl.adapters.sampler_engines.sglang_server import dense_layer_count

    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 4, "first_k_dense_replace": 1,
                                                                      "num_experts": 896}}))
    assert dense_layer_count(tmp_path) == 1
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 8}))
    assert dense_layer_count(tmp_path) == 0
    assert dense_layer_count(tmp_path / "missing") == 0


def test_decode_routed_experts_with_top_k_takes_only_the_two_exact_layouts(tmp_path):
    """num_experts_per_tok fixes the row layout: Kimi-K3.1-Pro-4layer on
    sglang-miles-kimi-k31@35708f76 answers [T, 4, 16] (layer 0 dense, all 0;
    measured on B300, GPU-A), the island keeps [T, 3, 16]."""
    from yeto.rl.adapters.sampler_engines.sglang_server import routed_top_k

    rows = 5
    moe = (np.arange(rows * 3 * 16, dtype="<i4").reshape(rows, 3, 16) % 895) + 1
    wire = np.concatenate([np.zeros((rows, 1, 16), dtype="<i4"), moe], axis=1)
    kw = {"prompt_tokens": 3, "completion_tokens": 3, "moe_layers": 3, "dense_layers": 1, "top_k": 16}
    got = decode_routed_experts({"routed_experts": _b64(wire)}, **kw)
    assert got.shape == (rows, 3, 16) and np.array_equal(got.astype("<i4"), moe)
    same = decode_routed_experts({"routed_experts": _b64(np.ascontiguousarray(moe))}, **kw)
    assert np.array_equal(same, got)
    # dense rows holding the pad are no routing either
    padded = wire.copy()
    padded[:, 0] = -1
    assert np.array_equal(decode_routed_experts({"routed_experts": _b64(padded)}, **kw), got)
    # the model layout with ids in the dense rows is refused, even where L_moe would not divide
    bad = wire.copy()
    bad[0, 0, 0] = 7
    with pytest.raises(SamplerEngineError, match="dense layer"):
        decode_routed_experts({"routed_experts": _b64(bad)}, **kw)
    # a width that splits into layers but not into top-16 rows is refused (3 x 8 = 4 x 6 = 24)
    with pytest.raises(SamplerEngineError, match="top-16"):
        decode_routed_experts({"routed_experts": _b64(np.ones((rows, 24), dtype="<i4"))}, **kw)
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_experts_per_tok": 16}}))
    assert routed_top_k(tmp_path) == 16
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 4}))
    assert routed_top_k(tmp_path) is None
    assert routed_top_k(tmp_path / "missing") is None


def test_routed_top_k_reads_the_kimi_k31_spelling(tmp_path):
    """Kimi-K3.1-Pro's config.json (d9e99b07) names the top-k num_experts_per_token."""
    from yeto.rl.adapters.sampler_engines.sglang_server import routed_top_k

    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 93,
                                                                      "num_experts_per_token": 16}}))
    assert routed_top_k(tmp_path) == 16
