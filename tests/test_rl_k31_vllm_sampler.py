"""The sampler's upstream vLLM line (platform cuda / rocm) for Kimi-K3.1: pins, argv,
R3 decoding of the per-choice ``.npy`` routed experts, top-K, fp32 head, ROCm rules,
K3.1 presets and the in-place reload calls. The legacy line (vllm-ascend) keeps its argv."""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import re

import numpy as np
import pytest

from yeto.rl import sampler_pins
from yeto.rl.adapters.sampler_engines import from_spec
from yeto.rl.adapters.sampler_engines.quant_tier import QuantTierSpecError
from yeto.rl.adapters.sampler_engines.quant_tier import apply as apply_rollout_quant
from yeto.rl.adapters.sampler_engines.vllm_server import (
    CUDA_ENV_PRESET,
    DEV_MODE_ENV,
    K31_PRESETS,
    ROCM_ENV_PRESET,
    VllmOptions,
    VllmServerEngine,
    decode_routed_experts,
    drop_cp_sizes,
    merge_overrides,
)
from yeto.rl.engine.sampler_engine import SamplerEngineError, SamplingParams, returns_routed_experts, version_fields

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

DIGEST = re.compile(r"docker:docker\.io/vllm/vllm-openai(-rocm)?@sha256:[0-9a-f]{64}")
SHA = re.compile(r"[0-9a-f]{40}")
LAYERS, DENSE, TOPK = 4, 1, 16  # _config: num_hidden_layers 4, first_k_dense_replace 1
ROCM_CARDS = json.dumps([{"gpu": i} for i in range(8)])


def _npy(array):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _routing(rows, *, layers=LAYERS, dense=DENSE, dtype=np.uint16, seed=0):
    """A server payload: every model layer, the dense ones all zero."""
    rng = np.random.default_rng(seed)
    out = np.zeros((rows, layers, TOPK), dtype=dtype)
    for t in range(rows):
        for layer in range(dense, layers):
            out[t, layer] = rng.choice(896, TOPK, replace=False)
    return out


def _engine(tmp_path, platform="cuda", *, routes=None, config=None, run=None, **options):
    http = _Http({("GET", "/health"): {}, **(routes or {})})
    options.setdefault("tp", 8)
    engine = VllmServerEngine(config or _config(tmp_path, card_model="B300", device_family="nvidia"),
                              VllmOptions(platform=platform, **options), spawn=lambda a, e: _Process(),
                              http=http, sleep=lambda s: None, run=run or (lambda argv: ROCM_CARDS))
    return engine, http


# ------------------------------------------------------------------ pins
def test_vllm_sampler_pins_name_the_dedicated_kimi_k31_images_by_digest():
    for image in (sampler_pins.VLLM_SAMPLER_IMAGE_ROCM, sampler_pins.VLLM_SAMPLER_IMAGE_ROCM_NIGHTLY, sampler_pins.VLLM_SAMPLER_BASE_IMAGE_CUDA):
        assert DIGEST.fullmatch(image) or re.fullmatch(
            r"docker:[a-z0-9.-]+/[a-z0-9._/-]+/vllm-openai-rocm@sha256:[0-9a-f]{64}", image), image
    # the mirrored A-card image keeps the upstream digest
    assert sampler_pins.VLLM_SAMPLER_IMAGE_ROCM.endswith(
        "@sha256:5f47cc1004d9a7ac65f8a913725edb20c62c99a07e31cbf4ba21741bd0b4f9b1")
    assert SHA.fullmatch(sampler_pins.VLLM_SAMPLER_INTERFACE_COMMIT) and SHA.fullmatch(sampler_pins.VLLM_SAMPLER_MAIN_COMMIT)
    assert sampler_pins.VLLM_SAMPLER_IMAGE_ROCM_TAG == "vllm/vllm-openai-rocm:kimi-k31"
    assert sampler_pins.VLLM_SAMPLER_BASE_IMAGE_CUDA_TAG == "vllm/vllm-openai:kimi-k31"
    # the nightly that did not start K3.1 on MI355X is recorded apart, never the pin
    assert sampler_pins.VLLM_SAMPLER_IMAGE_ROCM_NIGHTLY != sampler_pins.VLLM_SAMPLER_IMAGE_ROCM
    assert sampler_pins.VLLM_SAMPLER_MAIN_VERSION.endswith("+g" + sampler_pins.VLLM_SAMPLER_MAIN_COMMIT[:9])
    assert sampler_pins.VLLM_SAMPLER_MAIN_COMMIT in sampler_pins.VLLM_SAMPLER_MAIN_WHEEL_CUDA
    assert sampler_pins.sampler_vllm_version("cuda") == sampler_pins.VLLM_SAMPLER_CUDA_VERSION
    assert sampler_pins.sampler_vllm_version("rocm") == sampler_pins.VLLM_SAMPLER_ROCM_VERSION
    assert sampler_pins.sampler_vllm_version("npu") is None
    for platform, label in sampler_pins.VLLM_SAMPLER_ENGINE_VERSIONS.items():
        assert label.endswith(sampler_pins.sampler_vllm_version(platform))
    # the yeto sampler image on the CUDA base, by digest, from a full yeto commit
    assert re.fullmatch(r"docker:ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}", sampler_pins.VLLM_SAMPLER_IMAGE_CUDA)
    assert SHA.fullmatch(sampler_pins.VLLM_SAMPLER_YETO_COMMIT)
    assert sampler_pins.VLLM_SAMPLER_IMAGE_CUDA != sampler_pins.SGLANG_SAMPLER_IMAGE_CUDA
    assert sampler_pins.VLLM_SAMPLER_IMAGE_MANIFEST == sampler_pins.SGLANG_SAMPLER_IMAGE_MANIFEST
    # the sampler vLLM moves apart from the sampler SGLang and the Miles training pins
    assert sampler_pins.SGLANG_SAMPLER_IMAGE_ROCM != sampler_pins.VLLM_SAMPLER_IMAGE_ROCM


def test_vllm_sampler_pins_are_exported_from_yeto_rl():
    import yeto.rl as rl

    assert rl.VLLM_SAMPLER_IMAGE_ROCM == sampler_pins.VLLM_SAMPLER_IMAGE_ROCM
    assert rl.VLLM_SAMPLER_BASE_IMAGE_CUDA == sampler_pins.VLLM_SAMPLER_BASE_IMAGE_CUDA
    assert rl.VLLM_SAMPLER_IMAGE_CUDA == sampler_pins.VLLM_SAMPLER_IMAGE_CUDA
    assert rl.sampler_vllm_version("rocm") == sampler_pins.VLLM_SAMPLER_ROCM_VERSION


# ------------------------------------------------------------------ argv / env
def test_legacy_argv_is_unchanged_and_refuses_upstream_only_fields(tmp_path):
    config = _config(tmp_path, card_model="910B4", device_family="npu", precision="bf16", quant_scheme="bf16",
                     topology="tp4", max_model_len=4096)
    for platform in (None, "npu"):
        engine = VllmServerEngine(config, VllmOptions(platform=platform, tp=4, top_logprobs_num=128))
        argv = engine.argv(config.base_dir)
        assert argv[:5] == ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", str(config.base_dir)]
        assert "--max-logprobs" not in argv and "--enable-return-routed-experts" not in argv
        for name in ("return_routed_experts", "fp32_lm_head"):
            with pytest.raises(SamplerEngineError, match=f"{name} needs upstream vLLM"):
                VllmServerEngine(config, VllmOptions(platform=platform, tp=4, **{name: True}))
    with pytest.raises(SamplerEngineError, match="platform must be one of"):
        VllmServerEngine(config, VllmOptions(platform="ppu"))


def test_upstream_argv_carries_r3_top_k_fp32_head_and_the_kimi_parsers(tmp_path):
    engine, _ = _engine(tmp_path, return_routed_experts=True, fp32_lm_head=True, top_logprobs_num=128,
                        max_num_seqs=16, tool_call_parser="kimi_k3", reasoning_parser="kimi_k3")
    base = engine.config.base_dir
    argv = engine.argv(base)
    assert argv[:5] == ["python", "-m", "vllm.entrypoints.cli.main", "serve", str(base)]
    assert argv[argv.index("--tensor-parallel-size") + 1] == "8" and argv[argv.index("--max-num-seqs") + 1] == "16"
    assert argv[argv.index("--max-logprobs") + 1] == "128" and "--enable-return-routed-experts" in argv
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {"head_dtype": "float32"}
    assert argv[argv.index("--tool-call-parser") + 1] == "kimi_k3" and "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--reasoning-parser") + 1] == "kimi_k3"
    assert returns_routed_experts(engine) and engine.top_logprobs_num == 128
    assert version_fields(engine, SamplingParams(max_tokens=4))["fp32_lm_head"] is True
    assert engine.env()["VLLM_ENGINE_READY_TIMEOUT_S"] == CUDA_ENV_PRESET["VLLM_ENGINE_READY_TIMEOUT_S"]
    # K at most the server default needs no flag; --hf-overrides only through the options
    small, _ = _engine(tmp_path, top_logprobs_num=20)
    assert "--max-logprobs" not in small.argv(base) and "--hf-overrides" not in small.argv(base)
    with pytest.raises(SamplerEngineError, match="hf_overrides"):
        _engine(tmp_path, config=_config(tmp_path, extra_args=("--hf-overrides", "{}")))
    with pytest.raises(SamplerEngineError, match="head_dtype"):
        _engine(tmp_path, fp32_lm_head=True, hf_overrides={"head_dtype": "bfloat16"})[0].argv(base)


def test_rocm_env_preset_card_check_and_k31_presets(tmp_path):
    config = _config(tmp_path, extra_env={"HSA_NO_SCRATCH_RECLAIM": "0"})
    engine, _ = _engine(tmp_path, "rocm", config=config, k31_preset="amd-article", fp32_lm_head=True)
    env = engine.env()
    assert env["VLLM_ROCM_USE_AITER"] == "1" and env["HSA_NO_SCRATCH_RECLAIM"] == "0"  # extra_env wins
    assert env["VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA"] == "1" and env["VLLM_ROCM_QUICK_REDUCE_QUANTIZATION"] == "INT4"
    argv = engine.argv(config.base_dir)
    assert argv[argv.index("--decode-context-parallel-size") + 1] == "8"
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8" and "--language-model-only" in argv
    assert json.loads(argv[argv.index("--attention-config") + 1])["indexer_kv_dtype"] == "mxfp4"
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {
        "head_dtype": "float32", "text_config": {"dsa_config": {"indexer_use_fp8": True}}}
    assert "--quantization-config" in argv  # the article's FP8 KDA input projections
    rl, _ = _engine(tmp_path, "rocm", config=config, k31_preset="rl-precision")
    rl_argv, rl_env = rl.argv(config.base_dir), rl.env()
    assert "--quantization-config" not in rl_argv
    assert rl_argv[rl_argv.index("--mamba-ssm-cache-dtype") + 1] == "float32"
    assert rl_env["VLLM_ROCM_QUICK_REDUCE_QUANTIZATION"] == "NONE" and rl_env["AITER_FLYDSL_STAGE2_FP8"] == "0"
    # AITER_BF16_FP8_MOE_BOUND=0 crashed rocm_aiter_grouped_topk (hipErrorLaunchOutOfResources)
    # under 64 concurrent top-128 requests on MI355X: never set by either ROCm preset
    assert "AITER_BF16_FP8_MOE_BOUND" not in rl_env and "AITER_BF16_FP8_MOE_BOUND" not in env
    assert all("AITER_BF16_FP8_MOE_BOUND" not in p["env"] for p in K31_PRESETS.values())
    # extra_args come last (they win in vLLM's argparse)
    late, _ = _engine(tmp_path, "rocm", config=_config(tmp_path, extra_args=("--gpu-memory-utilization", "0.85")),
                      k31_preset="rl-precision")
    late_argv = late.argv(config.base_dir)
    assert late_argv[-2:] == ["--gpu-memory-utilization", "0.85"]
    with pytest.raises(SamplerEngineError, match="rocm preset"):
        _engine(tmp_path, "cuda", k31_preset="amd-article")
    with pytest.raises(SamplerEngineError, match="k31_preset must be one of"):
        _engine(tmp_path, "rocm", k31_preset="fast")
    assert {p["platform"] for p in K31_PRESETS.values()} == {"cuda", "rocm"}
    assert set(ROCM_ENV_PRESET) <= set(engine.env())
    few, _ = _engine(tmp_path, "rocm", run=lambda argv: json.dumps([{"gpu": 0}]))
    with pytest.raises(SamplerEngineError, match="lists 1 GPU"):
        few.start(None, 0)


def test_cuda_k31_preset_is_the_vllm_recipe_baseline(tmp_path):
    engine, _ = _engine(tmp_path, k31_preset="rl-precision-cuda")
    argv = engine.argv(engine.config.base_dir)
    assert argv[argv.index("--load-format") + 1] == "fastsafetensors" and "--language-model-only" in argv
    assert "--decode-context-parallel-size" not in argv and "--mamba-ssm-cache-dtype" in argv


def test_merge_overrides_merges_nested_and_refuses_a_conflict():
    assert merge_overrides({"a": {"b": 1}}, {"a": {"c": 2}}, {"d": 3}) == {"a": {"b": 1, "c": 2}, "d": 3}
    with pytest.raises(SamplerEngineError, match="'b'"):
        merge_overrides({"a": {"b": 1}}, {"a": {"b": 2}})


# ------------------------------------------------------------------ R3
def test_decode_routed_experts_drops_the_dense_rows_of_the_model_layer_axis():
    payload = _routing(6)
    out = decode_routed_experts(_npy(payload), prompt_tokens=4, completion_tokens=3, moe_layers=3, dense_layers=1)
    assert out.dtype == np.uint16 and out.shape == (6, 3, TOPK)
    assert np.array_equal(out, payload[:, 1:])
    # a server that already sends L_moe layers, uint8 ids (<= 256 experts) and int32 are read as is
    small = _routing(6, layers=3, dense=0, dtype=np.uint8) % 200
    assert decode_routed_experts(_npy(small), prompt_tokens=4, completion_tokens=3, moe_layers=3,
                                 dense_layers=1).shape == (6, 3, TOPK)
    assert decode_routed_experts(_npy(payload.astype(np.int32)), prompt_tokens=4, completion_tokens=3,
                                 moe_layers=3, dense_layers=1).dtype == np.uint16
    assert decode_routed_experts(None, prompt_tokens=4, completion_tokens=3, moe_layers=3) is None


@pytest.mark.parametrize("payload, kw, match", [
    (_routing(5), {}, "5 rows for 6 positions"),
    (_routing(6, layers=5), {}, "5 layers"),
    (np.zeros((6, LAYERS, TOPK), dtype=np.uint16), {}, "all 0"),
    (_routing(6, dense=0), {}, "dense layer"),
    (_routing(6).astype(np.float32), {}, "integer"),
])
def test_decode_routed_experts_refuses_a_wrong_payload(payload, kw, match):
    with pytest.raises(SamplerEngineError, match=match):
        decode_routed_experts(_npy(payload), prompt_tokens=4, completion_tokens=3, moe_layers=3, dense_layers=1, **kw)


def test_decode_routed_experts_refuses_raw_bytes():
    raw = base64.b64encode(_routing(6).tobytes()).decode("ascii")  # the SGLang wire, not .npy
    with pytest.raises(SamplerEngineError, match=r"\.npy"):
        decode_routed_experts(raw, prompt_tokens=4, completion_tokens=3, moe_layers=3, dense_layers=1)


def _completions(routing_rows=None, *, k=2):
    def handler(body):
        prompts = body["prompt"]
        choices = []
        for i in range(len(prompts) * body["n"]):
            prompt_ids = [1, 2, 3, 4]
            choice = {"index": i, "text": "ab", "finish_reason": "stop",
                      "logprobs": {"tokens": ["token_id:7", "token_id:8", "token_id:9"],
                                   "token_logprobs": [-0.1, -0.2, -0.3],
                                   "top_logprobs": [{"token_id:7": -0.1, "token_id:5": -2.0, "token_id:6": -3.0}] * 3}}
            if body.get("return_token_ids"):
                choice["prompt_token_ids"] = prompt_ids
            if routing_rows is not None:
                choice["routed_experts"] = _npy(_routing(routing_rows, seed=i))
            choices.append(choice)
        return {"choices": choices}
    return handler


def test_completions_carry_r3_top_k_and_the_prompt_ids(tmp_path):
    engine, http = _engine(tmp_path, return_routed_experts=True, top_logprobs_num=2,
                           pin_version=sampler_pins.sampler_vllm_version("cuda"),
                           routes={("POST", "/v1/completions"): _completions(6),
                                   ("GET", "/version"): {"version": sampler_pins.VLLM_SAMPLER_CUDA_VERSION}})
    report = engine.start(None, 0)
    assert report.detail["server_info"]["vllm_pin_match"] is True
    (group,) = engine.generate(["hi"], SamplingParams(max_tokens=3, n=2))
    body = [b for m, p, b in http.calls if p == "/v1/completions"][-1]
    assert body["return_token_ids"] and body["return_tokens_as_token_ids"] and body["logprobs"] == 2
    first = group[0]
    assert first.token_ids == (7, 8, 9) and first.prompt_token_ids == (1, 2, 3, 4)
    assert first.routed_experts.shape == (6, LAYERS - DENSE, TOPK) and first.routed_experts.dtype == np.uint16
    assert first.top_logprobs_ids.shape == (3, 2) and list(first.top_logprobs_ids[0]) == [7, 5]
    assert not np.array_equal(group[0].routed_experts, group[1].routed_experts)
    # token-id prompts take their own ids for the R3 rows
    (tokens,) = engine.generate_tokens([[1, 2, 3, 4]], SamplingParams(max_tokens=3))
    assert tokens[0].routed_experts.shape[0] == 6


def test_r3_on_but_missing_in_the_response_is_an_error(tmp_path):
    engine, _ = _engine(tmp_path, return_routed_experts=True,
                        routes={("POST", "/v1/completions"): _completions(None)})
    engine.start(None, 0)
    with pytest.raises(SamplerEngineError, match="no routed_experts"):
        engine.generate(["hi"], SamplingParams(max_tokens=3))


def test_version_mismatch_is_recorded_not_refused(tmp_path):
    engine, _ = _engine(tmp_path, "rocm", pin_version=sampler_pins.sampler_vllm_version("rocm"),
                        routes={("GET", "/version"): {"version": "0.33.0"}})
    info = engine.start(None, 0).detail["server_info"]
    assert info["vllm_pin_match"] is False and info["vllm_pin_version"] == sampler_pins.VLLM_SAMPLER_ROCM_VERSION
    unpinned, _ = _engine(tmp_path, "rocm", routes={("GET", "/version"): {"version": "0.33.0"}})
    assert unpinned.start(None, 0).detail["server_info"] == {"version": "0.33.0"}


# ------------------------------------------------------------------ reload
def test_in_place_reload_pauses_reloads_from_disk_and_resumes(tmp_path):
    def rpc(body):
        assert body["method"] == "reload_weights"
        return {"results": [None] * 8}

    routes = {("POST", "/pause?mode=wait"): {"status": "paused"}, ("POST", "/collective_rpc"): rpc,
              ("POST", "/resume"): {"status": "resumed"}}
    engine, http = _engine(tmp_path, config=_config(tmp_path, in_place_reload=True), routes=routes)
    assert engine.env()[DEV_MODE_ENV] == "1"
    engine.start(None, 0)
    release = tmp_path / "r1"
    release.mkdir()
    (release / "config.json").write_text((engine.config.base_dir / "config.json").read_text())
    report = engine.load(release, 1)
    assert report.reload_mode == "in_place" and engine.served_version() == 1
    posts = [(p, b) for m, p, b in http.calls if m == "POST"]
    assert [p for p, _ in posts] == ["/pause?mode=wait", "/collective_rpc", "/resume"]
    assert posts[1][1]["kwargs"] == {"weights_path": str(release)}


def test_in_place_reload_resumes_after_a_failed_rpc(tmp_path):
    routes = {("POST", "/pause?mode=wait"): {"status": "paused"}, ("POST", "/resume"): {"status": "resumed"}}
    engine, http = _engine(tmp_path, config=_config(tmp_path, in_place_reload=True), routes=routes)
    engine.start(None, 0)
    release = tmp_path / "r1"
    release.mkdir()
    (release / "config.json").write_text((engine.config.base_dir / "config.json").read_text())
    with pytest.raises(SamplerEngineError, match="collective_rpc"):
        engine.load(release, 1)
    assert [p for m, p, b in http.calls if m == "POST"][-1] == "/resume"


def test_restart_reload_stays_the_default_upstream(tmp_path):
    engine, _ = _engine(tmp_path)
    assert DEV_MODE_ENV not in engine.env()
    engine.start(None, 0)
    release = tmp_path / "r1"
    release.mkdir()
    (release / "config.json").write_text((engine.config.base_dir / "config.json").read_text())
    assert engine.load(release, 1).reload_mode == "restart"


# ------------------------------------------------------------------ spec
def test_spec_on_rocm_builds_the_upstream_adapter_with_the_k31_fields(tmp_path):
    base = tmp_path / "b"
    base.mkdir()
    server = {"base_dir": str(base), "card_model": "MI355X", "device_family": "rocm", "engine_version": "x",
              "precision": "mxfp4", "quant_scheme": "MXFP4_EXPERTS_RTN", "topology": "tp8", "max_model_len": 1024}
    section = {"adapter": "vllm-server", "server": server, "platform": "rocm",
               "options": {"tp": 8, "k31_preset": "rl-precision"},
               "rollout_fields": {"routed_experts": True, "top_logprobs_num": 128, "fp32_lm_head": True}}
    engine = from_spec.build(section)
    assert engine.options.platform == "rocm" and engine.upstream
    assert engine.options.return_routed_experts and engine.options.fp32_lm_head and engine.top_logprobs_num == 128
    with pytest.raises(ValueError, match="platform"):
        from_spec.build({**section, "options": {"tp": 8, "platform": "cuda"}})
    # tier 1 stays refused on upstream vLLM (no plan maps onto its ignore list)
    spec = {"engine": {"adapter": "vllm-server", "server": {**server, "quant_scheme": "bf16", "precision": "bf16"},
                       "options": {"platform": "rocm"}}, "rollout_quant": {"tier": 1}}
    with pytest.raises(QuantTierSpecError, match="ignore list is not mapped"):
        apply_rollout_quant(spec)


def test_mi355x_vllm_example_spec_builds_the_k31_engine(tmp_path):
    from pathlib import Path

    from yeto.rl.engine import engine_sampler_entry as entry

    from test_rl_k31_e2e_cpu import EXAMPLES, _filled

    nc = _filled(json.loads((EXAMPLES / "sampler-spec.nc-mi355x-vllm.json").read_text()), tmp_path)
    nc["peers"] = ["10.0.0.11:8431"]
    Path(nc["engine"]["server"]["base_dir"]).mkdir(parents=True)
    config = entry.island_config(nc)
    assert config.sampling.n == 8
    engine = entry.build_engine(nc)
    assert engine.options.platform == "rocm" and engine.options.k31_preset == "amd-article"
    assert returns_routed_experts(engine) and engine.top_logprobs_num == 128 and engine.fp32_lm_head
    argv = engine.argv(engine.config.base_dir)
    assert argv[0] == "/opt/runtime-venv/bin/python" and "--enable-return-routed-experts" in argv
    assert nc["engine"]["server"]["engine_version"].endswith(sampler_pins.VLLM_SAMPLER_ROCM_VERSION)
    assert engine.options.pin_version == sampler_pins.sampler_vllm_version("rocm")
    # R3 on: the preset's DCP8 is dropped (vLLM refuses R3 with DCP > 1)
    assert "--decode-context-parallel-size" not in argv
    assert engine.cp_overrides == [{"flag": "--decode-context-parallel-size", "size": 8, "axis": "DCP",
                                    "source": "k31_preset amd-article", "now": 1}]


# ------------------------------------------------------------------ R3 and context parallelism
def _cp_sizes(argv):
    return [(a, argv[i + 1]) for i, a in enumerate(argv)
            if a in ("--decode-context-parallel-size", "-dcp", "--prefill-context-parallel-size", "-pcp")]


def test_drop_cp_sizes_takes_both_spellings_and_keeps_size_one():
    args = ["--gpu-memory-utilization", "0.9", "-dcp", "8", "--prefill-context-parallel-size=2",
            "--decode-context-parallel-size", "1", "--max-num-seqs", "4"]
    kept, dropped = drop_cp_sizes(args)
    assert kept == ["--gpu-memory-utilization", "0.9", "--decode-context-parallel-size", "1", "--max-num-seqs", "4"]
    assert dropped == [{"flag": "-dcp", "size": 8, "axis": "DCP"},
                       {"flag": "--prefill-context-parallel-size", "size": 2, "axis": "PCP"}]
    with pytest.raises(SamplerEngineError, match="is not a size"):
        drop_cp_sizes(["-dcp", "eight"])


@pytest.mark.parametrize("preset", ["amd-article", "rl-precision"])
def test_r3_drops_the_rocm_preset_dcp_and_records_it(tmp_path, capsys, preset):
    extra = ("--decode-context-parallel-size", "8", "-pcp", "2", "--gpu-memory-utilization", "0.85")
    config = _config(tmp_path, extra_args=extra)
    engine, _ = _engine(tmp_path, "rocm", config=config, k31_preset=preset, return_routed_experts=True,
                        top_logprobs_num=128, routes={("GET", "/version"): {"version": "v"}})
    argv = engine.argv(config.base_dir)
    assert _cp_sizes(argv) == [] and "--enable-return-routed-experts" in argv
    assert argv[argv.index("--max-logprobs") + 1] == "128"
    assert argv[-2:] == ["--gpu-memory-utilization", "0.85"]  # the rest of extra_args stays, last
    assert "--kv-cache-dtype" in argv and "--enable-prefix-caching" in argv  # the preset's layout otherwise
    source = f"k31_preset {preset}"
    assert engine.cp_overrides == [
        {"flag": "--decode-context-parallel-size", "size": 8, "axis": "DCP", "source": source, "now": 1},
        {"flag": "--decode-context-parallel-size", "size": 8, "axis": "DCP", "source": "extra_args", "now": 1},
        {"flag": "-pcp", "size": 2, "axis": "PCP", "source": "extra_args", "now": 1}]
    report = engine.start(None, 0)
    assert report.detail["r3_cp_overrides"] == engine.cp_overrides  # the load record (manifest)
    out = capsys.readouterr().out
    assert "[vllm-server] R3 on" in out and "context parallelism (DCP/PCP > 1" in out and source in out


def test_without_r3_the_dcp_layout_is_left_alone(tmp_path):
    config = _config(tmp_path, extra_args=("-pcp", "2"))
    engine, _ = _engine(tmp_path, "rocm", config=config, k31_preset="amd-article",
                        routes={("GET", "/version"): {"version": "v"}})
    argv = engine.argv(config.base_dir)
    assert _cp_sizes(argv) == [("--decode-context-parallel-size", "8"), ("-pcp", "2")]
    assert engine.cp_overrides == [] and engine.r3_layout_note() is None
    assert "r3_cp_overrides" not in engine.start(None, 0).detail


def test_r3_without_context_parallelism_has_no_override(tmp_path):
    engine, _ = _engine(tmp_path, k31_preset="rl-precision-cuda", return_routed_experts=True,
                        config=_config(tmp_path, extra_args=("--decode-context-parallel-size", "1")))
    argv = engine.argv(engine.config.base_dir)
    assert _cp_sizes(argv) == [("--decode-context-parallel-size", "1")]
    assert engine.cp_overrides == [] and engine.r3_layout_note() is None


@pytest.mark.parametrize("extra_args, extra_env, match", [
    (("--pipeline-parallel-size", "2"), {}, "pipeline parallelism"),
    (("-pp=2",), {}, "pipeline parallelism"),
    (("--no-enable-prefix-caching",), {}, "requires prefix caching"),
    ((), {"VLLM_USE_V2_MODEL_RUNNER": "0"}, "Model Runner V2"),
])
def test_r3_refuses_the_other_layouts_vllm_rejects_before_a_start(tmp_path, extra_args, extra_env, match):
    config = _config(tmp_path, extra_args=extra_args, extra_env=extra_env)
    with pytest.raises(SamplerEngineError, match=match):
        _engine(tmp_path, "rocm", config=config, return_routed_experts=True)
    _engine(tmp_path, "rocm", config=_config(tmp_path, extra_args=extra_args, extra_env=extra_env))  # no R3: as is


def test_spec_with_routed_experts_drops_dcp_from_extra_args(tmp_path):
    base = tmp_path / "b"
    base.mkdir()
    server = {"base_dir": str(base), "card_model": "MI355X", "device_family": "rocm", "engine_version": "x",
              "precision": "mxfp4", "quant_scheme": "MXFP4_EXPERTS_RTN", "topology": "tp8", "max_model_len": 1024,
              "extra_args": ["--decode-context-parallel-size", "8"]}
    section = {"adapter": "vllm-server", "server": server, "platform": "rocm",
               "options": {"tp": 8, "k31_preset": "amd-article"}}
    with_r3 = from_spec.build({**section, "rollout_fields": {"routed_experts": True}})
    assert _cp_sizes(with_r3.argv(base)) == [] and len(with_r3.cp_overrides) == 2
    without = from_spec.build({**section, "rollout_fields": {"routed_experts": False}})
    assert _cp_sizes(without.argv(base)) == [("--decode-context-parallel-size", "8")] * 2


def test_a_release_is_served_with_the_tokenizer_of_the_base_checkpoint(tmp_path):
    """Found by the Qwen3-0.6B trainer + vLLM sampler rehearsal: the trainer's HF export
    writes tokenizer_config.json in its transformers-5 form (``extra_special_tokens`` a
    list); the restart of a vLLM 0.11 sampler (transformers 4.x) on that release died at
    start ten times in a row ('list' object has no attribute 'keys'). A release is now
    served with the base checkpoint's tokenizer, on both vLLM lines."""
    for platform in ("cuda", None):  # the upstream line and the legacy one
        engine, _ = _engine(tmp_path, platform)
        base = engine.config.base_dir
        assert "--tokenizer" not in engine.argv(base)  # the cold start reads the base itself
        argv = engine.argv(tmp_path / "r1-v1")
        assert argv[argv.index("--tokenizer") + 1] == str(base), platform
    pinned = _config(tmp_path, card_model="B300", device_family="nvidia")
    pinned = dataclasses.replace(pinned, extra_args=("--tokenizer", "/own/tokenizer"))
    engine, _ = _engine(tmp_path, config=pinned)
    argv = engine.argv(tmp_path / "r1-v1")
    assert argv.count("--tokenizer") == 1 and argv[argv.index("--tokenizer") + 1] == "/own/tokenizer"
