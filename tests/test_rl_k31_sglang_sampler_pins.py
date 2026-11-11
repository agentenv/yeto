"""The sampler-side SGLang pin (upstream main image) is independent of the Miles
training pin, and the SGLang server adapter records the build it talks to."""
from __future__ import annotations

import re

import pytest

from yeto.rl import sampler_pins
from yeto.rl.adapters.miles import pins
from yeto.rl.adapters.sampler_engines.sglang_server import (
    BUILD_ENV_KEYS,
    CUDA_MXFP4_DECODE_MOE_ENV,
    SERVER_INFO_KEYS,
    SERVER_INFO_PATHS,
    SGLangOptions,
    SGLangServerEngine,
    summarize_server_info,
)

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"docker:(docker\.io|ghcr\.io)/[a-z0-9._/-]+@sha256:[0-9a-f]{64}")


def test_sampler_pins_are_full_shas_and_digests_and_decoupled_from_miles():
    assert SHA.fullmatch(sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT)
    assert SHA.fullmatch(sampler_pins.SGLANG_SAMPLER_ROCM_COMMIT)
    assert SHA.fullmatch(sampler_pins.SGLANG_SAMPLER_YETO_COMMIT)
    for image in (sampler_pins.SGLANG_SAMPLER_BASE_IMAGE_CUDA, sampler_pins.SGLANG_SAMPLER_IMAGE_CUDA, sampler_pins.SGLANG_SAMPLER_IMAGE_ROCM):
        assert DIGEST.fullmatch(image), image
    assert sampler_pins.SGLANG_SAMPLER_UPSTREAM_REPOSITORY == "https://github.com/sgl-project/sglang"
    # the training-side pin is a different commit on a different branch; moving one must not move the other
    assert sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT != pins.SGLANG_K31_COMMIT
    assert pins.SGLANG_NEXT_COMMIT == pins.SGLANG_K31_COMMIT
    assert sampler_pins.SGLANG_SAMPLER_IMAGE_MANIFEST == pins.MILES_NEXT_IMAGE_MANIFEST == "/opt/yeto/image-manifest.json"
    assert sampler_pins.sampler_sglang_commit("cuda") == sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT
    assert sampler_pins.sampler_sglang_commit("rocm") == sampler_pins.SGLANG_SAMPLER_ROCM_COMMIT
    assert sampler_pins.sampler_sglang_commit("ppu") is None and sampler_pins.sampler_sglang_commit("npu") is None
    assert sampler_pins.SGLANG_SAMPLER_ENGINE_VERSIONS["cuda"].endswith(sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT[:8])
    assert str(sampler_pins.SGLANG_SAMPLER_ROCM_PR) in sampler_pins.SGLANG_SAMPLER_ENGINE_VERSIONS["rocm"]


def test_sampler_pins_are_exported_from_yeto_rl():
    import yeto.rl as rl

    assert rl.SGLANG_SAMPLER_UPSTREAM_COMMIT == sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT
    assert rl.SGLANG_SAMPLER_IMAGE_CUDA == sampler_pins.SGLANG_SAMPLER_IMAGE_CUDA
    assert rl.sampler_sglang_commit("cuda") == sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT


def test_server_info_is_read_from_the_new_path_first_and_keeps_the_k31_knobs():
    assert SERVER_INFO_PATHS == ("/server_info", "/get_server_info")
    for key in ("attention_backend", "dsa_decode_backend", "dcp_size", "enable_prefill_cp", "moe_runner_backend"):
        assert key in SERVER_INFO_KEYS
    info = {"version": "0.5.23", "tp_size": 2, "dsa_decode_backend": "trtllm", "dcp_size": 1,
            "enable_return_routed_experts": True, "internal_states": [{}], "launch_command": "..."}
    assert summarize_server_info(info) == {"version": "0.5.23", "tp_size": 2, "dsa_decode_backend": "trtllm",
                                           "dcp_size": 1, "enable_return_routed_experts": True}


def _engine(tmp_path, monkeypatch, routes, *, env=(), **options):
    for key in BUILD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env:
        monkeypatch.setenv(key, value)
    http = _Http({("GET", "/health"): {}, **routes})
    engine = SGLangServerEngine(_config(tmp_path, device_family="nvidia", card_model="H100", extra_env={}),
                                SGLangOptions(tp=8, platform="cuda", **options), http=http,
                                spawn=lambda *a, **k: _Process(), sleep=lambda s: None)
    return engine, http


def _ready(engine, http):
    """Start the fake server (``after_ready`` needs a live process) and return the
    GET paths ``after_ready`` asked, without the health poll."""
    engine.start(None, 0)
    before = len(http.calls)
    detail = engine.after_ready()
    return detail, [c[1] for c in http.calls[before:] if c[0] == "GET" and c[1] != "/health"]


def test_after_ready_prefers_server_info_and_records_the_build_against_the_pin(tmp_path, monkeypatch):
    engine, http = _engine(tmp_path, monkeypatch,
                           {("GET", "/server_info"): {"version": "0.5.23", "tp_size": 8, "enable_return_routed_experts": True}},
                           env=(("SGLANG_BUILD_COMMIT", sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT),
                                ("SGLANG_IMAGE_TAG", "lmsysorg/sglang:nightly-dev-20261109-842b4738")))
    detail, gets = _ready(engine, http)
    assert gets == ["/server_info"]
    info = detail["server_info"]
    assert info["version"] == "0.5.23" and info["sglang_build_commit"] == sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT
    assert info["sglang_image_tag"].startswith("lmsysorg/sglang:")
    assert info["sglang_pin_match"] is True and info["sglang_pin_commit"] == sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT


def test_after_ready_falls_back_to_get_server_info_and_flags_a_newer_build(tmp_path, monkeypatch):
    engine, http = _engine(tmp_path, monkeypatch,
                           {("GET", "/get_server_info"): {"version": "0.5.9", "server_args": {"tp_size": 8}}},
                           env=(("SGLANG_BUILD_COMMIT", "f" * 40),))
    detail, gets = _ready(engine, http)
    assert gets == ["/server_info", "/get_server_info"]
    assert detail["server_info"]["version"] == "0.5.9"
    assert detail["server_info"]["sglang_pin_match"] is False  # recorded, not refused


def test_a_server_without_build_env_records_nothing_about_the_build(tmp_path, monkeypatch):
    engine, http = _engine(tmp_path, monkeypatch, {("GET", "/server_info"): {"version": "0.5.23", "tp_size": 8}})
    info = _ready(engine, http)[0]["server_info"]
    assert "sglang_build_commit" not in info and "sglang_pin_match" not in info


@pytest.mark.parametrize("asked, expect", [(None, None), (True, "1"), (False, "0")])
def test_cuda_mxfp4_decode_moe_option_sets_the_upstream_env(tmp_path, monkeypatch, asked, expect):
    engine, _ = _engine(tmp_path, monkeypatch, {}, cuda_mxfp4_decode_moe=asked)
    assert engine.env().get(CUDA_MXFP4_DECODE_MOE_ENV) == expect
    assert CUDA_MXFP4_DECODE_MOE_ENV == "SGLANG_USE_CUDA_MXFP4_DECODE_MOE"


def test_extra_env_wins_over_the_mxfp4_option_and_rocm_never_sets_it(tmp_path, monkeypatch):
    http = _Http({})
    config = _config(tmp_path, device_family="nvidia", card_model="H100", extra_env={CUDA_MXFP4_DECODE_MOE_ENV: "0"})
    engine = SGLangServerEngine(config, SGLangOptions(tp=8, platform="cuda", cuda_mxfp4_decode_moe=True),
                                http=http, spawn=lambda *a, **k: _Process())
    assert engine.env()[CUDA_MXFP4_DECODE_MOE_ENV] == "0"
    rocm = SGLangServerEngine(_config(tmp_path), SGLangOptions(tp=8, platform="rocm", cuda_mxfp4_decode_moe=True),
                              http=http, spawn=lambda *a, **k: _Process())
    assert CUDA_MXFP4_DECODE_MOE_ENV not in rocm.env()
