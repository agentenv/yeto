"""ROCm (MI355X) support of the engine sampler: amd-smi / rocm-smi telemetry and
preflight parsers, the SGLang ROCm environment preset and card check, and the
spec's ``platform`` plumbing."""
from __future__ import annotations

import json

import pytest

from yeto.hw import card_telemetry as ct
from yeto.hw import card_tools
from yeto.rl.adapters.sampler_engines import from_spec
from yeto.rl.adapters.sampler_engines.sglang_server import (
    ROCM_ENV_PRESET,
    SGLangOptions,
    SGLangServerEngine,
    rocm_card_count,
)
from yeto.rl.engine.engine_sampler_entry import SamplerSpecError, build_engine
from yeto.rl.engine.sampler_engine import SamplerEngineError

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

AMD_SMI_METRIC = json.dumps([
    {"gpu": 0, "usage": {"gpu_activity": {"value": 37, "unit": "%"}},
     "power": {"socket_power": {"value": 812, "unit": "W"}},
     "temperature": {"edge": {"value": 41, "unit": "C"}, "hotspot": {"value": 58, "unit": "C"}},
     "mem_usage": {"total_vram": {"value": 294912, "unit": "MB"}, "used_vram": {"value": 1024, "unit": "MB"}}},
    {"gpu": 1, "usage": {"gpu_activity": {"value": "N/A", "unit": "%"}}, "power": {},
     "temperature": {"edge": {"value": 40, "unit": "C"}},
     "mem_usage": {"total_vram": {"value": 288, "unit": "GB"}, "used_vram": {"value": "N/A", "unit": "MB"}}},
])
ROCM_SMI = json.dumps({"card0": {"GPU use (%)": "12", "VRAM Total Memory (B)": str(288 * 2**30),
                                 "VRAM Total Used Memory (B)": str(2**30), "Average Graphics Package Power (W)": "650.0",
                                 "Temperature (Sensor junction) (C)": "61.0", "Unique ID": "0xabc"},
                       "card1": {"GPU use (%)": "0", "VRAM Total Memory (B)": str(288 * 2**30),
                                 "VRAM Total Used Memory (B)": "0", "Temperature (Sensor edge) (C)": "40.0"},
                       "system": {"Driver version": "6.12"}})


def test_amd_smi_metric_and_rocm_smi_json_become_card_samples():
    cards = ct.parse_amd_smi_metric(AMD_SMI_METRIC)
    assert [c["index"] for c in cards] == [0, 1] and set(cards[0]) == set(ct.CARD_KEYS)
    assert cards[0]["util_pct"] == 37 and cards[0]["power_w"] == 812 and cards[0]["temp_c"] == 58
    assert cards[0]["mem_total_mb"] == 294912 and cards[0]["mem_used_mb"] == 1024
    assert cards[1]["util_pct"] is None and cards[1]["mem_total_mb"] == 288 * 1024 and cards[1]["mem_used_mb"] is None
    cards = ct.parse_rocm_smi_json(ROCM_SMI)
    assert [c["index"] for c in cards] == [0, 1] and cards[0]["uuid"] == "0xabc"
    assert cards[0]["util_pct"] == 12 and cards[0]["mem_total_mb"] == 288 * 1024 and cards[0]["mem_used_mb"] == 1024
    assert cards[0]["power_w"] == 650 and cards[0]["temp_c"] == 61 and cards[1]["temp_c"] == 40
    with pytest.raises(ct.CardTelemetryError, match="no GPU entry"):
        ct.parse_amd_smi_metric("[]")
    with pytest.raises(ct.CardTelemetryError, match="no JSON"):
        ct.parse_rocm_smi_json("rocm-smi: command not found")


def test_rocm_collector_prefers_amd_smi_and_falls_back_to_rocm_smi():
    def run(argv):
        if argv[0] == "amd-smi":
            raise ct.CardTelemetryError("amd-smi: FileNotFoundError")
        return ROCM_SMI
    collector = ct.collector_for("rocm", run=run)
    collector.open()
    assert collector.source == "rocm-smi" and len(collector.sample()) == 2
    both = ct.collector_for("rocm", run=lambda argv: AMD_SMI_METRIC if argv[0] == "amd-smi" else ROCM_SMI)
    both.open()
    assert both.source == "amd-smi" and both.sample()[0]["power_w"] == 812
    def none(argv):
        raise ct.CardTelemetryError(f"{argv[0]}: missing")
    with pytest.raises(ct.CardTelemetryError, match="amd-smi: missing; rocm-smi: missing"):
        ct.collector_for("rocm", run=none).open()


def test_preflight_parsers_read_amd_smi_static_and_rocm_smi_product():
    assert card_tools.preflight_commands("rocm") == (("amd-smi", "static", "--json"),)
    static = json.dumps([{"gpu": 0, "asic": {"market_name": "AMD Instinct MI355X"},
                          "vram": {"size": {"value": 294912, "unit": "MB"}}},
                         {"gpu": 1, "asic": {"market_name": "AMD Instinct MI355X"}, "vram": {"size": {"value": 288, "unit": "GB"}}}])
    records = card_tools.parse_card_tool("rocm", static)
    assert [(r.index, r.card, round(r.hbm_gib)) for r in records] == [(0, "MI355X", 288), (1, "MI355X", 288)]
    rocm = json.dumps({"card0": {"Card Series": "AMD Instinct MI300X OAM", "VRAM Total Memory (B)": str(192 * 2**30)}})
    records = card_tools.parse_card_tool("rocm", rocm)
    assert records[0].card == "MI300X" and round(records[0].hbm_gib) == 192
    with pytest.raises(card_tools.CardToolError, match="no JSON"):
        card_tools.parse_card_tool("rocm", "not json")
    with pytest.raises(card_tools.CardToolError, match="market_name"):
        card_tools.parse_card_tool("rocm", json.dumps([{"gpu": 0, "asic": {}}]))


def test_rocm_card_count_and_sglang_start_check(tmp_path):
    assert rocm_card_count(lambda argv: json.dumps([{"gpu": i} for i in range(8)])) == 8
    def rocm_only(argv):
        if argv[0] == "amd-smi":
            raise OSError("missing")
        return json.dumps({"card0": {}, "card1": {}})
    assert rocm_card_count(rocm_only) == 2
    def none(argv):
        raise OSError("missing")
    with pytest.raises(SamplerEngineError, match="no AMD card tool"):
        rocm_card_count(none)
    spawned = []
    engine = SGLangServerEngine(_config(tmp_path), SGLangOptions(platform="rocm"), run=rocm_only,
                                spawn=lambda a, e: spawned.append((a, e)) or _Process(),
                                http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    with pytest.raises(SamplerEngineError, match="lists 2 GPU"):
        engine.start(None, 0)
    assert spawned == []  # refused before any process


def test_rocm_platform_injects_the_env_preset_and_extra_env_wins(tmp_path):
    spawned = []
    eight = lambda argv: json.dumps([{"gpu": i} for i in range(8)])  # noqa: E731
    config = _config(tmp_path, extra_env={"SGLANG_ROCM_RUNTIME_RESERVE_GB": "12", "HIP_VISIBLE_DEVICES": "0-7"})
    engine = SGLangServerEngine(config, SGLangOptions(platform="rocm", rocm_aiter=True), run=eight,
                                spawn=lambda a, e: spawned.append((a, e)) or _Process(),
                                http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    report = engine.start(None, 0)
    env = spawned[0][1]
    assert env["HSA_NO_SCRATCH_RECLAIM"] == "1" and env["ROCM_QUICK_REDUCE_QUANTIZATION"] == "NONE"
    assert env["AITER_SITUV2_A8W4"] == "1" and env["SGLANG_USE_AITER"] == "1"
    assert env["SGLANG_ROCM_RUNTIME_RESERVE_GB"] == "12" and env["HIP_VISIBLE_DEVICES"] == "0-7"  # extra_env wins
    assert "MX_PATCH" not in env  # the PPU W4A16 patch is not a ROCm thing
    assert report.detail["env"]["HSA_NO_SCRATCH_RECLAIM"] == "1"
    # cuda (default) platform: none of the preset, MX_PATCH as before
    plain = SGLangServerEngine(_config(tmp_path), spawn=lambda a, e: spawned.append((a, e)) or _Process(),
                               http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    plain.start(None, 0)
    assert not set(ROCM_ENV_PRESET) & set(spawned[1][1]) and spawned[1][1]["MX_PATCH"] == "1"
    with pytest.raises(SamplerEngineError, match="platform"):
        SGLangServerEngine(_config(tmp_path), SGLangOptions(platform="tpu"))
    with pytest.raises(SamplerEngineError, match="rocm_aiter"):
        SGLangServerEngine(_config(tmp_path), SGLangOptions(rocm_aiter=True))


def test_spec_platform_reaches_sglang_and_vllm_options_and_vllm_refuses_ppu(tmp_path):
    base = tmp_path / "b"
    base.mkdir()
    server = {"base_dir": str(base), "card_model": "MI355X", "device_family": "rocm", "engine_version": "x",
              "precision": "mxfp4", "quant_scheme": "MXFP4_EXPERTS_RTN", "topology": "tp8", "max_model_len": 1024}
    spec = {"engine": {"factory": from_spec.FACTORY, "adapter": "sglang-server", "server": server, "options": {}},
            "platform": "rocm", "sampling": {"max_tokens": 4}}
    engine = build_engine(spec)
    assert engine.options.platform == "rocm"
    with pytest.raises(SamplerSpecError, match="platform must be one of"):
        build_engine({**spec, "platform": "tpu"})
    vllm = {**spec, "engine": {"factory": from_spec.FACTORY, "adapter": "vllm-server",
                               "server": {**server, "precision": "bf16", "quant_scheme": "bf16", "topology": "tp4"},
                               "options": {"tp": 4}}}
    assert build_engine(vllm).options.platform == "rocm"
    with pytest.raises(SamplerSpecError, match="vllm-server runs on cuda / rocm / npu"):
        build_engine({**vllm, "platform": "ppu"})
    fake = {"engine": {"factory": from_spec.FACTORY, "adapter": "fake", "kwargs": {}}, "platform": "rocm",
            "sampling": {"max_tokens": 4}}
    build_engine(fake)  # the fake engine ignores the platform
