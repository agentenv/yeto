"""SGLang DP attention: --attn-dp-size (K3.1 pin) vs --dp N --enable-dp-attention,
resolved by a --help probe and by what GET /get_server_info reports."""
from __future__ import annotations

import pytest

from yeto.rl.adapters.sampler_engines.sglang_server import (
    DP_FLAG_ATTN_DP_SIZE,
    DP_FLAG_LEGACY,
    SGLangOptions,
    SGLangServerEngine,
    dp_attention_argv,
    dp_flag_from_help,
    dp_flag_from_server_info,
    summarize_server_info,
)
from yeto.rl.engine.sampler_engine import SamplerEngineError

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

NEW_HELP = "usage: launch_server [-h] --model-path ... --attn-dp-size ATTN_DP_SIZE --tp-size ..."
OLD_HELP = "usage: launch_server [-h] --model-path ... --dp-size DP_SIZE --enable-dp-attention ..."


def test_dp_flag_forms_and_help_probe():
    assert dp_attention_argv(DP_FLAG_ATTN_DP_SIZE, 8) == ["--attn-dp-size", "8", "--moe-a2a-backend", "deepep"]
    assert dp_attention_argv(DP_FLAG_LEGACY, 8) == ["--dp", "8", "--enable-dp-attention", "--moe-a2a-backend", "deepep"]
    assert dp_flag_from_help(NEW_HELP) == DP_FLAG_ATTN_DP_SIZE and dp_flag_from_help(OLD_HELP) == DP_FLAG_LEGACY
    with pytest.raises(SamplerEngineError, match="neither"):
        dp_flag_from_help("usage: something else")
    with pytest.raises(SamplerEngineError, match="not resolved"):
        dp_attention_argv("auto", 8)


def test_server_info_summary_reads_both_layouts():
    new = {"version": "0.5.9", "attn_dp_size": 8, "tp_size": 8, "enable_return_routed_experts": True,
           "internal_states": [{}], "launch_command": "..."}
    assert summarize_server_info(new) == {"version": "0.5.9", "attn_dp_size": 8, "tp_size": 8,
                                          "enable_return_routed_experts": True}
    old = {"version": "0.5.3", "server_args": {"dp_size": 8, "enable_dp_attention": True, "tp_size": 8}}
    assert summarize_server_info(old) == {"version": "0.5.3", "dp_size": 8, "enable_dp_attention": True, "tp_size": 8}
    assert dp_flag_from_server_info(summarize_server_info(new)) == DP_FLAG_ATTN_DP_SIZE
    assert dp_flag_from_server_info(summarize_server_info(old)) == DP_FLAG_LEGACY
    assert dp_flag_from_server_info({"version": "x"}) is None and summarize_server_info(None) == {}


def _engine(tmp_path, options, http, probe, spawned):
    return SGLangServerEngine(_config(tmp_path, topology="dp8"), options, help_probe=probe,
                              spawn=lambda a, e: spawned.append(a) or _Process(), http=http, sleep=lambda s: None)


def test_auto_probes_help_once_and_learns_from_server_info(tmp_path):
    probes = []
    def probe(argv):
        probes.append(argv)
        return NEW_HELP
    http = _Http({("GET", "/health"): {}, ("GET", "/get_server_info"): {"version": "0.5.9", "attn_dp_size": 8}})
    spawned = []
    engine = _engine(tmp_path, SGLangOptions(), http, probe, spawned)
    report = engine.start(None, 0)
    assert probes == [["python", "-m", "sglang.launch_server", "--help"]]
    assert "--attn-dp-size" in spawned[0] and "--enable-dp-attention" not in spawned[0]
    assert report.detail["server_info"] == {"version": "0.5.9", "attn_dp_size": 8}
    engine.start(None, 1)  # the second start asks no probe
    assert len(probes) == 1 and "--attn-dp-size" in spawned[1]


def test_auto_takes_the_new_form_when_the_probe_fails_and_switches_on_a_legacy_server(tmp_path):
    def probe(argv):
        raise OSError("no sglang here")
    http = _Http({("GET", "/health"): {}, ("GET", "/get_server_info"): {"server_args": {"enable_dp_attention": True,
                                                                                           "dp_size": 8}}})
    spawned = []
    engine = _engine(tmp_path, SGLangOptions(), http, probe, spawned)
    engine.start(None, 0)
    assert "--attn-dp-size" in spawned[0] and "dp_flag_probe_error" in engine.server_info
    assert engine.dp_flag == DP_FLAG_LEGACY  # learned from the running server
    engine.start(None, 1)
    assert "--enable-dp-attention" in spawned[1]


def test_explicit_flag_skips_the_probe_and_tp_topology_needs_neither(tmp_path):
    def probe(argv):
        raise AssertionError("no probe for an explicit flag")
    spawned = []
    engine = _engine(tmp_path, SGLangOptions(dp_attention_flag=DP_FLAG_LEGACY),
                     _Http({("GET", "/health"): {}}), probe, spawned)
    engine.start(None, 0)
    assert "--enable-dp-attention" in spawned[0] and engine.server_info == {}
    tp = SGLangServerEngine(_config(tmp_path), help_probe=probe, spawn=lambda a, e: spawned.append(a) or _Process(),
                            http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    tp.start(None, 0)
    assert not any(a.startswith("--attn-dp") or a == "--enable-dp-attention" for a in spawned[1])
    with pytest.raises(SamplerEngineError, match="dp_attention_flag"):
        SGLangServerEngine(_config(tmp_path), SGLangOptions(dp_attention_flag="dp"))


def test_server_that_resolved_a_wanted_feature_off_fails_the_start(tmp_path):
    http = _Http({("GET", "/health"): {}, ("GET", "/get_server_info"): {"attn_dp_size": 8, "version": "x",
                                                                           "enable_return_routed_experts": False}})
    spawned = []
    engine = _engine(tmp_path, SGLangOptions(return_routed_experts=True), http, lambda a: NEW_HELP, spawned)
    with pytest.raises(SamplerEngineError, match="enable_return_routed_experts"):
        engine.start(None, 0)
    http.routes[("GET", "/get_server_info")] = {"attn_dp_size": 4, "version": "x"}
    with pytest.raises(SamplerEngineError, match="attn_dp_size is 4"):
        _engine(tmp_path, SGLangOptions(), http, lambda a: NEW_HELP, spawned).start(None, 0)
