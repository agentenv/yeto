"""The platform rule of the SGLang in-place reload: ROCm (aiter shuffle) and
pre-Blackwell CUDA (bf16 upcast of MXFP4 experts) crash on
``update_weights_from_disk``, so they reload by restart whatever the spec's
``in_place_reload`` says; Blackwell keeps the in-place / file-level paths."""
from __future__ import annotations

import json

import pytest

from yeto.rl.adapters.sampler_engines.sglang_server import (
    IN_PLACE_RELOAD_OFF_ROCM,
    RELOAD_POLICY_IN_PLACE,
    RELOAD_POLICY_RESTART,
    RESTART_RELOAD_READ_BYTES_PER_S,
    RESTART_RELOAD_START_OVERHEAD_S,
    SGLangOptions,
    SGLangServerEngine,
    cuda_compute_capability,
    restart_reload_estimate_s,
)
from yeto.rl.engine.sampler_engine import RELOAD_IN_PLACE, RELOAD_RESTART, SamplerEngineError

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

EIGHT_AMD = json.dumps([{"gpu": i} for i in range(8)])


def _release(tmp_path, name="r1", size=1_000_000):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["KimiK31ForConditionalGeneration"],
                                               "num_hidden_layers": 4, "first_k_dense_replace": 1,
                                               "n_routed_experts": 896}))
    (d / "model-00001-of-00001.safetensors").write_bytes(b"\0" * size)
    return d


def _engine(tmp_path, run, spawned, http=None, **options):
    config = _config(tmp_path, in_place_reload=True, **options.pop("config", {}))
    return SGLangServerEngine(config, SGLangOptions(**options), run=run,
                              spawn=lambda a, e: spawned.append((a, e)) or _Process(),
                              http=http or _Http({("GET", "/health"): {}}), sleep=lambda s: None)


def test_cuda_compute_capability_reads_the_lowest_card_and_tolerates_a_missing_tool():
    assert cuda_compute_capability(lambda argv: "9.0\n9.0\n") == (9, 0)
    assert cuda_compute_capability(lambda argv: "10.3\n10.0\n") == (10, 0)
    assert cuda_compute_capability(lambda argv: "garbage") is None

    def missing(argv):
        raise OSError("nvidia-smi: not found")
    assert cuda_compute_capability(missing) is None


def test_restart_estimate_is_start_overhead_plus_disk_read():
    assert restart_reload_estimate_s(0) == RESTART_RELOAD_START_OVERHEAD_S
    tb = 1.5e12
    assert restart_reload_estimate_s(int(tb)) == RESTART_RELOAD_START_OVERHEAD_S + tb / RESTART_RELOAD_READ_BYTES_PER_S


def test_rocm_reloads_by_restart_even_when_the_spec_asks_for_in_place(tmp_path):
    spawned = []
    engine = _engine(tmp_path, lambda argv: EIGHT_AMD, spawned, platform="rocm", expert_file_update="auto")
    assert engine.in_place_reload_off == IN_PLACE_RELOAD_OFF_ROCM
    assert engine.expert_update_off == IN_PLACE_RELOAD_OFF_ROCM  # file-level updates need the in-place path
    assert not engine.reload_in_place_effective()
    report = engine.start(None, 0)
    policy = report.detail["reload_policy"]
    assert policy["mode"] == RELOAD_POLICY_RESTART and policy["configured_in_place"] is True
    assert policy["off_reason"] == IN_PLACE_RELOAD_OFF_ROCM and policy["platform"] == "rocm"
    assert policy["restart_estimate_s"] >= RESTART_RELOAD_START_OVERHEAD_S
    release = _release(tmp_path, size=2_000_000)
    report = engine.load(release, 1)
    assert report.reload_mode == RELOAD_RESTART and len(spawned) == 2  # stop -> start on the release
    assert spawned[1][0][spawned[1][0].index("--model-path") + 1] == str(release)
    assert report.detail["reload_policy"]["release_bytes"] == 2_000_000
    assert report.detail["reload_policy"]["restart_estimate_s"] == round(restart_reload_estimate_s(2_000_000), 1)
    with pytest.raises(SamplerEngineError, match="in-place reload is off"):
        engine.reload_in_place(release, 2)


def test_rocm_live_update_is_refused_because_it_needs_the_in_place_path(tmp_path):
    with pytest.raises(SamplerEngineError, match="live_update needs in-place reload"):
        _engine(tmp_path, lambda argv: EIGHT_AMD, [], platform="rocm", live_update=True)


def test_hopper_reloads_by_restart_and_blackwell_in_place(tmp_path):
    def nvidia(cap):
        def run(argv):
            assert argv[0] == "nvidia-smi" and "--query-gpu=compute_cap" in argv
            return f"{cap}\n{cap}\n"
        return run

    spawned = []
    hopper = _engine(tmp_path, nvidia("9.0"), spawned, config={"card_model": "H200", "device_family": "nvidia"})
    assert hopper.in_place_reload_off is None  # decided at preflight, not construction
    report = hopper.start(None, 0)
    assert hopper.compute_capability == (9, 0)
    assert report.detail["reload_policy"]["mode"] == RELOAD_POLICY_RESTART
    assert "sm90" in report.detail["reload_policy"]["off_reason"]
    report = hopper.load(_release(tmp_path, "h"), 1)
    assert report.reload_mode == RELOAD_RESTART and len(spawned) == 2

    spawned = []
    disk_calls = []
    http = _Http({("GET", "/health"): {},
                  ("POST", "/update_weights_from_disk"): lambda body: disk_calls.append(body) or {"success": True}})
    blackwell = _engine(tmp_path, nvidia("10.0"), spawned, http=http,
                        config={"card_model": "B200", "device_family": "nvidia"})
    report = blackwell.start(None, 0)
    assert blackwell.compute_capability == (10, 0) and blackwell.reload_in_place_effective()
    assert report.detail["reload_policy"]["mode"] == RELOAD_POLICY_IN_PLACE
    assert "restart_estimate_s" not in report.detail["reload_policy"]
    report = blackwell.load(_release(tmp_path, "b"), 1)
    assert report.reload_mode == RELOAD_IN_PLACE and len(spawned) == 1 and len(disk_calls) == 1


def test_the_rule_is_about_mxfp4_and_an_unreadable_capability_keeps_the_spec(tmp_path):
    def missing(argv):
        raise OSError("nvidia-smi: not found")

    spawned = []
    disk = _Http({("GET", "/health"): {}, ("POST", "/update_weights_from_disk"): {"success": True}})
    unknown = _engine(tmp_path, missing, spawned, http=disk, config={"card_model": "H100", "device_family": "nvidia"})
    report = unknown.start(None, 0)
    assert unknown.compute_capability is None and unknown.reload_in_place_effective()
    assert report.detail["reload_policy"]["compute_capability"] is None
    assert unknown.load(_release(tmp_path, "u"), 1).reload_mode == RELOAD_IN_PLACE

    spawned = []
    probed = []
    bf16 = _engine(tmp_path, lambda argv: probed.append(argv) or "9.0\n", spawned, http=disk,
                   config={"card_model": "H100", "device_family": "nvidia", "precision": "bf16",
                           "quant_scheme": "bf16"})
    bf16.start(None, 0)
    assert probed == [] and bf16.reload_in_place_effective()  # bf16 experts: no probe, no rule

    spawned = []
    rocm_bf16 = _engine(tmp_path, lambda argv: EIGHT_AMD, spawned, platform="rocm",
                        config={"precision": "bf16", "quant_scheme": "bf16"})
    assert rocm_bf16.in_place_reload_off is None

    spawned = []
    restart_spec = SGLangServerEngine(_config(tmp_path, in_place_reload=False), SGLangOptions(platform="rocm"),
                                      run=lambda argv: EIGHT_AMD,
                                      spawn=lambda a, e: spawned.append((a, e)) or _Process(),
                                      http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    report = restart_spec.start(None, 0)
    assert restart_spec.in_place_reload_off is None  # nothing to turn off
    assert report.detail["reload_policy"]["mode"] == RELOAD_POLICY_RESTART
    assert report.detail["reload_policy"]["configured_in_place"] is False
