"""CPU tests for the gated sampler snapshot mode (rl-trainer-cold-start-cost L8.2)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import config as mc
from yeto.rl.adapters.miles import sampler_snapshot as ss
from yeto.rl.adapters.miles.pins import SGLANG_NEXT_COMMIT
from yeto.rl.engine.algorithm import AlgorithmSpec

from test_rl_miles_adapter_config import make_config, sub

DISTRIBUTED_PASS = ss.CompatRecord(
    sglang_commit=SGLANG_NEXT_COMMIT, update_path=ss.UPDATE_PATH_DISTRIBUTED,
    result="pass", check="G-L8.2", evidence="test",
)


def snapshot_config(*, colocated: bool = False):
    cfg = make_config(colocated=colocated)
    return sub(cfg, "serving", offload_train=False, sampler_snapshot=True)


# The G-L8 passes were recorded on SGLang 2fa880182 (S24). The pin moved to 10afe11a7
# and 754605964 (S26 E6: the FP8 activation-cast fix in Qwen2/Qwen3, the compressed-tensors
# FP8 in-place reload); until G-L8 is re-run on it
# the snapshot mode is refused, as for any commit without a record.
VERIFIED_SGLANG_COMMIT = "2fa880182eefdb31e64d8ea70317be27f9f32121"


def _records_at_the_pin():
    return tuple(dataclasses.replace(r, sglang_commit=SGLANG_NEXT_COMMIT) for r in ss.load_compat_records())


def test_shipped_records_hold_the_l8_1_tensor_pass():
    record = ss.require_recorded_pass(ss.UPDATE_PATH_TENSOR, sglang_commit=VERIFIED_SGLANG_COMMIT)
    assert record.check == "G-L8.1"


def test_the_pinned_commit_waits_for_its_own_record():
    if SGLANG_NEXT_COMMIT == VERIFIED_SGLANG_COMMIT:
        pytest.skip("the pin is the verified commit")
    with pytest.raises(ss.SamplerSnapshotRefused, match="no recorded pass"):
        ss.require_recorded_pass(ss.UPDATE_PATH_TENSOR)


def test_refused_without_recorded_pass_names_the_missing_check():
    with pytest.raises(ss.SamplerSnapshotRefused) as exc:
        ss.require_recorded_pass(ss.UPDATE_PATH_DISTRIBUTED, records=())
    message = str(exc.value)
    assert "no recorded pass" in message
    assert "snapshot compatibility check" in message
    assert SGLANG_NEXT_COMMIT[:9] in message and "'distributed'" in message


def test_record_for_another_commit_does_not_count():
    other = ss.CompatRecord("0" * 40, ss.UPDATE_PATH_DISTRIBUTED, "pass", "G-L8.2", "")
    with pytest.raises(ss.SamplerSnapshotRefused, match="no recorded pass"):
        ss.require_recorded_pass(ss.UPDATE_PATH_DISTRIBUTED, records=(other,))


def test_recorded_fail_is_refused():
    failed = ss.CompatRecord(SGLANG_NEXT_COMMIT, ss.UPDATE_PATH_DISTRIBUTED, "fail", "G-L8.2", "")
    with pytest.raises(ss.SamplerSnapshotRefused, match="recorded 'fail'"):
        ss.require_recorded_pass(ss.UPDATE_PATH_DISTRIBUTED, records=(failed,))


def test_default_off_adds_no_sglang_config():
    argv = mc.translate_run_config(
        sub(make_config(colocated=False), "serving", offload_train=False), AlgorithmSpec()
    ).argv
    assert "--sglang-config" not in argv


def test_translation_refuses_without_record(monkeypatch):
    monkeypatch.setattr(ss, "load_compat_records", lambda text=None: ())
    with pytest.raises(mc.MilesConfigError, match="no recorded pass"):
        mc.translate_run_config(snapshot_config(), AlgorithmSpec())


def test_translation_with_record_sets_sampler_overrides_only(monkeypatch):
    monkeypatch.setattr(ss, "load_compat_records", lambda text=None: (DISTRIBUTED_PASS,))
    launch = mc.translate_run_config(snapshot_config(), AlgorithmSpec())
    argv = list(launch.argv)
    value = argv[argv.index("--sglang-config") + 1]
    decoded = ss.decode_sglang_config_arg(value)
    (model,) = decoded["sglang"]
    (group,) = model["server_groups"]
    assert model["name"] == "default"
    assert group == {"worker_type": "regular", "num_gpus": launch.placement.rollout_gpus,
                     "overrides": {"enable_memory_saver": True, "enable_weights_cpu_backup": True}}
    assert "--offload-rollout" not in argv  # the trainer side is unchanged


def test_translation_refuses_colocated(monkeypatch):
    monkeypatch.setattr(ss, "load_compat_records", lambda text=None: (DISTRIBUTED_PASS,))
    with pytest.raises(mc.MilesConfigError, match="dedicated sampler engines"):
        mc.translate_run_config(snapshot_config(colocated=True), AlgorithmSpec())


def test_translation_refuses_extra_sglang_config(monkeypatch):
    monkeypatch.setattr(ss, "load_compat_records", lambda text=None: (DISTRIBUTED_PASS,))
    request = SimpleNamespace(kind="fixed-partition", placement_map_arg=None, rollout_gpus=2)
    with pytest.raises(ss.SamplerSnapshotRefused, match="drop --sglang-config"):
        ss.snapshot_argv(request, extra_argv=("--sglang-config", "x.yaml"))


def test_cli_flag_is_off_by_default_and_parses():
    from yeto import cli

    base = ["launch", "--training-mode", "rl", "--gpu", "modal:2xh100", "--model", "x", "--data", "y"]
    assert cli.build_parser().parse_args(base).rl_sampler_snapshot is False
    assert cli.build_parser().parse_args([*base, "--rl-sampler-snapshot"]).rl_sampler_snapshot is True


def test_shipped_records_hold_the_l8_2_distributed_pass(monkeypatch):
    record = ss.require_recorded_pass(ss.UPDATE_PATH_DISTRIBUTED, sglang_commit=VERIFIED_SGLANG_COMMIT)
    assert record.check == "G-L8.2"
    at_the_pin = _records_at_the_pin()
    monkeypatch.setattr(ss, "load_compat_records", lambda text=None: at_the_pin)
    argv = mc.translate_run_config(snapshot_config(), AlgorithmSpec()).argv
    assert "--sglang-config" in argv
