"""CPU tests for the gated sampler snapshot mode (rl-trainer-cold-start-cost L8.2)."""

from __future__ import annotations

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


def test_shipped_records_hold_the_l8_1_tensor_pass():
    record = ss.require_recorded_pass(ss.UPDATE_PATH_TENSOR)
    assert record.sglang_commit == SGLANG_NEXT_COMMIT
    assert record.check == "G-L8.1"


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


def test_launch_gate_refuses_before_provisioning():
    with pytest.raises(ss.SamplerSnapshotRefused, match="fixed-partition"):
        ss.check_launch("colocated", records=(DISTRIBUTED_PASS,))
    with pytest.raises(ss.SamplerSnapshotRefused, match="no recorded pass"):
        ss.check_launch("fixed-partition", records=())
    ss.check_launch("fixed-partition", records=(DISTRIBUTED_PASS,))


def test_cli_flag_is_off_by_default_and_parses():
    from yeto import cli

    base = ["launch", "--training-mode", "rl", "--gpu", "modal:2xh100", "--model", "x", "--data", "y"]
    assert cli.build_parser().parse_args(base).rl_sampler_snapshot is False
    assert cli.build_parser().parse_args([*base, "--rl-sampler-snapshot"]).rl_sampler_snapshot is True
