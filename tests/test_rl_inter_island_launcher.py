"""--rl-island-scheduling launcher wiring (rl-inter-island-scheduling 0.13). No Ray."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_rl_launcher import _args
from yeto import launcher

ROOT = Path(__file__).resolve().parents[1]
BASE = "fd37129e"  # branch point before this change


@pytest.fixture(scope="module")
def head_launcher(tmp_path_factory):
    """yeto/launcher.py as it was before this change, loaded as yeto._launcher_base."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:yeto/launcher.py"],
                         check=True, capture_output=True, text=True).stdout
    path = tmp_path_factory.mktemp("base") / "launcher_base.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("yeto._launcher_base", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sft():
    return SimpleNamespace(training_mode="sft", quorum=2, grace_ms=0, grace_gamma=1.0,
                           grace_tau=1.0, pipeline=2, sync_interval_steps=24.0,
                           delta_correction="none", total_steps=10, outer_lr=0.7,
                           outer_momentum=0.9)


def test_legacy_syncer_command_is_byte_identical(head_launcher):
    for extra in ((), ("--rl-island-scheduling", "legacy")):
        args = _args(extra)
        assert args.rl_island_scheduling == "legacy"
        for n in (1, 2, 4):
            assert launcher.syncer_command(args, n) == head_launcher.syncer_command(args, n)
            assert (launcher.critic_syncer_command(args, n)
                    == head_launcher.critic_syncer_command(args, n))
    assert launcher.syncer_command(_sft(), 2) == head_launcher.syncer_command(_sft(), 2)


def test_legacy_learner_flags_unchanged():
    args = _args()
    assert "--rl-island-scheduling" not in launcher._ports_infra_flags(args)[1]


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")


def test_elastic_needs_hmac_key(monkeypatch):
    monkeypatch.delenv("YETO_ISLAND_HMAC_KEY")
    with pytest.raises(ValueError, match="YETO_ISLAND_HMAC_KEY"):
        launcher.syncer_command(_args(("--rl-island-scheduling", "elastic")), 2)
    assert "HMAC" not in launcher.syncer_command(_args(), 2)  # legacy needs no key


def test_elastic_syncer_flags_and_defaults():
    cmd = launcher.syncer_command(_args(("--rl-island-scheduling", "elastic")), 2)
    assert "k3y" not in cmd and "export YETO_ISLAND_HMAC_KEY" not in cmd  # 0.20: secret, not argv
    assert cmd.startswith('mkdir -p ~/yeto-output && : "${YETO_ISLAND_HMAC_KEY:?')
    cmd = launcher.syncer_command(_args(("--rl-island-scheduling", "elastic")), 2)
    assert (" --island-scheduling-mode elastic --quorum-theta 0.75 --carry-gamma 0.5"
            " --soft-deadline-s 900 --q-min 1 --max-carry-lag 2 --syncer-epoch 0") in cmd
    cmd = launcher.syncer_command(_args((
        "--rl-island-scheduling", "elastic", "--rl-quorum-theta", "0.6", "--rl-carry-gamma",
        "0.25", "--rl-soft-deadline-s", "300", "--rl-q-min", "2", "--rl-max-carry-lag", "1",
        "--rl-syncer-epoch", "4")), 3)
    assert (" --island-scheduling-mode elastic --quorum-theta 0.6 --carry-gamma 0.25"
            " --soft-deadline-s 300 --q-min 2 --max-carry-lag 1 --syncer-epoch 4") in cmd
    assert "--island-scheduling-mode elastic" in launcher.critic_syncer_command(
        _args(("--rl-island-scheduling", "elastic")), 2)
    assert " --rl-island-scheduling elastic --rl-syncer-epoch 0" in launcher._ports_infra_flags(
        _args(("--rl-island-scheduling", "elastic")))[1]


def test_invalid_combinations_refused():
    with pytest.raises(ValueError, match="need --rl-island-scheduling elastic"):
        launcher.syncer_command(_args(("--rl-quorum-theta", "0.5")), 2)
    with pytest.raises(ValueError, match="need --rl-island-scheduling elastic"):
        launcher.syncer_command(_args(("--rl-syncer-epoch", "1")), 2)
    with pytest.raises(ValueError):
        launcher.syncer_command(_args(("--rl-island-scheduling", "elastic",
                                       "--rl-quorum-theta", "1.5")), 2)
    with pytest.raises(ValueError, match="only applies to --training-mode rl"):
        launcher.syncer_command(SimpleNamespace(**vars(_sft()), rl_island_scheduling="elastic"), 2)
    with pytest.raises(SystemExit):
        _args(("--rl-island-scheduling", "auto"))


def test_learner_passes_mode_to_elastic_config():
    from yeto.rl import learner
    assert "island_scheduling" in Path(learner.__file__).read_text()
    from yeto.rl.engine.miles_adapter import elastic_wiring
    import inspect
    assert inspect.signature(elastic_wiring.build_elastic).parameters[
        "island_scheduling"].default == "legacy"


# ----------------------------------------------------------------- 0.17 syncer_epoch from status.json
def _no_sleep_clock():
    t = [0.0]

    def sleep(s):
        t[0] += s
    return sleep, (lambda: t[0])


def test_syncer_epoch_read_from_status_json(tmp_path):
    import json

    (tmp_path / "status.json").write_text(json.dumps(
        {"schema": "yeto.syncer.elastic-status/v1", "syncer_epoch": 6, "islands": {}}))
    reads = iter([None, None])  # the syncer is still starting for two polls
    real = launcher.syncer_status_reader(local_tape=str(tmp_path / "yeto-tape.jsonl"))
    sleep, clock = _no_sleep_clock()
    args = _args(("--rl-island-scheduling", "elastic"))
    assert launcher.resolve_island_syncer_epoch(
        args, lambda: next(reads, None) or real(), sleep=sleep, clock=clock) == 6
    assert args.rl_syncer_epoch == 6
    assert " --rl-syncer-epoch 6" in launcher._ports_infra_flags(args)[1]


def test_syncer_epoch_unreadable_falls_back_to_zero(tmp_path, capsys):
    import json

    (tmp_path / "status.json").write_text(json.dumps({"schema": "other", "syncer_epoch": 3}))
    sleep, clock = _no_sleep_clock()
    args = _args(("--rl-island-scheduling", "elastic"))
    reader = launcher.syncer_status_reader(local_tape=str(tmp_path / "yeto-tape.jsonl"))
    assert launcher.resolve_island_syncer_epoch(args, reader, timeout_s=10, sleep=sleep, clock=clock) == 0
    assert args.rl_syncer_epoch is None and "WARN" in capsys.readouterr().out
    assert " --rl-syncer-epoch 0" in launcher._ports_infra_flags(args)[1]


def test_syncer_epoch_explicit_or_legacy_never_reads():
    def boom():
        raise AssertionError("status.json must not be read")
    explicit = _args(("--rl-island-scheduling", "elastic", "--rl-syncer-epoch", "4"))
    assert launcher.resolve_island_syncer_epoch(explicit, boom) == 4
    legacy = _args()
    assert launcher.resolve_island_syncer_epoch(legacy, boom) == 0
    assert "--rl-island-scheduling" not in launcher._ports_infra_flags(legacy)[1]


# ----------------------------------------------------------------- 0.20 HMAC key as a secret
def _prov(args):
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    return args


def _fake_sky(monkeypatch):
    import types

    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))


def test_hmac_key_is_secret_on_syncer_task(monkeypatch, tmp_path):
    _fake_sky(monkeypatch)
    monkeypatch.setattr(launcher, "build_syncer_binary", lambda: tmp_path / "yeto-syncer")
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    task = launcher.make_syncer_task(_args(("--rl-island-scheduling", "elastic")), 2)
    assert task.secrets == {"YETO_ISLAND_HMAC_KEY": "k3y"}
    assert "k3y" not in task.run and "k3y" not in json.dumps(task.envs or {})
    legacy = launcher.make_syncer_task(_args(), 2)
    assert legacy.secrets is None and "HMAC" not in legacy.run


def test_hmac_key_in_modal_cfg_envs_and_island_secrets(monkeypatch, tmp_path):
    from test_rl_launcher import _prepare_rl_args
    from yeto.gpu_spec import parse_gpu_spec

    _fake_sky(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    args = _args(("--gpu", "modal:8xh100", "--rl-engine", "ports", "--rl-island-scheduling", "elastic"))
    _prepare_rl_args(_prov(args))
    (spec,) = parse_gpu_spec(args.gpu)
    task = launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    assert task.secrets["YETO_ISLAND_HMAC_KEY"] == "k3y"
    assert "k3y" not in task.run and "k3y" not in (task.setup or "")
    cfg = launcher.build_modal_island_config(args, spec, 0, task, "1.2.3.4:5000")
    assert cfg.envs["YETO_ISLAND_HMAC_KEY"] == "k3y" and "k3y" not in cfg.run_script
    legacy = _args(("--gpu", "modal:8xh100", "--rl-engine", "ports"))
    _prepare_rl_args(_prov(legacy))
    lt = launcher.make_miles_island_task(legacy, spec, 0, 1, "127.0.0.1:29400")
    assert "YETO_ISLAND_HMAC_KEY" not in (getattr(lt, "secrets", None) or {})
    assert "YETO_ISLAND_HMAC_KEY" not in launcher.build_modal_island_config(legacy, spec, 0, lt, "1.2.3.4:5000").envs


def test_head_job_ships_hmac_secret():
    src = Path(launcher.__file__).with_name("cli.py").read_text()
    assert "secrets.update(launcher.island_hmac_secret(args))" in src
    assert launcher.island_hmac_secret(_args()) == {}
    assert launcher.island_hmac_secret(_args(("--rl-island-scheduling", "elastic"))) == {
        "YETO_ISLAND_HMAC_KEY": "k3y"}
