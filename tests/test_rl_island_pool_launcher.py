"""Island pool launcher wiring: syncer membership flags and island flags. No Ray."""

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


HMAC_CHECK = ': "${YETO_ISLAND_HMAC_KEY:?island HMAC key secret missing}" && '


POOL = ("--rl-sync-quorum-theta", "0.75", "--rl-sync-max-base-lag", "2")


def test_fixed_roster_syncer_command_is_byte_identical(head_launcher):
    """secret-handling-hardening: the only change to the fixed-roster RL line is the
    island HMAC key check in front (the key itself is a secret env); SFT only
    gains the explicit --allow-unauthenticated-islands."""
    args = _args()
    assert not launcher.island_pool(args)
    for n in (1, 2, 4):
        assert launcher.syncer_command(args, n) == head_launcher.syncer_command(args, n).replace(
            "mkdir -p ~/yeto-output && ", "mkdir -p ~/yeto-output && " + HMAC_CHECK, 1)
        # P0-D8: one command per role; the HMAC check sits once in front.
        commands = launcher.role_syncer_commands(args, n)
        assert list(commands) == ["actor"]
        assert launcher.syncer_command(args, n).endswith(HMAC_CHECK + commands["actor"])
    assert launcher.syncer_command(_sft(), 2) == (
        head_launcher.syncer_command(_sft(), 2) + " --allow-unauthenticated-islands")


def test_fixed_roster_learner_flags_unchanged():
    args = _args()
    flags = launcher._ports_infra_flags(args)[1]
    assert "--rl-island-pool" not in flags and "--rl-sync-island-lease-s" not in flags


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")


def test_pool_and_fixed_roster_generate_a_key_when_none_is_set(monkeypatch):
    monkeypatch.delenv("YETO_ISLAND_HMAC_KEY")
    for extra in (POOL, ()):
        assert launcher.syncer_command(_args(extra), 2).startswith(
            "mkdir -p ~/yeto-output && " + HMAC_CHECK)


def test_pool_syncer_flags_and_defaults():
    cmd = launcher.syncer_command(_args(POOL), 2)
    assert "k3y" not in cmd and "export YETO_ISLAND_HMAC_KEY" not in cmd  # 0.20: secret, not argv
    assert cmd.startswith('mkdir -p ~/yeto-output && : "${YETO_ISLAND_HMAC_KEY:?')
    # only the given options reach the syncer; lease and syncer epoch always do
    assert " --quorum-theta 0.75 --island-lease-s 150.0 --syncer-epoch 0" in cmd
    for flag in ("--carry-gamma", "--soft-deadline-s", "--q-min", "--final-grace-s",
                 "--island-scheduling-mode", "--max-carry-lag"):
        assert flag not in cmd
    cmd = launcher.syncer_command(_args((
        "--rl-sync-quorum-theta", "0.6", "--rl-sync-carry-gamma", "0.25", "--rl-sync-soft-deadline-s", "300",
        "--rl-sync-q-min", "2", "--rl-sync-island-lease-s", "60", "--rl-sync-max-base-lag", "1",
        "--rl-syncer-epoch", "4")), 3)
    assert (" --quorum-theta 0.6 --q-min 2 --carry-gamma 0.25 --soft-deadline-s 300"
            " --island-lease-s 60.0 --syncer-epoch 4 --final-grace-s 300") in cmd
    assert "--island-lease-s 150.0" in launcher.role_syncer_commands(_args(POOL), 2)["actor"]
    assert launcher._ports_infra_flags(_args(POOL))[1].endswith(
        " --rl-island-pool --rl-syncer-epoch 0 --rl-sync-island-lease-s 150.0")


def test_invalid_combinations_refused():
    with pytest.raises(ValueError, match="--rl-syncer-epoch needs island pool options"):
        launcher.syncer_command(_args(("--rl-syncer-epoch", "1")), 2)
    with pytest.raises(ValueError, match=r"quorum-theta must be in \(0, 1\]"):
        launcher.syncer_command(_args(("--rl-sync-quorum-theta", "1.5")), 2)
    with pytest.raises(ValueError, match="only apply to --training-mode rl"):
        launcher.syncer_command(SimpleNamespace(**vars(_sft()), rl_sync_island_lease_s=30.0), 2)
    for removed in ("--rl-island-scheduling", "--rl-max-carry-lag"):
        with pytest.raises(SystemExit):
            _args((removed, "1"))


def test_pool_needs_rounds_that_may_merge_without_an_island():
    from test_rl_launcher import _prepare_rl_args

    with pytest.raises(ValueError, match="merge without some island"):
        _prepare_rl_args(_prov(_args(("--gpu", "modal:8xh100,modal:8xh100", "--rl-engine", "ports",
                                      "--rl-sync-island-lease-s", "30"))))


def test_learner_passes_the_pool_switch_to_the_island_config():
    from yeto.rl.adapters.miles import island_entry as learner
    assert "rl_island_pool" in Path(learner.__file__).read_text()
    from yeto.rl.adapters.miles import elastic_wiring
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
    args = _args(POOL)
    assert launcher.resolve_island_syncer_epoch(
        args, lambda: next(reads, None) or real(), sleep=sleep, clock=clock) == 6
    assert args.rl_syncer_epoch == 6
    assert " --rl-syncer-epoch 6" in launcher._ports_infra_flags(args)[1]


def test_syncer_epoch_unreadable_falls_back_to_zero(tmp_path, capsys):
    import json

    (tmp_path / "status.json").write_text(json.dumps({"schema": "other", "syncer_epoch": 3}))
    sleep, clock = _no_sleep_clock()
    args = _args(POOL)
    reader = launcher.syncer_status_reader(local_tape=str(tmp_path / "yeto-tape.jsonl"))
    assert launcher.resolve_island_syncer_epoch(args, reader, timeout_s=10, sleep=sleep, clock=clock) == 0
    assert args.rl_syncer_epoch is None and "WARN" in capsys.readouterr().out
    assert " --rl-syncer-epoch 0" in launcher._ports_infra_flags(args)[1]


def test_syncer_epoch_explicit_or_fixed_roster_never_reads():
    def boom():
        raise AssertionError("status.json must not be read")
    explicit = _args((*POOL, "--rl-syncer-epoch", "4"))
    assert launcher.resolve_island_syncer_epoch(explicit, boom) == 4
    fixed = _args()
    assert launcher.resolve_island_syncer_epoch(fixed, boom) == 0
    assert "--rl-island-pool" not in launcher._ports_infra_flags(fixed)[1]


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
    monkeypatch.setattr("yeto.launch.syncer.build_syncer_binary", lambda: tmp_path / "yeto-syncer")
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    task = launcher.make_syncer_task(_args(POOL), 2)
    assert task.secrets == {"YETO_ISLAND_HMAC_KEY": "k3y"}
    assert "k3y" not in task.run and "k3y" not in json.dumps(task.envs or {})
    fixed = launcher.make_syncer_task(_args(), 2)
    assert fixed.secrets == {"YETO_ISLAND_HMAC_KEY": "k3y"}  # secret-handling-hardening
    assert "k3y" not in fixed.run


def test_hmac_key_in_modal_cfg_envs_and_island_secrets(monkeypatch, tmp_path):
    from test_rl_launcher import _prepare_rl_args
    from yeto.gpu_spec import parse_gpu_spec

    _fake_sky(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    args = _args(("--gpu", "modal:8xh100", "--rl-engine", "ports", *POOL))
    _prepare_rl_args(_prov(args))
    (spec,) = parse_gpu_spec(args.gpu)
    task = launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    assert task.secrets["YETO_ISLAND_HMAC_KEY"] == "k3y"
    assert "k3y" not in task.run and "k3y" not in (task.setup or "")
    cfg = launcher.build_modal_island_config(args, spec, 0, task, "1.2.3.4:5000")
    assert cfg.envs["YETO_ISLAND_HMAC_KEY"] == "k3y" and "k3y" not in cfg.run_script
    fixed = _args(("--gpu", "modal:8xh100", "--rl-engine", "ports"))
    _prepare_rl_args(_prov(fixed))
    lt = launcher.make_miles_island_task(fixed, spec, 0, 1, "127.0.0.1:29400")
    # secret-handling-hardening: fixed-roster islands carry the key too.
    assert lt.secrets["YETO_ISLAND_HMAC_KEY"] == "k3y"
    assert launcher.build_modal_island_config(fixed, spec, 0, lt, "1.2.3.4:5000").envs[
        "YETO_ISLAND_HMAC_KEY"] == "k3y"


def test_head_job_ships_hmac_secret():
    src = Path(launcher.__file__).with_name("cli.py").read_text()
    assert "secrets.update(launcher.island_hmac_secret(args))" in src
    assert launcher.island_hmac_secret(_args()) == {"YETO_ISLAND_HMAC_KEY": "k3y"}
    assert launcher.island_hmac_secret(_sft()) == {}
    assert launcher.island_hmac_secret(_args(POOL)) == {
        "YETO_ISLAND_HMAC_KEY": "k3y"}
