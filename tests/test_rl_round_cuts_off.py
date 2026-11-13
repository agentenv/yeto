"""``--rl-round-cuts off``: no round cut wired, written or read; a relaunch continues
from the newest release (``yeto run up`` adds ``--rl-init-from-release latest``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from yeto import cli
from yeto.rl import up
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.cut_interval import check_round_cuts_off, round_cuts_off

BASE = ["launch", "--model", "gemma4", "--data", "org/data"]
NO_SYNC = ["--gpu", "modal:1xh100", "--training-mode", "rl", "--rl-single-island-no-sync",
           "--cluster-prefix", "run", "--rl-checkpoint-store", "s3://b/x"]
CUT_FLAGS = [["--rl-cut-every", "2"], ["--rl-cut-auto-mtbf-s", "21600"],
             ["--rl-cut-max-interval-s", "600"], ["--rl-cut-keep", "3"]]


def _args(extra):
    return cli.build_parser().parse_args(BASE + extra)


# ------------------------------------------------------------------ flag parsing / conflicts
def test_the_flag_defaults_to_auto_and_takes_off():
    assert _args(NO_SYNC).rl_round_cuts == "auto" and not round_cuts_off(_args(NO_SYNC))
    assert round_cuts_off(_args(NO_SYNC + ["--rl-round-cuts", "off"]))
    with pytest.raises(SystemExit):
        _args(NO_SYNC + ["--rl-round-cuts", "never"])


@pytest.mark.parametrize("flag", CUT_FLAGS + [["--rl-stop-after-rounds", "4"],
                                              ["--rl-resume-allow-config-change"]])
def test_off_refuses_the_cut_flags_before_the_launch(flag):
    from yeto import launcher

    args = _args(NO_SYNC + ["--rl-round-cuts", "off"] + flag)
    with pytest.raises(ValueError, match=f"--rl-round-cuts off: .*{flag[0]}"):
        launcher._check_resume_launch_flags(args)
    # the same flags with the default (auto) pass these checks
    if flag[0] not in ("--rl-cut-max-interval-s",):
        launcher._check_resume_launch_flags(_args(NO_SYNC + flag))


def test_off_refuses_a_convertible_node():
    with pytest.raises(ValueError, match="convertible node"):
        check_round_cuts_off(SimpleNamespace(rl_round_cuts="off", rl_sample_role="convertible"))


def test_off_passes_the_launch_checks_and_is_forwarded_to_the_island():
    from yeto import launcher

    args = _args(NO_SYNC + ["--rl-round-cuts", "off"])
    assert launcher._check_resume_launch_flags(args) is True  # the store stays the resume/buffer store
    _, flags = launcher._ports_infra_flags(args)
    assert " --rl-round-cuts off" in flags and "--rl-resume-store" in flags
    assert "--rl-cut-" not in flags and "--rl-island-cut-store" not in flags
    _, auto = launcher._ports_infra_flags(_args(NO_SYNC))
    assert "--rl-round-cuts" not in auto


# ------------------------------------------------------------------ island side (Miles)
def _learner(extra):
    from test_rl_engine_selection import _learner_argv

    from yeto.rl.adapters.miles import island_entry as learner

    argv = _learner_argv(("--rl-single-island-no-sync", "--rl-resume-store", "/s", *extra))
    i = argv.index("--syncer")
    del argv[i:i + 2]
    return learner, learner.parse_args(argv)


def test_the_island_builds_no_resume_wiring_with_off():
    learner, parsed = _learner(("--rl-round-cuts", "off"))
    learner._check_ports_infra_switches(parsed)
    miles_args, env = SimpleNamespace(), {}
    learner.apply_ports_infra_switches(parsed, miles_args, env)
    assert not hasattr(miles_args, "yeto_rl_resume") and miles_args.yeto_rl_round_cuts == "off"
    learner, parsed = _learner(())
    miles_args = SimpleNamespace()
    learner.apply_ports_infra_switches(parsed, miles_args, {})
    assert miles_args.yeto_rl_resume["store"] == "/s"  # auto: unchanged


@pytest.mark.parametrize("flag", CUT_FLAGS)
def test_the_island_refuses_the_cut_flags_with_off(flag, capsys):
    with pytest.raises(SystemExit):  # parse_args runs the island's switch checks
        _learner(("--rl-round-cuts", "off", *flag))
    assert "--rl-round-cuts off" in capsys.readouterr().err


class _Driver:
    def __init__(self):
        self.sync = LocalOnlySync(8)
        self.events = []
        self.rollout = SimpleNamespace(_in_executor=lambda fn, *a: ("executor", fn.__name__, *a))
        self.trainer = object()

    def export_local_resident(self, *, policy_version):
        return {"policy_version": policy_version}

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


def test_off_wires_no_cut_and_writes_nothing_to_the_store(capsys):
    from yeto.rl.adapters.miles import entry
    from yeto.rl.adapters.miles.buffer_rollout import READER_FUNCTION_PATHS

    driver = _Driver()
    store = mock.MagicMock(name="store")
    controller = SimpleNamespace(checkpoint_store=store, store=store, runtime_fingerprint="f")
    wiring = SimpleNamespace(controller=controller, ledger=None, every=1, allow_config_change=False)
    miles_args = SimpleNamespace(yeto_rl_round_cuts="off", global_batch_size=4,
                                 rollout_function_path=READER_FUNCTION_PATHS[0])
    with mock.patch("yeto.rl.adapters.miles.round_cut.wire_round_cuts") as wire:
        entry._wire_round_cuts(driver, elastic=wiring, miles_args=miles_args, algorithm=None,
                               base_model_revision="r")
        entry._wire_round_cuts_off(driver, miles_args=miles_args)
    wire.assert_not_called()
    assert driver.sync.round_cuts is None
    # the sync's start (where a cut would be restored): the store is never touched
    driver.sync.start(driver)
    assert store.mock_calls == []
    assert not any("round-cuts" in str(c) for c in store.mock_calls)
    (kind, fields), = driver.events
    assert kind == "rl_round_cuts_off" and fields["buffer_reader"] is True
    assert "round cuts are off" in capsys.readouterr().out
    assert callable(driver.buffer_start_without_cuts)


def test_off_without_the_flag_keeps_wiring_round_cuts():
    from yeto.rl.adapters.miles import entry

    driver = _Driver()
    entry._wire_round_cuts_off(driver, miles_args=SimpleNamespace())
    assert driver.events == [] and not hasattr(driver, "buffer_start_without_cuts")


def test_a_release_restart_reads_the_buffer_on_from_the_trained_position(monkeypatch):
    from yeto.rl.adapters.miles import sampler_island

    driver = _Driver()
    calls = []
    driver.buffer_start_without_cuts = lambda position: calls.append(position) or position
    resumed = SimpleNamespace(base_version=5, start_version=5, buffer_trained=640,
                              history=lambda: [], known_versions=lambda: [5])
    monkeypatch.setattr(sampler_island, "resumed_releases", lambda releases, limit: resumed)
    assert sampler_island._resume_single_trainer(driver, releases=None) == {"resumed_base": 5}
    assert calls == []  # the reader exists only once the sync starts
    driver.sync.start(driver)
    assert calls == [640] and driver.sync.resumed_at == 5


# ------------------------------------------------------------------ verl
def test_verl_off_gives_no_resume_plan_and_refuses_the_cut_flags():
    from yeto.rl.adapters.verl import round_cuts

    base = dict(rl_resume_store="/s", rl_island_cut_store=None, rl_cut_every=None, rl_cut_keep=None,
                rl_stop_after_rounds=None, rl_cut_auto_mtbf_s=None, rl_cut_max_interval_s=None,
                rl_resume_allow_config_change=False, rl_round_cuts="off")
    assert round_cuts.resume_config(SimpleNamespace(**base)) is None
    with pytest.raises(ValueError, match="--rl-round-cuts off"):
        round_cuts.resume_config(SimpleNamespace(**{**base, "rl_cut_every": 2}))


# ------------------------------------------------------------------ yeto run up restart
@pytest.fixture
def run_of(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up, "time", types.SimpleNamespace(
        time=time.time, sleep=lambda s: None, gmtime=time.gmtime, strftime=time.strftime))
    pool_cfg = tmp_path / "pool.json"
    pool_cfg.write_text(json.dumps({"shard_base": "s"}))

    def build(extra, *, restart, keys):
        cfg = {"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": str(pool_cfg)},
               "trainer": {"argv": ["--gpu", "modal:2x8xb300", *extra], "usd_per_h": 1},
               "budget": {}}
        run = up.Run(cfg)
        if restart:
            run.state["procs"] = {"trainer": {"pid": 2 ** 22 + 1, "log": "x"}}
        listed = []

        def objects():
            if isinstance(keys, Exception):
                raise keys
            return SimpleNamespace(list=lambda prefix: listed.append(prefix) or
                                   [k for k in keys if k.startswith(prefix)])

        run._objects = objects
        run.listed = listed
        return run

    return build


RELEASED = ["policy-releases/r1-v1/RELEASE.json", "policy-releases/LATEST.json"]


@pytest.mark.parametrize("extra,restart,keys,adds", [
    (["--rl-round-cuts", "off"], True, RELEASED, True),
    (["--rl-round-cuts=off"], True, RELEASED, True),
    (["--rl-round-cuts", "off"], False, RELEASED, False),       # the first launch
    (["--rl-round-cuts", "off"], True, [], False),               # no release yet: from the base
    (["--rl-round-cuts", "off"], True, ["sample-buffer/shards/a"], False),
    ([], True, RELEASED, False),                                 # auto: the round cut resumes
    (["--rl-round-cuts", "auto"], True, RELEASED, False),
    (["--rl-round-cuts", "off", "--rl-init-from-release", "r1-v1"], True, RELEASED, False),
    (["--rl-round-cuts", "off", "--rl-init-from-release=latest"], True, RELEASED, False),
])
def test_run_up_restart_adds_init_from_release(run_of, extra, restart, keys, adds):
    run = run_of(extra, restart=restart, keys=keys)
    argv = run.trainer_argv(now=1234.0)
    assert (argv[-2:] == ["--rl-init-from-release", "latest"]) is adds
    assert argv.count("--rl-init-from-release") + sum(a.startswith("--rl-init-from-release=")
                                                      for a in argv) <= 1
    if adds:
        assert "--rl-ledger-set-aside-before" in argv
        assert run.state["steps"]["trainer_init_from_release"]["selector"] == "latest"


def test_run_up_restart_fails_closed_when_the_store_cannot_be_listed(run_of):
    run = run_of(["--rl-round-cuts", "off"], restart=True, keys=OSError("no route"))
    with pytest.raises(RuntimeError, match="cannot list the store's policy releases"):
        run.trainer_argv()
    # the first launch never lists the store
    assert run_of(["--rl-round-cuts", "off"], restart=False, keys=OSError("x")).trainer_argv()


# ------------------------------------------------------------------ K3.1 examples
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "k31"


@pytest.mark.parametrize("script", ["launch-trainer-modal-b300.sh", "launch-convertible-modal.sh"])
def test_the_k31_full_parameter_launchers_turn_round_cuts_off(script):
    text = (EXAMPLES / script).read_text()
    assert "--rl-round-cuts off" in text
    assert "--rl-cut-every" not in text and "--rl-cut-auto-mtbf-s" not in text


@pytest.mark.parametrize("env_overrides", [{"STAGE": "1"}, {"STAGE": "2"}, {"STAGE": "2", "FORM": "synced-2"}])
def test_the_k31_trainer_example_argv_passes_the_launch_checks(tmp_path, env_overrides):
    from yeto import launcher

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    out = tmp_path / "argv.json"
    stub = bin_dir / "yeto"
    stub.write_text(f"#!{sys.executable}\nimport json, sys\njson.dump(sys.argv[1:], open({str(out)!r}, 'w'))\n")
    stub.chmod(0o755)
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "MODEL": "org/k31", "MODEL_REVISION": "r",
           "DATA": "d", "STORE_URI": "s3://b/r", "STORE_ENDPOINT": "https://e", "STORE_SECRET": "s",
           "EXPERT_SELECTION": "sel.json", "RL_IMAGE": "docker:i@sha256:0",
           "PARAMETER_LAYOUT_SHA256": "0" * 64, **env_overrides}
    subprocess.run(["bash", str(EXAMPLES / "launch-trainer-modal-b300.sh")], env=env, check=True)
    argv = json.loads(out.read_text())
    args = cli.build_parser().parse_args(argv)
    assert args.rl_round_cuts == "off"
    launcher._check_resume_launch_flags(args)
