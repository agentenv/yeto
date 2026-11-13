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

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401

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


INIT_V5 = {"path": "r5-v5", "version": 5, "policy_hash": "h5"}


def _resumed(monkeypatch, *, buffer_trained=640):
    from yeto.rl.adapters.miles import sampler_island

    resumed = SimpleNamespace(base_version=5, start_version=5, buffer_trained=buffer_trained,
                              history=lambda: [(5, "h5")], known_versions=lambda: [5])
    monkeypatch.setattr(sampler_island, "resumed_releases", lambda releases, limit: resumed)
    return sampler_island


def test_a_release_restart_reads_the_buffer_on_from_the_trained_position(monkeypatch):
    sampler_island = _resumed(monkeypatch)
    driver = _Driver()
    calls = []
    driver.buffer_start_without_cuts = lambda position: calls.append(position) or position
    assert sampler_island._resume_single_trainer(driver, releases=None, init_release=INIT_V5,
                                                 round_cuts_off=True) == {"resumed_base": 5}
    assert calls == []  # the reader exists only once the sync starts
    driver.sync.start(driver)
    assert calls == [640] and driver.sync.resumed_at == 5


def test_off_refuses_a_restart_over_releases_without_the_release_loaded(monkeypatch):
    # review blocker: base weights, numbering at v5, buffer from 640 -> [0, 640) never trained
    sampler_island = _resumed(monkeypatch)
    driver = _Driver()
    driver.buffer_start_without_cuts = lambda position: position
    with pytest.raises(sampler_island.RoundCutsOffRestartError,
                       match="--rl-round-cuts off: the store already holds policy releases up to v5.*"
                             "--rl-init-from-release latest"):
        sampler_island._resume_single_trainer(driver, releases=None, round_cuts_off=True)
    assert driver.sync.resumed_at == 0 and driver.events == []


def test_off_starts_from_the_base_over_a_store_without_releases(monkeypatch):
    from yeto.rl.adapters.miles import sampler_island

    monkeypatch.setattr(sampler_island, "resumed_releases", lambda releases, limit: None)
    assert sampler_island._resume_single_trainer(_Driver(), releases=None, round_cuts_off=True) == {}


def test_auto_without_a_round_cut_still_starts_from_the_base(monkeypatch):
    sampler_island = _resumed(monkeypatch)
    driver = _Driver()
    assert sampler_island._resume_single_trainer(driver, releases=None) == {"resumed_base": 5}
    assert driver.base_policy_is_start_state is True


def test_off_a_refused_start_position_reads_from_0_as_with_round_cuts(monkeypatch):
    sampler_island = _resumed(monkeypatch)
    driver = _Driver()
    calls = []

    def start(position):
        calls.append(position)
        if position:
            raise ValueError("beyond the index")
        return position

    driver.buffer_start_without_cuts = start
    sampler_island._resume_single_trainer(driver, releases=None, init_release=INIT_V5, round_cuts_off=True)
    driver.sync.start(driver)
    assert calls == [640, 0]
    kind, fields = driver.events[-1]
    assert kind == "rl_buffer_start_position_refused" and fields["position"] == 640


def test_off_refuses_a_sync_without_the_buffer_start_hook(monkeypatch):
    sampler_island = _resumed(monkeypatch)
    driver = _Driver()
    driver.sync = SimpleNamespace(resume_at=lambda *a, **k: None, round_cuts=None)
    driver.buffer_start_without_cuts = lambda position: position
    with pytest.raises(sampler_island.RoundCutsOffRestartError, match="no buffer_start_without_cuts hook"):
        sampler_island._resume_single_trainer(driver, releases=None, init_release=INIT_V5,
                                              round_cuts_off=True)


# ------------------------------------------------------------------ launcher: --if-released
def _trainer_task(extra=()):
    from test_rl_algorithm_provenance import _launcher_args

    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launcher import _island_task, _prepare_rl_args, make_miles_island_task

    args = _launcher_args("ports", ("--rl-single-island-no-sync", "--tuning", "full",
                                    "--rl-sample-role", "trainer",
                                    "--rl-checkpoint-store", "modal-volume://yeto-l54/run-a",
                                    "--rl-max-policy-age", "2", "--rl-algorithm-spec",
                                    "examples/rl_algorithms/decoupled-ppo-c2.json",
                                    "--cluster-prefix", "off-run", *extra), gpu="modal:1xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    return _island_task(make_miles_island_task, args, parse_gpu_spec(args.gpu)[0], 0, 1, "none", {})


def test_the_launcher_prepares_the_newest_release_on_every_off_trainer_node(_fake_sky, _no_modal_listing):
    run = str(_trainer_task(("--rl-round-cuts", "off")).run)
    prepare = run.index("-m yeto.rl.engine.init_from_release --release latest")
    assert "--if-released" in run[prepare:].split("\n", 1)[0]
    assert prepare < run.index("MASTER_ADDR=")  # before the node's Ray starts
    learner = run.split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    assert "--rl-init-from-release-dir ~/yeto-init-release" in learner
    assert "--rl-round-cuts off" in learner
    auto = str(_trainer_task().run)
    assert "init_from_release" not in auto and "--rl-init-from-release-dir" not in auto
    named = str(_trainer_task(("--rl-round-cuts", "off", "--rl-init-from-release", "r1-v1")).run)
    assert "--release r1-v1" in named and "--if-released" not in named


@pytest.mark.parametrize("fields,adds", [
    (dict(rl_round_cuts="off", tuning="full", rl_sample_role="trainer"), True),
    (dict(rl_round_cuts="off", tuning="full", rl_sample_role="both"), True),
    (dict(rl_round_cuts="off", tuning="full", rl_sample_role="sampler"), False),
    (dict(rl_round_cuts="off", tuning="lora", rl_sample_role="trainer"), False),
    (dict(rl_round_cuts="off", tuning="full", rl_sample_role="trainer", rl_init_from_release="latest"), False),
    (dict(rl_round_cuts="auto", tuning="full", rl_sample_role="trainer"), False),
])
def test_which_islands_prepare_if_released(fields, adds):
    from yeto.launch.rl_island_task import init_release_block, init_release_if_released

    args = SimpleNamespace(model_revision=None, rl_expert_selection=None, **fields)
    assert init_release_if_released(args) is adds
    assert ("--if-released" in init_release_block(args, "m")) is adds


def _release_store(tmp_path, versions=()):
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore

    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=5)
    for version in versions:
        releases.release(version, f"h{version}", lambda d: (d / "config.json").write_text("{}"))
    return releases


def test_if_released_prepares_latest_or_marks_the_start_from_the_base(tmp_path, capsys):
    from yeto.rl.engine.init_from_release import (NO_RELEASE, RECORD, island_model_path,
                                                  prepare_if_released)

    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    resolved = []
    assert prepare_if_released(_release_store(tmp_path), lambda: resolved.append(1) or str(base), out) is None
    assert resolved == [] and (out / NO_RELEASE).is_file() and not (out / RECORD).exists()
    assert island_model_path(out, base) == (str(base), None)
    assert "no policy release yet" in capsys.readouterr().out
    record = prepare_if_released(_release_store(tmp_path, (1, 2)), lambda: str(base), out)
    assert record["path"] == "r2-v2" and not (out / NO_RELEASE).exists()
    assert island_model_path(out, base)[1]["version"] == 2
    assert "adding --rl-init-from-release latest" in capsys.readouterr().out


def test_an_unprepared_node_still_refuses_to_start(tmp_path):
    from yeto.rl.engine.init_from_release import InitFromReleaseError, island_model_path

    with pytest.raises(InitFromReleaseError, match="did not prepare"):
        island_model_path(tmp_path, tmp_path)


def test_if_released_main_takes_only_latest(tmp_path):
    from yeto.rl.engine.init_from_release import InitFromReleaseError, main

    with pytest.raises(InitFromReleaseError, match="--release latest"):
        main(["--release", "r1-v1", "--if-released", "--model", "m", "--out", str(tmp_path),
              "--store", str(tmp_path)], environ={})
    _release_store(tmp_path)
    assert main(["--release", "latest", "--if-released", "--model", "org/not-downloaded",
                 "--out", str(tmp_path / "out"), "--store", str(tmp_path / "store")], environ={}) == 0


# ------------------------------------------------------------------ convertible trainer command
def test_the_convertible_trainer_command_continues_from_the_newest_release(tmp_path):
    from yeto.rl.adapters.miles.role_switch import continue_from_release

    calls = []

    def prepare(releases, base, out, *, expected_selection_sha256):
        calls.append((releases, out, expected_selection_sha256))

    command = ["python", "-m", "yeto.rl.adapters.miles.island_entry", "--model", "m",
               "--rl-sample-role", "trainer", "--rl-round-cuts", "off"]
    argv = continue_from_release(command, "R", prepare=prepare)
    assert argv[-2:] == ["--rl-init-from-release-dir", "~/yeto-init-release"]
    assert calls == [("R", os.path.expanduser("~/yeto-init-release"), None)]
    named = command + ["--rl-init-from-release-dir", str(tmp_path)]
    assert continue_from_release(named, "R", prepare=prepare) == named and calls[-1][1] == str(tmp_path)
    auto = command[:-2]
    assert continue_from_release(auto, "R", prepare=prepare) == auto and len(calls) == 2


# ------------------------------------------------------------------ verl
def test_verl_island_refuses_off():
    # verl has no --rl-init-from-release and no read-on from buffer_trained: a relaunch
    # would train base weights under the store's numbering
    from yeto.rl.adapters.verl import round_cuts

    base = dict(rl_resume_store="/s", rl_island_cut_store=None, rl_cut_every=None, rl_cut_keep=None,
                rl_stop_after_rounds=None, rl_cut_auto_mtbf_s=None, rl_cut_max_interval_s=None,
                rl_resume_allow_config_change=False, rl_round_cuts="off")
    with pytest.raises(ValueError, match="not supported with --rl-backend verl"):
        round_cuts.resume_config(SimpleNamespace(**base))
    with pytest.raises(ValueError, match="--rl-round-cuts off"):
        round_cuts.resume_config(SimpleNamespace(**{**base, "rl_cut_every": 2}))
    assert round_cuts.resume_config(SimpleNamespace(**{**base, "rl_round_cuts": "auto",
                                                       "rl_single_island_no_sync": True})) is not None


def test_the_launcher_refuses_off_with_verl():
    from yeto import launcher

    args = _args(NO_SYNC + ["--rl-round-cuts", "off", "--rl-backend", "verl"])
    with pytest.raises(ValueError, match="not supported with --rl-backend verl"):
        launcher._check_resume_launch_flags(args)
    launcher._check_resume_launch_flags(_args(NO_SYNC + ["--rl-backend", "verl"]))


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
                                   [k for k in keys if k.startswith(prefix)],
                                   read=lambda key: _record_bytes(key, keys))

        run._objects = objects
        run.listed = listed
        return run

    return build


RELEASED = ["policy-releases/r1-v1/RELEASE.json", "policy-releases/LATEST.json"]
TORN = "policy-releases/r7-v7/RELEASE.json"  # its content is torn (_record_bytes)
DERIVED = "policy-releases/r8-v1/RELEASE.json"  # a derived (re-quantized) release


def _release_record(seq, version, **extra):
    from yeto.rl.engine.policy_release import SCHEMA

    return {"schema": SCHEMA, "seq": seq, "version": version, "policy_hash": f"h{version}",
            "path": f"r{seq}-v{version}", "files": {}, "weights": "full", **extra}


def _record_bytes(key, keys):
    """The store content behind the fixture keys."""
    if key not in keys:
        return None
    if key == TORN:
        return b"{not json"
    if key == DERIVED:
        return json.dumps(_release_record(8, 1, derived_from={"scheme": "fp8"})).encode()
    return json.dumps(_release_record(1, 1)).encode()


@pytest.mark.parametrize("extra,restart,keys,adds", [
    (["--rl-round-cuts", "off"], True, RELEASED, True),
    (["--rl-round-cuts=off"], True, RELEASED, True),
    (["--rl-round-cuts", "off"], False, RELEASED, False),       # the first launch
    (["--rl-round-cuts", "off"], True, [], False),               # no release yet: from the base
    (["--rl-round-cuts", "off"], True, ["sample-buffer/shards/a"], False),
    # the latest resolution's view: a RELEASE.json without LATEST is a release ...
    (["--rl-round-cuts", "off"], True, ["policy-releases/r1-v1/RELEASE.json"], True),
    (["--rl-round-cuts", "off"], True, ["policy-releases/LATEST.json"], True),
    # ... a torn or a derived record is not (latest would find no trained release)
    (["--rl-round-cuts", "off"], True, [TORN], False),
    (["--rl-round-cuts", "off"], True, [DERIVED], False),
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
