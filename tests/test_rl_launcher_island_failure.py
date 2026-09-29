"""launcher.run() terminates, tears everything down and exits 4 when a fixed-roster
RL island fails for good (Modal and sky islands, with and without a syncer)."""

from __future__ import annotations

import types

import pytest

import yeto.launcher as launcher
import yeto.modal_runner as modal_runner
from test_rl_algorithm_provenance import _fake_sky, _launcher_args  # noqa: F401  (fixture)


class Status:
    def __init__(self, text):
        self.text = text

    def is_terminal(self):
        return self.text in ("SUCCEEDED", "FAILED")

    def __str__(self):
        return f"JobStatus.{self.text}"


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        if self.t > 36000:  # a hang in fake time: fail instead of looping forever
            raise AssertionError("controller did not terminate")


def _setup(monkeypatch, tmp_path, *, failing, clock, record):
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(launcher, "prepare_launch_args", launcher._prepare_rl_args)
    monkeypatch.setattr(launcher, "warn_if_model_wont_fit", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "make_syncer_task", lambda *a, **k: object())
    monkeypatch.setattr(launcher, "_tail", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_tail_modal", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "FAILED_RUN_DRAIN_S", 0.0, raising=False)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "terminate_and_verify",
                        lambda sky, name, **k: record.append(("down", name)) or True)
    monkeypatch.setattr(modal_runner, "resolve_syncer_for_modal", lambda addr, public: addr)

    class FakeModalOps:
        def __init__(self, app_name):
            self.app_name, self.owner = app_name, {}

        def define(self, cfg):
            pass

        def deploy(self):
            pass

        def spawn(self, cfg):
            call = f"fc-{cfg.learner_id}-{len(self.owner)}"
            self.owner[call] = cfg.learner_id
            record.append(("spawn", cfg.learner_id))
            return call

        def status(self, call_id):
            return "FAILED" if self.owner[call_id] in failing else "RUNNING"

        def cancel(self, call_id):
            record.append(("cancel", call_id))

        def stop_app(self):
            record.append(("stop_app",))

        def tail_logs(self, call_id, entries=100):
            return []

    class FakeSkyOps:  # sky side of RoutingOps (sky islands and the syncer)
        def job_status(self, cluster, job_id):
            if "-l" in cluster and int(cluster.split("-l")[1].split("-")[0]) in failing:
                return Status("FAILED")
            return Status("RUNNING")

        def job_alive(self, cluster, job_id):
            return False

        def cluster_up(self, cluster):
            return True

        def relaunch(self, task, cluster):
            record.append(("relaunch", cluster))
            return 7

        def down(self, cluster):
            record.append(("down", cluster))

        def rl_strict_failure(self, cluster, job_id):
            return None

        def now(self):
            return clock.now()

        def sleep(self, seconds):
            clock.sleep(seconds)

    monkeypatch.setattr(modal_runner, "ModalOps", FakeModalOps)
    monkeypatch.setattr(launcher, "SkySDKOps", FakeSkyOps)
    import sys

    sky = sys.modules["sky"]
    sky.launch = lambda task, **kw: f"rid-{kw.get('cluster_name')}"
    sky.stream_and_get = lambda rid: (1, types.SimpleNamespace(head_ip="10.0.0.1"))


@pytest.mark.parametrize("provider", ["modal", "sky"])
@pytest.mark.parametrize("syncer", [True, False])
@pytest.mark.parametrize("recover_timeout", [0.0, 600.0])
def test_island_failure_exits_4_and_tears_everything_down(
        monkeypatch, tmp_path, provider, syncer, recover_timeout):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing={0}, clock=clock, record=record)
    one = "modal:1xa100" if provider == "modal" else "aws:1xa100@us-east-1"
    gpu = f"{one},{one}" if syncer else one
    extra = ("--controller", "local", "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64)
    if not syncer:
        extra += ("--rl-single-island-no-sync",)
    args = _launcher_args("ports", extra, gpu=gpu)
    args.keep = False
    args.recover_timeout = recover_timeout
    args.controller_poll = 30.0
    code = launcher.run(args)
    assert code == launcher.ISLAND_FAILED_EXIT == 4
    assert clock.t < 36000
    names = launcher.learner_cluster_names(args.cluster_prefix, launcher.parse_gpu_spec(gpu))
    if provider == "modal":
        assert ("stop_app",) in record
    else:
        for name in names:
            assert ("down", name) in record
    if syncer:
        assert ("down", f"{args.cluster_prefix}-syncer") in record
    if recover_timeout > 0:  # bounded relaunches of the failing island
        relaunches = [r for r in record if r[0] in ("relaunch",)] + [
            r for r in record if r == ("spawn", 0)][1:]
        assert 1 <= len(relaunches) <= launcher.FIXED_ROSTER_MAX_RELAUNCHES
