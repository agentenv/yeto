"""launcher.run() terminates, tears everything down and exits 4 when a fixed-roster
RL island fails for good (Modal and sky islands, with and without a syncer)."""

from __future__ import annotations

import types

import pytest

import yeto.launcher as launcher
import yeto.modal_runner as modal_runner
from test_rl_algorithm_provenance import _fake_sky, _launcher_args  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def _no_modal_listing(monkeypatch):
    # provider listing after teardown: nothing listed (= stopped) unless a test says so
    monkeypatch.setattr(launcher, "_list_modal_apps", lambda: [])
    monkeypatch.setattr(launcher, "MODAL_STOP_VERIFY_DELAY_S", 0.0)


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
        import time

        time.sleep(0.005)  # let the log-tail threads run (real time passes too)
        self.t += seconds
        if self.t > 36000:  # a hang in fake time: fail instead of looping forever
            raise AssertionError("controller did not terminate")


def _setup(monkeypatch, tmp_path, *, failing, clock, record, succeeding=()):
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
                        lambda sky, name, **k: record.append(("terminate", name)) or True)
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
            island = self.owner[call_id]
            return ("FAILED" if island in failing
                    else "SUCCEEDED" if island in succeeding else "RUNNING")

        def cancel(self, call_id):
            record.append(("cancel", call_id))

        def app_status(self):  # provider view after stop (P0 teardown check)
            return None if getattr(self, "_stopped", False) else ("deployed", 1)

        def stop_app(self):
            self._stopped = True
            record.append(("stop_app",))

        def tail_logs(self, call_id, entries=100):
            return []

    class FakeSkyOps:  # sky side of RoutingOps (sky islands and the syncer)
        def job_status(self, cluster, job_id):
            if "-l" in cluster:
                island = int(cluster.split("-l")[1].split("-")[0])
                if island in failing:
                    return Status("FAILED")
                if island in succeeding:
                    return Status("SUCCEEDED")
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
    # the launcher probes every syncer port before starting islands (s13-g3b)
    monkeypatch.setattr(launcher, "tcp_probe", lambda host, port, **kw: (True, f"{host}:{port} reachable"))


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
    # the controller downs the failed island itself (Modal: cancels its call;
    # sky: FakeSkyOps.down), the finally block tears down the rest
    if provider == "modal":
        assert [r for r in record if r[0] == "cancel" and r[1].startswith("fc-0-")]
        assert record.count(("stop_app",)) == 1
        if syncer:  # the surviving island is cancelled by the finally block
            assert [r for r in record if r[0] == "cancel" and r[1].startswith("fc-1-")]
    else:
        assert ("down", names[0]) in record  # controller
        assert ("terminate", names[0]) not in record  # not torn down twice
        for name in names[1:]:
            assert ("terminate", name) in record  # finally
    if syncer:
        assert ("terminate", f"{args.cluster_prefix}-syncer") in record
    if recover_timeout > 0:  # bounded relaunches of the failing island
        relaunches = [r for r in record if r[0] in ("relaunch",)] + [
            r for r in record if r == ("spawn", 0)][1:]
        assert 1 <= len(relaunches) <= launcher.FIXED_ROSTER_MAX_RELAUNCHES


@pytest.mark.parametrize("provider", ["modal", "sky"])
def test_error_after_finalized_is_success_no_recovery(monkeypatch, tmp_path, provider):
    """2a 7.6: both islands finalized, then island 1 dies in ray.shutdown (job FAILED)
    and the syncer stops -- a completed run, not an island failure."""
    from yeto.rl import event_echo

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing={1}, succeeding={0}, clock=clock, record=record)
    fin = {i: event_echo.format_record({"island_id": i, "time_unix": 1.0,
                                        "event": "rl_learner_finalized"}) for i in (0, 1)}

    def feed_all(*args, **kwargs):  # the tail thread delivers each island's finalized record
        collector = args[-1]
        if collector is not None:
            island = 0 if "-l0-" in str(collector.path) else 1
            collector.feed(fin[island])

    monkeypatch.setattr(launcher, "_tail_modal", feed_all)
    monkeypatch.setattr(launcher, "_tail", feed_all)
    one = "modal:1xa100" if provider == "modal" else "aws:1xa100@us-east-1"
    args = _launcher_args("ports", ("--controller", "local", "--rl-image",
                                    "docker:ghcr.io/x/y@sha256:" + "a" * 64), gpu=f"{one},{one}")
    args.keep, args.recover_timeout, args.controller_poll = False, 600.0, 30.0
    args.output = str(tmp_path / "out")
    code = launcher.run(args)
    assert code in (0, 2), code  # 2: Modal artifact not fetchable over ssh
    assert not [r for r in record if r[0] == "relaunch"]
    assert [r for r in record if r[0] == "spawn"] in ([("spawn", 0), ("spawn", 1)], [])


def _rows(app, *rows):
    return [{"app_id": i, "description": app, "state": st, "tasks": str(t),
             "created_at": c} for i, st, t, c in rows]


def _run_one_modal(monkeypatch, tmp_path, listing):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing={0}, clock=clock, record=record)
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync",
                                    "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64),
                          gpu="modal:1xa100")
    args.keep, args.recover_timeout = False, 0.0
    app = modal_runner.modal_app_name(args.cluster_prefix)
    monkeypatch.setattr(launcher, "_list_modal_apps", lambda: listing(app))
    code = launcher.run(args)
    from yeto import runs

    rec = __import__("json").loads((runs.run_dir(args.cluster_prefix) / "teardown.json").read_text())
    return code, rec


def test_modal_app_still_running_after_stop_warns_and_exits_5(monkeypatch, tmp_path, capsys):
    code, rec = _run_one_modal(monkeypatch, tmp_path, lambda app: _rows(
        app, ("ap-new", "deployed", 1, "2999-01-01T00:00:00+00:00")))
    assert code == launcher.TEARDOWN_UNVERIFIED_EXIT == 5
    assert "not confirmed stopped" in capsys.readouterr().err
    assert rec["confirmed_stopped"] is False and rec["this_run_app_ids"] == ["ap-new"]
    assert len(rec["checks"]) == launcher.MODAL_STOP_VERIFY_ATTEMPTS


def test_same_name_old_stopped_new_running_is_not_confirmed(monkeypatch, tmp_path):
    code, rec = _run_one_modal(monkeypatch, tmp_path, lambda app: _rows(
        app, ("ap-old", "stopped", 0, "2020-01-01T00:00:00+00:00"),
        ("ap-new", "deployed", 1, "2999-01-01T00:00:00+00:00")))
    assert code == 5 and rec["confirmed_stopped"] is False
    assert rec["this_run_app_ids"] == ["ap-new"]  # only this run's row, by created_at


def test_same_name_rows_all_stopped_is_confirmed(monkeypatch, tmp_path):
    code, rec = _run_one_modal(monkeypatch, tmp_path, lambda app: _rows(
        app, ("ap-old", "stopped", 0, "2020-01-01T00:00:00+00:00"),
        ("ap-new", "stopped", 0, "2999-01-01T00:00:00+00:00")))
    assert code == 4 and rec["confirmed_stopped"] is True  # 4: the island failed


def test_old_same_name_row_still_running_also_blocks(monkeypatch, tmp_path):
    code, rec = _run_one_modal(monkeypatch, tmp_path, lambda app: _rows(
        app, ("ap-old", "deployed", 2, "2020-01-01T00:00:00+00:00"),
        ("ap-new", "stopped", 0, "2999-01-01T00:00:00+00:00")))
    assert code == 5 and rec["confirmed_stopped"] is False


def test_run_manifest_records_image_and_pins(monkeypatch, tmp_path, capsys):
    import json as _json

    import yeto.rl as rl
    from yeto import runs

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing={0}, clock=clock, record=record)
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync"),
                          gpu="modal:1xa100")
    args.keep, args.recover_timeout = False, 0.0
    launcher.run(args)
    manifest = _json.loads((runs.run_dir(args.cluster_prefix) / "run_manifest.json").read_text())
    assert manifest["rl_image"] == rl.MILES_NEXT_IMAGE  # the ports default, no override
    assert manifest["miles_commit"] == rl.MILES_NEXT_COMMIT
    assert manifest["sglang_commit"] == rl.SGLANG_NEXT_COMMIT
    out = capsys.readouterr().out
    assert f"image {rl.MILES_NEXT_IMAGE}" in out and rl.MILES_NEXT_COMMIT in out


def test_stalled_run_exits_6_and_tears_down(monkeypatch, tmp_path, capsys):
    """INFRA 2.4 s29: the syncer connection died; islands keep RUNNING with no new event."""
    from yeto.rl import event_echo

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)

    def one_event(*args, **kwargs):  # each island logs one event, then nothing
        collector = args[-1]
        if collector is not None:
            collector.feed(event_echo.format_record(
                {"island_id": 0, "time_unix": 1.0, "event": "rl_engine_selected"}))

    monkeypatch.setattr(launcher, "_tail_modal", one_event)
    args = _launcher_args("ports", ("--controller", "local", "--rl-image",
                                    "docker:ghcr.io/x/y@sha256:" + "a" * 64),
                          gpu="modal:1xa100,modal:1xa100")
    args.keep, args.recover_timeout, args.controller_poll = False, 600.0, 30.0
    args.rl_stall_timeout = 900.0
    assert launcher.run(args) == launcher.RUN_STALLED_EXIT == 6
    assert 900 <= clock.t < 1000  # bounded by the stall timeout, no endless waiting
    assert "run stalled" in capsys.readouterr().err
    assert record.count(("stop_app",)) == 1  # every resource torn down
    assert not [r for r in record if r[0] == "relaunch"]


def test_modal_container_change_exits_7_and_stops_the_app(monkeypatch, tmp_path, capsys):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)

    def moved(*args, **kwargs):  # Modal re-ran the island in a second container
        guard = kwargs["guard"]
        guard.feed("[modal-island 0] rank 0 container ta-A")
        guard.feed("[modal-island 0] rank 0 container ta-B")

    monkeypatch.setattr(launcher, "_tail_modal", moved)
    args = _launcher_args("ports", ("--controller", "local", "--rl-image",
                                    "docker:ghcr.io/x/y@sha256:" + "a" * 64),
                          gpu="modal:1xa100,modal:1xa100")
    args.keep, args.recover_timeout, args.controller_poll = False, 600.0, 30.0
    args.rl_stall_timeout = 900.0
    assert launcher.run(args) == launcher.CONTAINER_CHANGED_EXIT == 7
    err = capsys.readouterr().err
    assert "Modal container changed" in err and "ta-A -> ta-B" in err
    assert ("stop_app",) in record


def test_stall_detection_disabled_with_zero(monkeypatch, tmp_path):
    ctl = launcher.FleetController(
        learners={}, syncer=None, sky_ops=types.SimpleNamespace(now=lambda: 0.0),
        poll_interval=30, recover_timeout=0, progress_probe=lambda: 5, stall_timeout=0.0)
    ctl._check_stall()  # never raises when disabled
