"""launcher.run() copies its fleet.jsonl into the run store when it keeps island records
there (``dashboard.fleet_store``); see test_fleet_log_store.py for the unit cases."""

from __future__ import annotations

import pytest

import test_k31_customer_plan_golden as golden
import yeto.launcher as launcher
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401  (autouse fixtures)
from test_rl_launcher_island_failure import _setup
from test_rl_ssh_resident import NAMES, FakeCluster, _cards  # noqa: F401  (autouse fixture)
from test_rl_ssh_resident_launch import _args, _drive, _finish_when_started
from yeto import runs


def _store_keys(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.jsonl"))


def test_a_launch_with_island_records_leaves_its_fleet_log_in_the_store(monkeypatch, tmp_path):
    """launcher.run() of a resident ssh: island with its island records in a local run store
    (``--island-lifecycle on``): at its end the store holds the run's fleet.jsonl, the
    teardown's lines included, under the ``fleet`` lane."""
    cluster = FakeCluster(NAMES)
    monkeypatch.setattr("yeto.launch.ssh_resident.SshOps", lambda: cluster)
    store = tmp_path / "run-store"
    record, clock = [], _drive(monkeypatch)
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path, "--island-lifecycle", "on", "--rl-checkpoint-store", str(store))
    _finish_when_started(clock, cluster)
    assert launcher.run(args) == 0
    local = (runs.run_dir(args.cluster_prefix) / "fleet.jsonl").read_bytes()
    keys = [k for k in _store_keys(store) if k.startswith("fleet/log/")]
    assert keys and {k.split("/")[2] for k in keys} == {"fleet"}
    assert b"".join((store / k).read_bytes() for k in keys) == local
    assert b'"island_ready"' in local


def test_a_launch_without_island_records_uploads_nothing(monkeypatch, tmp_path):
    """One trainer island, island lifecycle off (the default for one island): main's
    behaviour, the store gets no fleet log even with a checkpoint store."""
    cluster = FakeCluster(NAMES)
    monkeypatch.setattr("yeto.launch.ssh_resident.SshOps", lambda: cluster)
    store = tmp_path / "run-store"
    record, clock = [], _drive(monkeypatch)
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    args = _args(tmp_path, "--rl-checkpoint-store", str(store))
    _finish_when_started(clock, cluster)
    assert launcher.run(args) == 0
    assert (runs.run_dir(args.cluster_prefix) / "fleet.jsonl").is_file()
    assert not [k for k in _store_keys(store) if k.startswith("fleet/")] if store.exists() else True


# -- the customer path ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(golden.CASES))
def test_the_customer_runs_keep_no_island_records_so_no_fleet_log_upload(case, tmp_path, monkeypatch):
    """Every customer K3.1 run (one buffer trainer, async GRPO, no fleet): the launch keeps
    no island records in the store (``island_lifecycle_enabled`` False, as on main), so the
    launcher builds its fleet log with ``objects=None`` and nothing is uploaded."""
    import yeto.cli as cli
    from yeto.gpu_spec import parse_gpu_spec

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    args = cli.parse_args(golden._argv(golden.CASES[case], tmp_path))  # noqa: SLF001
    assert not getattr(args, "dry_run", False) and getattr(args, "island_lifecycle", None) in (None, "auto")
    assert launcher.island_lifecycle_enabled(args, parse_gpu_spec(args.gpu)) is False


class Recording:
    def __init__(self):
        self.keys = []

    def write(self, key, data):
        self.keys.append(key)


def test_the_tape_uploader_default_keys_are_main_s(tmp_path):
    """Without ``prefix`` the uploader writes main's keys (``tapes/<island>/<incarnation>/<seq>``),
    and the trainer / syncer uploaders pass none."""
    import inspect

    from yeto.rl.engine import tape_upload

    assert tape_upload.chunk_key("trainer", "l0-h-20261127T000000Z", 3) == "tapes/trainer/l0-h-20261127T000000Z/00000003.jsonl"
    assert tape_upload.chunk_key("a/b", "", 0) == "tapes/a_b/_/00000000.jsonl"
    tape = tmp_path / "tape.jsonl"
    tape.write_text('{"a": 1}\n{"b": 2}\n')
    objects = Recording()
    up = tape_upload.TapeUploader(objects, tape, island="syncer", incarnation="h-1", interval_s=30.0)
    assert up.prefix == "tapes" and up.interval_s == 30.0 and up.flush() > 0
    assert objects.keys == ["tapes/syncer/h-1/00000000.jsonl"]
    for fn in (tape_upload.trainer_uploader, tape_upload.syncer_uploaders):
        assert "prefix" not in inspect.getsource(fn)


def test_the_store_copy_stops_when_the_fleet_controller_cannot_be_built(monkeypatch, tmp_path):
    """The fleet log (and its store copy) is built before the controller; a controller that
    raises in its constructor still gets the copy stopped by the launcher's teardown."""
    from yeto.dashboard import fleet_store

    stopped = []

    class Uploader:
        island = "fleet"

        def stop(self, timeout_s=None):
            stopped.append(timeout_s)
            return True

    monkeypatch.setattr(fleet_store, "start_uploader", lambda *a, **k: Uploader())
    cluster = FakeCluster(NAMES)
    monkeypatch.setattr("yeto.launch.ssh_resident.SshOps", lambda: cluster)
    record, clock = [], _drive(monkeypatch)
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)

    def broken(*a, **kw):
        raise RuntimeError("controller init failed")

    monkeypatch.setattr(launcher, "FleetController", broken)
    args = _args(tmp_path, "--island-lifecycle", "on", "--rl-checkpoint-store", str(tmp_path / "run-store"))
    with pytest.raises(RuntimeError, match="controller init failed"):
        launcher.run(args)
    assert stopped == [fleet_store.FINAL_UPLOAD_S]
