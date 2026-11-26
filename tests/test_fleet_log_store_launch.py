"""launcher.run() copies its fleet.jsonl into the run store when it keeps island records
there (``dashboard.fleet_store``); see test_fleet_log_store.py for the unit cases."""

from __future__ import annotations

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
