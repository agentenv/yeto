"""rl-multinode-island Q4 (C5): the island state dir (journal, cuts, ledger) is copied to an
off-island checkpoint store after every commit point and restored on an empty state dir
(machine replaced); --rl-checkpoint-store plumbing (cli -> launcher -> learner -> controller)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_rl_reconfig_recovery import SUCCEEDED, _island, read_epochs

from yeto import launcher
from yeto.rl.engine.controller import STORE_MANIFEST
from yeto.rl.engine.journal import read_journal


def _journal(root, kind):
    return [r for r in read_journal(root / "reconfig") if r["kind"] == kind]


def _manifest(store):
    return json.loads((store / STORE_MANIFEST).read_text(encoding="utf-8"))


# ------------------------------------------------------------- controller sync / restore
def test_store_is_synced_at_startup_and_after_commit_and_restored_on_an_empty_state_dir(tmp_path):
    store = tmp_path / "store"
    nodes = {"n0": 4, "n1": 4}
    kw = {"topology": (2, 4), "node_probe": lambda: dict(nodes), "checkpoint_store": store}
    clock = {"t": 1000.0}
    driver, ctl, *_ = _island(tmp_path / "island1", clock=clock, controller_kw=kw)
    # the topology/baseline record is already off-island (a run without any transaction
    # still leaves a restorable journal)
    assert ctl.last_store_sync["ok"] and ctl.last_store_sync["reason"] == "topology"
    assert _manifest(store)["config_epoch"] == 0 and (store / "reconfig/journal.jsonl").exists()
    assert not (store / "reconfig/journal.lock").exists()
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("up", "T4R4S0", 0, 60)
        return orig(rollout_id)

    driver.safe_point = safe_point
    driver.run()
    assert ctl.status("up")["phase"] == SUCCEEDED
    assert ctl.last_store_sync["ok"] and ctl.last_store_sync["reason"] == "SUCCEEDED"
    assert _manifest(store)["config_epoch"] == 1
    assert read_epochs(store / "reconfig").config_id == "T4R4S0"
    assert [r["phase"] for r in _journal(store, "phase")][-1] == SUCCEEDED  # the copy holds the commit
    assert _manifest(store)["seq"] <= read_journal(tmp_path / "island1/state/reconfig")[-1]["seq"]
    ctl.close()
    # node lost, cluster torn down: a new island on fresh machines with an EMPTY state
    # dir restores the store first and continues from the committed epoch
    _d2, ctl2, fork2, *_ = _island(tmp_path / "island2", clock=clock,
                                   controller_kw=dict(kw, node_probe=lambda: {"m0": 4, "m1": 4}))
    assert ctl2.journal.epochs.config_epoch == 1 and ctl2.journal.epochs.config_id == "T4R4S0"
    restored = _journal(tmp_path / "island2/state", "checkpoint_store")
    assert restored and restored[0]["action"] == "restore" and restored[0]["restored_from"]["config_epoch"] == 1
    assert restored[0]["restored_from"]["incarnation"] == ctl.incarnation["id"]
    assert ctl2.layout_baseline() == ctl.layout  # the baseline came with the journal
    assert ctl2.inspect().health == "RECOVERING"  # restart recovery of the 4 committed members (3.7)
    ctl2.close()


def test_local_journal_wins_over_the_store_and_a_missing_or_partial_store_is_ignored(tmp_path):
    store = tmp_path / "store"
    kw = {"topology": (2, 4), "node_probe": lambda: {"n0": 4, "n1": 4}, "checkpoint_store": store}
    _d, ctl, *_ = _island(tmp_path / "a", controller_kw=kw)
    ctl.close()
    # in-place restart: the local journal (same state dir) is used, nothing restored
    _d, ctl2, *_ = _island(tmp_path / "a", controller_kw=kw)
    assert not _journal(tmp_path / "a/state", "checkpoint_store")
    assert len(_journal(tmp_path / "a/state", "topology")) == 2
    ctl2.close()
    # a store without the manifest (sync interrupted) is not trusted
    (store / STORE_MANIFEST).unlink()
    _d, ctl3, *_ = _island(tmp_path / "b", controller_kw=kw)
    assert ctl3.journal.epochs.config_epoch == 0 and not _journal(tmp_path / "b/state", "checkpoint_store")
    assert len(_journal(tmp_path / "b/state", "topology")) == 1
    ctl3.close()
    # no store at all: node0-local behavior (single-node islands never set one)
    _d, ctl4, *_ = _island(tmp_path / "c", controller_kw={"topology": (2, 4), "node_probe": lambda: {"n0": 4, "n1": 4}})
    assert ctl4.checkpoint_store is None and ctl4.last_store_sync is None
    assert ctl4.sync_checkpoint_store("x") is False
    ctl4.close()


def test_store_sync_failure_never_fails_the_commit(tmp_path, caplog):
    store = tmp_path / "store"
    store.write_text("not a directory", encoding="utf-8")
    kw = {"topology": (2, 4), "node_probe": lambda: {"n0": 4, "n1": 4}, "checkpoint_store": store}
    _d, ctl, *_ = _island(tmp_path / "a", controller_kw=kw)
    assert ctl.inspect().health == "RUNNING"
    assert ctl.last_store_sync["ok"] is False and "Error" in ctl.last_store_sync["error"]
    assert any("checkpoint store sync (topology)" in r.message for r in caplog.records)
    ctl.close()


# ------------------------------------------------------------- launcher plumbing
def test_checkpoint_store_plan_bucket_uri_and_shared_path():
    assert launcher.rl_checkpoint_store_plan(SimpleNamespace()) is None
    assert launcher.rl_checkpoint_store_plan(SimpleNamespace(rl_checkpoint_store=None)) is None
    assert launcher.rl_checkpoint_store_plan(SimpleNamespace(rl_checkpoint_store="s3://bkt/run1/")) == (
        launcher.ELASTIC_CHECKPOINT_STORE_MOUNT + "/run1", "s3://bkt")
    # sky MOUNT accepts only a bucket root: nested prefix becomes a subdir of the mount
    assert launcher.rl_checkpoint_store_plan(SimpleNamespace(rl_checkpoint_store="s3://bkt/a/b")) == (
        launcher.ELASTIC_CHECKPOINT_STORE_MOUNT + "/a/b", "s3://bkt")
    assert launcher.rl_checkpoint_store_plan(SimpleNamespace(rl_checkpoint_store="gs://bkt")) == (
        launcher.ELASTIC_CHECKPOINT_STORE_MOUNT, "gs://bkt")
    assert launcher.rl_checkpoint_store_plan(SimpleNamespace(rl_checkpoint_store="/mnt/shared/run1/")) == (
        "/mnt/shared/run1", None)
    assert launcher.rl_checkpoint_store_plan(SimpleNamespace(rl_checkpoint_store="~/nfs/x")) == ("~/nfs/x", None)
    for bad in ("relative/path", "/mnt/../x", "s3://", "s3:///x"):
        with pytest.raises(ValueError, match="--rl-checkpoint-store"):
            launcher.rl_checkpoint_store_plan(SimpleNamespace(rl_checkpoint_store=bad))


def _two_node_elastic(monkeypatch, tmp_path, *extra, gpu="nebius:2x2xl40s"):
    from pathlib import Path

    from test_rl_launcher_multinode import _elastic_task

    cfg = Path(__file__).parent / "multinode_gpu/resources-2x2.json"
    return _elastic_task(monkeypatch, tmp_path, gpu, cfg, "T2R1S1", rollout=1, standby=1, extra=extra)


def test_bucket_store_is_mounted_on_the_island_and_the_learner_gets_the_mount_path(monkeypatch, tmp_path, capsys):
    args, _spec, task = _two_node_elastic(monkeypatch, tmp_path, "--rl-checkpoint-store", "s3://bkt/run1")
    assert f" --rl-elastic-checkpoint-store '{launcher.ELASTIC_CHECKPOINT_STORE_MOUNT}/run1'" in task.run
    mount = task.storage_mounts[launcher.ELASTIC_CHECKPOINT_STORE_MOUNT]
    assert "~/yeto-rl" in task.storage_mounts  # the spot completed-groups mount is kept alongside
    assert mount.source == "s3://bkt" and mount.mode == "mount" and mount.persistent is True
    assert "warning: --rl-checkpoint-store not set" not in capsys.readouterr().err


def test_shared_path_store_is_passed_through_without_a_mount(monkeypatch, tmp_path, capsys):
    args, _spec, task = _two_node_elastic(monkeypatch, tmp_path, "--rl-checkpoint-store", "/mnt/nfs/run1")
    assert " --rl-elastic-checkpoint-store /mnt/nfs/run1" in task.run
    assert launcher.ELASTIC_CHECKPOINT_STORE_MOUNT not in getattr(task, "storage_mounts", {})
    assert "warning: --rl-checkpoint-store not set" not in capsys.readouterr().err


def test_multi_node_island_without_a_store_warns_and_keeps_node0_local_state(monkeypatch, tmp_path, capsys):
    args, _spec, task = _two_node_elastic(monkeypatch, tmp_path)
    assert "--rl-elastic-checkpoint-store" not in task.run
    assert launcher.ELASTIC_CHECKPOINT_STORE_MOUNT not in getattr(task, "storage_mounts", {})
    err = capsys.readouterr().err
    assert "warning: --rl-checkpoint-store not set" in err and "node0's local disk" in err


def test_single_node_island_without_a_store_is_unchanged(monkeypatch, tmp_path, capsys):
    from test_rl_engine_selection import _island_task
    from test_rl_infra_switches import _elastic_cli, _elastic_files

    res, _ = _elastic_files(tmp_path)
    args = _elastic_cli(res, "--rl-elastic-initial-config", "c0", "--rl-placement", "fixed-partition",
                        "--rl-rollout-gpus", "1", "--gpu", "aws:2xa100@us-east-1")
    launcher._prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    assert "--rl-elastic-checkpoint-store" not in task.run
    assert "warning: --rl-checkpoint-store" not in capsys.readouterr().err


def test_store_flag_flows_from_cli_to_the_learner_and_needs_elastic(tmp_path, monkeypatch):
    from test_rl_engine_selection import _cli, _learner_argv
    from test_rl_infra_switches import ELASTIC_LEARNER

    from yeto.rl import learner

    with pytest.raises(ValueError, match="--rl-checkpoint-store need --rl-elastic"):
        launcher._check_ports_infra_switches(_cli(("--rl-checkpoint-store", "s3://b/x")), "ports")
    parsed = learner.parse_args(_learner_argv(ELASTIC_LEARNER + ("--rl-elastic-checkpoint-store", "/mnt/s")))
    miles_args = SimpleNamespace()
    learner.apply_ports_infra_switches(parsed, miles_args, {})
    assert miles_args.yeto_rl_elastic["checkpoint_store"] == "/mnt/s"
    parsed = learner.parse_args(_learner_argv(ELASTIC_LEARNER))
    miles_args = SimpleNamespace()
    learner.apply_ports_infra_switches(parsed, miles_args, {})
    assert "checkpoint_store" not in miles_args.yeto_rl_elastic
    with pytest.raises(SystemExit):
        learner.parse_args(_learner_argv(("--rl-elastic-checkpoint-store", "/mnt/s")))


def test_build_elastic_passes_the_store_to_the_controller(tmp_path):
    from test_rl_infra_switches import _elastic_files

    from yeto.rl.adapters.miles.elastic_wiring import build_elastic

    res, _ = _elastic_files(tmp_path)
    wiring = build_elastic(state_dir=tmp_path / "state", resources=res, attestation=None, profile=None,
                           initial_config="c0", runtime_fingerprint="fp", declared_cells=(),
                           checkpoint_store=str(tmp_path / "store"))
    assert wiring.controller.checkpoint_store == tmp_path / "store"
    wiring.controller.close()
    plain = build_elastic(state_dir=tmp_path / "state2", resources=res, attestation=None, profile=None,
                          initial_config="c0", runtime_fingerprint="fp", declared_cells=())
    assert plain.controller.checkpoint_store is None
    plain.controller.close()
