"""E3 (4.7) through the E1 IslandController (CPU protocol; needs infra-e3-controller.patch)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# E3 modules/fakes live on infra-e3 until integration (E1 applied the controller patch).
pytest.importorskip("yeto.rl.engine.trainer_transition")
pytest.importorskip("tests.rl_reshard_fakes")

from tests.test_rl_trainer_transition import SHA, CONFIGS, Pool, Publisher, _attestation, _spec, _world_trainer
from tests.rl_reshard_fakes import GBS, MBS, default_args
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.controller import REBUILT_OLD, SUCCEEDED, IslandController
from yeto.rl.engine.execution_profile import ExecutionProfile, ReadinessSnapshot
from yeto.rl.engine.journal import read_epochs


def _controller(tmp_path, ops):
    profile = ExecutionProfile(name="t", execution_mode="partitioned-serial",
                               outer_protocol="none").bind_algorithm(AlgorithmSpec())
    return IslandController(
        state_dir=tmp_path / "state", configs=CONFIGS, attestation=_attestation(), profile=profile,
        initial_config="T2R2", runtime_fingerprint="fp",
        trainer_edges=lambda: {"spec": _spec(), "args": ops.trainer._args, "global_batch_size": GBS,
                               "micro_batch_size": MBS, "ops": ops},
    )


class PoolWithStatus(Pool):
    """Fork membership mirror: every committed start/stop bumps the epoch."""

    epoch = 0

    def membership_status(self):
        return {"epoch": self.epoch}

    def remove_engines(self, members, *, epoch):
        assert epoch == self.epoch
        out = super().remove_engines(members, epoch=epoch)
        self.epoch += 1
        return out

    def add_engines(self, n, *, epoch, members):
        assert epoch == self.epoch
        out = super().add_engines(n, epoch=epoch, members=members)
        self.epoch += 1
        return out


def _driver(pool):
    return SimpleNamespace(rollout=pool, publisher=Publisher(), placement=None, ledger=None,
                           published_state=SimpleNamespace(policy_tensor_hash=lambda: "h"),
                           published_version=2, expected_token="t", config_epoch=0)


def _snapshot():
    return ReadinessSnapshot(rollout_id=2, optimizer_step=2, trained_policy_version=2,
                             published_policy_version=2, publication_complete=True, driver_safe_point=True)


def test_controller_commits_a_role_transfer(tmp_path):
    ranks, actor, ops, built = _world_trainer(2, tmp_path)
    pool = PoolWithStatus({"e0", "e1"})
    ctl = _controller(tmp_path, ops)
    ctl.open(pool)
    ctl.request("r1", "T1R3", 0, deadline_s=600)
    driver = _driver(pool)
    assert ctl.run_at_safe_point(driver, _snapshot()) == SUCCEEDED
    epochs = read_epochs(tmp_path / "state" / "reconfig")
    assert (epochs.config_epoch, epochs.config_id, set(epochs.members)) == (1, "T1R3", {"e0", "e1", "e2"})
    assert driver.config_epoch == 1 and ops.trainer.actual_layout()["dp"] == 1


def test_controller_rebuilds_old_on_failure(tmp_path):
    ranks, actor, ops, built = _world_trainer(2, tmp_path)
    pool = PoolWithStatus({"e0", "e1"})
    pool.fail_start = True
    ctl = _controller(tmp_path, ops)
    ctl.open(pool)
    ctl.request("r1", "T1R3", 0, deadline_s=600)
    assert ctl.run_at_safe_point(_driver(pool), _snapshot()) == REBUILT_OLD
    epochs = read_epochs(tmp_path / "state" / "reconfig")
    assert (epochs.config_epoch, epochs.config_id) == (0, "T2R2")
    assert ops.trainer.actual_layout()["dp"] == 2


def test_commit_cas_failure_enters_recovery_with_restore_old_hint(tmp_path, monkeypatch):
    import pytest

    from yeto.rl.engine.controller import RecoveryRequired
    from yeto.rl.engine.journal import EpochConflict, read_journal

    ranks, actor, ops, built = _world_trainer(2, tmp_path)
    pool = PoolWithStatus({"e0", "e1"})
    ctl = _controller(tmp_path, ops)
    ctl.open(pool)
    ctl.request("r1", "T1R3", 0, deadline_s=600)
    real = ctl.journal.compare_and_swap

    def cas(**kw):
        if kw["new"].config_id == "T1R3":
            raise EpochConflict("epochs file changed behind the single writer")
        return real(**kw)

    monkeypatch.setattr(ctl.journal, "compare_and_swap", cas)
    with pytest.raises(RecoveryRequired):
        ctl.run_at_safe_point(_driver(pool), _snapshot())
    hints = [r for r in read_journal(tmp_path / "state" / "reconfig") if r["kind"] == "trainer_recovery_hint"]
    assert hints and hints[-1]["action"] == "restore_old" and hints[-1]["cut_epoch"] == 0
    assert read_epochs(tmp_path / "state" / "reconfig").config_id == "T2R2"


def test_commit_failing_after_the_epochs_were_written_hints_restore_target(tmp_path, monkeypatch):
    """Review M1: os.replace succeeded, the directory fsync raised -> the commit is durable."""
    import pytest

    from yeto.rl.engine import journal as journal_mod
    from yeto.rl.engine.controller import RecoveryRequired
    from yeto.rl.engine.journal import read_journal

    ranks, actor, ops, built = _world_trainer(2, tmp_path)
    pool = PoolWithStatus({"e0", "e1"})
    ctl = _controller(tmp_path, ops)
    ctl.open(pool)
    ctl.request("r1", "T1R3", 0, deadline_s=600)
    real = journal_mod._fsync_dir
    armed = {"on": False}
    real_cas = ctl.journal.compare_and_swap

    def cas(**kw):
        armed["on"] = kw["new"].config_id == "T1R3"
        return real_cas(**kw)

    def fsync_dir(path):
        if armed["on"]:
            raise OSError("fsync failed")
        return real(path)

    monkeypatch.setattr(ctl.journal, "compare_and_swap", cas)
    monkeypatch.setattr(journal_mod, "_fsync_dir", fsync_dir)
    with pytest.raises(RecoveryRequired):
        ctl.run_at_safe_point(_driver(pool), _snapshot())
    monkeypatch.setattr(journal_mod, "_fsync_dir", real)
    hints = [r for r in read_journal(tmp_path / "state" / "reconfig") if r["kind"] == "trainer_recovery_hint"]
    assert hints[-1]["action"] == "restore_target" and hints[-1]["committed"] is True
    assert read_epochs(tmp_path / "state" / "reconfig").config_id == "T1R3"
