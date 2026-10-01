"""E1-D ④ (rl-infra-spec 3.3a/3.7, 2026-10-02 user ruling): a stop that keeps half-failing.

Expected, bounded sequence after the transaction deadline: exactly one REBUILD_OLD whose limit is
``deadline_wall + T_recovery`` (the transaction deadline is never reset), then RECOVERY_REQUIRED with
BOTH an island-level record (``request_id=None``) and a request-level terminal record for the request
(same error/epochs, ``cause``, reference to the island record). Nothing is prepared or trained after.
Chain 6r2 d4 failed on the GPU judge because only the island record existed (request terminal missing).
"""
from __future__ import annotations

import pytest

from yeto.rl.engine.controller import (
    QUIESCING,
    REBUILD_OLD,
    RECOVERY_REQUIRED,
    TRANSFERRING,
    VALIDATING,
    WAIT_SAFE,
    IslandController,
    Rejected,
    Timeouts,
)
from yeto.rl.engine.driver import DriverError
from yeto.rl.engine.journal import read_journal
from tests.test_rl_reconfig_e1 import FP, _attestation, _events, _profile, _setup, _up_then_down

RECOVERY, RETRY, DEADLINE = 3.0, 10.0, 60.0


def _run(tmp_path, stop_failures=99):
    driver, ctl, fork, pool, *_ = _setup(
        tmp_path, controller_kw={"timeouts": Timeouts(recovery=RECOVERY, retry_interval=RETRY)})
    with pytest.raises(DriverError, match="RECOVERY_REQUIRED"):
        _up_then_down(driver, ctl, pool, before_down=lambda: setattr(fork, "stop_failures", stop_failures))
    return driver, ctl, fork, pool, read_journal(tmp_path / "state/reconfig")


def test_permanent_stop_failure_sequence_and_time_limit(tmp_path):
    driver, ctl, fork, pool, journal = _run(tmp_path)
    req = next(r for r in journal if r["kind"] == "request" and r["request_id"] == "down")
    tx = req["tx_id"]
    phases = [r for r in journal if r["kind"] == "phase" and r.get("tx_id") == tx]
    assert [p["phase"] for p in phases] == [VALIDATING, WAIT_SAFE, QUIESCING, TRANSFERRING, REBUILD_OLD,
                                            RECOVERY_REQUIRED, RECOVERY_REQUIRED]
    rebuild = phases[4]
    island, request = phases[5], phases[6]
    # one bounded REBUILD_OLD; the overall limit is the transaction deadline + T_recovery and the
    # transaction deadline is not reset
    assert rebuild["cause"] == "stop_retry_deadline"
    assert rebuild["wall_time"] >= req["deadline_wall"]
    assert rebuild["deadline_wall"] == req["deadline_wall"]
    assert rebuild["recovery_deadline_wall"] == req["deadline_wall"] + RECOVERY
    assert request["wall_time"] <= req["deadline_wall"] + RECOVERY + RETRY  # one retry interval of slack
    assert request["wall_time"] - (req["deadline_wall"] - DEADLINE) <= DEADLINE + RECOVERY + RETRY
    # island + request records agree
    assert island["request_id"] is None and island["scope"] == "island"
    assert request["request_id"] == "down" and request["scope"] == "request"
    assert request["island_record_seq"] == island["seq"]
    assert island["cause"] == request["cause"] == "rebuild_old_failed"
    assert island["error"] == request["error"] and "ran past the deadline" in request["error"]
    assert (island["config_epoch"], island["fork_epoch"]) == (request["config_epoch"], request["fork_epoch"]) == (1, 1)
    assert request["deadline_wall"] == req["deadline_wall"]
    assert request["recovery_deadline_wall"] == req["deadline_wall"] + RECOVERY
    # status / inspect agree with the journal
    st = ctl.status("down")
    assert st["phase"] == RECOVERY_REQUIRED and st["terminal"] and st["error"] == request["error"]
    assert ctl.inspect().health == "RECOVERY_REQUIRED" and ctl.last_outcome["cause"] == "rebuild_old_failed"
    assert ctl.last_outcome["phase"] == RECOVERY_REQUIRED
    with pytest.raises(Rejected, match="RECOVERY_REQUIRED"):
        ctl.plan("T4R4S0", 1)
    # the stop was retried both before (deadline) and inside the rebuild (recovery budget), nothing else
    ops = [(r["op"], r["status"]) for r in journal if r["kind"] == "fork_op" and r["tx_id"] == tx]
    assert ops and all(op == "stop" for op, _ in ops)
    assert fork.incomplete is not None  # the fork still waits for the same stop


def test_nothing_prepared_or_trained_after_recovery_required(tmp_path):
    driver, ctl, fork, pool, journal = _run(tmp_path)
    # the ledger keeps real wall time, the controller a fake clock: compare by rollout. The down
    # request was made at the safe point of rollout 2, so only rollouts 0 and 1 may be prepared.
    ledger = [r for r in read_journal(tmp_path / "state/ledger") if r["kind"] in ("prepared", "optimizer_applied")]
    assert sorted(r["rollout_id"] for r in ledger if r["kind"] == "prepared") == [0, 1]
    assert sorted(r["rollout_id"] for r in ledger if r["kind"] == "optimizer_applied") == [0, 1]
    assert driver.ledger.unconsumed() == []
    events = _events(tmp_path)
    recs = [e for e in events if e["event"] == "rl_reconfiguration"]
    assert [e["result"] for e in recs] == ["SUCCEEDED", "RECOVERY_REQUIRED"]
    idx = events.index(recs[-1])
    assert not [e for e in events[idx:] if e["event"] == "rl_driver_phase" and e["phase"] in ("train", "generate")]
    # the driver refuses another round outright (ledger path guard)
    with pytest.raises(DriverError, match="RECOVERY_REQUIRED"):
        driver.run_round(99)


def test_replay_after_restart_keeps_both_terminal_records(tmp_path):
    driver, ctl, fork, pool, journal = _run(tmp_path)
    ctl.close()
    ctl2 = IslandController(state_dir=tmp_path / "state", configs=ctl.configs, attestation=_attestation(),
                            profile=_profile(), initial_config="T4R2S2", runtime_fingerprint=FP)
    assert ctl2.recovery_required and "rebuild of the old engine set failed" in ctl2.recovery_required
    assert ctl2._open_after_restart == []  # the request has a terminal record: nothing to resume
    assert ctl2.status("down")["phase"] == RECOVERY_REQUIRED
    # a second RECOVERY_REQUIRED on the same transaction adds no second request-level record
    n = len([r for r in journal if r["kind"] == "phase" and r["phase"] == RECOVERY_REQUIRED and r.get("request_id")])
    ctl2._enter_recovery(ctl.status("down")["tx_id"], "again")
    again = read_journal(tmp_path / "state/reconfig")
    assert len([r for r in again if r["kind"] == "phase" and r["phase"] == RECOVERY_REQUIRED and r.get("request_id")]) == n
    ctl2.close()
