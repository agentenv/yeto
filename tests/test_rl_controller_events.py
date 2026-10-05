"""dashboard 5.1/5.2: controller rl_cell_snapshot / rl_reconfig_phase events."""
from __future__ import annotations

from test_rl_reconfig_e1 import _attestation, _profile, _setup, _up_then_down, FP  # noqa: F401

from yeto.rl.engine.controller import COMMITTED, SUCCEEDED, IslandController
from yeto.rl.engine.journal import read_journal

PHASE_KEYS = {"tx_id", "txn_id", "request_id", "source", "target", "phase", "result", "t",
              "expected_epoch", "config_epoch", "reason"}
CELL_KEYS = {"cell_id", "role", "node", "gpu_uuid", "gpus", "state", "config", "epoch"}


def _wire(ctl, journal=True):
    got = []
    ctl.set_event_sink(lambda event, **f: got.append((event, f)), journal=journal)
    return got


def test_default_off_journal_unchanged(tmp_path):
    driver, ctl, *_ = _setup(tmp_path)
    _up_then_down(driver, ctl, _[1])
    kinds = {r["kind"] for r in read_journal(tmp_path / "state/reconfig")}
    assert not kinds & {"rl_cell_snapshot", "rl_reconfig_phase"}


def test_phase_and_snapshot_schema_and_order(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(tmp_path)
    got = _wire(ctl)
    _up_then_down(driver, ctl, pool)
    recs = read_journal(tmp_path / "state/reconfig")
    phases = [r for r in recs if r["kind"] == "phase" and r.get("tx_id")]
    evs = [f for e, f in got if e == "rl_reconfig_phase"]
    # one event per journaled phase transition, same order
    assert [(f["txn_id"], f["phase"]) for f in evs] == [(r["tx_id"], r["phase"]) for r in phases]
    for f in evs:
        assert set(f) == PHASE_KEYS
        assert f["result"] == (f["phase"] if f["phase"] in (COMMITTED, SUCCEEDED, "CANCELLED",
                                                           "REBUILT_OLD", "RECOVERY_REQUIRED") else None)
    up = [f for f in evs if f["request_id"] == "up"]
    assert up[0]["source"] == "T4R2S2" and up[0]["target"] == "T4R4S0" and up[0]["expected_epoch"] == 0
    assert COMMITTED in [f["phase"] for f in up] and up[-1]["phase"] == SUCCEEDED
    snaps = [(i, f) for i, (e, f) in enumerate(got) if e == "rl_cell_snapshot"]
    assert len(snaps) == 2  # one per transaction end
    for i, f in snaps:
        assert got[i - 1][0] == "rl_reconfig_phase" and got[i - 1][1]["phase"] == SUCCEEDED
        assert f["txn_id"] == got[i - 1][1]["txn_id"] and f["cause"] == "tx_end:SUCCEEDED"
        assert f["cells"] and all(set(c) == CELL_KEYS for c in f["cells"])
    assert len(snaps[0][1]["cells"]) == 4 and snaps[0][1]["cells"][0]["epoch"] == 1
    # journaled copies
    jk = [r["kind"] for r in recs if r["kind"].startswith("rl_")]
    assert jk.count("rl_reconfig_phase") == len(evs) and jk.count("rl_cell_snapshot") == 2


def test_replay_does_not_reemit(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(tmp_path)
    _wire(ctl)
    _up_then_down(driver, ctl, pool)
    ctl.close()
    driver.ledger.close()
    n = len(read_journal(tmp_path / "state/reconfig"))
    ctl2 = IslandController(state_dir=tmp_path / "state", configs=ctl.configs,
                            attestation=_attestation(), profile=_profile(),
                            initial_config="T4R2S2", runtime_fingerprint=FP)
    got = _wire(ctl2)
    assert got == []
    ctl2.close()
    after = read_journal(tmp_path / "state/reconfig")
    assert not [r for r in after[n:] if r["kind"].startswith("rl_")]


def test_sink_failure_never_fails_the_transaction(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(tmp_path)
    ctl.set_event_sink(lambda e, **f: 1 / 0)
    _up_then_down(driver, ctl, pool)
    assert ctl.status("down")["phase"] == SUCCEEDED
