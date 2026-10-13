"""rl-island-graceful-leave: LEAVE reason codes (design D7).

Parity with the Rust syncer, the exception -> code table, and the driver
closing an elastic bridge with the code of its departure."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from test_rl_elastic_lease_rejoin import MultiConnSyncer, _client, _wait
from test_rl_engine_driver import _driver, _engine, _strict_config
from yeto.rl.elastic_client import ElasticJoinRefused, ElasticRejoinError, Leave, LeaveReason
from yeto.rl.engine.bridges import ElasticAvgSync
from yeto.rl.engine.capabilities import CapabilityMismatch
from yeto.rl.engine.controller import RecoveryRequired, TransactionFailed
from yeto.rl.engine.cut import CutError
from yeto.rl.engine.driver import (
    DriverError, IslandRecoveryRequired, PolicyIdentityError, PublicationError, RebuildNotStarted,
    RoundFailedError,
)
from yeto.rl.engine.journal import EpochConflict
from yeto.rl.engine.leave_reason import leave_reason_for, sendable

REPO = Path(__file__).resolve().parents[1]


def test_reason_codes_match_the_rust_syncer():
    src = (REPO / "syncer/src/elastic.rs").read_text()
    rust = {m[0]: int(m[1]) for m in re.findall(r"pub const LEAVE_REASON_(\w+): u8 = (\d+);", src)}
    assert rust == {r.name: r.value for r in LeaveReason}


@pytest.mark.parametrize("error, reason", [
    (None, LeaveReason.FINISHED),
    (KeyboardInterrupt(), LeaveReason.STOPPED),
    (SystemExit(1), LeaveReason.STOPPED),
    (IslandRecoveryRequired("island is RECOVERY_REQUIRED: node lost"), LeaveReason.INFRA_FAILED),
    (RecoveryRequired("x"), LeaveReason.INFRA_FAILED),
    (RebuildNotStarted("x"), LeaveReason.INFRA_FAILED),
    (TransactionFailed.__new__(TransactionFailed), LeaveReason.INFRA_FAILED),
    (ElasticRejoinError("x"), LeaveReason.INFRA_FAILED),
    (CutError("x"), LeaveReason.INFRA_FAILED),
    (EpochConflict("x"), LeaveReason.INFRA_FAILED),
    (RuntimeError("CUDA out of memory"), LeaveReason.INFRA_FAILED),  # unmapped: D7 Q2
    (DriverError("x"), LeaveReason.TRAINING_FAILED),
    (PublicationError("x"), LeaveReason.TRAINING_FAILED),
    (PolicyIdentityError("x"), LeaveReason.TRAINING_FAILED),
    (RoundFailedError("x"), LeaveReason.TRAINING_FAILED),
    (CapabilityMismatch("x"), LeaveReason.CONFIG_MISMATCH),
    (ElasticJoinRefused("x"), LeaveReason.CONFIG_MISMATCH),
])
def test_exception_to_reason(error, reason):
    assert leave_reason_for(error) is reason


def test_strict_invariant_is_a_training_failure():
    from yeto.rl.core import StrictRlInvariantError

    err = StrictRlInvariantError.__new__(StrictRlInvariantError)
    assert leave_reason_for(err) is LeaveReason.TRAINING_FAILED


def test_an_island_never_sends_lease_expired_or_unknown_codes():
    assert sendable(LeaveReason.LEASE_EXPIRED) is LeaveReason.INFRA_FAILED
    assert sendable(99) is LeaveReason.INFRA_FAILED
    assert sendable(LeaveReason.STOPPED) is LeaveReason.STOPPED


def _elastic(tmp_path, fake, *, rounds=2):
    """The fake syncer cannot decode LEAVE (island-side decoder), so the
    client's sent LEAVE frames are recorded on ``fake.leaves`` instead."""
    engine = _engine(torch.tensor([1.0, 3.0]))
    c = _client(fake, [])
    fake.leaves = []
    send = c._send

    def recording_send(msg):
        send(msg)
        if isinstance(msg, Leave):
            fake.leaves.append(msg.reason)
    c._send = recording_send
    sync = ElasticAvgSync(_strict_config(tmp_path, engine, learner_id=0, rounds=rounds), client=c,
                          base_wait_s=5.0)
    return engine, sync, c


def _leaves(fake):
    return list(fake.leaves)


def _tape(tmp_path):
    return [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]


def test_a_finished_island_leaves_with_finished(tmp_path):
    fake = MultiConnSyncer()
    engine, sync, _ = _elastic(tmp_path, fake)
    _driver(engine, sync, tmp_path).run()
    assert _leaves(fake) == [LeaveReason.FINISHED]
    leave = [e for e in _tape(tmp_path) if e["event"] == "island_leave"]
    assert leave[-1]["reason_name"] == "FINISHED" and leave[-1]["error_type"] is None


@pytest.mark.parametrize("error, reason", [
    (IslandRecoveryRequired("island is RECOVERY_REQUIRED: node lost"), LeaveReason.INFRA_FAILED),
    (RoundFailedError("optimizer step failed"), LeaveReason.TRAINING_FAILED),
    (CapabilityMismatch("peer lacks x"), LeaveReason.CONFIG_MISMATCH),
    (KeyboardInterrupt(), LeaveReason.STOPPED),
    (RuntimeError("ray actor died"), LeaveReason.INFRA_FAILED),
])
def test_a_failing_island_leaves_with_the_reason_of_its_error(tmp_path, error, reason):
    fake = MultiConnSyncer()
    engine, sync, _ = _elastic(tmp_path, fake)

    def train_step(batch):
        raise error
    engine.trainer.train_step = train_step
    with pytest.raises(type(error)):
        _driver(engine, sync, tmp_path).run()
    assert _leaves(fake) == [reason]
    leave = [e for e in _tape(tmp_path) if e["event"] == "island_leave"][-1]
    assert (leave["reason"], leave["error_type"]) == (int(reason), type(error).__name__)


def test_a_failed_leave_is_reported_and_the_original_error_still_propagates(tmp_path):
    fake = MultiConnSyncer()
    engine, sync, client = _elastic(tmp_path, fake)

    def train_step(batch):
        def broken(reason=0):
            raise BrokenPipeError("syncer gone")
        client.leave = broken
        raise RoundFailedError("optimizer step failed")
    engine.trainer.train_step = train_step
    with pytest.raises(RoundFailedError):
        _driver(engine, sync, tmp_path).run()
    failed = [e for e in _tape(tmp_path) if e["event"] == "elastic_leave_failed"]
    assert len(failed) == 1
    assert (failed[0]["reason_name"], failed[0]["error_type"]) == ("TRAINING_FAILED", "BrokenPipeError")


def test_a_reclaim_leave_is_sent_once_and_not_overwritten_by_close(tmp_path):
    fake = MultiConnSyncer()
    engine, sync, _ = _elastic(tmp_path, fake)

    def train_step(batch):
        assert sync.leave(LeaveReason.PREEMPTED)  # reclaim listener thread in production
        raise RuntimeError("killed after the notice")
    engine.trainer.train_step = train_step
    with pytest.raises(RuntimeError):
        _driver(engine, sync, tmp_path).run()
    assert _leaves(fake) == [LeaveReason.PREEMPTED]
