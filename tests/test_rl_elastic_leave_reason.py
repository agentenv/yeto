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


# --- review follow-ups -------------------------------------------------------

def test_a_wrapped_profile_error_is_a_config_mismatch():
    from yeto.rl.engine.execution_profile import ProfileError

    try:
        try:
            raise ProfileError("eval overlap needs a fixed-partition placement")
        except ProfileError as error:
            raise DriverError(f"execution profile rejected: {error}") from error
    except DriverError as wrapped:
        assert leave_reason_for(wrapped) is LeaveReason.CONFIG_MISMATCH
    plain = DriverError("run reached max_rollouts without a stop")
    assert leave_reason_for(plain) is LeaveReason.TRAINING_FAILED


def test_the_miles_publication_error_is_a_training_failure():
    from yeto.rl.adapters.miles.publish import PublicationError as MilesPublicationError

    assert leave_reason_for(MilesPublicationError("read-back differs")) is LeaveReason.TRAINING_FAILED


def test_the_bridge_gives_the_client_the_syncer_lease(tmp_path):
    """The island heartbeats every lease/3 of the syncer's --island-lease-s, not
    the client's own default."""
    engine = _engine(torch.tensor([1.0, 3.0]))
    config = _strict_config(tmp_path, engine, learner_id=0, rounds=2)
    assert ElasticAvgSync(config, lease_s=45.0)._client_config().lease_s == 45.0
    from yeto.rl.elastic_client import DEFAULT_ISLAND_LEASE_S

    assert ElasticAvgSync(config)._client_config().lease_s == DEFAULT_ISLAND_LEASE_S


def test_the_launcher_passes_the_lease_to_every_elastic_island(monkeypatch):
    from test_rl_launcher import _args
    from yeto import launcher
    from yeto.rl.adapters.miles.island_entry import parse_args

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    flags = launcher._ports_infra_flags(_args(("--rl-island-scheduling", "elastic",
                                                "--rl-island-lease-s", "90")))[1]
    assert " --rl-island-lease-s 90.0" in flags
    default = launcher._ports_infra_flags(_args(("--rl-island-scheduling", "elastic")))[1]
    assert f" --rl-island-lease-s {launcher.DEFAULT_ISLAND_LEASE_S!r}" in default
    required = ["--model", "m", "--model-revision", "r", "--data", "d", "--learner-id", "0",
                "--reward-function", "f", "--reward-sha256", "s", "--source-sha256", "s",
                "--global-rounds", "1", "--groups-per-round", "1", "--samples-per-group", "1",
                "--over-sampling-batch-size", "1", "--optimizer-steps", "1",
                "--rollout-max-response-len", "1", "--completed-groups-path", "c",
                "--event-tape", "t", "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1",
                "--inner-lr", "1e-5", "--seq-len", "1", "--seed", "0", "--miles-root", "m",
                "--lora-r", "8", "--lora-targets", "attention", "--rl-single-island-no-sync"]
    assert parse_args(required + ["--rl-island-lease-s", "90"]).rl_island_lease_s == 90.0


class _RefusingSyncer(MultiConnSyncer):
    """Answers every JOIN with MSG_ERROR, like elastic_server.rs on a compat_group
    or backend identity mismatch."""

    def _serve(self, sock, idx):
        from yeto.protocol import MSG_ERROR, read_frame, write_frame

        read_frame(sock, 1 << 20)
        write_frame(sock, MSG_ERROR, b"compat_group mismatch, JOIN refused")


def test_a_refused_first_join_raises_elastic_join_refused():
    c = _client(_RefusingSyncer(), [])
    with pytest.raises(ElasticJoinRefused, match="compat_group mismatch"):
        c.join()
    c.close()


def test_concurrent_leaves_send_one_frame(tmp_path):
    import threading

    fake = MultiConnSyncer()
    engine, sync, client = _elastic(tmp_path, fake)
    client.join()
    barrier = threading.Barrier(8)
    results = []

    def leave(reason):
        barrier.wait()
        results.append(sync.leave(reason))
    threads = [threading.Thread(target=leave, args=(r,))
               for r in [LeaveReason.PREEMPTED, LeaveReason.INFRA_FAILED] * 4]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1 and len(_leaves(fake)) == 1
    sync.close()
