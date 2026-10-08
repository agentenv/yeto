"""0.22: lease renewal on its own connection, automatic re-JOIN, give-up after N. CPU, no Ray."""

from __future__ import annotations

import socket
import threading
import time

import pytest

from yeto.protocol import MSG_ERROR, read_frame, write_frame
from yeto.rl.elastic_client import (
    ELASTIC_REJOIN_FAILED_EXIT, DeltaTensor, ElasticBase, ElasticClientConfig, ElasticIslandClient,
    ElasticInit, ElasticRejoinError, Join, JoinAck, LeaseHeartbeat, decode, encode,
)

KEY = b"k"


class MultiConnSyncer:
    """Fake elastic server with one socketpair per connect(): membership by island id,
    heartbeats from a non-member answered like elastic.rs (MSG_ERROR ... rejoin required)."""

    def __init__(self, *, refuse_rejoin=False):
        self.members: set[int] = set()
        self.version, self.base = 0, [0.0, 0.0]
        self.frames: list[tuple[int, object]] = []  # (connection index, message)
        self.refuse_rejoin = refuse_rejoin
        self.joins = 0
        self.conns = []

    def connect(self, addr):
        a, b = socket.socketpair()
        idx = len(self.conns)
        self.conns.append(b)
        threading.Thread(target=self._serve, args=(b, idx), daemon=True).start()
        return a

    def _base(self, sock):
        write_frame(sock, *encode(ElasticBase(0, self.version, tuple(self.base)), KEY))

    def _serve(self, sock, idx):
        while True:
            try:
                t, p = read_frame(sock, 1 << 20)
            except Exception:
                return
            msg = decode(KEY, t, p)
            self.frames.append((idx, msg))
            if isinstance(msg, Join):
                self.joins += 1
                if self.joins > 1 and self.refuse_rejoin:
                    write_frame(sock, MSG_ERROR, b"join refused (test)")
                    continue
                self.members.add(msg.island_id)
                write_frame(sock, *encode(JoinAck(0, msg.island_id, self.joins, self.version,
                                                  bytes(32), self.joins > 1), KEY))
                self._base(sock)
            elif isinstance(msg, LeaseHeartbeat) and msg.island_id not in self.members:
                write_frame(sock, MSG_ERROR, f"{msg.island_id} is not a member (rejoin required)".encode())
            elif isinstance(msg, ElasticInit):
                self._base(sock)
            elif isinstance(msg, DeltaTensor):
                if msg.island_id not in self.members:
                    write_frame(sock, MSG_ERROR, f"{msg.island_id} is not a member".encode())
                    continue
                self.base = [b + u for b, u in zip(self.base, msg.update)]
                self.version += 1
                self._base(sock)

    def expire(self, island_id):
        self.members.discard(island_id)


def _client(fake, events, **kw):
    cfg = ElasticClientConfig(("x", 0), 0, lease_s=0.15, join_timeout_s=1.0, **kw)
    return ElasticIslandClient(cfg, KEY, connect=fake.connect, on_event=events.append)


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_heartbeats_use_their_own_connection_while_main_send_is_blocked():
    fake, events = MultiConnSyncer(), []
    c = _client(fake, events)
    c.join()
    assert c.hb_sock is not None and len(fake.conns) == 2
    with c._send_lock:  # a multi-MB DELTA_TENSOR in flight on the main connection
        n0 = sum(isinstance(m, LeaseHeartbeat) for _, m in fake.frames)
        assert _wait(lambda: sum(isinstance(m, LeaseHeartbeat) for _, m in fake.frames) >= n0 + 3)
    assert {i for i, m in fake.frames if isinstance(m, LeaseHeartbeat)} == {1}
    c.close()


def test_not_a_member_triggers_catch_up_rejoin_and_wait_returns_post_rejoin_base():
    fake, events = MultiConnSyncer(), []
    c = _client(fake, events)
    c.join()
    first = c.wait_base(newer_than=None, timeout_s=1.0)
    fake.expire(0)  # lease expired on the syncer while the island trained
    c.delta_tensor(base_version=first.outer_version, c_tokens=1, c_steps=1, update=[1.0, 1.0])
    base = c.wait_base(newer_than=first.outer_version, timeout_s=3.0)  # delta refused: no newer base
    assert base is not None and base.outer_version == first.outer_version
    rejoin = [e for e in events if e["event"] == "elastic_rejoin"]
    assert rejoin and rejoin[0]["incarnation"] == 1 and rejoin[0]["catch_up"] is True
    assert c.rejoins == 1 and c.ack.catch_up and 0 in fake.members
    joins = [m for _, m in fake.frames if isinstance(m, Join)]
    assert [j.incarnation for j in joins] == [0, 1]
    c.close()


def test_gives_up_after_n_consecutive_rejoin_failures():
    fake, events = MultiConnSyncer(refuse_rejoin=True), []
    c = _client(fake, events, max_rejoin_failures=3)
    c.config.join_timeout_s = 0.2
    c.join()
    first = c.wait_base(newer_than=None, timeout_s=1.0)
    fake.expire(0)
    with pytest.raises(ElasticRejoinError, match="3 consecutive re-JOINs failed"):
        c.wait_base(newer_than=first.outer_version, timeout_s=10.0)
    assert [e["failures"] for e in events if e["event"] == "elastic_rejoin_failed"] == [1, 2, 3]
    assert ELASTIC_REJOIN_FAILED_EXIT not in (4, 6)
    c.close()


def test_learner_maps_rejoin_exhaustion_to_its_own_exit_code():
    from yeto.rl.learner import _rejoin_exhausted

    try:
        try:
            raise ElasticRejoinError("x")
        except ElasticRejoinError as inner:
            raise RuntimeError("driver failed") from inner
    except RuntimeError as outer:
        assert _rejoin_exhausted(outer)
    assert not _rejoin_exhausted(RuntimeError("other"))


def test_launcher_island_lease_default_and_flag(monkeypatch):
    from test_rl_launcher import _args
    from yeto import launcher

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    cmd = launcher.syncer_command(_args(("--rl-island-scheduling", "elastic")), 2)
    assert " --island-lease-s 300.0" in cmd
    cmd = launcher.syncer_command(_args(("--rl-island-scheduling", "elastic", "--rl-island-lease-s", "450")), 2)
    assert " --island-lease-s 450.0" in cmd
    assert "--island-lease-s" not in launcher.syncer_command(_args(), 2)
    with pytest.raises(ValueError, match="--rl-island-lease-s"):
        launcher.syncer_command(_args(("--rl-island-lease-s", "450")), 2)  # legacy refuses it


def test_rejoin_events_reach_the_island_tape(tmp_path):
    import json

    import torch

    from test_rl_engine_driver import _driver, _engine, _strict_config
    from yeto.rl.engine.bridges import ElasticAvgSync

    fake, events = MultiConnSyncer(), []
    c = _client(fake, events)
    engine = _engine(torch.tensor([1.0, 3.0]))
    sync = ElasticAvgSync(_strict_config(tmp_path, engine, learner_id=0, rounds=3), client=c,
                          base_wait_s=5.0)
    orig = engine.trainer.train_step

    def train_step(batch):  # lease expires while round 1 trains
        if fake.version == 1 and 0 in fake.members:
            fake.expire(0)
            _wait(lambda: c.rejoins >= 1)
        return orig(batch)
    engine.trainer.train_step = train_step
    _driver(engine, sync, tmp_path).run()
    tape = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["incarnation"] for e in tape if e["event"] == "elastic_rejoin"] == [1]
    c.close()
