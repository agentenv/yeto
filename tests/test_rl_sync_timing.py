"""rl-trainer-start-stop L5.6: island-side sync timing. A fake elastic syncer
delays its ELASTIC_BASE reply; the island must report that delay as
``sync/sync_wait_seconds``. CPU, no Ray."""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
import torch

from yeto.protocol import read_frame, write_frame
from yeto.rl.elastic_client import (
    DeltaTensor, ElasticBase, ElasticClientConfig, ElasticInit, ElasticIslandClient, Join, JoinAck,
    Leave, decode, encode,
)
from yeto.rl.engine.bridges import DecoupledSync, ElasticAvgSync
from yeto.rl.engine.sync_timing import SyncTiming

from test_rl_engine_driver import _driver, _engine, _strict_config

KEY = b"k"
DELAY_S = 0.4


class DelayingSyncer:
    """Answers each DELTA_TENSOR with base v+1 after ``delay_s``."""

    def __init__(self, delay_s: float):
        self.a, self.b = socket.socketpair()
        self.delay_s = delay_s
        self.version, self.base = 0, None
        threading.Thread(target=self._serve, daemon=True).start()

    def _base(self):
        write_frame(self.b, *encode(ElasticBase(0, self.version, tuple(self.base)), KEY))

    def _serve(self):
        while True:
            try:
                t, p = read_frame(self.b, 1 << 20)
            except Exception:
                return
            msg = decode(KEY, t, p)
            if isinstance(msg, Join):
                write_frame(self.b, *encode(JoinAck(0, msg.island_id, 1, 0, bytes(32), False), KEY))
            elif isinstance(msg, ElasticInit):
                self.base = list(msg.params)
                self._base()
            elif isinstance(msg, DeltaTensor):
                time.sleep(self.delay_s)  # the other islands / the merge
                self.version += 1
                self.base = [b + u for b, u in zip(self.base, msg.update)]
                self._base()
            elif isinstance(msg, Leave):
                self.b.close()
                return

    def connect(self, addr):
        return self.a


def test_sync_timing_split():
    t = SyncTiming(submit_ts=100.0, global_ready_ts=103.0, recv_seconds=0.5)
    assert t.sync_wait_seconds == 2.5
    assert SyncTiming(100.0, 103.0).sync_wait_seconds == 3.0  # receive time not separated
    assert SyncTiming(100.0, 99.0).sync_wait_seconds == 0.0  # base was there before the submit
    assert SyncTiming(100.0, None).fields() == {
        "sync/submit_ts": 100.0, "sync/global_ready_ts": None, "sync/recv_seconds": None,
        "sync/sync_wait_seconds": None}


def test_elastic_island_measures_the_delayed_reply(tmp_path):
    fake = DelayingSyncer(DELAY_S)
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = ElasticIslandClient(
        ElasticClientConfig(("x", 0), 0, lease_s=30.0, join_timeout_s=2, heartbeat_connection=False),
        KEY, connect=fake.connect)
    sync = ElasticAvgSync(_strict_config(tmp_path, engine, learner_id=0, rounds=2), client=client,
                          base_wait_s=5.0)
    _driver(engine, sync, tmp_path).run()
    client.close()
    tape = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    timing = [e for e in tape if e["event"] == "rl_sync_timing"]
    assert [e["rollout_id"] for e in timing] == [0, 1]
    for e in timing:
        assert e["outer_sync"] == "elastic"
        assert e["sync/global_ready_ts"] >= e["sync/submit_ts"]
        assert 0.0 <= e["sync/recv_seconds"] < 0.2
        assert e["sync/sync_wait_seconds"] == pytest.approx(DELAY_S, abs=0.15)


def test_decoupled_remote_wait_is_the_blocked_time():
    sync = DecoupledSync(args=None)
    assert sync._remote_wait(time.sleep, 0.2) is None
    sync._remote_wait(lambda: None)
    assert 0.2 <= sync._remote_wait_seconds < 0.4
