"""0.26: a late island ends normally when the syncer finished (FINISHED or closed
connection after the final version) instead of BrokenPipe / exit 1. CPU, no Ray."""

from __future__ import annotations

import json
import socket
import threading

import pytest
import torch

from yeto.protocol import read_frame, write_frame
from yeto.rl.elastic_client import (
    MSG_FINISHED, DeltaTensor, ElasticBase, ElasticClientConfig, ElasticFinished, ElasticInit,
    ElasticIslandClient, Finished, Join, JoinAck, Leave, decode, encode,
)

from test_rl_engine_driver import _driver, _engine, _strict_config
from yeto.rl.engine.bridges import ElasticAvgSync

KEY = b"k"


class FinishingSyncer:
    """Other islands push the run to ``final`` with this island's first delta; the
    syncer then (optionally) sends FINISHED and goes away."""

    def __init__(self, *, final, send_finished):
        self.a, self.b = socket.socketpair()
        self.final, self.send_finished = final, send_finished
        self.version, self.base = 0, None
        self.leaves = 0
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
                self.version = self.final
                self.base = [b + u for b, u in zip(self.base, msg.update)]
                self._base()
                if self.send_finished:
                    write_frame(self.b, *encode(Finished(0, self.final, bytes([5]) * 32), KEY))
                    continue  # the grace window: the island answers with LEAVE
                self.b.close()
                return
            elif isinstance(msg, Leave):
                self.leaves += 1
                self.b.close()
                return

    def connect(self, addr):
        return self.a


def test_finished_frame_roundtrip():
    msg = Finished(3, 6, bytes([9]) * 32)
    t, payload = encode(msg, KEY)
    assert t == MSG_FINISHED == 25 and decode(KEY, t, payload) == msg
    assert len(payload) == 8 + 8 + 32 + 32


@pytest.mark.parametrize("send_finished", [True, False])
def test_late_island_ends_normally_when_the_syncer_finished(tmp_path, send_finished):
    fake = FinishingSyncer(final=3, send_finished=send_finished)
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = ElasticIslandClient(
        ElasticClientConfig(("x", 0), 0, lease_s=30.0, join_timeout_s=2, heartbeat_connection=False),
        KEY, connect=fake.connect)
    sync = ElasticAvgSync(_strict_config(tmp_path, engine, learner_id=0, rounds=3), client=client,
                          base_wait_s=5.0)
    _driver(engine, sync, tmp_path).run()  # no exception: a normal end (exit 0)
    tape = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    done = [e for e in tape if e["event"] == "elastic_finished"]
    # S17 M1: the island stops as soon as it applied the final outer version (here right after its
    # first delta), so it no longer pushes one more delta into the finished syncer; the FINISHED
    # path (elastic_finished) is then only taken when the push races the end (next test).
    assert sync.base_version == 3 and done == []
    assert sum(e["event"] == "rl_local_round" for e in tape) == 1
    if send_finished:  # LEAVE first so the syncer can exit before its grace window ends
        assert fake.leaves == 1
    client.close()


def test_disconnect_before_the_final_version_is_still_an_error():
    c = ElasticIslandClient(ElasticClientConfig(("x", 0), 0), KEY)
    c.final_outer_version = 6
    c.latest_base = ElasticBase(0, 4, (0.0,))
    c.disconnected = True
    c.raise_if_finished()  # not final yet: no ElasticFinished
    c.latest_base = ElasticBase(0, 6, (0.0,))
    with pytest.raises(ElasticFinished):
        c.raise_if_finished()


def test_push_racing_the_end_takes_the_finished_path(tmp_path):
    """The base the island holds is not final yet, but the syncer finished meanwhile:
    FINISHED arrives before the next push and the island ends via elastic_finished."""
    fake = FinishingSyncer(final=3, send_finished=True)
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = ElasticIslandClient(
        ElasticClientConfig(("x", 0), 0, lease_s=30.0, join_timeout_s=2, heartbeat_connection=False),
        KEY, connect=fake.connect)
    sync = ElasticAvgSync(_strict_config(tmp_path, engine, learner_id=0, rounds=4), client=client,
                          base_wait_s=5.0)
    _driver(engine, sync, tmp_path).run()
    tape = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    done = [e for e in tape if e["event"] == "elastic_finished"]
    assert len(done) == 1 and done[0]["final_outer_version"] == 3 and done[0]["reason"] == "FINISHED"
    assert fake.leaves == 1
    client.close()
