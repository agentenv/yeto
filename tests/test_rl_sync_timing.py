"""rl-trainer-start-stop L5.6: island-side sync timing. A fake island-pool syncer
delays its next base; the island must report that delay as
``sync/sync_wait_seconds``. CPU, no Ray."""

from __future__ import annotations

import json
import time

import pytest
import torch

from pool_fakes import FakePoolSyncer
from yeto.rl.engine.bridges import DecoupledSync, PoolAvgSync
from yeto.rl.engine.sync_timing import SyncTiming

from test_rl_engine_driver import _driver, _engine, _strict_config

DELAY_S = 0.4


def test_sync_timing_split():
    t = SyncTiming(submit_ts=100.0, global_ready_ts=103.0, recv_seconds=0.5)
    assert t.sync_wait_seconds == 2.5
    assert SyncTiming(100.0, 103.0).sync_wait_seconds == 3.0  # receive time not separated
    assert SyncTiming(100.0, 99.0).sync_wait_seconds == 0.0  # base was there before the submit
    assert SyncTiming(100.0, None).fields() == {
        "sync/submit_ts": 100.0, "sync/global_ready_ts": None, "sync/recv_seconds": None,
        "sync/sync_wait_seconds": None}


def test_pool_island_measures_the_delayed_reply(tmp_path):
    fake = FakePoolSyncer(base_delay_s=DELAY_S)
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = fake.client()
    sync = PoolAvgSync(_strict_config(tmp_path, engine, learner_id=0, rounds=2), client=client,
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
