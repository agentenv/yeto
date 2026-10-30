"""S29 spot elastic health test: regression tests for the fixes (docs/s29-elastic-defects.md)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import types

import pytest


class _Follower:
    def __init__(self, versions):
        self._v = tuple(versions)

    def drain(self):
        return 0

    @property
    def versions(self):
        return self._v


def _store(versions, base=0):
    from yeto.rl.spot_rollout.release_store import SubscriberReleaseStore

    return SubscriberReleaseStore(_Follower(versions), lambda cut: "/tmp/x", layout_fingerprint="f",
                                  base_version=base)


def test_d9_past_base_only_once_a_fragment_is_newer_than_the_base():
    assert not _store([-1] * 8).past_base()
    assert not _store([0] * 8).past_base()
    assert _store([0, 3, -1, -1, -1, -1, -1, -1]).past_base()


def test_d5_relay_backlog_is_the_groups_left_in_the_channel():
    from yeto.rl.spot_rollout.relay_buffer import RelaySampleBuffer

    buf = RelaySampleBuffer.__new__(RelaySampleBuffer)
    buf._depth = None
    assert buf.pending_groups() == 0
    buf._depth = types.SimpleNamespace(next_id=5, items=1)
    assert len(buf) == 4 and buf.pending_groups() == 1


def test_d3_whole_round_mini_batch_restores_the_config():
    pytest.importorskip("omegaconf")
    pytest.importorskip("ray")
    pytest.importorskip("verl")  # fully_async_runner imports verl's fully_async trainer
    from omegaconf import OmegaConf

    from yeto.rl.adapters.verl.fully_async_runner import whole_round_mini_batch

    cfg = OmegaConf.create({"actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 1}, "rollout": {"n": 4}}})
    with whole_round_mini_batch(cfg, 3):
        assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 3 and cfg.actor_rollout_ref.rollout.n == 1
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 1 and cfg.actor_rollout_ref.rollout.n == 4
    with whole_round_mini_batch(cfg, 4):  # a full round is left alone
        assert cfg.actor_rollout_ref.rollout.n == 4


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_d8_a_killed_islands_server_group_is_reaped(tmp_path, monkeypatch):
    from yeto.rl.adapters.sampler_engines import server

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    server.server_group_file("sglang", 31999).write_text(str(child.pid))
    assert server.reap_stale_server("sglang", 31999) == child.pid
    assert child.wait(timeout=10) == -signal.SIGKILL
    assert not server.server_group_file("sglang", 31999).exists()
    assert server.reap_stale_server("sglang", 31999) is None  # nothing left to reap


def test_d10_the_relay_pool_never_caps_discarded_pulls():
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncRolloutPool

    island = types.SimpleNamespace(required=1, seek_rollouter=None)
    assert FullyAsyncRolloutPool(island, max_pulls_factor=0).max_pulls is None
    assert FullyAsyncRolloutPool(island).max_pulls == 8


class _Board:
    """An in-memory relay board (ids restart when ``fresh`` is called, like a new syncer)."""

    def __init__(self):
        self.items, self.next_id = [], 1

    def set(self, data):
        self.items = [types.SimpleNamespace(item_id=self.next_id, producer=0, data=data)]
        self.next_id += 1
        return self.next_id - 1

    def get(self, newer_than=0, wait_s=0.0):
        item = self.items[-1] if self.items else None
        return item if item is not None and item.item_id > newer_than else None

    def close(self):
        pass

    def fresh(self):
        self.items, self.next_id = [], 1


def test_d9_version_board_holds_a_sampler_past_the_age_limit():
    from yeto.rl.spot_rollout.version_board import VersionPublisher, VersionReader

    board = _Board()
    pub = VersionPublisher(lambda: board, max_policy_age=4)
    reader = VersionReader(lambda: board, poll_s=0.0)
    assert not reader.behind(0)  # nothing published: no opinion
    assert pub.publish(3) and not pub.publish(2)  # never lower
    assert not reader.behind(0)  # 3 - 0 <= 4
    pub.publish(9)
    assert reader.behind(0) and reader.behind(None) and not reader.behind(5)


def test_d9_release_store_behind_uses_the_board_then_the_fragments():
    from yeto.rl.spot_rollout.version_board import VersionPublisher, VersionReader

    store = _store([-1] * 8)
    assert not store.behind(None)  # no board, no fragment: nothing known
    board = _Board()
    store.version_reader = VersionReader(lambda: board, poll_s=0.0)
    VersionPublisher(lambda: board, max_policy_age=4).publish(41)
    assert store.behind(0) and store.behind(None) and not store.behind(40)
    # no board: per-fragment versions alone never hold (the local deadlock of the first fix)
    assert not _store([0, 3, -1, -1, -1, -1, -1, -1]).behind(0)


def test_d4_a_fresh_syncer_fences_the_sampler():
    from yeto.rl.spot_rollout.version_board import VersionPublisher, VersionReader

    board = _Board()
    reader = VersionReader(lambda: board, poll_s=0.0)
    VersionPublisher(lambda: board, 4).publish(42)
    reader.refresh()
    assert not reader.regressed
    board.fresh()  # a syncer started without --resume
    reader.close()  # the reconnect that follows it
    VersionPublisher(lambda: board, 4).publish(1)
    reader.refresh()
    assert reader.regressed


def test_d10_a_round_of_stale_groups_is_discarded_and_the_round_waits():
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncRolloutPool

    pulls = iter([("stale", i) for i in range(40)] + [("fresh", 0)])
    kept = []

    class Collector:
        @property
        def full(self):
            return bool(kept)

        def offer(self, meta):
            if meta[0] == "fresh":
                kept.append(meta)

    island = types.SimpleNamespace(required=1, seek_rollouter=None)
    pool = FullyAsyncRolloutPool(island, max_pulls_factor=0)
    pulled = 0
    collector = Collector()
    while not collector.full:  # the pool's loop: with no cap it never raises on stale pulls
        if pool.max_pulls is not None and pulled >= pool.max_pulls:
            raise AssertionError("capped")
        collector.offer(next(pulls))
        pulled += 1
    assert pulled == 41 and kept == [("fresh", 0)]
