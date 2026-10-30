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
