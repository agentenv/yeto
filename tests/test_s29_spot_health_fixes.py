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


# ---------------------------------------------------------------------------- D4: FENCED leave
def test_d4_fenced_is_a_new_leave_reason_and_an_excused_departure():
    from yeto.launch.tapes import EXCUSED_LEAVE_REASONS
    from yeto.rl.pool_client import LeaveReason

    # the codes 0..6 keep their bytes; FENCED is appended (an older syncer: unknown_7)
    assert [r.name for r in sorted(LeaveReason, key=int)][:7] == [
        "FINISHED", "LEASE_EXPIRED", "STOPPED", "PREEMPTED", "INFRA_FAILED", "TRAINING_FAILED", "CONFIG_MISMATCH"]
    assert int(LeaveReason.FENCED) == 7 and "fenced" in EXCUSED_LEAVE_REASONS


def _channels(*, regressed=False, fenced=False):
    from yeto.rl.spot_rollout.sampler import SpotChannels

    ch = SpotChannels.__new__(SpotChannels)
    ch.version = types.SimpleNamespace(regressed=regressed, merged_version=0, highest=42)
    ch.client = types.SimpleNamespace(fenced=lambda: fenced, run_finished=lambda: False)
    return ch


def test_d4_spot_channels_name_the_fence():
    from yeto.rl.pool_client import LeaveReason

    assert _channels().stop_reason() is None  # run finished / syncer lost: FINISHED as before
    reason, detail = _channels(regressed=True).stop_reason()
    assert reason is LeaveReason.FENCED and detail["fenced"] == "other run"
    assert (detail["run_merged_version"], detail["highest_merged_version"]) == (0, 42)
    assert _channels(fenced=True).stop_reason() == (LeaveReason.FENCED, {"fenced": "superseded"})


def test_d4_a_fenced_sampler_leaves_fenced_on_its_tape(tmp_path, monkeypatch, capsys):
    import json

    from tests.test_stage1_gaps import _sampler_spec
    from yeto.rl.engine import engine_sampler_entry as entry
    from yeto.rl.engine.sampler_resume import RECLAIM_MARKER_ENV
    from yeto.rl.pool_client import LeaveReason

    monkeypatch.setenv(RECLAIM_MARKER_ENV, str(tmp_path / "reclaim-marker"))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        seen = []
        entry.run(_sampler_spec(tmp_path, num_rollouts=4), resource_sampler=lambda emit: None,
                  incarnation="aaaa0001", stop=lambda: any(e == "rl_sampler_round" for e, _ in seen),
                  stop_reason=lambda: (LeaveReason.FENCED, {"fenced": "other run"}),
                  on_event=lambda event, **fields: seen.append((event, fields)))
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    [leave] = [f for e, f in seen if e == "rl_sampler_leave"]
    assert leave["reason"] == "FENCED" and leave["fenced"] == "other run"
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    records = [json.loads(line) for line in tape.read_text().splitlines()]
    assert [r for r in records if r["event"] == "rl_sampler_leave"][0]["reason"] == "FENCED"
    assert "leaving FENCED" in capsys.readouterr().out


# ---------------------------------------------------------------------------- D7: local steps
def test_d7_the_local_step_line_and_the_global_rounds_help():
    from yeto.rl.engine.run_flags import GLOBAL_ROUNDS_HELP, local_training_steps_line

    assert "= global_rounds x fragments = 40 x 8 = 320" in local_training_steps_line(40, 8)
    assert "sync rounds" in GLOBAL_ROUNDS_HELP


SPOT = ["--rl-sample-role", "trainer", "--rl-sync-preset", "full-parameter",
        "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4"]
RELAY = ["--rl-sample-relay", "--rl-spot-subscribers", "2"]
GPU3 = ["--gpu", "modal:1xh200,modal:1xh200,modal:1xh200"]


def test_d7_the_launcher_prints_the_local_training_steps(monkeypatch, capsys):
    from tests.test_s27_verl_sample_role import _launch

    _launch(monkeypatch, SPOT + RELAY + GPU3)  # --total-steps 4, 4 fragments
    assert "local training steps = global_rounds x fragments = 4 x 4 = 16" in capsys.readouterr().out


# ---------------------------------------------------------------------------- D11: spot pool default
def _spot(monkeypatch, extra, *, relay=RELAY):
    from tests.test_s27_verl_sample_role import _launch
    from yeto.launch.syncer import _syncer_island_pool, island_lease_s, island_pool

    args, islands = _launch(monkeypatch, SPOT + list(relay) + GPU3 + list(extra))
    pool = island_pool(args)
    return args, islands, pool, (island_lease_s(args) if pool else None), _syncer_island_pool(args)


def test_d11_a_spot_rollout_run_is_an_island_pool_by_default(monkeypatch):
    args, islands, pool, lease, flags = _spot(monkeypatch, [])
    assert pool and lease == 150.0 and " --island-lease-s 150.0" in flags
    assert args.quorum == 1  # the async mode's quorum: rounds merge without some island
    assert all(i.rl_sync_island_lease_s == 150.0 for i in islands)


def test_d11_strict_values_get_a_quorum_below_the_islands(monkeypatch):
    args, _, pool, lease, _ = _spot(monkeypatch, ["--rl-sync-mode", "strict",
                                                  "--rl-sync-full-parameter-rounds", "async"])
    assert pool and lease == 150.0 and args.quorum == 2  # 3 trainer islands - 1


def test_d11_explicit_options_win(monkeypatch):
    _, _, pool, lease, _ = _spot(monkeypatch, ["--rl-sync-island-lease-s", "60"])
    assert pool and lease == 60.0
    # an explicit strict quorum with no grace/base lag: the fixed roster stays (and the warning)
    args, _, pool, _, flags = _spot(monkeypatch, ["--rl-sync-mode", "strict", "--rl-sync-full-parameter-rounds",
                                                  "async", "--rl-sync-quorum", "3"])
    assert not pool and flags == "" and args.quorum == 3


def test_d11_the_fixed_roster_switch_and_non_spot_runs_keep_the_roster(monkeypatch):
    _, _, pool, _, flags = _spot(monkeypatch, ["--rl-spot-fixed-roster"])
    assert not pool and flags == ""
    _, _, pool, _, flags = _spot(monkeypatch, [], relay=())
    assert not pool and flags == ""  # not a spot rollout run: unchanged
