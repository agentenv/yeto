"""A trainer island tells the samplers that follow it where it reads while a take
waits, so groups it passed over never hold a sampler at its backlog cap.

A fleet GPU run (island 0, an external vLLM pool and an SGLang pool, both following
one trainer on an object store; release every 2, max policy age 4): the pools wrote
rounds at v0 while the trainer trained five rounds, so every group after the newest
release (v4, read position 16 of the vLLM shard and 4 of the SGLang shard) was v0.
The trainer, at v5, passed all of them over (older than 5 - 4) and waited for new
groups; each pool counted 24 groups after the release's read position (its cap) and
waited for the trainer to read them. The read position only moved with a release,
the next release needed a trained round, and the round needed new groups: both
sides waited until the sample-wait limit.
"""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.buffer_payload import encode_group
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import (
    READ_POSITION_KEY, FileSampleBuffer, read_buffer_read_position, shard_index, write_buffer_read_position)
from yeto.rl.engine.sampler_loop import SamplerLoop
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

VLLM, SGLANG = "vllm-i0-0", "sglang-i0-0"
CAP = 24  # max_backlog_groups of both pools
GROUPS = 4  # rollout batch size (groups per round)
AGE = 4
_clock = itertools.count(1)


class _StagedStore(CheckpointStore):
    """A staged object store as on a fleet island: a local stage directory of its
    own per host, the objects shared (only they reach another host)."""

    def __init__(self, root, objects):
        super().__init__(root)
        self.objects = objects


def _write(writer, shard, n, version, groups):
    for _ in range(groups):
        sid = f"{shard}-{n[shard]}"
        n[shard] += 1
        ref = writer.write_payload(sid, encode_group([{"index": n[shard], "shard": shard}]))
        writer.append(SampleIndexEntry(sid, version, ref, 8, float(next(_clock))))


def _release(objects, seq=3, policy_hash="h-v4"):
    objects.write("policy-releases/LATEST.json",
                  json.dumps({"seq": seq, "policy_hash": policy_hash, "version": 4}).encode())


def _pool(buffer, shards, *, seq=3, policy_hash="h-v4"):
    """A pool's sampler loop that loaded the release ``seq`` (its read positions ``shards``)."""
    seen = SimpleNamespace(seq=seq, policy_hash=policy_hash, buffer_shards=dict(shards), buffer_selected=None,
                           released_at=0.0)
    return SamplerLoop(follower=SimpleNamespace(last_seen=seen, version=4), buffer=buffer,
                       generate=lambda r: None, num_rollouts=None, max_backlog_groups=CAP,
                       emit=lambda *a, **k: None)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _fleet_island(tmp_path, *, heartbeat):
    objects = DirectoryObjectStore(tmp_path / "objects")
    _release(objects)
    trainer_store = _StagedStore(tmp_path / "trainer-stage", objects)
    writers = {s: FileSampleBuffer(_StagedStore(tmp_path / f"{s}-stage", objects), batch_size=1,
                                   index=shard_index(s)) for s in (VLLM, SGLANG)}
    n = dict.fromkeys(writers, 0)
    # trained before the release: 16 vLLM and 4 SGLang groups, all at v0
    _write(writers[VLLM], VLLM, n, 0, 16)
    _write(writers[SGLANG], SGLANG, n, 0, 4)
    buffer = FileSampleBuffer(trainer_store, batch_size=GROUPS, merge_shards=True, expected_shards=(VLLM, SGLANG))
    clock = _Clock()
    reader = br.BatchReader(buffer, max_policy_age=AGE, wait_s=600.0, poll_s=2.0, clock=clock,
                            sleep=lambda s: None, prefetch=False, read_workers=1, heartbeat_s=0,
                            read_heartbeat=br.trainer_read_heartbeat(buffer) if heartbeat else None)
    for rollout_id in range(5):  # rounds 0-4 (versions 0-4): read position 20
        reader.take(rollout_id, GROUPS, current_version=rollout_id)
    assert buffer.read_cursor == 20
    assert buffer.shard_positions(20) == {VLLM: 16, SGLANG: 4}
    # meanwhile both pools generated at v0 up to their cap after the release's position
    _write(writers[VLLM], VLLM, n, 0, CAP)
    _write(writers[SGLANG], SGLANG, n, 0, CAP)
    pools = {s: _pool(writers[s], {VLLM: 16, SGLANG: 4}) for s in writers}
    return SimpleNamespace(objects=objects, reader=reader, buffer=buffer, writers=writers, n=n, pools=pools,
                           clock=clock)


def _run_round_5(island):
    """The trainer takes round 5 at v5 while the pools generate whenever their backlog
    is below the cap (at v4, the newest release they loaded)."""
    admitted = []

    def sleep(seconds):
        island.clock.now += seconds
        for shard, pool in island.pools.items():
            if pool.backlog() < CAP:
                admitted.append(shard)
                _write(island.writers[shard], shard, island.n, 4, GROUPS)

    island.reader.sleep = sleep
    groups, report = island.reader.take(5, GROUPS, current_version=5)
    return groups, report, admitted


def test_passed_over_groups_no_longer_hold_the_pools_at_their_cap(tmp_path):
    island = _fleet_island(tmp_path, heartbeat=True)
    assert {s: pool.backlog() for s, pool in island.pools.items()} == {VLLM: CAP, SGLANG: CAP}
    groups, report, admitted = _run_round_5(island)
    assert len(groups) == GROUPS
    assert report.stale_groups == 2 * CAP  # every v0 group after the release is over age (5 - 4 = 1)
    assert admitted  # a pool generated again
    assert {g[0]["shard"] for _, g in groups} <= {VLLM, SGLANG}
    beat = json.loads(island.objects.read(READ_POSITION_KEY))
    assert beat["release"] == {"seq": 3, "policy_hash": "h-v4"}
    assert beat["buffer_shards"] == {VLLM: 16 + CAP, SGLANG: 4 + CAP}


def test_without_the_heartbeat_both_sides_wait(tmp_path):
    """The deadlock as it happened: nothing tells the pools that the trainer passed
    their groups over, so neither side moves until the sample-wait limit."""
    island = _fleet_island(tmp_path, heartbeat=False)
    with pytest.raises(br.SampleBufferRolloutError, match="without a new group"):
        _run_round_5(island)
    assert {s: pool.backlog() for s, pool in island.pools.items()} == {VLLM: CAP, SGLANG: CAP}


def test_the_heartbeat_is_written_only_while_a_take_waits_and_only_when_it_moved(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "objects")
    writer = FileSampleBuffer(_StagedStore(tmp_path / "w", objects), batch_size=1, index=shard_index(VLLM))
    buffer = FileSampleBuffer(_StagedStore(tmp_path / "t", objects), batch_size=2, merge_shards=True,
                              expected_shards=(VLLM,))
    beats = []
    clock = _Clock()
    n = {VLLM: 0}
    _write(writer, VLLM, n, 1, 2)
    reader = br.BatchReader(buffer, max_policy_age=AGE, wait_s=600.0, poll_s=2.0, clock=clock,
                            sleep=lambda s: None, prefetch=False, read_workers=1, heartbeat_s=0,
                            read_heartbeat=lambda cursor, shards: beats.append((cursor, shards)),
                            read_heartbeat_s=30.0)
    reader.take(0, 2, current_version=1)
    assert beats == []  # the take did not wait: the release carries the position

    def sleep(seconds):
        clock.now += seconds
        if clock.now >= 70.0 and n[VLLM] < 4:
            _write(writer, VLLM, n, 1, 2)

    reader.sleep = sleep
    reader.take(1, 2, current_version=1)
    # once at the first wait, then every read_heartbeat_s while the cursor stayed (2 s polls)
    assert beats == [(2, {VLLM: 2}), (2, {VLLM: 2}), (2, {VLLM: 2})]


def test_a_failing_heartbeat_is_reported_once_and_never_stops_the_take(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "objects")
    writer = FileSampleBuffer(_StagedStore(tmp_path / "w", objects), batch_size=1, index=shard_index(VLLM))
    buffer = FileSampleBuffer(_StagedStore(tmp_path / "t", objects), batch_size=1, merge_shards=True,
                              expected_shards=(VLLM,))
    clock, events, calls = _Clock(), [], []
    n = {VLLM: 0}

    def broken(cursor, shards):
        calls.append(clock.now)
        raise OSError("store down")

    def sleep(seconds):
        clock.now += seconds
        if clock.now >= 100.0 and not n[VLLM]:
            _write(writer, VLLM, n, 1, 1)

    reader = br.BatchReader(buffer, max_policy_age=AGE, wait_s=600.0, poll_s=2.0, clock=clock, sleep=sleep,
                            prefetch=False, read_workers=1, heartbeat_s=0, read_heartbeat=broken,
                            read_heartbeat_s=30.0)
    reader.emit = lambda event, **fields: events.append((event, fields))
    groups, _ = reader.take(0, 1, current_version=1)
    assert len(groups) == 1
    assert calls == [0.0, 30.0, 60.0, 90.0]  # retried every read_heartbeat_s, not every poll
    failed = [f for e, f in events if e == br.READ_HEARTBEAT_FAILED_EVENT]
    assert len(failed) == 1 and "store down" in failed[0]["error"]


def test_heartbeat_round_trip_on_an_object_store_and_a_directory(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "objects")
    on_objects = SimpleNamespace(store=_StagedStore(tmp_path / "stage", objects))
    assert read_buffer_read_position(on_objects) is None
    write_buffer_read_position(on_objects, buffer_selected=9, buffer_shards={"a": 5}, train_step=None,
                               release={"seq": 2, "policy_hash": "x"}, clock=lambda: 7.0)
    assert not (tmp_path / "stage" / "sample-buffer").exists()  # the object, not the stage
    expected = {"buffer_selected": 9, "buffer_shards": {"a": 5}, "train_step": None, "time_unix": 7.0,
                "release": {"seq": 2, "policy_hash": "x"}}
    assert read_buffer_read_position(on_objects) == expected
    # another host's stage of the same objects sees it
    assert read_buffer_read_position(SimpleNamespace(store=_StagedStore(tmp_path / "other", objects))) == expected
    on_dir = SimpleNamespace(store=CheckpointStore(tmp_path / "dir"))
    (tmp_path / "dir" / "sample-buffer").mkdir(parents=True)
    write_buffer_read_position(on_dir, buffer_selected=3, buffer_shards=None, train_step=None, release=None,
                               clock=lambda: 1.0)
    assert read_buffer_read_position(on_dir)["buffer_selected"] == 3
    objects.write(READ_POSITION_KEY, b"{broken")
    assert read_buffer_read_position(on_objects) is None


def test_newest_release_identity(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "objects")
    buffer = SimpleNamespace(store=_StagedStore(tmp_path / "stage", objects))
    assert br.newest_release_identity(buffer) is None
    _release(objects, seq=5, policy_hash="abc")
    assert br.newest_release_identity(buffer) == {"seq": 5, "policy_hash": "abc"}
    local = CheckpointStore(tmp_path / "dir")
    (tmp_path / "dir" / "policy-releases").mkdir(parents=True)
    (tmp_path / "dir" / "policy-releases" / "LATEST.json").write_text(json.dumps({"seq": 1, "policy_hash": "p"}))
    assert br.newest_release_identity(SimpleNamespace(store=local)) == {"seq": 1, "policy_hash": "p"}
