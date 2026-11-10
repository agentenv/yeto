"""One trainer reading several sampler shards (``--rl-buffer-shards a,b,c``):
the merge interleaves the shards by arrival, a stopped shard never blocks the
reader, a lagging shard's over-age groups are dropped (and still advance its
read position), and a shard the trainer was not launched with is never merged."""

from __future__ import annotations

import itertools

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.buffer_payload import encode_group
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

_clock = itertools.count(1)


def _write(writer, shard, n, version, groups):
    for _ in range(groups):
        sid = f"{shard}-{n[shard]}"
        n[shard] += 1
        ref = writer.write_payload(sid, encode_group([{"index": n[shard], "shard": shard}]))
        writer.append(SampleIndexEntry(sid, version, ref, 8, float(next(_clock))))


def _reader(objects, shards, age=2):
    buffer = FileSampleBuffer(objects, batch_size=1, merge_shards=True, expected_shards=shards)
    return br.BatchReader(buffer, max_policy_age=age, wait_s=0.5, poll_s=0.01,
                          sleep=lambda _s: None, prefetch=False, read_workers=1)


def test_three_shards_one_stopped_one_lagging_keep_feeding_the_trainer(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    shards = ("a", "b", "c")
    writers = {s: FileSampleBuffer(objects, batch_size=1, index=shard_index(s)) for s in shards}
    n = dict.fromkeys(shards, 0)
    reader = _reader(objects, shards)
    took = {s: 0 for s in shards}
    stale_total = 0
    for version in range(1, 9):
        _write(writers["a"], "a", n, version, 3)              # up to date
        if version <= 2:
            _write(writers["b"], "b", n, version, 3)          # stops after version 2
        _write(writers["c"], "c", n, max(0, version - 4), 3)  # 4 versions behind
        out, report = reader.take(version, 3, current_version=version)
        assert len(out) == 3
        for _, group in out:
            took[group[0]["shard"]] += 1
        stale_total += report.stale_groups
    assert took["a"] > 0 and took["b"] > 0
    assert took["c"] == 0 and stale_total > 0  # all of c is over age (4 > 2) and dropped
    positions = reader.buffer.shard_positions(reader.buffer.read_cursor)
    assert set(positions) == set(shards)       # every shard is accounted
    assert positions["b"] == n["b"]            # the stopped shard is fully read
    assert not reader.buffer.data_ended        # b never ended: no data end, but no stall


def test_stopped_shards_do_not_block_a_live_one(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    shards = ("a", "b", "c")
    writers = {s: FileSampleBuffer(objects, batch_size=1, index=shard_index(s)) for s in shards}
    n = dict.fromkeys(shards, 0)
    reader = _reader(objects, shards)
    for version in range(1, 6):
        _write(writers["c"], "c", n, version, 2)  # a and b never write
        out, _ = reader.take(version, 2, current_version=version)
        assert [g[0]["shard"] for _, g in out] == ["c", "c"]


def test_data_ends_only_after_every_expected_shard_ends(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    shards = ("a", "b")
    writers = {s: FileSampleBuffer(objects, batch_size=1, index=shard_index(s)) for s in shards}
    n = dict.fromkeys(shards, 0)
    _write(writers["a"], "a", n, 1, 2)
    writers["a"].mark_data_end()
    buffer = FileSampleBuffer(objects, batch_size=1, merge_shards=True, expected_shards=shards)
    assert not buffer.data_ended
    _write(writers["b"], "b", n, 1, 1)
    writers["b"].mark_data_end()
    buffer.refresh()
    assert buffer.data_ended and len(buffer) == 3


def test_a_shard_not_named_is_never_merged(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    n = {"a": 0, "stray": 0}
    for s in n:
        _write(FileSampleBuffer(objects, batch_size=1, index=shard_index(s)), s, n, 1, 2)
    buffer = FileSampleBuffer(objects, batch_size=1, merge_shards=True, expected_shards=["a"])
    assert buffer.shard_positions(len(buffer)) == {"a": 2}
