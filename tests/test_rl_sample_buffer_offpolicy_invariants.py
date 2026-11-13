"""CPU only: invariants of the durable sample buffer and the trainer's off-policy
gate that the rest of the asynchronous RL path relies on.

* the age gate keeps a group iff its OLDEST token version is >= ``age_floor``
  (one sample of an older version makes the whole group old);
* the version statistics of a take count exactly the trained groups;
* a group id is written once: a second write of the same group is refused and
  the index does not change;
* a trainer restored at a cut's watermark trains every later group once and
  nothing below it, also across a restarted shard merger."""

from __future__ import annotations

import pytest

from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.policy_age import age_floor
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import (
    FileSampleBuffer,
    SampleBufferError,
    open_trainer_buffer,
    shard_index,
)
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry


def _buffer(root):
    return FileSampleBuffer(CheckpointStore(root), batch_size=2)


def _group(g, version, oldest=None):
    """Two samples of version ``version``; with ``oldest`` the second one is that older version."""
    return [sample(2 * g, g, version), sample(2 * g + 1, g, version if oldest is None else oldest)]


def _reader(root, age):
    return br.BatchReader(_buffer(root), max_policy_age=age, wait_s=0.0, poll_s=0.0)


@pytest.mark.parametrize("age", [1, 2, 4])
def test_the_gate_keeps_exactly_the_versions_at_or_above_the_floor(tmp_path, age):
    current = 8
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    versions = list(range(current - age - 2, current + 1))
    for g, version in enumerate(versions):
        writer.write(_group(g, version), group_key=f"g{g}")
    kept = [v for v in versions if v >= age_floor(current, age)]
    groups, report = _reader(tmp_path, age).take(0, len(kept), current_version=current)
    assert [g[0]["group_index"] for _, g in groups] == [versions.index(v) for v in kept]
    assert report.stale_groups == len(versions) - len(kept) == 2
    assert report.behavior_versions == {v: 1 for v in kept}
    assert sum(report.behavior_versions.values()) == report.groups == len(groups)


def test_one_old_sample_makes_the_whole_group_old(tmp_path):
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    writer.write(_group(0, 5, oldest=2), group_key="mixed")  # floor 4 at version 5, age 1
    writer.write(_group(1, 5), group_key="fresh")
    groups, report = _reader(tmp_path, 1).take(0, 1, current_version=5)
    assert [g[0]["group_index"] for _, g in groups] == [1]
    assert report.stale_groups == 1 and report.behavior_versions == {5: 1}


def test_a_group_id_is_written_once(tmp_path):
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    writer.write(_group(0, 1), group_key="g0")
    with pytest.raises(SampleBufferError, match="never overwritten"):
        writer.write(_group(0, 2), group_key="g0")
    assert len(_buffer(tmp_path)) == 1 and _buffer(tmp_path).entries()[0].behavior_version == 1


def test_a_restored_trainer_trains_every_group_after_the_watermark_once(tmp_path):
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    for g in range(8):
        writer.write(_group(g, 3), group_key=f"g{g}")
    first = _reader(tmp_path, 2)
    trained = []
    for rid in range(2):
        groups, _ = first.take(rid, 2, current_version=3)
        trained += [pos for pos, _ in groups]
    cut = first.watermark_before(1)  # a cut after round 0; round 1 is lost with the process
    assert cut == 2
    again = _reader(tmp_path, 2)
    again.reset(cut)
    rest = []
    for rid in range(1, 4):
        groups, _ = again.take(rid, 2, current_version=3)
        rest += [pos for pos, _ in groups]
    assert trained[:2] == [0, 1] and rest == list(range(2, 8))  # none below the cut, none twice


def test_a_restarted_merger_continues_each_shard_where_the_cut_left_it(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    writers = {s: FileSampleBuffer(objects, batch_size=1, index=shard_index(s)) for s in ("a", "b")}
    clock = 0.0
    for i in range(3):
        for s, w in writers.items():
            clock += 1.0
            w.append(SampleIndexEntry(f"{s}{i}", 1, "", 1, clock))
    trainer = open_trainer_buffer(objects, batch_size=1, shards=["a", "b"])
    taken = [item.entry.sample_id for item in trainer.select("trainer", 4)]
    trainer.commit_consumed(4)
    shards_at_cut = trainer.shard_positions(trainer.watermark)
    for s, w in writers.items():  # more groups while the trainer is down
        clock += 1.0
        w.append(SampleIndexEntry(f"{s}3", 1, "", 1, clock))
    restarted = open_trainer_buffer(objects, batch_size=1, shards=["a", "b"])
    restarted.reset_to(4)
    assert restarted.shard_positions(4) == shards_at_cut == {"a": 2, "b": 2}
    rest = [item.entry.sample_id for item in restarted.select("trainer", 99)]
    assert sorted(taken + rest) == sorted(f"{s}{i}" for s in "ab" for i in range(4))
    assert len(set(taken + rest)) == 8
