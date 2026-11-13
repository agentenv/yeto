"""Batched shard merge on an object-store index (ObjectIndex.append_batch, concurrent
read_new): same records as the per-record path, one LIST per batch, safe with
concurrent appenders."""

from __future__ import annotations

import random
import threading

import pytest

from yeto.rl.engine import sample_buffer as sb
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, ObjectIndex, shard_index
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry


class Counting(DirectoryObjectStore):
    def __init__(self, root):
        super().__init__(root)
        self.calls = {"list": 0, "read": 0, "create": 0}
        self._lock = threading.Lock()

    def _n(self, name):
        with self._lock:
            self.calls[name] += 1

    def list(self, prefix, *, start_after=""):
        self._n("list")
        return super().list(prefix, start_after=start_after)

    def read(self, key):
        self._n("read")
        return super().read(key)

    def create(self, key, data):
        self._n("create")
        return super().create(key, data)


def _fill(objects, shards=("a", "b", "c"), per_shard=60, seed=7, end=True):
    rnd = random.Random(seed)
    for shard in shards:
        w = FileSampleBuffer(objects, batch_size=1, index=shard_index(shard))
        t = 0.0
        for i in range(per_shard):
            t += rnd.choice((0.5, 1.0, 1.0, 2.0))  # ties across shards on purpose
            w.append(SampleIndexEntry(f"{shard}{i}", i // 10, "", 10 + i, t,
                                      provenance={"src": shard}, sampler_round={"round": i}))
        if end:
            w.mark_data_end()


def _old_write_records(self, records):
    """The per-record path before the batch (reference)."""
    with self._lock:
        for record in records:
            self._write_record(lambda pos, record=record: {**record, "pos": pos})


def _index_objects(objects):
    keys = objects.list("sample-buffer/index/")
    return [(k, objects.read(k)) for k in keys]


def test_batch_merge_equals_the_per_record_merge(tmp_path, monkeypatch):
    new, old = Counting(tmp_path / "new"), DirectoryObjectStore(tmp_path / "old")
    _fill(new)
    _fill(old)
    fast = FileSampleBuffer(new, batch_size=1, merge_shards=True)
    with monkeypatch.context() as m:
        m.setattr(FileSampleBuffer, "_write_records", _old_write_records)
        slow = FileSampleBuffer(old, batch_size=1, merge_shards=True)
    assert _index_objects(new) == _index_objects(old)  # every record object byte for byte
    assert [e.sample_id for e in fast.entries()] == [e.sample_id for e in slow.entries()]
    assert fast.data_ended and slow.data_ended and len(fast) == 180
    assert fast.shard_positions(len(fast)) == {"a": 60, "b": 60, "c": 60}


def test_batch_merge_lists_once_per_batch(tmp_path):
    objects = Counting(tmp_path)
    _fill(objects, per_shard=200)
    objects.calls.update(list=0, read=0, create=0)
    buf = FileSampleBuffer(objects, batch_size=1, merge_shards=True)
    assert len(buf) == 600
    # 600 records + the end record + the merger's trainer generation (sample_buffer.claim_reader_generation)
    assert objects.calls["create"] == 602
    # index read + shard names + 3 shard reads + the batch's one read_new (was ~2 per record)
    assert objects.calls["list"] <= 8
    assert objects.calls["read"] == 603  # each shard record once; nothing read back


def test_incremental_refresh_and_restart_resume(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    _fill(objects, per_shard=30, end=False)
    buf = FileSampleBuffer(objects, batch_size=1, merge_shards=True)
    assert len(buf) == 90
    w = FileSampleBuffer(objects, batch_size=1, index=shard_index("b"))
    w.append(SampleIndexEntry("late", 9, "", 1, 1e9))
    assert buf.refresh() == 91 and buf.entries()[-1].sample_id == "late"
    again = FileSampleBuffer(objects, batch_size=1, merge_shards=True)  # merges nothing twice
    assert [e.sample_id for e in again.entries()] == [e.sample_id for e in buf.entries()]


def test_append_batch_loses_races_without_gaps_or_duplicates(tmp_path):
    """Writers appending to one shared index (create-if-absent) while a batch goes in:
    every record lands once, positions are contiguous, each writer keeps its order."""
    objects = DirectoryObjectStore(tmp_path)
    n_batch, n_single, writers = 300, 120, 3
    start = threading.Barrier(writers + 1)

    def single(name):
        buf = FileSampleBuffer(objects, batch_size=1)
        start.wait()
        for i in range(n_single):
            buf.append(SampleIndexEntry(f"{name}-{i}", 0, "", 1, 1.0 + i))

    threads = [threading.Thread(target=single, args=(f"w{k}",)) for k in range(writers)]
    for t in threads:
        t.start()
    batcher = FileSampleBuffer(objects, batch_size=1)
    records = [{"schema": sb.SCHEMA, "sample_id": f"batch-{i}", "behavior_version": 0, "payload_ref": "",
                "tokens": 1, "appended_at": 1.0} for i in range(n_batch)]
    start.wait()
    batcher._write_records(records)
    for t in threads:
        t.join()
    final = ObjectIndex(objects).read_new()
    assert [r["pos"] for r in final] == list(range(len(final)))
    ids = [r["sample_id"] for r in final]
    assert len(ids) == len(set(ids)) == n_batch + writers * n_single
    for prefix, n in (("batch-", n_batch), *((f"w{k}-", n_single) for k in range(writers))):
        mine = [int(i.rsplit("-", 1)[1]) for i in ids if i.startswith(prefix)]
        assert mine == list(range(n))
    batcher.refresh()
    assert [e.sample_id for e in batcher.entries()] == ids


def test_read_new_stops_at_a_gap(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    idx = ObjectIndex(objects)
    for pos in (0, 1, 3):
        objects.create(idx.key(pos), (f'{{"schema": "{sb.SCHEMA}", "pos": {pos}, "end": true}}\n').encode())
    assert [r["pos"] for r in idx.read_new()] == [0, 1]
    objects.create(idx.key(2), (f'{{"schema": "{sb.SCHEMA}", "pos": 2, "end": true}}\n').encode())
    assert [r["pos"] for r in idx.read_new()] == [2, 3]


def test_append_after_end_refused(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    _fill(objects, per_shard=3)
    buf = FileSampleBuffer(objects, batch_size=1, merge_shards=True)
    with pytest.raises(sb.SampleBufferError, match="after the data end"):
        buf._write_records([{"schema": sb.SCHEMA, "sample_id": "x", "behavior_version": 0,
                             "payload_ref": "", "tokens": 1, "appended_at": 1.0}])
