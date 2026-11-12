"""The shard merge is idempotent: a merged record is identified by (shard, shard_pos).

Two mergers at once (a trainer not dead yet and the one that replaces it), a create
whose answer was lost, or a merge that failed half way leave every shard record in
the merged index once, and a restarted merger continues every shard after its
highest merged position. An index that already repeats a record (written before
this rule) is read with the repeat passed over: never selected, never counted."""

from __future__ import annotations

import json
import random
import threading

import pytest

from yeto.rl.engine.merged_release import BufferLane
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import (
    SCHEMA,
    FileSampleBuffer,
    ObjectIndex,
    UnmergedShards,
    merge_shards,
    open_trainer_buffer,
    shard_index,
)
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

SHARDS = ("a", "b")


class Hooked(DirectoryObjectStore):
    """``before[n]``: called before this store's n-th create of a merged index record
    (once). ``lose``: those creates are applied, then answered as lost. ``fail``: those
    raise instead of creating."""

    def __init__(self, root):
        super().__init__(root)
        self.before: dict[int, object] = {}
        self.lose: set[int] = set()
        self.fail: set[int] = set()
        self.creates = 0

    def create(self, key, data):
        if not key.startswith("sample-buffer/index/"):
            return super().create(key, data)
        n = self.creates
        self.creates += 1
        if hook := self.before.pop(n, None):
            hook()
        if n in self.fail:
            self.fail.discard(n)
            raise OSError("store unavailable")
        created = super().create(key, data)
        if n in self.lose:
            self.lose.discard(n)
            return False
        return created


def _writers(objects, shards=SHARDS):
    return {s: FileSampleBuffer(objects, batch_size=1, index=shard_index(s)) for s in shards}


def _add(writers, shard, count, clock):
    for _ in range(count):
        writer = writers[shard]
        clock[0] += 1.0
        writer.append(SampleIndexEntry(f"{shard}{len(writer)}", 0, "", 1, clock[0]))


def _merged(objects):
    """(shard, shard_pos, sample_id) of every merged record, in index order."""
    out = []
    for key in objects.list("sample-buffer/index/"):
        record = json.loads(objects.read(key))
        if not record.get("end"):
            out.append((record["shard"], record["shard_pos"], record["sample_id"]))
    return out


def _assert_every_record_once_in_shard_order(objects, counts):
    merged = _merged(objects)
    for shard, count in counts.items():
        assert [(pos, sid) for s, pos, sid in merged if s == shard] == [
            (i, f"{shard}{i}") for i in range(count)], shard
    assert len(merged) == sum(counts.values())


@pytest.mark.parametrize("at", range(6))
@pytest.mark.parametrize("other_knows_the_index", [True, False])
def test_two_mergers_interleaved_at_any_create_leave_every_record_once(tmp_path, at, other_knows_the_index):
    """The old trainer merges the same shard records between the new trainer's read of
    the index and its ``at``-th create. Before: the loser appended its whole batch
    again at the new positions, and a later restart skipped as many real records."""
    plain, hooked = DirectoryObjectStore(tmp_path), Hooked(tmp_path)
    writers, clock = _writers(plain), [0.0]
    _add(writers, "a", 2, clock)
    _add(writers, "b", 1, clock)
    old = open_trainer_buffer(plain, batch_size=1, shards=SHARDS) if other_knows_the_index else None
    new = open_trainer_buffer(hooked, batch_size=1, shards=SHARDS)
    if old is None:  # a merger that starts in the middle of the other's batch
        old = open_trainer_buffer(plain, batch_size=1, shards=SHARDS)
    _add(writers, "a", 3, clock)
    _add(writers, "b", 3, clock)
    hooked.creates = 0
    hooked.before[at] = old.refresh
    new.refresh()
    old.refresh()
    _assert_every_record_once_in_shard_order(plain, {"a": 5, "b": 4})
    assert [e.sample_id for e in new.entries()] == [e.sample_id for e in old.entries()] == [
        sid for _, _, sid in _merged(plain)]
    assert new.duplicate_merged_records == old.duplicate_merged_records == 0
    # the shards go on; each of the two, and a merger started later, continues correctly
    _add(writers, "a", 2, clock)
    _add(writers, "b", 2, clock)
    third = open_trainer_buffer(plain, batch_size=1, shards=SHARDS)
    _assert_every_record_once_in_shard_order(plain, {"a": 7, "b": 6})
    for writer in writers.values():
        writer.mark_data_end()
    assert new.refresh() == old.refresh() == third.refresh() == 13
    assert new.data_ended and old.data_ended and third.data_ended
    assert third.shard_positions(13) == new.shard_positions(13) == {"a": 7, "b": 6}
    _assert_every_record_once_in_shard_order(plain, {"a": 7, "b": 6})


class TakingTurns(DirectoryObjectStore):
    """Two stores whose creates of merged index records strictly alternate (while both
    mergers run): each merger loses about every second create to the other."""

    def __init__(self, root, name, turns):
        super().__init__(root)
        self.name, self.turns, self.lost = name, turns, 0

    def create(self, key, data):
        if not key.startswith("sample-buffer/index/"):
            return super().create(key, data)
        turns = self.turns
        with turns["cond"]:
            turns["cond"].wait_for(lambda: turns["next"] == self.name or len(turns["done"]) == 1, timeout=30)
            created = super().create(key, data)
            self.lost += not created
            turns["next"] = "two" if self.name == "one" else "one"
            turns["cond"].notify_all()
        return created


def test_two_mergers_that_take_turns_record_by_record_leave_every_record_once(tmp_path):
    turns = {"cond": threading.Condition(), "next": "one", "done": set()}
    one, two = TakingTurns(tmp_path, "one", turns), TakingTurns(tmp_path, "two", turns)
    plain = DirectoryObjectStore(tmp_path)
    writers, clock = _writers(plain), [0.0]
    mergers = {"one": open_trainer_buffer(one, batch_size=1, shards=SHARDS),
               "two": open_trainer_buffer(two, batch_size=1, shards=SHARDS)}
    _add(writers, "a", 20, clock)
    _add(writers, "b", 20, clock)
    errors = []

    def run(name):
        try:
            mergers[name].refresh()
        except Exception as error:  # noqa: BLE001 - reported by the assert below
            errors.append(error)
        finally:
            with turns["cond"]:
                turns["done"].add(name)
                turns["cond"].notify_all()

    threads = [threading.Thread(target=run, args=(name,)) for name in mergers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors
    assert one.lost + two.lost >= 10  # they did interleave
    for merger in mergers.values():
        merger.refresh()
    _assert_every_record_once_in_shard_order(plain, {"a": 20, "b": 20})
    restarted = open_trainer_buffer(plain, batch_size=1, shards=SHARDS)
    assert len(restarted) == 40 and restarted.duplicate_merged_records == 0
    assert [e.sample_id for e in mergers["one"].entries()] == [e.sample_id for e in restarted.entries()]


@pytest.mark.parametrize("lost", range(4))
def test_a_lost_create_answer_does_not_merge_the_record_twice(tmp_path, lost):
    """The create was applied, the merger was told it lost (an SDK retry answered 412).
    Before: it merged the record again one position later; the restarted merger then
    skipped one real record."""
    objects = Hooked(tmp_path)
    writers, clock = _writers(objects, ("a",)), [0.0]
    _add(writers, "a", 4, clock)
    objects.lose.add(lost)
    trainer = open_trainer_buffer(objects, batch_size=1, shards=["a"])
    assert [e.sample_id for e in trainer.entries()] == ["a0", "a1", "a2", "a3"]
    _add(writers, "a", 3, clock)
    restarted = open_trainer_buffer(objects, batch_size=1, shards=["a"])
    assert [e.sample_id for e in restarted.entries()] == [f"a{i}" for i in range(7)]
    _assert_every_record_once_in_shard_order(objects, {"a": 7})


@pytest.mark.parametrize("failed", range(4))
def test_a_merge_that_fails_half_way_is_finished_by_the_next_refresh(tmp_path, failed):
    """A store error on one create leaves the records read from the shards pending.
    Before: the shard readers had moved past them, so this merger never merged them
    and appended the later records after a hole."""
    objects = Hooked(tmp_path)
    writers, clock = _writers(objects), [0.0]
    trainer = open_trainer_buffer(objects, batch_size=1, shards=SHARDS)
    _add(writers, "a", 2, clock)
    _add(writers, "b", 2, clock)
    objects.creates = 0
    objects.fail.add(failed)
    with pytest.raises(OSError):
        trainer.refresh()
    _add(writers, "a", 1, clock)
    assert trainer.refresh() == 5
    _assert_every_record_once_in_shard_order(objects, {"a": 3, "b": 2})
    assert trainer.unmerged_shard_gaps == 0


def _write_index(objects, records):
    for pos, (shard, shard_pos) in enumerate(records):
        record = {"schema": SCHEMA, "pos": pos, "sample_id": f"{shard}{shard_pos}", "behavior_version": 0,
                  "payload_ref": "", "tokens": 1, "appended_at": float(shard_pos + 1), "shard": shard,
                  "shard_pos": shard_pos}
        assert objects.create(ObjectIndex(objects).key(pos), (json.dumps(record, sort_keys=True) + "\n").encode())


def test_an_index_that_repeats_merged_records_is_read_with_each_group_once(tmp_path, capsys):
    """An index written by two mergers before the merge was idempotent: a0 a1 a2 a3,
    then a2 a3 again. The repeats keep their positions but are never selected, and
    the shard records after them are merged (before: a4 and a5 were skipped for good)."""
    objects = DirectoryObjectStore(tmp_path)
    writers, clock = _writers(objects, ("a",)), [0.0]
    _add(writers, "a", 6, clock)
    _write_index(objects, [("a", 0), ("a", 1), ("a", 2), ("a", 3), ("a", 2), ("a", 3)])
    trainer = open_trainer_buffer(objects, batch_size=1, shards=["a"])
    assert trainer.duplicate_merged_records == 2
    assert capsys.readouterr().out.count("WARNING") == 1  # one warning, not one per repeat
    assert [(s, pos, sid) for s, pos, sid in _merged(objects)][6:] == [("a", 4, "a4"), ("a", 5, "a5")]
    assert len(trainer) == 8  # positions stay those of the index
    assert trainer.backlog(0).samples == 6
    assert [(item.position, item.entry.sample_id) for item in trainer.peek(8)] == [
        (0, "a0"), (1, "a1"), (2, "a2"), (3, "a3"), (6, "a4"), (7, "a5")]
    first = trainer.select("trainer", 4)
    assert [item.entry.sample_id for item in first] == ["a0", "a1", "a2", "a3"]
    assert trainer.shard_positions(trainer.read_cursor) == {"a": 4}
    rest = trainer.select("trainer", 4)
    assert [(item.position, item.entry.sample_id) for item in rest] == [(6, "a4"), (7, "a5")]
    assert trainer.read_cursor == 8 and trainer.shard_positions(8) == {"a": 6}
    assert trainer.shard_positions(5) == trainer.shard_positions(6) == {"a": 4}
    trainer.commit_consumed(8)
    # a restore from a cut inside the repeats continues after them
    resumed = open_trainer_buffer(objects, batch_size=1, shards=["a"])
    resumed.reset_to(5)
    assert [item.entry.sample_id for item in resumed.select("trainer", 9)] == ["a4", "a5"]
    assert resumed.backlog(0, start=4).samples == 2


def test_an_index_with_repeats_does_not_hide_unmerged_records_from_the_head(tmp_path):
    """The head counts the shard records above the merged position of their shard.
    Before: the merged count included the repeats, so as many unmerged records were
    taken for merged."""
    objects = DirectoryObjectStore(tmp_path)
    writers, clock = _writers(objects, ("a",)), [0.0]
    _add(writers, "a", 6, clock)
    _write_index(objects, [("a", 0), ("a", 1), ("a", 2), ("a", 3), ("a", 2), ("a", 3)])
    head = FileSampleBuffer(objects, batch_size=1, create_index=False)  # reads the index, merges nothing
    assert head.shard_positions(len(head)) == {"a": 4}
    unmerged = UnmergedShards(objects)
    stats = head.backlog(0, start=0, unmerged=unmerged)
    assert stats.samples == 6  # a0..a3 merged once + a4 a5 not merged yet
    pending, ended = unmerged.pending(head.shard_positions(len(head)))
    assert [e.sample_id for e in pending] == ["a4", "a5"] and not ended


def test_two_mergers_on_a_shared_directory_leave_every_record_once(tmp_path):
    """``shards/<shard>.jsonl`` merged into ``index.jsonl`` by two trainer buffers:
    the second one drops what the first appended since it read the shards."""
    store = CheckpointStore(tmp_path)
    writers = {s: FileSampleBuffer(store, batch_size=1, index=shard_index(s)) for s in SHARDS}
    clock = [0.0]
    old = open_trainer_buffer(store, batch_size=1, shards=SHARDS)
    new = open_trainer_buffer(store, batch_size=1, shards=SHARDS)
    _add(writers, "a", 3, clock)
    _add(writers, "b", 2, clock)
    real = new.index.shard_reader

    def reader_after_the_other_merged(shard, start):  # new has read the index; old merges now
        old.refresh()
        return real(shard, start)

    new.index.shard_reader = reader_after_the_other_merged
    new.refresh()
    lines = [json.loads(line) for line in (tmp_path / "sample-buffer" / "index.jsonl").read_text().splitlines()]
    assert sorted((r["shard"], r["shard_pos"]) for r in lines) == [
        ("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1)]
    _add(writers, "a", 1, clock)
    restarted = open_trainer_buffer(store, batch_size=1, shards=SHARDS)
    assert len(restarted) == 6 and restarted.shard_positions(6) == {"a": 4, "b": 2}
    assert restarted.duplicate_merged_records == 0


class _Files:
    """Store access of the convertible pool's head (``trainer_remote.StoreFiles``)."""

    def __init__(self, root):
        self.root = root

    def read(self, rel):
        path = self.root / rel
        return path.read_bytes() if path.exists() else None

    def write(self, rel, data):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _clock_stepped_back(tmp_path):
    store = CheckpointStore(tmp_path)
    shard = FileSampleBuffer(store, batch_size=1, index=shard_index("a"))
    shard.append(SampleIndexEntry("a0", 0, "", 1, 100.0))
    shard.append(SampleIndexEntry("a1", 0, "", 1, 50.0))  # the sampler's wall clock stepped back
    shard.append(SampleIndexEntry("a2", 0, "", 1, 60.0))
    return store


def test_the_head_merge_keeps_a_shard_in_its_own_order_when_its_clock_stepped_back(tmp_path):
    """``merge_shards()`` sorted one batch by ``appended_at`` across and inside the
    shards: the later record went in first, and a reader that takes a record below
    its shard's highest merged position for a repeat never selected the earlier one."""
    store = _clock_stepped_back(tmp_path)
    assert merge_shards(_Files(tmp_path), ["a"]) == 3
    lines = [json.loads(line) for line in (tmp_path / "sample-buffer" / "index.jsonl").read_text().splitlines()]
    assert [(r["shard_pos"], r["sample_id"]) for r in lines] == [(0, "a0"), (1, "a1"), (2, "a2")]
    reader = FileSampleBuffer(store, batch_size=1, create_index=False)
    assert [s.entry.sample_id for s in reader.select("trainer", 10)] == ["a0", "a1", "a2"]
    assert reader.duplicate_merged_records == 0 and reader.unmerged_shard_gaps == 0


def test_an_index_merged_out_of_shard_order_is_read_with_every_record(tmp_path):
    """An index the older ``merge_shards()`` wrote: a1 a2 a0. The record that fills a
    position passed over earlier is a real record; a true repeat is still passed over."""
    store = _clock_stepped_back(tmp_path)
    shard_lines = (tmp_path / "sample-buffer" / "shards" / "a.jsonl").read_text().splitlines()
    by_pos = {r["pos"]: r for r in map(json.loads, shard_lines)}
    order = [1, 2, 0, 2]  # as sorted by appended_at, then a2 once more (a repeat)
    index = [{**by_pos[shard_pos], "shard": "a", "shard_pos": shard_pos, "pos": pos}
             for pos, shard_pos in enumerate(order)]
    (tmp_path / "sample-buffer" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    reader = FileSampleBuffer(store, batch_size=1, create_index=False)
    assert [(s.position, s.entry.sample_id) for s in reader.select("trainer", 10)] == [
        (0, "a1"), (1, "a2"), (2, "a0")]
    assert reader.duplicate_merged_records == 1 and reader.unmerged_shard_gaps == 0
    assert reader.shard_positions(4) == {"a": 3}


def _jsonl_index(tmp_path, shard_positions, shard="a"):
    """Write ``index.jsonl`` with the shard's records in the given order of shard positions."""
    shard_lines = (tmp_path / "sample-buffer" / "shards" / f"{shard}.jsonl").read_text().splitlines()
    by_pos = {r["pos"]: r for r in map(json.loads, shard_lines)}
    index = [{**by_pos[shard_pos], "shard": shard, "shard_pos": shard_pos, "pos": pos}
             for pos, shard_pos in enumerate(shard_positions)]
    (tmp_path / "sample-buffer" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))


def test_an_index_whose_shard_positions_go_down_is_read_with_every_record_once(tmp_path, capsys):
    """Appended at 10, 11, 9, 12: the older ``merge_shards()`` wrote g2 g0 g1 g3. A
    reader that took every position below the highest one for a repeat selected g2 and
    g3 only and warned about repeats that are none."""
    store = CheckpointStore(tmp_path)
    shard = FileSampleBuffer(store, batch_size=1, index=shard_index("a"))
    for n, at in enumerate((10.0, 11.0, 9.0, 12.0)):
        shard.append(SampleIndexEntry(f"g{n}", 0, "", 1, at))
    _jsonl_index(tmp_path, [2, 0, 1, 3])
    reader = FileSampleBuffer(store, batch_size=1, create_index=False)
    assert [s.entry.sample_id for s in reader.select("trainer", 10)] == ["g2", "g0", "g1", "g3"]
    assert reader.duplicate_merged_records == 0 and reader.unmerged_shard_gaps == 0
    assert "repeats" not in capsys.readouterr().out
    assert [reader.shard_positions(c) for c in range(5)] == [{}, {"a": 3}, {"a": 3}, {"a": 3}, {"a": 4}]
    # the head goes on merging the shard after its highest merged position
    shard.append(SampleIndexEntry("g4", 0, "", 1, 13.0))
    assert merge_shards(_Files(tmp_path), ["a"]) == 1
    assert [s.entry.sample_id for s in reader.select("trainer", 10)] == ["g4"]


def test_the_head_merge_continues_after_the_highest_merged_position_not_the_record_count(tmp_path):
    """An index that repeats two records: by count the head would continue two records
    too far (a4 and a5 never merged)."""
    store = CheckpointStore(tmp_path)
    shard = FileSampleBuffer(store, batch_size=1, index=shard_index("a"))
    for n in range(6):
        shard.append(SampleIndexEntry(f"a{n}", 0, "", 1, float(n + 1)))
    _jsonl_index(tmp_path, [0, 1, 2, 3, 2, 3])
    assert merge_shards(_Files(tmp_path), ["a"]) == 2
    reader = FileSampleBuffer(store, batch_size=1, create_index=False)
    assert [s.entry.sample_id for s in reader.select("trainer", 10)] == ["a0", "a1", "a2", "a3", "a4", "a5"]
    assert reader.duplicate_merged_records == 2 and merge_shards(_Files(tmp_path), ["a"]) == 0


def test_a_merger_restarted_on_an_index_with_a_hole_continues_after_it_and_repeats_nothing(tmp_path):
    """An index the count-based resume left: a0 a1, then a4 (a2 and a3 passed over for
    good). The restarted merger continues at a5; the hole is counted, not filled."""
    objects = DirectoryObjectStore(tmp_path)
    writers, clock = _writers(objects, ("a",)), [0.0]
    _add(writers, "a", 6, clock)
    _write_index(objects, [("a", 0), ("a", 1), ("a", 4)])
    trainer = open_trainer_buffer(objects, batch_size=1, shards=["a"])
    assert [sid for _, _, sid in _merged(objects)] == ["a0", "a1", "a4", "a5"]
    assert trainer.unmerged_shard_gaps == 2 and trainer.duplicate_merged_records == 0
    _add(writers, "a", 1, clock)
    again = open_trainer_buffer(objects, batch_size=1, shards=["a"])  # and once more
    assert [e.sample_id for e in again.entries()] == ["a0", "a1", "a4", "a5", "a6"]
    assert again.unmerged_shard_gaps == 2 and again.shard_positions(5) == {"a": 7}
    assert [s.entry.sample_id for s in again.select("trainer", 10)] == ["a0", "a1", "a4", "a5", "a6"]


def test_lane_readers_pass_over_a_repeated_position_and_keep_their_lanes(tmp_path):
    """Synced trainers read disjoint lanes of the index positions. A repeat keeps its
    position, so the lanes of the records after it do not move; its lane's reader
    moves its cursor past it and selects nothing for it."""
    objects = DirectoryObjectStore(tmp_path)
    writers, clock = _writers(objects, ("a",)), [0.0]
    _add(writers, "a", 6, clock)
    _write_index(objects, [("a", 0), ("a", 1), ("a", 2), ("a", 1), ("a", 3), ("a", 4), ("a", 5)])
    lanes = [FileSampleBuffer(objects, batch_size=1, create_index=False, lane=BufferLane(i, 2)) for i in range(2)]
    got = [[(s.position, s.entry.sample_id) for s in lane.select(f"t{i}", 10)] for i, lane in enumerate(lanes)]
    assert got == [[(0, "a0"), (2, "a2"), (4, "a3"), (6, "a5")], [(1, "a1"), (5, "a4")]]  # 3 is the repeat
    assert [lane.read_cursor for lane in lanes] == [7, 7]
    assert [lane.duplicate_merged_records for lane in lanes] == [1, 1]
    assert lanes[1].backlog(0, start=0).samples == 6
    assert [s.position for s in lanes[1].peek(5)] == []
    lanes[1].reset_to(2)
    assert [(s.position, s.entry.sample_id) for s in lanes[1].select("t1", 10)] == [(5, "a4")]


@pytest.mark.parametrize("seed", range(20))
def test_shard_positions_of_any_cursor_match_a_scan_of_the_index(tmp_path, seed):
    """``shard_positions`` is looked up, not scanned: for every cursor it is each shard's
    highest position below the cursor plus one, in any index (order, repeats, holes)."""
    rng = random.Random(seed)
    objects = DirectoryObjectStore(tmp_path)
    writers, clock = _writers(objects, ("a", "b", "c")), [0.0]
    for shard in writers:
        _add(writers, shard, 12, clock)
    records = [(shard, rng.randrange(12)) for shard in rng.choices("abc", k=40)]
    _write_index(objects, records)
    reader = FileSampleBuffer(objects, batch_size=1, create_index=False)
    for cursor in range(len(records) + 1):
        want: dict[str, int] = {}
        for shard, shard_pos in records[:cursor]:
            want[shard] = max(want.get(shard, 0), shard_pos + 1)
        assert reader.shard_positions(cursor) == dict(sorted(want.items()))
    firsts = {record: i for i, record in reversed(list(enumerate(records)))}
    assert [s.position for s in reader.select("t", 100)] == sorted(firsts.values())
    assert reader.duplicate_merged_records == len(records) - len(firsts)
    assert reader.unmerged_shard_gaps == sum(
        max(p for s, p in records if s == shard) + 1 - len({p for s, p in records if s == shard})
        for shard in {s for s, _ in records})
