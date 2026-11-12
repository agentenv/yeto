"""Shard ownership records: the fence of a sampler shard on an object store.

A fence is an ownership record created at the next position of the shard's index.
Every group created before it is merged, nothing the fenced incarnation writes
after it is - on the sampler (``ShardFenced``, nothing created) and again in the
trainer's merge (a writer that bypassed its own check). No clock takes part.
A shard without ownership records (a run whose trainer has no generation lease,
a shared-directory buffer) keeps the lease record and the trainer's gate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_stage1_gaps import _sampler_spec
from yeto.launch.nc import elastic
from yeto.launch.pool.data_path import R2DataPath
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.buffer_payload import encode_group
from yeto.rl.engine.incarnation_lease import (
    IncarnationLease,
    default_incarnation_gate,
    fence_shard_owner,
    read_lease,
)
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.resume import STORE_IMPL_ENV, CheckpointStore
from yeto.rl.engine.sample_buffer import (
    MERGE_EVENT,
    OWNER_SCHEMA,
    SCHEMA,
    FileSampleBuffer,
    ObjectIndex,
    SampleBufferError,
    ShardFenced,
    ShardTail,
    UnmergedShards,
    claim_shard_owner,
    open_trainer_buffer,
    shard_index,
    shard_prefix,
)
from yeto.rl.engine.task_dispense import DispenseMismatch
from yeto.rl.engine.trainer_lease import TrainerLease
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

SHARD = "s-0"


def _objects(tmp_path, *, trainer_lease=True):
    objects = DirectoryObjectStore(tmp_path)
    if trainer_lease:  # the run's trainer reads ownership records
        TrainerLease.claim(objects, incarnation="trainer-1")
    return objects


def _records(objects):
    """The store as the pool and the lease see it (``.objects``)."""
    return elastic.ObjectRecords(objects)


def _writer(objects, shard=SHARD, **kw):
    return FileSampleBuffer(objects, batch_size=1, index=shard_index(shard), **kw)


def _sampler(objects, incarnation, shard=SHARD, **kw):
    """A sampler incarnation at its join: its shard writer and its lease, claimed."""
    writer = _writer(objects, shard, **{k: v for k, v in kw.items() if k == "clock"})
    lease = IncarnationLease(_records(objects), shard, incarnation, buffer=writer, **kw)
    lease.claim(writer.entries())
    return writer, lease


def _entry(shard, incarnation, n, at=0.0):
    return SampleIndexEntry(f"{shard}-{incarnation}-g{n}", 0, "", 1, float(at),
                            sampler_round={"sampler": shard, "incarnation": incarnation, "rollout_id": n})


def _add(writer, incarnation, numbers, at=0.0):
    for n in numbers:
        writer.append(_entry(writer.shard, incarnation, n, at))


def _trainer(objects, shards=(SHARD,)):
    return open_trainer_buffer(objects, batch_size=1, shards=shards)


def _ids(buffer):
    return [e.sample_id.split("-", 2)[2] if e.sample_id.count("-") >= 3 else e.sample_id for e in buffer.entries()]


def _names(buffer):
    """``<incarnation>-g<n>`` of every merged entry."""
    return [e.sample_id[len(SHARD) + 1:] if e.sample_id.startswith(SHARD + "-") else e.sample_id
            for e in buffer.entries()]


def _shard(objects, shard=SHARD):
    return [json.loads(objects.read(key)) for key in objects.list(shard_prefix(shard))]


def _bypass(objects, incarnation, n, shard=SHARD, at=0.0):
    """A writer that does not check anything: one group record at the shard's next position."""
    pos = len(objects.list(shard_prefix(shard)))
    record = {"schema": SCHEMA, "pos": pos, "sample_id": f"{shard}-{incarnation}-g{n}", "behavior_version": 0,
              "payload_ref": "", "tokens": 1, "appended_at": float(at),
              "sampler_round": {"sampler": shard, "incarnation": incarnation, "rollout_id": n}}
    assert objects.create(ObjectIndex(objects, prefix=shard_prefix(shard)).key(pos),
                          (json.dumps(record, sort_keys=True) + "\n").encode())


# ----------------------------------------------------------------- the rule, without clocks

@pytest.mark.parametrize("sampler_skew", [0.0, -600.0, 600.0])
@pytest.mark.parametrize("pool_skew", [0.0, -600.0, 600.0])
def test_groups_before_the_fence_are_all_trained_and_none_after_it_whatever_the_clocks_say(
        tmp_path, sampler_skew, pool_skew):
    """Before: the fence was a lease record with the pool's ``claimed_at`` and the
    trainer dropped a fenced incarnation's groups ``appended_at`` less than 60 s
    before it (lost: the dispenser had settled them), and accepted its later groups
    when the sampler's clock ran more than 60 s behind."""
    objects = _objects(tmp_path)
    now = 1_000_000.0
    writer, lease = _sampler(objects, "inc-1", clock=lambda: now + sampler_skew)
    _add(writer, "inc-1", range(3), at=now + sampler_skew)  # the last seconds before the fence
    record = elastic.fence_shard(_records(objects), SHARD, by="t-9", clock=lambda: now + pool_skew)
    assert record["incarnation"] == "pool-fence:t-9"
    assert record["owner"] == {"epoch": 2, "pos": 4, "by": "pool", "created": True}
    with pytest.raises(ShardFenced):
        _add(writer, "inc-1", [3], at=now + sampler_skew)
    assert lease.fenced and lease.fenced_by == "pool-fence:t-9" and lease.check_before_write() is True
    _bypass(objects, "inc-1", 4, at=now + sampler_skew - 3600)  # a zombie that skips every check
    trainer = _trainer(objects)
    assert _names(trainer) == ["inc-1-g0", "inc-1-g1", "inc-1-g2"]
    assert trainer.merge_counters()["stale_incarnation_groups"] == 1
    gate = default_incarnation_gate(trainer)  # the lease gate has no say on an owned record
    assert [gate.verdict(e, i) for i, e in enumerate(trainer.entries())] == [None, None, None]
    assert [trainer.shard_epoch_at(i) for i in range(3)] == [1, 1, 1]


def test_a_writer_that_bypasses_the_sampler_check_gets_nothing_trained(tmp_path, capsys):
    objects = _objects(tmp_path)
    writer, _ = _sampler(objects, "inc-1")
    _add(writer, "inc-1", range(2))
    trainer = _trainer(objects)
    assert fence_shard_owner(_records(objects), SHARD, "pool-fence:t-1")[1] is True
    for n in (2, 3):
        _bypass(objects, "inc-1", n)
    events = []
    trainer.merge_events = lambda event, **fields: events.append((event, fields))
    assert trainer.refresh() == 2
    assert trainer.take_fenced_dropped() == [
        {"shard": SHARD, "epoch": 2, "owner": "pool-fence:t-1", "incarnation": "inc-1", "groups": 2}]
    assert trainer.take_fenced_dropped() == []
    assert [(e, f["reason"], f["shard"], f["epoch"], f["shard_pos"]) for e, f in events] == [
        (MERGE_EVENT, "stale_incarnation", SHARD, 2, 4), (MERGE_EVENT, "stale_incarnation", SHARD, 2, 5)]
    assert capsys.readouterr().out.count("WARNING") == 1
    for key in objects.list("sample-buffer/index/"):  # the merged index holds neither kind
        merged = json.loads(objects.read(key))
        assert merged["schema"] == SCHEMA and merged["sampler_round"]["rollout_id"] in (0, 1)


def test_the_take_report_counts_the_dropped_groups_of_a_fenced_incarnation(tmp_path):
    objects = _objects(tmp_path)
    writer = _writer(objects)
    writer.claim_owner("inc-1")
    for n in range(2):
        ref = writer.write_payload(f"p{n}", encode_group([{"index": n}]))
        writer.append(SampleIndexEntry(f"{SHARD}-inc-1-g{n}", 0, ref, 1, 1.0,
                                       sampler_round={"sampler": SHARD, "incarnation": "inc-1", "rollout_id": n}))
    fence_shard_owner(_records(objects), SHARD, "pool-fence:t-1")
    _bypass(objects, "inc-1", 2)
    reader = br.BatchReader(_trainer(objects), max_policy_age=2, wait_s=0.5, poll_s=0.01,
                            sleep=lambda _s: None, prefetch=False, read_workers=1)
    out, report = reader.take(0, 2, current_version=0)
    assert len(out) == 2 and report.stale_incarnation_groups == 1
    assert report.stale_incarnation_shards == [
        {"shard": SHARD, "epoch": 2, "owner": "pool-fence:t-1", "incarnation": "inc-1", "groups": 1}]
    assert report.metrics()["sample_buffer/stale_incarnation_groups"] == 1.0
    assert report.metrics()["sample_buffer/duplicate_merged_records"] == 0.0


class Hooked(DirectoryObjectStore):
    """``before_create``: called once before this store's next create of a shard record."""

    before_create = None

    def create(self, key, data):
        hook, self.before_create = (self.before_create, None) if "/shards/" in key else (None, self.before_create)
        if hook:
            hook()
        return super().create(key, data)


def test_a_zombie_and_the_fence_race_for_one_position_and_the_position_decides(tmp_path):
    """The zombie has read the index, the fence lands before its create: it loses the
    position, reads the fence and writes nothing. The other way round its group is
    in before the fence and is trained."""
    plain = _objects(tmp_path)
    slow = Hooked(tmp_path)
    writer = _writer(slow)
    writer.claim_owner("inc-1")
    _add(writer, "inc-1", [0])
    # the fence wins position 2
    slow.before_create = lambda: fence_shard_owner(_records(plain), SHARD, "pool-fence:a")
    with pytest.raises(ShardFenced):
        _add(writer, "inc-1", [1])
    assert [r.get("sample_id") or r["owner"]["incarnation"] for r in _shard(plain)] == [
        "inc-1", f"{SHARD}-inc-1-g0", "pool-fence:a"]
    # the group wins: the fence is created one position later
    other = _writer(slow, "s-1")
    other.claim_owner("inc-1")
    pool = Hooked(tmp_path)
    pool.before_create = lambda: _add(other, "inc-1", [0])
    owner, created = claim_shard_owner(pool, "s-1", "pool-fence:b", by="pool")
    assert created and owner.pos == 2 and owner.epoch == 2
    trainer = _trainer(plain, (SHARD, "s-1"))
    assert sorted(e.sample_id for e in trainer.entries()) == ["s-0-inc-1-g0", "s-1-inc-1-g0"]
    assert trainer.merge_counters()["stale_incarnation_groups"] == 0
    with pytest.raises(ShardFenced):
        _add(other, "inc-1", [1])


def test_a_fenced_incarnation_cannot_end_the_shard(tmp_path):
    """Before: a superseded incarnation that saw a stop request wrote the data end;
    the incarnation restarted in place on the shard then found it ended and sampled nothing."""
    objects = _objects(tmp_path)
    writer, lease = _sampler(objects, "inc-1")
    _add(writer, "inc-1", [0])
    elastic.fence_incarnation(_records(objects), SHARD, by="t-1")
    with pytest.raises(ShardFenced):
        writer.mark_data_end()
    assert lease.fenced
    new, _ = _sampler(objects, "inc-2")
    assert not new.data_ended
    _add(new, "inc-2", [0])
    # an end record created without the check is void as well
    pos = len(objects.list(shard_prefix(SHARD)))
    objects.create(ObjectIndex(objects, prefix=shard_prefix(SHARD)).key(pos),
                   (json.dumps({"schema": SCHEMA, "pos": pos, "end": True, "incarnation": "inc-1"}) + "\n").encode())
    trainer = _trainer(objects)
    assert _names(trainer) == ["inc-1-g0", "inc-2-g0"] and not trainer.data_ended
    assert not _writer(objects).data_ended
    new.refresh()
    new.mark_data_end()  # the owner ends it
    assert trainer.refresh() == 2 and trainer.data_ended


def test_an_incarnation_restarted_in_place_takes_the_shard_and_the_old_one_is_void(tmp_path):
    objects = _objects(tmp_path)
    old, old_lease = _sampler(objects, "inc-1")
    _add(old, "inc-1", range(2))
    trainer = _trainer(objects)
    new, new_lease = _sampler(objects, "inc-2")  # no pool fence at all: a supervisor restart
    assert new_lease.owned and new_lease.owner.epoch == 2
    assert old_lease.poll() is True and old_lease.fenced_by == "inc-2"
    with pytest.raises(ShardFenced):
        _add(old, "inc-1", [2])
    _add(new, "inc-2", range(2))
    _bypass(objects, "inc-1", 3)
    _add(new, "inc-2", [2])
    assert trainer.refresh() == 5
    assert _names(trainer) == ["inc-1-g0", "inc-1-g1", "inc-2-g0", "inc-2-g1", "inc-2-g2"]
    assert [trainer.shard_epoch_at(i) for i in range(5)] == [1, 1, 2, 2, 2]
    assert trainer.shard_owners() == {SHARD: {"incarnation": "inc-2", "epoch": 2, "by": "sampler", "pos": 3}}
    assert trainer.shard_positions(5) == {SHARD: 8}  # shard positions: ownership records hold some
    assert new.groups_from(8) == 0 and new.groups_from(4) == 3 and new.groups_from(0) == 5
    assert [e.sample_id for e in new.entries()] == [f"{SHARD}-inc-1-g0", f"{SHARD}-inc-1-g1",
                                                    f"{SHARD}-inc-2-g0", f"{SHARD}-inc-2-g1", f"{SHARD}-inc-2-g2"]
    assert new.fenced_writes_seen == 1


def test_a_replacement_continues_in_its_own_shard_after_the_pool_fenced_the_old_one(tmp_path):
    objects = _objects(tmp_path)
    old, _ = _sampler(objects, "inc-1", "s-0")
    _add(old, "inc-1", range(2))
    elastic.fence_shard(_records(objects), "s-0", by="t-1")
    new, lease = _sampler(objects, "inc-9", "s-1")  # continues_shard: s-0
    assert lease.owned and lease.owner.epoch == 1
    _add(new, "inc-9", range(2))
    with pytest.raises(ShardFenced):
        _add(old, "inc-1", [2])
    trainer = _trainer(objects, ("s-0", "s-1"))
    assert sorted(e.sample_id for e in trainer.entries()) == [
        "s-0-inc-1-g0", "s-0-inc-1-g1", "s-1-inc-9-g0", "s-1-inc-9-g1"]
    assert trainer.shard_owners()["s-0"]["incarnation"] == "pool-fence:t-1"


@pytest.mark.parametrize("merged_before_restart", [0, 2, 4])
def test_a_restarted_trainer_restores_every_shard_owner(tmp_path, merged_before_restart):
    """The restarted merger reads each shard from its highest merged position: the
    owner there comes from the last merged record (``shard_epoch`` + its incarnation)."""
    objects = _objects(tmp_path)
    old, _ = _sampler(objects, "inc-1")
    first = _trainer(objects)
    _add(old, "inc-1", range(2))
    if merged_before_restart >= 2:
        first.refresh()
    new, _ = _sampler(objects, "inc-2")
    _add(new, "inc-2", range(2))
    if merged_before_restart >= 4:
        first.refresh()
    _bypass(objects, "inc-1", 7)  # the old incarnation, after it lost the shard
    _add(new, "inc-2", [2])
    restarted = _trainer(objects)
    assert _names(restarted) == ["inc-1-g0", "inc-1-g1", "inc-2-g0", "inc-2-g1", "inc-2-g2"]
    assert restarted.merge_counters()["stale_incarnation_groups"] == 1
    assert restarted.shard_owners()[SHARD]["epoch"] == 2
    _bypass(objects, "inc-1", 8)
    again = _trainer(objects)  # nothing merged since: the same owner again
    assert len(again) == 5 and again.merge_counters()["stale_incarnation_groups"] == 1
    merged = [json.loads(objects.read(k)) for k in objects.list("sample-buffer/index/")]
    assert [r.get("shard_epoch") for r in merged] == [1, 1, 2, 2, 2]


def test_the_head_does_not_count_ownership_records_or_fenced_writes_as_backlog(tmp_path):
    objects = _objects(tmp_path)
    writer, _ = _sampler(objects, "inc-1")
    _add(writer, "inc-1", range(2))
    fence_shard_owner(_records(objects), SHARD, "pool-fence:t-1")
    _bypass(objects, "inc-1", 2)
    head = FileSampleBuffer(objects, batch_size=1, create_index=False)
    pending, ended = UnmergedShards(objects).pending(head.shard_positions(len(head)))
    assert [e.sample_id for e in pending] == [f"{SHARD}-inc-1-g0", f"{SHARD}-inc-1-g1"] and not ended


# ----------------------------------------------------------------- the lease object

def test_the_lease_of_an_owned_shard_reads_the_shard_index_not_the_lease_record(tmp_path):
    objects = _objects(tmp_path)
    now = [0.0]
    writer, lease = _sampler(objects, "inc-1", write_check_s=3600.0, monotonic=lambda: now[0])
    assert lease.owned and lease.owner.pos == 0 and lease.poll() is False
    record = read_lease(_records(objects), SHARD)  # still written, for observation
    assert record["incarnation"] == "inc-1" and record["owner"] == {"epoch": 1, "pos": 0, "by": "sampler"}
    # a lease record that names someone else decides nothing
    objects.write(f"sample-buffer/leases/{SHARD}.json", json.dumps({"sampler": SHARD, "incarnation": "x"}).encode())
    assert lease.poll() is False and lease.check_before_write() is False
    _add(writer, "inc-1", [0])
    fence_shard_owner(_records(objects), SHARD, "pool-fence:t-1")
    assert lease.check_before_write() is False  # no request here: the write itself reads and refuses
    with pytest.raises(ShardFenced):
        _add(writer, "inc-1", [1])
    assert lease.check_before_write() is True and lease.fenced_by == "pool-fence:t-1"  # no poll was needed
    assert lease.detail()["shard_owner"]["epoch"] == 2


class Flaky(DirectoryObjectStore):
    down = False

    def list(self, prefix, *, start_after=""):
        if self.down and "/shards/" in prefix:
            raise OSError("store unavailable")
        return super().list(prefix, start_after=start_after)


def test_an_unreadable_shard_index_is_no_fence_but_stops_the_writes_after_some_reads(tmp_path):
    objects = Flaky(tmp_path)
    TrainerLease.claim(objects, incarnation="trainer-1")
    writer, lease = _sampler(objects, "inc-1", max_read_failures=3)
    objects.down = True
    assert [lease.poll() for _ in range(2)] == [False, False]
    assert lease.check_before_write() is False and not lease.fenced
    assert lease.poll() is False
    assert lease.check_before_write() is True and not lease.fenced  # no verdict for too long: no write
    with pytest.raises(OSError):  # and a write cannot get in unread
        _add(writer, "inc-1", [0])
    objects.down = False
    assert lease.poll() is False and lease.check_before_write() is False
    _add(writer, "inc-1", [0])
    assert len(writer) == 1


def test_a_writer_that_never_claimed_is_fenced_by_any_owner_and_a_claimed_one_checks_its_groups(tmp_path):
    objects = _objects(tmp_path)
    plain = _writer(objects)
    plain.append(SampleIndexEntry("g0", 0, "", 1, 1.0))
    owner, _ = _sampler(objects, "inc-1")
    with pytest.raises(ShardFenced):
        plain.append(SampleIndexEntry("g1", 0, "", 1, 1.0))
    with pytest.raises(SampleBufferError, match="round record names"):
        owner.append(SampleIndexEntry("g2", 0, "", 1, 1.0))  # the trainer would drop it
    assert _names(_trainer(objects)) == ["g0"]


def test_a_retried_pool_fence_writes_one_record_and_reads_only_the_end_of_the_shard(tmp_path):
    class Counting(DirectoryObjectStore):
        reads = 0

        def read(self, key):
            if "/shards/" in key:
                Counting.reads += 1
            return super().read(key)

    objects = _objects(tmp_path)
    writer, _ = _sampler(objects, "inc-1")
    _add(writer, "inc-1", range(200))
    pool = _records(Counting(tmp_path))
    first = elastic.fence_incarnation(pool, SHARD, by="t-1")
    assert first["owner"] == {"epoch": 2, "pos": 201, "by": "pool", "created": True}
    # the sampler's claim through the lease record's hint, then the shard's last record (its group)
    assert Counting.reads == 2
    again = elastic.fence_incarnation(pool, SHARD, by="t-1")
    assert again["owner"] == {"epoch": 2, "pos": 201, "by": "pool", "created": False}
    assert len(_shard(objects)) == 202
    # without the hint: back from the end to the newest ownership record, not the whole shard
    os.remove(Path(tmp_path) / "sample-buffer" / "leases" / f"{SHARD}.json")
    Counting.reads = 0
    tail = ShardTail()
    owner, created = claim_shard_owner(pool.objects, SHARD, "pool-fence:t-2", by="pool", tail=tail)
    assert created and owner.epoch == 3 and owner.pos == 202 and Counting.reads == 1
    Counting.reads = 0
    owner, created = claim_shard_owner(pool.objects, SHARD, "pool-fence:t-3", by="pool", tail=tail)
    assert created and owner.epoch == 4 and Counting.reads == 0  # the tail: nothing new to read


# ----------------------------------------------------------------- legacy and mixed versions

def test_without_a_trainer_lease_the_sampler_and_the_pool_keep_the_lease_record(tmp_path):
    """A run whose trainer is older code (it would fail on a v2 record): no ownership
    record is written, the lease record and the trainer's gate work as before."""
    objects = _objects(tmp_path, trainer_lease=False)
    writer, lease = _sampler(objects, "inc-1", clock=lambda: 100.0)
    assert not lease.owned
    _add(writer, "inc-1", range(2), at=100.0)
    assert lease.poll() is False
    record = elastic.fence_shard(_records(objects), SHARD, by="t-1", clock=lambda: 1000.0)
    assert "owner" not in record and fence_shard_owner(_records(objects), SHARD, "pool-fence:x") is None
    assert lease.poll() is True and lease.fenced_by == "pool-fence:t-1"
    _add(writer, "inc-1", [2], at=2000.0)  # nothing in the shard stops a legacy writer
    assert all(r["schema"] == SCHEMA for r in _shard(objects))
    trainer = _trainer(objects)
    assert len(trainer) == 3 and trainer.shard_owners() == {}
    gate = default_incarnation_gate(trainer)
    verdicts = [gate.verdict(e, i) for i, e in enumerate(trainer.entries())]
    assert verdicts[:2] == [None, None] and "superseded by pool-fence:t-1" in verdicts[2]
    merged = [json.loads(objects.read(k)) for k in objects.list("sample-buffer/index/")]
    assert all("shard_epoch" not in r for r in merged)


def test_a_shard_started_under_the_lease_record_continues_under_ownership_records(tmp_path):
    """The trainer was upgraded in the middle of a run: the shard's legacy part is
    read as before (the gate judges it), the part after the first ownership record
    by the ownership records alone."""
    objects = _objects(tmp_path, trainer_lease=False)
    old, old_lease = _sampler(objects, "inc-1", clock=lambda: 100.0)
    _add(old, "inc-1", range(2), at=100.0)
    TrainerLease.claim(objects, incarnation="trainer-1")
    new, new_lease = _sampler(objects, "inc-2", clock=lambda: 1000.0)
    assert new_lease.owned and not old_lease.owned
    with pytest.raises(ShardFenced):  # the new code's writer sees the record, whatever its lease mode
        _add(old, "inc-1", [2], at=2000.0)
    _add(new, "inc-2", range(2), at=1000.0)
    trainer = _trainer(objects)
    assert _names(trainer) == ["inc-1-g0", "inc-1-g1", "inc-2-g0", "inc-2-g1"]
    assert [trainer.shard_epoch_at(i) for i in range(4)] == [0, 0, 1, 1]
    gate = default_incarnation_gate(trainer)
    assert [gate.verdict(e, i) for i, e in enumerate(trainer.entries())] == [None] * 4
    assert _trainer(objects).shard_owners()[SHARD]["incarnation"] == "inc-2"  # and after a restart


def test_a_shared_directory_buffer_keeps_the_lease_record(tmp_path):
    store = CheckpointStore(tmp_path)
    writer = FileSampleBuffer(store, batch_size=1, index=shard_index(SHARD))
    assert not writer.can_own
    lease = IncarnationLease(store, SHARD, "inc-1", buffer=writer)
    lease.claim()
    assert not lease.owned
    IncarnationLease(store, SHARD, "inc-2").claim()
    assert lease.poll() is True
    with pytest.raises(SampleBufferError, match="object store"):
        writer.claim_owner("inc-1")


def _ledger(objects, **kw):
    return sd.StoreLedger(objects, "task-dispense", rows=4, fingerprint=b"f", holder="s-0",
                          incarnation=kw.pop("incarnation", "i1"), **kw)


def test_older_sampler_code_is_refused_by_a_new_plan_and_new_code_continues_an_old_plan(tmp_path, monkeypatch):
    objects = sd.DirObjects(tmp_path / "new")
    lease_of = {"s-0": "i1"}.get
    new = _ledger(objects, lease_of_holder=lease_of)
    assert new.open()["version"] == 3 and new.lease_of_holder is None  # the lease record frees no claim
    with monkeypatch.context() as m:  # a sampler of the code before ownership records
        m.setattr(sd, "PLAN_VERSION", 2)
        m.setattr(sd, "LEGACY_PLAN_VERSIONS", ())
        with pytest.raises(DispenseMismatch, match="version"):
            _ledger(objects).open()
    old_run = sd.DirObjects(tmp_path / "old")
    with monkeypatch.context() as m:
        m.setattr(sd, "PLAN_VERSION", 2)
        m.setattr(sd, "LEGACY_PLAN_VERSIONS", ())
        _ledger(old_run).open()
    continued = _ledger(old_run, incarnation="i2", lease_of_holder=lease_of)
    assert continued.open()["version"] == 2 and continued.plan_version == 2
    assert continued.lease_of_holder is lease_of  # the old plan's rule stays


def test_the_pool_fence_of_a_stalled_incarnation_frees_its_claims_at_once(tmp_path):
    """``fence_incarnation`` retires the incarnation in the ledger: nothing reads the
    lease record to find a fenced holder any more."""
    objects = _objects(tmp_path)
    dispense = sd.DirObjects(tmp_path)  # the same bucket, as the ledger reads it
    ledger = sd.StoreLedger(dispense, "task-dispense", rows=4, fingerprint=b"f", holder=SHARD, incarnation="inc-1")
    ledger.open()
    assert ledger.claim(0, 0) and ledger.claim(1, 0)
    writer, lease = _sampler(objects, "inc-1")
    path = R2DataPath(_records(objects), (dispense, "task-dispense"))
    record = path.fence_incarnation(SHARD, by="t-1")
    assert record["owner"]["created"] and lease.poll() is True
    states, _ = ledger.scan()
    assert [states[p].open_attempt() for p in (0, 1)] == [None, None] and ledger.is_retired()
    other = sd.StoreLedger(dispense, "task-dispense", rows=4, fingerprint=b"f", holder="s-1", incarnation="j1")
    other.open()
    assert other.claim(0, 1)


# ----------------------------------------------------------------- the engine sampler

STATE_ENV = "YETO_TEST_OWNER_FENCE_STATE"


def object_store_impl(path, *, environ=None, **_kwargs):
    """A store with an object API over a directory (``YETO_STORE_IMPL``)."""
    store = CheckpointStore(path)
    store.objects = DirectoryObjectStore(Path(path) / "objects")
    return store


def pool_fencing_reward(item, completion) -> float:
    """At call ``fence_at`` the pool fences the island's incarnation."""
    path = Path(os.environ[STATE_ENV])
    state = json.loads(path.read_text())
    state["n"] += 1
    path.write_text(json.dumps(state))
    if state["n"] == state["fence_at"]:
        elastic.fence_incarnation(object_store_impl(state["store"]), "ppu4", by="t-1")
    return 0.5


def test_an_engine_sampler_fenced_by_the_pool_writes_nothing_more_and_leaves_fenced(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_IMPL_ENV, "tests.test_shard_ownership_fence:object_store_impl")
    state = tmp_path / "state.json"
    monkeypatch.setenv(STATE_ENV, str(state))
    state.write_text(json.dumps({"n": 0, "fence_at": 5, "store": str(tmp_path / "store")}))
    objects = object_store_impl(tmp_path / "store").objects
    TrainerLease.claim(objects, incarnation="trainer-1")
    # the lease record is never polled in time (30 s): the shard index alone stops the writes
    spec = _sampler_spec(tmp_path, num_rollouts=4, reward="tests.test_shard_ownership_fence:pool_fencing_reward",
                         buffer_shards=["ppu4"])
    rounds = entry.run(spec, resource_sampler=lambda emit: None, incarnation="aaaa0001")
    assert rounds <= 2
    records = _shard(objects, "ppu4")
    assert [r["owner"]["incarnation"] if r["schema"] == OWNER_SCHEMA else r["sampler_round"]["rollout_id"]
            for r in records] == ["aaaa0001", 0, 0, "pool-fence:t-1"]  # no group, no data end after it
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    events = [json.loads(line) for line in tape.read_text().splitlines()]
    (owner,) = [e for e in events if e["event"] == "rl_shard_owner"]
    assert (owner["shard"], owner["incarnation"], owner["by"], owner["epoch"]) == ("ppu4", "aaaa0001", "sampler", 1)
    (leave,) = [e for e in events if e["event"] == "rl_sampler_leave"]
    assert leave["reason"] == "FENCED" and leave["superseded_by"] == "pool-fence:t-1"
    assert len(_trainer(objects, ("ppu4",))) == 2


# ----------------------------------------------------------------- damaged shard records

def _damage(objects, pos, record, shard=SHARD):
    data = record if isinstance(record, bytes) else (json.dumps(record) + "\n").encode()
    objects.write(ObjectIndex(objects, prefix=shard_prefix(shard)).key(pos), data)


def test_an_unreadable_shard_record_does_not_stop_the_groups_of_the_other_shards(tmp_path):
    """Before: one shard record that is not JSON made every refresh, and every
    reopen of the trainer buffer, raise; no shard was merged any more."""
    objects = _objects(tmp_path, trainer_lease=False)
    writers = {s: _writer(objects, s) for s in ("s-0", "s-1")}
    for s, writer in writers.items():
        for n in range(2):
            writer.append(SampleIndexEntry(f"{s}-g{n}", 0, "", 1, 1.0))
    trainer = _trainer(objects, ("s-0", "s-1"))
    events = []
    trainer.merge_events = lambda event, **fields: events.append(fields)
    for s, writer in writers.items():
        for n in range(2, 4):
            writer.append(SampleIndexEntry(f"{s}-g{n}", 0, "", 1, 2.0))
    _damage(objects, 2, b"{not json")
    assert trainer.refresh() == 7
    assert sorted(e.sample_id for e in trainer.entries()) == [
        "s-0-g0", "s-0-g1", "s-0-g3", "s-1-g0", "s-1-g1", "s-1-g2", "s-1-g3"]
    assert trainer.merge_counters()["bad_shard_records"] == 1
    assert [(f["reason"], f["shard"], f["shard_pos"]) for f in events] == [("bad_record", "s-0", 2)]
    restarted = _trainer(objects, ("s-0", "s-1"))  # reopens, and merges what comes after
    writers["s-0"].append(SampleIndexEntry("s-0-g4", 0, "", 1, 3.0))
    assert restarted.refresh() == 8 and restarted.shard_positions(8) == {"s-0": 5, "s-1": 4}


@pytest.mark.parametrize("damage", ["schema", "missing_field", "position", "not_an_object"])
def test_the_merger_never_writes_a_record_it_cannot_read_back(tmp_path, damage):
    """Before: the shard record was created in the merged index first and parsed
    after: the merged index was unreadable for good, also once the shard was repaired."""
    objects = _objects(tmp_path, trainer_lease=False)
    writer = _writer(objects)
    for n in range(3):
        writer.append(SampleIndexEntry(f"g{n}", 0, "", 1, 1.0))
    good = _shard(objects)[1]
    _damage(objects, 1, {"schema": {**good, "schema": "yeto.rl.sample-buffer/v0"},
                         "missing_field": {k: v for k, v in good.items() if k != "tokens"},
                         "position": {**good, "pos": 7}, "not_an_object": [1, 2]}[damage])
    trainer = _trainer(objects)
    assert [e.sample_id for e in trainer.entries()] == ["g0", "g2"]
    assert trainer.merge_counters()["bad_shard_records"] == 1
    for key in objects.list("sample-buffer/index/"):
        assert json.loads(objects.read(key))["schema"] == SCHEMA
    assert [e.sample_id for e in _trainer(objects).entries()] == ["g0", "g2"]


def test_a_missing_shard_position_is_reported_and_not_skipped(tmp_path, capsys):
    objects = _objects(tmp_path, trainer_lease=False)
    writer = _writer(objects)
    for n in range(4):
        writer.append(SampleIndexEntry(f"g{n}", 0, "", 1, 1.0))
    key = ObjectIndex(objects, prefix=shard_prefix(SHARD)).key(1)
    saved = objects.read(key)
    os.remove(Path(tmp_path) / key)  # deleted from outside
    trainer = _trainer(objects)
    events = []
    trainer.merge_events = lambda event, **fields: events.append(fields)
    assert len(trainer) == 1 and trainer.blocked_shards == {SHARD: 1}
    assert "has no record at position 1" in capsys.readouterr().out
    trainer.refresh()
    assert events == []  # reported once per position (at open, before the hook was set)
    objects.write(key, saved)  # the object is back
    assert trainer.refresh() == 4 and trainer.blocked_shards == {}
