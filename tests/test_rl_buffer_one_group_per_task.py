"""The trainer's reader trains one group per store-dispenser task (``TaskSlots``).

The store dispenser gives a task (a slot) to one sampler at a time, yet two
groups of one slot can reach the buffer without any fence seeing it. The reader
is the last line of defense: the first group of a slot that is accepted for
training holds the slot, every later group of it is passed over and counted
(``TakeReport.duplicate_task_groups``, driver event
``rl_buffer_duplicate_task_skipped``). The actors of the interleavings are the
real modules over one fake bucket (``tests.store_fault_fakes``).
"""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from tests.buffer_rollout_procs import sample
from tests.store_fault_fakes import (
    NAMESPACE,
    FaultS3,
    Sampler,
    client_error,
    drain,
    group,
    pool_fence,
    reader,
    shard_buffer,
    slot_of,
    trained_slots,
    trainer,
    write_prompts,
)
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.buffer_payload import GroupWriter
from yeto.rl.engine.merged_release import BufferLane
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.task_dispense import DISPENSE_POSITION_KEY


def _slots(items) -> list[int]:
    return [item.metadata[DISPENSE_POSITION_KEY] for item in items]


def _write(s3: FaultS3, shard: str, key: str, slot: int | None, *, attempt: int | None = 0, version: int = 1,
           incarnation: str = "i0", plan: str | None = "plan-a", **kwargs) -> str:
    """One group of ``shard`` for ``slot`` of dispenser ``plan`` (a store-dispenser group
    when ``attempt`` is set)."""
    record = {"sampler": shard, "incarnation": incarnation, "rollout_id": 0, "last": True}
    if plan is not None and attempt is not None:
        record["dispense_plan"] = plan
    if slot is not None:
        record["dispense_position"] = slot
    if attempt is not None:
        record["dispense_attempt"] = attempt
    s3.now += 1  # the merge orders the shards' records by their time
    return GroupWriter(shard_buffer(s3, shard), shard).write(
        group(0, version, **kwargs), group_key=key, sampler_round=record).sample_id


def _next_task(s3: FaultS3, sampler: Sampler) -> int:
    """``sampler`` takes and writes the next task (a take that ends on a passed-over
    group and finds nothing after it reports nothing: it waits, then raises)."""
    s3.now += 1
    items = sampler.draw()
    assert sampler.write(items) == 1
    return _slots(items)[0]


def _ids(entries) -> list[str]:
    return [entry.sample_id for entry in entries]


def _duplicates(reports) -> list[dict]:
    return [record for report in reports for record in report.duplicate_tasks]


# ------------------------------------------------------------------ the interleavings
def test_a_stall_between_still_holds_and_the_write_trains_the_task_once(tmp_path):
    """The holder stalls past its dispense lease after ``still_holds`` answered;
    another sampler ends the expired attempt, takes the task and writes it; the
    holder wakes up and writes too. No fence applies: both groups are in the buffer."""
    s3 = FaultS3()
    prompts = write_prompts(tmp_path, 4)
    holder = Sampler(s3, prompts, "s0", "z1")
    other = Sampler(s3, prompts, "s1", "o1")
    items = holder.draw()
    [slot] = _slots(items)

    def stall(_key: str) -> None:
        s3.now += 1000
        taken = other.draw()
        assert _slots(taken) == [slot] and other.write(taken) == 1

    s3.before("put", "payloads/s0-", stall)
    assert holder.write(items) == 1
    later = _next_task(s3, other)
    buffer = trainer(s3)
    assert [slot_of(e) for e in buffer.entries()] == [slot, slot, later]  # both are in the buffer
    trained, reports = drain(reader(buffer))
    assert [slot_of(e) for e in trained] == [slot, later]
    [record] = _duplicates(reports)
    first, second, _ = buffer.entries()
    assert record == {"slot": slot, "position": 1, "sample_id": second.sample_id, "first_position": 0,
                      "first_sample_id": first.sample_id, "first_trained_here": True,
                      "plan": holder.prompts.plan_id,
                      "attempt": second.sampler_round["dispense_attempt"]}
    assert first.sampler_round["dispense_plan"] == second.sampler_round["dispense_plan"]
    assert sum(r.duplicate_task_groups for r in reports) == 1
    # settled by whichever attempt won the done/ create: the task is not lost
    assert trained_slots(s3) == {slot, later}


def test_a_listing_taken_before_a_late_settlement_trains_the_task_once(tmp_path):
    """The scanner lists while the claim looks expired, the holder wakes up, writes
    and settles ``trained``; the scanner then ends the attempt from its old listing
    and claims the next one. Both creates win: two groups."""
    s3 = FaultS3()
    prompts = write_prompts(tmp_path, 4)
    holder = Sampler(s3, prompts, "s0", "z1")
    scanner = Sampler(s3, prompts, "s1", "o1")
    items = holder.draw()
    [slot] = _slots(items)
    s3.now += 1000
    stale = scanner.objects.list_meta(NAMESPACE + "/")
    assert holder.write(items) == 1
    scanner.prompts.ledger.objects.list_meta = lambda prefix: stale
    taken = scanner.draw()
    assert _slots(taken) == [slot]
    s3.now += 1
    assert scanner.write(taken) == 1
    assert len(s3.keys("sample-buffer/payloads/")) == 2
    scanner.prompts.ledger.objects.list_meta = scanner.objects.list_meta
    later = _next_task(s3, holder)
    trained, reports = drain(reader(trainer(s3)))
    assert [slot_of(e) for e in trained] == [slot, later] and trained[0].sample_id.startswith("s0-")
    assert sum(r.duplicate_task_groups for r in reports) == 1


def test_a_group_written_but_not_settled_and_given_out_again_by_a_retire_is_trained_once(tmp_path):
    s3 = FaultS3()
    prompts = write_prompts(tmp_path, 4)
    sampler = Sampler(s3, prompts, "s0", "z1")
    items = sampler.draw()
    [slot] = _slots(items)
    s3.fail("create", "done/", client_error("SlowDown", "PutObject"))
    with pytest.raises(ClientError):
        sampler.write(items)
    s3.now += 120
    pool_fence(s3, "s0")  # the pool retires the node: its open claim is released
    replacement = Sampler(s3, prompts, "s1", "r1")
    again = replacement.draw()
    assert _slots(again) == [slot] and replacement.write(again) == 1
    later = _next_task(s3, replacement)
    trained, reports = drain(reader(trainer(s3)))
    assert [slot_of(e) for e in trained] == [slot, later]
    assert sum(r.duplicate_task_groups for r in reports) == 1


def test_every_task_of_a_run_with_such_duplicates_is_trained_exactly_once(tmp_path):
    s3 = FaultS3()
    prompts = write_prompts(tmp_path, 6)
    holder = Sampler(s3, prompts, "s0", "z1")
    other = Sampler(s3, prompts, "s1", "o1")
    for _ in range(2):  # two tasks written twice (the stall), then the rest once
        items = holder.draw()

        def stall(_key: str) -> None:
            s3.now += 1000
            other.write(other.draw())

        s3.before("put", "payloads/s0-", stall)
        holder.write(items)
    for _ in range(4):
        s3.now += 1
        other.write(other.draw())
    trained, reports = drain(reader(trainer(s3)))
    assert Counter(slot_of(e) for e in trained) == Counter(range(6))
    assert sum(r.duplicate_task_groups for r in reports) == 2 and trained_slots(s3) == set(range(6))


# ------------------------------------------------------------------ restarts
@pytest.mark.parametrize("cut_after", [0, 1, 2])
def test_a_reader_restarted_at_any_cut_judges_the_duplicate_as_the_first_reader_did(cut_after):
    """The slots held below the restored watermark come from the index, not from the
    process that trained them."""
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7), _write(s3, "s1", "b", 8), _write(s3, "s1", "c", 7, attempt=1),
           _write(s3, "s0", "d", 9)]
    whole, _reports = drain(reader(trainer(s3)))
    assert _ids(whole) == [ids[0], ids[1], ids[3]]
    first = reader(trainer(s3))
    for rollout_id in range(cut_after):
        first.take(rollout_id, 1, current_version=1)
    watermark = first.watermark_before(cut_after)
    assert watermark == cut_after
    again = reader(trainer(s3))  # a new process: it never saw the takes above
    again.reset(watermark)
    rest, reports = drain(again, start=cut_after)
    assert _ids(rest) == [ids[0], ids[1], ids[3]][cut_after:]
    assert [d["sample_id"] for d in _duplicates(reports)] == [ids[2]]
    assert _duplicates(reports)[0]["first_sample_id"] == ids[0]
    # the first group lies below the restored watermark: this process did not train it itself
    assert _duplicates(reports)[0]["first_trained_here"] is (cut_after == 0)
    assert sum(r.duplicate_task_unverified_groups for r in reports) == (0 if cut_after == 0 else 1)


def test_a_round_rolled_back_to_the_watermark_gives_its_slots_back():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7), _write(s3, "s1", "b", 7, attempt=1), _write(s3, "s0", "c", 8)]
    batch_reader = reader(trainer(s3))
    taken, _report = batch_reader.take(0, 1, current_version=1)
    assert [position for position, _ in taken] == [0]
    batch_reader.reset(0)  # the round is not in the restored cut: selected again
    trained, reports = drain(batch_reader)
    assert _ids(trained) == [ids[0], ids[2]]  # the first group again, not passed over for itself
    assert [d["sample_id"] for d in _duplicates(reports)] == [ids[1]]


def test_a_take_that_raises_holds_no_slot():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7), _write(s3, "s1", "b", 7, attempt=1)]
    batch_reader = reader(trainer(s3))
    s3.fail("get", "sample-buffer/payloads", client_error("SlowDown", "GetObject"))
    with pytest.raises(ClientError):
        batch_reader.take(0, 1, current_version=1)  # the first group is selected, never returned
    assert batch_reader.tasks.owner == {}
    batch_reader.reset(0)
    trained, _reports = drain(batch_reader)
    assert _ids(trained) == [ids[0]]


# ------------------------------------------------------------------ a slot is held by a trained group only
def test_a_first_group_passed_over_as_too_old_does_not_hold_its_task():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7, version=1), _write(s3, "s1", "b", 7, attempt=1, version=9)]
    batch_reader = reader(trainer(s3))
    taken, report = batch_reader.take(0, 1, current_version=12)  # max_policy_age 8: version 1 is too old
    assert _ids(batch_reader.buffer.entries()[p] for p, _ in taken) == [ids[1]]
    assert report.stale_groups == 1 and report.duplicate_task_groups == 0


@pytest.mark.parametrize("fault", ["nonfinite", "small"])
def test_a_first_group_dropped_for_its_payload_does_not_hold_its_task_even_in_the_same_selection(fault):
    s3 = FaultS3()
    bad = {"reward": float("nan")} if fault == "nonfinite" else {"n": 1}
    ids = [_write(s3, "s0", "a", 7, **bad), _write(s3, "s1", "b", 7, attempt=1), _write(s3, "s0", "c", 8)]
    batch_reader = reader(trainer(s3))
    taken, report = batch_reader.take(0, 2, current_version=1)  # the two groups of slot 7 in one selection
    assert _ids(batch_reader.buffer.entries()[p] for p, _ in taken) == ids[1:]
    assert report.duplicate_task_groups == 0
    assert (report.nonfinite_reward_groups, report.dropped_small_groups) == ((1, 0) if fault == "nonfinite" else (0, 1))


class _Gate:
    """An incarnation gate that fences every group of incarnation ``zombie``."""

    def verdict(self, entry, position):
        return "fenced" if (entry.sampler_round or {}).get("incarnation") == "zombie" else None


@pytest.mark.parametrize("restart", [False, True])
def test_a_fenced_incarnations_group_does_not_hold_its_task_before_or_after_a_restart(restart):
    """The usual order after an in-place restart: the fenced process writes late, the
    task (released at the fence) is written again by its successor."""
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7, incarnation="zombie"), _write(s3, "s0", "b", 8),
           _write(s3, "s1", "c", 7, attempt=1)]
    batch_reader = reader(trainer(s3), incarnation_gate=_Gate())
    if restart:  # the cut after the fenced group and the next one; a new process continues there
        batch_reader.take(0, 1, current_version=1)
        watermark = batch_reader.watermark_before(1)
        assert watermark == 2
        batch_reader = reader(trainer(s3), incarnation_gate=_Gate())
        batch_reader.reset(watermark)
    trained, reports = drain(batch_reader, start=1 if restart else 0)
    assert _ids(trained) == (ids[2:] if restart else ids[1:])
    assert sum(r.duplicate_task_groups for r in reports) == 0


def test_after_a_restart_a_first_group_below_the_watermark_holds_its_task_whatever_became_of_it():
    """The one judgement a restart changes (``TaskSlots``): the index does not say that
    the first group was passed over as too old, so the restarted reader counts it as
    trained. The task is then trained zero times, never twice."""
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7, version=1), _write(s3, "s0", "b", 8, version=9),
           _write(s3, "s1", "c", 7, attempt=1, version=9), _write(s3, "s0", "d", 9, version=9)]
    first = reader(trainer(s3))
    taken, report = first.take(0, 1, current_version=12)
    assert report.stale_groups == 1 and [p for p, _ in taken] == [1]
    again = reader(trainer(s3))
    again.reset(first.watermark_before(1))
    taken, report = again.take(1, 1, current_version=12)
    assert _ids(again.buffer.entries()[p] for p, _ in taken) == [ids[3]] and report.duplicate_task_groups == 1
    assert report.duplicate_task_unverified_groups == 1 and not report.duplicate_tasks[0]["first_trained_here"]
    assert report.metrics()["sample_buffer/duplicate_task_unverified_groups"] == 1.0
    taken, report = first.take(1, 1, current_version=12)  # the reader that passed it over trains the second
    assert _ids(first.buffer.entries()[p] for p, _ in taken) == [ids[2]] and report.duplicate_task_groups == 0


# ------------------------------------------------------------------ the short-round fill and the carried group
def test_a_duplicate_met_while_a_short_round_is_filled_is_passed_over():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7), _write(s3, "s1", "b", 7, attempt=1), _write(s3, "s0", "c", 8)]
    batch_reader = reader(trainer(s3), group_size=4)  # groups of 2: a round holds two
    taken, report = batch_reader.take(0, 1, current_version=1)
    assert _ids(batch_reader.buffer.entries()[p] for p, _ in taken) == [ids[0], ids[2]]
    assert report.duplicate_task_groups == 1 and report.end == 3 and report.padded_samples == 0


def test_a_group_carried_over_to_the_next_round_holds_its_task_once_it_is_trained():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 6, n=2), _write(s3, "s0", "b", 7, n=3),  # 2 + 3 > 4: b is carried over
           _write(s3, "s1", "c", 7, attempt=1, n=2), _write(s3, "s0", "d", 8, n=1)]
    batch_reader = reader(trainer(s3), group_size=4)
    taken, report = batch_reader.take(0, 1, current_version=1)
    assert [p for p, _ in taken] == [0] and report.carried_over_groups == 1
    assert batch_reader.tasks.owner == {("plan-a", 6): 0}  # the carried group is not trained yet
    taken, report = batch_reader.take(1, 1, current_version=1)
    assert _ids(batch_reader.buffer.entries()[p] for p, _ in taken) == [ids[1]]
    assert report.duplicate_task_groups == 1
    assert batch_reader.tasks.owner == {("plan-a", 6): 0, ("plan-a", 7): 1}


def test_a_carried_group_that_ages_out_while_it_waits_gives_its_task_to_the_next_group():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 6, n=2, version=9), _write(s3, "s0", "b", 7, n=3, version=4),
           _write(s3, "s1", "c", 7, attempt=1, n=3, version=9)]
    batch_reader = reader(trainer(s3), group_size=4)
    batch_reader.take(0, 1, current_version=12)  # floor 4: b is carried over
    taken, report = batch_reader.take(1, 1, current_version=13)  # floor 5: b is too old now
    assert _ids(batch_reader.buffer.entries()[p] for p, _ in taken) == [ids[2]]
    assert report.stale_groups == 1 and report.duplicate_task_groups == 0


def test_a_carried_group_and_a_duplicate_selected_after_a_raise_train_the_task_once():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 6, n=2), _write(s3, "s0", "b", 7, n=3),  # 2 + 3 > 4: b is carried over
           _write(s3, "s1", "c", 7, attempt=1, n=2), _write(s3, "s0", "d", 8, n=2)]
    batch_reader = reader(trainer(s3), group_size=4)
    batch_reader.take(0, 1, current_version=1)
    s3.fail("get", "sample-buffer/payloads", client_error("SlowDown", "GetObject"))
    with pytest.raises(ClientError):
        batch_reader.take(1, 2, current_version=1)  # the carried group is chosen, the next read fails
    trained, reports = drain(batch_reader, start=1)
    assert _ids(trained) == [ids[1], ids[3]] and [d["sample_id"] for d in _duplicates(reports)] == [ids[2]]


def test_a_first_group_with_a_damaged_payload_does_not_hold_its_task():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7), _write(s3, "s1", "b", 7, attempt=1), _write(s3, "s0", "c", 7, attempt=2),
           _write(s3, "s0", "d", 8)]
    s3.objects[f"sample-buffer/payloads/{ids[0]}.bin"] = (b"{cut", s3.now)
    batch_reader = reader(trainer(s3))
    taken, report = batch_reader.take(0, 2, current_version=1)
    assert _ids(batch_reader.buffer.entries()[p] for p, _ in taken) == [ids[1], ids[3]]
    assert (report.damaged_payload_groups, report.duplicate_task_groups) == (1, 1)
    assert report.duplicate_tasks[0]["first_sample_id"] == ids[1]


def test_a_take_that_ends_without_a_round_names_the_duplicates_it_passed_over():
    from yeto.rl.engine import buffer_payload as bp

    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7), _write(s3, "s1", "b", 7, attempt=1)]
    shard_buffer(s3, "s0").mark_data_end()
    shard_buffer(s3, "s1").mark_data_end()
    batch_reader = reader(trainer(s3))
    batch_reader.take(0, 1, current_version=1)
    with pytest.raises(br.SampleBufferExhausted, match="1 of a task already trained") as failure:
        batch_reader.take(1, 1, current_version=1)
    assert bp.passed_over_of(str(failure.value)) == {"duplicate_task": 1}
    assert [d["sample_id"] for d in failure.value.passed_over["duplicate_tasks"]] == [ids[1]]


# ------------------------------------------------------------------ what is a task
def test_the_same_slot_of_a_second_dispenser_namespace_is_a_task_of_its_own():
    """The run's dispenser is opened under a new namespace on the same buffer (more
    data, another seed, more passes: any plan change needs one). Its slot numbers
    start at 0 again; the round record names the dispenser."""
    s3 = FaultS3()
    first = [_write(s3, "s0", f"a{slot}", slot, plan="plan-a") for slot in range(3)]
    second = [_write(s3, "s0", f"b{slot}", slot, plan="plan-b") for slot in range(3)]
    again = _write(s3, "s1", "b1-again", 1, attempt=1, plan="plan-b")
    last = _write(s3, "s0", "b3", 3, plan="plan-b")
    trained, reports = drain(reader(trainer(s3)))
    assert _ids(trained) == first + second + [last]
    [record] = _duplicates(reports)
    assert (record["sample_id"], record["first_sample_id"], record["plan"]) == (again, second[1], "plan-b")


def test_a_restart_between_two_namespaces_does_not_starve_the_trainer():
    s3 = FaultS3()
    for slot in range(3):
        _write(s3, "s0", f"a{slot}", slot, plan="plan-a")
    old = reader(trainer(s3))
    drain(old)
    watermark = old.watermark_before(10**6)
    assert watermark == 3
    second = [_write(s3, "s0", f"b{slot}", slot, plan="plan-b") for slot in range(3)]
    new = reader(trainer(s3))
    new.reset(watermark)
    trained, reports = drain(new, start=100)
    assert _ids(trained) == second and not _duplicates(reports)


def test_two_real_dispensers_on_one_buffer_have_two_plan_ids_and_both_are_trained(tmp_path, monkeypatch):
    s3 = FaultS3()
    prompts = write_prompts(tmp_path, 2)
    first = Sampler(s3, prompts, "s0", "i0")
    slots = [_next_task(s3, first) for _ in range(2)]
    original = sd.StoreDispensedPrompts.open
    monkeypatch.setattr(sd.StoreDispensedPrompts, "open", classmethod(
        lambda cls, path, config, **kwargs: original(path, {**config, "namespace": "task-dispense-2"}, **kwargs)))
    second = Sampler(s3, prompts, "s1", "i0")  # the same rows again, under another namespace
    slots += [_next_task(s3, second) for _ in range(2)]
    assert slots == [0, 1, 0, 1] and first.prompts.plan_id != second.prompts.plan_id
    trained, reports = drain(reader(trainer(s3)))
    assert [slot_of(e) for e in trained] == [0, 1, 0, 1] and not _duplicates(reports)


def test_the_plan_id_names_the_namespace_and_the_plan_not_the_sampler(tmp_path):
    s3 = FaultS3()
    prompts = write_prompts(tmp_path, 3)
    a, b = Sampler(s3, prompts, "s0", "i0"), Sampler(s3, prompts, "s1", "i7", lease_s=60.0, max_redo=5)
    import hashlib

    digest = hashlib.sha256(s3.objects[f"{NAMESPACE}/plan.json"][0]).hexdigest()[:16]
    assert a.prompts.plan_id == b.prompts.plan_id == f"{NAMESPACE}:{digest}"


def test_records_with_an_attempt_and_no_plan_are_one_task_per_slot():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7, plan=None), _write(s3, "s1", "b", 7, attempt=1, plan=None),
           _write(s3, "s0", "c", 7, plan="plan-b")]
    trained, reports = drain(reader(trainer(s3)))
    assert _ids(trained) == [ids[0], ids[2]] and [d["sample_id"] for d in _duplicates(reports)] == [ids[1]]
    assert "plan" not in _duplicates(reports)[0]


def test_a_reset_forward_past_positions_another_process_trained_keeps_their_slots():
    """One reader object: a round, then a reset to a later watermark (the restored cut
    of another process that had read further). The positions in between were trained
    by that process."""
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 5), _write(s3, "s0", "b", 6), _write(s3, "s0", "c", 7),
           _write(s3, "s1", "d", 7, attempt=1), _write(s3, "s0", "e", 8)]
    batch_reader = reader(trainer(s3))
    batch_reader.take(0, 1, current_version=1)  # position 0
    batch_reader.reset(3)  # the restored cut is at 3: positions 1 and 2 were trained elsewhere
    trained, reports = drain(batch_reader, start=1)
    assert _ids(trained) == [ids[4]]
    assert [(d["sample_id"], d["first_trained_here"]) for d in _duplicates(reports)] == [(ids[3], False)]


def test_the_two_passes_of_a_row_are_two_tasks_and_both_are_trained(tmp_path, monkeypatch):
    """A slot is one row in one pass (``SlotMap``: rows x passes slots), and the
    record's ``dispense_position`` is the slot, not the row."""
    original = sd.StoreDispensedPrompts.open
    monkeypatch.setattr(sd.StoreDispensedPrompts, "open", classmethod(
        lambda cls, path, config, **kwargs: original(path, {**config, "passes": 2}, **kwargs)))
    s3 = FaultS3()
    sampler = Sampler(s3, write_prompts(tmp_path, 2), "s0", "i0")
    rows = []
    for _ in range(4):
        items = sampler.draw()
        rows += [item.metadata["row"] for item in items]
        s3.now += 1
        assert sampler.write(items) == 1
    assert sorted(rows) == [0, 0, 1, 1]
    trained, reports = drain(reader(trainer(s3, ("s0",))))
    assert sorted(slot_of(e) for e in trained) == [0, 1, 2, 3]
    assert sum(r.duplicate_task_groups for r in reports) == 0


def test_the_engine_sampler_records_the_attempt_only_for_a_store_dispenser_prompt(tmp_path):
    from yeto.rl.engine import engine_sampler as es
    from yeto.rl.engine.sampler_resume import SamplerRound

    assert es.DISPENSE_ATTEMPT_KEY == sd.ATTEMPT_KEY
    s3 = FaultS3()
    sampler = Sampler(s3, write_prompts(tmp_path, 2), "s0", "i0")
    sampler.write(sampler.draw())
    [entry] = sampler.buffer.entries()
    assert entry.sampler_round["dispense_position"] == 0 and entry.sampler_round["dispense_attempt"] == 0
    assert entry.sampler_round["dispense_plan"] == sampler.prompts.plan_id
    assert br.task_key(entry) == (sampler.prompts.plan_id, 0)
    sampler.round.prompts = SimpleNamespace()  # prompts of any other source: no plan in the record
    assert sampler.round._dispense_plan() is None
    syncer = es.PromptItem(prompt_id="row3", prompt="p", label=None, metadata={DISPENSE_POSITION_KEY: 3})
    assert es.dispense_position(syncer) == 3 and es.dispense_attempt(syncer) is None
    record = SamplerRound("s0", "i0", 0, dispense_position=3).record()  # what a syncer-dispensed group records
    assert "dispense_attempt" not in record and "dispense_plan" not in record
    assert br.task_key(SimpleNamespace(sampler_round=record)) is None


def test_groups_of_the_syncers_dispenser_are_not_judged():
    """Its round record has a position and no attempt: the same position twice
    (whatever it means there) is trained twice, as before."""
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7, attempt=None), _write(s3, "s1", "b", 7, attempt=None),
           _write(s3, "s0", "c", None, attempt=None)]
    trained, reports = drain(reader(trainer(s3)))
    assert _ids(trained) == ids and sum(r.duplicate_task_groups for r in reports) == 0


def test_a_miles_sampler_round_with_the_syncers_positions_is_not_judged(tmp_path):
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2)
    writer = br.SamplerRoundWriter(buffer, "s0", incarnation="i0")
    writer.write_round(0, [("g0", group(0)), ("g1", group(1))], None, positions=[5, 5])
    assert [br.task_key(entry) for entry in buffer.entries()] == [None, None]
    trained, reports = drain(reader(buffer))
    assert len(trained) == 2 and sum(r.duplicate_task_groups for r in reports) == 0


def test_the_reader_can_be_told_to_train_every_group():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7), _write(s3, "s1", "b", 7, attempt=1)]
    trained, reports = drain(reader(trainer(s3), one_group_per_task=False))
    assert _ids(trained) == ids and sum(r.duplicate_task_groups for r in reports) == 0


def test_a_reader_without_a_group_size_passes_the_duplicate_over_and_still_fills_the_round():
    s3 = FaultS3()
    ids = [_write(s3, "s0", "a", 7), _write(s3, "s1", "b", 7, attempt=1), _write(s3, "s0", "c", 8)]
    batch_reader = reader(trainer(s3), group_size=None)
    taken, report = batch_reader.take(0, 2, current_version=1)
    assert _ids(batch_reader.buffer.entries()[p] for p, _ in taken) == [ids[0], ids[2]]
    assert report.groups == 2 and report.duplicate_task_groups == 1
    assert report.metrics()["sample_buffer/duplicate_task_groups"] == 1.0
    assert sum(report.source_groups.values()) == 2  # the duplicate is in no accounting


def test_a_lane_reader_counts_an_earlier_group_of_another_lane_as_holding_its_task(tmp_path):
    """Synced trainers read one lane each; what the other lane's trainer did with a
    group is not known here, so an earlier group of the other lane holds its slot."""
    def lane_reader(index: int):
        buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2, lane=BufferLane(index, 2))
        return reader(buffer)

    writer = GroupWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2), "s0")
    for key, slot in (("a", 7), ("b", 7), ("c", 8), ("d", 9)):  # positions 0..3: lanes 0, 1, 0, 1
        writer.write(group(0), group_key=key, sampler_round={
            "sampler": "s0", "incarnation": "i0", "rollout_id": 0, "last": True,
            "dispense_position": slot, "dispense_attempt": 0})
    trained, reports = drain(lane_reader(0))
    assert [e.sample_id for e in trained] == ["s0-a", "s0-c"] and not _duplicates(reports)
    trained, reports = drain(lane_reader(1))
    assert [e.sample_id for e in trained] == ["s0-d"]
    assert [(d["sample_id"], d["first_sample_id"]) for d in _duplicates(reports)] == [("s0-b", "s0-a")]


# ------------------------------------------------------------------ to the driver's tape
def test_the_round_metadata_takes_the_duplicate_task_records():
    from yeto.rl.adapters.miles import rollout_meta_hook as hook

    sent = []
    record = {"slot": 7, "position": 4, "sample_id": "s1-b", "first_position": 1, "first_sample_id": "s0-a"}
    original, hook.put_to_sink = hook.put_to_sink, lambda item, sink=None: sent.append(item)
    try:
        hook.record_round_metadata(None, 3, buffer_duplicate_task_groups=1, buffer_duplicate_tasks=[record])
        with pytest.raises(RuntimeError, match="list of mappings"):
            hook.record_round_metadata(None, 3, buffer_duplicate_tasks=[7])
    finally:
        hook.put_to_sink = original
    assert sent[0]["buffer_duplicate_task_groups"] == 1 and sent[0]["buffer_duplicate_tasks"] == [record]


def test_reader_round_reports_its_duplicate_tasks_as_round_metadata(monkeypatch):
    from tests.test_rl_buffer_rollout import _import_plugin_with_stub_miles

    calls = []
    plugin = _import_plugin_with_stub_miles(monkeypatch, {"all": lambda args, samples, source: None})
    monkeypatch.setattr(plugin, "record_round_metadata", lambda args, rid, **counts: calls.append(counts))
    monkeypatch.setattr(plugin, "Sample", SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda **kw: kw)
    monkeypatch.setattr(plugin, "offpolicy_source_event", lambda *a, **kw: {})
    monkeypatch.setattr(plugin, "flat_metrics", lambda source: {})
    record = {"slot": 7, "position": 1, "sample_id": "s1-b", "first_position": 0, "first_sample_id": "s0-a",
              "first_trained_here": False}
    report = br.TakeReport(rollout_id=2, groups=1, stale_groups=0, start=0, end=3,
                           duplicate_task_groups=1, duplicate_tasks=[record])
    round_ = plugin._BufferRound.__new__(plugin._BufferRound)
    round_.args = SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path="all")
    round_.groups, round_.samples_per_group = 1, 2
    round_.reader = SimpleNamespace(
        take=lambda rid, n, current_version: ([(2, [sample(0, 0, 4), sample(1, 0, 4)])], report))
    out = round_(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=4))
    assert calls[0]["buffer_duplicate_task_groups"] == 1 and calls[0]["buffer_duplicate_tasks"] == [record]
    assert calls[0]["buffer_duplicate_task_unverified_groups"] == 1
    assert out["metrics"]["sample_buffer/duplicate_task_groups"] == 1.0


def test_driver_tapes_the_duplicate_tasks_a_round_passed_over(tmp_path):
    import dataclasses

    from tests.test_rl_policy_age import _driver

    record = {"slot": 7, "position": 4, "sample_id": "s1-b", "first_position": 1, "first_sample_id": "s0-a"}
    driver = _driver(tmp_path, 1)
    original = driver.rollout.generate

    def gen(rid, **kw):
        extra = {"buffer_duplicate_task_groups": 1, "buffer_duplicate_task_unverified_groups": 1,
                 "buffer_duplicate_tasks": [record]} if rid == 1 else {}
        return dataclasses.replace(original(rid, **kw), buffer_stale_groups=0, **extra)

    driver.rollout.generate = gen
    driver.run()
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    skipped = [e for e in events if e["event"] == "rl_buffer_duplicate_task_skipped"]
    assert [(e["rollout_id"], e["duplicate_groups"], e["unverified_groups"], e["tasks"]) for e in skipped] == [
        (1, 1, 1, [record])]
