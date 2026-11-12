"""The trainer's reader (``BatchReader``) and the faults of one payload.

A payload that does not exist or is not a group is a fault of the data: the
reader passes the group over, counts it (``TakeReport.damaged_payload_groups``,
driver event ``rl_buffer_damaged_payload_skipped``) and trains the next one, on
this take and after every restart from the cut's watermark. A read the store did
not answer is not: the take raises, and taking again on the same reader loses no
group.
"""

from __future__ import annotations

import json
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

from tests.buffer_rollout_procs import sample
from tests.store_fault_fakes import FaultS3, client_error, drain, group, reader, shard_buffer, trainer
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine import buffer_payload as bp
from yeto.rl.engine.buffer_payload import GroupWriter
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, SampleBufferError

PAYLOADS = "sample-buffer/payloads"
DAMAGES = ["missing", "truncated", "schema", "not_an_object"]


def _fill(s3: FaultS3, shard: str, count: int, *, start: int = 0, **kwargs) -> list[str]:
    writer = GroupWriter(shard_buffer(s3, shard), shard)
    return [writer.write(group(g, **kwargs), group_key=f"g{g}",
                         sampler_round={"sampler": shard, "incarnation": "i0", "rollout_id": g,
                                        "last": True, "dispense_position": 100 + g}).sample_id
            for g in range(start, start + count)]


def _ids(entries) -> list[str]:
    return [entry.sample_id for entry in entries]


def _damage(s3: FaultS3, sample_id: str, damage: str) -> None:
    key = f"{PAYLOADS}/{sample_id}.bin"
    data = s3.objects[key][0]
    if damage == "missing":
        del s3.objects[key]
    elif damage == "truncated":
        s3.objects[key] = (data[:len(data) // 2], s3.now)
    elif damage == "schema":
        s3.objects[key] = (json.dumps({"schema": "other/v9", "samples": []}).encode(), s3.now)
    elif damage == "not_an_object":
        s3.objects[key] = (b"[]", s3.now)
    else:
        raise AssertionError(damage)


# ------------------------------------------------------------------ damaged payloads
@pytest.mark.parametrize("damage", DAMAGES)
def test_a_damaged_payload_is_passed_over_and_the_next_group_is_trained(damage):
    s3 = FaultS3()
    ids = _fill(s3, "s0", 3)
    _damage(s3, ids[1], damage)
    trained, reports = drain(reader(trainer(s3, ("s0",))))
    assert _ids(trained) == [ids[0], ids[2]]
    assert [r.damaged_payload_groups for r in reports] == [0, 1]
    [record] = reports[1].damaged_payloads
    assert (record["position"], record["sample_id"], record["dispense_position"]) == (1, ids[1], 101)
    assert record["payload_ref"].endswith(f"{ids[1]}.bin") and record["reason"]
    assert (reports[1].start, reports[1].end) == (1, 3)  # the damaged position is consumed
    assert reports[1].metrics()["sample_buffer/damaged_payload_groups"] == 1.0
    assert reports[1].behavior_versions == {1: 1} and sum(reports[1].source_groups.values()) == 1


@pytest.mark.parametrize("damage", DAMAGES)
def test_a_trainer_restarted_at_the_cut_passes_over_the_same_damaged_payload(damage):
    """What crash-looped before: the relaunched trainer restores the cut and meets
    the same payload at the same position."""
    s3 = FaultS3()
    ids = _fill(s3, "s0", 3)
    _damage(s3, ids[1], damage)
    first = reader(trainer(s3, ("s0",)))
    first.take(0, 1, current_version=1)
    watermark = first.watermark_before(1)
    assert watermark == 1
    for _ in range(2):
        again = reader(trainer(s3, ("s0",)))
        again.reset(watermark)
        taken, report = again.take(1, 1, current_version=1)
        assert [position for position, _ in taken] == [2] and report.damaged_payload_groups == 1
        assert again.watermark_before(2) == 3


@pytest.mark.parametrize("damage", DAMAGES)
def test_a_damaged_payload_is_passed_over_by_a_reader_without_a_group_size(damage):
    s3 = FaultS3()
    ids = _fill(s3, "s0", 4)
    _damage(s3, ids[1], damage)
    batch_reader = reader(trainer(s3, ("s0",)), group_size=None)
    taken, report = batch_reader.take(0, 3, current_version=1)  # still three groups: one more is selected
    assert [position for position, _ in taken] == [0, 2, 3]
    assert report.groups == 3 and report.damaged_payload_groups == 1 and report.end == 4


def test_a_damaged_payload_met_while_a_short_round_is_filled_is_passed_over():
    """``_fill``: the round is topped up group by group; a damaged one is skipped."""
    s3 = FaultS3()
    ids = _fill(s3, "s0", 4)
    _damage(s3, ids[1], "truncated")
    batch_reader = reader(trainer(s3, ("s0",)), group_size=4)  # groups of 2: a round holds two
    taken, report = batch_reader.take(0, 1, current_version=1)
    assert [position for position, _ in taken] == [0, 2]
    assert report.damaged_payload_groups == 1 and report.damaged_payloads[0]["sample_id"] == ids[1]
    assert report.end == 3 and report.padded_samples == 0


def test_a_buffer_whose_every_group_is_damaged_ends_the_run_and_says_so():
    s3 = FaultS3()
    ids = _fill(s3, "s0", 2)
    for sample_id in ids:
        _damage(s3, sample_id, "missing")
    shard_buffer(s3, "s0").mark_data_end()
    with pytest.raises(br.SampleBufferExhausted, match="2 with a damaged payload"):
        reader(trainer(s3, ("s0",))).take(0, 1, current_version=1)


def test_a_missing_payload_file_of_a_shared_directory_buffer_is_passed_over(tmp_path):
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2)
    writer = GroupWriter(buffer, "s0")
    entries = [writer.write(group(g), group_key=f"g{g}") for g in range(3)]
    (tmp_path / "sample-buffer" / entries[1].payload_ref).unlink()
    taken, report = reader(buffer).take(0, 2, current_version=1)
    assert [position for position, _ in taken] == [0, 2]
    assert report.damaged_payloads[0]["reason"] == "payload is missing"


@pytest.mark.parametrize("data, reason", [
    (b"", "does not decode"), (b"{not json", "does not decode"), (b"\xff\xfe", "does not decode"),
    (b"[]", "not an object"), (b'"text"', "not an object"),
    (json.dumps({"schema": "other/v9", "samples": []}).encode(), "schema"),
    (json.dumps({"schema": bp.PAYLOAD_SCHEMA}).encode(), "not a list of objects"),
    (json.dumps({"schema": bp.PAYLOAD_SCHEMA, "samples": {"a": 1}}).encode(), "not a list of objects"),
    (json.dumps({"schema": bp.PAYLOAD_SCHEMA, "samples": [1]}).encode(), "not a list of objects"),
    (json.dumps({"schema": bp.PAYLOAD_SCHEMA, "samples": [{"x": {"__ndarray__": "AAAA"}}]}).encode(),
     "does not decode"),
])
def test_bytes_that_are_not_a_payload_decode_to_a_damaged_payload_error(data, reason):
    with pytest.raises(bp.DamagedPayload, match=reason) as failure:
        bp.decode_group(data)
    assert isinstance(failure.value, bp.SampleBufferRolloutError)  # what callers caught before


# ------------------------------------------------------------------ the store did not answer
@pytest.mark.parametrize("error", [
    client_error("SlowDown", "GetObject"), client_error("InternalError", "GetObject"),
    client_error("503", "GetObject"), client_error("AccessDenied", "GetObject"),
    EndpointConnectionError(endpoint_url="https://store"),
    ReadTimeoutError(endpoint_url="https://store"), ConnectionResetError("reset"), TimeoutError("slow"),
], ids=["throttle", "internal", "503", "403", "connect", "read_timeout", "reset", "timeout"])
def test_a_payload_read_the_store_failed_raises_and_is_never_counted_as_a_damaged_payload(error):
    s3 = FaultS3()
    ids = _fill(s3, "s0", 2)
    batch_reader = reader(trainer(s3, ("s0",)))
    s3.fail("get", PAYLOADS, error)
    with pytest.raises(type(error)):
        batch_reader.take(0, 1, current_version=1)
    trained, reports = drain(batch_reader)  # the store answers again: nothing was dropped
    assert _ids(trained) == ids and sum(r.damaged_payload_groups for r in reports) == 0


def test_a_missing_answer_is_not_believed_while_the_store_cannot_confirm_it():
    """The payload reads as absent and the existence check fails: a store in
    trouble, so the take raises and the group is trained once the store is back."""
    s3 = FaultS3()
    ids = _fill(s3, "s0", 2)
    batch_reader = reader(trainer(s3, ("s0",)))
    s3.fail("get", PAYLOADS, FaultS3.exceptions.NoSuchKey)
    s3.fail("head", PAYLOADS, client_error("InternalError", "HeadObject"))
    with pytest.raises(ClientError):
        batch_reader.take(0, 1, current_version=1)
    trained, reports = drain(batch_reader)
    assert _ids(trained) == ids and sum(r.damaged_payload_groups for r in reports) == 0


def test_a_missing_answer_the_second_look_contradicts_is_raised_not_dropped():
    s3 = FaultS3()
    ids = _fill(s3, "s0", 2)
    batch_reader = reader(trainer(s3, ("s0",)))
    s3.fail("get", PAYLOADS, FaultS3.exceptions.NoSuchKey)  # one wrong "no such key"; the object is there
    with pytest.raises(SampleBufferError, match="is missing"):
        batch_reader.take(0, 1, current_version=1)
    trained, reports = drain(batch_reader)
    assert _ids(trained) == ids and sum(r.damaged_payload_groups for r in reports) == 0


# ------------------------------------------------------------------ taking again in place
def test_a_take_repeated_on_the_same_reader_after_a_failed_read_trains_the_group():
    s3 = FaultS3()
    ids = _fill(s3, "s0", 2)
    batch_reader = reader(trainer(s3, ("s0",)))
    s3.fail("get", PAYLOADS, client_error("SlowDown", "GetObject"))
    with pytest.raises(ClientError):
        batch_reader.take(0, 1, current_version=1)
    taken, report = batch_reader.take(0, 1, current_version=1)
    assert _ids(batch_reader.buffer.entries()[position] for position, _ in taken) == [ids[0]]
    assert (report.start, report.end) == (0, 1) and batch_reader.watermark_before(1) == 1
    taken, _report = batch_reader.take(1, 1, current_version=1)
    assert [position for position, _ in taken] == [1]


def test_a_failed_take_gives_back_every_group_it_selected_also_for_a_smaller_take():
    s3 = FaultS3()
    _fill(s3, "s0", 5)
    batch_reader = reader(trainer(s3, ("s0",)))
    s3.fail("get", PAYLOADS, client_error("SlowDown", "GetObject"), skip=2)  # the third of three reads
    with pytest.raises(ClientError):
        batch_reader.take(0, 3, current_version=1)
    taken, report = batch_reader.take(0, 2, current_version=1)  # fewer than the failed take selected
    assert [position for position, _ in taken] == [0, 1] and (report.start, report.end) == (0, 2)
    assert batch_reader.watermark_before(1) == 2  # not 3: position 2 is selected, not trained
    taken, report = batch_reader.take(1, 3, current_version=1)
    assert [position for position, _ in taken] == [2, 3, 4] and (report.start, report.end) == (2, 5)


def test_a_read_that_fails_while_a_short_round_is_filled_loses_neither_group():
    s3 = FaultS3()
    _fill(s3, "s0", 3)
    batch_reader = reader(trainer(s3, ("s0",)), group_size=4)
    s3.fail("get", PAYLOADS, client_error("SlowDown", "GetObject"), skip=1)  # the fill's read
    with pytest.raises(ClientError):
        batch_reader.take(0, 1, current_version=1)
    taken, _report = batch_reader.take(0, 1, current_version=1)
    assert [position for position, _ in taken] == [0, 1]


def test_a_failed_take_keeps_the_group_carried_over_from_the_round_before():
    s3 = FaultS3()
    writer = GroupWriter(shard_buffer(s3, "s0"), "s0")
    for g, n in enumerate((2, 3, 2, 2)):  # round 0: 2 + (3 does not fit 4) -> carried over
        writer.write(group(g, n=n), group_key=f"g{g}")
    batch_reader = reader(trainer(s3, ("s0",)), group_size=4)
    taken, report = batch_reader.take(0, 1, current_version=1)
    assert [position for position, _ in taken] == [0] and report.carried_over_groups == 1
    s3.fail("get", PAYLOADS, client_error("SlowDown", "GetObject"))
    with pytest.raises(ClientError):
        batch_reader.take(1, 2, current_version=1)
    taken, _report = batch_reader.take(1, 2, current_version=1)
    assert [position for position, _ in taken] == [1, 2, 3]


def test_a_reset_to_the_watermark_forgets_the_groups_of_a_failed_take():
    s3 = FaultS3()
    _fill(s3, "s0", 3)
    batch_reader = reader(trainer(s3, ("s0",)))
    batch_reader.take(0, 1, current_version=1)
    s3.fail("get", PAYLOADS, client_error("SlowDown", "GetObject"))
    with pytest.raises(ClientError):
        batch_reader.take(1, 1, current_version=1)
    batch_reader.reset(0)
    taken, _report = batch_reader.take(0, 3, current_version=1)
    assert [position for position, _ in taken] == [0, 1, 2]


def test_a_read_ahead_that_failed_is_read_again_by_the_take():
    s3 = FaultS3()
    ids = _fill(s3, "s0", 1)
    batch_reader = reader(trainer(s3, ("s0",)))
    failed: Future = Future()
    failed.set_exception(client_error("SlowDown", "GetObject"))
    batch_reader._reads[f"payloads/{ids[0]}.bin"] = failed  # what the prefetch thread left
    taken, report = batch_reader.take(0, 1, current_version=1)
    assert [position for position, _ in taken] == [0] and report.prefetched == 0


# ------------------------------------------------------------------ to the driver's tape
def test_the_round_metadata_takes_the_damaged_payload_records():
    from yeto.rl.adapters.miles import rollout_meta_hook as hook

    sent = []
    record = {"position": 4, "sample_id": "s0-g4", "payload_ref": "payloads/s0-g4.bin", "reason": "payload is missing"}
    original, hook.put_to_sink = hook.put_to_sink, lambda item, sink=None: sent.append(item)
    try:
        hook.record_round_metadata(None, 3, buffer_damaged_payload_groups=1, buffer_damaged_payloads=[record])
        with pytest.raises(RuntimeError, match="list of mappings"):
            hook.record_round_metadata(None, 3, buffer_damaged_payloads=["s0-g4"])
    finally:
        hook.put_to_sink = original
    assert sent[0]["buffer_damaged_payload_groups"] == 1 and sent[0]["buffer_damaged_payloads"] == [record]


def test_reader_round_reports_the_damaged_payloads_of_all_its_takes_as_round_metadata(monkeypatch):
    from tests.test_rl_buffer_rollout import _import_plugin_with_stub_miles

    calls = []
    plugin = _import_plugin_with_stub_miles(monkeypatch, {"all": lambda args, samples, source: None})
    monkeypatch.setattr(plugin, "record_round_metadata",
                        lambda args, rid, **counts: calls.append(counts))
    monkeypatch.setattr(plugin, "Sample", SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda **kw: kw)
    monkeypatch.setattr(plugin, "offpolicy_source_event", lambda *a, **kw: {})
    monkeypatch.setattr(plugin, "flat_metrics", lambda source: {})
    good = [sample(0, 0, 4), sample(1, 0, 4)]
    convert = plugin.miles_groups  # the first take's only group does not convert: the round takes again
    monkeypatch.setattr(plugin, "miles_groups", lambda groups, **kw: (
        ([], ["buffer position 0: malformed"]) if groups[0][0] == 0 else convert(groups, **kw)))
    damaged = [{"position": p, "sample_id": f"s0-g{p}", "payload_ref": f"payloads/s0-g{p}.bin",
                "reason": "payload is missing"} for p in (1, 3)]
    takes = iter([
        ([(0, good)], br.TakeReport(rollout_id=2, groups=1, stale_groups=0, start=0, end=2,
                                      damaged_payload_groups=1, damaged_payloads=damaged[:1])),
        ([(2, good)], br.TakeReport(rollout_id=2, groups=1, stale_groups=0, start=2, end=4,
                                    damaged_payload_groups=1, damaged_payloads=damaged[1:])),
    ])
    round_ = plugin._BufferRound.__new__(plugin._BufferRound)
    round_.args = SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path="all")
    round_.groups, round_.samples_per_group = 1, 2
    round_.reader = SimpleNamespace(take=lambda rid, n, current_version: next(takes))
    out = round_(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=4))
    assert calls[0]["buffer_damaged_payload_groups"] == 2 and calls[0]["buffer_damaged_payloads"] == damaged
    assert out["metrics"]["sample_buffer/damaged_payload_groups"] == 2.0


def test_driver_tapes_the_damaged_payloads_a_round_passed_over(tmp_path):
    import dataclasses

    from tests.test_rl_policy_age import _driver

    record = {"position": 4, "sample_id": "s0-g4", "payload_ref": "payloads/s0-g4.bin", "reason": "payload is missing"}
    driver = _driver(tmp_path, 1)
    original = driver.rollout.generate

    def gen(rid, **kw):
        extra = {"buffer_damaged_payload_groups": 1, "buffer_damaged_payloads": [record]} if rid == 1 else {}
        return dataclasses.replace(original(rid, **kw), buffer_stale_groups=0, **extra)

    driver.rollout.generate = gen
    driver.run()
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    skipped = [e for e in events if e["event"] == "rl_buffer_damaged_payload_skipped"]
    assert [(e["rollout_id"], e["damaged_groups"], e["payloads"]) for e in skipped] == [(1, 1, [record])]


# ------------------------------------------------------------------ a record that disagrees with its payload
def _claim_version(s3: FaultS3, shard: str, pos: int, version: int) -> None:
    """The shard's index record at ``pos`` names ``version``; its payload is unchanged."""
    key = f"sample-buffer/shards/{shard}/{pos:012d}.json"
    s3.put_json(key, {**s3.json(key), "behavior_version": version})


@pytest.mark.parametrize("group_size", [2, None, 4])
def test_a_record_that_names_a_newer_version_than_its_payload_cannot_pass_an_over_age_group(group_size):
    """The record says version 9, the payload's tokens are version 1: against
    version 12 (age 8, floor 4) the group is over age, whatever its record says."""
    s3 = FaultS3()
    ids = _fill(s3, "s0", 1, version=9) + _fill(s3, "s0", 1, start=1, version=1) + \
        _fill(s3, "s0", 2, start=2, version=9)
    _claim_version(s3, "s0", 1, 9)
    batch_reader = reader(trainer(s3, ("s0",)), group_size=group_size)
    taken, report = batch_reader.take(0, 2 if group_size != 4 else 1, current_version=12)
    assert _ids(batch_reader.buffer.entries()[p] for p, _ in taken) == [ids[0], ids[2]]
    assert (report.stale_groups, report.stale_payload_groups, report.stale_positions) == (1, 1, [101])
    assert report.dropped_small_groups == 0 and report.damaged_payload_groups == 0
    assert report.behavior_versions == {9: 2} and sum(report.source_groups.values()) == 2
    assert report.metrics()["sample_buffer/stale_payload_groups"] == 1.0


def test_a_record_that_disagrees_with_a_payload_inside_the_age_window_is_trained():
    """Only the age gate looks at the payload's version: a disagreement that leaves
    the group inside the window changes nothing (the accounting follows the record)."""
    s3 = FaultS3()
    _fill(s3, "s0", 1, version=5)
    _claim_version(s3, "s0", 0, 9)
    taken, report = reader(trainer(s3, ("s0",))).take(0, 1, current_version=12)
    assert [p for p, _ in taken] == [0] and (report.stale_groups, report.stale_payload_groups) == (0, 0)
    assert report.behavior_versions == {9: 1}


def test_a_carried_group_ages_out_by_its_payloads_version_not_its_records():
    s3 = FaultS3()
    writer = GroupWriter(shard_buffer(s3, "s0"), "s0")
    writer.write(group(0, 9, n=2), group_key="g0")
    writer.write(group(1, 4, n=3), group_key="g1")  # 2 + 3 > 4: carried over
    writer.write(group(2, 9, n=3), group_key="g2")
    _claim_version(s3, "s0", 1, 9)
    batch_reader = reader(trainer(s3, ("s0",)), group_size=4)
    taken, report = batch_reader.take(0, 1, current_version=12)  # floor 4
    assert [p for p, _ in taken] == [0] and report.carried_over_groups == 1
    taken, report = batch_reader.take(1, 1, current_version=13)  # floor 5: the payload's version 4 is out
    assert [p for p, _ in taken] == [2] and report.stale_groups == 1


def test_a_payload_without_a_numeric_version_is_judged_by_its_record_alone():
    assert bp.payload_oldest_version([{"index": 0}]) is None and bp.payload_oldest_version([]) is None
    assert bp.payload_oldest_version(group(0, 3) + group(1, 5)) == 3
