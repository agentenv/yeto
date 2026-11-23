"""Full-parameter islands record ``rl_fragment_push`` / ``rl_fragment_bcast``.

The LoRA stack (``DecoupledSync``) records one ``rl_fragment_push`` per PUSH
and one ``rl_fragment_bcast`` per applied broadcast (``progress.py``). A
full-parameter island (``FullParameterSync``) records the same events in the
same schema, the actor's only, after each boundary's ``rl_full_parameter_round``:
one per fragment and round, never per tensor. A no-sync island
(``LocalOnlySync``: the single-trainer customer form) records none.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from types import SimpleNamespace


from test_verl_full_parameter_sync import (  # noqa: F401 - syncer_binary is a fixture
    _layout,
    _two_islands,
    needs_cargo,
    syncer_binary,
)
from yeto.rl.decoupled import FragmentSubmission
from yeto.rl.engine.full_parameter_sync import FullParameterSync

# The LoRA stack's push record: FragmentSubmission + realized_h, with the wall-clock
# stamps of the PULL and of the enqueue. A full-parameter push has no delta norm
# (its delta is streamed in chunks).
LORA_PUSH_FIELDS = ({f.name for f in dataclasses.fields(FragmentSubmission)}
                    | {"realized_h", "permit_received_unix", "enqueued_unix"})
PUSH_FIELDS = LORA_PUSH_FIELDS - {"delta_l2_norm"}
OPTIONAL_PUSH_FIELDS = {"wire_bytes"}  # the PUSH's bytes on the wire, when encoded ahead
BCAST_FIELDS = {"fragment_id", "version", "payload_bytes", "queue_seconds", "received_unix", "applied_unix"}
OPTIONAL_BCAST_FIELDS = {"catch_up", "overwrite"}


def _of(events, name):
    return [fields for event, fields in events if event == name]


def _check_schema(events, *, fragments: int, rounds_cap: int):
    pushes, bcasts = _of(events, "rl_fragment_push"), _of(events, "rl_fragment_bcast")
    assert pushes and bcasts
    for push in pushes:
        assert PUSH_FIELDS <= set(push) <= PUSH_FIELDS | OPTIONAL_PUSH_FIELDS, set(push) ^ PUSH_FIELDS
        assert push.get("wire_bytes", 1) > 0
        assert push["realized_h"] == push["c_steps"] >= 1
        assert push["permit_received_unix"] <= push["enqueued_unix"]
        assert abs(push["enqueued_unix"] - push["permit_received_unix"] - push["pull_to_push_seconds"]) < 0.05
        assert 0 <= push["fragment_id"] < fragments and push["global_step"] >= 1
    for bcast in bcasts:
        assert BCAST_FIELDS <= set(bcast) <= BCAST_FIELDS | OPTIONAL_BCAST_FIELDS, set(bcast)
        assert all(bcast[key] is True for key in OPTIONAL_BCAST_FIELDS & set(bcast))
        assert bcast["received_unix"] <= bcast["applied_unix"]
        assert abs(bcast["applied_unix"] - bcast["received_unix"] - bcast["queue_seconds"]) < 1e-3
        assert bcast["payload_bytes"] > 0
    # bounded: one push per fragment and round attempt, one record per applied broadcast
    assert max(Counter((p["fragment_id"], p["global_step"], p["round_attempt"]) for p in pushes).values()) == 1
    assert max(Counter((b["fragment_id"], b["version"]) for b in bcasts).values()) == 1
    assert len(pushes) <= rounds_cap * fragments and len(bcasts) <= rounds_cap * fragments
    # emitted after the round event of their boundary: none before the first round event
    first_round = next(i for i, (event, _) in enumerate(events) if event == "rl_full_parameter_round")
    assert all(event not in ("rl_fragment_push", "rl_fragment_bcast") for event, _ in events[:first_round])
    return pushes, bcasts


@needs_cargo
def test_strict_sweeps_record_every_push_and_every_applied_broadcast(syncer_binary, tmp_path, monkeypatch):
    rounds, fragments = 3, len(_layout().fragments.fragments)
    results, _log = _two_islands(syncer_binary, tmp_path, monkeypatch, rounds=rounds, fragments=fragments,
                                 async_rounds=False, matrix_merge="avg", pipeline=1, final_cut="download",
                                 extra=("--rl-sync-full-parameter-rounds", "sweep",
                                        "--rl-sync-final-cut", "download"))
    sizes = [fragment.numel * 4 for fragment in _layout().fragments.fragments]  # f32 BCAST
    for learner_id in range(2):
        events = results[learner_id]["events"]
        pushes, bcasts = _check_schema(events, fragments=fragments, rounds_cap=rounds)
        # a strict sweep: this island pushes every round of every fragment, once
        assert sorted(p["global_step"] for p in pushes) == list(range(1, rounds * fragments + 1))
        assert all(p["round_attempt"] == 1 and p["c_steps"] == 1 for p in pushes)
        assert {p["global_step"]: p["fragment_id"] for p in pushes} == {
            step: (step - 1) % fragments for step in range(1, rounds * fragments + 1)}
        # every sweep but the last is applied as broadcasts (the last arrives as FINAL)
        assert sorted((b["fragment_id"], b["version"]) for b in bcasts) == sorted(
            ((step - 1) % fragments, step) for step in range(1, (rounds - 1) * fragments + 1))
        assert all(set(b) == BCAST_FIELDS for b in bcasts)  # no catch-up, no overwrite
        assert [b["payload_bytes"] for b in bcasts] == [sizes[b["fragment_id"]] for b in bcasts]


@needs_cargo
def test_asynchronous_rounds_record_pushes_and_name_overwritten_rounds(syncer_binary, tmp_path, monkeypatch):
    rounds, fragments = 3, len(_layout().fragments.fragments)
    results, log = _two_islands(syncer_binary, tmp_path, monkeypatch, rounds=rounds, fragments=fragments,
                                async_rounds=True, extra=("--rl-sync-full-parameter-rounds", "async",
                                                          "--rl-sync-final-cut", "download"),
                                matrix_merge="rda", pipeline=2)
    for learner_id in range(2):
        events = results[learner_id]["events"]
        local_rounds = len(_of(events, "rl_full_parameter_round"))
        pushes, bcasts = _check_schema(events, fragments=fragments, rounds_cap=max(local_rounds, rounds))
        pushed = {(p["fragment_id"], p["global_step"]) for p in pushes}
        for bcast in bcasts:
            key = (bcast["fragment_id"], bcast["version"])
            # quorum 1: a round may close without this island; only such a round is an overwrite
            assert bcast.get("overwrite", False) == (key not in pushed and not bcast.get("catch_up")), (
                learner_id, bcast, sorted(pushed), log[-2000:])


def _fake_stream(records):
    return SimpleNamespace(drain_fragment_records=lambda: [dict(r) for r in records.pop(0)] if records else [])


def test_only_the_actor_records_reach_the_tape_and_every_role_is_drained():
    actor = [[{"event": "rl_fragment_push", "fragment_id": 0, "global_step": 1}]]
    critic = [[{"event": "rl_fragment_push", "fragment_id": 0, "global_step": 1}]]
    session = FullParameterSync.__new__(FullParameterSync)
    session.pairing = SimpleNamespace(streams={"actor": _fake_stream(actor), "critic": _fake_stream(critic)})
    emitted = []
    driver = SimpleNamespace(emit=lambda event, **fields: emitted.append((event, fields)))
    session._emit_fragment_records(driver)
    assert emitted == [("rl_fragment_push", {"fragment_id": 0, "global_step": 1})]
    assert actor == [] and critic == []  # drained, so a role's buffer never grows
    session._emit_fragment_records(driver)
    assert len(emitted) == 1
    # a stream without fragment records (a test double, another stack) adds nothing
    session.pairing = SimpleNamespace(streams={"actor": object()})
    session._emit_fragment_records(driver)
    assert len(emitted) == 1


def test_the_no_sync_island_records_no_fragment_events(tmp_path):
    """The customer's single trainer (``--rl-single-island-no-sync``) runs
    ``LocalOnlySync``: its tape has no push / broadcast record at all."""

    import test_rl_engine_driver as td
    from yeto.rl.engine.bridges import LocalOnlySync

    engine = td._engine()
    driver = td._driver(engine, LocalOnlySync(3), tmp_path, evaluate=lambda r: {"pass_rate": 0.5}, eval_interval=2)
    driver.run()
    tape = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    events = Counter(record.get("event") for record in tape)
    assert events  # the driver ran its rounds
    assert not {"rl_fragment_push", "rl_fragment_bcast", "rl_full_parameter_round"} & set(events)
    assert not hasattr(LocalOnlySync, "drain_fragment_records")


def test_the_fragment_record_buffer_is_bounded():
    from yeto.rl.engine.full_parameter_stream import FRAGMENT_RECORDS_MAX, FullParameterRoleStream

    stream = FullParameterRoleStream.__new__(FullParameterRoleStream)
    from collections import deque

    stream._fragment_records = deque(maxlen=FRAGMENT_RECORDS_MAX)
    for i in range(FRAGMENT_RECORDS_MAX + 10):
        stream._fragment_records.append({"event": "rl_fragment_push", "global_step": i})
    records = stream.drain_fragment_records()
    assert len(records) == FRAGMENT_RECORDS_MAX and records[-1]["global_step"] == FRAGMENT_RECORDS_MAX + 9
    assert stream.drain_fragment_records() == []


def _bare_stream(*, partial_quorum=True):
    from collections import deque

    from yeto.rl.engine.full_parameter_stream import FRAGMENT_RECORDS_MAX, FullParameterRoleStream

    stream = FullParameterRoleStream.__new__(FullParameterRoleStream)
    stream.config = SimpleNamespace(role="actor", total_fragment_steps=8, partial_quorum=partial_quorum)
    stream._answered, stream._link_generations, stream._pending = {}, None, {}
    stream.terminal_submitted = False
    stream._pushed_rounds, stream._staged_arrival = set(), {}
    stream._fragment_records = deque(maxlen=FRAGMENT_RECORDS_MAX)
    return stream


def test_a_background_push_records_its_enqueue_time_and_wire_bytes():
    import time

    from yeto.protocol import PullRequest

    stream = _bare_stream()
    now = time.monotonic()
    permit = PullRequest(1, 2, 1, received_at=now - 2.0)
    stream._pending = {1: permit}
    candidate = SimpleNamespace(permit=permit, base_version=0, c_steps=3, c_tokens=40, payload_bytes=800)
    # harvested now, queued by the worker half a second ago, int4 on the wire
    submission = stream.note_pushed(candidate, 1, enqueued_at=now - 0.5, wire_bytes=104)
    assert stream._pending == {} and abs(submission.pull_to_push_seconds - 1.5) < 0.05
    (record,) = stream.drain_fragment_records()
    assert record == {"event": "rl_fragment_push", "fragment_id": 1, "global_step": 2, "round_attempt": 1,
                      "base_version": 0, "c_steps": 3, "c_tokens": 40, "payload_bytes": 800,
                      "pull_to_push_seconds": submission.pull_to_push_seconds,
                      "permit_received_unix": record["permit_received_unix"],
                      "enqueued_unix": record["enqueued_unix"], "realized_h": 3, "wire_bytes": 104}
    assert abs(record["enqueued_unix"] - record["permit_received_unix"] - 1.5) < 0.05
    assert time.time() - record["enqueued_unix"] >= 0.45
    # without encoded bytes (a push streamed by the client) there is no wire_bytes
    stream._pending = {1: permit}
    stream.note_pushed(candidate, 1)
    assert "wire_bytes" not in stream.drain_fragment_records()[0]


def test_an_applied_round_is_an_overwrite_only_without_this_islands_push_and_a_partial_quorum():
    import time

    stream = _bare_stream()
    stream._pushed_rounds = {(0, 1), (0, 3), (1, 2)}
    stream._staged_arrival = {(0, 1): (time.monotonic() - 1.0, 4096)}
    stream._record_applied(0, 1, catch_up=False)  # pushed: a plain apply
    stream._record_applied(1, 4, catch_up=False)  # never pushed round 4: overwritten
    stream._record_applied(0, 5, catch_up=True)   # a catch-up jump is not an overwrite
    plain, overwritten, caught_up = stream.drain_fragment_records()
    assert set(plain) == BCAST_FIELDS | {"event"} and plain["payload_bytes"] == 4096 and plain["queue_seconds"] >= 0.95
    assert overwritten["overwrite"] is True and "payload_bytes" not in overwritten  # arrival unknown
    assert caught_up["catch_up"] is True and "overwrite" not in caught_up
    # applied rounds and older ones leave the set; (1, 2) went with round 4 of fragment 1
    assert stream._pushed_rounds == set()
    strict = _bare_stream(partial_quorum=False)
    strict._record_applied(0, 1, catch_up=False)
    assert "overwrite" not in strict.drain_fragment_records()[0]  # every strict round has this push
