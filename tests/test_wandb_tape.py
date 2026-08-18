"""Syncer event tape -> W&B metrics.

The tape is written by Rust (`append_tape` in syncer/src/server.rs) and
read by Python, so one of these tests parses the Rust format string and
fails if a field is added there without a decision on this side.
"""

import json
import re
import threading
import time
from pathlib import Path

from yeto.wandb_tape import TapeForwarder, follow_jsonl, replay, tape_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]

# The record the Rust test `event_tape_records_rendezvous_metrics` produces.
REAL_RECORD = json.loads(
    '{"protocol_version":4,"delta_semantics":"local_minus_raw_anchor","step":7,'
    '"fragment":1,"launch_base_version":3,"attempt":1,"gnorm":0.5,"ms":44,'
    '"quorum":2,"expected":[0,1],'
    '"expected_members":[{"id":0,"generation":10},{"id":1,"generation":20}],'
    '"responded":[0],"responded_members":[{"id":0,"generation":10}],'
    '"missed_grace":[1],"missed_members":[{"id":1,"generation":20}],'
    '"quorum_ms":11,"grace_ms":22,"sync_ms":33,'
    '"responders":[{"id":0,"generation":10,"base_version":3,"staleness":0,'
    '"c_steps":4,"c_tokens":40,"weight":40.0,"contribution":1}]}'
)


class _RecordingRun:
    enabled = True

    def __init__(self):
        self.logged = []
        self.summaries = []

    def log(self, metrics):
        self.logged.append(metrics)

    def summary(self, metrics):
        self.summaries.append(metrics)

    def finish(self, exit_code=0):
        pass


def test_rendezvous_timings_and_participation():
    m = tape_metrics(REAL_RECORD)
    assert m["global_step"] == 7
    assert m["sync/fragment"] == 1
    assert m["sync/gnorm"] == 0.5
    assert m["sync/merge_ms"] == 44
    assert m["sync/sync_ms"] == 33
    assert m["sync/quorum_ms"] == 11
    assert m["sync/grace_ms"] == 22
    assert m["sync/expected"] == 2
    assert m["sync/responded"] == 1
    assert m["sync/missed"] == 1
    assert m["sync/participation"] == 0.5


def test_per_island_series_are_namespaced_by_learner_id():
    m = tape_metrics(REAL_RECORD)
    assert m["learner/0/staleness"] == 0
    assert m["learner/0/contribution"] == 1
    assert m["learner/0/c_steps"] == 4
    assert m["learner/0/c_tokens"] == 40
    assert m["learner/0/base_version"] == 3
    # An island that missed the grace window logs an explicit zero so its
    # curve shows the dropout instead of holding its last value.
    assert m["learner/1/contribution"] == 0.0


def test_staleness_aggregates_across_responders():
    rec = dict(REAL_RECORD)
    rec["responders"] = [
        {"id": 0, "staleness": 1, "contribution": 0.5},
        {"id": 1, "staleness": 5, "contribution": 0.5},
    ]
    rec["missed_grace"] = []
    m = tape_metrics(rec)
    assert m["sync/staleness_max"] == 5
    assert m["sync/staleness_mean"] == 3


def test_null_timings_are_omitted_not_zeroed():
    rec = dict(REAL_RECORD, quorum_ms=None, grace_ms=None)
    m = tape_metrics(rec)
    # "never waited on grace" and "waited 0 ms" are different states.
    assert "sync/quorum_ms" not in m
    assert "sync/grace_ms" not in m


def test_a_round_with_no_responders_still_logs():
    rec = dict(REAL_RECORD, responders=[], responded=[], missed_grace=[0, 1])
    m = tape_metrics(rec)
    assert m["sync/responded"] == 0
    assert m["sync/participation"] == 0.0
    assert "sync/staleness_max" not in m


def test_every_rust_tape_field_is_consumed_or_deliberately_skipped():
    """Fail if the syncer starts emitting a field this reader ignores."""
    source = (REPO_ROOT / "syncer/src/server.rs").read_text()
    body = source.split("fn append_tape(", 1)[1].split("\nfn ", 1)[0]
    # The responders literal is built first, the record literal last.
    responder_fmt, record_fmt = body.split("let line = format!(", 1)
    top_level = set(re.findall(r'\\"(\w+)\\":', record_fmt))
    responder_keys = set(re.findall(r'\\"(\w+)\\":', responder_fmt))
    assert responder_keys, "could not find the responder literal in append_tape"

    consumed_top = {
        "step", "fragment", "attempt", "gnorm", "ms", "sync_ms", "quorum",
        "expected", "responded", "missed_grace", "launch_base_version",
        "quorum_ms", "grace_ms", "responders",
    }
    # Constants and the member-identity mirrors of the id lists: nothing a
    # time series can be drawn from.
    skipped_top = {
        "protocol_version", "delta_semantics",
        "expected_members", "responded_members", "missed_members",
    }
    unhandled = top_level - consumed_top - skipped_top
    assert not unhandled, f"new syncer tape field(s) not routed to W&B: {unhandled}"

    consumed_responder = {
        "id", "staleness", "contribution", "weight", "c_steps", "c_tokens",
        "base_version",
    }
    skipped_responder = {"generation"}
    unhandled = responder_keys - consumed_responder - skipped_responder
    assert not unhandled, f"new responder field(s) not routed to W&B: {unhandled}"


def test_replay_logs_every_record_and_summarizes(tmp_path):
    tape = tmp_path / "tape.jsonl"
    tape.write_text("\n".join(json.dumps(REAL_RECORD) for _ in range(3)) + "\n")
    run = _RecordingRun()
    assert replay(tape, run) == 3
    assert len(run.logged) == 3
    assert run.summaries == [{"sync/tape_records": 3}]


def test_replay_skips_corrupt_lines(tmp_path):
    tape = tmp_path / "tape.jsonl"
    tape.write_text(json.dumps(REAL_RECORD) + "\n{not json}\n" + json.dumps(REAL_RECORD) + "\n")
    run = _RecordingRun()
    assert replay(tape, run) == 2


def test_follow_yields_only_complete_lines(tmp_path):
    tape = tmp_path / "tape.jsonl"
    line = json.dumps(REAL_RECORD)
    tape.write_text(line + "\n" + line[:20])  # a torn tail
    stop = threading.Event()
    seen = []

    def drain():
        for rec in follow_jsonl(tape, stop=stop, poll_seconds=0.01):
            seen.append(rec)

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while len(seen) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(seen) == 1  # the half record is not a merge

    with open(tape, "a") as f:  # complete it
        f.write(line[20:] + "\n")
    deadline = time.monotonic() + 5
    while len(seen) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    t.join(timeout=5)
    assert len(seen) == 2


def test_follow_waits_for_a_tape_that_does_not_exist_yet(tmp_path):
    tape = tmp_path / "later.jsonl"
    stop = threading.Event()
    seen = []

    def drain():
        for rec in follow_jsonl(tape, stop=stop, poll_seconds=0.01):
            seen.append(rec)

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    time.sleep(0.05)
    tape.write_text(json.dumps(REAL_RECORD) + "\n")
    deadline = time.monotonic() + 5
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    t.join(timeout=5)
    assert len(seen) == 1


def test_forwarder_is_inert_for_a_disabled_run(tmp_path):
    from yeto.wandb_logger import NullRun

    forwarder = TapeForwarder(NullRun(), tmp_path / "tape.jsonl")
    forwarder.start()
    assert forwarder._thread is None
    forwarder.stop()


def test_forwarder_streams_a_growing_tape(tmp_path):
    tape = tmp_path / "tape.jsonl"
    tape.write_text(json.dumps(REAL_RECORD) + "\n")
    run = _RecordingRun()
    forwarder = TapeForwarder(run, tape)
    forwarder.start()
    deadline = time.monotonic() + 5
    while not run.logged and time.monotonic() < deadline:
        time.sleep(0.01)
    forwarder.stop()
    assert run.logged[0]["global_step"] == 7
    assert run.summaries[-1] == {"sync/tape_records": 1}
