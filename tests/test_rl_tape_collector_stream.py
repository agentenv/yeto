"""Launcher tape collection: chunked log streams and late final events (A2 rerun2)."""

from __future__ import annotations

import json
import threading
import time

from yeto.launcher import wait_for_tapes
from yeto.rl.event_echo import FINALIZED_EVENT, PREFIX, TapeCollector


def _rec(event, **extra):
    return json.dumps({"event": event, "island_id": 0, "time_unix": 1.0, **extra},
                      separators=(",", ":"))


def _tape(path):
    return [json.loads(x)["event"] for x in path.read_text().splitlines()]


def test_a_log_entry_with_several_lines_loses_no_record(tmp_path):
    """rerun2: the finalized record and the next print came in ONE Modal log entry;
    the whole entry was parsed as one line and discarded as malformed."""
    c = TapeCollector(tmp_path / "t.jsonl")
    c.feed(f"{PREFIX}{_rec('rl_driver_phase')}\n")
    c.feed(f"{PREFIX}{_rec(FINALIZED_EVENT)}\n[rl] learner 0 finalized (rl_engine=ports)\n")
    assert c.discarded == 0 and c.finalized
    assert c.close() is True
    assert _tape(tmp_path / "t.jsonl") == ["rl_driver_phase", FINALIZED_EVENT]


def test_a_record_cut_across_two_entries_is_joined(tmp_path):
    c = TapeCollector(tmp_path / "t.jsonl")
    line = PREFIX + _rec(FINALIZED_EVENT)
    c.feed("noise\n" + line[:25])  # half line, no newline yet
    assert not c.finalized and c.discarded == 0
    c.feed(line[25:] + "\nmore noise\n")
    assert c.finalized and c.discarded == 0 and c.close()


def test_a_really_broken_line_is_counted_and_the_next_record_kept(tmp_path):
    c = TapeCollector(tmp_path / "t.jsonl")
    c.feed(PREFIX + '{"event": "rl_x", "isla')  # cut, never completed
    c.feed(PREFIX + _rec(FINALIZED_EVENT) + "\n")  # a new record, not the rest
    assert c.finalized and c.discarded == 1
    assert c.close()


def test_a_held_tail_is_judged_at_close(tmp_path):
    c = TapeCollector(tmp_path / "t.jsonl")
    c.feed(PREFIX + _rec(FINALIZED_EVENT))  # complete record, no trailing newline
    assert c.finalized  # parsed immediately: it is a whole record
    c2 = TapeCollector(tmp_path / "u.jsonl")
    c2.feed(PREFIX + '{"event": "rl_x"')
    assert c2.close() is False and c2.discarded == 1


def test_the_final_event_arriving_after_the_check_started_is_waited_for(tmp_path):
    c = TapeCollector(tmp_path / "t.jsonl")
    stream_done = threading.Event()

    def stream():
        time.sleep(0.3)  # the island prints its last record late
        c.feed(PREFIX + _rec(FINALIZED_EVENT) + "\n")
        stream_done.wait(5)  # the stream stays open afterwards

    t = threading.Thread(target=stream, daemon=True)
    t.start()
    started = time.monotonic()
    assert wait_for_tapes({"l0": c}, ["l0"], [t], limit=10, poll=0.05) == "finalized"
    assert time.monotonic() - started < 5  # returned on the event, not the deadline
    stream_done.set()
    assert c.close()


def test_wait_ends_when_streams_end_or_at_the_deadline(tmp_path):
    c = TapeCollector(tmp_path / "t.jsonl")
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    assert wait_for_tapes({"l0": c}, ["l0"], [dead], limit=10, poll=0.01) == "streams_ended"
    alive = threading.Event()
    t = threading.Thread(target=alive.wait, args=(5,), daemon=True)
    t.start()
    clock = iter([0.0, 0.0, 1.0, 2.0, 3.0])
    assert wait_for_tapes({"l0": c}, ["l0"], [t], limit=2, clock=lambda: next(clock),
                          sleep=lambda s: None) == "deadline"
    alive.set()


def test_recover_from_the_island_tape_file_completes_a_closed_tape(tmp_path):
    """S1 2x1 L40S (s1-mn-20261004a): the stream delivered no run-phase line; the island's
    own ~/yeto-output/rl-island-0.jsonl holds the whole tape."""
    c = TapeCollector(tmp_path / "t.jsonl")
    c.feed(f"{PREFIX}{_rec('rl_driver_phase')}\n")
    assert c.close() is False and c.incomplete_marker.exists()
    island = tmp_path / "rl-island-0.jsonl"
    island.write_text(_rec("rl_driver_phase") + "\n" + _rec("rl_round_trained") + "\n"
                      + "not a record\n" + _rec(FINALIZED_EVENT) + "\n")
    assert c.recover_from_file(island) is True
    assert c.closed and c.count == 3 and c.discarded == 1 and not c.incomplete_marker.exists()
    assert _tape(tmp_path / "t.jsonl") == ["rl_driver_phase", "rl_round_trained", FINALIZED_EVENT]
    # a tape file without the finalized record stays incomplete (fail closed)
    d = TapeCollector(tmp_path / "u.jsonl")
    d.close()
    partial = tmp_path / "rl-island-1.jsonl"
    partial.write_text(_rec("rl_driver_phase") + "\n")
    assert d.recover_from_file(partial) is False and d.incomplete_marker.exists()


def test_launcher_recovers_echo_tape_over_rsync(tmp_path):
    from pathlib import Path

    from yeto.launcher import _recover_echo_tape

    c = TapeCollector(tmp_path / "t.jsonl")
    c.close()
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        Path(cmd[-1], "rl-island-0.jsonl").write_text(_rec("rl_driver_phase") + "\n" + _rec(FINALIZED_EVENT) + "\n")

    assert _recover_echo_tape("isl-l0", c, run=fake_run) is True
    assert seen["cmd"][:2] == ["rsync", "-az"] and seen["cmd"][2] == "isl-l0:yeto-output/rl-island-*.jsonl"
    assert _tape(tmp_path / "t.jsonl") == ["rl_driver_phase", FINALIZED_EVENT]

    def failing_run(cmd, **kw):
        raise RuntimeError("ssh: no route")

    e = TapeCollector(tmp_path / "e.jsonl")
    e.close()
    assert _recover_echo_tape("isl-l0", e, run=failing_run) is False and e.incomplete_marker.exists()
