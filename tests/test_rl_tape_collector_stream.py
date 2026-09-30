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
