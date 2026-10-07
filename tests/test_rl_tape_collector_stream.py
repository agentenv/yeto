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


def test_settle_echo_tapes_completes_cut_never_streamed_and_modal_tapes(tmp_path):
    """S14/A19 (-5r1 r7): a SUCCEEDED job's echo stream went silent after a learner
    restart (58/146 records); the island's own file holds the finalized record,
    so the tape is complete and the launcher must not exit 3. Also covers an
    island that never streamed (now recovered too) and a Modal island (tape
    pulled from its Volume). A file without the finalized record stays incomplete."""
    from yeto.launcher import settle_echo_tapes

    events = tmp_path / "events"
    cut = TapeCollector(events / "sky-cut.jsonl")
    cut.feed(f"{PREFIX}{_rec('rl_driver_phase')}\n")  # stream stopped here
    bad = TapeCollector(events / "sky-bad.jsonl")
    bad.feed(f"{PREFIX}{_rec('rl_driver_phase')}\n")
    recovered = []

    def fake_recover(name, collector):
        recovered.append(name)
        f = tmp_path / f"{name}-island.jsonl"
        body = _rec("rl_driver_phase") + "\n" + _rec("rl_round_trained") + "\n"
        if name != "sky-bad":
            body += _rec(FINALIZED_EVENT) + "\n"
        f.write_text(body)
        return collector.recover_from_file(f)

    modal_dir = tmp_path / "modal-tape"
    (modal_dir / "modal-ok" / "rank0").mkdir(parents=True)
    (modal_dir / "modal-ok" / "rank0" / "rl-island-0.jsonl").write_text(
        _rec("rl_driver_phase") + "\n" + _rec(FINALIZED_EVENT) + "\n")
    modal_cfgs = {"modal-ok": object(), "modal-none": object()}
    names = ["sky-cut", "sky-bad", "sky-silent", "modal-ok", "modal-none"]
    incomplete = settle_echo_tapes({"sky-cut": cut, "sky-bad": bad}, names, modal_cfgs, events,
                                   recover_sky=fake_recover, modal_tape_dir=modal_dir)
    assert incomplete == ["modal-none", "sky-bad"]
    assert recovered == ["sky-bad", "sky-cut", "sky-silent"]
    assert _tape(events / "sky-cut.jsonl") == ["rl_driver_phase", "rl_round_trained", FINALIZED_EVENT]
    assert not (events / "sky-cut.jsonl.incomplete").exists()
    assert _tape(events / "sky-silent.jsonl") == ["rl_driver_phase", "rl_round_trained", FINALIZED_EVENT]
    assert _tape(events / "modal-ok.jsonl") == ["rl_driver_phase", FINALIZED_EVENT]
    assert (events / "sky-bad.jsonl.incomplete").exists()
    assert (events / "modal-none.jsonl.incomplete").exists()
    assert cut.closed and bad.closed  # fail closed: nothing is written afterwards


def test_settle_echo_tapes_finalized_stream_needs_no_recovery(tmp_path):
    from yeto.launcher import settle_echo_tapes

    c = TapeCollector(tmp_path / "l0.jsonl")
    c.feed(f"{PREFIX}{_rec(FINALIZED_EVENT)}\n")

    def never(name, collector):
        raise AssertionError("recovery must not run for a finalized tape")

    assert settle_echo_tapes({"l0": c}, ["l0"], {}, tmp_path, recover_sky=never) == []


def test_tail_skips_sky_control_none_and_ends_when_the_stream_ends(monkeypatch, capsys):
    """S14/A19 root cause of the missing run-phase lines: skypilot 0.13's client
    yields None for every rich-status/heartbeat control payload off the main
    thread (the API server heartbeats after 30 s of log silence); _tail took
    the first None as end-of-stream. None is skipped; exhaustion is the end."""
    import sys
    from types import SimpleNamespace

    from yeto.launcher import _tail

    calls = []

    def tail_logs(cluster, job_id, follow, preload_content):
        calls.append((cluster, job_id, follow, preload_content))
        return iter([f"{PREFIX}{_rec('rl_driver_phase')}\n", None, None,
                     f"{PREFIX}{_rec(FINALIZED_EVENT)}\n"])

    monkeypatch.setitem(sys.modules, "sky", SimpleNamespace(tail_logs=tail_logs))
    import tempfile
    from pathlib import Path

    c = TapeCollector(Path(tempfile.mkdtemp()) / "t.jsonl")
    assert _tail("isl-l0", 7, "isl-l0", c) == 0
    assert calls == [("isl-l0", 7, True, False)]
    assert c.finalized and c.count == 2
    assert "[isl-l0] YETO_RL_EVENT" in capsys.readouterr().out
