"""A tape append that fails part-way (disk full mid-write) must leave no torn
bytes behind: otherwise the next record -- e.g. ``rl_learner_finalized`` -- is
glued onto them and the tape reads as incomplete/corrupt afterwards."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from yeto.rl import event_echo
from yeto.rl.event_echo import FINALIZED_EVENT, append_record, tape_is_complete


def _rec(event: str, n: int) -> dict:
    return {"island_id": "i0", "time_unix": float(n), "event": event, "n": n}


def test_failed_append_does_not_tear_the_next_record(tmp_path, monkeypatch):
    tape = tmp_path / "tape.jsonl"
    append_record(tape, _rec("start", 1))

    real_open = Path.open


    def torn_open(self, mode="r", *a, **kw):
        fh = real_open(self, mode, *a, **kw)
        if "a" not in mode:
            return fh
        orig_write = fh.write

        def write(data):
            orig_write(data[: len(data) // 2])
            fh.flush()
            raise OSError(errno.ENOSPC, "No space left on device")

        fh.write = write
        return fh

    monkeypatch.setattr(Path, "open", torn_open)
    with pytest.raises(OSError):
        append_record(tape, _rec("lost", 2))
    monkeypatch.undo()

    append_record(tape, _rec(FINALIZED_EVENT, 3))
    lines = tape.read_text(encoding="utf-8").splitlines()
    import json

    assert [json.loads(l)["event"] for l in lines] == ["start", FINALIZED_EVENT]
    assert tape_is_complete(tape)


def test_append_after_crash_torn_tail_starts_a_new_line(tmp_path):
    """A process killed mid-write leaves an unterminated tail; the next writer
    must not glue its record onto it."""
    tape = tmp_path / "tape.jsonl"
    append_record(tape, _rec("start", 1))
    with tape.open("a", encoding="utf-8") as fh:
        fh.write('{"event":"cut')
    append_record(tape, _rec(FINALIZED_EVENT, 2))
    assert tape_is_complete(tape)
