"""A journal append that fails part-way (disk full mid-write, a failed fsync)
leaves no torn bytes behind: the next append must not glue its record onto
them, or the journal reads as corrupt on the next start."""

from __future__ import annotations

import errno

import pytest

from yeto.rl.engine.journal import Journal, read_journal


class _TornHandle:
    """Wraps the journal's file: the next write lands half its bytes, then fails."""

    def __init__(self, inner, *, fail_on: str = "write") -> None:
        self.inner = inner
        self.fail_on = fail_on

    def write(self, data: bytes) -> int:
        if self.fail_on == "write":
            self.inner.write(data[: len(data) // 2])
            self.inner.flush()
            raise OSError(errno.ENOSPC, "No space left on device")
        return self.inner.write(data)

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.mark.parametrize("fail_on", ["write", "fsync"])
def test_failed_append_leaves_journal_readable(tmp_path, monkeypatch, fail_on):
    journal = Journal(tmp_path)
    journal.append("first", n=1)
    real = journal._fh
    if fail_on == "write":
        journal._fh = _TornHandle(real)
    else:
        import yeto.rl.engine.journal as mod

        def bad_fsync(_fd):
            raise OSError(errno.EIO, "fsync failed")

        monkeypatch.setattr(mod.os, "fsync", bad_fsync)
    with pytest.raises(OSError):
        journal.append("lost", n=2)
    journal._fh = real
    monkeypatch.undo()
    third = journal.append("third", n=3)
    assert [r["kind"] for r in journal.records] == ["first", "third"]
    assert third["seq"] == 2  # the failed append took no sequence number
    journal.close()

    assert [r["kind"] for r in read_journal(tmp_path)] == ["first", "third"]
    with Journal(tmp_path) as again:  # a restart reads it (no JournalCorrupt)
        assert [r["seq"] for r in again.records] == [1, 2]
