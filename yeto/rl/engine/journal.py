"""Durable reconfiguration journal (rl-infra-spec 3.2, design D4, f-design §1.3/§1.4).

Two files under the island's persistent state directory:

* ``journal.jsonl`` -- append-only write-ahead log. Every record is one JSON
  line written with ``flush`` + ``fsync`` before the call returns, so a record
  the caller saw returned survives a crash. A torn last line (crash during the
  write) is ignored on read and truncated on the next append; any other
  malformed line fails closed (:class:`JournalCorrupt`).
* ``epochs.json`` -- the authoritative committed state (``config_epoch``,
  ``config_id``, ``fork_membership_epoch``, ...). It changes only through
  :meth:`Journal.compare_and_swap`: read, compare the expected epochs, write
  a temp file, ``fsync``, ``rename``, ``fsync`` the directory. The rename is
  the single commit point of a transaction (D4 "durable compare-and-swap").

A process holds the writer lock (``flock`` on ``journal.lock``) for as long
as the :class:`Journal` object is open: one writer per island (D4 single
writer). Readers (status CLI) use :func:`read_journal` without the lock.
JSONL observation events (``EventTape``) are NOT this journal and never act
as the WAL.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JOURNAL_FILE = "journal.jsonl"
EPOCHS_FILE = "epochs.json"
LOCK_FILE = "journal.lock"
SCHEMA = "yeto.rl.reconfig-journal/v1"


class JournalError(RuntimeError):
    pass


class JournalCorrupt(JournalError):
    """A journal line other than a torn tail cannot be parsed."""


class JournalLocked(JournalError):
    """Another process holds the single-writer lock."""


class EpochConflict(JournalError):
    """compare_and_swap found epochs other than the expected ones."""


@dataclass(frozen=True)
class EpochState:
    config_epoch: int = 0
    config_id: str | None = None
    fork_membership_epoch: int = 0
    # Members serving the committed config (opaque yeto member ids).
    members: tuple[str, ...] = ()
    last_tx_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config_epoch": self.config_epoch,
            "config_id": self.config_id,
            "fork_membership_epoch": self.fork_membership_epoch,
            "members": sorted(self.members),
            "last_tx_id": self.last_tx_id,
        }

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "EpochState":
        if raw.get("schema") != SCHEMA:
            raise JournalCorrupt(f"epochs file has schema {raw.get('schema')!r}")
        return EpochState(
            config_epoch=int(raw["config_epoch"]),
            config_id=raw.get("config_id"),
            fork_membership_epoch=int(raw.get("fork_membership_epoch", 0)),
            members=tuple(sorted(raw.get("members") or ())),
            last_tx_id=raw.get("last_tx_id"),
        )


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _parse_lines(data: bytes) -> tuple[list[dict[str, Any]], int]:
    """Records plus the byte length of the valid prefix (a torn tail is dropped)."""
    records: list[dict[str, Any]] = []
    valid = 0
    lines = data.split(b"\n")
    for index, line in enumerate(lines):
        last = index == len(lines) - 1
        if last and line == b"":
            break
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("not an object")
        except ValueError as exc:
            if last:  # no trailing newline: the crash hit mid-write
                break
            raise JournalCorrupt(f"journal line {index + 1} is not valid JSON: {exc}") from exc
        if last:  # complete JSON but no newline: treat as torn too
            break
        records.append(record)
        valid += len(line) + 1
    return records, valid


def read_journal(state_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(state_dir).expanduser() / JOURNAL_FILE
    if not path.exists():
        return []
    return _parse_lines(path.read_bytes())[0]


def read_epochs(state_dir: str | Path) -> EpochState:
    path = Path(state_dir).expanduser() / EPOCHS_FILE
    if not path.exists():
        return EpochState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise JournalCorrupt(f"epochs file is not valid JSON: {exc}") from exc
    return EpochState.from_dict(raw)


class Journal:
    """Single-writer durable journal. Use as a context manager or call close()."""

    def __init__(self, state_dir: str | Path, *, wall_clock=time.time, clock=time.monotonic) -> None:
        self.dir = Path(state_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._wall = wall_clock
        self._clock = clock
        self._lock_fd = os.open(self.dir / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._lock_fd)
            raise JournalLocked(f"journal {self.dir} is held by another writer") from exc
        path = self.dir / JOURNAL_FILE
        data = path.read_bytes() if path.exists() else b""
        self._records, valid = _parse_lines(data)
        if valid != len(data):  # drop a torn tail before appending after it
            with open(path, "r+b") as handle:
                handle.truncate(valid)
                handle.flush()
                os.fsync(handle.fileno())
        self._seq = max((int(r.get("seq", 0)) for r in self._records), default=0)
        self._fh = open(path, "ab")
        self.epochs = read_epochs(self.dir)

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- records -------------------------------------------------------------
    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    def append(self, kind: str, **fields: Any) -> dict[str, Any]:
        if self._fh is None:
            raise JournalError("journal is closed")
        self._seq += 1
        record = {
            "seq": self._seq,
            "kind": kind,
            "wall_time": self._wall(),
            "monotonic": self._clock(),
            **fields,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        self._fh.write(line.encode("utf-8") + b"\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._records.append(json.loads(line))
        return record

    def iter_tx(self, tx_id: str) -> Iterator[dict[str, Any]]:
        return (r for r in self._records if r.get("tx_id") == tx_id)

    # -- durable CAS ----------------------------------------------------------
    def compare_and_swap(
        self,
        *,
        expected_config_epoch: int,
        expected_fork_epoch: int | None = None,
        new: EpochState,
    ) -> EpochState:
        current = read_epochs(self.dir)
        if current != self.epochs:
            raise EpochConflict("epochs file changed behind the single writer")
        if current.config_epoch != expected_config_epoch or (
            expected_fork_epoch is not None
            and current.fork_membership_epoch != expected_fork_epoch
        ):
            raise EpochConflict(
                f"expected config epoch {expected_config_epoch}/fork {expected_fork_epoch}, "
                f"found {current.config_epoch}/{current.fork_membership_epoch}"
            )
        if new.config_epoch < current.config_epoch:
            raise EpochConflict("config epoch may not go backwards")
        tmp = self.dir / (EPOCHS_FILE + ".tmp")
        with open(tmp, "wb") as handle:
            handle.write(json.dumps(new.to_dict(), sort_keys=True).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.dir / EPOCHS_FILE)
        _fsync_dir(self.dir)
        self.epochs = new
        return new
