"""A full-parameter island's data position, kept across its processes (S27 task assignment, P0).

Why: a full-parameter island counts its rounds locally from 0 in every process
(``full_parameter_sync`` module doc: the policy version is the local round
counter), so the driver's restart path (``IslandDriver._restore_data_cursor``,
keyed by a global rollout id above 0) never runs for it. Before this module a
relaunched island (a new process: Modal retry, launcher relaunch, peer
recovery) read its data from the first row again and trained rows it had
already trained.

What: after the sync boundary of every round, the driver appends one record
``{"round", "cursor", "slot"}`` to the island's log in the run's checkpoint
store. ``cursor`` is the rollout data cursor; it counts rows of the island's
own data slot (``data_split``). At start:

* a process that joins a running pool (the session reports
  ``joined_running_pool``: the syncer sent fragments newer than version 0, or a
  peer recovery restored the island) moves its data source to the last record;
* a process that starts the run writes a start record (its fresh cursor), so a
  record that an earlier run left in a reused store is never read.

What a relaunch draws again (until a central dispenser replaces this, P2):

* never the rows of the rounds that reached their boundary before the last
  record;
* again, the rows of the round that was running when the process stopped (that
  round never reached its boundary, so no merge used it);
* never again, prompts the rollout had drawn ahead at the last record (a
  fully_async rollout keeps groups in flight): those are lost, at most once;
* a process that starts before the first merge (every fragment still at
  version 0) cannot tell a relaunch from the start of the run: it starts at
  the first row.

Layout: one writer per island (the island's current process), one new file or
object per record, never an append (a bucket mount and an object store cannot
append): ``data-positions/island<id>/<seq>.json`` under the store directory, or
under the same key prefix in the store's object API (``CheckpointStore.objects``:
S3, OSS). A record that cannot be read (a write cut short) is skipped.

The store commit (a Modal Volume commit takes seconds) runs in a background
thread, so the round does not wait for it; a record written while a commit
runs is committed by one more commit. :meth:`DataPositionLog.close` waits for
the last one.

This module is engine-neutral: a cursor is the mapping the rollout pool reports.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_POSITIONS_DIR = "data-positions"
COMMIT_WAIT_S = 120.0
START_ROUND = -1  # the round of a start record


class DataPositionError(RuntimeError):
    """A data position record this island cannot continue from."""


@dataclass(frozen=True)
class DataPosition:
    """One record: the data cursor after local round ``round`` (:data:`START_ROUND`:
    the start of a run) of the island with data slot ``slot`` (None: every row)."""

    round: int
    cursor: dict[str, int] | None
    slot: str | None

    def encode(self) -> str:
        return json.dumps({"round": int(self.round), "cursor": self.cursor, "slot": self.slot},
                          sort_keys=True)

    @classmethod
    def decode(cls, text: str | bytes) -> "DataPosition":
        raw = json.loads(text)
        cursor = raw.get("cursor")
        return cls(int(raw["round"]),
                   None if cursor is None else {str(k): int(v) for k, v in dict(cursor).items()},
                   raw.get("slot"))



class _BackgroundCommit:
    """Run ``commit`` off the caller's thread; requests made while one runs are
    served by one more commit after it."""

    def __init__(self, commit: Callable[[], Any]) -> None:
        self._commit = commit
        self._lock = threading.Lock()
        self._pending = False
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []

    def request(self) -> None:
        with self._lock:
            self._pending = True
            if self._thread is not None:
                return  # the running thread sees the request before it exits
            self._thread = threading.Thread(target=self._run, name="data-position-commit",
                                            daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._thread = None
                    return
                self._pending = False
            try:
                self._commit()
            except Exception as error:  # noqa: BLE001 - the next record commits again
                self.errors.append(f"{type(error).__name__}: {error}")

    def wait(self, timeout_s: float = COMMIT_WAIT_S) -> bool:
        """Wait for the running commit (and the ones requested meanwhile); False
        when it did not end within ``timeout_s``."""
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout_s)
        return not thread.is_alive()


class DataPositionLog:
    """The data position records of one island (module doc), as files under a
    store directory (``root``) or as objects (``objects``)."""

    def __init__(self, *, learner_id: int, slot: str | None, root: str | Path | None = None,
                 objects: Any = None, commit: Callable[[], Any] = lambda: None) -> None:
        if (root is None) == (objects is None):
            raise ValueError("a data position log needs a store directory or an object store")
        self.learner_id = int(learner_id)
        self.slot = slot or None
        self.objects = objects
        self.prefix = f"{DATA_POSITIONS_DIR}/island{self.learner_id}/"
        self.dir = None if root is None else Path(root) / self.prefix
        self._names: list[str] | None = None
        self._commit = _BackgroundCommit(commit)

    # -- read ------------------------------------------------------------------
    def last(self) -> DataPosition | None:
        """The newest readable record, None when there is none. Raises when it was
        written for another data slot (its cursor counts other rows)."""
        record = self._last_record()
        if record is not None and record.slot != self.slot:
            raise DataPositionError(
                f"island {self.learner_id}: the last data position was written for data slot "
                f"{record.slot or 'all rows'}, this island reads {self.slot or 'all rows'}")
        return record

    def _last_record(self) -> DataPosition | None:
        for name in reversed(self._listed()):
            try:
                return DataPosition.decode(self._read(name))
            except (OSError, TypeError, ValueError, KeyError):
                continue  # a write cut short: an older record is the position
        return None

    def _listed(self) -> list[str]:
        if self._names is None:
            if self.objects is not None:
                names = [key[len(self.prefix):] for key in self.objects.list(self.prefix)]
            else:
                names = [p.name for p in self.dir.iterdir()] if self.dir.is_dir() else []
            self._names = sorted(n for n in names if n.endswith(".json"))
        return self._names

    def _read(self, name: str) -> str | bytes:
        if self.objects is not None:
            return self.objects.read(self.prefix + name)
        return (self.dir / name).read_text(encoding="utf-8")

    # -- write -----------------------------------------------------------------
    def start(self, cursor: Mapping[str, int] | None) -> None:
        """The start record of a run (the fresh cursor)."""
        self._append(DataPosition(START_ROUND, _plain(cursor), self.slot))

    def record(self, round_id: int, cursor: Mapping[str, int] | None) -> None:
        """The cursor after local round ``round_id`` reached its boundary."""
        self._append(DataPosition(int(round_id), _plain(cursor), self.slot))

    def _append(self, record: DataPosition) -> None:
        names = self._listed()
        name = f"{len(names):010d}.json"
        if self.objects is not None:
            self.objects.write(self.prefix + name, record.encode().encode())
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / name).write_text(record.encode(), encoding="utf-8")
            self._commit.request()
        names.append(name)

    @property
    def commit_errors(self) -> list[str]:
        return list(self._commit.errors)

    def close(self, timeout_s: float = COMMIT_WAIT_S) -> bool:
        """Wait for the last commit; False when it did not end in time."""
        return self._commit.wait(timeout_s)


def _plain(cursor: Mapping[str, int] | None) -> dict[str, int] | None:
    return None if cursor is None else {str(k): int(v) for k, v in dict(cursor).items()}


def data_position_log(store_path: str, *, learner_id: int, slot: str | None,
                      environ: Mapping[str, str] | None = None) -> DataPositionLog:
    """The island's log in the run's checkpoint store at ``store_path`` (the
    island's ``--rl-data-position-store``): by object key when the store has an
    object API, else as files under the store directory (committed in the
    background after each record)."""
    from .resume import store_for

    store = store_for(store_path, environ=environ)
    objects = getattr(store, "objects", None)
    if objects is not None:
        return DataPositionLog(learner_id=learner_id, slot=slot, objects=objects)
    return DataPositionLog(learner_id=learner_id, slot=slot, root=store.root, commit=store.commit)
