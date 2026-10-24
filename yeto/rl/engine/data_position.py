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

Layout, one writer per island (the island's current process):

* directory store: ``data-positions/island<id>.jsonl``, one JSON line per record;
* object store (``CheckpointStore.objects``: S3, OSS):
  ``data-positions/island<id>/<seq>.json``, one object per record (an object
  store cannot append).

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
    """The data position records of one island (module doc), in a directory store
    (``lines``) or an object store (``objects``)."""

    def __init__(self, *, learner_id: int, slot: str | None, root: str | Path | None = None,
                 objects: Any = None, prefix: str = DATA_POSITIONS_DIR,
                 commit: Callable[[], Any] = lambda: None) -> None:
        if (root is None) == (objects is None):
            raise ValueError("a data position log needs a store directory or an object store")
        self.learner_id = int(learner_id)
        self.slot = slot or None
        self.objects = objects
        name = f"island{self.learner_id}"
        self.path = None if root is None else Path(root) / DATA_POSITIONS_DIR / f"{name}.jsonl"
        self.prefix = f"{prefix.rstrip('/')}/{name}/"
        self._keys: list[str] | None = None
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
        if self.objects is not None:
            for key in reversed(self._listed()):
                try:
                    return DataPosition.decode(self.objects.read(key))
                except (TypeError, ValueError, KeyError):
                    continue  # unreadable object: an older one is the position
            return None
        if self.path is None or not self.path.is_file():
            return None
        record = None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                record = DataPosition.decode(line)
            except (TypeError, ValueError, KeyError):
                continue  # a torn last line (the process stopped during the write)
        return record

    def _listed(self) -> list[str]:
        if self._keys is None:
            self._keys = sorted(self.objects.list(self.prefix))
        return self._keys

    # -- write -----------------------------------------------------------------
    def start(self, cursor: Mapping[str, int] | None) -> None:
        """The start record of a run (the fresh cursor)."""
        self._append(DataPosition(START_ROUND, _plain(cursor), self.slot))

    def record(self, round_id: int, cursor: Mapping[str, int] | None) -> None:
        """The cursor after local round ``round_id`` reached its boundary."""
        self._append(DataPosition(int(round_id), _plain(cursor), self.slot))

    def _append(self, record: DataPosition) -> None:
        if self.objects is not None:
            keys = self._listed()
            key = f"{self.prefix}{len(keys):010d}.json"
            self.objects.write(key, record.encode().encode())
            keys.append(key)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.encode() + "\n")
        self._commit.request()

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
    object API, else as files under the store directory (committed after each
    record)."""
    from .resume import store_for

    store = store_for(store_path, environ=environ)
    objects = getattr(store, "objects", None)
    if objects is not None:
        return DataPositionLog(learner_id=learner_id, slot=slot, objects=objects)
    return DataPositionLog(learner_id=learner_id, slot=slot, root=store.root, commit=store.commit)
