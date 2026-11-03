"""The run's merged version on a relay board.

A spot sampler learns the policy version only from the fragments its subscriber
receives. Over a slow WAN the first fragment of a (re)joined sampler arrives
tens of minutes after it starts, and until then it generated on the base weights,
far past any policy-age limit. The relay trainers therefore put the run's
merged version on board :data:`VERSION_BOARD` after every publication
(:class:`VersionPublisher`); the sampler reads it at every round admission
(:class:`VersionReader`) and holds rounds while the weights it serves are more
than ``max_policy_age`` merged versions behind.

The same board fences a sampler out of another run. Every record carries
the run identity (``run``, from :data:`RUN_ID_ENVS`) and its publisher (the
trainer's learner id). Several relay trainers of one run write the same board
(the newest PUT wins), so a lagging or just restarted trainer may write a version
below one the sampler already read: that is a stale write and is ignored (the
reader keeps the highest version seen). Only a change of run identity, or board
item ids going backwards (a syncer started fresh, without ``--resume``), means
the run the sampler joined is gone; the sampler then stops.

Both sides are best effort: a board the trainer cannot set, or the sampler
cannot read, changes nothing (the subscriber's fragments still rule).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

VERSION_BOARD = 2
SCHEMA = "yeto.spot.merged-version/v1"
RUN_ID_ENVS = ("YETO_RUN_ID", "YETO_RUN_GROUP")


def run_identity_from_env() -> str | None:
    for name in RUN_ID_ENVS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def encode(merged_version: int, max_policy_age: int, *, run: str | None = None,
           publisher: int | None = None) -> bytes:
    doc: dict[str, Any] = {"schema": SCHEMA, "merged_version": int(merged_version),
                           "max_policy_age": int(max_policy_age)}
    if run is not None:
        doc["run"] = str(run)
    if publisher is not None:
        doc["publisher"] = int(publisher)
    return json.dumps(doc, sort_keys=True).encode()


def decode_record(data: bytes) -> dict[str, Any] | None:
    """{merged_version, max_policy_age, run, publisher} (run/publisher may be None)."""
    try:
        doc = json.loads(bytes(data).decode())
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return None
    run = doc.get("run")
    publisher = doc.get("publisher")
    return {"merged_version": int(doc["merged_version"]), "max_policy_age": int(doc["max_policy_age"]),
            "run": None if run is None else str(run),
            "publisher": None if publisher is None else int(publisher)}


def decode(data: bytes) -> tuple[int, int] | None:
    record = decode_record(data)
    if record is None:
        return None
    return record["merged_version"], record["max_policy_age"]


class VersionPublisher:
    """Trainer side: set the board to the newest merged version (never lower)."""

    def __init__(self, open_board: Callable[[], Any], max_policy_age: int, *,
                 run: str | None = None, publisher: int | None = None) -> None:
        self.open_board = open_board
        self.max_policy_age = int(max_policy_age)
        self.run = run
        self.publisher = publisher
        self.board: Any = None
        self.published = -1

    def publish(self, merged_version: int | None) -> bool:
        if merged_version is None or int(merged_version) <= self.published:
            return False
        try:
            if self.board is None:
                self.board = self.open_board()
            self.board.set(encode(int(merged_version), self.max_policy_age,
                                  run=self.run, publisher=self.publisher))
        except Exception as error:  # noqa: BLE001 - best effort, training goes on
            print(f"[yeto-spot] version board not set: {error}", flush=True)
            self.close()
            return False
        self.published = int(merged_version)
        return True

    def close(self) -> None:
        board, self.board = self.board, None
        if board is not None:
            try:
                board.close()
            except Exception:  # noqa: BLE001
                pass


class VersionReader:
    """Sampler side: the newest board value, polled at most every ``poll_s``."""

    def __init__(self, open_board: Callable[[], Any], *, poll_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.open_board = open_board
        self.poll_s = float(poll_s)
        self.clock = clock
        self.board: Any = None
        self.item_id = 0
        self.last_item_id = 0  # survives close(): ids going backwards mean a fresh syncer
        self.run: str | None = None
        self.publisher: int | None = None
        self.merged_version: int | None = None
        self.max_policy_age: int | None = None
        self.highest = -1
        self.regressed = False
        self._polled: float | None = None

    def refresh(self) -> int | None:
        now = self.clock()
        if self._polled is not None and now - self._polled < self.poll_s:
            return self.merged_version
        self._polled = now
        try:
            if self.board is None:
                self.board = self.open_board()
            item = self.board.get(newer_than=self.item_id)
        except Exception:  # noqa: BLE001 - unreadable board: keep the last value
            self.close()
            return self.merged_version
        if item is not None:
            record = decode_record(item.data)
            item_id = int(item.item_id)
            # a fresh syncer numbers its board items from 1 again
            if item_id < self.last_item_id:
                self.regressed = True  # the syncer of the run this sampler joined is gone
            self.item_id = item_id
            self.last_item_id = max(self.last_item_id, item_id)
            if record is not None:
                self.observe(record["merged_version"], record["max_policy_age"],
                             run=record["run"], publisher=record["publisher"])
        return self.merged_version

    def observe(self, merged_version: int, max_policy_age: int, *, run: str | None = None,
                publisher: int | None = None) -> None:
        if run is not None:
            if self.run is not None and run != self.run:
                self.regressed = True  # another run took the board
                return
            self.run = run
        if merged_version < self.highest:
            return  # a stale write of a lagging or restarted trainer of this run
        self.highest = merged_version
        self.publisher = publisher
        self.merged_version, self.max_policy_age = merged_version, max_policy_age

    def behind(self, served_version: int | None) -> bool:
        """The weights served (None: none yet) are past the run's policy-age limit."""
        run = self.refresh()
        if run is None or self.max_policy_age is None:
            return False
        served = -1 if served_version is None else int(served_version)
        return run - served > self.max_policy_age

    def close(self) -> None:
        board, self.board = self.board, None
        self.item_id = 0  # a reconnect may reach a fresh syncer, whose items start at 1 again
        if board is not None:
            try:
                board.close()
            except Exception:  # noqa: BLE001
                pass
