"""The sampler member table: which buffer shards a trainer consumes, kept in its store.

A trainer used to read exactly the shards of its ``--rl-buffer-shards``, fixed at
launch: a pool added while the run trains, or a shard numbered past that list,
wrote groups nobody trained on. With the store control plane each pool writes the
shards it ever assigned to ``fleet/members/<pool>.json`` in the store its samplers
write their shards to (the trainer's buffer store)::

    {"schema": "yeto.fleet.member-table/v1", "pool": "<pool>", "shard_base": "<base>",
     "epoch": N, "shards": ["<base>-0", ...], "retired": ["<base>-0", ...], "updated_at": T}

* one writer per document: the holder of the pool's store lock
  (``yeto.launch.pool.store_state``); ``epoch`` grows by one whenever ``shards`` or
  ``retired`` change;
* ``shards``: every shard the pool assigned (a member's, ever: a gone member's shard
  may still hold groups the trainer has not merged); ``retired``: shards with a final
  end record (a removed pool's, ``yeto fleet apply``), so the data end never waits on them.

The trainer (:class:`MemberTableReader`, used by ``sample_buffer.FileSampleBuffer``
when :data:`MEMBER_TABLE_ENV` is set) reads the tables before each round and adds
every new shard to the shards it merges; it never drops a shard (its unmerged groups
stay trainable). A shard is taken only when it starts with its table's ``shard_base``
followed by ``-``: a stray document cannot route another lane's shards here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from typing import Any

SCHEMA = "yeto.fleet.member-table/v1"
MEMBERS_DIR = "fleet/members"
MEMBER_TABLE_ENV = "YETO_RL_MEMBER_TABLE"  # "1": the trainer follows the member tables of its store
DEFAULT_REFRESH_S = 15.0


def table_key(pool: str) -> str:
    if not pool or "/" in pool or pool in (".", ".."):
        raise ValueError(f"bad pool name {pool!r}")
    return f"{MEMBERS_DIR}/{pool}.json"


def enabled(environ: Any) -> bool:
    return str(environ.get(MEMBER_TABLE_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


class MemberTableWriter:
    """One pool's table (module doc); :meth:`publish` writes it when it changed."""

    def __init__(self, objects: Any, pool: str, shard_base: str, *, clock: Callable[[], float] = time.time) -> None:
        self.objects, self.pool, self.shard_base, self.clock = objects, pool, shard_base, clock
        self.key = table_key(pool)
        self._last: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self.epoch: int | None = None

    def _load_epoch(self) -> int:
        data = self.objects.read(self.key)
        if data is None:
            return 0
        try:
            doc = json.loads(data)
            self._last = (tuple(doc.get("shards") or ()), tuple(doc.get("retired") or ()))
            return int(doc.get("epoch") or 0)
        except (ValueError, TypeError, AttributeError):
            return 0

    def publish(self, shards: Iterable[str], retired: Iterable[str] = ()) -> bool:
        """Write the table when ``shards`` / ``retired`` changed (epoch + 1). True: written."""
        if self.epoch is None:
            self.epoch = self._load_epoch()
        now = (tuple(sorted({s for s in shards if s}, key=_order)), tuple(sorted({s for s in retired if s}, key=_order)))
        if now == self._last:
            return False
        epoch = self.epoch + 1
        doc = {"schema": SCHEMA, "pool": self.pool, "shard_base": self.shard_base, "epoch": epoch,
               "shards": list(now[0]), "retired": list(now[1]), "updated_at": self.clock()}
        self.objects.write(self.key, json.dumps(doc).encode())
        self.epoch, self._last = epoch, now
        return True


def _order(shard: str) -> tuple[str, int]:
    base, _, tail = shard.rpartition("-")
    return (base, int(tail)) if tail.isdigit() else (shard, -1)


def read_tables(objects: Any) -> list[dict[str, Any]]:
    """Every member table of the store (unreadable or foreign documents left out)."""
    out = []
    for key in objects.list(MEMBERS_DIR + "/"):
        if not key.endswith(".json"):
            continue
        data = objects.read(key)
        if data is None:
            continue
        try:
            doc = json.loads(data)
        except ValueError:
            continue
        if isinstance(doc, dict) and doc.get("schema") == SCHEMA:
            out.append(doc)
    return out


class MemberTableReader:
    """The trainer's view of the member tables: :meth:`poll` returns the shards to add
    (new since the last poll) and the shards the tables retired, at most every ``refresh_s``."""

    def __init__(self, objects: Any, *, refresh_s: float = DEFAULT_REFRESH_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.objects, self.refresh_s, self.clock = objects, float(refresh_s), clock
        self.epochs: dict[str, int] = {}
        self.shards: set[str] = set()
        self.retired: set[str] = set()
        self._polled_at: float | None = None
        self.errors = 0

    @property
    def epoch(self) -> int:
        """The sum of every table's epoch (each only grows): changes whenever one does."""
        return sum(self.epochs.values())

    def poll(self, *, force: bool = False) -> tuple[set[str], set[str]]:
        now = self.clock()
        if not force and self._polled_at is not None and now - self._polled_at < self.refresh_s:
            return set(), set()
        self._polled_at = now
        try:
            tables = read_tables(self.objects)
        except Exception:  # noqa: BLE001 - the next poll reads again; the consumed set stays
            self.errors += 1
            return set(), set()
        added: set[str] = set()
        retired: set[str] = set()
        for doc in tables:
            pool, epoch = str(doc.get("pool")), int(doc.get("epoch") or 0)
            if epoch <= self.epochs.get(pool, -1):
                continue
            self.epochs[pool] = epoch
            prefix = f"{doc.get('shard_base')}-"
            for shard in doc.get("shards") or ():
                if str(shard).startswith(prefix) and shard not in self.shards:
                    self.shards.add(str(shard))
                    added.add(str(shard))
            for shard in doc.get("retired") or ():
                if str(shard).startswith(prefix) and shard not in self.retired:
                    self.retired.add(str(shard))
                    retired.add(str(shard))
        return added, retired
