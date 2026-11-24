"""A pool's control state in the run's store (``fleet/pools/<pool>/``).

The pool state (members, orders, shards, budget use, pending handoffs: the
:meth:`~.island_pool.IslandPool.snapshot` document) used to live only in a file
on the control host (``~/.yeto/nc/pool-<shard_base>.json``). Lose the host and
the pool loses track of billed machines; start a second controller and both
drive the pool. Here the document is ``fleet/pools/<pool>/state.json`` in the
run's store, written only by the holder of the pool's lock
(``fleet/pools/<pool>/lock``, :class:`~.store_lock.StoreLock`): every save checks
the lock's fencing token first and a fenced holder raises
:class:`~.store_lock.LockLost` instead of writing.

The local state file is imported once, read only: when the store has no state
for the pool yet and the local file exists, its document is written to the
store with ``imported_from``; from then on the store wins and the local file is
never read again for this pool (it is left as it is: an older yeto still reads
it). The shard numbers come from :class:`~.shard_alloc.ShardAllocator` (scope:
the pool's ``shard_base``), starting past the imported ``next_shard``.

:func:`open_pool_control` bundles the three for a pool: the lock (or
:class:`PoolLocked` when a running lease holds it), the state store and the
allocator.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .shard_alloc import ShardAllocator
from .store_lock import DEFAULT_TTL_S, StoreLock

POOLS_DIR = "fleet/pools"


class PoolLocked(RuntimeError):
    """Another controller holds the pool's lock (its lease runs)."""

    def __init__(self, pool: str, holder: dict[str, Any] | None) -> None:
        who = (holder or {}).get("holder")
        until = (holder or {}).get("until")
        super().__init__(f"pool {pool}: another controller holds its store lock (holder {who}, lease until "
                         f"{until}); it drives the pool, this one does not")
        self.pool, self.holder = pool, holder


def pool_dir(pool: str) -> str:
    if not pool or "/" in pool or pool in (".", ".."):
        raise ValueError(f"bad pool name {pool!r}")
    return f"{POOLS_DIR}/{pool}"


def state_key(pool: str) -> str:
    return f"{pool_dir(pool)}/state.json"


def lock_prefix(pool: str) -> str:
    return f"{pool_dir(pool)}/lock"


def read_state(objects: Any, pool: str) -> dict[str, Any] | None:
    """The pool's state document in the store (read only; None: none yet)."""
    data = objects.read(state_key(pool))
    if data is None:
        return None
    doc = json.loads(data)
    return doc if isinstance(doc, dict) else None


class PoolStateStore:
    """Saves and loads one pool's state document under its lock (module doc)."""

    def __init__(self, objects: Any, pool: str, lock: StoreLock, *, clock: Callable[[], float] = time.time) -> None:
        self.objects, self.pool, self.lock, self.clock = objects, pool, lock, clock
        self.key = state_key(pool)
        self.imported_from: str | None = None

    def load(self, legacy: Path | None = None) -> dict[str, Any] | None:
        """The state document: the store's; else the local file at ``legacy`` imported
        once (written to the store as it is, with ``imported_from``); else None."""
        doc = read_state(self.objects, self.pool)
        if doc is not None:
            return doc
        if legacy is None or not Path(legacy).exists():
            return None
        doc = json.loads(Path(legacy).read_text())
        if not isinstance(doc, dict):
            return None
        doc = dict(doc, imported_from=str(legacy), imported_at=self.clock())
        self.imported_from = str(legacy)
        self.save(doc)
        return doc

    def save(self, doc: dict[str, Any]) -> None:
        """Write the document after checking the fencing token (raises ``LockLost``)."""
        self.lock.check(force=True)  # one LIST: no write after another controller took the lock
        out = dict(doc, control={"token": self.lock.token, "holder": self.lock.holder, "saved_at": self.clock()})
        self.objects.write(self.key, json.dumps(out, indent=1, default=str).encode())

    def renew(self) -> None:
        self.lock.renew()

    def release(self) -> None:
        self.lock.release()

    close = release


@dataclass
class PoolControl:
    """A pool's store lock, state store and shard allocator (:func:`open_pool_control`)."""

    lock: StoreLock
    state: PoolStateStore
    allocator: ShardAllocator

    def close(self) -> None:
        self.lock.release()


def open_pool_control(objects: Any, pool: str, shard_base: str, *, holder: str | None = None,
                      ttl_s: float = DEFAULT_TTL_S, clock: Callable[[], float] = time.time) -> PoolControl:
    """Claim pool ``pool``'s store lock and build its state store and allocator. Raises
    :class:`PoolLocked` while another controller's lease runs."""
    lock = StoreLock.acquire(objects, lock_prefix(pool), holder=holder, ttl_s=ttl_s, clock=clock)
    if lock is None:
        raise PoolLocked(pool, StoreLock.holder_of(objects, lock_prefix(pool), now=clock()))
    return PoolControl(lock, PoolStateStore(objects, pool, lock, clock=clock),
                       ShardAllocator(objects, shard_base, owner=pool, clock=clock))
