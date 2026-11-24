"""The fleet's shard number allocator, kept in the run's store.

A pool names the shards of its samplers ``<shard_base>-<n>``. Without this
module ``n`` comes from the pool's own ``next_shard`` in its local state file:
lose that file (a new control host, a state reset) and the numbers start over,
and ``shard_limit`` caps how many members a pool may ever order.

Here every number is issued by the store, once:
``fleet/shards/<scope>/<n:08d>.json`` is created with an exclusive create
(S3 / R2 ``If-None-Match: *``); a number whose record exists is never issued
again, whoever asks and from whatever host. A claimer that loses the create
lists again and tries the new highest + 1. ``scope`` is the pool's
``shard_base`` (unique across the pools of a fleet), so the shard *names* are
unique across the fleet and the first numbers of a pool are the ones its
trainer's initial ``--rl-buffer-shards`` list already names; the numbers grow
monotonically per scope across pools, controllers and control hosts.

* :meth:`ShardAllocator.allocate` issues the next number (at least ``floor``:
  a pool imported from a local state file starts past its ``next_shard``).
* :meth:`ShardAllocator.mark_ended` records that a shard's data ended (its
  sampler wrote the end, or the pool retired it): :meth:`ended` answers from
  the store, and the number is never issued again anyway.
* :meth:`ShardAllocator.issued` lists what was issued (status, the member table).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

SHARDS_DIR = "fleet/shards"
_NUMBER = re.compile(r"(\d{8,})\.json")
MAX_RACES = 1000


class ShardAllocError(RuntimeError):
    pass


def _safe(scope: str) -> str:
    if not scope or "/" in scope or scope in (".", ".."):
        raise ShardAllocError(f"bad shard scope {scope!r}")
    return scope


class ShardAllocator:
    def __init__(self, objects: Any, scope: str, *, owner: str = "", clock: Callable[[], float] = time.time,
                 root: str = SHARDS_DIR) -> None:
        self.objects, self.scope, self.owner, self.clock = objects, _safe(scope), owner, clock
        self.root = root.rstrip("/")

    @property
    def prefix(self) -> str:
        return f"{self.root}/{self.scope}/"

    def key(self, n: int) -> str:
        return f"{self.prefix}{int(n):08d}.json"

    def ended_key(self, shard: str) -> str:
        return f"{self.prefix}ended/{shard}.json"

    def numbers(self) -> list[int]:
        out = []
        for key in self.objects.list(self.prefix):
            match = _NUMBER.fullmatch(key[len(self.prefix):])
            if match is not None:
                out.append(int(match.group(1)))
        return sorted(out)

    def high(self) -> int:
        """The highest number issued in this scope, -1 when none."""
        numbers = self.numbers()
        return numbers[-1] if numbers else -1

    def allocate(self, *, floor: int = 0, token: Any = None, base: str | None = None) -> int:
        """Issue the next number of this scope (>= ``floor``); its record names the pool,
        the lock token that issued it and the shard name."""
        nonce = uuid.uuid4().hex
        for _ in range(MAX_RACES):
            n = max(int(floor), self.high() + 1)
            shard = f"{base or self.scope}-{n}"
            record = {"n": n, "shard": shard, "scope": self.scope, "owner": self.owner, "token": token,
                      "nonce": nonce, "at": self.clock()}
            if self.objects.create(self.key(n), json.dumps(record).encode()):
                return n
            doc = self._doc(self.key(n)) or {}
            if doc.get("nonce") == nonce:  # our own create, its answer lost
                return n
        raise ShardAllocError(f"shard scope {self.scope}: lost {MAX_RACES} creates in a row")

    def issued(self) -> list[dict[str, Any]]:
        return [doc for n in self.numbers() if (doc := self._doc(self.key(n))) is not None]

    def mark_ended(self, shard: str, **detail: Any) -> bool:
        """Record that ``shard``'s data ended (idempotent: the first record stays)."""
        doc = {"shard": shard, "at": self.clock(), **detail}
        return bool(self.objects.create(self.ended_key(shard), json.dumps(doc, default=str).encode()))

    def ended(self, shard: str) -> dict[str, Any] | None:
        return self._doc(self.ended_key(shard))

    def ended_shards(self) -> list[str]:
        base = f"{self.prefix}ended/"
        return [key[len(base):-len(".json")] for key in self.objects.list(base) if key.endswith(".json")]

    def _doc(self, key: str) -> dict[str, Any] | None:
        data = self.objects.read(key)
        if data is None:
            return None
        try:
            doc = json.loads(data)
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None
