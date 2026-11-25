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
stay trainable). Lanes: a table is followed only when its ``shard_base`` is in one of the
trainer's lanes (:data:`MEMBER_LANES_ENV`: comma-separated lanes, ``*`` every table of
the store; an island lane ``i<k>`` matches a ``shard_base`` ending in ``-i<k>``, any other
lane ``L`` matches ``L`` and ``L-...`` (:func:`lane_matches`); unset: the bases of the
trainer's initial ``--rl-buffer-shards``), its
``pool`` names its own key, and a shard is taken only when it starts with the table's
``shard_base`` followed by ``-``: a table of another lane is ignored (reported once), so
it cannot route that lane's shards into this trainer. ``yeto run up`` sets ``*`` for a
single trainer (its store prefix is its one lane, and a pool added while it trains has a
new ``shard_base``); a fleet's island k trainer gets ``i<k>``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from typing import Any

SCHEMA = "yeto.fleet.member-table/v1"
MEMBERS_DIR = "fleet/members"
MEMBER_TABLE_ENV = "YETO_RL_MEMBER_TABLE"  # "1": the trainer follows the member tables of its store
MEMBER_LANES_ENV = "YETO_RL_MEMBER_LANES"  # the trainer's lanes (module doc); unset: its initial shard bases
DEFAULT_REFRESH_S = 15.0


def table_key(pool: str) -> str:
    if not pool or "/" in pool or pool in (".", ".."):
        raise ValueError(f"bad pool name {pool!r}")
    return f"{MEMBERS_DIR}/{pool}.json"


def enabled(environ: Any) -> bool:
    return str(environ.get(MEMBER_TABLE_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def shard_base(shard: str) -> str:
    """``<base>-<n>`` -> ``<base>`` (a name without a numbered tail is its own base)."""
    base, _, tail = str(shard).rpartition("-")
    return base if base and tail.isdigit() else str(shard)


def lanes_from(environ: Any, initial_shards: Iterable[str]) -> frozenset[str] | None:
    """The trainer's lanes: :data:`MEMBER_LANES_ENV` (``*``: None, every table), else the
    bases of its initial shards."""
    raw = str(environ.get(MEMBER_LANES_ENV, "")).strip()
    if raw:
        lanes = {part.strip() for part in raw.split(",") if part.strip()}
        return None if "*" in lanes else frozenset(lanes)
    return frozenset(shard_base(s) for s in initial_shards)


_ISLAND_LANE = re.compile(r"-?i\d+")


def lane_matches(shard_base: str, lane: str) -> bool:
    """Is a table's ``shard_base`` in ``lane``? ``*``: every one. An island lane (``i<k>``
    or ``-i<k>``) matches by suffix: a fleet compiles island k's pools to
    ``<base>-i<k>`` (``r-sglang-i2``). Any other lane matches by prefix: ``L`` itself or
    ``L-...``."""
    if lane == "*":
        return True
    if lane.startswith("-") or _ISLAND_LANE.fullmatch(lane):
        suffix = lane if lane.startswith("-") else f"-{lane}"
        return shard_base == suffix.lstrip("-") or shard_base.endswith(suffix)
    return shard_base == lane or shard_base.startswith(f"{lane}-")


def in_lanes(base: str, lanes: frozenset[str] | None) -> bool:
    return lanes is None or any(lane_matches(base, lane) for lane in lanes)


class MemberTableWriter:
    """One pool's table (module doc); :meth:`publish` writes it when it changed."""

    def __init__(self, objects: Any, pool: str, shard_base: str, *, clock: Callable[[], float] = time.time) -> None:
        self.objects, self.pool, self.shard_base, self.clock = objects, pool, shard_base, clock
        self.key = table_key(pool)
        self._last: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self.epoch: int | None = None

    def _stored(self) -> tuple[int, tuple[tuple[str, ...], tuple[str, ...]] | None]:
        data = self.objects.read(self.key)
        if data is None:
            return 0, None
        try:
            doc = json.loads(data)
            return int(doc.get("epoch") or 0), (tuple(doc.get("shards") or ()), tuple(doc.get("retired") or ()))
        except (ValueError, TypeError, AttributeError):
            return 0, None

    def publish(self, shards: Iterable[str], retired: Iterable[str] = ()) -> bool:
        """Write the table when ``shards`` / ``retired`` changed. The stored epoch is read
        again before each write, so the epoch only grows even past a table another holder
        wrote meanwhile (a reader never sees it go back). True: written."""
        if self.epoch is None:
            self.epoch, self._last = self._stored()
        now = (tuple(sorted({s for s in shards if s}, key=_order)), tuple(sorted({s for s in retired if s}, key=_order)))
        if now == self._last:
            return False
        stored, _ = self._stored()
        epoch = max(self.epoch, stored) + 1
        doc = {"schema": SCHEMA, "pool": self.pool, "shard_base": self.shard_base, "epoch": epoch,
               "shards": list(now[0]), "retired": list(now[1]), "updated_at": self.clock()}
        self.objects.write(self.key, json.dumps(doc).encode())
        self.epoch, self._last = epoch, now
        return True


def _order(shard: str) -> tuple[str, int]:
    base, _, tail = shard.rpartition("-")
    return (base, int(tail)) if tail.isdigit() else (shard, -1)


def read_tables(objects: Any) -> list[dict[str, Any]]:
    """Every member table of the store (unreadable documents, other schemas and a table
    whose ``pool`` is not its own key left out)."""
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
        if (isinstance(doc, dict) and doc.get("schema") == SCHEMA
                and key.rsplit("/", 1)[-1] == f"{doc.get('pool')}.json"):
            out.append(doc)
    return out


def table_shards(doc: dict[str, Any], field: str = "shards") -> list[str]:
    """The shards of ``field`` that start with the table's ``shard_base`` and ``-``."""
    prefix = f"{doc.get('shard_base')}-"
    return [str(s) for s in doc.get(field) or () if str(s).startswith(prefix)]


class MemberTableReader:
    """The trainer's view of the member tables: :meth:`poll` returns the shards to add
    (new since the last poll) and the shards the tables retired, at most every ``refresh_s``."""

    def __init__(self, objects: Any, *, refresh_s: float = DEFAULT_REFRESH_S,
                 clock: Callable[[], float] = time.monotonic, lanes: frozenset[str] | None = None) -> None:
        self.objects, self.refresh_s, self.clock = objects, float(refresh_s), clock
        self.lanes = lanes  # None: every table of the store (module doc)
        self.ignored: dict[str, str] = {}  # pool -> shard_base of a table outside the lanes
        self.ignored_new: dict[str, str] = {}  # of them: first seen by the last poll
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
        self.ignored_new = {}
        for doc in tables:
            pool, epoch = str(doc.get("pool")), int(doc.get("epoch") or 0)
            if not in_lanes(str(doc.get("shard_base")), self.lanes):
                if pool not in self.ignored:
                    self.ignored[pool] = self.ignored_new[pool] = str(doc.get("shard_base"))
                continue
            if epoch <= self.epochs.get(pool, -1):
                continue
            self.epochs[pool] = epoch
            for shard in table_shards(doc):
                if shard not in self.shards:
                    self.shards.add(shard)
                    added.add(shard)
            for shard in table_shards(doc, "retired"):
                if shard not in self.retired:
                    self.retired.add(shard)
                    retired.add(shard)
        return added, retired
