"""The offer ledger: what each machine offer did, measured, across runs.

One row per offer key (a sky offer's key, ``nc``, ``modal:B300x8``, ...): how long its
machines took from the order to a ready sampler (``ordered`` to ``member_ready``:
p50 / p90 of the newest :data:`MAX_SAMPLES`), how many orders came up at all (the
start success rate), how many ready machines were taken away (reclaimed: preempted,
lost) against those that ended any other way, the ready hours they ran (the reclaim
rate per hour, the hazard ``lambda``), and the sampler throughput per model.

The ledger is fed the pool's own events (:meth:`OfferLedger.observe`, any event
prefix: ``nc_elastic_ordered`` and ``ordered`` are one event) and kept in the store
under :data:`PREFIX` at the bucket root, so it accumulates over runs: every writer (one
pool of one run) writes only its own shard object (:func:`shard_key`), and a reader
merges every shard (:meth:`OfferLedger.load_all`). Two pools or two runs never
overwrite each other, and a shard that cannot be read is never replaced: its writer
writes nothing until it has read it. The ledger decides nothing itself: the pool's
replacement lead (:func:`replace_lead_s`) and the capacity broker read it.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

PREFIX = "fleet/offer_ledger/"
FORMAT = 1
MAX_SAMPLES = 200  # start-time samples kept per offer (the newest)
MIN_SAMPLES = 5  # fewer measured starts: the replacement lead keeps its prior
# what the pool calls the events this ledger reads (the name after its prefix)
ORDERED, READY = "ordered", "member_ready"
START_FAILED = frozenset({"order_failed", "grant_timeout", "start_timeout"})
RECLAIMED = frozenset({"evicted", "node_lost"})
ENDED = frozenset({"released", "expired", "released_externally", "released_by_earlier_pool", "ended_for_balance"})
# a machine moving to another pool (capacity broker phase 2): the source's shard forgets it
# without an end, the destination's takes it over with its offer and order / ready times
# without an order or a start, so a moved machine is one machine in the merged ledger
MOVED_OUT, MOVED_IN = "member_moved_out", "member_moved_in"
KNOWN = frozenset({ORDERED, READY, "offer_failed", *START_FAILED, *RECLAIMED, *ENDED, MOVED_OUT, MOVED_IN})


def percentile(values: list[float], q: float) -> float | None:
    """The ``q`` quantile (0..1) of ``values``, linear between the closest ranks (None: no value)."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class OfferStats:
    """The measurements of one offer."""

    offer: str
    orders: int = 0
    ready: int = 0
    start_failed: int = 0
    reclaimed: int = 0  # ready machines taken away (preempted, lost)
    ended: int = 0  # ready machines that ended any other way (released, lease end)
    ready_s: float = 0.0  # seconds its machines ran after they were ready (ended ones)
    start_s: list[float] = field(default_factory=list)  # order -> ready, the newest MAX_SAMPLES
    # model -> {"tokens": generated tokens, "s": seconds they took}
    throughput: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def start_p50_s(self) -> float | None:
        return percentile(self.start_s, 0.5)

    @property
    def start_p90_s(self) -> float | None:
        return percentile(self.start_s, 0.9)

    @property
    def start_success_rate(self) -> float | None:
        """Ready machines over the orders that came up or failed to (None: none of either yet)."""
        done = self.ready + self.start_failed
        return None if done == 0 else self.ready / done

    @property
    def reclaim_rate(self) -> float | None:
        """Ready machines reclaimed over those that ended (None: none ended yet)."""
        done = self.reclaimed + self.ended
        return None if done == 0 else self.reclaimed / done

    @property
    def hazard_per_h(self) -> float | None:
        """Reclaims per ready machine hour (the ``lambda`` of the lifetime model; None: no ready hour yet)."""
        return None if self.ready_s <= 0 else self.reclaimed / (self.ready_s / 3600.0)

    def tokens_per_s(self, model: str) -> float | None:
        row = self.throughput.get(model) or {}
        return None if not row.get("s") else float(row["tokens"]) / float(row["s"])

    def to_doc(self) -> dict[str, Any]:
        return {"orders": self.orders, "ready": self.ready, "start_failed": self.start_failed,
                "reclaimed": self.reclaimed, "ended": self.ended, "ready_s": round(self.ready_s, 3),
                "start_s": [round(s, 3) for s in self.start_s],
                "throughput": {m: dict(v) for m, v in self.throughput.items()}}

    @classmethod
    def from_doc(cls, offer: str, doc: Mapping[str, Any]) -> OfferStats:
        return cls(offer=offer, orders=int(doc.get("orders", 0)), ready=int(doc.get("ready", 0)),
                   start_failed=int(doc.get("start_failed", 0)), reclaimed=int(doc.get("reclaimed", 0)),
                   ended=int(doc.get("ended", 0)), ready_s=float(doc.get("ready_s", 0.0)),
                   start_s=[float(s) for s in doc.get("start_s") or ()][-MAX_SAMPLES:],
                   throughput={str(m): {"tokens": float(v.get("tokens", 0.0)), "s": float(v.get("s", 0.0))}
                               for m, v in (doc.get("throughput") or {}).items()})

    def add(self, other: OfferStats) -> None:
        """Add ``other``'s measurements (another writer's) to these."""
        self.orders += other.orders
        self.ready += other.ready
        self.start_failed += other.start_failed
        self.reclaimed += other.reclaimed
        self.ended += other.ended
        self.ready_s += other.ready_s
        self.start_s = (self.start_s + other.start_s)[-MAX_SAMPLES:]
        for model, row in other.throughput.items():
            mine = self.throughput.setdefault(model, {"tokens": 0.0, "s": 0.0})
            mine["tokens"] += float(row.get("tokens", 0.0))
            mine["s"] += float(row.get("s", 0.0))

    def summary(self) -> dict[str, Any]:
        return {"offer": self.offer, "orders": self.orders, "starts_measured": len(self.start_s),
                "start_p50_s": self.start_p50_s, "start_p90_s": self.start_p90_s,
                "start_success_rate": self.start_success_rate, "reclaim_rate": self.reclaim_rate,
                "hazard_per_h": self.hazard_per_h}


def _name(event: str) -> str | None:
    """The pool event name after its prefix (``nc_elastic_member_ready`` -> ``member_ready``)."""
    for name in sorted(KNOWN, key=len, reverse=True):
        if event == name or event.endswith("_" + name):
            return name
    return None


@dataclass
class OfferLedger:
    """Every offer's :class:`OfferStats`, plus the machines in flight (ticket -> what is
    known of it) so an end can be told from a start failure and its ready time counted."""

    offers: dict[str, OfferStats] = field(default_factory=dict)
    tickets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def stats(self, offer: str) -> OfferStats:
        if offer not in self.offers:
            self.offers[offer] = OfferStats(offer)
        return self.offers[offer]

    def get(self, offer: str) -> OfferStats | None:
        return self.offers.get(offer)

    # -- accumulation ----------------------------------------------------------
    def observe(self, event: Mapping[str, Any], *, offer_of: Callable[[str], str | None] | None = None,
                default_offer: str | None = None) -> str | None:
        """Count one pool event. Its offer: the event's ``offer``, else the one its ticket was
        ordered with (recorded at ``ordered``), else ``offer_of(ticket)``, else
        ``default_offer``. Returns the counted name (None: not an event of the ledger)."""
        name = _name(str(event.get("event") or ""))
        if name is None or name == "offer_failed" and not event.get("ticket"):
            return None  # an offer_failed without a ticket follows its order_failed: counted once
        ticket = event.get("ticket")
        ticket = None if ticket is None else str(ticket)
        known = self.tickets.get(ticket) if ticket is not None else None
        offer = event.get("offer") or (known or {}).get("offer")
        if offer is None and ticket is not None and offer_of is not None:
            offer = offer_of(ticket)
        offer = offer or default_offer
        if offer is None:
            return None
        t = float(event.get("t") or 0.0)
        if name == MOVED_OUT:
            if ticket is not None:
                self.tickets.pop(ticket, None)
            return name
        if name == MOVED_IN:
            if ticket is not None:
                moved = {"offer": str(offer)}
                for key, field_name in (("ordered", "ordered_at"), ("ready", "ready_at")):
                    if event.get(field_name) is not None:
                        moved[key] = float(event[field_name])
                self.tickets[ticket] = moved
            return name
        row = self.stats(str(offer))
        if name == ORDERED:
            row.orders += 1
            if ticket is not None:
                self.tickets[ticket] = {"offer": str(offer), "ordered": t}
        elif name == READY:
            if known is not None and known.get("ready") is not None:
                return None  # one ready per machine (a restarted sampler joins again)
            row.ready += 1
            took = event.get("order_to_ready_s")
            if took is None and known is not None and known.get("ordered") is not None:
                took = t - float(known["ordered"])
            if took is not None and float(took) >= 0:
                row.start_s.append(float(took))
                del row.start_s[:-MAX_SAMPLES]
            if ticket is not None:
                self.tickets.setdefault(ticket, {"offer": str(offer)})["ready"] = t
        elif name in START_FAILED or name == "offer_failed":
            if known is not None and known.get("ready") is not None:
                return None  # not a start failure: it was up
            row.start_failed += 1
            if ticket is not None:
                self.tickets.pop(ticket, None)
        else:  # the machine ended: reclaimed, or any other end
            if known is None or known.get("ready") is None:
                if name in RECLAIMED and known is not None:
                    row.start_failed += 1  # gone before it was ever ready
                if ticket is not None:
                    self.tickets.pop(ticket, None)
                return name
            if name in RECLAIMED:
                row.reclaimed += 1
            else:
                row.ended += 1
            row.ready_s += max(0.0, t - float(known["ready"]))
            self.tickets.pop(ticket, None)
        return name

    def record_throughput(self, offer: str, model: str, tokens: float, seconds: float) -> None:
        """Add ``tokens`` generated in ``seconds`` by a sampler of ``model`` on ``offer``."""
        if seconds <= 0 or tokens < 0:
            return
        row = self.stats(offer).throughput.setdefault(model, {"tokens": 0.0, "s": 0.0})
        row["tokens"] += float(tokens)
        row["s"] += float(seconds)

    # -- persistence -----------------------------------------------------------
    def to_doc(self) -> dict[str, Any]:
        return {"format": FORMAT, "offers": {k: v.to_doc() for k, v in sorted(self.offers.items())},
                "tickets": {k: dict(v) for k, v in sorted(self.tickets.items())}}

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any] | None) -> OfferLedger:
        doc = doc or {}
        if doc and int(doc.get("format", FORMAT)) > FORMAT:
            raise ValueError(f"offer ledger format {doc.get('format')} is newer than this yeto reads ({FORMAT})")
        return cls(offers={str(k): OfferStats.from_doc(str(k), v) for k, v in (doc.get("offers") or {}).items()},
                   tickets={str(k): dict(v) for k, v in (doc.get("tickets") or {}).items()})

    @classmethod
    def merged(cls, ledgers: Iterable[OfferLedger]) -> OfferLedger:
        """Every ledger's measurements added up (the tickets in flight of all of them)."""
        out = cls()
        for led in ledgers:
            for key, stats in led.offers.items():
                out.stats(key).add(stats)
            out.tickets.update({k: dict(v) for k, v in led.tickets.items()})
        return out

    @classmethod
    def load(cls, store: Any, key: str | None = None) -> OfferLedger:
        """One shard ``key`` from ``store`` (``read(key) -> bytes | None``): absent is an empty
        ledger; a failed read, unreadable JSON or a newer format raises (never taken for
        empty). Without ``key``: every writer's shard merged (:meth:`load_all`)."""
        if key is None:
            return cls.load_all(store)
        raw = store.read(key)
        return cls() if raw is None else cls.from_doc(json.loads(raw))

    @classmethod
    def load_all(cls, store: Any, *, prefix: str = PREFIX, skip: str | None = None,
                 unreadable: dict[str, str] | None = None) -> OfferLedger:
        """Every writer's shard under ``prefix`` merged (``skip``: one shard key left out). A
        shard that does not read raises, or with ``unreadable`` (a dict) is left out and
        recorded there (key -> error); a failed LIST raises either way."""
        parts = []
        for key in store.list(prefix):
            if key == skip or not key.endswith(".json"):
                continue
            try:
                parts.append(cls.load(store, key))
            except Exception as error:  # noqa: BLE001
                if unreadable is None:
                    raise
                unreadable[key] = repr(error)[:300]
        return cls.merged(parts)

    def save(self, store: Any, key: str) -> None:
        """Write this ledger as the shard ``key`` (``write(key, bytes)``): only its own writer does."""
        store.write(key, json.dumps(self.to_doc(), sort_keys=True).encode())


def shard_key(writer: str, prefix: str = PREFIX) -> str:
    """The shard object of one writer (one pool of one run)."""
    return f"{prefix}{re.sub(r'[^A-Za-z0-9._-]', '_', writer)[:200]}.json"


# -- the replacement lead ---------------------------------------------------------

LEAD_MARGIN_FRACTION = 0.25
LEAD_MARGIN_MIN_S = 300.0


def replace_lead_s(stats: OfferStats | None, *, prior_s: float, min_samples: int = MIN_SAMPLES) -> tuple[float, str]:
    """How long before a machine's hard end its replacement is ordered: the offer's measured
    p90 order-to-ready time plus a margin (a quarter of it, at least 5 min) once it has
    ``min_samples`` measured starts; ``prior_s`` before that. Returns the lead and why."""
    p90 = None if stats is None or len(stats.start_s) < min_samples else stats.start_p90_s
    if p90 is None:
        have = 0 if stats is None else len(stats.start_s)
        return float(prior_s), f"prior {prior_s:g} s ({have} measured start(s), {min_samples} needed)"
    margin = max(LEAD_MARGIN_MIN_S, LEAD_MARGIN_FRACTION * p90)
    return p90 + margin, f"p90 {p90:.0f} s of {len(stats.start_s)} measured starts + margin {margin:.0f} s"


# -- a pool's view of the ledger ----------------------------------------------------

@dataclass
class PoolLedger:
    """The ledger as one pool keeps it: an event sink (``EventLog.sinks``) that counts the
    pool's events under the member's offer (``offer_of(ticket)``: a sky pool's offer key;
    else ``default_offer``: the pool's one offer) into ``own`` (this writer's shard
    :func:`shard_key` of ``writer``), and the replacement lead of a member (:meth:`lead_s`,
    from every writer's measurements).

    Store traffic never runs in the pool's tick: the first read (its own shard and every
    other writer's, merged into ``others``) and each save after a machine came up or ended
    run on a background thread (``background=False``: inline, for tests). Until its own
    shard was read the writer writes nothing: a read that fails (a store blip, a newer
    format) is tried again at the next save, and reported once as the pool event
    ``offer_ledger_unreadable`` (sent from the pool's own thread, with its next event)."""

    own: OfferLedger
    writer: str
    default_offer: str | None = None
    offer_of: Callable[[str], str | None] | None = None
    store: Any = None
    log: Callable[[str], None] = print
    event: Callable[..., Any] | None = None  # EventLog: event(name, **detail)
    # () -> whether this writer may write now (False: the pool lost its store lock: a newer
    # controller drives it and writes the shard); None: always
    may_write: Callable[[], bool] | None = None
    background: bool = True
    prefix: str = PREFIX
    # {offer key: the key its history was counted under before} (``offers.legacy_aliases``:
    # the generated key without the instance type, where it names this offer alone): read
    # with the offer's own; written never (new counts go under the offer's key)
    aliases: dict[str, str] = field(default_factory=dict)
    others: OfferLedger = field(default_factory=OfferLedger)
    loaded: bool = False
    save_error: str | None = None
    _lock: Any = field(default_factory=threading.Lock, repr=False)
    _dirty: bool = False
    _thread: Any = None
    _unreadable: str | None = None  # the read error of its own shard, not reported yet
    _reported: bool = False
    _foreign: dict[str, str] = field(default_factory=dict)  # other writers' shards that did not read
    _foreign_reported: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return shard_key(self.writer, self.prefix)

    def start(self) -> None:
        """Read the store now (in the background): the leads see the history before the first save."""
        if self.store is not None:
            self._kick()

    def __call__(self, doc: Mapping[str, Any]) -> None:
        self._report()
        with self._lock:
            name = self.own.observe(doc, offer_of=self._offer_of, default_offer=self.default_offer)
        if name is None or name == ORDERED or self.store is None:
            return
        with self._lock:
            self._dirty = True
        self._kick()

    def _kick(self) -> None:
        if not self.background:
            self._run()
            return
        with self._lock:  # one worker at a time; it clears _thread under this lock before it exits
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="yeto-offer-ledger", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            self.flush()
        finally:
            with self._lock:
                self._thread = None
                again = (self._dirty and self.loaded and self.save_error is None
                         and (self.may_write is None or self.may_write()))
            if again and self.background:  # dirtied between the last write and the exit
                self._kick()

    def flush(self) -> None:
        """Read the store if not read yet, then write this writer's shard while it is dirty."""
        if not self.loaded and not self._read():
            return
        while True:
            with self._lock:
                if not self._dirty:
                    return
                if self.may_write is not None and not self.may_write():
                    return  # the pool lost its store lock: the controller that took it writes now
                self._dirty = False
                doc = OfferLedger.from_doc(self.own.to_doc())
            try:
                doc.save(self.store, self.key)
                self.save_error = None
            except Exception as error:  # noqa: BLE001 - the ledger never stops a pool
                if self.save_error is None:
                    self.log(f"pool: offer ledger not saved: {error!r}"[:300])
                self.save_error = repr(error)
                with self._lock:
                    self._dirty = True  # the next event tries again
                return

    def close(self, timeout: float = 10.0) -> None:
        """At the pool's end: wait for the background worker, then write what is left."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
        if self.store is not None:
            self.save_error = None
            self.flush()

    def _read(self) -> bool:
        """Read this writer's own shard (it alone decides whether the writer may write) and the
        other writers' (one that does not read is left out and reported by its key)."""
        try:
            stored = OfferLedger.load(self.store, self.key)
        except Exception as error:  # noqa: BLE001 - never taken for an empty ledger: nothing is written
            with self._lock:
                if not self._reported:
                    self._unreadable = repr(error)[:300]
            return False
        foreign: dict[str, str] = {}
        try:
            others = OfferLedger.load_all(self.store, prefix=self.prefix, skip=self.key, unreadable=foreign)
        except Exception as error:  # noqa: BLE001 - no LIST: this writer still writes its own shard
            others, foreign = OfferLedger(), {f"{self.prefix}*": repr(error)[:300]}
        with self._lock:
            self.own = OfferLedger.merged([stored, self.own])  # this run's counts on top of the stored ones
            self.others = others
            self._foreign.update({k: v for k, v in foreign.items() if k not in self._foreign_reported})
            self.loaded = True
        return True

    def _report(self) -> None:
        with self._lock:
            why, self._unreadable = self._unreadable, None
            report = why is not None and not self._reported
            if report:
                self._reported = True
            foreign, self._foreign = self._foreign, {}
            self._foreign_reported.update(foreign)
        if report:
            self._emit(self.key, why, "nothing is written to it until it reads")
        for key, error in sorted(foreign.items()):
            self._emit(key, error, "another writer's shard: left out of the merged history")

    def _emit(self, key: str, error: str, what: str) -> None:
        self.log(f"pool: offer ledger {key} not read ({error}): {what}")
        if self.event is not None:
            try:
                self.event("offer_ledger_unreadable", key=key, error=error, own=key == self.key)
            except Exception:  # noqa: BLE001
                pass

    def _offer_of(self, ticket: str) -> str | None:
        if self.offer_of is None:
            return None
        try:
            return self.offer_of(ticket)
        except Exception:  # noqa: BLE001 - unknown: the pool's default offer
            return None

    def offer(self, ticket: str) -> str | None:
        with self._lock:
            known = self.own.tickets.get(ticket) or {}
        return known.get("offer") or self._offer_of(ticket) or self.default_offer

    def stats(self, offer: str) -> OfferStats | None:
        """Every writer's measurements of ``offer`` (the others' as read, this writer's live),
        with those counted under its older key (:attr:`aliases`)."""
        keys = [offer] + ([self.aliases[offer]] if self.aliases.get(offer) else [])
        with self._lock:
            parts = [led.offers[k] for k in keys for led in (self.others, self.own) if k in led.offers]
        if not parts:
            return None
        out = OfferStats(offer)
        for part in parts:
            out.add(part)
        return out

    def lead_s(self, ticket: str, prior_s: float) -> float:
        offer = self.offer(ticket)
        return replace_lead_s(None if offer is None else self.stats(offer), prior_s=prior_s)[0]
