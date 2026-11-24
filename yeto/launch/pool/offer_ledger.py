"""The offer ledger: what each machine offer did, measured, across runs.

One row per offer key (a sky offer's key, ``nc``, ``modal:B300x8``, ...): how long its
machines took from the order to a ready sampler (``ordered`` to ``member_ready``:
p50 / p90 of the newest :data:`MAX_SAMPLES`), how many orders came up at all (the
start success rate), how many ready machines were taken away (reclaimed: preempted,
lost) against those that ended any other way, the ready hours they ran (the reclaim
rate per hour, the hazard ``lambda``), and the sampler throughput per model.

The ledger is fed the pool's own events (:meth:`OfferLedger.observe`, any event
prefix: ``nc_elastic_ordered`` and ``ordered`` are one event) and kept in the run
store as one JSON document (:meth:`OfferLedger.load` / :meth:`OfferLedger.save`, at
:data:`DEFAULT_KEY` of the store root, so it accumulates over runs). It decides
nothing itself: the pool's replacement lead (:func:`replace_lead_s`) and the
capacity broker read it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_KEY = "fleet/offer_ledger.json"
FORMAT = 1
MAX_SAMPLES = 200  # start-time samples kept per offer (the newest)
MIN_SAMPLES = 5  # fewer measured starts: the replacement lead keeps its prior
# what the pool calls the events this ledger reads (the name after its prefix)
ORDERED, READY = "ordered", "member_ready"
START_FAILED = frozenset({"order_failed", "grant_timeout", "start_timeout"})
RECLAIMED = frozenset({"evicted", "node_lost"})
ENDED = frozenset({"released", "expired", "released_externally", "released_by_earlier_pool", "ended_for_balance"})
KNOWN = frozenset({ORDERED, READY, "offer_failed", *START_FAILED, *RECLAIMED, *ENDED})


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
    def load(cls, store: Any, key: str = DEFAULT_KEY) -> OfferLedger:
        """Read the ledger from ``store`` (an object store: ``read(key) -> bytes | None``);
        an absent document is an empty ledger."""
        raw = store.read(key)
        return cls() if raw is None else cls.from_doc(json.loads(raw))

    def save(self, store: Any, key: str = DEFAULT_KEY) -> None:
        """Write the ledger to ``store`` (``write(key, bytes)``)."""
        store.write(key, json.dumps(self.to_doc(), sort_keys=True).encode())


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
    """The ledger as one pool uses it: an event sink (``EventLog.sinks``) that counts the
    pool's events under the member's offer (``offer_of(ticket)``: a sky pool's offer key;
    else ``default_offer``: the pool's one offer) and saves the ledger to ``store`` after
    a machine came up or ended (best effort: a failed save is retried at the next one), and
    the replacement lead of a member (:meth:`lead_s`)."""

    ledger: OfferLedger
    default_offer: str | None = None
    offer_of: Callable[[str], str | None] | None = None
    store: Any = None
    key: str = DEFAULT_KEY
    log: Callable[[str], None] = print
    save_error: str | None = None

    def __call__(self, doc: Mapping[str, Any]) -> None:
        name = self.ledger.observe(doc, offer_of=self._offer_of, default_offer=self.default_offer)
        if name is None or name == ORDERED or self.store is None:
            return
        try:
            self.ledger.save(self.store, self.key)
            self.save_error = None
        except Exception as error:  # noqa: BLE001 - the ledger never stops a pool
            if self.save_error is None:
                self.log(f"pool: offer ledger not saved: {error!r}"[:300])
            self.save_error = repr(error)

    def _offer_of(self, ticket: str) -> str | None:
        if self.offer_of is None:
            return None
        try:
            return self.offer_of(ticket)
        except Exception:  # noqa: BLE001 - unknown: the pool's default offer
            return None

    def offer(self, ticket: str) -> str | None:
        known = self.ledger.tickets.get(ticket) or {}
        return known.get("offer") or self._offer_of(ticket) or self.default_offer

    def lead_s(self, ticket: str, prior_s: float) -> float:
        offer = self.offer(ticket)
        return replace_lead_s(None if offer is None else self.ledger.get(offer), prior_s=prior_s)[0]
