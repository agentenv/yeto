"""A pool's side of the capacity broker: which offer its next order uses (6.3 follow-up).

The broker (:mod:`.broker`) writes a pool's count through the ``pool-scale`` file
(``ScaleFileSink``: ``target`` plus ``broker: {"offers": {"<pool>/<offer>": n},
"refill": "<pool>/<offer>"}``). The pool applies ``target`` as always; this module
reads the ``broker`` part so an order goes where the broker placed the node:

* :func:`read_plan` - the plan of a scale file document (None without a ``broker``
  key: a pool the broker does not drive, the customer's included, is unchanged);
* :func:`next_offer` - the offer of the next order: a node the broker added (an
  offer's count above what ran on it when the broker decided) and not ordered yet
  goes to that offer; any other order (a lost node's replacement) to the refill
  offer; an offer the provider cools down is skipped (the provider's own cursor then
  picks, as without a plan);
* :func:`victim_rank` - on a scale-down, members on an offer over its count go
  first, so a smaller target leaves the mix the broker asked for.

Running members are never moved for a plan (the broker's hysteresis): a plan
only steers new orders and the choice of scale-down victims.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Plan:
    offers: dict[str, int] = field(default_factory=dict)  # provider offer key -> nodes
    refill: str | None = None
    running: dict[str, int] | None = None  # what ran on each offer when the broker decided (None: not sent)


def _name(key: Any) -> str | None:
    """``<pool>/<offer>`` -> the provider's offer key (which may hold ``/`` itself)."""
    if not isinstance(key, str) or "/" not in key:
        return None
    return key.split("/", 1)[1] or None


def read_plan(doc: Mapping[str, Any] | None) -> Plan | None:
    """The broker plan of a ``pool-scale`` document; None when the broker wrote none.
    Malformed entries are dropped (a bad plan never stops the pool: the target holds)."""
    section = (doc or {}).get("broker")
    if not isinstance(section, Mapping):
        return None

    def counts(raw: Any) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, n in (raw.items() if isinstance(raw, Mapping) else ()):
            name = _name(key)
            if name is not None and isinstance(n, int) and not isinstance(n, bool) and n >= 0:
                out[name] = n
        return out

    running = counts(section.get("running")) if isinstance(section.get("running"), Mapping) else None
    return Plan(counts(section.get("offers")), _name(section.get("refill")), running)


def next_offer(plan: Plan | None, held: Iterable[str | None], *, ordered: Mapping[str, int] | None = None,
               cooling: Iterable[str] = (), known: Iterable[str] | None = None) -> str | None:
    """The offer key the next order should use (None: no preference, the provider picks).
    ``held``: the offer of every member that counts now; ``ordered``: orders the pool placed
    on each offer since it read this plan; ``cooling``: offers the provider holds back;
    ``known``: the provider's offers (a plan naming another is ignored)."""
    if plan is None:
        return None
    have = Counter(o for o in held if o)
    skip = set(cooling)
    allowed = set(known) if known is not None else None

    def ok(key: str | None) -> bool:
        return key is not None and key not in skip and (allowed is None or key in allowed)

    done = Counter(ordered or {})
    if plan.running is not None:
        # nodes the broker added on an offer (its count above what ran there at the decision)
        # that the pool has not ordered since; a loss is not one: its replacement goes to refill
        added = {k: n - plan.running.get(k, 0) - done[k] for k, n in plan.offers.items()}
    else:  # a plan without what ran: every offer below its count
        added = {k: n - have[k] for k, n in plan.offers.items()}
    short = sorted(((a, k) for k, a in added.items() if a > 0 and ok(k)),
                   key=lambda t: (-t[0], t[1] != plan.refill, t[1]))
    if short:
        return short[0][1]
    return plan.refill if ok(plan.refill) else None


def victim_rank(plan: Plan | None, offer: str | None, held: Iterable[str | None]) -> int:
    """Sort key part of a scale-down victim: 0 for a member on an offer over its count
    (it goes first), 1 otherwise; always 1 without a plan (the order is unchanged)."""
    if plan is None or offer is None:
        return 1
    return 0 if Counter(o for o in held if o)[offer] > plan.offers.get(offer, 0) else 1
