"""Machine offers a cloud rollout provider walks through, cheapest usable first.

An :class:`Offer` is one way to get one rollout node: sky ``Resources`` keyword
arguments plus the hourly price of the whole node. A provider orders the offer
at its cursor; a failed launch moves the cursor to the next offer (the pool's
breaker paces the retries), a successful one moves it back to the top.

:func:`candidates` is the Nebius candidate list of the spot watchdog (GPU kind x
region x spot / on-demand): on-demand H100 first, then every spot candidate
cheapest first, then on-demand H200 when allowed. :func:`offers_from_config`
reads either an explicit ``offers`` list or a ``candidates`` section of a
provider config.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeto.cost import prices as _prices

# Default 8-GPU hourly prices (USD), (gpu, region) -> (on-demand, spot): the Nebius rows of the
# unified price table (yeto.cost.prices; read with `sky gpus list H100:8|H200:8 --infra nebius
# --all-regions`, each row dated); a ``prices`` file replaces the table. H200 is NOT the same price
# as H100 (spot H100 eu-north1 $6.32, cheapest spot H200 $11.20). A region missing here has no such
# preset in the sky catalog; Nebius quotas: compute.instance.gpu.h100 only in eu-north1, gpu.h200 in
# eu-north1/eu-west1/us-central1.
PRICES_8GPU: dict[tuple[str, str], tuple[float, float]] = _prices.regional_8gpu("nebius")


@dataclass(frozen=True)
class Offer:
    key: str
    resources: dict[str, Any] = field(default_factory=dict)
    price_usd_h: float | None = None

    def record(self) -> dict[str, Any]:
        return {"key": self.key, "resources": dict(self.resources), "price_usd_h": self.price_usd_h}


@dataclass(frozen=True)
class Candidate:
    gpu: str
    region: str
    spot: bool
    price_hr: float  # for the whole VM (``count`` GPUs)
    count: int = 8

    @property
    def key(self) -> str:
        return f"{self.gpu}:{self.count}@{self.region}/{'spot' if self.spot else 'od'}"

    def offer(self, infra: str = "nebius") -> Offer:
        return Offer(key=self.key, price_usd_h=self.price_hr,
                     resources={"infra": f"{infra}/{self.region}", "accelerators": f"{self.gpu}:{self.count}",
                                "use_spot": self.spot})


def candidates(*, gpus: Iterable[str] = ("H100", "H200"), regions: Iterable[str] | None = None,
               count: int = 8, prices: dict[tuple[str, str], tuple[float, float]] | None = None,
               allow_ondemand_h200: bool = False, spot_only: bool = False,
               availability: dict[tuple[str, str], int] | None = None) -> list[Candidate]:
    """On-demand H100 first, then spot cheapest first (better availability first
    at equal price), then on-demand H200 if allowed."""
    prices = PRICES_8GPU if prices is None else prices
    gpus = [g.upper() for g in gpus]
    regs = set(regions) if regions is not None else None
    scale = count / 8.0
    avail = availability or {}
    pool = [(g, r, od * scale, sp * scale) for (g, r), (od, sp) in prices.items()
            if g in gpus and (regs is None or r in regs)]
    out: list[Candidate] = []
    if not spot_only:
        out += [Candidate(g, r, False, od, count) for g, r, od, _ in sorted(pool, key=lambda x: x[2]) if g == "H100"]
    spot = sorted(pool, key=lambda x: (x[3], -avail.get((x[0], x[1]), 0)))
    out += [Candidate(g, r, True, sp, count) for g, r, _, sp in spot]
    if allow_ondemand_h200 and not spot_only:
        out += [Candidate(g, r, False, od, count) for g, r, od, _ in sorted(pool, key=lambda x: x[2]) if g == "H200"]
    return out


def load_prices(path: str) -> dict[tuple[str, str], tuple[float, float]]:
    """A price table file: ``[[gpu, region, on_demand_usd_hr, spot_usd_hr], ...]`` (8 GPUs)."""
    rows = json.loads(Path(path).expanduser().read_text())
    return {(str(g).upper(), str(r)): (float(od), float(sp)) for g, r, od, sp in rows}


OFFER_KEYS = frozenset({"key", "resources", "price_usd_h"})
CANDIDATE_KEYS = frozenset({"gpus", "regions", "count", "prices", "allow_ondemand_h200", "spot_only", "infra"})


def max_price(offers: list[Offer]) -> float | None:
    """The dearest node hour a pool of ``offers`` may order (the price a budget must assume
    before it knows which offer an order lands on: a spot pool whose spot offers fail falls
    back to its on-demand one). None when any offer is unpriced: such a node would be
    counted at $0."""
    if not offers or any(o.price_usd_h is None for o in offers):
        return None
    return max(float(o.price_usd_h) for o in offers)  # type: ignore[arg-type]


def offers_from_config(section: dict[str, Any]) -> list[Offer]:
    """The offers of a provider config: ``offers`` (``[{resources, price_usd_h, key}]``),
    else ``candidates`` (the watchdog's Nebius list), else one offer of ``resources``
    (with ``price_usd_h``). An offer without ``price_usd_h`` takes the unified price
    table's price of its node (:func:`yeto.cost.prices.offer_usd_h`) when the table has one."""
    if section.get("offers") is not None and section.get("candidates") is not None:
        raise ValueError("sky provider: set 'offers' or 'candidates', not both")
    if section.get("offers") is not None:
        out = []
        for i, doc in enumerate(section["offers"]):
            if not isinstance(doc, dict) or not isinstance(doc.get("resources"), dict):
                raise ValueError(f"sky provider: offers[{i}] needs a 'resources' object")
            unknown = set(doc) - OFFER_KEYS
            if unknown:
                raise ValueError(f"sky provider: offers[{i}]: unknown keys {sorted(unknown)}")
            res = dict(doc["resources"])
            out.append(Offer(key=str(doc.get("key") or _key(res)), resources=res,
                             price_usd_h=_stated_or_table(doc.get("price_usd_h"), res, f"offers[{i}]")))
        if not out:
            raise ValueError("sky provider: offers is empty")
        return out
    if section.get("candidates") is not None:
        doc = dict(section["candidates"])
        unknown = set(doc) - CANDIDATE_KEYS
        if unknown:
            raise ValueError(f"sky provider: candidates: unknown keys {sorted(unknown)}")
        infra = str(doc.pop("infra", "nebius"))
        prices = doc.pop("prices", None)
        cands = candidates(prices=load_prices(prices) if isinstance(prices, str) else None, **doc)
        if not cands:
            raise ValueError("sky provider: candidates selects no offer (gpus / regions / prices)")
        return [c.offer(infra) for c in cands]
    if not isinstance(section.get("resources"), dict):
        raise ValueError("sky provider: resources (or offers / candidates) is required")
    res = dict(section["resources"])
    price = _stated_or_table(section.get("price_usd_h"), res, "resources")
    return [Offer(key=_key(res), resources=res, price_usd_h=price)]


def _stated_or_table(value: Any, resources: dict[str, Any], where: str) -> float | None:
    """The offer's stated ``price_usd_h``, else the unified price table's price of the node its
    resources order (None: neither: the offer is unpriced)."""
    stated = _price(value, where)
    return _prices.offer_usd_h(resources) if stated is None else stated


def _price(value: Any, where: str) -> float | None:
    """An offer's node hour in USD: unset, or a finite number >= 0 (a negative or nan
    price is what the pool bills a node launched with this offer: the spend cap would
    never trip)."""
    if value is None:
        return None
    try:
        price = None if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        price = None
    if price is None or not math.isfinite(price) or price < 0:
        raise ValueError(f"sky provider: {where} price_usd_h must be a finite number >= 0, not {value!r}")
    return price


def _key(resources: dict[str, Any]) -> str:
    """An offer's key when its config names none: ``<accelerators>@<where>[/<instance_type>]/<spot|od>``.
    The instance type is part of it when the resources name one (two offers of one region and
    GPU on two instance types are two offers: their ledger stats and breakers are their own);
    without one the key is as before."""
    where = resources.get("infra") or "/".join(str(resources[k]) for k in ("cloud", "region") if resources.get(k))
    spot = "spot" if resources.get("use_spot") else "od"
    kind = resources.get("instance_type")
    return f"{resources.get('accelerators', '?')}@{where or 'any'}/{f'{kind}/' if kind else ''}{spot}"


def legacy_key(offer: Offer) -> str | None:
    """The key ``offer`` had before instance types were part of the generated key: an offer
    whose key is generated and whose resources name an instance type; None otherwise (its
    key did not change)."""
    res = offer.resources
    if not res.get("instance_type") or offer.key != _key(res):
        return None
    return _key({k: v for k, v in res.items() if k != "instance_type"})


def legacy_aliases(offers: Iterable[Offer]) -> dict[str, str]:
    """``{key: legacy key}`` of the offers whose history (ledger stats, cooling breakers) may
    still be under their :func:`legacy_key`: only where that key names this offer alone. Two
    offers that shared it (the collision the instance type now tells apart), or another
    offer's current key, mixed their history there: it is read as nobody's. Judged by
    ``offers`` alone: another instance type of the same GPU and region that an older run
    ordered (an offer since removed, another pool's) counted under the same older key, and
    that history is read as this offer's."""
    offers = list(offers)
    current = {o.key for o in offers}
    old = {o.key: legacy_key(o) for o in offers}
    shared: dict[str, int] = {}
    for v in old.values():
        if v:
            shared[v] = shared.get(v, 0) + 1
    return {k: v for k, v in old.items() if v and shared[v] == 1 and v not in current}
