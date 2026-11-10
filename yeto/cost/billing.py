"""What the resources of one run cost so far.

* Modal app: billed cost from ``modal.billing.workspace_billing_report`` (hourly
  rows, matched by exact app id from the ledger, only hours that overlap the
  span the app belonged to this run), plus an estimate for the time after the
  newest billed hour (billing lags): duration x rate, where the rate is the
  ledger's ``usd_per_hour``, else the price table, else the newest complete
  billed hour of that app.
* NC lease: (released or now) - granted_at, times the lease rate. ``lease_until``
  does NOT bound it: a lease past its end that was never deleted keeps billing.
* sky cluster: price table x wall clock (if the ledger carries gpu / gpus).

A resource with no rate is "unpriced" (``usd=None``) - never $0 - and the
summary says so; the guard warns about unpriced resources.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .ledger import Resource

HOUR = 3600.0
NC_USD_PER_UCR_ENV = "YETO_NC_USD_PER_UCR"
NC_USD_PER_GPU_HOUR_ENV = "YETO_NC_USD_PER_GPU_HOUR"
NC_UCR_PER_GPU_HOUR_ENV = "YETO_NC_UCR_PER_GPU_HOUR"
# National Compute defaults: 1 credit (1e6 ucr) ~ 1 USD, and the 8-GPU MI355X node at
# 3e6 ucr per GPU-hour (24e6 ucr/h, ~24 USD/h); the env above overrides either
DEFAULT_NC_USD_PER_UCR = 1e-6
DEFAULT_NC_UCR_PER_GPU_HOUR = 3_000_000
# Modal list price per GPU-hour, used when neither the ledger nor a price table
# gives a rate (2x8 B300 ~ 136 USD/h). Keys are lower-case Modal gpu names.
DEFAULT_MODAL_USD_PER_GPU_HOUR = {"b300": 8.5, "b200": 6.25, "h200": 4.54, "h100": 3.95,
                                  "a100-80gb": 2.5, "a100": 2.1, "l40s": 1.95}


def default_modal_rate(gpu: Any, gpus: Any) -> float | None:
    unit = DEFAULT_MODAL_USD_PER_GPU_HOUR.get(str(gpu or "").lower().split(":")[0].rstrip("!"))
    return None if unit is None or not gpus else unit * float(gpus)


def _ts(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=UTC)).timestamp()
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()


def floor_hour(t: float) -> float:
    return math.floor(t / HOUR) * HOUR


def hour_label(t: float) -> str:
    return datetime.fromtimestamp(floor_hour(t), UTC).strftime("%Y-%m-%dT%H:00Z")


def split_by_hour(start: float, end: float, usd_per_hour: float) -> dict[str, float]:
    out: dict[str, float] = {}
    t = start
    while t < end:
        nxt = min(end, floor_hour(t) + HOUR)
        out[hour_label(t)] = out.get(hour_label(t), 0.0) + (nxt - t) / HOUR * usd_per_hour
        t = nxt
    return out


# -- Modal billing source ----------------------------------------------------------

@dataclass
class BillingRow:
    object_id: str
    description: str
    interval_start: float
    cost: float


def modal_billing_fetch(start: float, end: float) -> list[BillingRow]:
    """Production fetch: ``modal.billing.workspace_billing_report`` at hourly resolution."""
    import modal

    rows = modal.billing.workspace_billing_report(
        start=datetime.fromtimestamp(floor_hour(start), UTC),
        end=datetime.fromtimestamp(end, UTC), resolution="h")
    return [BillingRow(str(r["object_id"]), str(r.get("description") or ""),
                       _ts(r["interval_start"]), float(r["cost"])) for r in rows]


class ModalBilling:
    """Caches the billing report; refreshed at most every ``refresh_s``.
    A failing fetch keeps the previous rows (the estimate covers the gap) and
    records the error."""

    def __init__(self, fetch: Callable[[float, float], list[BillingRow]] = modal_billing_fetch, *,
                 refresh_s: float = 300.0) -> None:
        self.fetch = fetch
        self.refresh_s = refresh_s
        self._rows: list[BillingRow] = []
        self._at: float | None = None
        self._start: float | None = None
        self.error: str | None = None

    def rows(self, start: float, now: float) -> list[BillingRow]:
        if self._at is None or now - self._at >= self.refresh_s or (
                self._start is not None and start < self._start):
            try:
                self._rows = list(self.fetch(start, now))
                self._start = start
                self.error = None
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
            self._at = now
        return self._rows


# -- per-resource cost ---------------------------------------------------------------

@dataclass
class ItemCost:
    key: str
    kind: str
    name: str | None
    start: float
    end: float
    open: bool
    billed_usd: float = 0.0
    estimated_usd: float = 0.0
    priced: bool = True
    rate_usd_per_hour: float | None = None
    hours: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def usd(self) -> float | None:
        if not self.priced and self.billed_usd == 0.0:
            return None
        return self.billed_usd + self.estimated_usd

    @property
    def billable_s(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class CostSummary:
    now: float
    items: list[ItemCost]
    billing_error: str | None = None

    @property
    def total_usd(self) -> float:
        return sum(i.usd or 0.0 for i in self.items)

    @property
    def unpriced(self) -> list[ItemCost]:
        return [i for i in self.items if i.usd is None and i.kind != "modal_call"]

    def to_json(self) -> dict[str, Any]:
        return {"now": self.now, "total_usd": round(self.total_usd, 4),
                "unpriced": [i.key for i in self.unpriced], "billing_error": self.billing_error,
                "items": [{**{k: v for k, v in i.__dict__.items()}, "usd": i.usd,
                           "billable_s": i.billable_s} for i in self.items]}


@dataclass
class Rates:
    """Where rates come from when the ledger entry carries none."""
    prices: Mapping[str, Any] | None = None  # yeto.dashboard.cost price table
    nc_usd_per_ucr: float | None = DEFAULT_NC_USD_PER_UCR
    nc_usd_per_gpu_hour: float | None = None
    nc_ucr_per_gpu_hour: float | None = DEFAULT_NC_UCR_PER_GPU_HOUR  # when the ledger carries no rate

    @classmethod
    def from_env(cls, prices_path: str | None = None, environ=None) -> "Rates":
        env = os.environ if environ is None else environ
        prices = None
        path = prices_path or env.get("YETO_DASHBOARD_PRICES")
        if path:
            from ..dashboard.cost import load_prices

            prices = load_prices(path)

        def f(name):
            v = env.get(name)
            return float(v) if v else None

        usd_per_ucr = f(NC_USD_PER_UCR_ENV)
        ucr = f(NC_UCR_PER_GPU_HOUR_ENV)
        return cls(prices, DEFAULT_NC_USD_PER_UCR if usd_per_ucr is None else usd_per_ucr,
                   f(NC_USD_PER_GPU_HOUR_ENV), DEFAULT_NC_UCR_PER_GPU_HOUR if ucr is None else ucr)

    def table_rate(self, cloud: str, attrs: Mapping[str, Any]) -> float | None:
        if self.prices is None or not attrs.get("gpu") or not attrs.get("gpus"):
            return None
        from ..dashboard.cost import price_per_gpu_h

        unit = price_per_gpu_h(dict(self.prices), cloud, attrs.get("gpu"), attrs.get("price_key"))
        return None if unit is None else unit * float(attrs["gpus"])


def _modal_app_cost(r: Resource, rows: Iterable[BillingRow], now: float, rates: Rates) -> ItemCost:
    end = r.released_at if r.released_at is not None else now
    item = ItemCost(r.key, r.kind, r.name, r.created_at, end, r.open)
    lo, hi = floor_hour(r.created_at), end
    mine = sorted((b for b in rows if b.object_id == r.id and lo <= b.interval_start < hi),
                  key=lambda b: b.interval_start)
    for b in mine:
        item.billed_usd += b.cost
        lbl = hour_label(b.interval_start)
        item.hours[lbl] = item.hours.get(lbl, 0.0) + b.cost
    billed_until = (mine[-1].interval_start + HOUR) if mine else r.created_at
    rate = r.attrs.get("usd_per_hour")
    rate = float(rate) if rate is not None else rates.table_rate("modal", r.attrs)
    if rate is None:
        rate = default_modal_rate(r.attrs.get("gpu"), r.attrs.get("gpus"))
        if rate is not None:
            item.note = "default Modal list price"
    if rate is None:
        full = [b for b in mine if b.interval_start >= r.created_at and b.interval_start + HOUR <= end]
        if full:
            rate = full[-1].cost
            item.note = "rate from the newest complete billed hour"
    item.rate_usd_per_hour = rate
    gap_start = max(r.created_at, billed_until)
    if gap_start < end:
        if rate is None:
            item.priced = False
            item.note = "no rate for the unbilled tail (set usd_per_hour or a price table)"
        else:
            for lbl, usd in split_by_hour(gap_start, end, rate).items():
                item.estimated_usd += usd
                item.hours[lbl] = item.hours.get(lbl, 0.0) + usd
    if r.attrs.get("shared"):
        item.note = (item.note + "; " if item.note else "") + \
            "app shared with other runs: billed hours overlapping this run are counted in full"
    return item


def _nc_rate(r: Resource, rates: Rates) -> float | None:
    gpus = float(r.attrs.get("gpus") or 8)
    if r.attrs.get("usd_per_gpu_hour") is not None:
        return gpus * float(r.attrs["usd_per_gpu_hour"])
    if r.attrs.get("usd_per_hour") is not None:
        return float(r.attrs["usd_per_hour"])
    if rates.nc_usd_per_gpu_hour is not None:
        return gpus * rates.nc_usd_per_gpu_hour
    ucr = r.attrs.get("rate_ucr_per_gpu_hour")
    if ucr is None:
        ucr = rates.nc_ucr_per_gpu_hour
    if ucr is not None and rates.nc_usd_per_ucr is not None:
        return gpus * float(ucr) * rates.nc_usd_per_ucr
    return None


def _span_cost(r: Resource, start: float, now: float, rate: float | None, note: str = "") -> ItemCost:
    end = r.released_at if r.released_at is not None else now
    item = ItemCost(r.key, r.kind, r.name, start, max(start, end), r.open, note=note,
                    rate_usd_per_hour=rate)
    if rate is None:
        item.priced = False
        item.note = note or "no rate (ledger attrs / price table / env)"
    else:
        item.hours = split_by_hour(start, max(start, end), rate)
        item.estimated_usd = sum(item.hours.values())
    return item


def resource_costs(resources: Iterable[Resource], now: float, *, rates: Rates | None = None,
                   billing: ModalBilling | None = None) -> CostSummary:
    rates = rates or Rates()
    resources = [r for r in resources if not r.intent_only or r.open]
    apps = [r for r in resources if r.kind == "modal_app" and r.id]
    rows: list[BillingRow] = []
    if apps and billing is not None:
        rows = billing.rows(min(r.created_at for r in apps), now)
    items: list[ItemCost] = []
    for r in resources:
        if r.kind == "modal_app":
            if r.id is None:
                items.append(_span_cost(r, r.created_at, now, None, "app id unknown (intent only)"))
            else:
                items.append(_modal_app_cost(r, rows, now, rates))
        elif r.kind == "nc_lease":
            start = _ts(r.attrs.get("granted_at")) or r.created_at
            note = ""
            until = _ts(r.attrs.get("lease_until"))
            end = r.released_at if r.released_at is not None else now
            if until is not None and end > until:
                note = f"{(end - until) / 60:.0f} min past lease_until"
            items.append(_span_cost(r, start, now, _nc_rate(r, rates), note))
        elif r.kind == "sky_cluster":
            rate = r.attrs.get("usd_per_hour")
            rate = float(rate) if rate is not None else rates.table_rate(r.attrs.get("cloud") or "", r.attrs)
            items.append(_span_cost(r, r.created_at, now, rate))
        else:  # modal_call: its cost is in its app; listed for liveness / utilization
            end = r.released_at if r.released_at is not None else now
            items.append(ItemCost(r.key, r.kind, r.name, r.created_at, end, r.open, billed_usd=0.0,
                                  note="cost counted in its Modal app"))
    return CostSummary(now, items, billing_error=billing.error if billing is not None else None)
