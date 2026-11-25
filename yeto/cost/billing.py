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

import logging
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from . import prices
from .ledger import Resource

log = logging.getLogger(__name__)
_SKEW_WARNED: set[str] = set()  # one warning per resource, not one per guard tick

HOUR = 3600.0
NC_USD_PER_UCR_ENV = "YETO_NC_USD_PER_UCR"
NC_USD_PER_GPU_HOUR_ENV = "YETO_NC_USD_PER_GPU_HOUR"
NC_UCR_PER_GPU_HOUR_ENV = "YETO_NC_UCR_PER_GPU_HOUR"
# National Compute defaults: 1 credit (1e6 ucr) ~ 1 USD, and the 8-GPU MI355X node at
# 3e6 ucr per GPU-hour (24e6 ucr/h, ~24 USD/h); the env above overrides either. Both
# come from the unified price table (yeto.cost.prices, row nc:MI355X).
DEFAULT_NC_USD_PER_UCR = prices.nc_usd_per_ucr()
DEFAULT_NC_UCR_PER_GPU_HOUR = prices.nc_ucr_per_gpu_h()
# Modal price per GPU-hour, used when neither the ledger nor a price table gives a
# rate. Keys are lower-case Modal gpu names. From the unified price table (its Modal
# rows: modal.com/pricing per-second list prices x 3600, each row with its date; base
# region-agnostic rate; Modal bills 1.15-1.75x when a region is pinned).
MODAL_PRICES_SOURCE = prices.sources("modal")
DEFAULT_MODAL_USD_PER_GPU_HOUR = prices.per_gpu_usd_h("modal")


def default_modal_host(gpus: Any) -> tuple[float, float]:
    """``(cores, memory GiB)`` a Modal island of ``gpus`` GPUs requests by default
    (:data:`yeto.modal_runner.MODAL_CPU_CORES_PER_GPU` / ``MODAL_MEMORY_GIB_PER_GPU`` each):
    what a ledger entry without its host shape is priced at."""
    from yeto.modal_runner import MODAL_CPU_CORES_PER_GPU, MODAL_MEMORY_GIB_PER_GPU  # noqa: PLC0415

    n = float(gpus or 0)
    return MODAL_CPU_CORES_PER_GPU * n, MODAL_MEMORY_GIB_PER_GPU * n


def default_modal_rate(gpu: Any, gpus: Any, cpu: Any = None, memory_gib: Any = None,
                       disk_gib: Any = None) -> float | None:
    """USD/h of Modal containers of ``gpus`` x ``gpu`` that request ``cpu`` cores and
    ``memory_gib`` GiB in all (an ``ephemeral disk`` of ``disk_gib`` bills as memory at
    20:1): the GPU list price plus the CPU and memory list prices (``host.modal``), which
    Modal bills next to the GPU. A missing host shape is the per-GPU default request."""
    unit = DEFAULT_MODAL_USD_PER_GPU_HOUR.get(str(gpu or "").lower().split(":")[0].rstrip("!"))
    if unit is None or not gpus:
        return None
    dcpu, dmem = default_modal_host(gpus)
    hostpart = prices.modal_host_usd_h(dcpu if cpu is None else float(cpu),
                                       dmem if memory_gib is None else float(memory_gib),
                                       None if disk_gib is None else float(disk_gib))
    return unit * float(gpus) + (hostpart or 0.0)


def modal_shapes_rate(shapes: Iterable[tuple[Any, ...]]) -> float | None:
    """USD/h of an app whose functions hold ``(gpu, gpus[, cpu, memory_gib[, disk_gib]])``
    each (:func:`default_modal_rate`); None when any GPU type is missing from
    :data:`DEFAULT_MODAL_USD_PER_GPU_HOUR` (or there is no GPU at all)."""
    rates = [default_modal_rate(*shape) for shape in shapes]
    if not rates or any(r is None for r in rates):
        return None
    return sum(rates)


def _attrs_rate(attrs: Mapping[str, Any]) -> float | None:
    """:func:`default_modal_rate` of a ledger entry's ``gpu`` / ``gpus`` / ``cpu`` / ``memory_gib`` / ``disk_gib``."""
    return default_modal_rate(attrs.get("gpu"), attrs.get("gpus"), attrs.get("cpu"), attrs.get("memory_gib"),
                              attrs.get("disk_gib"))


# The unbilled tail of a Modal app is estimated at the higher of its list rate and the
# rate of its dearest billed hour (that hour's cost over the time the app ran in it, when
# it ran >= this long in it): network egress (a trainer publishing its weights) and use
# above the request bill on top of the list rate, and no list price foresees them.
OBSERVED_RATE_MIN_OVERLAP_S = 900.0


def observed_hour_rate(r: Resource, mine: list[BillingRow], end: float, *,
                       without_egress: bool = False) -> tuple[float, str] | None:
    """``(USD/h, hour label)`` of app ``r``'s dearest billed hour among those it ran in for
    at least :data:`OBSERVED_RATE_MIN_OVERLAP_S` (its cost over that overlap); None: none.
    ``without_egress``: the hour's cost less its billed egress (the tail counts the app's
    egress from its tape instead; a row without the breakdown keeps its whole cost)."""
    best: tuple[float, str] | None = None
    for b in mine:
        overlap = min(end, b.interval_start + HOUR) - max(r.created_at, b.interval_start)
        if overlap >= OBSERVED_RATE_MIN_OVERLAP_S:
            cost = b.cost - (b.egress or 0.0) if without_egress else b.cost
            rate = cost / (overlap / HOUR)
            if best is None or rate > best[0]:
                best = (rate, hour_label(b.interval_start))
    return best


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


def ts_or_none(value: Any) -> float | None:
    """``_ts`` for ledger / API fields that may be corrupt: an unparsable value is None
    (callers fall back to the ledger's own times) instead of raising through every guard tick."""
    try:
        return _ts(value)
    except (TypeError, ValueError, OverflowError):
        return None


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
    egress: float | None = None  # the row's "Network Egress" part (None: the report gave no breakdown)


EGRESS_RESOURCE = "Network Egress"  # the billing report's cost_by_resource key of egress


def _row_field(row: Any, name: str) -> Any:
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


def modal_billing_fetch(start: float, end: float) -> list[BillingRow]:
    """Production fetch: the workspace billing report at hourly resolution
    (``modal.Workspace.billing.report``, which breaks each row down by resource; older
    clients: ``modal.billing.workspace_billing_report``, whole rows only)."""
    import modal

    window = dict(start=datetime.fromtimestamp(floor_hour(start), UTC), end=datetime.fromtimestamp(end, UTC),
                  resolution="h")
    workspace = getattr(modal, "Workspace", None)
    rows = None
    if workspace is not None and hasattr(workspace, "from_context"):
        try:
            rows = workspace.from_context().billing.report(**window)
        except Exception:  # noqa: BLE001 - the older report still prices every row
            rows = None
    if rows is None:
        rows = modal.billing.workspace_billing_report(**window)
    out = []
    for r in rows:
        by = _row_field(r, "cost_by_resource")
        egress = None if not isinstance(by, Mapping) else float(by.get(EGRESS_RESOURCE) or 0.0)
        out.append(BillingRow(str(_row_field(r, "object_id")), str(_row_field(r, "description") or ""),
                              _ts(_row_field(r, "interval_start")), float(_row_field(r, "cost")), egress))
    return out


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
        from ..dashboard.cost import host_unit_prices, price_per_gpu_h

        table = dict(self.prices)
        unit = price_per_gpu_h(table, cloud, attrs.get("gpu"), attrs.get("price_key"))
        if unit is None:
            return None
        rate = unit * float(attrs["gpus"])
        hp = host_unit_prices(table, cloud) if cloud == "modal" else None
        if hp is not None:  # Modal bills the requested CPU and memory next to the GPU
            dcpu, dmem = default_modal_host(attrs["gpus"])
            cpu = dcpu if attrs.get("cpu") is None else float(attrs["cpu"])
            mem = prices.modal_billed_memory_gib(dmem if attrs.get("memory_gib") is None else float(attrs["memory_gib"]),
                                                 attrs.get("disk_gib"))
            rate += hp["cpu_core_h"] * cpu + hp["mem_gib_h"] * mem
        return rate


def _app_rows(r: Resource, rows: Iterable[BillingRow], end: float) -> list[BillingRow]:
    lo = floor_hour(r.created_at)
    return sorted((b for b in rows if b.object_id == r.id and lo <= b.interval_start < end),
                  key=lambda b: b.interval_start)


def _billed_until(r: Resource, rows: Iterable[BillingRow], now: float) -> float:
    """Where the billing report of app ``r`` ends (its created time without a billed hour)."""
    end = r.released_at if r.released_at is not None else now
    mine = _app_rows(r, rows, end)
    return (mine[-1].interval_start + HOUR) if mine else r.created_at


def _pool_call_rate(r: Resource) -> float | None:
    """The ``usd_per_hour`` a sampler pool's Modal call carries (the pool's ``price_usd_h``)."""
    if r.kind != "modal_call" or not r.attrs.get("pool") or r.attrs.get("usd_per_hour") is None:
        return None
    return float(r.attrs["usd_per_hour"])


def _modal_app_cost(r: Resource, rows: Iterable[BillingRow], now: float, rates: Rates, *,
                    tail_by_calls: bool = False, egress: Sequence[Any] | None = None,
                    egress_taken: set[float] | None = None) -> ItemCost:
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
        rate = _attrs_rate(r.attrs)
        if rate is not None:
            item.note = "default Modal list price (GPU + CPU + memory)"
    if rate is None:
        full = [b for b in mine if b.interval_start >= r.created_at and b.interval_start + HOUR <= end]
        if full:
            rate = full[-1].cost
            item.note = "rate from the newest complete billed hour"
    egress_events = [str(e) for e in r.attrs.get("egress_events") or ()]
    if not tail_by_calls and billed_until < end:
        observed = observed_hour_rate(r, mine, end, without_egress=bool(egress_events) and egress is not None)
        if observed is not None and (rate is None or observed[0] > rate):
            listed = "" if rate is None else f" (above the list rate {rate:.2f} USD/h: egress or use over request)"
            rate = observed[0]
            item.note = (item.note + "; " if item.note else "") + \
                f"unbilled tail at the rate of billed hour {observed[1]}{listed}"
    item.rate_usd_per_hour = rate
    gap_start = max(r.created_at, billed_until)
    if tail_by_calls:  # a sampler pool's app: its priced calls estimate the unbilled tail
        item.note = (item.note + "; " if item.note else "") + "unbilled tail estimated per sampler call"
    elif gap_start < end:
        if rate is None:
            item.priced = False
            item.note = "no rate for the unbilled tail (set usd_per_hour or a price table)"
        else:
            for lbl, usd in split_by_hour(gap_start, end, rate).items():
                item.estimated_usd += usd
                item.hours[lbl] = item.hours.get(lbl, 0.0) + usd
    if egress_events and egress is not None and gap_start < end:
        _add_egress(item, r, egress, egress_events, gap_start, end, egress_taken)
    if r.attrs.get("shared"):
        item.note = (item.note + "; " if item.note else "") + \
            "app shared with other runs: billed hours overlapping this run are counted in full"
    return item


def _add_egress(item: ItemCost, r: Resource, samples: Sequence[Any], events: list[str], start: float,
                end: float, taken: set[float] | None) -> None:
    """The bytes app ``r``'s tape says it sent out of Modal in its unbilled tail
    ``[start, end)`` (:mod:`.egress`) at the price table's ``host.modal.egress_gib``. A
    sample counts in one app only (``taken``: the times already counted)."""
    from . import egress as eg  # noqa: PLC0415

    host = prices.host("modal")
    price = None if host is None else host.egress_gib
    fallback = r.attrs.get("egress_fallback_bytes")
    sized, unsized = eg.bytes_in(samples, events, start, end, None if fallback is None else float(fallback))
    if taken is not None:
        sized = {t: b for t, b in sized.items() if t not in taken}
        taken.update(sized)
    if price is None:
        if sized or unsized:
            item.note = (item.note + "; " if item.note else "") + "egress not priced (no host.modal.egress_gib)"
        return
    usd_total = 0.0
    for t, size in sized.items():
        usd = size / eg.GIB * price
        usd_total += usd
        lbl = hour_label(t)
        item.hours[lbl] = item.hours.get(lbl, 0.0) + usd
    item.estimated_usd += usd_total
    gib = sum(sized.values()) / eg.GIB
    note = f"egress {gib:.1f} GiB out of Modal after the newest billed hour (tape) at {price:g} USD/GiB"
    if unsized:
        note += f"; {unsized} record(s) without a size and no fallback: not counted"
    item.note = (item.note + "; " if item.note else "") + note


def _intent_app_rate(r: Resource, rates: Rates) -> float | None:
    rate = r.attrs.get("usd_per_hour")
    if rate is not None:
        return float(rate)
    rate = rates.table_rate("modal", r.attrs)
    return rate if rate is not None else _attrs_rate(r.attrs)


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
    if end < start and r.key not in _SKEW_WARNED:
        _SKEW_WARNED.add(r.key)
        # clamped to a 0 h span below (the ledger already floors a release at the newest
        # earlier stamp, so this is a clock skew it could not see): said, never silent
        log.warning("cost: %s ends %.0f s before its start (clock skew?): billed as 0 h", r.key, start - end)
    item = ItemCost(r.key, r.kind, r.name, start, max(start, end), r.open, note=note,
                    rate_usd_per_hour=rate)
    if rate is None:
        item.priced = False
        item.note = note or "no rate (ledger attrs / price table / env)"
    else:
        item.hours = split_by_hour(start, max(start, end), rate)
        item.estimated_usd = sum(item.hours.values())
    return item


def _provider_cost(kind: str):
    """The registered cost function of an island provider's ledger kind (None: not one)."""
    from yeto.launch.pool import provider as providers  # noqa: PLC0415

    entry = providers.ledger_kinds().get(kind)
    return None if entry is None or entry.resource_cost is None else providers.resolve(entry.resource_cost)


def resource_costs(resources: Iterable[Resource], now: float, *, rates: Rates | None = None,
                   billing: ModalBilling | None = None, egress: Sequence[Any] | None = None) -> CostSummary:
    """The cost of every resource. ``egress``: the run's egress samples
    (:class:`yeto.cost.egress.EgressMeter`), counted on the Modal apps whose ledger entry
    names them (``egress_events``) after their newest billed hour; None: not counted."""
    rates = rates or Rates()
    egress_taken: set[float] = set()
    resources = [r for r in resources if not r.intent_only or r.open]
    apps = [r for r in resources if r.kind == "modal_app" and r.id]
    rows: list[BillingRow] = []
    if apps and billing is not None:
        rows = billing.rows(min(r.created_at for r in apps), now)
    # A sampler pool's Modal app (role rollout) is recorded without a rate; each of its
    # calls carries the pool's price (usd_per_hour). Those calls estimate the time after
    # the app's newest billed hour, so the samplers count against the budget before the
    # billing report shows them.
    priced_runs = {str(r.attrs.get("run")) for r in resources if _pool_call_rate(r) is not None}
    rollout_apps = [r for r in apps if r.attrs.get("role") == "rollout" and str(r.attrs.get("run")) in priced_runs]
    rollout_keys = {r.key for r in rollout_apps}
    # a call's own app: the run's newest rollout app deployed before the call started
    app_billed = sorted((r.created_at, str(r.attrs.get("run")), _billed_until(r, rows, now)) for r in rollout_apps)

    def covered_until(c: Resource) -> float:
        run, until = str(c.attrs.get("run")), 0.0
        for created, app_run, billed in app_billed:
            if app_run == run and created <= c.created_at:
                until = billed
        return until

    # a respawn on the same ticket cancels the call before it: that call ends where the next starts
    next_start: dict[str, float] = {}
    by_ticket: dict[tuple[str, str], list[Resource]] = {}
    for r in resources:
        if _pool_call_rate(r) is not None and r.attrs.get("ticket"):
            by_ticket.setdefault((str(r.attrs.get("run")), str(r.attrs["ticket"])), []).append(r)
    for calls in by_ticket.values():
        calls.sort(key=lambda c: c.created_at)
        for a, b in zip(calls, calls[1:]):
            next_start[a.key] = b.created_at
    items: list[ItemCost] = []
    for r in resources:
        if r.kind == "modal_app":
            if r.id is None:
                # deploying (image build): no app id yet, so no billing rows; the app's
                # GPU shape (recorded with the intent) prices the span, conservatively
                rate = _intent_app_rate(r, rates)
                items.append(_span_cost(
                    r, r.created_at, now, rate,
                    "app id unknown (deploying): priced at its GPU rate" if rate is not None
                    else "app id unknown (intent only) and no GPU shape / rate recorded"))
            else:
                items.append(_modal_app_cost(r, rows, now, rates, tail_by_calls=r.key in rollout_keys,
                                             egress=egress, egress_taken=egress_taken))
        elif _pool_call_rate(r) is not None:
            start = max(r.created_at, covered_until(r))
            if r.key in next_start and (r.released_at is None or r.released_at > next_start[r.key]):
                r = replace(r, released_at=next_start[r.key])
            note = "sampler pool call: estimate after its app's newest billed hour"
            if r.attrs.get("pending"):  # recorded at its spawn: bills from its scheduled_at only
                scheduled = r.attrs.get("scheduled_at")
                if scheduled is None:  # it waited for a GPU worker and never got one: nothing ran
                    start = r.released_at if r.released_at is not None else now
                    note = "sampler pool call: pending (not scheduled on a worker), not billed"
                else:
                    start = max(start, float(scheduled))
            items.append(_span_cost(r, start, now, _pool_call_rate(r), note))
        elif _provider_cost(r.kind) is not None:  # an island provider's machine (nc_lease)
            items.append(_provider_cost(r.kind)(r, now, rates))
        elif r.kind == "ssh_trainer":  # the operator's machines: the inventory's declared price
            rate = r.attrs.get("usd_per_hour")
            items.append(_span_cost(r, r.created_at, now, None if rate is None else float(rate),
                                    "" if rate is not None else "SSH nodes without price_per_card_hour"))
        elif r.kind == "sky_cluster":
            rate = r.attrs.get("usd_per_hour")
            note = ""
            if rate is None and r.attrs.get("price_usd_h") is not None:
                # a sampler pool's cluster records the offer it was launched with: its price
                # is per node, a group cluster (num_nodes) bills every node
                rate = float(r.attrs["price_usd_h"]) * max(1, int(r.attrs.get("num_nodes") or 1))
                note = "the pool offer's price"
            elif rate is not None and r.attrs.get("price_note"):
                note = str(r.attrs["price_note"])
            rate = float(rate) if rate is not None else rates.table_rate(r.attrs.get("cloud") or "", r.attrs)
            if rate is None and r.attrs.get("cloud") and r.attrs.get("gpu"):
                rate = prices.node_usd_h(str(r.attrs["cloud"]), r.attrs["gpu"], r.attrs.get("gpus"),
                                         spot=r.attrs.get("use_spot"), region=r.attrs.get("region"))
                note = "" if rate is None else "unified price table (list price)"
            if rate is None and r.attrs.get("role") == "syncer":
                # a syncer VM recorded without a price (a ledger written before it carried one):
                # its cloud's syncer row, else the dearest one (never 0)
                rate, note, _conservative = prices.syncer_vm_usd_h(str(r.attrs.get("cloud") or ""))
            items.append(_span_cost(r, r.created_at, now, rate, note))
        else:  # modal_call: its cost is in its app; listed for liveness / utilization
            end = r.released_at if r.released_at is not None else now
            items.append(ItemCost(r.key, r.kind, r.name, r.created_at, end, r.open, billed_usd=0.0,
                                  note="cost counted in its Modal app"))
    return CostSummary(now, items, billing_error=billing.error if billing is not None else None)
