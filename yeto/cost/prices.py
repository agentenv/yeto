"""The one machine price table of yeto.

Every price yeto uses for a machine it does not get a bill for yet comes from
``prices.json`` next to this module: one row per (provider, GPU type, GPU count
of the node, spot / on-demand, region), each a whole-node price in USD per
hour with its ``source`` and its ``as_of`` date. Billing (``yeto.cost.billing``),
the pool and run budgets, the dashboard's default table, the sky offers of a
sampler pool, the shape planner and the dry-runs read it through the functions
below; a run config (``usd_per_h``, ``price_usd_h``) or a price file
(``--prices``) still overrides a row.

A node of ``n`` GPUs matched by a row of ``gpus`` GPUs costs ``n / gpus x usd_h``.

Every row has a ``grade``: ``billing`` (a price list the bill follows) or
``illustrative`` (an example value). Everything that gates or counts money (billing,
the pool and run budgets, the breakers, offer pricing, the broker) reads billing rows
only: to them an illustrative row is no price, so an offer priced by nothing else stays
unpriced and a capped run refuses it. Only the dashboard's estimate shows illustrative
rows, marked as estimates.
GPU names compare case-insensitively without a ``!`` suffix or a ``:count``
(``"h100!"``, ``"H100:8"`` and ``"H100"`` are one GPU).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

TABLE_PATH = Path(__file__).with_name("prices.json")
TABLE_NAME = "yeto/cost/prices.json"
# a dry-run says so when a price it uses was read longer ago than this
DEFAULT_MAX_AGE_DAYS = 90
ANY = "*"
BILLING, ILLUSTRATIVE = "billing", "illustrative"
GRADES = (BILLING, ILLUSTRATIVE)


class PriceTableError(ValueError):
    pass


@dataclass(frozen=True)
class Price:
    """One row: a node of ``gpus`` x ``gpu`` at ``provider`` for ``usd_h`` USD per hour."""

    provider: str
    gpu: str
    gpus: int
    usd_h: float
    source: str
    as_of: str
    grade: str = BILLING
    spot: bool | None = None  # None: either
    region: str | None = None  # None: any region
    note: str | None = None
    native: Mapping[str, Any] | None = field(default=None, compare=False)

    @property
    def usd_per_gpu_h(self) -> float:
        return self.usd_h / self.gpus

    def node_usd_h(self, gpus: float) -> float:
        """A node of ``gpus`` GPUs of this row's type."""
        return self.usd_per_gpu_h * float(gpus)

    @property
    def key(self) -> str:
        where = f"@{self.region}" if self.region else ""
        kind = "" if self.spot is None else ("/spot" if self.spot else "/od")
        return f"{self.provider}:{self.gpu}x{self.gpus}{where}{kind}"

    def age_days(self, today: dt.date | None = None) -> int:
        return ((today or dt.date.today()) - dt.date.fromisoformat(self.as_of)).days

    def cited(self) -> str:
        return f"{self.key} {self.usd_h:g} USD/h (source: {self.source}, as of {self.as_of})"


@dataclass(frozen=True)
class HostPrice:
    """CPU and memory a provider bills next to the GPUs (Modal)."""

    provider: str
    cpu_core_h: float
    mem_gib_h: float
    source: str
    as_of: str
    egress_gib: float | None = None  # USD per GiB sent out of the provider (None: not billed / unknown)


@dataclass(frozen=True)
class ServicePrice:
    """A machine priced by the service it runs (a fleet's syncer) on ``provider``, not by its GPUs."""

    name: str
    provider: str
    usd_h: float
    source: str
    as_of: str
    grade: str = BILLING
    note: str | None = None

    def cited(self) -> str:
        return f"{self.name}@{self.provider} {self.usd_h:g} USD/h (source: {self.source}, as of {self.as_of})"


@dataclass(frozen=True)
class Table:
    rows: tuple[Price, ...]
    host: Mapping[str, HostPrice]
    services: Mapping[tuple[str, str], ServicePrice] = field(default_factory=dict)


def gpu_name(gpu: Any) -> str:
    """The comparable form of a GPU name (``"H100!"``, ``"h100:8"`` -> ``"h100"``)."""
    return str(gpu or "").strip().lower().split(":")[0].rstrip("!")


def _row(doc: Mapping[str, Any], i: int) -> Price:
    where = f"{TABLE_NAME} rows[{i}]"
    for key in ("provider", "gpu", "gpus", "usd_h", "source", "as_of", "grade"):
        if doc.get(key) in (None, ""):
            raise PriceTableError(f"{where}: {key} is required")
    if doc["grade"] not in GRADES:
        raise PriceTableError(f"{where}: grade must be one of {list(GRADES)}, not {doc['grade']!r}")
    try:
        dt.date.fromisoformat(str(doc["as_of"]))
    except ValueError:
        raise PriceTableError(f"{where}: as_of {doc['as_of']!r} is not an ISO date") from None
    gpus, usd = doc["gpus"], doc["usd_h"]
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 1:
        raise PriceTableError(f"{where}: gpus must be a whole number >= 1")
    if isinstance(usd, bool) or not isinstance(usd, (int, float)) or not usd >= 0:
        raise PriceTableError(f"{where}: usd_h must be a number >= 0")
    unknown = set(doc) - {"provider", "gpu", "gpus", "usd_h", "source", "as_of", "grade", "spot", "region", "note",
                          "native"}
    if unknown:
        raise PriceTableError(f"{where}: unknown keys {sorted(unknown)}")
    return Price(provider=str(doc["provider"]).lower(), gpu=str(doc["gpu"]), gpus=int(gpus), usd_h=float(usd),
                 source=str(doc["source"]), as_of=str(doc["as_of"]), grade=str(doc["grade"]), spot=doc.get("spot"),
                 region=doc.get("region"), note=doc.get("note"), native=doc.get("native"))


def parse(doc: Mapping[str, Any]) -> Table:
    rows = tuple(_row(r, i) for i, r in enumerate(doc.get("rows") or ()))
    seen: set[tuple[Any, ...]] = set()
    for r in rows:
        ident = (r.provider, gpu_name(r.gpu), r.gpus, r.spot, r.region)
        if ident in seen:
            raise PriceTableError(f"{TABLE_NAME}: two rows for {r.key}")
        seen.add(ident)
    host = {}
    for name, h in (doc.get("host") or {}).items():
        for key in ("cpu_core_h", "mem_gib_h", "source", "as_of"):
            if h.get(key) in (None, ""):
                raise PriceTableError(f"{TABLE_NAME} host.{name}: {key} is required")
        egress = h.get("egress_gib")
        if egress is not None and (isinstance(egress, bool) or not isinstance(egress, (int, float)) or egress < 0):
            raise PriceTableError(f"{TABLE_NAME} host.{name}: egress_gib must be a number >= 0")
        host[name.lower()] = HostPrice(name.lower(), float(h["cpu_core_h"]), float(h["mem_gib_h"]),
                                       str(h["source"]), str(h["as_of"]),
                                       None if egress is None else float(egress))
    services: dict[tuple[str, str], ServicePrice] = {}
    for name, entries in (doc.get("services") or {}).items():
        if name.startswith("_"):
            continue
        if not isinstance(entries, list):
            raise PriceTableError(f"{TABLE_NAME} services.{name}: a list of rows, one per provider")
        for i, sv in enumerate(entries):
            where = f"{TABLE_NAME} services.{name}[{i}]"
            for key in ("provider", "usd_h", "source", "as_of", "grade"):
                if sv.get(key) in (None, ""):
                    raise PriceTableError(f"{where}: {key} is required")
            if sv["grade"] not in GRADES:
                raise PriceTableError(f"{where}: grade must be one of {list(GRADES)}, not {sv['grade']!r}")
            usd = sv["usd_h"]
            if isinstance(usd, bool) or not isinstance(usd, (int, float)) or not usd >= 0:
                raise PriceTableError(f"{where}: usd_h must be a number >= 0")
            try:
                dt.date.fromisoformat(str(sv["as_of"]))
            except ValueError:
                raise PriceTableError(f"{where}: as_of {sv['as_of']!r} is not an ISO date") from None
            unknown = set(sv) - {"provider", "usd_h", "source", "as_of", "grade", "note"}
            if unknown:
                raise PriceTableError(f"{where}: unknown keys {sorted(unknown)}")
            key = (name.lower(), str(sv["provider"]).lower())
            if key in services:
                raise PriceTableError(f"{TABLE_NAME} services.{name}: two rows for {key[1]}")
            services[key] = ServicePrice(key[0], key[1], float(usd), str(sv["source"]), str(sv["as_of"]),
                                         str(sv["grade"]), sv.get("note"))
    return Table(rows, host, services)


@lru_cache(maxsize=4)
def table(path: str | None = None) -> Table:
    return parse(json.loads(Path(path or TABLE_PATH).read_text(encoding="utf-8")))


def rows(provider: str | None = None, *, path: str | None = None) -> list[Price]:
    out = list(table(path).rows)
    return out if provider is None else [r for r in out if r.provider == provider.lower()]


def lookup(provider: str, gpu: Any, *, spot: bool | None = None, region: str | None = None,
           grade: str | None = BILLING, path: str | None = None) -> Price | None:
    """The row that prices ``gpu`` at ``provider`` (``spot`` / ``region`` when known): the
    provider's own rows before the ``*`` rows; among them an exact region before a
    region-less row, and a row of the asked spot-ness before one that matches either.
    A row of a stated region or spot-ness never matches a different one. Only rows of
    ``grade`` (default: billing; None: any grade). None: no row."""
    want = gpu_name(gpu)
    best: tuple[tuple[int, int, int], Price] | None = None
    for r in table(path).rows:
        if gpu_name(r.gpu) != want or r.provider not in (provider.lower(), ANY):
            continue
        if grade is not None and r.grade != grade:
            continue
        if r.region is not None and r.region != region:
            continue
        if r.spot is not None and spot is not None and r.spot != spot:
            continue
        if r.spot is not None and spot is None and r.spot:  # spot unknown: never the spot price
            continue
        rank = (r.provider == ANY, r.region is None, r.spot is None)
        if best is None or rank < best[0]:
            best = (rank, r)
    return None if best is None else best[1]


def node_usd_h(provider: str, gpu: Any, gpus: Any, *, spot: bool | None = None, region: str | None = None,
               path: str | None = None) -> float | None:
    """USD per hour of a node of ``gpus`` x ``gpu`` at ``provider`` from a billing row (None: none)."""
    if not gpus:
        return None
    row = lookup(provider, gpu, spot=spot, region=region, path=path)
    return None if row is None else row.node_usd_h(float(gpus))


def catalog_node_usd_h(cloud: str, gpu: Any, gpus_per_node: Any, *, region: str | None = None,
                       instance_type: str | None = None) -> tuple[float | None, str]:
    """USD per hour of one node from the local sky catalog, at its ON-DEMAND list price (a spot
    node bills no more than that): ``instance_type``'s when named, else the dearest instance
    the catalog has for ``gpus_per_node`` x ``gpu`` (sky picks among them by the task's CPU and
    memory, which a price recorded before the launch does not know: never under). For a
    machine without a billing row in the price table. ``(None, why)`` when the catalog is not
    installed or has nothing for it; never raises."""
    try:
        from sky import catalog  # noqa: PLC0415 - optional; read only where sky launches
        from sky.catalog import common  # noqa: PLC0415

        local = os.path.exists(common.get_catalog_path(f"{str(cloud).lower()}/vms.csv"))
    except Exception as error:  # noqa: BLE001
        return None, f"no sky catalog ({type(error).__name__})"
    if not local:  # never fetched here (sky check / sky launch fetch it): not downloaded for a price
        return None, f"no local sky catalog for {cloud}"
    where = f" in {region}" if region else ""
    try:
        if instance_type:
            usd = catalog.get_hourly_cost(instance_type, use_spot=False, region=region, zone=None, clouds=cloud)
            return float(usd), f"sky catalog on-demand price of {instance_type}{where}"
        names, _fuzzy = catalog.get_instance_type_for_accelerator(str(gpu), int(gpus_per_node), use_spot=False,
                                                                 region=region, zone=None, clouds=cloud)
        costs = [(float(catalog.get_hourly_cost(n, use_spot=False, region=region, zone=None, clouds=cloud)), n)
                 for n in names or ()]
    except Exception as error:  # noqa: BLE001 - an unknown type or region: unpriced, said
        return None, f"sky catalog has no on-demand price for {instance_type or f'{gpus_per_node}x{gpu}'}{where} " \
                     f"({type(error).__name__})"
    if not costs:
        return None, f"sky catalog has no {gpus_per_node}x{gpu} instance on {cloud}{where}"
    usd, name = max(costs)
    return usd, f"sky catalog on-demand price of the dearest {gpus_per_node}x{gpu} instance ({name}){where}"


def host(provider: str, *, path: str | None = None) -> HostPrice | None:
    return table(path).host.get(provider.lower())


# Modal bills an ephemeral disk request as memory at 20:1 (500 GiB of disk raises the
# memory request to 25 GiB unless it is higher already: modal.com/docs/guide/resources).
MODAL_DISK_GIB_PER_BILLED_MEMORY_GIB = 20


def modal_billed_memory_gib(memory_gib: float, disk_gib: float | None = None) -> float:
    """The memory GiB Modal bills a container for: its request, raised to its disk / 20."""
    mem = float(memory_gib)
    return mem if not disk_gib else max(mem, float(disk_gib) / MODAL_DISK_GIB_PER_BILLED_MEMORY_GIB)


def modal_host_usd_h(cpu: float, memory_gib: float, disk_gib: float | None = None, *,
                     path: str | None = None) -> float | None:
    """USD per hour of the CPU cores and memory a Modal container requests (``host.modal``
    list prices; Modal bills max(request, usage), so the request is the floor). None: the
    table has no ``host.modal`` row."""
    h = host("modal", path=path)
    if h is None:
        return None
    return float(cpu) * h.cpu_core_h + modal_billed_memory_gib(memory_gib, disk_gib) * h.mem_gib_h


def modal_container_usd_h(gpu: Any, gpus: Any, *, cpu: float, memory_gib: float, disk_gib: float | None = None,
                          path: str | None = None) -> float | None:
    """USD per hour of Modal containers holding ``gpus`` x ``gpu`` with ``cpu`` cores and
    ``memory_gib`` GiB in all: the GPUs' billing row plus the requested CPU and memory at the
    ``host.modal`` list prices (Modal bills the three apart). None: the GPU has no billing
    row, or the table has no ``host.modal`` row (an estimate without the host part would be
    low: a Modal L40S trainer of 8 cores / 64 GiB bills ~1.46x its GPU line)."""
    node = node_usd_h("modal", gpu, gpus, path=path)
    hostpart = modal_host_usd_h(cpu, memory_gib, disk_gib, path=path)
    return None if node is None or hostpart is None else node + hostpart


SYNCER = "syncer"


def service(name: str, provider: str, *, grade: str | None = BILLING, path: str | None = None) -> ServicePrice | None:
    """The ``services`` row of ``name`` on ``provider``, of ``grade`` (default: billing; None: any). None: none."""
    row = table(path).services.get((name.lower(), str(provider or "").lower()))
    return None if row is None or (grade is not None and row.grade != grade) else row


def syncer_usd_h(provider: str, *, path: str | None = None) -> float | None:
    """USD per hour of a fleet's syncer machine on ``provider`` from a billing ``services.syncer`` row (None: none)."""
    row = service(SYNCER, provider, path=path)
    return None if row is None else row.usd_h


def syncer_vm_usd_h(provider: str, *, path: str | None = None) -> tuple[float | None, str, bool]:
    """``(USD per hour, where it comes from, conservative)`` of a syncer VM of its own on
    ``provider`` (a per-island fleet's syncer launch, a fleet launch's syncer cluster): the
    billing ``services.syncer`` row of that provider; without one, the dearest billing syncer
    row of any provider (``conservative`` True: never counted as 0, a spend may be over, not
    under). None only when the table has no syncer row at all (unpriced)."""
    row = service(SYNCER, provider, path=path)
    if row is not None:
        return row.usd_h, f"price table {row.cited()}", False
    rows_ = [r for (name, _p), r in table(path).services.items() if name == SYNCER and r.grade == BILLING]
    if not rows_:
        return None, f"no price table syncer row (for {provider or 'an unknown provider'} or any other)", True
    dearest = max(rows_, key=lambda r: r.usd_h)
    return dearest.usd_h, (f"no price table syncer row for {provider or 'an unknown provider'}: counted at the "
                           f"dearest syncer row, {dearest.cited()}"), True


# -- views the older tables had (kept in their shape for their readers) -------------

def per_gpu_usd_h(provider: str, *, path: str | None = None) -> dict[str, float]:
    """``{gpu name lower-case: USD per GPU-hour}`` of a provider's region-less, spot-less rows
    (Modal: the per-GPU list prices billing falls back to)."""
    return {gpu_name(r.gpu): r.usd_per_gpu_h for r in rows(provider, path=path)
            if r.region is None and r.spot is None and r.grade == BILLING}


def sources(provider: str, *, path: str | None = None) -> str:
    """Where a provider's prices come from, for a message: ``source (as of date)`` per distinct pair."""
    pairs = sorted({(r.source, r.as_of) for r in rows(provider, path=path) if r.grade == BILLING})
    return "; ".join(f"{s} (checked {d})" for s, d in pairs)


def regional_8gpu(provider: str, *, path: str | None = None) -> dict[tuple[str, str], tuple[float, float]]:
    """``{(GPU, region): (on-demand, spot)}`` of a provider's 8-GPU regional rows (the sky
    spot watchdog's Nebius candidate table). Only regions with both prices."""
    od: dict[tuple[str, str], float] = {}
    sp: dict[tuple[str, str], float] = {}
    for r in rows(provider, path=path):
        if r.gpus != 8 or r.region is None or r.spot is None or r.grade != BILLING:
            continue
        (sp if r.spot else od)[(r.gpu.upper(), r.region)] = r.usd_h
    return {k: (od[k], sp[k]) for k in od if k in sp}


def dashboard_table(*, path: str | None = None) -> dict[str, Any]:
    """The dashboard's price table form: ``{"prices": {"provider:GPU": USD per GPU-hour},
    "host_prices": {provider: {cpu_core_h, mem_gib_h}}, "estimate_keys": [...]}`` from the
    region-less, spot-less rows of either grade; ``estimate_keys`` are the illustrative ones
    (the dashboard marks a price from them as an estimate)."""
    t = table(path)
    picked = [r for r in t.rows if r.region is None and r.spot is None]
    out = {f"{r.provider}:{r.gpu}": r.usd_per_gpu_h for r in picked}
    return {"_note": f"built from the unified price table {TABLE_NAME} (each row carries its source and date)",
            "prices": out,
            "estimate_keys": sorted(f"{r.provider}:{r.gpu}" for r in picked if r.grade == ILLUSTRATIVE),
            "host_prices": {name: {"cpu_core_h": h.cpu_core_h, "mem_gib_h": h.mem_gib_h} for name, h in t.host.items()}}


# -- National Compute ---------------------------------------------------------------

NC_PROVIDER, NC_GPU = "nc", "MI355X"


def _nc_row(path: str | None = None) -> Price:
    row = lookup(NC_PROVIDER, NC_GPU, path=path)
    if row is None or not row.native:
        raise PriceTableError(f"{TABLE_NAME}: no {NC_PROVIDER}:{NC_GPU} row with its native ucr rate")
    return row


def nc_ucr_per_gpu_h(*, path: str | None = None) -> int:
    """The National Compute MI355X rate in ucr per GPU-hour."""
    return int(_nc_row(path).native["per_gpu_h"])


def nc_usd_per_ucr(*, path: str | None = None) -> float:
    return float(_nc_row(path).native["usd_per_unit"])


def nc_node_gpus(*, path: str | None = None) -> int:
    return _nc_row(path).gpus


def nc_node_usd_h(*, path: str | None = None) -> float:
    """One 8-GPU National Compute MI355X node hour in USD."""
    return _nc_row(path).usd_h


# -- sky offers ---------------------------------------------------------------------

def offer_row(resources: Mapping[str, Any], *, path: str | None = None) -> Price | None:
    """The row that prices a sky offer's ``resources`` (``infra`` ``"<cloud>/<region>"`` or
    ``cloud`` + ``region``, ``accelerators`` ``"GPU:count"``, ``use_spot``); None: not in the table.
    A spot offer the table has no spot price for is priced at its on-demand row (counted high,
    never low: :func:`spot_at_on_demand` tells the dry-run)."""
    infra = str(resources.get("infra") or "")
    cloud, _, region = infra.partition("/")
    cloud = (cloud or str(resources.get("cloud") or "")).lower()
    region = region.split("/")[0] or (str(resources["region"]) if resources.get("region") else "")
    acc = resources.get("accelerators")
    if not cloud or not acc or not isinstance(acc, str):
        return None
    spot = bool(resources.get("use_spot"))
    row = lookup(cloud, acc, spot=spot, region=region or None, path=path)
    if row is None and spot:
        row = lookup(cloud, acc, spot=False, region=region or None, path=path)
    return row


def spot_at_on_demand(resources: Mapping[str, Any], *, path: str | None = None) -> bool:
    """A spot offer whose table price is its on-demand row (no spot row for it)."""
    row = offer_row(resources, path=path) if resources.get("use_spot") else None
    return row is not None and row.spot is False


def offer_usd_h(resources: Mapping[str, Any], *, path: str | None = None) -> float | None:
    """USD per hour of the node a sky offer orders (None: its cloud / GPU / region is not in the table)."""
    row = offer_row(resources, path=path)
    if row is None:
        return None
    acc = str(resources["accelerators"])
    count = acc.partition(":")[2]
    try:
        gpus = float(count) if count else 1.0
    except ValueError:
        return None
    return row.node_usd_h(gpus)


# -- age ----------------------------------------------------------------------------

def stale(used: Iterable[Price], *, max_age_days: int = DEFAULT_MAX_AGE_DAYS,
          today: dt.date | None = None) -> list[str]:
    """One line per price in ``used`` read more than ``max_age_days`` ago (its source and date)."""
    out = []
    for r in sorted(set(used), key=lambda r: r.key):
        age = r.age_days(today)
        if age > max_age_days:
            out.append(f"price {r.key} {r.usd_h:g} USD/h is {age} days old (source: {r.source}, as of {r.as_of}); "
                       f"check it before trusting the spend estimate")
    return out
