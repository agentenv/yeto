"""Island providers: where a pool's machines come from, and nothing else.

A provider orders, reads, addresses and releases machines; it knows nothing of
samplers, shards or training. The pool (:class:`~.island_pool.IslandPool`) is
policy over :class:`~.island_pool.PoolOps`, which :class:`ProviderPoolOps`
composes from three parts:

* :class:`IslandProvider` - the machines (this module);
* :class:`NodeRoute` - what runs on a machine: setup, sampler start / restart,
  probe, stop (the National Compute SSH + docker route is
  ``yeto.launch.nc.cli.NcSshOps``; a Daytona sandbox runs the sampler as a process);
* :class:`Fencer` - the work contract's fence of a shard (``yeto.rl.engine.
  incarnation_lease`` through ``yeto.launch.nc.elastic.fence_shard``).

The provider contract (every method may raise; the pool treats a failed status
call as a control-plane blip, never as an end):

* ``order(slot) -> handle``: place one order for this run (record it before the
  create call so a crash leaves a handle cleanup can find);
* ``status(slot, handle) -> LeaseStatus``;
* ``release(slot, handle)``: end only a machine it created for this run, by id,
  and return once it reads ended (raise otherwise; the pool retries);
* ``node(slot, handle) -> NodeLease``: address + login of an active machine;
* ``balance() -> int | None`` (None: not known), ``account_free(slot) -> bool``
  (the slot holds no machine of anybody's: one machine per slot at a time);
* ``run_tickets() -> [(slot, handle)]``: open machines of this run it knows of
  (ledger rows, labels): the pool adopts them; cleanup releases them.

Optional: ``kind`` (registry name), ``caps`` (:class:`ProviderCaps`),
``ledger_kind``, ``attach(lease, *, ssh_run)`` (its own reach + host setup),
``sampler_route``, ``sweep()`` (delete every machine of this run: orphan cleanup),
``owned()`` (same as ``run_tickets``), ``hourly_cost(ticket=None)`` (one machine
hour in ``caps.currency``; the pool charges a member's budget with it when the
status carries no rate), ``released_by_us(ticket)``.

The registry maps a pool config's ``provider.kind`` to a factory and, for
providers whose machines are recorded in the run's resource ledger
(``yeto.cost.ledger``), the ledger kind with its cleanup release and its cost
(``yeto cleanup`` and ``yeto cost`` dispatch on it). ``provider.factory``
(``"package.module:function"``, called ``function(cfg, run_id)``) brings any
other provider without registering it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .status import LeaseStatus

# -- the contract ---------------------------------------------------------------


@dataclass(frozen=True)
class ProviderCaps:
    """What the pool may assume of a provider's machines.

    ``lease_bounded``: a machine has a lease end (``lease_until``) and must be
    replaced before it (National Compute: 4 h, not renewable); ``renewable``: the
    lease can be extended in place; ``notice_s``: warning before a preemption (0:
    none, e.g. Daytona spot); ``cold_start_s``: order to a ready sampler, a hint for
    ``replace_lead_s``; ``currency``: the unit of ``balance`` / ``hourly_cost``;
    ``balance``: ``balance()`` reads a real (shared) balance."""

    lease_bounded: bool = False
    renewable: bool = False
    notice_s: float = 0.0
    cold_start_s: float = 0.0
    currency: str = "usd"
    balance: bool = False


class IslandProvider(Protocol):
    def order(self, account: Any) -> str: ...
    def status(self, account: Any, ticket: str) -> LeaseStatus: ...
    def release(self, account: Any, ticket: str) -> Any: ...
    def node(self, account: Any, ticket: str) -> Any: ...
    def balance(self) -> int | None: ...
    def account_free(self, account: Any) -> bool: ...
    def run_tickets(self) -> list[tuple[Any, str]]: ...


class NodeRoute(Protocol):
    """What runs on a machine (``member`` is the pool's
    :class:`~.island_pool.Member`: ticket, shard, data slot, ``replaces_shard``).
    ``start`` sets the machine up and starts the sampler (idempotent: a restarted
    pool runs it again); ``probe`` is one look at the node
    (:class:`~.status.Probe`); ``stop`` stops the sampler, best effort (a lost node
    does not answer), before the provider releases the machine."""

    def start(self, member: Any, node: Any, shards: list[str]) -> None: ...
    def restart(self, member: Any, node: Any, shards: list[str]) -> None: ...
    def probe(self, member: Any) -> Any: ...
    def stop(self, member: Any) -> None: ...


class Fencer(Protocol):
    """The pool's side of the work contract: ``fence_shard`` (the node is taken for
    gone: its sampler stops at its next lease read, its claims go back at once) and
    ``fence_incarnation`` (lease only: the sampler is restarted in place).

    Optional, for a "drain" scale down and a ``hold_tasks`` pause:
    ``request_drain(shard, seq) -> seq`` (the shard's sampler claims no new task and
    answers ``seq`` once it holds none), ``drain_answered(shard, seq) -> bool`` and
    ``cancel_drain(shard)`` (claims again). ``yeto.rl.engine.store_dispense`` is the
    R2 implementation (``request_drain`` / ``drain_answered``)."""

    def fence_shard(self, shard: str, by: str) -> Any: ...
    def fence_incarnation(self, shard: str, by: str) -> Any: ...


def caps_of(provider: Any) -> ProviderCaps:
    return getattr(provider, "caps", None) or ProviderCaps()


# -- several providers in one pool ----------------------------------------------

SLOT_SEP = ":"


class MultiProvider:
    """Several providers behind one :class:`IslandProvider`: a slot is
    ``"<name>:<slot>"`` (``slots()`` lists them in the order given, so the pool
    fills the first provider's slots first). ``balance()`` reads the provider
    named ``balance_of`` (None: no shared balance gates the pool)."""

    kind = "multi"

    def __init__(self, providers: Mapping[str, Any], slots: Mapping[str, list[Any]], *,
                 balance_of: str | None = None):
        unknown = set(slots) - set(providers)
        if unknown:
            raise ValueError(f"slots for unknown providers {sorted(unknown)}")
        for name in providers:
            if SLOT_SEP in name:
                raise ValueError(f"provider name {name!r} must not contain {SLOT_SEP!r}")
        self.providers = dict(providers)
        self._slots = {name: list(v) for name, v in slots.items()}
        self.balance_of = balance_of

    def slots(self) -> list[str]:
        return [f"{name}{SLOT_SEP}{s}" for name, values in self._slots.items() for s in values]

    def split(self, account: Any) -> tuple[str, Any, Any]:
        name, sep, raw = str(account).partition(SLOT_SEP)
        if not sep or name not in self.providers:
            raise KeyError(f"slot {account!r} is not '<provider>{SLOT_SEP}<slot>' of {sorted(self.providers)}")
        slot: Any = raw
        for value in self._slots.get(name, ()):
            if str(value) == raw:
                slot = value
        return name, self.providers[name], slot

    def provider_of(self, account: Any) -> Any:
        return self.split(account)[1]

    def order(self, account: Any) -> str:
        _, p, slot = self.split(account)
        return p.order(slot)

    def status(self, account: Any, ticket: str) -> LeaseStatus:
        _, p, slot = self.split(account)
        return p.status(slot, ticket)

    def release(self, account: Any, ticket: str) -> Any:
        _, p, slot = self.split(account)
        return p.release(slot, ticket)

    def node(self, account: Any, ticket: str) -> Any:
        _, p, slot = self.split(account)
        return p.node(slot, ticket)

    def account_free(self, account: Any) -> bool:
        _, p, slot = self.split(account)
        return p.account_free(slot)

    def balance(self) -> int | None:
        return None if self.balance_of is None else self.providers[self.balance_of].balance()

    def run_tickets(self) -> list[tuple[str, str]]:
        out = []
        for name, p in self.providers.items():
            out += [(f"{name}{SLOT_SEP}{slot}", ticket) for slot, ticket in p.run_tickets()]
        return out

    def sweep(self) -> list[str]:
        done: list[str] = []
        for p in self.providers.values():
            sweep = getattr(p, "sweep", None)
            if sweep is not None:
                done += list(sweep())
        return done


# -- provider + route + fencer -> pool ops ---------------------------------------


class ProviderPoolOps:
    """:class:`~.island_pool.PoolOps` from a provider, node routes and a fencer.

    ``route``: one :class:`NodeRoute`, or (with a :class:`MultiProvider`) a mapping
    provider name -> route. Without a route the pool sees a member ready once its
    machine is active (no probe, no restart). Without a fencer nothing is fenced
    (the trainer's age filter stays the only line)."""

    start_in_background = False

    def __init__(self, provider: Any, route: Any = None, fencer: Any = None, *, clock=None):
        import time  # noqa: PLC0415

        self.provider, self.routes, self.fencer = provider, route, fencer
        self.clock = clock or time.time
        self._released: set[str] = set()  # tickets these ops released (not a person's release)
        if route is not None:  # the pool looks these up with getattr: only set with a route
            self.probe = self._probe
            self.restart = self._restart
        if fencer is not None:
            self.fence = self._fence
            self.fence_incarnation = self._fence_incarnation
            if getattr(fencer, "request_drain", None) is not None:
                self.request_drain = self._request_drain
                self.drained = self._drained
                self.cancel_drain = self._cancel_drain

    def route_of(self, member: Any) -> Any:
        if self.routes is None or not isinstance(self.routes, Mapping):
            return self.routes
        name = self.provider.split(member.account)[0]
        return self.routes[name]

    def order(self, account: Any) -> str:
        return self.provider.order(account)

    def poll(self, member: Any) -> LeaseStatus:
        return self.provider.status(member.account, member.ticket)

    def balance(self) -> int | None:
        return self.provider.balance()

    def account_free(self, account: Any) -> bool:
        return self.provider.account_free(account)

    def run_tickets(self) -> list[tuple[Any, str]]:
        run_tickets = getattr(self.provider, "run_tickets", None) or self.provider.owned
        return run_tickets()

    def hourly_cost(self, member: Any) -> float | None:
        """One machine hour of ``member`` from the provider (None: unknown)."""
        rate = getattr(self.provider, "hourly_cost", None)
        if rate is None:
            return None
        try:
            return rate(member.ticket)
        except TypeError:
            return rate()

    def ready(self, member: Any) -> bool:  # noqa: ARG002 - without a route, active is ready
        return True

    def start(self, member: Any, shards: list[str]) -> None:
        route = self.route_of(member)
        if route is not None:
            route.start(member, self.provider.node(member.account, member.ticket), shards)

    def _restart(self, member: Any, shards: list[str]) -> None:
        self.route_of(member).restart(member, self.provider.node(member.account, member.ticket), shards)

    def _probe(self, member: Any) -> Any:
        return self.route_of(member).probe(member)

    def _fence(self, member: Any) -> None:
        self.fencer.fence_shard(member.shard, by=member.ticket)

    def _fence_incarnation(self, member: Any) -> None:
        self.fencer.fence_incarnation(member.shard, by=member.ticket)

    def _request_drain(self, member: Any) -> None:
        member.drain_seq = self.fencer.request_drain(member.shard, seq=f"{member.ticket}:{self.clock():.3f}")

    def _drained(self, member: Any) -> bool:
        return member.drain_seq is not None and bool(self.fencer.drain_answered(member.shard, member.drain_seq))

    def _cancel_drain(self, member: Any) -> None:
        self.fencer.cancel_drain(member.shard)
        member.drain_seq = None

    def released_by_us(self, ticket: str) -> bool:
        """An ended-``released`` machine these ops (or the provider's own record) released:
        not a person's release, so the pool refills it."""
        own = getattr(self.provider, "released_by_us", None)
        return ticket in self._released or bool(own is not None and own(ticket))

    def drain(self, member: Any) -> None:
        route = self.route_of(member)
        if route is not None and member.shard:
            try:
                route.stop(member)
            except Exception:  # noqa: BLE001 - best effort: the release below is what stops billing
                pass
        self._released.add(member.ticket)
        self.provider.release(member.account, member.ticket)


# -- the registry ---------------------------------------------------------------


@dataclass(frozen=True)
class ProviderKind:
    """A registered provider kind.

    ``factory``: ``"module:function"`` or a callable ``(cfg, run_id, section)`` ->
    provider (``section``: the pool config's ``provider`` object without ``kind``);
    ``national``: its slots are National Compute accounts (allowed-account check,
    NC tools, the ucr balance); ``ledger_kind``: the resource-ledger kind its machines
    are recorded under (None: not in the ledger); ``release_resource``: ``(resource,
    ops, dry_run) -> (ok, action)`` for ``yeto cleanup``; ``resource_cost``:
    ``(resource, now, rates) -> billing.ItemCost`` for ``yeto cost``; ``pool_owned``:
    a trainer launch's backstop cleanup leaves these to the pool;
    ``cleanup_order``: position among the ledger kinds in a cleanup; ``pool_ops``:
    ``(cfg) -> PoolOps`` replacing the SSH node route; ``offline_problems``: ``(cfg) ->
    [str]`` for ``pool --dry-run``."""

    kind: str
    factory: str | Callable[..., Any]
    national: bool = False
    ledger_kind: str | None = None
    release_resource: str | Callable[..., Any] | None = None
    resource_cost: str | Callable[..., Any] | None = None
    pool_owned: bool = True
    cleanup_order: int = 2
    caps: ProviderCaps = field(default_factory=ProviderCaps)
    # ``(cfg) -> PoolOps`` for machines that are not SSH nodes (None: the SSH / sky route)
    pool_ops: str | Callable[..., Any] | None = None
    # ``(cfg) -> [problem]`` checked by ``pool --dry-run`` without an API call
    offline_problems: str | Callable[..., Any] | None = None


_REGISTRY: dict[str, ProviderKind] = {}


def register(entry: ProviderKind, *, replace: bool = False) -> ProviderKind:
    if entry.kind in _REGISTRY and not replace:
        raise ValueError(f"provider kind {entry.kind!r} is already registered")
    _REGISTRY[entry.kind] = entry
    return entry


def registered(kind: str) -> ProviderKind | None:
    return _REGISTRY.get(kind)


def kinds() -> list[str]:
    return sorted(_REGISTRY)


def resolve(ref: str | Callable[..., Any] | None) -> Callable[..., Any] | None:
    """A callable from ``"package.module:function"`` (imported now) or itself."""
    if ref is None or callable(ref):
        return ref
    module, _, name = str(ref).partition(":")
    if not name:
        raise ValueError(f"{ref!r} must be 'package.module:function'")
    return getattr(importlib.import_module(module), name)


def ledger_kinds() -> dict[str, ProviderKind]:
    """Resource-ledger kind -> its provider kind (cleanup and billing dispatch)."""
    return {e.ledger_kind: e for e in _REGISTRY.values() if e.ledger_kind}


def pool_owned_ledger_kinds() -> frozenset[str]:
    return frozenset(k for k, e in ledger_kinds().items() if e.pool_owned)


def kind_of(cfg: Mapping[str, Any]) -> str | None:
    """A pool config's provider kind (None: ``provider.factory``)."""
    section = cfg.get("provider") or {}
    return None if "factory" in section else str(section.get("kind", "nc"))


def is_national(cfg: Mapping[str, Any]) -> bool:
    kind = kind_of(cfg)
    entry = registered(kind) if kind is not None else None
    return bool(entry and entry.national)


def load(cfg: Mapping[str, Any], run_id: str, *, factories: Mapping[str, Callable[..., Any]] | None = None) -> Any:
    """The provider of a pool config. ``factories``: kind -> ``(cfg, run_id, section)``
    overriding the registered factory (the NC CLI wires its ledger, key and relay)."""
    section = dict(cfg.get("provider") or {"kind": "nc"})
    if "factory" in section:
        return resolve(str(section.pop("factory")))(cfg, run_id)
    kind = str(section.pop("kind", "nc"))
    entry = registered(kind)
    if entry is None:
        names = [repr(k) for k in _REGISTRY]
        listed = " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)
        raise ValueError(f"provider kind {kind!r}: {listed} are built in; use provider.factory for others")
    factory = (factories or {}).get(kind) or resolve(entry.factory)
    return factory(cfg, run_id, section)


# built-in kinds (imported lazily: the registry itself imports no cloud code)
register(ProviderKind(
    "nc", "yeto.launch.nc.provider:registry_factory", national=True, ledger_kind="nc_lease",
    release_resource="yeto.launch.nc.provider:cleanup_release", resource_cost="yeto.launch.nc.provider:ledger_cost",
    caps=ProviderCaps(lease_bounded=True, renewable=False, currency="ucr", balance=True, cold_start_s=3600)))
register(ProviderKind(
    "daytona", "yeto.launch.nc.daytona_provider:registry_factory",
    offline_problems="yeto.launch.nc.pool_run:daytona_problems",
    caps=ProviderCaps(lease_bounded=False, notice_s=0.0, currency="usd")))
register(ProviderKind(
    "sky", "yeto.launch.nc.sky_provider:registry_factory",
    offline_problems="yeto.launch.nc.sky_provider:offline_problems",
    caps=ProviderCaps(lease_bounded=False, notice_s=0.0, currency="usd", cold_start_s=1800)))
register(ProviderKind(
    "modal", "yeto.launch.nc.modal_provider:registry_factory",
    pool_ops="yeto.launch.nc.modal_provider:pool_ops",
    offline_problems="yeto.launch.nc.modal_provider:offline_problems",
    caps=ProviderCaps(lease_bounded=False, notice_s=0.0, currency="usd", cold_start_s=900)))
