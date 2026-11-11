"""The National Compute island provider (:class:`NCProvider`).

The contract (:class:`~yeto.launch.pool.provider.IslandProvider`),
:class:`~yeto.launch.pool.provider.MultiProvider` and the provider registry live
in :mod:`yeto.launch.pool.provider`. The pool's National Compute +
SSH ops (:class:`~.cli.NcSshOps`) compose this provider with the SSH + docker node
route and the shard fence.

Pool config ``provider``: ``{"kind": "nc", ...tool overrides}`` (default),
``{"kind": "daytona", ...}`` (:mod:`.daytona_provider`: Daytona GPU sandboxes), or
``{"factory": "package.module:function", ...}`` - ``function(cfg, run_id)``
returns a provider.

The resource-ledger kind ``nc_lease`` is registered to this provider:
:func:`cleanup_release` and :func:`ledger_cost` are what ``yeto cleanup`` and
``yeto cost`` run for it.
"""

from __future__ import annotations

from typing import Any

from ..pool import provider as registry
from ..pool.provider import IslandProvider, ProviderCaps
from ..pool.status import LeaseStatus
from .client import Ledger, NcClient, NodeLease, configure_tools, hourly_cost_ucr


class NCProvider:
    """National Compute through its order tools (guarded :class:`NcClient`)."""

    kind = "nc"
    caps = ProviderCaps(lease_bounded=True, renewable=False, currency="ucr", balance=True, cold_start_s=3600)

    def __init__(self, accounts: list[int], run_id: str, *, ledger: Ledger, allowed: set[int],
                 model: str = "mi355x", pubkey=None, before_order=None, run=None):
        self.accounts, self.run_id, self.ledger, self.allowed = list(accounts), run_id, ledger, allowed
        self.model, self.pubkey, self.before_order = model, pubkey, before_order
        self._run = run

    def client(self, account: int, *, ordering: bool = False) -> NcClient:
        kw = {"run": self._run} if self._run is not None else {}
        return NcClient(account, self.ledger, allowed_accounts=self.allowed if ordering else None, **kw)

    def order(self, account: int) -> str:
        c = self.client(account, ordering=True)
        if self.before_order is not None:
            self.before_order(account)
        return c.create(model=self.model, pubkey=self.pubkey() if callable(self.pubkey) else self.pubkey,
                        run=self.run_id)

    def status(self, account: int, ticket: str) -> LeaseStatus:
        return self.client(account).ticket_status(ticket)

    def release(self, account: int, ticket: str) -> dict[str, Any]:
        return self.client(account).delete(ticket)

    def node(self, account: int, ticket: str) -> NodeLease:
        return self.client(account).wait_active(ticket, timeout_s=60)

    def balance(self) -> int | None:
        return self.client(self.accounts[0]).balance()

    def account_free(self, account: int) -> bool:
        return NcClient.own_ticket(self.client(account).status()) is None

    def run_tickets(self) -> list[tuple[int, str]]:
        accounts = set(self.accounts)
        return [(int(r["account"]), str(r["ticket"])) for r in self.ledger.run_rows(self.run_id)
                if int(r["account"]) in accounts]

    def hourly_cost(self) -> int:
        return hourly_cost_ucr()


def is_national_compute(cfg: dict[str, Any]) -> bool:
    """The pool orders National Compute nodes (its accounts are NC accounts, its
    tools and ledger apply); False for ``provider.factory`` and ``kind: daytona``."""
    return registry.is_national(cfg)


def provider_kind(cfg: dict[str, Any]) -> str:
    """The provider kind of a pool config: a registered kind (``nc`` by default), or
    ``factory`` for a ``provider.factory`` (the sky factory reads as ``sky``)."""
    kind = registry.kind_of(cfg)
    if kind is not None:
        return kind
    factory = str((cfg.get("provider") or {}).get("factory", ""))
    return "sky" if factory.startswith("yeto.launch.nc.sky_provider:") else "factory"


def _with_tools(nc_factory):
    def factory(cfg: dict[str, Any], run_id: str, section: dict[str, Any]):
        configure_tools(section)
        return nc_factory(cfg, run_id)

    return factory


def registry_factory(cfg: dict[str, Any], run_id: str, section: dict[str, Any]) -> NCProvider:
    """The registered ``nc`` factory: the CLI's provider (its ledger, lease key, relay)."""
    from .cli import nc_provider  # noqa: PLC0415 - the CLI imports this module

    return _with_tools(nc_provider)(cfg, run_id, section)


def load_provider(cfg: dict[str, Any], run_id: str, *, nc_factory) -> IslandProvider:
    """The provider of a pool config (:func:`yeto.launch.pool.provider.load`).
    ``nc_factory(cfg, run_id)`` builds the National Compute one."""
    return registry.load(cfg, run_id, factories={"nc": _with_tools(nc_factory)})


# -- the resource ledger (``yeto cleanup`` / ``yeto cost`` dispatch on kind nc_lease) ---

def cleanup_release(r: Any, ops: Any, dry_run: bool) -> tuple[bool, str, Any]:
    """Release one ``nc_lease`` ledger entry through ``ops.nc_release`` (the NC lease
    manager's guarded down: it refuses tickets it did not create). Returns ``(ok,
    action, result)``; ``result`` None when nothing was released."""
    account = r.attrs.get("account")
    if account is None:
        return False, "NC lease without account in the ledger: not released", None
    if dry_run:
        return True, f"would release NC ticket {r.id} (account {account})", None
    res = ops.nc_release(int(account), r.id)
    return True, f"released NC ticket {r.id}: {res}", res


def ledger_cost(r: Any, now: float, rates: Any) -> Any:
    """The cost of one ``nc_lease`` ledger entry: granted to released (or now) at the
    ticket's rate; notes the minutes past ``lease_until``."""
    from yeto.cost import billing  # noqa: PLC0415

    start = billing._ts(r.attrs.get("granted_at")) or r.created_at  # noqa: SLF001
    note = ""
    until = billing._ts(r.attrs.get("lease_until"))  # noqa: SLF001
    end = r.released_at if r.released_at is not None else now
    if until is not None and end > until:
        note = f"{(end - until) / 60:.0f} min past lease_until"
    return billing._span_cost(r, start, now, billing._nc_rate(r, rates), note)  # noqa: SLF001
