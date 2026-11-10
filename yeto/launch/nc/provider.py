"""Lease providers: where the pool's nodes come from.

The elastic pool (:mod:`.lease_pool`) is policy over :class:`~.lease_pool.PoolOps`.
Its National Compute + SSH implementation (:class:`~.cli.NcSshOps`) splits in
two: the *provider* (order a node, read its lease, release it, balance, run
tickets) and the *node route* (SSH + docker on the node: setup, sampler,
probe, fence). This module is the provider contract and its registry, so a
deployment can bring its own supply (a fixed SSH fleet, a k8s node pool, another
cloud) and keep the node route and the whole eviction policy.

A provider must:

* ``order(account) -> ticket``: place one order, recorded with the run tag;
* ``status(account, ticket) -> LeaseStatus``: raise on a failed call (the pool
  treats that as a control-plane blip, never as an end);
* ``release(account, ticket)``: end only a ticket it created for this run, by id,
  and return once it reads ended (raise otherwise; the pool retries);
* ``node(account, ticket) -> NodeLease``: address + login of an active ticket;
* ``balance() -> int | None``, ``account_free(account) -> bool``;
* ``run_tickets() -> [(account, ticket)]``: open tickets of this run it knows of.

Pool config ``provider``: ``{"kind": "nc", ...tool overrides}`` (default), or
``{"factory": "package.module:function", ...}`` - ``function(cfg, run_id)``
returns a provider.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

from .client import Ledger, NcClient, NodeLease, configure_tools
from .elastic import LeaseStatus


class LeaseProvider(Protocol):
    def order(self, account: int) -> str: ...
    def status(self, account: int, ticket: str) -> LeaseStatus: ...
    def release(self, account: int, ticket: str) -> dict[str, Any]: ...
    def node(self, account: int, ticket: str) -> NodeLease: ...
    def balance(self) -> int | None: ...
    def account_free(self, account: int) -> bool: ...
    def run_tickets(self) -> list[tuple[int, str]]: ...


class NcProvider:
    """National Compute through its order tools (guarded :class:`NcClient`)."""

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


def load_provider(cfg: dict[str, Any], run_id: str, *, nc_factory) -> LeaseProvider:
    """The provider of a pool config. ``nc_factory(cfg, run_id)`` builds the National
    Compute one (the CLI wires its ledger, key and relay)."""
    section = dict(cfg.get("provider") or {"kind": "nc"})
    if "factory" in section:
        module, _, name = str(section.pop("factory")).partition(":")
        if not name:
            raise ValueError("provider.factory must be 'package.module:function'")
        return getattr(importlib.import_module(module), name)(cfg, run_id)
    kind = section.pop("kind", "nc")
    if kind != "nc":
        raise ValueError(f"provider kind {kind!r}: only 'nc' is built in; use provider.factory for others")
    configure_tools(section)
    return nc_factory(cfg, run_id)
