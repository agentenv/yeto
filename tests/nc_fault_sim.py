"""A fake National Compute order tool with fault injection, and the ownership
invariant it lets a test (or a longer-running harness) check.

The invariant: this tool changes only tickets it can prove are its own.

* :class:`FaultyNc` stands in for ``nc-query`` / ``nc-manage`` (pass it as ``run=`` to
  :class:`~yeto.launch.nc.client.NcClient`). It keeps the truth - every ticket, who
  placed it (``client`` through this fake, or ``foreign``: another session's user,
  :meth:`FaultyNc.foreign_step`) - and records every changing call with the ticket it
  hit and whether the ledger held that ticket at that moment.
* Each answer is drawn at random: normal, empty, an error document, a timeout, an
  answer without the ticket, one wrong "no order", ``created: false``, someone else's
  ticket, an answer lost after the call took effect.
* :func:`violations` lists the changing calls that hit a ticket outside the ledger.
* :func:`settle` ends the faults and lets the tool and then an operator clean up;
  :func:`leaks` lists tickets of the tool that are still open and that nobody was told
  about.
* :func:`run_scenario` drives one random sequence (a seed) of client calls.

Not simulated: two wrong "no order" answers in a row for one account (the client
trusts the second read), and another party acting between the status read and the
``nc-manage delete`` of one release (the order tool's delete takes no ticket id).
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from yeto.launch.nc import client as ncc

OPEN = ("queued", "active")


def _answer(doc: Any = None, *, code: int = 0, stdout: str | None = None, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=code, stdout=json.dumps(doc) if stdout is None else stdout, stderr=stderr)


@dataclass
class FaultyNc:
    ledger_path: Path
    accounts: tuple[int, ...]
    rng: random.Random
    faults: bool = True
    fault_rate: float = 0.45
    tickets: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> account, by, state, end_reason
    current: dict[int, str | None] = field(default_factory=dict)  # the session's ticket per account
    writes: list[dict[str, Any]] = field(default_factory=list)  # every changing call
    forbidden: list[str] = field(default_factory=list)  # calls that must never be made
    n: int = 0
    _flapped: set[int] = field(default_factory=set)

    # -- the truth -----------------------------------------------------------
    def open_ticket(self, account: int) -> dict[str, Any] | None:
        tid = self.current.get(account)
        return self.tickets[tid] if tid and self.tickets[tid]["state"] in OPEN else None

    def _place(self, account: int, by: str) -> dict[str, Any]:
        self.n += 1
        tid = f"{by[0]}{self.n}"
        self.tickets[tid] = {"id": tid, "account": account, "by": by, "state": "queued", "end_reason": None}
        self.current[account] = tid
        return self.tickets[tid]

    def _in_ledger(self, account: int, tid: str) -> bool:
        try:
            rows = json.loads(self.ledger_path.read_text())
        except (OSError, ValueError):
            return False
        return any(r.get("ticket") == tid and r.get("account") == account for r in rows)

    def foreign_step(self) -> None:
        """Another user of the account acts: places an order on a free account, ends
        their own ticket; National Compute grants or preempts a ticket."""
        account = self.rng.choice(self.accounts)
        t = self.open_ticket(account)
        roll = self.rng.random()
        if t is None:
            if roll < 0.5:
                self._place(account, "foreign")
        elif t["state"] == "queued" and roll < 0.6:
            t["state"] = "active"
        elif t["by"] == "foreign" and roll < 0.8:
            t.update(state="ended", end_reason="released")
        elif roll > 0.9:
            t.update(state="ended", end_reason="preempted")

    # -- the tools -------------------------------------------------------------
    def _view(self, account: int) -> dict[str, Any] | None:
        tid = self.current.get(account)
        if tid is None:
            return None
        t = self.tickets[tid]
        node = {"public_ip": "192.0.2.9", "ssh_user": "ubuntu", "host_key": "k"} if t["state"] == "active" else None
        return {"id": tid, "state": t["state"], "end_reason": t["end_reason"], "node": node, "others": []}

    def _fault(self, kinds: tuple[str, ...]) -> str:
        if not self.faults or self.rng.random() > self.fault_rate:
            return "normal"
        return self.rng.choice(kinds)

    def _broken(self, kind: str, argv: list[str]) -> SimpleNamespace:
        if kind == "timeout":
            raise subprocess.TimeoutExpired(argv, 300)
        if kind == "empty":
            return _answer(stdout="")
        if kind == "error_exit":
            return _answer(code=1, stdout="", stderr='{"ok":false,"error":"reading the current order failed: 502"}')
        if kind == "error_json":
            return _answer({"ok": False, "error": "gateway"})
        if kind == "garbage":
            return _answer(stdout='{"ok": true, "ticket": {"id": "')
        if kind == "no_ticket":
            return _answer({"ok": True, "account": 0})
        raise AssertionError(kind)

    BROKEN = ("timeout", "empty", "error_exit", "error_json", "garbage", "no_ticket")

    def __call__(self, argv: list[str], **kw: Any) -> SimpleNamespace:
        argv = list(argv)
        if "--all" in argv:
            self.forbidden.append(" ".join(argv))
        account = int(argv[argv.index("--account") + 1])
        if argv[0] == "nc-query":
            return self._status(argv, account)
        verb = argv[1]
        if verb == "create":
            return self._create(argv, account)
        if verb == "delete":
            return self._delete(argv, account)
        self.forbidden.append(" ".join(argv))
        return _answer({"ok": False, "error": "unknown"}, code=1)

    def _status(self, argv: list[str], account: int) -> SimpleNamespace:
        kind = self._fault((*self.BROKEN, "bad_ticket", "wrong_none"))
        if kind == "wrong_none" and (account in self._flapped or self.current.get(account) is None):
            kind = "normal"
        if kind != "wrong_none":
            self._flapped.discard(account)
        if kind in self.BROKEN:
            return self._broken(kind, argv)
        if kind == "bad_ticket":
            return _answer({"ok": True, "ticket": {"detail": "<html>502</html>"}})
        if kind == "wrong_none":  # one 404 for an account that has a ticket (never two in a row)
            self._flapped.add(account)
            return _answer({"ok": True, "account": account, "ticket": None})
        return _answer({"ok": True, "account": account, "balance": {"balance_ucredits": 10**10},
                        "ticket": self._view(account)})

    def _create(self, argv: list[str], account: int) -> SimpleNamespace:
        kind = self._fault((*self.BROKEN, "lost_timeout", "lost_garbage", "lost_exit", "lost_no_ticket",
                            "raced", "raced_then_fails"))
        if kind in ("raced", "raced_then_fails") and self.open_ticket(account) is None:
            self._place(account, "foreign")  # someone else ordered right before this call
        if kind == "raced_then_fails":
            return self._broken(self.rng.choice(("timeout", "error_exit", "garbage")), argv)
        if kind in self.BROKEN:  # failed before the request
            return self._broken(kind, argv)
        found = self.open_ticket(account)
        if found is not None:  # the tool's own check: an open order, none placed
            return _answer({"ok": True, "account": account, "created": False, "reason": "the account has an open order",
                            "ticket": self._view(account)})
        t = self._place(account, "client")
        self.writes.append({"op": "create", "account": account, "ticket": None, "by": None, "in_ledger": None})
        if kind.startswith("lost_"):  # placed; the answer does not come back
            lost = {"lost_timeout": "timeout", "lost_garbage": "garbage", "lost_exit": "error_exit",
                    "lost_no_ticket": "no_ticket"}[kind]
            return self._broken(lost, argv)
        return _answer({"ok": True, "account": account, "created": True, "ticket": {"id": t["id"], "state": "queued"}})

    def _delete(self, argv: list[str], account: int) -> SimpleNamespace:
        kind = self._fault((*self.BROKEN, "lost_timeout", "lost_garbage", "lost_exit"))
        if kind in self.BROKEN:
            return self._broken(kind, argv)
        t = self.open_ticket(account)
        self.writes.append({"op": "delete", "account": account, "ticket": t and t["id"], "by": t and t["by"],
                            "in_ledger": t is not None and self._in_ledger(account, t["id"])})
        if t is not None:
            t.update(state="ended", end_reason="released", ended_by="client")
        if kind.startswith("lost_"):
            return self._broken({"lost_timeout": "timeout", "lost_garbage": "garbage",
                                 "lost_exit": "error_exit"}[kind], argv)
        return _answer({"ok": True, "account": account, "released": t is not None})


def violations(nc: FaultyNc) -> list[str]:
    """Changing calls that hit a ticket the ledger did not hold, or that are never allowed."""
    out = [f"forbidden call: {c}" for c in nc.forbidden]
    for w in nc.writes:
        if w["op"] != "create" and w["ticket"] is not None and not (w["in_ledger"] and w["by"] == "client"):
            out.append(f"{w['op']} hit ticket {w['ticket']} (placed by {w['by']}, in ledger: {w['in_ledger']}) "
                       f"on account {w['account']}")
    out += [f"foreign ticket {t['id']} was ended by the client" for t in nc.tickets.values()
            if t["by"] == "foreign" and t.get("ended_by") == "client"]
    return out


@dataclass
class Scenario:
    nc: FaultyNc
    ledger: ncc.Ledger
    clients: dict[int, ncc.NcClient]
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def reported(self) -> set[str]:
        return {d["ticket"] for name, d in self.events if name == "order_unrecorded"}


def scenario(tmp_path: Path, seed: int, *, accounts: tuple[int, ...] = (1, 4, 5), fault_rate: float = 0.45) -> Scenario:
    ledger = ncc.Ledger(tmp_path / "tickets.json")
    nc = FaultyNc(ledger.path, accounts, random.Random(seed), fault_rate=fault_rate)
    sc = Scenario(nc, ledger, {})
    for a in accounts:
        sc.clients[a] = ncc.NcClient(a, ledger, run=nc, sleep=lambda s: None,
                                     event=lambda name, **detail: sc.events.append((name, detail)))
    return sc


EXPECTED = (ncc.NcError, subprocess.TimeoutExpired)  # what a call may raise under faults


def run_scenario(sc: Scenario, steps: int = 150) -> None:
    """``steps`` random client calls (order, read, release, release of a ticket that is
    not in the ledger, look at an account), with the other party acting in between."""
    rng = sc.nc.rng
    for _ in range(steps):
        if rng.random() < 0.3:
            sc.nc.foreign_step()
        account = rng.choice(sc.nc.accounts)
        c = sc.clients[account]
        op = rng.choice(("create", "create", "status", "delete", "delete_any", "free", "wait"))
        rows = [r for r in sc.ledger.open_rows() if r["account"] == account]
        try:
            if op == "create":
                c.create(run="run-sim")
            elif op == "free":
                c.account_free()
            elif op == "delete_any" and sc.nc.tickets:  # whatever the account shows, in the ledger or not
                c.delete(rng.choice(sorted(sc.nc.tickets)))
            elif rows and op == "status":
                c.ticket_status(rng.choice(rows)["ticket"])
            elif rows and op == "delete":
                c.delete(rng.choice(rows)["ticket"])
            elif rows and op == "wait":
                c.wait_active(rng.choice(rows)["ticket"], timeout_s=0, log=lambda line: None)
        except EXPECTED:
            pass


def settle(sc: Scenario, *, operator: bool = True) -> None:
    """The faults end. The tool looks at every account and releases what its ledger
    holds; then (``operator``) a person adopts each reported ticket that the tool's
    order did place - they can see that in the console - and releases it."""
    sc.nc.faults = False
    for account, c in sc.clients.items():
        c.account_free()
    for row in sc.ledger.open_rows():
        try:
            sc.clients[row["account"]].delete(row["ticket"])
        except ncc.NcRefused:  # ended, and the account now shows someone else's ticket
            pass
    if not operator:
        return
    for account, c in sc.clients.items():
        t = sc.nc.open_ticket(account)
        if t is not None and t["by"] == "client" and t["id"] in sc.reported():
            c.adopt(t["id"])
            c.delete(t["id"])


def leaks(sc: Scenario) -> list[str]:
    """Tickets the tool's orders placed that are still open and were never reported
    for an operator (after :func:`settle`: every one of them is a silent, billing leak)."""
    return [f"ticket {t['id']} on account {t['account']} is open, in the ledger: "
            f"{sc.nc._in_ledger(t['account'], t['id'])}, reported: {t['id'] in sc.reported()}"
            for t in sc.nc.tickets.values()
            if t["by"] == "client" and t["state"] in OPEN and t["id"] not in sc.reported()]


def still_open(sc: Scenario, by: str) -> list[str]:
    return sorted(t["id"] for t in sc.nc.tickets.values() if t["by"] == by and t["state"] in OPEN)
