"""A fake National Compute order tool with fault injection, and the ownership
invariant it lets a test (or a longer-running harness) check.

The invariant: this tool changes only tickets it can prove are its own.

* :class:`FaultyNc` stands in for ``nc-query`` / ``nc-manage`` (pass it as ``run=`` to
  :class:`~yeto.launch.nc.client.NcClient`). It keeps the truth - every ticket, who
  placed it (``client`` through this fake, or ``foreign``: another user of the account,
  :meth:`FaultyNc.foreign_step`) - and records every changing call with the ticket it
  hit and whether the ledger held that ticket at that moment.
* Each answer is drawn at random: normal, empty, an error document, a timeout, an
  answer without the ticket, one wrong "no order", ``created: false``, an answer
  without ``created``, someone else's ticket, an answer lost after the call took effect.
* Time is a :class:`SimClock` shared with the ledger and the clients: tickets are
  granted, get a lease end, are preempted; the scenario jumps hours and days ahead
  (stale pending orders).
* :func:`violations` lists the changing calls that hit a ticket outside the ledger, the
  nodes handed out for such a ticket, and the adoptions of a ticket that someone
  else placed long after the tool's last order.
* :func:`settle` ends the faults and lets the tool and then an operator clean up;
  :func:`leaks` lists tickets of the tool that are still open and that nobody was told
  about.
* :func:`run_scenario` drives one random sequence (a seed) of client calls.

The residual window. ``nc-manage delete`` takes no ticket id: it releases whatever
ticket the account holds when it runs. The client reads the status right before the
call, but between that read and the delete the ticket can end on its own and someone
else can order. The fake models exactly this point: after a status answer it may end
the tool's ticket and place a foreign one (``race_rate``). If the very next call on
the account is the delete, the hit is recorded as *residual*, not as a violation: no
client-side check can close it. If the client read the status again in between and
still deleted, it is a violation. What must hold after a residual hit
(:func:`residual_outcomes`): the client sends no further changing call on the account
until it has either named the released ticket (``detected``: event
``foreign_ticket_released``, the account halted) or halted the account because it
cannot tell what its delete released (``unconfirmed``: event ``delete_unconfirmed``),
or while the delete is still on record as unconfirmed (``pending``). Anything else is
``undetected`` and must not occur. A released ticket may vanish from the status at
once (``vanish_rate``), so that the evidence is never visible.

Not simulated: two wrong "no order" answers in a row for one account (the client
trusts the second read).
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from yeto.launch.nc import client as ncc

OPEN = ("queued", "active")
LEASE_S = 4 * 3600.0


def _answer(doc: Any = None, *, code: int = 0, stdout: str | None = None, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=code, stdout=json.dumps(doc) if stdout is None else stdout, stderr=stderr)


def _iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, UTC).isoformat()


@dataclass
class SimClock:
    t: float = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@dataclass
class FaultyNc:
    ledger_path: Path
    accounts: tuple[int, ...]
    rng: random.Random
    clock: SimClock = field(default_factory=SimClock)
    faults: bool = True
    fault_rate: float = 0.45
    race_rate: float = 0.04  # after a status answer: the tool's ticket ends, someone else orders
    vanish_rate: float = 0.15  # a ticket the client's delete released is not shown afterwards
    creates_after_residual: int = 0  # orders placed on an account whose residual hit is not accounted for
    _unaccounted: dict[int, int] = field(default_factory=dict)  # account -> index in writes of its residual hit
    accounted: Any = None  # accounted(account) -> bool: the client has halted / still records the delete
    tickets: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> account, by, state, times
    current: dict[int, str | None] = field(default_factory=dict)  # the session's ticket per account
    writes: list[dict[str, Any]] = field(default_factory=list)  # every changing call
    forbidden: list[str] = field(default_factory=list)  # things that must never happen
    last_order_end: dict[int, float] = field(default_factory=dict)  # the client's last order call per account
    n: int = 0
    _flapped: set[int] = field(default_factory=set)
    _raced: set[int] = field(default_factory=set)  # raced after the account's latest status answer

    # -- the truth -----------------------------------------------------------
    def open_ticket(self, account: int) -> dict[str, Any] | None:
        tid = self.current.get(account)
        return self.tickets[tid] if tid and self.tickets[tid]["state"] in OPEN else None

    def _place(self, account: int, by: str) -> dict[str, Any]:
        self.n += 1
        tid = f"{by[0]}{self.n}"
        self.tickets[tid] = {"id": tid, "account": account, "by": by, "state": "queued", "end_reason": None,
                             "placed_at": self.clock(), "granted_at": None, "lease_until": None,
                             "preempt_at": None}
        self.current[account] = tid
        return self.tickets[tid]

    def _grant(self, t: dict[str, Any]) -> None:
        t.update(state="active", granted_at=self.clock(), lease_until=self.clock() + LEASE_S)

    def _in_ledger(self, account: int, tid: str) -> bool:
        try:
            rows = json.loads(self.ledger_path.read_text())
        except (OSError, ValueError):
            return False
        return any(r.get("ticket") == tid and r.get("account") == account for r in rows)

    def _due(self) -> None:
        """Preemptions whose time has come."""
        for t in self.tickets.values():
            if t["state"] in OPEN and t["preempt_at"] is not None and self.clock() >= t["preempt_at"]:
                t.update(state="ended", end_reason="preempted")

    def foreign_step(self) -> None:
        """Another user of the account acts: places an order on a free account, ends
        their own ticket; National Compute grants a ticket or announces a preemption."""
        self._due()
        account = self.rng.choice(self.accounts)
        t = self.open_ticket(account)
        roll = self.rng.random()
        if t is None:
            if roll < 0.5:
                self._place(account, "foreign")
        elif t["state"] == "queued" and roll < 0.6:
            self._grant(t)
        elif t["by"] == "foreign" and roll < 0.8:
            t.update(state="ended", end_reason="released")
        elif t["state"] == "active" and t["preempt_at"] is None and roll > 0.85:
            t["preempt_at"] = self.clock() + self.rng.choice((30.0, 90.0, 400.0))

    def console_release(self, tid: str) -> None:
        """A person releases a ticket in the National Compute console."""
        self.tickets[tid].update(state="ended", end_reason="released", ended_by="console")

    # -- the tools -------------------------------------------------------------
    def _view(self, account: int) -> dict[str, Any] | None:
        tid = self.current.get(account)
        if tid is None:
            return None
        t = self.tickets[tid]
        node = {"public_ip": "192.0.2.9", "ssh_user": "ubuntu", "host_key": "k"} if t["state"] == "active" else None
        return {"id": tid, "state": t["state"], "end_reason": t["end_reason"], "node": node,
                "granted_at": _iso(t["granted_at"]), "lease_until": _iso(t["lease_until"]),
                "preempt_at": _iso(t["preempt_at"]), "others": []}

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
        self._due()
        if "--all" in argv:
            self.forbidden.append(" ".join(argv))
        account = int(argv[argv.index("--account") + 1])
        if argv[0] == "nc-query":
            return self._status(argv, account)
        verb = argv[1]
        if verb == "create":
            try:
                return self._create(argv, account)
            finally:
                self.last_order_end[account] = self.clock()
        if verb == "delete":
            return self._delete(argv, account)
        self.forbidden.append(" ".join(argv))
        return _answer({"ok": False, "error": "unknown"}, code=1)

    def race(self, account: int) -> bool:
        """The residual window: the tool's open ticket on ``account`` ends on its own and
        someone else orders, after the status answer the client is holding."""
        t = self.open_ticket(account)
        if t is None or t["by"] != "client":
            return False
        t.update(state="ended", end_reason="expired")
        self._place(account, "foreign")["raced_from"] = t["id"]
        self._raced.add(account)
        return True

    def _status(self, argv: list[str], account: int) -> SimpleNamespace:
        self._raced.discard(account)  # the client looks again: whatever it does next, it saw the account as it is
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
        out = _answer({"ok": True, "account": account, "balance": {"balance_ucredits": 10**10},
                       "ticket": self._view(account)})
        if self.faults and self.rng.random() < self.race_rate:
            self.race(account)
        return out

    def _create(self, argv: list[str], account: int) -> SimpleNamespace:
        kind = self._fault((*self.BROKEN, "lost_timeout", "lost_garbage", "lost_exit", "lost_no_ticket",
                            "raced", "raced_then_fails", "raced_no_created"))
        if kind.startswith("raced") and self.open_ticket(account) is None:
            self._place(account, "foreign")  # someone else ordered right before this call
        if kind == "raced_then_fails":
            return self._broken(self.rng.choice(("timeout", "error_exit", "garbage")), argv)
        if kind in self.BROKEN:  # failed before the request
            return self._broken(kind, argv)
        found = self.open_ticket(account)
        if found is not None:  # the tool's own check: an open order, none placed
            doc = {"ok": True, "account": account, "created": False, "reason": "the account has an open order",
                   "ticket": self._view(account)}
            if kind == "raced_no_created":  # an answer that does not say whether it placed the order
                del doc["created"]
            return _answer(doc)
        if account in self._unaccounted and not (self.accounted and self.accounted(account)):
            self.creates_after_residual += 1  # the client went on writing without knowing what it had released
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
        lost = kind.startswith("lost_")
        self.writes.append({"op": "delete", "account": account, "ticket": t and t["id"], "by": t and t["by"],
                            "in_ledger": t is not None and self._in_ledger(account, t["id"]),
                            "residual": t is not None and account in self._raced, "answered": not lost,
                            "expected": t and t.get("raced_from"), "at": self.clock()})
        if self.writes[-1]["residual"]:
            self._unaccounted[account] = len(self.writes) - 1
        self._raced.discard(account)
        released = None
        if t is not None:
            t.update(state="ended", end_reason="released", ended_by="client")
            released = {"state": "ended", "end_reason": "released"}
            if self.rng.random() < 0.5:  # the tool's answer may or may not name the ticket
                released["id"] = t["id"]
            if self.faults and self.rng.random() < self.vanish_rate:
                self.current[account] = None  # gone from the status at once
        if lost:
            return self._broken({"lost_timeout": "timeout", "lost_garbage": "garbage",
                                 "lost_exit": "error_exit"}[kind], argv)
        return _answer({"ok": True, "account": account, "released": t is not None, "ticket": released})


def violations(nc: FaultyNc) -> list[str]:
    """Changing calls that hit a ticket the ledger did not hold (outside the residual
    window, see the module doc), and whatever else must never happen."""
    out = [f"forbidden: {c}" for c in nc.forbidden]
    for w in nc.writes:
        if w["op"] == "create" or w["ticket"] is None or w.get("residual"):
            continue
        if not (w["in_ledger"] and w["by"] == "client"):
            out.append(f"{w['op']} hit ticket {w['ticket']} (placed by {w['by']}, in ledger: {w['in_ledger']}) "
                       f"on account {w['account']}")
    return out


def residual_outcomes(sc: Scenario) -> dict[str, list[str]]:
    """What became of each delete that released someone else's ticket inside the
    residual window (the foreign ticket ids):

    * ``detected``: the client named it (``foreign_ticket_released``) and halted the account;
    * ``unconfirmed``: the client halted the account because it cannot tell what its
      delete released (``delete_unconfirmed`` for the ticket the delete was meant for);
    * ``pending``: the delete is still on record as unconfirmed (no trusted status
      since); the account takes no other write;
    * ``undetected``: none of these - the client does not know and may go on writing."""
    named = {d["ticket"] for name, d in sc.events if name == "foreign_ticket_released"}
    unconfirmed = {(d["account"], d["expected"]) for name, d in sc.events if name == "delete_unconfirmed"}
    out: dict[str, list[str]] = {"detected": [], "unconfirmed": [], "pending": [], "undetected": []}
    for w in sc.nc.writes:
        if not w.get("residual"):
            continue
        pending = sc.ledger.pending_delete(w["account"])
        if w["ticket"] in named:
            out["detected"].append(w["ticket"])
        elif (w["account"], w["expected"]) in unconfirmed:
            out["unconfirmed"].append(w["ticket"])
        elif pending is not None and pending["pending_delete"] == w["expected"]:
            out["pending"].append(w["ticket"])
        else:
            out["undetected"].append(w["ticket"])
    return out


def check_invariants(sc: Scenario) -> list[str]:
    """The two standing invariants, for a test or a simulation that runs the client
    over :class:`FaultyNc` (call it at any point; an empty list is a pass):

    1. no changing call hit a ticket outside the ledger, no node of such a ticket was
       handed out, no foreign ticket was adopted late (:func:`violations`);
    2. no delete that took someone else's ticket in the residual window went
       unnoticed with the account still writable (:func:`residual_outcomes`), and no
       order was placed on an account before such a hit was accounted for."""
    out = violations(sc.nc)
    out += [f"residual hit on ticket {t} is undetected and the account is writable"
            for t in residual_outcomes(sc)["undetected"]]
    if sc.nc.creates_after_residual:
        out.append(f"{sc.nc.creates_after_residual} order(s) placed on an account with an unaccounted residual hit")
    return out


def false_halts(sc: Scenario) -> int:
    """``delete_unconfirmed`` halts for a delete that did release the tool's own ticket
    (the cost of halting whenever the target cannot be established)."""
    residual = {(w["account"], w["expected"]) for w in sc.nc.writes if w.get("residual")}
    return sum(1 for name, d in sc.events
               if name == "delete_unconfirmed" and (d["account"], d["expected"]) not in residual)


@dataclass
class Scenario:
    nc: FaultyNc
    ledger: ncc.Ledger
    clients: dict[int, ncc.NcClient]
    clock: SimClock
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def reported(self) -> set[str]:
        """Tickets reported as a possible unrecorded order of the tool (adoptable or stale)."""
        return {d["ticket"] for name, d in self.events if name == "order_unrecorded"}


def scenario(tmp_path: Path, seed: int, *, accounts: tuple[int, ...] = (1, 4, 5), fault_rate: float = 0.45,
             race_rate: float = 0.04, vanish_rate: float = 0.15) -> Scenario:
    clock = SimClock()
    ledger = ncc.Ledger(tmp_path / "tickets.json", clock=clock)
    nc = FaultyNc(ledger.path, accounts, random.Random(seed), clock, fault_rate=fault_rate, race_rate=race_rate,
                  vanish_rate=vanish_rate)
    nc.accounted = lambda account: ledger.halted(account) is not None or ledger.pending_delete(account) is not None
    sc = Scenario(nc, ledger, {}, clock)
    for a in accounts:
        sc.clients[a] = ncc.NcClient(a, ledger, run=nc, sleep=lambda s: None,
                                     event=lambda name, **detail: sc.events.append((name, detail)))
    return sc


EXPECTED = (ncc.NcError, subprocess.TimeoutExpired)  # what a call may raise under faults


def _stale_for(sc: Scenario, t: dict[str, Any]) -> bool:
    """``t`` was placed so long after the tool's last order call on the account that no
    pending order can be tied to it."""
    last = sc.nc.last_order_end.get(t["account"])
    return last is None or t["placed_at"] > last + ncc.ORDER_PENDING_MAX_S + ncc.ORDER_IN_FLIGHT_S


def run_scenario(sc: Scenario, steps: int = 150) -> None:
    """``steps`` random client calls (order, read, release, release / wait / adopt of a
    ticket that is not in the ledger, look at an account), with the other party acting
    and time passing in between."""
    rng, nc = sc.nc.rng, sc.nc
    for _ in range(steps):
        sc.clock.advance(rng.choice((1.0, 5.0, 20.0, 60.0, 60.0, 300.0, 3600.0, 2 * 86400.0)))
        if rng.random() < 0.3:
            nc.foreign_step()
        account = rng.choice(nc.accounts)
        c = sc.clients[account]
        op = rng.choice(("create", "create", "status", "delete", "delete_any", "free", "wait", "wait_any",
                         "adopt_stale"))
        rows = [r for r in sc.ledger.open_rows() if r["account"] == account]
        shown = nc.current.get(account)
        try:
            if op == "create":
                c.create(run="run-sim")
            elif op == "free":
                c.account_free()
            elif op == "delete_any" and nc.tickets:  # whatever ticket, in the ledger or not
                c.delete(shown if shown and rng.random() < 0.7 else rng.choice(sorted(nc.tickets)))
            elif op == "wait_any" and shown:  # the node of whatever the account shows
                in_ledger = nc._in_ledger(account, shown)
                lease = c.wait_active(shown, timeout_s=0, log=lambda line: None)
                if not in_ledger:
                    nc.forbidden.append(f"node {lease.ip} of ticket {shown} handed out, in no ledger")
            elif op == "adopt_stale" and shown:  # a careless operator, long after the tool's last order
                t = nc.tickets[shown]
                if t["by"] == "foreign" and t["state"] in OPEN and _stale_for(sc, t):
                    c.adopt(shown)
                    nc.forbidden.append(f"foreign ticket {shown} adopted long after the last order")
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
    holds (waiting out a ticket that is about to end on its own); then (``operator``) a
    person lifts the halts, adopts each reported ticket that the tool's order did
    place - they can see that in the console - or releases it there, and the tool
    releases what was adopted."""
    sc.nc.faults = False
    for _ in range(3):
        for c in sc.clients.values():
            try:
                c.account_free()
            except ncc.NcError:
                pass
        deferred = False
        for row in sc.ledger.open_rows():
            try:
                sc.clients[row["account"]].delete(row["ticket"])
            except ncc.NcReleaseDeferred:
                deferred = True
            except ncc.NcError:  # ended behind someone else's order; a halted account; a foreign release
                pass
        if not deferred:
            break
        sc.clock.advance(2 * ncc.DELETE_GUARD_S + 1)
        sc.nc._due()
    if not operator:
        return
    for account, c in sc.clients.items():
        sc.ledger.clear_halt(account)
        sc.nc._unaccounted.pop(account, None)  # the operator looked into it
        t = sc.nc.open_ticket(account)
        if t is None or t["by"] != "client":
            continue
        if not sc.nc._in_ledger(account, t["id"]):
            if t["id"] not in sc.reported():
                continue
            try:
                c.adopt(t["id"])
            except ncc.NcRefused:  # reported as stale: released by hand
                sc.nc.console_release(t["id"])
                continue
        for _ in range(3):
            try:
                c.delete(t["id"])
                break
            except ncc.NcReleaseDeferred:
                sc.clock.advance(2 * ncc.DELETE_GUARD_S + 1)
                sc.nc._due()


def leaks(sc: Scenario) -> list[str]:
    """Tickets the tool's orders placed that are still open and were never reported
    for an operator (after :func:`settle`: every one of them is a silent, billing leak).
    A ticket on a halted account counts as reported: the halt names the account."""
    return [f"ticket {t['id']} on account {t['account']} is open, in the ledger: "
            f"{sc.nc._in_ledger(t['account'], t['id'])}, reported: {t['id'] in sc.reported()}"
            for t in sc.nc.tickets.values()
            if t["by"] == "client" and t["state"] in OPEN and t["id"] not in sc.reported()
            and sc.ledger.halted(t["account"]) is None]


def still_open(sc: Scenario, by: str) -> list[str]:
    return sorted(t["id"] for t in sc.nc.tickets.values() if t["by"] == by and t["state"] in OPEN)
