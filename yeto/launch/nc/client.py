"""The National Compute order tool (``nc-manage`` / ``nc-query``) as a guarded client.

Rules this client enforces (the accounts are shared with production):

* an order is placed only when the account has no open ticket of this session
  (``others[]`` - tickets of other sessions - is ignored, never acted on);
* every ticket this client creates is recorded in the ledger
  (``<state_dir>/tickets.json``); ``delete`` refuses a ticket that is not in the
  ledger, and refuses unless the account's current ticket id equals the
  expected id right before the call;
* ``delete --all`` is never issued; only ``nc-manage delete --account N --yes``,
  which releases the current ticket of this session.

The tools read their own account file; no account secret passes through here.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..pool.status import LeaseStatus

OPEN_STATES = frozenset({"queued", "held", "provisioning", "active", "releasing"})
MI355X_RATE_UCR_PER_GPU_HOUR = 3_000_000
LEASE_HOURS = 4.0


class NcError(RuntimeError):
    pass


class NcRefused(NcError):
    """A guard refused the action; nothing was sent to National Compute."""


@dataclass(frozen=True)
class NodeLease:
    account: int
    ticket: str
    ip: str
    user: str
    host_key: str
    granted_at: str | None
    lease_until: str | None
    gpus: int | None
    rate_ucr_per_gpu_hour: int | None

    @property
    def lease_until_ts(self) -> float | None:
        return parse_time(self.lease_until)

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value).astimezone(UTC).timestamp()


@dataclass(frozen=True)
class NcTools:
    """Where the National Compute tools live. Defaults come from the environment
    (``NC_MANAGE``, ``NC_QUERY``, ``NC_TOOL_DIR``, ``NC_NODE_BIN``); a pool config's
    ``provider`` section overrides them (:func:`configure_tools`). Nothing here
    assumes a home-directory install."""

    manage: tuple[str, ...]
    query: tuple[str, ...]
    tool_dir: str | None
    node_bin: str

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> NcTools:
        o = dict(overrides or {})
        unknown = set(o) - {"manage", "query", "tool_dir", "node_bin"}
        if unknown:
            raise ValueError(f"nc tools: unknown keys {sorted(unknown)}")

        def argv(key: str, env: str, default: str) -> tuple[str, ...]:
            value = o.get(key) or os.environ.get(env) or default
            return tuple(shlex.split(value) if isinstance(value, str) else value)

        tool_dir = o.get("tool_dir") or os.environ.get("NC_TOOL_DIR")
        return cls(argv("manage", "NC_MANAGE", "nc-manage"), argv("query", "NC_QUERY", "nc-query"),
                   os.path.expanduser(tool_dir) if tool_dir else None,
                   str(o.get("node_bin") or os.environ.get("NC_NODE_BIN", "node")))


_TOOLS: NcTools | None = None


def configure_tools(overrides: dict[str, Any] | None) -> NcTools:
    """Pin the tools for this process (a pool config's ``provider`` section)."""
    global _TOOLS
    _TOOLS = NcTools.from_env(overrides)
    return _TOOLS


def tools() -> NcTools:
    return _TOOLS if _TOOLS is not None else NcTools.from_env()


def _tool(kind: str) -> list[str]:
    return list(getattr(tools(), kind))


COST_LEDGER_ENV = "YETO_RESOURCE_LEDGER"  # yeto.cost.ledger.LEDGER_ENV


def cost_record(op: str, ticket: str, *, name: str | None = None, **attrs: Any) -> None:
    """Record an NC lease (kind ``nc_lease``, id = ticket) in the run's cost ledger
    (``$YETO_RESOURCE_LEDGER``) so ``yeto cost``, the budget and ``yeto cleanup`` see
    it. ``op``: ``created`` / ``released`` / ``release_failed``. No ledger: nothing.
    A write error is printed, never raised (it must not break an order or a delete)."""
    path = os.environ.get(COST_LEDGER_ENV)
    if not path:
        return
    try:
        try:
            from yeto.cost.ledger import ResourceLedger  # noqa: PLC0415
        except ImportError:  # a tree without the cost guard: the same record format
            ResourceLedger = None  # noqa: N806
        if ResourceLedger is not None:
            led = ResourceLedger(path)
            if op == "release_failed":
                led.release_failed("nc_lease", ticket, name, attrs.get("error", ""))
            else:
                getattr(led, op)("nc_lease", ticket, name, **attrs)
            return
        rec = {"op": {"created": "create", "released": "release"}.get(op, op), "kind": "nc_lease",
               "id": str(ticket), "name": name, "ts": time.time(), "attrs": attrs}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(rec, default=str, sort_keys=True) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception as error:  # noqa: BLE001
        print(f"WARNING: cost ledger write for NC ticket {ticket} failed: {error!r}", file=sys.stderr, flush=True)


class Ledger:
    """Tickets created by this tool (append-only file, 0600)."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []

    def add(self, account: int, ticket: str, **extra: Any) -> None:
        rows = self.load()
        rows.append({"account": account, "ticket": ticket, "created_at": time.time(), **extra})
        self._write(rows)

    def mark(self, ticket: str, **fields: Any) -> None:
        rows = self.load()
        for row in rows:
            if row["ticket"] == ticket:
                row.update(fields)
        self._write(rows)

    def owns(self, account: int, ticket: str) -> bool:
        return any(r["account"] == account and r["ticket"] == ticket for r in self.load())

    def open_rows(self) -> list[dict[str, Any]]:
        return [r for r in self.load() if not r.get("ended")]

    def run_rows(self, run: str) -> list[dict[str, Any]]:
        """Open tickets this tool created for ``run`` (the tag written at create)."""
        return [r for r in self.open_rows() if r.get("run") == run]

    def _write(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(rows, f, indent=1)
        os.replace(tmp, self.path)


class NcClient:
    """One account. ``run`` is ``subprocess.run`` (tests pass a fake)."""

    def __init__(self, account: int, ledger: Ledger, *, run: Callable[..., Any] = subprocess.run,
                 sleep: Callable[[float], None] = time.sleep, allowed_accounts: set[int] | None = None):
        if allowed_accounts is not None and account not in allowed_accounts:
            raise NcRefused(f"account {account} is not in the allowed accounts {sorted(allowed_accounts)}")
        self.account = account
        self.ledger = ledger
        self.run = run
        self.sleep = sleep

    def _call(self, argv: list[str], env: dict[str, str] | None = None, timeout: float = 300) -> dict[str, Any]:
        if "--all" in argv:
            raise NcRefused("nc-manage --all is never used")
        full_env = None if env is None else {**os.environ, **env}
        out = self.run(argv, capture_output=True, text=True, timeout=timeout, check=False, env=full_env)
        text = (out.stdout or "").strip()
        try:
            doc = json.loads(text[text.index("{"):]) if "{" in text else {}
        except ValueError as exc:
            raise NcError(f"{argv[0]}: unparseable output (exit {out.returncode})") from exc
        if out.returncode != 0 and not doc:
            raise NcError(f"{argv[0]} exit {out.returncode}: {(out.stderr or '')[-300:]}")
        return doc

    def status(self) -> dict[str, Any]:
        return self._call([*_tool("query"), "status", "--account", str(self.account)])

    def balance(self) -> int | None:
        bal = (self.status().get("balance") or {}).get("balance_ucredits")
        return int(bal) if bal is not None else None

    @staticmethod
    def own_ticket(status: dict[str, Any]) -> dict[str, Any] | None:
        """The ticket of this session if open (``others[]`` is not looked at)."""
        ticket = status.get("ticket") or {}
        return ticket if ticket.get("state") in OPEN_STATES else None

    def ticket_status(self, ticket: str) -> LeaseStatus:
        """``ticket`` as a :class:`~.elastic.LeaseStatus`. The account's current
        ticket is the only one the tool reports; another id (or none) means ``ticket``
        ended (one open order per account). A failed call raises."""
        from ..pool.status import LeaseStatus  # noqa: PLC0415

        t = self.status().get("ticket") or {}
        if t.get("id") != ticket:
            return LeaseStatus("ended", end_reason=None)
        state = t.get("state")
        until, preempt = parse_time(t.get("lease_until")), parse_time(t.get("preempt_at"))
        gpus, rate = t.get("gpus"), t.get("rate_ucr_per_gpu_hour")
        if state == "active" and (t.get("node") or {}).get("public_ip"):
            return LeaseStatus("active", until, None, preempt, gpus, rate)
        if state == "releasing":  # the node is going away: retire it (the drain confirms ended)
            return LeaseStatus("ended", until, "releasing", preempt, gpus, rate)
        if state in OPEN_STATES:
            return LeaseStatus("waiting", until, None, preempt, gpus, rate)
        return LeaseStatus("ended", until, t.get("end_reason"), preempt, gpus, rate)

    def create(self, *, model: str = "mi355x", pubkey: str | None = None, run: str | None = None) -> str:
        st = self.status()
        if self.own_ticket(st):
            raise NcRefused(f"account {self.account} already has open ticket "
                            f"{st['ticket'].get('id')} ({st['ticket'].get('state')}); one order per account")
        env = {"NC_PUBKEY": pubkey} if pubkey else None
        doc = self._call([*_tool("manage"), "create", "--account", str(self.account),
                          "--model", model, "--yes"], env=env)
        ticket = (doc.get("ticket") or {}).get("id")
        if not doc.get("ok") or not ticket:
            raise NcError(f"create failed: {doc.get('error') or doc}")
        self.ledger.add(self.account, ticket, model=model, **({"run": run} if run else {}))
        cost_record("created", ticket, name=run, account=self.account, machine_type=model)
        return ticket

    def wait_active(self, ticket: str, *, timeout_s: float = 3600, poll_s: float = 20,
                    log: Callable[[str], None] = print) -> NodeLease:
        deadline = time.monotonic() + timeout_s
        while True:
            t = self.status().get("ticket") or {}
            if t.get("id") != ticket:
                raise NcError(f"account {self.account}: current ticket is {t.get('id')}, expected {ticket}")
            state = t.get("state")
            log(f"ticket {ticket} state={state} position={t.get('position')}")
            if state == "active" and (t.get("node") or {}).get("public_ip"):
                node = t["node"]
                lease = NodeLease(self.account, ticket, node["public_ip"], node.get("ssh_user") or "ubuntu",
                                  node.get("host_key") or "", t.get("granted_at"), t.get("lease_until"),
                                  t.get("gpus"), t.get("rate_ucr_per_gpu_hour"))
                self.ledger.mark(ticket, ip=lease.ip, lease_until=lease.lease_until)
                cost_record("created", ticket, account=self.account, granted_at=lease.granted_at,
                            lease_until=lease.lease_until, gpus=lease.gpus,
                            rate_ucr_per_gpu_hour=lease.rate_ucr_per_gpu_hour)
                return lease
            if state not in OPEN_STATES:
                self.ledger.mark(ticket, ended=True, end_reason=t.get("end_reason"))
                cost_record("released", ticket, account=self.account, end_reason=t.get("end_reason"))
                raise NcError(f"ticket {ticket} ended before active (end_reason={t.get('end_reason')})")
            if time.monotonic() > deadline:
                raise NcError(f"ticket {ticket} not active after {timeout_s:.0f}s (state {state})")
            self.sleep(poll_s)

    def delete(self, ticket: str) -> dict[str, Any]:
        """Release ``ticket``: only one this client created, only while it is the
        account's current ticket, then confirm it reads ``ended`` (recorded in the
        run's cost ledger either way)."""
        try:
            result = self._delete(ticket)
        except NcRefused:
            raise
        except Exception as error:
            cost_record("release_failed", ticket, error=repr(error)[:300])
            raise
        cost_record("released", ticket, account=self.account, end_reason=result.get("end_reason"),
                    ended_at=result.get("ended_at"))
        if not result.get("already"):  # this tool deleted it (a later pool instance must not call it external)
            try:
                self.ledger.mark(ticket, released_by="yeto", released_at=time.time())
            except Exception:  # noqa: BLE001 - bookkeeping only
                pass
        return result

    def _delete(self, ticket: str) -> dict[str, Any]:
        if not self.ledger.owns(self.account, ticket):
            raise NcRefused(f"ticket {ticket} on account {self.account} was not created by this tool")
        st = self.status()
        current = st.get("ticket") or {}
        if current.get("id") != ticket:
            if current.get("id") is None or current.get("state") not in OPEN_STATES:
                self.ledger.mark(ticket, ended=True)
                return {"ticket": ticket, "state": "ended", "already": True}
            raise NcRefused(f"account {self.account} current ticket is {current.get('id')}, "
                            f"not {ticket}: not deleting")
        if current.get("state") not in OPEN_STATES:
            self.ledger.mark(ticket, ended=True)
            return {"ticket": ticket, "state": current.get("state"), "already": True}
        doc = self._call([*_tool("manage"), "delete", "--account", str(self.account), "--yes"])
        for _ in range(12):
            t = self.status().get("ticket") or {}
            if t.get("id") in (None, ticket) and t.get("state") not in OPEN_STATES:
                self.ledger.mark(ticket, ended=True, ended_at=t.get("ended_at"), end_reason=t.get("end_reason"))
                return {"ticket": ticket, "state": t.get("state", "ended"), "ended_at": t.get("ended_at"),
                        "end_reason": t.get("end_reason"), "released": doc.get("released")}
            self.sleep(5)
        raise NcError(f"ticket {ticket}: delete sent but the ticket does not read ended")


def hourly_cost_ucr(gpus: int = 8, rate: int = MI355X_RATE_UCR_PER_GPU_HOUR) -> int:
    return gpus * rate
