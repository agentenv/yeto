"""The National Compute order tool (``nc-manage`` / ``nc-query``) as a guarded client.

Rules this client enforces (the accounts are shared with production):

* an order is placed only when the account has no open ticket of this session
  (``others[]`` - tickets of other sessions - is ignored, never acted on);
* every ticket this client creates is recorded in the ledger
  (``<state_dir>/tickets.json``), the order itself before it is sent; ``delete`` refuses a ticket that is not in the
  ledger, and refuses unless the account's current ticket id equals the
  expected id right before the call;
* ``delete --all`` is never issued; only ``nc-manage delete --account N --yes``,
  which releases the current ticket of this session.

The tools read their own account file; no account secret passes through here.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..pool.status import LeaseStatus

OPEN_STATES = frozenset({"queued", "held", "provisioning", "active", "releasing"})
MI355X_RATE_UCR_PER_GPU_HOUR = 3_000_000
LEASE_HOURS = 4.0
ORDER_IN_FLIGHT_S = 360.0  # an order call (300 s timeout) may still run this long after it was written down
ORDER_CONFIRM_READS, ORDER_CONFIRM_POLL_S = 3, 5.0  # status reads after an order call that failed


class NcError(RuntimeError):
    pass


class LedgerError(NcError):
    """The ticket ledger exists and cannot be read (it is not taken for empty)."""


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


_FILE_LOCKS: dict[str, threading.RLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def state_file_lock(path: Path) -> Iterator[None]:
    """Exclusive hold on a state file for one read-modify-write: a lock per path
    for the threads of this process (the pool reads tickets from several), and an
    ``flock`` on ``<name>.lock`` next to it for other processes on the same state
    dir (a manual command while a pool runs). The lock file is never replaced."""
    key = os.path.abspath(path)
    with _FILE_LOCKS_GUARD:
        thread_lock = _FILE_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing the descriptor drops the flock


def write_json_atomic(path: Path, doc: Any) -> None:
    """Replace ``path`` with ``doc``: a temporary file of its own (0600, same
    directory), synced, then renamed. A reader sees the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class Ledger:
    """Tickets created by this tool (append-only file, 0600). Every change is one
    read-modify-write under :func:`state_file_lock`. A ledger that exists and cannot
    be read is an error, never an empty ledger: the next write would replace every
    ticket this tool owns with one row."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            return []
        except OSError as error:
            raise LedgerError(f"ticket ledger {self.path}: cannot be read: {error!r}") from error
        try:
            rows = json.loads(text)
        except ValueError as error:
            raise LedgerError(f"ticket ledger {self.path}: not JSON ({error}); repair or move it aside "
                              "(its tickets may still be billing)") from error
        if not isinstance(rows, list) or not all(isinstance(r, dict) and "ticket" in r for r in rows):
            raise LedgerError(f"ticket ledger {self.path}: not a list of ticket rows")
        return rows

    def add(self, account: int, ticket: str, **extra: Any) -> None:
        with state_file_lock(self.path):
            rows = self.load()
            rows.append({"account": account, "ticket": ticket, "created_at": time.time(), **extra})
            self._write(rows)

    def mark(self, ticket: str, **fields: Any) -> None:
        with state_file_lock(self.path):
            rows = self.load()
            for row in rows:
                if row["ticket"] == ticket:
                    row.update(fields)
            self._write(rows)

    def owns(self, account: int, ticket: str) -> bool:
        return any(r["account"] == account and r["ticket"] == ticket for r in self.load())

    def open_rows(self) -> list[dict[str, Any]]:
        return [r for r in self.load() if r.get("ticket") and not r.get("ended")]

    # An order is written down before it is sent: a row without a ticket. The order
    # call fills in the ticket; a row that stays without one is an order whose
    # outcome this tool never learned (the process died inside the call, or the
    # call failed and the status after it could not be read).

    def pending_orders(self, account: int) -> list[dict[str, Any]]:
        return [r for r in self.load() if r.get("ticket") is None and r["account"] == account]

    def begin_order(self, account: int, *, before: str | None, **extra: Any) -> str:
        """Write down an order about to be sent on ``account`` (``before``: the id of
        the ended ticket the account shows, if any). The caller has just read that
        the account has no open order, so earlier pending orders of the account are
        over and dropped - except one whose call may still be running (another
        process on this state dir): two orders on one account are refused."""
        order = uuid.uuid4().hex
        with state_file_lock(self.path):
            rows, now = self.load(), time.time()
            for r in rows:
                if r.get("ticket") is None and r["account"] == account and not r.get("call_ended") \
                        and now - r["created_at"] < ORDER_IN_FLIGHT_S:
                    raise NcRefused(f"account {account}: an order of this tool is in flight "
                                    f"(sent {now - r['created_at']:.0f}s ago); one order per account")
            rows = [r for r in rows if not (r.get("ticket") is None and r["account"] == account)]
            rows.append({"account": account, "ticket": None, "order": order, "created_at": now,
                         "before": before, **extra})
            self._write(rows)
        return order

    def end_order(self, order: str, ticket: str | None, *, drop: bool = False,
                  lost_row: dict[str, Any] | None = None, **fields: Any) -> None:
        """The order call is over: ``ticket`` is the ticket it placed (the row becomes
        that ticket's row), or None when it failed and no ticket of it is known (the
        row stays pending, no longer in flight; ``drop``: no order was sent, forget it)."""
        with state_file_lock(self.path):
            rows = self.load()
            row = next((r for r in rows if r.get("order") == order and r.get("ticket") is None), None)
            if row is None and ticket is not None and lost_row is not None:
                row = {**lost_row, "ticket": None, "order": order, "created_at": time.time()}
                rows.append(row)  # its pending row was dropped meanwhile: the ticket is recorded all the same
            if row is None:
                return
            if drop:
                rows.remove(row)
            elif ticket is None:
                row.update(call_ended=time.time(), **fields)
            else:
                row.pop("before", None)
                row.update(ticket=ticket, **fields)
            self._write(rows)

    def mark_pending(self, account: int, **fields: Any) -> None:
        with state_file_lock(self.path):
            rows = self.load()
            for row in rows:
                if row.get("ticket") is None and row["account"] == account:
                    row.update(fields)
            self._write(rows)

    def run_rows(self, run: str) -> list[dict[str, Any]]:
        """Open tickets this tool created for ``run`` (the tag written at create)."""
        return [r for r in self.open_rows() if r.get("run") == run]

    def _write(self, rows: list[dict[str, Any]]) -> None:
        write_json_atomic(self.path, rows)


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
        if out.returncode != 0:  # a failed call decides nothing, whatever it printed
            detail = doc.get("error") or (out.stderr or "")[-300:]
            raise NcError(f"{argv[0]} exit {out.returncode}: {detail}")
        return doc

    def status(self) -> dict[str, Any]:
        """The account's status, only when the answer can be acted on: the tool
        succeeded and the answer names the ticket of this session - ``"ticket":
        null`` (no order) or a ticket with a state. Anything else (no output, an
        error document, no ``ticket`` field, a ticket without a state) raises: it
        says nothing about the ticket, and is never read as "no open ticket"."""
        doc = self._call([*_tool("query"), "status", "--account", str(self.account)])
        ticket = doc.get("ticket")
        if doc.get("ok", True) is not True or "ticket" not in doc or not (
                ticket is None or (isinstance(ticket, dict) and isinstance(ticket.get("state"), str))):
            raise NcError(f"{_tool('query')[0]} status: account {self.account}: the answer does not report "
                          f"the ticket (keys {sorted(doc)[:8]}, error {str(doc.get('error'))[:200]!r})")
        return doc

    def _current(self, expect: str) -> dict[str, Any]:
        """The account's current ticket ({}: none). An answer that does not show
        ``expect`` is read once more before anyone concludes that ``expect`` is
        gone: one wrong "no order" answer must not retire a node or close a ticket
        in the ledger."""
        for _ in range(2):
            current = self.status().get("ticket") or {}
            if current.get("id") == expect:
                break
        return current

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
        ended (one open order per account), once two answers in a row say so. A failed
        call, or an answer that does not report the ticket, raises (the pool reads
        that as a control-plane blip and decides nothing on it)."""
        from ..pool.status import LeaseStatus  # noqa: PLC0415

        t = self._current(ticket)
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
        """Order a node; the ticket id, recorded in the ledger. The order is written
        to the ledger before it is sent, so an order whose answer is lost (a timeout,
        output that does not parse, the tool failing after the request) is still
        found: the account's status is read right after, and an open ticket that was
        not there before the order is the one this call placed."""
        st = self.status()
        own = self.own_ticket(st)
        if own:
            pending = self.ledger.pending_orders(self.account)
            sent = datetime.fromtimestamp(pending[-1]["created_at"], UTC) if pending else None
            hint = "" if sent is None or self.ledger.owns(self.account, own.get("id")) else (
                f". An order this tool sent at {sent:%Y-%m-%dT%H:%MZ} was not recorded: ticket {own.get('id')} "
                "may be that order, still billing and in no ledger; check it in the National Compute console "
                "and release it there if no one else placed it")
            raise NcRefused(f"account {self.account} already has open ticket "
                            f"{own.get('id')} ({own.get('state')}); one order per account{hint}")
        before = (st.get("ticket") or {}).get("id")
        row = {"account": self.account, "model": model, **({"run": run} if run else {})}
        order = self.ledger.begin_order(before=before, **row)
        env = {"NC_PUBKEY": pubkey} if pubkey else None
        try:
            doc = self._call([*_tool("manage"), "create", "--account", str(self.account),
                              "--model", model, "--yes"], env=env)
            ticket = (doc.get("ticket") or {}).get("id")
            if doc.get("created") is False and ticket:  # the tool found an open order and placed none
                self.ledger.end_order(order, None, drop=True)  # no order was sent
                raise NcRefused(f"account {self.account}: the order was not placed, ticket {ticket} "
                                f"({doc['ticket'].get('state')}) was opened by someone else; not recording it")
            if not doc.get("ok") or not ticket:
                raise NcError(f"create failed: {doc.get('error') or doc}")
            fields: dict[str, Any] = {}
        except NcRefused:
            raise
        except Exception as error:  # the answer is lost; the order may have been placed
            ticket = self._placed_order(before)
            if ticket is None:
                self.ledger.end_order(order, None, error=repr(error)[:300])
                raise
            print(f"WARNING: NC account {self.account}: the order call failed ({error!r}) but placed ticket "
                  f"{ticket}; recorded in the ledger", file=sys.stderr, flush=True)
            fields = {"recorded_after": repr(error)[:300]}
        self.ledger.end_order(order, ticket, lost_row=row, **fields)
        cost_record("created", ticket, name=run, account=self.account, machine_type=model)
        return ticket

    def _placed_order(self, before: str | None) -> str | None:
        """After an order call that failed: the open ticket of this session that was
        not the account's ticket before the order (``before``), or None. Only called
        by the process that sent the order, right after its call, having read before
        it that the account had no open order; the order tool holds the account's
        lock during the call. A status that cannot be read decides nothing."""
        for attempt in range(ORDER_CONFIRM_READS):
            if attempt:
                self.sleep(ORDER_CONFIRM_POLL_S)
            try:
                own = self.own_ticket(self.status())
            except Exception:  # noqa: BLE001 - the order call's error is the one raised
                continue
            if own and own.get("id") and own["id"] != before:
                return str(own["id"])
        return None

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
        current = self._current(ticket)
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
        if doc.get("ok") is False:
            raise NcError(f"ticket {ticket}: delete failed: {doc.get('error') or doc}")
        absent = 0  # "no order" answers in a row: one alone does not confirm the release
        for _ in range(12):
            t = self.status().get("ticket") or {}
            absent = absent + 1 if t.get("id") is None else 0
            if (t.get("id") == ticket or absent >= 2) and t.get("state") not in OPEN_STATES:
                self.ledger.mark(ticket, ended=True, ended_at=t.get("ended_at"), end_reason=t.get("end_reason"))
                return {"ticket": ticket, "state": t.get("state", "ended"), "ended_at": t.get("ended_at"),
                        "end_reason": t.get("end_reason"), "released": doc.get("released")}
            self.sleep(5)
        raise NcError(f"ticket {ticket}: delete sent but the ticket does not read ended")


def hourly_cost_ucr(gpus: int = 8, rate: int = MI355X_RATE_UCR_PER_GPU_HOUR) -> int:
    return gpus * rate
