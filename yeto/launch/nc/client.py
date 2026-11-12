"""The National Compute order tool (``nc-manage`` / ``nc-query``) as a guarded client.

Rules this client enforces (the accounts are shared with production):

* an order is placed only when the account has no open ticket of this session
  (``others[]`` - tickets of other sessions - is ignored, never acted on);
* every ticket this client creates is recorded in the ledger
  (``<state_dir>/tickets.json``): the order before it is sent, the ticket id from
  the order's own answer. A ticket is this tool's only on that evidence, or when
  an operator adopted it by hand (``adopt``). A ticket of unknown origin is only
  read and reported: never recorded, deleted or reached. ``delete`` refuses a ticket that is not in the
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
import socket
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
ORDER_LOOK_TIMEOUT_S = 30.0  # the one status read after an order call that failed (it only reports)
ORDER_CLOSE_FREE_READS = 2  # status reads in a row showing the account free that close a pending order
ORDER_PENDING_MAX_S = 1800.0  # a ticket first seen later than this after the order cannot be tied to it
ADOPT_CLOCK_SLACK_S = 300.0  # a ticket granted this long before the order was sent is not that order
ADOPT_QUEUE_MAX_S = 4 * 3600.0  # nor one granted later than this after the order call ended
DELETE_ABSENT_READS = 2  # "no order" answers in a row after a delete that leave its target unknown
DELETE_GUARD_S = 120.0  # no delete this close to the ticket's lease end or preemption (YETO_NC_DELETE_GUARD_S)
PENDING_ONLY_FIELDS = ("before", "since", "observed_ticket", "observed_at", "stale_reported", "free_reads",
                       "call_ended", "error")  # of a row without a ticket

class NcError(RuntimeError):
    pass


class LedgerError(NcError):
    """The ticket ledger exists and cannot be read (it is not taken for empty)."""


class NcOrderUnrecorded(NcError):
    """An order call failed and the account then shows an open ticket that no ledger
    row holds: maybe that order, maybe not. Nothing was recorded; an operator decides."""


class NcReleaseDeferred(NcError):
    """The ticket is about to end on its own (lease end, preemption): no delete is sent
    now, because a delete releases whatever ticket the account holds at that moment."""


class NcForeignTicketReleased(NcError):
    """A delete of this tool released a ticket that is not its own. The account is halted."""


class NcDeleteUnconfirmed(NcError):
    """A delete of this tool was sent and what it released cannot be established: its
    ticket is no longer the account's and never read ended. The account is halted."""


class NcRefused(NcError):
    """A guard refused the action; nothing was sent to National Compute."""


class NcHalted(NcRefused):
    """The account is halted in the ledger: no order, delete or adopt until an operator clears it."""


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

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time):
        self.path = path
        self.clock = clock

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
            rows.append({"account": account, "ticket": ticket, "created_at": self.clock(), **extra})
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

    def legacy_rows(self) -> list[dict[str, Any]]:
        """Open rows written before orders were recorded with their order (no ``order``
        field, not adopted): still treated as this tool's, but with weaker evidence -
        the old code could record a ticket the order tool merely found on the account."""
        return [r for r in self.open_rows() if "order" not in r and "adopted_by" not in r]

    # An order is written down before it is sent: a row without a ticket (a pending
    # order). The order's answer fills in the ticket. A row that stays without one is
    # an order whose outcome this tool never learned (the call failed or timed out, or
    # the process died inside it). It is closed when the account reads free twice in a
    # row after the call, or when a later order on the account succeeds; only an
    # operator (``adopt``) ties a ticket to it.

    @staticmethod
    def window_end(row: dict[str, Any]) -> float:
        """When the order call of a pending row was over at the latest."""
        return row.get("call_ended") or row["created_at"] + ORDER_IN_FLIGHT_S

    def in_flight(self, row: dict[str, Any], now: float | None = None) -> bool:
        """The order call of a pending row may still be running (in another process)."""
        now = self.clock() if now is None else now
        return row.get("ticket") is None and "order" in row and not row.get("call_ended") \
            and now - row["created_at"] < ORDER_IN_FLIGHT_S

    def _pending(self, rows: list[dict[str, Any]], account: int) -> list[dict[str, Any]]:
        return [r for r in rows if r.get("ticket") is None and "order" in r and r["account"] == account]

    def pending_orders(self, account: int, *, in_flight: bool = True) -> list[dict[str, Any]]:
        """The account's orders without a recorded outcome, oldest first
        (``in_flight`` False: without those whose call may still be running)."""
        return [r for r in self._pending(self.load(), account) if in_flight or not self.in_flight(r)]

    def begin_order(self, account: int, *, before: str | None, **extra: Any) -> str:
        """Write down an order about to be sent on ``account`` (``before``: the id of
        the ended ticket the account shows, if any; ``owner``: the process). The
        account keeps one pending row: an earlier order that is not closed yet is
        folded into the new row (``since``: when the earliest of them was sent). An
        order whose call may still be running (another process on this state dir)
        refuses a second one."""
        order = uuid.uuid4().hex
        with state_file_lock(self.path):
            rows, now = self.load(), self.clock()
            earlier = self._pending(rows, account)
            for r in earlier:
                if self.in_flight(r, now):
                    raise NcRefused(f"account {account}: an order of this tool is in flight "
                                    f"(sent {now - r['created_at']:.0f}s ago); one order per account")
            carried = {"since": min(r.get("since", r["created_at"]) for r in earlier),
                       **{k: earlier[-1][k] for k in ("observed_ticket", "observed_at") if k in earlier[-1]}
                       } if earlier else {}
            rows = [r for r in rows if r not in earlier]
            rows.append({"account": account, "ticket": None, "order": order, "created_at": now,
                         "before": before, **carried, **extra})
            self._write(rows)
        return order

    def end_order(self, order: str, ticket: str | None, *, drop: bool = False,
                  lost_row: dict[str, Any] | None = None, **fields: Any) -> None:
        """The order call is over. ``ticket``: the id in its answer (the row becomes
        that ticket's row). None: it failed, its outcome is unknown (the row stays
        pending, no longer in flight). ``drop``: no order was sent, forget the row."""
        with state_file_lock(self.path):
            rows = self.load()
            row = next((r for r in rows if r.get("order") == order and r.get("ticket") is None), None)
            if row is None and ticket is not None and lost_row is not None:
                row = {**lost_row, "ticket": None, "order": order, "created_at": self.clock()}
                rows.append(row)  # its pending row was dropped meanwhile: the ticket is recorded all the same
            if row is None:
                return
            if drop and "since" not in row:
                rows.remove(row)
            elif ticket is None:  # (a row that carries an earlier unrecorded order is never dropped)
                row.update(call_ended=self.clock(), **fields)
            else:  # the account was free for this order: the earlier ones left nothing open
                for key in PENDING_ONLY_FIELDS:
                    row.pop(key, None)
                row.update(ticket=ticket, **fields)
            self._write(rows)

    def note_free(self, account: int) -> None:
        """A trusted status read showed the account without an open order. The
        ``ORDER_CLOSE_FREE_READS``-th such read in a row after the order call closes
        the account's pending order: it placed nothing that is open, and no ticket
        seen on the account later is tied to it."""
        if not self._pending(self.load(), account):
            return
        with state_file_lock(self.path):
            rows = self.load()
            for row in self._pending(rows, account):
                if self.in_flight(row):
                    continue
                row["free_reads"] = row.get("free_reads", 0) + 1
                if row["free_reads"] >= ORDER_CLOSE_FREE_READS:
                    rows.remove(row)
            self._write(rows)

    def mark_pending(self, account: int, **fields: Any) -> None:
        with state_file_lock(self.path):
            rows = self.load()
            for row in self._pending(rows, account):
                row.update(fields)
            self._write(rows)

    def adopt_order(self, account: int, ticket: str, **fields: Any) -> dict[str, Any]:
        """An operator's decision: the account's pending order placed ``ticket``, the
        ticket this tool saw and reported for it. That row becomes the ticket's row."""
        with state_file_lock(self.path):
            rows = self.load()
            pending = [r for r in self._pending(rows, account) if not self.in_flight(r)]
            if not pending or pending[-1].get("observed_ticket") != ticket:
                raise NcRefused(f"account {account}: the ledger holds no order of this tool that ticket "
                                f"{ticket} was reported for; not adopted")
            row = pending[-1]
            rows = [r for r in rows if r is row or r not in pending]
            for key in PENDING_ONLY_FIELDS:
                row.pop(key, None)
            row.update(ticket=ticket, **fields)
            self._write(rows)
            return dict(row)

    # A delete is written down before it is sent (a row with ``pending_delete``: the
    # ticket it is meant for). ``nc-manage delete`` names no ticket, so until the
    # account shows that ticket ended, what the delete released is not known. The
    # row stays, across processes and for any length of time, until that is settled
    # (:meth:`NcClient._judge_delete`); meanwhile the account takes no other write.

    def pending_delete(self, account: int) -> dict[str, Any] | None:
        return next((r for r in self.load() if r.get("pending_delete") and r["account"] == account), None)

    def begin_delete(self, account: int, ticket: str) -> None:
        with state_file_lock(self.path):
            rows = self.load()
            row = next((r for r in rows if r.get("pending_delete") and r["account"] == account), None)
            if row is not None and row["pending_delete"] != ticket:
                raise NcRefused(f"account {account}: a delete for ticket {row['pending_delete']} is not "
                                f"confirmed yet; no delete for {ticket}")
            if row is None:
                rows.append({"account": account, "ticket": None, "pending_delete": ticket,
                             "sent_at": self.clock(), "attempts": 1, "absent": 0})
            else:
                row.update(attempts=row.get("attempts", 1) + 1, last_sent_at=self.clock(), absent=0)
            self._write(rows)

    def update_delete(self, account: int, **fields: Any) -> None:
        with state_file_lock(self.path):
            rows = self.load()
            for row in rows:
                if row.get("pending_delete") and row["account"] == account:
                    row.update(fields)
            self._write(rows)

    def end_delete(self, account: int) -> None:
        with state_file_lock(self.path):
            rows = self.load()
            kept = [r for r in rows if not (r.get("pending_delete") and r["account"] == account)]
            if len(kept) != len(rows):
                self._write(kept)

    # An account is halted when a delete of this tool released a ticket that was not
    # its own, or when what a delete released cannot be established: every order,
    # delete and adopt on it is refused until an operator clears the halt
    # (``clear-halt``).

    def halted(self, account: int) -> dict[str, Any] | None:
        return next((r for r in self.load() if r.get("halt") and r["account"] == account), None)

    def halt(self, account: int, reason: str, **fields: Any) -> None:
        """Halt ``account`` (and close its pending delete: the halt carries it on)."""
        with state_file_lock(self.path):
            rows = self.load()
            kept = [r for r in rows if not (r.get("pending_delete") and r["account"] == account)]
            if not any(r.get("halt") and r["account"] == account for r in kept):
                kept.append({"account": account, "ticket": None, "halt": reason, "halted_at": self.clock(), **fields})
            self._write(kept)

    def clear_halt(self, account: int) -> bool:
        with state_file_lock(self.path):
            rows = self.load()
            kept = [r for r in rows if not (r.get("halt") and r["account"] == account)]
            if len(kept) != len(rows):
                self._write(kept)
            return len(kept) != len(rows)

    def run_rows(self, run: str) -> list[dict[str, Any]]:
        """Open tickets this tool created for ``run`` (the tag written at create)."""
        return [r for r in self.open_rows() if r.get("run") == run]

    def _write(self, rows: list[dict[str, Any]]) -> None:
        write_json_atomic(self.path, rows)


class NcClient:
    """One account. ``run`` is ``subprocess.run`` (tests pass a fake)."""

    def __init__(self, account: int, ledger: Ledger, *, run: Callable[..., Any] = subprocess.run,
                 sleep: Callable[[float], None] = time.sleep, allowed_accounts: set[int] | None = None,
                 event: Callable[..., Any] | None = None, delete_guard_s: float | None = None):
        if allowed_accounts is not None and account not in allowed_accounts:
            raise NcRefused(f"account {account} is not in the allowed accounts {sorted(allowed_accounts)}")
        self.account = account
        self.ledger = ledger
        self.run = run
        self.sleep = sleep
        self.event = event  # event(name, **detail): the pool's event log (None: a warning on stderr)
        self.clock = ledger.clock
        self.delete_guard_s = float(os.environ.get("YETO_NC_DELETE_GUARD_S", DELETE_GUARD_S)
                                    if delete_guard_s is None else delete_guard_s)

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

    def status(self, *, timeout: float = 300) -> dict[str, Any]:
        """The account's status, only when the answer can be acted on: the tool
        succeeded and the answer names the ticket of this session - ``"ticket":
        null`` (no order) or a ticket with a state. Anything else (no output, an
        error document, no ``ticket`` field, a ticket without a state) raises: it
        says nothing about the ticket, and is never read as "no open ticket"."""
        doc = self._call([*_tool("query"), "status", "--account", str(self.account)], timeout=timeout)
        ticket = doc.get("ticket")
        if doc.get("ok", True) is not True or "ticket" not in doc or not (
                ticket is None or (isinstance(ticket, dict) and isinstance(ticket.get("state"), str))):
            raise NcError(f"{_tool('query')[0]} status: account {self.account}: the answer does not report "
                          f"the ticket (keys {sorted(doc)[:8]}, error {str(doc.get('error'))[:200]!r})")
        self._judge_halt(ticket or {})
        self._judge_delete(ticket or {})
        self._note_order(doc)
        return doc

    def _judge_halt(self, t: dict[str, Any]) -> None:
        """The one way a halt ends without an operator: the account was halted because
        the target of a delete could not be established (``delete_unconfirmed``), and
        a trusted status answer now shows that very ticket, by id, ended. The delete
        (or the ticket's own end) is thereby accounted for. No timeout lifts a halt,
        and a halt for a foreign ticket that was released is never lifted here."""
        if not t.get("id") or t.get("state") in OPEN_STATES:
            return
        halt = self.ledger.halted(self.account)
        if halt is None or halt.get("kind") != "delete_unconfirmed" or halt.get("expected_ticket") != t["id"]:
            return
        self.ledger.clear_halt(self.account)
        self._emit("halt_cleared", account=self.account, ticket=t["id"], state=t.get("state"),
                   end_reason=t.get("end_reason"), halted_at=halt.get("halted_at"),
                   why=f"ticket {t['id']} now reads {t.get('state')}: the delete sent for it is accounted for")

    def _judge_delete(self, t: dict[str, Any]) -> None:
        """Every trusted status answer settles, if it can, a delete this tool sent
        and has not confirmed (``Ledger.pending_delete``, for ticket E):

        * E reads ended: the delete hit E, or E ended on its own while it was still
          the account's ticket. Confirmed; the record is cleared.
        * E reads open: nothing is known yet (the delete failed, or is in progress).
        * another ticket reads releasing / released: the delete took it. The account
          is halted, event ``foreign_ticket_released``, :class:`NcForeignTicketReleased`.
        * another ticket in any other state, or no order twice in a row: E is gone
          without ever reading ended, so what the delete released cannot be
          established. The account is halted, event ``delete_unconfirmed``,
          :class:`NcDeleteUnconfirmed`. An operator checks the console."""
        pending = self.ledger.pending_delete(self.account)
        if pending is None:
            return
        expected = pending["pending_delete"]
        if t.get("id") == expected:
            if t.get("state") not in OPEN_STATES:
                self.ledger.end_delete(self.account)
            elif pending.get("absent"):
                self.ledger.update_delete(self.account, absent=0)
            return
        if t.get("id") is None:
            absent = pending.get("absent", 0) + 1
            if absent < DELETE_ABSENT_READS:
                self.ledger.update_delete(self.account, absent=absent)
                return
            self._unconfirmed_delete(pending, "the account shows no order and the ticket never read ended")
        if self._released_like(t):
            self._foreign_release(expected, t["id"], f"it reads {t.get('state')} after a delete of this tool "
                                                     f"meant for {expected}")
        self._unconfirmed_delete(pending, f"the account now shows ticket {t['id']} ({t.get('state')}) and "
                                          f"{expected} never read ended")

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

    def _emit(self, name: str, **detail: Any) -> None:
        try:
            if self.event is not None:
                self.event(name, **detail)
                return
        except Exception as error:  # noqa: BLE001 - reporting never fails the call
            detail = dict(detail, event_error=repr(error)[:200])
        print(f"WARNING: NC {name}: " + " ".join(f"{k}={v}" for k, v in detail.items()), file=sys.stderr, flush=True)

    def _refuse_halted(self) -> None:
        halt = self.ledger.halted(self.account)
        if halt is not None:
            raise NcHalted(f"account {self.account} is halted ({halt['halt']}): no order, delete or adopt on it "
                           f"until an operator checked it and ran clear-halt --account {self.account} --yes")

    def _refuse_pending_delete(self, ticket: str | None = None) -> None:
        """No write on the account while a delete is unconfirmed (but the same delete again)."""
        pending = self.ledger.pending_delete(self.account)
        if pending is not None and pending["pending_delete"] != ticket:
            raise NcRefused(f"account {self.account}: a delete for ticket {pending['pending_delete']} is not "
                            "confirmed yet (the ticket has not read ended); no other write on the account")

    def unrecorded_order(self, status: dict[str, Any]) -> dict[str, Any] | None:
        """The open ticket of a status answer (from :meth:`status`) when it may be an
        order of this tool that was never recorded and can still be adopted: no
        ledger row holds it, and it was seen in time for the account's pending order
        (:meth:`_note_order`). It is left alone: not recorded, not deleted, not reached."""
        own = self.own_ticket(status)
        if not own or not own.get("id") or self.ledger.owns(self.account, own["id"]):
            return None
        pending = self.ledger.pending_orders(self.account, in_flight=False)
        return own if pending and pending[-1].get("observed_ticket") == own["id"] else None

    def _note_order(self, status: dict[str, Any]) -> None:
        """What every trusted status answer says about the account's pending order (an
        order of this tool without a recorded outcome, its call over).

        The account is free: counted; the second such answer in a row closes the order.
        The account shows an open ticket that no ledger row holds: it may be that
        order. Seen within ``ORDER_PENDING_MAX_S`` of the order it is noted and
        reported once (event ``order_unrecorded``, with the ``adopt`` command). Seen
        later it is reported once as ``stale`` and cannot be adopted."""
        pending = self.ledger.pending_orders(self.account, in_flight=False)
        if not pending:
            return
        own = self.own_ticket(status)
        if own is None:
            self.ledger.note_free(self.account)
            return
        row = pending[-1]
        if row.get("free_reads"):
            self.ledger.mark_pending(self.account, free_reads=0)
        if not own.get("id") or self.ledger.owns(self.account, own["id"]) or row.get("before") == own["id"] \
                or row.get("observed_ticket") == own["id"]:
            return
        now = self.clock()
        detail = {"account": self.account, "ticket": own["id"], "state": own.get("state"), "run": row.get("run"),
                  "order_sent_at": row.get("since", row["created_at"]), "order_ended_at": self.ledger.window_end(row)}
        if now <= self.ledger.window_end(row) + ORDER_PENDING_MAX_S:
            self.ledger.mark_pending(self.account, observed_ticket=own["id"], observed_at=now)
            self._emit("order_unrecorded", **detail,
                       action=f"check ticket {own['id']} in the National Compute console; if this tool's order "
                              f"placed it: adopt --account {self.account} --ticket {own['id']} --yes")
        elif row.get("stale_reported") != own["id"]:
            self.ledger.mark_pending(self.account, stale_reported=own["id"])
            self._emit("order_unrecorded", **detail, stale=True,
                       action=f"ticket {own['id']} was first seen {now - self.ledger.window_end(row):.0f}s after "
                              "an order of this tool with no recorded outcome: too late to tie it to that order; "
                              "it cannot be adopted. Check it in the National Compute console")

    def account_free(self) -> bool:
        """The account has no open order of this session (an unrecorded one is reported)."""
        return self.own_ticket(self.status()) is None

    def create(self, *, model: str = "mi355x", pubkey: str | None = None, run: str | None = None) -> str:
        """Order a node; the ticket id, recorded in the ledger. The order is written
        to the ledger before it is sent and the ticket id is taken from the order's
        own answer (``created: true``), nowhere else. When the call fails (a timeout,
        output that does not parse, an error) the order may still have been placed:
        the row stays pending, the account's status is read once, and an open ticket
        found there is reported (``order_unrecorded``) - never recorded by the tool."""
        self._refuse_halted()
        st = self.status()
        self._refuse_pending_delete()
        own = self.own_ticket(st)
        found = self.unrecorded_order(st)
        if own:
            hint = "" if found is None else (
                f". An order this tool sent on the account was not recorded: ticket {own.get('id')} may be that "
                "order, still billing and in no ledger; check it in the National Compute console, then adopt "
                "it (adopt --account N --ticket T --yes) or release it there")
            raise NcRefused(f"account {self.account} already has open ticket "
                            f"{own.get('id')} ({own.get('state')}); one order per account{hint}")
        before = (st.get("ticket") or {}).get("id")
        row = {"account": self.account, "model": model, **({"run": run} if run else {}),
               "owner": f"{socket.gethostname()}:{os.getpid()}"}
        order = self.ledger.begin_order(before=before, **row)
        env = {"NC_PUBKEY": pubkey} if pubkey else None
        try:
            doc = self._call([*_tool("manage"), "create", "--account", str(self.account),
                              "--model", model, "--yes"], env=env)
            ticket = (doc.get("ticket") or {}).get("id")
            if doc.get("created") is not False and not (doc.get("ok") is True and doc.get("created") is True
                                                        and ticket):
                raise NcError(f"create failed: the answer does not say an order was placed: "
                              f"{doc.get('error') or sorted(doc)}")
        except Exception as error:  # the outcome is unknown: the order may have been placed
            self.ledger.end_order(order, None, error=repr(error)[:300])
            found = self._look_after_failed_order()
            if found is not None:
                raise NcOrderUnrecorded(
                    f"account {self.account}: the order call failed ({repr(error)[:200]}) and the account now "
                    f"shows open ticket {found.get('id')} ({found.get('state')}), in no ledger: not recorded, "
                    "not deleted; no further order on this account until an operator adopts or releases it"
                ) from error
            raise
        if doc.get("created") is False:  # the tool found an open order and placed none
            self.ledger.end_order(order, None, drop=True)
            raise NcRefused(f"account {self.account}: the order was not placed, the account has ticket {ticket} "
                            f"({(doc.get('ticket') or {}).get('state')}) of someone else; not recording it")
        self.ledger.end_order(order, ticket, lost_row=row)
        cost_record("created", ticket, name=run, account=self.account, machine_type=model)
        return ticket

    def _look_after_failed_order(self) -> dict[str, Any] | None:
        """One status read after an order call that failed, to report a ticket it may
        have left (:meth:`unrecorded_order`). A status that cannot be read reports
        nothing now: the pending row stays and the next look at the account reports."""
        try:
            return self.unrecorded_order(self.status(timeout=ORDER_LOOK_TIMEOUT_S))
        except Exception:  # noqa: BLE001 - the order call's error is the one raised
            return None

    def adopt(self, ticket: str) -> dict[str, Any]:
        """An operator says the account's unrecorded order placed ``ticket``: record
        it (``adopted_by="operator"``), so this tool may use and release it. Refused
        unless the account's status shows ``ticket`` as its current, open ticket, the
        ledger holds a pending order of the account that this tool saw ``ticket`` for
        in time (:meth:`unrecorded_order`: not closed, not the ticket the account had
        before, first seen within ``ORDER_PENDING_MAX_S``), and the ticket's grant
        time, when it has one, fits the order: not before it was sent, not more than
        ``ADOPT_QUEUE_MAX_S`` after its call ended."""
        self._refuse_halted()
        st = self.status()
        self._refuse_pending_delete()
        own = self.own_ticket(st)
        if not own or own.get("id") != ticket:
            raise NcRefused(f"account {self.account}: {ticket} is not the account's current open ticket "
                            f"(it shows {(own or {}).get('id')}): not adopted")
        if self.ledger.owns(self.account, ticket):
            raise NcRefused(f"ticket {ticket} on account {self.account} is already in the ledger")
        if self.unrecorded_order(st) is None:
            raise NcRefused(f"account {self.account}: the ledger holds no order of this tool that ticket {ticket} "
                            "can be tied to (none without a recorded outcome, or it was closed, or the ticket "
                            "was first seen too long after it): not adopted")
        row = self.ledger.pending_orders(self.account, in_flight=False)[-1]
        sent, ended = row.get("since", row["created_at"]), self.ledger.window_end(row)
        granted = parse_time(own.get("granted_at"))
        if granted is not None and not sent - ADOPT_CLOCK_SLACK_S <= granted <= ended + ADOPT_QUEUE_MAX_S:
            raise NcRefused(f"ticket {ticket} on account {self.account} was granted at {own.get('granted_at')}, "
                            f"which does not fit the unrecorded order (sent "
                            f"{datetime.fromtimestamp(sent, UTC):%Y-%m-%dT%H:%M:%SZ}): not adopted")
        row = self.ledger.adopt_order(self.account, ticket, adopted_by="operator", adopted_at=self.clock())
        cost_record("created", ticket, name=row.get("run"), account=self.account, machine_type=row.get("model"),
                    adopted_by="operator")
        return row

    def wait_active(self, ticket: str, *, timeout_s: float = 3600, poll_s: float = 20,
                    log: Callable[[str], None] = print) -> NodeLease:
        if not self.ledger.owns(self.account, ticket):  # no address or host key of a ticket that is not ours
            raise NcRefused(f"ticket {ticket} on account {self.account} was not created by this tool")
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
        run's cost ledger either way).

        ``nc-manage delete`` takes no ticket id: it releases whatever ticket the
        account holds when it runs. So the status is read again right before the
        call (the delete is not sent unless it still shows ``ticket``, open), no
        delete is sent while the ticket is about to end on its own
        (:class:`NcReleaseDeferred`), and a delete that turns out to have released
        another ticket halts the account (:class:`NcForeignTicketReleased`)."""
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
                self.ledger.mark(ticket, released_by="yeto", released_at=self.clock())
            except Exception:  # noqa: BLE001 - bookkeeping only
                pass
        return result

    def _delete(self, ticket: str) -> dict[str, Any]:
        self._refuse_halted()
        if not self.ledger.owns(self.account, ticket):
            raise NcRefused(f"ticket {ticket} on account {self.account} was not created by this tool")
        self._refuse_pending_delete(ticket)
        current = self._current(ticket)  # (each status read also settles an earlier, unconfirmed delete)
        if current.get("id") != ticket:
            if current.get("id") is None or current.get("state") not in OPEN_STATES:
                self.ledger.mark(ticket, ended=True)
                return {"ticket": ticket, "state": "ended", "already": True}
            raise NcRefused(f"account {self.account} current ticket is {current.get('id')}, "
                            f"not {ticket}: not deleting")
        if current.get("state") not in OPEN_STATES:
            self.ledger.mark(ticket, ended=True)
            return {"ticket": ticket, "state": current.get("state"), "already": True}
        self._ending_soon(ticket, current)
        latest = self.status().get("ticket") or {}  # the last look before the call
        if latest.get("id") != ticket or latest.get("state") not in OPEN_STATES:
            raise NcError(f"ticket {ticket}: the account's ticket changed between two reads (now "
                          f"{latest.get('id')} {latest.get('state')}); no delete sent")
        self._ending_soon(ticket, latest)
        self.ledger.begin_delete(self.account, ticket)
        try:
            doc = self._call([*_tool("manage"), "delete", "--account", str(self.account), "--yes"])
            if doc.get("ok") is False:
                raise NcError(f"ticket {ticket}: delete failed: {doc.get('error') or doc}")
        except Exception:  # the delete may have taken effect all the same: look at once
            result = self._confirm_release(ticket, {}, reads=2)
            if result is not None:
                return result
            raise
        hit = (doc.get("ticket") or {}).get("id")
        if hit and hit != ticket:
            self._foreign_release(ticket, hit, "the delete answer names it")
        if hit == ticket:  # the answer names our ticket: the target is confirmed
            self.ledger.end_delete(self.account)
        result = self._confirm_release(ticket, doc, reads=12)
        if result is None:
            raise NcError(f"ticket {ticket}: delete sent but the ticket does not read ended")
        return result

    def _confirm_release(self, ticket: str, doc: dict[str, Any], *, reads: int) -> dict[str, Any] | None:
        """After a delete call: read the status until ``ticket`` reads ended (the
        result), up to ``reads`` times; None when it does not. Each read goes through
        :meth:`_judge_delete`, which halts the account when the delete took another
        ticket or its target cannot be established. A status that cannot be read is
        read again."""
        absent = 0
        for i in range(reads):
            if i:
                self.sleep(5)
            try:
                t = self.status().get("ticket") or {}
            except (NcForeignTicketReleased, NcDeleteUnconfirmed):
                raise
            except Exception:  # noqa: BLE001 - not an answer; the pending delete stays in the ledger
                continue
            if t.get("id") == ticket and t.get("state") in OPEN_STATES:
                continue
            target_known = self.ledger.pending_delete(self.account) is None
            absent = absent + 1 if t.get("id") is None else 0
            if t.get("id") == ticket or (target_known and (t.get("id") is not None or absent >= 2)):
                ended = t if t.get("id") == ticket else {"end_reason": "released"}
                self.ledger.mark(ticket, ended=True, ended_at=ended.get("ended_at"),
                                 end_reason=ended.get("end_reason"))
                return {"ticket": ticket, "state": ended.get("state", "ended"), "ended_at": ended.get("ended_at"),
                        "end_reason": ended.get("end_reason"), "released": doc.get("released")}
        return None

    @staticmethod
    def _released_like(t: dict[str, Any]) -> bool:
        return t.get("state") == "releasing" or (t.get("state") not in OPEN_STATES
                                                 and t.get("end_reason") == "released")

    def _ending_soon(self, ticket: str, current: dict[str, Any]) -> None:
        """No delete within ``delete_guard_s`` of the ticket's lease end or preemption:
        if it ends on its own and someone else orders before the delete lands, the
        delete releases their ticket. Later, a ticket that still reads open is deleted
        as usual (a lease does not end at ``lease_until`` by itself)."""
        now = self.clock()
        for name in ("preempt_at", "lease_until"):
            at = parse_time(current.get(name))
            if at is not None and abs(at - now) <= self.delete_guard_s:
                raise NcReleaseDeferred(f"ticket {ticket}: {name} {current.get(name)} is within "
                                        f"{self.delete_guard_s:.0f}s; no delete sent now, retry after it")

    def _foreign_release(self, ticket: str, other: str, why: str) -> None:
        reason = f"a delete meant for ticket {ticket} released ticket {other}"
        self.ledger.halt(self.account, reason, kind="foreign_ticket_released", expected_ticket=ticket,
                         released_ticket=other)
        self._emit("foreign_ticket_released", account=self.account, ticket=other, expected=ticket, reason=reason,
                   why=why, severity="critical",
                   action=f"tell the owner of ticket {other}; account {self.account} is halted for this tool "
                          f"until clear-halt --account {self.account} --yes")
        raise NcForeignTicketReleased(f"account {self.account}: {reason} ({why}); the account is halted")

    def _unconfirmed_delete(self, pending: dict[str, Any], why: str) -> None:
        ticket = pending["pending_delete"]
        reason = f"a delete meant for ticket {ticket} was sent and what it released is not known"
        self.ledger.halt(self.account, reason, kind="delete_unconfirmed", expected_ticket=ticket,
                         delete_sent_at=pending.get("sent_at"))
        self._emit("delete_unconfirmed", account=self.account, expected=ticket, reason=reason, why=why,
                   severity="critical",
                   delete_sent_at=pending.get("sent_at"),
                   action=f"check in the National Compute console which ticket of account {self.account} was "
                          f"released around that time; if it was not {ticket}, tell its owner. The account is "
                          f"halted for this tool until clear-halt --account {self.account} --yes")
        raise NcDeleteUnconfirmed(f"account {self.account}: {reason} ({why}); the account is halted")

def hourly_cost_ucr(gpus: int = 8, rate: int = MI355X_RATE_UCR_PER_GPU_HOUR) -> int:
    return gpus * rate
