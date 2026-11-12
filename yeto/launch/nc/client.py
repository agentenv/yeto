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
PENDING_ONLY_FIELDS = ("before", "since", "observed_ticket", "call_ended", "error")  # of a row without a ticket
ADOPT_CLOCK_SLACK_S = 300.0  # a ticket granted this long before the order was sent is not that order


class NcError(RuntimeError):
    pass


class LedgerError(NcError):
    """The ticket ledger exists and cannot be read (it is not taken for empty)."""


class NcOrderUnrecorded(NcError):
    """An order call failed and the account then shows an open ticket that no ledger
    row holds: maybe that order, maybe not. Nothing was recorded; an operator decides."""


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

    # An order is written down before it is sent: a row without a ticket. The order's
    # answer fills in the ticket. A row that stays without one is an order whose
    # outcome this tool never learned (the call failed or timed out, or the process
    # died inside it). Nothing fills it in later but an operator (``adopt``).

    @staticmethod
    def in_flight(row: dict[str, Any], now: float) -> bool:
        """The order call of a pending row may still be running (in another process)."""
        return row.get("ticket") is None and not row.get("call_ended") \
            and now - row["created_at"] < ORDER_IN_FLIGHT_S

    def pending_orders(self, account: int, *, in_flight: bool = True) -> list[dict[str, Any]]:
        """The account's orders without a recorded outcome, oldest first
        (``in_flight`` False: without those whose call may still be running)."""
        now = time.time()
        return [r for r in self.load() if r.get("ticket") is None and r["account"] == account
                and (in_flight or not self.in_flight(r, now))]

    def begin_order(self, account: int, *, before: str | None, **extra: Any) -> str:
        """Write down an order about to be sent on ``account`` (``before``: the id of
        the ended ticket the account shows, if any). The account keeps one pending
        row: an earlier order without a recorded outcome is folded into the new row
        (``since``: when the earliest of them was sent), never forgotten on the word
        of one status answer. An order whose call may still be running (another
        process on this state dir) refuses a second one."""
        order = uuid.uuid4().hex
        with state_file_lock(self.path):
            rows, now = self.load(), time.time()
            earlier = [r for r in rows if r.get("ticket") is None and r["account"] == account]
            for r in earlier:
                if self.in_flight(r, now):
                    raise NcRefused(f"account {account}: an order of this tool is in flight "
                                    f"(sent {now - r['created_at']:.0f}s ago); one order per account")
            carried = {"since": min(r.get("since", r["created_at"]) for r in earlier),
                       "observed_ticket": earlier[-1].get("observed_ticket")} if earlier else {}
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
                row = {**lost_row, "ticket": None, "order": order, "created_at": time.time()}
                rows.append(row)  # its pending row was dropped meanwhile: the ticket is recorded all the same
            if row is None:
                return
            if drop and "since" not in row:
                rows.remove(row)
            elif ticket is None:  # (a row that carries an earlier unrecorded order is never dropped)
                row.update(call_ended=time.time(), **fields)
            else:  # the account was free for this order: the earlier ones left nothing open
                for key in PENDING_ONLY_FIELDS:
                    row.pop(key, None)
                row.update(ticket=ticket, **fields)
            self._write(rows)

    def mark_pending(self, account: int, **fields: Any) -> None:
        with state_file_lock(self.path):
            rows = self.load()
            for row in rows:
                if row.get("ticket") is None and row["account"] == account:
                    row.update(fields)
            self._write(rows)

    def adopt_order(self, account: int, ticket: str, **fields: Any) -> dict[str, Any]:
        """An operator's decision: the account's latest pending order placed
        ``ticket``. That row becomes the ticket's row; the account's other pending
        rows are dropped. Refused without a pending order of the account."""
        with state_file_lock(self.path):
            rows, now = self.load(), time.time()
            pending = [r for r in rows if r.get("ticket") is None and r["account"] == account
                       and not self.in_flight(r, now)]
            if not pending:
                raise NcRefused(f"account {account}: the ledger holds no order of this tool without a "
                                f"recorded outcome; ticket {ticket} is not adopted")
            row = pending[-1]
            rows = [r for r in rows if r is row or r not in pending]
            for key in PENDING_ONLY_FIELDS:
                row.pop(key, None)
            row.update(ticket=ticket, **fields)
            self._write(rows)
            return dict(row)

    def run_rows(self, run: str) -> list[dict[str, Any]]:
        """Open tickets this tool created for ``run`` (the tag written at create)."""
        return [r for r in self.open_rows() if r.get("run") == run]

    def _write(self, rows: list[dict[str, Any]]) -> None:
        write_json_atomic(self.path, rows)


class NcClient:
    """One account. ``run`` is ``subprocess.run`` (tests pass a fake)."""

    def __init__(self, account: int, ledger: Ledger, *, run: Callable[..., Any] = subprocess.run,
                 sleep: Callable[[float], None] = time.sleep, allowed_accounts: set[int] | None = None,
                 event: Callable[..., Any] | None = None):
        if allowed_accounts is not None and account not in allowed_accounts:
            raise NcRefused(f"account {account} is not in the allowed accounts {sorted(allowed_accounts)}")
        self.account = account
        self.ledger = ledger
        self.run = run
        self.sleep = sleep
        self.event = event  # event(name, **detail): the pool's event log (None: a warning on stderr)

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

    def _emit(self, name: str, **detail: Any) -> None:
        try:
            if self.event is not None:
                self.event(name, **detail)
                return
        except Exception as error:  # noqa: BLE001 - reporting never fails the call
            detail = dict(detail, event_error=repr(error)[:200])
        print(f"WARNING: NC {name}: " + " ".join(f"{k}={v}" for k, v in detail.items()), file=sys.stderr, flush=True)

    def unrecorded_order(self, status: dict[str, Any]) -> dict[str, Any] | None:
        """The account's open ticket when it may be an order of this tool that was
        never recorded: no ledger row holds it, and the ledger has an order of the
        account without a recorded outcome (not one still in flight). It is reported
        once per ticket (event ``order_unrecorded``) and otherwise left alone: not
        recorded, not deleted. An operator checks it and runs ``adopt`` or releases
        it in the console."""
        own = self.own_ticket(status)
        if not own or not own.get("id") or self.ledger.owns(self.account, own["id"]):
            return None
        pending = [r for r in self.ledger.pending_orders(self.account, in_flight=False)
                   if r.get("before") != own["id"]]
        if not pending:
            return None
        if pending[-1].get("observed_ticket") != own["id"]:
            self.ledger.mark_pending(self.account, observed_ticket=own["id"])
            self._emit("order_unrecorded", account=self.account, ticket=own["id"], state=own.get("state"),
                       order_sent_at=pending[-1].get("since", pending[-1]["created_at"]),
                       run=pending[-1].get("run"),
                       action=f"check ticket {own['id']} in the National Compute console; if this tool's order "
                              f"placed it: adopt --account {self.account} --ticket {own['id']} --yes")
        return own

    def account_free(self) -> bool:
        """The account has no open order of this session (an unrecorded one is reported)."""
        st = self.status()
        self.unrecorded_order(st)
        return self.own_ticket(st) is None

    def create(self, *, model: str = "mi355x", pubkey: str | None = None, run: str | None = None) -> str:
        """Order a node; the ticket id, recorded in the ledger. The order is written
        to the ledger before it is sent and the ticket id is taken from the order's
        own answer, nowhere else. When the call fails (a timeout, output that does
        not parse, an error) the order may still have been placed: the row stays
        pending, the account's status is read once, and an open ticket found there is
        reported (``order_unrecorded``) - it is never recorded by this process."""
        st = self.status()
        own = self.own_ticket(st)
        if own:
            hint = "" if self.unrecorded_order(st) is None else (
                f". An order this tool sent on the account was not recorded: ticket {own.get('id')} may be that "
                "order, still billing and in no ledger; check it in the National Compute console, then adopt "
                "it (adopt --account N --ticket T --yes) or release it there")
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
            if doc.get("created") is not False and (not doc.get("ok") or not ticket):
                raise NcError(f"create failed: {doc.get('error') or doc}")
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
        it (``adopted_by="operator"``), so this tool may release it. Refused unless
        the ledger holds an order of the account without a recorded outcome, and the
        account's status shows ``ticket`` as its current, open ticket - not the
        ticket the account had before that order, and not one granted before the
        order was sent."""
        own = self.own_ticket(self.status())
        if not own or own.get("id") != ticket:
            raise NcRefused(f"account {self.account}: {ticket} is not the account's current open ticket "
                            f"(it shows {(own or {}).get('id')}): not adopted")
        if self.ledger.owns(self.account, ticket):
            raise NcRefused(f"ticket {ticket} on account {self.account} is already in the ledger")
        pending = self.ledger.pending_orders(self.account, in_flight=False)
        if not pending:
            raise NcRefused(f"account {self.account}: the ledger holds no order of this tool without a "
                            f"recorded outcome; ticket {ticket} is not adopted")
        sent = pending[-1].get("since", pending[-1]["created_at"])
        granted = parse_time(own.get("granted_at"))
        if pending[-1].get("before") == ticket or (granted is not None and granted < sent - ADOPT_CLOCK_SLACK_S):
            raise NcRefused(f"ticket {ticket} on account {self.account} is older than the unrecorded order "
                            f"(sent {datetime.fromtimestamp(sent, UTC):%Y-%m-%dT%H:%M:%SZ}): not adopted")
        row = self.ledger.adopt_order(self.account, ticket, adopted_by="operator", adopted_at=time.time())
        cost_record("created", ticket, name=row.get("run"), account=self.account, machine_type=row.get("model"),
                    adopted_by="operator")
        return row

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
