"""Node leases across runs: one run at a time on a node of an SSH inventory.

An inventory may name a ``lease_store`` at its top (:mod:`.node_inventory`): a store
every launcher of the site reaches (``s3://`` / ``oss://``, or a shared directory).
A run that starts something on a node of that inventory (an ``ssh:`` trainer island,
a ``kind: ssh`` sampler pool) first takes the node's lease there; a node another run
holds is skipped, and a run that finds no free node reports who holds them and waits.
Without ``lease_store`` nothing is leased and nothing here runs (as before).

A lease is the fleet control plane's store lock (:class:`~.pool.store_lock.StoreLock`)
under ``<lease_store>/nodes/<node>/lease``: a token created once (S3 / R2
``If-None-Match: *``, a hard link on a directory) names the holding run, and the
token's lease file carries its end (``ttl_s``, 10 minutes by default), which the
holder moves forward on its supervision tick (the sampler pool's member reads, the
trainer island's keeper thread). Two runs that claim one node at once create the same
token: one wins. A holder that vanished lets its lease run out; the next run then
claims the next token (which fences the old holder for good) and writes a takeover
event under ``nodes/<node>/events/``. A run that restarts (a new launcher of the same
run id) takes its own running lease back at once (``StoreLock.acquire(reclaim=True)``).

The holder is the run id and the part of the run: ``<run>/trainer`` for the run's
``ssh:`` trainer islands (one launcher; :func:`~.ssh_trainer_island.inventory_trainer_nodes`
keeps its islands on distinct nodes), ``<pool run>/pool`` for a sampler pool. A store
without an atomic create is refused (:class:`NodeLeaseError`).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .pool.store_lock import LockLost, StoreLock, StoreLockError

DEFAULT_TTL_S = 600.0
WAIT_POLL_S = 30.0  # a run that waits for a free node reads the leases this often
EVENT_SCHEMA = "yeto.node-lease.event/v1"
TAKEOVER_EVENT = "node_lease_taken_over"
LOST_EVENT = "node_lease_lost"
NO_LEASE_HINT = ("the node inventory declares no lease_store: its nodes are not leased across runs, so two runs "
                 "on one inventory may start on the same node (set lease_store: <store URI> at the top of the "
                 "inventory)")

_LOG = logging.getLogger(__name__)
_hinted = False


class NodeLeaseError(RuntimeError):
    """The leases cannot be used (a store without an atomic create, an unreadable store)."""


class NodeLeased(RuntimeError):
    """A node (or every free node) is held by another run; nothing was started on it.

    The sampler pool counts it as a refused order (``island_pool.REFUSED_ERRORS``: the
    slot is left out for a while, the next one is tried)."""

    def __init__(self, message: str, holders: Mapping[str, str | None] | None = None) -> None:
        super().__init__(message)
        self.holders = dict(holders or {})


@dataclass(frozen=True)
class LeaseResult:
    """One :meth:`NodeLeases.acquire`: ``ok`` (this run holds the node now), ``holder``
    (the run that holds it: this run when ``ok``), ``took_over`` (the claim ended another
    run's lease that had run out; ``previous`` names that run)."""

    ok: bool
    holder: str | None
    took_over: bool = False
    previous: str | None = None


def node_name(node: Any) -> str:
    return node if isinstance(node, str) else str(node.name)


def open_objects(uri: str, environ: Mapping[str, str] | None = None) -> Any:
    """The object API of a lease store: a directory (an absolute path or ``file://``),
    else the head's ``s3://`` / ``oss://`` client (the launcher's store credential)."""
    if uri.startswith("file://") or uri.startswith("/"):
        from yeto.rl.engine.object_store import DirectoryObjectStore  # noqa: PLC0415

        return DirectoryObjectStore(uri[len("file://"):] if uri.startswith("file://") else uri)
    from .trainer_island import head_store_objects  # noqa: PLC0415

    return head_store_objects(uri, environ)


def no_lease_hint(*, emit: Callable[[str], None] | None = None) -> str | None:
    """The one hint of an inventory without ``lease_store``, once per process: through
    ``emit`` (a dry-run's note), or to the log only (``emit`` None: a dry-run whose output
    is a customer golden, the trainer's ``run up`` / ``launch --dry-run``). The log line is
    a DEBUG record: the launcher logs at INFO into the customer's preflight log, which
    stays as before. Returns the hint the first time, then None."""
    global _hinted
    if _hinted:
        return None
    _hinted = True
    if emit is None:
        _LOG.debug(NO_LEASE_HINT)
    else:
        emit(NO_LEASE_HINT)
    return NO_LEASE_HINT


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class NodeLeases:
    """The node leases of run ``run_id`` in the lease store ``store_uri`` (module doc).

    ``objects``: the store's object API (tests pass a fake; default :func:`open_objects`,
    opened on first use). ``on_event(name, **detail)``: where a takeover or a lost lease
    is reported besides the store (default: a line on stderr)."""

    def __init__(self, store_uri: str, run_id: str, ttl_s: float = DEFAULT_TTL_S, *, objects: Any = None,
                 clock: Callable[[], float] = time.time,
                 on_event: Callable[..., None] | None = None) -> None:
        if not run_id or not str(run_id).strip():
            raise NodeLeaseError("node leases need a run id (the holder)")
        if float(ttl_s) <= 0:
            raise NodeLeaseError(f"node lease ttl_s must be positive, got {ttl_s!r}")
        self.store_uri, self.run_id, self.ttl_s = str(store_uri), str(run_id), float(ttl_s)
        self.clock = clock
        self.on_event = on_event
        self._objects = objects
        self._locks: dict[str, StoreLock] = {}
        self._lost: dict[str, str | None] = {}  # node -> the run that holds it now
        self.assigned: set[str] = set()  # nodes given to an island of this launch (inventory_trainer_nodes)
        self._mutex = threading.RLock()
        self._keeper: threading.Thread | None = None
        self._stop = threading.Event()

    # -- store
    @property
    def objects(self) -> Any:
        if self._objects is None:
            self._objects = open_objects(self.store_uri)
        return self._objects

    @staticmethod
    def prefix(node: Any) -> str:
        return f"nodes/{node_name(node)}/lease"

    @staticmethod
    def event_key(node: Any, token: int) -> str:
        return f"nodes/{node_name(node)}/events/{int(token):08d}-takeover.json"

    @property
    def renew_every_s(self) -> float:
        """A held lease is renewed once this much of it is gone (a quarter: three renewals
        may fail before it runs out)."""
        return self.ttl_s / 4

    def _doc(self, key: str) -> dict[str, Any] | None:
        data = self.objects.read(key)
        if data is None:
            return None
        try:
            doc = json.loads(data)
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None

    def _ended_holder(self, name: str) -> str | None:
        """The run of the newest token when its lease ran out without a release (None: no
        token, or it was released)."""
        prefix = self.prefix(name)
        tokens = StoreLock.tokens(self.objects, prefix)
        if not tokens:
            return None
        lease = self._doc(StoreLock.lease_key(prefix, tokens[-1]))
        if lease is not None and lease.get("released"):
            return None
        record = lease or self._doc(StoreLock.record_key(prefix, tokens[-1])) or {}
        holder = record.get("holder")
        return str(holder) if holder else None

    def _event(self, event: str, **detail: Any) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event, **detail)
                return
            except Exception:  # noqa: BLE001 - a report never fails the lease
                pass
        _stderr(f"[node-lease] {event}: " + " ".join(f"{k}={v}" for k, v in detail.items()))

    # -- reading
    def holder(self, node: Any) -> str | None:
        """The run that holds ``node`` now (None: free: no lease, released, or run out)."""
        found = StoreLock.holder_of(self.objects, self.prefix(node), now=self.clock())
        return None if found is None or not found.get("holder") else str(found["holder"])

    def held(self) -> list[str]:
        """The nodes this object holds the lease of."""
        with self._mutex:
            return sorted(self._locks)

    def lost(self) -> dict[str, str | None]:
        """The nodes whose lease this run lost (another run took them after the lease ran
        out), with the run that holds each now."""
        with self._mutex:
            return dict(self._lost)

    # -- claim / renew / release
    def acquire(self, node: Any) -> LeaseResult:
        """Take ``node``'s lease for this run (held already: renewed). Refused (``ok``
        False) while another run's lease runs; an expired lease of another run is taken
        over (``took_over``, and a takeover event)."""
        name = node_name(node)
        with self._mutex:
            lock = self._locks.get(name)
            if lock is not None and self._renew_lock(name, lock):
                return LeaseResult(True, self.run_id)
            prefix = self.prefix(name)
            current = StoreLock.holder_of(self.objects, prefix, now=self.clock())
            previous = self._ended_holder(name) if current is None else None
            try:  # refused while another run's lease runs; this run's own running lease is taken back
                lock = StoreLock.acquire(self.objects, prefix, holder=self.run_id, ttl_s=self.ttl_s,
                                         clock=self.clock, reclaim=True)
            except StoreLockError as error:
                raise NodeLeaseError(f"lease store {self.store_uri}: {error}") from None
            if lock is None:  # held by another run, or its claim won the same token
                current = StoreLock.holder_of(self.objects, prefix, now=self.clock())
                return LeaseResult(False, None if current is None else str(current.get("holder") or "") or None)
            self._locks[name] = lock
            self._lost.pop(name, None)
            took = previous is not None and previous != self.run_id
            if took:
                self._takeover(name, lock.token, previous)
            return LeaseResult(True, self.run_id, took, previous if took else None)

    def _takeover(self, name: str, token: int, previous: str | None) -> None:
        doc = {"schema": EVENT_SCHEMA, "event": TAKEOVER_EVENT, "node": name, "holder": self.run_id,
               "previous": previous, "token": int(token), "at": self.clock()}
        recorded = True
        try:
            self.objects.write(self.event_key(name, token), json.dumps(doc, sort_keys=True).encode())
        except Exception:  # noqa: BLE001 - the lease is taken; the store record is best effort
            recorded = False
        self._event(TAKEOVER_EVENT, node=name, holder=self.run_id, previous=previous, token=int(token),
                    recorded=recorded)

    def _renew_lock(self, name: str, lock: StoreLock) -> bool:
        """Renew one held lease; False once another run holds the node (recorded in
        :meth:`lost`). A store that does not answer keeps the lease while it runs."""
        try:
            lock.renew()
            return True
        except LockLost:
            self._locks.pop(name, None)
            try:
                now = self.holder(name)
            except Exception:  # noqa: BLE001 - who took it is information only
                now = None
            if now == self.run_id:  # a restarted launcher of this run took it back: still this run's
                return True
            self._lost[name] = now
            self._event(LOST_EVENT, node=name, holder=self.run_id, now_held_by=now)
            return False
        except Exception as error:  # noqa: BLE001 - not renewed this time; the lease still runs
            left = lock.lease_left()
            if left <= 0:
                self._event("node_lease_unconfirmed", node=name, error=repr(error)[:200])
            return left > 0

    def renew(self, node: Any) -> bool:
        """Move ``node``'s lease end forward (one not held: taken, e.g. by a restarted
        launcher of this run). False when another run holds the node."""
        name = node_name(node)
        with self._mutex:
            lock = self._locks.get(name)
            if lock is None:
                return self.acquire(name).ok
            return self._renew_lock(name, lock)

    def renew_due(self, *, force: bool = False) -> list[str]:
        """Renew every held lease past :attr:`renew_every_s` (``force``: all); returns the
        nodes lost to another run by this call."""
        lost = []
        with self._mutex:
            now = self.clock()
            for name, lock in sorted(self._locks.items()):
                if force or lock.renewed_at is None or now - lock.renewed_at >= self.renew_every_s:
                    if not self._renew_lock(name, lock):
                        lost.append(name)
        return lost

    def release(self, node: Any) -> None:
        """Give ``node``'s lease up now (the next run does not wait for it to run out).
        Never raises."""
        with self._mutex:
            lock = self._locks.pop(node_name(node), None)
        if lock is not None:
            lock.release()

    def release_all(self, nodes: Iterable[Any] | None = None) -> None:
        for name in (self.held() if nodes is None else [node_name(n) for n in nodes]):
            self.release(name)

    # -- the keeper (a holder with no tick of its own while it starts: preflight, weights)
    def start_keeper(self, interval_s: float | None = None) -> None:
        """Renew the held leases from a daemon thread every ``interval_s`` (default
        :attr:`renew_every_s`) until :meth:`stop_keeper`; idempotent."""
        if self._keeper is not None and self._keeper.is_alive():
            return
        every = float(interval_s or self.renew_every_s)
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(every):
                try:
                    self.renew_due()
                except Exception as error:  # noqa: BLE001 - the next round tries again
                    _LOG.warning("node lease renewal failed: %r", error)

        self._keeper = threading.Thread(target=loop, name="yeto-node-lease-keeper", daemon=True)
        self._keeper.start()

    def stop_keeper(self) -> None:
        self._stop.set()


def pick_nodes(candidates: Sequence[Any], need: int, leases: NodeLeases, *, wait_s: float | None = None,
               poll_s: float = WAIT_POLL_S, sleep: Callable[[float], None] = time.sleep,
               monotonic: Callable[[], float] = time.monotonic,
               log: Callable[[str], None] = _stderr) -> list[Any]:
    """``need`` of ``candidates`` whose leases this run takes, in ``candidates`` order.

    Nodes this run holds already come first (a restarted launcher keeps the nodes its
    running incarnation is on), then free ones; a node another run holds is skipped.
    Fewer free nodes than ``need``: the holders are reported (again whenever they change)
    and the leases read every ``poll_s`` until enough are free, or :class:`NodeLeased`
    after ``wait_s`` (None: no limit)."""
    need = int(need)
    if need <= 0:
        return []
    if len(candidates) < need:
        raise NodeLeased(f"{need} node(s) needed, the inventory offers {len(candidates)}")
    order = {node_name(n): i for i, n in enumerate(candidates)}
    began, reported = monotonic(), None
    while True:
        holders = {node_name(n): leases.holder(n) for n in candidates}
        ours = [n for n in candidates if holders[node_name(n)] == leases.run_id]
        free = [n for n in candidates if holders[node_name(n)] is None]
        want = (ours + free)[:need]
        if len(want) == need:
            taken: list[Any] = []
            for node in want:
                result = leases.acquire(node)
                if not result.ok:  # another run's claim won this node since the read
                    holders[node_name(node)] = result.holder or "(another run)"
                    break
                taken.append(node)
            if len(taken) == need:
                return sorted(taken, key=lambda n: order[node_name(n)])
            for node in taken:  # give back what this round took; wait like any short round
                if node not in ours:
                    leases.release(node)
            ours = [n for n in ours if holders[node_name(n)] == leases.run_id]
            free = [n for n in free if holders[node_name(n)] is None]
        busy = {name: h for name, h in holders.items() if h not in (None, leases.run_id)}
        text = ", ".join(f"{name} by run {h}" for name, h in sorted(busy.items()))
        if busy != reported:
            log(f"[node-lease] {need} node(s) needed, {len(ours) + len(free)} free; held: {text}; waiting "
                f"(the leases are read every {poll_s:.0f}s)")
            reported = busy
        if wait_s is not None and monotonic() - began >= wait_s:
            raise NodeLeased(f"{need} node(s) needed, {len(ours) + len(free)} free after {wait_s:.0f}s; held: "
                             f"{text}", busy)
        sleep(poll_s)


def for_inventory(nodes: Any, run_id: str, **kwargs: Any) -> NodeLeases | None:
    """The :class:`NodeLeases` of an inventory's node list (None: no ``lease_store``)."""
    from .node_inventory import lease_store_of  # noqa: PLC0415

    store = lease_store_of(nodes)
    return None if store is None else NodeLeases(store, run_id, **kwargs)
