"""Island side of the task dispenser (S27 P2): which rows of ``--data`` this island trains.

With ``--rl-data-split dispense`` the actor syncer runs a dispenser
(``syncer/src/dispense.rs``) and every training island takes its rows from it
instead of from a fixed slot. Plain rules:

* The dispenser hands out *positions* of one global sequence. Position ``p``
  is row ``pass_order(rows, seed, p // rows)[p % rows]`` of ``--data``
  (:class:`PositionMap`): pass 0 in data order, every later pass reshuffled
  (``data_split.pass_order``). The sequence does not depend on how many
  islands there are, so islands may join, leave and restart at any time.
* An island process (:class:`TaskSource`) claims a *block* of positions at a
  time (its own round size: a fast island claims more often, so it trains
  more rows), keeps up to ``prefetch_blocks`` blocks ready, and reports every
  position it took: trained, dropped (consumed, not trained), requeue (a
  stale group, trained again at most ``--dispense-max-redo`` times) or
  release (never used).
* An island process that dies, or is silent for the dispenser lease, loses
  the positions it did not report; they go first to the next claim of any
  island. A restarted island is a new process (new incarnation): it simply
  claims new blocks, it needs no data cursor of its own.
* Head unreachable: the island keeps taking from the blocks it already holds
  while its lease is still valid on its own clock, then waits (it never
  reports the data as ended). Reports wait in a queue until the head answers.
* The data ends only when the dispenser says so (``--rl-data-passes``);
  :meth:`TaskSource.take` then raises :class:`DispenseEnded`.

Wire format: ``docs/PROTOCOL.md`` "Task dispenser"; the golden bytes are
pinned on both sides (``syncer/src/dispense_service.rs`` and
``tests/test_s27_dispense.py``). Import-light: runs inside the Miles rollout
executor process; nothing here imports torch, numpy or Miles.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .data_split import pass_order

logger = logging.getLogger(__name__)

MAGIC = 0xD170C0DE
MSG_ERROR = 10
MSG_TASK_HELLO = 50
MSG_TASK_WELCOME = 51
MSG_TASK_CLAIM = 52
MSG_TASK_GRANT = 53
MSG_TASK_REPORT = 54
MSG_TASK_ACK = 55
MSG_TASK_HEARTBEAT = 56
TASK_PROTOCOL_VERSION = 1
TASK_HELLO_MAC_DOMAIN = b"yeto-task-hello-mac-v1\0"

GRANT, WAIT, END = 0, 1, 2
TRAINED, DROPPED, REQUEUE, RELEASE = 1, 2, 3, 4
KIND_NAMES = {TRAINED: "trained", DROPPED: "dropped", REQUEUE: "requeue", RELEASE: "release"}

# refusal prefixes (syncer/src/dispense.rs)
FENCED_PREFIX = "fenced: dispense"
MISMATCH_PREFIX = "dispense mismatch:"
REFUSED_PREFIX = "dispense refused:"

_HEADER = struct.Struct("<IBQ")
_MAX_ANSWER = 1 << 21

# a sample's metadata key: its dispenser position
DISPENSE_POSITION_KEY = "yeto_dispense_position"


class DispenseError(RuntimeError):
    """The dispenser refused a request (fatal for this island)."""


class DispenseMismatch(DispenseError):
    """The island's data (row count or fingerprint) differs from the run's."""


class DispenseFenced(ConnectionError):
    """This incarnation's lease is gone; its unreported positions went to others."""


class DispenseEnded(RuntimeError):
    """Every pass was given out and reported (``--rl-data-passes``): the data ended."""


# ---------------------------------------------------------------- wire codec


def encode_task_hello(learner_id: int, incarnation: int, rows: int, fingerprint: bytes) -> bytes:
    if len(fingerprint) != 32:
        raise ValueError("the data fingerprint must be 32 bytes")
    return struct.pack("<HIQQ", TASK_PROTOCOL_VERSION, int(learner_id), int(incarnation), int(rows)) + fingerprint


def seal_task_hello(body: bytes, key: bytes | None) -> bytes:
    """``body | HMAC-SHA256(key, TASK_HELLO_MAC_DOMAIN | body)``; no-op without a key."""
    if not key:
        return body
    return body + hmac.new(key, TASK_HELLO_MAC_DOMAIN + body, hashlib.sha256).digest()


@dataclass(frozen=True)
class Welcome:
    lease_s: float
    passes: int | None
    max_block: int
    reconnect: bool
    superseded: int | None
    reclaimed: int
    next: int


def decode_welcome(payload: bytes) -> Welcome:
    fmt = "<HQQIBBQQQ"
    if len(payload) != struct.calcsize(fmt):
        raise DispenseError(f"TASK_WELCOME has {len(payload)} bytes")
    version, lease_ms, passes, max_block, reconnect, has_sup, sup, reclaimed, nxt = struct.unpack(fmt, payload)
    if version != TASK_PROTOCOL_VERSION:
        raise DispenseError(f"the syncer speaks task protocol {version}, this island {TASK_PROTOCOL_VERSION}")
    return Welcome(lease_ms / 1000.0, passes or None, max_block, bool(reconnect),
                   sup if has_sup else None, reclaimed, nxt)


def encode_claim(seq: int, count: int) -> bytes:
    """``seq`` numbers this incarnation's CLAIM / REPORT requests (strictly
    increasing); a request resent after a lost answer repeats its ``seq``."""
    return struct.pack("<QI", int(seq), int(count))


@dataclass(frozen=True)
class Grant:
    status: int
    block: int
    retry_s: float
    ranges: tuple[tuple[int, int], ...]


def _ranges(payload: bytes, offset: int) -> tuple[tuple[tuple[int, int], ...], int]:
    (n,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    if n > (len(payload) - offset) // 16:
        raise DispenseError(f"range count {n} exceeds the payload")
    out = tuple(struct.unpack_from("<QQ", payload, offset + 16 * i) for i in range(n))
    return out, offset + 16 * n


def decode_grant(payload: bytes) -> Grant:
    status, block, retry_ms = struct.unpack_from("<BQI", payload, 0)
    ranges, end = _ranges(payload, 13)
    if end != len(payload) or status not in (GRANT, WAIT, END):
        raise DispenseError("malformed TASK_GRANT")
    return Grant(status, block, retry_ms / 1000.0, ranges)


def encode_report(seq: int, entries: Sequence[tuple[int, int, Sequence[tuple[int, int]]]]) -> bytes:
    """``entries``: ``(block, kind, ranges)``; ``seq`` as in :func:`encode_claim`."""
    out = [struct.pack("<QI", int(seq), len(entries))]
    for block, kind, ranges in entries:
        out.append(struct.pack("<QBI", int(block), int(kind), len(ranges)))
        out.extend(struct.pack("<QQ", int(a), int(b)) for a, b in ranges)
    return b"".join(out)


@dataclass(frozen=True)
class Ack:
    settled: int
    requeued: int
    exhausted: int


def decode_ack(payload: bytes) -> Ack:
    if len(payload) != 24:
        raise DispenseError(f"TASK_ACK has {len(payload)} bytes")
    return Ack(*struct.unpack("<QQQ", payload))


def ranges_of(positions: Iterable[int]) -> list[tuple[int, int]]:
    """Sorted distinct positions as ``[start, end)`` ranges."""
    out: list[tuple[int, int]] = []
    for p in sorted(set(int(p) for p in positions)):
        if out and out[-1][1] == p:
            out[-1] = (out[-1][0], p + 1)
        else:
            out.append((p, p + 1))
    return out


# ------------------------------------------------------------- positions -> rows


class PositionMap:
    """Position -> row of ``--data`` (module doc). Caches the two latest pass orders."""

    def __init__(self, rows: int, *, seed: int) -> None:
        if int(rows) < 1:
            raise ValueError("the data has no rows")
        self.rows, self.seed = int(rows), int(seed)
        self._orders: dict[int, list[int]] = {}

    def data_pass(self, position: int) -> int:
        return int(position) // self.rows

    def row(self, position: int) -> int:
        data_pass, index = divmod(int(position), self.rows)
        order = self._orders.get(data_pass)
        if order is None:
            order = pass_order(self.rows, seed=self.seed, data_pass=data_pass)
            self._orders[data_pass] = order
            for old in [p for p in self._orders if p < data_pass - 1]:
                del self._orders[old]
        return order[index]


def file_fingerprint(path: str | os.PathLike[str]) -> tuple[int, bytes]:
    """``(row count, SHA-256)`` of a JSONL prompt file: what every island must agree on."""
    digest, rows = hashlib.sha256(), 0
    with open(os.path.expanduser(str(path)), "rb") as handle:
        for line in handle:
            digest.update(line)
            rows += bool(line.strip())
    return rows, digest.digest()


# ------------------------------------------------------------------ connection


def _refusal(text: str) -> Exception:
    if text.startswith(FENCED_PREFIX):
        return DispenseFenced(text)
    if text.startswith(MISMATCH_PREFIX):
        return DispenseMismatch(text)
    return DispenseError(text)


class DispenseConnection:
    """One TCP connection to the dispenser; one answer per request, in order."""

    def __init__(self, address: tuple[str, int], *, learner_id: int, incarnation: int, rows: int,
                 fingerprint: bytes, key: bytes | None, timeout_s: float = 30.0) -> None:
        self.sock = socket.create_connection(address, timeout=timeout_s)
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._send(MSG_TASK_HELLO, seal_task_hello(
                encode_task_hello(learner_id, incarnation, rows, fingerprint), key))
            self.welcome = decode_welcome(self._answer(MSG_TASK_WELCOME))
        except BaseException:
            self.close()
            raise

    def _send(self, msg_type: int, payload: bytes) -> None:
        self.sock.sendall(_HEADER.pack(MAGIC, msg_type, len(payload)) + payload)

    def _read(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("the dispenser closed the connection")
            buf += chunk
        return bytes(buf)

    def _answer(self, expected: int) -> bytes:
        magic, msg_type, length = _HEADER.unpack(self._read(_HEADER.size))
        if magic != MAGIC or length > _MAX_ANSWER:
            raise ConnectionError(f"bad frame from the dispenser (magic {magic:#x}, {length} bytes)")
        payload = self._read(length)
        if msg_type == MSG_ERROR:
            raise _refusal(payload.decode("utf-8", "replace"))
        if msg_type != expected:
            raise ConnectionError(f"the dispenser answered type {msg_type}, expected {expected}")
        return payload

    def claim(self, seq: int, count: int) -> Grant:
        self._send(MSG_TASK_CLAIM, encode_claim(seq, count))
        return decode_grant(self._answer(MSG_TASK_GRANT))

    def report(self, seq: int, entries: Sequence[tuple[int, int, Sequence[tuple[int, int]]]]) -> Ack:
        self._send(MSG_TASK_REPORT, encode_report(seq, entries))
        return decode_ack(self._answer(MSG_TASK_ACK))

    def heartbeat(self) -> None:
        self._send(MSG_TASK_HEARTBEAT, b"")
        decode_ack(self._answer(MSG_TASK_ACK))

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


# ----------------------------------------------------------------- task source


class TaskSource:
    """The island's positions (module doc): prefetch, offline wait, reports.

    ``take`` and ``report`` are called by the rollout; one background thread
    talks to the dispenser (claims, reports, heartbeats, reconnects).
    ``connect(incarnation) -> DispenseConnection`` is injectable for tests."""

    def __init__(self, *, connect: Callable[[int], DispenseConnection], block: int,
                 prefetch_blocks: int = 2, lease_margin: float = 0.2,
                 clock: Callable[[], float] = time.monotonic,
                 emit: Callable[..., None] | None = None,
                 incarnation: Callable[[], int] | None = None,
                 retry_s: float = 1.0) -> None:
        if int(block) < 1 or int(prefetch_blocks) < 1:
            raise ValueError("block and prefetch_blocks must be >= 1")
        self._connect = connect
        self.block = int(block)
        self.prefetch_blocks = int(prefetch_blocks)
        self.lease_margin = float(lease_margin)
        self._clock = clock
        self._emit = emit or (lambda kind, **fields: None)
        self._new_incarnation = incarnation or (lambda: int.from_bytes(os.urandom(8), "little") or 1)
        self._retry_s = float(retry_s)
        self._cv = threading.Condition()
        self._ready: deque[tuple[int, int]] = deque()  # (position, block)
        self._owner: dict[int, int] = {}  # taken, unreported position -> block
        self._pending: list[tuple[int, int, list[int]]] = []  # (block, kind, positions) to send
        # the CLAIM / REPORT on the wire: (seq, "claim", count) or (seq, "report", entries);
        # kept across a reconnect and resent with the same seq
        self._inflight: tuple[int, str, Any] | None = None
        self._seq = 0
        self._ended = False
        # the latest claim got WAIT: with limited passes nothing fresh is left
        # (an unlimited dispenser never answers WAIT), so a draw returns short
        self._starved = False
        self._fatal: BaseException | None = None
        self._stopped = False
        self._closing = False  # no more claims
        self._conn: DispenseConnection | None = None
        self.incarnation = self._new_incarnation()
        self.lease_s: float | None = None
        self._contact: float | None = None  # last answer from the dispenser
        self._wait_until = 0.0
        self.stats = {"claims": 0, "granted": 0, "waits": 0, "fenced": 0, "reconnects": 0,
                      "offline_waits": 0, "lost_reports": 0}
        for kind in KIND_NAMES.values():
            self.stats[kind] = 0
        self._thread = threading.Thread(target=self._run, name="yeto-task-dispense", daemon=True)

    # ------------------------------------------------------------ public API

    def start(self) -> "TaskSource":
        """Open the first connection (raises on refusal) and start the worker."""
        self._open()
        self._thread.start()
        return self

    def take(self, count: int, *, timeout_s: float | None = None,
             drained_ok: bool = False) -> list[int]:
        """``count`` positions, in grant order. Waits while none are ready or the
        lease is not valid on this island's clock (head unreachable). Raises
        :class:`DispenseEnded` when the data ended and nothing is ready.
        ``drained_ok``: when nothing fresh is left (the latest claim got WAIT)
        return what there is at once, possibly nothing, instead of waiting for
        rows other islands may give back."""
        count = int(count)
        out: list[int] = []
        deadline = None if timeout_s is None else self._clock() + float(timeout_s)
        with self._cv:
            while len(out) < count:
                if self._fatal is not None:
                    raise self._fatal
                if self._ready and self._lease_valid():
                    position, block = self._ready.popleft()
                    self._owner[position] = block
                    out.append(position)
                    self._cv.notify_all()  # the worker may prefetch now
                    continue
                if not self._ready and (self._ended or self._starved) and out:
                    break  # a short draw at the end of the data: hand back what there is
                if not self._ready and self._ended:
                    raise DispenseEnded("the dispenser gave out every pass and every row was reported")
                if not self._ready and self._starved and drained_ok:
                    break  # nothing fresh is left: the caller decides (it holds rows)
                if self._ready:
                    self.stats["offline_waits"] += 1
                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"no dispensed rows within {timeout_s} s "
                                       f"({len(out)} of {count} taken)")
                self._cv.notify_all()
                self._cv.wait(timeout=0.5 if remaining is None else min(0.5, remaining))
        return out

    def report(self, *, trained: Iterable[int] = (), dropped: Iterable[int] = (),
               requeue: Iterable[int] = (), release: Iterable[int] = ()) -> None:
        """Queue reports of taken positions (sent by the worker; never blocks).
        A position this incarnation does not hold (taken before a fence) is
        counted in ``stats["lost_reports"]`` and ignored."""
        with self._cv:
            for kind, positions in ((TRAINED, trained), (DROPPED, dropped),
                                    (REQUEUE, requeue), (RELEASE, release)):
                by_block: dict[int, list[int]] = {}
                for position in positions:
                    block = self._owner.pop(int(position), None)
                    if block is None:
                        self.stats["lost_reports"] += 1
                        continue
                    by_block.setdefault(block, []).append(int(position))
                for block, items in by_block.items():
                    self._pending.append((block, kind, items))
            self._cv.notify_all()

    def taken(self) -> set[int]:
        """Positions taken and not reported yet."""
        with self._cv:
            return set(self._owner)

    def ready_count(self) -> int:
        """Positions granted and not taken yet (prefetched)."""
        with self._cv:
            return len(self._ready)

    def release_ready(self, *, stop_claims: bool = False) -> int:
        """Release the prefetched positions (granted, not taken); with
        ``stop_claims`` no new block is claimed afterwards. Returns how many."""
        with self._cv:
            if stop_claims:
                self._closing = True
            ready = [p for p, _ in self._ready]
            for position, block in self._ready:
                self._owner[position] = block
            self._ready.clear()
        if ready:
            self.report(release=ready)
        return len(ready)

    def flush(self, timeout_s: float = 30.0) -> bool:
        """Wait until every queued report is acknowledged; False on timeout."""
        deadline = self._clock() + float(timeout_s)
        with self._cv:
            while self._pending or self._inflight is not None:
                if self._fatal is not None:
                    raise self._fatal
                if self._clock() >= deadline:
                    return False
                self._cv.notify_all()
                self._cv.wait(timeout=0.1)
        return True

    def close(self, *, release: bool = True, timeout_s: float = 10.0) -> None:
        """Stop: with ``release`` every ready and taken, unreported position is
        released first (they go to the other islands at once)."""
        with self._cv:
            self._closing = True
            # a claim on the wire may still land; wait for it so its block is released too
            deadline = self._clock() + float(timeout_s)
            while (self._inflight is not None and self._inflight[1] == "claim"
                   and self._thread.is_alive() and self._clock() < deadline):
                self._cv.wait(timeout=0.05)
        if release and self._thread.is_alive():
            self.release_ready()
            self.report(release=sorted(self.taken()))
            try:
                self.flush(timeout_s)
            except BaseException as error:  # noqa: BLE001 - closing anyway
                logger.warning("task dispenser: release at close failed: %r", error)
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
        if self._conn is not None:
            self._conn.close()

    # --------------------------------------------------------------- worker

    def _lease_valid(self) -> bool:
        if self._contact is None or self.lease_s is None:
            return False
        return self._clock() - self._contact < self.lease_s * (1.0 - self.lease_margin)

    def _touch(self) -> None:
        self._contact = self._clock()

    def _open(self) -> None:
        conn = self._connect(self.incarnation)
        welcome = conn.welcome
        with self._cv:
            self._conn, self.lease_s = conn, welcome.lease_s
            self._touch()
            self._cv.notify_all()
        self._emit("rl_dispense_open", incarnation=f"{self.incarnation:016x}",
                   reconnect=welcome.reconnect, lease_s=welcome.lease_s, passes=welcome.passes,
                   superseded=None if welcome.superseded is None else f"{welcome.superseded:016x}",
                   reclaimed=welcome.reclaimed)

    def _reset_after_fence(self, error: BaseException) -> None:
        """The dispenser retired this incarnation: drop its blocks, open a new one."""
        with self._cv:
            dropped = len(self._ready)
            self._ready.clear()
            self._owner.clear()
            self._pending.clear()
            self._inflight = None
            self.stats["fenced"] += 1
            old = self.incarnation
            self.incarnation = self._new_incarnation()
            self._contact = None
        logger.warning("task dispenser fenced incarnation %016x (%s); %d ready positions dropped",
                       old, error, dropped)
        self._emit("rl_dispense_fenced", incarnation=f"{old:016x}", dropped_ready=dropped,
                   error=str(error))

    def _want_claim(self) -> bool:
        return (not self._ended and not self._closing and self._clock() >= self._wait_until
                and len(self._ready) < self.block * self.prefetch_blocks)

    def _step(self) -> bool:
        """One request to the dispenser; False when there was nothing to do."""
        with self._cv:
            conn = self._conn
            if conn is None:
                raise ConnectionError("no dispenser connection")
            if self._inflight is None:
                if self._pending:
                    self._seq += 1
                    self._inflight = (self._seq, "report", self._pending)
                    self._pending = []
                elif self._want_claim():
                    self._seq += 1
                    self._inflight = (self._seq, "claim", self.block)
            request = self._inflight
            beat = (request is None and self.lease_s is not None and self._contact is not None
                    and self._clock() - self._contact >= self.lease_s / 3.0)
        if request is None:
            if not beat:
                return False
            conn.heartbeat()
            with self._cv:
                self._touch()
            return True
        seq, what, body = request
        if what == "report":
            conn.report(seq, [(block, kind, ranges_of(items)) for block, kind, items in body])
            with self._cv:
                for _block, kind, items in body:
                    self.stats[KIND_NAMES[kind]] += len(items)
                self._inflight = None
                self._touch()
                self._cv.notify_all()
            return True
        grant = conn.claim(seq, body)
        with self._cv:
            self._inflight = None
            self._touch()
            self.stats["claims"] += 1
            self._starved = grant.status == WAIT
            if grant.status == GRANT:
                for start, end in grant.ranges:
                    self._ready.extend((p, grant.block) for p in range(start, end))
                self.stats["granted"] += sum(e - s for s, e in grant.ranges)
            elif grant.status == WAIT:
                self.stats["waits"] += 1
                self._wait_until = self._clock() + max(grant.retry_s, 0.05)
            else:
                self._ended = True
            self._cv.notify_all()
        return True

    def _run(self) -> None:
        backoff = self._retry_s
        while True:
            with self._cv:
                if self._stopped:
                    return
            try:
                if self._conn is None:
                    self._open()
                    self.stats["reconnects"] += 1
                    backoff = self._retry_s
                busy = self._step()
            except DispenseFenced as error:
                if self._conn is not None:
                    self._conn.close()
                self._conn = None
                self._reset_after_fence(error)
                continue
            except DispenseError as error:
                with self._cv:
                    self._fatal = error
                    self._cv.notify_all()
                logger.error("task dispenser refused this island: %s", error)
                self._emit("rl_dispense_refused", error=str(error))
                return
            except (OSError, ConnectionError, struct.error) as error:
                if self._conn is not None:
                    self._conn.close()
                self._conn = None
                logger.warning("task dispenser unreachable (%r); retrying in %.1f s", error, backoff)
                with self._cv:
                    self._cv.wait(timeout=backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            if not busy:
                with self._cv:
                    if not self._stopped:
                        self._cv.wait(timeout=0.2)


def tcp_connector(address: str | tuple[str, int], *, learner_id: int, rows: int, fingerprint: bytes,
                  key: bytes | None, timeout_s: float = 30.0) -> Callable[[int], DispenseConnection]:
    """``connect(incarnation)`` for :class:`TaskSource` over TCP (``host:port``)."""
    if isinstance(address, str):
        host, _, port = address.rpartition(":")
        address = (host, int(port))

    def connect(incarnation: int) -> DispenseConnection:
        return DispenseConnection(address, learner_id=learner_id, incarnation=incarnation, rows=rows,
                                  fingerprint=fingerprint, key=key, timeout_s=timeout_s)

    return connect
