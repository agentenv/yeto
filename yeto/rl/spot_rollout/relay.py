"""Client of the syncer's sample relay (S27 G-EA; syncer ``sample_relay.rs``).

The relay is a small in-memory queue on the head. Sampler islands on any
cloud PUT rollout groups into the channel of a training island; that island
TAKEs them. Nothing else is shared between the clouds: no volume, no object
store, no store credentials on the sampler.

Wire (each connection: first frame RELAY_HELLO, then request/answer pairs):

====  ============  ========================================================
type  name          payload
====  ============  ========================================================
60    RELAY_HELLO   version u16 (=2) | role u8 (1 producer, 2 consumer,
                    3 board) | channel u32 | client u32 | incarnation u64 |
                    stream u64, then HMAC-SHA256(key, RELAY_MAC_DOMAIN | body)
                    with a key
61    WELCOME       version u16 | relay_epoch u64 | max_item_bytes u64 |
                    max_items u32 | max_bytes u64 | depth
62    PUT           seq u64 | item bytes (producer; board: sets the board)
63    PUT_ACK       seq u64 | status u8 (0 stored, 1 duplicate, 2 full) |
                    item_id u64 | depth
64    TAKE          after u64 | ack_through u64 | max_items u32 |
                    max_bytes u64 | wait_ms u32 (consumer; board: newest)
65    ITEMS         n u32 | depth | n x (item_id u64 | producer u32 |
                    len u64 | bytes)
66    STATUS        request: empty; answer: depth
====  ============  ========================================================

``depth`` = items u32 | bytes u64 | acked_through u64 | next_id u64.

Delivery: a producer numbers its PUTs from 1 per (incarnation, stream);
resending the last number gets the same answer and stores nothing twice.
``incarnation`` is the sampler process's member incarnation
(``task_dispense.member_incarnation``: the same on its TASK_HELLO and
SUBSCRIBE); ``stream`` is random per producer object, so two writers of one
process never share a sequence. A full channel
answers FULL and stores nothing (the producer retries later). The consumer
reads after a cursor and acknowledges through an item id; only acknowledged
items are deleted, so a reconnecting or restarted consumer gets every
unacknowledged item again. A head restart loses the queue: the WELCOME then
carries a new ``relay_epoch`` and the clients reset their cursors.

Errors: the syncer answers ERROR (type 10) and closes. ``relay refused:``
(relay off, bad HELLO, too large item) is fatal; ``relay superseded:`` means
a newer consumer owns the channel; ``fenced: relay`` means the syncer's
island member table retired this producer's member (a newer incarnation of
the sampler joined on any channel, or its lease expired): fatal, the process
does not come back under the same incarnation (:class:`RelayFenced`).
"""

from __future__ import annotations

import secrets
import socket
import struct
import time
from dataclasses import dataclass

from yeto.protocol import (
    MSG_ERROR,
    ProtocolError,
    enable_tcp_keepalive,
    island_hmac_key_from_env,
    read_frame,
    seal_hello,
    write_frame,
)

MSG_RELAY_HELLO = 60
MSG_RELAY_WELCOME = 61
MSG_RELAY_PUT = 62
MSG_RELAY_PUT_ACK = 63
MSG_RELAY_TAKE = 64
MSG_RELAY_ITEMS = 65
MSG_RELAY_STATUS = 66

RELAY_VERSION = 2  # 2: RELAY_HELLO carries the producer stream
RELAY_MAC_DOMAIN = b"yeto-relay-hello-mac-v1\0"
ROLE_PRODUCER = 1
ROLE_CONSUMER = 2
ROLE_BOARD = 3

PUT_STORED = 0
PUT_DUPLICATE = 1
PUT_FULL = 2

REFUSED_PREFIX = "relay refused:"
SUPERSEDED_PREFIX = "relay superseded:"
FENCED_PREFIX = "fenced: relay"  # syncer sample_relay.rs FENCED_PREFIX

_HELLO = struct.Struct("<HBIIQQ")
_WELCOME = struct.Struct("<HQQIQ")
_DEPTH = struct.Struct("<IQQQ")
_PUT_ACK = struct.Struct("<QBQ")
_TAKE = struct.Struct("<QQIQI")
_ITEM = struct.Struct("<QIQ")


class RelayError(ConnectionError):
    """The relay connection failed (retryable)."""


class RelayRefused(ProtocolError):
    """The syncer refused the relay request (not retryable)."""


class RelaySuperseded(RelayError):
    """A newer consumer owns the channel."""


class RelayFenced(RelayRefused):
    """The syncer retired this producer's member (superseded or expired)."""


@dataclass(frozen=True)
class Depth:
    items: int
    bytes: int
    acked_through: int
    next_id: int

    @classmethod
    def decode(cls, data: bytes, offset: int = 0) -> "Depth":
        return cls(*_DEPTH.unpack_from(data, offset))


@dataclass(frozen=True)
class Welcome:
    epoch: int
    max_item_bytes: int
    max_items: int
    max_bytes: int
    depth: Depth


@dataclass(frozen=True)
class PutAck:
    seq: int
    status: int
    item_id: int
    depth: Depth


@dataclass(frozen=True)
class Item:
    item_id: int
    producer: int
    data: bytes


def encode_relay_hello(role: int, channel: int, client: int, incarnation: int,
                       key: bytes | None, stream: int = 0) -> bytes:
    body = _HELLO.pack(RELAY_VERSION, role, channel, client, incarnation, stream)
    return seal_hello(body, key, RELAY_MAC_DOMAIN)


def _raise_error(payload: bytes) -> None:
    message = payload.decode("utf-8", errors="replace")
    if message.startswith(SUPERSEDED_PREFIX):
        raise RelaySuperseded(message)
    if message.startswith(FENCED_PREFIX):
        raise RelayFenced(message)
    raise RelayRefused(message)


class RelayConnection:
    """One relay connection (blocking; one request at a time)."""

    def __init__(self, addr: tuple[str, int], role: int, channel: int, client: int, *,
                 incarnation: int | None = None, key: bytes | None = None,
                 timeout: float = 120.0, stream: int | None = None) -> None:
        self.addr = (str(addr[0]), int(addr[1]))
        self.role = int(role)
        self.channel = int(channel)
        self.client = int(client)
        self.incarnation = secrets.randbits(63) | 1 if incarnation is None else int(incarnation)
        # one per connection object, kept across its reconnects (resend dedup)
        self.stream = secrets.randbits(63) | 1 if stream is None else int(stream)
        self.key = key if key is not None else island_hmac_key_from_env()
        self.timeout = float(timeout)
        self.sock: socket.socket | None = None
        self.welcome: Welcome | None = None

    def connect(self) -> Welcome:
        self.close()
        try:
            sock = socket.create_connection(self.addr, timeout=self.timeout)
        except OSError as error:
            raise RelayError(f"cannot reach the relay at {self.addr}: {error}") from error
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        enable_tcp_keepalive(sock)
        self.sock = sock
        try:
            write_frame(sock, MSG_RELAY_HELLO,
                        encode_relay_hello(self.role, self.channel, self.client, self.incarnation, self.key,
                                           self.stream))
            msg_type, payload = self._read(1 << 16)
        except BaseException:
            self.close()
            raise
        if msg_type != MSG_RELAY_WELCOME:
            self.close()
            raise RelayError(f"relay answered type {msg_type} to RELAY_HELLO")
        version, epoch, max_item, max_items, max_bytes = _WELCOME.unpack_from(payload)
        if version != RELAY_VERSION:
            self.close()
            raise RelayRefused(f"relay version {version}, this client speaks {RELAY_VERSION}")
        self.welcome = Welcome(epoch, max_item, max_items, max_bytes, Depth.decode(payload, _WELCOME.size))
        return self.welcome

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def _read(self, limit: int) -> tuple[int, bytes]:
        assert self.sock is not None
        try:
            msg_type, payload = read_frame(self.sock, lambda t: 1 << 16 if t == MSG_ERROR else limit)
        except (OSError, ConnectionError, ValueError) as error:
            self.close()
            raise RelayError(f"relay connection lost: {error}") from error
        if msg_type == MSG_ERROR:
            self.close()
            _raise_error(payload)
        return msg_type, payload

    def request(self, msg_type: int, payload: bytes, *, expect: int, limit: int = 1 << 16,
                timeout: float | None = None) -> bytes:
        if self.sock is None:
            self.connect()
        assert self.sock is not None
        try:
            self.sock.settimeout(self.timeout if timeout is None else timeout)
            write_frame(self.sock, msg_type, payload)
        except OSError as error:
            self.close()
            raise RelayError(f"relay send failed: {error}") from error
        answer, body = self._read(limit)
        if answer != expect:
            self.close()
            raise RelayError(f"relay answered type {answer}, expected {expect}")
        return body

    def status(self) -> Depth:
        return Depth.decode(self.request(MSG_RELAY_STATUS, b"", expect=MSG_RELAY_STATUS))


class RelayProducer:
    """Put items into one channel; reconnects and resends on a broken link.
    ``incarnation``: the process's member incarnation (default
    ``task_dispense.member_incarnation()``, shared with its subscriber and
    task dispenser connections)."""

    def __init__(self, addr: tuple[str, int], channel: int, producer: int, *,
                 key: bytes | None = None, timeout: float = 120.0, retry_s: float = 2.0,
                 max_attempts: int | None = None, sleep=time.sleep,
                 incarnation: int | None = None) -> None:
        if incarnation is None:
            from yeto.rl.engine.task_dispense import member_incarnation

            incarnation = member_incarnation()
        self.conn = RelayConnection(addr, ROLE_PRODUCER, channel, producer, key=key, timeout=timeout,
                                    incarnation=incarnation)
        self.seq = 0
        self.retry_s = float(retry_s)
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.last_depth: Depth | None = None
        self.stored = 0
        self.full_answers = 0
        self.reconnects = 0

    @property
    def incarnation(self) -> int:
        return self.conn.incarnation

    def _with_retry(self, call):
        attempt = 0
        while True:
            attempt += 1
            try:
                if self.conn.sock is None:
                    self.conn.connect()
                    self.reconnects += 1
                return call()
            except RelayRefused:
                raise
            except RelayError:
                if self.max_attempts is not None and attempt >= self.max_attempts:
                    raise
                self.sleep(self.retry_s)

    def put(self, data: bytes) -> PutAck:
        """One attempt to store ``data`` (retrying a broken link). FULL is
        returned to the caller, which decides whether to wait and retry."""
        self.seq += 1
        seq = self.seq
        payload = struct.pack("<Q", seq) + bytes(data)

        def call() -> PutAck:
            body = self.conn.request(MSG_RELAY_PUT, payload, expect=MSG_RELAY_PUT_ACK)
            got_seq, status, item_id = _PUT_ACK.unpack_from(body)
            if got_seq != seq:
                raise RelayError(f"relay acknowledged seq {got_seq}, sent {seq}")
            return PutAck(got_seq, status, item_id, Depth.decode(body, _PUT_ACK.size))

        ack = self._with_retry(call)
        if ack.status == PUT_FULL:
            self.seq -= 1  # nothing stored: the next put reuses this number
            self.full_answers += 1
        else:
            self.stored += 1
        self.last_depth = ack.depth
        return ack

    def put_blocking(self, data: bytes, *, should_stop=lambda: False, wait_s: float = 2.0) -> PutAck | None:
        """Put, waiting while the channel is full; None when ``should_stop()``."""
        while True:
            ack = self.put(data)
            if ack.status != PUT_FULL:
                return ack
            if should_stop():
                return None
            self.sleep(wait_s)

    def depth(self) -> Depth:
        depth = self._with_retry(self.conn.status)
        self.last_depth = depth
        return depth

    def close(self) -> None:
        self.conn.close()


class RelayConsumer:
    """Take items from one channel. ``take`` returns new items; ``ack`` marks
    every item through an id as consumed (sent with the next take)."""

    def __init__(self, addr: tuple[str, int], channel: int, consumer: int = 0, *,
                 key: bytes | None = None, timeout: float = 120.0, retry_s: float = 2.0,
                 max_bytes: int = 256 << 20, sleep=time.sleep) -> None:
        self.conn = RelayConnection(addr, ROLE_CONSUMER, channel, consumer, key=key, timeout=timeout)
        self.retry_s = float(retry_s)
        self.max_bytes = int(max_bytes)
        self.sleep = sleep
        self.after = 0
        self.acked = 0
        self.epoch: int | None = None
        self.epoch_resets = 0
        self.last_depth: Depth | None = None

    def _connect(self) -> None:
        welcome = self.conn.connect()
        if self.epoch is not None and welcome.epoch != self.epoch:
            # The head restarted: its queue (and item ids) are new.
            self.epoch_resets += 1
            self.after = self.acked = 0
        elif self.epoch is not None:
            # Same relay, new connection: everything not acknowledged comes again.
            self.after = self.acked
        self.epoch = welcome.epoch

    def ack(self, item_id: int) -> None:
        self.acked = max(self.acked, int(item_id))

    def take(self, *, max_items: int = 64, wait_s: float = 5.0) -> list[Item]:
        while True:
            try:
                if self.conn.sock is None:
                    self._connect()
                payload = _TAKE.pack(self.after, self.acked, int(max_items), self.max_bytes,
                                     int(max(0.0, wait_s) * 1000))
                body = self.conn.request(MSG_RELAY_TAKE, payload, expect=MSG_RELAY_ITEMS,
                                         limit=self.max_bytes + (64 << 20) + 4096,
                                         timeout=self.conn.timeout + wait_s)
                break
            except RelayRefused:
                raise
            except RelaySuperseded:
                raise
            except RelayError:
                self.sleep(self.retry_s)
        (n,) = struct.unpack_from("<I", body)
        self.last_depth = Depth.decode(body, 4)
        offset = 4 + _DEPTH.size
        items = []
        for _ in range(n):
            item_id, producer, length = _ITEM.unpack_from(body, offset)
            offset += _ITEM.size
            items.append(Item(item_id, producer, body[offset:offset + length]))
            offset += length
        if items:
            self.after = max(self.after, items[-1].item_id)
        return items

    def close(self) -> None:
        self.conn.close()


class RelayBoard:
    """A named value on the relay (role 3): the newest PUT wins and stays
    readable by every client. Used for the trainer's fragment layout (the
    subscriber needs the tensor names) and small status records."""

    def __init__(self, addr: tuple[str, int], board: int, client: int = 0, *,
                 key: bytes | None = None, timeout: float = 120.0) -> None:
        self.conn = RelayConnection(addr, ROLE_BOARD, board, client, key=key, timeout=timeout)
        self.seq = 0

    def set(self, data: bytes) -> int:
        self.seq += 1
        payload = struct.pack("<Q", self.seq) + bytes(data)
        body = self.conn.request(MSG_RELAY_PUT, payload, expect=MSG_RELAY_PUT_ACK)
        _seq, _status, item_id = _PUT_ACK.unpack_from(body)
        return int(item_id)

    def get(self, *, newer_than: int = 0, wait_s: float = 0.0) -> Item | None:
        payload = _TAKE.pack(int(newer_than), 0, 1, 1 << 40, int(max(0.0, wait_s) * 1000))
        body = self.conn.request(MSG_RELAY_TAKE, payload, expect=MSG_RELAY_ITEMS,
                                 limit=(64 << 20) + 4096, timeout=self.conn.timeout + wait_s)
        (n,) = struct.unpack_from("<I", body)
        if n == 0:
            return None
        item_id, producer, length = _ITEM.unpack_from(body, 4 + _DEPTH.size)
        start = 4 + _DEPTH.size + _ITEM.size
        return Item(item_id, producer, body[start:start + length])

    def wait(self, *, timeout_s: float, poll_s: float = 5.0, sleep=time.sleep) -> Item:
        """The board's value, waiting up to ``timeout_s`` for the first PUT."""
        deadline = time.monotonic() + float(timeout_s)
        while True:
            try:
                item = self.get(wait_s=min(poll_s, 50.0))
            except RelayRefused:
                raise
            except RelayError:
                item = None
                sleep(poll_s)
            if item is not None:
                return item
            if time.monotonic() >= deadline:
                raise TimeoutError(f"relay board {self.conn.channel} is still empty after {timeout_s:.0f} s")

    def close(self) -> None:
        self.conn.close()
