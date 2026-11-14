"""A short or overrunning frame from the relay peer is a ``RelayError``.

The relay client decodes WELCOME, PUT_ACK, ITEMS and board answers from peer
bytes; a frame shorter than its layout (or an ITEMS answer declaring more items
than it carries) must end the connection as a retryable ``RelayError`` instead
of escaping as ``struct.error`` past the retry loops and ``RelayBoard.wait``.
A short MEMBER_RETIRED answer is a ``ConnectionError`` like any bad frame.
"""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from yeto.protocol import MAGIC, _HEADER, read_frame, write_frame
from yeto.rl.engine import task_dispense
from yeto.rl.spot_rollout import relay
from yeto.rl.spot_rollout.relay import (
    MSG_RELAY_ITEMS,
    MSG_RELAY_PUT_ACK,
    MSG_RELAY_WELCOME,
    RELAY_VERSION,
    RelayBoard,
    RelayConsumer,
    RelayError,
    RelayProducer,
)

DEPTH = relay._DEPTH.pack(0, 0, 0, 1)
GOOD_WELCOME = relay._WELCOME.pack(RELAY_VERSION, 7, 1 << 20, 64, 1 << 30) + DEPTH


def items(*entries: tuple[int, bytes], declared: int | None = None) -> bytes:
    n = len(entries) if declared is None else declared
    out = [struct.pack("<I", n), DEPTH]
    for item_id, data in entries:
        out.append(relay._ITEM.pack(item_id, 1, len(data)) + data)
    return b"".join(out)


class ScriptedRelay:
    """Answers connection ``i`` with ``script[i]``: a list of (type, payload),
    the first answering RELAY_HELLO and each next one the next request."""

    def __init__(self, script: list[list[tuple[int, bytes]]]) -> None:
        self.script = list(script)
        self.connections = 0
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.settimeout(0.2)
        self.addr = self.listener.getsockname()[:2]
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="scripted-relay", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                conn, _ = self.listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            answers = self.script.pop(0) if self.script else []
            self.connections += 1
            with conn:
                conn.settimeout(5.0)
                try:
                    for msg_type, payload in answers:
                        read_frame(conn, 1 << 20)
                        write_frame(conn, msg_type, payload)
                    read_frame(conn, 1 << 20)  # hold the connection until the client drops it
                except (OSError, ConnectionError, ValueError):
                    pass

    def close(self) -> None:
        self.stopping.set()
        self.listener.close()
        self.thread.join(timeout=5.0)
        assert not self.thread.is_alive()


@pytest.fixture
def scripted():
    servers = []

    def make(script):
        server = ScriptedRelay(script)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.close()


def test_short_welcome_is_a_relay_error_and_the_producer_retries(scripted):
    ack = relay._PUT_ACK.pack(1, relay.PUT_STORED, 5) + DEPTH
    server = scripted([[(MSG_RELAY_WELCOME, GOOD_WELCOME[:10])], [(MSG_RELAY_WELCOME, GOOD_WELCOME[:10])],
                       [(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_PUT_ACK, ack)]])
    conn = relay.RelayConnection(server.addr, relay.ROLE_PRODUCER, 0, 1, key=None, timeout=5.0)
    with pytest.raises(RelayError, match="short relay WELCOME"):
        conn.connect()
    assert conn.sock is None

    sleeps = []
    producer = RelayProducer(server.addr, 0, 1, key=None, timeout=5.0, sleep=sleeps.append,
                             incarnation=3, max_attempts=3)
    try:
        got = producer.put(b"x")
    finally:
        producer.close()
    assert got.item_id == 5 and sleeps == [producer.retry_s]
    assert server.connections == 3


def test_welcome_with_a_short_depth_is_a_relay_error(scripted):
    server = scripted([[(MSG_RELAY_WELCOME, GOOD_WELCOME[:-1])]])
    conn = relay.RelayConnection(server.addr, relay.ROLE_CONSUMER, 0, 0, key=None, timeout=5.0)
    with pytest.raises(RelayError, match="short relay depth"):
        conn.connect()
    assert conn.sock is None


def test_short_put_ack_is_a_relay_error_and_the_producer_retries(scripted):
    ack = relay._PUT_ACK.pack(1, relay.PUT_STORED, 9) + DEPTH
    server = scripted([[(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_PUT_ACK, ack[:5])],
                       [(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_PUT_ACK, ack)]])
    sleeps = []
    producer = RelayProducer(server.addr, 0, 1, key=None, timeout=5.0, sleep=sleeps.append,
                             incarnation=3, max_attempts=3)
    try:
        assert producer.put(b"x").item_id == 9
    finally:
        producer.close()
    assert len(sleeps) == 1 and server.connections == 2


def test_items_declaring_more_than_they_carry_retry_then_deliver(scripted):
    server = scripted([[(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_ITEMS, items((1, b"ab"), declared=3))],
                       [(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_ITEMS, items((1, b"abc"))[:-1])],
                       [(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_ITEMS, items((1, b"ab"), (2, b"c")))]])
    sleeps = []
    consumer = RelayConsumer(server.addr, 0, key=None, timeout=5.0, sleep=sleeps.append, gone_s=None)
    try:
        got = consumer.take(wait_s=0.0)
    finally:
        consumer.close()
    assert [(i.item_id, i.data) for i in got] == [(1, b"ab"), (2, b"c")]
    assert len(sleeps) == 2 and server.connections == 3
    assert consumer.after == 2


def test_decode_items_rejects_short_frames():
    with pytest.raises(RelayError, match="ITEMS count"):
        relay._decode_items(b"\x01\x00")
    with pytest.raises(RelayError, match="depth"):
        relay._decode_items(struct.pack("<I", 0) + DEPTH[:-2])
    with pytest.raises(RelayError, match="item header"):
        relay._decode_items(items((1, b"a"), declared=2))
    with pytest.raises(RelayError, match="declares 4 bytes"):
        relay._decode_items(items((1, b"abcd"))[:-1])
    depth, got = relay._decode_items(items((1, b"a"), declared=5), limit=1)
    assert [i.data for i in got] == [b"a"] and depth.next_id == 1


def test_short_board_frame_is_a_relay_error_and_wait_retries(scripted):
    server = scripted([[(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_ITEMS, items((4, b"lay"))[:6])],
                       [(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_ITEMS, items((4, b"lay"), declared=1)[:-2])],
                       [(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_ITEMS, items((4, b"lay")))]])
    board = RelayBoard(server.addr, 2, key=None, timeout=5.0)
    try:
        with pytest.raises(RelayError, match="short relay depth"):
            board.get()
        assert board.conn.sock is None
        sleeps = []
        item = board.wait(timeout_s=30.0, poll_s=0.0, sleep=sleeps.append)
    finally:
        board.close()
    assert item.item_id == 4 and item.data == b"lay"
    assert len(sleeps) == 1 and server.connections == 3


def test_short_board_put_ack_is_a_relay_error(scripted):
    server = scripted([[(MSG_RELAY_WELCOME, GOOD_WELCOME), (MSG_RELAY_PUT_ACK, b"\x01")]])
    board = RelayBoard(server.addr, 2, key=None, timeout=5.0)
    try:
        with pytest.raises(RelayError, match="PUT_ACK"):
            board.set(b"v")
        assert board.conn.sock is None
    finally:
        board.close()


def test_short_member_retired_is_a_connection_error():
    listener = socket.create_server(("127.0.0.1", 0))
    addr = listener.getsockname()[:2]

    def serve() -> None:
        conn, _ = listener.accept()
        with conn:
            read_frame(conn, 1 << 16)
            body = b"\x01\x02\x03"
            conn.sendall(_HEADER.pack(MAGIC, task_dispense.MSG_MEMBER_RETIRED, len(body)) + body)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        with pytest.raises(ConnectionError, match="MEMBER_RETIRED has 3 bytes"):
            task_dispense.retire_member(addr, 1, None, key=None, timeout_s=5.0)
    finally:
        thread.join(timeout=5.0)
        listener.close()
    assert not thread.is_alive()
    with pytest.raises(ConnectionError):
        task_dispense.decode_member_retired(b"\x00" * 26)
    good = task_dispense._MEMBER_RETIRED.pack(1, 2, 3, 4)
    assert task_dispense.decode_member_retired(good)["incarnation"] == 2
