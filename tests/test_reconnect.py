"""SyncerClient reconnection tests against a minimal in-process fake server.

Pure Python sockets/threading; no Rust build. The fake server accepts a
connection group (HELLO + DATA_HELLOs), abruptly drops it, accepts the
redialed group, and exercises a pull -> push round-trip over the new
connection (reassembling striped chunks like the real syncer does).
"""

import queue
import socket
import struct
import threading
import time

import pytest

from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import (
    _CHUNK_HEAD,
    _HEADER,
    DTYPE_F32,
    MAGIC,
    MSG_CHUNK,
    MSG_DATA_HELLO,
    MSG_HELLO,
    MSG_PULL_REQ,
    MSG_PUSH_FRAGMENT,
    SyncerClient,
    encode_hello,
    read_frame,
    write_frame,
)

LEARNER_ID = 7


def make_layout() -> FragmentLayout:
    return FragmentLayout([Fragment(MERGE_RDA, [("model.body.weight", 4)])])


def make_listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(20)
    return listener


def accept_group(listener: socket.socket, num_streams: int):
    """Accept 1 control + num_streams data sockets; return (hello_payload, socks)."""
    socks, hello = [], None
    while len(socks) < 1 + num_streams:
        s, _ = listener.accept()
        s.settimeout(20)
        msg_type, payload = read_frame(s)
        if msg_type == MSG_HELLO:
            hello = payload
        else:
            assert msg_type == MSG_DATA_HELLO
            learner_id, _stream_idx = struct.unpack("<IH", payload)
            assert learner_id == LEARNER_ID
        socks.append(s)
    assert hello is not None, "group arrived without a HELLO"
    return hello, socks


def drop_group(socks) -> None:
    """Abruptly kill every socket of a group (syncer crash / WAN drop)."""
    for s in socks:
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        s.close()


def read_push(socks, timeout: float = 15.0) -> bytes:
    """Read frames off every socket of a group until one complete
    PUSH_FRAGMENT arrives, reassembling CHUNK envelopes by msg_id exactly
    like the syncer. Returns the inner PUSH_FRAGMENT payload."""
    frames: queue.Queue = queue.Queue()

    def reader(sock):
        try:
            while True:
                frames.put(read_frame(sock))
        except (OSError, ConnectionError, ValueError):
            pass

    for s in socks:
        threading.Thread(target=reader, args=(s,), daemon=True).start()

    reasm: dict[int, tuple[bytearray, list[int]]] = {}
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "no PUSH_FRAGMENT within timeout"
        msg_type, payload = frames.get(timeout=remaining)
        if msg_type == MSG_PUSH_FRAGMENT:
            return payload
        if msg_type != MSG_CHUNK:
            continue  # e.g. HEARTBEAT
        msg_id, total, offset = _CHUNK_HEAD.unpack_from(payload)
        data = payload[_CHUNK_HEAD.size :]
        if msg_id not in reasm:
            reasm[msg_id] = (bytearray(total), [0])
        buf, filled = reasm[msg_id]
        buf[offset : offset + len(data)] = data
        filled[0] += len(data)
        if filled[0] < total:
            continue
        del reasm[msg_id]
        magic, inner_type, length = _HEADER.unpack_from(bytes(buf[: _HEADER.size]))
        assert magic == MAGIC and length == total - _HEADER.size
        if inner_type == MSG_PUSH_FRAGMENT:
            return bytes(buf[_HEADER.size :])


def push_args(tensor_bytes: bytes, global_step: int = 5):
    return dict(
        fragment_id=0,
        global_step=global_step,
        base_version=4,
        local_step=10,
        c_steps=3,
        c_tokens=300,
        tensor_bytes=tensor_bytes,
    )


@pytest.mark.timeout(25)
def test_reconnect_after_drop_round_trip():
    listener = make_listener()
    port = listener.getsockname()[1]
    layout = make_layout()
    client = SyncerClient(
        ("127.0.0.1", port), LEARNER_ID, layout, dtype=DTYPE_F32, num_streams=2, connect_timeout=10
    )
    try:
        client.start()
        hello1, group1 = accept_group(listener, 2)
        assert hello1 == encode_hello(LEARNER_ID, DTYPE_F32, layout, 2)

        # Abrupt syncer death: every socket of the group drops at once.
        drop_group(group1)
        time.sleep(0.3)  # let the client's threads notice

        # During the outage the training loop keeps going: pushes must be
        # enqueued-or-dropped without raising, and check_health must not
        # raise while reconnection is still being attempted.
        tensor = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        for _ in range(5):
            client.push_fragment(**push_args(tensor))
            client.heartbeat(1)
            client.check_health()
            time.sleep(0.05)

        # The client must redial the whole group and re-HELLO on its own.
        hello2, group2 = accept_group(listener, 2)
        assert hello2 == encode_hello(LEARNER_ID, DTYPE_F32, layout, 2)

        # Post-reconnect round trip: PULL_REQ -> PUSH_FRAGMENT.
        write_frame(group2[0], MSG_PULL_REQ, struct.pack("<IQ", 0, 5))
        deadline = time.monotonic() + 10
        pulls = []
        while not pulls:
            assert time.monotonic() < deadline, "pull request never delivered"
            pulls = client.drain_pulls()
            time.sleep(0.02)
        assert (pulls[0].fragment_id, pulls[0].global_step) == (0, 5)

        client.heartbeat(11)
        client.push_fragment(**push_args(tensor))
        payload = read_push(group2)
        learner_id, fragment_id, global_step, base_version, local_step, c_steps, c_tokens = struct.unpack_from(
            "<IIQQQIQ", payload
        )
        assert (learner_id, fragment_id, global_step) == (LEARNER_ID, 0, 5)
        assert (base_version, local_step, c_steps, c_tokens) == (4, 10, 3, 300)
        assert payload[44:] == tensor
        client.check_health()
        drop_group(group2)
    finally:
        client.close()
        listener.close()


@pytest.mark.timeout(25)
def test_max_reconnects_exhausted_fails_health_check():
    listener = make_listener()
    port = listener.getsockname()[1]
    layout = make_layout()
    client = SyncerClient(
        ("127.0.0.1", port),
        LEARNER_ID,
        layout,
        dtype=DTYPE_F32,
        num_streams=1,
        connect_timeout=10,
        max_reconnects=0,
    )
    try:
        client.start()
        _, group1 = accept_group(listener, 1)
        client.check_health()  # healthy while connected

        drop_group(group1)

        # No reconnect budget: the drop must surface through check_health,
        # but pushes must still not raise.
        deadline = time.monotonic() + 10
        while True:
            client.push_fragment(**push_args(b"\x00" * 16))
            try:
                client.check_health()
            except RuntimeError:
                break
            assert time.monotonic() < deadline, "check_health never raised"
            time.sleep(0.05)
    finally:
        client.close()
        listener.close()
