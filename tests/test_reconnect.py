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
    MSG_BCAST_FRAGMENT,
    MSG_DATA_HELLO,
    MSG_ERROR,
    MSG_HELLO,
    MSG_PULL_REQ,
    MSG_PUSH_FRAGMENT,
    PROTOCOL_VERSION,
    ProtocolError,
    SyncerClient,
    encode_hello,
    read_frame,
    write_frame,
)

LEARNER_ID = 7
RUN_ID = bytes(range(32))


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
    """Accept a complete versioned connection group."""
    socks, hello, data_hellos = [], None, []
    while len(socks) < 1 + num_streams:
        s, _ = listener.accept()
        s.settimeout(20)
        msg_type, payload = read_frame(s)
        if msg_type == MSG_HELLO:
            hello = payload
        else:
            assert msg_type == MSG_DATA_HELLO
            data_hellos.append(struct.unpack("<HIQ32sH", payload))
        socks.append(s)
    assert hello is not None, "group arrived without a HELLO"
    version, learner_id, generation, run_id, _dtype, _fragments = struct.unpack_from(
        "<HIQ32sBI", hello
    )
    assert (version, learner_id) == (PROTOCOL_VERSION, LEARNER_ID)
    assert run_id == RUN_ID
    assert generation != 0
    assert all(
        (data_version, data_id, data_generation)
        == (PROTOCOL_VERSION, LEARNER_ID, generation)
        for data_version, data_id, data_generation, data_run_id, _index in data_hellos
    )
    assert all(item[3] == RUN_ID for item in data_hellos)
    return hello, generation, socks


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
        round_attempt=1,
        base_version=4,
        local_step=10,
        c_steps=3,
        c_tokens=300,
        tensor_bytes=tensor_bytes,
    )


def test_broadcast_versions_are_monotonic_per_fragment():
    layout = make_layout()
    client = SyncerClient(
        ("127.0.0.1", 1),
        LEARNER_ID,
        layout,
        dtype=DTYPE_F32,
        num_streams=0,
        run_id=RUN_ID,
        allow_insecure_loopback=True,
    )
    client._gen = 1
    newer_data = struct.pack("<4f", 2.0, 2.0, 2.0, 2.0)
    older_data = struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)
    client._dispatch(
        1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 2) + newer_data
    )
    client._dispatch(
        1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 1) + older_data
    )
    # An identical same-version replay is idempotent and not queued twice.
    client._dispatch(
        1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 2) + newer_data
    )
    updates = client.drain_updates()
    assert [(update.version, update.data) for update in updates] == [(2, newer_data)]
    with pytest.raises(ProtocolError, match="conflicting BCAST_FRAGMENT"):
        client._dispatch(
            1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 2) + older_data
        )


def test_concurrent_broadcasts_enqueue_in_version_order():
    layout = make_layout()
    client = SyncerClient(
        ("127.0.0.1", 1),
        LEARNER_ID,
        layout,
        dtype=DTYPE_F32,
        num_streams=0,
        run_id=RUN_ID,
        allow_insecure_loopback=True,
    )
    client._gen = 1

    class BlockingBroadcastQueue:
        def __init__(self):
            self.items = []
            self.older_put_started = threading.Event()
            self.release_older = threading.Event()
            self.put_without_version_lock = threading.Event()

        def put(self, item):
            if not client._bcast_lock.locked():
                self.put_without_version_lock.set()
            if item.version == 2:
                self.older_put_started.set()
                assert self.release_older.wait(timeout=5)
            self.items.append(item)

    broadcasts = BlockingBroadcastQueue()
    client._bcasts = broadcasts
    data_v2 = struct.pack("<4f", 2.0, 2.0, 2.0, 2.0)
    data_v3 = struct.pack("<4f", 3.0, 3.0, 3.0, 3.0)
    dispatch_v2 = threading.Thread(
        target=client._dispatch,
        args=(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 2) + data_v2),
    )
    dispatch_v3 = threading.Thread(
        target=client._dispatch,
        args=(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 3) + data_v3),
    )
    dispatch_v2.start()
    assert broadcasts.older_put_started.wait(timeout=5)
    dispatch_v3.start()
    broadcasts.release_older.set()
    dispatch_v2.join(timeout=5)
    dispatch_v3.join(timeout=5)

    assert not dispatch_v2.is_alive()
    assert not dispatch_v3.is_alive()
    assert not broadcasts.put_without_version_lock.is_set()
    assert [update.version for update in broadcasts.items] == [2, 3]


def test_client_rejects_huge_broadcast_header_before_reading_payload():
    layout = make_layout()
    client = SyncerClient(
        ("127.0.0.1", 1),
        LEARNER_ID,
        layout,
        dtype=DTYPE_F32,
        num_streams=0,
        run_id=RUN_ID,
        allow_insecure_loopback=True,
    )
    reader, writer = socket.socketpair()
    try:
        writer.sendall(
            _HEADER.pack(MAGIC, MSG_BCAST_FRAGMENT, client._max_bcast_payload + 1)
        )
        with pytest.raises(ValueError, match="exceeds limit"):
            read_frame(reader, client._incoming_frame_limit)
    finally:
        reader.close()
        writer.close()


@pytest.mark.timeout(10)
def test_server_error_becomes_clear_fatal_protocol_error():
    listener = make_listener()
    port = listener.getsockname()[1]
    layout = make_layout()
    client = SyncerClient(
        ("127.0.0.1", port),
        LEARNER_ID,
        layout,
        dtype=DTYPE_F32,
        num_streams=0,
        connect_timeout=5,
        run_id=RUN_ID,
        allow_insecure_loopback=True,
    )

    def reject():
        sock, _ = listener.accept()
        try:
            msg_type, _payload = read_frame(sock)
            assert msg_type == MSG_HELLO
            write_frame(sock, MSG_ERROR, b"wire protocol version mismatch")
            time.sleep(0.1)
        finally:
            sock.close()

    thread = threading.Thread(target=reject, daemon=True)
    thread.start()
    try:
        client.start()
        deadline = time.monotonic() + 5
        while True:
            try:
                client.check_health()
            except RuntimeError as error:
                assert "syncer connection failed" in str(error)
                assert isinstance(error.__cause__, ProtocolError)
                assert "wire protocol version mismatch" in str(error.__cause__)
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        client.close()
        listener.close()
        thread.join(timeout=1)


@pytest.mark.timeout(25)
def test_reconnect_after_drop_round_trip():
    listener = make_listener()
    port = listener.getsockname()[1]
    layout = make_layout()
    client = SyncerClient(
        ("127.0.0.1", port),
        LEARNER_ID,
        layout,
        dtype=DTYPE_F32,
        num_streams=2,
        connect_timeout=10,
        run_id=RUN_ID,
        allow_insecure_loopback=True,
    )
    try:
        client.start()
        hello1, generation1, group1 = accept_group(listener, 2)
        assert hello1 == encode_hello(
            LEARNER_ID, DTYPE_F32, layout, 2, generation1, RUN_ID
        )

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
        hello2, generation2, group2 = accept_group(listener, 2)
        assert hello2 == encode_hello(
            LEARNER_ID, DTYPE_F32, layout, 2, generation2, RUN_ID
        )
        assert generation2 != generation1

        # Post-reconnect round trip: PULL_REQ -> PUSH_FRAGMENT.
        write_frame(group2[0], MSG_PULL_REQ, struct.pack("<IQI", 0, 5, 1))
        deadline = time.monotonic() + 10
        pulls = []
        while not pulls:
            assert time.monotonic() < deadline, "pull request never delivered"
            pulls = client.drain_pulls()
            time.sleep(0.02)
        assert (
            pulls[0].fragment_id,
            pulls[0].global_step,
            pulls[0].round_attempt,
        ) == (0, 5, 1)

        client.heartbeat(11)
        client.push_fragment(**push_args(tensor))
        payload = read_push(group2)
        learner_id, fragment_id, global_step, attempt, base_version, local_step, c_steps, c_tokens = struct.unpack_from(
            "<IIQIQQIQ", payload
        )
        assert (learner_id, fragment_id, global_step) == (LEARNER_ID, 0, 5)
        assert attempt == 1
        assert (base_version, local_step, c_steps, c_tokens) == (4, 10, 3, 300)
        assert payload[48:] == tensor
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
        run_id=RUN_ID,
        allow_insecure_loopback=True,
    )
    try:
        client.start()
        _, _, group1 = accept_group(listener, 1)
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
