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
    MSG_BCAST_FRAGMENT,
    MSG_CHUNK,
    MSG_DATA_HELLO,
    MSG_ERROR,
    MSG_HELLO,
    MSG_PULL_REQ,
    MSG_PUSH_FRAGMENT,
    PROTOCOL_VERSION,
    ProtocolError,
    SyncerClient,
    encode_hello,
    layout_fingerprint,
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
            data_hellos.append(struct.unpack("<HIQH", payload))
        socks.append(s)
    assert hello is not None, "group arrived without a HELLO"
    version, learner_id, generation, _dtype, _fragments = struct.unpack_from(
        "<HIQBI", hello
    )
    assert (version, learner_id) == (PROTOCOL_VERSION, LEARNER_ID)
    assert generation != 0
    assert all(
        (data_version, data_id, data_generation)
        == (PROTOCOL_VERSION, LEARNER_ID, generation)
        for data_version, data_id, data_generation, _index in data_hellos
    )
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
        ("127.0.0.1", 1), LEARNER_ID, layout, dtype=DTYPE_F32, num_streams=0
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
        ("127.0.0.1", 1), LEARNER_ID, layout, dtype=DTYPE_F32, num_streams=0
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
        ("127.0.0.1", 1), LEARNER_ID, layout, dtype=DTYPE_F32, num_streams=0
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
        ("127.0.0.1", port), LEARNER_ID, layout, dtype=DTYPE_F32, num_streams=2, connect_timeout=10
    )
    try:
        client.start()
        hello1, generation1, group1 = accept_group(listener, 2)
        assert hello1 == encode_hello(
            LEARNER_ID, DTYPE_F32, layout, 2, generation1
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
            LEARNER_ID, DTYPE_F32, layout, 2, generation2
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


def test_syncer_sockets_use_tcp_keepalive():
    """A silently dropped idle flow (NAT) must surface as a socket error, which
    test_max_reconnects_exhausted_fails_health_check shows becomes a health failure
    (an earlier run hung 30+ min on such a flow)."""
    import socket as _socket

    from yeto import protocol

    listener = make_listener()
    port = listener.getsockname()[1]
    client = SyncerClient(("127.0.0.1", port), LEARNER_ID, make_layout(), dtype=DTYPE_F32,
                          num_streams=1, connect_timeout=10, max_reconnects=0)
    try:
        client.start()
        accept_group(listener, 1)
        for sock in client._socks:
            assert sock.getsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE)  # macOS reports the flag bit (8), Linux 1
            if hasattr(_socket, "TCP_KEEPIDLE"):
                assert sock.getsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPIDLE) == protocol.TCP_KEEPIDLE_S
                assert sock.getsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL) == protocol.TCP_KEEPINTVL_S
                assert sock.getsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT) == protocol.TCP_KEEPCNT
        assert protocol.TCP_KEEPIDLE_S + protocol.TCP_KEEPINTVL_S * protocol.TCP_KEEPCNT < 300
    finally:
        client.close()
        listener.close()


@pytest.mark.timeout(25)
def test_lease_expired_error_redials_with_new_generation_even_without_reconnect_budget():
    """The syncer expires a lease with MSG_ERROR "island lease expired: ...".
    The client treats it as a re-JOIN request: it redials with a new
    connection generation instead of failing, without using max_reconnects."""
    listener = make_listener()
    port = listener.getsockname()[1]
    client = SyncerClient(
        ("127.0.0.1", port), LEARNER_ID, make_layout(), dtype=DTYPE_F32, num_streams=1,
        connect_timeout=10, max_reconnects=0,
    )
    try:
        client.start()
        _hello, first_generation, socks = accept_group(listener, 1)
        write_frame(socks[0], MSG_ERROR, b"island lease expired: re-JOIN required")
        _hello, second_generation, socks2 = accept_group(listener, 1)
        assert second_generation != first_generation
        assert client.lease_expiries == 1
        drop_group(socks)
        client.check_health()  # a re-JOIN is not a failure
        # The re-joined connection works: a pull is answered on it.
        write_frame(socks2[0], MSG_PULL_REQ, struct.pack("<IQI", 0, 1, 1))
        deadline = time.monotonic() + 5
        while not client.drain_pulls():
            assert time.monotonic() < deadline, "pull on the re-joined generation not received"
            time.sleep(0.01)
        drop_group(socks2)
    finally:
        client.close()
        listener.close()


def _redial(client):
    """What _connect_group does for a redial: every fragment awaits its catch-up."""
    with client._bcast_lock:
        client._catch_up_pending = [True] * client.layout.num_fragments


def test_first_broadcast_after_a_redial_is_flagged_and_a_rewound_one_is_taken_on_request():
    layout = make_layout()
    v2 = struct.pack("<4f", 2.0, 2.0, 2.0, 2.0)
    v1 = struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)
    for accept in (False, True):
        client = SyncerClient(("127.0.0.1", 1), LEARNER_ID, layout, dtype=DTYPE_F32, num_streams=0,
                              accept_rewound_catch_up=accept)
        client._gen = 1
        client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 2) + v2)
        assert [(u.version, u.catch_up) for u in client.drain_updates()] == [(2, False)]
        _redial(client)
        # The syncer resumed from an older checkpoint: its catch-up is version 1.
        client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 1) + v1)
        got = [(u.version, u.data, u.catch_up) for u in client.drain_updates()]
        assert got == ([(1, v1, True)] if accept else [])
        # Only the first broadcast after the redial may go backwards.
        client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 0) + v1)
        assert client.drain_updates() == []
        # A replay of the version held is dropped (the island keeps its values).
        _redial(client)
        held = (1, v1) if accept else (2, v2)
        client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, held[0]) + held[1])
        assert client.drain_updates() == []
        # A newer one is the catch-up of a merge the island missed.
        _redial(client)
        client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 5) + v2)
        assert [(u.version, u.catch_up) for u in client.drain_updates()] == [(5, True)]
        client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 6) + v1)
        assert [(u.version, u.catch_up) for u in client.drain_updates()] == [(6, False)]
        assert client.rewound_catch_ups == int(accept)


def test_a_rewound_catch_up_with_the_same_version_and_other_bytes_is_taken_on_request():
    layout = make_layout()
    a, b = struct.pack("<4f", 2.0, 2.0, 2.0, 2.0), struct.pack("<4f", 3.0, 3.0, 3.0, 3.0)
    client = SyncerClient(("127.0.0.1", 1), LEARNER_ID, layout, dtype=DTYPE_F32, num_streams=0,
                          accept_rewound_catch_up=True)
    client._gen = 1
    client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 3) + a)
    client.drain_updates()
    with pytest.raises(ProtocolError, match="conflicting BCAST_FRAGMENT"):  # not after a redial
        client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 3) + b)
    _redial(client)
    client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 3) + b)
    assert [(u.version, u.data, u.catch_up) for u in client.drain_updates()] == [(3, b, True)]


def _chunk_frames(fid, version, data, msg_id=7):
    """The CHUNK payloads of one BCAST_FRAGMENT, as the syncer stripes them."""
    from yeto.protocol import _CHUNK_HEAD, _HEADER, CHUNK_SIZE, MAGIC

    inner = _HEADER.pack(MAGIC, MSG_BCAST_FRAGMENT, 12 + len(data)) + struct.pack("<IQ", fid, version) + data
    return [bytearray(_CHUNK_HEAD.pack(msg_id, len(inner), off) + inner[off:off + CHUNK_SIZE])
            for off in range(0, len(inner), CHUNK_SIZE)]


def test_block_digest_is_computed_while_chunks_arrive_in_any_order():
    import random

    from yeto.protocol import BCAST_DIGEST_BLOCK, bcast_payload_digest

    numel = (2 * BCAST_DIGEST_BLOCK + 3 * 1024 * 1024 + 4) // 4  # not a whole number of blocks
    layout = FragmentLayout([Fragment(MERGE_RDA, [("w", numel)])])
    data = bytes(random.Random(0).randbytes(4 * numel))
    want = bcast_payload_digest(data)
    assert want != bcast_payload_digest(data[:-1] + b"\x01")
    for seed in (1, 2):
        client = SyncerClient(("127.0.0.1", 1), LEARNER_ID, layout, dtype=DTYPE_F32, num_streams=0)
        client._gen = 1
        seen = []
        client._dispatch = lambda gen, msg_type, payload, digest=None: seen.append((bytes(payload[12:]), digest))
        frames = _chunk_frames(0, 3, data)
        random.Random(seed).shuffle(frames)
        out = [client._reassemble(1, frame) for frame in frames]
        assert all(o is None for o in out[:-1])
        client._dispatch(1, *out[-1])
        assert seen == [(data, want)]
        client.close()


def test_chunked_and_direct_broadcasts_share_the_replay_and_conflict_checks():
    from yeto.protocol import BCAST_DIGEST_BLOCK

    numel = 2 * BCAST_DIGEST_BLOCK // 4
    layout = FragmentLayout([Fragment(MERGE_RDA, [("w", numel)])])
    data = bytes(4 * numel)
    other = b"\x01" + data[1:]
    client = SyncerClient(("127.0.0.1", 1), LEARNER_ID, layout, dtype=DTYPE_F32, num_streams=0)
    client._gen = 1
    for frame in _chunk_frames(0, 3, data):
        inner = client._reassemble(1, frame)
    client._dispatch(1, *inner)
    assert [u.version for u in client.drain_updates()] == [3]
    # The same version again, unchunked: a replay, dropped.
    client._dispatch(1, MSG_BCAST_FRAGMENT, struct.pack("<IQ", 0, 3) + data)
    assert client.drain_updates() == []
    # Other bytes under it, chunked: a conflict.
    for frame in _chunk_frames(0, 3, other, msg_id=8):
        inner = client._reassemble(1, frame)
    with pytest.raises(ProtocolError, match="conflicting BCAST_FRAGMENT"):
        client._dispatch(1, *inner)
    client.close()


def test_subclass_dispatch_overrides_take_the_block_digest():
    """A reassembled BCAST of two or more digest blocks reaches ``_dispatch``
    with its block digest; every override must accept it."""
    import inspect

    from yeto.protocol import SyncerClient
    from yeto.rl.spot_rollout.subscriber import SubscriberClient

    for cls in (SyncerClient, SubscriberClient):
        inspect.signature(cls._dispatch).bind(None, 0, 0, b"", b"\0" * 32)


@pytest.mark.timeout(30)
def test_flapping_syncer_keeps_backing_off_instead_of_redialing_at_the_start_interval(monkeypatch):
    """A syncer that accepts the group and drops it right away (crash loop,
    flapping link) must not reset the backoff on every accepted dial: the
    gap between redials keeps growing, otherwise every island hammers the
    syncer and floods the log at the start interval forever."""
    from yeto import protocol

    monkeypatch.setattr(protocol, "RECONNECT_BACKOFF_START", 0.1)
    monkeypatch.setattr(protocol, "RECONNECT_BACKOFF_CAP", 2.0)
    monkeypatch.setattr(protocol, "RECONNECT_STABLE_S", 30.0)
    listener = make_listener()
    port = listener.getsockname()[1]
    client = SyncerClient(("127.0.0.1", port), LEARNER_ID, make_layout(), dtype=DTYPE_F32,
                          num_streams=1, connect_timeout=10, max_reconnects=None)
    try:
        client.start()
        stamps = []
        for _ in range(5):
            _, _, group = accept_group(listener, 1)
            stamps.append(time.monotonic())
            drop_group(group)
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert gaps[-1] > 0.6, f"backoff reset on every flap: gaps {gaps}"
        client.check_health()
    finally:
        client.close()
        listener.close()


@pytest.mark.timeout(30)
def test_reconnect_budget_counts_consecutive_failures_not_lifetime_redials(monkeypatch):
    """max_reconnects (the island pool's max_rejoin_failures) bounds
    *consecutive* failed redials: a link that came back and stayed up
    restores the full budget, so a few blips over a multi-day run do not
    eventually kill a healthy island."""
    from yeto import protocol

    monkeypatch.setattr(protocol, "RECONNECT_BACKOFF_START", 0.05)
    monkeypatch.setattr(protocol, "RECONNECT_STABLE_S", 0.3)
    listener = make_listener()
    port = listener.getsockname()[1]
    client = SyncerClient(("127.0.0.1", port), LEARNER_ID, make_layout(), dtype=DTYPE_F32,
                          num_streams=1, connect_timeout=10, max_reconnects=1)
    try:
        client.start()
        _, _, group = accept_group(listener, 1)
        for _ in range(3):  # three separate blips, each after a stable period
            time.sleep(0.6)
            drop_group(group)
            _, _, group = accept_group(listener, 1)
            client.check_health()
        assert client.send_progress().reconnects == 3
        drop_group(group)
    finally:
        client.close()
        listener.close()


def test_enqueue_racing_a_teardown_leaves_no_phantom_backlog_for_the_stall_watchdog():
    """A sender that picked its queue just before a teardown must not leave
    bytes counted on the (session-long) stream meter after drop_pending: the
    next group would look stalled and be dropped by the watchdog."""
    import queue as _queue

    from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
    from yeto.protocol import DTYPE_F32, SyncerClient

    client = SyncerClient(
        ("unused", 0), 3,
        FragmentLayout([Fragment(MERGE_RDA, [("w", 4)])]),
        dtype=DTYPE_F32, num_streams=0, stream_stall_timeout=5.0,
    )

    meter = client._meters[0]
    real_enqueued = meter.enqueued
    at_meter = threading.Event()
    torn_down = threading.Event()

    def enqueued_after_teardown(nbytes):
        if not at_meter.is_set():  # the queue is picked; let the outage land
            at_meter.set()
            torn_down.wait(0.5)  # a teardown blocked on the client lock times out
        real_enqueued(nbytes)

    def teardown():
        at_meter.wait(5.0)
        client._teardown_group()
        torn_down.set()

    meter.enqueued = enqueued_after_teardown
    client._gen = 1
    client._connected.set()
    client._queues = [_queue.Queue()]
    tearer = threading.Thread(target=teardown)
    tearer.start()
    try:
        client._enqueue(0, b"x" * 64)
    finally:
        tearer.join(5.0)
    assert not tearer.is_alive() and torn_down.is_set()
    assert meter.idle
    assert meter.stalled_for(time.monotonic() + 3600) == 0.0
    assert client.send_progress().queued_chunks == 0
