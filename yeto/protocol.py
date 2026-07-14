"""Python side of the learner <-> syncer wire protocol (docs/PROTOCOL.md v2).

A learner owns a connection *group*: stream 0 carries control messages
(HELLO, PULL_REQ, HEARTBEAT, SHUTDOWN); large payloads (INIT_PARAMS,
PUSH_FRAGMENT, BCAST_FRAGMENT) are split into 4 MiB CHUNK envelopes striped
round-robin across extra data sockets, which multiplies WAN throughput over
what a single congestion-window-limited TCP stream can carry.

The training loop must never block on the WAN, so each socket gets a sender
thread (draining a queue) and a receiver thread (reassembling chunks and
sorting inbound messages into pull-request / broadcast inboxes consumed at
inner-step boundaries).

The client survives socket failures: a supervisor thread tears down the whole
connection group when any socket errors, then redials with exponential
backoff and re-sends HELLO/DATA_HELLO. On HELLO the syncer rebroadcasts every
initialized fragment at its current version, so a reconnected learner is
caught up within one message exchange; anything queued toward the old
connection (pushes, heartbeats, pending pull requests, partial chunk
reassemblies) is stale by then and is dropped. Broadcasts already received
are kept. The training loop keeps calling push_fragment/heartbeat/drain_*
during an outage; those calls silently drop instead of raising, and
check_health only raises once reconnection has been given up for good.
"""

from __future__ import annotations

import itertools
import json
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass

from .fragments import MERGE_ISO, FragmentLayout

MAGIC = 0xD170C0DE

MSG_HELLO = 1
MSG_INIT_PARAMS = 2
MSG_PULL_REQ = 3
MSG_PUSH_FRAGMENT = 4
MSG_BCAST_FRAGMENT = 5
MSG_HEARTBEAT = 6
MSG_SHUTDOWN = 7
MSG_DATA_HELLO = 8
MSG_CHUNK = 9
# BCAST_CONTROL (SCAFFOLD-lite): the token-normalized mean control vector c for
# one fragment, broadcast after its BCAST_FRAGMENT when the syncer runs with
# --inner-control-variate scaffold_lite. Same envelope as BCAST_FRAGMENT
# (fid u32, version u64, bulk-wire tensor bytes); off by default so non-scaffold
# sessions never see it. See yeto/scaffold.py.
MSG_BCAST_CONTROL = 10
# Identity-shuffle Option-II residual pair. Envelope: fid u32, version u64,
# residual byte length u64, then assigned-residual and shared-mean bytes.
MSG_BCAST_CONTROL_PAIR = 11

DTYPE_F32 = 1
DTYPE_BF16 = 2
# Session dtype 3: PUSH_FRAGMENT payloads are block-quantized 4-bit E3M0
# *deltas* against the fragment value at base_version; INIT_PARAMS and
# BCAST_FRAGMENT stay bf16 (see bulk_dtype and docs/PROTOCOL.md v3).
DTYPE_Q4 = 3

CHUNK_SIZE = 4 * 1024 * 1024

RECONNECT_BACKOFF_START = 1.0
RECONNECT_BACKOFF_CAP = 30.0
RECONNECT_DIAL_TIMEOUT = 20.0

_HEADER = struct.Struct("<IBQ")  # magic, type, payload length
_CHUNK_HEAD = struct.Struct("<QQQ")  # msg_id, total_len, offset


def bulk_dtype(dtype: int) -> int:
    """Dtype of INIT_PARAMS/BCAST_FRAGMENT tensors for a session dtype.

    Q4 applies only to push deltas (small dynamic range); full parameter
    payloads would not survive 4 bits, so they travel as bf16.
    """
    return DTYPE_BF16 if dtype == DTYPE_Q4 else dtype


def write_frame(sock: socket.socket, msg_type: int, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(MAGIC, msg_type, len(payload)) + payload)


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("syncer closed connection")
        got += r
    return bytes(buf)


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    magic, msg_type, length = _HEADER.unpack(_read_exact(sock, _HEADER.size))
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    return msg_type, _read_exact(sock, length)


def encode_hello(
    learner_id: int,
    dtype: int,
    layout: FragmentLayout,
    num_streams: int,
    layout_metadata: dict | bytes | str | None = None,
) -> bytes:
    parts = [struct.pack("<IBI", learner_id, dtype, layout.num_fragments)]
    for frag in layout.fragments:
        parts.append(struct.pack("<BI", frag.merge_mode, len(frag.tensors)))
        parts.append(struct.pack(f"<{len(frag.tensors)}Q", *(n for _, n in frag.tensors)))
        if frag.merge_mode == MERGE_ISO:
            # Iso fragments append (rows, cols) per tensor so the syncer can
            # take the 2D view; avg/RDA keep the original wire format.
            dims = [d for name, _ in frag.tensors for d in frag.shapes[name]]
            parts.append(struct.pack(f"<{len(dims)}Q", *dims))
    parts.append(struct.pack("<H", num_streams))
    if layout_metadata is not None:
        if isinstance(layout_metadata, bytes):
            meta = layout_metadata
        elif isinstance(layout_metadata, str):
            meta = layout_metadata.encode("utf-8")
        else:
            meta = json.dumps(layout_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        parts.append(struct.pack("<I", len(meta)))
        parts.append(meta)
    return b"".join(parts)


@dataclass
class PullRequest:
    fragment_id: int
    global_step: int


@dataclass
class BcastFragment:
    fragment_id: int
    version: int
    data: bytes  # raw tensor bytes in the session dtype


@dataclass
class BcastControlPair:
    fragment_id: int
    version: int
    local_data: bytes
    mean_data: bytes


class SyncerClient:
    """Non-blocking striped syncer connection owned by one learner process.

    Self-healing: any socket error tears down and redials the whole group.
    ``max_reconnects`` bounds the number of redial attempts (None = retry
    forever); once exhausted, check_health raises.
    """

    def __init__(
        self,
        addr: tuple[str, int],
        learner_id: int,
        layout: FragmentLayout,
        dtype: int = DTYPE_BF16,
        num_streams: int = 4,
        layout_metadata: dict | bytes | str | None = None,
        connect_timeout: float = 900.0,
        max_reconnects: int | None = None,
    ):
        self.addr = addr
        self.learner_id = learner_id
        self.layout = layout
        self.dtype = dtype
        self.num_streams = num_streams
        self.layout_metadata = layout_metadata
        self.connect_timeout = connect_timeout
        self.max_reconnects = max_reconnects
        self._queues: list[queue.Queue[bytes | None]] = []
        self._socks: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        self._pulls: queue.Queue[PullRequest] = queue.Queue()
        self._bcasts: queue.Queue[BcastFragment] = queue.Queue()
        # SCAFFOLD-lite mean-control broadcasts (MSG_BCAST_CONTROL). Empty and
        # never fed unless the syncer runs with control variates enabled.
        self._controls: queue.Queue[BcastFragment] = queue.Queue()
        self._control_pairs: queue.Queue[BcastControlPair] = queue.Queue()
        self._reasm: dict[int, tuple[bytearray, list[int]]] = {}
        self._reasm_lock = threading.Lock()
        self._msg_id = itertools.count()
        self._rr = itertools.count()
        self._err: BaseException | None = None
        self._last_err: BaseException | None = None
        self.shutdown = threading.Event()
        # Connection-group generation. Bumped on every teardown/redial so
        # threads and messages belonging to a dead group can be recognized
        # and ignored. Guarded by _lock together with _socks/_queues.
        self._lock = threading.RLock()
        self._gen = 0
        self._connected = threading.Event()
        self._failure = threading.Event()  # set by any thread of the live group
        self._closed = threading.Event()
        self._reconnects_used = 0
        self._supervisor: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------

    def _dial(self, timeout: float) -> socket.socket:
        sock = socket.create_connection(self.addr, timeout=timeout)
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    def _connect_one(self) -> socket.socket:
        last: OSError | None = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.connect_timeout:
            try:
                return self._dial(30)
            except OSError as e:  # syncer may not be up yet
                last = e
                time.sleep(2.0)
        raise ConnectionError(f"cannot reach syncer at {self.addr}: {last}")

    def start(self) -> None:
        self._connect_group(patient=True)
        self._supervisor = threading.Thread(target=self._supervise, name="yeto-supervisor", daemon=True)
        self._supervisor.start()

    def _connect_group(self, patient: bool) -> None:
        """Dial control + data sockets, send HELLO/DATA_HELLO, start workers.

        ``patient`` retries each dial for connect_timeout (initial startup,
        where the syncer may not be up yet); otherwise each socket gets a
        single attempt and the supervisor's backoff handles retries.
        """
        socks: list[socket.socket] = []
        try:
            control = self._connect_one() if patient else self._dial(RECONNECT_DIAL_TIMEOUT)
            write_frame(
                control,
                MSG_HELLO,
                encode_hello(
                    self.learner_id,
                    self.dtype,
                    self.layout,
                    self.num_streams,
                    self.layout_metadata,
                ),
            )
            socks.append(control)
            for idx in range(self.num_streams):
                s = self._connect_one() if patient else self._dial(RECONNECT_DIAL_TIMEOUT)
                write_frame(s, MSG_DATA_HELLO, struct.pack("<IH", self.learner_id, idx))
                socks.append(s)
        except BaseException:
            for s in socks:
                _close_socket(s)
            raise
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._socks = socks
            self._queues = []
            self._threads = []
            self._failure.clear()
            self._connected.set()
            for i, s in enumerate(socks):
                q: queue.Queue[bytes | None] = queue.Queue(maxsize=256)
                self._queues.append(q)
                ts = threading.Thread(
                    target=self._send_loop, args=(gen, s, q), name=f"yeto-send-{i}", daemon=True
                )
                tr = threading.Thread(target=self._recv_loop, args=(gen, s), name=f"yeto-recv-{i}", daemon=True)
                self._threads += [ts, tr]
                ts.start()
                tr.start()

    def _teardown_group(self) -> None:
        """Kill the current group's sockets/threads and drop state that a
        fresh HELLO makes obsolete (the syncer rebroadcasts every fragment at
        its current version on reconnect). Received broadcasts are kept."""
        with self._lock:
            self._gen += 1  # invalidates the old group's threads and messages
            self._connected.clear()
            socks, queues, threads = self._socks, self._queues, self._threads
            self._socks, self._queues, self._threads = [], [], []
        for q in queues:
            try:
                q.put_nowait(None)  # wake the sender; socket close covers a full queue
            except queue.Full:
                pass
        for s in socks:
            _close_socket(s)
        with self._reasm_lock:
            self._reasm.clear()  # partial inbound messages died with the sockets
        self._drain(self._pulls)  # stale pull requests; answering them is pointless
        for t in threads:
            t.join(timeout=5.0)

    def _supervise(self) -> None:
        """Reconnect loop: on any socket failure of the live group, tear the
        whole group down, back off exponentially, redial, resume."""
        while True:
            self._failure.wait()
            if self._closed.is_set() or self.shutdown.is_set():
                return
            self._teardown_group()
            backoff = RECONNECT_BACKOFF_START
            while True:
                if self._closed.is_set() or self.shutdown.is_set():
                    return
                if self.max_reconnects is not None and self._reconnects_used >= self.max_reconnects:
                    with self._lock:
                        self._err = self._last_err or ConnectionError("syncer connection lost")
                    return
                self._reconnects_used += 1
                if self._closed.wait(backoff):
                    return
                backoff = min(backoff * 2.0, RECONNECT_BACKOFF_CAP)
                try:
                    self._connect_group(patient=False)
                    break
                except (OSError, ConnectionError) as e:
                    self._last_err = e

    def close(self) -> None:
        self._closed.set()
        self._failure.set()  # unblock the supervisor so it exits instead of redialing
        with self._lock:
            queues, socks = list(self._queues), list(self._socks)
        for q in queues:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for s in socks:
            _close_socket(s)
        if self._supervisor is not None:
            self._supervisor.join(timeout=5.0)

    def check_health(self) -> None:
        """Raise only for unrecoverable failures. While a reconnect is being
        attempted this is a no-op; the training loop keeps stepping locally."""
        if self._err is not None:
            raise RuntimeError("syncer connection failed") from self._err

    # -- learner-facing API ------------------------------------------------------

    def send_init(self, fragment_id: int, tensor_bytes: bytes) -> None:
        self._send_large(MSG_INIT_PARAMS, struct.pack("<I", fragment_id) + tensor_bytes)

    def push_fragment(
        self,
        fragment_id: int,
        global_step: int,
        base_version: int,
        local_step: int,
        c_steps: int,
        c_tokens: int,
        tensor_bytes: bytes,
    ) -> None:
        head = struct.pack(
            "<IIQQQIQ",
            self.learner_id,
            fragment_id,
            global_step,
            base_version,
            local_step,
            c_steps,
            c_tokens,
        )
        self._send_large(MSG_PUSH_FRAGMENT, head + tensor_bytes)

    def heartbeat(self, local_step: int) -> None:
        self._enqueue(0, self._frame(MSG_HEARTBEAT, struct.pack("<IQ", self.learner_id, local_step)))

    def drain_pulls(self) -> list[PullRequest]:
        return self._drain(self._pulls)

    def drain_updates(self) -> list[BcastFragment]:
        return self._drain(self._bcasts)

    def drain_controls(self) -> list[BcastFragment]:
        """SCAFFOLD-lite mean-control (c) broadcasts received since the last
        drain. Always empty unless the syncer has control variates enabled."""
        return self._drain(self._controls)

    def drain_control_pairs(self) -> list[BcastControlPair]:
        """Identity-shuffled residual plus the shared residual mean."""
        return self._drain(self._control_pairs)

    @staticmethod
    def _drain(q: queue.Queue) -> list:
        out = []
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                return out

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _frame(msg_type: int, payload: bytes) -> bytes:
        return _HEADER.pack(MAGIC, msg_type, len(payload)) + payload

    def _enqueue(self, stream: int, data: bytes, gen: int | None = None) -> bool:
        """Enqueue-or-drop. Never raises and never blocks unboundedly: during
        an outage the message is dropped (the syncer's quorum tolerates
        missing responders, and learner counters only reset on broadcast
        receipt, so a lost push is retried implicitly by the next pull).
        ``gen`` pins the message to one connection group so a multi-chunk
        message never straddles two connections. Returns True if enqueued."""
        while True:
            with self._lock:
                if self._closed.is_set() or self.shutdown.is_set():
                    return False
                if not self._connected.is_set():
                    return False
                if gen is not None and gen != self._gen:
                    return False
                q = self._queues[stream]
            try:
                q.put(data, timeout=0.5)  # bounded wait, then re-check the group
                return True
            except queue.Full:
                continue

    def _send_large(self, msg_type: int, payload: bytes) -> None:
        inner = self._frame(msg_type, payload)
        if self.num_streams == 0:
            self._enqueue(0, inner)
            return
        with self._lock:
            if not self._connected.is_set():
                return  # outage: drop the whole message rather than send a torso
            gen = self._gen
        msg_id = next(self._msg_id)
        total = len(inner)
        for offset in range(0, total, CHUNK_SIZE):
            chunk = inner[offset : offset + CHUNK_SIZE]
            envelope = self._frame(MSG_CHUNK, _CHUNK_HEAD.pack(msg_id, total, offset) + chunk)
            stream = 1 + next(self._rr) % self.num_streams
            if not self._enqueue(stream, envelope, gen=gen):
                return  # group died mid-message; drop the remainder

    def _socket_failed(self, gen: int, exc: BaseException) -> None:
        if self.shutdown.is_set() or self._closed.is_set():
            return
        with self._lock:
            if gen != self._gen:
                return  # a thread of an already-torn-down group; ignore
            self._last_err = exc
            self._connected.clear()
            self._failure.set()

    def _send_loop(self, gen: int, sock: socket.socket, q: queue.Queue) -> None:
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                sock.sendall(item)
        except BaseException as e:
            self._socket_failed(gen, e)

    def _recv_loop(self, gen: int, sock: socket.socket) -> None:
        try:
            while True:
                msg_type, payload = read_frame(sock)
                if msg_type == MSG_CHUNK:
                    inner = self._reassemble(gen, payload)
                    if inner is not None:
                        self._dispatch(gen, *inner)
                else:
                    self._dispatch(gen, msg_type, payload)
        except BaseException as e:
            self._socket_failed(gen, e)

    def _reassemble(self, gen: int, payload: bytes) -> tuple[int, bytes] | None:
        msg_id, total, offset = _CHUNK_HEAD.unpack_from(payload)
        data = payload[_CHUNK_HEAD.size :]
        with self._reasm_lock:
            if gen != self._gen:
                return None  # chunk from a dead group; msg_ids restarted server-side
            if msg_id not in self._reasm:
                self._reasm[msg_id] = (bytearray(total), [0])
            buf, filled = self._reasm[msg_id]
            buf[offset : offset + len(data)] = data
            filled[0] += len(data)
            if filled[0] < total:
                return None
            del self._reasm[msg_id]
        magic, msg_type, length = _HEADER.unpack_from(bytes(buf[: _HEADER.size]))
        if magic != MAGIC or length != total - _HEADER.size:
            raise ValueError("corrupt reassembled frame")
        return msg_type, bytes(buf[_HEADER.size :])

    def _dispatch(self, gen: int, msg_type: int, payload: bytes) -> None:
        if msg_type == MSG_SHUTDOWN:
            self.shutdown.set()
            return
        with self._lock:
            if gen != self._gen:
                return  # late message from a dead group
        if msg_type == MSG_PULL_REQ:
            fid, step = struct.unpack("<IQ", payload)
            self._pulls.put(PullRequest(fid, step))
        elif msg_type == MSG_BCAST_FRAGMENT:
            fid, version = struct.unpack_from("<IQ", payload)
            self._bcasts.put(BcastFragment(fid, version, payload[12:]))
        elif msg_type == MSG_BCAST_CONTROL:
            fid, version = struct.unpack_from("<IQ", payload)
            self._controls.put(BcastFragment(fid, version, payload[12:]))
        elif msg_type == MSG_BCAST_CONTROL_PAIR:
            fid, version, local_len = struct.unpack_from("<IQQ", payload)
            split = 20 + local_len
            if split > len(payload):
                raise ValueError("truncated SCAFFOLD full control pair")
            self._control_pairs.put(
                BcastControlPair(fid, version, payload[20:split], payload[split:])
            )


def _close_socket(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
