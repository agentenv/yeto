"""Python side of the learner <-> syncer wire protocol (docs/PROTOCOL.md v4).

A learner owns a connection *group*: stream 0 carries control messages
(HELLO, PULL_REQ, HEARTBEAT, FINAL_MANIFEST/ACK, SHUTDOWN); large payloads
(INIT_PARAMS, PUSH_FRAGMENT, BCAST_FRAGMENT, FINAL_FRAGMENT) are split into
4 MiB CHUNK envelopes striped round-robin across extra data sockets, which
multiplies WAN throughput over what a single congestion-window-limited TCP
stream can carry.

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

import hashlib
import itertools
import queue
import secrets
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .fragments import MERGE_ISO, FragmentLayout

MAGIC = 0xD170C0DE
PROTOCOL_VERSION = 4

MSG_HELLO = 1
MSG_INIT_PARAMS = 2
MSG_PULL_REQ = 3
MSG_PUSH_FRAGMENT = 4
MSG_BCAST_FRAGMENT = 5
MSG_HEARTBEAT = 6
MSG_SHUTDOWN = 7
MSG_DATA_HELLO = 8
MSG_CHUNK = 9
MSG_ERROR = 10
MSG_FINAL_MANIFEST = 11
MSG_FINAL_ACK = 12
MSG_FINAL_FRAGMENT = 13
MSG_BUDGET_DONE = 14

FINALIZATION_REVISION = 1
FINALIZATION_TIMEOUT = 900.0

DTYPE_F32 = 1
DTYPE_BF16 = 2
# Session dtype 3: PUSH_FRAGMENT payloads are block-quantized 4-bit E3M0
# base-relative learner deltas; INIT_PARAMS and BCAST_FRAGMENT stay bf16
# (see bulk_dtype and docs/PROTOCOL.md v4).
DTYPE_Q4 = 3

CHUNK_SIZE = 4 * 1024 * 1024
MAX_HANDSHAKE_FRAME = 16 * 1024 * 1024
MAX_ERROR_FRAME = 64 * 1024
Q4_BLOCK = 256
MAX_PARTIAL_MESSAGES = 64

RECONNECT_BACKOFF_START = 1.0
RECONNECT_BACKOFF_CAP = 30.0
RECONNECT_DIAL_TIMEOUT = 20.0

_HEADER = struct.Struct("<IBQ")  # magic, type, payload length
_CHUNK_HEAD = struct.Struct("<QQQ")  # msg_id, total_len, offset
_HELLO_HEAD = struct.Struct("<HIQBI")  # version, learner, generation, dtype, fragments
_DATA_HELLO = struct.Struct("<HIQH")  # version, learner, generation, stream index
_FINAL_MANIFEST_HEAD = struct.Struct("<HQI")  # revision, global_step, fragments
_FINAL_ACK = struct.Struct("<HQ")  # revision, global_step
_BUDGET_DONE = struct.Struct("<Q")  # local steps


def bulk_dtype(dtype: int) -> int:
    """Dtype of INIT_PARAMS/BCAST_FRAGMENT tensors for a session dtype.

    Q4 applies only to push deltas (small dynamic range); full parameter
    payloads would not survive 4 bits, so they travel as bf16.
    """
    return DTYPE_BF16 if dtype == DTYPE_Q4 else dtype


def layout_fingerprint(layout: FragmentLayout) -> bytes:
    """Canonical semantic identity for tensor names, order, shapes, and modes."""
    digest = hashlib.sha256()
    digest.update(b"yeto-layout-v1\0")
    digest.update(struct.pack("<I", layout.num_fragments))
    for frag in layout.fragments:
        digest.update(struct.pack("<BI", frag.merge_mode, len(frag.tensors)))
        for name, numel in frag.tensors:
            encoded_name = name.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded_name)))
            digest.update(encoded_name)
            digest.update(struct.pack("<Q", numel))
            if frag.identity_shapes is not None:
                shape = frag.identity_shapes[name]
            elif frag.shapes is not None and name in frag.shapes:
                shape = frag.shapes[name]
            else:
                shape = (numel,)
            digest.update(struct.pack("<I", len(shape)))
            digest.update(struct.pack(f"<{len(shape)}Q", *shape))
    return digest.digest()


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


def read_frame(
    sock: socket.socket,
    max_payload: int | Callable[[int], int] = MAX_HANDSHAKE_FRAME,
) -> tuple[int, bytes]:
    """Read one frame after checking its type-aware limit before allocation."""
    magic, msg_type, length = _HEADER.unpack(_read_exact(sock, _HEADER.size))
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    limit = max_payload(msg_type) if callable(max_payload) else max_payload
    if length > limit:
        raise ValueError(
            f"frame type {msg_type} has length {length}, exceeds limit {limit}"
        )
    return msg_type, _read_exact(sock, length)


def encode_hello(
    learner_id: int,
    dtype: int,
    layout: FragmentLayout,
    num_streams: int,
    connection_generation: int = 0,
    session_contract_hash: bytes | None = None,
) -> bytes:
    if session_contract_hash is None:
        # Generic/legacy callers still bind the session to the semantic tensor
        # layout. Restricted profiles may supply a stronger opaque contract.
        session_contract_hash = layout_fingerprint(layout)
    if not isinstance(session_contract_hash, bytes) or len(session_contract_hash) != 32:
        raise ValueError("session contract hash must be exactly 32 bytes")
    parts = [
        _HELLO_HEAD.pack(
            PROTOCOL_VERSION,
            learner_id,
            connection_generation,
            dtype,
            layout.num_fragments,
        )
    ]
    for frag in layout.fragments:
        parts.append(struct.pack("<BI", frag.merge_mode, len(frag.tensors)))
        parts.append(struct.pack(f"<{len(frag.tensors)}Q", *(n for _, n in frag.tensors)))
        if frag.merge_mode == MERGE_ISO:
            # Iso fragments append (rows, cols) per tensor so the syncer can
            # take the 2D view; avg/RDA keep the original wire format.
            dims = [d for name, _ in frag.tensors for d in frag.shapes[name]]
            parts.append(struct.pack(f"<{len(dims)}Q", *dims))
    parts.append(layout_fingerprint(layout))
    parts.append(session_contract_hash)
    parts.append(struct.pack("<H", num_streams))
    return b"".join(parts)


def encode_data_hello(
    learner_id: int, connection_generation: int, stream_index: int
) -> bytes:
    return _DATA_HELLO.pack(
        PROTOCOL_VERSION, learner_id, connection_generation, stream_index
    )


def _tensor_nbytes(dtype: int, numel: int) -> int:
    width = {DTYPE_F32: 4, DTYPE_BF16: 2}.get(dtype)
    if width is None:
        raise ValueError(f"unknown tensor dtype {dtype}")
    return numel * width


def _q4_nbytes(numel: int) -> int:
    return -(-numel // Q4_BLOCK) * (4 + Q4_BLOCK // 2)


def _contiguous_bytes_view(
    part: bytes | bytearray | memoryview,
    index: int,
    label: str,
) -> memoryview:
    try:
        view = memoryview(part)
    except TypeError as exc:
        raise TypeError(f"{label} part {index} is not bytes-like") from exc
    if not view.c_contiguous:
        view.release()
        raise ValueError(f"{label} part {index} is not C-contiguous")
    try:
        return view.cast("B")
    except (TypeError, ValueError) as exc:
        view.release()
        raise ValueError(
            f"{label} part {index} cannot be viewed as contiguous bytes"
        ) from exc


class ProtocolError(RuntimeError):
    """The peer rejected this client as wire- or session-incompatible."""


def encode_final_manifest(global_step: int, versions: tuple[int, ...] | list[int]) -> bytes:
    """Encode a versioned description of the authoritative terminal cut."""
    return _FINAL_MANIFEST_HEAD.pack(
        FINALIZATION_REVISION, global_step, len(versions)
    ) + struct.pack(f"<{len(versions)}Q", *versions)


def decode_final_manifest(payload: bytes, expected_fragments: int) -> "FinalManifest":
    if len(payload) < _FINAL_MANIFEST_HEAD.size:
        raise ValueError("truncated FINAL_MANIFEST header")
    revision, global_step, count = _FINAL_MANIFEST_HEAD.unpack_from(payload)
    if revision != FINALIZATION_REVISION:
        raise ValueError(
            f"unsupported finalization revision {revision}; "
            f"expected {FINALIZATION_REVISION}"
        )
    if count != expected_fragments:
        raise ValueError(
            f"FINAL_MANIFEST has {count} fragments, expected {expected_fragments}"
        )
    expected_size = _FINAL_MANIFEST_HEAD.size + count * 8
    if len(payload) != expected_size:
        raise ValueError(
            f"FINAL_MANIFEST has {len(payload)} bytes, expected {expected_size}"
        )
    versions = struct.unpack_from(f"<{count}Q", payload, _FINAL_MANIFEST_HEAD.size)
    return FinalManifest(global_step, tuple(versions))


@dataclass
class PullRequest:
    fragment_id: int
    global_step: int
    round_attempt: int
    received_at: float = field(default_factory=time.monotonic, compare=False)


@dataclass
class BcastFragment:
    fragment_id: int
    version: int
    data: object  # raw tensor buffer, or a typed object-store reference bundle
    received_at: float = field(default_factory=time.monotonic, compare=False)
    payload_hash: bytes | None = field(default=None, compare=False)
    stored: bool = field(default=False, compare=False)
    discard: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class FinalFragment:
    fragment_id: int
    version: int
    data: object  # raw f32 tensor buffer, or a typed object-store reference bundle
    payload_hash: bytes | None = field(default=None, compare=False)
    stored: bool = field(default=False, compare=False)
    discard: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class FinalManifest:
    global_step: int
    versions: tuple[int, ...]


@dataclass
class StreamedInboundPayload:
    """One validated tensor payload committed directly into a typed sink."""

    data: object
    payload_hash: bytes
    discard: Callable[[], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload_hash, bytes) or len(self.payload_hash) != 32:
            raise ValueError("streamed inbound payload hash must be a SHA256")


@dataclass
class _StagedInboundChunk:
    offset: int
    nbytes: int
    token: object | None


@dataclass
class _StreamedReassembly:
    total: int
    sink: object
    transaction: object
    filled: int = 0
    ranges: list[tuple[int, int]] = field(default_factory=list)
    pending: dict[int, _StagedInboundChunk] = field(default_factory=dict)
    next_payload_offset: int = 0
    msg_type: int | None = None
    fragment_id: int | None = None
    version: int | None = None
    payload_bytes: int | None = None


@dataclass(frozen=True)
class _StreamedInboundFragment:
    msg_type: int
    fragment_id: int
    version: int
    payload: StreamedInboundPayload


@dataclass
class _Outbound:
    data: bytes | bytearray
    sent: threading.Event | None = None


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
        connect_timeout: float = 900.0,
        max_reconnects: int | None = None,
        finalization_timeout: float = FINALIZATION_TIMEOUT,
        session_contract_hash: bytes | None = None,
    ):
        if not 0 <= learner_id <= 0xFFFF_FFFF:
            raise ValueError(f"learner_id must fit u32, got {learner_id}")
        if dtype not in (DTYPE_F32, DTYPE_BF16, DTYPE_Q4):
            raise ValueError(f"unsupported session dtype {dtype}")
        if layout.num_fragments < 1:
            raise ValueError("layout must contain at least one fragment")
        if not 0 <= num_streams <= 256:
            raise ValueError(f"num_streams must be in [0, 256], got {num_streams}")
        if session_contract_hash is None:
            session_contract_hash = layout_fingerprint(layout)
        if not isinstance(session_contract_hash, bytes) or len(session_contract_hash) != 32:
            raise ValueError("session contract hash must be exactly 32 bytes")
        self.addr = addr
        self.learner_id = learner_id
        self.layout = layout
        self.dtype = dtype
        self.num_streams = num_streams
        self.session_contract_hash = session_contract_hash
        self.connect_timeout = connect_timeout
        self.max_reconnects = max_reconnects
        self.finalization_timeout = finalization_timeout
        self._max_bcast_payload = max(
            12 + _tensor_nbytes(bulk_dtype(dtype), fragment.numel)
            for fragment in layout.fragments
        )
        self._max_final_payload = max(
            12 + _tensor_nbytes(DTYPE_F32, fragment.numel)
            for fragment in layout.fragments
        )
        self._max_chunked_inner = _HEADER.size + max(
            self._max_bcast_payload, self._max_final_payload
        )
        self._queues: list[queue.Queue[_Outbound | None]] = []
        self._socks: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        self._pulls: queue.Queue[PullRequest] = queue.Queue()
        self._bcasts: queue.Queue[BcastFragment] = queue.Queue()
        self._bcast_seen: list[tuple[int, bytes] | None] = [
            None
        ] * layout.num_fragments
        self._bcast_lock = threading.Lock()
        # Raw globals are retained independently of the drain queue so a
        # final fragment that outruns its control-stream manifest can be
        # re-applied exactly after the manifest arrives.
        self._final_fragments: dict[int, FinalFragment] = {}
        self._final_manifest: FinalManifest | None = None
        self._final_ack_step: int | None = None
        self.finalizing = threading.Event()
        self.finalized = threading.Event()
        self._final_cond = threading.Condition()
        self._reasm: dict[
            int,
            tuple[bytearray, list[int], list[tuple[int, int]]]
            | _StreamedReassembly,
        ] = {}
        self._reasm_lock = threading.Lock()
        self._inbound_chunk_sink: object | None = None
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
        self.connection_generation = 0
        self._connected = threading.Event()
        self._failure = threading.Event()  # set by any thread of the live group
        self._closed = threading.Event()
        self._reconnects_used = 0
        self._reset_bcasts_on_reconnect = False
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
        connection_generation = secrets.randbits(64)
        while connection_generation == 0 or connection_generation == self.connection_generation:
            connection_generation = secrets.randbits(64)
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
                    connection_generation,
                    self.session_contract_hash,
                ),
            )
            socks.append(control)
            for idx in range(self.num_streams):
                s = self._connect_one() if patient else self._dial(RECONNECT_DIAL_TIMEOUT)
                write_frame(
                    s,
                    MSG_DATA_HELLO,
                    encode_data_hello(self.learner_id, connection_generation, idx),
                )
                socks.append(s)
        except BaseException:
            for s in socks:
                _close_socket(s)
            raise
        with self._lock:
            self._gen += 1
            gen = self._gen
            self.connection_generation = connection_generation
            self._socks = socks
            self._queues = []
            self._threads = []
            self._failure.clear()
            self._connected.set()
            for i, s in enumerate(socks):
                q: queue.Queue[_Outbound | None] = queue.Queue(maxsize=256)
                self._queues.append(q)
                ts = threading.Thread(
                    target=self._send_loop, args=(gen, s, q), name=f"yeto-send-{i}", daemon=True
                )
                tr = threading.Thread(target=self._recv_loop, args=(gen, s), name=f"yeto-recv-{i}", daemon=True)
                self._threads += [ts, tr]
                ts.start()
                tr.start()
        with self._final_cond:
            self._final_cond.notify_all()

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
        self._clear_partial_reassemblies()
        self._drain(self._pulls)  # stale pull requests; answering them is pointless
        with self._lock:
            reset_bcasts = self._reset_bcasts_on_reconnect
            self._reset_bcasts_on_reconnect = False
        if reset_bcasts:
            self._drain(self._bcasts)
            with self._bcast_lock:
                self._bcast_seen = [None] * self.layout.num_fragments
        for t in threads:
            t.join(timeout=5.0)
        with self._final_cond:
            self._final_cond.notify_all()

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
                    with self._final_cond:
                        self._final_cond.notify_all()
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
        self._clear_partial_reassemblies()
        with self._final_cond:
            self._final_cond.notify_all()

    def check_health(self) -> None:
        """Raise only for unrecoverable failures. While a reconnect is being
        attempted this is a no-op; the training loop keeps stepping locally."""
        if self._err is not None:
            raise RuntimeError(f"syncer connection failed: {self._err}") from self._err

    # -- learner-facing API ------------------------------------------------------

    def send_init(self, fragment_id: int, tensor_bytes: bytes) -> None:
        if not 0 <= fragment_id < self.layout.num_fragments:
            raise ValueError(f"INIT_PARAMS for unknown fragment {fragment_id}")
        expected = _tensor_nbytes(
            bulk_dtype(self.dtype), self.layout.fragments[fragment_id].numel
        )
        if len(tensor_bytes) != expected:
            raise ValueError(
                f"INIT_PARAMS fragment {fragment_id} has {len(tensor_bytes)} "
                f"tensor bytes, expected {expected}"
            )
        self._send_large(MSG_INIT_PARAMS, struct.pack("<I", fragment_id) + tensor_bytes)

    def send_init_parts(
        self,
        fragment_id: int,
        tensor_parts: Iterable[bytes | bytearray | memoryview],
    ) -> bool:
        """Stream INIT_PARAMS without materializing its complete tensor."""
        if not 0 <= fragment_id < self.layout.num_fragments:
            raise ValueError(f"INIT_PARAMS for unknown fragment {fragment_id}")
        expected = _tensor_nbytes(
            bulk_dtype(self.dtype), self.layout.fragments[fragment_id].numel
        )
        return self._send_large_parts(
            MSG_INIT_PARAMS,
            struct.pack("<I", fragment_id),
            tensor_parts,
            expected,
            label=f"INIT_PARAMS fragment {fragment_id}",
        )

    def push_fragment(
        self,
        fragment_id: int,
        global_step: int,
        round_attempt: int,
        base_version: int,
        local_step: int,
        c_steps: int,
        c_tokens: int,
        tensor_bytes: bytes,
    ) -> None:
        if not 0 <= fragment_id < self.layout.num_fragments:
            raise ValueError(f"PUSH_FRAGMENT for unknown fragment {fragment_id}")
        if round_attempt < 1:
            raise ValueError("round_attempt must be positive")
        if c_steps < 1:
            raise ValueError("c_steps must be positive")
        numel = self.layout.fragments[fragment_id].numel
        expected = (
            _q4_nbytes(numel)
            if self.dtype == DTYPE_Q4
            else _tensor_nbytes(self.dtype, numel)
        )
        if len(tensor_bytes) != expected:
            raise ValueError(
                f"PUSH_FRAGMENT fragment {fragment_id} has {len(tensor_bytes)} "
                f"delta bytes, expected {expected}"
            )
        head = struct.pack(
            "<IIQIQQIQ",
            self.learner_id,
            fragment_id,
            global_step,
            round_attempt,
            base_version,
            local_step,
            c_steps,
            c_tokens,
        )
        self._send_large(MSG_PUSH_FRAGMENT, head + tensor_bytes)

    def push_fragment_parts(
        self,
        fragment_id: int,
        global_step: int,
        round_attempt: int,
        base_version: int,
        local_step: int,
        c_steps: int,
        c_tokens: int,
        tensor_parts: Iterable[bytes | bytearray | memoryview],
        *,
        before_last_enqueue: Callable[[], None] | None = None,
    ) -> bool:
        """Stream one PUSH_FRAGMENT from bounded contiguous byte parts.

        The syncer observes the same logical inner frame as ``push_fragment``.
        Only one CHUNK-sized envelope is assembled at a time; neither the
        complete tensor nor the complete inner frame is materialized here.
        Returns ``True`` only after every chunk is queued and ``False`` when
        none could be queued. A failure after the first chunk poisons that
        connection generation and raises so callers cannot mark the push done.
        """
        if not 0 <= fragment_id < self.layout.num_fragments:
            raise ValueError(f"PUSH_FRAGMENT for unknown fragment {fragment_id}")
        if round_attempt < 1:
            raise ValueError("round_attempt must be positive")
        if c_steps < 1:
            raise ValueError("c_steps must be positive")
        numel = self.layout.fragments[fragment_id].numel
        expected = (
            _q4_nbytes(numel)
            if self.dtype == DTYPE_Q4
            else _tensor_nbytes(self.dtype, numel)
        )
        head = struct.pack(
            "<IIQIQQIQ",
            self.learner_id,
            fragment_id,
            global_step,
            round_attempt,
            base_version,
            local_step,
            c_steps,
            c_tokens,
        )
        return self._send_large_parts(
            MSG_PUSH_FRAGMENT,
            head,
            tensor_parts,
            expected,
            label=f"PUSH_FRAGMENT fragment {fragment_id}",
            before_last_enqueue=before_last_enqueue,
        )

    def heartbeat(self, local_step: int) -> None:
        self._enqueue(0, self._frame(MSG_HEARTBEAT, struct.pack("<IQ", self.learner_id, local_step)))

    def send_budget_done(
        self, local_steps: int, timeout: float | None = None
    ) -> int:
        """Send the exact frozen learner budget and wait until bytes leave."""
        if not 0 < local_steps <= 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError("local_steps must be a positive u64")
        timeout = self.finalization_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        payload = _BUDGET_DONE.pack(local_steps)
        with self._lock:
            if not self._connected.is_set():
                raise RuntimeError("cannot send BUDGET_DONE while disconnected")
            gen = self._gen
            self._reset_bcasts_on_reconnect = True
        sent = threading.Event()
        if not self._enqueue(
            0,
            self._frame(MSG_BUDGET_DONE, payload),
            gen=gen,
            sent=sent,
        ):
            raise RuntimeError("connection closed before BUDGET_DONE was queued")
        while not sent.wait(min(0.1, max(0.0, deadline - time.monotonic()))):
            self.check_health()
            with self._lock:
                if gen != self._gen:
                    raise RuntimeError("connection lost while sending BUDGET_DONE")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"BUDGET_DONE was not sent within {timeout:.1f}s"
                )
        return gen

    def wait_for_budget_restart(
        self, previous_generation: int, timeout: float | None = None
    ) -> None:
        """Wait for the fresh connection group used by consolidation."""
        timeout = self.finalization_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while True:
            self.check_health()
            with self._lock:
                ready = self._connected.is_set() and self._gen > previous_generation
            if ready:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"budget consolidation syncer did not reconnect within {timeout:.1f}s"
                )
            with self._final_cond:
                self._final_cond.wait(min(remaining, 0.5))

    def drain_pulls(self) -> list[PullRequest]:
        return self._drain(self._pulls)

    def drain_updates(self) -> list[BcastFragment]:
        return self._drain(self._bcasts)

    def install_inbound_chunk_sink(self, sink: object | None) -> None:
        """Select a transactional sink for future chunked fragment receives.

        Switching is allowed only between complete logical messages, so a
        reconnect or policy-boundary sink replacement cannot mix anchors.
        """

        if sink is not None:
            required = (
                "begin_message",
                "bind_fragment",
                "stage_chunk",
                "consume_chunk",
                "consume_staged_chunk",
                "finish_message",
                "abort_message",
            )
            if any(not callable(getattr(sink, name, None)) for name in required):
                raise TypeError(
                    "inbound chunk sink does not implement the transaction API"
                )
        with self._reasm_lock:
            if self._reasm:
                raise RuntimeError("cannot replace inbound chunk sink mid-message")
            self._inbound_chunk_sink = sink

    def wait_for_final_fragments(
        self, timeout: float | None = None
    ) -> tuple[FinalManifest, list[FinalFragment]]:
        """Wait for a manifest and its exact fragment versions.

        FINAL_MANIFEST travels on control while fragments may be striped over
        data sockets, so either can arrive first. The retained payload cache
        makes both orders equivalent and survives a connection-group redial.
        """
        timeout = self.finalization_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        with self._final_cond:
            while True:
                self.check_health()
                manifest = self._final_manifest
                if manifest is not None:
                    ready = []
                    missing = []
                    for fid, version in enumerate(manifest.versions):
                        cached = self._final_fragments.get(fid)
                        if cached is None or cached.version != version:
                            latest = None if cached is None else cached.version
                            missing.append((fid, version, latest))
                        else:
                            ready.append(cached)
                    if not missing:
                        return manifest, ready
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = (
                        "manifest not received"
                        if manifest is None
                        else f"missing fragment versions {missing}"
                    )
                    raise TimeoutError(
                        f"finalization timed out after {timeout:.1f}s: {detail}"
                    )
                self._final_cond.wait(min(remaining, 0.5))

    def acknowledge_finalization(
        self, manifest: FinalManifest, timeout: float | None = None
    ) -> None:
        """Confirm the exact cut and wait boundedly until ACK bytes are sent.

        The training loop may exit as soon as this returns: the sender thread
        has completed ``sendall`` for FINAL_ACK, so saving cannot race client
        teardown and silently lose the acknowledgement.
        """
        if manifest != self._final_manifest:
            raise RuntimeError("cannot acknowledge a stale final manifest")
        timeout = self.finalization_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        payload = _FINAL_ACK.pack(FINALIZATION_REVISION, manifest.global_step)
        self._final_ack_step = manifest.global_step
        while True:
            self.check_health()
            if self.shutdown.is_set():
                # The coordinator may receive an ACK from the previous live
                # generation just before a reconnect. Receipt of SHUTDOWN for
                # this exact manifest is equivalent confirmation.
                self.finalized.set()
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"finalization ACK was not sent within {timeout:.1f}s"
                )
            with self._lock:
                connected = self._connected.is_set()
                gen = self._gen
            if not connected:
                with self._final_cond:
                    self._final_cond.wait(min(remaining, 0.5))
                continue
            sent = threading.Event()
            if not self._enqueue(
                0,
                self._frame(MSG_FINAL_ACK, payload),
                gen=gen,
                sent=sent,
            ):
                continue
            while not sent.wait(min(0.1, max(0.0, deadline - time.monotonic()))):
                self.check_health()
                with self._lock:
                    if gen != self._gen:
                        break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"finalization ACK was not sent within {timeout:.1f}s"
                    )
            if sent.is_set():
                self.finalized.set()
                with self._final_cond:
                    self._final_cond.notify_all()
                return

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

    def _enqueue(
        self,
        stream: int,
        data: bytes,
        gen: int | None = None,
        sent: threading.Event | None = None,
    ) -> bool:
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
                q.put(
                    _Outbound(data, sent),
                    timeout=0.5,
                )  # bounded wait, then re-check the group
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

    def _send_large_parts(
        self,
        msg_type: int,
        payload_prefix: bytes,
        parts: Iterable[bytes | bytearray | memoryview],
        parts_size: int,
        *,
        label: str,
        before_last_enqueue: Callable[[], None] | None = None,
    ) -> bool:
        """Emit one logical frame without joining its bytes-like parts."""
        payload_size = len(payload_prefix) + parts_size
        fixed = _HEADER.pack(MAGIC, msg_type, payload_size) + payload_prefix
        total = len(fixed) + parts_size
        with self._lock:
            if (
                self._closed.is_set()
                or self.shutdown.is_set()
                or not self._connected.is_set()
            ):
                return False  # outage: explicitly report a whole-message drop
            gen = self._gen

        part_iter = iter(parts)
        part_index = 0
        pending: memoryview | None = None
        pending_offset = 0
        parts_seen = 0
        msg_id = next(self._msg_id)
        offset = 0
        envelope_prefix = _HEADER.size + _CHUNK_HEAD.size
        enqueued_chunks = 0

        try:
            while offset < total:
                inner_size = min(CHUNK_SIZE, total - offset)
                envelope = bytearray(envelope_prefix + inner_size)
                _HEADER.pack_into(
                    envelope,
                    0,
                    MAGIC,
                    MSG_CHUNK,
                    _CHUNK_HEAD.size + inner_size,
                )
                _CHUNK_HEAD.pack_into(
                    envelope,
                    _HEADER.size,
                    msg_id,
                    total,
                    offset,
                )

                inner_end = offset + inner_size
                write_at = envelope_prefix
                if offset < len(fixed):
                    fixed_end = min(inner_end, len(fixed))
                    fixed_bytes = fixed[offset:fixed_end]
                    envelope[write_at : write_at + len(fixed_bytes)] = fixed_bytes
                    write_at += len(fixed_bytes)

                remaining = envelope_prefix + inner_size - write_at
                while remaining:
                    if pending is None:
                        try:
                            part = next(part_iter)
                        except StopIteration:
                            raise ValueError(
                                f"{label} has {parts_seen} delta bytes, "
                                f"expected {parts_size}"
                            ) from None
                        pending = _contiguous_bytes_view(part, part_index, label)
                        part_index += 1
                        pending_offset = 0
                        if pending.nbytes == 0:
                            pending.release()
                            pending = None
                            continue
                        if parts_seen + pending.nbytes > parts_size:
                            raise ValueError(
                                f"{label} exceeds expected {parts_size} delta bytes"
                            )

                    available = pending.nbytes - pending_offset
                    take = min(remaining, available)
                    envelope[write_at : write_at + take] = pending[
                        pending_offset : pending_offset + take
                    ]
                    write_at += take
                    remaining -= take
                    pending_offset += take
                    parts_seen += take
                    if pending_offset == pending.nbytes:
                        pending.release()
                        pending = None

                if inner_end == total:
                    if parts_seen != parts_size:
                        raise ValueError(
                            f"{label} has {parts_seen} delta bytes, "
                            f"expected {parts_size}"
                        )
                    for part in part_iter:
                        extra = _contiguous_bytes_view(part, part_index, label)
                        part_index += 1
                        try:
                            if extra.nbytes:
                                raise ValueError(
                                    f"{label} exceeds expected {parts_size} delta bytes"
                                )
                        finally:
                            extra.release()
                    if before_last_enqueue is not None:
                        callback = before_last_enqueue
                        before_last_enqueue = None
                        callback()

                stream = (
                    0
                    if self.num_streams == 0
                    else 1 + next(self._rr) % self.num_streams
                )
                if not self._enqueue(stream, envelope, gen=gen):
                    if enqueued_chunks:
                        raise RuntimeError(
                            f"connection generation {gen} dropped after "
                            f"{enqueued_chunks} {label} chunks were queued"
                        )
                    return False
                enqueued_chunks += 1
                offset = inner_end
            return True
        except BaseException as exc:
            if enqueued_chunks:
                self._poison_outbound_generation(gen, exc)
            raise
        finally:
            if pending is not None:
                pending.release()

    def _poison_outbound_generation(self, gen: int, exc: BaseException) -> None:
        """Invalidate and promptly close a generation holding a partial frame."""
        with self._lock:
            if gen != self._gen:
                return
            self._last_err = exc
            self._connected.clear()
            self._failure.set()
            socks = list(self._socks)
        for sock in socks:
            _close_socket(sock)
        with self._final_cond:
            self._final_cond.notify_all()

    def _socket_failed(self, gen: int, exc: BaseException) -> None:
        if self.shutdown.is_set() or self._closed.is_set():
            return
        with self._lock:
            if gen != self._gen:
                return  # a thread of an already-torn-down group; ignore
            self._last_err = exc
            self._connected.clear()
            self._failure.set()
        with self._final_cond:
            self._final_cond.notify_all()

    def _protocol_failed(self, gen: int, exc: BaseException) -> None:
        with self._lock:
            if gen != self._gen:
                return
            error = exc if isinstance(exc, ProtocolError) else ProtocolError(str(exc))
            self._err = error
            self._last_err = error
            self._connected.clear()
            self._closed.set()
            self._failure.set()
        with self._final_cond:
            self._final_cond.notify_all()

    def _send_loop(self, gen: int, sock: socket.socket, q: queue.Queue) -> None:
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                sock.sendall(item.data)
                if item.sent is not None:
                    item.sent.set()
        except BaseException as e:
            self._socket_failed(gen, e)

    def _recv_loop(self, gen: int, sock: socket.socket) -> None:
        try:
            while True:
                msg_type, payload = read_frame(sock, self._incoming_frame_limit)
                if msg_type == MSG_CHUNK:
                    inner = self._reassemble(gen, payload)
                    if inner is not None:
                        if isinstance(inner, _StreamedInboundFragment):
                            self._dispatch_streamed(gen, inner)
                        else:
                            self._dispatch(gen, *inner)
                else:
                    self._dispatch(gen, msg_type, payload)
        except (ValueError, ProtocolError) as e:
            self._protocol_failed(gen, e)
        except BaseException as e:
            self._socket_failed(gen, e)

    def _incoming_frame_limit(self, msg_type: int) -> int:
        limits = {
            MSG_PULL_REQ: 16,
            MSG_BCAST_FRAGMENT: self._max_bcast_payload,
            MSG_FINAL_FRAGMENT: self._max_final_payload,
            MSG_FINAL_MANIFEST: _FINAL_MANIFEST_HEAD.size
            + 8 * self.layout.num_fragments,
            MSG_SHUTDOWN: 0,
            MSG_ERROR: MAX_ERROR_FRAME,
            MSG_CHUNK: _CHUNK_HEAD.size + CHUNK_SIZE,
        }
        try:
            return limits[msg_type]
        except KeyError:
            raise ValueError(f"unexpected syncer message type {msg_type}") from None

    def _reassemble(
        self, gen: int, payload: bytes
    ) -> tuple[int, memoryview] | _StreamedInboundFragment | None:
        if len(payload) < _CHUNK_HEAD.size:
            raise ValueError("chunk header truncated")
        msg_id, total, offset = _CHUNK_HEAD.unpack_from(payload)
        data = payload[_CHUNK_HEAD.size :]
        if not data:
            raise ValueError("empty chunk")
        if total < _HEADER.size:
            raise ValueError("chunked inner frame is shorter than its header")
        if total > self._max_chunked_inner:
            raise ValueError(
                f"chunked frame has length {total}, exceeds negotiated limit "
                f"{self._max_chunked_inner}"
            )
        end = offset + len(data)
        if end > total:
            raise ValueError("chunk overflow")
        with self._reasm_lock:
            if gen != self._gen:
                return None  # chunk from a dead group; msg_ids restarted server-side
            if msg_id not in self._reasm:
                if len(self._reasm) >= MAX_PARTIAL_MESSAGES:
                    raise ValueError("too many partial chunked messages")
                if self._inbound_chunk_sink is None:
                    self._reasm[msg_id] = (bytearray(total), [0], [])
                else:
                    sink = self._inbound_chunk_sink
                    self._reasm[msg_id] = _StreamedReassembly(
                        total=total,
                        sink=sink,
                        transaction=sink.begin_message(msg_id, total),
                    )
            partial = self._reasm[msg_id]
            if isinstance(partial, _StreamedReassembly):
                if partial.total != total:
                    self._reasm.pop(msg_id, None)
                    self._abort_streamed_partial(partial)
                    raise ValueError("chunk total changed within one message")
                try:
                    completed = self._stream_chunk_locked(
                        msg_id,
                        partial,
                        offset,
                        data,
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    self._reasm.pop(msg_id, None)
                    self._abort_streamed_partial(partial)
                    raise
                if completed is not None:
                    self._reasm.pop(msg_id, None)
                return completed
            buf, filled, ranges = partial
            if len(buf) != total:
                raise ValueError("chunk total changed within one message")
            if any(offset < old_end and old_start < end for old_start, old_end in ranges):
                raise ValueError("overlapping chunk")
            buf[offset:end] = data
            filled[0] += len(data)
            ranges.append((offset, end))
            if filled[0] < total:
                return None
            if filled[0] != total:
                raise ValueError("chunk byte count exceeds frame length")
            del self._reasm[msg_id]
        magic, msg_type, length = _HEADER.unpack_from(buf)
        if magic != MAGIC or length != total - _HEADER.size:
            raise ValueError("corrupt reassembled frame")
        return msg_type, memoryview(buf)[_HEADER.size :]

    def _stream_chunk_locked(
        self,
        msg_id: int,
        partial: _StreamedReassembly,
        offset: int,
        data: bytes,
    ) -> _StreamedInboundFragment | None:
        """Validate/reorder one bounded CHUNK and feed tensor bytes to its sink."""

        del msg_id
        total = partial.total
        end = offset + len(data)
        if any(
            offset < old_end and old_start < end
            for old_start, old_end in partial.ranges
        ):
            raise ValueError("overlapping chunk")
        if offset == 0:
            tensor_start = _HEADER.size + 12
            if len(data) < tensor_start:
                raise ValueError("first streamed fragment chunk is header-truncated")
            magic, msg_type, length = _HEADER.unpack_from(data)
            if magic != MAGIC or length != total - _HEADER.size:
                raise ValueError("corrupt streamed inner frame")
            if msg_type not in (MSG_BCAST_FRAGMENT, MSG_FINAL_FRAGMENT):
                raise ValueError(
                    f"transactional CHUNK has unsupported inner type {msg_type}"
                )
            fragment_id, version = struct.unpack_from("<IQ", data, _HEADER.size)
            if fragment_id >= self.layout.num_fragments:
                raise ValueError(
                    f"streamed fragment has unknown fragment {fragment_id}"
                )
            dtype = (
                bulk_dtype(self.dtype)
                if msg_type == MSG_BCAST_FRAGMENT
                else DTYPE_F32
            )
            expected_payload = _tensor_nbytes(
                dtype,
                self.layout.fragments[fragment_id].numel,
            )
            if length != 12 + expected_payload:
                raise ValueError(
                    f"streamed fragment {fragment_id} has {length - 12} tensor "
                    f"bytes, expected {expected_payload}"
                )
            if partial.msg_type is not None and (
                partial.msg_type,
                partial.fragment_id,
                partial.version,
                partial.payload_bytes,
            ) != (msg_type, fragment_id, version, expected_payload):
                raise ValueError("streamed fragment header changed")
            partial.msg_type = msg_type
            partial.fragment_id = fragment_id
            partial.version = version
            partial.payload_bytes = expected_payload
            partial.sink.bind_fragment(
                partial.transaction,
                msg_type,
                fragment_id,
                version,
                expected_payload,
            )

        partial.ranges.append((offset, end))
        partial.filled += len(data)
        tensor_start = _HEADER.size + 12
        data_start = max(offset, tensor_start)
        if end > data_start:
            relative_offset = data_start - tensor_start
            view = memoryview(data)[data_start - offset :]
            try:
                if (
                    partial.msg_type is not None
                    and relative_offset == partial.next_payload_offset
                ):
                    partial.sink.consume_chunk(
                        partial.transaction,
                        relative_offset,
                        view,
                    )
                    partial.next_payload_offset += view.nbytes
                    self._drain_staged_chunks_locked(partial)
                else:
                    if relative_offset in partial.pending:
                        raise ValueError("duplicate streamed tensor chunk offset")
                    token = partial.sink.stage_chunk(partial.transaction, view)
                    if token is None:
                        raise RuntimeError(
                            "inbound chunk sink returned no staged object"
                        )
                    partial.pending[relative_offset] = _StagedInboundChunk(
                        relative_offset,
                        view.nbytes,
                        token,
                    )
            finally:
                view.release()

        if partial.filled < total:
            return None
        if partial.filled != total:
            raise ValueError("chunk byte count exceeds frame length")
        if (
            partial.msg_type is None
            or partial.fragment_id is None
            or partial.version is None
            or partial.payload_bytes is None
        ):
            raise ValueError("streamed fragment completed without its header")
        self._drain_staged_chunks_locked(partial)
        if partial.pending or partial.next_payload_offset != partial.payload_bytes:
            raise ValueError("streamed fragment tensor coverage is incomplete")
        committed = partial.sink.finish_message(partial.transaction)
        if not isinstance(committed, StreamedInboundPayload):
            raise TypeError("inbound chunk sink returned an invalid committed payload")
        return _StreamedInboundFragment(
            partial.msg_type,
            partial.fragment_id,
            partial.version,
            committed,
        )

    @staticmethod
    def _drain_staged_chunks_locked(partial: _StreamedReassembly) -> None:
        while partial.next_payload_offset in partial.pending:
            staged = partial.pending.pop(partial.next_payload_offset)
            token = staged.token
            if token is None:
                raise RuntimeError("staged inbound chunk lost its object reference")
            partial.sink.consume_staged_chunk(
                partial.transaction,
                staged.offset,
                token,
                staged.nbytes,
            )
            staged.token = None
            partial.next_payload_offset += staged.nbytes

    @staticmethod
    def _abort_streamed_partial(partial: _StreamedReassembly) -> None:
        tokens = [
            staged.token
            for staged in partial.pending.values()
            if staged.token is not None
        ]
        partial.pending.clear()
        partial.sink.abort_message(partial.transaction, tokens)

    def _clear_partial_reassemblies(self) -> None:
        with self._reasm_lock:
            partials = list(self._reasm.values())
            self._reasm.clear()
        for partial in partials:
            if isinstance(partial, _StreamedReassembly):
                try:
                    self._abort_streamed_partial(partial)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    # Teardown must still close the entire connection group,
                    # but retain a diagnostic if the typed sink misbehaved.
                    self._last_err = exc

    def _dispatch(
        self, gen: int, msg_type: int, payload: bytes | memoryview
    ) -> None:
        with self._lock:
            if gen != self._gen:
                return  # late message from a dead group
            # Keep the generation check and every resulting state mutation
            # atomic with respect to teardown/redial. Otherwise an obsolete
            # receiver could pass the check, lose the race to a reconnect,
            # then publish a stale terminal manifest or shutdown afterward.
            self._dispatch_live(gen, msg_type, payload)

    def _dispatch_streamed(
        self,
        gen: int,
        inbound: _StreamedInboundFragment,
    ) -> None:
        with self._lock:
            if gen != self._gen:
                if inbound.payload.discard is not None:
                    inbound.payload.discard()
                return
            self._dispatch_streamed_live(inbound)

    def _dispatch_streamed_live(self, inbound: _StreamedInboundFragment) -> None:
        fragment_id = inbound.fragment_id
        version = inbound.version
        committed = inbound.payload
        digest = committed.payload_hash
        if inbound.msg_type == MSG_BCAST_FRAGMENT:
            with self._bcast_lock:
                seen = self._bcast_seen[fragment_id]
                if seen is not None:
                    seen_version, seen_digest = seen
                    if version < seen_version:
                        if committed.discard is not None:
                            committed.discard()
                        return
                    if version == seen_version:
                        if digest != seen_digest:
                            if committed.discard is not None:
                                committed.discard()
                            raise ProtocolError(
                                "conflicting BCAST_FRAGMENT payloads for fragment "
                                f"{fragment_id} version {version}"
                            )
                        if committed.discard is not None:
                            committed.discard()
                        return
                self._bcast_seen[fragment_id] = (version, digest)
                self._bcasts.put(
                    BcastFragment(
                        fragment_id,
                        version,
                        committed.data,
                        payload_hash=digest,
                        stored=True,
                        discard=committed.discard,
                    )
                )
            return
        if inbound.msg_type == MSG_FINAL_FRAGMENT:
            update = FinalFragment(
                fragment_id,
                version,
                committed.data,
                payload_hash=digest,
                stored=True,
                discard=committed.discard,
            )
            with self._final_cond:
                current = self._final_fragments.get(fragment_id)
                if current is not None:
                    current_digest = current.payload_hash
                    if current_digest is None:
                        current_digest = hashlib.sha256(
                            memoryview(current.data).cast("B")
                        ).digest()
                    if version < current.version:
                        if committed.discard is not None:
                            committed.discard()
                        return
                    if version == current.version:
                        if digest != current_digest:
                            if committed.discard is not None:
                                committed.discard()
                            self._set_protocol_error(
                                "conflicting FINAL_FRAGMENT payloads for fragment "
                                f"{fragment_id} version {version}"
                            )
                            return
                        if committed.discard is not None:
                            committed.discard()
                        return
                self._final_fragments[fragment_id] = update
                self._final_cond.notify_all()
            return
        if committed.discard is not None:
            committed.discard()
        self._set_protocol_error(
            f"unsupported streamed syncer message type {inbound.msg_type}"
        )

    def _dispatch_live(
        self, gen: int, msg_type: int, payload: bytes | memoryview
    ) -> None:
        if msg_type == MSG_ERROR:
            message = payload.decode("utf-8", errors="replace")
            self._protocol_failed(
                gen, ProtocolError(f"syncer rejected protocol session: {message}")
            )
            return
        if msg_type == MSG_SHUTDOWN:
            if payload:
                self._set_protocol_error("SHUTDOWN payload must be empty")
                return
            manifest = self._final_manifest
            if manifest is None or self._final_ack_step != manifest.global_step:
                self._set_protocol_error(
                    "syncer sent legacy SHUTDOWN before the versioned final "
                    "manifest was applied; upgrade both syncer and learner"
                )
                return
            self.shutdown.set()
            with self._final_cond:
                self._final_cond.notify_all()
            return
        if msg_type == MSG_PULL_REQ:
            if len(payload) != 16:
                raise ValueError("PULL_REQ payload must be 16 bytes")
            fid, step, round_attempt = struct.unpack("<IQI", payload)
            if fid >= self.layout.num_fragments:
                raise ValueError(f"PULL_REQ for unknown fragment {fid}")
            self._pulls.put(PullRequest(fid, step, round_attempt))
        elif msg_type == MSG_BCAST_FRAGMENT:
            if len(payload) < 12:
                raise ValueError("BCAST_FRAGMENT header truncated")
            fid, version = struct.unpack_from("<IQ", payload)
            if fid >= self.layout.num_fragments:
                raise ValueError(f"BCAST_FRAGMENT for unknown fragment {fid}")
            expected = _tensor_nbytes(
                bulk_dtype(self.dtype), self.layout.fragments[fid].numel
            )
            if len(payload) - 12 != expected:
                raise ValueError(
                    f"BCAST_FRAGMENT {fid} has {len(payload) - 12} tensor bytes, "
                    f"expected {expected}"
                )
            data = payload[12:]
            digest = hashlib.sha256(data).digest()
            with self._bcast_lock:
                seen = self._bcast_seen[fid]
                if seen is not None:
                    seen_version, seen_digest = seen
                    if version < seen_version:
                        return
                    if version == seen_version:
                        if digest != seen_digest:
                            raise ProtocolError(
                                f"conflicting BCAST_FRAGMENT payloads for fragment "
                                f"{fid} version {version}"
                            )
                        return
                self._bcast_seen[fid] = (version, digest)
                # Receiver threads run once per striped socket. Queue the
                # accepted update under the same lock as the monotonicity
                # check so two versions cannot validate in order but enqueue
                # in reverse order after a thread reschedule.
                self._bcasts.put(BcastFragment(fid, version, data))
        elif msg_type == MSG_FINAL_FRAGMENT:
            if len(payload) < 12:
                self._set_protocol_error("truncated FINAL_FRAGMENT")
                return
            fid, version = struct.unpack_from("<IQ", payload)
            if fid >= self.layout.num_fragments:
                self._set_protocol_error(f"FINAL_FRAGMENT has unknown fragment {fid}")
                return
            expected_size = 12 + _tensor_nbytes(
                DTYPE_F32, self.layout.fragments[fid].numel
            )
            if len(payload) != expected_size:
                self._set_protocol_error(
                    f"FINAL_FRAGMENT {fid} has {len(payload) - 12} tensor bytes, "
                    f"expected {expected_size - 12} f32 bytes"
                )
                return
            update = FinalFragment(fid, version, payload[12:])
            update_digest = hashlib.sha256(memoryview(update.data).cast("B")).digest()
            with self._final_cond:
                current = self._final_fragments.get(fid)
                if current is not None:
                    if version < current.version:
                        return
                    if version == current.version:
                        current_digest = current.payload_hash
                        if current_digest is None:
                            current_digest = hashlib.sha256(
                                memoryview(current.data).cast("B")
                            ).digest()
                        if update_digest != current_digest:
                            self._set_protocol_error(
                                "conflicting FINAL_FRAGMENT payloads for fragment "
                                f"{fid} version {version}"
                            )
                        return
                self._final_fragments[fid] = update
                self._final_cond.notify_all()
        elif msg_type == MSG_FINAL_MANIFEST:
            try:
                manifest = decode_final_manifest(payload, self.layout.num_fragments)
            except ValueError as exc:
                self._set_protocol_error(f"invalid FINAL_MANIFEST: {exc}")
                return
            with self._final_cond:
                if self._final_manifest is not None and self._final_manifest != manifest:
                    self._set_protocol_error(
                        "syncer sent conflicting FINAL_MANIFEST payloads"
                    )
                    return
                self._final_manifest = manifest
                self.finalizing.set()
                self._final_cond.notify_all()
        else:
            self._set_protocol_error(
                f"unsupported syncer message type {msg_type}; upgrade both peers"
            )

    def _set_protocol_error(self, message: str) -> None:
        self._protocol_failed(self._gen, ProtocolError(message))


def _close_socket(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
