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

import itertools
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
# Reserved for the protocol-level error frame implemented by protocol v4.
MSG_ERROR = 10
MSG_FINAL_MANIFEST = 11
MSG_FINAL_ACK = 12
MSG_FINAL_FRAGMENT = 13

FINALIZATION_REVISION = 1
FINALIZATION_TIMEOUT = 900.0

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
_FINAL_MANIFEST_HEAD = struct.Struct("<HQI")  # revision, global_step, fragments
_FINAL_ACK = struct.Struct("<HQ")  # revision, global_step


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


def encode_hello(learner_id: int, dtype: int, layout: FragmentLayout, num_streams: int) -> bytes:
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
    return b"".join(parts)


def encode_final_manifest(global_step: int, versions: tuple[int, ...] | list[int]) -> bytes:
    """Golden-compatible FINAL_MANIFEST encoder used by protocol tests."""
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


@dataclass
class BcastFragment:
    fragment_id: int
    version: int
    data: bytes  # raw tensor bytes in the session dtype


@dataclass
class FinalFragment:
    fragment_id: int
    version: int
    data: bytes  # authoritative coordinator tensor bytes, always f32


@dataclass(frozen=True)
class FinalManifest:
    global_step: int
    versions: tuple[int, ...]


@dataclass
class _Outbound:
    data: bytes
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
    ):
        self.addr = addr
        self.learner_id = learner_id
        self.layout = layout
        self.dtype = dtype
        self.num_streams = num_streams
        self.connect_timeout = connect_timeout
        self.max_reconnects = max_reconnects
        self.finalization_timeout = finalization_timeout
        self._queues: list[queue.Queue[_Outbound | None]] = []
        self._socks: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        self._pulls: queue.Queue[PullRequest] = queue.Queue()
        self._bcasts: queue.Queue[BcastFragment] = queue.Queue()
        # Raw globals are retained independently of the drain queue so a
        # final fragment that outruns its control-stream manifest can be
        # re-applied exactly after the manifest arrives.
        self._final_fragments: dict[int, FinalFragment] = {}
        self._final_manifest: FinalManifest | None = None
        self._final_ack_step: int | None = None
        self.finalizing = threading.Event()
        self.finalized = threading.Event()
        self._final_cond = threading.Condition()
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
                control, MSG_HELLO, encode_hello(self.learner_id, self.dtype, self.layout, self.num_streams)
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
        with self._reasm_lock:
            self._reasm.clear()  # partial inbound messages died with the sockets
        self._drain(self._pulls)  # stale pull requests; answering them is pointless
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
        with self._final_cond:
            self._final_cond.notify_all()

    def check_health(self) -> None:
        """Raise only for unrecoverable failures. While a reconnect is being
        attempted this is a no-op; the training loop keeps stepping locally."""
        if self._err is not None:
            raise RuntimeError(f"syncer connection failed: {self._err}") from self._err

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

    def wait_for_final_fragments(
        self, timeout: float | None = None
    ) -> tuple[FinalManifest, list[FinalFragment]]:
        """Wait for a manifest and its exact raw fragment versions.

        FINAL_MANIFEST travels on control while fragments may be striped over
        data sockets, so either can arrive first. The retained raw cache makes
        both orders equivalent and survives a connection-group redial.
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
        with self._lock:
            if gen != self._gen:
                return  # late message from a dead group
            # Keep the generation check and every resulting state mutation
            # atomic with respect to teardown/redial. Otherwise an obsolete
            # receiver could pass the check, lose the race to a reconnect,
            # then publish a stale terminal manifest or shutdown afterward.
            self._dispatch_live(msg_type, payload)

    def _dispatch_live(self, msg_type: int, payload: bytes) -> None:
        if msg_type == MSG_SHUTDOWN:
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
            fid, step = struct.unpack("<IQ", payload)
            self._pulls.put(PullRequest(fid, step))
        elif msg_type == MSG_BCAST_FRAGMENT:
            if len(payload) < 12:
                self._set_protocol_error("truncated BCAST_FRAGMENT")
                return
            fid, version = struct.unpack_from("<IQ", payload)
            if fid >= self.layout.num_fragments:
                self._set_protocol_error(f"BCAST_FRAGMENT has unknown fragment {fid}")
                return
            update = BcastFragment(fid, version, payload[12:])
            self._bcasts.put(update)
        elif msg_type == MSG_FINAL_FRAGMENT:
            if len(payload) < 12:
                self._set_protocol_error("truncated FINAL_FRAGMENT")
                return
            fid, version = struct.unpack_from("<IQ", payload)
            if fid >= self.layout.num_fragments:
                self._set_protocol_error(f"FINAL_FRAGMENT has unknown fragment {fid}")
                return
            expected_size = 12 + self.layout.fragments[fid].numel * 4
            if len(payload) != expected_size:
                self._set_protocol_error(
                    f"FINAL_FRAGMENT {fid} has {len(payload) - 12} tensor bytes, "
                    f"expected {expected_size - 12} f32 bytes"
                )
                return
            update = FinalFragment(fid, version, payload[12:])
            with self._final_cond:
                current = self._final_fragments.get(fid)
                if current is None or version >= current.version:
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
        with self._lock:
            self._err = RuntimeError(message)
        with self._final_cond:
            self._final_cond.notify_all()


def _close_socket(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
