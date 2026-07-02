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
"""

from __future__ import annotations

import itertools
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass

from .fragments import FragmentLayout

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

DTYPE_F32 = 1
DTYPE_BF16 = 2

CHUNK_SIZE = 4 * 1024 * 1024

_HEADER = struct.Struct("<IBQ")  # magic, type, payload length
_CHUNK_HEAD = struct.Struct("<QQQ")  # msg_id, total_len, offset


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
    parts.append(struct.pack("<H", num_streams))
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


class SyncerClient:
    """Non-blocking striped syncer connection owned by one learner process."""

    def __init__(
        self,
        addr: tuple[str, int],
        learner_id: int,
        layout: FragmentLayout,
        dtype: int = DTYPE_BF16,
        num_streams: int = 4,
        connect_timeout: float = 900.0,
    ):
        self.addr = addr
        self.learner_id = learner_id
        self.layout = layout
        self.dtype = dtype
        self.num_streams = num_streams
        self.connect_timeout = connect_timeout
        self._queues: list[queue.Queue[bytes | None]] = []
        self._socks: list[socket.socket] = []
        self._pulls: queue.Queue[PullRequest] = queue.Queue()
        self._bcasts: queue.Queue[BcastFragment] = queue.Queue()
        self._reasm: dict[int, tuple[bytearray, list[int]]] = {}
        self._reasm_lock = threading.Lock()
        self._msg_id = itertools.count()
        self._rr = itertools.count()
        self._err: BaseException | None = None
        self.shutdown = threading.Event()

    # -- lifecycle -------------------------------------------------------------

    def _connect_one(self) -> socket.socket:
        last: OSError | None = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.connect_timeout:
            try:
                sock = socket.create_connection(self.addr, timeout=30)
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except OSError as e:  # syncer may not be up yet
                last = e
                time.sleep(2.0)
        raise ConnectionError(f"cannot reach syncer at {self.addr}: {last}")

    def start(self) -> None:
        control = self._connect_one()
        write_frame(
            control, MSG_HELLO, encode_hello(self.learner_id, self.dtype, self.layout, self.num_streams)
        )
        self._socks.append(control)
        for idx in range(self.num_streams):
            s = self._connect_one()
            write_frame(s, MSG_DATA_HELLO, struct.pack("<IH", self.learner_id, idx))
            self._socks.append(s)
        for i, s in enumerate(self._socks):
            q: queue.Queue[bytes | None] = queue.Queue(maxsize=256)
            self._queues.append(q)
            threading.Thread(target=self._send_loop, args=(s, q), name=f"diloco-send-{i}", daemon=True).start()
            threading.Thread(target=self._recv_loop, args=(s,), name=f"diloco-recv-{i}", daemon=True).start()

    def close(self) -> None:
        for q in self._queues:
            q.put(None)
        for s in self._socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    def check_health(self) -> None:
        if self._err is not None:
            raise RuntimeError("syncer connection failed") from self._err

    # -- learner-facing API ------------------------------------------------------

    def send_init(self, fragment_id: int, tensor_bytes: bytes) -> None:
        self._send_large(MSG_INIT_PARAMS, struct.pack("<I", fragment_id) + tensor_bytes)

    def push_fragment(
        self,
        fragment_id: int,
        global_step: int,
        local_step: int,
        c_steps: int,
        c_tokens: int,
        tensor_bytes: bytes,
    ) -> None:
        head = struct.pack(
            "<IIQQIQ", self.learner_id, fragment_id, global_step, local_step, c_steps, c_tokens
        )
        self._send_large(MSG_PUSH_FRAGMENT, head + tensor_bytes)

    def heartbeat(self, local_step: int) -> None:
        self._enqueue(0, self._frame(MSG_HEARTBEAT, struct.pack("<IQ", self.learner_id, local_step)))

    def drain_pulls(self) -> list[PullRequest]:
        return self._drain(self._pulls)

    def drain_updates(self) -> list[BcastFragment]:
        return self._drain(self._bcasts)

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

    def _enqueue(self, stream: int, data: bytes) -> None:
        self.check_health()
        self._queues[stream].put(data)

    def _send_large(self, msg_type: int, payload: bytes) -> None:
        inner = self._frame(msg_type, payload)
        if self.num_streams == 0:
            self._enqueue(0, inner)
            return
        msg_id = next(self._msg_id)
        total = len(inner)
        for offset in range(0, total, CHUNK_SIZE):
            chunk = inner[offset : offset + CHUNK_SIZE]
            envelope = self._frame(MSG_CHUNK, _CHUNK_HEAD.pack(msg_id, total, offset) + chunk)
            stream = 1 + next(self._rr) % self.num_streams
            self._enqueue(stream, envelope)

    def _send_loop(self, sock: socket.socket, q: queue.Queue) -> None:
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                sock.sendall(item)
        except BaseException as e:
            self._err = e

    def _recv_loop(self, sock: socket.socket) -> None:
        try:
            while True:
                msg_type, payload = read_frame(sock)
                if msg_type == MSG_CHUNK:
                    inner = self._reassemble(payload)
                    if inner is not None:
                        self._dispatch(*inner)
                else:
                    self._dispatch(msg_type, payload)
        except BaseException as e:
            if not self.shutdown.is_set():
                self._err = e

    def _reassemble(self, payload: bytes) -> tuple[int, bytes] | None:
        msg_id, total, offset = _CHUNK_HEAD.unpack_from(payload)
        data = payload[_CHUNK_HEAD.size :]
        with self._reasm_lock:
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

    def _dispatch(self, msg_type: int, payload: bytes) -> None:
        if msg_type == MSG_PULL_REQ:
            fid, step = struct.unpack("<IQ", payload)
            self._pulls.put(PullRequest(fid, step))
        elif msg_type == MSG_BCAST_FRAGMENT:
            fid, version = struct.unpack_from("<IQ", payload)
            self._bcasts.put(BcastFragment(fid, version, payload[12:]))
        elif msg_type == MSG_SHUTDOWN:
            self.shutdown.set()
