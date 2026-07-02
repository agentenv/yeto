"""Python side of the learner <-> syncer wire protocol (docs/PROTOCOL.md v2).

The learner training loop must never block on the WAN, so socket I/O runs on
background threads:

  - a sender thread drains an outgoing queue (HELLO/INIT/PUSH frames),
  - a receiver thread parses inbound frames into two inboxes: pull requests
    (answered by the training loop at inner-step boundaries) and fragment
    broadcasts (applied between inner steps).
"""

from __future__ import annotations

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

DTYPE_F32 = 1
DTYPE_BF16 = 2

_HEADER = struct.Struct("<IBQ")  # magic, type, payload length


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


def encode_hello(learner_id: int, dtype: int, layout: FragmentLayout) -> bytes:
    parts = [struct.pack("<IBI", learner_id, dtype, layout.num_fragments)]
    for frag in layout.fragments:
        parts.append(struct.pack("<BI", frag.merge_mode, len(frag.tensors)))
        parts.append(struct.pack(f"<{len(frag.tensors)}Q", *(n for _, n in frag.tensors)))
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
    """Non-blocking syncer connection owned by one learner process.

    Usage:
        client = SyncerClient(addr, learner_id, layout, dtype)
        client.start()
        client.send_init(fid, tensor_bytes)   # learner 0 only
        ...each inner step boundary:
            for req in client.drain_pulls(): ...answer with push_fragment...
            for frag in client.drain_updates(): ...overwrite + reset counters...
    """

    def __init__(
        self,
        addr: tuple[str, int],
        learner_id: int,
        layout: FragmentLayout,
        dtype: int = DTYPE_BF16,
        connect_timeout: float = 900.0,
    ):
        self.addr = addr
        self.learner_id = learner_id
        self.layout = layout
        self.dtype = dtype
        self.connect_timeout = connect_timeout
        self._out: queue.Queue[tuple[int, bytes] | None] = queue.Queue(maxsize=64)
        self._pulls: queue.Queue[PullRequest] = queue.Queue()
        self._bcasts: queue.Queue[BcastFragment] = queue.Queue()
        self._sock: socket.socket | None = None
        self._err: BaseException | None = None
        self.shutdown = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        last: OSError | None = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.connect_timeout:
            try:
                self._sock = socket.create_connection(self.addr, timeout=30)
                break
            except OSError as e:  # syncer may not be up yet
                last = e
                time.sleep(2.0)
        if self._sock is None:
            raise ConnectionError(f"cannot reach syncer at {self.addr}: {last}")
        self._sock.settimeout(None)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        write_frame(self._sock, MSG_HELLO, encode_hello(self.learner_id, self.dtype, self.layout))
        for fn, name in ((self._send_loop, "diloco-send"), (self._recv_loop, "diloco-recv")):
            threading.Thread(target=fn, name=name, daemon=True).start()

    def close(self) -> None:
        self._out.put(None)
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()

    def check_health(self) -> None:
        if self._err is not None:
            raise RuntimeError("syncer connection failed") from self._err

    # -- learner-facing API ----------------------------------------------------

    def send_init(self, fragment_id: int, tensor_bytes: bytes) -> None:
        self._enqueue(MSG_INIT_PARAMS, struct.pack("<I", fragment_id) + tensor_bytes)

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
        self._enqueue(MSG_PUSH_FRAGMENT, head + tensor_bytes)

    def heartbeat(self, local_step: int) -> None:
        try:
            self._out.put_nowait((MSG_HEARTBEAT, struct.pack("<IQ", self.learner_id, local_step)))
        except queue.Full:
            pass  # best-effort

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

    # -- internals ---------------------------------------------------------------

    def _enqueue(self, msg_type: int, payload: bytes) -> None:
        self.check_health()
        self._out.put((msg_type, payload))

    def _send_loop(self) -> None:
        try:
            while True:
                item = self._out.get()
                if item is None:
                    return
                write_frame(self._sock, item[0], item[1])
        except BaseException as e:
            self._err = e

    def _recv_loop(self) -> None:
        try:
            while True:
                msg_type, payload = read_frame(self._sock)
                if msg_type == MSG_PULL_REQ:
                    fid, step = struct.unpack("<IQ", payload)
                    self._pulls.put(PullRequest(fid, step))
                elif msg_type == MSG_BCAST_FRAGMENT:
                    fid, version = struct.unpack_from("<IQ", payload)
                    self._bcasts.put(BcastFragment(fid, version, payload[12:]))
                elif msg_type == MSG_SHUTDOWN:
                    self.shutdown.set()
                    return
        except BaseException as e:
            self._err = e
