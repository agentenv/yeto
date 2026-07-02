"""Python side of the learner <-> syncer wire protocol (docs/PROTOCOL.md).

The learner training loop must never block on the WAN, so all socket I/O runs
on background threads:

  - a sender thread drains an outgoing queue (HELLO/INIT/PUSH frames), and
  - a receiver thread parses BCAST_FRAGMENT frames into an inbox that the
    training loop applies between inner steps.
"""

from __future__ import annotations

import queue
import socket
import struct
import threading
from dataclasses import dataclass

MAGIC = 0xD170C0DE

MSG_HELLO = 1
MSG_INIT_PARAMS = 2
MSG_PUSH_FRAGMENT = 3
MSG_BCAST_FRAGMENT = 4
MSG_HEARTBEAT = 5
MSG_SHUTDOWN = 6

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


@dataclass
class BcastFragment:
    fragment_id: int
    version: int
    data: bytes  # raw tensor bytes in the session dtype


class SyncerClient:
    """Non-blocking client owned by one learner process.

    Usage:
        client = SyncerClient(addr, learner_id, fragment_numels, dtype)
        client.start()
        client.send_init(fid, tensor_bytes)          # learner 0 only
        client.push_fragment(fid, base_version, steps, tokens, tensor_bytes)
        for frag in client.drain_updates(): ...      # between inner steps
    """

    def __init__(
        self,
        addr: tuple[str, int],
        learner_id: int,
        fragment_numels: list[int],
        dtype: int = DTYPE_BF16,
        connect_timeout: float = 600.0,
    ):
        self.addr = addr
        self.learner_id = learner_id
        self.fragment_numels = fragment_numels
        self.dtype = dtype
        self.connect_timeout = connect_timeout
        self._out: queue.Queue[tuple[int, bytes] | None] = queue.Queue(maxsize=64)
        self._in: queue.Queue[BcastFragment] = queue.Queue()
        self._sock: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._err: BaseException | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        deadline = threading.Event()
        last = None
        import time

        t0 = time.monotonic()
        while time.monotonic() - t0 < self.connect_timeout:
            try:
                self._sock = socket.create_connection(self.addr, timeout=30)
                break
            except OSError as e:  # syncer may not be up yet
                last = e
                deadline.wait(2.0)
        if self._sock is None:
            raise ConnectionError(f"cannot reach syncer at {self.addr}: {last}")
        self._sock.settimeout(None)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        hello = struct.pack("<IIB", self.learner_id, len(self.fragment_numels), self.dtype)
        hello += struct.pack(f"<{len(self.fragment_numels)}Q", *self.fragment_numels)
        write_frame(self._sock, MSG_HELLO, hello)

        for fn, name in ((self._send_loop, "diloco-send"), (self._recv_loop, "diloco-recv")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

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

    # -- learner-facing API --------------------------------------------------

    def send_init(self, fragment_id: int, tensor_bytes: bytes) -> None:
        self._enqueue(MSG_INIT_PARAMS, struct.pack("<I", fragment_id) + tensor_bytes)

    def push_fragment(
        self, fragment_id: int, base_version: int, steps: int, tokens: int, tensor_bytes: bytes
    ) -> None:
        head = struct.pack("<IIQIQ", self.learner_id, fragment_id, base_version, steps, tokens)
        self._enqueue(MSG_PUSH_FRAGMENT, head + tensor_bytes)

    def heartbeat(self, local_step: int) -> None:
        try:
            self._out.put_nowait((MSG_HEARTBEAT, struct.pack("<IQ", self.learner_id, local_step)))
        except queue.Full:
            pass  # heartbeats are best-effort

    def drain_updates(self) -> list[BcastFragment]:
        out = []
        while True:
            try:
                out.append(self._in.get_nowait())
            except queue.Empty:
                return out

    def wait_update(self, timeout: float) -> BcastFragment | None:
        try:
            return self._in.get(timeout=timeout)
        except queue.Empty:
            return None

    # -- internals -------------------------------------------------------------

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
                if msg_type == MSG_BCAST_FRAGMENT:
                    fid, version = struct.unpack_from("<IQ", payload)
                    self._in.put(BcastFragment(fid, version, payload[12:]))
                elif msg_type == MSG_SHUTDOWN:
                    return
        except BaseException as e:
            self._err = e
