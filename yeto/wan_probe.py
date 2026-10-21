"""Island-to-syncer link-bandwidth probe (S26 E3b).

E3/E3b showed that the sync share follows the island *container*: on a slow
container every TCP stream is slow at once (1-5 MB/s each, both directions),
so no stream scheduling can help. An island can measure its link to the head
before it loads a model and exit early when the link is too slow; the
launcher then requests a fresh container (``--wan-probe-relaunches``).

Each probe connection sends one ``MSG_BW_PROBE`` frame (HMAC-sealed like
HELLO), uploads ``upload_bytes`` raw bytes, reads the syncer's 8-byte
acknowledgement, then reads ``download_bytes`` raw bytes. The syncer never
connects a probe to a learner session.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .protocol import MAGIC, MSG_BW_PROBE, ProtocolError, seal_hello, write_frame

# Exit code of an island that rejected its own link (the launcher may relaunch it).
EXIT_SLOW_LINK = 75
PROBE_EVENT = "rl_wan_probe"
PROBE_MARKER = "[yeto-wan-probe]"
_BLOCK = memoryview(bytes(1 << 20))


@dataclass(frozen=True)
class ProbeResult:
    streams: int
    upload_bytes: int
    download_bytes: int
    upload_seconds: float
    download_seconds: float
    error: str | None = None

    @property
    def upload_mbps(self) -> float:
        return self.upload_bytes / self.upload_seconds / 1e6 if self.upload_seconds > 0 else 0.0

    @property
    def download_mbps(self) -> float:
        return self.download_bytes / self.download_seconds / 1e6 if self.download_seconds > 0 else 0.0

    def accepted(self, min_upload_mbps: float, min_download_mbps: float) -> bool:
        return (
            self.error is None
            and self.upload_mbps >= min_upload_mbps
            and self.download_mbps >= min_download_mbps
        )

    def fields(self) -> dict[str, object]:
        return {
            "wan_probe/streams": self.streams,
            "wan_probe/upload_bytes": self.upload_bytes,
            "wan_probe/download_bytes": self.download_bytes,
            "wan_probe/upload_mbps": round(self.upload_mbps, 3),
            "wan_probe/download_mbps": round(self.download_mbps, 3),
            "wan_probe/error": self.error,
        }


def encode_probe(upload_bytes: int, download_bytes: int, key: bytes | None) -> bytes:
    return seal_hello(struct.pack("<QQ", upload_bytes, download_bytes), key)


def _one_stream(addr, upload: int, download: int, key, timeout: float, out: dict) -> None:
    sock = socket.create_connection(addr, timeout=timeout)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        write_frame(sock, MSG_BW_PROBE, encode_probe(upload, download, key))
        start = time.monotonic()
        left = upload
        while left:
            take = min(left, len(_BLOCK))
            sock.sendall(_BLOCK[:take])
            left -= take
        ack = b""
        while len(ack) < 8:
            part = sock.recv(8 - len(ack))
            if not part:
                # The syncer refused the probe: it answers with an ERROR frame
                # and closes; the ERROR body is the reason.
                raise ConnectionError("syncer closed the probe before acknowledging the upload")
            ack += part
        if ack.startswith(struct.pack("<I", MAGIC)):
            raise ProtocolError("syncer answered the probe with a frame (refused)")
        (received,) = struct.unpack("<Q", ack)
        if received != upload:
            raise ProtocolError(f"syncer acknowledged {received} of {upload} probe bytes")
        acked = time.monotonic()
        left = download
        while left:
            part = sock.recv(min(left, 1 << 20))
            if not part:
                raise ConnectionError(f"probe download ended {left} bytes early")
            left -= len(part)
        out["up"] = acked - start
        out["down"] = time.monotonic() - acked
    finally:
        sock.close()


def probe_link(
    addr: tuple[str, int],
    *,
    streams: int,
    bytes_per_direction: int,
    key: bytes | None,
    timeout: float = 120.0,
    stream_fn: Callable = _one_stream,
) -> ProbeResult:
    """Measure aggregate upload and download rates over ``streams`` parallel
    connections; ``bytes_per_direction`` is split evenly over the streams.
    A rate is total bytes over the slowest stream's time in that direction."""
    if streams < 1 or bytes_per_direction < streams:
        raise ValueError("probe needs at least one stream and one byte per stream")
    per = bytes_per_direction // streams
    results = [dict() for _ in range(streams)]
    errors: list[str] = []

    def run(index: int) -> None:
        try:
            stream_fn(addr, per, per, key, timeout, results[index])
        except (OSError, ProtocolError) as error:
            errors.append(f"{type(error).__name__}: {error}")

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(streams)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout + 5)
    done = [r for r in results if "up" in r and "down" in r]
    if len(done) < streams and not errors:
        errors.append(f"{streams - len(done)} probe stream(s) did not finish in {timeout:.0f}s")
    return ProbeResult(
        streams=streams,
        upload_bytes=per * len(done),
        download_bytes=per * len(done),
        upload_seconds=max((r["up"] for r in done), default=0.0),
        download_seconds=max((r["down"] for r in done), default=0.0),
        error=errors[0] if errors else None,
    )

