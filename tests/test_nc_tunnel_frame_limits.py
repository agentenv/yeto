"""The ProxyCommand tunnel bounds what a peer can make it buffer.

A frame header declaring a 2**63-byte payload, or a handshake response that never
ends its header, used to grow the client's buffer without limit."""

from __future__ import annotations

import socket
import threading

import pytest

from yeto.launch.nc import tunnel


def test_frame_reader_rejects_oversized_declared_length():
    reader = tunnel.FrameReader()
    with pytest.raises(tunnel.TunnelError):
        reader.feed(b"\x82\x7f" + (2 ** 63).to_bytes(8, "big") + b"x" * 1024)
    assert len(reader.buf) <= 1034


def test_frame_reader_still_decodes_large_and_split_frames():
    payload = b"a" * 70000
    raw = bytes([0x82, 127]) + len(payload).to_bytes(8, "big") + payload + bytes([0x88, 0])
    reader = tunnel.FrameReader()
    out = reader.feed(raw[:5]) + reader.feed(raw[5:40000]) + reader.feed(raw[40000:])
    assert out == [(2, payload), (8, b"")]


def test_open_ws_bounds_an_endless_handshake_header():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def serve():
        conn, _ = srv.accept()
        with conn:
            conn.recv(4096)
            try:
                while not stop.is_set():
                    conn.sendall(b"X" * 65536)
            except OSError:
                pass

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        with pytest.raises(tunnel.TunnelError):
            tunnel.open_ws(tunnel.Endpoint(f"http://127.0.0.1:{port}", "c=1"), "h:22", timeout=5)
    finally:
        stop.set()
        srv.close()
        t.join(5)
    assert not t.is_alive()
