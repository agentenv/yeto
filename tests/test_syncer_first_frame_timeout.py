"""A connection that never sends its first frame is closed after
``--first-frame-timeout-s`` (``syncer/src/server.rs::handle_connection``);
a normal client handshake on the same syncer is unaffected."""

from __future__ import annotations

import socket
import time

from tests.test_dispense_syncer import FP, KEY, Syncer, syncer_binary  # noqa: F401 - the fixture
from yeto.rl.engine import task_dispense as td


def _closed_within(sock: socket.socket, limit_s: float) -> float | None:
    """Seconds until the peer closed ``sock`` (EOF/reset), or None if it stayed open."""
    sock.settimeout(limit_s)
    start = time.monotonic()
    try:
        data = sock.recv(1)
    except socket.timeout:
        return None
    except ConnectionResetError:
        return time.monotonic() - start
    assert data == b"", f"syncer sent unexpected bytes {data!r}"
    return time.monotonic() - start


def test_silent_connection_is_closed_and_handshake_still_works(syncer_binary, tmp_path):  # noqa: F811
    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", "1", "--first-frame-timeout-s", "1",
                    lease_s=600.0)
    silent = []
    try:
        silent = [socket.create_connection(("127.0.0.1", syncer.port), timeout=5) for _ in range(3)]
        # a normal client handshakes and works while the silent ones are pending
        conn = td.DispenseConnection(("127.0.0.1", syncer.port), learner_id=1, incarnation=11, rows=8,
                                     fingerprint=FP, key=KEY, timeout_s=5.0)
        assert conn.claim(1, 2).ranges == ((0, 2),)
        for sock in silent:
            waited = _closed_within(sock, 10.0)
            assert waited is not None, "silent connection was never closed"
        # the handshaken connection outlives the first-frame timeout
        time.sleep(1.5)
        assert conn.claim(2, 1).ranges == ((2, 3),)
        conn.close()
        assert syncer.proc.poll() is None
        assert "first frame" in (tmp_path / "syncer.out").read_text()
    finally:
        for sock in silent:
            sock.close()
        syncer.stop()


def test_silent_connection_is_closed_with_the_default_timeout(syncer_binary, tmp_path):  # noqa: F811
    """No flag: the built-in default (30 s) still closes a silent peer, which
    the syncer before the first-frame timeout held open indefinitely."""
    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", "1", lease_s=600.0)
    silent = socket.create_connection(("127.0.0.1", syncer.port), timeout=5)
    try:
        waited = _closed_within(silent, 45.0)
        assert waited is not None, "silent connection was never closed"
        assert 20.0 <= waited <= 45.0, waited
        assert syncer.proc.poll() is None
    finally:
        silent.close()
        syncer.stop()
