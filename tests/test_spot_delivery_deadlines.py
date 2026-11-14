"""Release delivery never waits without a wall-clock bound: a peer that drips
bytes (each read inside the socket timeout) and an rclone that hangs both end
in an error the pull loop retries, instead of stalling the island forever."""

from __future__ import annotations

import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from yeto.rl.spot import peer_fanout, s3_delivery


class _Drip(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "1000000")
        self.end_headers()
        try:
            for _ in range(200):  # 1 byte every 0.1 s: never idle for a 1 s socket timeout
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.1)
        except OSError:
            pass


def _call_bounded(fn, limit_s):
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as error:  # noqa: BLE001
            box["error"] = error

    t = threading.Thread(target=run, daemon=True)
    began = time.monotonic()
    t.start()
    t.join(limit_s)
    return (not t.is_alive()), time.monotonic() - began, box


def test_peer_fetch_has_a_total_deadline():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Drip)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/r/file"
        done, took, box = _call_bounded(lambda: peer_fanout.urllib_fetch(url, 1.0), 8.0)
        assert done, "a dripping peer held the fetch past its timeout"
        assert isinstance(box.get("error"), peer_fanout.PeerError)
        assert took < 5.0
    finally:
        server.shutdown()
        server.server_close()


def test_hung_rclone_is_cut_off(tmp_path, monkeypatch):
    rclone = tmp_path / "rclone"
    rclone.write_text("#!/bin/sh\nexec sleep 600\n")
    rclone.chmod(rclone.stat().st_mode | stat.S_IEXEC)
    for key, value in {"S3_BUCKET": "b", "S3_PREFIX": "p", "S3_REGION": "r", "RCLONE": str(rclone),
                       "S3_RCLONE_CALL_TIMEOUT_S": "1"}.items():
        monkeypatch.setenv(key, value)
    s3 = s3_delivery.S3()
    done, took, box = _call_bounded(lambda: s3.get_bytes("LATEST.json"), 10.0)
    assert done, "a hung rclone cat held the pull loop"
    assert isinstance(box.get("error"), RuntimeError)
    assert took < 6.0


def test_copy_timeout_defaults_are_generous(monkeypatch):
    for key, value in {"S3_BUCKET": "b", "S3_PREFIX": "p", "S3_REGION": "r"}.items():
        monkeypatch.setenv(key, value)
    s3 = s3_delivery.S3()
    assert s3.call_timeout_s >= 60
    assert s3.copy_timeout_s >= 3600
