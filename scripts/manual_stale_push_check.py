#!/usr/bin/env python3
"""Force a stale PUSH_FRAGMENT and verify the syncer drops it.

This is a small manual/integration check for the stale-commit guard. It uses
only the Python standard library and talks to the syncer wire protocol directly:

1. Start a one-fragment, one-learner syncer for two outer steps.
2. Send a fresh push for step 1 with base_version=0.
3. Send the step-2 push with stale base_version=0 after the next pull.
5. Assert the syncer logs "stale push dropped" and never "stale push admitted".
"""

from __future__ import annotations

import argparse
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path


MAGIC = 0xD170_C0DE
MSG_HELLO = 1
MSG_INIT_PARAMS = 2
MSG_PULL_REQ = 3
MSG_PUSH_FRAGMENT = 4
MSG_BCAST_FRAGMENT = 5
MSG_SHUTDOWN = 7
DTYPE_F32 = 1
MERGE_AVG = 0


def frame(msg_type: int, payload: bytes) -> bytes:
    return struct.pack("<IBQ", MAGIC, msg_type, len(payload)) + payload


def read_exact(sock: socket.socket, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise EOFError("socket closed")
        out.extend(chunk)
    return bytes(out)


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    header = read_exact(sock, 13)
    magic, msg_type, length = struct.unpack("<IBQ", header)
    if magic != MAGIC:
        raise RuntimeError(f"bad magic 0x{magic:08x}")
    return msg_type, read_exact(sock, length)


def wait_for_pull(sock: socket.socket, step: int, timeout_s: float = 10.0) -> tuple[int, int]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg_type, payload = read_frame(sock)
        if msg_type == MSG_PULL_REQ:
            fid, global_step = struct.unpack("<IQ", payload)
            if global_step == step:
                return fid, global_step
        elif msg_type in {MSG_BCAST_FRAGMENT, MSG_SHUTDOWN}:
            continue
        else:
            raise RuntimeError(f"unexpected frame type {msg_type}")
    raise TimeoutError(f"timed out waiting for pull step {step}")


def send_hello(sock: socket.socket) -> None:
    payload = bytearray()
    payload += struct.pack("<IBI", 0, DTYPE_F32, 1)  # learner_id, dtype, fragments
    payload += struct.pack("<BIQ", MERGE_AVG, 1, 1)  # merge_mode, tensors, numel
    payload += struct.pack("<H", 0)  # no data streams
    sock.sendall(frame(MSG_HELLO, bytes(payload)))


def send_init(sock: socket.socket, value: float) -> None:
    payload = struct.pack("<I", 0) + struct.pack("<f", value)
    sock.sendall(frame(MSG_INIT_PARAMS, payload))


def send_push(sock: socket.socket, step: int, base_version: int, value: float) -> None:
    payload = bytearray()
    payload += struct.pack("<IIQQQIQ", 0, 0, step, base_version, step, 1, 16)
    payload += struct.pack("<f", value)
    sock.sendall(frame(MSG_PUSH_FRAGMENT, bytes(payload)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--syncer-bin",
        type=Path,
        default=Path("syncer/target/debug/yeto-syncer"),
        help="path to a built yeto-syncer binary",
    )
    parser.add_argument("--host", default="localhost", help="host to connect to for the temporary syncer")
    parser.add_argument("--port", type=int, default=29591, help="localhost port for the temporary syncer")
    args = parser.parse_args()
    syncer_bin = args.syncer_bin
    if not syncer_bin.exists():
        raise SystemExit(f"{syncer_bin} does not exist; run `cd syncer && cargo build` first")

    port = args.port
    with tempfile.TemporaryDirectory(prefix="yeto-stale-push-") as td:
        root = Path(td)
        log_path = root / "syncer.log"
        with log_path.open("wb") as log:
            proc = subprocess.Popen(
                [
                    str(syncer_bin),
                    "--port",
                    str(port),
                    "--learners",
                    "1",
                    "--quorum",
                    "1",
                    "--grace-ms",
                    "0",
                    "--sync-interval-steps",
                    "0",
                    "--total-steps",
                    "2",
                    "--outer-lr",
                    "1.0",
                    "--outer-momentum",
                    "0.0",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parents[1],
            )
        try:
            deadline = time.monotonic() + 10.0
            sock = None
            while True:
                candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    candidate.connect((args.host, port))
                    sock = candidate
                    break
                except OSError:
                    candidate.close()
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
            assert sock is not None
            with sock:
                send_hello(sock)
                send_init(sock, 0.0)
                wait_for_pull(sock, 1)
                send_push(sock, step=1, base_version=0, value=-1.0)
                wait_for_pull(sock, 2)
                send_push(sock, step=2, base_version=0, value=-2.0)

                # Drain until shutdown or socket close so the syncer can log.
                try:
                    while True:
                        msg_type, _ = read_frame(sock)
                        if msg_type == MSG_SHUTDOWN:
                            break
                except EOFError:
                    pass

            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        text = log_path.read_text(errors="replace")
        if "stale push admitted" in text:
            print(text)
            raise SystemExit("FAIL: stale push was admitted")
        if "stale push dropped" not in text:
            print(text)
            raise SystemExit("FAIL: stale push drop was not observed")
        if "round had no fresh pushes after stale-drop filter" not in text:
            print(text)
            raise SystemExit("FAIL: no-op rebroadcast path was not observed")

        print("PASS: stale full/f32 push was dropped before merge")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
