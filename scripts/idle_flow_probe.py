"""One-off idle-flow probe for A5 (gpu-plan §6, gpu-plan-v2 A5 criterion 4).

Measures whether an idle TCP flow from a Modal CPU container to this machine
(the local head's syncer path) survives 60/180/350/600/900/1200/1800 s of
silence. Each idle point is its own connection, opened at the same time and
held silent for that long; then the client sends ``ping <t>`` and must read
``pong <t>`` within 15 s. The listener logs every accept/close.

Local side (this machine, the head's public IP)::

    python scripts/idle_flow_probe.py listen --port 29400

Remote side (Modal CPU container, ~$0.05, hard timeout 40 min)::

    /tmp/modal-venv/bin/modal run scripts/idle_flow_probe.py --host <SYNCER_PUBLIC_IP> --port 29400

Prints one JSON object: {"host", "port", "results": [{"idle_s", "alive", "error"}],
"idle_flow_timeout_s"}; ``idle_flow_timeout_s`` is the longest idle point that
survived when a longer one failed, else None (no loss up to the longest point).
Use it for --rl-elastic-idle-flow-timeout-s. No GPU, no yeto code involved.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time

IDLE_POINTS = (60, 180, 350, 600, 900, 1200, 1800)


def _probe_one(host: str, port: int, idle_s: float, timeout_s: float = 15.0) -> dict:
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.sendall(f"hello {idle_s}\n".encode())
            time.sleep(idle_s)
            sock.sendall(f"ping {idle_s}\n".encode())
            sock.settimeout(timeout_s)
            data = b""
            while b"\n" not in data:
                chunk = sock.recv(64)
                if not chunk:
                    raise ConnectionError("closed by peer/path")
                data += chunk
            ok = data.split(b"\n")[0].decode() == f"pong {idle_s}"
            return {"idle_s": idle_s, "alive": ok, "error": None if ok else f"bad reply {data!r}"}
    except Exception as exc:  # noqa: BLE001 - a dropped flow is the measurement
        return {"idle_s": idle_s, "alive": False, "error": repr(exc)}


def probe(host: str, port: int, idle_points=IDLE_POINTS) -> dict:
    results: list[dict] = []
    threads = [threading.Thread(target=lambda t=t: results.append(_probe_one(host, port, t)))
               for t in idle_points]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    results.sort(key=lambda r: r["idle_s"])
    failed = [r["idle_s"] for r in results if not r["alive"]]
    alive_below = [r["idle_s"] for r in results if r["alive"] and (not failed or r["idle_s"] < min(failed))]
    timeout = (max(alive_below) if alive_below else 0) if failed else None
    return {"host": host, "port": port, "results": results, "idle_flow_timeout_s": timeout}


def listen(port: int, host: str = "0.0.0.0", *, ready: threading.Event | None = None,
           stop: threading.Event | None = None) -> None:
    server = socket.create_server((host, port), reuse_port=False)
    server.settimeout(0.5)
    if ready is not None:
        ready.set()

    def serve(conn: socket.socket, addr) -> None:
        with conn:
            buf = b""
            while True:
                try:
                    chunk = conn.recv(64)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode(errors="replace")
                    print(f"{time.time():.0f} {addr} {text}", flush=True)
                    if text.startswith("ping "):
                        conn.sendall(("pong " + text[5:] + "\n").encode())
            print(f"{time.time():.0f} {addr} closed", flush=True)

    with server:
        while stop is None or not stop.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            threading.Thread(target=serve, args=(conn, addr), daemon=True).start()


try:  # the Modal entry point (only when run with `modal run`)
    import modal

    app = modal.App("yeto-idle-flow-probe")

    @app.function(cpu=0.25, timeout=40 * 60)
    def remote_probe(host: str, port: int) -> dict:
        return probe(host, port)

    @app.local_entrypoint()
    def main(host: str, port: int = 29400) -> None:
        print(json.dumps(remote_probe.remote(host, port), indent=2))
except ImportError:  # pragma: no cover - local listener only
    app = None


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "listen":
        listen(int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 29400)
    elif len(sys.argv) >= 2 and sys.argv[1] == "probe":  # local dry run
        print(json.dumps(probe(sys.argv[2], int(sys.argv[3]),
                               tuple(float(x) for x in sys.argv[4:]) or IDLE_POINTS), indent=2))
    else:
        print(__doc__)
