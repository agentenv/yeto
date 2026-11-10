"""SSH reach to National Compute nodes through the workspace gateway.

A leased node's port 22 answers only inside National Compute (the workspace
reaches it, the launcher host does not). The workspace gateway forwards HTTP
upgrades on an opened port (``port.open`` + the ``gardengate_port`` cookie), so
``relay.mjs`` runs in the workspace as a WebSocket -> TCP relay and this module
is the ``ProxyCommand`` on the launcher host: ``ssh`` (and every sky SSH node
pool step, which reads the OpenSSH config of the host) speaks to the node over
one WebSocket per connection. No dependency beyond the standard library.

    ProxyCommand python -m yeto.launch.nc proxy --account 1 %h %p

The cookie comes from ``port_open.mjs`` (the NC controller tool) and is cached
0600 in ``state_dir``; a refused handshake (gateway 401/403) refreshes it once.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

RELAY_PORT = 8791
HERE = Path(__file__).resolve().parent
RELAY_JS = HERE / "relay.mjs"
PORT_OPEN_JS = HERE / "port_open.mjs"


class TunnelError(RuntimeError):
    pass


class HandshakeRefused(TunnelError):
    def __init__(self, status: int, text: str):
        super().__init__(f"relay handshake refused: HTTP {status} {text[:120]!r}")
        self.status = status


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    cookie: str


def state_dir() -> Path:
    path = Path(os.environ.get("YETO_NC_STATE_DIR", "~/.yeto/nc")).expanduser()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def cookie_path(account: int, port: int = RELAY_PORT) -> Path:
    return state_dir() / f"relay-{account}-{port}.json"


def tool_dir() -> str:
    """The controller tool directory (``NC_TOOL_DIR`` or the pool's ``provider.tool_dir``)."""
    from .client import tools  # noqa: PLC0415

    path = tools().tool_dir
    if not path:
        raise TunnelError("the National Compute controller tool directory is not set: "
                          "set NC_TOOL_DIR or provider.tool_dir in the pool config")
    return path


def node_bin() -> str:
    from .client import tools  # noqa: PLC0415

    return tools().node_bin


def fetch_endpoint(account: int, port: int = RELAY_PORT, *, run=subprocess.run) -> Endpoint:
    """``port.open`` through the controller tool; store the cookie 0600 and return it."""
    out = run([node_bin(), str(PORT_OPEN_JS), "--tool-dir", tool_dir(), "--account", str(account),
               "--port", str(port)], capture_output=True, text=True, timeout=120, check=False)
    if out.returncode != 0:
        raise TunnelError(f"port_open.mjs failed (exit {out.returncode}): {out.stderr.strip()[-300:]}")
    doc = json.loads(out.stdout.strip().splitlines()[-1])
    path = cookie_path(account, port)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"base_url": doc["base_url"], "cookie": doc["cookie"]}, f)
    os.replace(tmp, path)
    return Endpoint(doc["base_url"], doc["cookie"])


def cached_endpoint(account: int, port: int = RELAY_PORT) -> Endpoint | None:
    try:
        doc = json.loads(cookie_path(account, port).read_text())
        return Endpoint(doc["base_url"], doc["cookie"])
    except (OSError, ValueError, KeyError):
        return None


# --- WebSocket client framing (RFC 6455, client side masks) -----------------

def encode_frame(payload: bytes, opcode: int = 2, mask: bytes | None = None) -> bytes:
    mask = os.urandom(4) if mask is None else mask
    n = len(payload)
    if n < 126:
        head = bytes([0x80 | opcode, 0x80 | n])
    elif n < 65536:
        head = bytes([0x80 | opcode, 0x80 | 126]) + n.to_bytes(2, "big")
    else:
        head = bytes([0x80 | opcode, 0x80 | 127]) + n.to_bytes(8, "big")
    masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
    return head + mask + masked


class FrameReader:
    """Incremental decoder of server frames: ``feed`` bytes, get ``(opcode, payload)``."""

    def __init__(self):
        self.buf = b""

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.buf += data
        out = []
        while len(self.buf) >= 2:
            op, ln, off = self.buf[0] & 15, self.buf[1] & 127, 2
            masked = self.buf[1] & 128
            if ln == 126:
                if len(self.buf) < 4:
                    break
                ln, off = int.from_bytes(self.buf[2:4], "big"), 4
            elif ln == 127:
                if len(self.buf) < 10:
                    break
                ln, off = int.from_bytes(self.buf[2:10], "big"), 10
            mask = b""
            if masked:
                mask, off = self.buf[off:off + 4], off + 4
            if len(self.buf) < off + ln:
                break
            payload = self.buf[off:off + ln]
            if masked:
                payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
            self.buf = self.buf[off + ln:]
            out.append((op, payload))
        return out


def handshake_request(endpoint: Endpoint, target: str, key: str) -> tuple[str, int, bytes]:
    url = urlsplit(endpoint.base_url)
    path = (url.path.rstrip("/") or "") + f"/?target={target}"
    host = url.hostname or ""
    port = url.port or (443 if url.scheme == "https" else 80)
    req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nCookie: {endpoint.cookie}\r\n\r\n")
    return host, port, req.encode()


def open_ws(endpoint: Endpoint, target: str, *, timeout: float = 30.0) -> tuple[socket.socket, bytes]:
    """Connect and upgrade; return the socket and any bytes read past the 101 header."""
    key = base64.b64encode(os.urandom(16)).decode()
    host, port, req = handshake_request(endpoint, target, key)
    raw = socket.create_connection((host, port), timeout=timeout)
    sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host) \
        if endpoint.base_url.startswith("https") else raw
    sock.sendall(req)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode(errors="replace")
    parts = status_line.split()
    status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    if status != 101:
        sock.close()
        raise HandshakeRefused(status, head.decode(errors="replace") + rest.decode(errors="replace"))
    sock.settimeout(None)
    return sock, rest


def pipe(sock: socket.socket, rest: bytes, stdin=None, stdout=None) -> None:
    """Copy stdin -> binary frames and frames -> stdout until either side closes."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    lock = threading.Lock()
    done = threading.Event()

    def up():
        try:
            while True:
                data = os.read(stdin.fileno(), 65536) if hasattr(stdin, "fileno") else stdin.read(65536)
                if not data:
                    break
                with lock:
                    sock.sendall(encode_frame(data))
        except OSError:
            pass
        finally:
            try:
                with lock:
                    sock.sendall(encode_frame(b"", opcode=8))
            except OSError:
                pass
            done.set()

    threading.Thread(target=up, daemon=True).start()
    reader = FrameReader()
    try:
        frames = reader.feed(rest)
        while True:
            for op, payload in frames:
                if op == 8:
                    return
                if op == 9:
                    with lock:
                        sock.sendall(encode_frame(payload, opcode=10))
                elif op in (0, 1, 2) and payload:
                    stdout.write(payload)
                    stdout.flush()
            chunk = sock.recv(65536)
            if not chunk:
                return
            frames = reader.feed(chunk)
    finally:
        sock.close()


def proxy(account: int, host: str, port: int, *, relay_port: int = RELAY_PORT) -> int:
    """``ProxyCommand`` entry: dial ``host:port`` through the workspace relay."""
    target = f"{host}:{port}"
    endpoint = cached_endpoint(account, relay_port) or fetch_endpoint(account, relay_port)
    try:
        sock, rest = open_ws(endpoint, target)
    except HandshakeRefused as exc:
        if exc.status not in (401, 403, 302, 502):
            raise
        endpoint = fetch_endpoint(account, relay_port)  # stale cookie, or the port was closed
        sock, rest = open_ws(endpoint, target)
    pipe(sock, rest)
    return 0


DEFAULT_RELAY_DIR = "/data/home/.yeto-nc"  # the National Compute workspace's persistent home


def relay_dir() -> str:
    """Directory of the relay in the account's workspace (``YETO_NC_RELAY_DIR``;
    default: the workspace's persistent home). One relay per directory."""
    value = os.environ.get("YETO_NC_RELAY_DIR") or DEFAULT_RELAY_DIR
    if not value.startswith("/") or any(c.isspace() or c in "'\"$`;&|" for c in value):
        raise ValueError(f"YETO_NC_RELAY_DIR {value!r} must be an absolute path without spaces or shell characters")
    return value.rstrip("/")


def relay_start_command(allow: list[str], *, port: int = RELAY_PORT, directory: str | None = None) -> str:
    """Shell (run in the workspace) that installs relay.mjs, writes the allow list and
    starts the relay once (an already running relay is reused; the allow file is
    re-read per connection)."""
    d = directory or relay_dir()
    src = base64.b64encode(RELAY_JS.read_bytes()).decode()
    allow_b64 = base64.b64encode(("\n".join(allow) + "\n").encode()).decode()
    alive = _relay_alive(d)
    return (
        f"mkdir -p {d} && chmod 700 {d} && "
        f"echo {src} | base64 -d > {d}/relay.mjs && "
        f"echo {allow_b64} | base64 -d > {d}/relay-allow && "
        f"{alive} || (RELAY_PORT={port} "
        f"RELAY_ALLOW={d}/relay-allow nohup node {d}/relay.mjs "
        f">{d}/relay.log 2>&1 </dev/null & echo $! > {d}/relay.pid); sleep 1; "
        f"{alive} && echo __RELAY_UP__ || echo __RELAY_DOWN__"
    )


# the relay by its PID file (no process-name matching: it would match this very shell);
# the cmdline check guards against a reused PID
def _relay_alive(d: str) -> str:
    return (f"{{ p=$(cat {d}/relay.pid 2>/dev/null); [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null "
            f"&& tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q '{d}/relay.mjs'; }}")


_RELAY_ALIVE = _relay_alive(DEFAULT_RELAY_DIR)
