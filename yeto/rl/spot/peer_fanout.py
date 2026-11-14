"""Near-region fan-out of policy releases: islands of one cluster pull a release
from a peer that already has it, and serve what they hold to the next one.

Object storage sits far from the sampler nodes (R2 / S3 across regions, one
egress per node); peers share a rack or a cluster network. With ``peers``
(``host:port`` of the other islands' file servers, from the sampler spec or
``YETO_RELEASE_PEERS``), :func:`pull_from_peers` asks each peer for the
release's cut manifest (``GET /manifests/<release>.json``); the first peer
whose manifest matches the digest the object-store pointer named serves the
files (``GET /<release>/<file>``) into a staging directory, and
``cut_manifest.place`` checks every size and sha256 before the local store
moves. A peer is untrusted transport, nothing more: the digest comes from the
authenticated pointer, the file hashes from the manifest behind that digest,
so a wrong or malicious peer can only make the pull fall back to the object
store (:func:`yeto.rl.spot.s3_delivery.pull` with ``peers``), never place a
wrong byte. After a pull the node's own :class:`ReleaseFileServer` serves the
release (it reads straight from the store: a release is served once its
directory and manifest are both in place; a half-staged one never is).

Minimal on purpose: one connection per file, no range requests, no peer
scoring; the file server is Python's ``ThreadingHTTPServer``. Throughput
tuning (parallel files, chunked transfer, peer choice by load) comes after
the first MI355X run measured the baseline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from yeto.rl.spot_rollout import cut_manifest as cm

MANIFESTS = "manifests"  # <store>/manifests/<release>.json: what the node can serve
PEERS_ENV = "YETO_RELEASE_PEERS"  # comma-separated host:port list
DEFAULT_PORT = 8431
_PEER = re.compile(r"^(?P<host>[A-Za-z0-9.\-\[\]:]+?):(?P<port>\d{1,5})$")
# (url, timeout seconds) -> body bytes; raises PeerError when the peer has no such object
Fetch = Callable[[str, float], bytes]


class PeerError(RuntimeError):
    """A peer did not deliver (unreachable, no such release, wrong digest, bad bytes)."""


def parse_peers(value: Iterable[str] | str | None) -> list[str]:
    """``host:port`` entries from a list or a comma-separated string (an entry
    without a port gets :data:`DEFAULT_PORT`); duplicates and blanks dropped."""
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else list(value)
    out: list[str] = []
    for raw in items:
        item = str(raw).strip()
        if not item:
            continue
        if _PEER.match(item) is None:
            if ":" in item.strip("[]"):
                raise ValueError(f"peer {item!r} is not host:port")
            item = f"{item}:{DEFAULT_PORT}"
        if item not in out:
            out.append(item)
    return out


def peers_from(spec: Any = None, environ: Any = None) -> list[str]:
    """The peer list of a sampler spec (``peers``) or of ``$YETO_RELEASE_PEERS``
    (the spec wins when it names any)."""
    env = os.environ if environ is None else environ
    if isinstance(spec, dict) and spec.get("peers"):
        return parse_peers(spec["peers"])
    return parse_peers(env.get(PEERS_ENV))


def urllib_fetch(url: str, timeout_s: float) -> bytes:
    """GET ``url``; ``timeout_s`` bounds the whole transfer, not only each socket
    read (a peer that drips bytes would otherwise hold the pull forever)."""
    deadline = time.monotonic() + float(timeout_s)
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - http to a peer
            parts: list[bytes] = []
            while True:
                if time.monotonic() > deadline:
                    raise PeerError(f"{url}: not delivered within {timeout_s:g} s")
                part = response.read1(cm.CHUNK)
                if not part:
                    return b"".join(parts)
                parts.append(part)
    except urllib.error.HTTPError as error:
        raise PeerError(f"{url}: HTTP {error.code}") from None
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise PeerError(f"{url}: {type(error).__name__}: {error}") from None


def manifest_path(store: Path, release: str) -> Path:
    return Path(store) / MANIFESTS / f"{release}.json"


def save_manifest(store: Path, manifest: dict[str, Any]) -> Path:
    """Record a placed release's manifest so this node can serve it (written
    after ``place``: the server checks both exist)."""
    path = manifest_path(store, str(manifest["release"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(cm.encode(manifest))
    os.replace(tmp, path)
    return path


def servable(store: Path, release: str) -> bool:
    """The release is fully placed and its manifest recorded."""
    try:
        cm.check_release_name(release)
    except cm.ManifestError:
        return False
    return (Path(store) / cm.RELEASES / release).is_dir() and manifest_path(store, release).is_file()


def pull_from_peer(peer: str, store: Path, release: str, *, digest: str | None, fetch: Fetch = urllib_fetch,
                   timeout_s: float = 60.0, file_timeout_s: float = 3600.0) -> dict[str, Any]:
    """Stage ``release`` from one peer and place it; the metrics row. Raises
    :class:`PeerError` (nothing placed) when the peer lacks the release, its
    manifest does not match ``digest``, or a file fails its hash."""
    cm.check_release_name(release)
    base = f"http://{peer}"
    try:
        manifest = json.loads(fetch(f"{base}/{MANIFESTS}/{release}.json", timeout_s))
    except (ValueError, UnicodeDecodeError) as error:
        raise PeerError(f"{peer}: manifest of {release} is not JSON: {error}") from None
    if not isinstance(manifest, dict) or manifest.get("release") != release:
        raise PeerError(f"{peer}: manifest names {manifest.get('release') if isinstance(manifest, dict) else None!r}")
    try:
        cm.check(manifest, digest)
    except cm.ManifestError as error:
        raise PeerError(f"{peer}: {error}") from None
    expected = digest or cm.digest(manifest)
    root = Path(store) / cm.RELEASES
    if manifest.get("base") and not (root / str(manifest["base"])).is_dir():
        raise PeerError(f"{peer}: delta base {manifest['base']} is not in the local store yet")
    staged = Path(store) / f"staged-peer-{release}"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    t0 = time.time()
    try:
        for name, meta in manifest["files"].items():
            target = staged / name
            if target.resolve().parent != target.parent.resolve() or ".." in Path(name).parts:
                raise PeerError(f"{peer}: manifest file name {name!r} escapes the release")
            target.parent.mkdir(parents=True, exist_ok=True)
            data = fetch(f"{base}/{release}/{name}", file_timeout_s)
            if len(data) != int(meta["size"]):
                raise PeerError(f"{peer}: {name} has {len(data)} bytes, manifest says {meta['size']}")
            target.write_bytes(data)
        try:
            cm.place(manifest, staged, Path(store), expected_digest=expected)
        except cm.ManifestError as error:
            raise PeerError(f"{peer}: {error}") from None
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    save_manifest(Path(store), manifest)
    seconds = time.time() - t0
    return {"side": "pull", "source": f"peer:{peer}", "release": release, "bytes": manifest["bytes"],
            "download_s": round(seconds, 1), "MBps": round(manifest["bytes"] / 1e6 / max(seconds, 1e-6), 2),
            "placed_at": round(time.time(), 1)}


def pull_from_peers(peers: Sequence[str], store: Path, release: str, *, digest: str | None,
                    fetch: Fetch = urllib_fetch, log: Callable[[str], None] = lambda m: None,
                    **kw: Any) -> dict[str, Any] | None:
    """The first peer that delivers ``release`` (in order); None when none did
    (the caller falls back to the object store). Each failure is logged."""
    for peer in peers:
        try:
            return pull_from_peer(peer, store, release, digest=digest, fetch=fetch, **kw)
        except PeerError as error:
            log(f"peer {peer} did not deliver {release}: {error}")
    return None


# ---------------------------------------------------------------- file server
class _Handler(BaseHTTPRequestHandler):
    store: Path = Path(".")
    server_version = "yeto-release-peer/1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - BaseHTTPRequestHandler's name
        return  # quiet: the island's tape is the log

    def _resolve(self) -> Path | None:
        """The store file this request names, only inside a servable release or
        ``manifests/``; None (404) for anything else."""
        parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
        if len(parts) < 2 or any(p in (".", "..") or p.startswith(".") for p in parts):
            return None
        if parts[0] == MANIFESTS:
            if len(parts) != 2 or not parts[1].endswith(".json"):
                return None
            release = parts[1][:-5]
            return manifest_path(self.store, release) if servable(self.store, release) else None
        release = parts[0]
        if not servable(self.store, release):
            return None
        path = (self.store / cm.RELEASES / release).joinpath(*parts[1:])
        return path if path.is_file() else None

    def do_HEAD(self) -> None:  # noqa: N802 - http.server's name
        self._serve(head=True)

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        self._serve(head=False)

    def _serve(self, *, head: bool) -> None:
        path = self._resolve()
        if path is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if path.suffix == ".json" else "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        if head:
            return
        with open(path, "rb") as handle:
            shutil.copyfileobj(handle, self.wfile, cm.CHUNK)


class ReleaseFileServer:
    """Serves the store's placed releases to peers (module doc) on a daemon thread.

    The thread can die (an exception out of ``serve_forever``); :meth:`alive`
    tells, :attr:`error` keeps the cause, and :meth:`ensure_alive` (called by
    the pull loop every pass) logs it and serves again: on the same socket
    first, on a freshly bound one after ``rebind_after`` deaths in a row. After
    ``max_restarts`` deaths in a row (or a failed bind) it gives up and
    :attr:`healthy` is False; peers fall back to the object store."""

    def __init__(self, store: Path, *, host: str = "0.0.0.0", port: int = DEFAULT_PORT,
                 poll_interval: float = 0.5, rebind_after: int = 3, max_restarts: int = 6) -> None:
        self._handler = type("StoreHandler", (_Handler,), {"store": Path(store)})
        self._server = self._bind(host, int(port))
        self._address = (host, self.port)  # port 0: keep the port the first bind got
        self._thread: threading.Thread | None = None
        self._poll = float(poll_interval)
        self._stopped = False
        self.rebind_after, self.max_restarts = int(rebind_after), int(max_restarts)
        self.error: str | None = None
        self.restarts = 0  # every restart
        self.deaths_in_a_row = 0  # deaths with no healthy check in between
        self.healthy = True

    def _bind(self, host: str, port: int) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer((host, port), self._handler)
        server.daemon_threads = True
        return server

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def _serve(self) -> None:
        try:
            self._server.serve_forever(poll_interval=self._poll)
        except BaseException as error:  # noqa: BLE001 - recorded for ensure_alive; the thread ends here
            self.error = f"{type(error).__name__}: {error}"

    def start(self) -> "ReleaseFileServer":
        self._thread = threading.Thread(target=self._serve, name="yeto-release-peer", daemon=True)
        self._thread.start()
        return self

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def ensure_alive(self, log: Callable[[str], None] = print) -> bool:
        """True when serving (or stopped on purpose); False when the thread had died
        (restarted, rebound, or given up: see :attr:`healthy`)."""
        if self._stopped or not self.healthy:
            return self._stopped
        if self.alive():
            self.deaths_in_a_row = 0
            return True
        self.deaths_in_a_row += 1
        cause = self.error or "no error recorded"
        if self.deaths_in_a_row > self.max_restarts:
            self.healthy = False
            log(f"release peer server unhealthy: died {self.deaths_in_a_row} times in a row ({cause}); "
                "peers fall back to the object store")
            return False
        if self.deaths_in_a_row >= self.rebind_after:
            try:
                self._server.server_close()
                self._server = self._bind(*self._address)
            except OSError as error:
                self.healthy = False
                log(f"release peer server unhealthy: rebind of {self._address} failed: {error}")
                return False
            log(f"release peer server rebound after {self.deaths_in_a_row} deaths in a row ({cause})")
        self.restarts += 1
        log(f"release peer server thread died ({cause}): restart {self.restarts}")
        self.start()
        return False

    def stop(self) -> None:
        self._stopped = True
        if self.alive():
            self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)
