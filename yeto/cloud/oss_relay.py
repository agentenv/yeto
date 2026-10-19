"""Cloud layer: the head's object API of an ``oss://`` store through a relay node
(S25 stage 1, gap 6; main-agent ruling 10-19: the head on this server reads and
writes OSS through ppu1).

The stage-1 bucket accepts only Alibaba intranet endpoints
(``PublicEndpointForbidden`` from this server). The islands on the PPU nodes
reach it directly (:mod:`yeto.cloud.oss_store`). The head instead runs
``ossutil`` on a relay node over SSH: every :class:`RelayOssObjectStore` call is
one ``ssh RELAY bash -c SCRIPT`` with one ``ossutil`` command; object bodies
travel on the SSH stdin and stdout. ``ossutil`` on the relay uses the relay's own
configuration (``~/.ossutilconfig`` with the intranet endpoint, runbook S4); no
credential passes through the head.

Switch: :data:`RELAY_ENV` names the relay's SSH target. Unset, the head talks
to OSS directly (``oss_store.head_objects``), so a public endpoint or CNAME,
once it exists, needs no code change.

Limits (first implementation):

* ``create`` is a stat followed by a copy, not an atomic conditional write;
  the head is the only writer of the keys it creates (trainer commands), and
  the sample buffer keeps one shard per sampler (``buffer_shards``), so no
  record depends on it;
* one SSH round trip per call (the head reads small objects: cuts, commands,
  index records); bulk weights never go through the head;
* the ``ossutil`` output formats parsed here (``ls -s`` lines, ``NoSuchKey`` /
  ``404`` on a missing key) follow ossutil 1.7 documentation only; not run
  against OSS.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from yeto.cloud.oss_store import SCHEME, parse_oss_uri
from yeto.rl.engine.object_store import ObjectStoreError, check_key

RELAY_ENV = "YETO_RL_STORE_OSS_RELAY"  # SSH target of the relay node; unset = direct
OSSUTIL_ENV = "YETO_RL_STORE_OSS_RELAY_OSSUTIL"  # ossutil command on the relay (default "ossutil")
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20")
_MISSING_MARKERS = ("NoSuchKey", "StatusCode=404", "404 Not Found")

# (argv, stdin bytes) -> (exit code, stdout bytes, stderr bytes)
Runner = Callable[[Sequence[str], bytes], tuple[int, bytes, bytes]]


def subprocess_runner(timeout_s: float = 600.0) -> Runner:
    def run(argv: Sequence[str], data: bytes) -> tuple[int, bytes, bytes]:
        done = subprocess.run(list(argv), input=data, capture_output=True, timeout=timeout_s)  # noqa: S603
        return done.returncode, done.stdout, done.stderr

    return run


@dataclass(frozen=True)
class Relay:
    """The relay node: its SSH target and the ``ossutil`` command there."""

    host: str
    ossutil: str = "ossutil"

    def argv(self, script: str) -> list[str]:
        """``ssh`` that runs ``script`` on the relay (ssh joins its arguments into
        one remote shell line, hence the quoting)."""
        return ["ssh", *SSH_OPTIONS, self.host, f"bash -c {shlex.quote(script)}"]


class RelayOssObjectStore:
    """:class:`~yeto.rl.engine.object_store.ObjectStore` over ``oss://BUCKET[/PREFIX]``
    through ``ossutil`` on a relay node (module doc)."""

    scheme = SCHEME
    buffer_shards = True  # as OssObjectStore: no record depends on create

    def __init__(self, uri: str, relay: Relay, *, runner: Runner | None = None) -> None:
        self.bucket, self.prefix = parse_oss_uri(uri)
        self.uri = f"{SCHEME}://{self.bucket}" + (f"/{self.prefix}" if self.prefix else "")
        self.relay = relay
        self.runner = runner or subprocess_runner()

    # ---------------------------------------------------------------- helpers
    def _url(self, key: str) -> str:
        key = check_key(key)
        return f"{SCHEME}://{self.bucket}/" + (f"{self.prefix}/{key}" if self.prefix else key)

    def _run(self, script: str, data: bytes = b"") -> tuple[int, bytes, str]:
        """(exit code, stdout, error text); ``data`` is the script's stdin."""
        rc, out, err = self.runner(self.relay.argv(script), data)
        detail = "" if rc == 0 else err.decode(errors="replace") + out.decode(errors="replace")[-2000:]
        return rc, out, detail

    def _fail(self, what: str, key: str, detail: str) -> ObjectStoreError:
        return ObjectStoreError(f"oss relay {self.relay.host}: {what} {self._url(key)} failed: {detail.strip()[-500:]}")

    @staticmethod
    def _missing(detail: str) -> bool:
        return any(marker in detail for marker in _MISSING_MARKERS)

    # ------------------------------------------------------------ ObjectStore
    def read(self, key: str) -> bytes | None:
        rc, out, detail = self._run(f"exec {self.relay.ossutil} cat {shlex.quote(self._url(key))}")
        if rc == 0:
            return out
        if self._missing(detail):
            return None
        raise self._fail("read", key, detail)

    def write(self, key: str, data: bytes) -> None:
        script = ('t=$(mktemp) && trap \'rm -f "$t"\' EXIT\n'
                  'cat > "$t"\n'
                  f'{self.relay.ossutil} cp -f "$t" {shlex.quote(self._url(key))} >/dev/null')
        rc, _, detail = self._run(script, data)
        if rc:
            raise self._fail("write", key, detail)

    def exists(self, key: str) -> bool:
        rc, _, detail = self._run(f"exec {self.relay.ossutil} stat {shlex.quote(self._url(key))} >/dev/null")
        if rc == 0:
            return True
        if self._missing(detail):
            return False
        raise self._fail("stat", key, detail)

    def create(self, key: str, data: bytes) -> bool:
        """Write ``key`` unless it exists; not atomic (module doc)."""
        if self.exists(key):
            return False
        self.write(key, data)
        return True

    def list(self, prefix: str, *, start_after: str = "") -> list[str]:
        full = f"{SCHEME}://{self.bucket}/" + (f"{self.prefix}/{prefix}" if self.prefix else prefix)
        rc, out, detail = self._run(f"exec {self.relay.ossutil} ls -s {shlex.quote(full)}")
        if rc:
            raise ObjectStoreError(f"oss relay {self.relay.host}: list {full} failed: {detail.strip()[-500:]}")
        root = f"{SCHEME}://{self.bucket}/" + (f"{self.prefix}/" if self.prefix else "")
        keys = sorted(line[len(root):] for line in out.decode().splitlines()
                      if line.startswith(root) and not line.endswith("/"))
        return [k for k in keys if k > start_after] if start_after else keys

    def delete(self, key: str) -> None:
        rc, _, detail = self._run(f"exec {self.relay.ossutil} rm {shlex.quote(self._url(key))} >/dev/null")
        if rc and not self._missing(detail):
            raise self._fail("delete", key, detail)


def relay_from_env(environ: Mapping[str, str]) -> Relay | None:
    host = environ.get(RELAY_ENV)
    return Relay(host, environ.get(OSSUTIL_ENV) or "ossutil") if host else None


def relay_shell(relay: Relay, command: str) -> str:
    """``command`` as one shell line that runs on the relay (for plan steps the
    head runs, e.g. a listing of the run directory)."""
    return shlex.join(["ssh", *SSH_OPTIONS, relay.host, command])
