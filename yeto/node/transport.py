"""How the launcher reaches a node: two primitives, ``exec`` and ``upload`` (put).

Every route that drives a node from the launcher needs only these (unified-sampler-pool D5):

* ``exec(command, timeout)``: run ``command`` with ``bash -c`` on the node and wait;
  the exit code and output (:class:`ExecResult`). A command that starts a sampler
  detaches it itself (``setsid``, ``docker run -d``), so a handle without
  background support works;
* ``upload(data, remote)``: write ``data`` to the absolute node path ``remote``
  (parents created; :func:`put` adds the mode and an atomic rename).

The handles:

* :class:`SshExec`: plain ``ssh`` (an own node, a sky VM, an NC lease's alias, the
  Daytona ssh gateway; optional chunked uploads for a gateway that cuts long sessions);
* :class:`SdkExec`: the Daytona toolbox API (``process.exec``, ``fs.upload_file``);
* :class:`StoreUploadExec`: any handle whose large uploads go through the object
  store (a presigned URL fetched on the node, sha256-checked).

Modal is the one route without them: a Modal function is *spawned* with its spec
(``yeto.launch.nc.modal_provider``); it reports through the store like every other
node (``yeto.node.records``), and nothing is executed on it from outside.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# a shell without sudo (the sandbox user is root) runs ``sudo X`` as ``X``
NO_SUDO_SHIM = 'command -v sudo >/dev/null 2>&1 || sudo() { while [ "${1#-}" != "$1" ]; do shift; done; "$@"; }; '


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class SandboxExec(Protocol):
    """The two primitives of a node handle. ``name``: for messages and events.

    ``exec``: run ``command`` with ``bash -c`` in the sandbox and wait (up to
    ``timeout`` s); the exit code and output. A command that starts the sampler
    detaches it itself (``setsid``), so a handle without background support works.
    ``upload``: write ``data`` to the absolute sandbox path ``remote`` (parents
    created; the caller sets the mode)."""

    name: str

    def exec(self, command: str, *, timeout: float = 600) -> ExecResult: ...
    def upload(self, data: bytes, remote: str) -> None: ...


@dataclass
class SshExec:
    """:class:`SandboxExec` over ``ssh``: ``target`` is the argv after ``ssh``'s options
    that names the sandbox (Daytona: ``["<token>@ssh.app.daytona.io"]``)."""

    name: str
    target: Sequence[str]
    options: Sequence[str] = ("-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30")
    run: Callable[..., Any] = subprocess.run
    chunk_bytes: int | None = None  # None: one ssh session per upload (as before)
    chunk_attempts: int = 3

    def exec(self, command: str, *, timeout: float = 600) -> ExecResult:
        out = self.run(["ssh", "-n", *self.options, *self.target, "bash -c " + shlex.quote(command)],
                       capture_output=True, text=True, timeout=timeout, check=False)
        return ExecResult(out.returncode, out.stdout or "", out.stderr or "")

    def upload(self, data: bytes, remote: str) -> None:
        if self.chunk_bytes and len(data) > self.chunk_bytes:
            self._upload_chunked(data, remote)
            return
        q = shlex.quote(remote)
        out = self.run(["ssh", *self.options, *self.target,
                        f"mkdir -p \"$(dirname {q})\" && cat > {q}.part && mv -f {q}.part {q}"],
                       input=data, capture_output=True, timeout=1800, check=False)
        if out.returncode != 0:
            raise RuntimeError(f"upload to {self.name}:{remote} failed: {_err(out)}")

    def _upload_chunked(self, data: bytes, remote: str) -> None:
        """``data`` in ``chunk_bytes`` pieces, one ssh session each (retried alone), then
        joined and checked by size: a gateway that cuts long sessions (Daytona's ssh
        gateway closed every 13 MB stream) costs one piece, not the whole file."""
        q = shlex.quote(remote)
        n = -(-len(data) // self.chunk_bytes)
        for i in range(n):
            piece = data[i * self.chunk_bytes:(i + 1) * self.chunk_bytes]
            for attempt in range(self.chunk_attempts):
                out = self.run(["ssh", *self.options, *self.target,
                                f"mkdir -p \"$(dirname {q})\" && cat > {q}.chunk.{i}"],
                               input=piece, capture_output=True, timeout=600, check=False)
                if out.returncode == 0:
                    break
            else:
                raise RuntimeError(f"upload to {self.name}:{remote} failed at chunk {i + 1} of {n}: {_err(out)}")
        parts = " ".join(f"{q}.chunk.{i}" for i in range(n))
        out = self.run(["ssh", "-n", *self.options, *self.target,
                        f"cat {parts} > {q}.part && rm -f {parts} && test $(wc -c < {q}.part) -eq {len(data)} "
                        f"&& mv -f {q}.part {q}"], capture_output=True, timeout=600, check=False)
        if out.returncode != 0:
            raise RuntimeError(f"upload to {self.name}:{remote} failed joining {n} chunks: {_err(out)}")


@dataclass
class SdkExec:
    """:class:`SandboxExec` over the Daytona SDK (toolbox API over HTTPS), no ssh:
    ``process.exec`` runs the command, ``fs.upload_file`` writes files (credentials
    included: they never leave the API). ``sandbox()`` returns the SDK sandbox (looked
    up on each call: a token or a sandbox object may go stale). Daytona's ssh gateway
    closed sessions on fresh GPU sandboxes again and again; the toolbox API is the
    provider's own control path."""

    name: str
    sandbox: Callable[[], Any]
    attempts: int = 3

    def _retry(self, fn: Callable[[], Any]) -> Any:
        for attempt in range(self.attempts):
            try:
                return fn()
            except Exception:  # noqa: BLE001 - a dropped API call: tried again, the last error raised
                if attempt + 1 == self.attempts:
                    raise

    def exec(self, command: str, *, timeout: float = 600) -> ExecResult:
        try:
            out = self._retry(lambda: self.sandbox().process.exec("bash -c " + shlex.quote(command),
                                                                  timeout=max(1, int(timeout))))
        except Exception as error:  # noqa: BLE001 - reads like an unreachable node
            return ExecResult(255, "", f"daytona exec: {error!r}"[:500])
        text = str(getattr(out, "result", "") or "")
        code = int(getattr(out, "exit_code", 1))
        return ExecResult(code, text, "" if code == 0 else text[-2000:])

    def upload(self, data: bytes, remote: str) -> None:
        q = shlex.quote(remote)
        made = self.exec(f"mkdir -p \"$(dirname {q})\"", timeout=60)
        if made.returncode != 0:
            raise RuntimeError(f"upload to {self.name}:{remote} failed: {made.stderr[-300:]}")
        try:
            self._retry(lambda: self.sandbox().fs.upload_file(data, remote))
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"upload to {self.name}:{remote} failed: {error!r}"[:400]) from error


@dataclass
class StoreUploadExec:
    """A :class:`SandboxExec` whose large uploads go through the object store: the launcher
    puts the bytes at ``<prefix>/_upload/<sha256>`` and the sandbox fetches a presigned URL
    with ``curl`` (checked by sha256). Daytona's ssh gateway closes long sessions (a 13 MB
    ``cat`` over ssh was cut after about 45 s, every time); small files still go over ssh,
    retried. ``env_file``: the pool's store env file (``AWS_*``, ``R2_BUCKET``)."""

    inner: SandboxExec
    env_file: str
    prefix: str
    threshold: int = 1 << 20
    attempts: int = 3
    client: Any = None

    @property
    def name(self) -> str:
        return self.inner.name

    def exec(self, command: str, *, timeout: float = 600) -> ExecResult:
        return self.inner.exec(command, timeout=timeout)

    def _store(self) -> tuple[Any, str]:
        env = dict(line.split("=", 1) for line in Path(self.env_file).expanduser().read_text().splitlines()
                   if "=" in line and not line.lstrip().startswith("#"))
        env = {k.strip().removeprefix("export "): v.strip().strip("'\"") for k, v in env.items()}
        if self.client is None:
            import boto3  # noqa: PLC0415

            self.client = boto3.client("s3", endpoint_url=env.get("AWS_ENDPOINT_URL"),
                                       aws_access_key_id=env.get("AWS_ACCESS_KEY_ID"),
                                       aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY"),
                                       region_name=env.get("AWS_REGION") or "auto")
        return self.client, env["R2_BUCKET"]

    def upload(self, data: bytes, remote: str) -> None:
        if len(data) <= self.threshold:
            for attempt in range(self.attempts):
                try:
                    return self.inner.upload(data, remote)
                except RuntimeError:
                    if attempt + 1 == self.attempts:
                        raise
        import hashlib  # noqa: PLC0415

        digest = hashlib.sha256(data).hexdigest()
        client, bucket = self._store()
        key = f"{self.prefix.strip('/')}/_upload/{digest}"
        client.put_object(Bucket=bucket, Key=key, Body=data)
        url = client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600)
        q = shlex.quote(remote)
        cmd = (f"mkdir -p \"$(dirname {q})\" && curl -fsS --retry 5 --retry-connrefused -o {q}.part {shlex.quote(url)} "
               f"&& echo '{digest}  {remote}.part' | sha256sum -c --quiet - && mv -f {q}.part {q}")
        out = self.inner.exec(cmd, timeout=1800)
        if out.returncode != 0:
            raise RuntimeError(f"upload to {self.name}:{remote} through the store failed: {out.stderr.strip()[-300:]}")


def _err(out: Any) -> str:
    err = out.stderr.decode(errors="replace") if isinstance(out.stderr, bytes) else str(out.stderr or "")
    return err.strip()[-300:]


Transport = SandboxExec  # the protocol's neutral name


def sh(box: SandboxExec, command: str, *, timeout: float = 600) -> ExecResult:
    """``exec`` with the no-sudo shim (a root shell without ``sudo`` runs ``sudo X`` as ``X``)."""
    return box.exec(NO_SUDO_SHIM + command, timeout=timeout)


def put(box: SandboxExec, data: bytes | str, remote: str, *, mode: str = "600") -> None:
    """Upload and set the mode (written to a temporary name first: never half a file)."""
    raw = data.encode() if isinstance(data, str) else data
    box.upload(raw, remote + ".up")
    q = shlex.quote(remote)
    out = sh(box, f"chmod {mode} {q}.up && mv -f {q}.up {q}", timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"{box.name}: placing {remote}: {out.stderr.strip()[-300:]}")


# -- a checked copy (the prompts file of every route) --------------------------------

def size_sha256_command(remote: str) -> str:
    """Node shell printing ``YETO_PROMPTS <bytes> <sha256>`` of ``remote``."""
    q = shlex.quote(remote)
    return f"echo YETO_PROMPTS $(wc -c < {q}) $(sha256sum {q} | cut -d' ' -f1)"


def check_copy(stdout: str, data: bytes, *, node: str, remote: str, local: Any, where: str = "node",
               after: str = "the copy") -> dict[str, Any]:
    """The output of :func:`size_sha256_command` against the local bytes; the copy's
    record, or RuntimeError naming both sides."""
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256(data).hexdigest()
    have = next((ln.split()[1:] for ln in (stdout or "").splitlines() if ln.startswith("YETO_PROMPTS ")), [])
    if have != [str(len(data)), digest]:
        raise RuntimeError(f"{node}: prompts file {remote} does not match {local} after {after} "
                           f"({where} size/sha256 {' '.join(have) or 'unreadable'}, local {len(data)} {digest[:12]})")
    return {"bytes": len(data), "sha256": digest}
