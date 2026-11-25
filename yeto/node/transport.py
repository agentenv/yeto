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

How ``ssh`` gets to a node is its reach (:func:`reach_options`, the profile's
``reach``): ``direct``; ``jump`` (``ProxyJump`` through ``via``); ``relay``
(``ProxyCommand`` through the WebSocket relay of :mod:`yeto.launch.nc.tunnel`:
``via: nc:<account>`` is a National Compute workspace relay, exactly the
``ProxyCommand`` of a lease's alias; ``via: wss://...`` a ``relay.mjs`` the site
runs itself, :mod:`yeto.node.relay_proxy`). Code upload, start and stop then all go
through it: the node opens no other inbound port.

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


# -- reach: the ssh options of a node's route ---------------------------------------

RELAY_URL_SCHEMES = ("ws://", "wss://", "http://", "https://")
_RELAY_URL_FORBIDDEN = frozenset(" \t\n'\"`$;&|<>\\")


class ReachError(ValueError):
    """A reach ``ssh`` cannot dial (unknown mode, a relay ``via`` that is neither
    ``nc:<account>`` nor a relay URL, an ``outbound`` / ``function`` node)."""


def nc_relay_identity_options() -> list[str]:
    """What a node behind an NC workspace relay is dialled with besides the relay: the host
    keys yeto pinned for NC nodes (``<nc state>/known_hosts``, after the user's own file) and
    yeto's NC key as one more identity, as an NC lease's alias is (``node_pool.ssh_host_block``)."""
    from yeto.launch.nc.node_pool import key_path  # noqa: PLC0415
    from yeto.launch.nc.tunnel import state_dir  # noqa: PLC0415

    return ["-o", f"UserKnownHostsFile=~/.ssh/known_hosts {state_dir() / 'known_hosts'}",
            "-o", f"IdentityFile={key_path()}"]


def relay_proxy_command(via: str) -> str:
    """The ``ProxyCommand`` that dials ``%h:%p`` through the relay ``via``.

    ``nc:<account>`` or ``nc:<account>:<relay port>``: the National Compute workspace
    relay of that account (:func:`yeto.launch.nc.node_pool.proxy_command`, the very
    line of a lease's ssh alias). A ``ws(s)://`` / ``http(s)://`` URL: a ``relay.mjs``
    the site runs behind an authenticating gateway (:mod:`yeto.node.relay_proxy`: its
    bearer token from the environment, never the argv; plaintext and credentials in
    the URL refused; checked here so a dry-run reports it before any ssh)."""
    import sys  # noqa: PLC0415

    text = str(via or "").strip()
    if text.startswith("nc:"):
        parts = text[3:].split(":")
        if not 1 <= len(parts) <= 2 or not all(p.isdigit() for p in parts):
            raise ReachError(f"relay via {via!r}: a National Compute relay is nc:<account> or nc:<account>:<port>")
        from yeto.launch.nc.node_pool import proxy_command  # noqa: PLC0415
        from yeto.launch.nc.tunnel import RELAY_PORT  # noqa: PLC0415

        return proxy_command(int(parts[0]), int(parts[1]) if len(parts) == 2 else RELAY_PORT)
    if text.startswith(RELAY_URL_SCHEMES) and len(text) > text.index("//") + 2 \
            and not set(text) & _RELAY_URL_FORBIDDEN:
        from yeto.launch.nc.tunnel import TunnelError  # noqa: PLC0415
        from yeto.node import relay_proxy  # noqa: PLC0415

        why = relay_proxy.url_problem(text)
        if why:
            raise ReachError(why)
        try:
            relay_proxy.token()  # present and readable (its value never enters the ProxyCommand)
        except TunnelError as error:
            raise ReachError(str(error)) from None
        root = str(Path(__file__).resolve().parents[2])  # the tree holding yeto/
        url = shlex.quote(text).replace("%", "%%")  # ssh expands %-tokens in a ProxyCommand: %% is a literal %
        return (f"env PYTHONPATH={shlex.quote(root)} {shlex.quote(sys.executable)} -P -m yeto.node.relay_proxy "
                f"--url {url} %h %p")
    raise ReachError(f"relay via {via!r} must be nc:<account>[:<port>] (a National Compute workspace relay) or a "
                     f"relay URL ({', '.join(RELAY_URL_SCHEMES)}) without blanks or shell characters")


def reach_options(reach: Any) -> list[str]:
    """``ssh`` options that dial a node through ``reach`` (a
    :class:`yeto.launch.node_inventory.Reach`; None: direct)."""
    mode = getattr(reach, "mode", None) or "direct"
    via = getattr(reach, "via", None)
    if mode == "direct":
        return []
    if mode == "jump":
        if not via:
            raise ReachError("reach jump needs via (the jump host)")
        return ["-o", f"ProxyJump={via}"]
    if mode == "relay":
        # the relay host must not stand in for the node: its key is pinned (known_hosts), never learned here
        extra = nc_relay_identity_options() if str(via or "").strip().startswith("nc:") else []
        return ["-o", "StrictHostKeyChecking=yes", *extra, "-o", f"ProxyCommand={relay_proxy_command(via)}"]
    raise ReachError(f"reach {mode!r}: the control host cannot open an ssh connection to such a node "
                     "(outbound: it only dials the store; function: the platform's calls)")


def node_ssh_options(node: Any) -> list[str]:
    """The ``ssh`` options of ``node``'s route: its profile's reach when it has one,
    else the inventory's ``proxy_jump`` (``Node.ssh_options``)."""
    reach = getattr(node, "reach", None)
    if reach is not None:
        if reach.mode == "relay" and getattr(node, "proxy_jump", None):
            raise ReachError(f"node {node.name}: reach relay and proxy_jump {node.proxy_jump} name two routes; "
                             "keep one")
        return reach_options(reach)
    return list(node.ssh_options())


def ssh_exec(node: Any, *, run: Callable[..., Any] = subprocess.run, **kw: Any) -> SshExec:
    """An :class:`SshExec` of ``node``: its ``host`` through its reach."""
    return SshExec(name=str(node.name), target=[*node_ssh_options(node), str(node.host)], run=run, **kw)


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
