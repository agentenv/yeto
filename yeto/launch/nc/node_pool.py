"""The SSH side of a leased National Compute node (and of the other SSH providers).

Per node, an OpenSSH ``Host`` alias in ``<state_dir>/ssh_config`` (included from
``~/.ssh/config``): the node IP, the login user, the lease key, the host key
pinned from the ticket, and the workspace relay as ``ProxyCommand``
(:mod:`.tunnel`); every step on the node (docker on NVMe, the model pull, the
sampler container: :mod:`.ssh_sampler`) rides that alias. The sky and Daytona
providers write their aliases into the same file (:func:`edit_ssh_config`).

Nothing is installed on the node besides docker: no k3s or other cluster software
(National Compute MI355X nodes ship the provider's single-node "health k3s"; it is
left alone).
"""

from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .client import NodeLease, state_file_lock
from .tunnel import RELAY_PORT, state_dir

SSH_CONFIG = Path(os.path.expanduser("~/.ssh/config"))


def host_alias(ticket: str) -> str:
    return f"yeto-nc-{ticket}"


def key_path() -> Path:
    return state_dir() / "id_ed25519"


def ensure_key(run: Callable[..., Any] = subprocess.run) -> str:
    """The lease key (generated once, never leaves the launcher host); returns the public key."""
    key = key_path()
    if not key.exists():
        run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "yeto-nc", "-f", str(key)], check=True)
    return Path(f"{key}.pub").read_text().strip()


def proxy_command(account: int, relay_port: int = RELAY_PORT) -> str:
    env = []
    root = str(Path(__file__).resolve().parents[3])
    env.append(f"PYTHONPATH={shlex.quote(root)}")
    for name in ("NC_TOOL_DIR", "NC_NODE_BIN", "YETO_NC_STATE_DIR"):
        if os.environ.get(name):
            env.append(f"{name}={shlex.quote(os.environ[name])}")
    return (f"env {' '.join(env)} {shlex.quote(sys.executable)} -P -m yeto.launch.nc proxy "
            f"--account {account} --relay-port {relay_port} %h %p")


def ssh_host_block(lease: NodeLease, *, relay_port: int = RELAY_PORT) -> str:
    known = state_dir() / "known_hosts"
    return "\n".join([
        f"Host {host_alias(lease.ticket)}",
        f"  HostName {lease.ip}",
        f"  User {lease.user}",
        f"  IdentityFile {key_path()}",
        "  IdentitiesOnly yes",
        f"  UserKnownHostsFile {known}",
        "  StrictHostKeyChecking yes",
        "  ServerAliveInterval 15",
        "  ServerAliveCountMax 4",
        f"  ProxyCommand {proxy_command(lease.account, relay_port)}",
        "",
    ])


def _blocks(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current = None
    for line in text.splitlines(keepends=True):
        if line.startswith("Host "):
            current = line.split()[1]
            out[current] = ""
        if current:
            out[current] += line
    return out


def write_ssh_alias(lease: NodeLease, *, ssh_config: Path = SSH_CONFIG, relay_port: int = RELAY_PORT) -> str:
    """Add the node's alias and pin its host key; make ``~/.ssh/config`` include our file."""
    ours = state_dir() / "ssh_config"
    block = ssh_host_block(lease, relay_port=relay_port)
    edit_ssh_config(ours, lambda blocks: blocks.__setitem__(host_alias(lease.ticket), block))
    if lease.host_key:
        known = state_dir() / "known_hosts"
        lines = [ln for ln in (known.read_text().splitlines() if known.exists() else [])
                 if not ln.startswith(f"{lease.ip} ")]
        lines.append(f"{lease.ip} {lease.host_key.strip()}")
        known.write_text("\n".join(lines) + "\n")
        os.chmod(known, 0o600)
    include_ours(ours, ssh_config=ssh_config)
    return host_alias(lease.ticket)


def include_ours(ours: Path, *, ssh_config: Path | None = None) -> None:
    """Make ``~/.ssh/config`` include our alias file (``ssh <alias>`` resolves only then)."""
    ssh_config = SSH_CONFIG if ssh_config is None else ssh_config
    include = f"Include {ours}"
    text = ssh_config.read_text() if ssh_config.exists() else ""
    if include not in text.splitlines():
        ssh_config.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        ssh_config.write_text(f"{include}\n{text}")  # Include must precede any Host block


def remove_ssh_alias(ticket: str) -> None:
    alias = host_alias(ticket)
    edit_ssh_config(state_dir() / "ssh_config", lambda blocks: blocks.pop(alias, None), create=False)


def edit_ssh_config(ours: Path, edit: Callable[[dict[str, str]], Any], *, create: bool = True) -> None:
    """One read-modify-write of the shared alias file: every provider (NC, sky,
    Daytona) of every process on the state dir writes this one file, so the edit
    holds ``state_file_lock`` (a thread lock plus an ``flock``; not reentrant across
    the flock, so ``edit`` must not call back in) and the new text replaces the file
    atomically (a 0600 temporary of its own, synced, renamed): a concurrent ssh never
    reads a half-written config and no writer drops another's alias."""
    with state_file_lock(ours):
        if not create and not ours.exists():
            return
        blocks = _blocks(ours.read_text()) if ours.exists() else {}
        edit(blocks)
        fd, tmp = tempfile.mkstemp(prefix=f".{ours.name}.", suffix=".tmp", dir=ours.parent)
        try:
            with os.fdopen(fd, "w") as f:
                f.write("".join(blocks.values()))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, ours)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


def ssh_run(alias: str, script: str, *, run: Callable[..., Any] = subprocess.run,
            timeout: float = 1800) -> subprocess.CompletedProcess:
    """``ssh -T alias bash -s`` with the script on stdin (no tty, BatchMode)."""
    return run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=60", alias, "bash", "-s"],
               input=script, capture_output=True, text=True, timeout=timeout, check=False)
