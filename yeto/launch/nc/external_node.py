"""A node an operator brought up by hand (the National Compute console, or any other
provider) made usable by the per-node commands of this tool.

``register-node --name N --host USER@IP`` writes what those commands look a node
up by:

- a lease record under the ticket ``ext-<N>`` (account ``0``: no account of this
  tool ordered it), so ``probe``, ``ssh-pull``, ``ssh-sampler`` and
  ``agentic-setup`` take ``--ticket ext-<N>`` as they take an ordered ticket;
- an ssh alias ``yeto-nc-ext-<N>`` that dials the node directly (no workspace
  relay), with the host key pinned at registration and the operator's own ssh
  keys (or ``--identity-file``).

Nothing is written to the order ledger: the node is not this tool's order, so no
command of this tool deletes, releases or replaces it (``down``, ``delete-ticket``,
``adopt`` and the pool refuse an ``ext-`` ticket). Releasing it stays with the
operator, where it was ordered. ``unregister-node`` drops the record and the alias
and does not touch the node.
"""

from __future__ import annotations

import ipaddress
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .client import NodeLease, parse_time
from .node_pool import edit_ssh_config, host_alias, include_ours
from .tunnel import state_dir

PREFIX = "ext-"
ACCOUNT = 0
NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
USER_RE = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
HOSTNAME_RE = re.compile(r"[A-Za-z0-9]([A-Za-z0-9.-]{0,252})")


def is_external(ticket: str | None) -> bool:
    return bool(ticket) and str(ticket).startswith(PREFIX)


def ticket_of(name: str) -> str:
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"node name {name!r}: lower-case letters, digits and '-', at most 40, "
                         "starting with a letter or digit")
    return PREFIX + name


def parse_host(host: str) -> tuple[str, str]:
    """``USER@HOST`` -> (user, host)."""
    user, sep, addr = host.partition("@")
    if not sep or not USER_RE.fullmatch(user):
        raise ValueError(f"--host {host!r}: expected USER@HOST")
    try:
        ipaddress.ip_address(addr)
    except ValueError:
        if not HOSTNAME_RE.fullmatch(addr):
            raise ValueError(f"--host {host!r}: {addr!r} is not an IP address or host name") from None
    return user, addr


def known_hosts_path() -> Path:
    return state_dir() / "known_hosts"


def host_block(lease: NodeLease, *, port: int = 22, identity_file: str | None = None) -> str:
    """A direct alias: no ``ProxyCommand``; the host key pinned in our known_hosts."""
    lines = [f"Host {host_alias(lease.ticket)}", f"  HostName {lease.ip}", f"  User {lease.user}"]
    if port != 22:
        lines.append(f"  Port {port}")
    if identity_file:
        lines += [f"  IdentityFile {identity_file}", "  IdentitiesOnly yes"]
    lines += [f"  UserKnownHostsFile {known_hosts_path()}", "  StrictHostKeyChecking yes",
              "  ServerAliveInterval 15", "  ServerAliveCountMax 4", ""]
    return "\n".join(lines)


def _known_name(ip: str, port: int) -> str:
    return ip if port == 22 else f"[{ip}]:{port}"


def scan_host_keys(ip: str, port: int = 22, *, run: Callable[..., Any] | None = None,
                   timeout: float = 30) -> list[str]:
    """``ssh-keyscan``: the node's host key lines (``<name> <type> <key>``)."""
    out = (run or subprocess.run)(["ssh-keyscan", "-T", "15", "-p", str(port), ip], capture_output=True, text=True,
              timeout=timeout, check=False)
    keys = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip() and not ln.startswith("#")]
    if not keys:
        raise RuntimeError(f"ssh-keyscan {ip}:{port} returned no host key: {(out.stderr or '')[-200:]}")
    return keys


def pin_host_keys(ip: str, port: int, keys: list[str]) -> None:
    """Replace the node's lines in our known_hosts (only lines of that name)."""
    name = _known_name(ip, port)
    path = known_hosts_path()
    kept = [ln for ln in (path.read_text().splitlines() if path.exists() else [])
            if ln.split(" ", 1)[0] != name]
    path.write_text("\n".join(kept + [f"{name} {k.split(' ', 1)[1]}" for k in keys]) + "\n")
    path.chmod(0o600)


def lease_of(name: str, host: str, *, host_key: str = "", lease_until: str | None = None,
             gpus: int | None = 8) -> NodeLease:
    user, ip = parse_host(host)
    if lease_until is not None:
        try:
            ok = parse_time(lease_until) is not None
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"--lease-until {lease_until!r} is not an ISO time")
    return NodeLease(account=ACCOUNT, ticket=ticket_of(name), ip=ip, user=user, host_key=host_key,
                     granted_at=None, lease_until=lease_until, gpus=gpus, rate_ucr_per_gpu_hour=None)


def register(name: str, host: str, *, save_lease: Callable[[NodeLease], None], port: int = 22,
             identity_file: str | None = None, lease_until: str | None = None, gpus: int | None = 8,
             ssh_config: Path | None = None, run: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Pin the host key, write the alias and the lease record, then check the login."""
    run = run or subprocess.run
    lease = lease_of(name, host, lease_until=lease_until, gpus=gpus)
    keys = scan_host_keys(lease.ip, port, run=run)
    lease = NodeLease(**{**lease.to_json(), "host_key": keys[0].split(" ", 1)[1]})
    pin_host_keys(lease.ip, port, keys)
    ours = state_dir() / "ssh_config"
    block = host_block(lease, port=port, identity_file=identity_file)
    alias = host_alias(lease.ticket)
    edit_ssh_config(ours, lambda blocks: blocks.__setitem__(alias, block))
    include_ours(ours, ssh_config=ssh_config)
    save_lease(lease)
    probe = run(["ssh", "-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", alias,
                 "nproc; (rocm-smi --showproductname 2>/dev/null || nvidia-smi -L 2>/dev/null) | "
                 "grep -ciE 'MI3|GPU [0-9]+:' || true"],
                capture_output=True, text=True, timeout=60, check=False)
    return {"ticket": lease.ticket, "alias": alias, "ip": lease.ip, "user": lease.user,
            "lease_until": lease.lease_until, "login_ok": probe.returncode == 0,
            "login_detail": ((probe.stdout or "") + (probe.stderr or "")).strip()[-300:]}


def unregister(name: str, *, drop_lease: Callable[[str], bool]) -> dict[str, Any]:
    """Drop the record and the alias; the node itself is not touched."""
    ticket = ticket_of(name)
    alias = host_alias(ticket)
    edit_ssh_config(state_dir() / "ssh_config", lambda blocks: blocks.pop(alias, None), create=False)
    return {"ticket": ticket, "lease_dropped": drop_lease(ticket), "alias_dropped": alias}


def refuse(cmd: str, ticket: str) -> str:
    return (f"{cmd}: {ticket} is a node an operator registered (register-node), not an order of this tool: "
            "this tool does not release, delete or replace it. Release it where it was ordered; "
            f"`unregister-node --name {ticket[len(PREFIX):]}` drops this tool's record of it")
