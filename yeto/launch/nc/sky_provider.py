"""SkyPilot lease provider (skeleton): sampler nodes on clouds sky can launch
(AWS, GCP, Kubernetes, ...), behind the same :class:`~.provider.LeaseProvider`
contract as National Compute. The pool's policy (eviction handling, fencing,
events, budget, breaker) and the SSH + docker node route are unchanged; only the
supply differs:

* ``order(slot)``: ``sky.launch`` a one-node cluster ``<prefix>-<run>-<n>``
  (spot by default) running no job, detached; the cluster name is the ticket;
* ``status``: ``sky.status`` of that cluster: ``UP`` -> active, ``INIT`` ->
  waiting, ``STOPPED`` / gone -> ended (``preempted``: sky clusters end that
  way when the cloud reclaims spot capacity; there is no lease end, so
  ``lease_until`` is None and only eviction-driven replacement applies);
* ``release``: ``sky.down`` (only clusters this provider launched for this run:
  the name prefix *and* the local record);
* ``node``: the head IP and ssh user from the cluster handle;
* ``attach``: an ssh alias to the head (sky's key, no relay), then the
  ``host_setup`` commands of the config.

"Accounts" are slots ``1..max_nodes``: one cluster per slot at a time.

Status: skeleton. CPU tests drive it with a fake ``sky`` module; it has not
launched a real cluster. Pool config::

    "provider": {"factory": "yeto.launch.nc.sky_provider:from_config",
                 "resources": {"cloud": "aws", "accelerators": "H100:8", "use_spot": true},
                 "cluster_prefix": "yeto-smp", "ssh_key": "~/.sky/clients/<id>/ssh/sky-key",
                 "host_setup": ["sudo docker info"]}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .client import NodeLease
from .elastic import LeaseStatus
from .tunnel import state_dir

UP, INIT, STOPPED = "UP", "INIT", "STOPPED"


def _status_name(value: Any) -> str:
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


class SkyProvider:
    def __init__(self, run_id: str, *, resources: dict[str, Any], slots: list[int], cluster_prefix: str = "yeto-smp",
                 ssh_key: str | None = None, host_setup: list[str] | None = None, sky: Any = None,
                 record: Path | None = None):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", cluster_prefix):
            raise ValueError("cluster_prefix: lower-case letters, digits and '-'")
        self.run_id, self.resources, self.slots = run_id, dict(resources), list(slots)
        self.prefix = f"{cluster_prefix}-{re.sub(r'[^a-z0-9-]', '-', run_id.lower())}"
        self.ssh_key = os.path.expanduser(ssh_key) if ssh_key else None
        self.host_setup = list(host_setup or ())
        self._sky = sky
        self.record = record or state_dir() / f"sky-clusters-{self.prefix}.json"

    # -- sky
    @property
    def sky(self):
        if self._sky is None:
            import sky  # noqa: PLC0415

            self._sky = sky
        return self._sky

    def _rows(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.record.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, rows: dict[str, dict[str, Any]]) -> None:
        self.record.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.record.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1))
        tmp.replace(self.record)

    def _ours(self, ticket: str) -> bool:
        return ticket.startswith(self.prefix + "-") and ticket in self._rows()

    def _record(self, ticket: str) -> dict[str, Any] | None:
        records = self.sky.get(self.sky.status([ticket]))
        return records[0] if records else None

    # -- LeaseProvider
    def order(self, account: int) -> str:
        rows = self._rows()
        n = 1 + max([int(r.get("n", 0)) for r in rows.values()] or [0])
        name = f"{self.prefix}-{n}"
        rows[name] = {"slot": account, "n": n, "open": True}
        self._save(rows)  # recorded before the launch: a crash mid-launch still leaves it ours to down
        task = self.sky.Task(name=name, run=None)
        task.set_resources(self.sky.Resources(**self.resources))
        self.sky.launch(task, cluster_name=name, detach_run=True, down=True)
        return name

    def status(self, account: int, ticket: str) -> LeaseStatus:
        record = self._record(ticket)  # raises on a failed call: a control-plane blip to the pool
        if record is None:
            return LeaseStatus("ended", end_reason="preempted")
        state = _status_name(record.get("status"))
        if state == UP:
            return LeaseStatus("active")
        if state == INIT:
            return LeaseStatus("waiting")
        return LeaseStatus("ended", end_reason="preempted" if state == STOPPED else state.lower())

    def release(self, account: int, ticket: str) -> dict[str, Any]:
        if not self._ours(ticket):
            raise PermissionError(f"cluster {ticket} was not launched by this provider for run {self.run_id}")
        if self._record(ticket) is not None:
            self.sky.get(self.sky.down(ticket))
        if self._record(ticket) is not None:
            raise RuntimeError(f"cluster {ticket}: down sent but it still exists")
        rows = self._rows()
        rows[ticket]["open"] = False
        self._save(rows)
        return {"ticket": ticket, "state": "ended"}

    def node(self, account: int, ticket: str) -> NodeLease:
        record = self._record(ticket)
        handle = (record or {}).get("handle")
        ip = getattr(handle, "head_ip", None)
        if not ip:
            raise RuntimeError(f"cluster {ticket} has no head IP yet")
        user = getattr(handle, "ssh_user", None) or "ubuntu"
        gpus = None
        return NodeLease(account, ticket, ip, user, "", None, None, gpus, None)

    def balance(self) -> int | None:
        return None  # clouds bill outside; the run's spend cap (Budget) is the guard

    def account_free(self, account: int) -> bool:
        return not any(r.get("open") and r.get("slot") == account for r in self._rows().values())

    def run_tickets(self) -> list[tuple[int, str]]:
        return [(int(r["slot"]), name) for name, r in self._rows().items() if r.get("open")]

    # -- node route hook
    def attach(self, lease: NodeLease, *, ssh_run) -> str:
        """An ssh alias to the head (direct, sky's key), then ``host_setup``."""
        from . import node_pool  # noqa: PLC0415

        alias = node_pool.host_alias(lease.ticket)
        ours = state_dir() / "ssh_config"
        blocks = node_pool._blocks(ours.read_text()) if ours.exists() else {}
        lines = [f"Host {alias}", f"  HostName {lease.ip}", f"  User {lease.user}",
                 "  StrictHostKeyChecking accept-new", f"  UserKnownHostsFile {state_dir() / 'known_hosts'}",
                 "  ServerAliveInterval 15", "  ServerAliveCountMax 4"]
        if self.ssh_key:
            lines += [f"  IdentityFile {self.ssh_key}", "  IdentitiesOnly yes"]
        blocks[alias] = "\n".join(lines) + "\n"
        ours.write_text("".join(blocks.values()))
        os.chmod(ours, 0o600)
        for command in self.host_setup:
            out = ssh_run(alias, command)
            if out.returncode != 0:
                raise RuntimeError(f"host setup {command[:40]!r} on {lease.ticket}: exit {out.returncode}")
        return alias


def from_config(cfg: dict[str, Any], run_id: str, *, sky: Any = None) -> SkyProvider:
    section = dict(cfg.get("provider") or {})
    section.pop("factory", None)
    unknown = set(section) - {"resources", "cluster_prefix", "ssh_key", "host_setup"}
    if unknown:
        raise ValueError(f"sky provider: unknown keys {sorted(unknown)}")
    if "resources" not in section:
        raise ValueError("sky provider: resources is required")
    return SkyProvider(run_id, slots=[int(a) for a in cfg["accounts"]], sky=sky, **section)
