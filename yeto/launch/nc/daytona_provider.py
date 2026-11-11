"""Daytona lease provider: sampler nodes as Daytona GPU sandboxes (AMD MI355X,
spot), behind the same :class:`~.provider.LeaseProvider` contract as National
Compute. The pool policy (eviction handling, fencing, claim release, refill,
budget, breaker, patrol) is unchanged; only the supply differs.

Mapping
-------

* ``order(slot)``: ``Daytona.create`` of one sandbox named ``<prefix>-<run>-<n>``
  (the name is the ticket; ``Daytona.get`` takes a name or an id) with labels
  ``yeto-run=<run>``, ``yeto-slot=<slot>``, ``yeto-owner=yeto``; GPU type
  ``MI355X``, ``gpus`` GPUs (1-8), ``spot`` (default true), ``auto_stop_interval=0``
  (auto-stop off: a stopped GPU sandbox is deleted), ``auto_delete_interval=0``,
  ``ttl_minutes`` (default 0 = off). The order is recorded locally *before*
  the create call, and a create that raises is checked by name: a sandbox that
  exists anyway is kept as the ticket (it is billed), otherwise the error goes to
  the pool (a provisioning failure for its breaker).
* ``status``: ``Daytona.get(name)``. ``started`` -> active; creating / starting /
  pending_build / pulling_snapshot / restoring / resizing / unknown -> waiting;
  ``destroyed`` / ``destroying`` with ``spot_evicted_at`` -> ended ``preempted``
  (an eviction: fence, release claims, refill); with ``queue_timed_out_at`` ->
  ended ``queue_timeout``; otherwise -> ended ``released`` (deleted by us or a
  person); ``stopped`` / ``stopping`` / ``archived`` / ``paused`` -> ended
  ``preempted`` (a stopped GPU sandbox is deleted by Daytona, the node is gone);
  ``error`` / ``build_failed`` -> ended ``error``. A not-found sandbox reads ended:
  ``released`` when this provider deleted it, else ``preempted``. Any other error
  is raised (the pool reads it as a control-plane blip, never an end). There is
  no lease end (``lease_until`` None) and no preemption notice.
* ``release``: only sandboxes of this run (name prefix *and* the local record or
  the ``yeto-run`` label): ``delete(wait=True)``, then confirmed destroyed / gone.
* ``run_tickets``: open records of this run **plus** every non-destroyed sandbox
  listed with label ``yeto-run=<run>`` (a sandbox whose create the process did not
  live to record is still found: the pool's patrol adopts it, and a stopping
  pool or ``release_orphans`` deletes it). :meth:`sweep` deletes all of them.
* ``node`` / ``attach``: SSH through the Daytona gateway
  (``<token>@ssh.app.daytona.io``, port 22), the token from
  ``sandbox.create_ssh_access(expires_in_minutes)``. ``attach`` writes an ssh
  alias (``node_pool.host_alias(ticket)``) and runs ``host_setup``; the token is
  renewed (alias rewritten) by ``status`` once less than a third of it is left.
  The token is kept only in the 0600 ssh config, never in the lease file.

Verified against the ``daytona`` Python SDK 0.231.1 (signatures and source, no
API call) and the docs (gpu-sandboxes, ssh-access) on 2026-11-11:
``Daytona(DaytonaConfig(api_key, api_url, target))`` (env ``DAYTONA_API_KEY``,
``DAYTONA_API_URL``, ``DAYTONA_TARGET``); ``CreateSandboxFromImageParams(image,
resources=Resources(cpu, memory, disk, gpu, gpu_type=[GpuType.MI355X]), labels,
env_vars, spot, auto_stop_interval, auto_delete_interval, ttl_minutes,
queue_timeout)``; ``CreateSandboxFromSnapshotParams(snapshot=...)`` takes no
resources (the snapshot's resources apply: the snapshot itself must be built
with MI355X x N); ``Daytona.create(params, timeout=)`` blocks until started;
``Daytona.get``, ``Daytona.delete(sandbox, timeout, wait)``,
``Daytona.list(ListSandboxesQuery(labels=, states=))``; ``SandboxState`` values;
``Sandbox.spot_evicted_at`` / ``queue_timed_out_at`` / ``gpu`` / ``labels``;
``DaytonaNotFoundError``; ``Sandbox.create_ssh_access(expires_in_minutes)`` ->
``SshAccessDto(token, expires_at, ssh_command)``. Docs: spot eviction runs
Started -> Deleting -> Deleted with ``spotEvictedAt`` set, without notice, the
evicted sandbox stays readable for 24 h; a spot create with no capacity fails at
once; spot needs Tier 2+; auto-stop defaults to 15 min, 0 disables it; at most 8
GPUs; ssh tokens default to 60 min.

Not verified (no API key; check on the first smoke run): the exact state string
an evicted sandbox reads (``destroying`` vs ``destroyed``), whether ``get`` of a
deleted non-spot sandbox raises not-found or returns ``destroyed``, the maximum
``expires_in_minutes`` of an ssh token, whether ``list`` returns destroyed
sandboxes, the outbound network limits of the account tier.

Pool config::

    "provider": {"kind": "daytona", "image": "rocm/vllm:<tag>", "gpus": 8,
                 "gpu_type": "MI355X", "spot": true, "cpu": 128, "memory_gib": 1536,
                 "disk_gib": 4096, "name_prefix": "yeto-smp"}

``accounts`` are slots ``1..max_nodes`` (one sandbox per slot at a time).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ..pool.provider import ProviderCaps
from .client import NodeLease
from .elastic import LeaseStatus
from .tunnel import state_dir

SSH_HOST = "ssh.app.daytona.io"
GPU_TYPES = frozenset({"MI355X", "H100", "H200", "B200", "B300", "L40S", "RTX-PRO-6000", "RTX-4090", "RTX-5090"})
MAX_GPUS = 8
LABEL_RUN, LABEL_SLOT, LABEL_OWNER = "yeto-run", "yeto-slot", "yeto-owner"

ACTIVE = frozenset({"started"})
WAITING = frozenset({"creating", "starting", "pending_build", "building_snapshot", "pulling_snapshot",
                     "restoring", "resizing", "unknown", "snapshotting", "forking", "resuming"})
GONE = frozenset({"destroyed", "destroying"})
STOPPED = frozenset({"stopped", "stopping", "archived", "archiving", "paused", "pausing"})
FAILED = frozenset({"error", "build_failed"})

CONFIG_KEYS = frozenset({"kind", "image", "snapshot", "gpus", "gpu_type", "spot", "cpu", "memory_gib", "disk_gib",
                         "name_prefix", "env", "ttl_minutes", "queue_timeout_min", "create_timeout_s",
                         "delete_timeout_s", "ssh_token_minutes", "host_setup", "target", "api_url_env",
                         "api_key_env", "price_usd_h"})
LEDGER_KIND = "daytona_sandbox"


def _record_ledger(fn: str, *args: Any, **kw: Any) -> None:
    from yeto.cost.ledger import record_safely  # noqa: PLC0415

    record_safely(fn, *args, **kw)


def _state(value: Any) -> str:
    return str(getattr(value, "value", value) or "unknown").rsplit(".", 1)[-1].lower()


def _not_found(error: BaseException) -> bool:
    return type(error).__name__ in ("DaytonaNotFoundError", "NotFoundException") or \
        getattr(error, "status_code", None) == 404


class DaytonaProvider:
    kind = "daytona"
    caps = ProviderCaps(lease_bounded=False, notice_s=0.0, currency="usd")

    def __init__(self, run_id: str, *, slots: list[int], image: str | None = None, snapshot: str | None = None,
                 gpus: int = 8, gpu_type: str = "MI355X", spot: bool = True, cpu: int | None = None,
                 memory_gib: int | None = None, disk_gib: int | None = None, name_prefix: str = "yeto-smp",
                 env: dict[str, str] | None = None, ttl_minutes: int = 0, queue_timeout_min: int | None = None,
                 create_timeout_s: float = 900, delete_timeout_s: float = 300, ssh_token_minutes: int = 1440,
                 host_setup: list[str] | None = None, target: str | None = None,
                 api_url_env: str = "DAYTONA_API_URL", api_key_env: str = "DAYTONA_API_KEY",
                 price_usd_h: float | None = None, sdk: Any = None, client: Any = None, record: Path | None = None, clock=time.time):
        validate(image=image, snapshot=snapshot, gpus=gpus, gpu_type=gpu_type, spot=spot, name_prefix=name_prefix,
                 ttl_minutes=ttl_minutes, ssh_token_minutes=ssh_token_minutes)
        self.run_id, self.slots = run_id, list(slots)
        self.prefix = f"{name_prefix}-{re.sub(r'[^a-z0-9-]', '-', run_id.lower())}"
        self.image, self.snapshot, self.gpus, self.gpu_type, self.spot = image, snapshot, int(gpus), gpu_type, spot
        self.cpu, self.memory_gib, self.disk_gib = cpu, memory_gib, disk_gib
        self.env, self.ttl_minutes, self.queue_timeout_min = dict(env or {}), int(ttl_minutes), queue_timeout_min
        self.create_timeout_s, self.delete_timeout_s = float(create_timeout_s), float(delete_timeout_s)
        self.ssh_token_minutes = int(ssh_token_minutes)
        self.host_setup = list(host_setup or ())
        self.target, self.api_url_env, self.api_key_env = target, api_url_env, api_key_env
        self._sdk, self._client, self.clock = sdk, client, clock
        self.price_usd_h = None if price_usd_h is None else float(price_usd_h)
        self.record = record or state_dir() / f"daytona-sandboxes-{self.prefix}.json"

    @property
    def sampler_route(self) -> Any:
        """Daytona sandboxes have no docker: the sampler runs as a process
        (:class:`yeto.launch.daytona_sampler.DaytonaSamplerRoute`)."""
        from ..daytona_sampler import DaytonaSamplerRoute, k31_disk_plan  # noqa: PLC0415

        return DaytonaSamplerRoute(gpus=self.gpus, plan=k31_disk_plan())

    # -- sdk
    @property
    def sdk(self):
        if self._sdk is None:
            import daytona  # noqa: PLC0415

            self._sdk = daytona
        return self._sdk

    @property
    def client(self):
        if self._client is None:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"daytona provider: {self.api_key_env} is not set")
            self._client = self.sdk.Daytona(self.sdk.DaytonaConfig(
                api_key=key, api_url=os.environ.get(self.api_url_env) or None, target=self.target))
        return self._client

    # -- local record
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

    def _get(self, ticket: str) -> Any | None:
        """The sandbox, None when Daytona has no such sandbox; other errors raise."""
        try:
            return self.client.get(ticket)
        except Exception as error:
            if _not_found(error):
                return None
            raise

    def _labelled(self) -> list[Any]:
        sdk = self.sdk
        rows = self.client.list(sdk.ListSandboxesQuery(labels={LABEL_RUN: self.run_id}))
        return [s for s in rows if _state(getattr(s, "state", None)) not in GONE
                and str(getattr(s, "name", "")).startswith(self.prefix + "-")]

    def _ours(self, ticket: str, sandbox: Any | None = None) -> bool:
        if not ticket.startswith(self.prefix + "-"):
            return False
        if ticket in self._rows():
            return True
        labels = dict(getattr(sandbox, "labels", None) or {}) if sandbox is not None else {}
        return labels.get(LABEL_RUN) == self.run_id

    def params(self, name: str, slot: int) -> Any:
        """The create parameters of one sandbox (auto-stop off, auto-delete on stop)."""
        sdk = self.sdk
        common = {"name": name, "labels": {LABEL_RUN: self.run_id, LABEL_SLOT: str(slot), LABEL_OWNER: "yeto"},
                  "env_vars": dict(self.env), "spot": self.spot, "auto_stop_interval": 0,
                  "auto_delete_interval": 0, "ttl_minutes": self.ttl_minutes, "public": False}
        if self.queue_timeout_min:
            common["queue_timeout"] = int(self.queue_timeout_min)
        if self.snapshot:
            return sdk.CreateSandboxFromSnapshotParams(snapshot=self.snapshot, **common)
        resources = sdk.Resources(cpu=self.cpu, memory=self.memory_gib, disk=self.disk_gib, gpu=self.gpus,
                                  gpu_type=[sdk.GpuType(self.gpu_type)])
        return sdk.CreateSandboxFromImageParams(image=self.image, resources=resources, **common)

    # -- LeaseProvider
    def order(self, account: int) -> str:
        client = self.client  # a missing key fails here, before anything is recorded
        rows = self._rows()
        n = 1 + max([int(r.get("n", 0)) for r in rows.values()] or [0])
        name = f"{self.prefix}-{n}"
        rows[name] = {"slot": account, "n": n, "open": True, "ordered_at": self.clock()}
        self._save(rows)  # recorded before the create: a crash mid-create still leaves it ours to delete
        attrs = self._ledger_attrs(account)
        _record_ledger("intent", LEDGER_KIND, name, **attrs)
        try:
            client.create(self.params(name, account), timeout=self.create_timeout_s)
        except Exception:
            try:
                exists = self._get(name) is not None
            except Exception:  # noqa: BLE001 - unknown: keep the record open, the patrol/status decide
                exists = True
            if exists:
                _record_ledger("created", LEDGER_KIND, name, name, **attrs)
                return name  # created (billed) but not started in time: the pool watches it
            rows = self._rows()
            rows[name]["open"] = False
            rows[name]["create_failed"] = True
            self._save(rows)
            _record_ledger("released", LEDGER_KIND, None, name, never_created=True)
            raise
        _record_ledger("created", LEDGER_KIND, name, name, **attrs)
        return name

    def _ledger_attrs(self, slot: int) -> dict[str, Any]:
        """What ``yeto cleanup`` needs to delete the sandbox and price it later."""
        return {"run_id": self.run_id, "prefix": self.prefix, "slot": slot, "gpus": self.gpus,
                "gpu_type": self.gpu_type, "usd_per_hour": self.price_usd_h,
                "api_url_env": self.api_url_env, "api_key_env": self.api_key_env}

    def hourly_cost(self, ticket: str | None = None) -> float | None:  # noqa: ARG002
        """USD per hour of one sandbox (``price_usd_h``), None when not configured."""
        return self.price_usd_h

    def status(self, account: int, ticket: str) -> LeaseStatus:
        sandbox = self._get(ticket)  # raises on a failed call: a control-plane blip to the pool
        if sandbox is None:
            released = self._rows().get(ticket, {}).get("released")
            return LeaseStatus("ended", end_reason="released" if released else "preempted")
        state = _state(getattr(sandbox, "state", None))
        if state in ACTIVE:
            self._renew_ssh(ticket, sandbox)
            return LeaseStatus("active", gpus=getattr(sandbox, "gpu", None) or self.gpus)
        if state in GONE:
            if getattr(sandbox, "spot_evicted_at", None):
                return LeaseStatus("ended", end_reason="preempted")
            if getattr(sandbox, "queue_timed_out_at", None):
                return LeaseStatus("ended", end_reason="queue_timeout")
            return LeaseStatus("ended", end_reason="released")
        if state in STOPPED:
            return LeaseStatus("ended", end_reason="preempted")
        if state in FAILED:
            return LeaseStatus("ended", end_reason="error")
        return LeaseStatus("waiting")

    def release(self, account: int, ticket: str) -> dict[str, Any]:
        sandbox = self._get(ticket)
        if not self._ours(ticket, sandbox):
            raise PermissionError(f"sandbox {ticket} was not created by this provider for run {self.run_id}")
        if sandbox is not None and _state(getattr(sandbox, "state", None)) not in GONE:
            self.client.delete(sandbox, timeout=self.delete_timeout_s, wait=True)
            after = self._get(ticket)
            if after is not None and _state(getattr(after, "state", None)) not in GONE:
                raise RuntimeError(f"sandbox {ticket}: delete sent but it reads {_state(after.state)}")
        rows = self._rows()
        row = rows.setdefault(ticket, {"slot": account, "n": 0})
        row.update(open=False, released=True)
        self._save(rows)
        _drop_alias(ticket)
        _record_ledger("released", LEDGER_KIND, ticket, ticket)
        return {"ticket": ticket, "state": "ended"}

    def node(self, account: int, ticket: str) -> NodeLease:
        sandbox = self._get(ticket)
        if sandbox is None or _state(getattr(sandbox, "state", None)) not in ACTIVE:
            raise RuntimeError(f"sandbox {ticket} is not started")
        # the token is not stored in the lease file: attach() mints it into the 0600 ssh config
        return NodeLease(account, ticket, SSH_HOST, "", "", None, None, getattr(sandbox, "gpu", None) or self.gpus,
                         None)

    def balance(self) -> int | None:
        return None  # Daytona bills outside; the run's spend cap (Budget) is the guard

    def account_free(self, account: int) -> bool:
        return not any(r.get("open") and r.get("slot") == account for r in self._rows().values())

    def run_tickets(self) -> list[tuple[int, str]]:
        out = {name: int(r["slot"]) for name, r in self._rows().items() if r.get("open")}
        for sandbox in self._labelled():  # raises on a failed list: the patrol skips this tick
            name = str(sandbox.name)
            if name not in out and not self._rows().get(name, {}).get("released"):
                out[name] = int((sandbox.labels or {}).get(LABEL_SLOT) or 0)
        return [(slot, name) for name, slot in out.items()]

    def sweep(self) -> list[str]:
        """Delete every sandbox of this run (label + record): the teardown backstop."""
        done = []
        for slot, ticket in self.run_tickets():
            self.release(slot, ticket)
            done.append(ticket)
        return done

    # -- node route hook
    def ssh_access(self, ticket: str, sandbox: Any | None = None) -> tuple[str, float | None]:
        """A fresh gateway token for ``ticket`` and its expiry (epoch seconds, None if unknown)."""
        sandbox = sandbox if sandbox is not None else self._get(ticket)
        if sandbox is None:
            raise RuntimeError(f"sandbox {ticket} is gone")
        dto = sandbox.create_ssh_access(expires_in_minutes=self.ssh_token_minutes)
        return str(dto.token), self.clock() + 60 * self.ssh_token_minutes

    def _write_alias(self, ticket: str, token: str, expires: float | None) -> str:
        from . import node_pool  # noqa: PLC0415

        alias = node_pool.host_alias(ticket)
        ours = state_dir() / "ssh_config"
        ours.parent.mkdir(parents=True, exist_ok=True)
        blocks = node_pool._blocks(ours.read_text()) if ours.exists() else {}
        blocks[alias] = "\n".join([f"Host {alias}", f"  HostName {SSH_HOST}", f"  User {token}", "  Port 22",
                                   "  StrictHostKeyChecking accept-new",
                                   f"  UserKnownHostsFile {state_dir() / 'known_hosts'}",
                                   "  ServerAliveInterval 15", "  ServerAliveCountMax 4", ""])
        ours.write_text("".join(blocks.values()))
        os.chmod(ours, 0o600)
        node_pool.include_ours(ours)  # without it ``ssh <alias>`` does not resolve on a fresh pool host
        rows = self._rows()
        if ticket in rows:
            rows[ticket]["ssh_expires_at"] = expires
            self._save(rows)
        return alias

    def _renew_ssh(self, ticket: str, sandbox: Any) -> None:
        row = self._rows().get(ticket) or {}
        expires = row.get("ssh_expires_at")
        if expires is None:
            return  # not attached yet
        if float(expires) - self.clock() > 60 * self.ssh_token_minutes / 3:
            return
        token, until = self.ssh_access(ticket, sandbox)
        self._write_alias(ticket, token, until)

    def attach(self, lease: NodeLease, *, ssh_run) -> str:
        """An ssh alias through the Daytona gateway (a fresh token), then ``host_setup``."""
        token, expires = self.ssh_access(lease.ticket)
        alias = self._write_alias(lease.ticket, token, expires)
        for command in self.host_setup:
            out = ssh_run(alias, command)
            if out.returncode != 0:
                raise RuntimeError(f"host setup {command[:40]!r} on {lease.ticket}: exit {out.returncode}")
        return alias


def _drop_alias(ticket: str) -> None:
    from . import node_pool  # noqa: PLC0415

    node_pool.remove_ssh_alias(ticket)


def validate(*, image: str | None, snapshot: str | None, gpus: Any, gpu_type: Any, spot: Any, name_prefix: Any,
             ttl_minutes: Any = 0, ssh_token_minutes: Any = 1440) -> None:
    if bool(image) == bool(snapshot):
        raise ValueError("daytona provider: set exactly one of image / snapshot")
    if isinstance(gpus, bool) or not isinstance(gpus, int) or not 1 <= gpus <= MAX_GPUS:
        raise ValueError(f"daytona provider: gpus must be an integer 1..{MAX_GPUS}, got {gpus!r}")
    if gpu_type not in GPU_TYPES:
        raise ValueError(f"daytona provider: gpu_type {gpu_type!r} is not one of {sorted(GPU_TYPES)}")
    if not isinstance(spot, bool):
        raise ValueError("daytona provider: spot must be true or false")
    if not isinstance(name_prefix, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", name_prefix):
        raise ValueError("daytona provider: name_prefix: lower-case letters, digits and '-'")
    if not isinstance(ttl_minutes, int) or ttl_minutes < 0:
        raise ValueError("daytona provider: ttl_minutes must be >= 0 (0 = no wall-clock limit)")
    if not isinstance(ssh_token_minutes, int) or ssh_token_minutes < 10:
        raise ValueError("daytona provider: ssh_token_minutes must be an integer >= 10")


def from_config(cfg: dict[str, Any], run_id: str, *, sdk: Any = None, client: Any = None) -> DaytonaProvider:
    section = dict(cfg.get("provider") or {})
    section.pop("factory", None)
    unknown = set(section) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"daytona provider: unknown keys {sorted(unknown)}")
    section.pop("kind", None)
    return DaytonaProvider(run_id, slots=[int(a) for a in cfg["accounts"]], sdk=sdk, client=client, **section)


def registry_factory(cfg: dict[str, Any], run_id: str, section: dict[str, Any]) -> DaytonaProvider:  # noqa: ARG001
    """The registered ``daytona`` factory (``yeto.launch.pool.provider``)."""
    return from_config(cfg, run_id)


def cleanup_release(r: Any, ops: Any, dry_run: bool) -> tuple[bool, str, Any]:
    """Release one ``daytona_sandbox`` ledger entry (``yeto cleanup``): through
    ``ops.daytona_delete(resource)`` when the cleanup ops have it, else a provider
    rebuilt from the ledger attrs, whose ``release`` deletes only this run's sandboxes."""
    name = r.id or r.name
    if dry_run:
        return True, f"would delete Daytona sandbox {name}", None
    delete = getattr(ops, "daytona_delete", None)
    if delete is not None:
        res = delete(r)
    else:
        a = r.attrs
        prefix, run_id = a.get("prefix"), a.get("run_id")
        if not prefix or not run_id:
            return False, f"Daytona sandbox {name} without run/prefix in the ledger: not deleted", None
        p = DaytonaProvider(run_id, slots=[int(a.get("slot") or 0)], image="ledger",
                            api_url_env=a.get("api_url_env") or "DAYTONA_API_URL",
                            api_key_env=a.get("api_key_env") or "DAYTONA_API_KEY")
        p.prefix = prefix
        res = p.release(int(a.get("slot") or 0), name)
    return True, f"deleted Daytona sandbox {name}", res


def ledger_cost(r: Any, now: float, rates: Any) -> Any:
    """Cost of one ``daytona_sandbox`` entry: created to released (or now) at its
    ``usd_per_hour`` (the configured ``price_usd_h``; unpriced when not set)."""
    from yeto.cost import billing  # noqa: PLC0415

    rate = r.attrs.get("usd_per_hour")
    return billing._span_cost(r, r.created_at, now, None if rate is None else float(rate),  # noqa: SLF001
                              "" if rate is not None else "no price_usd_h configured")
