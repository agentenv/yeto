"""The dispenser's sandbox gate: readiness filter and capacity-aware claims.

:class:`SandboxGate` is what :class:`yeto.rl.engine.store_dispense.StoreDispensedPrompts`
asks before a claim pass:

* ``view()``: the readiness manifest read once per scan (:class:`.ready.ReadyView`
  for this sampler's host). Not-ready tasks are not claimed; tasks whose
  validation failed are settled ``dropped`` with the reason; a task that
  becomes ready later is claimed at the next scan. The manifest is not in the
  dispenser plan, so every sampler keeps the same plan.
* ``free_groups()``: how many groups may start now, so a sampler claims only
  what it can run at once instead of holding leases while trials queue for a
  sandbox. The smaller of the process cap (``HARBOR_MAX_SANDBOXES`` through
  :mod:`yeto.rl.harness.harbor.capacity`, minus slots held) and the host's
  free room (``max_sandboxes`` of the host minus its running containers,
  every owner counted, probed at most every ``ttl_s``), divided by the group
  size. ``None``: no limit known.

  With an owner (:mod:`.trial_labels`: run label, shard, incarnation) the
  probe classifies the running containers as ``ours`` (this run), ``stale``
  (this run and shard, an earlier incarnation: orphans of a killed sampler)
  and ``foreign``; stale ones are removed (:func:`.trial_labels.sweep_stale`)
  and not counted, and the breakdown is kept in :attr:`SandboxGate.last_probe`
  (logged when the host has no room, and on the ``rl_dispense_wait_capacity``
  event).
* ``sweep_stale()``: the same removal at sampler start (dispenser recover).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from .docker import DockerHost
from .envs import EnvsConfig, HostConfig, parse_envs
from .ready import ReadyView, read_manifest
from .trial_labels import OWNER_KEYS, RUN_KEY, Owner, owner_from_env, sweep_stale

logger = logging.getLogger(__name__)

HOST_ENV = "YETO_SANDBOX_HOST"
DEFAULT_TTL_S = 15.0


class SandboxGate:
    def __init__(self, objects: Any, cfg: EnvsConfig, *, host: HostConfig | None, group_size: int,
                 provider: str | None = None, docker: DockerHost | None = None,
                 limits: Any = None, ttl_s: float = DEFAULT_TTL_S,
                 clock: Callable[[], float] = time.monotonic, owner: Owner | None = None) -> None:
        self.objects, self.cfg, self.host = objects, cfg, host
        self.group_size = max(1, int(group_size))
        self.provider = provider
        self.docker = docker
        self._limits = limits
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self._probe: tuple[float, int | None] | None = None
        self.owner = owner
        self.last_probe: dict[str, Any] = {}

    @property
    def host_name(self) -> str | None:
        return None if self.host is None else self.host.name

    def view(self) -> ReadyView:
        try:
            doc = read_manifest(self.objects, self.cfg.ready_key)
        except Exception as error:  # noqa: BLE001 - an unreadable manifest dispenses nothing this scan
            logger.warning("sandbox gate: manifest read failed: %r", error)
            doc = None
        return ReadyView(doc, host=self.host_name, require_validation=self.cfg.require_validation,
                         affinity=self.cfg.affinity)

    def _process_free(self) -> int | None:
        if not self.provider:
            return None
        limits = self._limits
        if limits is None:
            from yeto.rl.harness.harbor.capacity import process_limits  # noqa: PLC0415

            limits = process_limits()
        slots = limits.slots(self.provider)
        return None if slots is None else max(0, slots.limit - slots.held)

    def _host_free(self) -> int | None:
        if self.host is None or self.host.max_sandboxes is None or self.docker is None:
            return None
        now = self.clock()
        if self._probe is not None and now - self._probe[0] < self.ttl_s:
            return self._probe[1]
        try:
            free: int | None = max(0, int(self.host.max_sandboxes) - self._count())
        except Exception as error:  # noqa: BLE001 - unknown room: claim nothing until the probe answers
            logger.warning("sandbox gate: container count on %s failed: %r", self.host.name, error)
            free = 0
        self.last_probe["free"] = free
        if free == 0:
            logger.warning("sandbox gate: no room on %s (max_sandboxes=%s): %s", self.host.name,
                           self.host.max_sandboxes, self.last_probe)
        self._probe = (now, free)
        return free

    def _count(self) -> int:
        """Running containers on the host, this shard's stale incarnations removed first."""
        owner = self.owner
        if owner is None or not hasattr(self.docker, "labelled_containers"):
            n = int(self.docker.count_running())
            self.last_probe = {"host_running": n}
            return n
        rows = self.docker.labelled_containers(OWNER_KEYS)
        stale = sum(1 for r in rows if owner.is_stale(r["labels"]))
        removed = 0
        if stale:
            try:
                removed = sweep_stale(self.docker, owner)
            except Exception as error:  # noqa: BLE001 - counted as running; retried at the next probe
                logger.warning("sandbox gate: stale sweep on %s failed: %r", self.host.name, error)
        ours = sum(1 for r in rows if r["labels"].get(RUN_KEY) == owner.run)
        shard = sum(1 for r in rows if r["labels"].get(RUN_KEY) == owner.run
                    and r["labels"].get("ai.yeto.shard") == owner.shard and not owner.is_stale(r["labels"]))
        running = len(rows) - min(removed, stale)
        self.last_probe = {"host_running": len(rows), "ours": ours, "shard": shard, "stale": stale,
                           "removed": removed, "foreign": len(rows) - ours}
        return running

    def sweep_stale(self) -> int:
        """Remove this shard's trial containers of earlier incarnations (sampler start)."""
        if self.docker is None or self.owner is None or not hasattr(self.docker, "labelled_containers"):
            return 0
        try:
            n = sweep_stale(self.docker, self.owner)
        except Exception as error:  # noqa: BLE001 - the probe retries it
            logger.warning("sandbox gate: stale sweep failed: %r", error)
            return 0
        self._probe = None
        return n

    def free_groups(self) -> int | None:
        rooms = [r for r in (self._process_free(), self._host_free()) if r is not None]
        return None if not rooms else min(rooms) // self.group_size


def gate_from_spec(spec: Mapping[str, Any], objects: Any, *, environ: Mapping[str, str] | None = None,
                   docker_factory: Callable[[HostConfig], DockerHost] | None = None,
                   limits: Any = None) -> SandboxGate | None:
    """The gate of a sampler spec with ``agentic.envs`` (None without it). The
    sampler's host: ``YETO_SANDBOX_HOST`` (a host name), else the host whose
    ``docker_host`` equals ``HARBOR_ENV_KWARGS.docker_host`` or ``DOCKER_HOST``."""
    import json  # noqa: PLC0415

    agentic = spec.get("agentic") or {}
    section = agentic.get("envs") if isinstance(agentic, Mapping) else None
    if not section:
        return None
    env = os.environ if environ is None else environ
    cfg = parse_envs(section, run=str(section.get("run") or spec.get("run_prefix") or "run"))
    host = None
    if env.get(HOST_ENV):
        host = cfg.host(str(env[HOST_ENV]))
    else:
        try:
            kwargs = json.loads(env.get("HARBOR_ENV_KWARGS") or "{}")
        except ValueError:
            kwargs = {}
        dh = (kwargs.get("docker_host") if isinstance(kwargs, Mapping) else None) or env.get("DOCKER_HOST")
        host = cfg.host_of_docker(dh)
        if host is None and len(cfg.hosts) == 1:
            host = cfg.hosts[0]
    provider = ((agentic.get("task_defaults") or {}).get("provider")
                or str(env.get("HARBOR_ENV_TYPE", "") or "").strip().lower() or None)
    n = int((spec.get("sampling") or {}).get("n") or 1)
    owner = owner_from_env(env)
    docker = None
    if host is not None and (host.max_sandboxes is not None or owner is not None):
        docker = (docker_factory or (lambda h: DockerHost(h.docker_host)))(host)
    return SandboxGate(objects, cfg, host=host, group_size=n, provider=provider, docker=docker,
                       limits=limits, owner=owner)
