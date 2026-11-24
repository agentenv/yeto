"""The sandbox checks of an island fleet's agentic pools (``yeto run up`` preflight).

A fleet's pools run Harbor trials in sandboxes like a single trainer's, but the
sandbox side is shared: every island's pools draw on the same boat account, the
same AgentENV gateway or the same docker hosts. What the samplers check alone (one
process's ``agentic.max_in_flight`` against its own caps) does not see the sum.
Each check names the islands and pools it is about:

* :func:`provider_findings`: each agentic pool's sandbox provider
  (``agentic.task_defaults.provider``, else the pool env's ``HARBOR_ENV_TYPE``),
  a line per provider listing its pools, a warning when the islands train on
  trials of different providers, and the refusals a sampler would raise at start
  (remote-docker without ``harness.sandbox_caps.max_sandboxes``, a
  ``max_in_flight`` above the pool's own cap). A pool with ``agentic.envs`` reads
  the readiness manifest that ``yeto sandbox prepare`` writes in the run store:
  without ``sandbox.envs`` (or with ``sandbox.provider agentenv``) ``run up``
  prepares nothing and the pool would never claim a task. An ``agentic.envs.run``
  that is no label once the island's ``run_prefix`` (with its ``/trainer<k>``)
  is filled in fails every sampler of the pool at start.
* :func:`capacity_findings`: what the pools that share one sandbox resource hold
  at once (samplers x ``max_in_flight``) against that resource's limit: the boat
  account (:data:`BOAT_PLAN`, refused above it: the API answers 429 and trials
  abort), one AgentENV gateway (``harness.sandbox_caps.agentenv_max_active_n``,
  refused above it) and each docker host (``max_sandboxes`` of ``agentic.envs.hosts``:
  a warning, the dispenser gate waits for room). The count at the pools' targets
  decides; at the elastic maximum (replacements overlap) it is a warning.
* :func:`port_findings`: every shard of the pools that take a port of a records
  relay or tunnel range by shard index (``lo + shard_offset + n``) gets a port of the
  range, and no two shards of the fleet get the same one on one host.
* :func:`spend_lines`: the sandboxes' hourly price in the fleet's spend
  (``fleet.sandbox_usd_per_h``, :func:`yeto.breakers.fleet.sandbox_price`).

Import-light: no torch, ray, miles or harbor.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

REMOTE_DOCKER, AGENTENV, BOAT = "remote-docker", "agentenv", "boat"
# the boat plan (box_20, yeto.rl.harness.harbor.boat_client): one account for every sampler of the run;
# a run config's sandbox.boat replaces it (yeto.rl.up.boat_account)
BOAT_PLAN: dict[str, int | None] = {"max_active_n": 100, "starts_per_minute_n": 12, "starts_per_hour_n": 60}
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")  # yeto.sandbox.envs: agentic.envs.run


@dataclass(frozen=True)
class SandboxPool:
    """One agentic pool of the fleet as the sandbox checks see it."""

    name: str  # the compiled pool name, i<k>-<name>
    where: str  # "island k (name), pool i<k>-<name>"
    island: int | None
    provider: str | None
    max_in_flight: int | None
    shards: tuple[str, ...]  # the samplers at the pool's target (their ids)
    max_nodes: int  # samplers at the elastic maximum
    caps: Mapping[str, Any] = field(default_factory=dict)  # harness.sandbox_caps.* as the pool resolves them
    envs: Mapping[str, Any] | None = None  # agentic.envs of the rendered spec
    docker_host: str | None = None  # agentic.setup.sandbox.docker_host (or the env's), "envs": by shard
    gateway: str | None = None  # agentic.setup.agentenv.gateway_url (or E2B_API_URL)

    @property
    def nodes(self) -> int:
        return len(self.shards)

    @property
    def gated(self) -> bool:
        return bool(self.envs)

    def process_cap(self) -> int | None:
        """``harness.sandbox_caps.max_sandboxes`` for its provider (None: uncapped)."""
        caps = self.caps.get("max_sandboxes") or {}
        if not isinstance(caps, Mapping) or not self.provider:
            return None
        value = caps.get(self.provider, caps.get("*"))
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def per_sampler(self) -> int:
        """Sandboxes one of its samplers holds at once: ``max_in_flight`` (within its caps)."""
        held = [v for v in (self.max_in_flight, self.process_cap()) if v is not None]
        if self.provider == BOAT and self.caps.get("boat_max_active_n"):
            held.append(int(self.caps["boat_max_active_n"]))
        if self.provider == AGENTENV and self.caps.get("agentenv_max_active_n"):
            held.append(int(self.caps["agentenv_max_active_n"]))
        return min(held) if held else 0


def boat_account(sandbox: Mapping[str, Any] | None) -> dict[str, int | None]:
    """``sandbox.boat`` of a run config: the boat account's limits that every boat sampler
    of the run shares (``yeto.launch.rollout.pools.with_boat_shares``), over
    :data:`BOAT_PLAN`; ``starts_per_hour_n`` may be null (not held). Raises ValueError."""
    section = (sandbox or {}).get("boat")
    if section is None:
        return dict(BOAT_PLAN)
    if not isinstance(section, Mapping):
        raise ValueError("run config: sandbox.boat must be an object (the boat account's limits)")
    unknown = set(section) - set(BOAT_PLAN)
    if unknown:
        raise ValueError(f"run config: unknown sandbox.boat keys {sorted(unknown)} (known: {sorted(BOAT_PLAN)})")
    out = dict(BOAT_PLAN)
    for key, value in section.items():
        if value is None and key == "starts_per_hour_n":
            out[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"run config: sandbox.boat.{key} must be a whole number >= 1 (got {value!r})")
        out[key] = value
    return out


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _env_kwargs(env: Mapping[str, str]) -> dict[str, Any]:
    try:
        doc = json.loads(env.get("HARBOR_ENV_KWARGS") or "{}")
    except ValueError:
        return {}
    return doc if isinstance(doc, dict) else {}


def pool_of(name: str, where: str, island: int | None, spec: Mapping[str, Any] | None, *,
            env: Mapping[str, str], shards: Sequence[str], max_nodes: int,
            caps: Mapping[str, Any]) -> SandboxPool | None:
    """The :class:`SandboxPool` of a pool whose rendered spec is agentic (None: not agentic).
    ``env``: what its samplers' harness sees (the run's ``env`` and the pool's
    ``agentic_env_file``); ``caps``: its ``harness.sandbox_caps`` parameters."""
    agentic = (spec or {}).get("agentic")
    if not isinstance(agentic, Mapping):
        return None
    defaults = agentic.get("task_defaults") if isinstance(agentic.get("task_defaults"), Mapping) else {}
    provider = defaults.get("provider") or str(env.get("HARBOR_ENV_TYPE") or "").strip().lower() or None
    setup = agentic.get("setup") if isinstance(agentic.get("setup"), Mapping) else {}
    sandbox = setup.get("sandbox") if isinstance(setup.get("sandbox"), Mapping) else {}
    docker = sandbox.get("docker_host") or _env_kwargs(env).get("docker_host") or env.get("DOCKER_HOST") or None
    agentenv = setup.get("agentenv") if isinstance(setup.get("agentenv"), Mapping) else {}
    gateway = agentenv.get("gateway_url") or env.get("E2B_API_URL") or None
    envs = agentic.get("envs") if isinstance(agentic.get("envs"), Mapping) and agentic.get("envs") else None
    return SandboxPool(name=name, where=where, island=island, provider=None if provider is None else str(provider),
                       max_in_flight=_int(agentic.get("max_in_flight")), shards=tuple(shards),
                       max_nodes=max(int(max_nodes), len(shards)), caps=dict(caps), envs=envs,
                       docker_host=None if docker is None else str(docker),
                       gateway=None if gateway is None else str(gateway).rstrip("/"))


def _islands(pools: Sequence[SandboxPool]) -> str:
    return "; ".join(p.where for p in pools)


# ---------------------------------------------------------------------- providers


def provider_findings(pools: Sequence[SandboxPool], *, sandbox: Mapping[str, Any] | None
                      ) -> tuple[list[str], list[str]]:
    """``(problems, notes)`` of the agentic pools' sandbox providers (module doc).
    ``sandbox``: the run config's ``sandbox`` section."""
    problems: list[str] = []
    notes: list[str] = []
    if not pools:
        return problems, notes
    sandbox = dict(sandbox or {})
    run_provider = sandbox.get("provider", REMOTE_DOCKER)
    prepares = bool(sandbox.get("envs")) and run_provider == REMOTE_DOCKER
    by_provider: dict[str, list[SandboxPool]] = {}
    for p in pools:
        by_provider.setdefault(p.provider or "?", []).append(p)
    for provider, rows in sorted(by_provider.items()):
        shown = "unknown (no agentic.task_defaults.provider, no HARBOR_ENV_TYPE)" if provider == "?" else provider
        notes.append(f"fleet: sandbox provider {shown}: {_islands(rows)}")
    islands = {p.island: {q.provider for q in pools if q.island == p.island} for p in pools}
    kinds = {frozenset(v) for v in islands.values()}
    if len(kinds) > 1:
        notes.append("WARNING: fleet: the islands' trials run on different sandbox providers ("
                     + "; ".join(f"island {k}: {', '.join(sorted(str(x) for x in v))}" for k, v in sorted(
                         islands.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))
                     + "): their environments (and so their rewards) may differ; the syncer merges them as one task")
    for p in pools:
        if p.provider == REMOTE_DOCKER and p.process_cap() is None:
            problems.append(f"{p.where}: provider remote-docker needs harness.sandbox_caps.max_sandboxes (env "
                            "HARBOR_MAX_SANDBOXES, e.g. {\"remote-docker\": 8}) in the pool's agentic_env_file or "
                            "breakers.harness: its samplers refuse to start without it")
        cap = p.process_cap()
        if cap is not None and p.max_in_flight is not None and p.max_in_flight > cap:
            problems.append(f"{p.where}: agentic.max_in_flight {p.max_in_flight} exceeds its {p.provider} sandbox cap "
                            f"{cap} (harness.sandbox_caps.max_sandboxes): its samplers refuse to start")
        for provider, key in ((BOAT, "boat_max_active_n"), (AGENTENV, "agentenv_max_active_n")):
            limit = _int(p.caps.get(key))
            if p.provider == provider and limit is not None and p.max_in_flight is not None \
                    and p.max_in_flight > limit:
                problems.append(f"{p.where}: agentic.max_in_flight {p.max_in_flight} exceeds harness.sandbox_caps."
                                f"{key} {limit}: its samplers refuse to start")
        if p.gated and not prepares:
            why = (f"sandbox.provider is {run_provider}" if run_provider != REMOTE_DOCKER
                   else "the run config has no sandbox.envs")
            problems.append(f"{p.where}: its sampler spec has agentic.envs, so its dispenser takes only the tasks "
                            f"the readiness manifest of the run store marks ready, but {why}: `yeto run up` runs no "
                            "`yeto sandbox prepare` and the pool would never claim a task (set sandbox.envs to the "
                            "envs file, or drop agentic.envs from the pool's spec)")
        if p.gated:
            label = str(p.envs.get("run") or "").strip() if p.envs else ""
            if label and not LABEL.fullmatch(label):
                problems.append(f"{p.where}: agentic.envs.run {label!r} is not a label of [A-Za-z0-9_.-] (a fleet "
                                "pool's run_prefix ends in /trainer<k>; do not build the label from __RUN_PREFIX__): "
                                "its samplers fail at start")
    return problems, notes


# ---------------------------------------------------------------------- capacity


def _host_rows(pools: Sequence[SandboxPool], run_envs: Any
               ) -> dict[str, tuple[int | None, list[tuple[SandboxPool, int, int]]]]:
    """Docker host -> ``(max_sandboxes, [(pool, samplers at target, at max)])``."""
    from yeto.sandbox.envs import EnvsError, assign_host, parse_envs  # noqa: PLC0415

    out: dict[str, tuple[int | None, list[tuple[SandboxPool, int, int]]]] = {}
    for p in pools:
        if p.provider != REMOTE_DOCKER:
            continue
        cfg = run_envs
        if p.envs:
            try:
                cfg = parse_envs(p.envs, run=str(p.envs.get("run") or "run") if LABEL.fullmatch(
                    str(p.envs.get("run") or "run")) else "run")
            except (EnvsError, ValueError, TypeError):
                cfg = run_envs
        hosts: dict[str, list[str]] = {}
        if p.docker_host == "envs" and cfg is not None and cfg.hosts:
            try:
                for shard in p.shards:
                    hosts.setdefault(assign_host(cfg, shard).name, []).append(shard)
            except EnvsError:
                hosts = {}
        elif p.docker_host:
            host = cfg.host_of_docker(p.docker_host) if cfg is not None else None
            hosts[host.name if host is not None else p.docker_host] = list(p.shards)
        for name, shards in hosts.items():
            limit = None
            if cfg is not None:
                known = [h for h in cfg.hosts if h.name == name]
                limit = known[0].max_sandboxes if known else None
            cap, rows = out.get(name, (limit, []))
            peak = -(-len(shards) * p.max_nodes // max(1, p.nodes))  # its share of the elastic maximum
            rows.append((p, len(shards), peak))
            out[name] = (cap if cap is not None else limit, rows)
    return out


def capacity_findings(pools: Sequence[SandboxPool], *, run_envs: Any = None,
                      boat: Mapping[str, int | None] | None = None) -> tuple[list[str], list[str]]:
    """``(problems, notes)``: the sandboxes the pools sharing one resource hold at once
    against its limit (module doc). ``run_envs``: the run's ``sandbox.envs`` as an
    :class:`yeto.sandbox.envs.EnvsConfig` (None: none), for pools without ``agentic.envs``;
    ``boat``: the boat account's limits (``sandbox.boat``, default :data:`BOAT_PLAN`)."""
    problems: list[str] = []
    notes: list[str] = []

    def total(rows: Sequence[SandboxPool], *, peak: bool = False) -> int:
        return sum((p.max_nodes if peak else p.nodes) * p.per_sampler() for p in rows)

    def listed(rows: Sequence[SandboxPool]) -> str:
        return "; ".join(f"{p.where}: {p.nodes} sampler(s) x {p.per_sampler()}" for p in rows)

    boats = [p for p in pools if p.provider == BOAT]
    if boats:
        limits = dict(BOAT_PLAN, **(boat or {}))
        need, peak = total(boats), total(boats, peak=True)
        line = (f"the boat account holds {limits['max_active_n']} active sandboxes and the fleet's boat pools hold "
                f"{need} at once ({listed(boats)})")
        if need > limits["max_active_n"]:
            problems.append(f"{line}: creates above the account limit are refused (HTTP 429) and those trials "
                            "abort; lower agentic.max_in_flight or the pools' targets")
        elif peak > limits["max_active_n"]:
            notes.append(f"WARNING: {line}, {peak} at the pools' elastic maximum (replacements overlap): above "
                         f"{limits['max_active_n']} creates are refused")
        else:
            notes.append(f"fleet: {line}")
        rate = sum(p.nodes * int(p.caps.get("boat_starts_per_minute_n") or BOAT_PLAN["starts_per_minute_n"])
                   for p in boats)
        if rate > limits["starts_per_minute_n"]:
            notes.append(f"WARNING: fleet: each boat sampler may start harness.sandbox_caps.boat_starts_per_minute_n "
                         f"sandboxes a minute; together {rate} > the account's {limits['starts_per_minute_n']} "
                         f"({_islands(boats)}): a burst gets HTTP 429 (set YETO_BOAT_STARTS_PER_MINUTE per pool so "
                         "the sum stays within the plan)")
    gateways: dict[str, list[SandboxPool]] = {}
    for p in pools:
        if p.provider == AGENTENV:
            gateways.setdefault(p.gateway or "?", []).append(p)
    for gateway, rows in sorted(gateways.items()):
        limits = [v for v in (_int(p.caps.get("agentenv_max_active_n")) for p in rows) if v is not None]
        limit = min(limits) if limits else None
        need = total(rows)
        shown = "an unnamed gateway" if gateway == "?" else f"gateway {gateway}"
        if limit is not None and need > limit:
            problems.append(f"AgentENV {shown} holds {limit} microVMs (harness.sandbox_caps.agentenv_max_active_n) "
                            f"and the fleet's pools on it hold {need} at once ({listed(rows)}): lower "
                            "agentic.max_in_flight or the pools' targets")
        else:
            notes.append(f"fleet: AgentENV {shown}: {need} microVMs at once ({listed(rows)})"
                         + (f" of {limit}" if limit is not None else ""))
    for host, (limit, rows) in sorted(_host_rows(pools, run_envs).items()):
        need = sum(n * p.per_sampler() for p, n, _ in rows)
        peak = sum(m * p.per_sampler() for p, _, m in rows)
        shown = "; ".join(f"{p.where}: {n} sampler(s) x {p.per_sampler()}" for p, n, _ in rows)
        if limit is None:
            notes.append(f"fleet: sandbox host {host}: {need} trial containers at once ({shown}); no max_sandboxes, "
                         "so the dispenser gate cannot hold claims back when it is full")
        elif need > limit:
            notes.append(f"WARNING: fleet: sandbox host {host} holds {limit} trial containers (max_sandboxes) and the "
                         f"fleet's pools on it run {need} at once ({shown}): their samplers wait for room "
                         "(rl_dispense_wait_capacity); add a host to agentic.envs.hosts or lower max_in_flight")
        elif peak > limit:
            notes.append(f"WARNING: fleet: sandbox host {host}: {need} of {limit} trial containers at the pools' "
                         f"targets, {peak} at their elastic maximum ({shown})")
        else:
            notes.append(f"fleet: sandbox host {host}: {need} of {limit} trial containers ({shown})")
    return problems, notes


# ---------------------------------------------------------------------- ports


def _range(key: Sequence[str]) -> tuple[str, int, int] | None:
    """``(listen host part, lo, hi)`` of a :func:`yeto.rl.fleet.port_share_key`."""
    raw = str(key[2])
    ip, _, ports = raw.rpartition(":") if key[0] == "tunnel" else ("", "", raw)
    lo, _, hi = ports.partition("-")
    try:
        return ip, int(lo), int(hi or lo)
    except ValueError:
        return None  # the node setup names the bad range


def port_findings(rows: Sequence[tuple[str, Sequence[str] | None, int, int]]) -> list[str]:
    """Problems of the shards' listen ports. ``rows``: ``(where, port_share_key, shard_offset,
    shard_limit)`` of each pool. A shard past the range, or two shards of the fleet on one
    port of one relay / tunnel host, is named with its pools."""
    problems: list[str] = []
    taken: dict[tuple[str, str, str, int], list[str]] = {}
    for where, key, offset, limit in rows:
        if key is None:
            continue
        span = _range(key)
        if span is None:
            continue
        ip, lo, hi = span
        need = lo + int(offset) + int(limit) - 1
        if need > hi:
            problems.append(f"{where}: shards {offset}..{int(offset) + int(limit) - 1} of the {key[0]} port range "
                            f"{key[2]} need ports up to {need}, past its end {hi}: widen it to one port per shard of "
                            "every pool that shares it (the fleet gives each pool its own block, shard_offset)")
        for n in range(int(limit)):
            taken.setdefault((key[0], key[1], ip, lo + int(offset) + n), []).append(where)
    clashes: dict[tuple[str, str], set[int]] = {}
    for (kind, host, _ip, port), who in taken.items():
        if len(set(who)) > 1:
            clashes.setdefault((kind, " and ".join(sorted(set(who)))), set()).add(port)
    for (kind, who), ports in sorted(clashes.items()):
        problems.append(f"{who}: their shards listen on the same {kind} port(s) {sorted(ports)[:8]}: two nodes would "
                        "forward one port (set one listen range for the pools, the fleet then gives each its own "
                        "block of it, or give them ranges that do not overlap)")
    return problems


# ---------------------------------------------------------------------- spend


def spend_lines(pools: Sequence[SandboxPool], sandbox_usd_per_h: float | None) -> list[str]:
    """The sandboxes in the fleet's spend: ``fleet.sandbox_usd_per_h`` (counted with the
    pools, trainers and syncer, ``yeto.breakers.fleet.fleet_spend``), or a warning that a
    fleet with agentic pools counts its sandboxes at $0/h."""
    if not pools:
        return []
    providers = ", ".join(sorted({p.provider or "?" for p in pools}))
    if not sandbox_usd_per_h:
        return [f"fleet: fleet.sandbox_usd_per_h is unset: the fleet spend counts the sandboxes ({providers}) at "
                "$0/h (set it to their hourly price: boat sandboxes, the docker hosts)"]
    return [f"fleet: sandboxes ({providers}) at ${sandbox_usd_per_h:g}/h in the fleet spend"]
