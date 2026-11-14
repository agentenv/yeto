"""``python -m yeto.launch.nc``: National Compute nodes as sky SSH node pools.

    proxy    --account N HOST PORT        ssh ProxyCommand through the workspace relay
    preflight --account N                 tools, account free, balance, relay, sky
    up       --account N --yes [--route ssh|sky]  order -> active -> alias -> docker on NVMe (ssh, default)
                                          or host bootstrap -> sky ssh up -> AMD GPUs (sky; refused on
                                          nodes with the provider's health k3s)
    probe    --ticket T [--spec F]        card check, image pull, vllm --version, rocm-smi, spec preflight
    ssh-pull start|status --ticket T      rate-limited model pull (aws-cli container, <= 3000 MB/s)
    ssh-sampler start|stop|status --ticket T --shard S   engine sampler container
    hello    --ticket T                   sky hello task on the node's pool, then the cluster goes down
    pull     --ticket T --src S3 --dest D rate-limited model pull as a sky task
    sampler  --ticket T --spec F ...      engine sampler as a sky task (own buffer shard)
    down     --account N --ticket T       sky clusters + pool down, delete the ticket (only ours), confirm ended
    pool     --config F [--dry-run]       lease-aware, eviction-tolerant sampler pool loop
                                          (``route: ssh`` default; ``elastic`` knobs: .elastic);
                                          --dry-run validates and prints orders + spend, orders nothing
    pool-stop --config F                  release every node the pool created (running or not)
    pool-retry --config F                 after a configuration error: restart in place, same node(s)
    pool-status --config F                the pool's state, billed nodes and last events (read-only)
    pool-state-downgrade --config F       before a rollback of yeto: write the state file in the older format

A National Compute lease is NOT ended at ``lease_until``: the node stays active
and billed until its ticket is deleted. See docs/NC_POOL.md.

Environment: ``NC_MANAGE`` / ``NC_QUERY`` (the order tools), ``NC_TOOL_DIR``
(the controller tool, for ``port.open`` and the workspace shell), ``NC_NODE_BIN``,
``YETO_NC_STATE_DIR`` (default ``~/.yeto/nc``), ``YETO_NC_ACCOUNTS`` (accounts
this tool may order on, e.g. ``1``; required for ``up``/``pool``), ``YETO_SKY_BIN``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .. import agentic_setup
from ..pool import status as island_status
from ..pool.island_pool import Member
from . import elastic, node_pool, scale_args, ssh_sampler, tasks, tunnel
from .client import (Ledger, NcClient, NcError, NodeLease, hourly_cost_ucr, parse_time, state_file_lock,
                     write_json_atomic)
from .lease_pool import PoolConfig
from .provider import NCProvider, load_provider


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def ledger() -> Ledger:
    return Ledger(tunnel.state_dir() / "tickets.json")


def allowed_accounts() -> set[int]:
    raw = os.environ.get("YETO_NC_ACCOUNTS", "")
    return {int(x) for x in raw.replace(" ", "").split(",") if x}


def client(account: int, *, ordering: bool = False) -> NcClient:
    return NcClient(account, ledger(), allowed_accounts=allowed_accounts() if ordering else None)


def leases_path() -> Path:
    return tunnel.state_dir() / "leases.json"


def save_lease(lease: NodeLease) -> None:
    """Add a lease to the lease file: read-modify-write under the state file lock (the
    threads of the pool, and another command on the same state dir), then an atomic
    rename (a concurrent reader never sees a half-written file, no writer drops
    another's lease)."""
    path = leases_path()
    with state_file_lock(path):
        doc = json.loads(path.read_text()) if path.exists() else {}
        doc[lease.ticket] = lease.to_json()
        write_json_atomic(path, doc)


def load_lease(ticket: str) -> NodeLease:
    doc = json.loads(leases_path().read_text()) if leases_path().exists() else {}
    if ticket not in doc:
        raise SystemExit(f"no lease record for ticket {ticket} in {leases_path()}")
    return NodeLease(**doc[ticket])


# -- workspace relay ---------------------------------------------------------

def workspace_run(account: int, command: str, *, timeout_ms: int = 120_000,
                  run=subprocess.run) -> subprocess.CompletedProcess:
    return run([tunnel.node_bin(), "tool/wsrun.mjs", "--account", str(account), "--timeout-ms",
                str(timeout_ms), "--", command], cwd=tunnel.tool_dir(), capture_output=True,
               text=True, timeout=timeout_ms / 1000 + 60, check=False)


def ensure_relay(account: int, *, run=subprocess.run) -> None:
    """Start the relay in the account's workspace; allow port 22 of this tool's open nodes."""
    allow = sorted({f"{r['ip']}:22" for r in ledger().open_rows() if r.get("ip") and r["account"] == account})
    out = workspace_run(account, tunnel.relay_start_command(allow), run=run)
    if "__RELAY_UP__" not in (out.stdout or ""):
        raise NcError(f"workspace relay did not start: {(out.stdout or '')[-300:]} {(out.stderr or '')[-200:]}")


# -- steps -----------------------------------------------------------------

def preflight(account: int) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for tool in ("nc-manage", "nc-query", "ssh", "kubectl"):
        checks[f"tool:{tool}"] = bool(shutil.which(tool))
    checks["tool:sky"] = bool(shutil.which(node_pool.sky_cmd()[0]))
    checks["nc_tool_dir"] = Path(tunnel.tool_dir(), "tool", "core.mjs").exists()
    st = client(account).status()
    own = NcClient.own_ticket(st)
    checks["account_free"] = own is None
    checks["own_ticket"] = (own or {}).get("id")
    checks["balance_ucr"] = (st.get("balance") or {}).get("balance_ucredits")
    checks["hourly_cost_ucr"] = hourly_cost_ucr()
    checks["account_allowed"] = account in allowed_accounts()
    return checks


def up(account: int, *, model: str = "mi355x", wait_s: float = 3600, route: str = "ssh") -> NodeLease:
    c = client(account, ordering=True)
    pub = node_pool.ensure_key()
    ensure_relay(account)  # the relay is up before the order: no billed minutes spent on it
    ticket = c.create(model=model, pubkey=pub)
    _log(f"ordered ticket {ticket} on account {account}")
    try:
        lease = c.wait_active(ticket, timeout_s=wait_s, log=_log)
    except Exception:
        _log(f"ticket {ticket} did not become active; deleting it")
        c.delete(ticket)
        raise
    save_lease(lease)
    _log(f"active {lease.ip} lease_until {lease.lease_until}")
    try:
        attach_ssh(lease) if route == "ssh" else attach(lease)
    except Exception as exc:
        _log(f"attach failed on {ticket}: {exc} (the node stays up; `down` deletes it)")
        raise
    return lease


def reach(lease: NodeLease) -> str:
    """Relay allow-list + alias; wait until ssh answers."""
    ensure_relay(lease.account)
    alias = node_pool.write_ssh_alias(lease)
    for attempt in range(30):
        probe = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", alias, "echo SSH_OK"],
                               capture_output=True, text=True, timeout=120, check=False)
        if "SSH_OK" in probe.stdout:
            break
        _log(f"ssh {alias}: not yet ({probe.stderr.strip()[-160:]})")
        time.sleep(10)
    else:
        raise NcError(f"ssh {alias} never answered")
    _log(f"ssh {alias} ok")
    return alias


def pull_ended_without_done(status: str) -> bool:
    """The pull status says the pull is over but not done: a ``YETO_NC_PULL fail`` line
    of the current attempt, the pull container exited (any code; ``done`` is its last
    line) or is gone (``absent``: docker restarted,
    removed), or the no-docker route logged a nonzero ``YETO_NC_PULL exit``. Waiting
    for ``done`` then only bills the node until ``timeout_s``."""
    first = status.splitlines()[0].strip() if status.strip() else ""
    if first == "absent" or re.match(r"exited \d+$", first):
        return True
    current = status.rsplit("YETO_NC_PULL start", 1)[-1]  # an appended log keeps earlier attempts
    return "YETO_NC_PULL fail" in current or any(
        code != "0" for code in re.findall(r"YETO_NC_PULL exit (\d+)", current))


def attach_ssh(lease: NodeLease) -> str:
    """The SSH + docker route (no k3s): reach, docker on NVMe."""
    alias = reach(lease)
    out = node_pool.ssh_run(alias, ssh_sampler.docker_host_script(), timeout=1200)
    _log((out.stdout or "").strip().splitlines()[-1] if (out.stdout or "").strip() else f"docker: {out.stderr[-300:]}")
    if "YETO_NC_DOCKER_OK" not in (out.stdout or ""):
        raise NcError("docker host step failed")
    return alias


def attach(lease: NodeLease) -> str:
    """sky SSH node pool route: reach, host bootstrap, sky pool, AMD GPUs (idempotent).
    Refused on nodes with the provider's health k3s (node_pool)."""
    alias = reach(lease)
    out = node_pool.ssh_run(alias, node_pool.host_bootstrap_script())
    _log(out.stdout.strip().splitlines()[-1] if out.stdout.strip() else f"bootstrap: {out.stderr[-300:]}")
    if "YETO_NC_BOOTSTRAP_OK" not in out.stdout:
        raise NcError("host bootstrap failed")
    pool = node_pool.write_pool(lease.ticket)
    res = node_pool.sky_ssh_up(pool)
    _log(f"sky ssh up {pool}: exit {res.returncode} {(res.stdout or '')[-400:]} {(res.stderr or '')[-400:]}")
    if res.returncode != 0:
        raise NcError(f"sky ssh up {pool} failed")
    gpu = node_pool.ssh_run(alias, node_pool.amd_gpu_script(lease.gpus or 8), timeout=600)
    _log((gpu.stdout or "").strip()[-300:])
    if "YETO_NC_AMD_GPU_OK" not in gpu.stdout:
        raise NcError("AMD device plugin did not report the GPUs")
    return pool


def _sky():
    import sky

    return sky


def cluster_name(kind: str, ticket: str) -> str:
    return f"yeto-nc-{kind}-{ticket}"


def launch(task_doc: dict[str, Any], name: str, *, down: bool) -> int:
    sky = _sky()
    task = sky.Task.from_yaml_config(task_doc)
    _record_cluster(name)
    job_id, _handle = sky.stream_and_get(sky.launch(task, cluster_name=name))
    code = sky.tail_logs(name, job_id, follow=True)
    if down:
        sky.stream_and_get(sky.down(name))
        _forget_cluster(name)
    return int(code or 0)


def _clusters_path() -> Path:
    return tunnel.state_dir() / "clusters.json"


def _record_cluster(name: str) -> None:
    names = set(json.loads(_clusters_path().read_text())) if _clusters_path().exists() else set()
    names.add(name)
    _clusters_path().write_text(json.dumps(sorted(names)))


def _forget_cluster(name: str) -> None:
    if _clusters_path().exists():
        names = set(json.loads(_clusters_path().read_text())) - {name}
        _clusters_path().write_text(json.dumps(sorted(names)))


def down(account: int, ticket: str) -> dict[str, Any]:
    """Our sky clusters on the pool, the pool, the alias, then the ticket (guarded)."""
    if _clusters_path().exists():
        for name in json.loads(_clusters_path().read_text()):
            if name.endswith(f"-{ticket}"):
                try:
                    _sky().stream_and_get(_sky().down(name))
                except Exception as exc:  # noqa: BLE001  (the ticket goes regardless)
                    _log(f"sky down {name}: {exc}")
                _forget_cluster(name)
    pool = node_pool.pool_name(ticket)
    if node_pool.SKY_POOLS_FILE.exists() and pool in (node_pool.SKY_POOLS_FILE.read_text()):
        res = node_pool.sky_ssh_down(pool)
        _log(f"sky ssh down {pool}: exit {res.returncode}")
        node_pool.remove_pool(ticket)
    node_pool.remove_ssh_alias(ticket)
    result = client(account).delete(ticket)
    _log(f"ticket {ticket}: {result}")
    return result


# -- pool ops ----------------------------------------------------------------

class NcSkyOps:
    """:class:`~.lease_pool.PoolOps` over National Compute and sky."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def order(self, account: int) -> str:
        c = client(account, ordering=True)
        ensure_relay(account)
        return c.create(model=machine_type(self.cfg), pubkey=node_pool.ensure_key(), run=run_id(self.cfg))

    def poll(self, m: Member):
        t = client(m.account).status().get("ticket") or {}
        if t.get("id") != m.ticket or t.get("state") in ("ended", None):
            return "ended", None
        if t.get("state") == "active" and (t.get("node") or {}).get("public_ip"):
            if not leases_path().exists() or m.ticket not in json.loads(leases_path().read_text()):
                save_lease(client(m.account).wait_active(m.ticket, timeout_s=60))
            return "active", parse_time(t.get("lease_until"))
        return "waiting", None

    def start(self, m: Member, shards: list[str]) -> None:
        lease = load_lease(m.ticket)
        pool = attach(lease)
        if self.cfg.get("model_src"):
            launch(tasks.with_secrets(tasks.model_pull_task(pool, src=self.cfg["model_src"],
                                                            dest=self.cfg["model_dest"]),
                                      self.cfg.get("r2_env_file")), cluster_name("pull", m.ticket), down=True)
        base = json.loads(Path(self.cfg["spec"]).read_text())
        spec_path = tunnel.state_dir() / f"spec-{m.shard}.json"
        spec = tasks.apply_sampler_overrides(tasks.sampler_spec(base, sampler_id=m.shard, shards=shards),
                                             data_slot=m.data_slot, continues_shard=m.replaces_shard,
                                             **_overrides(self.cfg.get("sampler") or {}))
        if agentic_setup.parse(spec) is not None:  # agentic setup hook: ssh route only
            raise ValueError("agentic.setup runs through ssh_sampler.start_sampler (the ssh route); "
                             "the sky route cannot prepare an agentic node")
        tasks.write_spec(spec, spec_path)
        doc = tasks.with_secrets(tasks.sampler_task(
            pool, spec_path=str(spec_path), run_prefix=self.cfg["run_prefix"], workdir=self.cfg["workdir"],
            protected=frozenset(self.cfg.get("protected_prefixes", ()))), self.cfg.get("r2_env_file"))
        sky = _sky()
        _record_cluster(cluster_name("sampler", m.ticket))
        sky.launch(sky.Task.from_yaml_config(doc), cluster_name=cluster_name("sampler", m.ticket))

    def ready(self, m: Member) -> bool:
        sky = _sky()
        try:
            records = sky.get(sky.status([cluster_name("sampler", m.ticket)]))
        except Exception:  # noqa: BLE001
            return False
        return bool(records) and str(records[0].get("status")).endswith("UP")

    def drain(self, m: Member) -> None:
        down(m.account, m.ticket)

    def balance(self) -> int | None:
        return client(slot_list(self.cfg)[0]).balance()

    def account_free(self, account: int) -> bool:
        return client(account).account_free()


class NcSshOps:
    """:class:`~.lease_pool.PoolOps` over National Compute and the SSH + docker
    route (the route verified on MI355X nodes; the sky route is refused on nodes
    with the provider's health k3s). Config keys (``pool --config``): see
    :func:`pool_config` and the customer guide.

    Every ticket is ordered with the run tag in the ledger; drain and the patrol
    only ever delete a ticket the ledger holds for this run, by id, after the
    client's current-ticket check (never ``--all``, never another session's ticket)."""

    start_in_background = True  # node setup + model pull: tens of minutes

    def __init__(self, cfg: dict[str, Any], *, run=subprocess.run, provider=None):
        self.cfg = cfg
        self.run_id = run_id(cfg)
        self._run = run
        self.provider = provider if provider is not None else load_provider(cfg, self.run_id,
                                                                            nc_factory=nc_provider)

    # -- supply (the provider)
    def order(self, account: int) -> str:
        return self.provider.order(account)

    def poll(self, m: Member) -> island_status.LeaseStatus:
        status = self.provider.status(m.account, m.ticket)
        if status.state == "active" and (not leases_path().exists()
                                         or m.ticket not in json.loads(leases_path().read_text())):
            save_lease(self.provider.node(m.account, m.ticket))
        return status

    def balance(self) -> int | None:
        return self.provider.balance()

    def account_free(self, account: int) -> bool:
        return self.provider.account_free(account)

    def run_tickets(self) -> list[tuple[int, str]]:
        return self.provider.run_tickets()

    def slot_halt(self, account: Any) -> dict[str, Any] | None:
        """The ledger's halt of a National Compute account (None: not halted, or not NC)."""
        book = getattr(self.provider, "ledger", None)
        if not isinstance(self.provider, NCProvider) or book is None:
            return None
        return book.halted(int(account))

    def released_by_us(self, ticket: str) -> bool:
        """The ticket ledger says this tool deleted ``ticket`` (an earlier pool instance)."""
        try:
            return any(r.get("ticket") == ticket and r.get("released_by") == "yeto" for r in ledger().load())
        except Exception:  # noqa: BLE001
            return False

    # -- node
    def start(self, m: Member, shards: list[str]) -> None:
        lease = load_lease(m.ticket)
        attach = getattr(self.provider, "attach", None)  # a provider's own reach + host setup
        if attach is not None:
            attach(lease, ssh_run=lambda alias, cmd: ssh_sampler.ssh(alias, cmd, run=self._run, timeout=1200))
        else:
            attach_ssh(lease)
        pull = self.cfg.get("model_pull")
        if pull:
            self._pull(lease, pull)
        self._hooks(lease, m)
        self._start_sampler(lease, m, shards)

    def restart(self, m: Member, shards: list[str]) -> None:
        lease = load_lease(m.ticket)
        self._hooks(lease, m)
        self._start_sampler(lease, m, shards)

    @property
    def metrics_port(self) -> int | None:
        """The engine's ``/metrics`` port on the node (``elastic.metrics_port``; by default
        the sampler spec's ``engine.server.port``)."""
        if not hasattr(self, "_metrics_port"):
            value = island_status.ElasticConfig.from_dict(self.cfg.get("elastic")).metrics_port
            spec = None
            if value == "spec":
                try:
                    spec = json.loads(Path(os.path.expanduser(str(self.cfg["spec"]))).read_text())
                except (KeyError, OSError, ValueError) as error:
                    _log(f"pool: cannot read the sampler spec for its server port ({error!r}); "
                         "engine metrics are not read")
            self._metrics_port = island_status.resolve_metrics_port(value, spec)
        return self._metrics_port

    @property
    def reads_metrics(self) -> bool:
        return self.metrics_port is not None

    @property
    def sampler_route(self) -> Any:
        """A sampler route without docker (``start`` / ``probe_script`` / ``stop_command`` /
        ``pull``): the provider's ``sampler_route``, or pool config ``"sampler_route":
        "process"`` (:class:`yeto.launch.daytona_sampler.DaytonaSamplerRoute`); None: the
        docker route of :mod:`.ssh_sampler`."""
        if not hasattr(self, "_route"):
            route = getattr(self.provider, "sampler_route", None)
            kind = self.cfg.get("sampler_route")
            if route is None and kind == "process":
                from ..daytona_sampler import DaytonaSamplerRoute  # noqa: PLC0415

                route = DaytonaSamplerRoute(gpus=int(getattr(self.provider, "gpus", 8) or 8))
            elif route is None and kind not in (None, "docker"):
                raise NcError(f"sampler_route {kind!r}: 'docker' (default) or 'process'")
            self._route = route
        return self._route

    def probe(self, m: Member) -> island_status.Probe:
        alias = node_pool.host_alias(m.ticket)
        if self.sampler_route is not None:
            script = self.sampler_route.probe_script(m.shard, self._tape_glob(m.shard), self.metrics_port)
        else:
            script = elastic.probe_script(ssh_sampler.container_name(m.shard), self._tape_glob(m.shard),
                                          metrics_port=self.metrics_port)
        try:
            out = self._run(["ssh", "-n", "-T", "-o", "BatchMode=yes", "-o",
                             f"ConnectTimeout={int(self.cfg.get('probe_connect_timeout_s', 20))}", alias, script],
                            capture_output=True, text=True, timeout=float(self.cfg.get("probe_timeout_s", 90)),
                            check=False)
        except subprocess.TimeoutExpired:
            return island_status.Probe(node=island_status.PROBE_UNREACHABLE)
        return island_status.Probe.parse(out.stdout or "", out.returncode)

    def fence(self, m: Member) -> Any:
        return self.data_path().fence_shard(m.shard, by=m.ticket)

    def fence_incarnation(self, m: Member) -> Any:
        return self.data_path().fence_incarnation(m.shard, by=m.ticket)

    def clear_stale_handoff(self, m: Member, keep: Any) -> Any:
        clear = getattr(self.data_path(), "clear_stale_handoff", None)  # the syncer path keeps no handoff record
        return None if clear is None else clear(m.shard, keep=keep)

    def exit_evidence(self, m: Member) -> Any:
        read = getattr(self.data_path(), "exit_evidence", None)  # the syncer path has no such record
        return None if read is None else read(m.shard)

    def data_path(self) -> Any:
        """The work contract's data path of this pool (:func:`pool_data_path`)."""
        return pool_data_path(self.cfg, self._fence_store,
                              lambda: self._dispense_ledger() if pool_dispense(self.cfg) is not None else None)

    def _fence_store(self) -> Any:
        env = elastic.read_env_file(self.cfg["r2_env_file"])
        bucket = env.get(self.cfg.get("bucket_env", "R2_BUCKET"))
        if not bucket:
            raise NcError(f"no {self.cfg.get('bucket_env', 'R2_BUCKET')} in the store env file")
        uri = f"s3://{bucket}/runs/{self.cfg['run_prefix'].strip('/')}"
        return elastic.s3_fence_store(uri, self.cfg["r2_env_file"])

    # -- scale down in "drain" mode (the store dispenser's drain request, store_dispense)
    def _dispense_ledger(self) -> tuple[Any, str]:
        from yeto.rl.engine import store_dispense  # noqa: PLC0415

        dispense = pool_dispense(self.cfg)
        if dispense is None:
            raise NcError("no store dispenser (prompts.dispense.backend store) in this pool: nothing to drain")
        if dispense.get("store"):
            store = elastic.s3_fence_store(str(dispense["store"]), self.cfg["r2_env_file"])
        else:
            store = self._fence_store()
        return store.objects, str(dispense.get("namespace") or store_dispense.DEFAULT_NAMESPACE)

    def request_drain(self, m: Member) -> None:
        if self.cfg.get("data_path", "r2") == "r2" and pool_dispense(self.cfg) is None:
            raise NcError("no store dispenser (prompts.dispense.backend store) in this pool: nothing to drain")
        m.drain_seq = self.data_path().request_drain(m.shard, seq=f"{m.ticket}:{time.time():.3f}")

    def drained(self, m: Member) -> bool:
        if m.drain_seq is None:
            return False
        return bool(self.data_path().drain_answered(m.shard, m.drain_seq))

    def cancel_drain(self, m: Member) -> None:
        self.data_path().cancel_drain(m.shard)
        m.drain_seq = None

    def reassert_drain(self, m) -> None:
        again = getattr(self.data_path(), "reassert_drain", None)
        if again is not None:  # a data path whose drain state lives only in its server (syncer)
            again(m.shard)

    def drain(self, m: Member) -> None:
        """Best-effort sampler stop (a lost node does not answer), alias removed,
        then the guarded delete of the ticket and its ``ended`` confirmation."""
        if m.shard and leases_path().exists() and m.ticket in json.loads(leases_path().read_text()):
            name = ssh_sampler.container_name(m.shard)
            stop = (self.sampler_route.stop_command(m.shard, grace_s=30) if self.sampler_route is not None else
                    f"sudo docker stop -t 30 {name} >/dev/null 2>&1; sudo docker logs --tail 500 {name} "
                    f"> {ssh_sampler.NVME}/logs/{name}.last.log 2>&1; sudo docker rm -f {name} "
                    ">/dev/null 2>&1; echo stopped")
            try:
                self._run(["ssh", "-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                           node_pool.host_alias(m.ticket), stop],
                          capture_output=True, text=True, timeout=120, check=False)
            except subprocess.TimeoutExpired:
                _log(f"drain {m.ticket}: sampler stop timed out (node unreachable); deleting the ticket")
        node_pool.remove_ssh_alias(m.ticket)
        result = self.provider.release(m.account, m.ticket)
        _log(f"ticket {m.ticket}: {result}")

    # -- pieces
    def _pull(self, lease: NodeLease, pull: dict[str, Any]) -> None:
        route, alias = self.sampler_route, node_pool.host_alias(lease.ticket)
        if route is not None:  # dest: the sandbox path (default the spec's /models/kimi-k3.1-pro)
            route.pull(alias, r2_env_file=self.cfg["r2_env_file"], src_key=pull["src_key"],
                       **({"dest": pull["dest"]} if pull.get("dest") else {}), include=pull.get("include"),
                       max_mb_s=int(pull.get("max_mb_s", 2500)), run=self._run)
        else:
            ssh_sampler.pull_model(lease, r2_env_file=self.cfg["r2_env_file"], src_key=pull["src_key"],
                                   dest=pull.get("dest", ssh_sampler.MODEL_HOST), include=pull.get("include"),
                                   max_mb_s=int(pull.get("max_mb_s", 2500)), run=self._run)
        deadline = time.monotonic() + float(pull.get("timeout_s", 3600))
        while time.monotonic() < deadline:
            status = (route.pull_status(alias, run=self._run) if route is not None
                      else ssh_sampler.pull_status(lease, run=self._run))
            if "YETO_NC_PULL done" in status:
                return
            if pull_ended_without_done(status):
                raise NcError(f"model pull failed on {lease.ticket}: {status[-300:]}")
            time.sleep(float(pull.get("poll_s", 30)))
        raise NcError(f"model pull on {lease.ticket} not done in {pull.get('timeout_s', 3600)}s")

    def _hooks(self, lease: NodeLease, m: Member) -> None:
        """``node_hooks``: commands that rebuild what lives on a node (reverse
        tunnels to sandbox hosts, the records proxy's public base, host aliases),
        run on every new node and before every in-place restart. Each must be
        idempotent and exit 0. ``where``: ``node`` (over the alias) or ``local``."""
        fields = {"alias": node_pool.host_alias(lease.ticket), "ip": lease.ip, "ticket": lease.ticket,
                  "account": lease.account, "user": lease.user, "shard": m.shard, "run": self.run_id}
        for hook in self.cfg.get("node_hooks") or ():
            command = str(hook["command"]).format(**fields)
            timeout = float(hook.get("timeout_s", 600))
            if hook.get("where", "node") == "node":
                out = ssh_sampler.ssh(fields["alias"], command, run=self._run, timeout=timeout)
            else:
                out = self._run(command, shell=True, capture_output=True, text=True, timeout=timeout, check=False)
            if out.returncode != 0:
                raise NcError(f"node hook {hook.get('name', command[:40])!r} exit {out.returncode}: "
                              f"{(out.stderr or out.stdout or '')[-300:]}")
            _log(f"node hook {hook.get('name', '')} ok on {lease.ticket}")

    def _spec(self, m: Member, shards: list[str]) -> dict[str, Any]:
        return render_spec(self.cfg, m, shards)

    def _start_sampler(self, lease: NodeLease, m: Member, shards: list[str]) -> None:
        if self.sampler_route is not None:
            self.sampler_route.start(node_pool.host_alias(lease.ticket), spec=self._spec(m, shards), shard=m.shard,
                                     run_prefix=self.cfg["run_prefix"], r2_env_file=self.cfg["r2_env_file"],
                                     protected=frozenset(self.cfg.get("protected_prefixes", ())),
                                     agentic_env_file=self.cfg.get("agentic_env_file"),
                                     spec_dir=Path(self.cfg["spec"]).expanduser().parent, run=self._run)
            return
        ssh_sampler.start_sampler(lease, spec=self._spec(m, shards), shard=m.shard, run_prefix=self.cfg["run_prefix"],
                                  r2_env_file=self.cfg["r2_env_file"], image=self.cfg.get("image", ssh_sampler.IMAGE),
                                  protected=frozenset(self.cfg.get("protected_prefixes", ())),
                                  state_dir=tunnel.state_dir(), agentic_env_file=self.cfg.get("agentic_env_file"),
                                  run=self._run)

    def _tape_glob(self, shard: str) -> str:
        """The node path glob of the shard's event tapes (``event_tape`` of the spec,
        one file per incarnation: ``<stem>.<incarnation>.jsonl``)."""
        try:
            tape = json.loads(Path(self.cfg["spec"]).read_text()).get("event_tape")
        except (OSError, ValueError, KeyError):
            tape = None
        root = "" if self.sampler_route is not None else ssh_sampler.NVME  # a process sees /work itself
        if not tape:
            return f"{root}/work/sampler/events-{shard}*.jsonl"
        tape = str(tape).replace("__RUN_PREFIX__", self.cfg["run_prefix"].strip("/"))
        host = tape.replace("/work/", f"{root}/work/", 1) if tape.startswith("/work/") else tape
        stem = host[:-len(".jsonl")] if host.endswith(".jsonl") else host
        return f"{stem}*.jsonl"


SYNCER_KEYS = frozenset({"address", "key_env", "timeout_s"})


def syncer_address(value: Any) -> tuple[str, int]:
    """``"host:port"`` or ``[host, port]`` -> ``(host, port)``."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return str(value[0]), int(value[1])
    host, sep, port = str(value or "").rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(f"pool config: syncer.address {value!r} must be 'host:port'")
    return host, int(port)


def pool_data_path(cfg: dict[str, Any], records: Any, dispense: Any) -> Any:
    """The work contract's data path of a pool (``yeto.launch.pool.data_path``).

    ``data_path: r2`` (default): :class:`R2DataPath` over the run store (``records``,
    ``dispense``: values or callables). ``data_path: syncer``: :class:`SyncerDataPath`
    on the pool config's ``syncer`` section (``address`` ``"host:port"``, ``key_env``
    the environment variable holding the island HMAC key, default
    ``YETO_ISLAND_HMAC_KEY``; unset: the syncer accepts only its own host)."""
    from ..pool.data_path import R2DataPath, SyncerDataPath  # noqa: PLC0415

    if cfg.get("data_path", "r2") == "r2":
        return R2DataPath(records, dispense)
    section = dict(cfg.get("syncer") or {})
    if "address" not in section:
        raise NcError("pool config: data_path syncer needs syncer.address (the syncer's host:port)")
    key = os.environ.get(str(section.get("key_env") or "YETO_ISLAND_HMAC_KEY"), "")
    return SyncerDataPath(syncer_address(section["address"]), key=key.encode() if key else None,
                          timeout_s=float(section.get("timeout_s", 10.0)))


def render_spec(cfg: dict[str, Any], m: Member, shards: list[str]) -> dict[str, Any]:
    """The sampler spec of pool member ``m`` (the pool config's spec file and knobs)."""
    text = Path(cfg["spec"]).read_text().replace("__RUN_PREFIX__", cfg["run_prefix"].strip("/"))
    spec = tasks.apply_sampler_overrides(tasks.sampler_spec(json.loads(text), sampler_id=m.shard, shards=shards),
                                         data_slot=m.data_slot, continues_shard=m.replaces_shard,
                                         **_overrides(cfg.get("sampler") or {}))
    if m.replaces_shard and m.predecessor_lost:
        spec["handoff_wait_s"] = float(cfg.get("elastic", {}).get("lost_handoff_wait_s", 60.0))
    if m.replaces_shard and getattr(m, "predecessor_retired", False):
        spec["predecessor_retired"] = True  # the runtime may sweep the predecessor's orphans at once
    if cfg.get("data_path", "r2") == "syncer":
        spec["data_path"] = "syncer"  # the one sampler entry picks its data path from the spec
        section = cfg.get("syncer") or {}
        if section.get("address"):
            spec.setdefault("syncer", str(section["address"]))
        dispense = (spec.get("prompts") or {}).get("dispense")
        if isinstance(dispense, dict) and str(dispense.get("backend") or "tcp") == "tcp":
            # the syncer's task dispenser: each pool member claims as its own island, the
            # holder id the pool retires / drains it by (SyncerDataPath); one id for every
            # member of the base spec would make each new member supersede the others
            from yeto.rl.engine.task_dispense import sampler_holder_id  # noqa: PLC0415

            dispense = dict(dispense, learner_id=sampler_holder_id(m.shard))
            if spec.get("syncer"):
                dispense.setdefault("address", str(spec["syncer"]))
            spec["prompts"] = dict(spec["prompts"], dispense=dispense)
    backlog = cfg.get("backlog")
    if backlog:
        spec["max_backlog_groups"] = tasks.backlog_groups(
            groups_per_round=int(spec["prompts"]["groups_per_round"]), samplers=len(spec["buffer_shards"]),
            **backlog)
    return spec


def nc_provider(cfg: dict[str, Any], run: str) -> NCProvider:
    """The National Compute provider of a pool: this tool's ledger, lease key and relay."""
    def event(name: str, **detail: Any) -> None:  # the pool's events file and its event_command hook
        hook = island_status.command_hook(cfg["event_command"]) if cfg.get("event_command") else None
        island_status.EventLog(path=pool_events_path(cfg), run=run, log=_log, hook=hook)(name, **detail)

    return NCProvider([int(a) for a in cfg["accounts"]], run, ledger=ledger(), allowed=allowed_accounts(),
                      model=machine_type(cfg), pubkey=node_pool.ensure_key, before_order=ensure_relay, event=event)


def run_id(cfg: dict[str, Any]) -> str:
    """The run tag of a pool (ledger rows, events): ``run``, else ``shard_base``."""
    value = str(cfg.get("run") or cfg["shard_base"])
    if not value or any(c.isspace() for c in value):
        raise ValueError(f"pool run tag {value!r} must be a non-empty word")
    return value


POOL_KEYS = frozenset({"shard_limit", "target", "accounts", "slots", "shard_base", "run", "route", "replace_lead_s", "end_margin_s",
                       "stop_below_ucr", "min_balance_to_order_ucr", "data_slots", "elastic", "event_command",
                       "machine_type", "model", "spec", "run_prefix", "r2_env_file", "bucket_env", "image",
                       "protected_prefixes", "agentic_env_file", "sampler", "backlog", "model_pull", "node_hooks",
                       "probe_timeout_s", "probe_connect_timeout_s", "provider", "model_src", "model_dest",
                       "workdir", "comment", "sampler_route", "data_path", "syncer", "protected_accounts"})
DATA_PATHS = ("r2", "syncer")  # buffer shards + releases in the run store | syncer broadcast + sample relay
REQUIRED_KEYS = ("target", "shard_base")  # and "accounts" (or "slots" off National Compute)


def slot_list(cfg: dict[str, Any]) -> list[int]:
    """The pool's provider slots: ``accounts`` (National Compute accounts), or for
    Modal / Daytona / sky ``slots`` (``accounts`` stays an alias there)."""
    from yeto.launch.pool.provider import is_national  # noqa: PLC0415

    if "slots" in cfg and "accounts" in cfg:
        raise ValueError("pool config: set 'slots' or 'accounts', not both")
    if "slots" in cfg:
        if is_national(cfg):
            raise ValueError("pool config: National Compute pools list NC accounts in 'accounts', not 'slots'")
        return [int(a) for a in cfg["slots"]]
    if "accounts" not in cfg:
        raise ValueError("pool config: missing required keys ['accounts']"
                         + ("" if is_national(cfg) else " (or 'slots')"))
    return [int(a) for a in cfg["accounts"]]


def slot_name(cfg: dict[str, Any]) -> str:
    """``account`` for a National Compute pool, else ``slot``."""
    from yeto.launch.pool.provider import is_national  # noqa: PLC0415

    return "account" if is_national(cfg) else "slot"
# the SSH route renders the sampler spec and starts it with the store credentials
REQUIRED_SSH_KEYS = ("spec", "run_prefix", "r2_env_file")
MODEL_PULL_KEYS = frozenset({"src_key", "dest", "include", "max_mb_s", "timeout_s", "poll_s"})
NODE_HOOK_KEYS = frozenset({"name", "command", "where", "timeout_s"})
BACKLOG_KEYS = frozenset({"synced", "trainers", "fragments", "local_rounds", "pipeline", "explicit_rounds"})
DEFAULT_MACHINE_TYPE = "mi355x"


def machine_type(cfg: dict[str, Any]) -> str:
    """The National Compute machine type a pool orders (``machine_type``; the
    deprecated ``model`` key is read when it is the only one set)."""
    return str(cfg.get("machine_type") or cfg.get("model") or DEFAULT_MACHINE_TYPE)


def pool_config_warnings(cfg: dict[str, Any]) -> list[str]:
    """Deprecation notes of a pool config (printed by ``pool``, ``pool --dry-run``)."""
    notes = []
    if "model" in cfg and "machine_type" not in cfg:
        notes.append(f"pool config: key 'model' is deprecated (it is the National Compute machine type, "
                     f"not a Hugging Face model); rename it to 'machine_type': {cfg['model']!r}")
    try:
        spec = json.loads(Path(cfg["spec"]).expanduser().read_text()) if cfg.get("spec") else {}
    except (OSError, ValueError):
        spec = {}
    if spec.get("incarnation_lease", True) is False:
        notes.append("sampler spec: incarnation_lease is false: a fenced or replaced node's task claims are "
                     "not released at once (the other samplers wait out the dispense lease, lease_s) and its "
                     "sampler is not fenced; remove the key (default on) for an elastic pool")
    return notes


def pool_has_dispenser(cfg: dict[str, Any]) -> bool:
    """The pool's samplers take their tasks from a dispenser (no task goes to two
    of them, so the pool may add or remove samplers): the store dispenser
    (:func:`pool_dispense`), or on ``data_path: syncer`` the syncer's task
    dispenser (``prompts.dispense`` of the spec with the default ``tcp`` backend)."""
    if pool_dispense(cfg) is not None:
        return True
    if cfg.get("data_path", "r2") != "syncer":
        return False
    try:
        spec = json.loads(Path(os.path.expanduser(str(cfg["spec"]))).read_text()) if cfg.get("spec") else {}
    except (OSError, ValueError):
        return False
    found = (spec.get("prompts") or {}).get("dispense") if isinstance(spec, dict) else None
    return isinstance(found, dict) and str(found.get("backend") or "tcp") == "tcp"


def pool_dispense(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The store dispenser config of a pool's samplers (``sampler.dispense`` of the pool
    config, else ``prompts.dispense`` of the spec with ``backend: store``); None: none
    (each sampler reads its data slot, or every prompt)."""
    own = (cfg.get("sampler") or {}).get("dispense")
    if isinstance(own, dict) and own:
        return dict(own)
    try:
        spec = json.loads(Path(os.path.expanduser(str(cfg["spec"]))).read_text()) if cfg.get("spec") else {}
    except (OSError, ValueError):
        return None
    found = (spec.get("prompts") or {}).get("dispense") if isinstance(spec, dict) else None
    if isinstance(found, dict) and found.get("backend", "store") == "store" and found.get("backend"):
        return dict(found)
    return None


def _unknown(section: str, doc: Any, known: frozenset[str]) -> None:
    if not isinstance(doc, dict):
        raise TypeError(f"pool config: {section} must be an object")
    unknown = set(doc) - known
    if unknown:
        raise ValueError(f"pool config: unknown keys {sorted(unknown)} in {section} (known: {sorted(known)})")


def pool_config(cfg: dict[str, Any], *, allowed: set[int] | None = None) -> PoolConfig:
    """The :class:`PoolConfig` of a pool config file. Refuses unknown keys (also in
    ``sampler``, ``model_pull``, ``node_hooks``, ``backlog``, ``elastic``), missing
    required keys, accounts outside ``YETO_NC_ACCOUNTS`` (``allowed``), a ``target``
    above the number of accounts (one node per account) and ``model`` together with
    ``machine_type``."""
    unknown = set(cfg) - POOL_KEYS
    if unknown:
        raise ValueError(f"pool config: unknown keys {sorted(unknown)} (known: {sorted(POOL_KEYS)})")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if "accounts" not in cfg and "slots" not in cfg:
        missing.insert(1, "accounts")
    if cfg.get("route", "ssh") == "ssh":
        missing += [k for k in REQUIRED_SSH_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(f"pool config: missing required keys {missing}")
    if "model" in cfg and "machine_type" in cfg:
        raise ValueError("pool config: set 'machine_type' only ('model' is its deprecated name)")
    if "/" in machine_type(cfg):
        raise ValueError(f"pool config: machine type {machine_type(cfg)!r} looks like a Hugging Face model name; "
                         f"'machine_type' (formerly 'model') is the National Compute machine type, e.g. "
                         f"{DEFAULT_MACHINE_TYPE!r} (the served model is in the sampler spec)")
    accounts = slot_list(cfg)
    label = slot_name(cfg)
    if not accounts or len(set(accounts)) != len(accounts):
        raise ValueError(f"pool config: {label}s must be a non-empty list without repeats")
    if allowed is not None and not set(accounts) <= allowed:
        raise ValueError(f"pool config: accounts {sorted(set(accounts) - allowed)} are not in YETO_NC_ACCOUNTS")
    target = int(cfg["target"])
    if target < 1:
        raise ValueError("pool config: target must be >= 1")
    if target > len(accounts):
        if label == "account":
            raise ValueError(f"pool config: target {target} > {len(accounts)} accounts; a National Compute "
                             f"account holds one node at a time, so the pool can run at most "
                             f"{len(accounts)} samplers")
        raise ValueError(f"pool config: target {target} > {len(accounts)} slots; a slot holds one machine "
                         f"at a time, so the pool can run at most {len(accounts)} samplers")
    slots = cfg.get("data_slots")
    if slots is not None and int(slots) < target:
        raise ValueError(f"pool config: data_slots {slots} < target {target} (each member holds one slot)")
    _overrides(cfg.get("sampler") or {})
    if cfg.get("model_pull"):
        _unknown("model_pull", cfg["model_pull"], MODEL_PULL_KEYS)
        if "src_key" not in cfg["model_pull"]:
            raise ValueError("pool config: model_pull needs src_key")
    for i, hook in enumerate(cfg.get("node_hooks") or ()):
        _unknown(f"node_hooks[{i}]", hook, NODE_HOOK_KEYS)
        if "command" not in hook:
            raise ValueError(f"pool config: node_hooks[{i}] needs command")
        if hook.get("where", "node") not in ("node", "local"):
            raise ValueError(f"pool config: node_hooks[{i}].where must be node or local")
    if cfg.get("backlog"):
        _unknown("backlog", cfg["backlog"], BACKLOG_KEYS)
    if cfg.get("data_path", "r2") not in DATA_PATHS:
        raise ValueError(f"pool config: data_path {cfg.get('data_path')!r} is not one of {DATA_PATHS}")
    if cfg.get("syncer") is not None:
        _unknown("syncer", cfg["syncer"], SYNCER_KEYS)
    if cfg.get("data_path", "r2") == "syncer":
        syncer_address((cfg.get("syncer") or {}).get("address"))
    if cfg.get("route", "ssh") not in ("ssh", "sky"):
        raise ValueError(f"pool config: route {cfg.get('route')!r} is not ssh or sky")
    run_id(cfg)
    lead, margin = (_seconds(cfg, "replace_lead_s", 3600), _seconds(cfg, "end_margin_s", 600))
    price = pool_price(cfg)[0]
    if price is not None and not (math.isfinite(price) and price >= 0):
        raise ValueError(f"pool config: provider price_usd_h {price!r} must be a finite number >= 0")
    return PoolConfig(target=target, accounts=accounts, shard_base=cfg["shard_base"],
                      shard_limit=pool_shard_limit(cfg),
                      replace_lead_s=lead, end_margin_s=margin,
                      stop_below_ucr=int(cfg.get("stop_below_ucr", 20_000_000)),
                      min_balance_to_order_ucr=cfg.get("min_balance_to_order_ucr"),
                      data_slots=slots, elastic=island_status.ElasticConfig.from_dict(cfg.get("elastic")),
                      hourly_cost=pool_hourly_ucr(cfg), slot_name=label)


def _seconds(cfg: dict[str, Any], key: str, default: float) -> float:
    """A pool duration: a finite number >= 0 (not a bool, a string or nan)."""
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"pool config: {key} must be a finite number of seconds >= 0, not {value!r}")
    return float(value)


def pool_price(cfg: dict[str, Any]) -> tuple[float | None, str]:
    """One node hour of a pool as its provider config states it, and the currency:
    sky ``price_usd_h`` (or the first offer's), modal / daytona ``price_usd_h`` in USD;
    None for National Compute (the MI355X rate in ucr applies)."""
    from ..pool import provider as providers  # noqa: PLC0415

    kind = providers.kind_of(cfg) or "factory"
    entry = providers.registered(kind)
    currency = entry.caps.currency if entry is not None else "ucr"
    section = cfg.get("provider") or {}
    price = section.get("price_usd_h")
    if price is None and kind == "sky":
        from yeto.launch.rollout.offers import offers_from_config  # noqa: PLC0415

        try:
            price = offers_from_config(section)[0].price_usd_h
        except ValueError:
            price = None
    if price is None:
        return None, currency
    if isinstance(price, bool):
        raise ValueError(f"pool config: provider price_usd_h must be a number, not {price!r}")
    try:
        return float(price), currency
    except (TypeError, ValueError):
        raise ValueError(f"pool config: provider price_usd_h must be a number, not {price!r}") from None


def pool_hourly_ucr(cfg: dict[str, Any]) -> float:
    """One node hour of a pool in ucr-equivalents (the pool budget's unit; 0: the
    National Compute MI355X rate). A USD price converts at ``YETO_NC_USD_PER_UCR``."""
    from ..pool.money import to_ucr  # noqa: PLC0415

    price, currency = pool_price(cfg)
    return 0 if price is None else to_ucr(price, "usd" if currency == "ucr" else currency)


def pool_hourly_text(cfg: dict[str, Any], hourly_ucr: float) -> str:
    """One node hour for a message: credits for National Compute, USD otherwise."""
    from ..pool.money import to_usd  # noqa: PLC0415

    if pool_price(cfg)[0] is None:
        return f"{hourly_ucr:.0f} ucr ({hourly_ucr / 1e6:g} credits)"
    return f"${to_usd(hourly_ucr, 'ucr'):.2f}"


def pool_shard_limit(cfg: dict[str, Any]) -> int | None:
    """How many shard indexes (``<shard_base>-0`` ..) the run can take: ``shard_limit`` (the
    trainer's ``--rl-buffer-shards`` count) and the ports of the spec's reverse-tunnel range
    (``agentic.setup.tunnel.listen`` ``<ip>:<lo>-<hi>``: one port per shard index), the
    smaller one; None when neither bounds it. An order past it is refused before it is placed."""
    limits = [int(cfg["shard_limit"])] if cfg.get("shard_limit") is not None else []
    try:
        spec = json.loads(Path(os.path.expanduser(str(cfg["spec"]))).read_text()) if cfg.get("spec") else {}
        listen = str((((spec.get("agentic") or {}).get("setup") or {}).get("tunnel") or {}).get("listen") or "")
    except (OSError, ValueError, AttributeError):
        listen = ""
    ports = listen.rpartition(":")[2]
    lo, sep, hi = ports.partition("-")
    if sep and lo.isdigit() and hi.isdigit() and int(hi) >= int(lo):
        limits.append(int(hi) - int(lo) + 1)
    return min(limits) if limits else None


def pool_state_path(cfg: dict[str, Any]) -> Path:
    return tunnel.state_dir() / f"pool-{cfg['shard_base']}.json"


def pool_events_path(cfg: dict[str, Any]) -> Path:
    return tunnel.state_dir() / f"events-{run_id(cfg)}.jsonl"


def pool_ops(cfg: dict[str, Any]):
    """The pool ops of a pool config: a registered provider kind's own ``pool_ops``
    (a machine that is not an SSH node, e.g. a Modal FunctionCall), else the SSH +
    docker route (``route: ssh``) or the sky route."""
    from ..pool import provider as providers  # noqa: PLC0415

    kind = providers.kind_of(cfg)
    entry = providers.registered(kind) if kind is not None else None
    if entry is not None and entry.pool_ops is not None:
        return providers.resolve(entry.pool_ops)(cfg)
    route = cfg.get("route", "ssh")
    if route == "ssh":
        return NcSshOps(cfg)
    if route == "sky":
        return NcSkyOps(cfg)
    raise ValueError(f"pool config: route {route!r} is not ssh or sky")


def agentic_setup_cmd(args) -> int:
    """``agentic-setup``: ``--dry-run`` (local), the setup alone on ``--ticket``'s node,
    or its tunnel events. ``ssh-sampler start`` runs the same setup by itself."""
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text().replace("__RUN_PREFIX__", "run"))
    lines, problems = agentic_setup.dry_run_report(spec, base_dir=spec_path.parent)
    if args.dry_run or not args.ticket:
        if not lines and not problems:
            print("agentic-setup: the spec has no agentic.setup section")
        for line in lines:
            print(line)
        for problem in problems:
            print(f"problem: {problem}")
        print("agentic-setup --dry-run: " + ("ok" if not problems else f"{len(problems)} problem(s)"))
        return 0 if not problems else 1
    if problems:
        raise SystemExit("agentic-setup: " + "; ".join(problems))
    setup = agentic_setup.parse(spec, base_dir=spec_path.parent)
    lease = load_lease(args.ticket)
    alias = node_pool.host_alias(lease.ticket)
    node = agentic_setup.SshNode(alias, args.image, ssh_sampler.NVME, ssh_sampler.upload_code(alias),
                                 ssh_sampler.PYTHON, lease.user, prelude=ssh_sampler.SSH_HOME_LINK)
    if args.tunnel_events:
        for event in agentic_setup.tunnel_events(node, setup)[0]:
            print(json.dumps(event, sort_keys=True))
        return 0
    report = agentic_setup.Provisioner(setup, node, emit=lambda r: print(json.dumps(r, sort_keys=True)),
                                       store_env_file=f"/home/{lease.user}/.yeto-nc-r2.env").run()
    print(json.dumps(report))
    return 0


# -- entry -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m yeto.launch.nc",
                                description="National Compute nodes for yeto samplers (docs/NC_POOL.md). A lease "
                                "is NOT ended at lease_until: a node stays billed until it is deleted.")
    sub = p.add_subparsers(dest="cmd", required=True)
    px = sub.add_parser("proxy", help="ssh ProxyCommand through the workspace relay")
    px.add_argument("--account", type=int, required=True)
    px.add_argument("--relay-port", type=int, default=tunnel.RELAY_PORT)
    px.add_argument("host")
    px.add_argument("port", type=int)
    for name, text in (("preflight", "tools, account free, balance (read-only)"),
                       ("relay", "start the workspace relay of an account")):
        sp = sub.add_parser(name, help=text)
        sp.add_argument("--account", type=int, required=True)
    u = sub.add_parser("up", help="order one billed node and set it up (needs --yes; release it with down)")
    u.add_argument("--account", type=int, required=True)
    u.add_argument("--yes", action="store_true")
    u.add_argument("--machine-type", "--model", dest="machine_type", default=DEFAULT_MACHINE_TYPE,
                   help="National Compute machine type (--model is its deprecated name)")
    u.add_argument("--route", choices=("ssh", "sky"), default="ssh")
    pr = sub.add_parser("probe", help="card check, image pull, engine version, spec preflight on a node")
    pr.add_argument("--ticket", required=True)
    pr.add_argument("--spec")
    pr.add_argument("--image", default=ssh_sampler.IMAGE)
    sp = sub.add_parser("ssh-pull", help="rate-limited model pull from the bucket onto a node")
    sp.add_argument("action", choices=("start", "status"))
    sp.add_argument("--ticket", required=True)
    sp.add_argument("--r2-env-file")
    sp.add_argument("--src-key", default="models/Kimi-K3.1-Pro", help="model directory under the bucket")
    sp.add_argument("--dest", default="/mnt/nvme/models/Kimi-K3.1-Pro")
    sp.add_argument("--include")
    sp.add_argument("--max-mb-s", type=int, default=2500)
    ss = sub.add_parser("ssh-sampler", help="start / stop / status of a sampler container on a node")
    ss.add_argument("action", choices=("start", "stop", "status"))
    ss.add_argument("--ticket", required=True)
    ss.add_argument("--shard", required=True)
    ss.add_argument("--spec")
    ss.add_argument("--shards", default="")
    ss.add_argument("--run-prefix")
    ss.add_argument("--r2-env-file")
    ss.add_argument("--protect", default="", help="comma list of run prefixes that must not be written")
    ss.add_argument("--image", default=ssh_sampler.IMAGE)
    ss.add_argument("--synced", action="store_true", help="full-parameter synced group (default: no-sync)")
    ss.add_argument("--trainers", type=int, default=1, help="trainers the samplers feed (synced backlog sizing)")
    ss.add_argument("--fragments", type=int, default=1)
    ss.add_argument("--local-rounds", type=int, default=1, help="--local-rl-rounds-per-sync")
    ss.add_argument("--pipeline", type=int, default=0)
    ss.add_argument("--backlog-rounds", type=int, help="explicit cap in rounds (refused below the minimum)")
    ss.add_argument("--prompts", help="local prompts JSONL, uploaded to /work/sampler/prompts.jsonl")
    ss.add_argument("--agentic-env", help="local env file of an agentic sampler (HARBOR_*, DOCKER_HOST, "
                    "YETO_RECORDS_PROXY_*), a second --env-file of the container")
    _add_sampler_override_args(ss)
    ag = sub.add_parser("agentic-setup", help="prepare an agentic sampler node from the spec's agentic.setup")
    ag.add_argument("--spec", required=True)
    ag.add_argument("--dry-run", action="store_true", help="validate and print the plan; no SSH")
    ag.add_argument("--ticket")
    ag.add_argument("--image", default=ssh_sampler.IMAGE)
    ag.add_argument("--tunnel-events", action="store_true", help="print the node's tunnel up/down events")
    a = sub.add_parser("attach", help="sky route: host bootstrap and sky SSH node pool on a node")
    a.add_argument("--ticket", required=True)
    h = sub.add_parser("hello", help="sky route: a hello task on the node's pool")
    h.add_argument("--ticket", required=True)
    pl = sub.add_parser("pull", help="sky route: rate-limited model pull as a sky task")
    pl.add_argument("--ticket", required=True)
    pl.add_argument("--src", required=True)
    pl.add_argument("--dest", required=True)
    pl.add_argument("--include")
    pl.add_argument("--max-mb-s", type=int, default=tasks.MAX_PULL_MB_S)
    pl.add_argument("--r2-env-file", required=True)
    sm = sub.add_parser("sampler", help="sky route: engine sampler as a sky task")
    sm.add_argument("--ticket", required=True)
    sm.add_argument("--spec", required=True)
    sm.add_argument("--sampler-id", required=True)
    sm.add_argument("--shards", default="")
    sm.add_argument("--run-prefix", required=True)
    sm.add_argument("--workdir", default=str(Path(__file__).resolve().parents[3]))
    sm.add_argument("--r2-env-file", required=True)
    _add_sampler_override_args(sm)
    d = sub.add_parser("down", help="release one node this tool created (by ticket id, confirmed ended)")
    d.add_argument("--account", type=int, required=True)
    d.add_argument("--ticket", required=True)
    d.add_argument("--drain", dest="mode", action="store_const", const="drain", default="fence",
                   help="a pool member: let its trials in flight finish first (default: fence it now)")
    d.add_argument("--direct", action="store_true",
                   help="delete the ticket here even when a running pool owns it (the pool then sees a manual "
                   "release: it lowers its target and orders no replacement)")
    dt = sub.add_parser("delete-ticket", help="release one ticket this pool's run ordered (dry run unless --yes)",
                        description="Delete exactly one ticket, named by its id, that the ledger rows of the "
                        "pool's run or its pool state record. Refused for any other ticket, an account outside "
                        "the pool config, a protected account (2 and 3 always, plus protected_accounts) and while "
                        "the pool loop runs. No patterns, lists or --all.")
    dt.add_argument("--config", required=True, help="the pool config JSON of the run that ordered the ticket")
    dt.add_argument("--ticket", required=True, help="one exact ticket id")
    dt.add_argument("--yes", action="store_true", help="send the delete (default: print what would be deleted)")
    ad = sub.add_parser("adopt", help="record a ticket an unrecorded order of this tool placed (an operator's call)",
                        description="After an order call that failed, the account may show an open ticket that "
                        "is in no ledger (event order_unrecorded). Check in the National Compute console that this "
                        "tool's order placed it and no one else did, then adopt it: the ticket is recorded in the "
                        "ledger (adopted_by=operator) and this tool may use and release it (pool, down, pool-stop).")
    ad.add_argument("--account", type=int, required=True)
    ad.add_argument("--ticket", required=True)
    ad.add_argument("--yes", action="store_true", help="I checked that this ticket is this tool's order")
    ch = sub.add_parser("clear-halt", help="lift the halt of an account (a delete hit, or may have hit, a foreign ticket)")
    ch.add_argument("--account", type=int, required=True)
    ch.add_argument("--yes", action="store_true", help="I checked the account and told the ticket's owner")
    po = sub.add_parser("pool", help="the foreground sampler pool loop (docs/NC_POOL.md)",
                        description="Run the lease-aware sampler pool in the foreground. Keep it up for the "
                        "whole run: National Compute does not end a lease at lease_until (it stays billed).")
    po.add_argument("--config", required=True, help="the pool config JSON (docs/NC_POOL.md)")
    po.add_argument("--interval", type=float, default=60, help="seconds between control ticks (default 60)")
    po.add_argument("--dry-run", action="store_true",
                    help="validate the config, print the orders and the spend per hour; order nothing")
    pst = sub.add_parser("pool-stop", help="release every node the pool created",
                         description="Ask a running pool to drain every node, or (no pool process alive) "
                         "release the nodes of its state file and the run's open tickets in the ledger. "
                         "Only tickets this tool created for this run are deleted, by id.")
    pst.add_argument("--config", required=True, help="the pool config JSON the pool runs with")
    pst.add_argument("--wait", type=float, default=900, help="seconds to wait for a running pool (default 900)")
    prt = sub.add_parser("pool-retry", help="after a configuration error: restart the sampler on the same node(s)",
                         description="Check the fixed pool config and sampler spec, then have the pool loop "
                         "restart the sampler in place on every node halted by a configuration error.")
    prt.add_argument("--config", required=True, help="the fixed pool config JSON")
    psd = sub.add_parser("pool-state-downgrade", help="before a rollback of yeto: write the pool state file in "
                         "the older format",
                         description="Write the pool's state file in the format the yeto before the fence retry "
                         "reads (it fails on the newer file, also in pool-stop). Run it with the newer yeto, "
                         "with the pool loop stopped. Refused while a member is `fencing`.")
    psd.add_argument("--config", required=True, help="the pool config JSON the pool runs with")
    psd.add_argument("--drop-owed-fences", action="store_true",
                     help="give up shard fences still owed (the older code does not retry them)")
    psc = sub.add_parser("pool-scale", help="change a pool's sampler target while it runs (more or fewer nodes)",
                         description="Record the wanted target in the pool's state dir; the running loop "
                         "orders (scale up) or fences, drains and releases (scale down) at its next tick. "
                         "The target outlives a pool restart. Refused before any order when it exceeds the "
                         "accounts, max_nodes, max_orders, max_run_spend_ucr or shard_limit.")
    psc.add_argument("--config", required=True, help="the pool config JSON the pool runs with")
    scale_args.add_scale_args(psc, target_flag="--target")
    for name, what in (("pool-pause", "stop every automatic action of the pool on these nodes (manual "
                                      "maintenance); they keep billing and count toward the target"),
                       ("pool-resume", "end a pool-pause: the pool watches these nodes again")):
        pp = sub.add_parser(name, help=what)
        pp.add_argument("--config", required=True, help="the pool config JSON the pool runs with")
        pp.add_argument("--ticket", action="append", required=True, help="the node ticket (repeatable)")
        if name == "pool-pause":
            pp.add_argument("--hold-tasks", action="store_true",
                            help="also stop its sampler from claiming new tasks (store dispenser) while paused")
    ps = sub.add_parser("pool-status", help="the pool's state, billed nodes and last events (read-only)")
    ps.add_argument("--config", required=True, help="the pool config JSON")
    ps.add_argument("--events", type=int, default=20, help="last events to print (default 20)")
    args = p.parse_args(argv)

    if args.cmd == "proxy":
        return tunnel.proxy(args.account, args.host, args.port, relay_port=args.relay_port)
    if args.cmd == "preflight":
        print(json.dumps(preflight(args.account), indent=1))
        return 0
    if args.cmd == "relay":
        ensure_relay(args.account)
        print("relay up")
        return 0
    if args.cmd == "up":
        if not args.yes:
            raise SystemExit("up orders a billed node: pass --yes")
        lease = up(args.account, model=args.machine_type, route=args.route)
        print(json.dumps({"ticket": lease.ticket, "pool": node_pool.pool_name(lease.ticket),
                          "lease_until": lease.lease_until}))
        print(f"WARNING: National Compute does NOT end this lease at lease_until ({lease.lease_until}): the node "
              f"stays active and billed until it is deleted: python -m yeto.launch.nc down --account "
              f"{args.account} --ticket {lease.ticket}", file=sys.stderr)
        return 0
    if args.cmd == "probe":
        spec = json.loads(Path(args.spec).read_text()) if args.spec else None
        print(ssh_sampler.probe(load_lease(args.ticket), spec=spec, state_dir=tunnel.state_dir(), image=args.image))
        return 0
    if args.cmd == "ssh-pull":
        lease = load_lease(args.ticket)
        if args.action == "status":
            print(ssh_sampler.pull_status(lease))
        else:
            if not args.r2_env_file:
                raise SystemExit("start needs --r2-env-file")
            print(ssh_sampler.pull_model(lease, r2_env_file=args.r2_env_file, src_key=args.src_key, dest=args.dest,
                                         include=args.include, max_mb_s=args.max_mb_s))
        return 0
    if args.cmd == "ssh-sampler":
        lease = load_lease(args.ticket)
        if args.action == "stop":
            print(ssh_sampler.stop_sampler(lease, args.shard).stdout)
        elif args.action == "status":
            print(ssh_sampler.sampler_status(lease, args.shard))
        else:
            if not (args.spec and args.run_prefix and args.r2_env_file):
                raise SystemExit("start needs --spec, --run-prefix and --r2-env-file")
            text = Path(args.spec).read_text().replace("__RUN_PREFIX__", args.run_prefix.strip("/"))
            spec = tasks.apply_sampler_overrides(
                tasks.sampler_spec(json.loads(text), sampler_id=args.shard,
                                   shards=[s for s in args.shards.split(",") if s]), **_arg_overrides(args))
            shards = spec["buffer_shards"]
            spec["max_backlog_groups"] = tasks.backlog_groups(
                groups_per_round=int(spec["prompts"]["groups_per_round"]), synced=args.synced,
                trainers=args.trainers,
                samplers=len(shards), fragments=args.fragments, local_rounds=args.local_rounds,
                pipeline=args.pipeline, explicit_rounds=args.backlog_rounds)
            print(f"max_backlog_groups={spec['max_backlog_groups']} shards={shards}")
            if args.prompts:
                ssh_sampler.upload_file(node_pool.host_alias(lease.ticket), args.prompts,
                                        "/mnt/nvme/work/sampler/prompts.jsonl", mode="644")
                spec["prompts"] = dict(spec.get("prompts") or {}, path="/work/sampler/prompts.jsonl")
            print(ssh_sampler.start_sampler(lease, spec=spec, shard=args.shard, run_prefix=args.run_prefix,
                                            r2_env_file=args.r2_env_file, image=args.image,
                                            protected=frozenset(p for p in args.protect.split(",") if p),
                                            state_dir=tunnel.state_dir(), agentic_env_file=args.agentic_env,
                                            spec_dir=Path(args.spec).resolve().parent))
        return 0
    if args.cmd == "agentic-setup":
        return agentic_setup_cmd(args)
    if args.cmd == "attach":
        print(attach(load_lease(args.ticket)))
        return 0
    if args.cmd == "hello":
        pool = node_pool.pool_name(args.ticket)
        return launch(tasks.hello_task(pool), cluster_name("hello", args.ticket), down=True)
    if args.cmd == "pull":
        pool = node_pool.pool_name(args.ticket)
        doc = tasks.with_secrets(tasks.model_pull_task(pool, src=args.src, dest=args.dest, include=args.include,
                                                       max_mb_s=args.max_mb_s), args.r2_env_file)
        return launch(doc, cluster_name("pull", args.ticket), down=True)
    if args.cmd == "sampler":
        pool = node_pool.pool_name(args.ticket)
        spec_path = tunnel.state_dir() / f"spec-{args.sampler_id}.json"
        shards = [s for s in args.shards.split(",") if s]
        tasks.write_spec(tasks.apply_sampler_overrides(
            tasks.sampler_spec(json.loads(Path(args.spec).read_text()), sampler_id=args.sampler_id, shards=shards),
            **_arg_overrides(args)), spec_path)
        doc = tasks.with_secrets(tasks.sampler_task(pool, spec_path=str(spec_path), run_prefix=args.run_prefix,
                                                    workdir=args.workdir), args.r2_env_file)
        return launch(doc, cluster_name("sampler", args.ticket), down=False)
    if args.cmd == "adopt":
        if not args.yes:
            print(f"adopt records ticket {args.ticket} (account {args.account}) as this tool's own, to be released "
                  "by it. Check in the National Compute console that no one else placed it, then pass --yes.")
            return 2
        try:
            row = client(args.account).adopt(args.ticket)
        except NcError as error:
            print(f"adopt refused: {error}", file=sys.stderr)
            return 1
        print(json.dumps(row, indent=1, default=str))
        return 0
    if args.cmd == "clear-halt":
        halt = ledger().halted(args.account)
        if halt is None:
            print(f"account {args.account} is not halted")
            return 0
        print(json.dumps(halt, indent=1, default=str))
        try:  # what the account holds now, for the operator to compare with the console
            t = client(args.account).status().get("ticket")
            print(f"account {args.account} now: " + ("no order" if not t else
                  f"ticket {t.get('id')} state={t.get('state')} end_reason={t.get('end_reason')} "
                  f"lease_until={t.get('lease_until')}"))
        except Exception as error:  # noqa: BLE001 - shown, the decision stays with the operator
            print(f"account {args.account} now: the status could not be read ({error!r})"[:300])
        if ledger().halted(args.account) is None:
            print("the halt was lifted by this status read: the expected ticket reads ended")
            return 0
        if not args.yes:
            print("nothing changed. Check the account in the National Compute console, then pass --yes "
                  "to lift the halt")
            return 2
        ledger().clear_halt(args.account)
        print(f"account {args.account}: halt lifted")
        return 0
    if args.cmd == "down":
        if not args.direct:
            from . import pool_run  # noqa: PLC0415

            from ..pool.island_pool import PoolStateFormatError  # noqa: PLC0415

            try:
                rc = pool_run.down_member(args.ticket, mode=args.mode)
            except PoolStateFormatError as error:  # the ticket may be a pool's member: not deleted
                print(f"down: refused, nothing changed: {error} (--direct deletes the ticket without asking "
                      "any pool)", file=sys.stderr)
                return 2
            if rc is not None:
                return rc
        down(args.account, args.ticket)
        return 0
    if args.cmd == "delete-ticket":
        from ..pool.island_pool import PoolStateFormatError  # noqa: PLC0415
        from . import pool_run  # noqa: PLC0415

        try:
            return pool_run.delete_own_ticket(json.loads(Path(args.config).read_text()), args.config, args.ticket,
                                              yes=args.yes)
        except (PoolStateFormatError, ValueError, NcError) as error:
            print(f"delete-ticket: refused, nothing sent: {error}", file=sys.stderr)
            return 2
    if args.cmd in ("pool-pause", "pool-resume", "pool", "pool-stop", "pool-status", "pool-retry", "pool-scale",
                    "pool-state-downgrade"):
        from ..pool.island_pool import PoolStateFormatError  # noqa: PLC0415
        from . import pool_run  # noqa: PLC0415

        try:
            pool_run.check_state_file(json.loads(Path(args.config).read_text()))
            return _pool_command(args)
        except PoolStateFormatError as error:  # one line, no traceback: the first thing an operator reads
            print(f"{args.cmd}: refused, nothing changed: {error}", file=sys.stderr)
            return 2
    return 2


def _pool_command(args: Any) -> int:
    """The pool commands (their state file's format is checked by the caller)."""
    if args.cmd in ("pool-pause", "pool-resume"):
        from . import pool_run  # noqa: PLC0415

        cfg = json.loads(Path(args.config).read_text())
        return pool_run.pause_pool(cfg, args.config, args.ticket, resume=args.cmd == "pool-resume",
                                   hold_tasks=getattr(args, "hold_tasks", False))
    if args.cmd in ("pool", "pool-stop", "pool-status", "pool-retry", "pool-scale", "pool-state-downgrade"):
        from . import pool_run  # noqa: PLC0415

        cfg = json.loads(Path(args.config).read_text())
        if args.cmd == "pool-scale":
            try:
                return pool_run.scale_pool(cfg, args.config, args.target, retire=args.retire, add=args.add,
                                           remove=args.remove, mode=args.mode, force_static=args.force_static,
                                           dry_run=args.dry_run, source=args.source)
            except ValueError as error:
                print(f"pool-scale: refused, nothing changed: {error}")
                return 2
        if args.cmd == "pool-status":
            return pool_run.pool_status(cfg, args.config, events=args.events)
        if args.cmd == "pool-retry":
            return pool_run.retry_pool(cfg, args.config)
        if args.cmd == "pool-state-downgrade":
            return pool_run.downgrade_state_file(cfg, args.config, drop_owed_fences=args.drop_owed_fences)
        if args.cmd == "pool-stop":
            return pool_run.stop_pool(cfg, args.config, wait_s=args.wait)
        if args.dry_run:
            return pool_run.dry_run(cfg, args.config)
        return pool_run.run_pool(cfg, args.config, interval=args.interval)
    return 2


_OVERRIDE_KEYS = ("data_slot", "drop_zero_variance", "thinking_effort", "reload", "n", "max_tokens",
                  "continues_shard")
_POOL_KEYS = ("drop_zero_variance", "thinking_effort", "reload", "n", "max_tokens", "dispense")


def _add_sampler_override_args(p) -> None:
    """The run knobs of ``tasks.apply_sampler_overrides`` (unset: the spec file's value)."""
    p.add_argument("--data-slot", metavar="I/N", help="this sampler's prompt rows (i %% N == I)")
    p.add_argument("--drop-zv", dest="drop_zero_variance", action="store_true", default=None,
                   help="drop zero-variance groups")
    p.add_argument("--keep-zv", dest="drop_zero_variance", action="store_false")
    p.add_argument("--thinking-effort")
    p.add_argument("--reload", choices=tasks.RELOADS)
    p.add_argument("--n", type=int, help="samples per prompt")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--continues-shard", help="replacement: continue this shard's prompts (it is asked to stop)")
    p.add_argument("--dispense-namespace", help="take tasks from the store ledger under this run-store "
                   "path (every sampler of the run: the same value); excludes --data-slot")
    p.add_argument("--dispense-lease-s", type=float, help="claim lease (renewed every lease/3; default 900)")
    p.add_argument("--dispense-max-redo", type=int, help="infra/expired attempts before a task is exhausted")
    p.add_argument("--dispense-passes", type=int, help="passes over the data (default 1)")
    p.add_argument("--dispense-seed", type=int)
    p.add_argument("--dispense-holdout", help="local hold-out file; a row with one of its task ids is refused")
    p.add_argument("--dispense-mix", help="--data-mix source weights, comma separated (rows carry the source)")
    p.add_argument("--dispense-labels", help="task labels this sampler takes (comma separated)")
    p.add_argument("--dispense-by-label", action="store_true", default=False,
                   help="one stream per task label (weighted by row count)")


def _arg_overrides(args) -> dict:
    out = {k: getattr(args, k, None) for k in _OVERRIDE_KEYS}
    if getattr(args, "dispense_namespace", None):
        out["dispense"] = tasks.dispense_config(
            args.dispense_namespace, lease_s=args.dispense_lease_s, max_redo=args.dispense_max_redo,
            passes=args.dispense_passes, seed=args.dispense_seed, holdout_file=args.dispense_holdout,
            mix=args.dispense_mix, labels=args.dispense_labels, by_label=args.dispense_by_label)
    return out


def _overrides(cfg) -> dict:
    """A pool config's ``sampler`` knobs (the data slot comes from the member)."""
    unknown = set(cfg) - set(_POOL_KEYS)
    if unknown:
        raise ValueError(f"pool sampler config: unknown keys {sorted(unknown)}")
    if cfg.get("dispense") is not None:
        from yeto.rl.engine.store_dispense import check_store_config  # noqa: PLC0415

        try:
            check_store_config(cfg["dispense"])
        except Exception as error:  # noqa: BLE001 - DispenseError: a pool config error
            raise ValueError(f"pool sampler config: dispense: {error}") from error
    return {k: cfg.get(k) for k in _POOL_KEYS}


if __name__ == "__main__":
    sys.exit(main())
