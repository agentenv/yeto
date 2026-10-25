"""Two-cluster RL island (S27 G-DI): a trainer cluster and an engine cluster, one Ray.

``--rl-engine-cluster SPEC`` splits a sky RL island into two sky clusters. The ``--gpu``
entry stays the LOGICAL island: every island rule (trainer shape, placement, Miles and
learner flags) sees its node count. Its last ``n`` nodes (``n`` = SPEC's node count) are
provisioned as a second cluster of SPEC's machine type:

====================  ===========================  ======================================
cluster               nodes (island ranks)         runs
====================  ===========================  ======================================
``<name>``            0 .. N-n-1, the --gpu card   Ray head, island driver, trainer
``<name>-eng``        N-n .. N-1, SPEC's card      Ray workers, the rollout engines
====================  ===========================  ======================================

Why: one sky cluster has one machine type, so "H200 trainer + H100 engines" needs two.

Launch (:class:`MixedIsland`):

1. ``sky.launch`` of both clusters at the same time. The trainer cluster gets the island
   task (its job starts the Ray head and the driver at once). The engine cluster gets the
   same setup and no run; sky runs that setup as a detached job, and the launcher waits
   until the job SUCCEEDED.
2. When both are up: the two Nebius security groups admit each other
   (:mod:`yeto.cloud.nebius_sg_link`; sky's default rule admits only the cluster's own
   group). Then ``sky.exec`` submits the engine run with the trainer head's private
   address in ``YETO_ISLAND_HEAD_ADDR``. The engine run shifts its node ranks by the
   trainer node count and puts that address first in ``SKYPILOT_NODE_IPS``, so the
   island run script takes its worker branch and joins the head's Ray at port 6379.
3. The driver waits (``YETO_ISLAND_NODE_WAIT_S``) until every island node is in its Ray
   before it pins the trainer block to the Ray head (D3 head pin) and starts the engines.
   The trainer therefore always lands on the trainer cluster, whatever the IP order.

Network: the clusters must share one private network (same cloud, same region). Every
port between them is open (Ray 6379 and its worker / object-manager ports, the
dashboard 8265, NCCL and gloo sockets, the SGLang and router HTTP ports, the
torch.distributed rendezvous of the weight-update group). Nothing is opened to the
internet. NCCL runs on TCP sockets (``NCCL_IB_DISABLE=1``, the Nebius multi-node default)
on the default-route interface, with more socket threads (:data:`NCCL_SOCKET_ENVS`).

Failure and preemption: the fleet supervises the trainer cluster's job only.
* The engine cluster is lost (preempted, crashed): its Ray node dies, the engines' Ray
  actors and HTTP endpoints fail, the driver fails, and the trainer job exits non-zero.
  The fleet then treats the island as failed, exactly as it does for a multi-node island
  that loses a worker node.
* The trainer cluster is lost: the Ray head is gone, so the engine run's
  ``ray status`` loop ends and its job ends.
* Relaunch (fleet recovery) relaunches BOTH clusters (:meth:`MixedIsland.relaunch`):
  provision what is gone, rerun the trainer task, and ``sky.exec`` a new engine run
  with the (possibly new) head address. A relaunch whose engine side fails cancels the
  new trainer job and reports failure, so the fleet retries within --recover-timeout.
* Teardown downs both clusters (they are in the launcher's cluster list and the run
  registry); the security-group rules are removed first.
* The trainer keeps no state across a relaunch beyond what a one-cluster island keeps
  (sync state with the syncer, checkpoints in the store). An engine-only restart that
  keeps the trainer running is NOT implemented.

Default: without ``--rl-engine-cluster`` nothing here runs and every task, command and
cluster list is unchanged.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..gpu_spec import ClusterSpec, parse_gpu_spec
from ..rl.engine.multinode import ISLAND_NODE_WAIT_ENV

ENGINE_SUFFIX = "-eng"
HEAD_ADDR_ENV = "YETO_ISLAND_HEAD_ADDR"
# The driver's wait for the engine cluster's nodes (step 3 above). The engine run is
# submitted only after the engine cluster finished setup, so the wait normally lasts
# the difference of the two setups; 60 min bounds a stuck engine cluster.
NODE_WAIT_S = 3600
# sky runs a task's setup as a detached job (detach_setup, the sky 0.13 default):
# sky.launch returns once that job is submitted. The engine run must not start before
# the setup job ends (Miles/SGLang checkout, model prefetch), so the launcher waits.
SETUP_WAIT_S = 3600
SETUP_POLL_S = 15.0
SUPPORTED_CLOUDS = frozenset({"nebius"})
# TCP sockets between two VMs (no InfiniBand across Nebius fabrics). More threads and
# sockets per peer than NCCL's default 1 x 1; both clusters set the same values.
# Inference, not measured on this pair yet.
NCCL_SOCKET_ENVS = {"NCCL_SOCKET_NTHREADS": "4", "NCCL_NSOCKS_PERTHREAD": "4"}
RUN_ANCHOR = "set -e\ncd ~/sky_workdir\n"
_IFACE_LINE = ("export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"
               "$(awk '$2==\"00000000\"{print $1; exit}' /proc/net/route)}\n")


@dataclass(frozen=True)
class EngineSplit:
    """One island split into a trainer cluster and an engine cluster."""

    learner_id: int
    name: str  # the island's cluster name = the trainer cluster
    engine_name: str
    island: ClusterSpec  # the logical --gpu entry
    trainer: ClusterSpec
    engine: ClusterSpec

    @property
    def rank_offset(self) -> int:
        return self.trainer.num_nodes

    def record(self) -> dict[str, Any]:
        return {
            "trainer_cluster": self.name, "trainer": str(self.trainer),
            "engine_cluster": self.engine_name, "engine": str(self.engine),
            "engine_island_ranks": list(range(self.rank_offset, self.island.num_nodes)),
            "network": "same private network; the two security groups admit each other",
            "nccl": {"NCCL_IB_DISABLE": "1", **NCCL_SOCKET_ENVS},
            "node_wait_s": NODE_WAIT_S,
        }


def requested(args) -> bool:
    return bool(getattr(args, "rl_engine_cluster", None))


def _entries(raw: str, n: int) -> list[ClusterSpec | None]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--rl-engine-cluster is empty")
    if len(parts) == 1:
        parts = parts * n
    if len(parts) != n:
        raise ValueError(f"--rl-engine-cluster has {len(parts)} entries; give one, or one per "
                         f"--gpu entry ({n})")
    return [None if p.lower() == "none" else parse_gpu_spec(p)[0] for p in parts]


def plan(args, specs: list[ClusterSpec], names: list[str], *,
         role_of: Callable[[int], Any] = lambda m: None) -> dict[str, EngineSplit]:
    """The islands to split, by trainer cluster name; ``{}`` when the flag is unset.
    Raises ValueError for every combination this design does not cover."""
    if not requested(args):
        return {}
    if getattr(args, "training_mode", "sft") != "rl":
        raise ValueError("--rl-engine-cluster needs --training-mode rl")
    if (getattr(args, "rl_backend", None) or "miles") != "miles":
        raise ValueError("--rl-engine-cluster needs --rl-backend miles (the D3 head pin and the "
                         "engine recovery port are Miles adapter code; verl has neither)")
    from ..rl.engine import island_placement
    from . import trainer_loop
    from .topology import rl_island_layout, rl_island_spec, rl_trainer_shape

    for flag, bad in (("--rl-elastic", bool(getattr(args, "rl_elastic", False))),
                      ("--learner-instance-type", bool(getattr(args, "learner_instance_type", None)))):
        if bad:
            raise ValueError(f"--rl-engine-cluster cannot be combined with {flag}")
    held = trainer_loop.held_island(args)
    out: dict[str, EngineSplit] = {}
    for m, (spec, engine) in enumerate(zip(specs, _entries(args.rl_engine_cluster, len(specs)))):
        if engine is None:
            continue
        where = f"island {m} ({spec})"
        if spec.cloud not in SUPPORTED_CLOUDS or engine.cloud != spec.cloud:
            raise ValueError(f"--rl-engine-cluster on {where}: both clusters must be on "
                             f"{sorted(SUPPORTED_CLOUDS)} (got {spec.cloud} + {engine.cloud})")
        if not spec.region or engine.region != spec.region:
            raise ValueError(f"--rl-engine-cluster on {where}: give both clusters the same "
                             f"explicit region (one private network); got {spec.region} + {engine.region}")
        if engine.gpus_per_node != spec.gpus_per_node:
            raise ValueError(f"--rl-engine-cluster on {where}: {engine.gpus_per_node} GPUs per "
                             f"engine node, {spec.gpus_per_node} per island node; they must be equal")
        trainer_nodes = spec.num_nodes - engine.num_nodes
        if trainer_nodes != 1:
            raise ValueError(f"--rl-engine-cluster on {where}: the trainer cluster must be exactly "
                             f"1 node (island nodes {spec.num_nodes} - engine nodes "
                             f"{engine.num_nodes} = {trainer_nodes}); the D3 head pin places only "
                             "the Ray head's node")
        if m == held:
            raise ValueError(f"--rl-engine-cluster on {where}: the trainer scheduler's island")
        if role_of(m) is not None:
            raise ValueError(f"--rl-engine-cluster on {where}: island role {role_of(m)!r}")
        iargs = island_placement.island_args(args, m, len(specs))
        if getattr(iargs, "rl_placement", "colocated") != "fixed-partition":
            raise ValueError(f"--rl-engine-cluster on {where} needs --rl-placement fixed-partition")
        if int(getattr(iargs, "rl_standby_gpus", 0) or 0):
            raise ValueError(f"--rl-engine-cluster on {where}: no --rl-standby-gpus")
        if getattr(iargs, "rl_island_use_gpus_per_node", None) is not None:
            raise ValueError(f"--rl-engine-cluster on {where}: no --rl-island-use-gpus-per-node")
        island = rl_island_spec(iargs, spec)
        if rl_island_layout(iargs, island) is not None:
            raise ValueError(f"--rl-engine-cluster on {where}: no explicit island layout")
        t_nodes, t_gpn = rl_trainer_shape(iargs, island)
        rollout = int(getattr(iargs, "rollout_num_gpus", 0) or 0)
        if (t_nodes, t_gpn) != (1, spec.gpus_per_node) or rollout != engine.total_gpus:
            raise ValueError(f"--rl-engine-cluster on {where}: the trainer must fill the trainer "
                             f"node and the engines the engine cluster (trainer {t_nodes}x{t_gpn}, "
                             f"rollout GPUs {rollout}, engine cluster {engine.total_gpus} GPUs)")
        name = names[m]
        out[name] = EngineSplit(
            learner_id=m, name=name, engine_name=engine_cluster_name(name), island=spec,
            trainer=dataclasses.replace(spec, num_nodes=trainer_nodes), engine=engine)
    return out


def engine_cluster_name(name: str) -> str:
    from .sky_ops import sky_cluster_name

    return sky_cluster_name(f"{name}{ENGINE_SUFFIX}")


def run_for(run: str, *, engine: bool, rank_offset: int = 0) -> str:
    """The island run script for one of the two clusters: the interface default for both,
    and on the engine cluster the rank shift and the head address."""
    if run.count(RUN_ANCHOR) != 1:
        raise ValueError("two-cluster island: the run script has no single 'set -e; cd "
                         "~/sky_workdir' anchor")
    lines = ["# two-cluster island (yeto.launch.engine_cluster): NCCL/gloo on the default-route NIC\n",
             _IFACE_LINE]
    if engine:
        lines += [
            f'[ -n "${{{HEAD_ADDR_ENV}:-}}" ] || {{ echo "[yeto-island] engine cluster: '
            f'{HEAD_ADDR_ENV} is unset" >&2; exit 1; }}\n',
            f"SKYPILOT_NODE_RANK=$((SKYPILOT_NODE_RANK + {int(rank_offset)}))\n",
            f'SKYPILOT_NODE_IPS="$(printf \'%s\\n%s\' "${HEAD_ADDR_ENV}" "$SKYPILOT_NODE_IPS")"\n',
            f'echo "[yeto-island] engine cluster node: island rank $SKYPILOT_NODE_RANK, '
            f'Ray head ${HEAD_ADDR_ENV}"\n',
        ]
    return run.replace(RUN_ANCHOR, RUN_ANCHOR + "".join(lines), 1)


def _add_envs(task, envs: dict[str, str]) -> None:
    if callable(getattr(task, "update_envs", None)):
        task.update_envs(envs)
    else:
        task.envs = {**(getattr(task, "envs", None) or {}), **envs}


def _shape(sky, args, task, spec: ClusterSpec) -> None:
    from .rl_island_task import island_sky_resources

    task.num_nodes = spec.num_nodes
    task.set_resources(sky.Resources(**island_sky_resources(args, spec, in_vm_docker=False)))


@dataclass
class SplitTasks:
    trainer: Any
    engine_setup: Any
    engine_run: Any


def recover_s(args) -> float:
    """``--rl-engine-cluster-recover-s``: design v2 outage budget (0 = off, v1 semantics)."""
    return float(getattr(args, "rl_engine_cluster_recover_s", 0) or 0)


def engine_cell_names(args, split: EngineSplit) -> list[str]:
    per = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    return [f"e{i}" for i in range(split.engine.total_gpus // per)]


def with_declared_cells(run: str, cells: list[str], module: str) -> str:
    """Append the fork F-R1 cell declaration to the island's learner command (the
    Miles recovery port starts and stops those cells)."""
    lines = run.split("\n")
    hits = [i for i, line in enumerate(lines) if f"-m {module}" in line]
    if len(hits) != 1:
        raise ValueError(f"two-cluster island: {len(hits)} learner command lines in the run script")
    i = hits[0]
    flags = f" --rl-elastic-declare-cells --rl-elastic-cells {','.join(cells)}"
    head, sep, tail = lines[i].partition(" || LRC=$?")  # keep --rl-island-post-cmd's suffix last
    lines[i] = head + flags + sep + tail
    return "\n".join(lines)


def build_tasks(sky, args, split: EngineSplit, make_task: Callable[[], Any]) -> SplitTasks:
    """Three sky tasks from three fresh builds of the logical island task: the trainer
    cluster's (setup + run), the engine cluster's launch (setup only) and its run."""
    trainer = make_task()
    _shape(sky, args, trainer, split.trainer)
    trainer.run = run_for(trainer.run, engine=False)
    _add_envs(trainer, {**NCCL_SOCKET_ENVS, ISLAND_NODE_WAIT_ENV: str(NODE_WAIT_S)})
    if recover_s(args) > 0:  # design v2: the trainer survives the engine cluster
        from ..rl.engine.engine_outage import ENGINE_RECOVERY_ENV
        from .rl_backend import _rl_backend_module

        module = _rl_backend_module(args, "launch_flags").ISLAND_ENTRY_MODULE
        trainer.run = with_declared_cells(trainer.run, engine_cell_names(args, split), module)
        _add_envs(trainer, {ENGINE_RECOVERY_ENV: str(int(recover_s(args)))})
    setup = make_task()
    _shape(sky, args, setup, split.engine)
    setup.run = None
    run = make_task()
    _shape(sky, args, run, split.engine)
    run.run = run_for(run.run, engine=True, rank_offset=split.rank_offset)
    _add_envs(run, dict(NCCL_SOCKET_ENVS))
    return SplitTasks(trainer=trainer, engine_setup=setup, engine_run=run)


def head_address(handle) -> str:
    """The trainer cluster head's private address (sky's internal IP)."""
    ips = list(getattr(handle, "cached_internal_ips", None) or [])
    if not ips and callable(getattr(handle, "internal_ips", None)):
        ips = list(handle.internal_ips())
    if not ips or not ips[0]:
        raise RuntimeError("two-cluster island: the trainer cluster has no internal IP")
    return str(ips[0])


class MixedIsland:
    """Launch, relaunch and tear down one two-cluster island.

    ``submit(spec, task, name)`` returns a sky request id (``submit_island_launch``);
    ``link(region, a, b)`` / ``unlink(...)`` take sky's ``cluster_name_on_cloud`` of both
    clusters (default: :mod:`yeto.cloud.nebius_sg_link`)."""

    def __init__(self, split: EngineSplit, tasks: SplitTasks, *, submit: Callable,
                 link: Callable | None = None, unlink: Callable | None = None,
                 log: Callable[[str], Any] = print, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] = time.sleep) -> None:
        from ..cloud import nebius_sg_link

        self.split, self.tasks, self.submit_fn, self.log = split, tasks, submit, log
        self.clock, self.sleep = clock, sleep
        self.link_fn = link or nebius_sg_link.link
        self.unlink_fn = unlink or nebius_sg_link.unlink
        self.engine_job: int | None = None
        self.on_cloud: tuple[str, str] | None = None
        self.head: str | None = None
        self.engine_relaunches = 0
        self.stop_watch = False

    def submit(self) -> tuple[Any, Any]:
        s = self.split
        self.log(f"[launcher] two-cluster island {s.name}: trainer cluster {s.name} ({s.trainer}), "
                 f"engine cluster {s.engine_name} ({s.engine})")
        return (self.submit_fn(s.trainer, self.tasks.trainer, s.name),
                self.submit_fn(s.engine, self.tasks.engine_setup, s.engine_name))

    def finish(self, sky, rids: tuple[Any, Any]) -> tuple[Any, Any]:
        """Wait for both launches, link the networks, start the engine run. Returns the
        trainer cluster's ``(job_id, handle)``; on an engine-side failure the new trainer
        job is cancelled and the error raised."""
        s = self.split
        try:
            trainer = sky.stream_and_get(rids[0])
        except Exception:
            # let the engine launch finish too, so the teardown that follows finds a
            # settled cluster (not one still being provisioned)
            try:
                sky.stream_and_get(rids[1])
            except Exception as exc:  # noqa: BLE001
                self.log(f"[launcher] {s.engine_name}: launch also failed ({exc})")
            raise
        try:
            engine = sky.stream_and_get(rids[1])
            self._wait_setup(sky, engine[0])
            head = self.head = head_address(trainer[1])
            self.on_cloud = (str(getattr(trainer[1], "cluster_name_on_cloud", s.name)),
                             str(getattr(engine[1], "cluster_name_on_cloud", s.engine_name)))
            self.link_fn(s.trainer.region, *self.on_cloud)
            _add_envs(self.tasks.engine_run, {HEAD_ADDR_ENV: head})
            job, _h = sky.stream_and_get(sky.exec(self.tasks.engine_run, cluster_name=s.engine_name))
            self.engine_job = job
            self.log(f"[launcher] {s.engine_name}: engine run job {job} joins Ray head {head}:6379")
        except Exception:
            self._cancel(sky, s.name, trainer[0])
            raise
        return trainer

    def _wait_setup(self, sky, job_id) -> None:
        """Wait until the engine cluster's detached setup job ends; raise unless it
        succeeded. No job id (setup ran inside the launch): nothing to wait for."""
        if job_id is None:
            return
        name, start = self.split.engine_name, self.clock()
        while True:
            status = sky.get(sky.job_status(name, [int(job_id)])).get(int(job_id))
            state = getattr(status, "name", str(status))
            if status is not None and callable(getattr(status, "is_terminal", None)) and status.is_terminal():
                if state != "SUCCEEDED":
                    raise RuntimeError(f"{name}: setup job {job_id} ended {state}")
                self.log(f"[launcher] {name}: setup job {job_id} SUCCEEDED after "
                         f"{self.clock() - start:.0f}s")
                return
            if self.clock() - start > SETUP_WAIT_S:
                raise RuntimeError(f"{name}: setup job {job_id} still {state} after {SETUP_WAIT_S}s")
            self.sleep(SETUP_POLL_S)

    def _cancel(self, sky, cluster: str, job_id) -> None:
        if job_id is None:
            return
        try:
            sky.get(sky.cancel(cluster, job_ids=[int(job_id)]))
            self.log(f"[launcher] {cluster}: cancelled job {job_id} (engine cluster not ready)")
        except Exception as exc:  # noqa: BLE001
            self.log(f"[launcher] {cluster}: cancel of job {job_id} failed: {exc}")

    def relaunch(self, sky) -> Any:
        """Fleet recovery: both clusters again. Returns the new trainer job id or None."""
        s = self.split
        self._cancel(sky, s.engine_name, self.engine_job)
        try:
            job, handle = self.finish(sky, self.submit())
        except Exception as exc:  # noqa: BLE001
            self.log(f"[launcher] relaunch of two-cluster island {s.name} failed: {exc}")
            return None
        return job

    def relaunch_engine(self, sky) -> Any:
        """Design v2: bring back only the engine cluster (the trainer job keeps
        running): provision + setup, wait for the setup job, admit the groups again,
        start the engine run toward the same Ray head. Returns the engine job id."""
        s = self.split
        if self.head is None:
            raise RuntimeError(f"{s.name}: no trainer head address to rejoin")
        self._cancel(sky, s.engine_name, self.engine_job)
        engine = sky.stream_and_get(self.submit_fn(s.engine, self.tasks.engine_setup, s.engine_name))
        self._wait_setup(sky, engine[0])
        on_cloud = (self.on_cloud or (s.name, s.engine_name))[0]
        self.on_cloud = (on_cloud, str(getattr(engine[1], "cluster_name_on_cloud", s.engine_name)))
        self.link_fn(s.trainer.region, *self.on_cloud)
        job, _h = sky.stream_and_get(sky.exec(self.tasks.engine_run, cluster_name=s.engine_name))
        self.engine_job = job
        self.engine_relaunches += 1
        self.log(f"[launcher] {s.engine_name}: engine cluster back (relaunch {self.engine_relaunches}), "
                 f"engine run job {job} joins Ray head {self.head}:6379")
        return job

    def engine_lost(self, sky) -> bool:
        """The engine run is no longer alive (cluster gone or job ended)."""
        if self.engine_job is None:
            return False
        try:
            status = sky.get(sky.job_status(self.split.engine_name, [int(self.engine_job)])).get(
                int(self.engine_job))
        except Exception:  # noqa: BLE001 - the cluster is gone (preempted / terminated)
            return True
        return status is None or (callable(getattr(status, "is_terminal", None)) and status.is_terminal())

    def watch(self, sky, stop, *, poll_s: float = 60.0, trainer_alive: Callable[[], bool] = lambda: True) -> None:
        """Design v2 supervisor thread body: relaunch the engine cluster whenever its
        run is lost while the trainer runs. ``stop``: a threading.Event."""
        while not stop.wait(poll_s) and not self.stop_watch:
            if not trainer_alive() or not self.engine_lost(sky):
                continue
            self.log(f"[launcher] {self.split.engine_name}: engine run lost; relaunching the engine cluster only")
            try:
                self.relaunch_engine(sky)
            except Exception as exc:  # noqa: BLE001 - retried at the next poll
                self.log(f"[launcher] {self.split.engine_name}: engine relaunch failed: {exc}")

    def unlink(self) -> None:
        if self.on_cloud is not None:
            self.unlink_fn(self.split.trainer.region, *self.on_cloud)


class EngineClusterOps:
    """The fleet's sky ops with two-cluster islands: ``relaunch`` and ``down`` of a
    trainer cluster cover its engine cluster too; every other call goes to ``base``."""

    def __init__(self, base, islands: dict[str, MixedIsland], sky_module=None) -> None:
        self.base, self.islands, self._sky = base, islands, sky_module

    def _sky_mod(self):
        if self._sky is None:
            import sky

            return sky
        return self._sky

    def relaunch(self, task, cluster):
        island = self.islands.get(cluster)
        if island is None:
            return self.base.relaunch(task, cluster)
        return island.relaunch(self._sky_mod())

    def down(self, cluster):
        island = self.islands.get(cluster)
        if island is None:
            return self.base.down(cluster)
        island.unlink()
        try:
            self.base.down(island.split.engine_name)
        finally:
            self.base.down(cluster)

    def __getattr__(self, name):
        return getattr(self.base, name)


def nodes_by_cluster(splits: dict[str, EngineSplit]) -> dict[str, int]:
    """Physical node counts of both clusters of every split island."""
    out: dict[str, int] = {}
    for s in splits.values():
        out[s.name] = s.trainer.num_nodes
        out[s.engine_name] = s.engine.num_nodes
    return out


def dry_run_record(split: EngineSplit, make_task: Callable[[], Any], args) -> dict[str, Any]:
    """``yeto launch --dry-run`` view of a split island: both cluster shapes, the lines
    each run script gets, and whether both carry the same learner command."""
    import hashlib

    import sky

    tasks = build_tasks(sky, args, split, make_task)
    base = make_task().run
    added = {kind: [line for line in run.splitlines() if line not in base.splitlines()]
             for kind, run in (("trainer", tasks.trainer.run), ("engine", tasks.engine_run.run))}
    return {
        **split.record(),
        "trainer_run_added": added["trainer"],
        "engine_run_added": added["engine"],
        "engine_setup_has_run": tasks.engine_setup.run is not None,
        "trainer_envs_added": sorted(set(tasks.trainer.envs) - set(make_task().envs)),
        "run_sha256": {k: hashlib.sha256(t.run.encode()).hexdigest()
                       for k, t in (("trainer", tasks.trainer), ("engine", tasks.engine_run))},
    }
