"""bm-sim: the CPU learner that replaces ``yeto.rl.adapters.miles.island_entry``
on the simulated bare-metal trainer island.

The launcher's own run script starts it on rank 0 (the harness swaps only the
module name of the learner line, :func:`scenario.patch_learner`), with the argv
and the environment the launcher built for the ``ssh:`` island. It then does
what the Miles learner does around the trainer, on CPU:

1. the argv must parse with the real ``island_entry.parse_args`` and name the
   trainer role;
2. it joins the island's Ray (``RAY_ADDRESS``, the head the run script started)
   and waits until every inventory node joined with its inventory address
   (``SKYPILOT_NODE_IPS``), i.e. the worker ``--node-ip-address`` pinning holds;
3. it opens the island's store with ``resume.store_for`` from the exported store
   environment (a staged ``s3://`` store: one stage per node, object API);
4. it runs the commanded trainer incarnation (``trainer_remote.run_commanded_island``:
   head commands over the store, lifecycle events echoed on stdout) over a
   fake engine that trains on the durable sample buffer;
5. every round publishes a policy release (``PolicyReleaseStore``); every cut
   writes one shard per node: rank 0's in its own stage, the others' by a Ray
   task pinned to that node, which uploads it from that node's stage
   (``cut.push_shard_from_rank``, as a Miles rank does), then the pointer and a
   state snapshot are committed (``recovery.save`` -> ``store.commit``);
6. a later incarnation drops its local stage and restores the newest cut from
   the bucket alone (``store.reload``), checking every node's shard digest.

Without the scheduler's control environment (a resident island the fleet
supervises, ``ssh_resident``) the incarnation trains to the end of the data on
its own: each round waits for a full batch, the round after which the data has
ended and less than a batch is left is the last, and a clean end emits
``rl_learner_finalized`` (rank 0 exits 0). It cuts after every round (the
resume store's ``--rl-cut-every 1``), and a relaunched incarnation (the fleet's
recovery after a failure) restores the store's newest cut, as item 6.

Run as ``python3 -m tests.integration.bm_sim.fake_learner <island_entry argv>``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SHARD_BYTES = 64 * 1024
RAY_JOIN_TIMEOUT_S = 180.0
COLLECTIVE_TIMEOUT_S = 60.0  # a round's "collective" (every node takes part) fails after this
ROUND_S_ENV = "BM_SIM_ROUND_S"  # extra seconds per trained round (a slower trainer)
SAMPLE_WAIT_S = 600.0  # a resident round waits this long for a full batch
STATE_DIR = "~/bm-sim-state"  # this node's state dir (the pointer before its snapshot)
TENSOR = "base_model.model.layer.lora_A.weight"


def log(*parts: Any) -> None:
    print("[bm-sim-learner]", *parts, flush=True)


def node_ips(environ=os.environ) -> list[str]:
    return [line.strip() for line in environ.get("SKYPILOT_NODE_IPS", "").splitlines() if line.strip()]


def shard_payload(cut_id: str, rank: int) -> bytes:
    """Deterministic shard bytes, so a restore can check them without the writer."""
    seed = hashlib.sha256(f"{cut_id}/{rank}".encode()).digest()
    return (seed * (SHARD_BYTES // len(seed) + 1))[:SHARD_BYTES]


def shard_name(rank: int) -> str:
    return f"shard-node{rank:02d}.bin"


def _write_and_push_shard(store_path: str, cut_id: str, rank: int) -> dict[str, Any]:
    """On the node of ``rank``: write its shard into this node's stage and upload it."""
    from yeto.rl.engine.cut import push_shard_from_rank
    from yeto.rl.engine.resume import ROUND_CUTS, store_for

    store = store_for(Path(store_path).expanduser())
    final = Path(store.root) / ROUND_CUTS / cut_id / shard_name(rank)
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(shard_payload(cut_id, rank))
    pushed = push_shard_from_rank(final)
    return {"rank": rank, "host": os.uname().nodename, "staged": bool(getattr(store, "staged", False)),
            "stage_root": str(store.root), "pushed": pushed}


def _rank_alive(rank: int) -> str:
    return os.uname().nodename


# what a trainer rank on each node must see (the island's exports reach the Ray workers
# through the raylet the run script started)
NODE_ENV_KEYS = ("SKYPILOT_NODE_RANK", "BM_SIM_NODE", "NCCL_IB_DISABLE", "NCCL_IB_HCA", "NCCL_SOCKET_IFNAME",
                 "GLOO_SOCKET_IFNAME", "YETO_DEVICE_FAMILY", "YETO_RL_STORE_IMPL", "YETO_RL_STORE_S3_STAGE",
                 "YETO_TRAINER_INCARNATION")


def _rank_env(rank: int) -> dict[str, Any]:
    return {"rank": rank, "host": os.uname().nodename,
            "env": {k: os.environ.get(k) for k in NODE_ENV_KEYS},
            "store_secret": bool(os.environ.get("YETO_RL_STORE_S3_SECRET_ACCESS_KEY"))}


def node_envs(ranks: dict[int, str]) -> list[dict[str, Any]]:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    task = ray.remote(num_cpus=0)(_rank_env)
    return ray.get([task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)).remote(rank)
                    for rank, node in sorted(ranks.items())], timeout=COLLECTIVE_TIMEOUT_S)


def collective(ranks: dict[int, str], timeout_s: float = COLLECTIVE_TIMEOUT_S) -> list[str]:
    """Every node of the island takes part in each round, as the trainer's
    collectives do: one task pinned to each node. A dead node fails the round
    (an NCCL collective would time out the same way, later)."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    task = ray.remote(num_cpus=0)(_rank_alive)
    refs = [task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)).remote(rank)
            for rank, node in sorted(ranks.items())]
    try:
        return ray.get(refs, timeout=timeout_s)
    except Exception as error:  # noqa: BLE001 - one error type for the driver
        raise RuntimeError(f"[bm-sim-learner] round collective failed: {error!r}"[:2000]) from None


def join_ray(expected: list[str], timeout_s: float = RAY_JOIN_TIMEOUT_S) -> dict[str, str]:
    """Ray node id -> node IP, once every expected address joined."""
    import ray

    pythonpath = os.environ.get("PYTHONPATH", "")
    # Miles hands the driver's PYTHONPATH to every actor through the job runtime_env
    ray.init(address=os.environ["RAY_ADDRESS"], runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
             log_to_driver=True)
    deadline = time.time() + timeout_s
    while True:
        alive = {n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n.get("Alive")}
        if sorted(alive.values()) == sorted(expected):
            return alive
        if time.time() > deadline:
            raise SystemExit(f"[bm-sim-learner] Ray nodes {sorted(alive.values())} != inventory {sorted(expected)} "
                             f"after {timeout_s:.0f} s")
        time.sleep(1.0)


class SimRoundCuts:
    """The round-cut holder of ``LocalOnlySync`` for the simulated island (see the
    module doc, items 5 and 6). Cuts at the end of an incarnation (``final``)."""

    def __init__(self, store: Any, buffer: Any, *, store_path: str, ranks: dict[int, str],
                 incarnation: str, every_round: bool = False) -> None:
        from yeto.rl.engine.trainer_remote import StoreCutLog

        self.store = store
        self.buffer = buffer
        self.store_path = store_path
        self.ranks = ranks  # rank -> Ray node id
        self.incarnation = str(incarnation)  # a label: cut ids and the state dir
        self.every_round = every_round  # cut at every safe point, not only at the end
        self.state_dir = Path(STATE_DIR).expanduser() / f"i{self.incarnation}"
        self.log = StoreCutLog(store.objects)
        self.saved: list[dict[str, Any]] = []
        self.resumed: dict[str, Any] | None = None

    # LocalOnlySync / IslandTrainerAgent hooks ------------------------------------
    def buffer_watermark(self, _version: int) -> int:
        return int(self.buffer.watermark)

    def latest(self):
        return self.log.latest()

    def due(self, _rollout_id: int, *, final: bool = False) -> bool:
        return self.every_round or final

    def save(self, driver: Any, *, rollout_id: int, final: bool = False, trigger: str | None = None):
        if not (final or self.every_round):
            return None
        from yeto.rl.engine import recovery
        from yeto.rl.engine.resume import ROUND_CUTS, write_snapshot

        pointer = self.log.pointer()
        if pointer is not None and int(pointer.get("next_rollout_id", -1)) == int(rollout_id):
            return None  # nothing trained since this cut
        import ray
        import torch
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        cut_id = f"r{int(rollout_id):06d}-i{self.incarnation}"
        root = Path(self.store.root) / ROUND_CUTS / cut_id
        root.mkdir(parents=True, exist_ok=True)
        engine = driver.trainer.engine
        torch.save({"tensors": engine.tensors, "moments": engine.moments, "step": engine.scheduler_step},
                   root / "trainer.pt")
        push = ray.remote(num_cpus=0)(_write_and_push_shard)
        shards = ray.get([push.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False))
                          .remote(self.store_path, cut_id, rank) for rank, node in sorted(self.ranks.items())])
        for shard in shards:
            pushed = shard["pushed"] or {}
            if "object_key" not in pushed:
                raise RuntimeError(f"cut {cut_id}: rank {shard['rank']} shard was not uploaded: {shard}")
        log(f"cut {cut_id}: {len(shards)} shard(s) pushed from", sorted({s['host'] for s in shards}))
        cut = recovery.ConsistentCut(checkpoint_id=cut_id, trigger=trigger or recovery.MANUAL,
                                     outer_step=int(rollout_id), island_steps={"0": int(driver.local_step)},
                                     buffer_watermark=int(self.buffer.watermark))
        info = {"cut_id": cut_id, "next_rollout_id": int(rollout_id), "policy_version": int(rollout_id),
                "local_step": int(driver.local_step), "incarnation": self.incarnation,
                "shards": {str(s["rank"]): {"key": s["pushed"]["object_key"], "host": s["host"],
                                            "sha256": hashlib.sha256(shard_payload(cut_id, s["rank"])).hexdigest()}
                           for s in shards}}

        def write(record: dict[str, Any]) -> dict[str, Any]:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            (self.state_dir / "round-cut.json").write_text(
                json.dumps({**info, recovery.CUT_KEY: record}, sort_keys=True), encoding="utf-8")
            write_snapshot(self.store, self.state_dir, {"cut_id": cut_id, "next_rollout_id": int(rollout_id),
                                                        "incarnation": self.incarnation}, commit=False)
            return {}

        saved = recovery.save(cut, store=self.store, write=write)  # commits: shards, pointer, LATEST
        self.saved.append(saved)
        log(f"cut {cut_id} committed (trigger {cut.trigger}, watermark {cut.buffer_watermark})")
        return {**info, recovery.CUT_KEY: saved[recovery.CUT_KEY]}

    def resume(self, driver: Any) -> dict[str, Any] | None:
        pointer = self.log.pointer()
        if pointer is None:
            return None
        import torch
        from yeto.rl.engine import recovery
        from yeto.rl.engine.resume import ROUND_CUTS

        cut_id = pointer["cut_id"]
        # A fresh machine: nothing of the run on this node's disk but what the bucket has.
        stage_root = Path(self.store.root)
        if (stage_root / ROUND_CUTS).exists():
            shutil.rmtree(stage_root / ROUND_CUTS)
        self.store.reload()
        payload = torch.load(stage_root / ROUND_CUTS / cut_id / "trainer.pt", weights_only=True)
        engine = driver.trainer.engine
        for name, value in payload["tensors"].items():
            engine.tensors[name].copy_(value)
        for name, value in payload["moments"].items():
            engine.moments[name].copy_(value)
        engine.scheduler_step = int(payload["step"])
        for rank, shard in sorted(pointer["shards"].items()):
            body = self.store.objects.read(shard["key"])
            if body is None or hashlib.sha256(body).hexdigest() != shard["sha256"]:
                raise RuntimeError(f"cut {cut_id}: shard of rank {rank} ({shard['key']}) is missing or "
                                   "differs in the bucket")
        driver.local_step = int(pointer["local_step"])
        self.buffer.reset_to(int(pointer[recovery.CUT_KEY]["buffer_watermark"] or 0))
        self.resumed = pointer
        log(f"restored cut {cut_id} from the bucket (step {engine.scheduler_step}, "
            f"{len(pointer['shards'])} shard(s) checked)")
        return {"next_rollout_id": int(pointer["next_rollout_id"]), **pointer}


class RunToTheEnd:
    """The command channel of an incarnation nobody commands: one RUN to the end
    of the data (``trainer_remote.CommandReader`` shape)."""

    def __init__(self) -> None:
        self.sent = False

    def poll(self) -> list[Any]:
        from yeto.rl.engine.trainer_remote import RUN, Command

        if self.sent:
            return []
        self.sent = True
        return [Command(seq=1, verb=RUN, rounds=None)]


def wait_for_batch(buffer: Any, batch: int, timeout_s: float = SAMPLE_WAIT_S) -> None:
    deadline = time.time() + timeout_s
    while True:
        left = buffer.backlog(0)  # refreshes the index: samples after the read cursor, data end
        if left.samples >= batch:
            return
        if left.data_ended or time.time() > deadline:
            raise RuntimeError(f"[bm-sim-learner] {left.samples} sample(s) left (data ended: {left.data_ended}), "
                               f"a round needs {batch}")
        time.sleep(1.0)


class ReleasingPublisher:
    """The fake publisher, plus one policy release per published version."""

    def __init__(self, inner: Any, releases: Any) -> None:
        self.inner = inner
        self.releases = releases

    def publish(self, *args: Any, **kwargs: Any) -> Any:
        result = self.inner.publish(*args, **kwargs)
        version, digest = self.inner.engine.published
        if version > 0:
            self.releases.release(int(version), str(digest),
                                  lambda directory: (directory / "weights.bin").write_bytes(
                                      f"{version}:{digest}".encode()))
            log(f"policy release v{version} committed")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from yeto.rl import event_echo
    from yeto.rl.adapters.miles import island_entry

    args = island_entry.parse_args(argv)
    if args.rl_sample_role != "trainer" or not args.rl_single_island_no_sync:
        raise SystemExit(f"[bm-sim-learner] expected a no-sync trainer island, got role {args.rl_sample_role}")
    island_entry._check_single_island_no_sync(args)
    if args.rl_echo_events:
        event_echo.enable_echo()
    import torch
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.fake import fake_capabilities
    from yeto.rl.engine.fake_lifecycle import BufferFedFakeRollout, cuttable_fake_engine
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import store_for
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.engine.trainer_remote import INCARNATION_ENV, run_commanded_island

    expected = node_ips()
    alive = join_ray(expected)
    by_ip = {ip: node for node, ip in alive.items()}
    ranks = {rank: by_ip[ip] for rank, ip in enumerate(expected)}
    scheduled = INCARNATION_ENV in os.environ  # the trainer scheduler commands this incarnation
    incarnation = os.environ[INCARNATION_ENV] if scheduled else f"r{int(time.time())}"
    tape = EventTape(args.event_tape, 0)
    tape.append({"event": "bm_sim_ray_nodes", "nodes": expected, "incarnation": incarnation,
                 "node_envs": node_envs(ranks)})
    log(f"incarnation {incarnation}: Ray has the {len(expected)} inventory node(s) {expected}")

    store_path = os.path.expanduser(args.rl_resume_store)
    store = store_for(store_path)
    if not getattr(store, "staged", False):
        raise SystemExit(f"[bm-sim-learner] the island store is not staged: {store.describe()}")
    batch_size = int(args.groups_per_round)
    buffer = FileSampleBuffer(store, batch_size=batch_size)
    engine = cuttable_fake_engine(tensors={TENSOR: torch.zeros(1, 2)}, step_delta=1.0,
                                  groups=batch_size, samples_per_group=1)
    engine.rollout = BufferFedFakeRollout(engine, buffer)
    generate = engine.rollout.generate
    round_s = float(os.environ.get(ROUND_S_ENV) or 0.0)

    def generate_on_every_node(rollout_id: int, **kwargs: Any) -> Any:
        if not scheduled:
            wait_for_batch(buffer, batch_size)
        out = generate(rollout_id, **kwargs)
        collective(ranks)
        if round_s:
            time.sleep(round_s)
        left = None if scheduled else buffer.backlog(0)
        if left is not None and left.data_ended and left.samples < batch_size:
            # the last round (num_rollout counts the rounds of this process): the driver drains after it
            sync.num_rollout = rollout_id + 1 - int(getattr(sync, "resumed_at", 0) or 0)
        return out

    engine.rollout.generate = generate_on_every_node
    engine.publisher = ReleasingPublisher(engine.publisher, PolicyReleaseStore(store))
    sync = LocalOnlySync(1_000_000)
    sync.round_cuts = SimRoundCuts(store, buffer, store_path=store_path, ranks=ranks, incarnation=incarnation,
                                   every_round=not scheduled)
    driver = IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                          policy_state=engine.policy_state, publisher=engine.publisher,
                          placement=engine.placement, algorithm=AlgorithmSpec(),
                          capabilities=fake_capabilities("miles-like"), sync=sync, events=tape)
    if scheduled:
        handle = run_commanded_island(driver, store=store)
    else:
        from yeto.rl.engine.trainer_remote import IslandTrainerAgent, StoreCutLog

        cuts = StoreCutLog(store.objects)
        resume_from = cuts.latest()  # a relaunch continues from the newest cut
        log(f"incarnation {incarnation}: resume from {resume_from}")
        handle = IslandTrainerAgent(driver, commands=RunToTheEnd(), incarnation=0, resume_from=resume_from,
                                    emit=driver.emit, cuts=cuts,
                                    watermark=sync.round_cuts.buffer_watermark, poll_s=1.0).run()
    log(f"incarnation {incarnation} ended: {handle.leave_reason} (version {handle.trained_version})")
    tape.append({"event": event_echo.FINALIZED_EVENT})
    return 0


if __name__ == "__main__":
    sys.exit(main())
