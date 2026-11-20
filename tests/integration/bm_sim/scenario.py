"""bm-sim head: drive the ``ssh:`` trainer island of the real launcher on the
simulated bare-metal cluster (:mod:`.cluster`) and check every stage.

Runs on the head container of the cluster (``run.py`` starts it there). The
launcher's own code builds the island: the CLI parser and ``_prepare_rl_args``,
``make_miles_island_task`` (the island's run script), ``island_spec_from_inventory``
(the ``SshTrainerSpec`` from ``--rl-node-inventory``) and the trainer loop
(``run_with_trainer_loop`` -> ``ssh_binding`` -> ``SshTrainerIslands`` over real
``ssh``). Two things are swapped, both by :func:`patch_task` and nothing else:

* the island's setup script (the Miles / SGLang source checkout of the GPU
  image; the node image is the runtime here) becomes a no-op;
* the learner module of the rank-0 learner line becomes
  :mod:`.fake_learner` (same argv, same environment).

The fleet (the samplers) is :class:`FakeFleet`: it writes groups into the run's
store by object API, like the NC sampler pool, and marks the data end.

Scenarios (``--scenario``):

* ``happy``: inventory -> preflight (uname, card tool, docker pull on docker
  nodes) -> start on every node -> Ray join with the inventory addresses ->
  training rounds -> idle stop with a cut (shards pushed from every node's stage)
  and policy releases in the bucket -> a second incarnation restores the cut from
  the bucket -> data end -> FINISHED -> nothing left running on any node;
* ``node-loss``: a worker node is killed while the island runs (its container
  stops: power loss); the island must end on every node, the loop must count the
  failure, and once the node is back a new incarnation finishes the run;
* ``preflight``: an inventory that does not match the node (card count) and a
  docker pull that fails with the overlay "not a directory" error are refused
  before any start.

Every check is a :class:`Check` row in the JSON report (``--report``); the exit
code is 0 only when every check passed.
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable

LEARNER_MODULE = "yeto.rl.adapters.miles.island_entry"
FAKE_LEARNER_MODULE = "tests.integration.bm_sim.fake_learner"
SETUP_STUB = ": # bm-sim: the node image is the trainer runtime (no Miles/SGLang source setup)"
WORKDIR = "/opt/yeto"  # the yeto checkout on every node (read-only mount)
REPO = Path(__file__).resolve().parents[3]  # the checkout the head runs from
BATCH = 12  # --rollout-batch-size: samples per trainer round (divisible by the DP of 2 or 3 nodes x 2 cards)
MAX_AGE = 2  # --rl-max-policy-age
Q_ON = MAX_AGE + 1  # batches of backlog that start the trainer (S1Scheduler.q_on without alpha)
NODE_CONTAINER = "bm-sim-n{rank}"
SIM_IMAGE = "ghcr.io/example/bm-sim-trainer@sha256:" + "1" * 64  # what the docker shim "pulls" and "runs"
NETWORK = "bm-sim"  # cluster.NETWORK


def log(*parts: Any) -> None:
    print(time.strftime("%H:%M:%S"), "[bm-sim]", *parts, flush=True)


# ----------------------------------------------------------------------------- report
@dataclasses.dataclass
class Check:
    scenario: str
    name: str
    ok: bool
    detail: str = ""


class Report:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.checks: list[Check] = []
        self.notes: dict[str, Any] = {}

    def check(self, scenario: str, name: str, ok: bool, detail: Any = "") -> bool:
        row = Check(scenario, name, bool(ok), detail if isinstance(detail, str) else json.dumps(detail, default=str))
        self.checks.append(row)
        log(("PASS" if ok else "FAIL"), f"{scenario}: {name}", "" if ok else row.detail[:2000])
        self.save()
        return bool(ok)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def save(self) -> None:
        if self.path is not None:
            self.path.write_text(json.dumps({"ok": self.ok, "checks": [dataclasses.asdict(c) for c in self.checks],
                                             "notes": self.notes}, indent=2, default=str))


# ------------------------------------------------------------------- launcher pieces
def sky_stub() -> types.SimpleNamespace:
    """``make_miles_island_task`` builds a ``sky.Task``; the head needs no SkyPilot
    for an ``ssh:`` island, so a plain record object stands in for it."""

    class _Record:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

        def set_resources(self, resources: Any) -> None:
            self.resources = resources

        def set_storage_mounts(self, mounts: Any) -> None:
            self.storage_mounts = mounts

    return types.SimpleNamespace(Task=_Record, Resources=_Record, Storage=_Record,
                                 StorageMode=types.SimpleNamespace(MOUNT="MOUNT", COPY="COPY"))


def install_sky_stub() -> None:
    """Puts :func:`sky_stub` in ``sys.modules`` for the rest of the process (the
    bm-sim head is its own process). In-process tests install the stub themselves
    with ``monkeypatch.setitem(sys.modules, "sky", sky_stub())`` so it is undone
    after the test; with a ``sky`` already imported this is a no-op."""
    if "sky" not in sys.modules:
        sys.modules["sky"] = sky_stub()


def launcher_argv(*, inventory: Path, store_uri: str, nodes: int, cards: int, run_id: str,
                  extra: list[str] = ()) -> list[str]:
    """The ``yeto launch`` argv of a K3.1-form trainer island on the SSH nodes
    (scheduler form: the head starts and stops it from the buffer backlog).

    The code reaches the nodes as the launcher sends it: the run code pack in the
    bucket (``--rl-code-pack auto`` on an ``s3://`` store, packed from git by
    :func:`build_args` / ``launcher.run``). A tree without its git dir (a worktree
    mounted alone) runs the nodes' ``--rl-ssh-trainer-workdir`` checkout instead."""
    code = [] if git_checkout(REPO) else ["--rl-code-pack", "off"]
    return [
        "--gpu", f"ssh:{nodes}x{cards}xb300@bm-sim",
        "--model", "/data/models/bm-sim", "--data", "org/bm-sim-prompts",
        "--training-mode", "rl", "--tuning", "full", "--inner-lr", "1e-6",
        "--total-steps", "1000", "--rollout-batch-size", str(BATCH), "--n-samples-per-prompt", "1",
        "--rollout-max-response-len", "64", "--local-rl-rounds-per-sync", "1",
        "--reward-function", "yeto.rl.numeric_answer_reward:score", "--trust-remote-code",
        "--rl-single-island-no-sync", "--rl-sample-role", "trainer", "--controller", "local",
        "--rl-max-policy-age", str(MAX_AGE), "--rl-algorithm-spec", str(REPO / "examples/k31/algorithm-spec.async-age2-tis.json"),
        "--rl-cut-every", "1", "--rl-sample-wait-s", "30",
        "--rl-checkpoint-store", store_uri,
        "--rl-trainer-scheduler", "s1", "--rl-trainer-island", "0",
        "--rl-trainer-scheduler-interval", "1", "--rl-trainer-command-poll-seconds", "1",
        "--rl-trainer-start-seconds", "15", "--rl-trainer-start-seconds-fixed",
        "--rl-trainer-ready-seconds", "240", "--rl-trainer-drain-seconds", "300",
        "--rl-trainer-max-failures", "4",
        "--rl-node-inventory", str(inventory), "--rl-ssh-trainer-workdir", WORKDIR,
        "--cluster-prefix", run_id, *code, *extra,
    ]


SCHEDULER_FLAGS = ("--rl-trainer-scheduler", "--rl-trainer-island", "--rl-trainer-scheduler-interval",
                   "--rl-trainer-command-poll-seconds", "--rl-trainer-start-seconds", "--rl-trainer-ready-seconds",
                   "--rl-trainer-drain-seconds", "--rl-trainer-max-failures")


def resident_argv(**kwargs: Any) -> list[str]:
    """The K3.1 trainer-only form: no trainer scheduler, the fleet keeps the
    ``ssh:`` island up to the end of the data (``ssh_resident``)."""
    argv = launcher_argv(**kwargs)
    out, i = [], 0
    while i < len(argv):
        if argv[i] in SCHEDULER_FLAGS:
            i += 2
        elif argv[i] == "--rl-trainer-start-seconds-fixed":
            i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


def git_checkout(path: Path) -> bool:
    done = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.returncode == 0


def build_args(argv: list[str]) -> Any:
    install_sky_stub()
    from yeto import launch_preflight
    from yeto.cli import parse_args
    from yeto.launch import trainer_loop
    from yeto.launch.code_pack import prepare_run_code_pack
    from yeto.launcher import _prepare_rl_args

    args = parse_args(argv)
    # what `yeto launch` resolves before the islands are built (Hub revisions, hashes)
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    prepare_run_code_pack(args)  # the run code pack in the bucket, as launcher.run (no-op when off)
    launch_preflight.pre_cloud_checks(args, threads=False)  # as launcher.run, before any island
    trainer_loop.check(args)
    return args


def patch_task(task: Any) -> Any:
    """The two bm-sim substitutions (module doc); refuses a run script whose
    learner line is not the one the harness knows."""
    run = str(task.run)
    marker = f"python3 -m {LEARNER_MODULE} "
    if run.count(marker) != 1:
        raise RuntimeError(f"the island run script has {run.count(marker)} learner line(s) "
                           f"`{marker.strip()}`; bm-sim swaps exactly one")
    task.run = run.replace(marker, f"python3 -m {FAKE_LEARNER_MODULE} ")
    task.setup = SETUP_STUB
    return task


def build_island(args: Any, events_dir: Path) -> tuple[Any, Any]:
    """(TrainerIsland, its task) as ``launcher.run`` builds them for a held ``ssh:`` island."""
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch import trainer_loop
    from yeto.launch.rl_island_task import make_miles_island_task

    specs = parse_gpu_spec(args.gpu)
    held = trainer_loop.held_island(args)
    if held is None:
        raise RuntimeError("the bm-sim launch holds no trainer island")
    name = launcher.learner_cluster_names(args.cluster_prefix, specs)[held]
    task = launcher._island_task(make_miles_island_task, args, specs[held], held, len(specs), "none",
                                 launcher._island_overrides(args, specs), launcher._island_role(args, specs, held))
    task = patch_task(task)
    island = trainer_loop.TrainerIsland(name, specs[held].cloud, spec=specs[held], events_dir=events_dir,
                                        ssh=launcher._ssh_trainer_spec(args, specs[held], task))
    return island, task


# ------------------------------------------------------------------------- the fleet
class FakeFleet:
    """The samplers: ``phases`` is a list of ``(samples, wait)``: write that many
    samples (with the newest released version), then ``wait(fleet)`` (blocks until
    the head may go on). The data ends after the last phase."""

    def __init__(self, objects: Any, phases: list[tuple[int, Callable[["FakeFleet"], None] | None]], *,
                 name: str = "s0") -> None:
        from yeto.rl.engine.sample_buffer import FileSampleBuffer

        self.name = name  # sample ids are <name>-<n>: a restarted fleet takes another name
        self.objects = objects
        self.buffer = FileSampleBuffer(objects, batch_size=BATCH)
        self.phases = phases
        self.produced = 0
        self.aborted: BaseException | None = None
        self.stop = threading.Event()

    def latest_release(self) -> int:
        raw = self.objects.read("policy-releases/LATEST.json")
        return 0 if raw is None else int(json.loads(raw)["version"])

    def produce(self, n: int) -> None:
        from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

        version = self.latest_release()
        for _ in range(n):
            sample_id = f"{self.name}-{self.produced:05d}"
            ref = self.buffer.write_payload(sample_id, f"{sample_id}@v{version}".encode())
            self.buffer.append(SampleIndexEntry(sample_id, version, ref, tokens=6))
            self.produced += 1
        log(f"fleet: {n} sample(s) at v{version} (total {self.produced})")

    def run(self) -> list[int]:
        for samples, wait in self.phases:
            if self.stop.is_set():
                break
            self.produce(samples)
            if wait is not None:
                wait(self)
        self.buffer.mark_data_end()
        log("fleet: data end")
        if self.aborted is not None:
            raise self.aborted
        return [0]

    def abort(self, error: BaseException) -> None:
        log(f"fleet: aborted by the trainer loop: {error}")
        self.aborted = error
        self.stop.set()


def wait_until(what: str, predicate: Callable[[], Any], timeout_s: float, poll_s: float = 1.0) -> Any:
    deadline = time.time() + timeout_s
    while True:
        value = predicate()
        if value:
            return value
        if time.time() > deadline:
            raise TimeoutError(f"bm-sim: {what} not seen within {timeout_s:.0f} s")
        time.sleep(poll_s)


def loop_records(args: Any) -> list[dict[str, Any]]:
    from yeto.launch import trainer_loop
    from yeto.rl.engine.journal import read_journal

    return [r for r in read_journal(trainer_loop.journal_dir(args)) if r.get("kind") == "trainer_loop"]


# --------------------------------------------------------------- node-side probes
class Nodes:
    """SSH to the inventory nodes as the island does, and the docker host for
    power-loss injection (the head container holds the docker socket)."""

    def __init__(self, nodes: list[Any], docker_socket: str = "/var/run/docker.sock") -> None:
        self.nodes = nodes
        self.docker_socket = docker_socket

    def ssh(self, rank: int, script: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
        from yeto.launch.ssh_trainer_island import SshOps

        return SshOps().run(self.nodes[rank], script, timeout=timeout)

    def island_processes(self, rank: int) -> list[str]:
        """The processes of the island user on a node other than the probe's own
        ssh session (``ps``; a name filter would match the probe itself)."""
        done = self.ssh(rank, "ps -u \"$(id -un)\" -o pid=,ppid=,args= || true")
        own = set()
        rows = []
        for line in done.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            rows.append(parts)
        # the probe's own shell chain: sshd session -> bash -> ps
        for pid, ppid, args in rows:
            if args.startswith(("ps -u", "sshd:")) or "exec 3<&0" in args or args.startswith("bash -c"):
                own.add(pid)
        return [f"{pid} {args}" for pid, ppid, args in rows
                if pid not in own and ppid not in own and not args.startswith(("ps ", "sshd"))]

    def _docker(self, method: str, path: str, body: Any = None) -> tuple[int, bytes]:
        """One Docker Engine API call over the socket (no docker CLI on the head)."""
        conn = http.client.HTTPConnection("localhost", timeout=120)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.docker_socket)
        conn.sock = sock
        payload = None if body is None else json.dumps(body)
        conn.request(method, path, body=payload, headers={"Content-Type": "application/json"} if payload else {})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def power_off(self, rank: int) -> int:
        return self._docker("POST", f"/containers/{NODE_CONTAINER.format(rank=rank)}/kill")[0]

    def power_on(self, rank: int) -> int:
        return self._docker("POST", f"/containers/{NODE_CONTAINER.format(rank=rank)}/start")[0]

    def unplug(self, rank: int) -> int:
        """The node keeps running but leaves the cluster network (a partition)."""
        return self._docker("POST", f"/networks/{NETWORK}/disconnect",
                            {"Container": NODE_CONTAINER.format(rank=rank), "Force": True})[0]

    def plug(self, rank: int, ip: str) -> int:
        return self._docker("POST", f"/networks/{NETWORK}/connect",
                            {"Container": NODE_CONTAINER.format(rank=rank),
                             "EndpointConfig": {"IPAMConfig": {"IPv4Address": ip}}})[0]

    def console(self, rank: int, script: str) -> str:
        """Run ``script`` on the node as the island user without the network (the
        node's console: ``docker exec``)."""
        container = NODE_CONTAINER.format(rank=rank)
        status, data = self._docker("POST", f"/containers/{container}/exec",
                                    {"Cmd": ["bash", "-c", script], "User": "yeto", "AttachStdout": True,
                                     "AttachStderr": True})
        if status != 201:
            return f"exec create failed: {status} {data[:200]!r}"
        exec_id = json.loads(data)["Id"]
        _, raw = self._docker("POST", f"/exec/{exec_id}/start", {"Detach": False, "Tty": False})
        out, i = [], 0
        while i + 8 <= len(raw):  # the multiplexed stream: 8-byte header per frame
            size = int.from_bytes(raw[i + 4:i + 8], "big")
            out.append(raw[i + 8:i + 8 + size])
            i += 8 + size
        return b"".join(out).decode(errors="replace")


# ------------------------------------------------------------------------ scenarios
def store_env(store_env_file: Path) -> None:
    for line in store_env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def make_bucket(uri: str) -> None:
    import boto3

    from yeto.cloud.s3_store import ENDPOINT_ENV, KEY_ID_ENV, SECRET_ENV, parse_s3_uri

    bucket, _ = parse_s3_uri(uri)
    client = boto3.client("s3", endpoint_url=os.environ[ENDPOINT_ENV], aws_access_key_id=os.environ[KEY_ID_ENV],
                          aws_secret_access_key=os.environ[SECRET_ENV], region_name="us-east-1")
    try:
        client.create_bucket(Bucket=bucket)
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass


def jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def bucket_keys(objects: Any, prefix: str) -> list[str]:
    return sorted(objects.list(prefix))


def check_teardown(report: Report, scenario: str, nodes: Nodes, ranks: list[int]) -> None:
    for rank in ranks:
        left = wait_for_clean(nodes, rank, 60.0)
        report.check(scenario, f"node {rank}: nothing of the island left running", not left, left)


def wait_for_clean(nodes: Nodes, rank: int, timeout_s: float) -> list[str]:
    deadline = time.time() + timeout_s
    while True:
        left = nodes.island_processes(rank)
        if not left or time.time() > deadline:
            return left
        time.sleep(2.0)


def scenario_happy(report: Report, ctx: dict[str, Any], *, name: str = "happy",
                   inventory: Path | None = None) -> None:
    from yeto.launch import trainer_loop
    from yeto.launch.trainer_island import head_store_objects
    from yeto.rl.engine import recovery
    from yeto.rl.engine.trainer_remote import StoreCutLog

    run_id = f"bmsim-{name}-{int(time.time())}"
    store_uri = f"{ctx['store_root']}/{run_id}"
    args = build_args(launcher_argv(inventory=inventory or ctx["inventory"], store_uri=store_uri,
                                    nodes=ctx["nodes"], cards=ctx["cards"], run_id=run_id))
    events_dir = Path(ctx["work"]) / run_id / "events"
    island, task = build_island(args, events_dir)
    spec = island.ssh
    report.check(name, "inventory -> SshTrainerSpec (launcher path)",
                 [n.name for n in spec.nodes] == [f"n{k}" for k in range(ctx["nodes"])],
                 {"nodes": [n.name for n in spec.nodes], "addresses": spec.addresses(), "docker": spec.docker})
    objects = head_store_objects(store_uri)
    cuts = StoreCutLog(objects)

    def first_cut(fleet: FakeFleet) -> None:
        cut = wait_until("the idle-stop cut of incarnation 0", lambda: (
            (c := cuts.latest()) is not None and c.watermark == Q_ON * BATCH and c), 420.0, 2.0)
        log(f"cut {cut.cut_id} ({cut.trigger}, watermark {cut.watermark})")
        wait_until("incarnation 0 released", lambda: any(
            r.get("event") == "stopped" and r.get("incarnation") == 0 for r in loop_records(args)), 120.0)
        ctx.setdefault("cuts", {})[name] = cut

    fleet = FakeFleet(objects, [(Q_ON * BATCH, first_cut), (BATCH + 2, None)])
    started = time.time()
    try:
        result = trainer_loop.run_with_trainer_loop(args, fleet.run, island=island, abort=fleet.abort)
        report.check(name, "run_with_trainer_loop returned", result == [0], {"result": result})
    except BaseException as error:  # noqa: BLE001 - the report keeps it
        report.check(name, "run_with_trainer_loop returned", False, traceback.format_exc())
        if isinstance(error, KeyboardInterrupt):
            raise
    report.notes[f"{name}_seconds"] = round(time.time() - started, 1)
    records = loop_records(args)
    report.notes[f"{name}_journal"] = records
    stops = [(r.get("incarnation"), r.get("reason"), r.get("trigger"), r.get("released"))
             for r in records if r.get("event") == "stopped"]
    report.check(name, "incarnation 0 idle-stopped with a cut, incarnation 1 finished, both released",
                 stops == [(0, "STOPPED", recovery.SCHEDULED, True), (1, "FINISHED", recovery.MANUAL, True)], stops)
    starts = [r for r in records if r.get("event") == "started"]
    cut0 = ctx.get("cuts", {}).get(name)
    resume = [r.get("resume_from") or r.get("resume") for r in starts]
    report.check(name, "incarnation 1 resumed from the cut of incarnation 0",
                 len(starts) == 2 and cut0 is not None and cut0.cut_id in json.dumps(starts[1], default=str),
                 {"starts": starts, "cut0": None if cut0 is None else cut0.cut_id, "resume": resume})
    tape = [json.loads(line) for line in (events_dir / f"{island.name}-i0.jsonl").read_text().splitlines()] \
        if (events_dir / f"{island.name}-i0.jsonl").is_file() else []
    ray_nodes = [e for e in tape if e.get("event") == "bm_sim_ray_nodes"]
    report.check(name, "Ray: every inventory node joined at its inventory address",
                 bool(ray_nodes) and sorted(ray_nodes[0]["nodes"]) == sorted(spec.addresses()),
                 ray_nodes or "no bm_sim_ray_nodes event on the node-0 stream")
    rdma = spec.nodes[0].fabric in ("ib", "roce")
    envs = ray_nodes[0].get("node_envs", []) if ray_nodes else []
    want = [{"SKYPILOT_NODE_RANK": str(r), "BM_SIM_NODE": f"n{r}", "NCCL_IB_DISABLE": "0" if rdma else "1",
             "NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0", "YETO_DEVICE_FAMILY": "cuda",
             "YETO_RL_STORE_S3_STAGE": "~/.cache/yeto/store-stage",
             **({"NCCL_IB_HCA": "mlx5_0,mlx5_1"} if rdma else {})} for r in range(ctx["nodes"])]
    got = [{k: e["env"].get(k) for k in w} for e, w in zip(envs, want)]
    report.check(name, "every node's trainer ranks see the inventory env, the fabric defaults and the store",
                 got == want and len(envs) == ctx["nodes"] and all(e["store_secret"] for e in envs),
                 {"want": want, "got": envs})
    worker_logs = sorted(p.name for p in events_dir.glob(f"{island.name}-i*-node*.log"))
    report.check(name, "the worker nodes' streams reached the head", len(worker_logs) >= ctx["nodes"] - 1,
                 worker_logs)
    keys = bucket_keys(objects, "round-cuts/")
    for cut_id in sorted({k.split("/")[1] for k in keys}):
        shards = [k for k in keys if k.startswith(f"round-cuts/{cut_id}/shard-node")]
        report.check(name, f"cut {cut_id}: one shard per node in the bucket", len(shards) == ctx["nodes"], shards)
    releases = bucket_keys(objects, "policy-releases/")
    latest = fleet.latest_release()
    report.check(name, f"policy releases in the bucket, newest = trained version {Q_ON + 1}",
                 latest == Q_ON + 1 and any(k.endswith("LATEST.json") for k in releases),
                 {"latest": latest, "keys": releases[-6:]})
    if spec.docker:
        done = Nodes(list(spec.nodes)).ssh(0, "for f in /tmp/bm-sim-docker/*.json; do cat \"$f\"; echo; done")
        runs = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
        envs = runs[0]["env"] if runs else []
        report.check(name, "docker: the island ran in named host-network containers with the store env by -e",
                     bool(runs) and "YETO_RL_STORE_S3_URI" in envs and "YETO_TRAINER_INCARNATION" in envs,
                     runs[:1])
    check_teardown(report, name, Nodes(list(spec.nodes)), list(range(ctx["nodes"])))


def scenario_node_loss(report: Report, ctx: dict[str, Any], *, down_starts: int = 0) -> None:
    """A worker node loses power in the middle of a training round; it comes
    back after the loop tried ``down_starts`` more incarnations without it."""
    from yeto.launch import trainer_loop
    from yeto.launch.trainer_island import head_store_objects

    name = "node-loss" if not down_starts else "node-loss-long"
    run_id = f"bmsim-{name}-{int(time.time())}"
    store_uri = f"{ctx['store_root']}/{run_id}"
    args = build_args(launcher_argv(inventory=ctx["inventory"], store_uri=store_uri, nodes=ctx["nodes"],
                                    cards=ctx["cards"], run_id=run_id))
    events_dir = Path(ctx["work"]) / run_id / "events"
    island, _ = build_island(args, events_dir)
    # a slower trainer, so the node dies inside a round
    island = dataclasses.replace(island, ssh=dataclasses.replace(
        island.ssh, env={**island.ssh.env, "BM_SIM_ROUND_S": "4"}))
    nodes = Nodes(list(island.ssh.nodes))
    victim = ctx["nodes"] - 1
    objects = head_store_objects(store_uri)
    seen: dict[str, Any] = {}

    def kill_while_training(fleet: FakeFleet) -> None:
        wait_until("incarnation 0 trained a round", lambda: fleet.latest_release() >= 1, 420.0, 0.5)
        log(f"power off node {victim}")
        seen["power_off"] = nodes.power_off(victim)
        seen["power_off_at"] = time.time()
        wait_until("the loop to end the incarnation with the dead node", lambda: any(
            r.get("event") == "stopped" and r.get("incarnation") == 0 for r in loop_records(args)), 600.0)
        seen["noticed_after_s"] = round(time.time() - seen["power_off_at"], 1)
        seen["survivor_left"] = wait_for_clean(nodes, 0, 60.0)
        if down_starts:
            time.sleep(down_starts * 60.0)  # the node stays down while the loop wants to start again
            seen["down_for_s"] = round(time.time() - seen["power_off_at"], 1)
        log(f"power on node {victim}")
        seen["power_on"] = nodes.power_on(victim)
        wait_until(f"node {victim} back on ssh", lambda: nodes.ssh(victim, "true", 30.0).returncode == 0, 120.0, 3.0)

    fleet = FakeFleet(objects, [(2 * Q_ON * BATCH, kill_while_training), (BATCH, None)])
    try:
        result = trainer_loop.run_with_trainer_loop(args, fleet.run, island=island, abort=fleet.abort)
        report.check(name, "the run finished after the node came back", result == [0], {"result": result})
    except BaseException as error:  # noqa: BLE001
        report.check(name, "the run finished after the node came back", False, traceback.format_exc())
        if isinstance(error, KeyboardInterrupt):
            raise
    records = loop_records(args)
    report.notes[f"{name}_journal"] = records
    report.notes[f"{name}_probe"] = seen
    stops = [(r.get("incarnation"), r.get("reason"), r.get("released"), r.get("error"))
             for r in records if r.get("event") == "stopped"]
    first = stops[0] if stops else None
    report.check(name, "the incarnation with the dead node ended as a failure (not a clean stop)",
                 first is not None and first[0] == 0 and first[1] not in ("STOPPED", "FINISHED"), stops)
    report.check(name, "the surviving node's island processes were stopped by the release",
                 not seen.get("survivor_left"), seen.get("survivor_left"))
    report.check(name, "a later incarnation finished the run",
                 bool(stops) and stops[-1][1] == "FINISHED", stops)
    if down_starts:
        report.check(name, "the start after the failure waited for the node instead of failing on it",
                     len(stops) == 2 and stops[1][0] == 1 and stops[1][1] == "FINISHED", {"stops": stops, "probe": seen})
    check_teardown(report, name, nodes, list(range(ctx["nodes"])))


def scenario_partition(report: Report, ctx: dict[str, Any]) -> None:
    """A worker node leaves the cluster network in the middle of a round and keeps
    running (a switch port or cable fault, not a power loss): the release cannot
    reach it; the next start must stop what it kept running before it starts."""
    from yeto.launch import trainer_loop
    from yeto.launch.trainer_island import head_store_objects

    name = "node-partition"
    run_id = f"bmsim-partition-{int(time.time())}"
    store_uri = f"{ctx['store_root']}/{run_id}"
    args = build_args(launcher_argv(inventory=ctx["inventory"], store_uri=store_uri, nodes=ctx["nodes"],
                                    cards=ctx["cards"], run_id=run_id))
    island, _ = build_island(args, Path(ctx["work"]) / run_id / "events")
    island = dataclasses.replace(island, ssh=dataclasses.replace(
        island.ssh, env={**island.ssh.env, "BM_SIM_ROUND_S": "4"}))
    nodes = Nodes(list(island.ssh.nodes))
    victim = ctx["nodes"] - 1
    objects = head_store_objects(store_uri)
    seen: dict[str, Any] = {}

    def cut_off_while_training(fleet: FakeFleet) -> None:
        wait_until("incarnation 0 trained a round", lambda: fleet.latest_release() >= 1, 420.0, 0.5)
        log(f"unplug node {victim}")
        seen["unplug"] = nodes.unplug(victim)
        started = time.time()
        wait_until("the loop to end the incarnation with the cut-off node", lambda: any(
            r.get("event") == "stopped" and r.get("incarnation") == 0 for r in loop_records(args)), 600.0)
        seen["noticed_after_s"] = round(time.time() - started, 1)
        seen["victim_processes_while_cut_off"] = nodes.console(victim, "ps -u yeto -o pid=,args=")
        time.sleep(20.0)
        log(f"plug node {victim} back")
        seen["plug"] = nodes.plug(victim, ctx["ips"][f"n{victim}"])
        wait_until(f"node {victim} back on ssh", lambda: nodes.ssh(victim, "true", 30.0).returncode == 0, 120.0, 3.0)

    fleet = FakeFleet(objects, [(2 * Q_ON * BATCH, cut_off_while_training), (BATCH, None)])
    try:
        result = trainer_loop.run_with_trainer_loop(args, fleet.run, island=island, abort=fleet.abort)
        report.check(name, "the run finished after the node came back", result == [0], {"result": result})
    except BaseException as error:  # noqa: BLE001
        report.check(name, "the run finished after the node came back", False, traceback.format_exc())
        if isinstance(error, KeyboardInterrupt):
            raise
    records = loop_records(args)
    report.notes[f"{name}_journal"] = records
    report.notes[f"{name}_probe"] = seen
    stops = [(r.get("incarnation"), r.get("reason"), r.get("released")) for r in records
             if r.get("event") == "stopped"]
    report.check(name, "the incarnation with the cut-off node failed and its release was not confirmed",
                 bool(stops) and stops[0][0] == 0 and stops[0][1] not in ("STOPPED", "FINISHED")
                 and stops[0][2] is False, stops)
    report.check(name, "a later incarnation finished the run", bool(stops) and stops[-1][1] == "FINISHED", stops)
    check_teardown(report, name, nodes, list(range(ctx["nodes"])))


def head_restart_child(ctx: dict[str, Any], run_id: str, phase: str) -> int:
    """One launcher process of the head-restart scenario (``--head-child``)."""
    from yeto.launch import trainer_loop
    from yeto.launch.trainer_island import head_store_objects

    store_uri = f"{ctx['store_root']}/{run_id}"
    args = build_args(launcher_argv(inventory=ctx["inventory"], store_uri=store_uri, nodes=ctx["nodes"],
                                    cards=ctx["cards"], run_id=run_id))
    island, _ = build_island(args, Path(ctx["work"]) / run_id / "events")
    island = dataclasses.replace(island, ssh=dataclasses.replace(
        island.ssh, env={**island.ssh.env, "BM_SIM_ROUND_S": "4"}))
    objects = head_store_objects(store_uri)
    if phase == "first":  # killed by the parent while the trainer runs
        fleet = FakeFleet(objects, [(2 * Q_ON * BATCH, lambda _fleet: time.sleep(3600))], name="a")
    else:
        fleet = FakeFleet(objects, [(BATCH, None)], name="b")
    result = trainer_loop.run_with_trainer_loop(args, fleet.run, island=island, abort=fleet.abort)
    return 0 if result == [0] else 1


def scenario_head_restart(report: Report, ctx: dict[str, Any]) -> None:
    """The launcher process dies while the island trains (head crash, ssh sessions
    gone); the island keeps running on the nodes. A restarted launcher of the same
    run must release that orphan incarnation by its journal before it starts
    another, and finish the run."""
    from yeto.launch.trainer_island import head_store_objects

    name = "head-restart"
    run_id = f"bmsim-headrestart-{int(time.time())}"
    objects = head_store_objects(f"{ctx['store_root']}/{run_id}")
    child = [sys.executable, "-u", "-m", "tests.integration.bm_sim.scenario", "--work", ctx["work"],
             "--head-child", run_id]
    first = subprocess.Popen([*child, "first"])
    nodes: Nodes | None = None
    try:
        wait_until("the first head's trainer to release a policy", lambda: FakeFleet(objects, []).latest_release() >= 1,
                   420.0, 0.5)
    finally:
        first.kill()  # SIGKILL: no stop, no release, no journal close
        first.wait()
    log("head killed")
    from yeto.launch.node_inventory import load_inventory

    nodes = Nodes(list(load_inventory(Path(ctx["inventory"]))))
    time.sleep(3.0)
    orphans = {rank: nodes.island_processes(rank) for rank in range(ctx["nodes"])}
    report.check(name, "the island kept running on every node after the head died",
                 all(orphans.values()), orphans)
    second = subprocess.run([*child, "second"], timeout=1200)
    report.check(name, "the restarted head finished the run", second.returncode == 0, {"rc": second.returncode})

    class _Args:  # loop_records reads the journal of the run by its prefix
        cluster_prefix = run_id

    records = loop_records(_Args)
    report.notes[f"{name}_journal"] = records
    orphaned = [r for r in records if r.get("event") == "orphaned"]
    stops = [(r.get("incarnation"), r.get("reason")) for r in records if r.get("event") == "stopped"]
    report.check(name, "the restarted head released incarnation 0 as an orphan, then a new incarnation finished",
                 bool(orphaned) and orphaned[0].get("incarnation") == 0 and bool(stops) and stops[-1][1] == "FINISHED"
                 and stops[-1][0] != 0, {"orphaned": orphaned, "stops": stops})
    check_teardown(report, name, nodes, list(range(ctx["nodes"])))


class LauncherProcessFakes:
    """What ``launcher.run`` needs patched on the bm-sim head, nothing else: the
    Hub resolution of ``prepare_launch_args`` (the head is offline; the revisions
    are fixed as in :func:`build_args`) and the island task factory, whose task
    gets the two bm-sim substitutions (:func:`patch_task`)."""

    def __enter__(self) -> "LauncherProcessFakes":
        install_sky_stub()
        from yeto import launcher

        self.launcher = launcher
        self.saved = (launcher.prepare_launch_args, launcher.make_miles_island_task)
        factory = launcher.make_miles_island_task

        def prepare(args: Any) -> Any:
            args.model_revision, args.data_revision = "a" * 40, "b" * 40
            args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
            return launcher._prepare_rl_args(args)

        def task_factory(*a: Any, **k: Any) -> Any:
            return patch_task(factory(*a, **k))

        launcher.prepare_launch_args, launcher.make_miles_island_task = prepare, task_factory
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.launcher.prepare_launch_args, self.launcher.make_miles_island_task = self.saved


def resident_args(ctx: dict[str, Any], run_id: str, inventory: Path) -> Any:
    from yeto.cli import parse_args

    args = parse_args(resident_argv(inventory=inventory, store_uri=f"{ctx['store_root']}/{run_id}",
                                    nodes=ctx["nodes"], cards=ctx["cards"], run_id=run_id))
    for name, value in (("keep", False), ("recover_timeout", 600.0), ("controller_poll", 5.0)):
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    args.controller_poll = 5.0
    return args


def scenario_resident(report: Report, ctx: dict[str, Any], *, name: str = "resident",
                      inventory: Path | None = None, node_loss: bool = False) -> None:
    """``launcher.run`` of the K3.1 trainer-only form: the fleet starts the resident
    ``ssh:`` island, the samplers (:class:`FakeFleet`, a thread) fill the buffer,
    the trainer ends at the data end, the launcher verifies the teardown.

    ``node_loss``: a slower trainer; the last node loses power in the middle of a
    round and stays off while the fleet wants to relaunch the island; the run must
    still finish, the relaunched incarnation restoring the newest cut."""
    import yaml

    from yeto import launcher
    from yeto.launch.node_inventory import load_inventory
    from yeto.launch.trainer_island import head_store_objects
    from yeto.rl.engine.trainer_remote import StoreCutLog

    run_id = f"bmsim-{name}-{int(time.time())}"
    run_dir = Path(os.environ.get("YETO_RUNS_DIR", "/work/runs")) / run_id
    inventory = inventory or Path(ctx["inventory"])
    if node_loss:
        raw = yaml.safe_load(inventory.read_text())
        for node in raw["nodes"]:
            node["env"] = {**node.get("env", {}), "BM_SIM_ROUND_S": "4"}  # the node dies inside a round
        inventory = Path(ctx["work"]) / f"inventory-{name}.yaml"
        inventory.write_text(yaml.safe_dump(raw))
    args = resident_args(ctx, run_id, inventory)
    objects = head_store_objects(f"{ctx['store_root']}/{run_id}")
    nodes = Nodes(list(load_inventory(inventory)))
    victim = ctx["nodes"] - 1
    seen: dict[str, Any] = {}

    def first_batches(fleet: FakeFleet) -> None:
        if not node_loss:
            wait_until("the resident trainer to train the first batches",
                       lambda: fleet.latest_release() >= Q_ON, 420.0, 1.0)
            return
        wait_until("the resident trainer to train a round", lambda: fleet.latest_release() >= 1, 420.0, 0.5)
        log(f"power off node {victim}")
        seen["power_off"] = nodes.power_off(victim)
        seen["power_off_at"] = time.time()
        seen["release_at_power_off"] = fleet.latest_release()
        seen["survivor_left"] = wait_for_clean(nodes, 0, 400.0)  # the fleet failed and released the island
        seen["survivor_clean_after_s"] = round(time.time() - seen["power_off_at"], 1)
        wait_until("the fleet to see the island lost", lambda: any(
            r.get("event") == "island_lost" for r in jsonl(run_dir / "fleet.jsonl")), 600.0, 2.0)
        seen["lost_after_s"] = round(time.time() - seen["power_off_at"], 1)
        time.sleep(30.0)  # down while the fleet recovers: no relaunch may start without the node
        log(f"power on node {victim}")
        seen["power_on"] = nodes.power_on(victim)
        wait_until(f"node {victim} back on ssh", lambda: nodes.ssh(victim, "true", 30.0).returncode == 0,
                   120.0, 3.0)
        seen["down_for_s"] = round(time.time() - seen["power_off_at"], 1)
        wait_until("the relaunched trainer to train the first batches",
                   lambda: fleet.latest_release() >= Q_ON, 600.0, 1.0)

    fleet = FakeFleet(objects, [(Q_ON * BATCH, first_batches), (2 * BATCH, None)])
    samplers = threading.Thread(target=fleet.run, name="bm-sim-samplers", daemon=True)
    started = time.time()
    with LauncherProcessFakes():
        samplers.start()
        try:
            code = launcher.run(args)
        except BaseException as error:  # noqa: BLE001
            code = repr(error)
            report.notes[f"{name}_error"] = traceback.format_exc()
    samplers.join(timeout=60)
    report.notes[f"{name}_seconds"] = round(time.time() - started, 1)
    report.check(name, "launcher.run of the trainer-only launch exits 0", code == 0, {"exit": code})
    latest = fleet.latest_release()
    report.check(name, f"the resident trainer trained every batch (release v{Q_ON + 2})", latest == Q_ON + 2,
                 {"latest": latest})
    pointer = StoreCutLog(objects).pointer() or {}
    keys = bucket_keys(objects, "round-cuts/")
    shards = [k for k in keys if f"/{pointer.get('cut_id')}/" in k and "/shard-node" in k]
    report.check(name, "its newest cut has one shard per node in the bucket", len(shards) == ctx["nodes"],
                 {"cut": pointer.get("cut_id"), "keys": keys})
    if node_loss:
        report.notes[f"{name}_probe"] = seen
        report.check(name, "the power loss failed the island and released the surviving node",
                     not seen.get("survivor_left"), seen)
        jobs = [json.loads(path.read_text()) for path in sorted(run_dir.glob("ssh-islands/*-jobs.json"))]
        report.notes[f"{name}_jobs"] = jobs
        report.check(name, "the fleet relaunched the island once, as incarnation 1",
                     len(jobs) == 1 and sorted(jobs[0]) == ["0", "1"], jobs)
        ready = [(r.get("start_version"), r.get("watermark")) for path in sorted(run_dir.glob("events/*.jsonl"))
                 for r in jsonl(path) if r.get("event") == "rl_trainer_lifecycle" and r.get("phase") == "ready"]
        report.check(name, "the relaunched incarnation restored the cut it had at the power loss",
                     len(ready) == 2 and ready[0] == (0, 0) and ready[1][0] == seen.get("release_at_power_off")
                     and ready[1][1] == BATCH * ready[1][0], {"ready": ready, "probe": seen})
    check_teardown(report, name, nodes, list(range(ctx["nodes"])))


def scenario_preflight(report: Report, ctx: dict[str, Any]) -> None:
    import yaml

    from yeto.launch.node_inventory import load_inventory
    from yeto.launch.ssh_trainer_island import SshIslandError, SshTrainerIslands, SshTrainerSpec

    name = "preflight"
    raw = yaml.safe_load(Path(ctx["inventory"]).read_text())
    for node in raw["nodes"]:
        node["cards_per_node"] = ctx["cards"] * 2  # the node reports ctx["cards"]
    bad = Path(ctx["work"]) / "inventory-wrong-cards.yaml"
    bad.write_text(yaml.safe_dump(raw))
    nodes = tuple(load_inventory(bad))
    spec = SshTrainerSpec(nodes=nodes, run_id="bmsim-preflight", workdir=WORKDIR, run_script="true")
    islands = SshTrainerIslands(spec, name="t", jobs_path=Path(ctx["work"]) / "preflight-jobs.json")
    try:
        islands.preflight()
        report.check(name, "a card-count mismatch is refused before any start", False, "preflight passed")
    except Exception as error:  # noqa: BLE001
        report.check(name, "a card-count mismatch is refused before any start",
                     "card" in str(error).lower() or "count" in str(error).lower(), repr(error))
    docker_inventory = Path(ctx["work"]) / "inventory-docker.yaml"
    nodes = tuple(load_inventory(docker_inventory))
    probe = Nodes(list(nodes))
    message = ("failed to register layer: openat opt/sglang/python/sglang-0.5.24.dist-info/INSTALLER: "
               "not a directory")
    probe.ssh(0, f"printf %s {shlex.quote(message)} > /tmp/bm-sim-pull-fail")
    spec = SshTrainerSpec(nodes=nodes, run_id="bmsim-preflight", workdir=WORKDIR, run_script="true",
                          image=SIM_IMAGE)
    islands = SshTrainerIslands(spec, name="t", jobs_path=Path(ctx["work"]) / "preflight-jobs2.json")
    try:
        islands.preflight()
        report.check(name, "an unpackable image layer is refused at the preflight pull", False, "preflight passed")
    except SshIslandError as error:
        report.check(name, "an unpackable image layer is refused at the preflight pull",
                     "make_layer" in str(error) and "not a directory" in str(error), str(error))
    finally:
        probe.ssh(0, "rm -f /tmp/bm-sim-pull-fail")


SCENARIOS = {"preflight": scenario_preflight, "happy": scenario_happy, "node-loss": scenario_node_loss}


def scenario_happy_docker(report: Report, ctx: dict[str, Any]) -> None:
    scenario_happy(report, ctx, name="happy-docker", inventory=Path(ctx["work"]) / "inventory-docker.yaml")


SCENARIOS["happy-docker"] = scenario_happy_docker
SCENARIOS["node-partition"] = scenario_partition
SCENARIOS["head-restart"] = scenario_head_restart
SCENARIOS["resident"] = scenario_resident
SCENARIOS["resident-docker"] = lambda report, ctx: scenario_resident(
    report, ctx, name="resident-docker", inventory=Path(ctx["work"]) / "inventory-docker.yaml")
SCENARIOS["resident-node-loss"] = lambda report, ctx: scenario_resident(
    report, ctx, name="resident-node-loss", node_loss=True)
SCENARIOS["node-loss-long"] = lambda report, ctx: scenario_node_loss(report, ctx, down_starts=1)


DEFAULT_SCENARIOS = ("preflight", "happy", "happy-docker", "node-loss", "node-loss-long", "node-partition",
                     "head-restart", "resident", "resident-docker", "resident-node-loss")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", default="/work", help="the cluster's work dir (inventory, store env, ssh)")
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--head-child", nargs=2, metavar=("RUN_ID", "PHASE"), default=None,
                        help=argparse.SUPPRESS)  # one launcher process of the head-restart scenario
    opts = parser.parse_args(argv)
    work = Path(opts.work)
    cluster = json.loads((work / "cluster.json").read_text())
    store_env(work / "store.env")
    make_bucket(cluster["store_root"])
    ctx = {**cluster, "work": str(work), "inventory": str(work / "inventory.yaml")}
    if opts.head_child:
        return head_restart_child(ctx, *opts.head_child)
    report = Report(Path(opts.report) if opts.report else work / "report.json")
    report.notes["cluster"] = cluster
    for scenario in opts.scenario or DEFAULT_SCENARIOS:
        log(f"=== scenario {scenario}")
        try:
            SCENARIOS[scenario](report, ctx)
        except Exception:  # noqa: BLE001 - one scenario's crash is its failed check
            report.check(scenario, "scenario ran to its end", False, traceback.format_exc())
    report.save()
    log(f"{sum(c.ok for c in report.checks)}/{len(report.checks)} check(s) passed")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
