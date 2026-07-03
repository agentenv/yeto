"""SkyPilot orchestration: one syncer VM + one cluster per learner.

Flow:
  1. build the Rust syncer locally (release) — the binary is file-mounted to
     a cheap CPU VM whose TCP port is opened to the learners;
  2. launch the syncer cluster, read its head public IP;
  3. launch all learner clusters in parallel (each pinned to its cloud/region
     from the --gpu spec), with the repo synced as the workdir and the syncer
     address passed via env;
  4. stream all job logs with per-cluster prefixes while a fleet controller
     polls job/cluster health, re-provisions failed or preempted clusters
     with their original spec, and abandons (tears down) any learner that
     cannot be restored within --recover-timeout — the run continues with
     the shrunken fleet;
  5. tear everything down (unless --keep; abandoned learners are always
     torn down).
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from .gpu_spec import ClusterSpec, parse_gpu_spec

SYNCER_PORT = 29400
REPO_ROOT = Path(__file__).resolve().parent.parent

# Rough per-GPU training capacity sanity check (bf16 LoRA, GB).
GPU_MEM_GB = {"A100": 40, "A100-80GB": 80, "H100": 80, "H200": 141, "L4": 24, "A10G": 24, "T4": 16, "V100": 16, "L40S": 48}
MODEL_WEIGHT_GB = {"deepseek4flash": 568, "gemma4": 66}


def build_syncer_binary() -> Path:
    binary = REPO_ROOT / "syncer/target/release/yeto-syncer"
    print("[launcher] building syncer (cargo build --release)...")
    subprocess.run(["cargo", "build", "--release"], cwd=REPO_ROOT / "syncer", check=True)
    return binary


def make_syncer_task(args, num_learners: int):
    import sky

    binary = build_syncer_binary()
    cmd = (
        f"chmod +x ~/yeto-syncer && ~/yeto-syncer"
        f" --port {SYNCER_PORT}"
        f" --learners {num_learners}"
        f" --quorum {args.quorum}"
        f" --grace-ms {args.grace_ms}"
        f" --total-steps {args.total_steps}"
        f" --outer-lr {args.outer_lr}"
        f" --outer-momentum {args.outer_momentum}"
        f" --checkpoint-path ~/yeto-state.ckpt --resume"
        f" --event-tape ~/yeto-tape.jsonl"
    )
    task = sky.Task(
        name="yeto-syncer",
        run=cmd,
        file_mounts={"~/yeto-syncer": str(binary)},
    )
    # --syncer-region accepts "region" (AWS assumed) or "cloud/region".
    infra = args.syncer_region if "/" in args.syncer_region else f"aws/{args.syncer_region}"
    task.set_resources(
        sky.Resources(
            infra=infra,
            cpus="8+",
            memory=f"{args.syncer_memory}+",
            ports=[SYNCER_PORT],
            use_spot=False,
        )
    )
    return task


PICKLED_LOSS_FILE = ".yeto_loss.pkl"


def resolve_loss_function(loss_function) -> str:
    """Return the --loss-function string to pass to learners.

    A callable or a ``custom:<file.py>`` spec is loaded here (failing fast
    before any cloud spend), pickled by value into the workdir, and shipped
    to learners as ``pickle:.yeto_loss.pkl``. Named losses pass through.
    """
    from .losses import dump_pickled_loss, load_custom_loss

    if callable(loss_function):
        fn = loss_function
    elif isinstance(loss_function, str) and loss_function.startswith("custom:"):
        fn = load_custom_loss(loss_function)
    else:
        return loss_function
    dump_pickled_loss(fn, REPO_ROOT / PICKLED_LOSS_FILE)
    return f"pickle:{PICKLED_LOSS_FILE}"


def make_learner_task(args, spec: ClusterSpec, learner_id: int, num_learners: int, syncer_addr: str):
    import sky

    learner_flags = (
        f" --model {shlex.quote(args.model)}"
        f" --data {shlex.quote(args.data)}"
        f" --syncer $SYNCER_ADDR"
        f" --learner-id $LEARNER_ID"
        f" --num-learners {num_learners}"
        f" --loss-function {args.loss_function}"
        f" --train-on {args.train_on}"
        f" --tuning {args.tuning}"
        f" --lora-r {args.lora_r}"
        f" --seq-len {args.seq_len}"
        f" --micro-batch-size {args.micro_batch_size}"
        f" --grad-accum {args.grad_accum}"
        f" --inner-lr {args.inner_lr}"
        f" --fragments {args.fragments}"
        f" --tokenize {args.tokenize}"
        f" --stream-workers {args.stream_workers}"
        f" --wire-dtype {args.wire_dtype}"
        f" --wan-streams {args.wan_streams}"
        f" --output-dir ~/yeto-output"
    )
    if args.max_rows:
        learner_flags += f" --max-rows {args.max_rows}"
    run = (
        'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
        "torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK "
        "--nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE "
        "--master_addr=$MASTER_ADDR --master_port=29500 "
        f"-m yeto.learner{learner_flags}"
    )
    envs = {
        "SYNCER_ADDR": syncer_addr,
        "LEARNER_ID": str(learner_id),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    import os

    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if spec.num_nodes > 1:
        # Surface NCCL's chosen transport in the job logs so an EFA-less
        # fallback to TCP sockets is visible, not silent.
        envs["NCCL_DEBUG"] = "INFO"
    file_mounts = None
    if args.loss_function.startswith("pickle:"):
        # The pickled loss is gitignored, so the workdir sync skips it;
        # mount it into the workdir explicitly.
        file_mounts = {
            f"~/sky_workdir/{PICKLED_LOSS_FILE}": str(REPO_ROOT / PICKLED_LOSS_FILE)
        }
    task = sky.Task(
        name=f"yeto-learner-{learner_id}",
        setup="pip install -q -r requirements.txt",
        run=run,
        envs=envs,
        num_nodes=spec.num_nodes,
        workdir=str(REPO_ROOT),
        file_mounts=file_mounts,
    )
    infra = f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud
    resources_kwargs = {}
    if spec.num_nodes > 1:
        # Multi-node learner: inner DDP all-reduce crosses the node fabric,
        # so request the cloud's RDMA-class interconnect (EFA on AWS,
        # GPUDirect on GCP). Single-node clusters stay on NVLink and don't
        # need it. On AWS this also swaps in the EFA-ready DLAMI; SkyPilot
        # installs no EFA software itself, and if the pinned AMI is missing
        # NCCL silently falls back to TCP — NCCL_DEBUG below makes the
        # chosen transport visible in the job logs (look for
        # "NET/OFI Selected Provider is efa").
        resources_kwargs["network_tier"] = "best"
    task.set_resources(
        sky.Resources(
            infra=infra,
            accelerators=spec.accelerators,
            cpus=args.learner_cpus,
            instance_type=args.learner_instance_type,
            use_spot=args.spot,
            disk_size=args.disk_size,
            **resources_kwargs,
        )
    )
    return task


def warn_if_model_wont_fit(args, specs: list[ClusterSpec]) -> None:
    weight_gb = MODEL_WEIGHT_GB.get(args.model)
    if weight_gb is None:
        return
    for spec in specs:
        vram = GPU_MEM_GB.get(spec.gpu, 0) * spec.total_gpus
        if vram < weight_gb:
            print(
                f"[launcher] WARNING: {spec} has ~{vram} GB VRAM but {args.model} "
                f"needs ~{weight_gb} GB for frozen bf16 weights alone — expect OOM.",
                file=sys.stderr,
            )


def _tail(cluster: str, job_id: int, prefix: str) -> int:
    import sky

    while True:
        try:
            it = sky.tail_logs(cluster, job_id, follow=True, preload_content=False)
            for line in it:
                if line is None:
                    break
                print(f"[{prefix}] {line.rstrip()}", flush=True)
            return 0
        except Exception as e:  # transient stream drops: reconnect
            print(f"[{prefix}] log stream error: {e}; retrying", flush=True)
            time.sleep(5)


class SkySDKOps:
    """Thin adapter over the sky SDK: the only surface FleetController needs.

    Tests inject a fake with the same methods. Any sky call here may raise
    (e.g. the cluster no longer exists); the controller treats exceptions
    from job_status/cluster_up as "cluster gone".
    """

    def job_status(self, cluster: str, job_id: int):
        import sky

        return sky.get(sky.job_status(cluster, [job_id])).get(job_id)

    def cluster_up(self, cluster: str) -> bool:
        import sky

        records = sky.get(sky.status(cluster_names=[cluster]))
        if not records:
            return False
        record = records[0]
        status = (
            record.get("status")
            if isinstance(record, dict)
            else getattr(record, "status", None)
        )
        return status == sky.ClusterStatus.UP

    def relaunch(self, task, cluster: str):
        """Re-provision `cluster` (same spec) and submit `task` as a new job.

        Blocking; returns the new job id, or None if provisioning failed.
        """
        import sky

        try:
            job_id, _handle = sky.get(sky.launch(task, cluster_name=cluster))
            return job_id
        except Exception as e:
            print(f"[launcher] relaunch of {cluster} failed: {e}", file=sys.stderr)
            return None

    def down(self, cluster: str) -> None:
        import sky

        sky.get(sky.down(cluster))

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


# FleetController states.
RUNNING = "running"
RECOVERING = "recovering"
DONE = "done"
ABANDONED = "abandoned"


class _RelaunchAttempt:
    """Result slot for one background relaunch attempt."""

    def __init__(self):
        self.result = None  # new job id, or None if provisioning failed
        self.finished = False


class FleetController:
    """Supervises the syncer + learner fleet after the initial launch.

    Per-learner state machine (evaluated once per poll):

        running    --(job terminal-not-SUCCEEDED, or cluster not UP)--> recovering
        recovering --(relaunch returns a new job id)------------------> running
        recovering --(recover_timeout exceeded)-----------------------> abandoned
        running    --(job SUCCEEDED)----------------------------------> done

    An abandoned learner's cluster is torn down immediately (even with
    --keep) and the run continues with the remaining fleet: the syncer's
    quorum design tolerates missing learners, and a learner that comes back
    later is caught up by the syncer's full rebroadcast. The syncer follows
    the same running/recovering cycle but is never abandoned — past the
    timeout it keeps retrying and logs an error every poll (its relaunch
    resumes from the on-VM checkpoint via the --resume flag already in its
    run command). recover_timeout <= 0 disables recovery: a failed learner
    is torn down on the spot.

    At most one relaunch attempt is in flight per cluster, each in a
    background thread so one slow re-provision never blocks polling the
    others; `thread_cls` exists so tests can substitute a synchronous stub.
    """

    def __init__(
        self,
        learners: dict,
        syncer: tuple,
        sky_ops,
        poll_interval: float,
        recover_timeout: float,
        on_relaunch=None,
        thread_cls=threading.Thread,
    ):
        """`learners` maps cluster name -> (task, job_id); `syncer` is
        (name, task, job_id). `on_relaunch(name, new_job_id)` is called after
        every successful relaunch (production spawns a new log tail)."""
        self.ops = sky_ops
        self.poll_interval = poll_interval
        self.recover_timeout = recover_timeout
        self.on_relaunch = on_relaunch
        self.thread_cls = thread_cls
        self.learners = {
            name: self._make_record(name, task, job_id)
            for name, (task, job_id) in learners.items()
        }
        syncer_name, syncer_task, syncer_job = syncer
        self.syncer = self._make_record(syncer_name, syncer_task, syncer_job)
        self.downed_clusters: set = set()

    @staticmethod
    def _make_record(name, task, job_id):
        return {
            "name": name,
            "task": task,
            "job_id": job_id,
            "state": RUNNING,
            "failed_at": None,
            "attempt": None,
            "exit": None,
        }

    def run(self) -> dict:
        """Poll until every learner is done or abandoned.

        Returns {learner name: final status string}; raises RuntimeError
        (after downing the syncer) if every learner was abandoned.
        """
        while True:
            self._poll(self.syncer, is_syncer=True)
            for rec in self.learners.values():
                self._poll(rec, is_syncer=False)
            if all(r["state"] in (DONE, ABANDONED) for r in self.learners.values()):
                break
            self.ops.sleep(self.poll_interval)
        exit_codes = {name: rec["exit"] for name, rec in self.learners.items()}
        print(f"[launcher] learner jobs finished: {exit_codes}")
        if not any(rec["state"] == DONE for rec in self.learners.values()):
            print(
                "[launcher] ERROR: all learners abandoned; tearing down the syncer",
                file=sys.stderr,
            )
            self._down(self.syncer["name"])
            raise RuntimeError("all learners abandoned; nothing left to train")
        return exit_codes

    def _poll(self, rec, is_syncer: bool) -> None:
        if rec["state"] == RUNNING:
            verdict, status = self._probe(rec)
            if verdict is None:
                return  # healthy
            if verdict == "succeeded":
                rec["state"] = DONE
                rec["exit"] = str(status)
                print(f"[launcher] {rec['name']} job finished: {status}")
            else:
                self._enter_recovering(rec, verdict, is_syncer)
        elif rec["state"] == RECOVERING:
            self._drive_recovery(rec, is_syncer)

    def _probe(self, rec):
        """Classify a running cluster: (None, status) if healthy,
        ("succeeded", status), or (failure reason, status)."""
        name, job_id = rec["name"], rec["job_id"]
        try:
            status = self.ops.job_status(name, job_id)
        except Exception as e:
            return f"job status unavailable ({e})", None
        if status is not None and status.is_terminal():
            if "SUCCEEDED" in str(status):
                return "succeeded", status
            return f"job ended as {status}", status
        try:
            up = self.ops.cluster_up(name)
        except Exception as e:
            return f"cluster status unavailable ({e})", status
        if not up:
            return "cluster is not UP (preempted or deleted)", status
        return None, status

    def _enter_recovering(self, rec, reason: str, is_syncer: bool) -> None:
        rec["state"] = RECOVERING
        rec["failed_at"] = self.ops.now()
        print(
            f"[launcher] {rec['name']}: {reason}; starting recovery "
            f"(timeout {self.recover_timeout}s)",
            file=sys.stderr,
        )
        if not is_syncer and self.recover_timeout <= 0:
            self._abandon(rec, 0.0)
            return
        self._drive_recovery(rec, is_syncer)

    def _drive_recovery(self, rec, is_syncer: bool) -> None:
        attempt = rec["attempt"]
        if attempt is not None and attempt.finished:
            rec["attempt"] = None
            if attempt.result is not None:
                rec["job_id"] = attempt.result
                rec["state"] = RUNNING
                rec["failed_at"] = None
                print(
                    f"[launcher] {rec['name']} recovered: relaunched as job "
                    f"{attempt.result}"
                )
                if self.on_relaunch is not None:
                    self.on_relaunch(rec["name"], attempt.result)
                return
            print(
                f"[launcher] relaunch attempt for {rec['name']} failed; will retry",
                file=sys.stderr,
            )
        elapsed = self.ops.now() - rec["failed_at"]
        if self.recover_timeout <= 0 or elapsed > self.recover_timeout:
            if is_syncer:
                # The syncer is never abandoned: without it no learner can
                # make outer progress, so keep trying and complain loudly.
                print(
                    f"[launcher] ERROR: syncer unrecovered for {elapsed:.0f}s "
                    f"(recover timeout {self.recover_timeout}s exceeded); "
                    "still retrying — learners cannot sync until it returns",
                    file=sys.stderr,
                )
            else:
                self._abandon(rec, elapsed)
                return
        if rec["attempt"] is None:
            rec["attempt"] = self._start_relaunch(rec)

    def _start_relaunch(self, rec) -> _RelaunchAttempt:
        attempt = _RelaunchAttempt()
        name, task = rec["name"], rec["task"]

        def _run():
            try:
                attempt.result = self.ops.relaunch(task, name)
            except Exception as e:
                print(f"[launcher] relaunch of {name} raised: {e}", file=sys.stderr)
                attempt.result = None
            finally:
                attempt.finished = True
            if attempt.result is not None and rec["state"] == ABANDONED:
                # Abandoned while this attempt was in flight, but the
                # relaunch re-provisioned the cluster anyway: tear it back
                # down so nothing is left running unattended.
                self._down(name, force=True)

        thread = self.thread_cls(target=_run, daemon=True)
        thread.start()
        return attempt

    def _abandon(self, rec, elapsed: float) -> None:
        rec["state"] = ABANDONED
        rec["exit"] = f"ABANDONED after {elapsed:.0f}s"
        self._down(rec["name"])
        remaining = sum(1 for r in self.learners.values() if r["state"] != ABANDONED)
        print(
            f"[launcher] LEARNER {rec['name']} ABANDONED after {elapsed:.0f}s "
            f"(could not recover within {self.recover_timeout}s); "
            f"fleet continues with {remaining} learner(s)",
            file=sys.stderr,
        )

    def _down(self, name: str, force: bool = False) -> None:
        if name in self.downed_clusters and not force:
            return
        self.downed_clusters.add(name)
        print(f"[launcher] tearing down {name}")
        try:
            self.ops.down(name)
        except Exception as e:
            print(f"[launcher] teardown of {name} failed: {e}", file=sys.stderr)


def run(args) -> int:
    import sky

    specs = parse_gpu_spec(args.gpu)
    num_learners = len(specs)
    args.loss_function = resolve_loss_function(args.loss_function)
    warn_if_model_wont_fit(args, specs)
    prefix = args.cluster_prefix
    clusters: list[str] = []
    controller = None

    try:
        # 1. Syncer.
        syncer_cluster = f"{prefix}-syncer"
        print(f"[launcher] launching syncer cluster {syncer_cluster} in {args.syncer_region}")
        syncer_task = make_syncer_task(args, num_learners)
        rid = sky.launch(syncer_task, cluster_name=syncer_cluster)
        syncer_job, syncer_handle = sky.stream_and_get(rid)
        clusters.append(syncer_cluster)
        syncer_addr = f"{syncer_handle.head_ip}:{SYNCER_PORT}"
        print(f"[launcher] syncer up at {syncer_addr}")

        # 2. Learners, in parallel.
        tasks = {}
        rids = {}
        for m, spec in enumerate(specs):
            name = f"{prefix}-l{m}-{spec.region or spec.cloud}"
            task = make_learner_task(args, spec, m, num_learners, syncer_addr)
            tasks[name] = task
            print(f"[launcher] launching learner {m} on {spec} as {name}")
            rids[name] = (
                m,
                sky.launch(task, cluster_name=name, retry_until_up=args.retry_until_up),
            )

        results = {}
        errors = {}

        def resolve(name: str, m: int, rid) -> None:
            try:
                results[name] = sky.stream_and_get(rid)
            except Exception as e:
                errors[name] = e

        threads = [
            threading.Thread(target=resolve, args=(n, m, r), daemon=True)
            for n, (m, r) in rids.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for name in rids:
            if name not in errors:
                clusters.append(name)
        if errors:
            for name, e in errors.items():
                print(f"[launcher] ERROR launching {name}: {e}", file=sys.stderr)
            raise RuntimeError(f"{len(errors)} learner cluster(s) failed to provision")

        # 3. Stream logs while the fleet controller polls health, recovers
        #    failed/preempted clusters, and abandons learners that stay down
        #    past --recover-timeout.
        def spawn_tail(name: str, job_id: int) -> None:
            label = "syncer" if name == syncer_cluster else name
            threading.Thread(target=_tail, args=(name, job_id, label), daemon=True).start()

        spawn_tail(syncer_cluster, syncer_job)
        for name, (job_id, _handle) in results.items():
            spawn_tail(name, job_id)

        controller = FleetController(
            learners={name: (tasks[name], job_id) for name, (job_id, _h) in results.items()},
            syncer=(syncer_cluster, syncer_task, syncer_job),
            sky_ops=SkySDKOps(),
            poll_interval=args.controller_poll,
            recover_timeout=args.recover_timeout,
            on_relaunch=spawn_tail,
        )
        exit_codes = controller.run()
        failed = [n for n, s in exit_codes.items() if "SUCCEEDED" not in s]

        # Fetch instructions: prefer learner 0, else any learner that
        # finished (abandoned clusters are already gone).
        done = [n for n, s in exit_codes.items() if "SUCCEEDED" in s]
        source = next((n for n in done if "-l0-" in n), done[0])
        print(
            f"[launcher] fine-tuned model saved on {source}:~/yeto-output\n"
            f"  fetch with: scp -r {source}:yeto-output ./"
        )
        return 1 if failed else 0
    finally:
        # Clusters the controller already tore down (abandoned learners, or
        # the syncer after a total loss) are skipped — even with --keep.
        downed = controller.downed_clusters if controller is not None else set()
        remaining = [c for c in clusters if c not in downed]
        if args.keep:
            print(f"[launcher] keeping clusters: {remaining}")
        else:
            for name in remaining:
                print(f"[launcher] tearing down {name}")
                try:
                    sky.get(sky.down(name))
                except Exception as e:
                    print(f"[launcher] teardown of {name} failed: {e}", file=sys.stderr)
