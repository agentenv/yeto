"""SkyPilot orchestration: one syncer VM + one cluster per learner.

Flow:
  1. build the Rust syncer locally (release) — the binary is file-mounted to
     a cheap CPU VM whose TCP port is opened to the learners;
  2. launch the syncer cluster, read its head public IP;
  3. launch all learner clusters in parallel (each pinned to its cloud/region
     from the --gpu spec), with the repo synced as the workdir and the syncer
     address passed via env;
  4. stream all job logs with per-cluster prefixes until the learners exit;
  5. tear everything down (unless --keep).
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
        f" --tuning {args.tuning}"
        f" --lora-r {args.lora_r}"
        f" --seq-len {args.seq_len}"
        f" --micro-batch-size {args.micro_batch_size}"
        f" --grad-accum {args.grad_accum}"
        f" --inner-lr {args.inner_lr}"
        f" --fragments {args.fragments}"
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


def run(args) -> int:
    import sky

    specs = parse_gpu_spec(args.gpu)
    num_learners = len(specs)
    args.loss_function = resolve_loss_function(args.loss_function)
    warn_if_model_wont_fit(args, specs)
    prefix = args.cluster_prefix
    clusters: list[str] = []

    try:
        # 1. Syncer.
        syncer_cluster = f"{prefix}-syncer"
        print(f"[launcher] launching syncer cluster {syncer_cluster} in {args.syncer_region}")
        rid = sky.launch(make_syncer_task(args, num_learners), cluster_name=syncer_cluster)
        syncer_job, syncer_handle = sky.stream_and_get(rid)
        clusters.append(syncer_cluster)
        syncer_addr = f"{syncer_handle.head_ip}:{SYNCER_PORT}"
        print(f"[launcher] syncer up at {syncer_addr}")

        # 2. Learners, in parallel.
        rids = {}
        for m, spec in enumerate(specs):
            name = f"{prefix}-l{m}-{spec.region or spec.cloud}"
            task = make_learner_task(args, spec, m, num_learners, syncer_addr)
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

        # 3. Stream logs until the learners finish.
        tails = [threading.Thread(target=_tail, args=(syncer_cluster, syncer_job, "syncer"), daemon=True)]
        waiters = []
        for name, (job_id, _handle) in results.items():
            t = threading.Thread(target=_tail, args=(name, job_id, name), daemon=True)
            tails.append(t)
            waiters.append((name, job_id))
        for t in tails:
            t.start()

        exit_codes = {}
        for name, job_id in waiters:
            # tail_logs with follow returns the exit code when not iterating;
            # poll job status instead so tails stay independent.
            while True:
                statuses = sky.get(sky.job_status(name, [job_id]))
                status = statuses.get(job_id)
                if status is not None and status.is_terminal():
                    exit_codes[name] = str(status)
                    break
                time.sleep(20)
        print(f"[launcher] learner jobs finished: {exit_codes}")
        failed = [n for n, s in exit_codes.items() if "SUCCEEDED" not in s]

        learner0 = next(n for n in results if "-l0-" in n)
        print(
            f"[launcher] fine-tuned model saved on {learner0}:~/yeto-output\n"
            f"  fetch with: scp -r {learner0}:yeto-output ./"
        )
        return 1 if failed else 0
    finally:
        if args.keep:
            print(f"[launcher] keeping clusters: {clusters}")
        else:
            for name in clusters:
                print(f"[launcher] tearing down {name}")
                try:
                    sky.get(sky.down(name))
                except Exception as e:
                    print(f"[launcher] teardown of {name} failed: {e}", file=sys.stderr)
