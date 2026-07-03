#!/usr/bin/env python3
"""Yeto: efficient, low-cost post-training across clouds/regions via SkyPilot.

Example:
    yeto launch \
        --gpu aws:8xa100@us-east-2,aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
        --model deepseek4flash \
        --data armand0e/claude-fable-5-claude-code \
        --loss-function cross_entropy

`launch` follows SkyPilot's UX: it detaches — by default
(`--controller head`) the run is handed to one small on-demand head VM
that hosts both the syncer and the fleet controller, so this machine is
not needed after submission; `--controller local` instead runs a
detached worker on this machine. Either way the CLI streams the run's
log and Ctrl-C detaches instead of killing anything. Re-attach with
`yeto logs <run>`, inspect with `yeto status`, stop with
`yeto down <run>`. Runs are named by `--cluster-prefix`. Bare flags
(`yeto --gpu ...`) still work and are treated as `yeto launch ...`.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

from . import runs
from .losses import LOSS_FUNCTIONS

SUBCOMMANDS = ("launch", "shape", "status", "logs", "down", "_worker", "_head")


def _add_launch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--gpu",
        required=True,
        help="comma-separated learner clusters: cloud:[NODESx]COUNTxGPU[@region], "
        "e.g. aws:4x8xa100@us-east-2,gcp:8xa100@us-central1",
    )
    p.add_argument("--model", required=True, help="model alias (gemma4|deepseek4flash) or HF id")
    p.add_argument("--data", required=True, help="HF dataset id (messages-format chat traces)")
    def loss_spec(value: str) -> str:
        if value in LOSS_FUNCTIONS or value.startswith(("custom:", "pickle:")):
            return value
        raise argparse.ArgumentTypeError(
            f"expected one of {LOSS_FUNCTIONS} or custom:<file.py>[:<fn>]"
        )

    p.add_argument(
        "--loss-function",
        type=loss_spec,
        default="cross_entropy",
        help=f"one of {'|'.join(LOSS_FUNCTIONS)}, or custom:<file.py>[:<fn>] "
        "defining fn(logits, input_ids, weights) -> (loss, num_tokens); the "
        "callable is pickled by value and shipped to all learners",
    )

    tune = p.add_argument_group("fine-tuning")
    tune.add_argument("--tuning", choices=["lora", "full"], default="lora")
    tune.add_argument(
        "--shard",
        choices=["ddp", "fsdp"],
        default="ddp",
        help="per-learner multi-GPU strategy; fsdp shards the frozen base "
        "across the learner's GPUs/nodes (lora only) so the model no "
        "longer has to fit on one GPU",
    )
    tune.add_argument(
        "--train-on",
        choices=["assistant", "all"],
        default="assistant",
        help="which tokens carry loss: assistant-message tokens only (default) or every token",
    )
    tune.add_argument("--lora-r", type=int, default=16)
    tune.add_argument("--seq-len", type=int, default=2048)
    tune.add_argument("--micro-batch-size", type=int, default=1)
    tune.add_argument("--grad-accum", type=int, default=4)
    tune.add_argument("--inner-lr", type=float, default=3e-4)
    tune.add_argument("--max-rows", type=int, default=None, help="cap dataset rows per learner")
    tune.add_argument(
        "--tokenize",
        choices=["stream", "preload"],
        default="stream",
        help="stream: async tokenization in DataLoader workers (default); preload: all upfront",
    )
    tune.add_argument(
        "--stream-workers",
        type=int,
        default=2,
        help="tokenizer worker processes per learner rank (stream mode)",
    )

    sync = p.add_argument_group("async sync")
    sync.add_argument("--total-steps", type=int, default=64, help="outer steps T (one fragment each)")
    sync.add_argument("--fragments", type=int, default=8, help="fragments P (= sync interval H)")
    sync.add_argument("--quorum", type=int, default=1, help="minimum learners per outer step (K)")
    sync.add_argument(
        "--grace-ms",
        type=int,
        default=1000,
        help="cap on the post-quorum grace window; the actual wait adapts "
        "per round to the learners' compute slack",
    )
    sync.add_argument(
        "--grace-gamma",
        type=float,
        default=0.8,
        help="safety margin on the adaptive grace slack (γ < 1)",
    )
    sync.add_argument(
        "--grace-tau",
        type=float,
        default=2.0,
        help="compute-overlap budget for the grace window, in inner steps (τ)",
    )
    sync.add_argument(
        "--delta-correction",
        choices=["heloco", "none"],
        default="heloco",
        help="pre-merge correction of learner deltas against the outer "
        "momentum (HeLoCo, arXiv 2606.00271); shrinks/reorients stale "
        "deltas that oppose the global trajectory",
    )
    sync.add_argument("--outer-lr", type=float, default=0.7)
    sync.add_argument("--outer-momentum", type=float, default=0.9)
    sync.add_argument(
        "--fragment-pattern",
        choices=["binpack", "strided"],
        default="binpack",
        help="fragment grouping: size-balanced bin-packing or depth-interleaved "
        "transformer layers (Streaming DiLoCo strided pattern)",
    )
    sync.add_argument(
        "--merge-alpha",
        type=float,
        default=0.5,
        help="local weight when a learner applies a broadcast fragment "
        "(0 = overwrite, 0.5 = keep half the in-flight local progress)",
    )
    sync.add_argument(
        "--wire-dtype",
        choices=["bf16", "f32", "q4"],
        default="bf16",
        help="WAN tensor encoding; q4 sends pushes as 4-bit E3M0 block-quantized "
        "deltas (~4x less learner egress; broadcasts stay bf16)",
    )
    sync.add_argument("--wan-streams", type=int, default=4, help="parallel TCP streams per learner")

    infra = p.add_argument_group("infrastructure")
    infra.add_argument(
        "--spot",
        action="store_true",
        default=True,
        help="use spot instances for learners (default)",
    )
    infra.add_argument(
        "--on-demand",
        dest="spot",
        action="store_false",
        help="use on-demand instances for learners instead of spot",
    )
    infra.add_argument("--disk-size", type=int, default=512, help="learner disk (GB)")
    infra.add_argument(
        "--learner-cpus",
        default=None,
        help="vCPU hint per learner node (e.g. '8+') to steer instance selection",
    )
    infra.add_argument(
        "--learner-instance-type",
        default=None,
        help="pin learner nodes to an exact instance type (e.g. gr6.4xlarge)",
    )
    infra.add_argument(
        "--syncer-region",
        default="us-west-2",
        help="syncer VM placement: 'region' (AWS) or 'cloud/region', e.g. gcp/us-central1",
    )
    infra.add_argument("--syncer-memory", type=int, default=32, help="syncer RAM (GB)")
    infra.add_argument(
        "--controller",
        choices=["head", "local"],
        default="head",
        help="where the run's controller lives: 'head' (default) provisions "
        "one small on-demand VM that hosts both the syncer and the fleet "
        "controller, so this machine is not needed after submission; "
        "'local' runs a detached worker on this machine plus a separate "
        "syncer VM (this machine must stay up for the whole run)",
    )
    infra.add_argument("--cluster-prefix", default="yeto", help="cluster name prefix; also the run's name")
    infra.add_argument("--keep", action="store_true", help="do not tear down clusters at the end")
    infra.add_argument(
        "--retry-until-up",
        action="store_true",
        help="keep retrying learner provisioning until capacity is found",
    )
    infra.add_argument(
        "--recover-timeout",
        type=int,
        default=1200,
        help="seconds to keep re-provisioning a failed/preempted learner before "
        "tearing it down and continuing with the remaining fleet (0 tears the "
        "learner down on its first failure; the syncer is always recovered)",
    )
    infra.add_argument(
        "--controller-poll",
        type=int,
        default=30,
        help="fleet-controller health poll interval (seconds)",
    )


def parse_args(argv=None):
    """Parse launch flags only (kept for callers that predate subcommands)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_launch_args(p)
    return p.parse_args(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yeto",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", metavar="{launch,status,logs,down}")

    launch = sub.add_parser(
        "launch",
        help="submit a detached training run and stream its log",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_launch_args(launch)

    shape = sub.add_parser(
        "shape",
        help="compute the best fleet plan for a model and budget",
        description="Maximize effective training FLOPs under a $/hr budget, "
        "AWS spot quotas, and spot placement scores; prints the plan and the "
        "matching `yeto launch` line. Signals are fetched in parallel and "
        "cached for 1 hour.",
    )
    shape.add_argument("--model", required=True, help="model alias or HF id")
    shape.add_argument("--budget", type=float, required=True, help="fleet budget in $/hr (includes head VM)")
    shape.add_argument("--tuning", choices=["lora", "full"], default="lora")
    shape.add_argument("--seq-len", type=int, default=2048)
    shape.add_argument("--data", default=None, help="HF dataset id (fills the launch line; required with --apply)")
    shape.add_argument(
        "--apply",
        action="store_true",
        help="launch the computed plan immediately (hands off to `yeto launch`)",
    )
    shape.add_argument("--json", action="store_true", help="emit the plan as JSON instead of text")
    shape.add_argument(
        "--regions",
        default=None,
        help="comma-separated AWS regions, or 'all' for every catalog region "
        "(default: us-east-1,us-east-2,us-west-1,us-west-2)",
    )
    shape.add_argument(
        "--price-margin",
        type=float,
        default=0.15,
        help="headroom applied to catalog spot prices when enforcing the "
        "budget (they are estimates and move)",
    )
    shape.add_argument(
        "--head-cost",
        type=float,
        default=0.40,
        help="assumed $/hr for the head VM in budget math",
    )
    shape.add_argument(
        "--gpus",
        default=None,
        help="comma-separated GPU allowlist in sky names (e.g. A100,A100-80GB,H100); default: all known",
    )
    shape.add_argument(
        "--min-score",
        type=int,
        default=7,
        help="require spot placement score strictly greater than this "
        "(0 keeps fetching scores but stops gating on them)",
    )
    shape.add_argument(
        "--skip-capacity-check",
        action="store_true",
        help="plan on quota + price alone with NO placement-score API calls "
        "(useful when the daily score-config budget is exhausted; the plan "
        "is not verified against spot obtainability)",
    )
    shape.add_argument(
        "--strict-capacity-check",
        action="store_true",
        help="reject shapes whose placement score cannot be fetched instead "
        "of assuming the best score with a warning (the default)",
    )
    shape.add_argument("--max-islands", type=int, default=16, help="cap on learner islands (syncer fan-out)")
    shape.add_argument(
        "--weights-gb",
        type=float,
        default=None,
        help="override the model weight size estimate (bf16 GB)",
    )
    shape.add_argument("--no-cache", action="store_true", help="bypass the 1h signal cache")

    sub.add_parser("status", help="table of known runs")

    logs = sub.add_parser("logs", help="stream a run's launcher log (Ctrl-C detaches)")
    logs.add_argument("run", help="run name (its --cluster-prefix)")
    logs.add_argument(
        "--no-follow",
        action="store_true",
        help="print the log captured so far and exit instead of following",
    )

    down = sub.add_parser("down", help="stop a run's worker and tear down its clusters")
    down.add_argument("run", help="run name (its --cluster-prefix)")

    # Internal: the detached background worker `launch` spawns.
    worker = sub.add_parser("_worker")
    worker.add_argument("run")

    # Internal: the controller job that runs ON the head VM (head mode).
    head = sub.add_parser("_head")
    head.add_argument("args_json", help="JSON-serialized launch args")
    return p


# ---------------------------------------------------------------------------
# launch


def _spawn_worker(name: str) -> subprocess.Popen:
    """Start the detached worker process, stdout+stderr appended to the log.

    `start_new_session` puts it in its own session/process group so it
    survives this CLI exiting and never touches the controlling TTY;
    `-u` keeps its output unbuffered so log tailing is live.
    """
    env = dict(os.environ)
    env["YETO_RUNS_DIR"] = str(runs.RUNS_DIR)
    with open(runs.log_path(name), "ab") as log_f:
        return subprocess.Popen(
            [sys.executable, "-u", "-m", "yeto.cli", "_worker", name],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )


def _stream_log(name: str, follow: bool = True, alive=None) -> None:
    """Print the run's launcher.log; in follow mode, tail until the worker
    exits (then drain what's left). Raises KeyboardInterrupt through to the
    caller — Ctrl-C means "stop streaming", never "stop the run"."""
    if alive is None:
        def alive() -> bool:
            meta = runs.load_run(name) or {}
            return runs.is_alive(meta.get("pid"))

    with open(runs.log_path(name), "r", errors="replace") as f:
        while True:
            line = f.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                continue
            if not follow or not alive():
                rest = f.read()  # drain output flushed around worker exit
                if rest:
                    sys.stdout.write(rest)
                    sys.stdout.flush()
                return
            time.sleep(0.5)


def _final_state(meta: dict) -> tuple[str, int]:
    """(display state, exit code) for a run whose worker is gone."""
    state = meta.get("state") or "UNKNOWN"
    if state in (runs.PENDING, runs.RUNNING):
        # Worker died without recording a result (killed, OOM, crash).
        state = f"{runs.FAILED} (worker exited without recording a result)"
    code = meta.get("exit_code")
    if code is None:
        code = 0 if state == runs.SUCCEEDED else 1
    return state, int(code)


def _print_detach_hints(name: str) -> None:
    print(
        f"\n[yeto] Detached from run '{name}'; it continues in the background.\n"
        f"[yeto] To re-attach to the logs:\tyeto logs {name}\n"
        f"[yeto] To stop the run:\t\tyeto down {name}",
        flush=True,
    )


def cmd_launch(args) -> int:
    name = args.cluster_prefix
    existing = runs.load_run(name)
    if existing is not None and runs.is_alive(existing.get("pid")):
        print(
            f"[yeto] run '{name}' already has a live worker "
            f"(pid {existing['pid']}). Use `yeto logs {name}` to attach, "
            f"`yeto down {name}` to stop it, or pick a different "
            f"--cluster-prefix.",
            file=sys.stderr,
        )
        return 1
    if getattr(args, "controller", "local") == "head":
        return cmd_launch_head(args)

    args_dict = {k: v for k, v in vars(args).items() if k != "command"}
    runs.create_run(name, args_dict)
    proc = _spawn_worker(name)
    runs.update_run(name, pid=proc.pid)
    print(f"[yeto] run '{name}' submitted; worker pid {proc.pid}.")
    print(f"[yeto] log: {runs.log_path(name)}")
    print("[yeto] streaming logs — Ctrl-C detaches, the run keeps going.\n", flush=True)
    try:
        _stream_log(name, follow=True, alive=lambda: proc.poll() is None)
    except KeyboardInterrupt:
        _print_detach_hints(name)
        return 0
    state, code = _final_state(runs.load_run(name) or {"name": name})
    print(f"[yeto] run '{name}' finished: {state} (exit code {code})")
    return code


# ---------------------------------------------------------------------------
# head controller mode: launch submission (runs locally) + _head (runs on
# the head VM). Modeled on SkyPilot's managed-jobs controller: one small
# on-demand VM hosts both the syncer process and the fleet controller, so
# the submitting machine can go away right after `yeto launch` returns.

# The readiness marker exists because SkyPilot detaches setup: sky.launch
# can return while these installs are still running, and an exec'd job would
# race them. Setup touches the marker only if every install succeeded; the
# head job waits for it (bounded) before importing anything.
HEAD_READY_MARKER = "~/.yeto_head_ready"
HEAD_SETUP_PIP = (
    'pip install -q "skypilot[aws]>=0.12" && '
    "pip install -q torch --index-url https://download.pytorch.org/whl/cpu && "
    "pip install -q cloudpickle transformers==4.57.1 && "
    f"touch {HEAD_READY_MARKER}"
)
HEAD_WAIT_READY = (
    f"for i in $(seq 1 180); do [ -f {HEAD_READY_MARKER} ] && break; sleep 5; done; "
    f"[ -f {HEAD_READY_MARKER} ] || {{ echo 'head setup never completed' >&2; exit 1; }}"
)


def _serializable_args(args) -> dict:
    """vars(args) restricted to JSON-serializable launch flags, with the
    controller mode pinned to 'head' (this dict is what `_head` replays)."""
    out = {}
    for k, v in vars(args).items():
        if k == "command":
            continue
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    out["controller"] = "head"
    return out


def _make_head_task(args, syncer_binary):
    """The head VM's provisioning task: repo workdir, syncer binary, cloud
    credentials, and the pickled loss (when used) — no run command; the
    controller job is exec'd separately once the head's IP is known."""
    import sky

    from .launcher import PICKLED_LOSS_FILE, REPO_ROOT, SYNCER_PORT, WAN_TUNING

    file_mounts = {"~/yeto-syncer": str(syncer_binary)}
    aws_creds = os.path.expanduser("~/.aws")
    if os.path.isdir(aws_creds):
        # Same pattern as sky's jobs controller: ship the local credentials
        # so the head can launch, recover, and tear down learner clusters.
        file_mounts["~/.aws"] = aws_creds
    else:
        print(
            "[yeto] WARNING: ~/.aws not found; the head will have no cloud "
            "credentials and cannot launch or tear down learner clusters.",
            file=sys.stderr,
        )
    if args.loss_function.startswith("pickle:"):
        # The pickled loss is gitignored, so the workdir sync skips it;
        # mount it into the head's workdir explicitly (the head re-mounts
        # it onto learners from there).
        file_mounts[f"~/sky_workdir/{PICKLED_LOSS_FILE}"] = str(
            REPO_ROOT / PICKLED_LOSS_FILE
        )
    task = sky.Task(
        name="yeto-head",
        setup=f"{WAN_TUNING}; {HEAD_SETUP_PIP}",
        workdir=str(REPO_ROOT),
        file_mounts=file_mounts,
    )
    # Same placement grammar as the syncer VM: 'region' (AWS) or 'cloud/region'.
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


def _sky_launch_head(task, cluster: str):
    """Provision the head cluster; returns the launch handle."""
    import sky

    _job_id, handle = sky.stream_and_get(sky.launch(task, cluster_name=cluster))
    return handle


def _sky_exec_head(task, cluster: str) -> int:
    """Submit the controller job on the (already-provisioned) head."""
    import sky

    job_id, _handle = sky.stream_and_get(sky.exec(task, cluster_name=cluster))
    return job_id


def _sky_tail_logs(cluster: str, job_id: int, follow: bool):
    import sky

    return sky.tail_logs(cluster, job_id, follow=follow, preload_content=False)


def _stream_head_logs(cluster: str, job_id: int, follow: bool = True) -> None:
    """Print a head job's log lines; KeyboardInterrupt passes through to
    the caller — Ctrl-C means "stop streaming", never "stop the run"."""
    for line in _sky_tail_logs(cluster, job_id, follow):
        if line is None:
            break
        sys.stdout.write(line if line.endswith("\n") else line + "\n")
        sys.stdout.flush()


def _record_head_result(name: str, cluster: str, job_id: int) -> None:
    """Best-effort: map the head job's terminal status into the registry."""
    try:
        import sky

        status = sky.get(sky.job_status(cluster, [job_id])).get(job_id)
        if status is None or not status.is_terminal():
            return
        state = runs.SUCCEEDED if "SUCCEEDED" in str(status) else runs.FAILED
    except Exception:
        return
    runs.update_run(name, state=state, finished_at=time.time())


def cmd_launch_head(args) -> int:
    """Submit a head-controlled run: provision one small on-demand VM that
    hosts both the syncer and the fleet controller, hand it the launch
    args, and stream its log (Ctrl-C detaches; the run keeps going without
    this machine)."""
    import sky

    from . import launcher
    from .gpu_spec import parse_gpu_spec

    name = args.cluster_prefix
    head_cluster = f"{name}-head"
    specs = parse_gpu_spec(args.gpu)
    # Resolve the loss BEFORE serializing: a custom:<file.py> spec becomes
    # pickle:<file> here, and the pickle is file-mounted onto the head.
    args.loss_function = launcher.resolve_loss_function(args.loss_function)
    binary = launcher.build_syncer_binary()

    args_dict = _serializable_args(args)
    runs.create_run(name, args_dict)
    learner_names = launcher.learner_cluster_names(name, specs)
    runs.update_run(
        name,
        controller="head",
        head_cluster=head_cluster,
        clusters=[head_cluster] + learner_names,
    )

    print(f"[yeto] provisioning head cluster {head_cluster} in {args.syncer_region}")
    handle = _sky_launch_head(_make_head_task(args, binary), head_cluster)
    head_ip = handle.head_ip
    print(f"[yeto] head is up at {head_ip}; submitting the controller job")

    envs = {"SYNCER_PUBLIC_IP": str(head_ip)}
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
    job_task = sky.Task(
        name="yeto-head-job",
        run=(
            f"{HEAD_WAIT_READY}; "
            "cd ~/sky_workdir && PYTHONPATH=~/sky_workdir "
            f"python3 -m yeto.cli _head {shlex.quote(json.dumps(args_dict))}"
        ),
        envs=envs,
    )
    job_id = _sky_exec_head(job_task, head_cluster)
    runs.update_run(name, state=runs.SUBMITTED, head_job_id=job_id)
    print(f"[yeto] run '{name}' submitted: job {job_id} on {head_cluster}.")
    print("[yeto] this machine is no longer needed; the head supervises the fleet.")
    print("[yeto] streaming head logs — Ctrl-C detaches, the run keeps going.\n", flush=True)
    try:
        _stream_head_logs(head_cluster, job_id, follow=True)
    except KeyboardInterrupt:
        _print_detach_hints(name)
        return 0
    except Exception as e:
        # A dropped stream is not a dropped run: the head keeps going.
        print(f"\n[yeto] log stream lost ({e}); the run continues on the head.")
        _print_detach_hints(name)
        return 0
    _record_head_result(name, head_cluster, job_id)
    print(
        f"[yeto] head job ended; the head cluster {head_cluster} is still up — "
        f"tear everything down with: yeto down {name}"
    )
    return 0


def cmd_head(payload: str) -> int:
    """Controller job, running ON the head VM: start the syncer as a local
    subprocess, then supervise the learner fleet until the run ends."""
    from . import launcher
    from .gpu_spec import parse_gpu_spec

    args = argparse.Namespace(**json.loads(payload))
    num_learners = len(parse_gpu_spec(args.gpu))
    syncer = launcher.LocalSyncer(args, num_learners)
    syncer.start()
    syncer.start_log_forwarder()
    try:
        code = launcher.run(args, local_syncer=syncer)
    except BaseException:
        import traceback

        traceback.print_exc()
        return 1
    finally:
        syncer.stop()
    return int(code or 0)


# ---------------------------------------------------------------------------
# _worker


def cmd_worker(name: str) -> int:
    """Detached executor: replay the recorded launch args through
    yeto.launcher.run and record the outcome in the registry."""
    meta = runs.load_run(name)
    if meta is None or "args" not in meta:
        print(f"[yeto] no recorded args for run '{name}'", file=sys.stderr)
        return 1
    args = argparse.Namespace(**meta["args"])
    runs.update_run(name, state=runs.RUNNING, pid=os.getpid())

    def record_clusters(names) -> None:
        runs.update_run(name, clusters=list(names))

    from .launcher import run as launcher_run

    try:
        code = launcher_run(args, on_clusters=record_clusters)
    except BaseException:
        import traceback

        traceback.print_exc()
        runs.update_run(name, state=runs.FAILED, exit_code=1, finished_at=time.time())
        return 1
    code = int(code or 0)
    runs.update_run(
        name,
        state=runs.SUCCEEDED if code == 0 else runs.FAILED,
        exit_code=code,
        finished_at=time.time(),
    )
    return code


# ---------------------------------------------------------------------------
# status


def _humanize_ago(ts) -> str:
    if not ts:
        return "-"
    delta = max(0.0, time.time() - float(ts))
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{delta / 3600:.1f}h ago"
    return f"{delta / 86400:.1f}d ago"


def _last_log_line(name: str, limit: int = 60) -> str:
    try:
        with open(runs.log_path(name), "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return "-"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "-"
    last = lines[-1]
    return last if len(last) <= limit else last[: limit - 3] + "..."


def _display_clusters(clusters) -> str:
    if not clusters:
        return "-"
    joined = ",".join(clusters)
    return joined if len(joined) <= 40 else f"{len(clusters)} clusters"


def cmd_status() -> int:
    # Registry-only, by design: never calls the sky API, so it's instant.
    metas = runs.list_runs()
    if not metas:
        print("No runs. Start one with: yeto launch --gpu ... --model ... --data ...")
        return 0
    header = ("NAME", "STATE", "STARTED", "CLUSTERS", "LOG")
    rows = []
    for meta in metas:
        name = meta.get("name", "?")
        state = (
            runs.RUNNING
            if runs.is_alive(meta.get("pid"))
            else (meta.get("state") or "UNKNOWN")
        )
        rows.append(
            (
                name,
                state,
                _humanize_ago(meta.get("started_at")),
                _display_clusters(meta.get("clusters")),
                _last_log_line(name),
            )
        )
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(4)]
    for row in [header] + rows:
        lead = "  ".join(row[i].ljust(widths[i]) for i in range(4))
        print(f"{lead}  {row[4]}")
    for meta in metas:
        # Head-mode runs are supervised on their head VM; the registry only
        # has the state as of submission/teardown.
        if meta.get("controller") == "head" and meta.get("state") != runs.DOWN:
            print(
                f"[yeto] '{meta.get('name')}' is controlled from "
                f"{meta.get('head_cluster')}; live state: yeto logs {meta.get('name')}"
            )
    return 0


# ---------------------------------------------------------------------------
# logs


def cmd_logs(args) -> int:
    name = args.run
    meta = runs.load_run(name)
    if meta is None:
        known = ", ".join(m.get("name", "?") for m in runs.list_runs()) or "(none)"
        print(f"[yeto] unknown run '{name}'. Known runs: {known}", file=sys.stderr)
        return 1
    if meta.get("controller") == "head" and meta.get("head_job_id") is not None:
        # Head-mode run: the launcher log lives on the head VM; stream it.
        head_cluster, head_job = meta["head_cluster"], meta["head_job_id"]
        try:
            _stream_head_logs(head_cluster, head_job, follow=not args.no_follow)
        except KeyboardInterrupt:
            _print_detach_hints(name)
            return 0
        except Exception as e:
            print(
                f"[yeto] cannot stream job {head_job} from {head_cluster}: {e}",
                file=sys.stderr,
            )
            return 1
        return 0
    try:
        _stream_log(name, follow=not args.no_follow)
    except KeyboardInterrupt:
        _print_detach_hints(name)
        return 0
    except OSError as e:
        print(f"[yeto] cannot read log for '{name}': {e}", file=sys.stderr)
        return 1
    if not args.no_follow:
        state, code = _final_state(runs.load_run(name) or meta)
        print(f"[yeto] run '{name}' is not running: {state} (exit code {code})")
    return 0


# ---------------------------------------------------------------------------
# down


def _sky_down_cluster(cluster: str) -> None:
    """Tear one cluster down via the sky SDK (patched out in tests)."""
    import sky

    sky.get(sky.down(cluster))


def _signal_worker(pid: int, sig: int) -> None:
    """Signal the worker's whole process group (it is a session leader),
    falling back to the single pid."""
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def cmd_down(args) -> int:
    name = args.run
    meta = runs.load_run(name)
    if meta is None:
        known = ", ".join(m.get("name", "?") for m in runs.list_runs()) or "(none)"
        print(f"[yeto] unknown run '{name}'. Known runs: {known}", file=sys.stderr)
        return 1

    pid = meta.get("pid")
    if runs.is_alive(pid):
        print(f"[yeto] stopping worker pid {pid} (SIGTERM)")
        _signal_worker(int(pid), signal.SIGTERM)
        deadline = time.time() + 10
        while runs.is_alive(pid) and time.time() < deadline:
            time.sleep(0.2)
        if runs.is_alive(pid):
            print(f"[yeto] worker pid {pid} did not exit; sending SIGKILL")
            _signal_worker(int(pid), signal.SIGKILL)
    else:
        print("[yeto] worker is not running")

    clusters = meta.get("clusters") or []
    if clusters:
        print(f"[yeto] tearing down {len(clusters)} cluster(s): {', '.join(clusters)}")

        def _down_one(cluster: str) -> None:
            try:
                _sky_down_cluster(cluster)
                print(f"[yeto] {cluster}: down")
            except Exception as e:  # best-effort; the cluster may be gone
                print(f"[yeto] {cluster}: teardown failed: {e}", file=sys.stderr)

        threads = [
            threading.Thread(target=_down_one, args=(c,), daemon=True) for c in clusters
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        print("[yeto] no clusters recorded for this run")

    runs.update_run(
        name,
        state=runs.DOWN,
        finished_at=meta.get("finished_at") or time.time(),
    )
    print(f"[yeto] run '{name}' is down")
    return 0


# ---------------------------------------------------------------------------


def cmd_shape(args) -> int:
    from .shape.plan import build_shape, launch_argv, render, to_json_dict

    if args.apply and not args.data:
        print("[yeto] --apply needs --data <hf-dataset>", file=sys.stderr)
        return 1
    try:
        result = build_shape(
            model=args.model,
            budget=args.budget,
            tuning=args.tuning,
            seq_len=args.seq_len,
            regions=args.regions.split(",") if args.regions else None,
            gpus=args.gpus.split(",") if args.gpus else None,
            min_score=args.min_score,
            max_islands=args.max_islands,
            weights_gb_override=args.weights_gb,
            cache_enabled=not args.no_cache,
            price_margin=args.price_margin,
            head_cost=args.head_cost,
            skip_capacity_check=args.skip_capacity_check,
            strict_capacity_check=args.strict_capacity_check,
        )
    except (ValueError, RuntimeError) as e:
        print(f"[yeto] shape failed: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(to_json_dict(result, args.model, args.budget, args.tuning, args.data), indent=2))
    else:
        print(render(result, args.model, args.budget, args.tuning, args.data))
    if not result.plan.counts:
        return 1
    if args.apply:
        print("[yeto] applying plan — handing off to `yeto launch`", flush=True)
        return main(launch_argv(result, args.model, args.tuning, args.data))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.command == "launch":
        return cmd_launch(args)
    if args.command == "shape":
        return cmd_shape(args)
    if args.command == "status":
        return cmd_status()
    if args.command == "logs":
        return cmd_logs(args)
    if args.command == "down":
        return cmd_down(args)
    if args.command == "_worker":
        return cmd_worker(args.run)
    if args.command == "_head":
        return cmd_head(args.args_json)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
