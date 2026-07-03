#!/usr/bin/env python3
"""Yeto: efficient, low-cost post-training across clouds/regions via SkyPilot.

Example:
    yeto launch \
        --gpu aws:8xa100@us-east-2,aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
        --model deepseek4flash \
        --data armand0e/claude-fable-5-claude-code \
        --loss-function cross_entropy

`launch` follows SkyPilot's UX: it detaches — the run executes in a
background worker while the CLI streams its log, and Ctrl-C detaches
instead of killing anything. Re-attach with `yeto logs <run>`, inspect
with `yeto status`, stop with `yeto down <run>`. Runs are named by
`--cluster-prefix`. Bare flags (`yeto --gpu ...`) still work and are
treated as `yeto launch ...`.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

from . import runs
from .losses import LOSS_FUNCTIONS

SUBCOMMANDS = ("launch", "status", "logs", "down", "_worker")


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
    sync.add_argument("--grace-ms", type=int, default=1000, help="grace window after quorum")
    sync.add_argument("--outer-lr", type=float, default=0.7)
    sync.add_argument("--outer-momentum", type=float, default=0.9)
    sync.add_argument("--wire-dtype", choices=["bf16", "f32"], default="bf16")
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


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.command == "launch":
        return cmd_launch(args)
    if args.command == "status":
        return cmd_status()
    if args.command == "logs":
        return cmd_logs(args)
    if args.command == "down":
        return cmd_down(args)
    if args.command == "_worker":
        return cmd_worker(args.run)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
