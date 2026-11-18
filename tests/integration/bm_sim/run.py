"""bm-sim entry: bring the simulated bare-metal cluster up, run the scenarios on
its head, collect the report, tear it down.

On a Linux host with docker::

    python3 tests/integration/bm_sim/run.py --nodes 2 [--scenario happy ...] [--keep]

On a Modal CPU sandbox with docker enabled (no local docker needed; app
``bmw-bm-sim`` unless ``--modal-app``; the sandbox ends with the run)::

    python3 tests/integration/bm_sim/run.py --modal --nodes 3

The report (JSON, one row per check) lands in ``--out`` (default
``/tmp/bm-sim-out``) with the head's log and every node's island logs. Exit
code 0 only when every check passed. CI does not run this; the opt-in pytest
wrapper is ``tests/test_bm_sim_harness.py`` (``YETO_BM_SIM=1``).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODAL_APP = "bmw-bm-sim"
SANDBOX_REPO = "/repo"
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", "node_modules", "target", ".pytest_cache", ".mypy_cache"}


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), "[bm-sim run]", *parts, flush=True)


def local(opts: argparse.Namespace) -> int:
    sys.path.insert(0, str(HERE))
    import cluster  # noqa: E402 - this directory, not a package import

    work = Path(opts.work)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    info = cluster.up(work, opts.nodes, store=opts.store)
    log("cluster up:", json.dumps(info["ips"]))
    argv = ["docker", "exec", "bm-sim-head", "python3", "-u", "-m", "tests.integration.bm_sim.scenario",
            "--work", "/work", "--report", "/work/report.json"]
    for scenario in opts.scenario or ():
        argv += ["--scenario", scenario]
    rc = 1
    try:
        with open(out / "head.log", "w") as head_log:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                head_log.write(line)
            rc = proc.wait()
    finally:
        collect(work, out, opts.nodes)
        if not opts.keep:
            cluster.down()
    log(f"scenario exit {rc}; report {out / 'report.json'}")
    return rc


def collect(work: Path, out: Path, nodes: int) -> None:
    """The report, the head's run dirs and each node's island state (logs)."""
    for name in ("report.json", "cluster.json"):
        if (work / name).is_file():
            (out / name).write_bytes((work / name).read_bytes())
    for k in range(nodes):
        done = subprocess.run(["docker", "exec", f"bm-sim-n{k}", "bash", "-c",
                               "cd /home/yeto && tar czf - .yeto/trainer-islands 2>/dev/null || true"],
                              capture_output=True)
        if done.stdout:
            (out / f"node{k}-islands.tgz").write_bytes(done.stdout)
    subprocess.run(["bash", "-c", f"cd {work} && tar czf {out}/head-runs.tgz runs 2>/dev/null || true"])


def repo_tarball() -> bytes:
    """The worktree as it is (tracked and untracked files, no VCS or build dirs)."""
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "-co", "--exclude-standard", "-z"],
                            check=True, capture_output=True).stdout.decode().split("\0")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in filter(None, listed):
            if set(Path(rel).parts) & EXCLUDE_DIRS or not (REPO / rel).is_file():
                continue
            tar.add(REPO / rel, arcname=rel)
    return buf.getvalue()


def modal(opts: argparse.Namespace) -> int:
    import modal as m

    app = m.App.lookup(opts.modal_app, create_if_missing=True)
    image = (m.Image.debian_slim(python_version="3.12")
             .apt_install("ca-certificates", "curl", "gnupg", "iptables", "openssh-client", "git")
             .run_commands(
                 "install -m 0755 -d /etc/apt/keyrings",
                 "curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc",
                 'echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] '
                 'https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list',
                 "apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io"))
    secrets = [m.Secret.from_name(opts.modal_store_secret)] if opts.store == "r2" else []
    sandbox = m.Sandbox.create("sleep", str(opts.modal_timeout), app=app, image=image, cpu=opts.modal_cpu,
                               memory=opts.modal_memory_mib, timeout=opts.modal_timeout, secrets=secrets,
                               experimental_options={"enable_docker": True})
    log("sandbox", sandbox.object_id)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 1
    try:
        tarball = repo_tarball()
        sandbox.filesystem.write_bytes(tarball, "/tmp/repo.tgz")
        log(f"repo uploaded ({len(tarball) / 1e6:.1f} MB)")
        inner = [f"mkdir -p {SANDBOX_REPO} && tar xzf /tmp/repo.tgz -C {SANDBOX_REPO}",
                 # a git checkout again, so the resident launch packs its code as a customer's does
                 f"git -C {SANDBOX_REPO} init -q && git -C {SANDBOX_REPO} add -A && git -C {SANDBOX_REPO} "
                 "-c user.name=bm-sim -c user.email=bm-sim@example.invalid commit -qm bm-sim",
                 # gVisor: no NAT tables, so dockerd runs without iptables (the bridge still
                 # connects the containers and the host; the build uses the host network)
                 "(nohup dockerd --iptables=false --ip6tables=false > /tmp/dockerd.log 2>&1 &)",
                 "for _ in $(seq 60); do docker info >/dev/null 2>&1 && break; sleep 1; done",
                 f"cd {SANDBOX_REPO} && python3 -u tests/integration/bm_sim/run.py --nodes {opts.nodes} "
                 f"--store {opts.store} --out /tmp/bm-sim-out "
                 + " ".join(f"--scenario {s}" for s in opts.scenario or ())]
        if opts.store == "r2":
            inner.insert(0, "export YETO_RL_STORE_S3_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID "
                            "YETO_RL_STORE_S3_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY "
                            f"YETO_RL_STORE_S3_ENDPOINT_URL={opts.r2_endpoint} BM_SIM_R2_BUCKET={opts.r2_bucket}")
        proc = sandbox.exec("bash", "-c", " && ".join(inner), timeout=opts.modal_timeout - 60)
        with open(out / "sandbox.log", "w") as fh:
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
        rc = proc.wait()
        sandbox.exec("bash", "-c", "cd /tmp/bm-sim-out && tar czf /tmp/bm-sim-out.tgz .").wait()
        payload = sandbox.filesystem.read_bytes("/tmp/bm-sim-out.tgz")
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            tar.extractall(out, filter="data")
    finally:
        sandbox.terminate()
        log("sandbox terminated")
    log(f"exit {rc}; report {out / 'report.json'}")
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bm-sim: simulated bare-metal ssh trainer island")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--scenario", action="append", default=None,
                        help="a scenario of scenario.DEFAULT_SCENARIOS (default: all)")
    parser.add_argument("--store", choices=["local", "r2"], default="local")
    parser.add_argument("--work", default="/tmp/bm-sim")
    parser.add_argument("--out", default="/tmp/bm-sim-out")
    parser.add_argument("--keep", action="store_true", help="leave the cluster up after the run")
    parser.add_argument("--modal", action="store_true", help="run on a Modal CPU sandbox with docker")
    parser.add_argument("--modal-app", default=MODAL_APP)
    parser.add_argument("--modal-cpu", type=float, default=8.0)
    parser.add_argument("--modal-memory-mib", type=int, default=16384)
    parser.add_argument("--modal-timeout", type=int, default=5400)
    parser.add_argument("--modal-store-secret", default=None,
                        help="--store r2 on Modal: the Secret with AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
    parser.add_argument("--r2-endpoint", default=os.environ.get("BM_SIM_R2_ENDPOINT", ""))
    parser.add_argument("--r2-bucket", default=os.environ.get("BM_SIM_R2_BUCKET", ""))
    opts = parser.parse_args(argv)
    return modal(opts) if opts.modal else local(opts)


if __name__ == "__main__":
    sys.exit(main())
