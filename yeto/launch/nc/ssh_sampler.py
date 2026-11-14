"""Engine sampler on a National Compute node over SSH + ``docker run``.

The node is reached by its alias (:mod:`.node_pool`: workspace relay as
``ProxyCommand``) and described as a node of the SSH node inventory
(:mod:`yeto.launch.node_inventory`): the same card check (``Node.check_cards``
with the rocm card tool), the same mounts / ``docker_args`` shape as the SSH
trainer island. The provider's health k3s on the node is never touched: the
docker host step moves only docker's ``/var/lib/containerd`` and data root onto
NVMe.

One sampler = one container ``yeto-sampler-<shard>`` running
``python -m yeto.rl.engine.engine_sampler --spec`` (vllm-server adapter; the
adapter starts vLLM, ``PolicyFollower`` pulls releases from the run's store) with
the yeto tree of the launcher's commit mounted read-only at ``/yeto``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .. import agentic_setup
from yeto.rl.sampler_pins import VLLM_SAMPLER_IMAGE_ROCM

from ..node_inventory import Node, parse_node
from .client import NodeLease
from .node_pool import host_alias

NVME = "/mnt/nvme"
# the pinned digest of the verified K3.1 ROCm vLLM image (not its moving tag)
IMAGE = VLLM_SAMPLER_IMAGE_ROCM.removeprefix("docker:")
PYTHON = "/opt/runtime-venv/bin/python3"
ROCM_DOCKER_ARGS = ("--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
                    "--security-opt", "seccomp=unconfined", "--cap-add", "SYS_PTRACE",
                    "--ulimit", "memlock=-1", "--shm-size", "64g")
MODEL_HOST = f"{NVME}/models/Kimi-K3.1-Pro"
MOUNTS = {MODEL_HOST: "/models/kimi-k3.1-pro", f"{NVME}/work": "/work", f"{NVME}/cache/vllm": "/root/.cache"}
REPO_ROOT = Path(__file__).resolve().parents[3]


def inventory_entry(lease: NodeLease, *, image: str = IMAGE) -> dict[str, Any]:
    """The lease as an SSH node inventory entry (``--rl-node-inventory`` shape)."""
    return {"name": host_alias(lease.ticket), "host": host_alias(lease.ticket), "device_family": "rocm",
            "cpu_arch": "x86_64", "card": "MI355X", "cards_per_node": lease.gpus or 8, "hbm_gib": 288,
            "host_memory_gib": 2700, "fabric": "tcp", "address": lease.ip, "mounts": dict(MOUNTS),
            "image": image, "runtime": "docker", "docker_args": list(ROCM_DOCKER_ARGS),
            "roles": ["sampler"]}


def node_of(lease: NodeLease, *, image: str = IMAGE) -> Node:
    return parse_node(inventory_entry(lease, image=image), index=0)


def docker_host_script(nvme_root: str = NVME) -> str:
    """Docker on NVMe (Docker 29 keeps layers in /var/lib/containerd, GPU-B section 6.0).
    k3s, kubelet and /var/lib/rancher (the provider's health k3s) are left alone."""
    return f"""set -eu
N={shlex.quote(nvme_root)}
mountpoint -q "$N" || {{ echo "YETO_NC_DOCKER_FAIL $N is not a mount point"; exit 3; }}
sudo -n true || {{ echo "YETO_NC_DOCKER_FAIL passwordless sudo is required"; exit 3; }}
sudo mkdir -p "$N/models" "$N/cache/vllm" "$N/logs" "$N/work/sampler" "$N/work/stage" "$N/yeto" "$N/docker" "$N/state/containerd"
sudo chown "$(id -u):$(id -g)" "$N/models" "$N/cache" "$N/cache/vllm" "$N/logs" "$N/work" "$N/work/sampler" "$N/work/stage" "$N/yeto"
export DEBIAN_FRONTEND=noninteractive
command -v docker >/dev/null || {{ APT="sudo apt-get -o DPkg::Lock::Timeout=900"; $APT update -qq >/dev/null 2>&1 || {{ echo "YETO_NC_DOCKER_FAIL apt-get update"; exit 3; }}; $APT install -y -qq docker.io >/dev/null 2>&1 || {{ echo "YETO_NC_DOCKER_FAIL apt-get install docker.io"; exit 3; }}; }}
if ! mountpoint -q /var/lib/containerd || ! grep -q "$N/docker" /etc/docker/daemon.json 2>/dev/null; then
  sudo systemctl stop docker.socket docker containerd >/dev/null 2>&1 || true
  if ! mountpoint -q /var/lib/containerd; then
    sudo mkdir -p /var/lib/containerd
    sudo rsync -aHAX /var/lib/containerd/ "$N/state/containerd/" 2>/dev/null || true
    sudo mount --bind "$N/state/containerd" /var/lib/containerd
    grep -q " /var/lib/containerd none bind" /etc/fstab || echo "$N/state/containerd /var/lib/containerd none bind 0 0" | sudo tee -a /etc/fstab >/dev/null
  fi
  sudo mkdir -p /etc/docker; echo '{{"data-root":"'"$N"'/docker"}}' | sudo tee /etc/docker/daemon.json >/dev/null
  sudo systemctl start containerd docker
fi
sudo usermod -aG docker,video,render "$(id -un)" >/dev/null 2>&1 || true
echo "YETO_NC_DOCKER_OK $(sudo docker --version) root=$(sudo docker info -f '{{{{.DockerRootDir}}}}') containerd_bind=$(mountpoint -q /var/lib/containerd && echo y || echo n) rootfree=$(df -h / | awk 'NR==2{{print $4}}')"
"""


def container_name(shard: str) -> str:
    return f"yeto-sampler-{shard}"


def docker_run_argv(node: Node, *, name: str, command: Sequence[str], env: Mapping[str, str],
                    yeto_dir: str | None, env_file: str | None, detach: bool,
                    mounts: Mapping[str, str] | None = None,
                    extra_env_files: Sequence[str] = ()) -> list[str]:
    """``docker run`` on ``node``: its device flags and mounts, env values by ``-e K=V``
    (non-secret only; secrets come from ``--env-file`` on the node). ``extra_env_files``:
    further env files on the node, read after ``env_file`` (e.g. the Harbor settings
    of an agentic sampler)."""
    argv = ["sudo", "docker", "run", "-d" if detach else "--rm", "--name", name, "--restart=no",
            "--network", "host", "--ipc", "host", *node.docker_args]
    for host_path, container_path in sorted((mounts or node.mounts).items()):
        ro = ":ro" if container_path.startswith("/models") else ""
        argv += ["-v", f"{host_path}:{container_path}{ro}"]
    if yeto_dir:
        argv += ["-v", f"{yeto_dir}:/yeto:ro"]
    for path in (env_file, *extra_env_files):
        if path:
            argv += ["--env-file", path]
    for key, value in sorted(env.items()):
        argv += ["-e", f"{key}={value}"]
    return [*argv, "--entrypoint", "bash", node.image or IMAGE, "-c", shlex.join(list(command))]


def sampler_env(run_prefix: str) -> dict[str, str]:
    """The store variables of a run (bucket and endpoint come from the env file on the node)."""
    return {"PYTHONPATH": "/yeto", "PYTHONUNBUFFERED": "1", "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "1", "YETO_RL_STORE_IMPL": "yeto.cloud.s3_store:store_from_env",
            "YETO_RL_STORE_S3_STAGE": "/work/stage", "YETO_RL_STORE_S3_REGION": "auto",
            "YETO_RL_STORE_S3_CREDENTIALS": "aws-env", "YETO_NC_RUN_PREFIX": run_prefix,
            # release download: 8 processes x 4 files x 16 parts took a 207 GB K3.1 release
            # in 144 s (1.43 GB/s mean, 5.2 GB/s 5-second peak) on an MI355X node
            "YETO_RL_STORE_DOWNLOAD_PROCESSES": "8", "YETO_RL_STORE_FILE_CONCURRENCY": "32",
            # each node starts a release download at its own fixed point of 30 s: 16 nodes do not
            # send their first 512 range reads each in the same second (policy_release.download_delay_s)
            "YETO_RL_RELEASE_SPREAD_S": "30"}


# OpenSSH reads ~/.ssh of the passwd entry, not $HOME: an agentic env file that sets
# HOME=/work/harbor/home (DOCKER_HOST=ssh://<sandbox host>) needs that .ssh linked into
# the container user's home, or the docker CLI cannot resolve the host alias. The
# files on the node must be owned by the container user (root) with mode 0600 and name
# container paths (ssh refuses a config of another owner).
SSH_HOME_LINK = ('P="$(getent passwd "$(id -u)" | cut -d: -f6)"; '
                 'if [ -d "$HOME/.ssh" ] && [ "$HOME" != "$P" ] && [ ! -e "$P/.ssh" ]; then '
                 'ln -s "$HOME/.ssh" "$P/.ssh"; fi; ')


def sampler_command(spec_in_container: str) -> list[str]:
    # bucket and endpoint are only known inside (env file); the URI is built there
    return ["bash", "-c",
            (SSH_HOME_LINK +
             'export YETO_RL_STORE_S3_URI="s3://${R2_BUCKET}/runs/${YETO_NC_RUN_PREFIX}" '
             'YETO_RL_STORE_S3_ENDPOINT_URL="$AWS_ENDPOINT_URL"; '
             f"exec {PYTHON} -m yeto.rl.engine.engine_sampler --spec {shlex.quote(spec_in_container)}")]


PREFLIGHT_PY = r"""
import json, sys
from yeto.rl.engine import engine_sampler_entry as e
spec = e.load_spec(sys.argv[1])
spec = dict(spec, store=sys.argv[2], buffer_root=sys.argv[2])
cfg = e.island_config(spec)
factory = e._import(spec["engine"]["factory"])
stores = e.build_stores(spec, environ={})
reward = e.build_reward(spec)
print("YETO_NC_PREFLIGHT_OK", json.dumps({"sampler_id": spec["sampler_id"], "adapter": spec["engine"].get("adapter"),
      "preset": (spec["engine"].get("options") or {}).get("k31_preset"), "factory": spec["engine"]["factory"],
      "config": type(cfg).__name__, "stores": [type(s).__name__ for s in stores]}))
"""


def probe_command(spec_in_container: str | None) -> list[str]:
    """``vllm --version`` + ``rocm-smi`` + (with a spec) the sampler's spec/import/store
    checks against a scratch local store; no engine start, no object store."""
    script = ("set -u; echo YETO_NC_VLLM $(vllm --version 2>/dev/null | tail -1); "
              "rocm-smi --showproductname 2>&1 | grep -E 'GPU\\[' | head -16; "
              "echo YETO_NC_GPUS $(rocm-smi --showproductname 2>/dev/null | grep -c 'Card Series')")
    if spec_in_container:
        script += (f"; cd /tmp && PYTHONPATH=/yeto {PYTHON} -c {shlex.quote(PREFLIGHT_PY)} "
                   f"{shlex.quote(spec_in_container)} /work/dry-preflight")
    return ["bash", "-c", script]


def upload_code(alias: str, *, ref: str = "HEAD", run: Callable[..., Any] = subprocess.run) -> str:
    """``git archive`` of ``yeto/`` at ``ref`` into ``/mnt/nvme/yeto/<sha>`` (idempotent)."""
    sha = run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", ref], capture_output=True,
              text=True, check=True).stdout.strip()
    dest = f"{NVME}/yeto/{sha}"
    archive = subprocess.Popen(["git", "-C", str(REPO_ROOT), "archive", "--format=tar", ref, "yeto"],
                               stdout=subprocess.PIPE)
    try:
        run(["ssh", "-o", "BatchMode=yes", alias,
             (f"test -d {dest}/yeto && cat >/dev/null || (mkdir -p {dest}.tmp && tar x -C {dest}.tmp "
             f"&& mv {dest}.tmp {dest})")], stdin=archive.stdout, check=True, timeout=600)
    finally:
        # Drop the launcher's read end so an ssh that ended early gives
        # git archive EPIPE instead of a pipe that blocks it forever.
        if archive.stdout is not None:
            archive.stdout.close()
        archive.wait()
    return dest


def upload_file(alias: str, local: str | Path, remote: str, *, mode: str = "600",
                run: Callable[..., Any] = subprocess.run) -> None:
    run(["scp", "-q", "-o", "BatchMode=yes", str(local), f"{alias}:{remote}"], check=True, timeout=300)
    run(["ssh", "-n", "-o", "BatchMode=yes", alias, f"chmod {mode} {remote}"], check=True, timeout=60)


def ssh(alias: str, command: str, *, run: Callable[..., Any] = subprocess.run, timeout: float = 1800,
        input: str | None = None):
    argv = ["ssh", "-T", "-o", "BatchMode=yes", alias, command]
    if input is None:
        argv.insert(1, "-n")
    return run(argv, capture_output=True, text=True, timeout=timeout, check=False, input=input)


def start_sampler(lease: NodeLease, *, spec: Mapping[str, Any], shard: str, run_prefix: str,
                  r2_env_file: str, image: str = IMAGE, protected: frozenset[str] = frozenset(),
                  state_dir: Path, agentic_env_file: str | None = None, spec_dir: str | Path | None = None,
                  emit: Callable[[dict[str, Any]], None] | None = None,
                  run: Callable[..., Any] = subprocess.run) -> str:
    """Upload code, spec and env file; start the sampler container detached. Returns its name.

    ``agentic_env_file``: a local env file of an agentic sampler (``HARBOR_*``,
    ``DOCKER_HOST``, ``YETO_RECORDS_PROXY_*``), uploaded next to the spec and passed
    as a second ``--env-file``; refused for a spec without an ``agentic`` section.

    ``agentic.setup`` in the spec (:mod:`yeto.launch.agentic_setup`): the node is
    prepared first (Harbor, docker CLI, docker key, reverse tunnel, task
    directories, records secret; idempotent), and the setup's environment goes in
    as an env file ahead of ``agentic_env_file``. Relative local paths of the
    section resolve against ``spec_dir``; ``emit`` gets the setup events."""
    prefix = run_prefix.strip("/")
    if not prefix:
        raise ValueError("run_prefix is required")
    if prefix in protected:
        raise ValueError(f"run prefix {prefix!r} is protected (a buffer in use)")
    if spec.get("sampler_id") != shard:
        raise ValueError(f"spec sampler_id {spec.get('sampler_id')!r} is not the shard {shard!r}")
    if agentic_env_file and not spec.get("agentic"):
        raise ValueError("an agentic env file needs a spec with an agentic section")
    setup = agentic_setup.parse(spec, base_dir=spec_dir)
    if setup is not None and (problems := agentic_setup.local_problems(setup)):
        raise ValueError("agentic setup: " + "; ".join(problems))
    alias, node = host_alias(lease.ticket), node_of(lease, image=image)
    # a restarted NC sampler starts on the newest release, not a round on the base
    # weights (engine_sampler_entry ``start_on_latest``; a spec may say false)
    spec = {"start_on_latest": True, **spec}
    yeto_dir = upload_code(alias, run=run)
    local = state_dir / f"spec-{shard}.json"
    local.write_text(json.dumps(agentic_setup.strip(spec), indent=1))  # local paths stay here
    upload_file(alias, local, f"{NVME}/work/sampler/{shard}.json", mode="644", run=run)
    env_on_node = f"/home/{lease.user}/.yeto-nc-r2.env"
    upload_file(alias, r2_env_file, env_on_node, run=run)
    extra: list[str] = []
    env = sampler_env(prefix)
    # the release download's start spread is taken from this key (policy_release.download_delay_s):
    # with --network host the host name is the node's, which NC nodes need not keep distinct
    env["YETO_RL_RELEASE_SPREAD_KEY"] = f"{prefix}/{shard}"
    if setup is not None:
        target = agentic_setup.SshNode(alias, node.image or IMAGE, NVME, yeto_dir, PYTHON, lease.user,
                                       prelude=SSH_HOME_LINK, run=run)
        agentic_setup.Provisioner(setup, target, emit=emit or _print_event, store_env_file=env_on_node).run()
        setup_env = state_dir / f"{shard}.agentic-setup.env"
        setup_env.write_text("".join(f"{k}={v}\n" for k, v in sorted(
            setup.container_env(python_bin=str(Path(PYTHON).parent)).items())))
        extra.append(f"{NVME}/work/sampler/{shard}.agentic-setup.env")
        upload_file(alias, setup_env, extra[-1], mode="600", run=run)
        env["PYTHONPATH"] = setup.python_path()
    if agentic_env_file:
        extra.append(f"{NVME}/work/sampler/{shard}.agentic.env")  # after the setup's: the operator's file wins
        upload_file(alias, agentic_env_file, extra[-1], mode="600", run=run)
    name = container_name(shard)
    argv = docker_run_argv(node, name=name, command=sampler_command(f"/work/sampler/{shard}.json"),
                           env=env, yeto_dir=yeto_dir, env_file=env_on_node,
                           detach=True, extra_env_files=extra)
    out = ssh(alias, f"sudo docker rm -f {name} >/dev/null 2>&1; {shlex.join(argv)}", run=run, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(f"docker run {name}: {(out.stderr or '')[-400:]}")
    return name


def _print_event(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def stop_sampler(lease: NodeLease, shard: str, *, grace_s: int = 60, run: Callable[..., Any] = subprocess.run):
    name = container_name(shard)
    return ssh(host_alias(lease.ticket), f"sudo docker stop -t {int(grace_s)} {name} >/dev/null 2>&1; "
               f"sudo docker logs --tail 200 {name} > {NVME}/logs/{name}.last.log 2>&1; "
               f"sudo docker rm -f {name} >/dev/null 2>&1; echo stopped {name}", run=run, timeout=grace_s + 120)


def sampler_status(lease: NodeLease, shard: str, *, run: Callable[..., Any] = subprocess.run) -> str:
    name = container_name(shard)
    out = ssh(host_alias(lease.ticket), f"sudo docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' "
              f"{name} 2>/dev/null || echo absent", run=run, timeout=120)
    return (out.stdout or "").strip()


def probe(lease: NodeLease, *, spec: Mapping[str, Any] | None, state_dir: Path, image: str = IMAGE,
          run: Callable[..., Any] = subprocess.run) -> str:
    """Pull the image, card check on the host, then the probe container (``--rm``)."""
    alias, node = host_alias(lease.ticket), node_of(lease, image=image)
    lines = []
    for cmd in node.preflight_commands():
        out = ssh(alias, shlex.join(cmd), run=run, timeout=120)
        if out.returncode == 0 and out.stdout.strip():
            cards = node.check_cards(out.stdout)
            lines.append(f"YETO_NC_CARDS_OK {len(cards)} x {node.card} ({cmd[0]})")
            break
    else:
        lines.append("YETO_NC_CARDS_UNCHECKED no card tool answered")
    pull = ssh(alias, f"sudo docker pull -q {shlex.quote(image)}", run=run, timeout=1800)
    lines.append(f"pull: {(pull.stdout or pull.stderr).strip()[-200:]}")
    yeto_dir, spec_path = None, None
    if spec is not None:
        yeto_dir = upload_code(alias, run=run)
        local = state_dir / "spec-probe.json"
        local.write_text(json.dumps(spec, indent=1))
        upload_file(alias, local, f"{NVME}/work/sampler/probe.json", mode="644", run=run)
        spec_path = "/work/sampler/probe.json"
    argv = docker_run_argv(node, name="yeto-nc-probe", command=probe_command(spec_path), env={},
                           yeto_dir=yeto_dir, env_file=None, detach=False,
                           mounts={f"{NVME}/work": "/work"})
    out = ssh(alias, shlex.join(argv), run=run, timeout=900)
    lines.append((out.stdout or "").strip())
    if out.returncode != 0:
        lines.append(f"probe exit {out.returncode}: {(out.stderr or '').strip()[-600:]}")
    return "\n".join(lines)


PULL_IMAGE = "amazon/aws-cli:latest"
PULL_CONTAINER = "yeto-nc-pull"


PULL_JOBS = 16  # aws-cli processes; one classic-client process tops out near 275 MB/s on R2


def pull_body(src_key: str, dest: str, *, max_mb_s: int = 2500, include: str | None = None,
              jobs: int = PULL_JOBS) -> str:
    """Shell of a capped, resumable parallel ``aws s3 cp`` of ``s3://$R2_BUCKET/<src_key>``
    into ``dest``: a path where the shell runs (``/models/...`` in the aws-cli container,
    or the sandbox path of a no-docker sampler). Credentials come from the environment."""
    if max_mb_s > 3000:
        raise ValueError(f"pull cap {max_mb_s} MB/s exceeds the node limit 3000 MB/s")
    jobs = max(1, int(jobs))
    src = src_key.strip("/")
    one = ('rel=${1#' + src + '/}; case "$rel" in $YETO_PULL_INCLUDE) ;; *) exit 0;; esac; '
           f'f={shlex.quote(dest)}/$rel; '
           '[ -f "$f" ] && [ "$(wc -c < "$f" | tr -d " ")" = "$2" ] && exit 0; mkdir -p "$(dirname "$f")"; '
           'aws s3 cp --only-show-errors "s3://$R2_BUCKET/$1" "$f" --endpoint-url "$AWS_ENDPOINT_URL" '
           '|| { echo "YETO_NC_PULL fail $1"; exit 255; }')
    return (f"export YETO_PULL_INCLUDE={shlex.quote(include or '*')} && "
            "aws configure set default.s3.max_concurrent_requests 16 && "
            f"aws configure set default.s3.max_bandwidth {max(1, int(max_mb_s) // jobs)}MB/s && "
            f'echo "YETO_NC_PULL start $(date -u +%FT%TZ) jobs={jobs}" && '
            # the listing is read before any copy: a refused or empty listing is a failed
            # pull, not a "done" with nothing copied (a pipeline exits with xargs' status)
            f'{{ L=$(aws s3 ls --recursive "s3://$R2_BUCKET/{src}/" --endpoint-url "$AWS_ENDPOINT_URL") '
            f'|| {{ echo "YETO_NC_PULL fail listing s3://$R2_BUCKET/{src}/"; exit 3; }}; }} && '
            f'{{ [ -n "$L" ] || {{ echo "YETO_NC_PULL fail empty listing s3://$R2_BUCKET/{src}/"; exit 3; }}; }} && '
            'printf "%s\\n" "$L" | awk \'{print $4, $3}\' | sort -k2 -n -r '
            f"| xargs -r -P {jobs} -L1 sh -c {shlex.quote(one)} pull "
            '&& echo "YETO_NC_PULL done $(date -u +%FT%TZ)"')


def pull_model_command(src_key: str, dest: str, *, max_mb_s: int = 2500, include: str | None = None,
                       jobs: int = PULL_JOBS) -> str:
    """Node shell: a detached aws-cli container copying ``s3://$R2_BUCKET/<src_key>`` to
    ``dest`` with the bandwidth cap (~7 GB/s drops an MI355X node off the network for
    good, GPU-B section 4.1). Credentials come from the env file on the node.

    One ``aws s3 sync`` (classic transfer client, which ``max_bandwidth`` forces)
    reached ~275 MB/s on an MI355X node, 95 minutes for the 1.56 TB K3.1 base; so
    ``jobs`` ``aws s3 cp`` processes share the object list, each capped at
    ``max_mb_s / jobs``. An object already at ``dest`` with the listed size is skipped
    (resume); ``include`` is a shell glob on the key below ``src_key``."""
    if not dest.startswith(f"{NVME}/models/"):
        raise ValueError(f"dest must be under {NVME}/models")
    body = pull_body(src_key, dest.replace(f"{NVME}/models", "/models", 1), max_mb_s=max_mb_s, include=include,
                     jobs=jobs)
    return (f"sudo docker rm -f {PULL_CONTAINER} >/dev/null 2>&1; sudo docker run -d --name {PULL_CONTAINER} "
            f"--network host --env-file \"$HOME/.yeto-nc-r2.env\" -v {NVME}/models:/models "
            f"--entrypoint sh {PULL_IMAGE} -c {shlex.quote(body)}")


def pull_model(lease: NodeLease, *, r2_env_file: str, src_key: str, dest: str, include: str | None = None,
               max_mb_s: int = 2500, run: Callable[..., Any] = subprocess.run) -> str:
    alias = host_alias(lease.ticket)
    upload_file(alias, r2_env_file, f"/home/{lease.user}/.yeto-nc-r2.env", run=run)
    out = ssh(alias, pull_model_command(src_key, dest, max_mb_s=max_mb_s, include=include), run=run, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(f"pull container: {(out.stderr or '')[-300:]}")
    return PULL_CONTAINER


def pull_status(lease: NodeLease, *, run: Callable[..., Any] = subprocess.run) -> str:
    out = ssh(host_alias(lease.ticket), f"sudo docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' "
              f"{PULL_CONTAINER} 2>/dev/null || echo absent; sudo docker logs --tail 3 {PULL_CONTAINER} 2>&1; "
              f"du -sh {NVME}/models/* 2>/dev/null", run=run, timeout=120)
    return (out.stdout or "").strip()
