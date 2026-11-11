"""Engine sampler in a Daytona GPU sandbox: no docker, the sampler is a process.

A Daytona GPU sandbox is a VM booted from an OCI snapshot with the GPUs passed
through; there is no docker daemon, the GPU driver is Daytona's, and the sandbox
is reached through an exec handle (SDK ``process.exec`` or SSH through Daytona's
gateway). So the National Compute route (:mod:`yeto.launch.nc.ssh_sampler`:
docker on NVMe, ``docker run`` of the sampler image, an aws-cli container for the
model pull, a helper container for the agentic tunnel) becomes:

* the sampler image *is* the sandbox snapshot (``docker/daytona-sampler``: the
  pinned ``vllm-openai-rocm:kimi-k31`` digest plus the sampler's pure-Python
  dependencies, awscli and openssh-client);
* the sandbox paths are the container paths of the NC route (``/work``,
  ``/models/kimi-k3.1-pro``), so one sampler spec serves both routes;
* :func:`preflight` checks the sandbox before anything is started (GPUs, gfx950,
  ``/dev/kfd`` + ``/dev/dri``, ``/dev/shm``, disk, memory, image, outbound reach)
  and raises :class:`PreflightError` naming each failed check and its remedy;
* the yeto tree of the launcher's commit is uploaded to ``/work/yeto/<sha>``;
* the sampler runs as ``setsid bash <start script>``: its own process group,
  pid in ``/work/sampler/<shard>.pid``, exit code in ``<shard>.exit``, log in
  ``/work/logs/<shard>.log``; it is stopped by its process group, never by name;
* the agentic setup runs through :class:`ProcessProvisioner` (no docker: the
  helper steps run in the sandbox, the reverse tunnel is a process).

What drives the sandbox is a :class:`SandboxExec` the provider layer supplies
(:class:`SshExec` is one over plain ``ssh``). Nothing here creates, stops or
deletes a sandbox.

Network: the sampler only dials out (object store, the sandbox host's docker
over SSH, the reverse tunnel ``ssh -R`` to the sandbox host). Nothing has to
reach the sandbox from outside; the pool's probe runs through the exec handle.
"""

from __future__ import annotations

import io
import json
import shlex
import subprocess
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from yeto.rl.sampler_pins import VLLM_SAMPLER_IMAGE_ROCM, VLLM_SAMPLER_ROCM_VERSION

from . import agentic_setup
from .nc import elastic
from .nc.ssh_sampler import PYTHON, SSH_HOME_LINK, pull_body, sampler_env

WORK = "/work"
MODELS = "/models"
MODEL_DIR = f"{MODELS}/kimi-k3.1-pro"
CACHE = f"{WORK}/cache"
SAMPLER_DIR = f"{WORK}/sampler"
LOG_DIR = f"{WORK}/logs"
CODE_DIR = f"{WORK}/yeto"
STORE_ENV = f"{SAMPLER_DIR}/store.env"
CA_BUNDLE = f"{WORK}/ca-bundle.pem"
# the egress proxy of a Daytona domain allow list re-signs TLS with this CA
DAYTONA_PROXY_CA = "/etc/daytona/netleash/ca.crt"
BASE_IMAGE = VLLM_SAMPLER_IMAGE_ROCM.removeprefix("docker:")
GPU_ARCH = "gfx950"  # MI355X; the in-place expert reload is verified on gfx950 only
REQUIRED_MODULES = ("vllm", "aiter", "torch", "boto3", "safetensors", "zstandard", "httpx", "numpy")
GIB = 1 << 30
# a shell without sudo (the sandbox user is root) runs ``sudo X`` as ``X``
NO_SUDO_SHIM = 'command -v sudo >/dev/null 2>&1 || sudo() { while [ "${1#-}" != "$1" ]; do shift; done; "$@"; }; '


# -- the exec handle (the provider layer supplies it) ---------------------------

@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class SandboxExec(Protocol):
    """What this module needs of a sandbox. ``name``: for messages and events.

    ``exec``: run ``command`` with ``bash -c`` in the sandbox and wait (up to
    ``timeout`` s); the exit code and output. A command that starts the sampler
    detaches it itself (``setsid``), so a handle without background support works.
    ``upload``: write ``data`` to the absolute sandbox path ``remote`` (parents
    created; the caller sets the mode)."""

    name: str

    def exec(self, command: str, *, timeout: float = 600) -> ExecResult: ...
    def upload(self, data: bytes, remote: str) -> None: ...


@dataclass
class SshExec:
    """:class:`SandboxExec` over ``ssh``: ``target`` is the argv after ``ssh``'s options
    that names the sandbox (Daytona: ``["<token>@ssh.app.daytona.io"]``)."""

    name: str
    target: Sequence[str]
    options: Sequence[str] = ("-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30")
    run: Callable[..., Any] = subprocess.run

    def exec(self, command: str, *, timeout: float = 600) -> ExecResult:
        out = self.run(["ssh", "-n", *self.options, *self.target, "bash -c " + shlex.quote(command)],
                       capture_output=True, text=True, timeout=timeout, check=False)
        return ExecResult(out.returncode, out.stdout or "", out.stderr or "")

    def upload(self, data: bytes, remote: str) -> None:
        q = shlex.quote(remote)
        out = self.run(["ssh", *self.options, *self.target,
                        f"mkdir -p \"$(dirname {q})\" && cat > {q}.part && mv -f {q}.part {q}"],
                       input=data, capture_output=True, timeout=1800, check=False)
        if out.returncode != 0:
            err = out.stderr.decode(errors="replace") if isinstance(out.stderr, bytes) else str(out.stderr or "")
            raise RuntimeError(f"upload to {self.name}:{remote} failed: {err.strip()[-300:]}")


def sh(box: SandboxExec, command: str, *, timeout: float = 600) -> ExecResult:
    return box.exec(NO_SUDO_SHIM + command, timeout=timeout)


def put(box: SandboxExec, data: bytes | str, remote: str, *, mode: str = "600") -> None:
    """Upload and set the mode (written to a temporary name first: never half a file)."""
    raw = data.encode() if isinstance(data, str) else data
    box.upload(raw, remote + ".up")
    q = shlex.quote(remote)
    out = sh(box, f"chmod {mode} {q}.up && mv -f {q}.up {q}", timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"{box.name}: placing {remote}: {out.stderr.strip()[-300:]}")


# -- sizing ----------------------------------------------------------------------

@dataclass(frozen=True)
class DiskPlan:
    """The sandbox disk a sampler needs (GiB). At the peak the release cache holds
    the served release and the next one being downloaded (``PolicyFollower``
    prunes every other download first); overlays are symlinks."""

    base_gib: float
    release_gib: float
    releases_on_disk: int = 2
    image_gib: float = 60.0   # the snapshot root file system (ROCm + torch + vLLM)
    cache_gib: float = 64.0   # vLLM / AITER / triton JIT caches, compile cache
    agentic_gib: float = 64.0  # Harbor site, task directories, trial logs, event tapes
    headroom: float = 0.15

    @property
    def required_gib(self) -> int:
        need = (self.base_gib + self.release_gib * self.releases_on_disk + self.image_gib + self.cache_gib
                + self.agentic_gib)
        return int(-(-need * (1 + self.headroom) // 1))

    @property
    def data_gib(self) -> int:
        """What must be free under ``/work`` and ``/models`` once the snapshot is up."""
        need = self.base_gib + self.release_gib * self.releases_on_disk + self.cache_gib + self.agentic_gib
        return int(-(-need // 1))


def k31_disk_plan(*, base_gib: float = 1453.0, release_gib: float = 193.0) -> DiskPlan:
    """Kimi-K3.1 MXFP4 sampler: base checkpoint + 2 releases + image + caches.
    Defaults: the 1.56 TB (1453 GiB) Kimi-K3.1-Pro base and a 128-expert release
    (207 GB, 193 GiB); a 64-expert release is about 96 GiB."""
    return DiskPlan(base_gib=base_gib, release_gib=release_gib)


# -- preflight -------------------------------------------------------------------

@dataclass
class Requirements:
    gpus: int = 8
    gpu_arch: str = GPU_ARCH
    shm_gib: float = 64.0
    memory_gib: float = 512.0
    data_gib: float = 0.0  # free under /work and /models (DiskPlan.data_gib); 0: unchecked
    vllm_version: str = VLLM_SAMPLER_ROCM_VERSION
    python: str = PYTHON
    hot_reload: bool = True  # the spec's engine.server.in_place_reload
    store: bool = True       # the object store endpoint must answer
    ssh_hosts: tuple[str, ...] = ()  # host:port the sampler dials (sandbox docker, tunnel)
    modules: tuple[str, ...] = REQUIRED_MODULES


REMEDY = {
    "kfd": "/dev/kfd is missing or not read-write for the sandbox user: the sandbox has no AMD GPU "
           "(request gpuType MI355X) or the user lacks the video/render groups",
    "dri": "fewer /dev/dri/renderD* nodes than GPUs: create the sandbox with the full GPU count",
    "gpus": "torch sees fewer GPUs than required: check HIP_VISIBLE_DEVICES and the sandbox's GPU count",
    "arch": "the GPUs are not gfx950 (MI355X): the K3.1 MXFP4 path and its in-place expert reload "
            "(AITER_MXFP4_BF16) are refused elsewhere",
    "rocm": "torch cannot initialise HIP: the sandbox's amdgpu driver does not serve the image's ROCm "
            "user space; compare the driver line with the image's ROCm and ask Daytona for a matching host",
    "shm": "/dev/shm is smaller than required (vLLM TP workers exchange through it): ask for a larger "
           "sandbox memory size or remount /dev/shm with size=<N>g if the user may",
    "memory": "host memory is below the requirement: create the sandbox with more memory",
    "disk": "free disk under /work or /models is below the plan: create the sandbox with a larger disk "
            "(GPU sandboxes default to 512 GiB whatever the GPU count)",
    "write": "/work or /models is not writable: the snapshot must create both for the sandbox user",
    "python": "the sampler interpreter is missing: the snapshot is not the yeto Daytona sampler image",
    "modules": "the image lacks Python modules the sampler imports: rebuild the snapshot from "
               "docker/daytona-sampler",
    "vllm": "the image's vLLM is not the pinned build: rebuild the snapshot from the pinned digest",
    "aws": "awscli is missing: the model pull and store task sync need it (rebuild the snapshot)",
    "store": "the object store endpoint does not answer from the sandbox: Tier 1/2 Daytona organisations "
             "restrict egress; allow the endpoint's domain or use Tier 3+",
    "ssh": "an SSH host the sampler dials (sandbox docker, reverse tunnel) is not reachable: allow it in the "
           "Daytona network policy (Tier 3+: CIDR list) or move the sandbox host",
}

PREFLIGHT_PY = r"""
import importlib.util, json, sys
mods = sys.argv[1].split(",")
out = {"missing": [m for m in mods if importlib.util.find_spec(m) is None]}
try:
    import torch
    out["torch"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)
    out["available"] = bool(torch.cuda.is_available())
    n = torch.cuda.device_count() if out["available"] else 0
    out["count"] = n
    out["arch"] = sorted({torch.cuda.get_device_properties(i).gcnArchName.split(":")[0] for i in range(n)})
    out["names"] = sorted({torch.cuda.get_device_name(i) for i in range(n)})
except Exception as exc:
    out["error"] = repr(exc)[:300]
print("YETO_DAYTONA_PY " + json.dumps(out))
"""


def preflight_script(req: Requirements, *, env_file: str | None = STORE_ENV) -> str:
    """Sandbox shell printing ``YETO_DAYTONA_FACT <key> <value>`` lines and one
    ``YETO_DAYTONA_PY {json}`` line; read-only (it creates nothing), no secrets printed."""
    py = shlex.quote(req.python)
    lines = [
        "set -u",
        'f() { echo "YETO_DAYTONA_FACT $1 ${2:-}"; }',
        'f user "$(id -un) $(id -Gn | tr " " ,)"',
        'if [ -e /dev/kfd ] && [ -r /dev/kfd ] && [ -w /dev/kfd ]; then f kfd rw; '
        'elif [ -e /dev/kfd ]; then f kfd noaccess; else f kfd absent; fi',
        "f dri $(ls /dev/dri 2>/dev/null | grep -c '^renderD')",
        "f driver \"$(cat /sys/module/amdgpu/version 2>/dev/null || echo -) "
        "$(uname -r)\"",
        "f rocm_image \"$(cat /opt/rocm/.info/version 2>/dev/null || echo -)\"",
        "f shm_kib $(df -Pk /dev/shm 2>/dev/null | awk 'NR==2{print $2}')",
        "f mem_kib $(awk '/MemTotal/{print $2}' /proc/meminfo)",
        "f memlock \"$(ulimit -l)\"",
        f"for d in {WORK} {MODELS}; do f free_kib:$d $(df -Pk $d 2>/dev/null | awk 'NR==2{{print $4}}'); "
        "t=$d/.yeto-write-probe; if (: > $t) 2>/dev/null; then rm -f $t; f write:$d ok; else f write:$d no; fi; done",
        f"[ -x {py} ] && f python ok || f python absent",
        "command -v aws >/dev/null && f aws ok || f aws absent",
        "command -v docker >/dev/null && f docker_cli yes || f docker_cli no",
        f"f proxy_ca $([ -f {DAYTONA_PROXY_CA} ] && echo yes || echo no)",
        f"f vllm \"$({py} -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)\"",
        f"cd /tmp && {py} -c {shlex.quote(PREFLIGHT_PY)} {shlex.quote(','.join(req.modules))} 2>&1 | "
        "grep '^YETO_DAYTONA_PY' || echo 'YETO_DAYTONA_PY {\"error\": \"interpreter failed\"}'",
    ]
    if req.store:
        src = f"set -a; . {shlex.quote(env_file)}; set +a; " if env_file else ""
        lines.append(f"( {src}c=$(curl -s -o /dev/null -m 15 -w '%{{http_code}}' ${{AWS_CA_BUNDLE:+--cacert $AWS_CA_BUNDLE}} "
                     "\"${AWS_ENDPOINT_URL:-}\" 2>/dev/null); f store \"${c:-000}\" )")
    for hostport in req.ssh_hosts:
        host, _, port = hostport.rpartition(":")
        lines.append(f"timeout 10 bash -c {shlex.quote(f'</dev/tcp/{host}/{int(port)}')} 2>/dev/null "
                     f"&& f ssh:{hostport} open || f ssh:{hostport} closed")
    return "\n".join(lines) + "\n"


def parse_facts(text: str) -> tuple[dict[str, str], dict[str, Any]]:
    facts: dict[str, str] = {}
    py: dict[str, Any] = {}
    for line in (text or "").splitlines():
        if line.startswith("YETO_DAYTONA_FACT "):
            _, key, *rest = line.split(" ", 2)
            facts[key] = rest[0].strip() if rest else ""
        elif line.startswith("YETO_DAYTONA_PY "):
            try:
                py = json.loads(line.split(" ", 1)[1])
            except ValueError:
                py = {"error": "unreadable interpreter report"}
    return facts, py


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    warn: bool = False  # recorded, not refused

    @property
    def remedy(self) -> str:
        return "" if self.ok else REMEDY.get(self.name.split(":")[0], "")


# checks that depend on the host the sandbox landed on (another sandbox may pass them);
# every other check fails the same way on every sandbox of the same snapshot and size
NODE_CHECKS = frozenset({"exec", "kfd", "dri", "rocm", "gpus", "arch"})


class PreflightError(RuntimeError):
    """``config_error``: every failed check is one of the snapshot / size / network policy
    (not of the host), so a replacement sandbox would fail the same way: the pool keeps
    the sandbox and halts until ``pool-retry`` instead of ordering more of them."""

    def __init__(self, sandbox: str, failed: Sequence[Check]):
        self.failed = list(failed)
        self.config_error = bool(self.failed) and not any(c.name.split(":")[0] in NODE_CHECKS for c in self.failed)
        body = "; ".join(f"[{c.name}] {c.detail}: {c.remedy}" for c in self.failed)
        super().__init__(f"Daytona sampler preflight failed on {sandbox}: {body}")


def _int(value: str | None) -> int | None:
    try:
        return int(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def evaluate(facts: Mapping[str, str], py: Mapping[str, Any], req: Requirements) -> list[Check]:
    """The checks of a preflight report against ``req`` (pure; the tests feed it)."""
    out: list[Check] = []

    def add(name: str, ok: bool, detail: str, *, warn: bool = False) -> None:
        out.append(Check(name, ok, detail, warn))

    add("kfd", facts.get("kfd") == "rw", f"/dev/kfd {facts.get('kfd', 'unreported')}")
    dri = _int(facts.get("dri")) or 0
    add("dri", dri >= req.gpus, f"{dri} render nodes for {req.gpus} GPUs")
    add("python", facts.get("python") == "ok", f"{req.python} {facts.get('python', 'unreported')}")
    missing = list(py.get("missing") or [])
    add("modules", not missing and "error" not in py or not missing and bool(py.get("torch")),
        f"missing {missing}" if missing else "all importable")
    if py.get("error") and not py.get("torch"):
        add("rocm", False, f"interpreter: {py['error']}")
    else:
        add("rocm", bool(py.get("available")) and bool(py.get("hip")),
            f"torch {py.get('torch')} hip {py.get('hip')} available={py.get('available')} "
            f"driver {facts.get('driver', '-')} image ROCm {facts.get('rocm_image', '-')}"
            + (f" ({py['error']})" if py.get("error") else ""))
    count = int(py.get("count") or 0)
    add("gpus", count >= req.gpus, f"torch sees {count} of {req.gpus}")
    arch = list(py.get("arch") or [])
    if req.gpu_arch:
        add("arch", arch == [req.gpu_arch], f"archs {arch or '-'} names {py.get('names') or '-'}",
            warn=not req.hot_reload and arch != [req.gpu_arch])
    shm = (_int(facts.get("shm_kib")) or 0) / (1 << 20)
    add("shm", shm >= req.shm_gib, f"/dev/shm {shm:.0f} GiB (need {req.shm_gib:g})")
    mem = (_int(facts.get("mem_kib")) or 0) / (1 << 20)
    add("memory", mem >= req.memory_gib, f"{mem:.0f} GiB (need {req.memory_gib:g})")
    for d in (WORK, MODELS):
        add(f"write:{d}", facts.get(f"write:{d}") == "ok", f"{d} writable={facts.get(f'write:{d}', 'unreported')}")
    if req.data_gib:
        free = {d: (_int(facts.get(f"free_kib:{d}")) or 0) / (1 << 20) for d in (WORK, MODELS)}
        # one file system in the usual snapshot: count it once
        total = free[WORK] if abs(free[WORK] - free[MODELS]) < 1 else free[WORK] + free[MODELS]
        add("disk", total >= req.data_gib, f"free {total:.0f} GiB under {WORK} + {MODELS} (need {req.data_gib:g})")
    have = facts.get("vllm", "")
    add("vllm", have == req.vllm_version, f"vllm {have or '-'} (pin {req.vllm_version})",
        warn=have != req.vllm_version and bool(have))
    add("aws", facts.get("aws") == "ok", f"aws {facts.get('aws', 'unreported')}")
    memlock = facts.get("memlock", "")
    add("memlock", memlock == "unlimited", f"ulimit -l {memlock or '-'}", warn=memlock != "unlimited")
    if req.store:
        code = _int(facts.get("store")) or 0
        add("store", 0 < code < 500, f"object store endpoint HTTP {code or 'no answer'}"
            + (" (egress proxy CA present)" if facts.get("proxy_ca") == "yes" else ""))
    for hostport in req.ssh_hosts:
        add(f"ssh:{hostport}", facts.get(f"ssh:{hostport}") == "open",
            f"tcp {hostport} {facts.get(f'ssh:{hostport}', 'unreported')}")
    return out


def preflight(box: SandboxExec, req: Requirements, *, env_file: str | None = STORE_ENV,
              emit: Callable[[dict[str, Any]], None] | None = None) -> list[Check]:
    """Run the checks in the sandbox; raises :class:`PreflightError` on a failed one."""
    out = box.exec(preflight_script(req, env_file=env_file), timeout=600)
    facts, py = parse_facts(out.stdout)
    if not facts:
        raise PreflightError(box.name, [Check("exec", False, f"no report (exit {out.returncode}: "
                                                            f"{out.stderr.strip()[-300:]})")])
    checks = evaluate(facts, py, req)
    if emit is not None:
        emit({"event": "daytona_sampler_preflight", "sandbox": box.name,
              "checks": {c.name: ("ok" if c.ok else "warn" if c.warn else "fail") for c in checks},
              "driver": facts.get("driver"), "rocm_image": facts.get("rocm_image"), "vllm": facts.get("vllm")})
    failed = [c for c in checks if not c.ok and not c.warn]
    if failed:
        raise PreflightError(box.name, failed)
    return checks


def requirements_of(spec: Mapping[str, Any], *, gpus: int = 8, plan: DiskPlan | None = None,
                    **overrides: Any) -> Requirements:
    """The requirements a sampler spec implies (hot reload, the SSH hosts it dials)."""
    server = (spec.get("engine") or {}).get("server") or {}
    hosts: list[str] = []
    setup = agentic_setup.parse(spec) if (spec.get("agentic") or {}).get("setup") else None
    if setup is not None:
        hosts.append(f"{setup.sandbox.host}:{setup.sandbox.port}")
        if setup.tunnel is not None:
            hosts.append(f"{setup.tunnel.host}:{setup.tunnel.port}")
    req = Requirements(gpus=gpus, hot_reload=bool(server.get("in_place_reload")),
                       data_gib=float(plan.data_gib) if plan else 0.0, ssh_hosts=tuple(dict.fromkeys(hosts)))
    for key, value in overrides.items():
        setattr(req, key, value)
    return req


# -- code, env, start / stop -------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def code_tar(ref: str = "HEAD", *, run: Callable[..., Any] = subprocess.run) -> tuple[str, bytes]:
    """``(short sha, tar of yeto/)`` of ``ref`` (``git archive``; a clean commit only)."""
    sha = run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", ref], capture_output=True, text=True,
              check=True).stdout.strip()
    data = run(["git", "-C", str(REPO_ROOT), "archive", "--format=tar", ref, "yeto"], capture_output=True,
               check=True).stdout
    return sha, data


def upload_code(box: SandboxExec, sha: str, tar: bytes) -> str:
    """The tree at ``/work/yeto/<sha>`` (idempotent: an existing tree is kept)."""
    dest = f"{CODE_DIR}/{sha}"
    if sh(box, f"test -d {dest}/yeto", timeout=60).returncode == 0:
        return dest
    box.upload(tar, f"{dest}.tar")
    out = sh(box, f"rm -rf {dest}.tmp && mkdir -p {dest}.tmp && tar xf {dest}.tar -C {dest}.tmp && "
                  f"mv {dest}.tmp {dest} && rm -f {dest}.tar", timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"{box.name}: unpacking the yeto tree: {out.stderr.strip()[-300:]}")
    return dest


def process_env(run_prefix: str, yeto_dir: str, *, gpus: int = 8) -> dict[str, str]:
    """The NC container's environment for a process in the sandbox."""
    env = sampler_env(run_prefix)
    env.update({"PYTHONPATH": yeto_dir, "HIP_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpus)),
                "YETO_RL_STORE_S3_STAGE": f"{WORK}/stage", "XDG_CACHE_HOME": CACHE,
                "VLLM_CACHE_ROOT": f"{CACHE}/vllm", "TRITON_CACHE_DIR": f"{CACHE}/triton",
                "AITER_JIT_DIR": f"{CACHE}/aiter",
                # the release cache (served + next release, DiskPlan) on the sized disk under /work,
                # not ~/.cache on the root file system (the NC container mounts /root/.cache on NVMe)
                "YETO_CACHE_DIR": f"{CACHE}/yeto"})
    return env


def paths(shard: str) -> dict[str, str]:
    return {"spec": f"{SAMPLER_DIR}/{shard}.json", "script": f"{SAMPLER_DIR}/{shard}.start.sh",
            "pid": f"{SAMPLER_DIR}/{shard}.pid", "exit": f"{SAMPLER_DIR}/{shard}.exit",
            "log": f"{LOG_DIR}/{shard}.log"}


def ca_bundle_script() -> str:
    """A combined CA file when Daytona's egress proxy re-signs TLS (domain allow list):
    the system roots plus the proxy CA, for boto3 / requests / curl."""
    return (f"if [ -f {DAYTONA_PROXY_CA} ]; then cat /etc/ssl/certs/ca-certificates.crt {DAYTONA_PROXY_CA} "
            f"> {CA_BUNDLE}.tmp 2>/dev/null && mv -f {CA_BUNDLE}.tmp {CA_BUNDLE}; fi")


def start_script(env: Mapping[str, str], *, spec_path: str, env_files: Sequence[str],
                 python: str = PYTHON) -> str:
    """The sampler's start script: the env files (secrets) sourced in order, the
    non-secret environment, the store URI, then ``exec`` of the engine sampler."""
    lines = ["#!/bin/bash", "set -eu", "set -a"]
    lines += [f". {shlex.quote(p)}" for p in env_files]
    lines += ["set +a"]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items())]
    lines += [f"if [ -f {CA_BUNDLE} ]; then export AWS_CA_BUNDLE={CA_BUNDLE} REQUESTS_CA_BUNDLE={CA_BUNDLE} "
              f"SSL_CERT_FILE={CA_BUNDLE}; fi",
              'export YETO_RL_STORE_S3_URI="s3://${R2_BUCKET}/runs/${YETO_NC_RUN_PREFIX}" '
              'YETO_RL_STORE_S3_ENDPOINT_URL="$AWS_ENDPOINT_URL"',
              SSH_HOME_LINK.rstrip(),
              f"mkdir -p {WORK}/stage {CACHE}",
              f"cd {WORK}",
              f"exec {shlex.quote(python)} -m yeto.rl.engine.engine_sampler --spec {shlex.quote(spec_path)}"]
    return "\n".join(lines) + "\n"


def _alive(pidfile: str) -> str:
    """Shell test: the pid in ``pidfile`` is alive and is our start script (no name matching)."""
    return (f"{{ p=$(cat {pidfile} 2>/dev/null); [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null && "
            f"tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q {MARKER}; }}")


MARKER = "yeto-sampler-start"


def launch_command(shard: str) -> str:
    """Detach the start script in its own session (process group = its pid)."""
    p = paths(shard)
    # $0 of the session leader is the marker _alive reads in /proc/<pid>/cmdline
    inner = f"bash {shlex.quote(p['script'])} >> {shlex.quote(p['log'])} 2>&1; echo $? > {shlex.quote(p['exit'])}"
    # the braces matter: ``a && b && setsid ... &`` would background the whole list in a
    # subshell; ``$!`` would then be that subshell (not the session leader: the group kill
    # misses the sampler) and the subshell would hold the exec channel's stdout open
    return (f"mkdir -p {LOG_DIR} && rm -f {p['exit']} && "
            f"{{ setsid bash -c {shlex.quote(inner)} {MARKER} </dev/null >/dev/null 2>&1 & "
            f"echo $! > {p['pid']}; }}; sleep 1; {_alive(p['pid'])} && echo YETO_DAYTONA_STARTED $(cat {p['pid']}) "
            f"|| {{ echo YETO_DAYTONA_START_FAILED; tail -n 40 {p['log']} 2>/dev/null; exit 1; }}")


def stop_command(shard: str, *, grace_s: int = 60) -> str:
    """TERM the sampler's process group, KILL it after ``grace_s``; keep the log tail."""
    p = paths(shard)
    return (f"if {_alive(p['pid'])}; then p=$(cat {p['pid']}); kill -TERM -- -$p 2>/dev/null; "
            f"for i in $(seq {int(grace_s)}); do kill -0 -- -$p 2>/dev/null || break; sleep 1; done; "
            "kill -KILL -- -$p 2>/dev/null; fi; "
            f"rm -f {p['pid']}; tail -n 200 {p['log']} > {LOG_DIR}/{shard}.last.log 2>/dev/null; echo stopped {shard}")


def state_command(shard: str) -> str:
    """Prints ``<status> <exit>`` like ``docker inspect``: ``running -``, ``exited N``, ``absent -``."""
    p = paths(shard)
    return (f"if {_alive(p['pid'])}; then echo 'running -'; elif [ -f {p['exit']} ]; then "
            f"echo \"exited $(cat {p['exit']})\"; else echo 'absent -'; fi")


def probe_script(shard: str, tape_glob: str, *, metrics_port: int | None = None) -> str:
    """The pool's node probe (``YETO_NC_PROBE ...`` line) for a process sampler."""
    return elastic.probe_script(f"yeto-sampler-{shard}", tape_glob, metrics_port=metrics_port,
                                state_script=state_command(shard),
                                logs_script=f"tail -n 2000 {paths(shard)['log']} 2>/dev/null")


@dataclass
class SamplerStart:
    """One sampler start in a sandbox; the pieces the pool needs afterwards."""

    shard: str
    yeto_dir: str
    paths: dict[str, str] = field(default_factory=dict)
    setup_report: dict[str, str] = field(default_factory=dict)


def start_sampler(box: SandboxExec, *, spec: Mapping[str, Any], shard: str, run_prefix: str,
                  r2_env_file: str | Path, code: tuple[str, bytes], gpus: int = 8,
                  requirements: Requirements | None = None, protected: frozenset[str] = frozenset(),
                  agentic_env_file: str | Path | None = None, spec_dir: str | Path | None = None,
                  emit: Callable[[dict[str, Any]], None] | None = None,
                  environ: Mapping[str, str] | None = None) -> SamplerStart:
    """Preflight, upload code / spec / env files, agentic setup, start the sampler
    process detached (a running one of the shard is stopped first).

    ``code``: ``(sha, tar)`` of :func:`code_tar`. ``requirements``: default
    :func:`requirements_of` the spec. Refusals match the NC route's."""
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
    if setup is not None and (problems := agentic_setup.local_problems(setup, environ=environ)):
        raise ValueError("agentic setup: " + "; ".join(problems))
    emit = emit or (lambda record: None)
    out = sh(box, f"mkdir -p {SAMPLER_DIR} {LOG_DIR} {CODE_DIR} {CACHE} {WORK}/stage && chmod 700 {SAMPLER_DIR}; "
                  + ca_bundle_script(), timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"{box.name}: preparing {WORK}: {out.stderr.strip()[-300:]}")
    put(box, Path(r2_env_file).expanduser().read_bytes(), STORE_ENV)  # never echoed
    preflight(box, requirements or requirements_of(spec, gpus=gpus), emit=emit)
    yeto_dir = upload_code(box, *code)
    p = paths(shard)
    spec = {"start_on_latest": True, **spec}
    put(box, json.dumps(agentic_setup.strip(spec), indent=1), p["spec"], mode="644")
    env = process_env(prefix, yeto_dir, gpus=gpus)
    env_files = [STORE_ENV]
    report: dict[str, str] = {}
    if setup is not None:
        node = ProcessNode(box, yeto_dir=yeto_dir)
        report = ProcessProvisioner(setup, node, emit=emit, store_env_file=STORE_ENV,
                                    **({"environ": environ} if environ is not None else {})).run()
        setup_env = f"{SAMPLER_DIR}/{shard}.agentic-setup.env"
        put(box, "".join(f"{k}={v}\n" for k, v in sorted(
            setup.container_env(python_bin=str(Path(PYTHON).parent)).items())), setup_env)
        env_files.append(setup_env)
        env["PYTHONPATH"] = setup.python_path(base=yeto_dir)
    if agentic_env_file:
        env_files.append(f"{SAMPLER_DIR}/{shard}.agentic.env")  # after the setup's: the operator's file wins
        put(box, Path(agentic_env_file).expanduser().read_bytes(), env_files[-1])
    put(box, start_script(env, spec_path=p["spec"], env_files=env_files), p["script"], mode="700")
    sh(box, stop_command(shard, grace_s=30), timeout=150)
    out = sh(box, launch_command(shard), timeout=120)
    if out.returncode != 0 or "YETO_DAYTONA_STARTED" not in out.stdout:
        raise RuntimeError(f"{box.name}: sampler {shard} did not start: {out.stdout.strip()[-600:]}")
    emit({"event": "daytona_sampler_started", "sandbox": box.name, "shard": shard, "yeto": yeto_dir})
    return SamplerStart(shard, yeto_dir, p, report)


def stop_sampler(box: SandboxExec, shard: str, *, grace_s: int = 60) -> ExecResult:
    return sh(box, stop_command(shard, grace_s=grace_s), timeout=grace_s + 120)


def sampler_status(box: SandboxExec, shard: str) -> str:
    return sh(box, state_command(shard), timeout=120).stdout.strip()


# -- model pull ---------------------------------------------------------------------

PULL_PID = f"{SAMPLER_DIR}/pull.pid"
PULL_LOG = f"{LOG_DIR}/pull.log"


def pull_command(src_key: str, dest: str = MODEL_DIR, *, max_mb_s: int = 2500, include: str | None = None,
                 jobs: int = 16) -> str:
    """Detached capped parallel pull of the base checkpoint into ``dest`` (the NC body,
    run by the sandbox's awscli instead of an aws-cli container)."""
    if not dest.startswith(f"{MODELS}/"):
        raise ValueError(f"dest must be under {MODELS}")
    body = pull_body(src_key, dest, max_mb_s=max_mb_s, include=include, jobs=jobs)
    inner = (f"set -a; . {STORE_ENV}; set +a; [ -f {CA_BUNDLE} ] && export AWS_CA_BUNDLE={CA_BUNDLE}; "
             f"{body}; echo \"YETO_NC_PULL exit $?\"")
    return (f"mkdir -p {LOG_DIR} {shlex.quote(dest)} && {{ setsid bash -c {shlex.quote(inner)} "
            f">> {PULL_LOG} 2>&1 </dev/null & echo $! > {PULL_PID}; echo YETO_DAYTONA_PULL_STARTED; }}")


def pull_status_command() -> str:
    return f"tail -n 3 {PULL_LOG} 2>/dev/null; du -sh {MODELS}/* 2>/dev/null"


# -- the agentic setup without docker --------------------------------------------------

class ProcessNode:
    """:class:`agentic_setup.SshNode`'s interface over a :class:`SandboxExec`: the
    sandbox paths are the container paths (``nvme`` is empty), and "in the image"
    means in the sandbox itself."""

    def __init__(self, box: SandboxExec, *, yeto_dir: str, python: str = PYTHON, prelude: str = SSH_HOME_LINK):
        self.box, self.alias, self.image = box, box.name, "sandbox"
        self.nvme, self.yeto_dir, self.python, self.user, self.prelude = "", yeto_dir, python, "root", prelude

    def sh(self, command: str, *, timeout: float = 600, input: str | None = None):
        if input is not None:  # a command reading stdin: the text goes in as a file first
            tmp = f"/tmp/yeto-stdin-{abs(hash(command)) % 10**12}"
            put(self.box, input, tmp)
            command = f"{{ {command} ; }} < {tmp}; rc=$?; rm -f {tmp}; exit $rc"
        out = sh(self.box, command, timeout=timeout)
        return subprocess.CompletedProcess(["sandbox", command], out.returncode, out.stdout, out.stderr)

    def host(self, container_path: str) -> str:
        return container_path

    put = agentic_setup.SshNode.put

    def in_image(self, script: str, *, env: Mapping[str, str] | None = None, timeout: float = 900):
        exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in sorted((env or {}).items()))
        return self.sh(f"( {exports}{self.prelude}{script} )", timeout=timeout)

    def stream_tar(self, src: str, names: Sequence[str], remote_dir: str, *, timeout: float = 1800) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for name in names or ["."]:
                tar.add(str(Path(src) / name), arcname=name)
        tmp = f"{remote_dir.rstrip('/')}.upload.tar"
        self.box.upload(buf.getvalue(), tmp)
        out = self.sh(f"mkdir -p {shlex.quote(remote_dir)} && tar -C {shlex.quote(remote_dir)} -xf {tmp} "
                      f"--no-same-owner && rm -f {tmp}", timeout=timeout)
        if out.returncode != 0:
            raise agentic_setup.SetupError("tasks", f"copying {src} to the sandbox failed: "
                                                    f"{(out.stderr or '').strip()[-300:]}")


TUNNEL_DIR = "/tunnel"  # the paths _tunnel_config names
TUNNEL_PID = f"{TUNNEL_DIR}/agent.pid"


class ProcessProvisioner(agentic_setup.Provisioner):
    """The agentic setup steps with the docker pieces replaced: the store task sync
    runs the sandbox's awscli, the reverse tunnel is a detached
    ``agentic_node_agent tunnel`` process. The tunnel key sits in ``/tunnel`` (0700);
    unlike the NC helper container, the sampler process runs as the same user and
    could read it."""

    @property
    def tunnel_dir(self) -> str:
        return TUNNEL_DIR

    def _store_sync_root(self, dest: str) -> str:
        return shlex.quote(dest)

    def _store_sync_command(self, dest: str, body: str) -> str:
        return (f"( set -a; . {shlex.quote(str(self.store_env_file))}; set +a; "
                f"[ -f {CA_BUNDLE} ] && export AWS_CA_BUNDLE={CA_BUNDLE}; mkdir -p {shlex.quote(dest)}; "
                f"sh -c {shlex.quote(body)} )")

    def _tunnel_running(self) -> bool:
        want = self.setup.fingerprint()
        out = self.node.sh(f"{_alive_any(TUNNEL_PID)} && cat {TUNNEL_DIR}/fingerprint 2>/dev/null", timeout=60)
        return out.returncode == 0 and (out.stdout or "").strip() == want

    def _tunnel_start(self) -> Any:
        t = self.setup.tunnel
        assert t is not None
        events = self.node.host(self.setup.tunnel_events)
        argv = [self.node.python, "-m", "yeto.launch.agentic_node_agent", "tunnel", "--events", events,
                "--settle-s", f"{t.settle_s:g}", "--retry-s", f"{t.retry_s:g}", "--max-backoff-s",
                f"{t.max_backoff_s:g}", "--", *agentic_setup.tunnel_ssh_argv(self.setup)]
        inner = f"export PYTHONPATH={shlex.quote(self.node.yeto_dir)}; exec {shlex.join(argv)}"
        return self.node.sh(
            f"if {_alive_any(TUNNEL_PID)}; then kill -TERM -- -$(cat {TUNNEL_PID}) 2>/dev/null; sleep 1; fi; "
            f"echo {shlex.quote(self.setup.fingerprint())} > {TUNNEL_DIR}/fingerprint; "
            f"setsid bash -c {shlex.quote(inner)} >> {TUNNEL_DIR}/agent.log 2>&1 </dev/null & "
            f"echo $! > {TUNNEL_PID}", timeout=120)


def _alive_any(pidfile: str) -> str:
    return f"{{ p=$(cat {pidfile} 2>/dev/null); [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null; }}"


# -- the pool's sampler route ------------------------------------------------------------

class DaytonaSamplerRoute:
    """The sampler route a pool uses instead of the docker one (``NcSshOps`` takes it
    from ``provider.sampler_route`` or pool config ``"sampler_route": "process"``).
    The sandbox is reached by the provider's ssh alias (``node_pool.host_alias``);
    ``exec_for`` builds the exec handle of an alias (default :class:`SshExec`)."""

    def __init__(self, *, gpus: int = 8, plan: DiskPlan | None = None, code_ref: str = "HEAD",
                 requirements: Mapping[str, Any] | None = None,
                 exec_for: Callable[[str, Callable[..., Any]], SandboxExec] | None = None):
        self.gpus, self.plan, self.code_ref = int(gpus), plan, code_ref
        self.requirements = dict(requirements or {})
        self.exec_for = exec_for or (lambda alias, run: SshExec(alias, [alias], run=run))
        self._code: tuple[str, bytes] | None = None

    def code(self, run: Callable[..., Any] = subprocess.run) -> tuple[str, bytes]:
        if self._code is None:
            self._code = code_tar(self.code_ref, run=run)
        return self._code

    def start(self, alias: str, *, spec: Mapping[str, Any], shard: str, run_prefix: str, r2_env_file: str,
              protected: frozenset[str] = frozenset(), agentic_env_file: str | None = None,
              spec_dir: str | Path | None = None, emit: Callable[[dict[str, Any]], None] | None = None,
              run: Callable[..., Any] = subprocess.run) -> str:
        """Idempotent: a running sampler of the shard is stopped and started again."""
        req = requirements_of(spec, gpus=self.gpus, plan=self.plan, **self.requirements)
        start_sampler(self.exec_for(alias, run), spec=spec, shard=shard, run_prefix=run_prefix,
                      r2_env_file=r2_env_file, code=self.code(run), gpus=self.gpus, requirements=req,
                      protected=protected, agentic_env_file=agentic_env_file, spec_dir=spec_dir, emit=emit)
        return f"yeto-sampler-{shard}"

    def probe_script(self, shard: str, tape_glob: str, metrics_port: int | None = None) -> str:
        return probe_script(shard, tape_glob, metrics_port=metrics_port)

    def stop_command(self, shard: str, *, grace_s: int = 30) -> str:
        return NO_SUDO_SHIM + stop_command(shard, grace_s=grace_s)

    def stop(self, alias: str, shard: str, *, grace_s: int = 30, run: Callable[..., Any] = subprocess.run) -> ExecResult:
        """Best effort (a lost sandbox does not answer); the log tail is kept."""
        return stop_sampler(self.exec_for(alias, run), shard, grace_s=grace_s)

    def pull(self, alias: str, *, r2_env_file: str, src_key: str, dest: str = MODEL_DIR, include: str | None = None,
             max_mb_s: int = 2500, run: Callable[..., Any] = subprocess.run) -> None:
        box = self.exec_for(alias, run)
        sh(box, f"mkdir -p {SAMPLER_DIR} && chmod 700 {SAMPLER_DIR}; " + ca_bundle_script(), timeout=120)
        put(box, Path(r2_env_file).expanduser().read_bytes(), STORE_ENV)
        out = sh(box, pull_command(src_key, dest, max_mb_s=max_mb_s, include=include), timeout=300)
        if out.returncode != 0 or "YETO_DAYTONA_PULL_STARTED" not in out.stdout:
            raise RuntimeError(f"{alias}: model pull did not start: {out.stderr.strip()[-300:]}")

    def pull_status(self, alias: str, *, run: Callable[..., Any] = subprocess.run) -> str:
        return sh(self.exec_for(alias, run), pull_status_command(), timeout=120).stdout.strip()
