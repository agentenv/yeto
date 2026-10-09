"""Run a Yeto learner island on Modal.

Modal is not a SkyPilot cloud: there are no VMs and no SSH, only
functions that run in GPU containers. This module makes one island — the
same thing a `sky.Task` is for the other clouds — into a Modal function
call, and gives the launcher the four operations it supervises islands
with (spawn, status, relaunch, cancel) plus log tailing.

The island runs the SAME `run` script the SkyPilot task would have run:
the container exports the `SKYPILOT_*` variables that script reads
(node ips / rank / count) from Modal's cluster info, so torchrun and the
Ray bootstrap for RL islands are byte-for-byte the launcher's. Only
`setup` differs: dependencies are baked into the image (built once,
cached by Modal) instead of installed at boot on every node.

Two ways in:

* `yeto launch --gpu ...,modal:8xh100,...` — the launcher routes the entry
  here (see launcher.run) and supervises it next to the sky islands.
* `python -m yeto.modal_runner --learner-id N --syncer-addr H:P ...` — the
  manual-join form for a fleet started with `--external-learners`.

All `modal` imports are local to the functions that need them so this
module imports (and is unit-tested) without the SDK or credentials.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODAL_TOKEN_PATH = "~/.modal.toml"
# Where the repo lands in the container; the sky task scripts `cd
# ~/sky_workdir`, and HOME is /root in Modal containers.
CONTAINER_WORKDIR = "/root/sky_workdir"
DEFAULT_TIMEOUT_S = 60 * 60 * 24
DEFAULT_RETRIES = 10
# Kept out of the island image. Patterns are dockerignore-style (a bare
# name only matches at the root), so caches use **/. `.claude` holds agent
# worktrees whose files change mid-upload and abort the Modal build.
MODAL_WORKDIR_IGNORE = [
    ".git", ".venv", ".claude", "syncer/target", "RLinf-main",
    "**/__pycache__", "**/*.pyc",
]

# sky accelerator name -> Modal GPU string, and the whole-node GPU count
# Modal requires for every container of a multi-container (clustered)
# function since 2026-05-31.
MODAL_GPUS: dict[str, str] = {
    "H100": "H100",
    "H200": "H200",
    "B200": "B200",
    "A100-80GB": "A100-80GB",
    "L40S": "L40S",
    "L4": "L4",
    "A10G": "A10G",
    "T4": "T4",
}
# Modal silently upgrades a plain "H100" request to H200 when it can; the
# "!" suffix pins the exact type (needed for bitwise comparisons, where the
# model swap changes bf16 numerics).  Only H100 is upgraded, so only it
# takes the suffix.
MODAL_EXACT_PINNABLE = frozenset({"H100"})
# nvidia-smi --query-gpu=name must match these for an exact-GPU island.
MODAL_GPU_NAME_PATTERNS: dict[str, str] = {
    "H100": r"\bH100\b",
    "H200": r"\bH200\b",
    "B200": r"\bB200\b",
    "A100-80GB": r"\bA100\b.*\b80GB\b",
    "L40S": r"\bL40S\b",
    "L4": r"\bL4\b",
    "A10G": r"\bA10G\b",
    "T4": r"\bT4\b",
}
MODAL_FULL_NODE: dict[str, int] = {"H100": 8, "H200": 8, "B200": 8, "A100-80GB": 8}
# CPU / memory the runner reserves per GPU (Modal bills max(request,
# usage)); the shape planner prices the same reservation.
# Event tape on a Modal Volume: a Modal container's filesystem disappears
# when it exits (there is no node to `scp` from afterwards), so the island
# copies its ~/yeto-output tape files into a mounted Volume while it runs
# and commits them; the launcher pulls them back into the local run dir.
TAPE_MOUNT = "/yeto-tape"
TAPE_SOURCE_DIR = "/root/yeto-output"
TAPE_SYNC_INTERVAL_S = 30.0
TAPE_SUFFIXES = (".jsonl", ".json", ".csv", ".log", ".txt", ".pt")  # .pt: verl raw log-prob dumps
TAPE_MAX_FILE_BYTES = 256 * 1024 * 1024
MODAL_CPU_CORES_PER_GPU = 4
MODAL_MEMORY_GIB_PER_GPU = 32


def modal_available() -> bool:
    """True when a Modal token is configured (`modal token new`, or env)."""
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    return os.path.isfile(os.path.expanduser(MODAL_TOKEN_PATH))


def modal_credential_hint() -> str:
    return f"{MODAL_TOKEN_PATH} (run `modal token new`) or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET"


def modal_app_name(cluster_prefix: str) -> str:
    """Deterministic per-run app name: `yeto down <prefix>` stops it."""
    return f"yeto-{cluster_prefix}"


def modal_island_name(cluster_prefix: str, learner_id: int) -> str:
    """The launcher-side 'cluster name' of a Modal island (registry, logs,
    FleetController records); recognisable by its suffix."""
    return f"{cluster_prefix}-l{learner_id}-modal"


def is_modal_island(name: str) -> bool:
    return name.endswith("-modal")


def validate_modal_shape(gpu: str, gpus_per_node: int, num_nodes: int) -> None:
    """The rules Modal enforces at scheduling time, checked before any
    resource is touched: known GPU, 1..8 per container, and whole nodes
    only when there is more than one container."""
    if gpu not in MODAL_GPUS:
        raise ValueError(f"Modal has no {gpu}; known: {', '.join(sorted(MODAL_GPUS))}")
    if not 1 <= gpus_per_node <= 8:
        raise ValueError(f"Modal containers take 1-8 GPUs, not {gpus_per_node}")
    if num_nodes > 1:
        full = MODAL_FULL_NODE.get(gpu)
        if full is None:
            raise ValueError(f"Modal multi-container islands need a whole-node GPU (one of {', '.join(sorted(MODAL_FULL_NODE))}), not {gpu}")
        if gpus_per_node != full:
            raise ValueError(
                f"Modal multi-container islands must use whole nodes: {gpu}:{full} per container, "
                f"not {gpu}:{gpus_per_node} (Modal rule since 2026-05-31)"
            )


def is_public_address(host: str) -> bool:
    """Whether a Modal container (on Modal's network) can reach `host`.
    Private, loopback, link-local and unresolvable hosts are not public."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
        except (OSError, ValueError):
            return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified)


def resolve_syncer_for_modal(syncer_addr: str, public_override: str | None) -> str:
    """The syncer address a Modal island dials; raises when it is not
    reachable from Modal and no `--syncer-public-addr` was given."""
    if public_override:
        return public_override
    host = syncer_addr.rsplit(":", 1)[0].strip("[]")
    if is_public_address(host):
        return syncer_addr
    raise ValueError(
        f"syncer at {syncer_addr} is not reachable from Modal (private or unresolvable "
        "address). Use the default head controller mode (the syncer then sits on a "
        "public head VM), or pass --syncer-public-addr HOST:PORT for a tunnel to it."
    )


@dataclass(frozen=True)
class ModalIslandConfig:
    """Everything one Modal island needs; JSON-serialisable so the launcher,
    the manual CLI and the container all see the same thing."""

    app_name: str
    learner_id: int
    training_mode: str  # "sft" | "rl"
    gpu: str  # sky accelerator name
    gpus_per_node: int
    num_nodes: int
    run_script: str  # the sky task's `run` (SKYPILOT_* vars provided here)
    envs: dict[str, str] = field(default_factory=dict)
    region: str | None = None  # Modal region hint; None = unpinned, no surcharge
    rdma: bool = True
    image_ref: str | None = None  # RL: registry image, MUST be repo@sha256:<64 hex>
    # RL: the sky task's `setup`, run in the container before `run_script`
    # (as sky runs it on the node). It cannot be baked at build time: it
    # reads the repo (Miles bundle), which is mounted only at start-up.
    setup_script: str | None = None
    pip_requirements: tuple[str, ...] = ()  # SFT image: requirements to install
    volume_name: str | None = None  # RL spot checkpoints
    volume_mount: str | None = None
    # Request exactly `gpu` ("H100!" -- no H200 upgrade) and fail the
    # container at start-up unless nvidia-smi reports that type.
    gpu_exact: bool = False
    # Pull image_ref with a private-registry login: ``registry_creds`` (the
    # SKYPILOT_DOCKER_* triple launcher.registry_login_for resolved) when
    # given, else -- with ``registry_login`` -- the launching process's
    # environment.  Default: anonymous pull (public image).
    registry_login: bool = False
    registry_creds: dict[str, str] | None = None
    timeout_s: int = DEFAULT_TIMEOUT_S
    retries: int = DEFAULT_RETRIES
    workdir: str = str(REPO_ROOT)
    python_version: str = "3.12"
    # Stock Codex run bundle (ports engine, signed Codex agent): a local dir
    # holding the attested binary / package manifest / app-server schema,
    # mounted read-only at ``codex_mount`` (= ``/opt/yeto/codex``, the
    # CODEX_CONTAINER_BINARY_PATH parent) the same way sky file_mounts does.
    codex_dir: str | None = None
    codex_mount: str | None = None
    # Other sky file_mounts (container path -> local file or dir), mounted
    # read-only at start-up like the workdir.
    extra_mounts: dict[str, str] = field(default_factory=dict)
    # Event tape Volume (None = off, the pre-existing behaviour): the
    # container mirrors TAPE_SOURCE_DIR into <volume>/<tape_subdir>/rank<r>/.
    # rl-resume-from-checkpoint: the --rl-checkpoint-store modal-volume://NAME Volume
    # (v1), mounted read-write; the learner commits it after every cut (LATEST last).
    checkpoint_store_volume_name: str | None = None
    checkpoint_store_mount: str | None = None
    tape_volume_name: str | None = None
    tape_subdir: str | None = None
    # Model Volume (None = off): mounted read-only at model_volume_mount
    # (the Nebius model-store path, /mnt/yeto-models); holds HF snapshots
    # under hf/<name>/<rev[:8]>/ and torch_dist checkpoints (Flash-Next).
    model_volume_name: str | None = None
    model_volume_mount: str | None = None
    # Host resource overrides (None = per-GPU defaults below).
    cpu_override: int | None = None
    memory_gib_override: int | None = None

    @property
    def function_name(self) -> str:
        return f"island-{self.learner_id}"

    @property
    def gpu_request(self) -> str:
        pin = "!" if self.gpu_exact and self.gpu in MODAL_EXACT_PINNABLE else ""
        return f"{MODAL_GPUS[self.gpu]}{pin}:{self.gpus_per_node}"

    @property
    def cpu_request(self) -> int:
        if self.cpu_override is not None:
            return int(self.cpu_override)
        return MODAL_CPU_CORES_PER_GPU * self.gpus_per_node

    @property
    def memory_request_mib(self) -> int:
        if self.memory_gib_override is not None:
            return int(self.memory_gib_override) * 1024
        return MODAL_MEMORY_GIB_PER_GPU * self.gpus_per_node * 1024

    def validate(self) -> None:
        validate_modal_shape(self.gpu, self.gpus_per_node, self.num_nodes)
        if self.training_mode == "rl":
            if not self.image_ref or not (re.fullmatch(r"[^\s@]+@sha256:[0-9a-fA-F]{64}", self.image_ref)
                                          or ENGINE_BUILT_IMAGE_RE.fullmatch(self.image_ref)):
                raise ValueError(
                    "RL islands on Modal must pin the Miles image by digest "
                    "(<repository>@sha256:<64 hex>), the same digest --rl-image gives sky"
                )
        if (self.volume_name is None) != (self.volume_mount is None):
            raise ValueError("volume_name and volume_mount go together")
        if (self.tape_volume_name is None) != (self.tape_subdir is None):
            raise ValueError("tape_volume_name and tape_subdir go together")
        if self.tape_subdir is not None and (
            self.tape_subdir.startswith("/") or ".." in self.tape_subdir.split("/")
        ):
            raise ValueError(f"tape_subdir {self.tape_subdir!r} must be relative, without '..'")
        if self.tape_volume_name and self.volume_name == self.tape_volume_name:
            raise ValueError("the tape volume must differ from the checkpoint volume")
        if (self.checkpoint_store_volume_name is None) != (self.checkpoint_store_mount is None):
            raise ValueError("checkpoint_store_volume_name and checkpoint_store_mount go together")
        if self.checkpoint_store_volume_name and self.checkpoint_store_volume_name in (
                self.volume_name, self.tape_volume_name, self.model_volume_name):
            raise ValueError("the checkpoint store volume must differ from the other volumes")
        if (self.model_volume_name is None) != (self.model_volume_mount is None):
            raise ValueError("model_volume_name and model_volume_mount go together")
        if self.model_volume_name and self.model_volume_name in (self.volume_name, self.tape_volume_name):
            raise ValueError("the model volume must differ from the checkpoint/tape volumes")
        if (self.codex_dir is None) != (self.codex_mount is None):
            raise ValueError("codex_dir and codex_mount go together")
        if self.codex_dir is not None and not os.path.isdir(self.codex_dir):
            raise ValueError(f"codex_dir {self.codex_dir} is not a directory")
        for target, source in self.extra_mounts.items():
            if not target.startswith("/"):
                raise ValueError(f"extra_mounts target {target} must be absolute")
            if not os.path.exists(os.path.expanduser(source)):
                raise ValueError(f"extra_mounts source {source} does not exist")

    def to_json(self) -> str:
        # The JSON travels into the container (fn.spawn): the registry login
        # stays with the launching process, it only feeds the image build.
        data = asdict(self)
        data.pop("registry_creds", None)
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ModalIslandConfig":
        data = json.loads(text)
        data["pip_requirements"] = tuple(data.get("pip_requirements") or ())
        data["extra_mounts"] = dict(data.get("extra_mounts") or {})
        return cls(**data)


# ``<backend>-build:<40-hex commit>``: an engine image Modal builds from the
# backend's own recipe (``yeto.rl.engine.backends`` role ``image``), pinned by the
# engine commit and the engine's lock file; there is no registry digest.
ENGINE_BUILT_IMAGE_RE = re.compile(r"(?P<backend>[a-z][a-z0-9]*)-build:(?P<commit>[0-9a-f]{40})")


def image_ref_from_rl_image(rl_image: str) -> str:
    """`--rl-image docker:<repo>@sha256:<hex>` -> the registry reference."""
    if ENGINE_BUILT_IMAGE_RE.fullmatch(rl_image or ""):
        return rl_image  # rl-verl-backend: built on Modal from a pinned engine commit
    ref = rl_image[len("docker:"):] if rl_image.startswith("docker:") else rl_image
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-fA-F]{64}", ref):
        raise ValueError(f"--rl-image must pin a digest (docker:<repo>@sha256:<64 hex>), got {rl_image!r}")
    return ref


# SkyPilot's private-registry login variables.  The launcher reads them from
# its environment (ports engine only) and hands them to SkyPilot as the
# resources' docker_login_config (used for `docker login` at provisioning)
# and to Modal as the ``from_registry`` pull secret -- not into task envs or
# the container environment.  Use a token scoped to read:packages only.
DOCKER_LOGIN_ENV_VARS = (
    "SKYPILOT_DOCKER_USERNAME",
    "SKYPILOT_DOCKER_PASSWORD",
    "SKYPILOT_DOCKER_SERVER",
)


def registry_host(image_ref: str) -> str:
    """Registry host of ``[docker:]<repo>[@digest]`` (Docker Hub default)."""
    ref = image_ref[len("docker:"):] if image_ref.startswith("docker:") else image_ref
    first = ref.split("/", 1)[0]
    if "/" in ref and ("." in first or ":" in first or first == "localhost"):
        return first
    return "docker.io"


def registry_credentials(image_ref: str | None, environ) -> dict[str, str] | None:
    """The SKYPILOT_DOCKER_* login from ``environ`` if it is for the
    registry ``image_ref`` lives on, else None (public image, or a login
    for some other registry).  A partial login is an error only when its
    server is this image's registry (as in SkyPilot)."""
    present = {k: environ[k] for k in DOCKER_LOGIN_ENV_VARS if environ.get(k)}
    if not present or not image_ref:
        return None
    server = re.sub(r"^https?://", "", present.get("SKYPILOT_DOCKER_SERVER", "")).rstrip("/")
    if server.split("/", 1)[0] != registry_host(image_ref):
        return None
    if len(present) != len(DOCKER_LOGIN_ENV_VARS):
        missing = sorted(set(DOCKER_LOGIN_ENV_VARS) - set(present))
        raise ValueError(f"registry login needs all of {DOCKER_LOGIN_ENV_VARS}; missing {missing}")
    return present


def skypilot_env(rank: int, container_ips: list[str], gpus_per_node: int) -> dict[str, str]:
    """The variables the sky task scripts read, emulated for a container."""
    return {
        "SKYPILOT_NODE_IPS": "\n".join(container_ips),
        "SKYPILOT_NUM_NODES": str(len(container_ips)),
        "SKYPILOT_NODE_RANK": str(rank),
        "SKYPILOT_NUM_GPUS_PER_NODE": str(gpus_per_node),
    }


def check_gpu_names(gpu: str, names: list[str], count: int) -> None:
    """Raise unless exactly ``count`` GPUs of type ``gpu`` are visible."""
    pattern = MODAL_GPU_NAME_PATTERNS[gpu]
    wrong = [n for n in names if not re.search(pattern, n)]
    if wrong or len(names) != count:
        raise RuntimeError(
            f"requested {count}x {gpu} (exact) but the container sees {names!r}"
        )


def visible_gpu_names() -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def cluster_rank_and_ips(info) -> tuple[int, list[str]]:
    """Rank and node addresses from Modal's cluster info.  IPv4 first:
    `container_ips` are IPv6 on Modal, and the sky scripts build
    `"$MASTER_ADDR:6379"` (Ray) / torchrun endpoints that need a bare IPv4."""
    ipv4 = list(getattr(info, "container_ipv4_ips", None) or [])
    return int(info.rank), ipv4 or list(info.container_ips)


class TapeSync:
    """Mirror tape files of ``src`` into ``dst`` and commit the Volume,
    every ``interval_s`` in a daemon thread and once more on ``stop()``.
    Copies whole files that changed (size or mtime); never deletes."""

    def __init__(self, src: str, dst: str, commit, interval_s: float = TAPE_SYNC_INTERVAL_S,
                 extra: dict[str, str] | None = None) -> None:
        self.src, self.dst, self.commit, self.interval_s = Path(src), Path(dst), commit, interval_s
        self.extra = dict(extra or {})  # file name -> text written once
        self._seen: dict[str, tuple[int, int]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.syncs = 0
        self.errors: list[str] = []
        # rl-resume-from-checkpoint 4.2: a re-run of the island in a new container
        # (preemption + resume) must not replace the previous container's tape files.
        # The k-th container to mirror into dst writes INCARNATION-<k>.txt and, for k > 0,
        # names its copies "<stem>.inc<k><suffix>" (k = 0 keeps the old names).
        self.incarnation: int | None = None

    def _claim_incarnation(self) -> int:
        if self.incarnation is None:
            self.dst.mkdir(parents=True, exist_ok=True)
            taken = sorted(self.dst.glob("INCARNATION-*.txt"))
            k = len(taken)
            if k == 0 and any(p.is_file() and p.name.endswith(TAPE_SUFFIXES) for p in self.dst.iterdir()):
                k = 1  # files from a container that predates the markers
            marker = self.dst / f"INCARNATION-{k}.txt"
            marker.write_text(f"{os.environ.get('MODAL_TASK_ID', 'unknown')}\n", encoding="utf-8")
            self.incarnation = k
        return self.incarnation

    def _target_name(self, name: str) -> str:
        k = self._claim_incarnation()
        if k == 0:
            return name
        for suffix in TAPE_SUFFIXES:
            if name.endswith(suffix):
                return f"{name[:-len(suffix)]}.inc{k}{suffix}"
        return f"{name}.inc{k}"

    def sync_once(self) -> int:
        self.dst.mkdir(parents=True, exist_ok=True)
        self._claim_incarnation()
        for name, text in list(self.extra.items()):
            (self.dst / self._target_name(name)).write_text(text, encoding="utf-8")
            del self.extra[name]
        copied = 0
        if self.src.is_dir():
            for f in sorted(self.src.iterdir()):
                if not f.is_file() or not f.name.endswith(TAPE_SUFFIXES):
                    continue
                st = f.stat()
                if st.st_size > TAPE_MAX_FILE_BYTES:
                    continue
                key = (st.st_size, st.st_mtime_ns)
                if self._seen.get(f.name) == key:
                    continue
                target = self._target_name(f.name)
                tmp = self.dst / f".{target}.tmp"
                tmp.write_bytes(f.read_bytes())
                os.replace(tmp, self.dst / target)
                self._seen[f.name] = key
                copied += 1
        self.commit()
        self.syncs += 1
        return copied

    def _safe_sync(self) -> None:
        try:
            self.sync_once()
        except Exception as exc:  # noqa: BLE001 - the tape must never kill the island
            self.errors.append(repr(exc))
            print(f"[modal-tape] sync failed: {exc!r}", flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._safe_sync()

    def start(self) -> "TapeSync":
        self._safe_sync()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="modal-tape")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 60)
        self._safe_sync()


def _volume_commit(volume_name: str):
    def commit() -> None:
        import modal

        modal.Volume.from_name(volume_name).commit()

    return commit


def container_command(run_script: str) -> list[str]:
    """How the island's run script is executed inside the container."""
    return ["bash", "-lc", f"cd {shlex.quote(CONTAINER_WORKDIR)} && {run_script}"]


def _host_mem_used_bytes() -> dict:
    """Container-visible host memory: /proc/meminfo (MemTotal - MemAvailable)
    plus cgroup memory.current/peak when readable (None otherwise)."""
    out: dict = {}
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = int(v.split()[0]) * 1024
        out["meminfo_total"] = info.get("MemTotal")
        out["meminfo_used"] = info.get("MemTotal", 0) - info.get("MemAvailable", 0)
    except Exception:  # noqa: BLE001 - advisory
        pass
    for key, path in (("cgroup_current", "/sys/fs/cgroup/memory.current"),
                      ("cgroup_peak", "/sys/fs/cgroup/memory.peak")):
        try:
            with open(path) as f:
                out[key] = int(f.read().strip())
        except Exception:  # noqa: BLE001
            out[key] = None
    return out


HOSTMEM_DEFAULT_MULTINODE_S = 30.0


def hostmem_interval_s(cfg) -> float | None:
    """Sampling interval: env YETO_MODAL_HOSTMEM_SAMPLE_S wins ("0" disables);
    otherwise every node of a multi-node island is sampled by default."""
    raw = (cfg.envs or {}).get("YETO_MODAL_HOSTMEM_SAMPLE_S")
    if raw is not None and str(raw).strip() != "":
        v = float(raw)
        return v if v > 0 else None
    return HOSTMEM_DEFAULT_MULTINODE_S if cfg.num_nodes > 1 else None


class HostMemSampler:
    """Append host memory and nvidia-smi memory.used / utilization.gpu every
    interval to a .jsonl in the tape dir (so the tape Volume carries it home)
    and print the peaks at exit. Opt-in via env YETO_MODAL_HOSTMEM_SAMPLE_S;
    on by default for multi-node islands (``hostmem_interval_s``), because the
    driver's ``rl_resource_sample`` only covers the node the driver runs on
    and a disaggregated island's rollout node would otherwise go unsampled.
    Starts at container start, so the startup phase (weight load, Ray, engine
    init) is covered too."""

    def __init__(self, interval_s: float, path: str, node_rank: int | None = None):
        self.interval_s, self.path = max(1.0, interval_s), path
        self.node_rank = node_rank
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_host = 0
        self.peak_gpu: list[int] = []

    def sample(self) -> dict:
        rec = {"event": "modal_host_sample", "time_unix": time.time(), **_host_mem_used_bytes()}
        if self.node_rank is not None:
            rec["node_rank"] = self.node_rank
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20)
            rows = [[x.strip() for x in line.split(",")] for line in res.stdout.splitlines() if line.strip()]
            rec["gpu_mem_used_mib"] = [int(r[0]) for r in rows]
            rec["gpu_util_pct"] = [int(r[1]) if len(r) > 1 and r[1].isdigit() else None for r in rows]
        except Exception:  # noqa: BLE001
            rec["gpu_mem_used_mib"] = None
            rec["gpu_util_pct"] = None
        used = max(v or 0 for v in (rec.get("meminfo_used"), rec.get("cgroup_current")))
        self.peak_host = max(self.peak_host, used)
        for i, v in enumerate(rec["gpu_mem_used_mib"] or []):
            if i >= len(self.peak_gpu):
                self.peak_gpu.append(0)
            self.peak_gpu[i] = max(self.peak_gpu[i], v)
        return rec

    def _loop(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        while True:
            try:
                rec = self.sample()
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:  # noqa: BLE001
                pass
            if self._stop.wait(self.interval_s):
                return

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="modal-hostmem")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        print(f"[modal-hostmem] peak host used {self.peak_host / 2**30:.1f} GiB; "
              f"peak gpu used MiB {self.peak_gpu}", flush=True)


def island_main(cfg_json: str) -> int:
    """Body of the Modal function: runs in the container. Returns the run
    script's exit code (nonzero raises so Modal records the failure)."""
    cfg = ModalIslandConfig.from_json(cfg_json)
    if cfg.num_nodes > 1:
        import modal.experimental

        info = modal.experimental.get_cluster_info()
        rank, ips = cluster_rank_and_ips(info)
        all_ips = {"container_ips": list(getattr(info, "container_ips", []) or []),
                   "container_ipv4_ips": list(getattr(info, "container_ipv4_ips", []) or [])}
    else:
        rank, ips, all_ips = 0, ["127.0.0.1"], {}
    env = {**os.environ, **cfg.envs, **skypilot_env(rank, ips, cfg.gpus_per_node), "HOME": "/root"}
    # rl-resume-from-checkpoint: the learner's own Python cannot import the Modal client
    # (G2 A: "No module named 'grpclib'"); it commits the store Volume through this
    # runner's interpreter and import path instead.
    env["YETO_MODAL_PYTHON"] = sys.executable
    env["YETO_MODAL_SYSPATH"] = os.pathsep.join(p for p in sys.path if p)
    print(f"[modal-island {cfg.learner_id}] rank {rank}/{len(ips)} starting (node ips {ips})", flush=True)
    # The launcher watches this line: a second, different id on the same call means Modal
    # moved the island to a new container (preemption / reschedule) and re-ran the script.
    print(f"[modal-island {cfg.learner_id}] rank {rank} container {os.environ.get('MODAL_TASK_ID', 'unknown')}",
          flush=True)
    tape = None
    if cfg.tape_volume_name and cfg.tape_subdir:
        os.makedirs(TAPE_SOURCE_DIR, exist_ok=True)
        tape = TapeSync(
            TAPE_SOURCE_DIR, f"{TAPE_MOUNT}/{cfg.tape_subdir}/rank{rank}",
            _volume_commit(cfg.tape_volume_name),
        )
    hostmem = None
    interval = hostmem_interval_s(cfg)
    if interval is not None:
        hostmem = HostMemSampler(interval, f"{TAPE_SOURCE_DIR}/modal-hostmem-rank{rank}.jsonl",
                                 node_rank=rank)
        hostmem.start()
    try:
        return _island_body(cfg, rank, ips, all_ips, env, tape)
    finally:
        if hostmem is not None:
            hostmem.stop()
        if tape is not None:
            tape.stop()
            print(f"[modal-tape] rank {rank}: {tape.syncs} commit(s) to "
                  f"{cfg.tape_volume_name}:{cfg.tape_subdir}/rank{rank}, errors {len(tape.errors)}",
                  flush=True)


def _island_body(cfg: ModalIslandConfig, rank: int, ips: list[str], all_ips: dict,
                 env: dict, tape: "TapeSync | None") -> int:
    try:
        names = visible_gpu_names()
    except (OSError, subprocess.SubprocessError) as exc:
        if cfg.gpu_exact:
            raise RuntimeError(f"island {cfg.learner_id}: cannot read GPU names: {exc}") from exc
        names = []
    print(f"[modal-island {cfg.learner_id}] requested {cfg.gpu_request}, got {names}", flush=True)
    if tape is not None:
        tape.extra["modal-node.json"] = json.dumps({
            "rank": rank, "node_ips": ips, **all_ips, "hostname": socket.gethostname(),
            "gpu_request": cfg.gpu_request, "gpu_names": names,
            "start_unix": time.time(),
        }, sort_keys=True) + "\n"
        tape.start()
    if cfg.gpu_exact:
        check_gpu_names(cfg.gpu, names, cfg.gpus_per_node)
    if cfg.setup_script:
        code = subprocess.call(container_command(cfg.setup_script), env=env)
        if code != 0:
            raise RuntimeError(f"island {cfg.learner_id} rank {rank} setup exited with {code}")
    code = subprocess.call(container_command(cfg.run_script), env=env)
    if code != 0:
        raise RuntimeError(f"island {cfg.learner_id} rank {rank} exited with {code}")
    return code


class ModalOps:
    """The SDK surface the launcher uses, as one small class (tests
    substitute a fake). One instance per run/app."""

    def __init__(self, app_name: str) -> None:
        self.app_name = app_name
        self._app = None
        self._functions: dict[str, object] = {}

    # -- building --------------------------------------------------------------

    def _modal(self):
        import modal

        return modal

    def build_image(self, cfg: ModalIslandConfig):
        modal = self._modal()
        built = ENGINE_BUILT_IMAGE_RE.fullmatch(cfg.image_ref or "") if cfg.training_mode == "rl" else None
        if built:
            from yeto.rl.engine import backends

            image = backends.module("image", built["backend"]).modal_image(modal, built["commit"])
        elif cfg.training_mode == "rl":
            creds = cfg.registry_creds or (
                registry_credentials(cfg.image_ref, os.environ) if cfg.registry_login else None
            )
            if creds:  # private registry (e.g. MILES_NEXT_IMAGE on ghcr.io)
                image = modal.Image.from_registry(
                    cfg.image_ref,
                    secret=modal.Secret.from_dict(
                        {
                            "REGISTRY_USERNAME": creds["SKYPILOT_DOCKER_USERNAME"],
                            "REGISTRY_PASSWORD": creds["SKYPILOT_DOCKER_PASSWORD"],
                        }
                    ),
                )
            else:
                image = modal.Image.from_registry(cfg.image_ref)
        else:
            image = modal.Image.debian_slim(python_version=cfg.python_version)
            if cfg.pip_requirements:
                image = image.pip_install(*cfg.pip_requirements)
        image = image.env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
        # add_local_* with copy=False must be the last steps: Modal rejects
        # any build step after one (the files are mounted at start-up).
        image = image.add_local_dir(
            cfg.workdir,
            CONTAINER_WORKDIR,
            copy=False,
            ignore=MODAL_WORKDIR_IGNORE,
        )
        if cfg.codex_dir and cfg.codex_mount:
            # Codex run bundle (sky: file_mounts[codex_mount] = codex_dir).
            image = image.add_local_dir(cfg.codex_dir, cfg.codex_mount, copy=False)
        for target, source in sorted(cfg.extra_mounts.items()):
            source = os.path.expanduser(source)
            if os.path.isdir(source):
                image = image.add_local_dir(source, target, copy=False)
            else:
                image = image.add_local_file(source, target, copy=False)
        return image

    def define(self, cfg: ModalIslandConfig):
        """Register the island's function on this run's app (idempotent
        per function name); `deploy` publishes them."""
        modal = self._modal()
        cfg.validate()
        if self._app is None:
            self._app = modal.App(self.app_name)
        if cfg.function_name in self._functions:
            return self._functions[cfg.function_name]
        fn = island_main
        if cfg.num_nodes > 1:
            import modal.experimental

            fn = modal.experimental.clustered(size=cfg.num_nodes, rdma=cfg.rdma)(fn)
        kwargs: dict = dict(
            image=self.build_image(cfg),
            gpu=cfg.gpu_request,
            cpu=cfg.cpu_request,
            memory=cfg.memory_request_mib,
            timeout=cfg.timeout_s,
            retries=modal.Retries(max_retries=cfg.retries, initial_delay=0.0),
            secrets=[modal.Secret.from_dict(dict(cfg.envs))],
            name=cfg.function_name,
        )
        if cfg.region:
            kwargs["region"] = cfg.region
        volumes = {}
        if cfg.volume_name and cfg.volume_mount:
            volumes[cfg.volume_mount] = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)
        if cfg.checkpoint_store_volume_name and cfg.checkpoint_store_mount:
            # create_if_missing makes a v1 Volume (the default version); a failure to
            # look it up/create it fails the launch (never a silent container-local store)
            volumes[cfg.checkpoint_store_mount] = modal.Volume.from_name(
                cfg.checkpoint_store_volume_name, create_if_missing=True)
        if cfg.tape_volume_name:
            volumes[TAPE_MOUNT] = modal.Volume.from_name(cfg.tape_volume_name, create_if_missing=True)
        if cfg.model_volume_name and cfg.model_volume_mount:
            mvol = modal.Volume.from_name(cfg.model_volume_name)
            volumes[cfg.model_volume_mount] = mvol.read_only() if hasattr(mvol, "read_only") else mvol
        if volumes:
            kwargs["volumes"] = volumes
        self._functions[cfg.function_name] = self._app.function(**kwargs)(fn)
        return self._functions[cfg.function_name]

    def deploy(self) -> None:
        """Publish the app (builds images); blocking."""
        modal = self._modal()
        if self._app is None:
            raise RuntimeError("define() at least one island before deploy()")
        with modal.enable_output():
            self._app.deploy(name=self.app_name)

    # -- running -----------------------------------------------------------------

    def spawn(self, cfg: ModalIslandConfig) -> str:
        """Start the island; returns the function-call id (the island's
        'job id'). Works from any process once the app is deployed."""
        modal = self._modal()
        fn = modal.Function.from_name(self.app_name, cfg.function_name)
        call = fn.spawn(cfg.to_json())
        return str(call.object_id)

    def status(self, call_id: str) -> str:
        """RUNNING | SUCCEEDED | FAILED for a call id."""
        modal = self._modal()
        call = modal.FunctionCall.from_id(call_id)
        try:
            call.get(timeout=0)
        except TimeoutError:
            return "RUNNING"
        except Exception as exc:  # noqa: BLE001 - any raised result is a failed island
            if type(exc).__name__ in ("TimeoutError", "OutputExpiredError") and "timeout" in str(exc).lower():
                return "RUNNING"
            return "FAILED"
        return "SUCCEEDED"

    def cancel(self, call_id: str) -> None:
        modal = self._modal()
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def app_status(self) -> tuple[str, int] | None:
        """(state, running tasks) of this run's app from `modal app list`,
        or None when Modal lists no app by that name. Raises when the
        listing itself fails, so callers can say "unverified" rather than
        "stopped"."""
        proc = subprocess.run(
            [sys.executable, "-m", "modal", "app", "list", "--json"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"modal app list failed: {(proc.stdout + proc.stderr).strip() or proc.returncode}"
            )
        import json as _json

        # `modal app list --json` rows: app_id, description, state, tasks,
        # created_at, stopped_at (tasks is a string).
        for app in _json.loads(proc.stdout or "[]"):
            if app.get("description") == self.app_name:
                state = str(app.get("state", "")).lower()
                try:
                    tasks = int(app.get("tasks") or 0)
                except (TypeError, ValueError):
                    tasks = 0
                return state, tasks
        return None

    def stop_app(self) -> None:
        """Stop every function of this run's app (used by `yeto down`).

        `--yes` because Modal asks for confirmation and, with no terminal,
        aborts; a failure raises so `yeto down` reports it instead of
        claiming the app stopped."""
        proc = subprocess.run(
            [sys.executable, "-m", "modal", "app", "stop", "--yes", self.app_name],
            capture_output=True,
            text=True,
        )
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0 or "Aborted" in output:
            raise RuntimeError(f"modal app stop {self.app_name} failed: {output or proc.returncode}")

    def pull_tape(self, volume_name: str, remote: str, local_dir: str) -> None:
        """Download ``<volume>/<remote>`` into ``local_dir`` (`modal volume
        get`); raises on failure so the caller can report the tape missing."""
        os.makedirs(local_dir, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, "-m", "modal", "volume", "get", "--force", volume_name, remote, local_dir],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"modal volume get {volume_name} {remote} failed: "
                f"{(proc.stdout + proc.stderr).strip() or proc.returncode}"
            )

    def tail_logs(self, call_id: str, entries: int = 100):
        modal = self._modal()
        for entry in modal.FunctionCall.from_id(call_id).logs.tail(entries=entries):
            yield getattr(entry, "message", str(entry))

    def stream_logs(self, call_id: str):
        modal = self._modal()
        for entry in modal.FunctionCall.from_id(call_id).logs.stream():
            yield getattr(entry, "message", str(entry))


CONTAINER_LINE_RE = re.compile(r"\[modal-island (\d+)\] rank (\d+) container (\S+)")


class ContainerIdGuard:
    """Watches one Modal call's log stream for container-id changes.

    `--modal-retries 0` does not stop Modal from re-running the function in a new
    container after a preemption/reschedule (double billing, mixed state). The island
    prints its MODAL_TASK_ID; the first id per (island, rank) is remembered and a later
    different one trips the guard. ``on_change(message)`` is called once per change."""

    def __init__(self, on_change=None) -> None:
        self.first: dict[tuple[str, str], str] = {}
        self.changes: list[str] = []
        self.on_change = on_change

    @property
    def tripped(self) -> bool:
        return bool(self.changes)

    def feed(self, text: str) -> None:
        for part in str(text).split("\n"):
            m = CONTAINER_LINE_RE.search(part)
            if not m:
                continue
            key, cid = (m.group(1), m.group(2)), m.group(3)
            seen = self.first.setdefault(key, cid)
            if seen != cid:
                msg = (f"Modal container changed for island {key[0]} rank {key[1]}: "
                       f"{seen} -> {cid} (the run script was re-run in a new container)")
                self.changes.append(msg)
                if self.on_change is not None:
                    self.on_change(msg)


class _JobStatus:
    """Duck-types sky's job status for FleetController."""

    def __init__(self, text: str) -> None:
        self.text = text

    def is_terminal(self) -> bool:
        return self.text in ("SUCCEEDED", "FAILED")

    def __str__(self) -> str:
        return self.text


# Same markers as launcher.SYNCER_REFUSAL_MARKERS (kept here: modal_runner does
# not import the launcher at module level).
SYNCER_REFUSAL_MARKERS = ("backend identity mismatch, JOIN refused",
                          "session mismatch (HELLO refused")


class ModalIslandOps:
    """FleetController's `sky_ops` for Modal islands: names map to
    `ModalIslandConfig`s ('tasks'), job ids are call ids."""

    def __init__(self, ops: ModalOps) -> None:
        self.ops = ops
        self.calls: dict[str, str] = {}  # island name -> latest call id

    def job_status(self, name: str, job_id: str):
        return _JobStatus(self.ops.status(job_id))

    def cluster_up(self, name: str) -> bool:
        call_id = self.calls.get(name)
        return call_id is not None and self.ops.status(call_id) == "RUNNING"

    def relaunch(self, task: ModalIslandConfig, name: str):
        """Re-spawn with the SAME learner id (the config carries it);
        returns the new call id, or None when Modal refused."""
        try:
            call_id = self.ops.spawn(task)
        except Exception as exc:  # noqa: BLE001 - controller retries next poll
            print(f"[modal] relaunch of {name} failed: {exc}", file=sys.stderr)
            return None
        self.calls[name] = call_id
        # The call id is the handle for `modal.FunctionCall.from_id(..).cancel()`
        # (kill/resume tests) and for reading one island's logs.
        print(f"[modal] {name}: function call {call_id}", flush=True)
        return call_id

    def down(self, name: str) -> None:
        call_id = self.calls.get(name)
        if call_id:
            self.ops.cancel(call_id)

    def rl_strict_failure(self, name: str, job_id: str) -> str | None:
        try:
            for line in self.ops.tail_logs(job_id, entries=400):
                text = str(line).strip()
                if "[yeto-rl-strict-failure]" in text or "StrictRlInvariantError:" in text \
                        or any(m in text for m in SYNCER_REFUSAL_MARKERS):
                    return text
        except Exception:  # noqa: BLE001
            return None
        return None


class RoutingOps:
    """One `sky_ops` for a mixed fleet: Modal islands (by name) go to
    `modal_ops`, everything else to `sky_ops`. `now`/`sleep` come from
    the sky side so the controller's clock is unchanged."""

    def __init__(self, sky_ops, modal_ops: ModalIslandOps | None) -> None:
        self.sky_ops = sky_ops
        self.modal_ops = modal_ops

    def _for(self, name: str):
        if self.modal_ops is not None and is_modal_island(name):
            return self.modal_ops
        return self.sky_ops

    def job_status(self, name, job_id):
        return self._for(name).job_status(name, job_id)

    def cluster_up(self, name):
        return self._for(name).cluster_up(name)

    def relaunch(self, task, name):
        return self._for(name).relaunch(task, name)

    def down(self, name):
        return self._for(name).down(name)

    def rl_strict_failure(self, name, job_id):
        probe = getattr(self._for(name), "rl_strict_failure", None)
        return probe(name, job_id) if probe else None

    def now(self):
        return self.sky_ops.now()

    def sleep(self, seconds):
        return self.sky_ops.sleep(seconds)


# --- manual join CLI ---------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m yeto.modal_runner",
        description="Join a running Yeto fleet from a Modal GPU container "
        "(the fleet must have been launched with --external-learners).",
    )
    p.add_argument("--learner-id", type=int, required=True, help="slot printed by the launch log")
    p.add_argument("--num-learners", type=int, required=True)
    p.add_argument("--syncer-addr", default=None, help="HOST:PORT reachable from Modal")
    p.add_argument("--rl-single-island-no-sync", action="store_true",
                   help="RL island with no syncer (rl-algorithm-capabilities); no --syncer-addr")
    p.add_argument("--cluster-prefix", default="yeto", help="run name (names the Modal app)")
    p.add_argument("--gpu", default="H100", help="sky accelerator name")
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--region", default=None, help="Modal region hint (surcharge); default unpinned")
    p.add_argument("--gpu-exact", action="store_true",
                   help="pin the GPU type (H100! -- no H200 upgrade) and assert it in the container")
    p.add_argument("--training-mode", choices=["sft", "rl"], default="sft")
    p.add_argument("--rl-image", default=None, help="docker:<repo>@sha256:<hex> for RL islands")
    p.add_argument("--run-script", required=True, help="path to a file with the island run script")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for the container (repeatable)")
    p.add_argument("--requirements", default=str(REPO_ROOT / "requirements.txt"))
    p.add_argument("--follow", action="store_true", help="stream container logs until the island exits")
    return p.parse_args(argv)


def config_from_cli(ns: argparse.Namespace) -> ModalIslandConfig:
    envs = {"LEARNER_ID": str(ns.learner_id), "NUM_LEARNERS": str(ns.num_learners)}
    if not getattr(ns, "rl_single_island_no_sync", False):
        envs["SYNCER_ADDR"] = ns.syncer_addr
    for item in ns.env:
        key, _, value = item.partition("=")
        envs[key] = value
    for passthrough in ("HF_TOKEN", "WANDB_API_KEY", "CYBERGYM_API_KEY"):
        if os.environ.get(passthrough):
            envs.setdefault(passthrough, os.environ[passthrough])
    reqs: tuple[str, ...] = ()
    if ns.training_mode == "sft":
        reqs = tuple(
            line.strip()
            for line in Path(ns.requirements).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    return ModalIslandConfig(
        app_name=modal_app_name(ns.cluster_prefix),
        learner_id=ns.learner_id,
        training_mode=ns.training_mode,
        gpu=ns.gpu.upper() if ns.gpu.lower() != "a100-80gb" else "A100-80GB",
        gpus_per_node=ns.gpus_per_node,
        num_nodes=ns.num_nodes,
        run_script=Path(ns.run_script).read_text(encoding="utf-8"),
        envs=envs,
        region=ns.region,
        gpu_exact=ns.gpu_exact,
        registry_login=bool(ns.rl_image),
        image_ref=image_ref_from_rl_image(ns.rl_image) if ns.rl_image else None,
        pip_requirements=reqs,
    )


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    if ns.rl_single_island_no_sync:
        if ns.training_mode != "rl" or ns.syncer_addr is not None or ns.num_learners != 1:
            print("[modal] --rl-single-island-no-sync needs --training-mode rl, one learner "
                  "and no --syncer-addr", file=sys.stderr)
            return 2
    elif ns.syncer_addr is None:
        print("[modal] --syncer-addr is required", file=sys.stderr)
        return 2
    cfg = config_from_cli(ns)
    cfg.validate()
    if not modal_available():
        print(f"[modal] no Modal credentials: {modal_credential_hint()}", file=sys.stderr)
        return 1
    if not ns.rl_single_island_no_sync:
        resolve_syncer_for_modal(ns.syncer_addr, None)
    ops = ModalOps(cfg.app_name)
    ops.define(cfg)
    ops.deploy()
    call_id = ops.spawn(cfg)
    print(f"[modal] island {cfg.learner_id} spawned as {call_id} in app {cfg.app_name}")
    if ns.follow:
        for line in ops.stream_logs(call_id):
            print(line, flush=True)
        print(f"[modal] island {cfg.learner_id} ended: {ops.status(call_id)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
