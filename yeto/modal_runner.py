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
MODAL_FULL_NODE: dict[str, int] = {"H100": 8, "H200": 8, "B200": 8, "A100-80GB": 8}
# CPU / memory the runner reserves per GPU (Modal bills max(request,
# usage)); the shape planner prices the same reservation.
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
    timeout_s: int = DEFAULT_TIMEOUT_S
    retries: int = DEFAULT_RETRIES
    workdir: str = str(REPO_ROOT)
    python_version: str = "3.12"

    @property
    def function_name(self) -> str:
        return f"island-{self.learner_id}"

    @property
    def gpu_request(self) -> str:
        return f"{MODAL_GPUS[self.gpu]}:{self.gpus_per_node}"

    @property
    def cpu_request(self) -> int:
        return MODAL_CPU_CORES_PER_GPU * self.gpus_per_node

    @property
    def memory_request_mib(self) -> int:
        return MODAL_MEMORY_GIB_PER_GPU * self.gpus_per_node * 1024

    def validate(self) -> None:
        validate_modal_shape(self.gpu, self.gpus_per_node, self.num_nodes)
        if self.training_mode == "rl":
            if not self.image_ref or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-fA-F]{64}", self.image_ref):
                raise ValueError(
                    "RL islands on Modal must pin the Miles image by digest "
                    "(<repository>@sha256:<64 hex>), the same digest --rl-image gives sky"
                )
        if (self.volume_name is None) != (self.volume_mount is None):
            raise ValueError("volume_name and volume_mount go together")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ModalIslandConfig":
        data = json.loads(text)
        data["pip_requirements"] = tuple(data.get("pip_requirements") or ())
        return cls(**data)


def image_ref_from_rl_image(rl_image: str) -> str:
    """`--rl-image docker:<repo>@sha256:<hex>` -> the registry reference."""
    ref = rl_image[len("docker:"):] if rl_image.startswith("docker:") else rl_image
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-fA-F]{64}", ref):
        raise ValueError(f"--rl-image must pin a digest (docker:<repo>@sha256:<64 hex>), got {rl_image!r}")
    return ref


def skypilot_env(rank: int, container_ips: list[str], gpus_per_node: int) -> dict[str, str]:
    """The variables the sky task scripts read, emulated for a container."""
    return {
        "SKYPILOT_NODE_IPS": "\n".join(container_ips),
        "SKYPILOT_NUM_NODES": str(len(container_ips)),
        "SKYPILOT_NODE_RANK": str(rank),
        "SKYPILOT_NUM_GPUS_PER_NODE": str(gpus_per_node),
    }


def container_command(run_script: str) -> list[str]:
    """How the island's run script is executed inside the container."""
    return ["bash", "-lc", f"cd {shlex.quote(CONTAINER_WORKDIR)} && {run_script}"]


def island_main(cfg_json: str) -> int:
    """Body of the Modal function: runs in the container. Returns the run
    script's exit code (nonzero raises so Modal records the failure)."""
    cfg = ModalIslandConfig.from_json(cfg_json)
    if cfg.num_nodes > 1:
        import modal.experimental

        info = modal.experimental.get_cluster_info()
        rank, ips = int(info.rank), list(info.container_ips)
    else:
        rank, ips = 0, ["127.0.0.1"]
    env = {**os.environ, **cfg.envs, **skypilot_env(rank, ips, cfg.gpus_per_node), "HOME": "/root"}
    print(f"[modal-island {cfg.learner_id}] rank {rank}/{len(ips)} starting", flush=True)
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
        if cfg.training_mode == "rl":
            image = modal.Image.from_registry(cfg.image_ref)
        else:
            image = modal.Image.debian_slim(python_version=cfg.python_version)
            if cfg.pip_requirements:
                image = image.pip_install(*cfg.pip_requirements)
        image = image.env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
        # Must be the last step: Modal rejects any build step after an
        # add_local_* with copy=False (the files are mounted at start-up).
        return image.add_local_dir(
            cfg.workdir,
            CONTAINER_WORKDIR,
            copy=False,
            ignore=MODAL_WORKDIR_IGNORE,
        )

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
        if cfg.volume_name and cfg.volume_mount:
            kwargs["volumes"] = {
                cfg.volume_mount: modal.Volume.from_name(cfg.volume_name, create_if_missing=True)
            }
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

    def tail_logs(self, call_id: str, entries: int = 100):
        modal = self._modal()
        for entry in modal.FunctionCall.from_id(call_id).logs.tail(entries=entries):
            yield getattr(entry, "message", str(entry))

    def stream_logs(self, call_id: str):
        modal = self._modal()
        for entry in modal.FunctionCall.from_id(call_id).logs.stream():
            yield getattr(entry, "message", str(entry))


class _JobStatus:
    """Duck-types sky's job status for FleetController."""

    def __init__(self, text: str) -> None:
        self.text = text

    def is_terminal(self) -> bool:
        return self.text in ("SUCCEEDED", "FAILED")

    def __str__(self) -> str:
        return self.text


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
        return call_id

    def down(self, name: str) -> None:
        call_id = self.calls.get(name)
        if call_id:
            self.ops.cancel(call_id)

    def rl_strict_failure(self, name: str, job_id: str) -> str | None:
        try:
            for line in self.ops.tail_logs(job_id, entries=400):
                text = str(line).strip()
                if "[yeto-rl-strict-failure]" in text or "StrictRlInvariantError:" in text:
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
    p.add_argument("--syncer-addr", required=True, help="HOST:PORT reachable from Modal")
    p.add_argument("--cluster-prefix", default="yeto", help="run name (names the Modal app)")
    p.add_argument("--gpu", default="H100", help="sky accelerator name")
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--region", default=None, help="Modal region hint (surcharge); default unpinned")
    p.add_argument("--training-mode", choices=["sft", "rl"], default="sft")
    p.add_argument("--rl-image", default=None, help="docker:<repo>@sha256:<hex> for RL islands")
    p.add_argument("--run-script", required=True, help="path to a file with the island run script")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for the container (repeatable)")
    p.add_argument("--requirements", default=str(REPO_ROOT / "requirements.txt"))
    p.add_argument("--follow", action="store_true", help="stream container logs until the island exits")
    return p.parse_args(argv)


def config_from_cli(ns: argparse.Namespace) -> ModalIslandConfig:
    envs = {"SYNCER_ADDR": ns.syncer_addr, "LEARNER_ID": str(ns.learner_id), "NUM_LEARNERS": str(ns.num_learners)}
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
        image_ref=image_ref_from_rl_image(ns.rl_image) if ns.rl_image else None,
        pip_requirements=reqs,
    )


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    cfg = config_from_cli(ns)
    cfg.validate()
    if not modal_available():
        print(f"[modal] no Modal credentials: {modal_credential_hint()}", file=sys.stderr)
        return 1
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
