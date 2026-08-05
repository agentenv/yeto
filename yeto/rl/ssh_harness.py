"""Direct SSH acceptance harness for the current Miles RL boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from . import MILES_COMMIT, MILES_PEFT_VERSION, MILES_REPOSITORY
from .core import CanonicalTensorSpec, canonical_state, policy_hash, tensors_from_flat

PLAN_SCHEMA = 1
LEARNERS = 2
SYNCER_PORT = 29400
RAY_PORT = 6379
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_ROOT = ".cache/yeto-rl-ssh"

_RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,47}\Z")
_REMOTE_PATH = re.compile(r"[a-zA-Z0-9._/-]+\Z")
_SSH_TARGET = re.compile(r"(?:[a-zA-Z0-9._-]+@)?[a-zA-Z0-9._-]+\Z")
_HOST = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9.-]*\Z")
_DOCKER_DIGEST = re.compile(r".+@sha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class HarnessError(RuntimeError):
    pass


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value):
        raise HarnessError(
            "--run-id must be 1-48 letters, digits, dots, underscores, or hyphens"
        )
    return value


def _validate_remote_path(value: str, flag: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith(("/", "~"))
        or not _REMOTE_PATH.fullmatch(value)
        or ".." in path.parts
    ):
        raise HarnessError(f"{flag} must be a safe path relative to remote $HOME")
    return value.rstrip("/")


def _validate_target(value: str) -> str:
    if not _SSH_TARGET.fullmatch(value):
        raise HarnessError(f"invalid SSH target {value!r}")
    return value


def _target_host(target: str) -> str:
    return target.rsplit("@", 1)[-1]


def _validate_address(value: str) -> tuple[str, int]:
    host, separator, raw_port = value.rpartition(":")
    if not separator or not _HOST.fullmatch(host):
        raise HarnessError("--syncer-address must be a reachable HOST:29400")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise HarnessError("--syncer-address has a non-integer port") from exc
    if port != SYNCER_PORT:
        raise HarnessError(f"the Miles RL harness uses syncer port {SYNCER_PORT}")
    return host, port


def _docker_ref(value: str) -> str:
    if "," in value or "=" in value:
        raise HarnessError("the SSH harness requires one Docker image digest")
    image = value.removeprefix("docker:")
    if not _DOCKER_DIGEST.fullmatch(image):
        raise HarnessError(
            "the SSH harness requires --rl-image docker:REPO@sha256:DIGEST"
        )
    return image


def _syncer_source_sha256() -> str:
    root = REPO_ROOT / "syncer"
    files = [root / "Cargo.toml", root / "Cargo.lock"]
    files.extend(sorted((root / "src").rglob("*.rs")))
    if (root / "build.rs").is_file():
        files.append(root / "build.rs")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _local_data_sha256(path: str | Path) -> str:
    from ..provenance import file_sha256

    root = Path(path).expanduser()
    if root.is_symlink():
        raise HarnessError("local dataset may not be a symlink")
    if root.is_file():
        return file_sha256(root)
    if not root.is_dir():
        raise HarnessError(f"local dataset does not exist: {root}")

    files = []
    for item in root.rglob("*"):
        if item.is_symlink():
            raise HarnessError("local dataset may not contain symlinks")
        if item.is_file():
            files.append(item)
        elif not item.is_dir():
            raise HarnessError(f"local dataset contains an unsupported entry: {item}")
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _plan_digest(plan: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(plan).encode()).hexdigest()


def _write_plan(path: str | Path, plan: dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    envelope = {"plan": plan, "sha256": _plan_digest(plan)}
    _atomic_bytes(
        destination,
        (json.dumps(envelope, sort_keys=True, indent=2) + "\n").encode(),
    )
    return destination


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise HarnessError("unsupported SSH harness plan schema")
    _validate_run_id(str(plan.get("run_id", "")))
    _validate_remote_path(str(plan.get("remote_run", "")), "remote_run")
    if plan.get("remote_env_file") is not None:
        _validate_remote_path(plan["remote_env_file"], "remote_env_file")
    islands = plan.get("islands")
    if not isinstance(islands, list) or len(islands) != LEARNERS:
        raise HarnessError("plan must contain exactly two Miles islands")
    all_hosts = []
    node_counts = set()
    gpu_counts = set()
    for island in islands:
        hosts = island.get("hosts") if isinstance(island, dict) else None
        gpus = island.get("gpus_per_node") if isinstance(island, dict) else None
        if not isinstance(hosts, list) or not hosts or not isinstance(gpus, int) or gpus <= 0:
            raise HarnessError("each island needs hosts and a positive GPU count")
        node_counts.add(len(hosts))
        gpu_counts.add(gpus)
        all_hosts.extend(_validate_target(host) for host in hosts)
    if len(node_counts) != 1:
        raise HarnessError("both acceptance islands must use the same node count")
    if len(gpu_counts) != 1:
        raise HarnessError("both acceptance islands must use the same GPU count")
    if len(set(all_hosts)) != len(all_hosts):
        raise HarnessError("each SSH target may belong to only one island")
    _, port = _validate_address(str(plan.get("syncer_address", "")))
    if plan.get("syncer_port") != port:
        raise HarnessError("plan syncer port does not match its address")
    _docker_ref(str(plan.get("docker_image", "")))
    if plan.get("miles") != {
        "repository": MILES_REPOSITORY,
        "commit": MILES_COMMIT,
        "peft_version": MILES_PEFT_VERSION,
    }:
        raise HarnessError("plan does not use the current pinned Miles revision")
    for name in ("source_sha256", "reward_sha256", "syncer_source_sha256"):
        if not _SHA256.fullmatch(str(plan.get(name, ""))):
            raise HarnessError(f"plan has an invalid {name}")
    learner = plan.get("learner")
    if not isinstance(learner, dict):
        raise HarnessError("plan has no learner configuration")
    if not _REVISION.fullmatch(str(learner.get("model_revision", ""))):
        raise HarnessError("plan model revision must be an immutable commit")
    data_revision = learner.get("data_revision")
    data_local_path = learner.get("data_local_path")
    data_sha256 = learner.get("data_sha256")
    if data_local_path is None:
        if not _REVISION.fullmatch(str(data_revision or "")) or data_sha256 is not None:
            raise HarnessError("plan dataset revision must be an immutable commit")
    elif (
        not isinstance(data_local_path, str)
        or not Path(data_local_path).is_absolute()
        or data_revision is not None
        or not _SHA256.fullmatch(str(data_sha256 or ""))
        or not str(learner.get("data", "")).startswith("/workspace/data/dataset")
    ):
        raise HarnessError("plan has an invalid local dataset identity")
    for name in (
        "global_rounds",
        "groups_per_round",
        "samples_per_group",
        "over_sampling_batch_size",
        "optimizer_steps",
        "rollout_max_response_len",
        "lora_r",
        "seq_len",
        "wan_streams",
    ):
        if not isinstance(learner.get(name), int) or learner[name] <= 0:
            raise HarnessError(f"plan has an invalid learner {name}")
    sync_preset = learner.get("sync_preset", "strict-avg")
    if sync_preset not in {"strict-avg", "decoupled"}:
        raise HarnessError("plan has an invalid learner sync_preset")
    fragments = learner.get("fragments", 1)
    pipeline = learner.get("pipeline", 1)
    local_horizon = learner.get("local_horizon", 1)
    total_fragment_steps = learner.get(
        "total_fragment_steps", learner["global_rounds"]
    )
    for name, value in (
        ("fragments", fragments),
        ("pipeline", pipeline),
        ("local_horizon", local_horizon),
        ("total_fragment_steps", total_fragment_steps),
    ):
        if not isinstance(value, int) or value <= 0:
            raise HarnessError(f"plan has an invalid learner {name}")
    if sync_preset == "strict-avg":
        if (fragments, pipeline, local_horizon, total_fragment_steps) != (
            1,
            1,
            1,
            learner["global_rounds"],
        ):
            raise HarnessError("strict-avg plan has decoupled settings")
    elif (
        fragments < 2
        or pipeline > fragments
        or local_horizon < 2
        or total_fragment_steps != learner["global_rounds"] * fragments
    ):
        raise HarnessError("invalid decoupled learner settings")
    world = len(islands[0]["hosts"]) * islands[0]["gpus_per_node"]
    if learner["groups_per_round"] * learner["samples_per_group"] % world:
        raise HarnessError("plan Miles batch does not divide across island GPUs")
    dynamic_filter = learner.get("dynamic_sampling_filter_path")
    if dynamic_filter:
        parts = str(dynamic_filter).split(".")
        if len(parts) < 2 or any(not part.isidentifier() for part in parts):
            raise HarnessError(
                "plan has an invalid dynamic_sampling_filter_path"
            )
        if learner["over_sampling_batch_size"] <= learner["groups_per_round"]:
            raise HarnessError(
                "variance-aware filtering requires oversampling beyond the training batch"
            )
    max_replacements = learner.get("dynamic_sampling_max_replacements")
    if max_replacements is not None and (
        not isinstance(max_replacements, int) or max_replacements < 0
    ):
        raise HarnessError(
            "plan has an invalid dynamic_sampling_max_replacements"
        )
    timeout_minutes = learner.get("rl_distributed_timeout_minutes", 10)
    if not isinstance(timeout_minutes, int) or timeout_minutes <= 0:
        raise HarnessError(
            "plan has an invalid rl_distributed_timeout_minutes"
        )


def load_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = Path(path).expanduser().resolve()
    try:
        envelope = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read harness plan {plan_path}") from exc
    plan = envelope.get("plan") if isinstance(envelope, dict) else None
    digest = envelope.get("sha256") if isinstance(envelope, dict) else None
    if not isinstance(plan, dict) or digest != _plan_digest(plan):
        raise HarnessError("SSH harness plan digest mismatch")
    _validate_plan(plan)
    return plan_path, plan


def _has_option(values: Sequence[str], option: str) -> bool:
    return option in values or any(value.startswith(option + "=") for value in values)


def _strip_separator(values: Sequence[str]) -> list[str]:
    values = list(values)
    return values[1:] if values[:1] == ["--"] else values


def _parse_islands(values: Sequence[str], gpus_per_node: int) -> list[dict[str, Any]]:
    if len(values) != LEARNERS or gpus_per_node <= 0:
        raise HarnessError("prepare requires two --host groups and positive GPUs per node")
    islands = []
    for value in values:
        hosts = [_validate_target(item.strip()) for item in value.split(",") if item.strip()]
        if not hosts:
            raise HarnessError("each --host must contain at least one SSH target")
        islands.append({"hosts": hosts, "gpus_per_node": gpus_per_node})
    if len({len(island["hosts"]) for island in islands}) != 1:
        raise HarnessError("both --host groups must contain the same number of nodes")
    if len({host for island in islands for host in island["hosts"]}) != sum(
        len(island["hosts"]) for island in islands
    ):
        raise HarnessError("each SSH target may belong to only one island")
    return islands


def _resolved_launch_args(namespace, islands):
    from ..cli import build_parser
    from ..launcher import prepare_launch_args

    launch_args = _strip_separator(namespace.launch_args)
    for reserved in (
        "--gpu",
        "--training-mode",
        "--rl-runtime",
        "--controller",
        "--cluster-prefix",
    ):
        if _has_option(launch_args, reserved):
            raise HarnessError(f"{reserved} is set by the SSH harness")
    gpu = ",".join(
        f"ssh:{len(island['hosts'])}x{island['gpus_per_node']}xa100@island-{index}"
        for index, island in enumerate(islands)
    )
    argv = [
        "launch",
        "--gpu",
        gpu,
        "--training-mode",
        "rl",
        "--rl-runtime",
        "miles",
        "--controller",
        "local",
        "--cluster-prefix",
        namespace.run_id,
        *launch_args,
    ]
    args = build_parser().parse_args(argv)
    prepare_launch_args(args, allow_local_rl_data=True)
    return args


def prepare(namespace) -> Path:
    run_id = _validate_run_id(namespace.run_id)
    islands = _parse_islands(namespace.host, namespace.gpus_per_node)
    remote_root = _validate_remote_path(namespace.remote_root, "--remote-root")
    if namespace.remote_env_file is not None:
        _validate_remote_path(namespace.remote_env_file, "--remote-env-file")
    syncer_address = namespace.syncer_address or (
        f"{_target_host(islands[0]['hosts'][0])}:{SYNCER_PORT}"
    )
    _, syncer_port = _validate_address(syncer_address)
    args = _resolved_launch_args(namespace, islands)
    local_run = (
        Path(namespace.output_dir).expanduser()
        if namespace.output_dir
        else Path.home() / ".yeto" / "ssh-runs" / run_id
    ).resolve()
    plan_path = local_run / "plan.json"
    if plan_path.exists():
        raise HarnessError(f"plan already exists at {plan_path}")
    dataset = args._provenance["dataset"]
    data = args.data
    data_local_path = None
    data_sha256 = None
    if dataset["source"] == "local":
        source = Path(args.data).expanduser().absolute()
        data_sha256 = _local_data_sha256(source)
        source = source.resolve()
        data_local_path = str(source)
        data = "/workspace/data/dataset"
        if source.is_file():
            data += source.suffix

    plan = {
        "schema": PLAN_SCHEMA,
        "run_id": run_id,
        "created_unix_ns": time.time_ns(),
        "remote_run": f"{remote_root}/{run_id}",
        "remote_env_file": namespace.remote_env_file,
        "ssh_options": list(namespace.ssh_option),
        "islands": islands,
        "syncer_address": syncer_address,
        "syncer_port": syncer_port,
        "docker_image": _docker_ref(args.rl_image),
        "miles": {
            "repository": MILES_REPOSITORY,
            "commit": MILES_COMMIT,
            "peft_version": MILES_PEFT_VERSION,
        },
        "source_sha256": args.source_sha256,
        "reward_sha256": args.reward_sha256,
        "syncer_source_sha256": _syncer_source_sha256(),
        "learner": {
            "model": args.model,
            "model_revision": args.model_revision,
            "data": data,
            "data_revision": args.data_revision,
            "data_local_path": data_local_path,
            "data_sha256": data_sha256,
            "reward_function": args.reward_function,
            "global_rounds": args.total_steps,
            "sync_preset": getattr(args, "rl_sync_preset", "strict-avg"),
            "fragments": getattr(args, "fragments", 1),
            "pipeline": getattr(args, "pipeline", 1),
            "local_horizon": getattr(args, "local_rl_rounds_per_sync", 1),
            "total_fragment_steps": getattr(
                args, "rl_total_fragment_steps", args.total_steps
            ),
            "groups_per_round": args.rollout_batch_size,
            "samples_per_group": args.n_samples_per_prompt,
            "over_sampling_batch_size": args.over_sampling_batch_size,
            "dynamic_sampling_filter_path": getattr(
                args, "dynamic_sampling_filter_path", None
            ),
            "dynamic_sampling_max_replacements": getattr(
                args, "dynamic_sampling_max_replacements", None
            ),
            "rl_offload_train": bool(getattr(args, "rl_offload_train", False)),
            "rl_distributed_timeout_minutes": getattr(
                args, "rl_distributed_timeout_minutes", 10
            ),
            "optimizer_steps": 1,
            "rollout_max_response_len": args.rollout_max_response_len,
            "custom_generate_function_path": args.custom_generate_function_path,
            "use_session_server": args.use_session_server,
            "session_server_ip": args.session_server_ip,
            "session_server_port": args.session_server_port,
            "tito_model": args.tito_model,
            "expert_parallel": args.expert_parallel,
            "lora_r": args.lora_r,
            "lora_targets": args.lora_targets,
            "inner_lr": args.inner_lr,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "wan_streams": args.wan_streams,
            "trust_remote_code": args.trust_remote_code,
            "cybergym_url": args.cybergym_url,
            "cybergym_agent_id": args.cybergym_agent_id,
            "cybergym_timeout": args.cybergym_timeout,
        },
    }
    _validate_plan(plan)
    _write_plan(plan_path, plan)
    print(f"prepared {plan_path}")
    print(f"syncer {syncer_address} on {islands[0]['hosts'][0]}")
    return plan_path


def _require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise HarnessError(f"{name} is required on the controller")


def _run(
    command: Sequence[str], *, capture: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(command), flush=True)
    return subprocess.run(
        list(command), check=check, text=True, capture_output=capture
    )


def _ssh_command(plan: dict[str, Any], target: str, script: str) -> list[str]:
    return [
        "ssh",
        *plan.get("ssh_options", []),
        target,
        "bash",
        "-lc",
        shlex.quote(script),
    ]


def _ssh(
    plan: dict[str, Any],
    target: str,
    script: str,
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(_ssh_command(plan, target, script), capture=capture, check=check)


def _rsync_shell(plan: dict[str, Any]) -> str:
    return shlex.join(["ssh", *plan.get("ssh_options", [])])


def _remote_vars(plan: dict[str, Any]) -> str:
    return f'RUN="$HOME/{plan["remote_run"]}"'


def _all_hosts(plan: dict[str, Any]) -> list[str]:
    return [host for island in plan["islands"] for host in island["hosts"]]


def _attest_local(plan: dict[str, Any]) -> None:
    from ..provenance import python_spec_sha256, verify_source_tree_sha256

    verify_source_tree_sha256(plan["source_sha256"])
    if _syncer_source_sha256() != plan["syncer_source_sha256"]:
        raise HarnessError("syncer source changed after the plan was prepared")
    if python_spec_sha256(plan["learner"]["reward_function"], base_dir=REPO_ROOT) != plan[
        "reward_sha256"
    ]:
        raise HarnessError("reward source changed after the plan was prepared")
    data_local_path = plan["learner"].get("data_local_path")
    if data_local_path is not None and _local_data_sha256(data_local_path) != plan[
        "learner"
    ]["data_sha256"]:
        raise HarnessError("local dataset changed after the plan was prepared")


def deploy(plan_path: str | Path) -> None:
    plan_file, plan = load_plan(plan_path)
    _require_program("ssh")
    _require_program("rsync")
    _attest_local(plan)
    digest = _plan_digest(plan)
    local_data = plan["learner"].get("data_local_path")
    excludes = (
        ".git/",
        ".venv/",
        ".worktree/",
        ".env",
        ".env.*",
        "__pycache__/",
        ".pytest_cache/",
        "syncer/target/",
        "checkpoints/",
        "runs/",
        "wandb/",
        "build/",
        "dist/",
    )
    for target in _all_hosts(plan):
        data_directory = '\nmkdir -p "$RUN/data"' if local_data is not None else ""
        initialize = f"""set -euo pipefail
{_remote_vars(plan)}
if [ -f "$RUN/control/plan.sha256" ]; then
  test "$(cat "$RUN/control/plan.sha256")" = {shlex.quote(digest)}
else
  if [ -f "$RUN/control/deploying.sha256" ]; then
    test "$(cat "$RUN/control/deploying.sha256")" = {shlex.quote(digest)}
  else
    if [ -e "$RUN" ]; then test -z "$(find "$RUN" -mindepth 1 -maxdepth 1 -print -quit)"; fi
    mkdir -p "$RUN"/control
    printf '%s\n' {shlex.quote(digest)} > "$RUN/control/deploying.sha256"
  fi
fi
{data_directory}
"""
        _ssh(plan, target, initialize)
        _run(
            [
                "rsync",
                "-az",
                "--delete",
                *[item for pattern in excludes for item in ("--exclude", pattern)],
                "-e",
                _rsync_shell(plan),
                f"{REPO_ROOT}/",
                f"{target}:{plan['remote_run']}/source/",
            ]
        )
        _run(
            [
                "rsync",
                "-az",
                "-e",
                _rsync_shell(plan),
                str(plan_file),
                f"{target}:{plan['remote_run']}/control/plan.json",
            ]
        )
        if local_data is not None:
            source = Path(local_data)
            is_directory = source.is_dir()
            _run(
                [
                    "rsync",
                    "-az",
                    "-e",
                    _rsync_shell(plan),
                    str(source) + ("/" if is_directory else ""),
                    f"{target}:{plan['remote_run']}/data/"
                    f"{PurePosixPath(plan['learner']['data']).name}"
                    + ("/" if is_directory else ""),
                ]
            )
        _ssh(
            plan,
            target,
            f"""set -euo pipefail
{_remote_vars(plan)}
mv "$RUN/control/deploying.sha256" "$RUN/control/plan.sha256" 2>/dev/null || \
  test "$(cat "$RUN/control/plan.sha256")" = {shlex.quote(digest)}
""",
        )
    print(f"deployed {plan['run_id']} to {len(_all_hosts(plan))} host(s)")


def _host_setup_script(plan: dict[str, Any], gpus_per_node: int) -> str:
    return f"""set -euo pipefail
{_remote_vars(plan)}
test "$(cat "$RUN/control/plan.sha256")" = {shlex.quote(_plan_digest(plan))}
command -v docker >/dev/null
command -v git >/dev/null
command -v nvidia-smi >/dev/null
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge {gpus_per_node}
docker info >/dev/null
mkdir -p "$HOME/.cache/huggingface" "$RUN/miles"
docker pull {shlex.quote(plan['docker_image'])}
if [ ! -d "$RUN/miles/.git" ]; then
  rmdir "$RUN/miles"
  git clone --no-checkout {shlex.quote(MILES_REPOSITORY)} "$RUN/miles"
fi
git -C "$RUN/miles" fetch --depth 1 origin {shlex.quote(MILES_COMMIT)}
git -C "$RUN/miles" checkout --detach {shlex.quote(MILES_COMMIT)}
"""


def _build_syncer(plan: dict[str, Any]) -> None:
    script = f"""set -euo pipefail
{_remote_vars(plan)}
ACTUAL="$(docker run --rm --interactive \
  --volume "$RUN/source/syncer:/syncer:ro" \
  --entrypoint python3 {shlex.quote(plan['docker_image'])} - <<'PY'
import hashlib
from pathlib import Path

root = Path("/syncer")
files = [root / "Cargo.toml", root / "Cargo.lock"]
files.extend(sorted((root / "src").rglob("*.rs")))
if (root / "build.rs").is_file():
    files.append(root / "build.rs")
digest = hashlib.sha256()
for path in files:
    relative = path.relative_to(root).as_posix().encode()
    data = path.read_bytes()
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)
print(digest.hexdigest())
PY
)"
if [ "$ACTUAL" != {shlex.quote(plan['syncer_source_sha256'])} ]; then
  echo "remote syncer source identity mismatch" >&2
  exit 1
fi
CARGO="$HOME/.cargo/bin/cargo"
if [ ! -x "$CARGO" ]; then CARGO="$(command -v cargo || true)"; fi
test -n "$CARGO" || {{ echo 'cargo is required on the syncer host' >&2; exit 1; }}
command -v cc >/dev/null
"$CARGO" build --release --locked --manifest-path "$RUN/source/syncer/Cargo.toml"
mkdir -p "$RUN/state"
install -m 0755 "$RUN/source/syncer/target/release/yeto-syncer" "$RUN/state/yeto-syncer"
"""
    _ssh(plan, plan["islands"][0]["hosts"][0], script)


def _argv(command: Sequence[str], *options: tuple[str, Any]) -> list[str]:
    values = list(command)
    for flag, value in options:
        values.append(flag)
        if value is not None:
            values.append(str(value))
    return values


def _syncer_argv(plan: dict[str, Any]) -> list[str]:
    learner = plan["learner"]
    sync_preset = learner.get("sync_preset", "strict-avg")
    decoupled = sync_preset == "decoupled"
    return _argv(
        ["$RUN/state/yeto-syncer"],
        ("--port", plan["syncer_port"]),
        ("--learners", LEARNERS),
        ("--quorum", LEARNERS),
        ("--grace-ms", 0),
        ("--pipeline", learner.get("pipeline", 1) if decoupled else 1),
        (
            "--sync-interval-steps",
            learner.get("local_horizon", 1) if decoupled else 0,
        ),
        ("--delta-correction", "none"),
        (
            "--total-steps",
            learner.get("total_fragment_steps", learner["global_rounds"])
            if decoupled
            else learner["global_rounds"],
        ),
        ("--outer-lr", 0.7 if decoupled else 1),
        ("--outer-momentum", 0.9 if decoupled else 0),
        ("--max-base-lag", 0),
        ("--learner-weight", "equal"),
        ("--checkpoint-path", "$RUN/state/state.ckpt"),
        ("--checkpoint-every", 1),
        ("--resume", None),
        ("--event-tape", "$RUN/state/events.jsonl"),
    )


def _shell_join_with_run(values: Sequence[str]) -> str:
    return " ".join(
        value if value.startswith("$RUN/") else shlex.quote(value) for value in values
    )


def _start_syncer(plan: dict[str, Any]) -> None:
    command = _shell_join_with_run(_syncer_argv(plan))
    script = f"""set -euo pipefail
{_remote_vars(plan)}
PID_FILE="$RUN/state/syncer.pid"
if [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "syncer already running pid=$(cat "$PID_FILE")"
  exit 0
fi
nohup setsid {command} >> "$RUN/state/syncer.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
sleep 1
kill -0 "$PID"
"""
    _ssh(plan, plan["islands"][0]["hosts"][0], script)


def _wait_for_syncer(plan: dict[str, Any], timeout_s: int = 120) -> None:
    host, port = _validate_address(plan["syncer_address"])
    for target in _all_hosts(plan):
        _ssh(
            plan,
            target,
            f"""set -euo pipefail
deadline=$((SECONDS + {timeout_s}))
until timeout 2 bash -c 'exec 3<>/dev/tcp/{host}/{port}' 2>/dev/null; do
  [ "$SECONDS" -lt "$deadline" ] || exit 1
  sleep 2
done
""",
        )


def _learner_argv(plan: dict[str, Any], learner_id: int) -> list[str]:
    if learner_id not in range(LEARNERS):
        raise HarnessError("--learner-id must be 0 or 1")
    learner = plan["learner"]
    island = plan["islands"][learner_id]
    values = _argv(
        ["python3", "-m", "yeto.rl.learner"],
        ("--model", learner["model"]),
        ("--model-revision", learner["model_revision"]),
        ("--data", learner["data"]),
        ("--syncer", plan["syncer_address"]),
        ("--learner-id", learner_id),
        ("--reward-function", learner["reward_function"]),
        ("--reward-sha256", plan["reward_sha256"]),
        ("--source-sha256", plan["source_sha256"]),
        ("--global-rounds", learner["global_rounds"]),
        ("--sync-preset", learner.get("sync_preset", "strict-avg")),
        ("--fragments", learner.get("fragments", 1)),
        ("--pipeline", learner.get("pipeline", 1)),
        ("--local-horizon", learner.get("local_horizon", 1)),
        (
            "--total-fragment-steps",
            learner.get("total_fragment_steps", learner["global_rounds"]),
        ),
        ("--groups-per-round", learner["groups_per_round"]),
        ("--samples-per-group", learner["samples_per_group"]),
        ("--over-sampling-batch-size", learner["over_sampling_batch_size"]),
        (
            "--rl-distributed-timeout-minutes",
            learner.get("rl_distributed_timeout_minutes", 10),
        ),
        ("--optimizer-steps", learner["optimizer_steps"]),
        ("--rollout-max-response-len", learner["rollout_max_response_len"]),
        ("--completed-groups-path", "/workspace/state/island-checkpoint.pt"),
        ("--event-tape", "/workspace/output/events.jsonl"),
        ("--audit-dir", "/workspace/audit"),
        ("--actor-num-nodes", len(island["hosts"])),
        ("--actor-num-gpus-per-node", island["gpus_per_node"]),
        ("--lora-r", learner["lora_r"]),
        ("--lora-targets", learner["lora_targets"]),
        ("--inner-lr", learner["inner_lr"]),
        ("--seq-len", learner["seq_len"]),
        ("--seed", learner["seed"]),
        ("--wan-streams", learner["wan_streams"]),
        ("--miles-root", "/workspace/miles"),
    )
    if learner.get("dynamic_sampling_max_replacements") is not None:
        values.extend(
            (
                "--dynamic-sampling-max-replacements",
                str(learner["dynamic_sampling_max_replacements"]),
            )
        )
    if learner.get("dynamic_sampling_filter_path"):
        values.extend(
            (
                "--dynamic-sampling-filter-path",
                learner["dynamic_sampling_filter_path"],
            )
        )
    if learner.get("rl_offload_train"):
        values.append("--rl-offload-train")
    if learner.get("data_revision") is not None:
        values.extend(("--data-revision", learner["data_revision"]))
    if learner.get("expert_parallel") is not None:
        values.extend(("--expert-parallel", str(learner["expert_parallel"])))
    if learner.get("custom_generate_function_path"):
        values.extend(
            (
                "--custom-generate-function-path",
                learner["custom_generate_function_path"],
            )
        )
    if learner.get("use_session_server"):
        values.append("--use-session-server")
        if learner.get("session_server_ip"):
            values.extend(("--session-server-ip", learner["session_server_ip"]))
        if learner.get("session_server_port"):
            values.append("--session-server-port")
            values.extend(str(port) for port in learner["session_server_port"])
        if learner.get("tito_model"):
            values.extend(("--tito-model", learner["tito_model"]))
    if learner["trust_remote_code"]:
        values.append("--trust-remote-code")
    return values


def _container_name(plan: dict[str, Any], learner_id: int, node_id: int) -> str:
    return f"yeto-rl-{plan['run_id']}-i{learner_id}-n{node_id}"


def _node_start_script(
    plan: dict[str, Any], learner_id: int, node_id: int
) -> str:
    island = plan["islands"][learner_id]
    if node_id not in range(len(island["hosts"])):
        raise HarnessError("node ID is outside the island topology")
    name = _container_name(plan, learner_id, node_id)
    head = _target_host(island["hosts"][0])
    gpus = island["gpus_per_node"]
    gpu_request = '"device=' + ",".join(str(index) for index in range(gpus)) + '"'
    env_file = (
        f' --env-file "$HOME/{plan["remote_env_file"]}"'
        if plan.get("remote_env_file")
        else ""
    )
    learner = plan["learner"]
    data_volume = (
        '  --volume "$RUN/data:/workspace/data:ro" \\\n'
        if learner.get("data_local_path") is not None
        else ""
    )
    common = (
        f"python3 -m pip install -q --no-deps 'peft=={MILES_PEFT_VERSION}'; "
        "export PYTHONPATH=/workspace/yeto:/workspace/miles${PYTHONPATH:+:$PYTHONPATH}; "
        "ray stop --force >/dev/null 2>&1 || true; "
    )
    if node_id == 0:
        wait = f"""python3 - <<'PY'
import ray, time
ray.init(address={head!r} + ':{RAY_PORT}')
deadline = time.monotonic() + 300
while len([node for node in ray.nodes() if node['Alive']]) < {len(island['hosts'])}:
    if time.monotonic() >= deadline:
        raise RuntimeError('Ray island did not reach its fixed node count')
    time.sleep(1)
ray.shutdown()
PY
"""
        command = (
            common
            + f"HEAD_IP=$(getent ahostsv4 {shlex.quote(head)} | awk 'NR == 1 {{print $1}}'); "
            + "test -n \"$HEAD_IP\"; "
            + "ray start --head --node-ip-address=\"$HEAD_IP\" "
            f"--port={RAY_PORT} --include-dashboard=true; "
            + wait
            + "exec " + shlex.join(_learner_argv(plan, learner_id))
        )
    else:
        command = (
            common
            + f"until ray start --address={shlex.quote(head)}:{RAY_PORT}; do sleep 2; done; "
            + f"while ray status --address={shlex.quote(head)}:{RAY_PORT} >/dev/null 2>&1; "
            "do sleep 5; done"
        )
    return f"""set -euo pipefail
{_remote_vars(plan)}
if docker inspect {shlex.quote(name)} >/dev/null 2>&1; then
  STATUS="$(docker inspect --format '{{{{.State.Status}}}}' {shlex.quote(name)})"
  if [ "$STATUS" = running ]; then exit 0; fi
  echo 'container exists but is not running; use restart-learner' >&2
  exit 1
fi
mkdir -p "$RUN/island-{learner_id}"/{{state,output,audit}}
docker run --detach \
  --name {shlex.quote(name)} \
  --gpus {shlex.quote(gpu_request)} \
  --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --env PYTHONUNBUFFERED=1 \
  --env HF_HOME=/workspace/hf \
  --env HF_HUB_ENABLE_HF_TRANSFER=1{env_file} \
  --env NVTE_FLASH_ATTN=0 --env NVTE_FUSED_ATTN=0 --env NVTE_UNFUSED_ATTN=1 \
  --env CYBERGYM_URL={shlex.quote(learner['cybergym_url'])} \
  --env CYBERGYM_AGENT_ID={shlex.quote(learner['cybergym_agent_id'])} \
  --env CYBERGYM_TIMEOUT={shlex.quote(str(learner['cybergym_timeout']))} \
{data_volume}  --volume "$RUN/source:/workspace/yeto:ro" \
  --volume "$RUN/miles:/workspace/miles" \
  --volume "$RUN/island-{learner_id}/state:/workspace/state" \
  --volume "$RUN/island-{learner_id}/output:/workspace/output" \
  --volume "$RUN/island-{learner_id}/audit:/workspace/audit" \
  --volume "$HOME/.cache/huggingface:/workspace/hf" \
  --entrypoint bash {shlex.quote(plan['docker_image'])} -lc {shlex.quote(command)}
"""


def _start_island(plan: dict[str, Any], learner_id: int) -> None:
    island = plan["islands"][learner_id]
    for node_id, target in enumerate(island["hosts"]):
        _ssh(
            plan,
            target,
            _node_start_script(plan, learner_id, node_id),
        )


def start(plan_path: str | Path) -> None:
    _, plan = load_plan(plan_path)
    deploy(plan_path)
    for island in plan["islands"]:
        for target in island["hosts"]:
            _ssh(plan, target, _host_setup_script(plan, island["gpus_per_node"]))
    _build_syncer(plan)
    _start_syncer(plan)
    _wait_for_syncer(plan)
    for learner_id in range(LEARNERS):
        _start_island(plan, learner_id)
    print(f"started Miles RL acceptance run {plan['run_id']}")


def status(plan_path: str | Path) -> None:
    _, plan = load_plan(plan_path)
    for learner_id, island in enumerate(plan["islands"]):
        for node_id, target in enumerate(island["hosts"]):
            name = _container_name(plan, learner_id, node_id)
            syncer = ""
            if learner_id == node_id == 0:
                syncer = """if [ -s "$RUN/state/syncer.pid" ] && kill -0 "$(cat "$RUN/state/syncer.pid")" 2>/dev/null; then echo syncer=running; else echo syncer=stopped; fi
"""
            result = _ssh(
                plan,
                target,
                f"""set -u
{_remote_vars(plan)}
echo host={shlex.quote(target)} island={learner_id} node={node_id}
{syncer}if docker inspect {shlex.quote(name)} >/dev/null 2>&1; then
  docker inspect --format 'container={{{{.State.Status}}}} exit={{{{.State.ExitCode}}}}' {shlex.quote(name)}
  docker logs --tail 12 {shlex.quote(name)} 2>&1 || true
else echo container=missing; fi
""",
                capture=True,
                check=False,
            )
            print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)


def kill_learner(plan_path: str | Path, learner_id: int) -> None:
    _, plan = load_plan(plan_path)
    if learner_id not in range(LEARNERS):
        raise HarnessError("--learner-id must be 0 or 1")
    nodes = list(enumerate(plan["islands"][learner_id]["hosts"]))
    for node_id, target in reversed(nodes):
        name = _container_name(plan, learner_id, node_id)
        _ssh(plan, target, f"docker kill {shlex.quote(name)} >/dev/null 2>&1 || true")


def restart_learner(plan_path: str | Path, learner_id: int) -> None:
    _, plan = load_plan(plan_path)
    if learner_id not in range(LEARNERS):
        raise HarnessError("--learner-id must be 0 or 1")
    _wait_for_syncer(plan)
    for node_id, target in enumerate(plan["islands"][learner_id]["hosts"]):
        name = _container_name(plan, learner_id, node_id)
        _ssh(plan, target, f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true")
    _start_island(plan, learner_id)


def kill_syncer(plan_path: str | Path) -> None:
    _, plan = load_plan(plan_path)
    _ssh(
        plan,
        plan["islands"][0]["hosts"][0],
        f"""set -euo pipefail
{_remote_vars(plan)}
test -s "$RUN/state/syncer.pid"
PID="$(cat "$RUN/state/syncer.pid")"
kill -KILL -- -"$PID"
for _ in {{1..50}}; do
  kill -0 "$PID" 2>/dev/null || exit 0
  sleep 0.1
done
echo "syncer did not exit after SIGKILL" >&2
exit 1
""",
    )


def restart_syncer(plan_path: str | Path) -> None:
    _, plan = load_plan(plan_path)
    _start_syncer(plan)
    _wait_for_syncer(plan)


def stop(plan_path: str | Path) -> None:
    _, plan = load_plan(plan_path)
    for learner_id, island in enumerate(plan["islands"]):
        for node_id, target in enumerate(island["hosts"]):
            name = _container_name(plan, learner_id, node_id)
            _ssh(
                plan,
                target,
                f"docker inspect {shlex.quote(name)} >/dev/null 2>&1 && "
                f"docker stop --time 30 {shlex.quote(name)} || true",
            )
    _ssh(
        plan,
        plan["islands"][0]["hosts"][0],
        f"""set -u
{_remote_vars(plan)}
if [ -s "$RUN/state/syncer.pid" ]; then kill -TERM -- -"$(cat "$RUN/state/syncer.pid")" 2>/dev/null || true; fi
""",
    )


def collect(plan_path: str | Path) -> Path:
    plan_file, plan = load_plan(plan_path)
    _require_program("ssh")
    _require_program("rsync")
    artifacts = plan_file.parent / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for learner_id, island in enumerate(plan["islands"]):
        destination = artifacts / f"island-{learner_id}"
        destination.mkdir(parents=True, exist_ok=True)
        for node_id, target in enumerate(island["hosts"]):
            name = _container_name(plan, learner_id, node_id)
            logs = _ssh(
                plan,
                target,
                f"docker logs --timestamps {shlex.quote(name)} 2>&1",
                capture=True,
                check=False,
            )
            _atomic_bytes(
                destination / f"node-{node_id}.log", logs.stdout.encode()
            )
            inspect = _ssh(
                plan,
                target,
                f"docker inspect {shlex.quote(name)}",
                capture=True,
                check=False,
            )
            _atomic_bytes(
                destination / f"node-{node_id}.inspect.json",
                inspect.stdout.encode(),
            )
        head = island["hosts"][0]
        for remote_name in ("state", "output", "audit"):
            local = destination / remote_name
            local.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    "rsync",
                    "-az",
                    "-e",
                    _rsync_shell(plan),
                    f"{head}:{plan['remote_run']}/island-{learner_id}/{remote_name}/",
                    f"{local}/",
                ]
            )
    syncer = artifacts / "syncer"
    syncer.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-az",
            "-e",
            _rsync_shell(plan),
            f"{plan['islands'][0]['hosts'][0]}:{plan['remote_run']}/state/",
            f"{syncer}/",
        ]
    )
    _atomic_bytes(artifacts / "plan.json", plan_file.read_bytes())
    print(f"collected artifacts in {artifacts}")
    return artifacts


def _json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot parse JSON event tape {path}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise HarnessError(f"event tape {path} contains a non-object")
    return values


def _read_f32(path: Path, numel: int) -> tuple[torch.Tensor, bytes]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise HarnessError(f"cannot read audit tensor {path}") from exc
    if len(data) != numel * 4:
        raise HarnessError(f"audit tensor {path} has the wrong length")
    tensor = torch.frombuffer(bytearray(data), dtype=torch.float32).clone()
    if not torch.isfinite(tensor).all().item():
        raise HarnessError(f"audit tensor {path} contains NaN or Inf")
    return tensor, data


def _event_path(artifacts: Path, learner_id: int) -> Path:
    output = artifacts / f"island-{learner_id}" / "output" / "events.jsonl"
    return output if output.exists() else artifacts / f"island-{learner_id}" / "events.jsonl"


def _verify_oracle(plan: dict[str, Any], checkpoint, artifacts: Path) -> str:
    rounds = plan["learner"]["global_rounds"]
    syncer_events = [
        event
        for event in _json_lines(artifacts / "syncer" / "events.jsonl")
        if event.get("fragment") == 0 and "step" in event
    ]
    if [event.get("step") for event in syncer_events] != list(range(1, rounds + 1)):
        raise HarnessError("syncer tape is not one ordered commit per RL round")
    island_events = [
        _json_lines(_event_path(artifacts, learner_id))
        for learner_id in range(LEARNERS)
    ]
    expected_base = None
    specs = None
    identity = None
    for step, sync_event in enumerate(syncer_events, start=1):
        responders = sorted(sync_event.get("responders", []), key=lambda item: item.get("id", -1))
        if (
            sync_event.get("launch_base_version") != step - 1
            or sync_event.get("expected") != [0, 1]
            or sync_event.get("responded") != [0, 1]
            or [item.get("id") for item in responders] != [0, 1]
            or any(
                item.get("base_version") != step - 1
                or item.get("c_steps") != 1
                or item.get("c_tokens") != 1
                or item.get("contribution") != 0.5
                for item in responders
            )
        ):
            raise HarnessError(f"syncer commit v{step} violates fixed-roster f32 AVG")
        bases = []
        base_bytes = []
        deltas = []
        for learner_id in range(LEARNERS):
            audit = artifacts / f"island-{learner_id}" / "audit"
            stem = f"round-{step:08d}"
            try:
                metadata = json.loads((audit / f"{stem}.json").read_text())
                current_specs = tuple(
                    CanonicalTensorSpec(
                        item["name"], tuple(item["shape"]), item["dtype"], item["numel"]
                    )
                    for item in metadata["specs"]
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HarnessError(
                    f"cannot read learner {learner_id} audit for v{step}"
                ) from exc
            numel = sum(spec.numel for spec in current_specs)
            base, raw_base = _read_f32(audit / f"{stem}.base.f32", numel)
            delta, raw_delta = _read_f32(audit / f"{stem}.delta.f32", numel)
            current_identity = (
                metadata.get("base_model_revision"),
                metadata.get("lora_config_hash"),
                metadata.get("layout_hash"),
            )
            if specs is None:
                specs, identity = current_specs, current_identity
            if (
                metadata.get("schema") != 1
                or metadata.get("learner_id") != learner_id
                or metadata.get("base_version") != step - 1
                or metadata.get("target_step") != step
                or metadata.get("base_f32_sha256") != hashlib.sha256(raw_base).hexdigest()
                or metadata.get("delta_f32_sha256") != hashlib.sha256(raw_delta).hexdigest()
                or current_specs != specs
                or current_identity != identity
                or current_identity[0] != plan["learner"]["model_revision"]
                or current_identity[2] != checkpoint.layout_hash
                or sync_event.get("sync/layout_hash") != current_identity[2]
            ):
                raise HarnessError(
                    f"learner {learner_id} audit identity mismatch at v{step}"
                )
            if not any(
                event.get("event") == "rl_local_round"
                and event.get("local_round_id") == step
                and event.get("base_policy_version") == step - 1
                for event in island_events[learner_id]
            ):
                raise HarnessError(f"learner {learner_id} lacks local round v{step}")
            bases.append(base)
            base_bytes.append(raw_base)
            deltas.append(delta)
        if base_bytes[0] != base_bytes[1]:
            raise HarnessError(f"learners did not use the same f32 base at v{step - 1}")
        if expected_base is not None and not torch.equal(bases[0], expected_base):
            raise HarnessError(f"oracle v{step - 1} is not the next exact f32 base")
        merged = torch.zeros_like(bases[0])
        weight = torch.tensor(0.5, dtype=torch.float32)
        for delta in deltas:
            merged.add_(delta * weight)
        expected_base = bases[0] + merged
    if specs is None or identity is None or expected_base is None:
        raise HarnessError("oracle produced no final policy")
    if (
        checkpoint.global_step != rounds
        or len(checkpoint.fragments) != 1
        or checkpoint.fragments[0][0] != rounds
        or checkpoint.ledger
        != {learner_id: (rounds, rounds, rounds) for learner_id in range(LEARNERS)}
        or not torch.equal(checkpoint.fragments[0][1], expected_base)
    ):
        raise HarnessError("authoritative checkpoint differs from the f32 oracle")
    final = canonical_state(
        rounds,
        tensors_from_flat(expected_base, specs),
        base_model_revision=identity[0],
        lora_config_hash=identity[1],
        layout_hash=identity[2],
    )
    final_hash = policy_hash(final)
    for learner_id, events in enumerate(island_events):
        if not any(
            event.get("event") == "rl_policy_apply"
            and event.get("policy_version") == rounds
            and event.get("sync/global_policy_hash") == final_hash
            for event in events
        ):
            raise HarnessError(f"learner {learner_id} did not apply the final policy")
    return final_hash


def _container_succeeded(inspection: Any) -> bool:
    if not isinstance(inspection, list) or not inspection:
        return False
    state = inspection[0].get("State", {})
    return state.get("Status") == "exited" and state.get("ExitCode") == 0


def verify(plan_path: str | Path, export_dir: str | None = None) -> None:
    plan_file, plan = load_plan(plan_path)
    artifacts = plan_file.parent / "artifacts"
    for learner_id, island in enumerate(plan["islands"]):
        for node_id in range(len(island["hosts"])):
            try:
                inspection = json.loads(
                    (
                        artifacts
                        / f"island-{learner_id}"
                        / f"node-{node_id}.inspect.json"
                    ).read_text()
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise HarnessError(
                    f"missing island {learner_id} node {node_id} container status"
                ) from exc
            if not _container_succeeded(inspection):
                raise HarnessError(
                    f"island {learner_id} node {node_id} did not exit successfully"
                )
    from ..export import parse_checkpoint

    checkpoint_path = artifacts / "syncer" / "state.ckpt"
    checkpoint = parse_checkpoint(checkpoint_path)
    final_hash = _verify_oracle(plan, checkpoint, artifacts)
    print(
        f"verified v{checkpoint.global_step} fixed-roster checkpoint and ordered-f32 oracle "
        f"({final_hash})"
    )
    if export_dir:
        from .export import export_rl_checkpoint

        export_rl_checkpoint(
            checkpoint_path,
            Path(export_dir).expanduser(),
            model=plan["learner"]["model"],
            model_revision=plan["learner"]["model_revision"],
            rank=plan["learner"]["lora_r"],
            lora_targets=plan["learner"]["lora_targets"],
            trust_remote_code=plan["learner"]["trust_remote_code"],
        )
        print(f"exported standard PEFT adapter to {Path(export_dir).expanduser()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yeto-rl-ssh", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument(
        "--host",
        action="append",
        required=True,
        help="comma-separated SSH nodes for one island; pass exactly twice",
    )
    prepare_parser.add_argument("--gpus-per-node", type=int, default=1)
    prepare_parser.add_argument("--syncer-address", default=None)
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--output-dir", default=None)
    prepare_parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    prepare_parser.add_argument("--remote-env-file", default=None)
    prepare_parser.add_argument("--ssh-option", action="append", default=[])
    prepare_parser.add_argument("launch_args", nargs=argparse.REMAINDER)

    def plan_command(name: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)
        return command

    plan_command("deploy")
    plan_command("start")
    plan_command("status")
    kill = plan_command("kill-learner")
    kill.add_argument("--learner-id", type=int, required=True)
    restart = plan_command("restart-learner")
    restart.add_argument("--learner-id", type=int, required=True)
    plan_command("kill-syncer")
    plan_command("restart-syncer")
    plan_command("stop")
    plan_command("collect")
    verify_parser = plan_command("verify")
    verify_parser.add_argument("--export-dir", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "kill-learner":
            kill_learner(args.plan, args.learner_id)
        elif args.command == "restart-learner":
            restart_learner(args.plan, args.learner_id)
        elif args.command == "verify":
            verify(args.plan, args.export_dir)
        else:
            {
                "deploy": deploy,
                "start": start,
                "status": status,
                "kill-syncer": kill_syncer,
                "restart-syncer": restart_syncer,
                "stop": stop,
                "collect": collect,
            }[args.command](args.plan)
    except (HarnessError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"yeto-rl-ssh: {exc}") from exc


if __name__ == "__main__":
    main()
