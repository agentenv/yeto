"""Two-host SSH/Tailscale acceptance harness for strict Miles RL.

The harness deliberately reuses the public launch preflight to create one
canonical manifest, but runs the already-provisioned H200 hosts directly.
It never treats an SSH hostname as a public cloud fleet specification.
"""

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

from .checkpoint import validate_rl_final_checkpoint
from .core import canonical_state, tensors_from_flat
from .export import export_rl_checkpoint, specs_from_manifest
from .manifest import (
    MILES_COMMIT,
    MILES_REPOSITORY,
    canonical_json,
    manifest_sha256,
    path_tree_sha256,
    validate_manifest,
)

PLAN_SCHEMA = 1
SYNCER_PORT = 29400
LEARNERS = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_ROOT = ".cache/yeto-rl-ssh"

_RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,47}\Z")
_REMOTE_PATH = re.compile(r"[a-zA-Z0-9._/-]+\Z")
_SSH_TARGET = re.compile(r"(?:[a-zA-Z0-9._-]+@)?[a-zA-Z0-9._-]+\Z")
_TAILSCALE_HOST = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9.-]*\Z")
_DOCKER_DIGEST = re.compile(r".+@sha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HarnessError(RuntimeError):
    pass


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value):
        raise HarnessError(
            "--run-id must be 1-48 letters, digits, dots, underscores, or hyphens"
        )
    return value


def _validate_remote_root(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith(("/", "~"))
        or not _REMOTE_PATH.fullmatch(value)
        or ".." in path.parts
    ):
        raise HarnessError(
            "--remote-root must be a safe path relative to the remote home directory"
        )
    return value.rstrip("/")


def _validate_target(value: str) -> str:
    if not _SSH_TARGET.fullmatch(value):
        raise HarnessError(f"invalid SSH target {value!r}")
    return value


def _target_host(target: str) -> str:
    host = target.rsplit("@", 1)[-1]
    return host


def _validate_syncer_address(value: str) -> tuple[str, int]:
    host, separator, raw_port = value.rpartition(":")
    if not separator or not _TAILSCALE_HOST.fullmatch(host):
        raise HarnessError("--syncer-address must be a Tailscale HOST:PORT")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise HarnessError("--syncer-address has a non-integer port") from exc
    if not 1 <= port <= 65535:
        raise HarnessError("--syncer-address port is outside 1..65535")
    return host, port


def _docker_ref(value: str) -> str:
    if "," in value or "=" in value:
        raise HarnessError("the SSH harness requires one Docker image digest")
    image = value.removeprefix("docker:")
    if not _DOCKER_DIGEST.fullmatch(image):
        raise HarnessError(
            "the SSH harness requires --learner-image docker:REPO@sha256:DIGEST"
        )
    return image


def _has_option(values: Sequence[str], option: str) -> bool:
    return option in values or any(value.startswith(option + "=") for value in values)


def _strip_separator(values: Sequence[str]) -> list[str]:
    values = list(values)
    return values[1:] if values[:1] == ["--"] else values


def _syncer_source_sha256() -> str:
    root = REPO_ROOT / "syncer"
    files = [root / "Cargo.toml", root / "Cargo.lock"]
    files.extend(sorted((root / "src").rglob("*.rs")))
    build_script = root / "build.rs"
    if build_script.is_file():
        files.append(build_script)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _manifest_and_args(namespace) -> tuple[Any, str]:
    from ..cli import build_parser
    from ..launcher import prepare_launch_args

    launch_args = _strip_separator(namespace.launch_args)
    for reserved in ("--gpu", "--training-mode", "--rl-runtime", "--cluster-prefix"):
        if _has_option(launch_args, reserved):
            raise HarnessError(f"{reserved} is set by the two-host harness")
    synthetic_gpu = "ssh:1xh200@island-a,ssh:1xh200@island-b"
    argv = [
        "launch",
        "--gpu",
        synthetic_gpu,
        "--training-mode",
        "rl",
        "--rl-runtime",
        "miles",
        "--cluster-prefix",
        namespace.run_id,
        *launch_args,
    ]
    args = build_parser().parse_args(argv)
    prepare_launch_args(args)
    if args.gpu != synthetic_gpu or args.training_mode != "rl":
        raise HarnessError("the resolved launch arguments escaped strict RL mode")
    return args, shlex.join(argv)


def prepare(namespace) -> Path:
    from ..datasource import kind

    run_id = _validate_run_id(namespace.run_id)
    hosts = [_validate_target(value) for value in namespace.host]
    if len(hosts) != LEARNERS or len(set(hosts)) != LEARNERS:
        raise HarnessError("prepare requires exactly two distinct --host values")
    remote_root = _validate_remote_root(namespace.remote_root)
    syncer_address = namespace.syncer_address or f"{_target_host(hosts[0])}:{SYNCER_PORT}"
    syncer_host, syncer_port = _validate_syncer_address(syncer_address)
    if syncer_port != SYNCER_PORT:
        raise HarnessError(f"strict RL currently uses syncer port {SYNCER_PORT}")
    if namespace.remote_env_file is not None:
        _validate_remote_root(namespace.remote_env_file)

    args, resolved_argv = _manifest_and_args(namespace)
    manifest_text = args.rl_manifest_json
    manifest = validate_manifest(manifest_text, args.run_manifest_sha256)
    image = _docker_ref(args.learner_image)
    data_kind = kind(args.data)
    if data_kind not in {"hf", "local"}:
        raise HarnessError("the SSH harness supports pinned HF or local RL datasets")
    local_data = None
    remote_data_arg = manifest["dataset"]["identifier"]
    if data_kind == "local":
        local_data = str(Path(args.data).expanduser().resolve())
        source = Path(local_data)
        suffix = source.suffix if source.is_file() else ""
        remote_data_arg = f"/workspace/data/dataset{suffix}" if suffix else "/workspace/data"

    local_run = (
        Path(namespace.output_dir).expanduser()
        if namespace.output_dir
        else Path.home() / ".yeto" / "ssh-runs" / run_id
    ).resolve()
    plan_path = local_run / "plan.json"
    if plan_path.exists():
        raise HarnessError(f"plan already exists at {plan_path}")

    plan = {
        "schema": PLAN_SCHEMA,
        "run_id": run_id,
        "created_unix_ns": time.time_ns(),
        "repo_root": str(REPO_ROOT),
        "local_run_dir": str(local_run),
        "hosts": hosts,
        "ssh_options": list(namespace.ssh_option),
        "syncer_address": syncer_address,
        "syncer_host": syncer_host,
        "syncer_port": syncer_port,
        "remote_root": remote_root,
        "remote_run": f"{remote_root}/{run_id}",
        "remote_env_file": namespace.remote_env_file,
        "docker_image": image,
        "manifest_sha256": args.run_manifest_sha256,
        "source_sha256": args.source_sha256,
        "syncer_source_sha256": _syncer_source_sha256(),
        "data": {
            "kind": data_kind,
            "local_path": local_data,
            "learner_arg": remote_data_arg,
        },
        "learner": {
            "model": manifest["base_model"]["identifier"],
            "model_revision": manifest["base_model"]["revision"],
            "data_revision": manifest["dataset"]["revision"],
            "reward_function": args.reward_function,
            "reward_sha256": args.reward_sha256,
            "generate_function": args.rl_generate_function,
            "generate_sha256": args.rl_generate_sha256,
            "global_rounds": args.rl_global_rounds,
            "groups_per_round": args.rl_groups_per_island_round,
            "samples_per_group": args.rl_samples_per_group,
            "optimizer_steps": args.rl_local_optimizer_steps,
            "round_timeout_s": args.rl_round_timeout_s,
            "lora_r": args.lora_r,
            "lora_targets": args.lora_targets,
            "inner_lr": args.inner_lr,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "wan_streams": args.wan_streams,
            "trust_remote_code": args.trust_remote_code,
        },
        "syncer": {
            "grace_ms": args.grace_ms,
            "grace_gamma": args.grace_gamma,
            "grace_tau": args.grace_tau,
            "pipeline": args.pipeline,
            "sync_interval_steps": args.sync_interval_steps,
            "delta_correction": args.delta_correction,
            "outer_lr": args.outer_lr,
            "outer_momentum": args.outer_momentum,
            "total_steps": args.total_steps,
        },
        "resolved_launch_command": resolved_argv,
    }
    _atomic_bytes(local_run / "rl-manifest.json", manifest_text.encode("utf-8"))
    _atomic_bytes(
        plan_path,
        (json.dumps(plan, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(f"prepared {plan_path}")
    print(f"manifest {args.run_manifest_sha256}")
    print(f"syncer {syncer_address} on {hosts[0]}")
    print(f"next: yeto-rl-ssh start --plan {shlex.quote(str(plan_path))}")
    return plan_path


def load_plan(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    plan_path = Path(path).expanduser().resolve()
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read harness plan {plan_path}") from exc
    if plan.get("schema") != PLAN_SCHEMA:
        raise HarnessError("unsupported SSH harness plan schema")
    _validate_run_id(plan.get("run_id", ""))
    _validate_remote_root(plan.get("remote_run", ""))
    hosts = plan.get("hosts")
    if not isinstance(hosts, list) or len(hosts) != LEARNERS:
        raise HarnessError("plan does not contain the fixed two-host roster")
    for host in hosts:
        _validate_target(host)
    image = _docker_ref(plan.get("docker_image", ""))
    _validate_syncer_address(plan.get("syncer_address", ""))
    manifest_path = plan_path.parent / "rl-manifest.json"
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HarnessError(f"cannot read {manifest_path}") from exc
    manifest = validate_manifest(manifest_text, plan.get("manifest_sha256"))
    learner = plan.get("learner") or {}
    workload = manifest["workload"]
    custom_generate = manifest["generation"]["custom_generate"]
    expected = {
        "model": manifest["base_model"]["identifier"],
        "model_revision": manifest["base_model"]["revision"],
        "data_revision": manifest["dataset"]["revision"],
        "reward_function": manifest["reward"]["callable"],
        "reward_sha256": manifest["reward"]["source_sha256"],
        "generate_function": custom_generate["callable"] if custom_generate else None,
        "generate_sha256": custom_generate["source_sha256"] if custom_generate else None,
        "global_rounds": workload["global_rounds"],
        "groups_per_round": workload["groups_per_island_round"],
        "samples_per_group": workload["samples_per_group"],
        "optimizer_steps": workload["local_optimizer_steps"],
        "lora_r": manifest["lora"]["rank"],
        "lora_targets": manifest["lora"]["targets"],
        "inner_lr": manifest["optimizer"]["learning_rate"],
        "seq_len": manifest["generation"]["max_context_length"],
        "seed": manifest["generation"]["seed"],
        "trust_remote_code": manifest["base_model"]["trust_remote_code"],
    }
    syncer = plan.get("syncer") or {}
    strict_sync = {
        "pipeline": 1,
        "sync_interval_steps": 0.0,
        "delta_correction": "none",
        "outer_lr": 1.0,
        "outer_momentum": 0.0,
        "total_steps": workload["global_rounds"],
    }
    if (
        not _SHA256.fullmatch(str(plan.get("source_sha256", "")))
        or not _SHA256.fullmatch(str(plan.get("syncer_source_sha256", "")))
        or manifest["run_id"] != plan["run_id"]
        or workload["learners"] != LEARNERS
        or manifest["yeto_source_sha256"] != plan.get("source_sha256")
        or _docker_ref(manifest["learner_image"]) != image
        or any(learner.get(key) != value for key, value in expected.items())
        or any(syncer.get(key) != value for key, value in strict_sync.items())
    ):
        raise HarnessError("SSH harness plan does not match its canonical manifest")
    return plan_path, plan, manifest_text


def _require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise HarnessError(f"{name} is required on the controller")


def _run(
    command: Sequence[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(command), flush=True)
    return subprocess.run(
        list(command),
        check=check,
        text=True,
        capture_output=capture,
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


def _attest_local(plan: dict[str, Any], manifest_text: str) -> None:
    from ..provenance import python_spec_sha256, verify_source_tree_sha256

    verify_source_tree_sha256(plan["source_sha256"])
    if _syncer_source_sha256() != plan.get("syncer_source_sha256"):
        raise HarnessError("syncer source changed after the plan was prepared")
    manifest = validate_manifest(manifest_text, plan["manifest_sha256"])
    reward = manifest["reward"]
    if python_spec_sha256(reward["callable"], base_dir=REPO_ROOT) != reward["source_sha256"]:
        raise HarnessError("reward source changed after the plan was prepared")
    custom = manifest["generation"]["custom_generate"]
    if custom and python_spec_sha256(custom["callable"], base_dir=REPO_ROOT) != custom["source_sha256"]:
        raise HarnessError("generate source changed after the plan was prepared")
    data = plan["data"]
    if (
        data["kind"] == "local"
        and path_tree_sha256(data["local_path"])
        != manifest["dataset"]["content_sha256"]
    ):
        raise HarnessError("local dataset changed after the plan was prepared")


def deploy(plan_path: str | Path) -> None:
    plan_file, plan, manifest_text = load_plan(plan_path)
    _require_program("ssh")
    _require_program("rsync")
    _attest_local(plan, manifest_text)
    remote_run = plan["remote_run"]
    manifest_sha = plan["manifest_sha256"]
    excludes = (
        ".git/",
        ".venv/",
        ".worktree/",
        ".env",
        ".env.*",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        "syncer/target/",
        "checkpoints/",
        "runs/",
        "logs/",
        "wandb/",
        "build/",
        "dist/",
    )
    for target in plan["hosts"]:
        probe = _ssh(
            plan,
            target,
            f"""set -euo pipefail
{_remote_vars(plan)}
if [ -f "$RUN/control/manifest.sha256" ]; then
  test "$(cat "$RUN/control/manifest.sha256")" = {shlex.quote(manifest_sha)}
  echo deployed
else
  echo pending
fi
""",
            capture=True,
        )
        if probe.stdout.strip() == "deployed":
            print(f"{target}: matching deployment already present")
            continue
        initialize = f"""set -euo pipefail
{_remote_vars(plan)}
if [ -e "$RUN" ] && [ -n "$(find "$RUN" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  test -f "$RUN/control/deploying.sha256"
  test "$(cat "$RUN/control/deploying.sha256")" = {shlex.quote(manifest_sha)}
fi
mkdir -p "$RUN"/{{control,source,miles,state,cache,audit}}
printf '%s\\n' {shlex.quote(manifest_sha)} > "$RUN/control/deploying.sha256"
"""
        _ssh(plan, target, initialize)
        source_command = [
            "rsync",
            "-az",
            "--delete",
            *[item for pattern in excludes for item in ("--exclude", pattern)],
            "-e",
            _rsync_shell(plan),
            f"{REPO_ROOT}/",
            f"{target}:{remote_run}/source/",
        ]
        _run(source_command)
        _run(
            [
                "rsync",
                "-az",
                "-e",
                _rsync_shell(plan),
                str(plan_file.parent / "rl-manifest.json"),
                f"{target}:{remote_run}/control/rl-manifest.json",
            ]
        )
        data = plan["data"]
        if data["kind"] == "local":
            local = Path(data["local_path"])
            if local.is_dir():
                data_source = f"{local}/"
                data_destination = f"{target}:{remote_run}/data/"
                _ssh(plan, target, f'{_remote_vars(plan)}\nmkdir -p "$RUN/data"')
            else:
                suffix = local.suffix
                data_source = str(local)
                data_destination = f"{target}:{remote_run}/data/dataset{suffix}"
                _ssh(plan, target, f'{_remote_vars(plan)}\nmkdir -p "$RUN/data"')
            _run(
                [
                    "rsync",
                    "-az",
                    "--delete" if local.is_dir() else "--checksum",
                    "-e",
                    _rsync_shell(plan),
                    data_source,
                    data_destination,
                ]
            )
        finalize = f"""set -euo pipefail
{_remote_vars(plan)}
test "$(sha256sum "$RUN/control/rl-manifest.json" | awk '{{print $1}}')" = {shlex.quote(manifest_sha)}
mv "$RUN/control/deploying.sha256" "$RUN/control/manifest.sha256"
"""
        _ssh(plan, target, finalize)
    print(f"deployed {plan['run_id']} to {', '.join(plan['hosts'])}")


def _host_setup_script(plan: dict[str, Any]) -> str:
    image = shlex.quote(plan["docker_image"])
    return f"""set -euo pipefail
{_remote_vars(plan)}
test "$(cat "$RUN/control/manifest.sha256")" = {shlex.quote(plan['manifest_sha256'])}
command -v docker >/dev/null
command -v git >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | grep -qi H200
docker info >/dev/null
mkdir -p "$HOME/.cache/huggingface"
docker pull {image}
if [ -d "$RUN/miles/.git" ]; then
  test "$(git -C "$RUN/miles" config --get remote.origin.url | sed 's#\\.git$##;s#/$##')" = {shlex.quote(MILES_REPOSITORY)}
  test "$(git -C "$RUN/miles" rev-parse HEAD)" = {shlex.quote(MILES_COMMIT)}
  test -z "$(git -C "$RUN/miles" status --porcelain --untracked-files=all)"
else
  test -z "$(find "$RUN/miles" -mindepth 1 -maxdepth 1 -print -quit)"
  git clone --no-checkout {shlex.quote(MILES_REPOSITORY)} "$RUN/miles"
  git -C "$RUN/miles" fetch --depth 1 origin {shlex.quote(MILES_COMMIT)}
  git -C "$RUN/miles" checkout --detach {shlex.quote(MILES_COMMIT)}
fi
test "$(git -C "$RUN/miles" rev-parse HEAD)" = {shlex.quote(MILES_COMMIT)}
test -z "$(git -C "$RUN/miles" status --porcelain --untracked-files=all)"
"""


def _build_syncer(plan: dict[str, Any]) -> None:
    verifier = """import hashlib
from pathlib import Path
root = Path('/workspace/yeto/syncer')
files = [root / 'Cargo.toml', root / 'Cargo.lock']
files.extend(sorted((root / 'src').rglob('*.rs')))
if (root / 'build.rs').is_file():
    files.append(root / 'build.rs')
digest = hashlib.sha256()
for path in files:
    relative = path.relative_to(root).as_posix().encode('utf-8')
    data = path.read_bytes()
    digest.update(len(relative).to_bytes(8, 'big'))
    digest.update(relative)
    digest.update(len(data).to_bytes(8, 'big'))
    digest.update(data)
print(digest.hexdigest())
"""
    script = f"""set -euo pipefail
{_remote_vars(plan)}
ACTUAL="$(docker run --rm --volume "$RUN/source:/workspace/yeto:ro" --entrypoint python3 {shlex.quote(plan['docker_image'])} -c {shlex.quote(verifier)})"
test "$ACTUAL" = {shlex.quote(plan['syncer_source_sha256'])} || {{ echo 'remote syncer source identity mismatch' >&2; exit 1; }}
CARGO="$(command -v cargo || true)"
if [ -z "$CARGO" ] && [ -x "$HOME/.cargo/bin/cargo" ]; then CARGO="$HOME/.cargo/bin/cargo"; fi
test -n "$CARGO" || {{ echo 'cargo is required on the syncer host' >&2; exit 1; }}
command -v cc >/dev/null || {{ echo 'a C compiler is required on the syncer host' >&2; exit 1; }}
"$CARGO" build --release --locked --manifest-path "$RUN/source/syncer/Cargo.toml"
install -m 0755 "$RUN/source/syncer/target/release/yeto-syncer" "$RUN/state/yeto-syncer"
"""
    _ssh(plan, plan["hosts"][0], script)


def _syncer_argv(plan: dict[str, Any]) -> list[str]:
    sync = plan["syncer"]
    return [
        "$RUN/state/yeto-syncer",
        "--port", str(plan["syncer_port"]),
        "--learners", str(LEARNERS),
        "--quorum", str(LEARNERS),
        "--grace-ms", str(sync["grace_ms"]),
        "--grace-gamma", str(sync["grace_gamma"]),
        "--grace-tau", str(sync["grace_tau"]),
        "--pipeline", str(sync["pipeline"]),
        "--sync-interval-steps", str(sync["sync_interval_steps"]),
        "--delta-correction", str(sync["delta_correction"]),
        "--total-steps", str(sync["total_steps"]),
        "--outer-lr", str(sync["outer_lr"]),
        "--outer-momentum", str(sync["outer_momentum"]),
        "--checkpoint-path", "$RUN/state/yeto-state.ckpt",
        "--resume",
        "--mark-final-checkpoint",
        "--event-tape", "$RUN/state/syncer-events.jsonl",
        "--checkpoint-every", "1",
        "--rl-strict-avg",
        "--run-manifest-sha256", plan["manifest_sha256"],
        "--rl-round-timeout-s", str(plan["learner"]["round_timeout_s"]),
    ]


def _shell_join_with_run(values: Sequence[str]) -> str:
    return " ".join(value if value.startswith("$RUN/") else shlex.quote(value) for value in values)


def _start_syncer(plan: dict[str, Any]) -> None:
    command = _shell_join_with_run(_syncer_argv(plan))
    script = f"""set -euo pipefail
{_remote_vars(plan)}
PID_FILE="$RUN/state/syncer.pid"
if [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "syncer already running pid=$(cat "$PID_FILE")"
  exit 0
fi
nohup {command} >> "$RUN/state/syncer.log" 2>&1 &
PID=$!
printf '%s\\n' "$PID" > "$PID_FILE"
sleep 1
kill -0 "$PID"
echo "syncer started pid=$PID"
"""
    _ssh(plan, plan["hosts"][0], script)


def _wait_for_syncer(plan: dict[str, Any], timeout_s: int = 120) -> None:
    host, port = _validate_syncer_address(plan["syncer_address"])
    quoted_host = shlex.quote(host)
    for target in plan["hosts"]:
        script = f"""set -euo pipefail
deadline=$((SECONDS + {timeout_s}))
until timeout 2 bash -c 'exec 3<>/dev/tcp/{quoted_host}/{port}' 2>/dev/null; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo 'syncer is not reachable at {shlex.quote(plan['syncer_address'])}' >&2
    exit 1
  fi
  sleep 2
done
"""
        _ssh(plan, target, script)


def _learner_argv(plan: dict[str, Any], learner_id: int) -> list[str]:
    learner = plan["learner"]
    values = [
        "python3", "-m", "yeto.rl.learner",
        "--model", learner["model"],
        "--model-revision", learner["model_revision"],
        "--data", plan["data"]["learner_arg"],
        "--syncer", plan["syncer_address"],
        "--learner-id", str(learner_id),
        "--num-learners", str(LEARNERS),
        "--manifest-sha256", plan["manifest_sha256"],
        "--reward-function", learner["reward_function"],
        "--reward-sha256", learner["reward_sha256"],
        "--global-rounds", str(learner["global_rounds"]),
        "--groups-per-round", str(learner["groups_per_round"]),
        "--samples-per-group", str(learner["samples_per_group"]),
        "--optimizer-steps", str(learner["optimizer_steps"]),
        "--round-timeout-s", str(learner["round_timeout_s"]),
        "--lora-r", str(learner["lora_r"]),
        "--lora-targets", learner["lora_targets"],
        "--inner-lr", str(learner["inner_lr"]),
        "--seq-len", str(learner["seq_len"]),
        "--seed", str(learner["seed"]),
        "--wan-streams", str(learner["wan_streams"]),
        "--cache-dir", "/workspace/cache",
        "--audit-dir", "/workspace/audit",
        "--miles-root", "/workspace/miles",
        "--source-sha256", plan["source_sha256"],
    ]
    if learner["data_revision"]:
        values.extend(("--data-revision", learner["data_revision"]))
    if learner["generate_function"]:
        values.extend(("--generate-function", learner["generate_function"]))
        values.extend(("--generate-sha256", learner["generate_sha256"]))
    if learner["trust_remote_code"]:
        values.append("--trust-remote-code")
    return values


def _container_name(plan: dict[str, Any], learner_id: int) -> str:
    return f"yeto-rl-{plan['run_id']}-l{learner_id}"


def _start_learner(plan: dict[str, Any], learner_id: int, *, replace: bool = False) -> None:
    if learner_id not in range(LEARNERS):
        raise HarnessError("--learner-id must be 0 or 1")
    target = plan["hosts"][learner_id]
    name = _container_name(plan, learner_id)
    learner_command = shlex.join(_learner_argv(plan, learner_id))
    env_file = ""
    if plan.get("remote_env_file"):
        env_file = f' --env-file "$HOME/{plan["remote_env_file"]}"'
    data_mount = ""
    if plan["data"]["kind"] == "local":
        data_mount = ' --volume "$RUN/data:/workspace/data:ro"'
    replace_script = f'docker rm -f {shlex.quote(name)} >/dev/null\n' if replace else ""
    script = f"""set -euo pipefail
{_remote_vars(plan)}
if docker inspect {shlex.quote(name)} >/dev/null 2>&1; then
  STATUS="$(docker inspect --format '{{{{.State.Status}}}}' {shlex.quote(name)})"
  if [ "$STATUS" = running ] && [ {str(replace).lower()} = false ]; then
    echo {shlex.quote(name)} already running
    exit 0
  fi
  if [ {str(replace).lower()} = false ]; then
    echo "container {name} exists with status $STATUS; use restart-learner" >&2
    exit 1
  fi
fi
{replace_script}mkdir -p "$RUN"/{{cache,audit}}
docker run --detach \
  --name {shlex.quote(name)} \
  --gpus device=0 \
  --network host \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env PYTHONUNBUFFERED=1 \
  --env HF_HOME=/workspace/hf \
  --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  --env PYTHONPATH=/workspace/yeto:/workspace/miles{env_file} \
  --volume "$RUN/source:/workspace/yeto:ro" \
  --volume "$RUN/miles:/workspace/miles:ro" \
  --volume "$RUN/control:/workspace/control:ro" \
  --volume "$RUN/cache:/workspace/cache" \
  --volume "$RUN/audit:/workspace/audit" \
  --volume "$HOME/.cache/huggingface:/workspace/hf"{data_mount} \
  --entrypoint bash \
  {shlex.quote(plan['docker_image'])} \
  -lc {shlex.quote('set -euo pipefail; export YETO_RL_MANIFEST="$(cat /workspace/control/rl-manifest.json)"; cd /workspace/yeto; exec ' + learner_command)}
"""
    _ssh(plan, target, script)


def start(plan_path: str | Path) -> None:
    _, plan, _ = load_plan(plan_path)
    deploy(plan_path)
    for target in plan["hosts"]:
        _ssh(plan, target, _host_setup_script(plan))
    _build_syncer(plan)
    _start_syncer(plan)
    _wait_for_syncer(plan)
    for learner_id in range(LEARNERS):
        _start_learner(plan, learner_id)
    print(f"started strict RL run {plan['run_id']}")


def status(plan_path: str | Path) -> None:
    _, plan, _ = load_plan(plan_path)
    for learner_id, target in enumerate(plan["hosts"]):
        name = _container_name(plan, learner_id)
        syncer = ""
        if learner_id == 0:
            syncer = """if [ -s "$RUN/state/syncer.pid" ] && kill -0 "$(cat "$RUN/state/syncer.pid")" 2>/dev/null; then
  echo "syncer=running pid=$(cat "$RUN/state/syncer.pid")"
else
  echo "syncer=stopped"
fi
for suffix in final fatal; do [ ! -f "$RUN/state/yeto-state.ckpt.$suffix" ] || echo "syncer_$suffix=present"; done
"""
        script = f"""set -u
{_remote_vars(plan)}
echo "host={target} learner={learner_id}"
{syncer}if docker inspect {shlex.quote(name)} >/dev/null 2>&1; then
  docker inspect --format 'container={{{{.State.Status}}}} exit={{{{.State.ExitCode}}}} started={{{{.State.StartedAt}}}} finished={{{{.State.FinishedAt}}}}' {shlex.quote(name)}
  docker logs --tail 12 {shlex.quote(name)} 2>&1 || true
else
  echo "container=missing"
fi
"""
        result = _ssh(plan, target, script, capture=True, check=False)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)


def kill_learner(plan_path: str | Path, learner_id: int) -> None:
    _, plan, _ = load_plan(plan_path)
    if learner_id not in range(LEARNERS):
        raise HarnessError("--learner-id must be 0 or 1")
    name = _container_name(plan, learner_id)
    _ssh(plan, plan["hosts"][learner_id], f"docker kill {shlex.quote(name)}")


def restart_learner(plan_path: str | Path, learner_id: int) -> None:
    _, plan, _ = load_plan(plan_path)
    _wait_for_syncer(plan)
    _start_learner(plan, learner_id, replace=True)


def kill_syncer(plan_path: str | Path) -> None:
    _, plan, _ = load_plan(plan_path)
    script = f"""set -euo pipefail
{_remote_vars(plan)}
test -s "$RUN/state/syncer.pid"
PID="$(cat "$RUN/state/syncer.pid")"
kill -KILL "$PID"
for _ in {{1..50}}; do
  kill -0 "$PID" 2>/dev/null || exit 0
  sleep 0.1
done
echo "syncer pid $PID did not disappear after SIGKILL" >&2
exit 1
"""
    _ssh(plan, plan["hosts"][0], script)


def restart_syncer(plan_path: str | Path) -> None:
    _, plan, _ = load_plan(plan_path)
    _start_syncer(plan)
    _wait_for_syncer(plan)


def stop(plan_path: str | Path) -> None:
    _, plan, _ = load_plan(plan_path)
    for learner_id, target in enumerate(plan["hosts"]):
        name = _container_name(plan, learner_id)
        _ssh(
            plan,
            target,
            f"docker inspect {shlex.quote(name)} >/dev/null 2>&1 && "
            f"docker stop --time 30 {shlex.quote(name)} || true",
        )
    script = f"""set -u
{_remote_vars(plan)}
if [ -s "$RUN/state/syncer.pid" ]; then kill -TERM "$(cat "$RUN/state/syncer.pid")" 2>/dev/null || true; fi
"""
    _ssh(plan, plan["hosts"][0], script)


def collect(plan_path: str | Path) -> Path:
    plan_file, plan, _ = load_plan(plan_path)
    _require_program("ssh")
    _require_program("rsync")
    artifacts = plan_file.parent / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for learner_id, target in enumerate(plan["hosts"]):
        destination = artifacts / f"learner-{learner_id}"
        destination.mkdir(parents=True, exist_ok=True)
        name = _container_name(plan, learner_id)
        logs = _ssh(
            plan,
            target,
            f"docker logs --timestamps {shlex.quote(name)} 2>&1",
            capture=True,
            check=False,
        )
        _atomic_bytes(destination / "container.log", logs.stdout.encode("utf-8"))
        for remote_name in ("audit", "cache"):
            local = destination / remote_name
            local.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    "rsync", "-az", "-e", _rsync_shell(plan),
                    f"{target}:{plan['remote_run']}/{remote_name}/",
                    f"{local}/",
                ]
            )
    syncer = artifacts / "syncer"
    syncer.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync", "-az", "-e", _rsync_shell(plan),
            f"{plan['hosts'][0]}:{plan['remote_run']}/state/",
            f"{syncer}/",
        ]
    )
    _atomic_bytes(artifacts / "rl-manifest.json", (plan_file.parent / "rl-manifest.json").read_bytes())
    print(f"collected artifacts in {artifacts}")
    return artifacts


def _json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line]
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


def _verify_oracle(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    checkpoint,
    artifacts: Path,
) -> None:
    rounds = manifest["workload"]["global_rounds"]
    learners = manifest["workload"]["learners"]
    specs = specs_from_manifest(manifest)
    numel = sum(spec.numel for spec in specs)
    syncer_events = _json_lines(artifacts / "syncer" / "syncer-events.jsonl")
    if [event.get("committed_version") for event in syncer_events] != list(range(1, rounds + 1)):
        raise HarnessError("syncer event tape is not one ordered commit per global round")

    island_events: list[list[dict[str, Any]]] = []
    for learner_id in range(learners):
        events = _json_lines(artifacts / f"learner-{learner_id}" / "cache" / "events.jsonl")
        if [event.get("committed_version") for event in events] != list(range(1, rounds + 1)):
            raise HarnessError(f"learner {learner_id} lacks one apply event per round")
        island_events.append(events)

    expected_final: torch.Tensor | None = None
    for version in range(1, rounds + 1):
        sync_event = syncer_events[version - 1]
        if (
            sync_event.get("run_manifest_sha256") != plan["manifest_sha256"]
            or sync_event.get("fixed_roster") != learners
            or sync_event.get("responded") != list(range(learners))
            or sync_event.get("checkpoint_committed") is not True
        ):
            raise HarnessError(f"syncer commit v{version} violates the fixed-roster identity")
        digest_by_id = {
            item.get("id"): item.get("delta_sha256")
            for item in sync_event.get("delta_digests", [])
        }
        bases: list[torch.Tensor] = []
        base_bytes: list[bytes] = []
        deltas: list[torch.Tensor] = []
        for learner_id in range(learners):
            audit = artifacts / f"learner-{learner_id}" / "audit"
            stem = f"round-{version:08d}"
            try:
                metadata = json.loads((audit / f"{stem}.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HarnessError(f"cannot read learner {learner_id} audit for v{version}") from exc
            base, raw_base = _read_f32(audit / f"{stem}.base.f32", numel)
            delta, raw_delta = _read_f32(audit / f"{stem}.delta.f32", numel)
            delta_sha = hashlib.sha256(raw_delta).hexdigest()
            island_event = island_events[learner_id][version - 1]
            if (
                metadata.get("schema") != 1
                or metadata.get("learner_id") != learner_id
                or metadata.get("base_version") != version - 1
                or metadata.get("target_step") != version
                or metadata.get("run_manifest_sha256") != plan["manifest_sha256"]
                or metadata.get("layout_fingerprint") != checkpoint.layout_fingerprint
                or metadata.get("numel") != numel
                or metadata.get("base_f32_sha256") != hashlib.sha256(raw_base).hexdigest()
                or metadata.get("delta_sha256") != delta_sha
                or digest_by_id.get(learner_id) != delta_sha
                or island_event.get("delta_sha256") != delta_sha
            ):
                raise HarnessError(f"learner {learner_id} audit identity mismatch at v{version}")
            base_state = canonical_state(
                version - 1,
                tensors_from_flat(base, specs),
                expected_specs=specs,
                expected_layout_fingerprint=checkpoint.layout_fingerprint,
            )
            if metadata.get("base_policy_hash") != base_state.policy_hash:
                raise HarnessError(f"learner {learner_id} base hash mismatch at v{version}")
            workload = manifest["workload"]
            if (
                island_event.get("run_manifest_sha256") != plan["manifest_sha256"]
                or island_event.get("learner_id") != learner_id
                or island_event.get("base_version") != version - 1
                or island_event.get("base_policy_hash") != base_state.policy_hash
                or island_event.get("groups") != workload.get("groups_per_island_round")
                or island_event.get("samples_per_group") != workload.get("samples_per_group")
                or island_event.get("trajectories")
                != workload.get("groups_per_island_round") * workload.get("samples_per_group")
                or island_event.get("optimizer_steps") != workload.get("local_optimizer_steps")
                or island_event.get("rollout_identity_set")
                != [{"version": version - 1, "policy_hash": base_state.policy_hash}]
            ):
                raise HarnessError(f"learner {learner_id} workload identity mismatch at v{version}")
            bases.append(base)
            base_bytes.append(raw_base)
            deltas.append(delta)
        if any(value != base_bytes[0] for value in base_bytes[1:]):
            raise HarnessError(f"learners did not train from identical f32 base v{version - 1}")

        weight = torch.tensor(1.0 / learners, dtype=torch.float32)
        outer_gradient = torch.zeros(numel, dtype=torch.float32)
        for delta in deltas:  # learner-id order, matching the syncer's BTreeMap
            outer_gradient.add_(torch.mul(torch.neg(delta), weight))
        expected = torch.sub(bases[0], outer_gradient)
        expected_state = canonical_state(
            version,
            tensors_from_flat(expected, specs),
            expected_specs=specs,
            expected_layout_fingerprint=checkpoint.layout_fingerprint,
        )
        applied = {"version": version, "policy_hash": expected_state.policy_hash}
        for learner_id, events in enumerate(island_events):
            event = events[version - 1]
            if (
                event.get("committed_policy_hash") != expected_state.policy_hash
                or event.get("trainer_applied_identity") != applied
                or event.get("rollout_applied_identity") != applied
            ):
                raise HarnessError(
                    f"learner {learner_id} applied identity does not match the f32 oracle at v{version}"
                )
        if sync_event.get("policy_sha256") != expected_state.policy_hash:
            raise HarnessError(f"syncer policy does not match the f32 oracle at v{version}")
        if version < rounds:
            next_base_path = (
                artifacts / "learner-0" / "audit" / f"round-{version + 1:08d}.base.f32"
            )
            if next_base_path.read_bytes() != expected.numpy().astype("<f4", copy=False).tobytes():
                raise HarnessError(f"oracle v{version} is not the next round's exact f32 base")
        else:
            expected_final = expected

    if expected_final is None:
        raise HarnessError("oracle produced no final policy")
    final = torch.tensor(checkpoint.fragments[0], dtype=torch.float32)
    if not torch.equal(expected_final, final):
        raise HarnessError("authoritative checkpoint differs from the ordered-f32 oracle")


def verify(plan_path: str | Path, export_dir: str | None = None) -> None:
    plan_file, plan, manifest_text = load_plan(plan_path)
    artifacts = plan_file.parent / "artifacts"
    manifest = validate_manifest(manifest_text, plan["manifest_sha256"])
    if manifest_sha256(canonical_json(manifest)) != plan["manifest_sha256"]:
        raise HarnessError("manifest hash changed during verification")
    checkpoint_path = artifacts / "syncer" / "yeto-state.ckpt"
    checkpoint = validate_rl_final_checkpoint(checkpoint_path, plan["manifest_sha256"])
    if (
        checkpoint.global_step != manifest["workload"]["global_rounds"]
        or checkpoint.roster_size != LEARNERS
        or checkpoint.layout_fingerprint != manifest["canonical_lora"]["layout_fingerprint"]
    ):
        raise HarnessError("final checkpoint does not match the planned workload")
    _verify_oracle(plan, manifest, checkpoint, artifacts)
    print(
        f"verified v{checkpoint.global_step} fixed-roster checkpoint, final ACK marker, "
        f"and ordered-f32 oracle ({checkpoint.policy_sha256})"
    )
    if export_dir:
        provenance = export_rl_checkpoint(
            checkpoint_path,
            manifest_text,
            Path(export_dir).expanduser(),
        )
        print(
            "exported and clean-process reloaded standard PEFT adapter "
            f"({provenance['policy_sha256']})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yeto-rl-ssh", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="resolve and persist one canonical run plan")
    prepare_parser.add_argument("--host", action="append", required=True, help="SSH target; pass twice")
    prepare_parser.add_argument(
        "--syncer-address",
        default=None,
        help="Tailscale HOST:29400 reachable from both H200s; defaults to the first SSH host",
    )
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--output-dir", default=None)
    prepare_parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    prepare_parser.add_argument(
        "--remote-env-file",
        default=None,
        help="optional path relative to remote $HOME, read by docker on both hosts",
    )
    prepare_parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="one raw ssh option token; prefer ~/.ssh/config",
    )
    prepare_parser.add_argument(
        "launch_args",
        nargs=argparse.REMAINDER,
        help="Yeto launch model/data/workload flags after --",
    )

    def plan_command(name: str, help_text: str) -> argparse.ArgumentParser:
        value = commands.add_parser(name, help=help_text)
        value.add_argument("--plan", required=True)
        return value

    plan_command("deploy", "copy the attested source, manifest, and optional local data")
    plan_command("start", "deploy and start the syncer plus both learners")
    plan_command("status", "show remote process state and recent learner logs")
    kill_l = plan_command("kill-learner", "SIGKILL one learner container for recovery testing")
    kill_l.add_argument("--learner-id", type=int, required=True)
    restart_l = plan_command("restart-learner", "restart one logical learner with the same cache")
    restart_l.add_argument("--learner-id", type=int, required=True)
    plan_command("kill-syncer", "SIGKILL the syncer for checkpoint recovery testing")
    plan_command("restart-syncer", "resume the syncer from its authoritative checkpoint")
    plan_command("stop", "stop both learner containers and the syncer")
    plan_command("collect", "collect logs, audit tensors, and authoritative state")
    verify_parser = plan_command("verify", "verify final identity, f32 oracle, and optional PEFT export")
    verify_parser.add_argument("--export-dir", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "deploy":
            deploy(args.plan)
        elif args.command == "start":
            start(args.plan)
        elif args.command == "status":
            status(args.plan)
        elif args.command == "kill-learner":
            kill_learner(args.plan, args.learner_id)
        elif args.command == "restart-learner":
            restart_learner(args.plan, args.learner_id)
        elif args.command == "kill-syncer":
            kill_syncer(args.plan)
        elif args.command == "restart-syncer":
            restart_syncer(args.plan)
        elif args.command == "stop":
            stop(args.plan)
        elif args.command == "collect":
            collect(args.plan)
        elif args.command == "verify":
            verify(args.plan, args.export_dir)
        else:  # pragma: no cover - argparse owns command choices
            raise AssertionError(args.command)
    except (HarnessError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"yeto-rl-ssh: {exc}") from exc


if __name__ == "__main__":
    main()
