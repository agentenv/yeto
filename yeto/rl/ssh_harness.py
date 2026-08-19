"""Direct SSH acceptance harness for the current Miles RL boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from . import (
    CODEX_APP_SERVER_PROTOCOL_REVISION,
    CODEX_APP_SERVER_SCHEMA_SHA256,
    CODEX_BASE_INSTRUCTIONS_SHA256,
    CODEX_CLI_VERSION,
    CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH,
    CODEX_CONTAINER_BINARY_PATH,
    CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256,
    CODEX_HARNESS_AGENT,
    CODEX_HARNESS_AGENT_PATH,
    CODEX_HARNESS_AGENT_SHA256,
    CODEX_LINUX_BINARY_SHA256,
    CODEX_LINUX_BINARY_SIZE_BYTES,
    CODEX_LINUX_TARGET,
    CODEX_NPM_PACKAGE,
    CODEX_NPM_TARBALL_SHA256,
    CODEX_PACKAGE_MANIFEST_SHA256,
    CODEX_SUBMIT_TOOL_SCHEMA_SHA256,
    CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256,
    MILES_COMMIT,
    MILES_PEFT_VERSION,
    MILES_REPOSITORY,
    SECRLENV_AGENTS,
    SECRLENV_GENERATE,
    SECRLENV_GROUP_FILTER,
    SECRLENV_INFRASTRUCTURE_REPLACEMENTS,
    SECRLENV_REWARD,
    SECRLENV_ZERO_VARIANCE_REPLACEMENTS,
    SGLANG_COMMIT,
    SGLANG_REPOSITORY,
)
from .core import CanonicalTensorSpec, canonical_state, policy_hash, tensors_from_flat

PLAN_SCHEMA = 2
SYNCER_PORT = 29400
RAY_PORT = 6379
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_ROOT = ".cache/yeto-rl-ssh"
TMS_PRELOAD_PATCH_REPOSITORY = "https://github.com/fzyzcjy/torch_memory_saver"
TMS_PRELOAD_PATCH_BASE_COMMIT = "fdb09ad342f8d150e11afe719f798453cef742ad"
TMS_PRELOAD_PATCH_COMMIT = "d7a3a51bce723a80736dabf233017b927de03df8"
TMS_PRELOAD_PATCH_CONTAINER_PATH = (
    "/usr/local/lib/python3.12/dist-packages/"
    "torch_memory_saver_hook_mode_preload_cu13.abi3.so"
)
TMS_PRELOAD_BASE_BINARY_SHA256 = (
    "27fd2a983e6dca2bc30ebe834c32221b6a16afe9d44cd5ab78f06ba6d7a99b60"
)
TMS_TRAIN_DISK_BACKUP_CONTAINER_PATH = "/workspace/tms-disk-backup"
TMS_TRAIN_DISK_BACKUP_DEFAULT_CHUNK_MB = 256
TMS_TRAIN_DISK_BACKUP_ROOT = PurePosixPath("/data/yeto-rl/tms-disk-backup")
SYNCER_CHECKPOINT_MAGIC = 0xD170_5A7E
JIT_CACHE_SCHEMA = 1
JIT_CACHE_ROOT = PurePosixPath("/data/yeto-rl/jit-cache")
JIT_CACHE_MOUNTS = (
    ("deep-gemm", "/tmp/sglang_deep_gemm"),
    ("tvm-ffi", "/root/.cache/tvm-ffi"),
    ("triton", "/root/.triton"),
    ("flashinfer", "/root/.cache/flashinfer"),
    ("sglang", "/root/.cache/sglang"),
    ("cuda", "/root/.nv/ComputeCache"),
    ("tilelang", "/root/.tilelang"),
    ("cupy", "/root/.cupy/kernel_cache"),
    ("cutlass-python", "/tmp/root/cutlass_python_cache"),
    ("torchinductor", "/tmp/torchinductor_root"),
)


_RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,47}\Z")
_REMOTE_PATH = re.compile(r"[a-zA-Z0-9._/-]+\Z")
_REMOTE_ABSOLUTE_PATH = re.compile(r"/data/[a-zA-Z0-9._/-]+\Z")
_SSH_TARGET = re.compile(r"(?:[a-zA-Z0-9._-]+@)?[a-zA-Z0-9._-]+\Z")
_HOST = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9.-]*\Z")
_NETWORK_INTERFACE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}\Z")
_DOCKER_DIGEST = re.compile(r".+@sha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_IMAGE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]+\Z")
_EVAL_DATASET_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}\Z")
CODEX_REMOTE_BUNDLE_PATH = "codex/codex-x86_64-unknown-linux-musl"
CODEX_REMOTE_MANIFEST_PATH = "codex/codex-package.json"
CODEX_REMOTE_SCHEMA_PATH = "codex/codex_app_server_protocol.v2.schemas.json"
_CONTAINER_STATE_FORMAT = (
    '{"Status":{{json .State.Status}},'
    '"ExitCode":{{.State.ExitCode}},'
    '"OOMKilled":{{.State.OOMKilled}},'
    '"RestartCount":{{.RestartCount}}}'
)
_EVENT_EVIDENCE_FILTER = r"""
import json
import math
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
allowed = {
    "strict-learner": {
        "rl_local_round": ("event", "local_round_id", "base_policy_version"),
        "rl_policy_apply": ("event", "policy_version", "sync/global_policy_hash"),
    },
    "decoupled-learner": {
        "rl_policy_snapshot": ("event", "rl/fragment_versions", "rl/policy_hash"),
        "rl_policy_apply": ("event", "sync/global_policy_hash"),
        "rl_final_cut": ("event",),
    },
    "eval-learner": {
        "rl_policy_apply": (
            "event", "island_id", "policy_version", "sync/global_policy_hash"
        ),
        "rl_eval_result": (
            "event", "island_id", "rollout_id", "policy_version", "dataset_name",
            "sample_count", "rl/eval/result", "rl/eval/pass_at_1"
        ),
    },
    "syncer": {
        "syncer_commit": (
            "step", "fragment", "launch_base_version", "expected", "responded",
            "responders", "sync/layout_hash"
        ),
    },
}

if mode not in allowed or path.is_symlink() or not path.is_file():
    raise SystemExit("invalid event evidence source")
if path.stat().st_size > 64 * 1024 * 1024:
    raise SystemExit("event evidence source is too large")

sha256 = re.compile(r"[0-9a-f]{64}\Z")
safe_name = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
integer_fields = {
    "local_round_id", "base_policy_version", "policy_version", "island_id",
    "rollout_id", "sample_count", "step", "fragment", "launch_base_version",
    "id", "base_version", "c_steps", "c_tokens",
}
hash_fields = {"sync/global_policy_hash", "rl/policy_hash", "sync/layout_hash"}
list_integer_fields = {"rl/fragment_versions", "expected", "responded"}
float_fields = {"contribution", "rl/eval/result", "rl/eval/pass_at_1"}
responder_fields = ("id", "base_version", "c_steps", "c_tokens", "contribution")

def scalar(name, value):
    if name in integer_fields:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10**12:
            raise ValueError
    elif name in hash_fields:
        if not isinstance(value, str) or not sha256.fullmatch(value):
            raise ValueError
    elif name in list_integer_fields:
        if (
            not isinstance(value, list) or len(value) > 1024
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
        ):
            raise ValueError
    elif name in float_fields:
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            raise ValueError
    elif name == "dataset_name":
        if not isinstance(value, str) or not safe_name.fullmatch(value):
            raise ValueError
    elif name == "event":
        if not isinstance(value, str) or value not in {
            "rl_local_round", "rl_policy_apply", "rl_policy_snapshot",
            "rl_final_cut", "rl_eval_result"
        }:
            raise ValueError
    else:
        raise ValueError
    return value

count = 0
try:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if len(line.encode()) > 1024 * 1024:
                raise ValueError
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError
            if mode == "syncer":
                if "step" not in event:
                    continue
                fields = allowed[mode]["syncer_commit"]
            else:
                event_name = event.get("event")
                fields = allowed[mode].get(event_name)
                if fields is None:
                    continue
            evidence = {}
            for name in fields:
                if name not in event:
                    continue
                if name == "responders":
                    responders = event[name]
                    if not isinstance(responders, list) or len(responders) > 1024:
                        raise ValueError
                    evidence[name] = [
                        {
                            field: scalar(field, responder[field])
                            for field in responder_fields
                            if field in responder
                        }
                        for responder in responders
                        if isinstance(responder, dict)
                    ]
                    if len(evidence[name]) != len(responders):
                        raise ValueError
                else:
                    evidence[name] = scalar(name, event[name])
            encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
            if len(encoded) > 1024 * 1024:
                raise ValueError
            print(encoded)
            count += 1
            if count > 100000:
                raise ValueError
except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
    raise SystemExit("invalid event evidence source")
"""
_SYNCER_LIFECYCLE_EVIDENCE = r"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
    raise SystemExit("invalid syncer lifecycle source")
resume = 0
final_ack = 0
try:
    with path.open(encoding="utf-8", errors="strict") as source:
        for line in source:
            resume += line.count("resumed from checkpoint")
            final_ack += line.count("learner finalized authoritative parameters")
except (OSError, UnicodeError):
    raise SystemExit("invalid syncer lifecycle source")
print(json.dumps({"schema": 1, "resume_count": resume, "final_ack_count": final_ack},
                 sort_keys=True, separators=(",", ":")))
"""
_EVENT_EVIDENCE_FIELDS = {
    "strict-learner": {
        "rl_local_round": {
            "event",
            "local_round_id",
            "base_policy_version",
        },
        "rl_policy_apply": {
            "event",
            "policy_version",
            "sync/global_policy_hash",
        },
    },
    "decoupled-learner": {
        "rl_policy_snapshot": {
            "event",
            "rl/fragment_versions",
            "rl/policy_hash",
        },
        "rl_policy_apply": {
            "event",
            "sync/global_policy_hash",
        },
        "rl_final_cut": {"event"},
    },
    "eval-learner": {
        "rl_policy_apply": {
            "event",
            "island_id",
            "policy_version",
            "sync/global_policy_hash",
        },
        "rl_eval_result": {
            "event",
            "island_id",
            "rollout_id",
            "policy_version",
            "dataset_name",
            "sample_count",
            "rl/eval/result",
            "rl/eval/pass_at_1",
        },
    },
    "syncer": {
        "syncer_commit": {
            "step",
            "fragment",
            "launch_base_version",
            "expected",
            "responded",
            "responders",
            "sync/layout_hash",
        },
    },
}


class HarnessError(RuntimeError):
    pass


def _learner_count(plan: dict[str, Any]) -> int:
    islands = plan.get("islands")
    if not isinstance(islands, list) or not islands:
        raise HarnessError("plan must contain at least one Miles island")
    return len(islands)


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


def _validate_data_path(value: str, flag: str) -> str:
    path = PurePosixPath(value)
    if (
        not _REMOTE_ABSOLUTE_PATH.fullmatch(value)
        or ".." in path.parts
        or len(path.parts) < 3
    ):
        raise HarnessError(f"{flag} must be a safe absolute path below /data")
    return value.rstrip("/")


def _validate_target(value: str) -> str:
    if not _SSH_TARGET.fullmatch(value):
        raise HarnessError(f"invalid SSH target {value!r}")
    return value


def _remote_model_mount(reference: str | None) -> str | None:
    if reference is None or not reference.startswith("/"):
        return None
    path = PurePosixPath(reference)
    if (
        not _REMOTE_ABSOLUTE_PATH.fullmatch(reference)
        or ".." in path.parts
        or len(path.parts) < 4
    ):
        raise HarnessError(
            "remote model paths must be specific directories under /data"
        )
    if "snapshots" in path.parts:
        index = path.parts.index("snapshots")
        if index < 3 or index + 1 >= len(path.parts):
            raise HarnessError("remote Hugging Face snapshot path is incomplete")
        # Snapshot entries are relative symlinks into the repository's blobs/
        # directory, so mount the narrow models--ORG--NAME root, not /data/hf.
        path = PurePosixPath(*path.parts[:index])
    return path.as_posix()


def _remote_model_mounts(learner: dict[str, Any]) -> list[str]:
    mounts = {
        mount
        for mount in (
            _remote_model_mount(str(learner.get("model", ""))),
            _remote_model_mount(
                str(learner["rollout_model"])
                if learner.get("rollout_model") is not None
                else None
            ),
        )
        if mount is not None
    }
    return sorted(mounts)


def _target_host(target: str) -> str:
    return target.rsplit("@", 1)[-1]


def _syncer_host(plan: dict[str, Any]) -> str:
    target = plan.get("syncer_host")
    if target is None:
        target = plan["islands"][0]["hosts"][0]
    return _validate_target(str(target))


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


def _codex_adapter_identity() -> dict[str, str]:
    """Read hashes derived by the live Yeto Codex adapter, never user input."""

    try:
        from yeto_miles_secrlenv import codex_harness_agent

        identity = codex_harness_agent.codex_harness_identity()
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise HarnessError("cannot attest the Yeto Codex harness adapter") from exc
    required = {
        "base_instructions_sha256",
        "terminal_exec_tool_schema_sha256",
        "submit_tool_schema_sha256",
        "dynamic_tools_schema_sha256",
    }
    if not isinstance(identity, dict) or set(identity) != required:
        raise HarnessError("Yeto Codex adapter returned an invalid identity contract")
    normalized = {name: str(value).lower() for name, value in identity.items()}
    if any(not _SHA256.fullmatch(value) for value in normalized.values()):
        raise HarnessError("Yeto Codex adapter identity contains an invalid SHA256")
    expected = {
        "base_instructions_sha256": CODEX_BASE_INSTRUCTIONS_SHA256,
        "terminal_exec_tool_schema_sha256": (
            CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256
        ),
        "submit_tool_schema_sha256": CODEX_SUBMIT_TOOL_SCHEMA_SHA256,
        "dynamic_tools_schema_sha256": CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256,
    }
    if normalized != expected:
        raise HarnessError("Yeto Codex adapter identity does not match its Yeto pins")
    return normalized


def _codex_harness_contract(namespace, args) -> dict[str, Any]:
    """Attest the official Linux Codex artifact and the Yeto adapter surface."""

    from ..provenance import file_sha256

    binary_value = getattr(namespace, "codex_harness_binary", None)
    manifest_value = getattr(namespace, "codex_package_manifest", None)
    schema_value = getattr(namespace, "codex_app_server_schema", None)
    if not binary_value or not manifest_value or not schema_value:
        raise HarnessError(
            "the stock Codex agent requires --codex-harness-binary and "
            "--codex-package-manifest and --codex-app-server-schema"
        )
    binary = Path(binary_value).expanduser()
    manifest = Path(manifest_value).expanduser()
    schema = Path(schema_value).expanduser()
    for path, flag in (
        (binary, "--codex-harness-binary"),
        (manifest, "--codex-package-manifest"),
        (schema, "--codex-app-server-schema"),
    ):
        if path.is_symlink() or not path.is_file():
            raise HarnessError(f"{flag} must be a regular, non-symlink file")
    binary = binary.resolve()
    manifest = manifest.resolve()
    schema = schema.resolve()
    if not binary.stat().st_mode & 0o111:
        raise HarnessError("--codex-harness-binary must be executable")
    if binary.stat().st_size != CODEX_LINUX_BINARY_SIZE_BYTES:
        raise HarnessError("Codex Linux binary size does not match its Yeto pin")
    if file_sha256(binary) != CODEX_LINUX_BINARY_SHA256:
        raise HarnessError("Codex Linux binary SHA256 does not match its Yeto pin")
    if file_sha256(manifest) != CODEX_PACKAGE_MANIFEST_SHA256:
        raise HarnessError("Codex package manifest SHA256 does not match its Yeto pin")
    if file_sha256(schema) != CODEX_APP_SERVER_SCHEMA_SHA256:
        raise HarnessError("Codex app-server schema SHA256 does not match its Yeto pin")
    identity = _codex_adapter_identity()
    adapter_path = REPO_ROOT / CODEX_HARNESS_AGENT_PATH
    if adapter_path.is_symlink() or not adapter_path.is_file():
        raise HarnessError("Yeto Codex harness adapter is missing")
    if file_sha256(adapter_path) != CODEX_HARNESS_AGENT_SHA256:
        raise HarnessError("Yeto Codex harness adapter source does not match its pin")
    return {
        "agent_function_path": CODEX_HARNESS_AGENT,
        "agent_source_sha256": CODEX_HARNESS_AGENT_SHA256,
        "controller_binary_path": str(binary),
        "controller_package_manifest_path": str(manifest),
        "controller_app_server_schema_path": str(schema),
        "bundle_binary_path": CODEX_REMOTE_BUNDLE_PATH,
        "bundle_package_manifest_path": CODEX_REMOTE_MANIFEST_PATH,
        "bundle_app_server_schema_path": CODEX_REMOTE_SCHEMA_PATH,
        "container_binary_path": CODEX_CONTAINER_BINARY_PATH,
        "container_app_server_schema_path": (
            CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH
        ),
        "binary_sha256": CODEX_LINUX_BINARY_SHA256,
        "binary_size_bytes": CODEX_LINUX_BINARY_SIZE_BYTES,
        "cli_version": CODEX_CLI_VERSION,
        "npm_package": CODEX_NPM_PACKAGE,
        "target": CODEX_LINUX_TARGET,
        "npm_tarball_sha256": CODEX_NPM_TARBALL_SHA256,
        "package_manifest_sha256": CODEX_PACKAGE_MANIFEST_SHA256,
        "app_server_protocol_revision": CODEX_APP_SERVER_PROTOCOL_REVISION,
        "app_server_schema_sha256": CODEX_APP_SERVER_SCHEMA_SHA256,
        **identity,
        "reasoning_effort": args.codex_reasoning_effort,
        "backend": {
            "model": args.tito_model,
            "max_tokens": args.rollout_max_response_len,
            "reasoning_effort": "max",
            "thinking": {"type": "enabled"},
            "chat_template": args.tito_model,
            "chat_template_kwargs": dict(args.apply_chat_template_kwargs),
            "tito_allowed_append_roles": list(args.tito_allowed_append_roles),
        },
    }


def _validate_codex_harness(value: Any, learner: dict[str, Any]) -> None:
    required = {
        "agent_function_path",
        "agent_source_sha256",
        "controller_binary_path",
        "controller_package_manifest_path",
        "controller_app_server_schema_path",
        "bundle_binary_path",
        "bundle_package_manifest_path",
        "bundle_app_server_schema_path",
        "container_binary_path",
        "container_app_server_schema_path",
        "binary_sha256",
        "binary_size_bytes",
        "cli_version",
        "npm_package",
        "target",
        "npm_tarball_sha256",
        "package_manifest_sha256",
        "app_server_protocol_revision",
        "app_server_schema_sha256",
        "base_instructions_sha256",
        "terminal_exec_tool_schema_sha256",
        "submit_tool_schema_sha256",
        "dynamic_tools_schema_sha256",
        "reasoning_effort",
        "backend",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise HarnessError("plan has an invalid stock Codex harness contract")
    pinned = {
        "agent_function_path": CODEX_HARNESS_AGENT,
        "agent_source_sha256": CODEX_HARNESS_AGENT_SHA256,
        "bundle_binary_path": CODEX_REMOTE_BUNDLE_PATH,
        "bundle_package_manifest_path": CODEX_REMOTE_MANIFEST_PATH,
        "bundle_app_server_schema_path": CODEX_REMOTE_SCHEMA_PATH,
        "container_binary_path": CODEX_CONTAINER_BINARY_PATH,
        "container_app_server_schema_path": (
            CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH
        ),
        "binary_sha256": CODEX_LINUX_BINARY_SHA256,
        "binary_size_bytes": CODEX_LINUX_BINARY_SIZE_BYTES,
        "cli_version": CODEX_CLI_VERSION,
        "npm_package": CODEX_NPM_PACKAGE,
        "target": CODEX_LINUX_TARGET,
        "npm_tarball_sha256": CODEX_NPM_TARBALL_SHA256,
        "package_manifest_sha256": CODEX_PACKAGE_MANIFEST_SHA256,
        "app_server_protocol_revision": CODEX_APP_SERVER_PROTOCOL_REVISION,
        "app_server_schema_sha256": CODEX_APP_SERVER_SCHEMA_SHA256,
        "base_instructions_sha256": CODEX_BASE_INSTRUCTIONS_SHA256,
        "terminal_exec_tool_schema_sha256": (
            CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256
        ),
        "submit_tool_schema_sha256": CODEX_SUBMIT_TOOL_SCHEMA_SHA256,
        "dynamic_tools_schema_sha256": CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256,
        "reasoning_effort": "xhigh",
    }
    if any(value.get(name) != expected for name, expected in pinned.items()):
        raise HarnessError("plan does not use the pinned stock Codex runtime")
    for name in (
        "controller_binary_path",
        "controller_package_manifest_path",
        "controller_app_server_schema_path",
    ):
        path = Path(str(value.get(name, "")))
        if not path.is_absolute() or ".." in path.parts:
            raise HarnessError(f"stock Codex contract has an invalid {name}")
    backend = value.get("backend")
    expected_backend = {
        "model": "deepseekv4",
        "max_tokens": learner.get("rollout_max_response_len"),
        "reasoning_effort": "max",
        "thinking": {"type": "enabled"},
        "chat_template": "deepseekv4",
        "chat_template_kwargs": {
            "thinking_mode": "thinking",
            "reasoning_effort": "max",
            "drop_thinking": False,
        },
        "tito_allowed_append_roles": ["tool", "user"],
    }
    if backend != expected_backend:
        raise HarnessError("stock Codex backend/TITO contract drifted")
    if (
        learner.get("rl_model_recipe") != "deepseek-v4-flash"
        or learner.get("custom_agent_function_path") != CODEX_HARNESS_AGENT
        or learner.get("codex_reasoning_effort") != "xhigh"
        or learner.get("apply_chat_template_kwargs")
        != expected_backend["chat_template_kwargs"]
        or learner.get("tito_model") != "deepseekv4"
        or learner.get("tito_allowed_append_roles") != ["tool", "user"]
    ):
        raise HarnessError("stock Codex is restricted to the signed DSV4 agent path")


def _tms_preload_patch(path: str | Path) -> dict[str, str]:
    """Attest the narrow upstream fix over the pinned Miles image binary."""

    from ..provenance import file_sha256

    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise HarnessError("--tms-preload-patch must be a regular, non-symlink file")
    source = source.resolve()
    try:
        relative = source.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise HarnessError(
            "--tms-preload-patch must be inside the Yeto source tree"
        ) from exc
    source_path = relative.as_posix()
    if (
        not _REMOTE_PATH.fullmatch(source_path)
        or ".." in relative.parts
        or source.suffix != ".so"
    ):
        raise HarnessError("--tms-preload-patch has an unsafe source path")
    return {
        "repository": TMS_PRELOAD_PATCH_REPOSITORY,
        "base_commit": TMS_PRELOAD_PATCH_BASE_COMMIT,
        "patch_commit": TMS_PRELOAD_PATCH_COMMIT,
        "source_path": source_path,
        "container_path": TMS_PRELOAD_PATCH_CONTAINER_PATH,
        "base_binary_sha256": TMS_PRELOAD_BASE_BINARY_SHA256,
        "binary_sha256": file_sha256(source),
    }


def _validate_tms_preload_patch(value: Any) -> None:
    if not isinstance(value, dict):
        raise HarnessError("offloaded training requires an attested TMS preload patch")
    if set(value) != {
        "repository",
        "base_commit",
        "patch_commit",
        "source_path",
        "container_path",
        "base_binary_sha256",
        "binary_sha256",
    }:
        raise HarnessError("plan has an invalid TMS preload patch contract")
    if (
        value.get("repository") != TMS_PRELOAD_PATCH_REPOSITORY
        or value.get("base_commit") != TMS_PRELOAD_PATCH_BASE_COMMIT
        or value.get("patch_commit") != TMS_PRELOAD_PATCH_COMMIT
        or value.get("container_path") != TMS_PRELOAD_PATCH_CONTAINER_PATH
        or value.get("base_binary_sha256") != TMS_PRELOAD_BASE_BINARY_SHA256
    ):
        raise HarnessError("plan does not use the pinned TMS free-while-paused fix")
    source_path = str(value.get("source_path", ""))
    relative = PurePosixPath(source_path)
    if (
        not source_path
        or source_path.startswith(("/", "~"))
        or not _REMOTE_PATH.fullmatch(source_path)
        or ".." in relative.parts
        or relative.suffix != ".so"
    ):
        raise HarnessError("plan has an invalid TMS preload patch source path")
    if not _SHA256.fullmatch(str(value.get("binary_sha256", ""))):
        raise HarnessError("plan has an invalid TMS preload patch SHA256")


def _validate_tms_train_disk_backup_root(value: str) -> str:
    path = PurePosixPath(value)
    prefix = TMS_TRAIN_DISK_BACKUP_ROOT
    if (
        not value
        or not _REMOTE_ABSOLUTE_PATH.fullmatch(value)
        or ".." in path.parts
        or path.parts[: len(prefix.parts)] != prefix.parts
    ):
        raise HarnessError(
            "--tms-train-disk-backup-root must be the dedicated "
            "/data/yeto-rl/tms-disk-backup directory or one of its descendants"
        )
    return path.as_posix()


def _validate_tms_train_disk_backup_chunk_mb(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > 4096
    ):
        raise HarnessError(
            "--tms-train-disk-backup-chunk-mb must be an integer from 1 to 4096"
        )
    return value


def _tms_train_disk_backup_contract(
    host_root: str,
    run_id: str,
    islands: list[dict[str, Any]],
    chunk_mb: int,
) -> dict[str, Any]:
    root = _validate_tms_train_disk_backup_root(host_root)
    run_id = _validate_run_id(run_id)
    chunk_mb = _validate_tms_train_disk_backup_chunk_mb(chunk_mb)
    node_paths = {}
    for learner_id, island in enumerate(islands):
        for node_id, target in enumerate(island["hosts"]):
            node_paths[target] = (
                PurePosixPath(root)
                / run_id
                / f"island-{learner_id}-node-{node_id}"
            ).as_posix()
    return {
        "host_root": root,
        "container_path": TMS_TRAIN_DISK_BACKUP_CONTAINER_PATH,
        "chunk_mb": chunk_mb,
        "node_paths": node_paths,
    }


def _validate_tms_train_disk_backup(
    value: Any,
    plan: dict[str, Any],
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "host_root",
        "container_path",
        "chunk_mb",
        "node_paths",
    }:
        raise HarnessError(
            "offloaded training requires a TMS trainer disk-backup contract"
        )
    expected = _tms_train_disk_backup_contract(
        str(value.get("host_root", "")),
        str(plan.get("run_id", "")),
        plan["islands"],
        value.get("chunk_mb"),
    )
    if value != expected:
        raise HarnessError("plan has an invalid TMS trainer disk-backup contract")


def _validate_jit_cache_root(value: str) -> str:
    path = PurePosixPath(value)
    prefix = JIT_CACHE_ROOT
    if (
        not value
        or not _REMOTE_ABSOLUTE_PATH.fullmatch(value)
        or ".." in path.parts
        or path.parts[: len(prefix.parts)] != prefix.parts
    ):
        raise HarnessError(
            "--jit-cache-root must be the dedicated /data/yeto-rl/jit-cache "
            "directory or one of its descendants"
        )
    return path.as_posix()


def _jit_cache_compatibility_sha256(plan: dict[str, Any]) -> str:
    identity = {
        "schema": JIT_CACHE_SCHEMA,
        "accelerator": "H200",
        "docker_image": plan.get("docker_image"),
        "miles_commit": plan.get("miles", {}).get("commit"),
        "sglang_commit": plan.get("sglang", {}).get("commit"),
        "mounts": list(JIT_CACHE_MOUNTS),
    }
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


def _jit_cache_contract(
    host_root: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "host_root": _validate_jit_cache_root(host_root),
        "compatibility_sha256": _jit_cache_compatibility_sha256(plan),
    }


def _validate_jit_cache(value: Any, plan: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {
        "host_root",
        "compatibility_sha256",
    }:
        raise HarnessError("plan has an invalid JIT cache contract")
    expected = _jit_cache_contract(str(value.get("host_root", "")), plan)
    if value != expected:
        raise HarnessError("plan has an invalid JIT cache compatibility identity")


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


def _local_jsonl_record_count(path: str | Path) -> int:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise HarnessError("evaluation dataset must be one regular JSONL file")
    try:
        with source.open("rb") as handle:
            count = sum(1 for line in handle if line.strip())
    except OSError as error:
        raise HarnessError("cannot read evaluation dataset") from error
    if count <= 0:
        raise HarnessError("evaluation dataset is empty")
    return count


def _eval_checkpoint_contract(
    source_host: str,
    source_path: str,
    sha256: str,
    size_bytes: int,
    global_step: int,
) -> dict[str, Any]:
    """Pin a remote terminal checkpoint without staging it on the controller."""

    source_host = _validate_target(source_host)
    source_path = _validate_remote_path(source_path, "--eval-checkpoint-path")
    sha256 = str(sha256).lower()
    if not _SHA256.fullmatch(sha256):
        raise HarnessError("--eval-checkpoint-sha256 must be a lowercase SHA256")
    for flag, value in (
        ("--eval-checkpoint-size-bytes", size_bytes),
        ("--eval-checkpoint-global-step", global_step),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise HarnessError(f"{flag} must be a positive integer")
    return {
        "source_host": source_host,
        "source_path": source_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "global_step": global_step,
    }


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
    if not isinstance(islands, list) or not islands:
        raise HarnessError("plan must contain at least one Miles island")
    all_hosts = []
    node_counts = set()
    gpu_counts = set()
    accelerators = set()
    for island in islands:
        hosts = island.get("hosts") if isinstance(island, dict) else None
        gpus = island.get("gpus_per_node") if isinstance(island, dict) else None
        if (
            not isinstance(hosts, list)
            or not hosts
            or not isinstance(gpus, int)
            or gpus <= 0
        ):
            raise HarnessError("each island needs hosts and a positive GPU count")
        node_counts.add(len(hosts))
        gpu_counts.add(gpus)
        accelerator = island.get("accelerator", "H200")
        if accelerator != "H200":
            raise HarnessError("the direct SSH RL harness is pinned to H200")
        accelerators.add(accelerator)
        all_hosts.extend(_validate_target(host) for host in hosts)
    if len(node_counts) != 1:
        raise HarnessError("all acceptance islands must use the same node count")
    if len(gpu_counts) != 1:
        raise HarnessError("all acceptance islands must use the same GPU count")
    if accelerators != {"H200"}:
        raise HarnessError("all acceptance islands must record H200")
    if len(set(all_hosts)) != len(all_hosts):
        raise HarnessError("each SSH target may belong to only one island")
    _syncer_host(plan)
    _, port = _validate_address(str(plan.get("syncer_address", "")))
    if plan.get("syncer_port") != port:
        raise HarnessError("plan syncer port does not match its address")
    final_ack_timeout_s = plan.get("final_ack_timeout_s")
    if final_ack_timeout_s is not None and (
        not isinstance(final_ack_timeout_s, int)
        or isinstance(final_ack_timeout_s, bool)
        or final_ack_timeout_s <= 0
    ):
        raise HarnessError("plan has an invalid final_ack_timeout_s")
    if not _NETWORK_INTERFACE.fullmatch(str(plan.get("network_interface", "eno3"))):
        raise HarnessError("plan has an invalid network interface")
    _docker_ref(str(plan.get("docker_image", "")))
    if plan.get("miles") != {
        "repository": MILES_REPOSITORY,
        "commit": MILES_COMMIT,
        "peft_version": MILES_PEFT_VERSION,
    }:
        raise HarnessError("plan does not use the current pinned Miles revision")
    if plan.get("sglang") != {
        "repository": SGLANG_REPOSITORY,
        "commit": SGLANG_COMMIT,
    }:
        raise HarnessError("plan does not use the current pinned SGLang revision")
    jit_cache = plan.get("jit_cache")
    if jit_cache is not None:
        _validate_jit_cache(jit_cache, plan)
    for name in ("source_sha256", "reward_sha256", "syncer_source_sha256"):
        if not _SHA256.fullmatch(str(plan.get(name, ""))):
            raise HarnessError(f"plan has an invalid {name}")
    learner = plan.get("learner")
    if not isinstance(learner, dict):
        raise HarnessError("plan has no learner configuration")
    daemon = plan.get("secrlenv_daemon")
    if learner.get("custom_agent_function_path") in SECRLENV_AGENTS:
        if not isinstance(daemon, dict):
            raise HarnessError("secrlenv agent requires a pinned daemon contract")
        for name in ("source_root", "task_pack", "state_root"):
            _validate_data_path(str(daemon.get(name, "")), f"secrlenv {name}")
        for name in ("source_sha256", "task_pack_sha256"):
            if not _SHA256.fullmatch(str(daemon.get(name, ""))):
                raise HarnessError(f"secrlenv daemon has an invalid {name}")
        if (
            daemon.get("bind") != "127.0.0.1"
            or daemon.get("placement", "all-hosts")
            not in {"all-hosts", "island-heads"}
            or not isinstance(daemon.get("port"), int)
            or not 1024 <= daemon["port"] <= 65535
            or not isinstance(daemon.get("max_active_episodes"), int)
            or daemon["max_active_episodes"] <= 0
            or not _SAFE_IMAGE.fullmatch(str(daemon.get("operator_image", "")))
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(daemon.get("operator_image_id", ""))
            )
        ):
            raise HarnessError("secrlenv daemon contract is invalid")
        if not plan.get("remote_env_file"):
            raise HarnessError("secrlenv daemon requires --remote-env-file")
    elif daemon is not None:
        raise HarnessError("secrlenv daemon requires the secrlenv custom agent")
    tms_patch = plan.get("tms_preload_patch")
    tms_train_disk_backup = plan.get("tms_train_disk_backup")
    if learner.get("rl_offload_train"):
        _validate_tms_preload_patch(tms_patch)
        _validate_tms_train_disk_backup(tms_train_disk_backup, plan)
    else:
        if tms_patch is not None:
            _validate_tms_preload_patch(tms_patch)
        if tms_train_disk_backup is not None:
            raise HarnessError("TMS trainer disk backup requires --rl-offload-train")
    if learner.get("rl_model_recipe", "generic") not in {
        "generic",
        "deepseek-v4-flash",
    }:
        raise HarnessError("plan has invalid rl_model_recipe")
    if learner.get("model_mounts") != _remote_model_mounts(learner):
        raise HarnessError("plan has invalid remote model mounts")
    model_manifest_sha256 = learner.get("model_manifest_sha256")
    if str(learner.get("model", "")).startswith("/"):
        if not _SHA256.fullmatch(str(model_manifest_sha256 or "")):
            raise HarnessError(
                "remote-local model requires an immutable conversion manifest SHA256"
            )
    elif model_manifest_sha256 is not None:
        raise HarnessError("Hub model may not declare a local conversion manifest")
    if not _REVISION.fullmatch(str(learner.get("model_revision", ""))):
        raise HarnessError("plan model revision must be an immutable commit")
    rollout_model = learner.get("rollout_model")
    rollout_revision = learner.get("rollout_model_revision")
    if rollout_model is not None and rollout_model != learner.get("model"):
        if not _REVISION.fullmatch(str(rollout_revision or "")):
            raise HarnessError(
                "plan rollout model revision must be an immutable commit"
            )
    elif rollout_revision is not None and not _REVISION.fullmatch(
        str(rollout_revision)
    ):
        raise HarnessError("plan has invalid rollout model revision")
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
    evaluation = learner.get("evaluation")
    if evaluation is not None:
        if not isinstance(evaluation, dict):
            raise HarnessError("plan has an invalid learner evaluation")
        if evaluation.get("eval_only") is not True:
            raise HarnessError("SSH evaluation must use a separate eval-only run")
        if learner.get("custom_agent_function_path") not in SECRLENV_AGENTS:
            raise HarnessError("SSH evaluation is restricted to the secrlenv agent")
        if learner.get("reward_function") != SECRLENV_REWARD:
            raise HarnessError(
                "SSH evaluation requires the signed secrlenv reward contract"
            )
        if sync_preset != "strict-avg":
            raise HarnessError(
                "final SSH evaluation requires strict-avg synchronization"
            )
        if data_local_path is None or not PurePosixPath(
            str(learner.get("data", ""))
        ).suffix:
            raise HarnessError(
                "SSH evaluation requires one local training dataset file"
            )
        if (
            evaluation.get("data") != learner.get("data")
            or evaluation.get("data_sha256") != data_sha256
        ):
            raise HarnessError(
                "evaluation must reuse the exact training dataset path and SHA256"
            )
        if not _EVAL_DATASET_NAME.fullmatch(
            str(evaluation.get("dataset_name", ""))
        ):
            raise HarnessError("plan has an invalid evaluation dataset_name")
        interval = evaluation.get("interval")
        samples = evaluation.get("samples_per_prompt")
        prompt_count = evaluation.get("prompt_count")
        if (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval != 1
        ):
            raise HarnessError("eval-only Miles evaluation interval must be 1")
        if (
            not isinstance(samples, int)
            or isinstance(samples, bool)
            or samples <= 0
        ):
            raise HarnessError("plan has an invalid evaluation samples_per_prompt")
        if (
            not isinstance(prompt_count, int)
            or isinstance(prompt_count, bool)
            or prompt_count <= 0
        ):
            raise HarnessError("plan has an invalid evaluation prompt_count")
        if evaluation.get("skip_before_train") is not True:
            raise HarnessError(
                "post-training evaluation must skip pre-train evaluation"
            )
        if final_ack_timeout_s is None or final_ack_timeout_s <= 3600:
            raise HarnessError(
                "evaluation requires an explicitly extended final_ack_timeout_s"
            )
        temperature = evaluation.get("temperature")
        if temperature is not None and (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise HarnessError("plan has an invalid evaluation temperature")
        top_p = evaluation.get("top_p")
        if top_p is not None and (
            not isinstance(top_p, (int, float))
            or isinstance(top_p, bool)
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise HarnessError("plan has an invalid evaluation top_p")
        for name in ("max_prompt_len", "max_response_len", "max_context_len"):
            value = evaluation.get(name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value > learner["seq_len"]
            ):
                raise HarnessError(f"plan has an invalid evaluation {name}")
        checkpoint = plan.get("eval_checkpoint")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "source_host",
            "source_path",
            "sha256",
            "size_bytes",
            "global_step",
        }:
            raise HarnessError("eval-only plan requires an attested syncer checkpoint")
        try:
            _validate_target(str(checkpoint.get("source_host", "")))
            _validate_remote_path(
                str(checkpoint.get("source_path", "")),
                "eval checkpoint source_path",
            )
        except HarnessError as error:
            raise HarnessError("plan has an invalid eval checkpoint identity") from error
        if (
            not _SHA256.fullmatch(str(checkpoint.get("sha256", "")))
            or not isinstance(checkpoint.get("size_bytes"), int)
            or isinstance(checkpoint.get("size_bytes"), bool)
            or checkpoint["size_bytes"] <= 12
            or checkpoint.get("global_step") != learner["global_rounds"]
        ):
            raise HarnessError("plan has an invalid eval checkpoint identity")
        if (
            checkpoint["source_host"] == _syncer_host(plan)
            and checkpoint["source_path"]
            == f"{plan['remote_run']}/state/state.ckpt"
        ):
            raise HarnessError("eval checkpoint source and fresh destination coincide")
    elif plan.get("eval_checkpoint") is not None:
        raise HarnessError("eval checkpoint requires an eval-only learner")
    fragments = learner.get("fragments", 1)
    pipeline = learner.get("pipeline", 1)
    local_horizon = learner.get("local_horizon", 1)
    total_fragment_steps = learner.get("total_fragment_steps", learner["global_rounds"])
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
    dynamic_filter = learner.get("dynamic_sampling_filter_path")
    if dynamic_filter:
        parts = str(dynamic_filter).split(".")
        if len(parts) < 2 or any(not part.isidentifier() for part in parts):
            raise HarnessError("plan has an invalid dynamic_sampling_filter_path")
        if learner["over_sampling_batch_size"] <= learner["groups_per_round"]:
            raise HarnessError(
                "variance-aware filtering requires oversampling beyond the training batch"
            )
    max_replacements = learner.get("dynamic_sampling_max_replacements")
    if max_replacements is not None and (
        type(max_replacements) is not int or max_replacements < 0
    ):
        raise HarnessError("plan has an invalid dynamic_sampling_max_replacements")
    infrastructure_replacements = learner.get(
        "secrlenv_max_infrastructure_replacements"
    )
    agent_function = learner.get("custom_agent_function_path")
    if agent_function in SECRLENV_AGENTS:
        if learner.get("custom_generate_function_path") != SECRLENV_GENERATE:
            raise HarnessError(
                "secrlenv agent requires the signed generate wrapper"
            )
        if learner.get("reward_function") != SECRLENV_REWARD:
            raise HarnessError("secrlenv agent requires the signed reward function")
        if dynamic_filter != SECRLENV_GROUP_FILTER:
            raise HarnessError("secrlenv agent requires the signed group filter")
        if max_replacements != SECRLENV_ZERO_VARIANCE_REPLACEMENTS or isinstance(
            max_replacements, bool
        ):
            raise HarnessError(
                "secrlenv agent requires zero variance replacements"
            )
        if (
            infrastructure_replacements
            != SECRLENV_INFRASTRUCTURE_REPLACEMENTS
            or isinstance(infrastructure_replacements, bool)
        ):
            raise HarnessError(
                "secrlenv agent requires exactly one infrastructure replacement"
            )
        if learner["over_sampling_batch_size"] != learner["groups_per_round"] + 1:
            raise HarnessError(
                "secrlenv agent requires one reserved same-task retry slot"
            )
    elif infrastructure_replacements is not None:
        raise HarnessError(
            "secrlenv infrastructure replacements require a secrlenv agent"
        )
    timeout_minutes = learner.get("rl_distributed_timeout_minutes", 10)
    if not isinstance(timeout_minutes, int) or timeout_minutes <= 0:
        raise HarnessError("plan has an invalid rl_distributed_timeout_minutes")
    tensor_parallel = learner.get("tensor_parallel", 1)
    pipeline_parallel = learner.get("pipeline_parallel", 1)
    if (
        not isinstance(tensor_parallel, int)
        or tensor_parallel <= 0
        or not isinstance(pipeline_parallel, int)
        or pipeline_parallel <= 0
    ):
        raise HarnessError("plan has invalid tensor/pipeline parallelism")
    world = islands[0]["gpus_per_node"] * len(islands[0]["hosts"])
    if world % (tensor_parallel * pipeline_parallel):
        raise HarnessError("plan TP*PP does not divide the island GPU world")
    data_parallel = world // (tensor_parallel * pipeline_parallel)
    if (learner["groups_per_round"] * learner["samples_per_group"]) % data_parallel:
        raise HarnessError("plan Miles batch does not divide across data parallelism")
    rollout_gpus = learner.get("rollout_num_gpus_per_engine", 1)
    if not isinstance(rollout_gpus, int) or rollout_gpus <= 0 or world % rollout_gpus:
        raise HarnessError("plan has invalid rollout_num_gpus_per_engine")
    memory_fraction = learner.get("sglang_mem_fraction_static", 0.4)
    if not isinstance(memory_fraction, (int, float)) or not 0 < memory_fraction < 1:
        raise HarnessError("plan has invalid sglang_mem_fraction_static")
    deterministic = learner.get("sglang_deterministic_inference", True)
    if not isinstance(deterministic, bool):
        raise HarnessError("plan has invalid sglang_deterministic_inference")
    if deterministic and learner.get("sglang_attention_backend") in {
        "dsv4",
        "compressed",
    }:
        raise HarnessError(
            "dsv4/compressed attention is incompatible with deterministic inference"
        )
    expert_full_count = learner.get("expert_full_count", 0)
    if (
        not isinstance(expert_full_count, int)
        or isinstance(expert_full_count, bool)
        or not 0 <= expert_full_count <= 32
    ):
        raise HarnessError("plan has an invalid expert_full_count")
    if expert_full_count:
        expert_full_lr = learner.get("expert_full_lr")
        if (
            not isinstance(expert_full_lr, (int, float))
            or isinstance(expert_full_lr, bool)
            or not 0 < expert_full_lr < float("inf")
        ):
            raise HarnessError("plan has an invalid expert_full_lr")
        for name in (
            "expert_selection_sha256",
            "expert_selection_contract_sha256",
        ):
            if not _SHA256.fullmatch(str(learner.get(name, ""))):
                raise HarnessError(f"plan has an invalid {name}")
        if (
            learner.get("rl_model_recipe") != "deepseek-v4-flash"
            or learner.get("lora_targets") != "attention"
        ):
            raise HarnessError(
                "expert-full plan requires DeepSeek V4 and attention LoRA"
            )
    elif learner.get("expert_selection_sha256") is not None or learner.get(
        "expert_selection_contract_sha256"
    ) is not None:
        raise HarnessError("expert selection hashes require expert_full_count")
    if learner.get("rl_model_recipe") == "deepseek-v4-flash":
        expected_lora_targets = (
            "attention" if expert_full_count else "attention-routed-experts"
        )
        if (
            learner.get("lora_targets") != expected_lora_targets
            or tensor_parallel != 8
            or learner.get("expert_parallel") != 8
            or rollout_gpus != 8
            or learner.get("sglang_tp_size") != 8
            or learner.get("sglang_ep_size") != 8
            or learner.get("sglang_dp_size") not in (None, 1)
        ):
            raise HarnessError(
                "expanded DeepSeek V4 plan must use TP8/EP8 pipeline stages "
                "and per-node eight-GPU rollout replicas"
            )
    custom_agent = learner.get("custom_agent_function_path")
    if custom_agent is not None and (
        not isinstance(custom_agent, str)
        or not learner.get("custom_generate_function_path")
        or not learner.get("use_session_server")
        or not learner.get("tito_model")
    ):
        raise HarnessError("plan has an incomplete agentic session contract")
    codex_harness = plan.get("codex_harness")
    if custom_agent == CODEX_HARNESS_AGENT:
        _validate_codex_harness(codex_harness, learner)
    elif codex_harness is not None:
        raise HarnessError("stock Codex contract requires the signed Codex agent")
    roles = learner.get("tito_allowed_append_roles")
    if roles is not None and (
        not isinstance(roles, list)
        or not roles
        or any(role not in {"tool", "user", "system"} for role in roles)
        or len(set(roles)) != len(roles)
    ):
        raise HarnessError("plan has invalid tito_allowed_append_roles")
    if learner.get("cybergym_reward_scheme", "binary") not in {
        "binary",
        "shaped_v1",
    }:
        raise HarnessError("plan has an invalid cybergym_reward_scheme")
    if learner.get("cybergym_reward_view", "train") not in {"train", "final"}:
        raise HarnessError("plan has an invalid cybergym_reward_view")


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
    if not values or gpus_per_node <= 0:
        raise HarnessError(
            "prepare requires one or more --host groups and positive GPUs per node"
        )
    islands = []
    for value in values:
        hosts = [
            _validate_target(item.strip()) for item in value.split(",") if item.strip()
        ]
        if not hosts:
            raise HarnessError("each --host must contain at least one SSH target")
        islands.append(
            {"hosts": hosts, "gpus_per_node": gpus_per_node, "accelerator": "H200"}
        )
    if len({len(island["hosts"]) for island in islands}) != 1:
        raise HarnessError("all --host groups must contain the same number of nodes")
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
        f"ssh:{len(island['hosts'])}x{island['gpus_per_node']}xh200@island-{index}"
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
    prepare_launch_args(
        args,
        allow_local_rl_data=True,
        allow_remote_rl_model=True,
    )
    return args


def prepare(namespace) -> Path:
    run_id = _validate_run_id(namespace.run_id)
    islands = _parse_islands(namespace.host, namespace.gpus_per_node)
    remote_root = _validate_remote_path(namespace.remote_root, "--remote-root")
    if namespace.remote_env_file is not None:
        _validate_remote_path(namespace.remote_env_file, "--remote-env-file")
    syncer_host = getattr(namespace, "syncer_host", None) or islands[0]["hosts"][0]
    _validate_target(syncer_host)
    syncer_address = namespace.syncer_address or (
        f"{_target_host(syncer_host)}:{SYNCER_PORT}"
    )
    _, syncer_port = _validate_address(syncer_address)
    network_interface = str(getattr(namespace, "network_interface", "eno3"))
    if not _NETWORK_INTERFACE.fullmatch(network_interface):
        raise HarnessError("--network-interface is invalid")
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
    data_prompt_count = None
    if dataset["source"] == "local":
        source = Path(args.data).expanduser().absolute()
        data_sha256 = _local_data_sha256(source)
        source = source.resolve()
        data_local_path = str(source)
        data = "/workspace/data/dataset"
        if source.is_file():
            data += source.suffix
            if getattr(namespace, "eval_checkpoint_host", None) is not None:
                data_prompt_count = _local_jsonl_record_count(source)

    tms_patch_path = getattr(namespace, "tms_preload_patch", None)
    tms_preload_patch = (
        _tms_preload_patch(tms_patch_path) if tms_patch_path is not None else None
    )
    tms_train_disk_backup_root = getattr(
        namespace,
        "tms_train_disk_backup_root",
        None,
    )
    tms_train_disk_backup = None
    if bool(getattr(args, "rl_offload_train", False)):
        if tms_preload_patch is None:
            raise HarnessError(
                "--rl-offload-train requires --tms-preload-patch with the pinned "
                "free-while-paused fix"
            )
        if tms_train_disk_backup_root is None:
            raise HarnessError(
                "--rl-offload-train requires --tms-train-disk-backup-root"
            )
        tms_train_disk_backup = _tms_train_disk_backup_contract(
            tms_train_disk_backup_root,
            run_id,
            islands,
            getattr(
                namespace,
                "tms_train_disk_backup_chunk_mb",
                TMS_TRAIN_DISK_BACKUP_DEFAULT_CHUNK_MB,
            ),
        )
    elif tms_train_disk_backup_root is not None:
        raise HarnessError(
            "--tms-train-disk-backup-root requires --rl-offload-train"
        )

    plan = {
        "schema": PLAN_SCHEMA,
        "run_id": run_id,
        "created_unix_ns": time.time_ns(),
        "remote_run": f"{remote_root}/{run_id}",
        "remote_env_file": namespace.remote_env_file,
        "ssh_options": list(namespace.ssh_option),
        "islands": islands,
        "syncer_host": syncer_host,
        "syncer_address": syncer_address,
        "syncer_port": syncer_port,
        "network_interface": network_interface,
        "docker_image": _docker_ref(args.rl_image),
        "miles": {
            "repository": MILES_REPOSITORY,
            "commit": MILES_COMMIT,
            "peft_version": MILES_PEFT_VERSION,
        },
        "sglang": {
            "repository": SGLANG_REPOSITORY,
            "commit": SGLANG_COMMIT,
        },
        "source_sha256": args.source_sha256,
        "reward_sha256": args.reward_sha256,
        "syncer_source_sha256": _syncer_source_sha256(),
        "tms_preload_patch": tms_preload_patch,
        "tms_train_disk_backup": tms_train_disk_backup,
        "learner": {
            "model": args.model,
            "rollout_model": getattr(args, "rollout_model", None),
            "rollout_model_revision": getattr(args, "rollout_model_revision", None),
            "rl_model_recipe": args.rl_model_recipe,
            "model_mounts": _remote_model_mounts(
                {
                    "model": args.model,
                    "rollout_model": getattr(args, "rollout_model", None),
                }
            ),
            "model_manifest_sha256": namespace.model_manifest_sha256,
            "model_revision": args.model_revision,
            "data": data,
            "data_revision": args.data_revision,
            "data_local_path": data_local_path,
            "data_sha256": data_sha256,
            "reward_function": args.reward_function,
            # Telemetry settings only. WANDB_API_KEY belongs in the remote
            # env file alongside HF_TOKEN, so the plan keeps its promise of
            # storing the path and never the contents.
            "wandb": getattr(args, "wandb", False),
            "wandb_project": getattr(args, "wandb_project", "yeto"),
            "wandb_entity": getattr(args, "wandb_entity", None),
            "wandb_mode": getattr(args, "wandb_mode", "online"),
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
            "secrlenv_max_infrastructure_replacements": getattr(
                args, "secrlenv_max_infrastructure_replacements", None
            ),
            "rl_offload_train": bool(getattr(args, "rl_offload_train", False)),
            "rl_distributed_timeout_minutes": getattr(
                args, "rl_distributed_timeout_minutes", 10
            ),
            "optimizer_steps": 1,
            "rollout_max_response_len": args.rollout_max_response_len,
            "apply_chat_template_kwargs": getattr(
                args, "apply_chat_template_kwargs", None
            ),
            "custom_generate_function_path": args.custom_generate_function_path,
            "custom_agent_function_path": getattr(
                args, "custom_agent_function_path", None
            ),
            "codex_reasoning_effort": getattr(
                args, "codex_reasoning_effort", None
            ),
            "agent_max_seq_len": getattr(args, "agent_max_seq_len", None),
            "use_session_server": args.use_session_server,
            "session_server_ip": args.session_server_ip,
            "session_server_port": args.session_server_port,
            "tito_model": args.tito_model,
            "tito_allowed_append_roles": getattr(
                args, "tito_allowed_append_roles", None
            ),
            "tensor_parallel": args.tensor_parallel,
            "pipeline_parallel": args.pipeline_parallel,
            "expert_parallel": args.expert_parallel,
            "rollout_num_gpus_per_engine": args.rollout_num_gpus_per_engine,
            "sglang_tp_size": args.sglang_tp_size,
            "sglang_dp_size": args.sglang_dp_size,
            "sglang_ep_size": args.sglang_ep_size,
            "sglang_mem_fraction_static": args.sglang_mem_fraction_static,
            "sglang_attention_backend": args.sglang_attention_backend,
            "sglang_deterministic_inference": args.sglang_deterministic_inference,
            "sglang_page_size": args.sglang_page_size,
            "sglang_max_running_requests": args.sglang_max_running_requests,
            "sglang_chunked_prefill_size": args.sglang_chunked_prefill_size,
            "use_rollout_routing_replay": args.use_rollout_routing_replay,
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
            "cybergym_reward_scheme": os.environ.get(
                "CYBERGYM_REWARD_SCHEME", "binary"
            ),
            "cybergym_reward_view": os.environ.get("CYBERGYM_REWARD_VIEW", "train"),
        },
    }
    final_ack_timeout_s = getattr(namespace, "final_ack_timeout_s", None)
    if final_ack_timeout_s is not None:
        plan["final_ack_timeout_s"] = final_ack_timeout_s
    eval_checkpoint_host = getattr(namespace, "eval_checkpoint_host", None)
    eval_checkpoint_path = getattr(namespace, "eval_checkpoint_path", None)
    eval_checkpoint_sha256 = getattr(namespace, "eval_checkpoint_sha256", None)
    eval_checkpoint_size_bytes = getattr(
        namespace, "eval_checkpoint_size_bytes", None
    )
    eval_checkpoint_global_step = getattr(
        namespace, "eval_checkpoint_global_step", None
    )
    checkpoint_values = (
        eval_checkpoint_host,
        eval_checkpoint_path,
        eval_checkpoint_sha256,
        eval_checkpoint_size_bytes,
        eval_checkpoint_global_step,
    )
    if any(value is not None for value in checkpoint_values) and any(
        value is None for value in checkpoint_values
    ):
        raise HarnessError(
            "all remote eval checkpoint identity options are required together"
        )
    eval_interval = getattr(namespace, "eval_interval", None)
    eval_dataset_name = getattr(namespace, "eval_dataset_name", None)
    eval_options = {
        "temperature": getattr(namespace, "eval_temperature", None),
        "top_p": getattr(namespace, "eval_top_p", None),
        "max_prompt_len": getattr(namespace, "eval_max_prompt_len", None),
        "max_response_len": getattr(namespace, "eval_max_response_len", None),
        "max_context_len": getattr(namespace, "eval_max_context_len", None),
    }
    if eval_checkpoint_host is None:
        if (
            eval_interval is not None
            or eval_dataset_name is not None
            or getattr(namespace, "eval_samples_per_prompt", None) is not None
            or any(value is not None for value in eval_options.values())
        ):
            raise HarnessError("evaluation options require --eval-checkpoint-host")
    else:
        plan["eval_checkpoint"] = _eval_checkpoint_contract(
            eval_checkpoint_host,
            eval_checkpoint_path,
            eval_checkpoint_sha256,
            eval_checkpoint_size_bytes,
            eval_checkpoint_global_step,
        )
        eval_samples_per_prompt = getattr(
            namespace, "eval_samples_per_prompt", None
        )
        plan["learner"]["evaluation"] = {
            "eval_only": True,
            "dataset_name": eval_dataset_name,
            "data": data,
            "data_sha256": data_sha256,
            "interval": 1 if eval_interval is None else eval_interval,
            "samples_per_prompt": (
                1 if eval_samples_per_prompt is None else eval_samples_per_prompt
            ),
            "prompt_count": data_prompt_count,
            "skip_before_train": True,
            **eval_options,
        }
    jit_cache_root = getattr(namespace, "jit_cache_root", None)
    if jit_cache_root is not None:
        plan["jit_cache"] = _jit_cache_contract(jit_cache_root, plan)
    daemon_source_root = getattr(namespace, "secrlenv_source_root", None)
    if args.custom_agent_function_path in SECRLENV_AGENTS:
        required = {
            "--secrlenv-source-root": daemon_source_root,
            "--secrlenv-source-sha256": getattr(
                namespace, "secrlenv_source_sha256", None
            ),
            "--secrlenv-task-pack": getattr(namespace, "secrlenv_task_pack", None),
            "--secrlenv-task-pack-sha256": getattr(
                namespace, "secrlenv_task_pack_sha256", None
            ),
            "--secrlenv-operator-image": getattr(
                namespace, "secrlenv_operator_image", None
            ),
            "--secrlenv-operator-image-id": getattr(
                namespace, "secrlenv_operator_image_id", None
            ),
        }
        missing = [flag for flag, value in required.items() if not value]
        if missing:
            raise HarnessError(
                "secrlenv custom agent requires " + ", ".join(missing)
            )
        plan["secrlenv_daemon"] = {
            "source_root": daemon_source_root,
            "source_sha256": namespace.secrlenv_source_sha256,
            "task_pack": namespace.secrlenv_task_pack,
            "task_pack_sha256": namespace.secrlenv_task_pack_sha256,
            "state_root": f"/data/yeto-rl/secrlenv-runs/{run_id}",
            "bind": "127.0.0.1",
            "placement": "island-heads",
            "port": namespace.secrlenv_port,
            "operator_image": namespace.secrlenv_operator_image,
            "operator_image_id": namespace.secrlenv_operator_image_id,
            "max_active_episodes": namespace.secrlenv_max_active_episodes,
        }
    elif daemon_source_root is not None:
        raise HarnessError(
            "--secrlenv-source-root requires the secrlenv custom agent"
        )
    codex_binary = getattr(namespace, "codex_harness_binary", None)
    codex_manifest = getattr(namespace, "codex_package_manifest", None)
    codex_schema = getattr(namespace, "codex_app_server_schema", None)
    if args.custom_agent_function_path == CODEX_HARNESS_AGENT:
        plan["codex_harness"] = _codex_harness_contract(namespace, args)
    elif any(value is not None for value in (codex_binary, codex_manifest, codex_schema)):
        raise HarnessError(
            "Codex artifact options require the signed stock Codex agent"
        )
    if getattr(args, "expert_full_count", 0):
        plan["learner"].update(
            expert_full_count=args.expert_full_count,
            expert_full_lr=args.expert_full_lr,
            expert_selection_sha256=args.expert_selection_sha256,
            expert_selection_contract_sha256=(
                args.expert_selection_contract_sha256
            ),
        )
    _validate_plan(plan)
    _write_plan(plan_path, plan)
    print(f"prepared {plan_path}")
    print(f"syncer {syncer_address} on {syncer_host}")
    return plan_path


def _require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise HarnessError(f"{name} is required on the controller")


def _run(
    command: Sequence[str], *, capture: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(command), flush=True)
    return subprocess.run(list(command), check=check, text=True, capture_output=capture)


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


def _secrlenv_daemon_script(plan: dict[str, Any]) -> str:
    daemon = plan["secrlenv_daemon"]
    return f"""set -euo pipefail
{_remote_vars(plan)}
SOURCE={shlex.quote(daemon['source_root'])}
TASK_PACK={shlex.quote(daemon['task_pack'])}
STATE_ROOT={shlex.quote(daemon['state_root'])}
ENV_FILE="$HOME/{plan['remote_env_file']}"
test -d "$SOURCE/secrlenv_rl"
test -d "$TASK_PACK"
test -f "$ENV_FILE" && test ! -L "$ENV_FILE"
test "$(stat -c '%a' "$ENV_FILE")" = 600
SOURCE_SHA="$(cd "$SOURCE" && find secrlenv_rl -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{{print $1}}')"
test "$SOURCE_SHA" = {shlex.quote(daemon['source_sha256'])}
PYTHONPATH="$RUN/source:$SOURCE${{PYTHONPATH:+:$PYTHONPATH}}" \
  python3 -m yeto.rl.secrlenv_task_images \
    --task-pack "$TASK_PACK" \
    --expected-task-pack-sha256 {shlex.quote(daemon['task_pack_sha256'])} \
    --data-root /data
IMAGE_ID="$(docker image inspect --format '{{{{.Id}}}}' {shlex.quote(daemon['operator_image'])})"
test "$IMAGE_ID" = {shlex.quote(daemon['operator_image_id'])}
if [ -s "$STATE_ROOT/daemon.pid" ]; then
  PID="$(cat "$STATE_ROOT/daemon.pid")"
  ARGS="$(ps -p "$PID" -o args= 2>/dev/null || true)"
  case "$ARGS" in
    *secrlenv_rl.server*"$STATE_ROOT/state"*) ;;
    *) echo 'existing secrlenv daemon PID has the wrong identity' >&2; exit 1 ;;
  esac
else
  if python3 - <<'PY'
import socket
s = socket.socket()
try:
    s.bind(({daemon['bind']!r}, {daemon['port']}))
finally:
    s.close()
PY
  then :; else
    echo 'secrlenv daemon port is occupied by an unattested process' >&2
    exit 1
  fi
  mkdir -p "$STATE_ROOT/state"
  TOKEN_FILE="$STATE_ROOT/daemon.token"
  if [ ! -e "$TOKEN_FILE" ]; then
    umask 077
    python3 - <<'PY' >"$TOKEN_FILE"
import secrets

print(secrets.token_hex(32))
PY
  fi
  test -f "$TOKEN_FILE" && test ! -L "$TOKEN_FILE"
  test "$(stat -c '%a' "$TOKEN_FILE")" = 600
  cd "$SOURCE"
  nohup setsid env PYTHONPATH="$SOURCE${{PYTHONPATH:+:$PYTHONPATH}}" \
    python3 -m secrlenv_rl.server \
      --task-pack "$TASK_PACK" \
      --state-dir "$STATE_ROOT/state" \
      --repository-root "$SOURCE" \
      --token-file "$TOKEN_FILE" \
      --bind {shlex.quote(daemon['bind'])} \
      --port {daemon['port']} \
      --operator-image {shlex.quote(daemon['operator_image'])} \
      --max-active-episodes {daemon['max_active_episodes']} \
      >"$STATE_ROOT/daemon.log" 2>&1 < /dev/null &
  echo "$!" > "$STATE_ROOT/daemon.pid"
fi
python3 - <<'PY'
import json
import time
from urllib.request import urlopen

deadline = time.monotonic() + 60
last_error = None
while time.monotonic() < deadline:
    try:
        with urlopen("http://{daemon['bind']}:{daemon['port']}/healthz", timeout=2) as response:
            value = json.load(response)
        if (
            response.status == 200
            and value.get("ok") is True
            and value.get("task_pack_sha256") == {daemon['task_pack_sha256']!r}
            and value.get("max_active_episodes") == {daemon['max_active_episodes']}
        ):
            print("secrlenv_daemon=ready task_pack=" + value["task_pack_sha256"])
            raise SystemExit(0)
        last_error = "health identity mismatch"
    except Exception as exc:
        last_error = str(exc)
    time.sleep(1)
raise SystemExit("secrlenv daemon did not become ready: " + str(last_error))
PY
"""


def _start_secrlenv_daemons(plan: dict[str, Any]) -> None:
    if plan.get("secrlenv_daemon") is None:
        return
    for target in _secrlenv_daemon_hosts(plan):
        _ssh(plan, target, _secrlenv_daemon_script(plan))


def _stop_secrlenv_daemons(plan: dict[str, Any]) -> None:
    daemon = plan.get("secrlenv_daemon")
    if daemon is None:
        return
    for target in _secrlenv_daemon_hosts(plan):
        _ssh(
            plan,
            target,
            f"""set -euo pipefail
STATE_ROOT={shlex.quote(daemon['state_root'])}
if [ ! -s "$STATE_ROOT/daemon.pid" ]; then exit 0; fi
PID="$(cat "$STATE_ROOT/daemon.pid")"
ARGS="$(ps -p "$PID" -o args= 2>/dev/null || true)"
case "$ARGS" in
  *secrlenv_rl.server*"$STATE_ROOT/state"*) ;;
  '') exit 0 ;;
  *) echo 'refusing to stop a process with the wrong identity' >&2; exit 1 ;;
esac
kill -TERM -- -"$PID"
for _ in {{1..600}}; do
  kill -0 "$PID" 2>/dev/null || exit 0
  sleep 0.1
done
echo 'secrlenv daemon did not exit after SIGTERM' >&2
exit 1
""",
        )


def _secrlenv_daemon_status_script(plan: dict[str, Any]) -> str:
    daemon = plan.get("secrlenv_daemon")
    if daemon is None:
        return ""
    return f"""STATE_ROOT={shlex.quote(daemon['state_root'])}
if [ ! -s "$STATE_ROOT/daemon.pid" ]; then
  echo 'secrlenv_daemon=stopped'
else
  PID="$(cat "$STATE_ROOT/daemon.pid")"
  case "$PID" in
    ''|*[!0-9]*) echo 'secrlenv_daemon=identity-drifted' >&2; exit 1 ;;
  esac
  ARGS="$(ps -p "$PID" -o args= 2>/dev/null || true)"
  case "$ARGS" in
    *secrlenv_rl.server*"$STATE_ROOT/state"*) ;;
    '') echo 'secrlenv_daemon=stopped' ;;
    *) echo 'secrlenv_daemon=identity-drifted' >&2; exit 1 ;;
  esac
  if [ -n "$ARGS" ]; then
    python3 - "$PID" <<'PY' || exit 1
import json
import sys
from urllib.request import urlopen

try:
    with urlopen("http://{daemon['bind']}:{daemon['port']}/healthz", timeout=2) as response:
        value = json.load(response)
except Exception:
    print("secrlenv_daemon=unhealthy")
    raise SystemExit(1)
if not (
    response.status == 200
    and value.get("ok") is True
    and value.get("task_pack_sha256") == {daemon['task_pack_sha256']!r}
    and value.get("max_active_episodes") == {daemon['max_active_episodes']}
):
    print("secrlenv_daemon=identity-drifted")
    raise SystemExit(1)
print("secrlenv_daemon=running pid=" + sys.argv[1])
PY
  fi
fi
"""


def _all_hosts(plan: dict[str, Any]) -> list[str]:
    return [host for island in plan["islands"] for host in island["hosts"]]


def _secrlenv_daemon_hosts(plan: dict[str, Any]) -> list[str]:
    """Return the pinned daemon roster, retaining old-plan cleanup semantics."""

    daemon = plan.get("secrlenv_daemon")
    if daemon is None:
        return []
    placement = daemon.get("placement", "all-hosts")
    if placement == "all-hosts":
        return _all_hosts(plan)
    if placement == "island-heads":
        return [island["hosts"][0] for island in plan["islands"]]
    raise HarnessError("secrlenv daemon has an invalid placement")


def _deployment_hosts(plan: dict[str, Any]) -> list[str]:
    hosts = _all_hosts(plan)
    syncer_host = _syncer_host(plan)
    return hosts if syncer_host in hosts else [*hosts, syncer_host]


def _attest_local(plan: dict[str, Any]) -> None:
    from ..provenance import file_sha256, python_spec_sha256, verify_source_tree_sha256

    verify_source_tree_sha256(plan["source_sha256"])
    if _syncer_source_sha256() != plan["syncer_source_sha256"]:
        raise HarnessError("syncer source changed after the plan was prepared")
    tms_patch = plan.get("tms_preload_patch")
    if tms_patch is not None:
        patch_path = REPO_ROOT / tms_patch["source_path"]
        if (
            patch_path.is_symlink()
            or not patch_path.is_file()
            or file_sha256(patch_path) != tms_patch["binary_sha256"]
        ):
            raise HarnessError("TMS preload patch changed after the plan was prepared")
    if (
        python_spec_sha256(plan["learner"]["reward_function"], base_dir=REPO_ROOT)
        != plan["reward_sha256"]
    ):
        raise HarnessError("reward source changed after the plan was prepared")
    learner = plan["learner"]
    codex_harness = plan.get("codex_harness")
    if codex_harness is not None:
        _validate_codex_harness(codex_harness, learner)
        binary = Path(codex_harness["controller_binary_path"])
        manifest = Path(codex_harness["controller_package_manifest_path"])
        schema = Path(codex_harness["controller_app_server_schema_path"])
        for path, expected_size, expected_sha256, executable in (
            (
                binary,
                codex_harness["binary_size_bytes"],
                codex_harness["binary_sha256"],
                True,
            ),
            (manifest, None, codex_harness["package_manifest_sha256"], False),
            (schema, None, codex_harness["app_server_schema_sha256"], False),
        ):
            if (
                path.is_symlink()
                or not path.is_file()
                or (executable and not path.stat().st_mode & 0o111)
                or (expected_size is not None and path.stat().st_size != expected_size)
                or file_sha256(path) != expected_sha256
            ):
                raise HarnessError("Codex controller artifact changed after prepare")
        adapter_path = REPO_ROOT / CODEX_HARNESS_AGENT_PATH
        if (
            adapter_path.is_symlink()
            or not adapter_path.is_file()
            or file_sha256(adapter_path) != codex_harness["agent_source_sha256"]
            or _codex_adapter_identity()
            != {
                name: codex_harness[name]
                for name in (
                    "base_instructions_sha256",
                    "terminal_exec_tool_schema_sha256",
                    "submit_tool_schema_sha256",
                    "dynamic_tools_schema_sha256",
                )
            }
        ):
            raise HarnessError("Yeto Codex adapter changed after prepare")
    data_local_path = plan["learner"].get("data_local_path")
    if (
        data_local_path is not None
        and _local_data_sha256(data_local_path) != plan["learner"]["data_sha256"]
    ):
        raise HarnessError("local dataset changed after the plan was prepared")
    evaluation = plan["learner"].get("evaluation")
    if evaluation is not None and _local_jsonl_record_count(
        data_local_path
    ) != evaluation.get("prompt_count"):
        raise HarnessError("evaluation dataset record count changed after prepare")


def deploy(plan_path: str | Path) -> None:
    plan_file, plan = load_plan(plan_path)
    _require_program("ssh")
    _require_program("rsync")
    _attest_local(plan)
    digest = _plan_digest(plan)
    local_data = plan["learner"].get("data_local_path")
    eval_checkpoint = plan.get("eval_checkpoint")
    if eval_checkpoint is not None:
        _require_program("scp")
        source_path = eval_checkpoint["source_path"]
        _ssh(
            plan,
            eval_checkpoint["source_host"],
            f"""set -euo pipefail
SOURCE="$HOME/{source_path}"
test -f "$SOURCE" && test ! -L "$SOURCE"
test "$(stat -c %s "$SOURCE")" = {eval_checkpoint['size_bytes']}
printf '%s  %s\n' {shlex.quote(eval_checkpoint['sha256'])} "$SOURCE" | sha256sum --check -
python3 - "$SOURCE" {eval_checkpoint['global_step']} <<'PY'
import sys
from pathlib import Path

header = Path(sys.argv[1]).open("rb").read(12)
expected_step = int(sys.argv[2])
if len(header) != 12 or int.from_bytes(header[:4], "little") != {SYNCER_CHECKPOINT_MAGIC}:
    raise SystemExit("invalid syncer checkpoint header")
if int.from_bytes(header[4:12], "little") != expected_step:
    raise SystemExit("syncer checkpoint global step mismatch")
PY
""",
        )
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
        "compare-report/",
        "wandb/",
        "build/",
        "dist/",
    )
    learner_hosts = set(_all_hosts(plan))
    for target in _deployment_hosts(plan):
        deploy_data = local_data is not None and target in learner_hosts
        deploy_checkpoint = eval_checkpoint is not None and target == _syncer_host(plan)
        deploy_codex = plan.get("codex_harness") is not None and target in learner_hosts
        data_directory = '\nmkdir -p "$RUN/data"' if deploy_data else ""
        state_directory = '\nmkdir -p "$RUN/state"' if deploy_checkpoint else ""
        codex_directory = '\nmkdir -p "$RUN/codex"' if deploy_codex else ""
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
{data_directory}{state_directory}{codex_directory}
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
        if deploy_data:
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
        if deploy_codex:
            codex = plan["codex_harness"]
            for source_path, bundle_path in (
                (codex["controller_binary_path"], codex["bundle_binary_path"]),
                (
                    codex["controller_package_manifest_path"],
                    codex["bundle_package_manifest_path"],
                ),
                (
                    codex["controller_app_server_schema_path"],
                    codex["bundle_app_server_schema_path"],
                ),
            ):
                _run(
                    [
                        "rsync",
                        "-az",
                        "-e",
                        _rsync_shell(plan),
                        source_path,
                        f"{target}:{plan['remote_run']}/{bundle_path}",
                    ]
                )
            _ssh(
                plan,
                target,
                f"""set -euo pipefail
{_remote_vars(plan)}
CODEX="$RUN/{codex['bundle_binary_path']}"
MANIFEST="$RUN/{codex['bundle_package_manifest_path']}"
SCHEMA="$RUN/{codex['bundle_app_server_schema_path']}"
test -f "$CODEX" && test ! -L "$CODEX" && test -x "$CODEX"
test "$(stat -c %s "$CODEX")" = {codex['binary_size_bytes']}
printf '%s  %s\n' {shlex.quote(codex['binary_sha256'])} "$CODEX" | sha256sum --check -
test -f "$MANIFEST" && test ! -L "$MANIFEST"
printf '%s  %s\n' {shlex.quote(codex['package_manifest_sha256'])} "$MANIFEST" | sha256sum --check -
test -f "$SCHEMA" && test ! -L "$SCHEMA"
printf '%s  %s\n' {shlex.quote(codex['app_server_schema_sha256'])} "$SCHEMA" | sha256sum --check -
""",
            )
        if deploy_checkpoint:
            if eval_checkpoint["source_host"] == target:
                _ssh(
                    plan,
                    target,
                    f"""set -euo pipefail
{_remote_vars(plan)}
cp --reflink=auto -- "$HOME/{eval_checkpoint['source_path']}" "$RUN/state/state.ckpt"
""",
                )
            else:
                _run(
                    [
                        "scp",
                        "-3",
                        *plan.get("ssh_options", []),
                        (
                            f"{eval_checkpoint['source_host']}:"
                            f"{eval_checkpoint['source_path']}"
                        ),
                        f"{target}:{plan['remote_run']}/state/state.ckpt",
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
    print(f"deployed {plan['run_id']} to {len(_deployment_hosts(plan))} host(s)")


def _host_setup_script(plan: dict[str, Any], gpus_per_node: int) -> str:
    sglang = plan["sglang"]
    miles = plan["miles"]
    learner = plan["learner"]
    if learner.get("model_manifest_sha256"):
        model_manifest_check = (
            f"printf '%s  %s\\n' "
            f"{shlex.quote(learner['model_manifest_sha256'])} "
            f"{shlex.quote(learner['model'] + '/conversion_manifest.json')} "
            "| sha256sum --check -\n"
        )
    else:
        model_manifest_check = ""
    tms_patch = plan.get("tms_preload_patch")
    if tms_patch is not None:
        tms_patch_check = f"""TMS_PATCH="$RUN/source/{tms_patch['source_path']}"
test -f "$TMS_PATCH" && test ! -L "$TMS_PATCH"
printf '%s  %s\n' {shlex.quote(tms_patch['binary_sha256'])} "$TMS_PATCH" | sha256sum --check -
TMS_BASE_ACTUAL="$(docker run --rm --entrypoint sha256sum {shlex.quote(plan['docker_image'])} {shlex.quote(tms_patch['container_path'])} | awk '{{print $1}}')"
test "$TMS_BASE_ACTUAL" = {shlex.quote(tms_patch['base_binary_sha256'])}
"""
    else:
        tms_patch_check = ""
    jit_cache = plan.get("jit_cache")
    if jit_cache is not None:
        cache_names = " ".join(shlex.quote(name) for name, _ in JIT_CACHE_MOUNTS)
        jit_cache_setup = f"""JIT_CACHE_ROOT={shlex.quote(jit_cache['host_root'])}
JIT_CACHE_COMPAT={shlex.quote(jit_cache['compatibility_sha256'])}
JIT_RUNTIME_ID="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader,nounits | sort -u)"
test "$(printf '%s\n' "$JIT_RUNTIME_ID" | sed '/^$/d' | wc -l)" -eq 1
JIT_RUNTIME_SHA="$(printf '%s' "$JIT_RUNTIME_ID" | sha256sum | awk '{{print $1}}')"
JIT_CACHE_HOST="$JIT_CACHE_ROOT/$JIT_CACHE_COMPAT/$JIT_RUNTIME_SHA"
umask 077
mkdir -p "$JIT_CACHE_HOST"
test -d "$JIT_CACHE_HOST" && test ! -L "$JIT_CACHE_HOST"
test "$(realpath -m "$JIT_CACHE_HOST")" = "$JIT_CACHE_HOST"
case "$JIT_CACHE_HOST" in "$JIT_CACHE_ROOT/$JIT_CACHE_COMPAT/"*) ;; *) exit 1 ;; esac
chmod 0700 "$JIT_CACHE_HOST"
for JIT_CACHE_NAME in {cache_names}; do
  JIT_CACHE_DIR="$JIT_CACHE_HOST/$JIT_CACHE_NAME"
  mkdir -p "$JIT_CACHE_DIR"
  test -d "$JIT_CACHE_DIR" && test ! -L "$JIT_CACHE_DIR"
  test "$(realpath -m "$JIT_CACHE_DIR")" = "$JIT_CACHE_DIR"
  chmod 0700 "$JIT_CACHE_DIR"
done
printf '%s\n' "$JIT_CACHE_HOST" > "$RUN/control/jit-cache-host-path"
chmod 0600 "$RUN/control/jit-cache-host-path"
"""
    else:
        jit_cache_setup = ""
    return f"""set -euo pipefail
{_remote_vars(plan)}
test "$(cat "$RUN/control/plan.sha256")" = {shlex.quote(_plan_digest(plan))}
command -v docker >/dev/null
command -v git >/dev/null
command -v nvidia-smi >/dev/null
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge {gpus_per_node}
docker info >/dev/null
{model_manifest_check}mkdir -p "$HOME/.cache/huggingface" "$RUN/miles" "$RUN/sglang"
if ! docker image inspect {shlex.quote(plan['docker_image'])} >/dev/null 2>&1; then
  docker pull {shlex.quote(plan['docker_image'])}
fi
docker image inspect {shlex.quote(plan['docker_image'])} >/dev/null
{tms_patch_check}{jit_cache_setup}if [ ! -d "$RUN/miles/.git" ]; then
  rmdir "$RUN/miles"
  git clone --no-checkout {shlex.quote(MILES_REPOSITORY)} "$RUN/miles"
fi
git -C "$RUN/miles" remote set-url origin {shlex.quote(MILES_REPOSITORY)}
git -C "$RUN/miles" fetch --depth 1 origin {shlex.quote(miles['commit'])}
git -C "$RUN/miles" checkout --detach {shlex.quote(miles['commit'])}
if [ ! -d "$RUN/sglang/.git" ]; then
  rmdir "$RUN/sglang"
  git clone --no-checkout {shlex.quote(sglang['repository'])} "$RUN/sglang"
fi
git -C "$RUN/sglang" remote set-url origin {shlex.quote(SGLANG_REPOSITORY)}
git -C "$RUN/sglang" fetch --depth 1 origin {shlex.quote(sglang['commit'])}
git -C "$RUN/sglang" checkout --detach {shlex.quote(sglang['commit'])}
"""


def _syncer_host_setup_script(plan: dict[str, Any]) -> str:
    return f"""set -euo pipefail
{_remote_vars(plan)}
test "$(cat "$RUN/control/plan.sha256")" = {shlex.quote(_plan_digest(plan))}
command -v docker >/dev/null
command -v systemctl >/dev/null
command -v systemd-run >/dev/null
test -d /run/systemd/system
docker info >/dev/null
if ! docker image inspect {shlex.quote(plan['docker_image'])} >/dev/null 2>&1; then
  docker pull {shlex.quote(plan['docker_image'])}
fi
docker image inspect {shlex.quote(plan['docker_image'])} >/dev/null
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
    _ssh(plan, _syncer_host(plan), script)


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
        ("--learners", _learner_count(plan)),
        ("--quorum", _learner_count(plan)),
        ("--quorum-timeout-s", 900),
        ("--final-ack-timeout-s", plan.get("final_ack_timeout_s", 3600)),
        ("--grace-ms", 0),
        ("--pipeline", learner.get("pipeline", 1) if decoupled else 1),
        (
            "--sync-interval-steps",
            learner.get("local_horizon", 1) if decoupled else 0,
        ),
        ("--delta-correction", "none"),
        (
            "--total-steps",
            (
                learner.get("total_fragment_steps", learner["global_rounds"])
                if decoupled
                else learner["global_rounds"]
            ),
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


def _syncer_unit_name(plan: dict[str, Any]) -> str:
    return f"yeto-rl-syncer-{plan['run_id']}.service"


def _legacy_syncer_pid_function() -> str:
    return r"""legacy_syncer_pid() {
  [ ! -e "$UNIT_FILE" ] || return 1
  [ -s "$PID_FILE" ] || return 1
  PID="$(cat "$PID_FILE")"
  case "$PID" in
    ''|*[!0-9]*) return 1 ;;
  esac
  [ "$PID" -gt 1 ] || return 1
  EXPECTED_EXE="$(readlink -f "$SYNCER_BINARY" 2>/dev/null || true)"
  ACTUAL_EXE="$(readlink -f "/proc/$PID/exe" 2>/dev/null || true)"
  [ -n "$EXPECTED_EXE" ] && [ "$ACTUAL_EXE" = "$EXPECTED_EXE" ] || return 1
  [ -r "/proc/$PID/cmdline" ] || return 1
  tr '\0' '\n' < "/proc/$PID/cmdline" | grep -Fqx -- "$CHECKPOINT_PATH" || return 1
  tr '\0' '\n' < "/proc/$PID/cmdline" | grep -Fqx -- "$EVENT_TAPE" || return 1
  printf '%s\n' "$PID"
}
"""


def _syncer_runtime_identity_functions() -> str:
    """Shell helpers that bind the listener to this run's exact systemd unit."""

    return r"""syncer_unit_pid() {
  systemctl is-active --quiet "$UNIT" || return 1
  [ -s "$UNIT_FILE" ] && [ "$(cat "$UNIT_FILE")" = "$UNIT" ] || return 1
  PID="$(systemctl show --property=MainPID --value "$UNIT")"
  case "$PID" in
    ''|*[!0-9]*) return 1 ;;
  esac
  [ "$PID" -gt 1 ] && kill -0 "$PID" 2>/dev/null || return 1
  FRAGMENT="$(systemctl show --property=FragmentPath --value "$UNIT")"
  [ "$FRAGMENT" = "/run/systemd/transient/$UNIT" ] || return 1
  [ "$(systemctl show --property=Restart --value "$UNIT")" = no ] || return 1
  [ "$(systemctl show --property=NRestarts --value "$UNIT")" = 0 ] || return 1
  CONTROL_GROUP="$(systemctl show --property=ControlGroup --value "$UNIT")"
  case "$CONTROL_GROUP" in
    /system.slice/*) ;;
    *) return 1 ;;
  esac
  CGROUP_PROCS="/sys/fs/cgroup$CONTROL_GROUP/cgroup.procs"
  [ -r "$CGROUP_PROCS" ] && grep -Fqx -- "$PID" "$CGROUP_PROCS" || return 1
  [ -r "/proc/$PID/cmdline" ] || return 1
  mapfile -d '' -t ACTUAL_ARGV < "/proc/$PID/cmdline"
  START=-1
  for INDEX in "${!ACTUAL_ARGV[@]}"; do
    if [ "${ACTUAL_ARGV[$INDEX]}" = "$WRAPPER" ]; then
      START=$INDEX
      break
    fi
  done
  [ "$START" -ge 0 ] || return 1
  [ "$((${#ACTUAL_ARGV[@]} - START))" -eq "${#EXPECTED_UNIT_ARGV[@]}" ] || return 1
  for INDEX in "${!EXPECTED_UNIT_ARGV[@]}"; do
    [ "${ACTUAL_ARGV[$((START + INDEX))]}" = "${EXPECTED_UNIT_ARGV[$INDEX]}" ] || return 1
  done
  printf '%s\n' "$PID"
}

syncer_child_pid() {
  CHILD_PID=$1
  case "$CHILD_PID" in
    ''|*[!0-9]*) return 1 ;;
  esac
  [ "$CHILD_PID" -gt 1 ] && kill -0 "$CHILD_PID" 2>/dev/null || return 1
  EXPECTED_EXE="$(readlink -f "$SYNCER_BINARY" 2>/dev/null || true)"
  ACTUAL_EXE="$(readlink -f "/proc/$CHILD_PID/exe" 2>/dev/null || true)"
  [ -n "$EXPECTED_EXE" ] && [ "$ACTUAL_EXE" = "$EXPECTED_EXE" ] || return 1
  [ -r "/proc/$CHILD_PID/cmdline" ] || return 1
  mapfile -d '' -t CHILD_ARGV < "/proc/$CHILD_PID/cmdline"
  [ "${#CHILD_ARGV[@]}" -eq "${#EXPECTED_SYNCER_ARGV[@]}" ] || return 1
  for INDEX in "${!EXPECTED_SYNCER_ARGV[@]}"; do
    [ "${CHILD_ARGV[$INDEX]}" = "${EXPECTED_SYNCER_ARGV[$INDEX]}" ] || return 1
  done
  printf '%s\n' "$CHILD_PID"
}

syncer_listener_inodes() {
  PORT_HEX="$(printf '%04X' "$SYNCER_PORT")"
  for TABLE in /proc/net/tcp /proc/net/tcp6; do
    [ ! -r "$TABLE" ] || awk -v port="$PORT_HEX" '
      $4 == "0A" {
        split($2, address, ":")
        if (toupper(address[2]) == port) print $10
      }
    ' "$TABLE"
  done | sort -u
}

syncer_pid_socket_inodes() {
  SOCKET_PID=$1
  for FD in "/proc/$SOCKET_PID/fd"/*; do
    LINK="$(readlink "$FD" 2>/dev/null || true)"
    case "$LINK" in
      'socket:['*']')
        INODE="${LINK#socket:[}"
        printf '%s\n' "${INODE%]}"
        ;;
    esac
  done | sort -u
}

syncer_listener_owned_by_pid() {
  OWNER_PID="$(syncer_child_pid "$1")" || return 1
  LISTEN_INODES="$(syncer_listener_inodes)"
  [ -n "$LISTEN_INODES" ] || return 1
  OWNED_INODES="$(syncer_pid_socket_inodes "$OWNER_PID")"
  [ -n "$OWNED_INODES" ] || return 1
  while read -r INODE; do
    printf '%s\n' "$OWNED_INODES" | grep -Fqx -- "$INODE" || return 1
  done <<< "$LISTEN_INODES"
  printf '%s\n' "$OWNER_PID"
}

syncer_listener_owned_by_unit() {
  PID="$(syncer_unit_pid)" || return 1
  CONTROL_GROUP="$(systemctl show --property=ControlGroup --value "$UNIT")"
  CGROUP_PROCS="/sys/fs/cgroup$CONTROL_GROUP/cgroup.procs"
  LISTEN_INODES="$(syncer_listener_inodes)"
  [ -n "$LISTEN_INODES" ] || return 1
  while read -r INODE; do
    OWNER_COUNT=0
    while read -r CGPID; do
      case "$CGPID" in ''|*[!0-9]*) continue ;; esac
      if syncer_pid_socket_inodes "$CGPID" | grep -Fqx -- "$INODE"; then
        syncer_child_pid "$CGPID" >/dev/null || return 1
        OWNER_COUNT=$((OWNER_COUNT + 1))
      fi
    done < "$CGROUP_PROCS"
    [ "$OWNER_COUNT" -eq 1 ] || return 1
  done <<< "$LISTEN_INODES"
  printf '%s\n' "$PID"
}
"""


def _syncer_start_script(plan: dict[str, Any]) -> str:
    command = _shell_join_with_run(_syncer_argv(plan))
    unit = shlex.quote(_syncer_unit_name(plan))
    eval_checkpoint = plan.get("eval_checkpoint")
    checkpoint_check = ""
    if eval_checkpoint is not None:
        checkpoint_check = f"""test -f "$RUN/state/state.ckpt"
test "$(stat -c %s "$RUN/state/state.ckpt")" = {eval_checkpoint['size_bytes']}
printf '%s  %s\n' {shlex.quote(eval_checkpoint['sha256'])} "$RUN/state/state.ckpt" | sha256sum --check -
"""
    return f"""set -euo pipefail
{_remote_vars(plan)}
{checkpoint_check}command -v systemctl >/dev/null
command -v systemd-run >/dev/null
test -d /run/systemd/system
UNIT={unit}
UNIT_FILE="$RUN/state/syncer.unit"
PID_FILE="$RUN/state/syncer.pid"
EXIT_FILE="$RUN/state/syncer.exit"
WRAPPER="$RUN/state/run-syncer"
LOG_FILE="$RUN/state/syncer.log"
SYNCER_BINARY="$RUN/state/yeto-syncer"
CHECKPOINT_PATH="$RUN/state/state.ckpt"
EVENT_TAPE="$RUN/state/events.jsonl"
SYNCER_PORT={plan['syncer_port']}
EXPECTED_SYNCER_ARGV=({command})
EXPECTED_UNIT_ARGV=("$WRAPPER" "$EXIT_FILE" "${{EXPECTED_SYNCER_ARGV[@]}}")
{_legacy_syncer_pid_function()}
{_syncer_runtime_identity_functions()}
if systemctl is-active --quiet "$UNIT"; then
  PID="$(syncer_listener_owned_by_unit)" || {{
    echo "active syncer unit or listener identity drifted: $UNIT" >&2
    exit 1
  }}
  printf '%s\n' "$PID" > "$PID_FILE"
  echo "syncer already running unit=$UNIT pid=$PID"
  exit 0
fi
LEGACY_PID="$(legacy_syncer_pid || true)"
if [ -n "$LEGACY_PID" ] && [ ! -s "$EXIT_FILE" ]; then
  echo "legacy syncer already running pid=$LEGACY_PID"
  exit 0
fi
if [ -n "$LEGACY_PID" ]; then
  echo "running legacy syncer conflicts with recorded terminal state" >&2
  exit 1
fi
if [ -n "$(syncer_listener_inodes)" ]; then
  echo "stale or unrelated process already listens on syncer port $SYNCER_PORT" >&2
  exit 1
fi
rm -f "$PID_FILE"
# A failed transient unit remains loaded until reset. Give systemd a bounded
# interval to garbage-collect it before recreating this run's deterministic unit.
systemctl reset-failed "$UNIT" >/dev/null 2>&1 || true
for _ in {{1..50}}; do
  [ "$(systemctl show --property=LoadState --value "$UNIT" 2>/dev/null || true)" = not-found ] && break
  sleep 0.1
done
if [ "$(systemctl show --property=LoadState --value "$UNIT" 2>/dev/null || true)" != not-found ]; then
  echo "stale syncer unit could not be unloaded: $UNIT" >&2
  exit 1
fi
rm -f "$PID_FILE" "$EXIT_FILE"
cat > "$WRAPPER" <<'SH'
#!/usr/bin/env bash
set -uo pipefail

EXIT_FILE=$1
shift
CHILD=""

record_exit() {{
  local result=$1 code=$2 status=$3 tmp
  tmp="$EXIT_FILE.tmp.$$"
  umask 077
  printf 'result=%s\nexit_code=%s\nexit_status=%s\ntimestamp=%s\n' \
    "$result" "$code" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmp"
  mv -f "$tmp" "$EXIT_FILE"
}}

terminate() {{
  local signal=$1
  trap - HUP INT TERM
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
    kill -"$signal" "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
  fi
  record_exit signal killed "$signal"
  exit 0
}}

trap 'terminate 1' HUP
trap 'terminate 2' INT
trap 'terminate 15' TERM

"$@" &
CHILD=$!
wait "$CHILD"
RC=$?
if [ "$RC" -eq 0 ]; then
  record_exit success exited 0
elif [ "$RC" -gt 128 ]; then
  record_exit signal killed "$((RC - 128))"
else
  record_exit exit-code exited "$RC"
fi
exit "$RC"
SH
chmod 0700 "$WRAPPER"
printf '%s\n' "$UNIT" > "$UNIT_FILE"
chmod 0600 "$UNIT_FILE"
SYNCER_PYTHONPATH="$RUN/source${{PYTHONPATH:+:$PYTHONPATH}}"
systemd-run --quiet \
  --unit="$UNIT" \
  --service-type=exec \
  --property=Restart=no \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=30s \
  --property=StandardInput=null \
  --property="StandardOutput=append:$LOG_FILE" \
  --property="StandardError=append:$LOG_FILE" \
  --setenv="PYTHONPATH=$SYNCER_PYTHONPATH" \
  "$WRAPPER" "$EXIT_FILE" {command}
sleep 1
if ! systemctl is-active --quiet "$UNIT"; then
  systemctl show "$UNIT" \
    --property=ActiveState,SubState,Result,ExecMainCode,ExecMainStatus \
    --no-pager >&2 || true
  echo "syncer did not remain active after launch" >&2
  exit 1
fi
PID="$(systemctl show --property=MainPID --value "$UNIT")"
test "$PID" -gt 1
kill -0 "$PID"
printf '%s\n' "$PID" > "$PID_FILE"
chmod 0600 "$PID_FILE"
"""


def _start_syncer(plan: dict[str, Any]) -> None:
    _ssh(plan, _syncer_host(plan), _syncer_start_script(plan))


def _wait_for_syncer(plan: dict[str, Any], timeout_s: int = 120) -> None:
    host, port = _validate_address(plan["syncer_address"])
    command = _shell_join_with_run(_syncer_argv(plan))
    unit = shlex.quote(_syncer_unit_name(plan))
    _ssh(
        plan,
        _syncer_host(plan),
        f"""set -euo pipefail
{_remote_vars(plan)}
UNIT={unit}
UNIT_FILE="$RUN/state/syncer.unit"
WRAPPER="$RUN/state/run-syncer"
SYNCER_BINARY="$RUN/state/yeto-syncer"
CHECKPOINT_PATH="$RUN/state/state.ckpt"
EVENT_TAPE="$RUN/state/events.jsonl"
EXIT_FILE="$RUN/state/syncer.exit"
SYNCER_PORT={port}
EXPECTED_SYNCER_ARGV=({command})
EXPECTED_UNIT_ARGV=("$WRAPPER" "$EXIT_FILE" "${{EXPECTED_SYNCER_ARGV[@]}}")
{_syncer_runtime_identity_functions()}
SYNCER_ADDRESS_HOST={shlex.quote(host)}
command -v getent >/dev/null
command -v ip >/dev/null
RESOLVED_IPS="$(getent ahosts "$SYNCER_ADDRESS_HOST" | awk '{{print $1}}' | sort -u)"
LOCAL_IPS="$(ip -o addr show | awk '{{split($4, address, "/"); print address[1]}}' | sort -u)"
[ -n "$RESOLVED_IPS" ] && [ -n "$LOCAL_IPS" ] || {{
  echo "cannot resolve syncer address ownership" >&2
  exit 1
}}
ADDRESS_IS_LOCAL=0
while read -r RESOLVED_IP; do
  if printf '%s\n' "$LOCAL_IPS" | grep -Fqx -- "$RESOLVED_IP"; then
    ADDRESS_IS_LOCAL=1
  fi
done <<< "$RESOLVED_IPS"
[ "$ADDRESS_IS_LOCAL" -eq 1 ] || {{
  echo "syncer address does not resolve to the configured syncer host" >&2
  exit 1
}}
deadline=$((SECONDS + {timeout_s}))
until syncer_listener_owned_by_unit >/dev/null; do
  [ "$SECONDS" -lt "$deadline" ] || {{
    echo "syncer unit/listener identity did not become ready" >&2
    exit 1
  }}
  sleep 2
done
""",
    )
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
    if learner_id not in range(_learner_count(plan)):
        raise HarnessError("--learner-id is outside the fixed island roster")
    learner = plan["learner"]
    island = plan["islands"][learner_id]
    values = _argv(
        ["python3", "-m", "yeto.rl.learner"],
        ("--model", learner["model"]),
        ("--rl-model-recipe", learner.get("rl_model_recipe", "generic")),
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
        ("--tensor-parallel", learner.get("tensor_parallel", 1)),
        ("--pipeline-parallel", learner.get("pipeline_parallel", 1)),
        (
            "--rollout-num-gpus-per-engine",
            learner.get("rollout_num_gpus_per_engine", 1),
        ),
        (
            "--sglang-mem-fraction-static",
            learner.get("sglang_mem_fraction_static", 0.4),
        ),
        ("--lora-r", learner["lora_r"]),
        ("--lora-targets", learner["lora_targets"]),
        ("--inner-lr", learner["inner_lr"]),
        ("--seq-len", learner["seq_len"]),
        ("--seed", learner["seed"]),
        ("--wan-streams", learner["wan_streams"]),
        ("--miles-root", "/workspace/miles"),
    )
    if learner.get("wandb"):
        values.append("--wandb")
        values.extend(("--wandb-project", str(learner.get("wandb_project", "yeto"))))
        values.extend(("--wandb-mode", str(learner.get("wandb_mode", "online"))))
        if learner.get("wandb_entity"):
            values.extend(("--wandb-entity", str(learner["wandb_entity"])))
    evaluation = learner.get("evaluation")
    if evaluation is not None:
        values.append("--eval-only")
        values.extend(
            (
                "--eval-dataset-name",
                evaluation["dataset_name"],
                "--eval-data-sha256",
                evaluation["data_sha256"],
                "--eval-interval",
                str(evaluation["interval"]),
                "--eval-samples-per-prompt",
                str(evaluation["samples_per_prompt"]),
            )
        )
        for flag, name in (
            ("--eval-temperature", "temperature"),
            ("--eval-top-p", "top_p"),
            ("--eval-max-prompt-len", "max_prompt_len"),
            ("--eval-max-response-len", "max_response_len"),
            ("--eval-max-context-len", "max_context_len"),
        ):
            if evaluation.get(name) is not None:
                values.extend((flag, str(evaluation[name])))
    if learner.get("rollout_model") is not None:
        values.extend(("--rollout-model", learner["rollout_model"]))
        values.extend(
            (
                "--rollout-model-revision",
                learner["rollout_model_revision"],
            )
        )
    if learner.get("expert_full_count", 0):
        values.extend(
            (
                "--expert-full-count",
                str(learner["expert_full_count"]),
                "--expert-full-lr",
                str(learner["expert_full_lr"]),
                "--expert-selection-sha256",
                learner["expert_selection_sha256"],
                "--expert-selection-contract-sha256",
                learner["expert_selection_contract_sha256"],
            )
        )
    if learner.get("apply_chat_template_kwargs"):
        values.extend(
            (
                "--apply-chat-template-kwargs",
                json.dumps(
                    learner["apply_chat_template_kwargs"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
    if learner.get("dynamic_sampling_max_replacements") is not None:
        values.extend(
            (
                "--dynamic-sampling-max-replacements",
                str(learner["dynamic_sampling_max_replacements"]),
            )
        )
    if learner.get("secrlenv_max_infrastructure_replacements") is not None:
        values.extend(
            (
                "--secrlenv-max-infrastructure-replacements",
                str(learner["secrlenv_max_infrastructure_replacements"]),
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
    for flag, name in (
        ("--sglang-tp-size", "sglang_tp_size"),
        ("--sglang-dp-size", "sglang_dp_size"),
        ("--sglang-ep-size", "sglang_ep_size"),
        ("--sglang-attention-backend", "sglang_attention_backend"),
        ("--sglang-page-size", "sglang_page_size"),
        ("--sglang-max-running-requests", "sglang_max_running_requests"),
        ("--sglang-chunked-prefill-size", "sglang_chunked_prefill_size"),
    ):
        if learner.get(name) is not None:
            values.extend((flag, str(learner[name])))
    if learner.get("use_rollout_routing_replay"):
        values.append("--use-rollout-routing-replay")
    values.append(
        "--sglang-deterministic-inference"
        if learner.get("sglang_deterministic_inference", True)
        else "--no-sglang-deterministic-inference"
    )
    if learner.get("custom_generate_function_path"):
        values.extend(
            (
                "--custom-generate-function-path",
                learner["custom_generate_function_path"],
            )
        )
    if learner.get("custom_agent_function_path"):
        values.extend(
            (
                "--custom-agent-function-path",
                learner["custom_agent_function_path"],
            )
        )
    if plan.get("codex_harness") is not None:
        values.extend(
            (
                "--codex-harness-contract",
                _canonical_json(plan["codex_harness"]),
            )
        )
    if learner.get("agent_max_seq_len") is not None:
        values.extend(("--agent-max-seq-len", str(learner["agent_max_seq_len"])))
    if learner.get("use_session_server"):
        values.append("--use-session-server")
        if learner.get("session_server_ip"):
            values.extend(("--session-server-ip", learner["session_server_ip"]))
        if learner.get("session_server_port"):
            values.append("--session-server-port")
            values.extend(str(port) for port in learner["session_server_port"])
        if learner.get("tito_model"):
            values.extend(("--tito-model", learner["tito_model"]))
        if learner.get("tito_allowed_append_roles"):
            values.append("--tito-allowed-append-roles")
            values.extend(learner["tito_allowed_append_roles"])
    if learner["trust_remote_code"]:
        values.append("--trust-remote-code")
    return values


def _container_name(plan: dict[str, Any], learner_id: int, node_id: int) -> str:
    return f"yeto-rl-{plan['run_id']}-i{learner_id}-n{node_id}"


def _node_start_script(plan: dict[str, Any], learner_id: int, node_id: int) -> str:
    island = plan["islands"][learner_id]
    if node_id not in range(len(island["hosts"])):
        raise HarnessError("node ID is outside the island topology")
    daemon_on_node = island["hosts"][node_id] in _secrlenv_daemon_hosts(plan)
    name = _container_name(plan, learner_id, node_id)
    head = _target_host(island["hosts"][0])
    gpus = island["gpus_per_node"]
    gpu_request = '"device=' + ",".join(str(index) for index in range(gpus)) + '"'
    network_interface = shlex.quote(plan.get("network_interface", "eno3"))
    env_file = (
        f' --env-file "$HOME/{plan["remote_env_file"]}"'
        if plan.get("remote_env_file")
        else ""
    )
    learner = plan["learner"]
    agent_env = (
        "  --env MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 \\\n"
        if learner.get("custom_agent_function_path")
        else ""
    )
    # The run id is the fleet's W&B group, so both islands land on one view.
    wandb_env = (
        f"  --env YETO_RUN_GROUP={shlex.quote(str(plan['run_id']))} \\\n"
        if learner.get("wandb")
        else ""
    )
    daemon_env = (
        "  --env SECRLENV_DAEMON_URL="
        f"http://127.0.0.1:{plan['secrlenv_daemon']['port']} "
        "--env SECRLENV_TASK_PACK_SHA256="
        f"{plan['secrlenv_daemon']['task_pack_sha256']} "
        "--env SECRLENV_BEARER_TOKEN_FILE=/run/secrlenv/daemon.token \\\n"
        if daemon_on_node
        else ""
    )
    daemon_volume = (
        "  --volume "
        f"{shlex.quote(plan['secrlenv_daemon']['state_root'])}/daemon.token:"
        "/run/secrlenv/daemon.token:ro \\\n"
        if daemon_on_node
        else ""
    )
    reward_scheme = shlex.quote(learner.get("cybergym_reward_scheme", "binary"))
    reward_view = shlex.quote(learner.get("cybergym_reward_view", "train"))
    data_volume = (
        '  --volume "$RUN/data:/workspace/data:ro" \\\n'
        if learner.get("data_local_path") is not None
        else ""
    )
    model_volumes = "".join(
        f"  --volume {shlex.quote(f'{mount}:{mount}:ro')} \\\n"
        for mount in learner.get("model_mounts", [])
    )
    codex = plan.get("codex_harness")
    if codex is not None:
        codex_contract_sha256 = _plan_digest(codex)
        codex_setup = f"""CODEX_HOST="$RUN/{codex['bundle_binary_path']}"
CODEX_MANIFEST_HOST="$RUN/{codex['bundle_package_manifest_path']}"
CODEX_SCHEMA_HOST="$RUN/{codex['bundle_app_server_schema_path']}"
test -f "$CODEX_HOST" && test ! -L "$CODEX_HOST" && test -x "$CODEX_HOST"
test "$(stat -c %s "$CODEX_HOST")" = {codex['binary_size_bytes']}
printf '%s  %s\n' {shlex.quote(codex['binary_sha256'])} "$CODEX_HOST" | sha256sum --check -
test -f "$CODEX_MANIFEST_HOST" && test ! -L "$CODEX_MANIFEST_HOST"
printf '%s  %s\n' {shlex.quote(codex['package_manifest_sha256'])} "$CODEX_MANIFEST_HOST" | sha256sum --check -
test -f "$CODEX_SCHEMA_HOST" && test ! -L "$CODEX_SCHEMA_HOST"
printf '%s  %s\n' {shlex.quote(codex['app_server_schema_sha256'])} "$CODEX_SCHEMA_HOST" | sha256sum --check -
"""
        codex_env = (
            "  --env YETO_CODEX_BINARY_PATH="
            f"{shlex.quote(codex['container_binary_path'])} "
            "--env YETO_CODEX_BINARY_SHA256="
            f"{codex['binary_sha256']} "
            f"--env YETO_CODEX_BINARY_SIZE_BYTES={codex['binary_size_bytes']} "
            f"--env YETO_CODEX_VERSION={shlex.quote(codex['cli_version'])} "
            "--env YETO_CODEX_APP_SERVER_PROTOCOL_REVISION="
            f"{codex['app_server_protocol_revision']} "
            "--env YETO_CODEX_APP_SERVER_SCHEMA_SHA256="
            f"{codex['app_server_schema_sha256']} "
            "--env YETO_CODEX_BASE_INSTRUCTIONS_SHA256="
            f"{codex['base_instructions_sha256']} "
            "--env YETO_CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256="
            f"{codex['terminal_exec_tool_schema_sha256']} "
            "--env YETO_CODEX_SUBMIT_TOOL_SCHEMA_SHA256="
            f"{codex['submit_tool_schema_sha256']} "
            "--env YETO_CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256="
            f"{codex['dynamic_tools_schema_sha256']} "
            f"--env YETO_CODEX_REASONING_EFFORT={codex['reasoning_effort']} "
            "--env YETO_CODEX_BACKEND_MAX_TOKENS="
            f"{codex['backend']['max_tokens']} "
            "--env YETO_CODEX_BACKEND_REASONING_EFFORT="
            f"{codex['backend']['reasoning_effort']} "
            "--env YETO_CODEX_BACKEND_THINKING="
            f"{codex['backend']['thinking']['type']} "
            f"--env YETO_CODEX_CHAT_TEMPLATE={codex['backend']['chat_template']} "
            "--env YETO_CODEX_CHAT_TEMPLATE_KWARGS="
            f"{shlex.quote(_canonical_json(codex['backend']['chat_template_kwargs']))} "
            "--env YETO_CODEX_TITO_ALLOWED_APPEND_ROLES=tool,user "
            f"--env YETO_CODEX_HARNESS_CONTRACT_SHA256={codex_contract_sha256} \\\n"
        )
        codex_volumes = (
            "  --volume \"$CODEX_HOST:"
            f"{codex['container_binary_path']}:ro\" \\\n"
            "  --volume \"$CODEX_MANIFEST_HOST:/opt/yeto/codex/"
            "codex-package.json:ro\" \\\n"
            "  --volume \"$CODEX_SCHEMA_HOST:"
            f"{codex['container_app_server_schema_path']}:ro\" \\\n"
        )
    else:
        codex_setup = ""
        codex_env = ""
        codex_volumes = ""
    tms_patch = plan.get("tms_preload_patch")
    tms_patch_volume = (
        "  --volume "
        f'"$RUN/source/{tms_patch["source_path"]}:'
        f'{tms_patch["container_path"]}:ro" \\\n'
        if tms_patch is not None
        else ""
    )
    tms_train_disk_backup = plan.get("tms_train_disk_backup")
    if tms_train_disk_backup is not None:
        target = island["hosts"][node_id]
        host_path = tms_train_disk_backup["node_paths"][target]
        container_path = tms_train_disk_backup["container_path"]
        chunk_mb = tms_train_disk_backup["chunk_mb"]
        tms_disk_setup = f"""TMS_DISK_BACKUP_HOST={shlex.quote(host_path)}
mkdir -p "$TMS_DISK_BACKUP_HOST"
test -d "$TMS_DISK_BACKUP_HOST" && test ! -L "$TMS_DISK_BACKUP_HOST"
test "$(realpath -m "$TMS_DISK_BACKUP_HOST")" = "$TMS_DISK_BACKUP_HOST"
"""
        tms_disk_env = (
            "  --env "
            f"MILES_TMS_TRAIN_DISK_BACKUP_DIR={shlex.quote(container_path)} "
            "--env "
            f"MILES_TMS_TRAIN_DISK_BACKUP_CHUNK_MB={chunk_mb} "
        )
        tms_disk_volume = (
            '  --volume "$TMS_DISK_BACKUP_HOST:'
            f'{container_path}" '
        )
    else:
        tms_disk_setup = ""
        tms_disk_env = ""
        tms_disk_volume = ""
    jit_cache = plan.get("jit_cache")
    if jit_cache is not None:
        jit_cache_prefix = (
            PurePosixPath(jit_cache["host_root"])
            / jit_cache["compatibility_sha256"]
        ).as_posix()
        cache_names = " ".join(shlex.quote(name) for name, _ in JIT_CACHE_MOUNTS)
        jit_cache_setup = f"""JIT_CACHE_FILE="$RUN/control/jit-cache-host-path"
test -f "$JIT_CACHE_FILE" && test ! -L "$JIT_CACHE_FILE"
JIT_CACHE_HOST="$(cat "$JIT_CACHE_FILE")"
case "$JIT_CACHE_HOST" in {shlex.quote(jit_cache_prefix + '/')}*) ;; *) exit 1 ;; esac
test -d "$JIT_CACHE_HOST" && test ! -L "$JIT_CACHE_HOST"
test "$(realpath -m "$JIT_CACHE_HOST")" = "$JIT_CACHE_HOST"
for JIT_CACHE_NAME in {cache_names}; do
  JIT_CACHE_DIR="$JIT_CACHE_HOST/$JIT_CACHE_NAME"
  test -d "$JIT_CACHE_DIR" && test ! -L "$JIT_CACHE_DIR"
  test "$(realpath -m "$JIT_CACHE_DIR")" = "$JIT_CACHE_DIR"
done
"""
        jit_cache_volumes = "".join(
            "  --volume "
            f'"$JIT_CACHE_HOST/{name}:{container_path}" \\\n'
            for name, container_path in JIT_CACHE_MOUNTS
        )
    else:
        jit_cache_setup = ""
        jit_cache_volumes = ""
    expert_full = bool(learner.get("expert_full_count", 0))
    if learner.get("rl_model_recipe") == "deepseek-v4-flash":
        attention_env = ""
        expert_env = (
            "--env YETO_DSV4_EXPERT_FULL=1 "
            f"--env YETO_DSV4_EXPERT_FULL_COUNT={learner['expert_full_count']} "
            f"--env YETO_DSV4_EXPERT_FULL_LR={learner['expert_full_lr']} "
            "--env NVTE_GROUPED_LINEAR_SINGLE_PARAM=0 "
            if expert_full
            else "--env YETO_DSV4_CLONE_ONLY_LORA=1 "
        )
        recipe_env = (
            "  --env YETO_DSV4_EXPERT_CLONE=1 "
            f"{expert_env}"
            "--env SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1 "
            "--env SGLANG_DSV4_FP4_EXPERTS=0 "
            "--env SGLANG_HEALTH_CHECK_TIMEOUT=120 "
            "--env SGLANG_DG_CACHE_DIR_PER_PROCESS=1 "
            "--env SGLANG_OPT_FP8_WO_A_GEMM=0 "
            "--env SGLANG_OPT_FUSE_WQA_WKV=0 \\\n"
        )
        diagnostics_env = (
            "  --env PYTHONFAULTHANDLER=1 "
            "--env TORCH_SHOW_CPP_STACKTRACES=1 "
            "--env TORCH_DISABLE_ADDR2LINE=1 "
            "--env TORCH_NCCL_DUMP_ON_TIMEOUT=1 "
            "--env TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 "
            "--env TORCH_FR_BUFFER_SIZE=1048576 "
            "--env NCCL_DEBUG=INFO "
            "--env NCCL_DEBUG_SUBSYS=INIT,NET "
            "--env YETO_TMS_POST_PAUSE_IDLE_S=30 \\\n"
        )
    else:
        attention_env = (
            "  --env NVTE_FLASH_ATTN=0 --env NVTE_FUSED_ATTN=0 "
            "--env NVTE_UNFUSED_ATTN=1 \\\n"
        )
        recipe_env = ""
        diagnostics_env = ""
    common = (
        f"python3 -m pip install -q --no-deps 'peft=={MILES_PEFT_VERSION}'; "
        "export PYTHONPATH=/workspace/sglang/python:/workspace/yeto:/workspace/miles${PYTHONPATH:+:$PYTHONPATH}; "
        f"NETWORK_INTERFACE={network_interface}; "
        "NODE_IP=$(python3 -c 'import fcntl, socket, struct, sys; "
        "sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); "
        "data = fcntl.ioctl(sock.fileno(), 0x8915, "
        "struct.pack(\"256s\", sys.argv[1].encode()[:15])); "
        "print(socket.inet_ntoa(data[20:24]))' \"$NETWORK_INTERFACE\"); "
        "test -n \"$NODE_IP\"; "
        "export MILES_HOST_IP=\"$NODE_IP\"; "
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
            + 'HEAD_IP="$NODE_IP"; '
            + 'ray start --head --node-ip-address="$NODE_IP" '
            f"--port={RAY_PORT} --include-dashboard=true; "
            + wait
            + "exec "
            + shlex.join(_learner_argv(plan, learner_id))
        )
    else:
        command = (
            common
            + f"until ray start --address={shlex.quote(head)}:{RAY_PORT} "
            + '--node-ip-address="$NODE_IP"; do sleep 2; done; '
            + f"while ray status --address={shlex.quote(head)}:{RAY_PORT} >/dev/null 2>&1; "
            "do sleep 5; done"
        )
    return f"""set -euo pipefail
{_remote_vars(plan)}
if docker inspect {shlex.quote(name)} >/dev/null 2>&1; then
  echo 'refusing to reuse a same-name learner container; use status or explicit restart-learner' >&2
  exit 1
fi
{codex_setup}{tms_disk_setup}{jit_cache_setup}mkdir -p "$RUN/island-{learner_id}"/{{state,output,audit,cores}}
docker run --detach \
  --name {shlex.quote(name)} \
  --gpus {shlex.quote(gpu_request)} \
  --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  --ulimit stack=67108864 --ulimit core=-1 \
  --env PYTHONUNBUFFERED=1 \
  --env HF_HOME=/workspace/hf \
  --env HF_HUB_ENABLE_HF_TRANSFER=1{env_file} \
  --env NCCL_IB_DISABLE=1 --env NCCL_SOCKET_IFNAME={network_interface} \
  --env GLOO_SOCKET_IFNAME={network_interface} --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
{attention_env}{recipe_env}{diagnostics_env}{tms_disk_env}{agent_env}{wandb_env}{daemon_env}{codex_env}  --env CYBERGYM_URL={shlex.quote(learner['cybergym_url'])} \
  --env CYBERGYM_AGENT_ID={shlex.quote(learner['cybergym_agent_id'])} \
  --env CYBERGYM_TIMEOUT={shlex.quote(str(learner['cybergym_timeout']))} \
  --env CYBERGYM_REWARD_SCHEME={reward_scheme} \
  --env CYBERGYM_REWARD_VIEW={reward_view} \
{data_volume}{model_volumes}{tms_patch_volume}{tms_disk_volume}{jit_cache_volumes}{daemon_volume}{codex_volumes}  --volume "$RUN/source:/workspace/yeto:ro" \
  --volume "$RUN/sglang:/workspace/sglang:ro" \
  --volume "$RUN/miles:/workspace/miles" \
  --volume "$RUN/island-{learner_id}/state:/workspace/state" \
  --volume "$RUN/island-{learner_id}/output:/workspace/output" \
  --volume "$RUN/island-{learner_id}/audit:/workspace/audit" \
  --volume "$RUN/island-{learner_id}/cores:/var/lib/vastai_kaalia/data" \
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
    _start_secrlenv_daemons(plan)
    for island in plan["islands"]:
        for target in island["hosts"]:
            _ssh(plan, target, _host_setup_script(plan, island["gpus_per_node"]))
    if _syncer_host(plan) not in _all_hosts(plan):
        _ssh(plan, _syncer_host(plan), _syncer_host_setup_script(plan))
    _build_syncer(plan)
    _start_syncer(plan)
    _wait_for_syncer(plan)
    for learner_id in range(_learner_count(plan)):
        _start_island(plan, learner_id)
    print(f"started Miles RL acceptance run {plan['run_id']}")


def _syncer_status_script(plan: dict[str, Any]) -> str:
    unit = shlex.quote(_syncer_unit_name(plan))
    command = _shell_join_with_run(_syncer_argv(plan))
    return f"""UNIT={unit}
UNIT_FILE="$RUN/state/syncer.unit"
PID_FILE="$RUN/state/syncer.pid"
EXIT_FILE="$RUN/state/syncer.exit"
WRAPPER="$RUN/state/run-syncer"
SYNCER_BINARY="$RUN/state/yeto-syncer"
CHECKPOINT_PATH="$RUN/state/state.ckpt"
EVENT_TAPE="$RUN/state/events.jsonl"
SYNCER_PORT={plan['syncer_port']}
EXPECTED_SYNCER_ARGV=({command})
EXPECTED_UNIT_ARGV=("$WRAPPER" "$EXIT_FILE" "${{EXPECTED_SYNCER_ARGV[@]}}")
{_legacy_syncer_pid_function()}
{_syncer_runtime_identity_functions()}
if command -v systemctl >/dev/null && systemctl is-active --quiet "$UNIT"; then
  PID="$(syncer_listener_owned_by_unit)" || {{
    echo "syncer=identity-drifted unit=$UNIT" >&2
    exit 1
  }}
  echo "syncer=running unit=$UNIT pid=$PID"
elif [ -s "$EXIT_FILE" ]; then
  RESULT=unknown
  EXIT_CODE=unknown
  EXIT_STATUS=unknown
  TIMESTAMP=unknown
  while IFS='=' read -r KEY VALUE; do
    case "$VALUE" in
      *[!A-Za-z0-9_.:+-]*|'') VALUE=unknown ;;
    esac
    case "$KEY" in
      result) RESULT=$VALUE ;;
      exit_code) EXIT_CODE=$VALUE ;;
      exit_status) EXIT_STATUS=$VALUE ;;
      timestamp) TIMESTAMP=$VALUE ;;
    esac
  done < "$EXIT_FILE"
  echo "syncer=stopped unit=$UNIT result=$RESULT exit_code=$EXIT_CODE exit_status=$EXIT_STATUS timestamp=$TIMESTAMP"
elif LEGACY_PID="$(legacy_syncer_pid || true)" && [ -n "$LEGACY_PID" ]; then
  PID="$(syncer_listener_owned_by_pid "$LEGACY_PID")" || {{
    echo "syncer=legacy-identity-drifted" >&2
    exit 1
  }}
  echo "syncer=running mode=legacy pid=$PID"
elif command -v systemctl >/dev/null && systemctl show "$UNIT" >/dev/null 2>&1; then
  echo "syncer=identity-unverified unit=$UNIT" >&2
  exit 1
else
  echo "syncer=stopped"
fi
"""


def status(plan_path: str | Path) -> None:
    _, plan = load_plan(plan_path)
    daemon_hosts = set(_secrlenv_daemon_hosts(plan))
    if _syncer_host(plan) not in _all_hosts(plan):
        result = _ssh(
            plan,
            _syncer_host(plan),
            f"""set -u
{_remote_vars(plan)}
echo host={shlex.quote(_syncer_host(plan))} role=syncer
{_syncer_status_script(plan)}
""",
            capture=True,
            check=False,
        )
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    for learner_id, island in enumerate(plan["islands"]):
        for node_id, target in enumerate(island["hosts"]):
            name = _container_name(plan, learner_id, node_id)
            syncer = ""
            if _syncer_host(plan) == target:
                syncer = _syncer_status_script(plan)
            daemon = (
                _secrlenv_daemon_status_script(plan)
                if target in daemon_hosts
                else ""
            )
            result = _ssh(
                plan,
                target,
                f"""set -u
{_remote_vars(plan)}
echo host={shlex.quote(target)} island={learner_id} node={node_id}
{syncer}{daemon}if docker inspect {shlex.quote(name)} >/dev/null 2>&1; then
  printf 'container='
  docker inspect --format {shlex.quote(_CONTAINER_STATE_FORMAT)} {shlex.quote(name)}
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
    if learner_id not in range(_learner_count(plan)):
        raise HarnessError("--learner-id is outside the fixed island roster")
    nodes = list(enumerate(plan["islands"][learner_id]["hosts"]))
    for node_id, target in reversed(nodes):
        name = _container_name(plan, learner_id, node_id)
        _ssh(plan, target, f"docker kill {shlex.quote(name)} >/dev/null 2>&1 || true")


def restart_learner(plan_path: str | Path, learner_id: int) -> None:
    _, plan = load_plan(plan_path)
    if learner_id not in range(_learner_count(plan)):
        raise HarnessError("--learner-id is outside the fixed island roster")
    _wait_for_syncer(plan)
    for node_id, target in enumerate(plan["islands"][learner_id]["hosts"]):
        name = _container_name(plan, learner_id, node_id)
        _ssh(plan, target, f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true")
    _start_island(plan, learner_id)


def kill_syncer(plan_path: str | Path) -> None:
    _, plan = load_plan(plan_path)
    unit = shlex.quote(_syncer_unit_name(plan))
    command = _shell_join_with_run(_syncer_argv(plan))
    _ssh(
        plan,
        _syncer_host(plan),
        f"""set -euo pipefail
{_remote_vars(plan)}
UNIT={unit}
UNIT_FILE="$RUN/state/syncer.unit"
PID_FILE="$RUN/state/syncer.pid"
EXIT_FILE="$RUN/state/syncer.exit"
WRAPPER="$RUN/state/run-syncer"
SYNCER_BINARY="$RUN/state/yeto-syncer"
CHECKPOINT_PATH="$RUN/state/state.ckpt"
EVENT_TAPE="$RUN/state/events.jsonl"
SYNCER_PORT={plan['syncer_port']}
EXPECTED_SYNCER_ARGV=({command})
EXPECTED_UNIT_ARGV=("$WRAPPER" "$EXIT_FILE" "${{EXPECTED_SYNCER_ARGV[@]}}")
{_legacy_syncer_pid_function()}
{_syncer_runtime_identity_functions()}
if command -v systemctl >/dev/null && systemctl is-active --quiet "$UNIT"; then
  PID="$(syncer_listener_owned_by_unit)" || {{
    echo "refusing to kill a syncer unit with drifted identity" >&2
    exit 1
  }}
  systemctl kill --signal=SIGKILL --kill-who=all "$UNIT"
  TMP="$EXIT_FILE.tmp.$$"
  umask 077
  printf 'result=signal\nexit_code=killed\nexit_status=9\ntimestamp=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TMP"
  mv -f "$TMP" "$EXIT_FILE"
  for _ in {{1..50}}; do
    if ! systemctl is-active --quiet "$UNIT" \
      && ! kill -0 "$PID" 2>/dev/null \
      && [ -z "$(syncer_listener_inodes)" ]; then
      rm -f "$PID_FILE"
      exit 0
    fi
    sleep 0.1
  done
  echo "syncer unit did not exit after SIGKILL: $UNIT pid=$PID" >&2
  exit 1
fi
PID="$(legacy_syncer_pid || true)"
if [ -z "$PID" ]; then
  rm -f "$PID_FILE"
  echo "syncer is not running" >&2
  exit 1
fi
PID="$(syncer_listener_owned_by_pid "$PID")" || {{
  echo "refusing to kill a legacy syncer with drifted identity" >&2
  exit 1
}}
kill -KILL -- -"$PID"
for _ in {{1..50}}; do
  if ! kill -0 "$PID" 2>/dev/null && [ -z "$(syncer_listener_inodes)" ]; then
    rm -f "$PID_FILE"
    exit 0
  fi
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


def _syncer_stop_script(plan: dict[str, Any]) -> str:
    unit = shlex.quote(_syncer_unit_name(plan))
    command = _shell_join_with_run(_syncer_argv(plan))
    return f"""set -euo pipefail
{_remote_vars(plan)}
UNIT={unit}
UNIT_FILE="$RUN/state/syncer.unit"
PID_FILE="$RUN/state/syncer.pid"
EXIT_FILE="$RUN/state/syncer.exit"
WRAPPER="$RUN/state/run-syncer"
SYNCER_BINARY="$RUN/state/yeto-syncer"
CHECKPOINT_PATH="$RUN/state/state.ckpt"
EVENT_TAPE="$RUN/state/events.jsonl"
SYNCER_PORT={plan['syncer_port']}
EXPECTED_SYNCER_ARGV=({command})
EXPECTED_UNIT_ARGV=("$WRAPPER" "$EXIT_FILE" "${{EXPECTED_SYNCER_ARGV[@]}}")
{_legacy_syncer_pid_function()}
{_syncer_runtime_identity_functions()}
LOAD_STATE="$(systemctl show --property=LoadState --value "$UNIT" 2>/dev/null || true)"
if systemctl is-active --quiet "$UNIT"; then
  PID="$(syncer_listener_owned_by_unit)" || {{
    echo "refusing to stop a syncer unit with drifted identity" >&2
    exit 1
  }}
  systemctl stop "$UNIT"
  STOPPED=0
  for _ in {{1..100}}; do
    ACTIVE_STATE="$(systemctl show --property=ActiveState --value "$UNIT" 2>/dev/null || true)"
    if {{ [ -z "$ACTIVE_STATE" ] || [ "$ACTIVE_STATE" = inactive ] || [ "$ACTIVE_STATE" = failed ]; }} \
      && ! kill -0 "$PID" 2>/dev/null \
      && [ -z "$(syncer_listener_inodes)" ]; then
      STOPPED=1
      break
    fi
    sleep 0.1
  done
  if [ "$STOPPED" -ne 1 ]; then
    echo "syncer unit did not become inactive with its process dead: $UNIT pid=$PID" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
  exit 0
fi
if [ -n "$LOAD_STATE" ] && [ "$LOAD_STATE" != not-found ]; then
  if [ -s "$EXIT_FILE" ] && [ -z "$(syncer_listener_inodes)" ]; then
    rm -f "$PID_FILE"
    exit 0
  fi
  echo "refusing to act on an inactive syncer unit without exact run evidence" >&2
  exit 1
fi
LEGACY_PID="$(legacy_syncer_pid || true)"
if [ -n "$LEGACY_PID" ]; then
  LEGACY_PID="$(syncer_listener_owned_by_pid "$LEGACY_PID")" || {{
    echo "refusing to stop a legacy syncer with drifted identity" >&2
    exit 1
  }}
  kill -TERM -- -"$LEGACY_PID"
  for _ in {{1..100}}; do
    if ! kill -0 "$LEGACY_PID" 2>/dev/null \
      && [ -z "$(syncer_listener_inodes)" ]; then
      rm -f "$PID_FILE"
      exit 0
    fi
    sleep 0.1
  done
  echo "legacy syncer did not exit after SIGTERM: pid=$LEGACY_PID" >&2
  exit 1
fi
rm -f "$PID_FILE"
"""


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
        _syncer_host(plan),
        _syncer_stop_script(plan),
    )
    _stop_secrlenv_daemons(plan)


def _collect_event_evidence(
    plan: dict[str, Any],
    target: str,
    remote_relative_path: str,
    mode: str,
    destination: Path,
) -> None:
    relative = PurePosixPath(remote_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise HarnessError("event evidence path must stay inside the run root")
    result = _ssh(
        plan,
        target,
        f"{_remote_vars(plan)}\n"
        "python3 -c "
        f"{shlex.quote(_EVENT_EVIDENCE_FILTER)} "
        f'"$RUN/{relative.as_posix()}" {shlex.quote(mode)}',
        capture=True,
    )
    payload = result.stdout.encode()
    if len(payload) > 64 * 1024 * 1024:
        raise HarnessError("filtered event evidence is unexpectedly large")
    try:
        records = [json.loads(line) for line in result.stdout.splitlines() if line]
    except json.JSONDecodeError as error:
        raise HarnessError("filtered event evidence is malformed") from error
    schema = _EVENT_EVIDENCE_FIELDS.get(mode)
    if schema is None:
        raise HarnessError("filtered event evidence uses an unknown mode")
    for record in records:
        if not isinstance(record, dict):
            raise HarnessError("filtered event evidence contains a non-object")
        event_name = "syncer_commit" if mode == "syncer" else record.get("event")
        if event_name not in schema or set(record) != schema[event_name]:
            raise HarnessError("filtered event evidence has unapproved fields")
        for name, value in record.items():
            if name in {
                "event",
                "dataset_name",
                "sync/global_policy_hash",
                "rl/policy_hash",
                "sync/layout_hash",
            } and (not isinstance(value, str) or len(value) > 128):
                raise HarnessError("filtered event evidence has an unsafe string")
    _atomic_bytes(destination, payload)


def _parse_syncer_lifecycle_evidence(raw: str) -> dict[str, int]:
    if len(raw.encode()) > 4096:
        raise HarnessError("syncer lifecycle evidence is unexpectedly large")
    try:
        evidence = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise HarnessError("syncer lifecycle evidence is malformed") from error
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema",
        "resume_count",
        "final_ack_count",
    }:
        raise HarnessError("syncer lifecycle evidence has unapproved fields")
    if (
        evidence.get("schema") != 1
        or any(
            isinstance(evidence.get(name), bool)
            or not isinstance(evidence.get(name), int)
            or not 0 <= evidence[name] <= 1_000_000
            for name in ("resume_count", "final_ack_count")
        )
    ):
        raise HarnessError("syncer lifecycle evidence has invalid counters")
    return evidence


def _collect_capacity_preflight(plan: dict[str, Any], artifacts: Path) -> None:
    """Refuse a large training transfer before it can fill the controller disk."""

    if plan["learner"].get("evaluation") is not None:
        return
    total_bytes = (len(plan["islands"]) + 1) * 64 * 1024 * 1024
    if plan["learner"].get("sync_preset", "strict-avg") != "decoupled":
        for learner_id, island in enumerate(plan["islands"]):
            result = _ssh(
                plan,
                island["hosts"][0],
                f"""set -euo pipefail
{_remote_vars(plan)}
find "$RUN/island-{learner_id}/audit" -maxdepth 1 -type f -name 'round-*.json' -printf '%s\n'
find "$RUN/island-{learner_id}/audit" -maxdepth 1 -type f -name 'round-*.base.f32' -printf '%s\n'
find "$RUN/island-{learner_id}/audit" -maxdepth 1 -type f -name 'round-*.delta.f32' -printf '%s\n'
""",
                capture=True,
            )
            sizes = result.stdout.splitlines()
            if any(not re.fullmatch(r"[0-9]+", size) for size in sizes):
                raise HarnessError("cannot determine remote audit evidence size")
            total_bytes += sum(map(int, sizes))
    checkpoint = _ssh(
        plan,
        _syncer_host(plan),
        f"""set -euo pipefail
{_remote_vars(plan)}
stat -c %s "$RUN/state/state.ckpt"
""",
        capture=True,
    )
    if not re.fullmatch(r"[0-9]+\n?", checkpoint.stdout):
        raise HarnessError("cannot determine remote checkpoint size")
    total_bytes += int(checkpoint.stdout)
    reserve = 5 * 1024 * 1024 * 1024
    available = shutil.disk_usage(artifacts).free
    if total_bytes + reserve > available:
        raise HarnessError(
            "controller lacks space for privacy-safe training evidence; keep the "
            "checkpoint remote and launch the eval handoff there"
        )


def collect(plan_path: str | Path) -> Path:
    plan_file, plan = load_plan(plan_path)
    _require_program("ssh")
    _require_program("rsync")
    artifacts = plan_file.parent / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    _collect_capacity_preflight(plan, artifacts)
    eval_only = plan["learner"].get("evaluation") is not None
    sync_preset = plan["learner"].get("sync_preset", "strict-avg")
    learner_event_mode = (
        "eval-learner"
        if eval_only
        else (
            "decoupled-learner"
            if sync_preset == "decoupled"
            else "strict-learner"
        )
    )
    for learner_id, island in enumerate(plan["islands"]):
        destination = artifacts / f"island-{learner_id}"
        destination.mkdir(parents=True, exist_ok=True)
        for node_id, target in enumerate(island["hosts"]):
            name = _container_name(plan, learner_id, node_id)
            inspect = _ssh(
                plan,
                target,
                "docker inspect --format "
                f"{shlex.quote(_CONTAINER_STATE_FORMAT)} {shlex.quote(name)}",
                capture=True,
                check=False,
            )
            inspection = _parse_container_inspection(inspect.stdout)
            _atomic_bytes(
                destination / f"node-{node_id}.inspect.json",
                (_canonical_json(inspection) + "\n").encode(),
            )
        head = island["hosts"][0]
        output = destination / "output"
        output.mkdir(parents=True, exist_ok=True)
        _collect_event_evidence(
            plan,
            head,
            f"island-{learner_id}/output/events.jsonl",
            learner_event_mode,
            output / "events.jsonl",
        )
        if not eval_only and sync_preset != "decoupled":
            audit = destination / "audit"
            audit.mkdir(parents=True, exist_ok=True)
            audit_includes = [
                f"--include=/round-{step:08d}{suffix}"
                for step in range(1, plan["learner"]["global_rounds"] + 1)
                for suffix in (".json", ".base.f32", ".delta.f32")
            ]
            _run(
                [
                    "rsync",
                    "-az",
                    *audit_includes,
                    "--exclude=*",
                    "-e",
                    _rsync_shell(plan),
                    f"{head}:{plan['remote_run']}/island-{learner_id}/audit/",
                    f"{audit}/",
                ]
            )
    syncer = artifacts / "syncer"
    syncer.mkdir(parents=True, exist_ok=True)
    if eval_only:
        lifecycle = _ssh(
            plan,
            _syncer_host(plan),
            f"{_remote_vars(plan)}\n"
            "python3 -c "
            f"{shlex.quote(_SYNCER_LIFECYCLE_EVIDENCE)} "
            '"$RUN/state/syncer.log"',
            capture=True,
        )
        evidence = _parse_syncer_lifecycle_evidence(lifecycle.stdout)
        _atomic_bytes(
            syncer / "lifecycle-evidence.json",
            (_canonical_json(evidence) + "\n").encode(),
        )
        syncer_includes = ("--include=/syncer.exit",)
    else:
        _collect_event_evidence(
            plan,
            _syncer_host(plan),
            "state/events.jsonl",
            "syncer",
            syncer / "events.jsonl",
        )
        syncer_includes = (
            "--include=/state.ckpt",
            "--include=/syncer.exit",
        )
    _run(
        [
            "rsync",
            "-az",
            *syncer_includes,
            "--exclude=*",
            "-e",
            _rsync_shell(plan),
            f"{_syncer_host(plan)}:{plan['remote_run']}/state/",
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
    return (
        output
        if output.exists()
        else artifacts / f"island-{learner_id}" / "events.jsonl"
    )


def _verify_oracle(plan: dict[str, Any], checkpoint, artifacts: Path) -> str:
    rounds = plan["learner"]["global_rounds"]
    roster = list(range(_learner_count(plan)))
    syncer_events = [
        event
        for event in _json_lines(artifacts / "syncer" / "events.jsonl")
        if event.get("fragment") == 0 and "step" in event
    ]
    if [event.get("step") for event in syncer_events] != list(range(1, rounds + 1)):
        raise HarnessError("syncer tape is not one ordered commit per RL round")
    island_events = [
        _json_lines(_event_path(artifacts, learner_id)) for learner_id in roster
    ]
    expected_base = None
    specs = None
    identity = None
    for step, sync_event in enumerate(syncer_events, start=1):
        responders = sorted(
            sync_event.get("responders", []), key=lambda item: item.get("id", -1)
        )
        if (
            sync_event.get("launch_base_version") != step - 1
            or sync_event.get("expected") != roster
            or sync_event.get("responded") != roster
            or [item.get("id") for item in responders] != roster
            or any(
                item.get("base_version") != step - 1
                or item.get("c_steps") != 1
                or item.get("c_tokens") != 1
                or item.get("contribution") != 1.0 / len(roster)
                for item in responders
            )
        ):
            raise HarnessError(f"syncer commit v{step} violates fixed-roster f32 AVG")
        bases = []
        base_bytes = []
        deltas = []
        for learner_id in roster:
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
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
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
                or metadata.get("base_f32_sha256")
                != hashlib.sha256(raw_base).hexdigest()
                or metadata.get("delta_f32_sha256")
                != hashlib.sha256(raw_delta).hexdigest()
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
        if any(raw != base_bytes[0] for raw in base_bytes[1:]):
            raise HarnessError(f"learners did not use the same f32 base at v{step - 1}")
        if expected_base is not None and not torch.equal(bases[0], expected_base):
            raise HarnessError(f"oracle v{step - 1} is not the next exact f32 base")
        merged = torch.zeros_like(bases[0])
        weight = torch.tensor(1.0 / len(roster), dtype=torch.float32)
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
        != {learner_id: (rounds, rounds, rounds) for learner_id in roster}
        or not torch.equal(checkpoint.fragments[0][1], expected_base)
    ):
        raise HarnessError("authoritative checkpoint differs from the f32 oracle")
    # torch.equal deliberately treats +0.0 and -0.0 as the same f32 value.
    # Rust's outer-gradient sign flip can preserve a different zero sign than
    # the independently evaluated PyTorch expression even when the two cuts
    # are numerically exact. policy_hash is byte-exact, so after validating
    # the oracle above, hash the authoritative checkpoint bytes that learners
    # actually receive and apply.
    authoritative_final = checkpoint.fragments[0][1]
    final = canonical_state(
        rounds,
        tensors_from_flat(authoritative_final, specs),
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


def _verify_decoupled(plan: dict[str, Any], checkpoint, artifacts: Path) -> str:
    learner = plan["learner"]
    fragments = learner["fragments"]
    total_steps = learner["total_fragment_steps"]
    local_horizon = learner["local_horizon"]
    terminal_versions = tuple(range(total_steps - fragments + 1, total_steps + 1))
    versions = tuple(version for version, _, _ in checkpoint.fragments)
    if (
        checkpoint.global_step != total_steps
        or len(checkpoint.fragments) != fragments
        or versions != terminal_versions
        or not _SHA256.fullmatch(checkpoint.layout_hash or "")
    ):
        raise HarnessError("authoritative checkpoint is not a complete decoupled cut")

    syncer_events = sorted(
        (
            event
            for event in _json_lines(artifacts / "syncer" / "events.jsonl")
            if "step" in event
        ),
        key=lambda event: event.get("step", -1),
    )
    if [event.get("step") for event in syncer_events] != list(
        range(1, total_steps + 1)
    ):
        raise HarnessError("syncer tape is not one complete decoupled schedule")

    roster = list(range(_learner_count(plan)))
    ledger = {learner_id: [0, 0, 0] for learner_id in roster}
    for event in syncer_events:
        step = event["step"]
        if event.get("fragment") != (step - 1) % fragments:
            raise HarnessError("decoupled syncer tape violates fragment schedule")
        base_version = max(0, step - fragments)
        responders = sorted(
            event.get("responders", []), key=lambda item: item.get("id", -1)
        )
        if (
            event.get("launch_base_version") != base_version
            or event.get("expected") != roster
            or event.get("responded") != roster
            or event.get("sync/layout_hash") != checkpoint.layout_hash
            or [item.get("id") for item in responders] != roster
        ):
            raise HarnessError(
                f"decoupled syncer commit {step} violates the fixed roster"
            )
        for responder in responders:
            learner_id = responder["id"]
            c_steps = responder.get("c_steps")
            c_tokens = responder.get("c_tokens")
            if (
                responder.get("base_version") != base_version
                or not isinstance(c_steps, int)
                or c_steps < local_horizon
                or not isinstance(c_tokens, int)
                or c_tokens < 0
            ):
                raise HarnessError(
                    f"decoupled syncer commit {step} has invalid progress evidence"
                )
            ledger[learner_id][0] += 1
            ledger[learner_id][1] += c_steps
            ledger[learner_id][2] += c_tokens
    expected_ledger = {
        learner_id: tuple(progress) for learner_id, progress in ledger.items()
    }
    if checkpoint.ledger != expected_ledger:
        raise HarnessError("authoritative checkpoint differs from decoupled tape")

    final_hashes = []
    for learner_id in roster:
        events = _json_lines(_event_path(artifacts, learner_id))
        snapshots = [
            event
            for event in events
            if event.get("event") == "rl_policy_snapshot"
            and event.get("rl/fragment_versions") == list(terminal_versions)
        ]
        if not snapshots:
            raise HarnessError(f"learner {learner_id} lacks the final policy snapshot")
        final_hash = snapshots[-1].get("rl/policy_hash")
        if not isinstance(final_hash, str) or not _SHA256.fullmatch(final_hash):
            raise HarnessError(f"learner {learner_id} has an invalid final policy hash")
        if not any(
            event.get("event") == "rl_policy_apply"
            and event.get("sync/global_policy_hash") == final_hash
            for event in events
        ):
            raise HarnessError(f"learner {learner_id} did not apply the final policy")
        if not any(event.get("event") == "rl_final_cut" for event in events):
            raise HarnessError(
                f"learner {learner_id} did not acknowledge the final cut"
            )
        final_hashes.append(final_hash)
    if len(set(final_hashes)) != 1:
        raise HarnessError("learners disagree on the final decoupled policy hash")
    return final_hashes[0]


def _parse_container_inspection(raw: str) -> dict[str, Any]:
    """Accept only the four non-sensitive fields selected on the remote host."""

    if len(raw.encode()) > 4096:
        raise HarnessError("learner container state record is unexpectedly large")
    try:
        inspection = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise HarnessError("learner container state record is malformed") from error
    if not isinstance(inspection, dict) or set(inspection) != {
        "Status",
        "ExitCode",
        "OOMKilled",
        "RestartCount",
    }:
        raise HarnessError("learner container state record has unapproved fields")
    if (
        not isinstance(inspection["Status"], str)
        or not inspection["Status"]
        or not isinstance(inspection["ExitCode"], int)
        or isinstance(inspection["ExitCode"], bool)
        or inspection["ExitCode"] < 0
        or not isinstance(inspection["OOMKilled"], bool)
        or not isinstance(inspection["RestartCount"], int)
        or isinstance(inspection["RestartCount"], bool)
        or inspection["RestartCount"] < 0
    ):
        raise HarnessError("learner container state record has invalid values")
    return inspection


def _container_succeeded(inspection: Any) -> bool:
    if not isinstance(inspection, dict) or set(inspection) != {
        "Status",
        "ExitCode",
        "OOMKilled",
        "RestartCount",
    }:
        return False
    return (
        isinstance(inspection.get("Status"), str)
        and inspection.get("Status") == "exited"
        and isinstance(inspection.get("ExitCode"), int)
        and not isinstance(inspection.get("ExitCode"), bool)
        and inspection.get("ExitCode") == 0
        and inspection.get("OOMKilled") is False
        and isinstance(inspection.get("RestartCount"), int)
        and not isinstance(inspection.get("RestartCount"), bool)
        and inspection["RestartCount"] == 0
    )


def _read_successful_syncer_exit(path: Path) -> None:
    try:
        if path.stat().st_size > 4096:
            raise HarnessError("eval syncer exit record is unexpectedly large")
        lines = path.read_text(encoding="utf-8").splitlines()
        pairs = [line.split("=", 1) for line in lines]
    except OSError as error:
        raise HarnessError("missing eval syncer exit record") from error
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(
        pairs
    ):
        raise HarnessError("eval syncer exit record is malformed")
    fields = dict(pairs)
    if (
        fields.get("result") != "success"
        or fields.get("exit_code") != "exited"
        or fields.get("exit_status") != "0"
    ):
        raise HarnessError("eval syncer did not exit successfully after FINAL_ACK")


def _eval_metric(event: dict[str, Any], name: str) -> float:
    value = event.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise HarnessError(f"eval event has an invalid {name}")
    return float(value)


def _verify_eval(
    plan: dict[str, Any], artifacts: Path
) -> tuple[str, int, int, float, float]:
    learner = plan["learner"]
    evaluation = learner["evaluation"]
    expected_version = learner["global_rounds"]
    expected_samples = (
        evaluation["prompt_count"] * evaluation["samples_per_prompt"]
    )
    if (artifacts / "syncer" / "state.ckpt").exists():
        raise HarnessError("eval artifact collection must not contain state.ckpt")
    _read_successful_syncer_exit(artifacts / "syncer" / "syncer.exit")
    try:
        lifecycle = _parse_syncer_lifecycle_evidence(
            (artifacts / "syncer" / "lifecycle-evidence.json").read_text(
                encoding="utf-8"
            )
        )
    except OSError as error:
        raise HarnessError("missing eval syncer lifecycle evidence") from error
    if lifecycle["resume_count"] != 1:
        raise HarnessError("eval syncer lacks one checkpoint resume event")
    if lifecycle["final_ack_count"] != _learner_count(plan):
        raise HarnessError("eval syncer lacks one FINAL_ACK per learner")

    hashes = set()
    results = []
    pass_at_1 = []
    for learner_id in range(_learner_count(plan)):
        events = _json_lines(_event_path(artifacts, learner_id))
        apply_indices = [
            index
            for index, event in enumerate(events)
            if event.get("event") == "rl_policy_apply"
            and event.get("policy_version") == expected_version
        ]
        eval_indices = [
            index
            for index, event in enumerate(events)
            if event.get("event") == "rl_eval_result"
            and event.get("dataset_name") == evaluation["dataset_name"]
        ]
        if len(apply_indices) != 1 or len(eval_indices) != 1:
            raise HarnessError(
                f"eval learner {learner_id} lacks one final apply/result event"
            )
        if apply_indices[0] >= eval_indices[0]:
            raise HarnessError(
                f"eval learner {learner_id} evaluated before final policy apply"
            )
        apply_event = events[apply_indices[0]]
        eval_event = events[eval_indices[0]]
        policy_hash = apply_event.get("sync/global_policy_hash")
        if not isinstance(policy_hash, str) or not _SHA256.fullmatch(policy_hash):
            raise HarnessError(f"eval learner {learner_id} has no final policy hash")
        if (
            apply_event.get("island_id") != learner_id
            or eval_event.get("island_id") != learner_id
            or eval_event.get("rollout_id") != 0
            or eval_event.get("policy_version") != expected_version
            or eval_event.get("sample_count") != expected_samples
        ):
            raise HarnessError(
                f"eval learner {learner_id} result has the wrong policy/sample identity"
            )
        hashes.add(policy_hash)
        results.append(_eval_metric(eval_event, "rl/eval/result"))
        pass_at_1.append(_eval_metric(eval_event, "rl/eval/pass_at_1"))
    if len(hashes) != 1:
        raise HarnessError("eval learners applied different terminal policies")
    return (
        evaluation["dataset_name"],
        evaluation["prompt_count"],
        evaluation["samples_per_prompt"],
        statistics.fmean(results),
        statistics.fmean(pass_at_1),
    )


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
    if plan["learner"].get("evaluation") is not None:
        if export_dir:
            raise HarnessError("eval-only verification cannot export a checkpoint")
        dataset, prompts, samples, result, pass_at_1 = _verify_eval(plan, artifacts)
        print(
            f"verified terminal eval {dataset}: {prompts} prompts x {samples} "
            f"sample(s), result={result:.6f}, pass@1={pass_at_1:.6f}"
        )
        return
    from ..export import parse_checkpoint

    checkpoint_path = artifacts / "syncer" / "state.ckpt"
    checkpoint = parse_checkpoint(checkpoint_path)
    learner = plan["learner"]
    sync_preset = learner.get("sync_preset", "strict-avg")
    if sync_preset == "decoupled":
        final_hash = _verify_decoupled(plan, checkpoint, artifacts)
        print(
            f"verified outer fragment step {checkpoint.global_step} fixed-roster "
            f"decoupled cut ({final_hash})"
        )
    else:
        final_hash = _verify_oracle(plan, checkpoint, artifacts)
        print(
            f"verified v{checkpoint.global_step} fixed-roster checkpoint and "
            f"ordered-f32 oracle ({final_hash})"
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
            sync_preset=sync_preset,
            fragments=learner.get("fragments", 1),
            pipeline=learner.get("pipeline", 1),
            local_horizon=learner.get("local_horizon", 1),
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
        help="comma-separated SSH nodes for one island; repeat for each island",
    )
    prepare_parser.add_argument("--gpus-per-node", type=int, default=1)
    prepare_parser.add_argument(
        "--syncer-host",
        default=None,
        help="SSH target that runs the syncer; defaults to the first learner host",
    )
    prepare_parser.add_argument("--syncer-address", default=None)
    prepare_parser.add_argument("--network-interface", default="eno3")
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--output-dir", default=None)
    prepare_parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    prepare_parser.add_argument("--remote-env-file", default=None)
    prepare_parser.add_argument("--final-ack-timeout-s", type=int, default=None)
    prepare_parser.add_argument("--eval-checkpoint-host", default=None)
    prepare_parser.add_argument("--eval-checkpoint-path", default=None)
    prepare_parser.add_argument("--eval-checkpoint-sha256", default=None)
    prepare_parser.add_argument("--eval-checkpoint-size-bytes", type=int, default=None)
    prepare_parser.add_argument("--eval-checkpoint-global-step", type=int, default=None)
    prepare_parser.add_argument("--eval-dataset-name", default=None)
    prepare_parser.add_argument("--eval-interval", type=int, default=None)
    prepare_parser.add_argument("--eval-samples-per-prompt", type=int, default=None)
    prepare_parser.add_argument("--eval-temperature", type=float, default=None)
    prepare_parser.add_argument("--eval-top-p", type=float, default=None)
    prepare_parser.add_argument("--eval-max-prompt-len", type=int, default=None)
    prepare_parser.add_argument("--eval-max-response-len", type=int, default=None)
    prepare_parser.add_argument("--eval-max-context-len", type=int, default=None)
    prepare_parser.add_argument("--model-manifest-sha256", default=None)
    prepare_parser.add_argument("--secrlenv-source-root", default=None)
    prepare_parser.add_argument("--secrlenv-source-sha256", default=None)
    prepare_parser.add_argument("--secrlenv-task-pack", default=None)
    prepare_parser.add_argument("--secrlenv-task-pack-sha256", default=None)
    prepare_parser.add_argument("--secrlenv-operator-image", default=None)
    prepare_parser.add_argument("--secrlenv-operator-image-id", default=None)
    prepare_parser.add_argument("--secrlenv-port", type=int, default=28765)
    prepare_parser.add_argument(
        "--secrlenv-max-active-episodes", type=int, default=16
    )
    prepare_parser.add_argument(
        "--codex-harness-binary",
        default=None,
        help="controller path to the pinned official Codex Linux binary",
    )
    prepare_parser.add_argument(
        "--codex-package-manifest",
        default=None,
        help="controller path to the pinned codex-package.json identity",
    )
    prepare_parser.add_argument(
        "--codex-app-server-schema",
        default=None,
        help="controller path to the pinned experimental v2 app-server schema",
    )
    prepare_parser.add_argument("--tms-preload-patch", default=None)
    prepare_parser.add_argument(
        "--tms-train-disk-backup-root",
        default=None,
    )
    prepare_parser.add_argument(
        "--tms-train-disk-backup-chunk-mb",
        type=int,
        default=TMS_TRAIN_DISK_BACKUP_DEFAULT_CHUNK_MB,
    )
    prepare_parser.add_argument("--jit-cache-root", default=None)
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
