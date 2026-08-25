"""Fail-closed direct launcher and reconciler for the Milestone-1 dense gate.

This deliberately does not use :mod:`yeto.rl.ssh_harness`.  A metadata-only
Miles probe first freezes the real full-parameter layout and fragment count.
This tool then binds that evidence, all executable inputs, and the exact
four-GPU-per-island topology into one canonical manifest shared by both
containers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "yeto-m1-dense-full-direct-launch-input-v1"
MANIFEST_SCHEMA = "yeto-m1-dense-full-direct-launch-manifest-v2"
PROBE_SCHEMA = "yeto-miles-full-parameter-manifest-probe-v1"
MODEL_REPO = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MILES_IMAGE_REPOSITORY = "radixark/miles"
MILES_IMAGE_DIGEST = (
    "sha256:4f6644fecb60dd784fc1804f2fd601bc7f4ead509d0daad311029f6f9f8478ea"
)
SYNCER_BINARY_SHA256 = (
    "f0869723f7454a446f2eb7553d957ed3b80483d92e81685cdbf6117b66f51b7d"
)
SYNCER_BUILD_MANIFEST_SHA256 = (
    "65902c13253202c457b127d30401a4fb490ccd3806360797ccc31c132a6526cd"
)
TRAIN_DATA_SHA256 = "de1c7b371ed5bc71fcfcf5563284ba7a88320b465130978ec051a0cb662f338f"
HELDOUT_DATA_SHA256 = "b433f81220bcad6d96eaecb2c16dc79a596026d7d7ec653584f953a60507f3e9"
DATA_MANIFEST_SHA256 = (
    "72e6dddc20aa834aa2ebff7f6eb154f561e7f340f44b6e30803747196e3e735c"
)
DOCKER_IMAGE_INVENTORY_SHA256 = (
    "0dcf687ecee23cba1f3cc9e2cbaf154bff1b636dd29f382783a392377f5a42d3"
)
TASK_PACK_SHA256 = "cf1d277e1e5a91d42445c08df67bc1f164e1602211b016ad2ca2f565b0dfb759"
MODEL_CACHE_SNAPSHOT = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/" + MODEL_REVISION
)
REWARD_FUNCTION = "yeto_miles_secrlenv.reward:reward_func"
CUSTOM_GENERATE_FUNCTION = "yeto_miles_secrlenv.generate.generate"
CUSTOM_AGENT_FUNCTION = "yeto_miles_secrlenv.codex_harness_agent.run"
DYNAMIC_FILTER_FUNCTION = "yeto_miles_secrlenv.reward.check_group"
MAX_FRAGMENT_BYTES = 2 * 1024**3
PRODUCTION_SEQUENCE_LENGTH = 4096
ISLAND_COUNT = 2
GPUS_PER_ISLAND = 4
TRAINER_GPUS = 2
INFERENCE_GPUS = 2
INFERENCE_ENGINES = 2
INFERENCE_TP = 1
LAUNCH_MODES = {"single-island-gate": 1, "two-island-final": ISLAND_COUNT}
FINAL_MIN_ROUNDS = 3
FINAL_MAX_ROUNDS = 8
EVAL_DATASET_NAME = "heldout"
EVAL_PROMPT_COUNT = 2
EVAL_SAMPLES_PER_PROMPT = 1
EVAL_SUMMARY_SCHEMA = "yeto-m1-dense-heldout-summary-v1"
FINAL_REPORT_SCHEMA = "yeto-m1-dense-final-report-v1"
LAUNCH_BUNDLE_SCHEMA = "yeto-m1-dense-launch-bundle-v1"
LAUNCH_BUNDLE_FILES = (
    "tools/probes/m1_dense_full_direct_launch.py",
    "tools/probes/m1_dense_full_final_report.py",
    "tools/probes/run_m1_dense_full_island.sh",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}\Z")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{16,}\Z")
_PORT_KEYS = frozenset(
    {
        "ray_gcs",
        "ray_dashboard",
        "ray_client",
        "rollout_engine_base",
        "sglang_router",
        "sglang_router_prometheus",
        "train_master_base",
        "session_server",
    }
)
_PROFILE_KEYS = frozenset(
    {
        "seq_len",
        "rollout_max_response_len",
        "groups_per_round",
        "samples_per_group",
        "over_sampling_batch_size",
        "inner_lr",
        "seed",
        "sglang_mem_fraction_static",
        "reward_function",
        "reward_source_path",
        "reward_sha256",
        "custom_generate_function_path",
        "custom_agent_function_path",
        "dynamic_sampling_filter_path",
        "dynamic_sampling_max_replacements",
        "secrlenv_max_infrastructure_replacements",
        "tito_model",
    }
)


class LaunchContractError(RuntimeError):
    """The proposed launch is not exactly the attested Milestone-1 run."""


def _validate_rounds(launch_mode: str, rounds: Any) -> int:
    if type(rounds) is not int:
        raise LaunchContractError("round count must be an integer")
    if launch_mode == "single-island-gate":
        if rounds != 1:
            raise LaunchContractError("single-island gate must be exactly one round")
    elif launch_mode == "two-island-final":
        if not FINAL_MIN_ROUNDS <= rounds <= FINAL_MAX_ROUNDS:
            raise LaunchContractError(
                f"two-island final requires {FINAL_MIN_ROUNDS}..{FINAL_MAX_ROUNDS} rounds"
            )
    else:
        raise LaunchContractError("launch_mode is not an allowlisted M1 mode")
    return rounds


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise LaunchContractError(f"duplicate JSON key: {name!r}")
        result[name] = value
    return result


def _load_json(path: Path, *, canonical: bool = False, private: bool = False) -> Any:
    if path.is_symlink() or not path.is_file():
        raise LaunchContractError(f"not a regular file: {path}")
    if private and path.stat().st_mode & 0o077:
        raise LaunchContractError(f"private JSON has permissive mode: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, ValueError) as exc:
        raise LaunchContractError(f"invalid JSON: {path}") from exc
    if canonical and raw != _canonical(payload):
        raise LaunchContractError(f"JSON is not canonical: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LaunchContractError(f"{name} must be a lowercase SHA256")
    return value


def _expect_keys(value: Any, expected: set[str] | frozenset[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise LaunchContractError(
            f"{name} keys differ: expected={sorted(expected)!r} actual={actual!r}"
        )
    return value


def _absolute(value: Any, name: str) -> Path:
    if not isinstance(value, str):
        raise LaunchContractError(f"{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise LaunchContractError(f"{name} must be an absolute normalized path")
    return path


def _regular(path: Path, name: str, *, executable: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise LaunchContractError(f"{name} is not a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise LaunchContractError(f"{name} is not executable: {path}")


def _directory(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise LaunchContractError(f"{name} is not a real directory: {path}")


def _attest_file(path: Path, expected: str, name: str) -> None:
    _regular(path, name)
    actual = _sha256(path)
    if actual != expected:
        raise LaunchContractError(
            f"{name} SHA256 mismatch: expected={expected} actual={actual}"
        )


def _is_linux_x86_64_elf(path: Path) -> bool:
    with path.open("rb") as stream:
        header = stream.read(20)
    return (
        len(header) == 20
        and header[:4] == b"\x7fELF"
        and header[4] == 2  # ELFCLASS64
        and header[5] == 1  # little endian
        and int.from_bytes(header[18:20], "little") == 62  # EM_X86_64
    )


def _validate_syncer_build_manifest(
    path: Path,
    *,
    expected_sha256: str,
    binary_path: Path,
    binary_sha256: str,
) -> dict[str, Any]:
    _attest_file(path, expected_sha256, "syncer build manifest")
    payload = _load_json(path, canonical=True, private=True)
    payload = _expect_keys(
        payload, {"schema", "binary", "build", "source"}, "syncer build manifest"
    )
    binary = _expect_keys(
        payload["binary"],
        {"linkage", "mode", "sha256", "size_bytes"},
        "syncer build binary",
    )
    build = _expect_keys(
        payload["build"],
        {
            "alpine_version",
            "cargo_locked",
            "cargo_version",
            "image_index_digest",
            "image_linux_amd64_digest",
            "musl_dev_version",
            "platform",
            "rust_commit",
            "rust_version",
        },
        "syncer build",
    )
    source = _expect_keys(
        payload["source"],
        {"cargo_lock_sha256", "cargo_toml_sha256", "files"},
        "syncer source",
    )
    if (
        payload["schema"] != "yeto-m1-syncer-linux-musl-build/v1"
        or binary["linkage"] != "static-pie"
        or binary["mode"] != "0700"
        or binary["sha256"] != binary_sha256
        or binary["size_bytes"] != binary_path.stat().st_size
        or build["cargo_locked"] is not True
        or build["platform"] != "linux/amd64"
        or build["alpine_version"] != "3.22.1"
        or build["musl_dev_version"] != "1.2.5-r12"
        or build["rust_version"] != "1.88.0"
        or build["rust_commit"] != "6b00bc3880198600130e1cf62b8f8a93494488cc"
        or not isinstance(build["cargo_version"], str)
        or not build["cargo_version"].startswith("1.88.0 ")
        or not _DIGEST.fullmatch(str(build["image_index_digest"]))
        or not _DIGEST.fullmatch(str(build["image_linux_amd64_digest"]))
        or not isinstance(source["files"], list)
        or not source["files"]
    ):
        raise LaunchContractError("syncer build manifest does not bind the binary")
    _expect_sha(source["cargo_lock_sha256"], "syncer Cargo.lock")
    _expect_sha(source["cargo_toml_sha256"], "syncer Cargo.toml")
    for index, row in enumerate(source["files"]):
        row = _expect_keys(row, {"path", "sha256"}, f"syncer source file {index}")
        if not isinstance(row["path"], str) or not row["path"].startswith("src/"):
            raise LaunchContractError("syncer source manifest path is invalid")
        _expect_sha(row["sha256"], f"syncer source file {index}")
    return payload


def _validate_daemon_contract(
    path: Path, *, expected_sha256: str, task_pack_sha256: str
) -> dict[str, Any]:
    _attest_file(path, expected_sha256, "SecRLEnv daemon contract")
    payload = _load_json(path, canonical=True, private=True)
    payload = _expect_keys(
        payload,
        {
            "source_root",
            "source_sha256",
            "task_pack",
            "task_pack_sha256",
            "state_root",
            "bind",
            "port",
            "operator_image",
            "operator_image_id",
            "max_active_episodes",
            "enable_dind_debug",
            "dind_image",
            "dind_image_id",
            "dind_debug_script_sha256",
        },
        "SecRLEnv daemon contract",
    )
    if (
        payload["task_pack_sha256"] != task_pack_sha256
        or payload["enable_dind_debug"] is not True
        or not isinstance(payload["dind_image"], str)
        or not payload["dind_image"]
        or not _DIGEST.fullmatch(str(payload["dind_image_id"]))
        or not _SHA256.fullmatch(str(payload["dind_debug_script_sha256"]))
        or not _DIGEST.fullmatch(str(payload["operator_image_id"]))
        or payload["bind"] != "127.0.0.1"
        or type(payload["port"]) is not int
        or not 1024 <= payload["port"] <= 65535
        or type(payload["max_active_episodes"]) is not int
        or payload["max_active_episodes"] < 2
    ):
        raise LaunchContractError(
            "SecRLEnv daemon must bind the task pack and flagless DinD debug"
        )
    _expect_sha(payload["source_sha256"], "SecRLEnv source")
    return payload


def _validate_data_bundle(
    manifest_path: Path,
    inventory_path: Path,
    *,
    task_pack_sha256: str,
    train_sha256: str,
    heldout_sha256: str,
    inventory_sha256: str,
) -> dict[str, Any]:
    bundle = _load_json(manifest_path, canonical=True, private=True)
    inventory = _load_json(inventory_path, canonical=True, private=True)
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != "qwen35-m1-dense-secrlenv-gate-input/v1"
        or bundle.get("outputs", {}).get("train", {}).get("sha256") != train_sha256
        or bundle.get("outputs", {}).get("heldout", {}).get("sha256") != heldout_sha256
        or bundle.get("outputs", {}).get("docker_images", {}).get("sha256")
        != inventory_sha256
        or bundle.get("source", {}).get("reduced_task_pack", {}).get("manifest_sha256")
        != task_pack_sha256
    ):
        raise LaunchContractError("SecRLEnv train/heldout bundle identity differs")
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema") != "secrlenv-minimal-immutable-image-inventory/v1"
        or inventory.get("source_task_pack_sha256") != task_pack_sha256
        or inventory.get("task_count") != 4
        or inventory.get("image_count") != len(inventory.get("images", []))
        or inventory.get("service_binding_count") != len(inventory.get("bindings", []))
    ):
        raise LaunchContractError("SecRLEnv immutable image inventory differs")
    for index, row in enumerate(inventory["images"]):
        row = _expect_keys(row, {"image_id", "immutable"}, f"task image {index}")
        if (
            not _DIGEST.fullmatch(str(row["image_id"]))
            or not isinstance(row["immutable"], str)
            or not row["immutable"].endswith("@" + row["image_id"])
        ):
            raise LaunchContractError("SecRLEnv task image is not immutable")
    return inventory


def _closed_conversion_files(
    root: Path, *, omit_manifest: bool = False
) -> list[dict[str, object]]:
    _directory(root, "conversion artifact root")
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            raise LaunchContractError(f"conversion artifact contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise LaunchContractError(f"conversion artifact is not regular: {path}")
        if omit_manifest and relative.as_posix() == "conversion-manifest.json":
            continue
        size = path.stat().st_size
        if size < 1:
            raise LaunchContractError(f"conversion artifact is empty: {path}")
        records.append(
            {"path": relative.as_posix(), "size": size, "sha256": _sha256(path)}
        )
    if not records:
        raise LaunchContractError(f"conversion artifact root is empty: {root}")
    return records


def _validate_conversion_artifacts(
    manifest_path: Path,
    *,
    expected_sha256: str,
    image_digest: str,
    model_root: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    release_marker = checkpoint_root / "latest_checkpointed_iteration.txt"
    _regular(release_marker, "Megatron release marker")
    try:
        marker_value = release_marker.read_text().strip()
    except (OSError, UnicodeError) as exc:
        raise LaunchContractError("Megatron release marker is unreadable") from exc
    if marker_value != "release":
        raise LaunchContractError("Megatron checkpoint is not a release checkpoint")
    _attest_file(manifest_path, expected_sha256, "conversion manifest")
    payload = _load_json(manifest_path, canonical=True, private=True)
    payload = _expect_keys(
        payload,
        {
            "schema",
            "image_digest",
            "model_repo",
            "model_revision",
            "model_config_sha256",
            "model_files",
            "checkpoint_files",
            "conversion_source_file_count",
            "conversion_source_bytes",
            "conversion_source_aggregate_sha256",
        },
        "conversion manifest",
    )
    if (
        payload["schema"] != "yeto-qwen35-megatron-conversion-v1"
        or payload["image_digest"] != image_digest
        or payload["model_repo"] != MODEL_REPO
        or payload["model_revision"] != MODEL_REVISION
        or payload["model_files"] != _closed_conversion_files(model_root)
        or payload["checkpoint_files"]
        != _closed_conversion_files(checkpoint_root, omit_manifest=True)
        or type(payload["conversion_source_file_count"]) is not int
        or payload["conversion_source_file_count"] < 1
        or type(payload["conversion_source_bytes"]) is not int
        or payload["conversion_source_bytes"] < 1
    ):
        raise LaunchContractError("conversion manifest or artifacts differ")
    _expect_sha(payload["model_config_sha256"], "conversion model config")
    _expect_sha(payload["conversion_source_aggregate_sha256"], "conversion source")
    config = next(
        (row for row in payload["model_files"] if row.get("path") == "config.json"),
        None,
    )
    if config is None or config.get("sha256") != payload["model_config_sha256"]:
        raise LaunchContractError("conversion manifest does not bind model config")
    metadata_root = model_root / ".cache" / "huggingface" / "download"
    if metadata_root.is_symlink() or not metadata_root.is_dir():
        raise LaunchContractError("Hugging Face revision metadata is absent")
    metadata = sorted(metadata_root.rglob("*.metadata"))
    if len(metadata) != len(payload["model_files"]):
        raise LaunchContractError("Hugging Face revision metadata count differs")
    for path in metadata:
        if path.is_symlink() or not path.is_file():
            raise LaunchContractError("Hugging Face revision metadata is not regular")
        try:
            revision = path.read_text().splitlines()[0]
        except (OSError, UnicodeError, IndexError) as exc:
            raise LaunchContractError(
                "Hugging Face revision metadata is invalid"
            ) from exc
        if revision != MODEL_REVISION:
            raise LaunchContractError("Hugging Face revision metadata differs")
    return payload


def _source_hashes(yeto_root: Path, miles_root: Path) -> tuple[str, str]:
    from yeto.provenance import source_tree_sha256
    from yeto.rl.miles import miles_execution_source_sha256

    return (
        source_tree_sha256(yeto_root / "yeto"),
        miles_execution_source_sha256(miles_root),
    )


def _validate_launch_bundle(value: Any) -> dict[str, Any]:
    bundle = _expect_keys(
        value, {"schema", "files", "aggregate_sha256"}, "launch bundle"
    )
    files = bundle["files"]
    if bundle["schema"] != LAUNCH_BUNDLE_SCHEMA or not isinstance(files, list):
        raise LaunchContractError("launch bundle schema or files differ")
    if len(files) != len(LAUNCH_BUNDLE_FILES):
        raise LaunchContractError("launch bundle file count differs")
    for expected_path, row in zip(LAUNCH_BUNDLE_FILES, files, strict=True):
        row = _expect_keys(row, {"path", "sha256"}, "launch bundle file")
        if row["path"] != expected_path:
            raise LaunchContractError("launch bundle path ordering differs")
        _expect_sha(row["sha256"], f"launch bundle {expected_path}")
    aggregate = hashlib.sha256(
        _canonical({"schema": LAUNCH_BUNDLE_SCHEMA, "files": files})
    ).hexdigest()
    if _expect_sha(bundle["aggregate_sha256"], "launch bundle aggregate") != aggregate:
        raise LaunchContractError("launch bundle aggregate differs")
    return bundle


def _launch_bundle(yeto_root: Path) -> dict[str, Any]:
    _directory(yeto_root, "launch bundle Yeto root")
    files = []
    for relative in LAUNCH_BUNDLE_FILES:
        path = yeto_root / relative
        _regular(path, f"launch bundle {relative}")
        files.append({"path": relative, "sha256": _sha256(path)})
    bundle = {
        "schema": LAUNCH_BUNDLE_SCHEMA,
        "files": files,
        "aggregate_sha256": hashlib.sha256(
            _canonical({"schema": LAUNCH_BUNDLE_SCHEMA, "files": files})
        ).hexdigest(),
    }
    return _validate_launch_bundle(bundle)


def _attest_launch_bundle(value: Any, yeto_root: Path) -> dict[str, Any]:
    expected = _validate_launch_bundle(value)
    observed = _launch_bundle(yeto_root)
    if observed != expected:
        raise LaunchContractError("launch bundle changed after manifest creation")
    return expected


def _probe_source_sha(source: Any, name: str, field: str) -> str:
    source = _expect_keys(source, {"path", field}, f"probe {name} source")
    if not isinstance(source["path"], str) or not source["path"].startswith("/"):
        raise LaunchContractError(f"probe {name} source path is malformed")
    return _expect_sha(source[field], f"probe {name} execution source")


def _validate_probe(
    probe: Any,
    *,
    image_digest: str,
    yeto_source_sha256: str,
    miles_source_sha256: str,
) -> dict[str, Any]:
    if not isinstance(probe, dict) or probe.get("schema") != PROBE_SCHEMA:
        raise LaunchContractError("metadata probe has the wrong schema")
    required = {
        "schema",
        "observed_utc",
        "algorithm",
        "probe_mode",
        "role",
        "fragment_strategy",
        "model_repo",
        "model_revision",
        "model_config_sha256",
        "model_path",
        "checkpoint_path",
        "conversion_manifest",
        "miles_image_digest",
        "yeto_source",
        "miles_source",
        "hardware",
        "megatron_bridge",
        "actor_topology",
        "sequence_length",
        "parameter_layout_hash",
        "owner_count",
        "minimum_fragment_count",
        "derived_fragment_count",
        "parameter_tensor_count",
        "parameter_scalar_count",
        "max_fragment_bytes_limit",
        "max_chunk_bytes",
        "observed_max_fragment_bytes",
        "owner_plans",
        "fragments",
    }
    _expect_keys(probe, required, "metadata probe")
    if (
        probe["algorithm"] != "grpo"
        or probe["probe_mode"] != "full_parameter_manifest_only"
        or probe["role"] != "actor"
        or probe["fragment_strategy"] != "owner_affine"
        or probe["model_repo"] != MODEL_REPO
        or probe["model_revision"] != MODEL_REVISION
        or probe["sequence_length"] != PRODUCTION_SEQUENCE_LENGTH
    ):
        raise LaunchContractError(
            "metadata probe describes a different algorithm/model"
        )
    if probe["miles_image_digest"] != image_digest:
        raise LaunchContractError("metadata probe image digest differs")
    if (
        _probe_source_sha(probe["yeto_source"], "Yeto", "source_tree_sha256")
        != yeto_source_sha256
    ):
        raise LaunchContractError("metadata probe Yeto source differs")
    if (
        _probe_source_sha(probe["miles_source"], "Miles", "execution_source_sha256")
        != miles_source_sha256
    ):
        raise LaunchContractError("metadata probe Miles source differs")
    if not isinstance(probe["model_revision"], str) or not _COMMIT.fullmatch(
        probe["model_revision"]
    ):
        raise LaunchContractError("metadata probe model revision is not immutable")
    _expect_sha(probe["model_config_sha256"], "probe model config")
    _expect_sha(probe["parameter_layout_hash"], "probe parameter layout")
    topology = _expect_keys(
        probe["actor_topology"],
        {
            "actor_num_nodes",
            "actor_num_gpus_per_node",
            "tp_size",
            "pp_size",
            "ep_size",
            "cp_size",
            "dp_size",
        },
        "probe actor topology",
    )
    if topology != {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": 1,
        "cp_size": 1,
        "dp_size": 1,
    }:
        raise LaunchContractError("metadata probe did not use the TP2/DP1 trainer")
    if probe["owner_count"] != 2 or probe["minimum_fragment_count"] != 2:
        raise LaunchContractError("metadata probe owner/minimum fragment count differs")
    fragments = probe["fragments"]
    derived = probe["derived_fragment_count"]
    if type(derived) is not int or derived < 2 or not isinstance(fragments, list):
        raise LaunchContractError("metadata probe derived fragment count is invalid")
    ids = [row.get("fragment_id") for row in fragments if isinstance(row, dict)]
    if len(fragments) != derived or ids != list(range(derived)):
        raise LaunchContractError(
            "metadata probe fragments are not exact and contiguous"
        )
    total_numel = 0
    for index, row in enumerate(fragments):
        required_fragment = {
            "fragment_id",
            "role",
            "shard_id",
            "plan_hash",
            "tensor_count",
            "numel",
            "fp32_bytes",
        }
        _expect_keys(row, required_fragment, f"probe fragment {index}")
        if (
            row["role"] != "actor"
            or type(row["numel"]) is not int
            or row["numel"] <= 0
            or row["fp32_bytes"] != row["numel"] * 4
            or row["fp32_bytes"] > MAX_FRAGMENT_BYTES
        ):
            raise LaunchContractError(f"probe fragment {index} violates the FP32 bound")
        _expect_sha(row["plan_hash"], f"probe fragment {index} plan")
        total_numel += row["numel"]
    if (
        probe["max_fragment_bytes_limit"] != MAX_FRAGMENT_BYTES
        or probe["observed_max_fragment_bytes"]
        != max(row["fp32_bytes"] for row in fragments)
        or probe["observed_max_fragment_bytes"] > MAX_FRAGMENT_BYTES
        or probe["parameter_scalar_count"] != total_numel
    ):
        raise LaunchContractError("metadata probe fragment totals or bounds differ")
    owners = probe["owner_plans"]
    if not isinstance(owners, list) or len(owners) != 2:
        raise LaunchContractError("metadata probe must contain both TP2 owners")
    topology_keys = {
        "tp_rank",
        "tp_size",
        "pp_rank",
        "pp_size",
        "ep_rank",
        "ep_size",
        "cp_rank",
        "cp_size",
        "dp_rank",
        "dp_size",
    }
    covered: list[int] = []
    owner_scalars = 0
    owner_tensors = 0
    for owner_index, owner in enumerate(owners):
        owner = _expect_keys(
            owner,
            {
                "role",
                "shard_id",
                "topology",
                "manifest_layout_hash",
                "plan_hash",
                "parameter_tensor_count",
                "parameter_scalar_count",
                "fragment_ids",
                "fragment_count",
                "max_fragment_bytes",
            },
            f"probe owner {owner_index}",
        )
        topology_row = _expect_keys(
            owner["topology"], topology_keys, f"probe owner {owner_index} topology"
        )
        expected_topology = {
            "tp_rank": owner_index,
            "tp_size": 2,
            "pp_rank": 0,
            "pp_size": 1,
            "ep_rank": 0,
            "ep_size": 1,
            "cp_rank": 0,
            "cp_size": 1,
            "dp_rank": 0,
            "dp_size": 1,
        }
        ids = owner["fragment_ids"]
        owned_rows = [
            row
            for row in fragments
            if row["shard_id"] == owner["shard_id"]
            and row["plan_hash"] == owner["plan_hash"]
        ]
        if (
            owner["role"] != "actor"
            or topology_row != expected_topology
            or not isinstance(ids, list)
            or not ids
            or ids != sorted(ids)
            or owner["fragment_count"] != len(ids)
            or [row["fragment_id"] for row in owned_rows] != ids
            or owner["max_fragment_bytes"]
            != max(row["fp32_bytes"] for row in owned_rows)
            or type(owner["parameter_scalar_count"]) is not int
            or owner["parameter_scalar_count"] <= 0
            or type(owner["parameter_tensor_count"]) is not int
            or owner["parameter_tensor_count"] <= 0
        ):
            raise LaunchContractError("metadata probe owner plan is inconsistent")
        _expect_sha(owner["manifest_layout_hash"], "owner manifest layout")
        _expect_sha(owner["plan_hash"], "owner plan")
        covered.extend(ids)
        owner_scalars += owner["parameter_scalar_count"]
        owner_tensors += owner["parameter_tensor_count"]
    if (
        sorted(covered) != list(range(derived))
        or len(set(covered)) != derived
        or owner_scalars != probe["parameter_scalar_count"]
        or owner_tensors != probe["parameter_tensor_count"]
    ):
        raise LaunchContractError("metadata probe owner plans do not close the layout")
    conversion = _expect_keys(
        probe["conversion_manifest"],
        {
            "path",
            "sha256",
            "schema",
            "model_file_count",
            "checkpoint_file_count",
            "conversion_source_aggregate_sha256",
        },
        "probe conversion manifest",
    )
    _expect_sha(conversion["sha256"], "probe conversion manifest")
    _expect_sha(
        conversion["conversion_source_aggregate_sha256"],
        "probe conversion source",
    )
    if (
        conversion["schema"] != "yeto-qwen35-megatron-conversion-v1"
        or type(conversion["model_file_count"]) is not int
        or conversion["model_file_count"] <= 0
        or type(conversion["checkpoint_file_count"]) is not int
        or conversion["checkpoint_file_count"] <= 0
    ):
        raise LaunchContractError("probe conversion inventory is invalid")
    bridge = _expect_keys(
        probe["megatron_bridge"],
        {
            "distribution_name",
            "distribution_version",
            "direct_url",
            "direct_url_commit",
        },
        "probe Megatron Bridge",
    )
    direct_url = bridge["direct_url"]
    vcs = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    if (
        not isinstance(bridge["distribution_name"], str)
        or not bridge["distribution_name"]
        or not isinstance(bridge["distribution_version"], str)
        or not bridge["distribution_version"]
        or not isinstance(direct_url, dict)
        or not isinstance(direct_url.get("url"), str)
        or not direct_url["url"]
        or not isinstance(vcs, dict)
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != bridge["direct_url_commit"]
        or not isinstance(bridge["direct_url_commit"], str)
        or _COMMIT.fullmatch(bridge["direct_url_commit"]) is None
    ):
        raise LaunchContractError("probe Megatron Bridge commit is not immutable")
    return probe


def _validated_input(path: Path) -> dict[str, Any]:
    payload = _load_json(path, canonical=True, private=True)
    return _expect_keys(
        payload,
        {
            "schema",
            "launch_mode",
            "run_id",
            "rounds",
            "image_repository",
            "paths",
            "expected",
            "hardware",
            "ports",
            "profile",
            "harness",
            "secrlenv",
        },
        "launch input",
    )


def _port_contract(
    value: Any, *, island_count: int
) -> tuple[int, list[dict[str, int]]]:
    value = _expect_keys(value, {"syncer", "islands"}, "ports")
    syncer_port = value["syncer"]
    islands = value["islands"]
    if (
        type(syncer_port) is not int
        or not isinstance(islands, list)
        or len(islands) != island_count
    ):
        raise LaunchContractError("port contract has the wrong shape")
    all_ports = [syncer_port]
    checked = []
    for index, item in enumerate(islands):
        row = _expect_keys(item, _PORT_KEYS, f"island {index} ports")
        if any(
            type(port) is not int or not 1024 <= port <= 65535 for port in row.values()
        ):
            raise LaunchContractError(f"island {index} has an invalid port")
        all_ports.extend(row.values())
        checked.append(dict(row))
    if not 1024 <= syncer_port <= 65535 or len(set(all_ports)) != len(all_ports):
        raise LaunchContractError(
            "every syncer/Ray/Miles/session port must be disjoint"
        )
    return syncer_port, checked


def _hardware_contract(value: Any, *, island_count: int) -> tuple[list[str], list[str]]:
    value = _expect_keys(value, {"gpu_uuids", "container_names"}, "hardware")
    uuids = value["gpu_uuids"]
    names = value["container_names"]
    if (
        not isinstance(uuids, list)
        or len(uuids) != island_count * GPUS_PER_ISLAND
        or len(set(uuids)) != island_count * GPUS_PER_ISLAND
        or any(
            not isinstance(item, str) or not _GPU_UUID.fullmatch(item) for item in uuids
        )
        or not isinstance(names, list)
        or len(names) != island_count
        or len(set(names)) != island_count
        or any(
            not isinstance(item, str) or not _RUN_ID.fullmatch(item) for item in names
        )
    ):
        raise LaunchContractError("hardware does not match the explicit launch mode")
    return uuids, names


def _artifact_contract(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise LaunchContractError("Codex harness artifacts are absent")
    checked = []
    destinations = set()
    for index, row in enumerate(rows):
        row = _expect_keys(
            row,
            {"host_path", "container_path", "sha256", "size_bytes", "executable"},
            f"harness artifact {index}",
        )
        host = _absolute(row["host_path"], f"harness artifact {index} host")
        container = _absolute(
            row["container_path"], f"harness artifact {index} container"
        )
        digest = _expect_sha(row["sha256"], f"harness artifact {index}")
        _regular(
            host, f"harness artifact {index}", executable=row["executable"] is True
        )
        if (
            type(row["size_bytes"]) is not int
            or row["size_bytes"] <= 0
            or type(row["executable"]) is not bool
            or host.stat().st_size != row["size_bytes"]
            or _sha256(host) != digest
            or str(container) in destinations
        ):
            raise LaunchContractError(f"harness artifact {index} identity differs")
        destinations.add(str(container))
        checked.append(dict(row))
    return checked


def _validate_harness_artifact_binding(
    contract: dict[str, Any], artifacts: list[dict[str, Any]]
) -> None:
    by_destination = {row["container_path"]: row for row in artifacts}
    expected = {
        contract["container_binary_path"]: (
            contract["controller_binary_path"],
            contract["binary_sha256"],
            contract["binary_size_bytes"],
            True,
        ),
        "/opt/yeto/codex/codex-package.json": (
            contract["controller_package_manifest_path"],
            contract["package_manifest_sha256"],
            None,
            False,
        ),
        contract["container_app_server_schema_path"]: (
            contract["controller_app_server_schema_path"],
            contract["app_server_schema_sha256"],
            None,
            False,
        ),
    }
    if set(by_destination) != set(expected):
        raise LaunchContractError("Codex harness artifact destinations differ")
    for destination, (host_path, digest, size, executable) in expected.items():
        row = by_destination[destination]
        if (
            row["host_path"] != host_path
            or row["sha256"] != digest
            or row["executable"] is not executable
            or (size is not None and row["size_bytes"] != size)
        ):
            raise LaunchContractError(
                f"Codex harness artifact identity differs: {destination}"
            )


def _validate_profile_semantics(profile: Any) -> dict[str, Any]:
    profile = _expect_keys(profile, _PROFILE_KEYS, "profile")
    if (
        type(profile["seq_len"]) is not int
        or not 512 <= profile["seq_len"] <= 8192
        or type(profile["rollout_max_response_len"]) is not int
        or not 1 <= profile["rollout_max_response_len"] <= profile["seq_len"]
        or type(profile["groups_per_round"]) is not int
        or profile["groups_per_round"] < 1
        or type(profile["samples_per_group"]) is not int
        or profile["samples_per_group"] != 3
        or type(profile["over_sampling_batch_size"]) is not int
        or profile["over_sampling_batch_size"] <= profile["groups_per_round"]
        or type(profile["seed"]) is not int
        or type(profile["sglang_mem_fraction_static"]) not in (int, float)
        or not 0 < profile["sglang_mem_fraction_static"] < 0.8
        or type(profile["inner_lr"]) not in (int, float)
        or not 0 < profile["inner_lr"] <= 1e-3
        or profile["secrlenv_max_infrastructure_replacements"] != 1
        or profile["dynamic_sampling_max_replacements"] != 0
        or profile["tito_model"] != "qwen35"
        or profile["reward_function"] != REWARD_FUNCTION
        or profile["custom_generate_function_path"] != CUSTOM_GENERATE_FUNCTION
        or profile["custom_agent_function_path"] != CUSTOM_AGENT_FUNCTION
        or profile["dynamic_sampling_filter_path"] != DYNAMIC_FILTER_FUNCTION
    ):
        raise LaunchContractError("learner smoke profile is unsafe or malformed")
    _absolute(profile["reward_source_path"], "reward source")
    _expect_sha(profile["reward_sha256"], "reward source")
    return profile


def _evaluation_contract(
    *,
    launch_mode: str,
    rounds: int,
    profile: dict[str, Any],
    data: dict[str, Any],
    run_root: Path,
    yeto_root: Path,
    island_count: int,
) -> dict[str, Any]:
    enabled = launch_mode == "two-island-final"
    return {
        "enabled": enabled,
        "cadence": "initial-and-terminal" if enabled else "none",
        "dataset_name": EVAL_DATASET_NAME if enabled else None,
        "container_data_path": data["container_heldout_path"] if enabled else None,
        "data_sha256": data["heldout_sha256"] if enabled else None,
        "prompt_count": EVAL_PROMPT_COUNT if enabled else 0,
        "samples_per_prompt": EVAL_SAMPLES_PER_PROMPT if enabled else 0,
        "temperature": 0.0 if enabled else None,
        "top_p": 1.0 if enabled else None,
        "max_prompt_len": (
            profile["seq_len"] - profile["rollout_max_response_len"]
            if enabled
            else None
        ),
        "max_response_len": (profile["rollout_max_response_len"] if enabled else None),
        "max_context_len": profile["seq_len"] if enabled else None,
        "policy_versions": [0, rounds] if enabled else [],
        # Miles receives a positive interval so it constructs the real eval
        # datasets.  Keeping it just outside the rollout budget prevents the
        # stock loop from adding unbound periodic evals; the dense hook invokes
        # the exact initial and terminal publications itself.
        "miles_eval_interval": rounds + 1 if enabled else None,
        "summary_schema": EVAL_SUMMARY_SCHEMA if enabled else None,
        "island_summary_paths": (
            [
                str(run_root / f"island-{index}" / "eval-summary.json")
                for index in range(island_count)
            ]
            if enabled
            else []
        ),
        "metric_history_dirs": (
            [
                str(run_root / f"island-{index}" / "miles-metrics")
                for index in range(island_count)
            ]
            if enabled
            else []
        ),
        "trajectory_evidence_globs": (
            [
                str(
                    run_root
                    / f"island-{index}"
                    / "audit"
                    / f"trajectory-evidence-{index}-*"
                )
                for index in range(island_count)
            ]
            if enabled
            else []
        ),
        "aggregate_report_path": (
            str(run_root / "final-report.json") if enabled else None
        ),
        "post_run_argv_template": (
            [
                "python3",
                str(yeto_root / "tools/probes/m1_dense_full_direct_launch.py"),
                "report",
                "--manifest",
                str(run_root / "manifest.json"),
                "--expected-sha256",
                "<MANIFEST_SHA256>",
            ]
            if enabled
            else []
        ),
    }


def _harness_environment(contract: dict[str, Any]) -> dict[str, str]:
    backend = contract.get("backend")
    thinking = backend.get("thinking") if isinstance(backend, dict) else None
    kwargs = backend.get("chat_template_kwargs") if isinstance(backend, dict) else None
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
    _expect_keys(contract, required, "Codex harness contract")
    backend = _expect_keys(
        backend,
        {
            "model",
            "max_tokens",
            "reasoning_effort",
            "thinking",
            "chat_template",
            "chat_template_kwargs",
            "tito_allowed_append_roles",
        },
        "Codex backend contract",
    )
    if (
        contract["reasoning_effort"] != "xhigh"
        or type(contract["binary_size_bytes"]) is not int
        or contract["binary_size_bytes"] < 1
        or type(backend["max_tokens"]) is not int
        or backend["max_tokens"] < 1
    ):
        raise LaunchContractError("Codex harness is not the pinned xhigh contract")
    for name in (
        "agent_source_sha256",
        "binary_sha256",
        "npm_tarball_sha256",
        "package_manifest_sha256",
        "app_server_schema_sha256",
        "base_instructions_sha256",
        "terminal_exec_tool_schema_sha256",
        "submit_tool_schema_sha256",
        "dynamic_tools_schema_sha256",
    ):
        _expect_sha(contract[name], f"Codex harness {name}")
    if (
        not isinstance(backend, dict)
        or backend.get("model") != "qwen35"
        or backend.get("reasoning_effort") != "xhigh"
        or thinking != {"type": "enabled"}
        or backend.get("chat_template") != "qwen35"
        or kwargs != {"clear_thinking": False}
        or backend.get("tito_allowed_append_roles") != ["tool", "user"]
    ):
        raise LaunchContractError("Codex backend is not the pinned xhigh profile")
    contract_hash = hashlib.sha256(_canonical(contract).rstrip(b"\n")).hexdigest()
    return {
        "YETO_CODEX_BINARY_PATH": str(contract["container_binary_path"]),
        "YETO_CODEX_BINARY_SHA256": str(contract["binary_sha256"]),
        "YETO_CODEX_BINARY_SIZE_BYTES": str(contract["binary_size_bytes"]),
        "YETO_CODEX_VERSION": str(contract["cli_version"]),
        "YETO_CODEX_APP_SERVER_PROTOCOL_REVISION": str(
            contract["app_server_protocol_revision"]
        ),
        "YETO_CODEX_APP_SERVER_SCHEMA_SHA256": str(
            contract["app_server_schema_sha256"]
        ),
        "YETO_CODEX_BASE_INSTRUCTIONS_SHA256": str(
            contract["base_instructions_sha256"]
        ),
        "YETO_CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": str(
            contract["terminal_exec_tool_schema_sha256"]
        ),
        "YETO_CODEX_SUBMIT_TOOL_SCHEMA_SHA256": str(
            contract["submit_tool_schema_sha256"]
        ),
        "YETO_CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256": str(
            contract["dynamic_tools_schema_sha256"]
        ),
        "YETO_CODEX_REASONING_EFFORT": "xhigh",
        "YETO_CODEX_BACKEND_MAX_TOKENS": str(backend["max_tokens"]),
        "YETO_CODEX_BACKEND_REASONING_EFFORT": "xhigh",
        "YETO_CODEX_BACKEND_THINKING": "enabled",
        "YETO_CODEX_CHAT_TEMPLATE": str(backend["chat_template"]),
        "YETO_CODEX_CHAT_TEMPLATE_KWARGS": json.dumps(
            kwargs, sort_keys=True, separators=(",", ":")
        ),
        "YETO_CODEX_TITO_ALLOWED_APPEND_ROLES": "tool,user",
        "YETO_CODEX_HARNESS_CONTRACT_SHA256": contract_hash,
    }


def _learner_argv(
    *,
    manifest: dict[str, Any],
    island_id: int,
    run_dir: str,
) -> list[str]:
    profile = manifest["profile"]
    model = manifest["model"]
    data = manifest["data"]
    provenance = manifest["provenance"]
    harness = manifest["harness"]
    ports = manifest["topology"]["islands"][island_id]["ports"]
    argv = [
        "python3",
        "-m",
        "yeto.rl.learner",
        "--model",
        model["repo"],
        "--rollout-model",
        model["repo"],
        "--model-revision",
        model["revision"],
        "--rollout-model-revision",
        model["revision"],
        "--data",
        data["container_train_path"],
        "--syncer",
        f"127.0.0.1:{manifest['syncer']['port']}",
        "--learner-id",
        str(island_id),
        "--num-learners",
        str(manifest["topology"]["island_count"]),
        "--reward-function",
        profile["reward_function"],
        "--reward-sha256",
        profile["reward_sha256"],
        "--source-sha256",
        provenance["yeto_source_sha256"],
        "--global-rounds",
        str(manifest["rounds"]),
        "--parameter-mode",
        "full",
        "--sync-preset",
        "dense-full",
        "--fragments",
        str(model["fragment_count"]),
        "--parameter-layout-sha256",
        model["parameter_layout_hash"],
        "--pipeline",
        "1",
        "--local-horizon",
        "1",
        "--total-fragment-steps",
        str(manifest["syncer"]["total_steps"]),
        "--groups-per-round",
        str(profile["groups_per_round"]),
        "--samples-per-group",
        str(profile["samples_per_group"]),
        "--over-sampling-batch-size",
        str(profile["over_sampling_batch_size"]),
        "--dynamic-sampling-filter-path",
        profile["dynamic_sampling_filter_path"],
        "--dynamic-sampling-max-replacements",
        str(profile["dynamic_sampling_max_replacements"]),
        "--secrlenv-max-infrastructure-replacements",
        str(profile["secrlenv_max_infrastructure_replacements"]),
        "--optimizer-steps",
        "1",
        "--rollout-max-response-len",
        str(profile["rollout_max_response_len"]),
        "--custom-generate-function-path",
        profile["custom_generate_function_path"],
        "--custom-agent-function-path",
        profile["custom_agent_function_path"],
        "--codex-harness-contract",
        json.dumps(harness["contract"], sort_keys=True, separators=(",", ":")),
        "--codex-reasoning-effort",
        "xhigh",
        "--apply-chat-template-kwargs",
        json.dumps(
            harness["contract"]["backend"]["chat_template_kwargs"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--agent-max-seq-len",
        str(profile["seq_len"]),
        "--use-session-server",
        "--session-server-ip",
        "127.0.0.1",
        "--session-server-port",
        str(ports["session_server"]),
        "--tito-model",
        profile["tito_model"],
        "--tito-allowed-append-roles",
        "tool",
        "user",
        "--completed-groups-path",
        f"{run_dir}/completed-groups.jsonl",
        "--event-tape",
        f"{run_dir}/learner-events.jsonl",
        "--audit-dir",
        f"{run_dir}/audit",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "2",
        "--tensor-parallel",
        "2",
        "--pipeline-parallel",
        "1",
        "--expert-parallel",
        "1",
        "--rollout-num-gpus",
        "2",
        "--rollout-num-gpus-per-engine",
        str(INFERENCE_TP),
        "--sglang-tp-size",
        str(INFERENCE_TP),
        "--sglang-dp-size",
        "1",
        "--sglang-ep-size",
        "1",
        "--sglang-mem-fraction-static",
        str(profile["sglang_mem_fraction_static"]),
        "--inner-lr",
        str(profile["inner_lr"]),
        "--seq-len",
        str(profile["seq_len"]),
        "--seed",
        str(profile["seed"]),
        "--miles-root",
        "/root/miles",
        "--miles-source-sha256",
        provenance["miles_source_sha256"],
        "--megatron-ref-load",
        model["container_checkpoint_path"],
    ]
    evaluation = manifest["evaluation"]
    if evaluation["enabled"]:
        argv.extend(
            (
                "--eval-data",
                evaluation["container_data_path"],
                "--eval-data-sha256",
                evaluation["data_sha256"],
                "--eval-dataset-name",
                evaluation["dataset_name"],
                "--eval-interval",
                str(evaluation["miles_eval_interval"]),
                "--eval-samples-per-prompt",
                str(evaluation["samples_per_prompt"]),
                "--eval-temperature",
                str(evaluation["temperature"]),
                "--eval-top-p",
                str(evaluation["top_p"]),
                "--eval-max-prompt-len",
                str(evaluation["max_prompt_len"]),
                "--eval-max-response-len",
                str(evaluation["max_response_len"]),
                "--eval-max-context-len",
                str(evaluation["max_context_len"]),
                "--eval-summary-path",
                f"{run_dir}/eval-summary.json",
            )
        )
    return argv


def _syncer_argv(
    binary: str,
    *,
    port: int,
    learners: int,
    total_steps: int,
    fragments: int,
    run: str,
) -> list[str]:
    return [
        binary,
        "--port",
        str(port),
        "--learners",
        str(learners),
        "--quorum",
        str(learners),
        "--quorum-timeout-s",
        "900",
        "--final-ack-timeout-s",
        "3600",
        "--grace-ms",
        "0",
        "--pipeline",
        "1",
        "--sync-interval-steps",
        "0",
        "--delta-correction",
        "none",
        "--total-steps",
        str(total_steps),
        "--policy-sweep-fragments",
        str(fragments),
        "--outer-lr",
        "1",
        "--outer-momentum",
        "0",
        "--max-base-lag",
        "0",
        "--learner-weight",
        "equal",
        "--checkpoint-path",
        f"{run}/syncer/state.ckpt",
        "--checkpoint-every",
        "1",
        "--resume",
        "--event-tape",
        f"{run}/syncer/events.jsonl",
    ]


def _build(args: argparse.Namespace) -> None:
    source = _validated_input(args.input)
    if source["schema"] != INPUT_SCHEMA:
        raise LaunchContractError("launch input has the wrong schema")
    launch_mode = source["launch_mode"]
    if launch_mode not in LAUNCH_MODES:
        raise LaunchContractError("launch_mode is not an allowlisted M1 mode")
    island_count = LAUNCH_MODES[launch_mode]
    run_id = source["run_id"]
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise LaunchContractError("run_id is not a safe bounded identifier")
    rounds = _validate_rounds(launch_mode, source["rounds"])
    repository = source["image_repository"]
    if repository != MILES_IMAGE_REPOSITORY:
        raise LaunchContractError("image_repository is not the attested public image")

    paths = _expect_keys(
        source["paths"],
        {
            "yeto_root",
            "miles_root",
            "model_root",
            "checkpoint_root",
            "probe_evidence",
            "train_data",
            "heldout_data",
            "data_manifest",
            "docker_image_inventory",
            "syncer_binary",
            "syncer_build_manifest",
            "run_root",
        },
        "paths",
    )
    resolved = {
        name: _absolute(value, f"paths.{name}") for name, value in paths.items()
    }
    for name in ("yeto_root", "miles_root", "model_root", "checkpoint_root"):
        _directory(resolved[name], name)
    run_root = resolved["run_root"]
    if run_root.exists() or run_root.is_symlink():
        raise LaunchContractError("run root must be fresh")
    _directory(run_root.parent, "run root parent")

    expected = _expect_keys(
        source["expected"],
        {
            "probe_evidence_sha256",
            "train_data_sha256",
            "heldout_data_sha256",
            "data_manifest_sha256",
            "docker_image_inventory_sha256",
            "syncer_binary_sha256",
            "syncer_build_manifest_sha256",
        },
        "expected",
    )
    for name, digest in expected.items():
        _expect_sha(digest, f"expected.{name}")
    pinned_expected = {
        "train_data_sha256": TRAIN_DATA_SHA256,
        "heldout_data_sha256": HELDOUT_DATA_SHA256,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "docker_image_inventory_sha256": DOCKER_IMAGE_INVENTORY_SHA256,
        "syncer_binary_sha256": SYNCER_BINARY_SHA256,
        "syncer_build_manifest_sha256": SYNCER_BUILD_MANIFEST_SHA256,
    }
    for name, pinned in pinned_expected.items():
        if expected[name] != pinned:
            raise LaunchContractError(f"expected.{name} is not the sealed M1 artifact")
    for name, path_name in (
        ("probe_evidence_sha256", "probe_evidence"),
        ("train_data_sha256", "train_data"),
        ("heldout_data_sha256", "heldout_data"),
        ("data_manifest_sha256", "data_manifest"),
        ("docker_image_inventory_sha256", "docker_image_inventory"),
        ("syncer_binary_sha256", "syncer_binary"),
        ("syncer_build_manifest_sha256", "syncer_build_manifest"),
    ):
        _attest_file(resolved[path_name], expected[name], path_name)
    _regular(resolved["syncer_binary"], "syncer binary", executable=True)
    if not _is_linux_x86_64_elf(resolved["syncer_binary"]):
        raise LaunchContractError("syncer binary must be a staged Linux x86_64 ELF")
    syncer_build = _validate_syncer_build_manifest(
        resolved["syncer_build_manifest"],
        expected_sha256=expected["syncer_build_manifest_sha256"],
        binary_path=resolved["syncer_binary"],
        binary_sha256=expected["syncer_binary_sha256"],
    )

    yeto_sha, miles_sha = _source_hashes(resolved["yeto_root"], resolved["miles_root"])
    probe = _load_json(resolved["probe_evidence"], canonical=True, private=True)
    digest = probe.get("miles_image_digest") if isinstance(probe, dict) else None
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise LaunchContractError("probe image digest is invalid")
    if digest != MILES_IMAGE_DIGEST:
        raise LaunchContractError("probe image is not the attested amd64 Miles image")
    probe = _validate_probe(
        probe,
        image_digest=digest,
        yeto_source_sha256=yeto_sha,
        miles_source_sha256=miles_sha,
    )
    conversion_path = resolved["checkpoint_root"] / "conversion-manifest.json"
    conversion_sha = probe["conversion_manifest"]["sha256"]
    conversion_manifest = _validate_conversion_artifacts(
        conversion_path,
        expected_sha256=conversion_sha,
        image_digest=digest,
        model_root=resolved["model_root"],
        checkpoint_root=resolved["checkpoint_root"],
    )
    if (
        len(conversion_manifest["model_files"])
        != probe["conversion_manifest"]["model_file_count"]
        or len(conversion_manifest["checkpoint_files"])
        != probe["conversion_manifest"]["checkpoint_file_count"]
        or conversion_manifest["conversion_source_aggregate_sha256"]
        != probe["conversion_manifest"]["conversion_source_aggregate_sha256"]
    ):
        raise LaunchContractError("probe conversion inventory differs")
    config_path = resolved["model_root"] / "config.json"
    _attest_file(config_path, probe["model_config_sha256"], "model config")

    uuids, names = _hardware_contract(source["hardware"], island_count=island_count)
    syncer_port, island_ports = _port_contract(
        source["ports"], island_count=island_count
    )
    profile = _validate_profile_semantics(source["profile"])
    reward_path = _absolute(profile["reward_source_path"], "reward source")
    reward_sha = _expect_sha(profile["reward_sha256"], "reward source")
    _attest_file(reward_path, reward_sha, "reward source")
    if (
        type(profile["seq_len"]) is not int
        or not 512 <= profile["seq_len"] <= 8192
        or type(profile["rollout_max_response_len"]) is not int
        or not 1 <= profile["rollout_max_response_len"] <= profile["seq_len"]
        or type(profile["groups_per_round"]) is not int
        or profile["groups_per_round"] < 1
        or type(profile["samples_per_group"]) is not int
        or profile["samples_per_group"] < 2
        or type(profile["over_sampling_batch_size"]) is not int
        or profile["over_sampling_batch_size"] <= profile["groups_per_round"]
        or type(profile["seed"]) is not int
        or type(profile["sglang_mem_fraction_static"]) not in (int, float)
        or not 0 < profile["sglang_mem_fraction_static"] < 0.8
        or type(profile["inner_lr"]) not in (int, float)
        or not 0 < profile["inner_lr"] <= 1e-3
        or profile["secrlenv_max_infrastructure_replacements"] != 1
        or profile["dynamic_sampling_max_replacements"] != 0
        or profile["samples_per_group"] != 3
        or profile["tito_model"] != "qwen35"
        or profile["reward_function"] != REWARD_FUNCTION
        or profile["custom_generate_function_path"] != CUSTOM_GENERATE_FUNCTION
        or profile["custom_agent_function_path"] != CUSTOM_AGENT_FUNCTION
        or profile["dynamic_sampling_filter_path"] != DYNAMIC_FILTER_FUNCTION
    ):
        raise LaunchContractError("learner smoke profile is unsafe or malformed")

    harness_input = _expect_keys(
        source["harness"], {"contract_path", "contract_sha256", "artifacts"}, "harness"
    )
    harness_path = _absolute(harness_input["contract_path"], "harness contract")
    harness_sha = _expect_sha(harness_input["contract_sha256"], "harness contract")
    _attest_file(harness_path, harness_sha, "harness contract")
    harness_contract = _load_json(harness_path, canonical=True, private=True)
    if not isinstance(harness_contract, dict):
        raise LaunchContractError("harness contract is not an object")
    harness_env = _harness_environment(harness_contract)
    if (
        harness_contract["backend"].get("max_tokens")
        != profile["rollout_max_response_len"]
    ):
        raise LaunchContractError(
            "Codex backend max_tokens must equal rollout_max_response_len"
        )
    harness_artifacts = _artifact_contract(harness_input["artifacts"])
    _validate_harness_artifact_binding(harness_contract, harness_artifacts)

    secrlenv = _expect_keys(
        source["secrlenv"],
        {
            "host_health_url",
            "container_url",
            "task_pack_sha256",
            "bearer_token_host_path",
            "bearer_token_sha256",
            "daemon_contract_path",
            "daemon_contract_sha256",
        },
        "secrlenv",
    )
    for name in ("host_health_url", "container_url"):
        if not isinstance(secrlenv[name], str) or not secrlenv[name].startswith(
            "http://"
        ):
            raise LaunchContractError(f"secrlenv {name} must be an explicit HTTP URL")
    token_path = _absolute(secrlenv["bearer_token_host_path"], "SecRLEnv token")
    daemon_contract_path = _absolute(
        secrlenv["daemon_contract_path"], "SecRLEnv daemon contract"
    )
    _attest_file(
        token_path,
        _expect_sha(secrlenv["bearer_token_sha256"], "SecRLEnv token"),
        "SecRLEnv token",
    )
    _attest_file(
        daemon_contract_path,
        _expect_sha(secrlenv["daemon_contract_sha256"], "SecRLEnv daemon contract"),
        "SecRLEnv daemon contract",
    )
    _expect_sha(secrlenv["task_pack_sha256"], "SecRLEnv task pack")
    if secrlenv["task_pack_sha256"] != TASK_PACK_SHA256:
        raise LaunchContractError("SecRLEnv task pack is not the sealed M1 pack")
    daemon_contract = _validate_daemon_contract(
        daemon_contract_path,
        expected_sha256=secrlenv["daemon_contract_sha256"],
        task_pack_sha256=secrlenv["task_pack_sha256"],
    )
    daemon_base_url = f"http://127.0.0.1:{daemon_contract['port']}"
    if (
        secrlenv["container_url"] != daemon_base_url
        or secrlenv["host_health_url"] != f"{daemon_base_url}/healthz"
    ):
        raise LaunchContractError(
            "SecRLEnv URLs must use the attested host-loopback daemon"
        )
    launch_ports = {
        syncer_port,
        *(port for row in island_ports for port in row.values()),
    }
    if daemon_contract["port"] in launch_ports:
        raise LaunchContractError("SecRLEnv daemon port overlaps a launch listener")
    task_images = _validate_data_bundle(
        resolved["data_manifest"],
        resolved["docker_image_inventory"],
        task_pack_sha256=secrlenv["task_pack_sha256"],
        train_sha256=expected["train_data_sha256"],
        heldout_sha256=expected["heldout_data_sha256"],
        inventory_sha256=expected["docker_image_inventory_sha256"],
    )

    fragment_count = probe["derived_fragment_count"]
    total_steps = rounds * fragment_count
    islands = []
    for island_id in range(island_count):
        start = island_id * GPUS_PER_ISLAND
        local = uuids[start : start + GPUS_PER_ISLAND]
        islands.append(
            {
                "island_id": island_id,
                "container_name": names[island_id],
                "host_gpu_indices": list(range(start, start + GPUS_PER_ISLAND)),
                "gpu_uuids": local,
                "trainer_gpu_uuids": local[:TRAINER_GPUS],
                "inference_gpu_uuids": local[TRAINER_GPUS:],
                "ports": island_ports[island_id],
                "host_run_dir": str(run_root / f"island-{island_id}"),
                "container_run_dir": "/evidence",
            }
        )
    data_contract = {
        "host_train_path": str(resolved["train_data"]),
        "container_train_path": "/workspace/data/train.jsonl",
        "train_sha256": expected["train_data_sha256"],
        "host_heldout_path": str(resolved["heldout_data"]),
        "container_heldout_path": "/workspace/data/heldout.jsonl",
        "heldout_sha256": expected["heldout_data_sha256"],
        "manifest_path": str(resolved["data_manifest"]),
        "manifest_sha256": expected["data_manifest_sha256"],
        "docker_image_inventory_path": str(resolved["docker_image_inventory"]),
        "docker_image_inventory_sha256": expected["docker_image_inventory_sha256"],
        "docker_image_inventory": task_images,
    }
    evaluation = _evaluation_contract(
        launch_mode=launch_mode,
        rounds=rounds,
        profile=profile,
        data=data_contract,
        run_root=run_root,
        yeto_root=resolved["yeto_root"],
        island_count=island_count,
    )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "launch_mode": launch_mode,
        "run_id": run_id,
        "rounds": rounds,
        "image": {
            "repository": repository,
            "digest": digest,
            "reference": f"{repository}@{digest}",
        },
        "launch_bundle": _launch_bundle(resolved["yeto_root"]),
        "provenance": {
            "probe_evidence_path": str(resolved["probe_evidence"]),
            "probe_evidence_sha256": expected["probe_evidence_sha256"],
            "yeto_root": str(resolved["yeto_root"]),
            "yeto_source_sha256": yeto_sha,
            "miles_root": str(resolved["miles_root"]),
            "miles_source_sha256": miles_sha,
            "megatron_bridge": probe["megatron_bridge"],
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": probe["model_revision"],
            "host_model_path": str(resolved["model_root"]),
            "container_model_path": "/models/hf",
            "container_hf_cache_snapshot_path": MODEL_CACHE_SNAPSHOT,
            "config_sha256": probe["model_config_sha256"],
            "host_checkpoint_path": str(resolved["checkpoint_root"]),
            "container_checkpoint_path": "/models/torch_dist",
            "conversion_manifest_sha256": conversion_sha,
            "conversion_manifest": conversion_manifest,
            "parameter_layout_hash": probe["parameter_layout_hash"],
            "fragment_count": fragment_count,
            "observed_max_fragment_bytes": probe["observed_max_fragment_bytes"],
        },
        "data": data_contract,
        "evaluation": evaluation,
        "profile": dict(profile),
        "harness": {
            "contract": harness_contract,
            "contract_path": str(harness_path),
            "contract_sha256": harness_sha,
            "artifacts": harness_artifacts,
            "environment": harness_env,
        },
        "secrlenv": {**secrlenv, "daemon_contract": daemon_contract},
        "topology": {
            "host_gpu_count": island_count * GPUS_PER_ISLAND,
            "island_count": island_count,
            "gpus_per_island": 4,
            "trainer_tp": 2,
            "inference_engines": INFERENCE_ENGINES,
            "inference_tp": INFERENCE_TP,
            "cross_island_collective": False,
            "islands": islands,
        },
        "syncer": {
            "binary_path": str(resolved["syncer_binary"]),
            "binary_sha256": expected["syncer_binary_sha256"],
            "build_manifest_path": str(resolved["syncer_build_manifest"]),
            "build_manifest_sha256": expected["syncer_build_manifest_sha256"],
            "build_manifest": syncer_build,
            "platform": "linux-x86_64",
            "port": syncer_port,
            "total_steps": total_steps,
            "policy_sweep_fragments": fragment_count,
        },
        "launch": {"host_run_root": str(run_root), "uses_ssh_harness": False},
        "learners": [],
    }
    manifest["syncer"]["argv"] = _syncer_argv(
        manifest["syncer"]["binary_path"],
        port=syncer_port,
        learners=island_count,
        total_steps=total_steps,
        fragments=fragment_count,
        run=str(run_root),
    )
    manifest["learners"] = [
        {
            "island_id": island_id,
            "argv": _learner_argv(
                manifest=manifest, island_id=island_id, run_dir="/evidence"
            ),
        }
        for island_id in range(island_count)
    ]
    _validate_manifest(manifest)
    _write_exclusive(args.output, manifest)
    print(
        json.dumps(
            {
                "fragment_count": fragment_count,
                "manifest_sha256": _sha256(args.output),
                "rounds": rounds,
                "total_steps": total_steps,
            },
            sort_keys=True,
        )
    )


def _write_exclusive(path: Path, payload: Any) -> None:
    if (
        path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise LaunchContractError(
            "manifest output must be a fresh path in a real directory"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = _canonical(payload)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            if stream.write(encoded) != len(encoded):
                raise LaunchContractError("short manifest write")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _flag_value(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1:
        raise LaunchContractError(f"learner argv must contain {flag} exactly once")
    index = argv.index(flag)
    if index + 1 == len(argv) or argv[index + 1].startswith("--"):
        raise LaunchContractError(f"learner argv flag has no value: {flag}")
    return argv[index + 1]


def _validate_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise LaunchContractError("launch manifest has the wrong schema")
    _expect_keys(
        payload,
        {
            "schema",
            "launch_mode",
            "run_id",
            "rounds",
            "image",
            "launch_bundle",
            "provenance",
            "model",
            "data",
            "evaluation",
            "profile",
            "harness",
            "secrlenv",
            "topology",
            "syncer",
            "launch",
            "learners",
        },
        "launch manifest",
    )
    launch_mode = payload.get("launch_mode")
    if launch_mode not in LAUNCH_MODES:
        raise LaunchContractError("launch manifest mode is invalid")
    _validate_rounds(launch_mode, payload.get("rounds"))
    island_count = LAUNCH_MODES[launch_mode]
    image = _expect_keys(
        payload.get("image"), {"repository", "digest", "reference"}, "image"
    )
    if (
        not isinstance(image, dict)
        or image.get("repository") != MILES_IMAGE_REPOSITORY
        or image.get("digest") != MILES_IMAGE_DIGEST
        or image.get("reference") != f"{image.get('repository')}@{image.get('digest')}"
    ):
        raise LaunchContractError("launch manifest image is not digest-pinned")
    _validate_launch_bundle(payload.get("launch_bundle"))
    model = _expect_keys(
        payload.get("model"),
        {
            "repo",
            "revision",
            "host_model_path",
            "container_model_path",
            "container_hf_cache_snapshot_path",
            "config_sha256",
            "host_checkpoint_path",
            "container_checkpoint_path",
            "conversion_manifest_sha256",
            "conversion_manifest",
            "parameter_layout_hash",
            "fragment_count",
            "observed_max_fragment_bytes",
        },
        "launch manifest model",
    )
    if (
        not isinstance(model, dict)
        or model.get("repo") != MODEL_REPO
        or model.get("revision") != MODEL_REVISION
        or model.get("container_model_path") != "/models/hf"
        or model.get("container_hf_cache_snapshot_path") != MODEL_CACHE_SNAPSHOT
        or model.get("container_checkpoint_path") != "/models/torch_dist"
        or not isinstance(model.get("conversion_manifest"), dict)
    ):
        raise LaunchContractError("launch manifest model differs")
    _absolute(model["host_model_path"], "manifest host model")
    _absolute(model["host_checkpoint_path"], "manifest host checkpoint")
    for name in (
        "config_sha256",
        "conversion_manifest_sha256",
        "parameter_layout_hash",
    ):
        _expect_sha(model[name], f"manifest model {name}")
    profile = _validate_profile_semantics(payload.get("profile"))
    harness = _expect_keys(
        payload.get("harness"),
        {"contract", "contract_path", "contract_sha256", "artifacts", "environment"},
        "manifest harness",
    )
    expected_harness_environment = _harness_environment(harness["contract"])
    if harness["environment"] != expected_harness_environment:
        raise LaunchContractError("manifest harness environment differs")
    if not isinstance(harness["artifacts"], list):
        raise LaunchContractError("manifest harness artifacts are malformed")
    _validate_harness_artifact_binding(harness["contract"], harness["artifacts"])
    if (
        harness["contract"]["backend"].get("max_tokens")
        != profile["rollout_max_response_len"]
    ):
        raise LaunchContractError("manifest Codex backend token budget differs")
    fragments = model.get("fragment_count")
    if type(fragments) is not int or fragments < 2:
        raise LaunchContractError("launch manifest fragment count is invalid")
    syncer = _expect_keys(
        payload.get("syncer"),
        {
            "binary_path",
            "binary_sha256",
            "build_manifest_path",
            "build_manifest_sha256",
            "build_manifest",
            "platform",
            "port",
            "total_steps",
            "policy_sweep_fragments",
            "argv",
        },
        "syncer",
    )
    if (
        not isinstance(syncer, dict)
        or syncer.get("platform") != "linux-x86_64"
        or syncer.get("binary_sha256") != SYNCER_BINARY_SHA256
        or syncer.get("build_manifest_sha256") != SYNCER_BUILD_MANIFEST_SHA256
        or type(syncer.get("port")) is not int
        or not 1024 <= syncer["port"] <= 65535
        or syncer.get("policy_sweep_fragments") != fragments
        or syncer.get("total_steps") != payload["rounds"] * fragments
    ):
        raise LaunchContractError("launch manifest syncer budget differs")
    data = _expect_keys(
        payload.get("data"),
        {
            "host_train_path",
            "container_train_path",
            "train_sha256",
            "host_heldout_path",
            "container_heldout_path",
            "heldout_sha256",
            "manifest_path",
            "manifest_sha256",
            "docker_image_inventory_path",
            "docker_image_inventory_sha256",
            "docker_image_inventory",
        },
        "data",
    )
    if (
        data["container_train_path"] != "/workspace/data/train.jsonl"
        or data["container_heldout_path"] != "/workspace/data/heldout.jsonl"
        or data["train_sha256"] != TRAIN_DATA_SHA256
        or data["heldout_sha256"] != HELDOUT_DATA_SHA256
        or data["manifest_sha256"] != DATA_MANIFEST_SHA256
        or data["docker_image_inventory_sha256"] != DOCKER_IMAGE_INVENTORY_SHA256
    ):
        raise LaunchContractError("launch manifest data is not the sealed M1 input")
    launch = _expect_keys(
        payload.get("launch"), {"host_run_root", "uses_ssh_harness"}, "launch"
    )
    run_root = _absolute(launch["host_run_root"], "launch run root")
    expected_evaluation = _evaluation_contract(
        launch_mode=launch_mode,
        rounds=payload["rounds"],
        profile=profile,
        data=data,
        run_root=run_root,
        yeto_root=_absolute(
            payload.get("provenance", {}).get("yeto_root"),
            "manifest Yeto root",
        ),
        island_count=island_count,
    )
    if payload.get("evaluation") != expected_evaluation:
        raise LaunchContractError("launch manifest evaluation contract differs")
    secrlenv = payload.get("secrlenv")
    if (
        not isinstance(secrlenv, dict)
        or secrlenv.get("task_pack_sha256") != TASK_PACK_SHA256
    ):
        raise LaunchContractError("launch manifest task pack differs")
    daemon_contract = secrlenv.get("daemon_contract")
    daemon_port = (
        daemon_contract.get("port") if isinstance(daemon_contract, dict) else None
    )
    daemon_base_url = f"http://127.0.0.1:{daemon_port}"
    if (
        type(daemon_port) is not int
        or not 1024 <= daemon_port <= 65535
        or secrlenv.get("container_url") != daemon_base_url
        or secrlenv.get("host_health_url") != f"{daemon_base_url}/healthz"
    ):
        raise LaunchContractError("launch manifest daemon URLs are not host-loopback")
    expected_syncer_argv = _syncer_argv(
        syncer.get("binary_path"),
        port=syncer.get("port"),
        learners=island_count,
        total_steps=syncer.get("total_steps"),
        fragments=fragments,
        run=payload.get("launch", {}).get("host_run_root"),
    )
    if syncer.get("argv") != expected_syncer_argv:
        raise LaunchContractError("launch manifest syncer argv differs")
    topology = _expect_keys(
        payload.get("topology"),
        {
            "host_gpu_count",
            "island_count",
            "gpus_per_island",
            "trainer_tp",
            "inference_engines",
            "inference_tp",
            "cross_island_collective",
            "islands",
        },
        "topology",
    )
    if (
        not isinstance(topology, dict)
        or topology.get("host_gpu_count") != island_count * GPUS_PER_ISLAND
        or topology.get("island_count") != island_count
        or topology.get("gpus_per_island") != 4
        or topology.get("trainer_tp") != 2
        or topology.get("inference_engines") != INFERENCE_ENGINES
        or topology.get("inference_tp") != INFERENCE_TP
        or topology.get("cross_island_collective") is not False
    ):
        raise LaunchContractError("launch manifest topology differs")
    islands = topology.get("islands")
    learners = payload.get("learners")
    if (
        not isinstance(islands, list)
        or not isinstance(learners, list)
        or len(islands) != island_count
        or len(learners) != island_count
    ):
        raise LaunchContractError("launch manifest island count differs from its mode")
    all_uuids: list[str] = []
    all_ports = [syncer.get("port")]
    run_dirs = []
    for island_id, (island, learner) in enumerate(zip(islands, learners, strict=True)):
        island = _expect_keys(
            island,
            {
                "island_id",
                "container_name",
                "host_gpu_indices",
                "gpu_uuids",
                "trainer_gpu_uuids",
                "inference_gpu_uuids",
                "ports",
                "host_run_dir",
                "container_run_dir",
            },
            f"island {island_id}",
        )
        learner = _expect_keys(learner, {"island_id", "argv"}, f"learner {island_id}")
        if (
            island.get("island_id") != island_id
            or learner.get("island_id") != island_id
            or island.get("host_gpu_indices")
            != list(
                range(
                    island_id * GPUS_PER_ISLAND,
                    (island_id + 1) * GPUS_PER_ISLAND,
                )
            )
            or island.get("container_run_dir") != "/evidence"
            or island.get("host_run_dir") != str(run_root / f"island-{island_id}")
        ):
            raise LaunchContractError("launch manifest island ordering differs")
        uuids = island.get("gpu_uuids")
        if (
            not isinstance(uuids, list)
            or len(uuids) != 4
            or island.get("trainer_gpu_uuids") != uuids[:2]
            or island.get("inference_gpu_uuids") != uuids[2:]
        ):
            raise LaunchContractError("island trainer/inference GPU split differs")
        all_uuids.extend(uuids)
        ports = island.get("ports")
        _expect_keys(ports, _PORT_KEYS, f"manifest island {island_id} ports")
        if any(
            type(port) is not int or not 1024 <= port <= 65535
            for port in ports.values()
        ):
            raise LaunchContractError("manifest island port is invalid")
        all_ports.extend(ports.values())
        run_dirs.append(_absolute(island.get("host_run_dir"), "island run dir"))
        argv = learner.get("argv")
        if not isinstance(argv, list) or argv[:3] != [
            "python3",
            "-m",
            "yeto.rl.learner",
        ]:
            raise LaunchContractError(
                "learner does not directly invoke yeto.rl.learner"
            )
        if argv != _learner_argv(
            manifest=payload,
            island_id=island_id,
            run_dir=island.get("container_run_dir"),
        ):
            raise LaunchContractError("learner direct argv differs from its manifest")
        expected_flags = {
            "--model": MODEL_REPO,
            "--rollout-model": MODEL_REPO,
            "--model-revision": MODEL_REVISION,
            "--rollout-model-revision": MODEL_REVISION,
            "--learner-id": str(island_id),
            "--num-learners": str(island_count),
            "--global-rounds": str(payload["rounds"]),
            "--parameter-mode": "full",
            "--sync-preset": "dense-full",
            "--fragments": str(fragments),
            "--local-horizon": "1",
            "--total-fragment-steps": str(syncer["total_steps"]),
            "--actor-num-nodes": "1",
            "--actor-num-gpus-per-node": "2",
            "--tensor-parallel": "2",
            "--pipeline-parallel": "1",
            "--rollout-num-gpus": "2",
            "--rollout-num-gpus-per-engine": str(INFERENCE_TP),
            "--sglang-tp-size": str(INFERENCE_TP),
            "--tito-model": "qwen35",
            "--codex-reasoning-effort": "xhigh",
            "--apply-chat-template-kwargs": '{"clear_thinking":false}',
            "--megatron-ref-load": "/models/torch_dist",
            "--miles-source-sha256": payload["provenance"]["miles_source_sha256"],
            "--source-sha256": payload["provenance"]["yeto_source_sha256"],
            "--session-server-port": str(ports["session_server"]),
        }
        evaluation = payload["evaluation"]
        if evaluation["enabled"]:
            expected_flags.update(
                {
                    "--eval-data": evaluation["container_data_path"],
                    "--eval-data-sha256": evaluation["data_sha256"],
                    "--eval-dataset-name": evaluation["dataset_name"],
                    "--eval-interval": str(evaluation["miles_eval_interval"]),
                    "--eval-samples-per-prompt": str(evaluation["samples_per_prompt"]),
                    "--eval-temperature": str(evaluation["temperature"]),
                    "--eval-top-p": str(evaluation["top_p"]),
                    "--eval-max-prompt-len": str(evaluation["max_prompt_len"]),
                    "--eval-max-response-len": str(evaluation["max_response_len"]),
                    "--eval-max-context-len": str(evaluation["max_context_len"]),
                    "--eval-summary-path": "/evidence/eval-summary.json",
                }
            )
        elif any(flag.startswith("--eval-") for flag in argv):
            raise LaunchContractError("single-island gate unexpectedly enables eval")
        for flag, expected in expected_flags.items():
            if _flag_value(argv, flag) != expected:
                raise LaunchContractError(f"learner {island_id} {flag} differs")
        if "--colocate" in argv:
            raise LaunchContractError("dense full learner must not colocate inference")
    if len(set(all_uuids)) != island_count * GPUS_PER_ISLAND or len(
        set(all_ports)
    ) != len(all_ports):
        raise LaunchContractError("island GPU UUIDs or ports overlap")
    if daemon_port in all_ports:
        raise LaunchContractError("daemon and launch ports overlap")
    resolved_run_dirs = [path.resolve(strict=False) for path in run_dirs]
    for index, first in enumerate(resolved_run_dirs):
        for second in resolved_run_dirs[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise LaunchContractError("island run directories overlap")
    _hardware_contract(
        {
            "gpu_uuids": all_uuids,
            "container_names": [island["container_name"] for island in islands],
        },
        island_count=island_count,
    )
    if launch["uses_ssh_harness"] is not False:
        raise LaunchContractError("direct manifest unexpectedly selects ssh_harness")
    return payload


def _load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected = _expect_sha(expected_sha256, "expected manifest")
    if _sha256(path) != expected:
        raise LaunchContractError("launch manifest SHA256 differs")
    payload = _load_json(path, canonical=True, private=True)
    return _validate_manifest(payload)


def _verify(args: argparse.Namespace) -> None:
    payload = _load_manifest(args.manifest, args.expected_sha256)
    if args.live:
        _verify_live_inputs(payload)
    print(
        json.dumps(
            {"manifest_sha256": args.expected_sha256, "verified": True}, sort_keys=True
        )
    )


def _report(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest, args.expected_sha256)
    if manifest["evaluation"]["aggregate_report_path"] is None:
        raise LaunchContractError("single-island gate has no terminal report")
    _verify_live_inputs(manifest)
    try:
        import m1_dense_full_final_report as final_report
    except ImportError as exc:
        raise LaunchContractError(
            f"terminal M1 reconciler is unavailable: {exc}"
        ) from exc
    try:
        if final_report.FINAL_REPORT_SCHEMA != FINAL_REPORT_SCHEMA:
            raise final_report.FinalReportError(
                "terminal report schema differs from the launcher"
            )
        report = final_report.build_final_report(
            manifest,
            args.expected_sha256,
            daemon_health=_secrlenv_ready(manifest),
        )
        output = Path(manifest["evaluation"]["aggregate_report_path"])
        final_report.write_report_exclusive(output, report)
    except final_report.FinalReportError as exc:
        raise LaunchContractError(f"terminal M1 reconciliation failed: {exc}") from exc
    print(
        json.dumps(
            {
                "final_policy_hash": report["final_policy"]["policy_hash"],
                "manifest_sha256": args.expected_sha256,
                "report_path": str(output),
                "report_sha256": _sha256(output),
                "status": "passed",
            },
            sort_keys=True,
        )
    )


def _verify_live_inputs(manifest: dict[str, Any]) -> None:
    provenance = manifest["provenance"]
    yeto_root = Path(provenance["yeto_root"])
    miles_root = Path(provenance["miles_root"])
    yeto_sha, miles_sha = _source_hashes(yeto_root, miles_root)
    if (
        yeto_sha != provenance["yeto_source_sha256"]
        or miles_sha != provenance["miles_source_sha256"]
    ):
        raise LaunchContractError(
            "Yeto or Miles source changed after manifest creation"
        )
    _attest_launch_bundle(manifest["launch_bundle"], yeto_root)
    probe_path = Path(provenance["probe_evidence_path"])
    _attest_file(
        probe_path, provenance["probe_evidence_sha256"], "metadata probe evidence"
    )
    probe = _validate_probe(
        _load_json(probe_path, canonical=True, private=True),
        image_digest=manifest["image"]["digest"],
        yeto_source_sha256=yeto_sha,
        miles_source_sha256=miles_sha,
    )
    model = manifest["model"]
    if (
        probe["model_revision"] != model["revision"]
        or probe["model_config_sha256"] != model["config_sha256"]
        or probe["parameter_layout_hash"] != model["parameter_layout_hash"]
        or probe["derived_fragment_count"] != model["fragment_count"]
        or probe["observed_max_fragment_bytes"] != model["observed_max_fragment_bytes"]
        or probe["conversion_manifest"]["sha256"] != model["conversion_manifest_sha256"]
        or probe["megatron_bridge"] != provenance["megatron_bridge"]
    ):
        raise LaunchContractError("metadata probe differs from the launch manifest")
    _attest_file(
        Path(model["host_model_path"]) / "config.json",
        model["config_sha256"],
        "model config",
    )
    conversion = _validate_conversion_artifacts(
        Path(model["host_checkpoint_path"]) / "conversion-manifest.json",
        expected_sha256=model["conversion_manifest_sha256"],
        image_digest=manifest["image"]["digest"],
        model_root=Path(model["host_model_path"]),
        checkpoint_root=Path(model["host_checkpoint_path"]),
    )
    if conversion != model["conversion_manifest"]:
        raise LaunchContractError("conversion manifest changed semantically")
    data = manifest["data"]
    for path_name, hash_name, label in (
        ("host_train_path", "train_sha256", "training data"),
        ("host_heldout_path", "heldout_sha256", "heldout data"),
        ("manifest_path", "manifest_sha256", "data manifest"),
        (
            "docker_image_inventory_path",
            "docker_image_inventory_sha256",
            "Docker image inventory",
        ),
    ):
        _attest_file(Path(data[path_name]), data[hash_name], label)
    syncer = manifest["syncer"]
    binary = Path(syncer["binary_path"])
    _attest_file(binary, syncer["binary_sha256"], "syncer binary")
    _regular(binary, "syncer binary", executable=True)
    if not _is_linux_x86_64_elf(binary):
        raise LaunchContractError("syncer binary is not Linux x86_64")
    syncer_build = _validate_syncer_build_manifest(
        Path(syncer["build_manifest_path"]),
        expected_sha256=syncer["build_manifest_sha256"],
        binary_path=binary,
        binary_sha256=syncer["binary_sha256"],
    )
    if syncer_build != syncer["build_manifest"]:
        raise LaunchContractError("syncer build manifest changed semantically")
    harness = manifest["harness"]
    _attest_file(
        Path(harness["contract_path"]), harness["contract_sha256"], "harness contract"
    )
    for row in harness["artifacts"]:
        _attest_file(Path(row["host_path"]), row["sha256"], "harness artifact")
    secrlenv = manifest["secrlenv"]
    _attest_file(
        Path(secrlenv["bearer_token_host_path"]),
        secrlenv["bearer_token_sha256"],
        "SecRLEnv token",
    )
    daemon = _validate_daemon_contract(
        Path(secrlenv["daemon_contract_path"]),
        expected_sha256=secrlenv["daemon_contract_sha256"],
        task_pack_sha256=secrlenv["task_pack_sha256"],
    )
    if daemon != secrlenv["daemon_contract"]:
        raise LaunchContractError("SecRLEnv daemon contract changed semantically")
    inventory = _validate_data_bundle(
        Path(data["manifest_path"]),
        Path(data["docker_image_inventory_path"]),
        task_pack_sha256=secrlenv["task_pack_sha256"],
        train_sha256=data["train_sha256"],
        heldout_sha256=data["heldout_sha256"],
        inventory_sha256=data["docker_image_inventory_sha256"],
    )
    if inventory != data["docker_image_inventory"]:
        raise LaunchContractError("SecRLEnv image inventory changed semantically")
    profile = manifest["profile"]
    _attest_file(
        Path(profile["reward_source_path"]),
        profile["reward_sha256"],
        "SecRLEnv reward source",
    )


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)


def _assert_host_ready(manifest: dict[str, Any]) -> dict[str, Any]:
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise LaunchContractError("the launch host must be Linux x86_64")
    for executable in ("docker", "nvidia-smi", "pgrep"):
        if shutil.which(executable) is None:
            raise LaunchContractError(f"launch host lacks {executable}")
    running = _run(["docker", "ps", "-q"]).stdout.split()
    if running:
        raise LaunchContractError("dedicated H100 host already has running containers")
    all_containers = _run(["docker", "ps", "-aq"]).stdout.split()
    names = {island["container_name"] for island in manifest["topology"]["islands"]}
    if all_containers:
        inspected = json.loads(_run(["docker", "inspect", *all_containers]).stdout)
        for row in inspected:
            labels = row.get("Config", {}).get("Labels") or {}
            existing_name = str(row.get("Name", "")).removeprefix("/")
            if labels.get("yeto.protected") == "true" or existing_name in names:
                raise LaunchContractError("protected/reserved container already exists")
    ray = subprocess.run(
        ["pgrep", "-f", r"(^|/)(raylet|gcs_server)( |$)"],
        text=True,
        capture_output=True,
        check=False,
    )
    if ray.returncode == 0:
        raise LaunchContractError("launch host already has a Ray runtime")
    syncer_help = subprocess.run(
        [manifest["syncer"]["binary_path"], "--help"],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if (
        syncer_help.returncode != 0
        or "--policy-sweep-fragments" not in syncer_help.stdout
    ):
        raise LaunchContractError("staged syncer is not executable with dense sweeps")
    image = manifest["image"]
    image_data = json.loads(
        _run(["docker", "image", "inspect", image["reference"]]).stdout
    )
    repo_digests = set(image_data[0].get("RepoDigests") or [])
    if image["reference"] not in repo_digests:
        raise LaunchContractError(
            "local Docker image does not expose the exact RepoDigest"
        )
    for row in manifest["data"]["docker_image_inventory"]["images"]:
        task_image = json.loads(
            _run(["docker", "image", "inspect", row["immutable"]]).stdout
        )
        if len(task_image) != 1 or task_image[0].get("Id") != row["image_id"]:
            raise LaunchContractError("local SecRLEnv task image identity differs")
    daemon = manifest["secrlenv"]["daemon_contract"]
    for label, reference_key, id_key in (
        ("operator", "operator_image", "operator_image_id"),
        ("DinD debug", "dind_image", "dind_image_id"),
    ):
        inspected_image = json.loads(
            _run(["docker", "image", "inspect", daemon[reference_key]]).stdout
        )
        if len(inspected_image) != 1 or inspected_image[0].get("Id") != daemon[id_key]:
            raise LaunchContractError(f"local SecRLEnv {label} image identity differs")

    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.splitlines()
    expected_gpu_count = manifest["topology"]["host_gpu_count"]
    if len(query) != expected_gpu_count:
        raise LaunchContractError(
            "launch host GPU count differs from the explicit launch mode"
        )
    observed = []
    for expected_index, line in enumerate(query):
        parts = [item.strip() for item in line.split(",", 3)]
        if (
            len(parts) != 4
            or int(parts[0]) != expected_index
            or "H100" not in parts[2]
            or int(parts[3]) != 0
        ):
            raise LaunchContractError("GPU inventory is not an idle indexed 8xH100")
        observed.append(parts[1])
    expected_uuids = [
        uuid
        for island in manifest["topology"]["islands"]
        for uuid in island["gpu_uuids"]
    ]
    if observed != expected_uuids:
        raise LaunchContractError("live GPU UUID ordering differs from the manifest")
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if apps.returncode != 0 or apps.stdout.strip():
        raise LaunchContractError("one or more H100s have active compute processes")
    for port in [
        manifest["syncer"]["port"],
        *(
            port
            for island in manifest["topology"]["islands"]
            for port in island["ports"].values()
        ),
    ]:
        with socket.socket() as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as exc:
                raise LaunchContractError(
                    f"reserved launch port is busy: {port}"
                ) from exc
    root = Path(manifest["launch"]["host_run_root"])
    if root.exists() or root.is_symlink():
        raise LaunchContractError("launch run root is not fresh")
    _directory(root.parent, "launch run root parent")
    topo = _run(["nvidia-smi", "topo", "-m"]).stdout
    return {"gpu_inventory": query, "gpu_topology": topo}


def _secrlenv_ready(manifest: dict[str, Any]) -> dict[str, Any]:
    contract = manifest["secrlenv"]
    try:
        token = Path(contract["bearer_token_host_path"]).read_text().strip()
    except (OSError, UnicodeError) as exc:
        raise LaunchContractError("SecRLEnv bearer token is unreadable") from exc
    if not token or any(character.isspace() for character in token):
        raise LaunchContractError("SecRLEnv bearer token is empty or malformed")
    request = urllib.request.Request(contract["host_health_url"])
    request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
    except Exception as exc:
        raise LaunchContractError("SecRLEnv daemon health preflight failed") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("ok") is not True
        or payload.get("active_episodes") != 0
        or payload.get("task_pack_sha256") != contract["task_pack_sha256"]
    ):
        raise LaunchContractError("SecRLEnv daemon is busy or has the wrong task pack")
    return payload


def _container_command(
    manifest: dict[str, Any], manifest_path: Path, manifest_sha: str, island_id: int
) -> list[str]:
    island = manifest["topology"]["islands"][island_id]
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        island["container_name"],
        "--label",
        "yeto.protected=true",
        "--label",
        f"yeto.m1.run={manifest['run_id']}",
        "--network",
        "host",
        "--gpus",
        '"device=' + ",".join(island["gpu_uuids"]) + '"',
        "--shm-size",
        "64g",
        "--ulimit",
        "memlock=-1:-1",
        "--env",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "PYTHONPATH=/root/miles:/root/yeto:/root/Megatron-LM",
        "--env",
        "HF_HOME=/root/.cache/huggingface",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "HF_HUB_DISABLE_TELEMETRY=1",
        "--env",
        "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "--env",
        "NCCL_NVLS_ENABLE=1",
        "--env",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "--env",
        "MILES_EXPERIMENTAL_FT_TRAINER=0",
        "--env",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1",
        "--env",
        f"YETO_M1_MANIFEST_SHA256={manifest_sha}",
        "--env",
        f"SECRLENV_DAEMON_URL={manifest['secrlenv']['container_url']}",
        "--env",
        f"SECRLENV_TASK_PACK_SHA256={manifest['secrlenv']['task_pack_sha256']}",
        "--env",
        "SECRLENV_BEARER_TOKEN_FILE=/run/secrlenv/daemon.token",
    ]
    if manifest["evaluation"]["enabled"]:
        command.extend(("--env", "MILES_CI_GATE_RECORD_DIR=/evidence/miles-metrics"))
    for name, value in sorted(manifest["harness"]["environment"].items()):
        command.extend(("--env", f"{name}={value}"))
    volumes = [
        (manifest["provenance"]["yeto_root"], "/root/yeto", "ro"),
        (manifest["provenance"]["miles_root"], "/root/miles", "ro"),
        (manifest["model"]["host_model_path"], "/models/hf", "ro"),
        (
            manifest["model"]["host_model_path"],
            manifest["model"]["container_hf_cache_snapshot_path"],
            "ro",
        ),
        (manifest["model"]["host_checkpoint_path"], "/models/torch_dist", "ro"),
        (manifest["data"]["host_train_path"], "/workspace/data/train.jsonl", "ro"),
        (
            manifest["data"]["host_heldout_path"],
            manifest["data"]["container_heldout_path"],
            "ro",
        ),
        (str(manifest_path), "/run/yeto-m1/manifest.json", "ro"),
        (island["host_run_dir"], "/evidence", "rw"),
        (
            manifest["secrlenv"]["bearer_token_host_path"],
            "/run/secrlenv/daemon.token",
            "ro",
        ),
    ]
    volumes.extend(
        (row["host_path"], row["container_path"], "ro")
        for row in manifest["harness"]["artifacts"]
    )
    for source, destination, mode in volumes:
        command.extend(("--volume", f"{source}:{destination}:{mode}"))
    command.extend(
        (
            manifest["image"]["reference"],
            "bash",
            "/root/yeto/tools/probes/run_m1_dense_full_island.sh",
            "/run/yeto-m1/manifest.json",
            manifest_sha,
            str(island_id),
        )
    )
    return command


def _cleanup_failed_launch(
    *,
    run_root: Path,
    syncer_process: subprocess.Popen[bytes] | None,
    launched: list[str],
    error: BaseException,
) -> None:
    container_results = []
    for container_id in reversed(launched):
        try:
            stopped = subprocess.run(
                ["docker", "stop", "--time", "30", container_id],
                text=True,
                capture_output=True,
                check=False,
                timeout=45,
            )
            returncode = stopped.returncode
            cleanup_error = None
        except (OSError, subprocess.SubprocessError) as exc:
            returncode = None
            cleanup_error = f"{type(exc).__name__}: {exc}"
        container_results.append(
            {
                "container_id": container_id,
                "stop_returncode": returncode,
                "stopped": returncode == 0,
                "cleanup_error": cleanup_error,
            }
        )
    syncer_signal = None
    syncer_cleanup_error = None
    if syncer_process is not None and syncer_process.poll() is None:
        syncer_signal = "SIGTERM"
        try:
            os.killpg(syncer_process.pid, signal.SIGTERM)
            syncer_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            syncer_signal = "SIGKILL"
            try:
                os.killpg(syncer_process.pid, signal.SIGKILL)
                syncer_process.wait(timeout=10)
            except (OSError, subprocess.SubprocessError) as exc:
                syncer_cleanup_error = f"{type(exc).__name__}: {exc}"
        except (OSError, subprocess.SubprocessError) as exc:
            syncer_cleanup_error = f"{type(exc).__name__}: {exc}"
    evidence = {
        "schema": "yeto-m1-dense-full-launch-failure-v1",
        "error_type": type(error).__name__,
        "error": str(error),
        "created_containers": launched,
        "container_cleanup": container_results,
        "syncer_pid": syncer_process.pid if syncer_process is not None else None,
        "syncer_returncode": (
            syncer_process.poll() if syncer_process is not None else None
        ),
        "syncer_signal": syncer_signal,
        "syncer_cleanup_error": syncer_cleanup_error,
    }
    failure_path = run_root / "launch-failure.json"
    try:
        failure_path.write_bytes(_canonical(evidence))
        os.chmod(failure_path, 0o600)
    except OSError:
        # Preserve the original launch exception even if the filesystem fails.
        pass


def _launch(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest, args.expected_sha256)
    _verify_live_inputs(manifest)
    host = _assert_host_ready(manifest)
    daemon_health = _secrlenv_ready(manifest)
    run_root = Path(manifest["launch"]["host_run_root"])
    os.mkdir(run_root, 0o700)
    for name in (
        "syncer",
        *(f"island-{index}" for index in range(manifest["topology"]["island_count"])),
    ):
        os.mkdir(run_root / name, 0o700)
    attestation = {
        "schema": "yeto-m1-dense-full-host-preflight-v1",
        "manifest_sha256": args.expected_sha256,
        "secrlenv_health": daemon_health,
        **host,
    }
    (run_root / "host-preflight.json").write_bytes(_canonical(attestation))
    os.chmod(run_root / "host-preflight.json", 0o600)
    shutil.copyfile(args.manifest, run_root / "manifest.json")
    os.chmod(run_root / "manifest.json", 0o600)
    if _sha256(run_root / "manifest.json") != args.expected_sha256:
        raise LaunchContractError("preserved launch manifest copy differs")

    syncer_log = (run_root / "syncer" / "syncer.log").open("xb")
    syncer_process: subprocess.Popen[bytes] | None = None
    try:
        syncer_process = subprocess.Popen(
            manifest["syncer"]["argv"],
            stdin=subprocess.DEVNULL,
            stdout=syncer_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException as exc:
        try:
            syncer_log.close()
        except OSError:
            pass
        _cleanup_failed_launch(
            run_root=run_root,
            syncer_process=None,
            launched=[],
            error=exc,
        )
        raise
    assert syncer_process is not None
    launched: list[str] = []
    try:
        syncer_log.close()
        (run_root / "syncer" / "pid").write_text(f"{syncer_process.pid}\n")
        os.chmod(run_root / "syncer" / "pid", 0o600)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if syncer_process.poll() is not None:
                raise LaunchContractError("syncer exited during launch")
            try:
                with socket.create_connection(
                    ("127.0.0.1", manifest["syncer"]["port"]), 1
                ):
                    break
            except OSError:
                time.sleep(0.25)
        else:
            raise LaunchContractError("syncer did not listen within 30 seconds")

        for island_id in range(manifest["topology"]["island_count"]):
            command = _container_command(
                manifest, run_root / "manifest.json", args.expected_sha256, island_id
            )
            command_path = run_root / f"island-{island_id}" / "docker-argv.json"
            command_path.write_bytes(_canonical(command))
            os.chmod(command_path, 0o600)
            result = _run(command)
            container_id = result.stdout.strip()
            if not re.fullmatch(r"[0-9a-f]{64}", container_id):
                raise LaunchContractError(
                    f"island {island_id} returned no container ID"
                )
            launched.append(container_id)
            id_path = run_root / f"island-{island_id}" / "container-id"
            id_path.write_text(container_id + "\n")
            os.chmod(id_path, 0o600)
            marker_path = run_root / f"island-{island_id}" / "container-started.json"
            marker_deadline = time.monotonic() + 90
            while time.monotonic() < marker_deadline:
                if marker_path.is_file() and not marker_path.is_symlink():
                    marker = _load_json(marker_path, canonical=True, private=True)
                    if marker != {
                        "schema": "yeto-m1-dense-full-island-started-v1",
                        "island_id": island_id,
                        "manifest_sha256": args.expected_sha256,
                    }:
                        raise LaunchContractError(
                            f"island {island_id} startup marker differs"
                        )
                    break
                state = _run(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        "{{.State.Running}} {{.State.ExitCode}}",
                        container_id,
                    ]
                ).stdout.strip()
                if not state.startswith("true "):
                    raise LaunchContractError(
                        f"island {island_id} exited before its startup marker: {state}"
                    )
                time.sleep(0.5)
            else:
                raise LaunchContractError(
                    f"island {island_id} did not start Ray within 90 seconds"
                )
    except BaseException as exc:
        _cleanup_failed_launch(
            run_root=run_root,
            syncer_process=syncer_process,
            launched=launched,
            error=exc,
        )
        raise
    print(
        json.dumps(
            {
                "containers": launched,
                "manifest_sha256": args.expected_sha256,
                "run_root": str(run_root),
                "syncer_pid": syncer_process.pid,
            },
            sort_keys=True,
        )
    )


def _verify_bridge(identity: dict[str, Any]) -> None:
    distribution = importlib.metadata.distribution(identity["distribution_name"])
    if distribution.version != identity["distribution_version"]:
        raise LaunchContractError("installed Megatron Bridge version differs")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        raise LaunchContractError(
            "installed Megatron Bridge has no direct URL identity"
        )
    payload = json.loads(direct_url, object_pairs_hook=_unique_object)
    vcs = payload.get("vcs_info") if isinstance(payload, dict) else None
    if (
        payload != identity["direct_url"]
        or not isinstance(vcs, dict)
        or vcs.get("commit_id") != identity["direct_url_commit"]
    ):
        raise LaunchContractError("installed Megatron Bridge commit differs")


def _container_gpu_uuids() -> list[str]:
    rows = _run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    ).stdout.splitlines()
    parsed = [[value.strip() for value in row.split(",")] for row in rows]
    if [int(row[0]) for row in parsed] != list(range(4)):
        raise LaunchContractError("container does not see exactly four indexed GPUs")
    return [row[1] for row in parsed]


def _install_ray_placement_guard(manifest: dict[str, Any], island_id: int) -> None:
    from miles.ray import placement_group as placement_module

    original = placement_module.create_placement_groups
    evidence_path = Path("/evidence/ray-placement.json")

    def checked_create_placement_groups(miles_args):
        groups = original(miles_args)
        actor = groups.get("actor") if isinstance(groups, dict) else None
        rollout = groups.get("rollout") if isinstance(groups, dict) else None
        if (
            not isinstance(actor, tuple)
            or len(actor) != 3
            or not isinstance(rollout, tuple)
            or len(rollout) != 3
            or actor[0] is not rollout[0]
        ):
            raise LaunchContractError("Miles returned an invalid placement group")
        try:
            actor_bundles = [int(value) for value in actor[1]]
            actor_gpus = [int(value) for value in actor[2]]
            rollout_bundles = [int(value) for value in rollout[1]]
            rollout_gpus = [int(value) for value in rollout[2]]
        except (TypeError, ValueError) as exc:
            raise LaunchContractError("Miles placement IDs are malformed") from exc
        if (
            actor_gpus != [0, 1, 2, 3]
            or len(actor_bundles) != 4
            or len(set(actor_bundles)) != 4
            or rollout_gpus != [2, 3]
            or rollout_bundles != actor_bundles[2:]
            or set(actor_bundles[:2]) & set(rollout_bundles)
        ):
            raise LaunchContractError(
                "Miles did not isolate trainer GPUs 0-1 from inference GPUs 2-3"
            )
        island = manifest["topology"]["islands"][island_id]
        _write_exclusive(
            evidence_path,
            {
                "schema": "yeto-m1-dense-full-ray-placement-v1",
                "island_id": island_id,
                "manifest_sha256": _expect_sha(
                    os.environ.get("YETO_M1_MANIFEST_SHA256"),
                    "container manifest environment",
                ),
                "actor_bundle_indices": actor_bundles[:2],
                "actor_local_gpu_ids": actor_gpus[:2],
                "actor_gpu_uuids": island["trainer_gpu_uuids"],
                "inference_bundle_indices": rollout_bundles,
                "inference_local_gpu_ids": rollout_gpus,
                "inference_gpu_uuids": island["inference_gpu_uuids"],
                "inference_engine_count": INFERENCE_ENGINES,
                "inference_tp": INFERENCE_TP,
            },
        )
        return groups

    placement_module.create_placement_groups = checked_create_placement_groups


def _exec_island(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest, args.expected_sha256)
    if args.island_id not in range(manifest["topology"]["island_count"]):
        raise LaunchContractError("invalid island id")
    island = manifest["topology"]["islands"][args.island_id]
    if _container_gpu_uuids() != island["gpu_uuids"]:
        raise LaunchContractError("container GPU UUID order differs from its island")
    _attest_launch_bundle(manifest["launch_bundle"], Path("/root/yeto"))
    _verify_bridge(manifest["provenance"]["megatron_bridge"])
    yeto_sha, miles_sha = _source_hashes(Path("/root/yeto"), Path("/root/miles"))
    if (
        yeto_sha != manifest["provenance"]["yeto_source_sha256"]
        or miles_sha != manifest["provenance"]["miles_source_sha256"]
    ):
        raise LaunchContractError("container execution source differs")
    _attest_file(
        Path("/models/hf/config.json"),
        manifest["model"]["config_sha256"],
        "model config",
    )
    conversion = _validate_conversion_artifacts(
        Path("/models/torch_dist/conversion-manifest.json"),
        expected_sha256=manifest["model"]["conversion_manifest_sha256"],
        image_digest=manifest["image"]["digest"],
        model_root=Path("/models/hf"),
        checkpoint_root=Path("/models/torch_dist"),
    )
    if conversion != manifest["model"]["conversion_manifest"]:
        raise LaunchContractError("container conversion artifacts differ")
    _attest_file(
        Path("/workspace/data/train.jsonl"),
        manifest["data"]["train_sha256"],
        "train data",
    )
    if manifest["evaluation"]["enabled"]:
        _attest_file(
            Path(manifest["data"]["container_heldout_path"]),
            manifest["data"]["heldout_sha256"],
            "heldout data",
        )
    argv = manifest["learners"][args.island_id]["argv"]
    if argv[:3] != ["python3", "-m", "yeto.rl.learner"]:
        raise LaunchContractError("manifest learner entry is not direct")
    from yeto.rl import learner

    _install_ray_placement_guard(manifest, args.island_id)

    original_parse_args = learner.parse_args
    learner_argv = argv[3:]
    ports = island["ports"]

    def parse_with_local_ports(values=None):
        parsed = original_parse_args(learner_argv if values is None else values)
        parsed.rollout_engine_base_port = ports["rollout_engine_base"]
        parsed.sglang_router_port = ports["sglang_router"]
        parsed.sglang_router_prometheus_port = ports["sglang_router_prometheus"]
        parsed.train_master_base_port = ports["train_master_base"]
        return parsed

    learner.parse_args = parse_with_local_ports
    learner.main(learner_argv)


def _island_runtime(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest, args.expected_sha256)
    _attest_launch_bundle(manifest["launch_bundle"], Path("/root/yeto"))
    if args.island_id not in range(manifest["topology"]["island_count"]):
        raise LaunchContractError("invalid island id")
    ports = manifest["topology"]["islands"][args.island_id]["ports"]
    # Intentionally machine-only: the inner shell reads these four fields
    # without eval or interpolation.
    print(
        ports["ray_gcs"],
        ports["ray_dashboard"],
        ports["ray_client"],
        manifest["topology"]["islands"][args.island_id]["container_run_dir"],
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=_build)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--live", action="store_true")
    verify.set_defaults(function=_verify)
    report = subparsers.add_parser("report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--expected-sha256", required=True)
    report.set_defaults(function=_report)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--manifest", type=Path, required=True)
    launch.add_argument("--expected-sha256", required=True)
    launch.set_defaults(function=_launch)
    execute = subparsers.add_parser("exec-island")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--expected-sha256", required=True)
    execute.add_argument("--island-id", type=int, required=True)
    execute.set_defaults(function=_exec_island)
    runtime = subparsers.add_parser("island-runtime")
    runtime.add_argument("--manifest", type=Path, required=True)
    runtime.add_argument("--expected-sha256", required=True)
    runtime.add_argument("--island-id", type=int, required=True)
    runtime.set_defaults(function=_island_runtime)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
