#!/usr/bin/env python3
"""Run and seal the frozen one-A100 active-CPLG acquisition.

The program deliberately ends at the GPU acquisition boundary.  It verifies
the frozen launch and input identities, runs the stock/candidate pair through
``compare_diloco.py``, checks only producer and acquisition completeness, and
publishes a checksummed acquisition manifest and terminal receipt.  Scientific
replay, gate arithmetic, resampling, and verdict publication belong to the
separate post-teardown CPU phase and are not implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE = REPO_ROOT / "scripts" / "compare_diloco.py"
STOCK_ARM = "cplg_m1_stock"
CANDIDATE_ARM = "cplg_m1_candidate"
BASE_ARM = "base (untrained)"
MODEL_PATH = "/home/shou/models/Qwen3.5-9B"
DATA_PATH = "/home/shou/data/Capybara-local/train.parquet"
CONFIG_SHA256 = "5afe2d4900051fda1ac99cc682c489dfeae85f0eb34d1816646b5bff5f0c26df"
FRAGMENT_ORDER = [0, 1, 2, 3] * 8
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
RUN_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
CHECKSUM_LINE_RE = re.compile(r"([0-9a-f]{64})  (.+)\Z")
MANIFEST_SCHEMA = "yeto.cplg-sgd.online-e1-acquisition-manifest.v1"
TERMINAL_SCHEMA = "yeto.cplg-sgd.online-e1-acquisition-terminal.v1"

COMPARE_FLAGS = frozenset(
    {
        "--strict-quorum",
        "--barrier-sync",
        "--deterministic-commit-order",
        "--skip-baseline",
    }
)
COMPARE_OPTION_ORDER = (
    "--model",
    "--data",
    "--seq-len",
    "--micro-batch-size",
    "--inner-lr",
    "--tuning",
    "--shard",
    "--lora-r",
    "--lora-alpha",
    "--eval-rows",
    "--max-rows",
    "--shuffle-rows-seed",
    "--training-seed",
    "--device",
    "--gpu-slots",
    "--delta-correction",
    "--matrix-merge",
    "--outer-momentum",
    "--outer-lr",
    "--token-budget",
    "--syncer-total-steps",
    "--learner-max-steps",
    "--fixed-window-microsteps",
    "--delta-norm-ref",
    "--syncer-checkpoint-every",
    "--strict-quorum",
    "--barrier-sync",
    "--deterministic-commit-order",
    "--settings",
    "--skip-baseline",
    "--cplg-online-run-id",
    "--cplg-online-run-config-sha256",
    "--cplg-online-source-commit",
    "--arm-timeout-min",
    "--work-dir",
    "--report-dir",
)


class RunnerError(RuntimeError):
    """The frozen acquisition contract was not satisfied."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    _require_regular(path, "SHA-256 input")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish(path: Path, payload: bytes) -> None:
    """Publish one fresh file without replacing pre-existing evidence."""

    path = _absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(path):
        raise RunnerError(f"refusing to replace stale evidence: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RunnerError(f"refusing to replace stale evidence: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json(path: Path, value: dict[str, Any]) -> str:
    path = _absolute(path)
    sidecar = path.with_name(path.name + ".sha256")
    if _path_exists(path) or _path_exists(sidecar):
        raise RunnerError(f"checksummed output is not fresh: {path}")
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    digest = _sha256_bytes(raw)
    _atomic_publish(path, raw)
    _atomic_publish(sidecar, f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def _require_regular(path: Path, label: str, *, nonempty: bool = True) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RunnerError(f"missing {label}: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RunnerError(f"{label} is not a regular non-symlink file: {path}")
    if nonempty and metadata.st_size == 0:
        raise RunnerError(f"{label} is empty: {path}")


def _regular_bytes(path: Path, label: str, *, nonempty: bool = True) -> bytes:
    _require_regular(path, label, nonempty=nonempty)
    return path.read_bytes()


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def object_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise RunnerError(f"{label}: duplicate JSON field {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{label}: invalid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise RunnerError(f"{label}: expected a JSON object")
    return value


def _checksummed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw = _regular_bytes(path, label)
    digest = _sha256_bytes(raw)
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if _regular_bytes(sidecar, f"{label} checksum") != expected:
        raise RunnerError(f"{label}: checksum sidecar is not basename-bound or valid")
    return _strict_json(raw, label), digest


def _closed_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise RunnerError(
            f"{label}: closed fields differ; missing={sorted(fields - value.keys())}, "
            f"unexpected={sorted(value.keys() - fields)}"
        )


def _require_exact_fields(
    value: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    for field, wanted in expected.items():
        if field not in value:
            raise RunnerError(f"{label} is missing {field!r}")
        observed = value[field]
        if type(observed) is not type(wanted) or observed != wanted:
            raise RunnerError(
                f"{label} {field} differs from the frozen contract: "
                f"expected {wanted!r}, observed {observed!r}"
            )


def _cli_number(value: Any, label: str) -> str:
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return format(value, ".15g")
    raise RunnerError(f"{label} must be a finite exact JSON number")


def _parse_compare_options(argv: list[str]) -> tuple[dict[str, str], set[str]]:
    if not argv:
        raise RunnerError("missing compare argv after --")
    values: dict[str, str] = {}
    flags: set[str] = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in COMPARE_FLAGS:
            if token in flags:
                raise RunnerError(f"compare argv declares {token} more than once")
            flags.add(token)
            index += 1
            continue
        if not token.startswith("--") or "=" in token:
            raise RunnerError(f"noncanonical or positional compare argument: {token!r}")
        if token in values:
            raise RunnerError(f"compare argv declares {token} more than once")
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise RunnerError(f"compare argv has no explicit value after {token}")
        values[token] = argv[index + 1]
        index += 2
    return values, flags


def _expected_compare_values(
    config: dict[str, Any],
    *,
    run_id: str,
    run_config_sha256: str,
    source_commit: str,
) -> dict[str, str]:
    workload = config["workload"]
    return {
        "--settings": ",".join(workload["arms_in_order"]),
        "--model": MODEL_PATH,
        "--data": DATA_PATH,
        "--tuning": workload["tuning"],
        "--shard": workload["shard"],
        "--seq-len": str(workload["sequence_length"]),
        "--micro-batch-size": str(workload["micro_batch_size"]),
        "--inner-lr": _cli_number(
            workload["inner_learning_rate"], "inner_learning_rate"
        ),
        "--lora-r": str(workload["lora_rank"]),
        "--lora-alpha": str(workload["lora_alpha"]),
        "--eval-rows": str(workload["evaluation_rows"]),
        "--max-rows": str(workload["max_rows"]),
        "--shuffle-rows-seed": str(workload["shuffle_rows_seed"]),
        "--training-seed": str(workload["training_seed"]),
        "--device": workload["device"],
        "--gpu-slots": str(workload["gpu_slots"]),
        "--delta-correction": workload["delta_correction"],
        "--matrix-merge": workload["matrix_merge"],
        "--outer-momentum": _cli_number(workload["outer_momentum"], "outer_momentum"),
        "--outer-lr": _cli_number(
            workload["outer_learning_rate"], "outer_learning_rate"
        ),
        "--token-budget": str(workload["compare_token_budget_per_arm"]),
        "--syncer-total-steps": str(workload["outer_commits_per_arm"]),
        "--learner-max-steps": str(workload["learner_max_steps_liveness_cap"]),
        "--fixed-window-microsteps": str(workload["fixed_window_microsteps"]),
        "--arm-timeout-min": str(workload["arm_timeout_minutes"]),
        "--delta-norm-ref": "0",
        "--syncer-checkpoint-every": "1",
        "--cplg-online-run-id": run_id,
        "--cplg-online-run-config-sha256": run_config_sha256,
        "--cplg-online-source-commit": source_commit,
    }


def _build_compare_argv(
    config: dict[str, Any],
    *,
    run_id: str,
    run_config_sha256: str,
    source_commit: str,
    work_dir: Path,
    report_dir: Path,
) -> list[str]:
    values = _expected_compare_values(
        config,
        run_id=run_id,
        run_config_sha256=run_config_sha256,
        source_commit=source_commit,
    )
    values["--work-dir"] = str(_absolute(work_dir))
    values["--report-dir"] = str(_absolute(report_dir))
    argv: list[str] = []
    for name in COMPARE_OPTION_ORDER:
        argv.append(name)
        if name not in COMPARE_FLAGS:
            argv.append(values[name])
    return argv


def _validate_frozen_config(
    config: dict[str, Any],
    compare_argv: list[str],
    *,
    run_id: str,
    run_config_sha256: str,
    source_commit: str,
) -> tuple[Path, Path]:
    try:
        workload = config["workload"]
        resource = config["resource_envelope"]["gpu_acquisition"]
    except (KeyError, TypeError) as exc:
        raise RunnerError("frozen configuration is missing required sections") from exc
    if type(workload) is not dict or type(resource) is not dict:
        raise RunnerError("frozen workload and GPU resource sections must be objects")
    _require_exact_fields(
        config,
        {
            "schema_version": 1,
            "schema": "cplg_sgd_active_e1_scientific_config_v1",
            "run_id": "exp2-cplg-active-e1-m1-r1",
        },
        "frozen configuration",
    )
    if run_id != config["run_id"]:
        raise RunnerError(
            f"runner run ID {run_id!r} differs from frozen {config['run_id']!r}"
        )
    _require_exact_fields(
        resource,
        {
            "provider": "gcp",
            "project": "model-training-497007",
            "zone": "us-central1-c",
            "machine_type": "a2-highgpu-1g",
            "accelerator_count": 1,
            "maximum_total_accelerators": 1,
            "provisioning_model": "SPOT",
            "termination_action": "DELETE",
            "maximum_run_duration_seconds": 3600,
            "boot_disk_size_gb": 250,
            "boot_disk_type": "pd-ssd",
            "image": (
                "projects/model-training-497007/global/images/"
                "yeto-optimizer-a100-20260714"
            ),
            "expected_source_image_id": "7290368630472593484",
        },
        "frozen GPU resource",
    )
    _require_exact_fields(
        workload,
        {
            "arms_in_order": [STOCK_ARM, CANDIDATE_ARM],
            "outer_optimizer_by_arm": {
                STOCK_ARM: "nesterov",
                CANDIDATE_ARM: "cplg-sgd",
            },
            "result_rows_in_order": [BASE_ARM, STOCK_ARM, CANDIDATE_ARM],
            "execution_order": "sequential stock then candidate",
            "sequence_length": 128,
            "micro_batch_size": 1,
            "raw_local_training_tokens_per_arm": 4352,
            "compare_token_budget_per_arm": 4352,
            "expected_terminal_local_steps_per_arm": 34,
            "learner_max_steps_liveness_cap": 96,
            "learners": 1,
            "fragments": 4,
            "quorum": 1,
            "fixed_window_microsteps": 4,
            "outer_commits_per_arm": 32,
            "commits_per_fragment_per_arm": 8,
            "fragment_order": FRAGMENT_ORDER,
            "wire_dtype": "f32",
            "merge_alpha": 0.0,
            "matrix_merge": "rda",
            "delta_correction": "none",
            "outer_learning_rate": 0.28,
            "outer_learning_rate_f32_bits": "0x3e8f5c29",
            "outer_momentum": 0.0,
            "outer_momentum_f32_bits": "0x00000000",
            "inner_optimizer": "adamw",
            "inner_learning_rate": 0.001,
            "adamw_betas": [0.9, 0.999],
            "adamw_epsilon": 1e-08,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "scheduler": "linear warmup for 10 optimizer steps, then constant",
            "gradient_clip_norm": 1.0,
            "gradient_checkpointing": "off",
            "loss_function": "cross_entropy",
            "train_on": "assistant",
            "tuning": "lora",
            "lora_rank": 2,
            "lora_alpha": 4,
            "lora_targets": "all-linear",
            "shard": "ddp",
            "grad_accumulation": 1,
            "tokenization": "preload",
            "shuffle_rows_seed": 271,
            "training_seed": 271271,
            "max_rows": 5000,
            "evaluation_rows": 8,
            "device": "cuda",
            "gpu_slots": 1,
            "arm_timeout_minutes": 20,
            "strict_quorum": True,
            "barrier_sync": True,
            "deterministic_commit_order": True,
            "skip_baseline_training": True,
        },
        "frozen workload",
    )
    values, flags = _parse_compare_options(compare_argv)
    expected = _expected_compare_values(
        config,
        run_id=run_id,
        run_config_sha256=run_config_sha256,
        source_commit=source_commit,
    )
    expected_names = set(expected) | {"--work-dir", "--report-dir"}
    if set(values) != expected_names:
        raise RunnerError(
            "compare value options differ from the exact frozen allowlist; "
            f"missing={sorted(expected_names - values.keys())}, "
            f"unexpected={sorted(values.keys() - expected_names)}"
        )
    if flags != COMPARE_FLAGS:
        raise RunnerError(
            "compare flags differ from the exact frozen allowlist; "
            f"missing={sorted(COMPARE_FLAGS - flags)}, "
            f"unexpected={sorted(flags - COMPARE_FLAGS)}"
        )
    for name, wanted in expected.items():
        if values[name] != wanted:
            raise RunnerError(
                f"compare {name} differs from frozen configuration: "
                f"expected {wanted!r}, observed {values[name]!r}"
            )
    work_dir = _absolute(Path(values["--work-dir"]))
    report_dir = _absolute(Path(values["--report-dir"]))
    if (
        work_dir == report_dir
        or work_dir in report_dir.parents
        or report_dir in work_dir.parents
    ):
        raise RunnerError("work-dir and report-dir must be disjoint siblings")
    if work_dir.parent != report_dir.parent:
        raise RunnerError("work-dir and report-dir must share one acquisition root")
    return work_dir, report_dir


def _check_compare_contract(
    argv: list[str],
    *,
    run_id: str,
    run_config_sha256: str,
    source_commit: str,
    config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Compatibility entry point used by tests and launch-spec review."""

    if config is None:
        config, digest = _checksummed_json(
            REPO_ROOT / "experiments/optimizer/cplg-sgd-active-e1-r1-config.json",
            "frozen run configuration",
        )
        if digest != CONFIG_SHA256:
            raise RunnerError("repository frozen configuration digest differs")
    return _validate_frozen_config(
        config,
        argv,
        run_id=run_id,
        run_config_sha256=run_config_sha256,
        source_commit=source_commit,
    )


def _parse_checksum_manifest(path: Path, label: str) -> list[tuple[str, Path]]:
    raw = _regular_bytes(path, label)
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RunnerError(f"{label}: not UTF-8") from exc
    entries: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for line_number, line in enumerate(lines, 1):
        match = CHECKSUM_LINE_RE.fullmatch(line)
        if match is None:
            raise RunnerError(f"{label}:{line_number}: noncanonical checksum line")
        target = _absolute(Path(match.group(2)))
        if target in seen:
            raise RunnerError(f"{label}: duplicate checksum target {target}")
        seen.add(target)
        entries.append((match.group(1), target))
    if not entries:
        raise RunnerError(f"{label}: empty checksum manifest")
    return entries


def _verify_checksum_entries(
    manifest: Path,
    label: str,
    *,
    expected_root: Path,
    exact_target: Path | None = None,
) -> str:
    expected_root = _absolute(expected_root)
    entries = _parse_checksum_manifest(manifest, label)
    if exact_target is not None:
        exact_target = _absolute(exact_target)
        if [target for _digest, target in entries] != [exact_target]:
            raise RunnerError(f"{label}: does not bind exactly {exact_target}")
    for wanted, target in entries:
        if target != expected_root and expected_root not in target.parents:
            raise RunnerError(f"{label}: target escapes frozen input root: {target}")
        relative = (
            target.relative_to(expected_root) if target != expected_root else None
        )
        cursor = expected_root
        if relative is not None:
            for part in relative.parts[:-1]:
                cursor /= part
                metadata = cursor.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise RunnerError(f"{label}: unsafe directory component {cursor}")
        if _sha256_file(target) != wanted:
            raise RunnerError(f"{label}: SHA-256 mismatch for {target}")
    return _sha256_file(manifest)


def _verify_input_provenance(
    input_manifest: Path,
    *,
    root: Path,
    config: dict[str, Any],
    source_commit: str,
) -> dict[str, str]:
    root = _absolute(root)
    input_manifest = _absolute(input_manifest)
    if input_manifest != root / "input-provenance.sha256":
        raise RunnerError(
            "input manifest must be acquisition-root/input-provenance.sha256"
        )
    entries = _parse_checksum_manifest(input_manifest, "input provenance manifest")
    provenance_dir = root / "input-provenance"
    by_name: dict[str, tuple[str, Path]] = {}
    for digest, target in entries:
        if target.parent != provenance_dir:
            raise RunnerError(
                f"input provenance manifest target is outside {provenance_dir}: {target}"
            )
        if target.name in by_name:
            raise RunnerError(f"duplicate input provenance basename {target.name!r}")
        if _sha256_file(target) != digest:
            raise RunnerError(f"input provenance SHA-256 mismatch: {target}")
        by_name[target.name] = (digest, target)
    required = {
        "verification.log",
        "yeto-model-files.sha256",
        "yeto-data.sha256",
        "yeto-runtime.txt",
        "yeto-optimizer-image.json",
    }
    if set(by_name) != required:
        raise RunnerError(
            "input provenance inventory differs; "
            f"missing={sorted(required - by_name.keys())}, "
            f"unexpected={sorted(by_name.keys() - required)}"
        )
    model_digest = _verify_checksum_entries(
        by_name["yeto-model-files.sha256"][1],
        "model checksum manifest",
        expected_root=Path(MODEL_PATH),
    )
    data_digest = _verify_checksum_entries(
        by_name["yeto-data.sha256"][1],
        "data checksum manifest",
        expected_root=Path(DATA_PATH),
        exact_target=Path(DATA_PATH),
    )
    runtime_path = by_name["yeto-runtime.txt"][1]
    runtime = _regular_bytes(runtime_path, "runtime manifest").decode("utf-8")
    for marker in ("torch=", "transformers=", "cuda=", "rustc ", "cargo "):
        if marker not in runtime:
            raise RunnerError(f"runtime manifest is missing {marker!r}")
    image_path = by_name["yeto-optimizer-image.json"][1]
    image = _strict_json(_regular_bytes(image_path, "image manifest"), "image manifest")
    _require_exact_fields(
        image,
        {
            "schema_version": 1,
            "repo_url": "https://github.com/agentenv/yeto.git",
            "model_files_included": True,
            "huggingface_cache_included": False,
            "credentials_included": False,
            "run_artifacts_included": False,
            "model_checksum_manifest": "/etc/yeto-model-files.sha256",
            "data_checksum_manifest": "/etc/yeto-data.sha256",
            "runtime_manifest": "/etc/yeto-runtime.txt",
        },
        "image manifest",
    )
    spec_path = root / "spec.json"
    spec = _strict_json(_regular_bytes(spec_path, "immutable cloud spec"), "cloud spec")
    resource = config["resource_envelope"]["gpu_acquisition"]
    _require_exact_fields(
        spec,
        {"run_id": config["run_id"], "repo_commit": source_commit},
        "cloud spec",
    )
    cloud = spec.get("cloud")
    if type(cloud) is not dict:
        raise RunnerError("cloud spec cloud section must be an object")
    _require_exact_fields(
        cloud,
        {
            "image": resource["image"],
            "expected_source_image_id": resource["expected_source_image_id"],
            "machine_type": resource["machine_type"],
            "accelerator_count": resource["accelerator_count"],
        },
        "cloud spec image/runtime",
    )
    bindings = spec.get("scientific_bindings")
    if (
        type(bindings) is not dict
        or bindings.get("scientific_config_sha256") != CONFIG_SHA256
    ):
        raise RunnerError(
            "cloud spec does not bind the frozen scientific config digest"
        )
    return {
        "input_manifest_sha256": _sha256_file(input_manifest),
        "model_manifest_sha256": model_digest,
        "data_manifest_sha256": data_digest,
        "runtime_manifest_sha256": _sha256_file(runtime_path),
        "image_manifest_sha256": _sha256_file(image_path),
        "image": resource["image"],
        "source_image_id": resource["expected_source_image_id"],
    }


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_clean_checkout(expected: str) -> None:
    actual = _git_head()
    if COMMIT_RE.fullmatch(actual) is None:
        raise RunnerError("source checkout has no full lowercase commit identity")
    if actual != expected:
        raise RunnerError(f"source HEAD {actual!r} differs from frozen {expected!r}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RunnerError("source checkout is dirty before or after acquisition")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    raw = _regular_bytes(path, label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RunnerError(f"{label}: not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise RunnerError(f"{label}:{line_number}: blank JSONL row")
        rows.append(_strict_json(line.encode("utf-8"), f"{label}:{line_number}"))
    if not rows:
        raise RunnerError(f"{label}: empty JSONL")
    return rows


def _verify_initial_receipt(
    path: Path,
    *,
    run_id: str,
    config_sha256: str,
    source_commit: str,
    arm: str,
    optimizer: str,
) -> dict[str, Any]:
    value, _digest = _checksummed_json(path, f"{arm} initial-state receipt")
    _closed_fields(
        value,
        {
            "schema_version",
            "run_id",
            "run_config_sha256",
            "source_commit",
            "arm",
            "outer_optimizer",
            "layout_sha256",
            "initial_state_sha256",
            "fragments",
            "expected_commits",
        },
        str(path),
    )
    _require_exact_fields(
        value,
        {
            "schema_version": 1,
            "run_id": run_id,
            "run_config_sha256": config_sha256,
            "source_commit": source_commit,
            "arm": arm,
            "outer_optimizer": optimizer,
            "fragments": 4,
            "expected_commits": 32,
        },
        str(path),
    )
    for field in ("layout_sha256", "initial_state_sha256"):
        if type(value[field]) is not str or SHA256_RE.fullmatch(value[field]) is None:
            raise RunnerError(f"{path}: noncanonical {field}")
    return value


def _verify_completion_receipt(
    path: Path,
    *,
    arm: str,
    run_id: str,
    event_tape: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    value, _digest = _checksummed_json(path, f"{arm} completion receipt")
    _closed_fields(
        value,
        {
            "schema_version",
            "run_id",
            "arm",
            "terminal_local_steps",
            "raw_training_tokens",
            "final_global_step",
            "commits_observed",
            "commits_per_fragment",
            "interval_start_ns",
            "interval_end_ns",
            "interval_ns",
            "event_tape_sha256",
            "final_checkpoint_sha256",
            "ledger_head",
            "ledger_rows",
            "writer_dropped",
            "writer_abandoned",
            "writer_pending",
            "writer_errors",
        },
        str(path),
    )
    _require_exact_fields(
        value,
        {
            "schema_version": 1,
            "run_id": run_id,
            "arm": arm,
            "terminal_local_steps": 34,
            "raw_training_tokens": 4352,
            "commits_observed": 32,
            "commits_per_fragment": [8, 8, 8, 8],
            "writer_dropped": 0,
            "writer_abandoned": 0,
            "writer_pending": 0,
            "writer_errors": 0,
        },
        str(path),
    )
    for field in (
        "final_global_step",
        "interval_start_ns",
        "interval_end_ns",
        "interval_ns",
    ):
        if type(value[field]) is not int or value[field] < 0:
            raise RunnerError(f"{path}: {field} must be a nonnegative exact integer")
    if value["interval_end_ns"] < value["interval_start_ns"]:
        raise RunnerError(f"{path}: monotonic interval ends before it starts")
    if value["interval_ns"] != value["interval_end_ns"] - value["interval_start_ns"]:
        raise RunnerError(f"{path}: interval_ns is not the exact endpoint difference")
    if value["event_tape_sha256"] != _sha256_file(event_tape):
        raise RunnerError(f"{path}: event tape SHA-256 mismatch")
    if value["final_checkpoint_sha256"] != _sha256_file(checkpoint):
        raise RunnerError(f"{path}: checkpoint SHA-256 mismatch")
    if arm == STOCK_ARM:
        if value["ledger_head"] is not None or value["ledger_rows"] is not None:
            raise RunnerError(f"{path}: stock receipt must have null ledger fields")
    elif (
        type(value["ledger_head"]) is not str
        or SHA256_RE.fullmatch(value["ledger_head"]) is None
        or value["ledger_rows"] != 32
    ):
        raise RunnerError(f"{path}: candidate ledger head/count is invalid")
    return value


def _verify_learner_receipt(path: Path, arm: str) -> dict[str, Any]:
    value, _digest = _checksummed_json(path, f"{arm} learner receipt")
    _closed_fields(
        value,
        {
            "schema",
            "learner_id",
            "local_step",
            "raw_tokens",
            "global_step",
            "reconnect_count",
            "terminal_status",
        },
        str(path),
    )
    _require_exact_fields(
        value,
        {
            "schema": "yeto.learner_completion.v1",
            "learner_id": 0,
            "local_step": 34,
            "raw_tokens": 4352,
            "reconnect_count": 0,
            "terminal_status": "syncer_shutdown",
        },
        str(path),
    )
    if type(value["global_step"]) is not int or value["global_step"] < 0:
        raise RunnerError(f"{path}: global_step must be a nonnegative exact integer")
    return value


def _verify_event_schedule(path: Path, arm: str) -> None:
    rows = _read_jsonl(path, f"{arm} event tape")
    if len(rows) != 32:
        raise RunnerError(f"{arm} event tape has {len(rows)} rows instead of 32")
    fragments: list[int] = []
    for commit, row in enumerate(rows, 1):
        if row.get("commit_seq") != commit:
            raise RunnerError(f"{arm} event tape commit sequence is not contiguous")
        fragment = row.get("fragment")
        if type(fragment) is not int:
            raise RunnerError(f"{arm} event tape has a noninteger fragment")
        fragments.append(fragment)
    if fragments != FRAGMENT_ORDER:
        raise RunnerError(f"{arm} event tape fragment order differs from frozen order")


def _verify_export(root: Path, arm: str) -> None:
    metadata = root.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RunnerError(f"{arm} export is not a regular directory: {root}")
    _require_regular(root / "adapter_config.json", f"{arm} adapter config")
    payloads = [
        path
        for path in (root / "adapter_model.safetensors", root / "adapter_model.bin")
        if _path_exists(path)
    ]
    if len(payloads) != 1:
        raise RunnerError(f"{arm} export must have exactly one adapter payload")
    _require_regular(payloads[0], f"{arm} adapter payload")


def _verify_required_outputs(
    work_dir: Path,
    report_dir: Path,
    *,
    run_id: str,
    config_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    for path, label in (
        (report_dir / "results.jsonl", "comparison results"),
        (report_dir / "report.md", "comparison report"),
        (work_dir / "train.jsonl", "materialized training rows"),
        (work_dir / "eval.jsonl", "materialized evaluation rows"),
    ):
        _require_regular(path, label)
    eval_rows = _read_jsonl(work_dir / "eval.jsonl", "materialized evaluation rows")
    if len(eval_rows) != 8:
        raise RunnerError(
            f"GPU evaluation split has {len(eval_rows)} rows instead of 8"
        )
    results = _read_jsonl(report_dir / "results.jsonl", "comparison results")
    if [row.get("arm") for row in results] != [BASE_ARM, STOCK_ARM, CANDIDATE_ARM]:
        raise RunnerError("comparison result rows are not the frozen closed order")
    for row in results:
        loss = row.get("eval_loss")
        if type(loss) not in (int, float) or not math.isfinite(loss):
            raise RunnerError(f"comparison result for {row.get('arm')!r} is not finite")

    initial: dict[str, dict[str, Any]] = {}
    completion: dict[str, dict[str, Any]] = {}
    learner: dict[str, dict[str, Any]] = {}
    for arm, optimizer in (
        (STOCK_ARM, "nesterov"),
        (CANDIDATE_ARM, "cplg-sgd"),
    ):
        arm_root = work_dir / arm
        for path, label in (
            (arm_root / "tape.jsonl", f"{arm} event tape"),
            (arm_root / "state.ckpt", f"{arm} checkpoint"),
            (arm_root / "syncer.log", f"{arm} syncer log"),
            (arm_root / "learner-0.log", f"{arm} learner log"),
            (arm_root / "export.log", f"{arm} export log"),
        ):
            _require_regular(path, label)
        _verify_event_schedule(arm_root / "tape.jsonl", arm)
        _verify_export(arm_root / "export", arm)
        initial[arm] = _verify_initial_receipt(
            arm_root / "cplg_online_initial_state.json",
            run_id=run_id,
            config_sha256=config_sha256,
            source_commit=source_commit,
            arm=arm,
            optimizer=optimizer,
        )
        completion[arm] = _verify_completion_receipt(
            arm_root / "cplg_online_completion.json",
            arm=arm,
            run_id=run_id,
            event_tape=arm_root / "tape.jsonl",
            checkpoint=arm_root / "state.ckpt",
        )
        learner[arm] = _verify_learner_receipt(
            arm_root / "learner-0" / "learner_completion.json", arm
        )
    for field in ("layout_sha256", "initial_state_sha256"):
        if initial[STOCK_ARM][field] != initial[CANDIDATE_ARM][field]:
            raise RunnerError(f"fresh arms differ at sealed initial {field}")
    if learner[STOCK_ARM]["global_step"] != learner[CANDIDATE_ARM]["global_step"]:
        raise RunnerError("fresh arms differ at learner terminal global step")

    stock = work_dir / STOCK_ARM
    for basename in ("cplg_action_ledger.jsonl", "cplg_action_ledger_manifest.json"):
        if _path_exists(stock / basename) or _path_exists(stock / f"{basename}.sha256"):
            raise RunnerError(
                f"stock arm unexpectedly contains candidate ledger {basename}"
            )
    candidate = work_dir / CANDIDATE_ARM
    ledger = candidate / "cplg_action_ledger.jsonl"
    _require_regular(ledger, "candidate action ledger")
    ledger_rows = _read_jsonl(ledger, "candidate action ledger")
    if len(ledger_rows) != 32:
        raise RunnerError("candidate action ledger does not contain exactly 32 rows")
    ledger_manifest, _digest = _checksummed_json(
        candidate / "cplg_action_ledger_manifest.json",
        "candidate action-ledger manifest",
    )
    if ledger_manifest.get("ledger_head") != completion[CANDIDATE_ARM]["ledger_head"]:
        raise RunnerError(
            "candidate ledger manifest head differs from completion receipt"
        )
    if ledger_manifest.get("ledger_rows") != 32:
        raise RunnerError("candidate ledger manifest does not bind exactly 32 rows")
    return {
        "layout_sha256": initial[STOCK_ARM]["layout_sha256"],
        "initial_state_sha256": initial[STOCK_ARM]["initial_state_sha256"],
        "terminal_local_steps": 34,
        "raw_training_tokens": 4352,
        "commits_per_arm": 32,
        "evaluation_rows": 8,
    }


def _walk_regular_files(root: Path) -> list[Path]:
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise RunnerError(f"acquisition tree root is missing: {root}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RunnerError(f"acquisition tree root is not a regular directory: {root}")
    files: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            path = directory / entry.name
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise RunnerError(f"acquisition tree contains a symlink: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(path)
            else:
                raise RunnerError(f"acquisition tree contains a special file: {path}")

    visit(root)
    return files


def _manifest_files(
    root: Path,
    *,
    excluded: set[Path],
    included: Iterable[Path] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    root = _absolute(root)
    excluded = {_absolute(path) for path in excluded}
    candidates: list[Path] = []
    if included is None:
        candidates = _walk_regular_files(root)
    else:
        for entry in included:
            entry = _absolute(entry)
            if entry != root and root not in entry.parents:
                raise RunnerError(f"manifest input escapes acquisition root: {entry}")
            metadata = entry.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RunnerError(f"acquisition tree contains a symlink: {entry}")
            if stat.S_ISDIR(metadata.st_mode):
                candidates.extend(_walk_regular_files(entry))
            elif stat.S_ISREG(metadata.st_mode):
                candidates.append(entry)
            else:
                raise RunnerError(f"acquisition tree contains a special file: {entry}")
    files: list[dict[str, Any]] = []
    total_bytes = 0
    seen: set[Path] = set()
    for path in sorted(
        candidates, key=lambda value: value.relative_to(root).as_posix()
    ):
        if path in excluded or path in seen:
            continue
        seen.add(path)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RunnerError(f"acquisition tree contains a non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        files.append(
            {"path": relative, "bytes": metadata.st_size, "sha256": _sha256_file(path)}
        )
        total_bytes += metadata.st_size
    if not files:
        raise RunnerError("acquisition tree contains no sealable files")
    return files, total_bytes


def _fsync_files(paths: Iterable[Path]) -> None:
    directories: set[Path] = set()
    for path in paths:
        _require_regular(path, "durable acquisition object", nonempty=False)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directories.add(path.parent)
    for directory in sorted(directories):
        _fsync_directory(directory)


def _stable_manifest_inputs(
    *, root: Path, work_dir: Path, report_dir: Path, input_manifest: Path
) -> list[Path]:
    inputs = [work_dir, report_dir, input_manifest, root / "input-provenance"]
    for basename in ("spec.json", "command.sh", "git-status.txt", "git-diff.patch"):
        path = root / basename
        _require_regular(path, f"frozen launch evidence {basename}", nonempty=False)
        inputs.append(path)
    return inputs


def run_acquisition(
    *,
    run_id: str,
    run_config: Path,
    input_manifest: Path,
    expected_source_commit: str,
    manifest_output: Path,
    terminal_output: Path,
    compare_argv: list[str],
    python: str,
) -> dict[str, Any]:
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise RunnerError(f"noncanonical run ID {run_id!r}")
    if COMMIT_RE.fullmatch(expected_source_commit) is None:
        raise RunnerError("expected source commit must be 40 lowercase hex characters")
    run_config = _absolute(run_config)
    config, config_sha256 = _checksummed_json(run_config, "frozen run configuration")
    if config_sha256 != CONFIG_SHA256:
        raise RunnerError(
            f"frozen run-config SHA-256 must be {CONFIG_SHA256}, got {config_sha256}"
        )
    work_dir, report_dir = _validate_frozen_config(
        config,
        compare_argv,
        run_id=run_id,
        run_config_sha256=config_sha256,
        source_commit=expected_source_commit,
    )
    root = work_dir.parent
    manifest_output = _absolute(manifest_output)
    terminal_output = _absolute(terminal_output)
    input_manifest = _absolute(input_manifest)
    if manifest_output != report_dir / "acquisition_manifest.json":
        raise RunnerError("manifest output must be report/acquisition_manifest.json")
    if terminal_output != report_dir / "acquisition_terminal.json":
        raise RunnerError("terminal output must be report/acquisition_terminal.json")
    for path, label in ((work_dir, "work-dir"), (report_dir, "report-dir")):
        if _path_exists(path):
            raise RunnerError(f"{label} must be fresh and absent before launch: {path}")
    for output in (manifest_output, terminal_output):
        if _path_exists(output) or _path_exists(
            output.with_name(output.name + ".sha256")
        ):
            raise RunnerError(f"output must be fresh before launch: {output}")
    _require_clean_checkout(expected_source_commit)
    provenance = _verify_input_provenance(
        input_manifest,
        root=root,
        config=config,
        source_commit=expected_source_commit,
    )
    canonical_compare_argv = _build_compare_argv(
        config,
        run_id=run_id,
        run_config_sha256=config_sha256,
        source_commit=expected_source_commit,
        work_dir=work_dir,
        report_dir=report_dir,
    )

    try:
        subprocess.run(
            [python, str(COMPARE), *canonical_compare_argv],
            cwd=REPO_ROOT,
            check=True,
        )
        _require_clean_checkout(expected_source_commit)
        evidence = _verify_required_outputs(
            work_dir,
            report_dir,
            run_id=run_id,
            config_sha256=config_sha256,
            source_commit=expected_source_commit,
        )
        frozen_copy = report_dir / "frozen_run_config.json"
        _atomic_publish(frozen_copy, run_config.read_bytes())
        _atomic_publish(
            frozen_copy.with_name(frozen_copy.name + ".sha256"),
            f"{config_sha256}  {frozen_copy.name}\n".encode("ascii"),
        )
        excluded = {
            manifest_output,
            manifest_output.with_name(manifest_output.name + ".sha256"),
            terminal_output,
            terminal_output.with_name(terminal_output.name + ".sha256"),
        }
        stable_inputs = _stable_manifest_inputs(
            root=root,
            work_dir=work_dir,
            report_dir=report_dir,
            input_manifest=input_manifest,
        )
        files, total_bytes = _manifest_files(
            root, excluded=excluded, included=stable_inputs
        )
        _fsync_files(root / item["path"] for item in files)
        # Re-hash after fsync so the manifest describes the exact durable bytes.
        files, total_bytes = _manifest_files(
            root, excluded=excluded, included=stable_inputs
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "ACQUIRED",
            "scientific_verdict": None,
            "run_id": run_id,
            "source_commit": expected_source_commit,
            "run_config_sha256": config_sha256,
            "provenance": provenance,
            "arms_in_order": [STOCK_ARM, CANDIDATE_ARM],
            "evidence": evidence,
            "files": files,
            "file_count": len(files),
            "total_bytes": total_bytes,
        }
        manifest_sha256 = _publish_json(manifest_output, manifest)
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "status": "GPU_ACQUISITION_COMPLETE",
            "scientific_verdict": None,
            "run_id": run_id,
            "source_commit": expected_source_commit,
            "run_config_sha256": config_sha256,
            "acquisition_manifest": manifest_output.name,
            "acquisition_manifest_sha256": manifest_sha256,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "gpu_analysis_performed": False,
            "next_action": "round_trip_verify_then_delete_gpu_before_cpu_analysis",
        }
        terminal_sha256 = _publish_json(terminal_output, terminal)
        return {**terminal, "acquisition_terminal_sha256": terminal_sha256}
    except Exception as exc:
        failure = {
            "schema": TERMINAL_SCHEMA,
            "status": "INFRA_FAILURE",
            "scientific_verdict": None,
            "run_id": run_id,
            "source_commit": expected_source_commit,
            "run_config_sha256": config_sha256,
            "acquisition_manifest": None,
            "acquisition_manifest_sha256": None,
            "error_class": type(exc).__name__,
            "error": str(exc),
            "gpu_analysis_performed": False,
            "next_action": "incomplete_run_preserve_evidence_and_teardown",
        }
        try:
            _publish_json(terminal_output, failure)
        except Exception as publish_exc:
            raise RunnerError(
                f"runner failed ({exc}); failure receipt publication also failed: {publish_exc}"
            ) from exc
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--terminal-output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("compare_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.compare_argv[:1] == ["--"]:
        args.compare_argv = args.compare_argv[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_acquisition(
            run_id=args.run_id,
            run_config=args.run_config,
            input_manifest=args.input_manifest,
            expected_source_commit=args.expected_source_commit,
            manifest_output=args.manifest_output,
            terminal_output=args.terminal_output,
            compare_argv=args.compare_argv,
            python=args.python,
        )
    except (RunnerError, OSError, subprocess.CalledProcessError) as exc:
        print(f"CPLG acquisition failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
