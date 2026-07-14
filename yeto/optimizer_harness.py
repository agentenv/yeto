"""Safe, reproducible orchestration for optimizer experiments on GCP Spot VMs.

The harness is deliberately narrower than a general cloud launcher.  It owns
one named VM at a time, records the provider-assigned instance id, and refuses
destructive operations unless the live id and management labels still match.
Experiment specifications are JSON so the controller has no optional parser
dependency and remains usable from the stock Python environment on a laptop.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
LABEL_OWNER = "yeto-optimizer-harness"
DEFAULT_MAX_TOTAL_ACCELERATORS = 8
RUN_ID_RE = re.compile(r"[a-z][a-z0-9-]{2,62}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
GCP_NAME_RE = re.compile(r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?\Z")
GCP_IMAGE_PATH_RE = re.compile(r"projects/([^/]+)/global/images/([^/]+)\Z")
ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
SAFE_IMAGE_PATH_PREFIXES = (
    "/home/",
    "/root/",
    "/tmp/",
    "/var/tmp/",
)


class HarnessError(RuntimeError):
    """A specification, safety, cloud, or remote-execution failure."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HarnessError(f"{where} must be a JSON object")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{where} must be a non-empty string")
    if "\x00" in value:
        raise HarnessError(f"{where} may not contain NUL")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HarnessError(f"{where} must be an integer >= {minimum}")
    return value


def _string_list(value: Any, where: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise HarnessError(f"{where} must be a {qualifier}JSON array")
    return [_string(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _absolute_remote_path(value: Any, where: str) -> str:
    path = _string(value, where)
    parsed = PurePosixPath(path)
    if not parsed.is_absolute() or ".." in parsed.parts:
        raise HarnessError(f"{where} must be an absolute normalized POSIX path")
    if str(parsed) in ("/", "/home", "/root", "/tmp", "/var", "/var/tmp"):
        raise HarnessError(f"{where} is too broad: {path}")
    return str(parsed)


def _image_resource_path(value: Any, where: str, *, default_project: str | None) -> str:
    """Return one exact ``projects/.../global/images/...`` resource path."""
    raw = _string(value, where)
    prefix = "https://www.googleapis.com/compute/v1/"
    if raw.startswith(prefix):
        raw = raw.removeprefix(prefix)
    if GCP_IMAGE_PATH_RE.fullmatch(raw):
        return raw
    if default_project is not None and GCP_NAME_RE.fullmatch(raw):
        return f"projects/{default_project}/global/images/{raw}"
    raise HarnessError(
        f"{where} must identify one exact GCP image by name or resource path"
    )


def _flag_value(command: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(command):
        if token == flag:
            if index + 1 >= len(command) or command[index + 1].startswith("--"):
                return ""
            return command[index + 1]
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _flag_occurrences(command: Sequence[str], flag: str) -> int:
    return sum(token == flag or token.startswith(f"{flag}=") for token in command)


def _set_flag(command: list[str], flag: str, value: str) -> None:
    for index, token in enumerate(command):
        if token == flag:
            if index + 1 >= len(command) or command[index + 1].startswith("--"):
                raise HarnessError(f"command flag {flag} has no value")
            command[index + 1] = value
            return
        if token.startswith(f"{flag}="):
            command[index] = f"{flag}={value}"
            return
    raise HarnessError(f"command is missing required flag {flag}")


def _remove_flag(command: list[str], flag: str, *, takes_value: bool) -> None:
    for index, token in enumerate(command):
        if token == flag:
            del command[index : index + (2 if takes_value else 1)]
            return
        if token.startswith(f"{flag}="):
            del command[index]
            return


def _validate_labels(labels: Mapping[str, Any], run_id: str) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, raw_value in labels.items():
        value = _string(raw_value, f"cloud.labels.{key}")
        if not GCP_NAME_RE.fullmatch(key) or not re.fullmatch(
            r"[-a-z0-9_]{0,63}", value
        ):
            raise HarnessError(f"invalid GCP label: {key}={value}")
        clean[key] = value
    if clean.get("managed-by") != LABEL_OWNER:
        raise HarnessError(f"cloud.labels.managed-by must be {LABEL_OWNER!r}")
    if clean.get("run-id") != run_id:
        raise HarnessError("cloud.labels.run-id must exactly match run_id")
    return clean


def _validate_sanitize_path(path: str, remote_run_dir: str) -> None:
    if not path.startswith(SAFE_IMAGE_PATH_PREFIXES):
        raise HarnessError(
            f"image.sanitize_paths path is outside allowed prefixes: {path}"
        )
    if path in SAFE_IMAGE_PATH_PREFIXES or path.rstrip("/") in (
        "/home",
        "/root",
        "/tmp",
        "/var/tmp",
    ):
        raise HarnessError(f"image.sanitize_paths path is too broad: {path}")
    # Run directories are allowed, as are explicit credential/history files.
    allowed_exact = {
        "/home/shou/.cache/huggingface",
        "/home/shou/.cache/huggingface/token",
        "/home/shou/.config/huggingface",
        "/home/shou/.config/gcloud",
        "/home/shou/.config/gh",
        "/home/shou/.docker",
        "/home/shou/.netrc",
        "/home/shou/.git-credentials",
        "/home/shou/.npmrc",
        "/home/shou/.pypirc",
        "/home/shou/.aws",
        "/home/shou/.bash_history",
        "/home/shou/.ssh",
        "/root/.config/gcloud",
        "/root/.config/huggingface",
        "/root/.config/gh",
        "/root/.docker",
        "/root/.netrc",
        "/root/.git-credentials",
        "/root/.npmrc",
        "/root/.pypirc",
        "/root/.aws",
        "/root/.bash_history",
        "/root/.ssh",
    }
    if path != remote_run_dir and path not in allowed_exact:
        raise HarnessError(
            "image.sanitize_paths accepts only the exact run directory and "
            f"predeclared credential/history files; got {path}"
        )


@dataclass(frozen=True)
class ExperimentSpec:
    path: Path
    raw: dict[str, Any]
    run_id: str
    repo_url: str
    repo_commit: str
    cloud: dict[str, Any]
    execution: dict[str, Any]
    artifacts: dict[str, Any]
    checks: dict[str, Any]
    analysis: tuple[dict[str, Any], ...]
    image: dict[str, Any] | None

    @property
    def project(self) -> str:
        return self.cloud["project"]

    @property
    def zone(self) -> str:
        return self.cloud["zone"]

    @property
    def instance_name(self) -> str:
        return self.cloud["instance_name"]

    @property
    def remote_run_dir(self) -> str:
        return self.execution["remote_run_dir"]

    @property
    def remote_repo_dir(self) -> str:
        return self.execution["remote_repo_dir"]

    @property
    def artifact_uri(self) -> str:
        return self.artifacts["uri"].rstrip("/")

    @property
    def command(self) -> tuple[str, ...]:
        return tuple(self.execution["command"])

    @property
    def env(self) -> dict[str, str]:
        return dict(self.execution["env"])


def load_spec(path: str | Path) -> ExperimentSpec:
    source = Path(path).expanduser().resolve()
    try:
        raw_value = json.loads(
            source.read_text(), object_pairs_hook=_unique_json_object
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read experiment spec {source}: {exc}") from exc
    raw = _object(raw_value, "spec")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise HarnessError(f"schema_version must be {SCHEMA_VERSION}")

    run_id = _string(raw.get("run_id"), "run_id")
    if not RUN_ID_RE.fullmatch(run_id):
        raise HarnessError(
            "run_id must be a lowercase GCP-compatible slug (3-63 chars)"
        )
    repo_url = _string(raw.get("repo_url"), "repo_url")
    repo_commit = _string(raw.get("repo_commit"), "repo_commit")
    if not COMMIT_RE.fullmatch(repo_commit):
        raise HarnessError("repo_commit must be a full 40-character lowercase SHA")

    cloud = _object(raw.get("cloud"), "cloud")
    if cloud.get("provider") != "gcp":
        raise HarnessError("cloud.provider must be 'gcp'")
    for key in ("project", "zone", "instance_name", "machine_type"):
        cloud[key] = _string(cloud.get(key), f"cloud.{key}")
    if not GCP_NAME_RE.fullmatch(cloud["instance_name"]):
        raise HarnessError("cloud.instance_name is not a valid GCP resource name")
    if cloud.get("provisioning_model") != "SPOT":
        raise HarnessError(
            "paid optimizer experiments must explicitly use provisioning_model=SPOT"
        )
    if cloud.get("termination_action") != "DELETE":
        raise HarnessError("cloud.termination_action must be DELETE")
    cloud["boot_disk_size_gb"] = _integer(
        cloud.get("boot_disk_size_gb", 250), "cloud.boot_disk_size_gb", minimum=100
    )
    if "boot_disk_type" in cloud:
        cloud["boot_disk_type"] = _string(
            cloud["boot_disk_type"], "cloud.boot_disk_type"
        )
        if cloud["boot_disk_type"] not in ("pd-standard", "pd-balanced", "pd-ssd"):
            raise HarnessError(
                "cloud.boot_disk_type must be pd-standard, pd-balanced, or pd-ssd"
            )
    inferred_accelerators = re.fullmatch(r"a2-highgpu-([1248])g", cloud["machine_type"])
    if "accelerator_count" in cloud:
        cloud["accelerator_count"] = _integer(
            cloud["accelerator_count"], "cloud.accelerator_count", minimum=1
        )
    elif inferred_accelerators is not None:
        cloud["accelerator_count"] = int(inferred_accelerators.group(1))
    else:
        raise HarnessError(
            "cloud.accelerator_count is required when it cannot be inferred "
            "from an a2-highgpu-{1,2,4,8}g machine type"
        )
    cloud["max_total_accelerators"] = _integer(
        cloud.get("max_total_accelerators", DEFAULT_MAX_TOTAL_ACCELERATORS),
        "cloud.max_total_accelerators",
        minimum=1,
    )
    if cloud["accelerator_count"] > cloud["max_total_accelerators"]:
        raise HarnessError(
            "cloud.accelerator_count exceeds cloud.max_total_accelerators"
        )
    if "max_run_duration_seconds" in cloud:
        cloud["max_run_duration_seconds"] = _integer(
            cloud["max_run_duration_seconds"],
            "cloud.max_run_duration_seconds",
            minimum=60,
        )
        if cloud["max_run_duration_seconds"] > 86_400:
            raise HarnessError("cloud.max_run_duration_seconds must be <= 86400")
    image_sources = [key for key in ("image", "machine_image") if cloud.get(key)]
    if len(image_sources) != 1:
        raise HarnessError("cloud must set exactly one of image or machine_image")
    cloud[image_sources[0]] = _string(
        cloud[image_sources[0]], f"cloud.{image_sources[0]}"
    )
    if "expected_source_image_id" in cloud:
        cloud["expected_source_image_id"] = _string(
            cloud["expected_source_image_id"],
            "cloud.expected_source_image_id",
        )
        if not re.fullmatch(r"[0-9]+", cloud["expected_source_image_id"]):
            raise HarnessError(
                "cloud.expected_source_image_id must contain only digits"
            )
        if image_sources[0] != "image":
            raise HarnessError(
                "cloud.expected_source_image_id is valid only with cloud.image"
            )
        _image_resource_path(
            cloud["image"], "cloud.image", default_project=cloud["project"]
        )
    cloud["labels"] = _validate_labels(
        _object(cloud.get("labels"), "cloud.labels"), run_id
    )
    if not isinstance(cloud.get("adopt_only", False), bool):
        raise HarnessError("cloud.adopt_only must be boolean")
    cloud["adopt_only"] = cloud.get("adopt_only", False)
    scopes = _string_list(
        cloud.get("scopes", ["storage-rw"]), "cloud.scopes", nonempty=True
    )
    cloud["scopes"] = scopes

    execution = _object(raw.get("execution"), "execution")
    source_mode = execution.get("source_mode", "preinstalled_exact")
    if source_mode not in ("preinstalled_exact", "checkout"):
        raise HarnessError(
            "execution.source_mode must be preinstalled_exact or checkout"
        )
    execution["source_mode"] = source_mode
    execution["remote_repo_dir"] = _absolute_remote_path(
        execution.get("remote_repo_dir"), "execution.remote_repo_dir"
    )
    execution["remote_run_dir"] = _absolute_remote_path(
        execution.get("remote_run_dir"), "execution.remote_run_dir"
    )
    if "/runs/" not in f"{execution['remote_run_dir'].rstrip('/')}/":
        raise HarnessError("execution.remote_run_dir must live below a runs directory")
    execution["command"] = _string_list(
        execution.get("command"), "execution.command", nonempty=True
    )
    env = _object(execution.get("env", {}), "execution.env")
    clean_env: dict[str, str] = {}
    for key, value in env.items():
        if not ENV_NAME_RE.fullmatch(key):
            raise HarnessError(f"invalid execution.env name: {key}")
        clean_env[key] = _string(value, f"execution.env.{key}")
    execution["env"] = clean_env
    execution["required_paths"] = [
        _absolute_remote_path(path, f"execution.required_paths[{index}]")
        for index, path in enumerate(execution.get("required_paths", []))
    ]
    execution["required_executables"] = [
        _absolute_remote_path(path, f"execution.required_executables[{index}]")
        for index, path in enumerate(execution.get("required_executables", []))
    ]
    if len(execution["required_executables"]) != len(
        set(execution["required_executables"])
    ):
        raise HarnessError("execution.required_executables must not contain duplicates")
    undeclared_executables = sorted(
        set(execution["required_executables"]) - set(execution["required_paths"])
    )
    if undeclared_executables:
        raise HarnessError(
            "every execution.required_executables entry must also be an exact "
            f"execution.required_paths entry: {undeclared_executables}"
        )
    execution["completion_paths"] = [
        _absolute_remote_path(path, f"execution.completion_paths[{index}]")
        for index, path in enumerate(execution.get("completion_paths", []))
    ]
    execution["checksum_manifests"] = [
        _absolute_remote_path(path, f"execution.checksum_manifests[{index}]")
        for index, path in enumerate(execution.get("checksum_manifests", []))
    ]
    execution["input_checksum_manifests"] = [
        _absolute_remote_path(path, f"execution.input_checksum_manifests[{index}]")
        for index, path in enumerate(execution.get("input_checksum_manifests", []))
    ]
    execution["input_provenance_paths"] = [
        _absolute_remote_path(path, f"execution.input_provenance_paths[{index}]")
        for index, path in enumerate(execution.get("input_provenance_paths", []))
    ]
    provenance_basenames = [
        Path(path).name for path in execution["input_provenance_paths"]
    ]
    if len(provenance_basenames) != len(set(provenance_basenames)):
        raise HarnessError("execution.input_provenance_paths basenames must be unique")
    for manifest in execution["input_checksum_manifests"]:
        if manifest not in execution["input_provenance_paths"]:
            raise HarnessError(
                "every execution.input_checksum_manifests entry must also be an "
                "execution.input_provenance_paths entry"
            )
    for path in execution["input_provenance_paths"]:
        if path not in execution["required_paths"]:
            raise HarnessError(
                "every execution.input_provenance_paths entry must also be an "
                "execution.required_paths entry"
            )
    run_prefix = execution["remote_run_dir"].rstrip("/") + "/"
    for manifest in execution["checksum_manifests"]:
        if not manifest.startswith(run_prefix):
            raise HarnessError(
                "execution.checksum_manifests must live below execution.remote_run_dir"
            )
        if manifest not in execution["completion_paths"]:
            raise HarnessError(
                "every execution.checksum_manifests entry must also be a completion path"
            )
    if not isinstance(execution.get("legacy_completion", False), bool):
        raise HarnessError("execution.legacy_completion must be boolean")
    execution["legacy_completion"] = execution.get("legacy_completion", False)
    if not execution["legacy_completion"] and not execution["completion_paths"]:
        raise HarnessError(
            "execution.completion_paths must be non-empty unless legacy_completion is true"
        )

    artifacts = _object(raw.get("artifacts"), "artifacts")
    uri = _string(artifacts.get("uri"), "artifacts.uri").rstrip("/")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]{1,220}/.+", uri):
        raise HarnessError(
            "artifacts.uri must be a GCS URI with a non-empty object prefix"
        )
    if run_id not in uri:
        raise HarnessError(
            "artifacts.uri must include run_id to prevent run collisions"
        )
    artifacts["uri"] = uri
    artifacts["sync_interval_seconds"] = _integer(
        artifacts.get("sync_interval_seconds", 120),
        "artifacts.sync_interval_seconds",
        minimum=30,
    )

    checks = _object(raw.get("checks", {}), "checks")
    command = execution["command"]
    if checks.get("require_injected_baseline", False):
        value = _flag_value(command, "--baseline-loss")
        if value is None or value == "":
            raise HarnessError(
                "checks.require_injected_baseline requires --baseline-loss VALUE"
            )
        try:
            finite = math.isfinite(float(value))
        except ValueError:
            finite = False
        if not finite:
            raise HarnessError("--baseline-loss must be finite")
        if checks.get("injected_baseline_report_only") is not True:
            raise HarnessError(
                "an injected baseline must declare injected_baseline_report_only=true; "
                "a naked float is not compatible causal provenance"
            )
    if checks.get("require_skip_baseline", False):
        if _flag_occurrences(command, "--skip-baseline") != 1:
            raise HarnessError(
                "checks.require_skip_baseline requires exactly one --skip-baseline"
            )
        if _flag_occurrences(command, "--baseline-loss"):
            raise HarnessError(
                "--skip-baseline experiments may not also declare --baseline-loss"
            )
    expected_flags = _object(checks.get("expected_flags", {}), "checks.expected_flags")
    for flag, expected in expected_flags.items():
        if _flag_occurrences(command, flag) != 1:
            raise HarnessError(f"command must contain exactly one {flag}")
        expected_text = str(expected)
        actual = _flag_value(command, flag)
        if actual != expected_text:
            raise HarnessError(
                f"command flag {flag} must be {expected_text!r}; found {actual!r}"
            )
    strict_budget_raw = checks.get("strict_quorum_step_budget")
    if strict_budget_raw is not None:
        strict_budget = _object(strict_budget_raw, "checks.strict_quorum_step_budget")
        fragments = _integer(
            strict_budget.get("fragments"),
            "checks.strict_quorum_step_budget.fragments",
            minimum=1,
        )
        min_headroom = _integer(
            strict_budget.get("min_headroom_steps"),
            "checks.strict_quorum_step_budget.min_headroom_steps",
            minimum=0,
        )
        empirical_upper_raw = strict_budget.get("empirical_shutdown_upper_bound_steps")
        post_empirical_headroom_raw = strict_budget.get(
            "min_post_empirical_headroom_steps"
        )
        if (empirical_upper_raw is None) != (post_empirical_headroom_raw is None):
            raise HarnessError(
                "checks.strict_quorum_step_budget."
                "empirical_shutdown_upper_bound_steps and "
                "min_post_empirical_headroom_steps must be set together"
            )
        empirical_upper: int | None = None
        post_empirical_headroom: int | None = None
        if empirical_upper_raw is not None:
            empirical_upper = _integer(
                empirical_upper_raw,
                "checks.strict_quorum_step_budget.empirical_shutdown_upper_bound_steps",
                minimum=1,
            )
            post_empirical_headroom = _integer(
                post_empirical_headroom_raw,
                "checks.strict_quorum_step_budget.min_post_empirical_headroom_steps",
                minimum=0,
            )
        if _flag_occurrences(command, "--strict-quorum") != 1:
            raise HarnessError(
                "checks.strict_quorum_step_budget requires exactly one --strict-quorum"
            )
        numeric_flags: dict[str, int] = {}
        for flag in (
            "--fixed-window-microsteps",
            "--syncer-total-steps",
            "--learner-max-steps",
        ):
            if _flag_occurrences(command, flag) != 1:
                raise HarnessError(
                    "checks.strict_quorum_step_budget requires exactly one " + flag
                )
            value = _flag_value(command, flag)
            try:
                parsed = int(value or "")
            except ValueError as exc:
                raise HarnessError(f"command flag {flag} must be an integer") from exc
            if parsed <= 0:
                raise HarnessError(f"command flag {flag} must be positive")
            numeric_flags[flag] = parsed
        ideal_steps = (
            math.ceil(numeric_flags["--syncer-total-steps"] / fragments)
            * numeric_flags["--fixed-window-microsteps"]
        )
        ideal_required_steps = ideal_steps + min_headroom
        empirical_required_steps = (
            None
            if empirical_upper is None or post_empirical_headroom is None
            else empirical_upper + post_empirical_headroom
        )
        required_steps = max(
            ideal_required_steps,
            empirical_required_steps or 0,
        )
        if numeric_flags["--learner-max-steps"] < required_steps:
            empirical_detail = ""
            if empirical_required_steps is not None:
                empirical_detail = (
                    ", empirical_shutdown_upper_bound_steps="
                    f"{empirical_upper}, min_post_empirical_headroom_steps="
                    f"{post_empirical_headroom}, "
                    f"empirical_required>={empirical_required_steps}"
                )
            raise HarnessError(
                "strict-quorum learner cap cannot reach the declared syncer budget "
                "with its required liveness headroom: "
                f"--learner-max-steps={numeric_flags['--learner-max-steps']}, "
                f"ideal={ideal_steps}, min_headroom_steps={min_headroom}, "
                f"ideal_required>={ideal_required_steps}"
                f"{empirical_detail}, "
                f"required>={required_steps}"
            )
        normalized_budget = {
            "fragments": fragments,
            "min_headroom_steps": min_headroom,
            "ideal_learner_steps": ideal_steps,
            "required_learner_steps": required_steps,
        }
        if empirical_required_steps is not None:
            normalized_budget.update(
                {
                    "empirical_shutdown_upper_bound_steps": empirical_upper,
                    "min_post_empirical_headroom_steps": post_empirical_headroom,
                    "empirical_required_learner_steps": empirical_required_steps,
                }
            )
        checks["strict_quorum_step_budget"] = normalized_budget
    expected_arms = _string_list(
        checks.get("expected_arms", []), "checks.expected_arms"
    )
    if expected_arms:
        if _flag_occurrences(command, "--settings") != 1:
            raise HarnessError("command must contain exactly one --settings")
        settings = _flag_value(command, "--settings")
        actual_arms = set((settings or "").split(","))
        if set(expected_arms) != actual_arms:
            raise HarnessError(
                f"--settings arms {sorted(actual_arms)} do not match expected_arms "
                f"{sorted(expected_arms)}"
            )
    checks["expected_arms"] = expected_arms

    raw_analysis = raw.get("analysis", [])
    if not isinstance(raw_analysis, list):
        raise HarnessError("analysis must be a JSON array")
    analysis: list[dict[str, Any]] = []
    seen_hooks: set[str] = set()
    for index, value in enumerate(raw_analysis):
        hook = _object(value, f"analysis[{index}]")
        name = _string(hook.get("name"), f"analysis[{index}].name")
        if not RUN_ID_RE.fullmatch(name) or name in seen_hooks:
            raise HarnessError(f"analysis hook name is invalid or duplicated: {name}")
        seen_hooks.add(name)
        analysis.append(
            {
                "name": name,
                "command": _string_list(
                    hook.get("command"), f"analysis[{index}].command", nonempty=True
                ),
            }
        )

    image_value = raw.get("image")
    image: dict[str, Any] | None = None
    if image_value is not None:
        image = _object(image_value, "image")
        for key in ("name", "family", "storage_location"):
            image[key] = _string(image.get(key), f"image.{key}")
        if not GCP_NAME_RE.fullmatch(image["name"]) or not GCP_NAME_RE.fullmatch(
            image["family"]
        ):
            raise HarnessError("image.name and image.family must be valid GCP names")
        paths = [
            _absolute_remote_path(path, f"image.sanitize_paths[{index}]")
            for index, path in enumerate(image.get("sanitize_paths", []))
        ]
        if execution["remote_run_dir"] not in paths:
            raise HarnessError(
                "image.sanitize_paths must include the exact remote run directory"
            )
        for path in paths:
            _validate_sanitize_path(path, execution["remote_run_dir"])
        image["sanitize_paths"] = paths
        default_canary_name = f"{image['name']}-canary"
        image["canary_name"] = _string(
            image.get("canary_name", default_canary_name), "image.canary_name"
        )
        image["canary_machine_type"] = _string(
            image.get("canary_machine_type", "a2-highgpu-1g"),
            "image.canary_machine_type",
        )
        image["canary_zone"] = _string(
            image.get("canary_zone", cloud["zone"]), "image.canary_zone"
        )
        if not GCP_NAME_RE.fullmatch(image["canary_name"]):
            raise HarnessError("image.canary_name must be a valid GCP instance name")

    return ExperimentSpec(
        path=source,
        raw=raw,
        run_id=run_id,
        repo_url=repo_url,
        repo_commit=repo_commit,
        cloud=cloud,
        execution=execution,
        artifacts=artifacts,
        checks=checks,
        analysis=tuple(analysis),
        image=image,
    )


def matched_sgd_command(
    spec: ExperimentSpec,
    analysis: Mapping[str, Any],
    *,
    subdirectory: str = "matched-sgd",
) -> list[str]:
    """Derive the frozen matched-control argv from a completed scale fit."""
    if not RUN_ID_RE.fullmatch(subdirectory):
        raise HarnessError("matched subdirectory must be a lowercase slug")
    try:
        source_outer_lr = float(analysis["source_outer_lr"])
        eta_match = float(analysis["eta_match"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessError("analysis has no numeric source_outer_lr/eta_match") from exc
    configured_outer_lr = _flag_value(spec.command, "--outer-lr")
    if configured_outer_lr is None or not math.isclose(
        source_outer_lr, float(configured_outer_lr), rel_tol=0, abs_tol=1e-15
    ):
        raise HarnessError(
            "analysis source_outer_lr does not match the exact control command"
        )
    if not math.isfinite(eta_match) or not 0 < eta_match <= 2:
        raise HarnessError(
            "eta_match must be finite and in the predeclared safety range (0,2]"
        )
    if analysis.get("same_state_causal_fit") is not False:
        raise HarnessError(
            "analysis must explicitly identify the fit as non-causal closed-loop projection"
        )
    if "scaffold_sgd" not in spec.checks.get("expected_arms", []):
        raise HarnessError("base spec has no exact scaffold_sgd control arm")

    command = list(spec.command)
    _set_flag(command, "--settings", "scaffold_sgd")
    _set_flag(command, "--outer-lr", repr(eta_match))
    _set_flag(
        command,
        "--work-dir",
        f"{spec.remote_run_dir}/{subdirectory}/work",
    )
    _set_flag(
        command,
        "--report-dir",
        f"{spec.remote_run_dir}/{subdirectory}/report",
    )
    _remove_flag(command, "--syncer-probe-capture", takes_value=False)
    _remove_flag(command, "--syncer-probe-capture-every", takes_value=True)
    return command


def state_path(spec: ExperimentSpec, state_dir: str | Path | None = None) -> Path:
    root = (
        Path(state_dir).expanduser()
        if state_dir is not None
        else Path(
            os.environ.get("YETO_OPTIMIZER_STATE_DIR", ".optimizer-harness/state")
        )
    )
    return root.resolve() / f"{spec.run_id}.json"


def load_state(
    spec: ExperimentSpec, state_dir: str | Path | None = None
) -> dict[str, Any]:
    path = state_path(spec, state_dir)
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read state {path}: {exc}") from exc
    if state.get("run_id") != spec.run_id or state.get("project") != spec.project:
        raise HarnessError(f"state {path} does not belong to this spec")
    return state


def save_state(
    spec: ExperimentSpec, state: Mapping[str, Any], state_dir: str | Path | None = None
) -> Path:
    path = state_path(spec, state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(state), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


@dataclass
class CommandRunner:
    dry_run: bool = False
    verbose: bool = True

    def run(
        self,
        command: Sequence[str],
        *,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if self.verbose or self.dry_run:
            print("+ " + shlex.join(command), file=sys.stderr)
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, "", "")
        result = subprocess.run(
            list(command), text=True, capture_output=capture, check=False
        )
        if check and result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise HarnessError(
                f"command failed ({result.returncode}): {shlex.join(command)}"
                + (f"\n{detail}" if detail else "")
            )
        return result


def _gcloud_prefix(spec: ExperimentSpec) -> list[str]:
    return ["gcloud", "compute"]


def describe_command(spec: ExperimentSpec) -> list[str]:
    return _gcloud_prefix(spec) + [
        "instances",
        "describe",
        spec.instance_name,
        f"--project={spec.project}",
        f"--zone={spec.zone}",
        "--format=json",
    ]


def launch_command(
    spec: ExperimentSpec, ownership_nonce: str | None = None
) -> list[str]:
    cloud = spec.cloud
    labels = dict(cloud["labels"])
    if ownership_nonce is not None:
        if not re.fullmatch(r"[a-f0-9]{16}", ownership_nonce):
            raise HarnessError(
                "ownership nonce must be 16 lowercase hexadecimal characters"
            )
        labels["ownership-nonce"] = ownership_nonce
    command = _gcloud_prefix(spec) + [
        "instances",
        "create",
        spec.instance_name,
        f"--project={spec.project}",
        f"--zone={spec.zone}",
        f"--machine-type={cloud['machine_type']}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        "--metadata=block-project-ssh-keys=true",
        f"--boot-disk-size={cloud['boot_disk_size_gb']}GB",
        f"--scopes={','.join(cloud['scopes'])}",
        "--labels="
        + ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
        "--format=json",
        "--quiet",
    ]
    if cloud.get("boot_disk_type"):
        command.append(f"--boot-disk-type={cloud['boot_disk_type']}")
    if cloud.get("max_run_duration_seconds"):
        command.append(f"--max-run-duration={cloud['max_run_duration_seconds']}s")
    if cloud.get("image"):
        command.append(f"--image={cloud['image']}")
    else:
        command.append(f"--source-machine-image={cloud['machine_image']}")
    if cloud.get("network"):
        command.append(f"--network={cloud['network']}")
    if cloud.get("subnet"):
        command.append(f"--subnet={cloud['subnet']}")
    return command


def ssh_command(spec: ExperimentSpec, remote_command: str) -> list[str]:
    return _gcloud_prefix(spec) + [
        "ssh",
        spec.instance_name,
        f"--project={spec.project}",
        f"--zone={spec.zone}",
        f"--command={remote_command}",
    ]


def _instance_id(description: Mapping[str, Any]) -> str:
    value = description.get("id")
    if value is None:
        raise HarnessError("GCP instance description has no provider id")
    return str(value)


def verify_description(
    spec: ExperimentSpec,
    description: Mapping[str, Any],
    *,
    expected_id: str | None = None,
    expected_nonce: str | None = None,
    require_spec_labels: bool = True,
) -> str:
    if description.get("name") != spec.instance_name:
        raise HarnessError("live GCP instance name does not match spec")
    instance_id = _instance_id(description)
    if expected_id is not None and instance_id != str(expected_id):
        raise HarnessError(
            f"REFUSING: live instance id {instance_id} != recorded exact id {expected_id}"
        )
    self_link = str(description.get("selfLink", ""))
    if f"/projects/{spec.project}/" not in self_link or not self_link.endswith(
        f"/zones/{spec.zone}/instances/{spec.instance_name}"
    ):
        raise HarnessError(
            "live GCP instance project/zone identity does not match spec"
        )
    labels = description.get("labels", {})
    if require_spec_labels:
        for key, value in spec.cloud["labels"].items():
            if labels.get(key) != value:
                raise HarnessError(
                    f"REFUSING: live instance label {key!r} does not match"
                )
    if expected_nonce is not None and labels.get("ownership-nonce") != expected_nonce:
        raise HarnessError("REFUSING: live instance ownership nonce does not match")
    scheduling = description.get("scheduling", {})
    if scheduling.get("provisioningModel") != "SPOT":
        raise HarnessError("live instance is not Spot")
    if scheduling.get("instanceTerminationAction") != "DELETE":
        raise HarnessError("live instance termination action is not DELETE")
    expected_max_run = spec.cloud.get("max_run_duration_seconds")
    if expected_max_run is not None:
        live_max_run = scheduling.get("maxRunDuration")
        if not isinstance(live_max_run, Mapping) or str(
            live_max_run.get("seconds")
        ) != str(expected_max_run):
            raise HarnessError(
                "REFUSING: live instance max run duration does not match spec"
            )
    return instance_id


def describe(spec: ExperimentSpec, runner: CommandRunner) -> dict[str, Any]:
    result = runner.run(describe_command(spec))
    if runner.dry_run:
        return {}
    try:
        return _object(json.loads(result.stdout), "gcloud instance description")
    except json.JSONDecodeError as exc:
        raise HarnessError(f"gcloud returned invalid instance JSON: {exc}") from exc


def _boot_disk_self_link(description: Mapping[str, Any]) -> str:
    disks = description.get("disks", [])
    boot = [disk for disk in disks if disk.get("boot")]
    if len(boot) != 1 or not boot[0].get("source"):
        raise HarnessError("expected exactly one identified boot disk")
    return str(boot[0]["source"])


def _numeric_provider_id(value: Any, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise HarnessError(f"{where} must be a numeric provider id")
    normalized = str(value)
    if not re.fullmatch(r"[0-9]+", normalized):
        raise HarnessError(f"{where} must be a numeric provider id")
    return normalized


def _verified_boot_disk_provenance(
    spec: ExperimentSpec,
    description: Mapping[str, Any],
    runner: CommandRunner,
) -> dict[str, str] | None:
    """Verify and return the pinned image provenance of one exact boot disk."""
    expected_source_image_id = spec.cloud.get("expected_source_image_id")
    if expected_source_image_id is None:
        return None
    boot_disk_self_link = _boot_disk_self_link(description)
    disk_name = boot_disk_self_link.rstrip("/").split("/")[-1]
    if not disk_name:
        raise HarnessError("could not determine exact boot disk name")
    result = runner.run(disk_describe_command(spec, disk_name))
    disk = _json_object_from_result(result, "boot disk description")
    disk_id = _numeric_provider_id(disk.get("id"), "boot disk id")
    if disk.get("selfLink") != boot_disk_self_link:
        raise HarnessError(
            "REFUSING: described boot disk self-link differs from the instance attachment"
        )
    instance_self_link = _string(description.get("selfLink"), "instance selfLink")
    if disk.get("users") != [instance_self_link]:
        raise HarnessError(
            "REFUSING: described boot disk is not bound only to the exact instance"
        )
    source_image_id = _numeric_provider_id(
        disk.get("sourceImageId"), "boot disk sourceImageId"
    )
    if source_image_id != expected_source_image_id:
        raise HarnessError(
            "REFUSING: boot disk sourceImageId differs from "
            "cloud.expected_source_image_id"
        )
    expected_image_path = _image_resource_path(
        spec.cloud["image"], "cloud.image", default_project=spec.project
    )
    source_image = _string(disk.get("sourceImage"), "boot disk sourceImage")
    source_image_path = _image_resource_path(
        source_image, "boot disk sourceImage", default_project=None
    )
    if source_image_path != expected_image_path:
        raise HarnessError(
            "REFUSING: boot disk sourceImage path differs from the pinned cloud.image"
        )
    return {
        "name": disk_name,
        "id": disk_id,
        "self_link": boot_disk_self_link,
        "user": instance_self_link,
        "source_image_id": source_image_id,
        "source_image": source_image,
        "source_image_path": source_image_path,
    }


def _base_state(
    spec: ExperimentSpec,
    description: Mapping[str, Any],
    ownership_nonce: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": spec.run_id,
        "spec_path": str(spec.path),
        "project": spec.project,
        "zone": spec.zone,
        "instance_name": spec.instance_name,
        "instance_id": _instance_id(description),
        "instance_self_link": description.get("selfLink"),
        "boot_disk_self_link": _boot_disk_self_link(description),
        "ownership_nonce": ownership_nonce,
        "labels": dict(description.get("labels", {})),
        "status": description.get("status"),
        "repo_commit": spec.repo_commit,
        "artifact_uri": spec.artifact_uri,
    }


def _active_project_accelerators(spec: ExperimentSpec, runner: CommandRunner) -> int:
    """Count accelerators attached to every non-terminated VM in the project.

    The user-facing campaign cap is aggregate, so this deliberately includes
    accelerators on instances not owned by this harness.  Such instances are
    never mutated; they merely consume capacity in the fail-closed launch
    calculation.
    """
    result = runner.run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={spec.project}",
            "--format=json",
        ]
    )
    if runner.dry_run:
        return 0
    try:
        instances = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HarnessError(
            f"gcloud returned invalid instance inventory JSON: {exc}"
        ) from exc
    if not isinstance(instances, list):
        raise HarnessError("gcloud instance inventory must be a JSON array")
    total = 0
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise HarnessError(f"instance inventory row {index} is not an object")
        if instance.get("status") == "TERMINATED":
            continue
        accelerators = instance.get("guestAccelerators", [])
        if not isinstance(accelerators, list):
            raise HarnessError(
                f"instance inventory row {index} guestAccelerators is not an array"
            )
        for accelerator in accelerators:
            if not isinstance(accelerator, dict):
                raise HarnessError(
                    f"instance inventory row {index} has an invalid accelerator entry"
                )
            count = accelerator.get("acceleratorCount")
            if isinstance(count, bool):
                raise HarnessError("acceleratorCount may not be boolean")
            try:
                parsed = int(count)
            except (TypeError, ValueError) as exc:
                raise HarnessError(
                    f"instance inventory row {index} has invalid acceleratorCount {count!r}"
                ) from exc
            if parsed < 0:
                raise HarnessError("acceleratorCount may not be negative")
            total += parsed
    return total


def launch(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
    *,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise HarnessError("launch requires --yes")
    if spec.cloud.get("adopt_only"):
        raise HarnessError(
            "this provenance spec is adopt_only and may not launch a new VM"
        )
    path = state_path(spec, state_dir)
    if path.exists():
        raise HarnessError(
            f"state already exists: {path}; inspect it instead of relaunching"
        )
    existing = runner.run(["gcloud", "storage", "ls", spec.artifact_uri], check=False)
    if not runner.dry_run and existing.returncode == 0 and existing.stdout.strip():
        raise HarnessError(
            f"artifact prefix is not empty: {spec.artifact_uri}; choose a new run_id"
        )
    active_accelerators = _active_project_accelerators(spec, runner)
    requested_accelerators = int(spec.cloud["accelerator_count"])
    max_total_accelerators = int(spec.cloud["max_total_accelerators"])
    if active_accelerators + requested_accelerators > max_total_accelerators:
        raise HarnessError(
            "REFUSING: accelerator aggregate would exceed the campaign cap: "
            f"active={active_accelerators} requested={requested_accelerators} "
            f"cap={max_total_accelerators}"
        )
    ownership_nonce = secrets.token_hex(8)
    runner.run(launch_command(spec, ownership_nonce))
    if runner.dry_run:
        return {"dry_run": True}
    description = describe(spec, runner)
    verify_description(spec, description, expected_nonce=ownership_nonce)
    state = _base_state(spec, description, ownership_nonce)
    # The provider has already created a billable, exactly owned VM. Persist
    # its deletion identity before any later provenance check can fail, so a
    # mismatch is quarantined rather than orphaned outside harness state.
    state["status"] = "PROVENANCE_PENDING"
    save_state(spec, state, state_dir)
    try:
        boot_disk = _verified_boot_disk_provenance(spec, description, runner)
    except HarnessError as exc:
        state["status"] = "PROVENANCE_FAILED"
        state["provenance_error"] = str(exc)
        save_state(spec, state, state_dir)
        raise
    if boot_disk is not None:
        state["boot_disk"] = boot_disk
        state["boot_disk_id"] = boot_disk["id"]
        state["source_image_id"] = boot_disk["source_image_id"]
    state["status"] = description.get("status")
    save_state(spec, state, state_dir)
    return state


def adopt_label_command(spec: ExperimentSpec, ownership_nonce: str) -> list[str]:
    labels = dict(spec.cloud["labels"])
    labels["ownership-nonce"] = ownership_nonce
    return _gcloud_prefix(spec) + [
        "instances",
        "add-labels",
        spec.instance_name,
        f"--project={spec.project}",
        f"--zone={spec.zone}",
        "--labels="
        + ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
        "--quiet",
    ]


def adopt(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
    *,
    exact_instance_id: str,
    confirmed: bool,
) -> dict[str, Any]:
    """Adopt one already-running instance, only by an explicit provider id."""
    if not confirmed:
        raise HarnessError("adopt requires --yes")
    path = state_path(spec, state_dir)
    if path.exists():
        raise HarnessError(f"state already exists: {path}")
    description = describe(spec, runner)
    if not runner.dry_run:
        verify_description(
            spec,
            description,
            expected_id=exact_instance_id,
            require_spec_labels=False,
        )
    ownership_nonce = secrets.token_hex(8)
    runner.run(adopt_label_command(spec, ownership_nonce))
    if runner.dry_run:
        return {"dry_run": True, "ownership_nonce": ownership_nonce}
    description = describe(spec, runner)
    verify_description(
        spec,
        description,
        expected_id=exact_instance_id,
        expected_nonce=ownership_nonce,
    )
    state = _base_state(spec, description, ownership_nonce)
    # add-labels has already mutated the live instance. Record the exact owned
    # identity before source-image verification so a failed adoption remains
    # recoverable through the explicit abandonment path.
    state["status"] = "PROVENANCE_PENDING"
    save_state(spec, state, state_dir)
    try:
        boot_disk = _verified_boot_disk_provenance(spec, description, runner)
    except HarnessError as exc:
        state["status"] = "PROVENANCE_FAILED"
        state["provenance_error"] = str(exc)
        save_state(spec, state, state_dir)
        raise
    if boot_disk is not None:
        state["boot_disk"] = boot_disk
        state["boot_disk_id"] = boot_disk["id"]
        state["source_image_id"] = boot_disk["source_image_id"]
    state["status"] = description.get("status")
    save_state(spec, state, state_dir)
    return state


def _remote_manifest(spec: ExperimentSpec) -> str:
    public_spec = json.dumps(spec.raw, indent=2, sort_keys=True) + "\n"
    return base64.b64encode(public_spec.encode()).decode()


def _runner_body(spec: ExperimentSpec) -> str:
    """Render the inner runner program before quoting it as one shell word."""
    completion = " ".join(
        shlex.quote(path) for path in spec.execution["completion_paths"]
    )
    checksum_verification = "\n".join(
        f"""  if [ \"$code\" -eq 0 ]; then
    manifest={shlex.quote(manifest)}
    if [ ! -s \"$manifest\" ]; then
      echo 'declared checksum manifest is missing' >&2
      code=14
    elif ! (cd \"$(dirname \"$manifest\")\" && sha256sum -c \"$(basename \"$manifest\")\" >/dev/null); then
      echo 'declared checksum manifest verification failed' >&2
      code=14
    fi
  fi"""
        for manifest in spec.execution["checksum_manifests"]
    )
    return f"""set +e
run="$1"; shift
"$@"
code=$?
{checksum_verification}
if [ "$code" -eq 0 ]; then
  if sha256sum {completion} > "$run/final-manifest.sha256.tmp"; then
    mv "$run/final-manifest.sha256.tmp" "$run/final-manifest.sha256"
  else
    rm -f "$run/final-manifest.sha256.tmp"
    code=15
  fi
fi
printf "%s\\n" "$code" > "$run/runner.exit.tmp"
mv "$run/runner.exit.tmp" "$run/runner.exit"
exit "$code"
"""


def _backup_body() -> str:
    """Render the inner artifact-upload program before shell quoting."""
    return """set +e
pid="$1"; run="$2"; artifact="$3"; interval="$4"
while kill -0 "$pid" 2>/dev/null; do
  gcloud storage rsync --recursive "$run" "$artifact" >> "$run/backup-transfer.log" 2>&1
  sleep "$interval"
done
gcloud storage rsync --recursive "$run" "$artifact" >> "$run/backup-transfer.log" 2>&1
"""


def _render_bash_c(body: str) -> str:
    """Render a nested Bash program without embedding its syntax in SSH text.

    ``gcloud compute ssh --command`` adds another remote-shell parse beyond the
    local ``subprocess`` boundary.  A normally shell-quoted ``bash -c`` body is
    therefore not stable when that body itself contains quotes: the remote
    transport can consume one quoting layer and split the program into Bash's
    positional arguments.  Base64 keeps the transported command lexical-only;
    command substitution decodes the program into one double-quoted ``-c``
    argument, whose contents are not reparsed by the outer shell.
    """

    encoded = base64.b64encode(body.encode()).decode()
    return f'bash -c "$(printf %s {encoded} | base64 -d)"'


def detached_runtime_smoke_script(spec: ExperimentSpec) -> str:
    """Render executable probes with the detached command's spec env assignments.

    Each executable is an absolute, already-declared required path.  Invoking
    ``--version`` catches missing execute permission and broken runtime/linker
    dependencies without starting experiment work.  Empty contracts render a
    no-op so existing specs remain behavior compatible.
    """

    executables = spec.execution["required_executables"]
    if not executables:
        return "true"
    env_assignments = [f"{key}={value}" for key, value in sorted(spec.env.items())]
    lines = []
    for executable in executables:
        quoted = shlex.quote(executable)
        error = shlex.quote(f"required executable is not executable: {executable}")
        lines.append(f"test -x {quoted} || {{ echo {error} >&2; exit 21; }}")
        lines.append(
            shlex.join(["env", *env_assignments, executable, "--version"])
            + " >/dev/null"
        )
    return "\n".join(lines)


def start_script(spec: ExperimentSpec) -> str:
    run = shlex.quote(spec.remote_run_dir)
    repo = shlex.quote(spec.remote_repo_dir)
    commit = shlex.quote(spec.repo_commit)
    artifact = shlex.quote(spec.artifact_uri)
    interval = int(spec.artifacts["sync_interval_seconds"])
    command = shlex.join(spec.command)
    env = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(spec.env.items())
    )
    runner_argv = "env " + (env + " " if env else "") + command
    runner_bash = _render_bash_c(_runner_body(spec))
    backup_bash = _render_bash_c(_backup_body())
    required = "\n".join(
        f"test -e {shlex.quote(path)} || {{ echo 'missing required path: {path}' >&2; exit 3; }}"
        for path in spec.execution["required_paths"]
    )
    runtime_smoke = detached_runtime_smoke_script(spec)
    input_manifest_verification = (
        "\n".join(
            f"echo {shlex.quote('VERIFY ' + manifest)}; "
            f"sha256sum -c {shlex.quote(manifest)}"
            for manifest in spec.execution["input_checksum_manifests"]
        )
        or "true"
    )
    input_provenance_copy = "\n".join(
        f"test -f {shlex.quote(path)} && test ! -L {shlex.quote(path)} || {{ "
        f"echo 'input provenance is not a regular non-symlink file: {path}' >&2; "
        "exit 16; }; "
        f'cp -- {shlex.quote(path)} "$run/input-provenance/"'
        for path in spec.execution["input_provenance_paths"]
    )
    encoded = shlex.quote(_remote_manifest(spec))
    if spec.execution["source_mode"] == "checkout":
        source_setup = f"""if [ -e \"$repo\" ]; then
  echo 'run-specific checkout path already exists' >&2
  exit 15
fi
mkdir -p \"$(dirname \"$repo\")\"
git clone --filter=blob:none --no-checkout {shlex.quote(spec.repo_url)} \"$repo\"
git -C \"$repo\" fetch --depth=1 origin {commit}
git -C \"$repo\" checkout --detach {commit}
"""
    else:
        source_setup = ""
    return f"""set -eu
run={run}
repo={repo}
{source_setup}test \"$(git -C \"$repo\" rev-parse HEAD)\" = {commit} || {{
  echo 'remote repo commit does not match pinned spec' >&2
  exit 2
}}
dirty=$(git -C \"$repo\" status --porcelain=v1 --untracked-files=all)
[ -z \"$dirty\" ] || {{
  echo 'remote repo is dirty or has untracked files; refusing unpinned execution' >&2
  printf '%s\\n' \"$dirty\" >&2
  exit 6
}}
{required}
{runtime_smoke}
mkdir -p \"$run\"
if [ -e \"$run/spec.json\" ]; then
  echo 'run directory was already initialized; choose a new run_id or use status' >&2
  exit 7
fi
if [ -f \"$run/runner.pid\" ] && kill -0 \"$(cat \"$run/runner.pid\")\" 2>/dev/null; then
  echo 'runner is already active' >&2
  exit 4
fi
mkdir -p \"$run/input-provenance\"
{input_provenance_copy}
{{
{input_manifest_verification}
}} > \"$run/input-provenance/verification.log\" 2>&1
find \"$run/input-provenance\" -type f -print0 | sort -z | \
  xargs -0 sha256sum > \"$run/input-provenance.sha256\"
printf %s {encoded} | base64 -d > \"$run/spec.json\"
printf '%s\\n' {shlex.quote(command)} > \"$run/command.sh\"
printf '%s\\n' \"$run/runner.log\" > \"$run/active-log-path\"
printf 'clean detached checkout %s\\n' \\
  \"$(git -C \"$repo\" rev-parse HEAD)\" > \"$run/git-status.txt\"
printf '# no tracked or untracked diff; clean detached checkout %s\\n' \\
  \"$(git -C \"$repo\" rev-parse HEAD)\" > \"$run/git-diff.patch\"
cd \"$repo\"
nohup {runner_bash} _ \"$run\" {runner_argv} > \"$run/runner.log\" 2>&1 < /dev/null &
runner=$!
printf '%s\\n' \"$runner\" > \"$run/runner.pid\"
nohup {backup_bash} _ \"$runner\" \"$run\" {artifact} {interval} >/dev/null 2>&1 < /dev/null &
backup=$!
printf '%s\\n' \"$backup\" > \"$run/backup.pid\"
printf '{{\"runner_pid\":%s,\"backup_pid\":%s}}\\n' \"$runner\" \"$backup\"
"""


def require_live_owned_instance(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_state(spec, state_dir)
    description = describe(spec, runner)
    if not runner.dry_run:
        verify_description(
            spec,
            description,
            expected_id=str(state["instance_id"]),
            expected_nonce=str(state["ownership_nonce"]),
        )
        if _boot_disk_self_link(description) != state.get("boot_disk_self_link"):
            raise HarnessError(
                "REFUSING: live boot disk identity does not match recorded state"
            )
    return state, description


def start(
    spec: ExperimentSpec, runner: CommandRunner, state_dir: str | Path | None
) -> dict[str, Any]:
    state, _ = require_live_owned_instance(spec, runner, state_dir)
    result = runner.run(ssh_command(spec, start_script(spec)))
    if runner.dry_run:
        return {"dry_run": True}
    try:
        remote = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise HarnessError(
            f"remote start did not return PID metadata: {result.stdout}"
        ) from exc
    state.update(remote)
    state["status"] = "RUNNING_EXPERIMENT"
    save_state(spec, state, state_dir)
    return state


def remote_status_script(spec: ExperimentSpec) -> str:
    run = shlex.quote(spec.remote_run_dir)
    return f"""set -eu
run={run}
runner=''
backup=''
[ -f \"$run/runner.pid\" ] && runner=$(cat \"$run/runner.pid\")
[ -f \"$run/backup.pid\" ] && backup=$(cat \"$run/backup.pid\")
runner_live=false; backup_live=false
[ -n \"$runner\" ] && kill -0 \"$runner\" 2>/dev/null && runner_live=true
[ -n \"$backup\" ] && kill -0 \"$backup\" 2>/dev/null && backup_live=true
printf '{{\"runner_pid\":\"%s\",\"runner_live\":%s,\"backup_pid\":\"%s\",\"backup_live\":%s}}\\n' \
  \"$runner\" \"$runner_live\" \"$backup\" \"$backup_live\"
log=\"$run/runner.log\"
[ -s \"$run/active-log-path\" ] && log=$(cat \"$run/active-log-path\")
tail -40 \"$log\" 2>/dev/null || true
"""


def status(
    spec: ExperimentSpec, runner: CommandRunner, state_dir: str | Path | None
) -> str:
    state, description = require_live_owned_instance(spec, runner, state_dir)
    if runner.dry_run:
        runner.run(ssh_command(spec, remote_status_script(spec)))
        return json.dumps({"state": state, "instance": description}, indent=2)
    remote = runner.run(ssh_command(spec, remote_status_script(spec)))
    header, _, log = remote.stdout.partition("\n")
    try:
        remote_state = json.loads(header)
    except json.JSONDecodeError:
        remote_state = {"parse_error": header}
    summary = {
        "instance_id": state["instance_id"],
        "instance_status": description.get("status"),
        "remote": remote_state,
        "artifact_uri": spec.artifact_uri,
    }
    return json.dumps(summary, indent=2, sort_keys=True) + "\n" + log


def logs(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
    lines: int,
) -> str:
    require_live_owned_instance(spec, runner, state_dir)
    lines = max(1, min(lines, 5000))
    run = shlex.quote(spec.remote_run_dir)
    command = (
        f'run={run}; log="$run/runner.log"; '
        '[ -s "$run/active-log-path" ] && log=$(cat "$run/active-log-path"); '
        f'tail -{lines} "$log"'
    )
    return runner.run(ssh_command(spec, command)).stdout


def sync(
    spec: ExperimentSpec, runner: CommandRunner, state_dir: str | Path | None
) -> None:
    require_live_owned_instance(spec, runner, state_dir)
    run = shlex.quote(spec.remote_run_dir)
    uri = shlex.quote(spec.artifact_uri)
    runner.run(ssh_command(spec, f"gcloud storage rsync --recursive {run} {uri}"))


def assert_remote_complete(
    spec: ExperimentSpec,
    runner: CommandRunner,
    additional_completion_paths: Sequence[str] = (),
    additional_checksum_manifests: Sequence[str] = (),
) -> None:
    runner.run(
        ssh_command(
            spec,
            remote_completion_check_script(
                spec,
                additional_completion_paths,
                additional_checksum_manifests,
            ),
        )
    )


def remote_completion_check_script(
    spec: ExperimentSpec,
    additional_completion_paths: Sequence[str] = (),
    additional_checksum_manifests: Sequence[str] = (),
    *,
    require_runner_stopped: bool = True,
) -> str:
    """Return the single source of truth for a remotely complete run."""
    run = shlex.quote(spec.remote_run_dir)
    checks = ["set -eu", f"run={run}"]
    if require_runner_stopped:
        checks.extend(
            [
                'runner=$(cat "$run/runner.pid" 2>/dev/null || true)',
                'if [ -n "$runner" ] && kill -0 "$runner" 2>/dev/null; then '
                "echo 'REFUSING: experiment runner is still live' >&2; exit 8; fi",
            ]
        )
    for path in [*spec.execution["completion_paths"], *additional_completion_paths]:
        checks.append(
            f"test -s {shlex.quote(path)} || {{ echo 'missing completion artifact: "
            f"{path}' >&2; exit 9; }}"
        )
    if not spec.execution["legacy_completion"]:
        checks.extend(
            [
                'test "$(cat "$run/runner.exit" 2>/dev/null)" = 0 || { '
                "echo 'runner exit status is missing or nonzero' >&2; exit 11; }",
                'test -s "$run/final-manifest.sha256" || { '
                "echo 'final checksum manifest is missing' >&2; exit 12; }",
                'cd / && sha256sum -c "$run/final-manifest.sha256" >/dev/null || { '
                "echo 'final checksum verification failed' >&2; exit 13; }",
            ]
        )
    for manifest in [
        *spec.execution["checksum_manifests"],
        *additional_checksum_manifests,
    ]:
        checks.extend(
            [
                f"test -s {shlex.quote(manifest)} || {{ echo 'missing follow-up "
                f"checksum manifest: {manifest}' >&2; exit 19; }}",
                f"manifest={shlex.quote(manifest)}; "
                '(cd "$(dirname "$manifest")" && '
                'sha256sum -c "$(basename "$manifest")" >/dev/null) || { '
                "echo 'follow-up checksum verification failed' >&2; exit 20; }",
            ]
        )
    return "\n".join(checks)


def matched_start_script(
    spec: ExperimentSpec,
    analysis: Mapping[str, Any],
    *,
    subdirectory: str,
) -> tuple[str, list[str], str]:
    if not spec.execution["completion_paths"]:
        raise HarnessError("matched follow-up requires a base completion path")
    command = matched_sgd_command(spec, analysis, subdirectory=subdirectory)
    run = spec.remote_run_dir
    subrun = f"{run}/{subdirectory}"
    result_path = f"{subrun}/report/results.jsonl"
    encoded_analysis = base64.b64encode(
        (json.dumps(dict(analysis), indent=2, sort_keys=True) + "\n").encode()
    ).decode()
    env_argv = [
        "env",
        *[f"{key}={value}" for key, value in sorted(spec.env.items())],
        *command,
    ]
    script = f"""set -eu
run={shlex.quote(run)}
subrun={shlex.quote(subrun)}
repo={shlex.quote(spec.remote_repo_dir)}
runner=$(cat \"$run/runner.pid\" 2>/dev/null || true)
if [ -n \"$runner\" ] && kill -0 \"$runner\" 2>/dev/null; then
  echo 'base experiment runner is still live' >&2
  exit 16
fi
test -s {shlex.quote(spec.execution["completion_paths"][0])} || {{
  echo 'base experiment completion artifact is missing' >&2
  exit 17
}}
if [ -e \"$subrun\" ]; then
  echo 'matched follow-up directory already exists' >&2
  exit 18
fi
mkdir -p \"$subrun\"
printf %s {shlex.quote(encoded_analysis)} | base64 -d > \"$subrun/scale-analysis.json\"
printf '%s\\n' {shlex.quote(shlex.join(command))} > \"$subrun/command.sh\"
printf '%s\\n' \"$subrun/runner.log\" > \"$run/active-log-path\"
cd \"$repo\"
nohup bash -c '
  set +e
  subrun="$1"; result="$2"; shift 2
  "$@"
  code=$?
  printf "%s\\n" "$code" > "$subrun/runner.exit.tmp"
  mv "$subrun/runner.exit.tmp" "$subrun/runner.exit"
  if [ "$code" -eq 0 ]; then
    sha256sum "$result" > "$subrun/final-manifest.sha256.tmp" && \
      mv "$subrun/final-manifest.sha256.tmp" "$subrun/final-manifest.sha256"
  fi
  exit "$code"
' _ \"$subrun\" {shlex.quote(result_path)} {shlex.join(env_argv)} \
  > \"$subrun/runner.log\" 2>&1 < /dev/null &
matched=$!
printf '%s\\n' \"$matched\" > \"$subrun/runner.pid\"
printf '%s\\n' \"$matched\" > \"$run/runner.pid\"
nohup bash -c '
  set +e
  pid="$1"; run="$2"; artifact="$3"; interval="$4"
  while kill -0 "$pid" 2>/dev/null; do
    gcloud storage rsync --recursive "$run" "$artifact" >> "$run/backup-transfer.log" 2>&1
    sleep "$interval"
  done
  gcloud storage rsync --recursive "$run" "$artifact" >> "$run/backup-transfer.log" 2>&1
' _ \"$matched\" \"$run\" {shlex.quote(spec.artifact_uri)} \
  {int(spec.artifacts["sync_interval_seconds"])} >/dev/null 2>&1 < /dev/null &
backup=$!
printf '%s\\n' \"$backup\" > \"$run/backup.pid\"
printf '{{\"runner_pid\":%s,\"backup_pid\":%s}}\\n' \"$matched\" \"$backup\"
"""
    return script, command, result_path


def start_matched(
    spec: ExperimentSpec,
    analysis: Mapping[str, Any],
    runner: CommandRunner,
    state_dir: str | Path | None,
    *,
    subdirectory: str,
) -> dict[str, Any]:
    state, _ = require_live_owned_instance(spec, runner, state_dir)
    assert_remote_complete(spec, runner)
    script, command, result_path = matched_start_script(
        spec, analysis, subdirectory=subdirectory
    )
    result = runner.run(ssh_command(spec, script))
    if runner.dry_run:
        return {"dry_run": True, "command": command, "result_path": result_path}
    try:
        remote = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise HarnessError(
            f"matched start did not return PID metadata: {result.stdout}"
        ) from exc
    state["status"] = "RUNNING_MATCHED_SGD"
    state["runner_pid"] = remote["runner_pid"]
    state["backup_pid"] = remote["backup_pid"]
    state.setdefault("additional_completion_paths", []).append(result_path)
    state.setdefault("additional_checksum_manifests", []).append(
        f"{spec.remote_run_dir}/{subdirectory}/final-manifest.sha256"
    )
    state["matched_sgd"] = {
        "eta_match": float(analysis["eta_match"]),
        "subdirectory": subdirectory,
        "command": command,
        "result_path": result_path,
    }
    save_state(spec, state, state_dir)
    return state["matched_sgd"]


def stop_remote_backup(spec: ExperimentSpec, runner: CommandRunner) -> None:
    run = shlex.quote(spec.remote_run_dir)
    script = f"""set -eu
run={run}
backup=$(cat \"$run/backup.pid\" 2>/dev/null || true)
if [ -n \"$backup\" ] && kill -0 \"$backup\" 2>/dev/null; then
  kill -TERM \"$backup\"
  for _i in $(seq 1 30); do
    kill -0 \"$backup\" 2>/dev/null || break
    sleep 1
  done
  if kill -0 \"$backup\" 2>/dev/null; then
    echo 'backup process did not stop cleanly' >&2
    exit 14
  fi
fi
"""
    runner.run(ssh_command(spec, script))


def analyze(
    spec: ExperimentSpec, runner: CommandRunner, state_dir: str | Path | None
) -> None:
    state, _ = require_live_owned_instance(spec, runner, state_dir)
    assert_remote_complete(
        spec,
        runner,
        state.get("additional_completion_paths", []),
        state.get("additional_checksum_manifests", []),
    )
    if not spec.analysis:
        raise HarnessError("spec defines no analysis hooks")
    run = shlex.quote(spec.remote_run_dir)
    repo = shlex.quote(spec.remote_repo_dir)
    commands = [f"set -eu\nmkdir -p {run}/analysis\ncd {repo}"]
    for hook in spec.analysis:
        log = shlex.quote(f"{spec.remote_run_dir}/analysis/{hook['name']}.log")
        commands.append(f"{shlex.join(hook['command'])} 2>&1 | tee {log}")
    commands.append(
        f"gcloud storage rsync --recursive {run} {shlex.quote(spec.artifact_uri)}"
    )
    runner.run(ssh_command(spec, "\n".join(commands)), capture=False)


def delete_command(spec: ExperimentSpec) -> list[str]:
    return _gcloud_prefix(spec) + [
        "instances",
        "delete",
        spec.instance_name,
        f"--project={spec.project}",
        f"--zone={spec.zone}",
        "--quiet",
    ]


def delete(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
    *,
    exact_instance_id: str,
    confirmed: bool,
) -> None:
    if not confirmed:
        raise HarnessError("delete requires --yes")
    state, description = require_live_owned_instance(spec, runner, state_dir)
    recorded = str(state["instance_id"])
    if exact_instance_id != recorded:
        raise HarnessError(
            f"REFUSING: --instance-id {exact_instance_id} != recorded exact id {recorded}"
        )
    if description.get("status") == "RUNNING":
        assert_remote_complete(
            spec,
            runner,
            state.get("additional_completion_paths", []),
            state.get("additional_checksum_manifests", []),
        )
        sync(spec, runner, state_dir)
        stop_remote_backup(spec, runner)
    runner.run(delete_command(spec))
    if runner.dry_run:
        return
    probe = runner.run(describe_command(spec), check=False)
    if probe.returncode == 0:
        raise HarnessError("delete returned but exact instance still exists")
    state["status"] = "DELETED"
    save_state(spec, state, state_dir)


def disk_describe_command(
    spec: ExperimentSpec, disk_name: str, *, zone: str | None = None
) -> list[str]:
    return _gcloud_prefix(spec) + [
        "disks",
        "describe",
        disk_name,
        f"--project={spec.project}",
        f"--zone={zone or spec.zone}",
        "--format=json",
    ]


REMOTE_INCOMPLETE_EXIT_CODES = frozenset((8, 9, 11, 12, 13, 19, 20))
REMOTE_COMPLETED_EXIT_CODE = 21


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _recorded_boot_disk_name(state: Mapping[str, Any]) -> str:
    self_link = _string(state.get("boot_disk_self_link"), "state.boot_disk_self_link")
    name = self_link.rstrip("/").split("/")[-1]
    if not name:
        raise HarnessError("recorded boot disk self-link has no disk name")
    return name


def _expected_ownership_labels(
    spec: ExperimentSpec, state: Mapping[str, Any]
) -> dict[str, str]:
    expected = {
        **spec.cloud["labels"],
        "ownership-nonce": _string(
            state.get("ownership_nonce"), "state.ownership_nonce"
        ),
    }
    recorded = state.get("labels")
    if recorded is not None:
        if not isinstance(recorded, dict) or recorded != expected:
            raise HarnessError(
                "REFUSING: recorded ownership labels differ from the pinned spec/nonce"
            )
    return expected


def _verify_abandonment_attachment(
    spec: ExperimentSpec,
    state: Mapping[str, Any],
    description: Mapping[str, Any],
) -> tuple[str, str]:
    """Bind the destructive request to the recorded VM and auto-delete disk."""
    recorded_instance_link = _string(
        state.get("instance_self_link"), "state.instance_self_link"
    )
    if description.get("selfLink") != recorded_instance_link:
        raise HarnessError(
            "REFUSING: live instance self-link differs from recorded state"
        )
    if description.get("labels") != _expected_ownership_labels(spec, state):
        raise HarnessError(
            "REFUSING: live instance labels differ from recorded ownership labels"
        )
    disks = description.get("disks", [])
    boot = [disk for disk in disks if isinstance(disk, dict) and disk.get("boot")]
    if len(boot) != 1:
        raise HarnessError("REFUSING: expected exactly one live boot disk attachment")
    recorded_disk_link = _string(
        state.get("boot_disk_self_link"), "state.boot_disk_self_link"
    )
    if boot[0].get("source") != recorded_disk_link:
        raise HarnessError(
            "REFUSING: live boot disk identity does not match recorded state"
        )
    if boot[0].get("autoDelete") is not True:
        raise HarnessError(
            "REFUSING: recorded boot disk is not attached with auto-delete"
        )
    return recorded_instance_link, recorded_disk_link


def abandonment_script(
    spec: ExperimentSpec,
    state: Mapping[str, Any],
    *,
    boot_disk_id: str,
    reason: str,
    requested_at_utc: str,
) -> tuple[str, dict[str, Any], str]:
    """Render the fail-closed remote stop, record, sync, and verify operation."""
    record = {
        "schema": "optimizer_harness_abandonment_v1",
        "run_id": spec.run_id,
        "reason": reason,
        "requested_at_utc": requested_at_utc,
        "instance_id": str(state["instance_id"]),
        "instance_self_link": state["instance_self_link"],
        "ownership_nonce": state["ownership_nonce"],
        "boot_disk_self_link": state["boot_disk_self_link"],
        "boot_disk_id": boot_disk_id,
        "labels": _expected_ownership_labels(spec, state),
        "artifact_uri": spec.artifact_uri,
        "state": "ABANDON_REQUESTED",
    }
    encoded_record = base64.b64encode(
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    ).decode()
    record_sha256 = hashlib.sha256(
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    completion = remote_completion_check_script(
        spec,
        state.get("additional_completion_paths", []),
        state.get("additional_checksum_manifests", []),
        require_runner_stopped=False,
    )
    encoded_completion = base64.b64encode(completion.encode()).decode()
    incomplete_codes = "|".join(
        str(code) for code in sorted(REMOTE_INCOMPLETE_EXIT_CODES)
    )
    run = shlex.quote(spec.remote_run_dir)
    artifact = shlex.quote(spec.artifact_uri)
    return (
        f"""set -euo pipefail
run={run}
artifact={artifact}
mkdir -p "$run"
check_file="$run/.abandon-completion-check.$$"
record_tmp="$run/.abandonment.json.tmp.$$"
trap 'rm -f "$check_file" "$record_tmp"' EXIT
printf %s {shlex.quote(encoded_completion)} | base64 -d > "$check_file"
set +e
bash "$check_file"
completion_rc=$?
set -e
case "$completion_rc" in
  0)
    echo 'REFUSING: run is complete; use delete' >&2
    exit {REMOTE_COMPLETED_EXIT_CODE}
    ;;
  {incomplete_codes})
    ;;
  *)
    echo "REFUSING: could not classify remote completion (exit $completion_rc)" >&2
    exit 22
    ;;
esac

collect_tree() {{
  local parent="$1" child
  for child in $(pgrep -P "$parent" 2>/dev/null || true); do
    collect_tree "$child"
  done
  printf '%s\n' "$parent"
}}

stop_pid_file() {{
  local file="$1" label="$2" pid cmdline tree live
  pid=$(cat "$file" 2>/dev/null || true)
  [ -n "$pid" ] || return 0
  case "$pid" in
    *[!0-9]*)
      echo "REFUSING: invalid $label pid file" >&2
      exit 16
      ;;
  esac
  kill -0 "$pid" 2>/dev/null || return 0
  [ -r "/proc/$pid/cmdline" ] || {{
    echo "REFUSING: cannot authenticate live $label pid $pid" >&2
    exit 16
  }}
  cmdline=$(tr '\\000' ' ' < "/proc/$pid/cmdline")
  case "$cmdline" in
    *"$run"*) ;;
    *)
      echo "REFUSING: live $label pid $pid is not bound to this run" >&2
      exit 16
      ;;
  esac
  tree=$(collect_tree "$pid")
  kill -TERM $tree 2>/dev/null || true
  for _i in $(seq 1 30); do
    live=''
    for _pid in $tree; do
      kill -0 "$_pid" 2>/dev/null && live="$live $_pid"
    done
    [ -z "$live" ] && return 0
    sleep 1
  done
  kill -KILL $live 2>/dev/null || true
  sleep 1
  for _pid in $live; do
    if kill -0 "$_pid" 2>/dev/null; then
      echo "REFUSING: $label process $_pid did not stop" >&2
      exit 17
    fi
  done
}}

stop_pid_file "$run/runner.pid" runner
stop_pid_file "$run/backup.pid" backup
printf %s {shlex.quote(encoded_record)} | base64 -d > "$record_tmp"
if [ -e "$run/abandonment.json" ]; then
  cmp -s "$record_tmp" "$run/abandonment.json" || {{
    echo 'REFUSING: conflicting remote abandonment record already exists' >&2
    exit 18
  }}
  rm -f "$record_tmp"
else
  mv "$record_tmp" "$run/abandonment.json"
fi
gcloud storage rsync --recursive "$run" "$artifact"
gcloud storage cat "$artifact/abandonment.json" | cmp - "$run/abandonment.json"
printf 'ABANDONMENT_RECORD_SHA256=%s\n' "$(sha256sum "$run/abandonment.json" | cut -d' ' -f1)"
""",
        record,
        record_sha256,
    )


def _explicit_not_found(result: subprocess.CompletedProcess[str]) -> bool:
    detail = f"{result.stdout or ''}\n{result.stderr or ''}"
    return (
        result.returncode != 0
        and re.search(
            r"(?:was\s+not\s+found|resource[^\n]*not\s+found|HTTPError\s+404)",
            detail,
            re.IGNORECASE,
        )
        is not None
    )


def _verify_absent(
    runner: CommandRunner,
    command: Sequence[str],
    resource: str,
    *,
    attempts: int = 60,
) -> None:
    for attempt in range(attempts):
        result = runner.run(command, check=False)
        if _explicit_not_found(result):
            return
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise HarnessError(
                f"could not verify {resource} absence"
                + (f": {detail}" if detail else "")
            )
        if attempt + 1 < attempts:
            time.sleep(1)
    raise HarnessError(f"{resource} still exists after delete")


def abandon(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
    *,
    exact_instance_id: str,
    reason: str,
    confirmed: bool,
) -> dict[str, Any]:
    """Preserve and delete one incomplete, exactly-owned experiment VM."""
    if not confirmed:
        raise HarnessError("abandon requires --yes")
    reason = _string(reason, "--reason").strip()
    state, description = require_live_owned_instance(spec, runner, state_dir)
    recorded_id = str(state["instance_id"])
    if exact_instance_id != recorded_id:
        raise HarnessError(
            f"REFUSING: --instance-id {exact_instance_id} != recorded exact id {recorded_id}"
        )
    if state.get("status") in ("DELETED", "ABANDONED"):
        raise HarnessError(f"run is already {state['status']}")
    original_state = json.loads(json.dumps(state))

    disk_name = _recorded_boot_disk_name(state)
    if runner.dry_run:
        instance_self_link = _string(
            state.get("instance_self_link"), "state.instance_self_link"
        )
        disk_self_link = _string(
            state.get("boot_disk_self_link"), "state.boot_disk_self_link"
        )
        disk_id = "DRY_RUN_DISK_ID"
    else:
        instance_self_link, disk_self_link = _verify_abandonment_attachment(
            spec, state, description
        )
        disk_result = runner.run(disk_describe_command(spec, disk_name))
        disk_description = _json_object_from_result(
            disk_result, "boot disk description"
        )
        recorded_disk_id = None
        if isinstance(state.get("boot_disk"), dict):
            recorded_disk_id = state["boot_disk"].get("id")
        disk_id = verify_disk_description(
            disk_description,
            expected_id=(
                str(recorded_disk_id) if recorded_disk_id is not None else None
            ),
            expected_self_link=disk_self_link,
            expected_user=instance_self_link,
        )

    existing_abandonment = state.get("abandonment")
    if existing_abandonment is not None:
        existing_abandonment = _object(existing_abandonment, "state.abandonment")
        if existing_abandonment.get("reason") != reason:
            raise HarnessError(
                "REFUSING: --reason differs from the recorded abandonment attempt"
            )
        if str(existing_abandonment.get("instance_id")) != recorded_id:
            raise HarnessError(
                "REFUSING: abandonment attempt instance id differs from recorded state"
            )
        if str(existing_abandonment.get("boot_disk_id")) != disk_id:
            raise HarnessError(
                "REFUSING: abandonment attempt boot disk id differs from the live disk"
            )
        requested_at = _string(
            existing_abandonment.get("requested_at_utc"),
            "state.abandonment.requested_at_utc",
        )
    else:
        requested_at = _utc_now()
    remote_script, record, record_sha256 = abandonment_script(
        spec,
        state,
        boot_disk_id=disk_id,
        reason=reason,
        requested_at_utc=requested_at,
    )

    if not runner.dry_run:
        completion_probe = runner.run(
            ssh_command(
                spec,
                remote_completion_check_script(
                    spec,
                    state.get("additional_completion_paths", []),
                    state.get("additional_checksum_manifests", []),
                    require_runner_stopped=False,
                ),
            ),
            check=False,
        )
        if completion_probe.returncode == 0:
            raise HarnessError("REFUSING: run is complete; use delete")
        if completion_probe.returncode not in REMOTE_INCOMPLETE_EXIT_CODES:
            detail = (completion_probe.stderr or completion_probe.stdout).strip()
            raise HarnessError(
                "could not safely classify run as incomplete; VM was not deleted"
                + (f": {detail}" if detail else "")
            )

        # Persist an idempotency key before the remote record is written.  A
        # failed GCS sync can then be retried without conflicting timestamps.
        state["status"] = "ABANDONING_PRESERVATION"
        state["abandonment"] = {
            **record,
            "record_sha256": record_sha256,
            "boot_disk_id": disk_id,
        }
        save_state(spec, state, state_dir)

    preservation = runner.run(ssh_command(spec, remote_script), check=False)
    if preservation.returncode == REMOTE_COMPLETED_EXIT_CODE:
        if not runner.dry_run:
            save_state(spec, original_state, state_dir)
        raise HarnessError("REFUSING: run is complete; use delete")
    if preservation.returncode != 0:
        detail = (preservation.stderr or preservation.stdout).strip()
        raise HarnessError(
            "abandonment preservation failed; VM was not deleted"
            + (f": {detail}" if detail else "")
        )
    if not runner.dry_run:
        match = re.search(
            r"^ABANDONMENT_RECORD_SHA256=([0-9a-f]{64})$",
            preservation.stdout,
            re.MULTILINE,
        )
        if match is None or match.group(1) != record_sha256:
            raise HarnessError(
                "abandonment preservation did not verify the exact remote record; "
                "VM was not deleted"
            )

        # Reauthenticate every mutable identity after the potentially long final sync.
        state_after_sync, description_after_sync = require_live_owned_instance(
            spec, runner, state_dir
        )
        if str(state_after_sync["instance_id"]) != recorded_id:
            raise HarnessError(
                "REFUSING: recorded instance id changed during abandonment"
            )
        current_instance_link, current_disk_link = _verify_abandonment_attachment(
            spec, state_after_sync, description_after_sync
        )
        disk_after_sync_result = runner.run(disk_describe_command(spec, disk_name))
        disk_after_sync = _json_object_from_result(
            disk_after_sync_result, "boot disk description"
        )
        verify_disk_description(
            disk_after_sync,
            expected_id=disk_id,
            expected_self_link=current_disk_link,
            expected_user=current_instance_link,
        )

        state["status"] = "ABANDONING"
        save_state(spec, state, state_dir)

    runner.run(delete_command(spec))
    if runner.dry_run:
        return {"dry_run": True, "reason": reason}

    _verify_absent(runner, describe_command(spec), "exact instance")
    _verify_absent(
        runner,
        disk_describe_command(spec, disk_name),
        "recorded auto-delete boot disk",
    )
    completed_at = _utc_now()
    state["status"] = "ABANDONED"
    state["abandon_reason"] = reason
    state["abandoned_at_utc"] = completed_at
    state["abandonment"]["completed_at_utc"] = completed_at
    save_state(spec, state, state_dir)
    return state


def image_create_command(
    spec: ExperimentSpec,
    disk_name: str,
    *,
    source_instance_id: str,
    source_disk_id: str,
    image_nonce: str,
) -> list[str]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    image = spec.image
    labels = {
        "managed-by": LABEL_OWNER,
        "image-nonce": image_nonce,
        "source-disk-id": source_disk_id,
        "source-instance-id": source_instance_id,
        "source-run": spec.run_id,
    }
    return _gcloud_prefix(spec) + [
        "images",
        "create",
        image["name"],
        f"--project={spec.project}",
        f"--source-disk={disk_name}",
        f"--source-disk-zone={spec.zone}",
        f"--storage-location={image['storage_location']}",
        "--labels="
        + ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
        f"--description=Sanitized Yeto optimizer image from {spec.run_id} at {spec.repo_commit}",
        "--format=json",
        "--quiet",
    ]


def stop_command(spec: ExperimentSpec) -> list[str]:
    return _gcloud_prefix(spec) + [
        "instances",
        "stop",
        spec.instance_name,
        f"--project={spec.project}",
        f"--zone={spec.zone}",
        "--quiet",
    ]


def sanitize_script(spec: ExperimentSpec) -> str:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    run = shlex.quote(spec.remote_run_dir)
    artifact = shlex.quote(spec.artifact_uri)
    paths = " ".join(shlex.quote(path) for path in spec.image["sanitize_paths"])
    manifest = {
        "schema_version": 1,
        "source_run": spec.run_id,
        "repo_url": spec.repo_url,
        "repo_commit": spec.repo_commit,
        "model_files_included": True,
        "huggingface_cache_included": False,
        "credentials_included": False,
        "run_artifacts_included": False,
        "model_checksum_manifest": "/etc/yeto-model-files.sha256",
        "model_symlink_manifest": "/etc/yeto-model-symlinks.txt",
        "data_checksum_manifest": "/etc/yeto-data.sha256",
        "runtime_manifest": "/etc/yeto-runtime.txt",
    }
    encoded = shlex.quote(
        base64.b64encode((json.dumps(manifest, indent=2) + "\n").encode()).decode()
    )
    return f"""set -eu
run={run}
runner=$(cat \"$run/runner.pid\" 2>/dev/null || true)
if [ -n \"$runner\" ] && kill -0 \"$runner\" 2>/dev/null; then
  echo 'REFUSING: experiment runner is still live' >&2
  exit 5
fi
root_source=$(findmnt -n -o SOURCE --target /)
for required in /home/shou/models/Qwen3.5-9B \
  /home/shou/data/Capybara-local/train.parquet /home/shou/venv /home/shou/yeto; do
  test -e \"$required\" || {{ echo \"missing image input: $required\" >&2; exit 21; }}
  test \"$(findmnt -n -o SOURCE --target \"$required\")\" = \"$root_source\" || {{
    echo \"image input is not on the boot filesystem: $required\" >&2
    exit 22
  }}
done
if find -L /home/shou/models/Qwen3.5-9B -type l -print -quit | grep -q .; then
  echo 'model tree contains a dangling symlink' >&2
  exit 23
fi
find /home/shou/models/Qwen3.5-9B -type l -print0 | while IFS= read -r -d '' link; do
  target=$(readlink -f \"$link\")
  case \"$target\" in
    /home/shou/models/Qwen3.5-9B/*) ;;
    *) echo \"model symlink escapes model root: $link -> $target\" >&2; exit 24 ;;
  esac
done
gcloud storage rsync --recursive \"$run\" {artifact}
backup=$(cat \"$run/backup.pid\" 2>/dev/null || true)
if [ -n \"$backup\" ] && kill -0 \"$backup\" 2>/dev/null; then
  kill -TERM \"$backup\"
  for _i in $(seq 1 30); do
    kill -0 \"$backup\" 2>/dev/null || break
    sleep 1
  done
  kill -0 \"$backup\" 2>/dev/null && kill -KILL \"$backup\"
fi
for path in {paths}; do
  sudo rm -rf -- \"$path\"
done
find /home/shou/models/Qwen3.5-9B -type f -print0 | sort -z | \
  xargs -0 sha256sum | sudo tee /etc/yeto-model-files.sha256 >/dev/null
find /home/shou/models/Qwen3.5-9B -type l -printf '%P -> %l\\n' | sort | \
  sudo tee /etc/yeto-model-symlinks.txt >/dev/null
sha256sum /home/shou/data/Capybara-local/train.parquet | \
  sudo tee /etc/yeto-data.sha256 >/dev/null
{{
  /home/shou/venv/bin/python -c 'import torch, transformers; print("torch=" + torch.__version__); print("transformers=" + transformers.__version__); print("cuda=" + str(torch.version.cuda))'
  /home/shou/.cargo/bin/rustc --version
  /home/shou/.cargo/bin/cargo --version
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1
  git -C /home/shou/yeto rev-parse HEAD
}} | sudo tee /etc/yeto-runtime.txt >/dev/null
sudo find /tmp /var/tmp -depth -mindepth 1 -delete
sudo journalctl --rotate >/dev/null 2>&1 || true
sudo journalctl --vacuum-time=1s >/dev/null 2>&1 || true
if command -v cloud-init >/dev/null 2>&1; then sudo cloud-init clean --logs; fi
suspects=$(sudo find /home/shou /root -xdev \
  \\( -name .netrc -o -name .git-credentials -o -name credentials \
     -o -name application_default_credentials.json -o -name access_tokens.db \
     -o -name credentials.db -o -name stored_tokens -o -name hosts.yml \
     -o -name .npmrc -o -name .pypirc -o -name id_rsa -o -name id_ed25519 \
     -o -name token \\) -print)
if [ -n \"$suspects\" ]; then
  echo 'REFUSING: credential-like files remain after image sanitation:' >&2
  printf '%s\\n' \"$suspects\" >&2
  exit 10
fi
printf %s {encoded} | base64 -d | sudo tee /etc/yeto-optimizer-image.json >/dev/null
sudo rm -f /var/lib/dbus/machine-id
sudo truncate -s 0 /etc/machine-id
sudo sync
printf 'IMAGE_MANIFEST_SHA256=%s\\n' "$(sha256sum /etc/yeto-optimizer-image.json | cut -d' ' -f1)"
"""


def verify_disk_description(
    description: Mapping[str, Any],
    *,
    expected_id: str | None,
    expected_self_link: str,
    expected_user: str,
) -> str:
    disk_id = _string(str(description.get("id", "")), "disk.id")
    if expected_id is not None and disk_id != str(expected_id):
        raise HarnessError(
            f"REFUSING: live boot disk id {disk_id} != recorded exact id {expected_id}"
        )
    if description.get("selfLink") != expected_self_link:
        raise HarnessError(
            "REFUSING: live boot disk self-link differs from the recorded disk"
        )
    users = description.get("users", [])
    if users != [expected_user]:
        raise HarnessError(
            "REFUSING: boot disk users do not identify only the source instance"
        )
    return disk_id


def verify_candidate_image_description(
    spec: ExperimentSpec,
    description: Mapping[str, Any],
    *,
    source_instance_id: str,
    source_disk_id: str,
    source_disk_self_link: str,
    image_nonce: str,
) -> str:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    image_id = _string(str(description.get("id", "")), "image.id")
    if description.get("name") != spec.image["name"]:
        raise HarnessError("created image name differs from the recipe")
    self_link = str(description.get("selfLink", ""))
    if f"/projects/{spec.project}/" not in self_link or not self_link.endswith(
        f"/global/images/{spec.image['name']}"
    ):
        raise HarnessError(
            "created image project/name identity differs from the recipe"
        )
    if description.get("status") != "READY":
        raise HarnessError("created image is not READY")
    if description.get("sourceDisk") != source_disk_self_link:
        raise HarnessError("created image sourceDisk differs from the exact boot disk")
    if str(description.get("sourceDiskId", "")) != source_disk_id:
        raise HarnessError(
            "created image sourceDiskId differs from the exact boot disk id"
        )
    labels = description.get("labels", {})
    expected_labels = {
        "managed-by": LABEL_OWNER,
        "image-nonce": image_nonce,
        "source-disk-id": source_disk_id,
        "source-instance-id": source_instance_id,
        "source-run": spec.run_id,
    }
    for key, value in expected_labels.items():
        if labels.get(key) != value:
            raise HarnessError(f"created image label {key!r} does not match")
    if description.get("family"):
        raise HarnessError("candidate image was prematurely assigned to a family")
    return image_id


def create_image(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
    *,
    exact_instance_id: str,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise HarnessError("create-image requires --yes")
    state, description = require_live_owned_instance(spec, runner, state_dir)
    recorded = str(state["instance_id"])
    if exact_instance_id != recorded:
        raise HarnessError(
            f"REFUSING: --instance-id {exact_instance_id} != recorded exact id {recorded}"
        )
    if spec.image is None:
        raise HarnessError("spec has no image section")

    def persist_state() -> None:
        if not runner.dry_run:
            save_state(spec, state, state_dir)

    collision = runner.run(
        [
            "gcloud",
            "compute",
            "images",
            "describe",
            spec.image["name"],
            f"--project={spec.project}",
            "--format=json",
        ],
        check=False,
    )
    if not runner.dry_run and collision.returncode == 0:
        raise HarnessError(
            f"output image already exists: {spec.image['name']}; refusing to sanitize/stop"
        )
    assert_remote_complete(
        spec,
        runner,
        state.get("additional_completion_paths", []),
        state.get("additional_checksum_manifests", []),
    )
    boot_disk_self_link = _boot_disk_self_link(description)
    disk_name = boot_disk_self_link.rstrip("/").split("/")[-1]
    if not disk_name:
        raise HarnessError("could not determine boot disk name")
    source_instance_self_link = _string(
        description.get("selfLink"), "source instance selfLink"
    )
    disk_before_result = runner.run(disk_describe_command(spec, disk_name))
    if runner.dry_run:
        source_disk_id = "DRY_RUN_DISK_ID"
    else:
        disk_before = _json_object_from_result(
            disk_before_result, "boot disk description"
        )
        source_disk_id = verify_disk_description(
            disk_before,
            expected_id=None,
            expected_self_link=boot_disk_self_link,
            expected_user=source_instance_self_link,
        )
    state["boot_disk"] = {
        "name": disk_name,
        "id": source_disk_id,
        "self_link": boot_disk_self_link,
        "user": source_instance_self_link,
    }
    state["status"] = "SANITIZING"
    persist_state()
    sanitize_result = runner.run(ssh_command(spec, sanitize_script(spec)))
    if runner.dry_run:
        state["image_manifest_sha256"] = "DRY_RUN"
    else:
        match = re.search(
            r"^IMAGE_MANIFEST_SHA256=([0-9a-f]{64})$", sanitize_result.stdout, re.M
        )
        if match is None:
            raise HarnessError("sanitizer did not return the image manifest checksum")
        state["image_manifest_sha256"] = match.group(1)
    state["status"] = "SANITIZED"
    persist_state()
    # A custom disk image from a stopped VM is filesystem-consistent and does
    # not need gcloud's unsafe --force path for an attached, running disk.
    state["status"] = "SOURCE_STOPPING"
    persist_state()
    runner.run(stop_command(spec))
    if not runner.dry_run:
        stopped = describe(spec, runner)
        verify_description(
            spec,
            stopped,
            expected_id=recorded,
            expected_nonce=str(state["ownership_nonce"]),
        )
        if stopped.get("status") != "TERMINATED":
            raise HarnessError(
                f"instance did not reach TERMINATED before imaging: {stopped.get('status')}"
            )
        disk_after_result = runner.run(disk_describe_command(spec, disk_name))
        disk_after = _json_object_from_result(
            disk_after_result, "stopped boot disk description"
        )
        verify_disk_description(
            disk_after,
            expected_id=source_disk_id,
            expected_self_link=boot_disk_self_link,
            expected_user=source_instance_self_link,
        )
    state["status"] = "SOURCE_STOPPED"
    persist_state()
    image_nonce = secrets.token_hex(8)
    state["image"] = {
        "name": spec.image["name"],
        "family": spec.image["family"],
        "image_nonce": image_nonce,
        "source_instance_id": recorded,
        "source_disk_id": source_disk_id,
        "source_disk_self_link": boot_disk_self_link,
        "family_promoted": False,
        "status": "CREATING",
    }
    state["status"] = "IMAGE_CREATING"
    persist_state()
    runner.run(
        image_create_command(
            spec,
            disk_name,
            source_instance_id=recorded,
            source_disk_id=source_disk_id,
            image_nonce=image_nonce,
        )
    )
    if runner.dry_run:
        return {"dry_run": True}
    image_result = runner.run(image_describe_command(spec))
    image_description = _json_object_from_result(image_result, "image description")
    image_id = verify_candidate_image_description(
        spec,
        image_description,
        source_instance_id=recorded,
        source_disk_id=source_disk_id,
        source_disk_self_link=boot_disk_self_link,
        image_nonce=image_nonce,
    )
    state["image"].update(
        {
            "id": image_id,
            "self_link": image_description.get("selfLink"),
            "status": "READY",
        }
    )
    state["status"] = "IMAGE_CANDIDATE_READY"
    save_state(spec, state, state_dir)
    return state["image"]


def image_describe_command(spec: ExperimentSpec) -> list[str]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    return _gcloud_prefix(spec) + [
        "images",
        "describe",
        spec.image["name"],
        f"--project={spec.project}",
        "--format=json",
    ]


def canary_describe_command(spec: ExperimentSpec) -> list[str]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    return _gcloud_prefix(spec) + [
        "instances",
        "describe",
        spec.image["canary_name"],
        f"--project={spec.project}",
        f"--zone={spec.image['canary_zone']}",
        "--format=json",
    ]


def canary_launch_command(
    spec: ExperimentSpec, ownership_nonce: str, source_image_id: str
) -> list[str]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    if not re.fullmatch(r"[a-f0-9]{16}", ownership_nonce):
        raise HarnessError(
            "ownership nonce must be 16 lowercase hexadecimal characters"
        )
    labels = {
        "managed-by": LABEL_OWNER,
        "ownership-nonce": ownership_nonce,
        "role": "image-canary",
        "source-image-id": source_image_id,
        "source-run": spec.run_id,
    }
    image_ref = f"projects/{spec.project}/global/images/{spec.image['name']}"
    command = _gcloud_prefix(spec) + [
        "instances",
        "create",
        spec.image["canary_name"],
        f"--project={spec.project}",
        f"--zone={spec.image['canary_zone']}",
        f"--machine-type={spec.image['canary_machine_type']}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        "--metadata=block-project-ssh-keys=true",
        "--boot-disk-auto-delete",
        f"--boot-disk-size={spec.cloud['boot_disk_size_gb']}GB",
        f"--scopes={','.join(spec.cloud['scopes'])}",
        "--labels="
        + ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
        f"--image={image_ref}",
        "--format=json",
        "--quiet",
    ]
    if spec.cloud.get("network"):
        command.append(f"--network={spec.cloud['network']}")
    if spec.cloud.get("subnet"):
        command.append(f"--subnet={spec.cloud['subnet']}")
    return command


def canary_ssh_command(spec: ExperimentSpec, remote_command: str) -> list[str]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    return _gcloud_prefix(spec) + [
        "ssh",
        spec.image["canary_name"],
        f"--project={spec.project}",
        f"--zone={spec.image['canary_zone']}",
        "--ssh-flag=-o ConnectTimeout=10",
        f"--command={remote_command}",
    ]


def canary_delete_command(spec: ExperimentSpec) -> list[str]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    return _gcloud_prefix(spec) + [
        "instances",
        "delete",
        spec.image["canary_name"],
        f"--project={spec.project}",
        f"--zone={spec.image['canary_zone']}",
        "--quiet",
    ]


def verify_canary_description(
    spec: ExperimentSpec,
    description: Mapping[str, Any],
    *,
    expected_id: str | None,
    expected_nonce: str,
    expected_image_id: str,
) -> str:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    name = spec.image["canary_name"]
    zone = spec.image["canary_zone"]
    if description.get("name") != name:
        raise HarnessError("live canary name does not match the image recipe")
    instance_id = _instance_id(description)
    if expected_id is not None and instance_id != str(expected_id):
        raise HarnessError(
            f"REFUSING: live canary id {instance_id} != recorded exact id {expected_id}"
        )
    self_link = str(description.get("selfLink", ""))
    if f"/projects/{spec.project}/" not in self_link or not self_link.endswith(
        f"/zones/{zone}/instances/{name}"
    ):
        raise HarnessError("live canary project/zone identity does not match")
    labels = description.get("labels", {})
    expected_labels = {
        "managed-by": LABEL_OWNER,
        "ownership-nonce": expected_nonce,
        "role": "image-canary",
        "source-image-id": expected_image_id,
        "source-run": spec.run_id,
    }
    for key, value in expected_labels.items():
        if labels.get(key) != value:
            raise HarnessError(f"REFUSING: live canary label {key!r} does not match")
    scheduling = description.get("scheduling", {})
    if scheduling.get("provisioningModel") != "SPOT":
        raise HarnessError("live canary is not Spot")
    if scheduling.get("instanceTerminationAction") != "DELETE":
        raise HarnessError("live canary termination action is not DELETE")
    disk = _boot_disk_self_link(description)
    if not disk.endswith(f"/zones/{zone}/disks/{name}"):
        raise HarnessError("live canary boot disk identity does not match")
    boot = [item for item in description.get("disks", []) if item.get("boot")]
    if len(boot) != 1 or boot[0].get("autoDelete") is not True:
        raise HarnessError(
            "live canary boot disk is not uniquely identified and auto-deleting"
        )
    return instance_id


def verify_canary_disk_description(
    description: Mapping[str, Any],
    *,
    expected_id: str | None,
    expected_self_link: str,
    expected_user: str,
    expected_image_id: str,
    expected_image_self_link: str,
) -> str:
    disk_id = verify_disk_description(
        description,
        expected_id=expected_id,
        expected_self_link=expected_self_link,
        expected_user=expected_user,
    )
    if str(description.get("sourceImageId", "")) != expected_image_id:
        raise HarnessError(
            "canary disk sourceImageId differs from the exact candidate image"
        )
    if description.get("sourceImage") != expected_image_self_link:
        raise HarnessError(
            "canary disk sourceImage differs from the exact candidate image"
        )
    return disk_id


def _json_object_from_result(
    result: subprocess.CompletedProcess[str], where: str
) -> dict[str, Any]:
    try:
        return _object(json.loads(result.stdout), where)
    except json.JSONDecodeError as exc:
        raise HarnessError(f"gcloud returned invalid {where} JSON: {exc}") from exc


def require_exact_created_image(
    spec: ExperimentSpec, state: Mapping[str, Any], runner: CommandRunner
) -> dict[str, Any]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    recorded = _object(state.get("image"), "state.image")
    if recorded.get("name") != spec.image["name"] or not recorded.get("id"):
        raise HarnessError("state has no exact created-image identity")
    result = runner.run(image_describe_command(spec))
    if runner.dry_run:
        return {}
    live = _json_object_from_result(result, "image description")
    if str(live.get("id", "")) != str(recorded["id"]):
        raise HarnessError(
            "REFUSING: live image id differs from recorded exact image id"
        )
    if live.get("name") != recorded["name"] or live.get("status") != "READY":
        raise HarnessError("recorded image is not the live READY image")
    return live


def create_canary(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
) -> dict[str, Any]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    state = load_state(spec, state_dir)
    require_exact_created_image(spec, state, runner)
    source_image_id = str(state["image"]["id"])
    source_image_self_link = _string(
        state["image"].get("self_link"), "state.image.self_link"
    )
    existing_state = state.get("canary")
    if existing_state is not None:
        existing_state = _object(existing_state, "state.canary")
        if existing_state.get("status") not in ("LAUNCHING", "GONE", "DELETED"):
            raise HarnessError("a live or completed canary is already recorded")
    if existing_state and existing_state.get("status") == "LAUNCHING":
        nonce = _string(
            existing_state.get("ownership_nonce"), "state.canary.ownership_nonce"
        )
    else:
        nonce = secrets.token_hex(8)
        state["canary"] = {
            "name": spec.image["canary_name"],
            "zone": spec.image["canary_zone"],
            "image_id": source_image_id,
            "ownership_nonce": nonce,
            "status": "LAUNCHING",
        }
        save_state(spec, state, state_dir)

    probe = runner.run(canary_describe_command(spec), check=False)
    if runner.dry_run:
        runner.run(canary_launch_command(spec, nonce, source_image_id))
        return {"dry_run": True}
    if probe.returncode == 0:
        description = _json_object_from_result(probe, "canary description")
        instance_id = verify_canary_description(
            spec,
            description,
            expected_id=None,
            expected_nonce=nonce,
            expected_image_id=source_image_id,
        )
    else:
        runner.run(canary_launch_command(spec, nonce, source_image_id))
        result = runner.run(canary_describe_command(spec))
        description = _json_object_from_result(result, "canary description")
        instance_id = verify_canary_description(
            spec,
            description,
            expected_id=None,
            expected_nonce=nonce,
            expected_image_id=source_image_id,
        )
    canary_boot_self_link = _boot_disk_self_link(description)
    canary_disk_name = canary_boot_self_link.rstrip("/").split("/")[-1]
    disk_result = runner.run(
        disk_describe_command(spec, canary_disk_name, zone=spec.image["canary_zone"])
    )
    disk_description = _json_object_from_result(disk_result, "canary disk description")
    canary_disk_id = verify_canary_disk_description(
        disk_description,
        expected_id=None,
        expected_self_link=canary_boot_self_link,
        expected_user=_string(description.get("selfLink"), "canary selfLink"),
        expected_image_id=source_image_id,
        expected_image_self_link=source_image_self_link,
    )
    state["canary"].update(
        {
            "id": instance_id,
            "self_link": description.get("selfLink"),
            "boot_disk_id": canary_disk_id,
            "boot_disk_self_link": canary_boot_self_link,
            "status": description.get("status"),
        }
    )
    state["status"] = "IMAGE_CANARY_CREATED"
    save_state(spec, state, state_dir)
    return dict(state["canary"])


def canary_verification_script(
    spec: ExperimentSpec, canary_id: str, source_image_id: str
) -> str:
    artifact = shlex.quote(spec.artifact_uri)
    repo = shlex.quote(spec.remote_repo_dir)
    commit = shlex.quote(spec.repo_commit)
    object_uri = shlex.quote(
        f"{spec.artifact_uri}/image-canary/{source_image_id}/{canary_id}/smoke.txt"
    )
    return f"""set -eu
test "$(git -C {repo} rev-parse HEAD)" = {commit}
test -s /etc/yeto-model-files.sha256
test -e /etc/yeto-model-symlinks.txt
test -s /etc/yeto-data.sha256
test -s /etc/yeto-runtime.txt
test -s /etc/yeto-optimizer-image.json
test ! -e {shlex.quote(spec.remote_run_dir)}
test -s /etc/machine-id
test -n "$(tr -d '[:space:]' < /etc/machine-id)"
suspects=$(sudo find /home/shou /root -xdev \
  \\( -name .netrc -o -name .git-credentials -o -name credentials \
     -o -name application_default_credentials.json -o -name access_tokens.db \
     -o -name credentials.db -o -name stored_tokens -o -name hosts.yml \
     -o -name .npmrc -o -name .pypirc -o -name id_rsa -o -name id_ed25519 \
     -o -name token \\) -print)
test -z "$suspects" || {{ printf '%s\n' "$suspects" >&2; exit 31; }}
sudo sha256sum -c /etc/yeto-model-files.sha256 >/dev/null
sudo sha256sum -c /etc/yeto-data.sha256 >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'A100'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/shou/venv/bin/python - <<'PY'
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root = "/home/shou/models/Qwen3.5-9B"
manifest = json.load(open("/etc/yeto-optimizer-image.json"))
assert manifest["credentials_included"] is False
assert manifest["run_artifacts_included"] is False
assert manifest["repo_commit"] == {spec.repo_commit!r}
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    root,
    dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True,
).to("cuda").eval()
inputs = tokenizer("image canary", return_tensors="pt")["input_ids"].to("cuda")
with torch.no_grad():
    logits = model(input_ids=inputs).logits
assert logits.shape[:2] == inputs.shape
print(
    "offline_forward_ok",
    tuple(logits.shape),
    torch.__version__,
    "peak_bytes",
    torch.cuda.max_memory_allocated(),
)
PY
cd {repo}
/home/shou/venv/bin/python -m pytest -q tests/test_learner_units.py tests/test_compare_action_probe.py
/home/shou/.cargo/bin/cargo test --release --manifest-path syncer/Cargo.toml
printf '%s\n' {shlex.quote("canary-" + canary_id)} > /tmp/yeto-image-canary-smoke
gcloud storage cp /tmp/yeto-image-canary-smoke {object_uri} >/dev/null
gcloud storage cp {object_uri} /tmp/yeto-image-canary-smoke.download >/dev/null
cmp /tmp/yeto-image-canary-smoke /tmp/yeto-image-canary-smoke.download
gcloud storage rm {object_uri} >/dev/null
printf '%s\n' "IMAGE_CANARY_OK image={spec.image["name"] if spec.image else ""} image_id={source_image_id} canary_id={canary_id} artifact={artifact}"
"""


def test_canary(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
    *,
    exact_canary_id: str,
) -> str:
    state = load_state(spec, state_dir)
    require_exact_created_image(spec, state, runner)
    canary = _object(state.get("canary"), "state.canary")
    recorded = _string(str(canary.get("id", "")), "state.canary.id")
    if exact_canary_id != recorded:
        raise HarnessError(
            f"REFUSING: --canary-id {exact_canary_id} != recorded exact id {recorded}"
        )
    result = runner.run(canary_describe_command(spec), check=False)
    if not runner.dry_run and result.returncode:
        canary["status"] = "INCONCLUSIVE"
        state["status"] = "IMAGE_CANARY_INCONCLUSIVE"
        save_state(spec, state, state_dir)
        raise HarnessError("recorded canary is absent or preempted before verification")
    if not runner.dry_run:
        description = _json_object_from_result(result, "canary description")
        verify_canary_description(
            spec,
            description,
            expected_id=recorded,
            expected_nonce=_string(
                canary.get("ownership_nonce"), "state.canary.ownership_nonce"
            ),
            expected_image_id=_string(
                str(canary.get("image_id", "")), "state.canary.image_id"
            ),
        )
        boot_self_link = _string(
            canary.get("boot_disk_self_link"), "state.canary.boot_disk_self_link"
        )
        disk_result = runner.run(
            disk_describe_command(
                spec,
                boot_self_link.rstrip("/").split("/")[-1],
                zone=spec.image["canary_zone"],
            )
        )
        verify_canary_disk_description(
            _json_object_from_result(disk_result, "canary disk description"),
            expected_id=_string(
                str(canary.get("boot_disk_id", "")), "state.canary.boot_disk_id"
            ),
            expected_self_link=boot_self_link,
            expected_user=_string(canary.get("self_link"), "state.canary.self_link"),
            expected_image_id=_string(
                str(canary.get("image_id", "")), "state.canary.image_id"
            ),
            expected_image_self_link=_string(
                state["image"].get("self_link"), "state.image.self_link"
            ),
        )
    source_image_id = _string(str(canary.get("image_id", "")), "state.canary.image_id")
    script = canary_verification_script(spec, recorded, source_image_id)
    ready: subprocess.CompletedProcess[str] | None = None
    for attempt in range(12):
        ready = runner.run(canary_ssh_command(spec, "true"), check=False)
        if runner.dry_run or ready.returncode == 0:
            break
        if attempt != 11:
            time.sleep(10)
    if ready is None or ready.returncode:
        detail = "" if ready is None else (ready.stderr or ready.stdout).strip()
        if not runner.dry_run:
            canary["status"] = "FAILED"
            state["status"] = "IMAGE_CANARY_FAILED"
            save_state(spec, state, state_dir)
        raise HarnessError(
            "image canary SSH readiness failed" + (f"\n{detail}" if detail else "")
        )
    last = runner.run(canary_ssh_command(spec, script), check=False)
    if last.returncode:
        detail = (last.stderr or last.stdout).strip()
        if not runner.dry_run:
            canary["status"] = "FAILED"
            state["status"] = "IMAGE_CANARY_FAILED"
            save_state(spec, state, state_dir)
        raise HarnessError(
            "image canary verification failed" + (f"\n{detail}" if detail else "")
        )
    if not runner.dry_run:
        canary["status"] = "PASSED"
        canary["verification_output_sha256"] = hashlib.sha256(
            (last.stdout + last.stderr).encode()
        ).hexdigest()
        state["status"] = "IMAGE_CANARY_PASSED"
        save_state(spec, state, state_dir)
    return last.stdout


def image_promote_command(spec: ExperimentSpec) -> list[str]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    return _gcloud_prefix(spec) + [
        "images",
        "update",
        spec.image["name"],
        f"--project={spec.project}",
        f"--family={spec.image['family']}",
        "--update-labels=canary-status=passed",
        "--quiet",
    ]


def image_from_family_command(spec: ExperimentSpec) -> list[str]:
    if spec.image is None:
        raise HarnessError("spec has no image section")
    return _gcloud_prefix(spec) + [
        "images",
        "describe-from-family",
        spec.image["family"],
        f"--project={spec.project}",
        "--format=json",
    ]


def promote_image(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
    *,
    exact_canary_id: str,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise HarnessError("promote-image requires --yes")
    if spec.image is None:
        raise HarnessError("spec has no image section")
    state = load_state(spec, state_dir)
    image = _object(state.get("image"), "state.image")
    canary = _object(state.get("canary"), "state.canary")
    if canary.get("status") != "PASSED":
        raise HarnessError("image promotion requires a stored PASSED canary result")
    recorded_canary_id = _string(str(canary.get("id", "")), "state.canary.id")
    if exact_canary_id != recorded_canary_id:
        raise HarnessError(
            f"REFUSING: --canary-id {exact_canary_id} != recorded exact id {recorded_canary_id}"
        )
    live_image = require_exact_created_image(spec, state, runner)
    if not runner.dry_run:
        verify_candidate_image_description(
            spec,
            live_image,
            source_instance_id=_string(
                str(image.get("source_instance_id", "")),
                "state.image.source_instance_id",
            ),
            source_disk_id=_string(
                str(image.get("source_disk_id", "")), "state.image.source_disk_id"
            ),
            source_disk_self_link=_string(
                image.get("source_disk_self_link"), "state.image.source_disk_self_link"
            ),
            image_nonce=_string(image.get("image_nonce"), "state.image.image_nonce"),
        )
        canary_result = runner.run(canary_describe_command(spec))
        canary_description = _json_object_from_result(
            canary_result, "canary description"
        )
        verify_canary_description(
            spec,
            canary_description,
            expected_id=recorded_canary_id,
            expected_nonce=_string(
                canary.get("ownership_nonce"), "state.canary.ownership_nonce"
            ),
            expected_image_id=_string(str(image.get("id", "")), "state.image.id"),
        )
        disk_result = runner.run(
            disk_describe_command(
                spec,
                _string(
                    canary.get("boot_disk_self_link"),
                    "state.canary.boot_disk_self_link",
                )
                .rstrip("/")
                .split("/")[-1],
                zone=spec.image["canary_zone"],
            )
        )
        verify_canary_disk_description(
            _json_object_from_result(disk_result, "canary disk description"),
            expected_id=_string(
                str(canary.get("boot_disk_id", "")), "state.canary.boot_disk_id"
            ),
            expected_self_link=_string(
                canary.get("boot_disk_self_link"), "state.canary.boot_disk_self_link"
            ),
            expected_user=_string(canary.get("self_link"), "state.canary.self_link"),
            expected_image_id=_string(str(image.get("id", "")), "state.image.id"),
            expected_image_self_link=_string(
                image.get("self_link"), "state.image.self_link"
            ),
        )
    runner.run(image_promote_command(spec))
    if runner.dry_run:
        return {"dry_run": True}
    promoted_result = runner.run(image_describe_command(spec))
    promoted = _json_object_from_result(promoted_result, "promoted image description")
    if str(promoted.get("id", "")) != str(image["id"]):
        raise HarnessError("promoted image id differs from the recorded exact image")
    family = str(promoted.get("family", ""))
    if family != spec.image["family"] and not family.endswith(
        f"/images/{spec.image['family']}"
    ):
        raise HarnessError("image did not enter the requested family")
    if promoted.get("labels", {}).get("canary-status") != "passed":
        raise HarnessError("promoted image lacks the canary-status=passed label")
    family_result = runner.run(image_from_family_command(spec))
    family_image = _json_object_from_result(family_result, "image family description")
    if str(family_image.get("id", "")) != str(image["id"]):
        raise HarnessError(
            "image family does not resolve to the verified exact image id"
        )
    image["family_promoted"] = True
    image["status"] = "PROMOTED"
    state["status"] = "IMAGE_PROMOTED"
    save_state(spec, state, state_dir)
    return dict(image)


def delete_canary(
    spec: ExperimentSpec,
    runner: CommandRunner,
    state_dir: str | Path | None,
    *,
    exact_canary_id: str,
    confirmed: bool,
) -> None:
    if not confirmed:
        raise HarnessError("delete-canary requires --yes")
    state = load_state(spec, state_dir)
    canary = _object(state.get("canary"), "state.canary")
    recorded = _string(str(canary.get("id", "")), "state.canary.id")
    if exact_canary_id != recorded:
        raise HarnessError(
            f"REFUSING: --canary-id {exact_canary_id} != recorded exact id {recorded}"
        )
    probe = runner.run(canary_describe_command(spec), check=False)
    if runner.dry_run:
        runner.run(canary_delete_command(spec))
        return
    if probe.returncode != 0:
        canary["status"] = "GONE"
        state["status"] = "IMAGE_CANARY_GONE"
        save_state(spec, state, state_dir)
        return
    description = _json_object_from_result(probe, "canary description")
    verify_canary_description(
        spec,
        description,
        expected_id=recorded,
        expected_nonce=_string(
            canary.get("ownership_nonce"), "state.canary.ownership_nonce"
        ),
        expected_image_id=_string(
            str(canary.get("image_id", "")), "state.canary.image_id"
        ),
    )
    boot_self_link = _string(
        canary.get("boot_disk_self_link"), "state.canary.boot_disk_self_link"
    )
    disk_name = boot_self_link.rstrip("/").split("/")[-1]
    disk_probe = runner.run(
        disk_describe_command(spec, disk_name, zone=spec.image["canary_zone"])
    )
    disk_description = _json_object_from_result(disk_probe, "canary disk description")
    verify_canary_disk_description(
        disk_description,
        expected_id=_string(
            str(canary.get("boot_disk_id", "")), "state.canary.boot_disk_id"
        ),
        expected_self_link=boot_self_link,
        expected_user=_string(canary.get("self_link"), "state.canary.self_link"),
        expected_image_id=_string(
            str(canary.get("image_id", "")), "state.canary.image_id"
        ),
        expected_image_self_link=_string(
            state["image"].get("self_link"), "state.image.self_link"
        ),
    )
    runner.run(canary_delete_command(spec))
    after = runner.run(canary_describe_command(spec), check=False)
    if after.returncode == 0:
        raise HarnessError("delete returned but the exact canary still exists")
    disk_after = runner.run(
        disk_describe_command(spec, disk_name, zone=spec.image["canary_zone"]),
        check=False,
    )
    if disk_after.returncode == 0:
        raise HarnessError(
            "canary was deleted but its recorded auto-delete boot disk remains; leaving it untouched"
        )
    canary["status"] = "DELETED"
    state["status"] = "IMAGE_CANARY_DELETED"
    save_state(spec, state, state_dir)


def validate_cloud(spec: ExperimentSpec, runner: CommandRunner) -> None:
    runner.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"]
    )
    bucket = "gs://" + spec.artifact_uri.removeprefix("gs://").split("/", 1)[0]
    runner.run(["gcloud", "storage", "ls", bucket])
    source = spec.cloud.get("image") or spec.cloud.get("machine_image")
    if spec.cloud.get("image"):
        image_project = spec.project
        image_name = source
        match = re.fullmatch(r"projects/([^/]+)/global/images/(.+)", source)
        if match:
            image_project, image_name = match.groups()
        runner.run(
            [
                "gcloud",
                "compute",
                "images",
                "describe",
                image_name,
                f"--project={image_project}",
            ]
        )
    else:
        runner.run(
            [
                "gcloud",
                "compute",
                "machine-images",
                "describe",
                source,
                f"--project={spec.project}",
            ]
        )
    validate_compute_quota(spec, runner)


def validate_compute_quota(spec: ExperimentSpec, runner: CommandRunner) -> None:
    """Fail before launch when the requested machine cannot fit regional quota."""
    machine_result = runner.run(
        [
            "gcloud",
            "compute",
            "machine-types",
            "describe",
            spec.cloud["machine_type"],
            f"--project={spec.project}",
            f"--zone={spec.zone}",
            "--format=json",
        ]
    )
    region = spec.zone.rsplit("-", 1)[0]
    region_result = runner.run(
        [
            "gcloud",
            "compute",
            "regions",
            "describe",
            region,
            f"--project={spec.project}",
            "--format=json",
        ]
    )
    if runner.dry_run:
        return
    machine = _json_object_from_result(machine_result, "machine type description")
    region_description = _json_object_from_result(region_result, "region description")
    guest_cpus = machine.get("guestCpus")
    if type(guest_cpus) not in (int, float) or guest_cpus <= 0:
        raise HarnessError("machine type description has no positive guestCpus")
    accelerator_total = 0
    for accelerator in machine.get("accelerators", []):
        if not isinstance(accelerator, dict):
            raise HarnessError("machine type accelerator description is malformed")
        count = accelerator.get("guestAcceleratorCount")
        if type(count) not in (int, float) or count < 0:
            raise HarnessError("machine type accelerator count is malformed")
        accelerator_total += int(count)
    requested_accelerators = int(spec.cloud["accelerator_count"])
    if accelerator_total != requested_accelerators:
        raise HarnessError(
            "machine type accelerator count differs from the spec: "
            f"machine={accelerator_total} spec={requested_accelerators}"
        )
    quota_rows = region_description.get("quotas")
    if not isinstance(quota_rows, list):
        raise HarnessError("region description has no quota list")
    quotas = {
        row.get("metric"): row
        for row in quota_rows
        if isinstance(row, dict) and isinstance(row.get("metric"), str)
    }

    required: list[tuple[str, int]] = []
    if str(spec.cloud["machine_type"]).startswith("a2-"):
        required.append(("A2_CPUS", int(guest_cpus)))
    if requested_accelerators:
        accelerator_metric = "NVIDIA_A100_GPUS"
        if spec.cloud["provisioning_model"] == "SPOT":
            accelerator_metric = "PREEMPTIBLE_NVIDIA_A100_GPUS"
        required.append((accelerator_metric, requested_accelerators))

    for metric, requested in required:
        row = quotas.get(metric)
        if row is None:
            raise HarnessError(f"regional quota {metric} is absent")
        limit = row.get("limit")
        usage = row.get("usage", 0)
        if not isinstance(limit, (int, float)) or not isinstance(usage, (int, float)):
            raise HarnessError(f"regional quota {metric} is malformed")
        available = float(limit) - float(usage)
        if available < requested:
            raise HarnessError(
                f"regional quota {metric} is insufficient in {region}: "
                f"limit={limit:g} usage={usage:g} available={available:g} "
                f"requested={requested}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "doctor", "render", "start", "status", "sync", "analyze"):
        command = subparsers.add_parser(name)
        command.add_argument("spec")
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("spec")
    launch_parser.add_argument("--yes", action="store_true")
    adopt_parser = subparsers.add_parser("adopt")
    adopt_parser.add_argument("spec")
    adopt_parser.add_argument("--instance-id", required=True)
    adopt_parser.add_argument("--yes", action="store_true")
    log_parser = subparsers.add_parser("logs")
    log_parser.add_argument("spec")
    log_parser.add_argument("--lines", type=int, default=100)
    delete_parser = subparsers.add_parser(
        "delete", help="delete an exactly-owned, successfully completed run"
    )
    delete_parser.add_argument("spec")
    delete_parser.add_argument("--instance-id", required=True)
    delete_parser.add_argument("--yes", action="store_true")
    abandon_parser = subparsers.add_parser(
        "abandon",
        help="preserve and delete an exactly-owned incomplete or failed run",
    )
    abandon_parser.add_argument("spec")
    abandon_parser.add_argument("--instance-id", required=True)
    abandon_parser.add_argument(
        "--reason",
        required=True,
        help="nonempty audit reason stored remotely, in GCS, and in local state",
    )
    abandon_parser.add_argument("--yes", action="store_true")
    image_parser = subparsers.add_parser("create-image")
    image_parser.add_argument("spec")
    image_parser.add_argument("--instance-id", required=True)
    image_parser.add_argument("--yes", action="store_true")
    canary_create_parser = subparsers.add_parser("create-canary")
    canary_create_parser.add_argument("spec")
    canary_test_parser = subparsers.add_parser("test-canary")
    canary_test_parser.add_argument("spec")
    canary_test_parser.add_argument("--canary-id", required=True)
    image_promote_parser = subparsers.add_parser("promote-image")
    image_promote_parser.add_argument("spec")
    image_promote_parser.add_argument("--canary-id", required=True)
    image_promote_parser.add_argument("--yes", action="store_true")
    canary_delete_parser = subparsers.add_parser("delete-canary")
    canary_delete_parser.add_argument("spec")
    canary_delete_parser.add_argument("--canary-id", required=True)
    canary_delete_parser.add_argument("--yes", action="store_true")
    matched_parser = subparsers.add_parser("render-matched")
    matched_parser.add_argument("spec")
    matched_parser.add_argument("analysis")
    matched_parser.add_argument("--subdirectory", default="matched-sgd")
    start_matched_parser = subparsers.add_parser("start-matched")
    start_matched_parser.add_argument("spec")
    start_matched_parser.add_argument("analysis")
    start_matched_parser.add_argument("--subdirectory", default="matched-sgd")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_spec(args.spec)
        runner = CommandRunner(dry_run=args.dry_run)
        if args.action == "validate":
            print(f"valid: {spec.run_id} ({spec.repo_commit})")
        elif args.action == "doctor":
            validate_cloud(spec, runner)
            print("cloud prerequisites: ok")
        elif args.action == "render":
            if spec.cloud.get("adopt_only"):
                print("# launch disabled: this is an adopt_only provenance spec")
            else:
                print(shlex.join(launch_command(spec)))
            print("\n# remote start script\n" + start_script(spec))
        elif args.action == "render-matched":
            try:
                analysis = _object(
                    json.loads(Path(args.analysis).read_text()), "analysis"
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise HarnessError(
                    f"cannot read analysis {args.analysis}: {exc}"
                ) from exc
            print(
                shlex.join(
                    matched_sgd_command(
                        spec,
                        analysis,
                        subdirectory=args.subdirectory,
                    )
                )
            )
        elif args.action == "start-matched":
            try:
                analysis = _object(
                    json.loads(Path(args.analysis).read_text()), "analysis"
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise HarnessError(
                    f"cannot read analysis {args.analysis}: {exc}"
                ) from exc
            print(
                json.dumps(
                    start_matched(
                        spec,
                        analysis,
                        runner,
                        args.state_dir,
                        subdirectory=args.subdirectory,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.action == "launch":
            print(
                json.dumps(
                    launch(
                        spec,
                        runner,
                        args.state_dir,
                        confirmed=args.yes,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.action == "adopt":
            print(
                json.dumps(
                    adopt(
                        spec,
                        runner,
                        args.state_dir,
                        exact_instance_id=args.instance_id,
                        confirmed=args.yes,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.action == "start":
            print(
                json.dumps(
                    start(spec, runner, args.state_dir), indent=2, sort_keys=True
                )
            )
        elif args.action == "status":
            print(status(spec, runner, args.state_dir), end="")
        elif args.action == "logs":
            print(logs(spec, runner, args.state_dir, args.lines), end="")
        elif args.action == "sync":
            sync(spec, runner, args.state_dir)
        elif args.action == "analyze":
            analyze(spec, runner, args.state_dir)
        elif args.action == "create-image":
            result = create_image(
                spec,
                runner,
                args.state_dir,
                exact_instance_id=args.instance_id,
                confirmed=args.yes,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.action == "create-canary":
            result = create_canary(spec, runner, args.state_dir)
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.action == "test-canary":
            print(
                test_canary(
                    spec,
                    runner,
                    args.state_dir,
                    exact_canary_id=args.canary_id,
                ),
                end="",
            )
        elif args.action == "promote-image":
            result = promote_image(
                spec,
                runner,
                args.state_dir,
                exact_canary_id=args.canary_id,
                confirmed=args.yes,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.action == "delete-canary":
            delete_canary(
                spec,
                runner,
                args.state_dir,
                exact_canary_id=args.canary_id,
                confirmed=args.yes,
            )
        elif args.action == "delete":
            delete(
                spec,
                runner,
                args.state_dir,
                exact_instance_id=args.instance_id,
                confirmed=args.yes,
            )
        elif args.action == "abandon":
            print(
                json.dumps(
                    abandon(
                        spec,
                        runner,
                        args.state_dir,
                        exact_instance_id=args.instance_id,
                        reason=args.reason,
                        confirmed=args.yes,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:  # pragma: no cover - argparse enforces the choices
            raise HarnessError(f"unknown action: {args.action}")
    except HarnessError as exc:
        print(f"optimizer-harness: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
