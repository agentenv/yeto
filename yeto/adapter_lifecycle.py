"""Safe lineage and compatibility checks for causal-LM adapter reuse."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .provenance import PROVENANCE_FILE

ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"
ADAPTER_MODES = ("resume", "branch")
LEARNER_PARENT_PATH = "~/yeto-parent-adapter"
HEAD_PARENT_PATH = "~/yeto-parent-adapter-src"

# Resume means "continue this recipe from these adapter weights".  Fields
# that control duration, output placement, or infrastructure are deliberately
# absent; changing any field below is a branch, not a resume.
RESUME_RECIPE_FIELDS = (
    "tuning",
    "base_quantization",
    "shard",
    "lora_r",
    "lora_alpha",
    "lora_targets",
    "loss_function",
    "loss_sha256",
    "train_on",
    "assistant_mask_mode",
    "data_format",
    "seq_len",
    "micro_batch_size",
    "grad_accum",
    "gradient_checkpointing",
    "inner_lr",
    "weight_decay",
    "warmup_steps",
    "seed",
    "max_rows",
    "tokenize",
    "stream_workers",
    "fragments",
    "fragment_pattern",
    "matrix_merge",
    "merge_alpha",
    "wire_dtype",
    "wan_streams",
    "attention_backend",
    "kernel_backend",
)

_LEARNER_RECIPE_DEFAULTS = {
    "gradient_checkpointing": "auto",
    "weight_decay": 0.01,
    "warmup_steps": 10,
    "matrix_merge": "rda",
}

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


def _sha256(value: str, *, flag: str = "--adapter-sha256") -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{flag} must be exactly 64 hexadecimal characters")
    return value.lower()


def selected_parent(args) -> tuple[str | None, str | None]:
    resume = getattr(args, "resume_from", None)
    branch = getattr(args, "branch_from", None)
    if resume and branch:
        raise ValueError("pass only one of --resume-from or --branch-from")
    if resume:
        return "resume", str(resume)
    if branch:
        return "branch", str(branch)
    return None, None


def prepare_parent_source(args) -> None:
    """Validate and attest a launch-time local/cloud parent before GPU spend."""
    _mode, source = selected_parent(args)
    if source is None:
        return
    from .datasource import kind

    source_kind = kind(source)
    expected = getattr(args, "adapter_sha256", None)
    if expected is not None:
        expected = _sha256(expected)
        args.adapter_sha256 = expected
    if source_kind == "hf":
        raise ValueError(
            "--resume-from/--branch-from currently require a local directory "
            "or cloud object-store URI; download Hub adapters first"
        )
    if source_kind == "cloud":
        if not expected:
            raise ValueError(
                "cloud parent adapters require --adapter-sha256 so mutable "
                "object-store contents are attested before training"
            )
        return
    actual = directory_sha256(source)
    if expected is not None and actual != expected:
        raise ValueError(
            f"adapter SHA256 mismatch: expected {expected}, got {actual}"
        )
    args.adapter_sha256 = actual


def learner_parent_arg(args) -> str | None:
    _mode, source = selected_parent(args)
    return LEARNER_PARENT_PATH if source is not None else None


def learner_parent_mounts(args) -> dict[str, str]:
    _mode, source = selected_parent(args)
    if source is None:
        return {}
    from .datasource import kind

    if kind(source) == "hf":
        raise ValueError("parent adapter was not materialized before learner launch")
    return {LEARNER_PARENT_PATH: os.path.expanduser(source)}


def head_stage_parent(args) -> dict[str, str]:
    """Stage a submitter-local parent onto the persistent head controller."""
    mode, source = selected_parent(args)
    if source is None:
        return {}
    from .datasource import kind

    if kind(source) != "local":
        return {}
    if mode == "resume":
        args.resume_from = HEAD_PARENT_PATH
    else:
        args.branch_from = HEAD_PARENT_PATH
    return {HEAD_PARENT_PATH: os.path.expanduser(source)}


def training_recipe(args) -> dict[str, Any]:
    """Serializable training identity recorded in every causal artifact."""
    return {
        field: getattr(args, field, _LEARNER_RECIPE_DEFAULTS.get(field))
        for field in RESUME_RECIPE_FIELDS
    }


def directory_sha256(path: str | Path) -> str:
    """Hash a directory by relative file name and bytes, rejecting symlinks."""
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in root.rglob("*") if item.is_file() or item.is_symlink()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"adapter directory is empty: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise ValueError(
                f"adapter directory contains a symbolic link: {relative}"
            )
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        bytes_read = 0
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                bytes_read += len(chunk)
        if bytes_read != size:
            raise ValueError(
                f"adapter file changed while it was being hashed: {relative}"
            )
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"adapter has no {description}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read adapter {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"adapter {description} must contain a JSON object: {path}")
    return value


def inspect_parent_adapter(
    args,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Validate the selected parent and return self-contained lineage data."""
    mode, source = selected_parent(args)
    if source is None:
        return None
    if getattr(args, "tuning", None) != "lora":
        raise ValueError("--resume-from/--branch-from require --tuning lora")

    root = Path(os.path.expanduser(source)).resolve()
    adapter_config = _read_json(root / ADAPTER_CONFIG_FILE, "adapter_config.json")
    peft_type = adapter_config.get("peft_type")
    if not isinstance(peft_type, str) or peft_type.upper() != "LORA":
        raise ValueError("parent adapter_config.json must declare peft_type=LORA")
    task_type = adapter_config.get("task_type")
    if task_type is not None and str(task_type).upper() != "CAUSAL_LM":
        raise ValueError(
            "parent adapter_config.json must target a causal language model"
        )
    if not (root / ADAPTER_WEIGHTS_FILE).is_file():
        raise ValueError(
            f"adapter has no safe {ADAPTER_WEIGHTS_FILE}: {root}; "
            "legacy pickle weights are not accepted"
        )
    digest = directory_sha256(root)
    expected = (
        _sha256(expected_sha256)
        if expected_sha256 is not None
        else None
    )
    if expected is not None and digest != expected:
        raise ValueError(
            "adapter SHA256 mismatch: expected "
            f"{expected}, got {digest}"
        )
    # Launcher mount paths deliberately use ``~``. Inspection expands them;
    # persist that concrete path because PEFT does not promise shell-style
    # tilde expansion when it receives a Python string.
    if mode == "resume":
        args.resume_from = str(root)
    else:
        args.branch_from = str(root)

    actual_r = adapter_config.get("r")
    actual_alpha = adapter_config.get("lora_alpha")
    if actual_r is not None and int(actual_r) != int(getattr(args, "lora_r", actual_r)):
        raise ValueError(
            f"parent adapter uses lora_r={actual_r}; pass --lora-r {actual_r}"
        )
    if actual_alpha is not None and int(actual_alpha) != int(
        getattr(args, "lora_alpha", actual_alpha)
    ):
        raise ValueError(
            "parent adapter uses lora_alpha="
            f"{actual_alpha}; pass --lora-alpha {actual_alpha}"
        )

    manifest_path = root / PROVENANCE_FILE
    manifest = (
        _read_json(manifest_path, PROVENANCE_FILE)
        if manifest_path.is_file()
        else None
    )
    current_provenance = getattr(args, "_provenance", {}) or {}
    current_model = current_provenance.get("model") or {}
    compared_manifest_base = False
    if manifest is not None:
        previous_model = manifest.get("model") or {}
        for field in ("resolved_identifier", "resolved_revision"):
            before = previous_model.get(field)
            after = current_model.get(field)
            if before is not None and after is not None and before != after:
                raise ValueError(
                    f"parent adapter base model {field} differs from this run"
                )
            compared_manifest_base = compared_manifest_base or before is not None
    if not compared_manifest_base:
        adapter_base = adapter_config.get("base_model_name_or_path")
        current_base = current_model.get("resolved_identifier")
        if adapter_base and current_base and adapter_base != current_base:
            raise ValueError(
                "legacy adapter base_model_name_or_path differs from this run"
            )

    if mode == "resume":
        if manifest is None:
            raise ValueError(
                "--resume-from requires a Yeto provenance manifest; use "
                "--branch-from for a reviewed legacy PEFT adapter"
            )
        previous = (manifest.get("artifact") or {}).get("training_recipe")
        if not isinstance(previous, dict):
            raise ValueError(
                "parent artifact predates resumable training recipes; use "
                "--branch-from to start a new lineage"
            )
        previous_dataset = manifest.get("dataset") or {}
        current_dataset = current_provenance.get("dataset") or {}
        dataset_changed = {
            field: (previous_dataset.get(field), current_dataset.get(field))
            for field in ("source", "resolved_identifier", "resolved_revision")
            if previous_dataset.get(field) != current_dataset.get(field)
        }
        if dataset_changed:
            detail = ", ".join(
                f"{field}: {before!r} -> {after!r}"
                for field, (before, after) in dataset_changed.items()
            )
            raise ValueError(
                f"resume dataset differs from the parent ({detail}); use "
                "--branch-from for intentional changes"
            )
        if manifest.get("trust_remote_code", False) != bool(
            current_provenance.get("trust_remote_code", False)
        ):
            raise ValueError(
                "resume trust_remote_code setting differs from the parent; "
                "use --branch-from for intentional changes"
            )
        current = training_recipe(args)
        changed = {
            field: (previous.get(field), current.get(field))
            for field in RESUME_RECIPE_FIELDS
            if previous.get(field) != current.get(field)
        }
        if changed:
            detail = ", ".join(
                f"{field}: {before!r} -> {after!r}"
                for field, (before, after) in changed.items()
            )
            raise ValueError(
                f"resume recipe differs from the parent ({detail}); use "
                "--branch-from for intentional changes"
            )

    return {
        "mode": mode,
        "sha256": digest,
        "source_artifact_kind": manifest.get("artifact_kind") if manifest else None,
        "source_model": manifest.get("model") if manifest else None,
        "source_dataset": manifest.get("dataset") if manifest else None,
        "source_training_recipe": (
            (manifest.get("artifact") or {}).get("training_recipe")
            if manifest
            else None
        ),
        "legacy_source": manifest is None,
        "adapter_config": {
            key: adapter_config.get(key)
            for key in (
                "peft_type",
                "task_type",
                "base_model_name_or_path",
                "r",
                "lora_alpha",
                "target_modules",
            )
            if key in adapter_config
        },
    }


def training_artifact_metadata(args) -> dict[str, Any]:
    recipe = getattr(args, "_training_recipe", None) or training_recipe(args)
    metadata: dict[str, Any] = {"training_recipe": recipe}
    lineage = getattr(args, "_adapter_lineage", None)
    if lineage:
        metadata["parent_adapter"] = lineage
    return metadata
