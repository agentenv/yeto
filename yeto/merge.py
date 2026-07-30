"""Merge a causal-LM PEFT adapter into its base model for deployment."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from .adapter_lifecycle import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHTS_FILE,
    directory_sha256,
)
from .provenance import PROVENANCE_FILE


def _read_manifest(adapter_dir: Path) -> dict | None:
    path = adapter_dir / PROVENANCE_FILE
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read adapter provenance {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"adapter provenance must contain a JSON object: {path}")
    return value


def _base_from_manifest(manifest: dict | None) -> tuple[str | None, str | None]:
    model = (manifest or {}).get("model") or {}
    return model.get("resolved_identifier"), model.get("resolved_revision")


def _torch_dtype(name: str):
    return {
        "auto": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "f32": torch.float32,
    }[name]


def merge_adapter(args) -> Path:
    """Load, safe-merge, shard, attest, and atomically publish one adapter."""
    adapter_dir = Path(args.adapter_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not adapter_dir.is_dir():
        raise ValueError(f"adapter directory does not exist: {adapter_dir}")
    if not (adapter_dir / ADAPTER_CONFIG_FILE).is_file():
        raise ValueError(f"adapter has no {ADAPTER_CONFIG_FILE}: {adapter_dir}")
    try:
        adapter_config = json.loads(
            (adapter_dir / ADAPTER_CONFIG_FILE).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read adapter configuration: {exc}") from exc
    if not isinstance(adapter_config, dict):
        raise ValueError("adapter_config.json must contain a JSON object")
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("adapter_config.json must declare peft_type=LORA")
    task_type = adapter_config.get("task_type")
    if task_type is not None and str(task_type).upper() != "CAUSAL_LM":
        raise ValueError("adapter must target a causal language model")
    if not (adapter_dir / ADAPTER_WEIGHTS_FILE).is_file():
        raise ValueError(
            f"adapter has no safe {ADAPTER_WEIGHTS_FILE}: {adapter_dir}; "
            "legacy pickle weights are not accepted"
        )
    if adapter_dir.resolve() == output_dir.resolve():
        raise ValueError("--output-dir must differ from --adapter-dir")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")

    manifest = _read_manifest(adapter_dir)
    recorded_model, recorded_revision = _base_from_manifest(manifest)
    model_id = args.model or recorded_model
    if not model_id:
        raise ValueError(
            "adapter has no Yeto base-model provenance; pass --model explicitly"
        )
    revision = args.model_revision or recorded_revision

    runtime_args = SimpleNamespace(
        model=model_id,
        model_revision=revision,
        data=None,
        trust_remote_code=bool(args.trust_remote_code),
        source_sha256=None,
    )
    from .provenance import pin_runtime_provenance

    current = pin_runtime_provenance(runtime_args, include_data=False)
    resolved = current["model"]
    if recorded_model is not None:
        if resolved.get("resolved_identifier") != recorded_model:
            raise ValueError(
                "--model resolves to a different base than the adapter provenance"
            )
        if (
            recorded_revision is not None
            and resolved.get("resolved_revision") != recorded_revision
        ):
            raise ValueError(
                "--model-revision differs from the adapter's immutable base revision"
            )
    adapter_base = adapter_config.get("base_model_name_or_path")
    if (
        adapter_base
        and resolved.get("resolved_identifier")
        and adapter_base != resolved["resolved_identifier"]
    ):
        raise ValueError(
            "adapter_config.json names a different base than the selected model"
        )

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {
        "revision": runtime_args.model_revision,
        "trust_remote_code": runtime_args.trust_remote_code,
        "use_safetensors": True,
    }
    dtype = _torch_dtype(args.dtype)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    base = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    base.to(torch.device(args.device))
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft_model.merge_and_unload(safe_merge=True)

    tokenizer_source = (
        adapter_dir
        if (adapter_dir / "tokenizer_config.json").is_file()
        else model_id
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if tokenizer_source == adapter_dir else runtime_args.model_revision,
        trust_remote_code=runtime_args.trust_remote_code,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    adapter_digest = directory_sha256(adapter_dir)
    try:
        merged.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer.save_pretrained(temporary)
        from .provenance import write_provenance_manifest

        write_provenance_manifest(
            temporary,
            runtime_args,
            artifact_kind="causal-lm-merged-model",
            extra={
                "parent_adapter_sha256": adapter_digest,
                "parent_artifact_kind": (
                    manifest.get("artifact_kind") if manifest else None
                ),
                "safe_merge": True,
                "safe_serialization": True,
                "max_shard_size": args.max_shard_size,
            },
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"merged adapter {adapter_digest[:12]} into {model_id} and saved "
        f"SafeTensors shards (max {args.max_shard_size}) to {output_dir}"
    )
    return output_dir
