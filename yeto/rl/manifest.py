"""Canonical static run identity for the pinned Miles RL integration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MILES_REPOSITORY = "https://github.com/radixark/miles"
MILES_COMMIT = "dfc66ff38752bfa2c5d325e0037ebc4b537c06de"
MANIFEST_SCHEMA = 1

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

ATTENTION_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
ALL_LINEAR_TARGET_MODULES = ATTENTION_TARGET_MODULES + (
    "gate_proj",
    "up_proj",
    "down_proj",
)


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def manifest_sha256(value: Mapping[str, Any] | str) -> str:
    text = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_manifest(text: str, expected_sha256: str | None = None) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported RL run manifest")
    if canonical_json(value) != text:
        raise ValueError("RL run manifest is not canonical JSON")
    actual = manifest_sha256(text)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("RL run manifest SHA256 mismatch")
    if value.get("mode") != "rl-strict-avg" or not value.get("run_id"):
        raise ValueError("RL run manifest has an invalid mode or run ID")
    miles = value.get("miles") or {}
    if miles.get("repository") != MILES_REPOSITORY or miles.get("commit") != MILES_COMMIT:
        raise ValueError("RL run manifest does not pin the supported Miles commit")
    if not is_immutable_image(value.get("learner_image")):
        raise ValueError("RL run manifest does not pin an immutable learner image")
    topology = value.get("topology") or {}
    if topology.get("trainer") != "megatron" or topology.get("rollout") != "sglang":
        raise ValueError("RL run manifest uses an unsupported runtime topology")
    if any(topology.get(name) != 1 for name in (
        "tensor_parallel", "pipeline_parallel", "expert_parallel", "data_parallel"
    )):
        raise ValueError("RL run manifest uses an unsupported parallel topology")
    base = value.get("base_model") or {}
    if not re.fullmatch(r"[0-9a-f]{40}", str(base.get("revision", ""))):
        raise ValueError("RL run manifest does not pin the base model commit")
    if not base.get("identifier") or base.get("trust_remote_code") is not True:
        raise ValueError("RL run manifest has an invalid base model")
    tokenizer = value.get("tokenizer") or {}
    if (
        tokenizer.get("identifier") != base["identifier"]
        or tokenizer.get("revision") != base["revision"]
        or tokenizer.get("chat_template") != "model-repository-default"
    ):
        raise ValueError("RL tokenizer identity does not match the base model")

    dataset = value.get("dataset") or {}
    revision = dataset.get("revision")
    content_hash = dataset.get("content_sha256")
    if not dataset.get("identifier") or dataset.get("assignment") != "miles-seed-plus-logical-learner-id":
        raise ValueError("RL run manifest has an invalid dataset assignment")
    if revision is None:
        if not _SHA256.fullmatch(str(content_hash or "")):
            raise ValueError("RL local dataset has no valid content hash")
    elif not re.fullmatch(r"[0-9a-f]{40}", str(revision)) or content_hash is not None:
        raise ValueError("RL Hub dataset revision is not immutable")

    lora = value.get("lora") or {}
    if (
        not _positive_int(lora.get("rank"))
        or lora.get("alpha") != lora.get("rank")
        or lora.get("target_modules") != list(target_modules(str(lora.get("targets", ""))))
        or lora.get("dropout") != 0.0
        or lora.get("bias") != "none"
        or lora.get("mapping") != "peft-canonical-v1"
    ):
        raise ValueError("RL run manifest has an unsupported LoRA configuration")

    workload = value.get("workload") or {}
    workload_names = (
        "learners",
        "groups_per_island_round",
        "samples_per_group",
        "local_optimizer_steps",
        "global_rounds",
    )
    if any(not _positive_int(workload.get(name)) for name in workload_names):
        raise ValueError("RL run manifest workload values must be positive integers")
    if workload["learners"] < 2:
        raise ValueError("RL strict averaging requires at least two islands")
    if (
        workload["groups_per_island_round"] * workload["samples_per_group"]
    ) % workload["local_optimizer_steps"]:
        raise ValueError("RL run manifest G*K must be divisible by U")

    if value.get("grpo") != {
        "advantage_estimator": "grpo",
        "rewards_normalization": True,
        "clip_low": 0.2,
        "clip_high": 0.28,
        "kl_loss_coefficient": 0.0,
        "entropy_coefficient": 0.0,
    }:
        raise ValueError("RL run manifest has an unsupported GRPO configuration")
    optimizer = value.get("optimizer") or {}
    learning_rate = optimizer.get("learning_rate")
    if (
        optimizer.get("name") != "adam"
        or type(learning_rate) not in {int, float}
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or optimizer.get("schedule") != "constant"
        or optimizer.get("betas") != [0.9, 0.999]
        or optimizer.get("weight_decay") != 0.0
        or optimizer.get("gradient_clip") != 1.0
        or optimizer.get("reset_each_global_round") is not True
    ):
        raise ValueError("RL run manifest has an unsupported optimizer configuration")
    generation = value.get("generation") or {}
    context = generation.get("max_context_length")
    prompt = generation.get("max_prompt_length")
    response = generation.get("max_response_length")
    if (
        not _positive_int(context)
        or not _positive_int(prompt)
        or not _positive_int(response)
        or prompt + response != context
        or generation.get("temperature") != 1.0
        or type(generation.get("seed")) is not int
    ):
        raise ValueError("RL run manifest has an invalid generation configuration")
    custom_generate = generation.get("custom_generate")
    if custom_generate is not None:
        if not isinstance(custom_generate, dict):
            raise ValueError("RL run manifest has an invalid generate function")
        callable_spec = str(custom_generate.get("callable", ""))
        module, separator, function = callable_spec.partition(":")
        if (
            not separator
            or not module
            or module.endswith(".py")
            or not function.isidentifier()
            or not _SHA256.fullmatch(str(custom_generate.get("source_sha256", "")))
        ):
            raise ValueError("RL run manifest has an invalid generate function")

    reward = value.get("reward") or {}
    callable_spec = str(reward.get("callable", ""))
    module, separator, function = callable_spec.partition(":")
    if (
        not separator
        or not module
        or module.endswith(".py")
        or not function.isidentifier()
        or not _SHA256.fullmatch(str(reward.get("source_sha256", "")))
    ):
        raise ValueError("RL run manifest has no valid reward source hash")
    if not _SHA256.fullmatch(str(value.get("yeto_source_sha256", ""))):
        raise ValueError("RL run manifest has no valid Yeto source hash")
    _validate_canonical_layout(value.get("canonical_lora"))
    return value


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _validate_canonical_layout(value: Any) -> None:
    from ..protocol import layout_fingerprint
    from .core import CanonicalTensorSpec, build_avg_layout

    if not isinstance(value, dict) or not isinstance(value.get("tensors"), list):
        raise ValueError("RL run manifest has no canonical LoRA layout")
    try:
        specs = tuple(
            CanonicalTensorSpec(
                item["name"],
                tuple(int(dim) for dim in item["shape"]),
                int(item["numel"]),
            )
            for item in value["tensors"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("RL run manifest has an invalid canonical LoRA layout") from exc
    if tuple(sorted(specs, key=lambda spec: spec.name)) != specs:
        raise ValueError("RL run manifest canonical LoRA tensors are not sorted")
    actual = layout_fingerprint(build_avg_layout(specs)).hex()
    if value.get("layout_fingerprint") != actual:
        raise ValueError("RL run manifest canonical LoRA fingerprint mismatch")


def is_immutable_image(value: str | None) -> bool:
    if not value:
        return False
    images = [part.partition("=")[2] or part for part in value.split(",")]
    return all(
        bool(
            re.search(r"@sha256:[0-9a-f]{64}\Z", image)
            or re.fullmatch(r"ami-[0-9a-f]{8,17}", image)
            or re.fullmatch(
                r"(?:https://www\.googleapis\.com/compute/v1/)?"
                r"projects/[A-Za-z0-9._-]+/global/images/[A-Za-z0-9._-]+",
                image,
            )
        )
        for image in images
    )


def target_modules(choice: str) -> tuple[str, ...]:
    if choice == "attention":
        return ATTENTION_TARGET_MODULES
    if choice in {"auto", "all-linear"}:
        return ALL_LINEAR_TARGET_MODULES
    raise ValueError(f"unsupported RL LoRA target selection {choice!r}")


def path_tree_sha256(path: str | Path) -> str:
    """Stable content identity for a local dataset file or directory."""

    root = Path(path).expanduser().resolve()
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"RL dataset {root} contains no files")
    digest = hashlib.sha256()
    for file in files:
        relative = "" if root.is_file() else file.relative_to(root).as_posix()
        name = relative.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(args, *, learners: int, reward_sha256: str) -> dict[str, Any]:
    provenance = getattr(args, "_provenance", {})
    model = provenance.get("model", {})
    dataset = provenance.get("dataset", {})
    lora_targets = target_modules(args.lora_targets)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "mode": "rl-strict-avg",
        "run_id": args.cluster_prefix,
        "base_model": {
            "identifier": model.get("resolved_identifier", args.model),
            "revision": model.get("resolved_revision", args.model_revision),
            "requested_identifier": getattr(args, "model_requested_identifier", args.model),
            "requested_revision": getattr(args, "model_requested_revision", None),
            "trust_remote_code": bool(args.trust_remote_code),
        },
        "tokenizer": {
            "identifier": model.get("resolved_identifier", args.model),
            "revision": model.get("resolved_revision", args.model_revision),
            "chat_template": "model-repository-default",
        },
        "dataset": {
            "identifier": dataset.get("resolved_identifier", args.data),
            "revision": dataset.get("resolved_revision", getattr(args, "data_revision", None)),
            "requested_identifier": getattr(args, "data_requested_identifier", args.data),
            "requested_revision": getattr(args, "data_requested_revision", None),
            "content_sha256": getattr(args, "rl_data_sha256", None),
            "assignment": "miles-seed-plus-logical-learner-id",
        },
        "lora": {
            "rank": args.lora_r,
            "alpha": args.lora_r,
            "targets": args.lora_targets,
            "target_modules": list(lora_targets),
            "dropout": 0.0,
            "bias": "none",
            "mapping": "peft-canonical-v1",
        },
        "workload": {
            "learners": learners,
            "groups_per_island_round": args.rl_groups_per_island_round,
            "samples_per_group": args.rl_samples_per_group,
            "local_optimizer_steps": args.rl_local_optimizer_steps,
            "global_rounds": args.rl_global_rounds,
        },
        "grpo": {
            "advantage_estimator": "grpo",
            "rewards_normalization": True,
            "clip_low": 0.2,
            "clip_high": 0.28,
            "kl_loss_coefficient": 0.0,
            "entropy_coefficient": 0.0,
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": args.inner_lr,
            "schedule": "constant",
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "reset_each_global_round": True,
        },
        "generation": {
            "max_context_length": args.seq_len,
            "max_prompt_length": args.seq_len // 2,
            "max_response_length": args.seq_len - args.seq_len // 2,
            "temperature": 1.0,
            "seed": args.seed,
            "custom_generate": (
                {
                    "callable": args.rl_generate_function,
                    "source_sha256": args.rl_generate_sha256,
                }
                if getattr(args, "rl_generate_function", None)
                else None
            ),
        },
        "reward": {"callable": args.reward_function, "source_sha256": reward_sha256},
        "miles": {"repository": MILES_REPOSITORY, "commit": MILES_COMMIT},
        "yeto_source_sha256": args.source_sha256,
        "learner_image": args.learner_image,
        "topology": {
            "trainer": "megatron",
            "rollout": "sglang",
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "expert_parallel": 1,
            "data_parallel": 1,
        },
        "canonical_lora": args.rl_canonical_layout,
    }
    text = canonical_json(manifest)
    if not _SHA256.fullmatch(manifest_sha256(text)):
        raise AssertionError("manifest SHA256 encoding failed")
    return manifest
