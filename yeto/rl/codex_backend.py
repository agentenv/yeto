"""Closed stock-Codex backend profiles for the signed RL harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

QWEN38_MODEL = "Qwen/Qwen3.8-27B"
QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN35_MODEL = "Qwen/Qwen3.5-4B"
QWEN35_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN35_08B_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN35_08B_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"

_PROFILES: dict[str, dict[str, Any]] = {
    "deepseekv4": {
        "model": "deepseekv4",
        "rl_model_recipe": "deepseek-v4-flash",
        "backend_reasoning_effort": "max",
        "thinking": {"type": "enabled"},
        "chat_template": "deepseekv4",
        "chat_template_kwargs": {
            "thinking_mode": "thinking",
            "reasoning_effort": "max",
            "drop_thinking": False,
        },
        "tito_allowed_append_roles": ["tool", "user"],
    },
    "qwen38": {
        "model": "qwen38",
        "rl_model_recipe": "generic",
        "backend_reasoning_effort": "xhigh",
        "thinking": {"type": "enabled"},
        "chat_template": "qwen38",
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        },
        "tito_allowed_append_roles": ["tool", "user"],
        "model_identifier": QWEN38_MODEL,
        "model_revision": QWEN38_REVISION,
        "identity_label": "Qwen3.8",
    },
    "qwen35": {
        "model": "qwen35",
        "rl_model_recipe": "generic",
        "backend_reasoning_effort": "xhigh",
        "thinking": {"type": "enabled"},
        "chat_template": "qwen35",
        # Miles' Qwen3.5 TITO profile owns the fixed Qwen3.5 template and
        # requires retained reasoning across append-only tool/user turns.
        "chat_template_kwargs": {"clear_thinking": False},
        "tito_allowed_append_roles": ["tool", "user"],
        "model_identifier": QWEN35_MODEL,
        "model_revision": QWEN35_REVISION,
        "identity_label": "Qwen3.5",
    },
    # This is a distinct, closed model identity while deliberately reusing
    # Miles' model-family-level Qwen3.5 TITO implementation.  The profile name
    # is carried separately from ``--tito-model qwen35`` so selecting the 0.8B
    # checkpoint never weakens the existing 4B allowlist entry.
    "qwen35_08b": {
        "model": "qwen35",
        "rl_model_recipe": "generic",
        "backend_reasoning_effort": "xhigh",
        "thinking": {"type": "enabled"},
        "chat_template": "qwen35_08b",
        "chat_template_kwargs": {"clear_thinking": False},
        "tito_allowed_append_roles": ["tool", "user"],
        "model_identifier": QWEN35_08B_MODEL,
        "model_revision": QWEN35_08B_REVISION,
        "identity_label": "Qwen3.5-0.8B",
        "tito_model": "qwen35",
    },
}


def stock_codex_backend_profile(profile_name: str) -> dict[str, Any]:
    """Return one immutable allowlisted profile as a defensive copy."""

    try:
        profile = _PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError("unsupported stock Codex backend profile") from exc
    return deepcopy(profile)


def stock_codex_tito_model(profile_name: str) -> str:
    """Return the Miles tokenizer family for one exact Codex backend profile."""

    profile = stock_codex_backend_profile(profile_name)
    return str(profile.get("tito_model", profile["model"]))


def stock_codex_backend_contract(
    profile_name: str,
    max_tokens: int,
) -> dict[str, Any]:
    profile = stock_codex_backend_profile(profile_name)
    return {
        "profile": profile_name,
        "tito_model": stock_codex_tito_model(profile_name),
        "model": profile["model"],
        "max_tokens": max_tokens,
        "reasoning_effort": profile["backend_reasoning_effort"],
        "thinking": profile["thinking"],
        "chat_template": profile["chat_template"],
        "chat_template_kwargs": profile["chat_template_kwargs"],
        "tito_allowed_append_roles": profile["tito_allowed_append_roles"],
    }


def validate_stock_codex_fields(
    *,
    tito_model: str,
    codex_backend_profile: str | None = None,
    rl_model_recipe: str,
    model: str,
    model_revision: str,
    rollout_model: str | None,
    rollout_model_revision: str | None,
    apply_chat_template_kwargs: dict[str, Any] | None,
    tito_allowed_append_roles: list[str] | None,
    codex_reasoning_effort: str | None,
    lora_targets: str,
    expert_full_count: int,
) -> dict[str, Any]:
    """Validate the complete model-facing stock-Codex identity surface."""

    if codex_reasoning_effort != "xhigh":
        raise ValueError("the stock Codex harness requires xhigh reasoning")
    profile_name = codex_backend_profile or tito_model
    profile = stock_codex_backend_profile(profile_name)
    expected_tito_model = stock_codex_tito_model(profile_name)
    if tito_model != expected_tito_model:
        raise ValueError(
            "stock Codex backend profile does not match the Miles TITO family"
        )
    if rl_model_recipe != profile["rl_model_recipe"]:
        raise ValueError("stock Codex model recipe does not match its profile")
    if apply_chat_template_kwargs != profile["chat_template_kwargs"]:
        raise ValueError("stock Codex chat-template kwargs do not match its profile")
    if tito_allowed_append_roles != profile["tito_allowed_append_roles"]:
        raise ValueError("stock Codex append roles do not match its profile")
    expected_model = profile.get("model_identifier")
    expected_revision = profile.get("model_revision")
    if expected_model is not None and (
        model != expected_model
        or model_revision != expected_revision
        or rollout_model not in {None, expected_model}
        or rollout_model_revision not in {None, expected_revision}
        or lora_targets != "attention"
        or expert_full_count != 0
    ):
        raise ValueError(
            f"stock Codex {profile['identity_label']} model identity drifted"
        )
    return profile
