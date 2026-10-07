"""Closed stock-Codex backend profiles for the signed RL harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from yeto.rl.profiles.qwen3_8_next import (
    HF_REPO_4LAYER,
    HF_REPO_FULL,
    HF_REVISION_4LAYER,
    HF_REVISION_FULL,
    MODEL_NAMES as _FN_MODEL_NAMES,
)

QWEN38_MODEL = "Qwen/Qwen3.8-27B"
QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN35_MODEL = "Qwen/Qwen3.5-4B"
QWEN35_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN35_08B_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN35_08B_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
# Qwen3.8-Flash-Next identities are owned by ``yeto/rl/profiles/qwen3_8_next.py``
# (single source of truth shared with the FN launch scripts); re-exported here
# so the codex allowlist and its tests read the same constants.
QWEN38_NEXT_MODEL = HF_REPO_FULL
QWEN38_NEXT_REVISION = HF_REVISION_FULL
QWEN38_NEXT_4LAYER_MODEL = HF_REPO_4LAYER
QWEN38_NEXT_4LAYER_REVISION = HF_REVISION_4LAYER

# Qwen3.8 reasoning-effort template kwargs.  ``enable_thinking`` is the only key
# the HF template reads; ``preserve_thinking`` / ``reasoning_effort`` are consumed
# by the fork's fixed template (``consistant_kwargs`` -> must stay equal across turns).
_QWEN38_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {
    "enable_thinking": True,
    "preserve_thinking": True,
    "reasoning_effort": "xhigh",
}

# LoRA surface of the FN per-expert LoRA layout (``s1-runs/s14-fnsmoke-modal.sh``:
# ``--lora-targets all-linear --rl-lora-expert-rank 8``).  Profiles that do not
# declare ``lora_targets`` / ``lora_expert_rank`` keep the legacy ``attention`` / 0.
_FN_LORA_TARGETS = "all-linear"
_FN_LORA_EXPERT_RANK = 8
DEFAULT_LORA_TARGETS = "attention"
DEFAULT_LORA_EXPERT_RANK = 0


def _flash_next_profile(variant: str, chat_template: str) -> dict[str, Any]:
    """FN codex profile (design D1): own identity, family-level ``qwen4exp`` TITO."""

    repo = {"full": HF_REPO_FULL, "4layer": HF_REPO_4LAYER}[variant]
    revision = {"full": HF_REVISION_FULL, "4layer": HF_REVISION_4LAYER}[variant]
    return {
        "model": "qwen4exp",
        "tito_model": "qwen4exp",
        # FN launch scripts pass no ``--rl-model-recipe`` -> launcher default.
        "rl_model_recipe": "generic",
        "backend_reasoning_effort": "xhigh",
        "thinking": {"type": "enabled"},
        "chat_template": chat_template,
        "chat_template_kwargs": deepcopy(_QWEN38_CHAT_TEMPLATE_KWARGS),
        "tito_allowed_append_roles": ["tool", "user"],
        "model_identifier": repo,
        "model_revision": revision,
        "identity_label": _FN_MODEL_NAMES[variant],
        "lora_targets": _FN_LORA_TARGETS,
        "lora_expert_rank": _FN_LORA_EXPERT_RANK,
    }

# Qwen3.8-Flash-Next (full + 4-layer debug variant): single source of truth is
# ``yeto/rl/profiles/qwen3_8_next.py``; re-exported here for the tests.
QWEN38_NEXT_MODEL = HF_REPO_FULL
QWEN38_NEXT_REVISION = HF_REVISION_FULL
QWEN38_NEXT_4LAYER_MODEL = HF_REPO_4LAYER
QWEN38_NEXT_4LAYER_REVISION = HF_REVISION_4LAYER

# LoRA layout defaults for profiles that do not declare one (legacy
# attention-only LoRA, no routed-expert LoRA).
DEFAULT_LORA_TARGETS = "attention"
DEFAULT_LORA_EXPERT_RANK = 0

_QWEN38_NEXT_KWARGS: dict[str, Any] = {
    "enable_thinking": True,
    "preserve_thinking": True,
    "reasoning_effort": "xhigh",
}


def _fn_profile(variant: str, model: str, revision: str) -> dict[str, Any]:
    """rl-fn-codex-rollout D1: FN identity on the fork's ``qwen4exp`` TITO family.

    Both FN variants keep their own identity (label / HF repo / revision) while
    sharing the family-level ``Qwen38SmallTITOTokenizer`` fixed template, the
    same way ``qwen35_08b`` reuses ``qwen35``.  The launcher passes no
    ``--rl-model-recipe`` for FN (``s1-runs/s14-fnsmoke-modal.sh``), so the
    recipe stays ``generic``; LoRA layout is ``all-linear`` with per-expert
    rank 8 (``--lora-targets all-linear --rl-lora-expert-rank 8``).
    """

    return {
        "model": "qwen4exp",
        "tito_model": "qwen4exp",
        "rl_model_recipe": "generic",
        "backend_reasoning_effort": "xhigh",
        "thinking": {"type": "enabled"},
        "chat_template": f"qwen38_next_{variant}" if variant != "full" else "qwen38_next",
        "chat_template_kwargs": dict(_QWEN38_NEXT_KWARGS),
        "tito_allowed_append_roles": ["tool", "user"],
        "model_identifier": model,
        "model_revision": revision,
        "identity_label": _FN_MODEL_NAMES[variant],
        "lora_targets": "all-linear",
        "lora_expert_rank": 8,
        # ``qwen3.8_small_and_flash_next_fixed.jinja`` renders every assistant
        # <think> block when ``preserve_thinking`` is undefined or true.
        "keeps_history_reasoning": True,
    }


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
        "keeps_history_reasoning": True,
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
        "keeps_history_reasoning": True,
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
        # ``qwen3.5_fixed.jinja`` + ``preserve_thinking: True`` keeps every
        # historical <think> block (rl-fn-codex-rollout D2 / upstream 5.1).
        "keeps_history_reasoning": True,
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
        "keeps_history_reasoning": True,
    },
    "qwen38_next": _fn_profile("full", QWEN38_NEXT_MODEL, QWEN38_NEXT_REVISION),
    "qwen38_next_4layer": _fn_profile(
        "4layer", QWEN38_NEXT_4LAYER_MODEL, QWEN38_NEXT_4LAYER_REVISION
    ),
}


def stock_codex_keeps_history_reasoning(profile_name: str) -> bool:
    """5.1: whether the profile's fixed TITO template keeps historical reasoning.

    Every supported profile must declare it explicitly; the gateway's
    ``ChainRegistry`` is initialised from this value, so a missing declaration
    is a profile bug, not a default.
    """

    profile = stock_codex_backend_profile(profile_name)
    try:
        value = profile["keeps_history_reasoning"]
    except KeyError as exc:
        raise ValueError(
            f"stock Codex profile {profile_name!r} does not declare keeps_history_reasoning"
        ) from exc
    if not isinstance(value, bool):
        raise ValueError("keeps_history_reasoning must be a bool")
    return value


def stock_codex_lora_layout(profile_name: str) -> tuple[str, int]:
    """Return the declared ``(lora_targets, lora_expert_rank)`` of one profile.

    Profiles without a declaration keep the legacy ``attention`` / 0 layout.
    """

    profile = stock_codex_backend_profile(profile_name)
    return (
        str(profile.get("lora_targets", DEFAULT_LORA_TARGETS)),
        int(profile.get("lora_expert_rank", DEFAULT_LORA_EXPERT_RANK)),
    )


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
    lora_expert_rank: int = DEFAULT_LORA_EXPERT_RANK,
) -> dict[str, Any]:
    """Validate the complete model-facing stock-Codex identity surface.

    ``lora_targets`` / ``lora_expert_rank`` are compared against the profile's
    own declaration (``stock_codex_lora_layout``); undeclared profiles keep the
    historical ``attention`` / 0 requirement unchanged.
    """

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
    expected_lora_targets, expected_lora_expert_rank = stock_codex_lora_layout(
        profile_name
    )
    if expected_model is not None and (
        model != expected_model
        or model_revision != expected_revision
        or rollout_model not in {None, expected_model}
        or rollout_model_revision not in {None, expected_revision}
        or lora_targets != expected_lora_targets
        or int(lora_expert_rank or 0) != expected_lora_expert_rank
        or expert_full_count != 0
    ):
        raise ValueError(
            f"stock Codex {profile['identity_label']} model identity drifted"
        )
    return profile
