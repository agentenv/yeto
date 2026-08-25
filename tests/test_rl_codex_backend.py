from __future__ import annotations

import copy

import pytest

from yeto.rl.codex_backend import (
    QWEN35_MODEL,
    QWEN35_REVISION,
    QWEN38_MODEL,
    QWEN38_REVISION,
    stock_codex_backend_contract,
    stock_codex_backend_profile,
    validate_stock_codex_fields,
)

QWEN38_KWARGS = {
    "enable_thinking": True,
    "preserve_thinking": True,
    "reasoning_effort": "xhigh",
}

QWEN35_KWARGS = {"clear_thinking": False}


def _validate_qwen38(**overrides):
    values = {
        "tito_model": "qwen38",
        "rl_model_recipe": "generic",
        "model": QWEN38_MODEL,
        "model_revision": QWEN38_REVISION,
        "rollout_model": None,
        "rollout_model_revision": None,
        "apply_chat_template_kwargs": copy.deepcopy(QWEN38_KWARGS),
        "tito_allowed_append_roles": ["tool", "user"],
        "codex_reasoning_effort": "xhigh",
        "lora_targets": "attention",
        "expert_full_count": 0,
    }
    values.update(overrides)
    return validate_stock_codex_fields(**values)


def _validate_qwen35(**overrides):
    values = {
        "tito_model": "qwen35",
        "rl_model_recipe": "generic",
        "model": QWEN35_MODEL,
        "model_revision": QWEN35_REVISION,
        "rollout_model": None,
        "rollout_model_revision": None,
        "apply_chat_template_kwargs": copy.deepcopy(QWEN35_KWARGS),
        "tito_allowed_append_roles": ["tool", "user"],
        "codex_reasoning_effort": "xhigh",
        "lora_targets": "attention",
        "expert_full_count": 0,
    }
    values.update(overrides)
    return validate_stock_codex_fields(**values)


def test_qwen38_stock_codex_profile_is_closed_and_defensively_copied():
    profile = stock_codex_backend_profile("qwen38")
    assert profile["model_identifier"] == QWEN38_MODEL
    assert profile["model_revision"] == QWEN38_REVISION
    assert profile["backend_reasoning_effort"] == "xhigh"
    assert profile["chat_template_kwargs"] == QWEN38_KWARGS
    assert profile["tito_allowed_append_roles"] == ["tool", "user"]

    profile["chat_template_kwargs"]["reasoning_effort"] = "low"
    assert stock_codex_backend_profile("qwen38")["chat_template_kwargs"] == (
        QWEN38_KWARGS
    )
    with pytest.raises(ValueError, match="unsupported"):
        stock_codex_backend_profile("qwen")


def test_qwen38_stock_codex_backend_contract_binds_xhigh_native_profile():
    assert stock_codex_backend_contract("qwen38", 32768) == {
        "model": "qwen38",
        "max_tokens": 32768,
        "reasoning_effort": "xhigh",
        "thinking": {"type": "enabled"},
        "chat_template": "qwen38",
        "chat_template_kwargs": QWEN38_KWARGS,
        "tito_allowed_append_roles": ["tool", "user"],
    }
    assert _validate_qwen38()["model_identifier"] == QWEN38_MODEL


def test_qwen35_stock_codex_backend_contract_binds_fixed_tito_profile():
    assert stock_codex_backend_contract("qwen35", 32768) == {
        "model": "qwen35",
        "max_tokens": 32768,
        "reasoning_effort": "xhigh",
        "thinking": {"type": "enabled"},
        "chat_template": "qwen35",
        "chat_template_kwargs": QWEN35_KWARGS,
        "tito_allowed_append_roles": ["tool", "user"],
    }
    assert _validate_qwen35()["model_identifier"] == QWEN35_MODEL
    with pytest.raises(ValueError, match="Qwen3.5 model identity"):
        _validate_qwen35(model_revision="0" * 40)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "Qwen/Qwen3.6-27B"},
        {"model_revision": "0" * 40},
        {"rollout_model": "Qwen/Qwen3.8-27B-FP8"},
        {"rollout_model_revision": "0" * 40},
        {"lora_targets": "all"},
        {"expert_full_count": 1},
        {"codex_reasoning_effort": "high"},
        {"tito_allowed_append_roles": ["tool"]},
        {
            "apply_chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": False,
                "reasoning_effort": "xhigh",
            }
        },
    ],
)
def test_qwen38_stock_codex_profile_fails_closed_on_identity_drift(overrides):
    with pytest.raises(ValueError):
        _validate_qwen38(**overrides)


def test_existing_deepseek_v4_stock_codex_profile_is_unchanged():
    assert stock_codex_backend_contract("deepseekv4", 4096) == {
        "model": "deepseekv4",
        "max_tokens": 4096,
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
