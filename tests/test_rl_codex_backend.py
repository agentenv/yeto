from __future__ import annotations

import copy

import pytest

from yeto.rl.codex_backend import (
    QWEN35_08B_MODEL,
    QWEN35_08B_REVISION,
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
        "profile": "qwen38",
        "tito_model": "qwen38",
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
        "profile": "qwen35",
        "tito_model": "qwen35",
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


def test_qwen35_08b_is_a_distinct_closed_profile_using_qwen35_tito():
    profile = stock_codex_backend_profile("qwen35_08b")
    assert profile["model"] == "qwen35"
    assert profile["tito_model"] == "qwen35"
    assert profile["model_identifier"] == QWEN35_08B_MODEL
    assert profile["model_revision"] == QWEN35_08B_REVISION
    assert profile["chat_template"] == "qwen35_08b"
    assert stock_codex_backend_contract("qwen35_08b", 32768) == {
        "profile": "qwen35_08b",
        "tito_model": "qwen35",
        "model": "qwen35",
        "max_tokens": 32768,
        "reasoning_effort": "xhigh",
        "thinking": {"type": "enabled"},
        "chat_template": "qwen35_08b",
        "chat_template_kwargs": QWEN35_KWARGS,
        "tito_allowed_append_roles": ["tool", "user"],
    }
    assert (
        validate_stock_codex_fields(
            tito_model="qwen35",
            codex_backend_profile="qwen35_08b",
            rl_model_recipe="generic",
            model=QWEN35_08B_MODEL,
            model_revision=QWEN35_08B_REVISION,
            rollout_model=None,
            rollout_model_revision=None,
            apply_chat_template_kwargs=copy.deepcopy(QWEN35_KWARGS),
            tito_allowed_append_roles=["tool", "user"],
            codex_reasoning_effort="xhigh",
            lora_targets="attention",
            expert_full_count=0,
        )["tito_model"]
        == "qwen35"
    )
    with pytest.raises(ValueError, match="Miles TITO family"):
        validate_stock_codex_fields(
            tito_model="qwen35_08b",
            codex_backend_profile="qwen35_08b",
            rl_model_recipe="generic",
            model=QWEN35_08B_MODEL,
            model_revision=QWEN35_08B_REVISION,
            rollout_model=None,
            rollout_model_revision=None,
            apply_chat_template_kwargs=copy.deepcopy(QWEN35_KWARGS),
            tito_allowed_append_roles=["tool", "user"],
            codex_reasoning_effort="xhigh",
            lora_targets="attention",
            expert_full_count=0,
        )
    with pytest.raises(ValueError, match="Qwen3.5-0.8B model identity"):
        validate_stock_codex_fields(
            tito_model="qwen35",
            codex_backend_profile="qwen35_08b",
            rl_model_recipe="generic",
            model=QWEN35_MODEL,
            model_revision=QWEN35_REVISION,
            rollout_model=None,
            rollout_model_revision=None,
            apply_chat_template_kwargs=copy.deepcopy(QWEN35_KWARGS),
            tito_allowed_append_roles=["tool", "user"],
            codex_reasoning_effort="xhigh",
            lora_targets="attention",
            expert_full_count=0,
        )


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
        "profile": "deepseekv4",
        "tito_model": "deepseekv4",
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


# ----------------------------------------------------------- rl-fn-codex-rollout 0.1 (D1)

from yeto.rl.codex_backend import (  # noqa: E402
    QWEN38_NEXT_4LAYER_MODEL,
    QWEN38_NEXT_4LAYER_REVISION,
    QWEN38_NEXT_MODEL,
    QWEN38_NEXT_REVISION,
    stock_codex_lora_layout,
)
from yeto.rl.profiles import qwen3_8_next as fn  # noqa: E402

FN_PROFILES = {
    "qwen38_next": (QWEN38_NEXT_MODEL, QWEN38_NEXT_REVISION, "Qwen3.8-Flash-Next"),
    "qwen38_next_4layer": (
        QWEN38_NEXT_4LAYER_MODEL,
        QWEN38_NEXT_4LAYER_REVISION,
        "Qwen3.8-Flash-Next-4layer",
    ),
}


def _validate_fn(profile_name: str, **overrides):
    model, revision, _ = FN_PROFILES[profile_name]
    values = {
        "tito_model": "qwen4exp",
        "codex_backend_profile": profile_name,
        "rl_model_recipe": "generic",
        "model": model,
        "model_revision": revision,
        "rollout_model": None,
        "rollout_model_revision": None,
        "apply_chat_template_kwargs": copy.deepcopy(QWEN38_KWARGS),
        "tito_allowed_append_roles": ["tool", "user"],
        "codex_reasoning_effort": "xhigh",
        "lora_targets": "all-linear",
        "expert_full_count": 0,
        "lora_expert_rank": 8,
    }
    values.update(overrides)
    return validate_stock_codex_fields(**values)


def test_fn_profiles_pin_identity_from_the_qwen3_8_next_profile_module():
    assert QWEN38_NEXT_MODEL == fn.HF_REPO_FULL == "Qwen/Qwen3.8-Flash-Next"
    assert QWEN38_NEXT_REVISION == fn.HF_REVISION_FULL
    assert QWEN38_NEXT_4LAYER_MODEL == fn.HF_REPO_4LAYER
    assert QWEN38_NEXT_4LAYER_REVISION == fn.HF_REVISION_4LAYER
    for name, (model, revision, label) in FN_PROFILES.items():
        profile = stock_codex_backend_profile(name)
        assert profile["model"] == "qwen4exp" and profile["tito_model"] == "qwen4exp"
        assert profile["model_identifier"] == model
        assert profile["model_revision"] == revision
        assert profile["identity_label"] == label
        assert profile["rl_model_recipe"] == "generic"
        assert profile["chat_template_kwargs"] == QWEN38_KWARGS
        assert profile["tito_allowed_append_roles"] == ["tool", "user"]
        assert profile["lora_targets"] == "all-linear"
        assert profile["lora_expert_rank"] == 8
        assert stock_codex_lora_layout(name) == ("all-linear", 8)
        assert stock_codex_backend_contract(name, 4096)["tito_model"] == "qwen4exp"


@pytest.mark.parametrize("profile_name", sorted(FN_PROFILES))
def test_fn_full_lora_configuration_passes_validation(profile_name):
    # Scenario: FN 全尺寸 LoRA 配置通过校验 (and the 4-layer twin).
    profile = _validate_fn(profile_name)
    assert profile["chat_template_kwargs"] == QWEN38_KWARGS
    assert profile["identity_label"] == FN_PROFILES[profile_name][2]


@pytest.mark.parametrize(
    "overrides",
    [
        {"lora_targets": "attention"},
        {"lora_expert_rank": 0},
        {"lora_expert_rank": 16},
        {"model": QWEN38_MODEL},
        {"model_revision": QWEN38_NEXT_4LAYER_REVISION},
        {"expert_full_count": 1},
    ],
)
def test_fn_lora_target_drift_is_rejected_with_identity_label(overrides):
    # Scenario: LoRA 目标漂移被拒.
    with pytest.raises(ValueError, match="Qwen3.8-Flash-Next model identity drifted"):
        _validate_fn("qwen38_next", **overrides)


def test_fn_profile_requires_the_qwen4exp_tito_family():
    with pytest.raises(ValueError, match="Miles TITO family"):
        _validate_fn("qwen38_next", tito_model="qwen38")


def test_legacy_profiles_keep_attention_zero_layout():
    # Scenario: 旧 profile 不受影响.
    for name in ("deepseekv4", "qwen38", "qwen35", "qwen35_08b"):
        assert stock_codex_lora_layout(name) == ("attention", 0)
        assert "lora_targets" not in stock_codex_backend_profile(name)
    assert _validate_qwen35()["model_identifier"] == QWEN35_MODEL
    assert _validate_qwen35(lora_expert_rank=0)["model_identifier"] == QWEN35_MODEL
    assert validate_stock_codex_fields(
        tito_model="qwen35",
        codex_backend_profile="qwen35_08b",
        rl_model_recipe="generic",
        model=QWEN35_08B_MODEL,
        model_revision=QWEN35_08B_REVISION,
        rollout_model=None,
        rollout_model_revision=None,
        apply_chat_template_kwargs=copy.deepcopy(QWEN35_KWARGS),
        tito_allowed_append_roles=["tool", "user"],
        codex_reasoning_effort="xhigh",
        lora_targets="attention",
        expert_full_count=0,
    )["identity_label"] == "Qwen3.5-0.8B"
    with pytest.raises(ValueError, match="Qwen3.5 model identity"):
        _validate_qwen35(lora_expert_rank=8)
    with pytest.raises(ValueError, match="Qwen3.8 model identity"):
        _validate_qwen38(lora_targets="all-linear")


# ----------------------------------------------------------- rl-fn-codex-rollout 0.1

from yeto.rl.codex_backend import (  # noqa: E402
    QWEN38_NEXT_4LAYER_MODEL,
    QWEN38_NEXT_4LAYER_REVISION,
    QWEN38_NEXT_MODEL,
    QWEN38_NEXT_REVISION,
)
from yeto.rl.profiles import qwen3_8_next as fn_profile  # noqa: E402


def _validate_fn(profile_name="qwen38_next", **overrides):
    model, revision = {
        "qwen38_next": (QWEN38_NEXT_MODEL, QWEN38_NEXT_REVISION),
        "qwen38_next_4layer": (QWEN38_NEXT_4LAYER_MODEL, QWEN38_NEXT_4LAYER_REVISION),
    }[profile_name]
    values = {
        "tito_model": "qwen4exp",
        "codex_backend_profile": profile_name,
        "rl_model_recipe": "generic",
        "model": model,
        "model_revision": revision,
        "rollout_model": None,
        "rollout_model_revision": None,
        "apply_chat_template_kwargs": copy.deepcopy(QWEN38_KWARGS),
        "tito_allowed_append_roles": ["tool", "user"],
        "codex_reasoning_effort": "xhigh",
        "lora_targets": "all-linear",
        "expert_full_count": 0,
        "lora_expert_rank": 8,
    }
    values.update(overrides)
    return validate_stock_codex_fields(**values)


@pytest.mark.parametrize(
    ("profile_name", "label", "repo", "revision"),
    [
        ("qwen38_next", "Qwen3.8-Flash-Next", fn_profile.HF_REPO_FULL, fn_profile.HF_REVISION_FULL),
        ("qwen38_next_4layer", "Qwen3.8-Flash-Next-4layer", fn_profile.HF_REPO_4LAYER, fn_profile.HF_REVISION_4LAYER),
    ],
)
def test_flash_next_profiles_share_qwen4exp_tito_and_pin_fn_constants(profile_name, label, repo, revision):
    profile = stock_codex_backend_profile(profile_name)
    assert profile["model"] == "qwen4exp" and profile["tito_model"] == "qwen4exp"
    assert profile["model_identifier"] == repo and profile["model_revision"] == revision
    assert profile["identity_label"] == label == fn_profile.MODEL_NAMES["full" if profile_name == "qwen38_next" else "4layer"]
    assert profile["chat_template_kwargs"] == QWEN38_KWARGS
    assert profile["tito_allowed_append_roles"] == ["tool", "user"]
    assert profile["lora_targets"] == "all-linear" and profile["lora_expert_rank"] == 8
    assert stock_codex_backend_contract(profile_name, 4096) == {
        "profile": profile_name,
        "tito_model": "qwen4exp",
        "model": "qwen4exp",
        "max_tokens": 4096,
        "reasoning_effort": "xhigh",
        "thinking": {"type": "enabled"},
        "chat_template": profile_name,
        "chat_template_kwargs": QWEN38_KWARGS,
        "tito_allowed_append_roles": ["tool", "user"],
    }
    # Scenario: FN LoRA configuration passes and kwargs equal the declaration.
    validated = _validate_fn(profile_name)
    assert validated["model_identifier"] == repo
    assert validated["chat_template_kwargs"] == QWEN38_KWARGS


def test_flash_next_profile_is_distinct_from_dense_qwen38():
    dense = stock_codex_backend_profile("qwen38")
    assert "lora_targets" not in dense and "lora_expert_rank" not in dense
    assert stock_codex_backend_profile("qwen38_next")["model_identifier"] != dense["model_identifier"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"lora_targets": "attention"},  # Scenario: LoRA target drift rejected
        {"lora_expert_rank": 0},
        {"lora_expert_rank": 16},
        {"model": QWEN38_MODEL},
        {"model_revision": "0" * 40},
        {"model": QWEN38_NEXT_4LAYER_MODEL, "model_revision": QWEN38_NEXT_4LAYER_REVISION},
        {"rollout_model": QWEN38_NEXT_4LAYER_MODEL},
        {"expert_full_count": 1},
    ],
)
def test_flash_next_identity_drift_fails_closed_with_fn_label(overrides):
    with pytest.raises(ValueError, match="Qwen3.8-Flash-Next model identity"):
        _validate_fn(**overrides)


def test_flash_next_profile_requires_qwen4exp_tito_family():
    with pytest.raises(ValueError, match="Miles TITO family"):
        _validate_fn(tito_model="qwen38")


def test_legacy_profiles_default_to_attention_and_zero_expert_rank():
    # Scenario: old profiles are unaffected (attention / expert rank 0 passes, both
    # with the implicit default and an explicit 0).
    assert _validate_qwen38()["identity_label"] == "Qwen3.8"
    assert _validate_qwen38(lora_expert_rank=0)["identity_label"] == "Qwen3.8"
    assert _validate_qwen35(lora_expert_rank=None)["identity_label"] == "Qwen3.5"
    with pytest.raises(ValueError, match="Qwen3.8 model identity"):
        _validate_qwen38(lora_expert_rank=8)
    with pytest.raises(ValueError, match="Qwen3.5 model identity"):
        _validate_qwen35(lora_targets="all-linear")
