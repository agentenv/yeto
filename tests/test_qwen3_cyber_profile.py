"""rlf-codex-rl 5.6: Qwen3-30B-A3B cyber baseline profile (CPU-only contracts).

Covers the S21 audit gaps for task 5.6: the profile passes the closed
allowlist checks, and the plan budgets (thinking mode, 32K window, 1024
tokens/turn, 48 turns, 8192-byte tool output) are registered values rather
than unsigned defaults.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from yeto.rl import codex_backend as cb
from yeto.rl.codex_backend import (
    QWEN3_30B_A3B_MODEL,
    QWEN3_30B_A3B_REVISION,
    stock_codex_backend_contract,
    stock_codex_backend_profile,
    stock_codex_keeps_history_reasoning,
    stock_codex_reasoning_markers,
    validate_stock_codex_fields,
)
from yeto.rl.harness.codex import codex_openenv_agent_function as adapter
from yeto.rl.harness.codex import pins

PROFILE = "qwen3_30b_a3b"

QWEN3_KWARGS = {
    "enable_thinking": True,
    "preserve_thinking": True,
    "reasoning_effort": "xhigh",
}


def test_profile_is_allowlisted_and_resolvable():
    assert PROFILE in pins.OPENENV_BACKEND_PROFILES
    assert adapter.profile_identity(PROFILE) == (
        QWEN3_30B_A3B_MODEL,
        QWEN3_30B_A3B_REVISION,
    )
    assert adapter.resolve_backend_profile(PROFILE, env={}) == PROFILE
    assert (
        adapter.resolve_backend_profile(env={adapter.RUNTIME_PROFILE_ENV: PROFILE})
        == PROFILE
    )
    # every runtime profile's chat template equals its profile name
    assert stock_codex_backend_contract(PROFILE, 1)["chat_template"] == PROFILE


def test_thinking_mode_and_pinned_identity():
    profile = stock_codex_backend_profile(PROFILE)
    assert profile["thinking"] == {"type": "enabled"}
    assert profile["chat_template_kwargs"] == QWEN3_KWARGS
    assert profile["model_identifier"] == "Qwen/Qwen3-30B-A3B"
    assert profile["model_revision"] == QWEN3_30B_A3B_REVISION
    assert len(profile["model_revision"]) == 40
    assert profile["identity_label"] == "Qwen3-30B-A3B"
    assert stock_codex_keeps_history_reasoning(PROFILE) is True
    assert stock_codex_reasoning_markers(PROFILE) == ("<think>", "</think>")


def test_budget_fields_match_the_5_6_plan():
    profile = stock_codex_backend_profile(PROFILE)
    assert profile["max_seq_len"] == 32_768  # 32K single-window baseline
    assert profile["max_tokens_per_turn"] == 1_024
    assert profile["max_turns"] == 48
    assert profile["tool_output_bytes"] == 8_192


def _validate_qwen3(**overrides):
    values = {
        "tito_model": "qwen3",
        "codex_backend_profile": PROFILE,
        "rl_model_recipe": "generic",
        "model": QWEN3_30B_A3B_MODEL,
        "model_revision": QWEN3_30B_A3B_REVISION,
        "rollout_model": None,
        "rollout_model_revision": None,
        "apply_chat_template_kwargs": dict(QWEN3_KWARGS),
        "tito_allowed_append_roles": ["tool", "user"],
        "codex_reasoning_effort": "xhigh",
        "lora_targets": "attention",
        "expert_full_count": 0,
    }
    values.update(overrides)
    return validate_stock_codex_fields(**values)


def test_validate_stock_codex_fields_accepts_then_rejects_drift():
    assert _validate_qwen3()["identity_label"] == "Qwen3-30B-A3B"
    with pytest.raises(ValueError, match="Qwen3-30B-A3B model identity drifted"):
        _validate_qwen3(model_revision="0" * 40)
    with pytest.raises(ValueError, match="Qwen3-30B-A3B model identity drifted"):
        _validate_qwen3(rollout_model="Qwen/Qwen3-32B")


def _harness_tool_output_bytes(env_extra: dict[str, str]) -> int:
    """Resolve MAX_TOOL_OUTPUT_BYTES in a child interpreter (it is bound at
    import time from the profile named by YETO_CODEX_CHAT_TEMPLATE)."""
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("YETO_CODEX_CHAT_TEMPLATE", None)
    env.update(env_extra)
    env["PYTHONPATH"] = str(repo)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from yeto.rl.harness.codex import codex_harness_agent as h;"
            " print(h.MAX_TOOL_OUTPUT_BYTES)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def test_tool_output_budget_is_profile_driven():
    assert _harness_tool_output_bytes({"YETO_CODEX_CHAT_TEMPLATE": PROFILE}) == 8_192
    # profiles without a declaration keep the signed 32 KiB default
    assert _harness_tool_output_bytes({"YETO_CODEX_CHAT_TEMPLATE": "qwen35_08b"}) == 32_768
    assert _harness_tool_output_bytes({}) == 32_768


def test_existing_profiles_are_unchanged():
    for name in ("qwen35_08b", "qwen35", "qwen38_next", "qwen38_next_4layer"):
        profile = cb.stock_codex_backend_profile(name)
        assert "tool_output_bytes" not in profile, name
        assert "max_turns" not in profile, name
        assert cb.stock_codex_keeps_history_reasoning(name) is True
