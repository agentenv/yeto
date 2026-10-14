"""Island-entry wiring of DeepSeek V4 expert-full on ports (P0 1.7b.2). No Miles, no GPU."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import island_entry
from yeto.rl.core import canonical_lora_config_hash

_DSV4_ENV = (
    "YETO_DSV4_EXPERT_CLONE", "YETO_DSV4_CLONE_ONLY_LORA", "YETO_DSV4_EXPERT_FULL",
    "YETO_DSV4_EXPERT_FULL_PORTS", "YETO_DSV4_EXPERT_FULL_COUNT", "YETO_DSV4_EXPERT_FULL_LR",
    "NVTE_GROUPED_LINEAR_SINGLE_PARAM", "SGLANG_OPT_FUSE_WQA_WKV",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in _DSV4_ENV:
        # setenv first so monkeypatch restores the variable the code under test writes
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    return monkeypatch


def _args(**overrides):
    values = dict(
        lora_r=8, expert_full_count=4, expert_full_lr=1e-6,
        expert_selection_sha256="1" * 64, expert_selection_contract_sha256="2" * 64,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(("engine", "runtime", "other"), [
    ("ports", "YETO_DSV4_EXPERT_FULL_PORTS", "YETO_DSV4_EXPERT_FULL"),
    ("legacy", "YETO_DSV4_EXPERT_FULL", "YETO_DSV4_EXPERT_FULL_PORTS"),
])
def test_expert_full_env_selects_the_runtime_of_the_engine(clean_env, engine, runtime, other):
    island_entry._configure_deepseek_v4_env(_args(), rl_engine=engine, clone_only_lora=False, expert_full=True)

    assert os.environ[runtime] == "1"
    assert other not in os.environ
    assert os.environ["YETO_DSV4_EXPERT_FULL_COUNT"] == "4"
    assert os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] == "0"


def test_expert_full_env_refuses_the_other_layouts_runtime(clean_env):
    clean_env.setenv("YETO_DSV4_EXPERT_FULL", "1")
    with pytest.raises(ValueError, match="other Miles layout"):
        island_entry._configure_deepseek_v4_env(_args(), rl_engine="ports", clone_only_lora=False, expert_full=True)


def test_plain_lora_sets_no_deepseek_v4_switch(clean_env):
    island_entry._configure_deepseek_v4_env(_args(), rl_engine="ports", clone_only_lora=False, expert_full=False)
    assert not any(name in os.environ for name in _DSV4_ENV)


def test_lora_identity_hash_tells_expert_selections_apart():
    targets = ["linear_qkv"]
    plain = island_entry._lora_config_hash(_args(expert_full_count=0), targets)
    assert plain == canonical_lora_config_hash(rank=8, target_modules=targets)  # unchanged
    hashes = {
        island_entry._lora_config_hash(_args(**change), targets)
        for change in ({}, {"expert_full_count": 5}, {"expert_full_lr": 2e-6},
                       {"expert_selection_sha256": "3" * 64})
    }
    assert len(hashes) == 4 and plain not in hashes


def test_dense_training_contract_has_no_expert_section():
    assert island_entry._expert_full_contract(_args(expert_full_count=0)) == {}
    assert island_entry._expert_full_contract(_args())["expert_full"]["count"] == 4
