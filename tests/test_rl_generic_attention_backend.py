"""The generic recipe trains with cuDNN fused attention by default.

Unfused attention materializes the S x S score matrix: a 32k-token packed
sequence asked for ~47 GiB in one baddbmm and OOMed on the first step of a
1xH100 Qwen3-4B run.  Deterministic trainers keep unfused (Transformer Engine
drops cuDNN fused attention under NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 on
pre-Hopper cards)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from yeto.rl.engine import run_config as rc

GOLDEN = Path(__file__).parent / "golden" / "decoupling"


def test_generic_recipe_defaults_to_fused_attention():
    assert rc.generic_attention_backend(SimpleNamespace()) == "fused"
    assert rc.generic_attention_backend(SimpleNamespace(rl_deterministic_trainer=False)) == "fused"


def test_deterministic_trainer_keeps_unfused_attention():
    assert rc.generic_attention_backend(SimpleNamespace(rl_deterministic_trainer=True)) == "unfused"


def test_generic_golden_argv_carries_fused_attention():
    # these decoupling goldens are generic-recipe (Qwen3) runs without --deterministic
    for name in ("grpo_default", "grpo_tis", "decoupled", "drgrpo", "seq_adv_maxrl", "elastic"):
        argv = json.loads((GOLDEN / f"{name}.json").read_text())["miles_argv"]
        assert argv[argv.index("--attention-backend") + 1] == "fused", name


def test_harness_pins_no_nvte_attention_env():
    # Megatron sets NVTE_*_ATTN from --attention-backend and asserts on a pin that disagrees
    source = Path(rc.__file__).parents[1].joinpath("ssh_harness.py").read_text()
    assert "NVTE_UNFUSED_ATTN=1" not in source
