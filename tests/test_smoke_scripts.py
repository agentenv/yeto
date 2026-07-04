"""Pure-logic tests for the smoke/comparison harnesses (nothing launches)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


smoke = _load("smoke_models")
compare = _load("compare_diloco")


def test_every_alias_has_a_tier():
    from yeto.models import MODEL_ALIASES

    for alias in MODEL_ALIASES:
        assert smoke.tier_of(alias) in smoke.TIERS


def test_tier_selection_is_cumulative_and_ordered():
    args = SimpleNamespace(only=None, skip=None, tier="small")
    small = smoke.select_models(args)
    assert small, "small tier should not be empty"
    assert "lfm25-230m" in small
    from yeto.models import MODEL_WEIGHT_GB

    sizes = [MODEL_WEIGHT_GB.get(a) for a in small]
    assert all(s is not None and s <= smoke.TIERS["small"] for s in sizes)
    assert sizes == sorted(sizes), "cheapest models must run first"
    args_med = SimpleNamespace(only=None, skip=None, tier="medium")
    assert set(small) <= set(smoke.select_models(args_med))


def test_only_and_skip():
    args = SimpleNamespace(only="lfm25-230m,qwen35-4b", skip="qwen35-4b", tier="small")
    assert smoke.select_models(args) == ["lfm25-230m"]
    import pytest

    with pytest.raises(SystemExit):
        smoke.select_models(SimpleNamespace(only="not-a-model", skip=None, tier="small"))


def test_run_names_are_cluster_safe():
    from yeto.models import MODEL_ALIASES

    for alias in MODEL_ALIASES:
        name = smoke.run_name(alias)
        assert all(c.isalnum() or c == "-" for c in name), name
        assert name.startswith("smk-")


def test_launch_command_uses_auto_planner():
    args = SimpleNamespace(
        data="org/ds", budget=15.0, total_steps=8, fragments=4,
        seq_len=1024, max_rows=512,
    )
    cmd = smoke.launch_command("lfm25-230m", args)
    assert "--gpu" not in cmd, "auto smoke must let the shape planner pick the fleet"
    assert "--budget" in cmd and "--confirm" in cmd
    # auto knobs stay at their defaults: no explicit micro-batch/lora-targets
    assert "--micro-batch-size" not in cmd and "--lora-targets" not in cmd


def test_compare_presets_cover_the_paper_axes():
    arms = compare.select_arms("all")
    names = {a.name for a in arms}
    assert {"m2", "m4", "alpha0", "q4", "serial", "noheloco", "strided"} <= names
    assert compare.PRESETS["alpha0"].merge_alpha == 0.0
    assert compare.PRESETS["serial"].pipeline == 1
    assert compare.PRESETS["q4"].wire_dtype == "q4"


def test_token_budget_split_is_fair_across_learners():
    # Same budget, M learners -> per-learner steps shrink by ~M.
    b1 = compare.steps_for(1_000_000, 2, 512, 1)
    b4 = compare.steps_for(1_000_000, 2, 512, 4)
    assert b1 == 977 and b4 == 245  # ceil semantics
    assert compare.steps_for(1, 2, 512, 8) == 1  # never zero steps


def test_learner_command_arm_overrides():
    args = SimpleNamespace(
        model="lfm25-230m", lora_r=16, lora_alpha=32, seq_len=512,
        micro_batch_size=2, inner_lr=3e-4, device="cpu",
    )
    arm = compare.PRESETS["q4"]
    cmd = compare.learner_command(
        args, Path("/tmp/w/q4"), learner_id=1, num_learners=2,
        syncer="127.0.0.1:1", max_steps=10, arm=arm,
    )
    assert cmd[cmd.index("--wire-dtype") + 1] == "q4"
    assert cmd[cmd.index("--data") + 1].endswith("train.jsonl")
    base = compare.learner_command(
        args, Path("/tmp/w/baseline"), learner_id=0, num_learners=1,
        syncer="none", max_steps=10, arm=None,
    )
    assert "--wire-dtype" not in base  # baseline never touches sync knobs


def test_syncer_command_quorum_defaults_to_all_learners():
    arm = compare.PRESETS["m4"]
    cmd = compare.syncer_command(arm, 1234, Path("/tmp/w/m4"), total_steps=100)
    assert cmd[cmd.index("--quorum") + 1] == "4"
    assert cmd[cmd.index("--checkpoint-every") + 1] == "1"
    assert cmd[cmd.index("--pipeline") + 1] == "2"
