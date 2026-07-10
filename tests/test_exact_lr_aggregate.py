import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aggregate_exact_lr_probe", ROOT / "scripts" / "aggregate_exact_lr_probe.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def _action(multiplier, anchor, oracle_loss, oracle_utility):
    return {
        "multiplier": multiplier,
        "anchor_panel_losses": anchor,
        "oracle_loss": oracle_loss,
        "oracle_utility": oracle_utility,
        "oracle_utility_se": 0.01,
        "oracle_gain_vs_baseline": 1.0 - oracle_loss,
    }


def _record(seed=1):
    actions = [
        _action(1.0, [1.0, 1.0, 1.0, 1.0], 1.0, 0.1),
        _action(1.25, [0.9, 0.9, 0.9, 0.9], 0.8, 0.3),
        _action(1.5, [0.95, 0.95, 0.95, 0.95], 0.9, 0.2),
    ]
    return {
        "seed": seed,
        "step": 1,
        "fragment": 0,
        "anchor_current_panel_losses": [1.2, 1.2, 1.2, 1.2],
        "baseline_oracle_utility": 0.1,
        "actions": actions,
    }


def test_risk_aware_rule_uses_largest_confident_positive_step():
    result = MOD.replay_rule(
        [_record()],
        family="positive_utility_largest",
        fallback=1.0,
        min_gain=0.0,
        z=0.0,
        min_win_rate=1.0,
    )

    assert result["chosen_multiplier_distribution"] == {"1.5": 1}
    assert result["mean_gain_vs_multiplier_1"] == pytest.approx(0.1)


def test_shrink_rule_can_beat_maximum_fallback():
    result = MOD.replay_rule(
        [_record()],
        family="shrink_from_fallback",
        fallback=1.5,
        min_gain=0.0,
        z=0.0,
        min_win_rate=1.0,
    )

    assert result["chosen_multiplier_distribution"] == {"1.25": 1}
    assert result["mean_gain_vs_fallback"] == pytest.approx(0.1)
