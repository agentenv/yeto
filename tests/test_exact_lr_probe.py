import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_exact_lr_probe", ROOT / "scripts" / "replay_exact_lr_probe.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_selector_falls_back_to_multiplier_one_when_no_action_passes():
    losses = {
        0.5: [1.001] * 8,
        0.75: [1.0005] * 8,
        1.0: [1.0] * 8,
        1.25: [1.0001] * 8,
    }

    result = MOD.select_multiplier(losses, tuple(losses))

    assert result["chosen_multiplier"] == 1.0
    assert result["fallback_reason"] == "no_action_passed"


def test_selector_tie_breaking_is_deterministic_and_conservative():
    losses = {
        1.25: [0.999] * 8,
        1.0: [1.0] * 8,
        0.75: [0.999] * 8,
        1.5: [1.001] * 8,
    }

    forward = MOD.select_multiplier(losses, (1.25, 1.0, 0.75, 1.5))
    reverse = MOD.select_multiplier(
        dict(reversed(list(losses.items()))), (1.5, 0.75, 1.0, 1.25)
    )

    assert forward["chosen_multiplier"] == 0.75
    assert reverse["chosen_multiplier"] == 0.75
    assert forward["fallback_reason"] is None


def test_anchor_oracle_disjointness_metadata_reports_canonical_overlap():
    disjoint = MOD.anchor_oracle_disjointness_metadata(
        ("anchor-a", "anchor-b"), ("oracle-a", "oracle-b")
    )
    overlap = MOD.anchor_oracle_disjointness_metadata(
        ("anchor-a", "shared"), ("shared", "oracle-b")
    )

    assert disjoint == {
        "canonicalization": MOD.CANONICALIZATION,
        "anchor_row_count": 2,
        "anchor_unique_row_count": 2,
        "oracle_row_count": 2,
        "oracle_unique_row_count": 2,
        "overlap_count": 0,
        "verified_disjoint": True,
        "overlap_sha256": MOD.hashlib.sha256(b"").hexdigest(),
    }
    assert overlap["verified_disjoint"] is False
    assert overlap["overlap_count"] == 1


def test_paired_statistics_fail_closed_on_invalid_series():
    with pytest.raises(ValueError, match="equal length"):
        MOD.paired_decision(
            [1.0, 1.1], [1.0], min_gain=0.0, lcb_z=0.0, min_win_rate=0.0
        )
    with pytest.raises(ValueError, match="invalid value"):
        MOD.paired_decision(
            [1.0, float("nan")],
            [0.9, 0.9],
            min_gain=0.0,
            lcb_z=0.0,
            min_win_rate=0.0,
        )


def test_target_block_selection_skips_prompt_only_prefix_blocks():
    weights = MOD.torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )

    assert MOD.target_block_indices(weights, 2) == [1, 3]
    with pytest.raises(ValueError, match="only 2"):
        MOD.target_block_indices(weights, 3)


def _action(multiplier, gain):
    return {
        "multiplier": multiplier,
        "oracle_gain_vs_baseline": gain,
        "step_norm_ratio": multiplier,
    }


def test_summary_aggregates_chosen_fixed_and_oracle_policy_math():
    records = [
        {
            "seed": 223,
            "chosen_multiplier": 0.5,
            "chosen_oracle_gain_vs_baseline": 0.2,
            "step_norm_ratio": 0.5,
            "best_oracle_multiplier": 0.5,
            "best_oracle_gain_vs_baseline": 0.2,
            "selection": {"fallback_reason": None},
            "actions": [
                _action(0.5, 0.2),
                _action(1.0, 0.0),
                _action(1.5, 0.1),
            ],
        },
        {
            "seed": 239,
            "chosen_multiplier": 1.0,
            "chosen_oracle_gain_vs_baseline": 0.0,
            "step_norm_ratio": 1.0,
            "best_oracle_multiplier": 1.5,
            "best_oracle_gain_vs_baseline": 0.3,
            "selection": {"fallback_reason": "no_action_passed"},
            "actions": [
                _action(0.5, -0.1),
                _action(1.0, 0.0),
                _action(1.5, 0.3),
            ],
        },
    ]

    summary = MOD.summarize(records)

    assert summary["records"] == 2
    assert summary["mean_chosen_oracle_gain_vs_baseline"] == pytest.approx(0.1)
    assert summary["mean_step_norm_ratio"] == pytest.approx(0.75)
    assert summary["selection_rate"] == pytest.approx(0.5)
    assert summary["fallback_rate"] == pytest.approx(0.5)
    assert summary["best_fixed_multiplier"] == 1.5
    assert summary["best_fixed_mean_oracle_gain_vs_baseline"] == pytest.approx(0.2)
    assert summary["mean_best_oracle_gain_vs_baseline"] == pytest.approx(0.25)
    assert summary["oracle_headroom_captured"] == pytest.approx(0.4)
    assert summary["chosen_multiplier_distribution"] == {"0.5": 1, "1": 1}
    assert summary["best_oracle_multiplier_distribution"] == {"0.5": 1, "1.5": 1}
