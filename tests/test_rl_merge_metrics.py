"""rl-trainer-start-stop L5.5 / L5.7: merge metrics folded from the elastic
syncer tape, and the tape clock fields. No Ray, no GPU."""
from __future__ import annotations

import json

import pytest

from yeto.rl.engine.merge_metrics import MergeMetricsFolder
from yeto.wandb_tape import record_metrics, tape_metrics


def _fold(records):
    folder = MergeMetricsFolder()
    return [m for m in (folder.feed(r) for r in records) if m is not None]


def test_folder_hand_computed_shares_carry_and_rejections():
    # Round v1: island 1 on time (w = 10^2/1 = 100), island 3 late by one
    # (w = 30^2/1 * 0.5 = 450), island 9 not a member, island 2 unknown version.
    records = [
        {"kind": "delta_carried_over", "island_id": 3, "base": 0, "lag": 1, "discount": 0.5},
        {"kind": "delta_accepted", "island_id": 1},
        {"kind": "delta_rejected", "island_id": 9, "reason": "not_member"},
        {"kind": "delta_rejected", "island_id": 2, "reason": "unknown_version"},
        {"kind": "outer_step", "base_version": 1, "arrived": 1, "timed_out": True,
         "raw_weights": [[1, 1, 100.0], [3, 0, 450.0]],
         "weights": [[1, 1, 100 / 550], [3, 0, 450 / 550]], "absent": [2, 3],
         "rejected": [[9, "not_member"], [2, "unknown_version"]],
         "time_unix": 1000.0, "step_seconds": 4.0, "last_arrival_island": 1},
        {"kind": "delta_accepted", "island_id": 1},
        {"kind": "outer_step", "base_version": 2, "arrived": 1, "timed_out": False,
         "raw_weights": [[1, 2, 100.0]], "weights": [[1, 2, 1.0]], "absent": [], "rejected": []},
    ]
    first, second = _fold(records)
    assert first["global_step"] == 2 and first["merge/base_version"] == 1
    assert first["merge/island/1/weight_share"] == pytest.approx(100 / 550)
    assert first["merge/island/3/weight_share"] == pytest.approx(450 / 550)
    assert first["merge/carried_weight_share"] == pytest.approx(450 / 550)
    assert first["merge/submitted"] == 4
    assert first["merge/carried_fraction"] == 0.25
    assert first["merge/carried_discount_mean"] == 0.5
    assert first["merge/rejected_fraction"] == 0.5
    assert first["merge/rejected_fraction/not_member"] == 0.25
    assert first["merge/rejected_fraction/unknown_version"] == 0.25
    assert first["merge/step_seconds"] == 4.0 and first["merge/last_arrival_island"] == 1
    # The folder starts again after each step; absent values stay absent.
    assert second["merge/submitted"] == 1 and second["merge/rejected_fraction"] == 0.0
    assert second["merge/carried_weight_share"] == 0.0
    assert "merge/carried_discount_mean" not in second and "merge/step_seconds" not in second


def test_old_tape_without_rejected_list_counts_rejection_events():
    records = [
        {"kind": "delta_accepted", "island_id": 1},
        {"kind": "delta_rejected", "island_id": 2, "reason": "carry_lag_exceeded"},
        {"kind": "outer_step", "base_version": 0, "arrived": 1, "timed_out": False,
         "raw_weights": [[1, 0, 1.0]], "weights": [[1, 0, 1.0]], "absent": [2]},
    ]
    (m,) = _fold(records)
    assert m["merge/rejected_fraction/carry_lag_exceeded"] == 0.5


def test_wandb_routes_elastic_records_through_the_folder_only():
    folder = MergeMetricsFolder()
    assert record_metrics({"kind": "pool_join", "island_id": 1}, folder) is None
    assert record_metrics({"kind": "delta_accepted", "island_id": 1}, folder) is None
    step = {"kind": "outer_step", "base_version": 0, "weights": [[1, 0, 1.0]]}
    assert record_metrics(step, folder)["merge/island/1/weight_share"] == 1.0
    strict = {"step": 3, "responders": []}
    assert record_metrics(strict, folder) == tape_metrics(strict)
