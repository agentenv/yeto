"""rl-algo-supplement 2.6: dual_clipfrac and the over-sampling tally on the ALGO side."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles.rollout import handle_from_metadata
from yeto.rl.engine.fake import FakeEngine
from yeto.rl.engine.ports import OVER_SAMPLING_KEYS, over_sampling_fields

from tests.test_rl_driver_profiles import _driver, _events, _profile
from tests.test_rl_miles_adapter_rollout import H, group

STATS = {"submit_calls": 3, "refill_calls": 2, "submitted_groups": 12, "submit_batch_size_first": 4,
         "submit_batch_size_max": 4, "completed_groups": 9, "filtered_groups": 5, "kept_groups": 2,
         "surplus_groups": 2, "inflight_groups_at_end": 3, "resumed_groups": 0, "failed_groups": 0}


def test_over_sampling_fields_validates():
    assert set(STATS) == set(OVER_SAMPLING_KEYS)
    assert over_sampling_fields(STATS) == STATS
    assert over_sampling_fields(None) is None
    assert over_sampling_fields({**STATS, "kept_groups": -1}) is None
    assert over_sampling_fields({**STATS, "kept_groups": 1.0}) is None
    assert over_sampling_fields({k: v for k, v in STATS.items() if k != "refill_calls"}) is None
    # extra keys are dropped, not passed through
    assert over_sampling_fields({**STATS, "x": 1}) == STATS


@pytest.mark.parametrize("stats", [STATS, None])
def test_fork_over_sampling_tally_reaches_the_handle(tmp_path, stats):
    kept = [group(0, [1.0, 0.0]), group(1, [0.0, 1.0])]
    args = SimpleNamespace()
    if stats is not None:
        args.rollout_over_sampling_stats = dict(stats)
    hook.record_trained_groups(args, kept)
    os.environ[hook.META_SINK_ENV] = f"dir:{tmp_path}"
    try:
        hook.extract_rollout_metadata(args, kept, SimpleNamespace(sample_group_index=6))
    finally:
        os.environ.pop(hook.META_SINK_ENV)
    payload = json.loads((tmp_path / "rollout-3.json").read_text())
    h = handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash=H, data_pack=None)
    if stats is None:  # older image: unknown, no field
        assert "over_sampling" not in payload and h.over_sampling is None
    else:
        assert args.rollout_over_sampling_stats is None  # consumed once per rollout
        assert payload["over_sampling"] == STATS and h.over_sampling == STATS


def test_fake_over_sampled_batch_reports_the_tally():
    e = FakeEngine(tensors={"w": torch.zeros(2)}, groups=2, over_sampling_groups=5)
    rollout = e.rollout
    batch = rollout._generate_over_sampled(0, 0, "d", "t")
    o = batch.over_sampling
    assert over_sampling_fields(o) == o
    assert o["submitted_groups"] == 5 == batch.submitted_groups
    assert o["kept_groups"] == len(batch.groups) == 2
    assert o["inflight_groups_at_end"] == batch.aborted_in_flight_groups == 3
    assert (o["submitted_groups"] + o["resumed_groups"]
            == o["completed_groups"] + o["failed_groups"] + o["inflight_groups_at_end"])


def test_fake_dual_clipfrac_reaches_rl_round_trained(tmp_path):
    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, dual_clipfrac=0.125)
    driver, _ = _driver(engine, tmp_path, profile=_profile("colocated-serial"), observe=True)
    driver.run()
    trained = [e for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_round_trained"]
    assert trained and all(e["train_metrics"]["dual_clipfrac"] == 0.125 for e in trained)


def test_fake_default_reports_no_dual_clipfrac(tmp_path):
    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)}, step_delta=1.0)
    driver, _ = _driver(engine, tmp_path, profile=_profile("colocated-serial"), observe=True)
    driver.run()
    trained = [e for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_round_trained"]
    assert trained and all("dual_clipfrac" not in e["train_metrics"] for e in trained)
