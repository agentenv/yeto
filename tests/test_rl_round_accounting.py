"""INFRA channels for 1b/2a: per-round trained counts, GSPO clip fraction, non-zero advantages."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook
from yeto.rl.engine.miles_adapter import state_plugin
from yeto.rl.engine.miles_adapter import trainer as tr
from yeto.rl.engine.miles_adapter.rollout import handle_from_metadata

H = "a" * 64


def test_step_loss_recorder_keeps_clipfrac_and_skips_non_last_stage():
    state_plugin._STEP_LOSSES.clear()
    state_plugin._record_step_losses(({"train/pg_clipfrac": 0.25, "train/loss": 1.0}, 0.5, "ok"))
    state_plugin._record_step_losses(({}, 0.5, "ok"))  # not last PP stage
    state_plugin._record_step_losses(({"train/loss": 1.0}, 0.5, "ok"))
    assert state_plugin.step_losses(None) == [
        {"pg_clipfrac": 0.25, "loss_tokens": None},
        {"pg_clipfrac": None, "loss_tokens": None},
    ]
    assert state_plugin.step_losses(None) == []


def test_gspo_masked_fraction_uses_optional_seq_adv(monkeypatch):
    losses = [{"pg_clipfrac": 1.0, "loss_tokens": None}]
    monkeypatch.setitem(sys.modules, "yeto.rl.algos.seq_adv", None)  # module absent
    assert tr.clipfrac_masked_fraction(losses) is None
    fake = types.ModuleType("yeto.rl.algos.seq_adv")
    seen = {}

    def clipfrac_from_losses(step_losses, token_counts=None):
        seen["args"] = (step_losses, token_counts)
        return 1.0

    fake.clipfrac_from_losses = clipfrac_from_losses
    monkeypatch.setitem(sys.modules, "yeto.rl.algos.seq_adv", fake)
    assert tr.clipfrac_masked_fraction(losses) == 1.0
    assert seen["args"] == (losses, None)
    assert tr._mean_clipfrac([{"pg_clipfrac": 0.2}, {"pg_clipfrac": 0.4}]) == pytest.approx(0.3)
    assert tr._mean_clipfrac([{"pg_clipfrac": None}]) is None


def _group():
    return {"group_id": "g0", "sample_ids": ["s0"], "policy_token": "t", "reward_mean": 0.0,
            "reward_std": 0.0, "token_count": 1}


def test_nonzero_advantages_travel_through_rollout_metadata():
    args = SimpleNamespace(**{hook.ROUND_METADATA_ATTR: {"nonzero_advantages": 7}})
    assert hook._round_metadata(args) == {"nonzero_advantages": 7}
    assert hook._round_metadata(SimpleNamespace()) == {}  # default: key set unchanged
    with pytest.raises(RuntimeError, match="unknown"):
        hook._round_metadata(SimpleNamespace(**{hook.ROUND_METADATA_ATTR: {"x": 1}}))
    with pytest.raises(RuntimeError, match="non-negative"):
        hook._round_metadata(SimpleNamespace(**{hook.ROUND_METADATA_ATTR: {"nonzero_advantages": -1}}))
    payload = {"schema": hook.METADATA_SCHEMA, "rollout_id": 3, "groups": [_group()],
               "completed": 1, "aborted": 0, "nonzero_advantages": 7}
    h = handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash=H, data_pack=None)
    assert h.nonzero_advantages == 7
    payload.pop("nonzero_advantages")
    assert handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash=H,
                                data_pack=None).nonzero_advantages is None
