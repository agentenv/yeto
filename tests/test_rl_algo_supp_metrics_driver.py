"""rl-algo-supplement 2.6 (INFRA patch algo-supp-metrics.patch): the driver puts the
engine's over-sampling tally into rl_round_labels as rl/over_sampling/*."""

from __future__ import annotations

import torch

from yeto.rl.engine.fake import FakeEngine
from yeto.rl.engine.ports import OVER_SAMPLING_KEYS

from tests.test_rl_driver_profiles import _driver, _events, _profile

NAME = "base_model.model.layer.lora_A.weight"


def _labels(tmp_path, **kw):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0, **kw)
    driver, _ = _driver(engine, tmp_path, profile=_profile("colocated-serial"), observe=True)
    driver.run()
    return [e for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_round_labels"]


def test_over_sampling_tally_in_round_labels(tmp_path):
    labels = _labels(tmp_path, groups=2, over_sampling_groups=5)
    assert labels
    for e in labels:
        assert {f"rl/over_sampling/{k}" for k in OVER_SAMPLING_KEYS} <= set(e)
        assert e["rl/over_sampling/submitted_groups"] == 5
        assert e["rl/over_sampling/kept_groups"] == e["rl/groups"]


def test_no_over_sampling_no_field(tmp_path):
    labels = _labels(tmp_path)
    assert labels and not any(k.startswith("rl/over_sampling/") for e in labels for k in e)
