"""Per-algorithm zero-gradient invariant (rl-algorithm-capabilities 4.1/4.2).

Ships with ``p0-driver.patch`` (driver.py/trainer.py owned by INFRA): needs
``TrainStepMetrics.masked_fraction`` and ``_check_gradient`` calling
``AlgorithmSpec.expects_gradient``.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.core import StrictRlInvariantError
from yeto.rl.engine import algorithm as alg
from yeto.rl.engine.algorithm import AlgorithmSpec, LossSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import EventTape, IslandDriver, TrainStepMetrics
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.miles_adapter.trainer import masked_fraction

NAME = "base_model.model.layer.lora_A.weight"
MASKING = AlgorithmSpec(loss=LossSpec(eps_clip=0.123))


@pytest.fixture
def masking_mechanism():
    alg.register_mechanism("features", "test_full_mask", lambda s: s.loss.eps_clip == 0.123,
                           masks_tokens=True)
    yield
    alg.unregister(mechanism=("features", "test_full_mask"))


def _run(tmp_path, algorithm, *, features=frozenset(), **engine_kwargs):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, **engine_kwargs)
    driver = IslandDriver(
        learner_id=0,
        rollout=engine.rollout,
        trainer=engine.trainer,
        policy_state=engine.policy_state,
        publisher=engine.publisher,
        placement=engine.placement,
        algorithm=algorithm,
        sync=LocalOnlySync(3),
        events=EventTape(tmp_path / "events.jsonl", 0),
        capabilities=fake_capabilities(features=features),
    )
    return engine, driver


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]


# -- 4.1 ---------------------------------------------------------------------


def test_train_step_metrics_masked_fraction_optional():
    assert TrainStepMetrics(grad_norm=1.0).masked_fraction is None
    assert TrainStepMetrics(grad_norm=1.0, masked_fraction=0.25).masked_fraction == 0.25


def test_adapter_reads_masked_fraction_when_present():
    assert masked_fraction([SimpleNamespace(metrics={"train/masked_fraction": 0.5})]) == 0.5
    assert masked_fraction([{"masked_fraction": 1.0}]) == 1.0
    assert masked_fraction([SimpleNamespace(outcome="normal")]) is None
    assert masked_fraction([]) is None
    assert masked_fraction([{"masked_fraction": math.nan}]) is None


def test_miles_trainer_step_metrics_carry_masked_fraction():
    from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup

    group = MilesTrainerGroup.__new__(MilesTrainerGroup)
    group.last_grad_norm = 1.0
    group.last_applied_lrs = None
    group.last_masked_fraction = None
    assert group.step_metrics().masked_fraction is None
    group.last_masked_fraction = 0.75
    assert group.step_metrics().masked_fraction == 0.75


# -- 4.2 ---------------------------------------------------------------------


def test_default_grpo_zero_grad_still_fails_even_if_fully_masked(tmp_path):
    _, driver = _run(tmp_path, AlgorithmSpec(), zero_grad_rounds={1},
                     masked_fraction_rounds={1: 1.0})
    with pytest.raises(StrictRlInvariantError) as info:
        driver.run()
    assert info.value.metric == "zero_grad_norm_with_nonzero_advantages"


def test_declared_full_mask_round_is_not_a_failure(tmp_path, masking_mechanism):
    _, driver = _run(tmp_path, MASKING, features={"eps_clip", "test_full_mask"},
                     zero_grad_rounds={1}, masked_fraction_rounds={1: 1.0})
    final = driver.run()
    assert final.policy_version == 3
    masked = [e for e in _events(tmp_path) if e["event"] == "rl_zero_gradient_masked"]
    assert masked == [dict(masked[0], rollout_id=1, masked_fraction=1.0)]


def test_masking_mechanism_without_full_mask_still_fails(tmp_path, masking_mechanism):
    for fraction in ({1: 0.9}, {}):
        _, driver = _run(tmp_path, MASKING, features={"eps_clip", "test_full_mask"},
                         zero_grad_rounds={1}, masked_fraction_rounds=fraction)
        with pytest.raises(StrictRlInvariantError, match="grad_norm 0"):
            driver.run()


def test_nonfinite_grad_norm_fails_for_any_algorithm(tmp_path, masking_mechanism):
    for spec, features in ((AlgorithmSpec(), frozenset()),
                           (MASKING, {"eps_clip", "test_full_mask"})):
        engine, driver = _run(tmp_path, spec, features=features)
        engine.trainer.step_metrics = lambda: TrainStepMetrics(
            grad_norm=math.nan, masked_fraction=1.0, applied_lrs=(1e-5,)
        )
        with pytest.raises(StrictRlInvariantError) as info:
            driver.run()
        assert info.value.metric == "nonfinite_grad_norm"
