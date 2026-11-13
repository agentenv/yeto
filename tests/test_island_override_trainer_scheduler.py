"""A per-island rl_max_policy_age on the island the trainer scheduler drives
would make the head's backlog (global age) disagree with the trainer's reader
(overridden age): the launch refuses that combination."""

from types import SimpleNamespace

import pytest

from yeto import island_overrides as io_


def _args(**kw):
    base = dict(training_mode="rl", rl_negative_test_run=True, rl_trainer_scheduler="s1",
                rl_trainer_island=1, rl_max_policy_age=2, gpu="modal:H100:8,modal:H100:8")
    base.update(kw)
    return SimpleNamespace(**base)


def test_policy_age_override_on_scheduled_trainer_island_is_refused():
    with pytest.raises(ValueError, match="trainer scheduler"):
        io_.overrides_of(_args(rl_island_override=["1:rl_max_policy_age=5"]), 2)


def test_policy_age_override_on_a_sampler_island_still_allowed():
    out = io_.overrides_of(_args(rl_island_override=["0:rl_max_policy_age=5"]), 2)
    assert out == {0: {"rl_max_policy_age": 5}}


def test_other_keys_on_trainer_island_still_allowed():
    out = io_.overrides_of(_args(rl_island_override=["1:rl_lr_schedule=constant"]), 2)
    assert out == {1: {"rl_lr_schedule": "constant"}}


def test_policy_age_override_without_scheduler_still_allowed():
    out = io_.overrides_of(_args(rl_trainer_scheduler="off", rl_trainer_island=None,
                                 rl_island_override=["1:rl_max_policy_age=5"]), 2)
    assert out == {1: {"rl_max_policy_age": 5}}


@pytest.mark.parametrize("island", [0, 1])
def test_policy_age_override_in_a_convertible_pool_is_refused(island):
    args = _args(rl_trainer_scheduler="off", rl_trainer_island=None,
                 rl_node_class=["island0=convertible", "island1=convertible"],
                 rl_island_override=[f"{island}:rl_max_policy_age=5"])
    with pytest.raises(ValueError, match="trainer scheduler"):
        io_.overrides_of(args, 2)
