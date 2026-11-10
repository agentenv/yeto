import math

import pytest

from tests.test_rl_algorithm_gradient import _events, _run
from yeto.rl.core import StrictRlInvariantError
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.nan_skip import ENV, NanSkipConfigError, NanSkipPolicy
from yeto.rl.engine.driver import TrainStepMetrics


def _grad_norms(engine, values):
    calls = iter(values)
    engine.trainer.step_metrics = lambda: TrainStepMetrics(grad_norm=next(calls), applied_lrs=(1e-5,))


def test_unset_keeps_the_strict_refusal(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    engine, driver = _run(tmp_path, AlgorithmSpec())
    _grad_norms(engine, [1.0, math.nan, 1.0])
    with pytest.raises(StrictRlInvariantError) as info:
        driver.run()
    assert info.value.metric == "nonfinite_grad_norm"


def test_one_nan_round_is_skipped_and_counted(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "max_consecutive=1,max_fraction=0.5,min_rounds=1")
    engine, driver = _run(tmp_path, AlgorithmSpec())
    _grad_norms(engine, [1.0, math.nan, 1.0])
    driver.run()
    skips = [e for e in _events(tmp_path) if e["event"] == "rl_nan_skip"]
    assert len(skips) == 1 and skips[0]["nan_skipped"] == 1 and skips[0]["grad_norm"] == "nan"
    assert driver.nan_skip.rounds == 3 and driver.nan_skip.consecutive == 0
    trained = [e for e in _events(tmp_path) if e["event"] == "rl_round_trained"]
    assert [e["nan_skipped"] for e in trained] == [0, 1, 1]


def test_consecutive_nan_rounds_trip_the_breaker(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "max_consecutive=1,max_fraction=1,min_rounds=1")
    engine, driver = _run(tmp_path, AlgorithmSpec())
    _grad_norms(engine, [math.inf, math.nan, 1.0])
    with pytest.raises(StrictRlInvariantError) as info:
        driver.run()
    assert info.value.metric == "nan_skip_breaker"


def test_fraction_breaker_waits_for_min_rounds():
    policy = NanSkipPolicy(max_consecutive=5, max_fraction=0.2, min_rounds=4)
    assert policy.skipped_round() is None  # 1/1 but below min_rounds
    policy.finite_round(); policy.finite_round()
    assert policy.skipped_round() is not None  # 2/4 > 20%


@pytest.mark.parametrize("raw", ["max_consecutive", "bogus=1", "max_fraction=0", "max_consecutive=0"])
def test_bad_config_refused(raw):
    with pytest.raises(NanSkipConfigError):
        NanSkipPolicy.from_env({ENV: raw})


def test_on_means_defaults():
    assert NanSkipPolicy.from_env({ENV: "on"}) == NanSkipPolicy()
    assert NanSkipPolicy.from_env({}) is None


def test_kimi_k31_extra_argv_takes_the_megatron_nan_check_off():
    from yeto.rl.adapters.miles.config import train_extra_argv

    assert train_extra_argv("kimi_k31", "--no-check-for-nan-in-loss-and-grad") == (
        "--no-check-for-nan-in-loss-and-grad",)
