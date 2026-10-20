"""S25 T-G1C: the token-diversity test reward and the G1 driver's wall-clock limits (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from yeto.rl import token_diversity_reward as tdr
from yeto.rl.engine import engine_sampler_entry as entry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "probes" / "s25_g1"))
from run_limits import Deadline, wait_until  # noqa: E402


def test_token_diversity_reward_scores_token_ids_not_words():
    import asyncio
    from types import SimpleNamespace

    one_word = SimpleNamespace(tokens=[1, 2] + [7] * 8, response_length=8)  # "PrototypePrototype..."
    varied = SimpleNamespace(tokens=[1, 2, 3, 4, 3, 5], response_length=4)
    assert asyncio.run(tdr.score(None, one_word)) == pytest.approx(1 / 8)
    assert asyncio.run(tdr.score(None, varied)) == pytest.approx(3 / 4)
    assert tdr.diversity([]) == 0.0
    with pytest.raises(ValueError):
        tdr.response_ids(SimpleNamespace(tokens=[1], response_length=3))


def test_miles_reward_wrapper_passes_token_ids():
    from yeto.rl.engine.engine_sampler import PromptItem
    from yeto.rl.engine.sampler_engine import Completion

    reward = entry.miles_reward(tdr.score)
    done = Completion(text="x", token_ids=(5, 5, 6, 7), logprobs=(0.0,) * 4, finish_reason="length",
                      prompt_token_ids=(1, 2))
    assert reward(PromptItem("p", "q"), done) == pytest.approx(3 / 4)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def test_sampler_wait_after_the_trainers_has_a_limit():
    clock = _Clock()
    assert wait_until(lambda: False, 300, poll_s=15, clock=clock, sleep=clock.sleep) is False
    assert 300 <= clock.now < 315
    clock.now = 0.0
    assert wait_until(lambda: clock.now >= 60, 300, poll_s=15, clock=clock, sleep=clock.sleep) is True


def test_run_deadline_ends_a_wait_early():
    clock = _Clock()
    deadline = Deadline(100, clock=clock)
    assert not deadline.passed()
    assert wait_until(lambda: False, 1000, poll_s=10, clock=clock, sleep=clock.sleep,
                      deadline=deadline) is False
    assert deadline.passed() and clock.now == 100
    with pytest.raises(ValueError):
        Deadline(0)
