"""A Miles trainer over a sample buffer may run a larger policy-age limit than the
in-island carry-over (stage 3, at most 4): its buffer reader drops whole groups."""
import pytest

from yeto.rl.adapters.miles.policy_age import SUPPORT
from yeto.rl.engine.policy_age import PolicyAgeError


def test_a_buffer_trainer_takes_a_larger_limit_than_carry_over():
    SUPPORT.check(4)
    with pytest.raises(PolicyAgeError, match="at most 4"):
        SUPPORT.check(12)
    SUPPORT.check(12, buffer_trainer=True)
    with pytest.raises(PolicyAgeError, match="sample buffer runs a policy-age limit of at most 16"):
        SUPPORT.check(17, buffer_trainer=True)
