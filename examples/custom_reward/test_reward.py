"""Test of the minimal custom reward; runs without any training framework.

    python -m pytest examples/custom_reward/test_reward.py
"""

from __future__ import annotations

import sys

from examples.custom_reward.reward import ends_with_label
from yeto.rl.rewards.types import Trajectory


def test_ends_with_label():
    assert ends_with_label(Trajectory(response="so it is 42", label="42")).value == 1.0
    miss = ends_with_label(Trajectory(response="so it is 41", label="42"))
    assert miss.value == 0.0 and miss.metadata == {"success": False, "response_chars": 11}
    assert ends_with_label(Trajectory(response="x", label=None)).value == 0.0
    assert not {m.split(".")[0] for m in sys.modules} & {"miles", "verl", "megatron"}
