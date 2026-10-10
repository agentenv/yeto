"""Declarations about what a reward function writes into ``sample.metadata``.

``loss.positive_lm_source='success'`` (VAPO positive-example LM loss) needs the
reward function to write the boolean ``sample.metadata['success']``. The launch
check ``positive_lm_success`` (yeto.rl.algos.critic) asks this module.

Kept apart from the reward modules on purpose: their source hashes are part of
the decoupling golden samples (tests/decoupling_golden.py).
"""

from __future__ import annotations

import importlib

SUCCESS_DECLARATION_ATTR = "yeto_writes_success"

# In-package Miles reward entry points that call yeto.rl.math_reward.set_success.
SUCCESS_WRITING_REWARDS = frozenset({
    "yeto.rl.gsm8k_reward:score",
    "yeto.rl.math_reward:reward_func",
})


def writes_success(fn):
    """Declare that reward function ``fn`` writes ``sample.metadata['success']``."""

    setattr(fn, SUCCESS_DECLARATION_ATTR, True)
    return fn


def reward_declares_success(reward_function: str) -> bool | None:
    """True: declared; False: importable but undeclared; None: cannot import here."""

    ref = str(reward_function)
    if ref in SUCCESS_WRITING_REWARDS:
        return True
    module, _, name = ref.partition(":")
    try:
        fn = getattr(importlib.import_module(module), name)
    except Exception:  # noqa: BLE001 - any import failure means "cannot confirm"
        return None
    return bool(getattr(fn, SUCCESS_DECLARATION_ATTR, False))
