"""Rule-based math-answer reward for Miles RL (MATH-500 style rows).

A smoke-test environment, not a product reward: it needs no external
service, so it is the quickest way to check that an RL island trains at
all. Launch with

    --data HuggingFaceH4/MATH-500 --rl-prompt-column problem --rl-label-column answer
    --reward-function examples/math_reward.py:reward_func

(see examples/README.md for the full line). The grader itself lives in
`yeto.rl.math_reward`, so an island that runs from the installed package
rather than the repo — a Modal island never gets `examples/` mounted —
can name it as `yeto.rl.math_reward:reward_func` instead.
"""

from __future__ import annotations

from yeto.rl.math_reward import reward_func, score

__all__ = ["reward_func", "score"]
