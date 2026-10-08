"""Dense length/termination reward for the Flash-Next 4-layer stage-A rerun (fnrun.sh fn8r).

A test reward, not a product reward. The 4-layer debug slice of Qwen3.8-Flash-Next never
solves gsm8k: with ``gsm8k_reward:score`` every reward was 0, every GRPO group had zero
variance, so advantage, loss and grad_norm were all 0 and the LoRA B matrices never left
their zero init (evidence/fn/FN-A-RESULT.md section 4). This reward is continuous in the
response length, so samples of one prompt differ even when all of them are truncated:

    score = 0.5 * finished + 0.5 * (1 - min(len(response chars), CHAR_CAP) / CHAR_CAP)

``finished`` is 1.0 when Miles marks the sample COMPLETED (EOS before
``--rollout-max-response-len``), 0.0 when TRUNCATED/ABORTED or unknown. Range [0, 1];
shorter and properly terminated answers score higher. Launch with

    --reward-function yeto.rl.length_reward:reward_func
"""

from __future__ import annotations

# Neutral form: yeto.rl.rewards.builtin.length_reward (decoupling 3.2); this
# module keeps the Miles entry point and its signature.
from yeto.rl.rewards.builtin import LENGTH_CHAR_CAP as CHAR_CAP
from yeto.rl.rewards.builtin import length_reward, length_score as score
from yeto.rl.rewards.types import Trajectory

__all__ = ["CHAR_CAP", "score", "reward_func"]


def _status_text(sample) -> str:
    status = getattr(sample, "status", None)
    name = getattr(status, "name", None) or getattr(status, "value", None) or str(status or "")
    return str(name)


async def reward_func(args, sample, **kwargs) -> float:
    traj = Trajectory(response=sample.response or "", status=_status_text(sample))
    return length_reward(traj).value
