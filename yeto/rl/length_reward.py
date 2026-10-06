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

CHAR_CAP = 1024


def _finished(sample) -> float:
    status = getattr(sample, "status", None)
    name = getattr(status, "name", None) or getattr(status, "value", None) or str(status or "")
    return 1.0 if str(name).upper().endswith("COMPLETED") else 0.0


def score(response: str, finished: float) -> float:
    n = min(len(response or ""), CHAR_CAP)
    return 0.5 * float(finished) + 0.5 * (1.0 - n / CHAR_CAP)


async def reward_func(args, sample, **kwargs) -> float:
    return score(sample.response or "", _finished(sample))
