# Test-only reward (M2): a sliced 5-layer MoE never solves gsm8k, so every reward is 0, the GRPO advantage is 0 and
# grad_norm is 0 -- the backward check can't be exercised. Response length varies across the n samples of a prompt,
# so the group advantage is non-zero and a real gradient flows through the cross-node EP group.
async def score(args, sample, **kwargs):
    return min(len(sample.response or ""), 2000) / 2000.0
