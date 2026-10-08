"""Synthetic two-segment rollouts for the GPU G1 of ``cross_segment_per_sample``
(change ``rl-algo-critic-family`` task 6.4; progress "S13 fork G1 结果").

TEST ONLY. ``cross_segment_per_sample`` normally needs compacted rollouts, which
only the Codex OpenEnv harness produces. Task 6.4 asks for the trainer-side
variant on *artificial* two-segment data: this reward wraps
``yeto.rl.gsm8k_reward:score`` (same reward and ``success`` field) and labels
every sample as one segment of a pretend two-segment rollout:

* even ``sample.index`` -> segment 0 (first segment): ``tokens_after =
  SYNTHETIC_TOKENS_AFTER`` (a later segment of that many optimised tokens),
* odd ``sample.index``  -> segment 1 (last segment): ``tokens_after = 0``.

The fork multiplies the local GAE advantage by ``(gamma*lambda)^tokens_after``
(eq. 14). The labels are fabricated by construction and say nothing about real
compaction; the launcher only accepts this reward with that variant (see
``yeto.launcher.compactionrl_launch_env``).
"""

from __future__ import annotations

from yeto.rl.gsm8k_reward import score as _gsm8k_score

SYNTHETIC_SEGMENTS_REWARD = "yeto.rl.synthetic_segments:score"
SYNTHETIC_TOKENS_AFTER = 64


def segment_labels(index: int) -> dict[str, int]:
    first = int(index) % 2 == 0
    return {
        "tokens_after": SYNTHETIC_TOKENS_AFTER if first else 0,
        "synthetic_segment_index": 0 if first else 1,
    }


async def score(args, sample, **kwargs):
    value = await _gsm8k_score(args, sample, **kwargs)
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata.update(segment_labels(getattr(sample, "index", 0) or 0))
    return value
