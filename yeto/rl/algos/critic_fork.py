"""Miles fork pin for the critic family (rl-algo-critic-family, S13 fork merge).

``yeto-critic-family`` on michaellchung/miles (local commit, not pushed, not in an
image) merges the critic-family fork branches onto yeto/ports 039471508:

* yeto-gae-variant ce96fc060 -- ``--gae-variant`` / ``--gae-lambd-mode`` /
  ``--gae-length-alpha`` / ``--gae-critic-lambd`` (decoupled, length_adaptive,
  cross_segment, now ``cross_segment_whole_rollout``);
* yeto-vapo cbf8c4737 -- ``--positive-example-lm-loss-coef`` /
  ``--positive-example-reward-threshold``;
* yeto-sao 6b5bd88c -- ``--policy-objective sao_dis``, ``--sao-dis-eps-*``,
  ``--value-loss-type classification`` (HL-Gauss / two-hot value bins);
* e07e51c07 -- ``--num-critic-epochs`` (alias ``--critic-updates-per-step``) and
  ``--critic-freeze-attention`` (SAO agentenv/miles 16a9bea4 semantics);
* ffe769c1e -- ``--gae-variant cross_segment_per_sample`` (CompactionRL, one
  sample per compaction segment; reads ``metadata.tokens_after`` /
  ``gae_length``).
* 70e3d7761 -- ``--positive-example-source {success,reward}`` (default success:
  boolean ``metadata.success``/``is_correct``), positive-example NLL normalized
  by the global positive-token count of the optimizer step (VAPO eq. 9); the
  legacy one-sample-per-rollout ``cross_segment`` renamed to the explicit control
  mode ``cross_segment_whole_rollout`` (ambiguous ``cross_segment`` refused).

The spec-level ``*_not_at_pin`` rejections check :data:`CRITIC_FORK_PIN`
against :data:`FORK_COMMITS` (the loss_variants precedent). This is a
declaration pin only: the ports image still runs ``yeto.rl.MILES_NEXT_COMMIT``
(c35702e), which does not carry these flags, and the mechanisms stay
undeclared (``--rl-allow-unverified-mechanism``) until GPU G1.
"""

from __future__ import annotations

# Full SHAs of fork commits carrying every critic-family flag above.
FORK_COMMITS: frozenset[str] = frozenset({
    "70e3d77618841235330fee2f4634b1ddd6b924da",  # yeto-critic-family (local)
})

CRITIC_FORK_PIN = "70e3d77618841235330fee2f4634b1ddd6b924da"


def fork_carries_critic_family(commit: str | None = None) -> bool:
    return (commit or CRITIC_FORK_PIN) in FORK_COMMITS
