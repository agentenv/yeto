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
* 5182e37f0 -- SAO: with ``sao_dis`` the critic re-runs a forward after its
  update so the shipped values are post-update (agentenv/miles 16a9bea4).

S13 rebase (progress "S13 fork rebase c35702e + overlay"): the seven commits
above were cherry-picked onto the image's Miles c35702e as branch
``yeto-critic-c357`` (local, not pushed); HEAD 6e7365b60. S13 tied fix:
6574a9c82 on ``yeto-critic-c357-tied`` (critic value head replaces
``language_model.output_layer`` of multimodal bridge wrappers, so tied
Qwen3.5 critics need no ``lm_head.weight``) is the pin. The
losses.py / math_utils.py conflicts keep c35702e's ``--policy-loss-variant``
dispatch (cispo/sapo/gmpo) under the non-SAO branch; ``sao_dis`` with a
non-default variant is refused by the fork.

The spec-level ``*_not_at_pin`` rejections check :data:`CRITIC_FORK_PIN`
against :data:`FORK_COMMITS` (the loss_variants precedent). The ports image
still ships ``yeto.rl.MILES_NEXT_COMMIT`` (c35702e); a run gets the pinned code
only through the island-setup overlay (:mod:`yeto.rl.adapters.miles.overlay`, patch
c35702e..pin, sha256-checked, recorded as "image c35702e + overlay <sha256>").
The mechanisms stay undeclared (``--rl-allow-unverified-mechanism``) until GPU G1.
"""

from __future__ import annotations

# Full SHAs of fork commits carrying every critic-family flag above.
FORK_COMMITS: frozenset[str] = frozenset({
    "6e7365b602ecf5d83d16169e87f4d78f4daf5f86",  # yeto-critic-c357 (local, on c35702e)
    "6574a9c82edeed17f4b00bc64db6c7cd6b45cb39",  # yeto-critic-c357-tied: + critic value head on language_model (Qwen3.5 tied)
})

CRITIC_FORK_PIN = "6574a9c82edeed17f4b00bc64db6c7cd6b45cb39"


def fork_carries_critic_family(commit: str | None = None) -> bool:
    return (commit or CRITIC_FORK_PIN) in FORK_COMMITS
