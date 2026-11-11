"""Miles fork pin for the critic family (rl-algo-critic-family).

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

Rebase: the seven commits
above were cherry-picked onto the image's Miles c35702e as branch
``yeto-critic-c357`` (local, not pushed); HEAD 6e7365b60. Tied-weights fix:
6574a9c82 on ``yeto-critic-c357-tied`` (critic value head replaces
``language_model.output_layer`` of multimodal bridge wrappers, so tied
Qwen3.5 critics need no ``lm_head.weight``) was the pin on ports-v1. The
losses.py / math_utils.py conflicts keep c35702e's ``--policy-loss-variant``
dispatch (cispo/sapo/gmpo) under the non-SAO branch; ``sao_dis`` with a
non-default variant is refused by the fork.

k31: the B8 batch (Miles k31/b8 6442e19e, head efc83590; squashing the fork commits above
plus cfafdaec's GAE variants / VAPO) carries the whole family in the main k31
lineage, so :data:`CRITIC_FORK_LINEAGE` is the k31 lineage from k31/b8 on
(``pins.K31_FEATURES["critic_family"]``) and no overlay patches the image any
more (the c35702e island-setup overlay and its sha256 record are gone). The CLI
names are the fork's (``--policy-objective``, ``--gae-variant``,
``--num-critic-epochs`` ...); options live in Miles' YetoAlgoConfig trait.

The spec-level ``*_not_at_pin`` rejections check the run's Miles pin
(:func:`yeto.rl.adapters.miles.pins.current_miles_commit`) against
:data:`FORK_COMMITS` (the loss_variants precedent). The mechanisms stay
undeclared (``--rl-allow-unverified-mechanism``) until a GPU run verifies them.
"""

from __future__ import annotations

from yeto.rl import CRITIC_FORK_LINEAGE

# Full SHAs of Miles commits carrying every critic-family flag above; derived
# from the k31 lineage in adapters/miles/pins.py (feature ``critic_family``).
FORK_COMMITS: frozenset[str] = frozenset(CRITIC_FORK_LINEAGE)

# The first commit carrying the family (k31/b8).
CRITIC_FORK_PIN = CRITIC_FORK_LINEAGE[0]


def fork_carries_critic_family(commit: str | None = None) -> bool:
    """Whether ``commit`` (default: the run's Miles pin) carries the critic family."""
    if commit is None:
        from yeto.rl import current_miles_commit

        commit = current_miles_commit()
    return commit in FORK_COMMITS
