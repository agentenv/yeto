"""VAPO declaration (change ``rl-algo-critic-family`` task 7.2, design D7).

VAPO (arXiv 2504.05118v3) = PPO with a critic plus, from its sec. 5.1 list:
value pretraining (50 steps; run as the warm-up stage W, design D5),
decoupled GAE (value target lambda 1.0), length-adaptive policy lambda
``1 - 1/(alpha*l)`` with alpha 0.05, clip-higher (eps_low 0.2, eps_high 0.28),
token-level policy loss, positive-example LM loss (weight 0.1) and group
sampling (16 samples per prompt; a run-level setting, not part of the spec).

Translation: PPO (shared actor/critic, ``critic_argv``) + the fork's
``--gae-variant decoupled`` / ``--gae-lambd-mode length_adaptive`` +
``--positive-example-lm-loss-coef`` / ``--positive-example-source success``
(fork 70e3d7761: positives from the reward function's explicit boolean success
field; NLL normalized by the global positive-token count, eq. 9). Values not in
the paper are not invented: every configuration that differs from the paper or
that the paper does not give is listed in :data:`DEVIATIONS` (design D7). Not a registration module (no fields/flags here);
execution is not declared available before GPU G1 (task 7.3).
"""

from __future__ import annotations

from typing import Any

from yeto.rl.engine.algorithm import AlgorithmSpec, load_extensions

# (spec path, value, paper source) -- arXiv 2504.05118v3.
PAPER_PARAMETERS: tuple[tuple[str, Any, str], ...] = (
    ("advantage.gamma", 1.0, "sec. 5.1 basic PPO: gamma 1.0"),
    ("advantage.gae_variant", "decoupled", "sec. 4.1; sec. 5.1 item 2"),
    ("advantage.critic_lambd", 1.0, "sec. 4.1; sec. 5.1 item 2: value returns with lambda=1.0"),
    ("advantage.lambd_mode", "length_adaptive", "sec. 4.2 eq. 5; sec. 5.1 item 3"),
    ("advantage.alpha", 0.05, "sec. 5.1 item 3: alpha = 0.05"),
    ("loss.eps_clip", 0.2, "sec. 4.3 eq. 8; sec. 5.1 item 4: eps_low 0.2"),
    ("loss.eps_clip_high", 0.28, "sec. 5.1 item 4: eps_high 0.28"),
    ("loss.aggregation", "token", "sec. 4.2 eq. 7; sec. 5.1 item 5"),
    ("loss.positive_lm_coef", 0.1, "sec. 4.3 eq. 9-10; sec. 5.1 item 6: weight 0.1"),
    ("critic.critic_lr", 2e-6, "sec. 5.1 basic PPO: critic lr 2e-6"),
    ("critic.warmup_steps", 50, "sec. 4.1; sec. 5.1 item 1: value warmup 50 steps"),
)

# Run-level settings from the paper (not part of the algorithm hash).
PAPER_RUN_SETTINGS: dict[str, Any] = {
    "actor_lr": 1e-6,  # sec. 5.1 basic PPO
    "lr_schedule": "warmup-constant",  # sec. 5.1 (warmup length not given)
    "n_samples_per_prompt": 16,  # sec. 5.1 item 7 (group sampling)
    "prompts_per_rollout": 512,  # sec. 5.1 item 7
    "mini_batch_size": 512,  # sec. 5.1 item 7 (unit not stated)
    "eval": "AIME24 avg@32, top_p 0.7, temperature 1.0",  # sec. 5.1
}

# Every configuration that differs from the paper or that the paper does not
# give (user decision E, 2026-10-07; mirrored in design D7):
# (item, yeto value, source of the value, relation to the paper).
DEVIATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("critic.init", "copy_actor_backbone + 50-step warm-up (stage W) with value-quality "
     "metrics (critic_warmup.value_quality)", "design D5; user decision B",
     "differs: paper initializes the value model from a reward model; no RM in this stack"),
    ("loss.positive_lm_source", "success (explicit boolean sample.metadata success/is_correct "
     "from the reward function; missing -> error)", "user decision C",
     "paper: 'correct answers' without a mechanism; reward>threshold only if the spec "
     "declares the reward binary success (positive_lm_source='reward')"),
    ("positive LM normalization", "sum over positive tokens / global positive-token count of "
     "the optimizer step (all micro-batches and DP ranks)", "paper eq. 9; user decision D",
     "matches eq. 9; the batch it is counted over = one Miles optimizer step"),
    ("critic.value_clip", "0.2", "Miles default (engineering baseline)", "not given in paper"),
    ("kl", "none (kl_coef 0)", "Miles shared PPO requires kl_coef 0 (engineering baseline)",
     "not given in paper for VAPO"),
    ("advantage.lambd", "1.0 (unused under length_adaptive)", "spec default fill",
     "not applicable in paper"),
    ("advantage.alpha", "0.05", "paper sec. 5.1 (user decision A: kept)",
     "same as paper; differs from the fork default 1.5 (SAO / CompactionRL)"),
    ("lr warmup steps", "run-level, not fixed", "-", "paper: 'warmup-constant', length not given"),
    ("mini-batch 512 unit", "run-level", "-", "paper does not say prompts or samples"),
    ("updates per batch", "1 actor / 1 critic (Miles default)", "Miles default",
     "not given in paper"),
    ("max response length", "run-level", "-", "not given in paper"),
)
# Back-compat summary view of DEVIATIONS.
NOT_IN_PAPER: dict[str, str] = {item: f"{value} ({relation})"
                                for item, value, _src, relation in DEVIATIONS}


def vapo_spec(**overrides: Any) -> AlgorithmSpec:
    """The VAPO spec with the paper values; ``overrides`` replace group dicts' keys."""

    load_extensions()  # gamma / critic_lambd / positive_lm_* are extension fields
    groups: dict[str, dict[str, Any]] = {
        "advantage": {"estimator": "ppo"},
        "execution": {"needs_critic": True},
        "loss": {"positive_lm_source": "success"},
        "critic": {},
    }
    for path, value, _ in PAPER_PARAMETERS:
        group, name = path.split(".")
        groups[group][name] = value
    for group, values in overrides.items():
        groups.setdefault(group, {}).update(values)
    return AlgorithmSpec(**groups)
