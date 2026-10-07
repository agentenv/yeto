"""VAPO declaration (change ``rl-algo-critic-family`` task 7.2, design D7).

VAPO (arXiv 2504.05118v3) = PPO with a critic plus, from its sec. 5.1 list:
value pretraining (50 steps; run as the warm-up stage W, design D5),
decoupled GAE (value target lambda 1.0), length-adaptive policy lambda
``1 - 1/(alpha*l)`` with alpha 0.05, clip-higher (eps_low 0.2, eps_high 0.28),
token-level policy loss, positive-example LM loss (weight 0.1) and group
sampling (16 samples per prompt; a run-level setting, not part of the spec).

Translation: PPO (shared actor/critic, ``critic_argv``) + the fork's
``--gae-variant decoupled`` / ``--gae-lambd-mode length_adaptive`` +
``--positive-example-lm-loss-coef``. Values not in the paper are not invented:
see :data:`NOT_IN_PAPER`. Not a registration module (no fields/flags here);
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

# Yeto choices the paper does not fix (need user confirmation).
NOT_IN_PAPER: dict[str, str] = {
    "loss.positive_lm_reward_threshold": "paper: 'correct answers'; yeto uses reward > 0.0",
    "critic.init": "paper initializes the value model from a reward model; yeto copies the "
                   "actor backbone (design D5) -- no reward model in this stack",
    "critic.value_clip": "not given; Miles default 0.2",
    "advantage.lambd": "unused under length_adaptive; filled with 1.0",
    "kl": "not given for VAPO; Miles shared PPO requires kl_coef 0",
}


def vapo_spec(**overrides: Any) -> AlgorithmSpec:
    """The VAPO spec with the paper values; ``overrides`` replace group dicts' keys."""

    load_extensions()  # gamma / critic_lambd / positive_lm_* are extension fields
    groups: dict[str, dict[str, Any]] = {
        "advantage": {"estimator": "ppo"},
        "execution": {"needs_critic": True},
        "loss": {"positive_lm_reward_threshold": 0.0},
        "critic": {},
    }
    for path, value, _ in PAPER_PARAMETERS:
        group, name = path.split(".")
        groups[group][name] = value
    for group, values in overrides.items():
        groups.setdefault(group, {}).update(values)
    return AlgorithmSpec(**groups)
