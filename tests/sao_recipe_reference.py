"""Frozen copy of the legacy SAO online recipe (reference only; not imported by yeto).

Verbatim body of ``apply_sao_online_recipe`` from agentenv/miles
``feat/sao-tbench21-e2e-validation`` @ 16a9bea409de61549e233dda8a684e8cdd1f7448,
``miles/backends/training_utils/sao.py`` (introduced in e25048ed). The
streaming entry ``yeto.rl.sao_streaming_runtime`` requires
``--sao-online-recipe`` and runs this on the Miles side.
"""

from typing import Any


def apply_sao_online_recipe(args: Any) -> None:
    domain = getattr(args, "sao_online_recipe", None)
    if domain is None:
        return
    if getattr(args, "value_pretrain_manifest", None) is not None:
        raise ValueError("--sao-online-recipe is for online training, not offline value pretraining")
    if args.n_samples_per_prompt != 1:
        raise ValueError("SAO requires exactly one rollout per prompt")

    args.advantage_estimator = "gae_adaptive"
    args.policy_objective = "sao_dis"
    args.gae_adaptive_mode = "adaptive"
    args.gae_adaptive_alpha = 1.5
    args.gae_adaptive_min_length = 1
    args.gamma = 1.0
    args.critic_lambd = 1.0
    args.num_critic_epochs = 2
    args.critic_freeze_attention = True
    args.lr = 1e-6
    args.kl_coef = 0.0
    args.kl_loss_coef = 0.0
    args.use_kl_loss = False
    args.entropy_coef = 0.0
    args.critic_lr = 5e-6
    args.critic_lr_warmup_iters = 10
    if getattr(args, "sao_compaction", False):
        if getattr(args, "group_rm", False):
            raise ValueError("--sao-compaction requires one terminal reward per trajectory")
        if getattr(args, "generate_multi_samples", False):
            raise ValueError("--sao-compaction owns segment assembly; remove --generate-multi-samples")
        args.calculate_per_token_loss = True
        args.use_dynamic_global_batch_size = True

    if domain == "coding":
        args.sao_dis_eps_low = 0.8
        args.sao_dis_eps_high = 3.0
    elif domain == "reasoning":
        args.sao_dis_eps_low = 0.3
        args.sao_dis_eps_high = 5.0
    else:
        raise ValueError(f"unknown SAO online recipe: {domain}")
