"""Fixed case list for 2.6 attempt 3 (committed before the run)."""
from yeto.rl.engine.algorithm import (
    BOUNDED_NONZERO_STD_FILTER, STOCK_NONZERO_STD_FILTER, AdvantageSpec, AlgorithmSpec,
    CorrectionSpec, KlSpec, LossSpec, PluginRef, SamplingSpec,
)

REF = PluginRef.from_path("yeto.rl.engine.algorithm.plugin_source_sha256")
ICEPOP = PluginRef("miles.backends.training_utils.loss_hub.corrections.icepop_function", "0" * 64)


def cases():
    """[(name, AlgorithmSpec, {namespace attr: expected})]."""

    c = [("default_grpo", AlgorithmSpec(), {"advantage_estimator": "grpo", "kl_coef": 0.0}),
         ("v1_kl0_bounded", AlgorithmSpec(kl_coef=0.0, dynamic_sampling_filter=BOUNDED_NONZERO_STD_FILTER,
                                          dynamic_sampling_max_replacements=2),
          {"kl_coef": 0.0, "dynamic_sampling_filter_path": BOUNDED_NONZERO_STD_FILTER,
           "yeto_rl_dynamic_sampling_max_replacements": 2})]
    one = [
        ("eps_clip", dict(loss=LossSpec(eps_clip=0.25)), dict(eps_clip=0.25)),
        ("clip_higher", dict(loss=LossSpec(eps_clip_high=0.28)), dict(eps_clip_high=0.28)),
        ("dual_clip", dict(loss=LossSpec(eps_clip_c=3)), dict(eps_clip_c=3.0)),
        ("token_agg", dict(loss=LossSpec(aggregation="token")), dict(calculate_per_token_loss=True)),
        ("constant_agg", dict(loss=LossSpec(aggregation="constant", reducer=REF)),
         dict(custom_pg_loss_reducer_function_path=REF.path)),
        ("custom_loss", dict(loss=LossSpec(variant="custom_loss", custom_loss=REF)),
         dict(loss_type="custom_loss", custom_loss_function_path=REF.path)),
        ("no_std", dict(advantage=AdvantageSpec(std_normalization=False)), dict(grpo_std_normalization=False)),
        ("no_reward_norm", dict(advantage=AdvantageSpec(rewards_normalization=False)),
         dict(rewards_normalization=False)),
        ("whiten", dict(advantage=AdvantageSpec(whiten=True)), dict(normalize_advantages=True)),
        ("reward_postprocess", dict(advantage=AdvantageSpec(reward_postprocess=REF)),
         dict(custom_reward_post_process_path=REF.path)),
        ("gspo", dict(advantage=AdvantageSpec(estimator="gspo"), loss=LossSpec(eps_clip=3e-4, eps_clip_high=4e-4)),
         dict(advantage_estimator="gspo", eps_clip=3e-4, eps_clip_high=4e-4)),
        ("rpp_reward_kl", dict(advantage=AdvantageSpec(estimator="reinforce_plus_plus", whiten=True),
                               kl=KlSpec(placement="reward", coef=0.05)),
         dict(advantage_estimator="reinforce_plus_plus", kl_coef=0.05, normalize_advantages=True)),
        ("rpp_baseline", dict(advantage=AdvantageSpec(estimator="reinforce_plus_plus_baseline", whiten=True)),
         dict(advantage_estimator="reinforce_plus_plus_baseline", normalize_advantages=True)),
    ]
    for est in ("k1", "k2", "k3", "low_var_kl"):
        one.append((f"kl_loss_{est}", dict(kl=KlSpec(placement="loss", coef=0.01, estimator=est)),
                    dict(use_kl_loss=True, kl_loss_coef=0.01, kl_loss_type=est, kl_coef=0.0)))
    one += [
        ("kl_unbiased", dict(kl=KlSpec(placement="loss", coef=0.01, estimator="k3", unbiased=True)),
         dict(use_unbiased_kl=True)),
        ("entropy", dict(entropy_coef=0.001), dict(entropy_coef=0.001)),
        ("tis", dict(correction=CorrectionSpec(method="tis", tis_clip=2, tis_clip_low=0.5)),
         dict(use_tis=True, tis_clip=2.0, tis_clip_low=0.5)),
        ("custom_tis", dict(correction=CorrectionSpec(method="custom", function=ICEPOP, tis_clip=5,
                                                      tis_clip_low=0.5, mismatch_metrics=True)),
         dict(use_tis=True, custom_tis_function_path=ICEPOP.path, get_mismatch_metrics=True)),
        ("rollout_logprobs", dict(correction=CorrectionSpec(use_rollout_logprobs=True)),
         dict(use_rollout_logprobs=True)),
        ("opsm", dict(correction=CorrectionSpec(method="opsm", opsm_delta=1e-4)), dict(use_opsm=True, opsm_delta=1e-4)),
        ("stock_filter_oversample", dict(sampling=SamplingSpec(filter=STOCK_NONZERO_STD_FILTER,
                                                               over_sampling_batch_size=4)),
         dict(dynamic_sampling_filter_path=STOCK_NONZERO_STD_FILTER, over_sampling_batch_size=4)),
        ("combo_dapo", dict(loss=LossSpec(eps_clip=0.2, eps_clip_high=0.28, aggregation="token"),
                            advantage=AdvantageSpec(std_normalization=False), entropy_coef=0.001,
                            sampling=SamplingSpec(filter=BOUNDED_NONZERO_STD_FILTER, max_replacements=2)),
         dict(eps_clip=0.2, eps_clip_high=0.28, calculate_per_token_loss=True,
              grpo_std_normalization=False, entropy_coef=0.001)),
        ("combo_kl_tis_dual", dict(kl=KlSpec(placement="loss", coef=0.02, estimator="k2"),
                                   correction=CorrectionSpec(method="tis", tis_clip=3, tis_clip_low=0.1),
                                   loss=LossSpec(eps_clip_c=2.5)),
         dict(use_kl_loss=True, kl_loss_type="k2", use_tis=True, tis_clip=3.0, eps_clip_c=2.5)),
    ]
    c += [(n, AlgorithmSpec(**ch), exp) for n, ch, exp in one]
    return c
