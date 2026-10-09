"""rl-algo-critic-family 9.3: CompactionRL declaration and translation (CPU)."""

from __future__ import annotations

import dataclasses

from yeto.rl import critic_warmup as cw
from yeto.rl.algos import critic
from yeto.rl.algos.compactionrl import DESIGN_PARAMETERS, PAPER_PARAMETERS, compactionrl_spec
from yeto.rl.algos.vapo import vapo_spec
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.adapters.miles import algorithm_flags as af
from yeto.rl.adapters.miles import config as mc
from yeto.rl.engine.run_config import CriticRunConfig, LrSchedule

from test_rl_miles_adapter_config import make_config

CRL_ARGV = [
    "--calculate-per-token-loss",
    "--gamma", "1.0", "--lambd", "1.0", "--value-clip", "0.2", "--critic-lr", "3e-06",
    "--critic-updates-per-step", "2", "--num-critic-only-steps", "0",
    "--gae-variant", "cross_segment_per_sample",
    "--gae-lambd-mode", "length_adaptive", "--gae-length-alpha", "1.5",
]
# 506b4ba4...c932 while the spec value was the ambiguous 'cross_segment'.
CRL_SHA = "16fb68d50b365b0bf182ee1989b650da6601efe38ac0ab5c3ec7ba4526bf3c4f"
CRL_EXTRA = (
    "--advantage-estimator ppo --gae-variant cross_segment_per_sample --gae-lambd-mode length_adaptive "
    "--gae-length-alpha 1.5 --calculate-per-token-loss --critic-lr 3e-6 "
    "--critic-updates-per-step 2 --num-critic-only-steps 50"
)
UNDECLARED = ("advantage_estimators:ppo", "execution:critic", "features:gae_cross_segment",
              "features:gae_length_adaptive", "features:critic_multi_update")


def test_spec_carries_the_values():
    spec = compactionrl_spec()
    assert spec.rejections() == []
    for path, value, _ in PAPER_PARAMETERS + DESIGN_PARAMETERS:
        assert spec.get_path(path) == value, path
    assert spec.kl.placement == "none" and spec.kl_coef is None  # kl = 0
    assert spec.critic.init == "copy_actor_backbone"
    assert spec.sha256() == CRL_SHA


def test_translation_snapshot():
    assert af.algorithm_argv(compactionrl_spec()) == CRL_ARGV
    assert "--kl-coef" not in CRL_ARGV and "--use-kl-loss" not in CRL_ARGV


def test_dry_run_snapshot_and_absorption_equal_the_declaration():
    refused = af.dry_run(["--dry-run", "--extra", CRL_EXTRA])
    assert refused["verdict"] == "rejected"
    for name in ("gae_cross_segment", "critic_multi_update"):
        assert name in refused["error"]  # not declared before GPU G1 (task 9.4)
    flags = [x for name in UNDECLARED for x in ("--rl-allow-unverified-mechanism", name)]
    result = af.dry_run(["--dry-run", "--extra", CRL_EXTRA, *flags])
    assert result["verdict"] == "accepted"
    assert result["miles_argv"] == ["--advantage-estimator", "ppo", *CRL_ARGV]
    assert result["algorithm_spec_sha256"] == CRL_SHA
    assert result["remaining_extra_argv"] == []


def test_stage_w_has_50_steps_and_the_same_critic_schedule():
    spec = compactionrl_spec()
    cfg = make_config()
    cfg = dataclasses.replace(cfg, algorithm=dataclasses.replace(
        cfg.algorithm, advantage_estimator="ppo",
        # critic_updates_per_step=2 needs a non-decaying schedule (critic_lr_horizon)
        lr_schedule=LrSchedule("constant", 3),
        critic=CriticRunConfig(critic_load="/w", init_sha256="a" * 64)))
    main = list(mc.translate_run_config(cfg, spec).argv)
    stage = cw.warmup_stage_argv(main, spec, actor_checkpoint="/ckpt/actor", critic_save="/c")
    values = lambda argv, flag: [argv[i + 1] for i, t in enumerate(argv) if t == flag]
    assert values(stage, "--num-critic-only-steps") == ["50"]
    for flag in ("--gae-variant", "--gae-lambd-mode", "--gae-length-alpha",
                 "--critic-updates-per-step", "--critic-lr"):
        assert values(stage, flag) == values(main, flag) != [], flag


def test_hash_distinguishes_the_components():
    base = compactionrl_spec().sha256()
    for change in (dict(advantage={"gae_variant": "vanilla"}), dict(advantage={"alpha": 1.0}),
                   dict(critic={"critic_updates_per_step": 1}), dict(critic={"warmup_steps": 0}),
                   dict(loss={"aggregation": "default"})):
        assert compactionrl_spec(**change).sha256() != base, change


def test_ablation_arm_without_cross_segment_drops_only_that_flag():
    argv = af.algorithm_argv(compactionrl_spec(advantage={"gae_variant": "vanilla"}))
    assert argv == [t for t in CRL_ARGV if t not in ("--gae-variant", "cross_segment_per_sample")]


def test_single_update_and_plain_ppo_argv_unchanged():
    ppo = AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True})
    assert af.algorithm_argv(ppo) == [
        "--gamma", "1.0", "--lambd", "1.0", "--value-clip", "0.2", "--num-critic-only-steps", "0"]
    assert "--critic-updates-per-step" not in af.algorithm_argv(vapo_spec())
    assert "--critic-updates-per-step" in critic.FORK_FLAGS


def test_ambiguous_cross_segment_is_refused():
    flags = [x for name in UNDECLARED for x in ("--rl-allow-unverified-mechanism", name)]
    legacy = CRL_EXTRA.replace("cross_segment_per_sample", "cross_segment")
    result = af.dry_run(["--dry-run", "--extra", legacy, *flags])
    assert result["verdict"] == "rejected"
    assert "cross_segment_per_sample" in result["error"]
    assert "cross_segment_whole_rollout" in result["error"]
    import pytest
    from yeto.rl.engine.algorithm import AlgorithmSpecError

    with pytest.raises(AlgorithmSpecError, match="ambiguous"):
        compactionrl_spec(advantage={"gae_variant": "cross_segment"})


def test_whole_rollout_control_arm_is_explicit():
    from yeto.rl.algos.compactionrl import compactionrl_whole_rollout_control_spec

    spec = compactionrl_whole_rollout_control_spec()
    assert spec.rejections() == []
    assert spec.sha256() != compactionrl_spec().sha256()
    argv = af.algorithm_argv(spec)
    assert argv == [("cross_segment_whole_rollout" if t == "cross_segment_per_sample" else t)
                    for t in CRL_ARGV]
    extra = CRL_EXTRA.replace("cross_segment_per_sample", "cross_segment_whole_rollout")
    flags = [x for name in UNDECLARED if name != "features:gae_cross_segment"
             for x in ("--rl-allow-unverified-mechanism", name)]
    refused = af.dry_run(["--dry-run", "--extra", extra, *flags])
    assert refused["verdict"] == "rejected" and "gae_cross_segment_whole_rollout" in refused["error"]
    flags += ["--rl-allow-unverified-mechanism", "features:gae_cross_segment_whole_rollout"]
    result = af.dry_run(["--dry-run", "--extra", extra, *flags])
    assert result["verdict"] == "accepted"
    assert result["algorithm_spec_sha256"] == spec.sha256()


def test_linear_schedule_shorter_than_critic_steps_is_refused():
    """S14 SAO G1: --lr-decay-iters = actor steps is shared with a critic that
    takes 2 updates per step, so the critic LR hit 0 halfway (critic_lr_horizon)."""
    from yeto.rl.algos import sao
    from yeto.rl.engine.algorithm import launch_problems

    values = {"num_rollout": 12, "rollout_batch_size": 4, "n_samples_per_prompt": 8,
              "global_batch_size": 32, "lr_decay_iters": 12, "lr_decay_style": "linear"}
    for spec in (sao.sao_algorithm_spec("coding"), compactionrl_spec()):
        problems = launch_problems(spec, values)
        assert any("critic_lr_horizon" in p and "24 optimizer steps" in p for p in problems), problems
        assert not any("critic_lr_horizon" in p
                       for p in launch_problems(spec, {**values, "lr_decay_style": "constant"}))
        assert not any("critic_lr_horizon" in p
                       for p in launch_problems(spec, {**values, "lr_decay_iters": 24}))
    one = AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True})
    assert not any("critic_lr_horizon" in p for p in launch_problems(one, values))
