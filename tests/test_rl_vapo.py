"""rl-algo-critic-family 7.2: VAPO declaration and translation (CPU)."""

from __future__ import annotations

import dataclasses

import pytest

from yeto.rl import critic_warmup as cw
from yeto.rl.algos.vapo import PAPER_PARAMETERS, vapo_spec
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.miles_adapter import algorithm_flags as af
from yeto.rl.engine.miles_adapter import config as mc
from yeto.rl.engine.run_config import CriticRunConfig

from test_rl_miles_adapter_config import make_config

VAPO_ARGV = [
    "--eps-clip", "0.2", "--eps-clip-high", "0.28", "--calculate-per-token-loss",
    "--gamma", "1.0", "--lambd", "1.0", "--value-clip", "0.2", "--critic-lr", "2e-06",
    "--num-critic-only-steps", "0",
    "--gae-variant", "decoupled", "--gae-critic-lambd", "1.0",
    "--gae-lambd-mode", "length_adaptive", "--gae-length-alpha", "0.05",
    "--positive-example-lm-loss-coef", "0.1", "--positive-example-source", "success",
]
# 7ee1dde4...386f before user decision C (reward > 0.0 positives); now explicit success.
VAPO_SHA = "5e9b38edb3eec5e160d221c7f62275e6ebd4a9e163d743ddca94a9b406922113"
VAPO_EXTRA = (
    "--advantage-estimator ppo --gae-variant decoupled --gae-lambd-mode length_adaptive "
    "--gae-length-alpha 0.05 --eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss "
    "--positive-example-lm-loss-coef 0.1 --positive-example-source success "
    "--critic-lr 2e-6 --num-critic-only-steps 50"
)
UNDECLARED = ("advantage_estimators:ppo", "execution:critic", "features:gae_decoupled",
              "features:gae_length_adaptive", "features:positive_example_lm_loss")


def _ppo(**groups) -> AlgorithmSpec:
    advantage = {"estimator": "ppo", **groups.pop("advantage", {})}
    return AlgorithmSpec(advantage=advantage, execution={"needs_critic": True}, **groups)


def test_vapo_spec_carries_the_paper_values():
    spec = vapo_spec()
    assert spec.rejections() == []
    for path, value, _source in PAPER_PARAMETERS:
        assert spec.get_path(path) == value, path
    assert spec.loss.positive_lm_source == "success"
    assert spec.loss.positive_lm_reward_threshold is None
    assert spec.critic.init == "copy_actor_backbone"
    assert spec.sha256() == VAPO_SHA


def test_vapo_translation_snapshot():
    assert af.algorithm_argv(vapo_spec()) == VAPO_ARGV
    # the warm-up length is not a main-stage flag: it is stage W (design D5)
    assert "50" not in VAPO_ARGV


def test_dry_run_snapshot_and_absorption_equal_the_declaration():
    refused = af.dry_run(["--dry-run", "--extra", VAPO_EXTRA])
    assert refused["verdict"] == "rejected"
    for name in ("gae_decoupled", "gae_length_adaptive", "positive_example_lm_loss"):
        assert name in refused["error"]  # not declared before GPU G1 (task 7.3)
    flags = [x for name in UNDECLARED for x in ("--rl-allow-unverified-mechanism", name)]
    result = af.dry_run(["--dry-run", "--extra", VAPO_EXTRA, *flags])
    assert result["verdict"] == "accepted"
    assert result["miles_argv"] == ["--advantage-estimator", "ppo", *VAPO_ARGV]
    assert result["algorithm_spec_sha256"] == VAPO_SHA
    assert result["remaining_extra_argv"] == []


def test_stage_w_runs_the_50_step_value_pretraining_with_the_vapo_gae():
    spec = vapo_spec()
    cfg = make_config()
    cfg = dataclasses.replace(cfg, algorithm=dataclasses.replace(
        cfg.algorithm, advantage_estimator="ppo",
        critic=CriticRunConfig(critic_load="/w", init_sha256="a" * 64)))
    main = list(mc.translate_run_config(cfg, spec).argv)
    stage = cw.warmup_stage_argv(main, spec, actor_checkpoint="/ckpt/actor", critic_save="/c")
    values = lambda argv, flag: [argv[i + 1] for i, t in enumerate(argv) if t == flag]
    assert values(main, "--num-critic-only-steps") == ["0"]
    assert values(stage, "--num-critic-only-steps") == ["50"]
    for flag in ("--gae-variant", "--gae-critic-lambd", "--gae-lambd-mode", "--gae-length-alpha"):
        assert values(stage, flag) == values(main, flag) != [], flag


def test_hash_distinguishes_the_vapo_components():
    base = vapo_spec().sha256()
    for change in (dict(advantage={"alpha": 1.5}), dict(advantage={"critic_lambd": 0.95}),
                   dict(loss={"positive_lm_coef": 0.2}),
                   dict(loss={"positive_lm_source": "reward",
                              "positive_lm_reward_threshold": 0.5})):
        assert vapo_spec(**change).sha256() != base, change


def test_decoupled_fills_the_critic_lambda_explicitly():
    implicit = _ppo(advantage={"gae_variant": "decoupled"})
    explicit = _ppo(advantage={"gae_variant": "decoupled", "critic_lambd": 1.0})
    assert implicit.advantage.critic_lambd == 1.0
    assert implicit.sha256() == explicit.sha256()
    assert implicit.rejections() == []
    assert "--gae-lambd-mode" not in af.algorithm_argv(implicit)


@pytest.mark.parametrize("spec, message", [
    (lambda: _ppo(advantage={"critic_lambd": 0.9}), "critic_lambd"),
    (lambda: _ppo(critic={"value_loss": "hl_gauss"}), "hl_gauss"),
    (lambda: _ppo(loss={"positive_lm_coef": 0.1}), "positive_lm_source"),
    (lambda: _ppo(loss={"positive_lm_coef": 0.1, "positive_lm_source": "reward"}),
     "positive_lm_reward_threshold"),
    (lambda: _ppo(loss={"positive_lm_coef": 0.1, "positive_lm_source": "success",
                        "positive_lm_reward_threshold": 0.0}), "only applies"),
    (lambda: _ppo(loss={"positive_lm_reward_threshold": 0.0}), "only apply"),
    (lambda: _ppo(loss={"positive_lm_source": "success"}), "only apply"),
    (lambda: AlgorithmSpec(advantage={"gae_variant": "decoupled"}), "only apply to critic"),
])
def test_rejections(spec, message):
    assert any(message in p for p in spec().rejections())


def test_plain_ppo_and_grpo_argv_unchanged():
    assert af.algorithm_argv(_ppo()) == [
        "--gamma", "1.0", "--lambd", "1.0", "--value-clip", "0.2", "--num-critic-only-steps", "0"]
    assert af.algorithm_argv(AlgorithmSpec()) == []


def test_reward_source_is_an_explicit_binary_success_declaration():
    spec = vapo_spec(loss={"positive_lm_source": "reward", "positive_lm_reward_threshold": 0.0})
    assert spec.rejections() == []
    argv = af.algorithm_argv(spec)
    assert argv[-6:] == ["--positive-example-lm-loss-coef", "0.1", "--positive-example-source",
                         "reward", "--positive-example-reward-threshold", "0.0"]
    extra = VAPO_EXTRA.replace("--positive-example-source success",
                               "--positive-example-source reward --positive-example-reward-threshold 0")
    flags = [x for name in UNDECLARED for x in ("--rl-allow-unverified-mechanism", name)]
    result = af.dry_run(["--dry-run", "--extra", extra, *flags])
    assert result["verdict"] == "accepted"
    assert result["algorithm_spec_sha256"] == spec.sha256() != VAPO_SHA


def test_deviation_list_covers_every_non_paper_choice():
    from yeto.rl.algos.vapo import DEVIATIONS, NOT_IN_PAPER

    items = {item for item, *_ in DEVIATIONS}
    for needed in ("critic.init", "loss.positive_lm_source", "positive LM normalization",
                   "critic.value_clip", "kl", "advantage.alpha"):
        assert needed in items
    assert all(len(row) == 4 and all(row) for row in DEVIATIONS)
    assert set(NOT_IN_PAPER) == items
    assert vapo_spec().advantage.alpha == 0.05  # decision A: paper value kept
