"""rl-algo-critic-family 5.1/5.2: critic warm-up stage W (CPU)."""

from __future__ import annotations

import dataclasses

import pytest

from yeto.rl import critic_warmup as cw
from yeto.rl.engine.algorithm import AlgorithmSpec, AlgorithmSpecError
from yeto.rl.engine.miles_adapter import config as mc
from yeto.rl.engine.run_config import CriticRunConfig

from test_rl_miles_adapter_config import make_config

WARM = AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True},
                     critic={"warmup_steps": 50, "critic_lr": 3e-6})


def _main_argv(product="/w", digest="a" * 64):
    cfg = make_config(with_eval=True)
    cfg = dataclasses.replace(cfg, algorithm=dataclasses.replace(
        cfg.algorithm, advantage_estimator="ppo",
        critic=CriticRunConfig(critic_load=product, init_sha256=digest)))
    return list(mc.translate_run_config(cfg, WARM).argv)


def _values(argv, flag):
    return [argv[i + 1] for i, t in enumerate(argv) if t == flag]


# -- 5.1 ------------------------------------------------------------------------------


def test_main_stage_runs_zero_critic_only_steps_and_loads_the_product():
    argv = _main_argv()
    assert _values(argv, "--num-critic-only-steps") == ["0"]
    assert _values(argv, "--critic-load") == ["/w"]
    assert cw.main_stage_critic_args(argv) == {"--num-critic-only-steps": "0",
                                              "--critic-load": "/w"}


def test_stage_w_argv_from_the_main_stage():
    main = _main_argv()
    stage = cw.warmup_stage_argv(main, WARM, actor_checkpoint="/ckpt/actor",
                                 critic_save="/cache/w")
    for flag, value in (("--num-rollout", "50"), ("--num-critic-only-steps", "50"),
                        ("--critic-load", "/ckpt/actor"), ("--critic-save", "/cache/w"),
                        ("--save-interval", "50")):
        assert _values(stage, flag) == [value], flag
    for gone in ("--rollout-sample-filter-path", "--rollout-all-samples-process-path",
                 "--buffer-filter-path", "--eval-interval", "--eval-prompt-data",
                 "--skip-eval-before-train"):
        assert gone not in stage, gone
    # model / batch / algorithm flags are the main stage's
    for flag in ("--hf-checkpoint", "--rollout-batch-size", "--advantage-estimator", "--gamma",
                 "--lambd", "--value-clip", "--critic-lr", "--lr"):
        assert _values(stage, flag) == _values(main, flag), flag


def test_stage_w_needs_a_warm_copied_critic():
    for spec in (AlgorithmSpec(), AlgorithmSpec(advantage_estimator="ppo"),
                 AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True},
                               critic={"init": "load", "load": "/c"})):
        with pytest.raises(AlgorithmSpecError, match="warmup_steps > 0"):
            cw.warmup_stage_argv([], spec, actor_checkpoint="/a", critic_save="/w")


def test_dry_run_snapshot_both_stages():
    result = cw.dry_run(["--dry-run", "--extra",
                         "--advantage-estimator ppo --num-critic-only-steps 50 --critic-lr 3e-6",
                         "--actor-checkpoint", "/ckpt/actor", "--critic-save", "/cache/w"])
    assert result["verdict"] == "accepted"
    assert result["main_argv"] == [
        "--advantage-estimator", "ppo", "--gamma", "1.0", "--lambd", "1.0", "--value-clip",
        "0.2", "--critic-lr", "3e-06", "--num-critic-only-steps", "0",
        "--critic-load", "<stage-W product>"]
    assert result["stage_w_argv"] == [
        "--advantage-estimator", "ppo", "--gamma", "1.0", "--lambd", "1.0", "--value-clip",
        "0.2", "--critic-lr", "3e-06", "--num-rollout", "50", "--num-critic-only-steps", "50",
        "--critic-load", "/ckpt/actor", "--critic-save", "/cache/w", "--save-interval", "50"]
    assert result["algorithm_spec_sha256"] == WARM.sha256()
    bad = cw.dry_run(["--dry-run", "--extra", "--advantage-estimator ppo",
                      "--actor-checkpoint", "/a", "--critic-save", "/w"])
    assert bad["verdict"] == "rejected"


# -- 5.2 ------------------------------------------------------------------------------


def _checkpoint(path, content):
    path.mkdir(parents=True, exist_ok=True)
    (path / "iter_0000001").mkdir(exist_ok=True)
    (path / "iter_0000001" / "model.bin").write_bytes(content)
    return str(path)


def test_checkpoint_hash_is_content_based(tmp_path):
    a = cw.checkpoint_sha256(_checkpoint(tmp_path / "a", b"x"))
    assert a == cw.checkpoint_sha256(_checkpoint(tmp_path / "b", b"x"))
    assert a != cw.checkpoint_sha256(_checkpoint(tmp_path / "c", b"y"))
    with pytest.raises(cw.CriticWarmupError):
        cw.checkpoint_sha256(tmp_path / "missing")


def test_two_islands_share_one_product_and_actor_is_checked(tmp_path):
    actor = _checkpoint(tmp_path / "actor", b"actor-weights")
    runs = []

    def run_stage(critic_save):
        runs.append(critic_save)
        _checkpoint(tmp_path / critic_save, b"critic-after-50-steps")

    first = cw.ensure_warmup(WARM, actor_checkpoint=actor, cache_root=tmp_path / "cache",
                             run_stage=run_stage)
    second = cw.ensure_warmup(WARM, actor_checkpoint=actor, cache_root=tmp_path / "cache",
                              run_stage=run_stage)
    assert len(runs) == 1 and first == second  # island 2 reuses by hash
    assert first.actor_sha256 == cw.checkpoint_sha256(actor)
    config = first.critic_run_config()
    assert (config.critic_load, config.init_sha256) == (first.critic_checkpoint,
                                                       first.critic_sha256)


def test_actor_changed_during_warmup_is_refused(tmp_path):
    actor = _checkpoint(tmp_path / "actor", b"actor-weights")

    def run_stage(critic_save):
        _checkpoint(tmp_path / critic_save, b"critic")
        _checkpoint(tmp_path / "actor", b"actor-trained!")  # stage W touched the actor

    with pytest.raises(cw.CriticWarmupError, match="actor checkpoint changed"):
        cw.ensure_warmup(WARM, actor_checkpoint=actor, cache_root=tmp_path / "cache",
                         run_stage=run_stage)


def test_tampered_or_foreign_product_is_refused(tmp_path):
    actor = _checkpoint(tmp_path / "actor", b"actor-weights")
    product = cw.ensure_warmup(WARM, actor_checkpoint=actor, cache_root=tmp_path / "cache",
                               run_stage=lambda d: _checkpoint(tmp_path / d, b"critic"))
    other = AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True},
                          critic={"warmup_steps": 50, "critic_lr": 1e-6})
    with pytest.raises(cw.CriticWarmupError, match="algorithm hash differs"):
        cw.load_product(product.critic_checkpoint, other, actor_sha256=product.actor_sha256)
    with pytest.raises(cw.CriticWarmupError, match="initial actor"):
        cw.load_product(product.critic_checkpoint, WARM, actor_sha256="0" * 64)
    _checkpoint(tmp_path / product.critic_checkpoint, b"critic-tampered")
    with pytest.raises(cw.CriticWarmupError, match="content differs"):
        cw.load_product(product.critic_checkpoint, WARM, actor_sha256=product.actor_sha256)


def test_main_stage_receipt_source_is_the_product_hash():
    cfg = make_config()
    cfg = dataclasses.replace(cfg, algorithm=dataclasses.replace(
        cfg.algorithm, advantage_estimator="ppo",
        critic=CriticRunConfig(critic_load="/w", init_sha256="b" * 64)))
    launch = mc.translate_run_config(cfg, WARM)
    assert launch.runtime_attrs["yeto_rl_critic_init_sha256"] == "b" * 64
    assert "yeto_rl_critic_init_sha256" not in mc.translate_run_config(
        make_config(), AlgorithmSpec()).runtime_attrs
