"""S13 fork merge: critic-family specs pass the fork pin (CPU only, no Ray)."""

from __future__ import annotations

import pytest

from yeto.rl.algos import critic, critic_fork, sao
from yeto.rl.algos.compactionrl import compactionrl_spec
from yeto.rl.algos.vapo import vapo_spec
from yeto.rl.engine.algorithm import AlgorithmSpec, load_extensions
from yeto.rl.engine.miles_adapter import algorithm_flags as af

load_extensions()

PIN = "e07e51c07f5e38a32dfb31d98eebd3d6e6daa4b8"
SAO_UNDECLARED = ("advantage_estimators:ppo", "execution:critic", "features:critic_multi_update",
                  "features:gae_decoupled", "features:gae_length_adaptive", "features:sao_dis",
                  "features:value_hl_gauss")


def _allow(names):
    return [x for name in names for x in ("--rl-allow-unverified-mechanism", name)]


def _extra(spec):
    argv = af.algorithm_argv(spec)
    if "--value-reward-range" in argv:  # translation constant, not absorbable
        i = argv.index("--value-reward-range")
        argv = argv[:i] + argv[i + 3:]
    return " ".join(["--advantage-estimator", "ppo", *argv])


def test_pin_is_the_merged_fork_commit():
    assert critic_fork.CRITIC_FORK_PIN == PIN and PIN in critic_fork.FORK_COMMITS
    assert sao.FORK_COMMITS is critic_fork.FORK_COMMITS


@pytest.mark.parametrize("make", [vapo_spec, compactionrl_spec,
                                  lambda: sao.sao_algorithm_spec("coding"),
                                  lambda: sao.sao_algorithm_spec("reasoning")])
def test_no_pin_rejection(make):
    assert make().rejections() == []


@pytest.mark.parametrize("domain", ["coding", "reasoning"])
def test_sao_dry_run_needs_unverified_flags_then_absorbs_to_the_same_hash(domain):
    spec = sao.sao_algorithm_spec(domain)
    refused = af.dry_run(["--dry-run", "--extra", _extra(spec)])
    assert refused["verdict"] == "rejected"
    for name in ("sao_dis", "value_hl_gauss", "critic_multi_update"):
        assert name in refused["error"]
    result = af.dry_run(["--dry-run", "--extra", _extra(spec), *_allow(SAO_UNDECLARED)])
    assert result["verdict"] == "accepted", result.get("error")
    assert result["algorithm_spec_sha256"] == spec.sha256()
    assert result["remaining_extra_argv"] == []
    assert result["miles_argv"] == ["--advantage-estimator", "ppo", *af.algorithm_argv(spec)]


def test_sao_rejected_when_the_pin_is_not_a_fork_commit(monkeypatch):
    monkeypatch.setattr(sao, "CRITIC_FORK_PIN", "0" * 40)
    problems = " ".join(sao.sao_algorithm_spec("coding").rejections())
    assert "sao_dis" in problems


def test_hl_gauss_without_sao_stays_refused():
    spec = AlgorithmSpec(advantage_estimator="ppo", execution={"needs_critic": True},
                         critic={"value_loss": "hl_gauss"})
    assert any("hl_gauss" in p for p in spec.rejections())


def test_num_critic_epochs_is_an_alias_of_critic_updates_per_step():
    base = AlgorithmSpec(advantage_estimator="ppo", execution={"needs_critic": True})
    a, _, _ = af.absorb_extra_argv(base, ["--num-critic-epochs", "2"])
    b, _, _ = af.absorb_extra_argv(base, ["--critic-updates-per-step", "2"])
    assert a.critic.critic_updates_per_step == b.critic.critic_updates_per_step == 2
    assert a.sha256() == b.sha256()
    argv = critic.critic_argv(a)
    assert argv[argv.index("--critic-updates-per-step") + 1] == "2"
    assert "--num-critic-epochs" not in argv  # one spelling emitted
    with pytest.raises(af.AlgorithmFlagConflict):
        af.absorb_extra_argv(base, ["--num-critic-epochs", "2", "--critic-updates-per-step", "3"])
    assert "--critic-updates-per-step" not in critic.critic_argv(base)  # default 1: not emitted


@pytest.mark.parametrize("flag, value", [("--value-target-type", "two_hot"),
                                         ("--hl-gauss-sigma-ratio", "0.5"),
                                         ("--value-loss-type", "huber")])
def test_value_flags_outside_the_spec_are_refused(flag, value):
    with pytest.raises(af.AlgorithmSpecError):
        af.absorb_extra_argv(AlgorithmSpec(advantage_estimator="ppo"), [flag, value])


def test_value_reward_range_cannot_pass_raw():
    with pytest.raises(af.UnmappedAlgorithmFlag):
        af.absorb_extra_argv(AlgorithmSpec(), ["--value-reward-range", "0", "2"])
