"""rl-algo-critic-family 8.3: SAO recipe -> AlgorithmSpec (CPU only, no Ray)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.sao_recipe_reference import apply_sao_online_recipe
from yeto.rl import sao_streaming_runtime as legacy
from yeto.rl.algos import sao
from yeto.rl.engine.algorithm import AlgorithmSpec, AlgorithmSpecError, load_extensions
from yeto.rl.engine.miles_adapter.algo_flag_rows import sao_fork_argv
from yeto.rl.engine.miles_adapter.algorithm_flags import algorithm_argv

load_extensions()

# Settings of the legacy recipe the spec does not own (deployment knobs of the
# streaming entry / Miles-side attention freeze), listed so the comparison is
# exhaustive rather than silent.
NOT_IN_SPEC = {"lr", "critic_freeze_attention", "kl_loss_coef"}


def _legacy(domain):
    args = SimpleNamespace(sao_online_recipe=domain, n_samples_per_prompt=1)
    apply_sao_online_recipe(args)
    out = dict(vars(args))
    out.pop("sao_online_recipe"); out.pop("n_samples_per_prompt")
    return out


@pytest.mark.parametrize("domain", ["coding", "reasoning"])
def test_spec_settings_equal_legacy_recipe(domain):
    legacy_settings = _legacy(domain)
    got = sao.recipe_settings(sao.sao_algorithm_spec(domain))
    extra = {"value_loss_type", "value_num_bins"}  # recipe leaves these to the Miles CLI
    assert set(legacy_settings) - NOT_IN_SPEC == set(got) - extra
    for key in set(got) - extra:
        assert got[key] == legacy_settings[key], key
    assert got["value_loss_type"] == "classification" and got["value_num_bins"] == 51
    assert legacy_settings["lr"] == sao.SAO_ACTOR_LR


def test_spec_identity_and_critic_config():
    spec = sao.sao_algorithm_spec("coding")
    c, a = spec.critic, spec.advantage
    assert spec.execution.needs_critic and spec.advantage.estimator == "ppo"
    assert (a.gamma, a.lambd, a.lambd_mode, a.alpha, a.gae_variant) == (
        1.0, 1.0, "length_adaptive", 1.5, "decoupled")
    assert (c.value_loss, c.hl_gauss_bins, c.critic_updates_per_step, c.warmup_steps,
            c.param_mode, c.critic_lr, c.critic_lr_warmup) == ("hl_gauss", 51, 2, 0, "full", 5e-6, 10)
    assert sao.sao_algorithm_spec("coding").sha256() == spec.sha256()
    assert sao.sao_algorithm_spec("reasoning").sha256() != spec.sha256()
    again = AlgorithmSpec.from_dict(spec.to_dict())
    assert again.sha256() == spec.sha256()


def _fake_runtime(tmp_path: Path, actor_steps, critic_steps):
    actor = SimpleNamespace(learner_id="L", component=SimpleNamespace(model_revision="R"),
                            optimizer_steps_per_round=actor_steps)
    critic = SimpleNamespace(optimizer_steps_per_round=critic_steps)
    return SimpleNamespace(streams=SimpleNamespace(actor=actor, critic=critic),
                           trajectory_evidence_kind="secrlenv",
                           trajectory_evidence_dir=tmp_path / "fresh")


def _miles_args(actor_steps, epochs):
    return SimpleNamespace(
        sao_online_recipe="coding", use_critic=True, num_critic_only_steps=0, start_rollout_id=0,
        lora_rank=0, external_policy_sync_path=legacy._SYNC_FACTORY,
        rollout_function_path=legacy._ROLLOUT_FUNCTION, yeto_rl_learner_id="L",
        yeto_rl_base_model_revision="R", num_steps_per_rollout=actor_steps,
        num_critic_epochs=epochs)


@pytest.mark.parametrize("actor_steps", [1, 3])
def test_role_contract_matches_legacy_accounting(tmp_path, actor_steps):
    spec = sao.sao_algorithm_spec("reasoning")
    contract = sao.sao_role_contract(spec, actor_steps)
    steps = contract["optimizer_steps_per_round"]
    # the legacy entry's own validator accepts exactly these numbers
    legacy._validate_miles_runtime(
        _miles_args(actor_steps, spec.critic.critic_updates_per_step),
        _fake_runtime(tmp_path, steps["actor"], steps["critic"]))
    with pytest.raises(ValueError, match="optimizer accounting"):
        legacy._validate_miles_runtime(
            _miles_args(actor_steps, spec.critic.critic_updates_per_step),
            _fake_runtime(tmp_path, steps["actor"], steps["actor"]))
    from yeto.rl.local_learner import _ROLES_BY_ALGORITHM

    assert frozenset(contract["roles"]) == _ROLES_BY_ALGORITHM["sao"]
    assert contract["num_critic_only_steps"] == 0
    assert contract["separate_layouts"] and contract["syncer_per_role"]
    assert contract["lockstep_paired_fragments"]


def test_fork_argv():
    spec = sao.sao_algorithm_spec("coding")
    argv = sao_fork_argv(spec)
    assert argv == [
        "--policy-objective", "sao_dis", "--sao-dis-eps-low", "0.8", "--sao-dis-eps-high", "3.0",
        "--value-loss-type", "classification", "--value-num-bins", "51",
        "--value-target-type", "hl_gauss", "--hl-gauss-sigma-ratio", "0.75",
        "--value-reward-range", "0.0", "1.0",
        "--gae-variant", "decoupled", "--gae-lambd-mode", "length_adaptive",
        "--gae-length-alpha", "1.5", "--gae-critic-lambd", "1.0"]
    assert all(t in algorithm_argv(spec) for t in ("--policy-objective", "--gae-critic-lambd"))
    assert sao_fork_argv(AlgorithmSpec()) == []
    assert "--policy-objective" not in algorithm_argv(AlgorithmSpec(advantage_estimator="ppo"))


def test_rejections():
    spec = sao.sao_algorithm_spec("coding")
    assert spec.rejections() == []  # critic fork pin carries SAO (critic_fork.py, e07e51c07)
    stray = AlgorithmSpec(loss={"sao_dis_eps_low": 0.3})
    assert any("only apply" in p for p in stray.rejections())
    no_critic = AlgorithmSpec(loss={"policy_objective": "sao_dis", "sao_dis_eps_low": 0.3,
                                    "sao_dis_eps_high": 5.0})
    assert any("needs the critic" in p for p in no_critic.rejections())
    with pytest.raises(AlgorithmSpecError):
        AlgorithmSpec(loss={"policy_objective": "dis"})
    with pytest.raises(AlgorithmSpecError):
        sao.sao_algorithm_spec("math")


def test_default_specs_untouched():
    assert "policy_objective" not in AlgorithmSpec().canonical_json()
    assert "policy_objective" not in AlgorithmSpec(advantage_estimator="ppo").canonical_json()
