"""rl-algo-critic-family 2.1-2.4: critic fields, translation, pre-GPU rejections."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from yeto.rl.algos.critic import critic_run_problems
from yeto.rl.adapters.miles.algo_flag_rows import critic_argv
from yeto.rl.engine.algorithm import (
    AlgorithmSpec,
    AlgorithmSpecError,
    CriticSpec,
    launch_problems,
)
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.capabilities import CapabilityMismatch, ExecutionCapabilities
from yeto.rl.engine.driver import DriverError, EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.adapters.miles import algorithm_flags as af
from yeto.rl.adapters.miles import config as mc
from yeto.rl.engine.run_config import CriticRunConfig

from test_rl_miles_adapter_config import make_config

# Hashes computed on integ-decl de27c301 (before this change); see
# openspec/changes/rl-algo-critic-family/evidence/hash-critic-base.txt.
GRPO_DEFAULT_SHA = "27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea"
KL_LOSS_SHA = "174327cb90769207665c39aef391f4961db09cab9cc5fb794f778af89fc7f0e6"
RPP_WHITEN_SHA = "8eececdaec7927387669e975eec0d84511a976675b6c2536e77e7d035891607d"
NAME = "base_model.model.layer.lora_A.weight"


def ppo(**critic) -> AlgorithmSpec:
    return AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True},
                         critic=critic or None)


# -- 2.1 fields and identity ------------------------------------------------------


def test_grpo_golden_hashes_unchanged():
    assert AlgorithmSpec().sha256() == GRPO_DEFAULT_SHA
    assert AlgorithmSpec(kl={"placement": "loss", "coef": 0.01, "estimator": "k3"}).sha256() \
        == KL_LOSS_SHA
    rpp = AlgorithmSpec(advantage={"estimator": "reinforce_plus_plus", "whiten": True})
    assert rpp.sha256() == RPP_WHITEN_SHA
    for spec in (AlgorithmSpec(), rpp):
        assert "critic" not in json.loads(spec.canonical_json())
        assert "lambd" not in spec.canonical_json()


def test_critic_spec_is_explicit_in_the_identity():
    spec = ppo()
    payload = json.loads(spec.canonical_json())
    assert payload["schema"] == "yeto-rl-algorithm-spec-v2"
    assert payload["critic"] == {
        "critic_lr": None, "critic_lr_warmup": None, "critic_updates_per_step": 1,
        "hl_gauss_bins": None, "init": "copy_actor_backbone", "load": None,
        "lora_alpha": None, "lora_rank": None, "lora_target_modules": None,
        "param_mode": "full", "value_clip": 0.2, "value_loss": "mse", "warmup_steps": 0,
    }
    assert {k: payload["advantage"][k] for k in ("lambd", "lambd_mode", "gae_variant")} == {
        "lambd": 1.0, "lambd_mode": "fixed", "gae_variant": "vanilla"}
    # explicit defaults == implicit defaults; JSON round trip keeps the hash
    assert ppo(value_clip=0.2, param_mode="full").sha256() == spec.sha256()
    assert AlgorithmSpec.from_dict(json.loads(spec.canonical_json())).sha256() == spec.sha256()
    assert AlgorithmSpec(advantage_estimator="ppo").sha256() == spec.sha256()  # v1 ppo


@pytest.mark.parametrize("change", [
    dict(advantage={"estimator": "ppo", "gamma": 0.99}),
    dict(advantage={"estimator": "ppo", "lambd": 0.95}),
    dict(critic={"value_clip": 0.3}),
    dict(critic={"critic_lr": 1e-6}),
    dict(critic={"warmup_steps": 50}),
])
def test_critic_fields_change_the_hash(change):
    base = ppo()
    payload = {"advantage": {"estimator": "ppo"}, "execution": {"needs_critic": True}, **change}
    assert AlgorithmSpec(**payload).sha256() != base.sha256()


@pytest.mark.parametrize("payload, field", [
    (dict(critic={"value_clip": 0.2}), "critic.value_clip"),
    (dict(critic={"param_mode": "full"}), "critic.param_mode"),
    (dict(advantage={"lambd": 0.95}), "advantage.lambd"),
    (dict(advantage={"gae_variant": "vanilla"}), "advantage.gae_variant"),
])
def test_critic_fields_on_a_non_critic_algorithm_rejected(payload, field):
    text = "; ".join(AlgorithmSpec(**payload).rejections())
    assert "[critic_fields_without_critic]" in text and field in text
    assert "only apply to critic algorithms" in text


def test_value_clip_flag_under_grpo_rejected_before_gpu():
    spec, _, _ = af.absorb_extra_argv(AlgorithmSpec(), ["--value-clip", "0.2"])
    assert any("critic.value_clip" in p for p in spec.rejections())
    with pytest.raises(mc.MilesConfigError, match="only apply to critic algorithms"):
        mc.translate_run_config(make_config(), AlgorithmSpec(), extra_argv=["--value-clip", "0.2"])


def test_critic_field_validation():
    with pytest.raises(AlgorithmSpecError, match="critic.param_mode"):
        CriticSpec(param_mode="qlora")
    with pytest.raises(AlgorithmSpecError, match="critic.value_clip"):
        CriticSpec(value_clip=0)
    with pytest.raises(AlgorithmSpecError, match="advantage.lambd"):
        AlgorithmSpec(advantage={"estimator": "ppo", "lambd": 1.5},
                      execution={"needs_critic": True})
    with pytest.raises(AlgorithmSpecError, match="critic.lora_target_modules"):
        CriticSpec(lora_target_modules="q_proj")
    assert CriticSpec(lora_target_modules=["q_proj"]).lora_target_modules == ("q_proj",)


# -- 2.2 translation and absorption -----------------------------------------------


def test_v1_accepts_ppo_with_a_critic():
    spec = AlgorithmSpec(advantage_estimator="ppo")
    assert spec.execution.needs_critic and spec.rejections() == []
    assert spec.schema == "yeto-rl-algorithm-spec-v2"


def test_critic_argv_is_explicit():
    assert af.algorithm_argv(ppo()) == [
        "--gamma", "1.0", "--lambd", "1.0", "--value-clip", "0.2",
        "--num-critic-only-steps", "0"]
    spec = AlgorithmSpec(advantage={"estimator": "ppo", "gamma": 0.99, "lambd": 0.95},
                         execution={"needs_critic": True},
                         critic={"value_clip": 0.3, "critic_lr": 1e-6, "critic_lr_warmup": 5})
    assert af.algorithm_argv(spec) == [
        "--gamma", "0.99", "--lambd", "0.95", "--value-clip", "0.3", "--critic-lr", "1e-06",
        "--critic-lr-warmup-iters", "5", "--num-critic-only-steps", "0"]
    assert critic_argv(AlgorithmSpec()) == []
    loaded = ppo(init="load", load="/ckpt/critic")
    assert af.algorithm_argv(loaded)[-2:] == ["--critic-load", "/ckpt/critic"]


def test_absorb_critic_flags_equals_direct_spec():
    spec, rest, absorbed = af.absorb_extra_argv(AlgorithmSpec(), [
        "--advantage-estimator", "ppo", "--gamma", "0.99", "--lambd", "0.95",
        "--value-clip", "0.3", "--critic-lr", "1e-6", "--micro-batch-size", "1"])
    direct = AlgorithmSpec(advantage={"estimator": "ppo", "gamma": 0.99, "lambd": 0.95},
                           execution={"needs_critic": True},
                           critic={"value_clip": 0.3, "critic_lr": 1e-6})
    assert spec.sha256() == direct.sha256() and rest == ("--micro-batch-size", "1")
    assert absorbed["--gamma"] == "0.99"
    loaded, _, _ = af.absorb_extra_argv(ppo(), ["--critic-load", "/c"])
    assert (loaded.critic.init, loaded.critic.load) == ("load", "/c")
    warm, _, _ = af.absorb_extra_argv(ppo(), ["--num-critic-only-steps", "50"])
    assert warm.critic.warmup_steps == 50
    assert "--num-critic-only-steps', '0'" in str(af.algorithm_argv(warm))  # main stage: 0


def test_absorb_conflicts_name_both_values():
    with pytest.raises(af.AlgorithmFlagConflict, match=r"critic.value_clip.*0\.4.*0\.3"):
        af.absorb_extra_argv(ppo(value_clip=0.3), ["--value-clip", "0.4"])
    with pytest.raises(af.AlgorithmFlagConflict, match="--lambd"):
        af.absorb_extra_argv(ppo(), ["--lambd", "0.9", "--lambd", "0.8"])
    with pytest.raises(af.AlgorithmFlagConflict, match="critic.load"):
        af.absorb_extra_argv(ppo(init="load", load="/a"), ["--critic-load", "/c"])
    tuned, _, _ = af.absorb_extra_argv(ppo(), ["--value-clip", "0.3"])  # filled default
    assert tuned.critic.value_clip == 0.3


def test_gamma_stays_shared_with_reinforce_plus_plus():
    rpp = AlgorithmSpec(advantage={"estimator": "reinforce_plus_plus", "whiten": True,
                                   "gamma": 0.99})
    assert af.algorithm_argv(rpp) == ["--normalize-advantages", "--gamma", "0.99"]
    grpo = AlgorithmSpec(advantage={"gamma": 0.9})
    assert any("[seq_adv_gamma]" in p for p in grpo.rejections())
    tuned = AlgorithmSpec(advantage={"estimator": "ppo", "gamma": 0.99},
                          execution={"needs_critic": True})
    assert tuned.rejections() == []
    assert af.algorithm_argv(tuned).count("--gamma") == 1


def test_dry_run_snapshot_contains_ppo_and_critic_flags():
    allow = ["--rl-allow-unverified-mechanism", "advantage_estimators:ppo",
             "--rl-allow-unverified-mechanism", "execution:critic"]
    result = af.dry_run(["--dry-run", "--extra",
                         "--advantage-estimator ppo --gamma 0.99 --critic-lr 1e-6", *allow])
    assert result["verdict"] == "accepted", result.get("error")
    assert result["miles_argv"] == [
        "--advantage-estimator", "ppo", "--gamma", "0.99", "--lambd", "1.0",
        "--value-clip", "0.2", "--critic-lr", "1e-06", "--num-critic-only-steps", "0"]
    assert result["algorithm_spec"]["critic"]["value_clip"] == 0.2


# -- 2.3 pre-GPU rejections ---------------------------------------------------------


CRITIC_CAPS = dict(advantage_estimators={"grpo", "ppo"},
                   execution=ExecutionCapabilities(critic=True, rollout_logprobs=True))


class _DecoupledLike(LocalOnlySync):
    OUTER_SYNC_KIND = "decoupled"


def _driver(tmp_path, spec, *, sync=None, elastic_hook=None):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)})
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, capabilities=fake_capabilities(**CRITIC_CAPS),
        algorithm=spec, sync=sync or LocalOnlySync(1),
        events=EventTape(tmp_path / "events.jsonl", 0), elastic_hook=elastic_hook,
    )
    return engine, driver


@pytest.mark.parametrize("spec, kwargs, error, message", [
    (AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True},
                   kl={"placement": "reward", "coef": 0.1}), {}, CapabilityMismatch,
     "critic_reward_kl"),
    (ppo(param_mode="lora"), {}, CapabilityMismatch, "planned.*not\\s+implemented"),
    (ppo(), {"elastic_hook": object()}, DriverError, "indep-dp"),
    (ppo(), {"sync": _DecoupledLike(1)}, DriverError, "decoupled outer sync"),
], ids=["kl_coef", "param_mode_lora", "elastic", "decoupled"])
def test_fake_composition_root_rejects_before_any_engine_verb(tmp_path, spec, kwargs, error,
                                                               message):
    engine, driver = _driver(tmp_path, spec, **kwargs)
    with pytest.raises(error, match=message):
        driver.run()
    assert engine.calls == []


def test_declared_critic_passes_the_handshake(tmp_path):
    engine, driver = _driver(tmp_path, ppo())
    assert driver._critic_run_problems() == []
    assert ppo().rejections() == []


@pytest.mark.parametrize("values, message", [
    ({"elastic": True}, "indep-dp"),
    ({"extra_argv": ("--indep-dp",)}, "independent DP"),
    ({"extra_argv": ("--deploy-component", "trainer")}, "deploy-component trainer"),
    ({"extra_argv": ("--critic-num-gpus-per-node", "4"), "actor_num_gpus_per_node": 2},
     "critic-num-gpus-per-node 4 differs"),
    ({"extra_argv": ("--critic-num-nodes=2",), "actor_num_nodes": 1}, "critic-num-nodes 2"),
    ({"sync_preset": "decoupled"}, "decoupled outer sync"),
])
def test_run_level_rejections(values, message):
    problems = critic_run_problems(ppo(), values)
    assert len(problems) == 1 and message in problems[0]
    assert critic_run_problems(AlgorithmSpec(), values) == []  # GRPO unaffected
    assert any(message in p for p in launch_problems(ppo(), values))


def test_equal_critic_gpu_count_is_accepted():
    assert critic_run_problems(ppo(), {"extra_argv": ("--critic-num-gpus-per-node", "2"),
                                       "actor_num_gpus_per_node": 2}) == []


def _ppo_config(**critic):
    cfg = make_config()
    return dataclasses.replace(cfg, algorithm=dataclasses.replace(
        cfg.algorithm, advantage_estimator="ppo",
        critic=CriticRunConfig(**critic) if critic else None))


def test_translate_rejects_mismatched_critic_gpus_before_gpu():
    with pytest.raises(mc.MilesConfigError, match="critic-num-gpus-per-node"):
        mc.translate_run_config(_ppo_config(), ppo(),
                                extra_argv=("--critic-num-gpus-per-node", "7"))
    with pytest.raises(mc.FaultToleranceArgsError):
        mc.translate_run_config(_ppo_config(), ppo(), extra_argv=("--indep-dp",))


# -- 2.4 messages and run config ------------------------------------------------------


def test_capability_message_has_no_legacy_claim():
    caps = fake_capabilities(advantage_estimators={"grpo", "ppo"})
    with pytest.raises(CapabilityMismatch) as error:
        caps.check(layout="lora", placement="colocated", execution_mode="colocated-serial",
                   algorithm=ppo())
    text = str(error.value)
    assert "legacy" not in text and "execution:critic" in text
    rejection = "; ".join(AlgorithmSpec(advantage={"estimator": "ppo"}).rejections())
    assert "needs a critic" in rejection and "legacy" not in rejection


def test_translate_ppo_run_config():
    argv = mc.translate_run_config(_ppo_config(), ppo()).argv
    i = argv.index("--advantage-estimator")
    assert argv[i + 1] == "ppo"
    assert "--critic-load" not in argv  # no warm-up: Miles copies the actor checkpoint
    tail = list(argv[argv.index("--gamma"):argv.index("--gamma") + 8])
    assert tail == ["--gamma", "1.0", "--lambd", "1.0", "--value-clip", "0.2",
                    "--num-critic-only-steps", "0"]


def test_critic_run_config_validation_and_warmup_product():
    digest = "a" * 64
    with pytest.raises(ValueError, match="together"):
        CriticRunConfig(critic_load="/w")
    with pytest.raises(ValueError, match="hex"):
        CriticRunConfig(critic_load="/w", init_sha256="XYZ")
    warm = ppo(warmup_steps=50)
    with pytest.raises(mc.MilesConfigError, match="warm-up"):
        mc.translate_run_config(_ppo_config(), warm)
    argv = mc.translate_run_config(_ppo_config(critic_load="/w", init_sha256=digest), warm).argv
    assert argv[argv.index("--critic-load") + 1] == "/w"
    assert argv[argv.index("--num-critic-only-steps") + 1] == "0"
    with pytest.raises(mc.MilesConfigError, match="no critic"):
        cfg = make_config()
        mc.translate_run_config(dataclasses.replace(cfg, algorithm=dataclasses.replace(
            cfg.algorithm, critic=CriticRunConfig(critic_load="/w", init_sha256=digest))),
            AlgorithmSpec())
    with pytest.raises(mc.MilesConfigError, match="warm-up product"):
        mc.translate_run_config(_ppo_config(critic_load="/w", init_sha256=digest), ppo())
