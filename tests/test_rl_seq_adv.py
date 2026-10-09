"""Spec fields, rejections, zero-gradient rules and capability allowance of
rl-algo-seq-and-adv (tasks 2.1, 2.2, 2.3, 3.1, 3.2, 3.6, 4.1, 4.4, 5.1, 5.5, 6.1, 6.2).

The fake-driver GSPO tests need ``TrainStepMetrics.masked_fraction`` and the
driver calling ``AlgorithmSpec.expects_gradient`` (P0 4.2,
infra-drafts/p0-driver.patch, INFRA-owned driver.py); they skip until that
lands.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.algos import grpo_knobs  # noqa: E402
from yeto.rl.algos import reward_pipeline as rp  # noqa: E402
from yeto.rl.algos import seq_adv as sa  # noqa: E402
from yeto.rl.core import StrictRlInvariantError  # noqa: E402
from yeto.rl.engine import algorithm as alg  # noqa: E402
from yeto.rl.engine.algorithm import (  # noqa: E402
    AlgorithmSpec,
    AlgorithmSpecError,
    check_unverified_allowance,
    load_extensions,
)
from yeto.rl.engine.bridges import LocalOnlySync  # noqa: E402
from yeto.rl.engine.capabilities import CapabilityMismatch  # noqa: E402
from yeto.rl.engine.driver import EventTape, IslandDriver, TrainStepMetrics  # noqa: E402
from yeto.rl.engine.fake import FakeEngine, fake_capabilities  # noqa: E402
from yeto.rl.adapters.miles import algorithm_flags as af  # noqa: E402
from yeto.rl.adapters.miles.entry import miles_capabilities  # noqa: E402

load_extensions()
NAME = "base_model.model.layer.lora_A.weight"
DRIVER_USES_SPEC_RULE = "masked_fraction" in getattr(TrainStepMetrics, "__dataclass_fields__", {})
needs_p0_driver = pytest.mark.skipif(
    not DRIVER_USES_SPEC_RULE,
    reason="needs P0 4.2 driver patch (TrainStepMetrics.masked_fraction + expects_gradient)",
)
DISPATCHER = None


def _dispatcher():
    return rp.dispatcher_ref().to_dict()


def gspo(**loss):
    return AlgorithmSpec(advantage={"estimator": "gspo"}, loss=loss or {"eps_clip": 3e-4,
                                                                         "eps_clip_high": 4e-4})


def rpp(estimator="reinforce_plus_plus", **advantage):
    return AlgorithmSpec(advantage={"estimator": estimator, "whiten": True, **advantage},
                         kl={"placement": "reward", "coef": 0.01})


def transform_spec(transform, *, estimator="grpo", binary=True, gdpo=None, plugins=True,
                   shapers=()):
    advantage = {"estimator": estimator, "transform": transform, "reward_binary": binary,
                 "reward_postprocess": _dispatcher(), "reward_shapers": list(shapers)}
    if gdpo is not None:
        advantage["gdpo"] = gdpo
    spec = AlgorithmSpec(advantage=advantage)
    return grpo_knobs.with_pipeline_plugins(spec) if plugins else spec


GDPO = {"components": [{"name": "format", "weight": 0.5}, {"name": "correctness", "weight": 1.0}],
        "whiten": True}


def _group(std, mean=0.5):
    return SimpleNamespace(reward_std=std, reward_mean=mean, sample_ids=("a", "b"))


def _batch(*groups, **extra):
    return SimpleNamespace(groups=tuple(groups), **extra)


# -- 2.1 GSPO clip hint and transform combination ----------------------------------


@pytest.mark.parametrize("loss", [{}, {"eps_clip": 3e-4}, {"eps_clip_high": 4e-4}])
def test_gspo_without_explicit_clip_is_rejected_with_hint(loss):
    spec = AlgorithmSpec(advantage={"estimator": "gspo"}, loss=loss)
    text = "; ".join(spec.rejections())
    assert "[sequence_ratio_without_clip]" in text
    assert "engine default (Miles --eps-clip 0.2)" in text and "3e-4 (low) / 4e-4 (high)" in text


def test_gspo_with_clip_accepted():
    assert gspo().rejections() == []


@pytest.mark.parametrize("transform", ["maxrl", "mapo", "gdpo"])
def test_gspo_with_advantage_transform_not_opened(transform):
    spec = transform_spec(transform, estimator="gspo", gdpo=GDPO if transform == "gdpo" else None)
    spec = spec.replace(loss={"eps_clip": 3e-4, "eps_clip_high": 4e-4})
    text = "; ".join(spec.rejections())
    assert "[seq_adv_gspo_transform]" in text and "not opened" in text


# -- 2.2 clip fraction -> masked_fraction ---------------------------------------------


def test_clipfrac_present_single_step():
    assert sa.clipfrac_from_losses([{"pg_clipfrac": torch.tensor(0.25)}]) == 0.25
    assert sa.clipfrac_from_losses([{"train/pg_clipfrac": 1.0}]) == 1.0


def test_clipfrac_multi_minibatch_token_weighted():
    losses = [{"pg_clipfrac": 1.0}, {"pg_clipfrac": 0.0}, {"pg_clipfrac": 1.0}]
    assert sa.clipfrac_from_losses(losses, token_counts=[30, 10, 60]) == pytest.approx(0.9)
    assert sa.clipfrac_from_losses(losses) == pytest.approx(2 / 3)
    full = [{"pg_clipfrac": 1.0}] * 3
    assert sa.clipfrac_from_losses(full, token_counts=[5, 7, 9]) == 1.0


@pytest.mark.parametrize("losses,counts", [
    ([], None), ([{"loss": 1.0}], None), ([{"pg_clipfrac": 1.0}, {}], None),
    ([{"pg_clipfrac": math.nan}], None), ([{"pg_clipfrac": 1.0}], [0]),
    ([{"pg_clipfrac": 1.0}], [1, 2]),
])
def test_clipfrac_missing_is_none(losses, counts):
    assert sa.clipfrac_from_losses(losses, token_counts=counts) is None


# -- 2.3 GSPO expects_gradient ---------------------------------------------------------


@pytest.mark.parametrize("masked,expected", [(1.0, False), (1.0 - 1e-12, False), (None, True),
                                             (0.999, True), (0.0, True)])
def test_gspo_expects_gradient(masked, expected):
    batch = _batch(_group(0.5))
    metrics = SimpleNamespace(masked_fraction=masked)
    assert gspo().expects_gradient(batch, metrics) is expected
    assert sa.expects_gradient(gspo(), batch, metrics) is expected
    # default GRPO is never relaxed by a clip fraction
    assert AlgorithmSpec().expects_gradient(batch, metrics) is True


def _driver(tmp_path, spec, caps, **engine_kwargs):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, **engine_kwargs)
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, capabilities=caps, algorithm=spec,
        sync=LocalOnlySync(2), events=EventTape(tmp_path / "events.jsonl", 0),
    )
    return engine, driver


def _gspo_caps():
    return fake_capabilities(advantage_estimators={"grpo", "gspo"},
                             features={"eps_clip", "clip_higher"})


@needs_p0_driver
def test_fake_driver_gspo_fully_clipped_zero_grad_passes(tmp_path):
    _, driver = _driver(tmp_path, gspo(), _gspo_caps(), zero_grad_rounds={1},
                        masked_fraction_rounds={1: 1.0})
    assert driver.run().policy_version == 2


@needs_p0_driver
@pytest.mark.parametrize("fractions", [{}, {1: 0.5}])
def test_fake_driver_gspo_zero_grad_fails_unless_fully_clipped(tmp_path, fractions):
    _, driver = _driver(tmp_path, gspo(), _gspo_caps(), zero_grad_rounds={1},
                        masked_fraction_rounds=fractions)
    with pytest.raises(StrictRlInvariantError) as info:
        driver.run()
    assert info.value.metric == "zero_grad_norm_with_nonzero_advantages"


def test_fake_driver_nonfinite_grad_norm_fails_for_gspo(tmp_path):
    engine, driver = _driver(tmp_path, gspo(), _gspo_caps())
    engine.trainer.step_metrics = lambda: TrainStepMetrics(grad_norm=math.nan, applied_lrs=(1e-5,))
    with pytest.raises(StrictRlInvariantError) as info:
        driver.run()
    assert info.value.metric == "nonfinite_grad_norm"


# -- 3.1 --gamma mapping -----------------------------------------------------------------


def test_gamma_mapped_and_lambd_mapped_by_critic():
    assert "--gamma" in af.mapped_flags()
    assert "--gamma" not in af.UNMAPPED_OBJECTIVE_FLAGS
    assert "--lambd" not in af.UNMAPPED_OBJECTIVE_FLAGS and "--lambd" in af.mapped_flags()
    assert af.MAPPINGS["--gamma"].field == "advantage.gamma"


def test_default_gamma_not_emitted_and_hash_unchanged():
    assert af.algorithm_argv(AlgorithmSpec()) == []
    assert AlgorithmSpec().sha256() == AlgorithmSpec(advantage={"gamma": 1.0}).sha256()
    assert af.algorithm_argv(rpp()) == ["--normalize-advantages"]
    assert af.algorithm_argv(rpp(gamma=0.99)) == ["--normalize-advantages", "--gamma", "0.99"]


def test_gamma_absorbed_from_extra_argv():
    spec, rest, absorbed = af.absorb_extra_argv(rpp(), ["--gamma", "0.99", "--foo"])
    assert spec.advantage.gamma == 0.99 and rest == ("--foo",) and absorbed == {"--gamma": "0.99"}
    # rl-algo-critic-family 2.2: --lambd is mapped now (advantage.lambd), and a
    # non-critic spec refuses it as a critic-only field.
    spec, _, _ = af.absorb_extra_argv(rpp(), ["--lambd", "0.95"])
    assert any("critic_fields_without_critic" in p for p in spec.rejections())


# -- 3.2 gamma / whiten rules ---------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "reinforce_plus_plus_baseline", "ppo"])
def test_gamma_with_other_estimators_rejected(estimator):
    spec = AlgorithmSpec(advantage={"estimator": estimator, "gamma": 0.9, "whiten": True},
                         loss={"eps_clip": 0.2, "eps_clip_high": 0.2})
    text = "; ".join(spec.rejections())
    assert "[seq_adv_gamma]" in text and "only applies to advantage.estimator='reinforce_plus_plus'" in text


def test_non_default_gamma_with_rpp_not_opened():
    text = "; ".join(rpp(gamma=0.99).rejections())
    assert "not opened" in text and "supported values: [1.0]" in text


@pytest.mark.parametrize("value", [0.0, 1.5, -1, math.inf, "0.9", True])
def test_gamma_value_validated(value):
    with pytest.raises(AlgorithmSpecError, match="advantage.gamma"):
        rpp(gamma=value)


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_rpp_family_whiten_rules(estimator):
    assert rpp(estimator).rejections() == []  # reward KL is allowed for the rpp family
    unwhitened = AlgorithmSpec(advantage={"estimator": estimator})
    assert any("[rpp_requires_whiten]" in p for p in unwhitened.rejections())
    # the Miles adapter: refused until the estimator is declared (7.5); once
    # declared, the mandated whiten is claimed by the estimator (P0
    # ESTIMATOR_COMPANIONS, alignment §7b) and the spec is accepted
    caps = miles_capabilities("sha256:" + "0" * 64)
    check = dict(layout="lora", placement="colocated", execution_mode="colocated-serial")
    if estimator in MILES_2A_DECLARED:
        caps.check(**check, algorithm=rpp(estimator))
    else:
        with pytest.raises(CapabilityMismatch, match=f"'{estimator}' not supported"):
            caps.check(**check, algorithm=rpp(estimator))


# -- 3.6 REINFORCE++ / baseline expects_gradient ----------------------------------------


def test_rpp_expects_gradient_rule():
    no_kl = AlgorithmSpec(advantage={"estimator": "reinforce_plus_plus", "whiten": True})
    varied, flat_same, flat_diff = (_batch(_group(0.5)), _batch(_group(0.0), _group(0.0)),
                                    _batch(_group(0.0, 1.0), _group(0.0, 0.0)))
    assert sa.expects_gradient(no_kl, varied) is True
    assert sa.expects_gradient(no_kl, flat_same) is False
    assert sa.expects_gradient(no_kl, flat_diff) is True  # means differ: not all equal
    # reward KL alone: size unreported (0 on round 0, LoRA B=0) -> no verdict, R0 rule
    assert sa.expects_gradient(rpp(), flat_same) is False
    assert sa.expects_gradient(rpp(), flat_diff) is True
    unknown = _batch(SimpleNamespace(reward_std=None, reward_mean=None))
    assert sa.expects_gradient(no_kl, unknown) is True  # unreadable -> expect gradient
    # P0 hook (relax-only) agrees wherever the default already expects a gradient
    assert no_kl.expects_gradient(varied) is True and no_kl.expects_gradient(flat_same) is False


def test_rpp_baseline_expects_gradient_is_default():
    spec = rpp("reinforce_plus_plus_baseline")
    assert spec.expects_gradient(_batch(_group(0.5))) is True
    assert spec.expects_gradient(_batch(_group(0.0), _group(0.0, 1.0))) is False


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_fake_driver_rpp_rounds(tmp_path, estimator):
    caps = fake_capabilities(advantage_estimators={"grpo", estimator},
                             features={"whiten_advantages"})
    # no reward KL: a round with identical rewards everywhere has all-equal advantages
    no_kl = AlgorithmSpec(advantage={"estimator": estimator, "whiten": True})
    _, driver = _driver(tmp_path, no_kl, caps, zero_grad_rounds={1},
                        constant_reward_rounds={1})
    assert driver.run().policy_version == 2  # not expected -> zero grad is fine
    _, driver = _driver(tmp_path, rpp(estimator), caps, zero_grad_rounds={1})
    with pytest.raises(StrictRlInvariantError, match=r"grad_norm (is )?0"):
        driver.run()  # expected -> zero grad fails


# -- 4.1 transform rejections --------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"])
@pytest.mark.parametrize("transform", ["maxrl", "mapo"])
def test_transform_requires_grpo(estimator, transform):
    spec = transform_spec(transform, estimator=estimator).replace(
        advantage={**transform_spec(transform).advantage.to_dict(), "estimator": estimator,
                   "whiten": True},
        loss={"eps_clip": 0.2, "eps_clip_high": 0.2})
    text = "; ".join(spec.rejections())
    assert "only combines with advantage.estimator='grpo'" in text


@pytest.mark.parametrize("transform", ["maxrl", "mapo"])
def test_binary_transforms_need_binary_reward_and_no_overlong_penalty(transform):
    assert transform_spec(transform).rejections() == []
    text = "; ".join(transform_spec(transform, binary=False).rejections())
    assert "[binary_reward_required]" in text and transform in text
    shaped = transform_spec(transform, shapers=[{"name": "overlong_penalty", "max_length": 8,
                                                 "cache_length": 4}])
    assert any("[seq_adv_binary_overlong]" in p for p in shaped.rejections())


def test_transform_identity_required():
    text = "; ".join(transform_spec("maxrl", plugins=False).rejections())
    assert "[grpo_knobs_pipeline_plugins]" in text and "yeto.rl.algos.seq_adv" in text
    listed = [p.path for p in transform_spec("maxrl").plugins]
    assert any(p.startswith("yeto.rl.algos.seq_adv.") for p in listed)


# -- 4.4 MaxRL all-wrong round in the fake driver ----------------------------------------


@pytest.mark.parametrize("transform", ["maxrl", "mapo"])
def test_fake_driver_all_wrong_round_zero_grad_passes(tmp_path, transform):
    caps = fake_capabilities(features={transform, "plugins"},
                             reward_postprocessors={"custom_reward_postprocess"})
    _, driver = _driver(tmp_path, transform_spec(transform), caps, zero_grad_rounds={1},
                        constant_reward_rounds={1})
    assert driver.run().policy_version == 2


# -- 5.1 GDPO declaration ---------------------------------------------------------------------


def test_gdpo_declaration_normalized_and_hashed():
    spec = transform_spec("gdpo", gdpo=GDPO)
    assert spec.rejections() == []
    stored = json.loads(spec.canonical_json())["advantage"]["gdpo"]
    assert [c["name"] for c in stored["components"]] == ["correctness", "format"]
    reordered = {"components": list(reversed(GDPO["components"])), "whiten": True}
    assert transform_spec("gdpo", gdpo=reordered).sha256() == spec.sha256()
    heavier = {"components": [{"name": "format", "weight": 0.75},
                              {"name": "correctness", "weight": 1.0}]}
    assert transform_spec("gdpo", gdpo=heavier).sha256() != spec.sha256()


@pytest.mark.parametrize("gdpo,match", [
    ({"components": []}, "components is empty"),
    ({"components": [{"name": "a", "weight": 1}, {"name": "a", "weight": 2}]}, "repeat"),
    ({"components": [{"name": "a", "weight": math.nan}]}, "finite number"),
    ({"components": [{"name": "a", "weight": math.inf}]}, "finite number"),
    ({"components": [{"name": "a"}]}, "exactly \\{name, weight\\}"),
    ({"components": [{"name": "", "weight": 1}]}, "non-empty string"),
    ({"weights": {}}, "must be an object"),
])
def test_gdpo_declaration_rejected(gdpo, match):
    with pytest.raises(AlgorithmSpecError, match=match):
        transform_spec("gdpo", gdpo=gdpo)


def test_gdpo_selection_consistency():
    assert any("[seq_adv_gdpo]" in p for p in transform_spec("gdpo").rejections())
    maxrl_with_gdpo = transform_spec("maxrl", gdpo=GDPO)
    assert any("requires advantage.transform='gdpo'" in p for p in maxrl_with_gdpo.rejections())
    unwhitened = transform_spec("gdpo", gdpo={**GDPO, "whiten": False})
    assert any("whiten=false is expressible but not opened" in p for p in unwhitened.rejections())


def test_gdpo_runtime_attrs_only_when_selected():
    assert "yeto_rl_seq_adv" not in AlgorithmSpec().to_legacy_runtime_attrs()
    attrs = transform_spec("gdpo", gdpo=GDPO).to_legacy_runtime_attrs()
    config = attrs[sa.SEQ_ADV_ATTR]["config"]
    assert config["gdpo"]["components"][0] == {"name": "correctness", "weight": 1.0}


# -- 5.5 GDPO expects_gradient ---------------------------------------------------------------


def test_gdpo_expects_gradient():
    spec = transform_spec("gdpo", gdpo=GDPO)
    assert sa.expects_gradient(spec, _batch(_group(0.0), nonzero_advantages=3)) is True
    assert sa.expects_gradient(spec, _batch(_group(0.5), nonzero_advantages=0)) is False
    assert sa.expects_gradient(spec, _batch(_group(0.0))) is True  # unreadable -> expect
    # P0 hook: relaxes when the dispatcher reports no non-zero advantage
    assert spec.expects_gradient(_batch(_group(0.5), nonzero_advantages=0)) is False
    assert spec.expects_gradient(_batch(_group(0.5), nonzero_advantages=2)) is True
    assert spec.expects_gradient(_batch(_group(0.5))) is True


# -- 6.1 / 6.2 allowance and fake declaration ------------------------------------------------


def _mechanism_spec(name):
    return {
        "gspo": gspo(),
        "reinforce_plus_plus": rpp(),
        "reinforce_plus_plus_baseline": rpp("reinforce_plus_plus_baseline"),
        "maxrl": transform_spec("maxrl"),
        "mapo": transform_spec("mapo"),
        "gdpo": transform_spec("gdpo", gdpo=GDPO),
    }[name]


# Declared besides the mechanism itself (the rest of each spec, all R0-declared in Miles
# except the dispatcher / plugins / clip / whiten features).
SUPPORTING = ["custom_reward_postprocess", "plugins", "eps_clip", "clip_higher",
              "whiten_advantages"]


def fake_seq_adv_capabilities(**overrides):
    """The fake engine declaring all six mechanisms (task 6.2; see 2a-shared.patch)."""

    return fake_capabilities(
        advantage_estimators={"grpo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"},
        reward_postprocessors={"custom_reward_postprocess"},
        features={"maxrl", "mapo", "gdpo", "plugins", "eps_clip", "clip_higher",
                  "whiten_advantages"},
        **overrides,
    )


@pytest.mark.parametrize("name", sorted(sa.MECHANISMS))
def test_allowance_launches_single_island_and_refuses_otherwise(tmp_path, name):
    spec = _mechanism_spec(name)
    assert spec.rejections() == []
    assert sa.MECHANISMS[name] in spec.required_mechanisms()
    # not allowed: refused before any engine call
    engine, driver = _driver(tmp_path, spec, fake_capabilities())
    with pytest.raises(CapabilityMismatch, match=f"'{name}' not supported"):
        driver.run()
    assert engine.calls == []
    # allowed on a single island without outer sync (plus the supporting mechanisms)
    needed = sorted(f"{d}:{n}" for d, n in spec.required_mechanisms() if n in {name, *SUPPORTING})
    assert f"{sa.MECHANISMS[name][0]}:{name}" in needed
    names = check_unverified_allowance(needed, islands=1, outer_sync=False)
    engine, driver = _driver(tmp_path, spec, fake_capabilities().with_unverified(names))
    assert driver.run().policy_version == 2
    for islands, outer in ((2, True), (1, True)):
        with pytest.raises(AlgorithmSpecError, match="single-island run without outer sync"):
            check_unverified_allowance(needed, islands=islands, outer_sync=outer)


@pytest.mark.parametrize("name", sorted(sa.MECHANISMS))
def test_fake_declaration_launches_each_mechanism(tmp_path, name):
    _, driver = _driver(tmp_path, _mechanism_spec(name), fake_seq_adv_capabilities())
    assert driver.run().policy_version == 2


# Mechanisms of this change the Miles adapter declares (7.5; one per commit).
MILES_2A_DECLARED = {"maxrl", "mapo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "gdpo"}


def test_miles_adapter_declarations_follow_g1():
    """7.5: exactly MILES_2A_DECLARED of this change is declared; the rest refused."""

    caps = miles_capabilities("sha256:" + "0" * 64)
    mine = {"gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline",
            "maxrl", "mapo", "gdpo"}
    estimators = {"gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"}
    assert (set(caps.advantage_estimators) | set(caps.features)) & mine == MILES_2A_DECLARED
    assert caps.advantage_estimators == frozenset({"grpo"} | (MILES_2A_DECLARED & estimators))
    check = dict(layout="lora", placement="colocated", execution_mode="colocated-serial")
    for name in sorted(mine):
        try:
            caps.check(**check, algorithm=_mechanism_spec(name))
            text = ""
        except CapabilityMismatch as exc:
            text = str(exc)
        if name in MILES_2A_DECLARED:
            assert f"'{name}' not supported" not in text, text
        else:  # undeclared mechanisms of this change are still refused
            assert f"'{name}' not supported" in text


def test_default_spec_unchanged():
    spec = AlgorithmSpec()
    assert spec.rejections() == [] and spec.to_legacy_runtime_attrs() == {
        "yeto_rl_dynamic_sampling_max_replacements": None}
    assert af.algorithm_argv(spec) == []


def test_p0_hook_tightens_for_gdpo_and_rpp():
    """D8 via the P0 hook (True verdicts require a gradient the R0 rule misses)."""

    flat = _batch(_group(0.0), _group(0.0))
    assert rpp().expects_gradient(flat) is False  # reward KL alone: no verdict (round 0 KL = 0)
    no_kl = AlgorithmSpec(advantage={"estimator": "reinforce_plus_plus", "whiten": True})
    assert no_kl.expects_gradient(_batch(_group(0.0, 1.0), _group(0.0, 0.0))) is True
    assert no_kl.expects_gradient(flat) is False
    gdpo = transform_spec("gdpo", gdpo=GDPO)
    assert gdpo.expects_gradient(_batch(_group(0.0), nonzero_advantages=2)) is True
    assert gdpo.expects_gradient(_batch(_group(0.0), nonzero_advantages=0)) is False
    # default GRPO untouched
    assert AlgorithmSpec().expects_gradient(flat) is False


def test_fake_driver_rpp_reward_kl_round0_identical_rewards_is_not_a_failure(tmp_path):
    """Review item 6: round 0 (LoRA B = 0, policy == reference, reward KL = 0) with
    identical rewards everywhere has all-equal advantages; zero grad must pass."""

    caps = fake_capabilities(advantage_estimators={"grpo", "reinforce_plus_plus"},
                             features={"whiten_advantages"})
    _, driver = _driver(tmp_path, rpp(), caps, zero_grad_rounds={0, 1},
                        constant_reward_rounds={0, 1})
    assert driver.run().policy_version == 2
    # other cases are not relaxed: differing group means still require a gradient
    no_kl = AlgorithmSpec(advantage={"estimator": "reinforce_plus_plus", "whiten": True})
    assert rpp().expects_gradient(_batch(_group(0.0, 1.0), _group(0.0, 0.0))) is True
    assert no_kl.expects_gradient(_batch(_group(0.5))) is True


@pytest.mark.parametrize("cp,rejected", [(1, False), (2, True)])
def test_gspo_context_parallel_refused_before_launch(cp, rejected):
    problems = alg.launch_problems(gspo(), {"rollout_batch_size": 4,
                                            "rollout_max_response_len": 384,
                                            "context_parallel_size": cp, "multi_lora": False})
    assert any("[seq_adv]" in p and "context parallel" in p for p in problems) is rejected
    assert not any("[seq_adv]" in p for p in alg.launch_problems(
        AlgorithmSpec(), {"rollout_batch_size": 4, "rollout_max_response_len": 384,
                          "context_parallel_size": 2, "multi_lora": False}))


# -- 5.5 GDPO through the fake driver (INFRA R2: RolloutBatchHandle.nonzero_advantages) --------


def _gdpo_driver(tmp_path, nonzero, **engine_kwargs):
    import dataclasses

    caps = fake_capabilities(features={"gdpo"},
                             reward_postprocessors={"custom_reward_postprocess"})
    engine, driver = _driver(tmp_path, transform_spec("gdpo", gdpo=GDPO), caps, **engine_kwargs)
    generate = engine.rollout.generate

    def with_count(*args, **kwargs):
        batch = generate(*args, **kwargs)
        return dataclasses.replace(batch, nonzero_advantages=nonzero(batch.rollout_id))

    engine.rollout.generate = with_count
    return driver


def test_fake_driver_gdpo_no_nonzero_advantage_zero_grad_passes(tmp_path):
    driver = _gdpo_driver(tmp_path, lambda r: 0 if r == 1 else 5, zero_grad_rounds={1})
    assert driver.run().policy_version == 2


@pytest.mark.parametrize("count", [3, None])  # reported non-zero, or unreadable
def test_fake_driver_gdpo_expected_zero_grad_fails(tmp_path, count):
    # scalar reward constant (R0 rule: no gradient) but the reward vector is not
    driver = _gdpo_driver(tmp_path, lambda r: count, zero_grad_rounds={1},
                          constant_reward_rounds={1})
    with pytest.raises(StrictRlInvariantError, match=r"grad_norm (is )?0"):
        driver.run()


def test_transforms_report_round_counter_for_the_metadata_hook(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import rollout_meta_hook as hook

    sink = f"dir:{tmp_path}"
    monkeypatch.setenv(hook.META_SINK_ENV, sink)
    hook.put_policy_token("yeto:7:" + "a" * 64, sink)
    spec = transform_spec("maxrl")

    class S(SimpleNamespace):
        def get_reward_value(self, args):
            return self.reward

    args = SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True,
                           grpo_std_normalization=True, n_samples_per_prompt=4,
                           rollout_batch_size=2, multi_lora=False)
    for key, value in spec.to_legacy_runtime_attrs().items():
        setattr(args, key, value)
    # Sample.rollout_id is a trajectory key (here 100+i), not the training round (7)
    samples = [S(reward=r, group_index=g, index=i, rollout_id=100 + i, metadata={},
                 response_length=4)
               for i, (g, r) in enumerate([(0, 1.0), (0, 0.0), (1, 0.0), (1, 0.0)])]
    rp.post_process(args, samples)
    records = [json.loads(p.read_text()) for p in tmp_path.iterdir()
               if p.name != hook.POLICY_TOKEN_FILE and not p.name.startswith(".")]
    counts = [r for r in records if r.get("nonzero_advantages") is not None]
    assert [(r["rollout_id"], r["nonzero_advantages"]) for r in counts] == [(7, 2)]


def test_no_sink_no_round_record():
    assert sa._current_round_id() is None  # CPU: no sink env, no Ray


def test_trainer_reads_gspo_clipfrac_through_seq_adv():
    """2.2: the adapter (INFRA R1) fills masked_fraction from per-step pg_clipfrac."""

    from yeto.rl.adapters.miles import trainer

    assert "gspo" in trainer.CLIPFRAC_MASKED_ESTIMATORS
    assert trainer.clipfrac_masked_fraction([{"pg_clipfrac": 1.0}, {"pg_clipfrac": 1.0}]) == 1.0
    assert trainer.clipfrac_masked_fraction([{"pg_clipfrac": 1.0}, {"pg_clipfrac": 0.0}]) == 0.5
    assert trainer.clipfrac_masked_fraction([{"pg_clipfrac": 1.0}, {}]) is None
    tokens = [{"pg_clipfrac": 1.0, "loss_tokens": 30}, {"pg_clipfrac": 0.0, "loss_tokens": 10}]
    assert trainer.clipfrac_masked_fraction(tokens) == pytest.approx(0.75)


@pytest.mark.parametrize("name", ["gspo", "rpp", "rpp_baseline", "maxrl", "mapo", "gdpo"])
def test_example_specs_are_current(name):
    """G1 uses these files: their plugin hashes must match the current sources."""

    from pathlib import Path

    path = (Path(__file__).resolve().parents[1]
            / "openspec/changes/rl-algo-seq-and-adv/examples" / f"{name}.json")
    spec = AlgorithmSpec.from_json_file(str(path))
    assert spec.rejections() == [], "regenerate with examples/make_examples.py"
