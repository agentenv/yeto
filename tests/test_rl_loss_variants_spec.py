"""rl-algo-loss-variants (route B) on the yeto side: reference self-checks (2.1),
spec fields and hash (2.2), rejections (2.3), gradient rules (2.4), Miles
adapter not opened (2.5), fake declaration (5.1), flag translation and
absorption (4.4 yeto part, before the pin moves)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import rl_loss_variant_reference as ref  # noqa: E402

from yeto.rl.algos import loss_variants as lv  # noqa: E402
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
from yeto.rl.engine.miles_adapter import algorithm_flags as af  # noqa: E402
from yeto.rl.engine.miles_adapter.entry import miles_capabilities  # noqa: E402

load_extensions()
NAME = "base_model.model.layer.lora_A.weight"
FP = "sha256:" + "0" * 64
RUN = {"rollout_batch_size": 4, "rollout_max_response_len": 384,
       "context_parallel_size": 1, "multi_lora": False}


CISPO_CLIP = {"eps_clip": 0.2, "eps_clip_high": 0.28, "aggregation": "token"}


def spec(variant, **loss):
    if variant == "cispo":
        loss = {**CISPO_CLIP, **loss}  # CISPO: explicit clip range, token aggregation
    return AlgorithmSpec.from_dict({"schema": alg.ALGORITHM_SPEC_SCHEMA_V2,
                                    "loss": {"policy_loss_variant": variant, **loss}})


def _t(*values):
    return torch.tensor(values, dtype=torch.float64)


# -- 2.1 reference formulas: hand-computed values --------------------------------------


def test_reference_cispo_hand_values():
    logp = _t(-0.5, -1.0, -0.2).requires_grad_()
    old = _t(-0.5 - math.log(1.5), -1.0 - math.log(0.5), -0.2)  # rho = 1.5, 0.5, 1.0
    adv = _t(1.0, -2.0, 1.0)
    loss = ref.cispo(logp, old, adv, 0.2, 0.28)
    # w = clip(rho) = 1.28, 0.8, 1.0 ; loss = -w * A * logp
    assert loss.detach().tolist() == pytest.approx([0.64, -1.6, 0.2])
    loss.sum().backward()
    # out-of-range tokens keep a gradient: d/dlogp = -w * A
    assert logp.grad.tolist() == pytest.approx([-1.28, 1.6, -1.0])


def test_reference_sapo_hand_values():
    logp = _t(0.0, 0.0, math.log(2.0), 0.0)
    old = _t(0.0, 0.0, 0.0, 0.0)  # rho = 1, 1, 2, 1
    adv = _t(1.0, -1.0, 1.0, 0.0)
    loss = ref.sapo(logp, old, adv, 1.0, 1.05)
    # f(1) = 0.5 * 4 / tau ; f(2) with tau 1 = sigmoid(1) * 4
    expected = [-2.0, 2.0 / 1.05, -4 / (1 + math.exp(-1.0)), 0.0]
    assert loss.tolist() == pytest.approx(expected)
    r = torch.tensor(1.0, requires_grad=True)
    (torch.sigmoid(1.0 * (r - 1)) * 4 / 1.0).backward()
    assert r.grad.item() == pytest.approx(1.0)  # PPO-like slope on-policy


def test_reference_gmpo_hand_values_and_log_clip():
    logp = _t(0.1, 0.5, 0.0)
    old = _t(0.0, 0.0, 0.0)
    mask = torch.tensor([1, 1, 0])
    loss, clip = ref.gmpo_sequence(logp, old, _t(1.0, 1.0, 1.0), mask)
    # A > 0: l = [0.1, 0.4 (0.5 clipped at delta_h)], mean 0.25 over 2 valid tokens
    assert loss[0].item() == pytest.approx(-math.exp(0.25))
    assert clip.item() == pytest.approx(0.5)
    # A < 0: one-sided, log r = 0.5 > delta_h is NOT clipped -> l = [0.1, 0.5]
    neg, neg_clip = ref.gmpo_sequence(logp, old, _t(-1.0, -1.0, -1.0), mask)
    assert neg[0].item() == pytest.approx(math.exp(0.3))
    assert neg_clip.item() == 0.0
    # a clipped token contributes no gradient
    lp = logp.clone().requires_grad_()
    ref.gmpo_sequence(lp, old, _t(1.0, 1.0, 1.0), mask)[0].sum().backward()
    assert lp.grad[1].item() == 0.0 and lp.grad[0].item() != 0.0


def test_reference_gmpo_positive_advantage_far_below_keeps_gradient():
    lp = _t(-2.0, 0.0).requires_grad_()  # log r = -2 << -delta_l
    loss, clip = ref.gmpo_sequence(lp, _t(0.0, 0.0), _t(1.0, 1.0), torch.tensor([1, 1]))
    assert loss[0].item() == pytest.approx(-math.exp(-1.0))  # not clipped: mean(-2, 0)
    assert clip.item() == 0.0
    loss[0].backward()
    assert lp.grad[0].item() != 0.0


def test_reference_gmpo_asymmetric_bounds_and_negative_advantage():
    old = _t(0.0, 0.0, 0.0)
    mask = torch.tensor([1, 1, 1])
    logp = _t(-0.5, 0.5, 0.1)
    # A < 0, delta_l=0.2, delta_h=0.6: -0.5 -> -0.2 (clipped), 0.5 kept, 0.1 kept
    loss, clip = ref.gmpo_sequence(logp, old, _t(-2.0, -2.0, -2.0), mask, 0.2, 0.6)
    assert loss[0].item() == pytest.approx(2.0 * math.exp((-0.2 + 0.5 + 0.1) / 3))
    assert clip.item() == pytest.approx(1 / 3)
    # A > 0 with the same bounds: 0.5 < delta_h 0.6 kept, -0.5 kept -> nothing clipped
    loss, clip = ref.gmpo_sequence(logp, old, _t(1.0, 1.0, 1.0), mask, 0.2, 0.6)
    assert loss[0].item() == pytest.approx(-math.exp(0.1 / 3))
    assert clip.item() == 0.0
    # zero advantage: no loss, clip fraction undefined (no A != 0 token)
    loss, clip = ref.gmpo_sequence(logp, old, _t(0.0, 0.0, 0.0), mask)
    assert loss.abs().sum().item() == 0.0 and clip is None


def test_reference_gmpo_context_parallel_matches_unsplit():
    g = torch.Generator().manual_seed(0)
    logp = torch.randn(7, generator=g, dtype=torch.float64) * 0.3
    old = torch.randn(7, generator=g, dtype=torch.float64) * 0.3
    adv = torch.full((7,), 0.7, dtype=torch.float64)
    mask = torch.tensor([1, 1, 0, 1, 1, 1, 0])
    whole, _ = ref.gmpo_sequence(logp, old, adv, mask)
    for shards in (2, 3):
        parts = ref.gmpo_sharded(logp, old, adv, mask, shards)
        assert torch.allclose(torch.cat(parts), whole)


def test_reference_composition_with_tis_and_icepop():
    logp, old, rollout = _t(-0.5, -1.0), _t(-0.6, -1.0), _t(-0.5, -1.5)
    adv = _t(1.0, -1.0)
    base = ref.cispo(logp, old, adv, 0.2, 0.2)
    w = ref.tis_weight(old, rollout, 0.0, 2.0)  # exp(-0.1), exp(0.5)
    assert (base * w).tolist() == pytest.approx(
        [base[0].item() * math.exp(-0.1), base[1].item() * math.exp(0.5)])
    ice = ref.icepop_weight(old, rollout, 0.5, 1.5)  # second token outside -> 0
    assert (base * ice).tolist() == pytest.approx([base[0].item() * math.exp(-0.1), 0.0])


# -- 2.2 fields and hash --------------------------------------------------------------


def test_default_variant_hash_and_argv_unchanged():
    default = AlgorithmSpec()
    explicit = spec("policy_loss")
    assert explicit.sha256() == default.sha256()
    assert "policy_loss_variant" not in default.canonical_json()
    assert af.algorithm_argv(explicit) == af.algorithm_argv(default) == []
    # stray defaults of variant parameters do not change the identity either
    assert spec("policy_loss", sapo_tau_pos=1.0, gmpo_log_clip_low=0.4).sha256() == default.sha256()


def test_each_variant_hashes_differently_and_carries_its_parameters():
    hashes = {AlgorithmSpec().sha256()}
    for name in lv.VARIANTS:
        s = spec(name)
        hashes.add(s.sha256())
        loss = json.loads(s.canonical_json())["loss"]
        assert loss["policy_loss_variant"] == name
        own = {p for p, _ in lv.VARIANT_PARAMS[name]}
        others = {p for v, ps in lv.VARIANT_PARAMS.items() if v != name for p, _ in ps}
        assert own <= set(loss) and not (others & set(loss))  # defaults emitted when selected
    assert len(hashes) == 4
    assert spec("sapo", sapo_tau_pos=1.2).sha256() != spec("sapo").sha256()
    assert spec("gmpo", gmpo_log_clip_high=0.3).sha256() != spec("gmpo").sha256()
    assert spec("cispo", eps_clip=0.1, eps_clip_high=0.28).sha256() != spec("cispo").sha256()
    # round trip
    for name in lv.VARIANTS:
        s = spec(name, **({"sapo_tau_neg": 2.0} if name == "sapo" else {}))
        assert AlgorithmSpec.from_dict(json.loads(s.canonical_json())) == s


@pytest.mark.parametrize("field", ["sapo_tau_pos", "sapo_tau_neg",
                                   "gmpo_log_clip_low", "gmpo_log_clip_high"])
@pytest.mark.parametrize("bad", [0, -1.0, math.inf, math.nan, "1", True])
def test_invalid_parameters_name_the_field(field, bad):
    name = field.split("_")[0]
    with pytest.raises(AlgorithmSpecError, match=f"loss.{field}"):
        spec(name, **{field: bad})


def test_unknown_variant_rejected():
    with pytest.raises(AlgorithmSpecError, match="loss.policy_loss_variant"):
        spec("ppo")


@pytest.mark.parametrize("variant,field", [
    ("policy_loss", "sapo_tau_pos"), ("cispo", "sapo_tau_neg"), ("gmpo", "sapo_tau_pos"),
    ("sapo", "gmpo_log_clip_low"), ("cispo", "gmpo_log_clip_high"),
])
def test_parameter_of_another_variant_rejected(variant, field):
    text = "; ".join(spec(variant, **{field: 0.7}).rejections())
    assert "[loss_variant_params]" in text and f"loss.{field}" in text


# -- 2.3 rejections, before any engine verb -------------------------------------------


def _driver(tmp_path, algorithm, caps, **engine_kwargs):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, **engine_kwargs)
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, capabilities=caps, algorithm=algorithm,
        sync=LocalOnlySync(3), events=EventTape(tmp_path / "events.jsonl", 0),
    )
    return engine, driver


REJECTED = [
    ("loss_variant_sequence_ratio",
     dict(advantage={"estimator": "gspo"}, loss={"eps_clip": 3e-4, "eps_clip_high": 4e-4}),
     "advantage.estimator='grpo'"),
    ("loss_variant_dual_clip", dict(loss={"eps_clip_c": 3.0}), "Drop loss.eps_clip_c"),
]


@pytest.mark.parametrize("variant", lv.VARIANTS)
@pytest.mark.parametrize("rule,extra,alternative", REJECTED, ids=[r[0] for r in REJECTED])
def test_rejections_fail_in_fake_root_before_any_engine_verb(tmp_path, variant, rule, extra,
                                                             alternative):
    payload = {"schema": alg.ALGORITHM_SPEC_SCHEMA_V2, **extra}
    payload["loss"] = {**payload.get("loss", {}), "policy_loss_variant": variant}
    if variant in ("sapo", "gmpo"):
        payload["loss"].pop("eps_clip", None)
        payload["loss"].pop("eps_clip_high", None)
    else:
        payload["loss"] = {**CISPO_CLIP, **payload["loss"]}
    s = AlgorithmSpec.from_dict(payload)
    text = "; ".join(s.rejections())
    assert f"[{rule}]" in text and alternative in text
    caps = fake_capabilities(advantage_estimators={"grpo", "gspo"},
                             features={"eps_clip", "clip_higher", "dual_clip"})
    engine, driver = _driver(tmp_path, s, caps)
    with pytest.raises(CapabilityMismatch, match=rule):
        driver.run()
    assert engine.calls == []


def test_gmpo_gspo_message_names_both_ratios():
    s = AlgorithmSpec.from_dict({"schema": alg.ALGORITHM_SPEC_SCHEMA_V2,
                                 "advantage": {"estimator": "gspo"},
                                 "loss": {"policy_loss_variant": "gmpo"}})
    assert "both define the policy ratio" in "; ".join(s.rejections())


def test_variant_with_custom_loss_rejected():
    ref_ = alg.PluginRef.from_path("yeto.rl.engine.algorithm.plugin_source_sha256")
    s = AlgorithmSpec.from_dict({"schema": alg.ALGORITHM_SPEC_SCHEMA_V2, "loss": {
        "variant": "custom_loss", "custom_loss": ref_.to_dict(), "policy_loss_variant": "cispo",
        **CISPO_CLIP}})
    assert "[loss_variant_custom_loss]" in "; ".join(s.rejections())


@pytest.mark.parametrize("variant", ["sapo", "gmpo"])
def test_unused_clip_bounds_rejected(variant):
    text = "; ".join(spec(variant, eps_clip=0.2).rejections())
    assert "[loss_variant_unused_clip]" in text
    assert spec("cispo", eps_clip=0.2, eps_clip_high=0.28).rejections() == []


def test_variants_combine_with_tis_and_entropy():
    for name in lv.VARIANTS:
        s = AlgorithmSpec.from_dict({
            "schema": alg.ALGORITHM_SPEC_SCHEMA_V2,
            "loss": {"policy_loss_variant": name, **(CISPO_CLIP if name == "cispo" else {})},
            "correction": {"method": "tis", "tis_clip": 2.0, "tis_clip_low": 0.0},
            "entropy_coef": 0.001})
        assert s.rejections() == [], name


# -- 2.4 gradient rules ----------------------------------------------------------------


def _group(std):
    return SimpleNamespace(reward_std=std, reward_mean=0.5, sample_ids=("a", "b"))


def _batch(*groups):
    return SimpleNamespace(groups=tuple(groups))


def _metrics(masked, clip=None):
    return SimpleNamespace(masked_fraction=masked, clip_fraction=clip)


@pytest.mark.parametrize("variant", ["cispo", "sapo"])
def test_cispo_sapo_expect_gradient_even_fully_clipped(variant):
    s = spec(variant)
    assert s.expects_gradient(_batch(_group(0.5)), _metrics(1.0)) is True
    assert s.expects_gradient(_batch(_group(0.0)), _metrics(None)) is False  # GRPO rule


def test_gmpo_full_clip_lifts_expectation_unknown_falls_back():
    s = spec("gmpo")
    batch = _batch(_group(0.5))
    assert s.gradient_expectation(batch, _metrics(None, 1.0)) == (
        False, "gradient_rule:loss_variant_gmpo_full_clip (losses:gmpo)")
    for clip in (None, 0.99, math.nan, True, "1.0"):
        assert s.expects_gradient(batch, _metrics(None, clip)) is True
    assert s.expects_gradient(batch, None) is True


def test_gmpo_rule_reads_clip_fraction_not_the_correction_mask():
    # corrections fill masked_fraction with their own mask; GMPO must not read it
    s = spec("gmpo")
    batch = _batch(_group(0.5))
    assert s.expects_gradient(batch, _metrics(1.0, 0.3)) is True
    assert s.expects_gradient(batch, _metrics(0.2, 1.0)) is False


def test_gmpo_rule_does_not_touch_other_variants():
    assert lv.gmpo_gradient_rule(spec("cispo"), None, _metrics(1.0, 1.0)) is None
    assert AlgorithmSpec().expects_gradient(_batch(_group(0.5)), _metrics(1.0, 1.0)) is True
    # CISPO: every ratio out of range (clip fraction 1) still expects a gradient
    assert spec("cispo").expects_gradient(_batch(_group(0.5)), _metrics(None, 1.0)) is True


def _caps():
    return fake_capabilities(features={"eps_clip", "clip_higher"})


@pytest.mark.parametrize("variant", lv.VARIANTS)
def test_nonfinite_grad_norm_fails_for_every_variant(tmp_path, variant):
    engine, driver = _driver(tmp_path, spec(variant), _caps())
    engine.trainer.step_metrics = lambda: TrainStepMetrics(
        grad_norm=math.nan, masked_fraction=1.0, applied_lrs=(1e-5,))
    with pytest.raises(StrictRlInvariantError) as info:
        driver.run()
    assert info.value.metric == "nonfinite_grad_norm"


def _with_clip_fraction(engine, rounds):
    import dataclasses

    original = engine.trainer.step_metrics

    def metrics():
        m = original()
        return dataclasses.replace(m, clip_fraction=rounds.get(m.train_step))

    engine.trainer.step_metrics = metrics


def test_fake_driver_gmpo_full_clip_round_passes_cispo_fails(tmp_path):
    engine, driver = _driver(tmp_path, spec("gmpo"), _caps(), zero_grad_rounds={1})
    _with_clip_fraction(engine, {1: 1.0})
    assert driver.run().policy_version == 3
    engine, driver = _driver(tmp_path / "c", spec("cispo"), _caps(), zero_grad_rounds={1})
    _with_clip_fraction(engine, {1: 1.0})
    with pytest.raises(StrictRlInvariantError) as info:
        driver.run()
    assert info.value.metric == "zero_grad_norm_with_nonzero_advantages"


def test_fake_driver_gmpo_correction_mask_does_not_relax(tmp_path):
    engine, driver = _driver(tmp_path, spec("gmpo"), _caps(), zero_grad_rounds={1},
                             masked_fraction_rounds={1: 1.0})
    _with_clip_fraction(engine, {1: 0.5})
    with pytest.raises(StrictRlInvariantError):
        driver.run()


# -- 2.5 / 5.1 declarations ------------------------------------------------------------


@pytest.mark.parametrize("variant", lv.VARIANTS)
def test_miles_adapter_does_not_open_variants(variant):
    caps = miles_capabilities(FP)
    assert variant not in caps.losses
    with pytest.raises(CapabilityMismatch, match=f"losses mechanism {variant!r} not supported"):
        caps.check(layout="lora", placement="colocated", execution_mode="colocated-serial",
                   algorithm=spec(variant))


@pytest.mark.parametrize("variant", lv.VARIANTS)
def test_fake_declares_variants_and_runs(tmp_path, variant):
    assert {"policy_loss", *lv.VARIANTS} <= set(fake_capabilities().losses)
    _, driver = _driver(tmp_path, spec(variant), fake_capabilities())
    assert driver.run().policy_version == 3


def test_single_island_allowance_admits_variant_on_miles_capabilities():
    names = check_unverified_allowance(["losses:cispo"], islands=1, outer_sync=False)
    caps = miles_capabilities(FP, unverified_mechanisms=names)
    caps.check(layout="lora", placement="colocated", execution_mode="colocated-serial",
               algorithm=spec("cispo"))
    with pytest.raises(AlgorithmSpecError, match="outer sync on"):
        check_unverified_allowance(["losses:cispo"], islands=1, outer_sync=True)


def test_dry_run_reports_expressible_not_opened(monkeypatch):
    monkeypatch.setattr(lv, "FORK_COMMITS", frozenset())  # a pin without the fork commit
    result = af.dry_run(["--dry-run", "--extra",
                         "--policy-loss-variant cispo --eps-clip 0.2 --eps-clip-high 0.28 "
                         "--calculate-per-token-loss"])
    assert result["verdict"] == "rejected"
    assert "losses mechanism 'cispo' not supported" in result["error"]
    allowed = af.dry_run(["--dry-run", "--extra", "--policy-loss-variant sapo",
                          "--rl-allow-unverified-mechanism", "losses:sapo"])
    assert allowed["verdict"] == "accepted"
    # the dry run also reports the launch checks (pin): a real launch is still refused
    assert any("[loss_variants]" in w and "not opened" in w for w in allowed["launch_warnings"])
    assert af.dry_run(["--dry-run"])["launch_warnings"] == []
    assert allowed["miles_argv"][-6:] == ["--policy-loss-variant", "sapo", "--sapo-tau-pos",
                                          "1.0", "--sapo-tau-neg", "1.05"]


# -- launch checks (pin, GMPO CP) ------------------------------------------------------


@pytest.mark.parametrize("variant", lv.VARIANTS)
def test_variant_refused_until_the_pin_has_the_fork_commit(variant, monkeypatch):
    # the real pin (5c1b49eb) carries the fork commit: open
    assert lv.pinned_miles_commit() in lv.FORK_COMMITS
    assert not any("[loss_variants]" in p for p in alg.launch_problems(spec(variant), RUN))
    monkeypatch.setattr(lv, "FORK_COMMITS", frozenset())  # a pin without it: refused
    problems = alg.launch_problems(spec(variant), RUN)
    assert any("[loss_variants]" in p and "Expressible but not opened" in p for p in problems)
    monkeypatch.setattr(lv, "FORK_COMMITS", frozenset({lv.pinned_miles_commit()}))
    assert not any("[loss_variants]" in p for p in alg.launch_problems(spec(variant), RUN))
    assert not any("[loss_variants]" in p for p in alg.launch_problems(AlgorithmSpec(), RUN))


@pytest.mark.parametrize("cp,rejected", [(1, False), (2, True)])
def test_gmpo_context_parallel_refused(cp, rejected, monkeypatch):
    monkeypatch.setattr(lv, "FORK_COMMITS", frozenset({lv.pinned_miles_commit()}))
    problems = alg.launch_problems(spec("gmpo"), {**RUN, "context_parallel_size": cp})
    assert any("context parallel" in p for p in problems) is rejected
    assert not alg.launch_problems(spec("cispo"), {**RUN, "context_parallel_size": 2})


# -- translation and absorption (4.4, yeto part) ----------------------------------------


def test_translation_emits_variant_and_explicit_parameters():
    assert af.algorithm_argv(spec("cispo", eps_clip=0.2, eps_clip_high=0.28)) == [
        "--eps-clip", "0.2", "--eps-clip-high", "0.28", "--calculate-per-token-loss",
        "--policy-loss-variant", "cispo"]
    assert af.algorithm_argv(spec("gmpo", gmpo_log_clip_low=0.3))[-6:] == [
        "--policy-loss-variant", "gmpo", "--gmpo-log-clip-low", "0.3",
        "--gmpo-log-clip-high", "0.4"]


def test_fork_flag_names_match_the_fork_parser():
    assert lv.FORK_FLAGS == {"--policy-loss-variant", "--sapo-tau-pos", "--sapo-tau-neg",
                             "--gmpo-log-clip-low", "--gmpo-log-clip-high"}
    assert lv.FORK_FLAGS <= af.mapped_flags()


def test_absorb_extra_argv_into_the_spec():
    s, remaining, absorbed = af.absorb_extra_argv(
        AlgorithmSpec(), ["--policy-loss-variant", "sapo", "--sapo-tau-neg=1.2", "--foo"])
    assert remaining == ("--foo",)
    assert s == spec("sapo", sapo_tau_neg=1.2)
    assert absorbed == {"--policy-loss-variant": "sapo", "--sapo-tau-neg": "1.2"}
    # translating back gives the same fork flags
    assert af.algorithm_argv(s) == ["--policy-loss-variant", "sapo", "--sapo-tau-pos", "1.0",
                                    "--sapo-tau-neg", "1.2"]


def test_absorbed_parameter_conflict_lists_both_values():
    with pytest.raises(af.AlgorithmFlagConflict, match=r"sapo_tau_pos=1\.2.*sapo_tau_pos=1\.5"):
        af.absorb_extra_argv(spec("sapo", sapo_tau_pos=1.5), ["--sapo-tau-pos", "1.2"])
    with pytest.raises(af.AlgorithmFlagConflict, match="policy_loss_variant"):
        af.absorb_extra_argv(spec("cispo"), ["--policy-loss-variant", "gmpo"])
    with pytest.raises(AlgorithmSpecError, match="loss.sapo_tau_pos"):
        af.absorb_extra_argv(AlgorithmSpec(), ["--policy-loss-variant", "sapo",
                                               "--sapo-tau-pos", "-1"])


@pytest.mark.parametrize("loss", [{}, {"eps_clip": 0.2}, {"eps_clip_high": 0.28}])
def test_cispo_requires_explicit_clip_range(loss):
    s = AlgorithmSpec.from_dict({"schema": alg.ALGORITHM_SPEC_SCHEMA_V2,
                                 "loss": {"policy_loss_variant": "cispo", **loss}})
    text = "; ".join(s.rejections())
    assert "[loss_variant_cispo_clip]" in text and "loss.eps_clip" in text
    ok = spec("cispo")
    assert ok.rejections() == []
    loss_json = json.loads(ok.canonical_json())["loss"]
    assert (loss_json["eps_clip"], loss_json["eps_clip_high"]) == (0.2, 0.28)
    assert af.algorithm_argv(ok)[:4] == ["--eps-clip", "0.2", "--eps-clip-high", "0.28"]


def test_absorbed_stray_parameter_is_rejected():
    s, _, _ = af.absorb_extra_argv(AlgorithmSpec(), ["--gmpo-log-clip-low", "0.2"])
    assert "[loss_variant_params]" in "; ".join(s.rejections())


# -- review round 3: GMPO global num/den, per-token loss ---------------------------------


def test_reference_gmpo_global_clip_excludes_zero_advantage_sequences():
    old = _t(0.0, 0.0)
    m = torch.tensor([1, 1])
    seqs = [(_t(0.5, 0.6), old, _t(1.0, 1.0), m),     # both clipped
            (_t(0.1, 0.5), old, _t(-1.0, -1.0), m),   # A<0: none clipped
            (_t(0.9, 0.9), old, _t(0.0, 0.0), m)]     # A=0: not counted
    assert ref.gmpo_global_clip(seqs) == (2.0, 4.0)
    assert ref.gmpo_global_clip(seqs[:1] + seqs[2:]) == (2.0, 2.0)


def _steps(*pairs):
    return [{"pg_clipfrac": 0.1, "metrics": {"gmpo_clip_num": n, "gmpo_clip_den": d,
                                             "pg_clipfrac": 0.1}} for n, d in pairs]


def test_gmpo_clip_fraction_from_fork_counts():
    assert lv.gmpo_clip_fraction(_steps((2.0, 4.0), (3.0, 3.0))) == pytest.approx(5 / 7)
    # CP duplication / micro-batch averaging scale num and den alike
    assert lv.gmpo_clip_fraction(_steps((4.0, 4.0), (0.5, 0.5))) == 1.0
    assert lv.gmpo_clip_fraction([{"gmpo_clip_num": torch.tensor(1.0),
                                   "gmpo_clip_den": torch.tensor(2.0)}]) == 0.5


@pytest.mark.parametrize("steps", [
    [], None, _steps((0.0, 0.0)), [{"metrics": {"pg_clipfrac": 1.0}}],
    _steps((1.0, 1.0)) + [{"metrics": {}}], _steps((math.nan, 1.0)), _steps((-1.0, 1.0)),
    _steps((2.0, 1.0)), [{"metrics": {"gmpo_clip_num": True, "gmpo_clip_den": 1.0}}],
])
def test_gmpo_clip_fraction_unknown(steps):
    assert lv.gmpo_clip_fraction(steps) is None


def test_gmpo_rule_on_global_fraction():
    s = spec("gmpo")
    batch = _batch(_group(0.5))
    full = lv.gmpo_clip_fraction(_steps((3.0, 3.0), (1.0, 1.0)))
    assert s.expects_gradient(batch, _metrics(None, full)) is False
    partial = lv.gmpo_clip_fraction(_steps((3.0, 3.0), (0.0, 1.0)))
    assert s.expects_gradient(batch, _metrics(None, partial)) is True


def test_gmpo_with_per_token_loss_rejected_before_launch(tmp_path):
    s = spec("gmpo", aggregation="token")
    assert "[loss_variant_gmpo_per_token]" in "; ".join(s.rejections())
    caps = fake_capabilities(loss_aggregations={"default", "token"})
    engine, driver = _driver(tmp_path, s, caps)
    with pytest.raises(CapabilityMismatch, match="loss_variant_gmpo_per_token"):
        driver.run()
    assert engine.calls == []
    for name in ("cispo", "sapo"):
        assert spec(name, aggregation="token").rejections() == []


@pytest.mark.parametrize("aggregation", ["default", "constant"])
def test_cispo_requires_token_aggregation(aggregation):
    loss = {"policy_loss_variant": "cispo", "eps_clip": 0.2, "eps_clip_high": 0.28,
            "aggregation": aggregation}
    if aggregation == "constant":
        loss["reducer"] = alg.PluginRef.from_path(
            "yeto.rl.engine.algorithm.plugin_source_sha256").to_dict()
    s = AlgorithmSpec.from_dict({"schema": alg.ALGORITHM_SPEC_SCHEMA_V2, "loss": loss})
    text = "; ".join(s.rejections())
    assert "[loss_variant_cispo_aggregation]" in text and "loss.aggregation='token'" in text


def test_reference_sapo_is_a_sequence_mean():
    a, b = _t(1.0, 1.0, 1.0), _t(4.0)
    ma, mb = torch.tensor([1, 1, 1]), torch.tensor([1])
    assert ref.sample_mean([a, b], [ma, mb]).item() == pytest.approx(2.5)  # (1 + 4) / 2
    assert ref.token_mean(torch.cat([a, b]), torch.cat([ma, mb])).item() == pytest.approx(7 / 4)


# -- 4.4 translation through the Miles adapter + provenance on the pinned fork ----------


@pytest.mark.parametrize("variant", lv.VARIANTS)
def test_translate_run_config_and_provenance_carry_variant_and_fork_commit(variant):
    from test_rl_miles_adapter_config import make_config

    from yeto.rl import MILES_NEXT_COMMIT
    from yeto.rl.engine.miles_adapter import config as mc
    from yeto.rl.engine.miles_adapter.entry import selection_event

    assert lv.fork_supports_variants(MILES_NEXT_COMMIT)
    s = spec(variant)
    launch = mc.translate_run_config(make_config(), s)
    argv = list(launch.argv)
    i = argv.index("--policy-loss-variant")
    assert argv[i + 1] == variant
    event = selection_event(launch=launch, algorithm=launch.algorithm,
                            miles_commit=MILES_NEXT_COMMIT)
    assert event["miles_commit"] == MILES_NEXT_COMMIT
    recorded = json.loads(event["rl/algorithm_spec"])["loss"]
    assert recorded["policy_loss_variant"] == variant
    for name, value in lv.variant_params(s).items():
        assert recorded[name] == value
