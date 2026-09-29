"""Capability declaration, execution requirements, rejection matrix and the
unverified-mechanism allowance (rl-algorithm-capabilities 3.1-3.4, 5.5 core)."""

from __future__ import annotations

import json

import pytest
import torch

from yeto.rl.elastic_benchmark.capabilities import attestation_from_dict
from yeto.rl.engine import algorithm as alg
from yeto.rl.engine.algorithm import (  # noqa: I001
    AdvantageSpec,
    AlgorithmSpec,
    AlgorithmSpecError,
    CorrectionSpec,
    ExecutionSpec,
    KlSpec,
    LossSpec,
    check_unverified_allowance,
)
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.capabilities import (
    CapabilityMismatch,
    EngineCapabilities,
    ExecutionCapabilities,
)
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.miles_adapter.entry import miles_capabilities

FP = "sha256:" + "e" * 64
NAME = "base_model.model.layer.lora_A.weight"


def _check(caps, spec, **kw):
    caps.check(layout="lora", placement="colocated", execution_mode="colocated-serial",
               algorithm=spec, **kw)


# -- 3.1 ------------------------------------------------------------------------


def test_new_dimensions_roundtrip_and_pr66_reader_unchanged():
    old = EngineCapabilities(
        engine="e", runtime_fingerprint=FP, parameter_layouts=["lora"],
        placements=["colocated"], advantage_estimators=["grpo"],
        dynamic_sampling_filters=[], execution_modes=["colocated-serial"],
        certified_edges=[("A", "B", "rollout-only")], partitioned_driver=True,
    )
    new = EngineCapabilities(
        engine="e", runtime_fingerprint=FP, parameter_layouts=["lora"],
        placements=["colocated"], advantage_estimators=["grpo", "gspo"],
        dynamic_sampling_filters=[], execution_modes=["colocated-serial"],
        certified_edges=[("A", "B", "rollout-only")], partitioned_driver=True,
        losses=["policy_loss"], loss_aggregations=["default", "token"],
        kl_placements=["none", "loss"], corrections=["none", "tis"],
        reward_postprocessors=["custom_reward_postprocess"], features=["clip_higher"],
        execution=ExecutionCapabilities(critic=False, max_policy_staleness=0,
                                        rollout_logprobs=True),
        unverified_mechanisms=["dual_clip"],
    )
    assert EngineCapabilities.from_json(new.to_json()) == new
    payload = json.loads(new.to_json())
    assert payload["execution"] == {"critic": False, "max_policy_staleness": 0,
                                    "rollout_logprobs": True}
    assert payload["corrections"] == ["none", "tis"]
    a_old = attestation_from_dict(json.loads(old.to_json()))
    a_new = attestation_from_dict(payload)
    for field in ("runtime_fingerprint", "execution_modes", "certified_edges",
                  "optimized_paths", "auto_controller", "partitioned_driver"):
        assert getattr(a_new, field) == getattr(a_old, field), field


def test_old_declaration_reads_as_r0_mechanisms():
    payload = json.loads(fake_capabilities().to_json())
    for key in ("losses", "loss_aggregations", "kl_placements", "corrections",
                "reward_postprocessors", "features", "execution", "unverified_mechanisms"):
        payload.pop(key)
    caps = EngineCapabilities.from_attestation_dict(payload)
    assert caps.losses == {"policy_loss"} and caps.kl_placements == {"none", "reward"}
    assert caps.execution == ExecutionCapabilities()


# -- 3.2 ------------------------------------------------------------------------

# One spec per P0 mechanism beyond R0 (each produces that mechanism; some
# produce companions too). Tests pick the ones an engine has NOT declared, so
# a later declaration (a follow-up change's G1) never turns them into no-ops.
_REF = alg.PluginRef.from_path("yeto.rl.engine.algorithm.plugin_source_sha256")
CANDIDATES = {
    "advantage_estimators:gspo": dict(advantage=dict(estimator="gspo"),
                                      loss=dict(eps_clip=3e-4, eps_clip_high=4e-4)),
    "advantage_estimators:reinforce_plus_plus": dict(
        advantage=dict(estimator="reinforce_plus_plus", whiten=True)),
    "advantage_estimators:reinforce_plus_plus_baseline": dict(
        advantage=dict(estimator="reinforce_plus_plus_baseline", whiten=True)),
    "corrections:tis": dict(correction=dict(method="tis", tis_clip=2.0, tis_clip_low=0.0)),
    "corrections:opsm": dict(correction=dict(method="opsm", opsm_delta=1e-4)),
    "corrections:custom": dict(correction=dict(method="custom", function=_REF,
                                               tis_clip=5.0, tis_clip_low=0.5)),
    "features:clip_higher": dict(loss=dict(eps_clip_high=0.28)),
    "features:dual_clip": dict(loss=dict(eps_clip_c=3.0)),
    "features:eps_clip": dict(loss=dict(eps_clip=0.25)),
    "features:custom_pg_loss_reducer": dict(loss=dict(reducer=_REF)),
    "features:entropy_bonus": dict(entropy_coef=0.001),
    "features:kl_unbiased": dict(kl=dict(placement="loss", coef=0.01, estimator="k3",
                                         unbiased=True)),
    "features:mismatch_metrics": dict(correction=dict(method="tis", tis_clip=2.0,
                                                      tis_clip_low=0.0, mismatch_metrics=True)),
    "features:no_grpo_std_normalization": dict(advantage=dict(std_normalization=False)),
    "features:no_rewards_normalization": dict(advantage=dict(rewards_normalization=False)),
    "features:over_sampling": dict(sampling=dict(filter=alg.BOUNDED_NONZERO_STD_FILTER,
                                                 over_sampling_batch_size=8)),
    "features:overlong_filter": dict(sampling=dict(overlong_filter=True)),
    "features:plugins": dict(plugins=(_REF,)),
    "features:rollout_logprobs_as_old": dict(correction=dict(use_rollout_logprobs=True)),
    "features:whiten_advantages": dict(advantage=dict(whiten=True)),
    "kl_placements:loss": dict(kl=dict(placement="loss", coef=0.01, estimator="k3")),
    "loss_aggregations:token": dict(loss=dict(aggregation="token")),
    "loss_aggregations:constant": dict(loss=dict(aggregation="constant", reducer=_REF)),
    "losses:custom_loss": dict(loss=dict(variant="custom_loss", custom_loss=_REF)),
    "reward_postprocessors:custom_reward_postprocess": dict(
        advantage=dict(reward_postprocess=_REF)),
}
R0_MECHANISMS_P0 = {
    "advantage_estimators:grpo", "losses:policy_loss", "loss_aggregations:default",
    "kl_placements:none", "kl_placements:reward", "corrections:none",
    f"dynamic_sampling_filters:{alg.BOUNDED_NONZERO_STD_FILTER}",
    f"dynamic_sampling_filters:{alg.STOCK_NONZERO_STD_FILTER}",
}
# P0 built-ins without a candidate, with the reason.
EXEMPT = {"advantage_estimators:ppo": "critic: refused by the rejection matrix"}


def test_candidates_cover_the_p0_mechanism_registry():
    registered = alg.mechanism_names()
    assert set(CANDIDATES) <= registered
    p0_builtin = set(CANDIDATES) | set(EXEMPT) | R0_MECHANISMS_P0
    # every P0 built-in beyond R0 has a candidate (extension mechanisms may add more)
    builtin = {
        f"{m.dimension}:{m.name}" for m in alg.registered_mechanisms()
        if m.detect.__module__ == alg.__name__  # registered by P0 itself
    }
    missing = sorted(builtin - p0_builtin)
    assert not missing, f"P0 mechanisms without a candidate: {missing}"
    assert not (set(CANDIDATES) - builtin), "a candidate names a non-P0 mechanism"
    for mechanism, kwargs in CANDIDATES.items():
        dimension, name = mechanism.split(":")
        assert (dimension, name) in _combine(kwargs).required_mechanisms(), mechanism


def undeclared(caps):
    """[(mechanism, kwargs)] of candidates ``caps`` does not declare (non-empty)."""

    declared = caps.declared_mechanisms()
    found = [(m, kw) for m, kw in CANDIDATES.items() if m not in declared]
    assert found, "every candidate is declared; extend CANDIDATES"
    return found


def _combine(*kwargs_list):
    merged = {}
    for kw in kwargs_list:
        for group, value in kw.items():
            if isinstance(value, dict):
                merged.setdefault(group, {}).update(value)
            else:
                merged[group] = value
    return AlgorithmSpec(**merged)


def _touches(kw):
    return {(g, f) for g, v in kw.items() for f in (v if isinstance(v, dict) else (None,))}


def _disjoint_pair(items):
    for i, (m1, k1) in enumerate(items):
        for m2, k2 in items[i + 1:]:
            if _touches(k1) & _touches(k2) or m1.split(":")[0] == m2.split(":")[0] == "corrections":
                continue
            try:
                spec = _combine(k1, k2)
            except alg.AlgorithmSpecError:
                continue
            if not spec.rejections():
                return (m1, k1), (m2, k2)
    pytest.fail("fewer than two combinable undeclared candidates; extend CANDIDATES")


def test_expressible_but_not_enabled_is_rejected_with_options():
    caps = miles_capabilities(FP)
    (m1, k1), (m2, k2) = _disjoint_pair(undeclared(caps))
    with pytest.raises(CapabilityMismatch) as info:
        _check(caps, _combine(k1, k2))
    text = str(info.value)
    # every problem at once, each with the supported options
    labels = {"advantage_estimators": "advantage estimator",
              "dynamic_sampling_filters": "dynamic sampling filter"}
    for mechanism in (m1, m2):
        dimension, name = mechanism.split(":")
        supported = sorted(caps.mechanisms(dimension))
        label = labels.get(dimension, f"{dimension} mechanism")
        assert f"{label} {name!r} not supported (supported: {supported}" in text
    assert "expressible but not enabled" in text


def test_miles_accepts_each_declared_mechanism_and_rejects_overlong_filter():
    caps = miles_capabilities(FP)
    declared = caps.declared_mechanisms()
    covered = []
    for mechanism in sorted(declared):
        kwargs = CANDIDATES.get(mechanism)
        if kwargs is None:
            continue  # R0 (default spec) or an extension mechanism not registered here
        spec = _combine(kwargs)
        needs = {f"{d}:{n}" for d, n in spec.required_mechanisms()}
        if needs <= declared:
            _check(caps, spec)  # accepted
            covered.append(mechanism)
    _check(caps, AlgorithmSpec())  # the R0 declarations
    assert covered, "no declared non-R0 mechanism was exercised"
    assert {"corrections:tis", "corrections:opsm"} <= set(covered)
    assert "features:overlong_filter" not in declared
    with pytest.raises(CapabilityMismatch, match="'overlong_filter' not supported"):
        _check(caps, _combine(CANDIDATES["features:overlong_filter"]))


def test_critic_rejected_pointing_to_legacy():
    caps = fake_capabilities(advantage_estimators={"grpo", "ppo"})
    spec = AlgorithmSpec(advantage=AdvantageSpec(estimator="ppo"),
                         execution=ExecutionSpec(needs_critic=True))
    with pytest.raises(CapabilityMismatch, match="critic.*--rl-engine legacy"):
        _check(caps, spec)
    # without declaring the critic need, the rejection matrix still points to legacy
    with pytest.raises(CapabilityMismatch, match="needs a critic.*--rl-engine legacy"):
        _check(caps, AlgorithmSpec(advantage=AdvantageSpec(estimator="ppo")))


def test_staleness_requirement_not_met():
    caps = fake_capabilities()
    with pytest.raises(CapabilityMismatch, match="policy age 1.*max_policy_staleness=0"):
        _check(caps, AlgorithmSpec(), max_policy_age=1)
    stale_mode = fake_capabilities(execution=ExecutionCapabilities(max_policy_staleness=2))
    with pytest.raises(CapabilityMismatch, match="policy age 2"):
        _check(stale_mode, AlgorithmSpec())
    with pytest.raises(CapabilityMismatch, match="max_policy_staleness=1.*A6"):
        _check(caps, AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1)))
    _check(caps, AlgorithmSpec(), max_policy_age=0)


def test_rollout_logprobs_requirement():
    spec = AlgorithmSpec(execution=ExecutionSpec(needs_rollout_logprobs=True))
    with pytest.raises(CapabilityMismatch, match="rollout logprobs"):
        _check(fake_capabilities(execution=ExecutionCapabilities()), spec)
    _check(fake_capabilities(), spec)


# -- 3.3 rejection matrix, before any GPU process (fake composition root) ----------


def _driver(tmp_path, spec, caps):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)})
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, capabilities=caps, algorithm=spec,
        sync=LocalOnlySync(1), events=EventTape(tmp_path / "events.jsonl", 0),
    )
    return engine, driver


ALL_DECLARED = dict(
    advantage_estimators=set(alg.ADVANTAGE_ESTIMATORS),
    losses=set(alg.LOSS_VARIANTS), loss_aggregations=set(alg.LOSS_AGGREGATIONS),
    kl_placements=set(alg.KL_PLACEMENTS), corrections=set(alg.CORRECTION_METHODS),
    features={"eps_clip", "clip_higher", "dual_clip", "rollout_logprobs_as_old",
              "mismatch_metrics"},
)


@pytest.fixture
def binary_mechanism():
    alg.register_mechanism("features", "test_binary_only",
                           lambda s: s.loss.eps_clip_c == 7.0, requires_binary_reward=True)
    yield
    alg.unregister(mechanism=("features", "test_binary_only"))


MATRIX = [
    ("tis_with_rollout_logprobs",
     AlgorithmSpec(correction=CorrectionSpec(method="tis", tis_clip=2, tis_clip_low=0,
                                             use_rollout_logprobs=True)),
     "mutually exclusive"),
    ("reward_kl_with_loss_kl", None, "reward KL.*loss KL"),
    ("sequence_ratio_without_clip",
     AlgorithmSpec(advantage=AdvantageSpec(estimator="gspo")), "loss.eps_clip and"),
    ("binary_reward_required",
     AlgorithmSpec(loss=LossSpec(eps_clip_c=7.0)), "require a binary"),
    ("reward_kl_ignored", AlgorithmSpec(kl=KlSpec(placement="reward", coef=0.1)),
     "placement='loss'"),
]


@pytest.mark.parametrize("name, spec, message", MATRIX, ids=[m[0] for m in MATRIX])
def test_rejection_matrix_before_any_engine_verb(tmp_path, binary_mechanism, name, spec, message):
    if spec is None:
        # One placement per spec: reward+loss KL can only arrive via extra argv.
        from yeto.rl.engine.miles_adapter.algorithm_flags import absorb_extra_argv

        with pytest.raises(AlgorithmSpecError, match=message):
            absorb_extra_argv(AlgorithmSpec(), ["--kl-coef", "0.1", "--use-kl-loss",
                                                "--kl-loss-coef", "0.1",
                                                "--kl-loss-type", "k1"])
        return
    caps = fake_capabilities(**ALL_DECLARED)
    engine, driver = _driver(tmp_path, spec, caps)
    with pytest.raises(CapabilityMismatch, match=message):
        driver.run()
    assert engine.calls == []  # nothing started: rejected in the handshake


def test_binary_reward_declared_passes(binary_mechanism):
    spec = AlgorithmSpec(loss=LossSpec(eps_clip_c=7.0),
                         advantage=AdvantageSpec(reward_binary=True))
    assert not [p for p in spec.rejections() if "binary" in p]


# -- 3.4 declarations ----------------------------------------------------------------


def test_miles_and_fake_declarations():
    # rl-algo-mismatch-correction 7.3 (G1 passed): the adapter declares tis and
    # OPSM (trainer pi_old); the fake declares every correction for CPU tests.
    expected = {
        # rl-algo-seq-and-adv 7.5 G1: maxrl/mapo (gdpo held back)
        "miles": ({"none", "tis", "opsm", "opsm_trainer"}, {"maxrl", "mapo"}),
        "fake": ({"none", "tis", "opsm", "custom", "mismatch_observe", "icepop",
                  "opsm_trainer", "opsm_rollout", "mis", "mis_mask"},
                 {"mismatch_metrics", "rollout_logprobs_as_old"}),
    }
    for kind, caps in (("miles", miles_capabilities(FP)), ("fake", fake_capabilities())):
        corrections, features = expected[kind]
        assert caps.advantage_estimators == {"grpo"}
        assert caps.losses == {"policy_loss"} and caps.loss_aggregations == {"default"}
        assert caps.kl_placements == {"none", "reward"} and caps.corrections == corrections
        assert caps.reward_postprocessors == frozenset() and caps.features == features
        assert caps.execution == ExecutionCapabilities(
            critic=False, max_policy_staleness=0, rollout_logprobs=True)
        assert caps.unverified_mechanisms == frozenset()


def test_fake_root_default_grpo_starts_other_mechanisms_rejected(tmp_path):
    engine, driver = _driver(tmp_path, AlgorithmSpec(), fake_capabilities())
    assert driver.run().policy_version == 1
    for _mechanism, kwargs in undeclared(fake_capabilities()):
        spec = _combine(kwargs)
        engine, driver = _driver(tmp_path, spec, fake_capabilities())
        with pytest.raises(CapabilityMismatch, match="not supported"):
            driver.run()
        assert engine.calls == []


# -- 5.5 allowance (capability side) -------------------------------------------------


def test_unverified_allowance_single_island_without_outer_sync(tmp_path):
    mechanism, kwargs = undeclared(fake_capabilities())[0]
    spec = _combine(kwargs)
    declared = fake_capabilities().declared_mechanisms()
    needed = sorted({f"{d}:{n}" for d, n in spec.required_mechanisms()} - declared)
    assert mechanism in needed
    names = check_unverified_allowance(needed, islands=1, outer_sync=False)
    caps = fake_capabilities().with_unverified(names)
    engine, driver = _driver(tmp_path, spec, caps)  # LocalOnlySync: no outer sync
    assert driver.run().policy_version == 1
    assert json.loads(caps.to_json())["unverified_mechanisms"] == needed
    # the allowance does not enter the algorithm hash
    assert spec.sha256() == _combine(kwargs).sha256()


def test_unverified_allowance_refused_with_outer_sync_multi_island_unknown():
    with pytest.raises(AlgorithmSpecError, match="without outer sync.*2 island"):
        check_unverified_allowance(["features:clip_higher"], islands=2, outer_sync=False)
    with pytest.raises(AlgorithmSpecError, match="outer sync on"):
        check_unverified_allowance(["features:clip_higher"], islands=1, outer_sync=True)
    with pytest.raises(AlgorithmSpecError, match="unknown mechanism"):
        check_unverified_allowance(["clip_higher"], islands=1, outer_sync=False)  # bare name
    with pytest.raises(CapabilityMismatch, match="unknown mechanism"):
        fake_capabilities().with_unverified(["features:no_such_mechanism"])


def test_unverified_allowance_is_dimension_qualified(tmp_path):
    # "none" exists in two dimensions; allowing kl_placements:none must not
    # admit corrections:none-like names elsewhere.
    caps = fake_capabilities(corrections=set()).with_unverified(["kl_placements:none"])
    with pytest.raises(CapabilityMismatch, match="corrections mechanism 'none'"):
        _check(caps, AlgorithmSpec())


def test_unverified_allowance_does_not_bypass_rejection_matrix(tmp_path):
    spec = AlgorithmSpec(advantage=AdvantageSpec(estimator="gspo"))  # no explicit clip
    caps = fake_capabilities().with_unverified(["advantage_estimators:gspo"])
    engine, driver = _driver(tmp_path, spec, caps)
    with pytest.raises(CapabilityMismatch, match="sequence-level ratio") as info:
        driver.run()
    assert "'gspo' not supported" not in str(info.value)  # only the matrix problem
    assert engine.calls == []


def test_unverified_allowance_exempts_only_named(tmp_path):
    (m1, k1), (m2, k2) = _disjoint_pair(undeclared(fake_capabilities()))
    caps = fake_capabilities().with_unverified([m1])
    engine, driver = _driver(tmp_path, _combine(k1, k2), caps)
    with pytest.raises(CapabilityMismatch, match=f"{m2.split(':')[1]!r} not supported") as info:
        driver.run()
    assert f"{m1.split(':')[1]!r} not supported" not in str(info.value)


# -- 1a-shared: OPSM combinations, named custom functions, always-emit fields -----


def test_opsm_combines_with_tis_and_translates_both():
    from yeto.rl.engine.miles_adapter.algorithm_flags import algorithm_argv

    spec = AlgorithmSpec(correction=CorrectionSpec(method="tis", tis_clip=2, tis_clip_low=0,
                                                   opsm_delta=1e-4))
    assert ("corrections", "opsm") in spec.required_mechanisms()
    assert ("corrections", "tis") in spec.required_mechanisms()
    argv = algorithm_argv(spec)
    assert "--use-tis" in argv and argv[argv.index("--use-opsm") + 2] == "0.0001"
    _check(miles_capabilities(FP), spec)
    with pytest.raises(AlgorithmSpecError, match="opsm_delta requires"):
        CorrectionSpec(opsm_delta=1e-4)


def test_named_custom_function_only_exempt_when_its_mechanism_detects():
    from yeto.rl.engine.algorithm import PluginRef

    ref = PluginRef.from_path("yeto.rl.engine.algorithm.plugin_source_sha256")
    spec = AlgorithmSpec(correction=CorrectionSpec(method="custom", function=ref,
                                                   tis_clip=5, tis_clip_low=0.5))
    assert ("corrections", "custom") in spec.required_mechanisms()
    alg.register_named_correction_function(ref.path)
    try:
        # named but no mechanism claims it: still generic custom (no escape)
        assert ("corrections", "custom") in spec.required_mechanisms()
        alg.register_mechanism("corrections", "t_named",
                               lambda s: s.correction.function is not None
                               and s.correction.function.path == ref.path)
        try:
            required = spec.required_mechanisms()
            assert ("corrections", "t_named") in required
            assert ("corrections", "custom") not in required
            # the adapter declaring only generic 'custom' does not admit it
            with pytest.raises(CapabilityMismatch, match="'t_named' not supported"):
                _check(fake_capabilities(corrections={"none", "custom"}), spec)
            # a different function (other path) still needs 'custom'
            other = PluginRef.from_path("yeto.rl.engine.algorithm.load_extensions")
            spec2 = AlgorithmSpec(correction=CorrectionSpec(method="custom", function=other,
                                                            tis_clip=5, tis_clip_low=0.5))
            assert ("corrections", "custom") in spec2.required_mechanisms()
        finally:
            alg.unregister(mechanism=("corrections", "t_named"))
    finally:
        alg.NAMED_CORRECTION_FUNCTIONS.discard(ref.path)
    with pytest.raises(ValueError, match="must be in"):
        alg.register_named_correction_function("examples.x.fn")


def test_always_emit_field_only_when_its_mechanism_applies():
    from yeto.rl.engine.algorithm import CorrectionSpec as C

    alg.register_field("correction", "t_src", default="trainer",
                       parse=lambda p, v: v, always_emit=lambda g: g.opsm_delta is not None)
    try:
        assert AlgorithmSpec().sha256() == AlgorithmSpec().replace().sha256()
        assert "t_src" not in AlgorithmSpec().canonical_json()
        tis = AlgorithmSpec(correction=C(method="tis", tis_clip=2, tis_clip_low=0))
        assert "t_src" not in tis.canonical_json()
        opsm = AlgorithmSpec(correction=C(method="opsm", opsm_delta=1e-4))
        assert '"t_src":"trainer"' in opsm.canonical_json()
    finally:
        alg.unregister(field=("correction", "t_src"))


def test_reward_dispatcher_rejection_explains_the_pending_declaration():
    caps = miles_capabilities(FP)
    with pytest.raises(CapabilityMismatch, match="maxrl/mapo is not declared yet"):
        _check(caps, _combine(CANDIDATES["reward_postprocessors:custom_reward_postprocess"]))
