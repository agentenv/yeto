"""AlgorithmSpec v2 (change rl-algorithm-capabilities, tasks 1.1-1.5)."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from yeto.rl.engine import algorithm as alg
from yeto.rl.engine.algorithm import (
    ALGORITHM_SPEC_SCHEMA,
    ALGORITHM_SPEC_SCHEMA_V2,
    BOUNDED_NONZERO_STD_FILTER,
    AdvantageSpec,
    AlgorithmSpec,
    AlgorithmSpecError,
    CorrectionSpec,
    ExecutionSpec,
    KlSpec,
    LossSpec,
    PluginIdentityError,
    PluginRef,
    SamplingSpec,
)

BOUNDED = AlgorithmSpec(
    dynamic_sampling_filter=BOUNDED_NONZERO_STD_FILTER,
    dynamic_sampling_max_replacements=4,
)
# R0 golden (tests/test_rl_engine_algorithm.py pins the canonical JSON).
R0_BOUNDED_JSON = (
    '{"advantage_estimator":"grpo","dynamic_sampling_filter":'
    '"yeto.rl.filters.bounded_nonzero_reward_std",'
    '"dynamic_sampling_max_replacements":4,"kl_coef":null,'
    '"loss":"policy_loss","schema":"yeto-rl-algorithm-spec-v1"}'
)
R0_DEFAULT_JSON = (
    '{"advantage_estimator":"grpo","dynamic_sampling_filter":null,'
    '"dynamic_sampling_max_replacements":null,"kl_coef":null,'
    '"loss":"policy_loss","schema":"yeto-rl-algorithm-spec-v1"}'
)
FILE_REF = PluginRef.from_path("yeto.rl.engine.algorithm.plugin_source_sha256")


# -- 1.1 field-level validation ------------------------------------------------


@pytest.mark.parametrize(
    "build, field",
    [
        (lambda: LossSpec(eps_clip_high=-0.1), "loss.eps_clip_high"),
        (lambda: LossSpec(eps_clip=float("nan")), "loss.eps_clip"),
        (lambda: LossSpec(eps_clip="0.2"), "loss.eps_clip"),
        (lambda: LossSpec(eps_clip_c=1.0), "loss.eps_clip_c"),
        (lambda: LossSpec(aggregation="sum"), "loss.aggregation"),
        (lambda: LossSpec(variant="cispo"), "loss.variant"),
        (lambda: KlSpec(placement="value"), "kl.placement"),
        (lambda: KlSpec(placement="loss", coef=float("inf"), estimator="k1"), "kl.coef"),
        (lambda: KlSpec(placement="loss", coef=0.1, estimator="k9"), "kl.estimator"),
        (lambda: KlSpec(placement="loss", coef=0.1), "kl.estimator"),
        (lambda: KlSpec(placement="reward"), "kl.coef"),
        (lambda: KlSpec(unbiased="yes"), "kl.unbiased"),
        (lambda: AdvantageSpec(estimator="dapo"), "advantage.estimator"),
        (lambda: AdvantageSpec(whiten=1), "advantage.whiten"),
        (lambda: CorrectionSpec(method="tis"), "correction.tis_clip"),
        (lambda: CorrectionSpec(method="opsm"), "correction.opsm_delta"),
        (lambda: CorrectionSpec(method="bogus"), "correction.method"),
        (lambda: CorrectionSpec(method="tis", tis_clip=-1, tis_clip_low=0), "correction.tis_clip"),
        (lambda: SamplingSpec(max_replacements=-1, filter=BOUNDED_NONZERO_STD_FILTER),
         "sampling.max_replacements"),
        (lambda: SamplingSpec(over_sampling_batch_size=0), "sampling.over_sampling_batch_size"),
        (lambda: SamplingSpec(filter="x.y"), "sampling.filter"),
        (lambda: ExecutionSpec(max_policy_staleness=-1), "execution.max_policy_staleness"),
        (lambda: ExecutionSpec(needs_critic="no"), "execution.needs_critic"),
        (lambda: AlgorithmSpec(entropy_coef=float("nan")), "entropy_coef"),
        (lambda: AlgorithmSpec.from_dict({"schema": ALGORITHM_SPEC_SCHEMA_V2,
                                          "loss": {"eps_clip_hi": 0.3}}), "loss.eps_clip_hi"),
    ],
)
def test_invalid_field_names_the_field(build, field):
    with pytest.raises(AlgorithmSpecError, match=field.replace(".", r"\.")):
        build()


def test_groups_are_frozen():
    spec = AlgorithmSpec()
    with pytest.raises(Exception):
        spec.loss.eps_clip = 0.3  # type: ignore[misc]
    with pytest.raises(Exception):
        spec.entropy_coef = 1.0  # type: ignore[misc]


# -- 1.2 v1/v2 from_dict + v1 mapping ------------------------------------------


def test_v1_json_roundtrip_is_identical():
    for spec, text in ((BOUNDED, R0_BOUNDED_JSON), (AlgorithmSpec(), R0_DEFAULT_JSON)):
        loaded = AlgorithmSpec.from_dict(json.loads(text))
        assert loaded == spec
        assert loaded.canonical_json() == text
    kl = AlgorithmSpec.from_dict({"advantage_estimator": "grpo", "kl_coef": 0.0})
    assert kl.kl == KlSpec(placement="reward", coef=0.0)
    assert kl.kl_coef == 0.0
    assert AlgorithmSpec.from_dict(json.loads(kl.canonical_json())) == kl


def test_v1_fields_map_to_groups():
    assert BOUNDED.sampling == SamplingSpec(filter=BOUNDED_NONZERO_STD_FILTER, max_replacements=4)
    assert BOUNDED.advantage.estimator == "grpo" and BOUNDED.loss.variant == "policy_loss"
    assert AlgorithmSpec().kl == KlSpec()  # kl_coef None -> placement none
    assert AlgorithmSpec(kl_coef=0.5).kl == KlSpec(placement="reward", coef=0.5)


def test_v2_roundtrip():
    spec = AlgorithmSpec(
        loss=LossSpec(eps_clip=0.2, eps_clip_high=0.28, aggregation="token"),
        kl=KlSpec(placement="loss", coef=0.01, estimator="k3"),
        advantage=AdvantageSpec(reward_postprocess=FILE_REF),
        entropy_coef=0.001,
    )
    payload = json.loads(spec.canonical_json())
    assert payload["schema"] == ALGORITHM_SPEC_SCHEMA_V2
    assert AlgorithmSpec.from_dict(payload) == spec
    assert AlgorithmSpec.from_dict(payload).sha256() == spec.sha256()


# -- 1.3 canonicalization -------------------------------------------------------


def test_r0_golden_hashes_unchanged():
    assert BOUNDED.canonical_json() == R0_BOUNDED_JSON
    assert BOUNDED.sha256() == hashlib.sha256(R0_BOUNDED_JSON.encode()).hexdigest()
    assert AlgorithmSpec().canonical_json() == R0_DEFAULT_JSON
    # Built from v2 groups, a v1-expressible spec still canonicalizes as v1.
    v2_built = AlgorithmSpec(sampling=SamplingSpec(filter=BOUNDED_NONZERO_STD_FILTER,
                                                   max_replacements=4))
    assert v2_built.canonical_json() == R0_BOUNDED_JSON
    assert AlgorithmSpec.from_dict(BOUNDED.structured_dict()) == BOUNDED


@pytest.mark.parametrize(
    "change",
    [
        dict(loss=LossSpec(eps_clip_high=0.28)),
        dict(loss=LossSpec(eps_clip=0.2)),
        dict(loss=LossSpec(aggregation="token")),
        dict(advantage=AdvantageSpec(std_normalization=False)),
        dict(advantage=AdvantageSpec(estimator="gspo")),
        dict(kl=KlSpec(placement="loss", coef=0.01, estimator="k1")),
        dict(correction=CorrectionSpec(method="opsm", opsm_delta=1e-4)),
        dict(sampling=SamplingSpec(over_sampling_batch_size=8)),
        dict(execution=ExecutionSpec(needs_rollout_logprobs=True)),
        dict(entropy_coef=0.01),
        dict(plugins=(FILE_REF,)),
    ],
)
def test_new_field_switches_to_v2_and_changes_hash(change):
    spec = AlgorithmSpec(**change)
    assert json.loads(spec.canonical_json())["schema"] == ALGORITHM_SPEC_SCHEMA_V2
    assert spec.sha256() != AlgorithmSpec().sha256()
    assert spec.sha256() != BOUNDED.sha256()


def test_int_float_and_order_equivalence():
    a = AlgorithmSpec.from_dict({"schema": ALGORITHM_SPEC_SCHEMA_V2,
                                 "loss": {"eps_clip": 0.2, "eps_clip_high": 0.28},
                                 "entropy_coef": 0})
    b = AlgorithmSpec.from_dict({"entropy_coef": 0.0,
                                 "loss": {"eps_clip_high": 0.28, "eps_clip": 0.2},
                                 "schema": ALGORITHM_SPEC_SCHEMA_V2})
    assert a.sha256() == b.sha256()
    c = AlgorithmSpec(loss=LossSpec(eps_clip_c=3))
    d = AlgorithmSpec(loss=LossSpec(eps_clip_c=3.0))
    assert c.canonical_json() == d.canonical_json()
    assert AlgorithmSpec(kl_coef=0).sha256() == AlgorithmSpec(kl_coef=0.0).sha256()
    # enums are lower-cased; plugins sorted by path
    assert AlgorithmSpec(kl=KlSpec(placement="LOSS", coef=1, estimator="K3")).kl.placement == "loss"
    other = PluginRef.from_path("yeto.rl.engine.algorithm.load_extensions")
    assert (AlgorithmSpec(plugins=(FILE_REF, other)).sha256()
            == AlgorithmSpec(plugins=(other, FILE_REF)).sha256())


def test_extension_fields_do_not_change_existing_hashes():
    spec = AlgorithmSpec(loss=LossSpec(eps_clip_high=0.28))
    before = spec.sha256()
    alg.register_field("advantage", "gamma_test", default=1.0,
                       parse=lambda path, v: alg._number(path, v, optional=False))
    try:
        assert spec.sha256() == before
        assert AlgorithmSpec().canonical_json() == R0_DEFAULT_JSON
        adv = AdvantageSpec().with_ext(gamma_test=0.9)
        assert adv.gamma_test == 0.9 and AdvantageSpec().gamma_test == 1.0
        changed = AlgorithmSpec(advantage=adv)
        assert json.loads(changed.canonical_json())["advantage"]["gamma_test"] == 0.9
        assert AlgorithmSpec.from_dict(json.loads(changed.canonical_json())) == changed
        with pytest.raises(AlgorithmSpecError, match=r"advantage\.gamma_test"):
            AdvantageSpec().with_ext(gamma_test="x")
        # default value normalizes away (still v1)
        assert AlgorithmSpec(advantage=AdvantageSpec().with_ext(gamma_test=1)).canonical_json() \
            == R0_DEFAULT_JSON
    finally:
        alg.unregister(field=("advantage", "gamma_test"))


# -- 1.4 plugin identity ---------------------------------------------------------


def test_plugin_hash_matches():
    FILE_REF.verify()
    with open(alg.__file__, "rb") as handle:
        assert FILE_REF.sha256 == hashlib.sha256(handle.read()).hexdigest()


def test_plugin_hash_mismatch_reports_both():
    bad = PluginRef(FILE_REF.path, "0" * 64)
    with pytest.raises(PluginIdentityError, match=f"spec {'0' * 64}, runtime {FILE_REF.sha256}"):
        bad.verify()
    with pytest.raises(PluginIdentityError):
        AlgorithmSpec(plugins=(bad,)).verify_plugins()


def test_plugin_import_failure():
    with pytest.raises(PluginIdentityError, match="not importable"):
        PluginRef("yeto.no_such_module.fn", "a" * 64).verify()
    with pytest.raises(PluginIdentityError, match="not a callable"):
        PluginRef("yeto.rl.engine.algorithm.no_such_fn", FILE_REF.sha256).verify()


def test_plugin_namespace_not_allowed():
    with pytest.raises(AlgorithmSpecError, match="allowed namespaces"):
        PluginRef("os.path.join", "a" * 64)
    with pytest.raises(AlgorithmSpecError, match="allowed namespaces"):
        PluginRef.from_path("json.dumps")
    PluginRef("miles.backends.training_utils.loss_hub.corrections.icepop_function", "b" * 64)


def test_plugin_identity_enters_hash():
    a = AlgorithmSpec(plugins=(FILE_REF,))
    b = AlgorithmSpec(plugins=(PluginRef(FILE_REF.path, "c" * 64),))
    assert a.sha256() != b.sha256()


# -- 1.5 derived methods ------------------------------------------------------------


def _groups(*stds):
    return SimpleNamespace(groups=[SimpleNamespace(reward_std=s) for s in stds])


def _r0_driver_rule(batch):
    # yeto/rl/engine/driver.py (R0) _check_gradient: advantages_nonzero
    return any(g.reward_std > 0 for g in batch.groups)


@pytest.mark.parametrize("stds", [(0.0,), (0.0, 0.0), (0.5,), (0.0, 0.1), (0.3, 0.2)])
@pytest.mark.parametrize("masked", [None, 0.0, 0.5, 1.0])
def test_default_grpo_expects_gradient_equals_r0(stds, masked):
    batch = _groups(*stds)
    metrics = SimpleNamespace(masked_fraction=masked)
    for spec in (AlgorithmSpec(), BOUNDED):
        assert spec.expects_gradient(batch, metrics) == _r0_driver_rule(batch)
        assert spec.expects_gradient(batch) == _r0_driver_rule(batch)


def test_masking_mechanism_lifts_expectation_only_when_fully_masked():
    alg.register_mechanism("features", "test_mask", lambda s: s.loss.eps_clip == 0.123,
                           masks_tokens=True)
    try:
        spec = AlgorithmSpec(loss=LossSpec(eps_clip=0.123))
        batch = _groups(0.5)
        assert spec.expects_gradient(batch, SimpleNamespace(masked_fraction=1.0)) is False
        assert spec.expects_gradient(batch, SimpleNamespace(masked_fraction=0.99)) is True
        assert spec.expects_gradient(batch, SimpleNamespace(masked_fraction=None)) is True
        assert AlgorithmSpec().expects_gradient(batch, SimpleNamespace(masked_fraction=1.0))
    finally:
        alg.unregister(mechanism=("features", "test_mask"))


def test_required_mechanisms_default_grpo():
    assert AlgorithmSpec().required_mechanisms() == {
        ("advantage_estimators", "grpo"), ("losses", "policy_loss"),
        ("loss_aggregations", "default"), ("kl_placements", "none"), ("corrections", "none"),
    }
    assert ("dynamic_sampling_filters", BOUNDED_NONZERO_STD_FILTER) in BOUNDED.required_mechanisms()
    assert ("features", "clip_higher") in AlgorithmSpec(
        loss=LossSpec(eps_clip_high=0.3)).required_mechanisms()
    assert ("kl_placements", "reward") in AlgorithmSpec(kl_coef=0.0).required_mechanisms()


def test_v1_constructor_keeps_r0_validation():
    with pytest.raises(AlgorithmSpecError, match="grpo"):
        AlgorithmSpec(advantage_estimator="ppo")
    with pytest.raises(AlgorithmSpecError, match="unknown algorithm spec fields"):
        AlgorithmSpec.from_dict({"advantage_estimator": "grpo", "eps_clip": 0.2})


# -- shared extension points (1b-shared.patch) ------------------------------------


def test_runtime_attrs_extension_and_clash():
    alg.register_runtime_attrs("t_attrs", lambda s: {"yeto_algo_test": s.entropy_coef}
                               if s.entropy_coef else {})
    try:
        assert AlgorithmSpec().to_legacy_runtime_attrs() == {
            "yeto_rl_dynamic_sampling_max_replacements": None}
        assert AlgorithmSpec(entropy_coef=0.5).to_legacy_runtime_attrs()["yeto_algo_test"] == 0.5
        alg.register_runtime_attrs(
            "t_clash", lambda s: {"yeto_rl_dynamic_sampling_max_replacements": 1})
        with pytest.raises(AlgorithmSpecError, match="redefines"):
            AlgorithmSpec().to_legacy_runtime_attrs()
    finally:
        alg.unregister(runtime_attrs="t_attrs")
        alg.unregister(runtime_attrs="t_clash")
    with pytest.raises(ValueError, match="already registered"):
        alg.register_runtime_attrs("x", dict)
        alg.register_runtime_attrs("x", dict)
    alg.unregister(runtime_attrs="x")


def test_launch_and_island_checks():
    alg.register_launch_check("t_launch", lambda s, run: [f"cp={run['context_parallel_size']}"]
                              if s.entropy_coef == 0.25 else [])
    alg.register_island_check("t_island", lambda s, isl: ["rev"]
                              if s.entropy_coef == 0.25 and isl["base_model_revision"] else [])
    try:
        spec = AlgorithmSpec(entropy_coef=0.25)
        assert alg.launch_problems(spec, {"context_parallel_size": 1}) == ["[t_launch] cp=1"]
        assert alg.launch_problems(AlgorithmSpec(), {"context_parallel_size": 1}) == []
        assert alg.island_problems(spec, {"base_model_revision": "a"}) == ["[t_island] rev"]
    finally:
        alg.unregister(launch_check="t_launch", island_check="t_island")


def test_gradient_rule_only_relaxes_and_is_bound_to_its_mechanism():
    # A rule that would relax *every* round is consulted only for specs that
    # require its mechanism: default GRPO stays exactly the R0 rule.
    alg.register_gradient_rule("t_rule", lambda s, b, m: False,
                               mechanism="features:entropy_bonus")
    try:
        batch = _groups(0.5)
        assert AlgorithmSpec(entropy_coef=0.25).gradient_expectation(batch) == (
            False, "gradient_rule:t_rule (features:entropy_bonus)")
        for spec in (AlgorithmSpec(), BOUNDED):
            for stds in ((0.5,), (0.0,), (0.0, 0.2)):
                b = _groups(*stds)
                assert spec.expects_gradient(b, SimpleNamespace(masked_fraction=1.0)) \
                    == _r0_driver_rule(b)
        assert AlgorithmSpec(entropy_coef=0.25).expects_gradient(_groups(0.0)) is False
        with pytest.raises(ValueError, match="dimension:name"):
            alg.register_gradient_rule("t_bad", lambda s, b, m: None, mechanism="entropy")
    finally:
        alg.unregister(gradient_rule="t_rule")


def test_masked_fraction_must_be_a_real_fraction():
    alg.register_mechanism("features", "t_mask2", lambda s: s.loss.eps_clip == 0.125,
                           masks_tokens=True)
    try:
        spec = AlgorithmSpec(loss=LossSpec(eps_clip=0.125))
        for bad in (True, 2.0, -1, "1.0", float("nan")):
            assert spec.expects_gradient(_groups(0.5), SimpleNamespace(masked_fraction=bad))
        assert not spec.expects_gradient(_groups(0.5), SimpleNamespace(masked_fraction=1))
    finally:
        alg.unregister(mechanism=("features", "t_mask2"))


def test_register_field_requires_hashable_values():
    with pytest.raises(ValueError, match="hashable"):
        alg.register_field("loss", "t_list_default", default=[], parse=lambda p, v: v)
    alg.register_field("loss", "t_list", default=None, parse=lambda p, v: v)
    try:
        with pytest.raises(AlgorithmSpecError, match=r"loss\.t_list.*unhashable"):
            LossSpec().with_ext(t_list=[1, 2])
        assert LossSpec().with_ext(t_list=(1, 2)).t_list == (1, 2)
    finally:
        alg.unregister(field=("loss", "t_list"))


def test_load_extensions_retries_after_a_failed_import(monkeypatch):
    import yeto.rl.algos as algos

    monkeypatch.setattr(alg, "_EXTENSIONS_LOADED", False)
    monkeypatch.setattr(algos, "EXTENSION_MODULES", ("yeto.rl.algos.no_such_extension",))
    with pytest.raises(ImportError):
        alg.load_extensions()
    assert alg._EXTENSIONS_LOADED is False
    monkeypatch.setattr(algos, "EXTENSION_MODULES", ())
    alg.load_extensions()
    assert alg._EXTENSIONS_LOADED is True
