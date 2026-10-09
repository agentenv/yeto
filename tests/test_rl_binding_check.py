"""yeto-framework-decoupling 4.4 (neutral filter names + legacy normalization)
and 4.4a (training-side binding check, design D6 / D6a)."""

from __future__ import annotations

import pytest

from yeto.rl.adapters.miles import binding
from yeto.rl.adapters.miles.algorithm_flags import legacy_algorithm_argv
from yeto.rl.engine import algorithm as alg
from yeto.rl.engine.algorithm import AlgorithmSpec, SamplingSpec
from yeto.rl.engine.cut import AlgorithmIdentity
from yeto.rl.engine.execution_profile import ProfileError, check_algorithm_contract

B_OLD = "yeto.rl.filters.bounded_nonzero_reward_std"
S_OLD = "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"

# AlgorithmSpec.sha256() of the same specs on agentenv/main 80e944b6 (pre-4.4,
# filter stored as the Miles path) -> after 4.4.  Recorded in hash-migration.md.
SPECS = {
    "v1_bounded_r2": (lambda f: AlgorithmSpec(advantage_estimator="grpo", dynamic_sampling_filter=f,
                                              dynamic_sampling_max_replacements=2), B_OLD,
                      "32ba45a6d3e466e6b664f938779959fdae8767236d5177eabf59f62957e5a47a",
                      "b669e6c4de10a52d95ecc45294bbfeba857dddd0769abc358528729407bae37e"),
    "v2_stock": (lambda f: AlgorithmSpec(sampling=SamplingSpec(filter=f)), S_OLD,
                 "1129f5c0c80f08bf3d69c08e14e1c803c867e6508e1d55fc53934e671a774e2a",
                 "05d7bc4b978f11caf6489e530081dcc62ab2131b2261445ee10eefebeb012bb8"),
    "v2_bounded_r4": (lambda f: AlgorithmSpec(sampling=SamplingSpec(filter=f, max_replacements=4)),
                      B_OLD,
                      "f25cf271d45d38253ea1f255095fe5c0a9b478841a6f90270533382b7adb41dc",
                      "ac623863613bc25aa7506f31b50e799b5a272919ecdcf05885ff72a9e38d9038"),
}


def test_every_legacy_name_normalizes_to_a_neutral_name():
    assert set(alg.LEGACY_FILTER_NAMES) == {B_OLD, S_OLD}
    for old, new in alg.LEGACY_FILTER_NAMES.items():
        assert alg.normalize_filter_name(old) == new
        assert alg.normalize_filter_name(new) == new
        assert "." not in new and new in alg.DYNAMIC_SAMPLING_FILTERS
    assert alg.normalize_filter_name(None) is None


@pytest.mark.parametrize("name", sorted(SPECS))
def test_old_and_new_names_give_the_same_new_hash(name):
    build, old_name, old_hash, new_hash = SPECS[name]
    neutral = alg.LEGACY_FILTER_NAMES[old_name]
    from_old, from_new = build(old_name), build(neutral)
    assert from_old.dynamic_sampling_filter == neutral
    assert from_old.sha256() == from_new.sha256() == new_hash != old_hash


def test_miles_argv_is_byte_identical_to_the_pre_rename_argv():
    spec = SPECS["v1_bounded_r2"][0](B_OLD)
    assert legacy_algorithm_argv(spec)[2:] == ["--dynamic-sampling-filter-path", B_OLD]
    assert binding.filter_argv(alg.STOCK_NONZERO_STD_FILTER) == ["--dynamic-sampling-filter-path", S_OLD]


def test_old_version_island_is_refused():
    """A launcher/peer on the pre-4.4 code binds the old hash: contract check refuses."""
    build, old_name, old_hash, new_hash = SPECS["v2_bounded_r4"]

    class Profile:  # minimal stand-in for an ExecutionProfile bound by the old version
        name = "old-version-island"
        algorithm_spec_sha256 = old_hash
        max_policy_age = 0

    with pytest.raises(ProfileError, match=old_hash):
        check_algorithm_contract(Profile(), build(old_name))


def test_checkpoint_cut_with_old_hash_does_not_match_the_run():
    build, old_name, old_hash, new_hash = SPECS["v1_bounded_r2"]
    cut = AlgorithmIdentity(algorithm_spec_sha256=old_hash)
    run = AlgorithmIdentity(algorithm_spec_sha256=build(old_name).sha256())
    assert cut != run  # verify_cut reports "algorithm identity differs" and refuses the resume


# ---------------------------------------------------------------- 4.4a
def test_binding_check_passes_for_the_real_table():
    report = binding.check_spec(SPECS["v1_bounded_r2"][0](B_OLD))
    (row,) = report["filters"]
    assert row["argv"] == "same" and row["mapped"] == B_OLD
    assert row["identity"]["module"] == "yeto.rl.filters" and row["identity"]["source_sha256"]
    assert row["call"] == "same"
    stock = binding.check_filter(alg.STOCK_NONZERO_STD_FILTER)
    assert stock["argv"] == "same"
    assert stock["call"] == "same" or stock["call"].startswith("skipped")  # needs Miles


def test_binding_to_another_function_fails_on_identity():
    wrong = {alg.BOUNDED_NONZERO_STD_FILTER: "yeto.rl.filters._group_key"}
    with pytest.raises(binding.BindingError, match="①"):
        binding.check_filter(alg.BOUNDED_NONZERO_STD_FILTER, paths=wrong)
    # same argv spelling but a different implementation resolved behind it (e.g. a
    # shadowing module on the path): identity (②) catches it
    shadowed = iter([{"module": "m", "qualname": "f", "source_sha256": "aa"},
                     {"module": "m", "qualname": "f", "source_sha256": "bb"}])
    with pytest.raises(binding.BindingError, match="②"):
        binding.check_filter(alg.BOUNDED_NONZERO_STD_FILTER, identify=lambda path: next(shadowed))


def test_binding_with_a_misspelled_parameter_fails_on_argv():
    wrong = {alg.BOUNDED_NONZERO_STD_FILTER: B_OLD + "_"}
    with pytest.raises(binding.BindingError, match="① argv"):
        binding.check_filter(alg.BOUNDED_NONZERO_STD_FILTER, paths=wrong)


def test_binding_call_check_compares_results():
    calls = iter([lambda args, s: True, lambda args, s: False])
    with pytest.raises(binding.BindingError, match="③ call"):
        binding.check_filter(alg.BOUNDED_NONZERO_STD_FILTER, loader=lambda path: next(calls))


def test_launch_preflight_refuses_a_broken_binding(monkeypatch):
    from yeto.rl.adapters.miles import entry

    monkeypatch.setattr(binding, "FILTER_PATHS", {alg.BOUNDED_NONZERO_STD_FILTER: "yeto.rl.filters._group_key"})
    spec = SPECS["v1_bounded_r2"][0](B_OLD)
    with pytest.raises(binding.BindingError):
        binding.check_spec(spec)
    assert "check_spec(algorithm)" in open(entry.__file__).read()  # wired into preflight()


@pytest.mark.skip(reason="verl adapter not implemented yet (rl-verl-backend): its mapping table "
                  "gets the same three checks (D6a)")
def test_verl_binding_check_placeholder():
    raise AssertionError("implement with the verl adapter")
