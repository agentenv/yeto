"""Mapping table, absorption and translation (rl-algorithm-capabilities 2.1, 2.3-2.5)."""

from __future__ import annotations

import pytest

from yeto.rl.engine.algorithm import (
    BOUNDED_NONZERO_STD_FILTER,
    STOCK_NONZERO_STD_FILTER,
    AdvantageSpec,
    AlgorithmSpec,
    AlgorithmSpecError,
    CorrectionSpec,
    KlSpec,
    LossSpec,
    PluginRef,
    SamplingSpec,
)
from yeto.rl.engine.miles_adapter import algorithm_flags as af
from yeto.rl.engine.miles_adapter import config as mc
from test_rl_miles_adapter_config import make_config, sub

REF = PluginRef.from_path("yeto.rl.engine.algorithm.plugin_source_sha256")
# rl-algo-grpo-knobs: the only reward post-process of the ports path is the
# yeto dispatcher, and a loss-placed KL must name its reference model.
KL_REF = {"source": "Qwen/Qwen3-0.6B", "revision": "rev-a"}


def _registered(group, name):
    from yeto.rl.engine import algorithm as alg

    alg.load_extensions()
    return name in alg._FIELDS.get(group, {})


# Branch on whether rl-algo-grpo-knobs has registered its fields (not on an
# exception): with them, the dispatcher is the only reward post-process.
KNOBS = _registered("advantage", "reward_shapers")
DISPATCHER = (PluginRef.from_path("yeto.rl.algos.reward_pipeline.post_process")
              if KNOBS else REF)
# With rl-algo-grpo-knobs the only pg_loss reducer is the Dr.GRPO one.
REDUCER = (PluginRef.from_path("yeto.rl.algos.reducers.constant_denominator_reducer")
           if _registered("loss", "constant_denominator") else REF)


def complete(spec):
    """Add the rl-algo-grpo-knobs companions argv cannot carry (no argv change)."""

    if (spec.advantage.reward_postprocess is not None
            and _registered("advantage", "reward_shapers")
            and not spec.advantage.reward_shapers):
        spec = spec.replace(advantage=spec.advantage.with_ext(reward_shapers=[
            {"name": "overlong_penalty", "max_length": 1024, "cache_length": 128}]))
    if (spec.kl.placement == "loss" and _registered("kl", "ref_model")
            and spec.kl.ref_model is None):
        spec = spec.replace(kl=spec.kl.with_ext(ref_model=KL_REF))
    if (spec.loss.reducer is not None and _registered("loss", "constant_denominator")
            and spec.loss.aggregation != "constant"):
        spec = spec.replace(loss=spec.loss.__class__.from_dict(
            {**spec.loss.to_dict(), "aggregation": "constant", "constant_denominator": 1024}))
    if "yeto.rl.algos.grpo_knobs" in _extension_modules():
        # rl-algo-grpo-knobs F1: pipeline code modules pinned in spec.plugins
        from yeto.rl.algos.grpo_knobs import with_pipeline_plugins

        spec = with_pipeline_plugins(spec)
    return spec


def _extension_modules():
    import yeto.rl.algos as algos

    return algos.EXTENSION_MODULES


R0_ARGV = mc.translate_run_config(make_config(), AlgorithmSpec()).argv


# -- 2.1 table ----------------------------------------------------------------


def test_mapping_table_is_subset_of_objective_list():
    assert af.mapped_flags() <= af.objective_flags()
    assert not (af.mapped_flags() & af.UNMAPPED_OBJECTIVE_FLAGS)
    for flag, row in af.MAPPINGS.items():
        assert row.flag == flag and flag.startswith("--")
        assert callable(row.parse) and callable(row.absorb) and callable(row.translate)
        assert row.field.split(".")[0] in {
            "advantage", "loss", "kl", "correction", "sampling", "entropy_coef", "critic",
        }


def test_design_d3_flags_are_all_mapped():
    d3 = {
        "--advantage-estimator", "--eps-clip", "--eps-clip-high", "--eps-clip-c",
        "--calculate-per-token-loss", "--disable-grpo-std-normalization",
        "--disable-rewards-normalization", "--normalize-advantages", "--kl-coef",
        "--use-kl-loss", "--kl-loss-coef", "--kl-loss-type", "--use-unbiased-kl",
        "--entropy-coef", "--use-tis", "--tis-clip", "--tis-clip-low",
        "--custom-tis-function-path", "--use-rollout-logprobs", "--get-mismatch-metrics",
        "--use-opsm", "--opsm-delta", "--custom-pg-loss-reducer-function-path",
        "--custom-reward-post-process-path", "--loss-type", "--custom-loss-function-path",
        "--dynamic-sampling-filter-path", "--over-sampling-batch-size",
    }
    # The reviewed P0 rows are exactly D3; the table is D3 plus the rows that
    # registered extension modules declared (register_flag) -- nothing else.
    assert af.BUILTIN_FLAGS == d3
    assert af.mapped_flags() == d3 | af.EXTENSION_FLAGS
    assert not (af.EXTENSION_FLAGS & d3)


def test_objective_flags_are_adapter_owned():
    assert af.objective_flags() <= mc.ADAPTER_OWNED_FLAGS


# -- 2.3 absorb / conflict / reject ----------------------------------------------


def test_absorb_equals_direct_spec_and_leaves_other_argv():
    spec, rest, absorbed = af.absorb_extra_argv(
        AlgorithmSpec(), ["--eps-clip-high", "0.28", "--foo", "1", "--calculate-per-token-loss"]
    )
    direct = AlgorithmSpec(loss=LossSpec(eps_clip_high=0.28, aggregation="token"))
    assert spec == direct and spec.sha256() == direct.sha256()
    assert rest == ("--foo", "1")
    assert absorbed == {"--eps-clip-high": "0.28", "--calculate-per-token-loss": True}
    spec2, _, _ = af.absorb_extra_argv(AlgorithmSpec(), ["--eps-clip-high=0.28",
                                                         "--calculate-per-token-loss"])
    assert spec2 == direct


def test_absorb_same_value_is_not_a_conflict():
    base = AlgorithmSpec(loss=LossSpec(eps_clip_high=0.28))
    spec, _, _ = af.absorb_extra_argv(base, ["--eps-clip-high", "0.28"])
    assert spec == base


def test_conflict_names_flag_and_both_values():
    base = AlgorithmSpec(loss=LossSpec(eps_clip_high=0.28))
    with pytest.raises(af.AlgorithmFlagConflict, match=r"--eps-clip-high.*0\.3.*0\.28"):
        af.absorb_extra_argv(base, ["--eps-clip-high", "0.3"])
    with pytest.raises(af.AlgorithmFlagConflict, match="reward KL.*loss KL"):
        af.absorb_extra_argv(AlgorithmSpec(), ["--kl-coef", "0.1", "--use-kl-loss",
                                               "--kl-loss-coef", "0.1", "--kl-loss-type", "k1"])
    with pytest.raises(af.AlgorithmFlagConflict, match="given twice"):
        af.absorb_extra_argv(AlgorithmSpec(), ["--eps-clip", "0.2", "--eps-clip", "0.3"])
    with pytest.raises(mc.MilesConfigError, match=r"--eps-clip-high.*0\.3.*0\.28"):
        mc.translate_run_config(make_config(), base, extra_argv=("--eps-clip-high", "0.3"))


@pytest.mark.parametrize("flag", ["--gamma", "--value-clip", "--lambd", "--use-routing-replay",
                                  "--rollout-temperature", "--partial-rollout"])
def test_unmapped_objective_flag_rejected(flag):
    if flag in af.mapped_flags():
        pytest.skip(f"{flag} is mapped by a registered extension (EXTENSION_FLAGS)")
    argv = [flag] if flag in ("--use-routing-replay", "--partial-rollout") else [flag, "0.9"]
    with pytest.raises(mc.MilesConfigError, match=flag):
        mc.check_extra_argv(argv)
    with pytest.raises(af.UnmappedAlgorithmFlag, match=flag):
        af.absorb_extra_argv(AlgorithmSpec(), argv)
    with pytest.raises(mc.MilesConfigError, match=flag):
        mc.translate_run_config(make_config(), AlgorithmSpec(), extra_argv=tuple(argv))


def test_translate_absorbs_and_records():
    launch = mc.translate_run_config(
        make_config(), AlgorithmSpec(), extra_argv=("--eps-clip-high", "0.28")
    )
    assert launch.algorithm == AlgorithmSpec(loss=LossSpec(eps_clip_high=0.28))
    assert launch.algorithm_sha256 == launch.algorithm.sha256()
    assert launch.absorbed_flags == {"--eps-clip-high": "0.28"}
    argv = list(launch.argv)
    assert argv.count("--eps-clip-high") == 1 and argv[argv.index("--eps-clip-high") + 1] == "0.28"


def test_still_owned_flags_rejected():
    with pytest.raises(mc.MilesConfigError, match="owned by the ports adapter"):
        mc.check_extra_argv(["--colocate"])


# -- 2.4 translation -----------------------------------------------------------------


def test_default_and_v1_argv_unchanged():
    for spec in (AlgorithmSpec(), AlgorithmSpec(kl_coef=0.0),
                 AlgorithmSpec(dynamic_sampling_filter=BOUNDED_NONZERO_STD_FILTER,
                               dynamic_sampling_max_replacements=2)):
        assert af.algorithm_argv(spec) == []
    assert mc.translate_run_config(make_config(), AlgorithmSpec()).argv == R0_ARGV


CASES = [
    (dict(loss=LossSpec(eps_clip=0.25)), ["--eps-clip", "0.25"]),
    (dict(loss=LossSpec(eps_clip_high=0.28)), ["--eps-clip-high", "0.28"]),
    (dict(loss=LossSpec(eps_clip_c=3)), ["--eps-clip-c", "3.0"]),
    (dict(loss=LossSpec(aggregation="token")), ["--calculate-per-token-loss"]),
    (dict(loss=LossSpec(reducer=REDUCER)),
     ["--custom-pg-loss-reducer-function-path", REDUCER.path]),
    (dict(loss=LossSpec(variant="custom_loss", custom_loss=REF)),
     ["--loss-type", "custom_loss", "--custom-loss-function-path", REF.path]),
    (dict(advantage=AdvantageSpec(std_normalization=False)), ["--disable-grpo-std-normalization"]),
    (dict(advantage=AdvantageSpec(rewards_normalization=False)),
     ["--disable-rewards-normalization"]),
    (dict(advantage=AdvantageSpec(whiten=True)), ["--normalize-advantages"]),
    (dict(advantage=AdvantageSpec(reward_postprocess=DISPATCHER)),
     ["--custom-reward-post-process-path", DISPATCHER.path]),
    (dict(kl=KlSpec(placement="loss", coef=0.01, estimator="k3", unbiased=True)),
     ["--use-kl-loss", "--kl-loss-coef", "0.01", "--kl-loss-type", "k3", "--use-unbiased-kl"]),
    (dict(entropy_coef=0.001), ["--entropy-coef", "0.001"]),
    (dict(correction=CorrectionSpec(method="tis", tis_clip=2, tis_clip_low=0.5)),
     ["--use-tis", "--tis-clip", "2.0", "--tis-clip-low", "0.5"]),
    (dict(correction=CorrectionSpec(method="custom", function=REF, tis_clip=5,
                                    tis_clip_low=0.5, mismatch_metrics=True)),
     ["--use-tis", "--tis-clip", "5.0", "--tis-clip-low", "0.5",
      "--custom-tis-function-path", REF.path, "--get-mismatch-metrics"]),
    (dict(correction=CorrectionSpec(use_rollout_logprobs=True)), ["--use-rollout-logprobs"]),
    (dict(correction=CorrectionSpec(method="opsm", opsm_delta=1e-4)),
     ["--use-opsm", "--opsm-delta", "0.0001"]),
]


@pytest.mark.parametrize("change, fragment", CASES)
def test_each_mapped_field_translates(change, fragment):
    spec = AlgorithmSpec(**change)
    assert af.algorithm_argv(spec) == fragment
    # absorption of the translation reproduces the spec (round trip)
    absorbed, rest, _ = af.absorb_extra_argv(AlgorithmSpec(), fragment)
    assert absorbed == spec and rest == ()
    # absorbed -> completed (1b: ref_model / reward_shapers) -> translated
    assert complete(absorbed) == complete(spec)
    argv = list(mc.translate_run_config(make_config(), complete(absorbed)).argv)
    i = argv.index(fragment[0])
    assert argv[i:i + len(fragment)] == fragment
    # the rest of the argv is the default GRPO argv, byte for byte
    assert argv[:i] + argv[i + len(fragment):] == list(R0_ARGV)


def test_r0_positioned_fields_translate():
    gspo = AlgorithmSpec(advantage=AdvantageSpec(estimator="gspo"),
                         loss=LossSpec(eps_clip=3e-4, eps_clip_high=4e-4))
    cfg = sub(make_config(), "algorithm", advantage_estimator="gspo")
    argv = list(mc.translate_run_config(cfg, gspo).argv)
    assert argv[argv.index("--advantage-estimator") + 1] == "gspo"
    stock = AlgorithmSpec(sampling=SamplingSpec(filter=STOCK_NONZERO_STD_FILTER))
    argv = list(mc.translate_run_config(make_config(), stock).argv)
    assert argv[argv.index("--dynamic-sampling-filter-path") + 1] == STOCK_NONZERO_STD_FILTER
    over = AlgorithmSpec(sampling=SamplingSpec(filter=BOUNDED_NONZERO_STD_FILTER,
                                               over_sampling_batch_size=4))
    argv = list(mc.translate_run_config(make_config(), over).argv)
    assert argv[argv.index("--over-sampling-batch-size") + 1] == "4"
    with pytest.raises(mc.UnmappedConfigError, match="over_sampling_batch_size"):
        mc.translate_run_config(make_config(), AlgorithmSpec(
            sampling=SamplingSpec(filter=BOUNDED_NONZERO_STD_FILTER,
                                  over_sampling_batch_size=8)))


# -- 2.5 KL placement -------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["grpo", "gspo"])
def test_reward_kl_rejected_for_grpo_gspo(estimator):
    spec = AlgorithmSpec(advantage=AdvantageSpec(estimator=estimator),
                         kl=KlSpec(placement="reward", coef=0.1),
                         loss=LossSpec(eps_clip=3e-4, eps_clip_high=4e-4))
    problems = spec.rejections()
    assert any("placement='loss'" in p for p in problems)
    cfg = sub(make_config(), "algorithm", advantage_estimator=estimator)
    with pytest.raises(mc.MilesConfigError, match="placement='loss'"):
        mc.translate_run_config(cfg, spec)


def test_reward_kl_allowed_for_reinforce_plus_plus():
    spec = AlgorithmSpec(advantage=AdvantageSpec(estimator="reinforce_plus_plus", whiten=True),
                         kl=KlSpec(placement="reward", coef=0.1))
    assert spec.rejections() == []
    assert any("whiten" in p for p in AlgorithmSpec(
        advantage=AdvantageSpec(estimator="reinforce_plus_plus")).rejections())
    cfg = sub(make_config(), "algorithm", advantage_estimator="reinforce_plus_plus")
    argv = list(mc.translate_run_config(cfg, spec).argv)
    assert argv[argv.index("--kl-coef") + 1] == "0.1"


def test_loss_kl_translates():
    spec = complete(AlgorithmSpec(kl=KlSpec(placement="loss", coef=0.02, estimator="low_var_kl")))
    argv = list(mc.translate_run_config(make_config(), spec).argv)
    assert "--kl-coef" not in argv
    i = argv.index("--use-kl-loss")
    assert argv[i:i + 5] == ["--use-kl-loss", "--kl-loss-coef", "0.02", "--kl-loss-type",
                             "low_var_kl"]


@pytest.mark.parametrize(
    "kl_coef, kl_flag, rejected",
    [(None, None, False), (0.0, "0.0", False), (0.1, None, True)],
)
def test_v1_kl_coef_inputs(kl_coef, kl_flag, rejected):
    spec = AlgorithmSpec(kl_coef=kl_coef)
    if rejected:
        assert spec.kl == KlSpec(placement="reward", coef=0.1)
        with pytest.raises(mc.MilesConfigError, match="ignores a KL placed in the reward"):
            mc.translate_run_config(make_config(), spec)
        return
    argv = list(mc.translate_run_config(make_config(), spec).argv)
    if kl_flag is None:
        # placement none: no --kl-coef, so Miles' kl_coef stays 0 -> no reference model
        assert spec.kl.placement == "none" and "--kl-coef" not in argv
        assert "--use-kl-loss" not in argv
    else:
        assert argv[argv.index("--kl-coef") + 1] == kl_flag
        assert spec.sha256() == AlgorithmSpec.from_dict(
            {"advantage_estimator": "grpo", "kl_coef": 0.0}).sha256()


def test_reward_and_loss_kl_cannot_coexist():
    with pytest.raises(AlgorithmSpecError):
        KlSpec(placement="reward", coef=0.1, estimator="k1")


def test_custom_config_path_refused():
    with pytest.raises(mc.MilesConfigError, match="--custom-config-path"):
        mc.check_extra_argv(["--custom-config-path", "x.yaml"])
    with pytest.raises(mc.MilesConfigError, match="--custom-config-path"):
        mc.translate_run_config(make_config(), AlgorithmSpec(),
                                extra_argv=("--custom-config-path=x.yaml",))
