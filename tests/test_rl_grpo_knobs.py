"""CPU tests for rl-algo-grpo-knobs (no Miles needed)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from yeto.rl.algos import grpo_knobs as gk
from yeto.rl.algos import reward_pipeline as rp
from yeto.rl.algos import sample_filters as sf
from yeto.rl.engine.algorithm import (
    BOUNDED_NONZERO_STD_FILTER,
    AlgorithmSpec,
    AlgorithmSpecError,
    PluginRef,
)
from yeto.rl.engine.miles_adapter.algorithm_flags import algorithm_argv

DEFAULT_SHA = AlgorithmSpec(advantage_estimator="grpo").sha256()
OVERLONG = {"name": "overlong_penalty", "max_length": 100, "cache_length": 20}


def dispatcher():
    return rp.dispatcher_ref().to_dict()


def reducer():
    return PluginRef.from_path(gk.REDUCER_PATH).to_dict()


def pipeline_spec(**adv):
    return gk.with_pipeline_plugins(AlgorithmSpec(advantage={"reward_postprocess": dispatcher(), **adv}))


# ---------------------------------------------------------------- 5.1 / hashes


def test_modules_import_without_gpu():
    import yeto.rl.algos.reducers  # noqa: F401

    assert "grpo_default" in rp.ADV_TRANSFORMS and "overlong_penalty" in rp.REWARD_SHAPERS


def test_default_hash_and_argv_unchanged():
    spec = AlgorithmSpec(advantage_estimator="grpo")
    assert spec.is_v1_expressible()
    same = AlgorithmSpec.from_dict(spec.structured_dict())
    assert same.sha256() == DEFAULT_SHA and algorithm_argv(same) == []
    assert gk.runtime_attrs(spec) == {} and spec.rejections() == []
    # an explicit default transform / empty shaper list is not a new field
    explicit = AlgorithmSpec(advantage={"transform": "grpo_default", "reward_shapers": []})
    assert explicit.sha256() == DEFAULT_SHA


# ---------------------------------------------------------------- 2.1 constraints


@pytest.mark.parametrize("group,payload,field", [
    ("loss", {"eps_clip_c": 1.0}, "loss.eps_clip_c must be > 1.0"),
    ("loss", {"eps_clip_c": 0.5}, "loss.eps_clip_c must be > 1.0"),
    ("loss", {"eps_clip_high": math.inf}, "loss.eps_clip_high must be finite"),
    ("loss", {"eps_clip_high": 0.0}, "loss.eps_clip_high must be > 0.0"),
    ("loss", {"eps_clip_high": -0.1}, "loss.eps_clip_high must be > 0.0"),
    (None, {"entropy_coef": math.nan}, "entropy_coef must be finite"),
    (None, {"entropy_coef": math.inf}, "entropy_coef must be finite"),
])
def test_field_constraints(group, payload, field):
    with pytest.raises(AlgorithmSpecError, match=field.replace(".", r"\.")):
        AlgorithmSpec(**({group: payload} if group else payload))


def test_over_sampling_needs_filter_and_batch():
    spec = AlgorithmSpec(sampling={"over_sampling_batch_size": 8})
    assert any("sampling.over_sampling_batch_size requires a dynamic sampling filter" in p
               for p in spec.rejections())
    ok = AlgorithmSpec(sampling={"filter": BOUNDED_NONZERO_STD_FILTER, "over_sampling_batch_size": 8})
    assert ok.rejections() == []
    problems = gk.launch_problems(ok, rollout_batch_size=16, rollout_max_response_len=1024)
    assert problems == [
        "sampling.over_sampling_batch_size=8 must be >= the rollout batch size 16"
    ]
    assert gk.launch_problems(ok, rollout_batch_size=8, rollout_max_response_len=1024) == []


# ---------------------------------------------------------------- 2.2 translation


@pytest.mark.parametrize("kwargs,fragment", [
    ({"loss": {"eps_clip": 0.2, "eps_clip_high": 0.28}}, ["--eps-clip", "0.2", "--eps-clip-high", "0.28"]),
    ({"loss": {"eps_clip_c": 3}}, ["--eps-clip-c", "3.0"]),
    ({"loss": {"aggregation": "token"}}, ["--calculate-per-token-loss"]),
    ({"advantage": {"std_normalization": False}}, ["--disable-grpo-std-normalization"]),
    ({"entropy_coef": 0.001}, ["--entropy-coef", "0.001"]),
    ({"kl": {"placement": "loss", "coef": 0.001, "estimator": "k3",
             "ref_model": {"source": "Qwen/Qwen3-0.6B", "revision": "abc"}}},
     ["--use-kl-loss", "--kl-loss-coef", "0.001", "--kl-loss-type", "k3"]),
])
def test_translation_fragments(kwargs, fragment):
    spec = AlgorithmSpec(**kwargs)
    assert spec.rejections() == []
    assert algorithm_argv(spec) == fragment


def test_over_sampling_translated_by_config_position():
    # --over-sampling-batch-size is emitted in its R0 position by
    # translate_run_config (from the config batch, checked against the spec).
    spec = AlgorithmSpec(sampling={"filter": BOUNDED_NONZERO_STD_FILTER, "over_sampling_batch_size": 8})
    assert algorithm_argv(spec) == []


# ---------------------------------------------------------------- 3.1 / 3.2 KL ref


REF = {"source": "Qwen/Qwen3-0.6B", "revision": "rev-a"}


def kl_spec(**ref):
    return AlgorithmSpec(kl={"placement": "loss", "coef": 0.001, "estimator": "k3",
                             "ref_model": {**REF, **ref}})


def test_ref_model_enters_hash():
    assert kl_spec().sha256() != kl_spec(revision="rev-b").sha256()
    assert kl_spec().sha256() == kl_spec().sha256()
    assert json.loads(kl_spec().canonical_json())["kl"]["ref_model"] == REF


def test_ref_model_required_for_loss_kl():
    spec = AlgorithmSpec(kl={"placement": "loss", "coef": 0.001, "estimator": "k3"})
    assert any("kl.ref_model {source, revision} is required" in p for p in spec.rejections())
    with pytest.raises(AlgorithmSpecError, match="kl.ref_model must be an object"):
        AlgorithmSpec(kl={"placement": "loss", "coef": 0.001, "estimator": "k3",
                          "ref_model": {"source": "x"}})
    bad = AlgorithmSpec(kl={"placement": "reward", "coef": 0.0, "ref_model": REF})
    assert any("kl.ref_model applies to kl.placement='loss' only" in p for p in bad.rejections())


def test_unknown_kl_estimator_lists_allowed():
    with pytest.raises(AlgorithmSpecError, match=r"kl.estimator must be one of \['k1', 'k2', 'k3', 'low_var_kl'\]"):
        AlgorithmSpec(kl={"placement": "loss", "coef": 0.001, "estimator": "k9", "ref_model": REF})


def test_check_ref_model_against_base():
    gk.check_ref_model(kl_spec(), "rev-a")
    gk.check_ref_model(AlgorithmSpec(), None)
    with pytest.raises(AlgorithmSpecError, match="does not match this island's base_model_revision 'rev-b'"):
        gk.check_ref_model(kl_spec(), "rev-b")


# ---------------------------------------------------------------- 4.3 constant


def constant_spec(**loss):
    return AlgorithmSpec(advantage={"std_normalization": False},
                         loss={"aggregation": "constant", "reducer": reducer(),
                               "constant_denominator": 4096, **loss})


def test_constant_translation_and_hash():
    spec = constant_spec()
    assert spec.rejections() == []
    assert algorithm_argv(spec) == [
        "--disable-grpo-std-normalization",
        "--custom-pg-loss-reducer-function-path", gk.REDUCER_PATH,
    ]
    assert constant_spec(constant_denominator=1000).sha256() != spec.sha256()
    cfg = gk.runtime_attrs(spec)[rp.PIPELINE_ATTR]["config"]
    assert cfg["reducer"] == {"denominator": 4096.0} and "reward_pipeline" not in cfg


def test_constant_rejections():
    missing = AlgorithmSpec(loss={"aggregation": "constant", "reducer": reducer()})
    assert any("requires loss.constant_denominator" in p for p in missing.rejections())
    with pytest.raises(AlgorithmSpecError, match="loss.constant_denominator must be a finite number > 0"):
        constant_spec(constant_denominator=0)
    # token aggregation and constant are one enum: the argv switch conflicts.
    from yeto.rl.engine.miles_adapter.algorithm_flags import AlgorithmFlagConflict, absorb_extra_argv

    with pytest.raises(AlgorithmFlagConflict, match="loss.aggregation"):
        absorb_extra_argv(constant_spec(), ["--calculate-per-token-loss"])
    assert gk.launch_problems(constant_spec(), rollout_batch_size=1, rollout_max_response_len=1,
                              context_parallel_size=2) == [
        "loss.aggregation='constant' requires context parallel size 1, got 2"
    ]
    stray = AlgorithmSpec(loss={"reducer": reducer()})
    assert any("requires loss.aggregation='constant'" in p for p in stray.rejections())


# ---------------------------------------------------------------- 5.3 / 5.4 dispatcher wiring


def test_dispatcher_argv_only_when_needed():
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    assert spec.rejections() == []
    assert algorithm_argv(spec) == ["--custom-reward-post-process-path", rp.DISPATCHER_PATH]
    assert algorithm_argv(AlgorithmSpec()) == []
    lone = pipeline_spec()
    assert any("no reward shaper or non-default advantage.transform" in p for p in lone.rejections())
    missing = gk.with_pipeline_plugins(AlgorithmSpec(advantage={"reward_shapers": [OVERLONG]}))
    assert any("need the dispatcher" in p for p in missing.rejections())
    other = AlgorithmSpec(advantage={"reward_postprocess": PluginRef.from_path(
        "yeto.rl.algos.reducers.constant_denominator_reducer").to_dict()})
    assert any("uses only the yeto dispatcher" in p for p in other.rejections())


def test_runtime_attrs_hash_checked():
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    payload = gk.runtime_attrs(spec)[rp.PIPELINE_ATTR]
    args = SimpleNamespace(**{rp.PIPELINE_ATTR: payload})
    cfg = rp.read_plugins(args)
    assert cfg["algorithm_spec_sha256"] == spec.sha256()
    assert cfg["reward_pipeline"]["reward_shapers"] == [OVERLONG]
    # JSON string form (argparse namespace set from a serialized value) also works
    assert rp.read_plugins(SimpleNamespace(**{rp.PIPELINE_ATTR: json.dumps(payload)})) == cfg
    tampered = json.loads(json.dumps(payload))
    tampered["config"]["reward_pipeline"]["reward_shapers"][0]["cache_length"] = 10
    with pytest.raises(rp.RewardPipelineError, match="hash mismatch"):
        rp.read_plugins(SimpleNamespace(**{rp.PIPELINE_ATTR: tampered}))
    with pytest.raises(rp.RewardPipelineError, match="is missing"):
        rp.read_plugins(SimpleNamespace())


def test_multi_lora_refused():
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    assert gk.launch_problems(spec, rollout_batch_size=1, rollout_max_response_len=1024,
                              multi_lora=True)[0].startswith("multi-LoRA")
    args = SimpleNamespace(multi_lora=True, **gk.runtime_attrs(spec))
    with pytest.raises(rp.RewardPipelineError, match="multi-LoRA"):
        rp.post_process(args, [])


class S(SimpleNamespace):
    def get_reward_value(self, args):
        return self.reward


def _args(spec, **kw):
    base = dict(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True,
                n_samples_per_prompt=4, rollout_batch_size=2, multi_lora=False)
    base.update(kw)
    return SimpleNamespace(**base, **gk.runtime_attrs(spec))


def test_fallback_warning_event(tmp_path, caplog):
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    tape = tmp_path / "events.jsonl"
    args = _args(spec, yeto_rl_event_tape=str(tape), yeto_rl_learner_id=0)
    samples = [S(reward=float(i % 2), group_index=None, index=i, rollout_id=None,
                 response_length=10, metadata=None) for i in range(5)]
    rp.post_process(args, samples)
    events = [json.loads(line) for line in tape.read_text().splitlines()]
    fallback = [e for e in events if e["event"] == rp.FALLBACK_EVENT]
    assert fallback == [{**fallback[0], "samples": 5, "expected_fixed_fanout": 8,
                         "missing_group_index": 5}]


def test_extension_point_identity_transform():
    """A transform registered at runtime runs through the unchanged entry point (5.5)."""

    seen = {}

    def identity(args, samples, rewards, groups, params):
        seen["groups"], seen["params"] = groups, params
        return list(rewards)

    # A real extension registers a module-level function of a yeto module;
    # this test function stands in as part of reward_pipeline.
    identity.__module__, identity.__qualname__ = rp.__name__, "test_identity"

    rp.register_advantage_transform("test_identity", identity,
                                    validate=lambda params, spec: [] if params.get("k") == 1 else ["k must be 1"])
    try:
        spec = pipeline_spec(transform="test_identity", transform_params={"k": 1})
        assert spec.rejections() == []
        assert json.loads(spec.canonical_json())["advantage"]["transform"] == "test_identity"
        bad = pipeline_spec(transform="test_identity", transform_params={"k": 2})
        assert any("advantage.transform[test_identity]: k must be 1" in p for p in bad.rejections())
        samples = [S(reward=r, group_index=0, index=i, rollout_id=None, response_length=5, metadata=None)
                   for i, r in enumerate([1.0, 0.0, 0.5])]
        shaped, adv = rp.post_process(_args(spec), samples)
        assert shaped == adv == [1.0, 0.0, 0.5]
        assert seen == {"groups": [[0, 1, 2]], "params": {"k": 1}}
    finally:
        rp.ADV_TRANSFORMS.pop("test_identity")


def test_unregistered_stage_names_rejected():
    with pytest.raises(AlgorithmSpecError, match="not a registered reward shaper"):
        pipeline_spec(reward_shapers=[{"name": "nope"}])
    with pytest.raises(AlgorithmSpecError, match="advantage.transform must be one of"):
        pipeline_spec(transform="nope")


# ---------------------------------------------------------------- 6.1 / 6.2 overlong


def test_overlong_values():
    assert [rp.overlong_penalty_value(n, 100, 20) for n in (80, 90, 100, 101)] == [0.0, -0.5, -1.0, -1.0]


def test_overlong_multi_segment_consistent():
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    samples = [
        S(reward=1.0, group_index=0, index=0, rollout_id=7, response_length=50, metadata=None),
        S(reward=1.0, group_index=0, index=1, rollout_id=7, response_length=40, metadata=None),
        S(reward=0.0, group_index=0, index=2, rollout_id=8, response_length=30, metadata=None),
    ]
    shaped, adv = rp.post_process(_args(spec), samples)
    assert shaped == [0.5, 0.5, 0.0]  # rollout 7 total 90 -> -0.5 on both segments
    assert adv[0] == adv[1]


@pytest.mark.parametrize("params,message", [
    ({"max_length": 100, "cache_length": 120}, "cache_length=120, max_length=100"),
    ({"max_length": 100, "cache_length": 0}, "0 < cache_length <= max_length"),
    ({"max_length": 100}, "cache_length must be an integer"),
])
def test_overlong_bad_params(params, message):
    spec = pipeline_spec(reward_shapers=[{"name": "overlong_penalty", **params}])
    assert any(message in p for p in spec.rejections())


def test_overlong_limit_vs_generation_length():
    spec = pipeline_spec(reward_shapers=[{"name": "overlong_penalty", "max_length": 2048, "cache_length": 256}])
    assert gk.launch_problems(spec, rollout_batch_size=1, rollout_max_response_len=1024) == [
        "overlong_penalty.max_length=2048 exceeds the generation limit rollout_max_response_len=1024"
    ]


def test_raw_and_shaped_reward_event(caplog):
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    samples = [S(reward=1.0, group_index=0, index=i, rollout_id=None, response_length=n, metadata=None)
               for i, n in enumerate((80, 90, 100, 101))]
    with caplog.at_level("WARNING"):
        shaped, _ = rp.post_process(_args(spec), samples)
    assert [s.metadata[rp.RAW_REWARD_METADATA_KEY] for s in samples] == [1.0] * 4
    event = rp.reward_summary_event([1.0] * 4, shaped)
    assert event["raw_reward"] == {"mean": 1.0, "min": 1.0, "max": 1.0}
    assert event["shaped_reward"] == {"mean": 0.375, "min": 0.0, "max": 1.0}
    assert event["shaped_samples"] == 3
    assert rp.REWARD_SUMMARY_EVENT in caplog.text


# ---------------------------------------------------------------- 6.3 overlong filter


def _group(statuses):
    return [SimpleNamespace(index=i, status=SimpleNamespace(value=s), remove_sample=False, metadata=None)
            for i, s in enumerate(statuses)]


def test_overlong_filter_marks_truncated_only():
    spec = gk.with_pipeline_plugins(AlgorithmSpec(sampling={"overlong_filter": True}))
    args = SimpleNamespace(**gk.runtime_attrs(spec))
    data = [_group(["completed", "truncated", "completed", "truncated"])]
    assert sf.apply_sample_filters(args, data) == {"overlong_filter": 2}
    assert [s.remove_sample for s in data[0]] == [False, True, False, True]
    assert data[0][1].metadata == {sf.FILTERED_BY_KEY: sf.OVERLONG_FILTER}
    assert sf.metadata_fields(args) == {"filtered_samples": {"overlong_filter": 2}}
    assert sf.group_filtered_samples(data[0]) == 2


def test_overlong_filter_default_is_noop():
    args = SimpleNamespace()
    data = [_group(["truncated", "truncated"])]
    assert sf.apply_sample_filters(args, data) is None
    assert [s.remove_sample for s in data[0]] == [False, False]
    assert not hasattr(args, sf.FILTER_COUNTS_ATTR) and sf.metadata_fields(args) == {}


# ---------------------------------------------------------------- shared hooks (1b-shared.patch)

import yeto.rl.engine.algorithm as _alg  # noqa: E402



def test_runtime_attrs_wired_through_spec():
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    attrs = spec.to_legacy_runtime_attrs()
    assert attrs[rp.PIPELINE_ATTR] == gk.runtime_attrs(spec)[rp.PIPELINE_ATTR]
    assert AlgorithmSpec().to_legacy_runtime_attrs() == {
        "yeto_rl_dynamic_sampling_max_replacements": None
    }


def test_translate_run_config_launch_checks():
    from test_rl_miles_adapter_config import make_config
    from yeto.rl.engine.miles_adapter import config as mc

    spec = pipeline_spec(reward_shapers=[{"name": "overlong_penalty", "max_length": 4096, "cache_length": 256}])
    with pytest.raises(mc.MilesConfigError, match="exceeds the generation limit"):
        mc.translate_run_config(make_config(), spec)
    ok = pipeline_spec(reward_shapers=[OVERLONG])
    launch = mc.translate_run_config(make_config(), ok)
    assert launch.argv[launch.argv.index("--custom-reward-post-process-path") + 1] == rp.DISPATCHER_PATH
    assert rp.read_plugins(SimpleNamespace(**launch.runtime_attrs))["algorithm_spec_sha256"] == ok.sha256()
    with pytest.raises(mc.MilesConfigError, match="multi-LoRA"):
        mc.translate_run_config(make_config(), ok, extra_argv=["--multi-lora"])
    default = mc.translate_run_config(make_config(), AlgorithmSpec())
    assert "--custom-reward-post-process-path" not in default.argv
    assert rp.PIPELINE_ATTR not in default.runtime_attrs


def test_island_check_ref_model():
    assert _alg.island_problems(kl_spec(), {"base_model_revision": "rev-a"}) == []
    problems = _alg.island_problems(kl_spec(), {"base_model_revision": "rev-b"})
    assert problems and "does not match this island's base_model_revision" in problems[0]


def _g(std, n, filtered):
    return SimpleNamespace(reward_std=std, sample_ids=tuple(range(n)), filtered_samples=filtered)


def test_overlong_gradient_rule():
    spec = AlgorithmSpec(sampling={"overlong_filter": True})
    full = SimpleNamespace(groups=(_g(0.5, 4, 4), _g(0.0, 4, 1)))
    part = SimpleNamespace(groups=(_g(0.5, 4, 3), _g(0.4, 4, 4)))
    unknown = SimpleNamespace(groups=(_g(0.5, 4, None),))
    assert gk.overlong_gradient_rule(spec, full) is False
    assert gk.overlong_gradient_rule(spec, part) is None
    assert gk.overlong_gradient_rule(spec, unknown) is None
    assert gk.overlong_gradient_rule(AlgorithmSpec(), full) is None


def test_expects_gradient_with_overlong_filter():
    spec = gk.with_pipeline_plugins(AlgorithmSpec(sampling={"overlong_filter": True}))
    assert spec.expects_gradient(SimpleNamespace(groups=(_g(0.5, 4, 4),))) is False
    assert spec.expects_gradient(SimpleNamespace(groups=(_g(0.5, 4, 3),))) is True
    assert AlgorithmSpec().expects_gradient(SimpleNamespace(groups=(_g(0.5, 4, 4),))) is True


# ---------------------------------------------------------------- 6.3 hook (1b-hook.patch)

from yeto.rl.engine.miles_adapter import rollout_meta_hook as rmh  # noqa: E402
import inspect as _inspect  # noqa: E402

needs_hook = pytest.mark.skipif("apply_sample_filters" not in _inspect.getsource(rmh.record_trained_groups),
                                reason="needs infra-drafts/1b-hook.patch")


def _hook_samples(statuses, group_index):
    return [SimpleNamespace(index=group_index * 10 + i, group_index=group_index, rollout_id=3,
                            status=SimpleNamespace(value=s), remove_sample=False, metadata=None,
                            reward=float(i % 2), response_length=5, weight_versions=None)
            for i, s in enumerate(statuses)]


@needs_hook
def test_hook_overlong_filter_and_metadata():
    spec = AlgorithmSpec(sampling={"overlong_filter": True})
    args = SimpleNamespace(**gk.runtime_attrs(spec))
    data = [_hook_samples(["completed", "truncated"], 0), _hook_samples(["truncated", "truncated"], 1)]
    rmh.record_trained_groups(args, data)
    meta = rmh.build_metadata(args, data)
    assert meta["filtered_samples"] == {"overlong_filter": 3}
    assert [g["filtered_samples"] for g in meta["groups"]] == [1, 2]
    assert [s.remove_sample for g in data for s in g] == [False, True, True, True]


@needs_hook
def test_hook_default_unchanged():
    args = SimpleNamespace()
    data = [_hook_samples(["completed", "truncated"], 0)]
    rmh.record_trained_groups(args, data)
    meta = rmh.build_metadata(args, data)
    assert "filtered_samples" not in meta and "filtered_samples" not in meta["groups"][0]
    assert [s.remove_sample for s in data[0]] == [False, False]
    assert set(meta) == {"schema", "rollout_id", "groups", "completed", "filtered", "aborted",
                         "trained_sample_indices"}


# ---------------------------------------------------------------- 3.2 learner (before outer sync)


def test_two_islands_ref_model_checked_before_joining(tmp_path):
    """Island B's base revision differs from the spec's KL reference: B fails in
    verify_ports_algorithm (called before any bridge, P0 test_learner_checks_before_the_bridge)."""

    from test_rl_miles_adapter_config import make_config
    from yeto.rl import learner as rl_learner
    from yeto.rl.engine.miles_adapter import config as mc

    spec = kl_spec()
    launch = mc.translate_run_config(make_config(), spec)
    results = {}
    for island, revision in ((0, "rev-a"), (1, "rev-b")):
        tape = tmp_path / f"tape{island}.jsonl"
        args = SimpleNamespace(rl_expected_algorithm_sha256=spec.sha256(), event_tape=str(tape),
                               learner_id=island, rl_allow_unverified_mechanism=None,
                               model_revision=revision)
        try:
            rl_learner.verify_ports_algorithm(args, SimpleNamespace(), launch)
            results[island] = "joined"
        except rl_learner.AlgorithmMismatchError as exc:
            results[island] = str(exc)
        results[f"tape{island}"] = ([json.loads(x) for x in tape.read_text().splitlines()]
                                    if tape.exists() else [])
    assert results[0] == "joined" and results["tape0"] == []
    assert "base_model_revision 'rev-b'" in results[1]
    [event] = results["tape1"]
    assert event["event"] == "rl_algorithm_island_rejected" and event["island_id"] == 1
    # a different reference revision is a different algorithm hash (outer-sync identity)
    assert kl_spec(revision="rev-b").sha256() != spec.sha256()


# ---------------------------------------------------------------- 7.1 examples

from pathlib import Path  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "rl_algorithms"


@pytest.mark.parametrize("name", ["dapo-like", "dr-grpo"])
def test_examples_build_and_translate(name):
    from test_rl_miles_adapter_config import make_config, sub
    from yeto.rl.engine.miles_adapter import config as mc

    spec = AlgorithmSpec.from_json_file(str(EXAMPLES / f"{name}.json"))
    # plugin hashes are the current sources (a dispatcher/reducer edit must
    # regenerate the examples: the hash is the plugin identity)
    spec.verify_plugins()
    assert spec.rejections() == []
    cfg = make_config()
    if spec.sampling.over_sampling_batch_size is not None:
        cfg = sub(cfg, "batch", over_sampling_batch_size=spec.sampling.over_sampling_batch_size,
                  groups_per_round=4)
    launch = mc.translate_run_config(cfg, spec)
    assert launch.algorithm_sha256 == spec.sha256()
    assert set(algorithm_argv(spec)) <= set(launch.argv)
    # default GRPO argv unchanged by the examples' existence
    assert "--custom-reward-post-process-path" not in mc.translate_run_config(make_config(), AlgorithmSpec()).argv


# ---------------------------------------------------------------- review F1-F4


def _foreign(args, samples, rewards, groups, params):
    return list(rewards)


def test_other_module_stages_enter_identity():
    """A transform registered by another yeto module is covered (F1)."""

    before = rp.stage_plugin_paths()
    base = pipeline_spec(reward_shapers=[OVERLONG])
    _foreign.__module__ = "yeto.rl.algos.reducers"  # stands in for e.g. yeto.rl.algos.seq_adv
    rp.register_advantage_transform("test_foreign", _foreign)
    try:
        paths = rp.stage_plugin_paths()
        assert "yeto.rl.algos.reducers._foreign" in paths and set(before) < set(paths)
        assert paths == tuple(sorted(paths, key=lambda p: p.rpartition(".")[0]))
        # a spec pinned before the new module registered is now incomplete
        assert any("plugins must list" in p for p in base.rejections())
        spec = pipeline_spec(transform="test_foreign")
        assert spec.rejections() == []
        assert "yeto.rl.algos.reducers._foreign" in {p.path for p in spec.plugins}
        cfg = gk.plugins_config(spec)["reward_pipeline"]
        assert cfg["pipeline_sha256"] == rp.combined_sha256(rp.stage_plugin_refs())
    finally:
        rp.ADV_TRANSFORMS.pop("test_foreign")


def test_post_process_rechecks_pipeline_identity():
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    payload = gk.runtime_attrs(spec)[rp.PIPELINE_ATTR]
    cfg = json.loads(json.dumps(payload["config"]))
    cfg["reward_pipeline"]["pipeline_sha256"] = "0" * 64
    args = _args(AlgorithmSpec())
    setattr(args, rp.PIPELINE_ATTR, {"config": cfg, "sha256": rp.canonical_sha256(cfg)})
    with pytest.raises(rp.RewardPipelineError, match="code identity"):
        rp.post_process(args, [])


def test_stale_plugin_hash_rejected():
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    stale = spec.replace(plugins=[{"path": p.path, "sha256": "1" * 64} for p in spec.plugins])
    assert any("source hash differs" in p for p in stale.rejections())


def test_unknown_transform_refused_at_launch():
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    rp.register_advantage_transform("test_gone", _foreign)
    try:
        gone = pipeline_spec(transform="test_gone")
    finally:
        rp.ADV_TRANSFORMS.pop("test_gone")
    assert gk.launch_problems(gone, rollout_batch_size=1, rollout_max_response_len=1024) == [
        "advantage transform 'test_gone' is not registered"
    ]
    assert gk.launch_problems(spec, rollout_batch_size=1, rollout_max_response_len=1024) == []


def test_plugins_bound_to_expected_algorithm_hash():
    spec = pipeline_spec(reward_shapers=[OVERLONG])
    attrs = gk.runtime_attrs(spec)
    ok = SimpleNamespace(yeto_rl_expected_algorithm_sha256=spec.sha256().upper(), **attrs)
    assert rp.read_plugins(ok)["algorithm_spec_sha256"] == spec.sha256()
    bad = SimpleNamespace(yeto_rl_expected_algorithm_sha256="f" * 64, **attrs)
    with pytest.raises(rp.RewardPipelineError, match="expects ffff"):
        rp.read_plugins(bad)


def test_sample_filters_fail_closed_and_hash_checked():
    data = [_group(["truncated"])]
    lost = SimpleNamespace(yeto_rl_expected_algorithm_sha256="a" * 64)
    with pytest.raises(rp.RewardPipelineError, match="did not reach"):
        sf.apply_sample_filters(lost, data)
    # runtime attrs delivered, no plugins payload: the spec selects no filter
    plain = SimpleNamespace(yeto_rl_expected_algorithm_sha256="a" * 64,
                            yeto_rl_dynamic_sampling_max_replacements=None)
    assert sf.apply_sample_filters(plain, data) is None
    spec = gk.with_pipeline_plugins(AlgorithmSpec(sampling={"overlong_filter": True}))
    assert spec.rejections() == []
    assert rp.SAMPLE_FILTERS_PATH in {p.path for p in spec.plugins}
    cfg = json.loads(json.dumps(gk.runtime_attrs(spec)[rp.PIPELINE_ATTR]["config"]))
    cfg["sample_filters_sha256"] = "0" * 64
    tampered = SimpleNamespace(**{rp.PIPELINE_ATTR: {"config": cfg, "sha256": rp.canonical_sha256(cfg)}})
    with pytest.raises(rp.RewardPipelineError, match="sample filters source"):
        sf.apply_sample_filters(tampered, data)
    unpinned = AlgorithmSpec(sampling={"overlong_filter": True})
    assert any("plugins must list" in p for p in unpinned.rejections())


# ---------------------------------------------------------------- 3.3 --ref-load binding


def test_ref_source_bound_to_base_model():
    spec = kl_spec()
    assert _alg.island_problems(spec, {"base_model_revision": "REV-A", "base_model": REF["source"]}) == []
    wrong = _alg.island_problems(spec, {"base_model_revision": "rev-a", "base_model": "Qwen/Other"})
    assert wrong and "kl.ref_model.source 'Qwen/Qwen3-0.6B' does not match" in wrong[0]
    override = _alg.island_problems(spec, {"base_model_revision": "rev-a", "base_model": REF["source"],
                                           "ref_load_override": "/ckpt/megatron"})
    assert override and "--megatron-ref-load" in override[0]
    # no KL loss: nothing to bind
    assert _alg.island_problems(AlgorithmSpec(), {"base_model_revision": "x", "base_model": "y",
                                                  "ref_load_override": "/z"}) == []


# ---------------------------------------------------------------- 3.2 fake composition root, outer sync

import torch  # noqa: E402


def _kl_island(tmp_path, spec, syncer, joins, *, learner_id, model_revision):
    """Ports island start order (learner.run_miles): verify_ports_algorithm, then
    the driver with strict-avg outer sync (the bridge joins in sync.start)."""

    import test_rl_engine_driver as td
    from yeto.rl import learner as rl_learner
    from yeto.rl.engine.bridges import StrictAvgSync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import fake_capabilities
    from yeto.rl.engine.miles_adapter import config as mc
    from test_rl_miles_adapter_config import make_config

    args = SimpleNamespace(rl_expected_algorithm_sha256=spec.sha256(), learner_id=learner_id,
                           event_tape=str(tmp_path / f"learner-{learner_id}.jsonl"),
                           rl_allow_unverified_mechanism=None, model_revision=model_revision,
                           model=REF["source"])
    rl_learner.verify_ports_algorithm(args, SimpleNamespace(), mc.translate_run_config(make_config(), spec))
    engine = td._engine(torch.tensor([1.0 + learner_id, 3.0]))

    def join(_bridge):
        joins.append(learner_id)
        return syncer.client(learner_id)

    sync = StrictAvgSync(td._strict_config(tmp_path, engine, learner_id=learner_id, rounds=1,
                                           tape=f"island-{learner_id}.jsonl"),
                         client_factory=join)
    caps = fake_capabilities(kl_placements={"none", "reward", "loss"},
                             features={"kl_loss_ref_model"})
    return IslandDriver(learner_id=learner_id, rollout=engine.rollout, trainer=engine.trainer,
                        policy_state=engine.policy_state, publisher=engine.publisher,
                        placement=engine.placement, algorithm=spec, sync=sync,
                        events=EventTape(tmp_path / f"island-{learner_id}.jsonl", learner_id),
                        capabilities=caps)


def test_fake_two_islands_ref_mismatch_fails_before_outer_sync(tmp_path):
    import test_rl_engine_driver as td
    from yeto.rl import learner as rl_learner

    spec = kl_spec(revision=td.MODEL_REVISION)
    engine = td._engine()
    syncer = td._strict_syncer(engine, learners=2, rounds=1)
    joins: list[int] = []
    # same reference on both islands: both join and finish one strict-avg round
    ok = [_kl_island(tmp_path, spec, syncer, joins, learner_id=i, model_revision=td.MODEL_REVISION)
          for i in (0, 1)]
    results, errors = td._run_threads(ok)
    assert errors == {} and sorted(joins) == [0, 1]
    assert torch.equal(results[0].tensors[td.NAME], results[1].tensors[td.NAME])
    # island 1's base differs from the spec's reference: refused before its bridge joins
    joins.clear()
    syncer = td._strict_syncer(engine, learners=2, rounds=1)
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(rl_learner.AlgorithmMismatchError, match="base_model_revision"):
        _kl_island(bad, spec, syncer, joins, learner_id=1, model_revision="0" * 40)
    assert joins == []
    [event] = [json.loads(x) for x in (bad / "learner-1.jsonl").read_text().splitlines()]
    assert event["event"] == "rl_algorithm_island_rejected"


# ---------------------------------------------------------------- 6.5 fake driver, zero-gradient invariant

import dataclasses as _dc  # noqa: E402

from yeto.rl.engine.ports import GroupMetadata  # noqa: E402


@_dc.dataclass(frozen=True)
class _FilteredGroup(GroupMetadata):
    # GroupMetadata.filtered_samples arrives with 1b-hook.patch; until then the
    # fake rollout supplies it through this subclass (same field name/type).
    filtered_samples: int | None = None


def _overlong_driver(tmp_path, filtered_per_group, *, grad_nan=False):
    import test_rl_engine_driver as td
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import fake_capabilities

    engine = td._engine(zero_grad_rounds={0})
    original = engine.rollout.generate

    def generate(rollout_id):
        batch = original(rollout_id)
        cls = GroupMetadata if "filtered_samples" in {f.name for f in _dc.fields(GroupMetadata)} \
            else _FilteredGroup
        groups = tuple(cls(**{f.name: getattr(g, f.name) for f in _dc.fields(g)
                              if f.name != "filtered_samples"},
                                      filtered_samples=(len(g.sample_ids) if filtered_per_group == "all"
                                                        else filtered_per_group))
                       for g in batch.groups)
        return _dc.replace(batch, groups=groups)

    engine.rollout.generate = generate
    if grad_nan:
        metrics = engine.trainer.step_metrics
        engine.trainer.step_metrics = lambda: _dc.replace(metrics(), grad_norm=float("nan"))
    spec = gk.with_pipeline_plugins(AlgorithmSpec(sampling={"overlong_filter": True}))
    caps = fake_capabilities(features={"overlong_filter", "plugins"})
    return IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                        policy_state=engine.policy_state, publisher=engine.publisher,
                        placement=engine.placement, algorithm=spec, sync=LocalOnlySync(1),
                        events=EventTape(tmp_path / "events.jsonl", 0), capabilities=caps)


def test_overlong_all_truncated_zero_grad_does_not_fail(tmp_path):
    _overlong_driver(tmp_path, "all").run()


def test_overlong_partial_truncated_zero_grad_still_fails(tmp_path):
    from yeto.rl.engine.driver import StrictRlInvariantError

    with pytest.raises(StrictRlInvariantError, match="adapter gradients are not flowing"):
        _overlong_driver(tmp_path, 1).run()


def test_overlong_nonfinite_grad_still_fails(tmp_path):
    from yeto.rl.engine.driver import StrictRlInvariantError

    with pytest.raises(StrictRlInvariantError, match="grad_norm=nan"):
        _overlong_driver(tmp_path, "all", grad_nan=True).run()


def test_learner_binds_ref_source_and_override(tmp_path):
    """3.3: the learner passes --model / --megatron-ref-load to the island check."""

    from test_rl_miles_adapter_config import make_config
    from yeto.rl import learner as rl_learner
    from yeto.rl.engine.miles_adapter import config as mc

    spec = kl_spec()
    launch = mc.translate_run_config(make_config(), spec)

    def verify(**kw):
        args = SimpleNamespace(rl_expected_algorithm_sha256=spec.sha256(), learner_id=0,
                               event_tape=str(tmp_path / "t.jsonl"), rl_allow_unverified_mechanism=None,
                               model_revision="REV-A", **kw)
        rl_learner.verify_ports_algorithm(args, SimpleNamespace(), launch)

    verify(model=REF["source"], megatron_ref_load=None)
    with pytest.raises(rl_learner.AlgorithmMismatchError, match="kl.ref_model.source"):
        verify(model="Qwen/Other", megatron_ref_load=None)
    with pytest.raises(rl_learner.AlgorithmMismatchError, match="--megatron-ref-load"):
        verify(model=REF["source"], megatron_ref_load="/ckpt/megatron")


# ---------------------------------------------------------------- 8.3 declarations (1b-declare.patch)

EXPECTED_1B_DECLARED = {
    "loss_aggregations": {"constant"},
    "features": {"kl_loss_ref_model", "no_grpo_std_normalization", "entropy_bonus", "overlong_penalty",
                 "eps_clip"},
    "kl_placements": {"loss"},
    "reward_postprocessors": {"custom_reward_postprocess"},
}


def test_declared_table_matches_final_declaration():
    """G1_DECLARED == the 1b part of ALGO-CAP's final MILES_DECLARED (integ-decl d6be0f1);
    custom_pg_loss_reducer is declared there separately, limited to the Dr.GRPO reducer."""

    assert {d: set(n) for d, n in gk.declared_mechanisms().items()} == EXPECTED_1B_DECLARED
    from yeto.rl.engine.miles_adapter import entry

    final = getattr(entry, "MILES_DECLARED", None)
    if final is not None:  # integrated branches: the 1b entries must be exactly there
        ours = {f"{d}:{n}" for d, names in EXPECTED_1B_DECLARED.items() for n in names}
        assert ours <= set(final)


def test_declared_caps_accept_1b_and_refuse_undeclared():
    from yeto.rl.engine.miles_adapter.entry import miles_capabilities

    caps = gk.merge_declared(miles_capabilities("sha256:" + "0" * 64))
    for spec in (pipeline_spec(reward_shapers=[OVERLONG]), AlgorithmSpec(loss={"eps_clip": 0.2}),
                 kl_spec(), AlgorithmSpec(entropy_coef=0.001)):
        missing = [f"{d}:{n}" for d, n in spec.required_mechanisms() if n not in getattr(caps, d)]
        assert missing == [], missing
    assert "token" not in gk.declared_mechanisms().get("loss_aggregations", frozenset())
    for name in ("clip_higher", "dual_clip", "over_sampling", "overlong_filter"):
        assert name not in gk.declared_mechanisms().get("features", frozenset())
    # token withdrawn after review (grad_norm identical to the baseline)
    assert "token" not in gk.declared_mechanisms().get("loss_aggregations", frozenset())


def test_only_the_drgrpo_reducer_is_accepted():
    other = PluginRef.from_path("yeto.rl.algos.reducers.configured_denominator").to_dict()
    for loss in ({"reducer": other}, {"aggregation": "constant", "constant_denominator": 10, "reducer": other}):
        assert any("the only pg_loss reducer on the ports path" in p
                   for p in AlgorithmSpec(loss=loss).rejections())
    assert constant_spec().rejections() == []


def test_emit_event_echoes_to_stdout(capsys):
    from yeto.rl import event_echo

    rp.emit_event(SimpleNamespace(yeto_rl_learner_id=0), {"event": "rl_reward_shaping", "samples": 2})
    out = capsys.readouterr().out.strip().splitlines()
    assert out and out[-1].startswith(event_echo.PREFIX)
    raw = event_echo.parse_line(out[-1])
    assert raw not in (None, event_echo.INVALID)
    record = json.loads(raw) if isinstance(raw, str) else raw
    assert record["event"] == "rl_reward_shaping" and record["island_id"] == 0 and record["samples"] == 2
