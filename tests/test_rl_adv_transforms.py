"""MaxRL / MAPO / GDPO advantage transforms (rl-algo-seq-and-adv 4.1, 4.2, 4.4, 5.2, 5.3, 5.4).

CPU only. References are independent float64 implementations of the paper
formulas (design D5/D6), compared element-wise with ``pytest.approx``
(rel 1e-5, abs 1e-6: the transforms compute in float32 exactly like Miles'
GRPO normalization).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import statistics
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.algos import reward_pipeline as rp  # noqa: E402
from yeto.rl.algos import seq_adv as sa  # noqa: E402
from yeto.rl.engine.algorithm import AlgorithmSpec, load_extensions  # noqa: E402

load_extensions()
APPROX = dict(rel=1e-5, abs=1e-6)


class _Sample(SimpleNamespace):
    def get_reward_value(self, args):
        return self.reward


def _s(reward, group, index=None, rollout=None, components=None, length=10):
    metadata = {} if components is None else {sa.REWARD_COMPONENTS_KEY: dict(components)}
    return _Sample(reward=reward, group_index=group, index=index, rollout_id=rollout,
                   metadata=metadata, response_length=length)


def _spec(transform, gdpo=None, estimator="grpo"):
    advantage = {"estimator": estimator, "transform": transform, "reward_binary": True,
                 "reward_postprocess": rp.dispatcher_ref().to_dict()}
    if gdpo is not None:
        advantage["gdpo"] = gdpo
    from yeto.rl.algos import grpo_knobs

    return grpo_knobs.with_pipeline_plugins(AlgorithmSpec(advantage=advantage))


def _args(spec, **extra):
    from yeto.rl.algos import grpo_knobs

    args = SimpleNamespace(advantage_estimator=spec.advantage.estimator,
                           rewards_normalization=True, grpo_std_normalization=True,
                           n_samples_per_prompt=4, rollout_batch_size=2, multi_lora=False,
                           **extra)
    for key, value in {**grpo_knobs.runtime_attrs(spec), **sa.runtime_attrs(spec)}.items():
        setattr(args, key, value)
    return args


def _run(transform, samples, gdpo=None):
    spec = _spec(transform, gdpo)
    assert spec.rejections() == []
    return rp.post_process(_args(spec), samples)[1]


# -- independent references (float64, paper formulas) -------------------------


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def ref_maxrl(rs):
    mu = sum(rs) / len(rs)
    if len(rs) == 1 or mu == 0:
        return [0.0] * len(rs)
    return [(r - mu) / mu for r in rs]


def ref_grpo(rs):
    mu = sum(rs) / len(rs)
    sd = _std(rs)
    return [(r - mu) / (sd + 1e-6) if sd > 0 else r - mu for r in rs]


def ref_mapo(rs):
    if len(rs) == 1:
        return [0.0]
    mu = sum(rs) / len(rs)
    lam = 1 - 4 * mu * (1 - mu)
    first = ref_grpo(rs)
    second = [(r - mu) / mu if mu > 0 else 0.0 for r in rs]
    return [(1 - lam) * a + lam * b for a, b in zip(first, second)]


def ref_gdpo(groups, weights):
    """groups: list of list of {name: value}; returns per-entry advantages."""

    names = sorted(weights)
    combined = []
    for entries in groups:
        per = [0.0] * len(entries)
        for name in names:
            normed = ref_grpo([e[name] for e in entries])
            per = [p + weights[name] * v for p, v in zip(per, normed)]
        combined.append(per)
    flat = [v for g in combined for v in g]
    white = ref_grpo(flat)
    out, i = [], 0
    for g in combined:
        out.append(white[i:i + len(g)])
        i += len(g)
    return out


# -- 4.2 MaxRL / MAPO numerics ------------------------------------------------

BINARY_GROUPS = {
    "g1": [[1.0]],
    "all_wrong": [[0.0, 0.0, 0.0, 0.0]],
    "all_right": [[1.0, 1.0, 1.0, 1.0]],
    "one_zero_zero_one": [[1.0, 0.0, 0.0, 1.0]],
    "mixed": [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 1.0]],
}


def _flat_batch(groups):
    samples, i = [], 0
    for g, rs in enumerate(groups):
        for r in rs:
            samples.append(_s(r, g, index=i))
            i += 1
    return samples


@pytest.mark.parametrize("case", sorted(BINARY_GROUPS))
@pytest.mark.parametrize("transform,ref", [("maxrl", ref_maxrl), ("mapo", ref_mapo)])
def test_binary_transforms_match_reference(case, transform, ref):
    groups = BINARY_GROUPS[case]
    out = _run(transform, _flat_batch(groups))
    expected = [v for rs in groups for v in ref(rs)]
    assert out == pytest.approx(expected, **APPROX)
    assert all(math.isfinite(v) for v in out)


def test_maxrl_spec_example():
    assert _run("maxrl", _flat_batch([[1.0, 0.0, 0.0, 1.0]])) == [1.0, -1.0, -1.0, 1.0]


def test_all_wrong_and_all_right_are_zero():
    for transform in ("maxrl", "mapo"):
        for rs in ([0.0] * 4, [1.0] * 4, [1.0], [0.0]):
            assert _run(transform, _flat_batch([rs])) == [0.0] * len(rs)


def test_mapo_p_half_is_grpo_exactly():
    rs = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    out = _run("mapo", _flat_batch([rs]))
    grpo = rp.grpo_default(_args(_spec("mapo")), _flat_batch([rs]), rs, [list(range(6))], {})
    assert out == grpo


def _multi_segment():
    # rollout 7: three segments of one rollout (reward 1); 8 and 9 single; group 0.
    return [
        _s(1.0, 0, index=0, rollout=7, components={"a": 1.0, "b": 0.0}),
        _s(1.0, 0, index=1, rollout=7, components={"a": 1.0, "b": 0.0}),
        _s(1.0, 0, index=2, rollout=7, components={"a": 1.0, "b": 0.0}),
        _s(0.0, 0, index=3, rollout=8, components={"a": 0.0, "b": 1.0}),
        _s(0.0, 0, index=4, rollout=9, components={"a": 0.0, "b": 0.0}),
        _s(1.0, 1, index=5, rollout=10, components={"a": 1.0, "b": 1.0}),
        _s(0.0, 1, index=6, rollout=11, components={"a": 0.0, "b": 1.0}),
    ]


@pytest.mark.parametrize("transform,ref", [("maxrl", ref_maxrl), ("mapo", ref_mapo)])
def test_multi_segment_rollout_counts_once_and_shares(transform, ref):
    out = _run(transform, _multi_segment())
    g0, g1 = ref([1.0, 0.0, 0.0]), ref([1.0, 0.0])
    assert out[0] == out[1] == out[2]
    assert out == pytest.approx([g0[0]] * 3 + g0[1:] + g1, **APPROX)


@pytest.mark.parametrize("transform", ["maxrl", "mapo"])
def test_inconsistent_segments_fail(transform):
    bad = [_s(1.0, 0, index=0, rollout=3), _s(0.0, 0, index=1, rollout=3),
           _s(0.0, 0, index=2, rollout=4)]
    with pytest.raises(ValueError, match=r"rollout 3 must share one reward.*\[1.0, 0.0\]"):
        _run(transform, bad)


# -- 4.1 runtime binary check ----------------------------------------------------


@pytest.mark.parametrize("transform", ["maxrl", "mapo"])
def test_non_binary_reward_fails_round(transform):
    with pytest.raises(sa.AdvantageTransformError, match="requires binary.*sample 1: 0.5"):
        _run(transform, _flat_batch([[1.0, 0.5, 0.0, 1.0]]))


def test_transform_event_reports_group_counts(caplog):
    with caplog.at_level("WARNING"):
        _run("maxrl", _flat_batch([[0.0] * 4, [1.0] * 4, [1.0, 0.0, 0.0, 0.0]]))
    line = next(r.getMessage() for r in caplog.records if sa.TRANSFORM_EVENT in r.getMessage())
    event = json.loads(line)
    assert event["all_zero_groups"] == 1 and event["all_one_groups"] == 1
    assert event["groups"] == 3 and event["nonzero_advantages"] == 4


# -- 4.4 zero-gradient equivalence (exhaustive, G <= 8) --------------------------


@pytest.mark.parametrize("fn", [sa.maxrl_values, sa.mapo_values])
def test_nonzero_std_iff_nonzero_output(fn):
    for size in range(1, 9):
        for combo in itertools.product((0.0, 1.0), repeat=size):
            values = torch.tensor(combo)
            std_positive = size > 1 and float(values.std()) > 0
            out = fn(list(combo))
            assert torch.isfinite(out).all()
            assert std_positive == bool((out != 0).any()), (fn.__name__, combo)


# -- 5.2 / 5.3 GDPO ----------------------------------------------------------------

GDPO = {"components": [{"name": "correctness", "weight": 1.0},
                       {"name": "format", "weight": 0.5}], "whiten": True}
W = {"correctness": 1.0, "format": 0.5}


def _gdpo_batch(groups):
    samples, i = [], 0
    for g, entries in enumerate(groups):
        for comps in entries:
            samples.append(_s(comps["correctness"], g, index=i, components=comps))
            i += 1
    return samples


GDPO_CASES = {
    "g1": [[{"correctness": 1.0, "format": 0.0}], [{"correctness": 0.0, "format": 1.0}]],
    "component_constant_in_group": [
        [{"correctness": 1.0, "format": 1.0}, {"correctness": 0.0, "format": 1.0},
         {"correctness": 1.0, "format": 1.0}],
        [{"correctness": 0.0, "format": 0.0}, {"correctness": 0.0, "format": 1.0}],
    ],
    "non_binary": [
        [{"correctness": 0.3, "format": 0.9}, {"correctness": 0.7, "format": 0.1},
         {"correctness": 0.5, "format": 0.5}, {"correctness": 0.1, "format": 0.2}],
    ],
}


@pytest.mark.parametrize("case", sorted(GDPO_CASES))
def test_gdpo_matches_reference(case):
    groups = GDPO_CASES[case]
    out = _run("gdpo", _gdpo_batch(groups), gdpo=GDPO)
    expected = [v for g in ref_gdpo(groups, W) for v in g]
    assert out == pytest.approx(expected, **APPROX)
    assert all(math.isfinite(v) for v in out)


def test_gdpo_constant_component_contributes_zero():
    group = GDPO_CASES["component_constant_in_group"][:1]  # format == 1 in every rollout
    zero_format = {"components": [{"name": "correctness", "weight": 1.0},
                                  {"name": "format", "weight": 0.0}], "whiten": True}
    assert _run("gdpo", _gdpo_batch(group), gdpo=GDPO) == \
        _run("gdpo", _gdpo_batch(group), gdpo=zero_format)


def test_gdpo_whole_batch_constant_only_subtracts_mean():
    groups = [[{"correctness": 1.0, "format": 1.0}] * 3, [{"correctness": 0.0, "format": 1.0}] * 2]
    out = _run("gdpo", _gdpo_batch(groups), gdpo=GDPO)
    assert out == [0.0] * 5


def test_gdpo_multi_segment():
    samples = _multi_segment()
    gd = {"components": [{"name": "a", "weight": 1.0}, {"name": "b", "weight": 2.0}]}
    out = _run("gdpo", samples, gdpo=gd)
    groups = [[{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, {"a": 0.0, "b": 0.0}],
              [{"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 1.0}]]
    ref = ref_gdpo(groups, {"a": 1.0, "b": 2.0})
    assert out[0] == out[1] == out[2]
    assert out == pytest.approx([ref[0][0]] * 3 + ref[0][1:] + ref[1], **APPROX)


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s.metadata.clear(), "sample 1 .*metadata\\['yeto_reward_components'\\] is missing"),
    (lambda s: s.metadata[sa.REWARD_COMPONENTS_KEY].pop("format"),
     "sample 1 .*lacks components \\['format'\\]"),
    (lambda s: s.metadata[sa.REWARD_COMPONENTS_KEY].update(length=1.0),
     "sample 1 .*undeclared components \\['length'\\]"),
    (lambda s: s.metadata[sa.REWARD_COMPONENTS_KEY].update(format=math.nan),
     "sample 1 .*'format' must be a finite number"),
    (lambda s: s.metadata[sa.REWARD_COMPONENTS_KEY].update(format=True),
     "sample 1 .*'format' must be a finite number"),
])
def test_gdpo_reward_vector_failures(mutate, match):
    samples = _gdpo_batch([[{"correctness": 1.0, "format": 0.0}, {"correctness": 0.0, "format": 1.0}]])
    mutate(samples[1])
    with pytest.raises(sa.AdvantageTransformError, match=match):
        _run("gdpo", samples, gdpo=GDPO)


def test_gdpo_segments_must_share_vector():
    samples = [_s(1.0, 0, index=0, rollout=3, components={"correctness": 1.0, "format": 1.0}),
               _s(1.0, 0, index=1, rollout=3, components={"correctness": 1.0, "format": 0.0}),
               _s(0.0, 0, index=2, rollout=4, components={"correctness": 0.0, "format": 0.0})]
    with pytest.raises(sa.AdvantageTransformError, match="rollout 3 must share one reward vector"):
        _run("gdpo", samples, gdpo=GDPO)


def test_gdpo_config_hash_checked():
    spec = _spec("gdpo", GDPO)
    args = _args(spec)
    payload = getattr(args, sa.SEQ_ADV_ATTR)
    payload["config"]["gdpo"]["components"][0]["weight"] = 9.0
    with pytest.raises(sa.AdvantageTransformError, match="hash mismatch"):
        rp.post_process(args, _gdpo_batch([[{"correctness": 1.0, "format": 0.0}]]))
    missing = _args(spec)
    delattr(missing, sa.SEQ_ADV_ATTR)
    with pytest.raises(sa.AdvantageTransformError, match="is missing"):
        rp.post_process(missing, _gdpo_batch([[{"correctness": 1.0, "format": 0.0}]]))


# -- 5.4 example reward function ----------------------------------------------------


def test_example_reward_writes_accepted_components(monkeypatch):
    from yeto.rl.algos import gdpo_reward

    monkeypatch.setattr(gdpo_reward, "_correct", lambda response, label: response.endswith("\\boxed{4}"))
    samples = []
    for i, response in enumerate(["so \\boxed{4}", "so \\boxed{5}", "no box", "<think>\\boxed{4}</think> x"]):
        sample = _Sample(response=response, label="4", metadata=None, group_index=0, index=i,
                         rollout_id=None, reward=None)
        sample.reward = asyncio.run(gdpo_reward.reward_func(None, sample))
        samples.append(sample)
    comps = [s.metadata[sa.REWARD_COMPONENTS_KEY] for s in samples]
    assert comps == [{"correctness": 1.0, "format": 1.0}, {"correctness": 0.0, "format": 1.0},
                     {"correctness": 0.0, "format": 0.0}, {"correctness": 0.0, "format": 0.0}]
    assert [s.reward for s in samples] == [1.0, 0.0, 0.0, 0.0]
    spec_gdpo = {"components": [{"name": n, "weight": 1.0} for n in gdpo_reward.COMPONENTS]}
    out = _run("gdpo", samples, gdpo=spec_gdpo)
    assert len(out) == 4 and all(math.isfinite(v) for v in out)


def test_gsm8k_style_label_and_binary_reward(monkeypatch):
    from yeto.rl.algos import gdpo_reward

    seen = []
    monkeypatch.setattr("yeto.rl.math_reward.score",
                        lambda response, truth: seen.append(truth) or float(truth == "1234"))
    sample = _Sample(response="so \\boxed{1234}", label="work ... #### 1,234", metadata=None)
    assert asyncio.run(gdpo_reward.correctness_reward(None, sample)) == 1.0
    assert seen == ["1234"]
