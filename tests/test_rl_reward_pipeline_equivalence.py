"""Dispatcher == Miles built-in ``_post_process_rewards`` (rl-algo-grpo-knobs 5.2).

Runs where upstream Miles imports (``/home/michael/work/miles-next-venv`` with
``PYTHONPATH=<yeto>:/home/michael/work/miles-next``); skipped otherwise.
Element-wise comparison with ``torch.equal`` (not approximate).
"""

from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace

import pytest

tdc = pytest.importorskip("miles.ray.rollout.train_data_conversion")
torch = pytest.importorskip("torch")
from miles.utils.types import Sample  # noqa: E402

from yeto.rl.algos import grpo_knobs, reward_pipeline as rp  # noqa: E402
from yeto.rl.engine.algorithm import AlgorithmSpec  # noqa: E402

# sha256 of miles/ray/rollout/train_data_conversion.py at MILES_NEXT_COMMIT
# 03947150 (identical at 9e4260d). A Miles upgrade that touches the built-in
# reward post-processing fails here: re-review grpo_default against it.
TRAIN_DATA_CONVERSION_SHA256 = "ff7448c017c01ce18533fc986c571aa69f7e438ac02fdfaba95d38aa5b72296e"


def test_upstream_source_pinned():
    with open(inspect.getsourcefile(tdc), "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert digest == TRAIN_DATA_CONVERSION_SHA256, (
        "Miles train_data_conversion.py changed: re-review "
        "yeto.rl.algos.reward_pipeline.grpo_default against _post_process_rewards"
    )


def _args(estimator="grpo", *, rewards_normalization=True, std=True, n=4, batch=2):
    spec = AlgorithmSpec(
        advantage={"estimator": "grpo", "reward_shapers": [], "transform": "grpo_default"}
    )
    args = SimpleNamespace(
        advantage_estimator=estimator,
        rewards_normalization=rewards_normalization,
        grpo_std_normalization=std,
        n_samples_per_prompt=n,
        rollout_batch_size=batch,
        reward_key=None,
        multi_lora=False,
    )
    # The dispatcher only runs with a pipeline section; grpo_default + no shaper
    # is the "identity" configuration compared with the built-in path.
    setattr(args, rp.PIPELINE_ATTR, rp.plugins_payload({
        "reward_pipeline": {"reward_shapers": [], "advantage_transform": "grpo_default",
                            "advantage_params": {}, "pipeline_sha256": rp.pipeline_sha256()},
        "algorithm_spec_sha256": spec.sha256(),
    }))
    return args


def _sample(reward, *, group=None, index=None, rollout=None, length=10):
    return Sample(group_index=group, index=index, rollout_id=rollout, reward=reward,
                  response_length=length)


def _batches():
    b = {}
    b["plain"] = [
        _sample(r, group=g, index=i) for i, (g, r) in enumerate(
            [(0, 1.0), (0, 0.0), (0, 0.0), (0, 1.0), (1, 0.5), (1, 0.25), (1, 1.0), (1, 0.0)]
        )
    ]
    # rollout 7 has three segments, rollout 8 two: one shared reward each.
    b["multi_segment"] = [
        _sample(1.0, group=0, index=0, rollout=7), _sample(1.0, group=0, index=1, rollout=7),
        _sample(1.0, group=0, index=2, rollout=7), _sample(0.0, group=0, index=3, rollout=8),
        _sample(0.0, group=0, index=4, rollout=8), _sample(0.5, group=0, index=5, rollout=9),
        _sample(0.2, group=1, index=6, rollout=10), _sample(0.9, group=1, index=7, rollout=11),
    ]
    b["g1"] = [_sample(0.7, group=0, index=0), _sample(0.3, group=1, index=1),
               _sample(1.0, group=2, index=2, rollout=5), _sample(1.0, group=2, index=3, rollout=5)]
    b["constant_group"] = [_sample(1.0, group=0, index=i) for i in range(4)] + [
        _sample(float(i % 2), group=1, index=4 + i) for i in range(4)
    ]
    # no group_index: fixed fan-out n*batch = 8 samples -> contiguous groups of 4.
    b["fanout"] = [_sample(float(v), index=i) for i, v in enumerate([1, 0, 0, 0, 1, 1, 0, 1])]
    # no group_index and not 8 samples -> whole batch one group (warning event).
    b["fallback"] = [_sample(float(v), index=i) for i, v in enumerate([1, 0, 0.5, 0, 1])]
    # no index / rollout ids at all: rows are their own rollouts.
    b["rowkeys"] = [_sample(float(v), group=0) for v in [1, 0, 0, 1]]
    b["ints"] = [_sample(v, group=i // 4, index=i) for i, v in enumerate([1, 0, 0, 0, 3, 1, 2, 2])]
    return b


CASES = [
    ("grpo", True, True),
    ("grpo", False, True),  # rewards_normalization off
    ("grpo", True, False),  # grpo_std_normalization off
    ("gspo", True, True),
    ("gspo", True, False),
    ("reinforce_plus_plus_baseline", True, True),
    ("reinforce_plus_plus", True, True),
    ("ppo", True, True),
]


@pytest.mark.parametrize("estimator,rnorm,std", CASES)
@pytest.mark.parametrize("batch", sorted(_batches()))
def test_dispatcher_equals_builtin(batch, estimator, rnorm, std, caplog):
    args = _args(estimator, rewards_normalization=rnorm, std=std)
    ours = rp.post_process(args, _batches()[batch])
    theirs = tdc._post_process_rewards(args, _batches()[batch], custom_reward_post_process_func=None)
    for mine, ref in zip(ours, theirs, strict=True):
        assert len(mine) == len(ref)
        assert torch.equal(torch.tensor(mine, dtype=torch.float64),
                           torch.tensor(ref, dtype=torch.float64))
        assert [type(x) for x in mine] == [type(x) for x in ref]


def test_fallback_warns(caplog):
    args = _args()
    with caplog.at_level("WARNING"):
        rp.post_process(args, _batches()["fallback"])
    assert rp.FALLBACK_EVENT in caplog.text and '"samples": 5' in caplog.text


def test_inconsistent_rollout_rewards_raise_in_both():
    bad = [_sample(1.0, group=0, index=0, rollout=3), _sample(0.0, group=0, index=1, rollout=3),
           _sample(0.5, group=0, index=2, rollout=4)]
    args = _args()
    with pytest.raises(ValueError, match="rollout 3 must share one reward") as mine:
        rp.post_process(args, bad)
    with pytest.raises(ValueError, match="rollout 3 must share one reward") as ref:
        tdc._post_process_rewards(args, bad, custom_reward_post_process_func=None)
    assert str(mine.value) == str(ref.value)


def test_miles_calls_dispatcher_through_custom_path():
    """The Miles hook contract: custom func replaces the built-in path verbatim."""

    args = _args()
    samples = _batches()["plain"]
    got = tdc._post_process_rewards(args, samples, custom_reward_post_process_func=rp.post_process)
    assert got == rp.post_process(args, samples)


def test_overlong_shaping_then_builtin_normalization():
    """With the overlong shaper the dispatcher == built-in normalization of shaped rewards."""

    spec = grpo_knobs.with_pipeline_plugins(AlgorithmSpec(advantage={
        "reward_postprocess": rp.dispatcher_ref().to_dict(),
        "reward_shapers": [{"name": "overlong_penalty", "max_length": 100, "cache_length": 20}],
    }))
    args = _args()
    setattr(args, rp.PIPELINE_ATTR, grpo_knobs.runtime_attrs(spec)[rp.PIPELINE_ATTR])
    lengths = [80, 90, 100, 101]
    samples = [_sample(1.0, group=0, index=i, length=n) for i, n in enumerate(lengths)]
    shaped, adv = rp.post_process(args, samples)
    assert shaped == [1.0, 0.5, 0.0, 0.0]
    reference = [_sample(r, group=0, index=i) for i, r in enumerate(shaped)]
    _, ref_adv = tdc._post_process_rewards(args, reference, custom_reward_post_process_func=None)
    assert torch.equal(torch.tensor(adv), torch.tensor(ref_adv))
    assert [s.metadata[rp.RAW_REWARD_METADATA_KEY] for s in samples] == [1.0] * 4
