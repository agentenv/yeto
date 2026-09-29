"""rl-algo-seq-and-adv against upstream Miles functions (tasks 2.4, 2.5, 3.3, 3.4, 3.5, 4.3).

Runs where upstream Miles imports::

    PYTHONPATH=<yeto>:/home/michael/work/miles-next \\
      /home/michael/work/miles-next-venv/bin/python -m pytest -q tests/test_rl_seq_adv_miles.py

and is skipped otherwise (``/tmp/yeto-venv`` has no Miles). Element-wise
comparisons use ``torch.equal`` against Miles, ``torch.allclose`` (float32
tolerance) against the hand formulas.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
mu = pytest.importorskip("miles.backends.training_utils.loss_hub.math_utils")
adv_mod = pytest.importorskip("miles.backends.training_utils.loss_hub.advantages")
tdc = pytest.importorskip("miles.ray.rollout.train_data_conversion")
arguments = pytest.importorskip("miles.utils.arguments")
from miles.utils.types import Sample  # noqa: E402

from yeto.rl.algos import grpo_knobs  # noqa: E402
from yeto.rl.algos import reward_pipeline as rp  # noqa: E402
from yeto.rl.algos import seq_adv as sa  # noqa: E402
from yeto.rl.engine.algorithm import AlgorithmSpec, load_extensions  # noqa: E402
from yeto.rl.engine.miles_adapter import algorithm_flags as af  # noqa: E402

load_extensions()


# -- 2.4 GSPO sequence ratio and clip --------------------------------------------------


def _gspo_inputs():
    torch.manual_seed(0)
    lengths = [5, 3, 7]
    old = [torch.randn(n) * 0.1 - 1.0 for n in lengths]
    new = [o + torch.randn(n) * 0.05 for o, n in zip(old, lengths)]
    masks = [torch.ones(n) for n in lengths]
    masks[2][-2:] = 0  # masked tail tokens do not enter the sequence mean
    return lengths, old, new, masks


def _hand_seq_ratio(old, new, masks):
    return [torch.exp(((n - o) * m).sum() / m.sum()) for o, n, m in zip(old, new, masks)]


def test_gspo_sequence_ratio_matches_hand_formula():
    lengths, old, new, masks = _gspo_inputs()
    kl = mu.compute_gspo_kl(new, old, new, masks)
    ratio = torch.exp(-kl)
    hand = torch.cat([r.expand(n) for r, n in zip(_hand_seq_ratio(old, new, masks), lengths)])
    assert torch.allclose(ratio, hand, rtol=1e-6, atol=1e-7)
    # every token of a sequence shares one ratio
    for chunk in ratio.split(lengths):
        assert torch.equal(chunk, chunk[0].expand_as(chunk))


@pytest.mark.parametrize("eps_low,eps_high", [(0.2, 0.2), (3e-4, 4e-4)])
def test_gspo_sequence_clip_matches_hand_formula(eps_low, eps_high):
    lengths, old, new, masks = _gspo_inputs()
    advantages = torch.cat([torch.full((n,), a) for n, a in zip(lengths, [1.0, -0.5, 2.0])])
    kl = mu.compute_gspo_kl(new, old, new, masks)
    losses, clipfrac = mu.compute_policy_loss(kl, advantages, eps_low, eps_high)
    ratio = torch.cat([r.expand(n) for r, n in zip(_hand_seq_ratio(old, new, masks), lengths)])
    unclipped = -ratio * advantages
    clipped = -ratio.clamp(1 - eps_low, 1 + eps_high) * advantages
    assert torch.allclose(losses, torch.maximum(unclipped, clipped), rtol=1e-6, atol=1e-7)
    assert torch.equal(clipfrac, (clipped > unclipped).float())
    for chunk in clipfrac.split(lengths):  # sequence-level: all or none
        assert chunk.min() == chunk.max()


def test_gspo_fully_clipped_round_has_clipfrac_one_and_zero_gradient():
    lengths = [4, 6]
    old = [torch.zeros(n) for n in lengths]
    # sequence log-ratio +0.1 (A>0) and -0.1 (A<0): both outside [1-3e-4, 1+4e-4]
    new = [torch.full((4,), 0.1, requires_grad=True), torch.full((6,), -0.1, requires_grad=True)]
    masks = [torch.ones(n) for n in lengths]
    advantages = torch.cat([torch.full((4,), 1.0), torch.full((6,), -1.0)])
    kl = mu.compute_gspo_kl(new, old, new, masks)
    losses, clipfrac = mu.compute_policy_loss(kl, advantages, 3e-4, 4e-4)
    assert torch.equal(clipfrac, torch.ones(10))
    losses.sum().backward()
    for tensor in new:
        assert torch.equal(tensor.grad, torch.zeros_like(tensor))
    # the adapter aggregate of per-mini-batch clip fractions is then exactly 1
    assert sa.clipfrac_from_losses([{"pg_clipfrac": clipfrac.mean()}] * 2, [4, 6]) == 1.0


def test_gspo_partially_clipped_has_gradient():
    old = [torch.zeros(4), torch.zeros(6)]
    new = [torch.full((4,), 0.1, requires_grad=True), torch.full((6,), 0.0, requires_grad=True)]
    advantages = torch.cat([torch.full((4,), 1.0), torch.full((6,), 1.0)])
    kl = mu.compute_gspo_kl(new, old, new, [torch.ones(4), torch.ones(6)])
    losses, clipfrac = mu.compute_policy_loss(kl, advantages, 3e-4, 4e-4)
    assert 0 < clipfrac.mean() < 1
    losses.sum().backward()
    assert torch.equal(new[0].grad, torch.zeros(4)) and new[1].grad.abs().sum() > 0


# -- 2.5 / 3.5 upstream parse_args -------------------------------------------------------

TINY_QWEN3 = {
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 64,
    "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 4096,
    "vocab_size": 1000, "rms_norm_eps": 1e-6, "rope_theta": 10000,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16", "hidden_act": "silu",
    "attention_bias": False,
}


def _parse(tmp_path, monkeypatch, spec):
    (tmp_path / "config.json").write_text(json.dumps(TINY_QWEN3))
    positioned = ["--advantage-estimator", spec.advantage_estimator]
    if spec.kl_coef is not None:
        positioned += ["--kl-coef", str(spec.kl_coef)]
    argv = [
        "train.py", "--train-backend", "fsdp", "--hf-checkpoint", str(tmp_path),
        "--ref-load", str(tmp_path), "--rollout-batch-size", "2", "--n-samples-per-prompt", "2",
        "--global-batch-size", "4", "--num-rollout", "1", "--colocate",
        "--actor-num-gpus-per-node", "1", *positioned, *af.algorithm_argv(spec),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return arguments.parse_args()


def test_gspo_explicit_clip_parses_upstream(tmp_path, monkeypatch):
    spec = AlgorithmSpec(advantage={"estimator": "gspo"},
                         loss={"eps_clip": 3e-4, "eps_clip_high": 4e-4})
    assert spec.rejections() == []
    args = _parse(tmp_path, monkeypatch, spec)
    assert args.advantage_estimator == "gspo"
    assert args.eps_clip == pytest.approx(3e-4) and args.eps_clip_high == pytest.approx(4e-4)


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_rpp_whiten_reward_kl_parses_upstream(tmp_path, monkeypatch, estimator):
    spec = AlgorithmSpec(advantage={"estimator": estimator, "whiten": True},
                         kl={"placement": "reward", "coef": 0.05})
    assert spec.rejections() == []
    args = _parse(tmp_path, monkeypatch, spec)
    assert args.advantage_estimator == estimator and args.normalize_advantages is True
    assert args.kl_coef == pytest.approx(0.05) and args.gamma == 1.0  # default not emitted


def test_mapped_gamma_parses_upstream(tmp_path, monkeypatch):
    # expressible (mapping row); the spec itself is refused as not opened
    spec = AlgorithmSpec(advantage={"estimator": "reinforce_plus_plus", "whiten": True,
                                    "gamma": 0.99})
    assert "--gamma" in af.algorithm_argv(spec)
    assert _parse(tmp_path, monkeypatch, spec).gamma == pytest.approx(0.99)


# -- 3.3 dispatcher vs Miles _post_process_rewards for rpp / rpp_baseline ------------------


def _args(estimator, transform="grpo_default", n=4, batch=2):
    spec = grpo_knobs.with_pipeline_plugins(AlgorithmSpec(
        advantage={"estimator": "grpo", "transform": "maxrl", "reward_binary": True,
                   "reward_postprocess": rp.dispatcher_ref().to_dict()}))
    args = SimpleNamespace(advantage_estimator=estimator, rewards_normalization=True,
                           grpo_std_normalization=True, n_samples_per_prompt=n,
                           rollout_batch_size=batch, reward_key=None, multi_lora=False)
    setattr(args, rp.PIPELINE_ATTR, rp.plugins_payload({
        "reward_pipeline": {"reward_shapers": [], "advantage_transform": transform,
                            "advantage_params": {}},
        "algorithm_spec_sha256": spec.sha256(),
    }))
    return args


def _sample(reward, *, group, index, rollout=None):
    return Sample(group_index=group, index=index, rollout_id=rollout, reward=reward,
                  response_length=10)


def _rpp_batches():
    return {
        "plain": [_sample(r, group=i // 4, index=i)
                  for i, r in enumerate([1.0, 0.0, 0.5, 1.0, 0.25, 0.25, 0.75, 0.0])],
        "multi_segment": [
            _sample(1.0, group=0, index=0, rollout=7), _sample(1.0, group=0, index=1, rollout=7),
            _sample(1.0, group=0, index=2, rollout=7), _sample(0.0, group=0, index=3, rollout=8),
            _sample(0.5, group=0, index=4, rollout=9), _sample(0.2, group=1, index=5, rollout=10),
            _sample(0.2, group=1, index=6, rollout=10), _sample(0.9, group=1, index=7, rollout=11),
        ],
        "g1": [_sample(0.7, group=0, index=0), _sample(0.3, group=1, index=1)],
    }


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
@pytest.mark.parametrize("batch", ["plain", "multi_segment", "g1"])
def test_rpp_dispatcher_equals_builtin(estimator, batch):
    args = _args(estimator)
    ours = rp.post_process(args, _rpp_batches()[batch])
    theirs = tdc._post_process_rewards(args, _rpp_batches()[batch],
                                       custom_reward_post_process_func=None)
    for mine, ref in zip(ours, theirs, strict=True):
        assert torch.equal(torch.tensor(mine, dtype=torch.float64),
                           torch.tensor(ref, dtype=torch.float64))
    rewards = [s.reward for s in _rpp_batches()[batch]]
    if estimator == "reinforce_plus_plus":
        assert ours[1] == rewards  # identity
    else:  # group mean subtracted, no std division (one entry per rollout)
        assert ours[1] != rewards or batch == "g1"


# -- 3.4 REINFORCE++ returns, baseline advantages and DP=1 whitening ------------------------


@pytest.fixture
def single_process_dp(tmp_path, monkeypatch):
    import torch.distributed as dist

    from miles.backends.training_utils import parallel

    state = SimpleNamespace(cp=SimpleNamespace(size=1), effective_dp=SimpleNamespace(group=None))
    monkeypatch.setattr(parallel, "_parallel_state", state)
    created = False
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
        created = True
    yield state
    if created:
        dist.destroy_process_group()


def _rpp_inputs():
    kl = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.05, -0.1]), torch.tensor([0.0, 0.4, 0.1, 0.2])]
    masks = [torch.ones(3), torch.ones(2), torch.tensor([1.0, 1.0, 1.0, 0.0])]
    rewards = torch.tensor([1.0, 0.0, 1.0])
    return kl, masks, rewards


def test_rpp_returns_include_reward_kl_and_whiten_to_zero_mean(single_process_dp):
    kl, masks, rewards = _rpp_inputs()
    coef = 0.5
    returns = mu.get_reinforce_plus_plus_returns(
        rewards=rewards, kl=[k.clone() for k in kl], loss_masks=masks,
        response_lengths=[3, 2, 4], total_lengths=[5, 4, 6], kl_coef=coef, gamma=1.0)
    for k, m, r, ret in zip(kl, masks, rewards, returns):
        token = -coef * k.clone()
        token[int(m.nonzero()[-1])] += r
        hand = torch.flip(torch.cumsum(torch.flip(token, [0]), 0), [0])  # gamma = 1
        assert torch.allclose(ret, hand, atol=1e-6)
    no_kl = mu.get_reinforce_plus_plus_returns(
        rewards=rewards, kl=[k.clone() for k in kl], loss_masks=masks,
        response_lengths=[3, 2, 4], total_lengths=[5, 4, 6], kl_coef=0.0, gamma=1.0)
    assert any(not torch.equal(a, b) for a, b in zip(returns, no_kl))  # reward KL enters
    whitened = adv_mod.normalize_advantages(SimpleNamespace(), returns, masks, [5, 4, 6], [3, 2, 4])
    flat, mask = torch.cat(whitened), torch.cat(masks)
    assert abs(float((flat * mask).sum() / mask.sum())) < 1e-6
    var = float(((flat * mask) ** 2).sum() / (mask.sum() - 1))
    assert var == pytest.approx(1.0, rel=1e-4)


def test_rpp_baseline_advantages_include_reward_kl(single_process_dp):
    kl, masks, _ = _rpp_inputs()
    centered = torch.tensor([0.5, -0.5, 0.0])  # rollout side subtracted the group mean
    coef = 0.1
    advs = mu.get_reinforce_plus_plus_baseline_advantages(
        rewards=centered, kl=kl, loss_masks=masks, kl_coef=coef)
    for k, c, a in zip(kl, centered, advs):
        assert torch.allclose(a, c - coef * k)
    whitened = adv_mod.normalize_advantages(SimpleNamespace(), advs, masks, [5, 4, 6], [3, 2, 4])
    flat, mask = torch.cat(whitened), torch.cat(masks)
    assert abs(float((flat * mask).sum() / mask.sum())) < 1e-6


# -- 4.3 MAPO at p=0.5 equals Miles' built-in GRPO normalization ------------------------------


@pytest.mark.parametrize("rows", [
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
])
def test_mapo_half_correct_equals_builtin_grpo(rows):
    samples = [_sample(r, group=0, index=i) for i, r in enumerate(rows)]
    samples += [_sample(r, group=1, index=len(rows) + i) for i, r in enumerate(reversed(rows))]
    args = _args("grpo", transform="mapo", n=len(rows), batch=2)
    ours = rp.post_process(args, samples)[1]
    theirs = tdc._post_process_rewards(args, samples, custom_reward_post_process_func=None)[1]
    assert torch.equal(torch.tensor(ours), torch.tensor(theirs))


def test_mapo_half_correct_multi_segment_equals_builtin_grpo():
    samples = [_sample(1.0, group=0, index=0, rollout=1), _sample(1.0, group=0, index=1, rollout=1),
               _sample(0.0, group=0, index=2, rollout=2), _sample(1.0, group=0, index=3, rollout=3),
               _sample(0.0, group=0, index=4, rollout=4)]
    args = _args("grpo", transform="mapo")
    ours = rp.post_process(args, samples)[1]
    theirs = tdc._post_process_rewards(args, samples, custom_reward_post_process_func=None)[1]
    assert torch.equal(torch.tensor(ours), torch.tensor(theirs))
