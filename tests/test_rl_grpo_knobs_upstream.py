"""rl-algo-grpo-knobs checks against upstream Miles source (2.3/2.4/3.3/4.2-4.4/6.4).

Run in ``/home/michael/work/miles-next-venv`` with
``PYTHONPATH=<yeto>:/home/michael/work/miles-next``; skipped where Miles
is not importable. ``megatron.training`` is not installed in that venv, so
argv is parsed with Miles' own argument provider
(``get_miles_extra_args_provider``) on a bare argparse parser -- the Miles
half of upstream ``parse_args``; the Megatron half (and Miles'
``validate_args``) is covered by ``test_full_parse_args`` only where
``megatron.training`` imports (the pinned image).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

arguments = pytest.importorskip("miles.utils.arguments")
torch = pytest.importorskip("torch")

from yeto.rl.algos import grpo_knobs as gk  # noqa: E402
from yeto.rl.algos import reducers  # noqa: E402
from yeto.rl.algos import reward_pipeline as rp  # noqa: E402
from yeto.rl.engine.algorithm import BOUNDED_NONZERO_STD_FILTER, AlgorithmSpec, PluginRef  # noqa: E402
from yeto.rl.engine.miles_adapter.algorithm_flags import algorithm_argv  # noqa: E402

MILES_REPO = Path(arguments.__file__).resolve().parents[2]  # the miles checkout on PYTHONPATH
REF = {"source": "Qwen/Qwen3-0.6B", "revision": "rev-a"}


def miles_parse(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    arguments.get_miles_extra_args_provider()(parser)
    ns, unknown = parser.parse_known_args(argv)
    assert unknown == []
    return ns


def _dispatcher():
    return rp.dispatcher_ref().to_dict()


SPECS = {
    "clip_higher": (AlgorithmSpec(loss={"eps_clip": 0.2, "eps_clip_high": 0.28}),
                    {"eps_clip": 0.2, "eps_clip_high": 0.28}),
    "dual_clip": (AlgorithmSpec(loss={"eps_clip_c": 3.0}), {"eps_clip_c": 3.0}),
    "token": (AlgorithmSpec(loss={"aggregation": "token"}), {"calculate_per_token_loss": True}),
    "no_std": (AlgorithmSpec(advantage={"std_normalization": False}), {"grpo_std_normalization": False}),
    "entropy": (AlgorithmSpec(entropy_coef=0.001), {"entropy_coef": 0.001}),
    "over_sampling": (
        AlgorithmSpec(sampling={"filter": BOUNDED_NONZERO_STD_FILTER, "over_sampling_batch_size": 64}),
        # emitted by translate_run_config from the config batch in its R0 slot
        {},
    ),
    "constant": (
        AlgorithmSpec(loss={"aggregation": "constant", "constant_denominator": 4096,
                            "reducer": PluginRef.from_path(gk.REDUCER_PATH).to_dict()}),
        {"custom_pg_loss_reducer_function_path": gk.REDUCER_PATH, "calculate_per_token_loss": False},
    ),
    "overlong": (
        gk.with_pipeline_plugins(AlgorithmSpec(advantage={"reward_postprocess": _dispatcher(), "reward_shapers": [
            {"name": "overlong_penalty", "max_length": 1024, "cache_length": 128}]})),
        {"custom_reward_post_process_path": rp.DISPATCHER_PATH},
    ),
    **{
        f"kl_{est}": (
            AlgorithmSpec(kl={"placement": "loss", "coef": 0.001, "estimator": est, "ref_model": REF}),
            {"use_kl_loss": True, "kl_loss_coef": 0.001, "kl_loss_type": est, "kl_coef": 0.0},
        )
        for est in ("k1", "k2", "k3", "low_var_kl")
    },
}


@pytest.mark.parametrize("name", sorted(SPECS))
def test_miles_parser_accepts_translation(name):
    spec, expected = SPECS[name]
    assert spec.rejections() == []
    ns = miles_parse(["--advantage-estimator", "grpo", *algorithm_argv(spec)])
    for key, value in expected.items():
        assert getattr(ns, key) == value, key
    if name == "over_sampling":
        assert miles_parse(["--over-sampling-batch-size", "64"]).over_sampling_batch_size == 64


def test_kl_loss_triggers_ref_load_branch():
    """``ray/specs/train.py:58``: with_ref = kl_coef != 0 or use_kl_loss."""

    source = (MILES_REPO / "miles/ray/specs/train.py").read_text()
    assert "(trainer_args.kl_coef != 0 or trainer_args.use_kl_loss)" in source
    spec, _ = SPECS["kl_k3"]
    ns = miles_parse(algorithm_argv(spec))
    assert ns.kl_coef != 0 or ns.use_kl_loss
    default = miles_parse([])
    assert not (default.kl_coef != 0 or default.use_kl_loss)


# ------------------------------------------------------------------ 2.4 compute_policy_loss


def _policy_loss():
    from miles.backends.training_utils.loss_hub import math_utils

    fn = math_utils.compute_policy_loss
    return getattr(fn, "_torchdynamo_orig_callable", fn)


@pytest.mark.parametrize("adv", [2.0, -2.0])
@pytest.mark.parametrize("ratio", [0.5, 0.9, 1.0, 1.1, 1.25, 1.5, 5.0])
def test_dual_clip_and_clip_higher(adv, ratio):
    eps, eps_high, c = 0.2, 0.28, 3.0
    # Miles evaluates the clip bounds in float32: compare at float32 precision.
    ppo_kl = -torch.log(torch.tensor([ratio], dtype=torch.float64))
    a = torch.tensor([adv], dtype=torch.float64)
    r = torch.exp(-ppo_kl).item()
    clipped = min(max(r, 1 - eps), 1 + eps_high)
    base = max(-r * adv, -clipped * adv)  # PPO clipped surrogate (upper bound 1 + eps_high)
    expected_dual = min(-c * adv, base) if adv < 0 else base  # dual-clip: loss <= -c*A for A<0
    got, clipfrac = _policy_loss()(ppo_kl, a, eps, eps_high, c)
    assert got.item() == pytest.approx(expected_dual, rel=1e-6, abs=1e-6)
    got_plain, _ = _policy_loss()(ppo_kl, a, eps, eps_high, None)
    assert got_plain.item() == pytest.approx(base, rel=1e-6, abs=1e-6)
    if adv > 0 and r > 1 + eps_high:
        assert got_plain.item() == pytest.approx(-(1 + eps_high) * adv) and clipfrac.item() == 1.0
    if adv < 0 and r > 1 + eps_high:
        assert got.item() == pytest.approx(min(-c * adv, -r * adv))


# ------------------------------------------------------------------ 4.2 / 4.4 reducer


def _git_show(rev_path: str) -> bytes:
    return subprocess.run(["git", "-C", str(MILES_REPO), "show", rev_path], check=True,
                          capture_output=True).stdout


def test_vendored_source_pinned():
    raw = _git_show(f"{reducers.SOURCE_COMMIT}:{reducers.SOURCE_PATH}")
    assert hashlib.sha256(raw).hexdigest() == reducers.SOURCE_SHA256
    blob = subprocess.run(["git", "-C", str(MILES_REPO), "rev-parse",
                           f"{reducers.SOURCE_COMMIT}:{reducers.SOURCE_PATH}"],
                          check=True, capture_output=True, text=True).stdout.strip()
    assert blob == reducers.SOURCE_BLOB


def _load_original(monkeypatch):
    import sys
    import types

    mpu = types.SimpleNamespace(get_context_parallel_world_size=lambda: 1)
    core = types.ModuleType("megatron.core")
    core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    namespace: dict = {}
    exec(compile(_git_show(f"{reducers.SOURCE_COMMIT}:{reducers.SOURCE_PATH}"), "orig", "exec"), namespace)
    return namespace["get_pg_loss_reducer"]


def _plugins_args(denominator):
    spec = AlgorithmSpec(loss={"aggregation": "constant", "constant_denominator": denominator,
                               "reducer": PluginRef.from_path(gk.REDUCER_PATH).to_dict()})
    return SimpleNamespace(**gk.runtime_attrs(spec))


def _batch():
    g = torch.Generator().manual_seed(0)
    lengths = [3, 5, 2]
    x = torch.randn(sum(lengths), generator=g, dtype=torch.float32)
    masks = [torch.tensor([1, 1, 0.]), torch.tensor([1, 0, 1, 1, 1.]), torch.tensor([0, 0.])]
    return [sum(lengths) + 7] * 3, lengths, masks, x


def test_reducer_matches_original_at_1000(monkeypatch):
    original = _load_original(monkeypatch)
    monkeypatch.setattr(reducers, "_cp_world_size", lambda: 1)
    total, lengths, masks, x = _batch()
    ours = reducers.constant_denominator_reducer(total, lengths, masks, False, args=_plugins_args(1000))(x)
    ref = original(total, lengths, masks, False)(x)
    assert torch.equal(ours, ref)


@pytest.mark.parametrize("denominator", [1.0, 7.0, 4096.0])
def test_reducer_is_masked_sum_over_d(monkeypatch, denominator):
    monkeypatch.setattr(reducers, "_cp_world_size", lambda: 1)
    total, lengths, masks, x = _batch()
    got = reducers.constant_denominator_reducer(total, lengths, masks, False,
                                                args=_plugins_args(denominator))(x)
    expected = sum((xi * m).sum() for xi, m in zip(x.split(lengths), masks)) / denominator
    assert torch.allclose(got, expected, rtol=0, atol=1e-6)


def test_reducer_errors(monkeypatch):
    monkeypatch.setattr(reducers, "_cp_world_size", lambda: 1)
    total, lengths, masks, _ = _batch()
    with pytest.raises(rp.RewardPipelineError, match="is missing"):
        reducers.constant_denominator_reducer(total, lengths, masks, False, args=SimpleNamespace())
    no_reducer = SimpleNamespace(**{rp.PIPELINE_ATTR: rp.plugins_payload({"overlong_filter": True})})
    with pytest.raises(RuntimeError, match="no reducer.denominator"):
        reducers.constant_denominator_reducer(total, lengths, masks, False, args=no_reducer)
    with pytest.raises(RuntimeError, match="calculate-per-token-loss"):
        reducers.constant_denominator_reducer(total, lengths, masks, True, args=_plugins_args(10))
    monkeypatch.setattr(reducers, "_cp_world_size", lambda: 2)
    with pytest.raises(AssertionError, match="cp_size == 1"):
        reducers.constant_denominator_reducer(total, lengths, masks, False, args=_plugins_args(10))


# ------------------------------------------------------------------ 6.4 remove_sample semantics


def test_remove_sample_semantics(monkeypatch):
    from miles.backends.training_utils import cp_utils
    from miles.ray.rollout import train_data_conversion as tdc
    from miles.utils.types import Sample

    def samples(remove):
        out = []
        for i, (reward, length) in enumerate([(1.0, 4), (0.0, 3), (1.0, 2), (0.0, 5)]):
            s = Sample(group_index=0, index=i, reward=reward, response_length=length,
                       tokens=list(range(length + 2)), status=Sample.Status.COMPLETED)
            if remove and i == 0:
                s.status = Sample.Status.TRUNCATED
                s.remove_sample = True
            out.append(s)
        return out

    args = SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True,
                           grpo_std_normalization=True, n_samples_per_prompt=4, rollout_batch_size=1,
                           reward_key=None, use_dynamic_global_batch_size=False)
    kept = tdc.convert_samples_to_train_data(args, samples(False), {}, None, None)
    removed = tdc.convert_samples_to_train_data(args, samples(True), {}, None, None)
    assert removed["loss_masks"][0] == [0, 0, 0, 0]
    assert removed["loss_masks"][1:] == kept["loss_masks"][1:]
    assert removed["rewards"] == kept["rewards"]  # advantage statistics unchanged
    assert removed["rollout_mask_sums"][0] == 0

    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    lengths = [4, 3, 2, 5]
    masks = [torch.tensor(m, dtype=torch.float32) for m in removed["loss_masks"]]
    denominators = [torch.tensor(d) for d in removed["rollout_mask_sums"]]
    x = torch.ones(sum(lengths))
    reducer = cp_utils.get_sum_of_sample_mean([0] * 4, lengths, masks, False, denominators=denominators)
    # each kept sample contributes its token mean (1.0); the removed one 0 --
    # the per-sample sum is later divided by global_batch_size (loss.py:197-210),
    # which still counts the removed sample: it dilutes the others' weight.
    assert reducer(x).item() == 3.0
    token = cp_utils.get_sum_of_sample_mean([0] * 4, lengths, masks, True)
    assert token(x).item() == float(sum(sum(m) for m in removed["loss_masks"]))


def _example(name):
    return AlgorithmSpec.from_json_file(str(Path(__file__).resolve().parents[1]
                                            / "examples" / "rl_algorithms" / f"{name}.json"))


EXAMPLE_EXPECTED = {
    "dapo-like": {"eps_clip": 0.2, "eps_clip_high": 0.28, "calculate_per_token_loss": True,
                  "custom_reward_post_process_path": rp.DISPATCHER_PATH,
                  "over_sampling_batch_size": 8},
    "dr-grpo": {"grpo_std_normalization": False,
                "custom_pg_loss_reducer_function_path": gk.REDUCER_PATH},
}


@pytest.mark.parametrize("name", sorted(SPECS) + [f"example:{n}" for n in sorted(EXAMPLE_EXPECTED)])
def test_full_parse_args(tmp_path, name):
    """Upstream ``parse_args`` + ``validate_parsed_args`` on the full translated
    argv (pinned image only: needs megatron.training)."""

    pytest.importorskip("megatron.training")
    import dataclasses
    import json

    from test_rl_miles_adapter_config import _TINY_QWEN3, make_config, sub
    from yeto.rl.engine.miles_adapter import config as mc

    (tmp_path / "config.json").write_text(json.dumps(_TINY_QWEN3))
    (tmp_path / "p.jsonl").write_text('{"messages":[{"role":"user","content":"hi"}],"label":"x"}\n')
    cfg = dataclasses.replace(make_config(), hf_checkpoint=str(tmp_path), ref_load=str(tmp_path))
    cfg = sub(cfg, "data", prompt_path=str(tmp_path / "p.jsonl"))
    cfg = sub(cfg, "trainable", target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    if name.startswith("example:"):
        spec, expected = _example(name[8:]), EXAMPLE_EXPECTED[name[8:]]
    else:
        spec, expected = SPECS[name]
    if spec.sampling.over_sampling_batch_size is not None:
        cfg = sub(cfg, "batch", over_sampling_batch_size=spec.sampling.over_sampling_batch_size)
    launch = mc.translate_run_config(cfg, spec)
    args = mc.parse_miles_args(launch)
    for key, value in expected.items():
        assert getattr(args, key) == value, key
    if name == "over_sampling":
        assert args.over_sampling_batch_size == 64
    plugins = gk.plugins_config(spec)
    if plugins is None:
        assert not hasattr(args, rp.PIPELINE_ATTR) or getattr(args, rp.PIPELINE_ATTR) is None
    else:
        assert rp.read_plugins(args) == {"schema": rp.PLUGINS_SCHEMA, **plugins}


@pytest.mark.parametrize("name", ["dapo-like", "dr-grpo"])
def test_examples_parse_upstream(name):
    spec = AlgorithmSpec.from_json_file(str(Path(__file__).resolve().parents[1]
                                            / "examples" / "rl_algorithms" / f"{name}.json"))
    ns = miles_parse(["--advantage-estimator", "grpo", *algorithm_argv(spec)])
    if name == "dapo-like":
        assert (ns.eps_clip, ns.eps_clip_high, ns.calculate_per_token_loss) == (0.2, 0.28, True)
        assert ns.custom_reward_post_process_path == rp.DISPATCHER_PATH
    else:
        assert ns.grpo_std_normalization is False
        assert ns.custom_pg_loss_reducer_function_path == gk.REDUCER_PATH
