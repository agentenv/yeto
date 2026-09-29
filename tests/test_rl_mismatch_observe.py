"""Numeric checks of the mismatch mechanisms against the Miles sources
(change rl-algo-mismatch-correction, tasks 2.2, 2.3, 2.4, 3.1, 3.2, 4.2, 4.3,
5.2, 5.3).

Needs Miles (``MILES_NEXT_COMMIT`` checkout on ``sys.path``) and torch; skipped
otherwise. Run in miles-next-venv:

    PYTHONPATH=/home/michael/work/miles-next:$PWD \
      /home/michael/work/miles-next-venv/bin/python -m pytest -q tests/test_rl_mismatch_observe.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import pathlib
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")
losses = pytest.importorskip("miles.backends.training_utils.loss_hub.losses")

from miles.backends.training_utils import cp_utils  # noqa: E402
from miles.backends.training_utils.loss_hub import corrections  # noqa: E402
from miles.backends.training_utils.loss_hub.math_utils import compute_opsm_mask  # noqa: E402
from miles.backends.training_utils.parallel import (  # noqa: E402
    GroupInfo,
    ParallelState,
    set_parallel_state,
)

from yeto.rl.algos import mismatch_correction as mc  # noqa: E402
from yeto.rl.algos import mismatch_observe  # noqa: E402
from yeto.rl.algos.vendor import miles_mis  # noqa: E402
from yeto.rl.engine.algorithm import AlgorithmSpec, PluginRef  # noqa: E402
from yeto.rl.engine.miles_adapter.algorithm_flags import algorithm_argv  # noqa: E402

MILES_ROOT = pathlib.Path(losses.__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def _single_rank():
    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(ParallelState(
        intra_dp=trivial, intra_dp_cp=trivial, cp=trivial, tp=trivial, pp=trivial,
        ep=trivial, etp=trivial, indep_dp=trivial, is_pp_last_stage=True,
    ))


def _inputs(seed=0, response_lens=(5, 9, 3), prompt=4, vocab=17):
    g = torch.Generator().manual_seed(seed)
    total = [prompt + r for r in response_lens]
    return dict(
        response_lengths=list(response_lens),
        total_lengths=total,
        unconcat_tokens=[torch.randint(0, vocab, (t,), generator=g) for t in total],
        log_probs=[torch.randn(r, generator=g) * 0.3 - 2 for r in response_lens],
        rollout_log_probs=[torch.randn(r, generator=g) * 0.3 - 2 for r in response_lens],
        advantages=[torch.full((r,), float(torch.randn(1, generator=g))) for r in response_lens],
        loss_masks=[(torch.rand(r, generator=g) > 0.2).float() for r in response_lens],
        logits=torch.randn(1, sum(total), vocab, generator=g),
    )


def _args(**overrides):
    base = dict(
        use_rollout_logprobs=False, use_sampling_support_replay=False,
        skip_actor_forward_only=False, use_opsm=False, opsm_delta=1e-4,
        advantage_estimator="grpo", get_mismatch_metrics=False, use_tis=False,
        tis_clip=2.0, tis_clip_low=0.0, eps_clip=0.2, eps_clip_high=0.28,
        custom_tis_function_path=None, custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False, qkv_format="thd", entropy_coef=0.0,
        use_kl_loss=False, use_unbiased_kl=False, kl_loss_type="k1", kl_loss_coef=0.0,
        rollout_temperature=1.0, log_probs_chunk_size=-1, true_on_policy_mode=False,
        allgather_cp=False, observe_training_entropy=False,
    )
    base.update(overrides)
    return Namespace(**base)


def _log_probs_from_logits(logits, *, unconcat_tokens, total_lengths, response_lengths, **_):
    out, offset = [], 0
    flat = logits[0]
    for tokens, total, response in zip(unconcat_tokens, total_lengths, response_lengths):
        seq = flat[offset:offset + total]
        logp = torch.log_softmax(seq[total - response - 1: total - 1], dim=-1)
        out.append(logp.gather(-1, tokens[total - response:].unsqueeze(-1)).squeeze(-1))
        offset += total
    return {"log_probs": out, "entropy": [torch.zeros_like(x) for x in out]}


def _policy_loss(args, data, *, with_mask_sums: bool):
    batch = {k: [t.clone() for t in v] if isinstance(v, list) and v and torch.is_tensor(v[0])
             else list(v) for k, v in data.items() if k != "logits"}
    if with_mask_sums:
        batch["rollout_mask_sums"] = [m.sum() for m in batch["loss_masks"]]
    logits = data["logits"].clone().requires_grad_(True)
    reducer = cp_utils.get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"],
        args.calculate_per_token_loss, args.qkv_format, None,
        denominators=batch.get("rollout_mask_sums"),
    )
    loss, metrics = losses.policy_loss_function(args, batch, logits, reducer)
    loss.backward()
    return loss.detach(), logits.grad.detach(), metrics


@pytest.fixture
def local_log_probs(monkeypatch):
    monkeypatch.setattr(losses, "get_log_probs_and_entropy", _log_probs_from_logits)
    monkeypatch.setattr(
        losses, "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format="thd",
        max_seq_lens=None: loss_masks,
    )


# -- 2.2 plugin metrics equal Miles' built-in TIS metrics ------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_observe_metrics_match_vanilla_tis(seed):
    data = _inputs(seed)
    kwargs = dict(train_log_probs=data["log_probs"], rollout_log_probs=data["rollout_log_probs"],
                  loss_masks=data["loss_masks"])
    pg_loss = torch.randn(sum(data["response_lengths"]))
    out, masks, metrics = mismatch_observe.observe_mismatch(_args(), pg_loss=pg_loss, **kwargs)
    _, _, reference = corrections.vanilla_tis_function(
        _args(tis_clip=2.0, tis_clip_low=0.0), pg_loss=pg_loss, **kwargs
    )
    assert out is pg_loss and masks is data["loss_masks"]
    for key in ("tis", "tis_abs"):
        assert torch.equal(metrics[key], reference[key])
    ratio = reference["tis"]
    expected = (~((ratio >= 0.5) & (ratio <= 5.0))).float()
    assert torch.equal(metrics["mismatch_outside_0p5_5"], expected)


# -- 2.3 observe-only leaves loss and gradient bit-identical ----------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("with_mask_sums", [False, True])
@pytest.mark.parametrize("use_tis", [True, False])
def test_observe_gradient_identical(local_log_probs, seed, with_mask_sums, use_tis):
    data = _inputs(seed)
    base_loss, base_grad, base_metrics = _policy_loss(_args(), data, with_mask_sums=with_mask_sums)
    obs_args = _args(use_tis=use_tis, get_mismatch_metrics=True,
                     custom_tis_function_path=mc.OBSERVE_PATH)
    loss, grad, metrics = _policy_loss(obs_args, data, with_mask_sums=with_mask_sums)
    assert torch.equal(loss, base_loss)
    assert torch.equal(grad, base_grad)
    assert torch.count_nonzero(grad) > 0
    for key in ("tis", "tis_abs", "mismatch_outside_0p5_5", "ois", "train_rollout_kl",
                "ess_ratio", "train_rollout_logprob_abs_diff"):
        assert key in metrics and torch.isfinite(torch.as_tensor(metrics[key])).all(), key
    assert "tis" not in base_metrics


# -- 3.2 IcePop: in-interval weight = ratio, outside = 0 --------------------------


def test_icepop_weights():
    ratios = torch.tensor([0.1, 0.5, 0.9, 1.0, 3.0, 5.0, 5.1, 50.0])
    rollout = torch.zeros_like(ratios)
    train = torch.log(ratios)
    pg_loss, masks, metrics = corrections.icepop_function(
        _args(tis_clip=5.0, tis_clip_low=0.5), pg_loss=torch.ones_like(ratios),
        train_log_probs=[train], rollout_log_probs=[rollout], loss_masks=[torch.ones_like(ratios)],
    )
    ratio = torch.exp(train - rollout)
    inside = (ratio >= 0.5) & (ratio <= 5.0)
    assert inside.tolist() == [False, True, True, True, True, True, False, False]
    assert torch.equal(pg_loss[inside], ratio[inside])
    assert torch.equal(pg_loss[~inside], torch.zeros(int((~inside).sum())))
    # The masked-token fraction the adapter derives (tis_clipfrac) marks exactly
    # the out-of-interval tokens.
    assert torch.equal(metrics["tis_clipfrac"], (~inside).float())


def test_icepop_source_hash_pinned():
    ref = PluginRef.from_path(mc.ICEPOP_PATH)
    assert ref.sha256 == mc.ICEPOP_SOURCE_SHA256
    ref.verify(import_callable=True)


def test_icepop_all_out_of_interval_zero_gradient(local_log_probs):
    data = _inputs(3)
    data["rollout_log_probs"] = [lp.detach() + 10.0 for lp in data["log_probs"]]
    args = _args(use_tis=True, get_mismatch_metrics=True, tis_clip=5.0, tis_clip_low=0.5,
                 custom_tis_function_path=mc.ICEPOP_PATH)
    _, grad, metrics = _policy_loss(args, data, with_mask_sums=True)
    assert torch.count_nonzero(grad) == 0
    n = len(data["loss_masks"])
    assert float(metrics["tis_clipfrac"]) == pytest.approx(n)  # sum of per-sample means
    fraction = mc.masked_fraction_from_metrics(
        AlgorithmSpec(correction={"method": "custom", "function": mc.icepop_ref().to_dict(),
                                  "tis_clip": 5.0, "tis_clip_low": 0.5}),
        {"tis_clipfrac": float(metrics["tis_clipfrac"]) / n},
    )
    assert fraction == 1.0


# -- 4.3 OPSM mask condition ---------------------------------------------------


def test_opsm_mask_condition():
    delta = 0.05
    old = [torch.zeros(4) for _ in range(4)]
    # seq_kl = mean(old - new): 0.1 > delta for rows 0/1, 0.0 for rows 2/3.
    new = [torch.full((4,), -0.1), torch.full((4,), -0.1), torch.zeros(4), torch.zeros(4)]
    advantages = [torch.full((4,), a) for a in (-1.0, 1.0, -1.0, 1.0)]
    masks = [torch.ones(4) for _ in range(4)]
    mask, clipfrac = compute_opsm_mask(Namespace(opsm_delta=delta), new, old, advantages, masks)
    per_sequence = mask.view(4, 4)
    assert per_sequence[:, 0].tolist() == [0.0, 1.0, 1.0, 1.0]  # only adv<0 and kl>delta
    assert (per_sequence == per_sequence[:, :1]).all()
    assert float(clipfrac) == pytest.approx(1.0)  # 4 masked tokens / 4 tokens (not a fraction)


def test_opsm_all_sequences_masked_zero_gradient(local_log_probs):
    data = _inputs(4)
    data["advantages"] = [-torch.ones_like(a) for a in data["advantages"]]
    data["log_probs"] = [lp.detach() + 5.0 for lp in data["log_probs"]]  # large seq_kl
    _, grad, _ = _policy_loss(_args(use_opsm=True, opsm_delta=1e-4), data, with_mask_sums=False)
    assert torch.count_nonzero(grad) == 0


# -- 5.2 vendored MIS equals the Miles original -----------------------------------


def _original_mis():
    path = MILES_ROOT / "examples/infra_features/train_infer_mismatch_helper/mis.py"
    spec = importlib.util.spec_from_file_location("miles_examples_mis_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def test_vendored_mis_is_verbatim():
    _, path = _original_mis()
    vendored = pathlib.Path(miles_mis.__file__).read_text()
    header, _, body = vendored.partition(
        "# Do not edit below this line; re-copy on Miles upgrades (docs/MILES_RL.md).\n"
    )
    assert body == path.read_text()
    assert "9e4260de047a704208535c0e90c531929879ab40" in header and "Apache" in header


@pytest.mark.parametrize("level", ["token", "sequence", "geometric"])
@pytest.mark.parametrize("mode", ["truncate", "clip", "mask"])
@pytest.mark.parametrize("normalize", [False, True])
def test_vendored_mis_matches_original(level, mode, normalize):
    if normalize and level == "geometric":
        pytest.skip("Miles raises for geometric batch normalization (rejected by the spec)")
    original, _ = _original_mis()
    data = _inputs(5)
    low = None if mode == "truncate" else 0.8
    spec = AlgorithmSpec(correction={
        "method": "custom", "function": mc.mis_ref().to_dict(), "mis_level": level,
        "mis_mode": mode, "mis_upper_bound": 1.2, "mis_batch_normalize": normalize,
        **({"mis_lower_bound": low} if low is not None else {}),
    })
    assert spec.rejections() == []
    args = Namespace(use_tis=True, **mc.mis_config(spec))
    kwargs = dict(train_log_probs=data["log_probs"], rollout_log_probs=data["rollout_log_probs"],
                  loss_masks=data["loss_masks"])
    a = miles_mis.compute_mis_weights(args, **kwargs)
    b = original.compute_mis_weights(args, **kwargs)
    for x, y in zip(a[0], b[0]):
        assert torch.equal(x, y)
    for x, y in zip(a[1], b[1]):
        assert torch.equal(x, y)
    assert a[2].keys() == b[2].keys()
    for key in a[2]:
        for x, y in zip(a[2][key], b[2][key]):
            assert torch.equal(x, y), key


def test_mis_mask_all_rejected_masks_everything():
    data = _inputs(6)
    far = [lp + 3.0 for lp in data["rollout_log_probs"]]
    spec = AlgorithmSpec(correction={
        "method": "custom", "function": mc.mis_ref().to_dict(), "mis_level": "token",
        "mis_mode": "mask", "mis_lower_bound": 0.5, "mis_upper_bound": 2.0,
    })
    args = Namespace(use_tis=True, **mc.mis_config(spec))
    _, masks, metrics = miles_mis.compute_mis_weights(
        args, train_log_probs=data["log_probs"], rollout_log_probs=far,
        loss_masks=data["loss_masks"],
    )
    assert all(torch.count_nonzero(m) == 0 for m in masks)
    assert "mask_fraction_low" in "".join(metrics)  # -> mis_tis_mask_fraction_* in Miles logs


# -- upstream parser accepts every generated fragment (2.4/3.1/3.2/4.2/5.3) -------


def _parser():
    from miles.utils.arguments import get_miles_extra_args_provider

    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    return parser


SPECS = {
    "observe": lambda: mc.observe_spec(),
    "tis": lambda: AlgorithmSpec(correction={"method": "tis", "tis_clip": 2.0, "tis_clip_low": 0.0}),
    "icepop": lambda: AlgorithmSpec(correction={
        "method": "custom", "function": mc.icepop_ref().to_dict(), "tis_clip": 5.0,
        "tis_clip_low": 0.5, "mismatch_metrics": True}),
    "opsm_trainer": lambda: AlgorithmSpec(correction={"method": "opsm", "opsm_delta": 1e-4}),
    "opsm_rollout": lambda: AlgorithmSpec(correction={
        "method": "opsm", "opsm_delta": 1e-4, "opsm_old_logprob_source": "rollout",
        "use_rollout_logprobs": True}),
    "mis": lambda: AlgorithmSpec(correction={
        "method": "custom", "function": mc.mis_ref().to_dict(), "mis_level": "geometric",
        "mis_mode": "mask", "mis_lower_bound": 0.999, "mis_upper_bound": 1.001}),
}


@pytest.mark.parametrize("name", sorted(SPECS))
def test_upstream_parser_accepts_fragment(name):
    spec = SPECS[name]()
    argv = algorithm_argv(spec)
    namespace, rest = _parser().parse_known_args(argv)
    assert rest == []
    c = spec.correction
    if c.method == "tis" or c.function is not None:
        assert namespace.use_tis is True
    if c.tis_clip is not None:
        assert namespace.tis_clip == c.tis_clip and namespace.tis_clip_low == c.tis_clip_low
    if c.function is not None:
        assert namespace.custom_tis_function_path == c.function.path
    assert namespace.get_mismatch_metrics is c.mismatch_metrics
    assert namespace.use_opsm is (c.opsm_delta is not None)
    if c.opsm_delta is not None:
        assert namespace.opsm_delta == c.opsm_delta
    assert namespace.use_rollout_logprobs is c.use_rollout_logprobs
    # Miles' own cross-flag checks (arguments.py miles_validate_args 3453-3459).
    assert not (namespace.use_rollout_logprobs and namespace.use_tis)
    assert not namespace.get_mismatch_metrics or namespace.custom_tis_function_path
    if name == "mis":
        from miles.utils.file_arg_utils import resolve_file_arg
        import yaml

        loaded = yaml.safe_load(resolve_file_arg(namespace.custom_config_path))
        assert loaded == mc.mis_config(spec)
        assert importlib.util.find_spec(mc.MIS_PATH.rpartition(".")[0]) is not None


def test_icepop_hash_is_file_sha():
    path = pathlib.Path(corrections.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == mc.ICEPOP_SOURCE_SHA256
