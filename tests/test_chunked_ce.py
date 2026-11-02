import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

transformers = pytest.importorskip("transformers")

from yeto import chunked_ce  # noqa: E402
from yeto.causal_kernels import validate_kernel_request  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402

VOCAB = 512
HIDDEN = 64


def _tiny_qwen3(dtype=torch.float32, tie=True, seed=0):
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=tie,
    )
    config._attn_implementation = "sdpa"
    model = transformers.Qwen3ForCausalLM(config).to(dtype)
    return model


def _batch(seq=97, batch=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (batch, seq), generator=g)
    weights = (torch.rand(batch, seq, generator=g) > 0.4).float()
    return ids, weights


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _native(model, ids, weights):
    model.zero_grad(set_to_none=True)
    loss, n = sft_loss(model(input_ids=ids).logits, ids, "cross_entropy", weights)
    loss.backward()
    return loss.detach(), n, _grads(model)


def _chunked(model, ids, weights, chunk):
    model.zero_grad(set_to_none=True)
    loss, n = chunked_ce.chunked_sft_forward(model, ids, weights, chunk_tokens=chunk)
    loss.backward()
    return loss.detach(), n, _grads(model)


def _assert_match(a, b, rtol, atol):
    assert int(a[1]) == int(b[1])
    torch.testing.assert_close(a[0].float(), b[0].float(), rtol=rtol, atol=atol)
    assert a[2].keys() == b[2].keys()
    for name in a[2]:
        torch.testing.assert_close(
            a[2][name].float(), b[2][name].float(), rtol=rtol, atol=atol, msg=name
        )


@pytest.mark.parametrize("tie", [True, False])
@pytest.mark.parametrize("chunk", [1, 7, 64, 4096])
def test_fp32_loss_and_grads_match_native(tie, chunk):
    model = _tiny_qwen3(tie=tie)
    ids, weights = _batch()
    _assert_match(_native(model, ids, weights), _chunked(model, ids, weights, chunk), 1e-5, 1e-5)


def test_fractional_weights_and_no_weights_match():
    model = _tiny_qwen3()
    ids, weights = _batch()
    weights = weights * torch.rand_like(weights) * 2
    _assert_match(_native(model, ids, weights), _chunked(model, ids, weights, 13), 1e-5, 1e-5)
    _assert_match(_native(model, ids, None), _chunked(model, ids, None, 13), 1e-5, 1e-5)


def test_ignore_index_labels_match():
    # Input IDs never contain -100, so exercise ignore_index on the helper.
    model = _tiny_qwen3()
    hidden = torch.randn(50, 64, dtype=torch.float32, requires_grad=True)
    weight = model.lm_head.weight
    targets = torch.randint(0, VOCAB, (50,))
    targets[::3] = -100
    w = torch.rand(50)
    ref = (torch.nn.functional.cross_entropy(
        hidden @ weight.T, targets, ignore_index=-100, reduction="none") * w).sum()
    got = chunked_ce.chunked_linear_cross_entropy(hidden, weight, targets, w, chunk_tokens=8)
    torch.testing.assert_close(got, ref)


def test_all_masked_batch_backward():
    model = _tiny_qwen3()
    ids, _ = _batch()
    weights = torch.zeros_like(ids, dtype=torch.float32)
    loss, n = chunked_ce.chunked_sft_forward(model, ids, weights, chunk_tokens=8)
    loss.backward()
    assert float(loss) == 0.0 and int(n) == 0


def test_bf16_matches_native():
    model = _tiny_qwen3(dtype=torch.bfloat16)
    ids, weights = _batch()
    _assert_match(_native(model, ids, weights), _chunked(model, ids, weights, 32), 2e-2, 2e-2)


def test_lora_with_gradient_checkpointing_matches():
    peft = pytest.importorskip("peft")
    model = _tiny_qwen3()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = peft.get_peft_model(
        model, peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", init_lora_weights=False)
    )
    model.train()
    ids, weights = _batch()
    chunked_ce.require_chunked_ce_support(model)
    a = _native(model, ids, weights)
    b = _chunked(model, ids, weights, 16)
    assert a[2] and all("lora" in k for k in a[2])
    _assert_match(a, b, 1e-5, 1e-5)


def test_lora_on_lm_head_is_refused():
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        _tiny_qwen3(tie=False), peft.LoraConfig(r=4, target_modules=["lm_head", "q_proj"])
    )
    with pytest.raises(ValueError, match="plain"):
        chunked_ce.require_chunked_ce_support(model)


class _VocabTensorTracker(TorchDispatchMode):
    """Record the row count of every vocab-wide tensor any op produces and the
    peak of live vocab-wide bytes, across forward and backward."""

    def __init__(self):
        super().__init__()
        self.max_rows = 0
        self.live = {}
        self.peak_bytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in torch.utils._pytree.tree_leaves(out):
            if (
                isinstance(t, torch.Tensor)
                and t.dim()
                and t.shape[-1] == VOCAB
                and t.numel() != VOCAB * HIDDEN  # lm_head weight views
            ):
                self.max_rows = max(self.max_rows, t.numel() // VOCAB)
                key = id(t)
                self.live[key] = t.numel() * t.element_size()
                weakref.finalize(t, self.live.pop, key, None)
                self.peak_bytes = max(self.peak_bytes, sum(self.live.values()))
        return out


def _track(model, fn):
    model.zero_grad(set_to_none=True)
    tracker = _VocabTensorTracker()
    with tracker:
        loss, _ = fn()
        loss.backward()
    return tracker


def test_full_sequence_logits_never_materialized():
    model = _tiny_qwen3(tie=False)
    ids, _ = _batch(seq=257, batch=2)
    weights = torch.ones_like(ids, dtype=torch.float32)
    chunk = 16
    native = _track(model, lambda: sft_loss(model(input_ids=ids).logits, ids, "cross_entropy", weights))
    chunked = _track(model, lambda: chunked_ce.chunked_sft_forward(model, ids, weights, chunk_tokens=chunk))
    assert native.max_rows >= 2 * 256
    assert chunked.max_rows <= chunk
    assert chunked.peak_bytes * 10 < native.peak_bytes


def test_validate_kernel_request_chunked_any_device():
    validate_kernel_request("chunked", "cross_entropy", torch.device("cpu"), torch.float32,
                            tuning="lora", shard="ddp")
    with pytest.raises(ValueError):
        validate_kernel_request("chunked", "custom:x.py", torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError):
        validate_kernel_request("chunked", "cross_entropy", torch.device("cpu"), torch.float32,
                                shard="fsdp")


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("native", "torch-cross-entropy"),
        ("liger", "liger-fused-linear-cross-entropy"),
        ("chunked", chunked_ce.CHUNKED_LOSS_IMPLEMENTATION),
    ],
)
def test_kernel_recipe_names_the_loss_that_runs(backend, expected):
    from yeto.learner import loss_implementation_for

    assert loss_implementation_for(backend) == expected


def test_every_kernel_backend_has_a_recipe_loss_name():
    from yeto.causal_kernels import KERNEL_BACKENDS
    from yeto.learner import loss_implementation_for

    names = {loss_implementation_for(b) for b in KERNEL_BACKENDS}
    assert len(names) == len(KERNEL_BACKENDS)
    with pytest.raises(ValueError):
        loss_implementation_for("bogus")
