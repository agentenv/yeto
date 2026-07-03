"""Unit tests for learner helpers that run without GPUs or a process group."""

import torch

from yeto.learner import allreduce_trainable_grads, normalize_param_name


# --- normalize_param_name -------------------------------------------------


def test_clean_names_pass_through():
    name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    assert normalize_param_name(name) == name


def test_strips_fsdp_prefix():
    assert (
        normalize_param_name("_fsdp_wrapped_module.base_model.model.lm_head.weight")
        == "base_model.model.lm_head.weight"
    )


def test_strips_nested_fsdp_prefixes():
    # Nested FSDP wrapping (auto_wrap_policy) inserts the segment at every
    # wrapped level.
    name = (
        "_fsdp_wrapped_module.base_model.model.model.layers.0."
        "_fsdp_wrapped_module.self_attn.q_proj.lora_B.default.weight"
    )
    assert (
        normalize_param_name(name)
        == "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    )


def test_strips_checkpoint_wrapper_prefix():
    name = (
        "_fsdp_wrapped_module._checkpoint_wrapped_module.layers.0.lora_A.default.weight"
    )
    assert normalize_param_name(name) == "layers.0.lora_A.default.weight"


def test_normalized_names_match_unwrapped_layout_names():
    # Fragment layouts are keyed by parameter name, so an fsdp-lora learner
    # must expose the exact names a ddp/single-GPU learner would.
    unwrapped = [
        "base_model.model.model.embed_tokens.weight",
        "base_model.model.model.layers.1.mlp.up_proj.lora_A.default.weight",
    ]
    wrapped = ["_fsdp_wrapped_module." + n for n in unwrapped]
    assert [normalize_param_name(n) for n in wrapped] == unwrapped


# --- allreduce_trainable_grads --------------------------------------------


def _param(grad):
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = grad
    return p


def test_allreduce_noop_when_world_is_one(monkeypatch):
    import yeto.learner as learner

    def boom(*a, **k):
        raise AssertionError("dist.all_reduce must not be called for world == 1")

    monkeypatch.setattr(learner.dist, "all_reduce", boom)
    p = _param(torch.ones(3))
    allreduce_trainable_grads([p], world=1)
    assert torch.equal(p.grad, torch.ones(3))


def test_allreduce_divides_by_world(monkeypatch):
    import yeto.learner as learner

    world = 4

    def fake_all_reduce(t, op=None):
        # Every rank holds the same grad, so SUM yields world * grad.
        t.mul_(world)

    monkeypatch.setattr(learner.dist, "all_reduce", fake_all_reduce)
    g = torch.tensor([1.0, -2.0, 0.5])
    p = _param(g.clone())
    allreduce_trainable_grads([p], world=world)
    # SUM over identical ranks then /world == the original grad (DDP mean).
    assert torch.allclose(p.grad, g)


def test_allreduce_skips_none_grads(monkeypatch):
    import yeto.learner as learner

    calls = []

    def fake_all_reduce(t, op=None):
        calls.append(t)

    monkeypatch.setattr(learner.dist, "all_reduce", fake_all_reduce)
    with_grad = _param(torch.full((3,), 2.0))
    without_grad = _param(None)
    allreduce_trainable_grads([with_grad, without_grad], world=2)
    assert len(calls) == 1
    assert without_grad.grad is None
    assert torch.allclose(with_grad.grad, torch.ones(3))  # 2.0 (sum stub is id) / 2
