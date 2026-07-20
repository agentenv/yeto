"""Causal-LM auto-batch resolution and probe-purity tests."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from yeto import autobatch


def _args(mb="auto", seq_len=2048, grad_accum=4):
    return SimpleNamespace(micro_batch_size=mb, seq_len=seq_len, grad_accum=grad_accum)


class Tok:
    def __len__(self):
        return 1000


class TinyCausalLM(torch.nn.Module):
    def __init__(self, *, consume_cpu_rng: bool = False):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.dropout = torch.nn.Dropout(0.25)
        self.output = torch.nn.Linear(8, 32)
        self.consume_cpu_rng = consume_cpu_rng

    def forward(self, input_ids):
        if self.consume_cpu_rng:
            torch.rand(())
        hidden = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(logits=self.output(hidden))


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.device == expected.device
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_equal(actual_item, expected_item)
        return
    assert actual == expected


def test_int_or_auto():
    assert autobatch.int_or_auto("auto") == "auto"
    assert autobatch.int_or_auto("8") == 8


def test_explicit_size_skips_probe(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("probe must not run")

    monkeypatch.setattr(autobatch, "_probe_once", boom)
    got = autobatch.resolve_micro_batch_size(
        _args(mb=3), None, {}, None, Tok(), SimpleNamespace(type="cuda"), 1
    )
    assert got == 3


def test_cpu_returns_one_without_probe(monkeypatch):
    monkeypatch.setattr(autobatch, "_probe_once", lambda *a: (_ for _ in ()).throw(AssertionError))
    got = autobatch.resolve_micro_batch_size(
        _args(), None, {}, None, Tok(), SimpleNamespace(type="cpu"), 1
    )
    assert got == 1


def test_probe_uses_largest_fitting_exact_divisor(monkeypatch, caplog):
    sizes = []

    def probe(model, params, opt, seq_len, vocab, device, mb):
        sizes.append(mb)
        if mb >= 10:
            raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(autobatch, "_probe_once", probe)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    caplog.set_level("INFO", logger="learner")
    got = autobatch.resolve_micro_batch_size(
        _args(grad_accum=18), None, {}, None, Tok(), SimpleNamespace(type="cuda"), 1
    )
    assert got == 9
    assert autobatch.exact_grad_accum(18, got) == 2
    assert got * autobatch.exact_grad_accum(18, got) == 18
    assert sizes == [1, 2, 3, 6, 9, 18]
    assert "largest fitting exact micro-batch=9" in caplog.text
    assert "effective batch=18" in caplog.text


def test_non_power_of_two_budget_is_not_ceil_increased(monkeypatch):
    sizes = []

    def probe(model, params, opt, seq_len, vocab, device, mb):
        sizes.append(mb)
        if mb > 4:
            raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(autobatch, "_probe_once", probe)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    micro_batch = autobatch.resolve_micro_batch_size(
        _args(grad_accum=6), None, {}, None, Tok(), SimpleNamespace(type="cuda"), 1
    )
    grad_accum = autobatch.exact_grad_accum(6, micro_batch)

    assert sizes == [1, 2, 3, 6]
    assert (micro_batch, grad_accum) == (3, 2)
    assert micro_batch * grad_accum == 6


def test_oom_at_one_is_a_clear_error(monkeypatch):
    def probe(*a, **k):
        raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(autobatch, "_probe_once", probe)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(RuntimeError, match="does not fit even micro-batch 1"):
        autobatch.resolve_micro_batch_size(
            _args(), None, {}, None, Tok(), SimpleNamespace(type="cuda"), 1
        )


def test_exact_grad_accum_rejects_an_inexact_recipe():
    assert autobatch.exact_grad_accum(12, 6) == 2
    with pytest.raises(ValueError, match="does not divide"):
        autobatch.exact_grad_accum(10, 4)


def test_rebalance_grad_accum_legacy_diffusion_behavior():
    assert autobatch.rebalance_grad_accum(4, 1) == 4
    assert autobatch.rebalance_grad_accum(4, 2) == 2
    assert autobatch.rebalance_grad_accum(4, 8) == 1
    assert autobatch.rebalance_grad_accum(1, 64) == 1


def test_probe_preserves_fresh_adamw_params_state_scheduler_grads_and_cpu_rng():
    torch.manual_seed(314159)
    model = TinyCausalLM()
    params = dict(model.named_parameters())
    opt = torch.optim.AdamW(
        params.values(), lr=3e-3, weight_decay=0.4, foreach=False, fused=False
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.75**step)

    original_grads = {}
    for param in params.values():
        param.grad = torch.full_like(param, 0.125)
        original_grads[param] = (param.grad, param.grad.detach().clone())
    parameter_values = {name: param.detach().clone() for name, param in params.items()}
    parameter_versions = {name: param._version for name, param in params.items()}
    optimizer_state = _clone_tree(opt.state_dict())
    scheduler_state = _clone_tree(sched.state_dict())
    cpu_rng = torch.get_rng_state().clone()

    autobatch._probe_once(model, params, opt, 7, 31, torch.device("cpu"), 3)

    for name, param in params.items():
        assert torch.equal(param, parameter_values[name])
        assert param._version == parameter_versions[name]
        original, value = original_grads[param]
        assert param.grad is original
        assert torch.equal(param.grad, value)
    _assert_tree_equal(opt.state_dict(), optimizer_state)
    _assert_tree_equal(sched.state_dict(), scheduler_state)
    assert torch.equal(torch.get_rng_state(), cpu_rng)


def test_probe_preserves_materialized_adamw_state_and_scheduler():
    torch.manual_seed(2718)
    model = TinyCausalLM()
    params = dict(model.named_parameters())
    opt = torch.optim.AdamW(
        params.values(), lr=1e-2, weight_decay=0.3, foreach=False, fused=False
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

    ids = torch.randint(0, 31, (2, 5))
    model(input_ids=ids).logits.sum().backward()
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    assert opt.state

    parameter_values = {name: param.detach().clone() for name, param in params.items()}
    parameter_versions = {name: param._version for name, param in params.items()}
    optimizer_state = _clone_tree(opt.state_dict())
    scheduler_state = _clone_tree(sched.state_dict())
    torch.manual_seed(1618)
    cpu_rng = torch.get_rng_state().clone()

    autobatch._probe_once(model, params, opt, 5, 31, torch.device("cpu"), 2)

    for name, param in params.items():
        assert torch.equal(param, parameter_values[name])
        assert param._version == parameter_versions[name]
        assert param.grad is None
    _assert_tree_equal(opt.state_dict(), optimizer_state)
    _assert_tree_equal(sched.state_dict(), scheduler_state)
    assert torch.equal(torch.get_rng_state(), cpu_rng)


def test_probe_restores_state_when_forward_raises_oom():
    class OOMModel(TinyCausalLM):
        def forward(self, input_ids):
            super().forward(input_ids)
            torch.rand(())
            raise torch.cuda.OutOfMemoryError("synthetic after forward")

    torch.manual_seed(42)
    model = OOMModel()
    params = dict(model.named_parameters())
    opt = torch.optim.AdamW(
        params.values(), lr=1e-2, weight_decay=0.5, foreach=False, fused=False
    )
    parameter_values = {name: param.detach().clone() for name, param in params.items()}
    optimizer_state = _clone_tree(opt.state_dict())
    cpu_rng = torch.get_rng_state().clone()

    with pytest.raises(torch.cuda.OutOfMemoryError):
        autobatch._probe_once(model, params, opt, 5, 31, torch.device("cpu"), 2)

    for name, param in params.items():
        assert torch.equal(param, parameter_values[name])
        assert param.grad is None
    _assert_tree_equal(opt.state_dict(), optimizer_state)
    assert torch.equal(torch.get_rng_state(), cpu_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_probe_preserves_cpu_and_cuda_rng():
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(123)
    torch.cuda.manual_seed(456)
    model = TinyCausalLM(consume_cpu_rng=True).to(device)
    params = dict(model.named_parameters())
    opt = torch.optim.AdamW(
        params.values(), lr=1e-3, weight_decay=0.2, foreach=False, fused=False
    )
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()

    autobatch._probe_once(model, params, opt, 5, 31, device, 2)

    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
