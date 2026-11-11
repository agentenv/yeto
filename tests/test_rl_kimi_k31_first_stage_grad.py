from types import SimpleNamespace

import torch
from torch import nn

from yeto.rl.adapters.miles.models.kimi_k31 import first_stage_grad


class _Checkpoint(torch.autograd.Function):
    """Megatron's tensor_parallel.checkpoint, trimmed: reentrant, recompute in backward."""

    @staticmethod
    def forward(ctx, run_function, hidden):
        ctx.run_function = run_function
        ctx.save_for_backward(hidden)
        with torch.no_grad():
            return run_function(hidden)

    @staticmethod
    def backward(ctx, grad):
        (hidden,) = ctx.saved_tensors
        hidden = hidden.detach().requires_grad_(hidden.requires_grad)
        with torch.enable_grad():
            out = ctx.run_function(hidden)
        torch.autograd.backward(out, grad)
        return None, hidden.grad


class _FirstStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8).requires_grad_(False)
        self.expert = nn.Linear(8, 8, bias=False)

    def forward(self, tokens):
        return _Checkpoint.apply(self.expert, self.embedding(tokens))


class _Wrapper(nn.Module):  # DDP / Float16Module: the model sits under .module
    def __init__(self, inner):
        super().__init__()
        self.module = inner


def _grad(stage):
    stage.train()
    out = stage(torch.tensor([1, 2, 3]))
    if out.requires_grad:
        out.sum().backward()
    return stage.expert.weight.grad


def test_frozen_embedding_loses_first_stage_grad_without_hook():
    assert _grad(_FirstStage()) is None


def test_hook_through_wrappers_restores_grad_and_is_idempotent(monkeypatch):
    monkeypatch.delenv(first_stage_grad.CHAIN_ENV, raising=False)
    stage = _FirstStage()
    model = [_Wrapper(_Wrapper(stage))]
    for step in range(2):
        first_stage_grad.before_train_step(None, 0, step, model, None, None)
    assert len(stage.embedding._forward_hooks) == 1
    grad = _grad(stage)
    assert grad is not None and grad.abs().sum() > 0
    assert stage.embedding.weight.grad is None


def test_later_stage_without_embedding_is_untouched():
    assert first_stage_grad.install([_Wrapper(nn.Linear(2, 2))]) == 0


def test_eval_forward_is_left_alone():
    stage = _FirstStage()
    first_stage_grad.install(stage)
    stage.eval()
    assert not stage(torch.tensor([1])).requires_grad


calls = []


def _recorder(*a):
    calls.append(a)


def test_chained_hook_runs_after(monkeypatch):
    monkeypatch.setenv(first_stage_grad.CHAIN_ENV, f"{__name__}._recorder")
    calls.clear()
    args = SimpleNamespace()
    first_stage_grad.before_train_step(args, 3, 4, [_Wrapper(_FirstStage())], "opt", "sched")
    assert calls and calls[0][:3] == (args, 3, 4) and calls[0][4:] == ("opt", "sched")
