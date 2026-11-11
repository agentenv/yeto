import math

import torch
from torch import nn

from yeto.rl.adapters.miles.models.kimi_k31 import first_stage_grad, nan_inject


class _Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8).requires_grad_(False)
        self.expert = nn.Linear(8, 1, bias=False)

    def forward(self, tokens):
        return self.expert(self.embedding(tokens)).sum()


class _Wrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.module = inner


def _step(model, stage):
    stage.zero_grad(set_to_none=True)
    loss = stage(torch.tensor([1, 2, 3]))
    loss.backward()
    return float(loss), float(stage.expert.weight.grad.norm())


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(nan_inject.ENV, raising=False)
    stage = _Stage(); model = [_Wrapper(stage)]
    stage.train()
    first_stage_grad.before_train_step(None, 3, 0, model, None, None)
    assert all(math.isfinite(x) for x in _step(model, stage))


def test_target_rollout_gets_one_nan_forward_then_clean(monkeypatch):
    monkeypatch.setenv(nan_inject.ENV, "2")
    monkeypatch.delenv(first_stage_grad.CHAIN_ENV, raising=False)
    stage = _Stage(); model = [_Wrapper(stage)]
    stage.train()
    first_stage_grad.before_train_step(None, 1, 0, model, None, None)
    assert all(math.isfinite(x) for x in _step(model, stage))
    first_stage_grad.before_train_step(None, 2, 0, model, None, None)
    loss, grad_norm = _step(model, stage)
    assert math.isnan(loss) and math.isnan(grad_norm)  # what Megatron's get_grad_norm then sees
    assert all(math.isfinite(x) for x in _step(model, stage))  # one-shot: the next forward is clean
    first_stage_grad.before_train_step(None, 3, 0, model, None, None)
    assert all(math.isfinite(x) for x in _step(model, stage))
    assert len(stage.embedding._forward_hooks) == 1  # only the first-stage grad hook remains


def test_eval_forward_never_fires(monkeypatch):
    monkeypatch.setenv(nan_inject.ENV, "0")
    stage = _Stage(); model = [_Wrapper(stage)]
    nan_inject.maybe_arm(0, model)
    stage.eval()
    with torch.no_grad():
        assert math.isfinite(float(stage(torch.tensor([1]))))
    stage.train()
    assert math.isnan(float(stage(torch.tensor([1]))))
