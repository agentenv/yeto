"""Regression test for the Miles LoRA gradient-flow bug.

Megatron's distributed data parallel registers a hook on each trainable
parameter's ``AccumulateGrad`` node at construction time and routes the
gradient into ``main_grad`` from there.  PyTorch's ``Tensor.data`` setter
resets that accumulator whenever the new storage has a different dtype
(``VariableHooks::set_data``), so a bf16<-fp32 swap silently drops the hook
even after the original storage is restored.  The policy-sync integration
layer used to do exactly that swap to expose fp32 optimizer masters, which
left every LoRA optimizer step a no-op.  This test pins the PyTorch
behaviour so a future refactor cannot reintroduce it; it needs neither a
GPU nor Miles.
"""

from __future__ import annotations

import torch


def _parameter_with_accumulator_hook() -> tuple[torch.nn.Parameter, list[int], object]:
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    parameter.main_grad = torch.zeros(2, dtype=torch.float32)
    fired: list[int] = []

    def hook(*_unused) -> None:
        fired.append(1)
        parameter.main_grad += parameter.grad.float()

    # Autograd only holds the accumulator weakly; Megatron's DDP keeps every
    # node it hooked in ``self.grad_accs`` for exactly this reason.
    accumulator = _accumulator(parameter)
    accumulator.register_hook(hook)
    return parameter, fired, accumulator


def _accumulator(parameter: torch.Tensor):
    return parameter.expand_as(parameter).grad_fn.next_functions[0][0]


def _backward(parameter: torch.nn.Parameter) -> None:
    (parameter.float() * 2).sum().backward()


def test_cross_dtype_data_swap_drops_grad_accumulator_hook():
    parameter, fired, accumulator = _parameter_with_accumulator_hook()
    original = parameter.data

    parameter.data = torch.zeros(2, dtype=torch.float32)
    parameter.data = original

    assert _accumulator(parameter) is not accumulator
    _backward(parameter)
    assert fired == []
    assert parameter.grad.tolist() == [2.0, 2.0]
    assert parameter.main_grad.tolist() == [0.0, 0.0]


def test_same_dtype_data_swap_keeps_grad_accumulator_hook():
    parameter, fired, accumulator = _parameter_with_accumulator_hook()
    original = parameter.data

    parameter.data = torch.zeros(2, dtype=torch.bfloat16)
    parameter.data = original

    assert _accumulator(parameter) is accumulator
    _backward(parameter)
    assert fired == [1]
    assert parameter.main_grad.tolist() == [2.0, 2.0]


def test_replacing_module_parameter_entry_keeps_grad_accumulator_hook():
    """The fix: swap the module's registered entry, never the Parameter's storage."""

    parameter, fired, accumulator = _parameter_with_accumulator_hook()
    module = torch.nn.Module()
    module.weight = parameter

    master = torch.zeros(4, dtype=torch.float32)[:2]
    module._parameters["weight"] = torch.nn.Parameter(master, requires_grad=False)
    assert module.weight.dtype == torch.float32
    assert module.weight.data_ptr() == master.data_ptr()
    module._parameters["weight"] = parameter

    assert module.weight is parameter
    assert _accumulator(parameter) is accumulator
    _backward(parameter)
    assert fired == [1]
    assert parameter.main_grad.tolist() == [2.0, 2.0]
