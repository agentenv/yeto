"""The vendored DP-invariant optimizer state (yeto.rl.adapters.miles.dp_invariant_state), CPU part.

Vendored with the module from Miles fork 28bc6cc24
tests/fast/backends/megatron_utils/test_lora_dp_invariant_state.py; the
Megatron-backed and fork-checkpoint parts are in
test_rl_dp_invariant_state_megatron.py (skipped without Megatron / the fork).
"""

import random
import shutil
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yeto.rl.adapters.miles import dp_invariant_state as dps

_NAMES = ["layers.0.lora_A.weight", "layers.0.lora_B.weight", "layers.1.lora_A.weight"]

class _Model(torch.nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.params = {name: torch.nn.Parameter(torch.randn(3, 2, generator=g)) for name in _NAMES}
        for index, param in enumerate(self.params.values()):
            self.register_parameter(f"p{index}", param)

    def named_parameters(self, *args, **kwargs):
        return list(self.params.items())


def _optimizer(model: _Model, order: list[str]) -> torch.optim.Optimizer:
    params = model.params
    return torch.optim.AdamW(
        [
            {"params": [params[n] for n in order if n.startswith("layers.0")], "lr": 0.1},
            {"params": [params[n] for n in order if n.startswith("layers.1")], "lr": 0.01, "weight_decay": 0.0},
        ]
    )


def _step(model: _Model, optimizer: torch.optim.Optimizer, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    optimizer.zero_grad()
    loss = sum((p * torch.randn(p.shape, generator=g)).sum() for p in model.params.values())
    loss.backward()
    optimizer.step()


def _named(model):
    return model.named_parameters()


def _export(model, optimizer):
    return dps.export_named_optimizer_state(optimizer, _named(model))


def _load(model, optimizer, shards):
    dps.load_named_optimizer_state(optimizer, _named(model), dps.merge_named_optimizer_states(shards))


class TestNamedOptimizerState:
    def test_state_is_keyed_by_name_and_survives_a_different_param_order(self):
        src = _Model(0)
        src_opt = _optimizer(src, _NAMES)
        _step(src, src_opt, 1)
        named = _export(src, src_opt)

        tgt = _Model(0)
        tgt_opt = _optimizer(tgt, list(reversed(_NAMES)))
        _load(tgt, tgt_opt, [named])
        for name in _NAMES:
            s, t = src_opt.state[src.params[name]], tgt_opt.state[tgt.params[name]]
            torch.testing.assert_close(t["exp_avg"], s["exp_avg"], rtol=0, atol=0)
            torch.testing.assert_close(t["step"], s["step"], rtol=0, atol=0)
            torch.testing.assert_close(tgt.params[name].data, src.params[name].data, rtol=0, atol=0)
        assert tgt_opt.param_groups[1]["lr"] == 0.01

    def test_replicated_shards_must_agree(self):
        model = _Model(0)
        opt = _optimizer(model, _NAMES)
        _step(model, opt, 1)
        a = _export(model, opt)
        _step(model, opt, 2)
        b = _export(model, opt)
        assert set(dps.merge_named_optimizer_states([a, a])) == set(_NAMES)
        with pytest.raises(dps.DpInvariantStateError, match="disagree"):
            dps.merge_named_optimizer_states([a, b])

    def test_missing_names_and_mixed_hyperparameters_are_refused(self):
        model = _Model(0)
        opt = _optimizer(model, _NAMES)
        _step(model, opt, 1)
        merged = dps.merge_named_optimizer_states([_export(model, opt)])
        merged.pop(_NAMES[0])
        with pytest.raises(KeyError, match="lacks parameters"):
            dps.load_named_optimizer_state(opt, _named(model), merged)

        merged = dps.merge_named_optimizer_states([_export(model, opt)])
        mixed = torch.optim.AdamW([{"params": [model.params[_NAMES[0]], model.params[_NAMES[2]]]}])
        with pytest.raises(dps.DpInvariantStateError, match="different hyperparameters"):
            dps.load_named_optimizer_state(mixed, [(n, model.params[n]) for n in (_NAMES[0], _NAMES[2])], merged)

    def test_an_optimizer_over_non_model_tensors_is_refused(self):
        model = _Model(0)
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(2))])
        with pytest.raises(NotImplementedError, match="maps to no model parameter"):
            _export(model, opt)

    def test_an_unknown_optimizer_wrapper_is_refused(self):
        with pytest.raises(NotImplementedError, match="does not support"):
            _export(_Model(0), SimpleNamespace())


class _Float16Optimizer:
    """The attribute layout of Megatron Float16OptimizerWithFloat16Params (whose ctor needs CUDA tensors)."""

    def __init__(self, model, lr=0.1):
        self.float16_groups = [[p for p in model.parameters()]]
        self.fp32_from_float16_groups = [[p.detach().clone().float() for p in self.float16_groups[0]]]
        self.fp32_from_fp32_groups = [[]]
        self.optimizer = torch.optim.Adam(self.fp32_from_float16_groups[0], lr=lr, foreach=False)

    def step(self, grads):
        for main, g in zip(self.fp32_from_float16_groups[0], grads, strict=True):
            main.grad = g.clone()
        self.optimizer.step()
        for model_p, main in zip(self.float16_groups[0], self.fp32_from_float16_groups[0], strict=True):
            model_p.data.copy_(main)


class _Bf16Model(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lora_A = torch.nn.Parameter(torch.randn(3, 5, generator=g).bfloat16())
        self.lora_B = torch.nn.Parameter(torch.randn(7, generator=g).bfloat16())


class TestFloat16Optimizer:
    def test_fp32_main_params_and_state_roundtrip_by_name(self):
        src = _Bf16Model(0)
        opt = _Float16Optimizer(src)
        opt.step([torch.randn(3, 5), torch.randn(7)])
        named = _export(src, opt)
        assert named["entries"]["lora_A"]["tensors"]["param"].dtype == torch.float32

        tgt = _Bf16Model(1)
        tgt_opt = _Float16Optimizer(tgt)
        _load(tgt, tgt_opt, [named])
        for s_main, t_main in zip(opt.fp32_from_float16_groups[0], tgt_opt.fp32_from_float16_groups[0], strict=True):
            torch.testing.assert_close(t_main, s_main, rtol=0, atol=0)  # full fp32 precision, not the bf16 copy
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(tgt_opt.optimizer.state[t_main][key], opt.optimizer.state[s_main][key])
        grads = [torch.randn(3, 5), torch.randn(7)]
        opt.step(grads)
        tgt_opt.step(grads)
        for s_main, t_main in zip(opt.fp32_from_float16_groups[0], tgt_opt.fp32_from_float16_groups[0], strict=True):
            torch.testing.assert_close(t_main, s_main, rtol=0, atol=0)


# --- DistributedOptimizer: the real Megatron range/state accessors on a CPU-constructible object -------------------



class TestRngState:
    def test_capture_then_restore_replays_every_cpu_generator(self):
        state = dps.capture_rng_state()
        first = (random.random(), np.random.rand(), torch.rand(2))

        dps.restore_rng_state(state)
        second = (random.random(), np.random.rand(), torch.rand(2))

        assert first[:2] == second[:2]
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_the_state_survives_torch_save(self, tmp_path):
        state = dps.capture_rng_state()
        expected = torch.rand(3)
        torch.save(state, tmp_path / "rng.pt")

        dps.restore_rng_state(torch.load(tmp_path / "rng.pt", weights_only=False))

        torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)




def test_a_megatron_stub_optimizer_has_no_slots():
    """A partial-experts run freezes every dense weight; Megatron then chains a stub
    (``is_stub_optimizer``, no inner optimizer) for the empty dense group."""
    model = _Model()
    inner = _optimizer(model, list(_NAMES))
    chained = SimpleNamespace(chained_optimizers=[SimpleNamespace(is_stub_optimizer=True, optimizer=None), inner])
    names = [slot.name for slot in dps.optimizer_slots(chained, _named(model))]
    assert names == [slot.name for slot in dps.optimizer_slots(inner, _named(model))]
    assert names
