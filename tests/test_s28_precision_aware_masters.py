"""S28 G-RCP: the full-parameter boundary reads FP32 masters under Megatron's
precision-aware optimizer (bf16 Adam moments kept by TE FusedAdam).

The fake inner optimizer mirrors the two TE FusedAdam behaviours the boundary relies
on (read on n1 from TransformerEngineNPU fused_adam.py): ``master_param`` exists only
after the first step(), created as the bf16 shard cast to FP32; ``get_unscaled_state``
returns an FP32 master as is and ``set_scaled_state`` stores it. Megatron's
precision-aware param groups hold the bf16 *shards* of the model params
(distrib_optimizer.py "shard_float16_params_this_group").
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto.rl import megatron_masters as mm
from yeto.rl.adapters.verl import megatron_engine as me


class _FusedAdamLike:
    def __init__(self, shards, *, store_param_remainders=False):
        self.param_groups = [{"params": shards}]
        self.state: dict = {}
        self.store_param_remainders = store_param_remainders

    def step(self):  # what _init_or_fix_state does to a master on the first step
        for p in self.param_groups[0]["params"]:
            st = self.state.setdefault(p, {})
            st.setdefault("master_param", p.detach().float())
            st.setdefault("exp_avg", torch.zeros_like(p))
            st.setdefault("exp_avg_sq", torch.zeros_like(p))
            st["master_param"] += 0.125  # the master moves away from the bf16 copy

    def get_unscaled_state(self, p, name):
        value = self.state[p][name]
        assert value.dtype == torch.float32
        return value.detach()

    def set_scaled_state(self, p, name, value):
        self.state.setdefault(p, {})[name] = value.detach().to(torch.float32)


class _Range:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _PaRank:
    """One DP rank of a precision-aware DistributedOptimizer (flat split across params)."""

    def __init__(self, params, *, dp_rank, dp_size, **config):
        total = sum(p.numel() for p in params)
        per = -(-total // dp_size)
        lo, hi = dp_rank * per, min((dp_rank + 1) * per, total)
        param_map, shards, self.model_param_group_index_map = {}, [], {}
        offset = 0
        for p in params:
            s, e = max(lo, offset) - offset, min(hi, offset + p.numel()) - offset
            if e > s:
                param_map[p] = {"param": _Range(s, e)}
                self.model_param_group_index_map[p] = (0, len(shards))
                shards.append(p.detach().view(-1)[s:e])  # a view, like Megatron's shard_model_param
            offset += p.numel()
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": param_map}]}]
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True, bf16=True,
                                      main_params_dtype=torch.float32, optimizer_cpu_offload=False,
                                      store_param_remainders=False)
        for k, v in config.items():
            setattr(self.config, k, v)
        self.optimizer = _FusedAdamLike(shards, store_param_remainders=self.config.store_param_remainders)

    def _get_main_param_and_optimizer_states(self, p):  # Megatron's: KeyError before step 1
        g, o = self.model_param_group_index_map[p]
        shard = self.optimizer.param_groups[g]["params"][o]
        return {"param": self.optimizer.state[shard]["master_param"]}


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g).to(torch.bfloat16)) for s in ((3, 5), (7,), (2, 4))]


def _gather(ranks, params):
    parts = []
    for r in ranks:
        parts.append(mm.full_masters(r, params, all_reduce_sum=lambda t: None))
    return [sum(rank[i] for rank in parts) for i in range(len(params))]


def test_before_the_first_step_the_master_is_the_bf16_shard_in_fp32():
    params = _params()
    ranks = [_PaRank(params, dp_rank=r, dp_size=2) for r in range(2)]
    with pytest.raises(KeyError):  # why the boundary does not call Megatron's accessor here
        ranks[0]._get_main_param_and_optimizer_states(params[0])
    for p, m in zip(params, _gather(ranks, params), strict=True):
        assert m.dtype == torch.float32
        assert torch.equal(m, p.detach().float())


def test_after_a_step_the_boundary_reads_fusedadam_fp32_master_not_the_bf16_copy():
    params = _params()
    ranks = [_PaRank(params, dp_rank=r, dp_size=2) for r in range(2)]
    for r in ranks:
        r.optimizer.step()
    for p, m in zip(params, _gather(ranks, params), strict=True):
        assert torch.equal(m, p.detach().float() + 0.125)  # a mutated read (bf16 copy) fails here


def test_write_sets_the_fusedadam_master_and_the_bf16_model_copy():
    params = _params()
    ranks = [_PaRank(params, dp_rank=r, dp_size=2) for r in range(2)]
    ranks[0].optimizer.step()  # one rank initialised, the other not: both paths
    g = torch.Generator().manual_seed(9)
    targets = [torch.randn(p.shape, generator=g) for p in params]
    for r in ranks:
        assert mm.write_masters(r, params, targets) is True
    for p, t, m in zip(params, targets, _gather(ranks, params), strict=True):
        assert torch.equal(m, t)  # FP32, not rounded through bf16
        assert torch.equal(p.detach(), t.to(torch.bfloat16))


@pytest.mark.parametrize("config, match", [
    ({"store_param_remainders": True}, "INT16 remainders"),
    ({"optimizer_cpu_offload": True}, "optimizer_cpu_offload"),
    ({"main_params_dtype": torch.float16}, "not FP32"),
])
def test_masters_that_are_not_exact_fp32_shards_are_still_refused(config, match):
    params = _params()
    rank = _PaRank(params, dp_rank=0, dp_size=1, **config)
    with pytest.raises(mm.MegatronMasterError, match=match):
        mm.full_masters(rank, params, all_reduce_sum=lambda t: None)


def test_bf16_moment_overrides_and_exclusion_with_offload():
    o = me.optimizer_memory_overrides(0.0, bf16_moments=True)
    assert o["optim.use_precision_aware_optimizer"] == "True"
    assert o["optim.exp_avg_dtype"] == o["optim.exp_avg_sq_dtype"] == "bf16"
    assert o["++optim.override_optimizer_config.store_param_remainders"] == "False"
    assert not any("cpu_offload" in k or "main_grads_dtype" in k for k in o)
    assert me.optimizer_memory_overrides(0.0) == {}
    with pytest.raises(me.MegatronShapeError):
        me.optimizer_memory_overrides(0.5, bf16_moments=True)
