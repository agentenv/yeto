"""S28 G-120: the full-parameter boundary reads FP32 masters under Megatron's CPU
optimizer offload (``HybridDeviceOptimizer``, the 120B island's optimizer).

The fake mirrors megatron-core 0.18.2 ``cpu_offloading/hybrid_optimizer.py`` (read on
npu1 n1): Megatron builds it with ``param_update_in_fp32=True``; ``param_to_fp32_param``
maps each bf16 shard of the param groups to an FP32 clone (on the host for the
offloaded share); step() updates the clone, puts it into ``state[shard]["master_param"]``
and a post-step hook copies it back into the bf16 shard.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto.rl import megatron_masters as mm


class _HybridLike:
    def __init__(self, shards, *, fraction=1.0, param_update_in_fp32=True):
        self.param_groups = [{"params": shards}]
        self.state: dict = {}
        self.param_update_in_fp32 = param_update_in_fp32
        self.param_to_fp32_param = {}
        total, moved = sum(s.numel() for s in shards), 0
        for s in shards:
            clone = s.detach().clone().cpu()  # host copy (pinned in Megatron)
            moved += s.numel()
            if param_update_in_fp32:
                clone = clone.float()
                self.param_to_fp32_param[s] = clone
        assert moved <= total and fraction >= 0

    def step(self, delta=0.001):
        for shard, master in self.param_to_fp32_param.items():
            master.add_(delta)  # an update below bf16 resolution near 1: master != bf16 copy
            self.state.setdefault(shard, {})["master_param"] = master
            shard.data.copy_(master)  # the copy-back hook


class _Range:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _HdoRank:
    def __init__(self, params, *, dp_rank, dp_size, **opt):
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
                shards.append(p.detach().view(-1)[s:e])
            offset += p.numel()
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": param_map}]}]
        # what yeto's optimizer_memory_overrides(1.0) resolves to; store_param_remainders
        # keeps Megatron's default True: HDO never sees it (only TE FusedAdam does)
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True, bf16=True,
                                      main_params_dtype=torch.float32, optimizer_cpu_offload=True,
                                      store_param_remainders=True)
        self.optimizer = _HybridLike(shards, **opt)


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter((1 + 0.1 * torch.randn(s, generator=g)).to(torch.bfloat16))
            for s in ((3, 5), (7,), (2, 4))]


def _gather(ranks, params):
    parts = [mm.full_masters(r, params, all_reduce_sum=lambda t: None, device="cpu") for r in ranks]
    return [sum(rank[i] for rank in parts) for i in range(len(params))]


def test_manifest_accepts_the_hybrid_offload():
    params = _params()
    found, owned = mm.distributed_ranges(_HdoRank(params, dp_rank=0, dp_size=1))
    assert found and len(owned) == len(params)


def test_before_a_step_the_master_is_the_fp32_clone():
    params = _params()
    ranks = [_HdoRank(params, dp_rank=r, dp_size=2) for r in range(2)]
    for p, m in zip(params, _gather(ranks, params), strict=True):
        assert m.dtype == torch.float32 and torch.equal(m, p.detach().float())


def test_after_steps_the_boundary_reads_the_host_fp32_master_not_the_bf16_copy():
    params = _params()
    ranks = [_HdoRank(params, dp_rank=r, dp_size=2) for r in range(2)]
    start = [p.detach().float().clone() for p in params]
    for r in ranks:
        r.optimizer.step()
        r.optimizer.step()
    for p, s, m in zip(params, start, _gather(ranks, params), strict=True):
        assert torch.allclose(m, s + 0.002, atol=1e-6)
        assert not torch.equal(m, p.detach().float())  # reading the bf16 copy would fail here
        # what a bf16 release writes: the master cast down is the model copy (copy-back hook)
        assert torch.equal(m.to(torch.bfloat16), p.detach())


def test_write_sets_the_host_master_and_the_bf16_model_copy():
    params = _params()
    ranks = [_HdoRank(params, dp_rank=r, dp_size=2) for r in range(2)]
    g = torch.Generator().manual_seed(9)
    targets = [torch.randn(p.shape, generator=g) for p in params]
    for r in ranks:
        assert mm.write_masters(r, params, targets) is True
    for p, t, m in zip(params, targets, _gather(ranks, params), strict=True):
        assert torch.equal(m, t)
        assert torch.equal(p.detach(), t.to(torch.bfloat16))


def test_hybrid_without_fp32_update_is_refused():
    params = _params()
    rank = _HdoRank(params, dp_rank=0, dp_size=1, param_update_in_fp32=False)
    with pytest.raises(mm.MegatronMasterError, match="param_update_in_fp32"):
        mm.full_masters(rank, params, all_reduce_sum=lambda t: None)
