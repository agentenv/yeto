"""rl-infra-spec 2.4 (decision (a)): DP-sharded FP32 masters under Megatron's DistributedOptimizer.

Ranges come from the real ``DistributedOptimizer._build_model_gbuf_param_range_map`` and the
state accessor is the real ``_get_main_param_and_optimizer_states`` (same CPU-constructible
double as the fork-M5 v2 tests); DP=2 ranks run in one process and the DP all-reduce is the sum
of both ranks' contributions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

_dist = pytest.importorskip("megatron.core.optimizer.distrib_optimizer")

from yeto.rl.engine.miles_adapter import state_plugin as sp  # noqa: E402


class _DistOptRank:
    _get_main_param_and_optimizer_states = _dist.DistributedOptimizer._get_main_param_and_optimizer_states

    def __init__(self, params, full_main, *, dp_rank, dp_size):
        index_map, offset = {}, 0
        for p in params:
            index_map[p] = (offset, offset + p.numel(), 0)
            offset += p.numel()
        shard = -(-offset // dp_size)
        world = _dist.Range(dp_rank * shard, min((dp_rank + 1) * shard, offset))
        param_map = _dist.DistributedOptimizer._build_model_gbuf_param_range_map(index_map, world, 0)
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": param_map}]}]
        self.model_param_group_index_map, mains = {}, []
        for p, ranges in param_map.items():
            r = ranges["param"]
            self.model_param_group_index_map[p] = (0, len(mains))
            mains.append(full_main[p].reshape(-1)[r.start : r.end].clone())
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        self.optimizer = torch.optim.AdamW([{"params": mains or [torch.zeros(0)], "lr": 0.1}])


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    # odd sizes so the DP split cuts through a parameter
    return [torch.nn.Parameter(torch.randn(s, generator=g).to(torch.bfloat16)) for s in ((3, 5), (7,), (2, 4))]


def _masters(params, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {p: torch.randn(p.shape, generator=g, dtype=torch.float32) for p in params}


def test_dp2_export_gathers_exactly_the_unsharded_masters():
    params = _params()
    ref = _masters(params)
    ranks = [_DistOptRank(params, ref, dp_rank=r, dp_size=2) for r in range(2)]
    contributions = []
    for opt in ranks:
        sp.full_masters(opt, params, all_reduce_sum=lambda t: contributions.append(t.clone()))
    total = contributions[0] + contributions[1]
    expected = torch.cat([ref[p].reshape(-1) for p in params])
    assert torch.equal(total, expected)  # bitwise: each element comes from exactly one rank
    # DP=1 (one rank owns everything) gives the same masters without a reduction partner
    single = _DistOptRank(params, ref, dp_rank=0, dp_size=1)
    got = sp.full_masters(single, params, all_reduce_sum=lambda t: None)
    for p, m in zip(params, got, strict=True):
        assert torch.equal(m, ref[p])


def test_dp2_apply_writes_each_rank_range_and_the_model_copy():
    params = _params()
    zeros = {p: torch.zeros(p.shape) for p in params}
    ranks = [_DistOptRank(params, zeros, dp_rank=r, dp_size=2) for r in range(2)]
    targets = [t for t in _masters(params, seed=7).values()]
    for opt in ranks:
        assert sp.write_masters(opt, params, targets) is True
    gathered = []
    for opt in ranks:
        gathered.append(sp.full_masters(opt, params, all_reduce_sum=lambda t: None))
    for i, (p, t) in enumerate(zip(params, targets, strict=True)):
        assert torch.equal(gathered[0][i] + gathered[1][i], t)  # disjoint ranges, zero elsewhere
        assert torch.equal(p.detach(), t.to(torch.bfloat16))


def test_no_master_anywhere_is_still_refused():
    params = _params()
    with pytest.raises(sp.StatePluginError, match="no FP32 optimizer master"):
        sp.full_masters(None, params)
