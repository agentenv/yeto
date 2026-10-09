"""M2 (EP>1, dense DP>1, DistributedOptimizer): sharded masters under a ChainedOptimizer.

Two leaves per rank: a dense leaf sharding over the dense DP group (size 2) and
an expert leaf over the expert DP group (size 1; each EP rank holds its own
experts). Parameters carry no ``main_param``; masters live only in the leaf
shards. World 2 is simulated in one process: pass 1 records each rank's
contribution per (group, call), pass 2 reduces with the recorded group sums.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.miles import state_plugin as sp

DENSE_GROUP, EXPERT_GROUP = ("dense-dp", (0, 1)), None  # expert DP groups are per rank


class _Range:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Leaf:
    def __init__(self, params, full_main, *, dp_rank, dp_size, group, megatron_handles=False):
        sizes = [p.numel() for p in params]
        total = sum(sizes)
        shard = -(-total // dp_size)
        lo, hi = dp_rank * shard, min((dp_rank + 1) * shard, total)
        param_map, self._mains, off = {}, {}, 0
        for p, n in zip(params, sizes, strict=True):
            s, e = max(lo, off) - off, min(hi, off + n) - off
            if e > s:
                param_map[p] = {"param": _Range(s, e)}
                self._mains[id(p)] = full_main[p].reshape(-1)[s:e].clone()
                if megatron_handles:  # real Megatron: main_param is the owned FP32 shard
                    p.main_param, p.main_param_sharded = self._mains[id(p)], True
            off += n
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": param_map}]}]
        self.model_param_group_index_map = {p: (0, i) for i, p in enumerate(param_map)}
        self.buffers = [SimpleNamespace(param_index_map={p: None for p in params})]
        self.data_parallel_group = group
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)

    def _get_main_param_and_optimizer_states(self, param):
        return {"param": self._mains[id(param)]}


def _bf16(shapes, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g).to(torch.bfloat16)) for s in shapes]


def _ref(params, seed):
    g = torch.Generator().manual_seed(seed)
    return {p: torch.randn(p.shape, generator=g, dtype=torch.float32) for p in params}


def _world(zero=False, megatron_handles=False):
    dense = _bf16(((3, 5), (7,), (2, 4)), 0)  # attention LoRA, replicated across EP ranks
    experts = [_bf16(((4, 3),), 10 + r) for r in range(2)]  # EP rank r's own expert params
    ref = {**_ref(dense, 1), **_ref(experts[0], 2), **_ref(experts[1], 3)}
    if zero:
        ref = {p: torch.zeros(p.shape) for p in ref}
    ranks = []
    for r in range(2):
        d = _Leaf(dense, ref, dp_rank=r, dp_size=2, group=DENSE_GROUP, megatron_handles=megatron_handles)
        e = _Leaf(
            experts[r], ref, dp_rank=0, dp_size=1, group=("expert-dp", (r,)), megatron_handles=megatron_handles
        )
        ranks.append((SimpleNamespace(chained_optimizers=[d, e]), dense + experts[r]))
    return ranks, ref


def _gather_all(ranks):
    seen = []
    record = {}
    for r, (opt, params) in enumerate(ranks):
        calls = []

        def rec(flat, leaf, calls=calls):
            calls.append(leaf.data_parallel_group)
            record.setdefault((leaf.data_parallel_group, len(calls)), []).append(flat.clone())

        sp.full_masters(opt, params, reduce=rec)
        seen.append(calls)
    outs = []
    for opt, params in ranks:
        n = [0]

        def red(flat, leaf, n=n):
            n[0] += 1
            flat.copy_(sum(record[(leaf.data_parallel_group, n[0])]))

        outs.append(sp.full_masters(opt, params, reduce=red))
    return outs, seen


@pytest.mark.parametrize("megatron_handles", [False, True])
def test_chained_export_gathers_full_masters_in_the_right_groups(megatron_handles):
    ranks, ref = _world(megatron_handles=megatron_handles)
    assert all(sp.needs_gather(opt, params) for opt, params in ranks)
    outs, seen = _gather_all(ranks)
    assert seen == [[DENSE_GROUP, ("expert-dp", (0,))], [DENSE_GROUP, ("expert-dp", (1,))]]
    for (opt, params), got in zip(ranks, outs, strict=True):
        for p, m in zip(params, got, strict=True):
            assert m.dtype == torch.float32 and torch.equal(m, ref[p])


@pytest.mark.parametrize("megatron_handles", [False, True])
def test_chained_apply_then_export_roundtrips(megatron_handles):
    ranks, _ = _world(zero=True, megatron_handles=megatron_handles)
    targets = {p: t for opt, params in ranks for p, t in _ref(params, 99).items()}
    for opt, params in ranks:
        assert sp.write_masters(opt, params, [targets[p] for p in params]) is True
        for p in params:
            assert torch.equal(p.detach(), targets[p].to(torch.bfloat16))
    outs, _ = _gather_all(ranks)
    for (opt, params), got in zip(ranks, outs, strict=True):
        for p, m in zip(params, got, strict=True):
            assert torch.equal(m, targets[p])


def test_masters_override_in_module_context():
    ranks, ref = _world()
    opt, params = ranks[0]
    module = torch.nn.Module()
    for i, p in enumerate(params):
        module.register_parameter(f"p{i}", p)
    assert sp.needs_gather(opt, params)
    with pytest.raises(sp.StatePluginError, match="no FP32 optimizer master"):
        with sp.masters_as_module_parameters([module], params):
            pass
    masters = [ref[p] for p in params]
    with sp.masters_as_module_parameters([module], params, masters):
        for i, p in enumerate(params):
            assert torch.equal(module._parameters[f"p{i}"], ref[p])
    sp.assert_grad_flow_intact([module], params)


def test_default_reduce_uses_the_owning_leaf_group(monkeypatch):
    import torch.distributed as dist

    ranks, _ = _world()
    opt, params = ranks[1]
    groups = []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None, group=None: groups.append(group))
    sp.full_masters(opt, params)
    assert groups == [DENSE_GROUP, ("expert-dp", (1,))]


def test_unsharded_masters_need_no_gather():
    p = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    p.main_param = torch.ones(3)
    ranks, _ = _world()
    assert not sp.needs_gather(ranks[0][0], [p])
    assert not sp.needs_gather(None, [p])


def test_sharded_main_param_is_not_a_complete_master():
    """M2 rerun: Megatron's DistributedOptimizer sets ``main_param`` to the owned FP32 shard."""
    p = torch.nn.Parameter(torch.zeros(6, dtype=torch.bfloat16))
    p.main_param, p.main_param_sharded = torch.ones(3), True
    assert not sp.has_complete_master(p)
    q = torch.nn.Parameter(torch.zeros(6, dtype=torch.bfloat16))
    q.main_param = torch.ones(3)  # partial without the flag
    assert not sp.has_complete_master(q)
    assert sp.has_complete_master(torch.nn.Parameter(torch.zeros(2)))
