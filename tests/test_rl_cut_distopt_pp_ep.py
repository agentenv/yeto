"""Same-shape cut under DistributedOptimizer with PP2 (DP1/DP2) and EP2 (dense DP2, expert DP1).

CPU protocol test, NOT GPU evidence. Every rank is simulated in one process
with duck-typed Megatron DistOpt leaves (``gbuf_ranges``, main-param shards,
Adam moments) and the REAL fork-M5 helpers (``dp_invariant_state`` at the
Miles image pin, read from a local Miles checkout; skipped when absent).
save -> fresh rebuild -> restore -> re-export must be element-wise equal, and
the full FP32 masters gathered for the republish (state_plugin.full_masters)
must equal the pre-cut ones. A different DP/EP layout is refused.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.engine.miles_adapter import cut_plugin as cp
from yeto.rl.engine.miles_adapter import state_plugin as sp

MILES_CHECKOUT = os.environ.get("YETO_MILES_CHECKOUT", "/home/michael/work/miles-m3")
MILES_PIN = "c35702e"
DPS_PATH = "miles/backends/megatron_utils/lora/dp_invariant_state.py"


def _load_dps():
    try:
        src = subprocess.run(["git", "-C", MILES_CHECKOUT, "show", f"{MILES_PIN}:{DPS_PATH}"],
                             capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    spec = importlib.util.spec_from_loader("_yeto_test_dps", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(src, DPS_PATH, "exec"), module.__dict__)  # noqa: S102 - pinned local source
    return module


DPS = _load_dps()
needs_dps = pytest.mark.skipif(DPS is None, reason="Miles checkout with the fork-M5 helpers not available")


class _Range:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _Leaf:
    """One DistOpt instance: owns ``[lo, hi)`` of the flattened params of its DP group."""

    def __init__(self, params, *, dp_rank, dp_size, group, pad=0):
        sizes = [p.numel() for p in params]
        total = sum(sizes) + pad
        shard = -(-total // dp_size)
        lo, hi = dp_rank * shard, min((dp_rank + 1) * shard, total)
        param_map, mains, off = {}, [], 0
        for p, n in zip(params, sizes, strict=True):
            s, e = max(lo, off) - off, min(hi, off + n) - off
            if e > s:
                param_map[p] = {"param": _Range(s, e)}
                mains.append(p.detach().float().reshape(-1)[s:e].clone())
            off += n
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": param_map}]}]
        self.model_param_group_index_map = {p: (0, i) for i, p in enumerate(param_map)}
        self.buffers = [SimpleNamespace(param_index_map={p: None for p in params})]
        self.data_parallel_group = group
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        self.optimizer = SimpleNamespace(param_groups=[{"params": mains, "lr": 0.01, "betas": (0.9, 0.95)}],
                                         state={})
        self._ranges = {id(p): (r["param"].start, r["param"].end) for p, r in param_map.items()}

    def _main(self, p):
        g, i = self.model_param_group_index_map[p]
        return self.optimizer.param_groups[g]["params"][i]

    def _get_main_param_and_optimizer_states(self, p):
        main = self._main(p)
        st = self.optimizer.state.setdefault(main, {})  # Megatron indexes a defaultdict
        return {"param": main, **{k: st[k] for k in ("exp_avg", "exp_avg_sq") if k in st}}

    def _set_main_param_and_optimizer_states(self, p, tensors):
        main = self._main(p)
        st = self.optimizer.state[main]
        for key, value in tensors.items():
            if key == "param":
                main.copy_(value)
            elif key in st:
                st[key].copy_(value)

    def _init_optimizer_states_with_dummy_values(self):
        for main in self.optimizer.param_groups[0]["params"]:
            self.optimizer.state[main] = {"exp_avg": torch.zeros_like(main), "exp_avg_sq": torch.zeros_like(main)}

    def train(self, seed):
        g = torch.Generator().manual_seed(seed)
        for main in self.optimizer.param_groups[0]["params"]:
            main.add_(torch.randn(main.shape, generator=g))
            self.optimizer.state[main] = {"exp_avg": torch.randn(main.shape, generator=g),
                                          "exp_avg_sq": torch.rand(main.shape, generator=g)}


class _Backend(cp.MilesCutBackend):
    def __init__(self, coord):
        self._coord = coord

    def coord(self):
        return dict(self._coord)

    def is_adapter(self, name):
        return "lora" in name

    def _dps(self):
        return DPS

    def megatron_counters(self):
        return {}


class _Sched:
    def __init__(self):
        self.num_steps = 0

    def state_dict(self):
        return {"num_steps": self.num_steps, "max_lr": 0.01}

    def load_state_dict(self, state):
        self.num_steps += state["num_steps"]


def _module(named):
    m = torch.nn.Module()
    for name, t in named:
        m.register_parameter(name, torch.nn.Parameter(t.clone().to(torch.bfloat16)))
    return m


def _tensors(seed, shapes):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=g) for s in shapes]


def _world(layout):
    """Freshly built ranks. layout: 'pp2dp1' | 'pp2dp2' | 'ep2' (dense DP2, expert DP1)."""
    ranks = []
    if layout.startswith("pp2"):
        dp_size = 1 if layout == "pp2dp1" else 2
        for pp in range(2):
            init = _tensors(100 + pp, ((3, 5), (7,), (2, 4)))
            for dp in range(dp_size):
                model = _module([(f"lora_{i}", t) for i, t in enumerate(init)])
                params = list(model.parameters())
                leaf = _Leaf(params, dp_rank=dp, dp_size=dp_size, group=("dp", pp))
                coord = {"global_rank": pp * dp_size + dp, "tp": 0, "pp": pp, "dp": dp, "dp_size": dp_size,
                         "tp_size": 1, "pp_size": 2, "cp_size": 1, "ep_size": 1}
                ranks.append(_rank(model, leaf, coord, pp=2))
    else:
        dense = _tensors(7, ((3, 5), (7,), (2, 4)))
        for ep in range(2):
            expert = _tensors(20 + ep, ((4, 3), (5,)))
            model = _module([(f"lora_d{i}", t) for i, t in enumerate(dense)]
                            + [(f"experts_lora_{i}", t) for i, t in enumerate(expert)])
            params = list(model.parameters())
            d = _Leaf(params[:3], dp_rank=ep, dp_size=2, group=("dense-dp", (0, 1)))
            e = _Leaf(params[3:], dp_rank=0, dp_size=1, group=("expert-dp", (ep,)))
            coord = {"global_rank": ep, "tp": 0, "pp": 0, "dp": ep, "dp_size": 2, "tp_size": 1, "pp_size": 1,
                     "cp_size": 1, "ep_size": 2, "ep": ep, "etp_size": 1, "edp": 0, "edp_size": 1}
            ranks.append(_rank(model, SimpleNamespace(chained_optimizers=[d, e]), coord, ep=2))
    return ranks


def _rank(model, optimizer, coord, *, pp=1, ep=1):
    args = SimpleNamespace(fp16=False, use_distributed_optimizer=True, pipeline_model_parallel_size=pp,
                           expert_model_parallel_size=ep, tensor_model_parallel_size=1)
    return SimpleNamespace(args=args, model=[model], optimizer=optimizer, opt_param_scheduler=_Sched(),
                           _yeto_cut_backend=_Backend(coord))


def _leaves(rank):
    return list(getattr(rank.optimizer, "chained_optimizers", None) or [rank.optimizer])


def _train(ranks):
    for r, rank in enumerate(ranks):
        for k, leaf in enumerate(_leaves(rank)):
            leaf.train(1000 + 10 * r + k)
        rank.opt_param_scheduler.num_steps = 8


def _gathered_masters(ranks):
    """Full FP32 masters per rank via state_plugin.full_masters, group sums simulated in one process."""
    record = {}
    for rank in ranks:
        calls = [0]

        def rec(flat, leaf, calls=calls):
            calls[0] += 1
            record.setdefault((leaf.data_parallel_group, calls[0]), []).append(flat.clone())

        sp.full_masters(rank.optimizer, list(rank.model[0].parameters()), reduce=rec)
    out = []
    for rank in ranks:
        n = [0]

        def red(flat, leaf, n=n):
            n[0] += 1
            flat.copy_(sum(record[(leaf.data_parallel_group, n[0])]))

        out.append(sp.full_masters(rank.optimizer, list(rank.model[0].parameters()), reduce=red))
    return out


def _export(rank):
    named = [(n, p) for n, p in rank.model[0].named_parameters()]
    return rank._yeto_cut_backend.export_optimizer(rank.optimizer, named)


def _save(ranks, tmp_path):
    files = []
    for rank in ranks:
        s = cp.save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
        assert "refused" not in s, s
        files.append(s)
    return files


@needs_dps
@pytest.mark.parametrize("layout", ["pp2dp1", "pp2dp2", "ep2"])
def test_save_restore_reexport_is_elementwise_equal(tmp_path, layout):
    ranks = _world(layout)
    _train(ranks)
    before = [_export(r) for r in ranks]
    masters = _gathered_masters(ranks)
    files = _save(ranks, tmp_path)
    if layout == "ep2":  # each EP rank keeps its own expert params (same names, different values)
        assert not torch.equal(masters[0][3], masters[1][3])
    fresh = _world(layout)
    for rank in fresh:
        out = cp.restore_cut_shard(rank, directory=str(tmp_path), files=files, cut_id="c1")
        assert "refused" not in out, out
        assert "diff" not in out, out
    for rank, want in zip(fresh, before, strict=True):
        got = _export(rank)
        assert cp.optimizer_ranges(got) == cp.optimizer_ranges(want)
        assert cp.state_digest(got) == cp.state_digest(want)
        for name, e in want["entries"].items():
            for key, t in e["tensors"].items():
                assert torch.equal(got["entries"][name]["tensors"][key], t), (name, key)
    for got, want in zip(_gathered_masters(fresh), masters, strict=True):
        for a, b in zip(got, want, strict=True):
            assert torch.equal(a, b)


@needs_dps
def test_ep_layout_change_is_refused(tmp_path):
    ranks = _world("ep2")
    _train(ranks)
    files = _save(ranks, tmp_path)
    fresh = _world("ep2")
    fresh[0]._yeto_cut_backend._coord["ep_size"] = 4
    out = cp.restore_cut_shard(fresh[0], directory=str(tmp_path), files=files, cut_id="c1")
    assert "same-shape" in out["refused"] and out["refusal_kind"] == "refused"


@needs_dps
def test_dp_layout_change_is_refused(tmp_path):
    ranks = _world("pp2dp2")
    _train(ranks)
    files = _save(ranks, tmp_path)
    fresh = _world("pp2dp1")
    out = cp.restore_cut_shard(fresh[0], directory=str(tmp_path), files=files, cut_id="c1")
    assert "refused" in out


@needs_dps
def test_changed_shard_ranges_are_refused_before_any_write(tmp_path):
    ranks = _world("pp2dp2")
    _train(ranks)
    files = _save(ranks, tmp_path)
    fresh = _world("pp2dp2")
    rank = fresh[0]
    model = rank.model[0]
    rank.optimizer = _Leaf(list(model.parameters()), dp_rank=0, dp_size=2, group=("dp", 0), pad=6)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    out = cp.restore_cut_shard(rank, directory=str(tmp_path), files=files, cut_id="c1")
    assert "ranges differ" in out["refused"]
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p.detach())


def test_local_states_cover_with_fillers():
    t = torch.arange(3.0)
    state = {"format": "f", "entries": {"a": {"numel": 10, "shape": (10,), "start": 4, "end": 7,
                                              "tensors": {"param": t}, "scalars": {}, "hyper": {}},
                                        "b": {"numel": 3, "shape": (3,), "start": 0, "end": 3,
                                              "tensors": {"param": t}, "scalars": {}, "hyper": {}}}}
    states = cp.local_states(state)
    spans = sorted((n, e["start"], e["end"]) for s in states for n, e in s["entries"].items())
    assert spans == [("a", 0, 4), ("a", 4, 7), ("a", 7, 10), ("b", 0, 3)]
    assert all(len(s["entries"]) and s["format"] == "f" for s in states)
    assert cp.range_problems(state, state) == []
    other = {"entries": {"a": {**state["entries"]["a"], "start": 5}}}
    assert len(cp.range_problems(state, other)) == 2
