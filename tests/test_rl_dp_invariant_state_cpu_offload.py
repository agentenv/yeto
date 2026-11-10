"""CPU tests: name-keyed optimizer state of a DistributedOptimizer under --optimizer-cpu-offload.

Vendored with yeto.rl.adapters.miles.dp_invariant_state from Miles fork 28bc6cc24
tests/fast/backends/megatron_utils/test_dp_invariant_state_cpu_offload.py (needs Megatron).

Megatron runs CPU offload as the precision-aware DistributedOptimizer over a HybridDeviceOptimizer (HDO). The HDO
here is CPU-constructible: it keeps the real HDO state plumbing (``_sync_*``, ``_update_fp32_params_by_new_state``,
``_move_new_state_to_right_device``) and the real DistributedOptimizer accessors; only the CUDA streams and the
d2h/h2d copies of ``HybridDeviceOptimizer.step`` are replaced by their CPU equivalent.
"""

from types import SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.miles import dp_invariant_state as dps

_dist = pytest.importorskip("megatron.core.optimizer.distrib_optimizer")
_hdo = pytest.importorskip("megatron.core.optimizer.cpu_offloading")

LR, WD = 0.1, 0.01


class _CpuHdo(_hdo.HybridDeviceOptimizer):
    """A HybridDeviceOptimizer with offload fraction 1 and fp32 updates, stepped on the CPU."""

    def __init__(self, params):
        torch.optim.Optimizer.__init__(
            self,
            [{"params": params, "lr": LR, "weight_decay": WD}],
            defaults={
                "lr": LR,
                "offload_fraction": 1.0,
                "cpu_optimizer_cls": torch.optim.AdamW,
                "gpu_optimizer_cls": torch.optim.AdamW,
                "param_update_in_fp32": True,
                "pin_cpu_grads": False,
                "pin_cpu_params": False,
                "overlap_cpu_optimizer_d2h_h2d": False,
            },
        )
        self.param_update_in_fp32 = True
        # As in HybridDeviceOptimizer._get_sub_optimizer_param_groups: a low-precision parameter gets an fp32
        # copy; a native fp32 parameter gets only its offloaded host copy (taken now, at build time).
        self.param_to_fp32_param = {p: p.detach().float().clone() for p in params if p.dtype != torch.float32}
        self.gpu_params_map_cpu_copy = {p: p.detach().clone() for p in params if p.dtype == torch.float32}
        self.fp32_param_to_orig_param = {v: k for k, v in self.param_to_fp32_param.items()}
        self.param_to_inner_param = {p: self.param_to_fp32_param.get(p, self.gpu_params_map_cpu_copy.get(p))
                                     for p in params}
        self.inner_param_to_orig_param = {v: k for k, v in self.param_to_inner_param.items()}
        inner = list(self.param_to_inner_param.values())
        self.cpu_optimizers = [torch.optim.AdamW([{"params": inner, "lr": LR, "weight_decay": WD}], foreach=False)]
        self.gpu_optimizer = None

    def step(self, closure=None):
        self._sync_hdo_param_groups_to_sub_optimizers()
        for param, inner in self.param_to_inner_param.items():
            inner.grad = None if param.grad is None else param.grad.float()
        self.cpu_optimizers[0].step()
        with torch.no_grad():
            for param, inner in self.param_to_inner_param.items():
                param.copy_(inner)  # the copy-back hook: the HDO writes into the parameters it was built over
        self._sync_sub_optimizers_state_to_hdo()


class _OffloadDistOpt:
    """One DP rank of a precision-aware DistributedOptimizer with CPU offload over a flat bf16 buffer.

    As in Megatron, the HDO is built over bf16 views of this rank's ranges of the model parameters.
    """

    _get_main_param_and_optimizer_states = _dist.DistributedOptimizer._get_main_param_and_optimizer_states
    _set_main_param_and_optimizer_states = _dist.DistributedOptimizer._set_main_param_and_optimizer_states
    _init_optimizer_states_with_dummy_values = _dist.DistributedOptimizer._init_optimizer_states_with_dummy_values

    def __init__(self, params, *, dp_rank, dp_size, exp_avg_dtype=torch.float32):
        index_map, offset = {}, 0
        for p in params:
            index_map[p] = (offset, offset + p.numel(), 0)
            offset += p.numel()
        shard = -(-offset // dp_size)
        world = _dist.Range(dp_rank * shard, min((dp_rank + 1) * shard, offset))
        self.param_map = _dist.DistributedOptimizer._build_model_gbuf_param_range_map(index_map, world, 0)
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": self.param_map}]}]
        self.model_param_group_index_map, shards = {}, []
        for p, ranges in self.param_map.items():
            r = ranges["param"]
            self.model_param_group_index_map[p] = (0, len(shards))
            shards.append(p.detach().view(-1)[r.start : r.end])
        self.config = SimpleNamespace(
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=True,
            optimizer_cpu_offload=True,
            main_params_dtype=torch.float32,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=torch.float32,
        )
        self.optimizer = _CpuHdo(shards)

    def step(self, full_grads):
        for p, ranges in self.param_map.items():
            gi, go = self.model_param_group_index_map[p]
            r = ranges["param"]
            sharded = self.optimizer.param_groups[gi]["params"][go]
            sharded.grad = full_grads[p].reshape(-1)[r.start : r.end].to(sharded.dtype)
        self.optimizer.step()

    def gather_masters_into(self, full):
        for p, ranges in self.param_map.items():
            gi, go = self.model_param_group_index_map[p]
            r = ranges["param"]
            sharded = self.optimizer.param_groups[gi]["params"][go]
            full[p].reshape(-1)[r.start : r.end] = self.optimizer.param_to_inner_param[sharded].detach()


class _Model(torch.nn.Module):
    """bf16 parameters, or native fp32 ones (Megatron keeps e.g. KDA ``A_log`` and mHC ``alpha_*`` in fp32;
    the distributed optimizer puts each dtype in its own buffer, so one test model holds one dtype)."""

    def __init__(self, seed, dtype=torch.bfloat16):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(3, 5, generator=g).to(dtype))
        self.b = torch.nn.Parameter(torch.randn(7, generator=g).to(dtype))


def _params(model):
    return [p for _, p in model.named_parameters()]


def _grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    return {p: torch.randn(p.shape, generator=g) for p in _params(model)}


def _export(model, opt):
    return dps.export_named_optimizer_state(opt, list(model.named_parameters()))


def _load(model, opt, shards):
    named = list(model.named_parameters())
    dps.load_named_optimizer_state(opt, named, dps.merge_named_optimizer_states(shards))


def _reference(model):
    mains = [p.detach().float().reshape(-1).clone() for p in _params(model)]
    opt = torch.optim.AdamW([{"params": mains, "lr": LR, "weight_decay": WD}], foreach=False)

    def step(grads):
        for m, p in zip(mains, _params(model), strict=True):
            m.grad = grads[p].reshape(-1).to(p.dtype).float()
        opt.step()

    return mains, opt, step


class TestCpuOffloadRoundTrip:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "native_fp32"])
    @pytest.mark.parametrize("save_dp, load_dp", [(2, 2), (1, 1), (2, 1), (1, 3)])
    def test_the_next_step_after_a_restore_matches_an_unsharded_run(self, save_dp, load_dp, dtype):
        model = _Model(0, dtype)
        ref_mains, ref_opt, ref_step = _reference(model)
        savers = [_OffloadDistOpt(_params(model), dp_rank=r, dp_size=save_dp) for r in range(save_dp)]
        for seed in (1, 2):
            grads = _grads(model, seed)
            for opt in savers:
                opt.step(grads)
            ref_step(grads)
        shards = [_export(model, opt) for opt in savers]
        assert all(e["scalars"]["step"].item() == 2 for s in shards for e in s["entries"].values())
        saved_model = {n: p.detach().clone() for n, p in model.named_parameters()}

        # A rebuilt trainer, in Miles' order: model and a fresh HDO (no state; its host copies are snapshots of
        # the freshly initialized weights) are built first, then the restore writes the model weights.
        target = _Model(7, dtype)
        loaders = [_OffloadDistOpt(_params(target), dp_rank=r, dp_size=load_dp) for r in range(load_dp)]
        with torch.no_grad():
            for (_, p), (_, src) in zip(target.named_parameters(), model.named_parameters(), strict=True):
                p.copy_(src)
        for opt in loaders:
            assert not opt.optimizer.state
            _load(target, opt, shards)
        for name, p in target.named_parameters():
            # the dummy step of the load (zero grads, weight decay) must not touch the model
            torch.testing.assert_close(p.detach(), saved_model[name], rtol=0, atol=0)
        for opt in loaders:
            for sub in opt.optimizer.sub_optimizers:
                assert {s["step"].item() for s in sub.state.values()} == {2.0}

        grads = _grads(model, 3)
        for opt in loaders:
            opt.step({tp: grads[sp] for tp, sp in zip(_params(target), _params(model), strict=True)})
        ref_step(grads)
        gathered = {p: torch.full(p.shape, float("nan")) for p in _params(target)}
        for opt in loaders:
            opt.gather_masters_into(gathered)
        for p, ref in zip(_params(target), ref_mains, strict=True):
            torch.testing.assert_close(gathered[p].reshape(-1), ref.detach(), rtol=0, atol=0)

    @staticmethod
    def _restored(dtype, save_dp=2, load_dp=2):
        """(saved shards, restored loaders, target model) after two steps, a save and a restore."""
        model = _Model(0, dtype)
        savers = [_OffloadDistOpt(_params(model), dp_rank=r, dp_size=save_dp) for r in range(save_dp)]
        for seed in (1, 2):
            grads = _grads(model, seed)
            for opt in savers:
                opt.step(grads)
        shards = [_export(model, opt) for opt in savers]
        target = _Model(7, dtype)
        loaders = [_OffloadDistOpt(_params(target), dp_rank=r, dp_size=load_dp) for r in range(load_dp)]
        with torch.no_grad():
            for (_, p), (_, src) in zip(target.named_parameters(), model.named_parameters(), strict=True):
                p.copy_(src)
        for opt in loaders:
            _load(target, opt, shards)
        return shards, loaders, target

    @staticmethod
    def _same_state(a_shards, b_shards):
        a, b = dps.merge_named_optimizer_states(a_shards), dps.merge_named_optimizer_states(b_shards)
        return a.keys() == b.keys() and all(
            a[n]["tensors"].keys() == b[n]["tensors"].keys()
            and all(torch.equal(a[n]["tensors"][k], b[n]["tensors"][k]) for k in a[n]["tensors"])
            for n in a
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "native_fp32"])
    def test_the_export_after_a_restore_equals_the_saved_state(self, dtype):
        shards, loaders, target = self._restored(dtype)
        assert self._same_state(shards, [_export(target, opt) for opt in loaders])

    def test_a_restore_that_leaves_the_stepped_tensors_stale_is_seen_by_the_export(self, monkeypatch):
        """Regression (T-C4 N7.6c): inject the T-G3 bug (the sub-optimizer keeps stepping its build-time host copy
        of a native fp32 parameter). The export, which feeds the cut's restore digest, must differ from the cut;
        the HDO's own "master_param" still holds the loaded value, so a digest over it would not see the bug."""
        monkeypatch.setattr(dps, "_bind_masters_to_stepped_tensors", lambda hdo: None)
        shards, loaders, target = self._restored(torch.float32)
        assert not self._same_state(shards, [_export(target, opt) for opt in loaders])
        hdo_masters = {}
        for opt in loaders:
            for p, ranges in opt.param_map.items():
                gi, go = opt.model_param_group_index_map[p]
                sharded = opt.optimizer.param_groups[gi]["params"][go]
                hdo_masters[(id(p), ranges["param"].start)] = opt.optimizer.state[sharded]["master_param"]
        merged = dps.merge_named_optimizer_states(shards)
        names = {id(p): n for n, p in target.named_parameters()}
        for (pid, start), master in hdo_masters.items():
            saved = merged[names[pid]]["tensors"]["param"][start : start + master.numel()]
            assert torch.equal(master.reshape(-1), saved)  # the old read path: bug invisible

    def test_a_fresh_optimizer_exports_its_ranges_without_creating_state(self):
        model = _Model(0)
        opt = _OffloadDistOpt(_params(model), dp_rank=1, dp_size=2)
        fresh = _export(model, opt)
        assert not opt.optimizer.state
        assert {n: (e["start"], e["end"]) for n, e in fresh["entries"].items()} == {
            n: (opt.param_map[p]["param"].start, opt.param_map[p]["param"].end) for n, p in model.named_parameters()
            if p in opt.param_map
        }

    def test_the_export_holds_hyperparameters_only_and_loads_weights_only(self, tmp_path):
        model = _Model(0)
        opt = _OffloadDistOpt(_params(model), dp_rank=0, dp_size=1)
        opt.step(_grads(model, 1))
        named = _export(model, opt)
        hyper = next(iter(named["entries"].values()))["hyper"]
        assert hyper["lr"] == LR and not set(hyper) & dps._HDO_WIRING_KEYS
        torch.save(named, tmp_path / "state.pt")
        torch.load(tmp_path / "state.pt", weights_only=True)  # no classes inside a shard


class TestRefusals:
    def test_low_precision_moments_are_refused(self):
        model = _Model(0)
        opt = _OffloadDistOpt(_params(model), dp_rank=0, dp_size=1, exp_avg_dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="exp_avg_dtype"):
            _export(model, opt)

    def test_precision_aware_without_offload_is_refused(self):
        model = _Model(0)
        opt = _OffloadDistOpt(_params(model), dp_rank=0, dp_size=1)
        opt.optimizer = torch.optim.AdamW([torch.zeros(1)])
        with pytest.raises(NotImplementedError, match="precision-aware optimizer on GPU"):
            _export(model, opt)
