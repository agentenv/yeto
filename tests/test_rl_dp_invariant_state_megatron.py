"""The vendored DP-invariant optimizer state on Megatron's DistributedOptimizer (CPU-constructible).

Vendored with the module from Miles fork 28bc6cc24
tests/fast/backends/megatron_utils/test_lora_dp_invariant_state.py. Needs
Megatron (``megatron.core.optimizer.distrib_optimizer``); the save/load
round trips through the fork's LoRA checkpoint (``save_lora_checkpoint`` /
``load_lora_adapter``) and its argument validation exist only on the ports-v1
fork and skip elsewhere.
"""

import random
import shutil
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yeto.rl.adapters.miles import dp_invariant_state as dps
from test_rl_dp_invariant_state import _NAMES, _Model, _export, _load, _named, _optimizer, _step  # noqa: F401

_dist = pytest.importorskip("megatron.core.optimizer.distrib_optimizer")


def _fork_lora_utils():
    """The ports-v1 fork's LoRA checkpoint helpers; None on upstream / k31 Miles."""
    try:
        from miles.backends.megatron_utils.lora import utils as lora_utils
    except ImportError:
        return None
    return lora_utils if hasattr(lora_utils, "save_lora_checkpoint") else None


lora_utils = _fork_lora_utils()
needs_fork_lora_checkpoint = pytest.mark.skipif(
    lora_utils is None, reason="the fork's LoRA checkpoint (save_lora_checkpoint / load_lora_adapter) is not here")

class _DistOpt:
    """One DP rank of a Megatron DistributedOptimizer over one flat bf16 buffer holding ``params`` in order.

    The ranges come from ``DistributedOptimizer._build_model_gbuf_param_range_map`` and the state accessors are the
    real DistributedOptimizer methods; only the CUDA grad buffer is replaced by the index arithmetic it performs.
    """

    _get_main_param_and_optimizer_states = _dist.DistributedOptimizer._get_main_param_and_optimizer_states
    _set_main_param_and_optimizer_states = _dist.DistributedOptimizer._set_main_param_and_optimizer_states
    _init_optimizer_states_with_dummy_values = _dist.DistributedOptimizer._init_optimizer_states_with_dummy_values

    def __init__(self, params, full_main, *, dp_rank, dp_size, lr=0.1):
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
        self.param_map = param_map
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        self.optimizer = torch.optim.AdamW(
            [{"params": mains, "lr": lr, "weight_decay": 0.01}] if mains else [{"params": [torch.zeros(0)]}],
            foreach=False,
        )

    def step(self, full_grads):
        for p, ranges in self.param_map.items():
            gi, go = self.model_param_group_index_map[p]
            r = ranges["param"]
            self.optimizer.param_groups[gi]["params"][go].grad = full_grads[p].reshape(-1)[r.start : r.end].clone()
        self.optimizer.step()

    def gather_into(self, full):
        for p, ranges in self.param_map.items():
            gi, go = self.model_param_group_index_map[p]
            r = ranges["param"]
            full[p].reshape(-1)[r.start : r.end] = self.optimizer.param_groups[gi]["params"][go].detach()

    def state_dict(self):
        return {}


def _params_of(model):
    return [p for _, p in model.named_parameters()]


def _grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    return {p: torch.randn(p.shape, generator=g) for p in _params_of(model)}


class TestDistributedOptimizerGatherReshard:
    """Save at one DP size, gather through the files, reshard at another; the next step must match exactly."""

    def _reference(self, model, init):
        mains = [init[p].clone().reshape(-1) for p in _params_of(model)]
        opt = torch.optim.AdamW([{"params": mains, "lr": 0.1, "weight_decay": 0.01}], foreach=False)

        def step(grads):
            for m, p in zip(mains, _params_of(model), strict=True):
                m.grad = grads[p].reshape(-1).clone()
            opt.step()

        return mains, step

    @pytest.mark.parametrize("save_dp, load_dp", [(2, 1), (2, 3), (1, 4), (4, 2)])
    def test_the_next_step_after_a_dp_change_matches_an_unsharded_run(self, save_dp, load_dp):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        ref_mains, ref_step = self._reference(model, init)

        savers = [_DistOpt(_params_of(model), init, dp_rank=r, dp_size=save_dp) for r in range(save_dp)]
        for seed in (1, 2):
            for opt in savers:
                opt.step(_grads(model, seed))
            ref_step(_grads(model, seed))
        shards = [_export(model, opt) for opt in savers]

        zeros = {p: torch.zeros(p.shape) for p in _params_of(model)}
        loaders = [_DistOpt(_params_of(model), zeros, dp_rank=r, dp_size=load_dp) for r in range(load_dp)]
        for opt in loaders:
            _load(model, opt, shards)
            opt.step(_grads(model, 3))
        ref_step(_grads(model, 3))

        gathered = {p: torch.full(p.shape, float("nan")) for p in _params_of(model)}
        for opt in loaders:
            opt.gather_into(gathered)
        for p, ref in zip(_params_of(model), ref_mains, strict=True):
            torch.testing.assert_close(gathered[p].reshape(-1), ref.detach(), rtol=0, atol=0)

    def test_a_missing_dp_shard_is_detected(self):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        savers = [_DistOpt(_params_of(model), init, dp_rank=r, dp_size=2) for r in range(2)]
        for opt in savers:
            opt.step(_grads(model, 1))
        with pytest.raises(dps.DpInvariantStateError, match="not fully covered"):
            dps.merge_named_optimizer_states([_export(model, savers[0])])

    def test_chained_optimizers_are_unwrapped(self):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        dist = _DistOpt(_params_of(model), init, dp_rank=0, dp_size=1)
        dist.step(_grads(model, 1))
        chained = SimpleNamespace(chained_optimizers=[dist])
        assert set(_export(model, chained)["entries"]) == {"lora_A", "lora_B"}




def _patch_parallel_state(monkeypatch, *, dp_rank: int, dp_size: int = 1, cp_size: int = 1, ep_size: int = 1):
    group = lambda r, n=1: SimpleNamespace(rank=r, size=n)  # noqa: E731
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(
            tp=group(0), pp=group(0), intra_dp=group(dp_rank, dp_size), cp=group(0, cp_size), ep=group(0, ep_size)
        ),
    )


class _AdapterModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.full((2, 2), value))
        self.lora_B = torch.nn.Parameter(torch.full((2, 2), value + 1))


@needs_fork_lora_checkpoint
class TestSaveLoadDpInvariant:
    def _save(self, model, optimizer, path, *, flag: bool):
        args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=False, lora_dp_invariant_state=flag)
        lora_utils.save_lora_checkpoint(
            [model],
            args,
            str(path),
            publisher=SimpleNamespace(write_adapter=lambda *_: None),
            optimizer=optimizer,
            opt_param_scheduler=SimpleNamespace(state_dict=lambda: {"lr": 0.5}),
            iteration=4,
        )

    def test_default_format_writes_no_dp_invariant_files(self, tmp_path, monkeypatch):
        _patch_parallel_state(monkeypatch, dp_rank=0)
        model = _AdapterModel(1.0)
        self._save(model, torch.optim.Adam(model.parameters()), tmp_path / "ckpt", flag=False)

        assert sorted(p.name for p in (tmp_path / "ckpt").iterdir()) == [
            "adapter_megatron_rank0.pt",
            "training_state_rank0.pt",
        ]

    def _save_dp2(self, tmp_path, monkeypatch):
        """Two DP ranks save into one directory; returns (model, optimizer, dir, RNG draw expected on dp rank 0)."""
        model = _AdapterModel(1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model.lora_A.grad = torch.ones(2, 2)
        model.lora_B.grad = torch.ones(2, 2)
        optimizer.step()

        merged_dir = tmp_path / "merged"
        merged_dir.mkdir()
        expected = None
        for dp_rank in (1, 0):
            _patch_parallel_state(monkeypatch, dp_rank=dp_rank, dp_size=2)
            torch.manual_seed(100 + dp_rank)
            if dp_rank == 0:
                expected = torch.rand(1)
                torch.manual_seed(100)
            self._save(model, optimizer, tmp_path / f"dp{dp_rank}", flag=True)
            for f in (tmp_path / f"dp{dp_rank}").iterdir():
                if "dp_invariant" in f.name or "named" in f.name:
                    shutil.copy(f, merged_dir / f.name)
        return model, optimizer, merged_dir, expected

    def _load(self, merged_dir, *, rng_policy="exact"):
        target = _AdapterModel(0.0)
        target_opt = torch.optim.Adam(target.parameters(), lr=0.1)
        scheduler_loads = []
        result = lora_utils.load_lora_adapter(
            [target],
            str(merged_dir),
            optimizer=target_opt,
            opt_param_scheduler=SimpleNamespace(load_state_dict=scheduler_loads.append),
            dp_invariant=True,
            rng_policy=rng_policy,
        )
        return result, target, target_opt, scheduler_loads

    def test_same_dp_size_restores_optimizer_and_exact_rng(self, tmp_path, monkeypatch):
        model, optimizer, merged_dir, expected = self._save_dp2(tmp_path, monkeypatch)
        assert sorted(p.name for p in merged_dir.iterdir()) == [
            "adapter_dp_invariant_tp0_pp0.pt",
            "training_state_named_tp0_pp0_dp0.pt",
            "training_state_named_tp0_pp0_dp1.pt",
        ]
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=2)
        torch.manual_seed(7)

        result, target, target_opt, scheduler_loads = self._load(merged_dir)

        assert result == (True, 4, True)
        torch.testing.assert_close(target.lora_A.data, model.lora_A.data)
        assert scheduler_loads == [{"lr": 0.5}]
        torch.testing.assert_close(torch.rand(1), expected, rtol=0, atol=0)

    def test_a_dp_change_loads_optimizer_by_name_under_keep_on_dp_change(self, tmp_path, monkeypatch):
        """DP=2 -> DP=1: optimizer state comes back by name; RNG stays the fresh one of the new process."""
        model, optimizer, merged_dir, _ = self._save_dp2(tmp_path, monkeypatch)
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)
        torch.manual_seed(7)
        fresh = torch.rand(1)
        torch.manual_seed(7)

        result, _, target_opt, _ = self._load(merged_dir, rng_policy="keep_on_dp_change")

        assert result == (True, 4, True)
        torch.testing.assert_close(torch.rand(1), fresh, rtol=0, atol=0)
        saved_state = optimizer.state_dict()["state"]
        loaded_state = target_opt.state_dict()["state"]
        for index in saved_state:
            torch.testing.assert_close(loaded_state[index]["exp_avg"], saved_state[index]["exp_avg"])

    def test_a_dp_change_is_refused_under_the_exact_rng_policy(self, tmp_path, monkeypatch):
        _, _, merged_dir, _ = self._save_dp2(tmp_path, monkeypatch)
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)

        with pytest.raises(dps.DpInvariantStateError, match="explicit policy"):
            self._load(merged_dir, rng_policy="exact")

    def test_an_unknown_rng_policy_is_refused(self, tmp_path, monkeypatch):
        _, _, merged_dir, _ = self._save_dp2(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="unknown RNG policy"):
            self._load(merged_dir, rng_policy="whatever")

    def test_dp_invariant_load_falls_back_to_rank_shards(self, tmp_path, monkeypatch):
        _patch_parallel_state(monkeypatch, dp_rank=0)
        model = _AdapterModel(3.0)
        self._save(model, torch.optim.Adam(model.parameters()), tmp_path / "ckpt", flag=False)
        target = _AdapterModel(0.0)

        loaded, iteration, _ = lora_utils.load_lora_adapter([target], str(tmp_path / "ckpt"), dp_invariant=True)

        assert (loaded, iteration) == (True, None)
        torch.testing.assert_close(target.lora_A.data, model.lora_A.data)

    @pytest.mark.parametrize("sizes", [dict(ep_size=2), dict(cp_size=2)])
    def test_expert_and_context_parallelism_are_refused_at_runtime(self, tmp_path, monkeypatch, sizes):
        _patch_parallel_state(monkeypatch, dp_rank=0, **sizes)
        model = _AdapterModel(1.0)
        with pytest.raises(dps.DpInvariantStateError, match="EP=1 and CP=1"):
            self._save(model, torch.optim.Adam(model.parameters()), tmp_path / "ckpt", flag=True)


@needs_fork_lora_checkpoint
class TestDistributedOptimizerEndToEnd:
    """save_lora_checkpoint at DP=2 with a bf16 model under a DistributedOptimizer, load_lora_adapter at DP=1."""

    def test_dp2_to_dp1_through_the_checkpoint_directory(self, tmp_path, monkeypatch):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        savers = [_DistOpt(_params_of(model), init, dp_rank=r, dp_size=2) for r in range(2)]
        for opt in savers:
            opt.step(_grads(model, 1))
        full = {p: torch.zeros(p.shape) for p in _params_of(model)}
        for opt in savers:
            opt.gather_into(full)
        for p in _params_of(model):
            p.data.copy_(full[p])

        merged_dir = tmp_path / "merged"
        merged_dir.mkdir()
        for dp_rank, opt in enumerate(savers):
            _patch_parallel_state(monkeypatch, dp_rank=dp_rank, dp_size=2)
            args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=False, lora_dp_invariant_state=True)
            out = tmp_path / f"dp{dp_rank}"
            lora_utils.save_lora_checkpoint(
                [model], args, str(out), publisher=SimpleNamespace(write_adapter=lambda *_: None),
                optimizer=opt, opt_param_scheduler=None, iteration=9,
            )
            for f in out.iterdir():
                if "dp_invariant" in f.name or "named" in f.name:
                    shutil.copy(f, merged_dir / f.name)

        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)
        target = _Bf16Model(5)
        zeros = {p: torch.zeros(p.shape) for p in _params_of(target)}
        loader = _DistOpt(_params_of(target), zeros, dp_rank=0, dp_size=1)
        result = lora_utils.load_lora_adapter(
            [target], str(merged_dir), optimizer=loader, dp_invariant=True, rng_policy="keep_on_dp_change"
        )

        assert result == (True, 9, True)
        gathered = {p: torch.zeros(p.shape) for p in _params_of(target)}
        loader.gather_into(gathered)
        for src_p, tgt_p in zip(_params_of(model), _params_of(target), strict=True):
            torch.testing.assert_close(gathered[tgt_p], full[src_p], rtol=0, atol=0)
            torch.testing.assert_close(tgt_p.data, src_p.data, rtol=0, atol=0)

    def test_the_exact_policy_refuses_a_dp_change_before_touching_the_model(self, tmp_path, monkeypatch):
        model = _AdapterModel(1.0)
        opt = torch.optim.Adam(model.parameters())
        for dp_rank in (0, 1):
            _patch_parallel_state(monkeypatch, dp_rank=dp_rank, dp_size=2)
            args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=False, lora_dp_invariant_state=True)
            lora_utils.save_lora_checkpoint(
                [model],
                args,
                str(tmp_path / f"dp{dp_rank}"),
                publisher=SimpleNamespace(write_adapter=lambda *_: None),
                optimizer=opt, opt_param_scheduler=None, iteration=1,
            )
        merged = tmp_path / "dp0"
        shutil.copy(tmp_path / "dp1" / "training_state_named_tp0_pp0_dp1.pt", merged)
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)
        target = _AdapterModel(0.0)
        with pytest.raises(dps.DpInvariantStateError, match="explicit policy"):
            lora_utils.load_lora_adapter([target], str(merged), dp_invariant=True, rng_policy="exact")
        assert torch.equal(target.lora_A.data, torch.zeros(2, 2))


@needs_fork_lora_checkpoint
class TestLoadValidatesBeforeWriting:
    def _save_bf16_dist(self, tmp_path, monkeypatch):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        opt = _DistOpt(_params_of(model), init, dp_rank=0, dp_size=1)
        opt.step(_grads(model, 1))
        _patch_parallel_state(monkeypatch, dp_rank=0, dp_size=1)
        args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=False, lora_dp_invariant_state=True)
        lora_utils.save_lora_checkpoint(
            [model],
            args,
            str(tmp_path / "ckpt"),
            publisher=SimpleNamespace(write_adapter=lambda *_: None),
            optimizer=opt,
            opt_param_scheduler=None,
            iteration=1,
        )
        return tmp_path / "ckpt"

    def test_an_optimizer_that_does_not_fit_leaves_the_adapter_untouched(self, tmp_path, monkeypatch):
        ckpt = self._save_bf16_dist(tmp_path, monkeypatch)
        target = _Bf16Model(5)
        before = {n: p.detach().clone() for n, p in target.named_parameters()}
        # the target optimizer also updates a parameter the checkpoint does not have
        extra = torch.nn.Parameter(torch.zeros(3))
        target.register_parameter("lora_C", extra)
        opt = torch.optim.AdamW(list(target.parameters()))

        with pytest.raises(KeyError, match="lacks parameters"):
            lora_utils.load_lora_adapter([target], str(ckpt), optimizer=opt, dp_invariant=True)
        for n, p in target.named_parameters():
            if n in before:
                assert torch.equal(p, before[n])

    def test_a_failing_optimizer_write_restores_the_adapter(self, tmp_path, monkeypatch):
        ckpt = self._save_bf16_dist(tmp_path, monkeypatch)
        target = _Bf16Model(5)
        before = {n: p.detach().clone() for n, p in target.named_parameters()}
        zeros = {p: torch.zeros(p.shape) for p in _params_of(target)}
        opt = _DistOpt(_params_of(target), zeros, dp_rank=0, dp_size=1)
        monkeypatch.setattr(
            dps, "load_named_optimizer_state", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("write failed"))
        )

        with pytest.raises(RuntimeError, match="write failed"):
            lora_utils.load_lora_adapter([target], str(ckpt), optimizer=opt, dp_invariant=True)
        for n, p in target.named_parameters():
            assert torch.equal(p, before[n])

    def test_tensor_valued_hyperparameters_are_compared_by_value(self):
        model = _Model(0)
        opt = torch.optim.Adam([{"params": list(model.params.values()), "lr": torch.tensor(0.1)}])
        _step(model, opt, 1)
        a, b = _export(model, opt), _export(model, opt)  # separate tensor objects, equal values
        merged = dps.merge_named_optimizer_states([a, b])
        dps.load_named_optimizer_state(opt, _named(model), merged)


@needs_fork_lora_checkpoint
class TestArgumentFailFast:
    """Validation runs on the args as ``parse_args`` leaves them, i.e. after ``set_default_megatron_args``."""

    @staticmethod
    def _args(**overrides):
        pytest.importorskip("megatron.training.arguments")
        from miles.backends.megatron_utils.arguments import set_default_megatron_args

        base = dict(
            lora_dp_invariant_state=True,
            lora_rank=8,
            optimizer="adam",
            fp16=False,
            seq_length=None,
            vocab_size=None,
            padded_vocab_size=None,
            tokenizer_model="tok",
            tokenizer_type="HuggingFaceTokenizer",
            multi_latent_attention=False,
            rope_type="rope",
            spec=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            use_precision_aware_optimizer=False,
            num_distributed_optimizer_instances=1,
        )
        return set_default_megatron_args(Namespace(**{**base, **overrides}))

    def test_the_default_megatron_config_is_accepted(self):
        from miles.utils.lora.arguments import validate_lora_dp_invariant_args

        args = self._args()
        assert args.bf16 and args.use_distributed_optimizer  # what a real launch gets
        validate_lora_dp_invariant_args(args)
        validate_lora_dp_invariant_args(self._args(optimizer="sgd"))  # bf16 without DistOpt
        validate_lora_dp_invariant_args(self._args(lora_dp_invariant_state=False, fp16=True))

    @pytest.mark.parametrize(
        "overrides, match",
        [
            (dict(fp16=True), "loss-scaler"),
            (dict(use_precision_aware_optimizer=True), "precision-aware"),
            (dict(num_distributed_optimizer_instances=2), "partial DP group"),
            (dict(context_parallel_size=2), "CP>1"),
            (dict(expert_model_parallel_size=4), "EP>1"),
            (dict(lora_rank=0), "needs LoRA"),
        ],
    )
    def test_unsupported_configs_fail_at_parse_time(self, overrides, match):
        from miles.utils.lora.arguments import validate_lora_dp_invariant_args

        with pytest.raises(ValueError, match=match):
            validate_lora_dp_invariant_args(self._args(**overrides))


class TestDistributedOptimizerLazyState:
    """A read of a fresh DistOpt used to leave empty state entries; the load then skipped the Adam init and Megatron
    copied only ``param``, dropping exp_avg / exp_avg_sq without a word (E2 GPU C1 diag 2)."""

    def _saved(self, model, dp_size=2):
        init = {p: p.detach().float() for p in _params_of(model)}
        savers = [_DistOpt(_params_of(model), init, dp_rank=r, dp_size=dp_size) for r in range(dp_size)]
        for opt in savers:
            opt.step(_grads(model, 1))
            opt.step(_grads(model, 2))
        return savers, [_export(model, opt) for opt in savers]

    def _fresh(self, model, dp_rank=0, dp_size=1):
        zeros = {p: torch.zeros(p.shape) for p in _params_of(model)}
        return _DistOpt(_params_of(model), zeros, dp_rank=dp_rank, dp_size=dp_size)

    def test_reading_a_fresh_optimizer_leaves_no_state_entry(self):
        model = _Bf16Model(0)
        fresh = self._fresh(model)
        _export(model, fresh)
        [slot.get() for slot in dps.optimizer_slots(fresh, _named(model))]
        assert len(fresh.optimizer.state) == 0

    def test_moments_survive_empty_entries_left_by_an_earlier_read(self):
        model = _Bf16Model(0)
        savers, shards = self._saved(model)
        merged = dps.merge_named_optimizer_states(shards)
        loader = self._fresh(model)
        # what the old read path (defaultdict access in Megatron's getter) left behind
        for group in loader.optimizer.param_groups:
            for main in group["params"]:
                loader.optimizer.state[main]
        assert loader.optimizer.state and not any(loader.optimizer.state.values())
        ranges_before = [(s.name, s.start, s.end) for s in dps.optimizer_slots(loader, _named(model))]

        _load(model, loader, shards)

        reread = _export(model, loader)["entries"]
        for name, full in merged.items():
            for key in ("param", "exp_avg", "exp_avg_sq"):
                assert torch.equal(reread[name]["tensors"][key], full["tensors"][key]), (name, key)
            assert torch.equal(reread[name]["scalars"]["step"], full["scalars"]["step"])
        assert [(s.name, s.start, s.end) for s in dps.optimizer_slots(loader, _named(model))] == ranges_before

    def test_a_saved_key_the_target_cannot_hold_is_refused(self):
        model = _Bf16Model(0)
        _, shards = self._saved(model)
        merged = dps.merge_named_optimizer_states(shards)
        for entry in merged.values():
            entry["tensors"]["momentum_buffer"] = torch.zeros_like(entry["tensors"]["param"])
        loader = self._fresh(model)
        with pytest.raises(dps.DpInvariantStateError, match="momentum_buffer"):
            dps.load_named_optimizer_state(loader, _named(model), merged)

    def test_a_target_key_the_save_lacks_is_still_refused(self):
        model = _Bf16Model(0)
        _, shards = self._saved(model)
        merged = dps.merge_named_optimizer_states(shards)
        for entry in merged.values():
            del entry["tensors"]["exp_avg_sq"]
        loader = self._fresh(model)
        loader.step(_grads(model, 5))
        with pytest.raises(dps.DpInvariantStateError, match="lacks \\['exp_avg_sq'\\]"):
            dps.load_named_optimizer_state(loader, _named(model), merged)


class _TwoBufferDistOpt(_DistOpt):
    """A DistOpt whose grad buffers hold ``lora_A`` and ``lora_B`` under two dtype keys, the first in two buckets."""

    def __init__(self, params, full_main, *, lr=0.1):
        a, b = params
        buffers = [((torch.bfloat16, torch.float32), [[a], []]), ((torch.float16, torch.float32), [[b]])]
        self.gbuf_ranges, self.param_map, self.model_param_group_index_map, mains = [{}], {}, {}, []
        for key, buckets in buffers:
            maps = []
            for bucket in buckets:
                index_map, offset = {}, 0
                for p in bucket:
                    index_map[p] = (offset, offset + p.numel(), 0)
                    offset += p.numel()
                param_map = _dist.DistributedOptimizer._build_model_gbuf_param_range_map(
                    index_map, _dist.Range(0, offset), 0
                )
                for p, ranges in param_map.items():
                    r = ranges["param"]
                    self.model_param_group_index_map[p] = (0, len(mains))
                    mains.append(full_main[p].reshape(-1)[r.start : r.end].clone())
                self.param_map.update(param_map)
                maps.append({"param_map": param_map})
            self.gbuf_ranges[0][key] = maps
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        self.optimizer = torch.optim.AdamW([{"params": mains, "lr": lr, "weight_decay": 0.01}], foreach=False)


class TestLazyStateAcrossBuffers:
    def test_partly_empty_entries_over_two_dtypes_and_buckets_restore_bitwise(self):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        saver = _TwoBufferDistOpt(_params_of(model), init)
        saver.step(_grads(model, 1))
        saver.step(_grads(model, 2))
        merged = dps.merge_named_optimizer_states([_export(model, saver)])

        loader = _TwoBufferDistOpt(_params_of(model), {p: torch.zeros(p.shape) for p in _params_of(model)})
        loader.optimizer.state[loader.optimizer.param_groups[0]["params"][1]]  # only lora_B gets an empty entry
        dps.load_named_optimizer_state(loader, _named(model), merged)

        reread = _export(model, loader)["entries"]
        assert set(reread) == {"lora_A", "lora_B"}
        for name, full in merged.items():
            for key in ("param", "exp_avg", "exp_avg_sq"):
                assert torch.equal(reread[name]["tensors"][key], full["tensors"][key]), (name, key)

    def test_a_cross_implementation_resume_names_the_cause(self):
        model = _Bf16Model(0)
        init = {p: p.detach().float() for p in _params_of(model)}
        saver = _TwoBufferDistOpt(_params_of(model), init)
        saver.step(_grads(model, 1))
        merged = dps.merge_named_optimizer_states([_export(model, saver)])
        loader = _TwoBufferDistOpt(_params_of(model), init)
        loader.step(_grads(model, 1))
        for main in loader.optimizer.param_groups[0]["params"]:  # an implementation that keeps no per-param step
            del loader.optimizer.state[main]["step"]
        loader._init_optimizer_states_with_dummy_values = lambda: None
        with pytest.raises(dps.DpInvariantStateError, match="different optimizer implementation"):
            dps.load_named_optimizer_state(loader, _named(model), merged)
