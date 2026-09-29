"""CPU tests for miles_adapter.state + state_plugin (task 3.4).

Uses a tiny torch model that reproduces the Megatron pieces the plugin
touches: BF16 adapter Parameters with FP32 ``main_param`` masters, a
``main_grad`` accumulated by a hook on each Parameter's grad accumulator (as
Megatron DDP does), a chained optimizer that copies masters to model params,
and an ``OptimizerParamScheduler``-like counter.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.engine.miles_adapter import state_plugin as sp  # noqa: E402
from yeto.rl.engine.miles_adapter.state import MilesPolicyState, PolicyStateError  # noqa: E402
from yeto.rl.engine.ports import PolicyState  # noqa: E402
from yeto.rl.engine.trainable_state import TrainableState, UnsupportedLayoutError  # noqa: E402

REV = "0" * 40
CFG = "1" * 64
PREFIX = "base_model.model.model.layers.0.self_attn.q_proj"


class LoraLinear(torch.nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(6, 4, generator=g), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.randn(2, 4, generator=g).to(torch.bfloat16))
        self.lora_B = torch.nn.Parameter(torch.randn(6, 2, generator=g).to(torch.bfloat16))

    def forward(self, x):
        return x @ self.weight.T + (x.to(torch.bfloat16) @ self.lora_A.T @ self.lora_B.T).float()


def attach_megatron_like_grad_hooks(params):
    """Megatron DDP: accumulate into ``param.main_grad`` from the grad-accumulator node."""

    fired = {id(p): 0 for p in params}
    keep = []
    for p in params:
        p.main_grad = torch.zeros(p.shape, dtype=torch.float32)
        acc = p.expand_as(p).grad_fn.next_functions[0][0]

        def hook(*_unused, param=p):
            if param.grad is not None:
                param.main_grad.add_(param.grad.float())
                param.grad = None
                fired[id(param)] += 1

        acc.register_hook(hook)
        keep.append(acc)
    return fired, keep


class ChildOptimizer:
    def __init__(self, params):
        self.params = params
        self.optimizer = torch.optim.Adam([p.main_param for p in params], lr=1e-2)

    def _copy_main_params_to_model_params(self):
        with torch.no_grad():
            for p in self.params:
                p.copy_(p.main_param.view(p.shape).to(p.dtype))


class Scheduler:
    def __init__(self):
        self.num_steps = 0

    def step(self, increment):
        self.num_steps += increment


class FakeRankActor:
    def __init__(self, main: bool, seed: int = 0):
        self.module = LoraLinear(seed)
        self.model = [self.module]
        params = [self.module.lora_A, self.module.lora_B]
        for p in params:
            p.main_param = p.detach().float().clone()
        self.optimizer = SimpleNamespace(chained_optimizers=[ChildOptimizer(params)])
        self.opt_param_scheduler = Scheduler()
        self.args = SimpleNamespace(global_batch_size=4, tensor_model_parallel_size=1,
                                    pipeline_model_parallel_size=1, expert_model_parallel_size=1)
        self._is_first_replica_megatron_main_rank = main
        self.backups = []
        self.weights_backuper = SimpleNamespace(backup=self.backups.append)
        ident = lambda t: t.clone()  # noqa: E731
        self._yeto_adapter_bindings = (
            sp.AdapterBinding(f"{PREFIX}.lora_A.weight", self.module.lora_A, ident, ident),
            sp.AdapterBinding(f"{PREFIX}.lora_B.weight", self.module.lora_B, ident, ident),
        )
        self.hook_counts, self._keep = attach_megatron_like_grad_hooks(params)

    def train_step(self):
        """One micro-step + optimizer step, like Megatron (masters then copy)."""

        x = torch.randn(3, 4, generator=torch.Generator().manual_seed(7))
        loss = self.module(x).pow(2).sum()
        loss.backward()
        child = self.optimizer.chained_optimizers[0]
        for p in child.params:
            p.main_param.grad = p.main_grad.clone()
        child.optimizer.step()
        child._copy_main_params_to_model_params()
        norm = sp.grad_norm(self)
        for p in child.params:
            p.main_grad.zero_()
        self.opt_param_scheduler.step(self.args.global_batch_size)
        return norm


class FakeTrainGroup:
    """``TrainGroup.run_plugin(fn_path, kwargs)`` of the yeto/ports Miles patch."""

    def __init__(self, ranks):
        self.ranks = ranks
        self.calls = []

    def run_plugin(self, fn_path, kwargs=None):
        module, _, name = fn_path.rpartition(".")
        fn = getattr(importlib.import_module(module), name)
        self.calls.append(fn_path)
        return [fn(rank, **(kwargs or {})) for rank in self.ranks]


def make(n_ranks=2):
    ranks = [FakeRankActor(main=(i == 0)) for i in range(n_ranks)]
    group = FakeTrainGroup(ranks)
    ps = MilesPolicyState(actor_model=group, base_model_revision=REV, config_hash=CFG)
    return ranks, group, ps


@pytest.fixture(autouse=True)
def _generic_optimizer_reset(monkeypatch):
    """The fake uses torch Adam; upstream's reset (Megatron Adam only) is out of scope here."""

    import sys

    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.optimizer_state_reset", None)


def moments(rank):
    opt = rank.optimizer.chained_optimizers[0].optimizer
    return [s["exp_avg"].abs().sum().item() for s in opt.state.values()]


def test_masters_context_restores_parameter_objects_and_grad_flow():
    rank = FakeRankActor(main=True)
    params = [rank.module.lora_A, rank.module.lora_B]
    ids = [id(p) for p in params]
    with sp.masters_as_module_parameters(rank.model, params):
        exposed = rank.module._parameters["lora_A"]
        assert exposed is not params[0]
        assert exposed.dtype == torch.float32 and not exposed.requires_grad
        assert exposed.data_ptr() == params[0].main_param.data_ptr()  # shares master storage
        assert params[0].dtype == torch.bfloat16  # original .data never reassigned
    assert [id(rank.module._parameters[n]) for n in ("lora_A", "lora_B")] == ids
    sp.assert_grad_flow_intact(rank.model, params)
    rank.train_step()
    assert all(c == 1 for c in rank.hook_counts.values())  # grad accumulator hook still fires


def test_export_and_apply_keep_grad_accumulator_hook():
    """Ports-side counterpart of #64: the PyTorch behaviour itself is pinned by
    ``tests/test_rl_grad_accumulator_hook.py`` (merged with #64); here, after
    export and apply, the next step still produces adapter grads
    (grad_norm > 0) and every accumulator hook fires."""

    ranks, group, ps = make()
    state = ps.export(policy_version=0)
    ps.apply(state, optimizer="reset", local_step=0)
    norms = [r.train_step() for r in ranks]
    assert all(n > 0 for n in norms)
    assert all(c == 1 for r in ranks for c in r.hook_counts.values())


def test_export_apply_export_hash_roundtrip_preserve():
    ranks, group, ps = make()
    assert isinstance(ps, PolicyState)
    for r in ranks:
        r.train_step()
    before_moments = moments(ranks[0])
    first = ps.export(policy_version=1)
    assert isinstance(first, TrainableState) and first.layout == "lora"
    assert first.tensor_names == (f"{PREFIX}.lora_A.weight", f"{PREFIX}.lora_B.weight")
    ps.apply(first, optimizer="preserve", local_step=1)
    second = ps.export(policy_version=1)
    assert second.policy_hash() == first.policy_hash()
    assert moments(ranks[0]) == before_moments and all(m > 0 for m in before_moments)
    assert ranks[0].backups == ["actor"]


def test_apply_reset_zeroes_moments_keeps_optimizer_and_aligns_scheduler():
    ranks, group, ps = make()
    for r in ranks:
        r.train_step()
    opt_before = ranks[0].optimizer.chained_optimizers[0].optimizer
    target = ps.export(policy_version=5)
    changed = {k: v + 0.5 for k, v in target.tensors.items()}
    from yeto.rl.core import canonical_state

    new = TrainableState.from_lora(
        canonical_state(5, changed, base_model_revision=REV, lora_config_hash=CFG)
    )
    ps.apply(new, optimizer="reset", local_step=3)
    assert ranks[0].optimizer.chained_optimizers[0].optimizer is opt_before
    for r in ranks:
        assert all(m == 0 for m in moments(r))
        assert r.opt_param_scheduler.num_steps == 3 * r.args.global_batch_size
        assert torch.equal(r.module.lora_A.main_param, changed[f"{PREFIX}.lora_A.weight"])
        assert torch.equal(r.module.lora_A.detach(), changed[f"{PREFIX}.lora_A.weight"].to(torch.bfloat16))
    assert ps.export(policy_version=5).policy_tensor_hash() == new.policy_tensor_hash()


def test_scheduler_ahead_rejected():
    ranks, group, ps = make(n_ranks=1)
    state = ps.export(policy_version=0)
    ranks[0].opt_param_scheduler.num_steps = 8
    with pytest.raises(sp.StatePluginError, match="ahead"):
        ps.apply(state, optimizer="preserve", local_step=1)


def test_layout_and_name_mismatches_rejected():
    ranks, group, ps = make(n_ranks=1)
    state = ps.export(policy_version=0)
    from yeto.rl.core import canonical_state

    other = {f"{PREFIX}.lora_A.weight": torch.zeros(2, 5), f"{PREFIX}.lora_B.weight": torch.zeros(6, 2)}
    bad = TrainableState.from_lora(canonical_state(0, other, base_model_revision=REV, lora_config_hash=CFG))
    with pytest.raises(PolicyStateError, match="layout hash"):
        ps.apply(bad, optimizer="preserve", local_step=0)
    wrong_cfg = TrainableState.from_lora(
        canonical_state(0, dict(state.tensors), base_model_revision=REV, lora_config_hash="2" * 64)
    )
    with pytest.raises(PolicyStateError, match="different base model"):
        ps.apply(wrong_cfg, optimizer="preserve", local_step=0)
    with pytest.raises(ValueError):
        ps.apply(state, optimizer="zero", local_step=0)


def test_unimplemented_layout_rejected_at_startup():
    with pytest.raises(UnsupportedLayoutError, match="full"):
        MilesPolicyState(actor_model=None, base_model_revision=REV, config_hash=CFG, layout="full")


def test_export_requires_exactly_one_main_rank():
    ranks = [FakeRankActor(main=True), FakeRankActor(main=True)]
    ps = MilesPolicyState(actor_model=FakeTrainGroup(ranks), base_model_revision=REV, config_hash=CFG)
    with pytest.raises(PolicyStateError, match="exactly one"):
        ps.export()


def test_sharded_master_rejected():
    rank = FakeRankActor(main=True)
    rank.module.lora_A.main_param = torch.zeros(3)
    with pytest.raises(sp.StatePluginError, match="complete FP32"):
        sp.export_state(rank, policy_version=0)


def test_state_plugins_wake_an_offloaded_actor_and_restore_sleep():
    from yeto.rl.engine.miles_adapter import state_plugin as sp

    calls = []

    class Actor:
        args = SimpleNamespace(offload_train=True)
        _asleep = True

        def wake_up(self):
            calls.append("wake")
            self._asleep = False

        def sleep(self):
            calls.append("sleep")
            self._asleep = True

    actor = Actor()
    with sp.trainer_resident(actor):
        assert actor._asleep is False
    assert calls == ["wake", "sleep"] and actor._asleep is True
    resident = Actor()
    resident._asleep = False
    with sp.trainer_resident(resident):
        pass
    assert calls == ["wake", "sleep"]


def test_grad_norm_prefers_recorded_train_step_value(monkeypatch):
    from yeto.rl.engine.miles_adapter import state_plugin as sp

    class Optimizer:
        def get_grad_norm(self):
            return 0.0  # recomputed after the step: stale

    monkeypatch.setattr(sp, "_STEP_GRAD_NORMS", [0.25, 0.5])
    actor = SimpleNamespace(optimizer=Optimizer(), model=[])
    assert sp.grad_norm(actor) == 0.5
    assert sp._STEP_GRAD_NORMS == []
    assert sp.grad_norm(actor) == 0.0
