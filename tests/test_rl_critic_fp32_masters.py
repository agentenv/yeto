"""rl-algo-critic-family (user decision 2026-10-07): critic two-island sync is over
the FP32 optimizer masters (CPU only; fake DistributedOptimizer, no Megatron, no Ray).

A fake DistributedOptimizer leaf holds bf16 model parameters and FP32 main shards
(``gbuf_ranges`` / ``_get_main_param_and_optimizer_states`` as read by
``state_plugin.full_masters`` / ``write_masters``); its ``step`` updates the FP32
shards from the bf16 grads and then casts main -> model, like Megatron. Whether the
real Megatron shard layout matches is unverified (needs GPU 4.5).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from yeto.rl import critic_state as cs
from yeto.rl.engine.algorithm import AlgorithmSpecError, check_unverified_allowance
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from yeto.rl.engine.miles_adapter import state_plugin as sp


class _Range:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _FakeDistOpt:
    """One DP rank: owns ``[start, end)`` of every parameter's flat FP32 master."""

    def __init__(self, params, full_main, *, dp_rank=0, dp_size=1, lr=1e-2):
        self.params = list(params)
        self.param_map, self.mains = {}, {}
        for p in self.params:
            n = p.numel()
            shard = -(-n // dp_size)
            start, end = min(dp_rank * shard, n), min((dp_rank + 1) * shard, n)
            self.param_map[p] = {"param": _Range(start, end)}
            main = full_main[p].reshape(-1)[start:end].clone()
            self.mains[id(p)] = (main, start, end)
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": self.param_map}]}]
        self.model_param_group_index_map = {p: (0, i) for i, p in enumerate(self.params)}
        self.buffers = [SimpleNamespace(param_index_map={p: None for p in self.params})]
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        self.data_parallel_group = None
        shards = [m for m, _, _ in self.mains.values()]
        self.inner = torch.optim.Adam(shards, lr=lr)

    def _get_main_param_and_optimizer_states(self, param):
        return {"param": self.mains[id(param)][0]}

    def step(self):
        """Grad (bf16 model grad) -> FP32 shard step -> main->model copy (bf16 cast)."""
        for p in self.params:
            main, s, e = self.mains[id(p)]
            main.grad = p.grad.detach().float().reshape(-1)[s:e].clone()
        self.inner.step()
        for p in self.params:
            main, s, e = self.mains[id(p)]
            flat = p.data.view(-1)
            flat[s:e].copy_(main.detach().to(p.dtype))  # each rank casts its owned range

    def zero_grad(self):
        for p in self.params:
            p.grad = None
        self.inner.zero_grad()

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state):
        self.inner.load_state_dict(state)


class _Sched:
    num_steps = 0

    def state_dict(self):
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state):
        self.num_steps = state["num_steps"]


def _island(seed, *, dp_rank=0, dp_size=1, body=None):
    torch.manual_seed(seed)
    if body is None:
        body = torch.nn.Module()
        body.embedding = torch.nn.Linear(5, 3, bias=False)
        body.output_layer = torch.nn.Linear(3, 1, bias=False)
        body = body.to(torch.bfloat16)
    params = [p for p in body.parameters() if p.requires_grad]
    g = torch.Generator().manual_seed(seed + 100)
    # FP32 masters that are NOT representable in bf16 (the real situation)
    full = {p: p.detach().float() + 1e-4 * torch.randn(p.shape, generator=g) for p in params}
    opt = _FakeDistOpt(params, full, dp_rank=dp_rank, dp_size=dp_size)
    return SimpleNamespace(model=[body], optimizer=opt, opt_param_scheduler=_Sched())


def _grad_step(actor):
    params = [p for p in actor.model[0].parameters() if p.requires_grad]
    loss = sum((p.float() ** 2).sum() for p in params)
    loss.backward()
    actor.optimizer.step()
    actor.optimizer.zero_grad()


def _model_from_masters(actor):
    masters = sp._critic_masters(actor, sp._critic_parameters(actor))
    for k, p in sp._critic_parameters(actor).items():
        assert torch.equal(p.detach(), masters[k].to(p.dtype)), k


def _average(a, b):
    return {k: (a[k] + b[k]) / 2 for k in a}


def _write(actor, tensors):
    return sp._import_critic_tensors(
        actor, by_rank={0: {"tensors": tensors, "sha256": cs.critic_weights_sha256(tensors)}})


def test_two_islands_average_on_fp32_masters_bitwise_and_model_from_masters():
    a, b = _island(0), _island(1)
    _grad_step(a), _grad_step(b)
    ea, eb = sp._export_critic_tensors(a), sp._export_critic_tensors(b)
    for e in (ea, eb):
        assert all(t.dtype == torch.float32 for t in e["tensors"].values())
    # export reads the masters, not the bf16 parameters
    params = sp._critic_parameters(a)
    assert any(not torch.equal(ea["tensors"][k], p.detach().float()) for k, p in params.items())
    avg = _average(ea["tensors"], eb["tensors"])
    want = cs.critic_weights_sha256(avg)
    ra, rb = _write(a, avg), _write(b, avg)
    assert "refused" not in ra and "refused" not in rb  # old bf16 write-back refusal is gone
    assert ra["weights_sha256"] == rb["weights_sha256"] == want
    assert sp._export_critic_tensors(a)["weights_sha256"] == sp._export_critic_tensors(b)["weights_sha256"] == want
    _model_from_masters(a), _model_from_masters(b)
    for k in params:
        assert torch.equal(sp._critic_parameters(a)[k], sp._critic_parameters(b)[k])


def test_next_optimizer_step_continues_from_the_average():
    a, b = _island(0), _island(1)
    avg = _average(sp._export_critic_tensors(a)["tensors"], sp._export_critic_tensors(b)["tensors"])
    _write(a, avg)
    # reference: an untouched optimizer whose masters are exactly the average
    ref = _island(0)
    for k, p in sp._critic_parameters(ref).items():
        p.data.copy_(avg[k].to(p.dtype))
    for p in ref.optimizer.params:
        main, s, e = ref.optimizer.mains[id(p)]
        key = next(k for k, q in sp._critic_parameters(ref).items() if q is p)
        main.data.copy_(avg[key].reshape(-1)[s:e])
    _grad_step(a), _grad_step(ref)
    after = sp._export_critic_tensors(a)["tensors"]
    assert cs.critic_weights_sha256(after) == cs.critic_weights_sha256(sp._export_critic_tensors(ref)["tensors"])
    # the stale (pre-average) masters did not overwrite the average
    stale = _island(0)
    _grad_step(stale)
    assert cs.critic_weights_sha256(after) != cs.critic_weights_sha256(sp._export_critic_tensors(stale)["tensors"])
    _model_from_masters(a)


def test_dp2_sharded_masters_gather_and_write_back(monkeypatch):
    """Two DP ranks of one island, in threads, with a summing DP all-reduce."""
    torch.manual_seed(3)
    body = torch.nn.Module()
    body.embedding = torch.nn.Linear(5, 3, bias=False)  # 15 elements: DP split cuts through
    body.output_layer = torch.nn.Linear(3, 1, bias=False)
    body = body.to(torch.bfloat16)
    # both DP ranks share one model object (enough for the gather/scatter check)
    ranks = [_island(3, dp_rank=r, dp_size=2, body=body) for r in range(2)]
    ref = sp._critic_masters(_island(3, body=body), sp._critic_parameters(ranks[0]))
    local = threading.local()
    barrier, slots = threading.Barrier(2), {}

    def all_reduce(leaf, flat):
        slots[local.rank] = flat
        barrier.wait()
        total = slots[0] + slots[1]
        barrier.wait()
        flat.copy_(total)

    monkeypatch.setattr(sp, "_dp_all_reduce_sum", all_reduce)
    monkeypatch.setattr(sp, "_rank", lambda: local.rank)
    out = {}

    def run(rank, fn):
        local.rank = rank
        out[rank] = fn(ranks[rank])

    def both(fn):
        ts = [threading.Thread(target=run, args=(r, fn)) for r in range(2)]
        [t.start() for t in ts], [t.join() for t in ts]
        return out[0], out[1]

    e0, e1 = both(sp._export_critic_tensors)
    assert e0["weights_sha256"] == e1["weights_sha256"]
    for k, m in ref.items():
        assert torch.equal(e0["tensors"][k], m)  # gathered full FP32 master
    target = {k: v + 3e-5 for k, v in e0["tensors"].items()}
    h = cs.critic_weights_sha256(target)
    by_rank = {r: {"tensors": target, "sha256": h} for r in range(2)}
    r0, r1 = both(lambda actor: sp._import_critic_tensors(actor, by_rank=by_rank))
    assert r0 == {"rank": 0, "weights_sha256": h} and r1 == {"rank": 1, "weights_sha256": h}
    for r in range(2):
        main, s, e = ranks[r].optimizer.mains[id(body.embedding.weight)]
        assert torch.equal(main, target["0:embedding.weight"].reshape(-1)[s:e])


def test_low_precision_without_any_master_is_refused_and_fp32_reads_itself():
    body = torch.nn.Linear(3, 1, bias=False).to(torch.bfloat16)
    actor = SimpleNamespace(model=[body], optimizer=torch.optim.Adam(body.parameters()))
    with pytest.raises(sp.StatePluginError, match="no FP32 optimizer master"):
        sp._export_critic_tensors(actor)
    fp32 = torch.nn.Linear(3, 1, bias=False)
    actor = SimpleNamespace(model=[fp32], optimizer=torch.optim.Adam(fp32.parameters()))
    e = sp._export_critic_tensors(actor)
    assert torch.equal(e["tensors"]["0:weight"], fp32.weight.detach())
    t = {"0:weight": e["tensors"]["0:weight"] + 0.5}
    assert _write(actor, t)["weights_sha256"] == cs.critic_weights_sha256(t)
    assert torch.equal(fp32.weight.detach(), t["0:weight"])


def test_round_cut_save_restore_hash_matches_the_channel(tmp_path):
    a = _island(0)
    _grad_step(a)
    a.opt_param_scheduler.num_steps = 8
    exported = sp._export_critic_tensors(a)
    saved = sp._save_critic_cut(a, directory=str(tmp_path), round_id=2)
    assert saved["weights_sha256"] == exported["weights_sha256"]
    opt_before = {k: v["exp_avg"].clone() for k, v in a.optimizer.inner.state_dict()["state"].items()}
    fresh = _island(9)
    got = sp._restore_critic_cut(fresh, directory=str(tmp_path), actor_round=2, critic_round=2)
    assert got == {"rank": 0, "weights_sha256": exported["weights_sha256"]}
    assert sp._export_critic_tensors(fresh)["weights_sha256"] == exported["weights_sha256"]
    _model_from_masters(fresh)
    for k, v in fresh.optimizer.inner.state_dict()["state"].items():
        assert torch.equal(v["exp_avg"], opt_before[k])
    assert fresh.opt_param_scheduler.num_steps == 8
    assert "refused" in sp._restore_critic_cut(fresh, directory=str(tmp_path), actor_round=3, critic_round=2)


def test_allowance_strict_avg_critic_on_two_islands_decoupled_still_refused():
    from yeto.rl.algos.critic import critic_run_problems
    from yeto.rl.engine.algorithm import AlgorithmSpec

    names = ["execution:critic", "features:gae_length_adaptive"]
    assert check_unverified_allowance(names, islands=2, outer_sync=True, sync_preset="strict-avg") == tuple(sorted(names))
    for preset in ("decoupled", None, "dense-full"):
        with pytest.raises(AlgorithmSpecError, match="single-island"):
            check_unverified_allowance(names, islands=2, outer_sync=True, sync_preset=preset)
    # a non-critic unverified mechanism keeps D11, even with the critic under strict-avg
    with pytest.raises(AlgorithmSpecError, match="single-island"):
        check_unverified_allowance(["execution:critic", "losses:cispo"], islands=2, outer_sync=True,
                                   sync_preset="strict-avg")
    # critic features without the critic itself are not the critic exception
    with pytest.raises(AlgorithmSpecError, match="single-island"):
        check_unverified_allowance(["features:gae_length_adaptive"], islands=2, outer_sync=True,
                                   sync_preset="strict-avg")
    # unknown names are still refused
    with pytest.raises(AlgorithmSpecError, match="unknown"):
        check_unverified_allowance(["execution:bogus"], islands=2, outer_sync=True, sync_preset="strict-avg")
    ppo = AlgorithmSpec(advantage_estimator="ppo")
    assert ppo.execution.needs_critic
    assert any("decoupled" in p for p in critic_run_problems(ppo, {"sync_preset": "decoupled"}))
    assert critic_run_problems(ppo, {"sync_preset": "strict-avg"}) == []


def test_launcher_two_island_critic_strict_avg_passes_decoupled_refused(tmp_path, _fake_sky, _no_modal_listing):  # noqa: F811
    from test_rl_algorithm_provenance import _launcher_args

    from yeto.launcher import _prepare_rl_args
    from yeto.rl.engine.algorithm import AlgorithmSpec

    spec_path = tmp_path / "ppo.json"
    spec_path.write_text(AlgorithmSpec(advantage_estimator="ppo").canonical_json())

    def args(preset, allow):
        a = _launcher_args("ports", gpu="aws:1xa100@us-east-1,aws:1xa100@us-west-2")
        a.controller = "local"
        a.rl_algorithm_spec = str(spec_path)
        a.rl_sync_preset = preset
        a.rl_allow_unverified_mechanism = allow
        return a

    ok = args("strict-avg", ["advantage_estimators:ppo", "execution:critic"])
    _prepare_rl_args(ok)
    assert ok.rl_allow_unverified_mechanism == ["advantage_estimators:ppo", "execution:critic"]
    with pytest.raises(Exception, match="unverified|execution:critic|critic"):
        _prepare_rl_args(args("strict-avg", None))  # still needs the explicit allowance
    with pytest.raises(Exception, match="decoupled"):
        _prepare_rl_args(args("decoupled", ["advantage_estimators:ppo", "execution:critic"]))


def test_critic_state_summary_wakes_an_asleep_colocated_critic():
    """s13-g1-modal-20261007a: the summary read paused critic memory (CUDA invalid argument)."""
    import types

    import torch

    from yeto.rl.engine.miles_adapter import state_plugin

    calls = []

    class Actor:
        def __init__(self):
            self.args = types.SimpleNamespace(offload_train=True)
            self._asleep = True
            self.model = [torch.nn.Linear(2, 1)]

        def wake_up(self):
            calls.append("wake")
            self._asleep = False

        def sleep(self):
            calls.append("sleep")
            self._asleep = True

    out = state_plugin.critic_state_summary(Actor())
    assert calls == ["wake", "sleep"]
    assert len(out["specs"]) == 2 and len(out["weights_sha256"]) == 64
