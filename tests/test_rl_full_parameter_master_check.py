"""The read-only model/master check after a full-parameter apply (SC2 P6, D4)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.adapters.miles.full_parameter import master_check as mc  # noqa: E402


def _param(values, *, master=None):
    p = torch.nn.Parameter(torch.tensor(values, dtype=torch.bfloat16))
    p.main_param = torch.tensor(values, dtype=torch.float32) if master is None else master
    return p


def test_equal_model_and_master_and_a_drifted_master():
    a, b = _param([1.0, 2.0]), _param([3.0, 4.5])
    out = mc.master_model_digest([("a", a), ("b", b)], {})
    assert out["equal"] and out["compared"] == 2 and out["skipped"] == 0
    assert "cpu_copies" not in out
    b.main_param = torch.tensor([3.0, 9.0])  # master moved, model not copied
    assert not mc.master_model_digest([("a", a), ("b", b)], {})["equal"]


def test_sharded_master_is_skipped_and_hybrid_cpu_copy_is_compared():
    a = _param([1.0, 2.0])
    s = _param([5.0, 6.0], master=torch.tensor([5.0]))  # a DP slice
    copy = torch.tensor([1.0, 2.0])
    out = mc.master_model_digest([("a", a), ("s", s)], {id(a.main_param): copy})
    assert out["compared"] == 1 and out["skipped"] == 1
    assert out["cpu_copies"] == 1 and out["cpu_copies_pinned"] is False  # not pinned on this host
    assert out["equal"] and out["cpu_copy_bf16_sha256"] == out["model_bf16_sha256"]
    bad = mc.master_model_digest([("a", a)], {id(a.main_param): torch.tensor([1.0, 7.0])})
    assert not bad["equal"]


def test_rank_plugin_reads_actor_model_and_hybrid_optimizer():
    model = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
    for p in model.parameters():
        p.main_param = p.detach().float().clone()
    inner = {model.weight.main_param: model.weight.main_param.clone()}
    actor = SimpleNamespace(model=[model], optimizer=SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=SimpleNamespace(param_to_inner_param=inner))]))
    out = mc.rank_master_model_check(actor)
    assert out["equal"] and out["compared"] == 1 and out["cpu_copies"] == 1


def test_off_by_default_and_budgeted(capsys):
    calls = []

    async def run_plugin(path, kwargs):
        calls.append(path)
        return [{"equal": True}]

    group = SimpleNamespace(run_plugin=run_plugin)
    mc._DONE["applies"] = 0
    asyncio.run(mc.report_master_check(group, "actor", environ={}))
    assert calls == [] and capsys.readouterr().err == ""
    for _ in range(3):
        asyncio.run(mc.report_master_check(group, "actor", environ={mc.MASTER_CHECK_ENV: "2"}))
    assert calls == [mc.MASTER_CHECK, mc.MASTER_CHECK]
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 2 and all(line.startswith(mc.TAG + " role=actor") for line in err)
    mc._DONE["applies"] = 0


def test_a_failing_plugin_never_raises(capsys):
    async def run_plugin(path, kwargs):
        raise RuntimeError("boom")

    mc._DONE["applies"] = 0
    asyncio.run(mc.report_master_check(SimpleNamespace(run_plugin=run_plugin), "actor",
                                       environ={mc.MASTER_CHECK_ENV: "1"}))
    assert "failed: RuntimeError: boom" in capsys.readouterr().err
    mc._DONE["applies"] = 0


def test_precision_aware_parameter_uses_its_hybrid_copy_as_master():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    out = mc.master_model_digest([("p", p)], {id(p): torch.tensor([1.0, 2.0])})
    assert out["compared"] == 1 and out["equal"]


def _dist_actor(views, masters, inner=None, model_values=(1.0, 2.0, 3.0, 4.0)):
    model = torch.nn.Linear(4, 1, bias=False).to(torch.bfloat16)
    child = SimpleNamespace(shard_float16_groups=[views], shard_fp32_from_float16_groups=masters,
                            optimizer=SimpleNamespace(param_to_inner_param=inner or {}))
    return SimpleNamespace(model=[model], optimizer=SimpleNamespace(chained_optimizers=[child]))


def test_distributed_ranges_compare_the_model_slice_with_its_fp32_shard():
    full = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    view = full[1:3]
    out = mc.rank_master_model_check(_dist_actor([view], [[torch.tensor([2.0, 3.0])]]))
    assert out["distributed"] and out["compared"] == 1 and out["equal"]
    bad = mc.rank_master_model_check(_dist_actor([view], [[torch.tensor([2.0, 5.0])]]))
    assert not bad["equal"]


def test_precision_aware_distributed_range_uses_the_hybrid_copy():
    view = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    copy = torch.tensor([1.0, 2.0])
    actor = _dist_actor([view], None, inner={view: copy})
    out = mc.rank_master_model_check(actor)
    assert out["compared"] == 1 and out["equal"] and out["cpu_copies"] == 1
    assert out["cpu_copies_pinned"] is False  # an unpinned CPU master is reported


def test_an_empty_range_is_skipped_and_no_ranges_falls_back_to_parameters():
    empty = torch.tensor([], dtype=torch.bfloat16)
    view = torch.tensor([1.0], dtype=torch.bfloat16)
    out = mc.distributed_digest([("e", empty, torch.tensor([])), ("v", view, None)], {})
    assert out["compared"] == 0 and out["skipped"] == 2 and not out["equal"]
    assert mc.distributed_ranges(SimpleNamespace()) == []
