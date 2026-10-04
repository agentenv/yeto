"""rl-infra-spec 4.2/4.3: rank-side cut shard save/restore, CPU analog of X3 (not GPU evidence)."""

from __future__ import annotations

import re

from types import SimpleNamespace

import pytest
import torch

from tests.rl_cut_fakes import make_rank, params, train_step
from yeto.rl.engine.miles_adapter import cut_plugin, state_plugin
from yeto.rl.engine.miles_adapter.cut_plugin import (
    CutPluginError,
    config_problems,
    restore_cut_shard,
    save_cut_shard,
)


def _batches(n):
    g = torch.Generator().manual_seed(7)
    return [torch.randn(3, 6, generator=g) for _ in range(n)]


def test_same_shape_restore_matches_continuous_run_bitwise(tmp_path):
    batches = _batches(3)
    # continuous run: 2 steps, then the frozen third batch
    cont = make_rank(seed=0)
    torch.manual_seed(99)
    for b in batches:
        train_step(cont, b)  # the RNG is global: finish the continuous run first
    # cut run: same 2 steps, save, "rebuild" (fresh rank, different init), restore, third batch
    run = make_rank(seed=0)
    torch.manual_seed(99)
    for b in batches[:2]:
        train_step(run, b)
    summary = save_cut_shard(run, directory=str(tmp_path), cut_id="c1")
    assert summary["has_optimizer_state"] and summary["has_rng"] and summary["scheduler_samples"] == 8
    fresh = make_rank(seed=5)
    torch.manual_seed(12345)  # the rebuilt process has other RNG state
    restored = restore_cut_shard(fresh, directory=str(tmp_path), files=[summary], cut_id="c1")
    assert restored["state_digest"] == summary["state_digest"]
    assert restored["rng_digest"] == summary["rng_digest"]
    assert fresh.opt_param_scheduler.num_steps == 8
    train_step(fresh, batches[2])
    for name, value in params(cont).items():
        assert torch.equal(value, params(fresh)[name]), name
    for p_c, p_f in zip(cont.optimizer.param_groups[0]["params"], fresh.optimizer.param_groups[0]["params"]):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(cont.optimizer.state[p_c][key], fresh.optimizer.state[p_f][key])


def test_restore_without_rng_would_diverge(tmp_path):
    """Guards the test above: dropout makes the next step RNG dependent."""
    batches = _batches(3)
    a, b = make_rank(0), make_rank(0)
    torch.manual_seed(1)
    train_step(a, batches[0])
    torch.manual_seed(2)
    train_step(b, batches[0])
    assert any(not torch.equal(x, params(b)[n]) for n, x in params(a).items())


def test_corrupted_truncated_and_foreign_shards_are_refused(tmp_path):
    rank = make_rank(0)
    train_step(rank, _batches(1)[0])
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    path = tmp_path / s["path"]
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    assert re.search("truncated", restore_cut_shard(make_rank(1), directory=str(tmp_path), files=[s], cut_id="c1")["refused"])
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    assert re.search("checksum", restore_cut_shard(make_rank(1), directory=str(tmp_path), files=[s], cut_id="c1")["refused"])
    path.write_bytes(data)
    assert re.search("not a shard of cut", restore_cut_shard(make_rank(1), directory=str(tmp_path), files=[s], cut_id="other")["refused"])


def test_layout_change_is_refused(tmp_path):
    rank = make_rank(0)
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    other = make_rank(0, coord={"global_rank": 0, "tp": 0, "pp": 0, "dp": 0, "dp_size": 2,
                                "cp_size": 1, "ep_size": 1})
    assert re.search("same-shape", restore_cut_shard(other, directory=str(tmp_path), files=[s], cut_id="c1")["refused"])


def test_refused_restore_writes_nothing(tmp_path):
    rank = make_rank(0)
    train_step(rank, _batches(1)[0])
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    fresh = make_rank(3)
    fresh._yeto_cut_backend.check_optimizer = lambda *a: (_ for _ in ()).throw(ValueError("mismatch"))
    before = params(fresh)
    assert "mismatch" in restore_cut_shard(fresh, directory=str(tmp_path), files=[s], cut_id="c1")["refused"]
    for n, v in before.items():
        assert torch.equal(v, params(fresh)[n])


def test_shard_is_immutable(tmp_path):
    rank = make_rank(0)
    save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    assert re.search("immutable", save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")["refused"])


def test_pending_step_records_mean_not_at_step_boundary(tmp_path):
    state_plugin._STEP_GRAD_NORMS.append(1.0)
    try:
        assert re.search("step boundary", save_cut_shard(make_rank(0), directory=str(tmp_path), cut_id="c1")["refused"])
    finally:
        state_plugin._STEP_GRAD_NORMS.clear()


def test_non_lora_trainable_parameters_are_refused(tmp_path):
    rank = make_rank(0)
    rank.model[0].base.weight.requires_grad_(True)
    assert re.search("other than LoRA", save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")["refused"])


@pytest.mark.parametrize(
    "args, match",
    [
        (SimpleNamespace(fp16=True), "fp16"),
        (SimpleNamespace(use_precision_aware_optimizer=True), "precision-aware"),
        (SimpleNamespace(num_distributed_optimizer_instances=2), "instances"),
        (SimpleNamespace(context_parallel_size=2), "CP>1"),
        (SimpleNamespace(expert_model_parallel_size=2), "EP>1"),
    ],
)
def test_unsupported_configurations(args, match):
    assert any(match in p for p in config_problems(args))


def test_unsupported_configuration_fails_before_writing(tmp_path):
    rank = make_rank(0, args=SimpleNamespace(fp16=True))
    assert re.search("fp16", save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")["refused"])
    assert not list(tmp_path.iterdir())


def test_distopt_peers_are_merged(tmp_path):
    """A DistOpt rank loads every DP shard of its (tp, pp) for the fork-M5 merge."""
    files = [
        {"path": "trainer_tp0_pp0_dp0.pt", "coord": {"tp": 0, "pp": 0, "dp": 0}},
        {"path": "trainer_tp0_pp0_dp1.pt", "coord": {"tp": 0, "pp": 0, "dp": 1}},
        {"path": "trainer_tp1_pp0_dp0.pt", "coord": {"tp": 1, "pp": 0, "dp": 0}},
    ]
    peers = cut_plugin._peer_entries(files, {"tp": 0, "pp": 0, "dp": 0})
    assert [p["path"] for p in peers] == ["trainer_tp0_pp0_dp1.pt"]


def test_plugin_paths_resolve():
    import importlib

    for path in (cut_plugin.SAVE_CUT_SHARD, cut_plugin.RESTORE_CUT_SHARD):
        module, name = path.rsplit(".", 1)
        assert callable(getattr(importlib.import_module(module), name))


def test_restore_into_a_trained_scheduler_is_refused_before_any_write(tmp_path):
    """Megatron load_state_dict ADDS num_steps: only a fresh trainer (scheduler at 0) may be restored."""
    rank = make_rank(0)
    train_step(rank, _batches(1)[0])
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    live = make_rank(3)
    train_step(live, _batches(1)[0])
    before = params(live)
    assert re.search("freshly built", restore_cut_shard(live, directory=str(tmp_path), files=[s], cut_id="c1")["refused"])
    assert live.opt_param_scheduler.num_steps == 4
    for n, v in before.items():
        assert torch.equal(v, params(live)[n])


def test_scheduler_hyper_parameter_mismatch_is_refused_before_any_write(tmp_path):
    rank = make_rank(0)
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    fresh = make_rank(3)
    fresh.opt_param_scheduler.lr0 = 0.5
    before = params(fresh)
    assert re.search("hyper-parameters", restore_cut_shard(fresh, directory=str(tmp_path), files=[s], cut_id="c1")["refused"])
    for n, v in before.items():
        assert torch.equal(v, params(fresh)[n])


def test_tp_pp_with_distributed_optimizer_is_refused():
    args = SimpleNamespace(tensor_model_parallel_size=2, use_distributed_optimizer=True)
    assert any("DistributedOptimizer" in p for p in config_problems(args))
    assert not config_problems(SimpleNamespace(tensor_model_parallel_size=2))
    assert not config_problems(SimpleNamespace(use_distributed_optimizer=True))


def test_save_reports_optimizer_coverage(tmp_path):
    s = save_cut_shard(make_rank(0), directory=str(tmp_path), cut_id="c1")
    assert s["adapter_names"] == s["optimizer_names"] == ["lora_A", "lora_B"]


def test_refusal_is_returned_but_a_failure_after_writing_raises(tmp_path, monkeypatch):
    """Miles marks a cell errored on any run_plugin exception (GPU C1, 2026-09-30):
    refusals before any write return {"refused"}; a failure after writing still raises."""
    rank = make_rank(0)
    train_step(rank, _batches(1)[0])
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    assert "freshly built" in restore_cut_shard(rank, directory=str(tmp_path), files=[s], cut_id="c1")["refused"]
    fresh = make_rank(4)

    def boom(*a, **k):
        raise RuntimeError("load failed")

    fresh._yeto_cut_backend.load_optimizer = boom
    with pytest.raises(RuntimeError, match="load failed"):
        restore_cut_shard(fresh, directory=str(tmp_path), files=[s], cut_id="c1")


def test_side_effect_free_state_removes_entries_created_by_a_read():
    from collections import defaultdict

    from yeto.rl.engine.miles_adapter.cut_plugin import side_effect_free_state

    inner = SimpleNamespace(state=defaultdict(dict))
    inner.state["kept"] = {"exp_avg": 1}
    leaf = SimpleNamespace(optimizer=inner)  # Megatron wrapper -> torch optimizer
    chained = SimpleNamespace(chained_optimizers=[leaf])
    with side_effect_free_state(chained):
        inner.state["new-empty"]
        inner.state["new-filled"]["x"] = 1
    assert set(inner.state) == {"kept", "new-filled"}


def test_side_effect_free_state_never_swallows_errors():
    """Defence in depth only: an exception raised by the read (e.g. a fixed fork-M5 refusing
    missing/extra keys) propagates unchanged; only empty entries created by the read go."""
    from collections import defaultdict

    from yeto.rl.engine.miles_adapter.cut_plugin import side_effect_free_state

    inner = SimpleNamespace(state=defaultdict(dict))

    class ForkError(RuntimeError):
        pass

    with pytest.raises(ForkError, match="lacks exp_avg"):
        with side_effect_free_state(SimpleNamespace(optimizer=inner)):
            inner.state["p"]
            raise ForkError("saved state of 'p' lacks exp_avg")
    assert not inner.state


def test_a_fixed_fork_setter_error_reaches_restore_cut(tmp_path):
    """After the fork fix, a key mismatch raises inside load (after writes began) and the
    restore fails -- not masked by the read wrapper."""
    rank = make_rank(0)
    train_step(rank, _batches(1)[0])
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    fresh = make_rank(4)

    def strict_load(optimizer, named, merged):
        raise ValueError("saved state has keys the destination lacks: ['exp_avg']")

    fresh._yeto_cut_backend.load_optimizer = strict_load
    with pytest.raises(ValueError, match="destination lacks"):
        restore_cut_shard(fresh, directory=str(tmp_path), files=[s], cut_id="c1")


def test_miles_backend_export_leaves_no_empty_state_entries(monkeypatch):
    """L4: MilesCutBackend.export_optimizer itself (fork-M5 export stubbed as Megatron reads it:
    ``optimizer.state[main_param]`` on a defaultdict) leaves the state as it was."""
    from collections import defaultdict

    main = torch.nn.Parameter(torch.zeros(3))
    inner = SimpleNamespace(state=defaultdict(dict), param_groups=[{"params": [main]}])
    distopt = SimpleNamespace(optimizer=inner)

    def export_named_optimizer_state(optimizer, named):
        entries = {}
        for n, _ in named:
            state = optimizer.optimizer.state[main]  # Megatron _get_main_param_and_optimizer_states
            entries[n] = {"tensors": {"param": main.detach().clone(), **dict(state)}}
        return {"entries": entries}

    backend = cut_plugin.MilesCutBackend()
    monkeypatch.setattr(backend, "_dps", lambda: SimpleNamespace(
        export_named_optimizer_state=export_named_optimizer_state))
    out = backend.export_optimizer(distopt, [("w", main)])
    assert list(out["entries"]) == ["w"] and not inner.state


def test_weight_version_counter_is_carried_by_the_cut(tmp_path):
    """GPU C1 plan-v6: a rebuilt trainer restarted its update_weights counter at 0 and the
    rollout executor refused the re-publication (version went backwards)."""
    rank = make_rank(0)
    rank.weight_updater = SimpleNamespace(weight_version=3)
    train_step(rank, _batches(1)[0])
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    fresh = make_rank(4)
    fresh.weight_updater = SimpleNamespace(weight_version=0)
    r = restore_cut_shard(fresh, directory=str(tmp_path), files=[s], cut_id="c1")
    assert "refused" not in r and fresh.weight_updater.weight_version == 3
    assert r["state_digest"] == s["state_digest"]  # the counter is part of the verified state
    no_updater = make_rank(5)
    assert "weight_updater" in restore_cut_shard(no_updater, directory=str(tmp_path), files=[s],
                                                 cut_id="c1")["refused"]
