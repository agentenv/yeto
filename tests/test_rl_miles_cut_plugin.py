"""rl-infra-spec 4.2/4.3: rank-side cut shard save/restore, CPU analog of X3 (not GPU evidence)."""

from __future__ import annotations

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
    with pytest.raises(CutPluginError, match="truncated"):
        restore_cut_shard(make_rank(1), directory=str(tmp_path), files=[s], cut_id="c1")
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(CutPluginError, match="checksum"):
        restore_cut_shard(make_rank(1), directory=str(tmp_path), files=[s], cut_id="c1")
    path.write_bytes(data)
    with pytest.raises(CutPluginError, match="not a shard of cut"):
        restore_cut_shard(make_rank(1), directory=str(tmp_path), files=[s], cut_id="other")


def test_layout_change_is_refused(tmp_path):
    rank = make_rank(0)
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    other = make_rank(0, coord={"global_rank": 0, "tp": 0, "pp": 0, "dp": 0, "dp_size": 2,
                                "cp_size": 1, "ep_size": 1})
    with pytest.raises(CutPluginError, match="same-shape"):
        restore_cut_shard(other, directory=str(tmp_path), files=[s], cut_id="c1")


def test_refused_restore_writes_nothing(tmp_path):
    rank = make_rank(0)
    train_step(rank, _batches(1)[0])
    s = save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    fresh = make_rank(3)
    fresh._yeto_cut_backend.check_optimizer = lambda *a: (_ for _ in ()).throw(ValueError("mismatch"))
    before = params(fresh)
    with pytest.raises(ValueError):
        restore_cut_shard(fresh, directory=str(tmp_path), files=[s], cut_id="c1")
    for n, v in before.items():
        assert torch.equal(v, params(fresh)[n])


def test_shard_is_immutable(tmp_path):
    rank = make_rank(0)
    save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    with pytest.raises(CutPluginError, match="immutable"):
        save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")


def test_pending_step_records_mean_not_at_step_boundary(tmp_path):
    state_plugin._STEP_GRAD_NORMS.append(1.0)
    try:
        with pytest.raises(CutPluginError, match="step boundary"):
            save_cut_shard(make_rank(0), directory=str(tmp_path), cut_id="c1")
    finally:
        state_plugin._STEP_GRAD_NORMS.clear()


def test_non_lora_trainable_parameters_are_refused(tmp_path):
    rank = make_rank(0)
    rank.model[0].base.weight.requires_grad_(True)
    with pytest.raises(CutPluginError, match="other than LoRA"):
        save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")


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
    with pytest.raises(CutPluginError, match="fp16"):
        save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
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
