"""plan-v3 G-4.5 test-only injection switches (CPU; off by default)."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
import torch

from tests.rl_cut_fakes import make_rank, params, train_step
from yeto.rl.engine.miles_adapter import cut_injection as ci
from yeto.rl.engine.miles_adapter import cut_plugin
from yeto.rl.engine.miles_adapter.trainer_rebuild import inject_rebuild_failures


class Killed(BaseException):
    pass


@pytest.fixture
def no_exit(monkeypatch):
    def fake(reason):
        raise Killed(reason)

    monkeypatch.setattr(ci, "kill_now", fake)


def test_all_switches_default_off(monkeypatch):
    for name in ci.ALL_ENVS:
        monkeypatch.delenv(name, raising=False)
    assert ci.save_kill_rank() is None and ci.restore_kill_rank() is None
    assert ci.restore_sleep() is None and ci.rebuild_fail_count() == 0 and ci.cursor_shift() is None


@pytest.mark.parametrize("name, raw", [(ci.SAVE_KILL_RANK_ENV, "x"), (ci.RESTORE_SLEEP_ENV, "0"),
                                       (ci.RESTORE_SLEEP_ENV, "0:-1"), (ci.REBUILD_FAIL_ENV, "0"),
                                       (ci.CURSOR_SHIFT_ENV, "-2")])
def test_invalid_values_raise(name, raw):
    readers = {ci.SAVE_KILL_RANK_ENV: ci.save_kill_rank, ci.RESTORE_SLEEP_ENV: ci.restore_sleep,
               ci.REBUILD_FAIL_ENV: ci.rebuild_fail_count, ci.CURSOR_SHIFT_ENV: ci.cursor_shift}
    with pytest.raises(ci.InjectionConfigError):
        readers[name]({name: raw})


def test_save_kill_leaves_half_temp_file_and_no_shard(tmp_path, monkeypatch, no_exit):
    monkeypatch.setenv(ci.SAVE_KILL_RANK_ENV, "0")
    with pytest.raises(Killed):
        cut_plugin.save_cut_shard(make_rank(0), directory=str(tmp_path), cut_id="c1")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["trainer_tp0_pp0_dp0.pt.tmp"] and (tmp_path / names[0]).stat().st_size > 0


def test_save_kill_other_rank_is_inert(tmp_path, monkeypatch, no_exit):
    monkeypatch.setenv(ci.SAVE_KILL_RANK_ENV, "1")
    cut_plugin.save_cut_shard(make_rank(0), directory=str(tmp_path), cut_id="c1")


def test_restore_kill_after_adapter_before_optimizer(tmp_path, monkeypatch, no_exit):
    rank = make_rank(0)
    train_step(rank, torch.randn(3, 6))
    s = cut_plugin.save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    fresh = make_rank(5)
    monkeypatch.setenv(ci.RESTORE_KILL_RANK_ENV, "0")
    with pytest.raises(Killed):
        cut_plugin.restore_cut_shard(fresh, directory=str(tmp_path), files=[s], cut_id="c1")
    assert all(torch.equal(v, params(fresh)[n]) for n, v in params(rank).items())  # adapter written
    assert not fresh.optimizer.state  # optimizer not
    assert fresh.opt_param_scheduler.num_steps == 0


def test_restore_sleep_targets_one_rank(tmp_path, monkeypatch):
    rank = make_rank(0)
    s = cut_plugin.save_cut_shard(rank, directory=str(tmp_path), cut_id="c1")
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    monkeypatch.setenv(ci.RESTORE_SLEEP_ENV, "0:7.5")
    cut_plugin.restore_cut_shard(make_rank(1), directory=str(tmp_path), files=[s], cut_id="c1")
    assert slept == [7.5]


def test_rebuild_failure_runs_inside_the_fork_function():
    module = SimpleNamespace()
    calls = []

    async def create_training_models(*a, **k):
        calls.append("create")
        return "actor", None

    module.create_training_models = create_training_models

    async def fork_rebuild(*a, **k):  # like the fork: stages, then create_training_models
        try:
            return await module.create_training_models()
        except RuntimeError as error:
            raise type("TrainerRebuildError", (RuntimeError,), {})(f"create: {error}") from error

    wrapped = inject_rebuild_failures(fork_rebuild, module, 1)
    with pytest.raises(RuntimeError, match="create: yeto test injection"):
        asyncio.run(wrapped())
    assert module.create_training_models is create_training_models  # restored
    assert asyncio.run(wrapped()) == ("actor", None) and calls == ["create"]


def test_cursor_shift_writes_where_miles_load_reads(tmp_path):
    args = SimpleNamespace(load=str(tmp_path), start_rollout_id=0)
    cursor = {"sample_offset": 8, "epoch_id": 0, "sample_group_index": 8, "sample_index": 64}
    path = ci.write_shifted_dataset_state(args, cursor, 2)
    assert path == os.path.join(str(tmp_path), "rollout/global_dataset_state_dict_-1.pt")
    state = torch.load(path, weights_only=True)
    assert state["sample_offset"] == 10 and state["sample_group_index"] == 10
    with pytest.raises(ci.InjectionConfigError, match="already exists"):
        ci.write_shifted_dataset_state(args, cursor, 2)
    with pytest.raises(ci.InjectionConfigError, match="args.load"):
        ci.write_shifted_dataset_state(SimpleNamespace(load=None), cursor, 1)


def test_state_summary_and_determinism_readouts(monkeypatch):
    rank = make_rank(0)
    s0 = cut_plugin.state_summary(rank)
    assert s0["moments_nonzero"] == 0 and s0["scheduler_samples"] == 0
    train_step(rank, torch.randn(3, 6))
    s1 = cut_plugin.state_summary(rank)
    assert s1["moments_nonzero"] == 2 and s1["state_digest"] != s0["state_digest"]
    assert s1["bf16_master_mismatch"] == []
    assert cut_plugin.state_summary(rank)["rng_digest"] == s1["rng_digest"]  # no randomness consumed
    for k, v in cut_plugin.DETERMINISM_ENV.items():
        monkeypatch.setenv(k, v)
    rank.args.deterministic_mode = True
    d = cut_plugin.rank_determinism(rank)
    assert d["env_ok"] and d["deterministic_mode"]
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    assert not cut_plugin.rank_determinism(rank)["env_ok"]
