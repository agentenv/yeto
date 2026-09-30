"""E2 harness sequence on a CPU fake island (protocol test only, NOT GPU evidence)."""

from __future__ import annotations

import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from tests.rl_cut_fakes import GBS, make_rank, train_step
from yeto.rl.engine.miles_adapter import LoopRunner, cut_plugin, e2_harness, state_plugin
from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup
from yeto.rl.engine.miles_adapter.trainer_rebuild import SwappableActor
from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle

ARGS = SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=1, num_steps_per_rollout=1,
                       global_batch_size=GBS, load="/ref", requested_load=None, start_rollout_id=0,
                       ref_load="/ref")


class RankGroup:
    def __init__(self, ranks):
        self.ranks = ranks
        self.disposed = False

    async def run_plugin(self, fn_path, kwargs=None):
        module, name = fn_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(module), name)
        return [fn(rank, **(kwargs or {})) for rank in self.ranks]

    async def train(self, rollout_id, payload):
        for rank in self.ranks:
            train_step(rank, payload)
            state_plugin._STEP_APPLIED_LRS.append(rank.optimizer.param_groups[0]["lr"])
        return [SimpleNamespace(outcome="NORMAL", metrics={}) for _ in self.ranks]

    async def dispose(self):
        self.disposed = True


class State:
    def __init__(self, version):
        self.policy_version = version

    def policy_tensor_hash(self):
        return hashlib.sha256(f"policy-{self.policy_version}".encode()).hexdigest()


class Rollout:
    def __init__(self):
        self.cursor = {"sample_offset": 0, "epoch_id": 0, "sample_group_index": 0, "sample_index": 0}

    def data_cursor(self):
        return dict(self.cursor)

    def members(self):
        return frozenset({"e0"})


class Driver:
    def __init__(self, trainer):
        self.trainer, self.rollout = trainer, Rollout()
        self.ledger = None
        self.local_step, self.config_epoch, self.at_safe_point = 0, 0, False
        self.published_state = self.published_version = None
        self.expected_token = "t"
        self.sync = SimpleNamespace(start=lambda d: SimpleNamespace(rollout_id=0, state=State(0)))
        self.publisher = SimpleNamespace(
            publish=lambda state: (_ for _ in ()).throw(AssertionError("no re-publication")))
        self.rounds_completed = 0
        self.colocated = True  # MilesTrainerGroup.onload is a no-op without offload_train
        self.policy_state = SimpleNamespace(export=lambda: self.published_state)

    def handshake(self):
        pass

    def publish(self, state, *, rollout_id):
        self.published_state, self.published_version = state, rollout_id

    def safe_point(self, rid):
        self.at_safe_point = True

    def _generate(self, rid):
        g = torch.Generator().manual_seed(100 + rid)
        self.rollout.cursor = {**self.rollout.cursor, "sample_offset": self.rollout.cursor["sample_offset"] + 2}
        return RolloutBatchHandle(rollout_id=rid, policy_version=rid, policy_hash="b" * 64,
                                  groups=(GroupMetadata(f"g{rid}", ("s0",), "tok", 0.0, 0.0, 8),),
                                  completed=0, aborted=0, payload=torch.randn(3, 6, generator=g))

    def run_round(self, rid):
        self.at_safe_point = False
        self.trainer.train_step(self._generate(rid))
        self.local_step += 1
        self.publish(State(rid + 1), rollout_id=rid + 1)

    def rebuild_trainer(self, rebuild, *, cut_policy_hash):  # must not be used by the harness
        raise AssertionError("the harness must not re-publish through driver.rebuild_trainer")


@pytest.fixture
def determinism(monkeypatch):
    for k, v in cut_plugin.DETERMINISM_ENV.items():
        monkeypatch.setenv(k, v)


def _rank(seed):
    rank = make_rank(seed)
    rank.args.deterministic_mode = True
    return rank


def _ctx(tmp_path, *, fresh_rank=_rank):
    actor = SwappableActor(RankGroup([_rank(0)]))
    trainer = MilesTrainerGroup(args=ARGS, actor_model=actor, learner_id=0, learner_generation=0,
                                parameter_layout_hash=lambda: "c" * 64, runner=LoopRunner(),
                                check_policy_tokens=False, release_refs=lambda a, p: None)
    driver = Driver(trainer)

    async def rebuild(args, executor, *, old_handles, worker_manager, trainer_pg_view):
        await old_handles["actor"].dispose()
        return RankGroup([fresh_rank(7)]), None

    plan = {"schema": e2_harness.PLAN_SCHEMA, "config": "C1", "warmup_rounds": 2,
            "out_dir": str(tmp_path), "expected_dp": 1}
    return e2_harness.HarnessContext(
        driver=driver, actor=actor, miles_args=ARGS, rollout_executor="ex", runner=LoopRunner(),
        algorithm=SimpleNamespace(sha256=lambda: "a" * 64), base_model_revision="rev",
        backend_fingerprint="fp", plan=plan, rebuild=rebuild, worker_manager="wm")


def test_full_sequence_passes_on_the_cpu_fake(tmp_path, determinism):
    results = e2_harness.run_harness(_ctx(tmp_path))
    assert results["pass"]
    names = set(results["criteria"])
    for case in "abcdeg":
        assert any(n.startswith(f"G-4.2({case})") for n in names), case
    assert "G-4.3(2) adapter/master/moments/step/RNG bitwise equal" in names
    saved = json.loads((tmp_path / "C1" / "results.json").read_text())
    assert saved["pass"] and (tmp_path / "C1" / "steps.jsonl").exists()


def test_missing_determinism_is_environment_blocked(tmp_path, monkeypatch):
    for k in cut_plugin.DETERMINISM_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(e2_harness.EnvironmentBlocked):
        e2_harness.run_harness(_ctx(tmp_path))
    saved = json.loads((tmp_path / "C1" / "results.json").read_text())
    assert saved["criteria"]["determinism_settings"]["pass"] is False


def test_a_rebuild_that_loses_rng_fails_the_bitwise_criterion(tmp_path, determinism, monkeypatch):
    real = cut_plugin.MilesCutBackend  # noqa: F841 - the fake backend is on the rank

    def lossy_rank(seed):
        rank = _rank(seed)
        backend = rank._yeto_cut_backend
        backend.restore_rng = lambda state: None  # a broken restore: RNG not restored
        return rank

    with pytest.raises(Exception) as info:
        e2_harness.run_harness(_ctx(tmp_path, fresh_rank=lossy_rank))
    # restore_cut's own digest check already refuses it (RECOVERY_REQUIRED inside the rebuild)
    assert "rng_digest" in str(info.value) or "failed" in str(info.value)
    saved = json.loads((tmp_path / "C1" / "results.json").read_text())
    assert saved["pass"] is False


def test_plan_file_is_optional_and_validated(tmp_path):
    assert e2_harness.load_plan({}, repo_root=tmp_path) is None
    (tmp_path / e2_harness.PLAN_FILE).write_text(json.dumps({"schema": "x"}))
    with pytest.raises(e2_harness.HarnessFailed, match="invalid"):
        e2_harness.load_plan({}, repo_root=tmp_path)
    with pytest.raises(e2_harness.HarnessFailed, match="does not exist"):
        e2_harness.load_plan({e2_harness.PLAN_ENV: str(tmp_path / "none.json")})


def test_plan_lora_dropout_must_match_the_ranks(tmp_path, determinism):
    ctx = _ctx(tmp_path)
    ctx.plan = {**ctx.plan, "lora_dropout": 0.05}
    for r in ctx.actor.target.ranks:
        r.args.lora_dropout = 0.0
    with pytest.raises(e2_harness.EnvironmentBlocked, match="dropout"):
        e2_harness.run_harness(ctx)


def test_failed_rebuild_records_the_attempts(tmp_path, determinism):
    ctx = _ctx(tmp_path)

    async def failing(args, executor, *, old_handles, worker_manager, trainer_pg_view):
        err = type("TrainerRebuildError", (RuntimeError,), {})("failed at start_pools")
        err.stage, err.cleanup_error, err.previous_view, err.view_restored = "start_pools", None, None, False
        raise err from RuntimeError("bundles [0] are in use by running cell inference-0")

    ctx.rebuild = failing
    with pytest.raises(Exception):
        e2_harness.run_harness(ctx)
    crit = json.loads((tmp_path / "C1" / "results.json").read_text())["criteria"]["harness_completed"]
    assert [a["stage"] for a in crit["attempts"]] == ["start_pools", "start_pools"]
    assert "in use by running cell" in crit["attempts"][0]["error"]
