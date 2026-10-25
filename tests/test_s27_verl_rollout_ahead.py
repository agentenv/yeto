"""S27 G-EV: rollout ahead on the verl separate-mode island (VERL-PARITY #10).

CPU only, no verl import. ``VerlRolloutPool`` runs over a stand-in for verl's
v1 separate trainer (``prepare_step`` hands the prompts to the rollout servers
and returns; ``replay_buffer.sample`` waits for them) and a stand-in
``transfer_queue``. Covered:

* the split generation: ``start_generate(r+1)`` submits with the published
  v_r and the prompts' ``global_steps`` of round r+1, and gives the trainer its
  own step back; ``finish_generate`` samples that batch after round r trained;
  a wrong publication or token is refused before any prompt is submitted;
* the serial ``generate`` is unchanged (same submissions, same records);
* a real ``IslandDriver`` (no sync) over the real pool, the separate-mode
  publisher, profile and capabilities: batch r >= 1 is submitted while the
  servers hold v_{r-1}, no weight update runs while a batch generates, the
  trainer trains round r with ``global_steps`` r + 1;
* two islands through the launcher-built Rust syncer (asynchronous rounds):
  the batches generated ahead bind to the round's published policy and both
  islands acknowledge the final cut;
* the profile and capabilities are those of the serial island without
  rollout ahead (FSDP2 and Megatron);
* the launcher sends ``--rl-rollout-ahead`` to verl's separate-mode islands
  (synced and single, FSDP2 and Megatron) and refuses the others before any
  machine starts; the island refuses them again.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_verl_full_parameter_launch import MILES_ONLY, _flags, _island_argv, _prepared
from test_s27_verl_full_parameter_sync import (  # noqa: F401 - syncer_binary is a fixture
    COMPONENT, MAX_FRAGMENT_BYTES, SHAPES, TOKENS, _Model, _plan, _start_syncer, _step, needs_cargo,
    syncer_binary,
)
from test_s27_verl_separate import _CheckpointManager, _SeparateIsland, _weights_hash
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.adapters.verl import island_entry, separate
from yeto.rl.adapters.verl.ports_impl import VerlPendingGeneration, VerlRolloutPool, round_base_policy
from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec

GROUP = 2  # samples per prompt group
PROMPTS = 2  # prompt groups per round (train_batch_size)


# ----------------------------------------------------------------------------- verl stand-ins
class _Rows:
    """What ``tq.kv_batch_get`` returns: indexable fields and ``to_padded_tensor``."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, name):
        return [r.get(name, [1, 1]) for r in self.rows]  # the nested response mask

    def to_padded_tensor(self):
        return {"uid": np.array([r["uid"] for r in self.rows], dtype=object),
                "prompts": torch.tensor([r["prompt"] for r in self.rows]),
                "rm_scores": torch.tensor([[0.0, r["reward"]] for r in self.rows]),
                "response_mask": torch.tensor([[1, 1] for _ in self.rows])}


class _TransferQueue:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    def kv_batch_get(self, *, keys, partition_id, select_fields):
        return _Rows([self.rows[k] for k in keys])


@pytest.fixture
def tq(monkeypatch):
    queue = _TransferQueue()
    monkeypatch.setitem(sys.modules, "transfer_queue", types.SimpleNamespace(kv_batch_get=queue.kv_batch_get))
    skip = types.ModuleType("verl.utils.skip")
    skip.SkipManager = SimpleNamespace(set_step=lambda step: None)
    for name, module in (("verl", types.ModuleType("verl")), ("verl.utils", types.ModuleType("verl.utils")),
                         ("verl.utils.skip", skip)):
        monkeypatch.setitem(sys.modules, name, module)
    return queue


class _Servers(_CheckpointManager):
    """verl's ``standalone_checkpoint_manager``; a weight update while a batch is still
    generating is the error rollout ahead must never make."""

    def __init__(self, island, trainer):
        super().__init__(island)
        self.trainer = trainer

    def update_weights(self, version):
        assert self.trainer.in_flight == 0, f"weights v{version} written while a batch generates"
        return super().update_weights(version)


class _SeparateTrainer:
    """verl's v1 separate trainer as ``VerlRolloutPool`` uses it."""

    def __init__(self, queue: _TransferQueue, learner_id: int = 0):
        self.queue = queue
        self.learner_id = learner_id
        self.global_steps = 1
        self.timing_raw = {}
        self.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=PROMPTS, gen_batch_size=None),
                                      actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=1.0)))
        self.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())  # rewards in the loop
        self.replay_buffer = self
        self.standalone_checkpoint_manager = None  # set by the island
        self.submitted: list[dict] = []  # one record per prepare_step
        self.sampled: list[int] = []  # global_steps of every replay-buffer sample
        self.in_flight = 0
        self._batches: dict[int, list[str]] = {}

    def _fetch_one_gen_batch(self):
        return [f"p{len(self.submitted)}-{i}" for i in range(PROMPTS)]

    def prepare_step(self):
        """Submit one batch to the rollout servers and return (fork ``agent_loop_tq``)."""
        prompts = self._fetch_one_gen_batch()
        servers = self.standalone_checkpoint_manager
        keys = []
        for p, _prompt in enumerate(prompts):
            uid = f"l{self.learner_id}-s{self.global_steps}-{p}"
            for k in range(GROUP):
                key = f"{uid}_{k}"
                self.queue.rows[key] = {"uid": uid, "prompt": [self.global_steps, p], "reward": 1.0}
                keys.append(key)
        self._batches[self.global_steps] = keys
        self.submitted.append({"global_steps": self.global_steps, "served": servers.calls[-1],
                               "weights": _weights_hash(servers.rollout_weights)})
        self.in_flight += 1
        return {"prep/submitted": float(len(prompts))}

    def on_sample_begin(self):
        return None

    def on_sample_end(self):
        return None

    def sample(self, *, global_steps, partition_id, batch_size):
        keys = self._batches.pop(global_steps)  # the batch tagged with this step
        self.in_flight -= 1
        self.sampled.append(global_steps)
        tags = [{"global_steps": global_steps} for _ in keys]
        return SimpleNamespace(keys=keys, partition_id=partition_id, tags=tags, extra_info={}), {}

    def _balance_batch(self, batch, metrics):
        return batch


def _island(tmp_path: Path, queue: _TransferQueue, learner_id: int = 0, tape=None):
    tape = [] if tape is None else tape
    island = _SeparateIsland(_Model(seed=0), tmp_path, learner_id, tape)
    trainer = _SeparateTrainer(queue, learner_id)
    trainer.standalone_checkpoint_manager = _Servers(island, trainer)
    island.trainer = trainer
    island.pending = None
    return island, tape


def _publish(island, version: int) -> str:
    island.trainer.standalone_checkpoint_manager.update_weights(version)
    island.published = (version, f"{version:064x}")
    return f"yeto:{version}:{version:064x}"


# ----------------------------------------------------------------------------- the split generation
def test_start_submits_with_the_published_policy_and_gives_the_trainer_its_step_back(tmp_path, tq):
    island, _tape = _island(tmp_path, tq)
    pool = VerlRolloutPool(island)
    t = island.trainer
    _publish(island, 0)
    first = pool.generate(0)
    assert (first.rollout_id, first.policy_version, t.global_steps) == (0, 0, 1)
    pending = pool.start_generate(1, expected_policy_version=f"yeto:0:{0:064x}")
    # the prompts of round 1 carry its step; the trainer is back on round 0's step to train it
    assert [s["global_steps"] for s in t.submitted] == [1, 2] and t.global_steps == 1
    assert t.in_flight == 1 and isinstance(pending, VerlPendingGeneration)
    batch = pool.finish_generate(pending)
    assert t.sampled == [1, 2] and t.global_steps == 2 and t.in_flight == 0
    assert (batch.rollout_id, batch.policy_version, batch.policy_hash) == (1, 0, f"{0:064x}")
    assert {g.policy_token for g in batch.groups} == {f"yeto:0:{0:064x}"}
    assert [g.group_id for g in batch.groups] == ["r1-l0-s2-0", "r1-l0-s2-1"]
    assert island.pending["metrics"] == {"prep/submitted": float(PROMPTS)}
    # the prompts drawn so far: two batches (the cursor the driver reads before a start)
    assert batch.data_cursor == {"sample_offset": 2 * PROMPTS}
    records = [json.loads(line) for line in (tmp_path / "verl-rollout-0.jsonl").read_text().splitlines()]
    assert [r.get("ahead") for r in records] == [None, True]
    assert [r["sample_submit_global_steps"] for r in records] == [[1], [2]]


def test_start_refuses_a_wrong_publication_or_token_before_any_prompt(tmp_path, tq):
    from yeto.rl.engine.driver import PolicyIdentityError

    island, _tape = _island(tmp_path, tq)
    pool = VerlRolloutPool(island)
    _publish(island, 0)
    with pytest.raises(RuntimeError, match="without the publication of v1"):
        pool.start_generate(2)
    with pytest.raises(PolicyIdentityError, match="driver expects"):
        pool.start_generate(1, expected_policy_version=f"yeto:0:{1:064x}")
    assert island.trainer.submitted == []


def test_a_cancelled_batch_is_never_collected(tmp_path, tq):
    island, _tape = _island(tmp_path, tq)
    pool = VerlRolloutPool(island)
    _publish(island, 0)
    pending = pool.start_generate(1)
    pool.cancel_generate(pending)
    with pytest.raises(RuntimeError, match="cancelled"):
        pool.finish_generate(pending)


def test_the_serial_generation_submits_and_samples_as_before(tmp_path, tq):
    island, _tape = _island(tmp_path, tq)
    pool = VerlRolloutPool(island)
    t = island.trainer
    for version in range(3):
        _publish(island, version)
        batch = pool.generate(version, expected_policy_version="ignored")
        assert batch.policy_version == version and t.global_steps == version + 1
    assert [s["global_steps"] for s in t.submitted] == t.sampled == [1, 2, 3]
    with pytest.raises(RuntimeError, match="without a verified publication"):
        pool.generate(5)
    record = json.loads((tmp_path / "verl-rollout-0.jsonl").read_text().splitlines()[0])
    assert "ahead" not in record and record["expected_policy_version"] == "ignored"


# ----------------------------------------------------------------------------- a real driver
SPEC = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1),
                     correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
POOLS = separate.SeparatePools(1, 1, 1, 1, rollout_tp=1).to_dict()


def _trainer_port(island, learner_id: int, *, layout_hash: str | None = None):
    from yeto.rl.contracts import LocalStepReceipt
    from yeto.rl.engine.driver import TrainStepMetrics

    class Trainer:
        publish_offloaded = False

        def __init__(self):
            self.trained: list[tuple[int, int, int]] = []  # (round, behavior version, trainer global_steps)

        def onload(self):
            return None

        def offload(self):
            return None

        def train_step(self, batch):
            self.trained.append((batch.rollout_id, batch.policy_version, island.trainer.global_steps))
            with torch.no_grad():
                for name, parameter in island.model.named_parameters():
                    parameter.add_(_step(learner_id, batch.rollout_id)[name])
            ids = tuple(s for g in batch.groups for s in g.sample_ids)
            batch_hash = hashlib.sha256("|".join(ids).encode()).hexdigest()
            base_version, base_hash = round_base_policy(island, batch)  # VerlTrainerGroup's rule
            receipt = LocalStepReceipt(
                algorithm="grpo", learner_id=learner_id, learner_generation=0,
                base_policy_version=base_version, base_policy_hash=base_hash,
                input_batch_hash=batch_hash, trajectory_ids=ids, trained_tokens=TOKENS, optimizer_steps=1,
                optimizer_step_succeeded=True, parameter_layout_hash=layout_hash or island.layout_hash)
            island.full_group.record_round(receipt, vfs.VerlRoundEvidence(
                batch.rollout_id, batch.policy_hash, batch_hash, TOKENS, ids))
            return receipt

        def step_metrics(self):
            return TrainStepMetrics(grad_norm=0.5, loss=0.1, pg_loss=0.1, lr=1e-6, train_step=0, applied_lrs=(1e-6,))

    return Trainer()


def _driver(island, tape_path: Path, plan: dict, sync, policy_state, learner_id: int = 0):
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.engine.driver import EventTape, IslandDriver

    options = separate.driver_options(plan, SPEC, verl_capabilities("sha256:" + "1" * 64))
    driver = IslandDriver(learner_id=learner_id, rollout=VerlRolloutPool(island),
                          trainer=_trainer_port(island, learner_id,
                                                layout_hash=(plan.get(vfs.PLAN_KEY) or {}).get(
                                                    "parameter_layout_sha256")),
                          policy_state=policy_state, publisher=separate.SeparatePublisher(island), algorithm=SPEC,
                          sync=sync, events=EventTape(tape_path, learner_id), **options)
    separate.wire_rollout_ahead(driver, plan)
    return driver


def _local_plan(ahead: int) -> dict:
    return {"separate": POOLS, "sync": "none", "groups_per_round": PROMPTS, "samples_per_group": GROUP,
            "rollout_ahead": ahead, "max_policy_age": 1 if ahead else 0}


def _run_local(tmp_path: Path, queue: _TransferQueue, *, ahead: int, rounds: int = 4):
    from yeto.rl.engine.bridges import LocalOnlySync

    island, tape = _island(tmp_path, queue)
    island.full_group = vfs.VerlFullParameterGroup(island)
    state = vfs.VerlFullParameterLocalState(island.full_group, component=COMPONENT,
                                            max_fragment_bytes=MAX_FRAGMENT_BYTES)
    island.layout_hash = state.layout_hash
    driver = _driver(island, tmp_path / "tape.jsonl", _local_plan(ahead), LocalOnlySync(rounds), state)
    final = driver.run()
    events = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines()]
    return island, driver, final, events


def test_the_next_batch_generates_on_the_published_policy_while_the_trainer_trains(tmp_path, tq):
    island, driver, final, events = _run_local(tmp_path, tq, ahead=1)
    t, servers = island.trainer, island.trainer.standalone_checkpoint_manager
    assert final.policy_version == 4 and servers.calls == [0, 1, 2, 3, 4]
    # batch 0 serially on v0; batch r >= 1 submitted while the servers held v_{r-1};
    # a known final round starts nothing ahead
    assert [(s["global_steps"], s["served"]) for s in t.submitted] == [(1, 0), (2, 0), (3, 1), (4, 2)]
    # round r trains with its own step; batches r >= 1 lag one version
    assert driver.trainer.trained == [(0, 0, 1), (1, 0, 2), (2, 1, 3), (3, 2, 4)]
    ahead = [e for e in events if e["event"] == "rl_rollout_ahead"]
    assert [(e["rollout_id"], e["policy_version"]) for e in ahead] == [(1, 0), (2, 1), (3, 2)]
    start = next(e for e in events if e["event"] == "rl_driver_start")
    assert (start["placement"], start["execution_mode"]) == ("fixed-partition", "partitioned-serial")
    profile = driver.profile
    assert (profile.max_policy_age, profile.algorithm_contract) == (1, "bounded-staleness")
    assert driver.capabilities.execution.max_policy_staleness == 1


def test_without_rollout_ahead_the_island_runs_its_serial_rounds(tmp_path, tq):
    island, driver, final, events = _run_local(tmp_path, tq, ahead=0)
    t = island.trainer
    assert [(s["global_steps"], s["served"]) for s in t.submitted] == [(1, 0), (2, 1), (3, 2), (4, 3)]
    assert driver.trainer.trained == [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 4)]
    assert not [e for e in events if e["event"] == "rl_rollout_ahead"]
    assert (driver.profile.max_policy_age, driver.profile.algorithm_contract) == (0, "on-policy")
    assert driver.rollout_ahead == 0


def test_the_serial_profile_and_capabilities_are_unchanged():
    from yeto.rl.adapters.verl.entry import verl_capabilities, with_partitioned_serial
    from yeto.rl.adapters.verl.megatron_ports import _execution_profile
    from yeto.rl.engine.execution_profile import ExecutionProfile

    plan = {"separate": POOLS, "sync": "none", "groups_per_round": 8, "samples_per_group": 4, "max_policy_age": 0}
    before = ExecutionProfile(name="verl-fsdp2-separate", execution_mode="partitioned-serial", outer_protocol="none",
                              algorithm_contract="on-policy", max_policy_age=0, groups_per_batch=8,
                              samples_per_group=4, algorithm_spec_sha256=SPEC.sha256(),
                              extra={"engine": "verl-fsdp2"})
    caps = verl_capabilities("sha256:" + "1" * 64)
    options = separate.driver_options(plan, SPEC, caps)
    assert options["profile"] == before and options["profile"].extra == before.extra
    assert options["capabilities"] == with_partitioned_serial(caps)
    # a plan with an age but no rollout ahead (the fully_async path) gets no age here either
    assert separate.policy_age_of({**plan, "max_policy_age": 2}) == 0
    megatron = _execution_profile(SPEC, {**plan, "megatron": {}})
    assert (megatron.max_policy_age, megatron.algorithm_contract) == (0, "on-policy")
    ahead = _execution_profile(SPEC, {**plan, "megatron": {}, "rollout_ahead": 1, "max_policy_age": 1})
    assert (ahead.max_policy_age, ahead.algorithm_contract) == (1, "bounded-staleness")
    mcaps = separate.with_policy_age(verl_capabilities("sha256:" + "1" * 64, megatron=True),
                                     {"rollout_ahead": 1, "max_policy_age": 1})
    assert mcaps.execution.max_policy_staleness == 1


def test_rollout_ahead_needs_the_separate_mode():
    driver = SimpleNamespace(rollout_ahead=0)
    with pytest.raises(ValueError, match="separate mode"):
        separate.wire_rollout_ahead(driver, {"rollout_ahead": 1})
    separate.wire_rollout_ahead(driver, {"rollout_ahead": 0})
    assert driver.rollout_ahead == 0


# ----------------------------------------------------------------------------- two islands, real syncer
def _run_synced(learner_id: int, plan: dict, tmp_path: Path, queue: _TransferQueue, results: dict) -> None:
    out = tmp_path / f"island-{learner_id}"
    out.mkdir()
    island, tape = _island(out, queue, learner_id)
    sync = None
    try:
        island.full_group = vfs.VerlFullParameterGroup(island)
        sync, policy_state, _stream = vfs.build_full_parameter_sync(island, plan, backend_identity_sha256="c" * 64)
        driver = _driver(island, out / "tape.jsonl", plan, sync, policy_state, learner_id)
        state = driver.run()
        results[learner_id] = {"island": island, "state": state, "driver": driver, "out": out}
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[learner_id] = error
    finally:
        if sync is not None:
            sync.close()


@needs_cargo
def test_two_islands_generate_ahead_through_the_real_syncer(syncer_binary, tmp_path, monkeypatch, tq):
    from test_s27_verl_full_parameter_sync import _layout

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-rollout-ahead-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    rounds, fragments = 4, len(_layout(matrix_merge="rda").fragments.fragments)
    process, port, profile = _start_syncer(syncer_binary, tmp_path, fragments=fragments, rounds=rounds, extra=(
        "--rl-sync-full-parameter-rounds", "async", "--rl-sync-final-cut", "download"))
    results: dict = {}
    try:
        threads = []
        for i in range(2):
            plan = _plan(i, port, profile, rounds=rounds, fragments=fragments, async_rounds=True,
                         matrix_merge="rda", pipeline=2, final_cut="download")
            plan.update(sync="strict", global_rounds=rounds, groups_per_round=PROMPTS, samples_per_group=GROUP,
                        separate=POOLS, rollout_ahead=1, max_policy_age=1)
            threads.append(threading.Thread(target=_run_synced, args=(i, plan, tmp_path, tq, results)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    for learner_id in range(2):
        result = results.get(learner_id)
        assert not isinstance(result, BaseException) and result is not None, (result, log[-3000:])
        island, driver = result["island"], result["driver"]
        events = [json.loads(line) for line in (result["out"] / "tape.jsonl").read_text().splitlines()]
        trained = driver.trainer.trained
        # every round after the first trained a batch generated by the previous publication
        assert trained[0][:2] == (0, 0) and all(v == r - 1 for r, v, _gs in trained[1:]), trained
        assert [gs for _r, _v, gs in trained] == [r + 1 for r, _v, _gs in trained]
        submitted = island.trainer.submitted
        assert [s["served"] for s in submitted[1:len(trained)]] == [r - 1 for r, _v, _gs in trained[1:]]
        assert any(e["event"] == "rl_rollout_ahead" for e in events)
        acks = [e for e in events if e["event"] == "rl_full_parameter_final_ack"]
        assert [a.get("acknowledged") for a in acks] == [True], acks
    for name in SHAPES:  # the downloaded final cut: the same merged weights on both islands
        assert torch.equal(results[0]["island"].model.values()[name], results[1]["island"].model.values()[name])


# ----------------------------------------------------------------------------- launcher and island entry
AHEAD = ("--rl-max-policy-age", "1", "--rl-rollout-ahead", "1")


def _verl(*extra, gpu="modal:2xh200,modal:2xh200"):
    return _launcher_args("ports", ("--rl-backend", "verl", "--reward-function", "gsm8k_reward:score", *extra),
                          gpu=gpu)


def _synced(*extra):
    return _verl("--rl-sync-preset", "full-parameter", "--rl-parameter-layout-sha256", "e" * 64,
                 "--fragments", "4", "--rl-sync-mode", "async", "--rl-placement", "fixed-partition",
                 "--rl-rollout-gpus", "1", *extra)


def _single(*extra):
    return _verl("--tuning", "full", "--rl-single-island-no-sync", "--rl-placement", "fixed-partition",
                 "--rl-rollout-gpus", "1", *extra, gpu="modal:2xh200")


@pytest.mark.parametrize("launch, islands", [(_synced, 2), (_single, 1)])
def test_the_launcher_sends_rollout_ahead_to_separate_mode_islands(launch, islands):
    args = _prepared(launch(*AHEAD))
    for learner_id in range(islands):
        argv = _island_argv(args, learner_id)
        parsed, ignored = island_entry.parse_args(argv)
        assert _flags(ignored) <= MILES_ONLY, _flags(ignored) - MILES_ONLY
        assert (parsed.rl_rollout_ahead, parsed.rl_max_policy_age) == (1, 1)
        # the separate mode with its synchronous trainer: no fully_async limit
        assert island_entry.rollout_ahead_of(parsed) == (1, 0)
        pools = island_entry.fsdp_separate_pools(parsed, {"after": {"GPU": 2.0}})
        assert (pools.trainer_gpus, pools.rollout_gpus) == (1, 1)


def test_the_default_command_has_no_rollout_ahead():
    argv = _island_argv(_prepared(_synced()), 0)
    assert "--rl-rollout-ahead" not in argv and "--rl-max-policy-age" not in argv
    parsed, _ignored = island_entry.parse_args(argv)
    assert island_entry.rollout_ahead_of(parsed) == (0, 0)


@pytest.mark.parametrize("launch, message", [
    (lambda: _verl("--rl-sync-preset", "full-parameter", "--rl-parameter-layout-sha256", "e" * 64,
                   "--fragments", "4", "--rl-sync-mode", "async", *AHEAD, gpu="modal:1xh200,modal:1xh200"),
     "fixed-partition"),
    (lambda: _verl("--rl-single-island-no-sync", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
                   *AHEAD, gpu="modal:2xh200"), "--tuning full"),
    (lambda: _synced("--rl-rollout-ahead", "1"), "--rl-max-policy-age >= 1"),
    (lambda: _synced("--rl-max-policy-age", "1"), "--rl-rollout-ahead 1"),
    (lambda: _verl("--rl-sync-preset", "full-parameter", "--rl-parameter-layout-sha256", "e" * 64,
                   "--fragments", "4", "--rl-sync-mode", "strict", "--rl-placement", "fixed-partition",
                   "--rl-rollout-gpus", "1", *AHEAD), "asynchronous rounds"),
])
def test_the_launcher_refuses_rollout_ahead_off_the_separate_mode(launch, message):
    with pytest.raises(ValueError, match=message):
        _prepared(launch())


def test_eval_runs_with_rollout_ahead_but_not_on_the_fully_async_path():
    from yeto.rl.adapters.verl import run_config_rules

    ahead = SimpleNamespace(rl_eval_interval=2, rl_max_policy_age=1, rl_rollout_ahead=1, tensor_parallel=1,
                            pipeline_parallel=1, expert_parallel=1, rl_sample_role=None, rl_island_sample_role=None)
    run_config_rules.check_eval(ahead)
    with pytest.raises(ValueError, match="fully_async path has no evaluate hook"):
        run_config_rules.check_eval(SimpleNamespace(**{**vars(ahead), "rl_rollout_ahead": 0}))


def test_the_megatron_shape_accepts_rollout_ahead():
    from yeto.rl.adapters.verl import run_config_rules

    shape = SimpleNamespace(tuning="full", rl_single_island_no_sync=True, rl_placement="fixed-partition",
                            rl_max_policy_age=1, rl_rollout_ahead=1)
    assert run_config_rules.megatron_problems(shape) == []
    assert run_config_rules.megatron_problems(SimpleNamespace(**{**vars(shape), "rl_rollout_ahead": 0})) == [
        "--rl-max-policy-age 0, or --rl-rollout-ahead 1"]


@pytest.mark.parametrize("change, message", [
    (dict(parameter_mode="lora"), "--tuning full"),
    (dict(rl_placement="colocated"), "fixed-partition"),
    (dict(rl_max_policy_age=0), "--rl-max-policy-age >= 1"),
    (dict(rl_sample_role="trainer"), "--rl-sample-role"),
    (dict(rl_rollout_ahead=2), "0..1"),
])
def test_the_island_refuses_rollout_ahead_off_the_separate_mode(change, message):
    args = SimpleNamespace(parameter_mode="full", rl_placement="fixed-partition", rl_max_policy_age=1,
                           rl_rollout_ahead=1, rl_sample_role=None)
    for key, value in change.items():
        setattr(args, key, value)
    with pytest.raises(SystemExit, match=message):
        island_entry.rollout_ahead_of(args)
