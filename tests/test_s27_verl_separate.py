"""S27 G-EK: the verl FSDP2 full-parameter island in verl's separate mode (VERL-PARITY #13).

CPU only. The separate mode (:mod:`yeto.rl.adapters.verl.separate`) is the
engine-neutral part of G-DC's Megatron island: a trainer pool, standalone vLLM
servers on their own cards and the NCCL checkpoint engine. These tests check:

* the pools from the launcher's island flags, and the Hydra overrides of an
  FSDP2 island in that mode (the colocated keys are replaced, the rest stays);
* a launch with ``--rl-placement fixed-partition`` reaches the island with the
  pool flags, and the island plans the separate mode from them;
* the Megatron island still builds byte-identical overrides (it now uses the
  shared module);
* two islands through the launcher-built Rust syncer, each a real
  ``IslandDriver`` over the separate ports: the full-parameter stream reads the
  cut from the trainer workers only, every publication goes through the
  standalone checkpoint manager (at least three per island), and the rollout
  side always serves the published weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_verl_full_parameter_launch import _island_argv, _prepared
from test_s27_verl_full_parameter_sync import (  # noqa: F401 - syncer_binary is a fixture
    SHAPES, TOKENS, _Island, _Model, _plan, _start_syncer, _step, needs_cargo, syncer_binary,
)
from yeto.rl.adapters.verl import config as vconf
from yeto.rl.adapters.verl import full_parameter as fp
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.adapters.verl import island_entry, separate

SEPARATE_GPU = "modal:2xh200,modal:2xh200"


def _pool_args(**kw):
    values = dict(rl_island_gpus_per_node=None, actor_num_gpus_per_node=1, actor_num_nodes=1, rollout_num_gpus=1,
                  rollout_num_gpus_per_engine=1, sglang_mem_fraction_static=None)
    values.update(kw)
    return SimpleNamespace(**values)


# ----------------------------------------------------------------------------- pools and overrides
def test_pools_come_from_the_launchers_island_flags():
    pools = separate.pools_from_args(_pool_args(), 2)
    assert (pools.trainer_gpus, pools.rollout_gpus, pools.rollout_tp) == (1, 1, 1)
    assert pools.gpu_memory_utilization == 0.8  # the shared default (G-DC's)
    assert separate.pools_from_args(_pool_args(sglang_mem_fraction_static=0.7), 2).gpu_memory_utilization == 0.7
    # the Megatron D0 shape: 8 trainer + 8 rollout cards on two nodes, TP2 engines
    d0 = separate.pools_from_args(_pool_args(actor_num_gpus_per_node=8, rollout_num_gpus=8,
                                             rollout_num_gpus_per_engine=2), 16)
    assert (d0.trainer_nodes, d0.trainer_gpus_per_node, d0.rollout_nodes, d0.rollout_gpus_per_node,
            d0.rollout_tp) == (1, 8, 1, 8, 2)
    with pytest.raises(SystemExit, match="!= 岛内 Ray"):
        separate.pools_from_args(_pool_args(), 3)
    with pytest.raises(SystemExit, match="--rollout-num-gpus 0"):
        separate.pools_from_args(_pool_args(rollout_num_gpus=0), 1)
    with pytest.raises(separate.SeparatePoolError, match="rollout TP"):
        separate.validate_pools(separate.SeparatePools(1, 1, 1, 1, rollout_tp=2))
    with pytest.raises(separate.SeparatePoolError, match="gpu_memory_utilization"):
        separate.validate_pools(separate.SeparatePools(1, 1, 1, 1, rollout_tp=1, gpu_memory_utilization=1.0))


def _run(**kw) -> vconf.VerlRunConfig:
    values = dict(model_path="/m", train_file="/t", val_file="/v", out_dir="/o", parameter_mode="full",
                  lora_rank=0, lora_alpha=0, lr=1e-6, groups_per_round=16, samples_per_group=4,
                  max_prompt_len=512, max_response_len=512, seed=1, total_rounds=8, correction="tis",
                  tis_upper=2.0, n_gpus=1, max_model_len=1024, experiment_name="x")
    values.update(kw)
    return vconf.VerlRunConfig(**values)


def _value(overrides, key):
    hits = [o.split("=", 1)[1] for o in overrides if o.split("=", 1)[0].lstrip("+") == key]
    assert len(hits) == 1, (key, hits)
    return hits[0]


def test_an_fsdp2_island_in_the_separate_mode_replaces_only_the_pool_keys():
    pools = separate.SeparatePools(1, 1, 1, 1, rollout_tp=1, gpu_memory_utilization=0.7)
    colocated = vconf.build_overrides(_run())
    out = separate.with_separate_pools(colocated, pools)
    replaced = [o for o in colocated if o.startswith(separate.POOL_PREFIXES)]
    assert len(replaced) == len(separate.POOL_PREFIXES)
    assert [o for o in colocated if o not in replaced] == out[:len(colocated) - len(replaced)]
    assert _value(out, "trainer.v1.trainer_mode") == separate.TRAINER_MODE
    assert _value(out, "actor_rollout_ref.hybrid_engine") == "False"
    assert _value(out, "actor_rollout_ref.actor.strategy") == "fsdp2"  # the engine stays FSDP2
    assert _value(out, "actor_rollout_ref.actor.fsdp_config.model_dtype") == "fp32"  # FP32 masters for the stream
    assert _value(out, "actor_rollout_ref.rollout.checkpoint_engine.backend") == "nccl"
    assert _value(out, "actor_rollout_ref.rollout.gpu_memory_utilization") == "0.7"
    assert (_value(out, "trainer.n_gpus_per_node"), _value(out, "actor_rollout_ref.rollout.n_gpus_per_node")) == \
        ("1", "1")
    # verl's separate_async asserts train_batch_size == parameter_sync_step * ppo_mini_batch_size
    assert _value(out, "data.train_batch_size") == _value(out, "actor_rollout_ref.actor.ppo_mini_batch_size")


def test_the_megatron_island_builds_the_same_overrides_from_the_shared_module():
    from yeto.rl.adapters.verl import megatron_engine as me

    shape = me.MegatronShape(num_experts=128)
    out = me.build_overrides(_run(n_gpus=8, groups_per_round=32, samples_per_group=16), shape,
                             eps_clip=0.2, eps_clip_high=0.28)
    at = out.index(f"trainer.v1.trainer_mode={separate.TRAINER_MODE}")
    assert out[at:at + 12] == separate.separate_overrides(shape.pools())
    assert me.ASSERTED_KEYS[6:11] == separate.ASSERTED_KEYS
    assert me.TRAINER_MODE == separate.TRAINER_MODE


# ----------------------------------------------------------------------------- launcher -> island
def _separate_launch(*extra):
    return _launcher_args("ports", (
        "--rl-backend", "verl", "--rl-sync-preset", "full-parameter", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", "4", "--reward-function", "gsm8k_reward:score", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", *extra), gpu=SEPARATE_GPU)


def test_a_fixed_partition_launch_reaches_the_island_as_the_separate_mode(tmp_path):
    args = _prepared(_separate_launch())
    argv = _island_argv(args, 1)
    parsed, ignored = island_entry.parse_args(argv)
    pool_flags = {"--rl-placement", "--rollout-num-gpus", "--actor-num-nodes", "--actor-num-gpus-per-node",
                  "--rollout-num-gpus-per-engine", "--sglang-mem-fraction-static", "--parameter-mode"}
    assert pool_flags <= set(argv) and not pool_flags & set(ignored)
    assert (parsed.parameter_mode, parsed.rl_placement, parsed.rollout_num_gpus,
            parsed.actor_num_gpus_per_node) == ("full", "fixed-partition", 1, 1)
    pools = island_entry.fsdp_separate_pools(parsed, {"after": {"GPU": 2.0}})
    assert (pools.trainer_gpus, pools.rollout_gpus) == (1, 1)
    run, overrides = island_entry.fsdp_island_run(
        parsed, model_path=str(tmp_path), train_file="/t", val_file="/v", out_dir=str(tmp_path / "o"),
        algo={"correction": "none", "tis_upper": 2.0}, chat={}, ray_info={"after": {"GPU": 2.0}}, max_prompt=512,
        limit=0, pools=pools)
    assert run.n_gpus == 1 and run.lora_rank == 0
    assert _value(overrides, "trainer.v1.trainer_mode") == separate.TRAINER_MODE
    # a colocated launch keeps the colocated island (no pools)
    colocated, _ = island_entry.parse_args(_island_argv(_prepared(_launcher_args("ports", (
        "--rl-backend", "verl", "--rl-sync-preset", "full-parameter", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", "4", "--reward-function", "gsm8k_reward:score"), gpu="modal:1xh200,modal:1xh200")), 0))
    assert island_entry.fsdp_separate_pools(colocated, {"after": {"GPU": 1.0}}) is None


def test_a_lora_islands_fixed_partition_is_not_the_separate_mode():
    parsed = SimpleNamespace(parameter_mode="lora", rl_placement="fixed-partition")
    assert island_entry.fsdp_separate_pools(parsed, {"after": {"GPU": 2.0}}) is None


def test_the_separate_mode_refuses_standby_cards():
    with pytest.raises(ValueError, match="standby"):
        _prepared(_separate_launch("--rl-standby-gpus", "1"))


# ----------------------------------------------------------------------------- ports over a real driver
class _CheckpointManager:
    """verl's ``standalone_checkpoint_manager``: copies the trainer's weights to the
    rollout servers (here: one dict), as the NCCL checkpoint engine does."""

    def __init__(self, island):
        self.island = island
        self.calls: list[int] = []
        self.rollout_weights: dict[str, torch.Tensor] = {}

    def update_weights(self, version):
        self.calls.append(int(version))
        self.rollout_weights = self.island.model.values()
        return {"sync/bytes": 4.0 * sum(v.numel() for v in self.rollout_weights.values())}


class _SeparateIsland(_Island):
    """``VerlIsland`` of a separate-mode island: ``call_workers`` reaches the trainer
    worker only; the rollout servers are behind the checkpoint manager."""

    def __init__(self, model, tmp_path: Path, learner_id: int, tape):
        super().__init__(model)
        self.learner_id = learner_id
        self.out = tmp_path
        self.published = None
        self.emit = lambda event, **fields: tape.append({"event": event, **fields})
        self.trainer = SimpleNamespace(standalone_checkpoint_manager=_CheckpointManager(self))
        self.trainer_reads: list[str] = []

    def call_workers(self, fn, *args):
        self.trainer_reads.append(fn.__name__)
        return super().call_workers(fn, *args)

    call_workers_shared = call_workers

    def append_jsonl(self, name, record):
        with (self.out / name).open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _weights_hash(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode() + values[name].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _ports(island: _SeparateIsland, learner_id: int, plan: dict, served: list):
    from yeto.rl.contracts import LocalStepReceipt
    from yeto.rl.engine.driver import TrainStepMetrics, policy_token
    from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle

    from yeto.rl.adapters.verl.ports_impl import MEMBER

    class Rollout:
        def generate(self, rollout_id, *, expected_policy_version=None):
            version, digest = island.published
            assert version == rollout_id
            # the standalone servers serve exactly the trainer weights of the last publication
            served.append((rollout_id, _weights_hash(island.trainer.standalone_checkpoint_manager.rollout_weights)))
            token = policy_token(version, digest)
            ids = (f"l{learner_id}-r{rollout_id}-s0", f"l{learner_id}-r{rollout_id}-s1")
            groups = (GroupMetadata(f"l{learner_id}-r{rollout_id}-g0", ids, token, 0.5, 0.5, TOKENS),)
            return RolloutBatchHandle(rollout_id, version, digest, groups, 1, 0, payload=None)

        def abort(self):
            return None

        def members(self):
            return frozenset({MEMBER})

    class Trainer:
        publish_offloaded = False

        def onload(self):
            return None

        def offload(self):
            return None

        def train_step(self, batch):
            with torch.no_grad():  # one optimizer step on the trainer cards
                for name, parameter in island.model.named_parameters():
                    parameter.add_(_step(learner_id, batch.rollout_id)[name])
            ids = tuple(s for g in batch.groups for s in g.sample_ids)
            batch_hash = hashlib.sha256("|".join(ids).encode()).hexdigest()
            receipt = LocalStepReceipt(
                algorithm="grpo", learner_id=learner_id, learner_generation=0,
                base_policy_version=batch.policy_version, base_policy_hash=batch.policy_hash,
                input_batch_hash=batch_hash, trajectory_ids=ids, trained_tokens=TOKENS, optimizer_steps=1,
                optimizer_step_succeeded=True, parameter_layout_hash=plan[vfs.PLAN_KEY]["parameter_layout_sha256"])
            island.full_group.record_round(receipt, vfs.VerlRoundEvidence(
                batch.rollout_id, batch.policy_hash, batch_hash, TOKENS, ids))
            return receipt

        def step_metrics(self):
            return TrainStepMetrics(grad_norm=0.5, loss=0.1, pg_loss=0.1, lr=1e-6, train_step=0, applied_lrs=(1e-6,))

    return Rollout(), Trainer()


def _run_separate_island(learner_id: int, plan: dict, tmp_path: Path, results: dict) -> None:
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.adapters.verl.ports_impl import VerlFullParameterPublisher
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.driver import EventTape, IslandDriver

    out = tmp_path / f"island-{learner_id}"
    out.mkdir()
    tape: list[dict] = []
    island = _SeparateIsland(_Model(seed=0), out, learner_id, tape)
    served: list = []
    sync = None
    try:
        # trainer._full_parameter_ports (verl-only import) for a synced island: the stream's
        # sync and policy state, and the publisher by plan
        island.full_group = vfs.VerlFullParameterGroup(island)
        sync, policy_state, _stream = vfs.build_full_parameter_sync(island, plan, backend_identity_sha256="c" * 64)
        publisher = separate.full_parameter_publisher(island, plan, VerlFullParameterPublisher)
        assert isinstance(publisher, separate.SeparatePublisher)
        spec = AlgorithmSpec()
        rollout, trainer = _ports(island, learner_id, plan, served)
        options = separate.driver_options(plan, spec, verl_capabilities("sha256:" + "1" * 64))
        driver = IslandDriver(learner_id=learner_id, rollout=rollout, trainer=trainer, policy_state=policy_state,
                              publisher=publisher, algorithm=spec, sync=sync,
                              events=EventTape(out / "tape.jsonl", learner_id), **options)
        state = driver.run()
        results[learner_id] = {"island": island, "state": state, "served": served, "tape": tape,
                               "rounds": driver.rounds_completed, "out": out}
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[learner_id] = error
    finally:
        if sync is not None:
            sync.close()


@needs_cargo
@pytest.mark.parametrize("mode", ["sweep", "async"])
def test_two_separate_verl_islands_publish_every_round_through_the_real_syncer(syncer_binary, tmp_path,
                                                                                monkeypatch, mode):
    from test_s27_verl_full_parameter_sync import _layout

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-separate-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    final_cut, async_rounds = "download", mode == "async"
    merge = "rda" if async_rounds else "avg"
    rounds, fragments = 4, len(_layout(matrix_merge=merge).fragments.fragments)
    # async: the launcher's S27 defaults (quorum 1, two rounds in flight, RDA, local init)
    process, port, profile = _start_syncer(syncer_binary, tmp_path, fragments=fragments, rounds=rounds, extra=(
        "--rl-sync-full-parameter-rounds", mode, "--rl-sync-final-cut", final_cut))
    pools = separate.SeparatePools(1, 1, 1, 1, rollout_tp=1).to_dict()
    results: dict = {}
    try:
        plans = []
        for i in range(2):
            plan = _plan(i, port, profile, rounds=rounds, fragments=fragments, async_rounds=async_rounds,
                         matrix_merge=merge, pipeline=2 if async_rounds else 1, final_cut=final_cut)
            plan.update(sync="strict", global_rounds=rounds, groups_per_round=1, samples_per_group=2,
                        separate=pools)
            plans.append(plan)
        threads = [threading.Thread(target=_run_separate_island, args=(i, plans[i], tmp_path, results))
                   for i in range(2)]
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
        island = result["island"]
        manager = island.trainer.standalone_checkpoint_manager
        # every publication went through the standalone checkpoint engine, in version order,
        # and there were at least three (the GPU check's criterion: the 2nd and later must not hang)
        assert manager.calls == list(range(len(manager.calls))) and len(manager.calls) >= 3, manager.calls
        records = [json.loads(line) for line in (result["out"] / f"verl-publish-{learner_id}.jsonl").read_text()
                   .splitlines()]
        assert [r["policy_version"] for r in records] == manager.calls
        assert {r["transport"] for r in records} == {separate.WEIGHT_TRANSPORT}
        # every round generated on the weights published for it (the rollout side never
        # saw a later or a half-applied state)
        assert [rid for rid, _h in result["served"]] == list(range(result["rounds"]))
        # the stream and the policy state read and wrote the trainer workers only
        assert {"remote_manifest", "remote_export_for_reader"} <= set(island.trainer_reads)
        start = next(json.loads(line) for line in (result["out"] / "tape.jsonl").read_text().splitlines()
                     if json.loads(line)["event"] == "rl_driver_start")
        assert (start["placement"], start["execution_mode"]) == ("fixed-partition", "partitioned-serial")
        assert result["rounds"] == rounds or (async_rounds and result["rounds"] >= rounds)
    if not async_rounds:  # strict sweeps: both islands served the same merged policy each round
        assert results[0]["served"] == results[1]["served"]
    # the final cut is downloaded: both islands end on the same merged policy
    assert results[0]["state"].policy_tensor_hash() == results[1]["state"].policy_tensor_hash()
    for name in SHAPES:
        assert torch.equal(results[0]["island"].model.values()[name], results[1]["island"].model.values()[name])


def test_a_colocated_island_keeps_its_driver_options_and_publisher():
    from yeto.rl.adapters.verl.ports_impl import VerlFullParameterPublisher

    assert separate.driver_options({"sync": "strict"}, None, None) == {}
    assert isinstance(separate.full_parameter_publisher(object(), {}, VerlFullParameterPublisher),
                      VerlFullParameterPublisher)


def test_the_separate_publisher_refuses_a_state_the_trainer_moved_past(tmp_path):
    from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState, WeightsChanged

    island = _SeparateIsland(_Model(seed=0), tmp_path, 0, [])

    def holds(_state):
        raise WeightsChanged("trained since")

    state = TrainerResidentState(layout="fragments", base_model_revision="a" * 40, config_hash="b" * 64,
                                 layout_hash="c" * 64, policy_version=2,
                                 digest=PolicyDigest("d" * 64, "e" * 64, 64, ()), materialize_fn=lambda v: None,
                                 weights_mark={"v": 1}, holds_fn=holds)
    with pytest.raises(WeightsChanged):
        separate.SeparatePublisher(island).publish(state)
    assert island.trainer.standalone_checkpoint_manager.calls == []


def test_parse_args_accepts_the_separate_flags():
    args, ignored = island_entry.parse_args([
        "--model", "m", "--model-revision", "a" * 40, "--data", "d", "--learner-id", "0", "--reward-function", "r",
        "--global-rounds", "1", "--groups-per-round", "1", "--samples-per-group", "1",
        "--rollout-max-response-len", "1", "--event-tape", "t", "--inner-lr", "1", "--seq-len", "1", "--seed", "1",
        "--parameter-mode", "full", "--rl-placement", "fixed-partition", "--rollout-num-gpus", "1",
        "--actor-num-gpus-per-node", "1", "--rollout-num-gpus-per-engine", "1"])
    assert ignored == [] and isinstance(args, argparse.Namespace)
    assert fp.MASTER_DTYPE == "torch.float32"


# ----------------------------------------------------------------------------- VERL-PARITY #14, first step
def test_a_full_parameter_trainer_releases_its_weights_through_the_buffer_publisher(tmp_path):
    """The full-parameter release (``buffer_ports.full_release_weights`` + ``merged_release.
    write_full_checkpoint``): the fully_async trainer actor's ``yeto_call_workers`` hook gives the
    full-parameter group the trainer workers, the island's HF writer (G-DS) writes the release, version
    0 (the base) is not released, a trained version is, and the release loads as the trainer's weights."""
    import asyncio

    from safetensors.torch import load_file

    from yeto.rl.adapters.verl import buffer_feed as bf
    from yeto.rl.adapters.verl.buffer_ports import BufferReleasePublisher, full_release_weights
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland
    from yeto.rl.adapters.verl.hf_export import VerlHfWriter
    from yeto.rl.engine.hf_export import COMPLETE_MARKER
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.local_learner import ComponentIdentity

    model = _Model(seed=0)
    worker = _Island(model).worker  # the trainer worker behind the actor

    class Actor:  # YetoBufferTrainer: the fully_async trainer actor
        param_version = 0

        def __call__(self, name, *args, timeout=None):
            if name == "yeto_call_workers":
                fn, *rest = args
                return [fn(worker, *rest)]
            return getattr(self, name)(*args)

        def yeto_param_version(self):
            return self.param_version

    actor = Actor()
    island = FullyAsyncIsland(actor, learner_id=0, limit=1, required=1, out_dir=str(tmp_path / "out"),
                              readback_dir=str(tmp_path / "rb"))
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"torch_dtype": "bfloat16"}')
    (base / "tokenizer.json").write_text("{}")
    group = vfs.VerlFullParameterGroup(island)
    state_port = vfs.VerlFullParameterLocalState(
        group, component=ComponentIdentity(role="actor", model_revision="a" * 40, config_hash="b" * 64),
        max_fragment_bytes=64)
    store = CheckpointStore(tmp_path / "store")
    feed = bf.BufferFeed(FileSampleBuffer(store, batch_size=1), vmap=lambda: island.vmap,
                         shape=bf.RowShape(prompt_length=4, response_length=4, pad_token_id=0),
                         add_group=print, record=print)
    releases = PolicyReleaseStore(store)
    publisher = BufferReleasePublisher(island, releases, feed, base_dir=base,
                                       weights=full_release_weights(VerlHfWriter(group, base, run=asyncio.run)))
    publisher.publish(state_port.export(policy_version=0))
    assert releases.latest() is None  # version 0 is the base checkpoint: the samplers start on it
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.25)
    actor.param_version = 1
    state = state_port.export(policy_version=1)
    result = publisher.publish(state)
    latest = releases.latest()
    assert (latest.version, latest.policy_hash) == (1, state.policy_tensor_hash())
    landed = Path(store.root) / releases.release_dir / latest.path
    # the weights and the base's non-weight files, plus the store's own release record
    assert sorted(p.name for p in landed.iterdir()) == ["RELEASE.json", "config.json", "model.safetensors",
                                                        "tokenizer.json"]
    assert COMPLETE_MARKER not in {p.name for p in landed.iterdir()}
    weights = load_file(str(landed / "model.safetensors"))
    assert sorted(weights) == sorted(SHAPES)
    for name, value in model.values().items():
        assert weights[name].dtype == torch.bfloat16 and torch.equal(weights[name], value.to(torch.bfloat16))
    assert result.manifest.payload_bytes == sum(latest.files.values())
    record = json.loads((tmp_path / "out" / "verl-buffer-release-0.jsonl").read_text().splitlines()[-1])
    assert record["released"] and record["merge"]["files"] == ["config.json", "model.safetensors", "tokenizer.json"]
