"""Engine ports of the verl Megatron island (S27 G-DC; runs inside verl's task runner).

One island, full parameters, no outer sync (``--rl-single-island-no-sync``):
the trainer cards run verl's Megatron engine, the rollout cards run
standalone vLLM servers (:mod:`.megatron_engine`, :mod:`.separate_trainer`).
Next to the FSDP2 ports of :mod:`.ports_impl`:

* rollout: ``ports_impl.VerlRolloutPool`` unchanged (verl submits the round's
  prompts, the standalone servers generate, the replay buffer returns them);
* trainer: ``ports_impl.VerlTrainerGroup`` plus a count of trained rounds
  (the weights version the resident policy handle is checked against);
* policy state (:class:`MegatronPolicyState`): the weights stay in the
  trainer. Each Megatron rank hashes the parameters it owns (dense ones on
  data-parallel rank 0, expert ones on expert-data-parallel rank 0, keyed by
  name and TP/PP/EP coordinates); the island combines the per-rank hashes
  into the policy hash. No tensor leaves the trainer: there is no export or
  apply (no outer sync on this island);
* publisher (:class:`MegatronPublisher`): verl's standalone NCCL checkpoint
  engine. There is no read-back of what vLLM loaded (the LoRA read-back of
  ``vllm_readback`` does not cover full weights); the train/rollout log-prob
  criteria of every round are the check that the rollout runs the published
  weights;
* placement: fixed partition (trainer pool, rollout pool).

``run_megatron_island`` wires them into the neutral ``IslandDriver``; the
final model can be exported to the run's store (``--rl-export-hf-rounds``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from yeto.rl.contracts import InferencePublicationManifest
from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState, WeightsChanged
from yeto.rl.engine.ports import PlacementDescription, PublicationResult
from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS

from .ports_impl import MEMBER, VerlIsland, VerlRolloutPool, VerlTrainerGroup, _inner_worker

DIGEST_PREFIX = b"yeto-verl-megatron-policy-v1\0"
PAYLOAD_PREFIX = b"yeto-verl-megatron-payload-v1\0"
HASH_THREADS = 8
READBACK_STATUS = "ENGINE_READBACK_NOT_IMPLEMENTED"


class MegatronTensorExportError(RuntimeError):
    """The Megatron island keeps its weights in the trainer (no export / apply)."""


# ----------------------------------------------------------------------------- worker side
def _parallel_coordinates() -> dict[str, int]:
    from megatron.core import parallel_state as mpu

    def rank(name: str, **kwargs) -> int:
        fn = getattr(mpu, name, None)
        return int(fn(**kwargs)) if callable(fn) else 0

    return {"tp": rank("get_tensor_model_parallel_rank"), "pp": rank("get_pipeline_model_parallel_rank"),
            "ep": rank("get_expert_model_parallel_rank"),
            "dp": rank("get_data_parallel_rank", with_context_parallel=True),
            "edp": rank("get_expert_data_parallel_rank")}


def is_expert_param(name: str, param: Any) -> bool:
    """Megatron marks expert parameters with ``allreduce=False`` (their gradients are
    reduced over the expert data-parallel group only); the name rule is the fallback."""
    flag = getattr(param, "allreduce", None)
    if flag is not None:
        return not bool(flag)
    return ".experts." in name


def owned_parameters(chunks, coords: dict[str, int]):
    """``(key, param)`` of the parameters this rank owns: a dense parameter on
    data-parallel rank 0, an expert parameter on expert-data-parallel rank 0. The key
    names the model chunk, the parameter and the coordinates that make it unique."""
    for index, chunk in enumerate(chunks):
        for name, param in chunk.named_parameters():
            expert = is_expert_param(name, param)
            if expert and coords["edp"] != 0:
                continue
            if not expert and coords["dp"] != 0:
                continue
            where = f"tp{coords['tp']}.pp{coords['pp']}" + (f".ep{coords['ep']}" if expert else "")
            yield f"{index}:{name}@{where}", param


def _chunks(worker) -> list:
    module = worker.actor.engine.module
    return list(module) if isinstance(module, (list, tuple)) else [module]


def remote_megatron_layout(worker_dict) -> list[list]:
    """``[key, shape, dtype]`` of the parameters this rank owns (no hashing)."""
    worker = _inner_worker(worker_dict)
    coords = _parallel_coordinates()
    return [[key, list(param.shape), str(param.dtype)] for key, param in owned_parameters(_chunks(worker), coords)]


def remote_megatron_digest(worker_dict) -> dict:
    """sha256 of every owned parameter's bytes (as the trainer holds them, bf16),
    copied to host and hashed in ``HASH_THREADS`` threads (hashlib releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    started = time.monotonic()
    worker = _inner_worker(worker_dict)
    coords = _parallel_coordinates()
    rows, futures, total = [], [], 0

    def sha(raw) -> str:
        return hashlib.sha256(memoryview(raw)).hexdigest()

    with torch.no_grad(), ThreadPoolExecutor(max_workers=HASH_THREADS, thread_name_prefix="yeto-meg-hash") as pool:
        for key, param in owned_parameters(_chunks(worker), coords):
            host = param.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy()
            total += int(host.nbytes)
            rows.append([key, list(param.shape), str(param.dtype)])
            futures.append(pool.submit(sha, host))
        for row, future in zip(rows, futures):
            row.append(future.result())
    rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
    return {"rank": rank, "coords": coords, "tensors": rows, "bytes": total,
            "seconds": round(time.monotonic() - started, 3)}


HF_STAGING = "yeto-verl-hf-export"


def _save_hf(worker, directory: Path, model_path: str, rank: int) -> None:
    """Megatron-Bridge writes the HF weights of the current model into ``directory``
    (collective: every trainer rank calls it); rank 0 adds the checkpoint's config,
    tokenizer and generation files, so the directory loads as the base repo does."""
    import shutil

    import torch

    engine = worker.actor.engine
    directory.mkdir(parents=True, exist_ok=True)
    engine.bridge.save_hf_weights(engine.module, str(directory))
    if rank == 0:
        for src in sorted(Path(model_path).iterdir()):
            if src.is_file() and not src.name.endswith((".safetensors", ".safetensors.index.json", ".bin")) \
                    and not (directory / src.name).exists():
                shutil.copy2(src, directory / src.name)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def remote_hf_export(worker_dict, config: dict, rollout_id: int, policy_version: int, final: bool,
                     model_path: str) -> dict | None:
    """``--rl-export-hf-rounds`` on the trainer workers: every rank joins the save
    into one staging directory of the trainer node; rank 0 runs the neutral
    ``HfRoundExports`` (marker, upload to the store, retries) and returns its record.
    The files never leave the trainer node before the upload."""
    import tempfile

    import torch

    from yeto.rl.engine.hf_export import ExportRounds, HfRoundExports
    from yeto.rl.engine.resume import store_for

    worker = _inner_worker(worker_dict)
    rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
    staging = Path(tempfile.gettempdir()) / HF_STAGING

    def write(_rollout_id: int, directory: Path) -> None:
        _save_hf(worker, Path(directory), model_path, rank)

    if rank == 0:
        exports = HfRoundExports(ExportRounds.parse(config["rounds"]), store_for(config["store"]), write,
                                 staging=staging)
        return exports.export(int(rollout_id), policy_version=int(policy_version), final=bool(final))
    write(rollout_id, staging / f"round-{int(rollout_id) + 1}")  # the directory rank 0 exports
    return None


# ----------------------------------------------------------------------------- island side
def call_workers(trainer, fn, *args) -> list:
    """``fn(worker_dict, *args)`` on every trainer worker (``VerlIsland.call_workers``
    before the island object exists)."""
    import ray

    return ray.get([w.__ray_call__.remote(fn, *args) for w in trainer.actor_rollout_wg.workers])


def combine_digests(records: list[dict]) -> PolicyDigest:
    """One policy digest from the per-rank records of :func:`remote_megatron_digest`."""
    rows = sorted(row for record in records for row in record["tensors"])
    keys = [row[0] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("two Megatron ranks own the same parameter key")
    lines = "\n".join(json.dumps(row, separators=(",", ":")) for row in rows).encode()
    policy = hashlib.sha256(DIGEST_PREFIX + lines).hexdigest()
    payload = hashlib.sha256(PAYLOAD_PREFIX + policy.encode()).hexdigest()
    return PolicyDigest(policy_tensor_hash=policy, payload_hash=payload,
                        payload_bytes=sum(int(r["bytes"]) for r in records),
                        specs=tuple((row[0], tuple(int(d) for d in row[1])) for row in rows),
                        hash_seconds=max((float(r.get("seconds", 0.0)) for r in records), default=0.0))


def layout_of(records: list[list[list]]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Sorted (key, shape) of :func:`remote_megatron_layout` over all ranks."""
    rows = sorted((row[0], tuple(int(d) for d in row[1])) for rank_rows in records for row in rank_rows)
    if len({key for key, _ in rows}) != len(rows):
        raise ValueError("two Megatron ranks own the same parameter key")
    return tuple(rows)


LAYOUT_PREFIX = b"yeto-verl-megatron-layout-v1\0"


def layout_hash_of(specs) -> str:
    """Hash of the island's parameter layout: the sorted (key, shape) of every owned
    parameter (the keys are Megatron names with TP/PP/EP coordinates, not canonical
    PEFT names, so ``policy_digest.layout_hash_of`` does not apply)."""
    rows = [[str(key), [int(d) for d in shape]] for key, shape in sorted(specs)]
    return hashlib.sha256(LAYOUT_PREFIX + json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def config_hash(model_path: str, shape: dict) -> str:
    """Identity of the trained model: the checkpoint's ``config.json`` and the parallel
    shape (the parameter keys carry TP/PP/EP coordinates)."""
    text = (Path(model_path) / "config.json").read_text()
    return hashlib.sha256(json.dumps({"config": json.loads(text), "shape": shape},
                                     sort_keys=True).encode()).hexdigest()


class MegatronIsland(VerlIsland):
    """``VerlIsland`` plus the trained-round counter (the weights version)."""

    def __init__(self, trainer, *, model_path: str, **kwargs):
        super().__init__(trainer, **kwargs)
        self.model_path = model_path
        self.trained_rounds = 0
        self.process_id = f"{os.getpid()}-{time.time_ns()}"


class MegatronTrainerGroup(VerlTrainerGroup):
    def train_step(self, batch):
        receipt = super().train_step(batch)
        self.island.trained_rounds += 1
        return receipt


class MegatronPolicyState:
    supports_unverified_rounds = False  # every publication hashes the policy
    supports_deferred_digest = False

    def __init__(self, island: MegatronIsland):
        self.island = island

    def _mark(self) -> dict:
        return {"process_id": self.island.process_id, "version": int(self.island.trained_rounds)}

    def _holds(self, state) -> None:
        if dict(state.weights_mark or {}) != self._mark():
            raise WeightsChanged(f"the trainer trained since policy v{state.policy_version} was hashed "
                                 f"({state.weights_mark} -> {self._mark()})")

    def _refuse(self, policy_version: int):
        raise MegatronTensorExportError(
            "verl Megatron island: the weights stay in the trainer (no tensor export, no outer sync)")

    def export_digest(self, *, policy_version: int, deferred: bool = False) -> TrainerResidentState:
        isl, started = self.island, time.monotonic()
        digest = combine_digests(isl.call_workers(remote_megatron_digest))
        if layout_hash_of(digest.specs) != isl.layout_hash:
            raise MegatronTensorExportError("the Megatron parameter layout changed since the island start")
        print(f"[rl] verl megatron digest v{policy_version}: total={time.monotonic() - started:.2f}s "
              f"hash={digest.hash_seconds:.2f}s bytes={digest.payload_bytes}", flush=True)
        return TrainerResidentState(
            layout=LAYOUT_FRAGMENTS, base_model_revision=isl.base_model_revision,
            config_hash=isl.lora_config_hash, layout_hash=isl.layout_hash,
            policy_version=int(policy_version), digest=digest, materialize_fn=self._refuse,
            weights_mark=self._mark(), holds_fn=self._holds)

    def export(self):
        self._refuse(-1)

    def apply(self, state, *, optimizer: str, local_step: int) -> None:
        self._refuse(int(getattr(state, "policy_version", -1)))


class MegatronPublisher:
    def __init__(self, island: MegatronIsland):
        self.island = island

    def publish(self, state) -> PublicationResult:
        isl, t = self.island, self.island.trainer
        version = int(state.policy_version)
        check = getattr(state, "check_holds", None)
        if callable(check):
            check()  # the trainer still holds the hashed weights
        started = time.monotonic()
        sync_metrics = dict(t.standalone_checkpoint_manager.update_weights(version) or {})
        update_seconds = time.monotonic() - started
        digest = state.policy_tensor_hash()
        payload_hash, payload_bytes = state.payload_digest
        record = {"policy_version": version, "status": READBACK_STATUS, "update_seconds": update_seconds,
                  "policy_tensor_hash": digest, "payload_bytes": int(payload_bytes),
                  "transport": "verl-nccl-checkpoint-engine",
                  "sync_metrics": {k: v for k, v in sync_metrics.items() if isinstance(v, (int, float, str))}}
        isl.append_jsonl(f"verl-publish-{isl.learner_id}.jsonl", record)
        isl.emit("rl_publication_check", **record)
        isl.published = (version, digest)
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=version,
            target_policy_hash=digest,
            target_manifest_hash=hashlib.sha256(f"{version}:{digest}".encode()).hexdigest(),
            payload_hash=payload_hash, payload_bytes=int(payload_bytes), complete=True)
        return PublicationResult(manifest, frozenset({MEMBER}))


class MegatronPlacement:
    def __init__(self, shape: dict):
        self.shape = dict(shape)

    def describe(self) -> PlacementDescription:
        trainer = tuple(f"trainer:{i}" for i in range(int(self.shape["trainer_gpus"])))
        rollout = tuple(f"rollout:{i}" for i in range(int(self.shape["rollout_gpus"])))
        return PlacementDescription("fixed-partition", trainer, rollout,
                                    extra={"weight_transport": "verl-nccl-checkpoint-engine",
                                           "shape": self.shape})


class MegatronHfExports:
    """The driver's ``hf_exports`` (``.due`` / ``.export``) of a Megatron island: the
    export runs on the trainer workers (:func:`remote_hf_export`)."""

    def __init__(self, island: MegatronIsland, config: dict):
        from yeto.rl.engine.hf_export import ExportRounds

        self.island = island
        self.config = dict(config)
        self.rounds = ExportRounds.parse(self.config["rounds"])

    def due(self, rollout_id: int, *, final: bool) -> bool:
        return self.rounds.due(rollout_id, final=final)

    def export(self, rollout_id: int, *, policy_version: int, final: bool) -> dict:
        results = self.island.call_workers(remote_hf_export, self.config, int(rollout_id), int(policy_version),
                                           bool(final), self.island.model_path)
        return next(r for r in results if r is not None)


def _execution_profile(spec, plan: dict):
    from yeto.rl.engine.execution_profile import ExecutionProfile

    return ExecutionProfile(
        name="verl-megatron-separate", execution_mode="partitioned-serial", outer_protocol="none",
        algorithm_contract="on-policy", max_policy_age=0,
        groups_per_batch=int(plan["groups_per_round"]), samples_per_group=int(plan["samples_per_group"]),
        algorithm_spec_sha256=spec.sha256(), extra={"engine": "verl-megatron"})


def hf_exports_for(plan: dict, island: MegatronIsland) -> MegatronHfExports | None:
    """The island's HF exports (``--rl-export-hf-rounds``), None when off."""
    config = plan.get("hf_exports")
    if not config:
        return None
    if int(plan["megatron"]["trainer_nodes"]) != 1:
        raise ValueError("verl Megatron HF export: the trainer pool must be one node (one staging directory)")
    return MegatronHfExports(island, config)


def run_megatron_island(trainer, agent_loop_manager, plan: dict, cfg: dict) -> dict:
    """The yeto driver over the Megatron ports (``verl_main.YetoTaskRunner``)."""
    from verl.utils.skip import SkipManager
    from verl.utils.tracking import DapoFilteredRewardTableLogger, Tracking, ValidationGenerationsLogger

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.hw.catalog import runtime_versions

    from .entry import verl_capabilities
    from .identity import backend_identity
    from .trainer import _fingerprint, check_asserted

    t0 = time.time()
    if plan.get("sync") != "none":
        raise ValueError("verl Megatron island: outer sync is not wired (use --rl-single-island-no-sync)")
    if plan.get("resume"):
        raise ValueError("verl Megatron island: round cuts are not wired")
    trainer.agent_loop_manager = agent_loop_manager
    SkipManager.init(trainer.config)
    trainer.logger = Tracking(project_name=trainer.config.trainer.project_name,
                              experiment_name=trainer.config.trainer.experiment_name,
                              default_backend=trainer.config.trainer.logger, config=cfg)
    names = dict(project_name=trainer.config.trainer.project_name,
                 experiment_name=trainer.config.trainer.experiment_name)
    trainer.validation_generations_logger = ValidationGenerationsLogger(**names)
    trainer.dapo_filtered_reward_logger = DapoFilteredRewardTableLogger(**names)
    trainer.global_steps = 1
    SkipManager.set_step(1)
    trainer.prev_step_profile = trainer.curr_step_profile = trainer.next_step_profile = False
    trainer.timing_raw = {}
    trainer._reissue_inflight_prompts()
    trainer.on_train_begin()

    learner_id = int(plan["learner_id"])
    asserted = check_asserted(plan.get("asserted") or {}, cfg)
    shape = dict(plan["megatron"])
    tape = EventTape(plan["event_tape"], learner_id)
    specs = layout_of(call_workers(trainer, remote_megatron_layout))
    layout_hash = layout_hash_of(specs)
    identity_hash = config_hash(plan["model_path"], shape)
    island = MegatronIsland(
        trainer, model_path=plan["model_path"], learner_id=learner_id, out_dir=plan["out_dir"],
        base_model_revision=plan["model_revision"], lora_config_hash=identity_hash,
        layout_hash=layout_hash, expected_specs=(), tis_upper=float(plan.get("tis_upper", 2.0)),
        thresholds_key=plan["thresholds_key"], emit=lambda event, **f: tape.append({"event": event, **f}))
    identity = backend_identity(train_backend="megatron", compat_group=plan.get("compat_group"))
    tape.append({"event": "rl_island_hardware", "compat_group": identity.compat_group, **runtime_versions()})
    tape.append({"event": "rl_verl_island_start", "plan": plan, "engine": "megatron",
                 "backend_identity": identity.to_dict(), "backend_identity_sha256": identity.sha256(),
                 "layout_hash": layout_hash, "config_hash": identity_hash, "n_specs": len(specs),
                 "asserted": asserted, "init_seconds": time.time() - t0})
    spec = AlgorithmSpec.from_dict(plan["algorithm_spec"]) if plan.get("algorithm_spec") else AlgorithmSpec()
    fingerprint = _fingerprint(cfg)
    driver = IslandDriver(
        learner_id=learner_id, rollout=VerlRolloutPool(island), trainer=MegatronTrainerGroup(island),
        policy_state=MegatronPolicyState(island), publisher=MegatronPublisher(island),
        placement=MegatronPlacement(shape), capabilities=verl_capabilities(fingerprint, megatron=True),
        algorithm=spec, sync=LocalOnlySync(int(plan["global_rounds"])), events=tape,
        profile=_execution_profile(spec, plan), layout=LAYOUT_FRAGMENTS)
    driver.data_slot = plan.get("data_slot")
    driver.hf_exports = hf_exports_for(plan, island)
    state = driver.run()
    result = {"learner_id": learner_id, "rounds_completed": driver.rounds_completed,
              "final_policy_version": state.policy_version,
              "final_policy_tensor_hash": state.policy_tensor_hash(), "seconds": time.time() - t0}
    tape.append({"event": "rl_verl_island_done", **result})
    tape.append({"event": "rl_learner_finalized"})
    return result
