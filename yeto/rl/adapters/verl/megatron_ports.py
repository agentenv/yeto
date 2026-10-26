"""Engine ports of the verl Megatron island (S27 G-DC; runs inside verl's task runner).

One island, full parameters, no outer sync (``--rl-single-island-no-sync``):
the trainer cards run verl's Megatron engine, the rollout cards run
standalone vLLM servers (:mod:`.megatron_engine`, :mod:`.separate_trainer`).
Next to the FSDP2 ports of :mod:`.ports_impl`:

* rollout: ``ports_impl.VerlRolloutPool`` unchanged (verl submits the round's
  prompts, the standalone servers generate, the replay buffer returns them);
* trainer: ``ports_impl.VerlTrainerGroup`` plus a count of trained rounds
  (the weights version the resident policy handle is checked against);
* policy state without an outer sync (:class:`MegatronPolicyState`): the
  weights stay in the trainer. Each Megatron rank hashes the parameters it
  owns (dense ones on data-parallel rank 0, expert ones on
  expert-data-parallel rank 0, keyed by name and TP/PP/EP coordinates); the
  island combines the per-rank hashes into the policy hash. No tensor leaves
  the trainer: there is no export or apply;
* publisher (:class:`MegatronPublisher`): verl's standalone NCCL checkpoint
  engine. There is no read-back of what vLLM loaded (the LoRA read-back of
  ``vllm_readback`` does not cover full weights); the train/rollout log-prob
  criteria of every round are the check that the rollout runs the published
  weights;
* placement: fixed partition (trainer pool, rollout pool).

``run_megatron_island`` wires them into the neutral ``IslandDriver``; the
final model can be exported to the run's store (``--rl-export-hf-rounds``).

S27 G-FG: with an outer sync (``plan["full_parameter"]``) the island runs the
FSDP2 island's full-parameter sync instead of :class:`MegatronPolicyState`:
the engine-neutral role stream over the Megatron trainer group of
:mod:`.megatron_full_parameter` (FP32 masters in HF names, the FSDP2 layout).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState, WeightsChanged

from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS

from .ports_impl import MEMBER, VerlIsland, VerlRolloutPool, VerlTrainerGroup, _inner_worker  # noqa: F401
from .separate import (READBACK_STATUS, SeparatePlacement, SeparatePublisher, execution_profile,  # noqa: F401
                       wire_rollout_ahead, with_policy_age)

DIGEST_PREFIX = b"yeto-verl-megatron-policy-v1\0"
PAYLOAD_PREFIX = b"yeto-verl-megatron-payload-v1\0"
HASH_THREADS = 8


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


def remote_megatron_grad_checks(worker_dict) -> dict:
    """S27 G-FL: the gradient checks this rank's DDP wrappers actually carry.

    Read off ``DistributedDataParallelConfig`` after verl built the model, so an
    override Hydra dropped cannot pass unnoticed
    (:mod:`yeto.rl.megatron_grad_checks`)."""
    import torch.distributed as dist

    from yeto.rl import megatron_grad_checks

    record = megatron_grad_checks.report(_chunks(_inner_worker(worker_dict)))
    record["rank"] = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0
    return record


def remote_ascend_plugin_report(worker_dict) -> dict:
    """S27 G-FL: what the Ascend plugin stack installed in this worker process.

    ``verl_main.setup_worker`` installed it before verl built the model; this only
    reads the record back for the island log. ``{"installed": False}`` when the worker
    never ran the install (the switch is off, or the setup hook did not run)."""
    import os

    from yeto.rl import ascend_plugin_stack

    report = ascend_plugin_stack.LAST_REPORT
    rank = int(os.environ.get("RANK", 0))
    if report is None:
        return {"rank": rank, "ran": False, "root": os.environ.get("YETO_ASCEND_PLUGIN_STACK", "")}
    return {"rank": rank, "ran": True, "root": report["root"], "installed": report["installed"],
            "failed": report["failed"], "switches": sorted(report["env"])}


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
    tokenizer and generation files, so the directory loads as the base repo does.
    After every rank finished, rank 0 leaves the writer's completion marker that
    ``HfRoundExports`` takes out (as ``adapters/verl/hf_export.py`` on FSDP2)."""
    import shutil

    import torch

    from yeto.rl.engine.hf_export import COMPLETE_MARKER

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
    if rank == 0:
        (directory / COMPLETE_MARKER).write_text("")


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
        state = TrainerResidentState(
            layout=LAYOUT_FRAGMENTS, base_model_revision=isl.base_model_revision,
            config_hash=isl.lora_config_hash, layout_hash=isl.layout_hash,
            policy_version=int(policy_version), digest=digest, materialize_fn=self._refuse,
            weights_mark=self._mark(), holds_fn=self._holds)
        self._last_verified = state
        return state

    def allow_unverified_rounds(self) -> None:
        """S27 G-EU (VERL-PARITY #26, ``--rl-verify-every N``): the rounds the driver
        does not verify publish :meth:`export_identity` (no per-rank SHA-256)."""
        self.supports_unverified_rounds = True

    def export_identity(self, *, policy_version: int) -> TrainerResidentState | None:
        """A round whose publication is not verified: the trainer weights version
        stands in for the content hash (``policy_digest.unverified_identity``);
        None before the first verified digest (the driver then verifies)."""
        from yeto.rl.engine.policy_digest import PolicyDigest, unverified_identity

        last = getattr(self, "_last_verified", None)
        if last is None:
            return None
        isl, mark = self.island, self._mark()
        identity = unverified_identity(int(policy_version), mark)
        print(f"[rl] verl megatron publication v{policy_version}: unverified (no digest; last verified "
              f"hash {last.policy_tensor_hash()[:12]})", flush=True)
        return TrainerResidentState(
            layout=LAYOUT_FRAGMENTS, base_model_revision=isl.base_model_revision,
            config_hash=isl.lora_config_hash, layout_hash=isl.layout_hash,
            policy_version=int(policy_version),
            digest=PolicyDigest(identity, identity, last.digest.payload_bytes, last.digest.specs),
            materialize_fn=self._refuse, weights_mark=mark, holds_fn=self._holds, verified=False)

    def export(self):
        self._refuse(-1)

    def apply(self, state, *, optimizer: str, local_step: int) -> None:
        self._refuse(int(getattr(state, "policy_version", -1)))


# S27 G-EK: the publisher and the placement are the engine-neutral ones of the
# separate mode (.separate; the FSDP2 full-parameter island runs them too)
class MegatronPublisher(SeparatePublisher):
    pass


class MegatronPlacement(SeparatePlacement):
    pass


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
    return execution_profile(spec, plan, name="verl-megatron-separate", engine="verl-megatron",
                             outer_protocol="none" if plan.get("sync") == "none" else "strict-avg")


def hf_exports_for(plan: dict, island: MegatronIsland) -> MegatronHfExports | None:
    """The island's HF exports (``--rl-export-hf-rounds``), None when off."""
    config = plan.get("hf_exports")
    if not config:
        return None
    if int(plan["megatron"]["trainer_nodes"]) != 1:
        raise ValueError("verl Megatron HF export: the trainer pool must be one node (one staging directory)")
    return MegatronHfExports(island, config)


def sync_ports(island: MegatronIsland, plan: dict, identity_sha256: str):
    """``(sync, policy state)`` of the island: without an outer sync the local-only
    sync over :class:`MegatronPolicyState`; with one (S27 G-FG) the FSDP2 island's
    full-parameter sync over the Megatron trainer group
    (:func:`.megatron_full_parameter.megatron_group`)."""
    from yeto.rl.engine.bridges import LocalOnlySync

    if plan.get("sync") == "none":
        return LocalOnlySync(int(plan["global_rounds"])), MegatronPolicyState(island)
    from .full_parameter_stream import full_parameter_policy
    from .megatron_full_parameter import megatron_group

    group = getattr(island, "full_group", None) or megatron_group(island, plan["model_path"])
    return full_parameter_policy(island, plan, backend_identity_sha256=identity_sha256, group=group)


def driver_options(plan: dict, spec, fingerprint: str) -> dict:
    """``IslandDriver`` keywords of the Megatron island: its placement, capabilities,
    partitioned-serial profile (outer protocol ``strict-avg`` when synced) and layout."""
    from .entry import verl_capabilities

    return {"placement": MegatronPlacement(dict(plan["megatron"])),
            "capabilities": with_policy_age(verl_capabilities(fingerprint, megatron=True), plan),
            "profile": _execution_profile(spec, plan), "layout": LAYOUT_FRAGMENTS}


def _check_grad_checks(trainer, shape: dict, tape) -> None:
    """S27 G-FL: every trainer rank must carry the gradient check the plan asked for.

    The check turns a non-finite gradient into a RuntimeError instead of letting
    ``clip_grads`` scale it into every parameter; a run that wants the old behaviour
    says so with ``--rl-no-grad-nan-check``."""
    from yeto.rl import megatron_grad_checks

    wanted = bool(shape.get("grad_nan_check", True))
    reports = call_workers(trainer, remote_megatron_grad_checks)
    tape.append({"event": "rl_megatron_grad_checks", "wanted": wanted, "ranks": reports})
    print(f"[yeto-verl] gradient NaN check wanted={wanted} ranks="
          f"{[ (r.get('rank'), r.get('check_for_nan_in_grad')) for r in reports ]}", flush=True)
    megatron_grad_checks.check_reports(reports, wanted=wanted)


def _log_ascend_plugin_stack(trainer, section: dict, tape) -> None:
    """S27 G-FL: print what every trainer worker installed, and put it on the tape.

    Reading the report back cannot fail the island: a worker that never ran the
    install reports ``ran: False`` and the line says so."""
    try:
        reports = call_workers(trainer, remote_ascend_plugin_report)
    except Exception as error:  # noqa: BLE001 - a log must not stop the island
        reports = [{"rank": -1, "ran": False, "error": repr(error)}]
    print(f"[yeto-verl] ascend plugin stack root={section.get('root')} "
          f"switches={len(section.get('env') or {})} planned={len(section.get('plugins') or [])}",
          flush=True)
    for name, reason in sorted((section.get("excluded") or {}).items()):
        print(f"[yeto-verl]   excluded {name}: {reason}", flush=True)
    for name, reason in sorted((section.get("unset") or {}).items()):
        print(f"[yeto-verl]   unset {name}: {reason}", flush=True)
    for report in sorted(reports, key=lambda row: int(row.get("rank", 0))):
        if not report.get("ran"):
            print(f"[yeto-verl]   rank {report.get('rank')}: the stack did not run "
                  f"({report.get('error') or 'no report in the worker'})", flush=True)
            continue
        print(f"[yeto-verl]   rank {report['rank']}: installed {len(report['installed'])} "
              f"plugin(s); not installed {sorted(report['failed']) or 'none'}", flush=True)
        for plugin, reason in sorted(report["failed"].items()):
            print(f"[yeto-verl]     NOT installed {plugin}: {reason}", flush=True)
    tape.append({"event": "rl_ascend_plugin_stack", "root": section.get("root"),
                 "env": section.get("env"), "unset": sorted(section.get("unset") or {}),
                 "excluded": sorted(section.get("excluded") or {}), "ranks": reports})


def run_megatron_island(trainer, agent_loop_manager, plan: dict, cfg: dict) -> dict:
    """The yeto driver over the Megatron ports (``verl_main.YetoTaskRunner``)."""
    from verl.utils.skip import SkipManager
    from verl.utils.tracking import DapoFilteredRewardTableLogger, Tracking, ValidationGenerationsLogger

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.hw.catalog import runtime_versions

    from .identity import backend_identity
    from .trainer import _fingerprint, check_asserted

    t0 = time.time()
    synced = plan.get("sync") != "none"  # S27 G-FG: the full-parameter outer sync (.megatron_full_parameter)
    if synced and not plan.get("full_parameter"):
        raise ValueError("a synced verl Megatron island needs the full-parameter sync plan")
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
    identity = backend_identity(train_backend="megatron", compat_group=plan.get("compat_group"))
    if synced:  # S27 G-FG: the FSDP2 island's HF-named layout, identity and policy-age binding
        from yeto.rl.engine.policy_age import bind_policy_age

        from .full_parameter_stream import full_parameter_identity

        specs: tuple = ()
        layout_hash = plan["full_parameter"]["parameter_layout_sha256"]
        _layout, identity_hash = full_parameter_identity(plan["model_config_sha256"])
        identity_sha = bind_policy_age(identity.sha256(), int(plan.get("max_policy_age") or 0))
    else:
        specs = layout_of(call_workers(trainer, remote_megatron_layout))
        layout_hash = layout_hash_of(specs)
        identity_hash = config_hash(plan["model_path"], shape)
        identity_sha = identity.sha256()
    island = MegatronIsland(
        trainer, model_path=plan["model_path"], learner_id=learner_id, out_dir=plan["out_dir"],
        base_model_revision=plan["model_revision"], lora_config_hash=identity_hash,
        layout_hash=layout_hash, expected_specs=(), tis_upper=float(plan.get("tis_upper", 2.0)),
        thresholds_key=plan["thresholds_key"], emit=lambda event, **f: tape.append({"event": event, **f}))
    sync, policy_state = sync_ports(island, plan, identity_sha)
    tape.append({"event": "rl_island_hardware", "compat_group": identity.compat_group, **runtime_versions()})
    _check_grad_checks(trainer, shape, tape)  # S27 G-FL: refuses a trainer without the NaN check
    if plan.get("ascend_plugin_stack"):  # S27 G-FL: one line per switch and per plugin
        _log_ascend_plugin_stack(trainer, plan["ascend_plugin_stack"], tape)
    tape.append({"event": "rl_verl_island_start", "plan": plan, "engine": "megatron",
                 "backend_identity": identity.to_dict(), "backend_identity_sha256": identity_sha,
                 "layout_hash": layout_hash, "config_hash": identity_hash, "n_specs": len(specs),
                 "parameter_mode": "full", "asserted": asserted, "init_seconds": time.time() - t0})
    spec = AlgorithmSpec.from_dict(plan["algorithm_spec"]) if plan.get("algorithm_spec") else AlgorithmSpec()
    fingerprint = _fingerprint(cfg)
    from .driver_extras import TestExitPublisher

    # TEST ONLY (leave/rejoin, as the FSDP2 island): end the first process after a given publication
    publisher = TestExitPublisher(MegatronPublisher(island), plan.get("test_exit_after_version"), tape,
                                  rejoined=lambda: bool(getattr(sync, "joined_running_pool", False)))
    dispense = None
    if plan.get("dispense"):  # S27 G-FG: a synced island claims its rows from the task dispenser (as FSDP2)
        from .task_dispense import install as install_dispense

        dispense = install_dispense(trainer, plan["dispense"],
                                    emit=lambda event, **f: tape.append({"event": event, **f}))
    driver = IslandDriver(
        learner_id=learner_id, rollout=VerlRolloutPool(island, dispense=dispense),
        trainer=MegatronTrainerGroup(island), policy_state=policy_state, publisher=publisher,
        algorithm=spec, sync=sync, events=tape, **driver_options(plan, spec, fingerprint))
    driver.data_slot = plan.get("data_slot")
    if dispense is not None and plan["dispense"].get("data_streams"):
        driver.data_streams = plan["dispense"]["data_streams"]  # S27 P3 streams (rl_driver_start)
    if synced:  # S27 G-DS: data positions and observation, as on a synced FSDP2 island
        from .driver_extras import wire_full_parameter_extras

        wire_full_parameter_extras(driver, island, {k: v for k, v in plan.items() if k != "hf_exports"})
    wire_rollout_ahead(driver, plan)  # S27 G-EV: --rl-rollout-ahead 1 (0: serial rounds)
    driver.hf_exports = hf_exports_for(plan, island)
    from .train_options import wire_debug, wire_verify  # S27 G-EU (VERL-PARITY #26, #35)

    wire_verify(driver, driver.policy_state, plan)
    wire_debug(island, plan)
    state = driver.run()
    result = {"learner_id": learner_id, "rounds_completed": driver.rounds_completed,
              "final_policy_version": state.policy_version,
              "final_policy_tensor_hash": state.policy_tensor_hash(), "seconds": time.time() - t0}
    tape.append({"event": "rl_verl_island_done", **result})
    tape.append({"event": "rl_learner_finalized"})
    return result
