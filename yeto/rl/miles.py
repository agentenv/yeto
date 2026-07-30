"""Adapter for the pinned Miles commit's Megatron/SGLang runtime."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import statistics
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from . import MILES_COMMIT, MILES_REPOSITORY
from .core import (
    CanonicalLoraState,
    LocalRoundStats,
    StrictRlInvariantError,
    canonical_state,
    policy_hash,
)

_CANONICAL_PREFIX = "base_model.model."


def verify_miles_revision(root: str | Path) -> Path:
    """Verify repository, commit, tracked files, and the imported package."""

    root = Path(root).expanduser().resolve()

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot verify Miles checkout at {root}") from exc

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    origin = git("config", "--get", "remote.origin.url")
    if commit != MILES_COMMIT:
        raise RuntimeError(
            f"Miles revision mismatch: expected {MILES_COMMIT}, got {commit}"
        )
    if origin.removesuffix(".git").rstrip("/") != MILES_REPOSITORY:
        raise RuntimeError(
            f"Miles origin mismatch: expected {MILES_REPOSITORY}, got {origin}"
        )
    if branch != "HEAD":
        raise RuntimeError("Miles checkout is not detached")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Miles checkout is not clean")

    miles = importlib.import_module("miles")
    package_path = Path(miles.__file__).resolve()
    if not package_path.is_relative_to(root):
        raise RuntimeError(f"imported Miles package {package_path} is outside {root}")
    return root


def _adapter_sides(actor) -> list[tuple[str, Any]]:
    """Return PEFT names paired with Bridge's actual conversion tasks."""

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(
        actor.args.hf_checkpoint,
        trust_remote_code=bool(actor.args.yeto_rl_trust_remote_code),
    )
    model_bridge = getattr(bridge, "_model_bridge", None)
    build_tasks = getattr(model_bridge, "build_adapter_conversion_tasks", None)
    if build_tasks is None:
        raise RuntimeError("pinned Megatron-Bridge lacks adapter conversion tasks")
    tasks_by_base = build_tasks(actor.model)
    sides: list[tuple[str, Any]] = []
    for base_name in sorted(tasks_by_base):
        tasks = sorted(
            tasks_by_base[base_name],
            key=lambda task: task.adapter_key or "",
        )
        for task in tasks:
            for side in (task.linear_in_task, task.linear_out_task):
                parameter = side.param_weight
                main = getattr(parameter, "main_param", None)
                if (
                    main is None
                    or main.dtype != torch.float32
                    or main.numel() != parameter.numel()
                ):
                    raise RuntimeError(
                        f"LoRA parameter {side.param_name!r} has no complete "
                        "FP32 optimizer master"
                    )
                converted = side.mapping.megatron_to_hf(
                    main.view(parameter.shape),
                    side.megatron_module,
                )
                if len(converted) != 1:
                    raise RuntimeError(
                        f"ambiguous LoRA mapping for {side.param_name!r}"
                    )
                raw_name = next(iter(converted))
                name = (
                    raw_name
                    if raw_name.startswith(_CANONICAL_PREFIX)
                    else _CANONICAL_PREFIX + raw_name
                )
                if not name.endswith((".lora_A.weight", ".lora_B.weight")):
                    raise RuntimeError(f"non-PEFT LoRA mapping {name!r}")
                sides.append((name, side))
    names = [name for name, _ in sides]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("Miles produced an empty or duplicate LoRA mapping")
    mapped = {id(side.param_weight) for _, side in sides}
    trainable = {
        id(parameter)
        for chunk in actor.model
        for parameter in chunk.parameters()
        if parameter.requires_grad
    }
    if mapped != trainable:
        raise RuntimeError(
            "Miles adapter conversion does not cover every trainable parameter"
        )
    return sorted(sides)


@torch.no_grad()
def _export_fp32_policy(actor) -> dict[str, torch.Tensor]:
    tensors = {}
    for name, side in _adapter_sides(actor):
        parameter = side.param_weight
        converted = side.mapping.megatron_to_hf(
            parameter.main_param.view(parameter.shape),
            side.megatron_module,
        )
        value = next(iter(converted.values()))
        tensors[name] = value.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous().clone()
    return tensors


def _optimizer_children(optimizer) -> list[Any]:
    return list(getattr(optimizer, "chained_optimizers", (optimizer,)))


def _reset_optimizer_state(actor, parameters: list[torch.Tensor]) -> int:
    parameter_ids = {id(parameter) for parameter in parameters}
    for child in _optimizer_children(actor.optimizer):
        optimizer = getattr(child, "optimizer", child)
        for parameter in list(optimizer.state):
            if id(parameter) in parameter_ids:
                optimizer.state.pop(parameter, None)
    return len(parameter_ids)


@torch.no_grad()
def _copy_masters_to_model(actor) -> None:
    for child in _optimizer_children(actor.optimizer):
        copy = getattr(child, "_copy_main_params_to_model_params", None)
        if copy is None:
            raise RuntimeError("pinned Megatron optimizer lacks main-to-model copy")
        copy()


def _restore_scheduler_progress(actor, policy_version: int) -> None:
    scheduler = actor.opt_param_scheduler
    batch_size = actor.args.global_batch_size
    target = policy_version * actor.args.num_steps_per_rollout * batch_size
    if scheduler is None or batch_size <= 0 or scheduler.num_steps % batch_size:
        raise RuntimeError("Miles scheduler progress is not an integral optimizer step")
    if scheduler.num_steps > target:
        raise RuntimeError("Miles scheduler is ahead of the committed policy")
    if scheduler.num_steps < target:
        scheduler.step(increment=target - scheduler.num_steps)


def _actor_export_policy(self):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    return _export_fp32_policy(self)


@torch.no_grad()
def _actor_apply_policy(
    self,
    tensors: Mapping[str, torch.Tensor],
    policy_version: int,
):
    sides = dict(_adapter_sides(self))
    if set(tensors) != set(sides):
        missing = sorted(set(sides) - set(tensors))
        extra = sorted(set(tensors) - set(sides))
        raise RuntimeError(
            f"global LoRA mapping mismatch: missing={missing}, extra={extra}"
        )

    mapped = {}
    for name, side in sides.items():
        value = tensors[name].detach().to(
            device=side.param_weight.device,
            dtype=torch.float32,
        )
        target = side.mapping.hf_to_megatron(value, side.megatron_module)
        if target.numel() != side.param_weight.numel():
            raise RuntimeError(f"global LoRA shape mismatch for {name!r}")
        mapped[name] = target.reshape(side.param_weight.shape).contiguous()

    _restore_scheduler_progress(self, policy_version)
    reset_parameter_count = _reset_optimizer_state(
        self,
        [side.param_weight.main_param for side in sides.values()],
    )
    for name, side in sides.items():
        side.param_weight.main_param.view(side.param_weight.shape).copy_(mapped[name])
    _copy_masters_to_model(self)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    self.weights_backuper.backup("actor")
    torch.cuda.empty_cache()

    identity = {
        "base_model_revision": self.args.yeto_rl_base_model_revision,
        "lora_config_hash": self.args.yeto_rl_lora_config_hash,
        "layout_hash": self.args.yeto_rl_layout_hash,
    }
    applied = canonical_state(policy_version, _export_fp32_policy(self), **identity)
    canonical_state(
        policy_version,
        tensors,
        expected_specs=applied.specs,
        **identity,
    )
    return reset_parameter_count, policy_hash(applied)


def _actor_optimizer_steps(self) -> int:
    scheduler = self.opt_param_scheduler
    batch = self.args.global_batch_size
    if scheduler is None or batch <= 0 or scheduler.num_steps % batch:
        raise RuntimeError("Miles optimizer step counter is not integral")
    return int(scheduler.num_steps // batch)


def _actor_train_metrics(self):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    metrics = getattr(self.args, "_yeto_rl_train_metrics", None)
    if hasattr(self.args, "_yeto_rl_train_metrics"):
        del self.args._yeto_rl_train_metrics
    return metrics


def _install_train_metric_capture() -> None:
    from miles.backends.megatron_utils import model

    original = model.log_train_step
    if getattr(original, "_yeto_rl_capture", False):
        return

    def log_train_step(*values, **kwargs):
        metrics = original(*values, **kwargs)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            args = kwargs["args"]
            args._yeto_rl_train_metrics = {
                key: float(metrics[key])
                for key in (
                    "train/train_rollout_kl",
                    "train/ess_ratio",
                    "train/pg_clipfrac",
                )
                if key in metrics
            }
        return metrics

    log_train_step._yeto_rl_capture = True
    model.log_train_step = log_train_step


def _install_colocated_lora_ipc_sync() -> None:
    """Keep CUDA IPC producers alive until SGLang finishes the transfer."""

    import ray
    from miles.backends.megatron_utils.update_weight import (
        update_weight_from_tensor,
    )
    from miles.backends.megatron_utils.update_weight.common import (
        _check_weight_sync_results,
    )

    original = update_weight_from_tensor._send_to_colocated_engine
    if getattr(original, "_yeto_rl_synchronized", False):
        return

    def synchronized_send(*values, **kwargs):
        refs, long_lived_tensors = original(*values, **kwargs)
        if kwargs.get("lora_config") is not None:
            results = ray.get(refs)
            _check_weight_sync_results(results, is_lora=True)
            group = kwargs.get("ipc_gather_group")
            if group is not None:
                torch.distributed.barrier(group=group)
        return refs, long_lived_tensors

    synchronized_send._yeto_rl_synchronized = True
    update_weight_from_tensor._send_to_colocated_engine = synchronized_send


def configure_miles_bridge(args) -> None:
    """Install Yeto actor methods inside each Miles Ray worker."""

    from megatron.bridge import AutoBridge
    from megatron.bridge.training import config as bridge_config

    original = AutoBridge.to_megatron_provider

    def configured_provider(self, *values, **kwargs):
        provider = original(self, *values, **kwargs)
        provider.attention_backend = args.attention_backend
        return provider

    AutoBridge.to_megatron_provider = configured_provider
    # The INIT adapter is replicated. Keep complete fp32 masters and Adam
    # state on every DP/EP rank; no sharded gather path is part of v0.
    args.use_distributed_optimizer = False
    original_ddp_config = bridge_config.DistributedDataParallelConfig

    def replicated_lora_ddp_config(*values, **kwargs):
        kwargs["use_distributed_optimizer"] = False
        return original_ddp_config(*values, **kwargs)

    bridge_config.DistributedDataParallelConfig = replicated_lora_ddp_config
    _install_train_metric_capture()
    _install_colocated_lora_ipc_sync()
    install_miles_actor_adapter()


def install_miles_actor_adapter() -> None:
    """Install methods in both the driver and each Miles Ray worker."""

    from miles.backends.megatron_utils.actor import MegatronTrainRayActor

    methods = {
        "yeto_rl_export_policy": _actor_export_policy,
        "yeto_rl_apply_policy": _actor_apply_policy,
        "yeto_rl_optimizer_steps": _actor_optimizer_steps,
        "yeto_rl_train_metrics": _actor_train_metrics,
    }
    for name, method in methods.items():
        existing = getattr(MegatronTrainRayActor, name, None)
        if existing is not None and (
            getattr(existing, "__module__", None),
            getattr(existing, "__qualname__", None),
        ) != (method.__module__, method.__qualname__):
            raise RuntimeError(f"Miles actor already defines incompatible {name}")
        setattr(MegatronTrainRayActor, name, method)


def _policy_token(version: int) -> str:
    return f"yeto:{version}"


def _version_from_token(token: object) -> int:
    prefix, separator, value = str(token).partition(":")
    if prefix != "yeto" or not separator:
        raise RuntimeError(f"invalid rollout policy token {token!r}")
    try:
        version = int(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid rollout policy token {token!r}") from exc
    if version < 0:
        raise RuntimeError(f"invalid rollout policy token {token!r}")
    return version


def _validate_rollout_groups(data: object, groups: int, samples: int) -> None:
    if not isinstance(data, list) or len(data) != groups:
        raise RuntimeError(
            f"Miles produced {len(data) if isinstance(data, list) else 0} "
            f"groups, expected {groups}"
        )
    for index, group in enumerate(data):
        if not isinstance(group, list) or len(group) != samples:
            raise RuntimeError(
                f"Miles group {index} contains "
                f"{len(group) if isinstance(group, list) else 0} trajectories, "
                f"expected {samples}"
            )
        for sample in group:
            status = getattr(getattr(sample, "status", None), "value", None)
            if isinstance(sample, list) or status not in {"completed", "truncated"}:
                raise RuntimeError("Miles returned an incomplete trajectory")


_ISLAND_CHECKPOINT_SCHEMA = 2


def _island_checkpoint_config(args) -> dict[str, Any]:
    return {
        "actor_num_gpus_per_node": args.actor_num_gpus_per_node,
        "actor_num_nodes": args.actor_num_nodes,
        "advantage_estimator": args.advantage_estimator,
        "model": args.yeto_rl_model,
        "dataset": args.yeto_rl_data,
        "base_model_revision": args.yeto_rl_base_model_revision,
        "data_revision": args.yeto_rl_data_revision,
        "seq_length": args.seq_length,
        "seed": args.seed,
        "expert_model_parallel_size": args.expert_model_parallel_size,
        "layout_hash": args.yeto_rl_layout_hash,
        "lr": args.lr,
        "lora_config_hash": args.yeto_rl_lora_config_hash,
        "n_samples_per_prompt": args.n_samples_per_prompt,
        "num_steps_per_rollout": args.num_steps_per_rollout,
        "over_sampling_batch_size": args.over_sampling_batch_size,
        "reward_sha256": args.yeto_rl_reward_sha256,
        "rollout_batch_size": args.rollout_batch_size,
        "rollout_max_response_len": args.rollout_max_response_len,
        "custom_generate_function_path": args.custom_generate_function_path,
        "use_session_server": args.use_session_server,
        "tito_model": args.tito_model,
    }


def _complete_group_for_policy(group: object, policy_version: int, size: int) -> bool:
    if not isinstance(group, list) or len(group) != size:
        return False
    for sample in group:
        status = getattr(getattr(sample, "status", None), "value", None)
        versions = getattr(sample, "weight_versions", None)
        if status not in {"completed", "truncated"} or not versions:
            return False
        try:
            observed = {_version_from_token(token) for token in versions}
        except RuntimeError:
            return False
        if observed != {policy_version}:
            return False
    return True


def _group_indices(group: object) -> tuple[int, ...] | None:
    if not isinstance(group, list):
        return None
    try:
        return tuple(int(sample.index) for sample in group)
    except (AttributeError, TypeError, ValueError):
        return None


def queue_completed_groups(args, all_samples, data_source) -> None:
    """Retain complete oversampling results until this round selects its batch."""

    source = getattr(data_source, "__self__", None)
    if source is None or not callable(getattr(source, "add_samples", None)):
        raise RuntimeError("pinned Miles did not provide a bound data source method")
    source.add_samples(
        [
            group
            for group in all_samples
            if _complete_group_for_policy(
                group,
                args.yeto_rl_policy_version,
                args.n_samples_per_prompt,
            )
        ]
    )


def _atomic_save_island_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _serialize_completed_groups(groups: list[list[Any]]) -> list[list[dict[str, Any]]]:
    serialized = []
    for group in groups:
        serialized_group = []
        for sample in group:
            to_dict = getattr(sample, "to_dict", None)
            if not callable(to_dict):
                raise RuntimeError("pinned Miles Sample lacks checkpoint serialization")
            serialized_group.append(to_dict())
        serialized.append(serialized_group)
    return serialized


def _deserialize_completed_groups(groups: object) -> list[list[Any]]:
    from miles.utils.types import Sample

    if not isinstance(groups, list):
        return []
    try:
        return [
            [Sample.from_dict(sample) for sample in group]
            for group in groups
            if isinstance(group, list)
        ]
    except (TypeError, ValueError) as error:
        raise RuntimeError("invalid Miles samples in island checkpoint") from error


def _restore_completed_groups(args, policy_version: int, data_source) -> None:
    if getattr(data_source, "_yeto_rl_checkpoint_loaded", False):
        return
    data_source._yeto_rl_checkpoint_loaded = True
    path = Path(args.yeto_rl_completed_groups_path).expanduser()
    if not path.is_file():
        return
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # local queue is disposable, global state is not
        print(f"[rl] discarded unreadable island checkpoint {path}: {error}")
        return
    expected_config = _island_checkpoint_config(args)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _ISLAND_CHECKPOINT_SCHEMA
        or payload.get("policy_version") != policy_version
        or payload.get("config") != expected_config
    ):
        print(f"[rl] discarded island checkpoint outside policy/config contract: {path}")
        return
    groups = [
        group
        for group in _deserialize_completed_groups(payload.get("completed_groups"))
        if _complete_group_for_policy(
            group, policy_version, args.n_samples_per_prompt
        )
    ]
    if groups:
        data_source.add_samples(groups)


def _save_completed_groups(
    args,
    policy_version: int,
    local_round_id: int,
    data_source,
    metrics: Mapping[str, Any] | None,
) -> None:
    buffer = getattr(data_source, "buffer", None)
    if not isinstance(buffer, list):
        raise RuntimeError("Miles data source lacks a completed-group queue")
    completed = [
        group
        for group in buffer
        if _complete_group_for_policy(
            group, policy_version, args.n_samples_per_prompt
        )
    ]
    buffer[:] = completed
    numeric_metrics = {
        str(name): float(value)
        for name, value in (metrics or {}).items()
        if isinstance(value, (int, float))
    }
    _atomic_save_island_checkpoint(
        Path(args.yeto_rl_completed_groups_path).expanduser(),
        {
            "schema_version": _ISLAND_CHECKPOINT_SCHEMA,
            "config": _island_checkpoint_config(args),
            "local_round_id": local_round_id,
            "policy_version": policy_version,
            "rollout_metrics": numeric_metrics,
            "local_round_stats": None,
            "completed_groups": _serialize_completed_groups(completed),
        },
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _run_rollout_with_metrics(generate, args, rollout_id, data_source, evaluation):
    from miles.rollout import sglang_rollout

    state = {
        "active": 0,
        "peak_active": 0,
        "cancelled": 0,
        "durations": [],
    }
    generate_state = getattr(sglang_rollout, "GenerateState", None)
    original = getattr(generate_state, "submit_generate_tasks", None)
    if original is None:
        raise RuntimeError("pinned Miles lacks GenerateState.submit_generate_tasks")

    def submit_generate_tasks(self, samples):
        before = set(self.pendings)
        original(self, samples)
        started = time.monotonic()
        tasks = self.pendings - before
        state["active"] += len(tasks)
        state["peak_active"] = max(state["peak_active"], state["active"])

        def finished(task):
            state["active"] -= 1
            state["durations"].append(time.monotonic() - started)
            if task.cancelled():
                state["cancelled"] += 1
                return
            try:
                group = task.result()
            except Exception:
                state["cancelled"] += 1
                return
            statuses = [
                getattr(getattr(sample, "status", None), "value", None)
                for sample in group
            ]
            if any(status not in {"completed", "truncated"} for status in statuses):
                state["cancelled"] += 1

        for task in tasks:
            task.add_done_callback(finished)

    generate_state.submit_generate_tasks = submit_generate_tasks
    try:
        return generate(args, rollout_id, data_source, evaluation=evaluation), state
    finally:
        generate_state.submit_generate_tasks = original


def generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    """Keep Miles rollout, adding group/version queue contracts."""

    from miles.rollout.sglang_rollout import generate_rollout as miles_generate

    if not evaluation:
        args.yeto_rl_policy_version = rollout_id
        _restore_completed_groups(args, rollout_id, data_source)
        buffer = getattr(data_source, "buffer", None)
        if not isinstance(buffer, list):
            raise RuntimeError("Miles data source lacks a completed-group queue")
        buffer[:] = [
            group
            for group in buffer
            if _complete_group_for_policy(
                group, rollout_id, args.n_samples_per_prompt
            )
        ]
    output, lifecycle = _run_rollout_with_metrics(
        miles_generate,
        args,
        rollout_id,
        data_source,
        evaluation,
    )
    if not evaluation:
        if lifecycle["active"] != 0:
            raise RuntimeError("Miles returned with rollout groups still active")
        _validate_rollout_groups(
            output.samples,
            args.rollout_batch_size,
            args.n_samples_per_prompt,
        )
        consumed = {_group_indices(group) for group in output.samples}
        data_source.buffer[:] = [
            group
            for group in data_source.buffer
            if _group_indices(group) not in consumed
        ]
        samples = [sample for group in output.samples for sample in group]
        tool_wait_seconds = sum(
            float(getattr(sample, "non_generation_time", 0.0))
            for sample in samples
        )
        round_metrics = {
            "active_groups": lifecycle["peak_active"],
            "cancelled_groups": lifecycle["cancelled"],
            "tool_wait_seconds": tool_wait_seconds,
            "group_p50_seconds": _percentile(lifecycle["durations"], 0.50),
            "group_p95_seconds": _percentile(lifecycle["durations"], 0.95),
            "group_p99_seconds": _percentile(lifecycle["durations"], 0.99),
        }
        _save_completed_groups(
            args,
            rollout_id,
            rollout_id + 1,
            data_source,
            {**(output.metrics or {}), **round_metrics},
        )
    return output


class MilesIslandRuntime:
    """Synchronous wrapper over the pinned Miles Ray APIs."""

    def __init__(self, args) -> None:
        self.args = args
        self.loop = asyncio.new_event_loop()
        self.rollout_manager = None
        self.actor_model = None
        self._owns_ray = False
        self._trainer_awake = False
        self._rollout_offloaded = True
        self._policy_version: int | None = None
        self._rollout_id = 0
        self._optimizer_reset_count = 0

    async def _onload_trainer(self) -> None:
        if self._trainer_awake:
            return
        await self.actor_model.onload()
        self._trainer_awake = True

    async def _offload_trainer(self) -> None:
        if self._trainer_awake and self.args.offload_train:
            await self.actor_model.offload()
            self._trainer_awake = False

    def _run(self, coroutine):
        return self.loop.run_until_complete(coroutine)

    async def _actor_call(self, method: str, *args, rank0: bool = False):
        results = await self.actor_model._broadcast(method, *args)
        expected = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        if not isinstance(results, list) or len(results) != expected:
            raise RuntimeError(
                f"Miles returned {len(results) if isinstance(results, list) else 0} "
                f"actor results, expected {expected}"
            )
        if rank0:
            exported = [result for result in results if result is not None]
            if len(exported) != 1:
                raise RuntimeError("Miles must export policy only on global rank 0")
            return exported[0]
        if not results or any(result != results[0] for result in results[1:]):
            raise RuntimeError(f"Miles actor ranks disagree on {method}")
        return results[0]

    async def _initialize(self) -> CanonicalLoraState:
        import ray
        from miles.ray.placement_group import (
            create_placement_groups,
            create_rollout_manager,
            create_training_models,
        )

        install_miles_actor_adapter()
        if not ray.is_initialized():
            ray.init(address="auto")
            self._owns_ray = True
        expected_gpus = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        )
        deadline = time.monotonic() + 300
        while True:
            visible_gpus = int(ray.cluster_resources().get("GPU", 0))
            if visible_gpus >= expected_gpus or time.monotonic() >= deadline:
                break
            await asyncio.sleep(2)
        if visible_gpus != expected_gpus:
            raise RuntimeError(
                f"Miles Ray cluster has {visible_gpus} GPUs, expected {expected_gpus}"
            )
        groups = create_placement_groups(self.args)
        self.rollout_manager, _ = create_rollout_manager(
            self.args, groups["rollout"]
        )
        self.actor_model, critic = await create_training_models(
            self.args, groups, self.rollout_manager
        )
        if critic is not None:
            raise RuntimeError("RL v0 does not support a Miles critic")
        self._trainer_awake = not self.args.offload_train
        await self._onload_trainer()
        tensors = await self._actor_call("yeto_rl_export_policy", rank0=True)
        await self._offload_trainer()
        return canonical_state(
            0,
            tensors,
            base_model_revision=self.args.yeto_rl_base_model_revision,
            lora_config_hash=self.args.yeto_rl_lora_config_hash,
            layout_hash=self.args.yeto_rl_layout_hash,
        )

    def initialize(self) -> CanonicalLoraState:
        return self._run(self._initialize())

    async def _engines(self) -> list[Any]:
        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()
        engines = list(info.rollout_engines)
        if not engines:
            raise RuntimeError("Miles created no updatable SGLang engine")
        return engines

    async def _pause_rollout(self) -> None:
        engines = await self._engines()
        # Retracted requests resume after a weight update; abort them so one
        # trajectory can never cross the global policy boundary.
        await asyncio.gather(
            *(engine.pause_generation.remote("abort") for engine in engines)
        )

    async def _resume_rollout(self) -> None:
        engines = await self._engines()
        await asyncio.gather(
            *(engine.continue_generation.remote() for engine in engines)
        )

    async def _set_rollout_version(self, version: int) -> None:
        token = _policy_token(version)
        engines = await self._engines()
        await asyncio.gather(
            *(engine.update_weight_version.remote(token) for engine in engines)
        )

    async def _apply_global_policy(self, state: CanonicalLoraState) -> None:
        from sglang.srt.constants import (
            GPU_MEMORY_TYPE_CUDA_GRAPH,
            GPU_MEMORY_TYPE_KV_CACHE,
            GPU_MEMORY_TYPE_WEIGHTS,
        )

        await self._pause_rollout()
        if not self._rollout_offloaded:
            await self.rollout_manager.offload.remote(
                tags=[
                    GPU_MEMORY_TYPE_CUDA_GRAPH,
                    GPU_MEMORY_TYPE_KV_CACHE,
                    GPU_MEMORY_TYPE_WEIGHTS,
                ]
            )
            self._rollout_offloaded = True
        await self._onload_trainer()
        reset_parameter_count, applied_hash = await self._actor_call(
            "yeto_rl_apply_policy",
            dict(state.tensors),
            state.policy_version,
        )
        expected_hash = policy_hash(state)
        if applied_hash != expected_hash:
            raise StrictRlInvariantError(
                "policy_hash_mismatch_after_apply",
                "policy hash mismatch after trainer apply",
            )
        await self._offload_trainer()
        await self.rollout_manager.onload_weights.remote()
        await self.actor_model.update_weights()
        # update_weights resumes generation; close the admission boundary
        # until KV/weights and the explicit version are all installed.
        await self._pause_rollout()
        await self.rollout_manager.onload_kv.remote()
        self._rollout_offloaded = False
        await self._set_rollout_version(state.policy_version)
        self._policy_version = state.policy_version
        self._rollout_id = state.policy_version
        await self._resume_rollout()
        self._optimizer_reset_count += 1
        self._append_event(
            {
                "event": "rl_policy_apply",
                "policy_version": state.policy_version,
                "optimizer_reset_count": self._optimizer_reset_count,
                "reset_parameter_count": reset_parameter_count,
                "rl/global_policy_version": state.policy_version,
                "rl/optimizer_reset_count": self._optimizer_reset_count,
                "sync/global_policy_hash": applied_hash,
            }
        )

    def apply_global_policy(self, state: CanonicalLoraState) -> None:
        self._run(self._apply_global_policy(state))

    def _rollout_batches(self, data_pack) -> list[Mapping[str, Any]]:
        import ray

        references = data_pack.get("data_ref")
        if not isinstance(references, list):
            raise RuntimeError("Miles returned an invalid rollout shard list")
        expected_dp = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        )
        if len(references) != expected_dp:
            raise RuntimeError(
                f"Miles returned {len(references)} DP shards, expected {expected_dp}"
            )
        return [ray.get(reference.inner) for reference in references]

    def _rollout_metrics(self, policy_version: int) -> dict[str, float]:
        path = Path(self.args.yeto_rl_completed_groups_path).expanduser()
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != _ISLAND_CHECKPOINT_SCHEMA
                or payload.get("policy_version") != policy_version
                or payload.get("config") != _island_checkpoint_config(self.args)
                or not isinstance(payload.get("rollout_metrics"), Mapping)
            ):
                raise RuntimeError("Miles island checkpoint lacks rollout metrics")
            return {
                name: float(value)
                for name, value in payload["rollout_metrics"].items()
            }
        except (TypeError, ValueError) as error:
            raise RuntimeError("Miles returned invalid Yeto group metrics") from error

    async def _run_local_round(
        self,
        expected_policy_version: int,
        groups: int,
        samples_per_group: int,
        optimizer_steps: int,
    ) -> LocalRoundStats:
        from sglang.srt.constants import (
            GPU_MEMORY_TYPE_CUDA_GRAPH,
            GPU_MEMORY_TYPE_KV_CACHE,
            GPU_MEMORY_TYPE_WEIGHTS,
        )

        if expected_policy_version != self._policy_version:
            raise RuntimeError("Miles round requested from an unapplied global policy")
        rollout_started = time.monotonic()
        data_pack = await self.rollout_manager.generate.remote(self._rollout_id)
        rollout_seconds = time.monotonic() - rollout_started
        await self._pause_rollout()
        batches = self._rollout_batches(data_pack)
        versions = [
            sample_versions
            for batch in batches
            for sample_versions in batch.get("weight_versions", [])
        ]
        expected_samples = groups * samples_per_group
        if not isinstance(versions, list) or len(versions) != expected_samples:
            raise RuntimeError(
                f"Miles produced {len(versions) if isinstance(versions, list) else 0} "
                f"versioned samples, expected {expected_samples}"
            )
        try:
            observed = {
                _version_from_token(token)
                for sample_versions in versions
                for token in sample_versions
            }
        except RuntimeError as error:
            raise StrictRlInvariantError(
                "mixed_version_group_count",
                str(error),
            ) from error
        if any(not sample_versions for sample_versions in versions) or observed != {
            expected_policy_version
        }:
            raise StrictRlInvariantError(
                "mixed_version_group_count",
                f"Miles rollout mixed policy versions: {observed}",
            )
        rollout_metrics = self._rollout_metrics(expected_policy_version)

        await self.rollout_manager.offload.remote(
            tags=[
                GPU_MEMORY_TYPE_CUDA_GRAPH,
                GPU_MEMORY_TYPE_KV_CACHE,
                GPU_MEMORY_TYPE_WEIGHTS,
            ]
        )
        self._rollout_offloaded = True
        before = await self._actor_call("yeto_rl_optimizer_steps")
        train_started = time.monotonic()
        await self.actor_model.train(self._rollout_id, data_pack)
        train_seconds = time.monotonic() - train_started
        self._trainer_awake = True
        after = await self._actor_call("yeto_rl_optimizer_steps")
        if after - before != optimizer_steps:
            raise RuntimeError(
                f"Miles performed {after - before} optimizer steps, "
                f"expected {optimizer_steps}"
            )
        train_metrics = await self._actor_call(
            "yeto_rl_train_metrics",
            rank0=True,
        )
        try:
            mean_kl = float(train_metrics["train/train_rollout_kl"])
            ess_ratio = float(train_metrics["train/ess_ratio"])
            clip_fraction = float(train_metrics["train/pg_clipfrac"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Miles did not return required GRPO train metrics") from error
        self._rollout_id += 1
        response_lengths = [
            int(value)
            for batch in batches
            for value in batch.get("response_lengths", [])
        ]
        sample_indices = [
            int(value)
            for batch in batches
            for value in batch.get("sample_indices", [])
        ]
        if (
            len(response_lengths) != expected_samples
            or len(sample_indices) != expected_samples
            or len(set(sample_indices)) != expected_samples
        ):
            raise RuntimeError("Miles DP rollout shards do not form one complete batch")
        raw_rewards = batches[0].get("raw_reward")
        if not isinstance(raw_rewards, list) or len(raw_rewards) != expected_samples:
            raise RuntimeError("Miles rollout lacks scalar raw rewards")
        if any(batch.get("raw_reward") != raw_rewards for batch in batches[1:]):
            raise RuntimeError("Miles DP rollout shards disagree on raw rewards")
        try:
            rewards = [float(value) for value in raw_rewards]
        except (TypeError, ValueError) as error:
            raise RuntimeError("Miles RL v0 requires scalar rewards") from error
        return LocalRoundStats(
            island_id=int(self.args.yeto_rl_learner_id),
            local_round_id=expected_policy_version + 1,
            base_policy_version=expected_policy_version,
            active_groups=int(rollout_metrics["active_groups"]),
            completed_groups=groups,
            cancelled_groups=int(rollout_metrics["cancelled_groups"]),
            completed_trajectories=expected_samples,
            action_tokens=sum(response_lengths),
            tool_wait_seconds=rollout_metrics["tool_wait_seconds"],
            group_p50_seconds=rollout_metrics["group_p50_seconds"],
            group_p95_seconds=rollout_metrics["group_p95_seconds"],
            group_p99_seconds=rollout_metrics["group_p99_seconds"],
            reward_mean=statistics.fmean(rewards),
            reward_std=statistics.pstdev(rewards),
            zero_variance_group_ratio=sum(
                len(set(rewards[index : index + samples_per_group])) == 1
                for index in range(0, expected_samples, samples_per_group)
            )
            / groups,
            mean_kl=mean_kl,
            ess_ratio=ess_ratio,
            clip_fraction=clip_fraction,
            delta_l2_norm=0.0,
            rollout_seconds=rollout_seconds,
            train_seconds=train_seconds,
        )

    def run_local_round(
        self,
        *,
        expected_policy_version: int,
        groups: int,
        samples_per_group: int,
        optimizer_steps: int,
    ) -> LocalRoundStats:
        return self._run(
            self._run_local_round(
                expected_policy_version,
                groups,
                samples_per_group,
                optimizer_steps,
            )
        )

    async def _export_local_policy(self) -> CanonicalLoraState:
        if not self._trainer_awake:
            await self._onload_trainer()
        tensors = await self._actor_call("yeto_rl_export_policy", rank0=True)
        await self._offload_trainer()
        if self._policy_version is None:
            raise RuntimeError("Miles has no applied global policy")
        return canonical_state(
            self._policy_version,
            tensors,
            base_model_revision=self.args.yeto_rl_base_model_revision,
            lora_config_hash=self.args.yeto_rl_lora_config_hash,
            layout_hash=self.args.yeto_rl_layout_hash,
        )

    def export_local_policy(self) -> CanonicalLoraState:
        return self._run(self._export_local_policy())

    def record_local_round(self, stats: LocalRoundStats) -> None:
        path = Path(self.args.yeto_rl_completed_groups_path).expanduser()
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise RuntimeError("cannot update Miles island checkpoint") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _ISLAND_CHECKPOINT_SCHEMA
            or payload.get("policy_version") != stats.base_policy_version
            or payload.get("config") != _island_checkpoint_config(self.args)
        ):
            raise RuntimeError("Miles island checkpoint changed before round commit")
        payload["local_round_id"] = stats.local_round_id
        payload["local_round_stats"] = asdict(stats)
        _atomic_save_island_checkpoint(path, payload)

    def _append_event(self, event: dict[str, Any]) -> None:
        path = Path(self.args.yeto_rl_event_tape).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "island_id": int(self.args.yeto_rl_learner_id),
            "time_unix": time.time(),
            **event,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            )

    async def _shutdown(self) -> None:
        if self.actor_model is not None:
            await self._offload_trainer()
        if self.rollout_manager is not None:
            await self.rollout_manager.dispose.remote()

    def shutdown(self) -> None:
        try:
            self._run(self._shutdown())
        finally:
            if self._owns_ray:
                import ray

                ray.shutdown()
            self.loop.close()
