"""Pinned Miles adapter implementing Yeto's strict global-policy boundary."""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import importlib
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .bridge import LocalRoundResult
from .core import CanonicalLoraState, PolicyIdentity, canonical_state
from .manifest import MILES_COMMIT, MILES_REPOSITORY

_CANONICAL_PREFIX = "base_model.model."


def verify_miles_revision(root: str | Path) -> Path:
    """Fail unless *root* and the imported package are the pinned Miles tree."""

    root = Path(root).expanduser().resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        origin = subprocess.run(
            ["git", "-C", str(root), "config", "--get", "remote.origin.url"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_changes = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot verify Miles checkout at {root}") from exc
    if commit != MILES_COMMIT:
        raise RuntimeError(f"Miles revision mismatch: expected {MILES_COMMIT}, got {commit}")
    if origin.removesuffix(".git").rstrip("/") != MILES_REPOSITORY:
        raise RuntimeError(f"Miles origin mismatch: expected {MILES_REPOSITORY}, got {origin}")
    if source_changes:
        raise RuntimeError("Miles checkout contains source changes outside the pinned commit")

    miles = importlib.import_module("miles")
    package_path = Path(miles.__file__).resolve()
    if not package_path.is_relative_to(root):
        raise RuntimeError(f"imported Miles package {package_path} is outside {root}")
    return root


def _adapter_sides(actor) -> list[tuple[str, Any]]:
    """Return canonical PEFT names paired with fixed-commit conversion tasks."""

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
                if main is None or main.dtype != torch.float32 or main.numel() != parameter.numel():
                    raise RuntimeError(
                        f"LoRA parameter {side.param_name!r} has no complete FP32 optimizer master"
                    )
                converted = side.mapping.megatron_to_hf(
                    main.view(parameter.shape), side.megatron_module
                )
                if len(converted) != 1:
                    raise RuntimeError(f"ambiguous LoRA mapping for {side.param_name!r}")
                raw_name = next(iter(converted))
                name = raw_name if raw_name.startswith(_CANONICAL_PREFIX) else _CANONICAL_PREFIX + raw_name
                if not name.endswith((".lora_A.weight", ".lora_B.weight")):
                    raise RuntimeError(f"non-PEFT LoRA mapping {name!r}")
                sides.append((name, side))
    names = [name for name, _ in sides]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("Miles produced an empty or duplicate canonical LoRA mapping")
    return sorted(sides, key=lambda item: item[0])


@torch.no_grad()
def _export_fp32_policy(actor) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, side in _adapter_sides(actor):
        parameter = side.param_weight
        main = parameter.main_param.view(parameter.shape)
        converted = side.mapping.megatron_to_hf(main, side.megatron_module)
        value = next(iter(converted.values()))
        tensors[name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    return tensors


def _optimizer_children(optimizer) -> list[Any]:
    return list(getattr(optimizer, "chained_optimizers", (optimizer,)))


def _new_optimizer(actor):
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler

    old_optimizer = actor.optimizer
    actor.optimizer = None
    actor.opt_param_scheduler = None
    for chunk in actor.model:
        for parameter in chunk.parameters():
            if parameter.requires_grad and hasattr(parameter, "main_param"):
                parameter.main_param = None
    del old_optimizer
    gc.collect()
    torch.cuda.empty_cache()

    values = {
        field.name: getattr(actor.args, field.name)
        for field in dataclasses.fields(OptimizerConfig)
        if hasattr(actor.args, field.name)
    }
    config = OptimizerConfig(**values)
    config.timers = None
    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=actor.model,
        use_gloo_process_groups=actor.args.enable_gloo_process_groups,
    )
    for child in _optimizer_children(optimizer):
        if getattr(child.optimizer, "state", None):
            raise RuntimeError("rebuilt Miles optimizer unexpectedly retained state")
    scheduler = get_optimizer_param_scheduler(actor.args, optimizer)
    if scheduler.num_steps != 0:
        raise RuntimeError("rebuilt Miles scheduler did not start at step zero")
    actor.optimizer = optimizer
    actor.opt_param_scheduler = scheduler


@torch.no_grad()
def _copy_masters_to_model(actor) -> None:
    for child in _optimizer_children(actor.optimizer):
        copy = getattr(child, "_copy_main_params_to_model_params", None)
        if copy is None:
            raise RuntimeError("pinned Megatron optimizer lacks main-to-model copy")
        copy()
    for chunk in actor.model:
        chunk.start_param_sync(force_sync=True)


def _actor_export_policy(self):
    return _export_fp32_policy(self)


@torch.no_grad()
def _actor_apply_policy(
    self,
    tensors: Mapping[str, torch.Tensor],
    version: int,
    layout_fingerprint: str,
    policy_hash: str,
):
    sides = dict(_adapter_sides(self))
    if set(tensors) != set(sides):
        missing = sorted(set(sides) - set(tensors))
        extra = sorted(set(tensors) - set(sides))
        raise RuntimeError(f"global LoRA mapping mismatch: missing={missing}, extra={extra}")

    mapped: dict[str, torch.Tensor] = {}
    for name, side in sides.items():
        value = tensors[name].detach().to(device=side.param_weight.device, dtype=torch.float32)
        target = side.mapping.hf_to_megatron(value, side.megatron_module)
        if target.numel() != side.param_weight.numel():
            raise RuntimeError(f"global LoRA shape mismatch for {name!r}")
        mapped[name] = target.reshape(side.param_weight.shape).contiguous()
        side.param_weight.data.copy_(mapped[name])

    _new_optimizer(self)
    for name, side in dict(_adapter_sides(self)).items():
        main = side.param_weight.main_param
        value = mapped[name]
        if main is None or main.dtype != torch.float32 or main.numel() != value.numel():
            raise RuntimeError(f"rebuilt optimizer master mismatch for {name!r}")
        main.view(side.param_weight.shape).copy_(value)
    _copy_masters_to_model(self)
    self.weights_backuper.backup("actor")

    applied = canonical_state(
        version,
        _export_fp32_policy(self),
        expected_layout_fingerprint=layout_fingerprint,
    )
    if applied.policy_hash != policy_hash:
        raise RuntimeError(
            f"trainer policy hash mismatch: expected {policy_hash}, got {applied.policy_hash}"
        )
    self._yeto_rl_policy_version = version
    self._yeto_rl_layout_fingerprint = layout_fingerprint
    return {"version": version, "policy_hash": applied.policy_hash}


def _actor_read_policy_identity(self):
    version = getattr(self, "_yeto_rl_policy_version", None)
    fingerprint = getattr(self, "_yeto_rl_layout_fingerprint", None)
    if version is None or fingerprint is None:
        raise RuntimeError("trainer has not installed a Yeto global policy")
    state = canonical_state(
        version,
        _export_fp32_policy(self),
        expected_layout_fingerprint=fingerprint,
    )
    return {"version": version, "policy_hash": state.policy_hash}


def _actor_optimizer_steps(self) -> int:
    scheduler = self.opt_param_scheduler
    batch = self.args.global_batch_size
    if scheduler is None or batch <= 0 or scheduler.num_steps % batch:
        raise RuntimeError("Miles optimizer step counter is not integral")
    return int(scheduler.num_steps // batch)


def install_miles_actor_adapter() -> None:
    """Install methods before Miles wraps its actor class with ``ray.remote``."""

    from miles.backends.megatron_utils.actor import MegatronTrainRayActor

    methods = {
        "yeto_rl_export_policy": _actor_export_policy,
        "yeto_rl_apply_policy": _actor_apply_policy,
        "yeto_rl_read_policy_identity": _actor_read_policy_identity,
        "yeto_rl_optimizer_steps": _actor_optimizer_steps,
    }
    for name, method in methods.items():
        existing = getattr(MegatronTrainRayActor, name, None)
        if existing not in (None, method):
            raise RuntimeError(f"Miles actor already defines incompatible {name}")
        setattr(MegatronTrainRayActor, name, method)


def _policy_token(identity: PolicyIdentity) -> str:
    return f"yeto:{identity.version}:{identity.policy_hash}"


def _identity_from_token(token: object) -> PolicyIdentity:
    prefix, separator, rest = str(token).partition(":")
    version, separator2, digest = rest.partition(":")
    if prefix != "yeto" or not separator or not separator2:
        raise RuntimeError(f"invalid rollout policy token {token!r}")
    return PolicyIdentity(int(version), digest)


def _validate_rollout_groups(data: object, groups: int, samples_per_group: int) -> None:
    if not isinstance(data, list) or len(data) != groups:
        raise RuntimeError(
            f"Miles produced {len(data) if isinstance(data, list) else 0} groups, "
            f"expected {groups}"
        )
    for index, group in enumerate(data):
        if not isinstance(group, list) or len(group) != samples_per_group:
            raise RuntimeError(
                f"Miles group {index} contains "
                f"{len(group) if isinstance(group, list) else 0} trajectories, "
                f"expected {samples_per_group}"
            )
        for sample in group:
            status = getattr(getattr(sample, "status", None), "value", None)
            if isinstance(sample, list) or status not in {"completed", "truncated"}:
                raise RuntimeError(
                    "Miles groups must contain exactly one complete trajectory per sample"
                )


def generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    """Pinned Miles rollout with Yeto's pre-flatten G/K contract check."""

    from miles.rollout.sglang_rollout import generate_rollout as miles_generate_rollout

    output = miles_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    if not evaluation:
        _validate_rollout_groups(
            output.samples,
            args.rollout_batch_size,
            args.n_samples_per_prompt,
        )
    return output


class MilesIslandRuntime:
    """Synchronous ``IslandRuntime`` wrapper over pinned Miles async APIs."""

    def __init__(self, args, manifest: Mapping[str, Any]) -> None:
        self.args = args
        self.manifest = manifest
        self.loop = asyncio.new_event_loop()
        self.rollout_manager = None
        self.actor_model = None
        self._owns_ray = False
        self._trainer_awake = False
        self._rollout_offloaded = True
        self._trainer_identity: PolicyIdentity | None = None
        self._rollout_id = 0

    def _run(self, coroutine):
        return self.loop.run_until_complete(coroutine)

    async def _actor_call(self, method: str, *args):
        if hasattr(self.actor_model, "_broadcast"):
            results = await self.actor_model._broadcast(method, *args)
        elif hasattr(self.actor_model, "_execute_first_alive"):
            results = await self.actor_model._execute_first_alive(method, *args)
        else:
            raise RuntimeError("unsupported pinned Miles RayTrainGroup")
        while isinstance(results, list) and len(results) == 1:
            results = results[0]
        if isinstance(results, list):
            raise RuntimeError("RL v0 expected exactly one Miles trainer rank")
        return results

    async def _initialize(self) -> Mapping[str, torch.Tensor]:
        import ray
        from miles.ray.placement_group import (
            create_placement_groups,
            create_rollout_manager,
            create_training_models,
        )

        install_miles_actor_adapter()
        if not ray.is_initialized():
            ray.init(include_dashboard=False)
            self._owns_ray = True
        if int(ray.cluster_resources().get("GPU", 0)) != 1:
            raise RuntimeError("RL v0 requires exactly one visible GPU per Miles island")
        groups = create_placement_groups(self.args)
        self.rollout_manager, _ = create_rollout_manager(self.args, groups["rollout"])
        self.actor_model, critic = await create_training_models(
            self.args, groups, self.rollout_manager
        )
        if critic is not None:
            raise RuntimeError("RL v0 does not support a Miles critic")
        await self.actor_model.onload()
        self._trainer_awake = True
        tensors = await self._actor_call("yeto_rl_export_policy")
        await self.actor_model.offload()
        self._trainer_awake = False
        await self._pause_rollout()
        return tensors

    def initialize(self) -> Mapping[str, torch.Tensor]:
        return self._run(self._initialize())

    async def _engines(self) -> list[Any]:
        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()
        engines = list(info.rollout_engines)
        if not engines:
            raise RuntimeError("Miles created no updatable SGLang engine")
        return engines

    async def _pause_rollout(self) -> None:
        engines = await self._engines()
        await asyncio.gather(*(engine.pause_generation.remote("retract") for engine in engines))

    def cancel_or_drain_rollouts(self) -> None:
        self._run(self._pause_rollout())

    async def _set_rollout_identity(self, identity: PolicyIdentity) -> None:
        token = _policy_token(identity)
        engines = await self._engines()
        await asyncio.gather(*(engine.update_weight_version.remote(token) for engine in engines))
        actual = await asyncio.gather(*(engine.get_weight_version.remote() for engine in engines))
        if any(value != token for value in actual):
            raise RuntimeError(f"SGLang policy identity mismatch: expected {token}, got {actual}")

    async def _apply_global_policy(self, state: CanonicalLoraState) -> None:
        from sglang.srt.constants import (
            GPU_MEMORY_TYPE_CUDA_GRAPH,
            GPU_MEMORY_TYPE_KV_CACHE,
            GPU_MEMORY_TYPE_WEIGHTS,
        )

        await self._pause_rollout()
        if not self._rollout_offloaded:
            await self.rollout_manager.offload.remote(
                tags=[GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS]
            )
            self._rollout_offloaded = True
        await self.actor_model.onload()
        self._trainer_awake = True
        result = await self._actor_call(
            "yeto_rl_apply_policy",
            dict(state.tensors),
            state.policy_version,
            state.layout_fingerprint,
            state.policy_hash,
        )
        if result != {"version": state.policy_version, "policy_hash": state.policy_hash}:
            raise RuntimeError(f"Miles trainer returned inconsistent policy identity {result}")
        await self.actor_model.offload()
        self._trainer_awake = False
        await self.rollout_manager.onload_weights.remote()
        await self.actor_model.update_weights()
        # The pinned Miles updater resumes generation after publishing. Restore
        # Yeto's stricter boundary before exposing the installed identity.
        await self._pause_rollout()
        await self.rollout_manager.onload_kv.remote()
        self._rollout_offloaded = False
        await self._set_rollout_identity(state.identity)
        self._trainer_identity = state.identity
        self._rollout_id = state.policy_version

    def apply_global_policy(self, state: CanonicalLoraState) -> None:
        self._run(self._apply_global_policy(state))

    @staticmethod
    def _rollout_batch(data_pack) -> Mapping[str, Any]:
        import ray

        references = data_pack.get("data_ref")
        if not isinstance(references, list) or len(references) != 1:
            raise RuntimeError("RL v0 expected one DP=1 Miles rollout shard")
        return ray.get(references[0].inner)

    async def _run_local_round(
        self,
        identity: PolicyIdentity,
        groups: int,
        samples_per_group: int,
        optimizer_steps: int,
    ) -> LocalRoundResult:
        from sglang.srt.constants import (
            GPU_MEMORY_TYPE_CUDA_GRAPH,
            GPU_MEMORY_TYPE_KV_CACHE,
            GPU_MEMORY_TYPE_WEIGHTS,
        )

        if identity != self._trainer_identity:
            raise RuntimeError("Miles local round requested from an uninstalled global policy")
        engines = await self._engines()
        await asyncio.gather(*(engine.continue_generation.remote() for engine in engines))
        rollout_started = time.monotonic()
        data_pack = await self.rollout_manager.generate.remote(self._rollout_id)
        await self._pause_rollout()
        batch = self._rollout_batch(data_pack)
        versions = batch.get("weight_versions")
        expected_samples = groups * samples_per_group
        if not isinstance(versions, list) or len(versions) != expected_samples:
            raise RuntimeError(
                f"Miles produced {len(versions) if isinstance(versions, list) else 0} "
                f"versioned samples, expected {expected_samples}"
            )
        observed = {
            _identity_from_token(token)
            for sample_versions in versions
            for token in sample_versions
        }
        if any(not sample_versions for sample_versions in versions) or observed != {identity}:
            raise RuntimeError(f"Miles rollout mixed policy identities: {observed}")
        rollout_seconds = time.monotonic() - rollout_started

        await self.rollout_manager.offload.remote(
            tags=[GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS]
        )
        self._rollout_offloaded = True
        await self.actor_model.onload()
        self._trainer_awake = True
        train_started = time.monotonic()
        before = await self._actor_call("yeto_rl_optimizer_steps")
        await self.actor_model.train(self._rollout_id, data_pack)
        after = await self._actor_call("yeto_rl_optimizer_steps")
        train_seconds = time.monotonic() - train_started
        if after - before != optimizer_steps:
            raise RuntimeError(
                f"Miles performed {after - before} optimizer steps, expected {optimizer_steps}"
            )
        self._rollout_id += 1
        return LocalRoundResult(
            groups=groups,
            samples_per_group=samples_per_group,
            optimizer_steps=optimizer_steps,
            rollout_identities=frozenset({identity}),
            rollout_seconds=rollout_seconds,
            train_seconds=train_seconds,
            stats={"optimizer_step_start": before, "optimizer_step_end": after},
        )

    def run_local_round(
        self,
        policy_identity: PolicyIdentity,
        *,
        groups: int,
        samples_per_group: int,
        optimizer_steps: int,
    ) -> LocalRoundResult:
        return self._run(
            self._run_local_round(
                policy_identity, groups, samples_per_group, optimizer_steps
            )
        )

    async def _export_local_policy(self) -> Mapping[str, torch.Tensor]:
        if not self._trainer_awake:
            await self.actor_model.onload()
            self._trainer_awake = True
        tensors = await self._actor_call("yeto_rl_export_policy")
        await self.actor_model.offload()
        self._trainer_awake = False
        return tensors

    def export_local_policy(self) -> Mapping[str, torch.Tensor]:
        return self._run(self._export_local_policy())

    def read_trainer_policy_identity(self) -> PolicyIdentity:
        if self._trainer_identity is None:
            raise RuntimeError("Miles trainer has no global policy identity")
        return self._trainer_identity

    async def _read_rollout_policy_identity(self) -> PolicyIdentity:
        engines = await self._engines()
        versions = await asyncio.gather(*(engine.get_weight_version.remote() for engine in engines))
        identities = {_identity_from_token(version) for version in versions}
        if len(identities) != 1:
            raise RuntimeError(f"SGLang engines disagree on policy identity: {identities}")
        return identities.pop()

    def read_rollout_policy_identity(self) -> PolicyIdentity:
        return self._run(self._read_rollout_policy_identity())

    async def _shutdown(self) -> None:
        if self.actor_model is not None and self._trainer_awake:
            await self.actor_model.offload()
            self._trainer_awake = False
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
