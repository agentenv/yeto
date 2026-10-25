"""Miles/Megatron-only checks of ``resolve_rl_run_config`` (yeto-framework-decoupling 4.2, audit E10).

``yeto.rl.engine.run_config`` resolves the neutral run (parallel layout,
batch, LR plan, reference model path) and calls these at the same points as
the pre-split ``build_miles_argv`` did, so the validation order and the error
texts are unchanged; only the Miles-specific rules live here.
"""

from __future__ import annotations

from pathlib import Path


def resolve_ref_load(args, model_path) -> str:
    """``--megatron-ref-load``: an absolute local Megatron *release* checkpoint."""
    configured = getattr(args, "megatron_ref_load", None)
    if configured is None:
        return str(model_path)
    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute():
        raise ValueError("--megatron-ref-load must be an absolute local path")
    if configured_path.is_symlink() or not configured_path.is_dir():
        raise ValueError("--megatron-ref-load must be a real local directory")
    release_marker = configured_path / "latest_checkpointed_iteration.txt"
    try:
        marker = release_marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError("--megatron-ref-load has no readable release marker") from exc
    if release_marker.is_symlink() or marker != "release":
        raise ValueError("--megatron-ref-load is not a release checkpoint")
    return str(configured_path.resolve())


def check_trainer_parallel(actor_gpus: int, tensor_parallel: int, pipeline_parallel: int) -> None:
    if tensor_parallel <= 0 or pipeline_parallel <= 0 or actor_gpus % (tensor_parallel * pipeline_parallel):
        raise ValueError("Miles actor world must be divisible by TP*PP")


def check_expert_parallel(actor_gpus: int, expert_parallel: int) -> None:
    if actor_gpus % expert_parallel:
        raise ValueError("expert parallelism must divide Miles actor world size")


def check_global_batch(global_batch: int, data_parallel: int) -> None:
    if global_batch % data_parallel:
        raise ValueError("Miles global batch must divide evenly across DP ranks")


def check_island_env(*, placement: str, env_pairs) -> None:
    """S27 G-EF: a colocated island runs Miles' ``--offload-train`` (the adapter
    forces it), whose torch_memory_saver refuses to start when PyTorch's
    allocator uses expandable segments (``torch_memory_saver.entrypoint``:
    "expandable_segments is not supported yet"). Refused before any GPU starts.
    ``env_pairs``: the island's ``KEY=VALUE`` environment options."""
    if placement != "colocated":
        return
    for item in env_pairs or ():
        key, _, value = str(item).partition("=")
        if key == "PYTORCH_CUDA_ALLOC_CONF" and "expandable_segments:True" in value:
            raise ValueError(
                "a colocated Miles island offloads the trainer with torch_memory_saver, which "
                "refuses PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; drop that environment "
                "option or use --rl-placement fixed-partition")


def _miles_pin_gap(first: str, feature: str) -> str | None:
    """Why the pinned Miles lacks ``feature``, first in fork commit ``first``; None when it has it."""
    from yeto.rl.adapters.miles import pins

    if first in pins.MILES_FORK_LINEAGE and pins.MILES_NEXT_COMMIT in pins.miles_fork_lineage_from(first):
        return None
    return f"Miles {first[:9]} or later ({feature}); the pinned image ships {pins.MILES_NEXT_COMMIT[:9]}"


def full_parameter_dp_replica_gap() -> str | None:
    """Why the pinned Miles cannot run the full-parameter sync at DP>1; None when it can.

    Miles :data:`~yeto.rl.adapters.miles.pins.MILES_DP_REPLICA_FULL_PARAMETER_COMMIT`
    and later give every parameter one owner rank per data-parallel replica group
    and apply the owner's bytes on the other replicas."""
    from yeto.rl.adapters.miles import pins

    return _miles_pin_gap(pins.MILES_DP_REPLICA_FULL_PARAMETER_COMMIT, "data-parallel replicas")


def full_parameter_dp_sharded_master_gap() -> str | None:
    """Why the pinned Miles cannot run the full-parameter sync at DP>1 on
    Megatron's distributed optimizer; None when it can.

    Miles :data:`~yeto.rl.adapters.miles.pins.MILES_DP_SHARDED_MASTER_COMMIT`
    and later read each rank's FP32 master range: the replicas contribute
    their ranges to the owner's export and an apply writes the whole model
    and the local range. Without it DP>1 needs the non-distributed optimizer."""
    from yeto.rl.adapters.miles import pins

    return _miles_pin_gap(pins.MILES_DP_SHARDED_MASTER_COMMIT, "distributed optimizer at DP>1")
