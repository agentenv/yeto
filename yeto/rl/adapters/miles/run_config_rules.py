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
