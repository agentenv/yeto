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


def island_offloads_trainer(*, placement: str, tuning: str = "lora", sample_role: str = "both",
                            offload_train: bool = False) -> bool:
    """Whether the island's Miles argv carries ``--offload-train`` (which loads
    torch_memory_saver into the trainer actors through ``LD_PRELOAD``).

    Mirrors ``config.translate_run_config``: ``--offload-train`` when the user
    asks for it (``--rl-offload-train``) or the trainer shares its GPUs with
    engines (colocated). A full-parameter trainer island (``--tuning full
    --rl-sample-role trainer``) starts no engine whatever ``--rl-placement``
    says (``run_config``: a trainer-only fixed partition, Miles
    ``--debug-train-only`` with ``--no-offload-train``), so it stays resident."""
    if offload_train:
        return True
    if tuning == "full" and sample_role == "trainer":
        return False
    return placement == "colocated"


def check_island_env(*, placement: str, env_pairs, tuning: str = "lora", sample_role: str = "both",
                     offload_train: bool = False) -> None:
    """An island that runs Miles' ``--offload-train`` (:func:`island_offloads_trainer`)
    loads torch_memory_saver, which refuses to start when PyTorch's allocator
    uses expandable segments (``torch_memory_saver.entrypoint``:
    "expandable_segments is not supported yet"). Refused before any GPU starts.
    A resident trainer (no ``--offload-train``) may use expandable segments; on
    the K3.1 full-parameter trainer they cut the allocator's reserved-minus-allocated
    gap from about 80 GiB to about 17 GiB per GPU.
    ``env_pairs``: the island's ``KEY=VALUE`` environment options."""
    if not island_offloads_trainer(placement=placement, tuning=tuning, sample_role=sample_role,
                                   offload_train=offload_train):
        return
    for item in env_pairs or ():
        key, _, value = str(item).partition("=")
        if key == "PYTORCH_CUDA_ALLOC_CONF" and "expandable_segments:True" in value:
            why = ("--rl-offload-train offloads" if offload_train
                   else "a colocated Miles island offloads")
            raise ValueError(
                f"{why} the trainer with torch_memory_saver, which "
                "refuses PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; drop that environment "
                "option, or run the trainer resident: --rl-placement fixed-partition, or a "
                "full-parameter trainer island (--tuning full --rl-sample-role trainer), "
                "without --rl-offload-train")


def check_train_extra_env(*, recipe: str | None, env_pairs) -> None:
    """``YETO_MILES_TRAIN_EXTRA_ARGV`` among the island's ``KEY=VALUE`` environment options
    (``--modal-env``) goes through the allow-list the island applies
    (:func:`yeto.rl.adapters.miles.config.train_extra_argv`, the same function), here,
    before any GPU starts (also under ``--dry-run``). k31-mfu16-x3 started a two-node
    B300 island for a flag the island then refused (``MilesConfigError: flags not
    allowed``) while the local dry run had passed."""
    from yeto.rl.adapters.miles.config import TRAIN_EXTRA_ENV, train_extra_argv
    from yeto.rl.engine.run_config import RECIPE_GENERIC

    for item in env_pairs or ():
        key, sep, value = str(item).partition("=")
        if sep and key == TRAIN_EXTRA_ENV:
            train_extra_argv(recipe or RECIPE_GENERIC, value)


def check_train_memory(island_args) -> None:
    """The trainer memory options of every island (``island_args``: one launch args copy
    per island) through the rules the island applies when it builds its run config
    (:func:`yeto.rl.engine.run_config._train_memory`, the same function), here, before
    any GPU starts (also under ``--dry-run``). They read only the launch options:
    ``--rl-max-tokens-per-gpu`` x CP against ``--seq-len``, the optimizer offload
    fraction, ``--rl-dp-optimizer``, the precision-aware optimizer and
    ``--rl-log-probs-chunk-size``. A Qwen3-4B launch with ``--rl-max-tokens-per-gpu
    16384`` and ``--seq-len 32768`` passed its dry run and was refused only by the
    island, after Modal had started the container (and retried it)."""
    from yeto.rl.engine.run_config import _train_memory

    for iargs in island_args:
        if (getattr(iargs, "rl_backend", None) or "miles") == "miles":  # --rl-island-engine: own rules
            _train_memory(iargs)


def _miles_pin_gap(first: str, feature: str, *, feature_name: str) -> str | None:
    """Why the pinned Miles lacks ``feature`` (registered as ``feature_name`` in
    the lineage tables, first carried by fork commit ``first`` on ports-v1);
    None when the pin carries it."""
    from yeto.rl.adapters.miles import pins

    if pins.pin_has(feature_name, pins.MILES_NEXT_COMMIT):
        return None
    return f"Miles {first[:9]} or later ({feature}); the pinned image ships {pins.MILES_NEXT_COMMIT[:9]}"


def full_parameter_dp_replica_gap() -> str | None:
    """Why the pinned Miles cannot run the full-parameter sync at DP>1; None when it can.

    Miles :data:`~yeto.rl.adapters.miles.pins.MILES_DP_REPLICA_FULL_PARAMETER_COMMIT`
    and later give every parameter one owner rank per data-parallel replica group
    and apply the owner's bytes on the other replicas."""
    from yeto.rl.adapters.miles import pins

    return _miles_pin_gap(pins.MILES_DP_REPLICA_FULL_PARAMETER_COMMIT, "data-parallel replicas",
                          feature_name="dp_replica_full_parameter")


def full_parameter_dp_sharded_master_gap() -> str | None:
    """Why the pinned Miles cannot run the full-parameter sync at DP>1 on
    Megatron's distributed optimizer; None when it can.

    Miles :data:`~yeto.rl.adapters.miles.pins.MILES_DP_SHARDED_MASTER_COMMIT`
    and later read each rank's FP32 master range: the replicas contribute
    their ranges to the owner's export and an apply writes the whole model
    and the local range. Without it DP>1 needs the non-distributed optimizer."""
    from yeto.rl.adapters.miles import pins

    return _miles_pin_gap(pins.MILES_DP_SHARDED_MASTER_COMMIT, "distributed optimizer at DP>1",
                          feature_name="dp_sharded_master")


def host_sync_free_gap() -> str | None:
    """Why the pinned Miles cannot take ``--no-host-sync-free``; None when it can.

    The switch is the typed ``DebugConfig.host_sync_free`` of Miles k31/b3-b7
    (pins feature ``host_sync_free_flag``); the ports-v1 fork only read the
    ``MILES_HOST_SYNC_FREE`` environment variable, and k31 before b3-b7 has no
    sync-free paths at all."""
    from yeto.rl.adapters.miles import pins

    return _miles_pin_gap(pins.MILES_K31_B3B7_COMMIT, "the typed --host-sync-free switch",
                          feature_name="host_sync_free_flag")


def check_host_sync_free(enabled: bool) -> None:
    """``--no-rl-host-sync-free`` on a pin without the switch: refused before launch."""
    if enabled:
        return
    gap = host_sync_free_gap()
    if gap is not None:
        raise ValueError(f"--no-rl-host-sync-free needs {gap}")
