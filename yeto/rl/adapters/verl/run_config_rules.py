"""verl-only checks of the neutral run config.

Two training engines (``megatron_engine.engine_for``):

* FSDP2: TP = PP = EP = 1, data parallel over the island's cards;
* Megatron: TP / PP / EP > 1; the run then trains full
  parameters on the Megatron engine island (``megatron_engine``).

No Megatron reference checkpoint either way (verl loads the HF model).
"""

from __future__ import annotations


def resolve_ref_load(args, model_path) -> str:
    if getattr(args, "megatron_ref_load", None) is not None:
        raise ValueError("--megatron-ref-load is a Miles/Megatron option; verl uses the HF model")
    return str(model_path)


def check_trainer_parallel(actor_gpus: int, tensor_parallel: int, pipeline_parallel: int) -> None:
    if tensor_parallel <= 0 or pipeline_parallel <= 0 or actor_gpus <= 0:
        raise ValueError("verl backend: trainer GPUs, TP and PP must be positive")
    if actor_gpus % (tensor_parallel * pipeline_parallel):
        raise ValueError("verl backend: TP*PP must divide the trainer GPUs (TP/PP > 1: Megatron engine)")


# EP > 1 trains on the Megatron engine island (megatron_engine).
EP_NEEDS_MEGATRON = ("verl backend expert parallelism (EP>1) only runs on the Megatron engine island "
                     "(megatron_engine): it requires full parameters, --rl-single-island-no-sync or "
                     "--rl-sync-preset full-parameter, --rl-placement fixed-partition")


def check_expert_parallel(actor_gpus: int, expert_parallel: int) -> None:
    if expert_parallel < 1 or actor_gpus % expert_parallel:
        raise ValueError(f"verl backend: EP={expert_parallel} must divide the trainer GPUs {actor_gpus}")


def check_global_batch(global_batch: int, data_parallel: int) -> None:
    if data_parallel < 1 or global_batch <= 0 or global_batch % data_parallel:
        raise ValueError("verl backend: the global batch must be positive and divide evenly across DP ranks")


def megatron_problems(args) -> list[str]:
    """Why a run with TP/PP/EP > 1 cannot run on the verl Megatron island ([] when it can);
    the island refuses the same combinations at start (``island_entry.megatron_island_plan``).

    One island without an outer sync (``--rl-single-island-no-sync``), or
    islands in a full-parameter session (``--rl-sync-preset full-parameter``,
    ``megatron_engine.sync_problems``)."""
    from yeto.rl.engine.selection import FULL_PARAMETER_SYNC_PRESET, SYNC_PRESET_ALIASES

    from .megatron_engine import sync_problems

    problems = []
    if getattr(args, "rl_single_island_no_sync", False):
        if getattr(args, "tuning", "lora") != "full":
            problems.append("--tuning full")
    else:  # the full-parameter preset trains every parameter whatever --tuning says
        preset = getattr(args, "rl_sync_preset", None)
        if SYNC_PRESET_ALIASES.get(preset, preset) != FULL_PARAMETER_SYNC_PRESET:
            problems.append("--rl-single-island-no-sync with --tuning full, or --rl-sync-preset full-parameter")
        problems += sync_problems(args)
    if getattr(args, "rl_placement", "colocated") != "fixed-partition":
        problems.append("--rl-placement fixed-partition")
    if int(getattr(args, "rl_max_policy_age", 0) or 0) != 0 and not int(getattr(args, "rl_rollout_ahead", 0) or 0):
        problems.append("--rl-max-policy-age 0, or --rl-rollout-ahead 1")  # Rollout ahead
    return problems


def check_engine_shape(args) -> None:
    """Launcher-side refusal of a Megatron-engine run the island cannot run."""
    if not megatron_run(args):
        return
    problems = megatron_problems(args)
    if problems:
        raise ValueError(f"{EP_NEEDS_MEGATRON}; missing: " + ", ".join(problems))


def megatron_run(args) -> bool:
    """The run trains on the verl Megatron island (TP/PP/EP/CP > 1)."""
    from .megatron_engine import ENGINE_MEGATRON, engine_for

    return engine_for(tensor_parallel=getattr(args, "tensor_parallel", 1) or 1,
                      pipeline_parallel=getattr(args, "pipeline_parallel", 1) or 1,
                      expert_parallel=getattr(args, "expert_parallel", 1) or 1,
                      context_parallel=getattr(args, "context_parallel", 1) or 1) == ENGINE_MEGATRON


def check_over_sampling(rollout_batch_size: int, over_sampling_batch_size: int) -> None:
    """The pinned verl fork has no usable
    over-sample cut-off in synchronous mode (``rollout.over_sample_rate`` is
    defined but never used), so over-sampling is refused, not ignored."""
    if over_sampling_batch_size != rollout_batch_size:
        raise ValueError(
            "the verl backend does not implement the over-sampling cut-off: --over-sampling-batch-size "
            f"{over_sampling_batch_size} must equal --rollout-batch-size {rollout_batch_size}")


# The advantage estimator the verl agentic path needs
# (``algorithm.adv_estimator``; registered by ``.advantage`` in the trainer process).
AGENTIC_ADV_ESTIMATOR = "yeto_grpo"


# The Miles agentic wiring has no verl spelling.  The verl agentic
# path is the yeto ``HarborAgentLoop`` (agent loop ``yeto_harbor``), selected per
# data row, not these flags; passing them used to be ignored without a word.
AGENTIC_FLAGS = (
    ("custom_generate_function_path", "--custom-generate-function-path"),
    ("custom_agent_function_path", "--custom-agent-function-path"),
    ("use_session_server", "--use-session-server"),
    ("tito_model", "--tito-model"),
    ("agent_max_seq_len", "--agent-max-seq-len"),
    ("codex_backend_profile", "--codex-backend-profile"),
    ("tito_allowed_append_roles", "--tito-allowed-append-roles"),
    ("session_server_ip", "--session-server-ip"),
    ("session_server_port", "--session-server-port"),
    ("session_server_workers", "--session-server-workers"),
)


def check_agentic(args) -> None:
    """Refuse Miles agentic options on the verl backend instead of ignoring them,
    and refuse a ``--rl-agentic-harbor`` shape the island cannot run."""
    given = [flag for attr, flag in AGENTIC_FLAGS if getattr(args, attr, None) not in (None, False, [], "")]
    if given:
        raise ValueError(
            "the verl backend does not support Miles agentic options: " + ", ".join(given)
            + ". The verl agentic path is the yeto HarborAgentLoop (agent loop yeto_harbor, "
            "selected by each data row's agent_name)")
    if getattr(args, "rl_agentic_harbor", False):
        problems = agentic_problems(args)
        if problems:
            raise ValueError("--rl-agentic-harbor on the verl backend only covers the synchronous colocated training island; missing: "
                             + ", ".join(problems))


def agentic_problems(args) -> list[str]:
    """Why ``--rl-agentic-harbor`` cannot run on this verl shape ([] when it can).

    Covered: the synchronous training island that both generates and
    trains (``trainer.py`` + ``AgentLoopManagerTQ``, the records proxy over its
    own rollout servers).  Not covered, and refused rather than ignored: the
    Megatron engine island (its own round loop, ``megatron_ports``), the
    fully_async rollouter (it replaces every row's ``agent_name``,
    ``fully_async_round.py:213``), the sampler / buffer-trainer split, and
    in-training evaluation (the held-out rows would need trials of their own).
    """
    problems = []
    if megatron_run(args):
        problems.append("TP=PP=EP=1 (the Megatron engine island has no agent loop manager)")
    if int(getattr(args, "rl_max_policy_age", 0) or 0) and not int(getattr(args, "rl_rollout_ahead", 0) or 0):
        problems.append("--rl-max-policy-age 0 (the fully_async rollouter rewrites every row's agent_name)")
    if (getattr(args, "rl_sample_role", None) or "both") != "both" or getattr(args, "rl_island_sample_role", None):
        problems.append("no sampler / buffer-trainer split")
    if getattr(args, "rl_eval_interval", None) is not None:
        problems.append("no --rl-eval-interval (the eval rows would need trials too; not wired)")
    return problems


def refuse_agentic_argv(ignored: list[str]) -> None:
    """Island-entry side of ``check_agentic``: unknown argv the parser left over."""
    flags = {flag for _, flag in AGENTIC_FLAGS}
    given = sorted({token.split("=", 1)[0] for token in ignored} & flags)
    if given:
        raise ValueError("the verl backend does not support Miles agentic options: " + ", ".join(given))


def hf_export_supported(args) -> bool:
    """``--rl-export-hf-rounds`` on verl (``launch.rl_ports._check_hf_export_launch_flags``):
    the Megatron island and synced full-parameter islands
    (``hf_export.VerlHfWriter``)."""
    from yeto.rl.engine.selection import FULL_PARAMETER_SYNC_PRESET

    return megatron_run(args) or (getattr(args, "training_mode", "sft") == "rl"
                                  and getattr(args, "rl_sync_preset", None) == FULL_PARAMETER_SYNC_PRESET)


def check_eval(args) -> None:
    """``--rl-eval-interval`` on verl (``eval_port``): the synchronous
    trainer runs verl's ``_validate`` behind the driver's evaluate hook; the
    fully_async and buffer-trainer paths and the Megatron island
    (``megatron_ports.run_megatron_island``) have no such hook, so they refuse the
    eval before launch (the island refuses it again)."""
    if getattr(args, "rl_eval_interval", None) is None:
        return
    if megatron_run(args):
        raise ValueError("--rl-eval-interval on verl does not cover the Megatron island "
                         "(TP/PP/EP > 1): it has no evaluate hook")
    if int(getattr(args, "rl_max_policy_age", 0) or 0) > 0 and not int(getattr(args, "rl_rollout_ahead", 0) or 0):
        # Rollout ahead keeps the synchronous trainer; the evaluation runs after
        # the publication, while no batch generates (the driver joined it before)
        raise ValueError("--rl-eval-interval on verl needs --rl-max-policy-age 0 (or --rl-rollout-ahead 1): "
                         "verl's _validate runs on the synchronous trainer; the fully_async path has no "
                         "evaluate hook")
    if ((getattr(args, "rl_sample_role", None) or "both") != "both"
            or getattr(args, "rl_island_sample_role", None)):
        raise ValueError("--rl-eval-interval on verl does not cover sampler / buffer-trainer islands "
                         "(no evaluate hook on that path)")
