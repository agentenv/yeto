"""verl-only checks of the neutral run config (decoupling 4.2 role; rl-verl-backend).

First step: one GPU per island, FSDP2, no Megatron reference checkpoint.
"""

from __future__ import annotations


def resolve_ref_load(args, model_path) -> str:
    if getattr(args, "megatron_ref_load", None) is not None:
        raise ValueError("--megatron-ref-load is a Miles/Megatron option; verl uses the HF model")
    return str(model_path)


def check_trainer_parallel(actor_gpus: int, tensor_parallel: int, pipeline_parallel: int) -> None:
    if (tensor_parallel, pipeline_parallel) != (1, 1):
        raise ValueError("verl FSDP2 backend: tensor/pipeline parallel must be 1")
    if actor_gpus != 1:
        raise ValueError("verl backend first step: one GPU per island")


# S26 E7 decision: the verl Megatron engine (the only verl path with expert
# parallelism) is not wired into this adapter; MoE runs with EP go to Miles.
EP_REFUSAL = ("verl 后端不支持专家并行（EP>1）：yeto 的 verl 适配层只接 FSDP2，"
              "未接 verl Megatron 引擎（S26 E7 结论：不做）。EP 训练请用 Miles 后端")


def check_expert_parallel(actor_gpus: int, expert_parallel: int) -> None:
    if expert_parallel != 1:
        raise ValueError(f"{EP_REFUSAL}；收到 expert_parallel={expert_parallel}")


def check_global_batch(global_batch: int, data_parallel: int) -> None:
    if data_parallel != 1 or global_batch <= 0:
        raise ValueError("verl backend first step: data parallel 1 and a positive batch")


def check_over_sampling(rollout_batch_size: int, over_sampling_batch_size: int) -> None:
    """agentic-rollout-utilization 6.2: the pinned verl fork has no usable
    over-sample cut-off in synchronous mode (``rollout.over_sample_rate`` is
    defined but never used), so over-sampling is refused, not ignored."""
    if over_sampling_batch_size != rollout_batch_size:
        raise ValueError(
            "verl 后端尚未实现多发截止（阶段 0，任务 6.2b）：--over-sampling-batch-size "
            f"{over_sampling_batch_size} 必须等于 --rollout-batch-size {rollout_batch_size}")


# S25 N10.3b: the advantage estimator the verl agentic path needs
# (``algorithm.adv_estimator``; registered by ``.advantage`` in the trainer process).
AGENTIC_ADV_ESTIMATOR = "yeto_grpo"


# S25 N10.4: the Miles agentic wiring has no verl spelling.  The verl agentic
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
    """Refuse Miles agentic options on the verl backend instead of ignoring them."""
    given = [flag for attr, flag in AGENTIC_FLAGS if getattr(args, attr, None) not in (None, False, [], "")]
    if given:
        raise ValueError(
            "verl 后端不支持 Miles agentic 参数：" + ", ".join(given)
            + "。verl 的 agentic 路径是 yeto HarborAgentLoop（agent loop 名 yeto_harbor，"
            "由数据行 agent_name 选择）")


def refuse_agentic_argv(ignored: list[str]) -> None:
    """Island-entry side of ``check_agentic``: unknown argv the parser left over."""
    flags = {flag for _, flag in AGENTIC_FLAGS}
    given = sorted({token.split("=", 1)[0] for token in ignored} & flags)
    if given:
        raise ValueError("verl 后端不支持 Miles agentic 参数：" + ", ".join(given))
