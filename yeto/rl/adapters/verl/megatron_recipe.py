"""verl Megatron full-parameter GRPO overrides for Nemotron-H models.

Source of the plan: the fork's official example
``examples/grpo_trainer/run_nemotron_nano_v3_30b_a3b_megatron.sh`` (fork
acad9875): one node of 8 GPUs, train TP8 / PP1 / EP8 / ETP1, rollout vLLM
TP8, bf16, full recompute (uniform, 1 layer), router in fp32, parameters and
optimizer offloaded to the host (``optimizer_cpu_offload``, precision-aware
optimizer), Megatron-Bridge model loading. :class:`NemotronMegatronPlan`
keeps those choices and lets the card count shrink (TP = EP = rollout TP =
cards by default, as in the example).

This is the plan for the bare verl trainer (``verl.trainer.main_ppo``). yeto's
own verl island (``config.py``) is LoRA on one card; full-parameter verl
islands in the multi-island syncer are not wired (GOAL B10).

Frozen vision: verl's ``actor.freeze_vision_tower`` exists in the fork's
config (``verl/workers/config/actor.py``) but no code reads it, so a VL model
would train its vision part. :func:`validate` therefore refuses a VL model;
the name rule for the freeze is :mod:`yeto.rl.hybrid_params`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

VL_ARCHITECTURES = ("NemotronH_Omni_Reasoning_V3",)


class NemotronPlanError(ValueError):
    pass


@dataclass(frozen=True)
class NemotronMegatronPlan:
    model_path: str
    train_file: str
    val_file: str
    out_dir: str
    architecture: str = "NemotronHForCausalLM"
    num_experts: int = 128
    n_gpus: int = 8
    nnodes: int = 1
    tp: int = 8
    pp: int = 1
    ep: int = 8
    etp: int = 1
    cp: int = 1
    rollout_tp: int = 8
    offload: bool = True
    groups_per_round: int = 32
    samples_per_group: int = 16
    max_prompt_len: int = 2048
    max_response_len: int = 2048
    lr: float = 1e-6
    total_steps: int = 3
    gpu_memory_utilization: float = 0.7
    reward_path: str = ""
    reward_name: str = "compute_score"
    experiment_name: str = "yeto-nemotron-verl"

    def to_dict(self) -> dict:
        return asdict(self)


def validate(p: NemotronMegatronPlan) -> None:
    if p.architecture in VL_ARCHITECTURES:
        raise NemotronPlanError(
            "verl 不冻结视觉部分（freeze_vision_tower 无代码读取）；VL 模型需先接入冻结规则")
    world = p.n_gpus * p.nnodes
    model_parallel = p.tp * p.pp * p.cp
    if world % model_parallel:
        raise NemotronPlanError(f"TP*PP*CP={model_parallel} 不整除卡数 {world}")
    if p.num_experts % p.ep:
        raise NemotronPlanError(f"EP={p.ep} 不整除专家数 {p.num_experts}")
    # Megatron: EP x ETP must divide the expert-side world (TP*CP*DP folded per PP stage).
    if (world // p.pp) % (p.ep * p.etp):
        raise NemotronPlanError(f"EP*ETP={p.ep * p.etp} 不整除每个流水级的卡数 {world // p.pp}")
    if p.rollout_tp > p.n_gpus or p.n_gpus % p.rollout_tp:
        raise NemotronPlanError(f"rollout TP={p.rollout_tp} 必须整除每机卡数 {p.n_gpus}")
    if p.groups_per_round <= 0 or p.samples_per_group <= 0 or p.total_steps <= 0:
        raise NemotronPlanError("groups_per_round / samples_per_group / total_steps 必须为正")


def build_overrides(p: NemotronMegatronPlan) -> list[str]:
    """Hydra overrides for ``verl.trainer.main_ppo`` (same keys as the example)."""
    validate(p)
    seq = p.max_prompt_len + p.max_response_len
    meg = "actor_rollout_ref.actor.megatron"
    tcfg = f"+{meg}.override_transformer_config"
    ocfg = "+actor_rollout_ref.actor.optim.override_optimizer_config"
    o = [
        "model_engine=megatron",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.0",
        f"data.train_files={p.train_file}",
        f"data.val_files={p.val_file}",
        "data.prompt_key=prompt",
        "data.return_raw_chat=True",
        "data.truncation=left",
        f"data.max_prompt_length={p.max_prompt_len}",
        f"data.max_response_length={p.max_response_len}",
        f"data.train_batch_size={p.groups_per_round}",
        f"actor_rollout_ref.model.path={p.model_path}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_fused_kernels=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.clip_ratio_c=10.0",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={seq}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={p.groups_per_round}",
        f"actor_rollout_ref.actor.optim.lr={p.lr!r}",
        "actor_rollout_ref.actor.optim.weight_decay=0.1",
        "actor_rollout_ref.actor.optim.clip_grad=1.0",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        f"{meg}.param_offload={p.offload}",
        f"{meg}.optimizer_offload={p.offload}",
        f"{meg}.tensor_model_parallel_size={p.tp}",
        f"{meg}.pipeline_model_parallel_size={p.pp}",
        f"{meg}.context_parallel_size={p.cp}",
        f"{meg}.expert_model_parallel_size={p.ep}",
        f"{meg}.expert_tensor_parallel_size={p.etp}",
        f"{meg}.use_mbridge=True",
        f"{meg}.dtype=bfloat16",
        f"{tcfg}.apply_rope_fusion=True",
        f"{tcfg}.moe_router_dtype=fp32",
        f"{tcfg}.recompute_method=uniform",
        f"{tcfg}.recompute_granularity=full",
        f"{tcfg}.recompute_num_layers=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={p.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={p.gpu_memory_utilization}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={seq}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        f"actor_rollout_ref.rollout.n={p.samples_per_group}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={seq}",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={seq}",
        "trainer.logger=[console,file]",
        "trainer.project_name=yeto-nemotron",
        f"trainer.experiment_name={p.experiment_name}",
        f"trainer.n_gpus_per_node={p.n_gpus}",
        f"trainer.nnodes={p.nnodes}",
        "trainer.critic_warmup=0",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.resume_mode=disable",
        f"trainer.total_training_steps={p.total_steps}",
        "trainer.total_epochs=1000",
        f"trainer.default_local_dir={p.out_dir}/ckpt",
        "ray_kwargs.ray_init.runtime_env.py_executable=null",
    ]
    if p.offload:
        o += [f"{ocfg}.optimizer_offload_fraction=1",
              f"{ocfg}.overlap_cpu_optimizer_d2h_h2d=True",
              f"{ocfg}.use_precision_aware_optimizer=True",
              f"{ocfg}.optimizer_cpu_offload=True"]
    if p.reward_path:
        o += [f"reward.custom_reward_function.path={p.reward_path}",
              f"reward.custom_reward_function.name={p.reward_name}"]
    return o


def image_commands(max_jobs: int = 16) -> list[str]:
    """Extra image layers on top of the verl uv venv: the megatron extra plus the
    Mamba training kernels the extra does not list (the example installs them by
    hand: causal-conv1d and mamba-ssm). Hopper only; mamba-ssm's own CUDA
    selective-scan is skipped (Mamba-2 training uses its Triton kernels)."""
    from .image import VENV_BIN
    from .pins import VERL_PYTHON, VERL_ROOT
    env = f"MAX_JOBS={max_jobs} TORCH_CUDA_ARCH_LIST=9.0"
    return [
        f"cd {VERL_ROOT} && UV_PYTHON=/usr/local/bin/python{VERL_PYTHON} uv sync --frozen --all-packages"
        " --extra vllm --extra megatron",
        f"uv pip install --python {VENV_BIN}/python ninja packaging wheel setuptools",
        f"{env} CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --python {VENV_BIN}/python --no-build-isolation"
        " 'causal-conv1d==1.6.2.post1'",
        f"{env} MAMBA_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE uv pip install --python {VENV_BIN}/python"
        " --no-build-isolation --no-deps 'mamba-ssm==2.3.2.post1'",
    ]
