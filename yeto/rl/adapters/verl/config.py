"""yeto run options -> checked verl hydra overrides (rl-verl-backend 1.3, design D4/D7/D11).

Rules enforced before anything starts (each has a CPU test):

* LoRA ``lora_alpha`` must equal ``rank`` (yeto canonical LoRA, D4);
* the verl source commit must equal the pin (D1);
* bypass mode (old log-probs = rollout log-probs) cannot be combined with any
  mismatch criterion (D7: the criteria would compare a tensor with itself);
* ``full_determinism`` is refused in training mode, and in diagnostic mode it
  requires ``enforce_eager`` (S16: vLLM's Inductor autotune refuses
  deterministic mode; deterministic sampling also makes the GRPO group
  identical, i.e. no learning signal);
* corrections: only TIS with lower bound 0 is mapped; IcePop is in the design
  but its verl mapping is not verified yet, so it is refused for now; every
  other correction is refused with "verl 后端不支持此修正";
* sampling defaults are T=1, top_p=1, top_k=-1; non-default top_p/top_k need
  ``allow_sampling_shift`` (the criteria then see a systematic shift).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .pins import VERL_COMMIT

MODES = ("train", "diagnose")
SUPPORTED_CORRECTIONS = ("none", "tis")
DESIGNED_NOT_MAPPED = ("icepop",)  # in design D7, mapping to verl not verified yet


class VerlConfigError(ValueError):
    """The run cannot be expressed on the verl backend as configured."""


@dataclass(frozen=True)
class VerlRunConfig:
    model_path: str
    train_file: str
    val_file: str
    out_dir: str
    lora_rank: int = 32
    lora_alpha: int = 32
    target_modules: str = "all-linear"
    lr: float = 1e-5
    groups_per_round: int = 32
    samples_per_group: int = 4
    max_prompt_len: int = 512
    max_response_len: int = 1024
    seed: int = 1
    total_rounds: int = 3
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    allow_sampling_shift: bool = False
    mode: str = "train"
    full_determinism: bool = False
    enforce_eager: bool = False
    bypass_mode: bool = False
    criteria_enabled: bool = True
    correction: str = "tis"
    tis_lower: float = 0.0
    tis_upper: float = 2.0
    lora_merge: bool = False
    engine_commit: str = VERL_COMMIT
    gpu_memory_utilization: float = 0.5
    n_gpus: int = 1
    model_dtype: str = "fp32"
    reward_path: str = ""
    reward_name: str = "compute_score"
    chat_template_kwargs: dict = field(default_factory=dict)
    experiment_name: str = "yeto-verl"

    def to_dict(self) -> dict:
        return asdict(self)


def validate(cfg: VerlRunConfig, *, expected_commit: str = VERL_COMMIT) -> None:
    """Raise :class:`VerlConfigError` naming the first violated rule."""
    if cfg.lora_rank <= 0:
        raise VerlConfigError("verl 后端第一版只支持 LoRA（lora_rank 必须 > 0）")
    if cfg.lora_alpha != cfg.lora_rank:
        raise VerlConfigError(
            f"lora_alpha ({cfg.lora_alpha}) 必须等于 rank ({cfg.lora_rank})：yeto 规范 LoRA 不带缩放")
    if cfg.engine_commit != expected_commit:
        raise VerlConfigError(
            f"verl 源码 commit {cfg.engine_commit} 与钉定的 {expected_commit} 不符")
    if cfg.mode not in MODES:
        raise VerlConfigError(f"mode 必须是 {MODES} 之一，得到 {cfg.mode!r}")
    if cfg.bypass_mode and cfg.criteria_enabled:
        raise VerlConfigError("旁路模式（old=rollout）与训推不一致判据互斥：判据会拿同一张量比自己")
    if cfg.full_determinism and cfg.mode == "train":
        raise VerlConfigError(
            "训练模式不允许 full_determinism：组内采样会完全相同、GRPO 无学习信号（只用于诊断）")
    if cfg.mode == "diagnose" and cfg.full_determinism and not cfg.enforce_eager:
        raise VerlConfigError("诊断模式开 full_determinism 必须同时 enforce_eager=True（否则 vLLM 起不来）")
    if cfg.correction in DESIGNED_NOT_MAPPED:
        raise VerlConfigError(f"verl 后端的 {cfg.correction} 映射尚未核实，暂不开放")
    if cfg.correction not in SUPPORTED_CORRECTIONS:
        raise VerlConfigError(f"verl 后端不支持此修正: {cfg.correction!r}")
    if cfg.correction == "tis":
        if cfg.tis_lower != 0.0:
            raise VerlConfigError("verl 后端不支持此修正: TIS 下界只能为 0")
        if not (math.isfinite(cfg.tis_upper) and cfg.tis_upper > 1.0):
            raise VerlConfigError(f"TIS 上界必须 > 1，得到 {cfg.tis_upper}")
    if cfg.temperature != 1.0:
        raise VerlConfigError("verl 后端第一版采样温度固定为 1（判据口径）")
    if (cfg.top_p != 1.0 or cfg.top_k != -1) and not cfg.allow_sampling_shift:
        raise VerlConfigError(
            "非默认 top_p/top_k 会让推理 logprob 系统性偏移（S16 实测 top_p=0.9 偏 -0.032）；"
            "确需如此请显式 allow_sampling_shift")
    if cfg.groups_per_round <= 0 or cfg.samples_per_group <= 0 or cfg.total_rounds <= 0:
        raise VerlConfigError("groups_per_round / samples_per_group / total_rounds 必须为正")
    if cfg.n_gpus != 1:
        raise VerlConfigError("verl 后端第一版每岛只支持 1 张卡（V1/V2 范围）")
    if cfg.model_dtype not in ("fp32", "bf16"):
        raise VerlConfigError(f"model_dtype 必须是 fp32 或 bf16，得到 {cfg.model_dtype!r}")


def build_overrides(cfg: VerlRunConfig) -> list[str]:
    """Hydra overrides for ``verl/trainer/config/ppo_trainer.yaml`` (fork acad9875).

    The trainer mode is yeto's ``yeto_sync`` (``trainer.v1.trainer_mode``),
    registered by ``yeto.rl.adapters.verl.trainer``; ``trainer.total_training_steps``
    is an upper bound, yeto's island loop decides how many rounds run.
    """
    validate(cfg)
    o = [
        "trainer.use_v1=True",
        "trainer.v1.trainer_mode=yeto_sync",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        f"algorithm.rollout_correction.bypass_mode={cfg.bypass_mode}",
        f"data.train_files={cfg.train_file}",
        f"data.val_files={cfg.val_file}",
        f"data.train_batch_size={cfg.groups_per_round}",
        f"data.max_prompt_length={cfg.max_prompt_len}",
        f"data.max_response_length={cfg.max_response_len}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"data.seed={cfg.seed}",
        f"actor_rollout_ref.model.path={cfg.model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.model.lora_rank={cfg.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={cfg.lora_alpha}",
        f"actor_rollout_ref.model.target_modules={cfg.target_modules}",
        f"actor_rollout_ref.model.lora.merge={cfg.lora_merge}",
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={cfg.model_dtype}",
        f"actor_rollout_ref.actor.optim.lr={cfg.lr!r}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg.groups_per_round}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.actor.data_loader_seed={cfg.seed}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.n={cfg.samples_per_group}",
        f"actor_rollout_ref.rollout.temperature={cfg.temperature}",
        f"actor_rollout_ref.rollout.top_p={cfg.top_p}",
        f"actor_rollout_ref.rollout.top_k={cfg.top_k}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384",
        f"actor_rollout_ref.rollout.full_determinism={cfg.full_determinism}",
        f"actor_rollout_ref.rollout.enforce_eager={cfg.enforce_eager}",
        "trainer.critic_warmup=0",
        "trainer.logger=[console,file]",
        "trainer.project_name=yeto-verl",
        f"trainer.experiment_name={cfg.experiment_name}",
        f"trainer.default_local_dir={cfg.out_dir}/ckpt",
        f"trainer.n_gpus_per_node={cfg.n_gpus}",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.resume_mode=disable",
        f"trainer.total_training_steps={cfg.total_rounds}",
        "trainer.total_epochs=1000",
        "ray_kwargs.ray_init.runtime_env.py_executable=null",
    ]
    if cfg.correction == "tis":
        o += ["algorithm.rollout_correction.rollout_is=token",
              f"algorithm.rollout_correction.rollout_is_threshold={cfg.tis_upper}"]
    if cfg.reward_path:
        o += [f"reward.custom_reward_function.path={cfg.reward_path}",
              f"reward.custom_reward_function.name={cfg.reward_name}"]
    for key, value in sorted(cfg.chat_template_kwargs.items()):
        o.append(f"+data.apply_chat_template_kwargs.{key}={value}")
    return o


# Overrides whose value the island re-reads from the resolved verl config and
# asserts after hydra composition (initialisation assertions, see the review doc).
ASSERTED_KEYS = (
    "actor_rollout_ref.model.lora_rank",
    "actor_rollout_ref.model.lora_alpha",
    "actor_rollout_ref.rollout.temperature",
    "actor_rollout_ref.rollout.top_p",
    "actor_rollout_ref.rollout.top_k",
    "actor_rollout_ref.rollout.calculate_log_probs",
    "algorithm.rollout_correction.bypass_mode",
    "actor_rollout_ref.rollout.full_determinism",
    "actor_rollout_ref.actor.strategy",
)
