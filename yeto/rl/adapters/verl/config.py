"""yeto run options -> checked verl hydra overrides.

Rules enforced before anything starts (each has a CPU test):

* LoRA ``lora_alpha`` must equal ``rank`` (yeto canonical LoRA);
* the verl source commit must equal the pin;
* bypass mode (old log-probs = rollout log-probs) cannot be combined with any
  mismatch criterion (the criteria would compare a tensor with itself);
* ``full_determinism`` is refused in training mode, and in diagnostic mode it
  requires ``enforce_eager`` (vLLM's Inductor autotune refuses
  deterministic mode; deterministic sampling also makes the GRPO group
  identical, i.e. no learning signal);
* corrections: only TIS with lower bound 0 is mapped; IcePop is in the design
  but its verl mapping is not verified yet, so it is refused for now; every
  other correction is refused with "verl backend does not support this correction";
* sampling defaults are T=1, top_p=1, top_k=-1; non-default top_p/top_k need
  ``allow_sampling_shift`` (the criteria then see a systematic shift).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .pins import VERL_COMMIT
from .run_config_rules import AGENTIC_ADV_ESTIMATOR

MODES = ("train", "diagnose")
SUPPORTED_CORRECTIONS = ("none", "tis")
DESIGNED_NOT_MAPPED = ("icepop",)  # designed, mapping to verl not verified yet
# ``parameter_mode``: the island's trainable set. "lora" is the original
# one-card adapter island; "full" trains every parameter on
# ``n_gpus`` cards with FSDP2 data parallelism and FP32 masters
# (:mod:`.full_parameter`), alone (``--rl-single-island-no-sync``) or synced
# through the unified full-parameter syncer (:mod:`.full_parameter_stream`).
PARAMETER_MODES = ("lora", "full")
FULL_PARAMETER_MODE = "full"


class VerlConfigError(ValueError):
    """The run cannot be expressed on the verl backend as configured."""


@dataclass(frozen=True)
class VerlRunConfig:
    model_path: str
    train_file: str
    val_file: str
    out_dir: str
    parameter_mode: str = "lora"
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
    # vLLM context length; None = the model's max_position_embeddings (262144 for
    # Qwen3-4B-Instruct-2507, whose KV cache then does not fit next to the actor)
    max_model_len: int | None = None
    n_gpus: int = 1
    model_dtype: str = "fp32"
    reward_path: str = ""
    reward_name: str = "compute_score"
    chat_template_kwargs: dict = field(default_factory=dict)
    experiment_name: str = "yeto-verl"
    # Keep the parameters and the optimizer state in host
    # memory, the card holding only what a step computes with. Both default off,
    # as before. param_offload is what lets a card smaller than the NVIDIA
    # reference hold a large full-parameter island at all: without it verl puts
    # the whole model on the card before FSDP2 shards it, and an 8x64 GiB
    # Ascend node dies inside fully_shard for a 30B model (the sharded copy no
    # longer fits next to the unsharded one). The parameters stay FP32, so the
    # full-parameter boundary still reads masters (full_parameter.MASTER_DTYPE).
    param_offload: bool = False
    optimizer_offload: bool = False
    # FSDP2's own CPU offload (torch CPUOffloadPolicy). This is the key that
    # actually offloads an fsdp2 island: verl reads param_offload /
    # optimizer_offload only on the FSDP1 path, and forces both off when
    # offload_policy is set (fork acad9875,
    # workers/engine/fsdp/transformer_impl.py:459-463).
    offload_policy: bool = False
    # Activation recomputation, when the model's implementation
    # supports it (``island_entry.gradient_checkpointing_for``). The HF
    # NemotronH of the Ascend image's transformers 5.3 does not, and verl's
    # FSDP2 engine dies on it right after building the model.
    gradient_checkpointing: bool = True
    # Load the model and tokenizer with the checkpoint's own code
    # (``--trust-remote-code``, the launcher's flag the Miles islands already
    # take). Needed for an architecture whose support lives in the repository
    # rather than in the installed transformers: the official Ascend verl
    # image's transformers 5.3 has nemotron_h, but its NemotronHModel sets
    # supports_gradient_checkpointing = False, while the checkpoint's own
    # modeling_nemotron_h.py sets it True. Default off: an NVIDIA island is
    # unchanged, and remote code is only run when the launch asks for it.
    trust_remote_code: bool = False
    # The agentic island. The rows select yeto's
    # Harbor agent loop (``agent_name: yeto_harbor``), so the advantage estimator
    # must be ``yeto_grpo`` (``run_config_rules.AGENTIC_ADV_ESTIMATOR``; the loop
    # refuses any other one, ``harbor_agent_loop.check_estimator``) and verl must
    # read the loop's registry entry from ``agent_loop_config`` (a YAML path).
    agentic: bool = False
    agent_loop_config: str = ""
    # Agent-loop worker processes. One by default on the agentic path: every
    # worker builds its own records proxy on its own port, and only one port is
    # tunnelled to the sandboxes (``modal_runner.harbor_relay_tunnel``).
    agent_loop_workers: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def full_parameter(cfg: VerlRunConfig) -> bool:
    return cfg.parameter_mode == FULL_PARAMETER_MODE


def validate(cfg: VerlRunConfig, *, expected_commit: str = VERL_COMMIT) -> None:
    """Raise :class:`VerlConfigError` naming the first violated rule."""
    if cfg.parameter_mode not in PARAMETER_MODES:
        raise VerlConfigError(
            f"parameter_mode must be one of {PARAMETER_MODES}, got {cfg.parameter_mode!r}")
    if full_parameter(cfg):
        if cfg.lora_rank != 0:
            raise VerlConfigError("full-parameter mode requires lora_rank=0 (the full-parameter island has no adapter)")
        if cfg.model_dtype != "fp32":
            raise VerlConfigError(
                "full-parameter sync requires FP32 masters: model_dtype must be fp32 (see the full_parameter module)")
    elif cfg.lora_rank <= 0:
        raise VerlConfigError("LoRA mode requires lora_rank > 0 (use parameter_mode=full for full parameters)")
    if not full_parameter(cfg) and cfg.lora_alpha != cfg.lora_rank:
        raise VerlConfigError(
            f"lora_alpha ({cfg.lora_alpha}) must equal rank ({cfg.lora_rank}): yeto canonical LoRA has no scaling")
    if cfg.engine_commit != expected_commit:
        raise VerlConfigError(
            f"verl source commit {cfg.engine_commit} does not match the pinned {expected_commit}")
    if cfg.mode not in MODES:
        raise VerlConfigError(f"mode must be one of {MODES}, got {cfg.mode!r}")
    if cfg.bypass_mode and cfg.criteria_enabled:
        raise VerlConfigError("bypass mode (old=rollout) excludes the train/rollout mismatch criteria: they would compare a tensor with itself")
    if cfg.full_determinism and cfg.mode == "train":
        raise VerlConfigError(
            "full_determinism is not allowed in train mode: every sample in a group would be identical and GRPO gets no learning signal (diagnostics only)")
    if cfg.mode == "diagnose" and cfg.full_determinism and not cfg.enforce_eager:
        raise VerlConfigError("full_determinism in diagnose mode requires enforce_eager=True (vLLM does not start otherwise)")
    if cfg.correction in DESIGNED_NOT_MAPPED:
        raise VerlConfigError(f"the verl backend's {cfg.correction} mapping is not verified yet and is not available")
    if cfg.correction not in SUPPORTED_CORRECTIONS:
        raise VerlConfigError(f"verl backend does not support this correction: {cfg.correction!r}")
    if cfg.correction == "tis":
        if cfg.tis_lower != 0.0:
            raise VerlConfigError("verl backend does not support this correction: the TIS lower bound must be 0")
        if not (math.isfinite(cfg.tis_upper) and cfg.tis_upper > 1.0):
            raise VerlConfigError(f"TIS upper bound must be > 1, got {cfg.tis_upper}")
    if cfg.temperature != 1.0:
        raise VerlConfigError("the verl backend fixes the sampling temperature at 1 (the criteria assume it)")
    if (cfg.top_p != 1.0 or cfg.top_k != -1) and not cfg.allow_sampling_shift:
        raise VerlConfigError(
            "non-default top_p/top_k shift the rollout log-probs systematically (measured: top_p=0.9 "
            "shifts them by -0.032); set allow_sampling_shift explicitly if this is intended")
    if cfg.groups_per_round <= 0 or cfg.samples_per_group <= 0 or cfg.total_rounds <= 0:
        raise VerlConfigError("groups_per_round / samples_per_group / total_rounds must be positive")
    if cfg.n_gpus < 1:
        raise VerlConfigError(f"n_gpus must be >= 1 (cards of a single-node island), got {cfg.n_gpus}")
    if cfg.agentic:
        if not cfg.agent_loop_config:
            raise VerlConfigError("an agentic island needs agent_loop_config (the yeto_harbor registry YAML path)")
        if cfg.agent_loop_workers < 1:
            raise VerlConfigError(f"agent_loop_workers must be >= 1, got {cfg.agent_loop_workers}")
    elif cfg.agent_loop_config:
        raise VerlConfigError("agent_loop_config only applies to an agentic island")
    if cfg.model_dtype not in ("fp32", "bf16"):
        raise VerlConfigError(f"model_dtype must be fp32 or bf16, got {cfg.model_dtype!r}")


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
        f"algorithm.adv_estimator={AGENTIC_ADV_ESTIMATOR if cfg.agentic else 'grpo'}",
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
        f"data.trust_remote_code={cfg.trust_remote_code}",
        f"actor_rollout_ref.model.trust_remote_code={cfg.trust_remote_code}",
        f"actor_rollout_ref.model.path={cfg.model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={cfg.gradient_checkpointing}",
        *(["actor_rollout_ref.model.lora_rank=0"] if full_parameter(cfg) else [
            f"actor_rollout_ref.model.lora_rank={cfg.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={cfg.lora_alpha}",
            f"actor_rollout_ref.model.target_modules={cfg.target_modules}",
            f"actor_rollout_ref.model.lora.merge={cfg.lora_merge}",
        ]),
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={cfg.model_dtype}",
        f"actor_rollout_ref.actor.optim.lr={cfg.lr!r}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg.groups_per_round}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={cfg.param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={cfg.optimizer_offload}",
        f"actor_rollout_ref.actor.fsdp_config.offload_policy={cfg.offload_policy}",
        f"actor_rollout_ref.actor.data_loader_seed={cfg.seed}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg.gpu_memory_utilization}",
        *([f"actor_rollout_ref.rollout.max_model_len={cfg.max_model_len}"] if cfg.max_model_len else []),
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
    if cfg.agentic:  # the rows name the loop; verl needs its registry entry
        o += [f"actor_rollout_ref.rollout.agent.agent_loop_config_path={cfg.agent_loop_config}",
              f"actor_rollout_ref.rollout.agent.num_workers={cfg.agent_loop_workers}"]
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
    # A silently dropped offload_policy would only show up as an
    # out-of-memory failure deep into the first step, so assert it resolved.
    "actor_rollout_ref.actor.fsdp_config.offload_policy",
)
