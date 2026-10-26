"""verl Megatron engine island: shape, checks and Hydra overrides (S27 G-DC, V8).

The verl FSDP2 island (``config.build_overrides``) has no tensor, pipeline or
expert parallelism. verl's Megatron engine (``model_engine=megatron``, model
built by Megatron-Bridge) has all three, so a run that asks for any of them
runs on this engine. The engine is chosen by the parallel shape
(:func:`engine_for`), not by a separate switch: EP/TP/PP > 1 exist only here.

Shape of the island (the Miles shape of S27 D0, ``s27-ch-d0a``):

* trainer pool: ``trainer_nodes`` x ``trainer_gpus_per_node`` cards, Megatron
  TP x PP x EP (ETP 1, CP 1), the distributed optimizer, no offload (the
  trainer cards hold weights, gradients and the optimizer state all the time);
* rollout pool: its own ``rollout_nodes`` x ``rollout_gpus_per_node`` cards,
  vLLM servers of ``rollout_tp`` cards each (verl's "standalone" replicas);
* trainer mode ``yeto_separate`` (:mod:`.separate_trainer`): verl's v1
  ``separate_async`` trainer with ``hybrid_engine=False``,
  ``parameter_sync_step=1`` and no warm-up batch. The yeto driver submits
  the prompts of round r only after it published policy r, so every sample
  of round r comes from policy r (age 0, on-policy, as D0);
* weights reach the rollout servers through verl's NCCL checkpoint engine.

Rollout engine: vLLM. The fork's ``uv.lock`` declares the ``vllm`` and
``sglang`` extras as conflicting, so one image holds one of them; the yeto verl
image and its tests already pin vLLM 0.29.0, and the fork's own Nemotron Nano
Megatron example (``examples/grpo_trainer/run_nemotron_nano_v3_30b_a3b_megatron.sh``)
runs vLLM.

Expert dispatch: ``alltoall`` with permute fusion and grouped GEMM. DeepEP
(``flex`` dispatcher) is not in the fork's lock (no ``deep_ep`` package for
CUDA 13 / torch 2.13), so it is not offered here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import config as vconf
from . import separate
from .separate import TRAINER_MODE  # noqa: F401 - re-exported (S27 G-EK: separate mode moved to .separate)

ENGINE_FSDP2 = "fsdp2"
ENGINE_MEGATRON = "megatron"
ROLLOUT_ENGINES = ("vllm",)
DISPATCHERS = ("alltoall",)
ROUTER_REPLAY_MODES = ("disabled", "R3")
# Megatron-Core selective recompute modules (``TransformerConfig.recompute_modules``)
RECOMPUTE_MODULES = ("core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "shared_experts")


class MegatronShapeError(ValueError):
    """The verl Megatron island cannot run this shape."""


def sync_problems(args) -> list[str]:
    """S27 G-FG: what a Megatron island with the full-parameter outer sync cannot
    run ([] when it can). Read from launcher arguments and from island arguments
    alike (the attribute names the two share or that only one of them has).

    * pipeline parallelism: the HF boundary (``yeto.rl.megatron_hf_boundary``)
      exports through Megatron-Bridge, which handles PP, but a tied embedding
      over two stages and the PP broadcasts have no test; refused until they do;
    * peer recovery: the verl stream's donor files are FSDP2 rank files
      (``peer_recovery.VerlRankShards``); a Megatron island rejoins by catch-up
      (the syncer's broadcast of the current fragments) instead."""
    problems = []
    if int(getattr(args, "pipeline_parallel", 1) or 1) != 1:
        problems.append("--pipeline-parallel 1 (PP with the outer sync is not verified)")
    if (getattr(args, "rl_peer_recovery", False) or getattr(args, "rl_peer_recovery_store", None)
            or getattr(args, "rl_peer_recover", False)):
        problems.append("no peer recovery (the Megatron island rejoins by catch-up: --rl-island-state-source "
                        "catch-up)")
    return problems


def engine_for(*, tensor_parallel: int = 1, pipeline_parallel: int = 1, expert_parallel: int = 1) -> str:
    """The verl training engine of a run: Megatron when any model parallel size is
    above 1 (the FSDP2 island has none), FSDP2 otherwise."""
    if max(int(tensor_parallel or 1), int(pipeline_parallel or 1), int(expert_parallel or 1)) > 1:
        return ENGINE_MEGATRON
    return ENGINE_FSDP2


@dataclass(frozen=True)
class MegatronShape:
    trainer_nodes: int = 1
    trainer_gpus_per_node: int = 8
    rollout_nodes: int = 1
    rollout_gpus_per_node: int = 8
    rollout_tp: int = 2
    tp: int = 1
    pp: int = 1
    ep: int = 8
    etp: int = 1
    cp: int = 1
    # dynamic batching: tokens of one packed micro-batch per trainer card (Miles
    # --rl-max-tokens-per-gpu); also the old-log-prob recompute budget
    max_tokens_per_gpu: int = 8192
    recompute_modules: tuple[str, ...] = ("moe", "layernorm")
    dispatcher: str = "alltoall"
    router_replay: str = "disabled"
    rollout_engine: str = "vllm"
    gpu_memory_utilization: float = 0.8
    num_experts: int | None = None

    @property
    def trainer_gpus(self) -> int:
        return self.trainer_nodes * self.trainer_gpus_per_node

    @property
    def rollout_gpus(self) -> int:
        return self.rollout_nodes * self.rollout_gpus_per_node

    @property
    def data_parallel(self) -> int:
        return self.trainer_gpus // (self.tp * self.pp * self.cp)

    def pools(self) -> separate.SeparatePools:
        """The trainer and rollout pools (the engine-neutral part, :mod:`.separate`)."""
        return separate.SeparatePools(
            trainer_nodes=self.trainer_nodes, trainer_gpus_per_node=self.trainer_gpus_per_node,
            rollout_nodes=self.rollout_nodes, rollout_gpus_per_node=self.rollout_gpus_per_node,
            rollout_tp=self.rollout_tp, gpu_memory_utilization=self.gpu_memory_utilization)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["recompute_modules"] = list(self.recompute_modules)
        out.update(trainer_gpus=self.trainer_gpus, rollout_gpus=self.rollout_gpus,
                   data_parallel=self.data_parallel)
        return out


def validate(shape: MegatronShape) -> None:
    """Raise :class:`MegatronShapeError` naming the first violated rule."""
    s = shape
    for name in ("trainer_nodes", "trainer_gpus_per_node", "rollout_nodes", "rollout_gpus_per_node",
                 "rollout_tp", "tp", "pp", "ep", "etp", "cp", "max_tokens_per_gpu"):
        if int(getattr(s, name)) < 1:
            raise MegatronShapeError(f"{name} 必须 >= 1，得到 {getattr(s, name)}")
    if s.cp != 1:
        raise MegatronShapeError("verl Megatron 岛暂不接 CP>1：Mamba 打包的 seq_idx 补丁只在 CP=1 下核对过")
    if s.etp != 1:
        raise MegatronShapeError("verl Megatron 岛只接 ETP=1（专家不做张量并行）")
    model_parallel = s.tp * s.pp * s.cp
    if s.trainer_gpus % model_parallel:
        raise MegatronShapeError(f"TP*PP*CP={model_parallel} 不整除训练卡数 {s.trainer_gpus}")
    if s.tp > s.trainer_gpus_per_node:
        raise MegatronShapeError(f"TP={s.tp} 跨节点（每节点 {s.trainer_gpus_per_node} 卡）")
    # Megatron: EP x ETP divides the expert-side world of one pipeline stage
    if (s.trainer_gpus // s.pp) % (s.ep * s.etp):
        raise MegatronShapeError(f"EP*ETP={s.ep * s.etp} 不整除每个流水级的卡数 {s.trainer_gpus // s.pp}")
    if s.num_experts is not None and s.num_experts % s.ep:
        raise MegatronShapeError(f"EP={s.ep} 不整除专家数 {s.num_experts}")
    separate.validate_pools(s.pools(), MegatronShapeError)
    if s.rollout_engine not in ROLLOUT_ENGINES:
        raise MegatronShapeError(f"rollout 引擎只接 {ROLLOUT_ENGINES}（镜像只带 vLLM），得到 {s.rollout_engine!r}")
    if s.dispatcher not in DISPATCHERS:
        raise MegatronShapeError(
            f"专家分发只接 {DISPATCHERS}：DeepEP（flex）不在 verl 分叉的 uv.lock 里，得到 {s.dispatcher!r}")
    if s.router_replay not in ROUTER_REPLAY_MODES:
        raise MegatronShapeError(f"router_replay 只接 {ROUTER_REPLAY_MODES}，得到 {s.router_replay!r}")
    unknown = sorted(set(s.recompute_modules) - set(RECOMPUTE_MODULES))
    if unknown:
        raise MegatronShapeError(f"未知的 selective 重算模块 {unknown}（可选 {RECOMPUTE_MODULES}）")


def validate_run(cfg: vconf.VerlRunConfig) -> None:
    """The run-level rules of a Megatron island on top of ``config.validate``."""
    vconf.validate(cfg)
    if not vconf.full_parameter(cfg):
        raise MegatronShapeError("verl Megatron 岛只接全参数（parameter_mode=full）；LoRA 走 FSDP2 岛")


# FSDP2-island overrides that a Megatron island replaces or drops (key prefixes;
# everything else of ``config.build_overrides`` -- data, sampling, algorithm,
# correction, reward, trainer bookkeeping -- is kept as is).
_FSDP_ONLY_PREFIXES = (
    *separate.POOL_PREFIXES,
    "actor_rollout_ref.model.use_remove_padding=",
    "actor_rollout_ref.model.enable_gradient_checkpointing=",
    "actor_rollout_ref.actor.strategy=",
    "actor_rollout_ref.actor.fsdp_config.",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=",
)


def build_overrides(cfg: vconf.VerlRunConfig, shape: MegatronShape, *,
                    eps_clip: float | None = None, eps_clip_high: float | None = None) -> list[str]:
    """Hydra overrides of a verl Megatron island (``verl/trainer/config/ppo_trainer.yaml``
    with ``model_engine=megatron``, fork acad9875)."""
    validate_run(cfg)
    validate(shape)
    base = [o for o in vconf.build_overrides(cfg) if not o.startswith(_FSDP_ONLY_PREFIXES)]
    meg = "actor_rollout_ref.actor.megatron"
    tcfg = f"++{meg}.override_transformer_config"
    tokens = int(shape.max_tokens_per_gpu)
    out = [
        "model_engine=megatron",
        *base,
        # trainer mode, separate_async settings and both pools (engine-neutral, .separate)
        *separate.separate_overrides(shape.pools()),
        # Mamba SSM state cache in fp32 (the checkpoint's mamba_ssm_cache_dtype; NeMo-RL's
        # Nemotron vLLM setting), written out so the rollout never falls back to bf16
        "++actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype=float32",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={tokens}",
        "actor_rollout_ref.model.trust_remote_code=False",
        "actor_rollout_ref.model.use_fused_kernels=False",
        # trainer: Megatron, distributed optimizer, nothing offloaded (dynamic batching:
        # use_dynamic_bsz=True comes from the shared overrides)
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={tokens}",
        f"{meg}.tensor_model_parallel_size={shape.tp}",
        f"{meg}.pipeline_model_parallel_size={shape.pp}",
        f"{meg}.expert_model_parallel_size={shape.ep}",
        f"{meg}.expert_tensor_parallel_size={shape.etp}",
        f"{meg}.context_parallel_size={shape.cp}",
        f"{meg}.use_distributed_optimizer=True",
        f"{meg}.param_offload=False",
        f"{meg}.optimizer_offload=False",
        f"{meg}.use_mbridge=True",
        f"{meg}.dtype=bfloat16",
        f"{meg}.use_remove_padding=True",
        f"{meg}.router_replay.mode={shape.router_replay}",
        f"{tcfg}.recompute_granularity=selective",
        f"{tcfg}.recompute_modules=[{','.join(shape.recompute_modules)}]",
        f"{tcfg}.recompute_method=null",
        f"{tcfg}.recompute_num_layers=null",
        f"{tcfg}.moe_router_dtype=fp32",
        f"{tcfg}.moe_token_dispatcher_type={shape.dispatcher}",
        f"{tcfg}.moe_permute_fusion=True",
        f"{tcfg}.moe_grouped_gemm=True",
        # RL keeps the routing statistics fixed (as the Miles recipe, nemotron_h.py)
        f"{tcfg}.moe_router_bias_update_rate=0.0",
        f"{tcfg}.moe_aux_loss_coeff=0.0",
    ]
    if shape.router_replay == "R3":
        # R3 replays the experts vLLM routed (``enable_return_routed_experts``)
        out.append("actor_rollout_ref.rollout.enable_rollout_routing_replay=True")
    if eps_clip is not None:
        out.append(f"actor_rollout_ref.actor.clip_ratio_low={float(eps_clip)!r}")
    if eps_clip_high is not None:
        out.append(f"actor_rollout_ref.actor.clip_ratio_high={float(eps_clip_high)!r}")
    return out


# The island re-reads these after Hydra composition and asserts the values
# (``trainer.check_asserted``), on top of ``config.ASSERTED_KEYS``.
ASSERTED_KEYS = (
    "actor_rollout_ref.model.lora_rank",
    "actor_rollout_ref.rollout.temperature",
    "actor_rollout_ref.rollout.top_p",
    "actor_rollout_ref.rollout.top_k",
    "actor_rollout_ref.rollout.calculate_log_probs",
    "algorithm.rollout_correction.bypass_mode",
    *separate.ASSERTED_KEYS,
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size",
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
    "actor_rollout_ref.actor.megatron.use_distributed_optimizer",
    "actor_rollout_ref.actor.megatron.param_offload",
    "actor_rollout_ref.actor.megatron.optimizer_offload",
)
