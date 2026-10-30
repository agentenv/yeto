"""verl Megatron engine island: shape, checks and Hydra overrides (S27 G-DC, V8).

The verl FSDP2 island (``config.build_overrides``) has no tensor, pipeline or
expert parallelism. verl's Megatron engine (``model_engine=megatron``, model
built by Megatron-Bridge) has all three, so a run that asks for any of them
runs on this engine. The engine is chosen by the parallel shape
(:func:`engine_for`), not by a separate switch: EP/TP/PP > 1 exist only here.

Shape of the island (the Miles shape of S27 D0, ``s27-ch-d0a``):

* trainer pool: ``trainer_nodes`` x ``trainer_gpus_per_node`` cards, Megatron
  TP x PP x EP x CP (ETP 1), the distributed optimizer, no offload (the
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

from yeto.rl import megatron_grad_checks as grad_checks

from . import config as vconf
from . import separate
from .separate import TRAINER_MODE  # noqa: F401 - re-exported (S27 G-EK: separate mode moved to .separate)

ENGINE_FSDP2 = "fsdp2"
ENGINE_MEGATRON = "megatron"
ROLLOUT_ENGINES = ("vllm",)
DISPATCHERS = ("alltoall",)
ROUTER_REPLAY_MODES = ("disabled", "R3")
# S27 G-FN: keys a model's own config.json uses to declare multi-token prediction.
# A hybrid (Mamba) model that declares none of them wants the MTP branch closed, and
# has to say so explicitly -- see MTP_OFF_PATTERN.
MTP_CONFIG_KEYS = ("mtp_num_layers", "num_nextn_predict_layers", "mtp_hybrid_override_pattern")
# Megatron-Bridge enters its MTP branch whenever mtp_hybrid_override_pattern is truthy
# and then multiplies the pattern by mtp_num_layers, which verl spells as None for
# "off" -- so the build dies with TypeError: can't multiply sequence by non-int of type
# NoneType. transformers lights the fuse by defaulting NemotronHConfig.
# mtp_hybrid_override_pattern to "*E" for a checkpoint whose config.json has no MTP key
# at all. No single side is wrong; the three together are (G-FJ2 traced this,
# SESSION27/VERL-ASCEND-MEGATRON.md conclusion C item 2, and it is not Ascend-specific:
# the same versions on CUDA break the same way). Writing the pattern as null keeps the
# branch closed, which is what the checkpoint asked for.
MTP_OFF_PATTERN = "null"
# Megatron-Core selective recompute modules (``TransformerConfig.recompute_modules``)
RECOMPUTE_MODULES = ("core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "shared_experts")


# S27 G-FN: Megatron's own optimizer-memory keys, written when the engine-neutral
# --rl-optimizer-offload-fraction asks for host memory. verl's own
# engine.optimizer_offload is a DIFFERENT switch and does not set Megatron's CPU
# offload: G-FJ2 read it back off the resolved OptimizerConfig (optimizer_cpu_offload
# stayed False) and the per-card peak did not move
# (SESSION27/VERL-ASCEND-MEGATRON.md conclusion B). So the fraction writes the real
# keys and leaves verl's key alone, which is what that report recommended.
# Measured on 910B3 with the whole state in host memory: 8 cards at TP=1/PP=2/EP=4
# peak at 41.97 GB per card (VERL-MEGATRON-PP.md section 7.9); the state it moves off
# the card is about 29 GB per rank there.
def bf16_moment_overrides() -> dict[str, str]:
    """S28 G-RCP: the precision-aware optimizer on the card with bf16 Adam moments and an
    FP32 master (``store_param_remainders`` off, so the full-parameter boundary can read
    the master: ``yeto.rl.megatron_masters``). Main grads stay FP32. Measured need: the
    PP=2 Nano island on 64 GB 910B3 ran out of memory by 20 MiB in FusedAdam's first
    moment allocation (VERL-MEGATRON-PP section 7.13.6); bf16 moments halve them."""
    return {
        "optim.use_precision_aware_optimizer": "True",
        "optim.exp_avg_dtype": "bf16",
        "optim.exp_avg_sq_dtype": "bf16",
        "++optim.override_optimizer_config.store_param_remainders": "False",
    }


def optimizer_memory_overrides(fraction: float, bf16_moments: bool = False) -> dict[str, str]:
    """Megatron optimizer keys for an offload fraction in (0, 1]; {} at 0.

    The offload runs Megatron's precision-aware path, so the Adam moments and the
    gradient-accumulation buffer go to bf16 with it (the same pairing Miles documents
    for this flag, ``adapters.miles.config``, and the one the Ascend SFT recipe runs)."""
    if bf16_moments and fraction:
        raise MegatronShapeError("bf16 moments on the card and optimizer_offload_fraction are exclusive "
                                 "(both pick the optimizer: on-card FusedAdam or the host HybridDeviceOptimizer)")
    if bf16_moments:
        return bf16_moment_overrides()
    if not fraction:
        return {}
    return {
        "optim.use_precision_aware_optimizer": "True",
        "optim.main_grads_dtype": "bf16",
        "optim.exp_avg_dtype": "bf16",
        "optim.exp_avg_sq_dtype": "bf16",
        "++optim.override_optimizer_config.optimizer_cpu_offload": "True",
        "++optim.override_optimizer_config.optimizer_offload_fraction": repr(float(fraction)),
        "++optim.override_optimizer_config.use_torch_optimizer_for_cpu_offload": "True",
    }



class MegatronShapeError(ValueError):
    """The verl Megatron island cannot run this shape."""


# S28 G-CPW: Mamba layers in verl's THD packing. verl packs several samples into one
# micro-batch, and a Mamba layer restarts its conv and SSM state at a sample boundary
# only when ``PackedSeqParams`` carries ``seq_idx``. verl never sets it; the fifth
# patch of ``patch_verl.py`` (``--profile megatron``) adds ``_yeto_seq_idx_args`` to
# ``verl/models/mcore/util.py``, which passes the pack's ``total_tokens`` so Megatron
# derives it. That patch lives in this repository, but whether it is in effect is a
# property of the verl tree the island actually imports: npu1's tree never got it
# (S28 G-CPW, 10-27 03:1xZ: util.py there is the unpatched acad9875 file), so the
# island probes the imported module itself (:func:`probe_seq_idx`) instead of trusting
# the repository. The S27 helper passed ``total_tokens`` at CP=1 only; Megatron-Core's
# Mamba CP all-to-alls each rank back to the whole sequence, so the right ``seq_idx``
# is the global pack's at every CP size (S28 CP2, report/CP2-PLAN.md section 2.2),
# which the S28 helper passes. The probe asks the helper at the island's own CP.
SEQ_IDX_MODULE = "verl.models.mcore.util"
SEQ_IDX_HELPER = "_yeto_seq_idx_args"
SEQ_IDX_REMEDY = ("apply patch_verl.py --profile megatron (fifth patch, seq_idx; CP>1 needs the S28 "
                  "version that passes total_tokens at every CP size) to the verl tree the island imports, "
                  "then restart Ray")
# A pack of two samples (3 and 5 padded tokens) that the probe hands to the helper
_PROBE_SEQLENS = (3, 5)


@dataclass(frozen=True)
class SeqIdxPatch:
    """What verl's THD packing in one process does for a Mamba layer at CP ``cp``.

    ``ready``: the helper gives ``PackedSeqParams`` the whole pack's ``total_tokens``
    at that CP size (so Megatron derives ``seq_idx``); ``error`` names what is missing."""

    source: str | None
    cp: int
    ready: bool
    error: str | None = None


def probe_seq_idx(cp: int = 1, module=None) -> SeqIdxPatch:
    """Probe the seq_idx patch in ``module`` (default: the imported ``SEQ_IDX_MODULE``).

    Behavioural, not textual: the helper is called on a two-sample pack at ``cp`` and
    must return that pack's ``total_tokens``. That also proves this Megatron-Core's
    ``PackedSeqParams`` takes the field (the helper returns ``{}`` otherwise)."""
    cp = int(cp)
    if module is None:
        import importlib

        try:
            module = importlib.import_module(SEQ_IDX_MODULE)
        except Exception as error:  # noqa: BLE001 - reported, the caller refuses
            return SeqIdxPatch(None, cp, False, f"cannot import {SEQ_IDX_MODULE}: {error!r}")
    source = getattr(module, "__file__", None) or getattr(module, "__name__", None)
    helper = getattr(module, SEQ_IDX_HELPER, None)
    if not callable(helper):
        return SeqIdxPatch(source, cp, False, f"{source} has no {SEQ_IDX_HELPER} (unpatched verl)")
    pack, want = list(_PROBE_SEQLENS), sum(_PROBE_SEQLENS)
    try:
        got = helper(cp, pack)
    except Exception as error:  # noqa: BLE001 - a changed signature is reported, not raised
        return SeqIdxPatch(source, cp, False, f"{SEQ_IDX_HELPER}({cp}, {pack}) raised {error!r}")
    if isinstance(got, dict) and got.get("total_tokens") == want:
        return SeqIdxPatch(source, cp, True)
    why = ("this Megatron-Core's PackedSeqParams may lack total_tokens" if cp == 1 else
           "the helper passes total_tokens at CP=1 only (the S27 version)")
    return SeqIdxPatch(source, cp, False, f"{SEQ_IDX_HELPER}({cp}, {pack}) returned {got!r}, "
                                          f"not total_tokens={want}: {why}")


def seq_idx_problem(*, mamba: bool, cp: int, patch: SeqIdxPatch | None) -> str | None:
    """Why a model cannot train on this verl tree at context parallelism ``cp``
    (None when it can). Only a model with Mamba layers needs ``seq_idx``."""
    if not mamba:
        return None
    if patch is None:
        return "含 Mamba 层的模型：没有查过实际运行的 verl 是否带 seq_idx 补丁（MegatronShape.seq_idx 为空）"
    if int(patch.cp) != int(cp):
        return f"含 Mamba 层的模型：seq_idx 补丁是按 CP={patch.cp} 查的，本岛是 CP={int(cp)}"
    if not patch.ready:
        return (f"含 Mamba 层的模型在 CP={int(cp)} 下需要 seq_idx 补丁，实际运行的 verl 里它没有生效："
                f"{patch.error}。不补的话一个 micro-batch 里的多条样本会串 Mamba 状态。补法：{SEQ_IDX_REMEDY}")
    return None


def has_mamba(model_config) -> bool:
    """The model's ``config.json`` has Mamba layers (Nemotron-H ``hybrid_override_pattern``
    or ``layers_block_type``; :func:`yeto.rl.hybrid_params.hybrid_pattern`)."""
    from yeto.rl import hybrid_params

    try:
        return "M" in hybrid_params.hybrid_pattern(model_config)
    except hybrid_params.HybridParamError:
        return False


def runtime_problems(shape: dict, reports: list[dict]) -> list[str]:
    """What the trainer workers really run, against the plan's shape (S28 G-CPW).

    ``reports`` come from ``megatron_ports.remote_megatron_runtime``: the Megatron
    parallel world sizes of each rank and its own probe of the seq_idx patch. Writing
    ``context_parallel_size`` into the overrides is not proof that Megatron built CP
    groups of that size, and the driver's verl tree is not proof of the workers'."""
    problems = []
    for report in sorted(reports, key=lambda row: int(row.get("rank", 0))):
        rank = report.get("rank")
        for name in ("tp", "pp", "ep", "cp"):
            got = (report.get("sizes") or {}).get(name)
            if got is None or int(got) != int(shape[name]):
                problems.append(f"rank {rank}: {name} world size {got}, the plan has {shape[name]}")
        seq_idx = report.get("seq_idx")
        problem = seq_idx_problem(mamba=bool(shape.get("mamba")), cp=int(shape["cp"]),
                                  patch=SeqIdxPatch(**seq_idx) if seq_idx else None)
        if problem:
            problems.append(f"rank {rank}: {problem}")
    return problems


def sync_problems(args) -> list[str]:
    """S27 G-FG: what a Megatron island with the full-parameter outer sync cannot
    run ([] when it can). Read from launcher arguments and from island arguments
    alike (the attribute names the two share or that only one of them has).

    * peer recovery: the verl stream's donor files are FSDP2 rank files
      (``peer_recovery.VerlRankShards``); a Megatron island rejoins by catch-up
      (the syncer's broadcast of the current fragments) instead.

    S27 G-FL: pipeline parallelism (PP > 1) is **gated**, not unimplemented. The HF
    boundary handles PP -- Bridge's conversion tasks span every stage and broadcast
    over the PP group, and the boundary writes the tied embedding's second copy on
    the last stage itself (``megatron_hf_boundary.tied_output_copies``), all of it
    verified against real Megatron-Core and real Megatron-Bridge on gloo. On a GPU
    island it does not start: in run ``s27-fl-1`` the trainer ranks of pipeline
    stage 1 crashed on both islands in the first sharded export (``export_hf``),
    **after** the manifest step finished on both ranks (226 conversion tasks each,
    0 refused); the receiver died first, then the owner. The earlier attribution to
    the manifest step was withdrawn in section 7.4. On Ascend the same path runs:
    HCCL PP2 passes all three collectives (section 7.7). The gate is ours and
    temporary: it stops a shape that cannot start from taking other people's cards,
    and it names the cause instead of leaving a segfault with no Python frame.
    ``--rl-allow-pipeline-parallel-sync`` lifts it for whoever retries. Details:
    ``SESSION27/VERL-MEGATRON-PP.md`` sections 7.4 and 7.7."""
    problems = []
    if int(getattr(args, "pipeline_parallel", 1) or 1) != 1 and not getattr(
            args, "rl_allow_pipeline_parallel_sync", False):
        problems.append(
            "--pipeline-parallel 1 (PP > 1 does not start on a GPU island: the stage-1 trainer rank "
            "segfaults in the first sharded export after the manifest step, run s27-fl-1; the HF "
            "boundary itself is verified on CPU and on Ascend HCCL, SESSION27/VERL-MEGATRON-PP.md "
            "sections 7.4 and 7.7. --rl-allow-pipeline-parallel-sync retries it anyway)")
    if (getattr(args, "rl_peer_recovery", False) or getattr(args, "rl_peer_recovery_store", None)
            or getattr(args, "rl_peer_recover", False)):
        problems.append("no peer recovery (the Megatron island rejoins by catch-up: --rl-island-state-source "
                        "catch-up)")
    return problems


def engine_for(*, tensor_parallel: int = 1, pipeline_parallel: int = 1, expert_parallel: int = 1,
               context_parallel: int = 1) -> str:
    """The verl training engine of a run: Megatron when any model parallel size is
    above 1 (the FSDP2 island has none), FSDP2 otherwise. S28 G-CPW: context
    parallelism counts, so ``--context-parallel 2`` alone does not land on FSDP2."""
    sizes = (tensor_parallel, pipeline_parallel, expert_parallel, context_parallel)
    if max(int(size or 1) for size in sizes) > 1:
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
    # S27 G-FL: Megatron's per-bucket non-finite gradient check
    # (yeto.rl.megatron_grad_checks); off only through --rl-no-grad-nan-check
    grad_nan_check: bool = True
    # S28 G-120: Megatron's weight-gradient accumulation fusion calls Apex's
    # fused_weight_gradient_mlp_cuda, which does not exist on Ascend (no Apex there).
    # Megatron's own default is True, so the key has to be written: a plugin-stack
    # island died in the linear backward with "NameError: name
    # 'fused_weight_gradient_mlp_cuda' is not defined" (tensor_parallel/layers.py:602,
    # 120B run mfu-a1d on npu1). False is also what the recipe runs (args.json), and
    # it matches yeto's neutral default (engine/run_config.py) and the Miles adapter's
    # --no-gradient-accumulation-fusion.
    gradient_accumulation_fusion: bool = False
    # S27 G-FN (B3, ruled by the main agent): False on a buffer trainer island
    # (--rl-sample-role trainer). It has no rollout engine, so there is nothing to
    # keep apart from the training cards: every card the island's Ray reports is a
    # training card and no rollout pool is planned or validated. The fixed partition
    # exists to separate training cards from rollout cards, which is a requirement of
    # verl's separate mode, not of the Megatron engine.
    rollout_pool: bool = True
    # S27 G-FN: host share of the Megatron optimizer state
    # (--rl-optimizer-offload-fraction); 0 keeps all of it on the card
    optimizer_offload_fraction: float = 0.0
    # S28 G-RCP: --rl-megatron-bf16-moments (bf16_moment_overrides)
    bf16_moments: bool = False
    # S27 G-FN: write mtp_hybrid_override_pattern=null (MTP_OFF_PATTERN). True for a
    # hybrid model whose config.json declares no MTP layers at all.
    close_mtp_pattern: bool = False
    # S28 G-120: uneven pipeline stages (Megatron num_layers_in_first/last_pipeline_stage;
    # the hybrid model takes them when its pattern has no '|'). 120B NemotronH has 88
    # layers, which PP=3 cannot split evenly; the Ascend SFT recipe runs 29/29/30.
    first_stage_layers: int | None = None
    last_stage_layers: int | None = None
    # S28 G-120: full-layer recompute of this many layers per stage (uniform); None
    # keeps the selective recompute of recompute_modules
    recompute_full_layers: int | None = None
    # S28 G-120C: verl's fused log-prob kernels (actor_rollout_ref.model.use_fused_kernels).
    # They replace the materialised [tokens, vocab] logits and their gradient with a
    # fused linear-cross-entropy, which on this model is 131 072 of the 273 984 bytes
    # of activation per token (47.8%), so they decide how long one sequence can be.
    # The engine used to write False with no comment, which only restated verl's own
    # default (trainer/config/model/hf_model.yaml), so nothing recorded a decision.
    # It stays False by default, and this field exists so that turning it on is an
    # explicit, asserted choice rather than an unwritten default. Measured blockers on
    # the Ascend 120B path are in infra-drafts/SESSION28/report/FCE-PLAN.md: verl's
    # fused forward only patches a GPTModel, while NemotronH loads as a MambaModel, and
    # the Triton kernel asserts .is_cuda and calls torch.cuda.nvtx.
    fused_logprob_kernels: bool = False
    # S28 G-CPW: the model has Mamba layers, and what the island's verl tree does for
    # their seq_idx at this shape's CP (probe_seq_idx; None = not probed). validate
    # refuses a Mamba model whose verl does not generate it.
    mamba: bool = False
    seq_idx: SeqIdxPatch | None = None
    # S28 G-120C: whether verl's old-log-prob pass computes the token entropy. verl hard
    # codes calculate_entropy=True there (trainer/ppo/ray_trainer.py:1300), but the value
    # only feeds the "actor/entropy" metric (experimental/separation/ray_trainer.py:522-536
    # pops it right after) and never the loss, and computing it is the old-log-prob peak:
    # a clone of the [tokens, vocab/TP] logits plus fp32 softmax products. Run g120c-rl1
    # (CP2, 80.7k tokens per CP rank) died there allocating 9.85 GiB. False turns it off
    # through verl_megatron_patches; "actor/entropy" then reads 0.0 (see that module).
    old_logprob_entropy: bool = True

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
        """The trainer and rollout pools (the engine-neutral part, :mod:`.separate`).

        Refused without a rollout pool (S27 G-FN): a buffer trainer island has no
        rollout cards, so a caller that asks for the two pools is asking for a
        shape this island does not have."""
        if not self.rollout_pool:
            raise MegatronShapeError(
                "verl Megatron 缓冲训练岛没有 rollout 池：全部卡都是训练卡（--rl-sample-role trainer）")
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
    names = ["trainer_nodes", "trainer_gpus_per_node", "tp", "pp", "ep", "etp", "cp", "max_tokens_per_gpu"]
    if s.rollout_pool:  # S27 G-FN: a buffer trainer island plans no rollout pool
        names += ["rollout_nodes", "rollout_gpus_per_node", "rollout_tp"]
    for name in names:
        if int(getattr(s, name)) < 1:
            raise MegatronShapeError(f"{name} 必须 >= 1，得到 {getattr(s, name)}")
    # S28 G-CPW: was a refusal of every CP>1. A Mamba model now needs the seq_idx patch
    # in effect in the verl it imports, at CP=1 and, CP-aware, at CP>1; a model without
    # Mamba layers needs no seq_idx.
    problem = seq_idx_problem(mamba=s.mamba, cp=s.cp, patch=s.seq_idx)
    if problem:
        raise MegatronShapeError(problem)
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
    if s.rollout_pool:
        separate.validate_pools(s.pools(), MegatronShapeError)
    elif not 0.0 < float(s.gpu_memory_utilization) < 1.0:
        raise MegatronShapeError(f"gpu_memory_utilization 必须在 (0, 1)，得到 {s.gpu_memory_utilization}")
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
    for name in ("first_stage_layers", "last_stage_layers", "recompute_full_layers"):
        value = getattr(s, name)
        if value is not None and int(value) < 1:
            raise MegatronShapeError(f"{name} 必须 >= 1，得到 {value}")
    if (s.first_stage_layers is not None or s.last_stage_layers is not None) and s.pp < 2:
        raise MegatronShapeError("first/last_stage_layers 只用于 PP >= 2")
    if not 0.0 <= float(s.optimizer_offload_fraction) <= 1.0:
        raise MegatronShapeError("optimizer_offload_fraction 必须在 [0, 1]，"
                                 f"得到 {s.optimizer_offload_fraction}")
    if s.bf16_moments and float(s.optimizer_offload_fraction) > 0:
        raise MegatronShapeError("bf16_moments 与 optimizer_offload_fraction > 0 互斥"
                                 "（两者各自决定优化器：卡上 FusedAdam 或主机 HybridDeviceOptimizer）")


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
    # S27 G-EQ2: the FSDP2 island maps --trust-remote-code; a Megatron island
    # pins it False below (its bridge needs native Megatron support anyway)
    "actor_rollout_ref.model.trust_remote_code=",
    "actor_rollout_ref.actor.strategy=",
    "actor_rollout_ref.actor.fsdp_config.",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=",
)


def check_token_budget(cfg: vconf.VerlRunConfig, shape: MegatronShape) -> None:
    """One whole row must fit in one micro-batch (S28 G-CPW).

    verl's dynamic batching packs at most ``max_token_len_per_gpu x CP`` tokens into a
    micro-batch (``workers/engine/utils.py`` ``prepare_micro_batches``: the Megatron
    engine sets ``sp_size`` to ``context_parallel_size``) and asserts that the longest
    sequence fits (``utils/seqlen_balancing.py`` ``rearrange_micro_batches``). That
    assertion fires in the first step, after the model has loaded; this is the same
    rule at plan time, and the rule of the neutral ``run_config._train_memory``
    (``--rl-max-tokens-per-gpu x CP >= --seq-len``)."""
    row = int(cfg.max_prompt_len) + int(cfg.max_response_len)
    budget = int(shape.max_tokens_per_gpu) * int(shape.cp)
    if budget < row:
        raise MegatronShapeError(f"max_tokens_per_gpu {shape.max_tokens_per_gpu} x CP {shape.cp} = {budget} "
                                 f"放不下一行（max_prompt_len + max_response_len = {row}）")


def recompute_overrides(tcfg: str, shape: MegatronShape) -> list[str]:
    """Megatron recompute keys: selective over ``recompute_modules`` (default), or
    full-layer uniform recompute of ``recompute_full_layers`` layers (S28 G-120)."""
    if shape.recompute_full_layers is None:
        return [f"{tcfg}.recompute_granularity=selective",
                f"{tcfg}.recompute_modules=[{','.join(shape.recompute_modules)}]",
                f"{tcfg}.recompute_method=null",
                f"{tcfg}.recompute_num_layers=null"]
    return [f"{tcfg}.recompute_granularity=full",
            f"{tcfg}.recompute_method=uniform",
            f"{tcfg}.recompute_num_layers={int(shape.recompute_full_layers)}"]


def build_overrides(cfg: vconf.VerlRunConfig, shape: MegatronShape, *,
                    eps_clip: float | None = None, eps_clip_high: float | None = None) -> list[str]:
    """Hydra overrides of a verl Megatron island (``verl/trainer/config/ppo_trainer.yaml``
    with ``model_engine=megatron``, fork acad9875).

    S27 G-FN (B3, B6): without a rollout pool (``shape.rollout_pool`` False, a buffer
    trainer island) the separate-mode block is left out -- verl's separate mode is
    what needs the two pools. The island then runs the fully_async config tree
    (``fully_async_ppo_trainer.yaml``, whose ``defaults`` include ``ppo_trainer``, so
    every key here still resolves), and ``fully_async_round.fully_async_run_overrides``
    writes ``trainer.n_gpus_per_node`` / ``trainer.nnodes`` from the island's own cards
    (``buffer_trainer_shape``) plus ``hybrid_engine=False`` and the nccl checkpoint
    engine backend."""
    validate_run(cfg)
    validate(shape)
    check_token_budget(cfg, shape)
    base = [o for o in vconf.build_overrides(cfg) if not o.startswith(_FSDP_ONLY_PREFIXES)]
    meg = "actor_rollout_ref.actor.megatron"
    tcfg = f"++{meg}.override_transformer_config"
    tokens = int(shape.max_tokens_per_gpu)
    out = [
        "model_engine=megatron",
        *base,
        # trainer mode, separate_async settings and both pools (engine-neutral, .separate)
        *(separate.separate_overrides(shape.pools()) if shape.rollout_pool else []),
        # Mamba SSM state cache in fp32 (the checkpoint's mamba_ssm_cache_dtype; NeMo-RL's
        # Nemotron vLLM setting), written out so the rollout never falls back to bf16
        "++actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype=float32",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={tokens}",
        "actor_rollout_ref.model.trust_remote_code=False",
        f"actor_rollout_ref.model.use_fused_kernels={shape.fused_logprob_kernels}",
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
        *recompute_overrides(tcfg, shape),
        f"{tcfg}.moe_router_dtype=fp32",
        f"{tcfg}.moe_token_dispatcher_type={shape.dispatcher}",
        f"{tcfg}.moe_permute_fusion=True",
        f"{tcfg}.moe_grouped_gemm=True",
        # RL keeps the routing statistics fixed (as the Miles recipe, nemotron_h.py)
        f"{tcfg}.moe_router_bias_update_rate=0.0",
        f"{tcfg}.moe_aux_loss_coeff=0.0",
        # S28 G-120: Megatron's weight-gradient accumulation fusion calls Apex's
        # fused_weight_gradient_mlp_cuda, a CUDA extension that does not exist on Ascend
        # (Apex is not installed there at all). Megatron's own default is True, and this
        # engine never wrote the key, so a plugin-stack island died in the linear backward
        # with "NameError: name 'fused_weight_gradient_mlp_cuda' is not defined"
        # (tensor_parallel/layers.py:602; 120B run mfu-a1d on npu1, 24 cards). The recipe
        # sets gradient_accumulation_fusion False as well (args.json), so this is the
        # recipe-aligned value, not a workaround. yeto's neutral default is already False
        # (engine/run_config.py) and the Miles adapter passes
        # --no-gradient-accumulation-fusion; this is the verl side of the same switch.
        f"{tcfg}.gradient_accumulation_fusion={shape.gradient_accumulation_fusion}",
        # S28 G-120C: read by verl_megatron_patches in the trainer process (a yeto-owned
        # top-level section: verl's dataclass-backed sections refuse unknown keys)
        f"+{OLD_LOGPROB_ENTROPY_KEY}={shape.old_logprob_entropy}",
    ]
    if shape.first_stage_layers is not None:  # S28 G-120
        out.append(f"{tcfg}.num_layers_in_first_pipeline_stage={int(shape.first_stage_layers)}")
    if shape.last_stage_layers is not None:
        out.append(f"{tcfg}.num_layers_in_last_pipeline_stage={int(shape.last_stage_layers)}")
    if shape.close_mtp_pattern:  # S27 G-FN: MTP_OFF_PATTERN
        out.append(f"{tcfg}.mtp_hybrid_override_pattern={MTP_OFF_PATTERN}")
    actor = "actor_rollout_ref.actor"
    # S27 G-FN (B13): the Megatron engine builds its LR scheduler inside init_model,
    # and Megatron asserts lr_decay_steps > 0 there. verl derives that from
    # optim.total_training_steps, whose default is -1, and both the synchronous and the
    # fully_async trainer only fill it in AFTER the workers exist
    # (ray_trainer.py:447, fully_async_trainer.set_total_train_steps). The FSDP2 engine
    # does not build that scheduler, so only a Megatron island trips it; run fn-6 on
    # npu1 died there in every worker. The island knows the count up front, so it writes
    # it: one yeto round is one parameter version. A trainer that recomputes it later
    # still wins, which is why this is not an asserted key. The LR itself is constant
    # (--rl-lr-schedule constant, lr_decay_style=constant), so the value only has to be
    # positive for the scheduler to build.
    out.append(f"{actor}.optim.total_training_steps={max(1, int(cfg.total_rounds))}")
    # S27 G-FL: verl's DDP config leaves check_for_nan_in_grad at Megatron's False, and
    # with it off a NaN gradient is scaled into every parameter instead of raising
    # (yeto.rl.megatron_grad_checks). The island re-reads the flag off the DDP wrappers.
    for field, value in grad_checks.ddp_overrides(check_for_nan_in_grad=shape.grad_nan_check).items():
        out.append(f"++{meg}.override_ddp_config.{field}={value}")
    # S27 G-FN: the optimizer-state memory keys, under the actor's optimizer config
    # ("optim.x" becomes actor_rollout_ref.actor.optim.x; a "++" prefix
    # stays in front of the whole key, as Hydra wants it)
    for key, value in optimizer_memory_overrides(float(shape.optimizer_offload_fraction),
                                                    bool(shape.bf16_moments)).items():
        path = key.removeprefix("++")
        out.append(f"{'++' if path != key else ''}{actor}.{path}={value}")
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
# S28 G-120C: the yeto-owned config key the trainer's run-time patch reads
OLD_LOGPROB_ENTROPY_KEY = "yeto_trainer.old_logprob_entropy"

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
    # S28 G-CPW: written without "++", so the plan's want is the value and a resolved
    # config that disagrees fails the island (trainer.check_asserted). The workers'
    # Megatron CP groups are checked on top (runtime_problems).
    "actor_rollout_ref.actor.megatron.context_parallel_size",
    "actor_rollout_ref.actor.megatron.use_distributed_optimizer",
    "actor_rollout_ref.actor.megatron.param_offload",
    "actor_rollout_ref.actor.megatron.optimizer_offload",
    # S28 G-120: assert Megatron really received it. Writing the override is not proof
    # that the consumer took it: TORCHDYNAMO_DISABLE was exported on the driver and
    # never reached the Ray workers (run mfu-a1c), and this key's own default (True)
    # had been silently in force because the engine never wrote it (run mfu-a1d).
    "actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion",
    # S28 G-120C: the fused log-prob kernels decide whether the [tokens, vocab] logits
    # are materialised, which is 47.8% of this model's activation per token. Assert it
    # for the same reason as the key above: the engine wrote it for months without
    # anything checking that verl took the value.
    "actor_rollout_ref.model.use_fused_kernels",
    # S28 G-120C: written on every island, so want is never None -- a real assertion
    OLD_LOGPROB_ENTROPY_KEY,
)

# Keys of ``separate.ASSERTED_KEYS`` that only verl's separate mode writes. A buffer
# trainer island plans no rollout pool, so ``build_overrides`` does not write them and
# they are not asserted (S27 G-FN). The other two keys of that tuple -- hybrid_engine
# and the rollout checkpoint engine backend -- the fully_async overrides still write.
_SEPARATE_ONLY_ASSERTED = (
    "trainer.v1.trainer_mode",
    "trainer.v1.separate_async.parameter_sync_step",
    "trainer.v1.separate_async.num_warmup_batches",
)


def asserted_keys(shape: MegatronShape) -> tuple[str, ...]:
    """The keys this island re-reads from verl's resolved config (S27 G-FN).

    :data:`ASSERTED_KEYS` for an island with a rollout pool; without one the
    separate-mode keys drop out, because nothing writes them and an unwritten key
    asserts nothing (``trainer.check_asserted`` skips a None)."""
    if shape.rollout_pool:
        return ASSERTED_KEYS
    return tuple(key for key in ASSERTED_KEYS if key not in _SEPARATE_ONLY_ASSERTED)
