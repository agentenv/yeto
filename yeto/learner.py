"""Yeto learner: inner AdamW optimization with
non-blocking fragment sync against the Rust syncer.

Runs standalone (single GPU/CPU) or under torchrun (DDP across the GPUs and
nodes of one learner cluster). Only global rank 0 talks to the syncer; applied
fragment broadcasts are redistributed to other ranks with torch.distributed.

Per inner step (one optimizer step):
  1. forward/backward/AdamW on the local shard;
  2. counters advance for every fragment (c_steps, c_tokens);
  3. pending PULL_REQs are answered — pack fragment p and push it with its
     counters (only once c_steps[p] >= 1, which self-clocks the syncer);
  4. received BCAST fragments overwrite local params (alpha = 0), reset that
     fragment's counters, and adopt the syncer's global step.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections import deque

import torch
import torch.distributed as dist

from .autobatch import int_or_auto, rebalance_grad_accum, resolve_micro_batch_size
from .data import StreamingPackedBlocks, build_packed_dataset
from .fragments import FragmentLayout, build_layout
from .losses import load_custom_loss, load_pickled_loss, sft_loss
from .protocol import DTYPE_BF16, DTYPE_F32, DTYPE_Q4, SyncerClient, bulk_dtype
from .tensor_io import (
    apply_fragment,
    fragment_flat,
    pack_flat,
    pack_fragment,
    quantize_q4,
    unpack_fragment,
)

log = logging.getLogger("learner")

from .models import MODEL_ALIASES  # single source; see yeto/models.py


def accelerator_model_dtype(device: torch.device | str) -> torch.dtype:
    """Storage/compute dtype for frozen base weights on the local device."""
    device = torch.device(device)
    if device.type == "cpu":
        return torch.float32
    if device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(device)
        if major < 8:
            return torch.float16
        return torch.bfloat16
    return torch.bfloat16


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Yeto learner")
    p.add_argument("--model", required=True, help="HF model id or an alias from yeto/models.py (gemma4, qwen35-9b, llama31-8b, gptoss-120b, ...)")
    p.add_argument("--data", required=True, help="HF dataset id")
    p.add_argument(
        "--syncer",
        required=True,
        help="host:port of the syncer, or 'none' for a standalone DDP "
        "baseline (no async sync; stops at --max-local-steps)",
    )
    p.add_argument("--learner-id", type=int, required=True)
    p.add_argument("--num-learners", type=int, required=True)
    p.add_argument("--loss-function", default="cross_entropy")
    p.add_argument(
        "--train-on",
        choices=["assistant", "all"],
        default="assistant",
        help="which tokens carry loss: assistant-message tokens only "
        "(default) or every token",
    )
    p.add_argument("--tuning", choices=["lora", "full"], default="lora")
    p.add_argument(
        "--shard",
        choices=["ddp", "fsdp"],
        default="ddp",
        help="multi-GPU strategy; fsdp+lora shards only the frozen base "
        "(adapters stay replicated, any --syncer works); fsdp+full is only "
        "supported with --syncer none (synchronous baseline)",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
        help="which linears get adapters: attention projections only, every "
        "linear, or auto (attention for MoE models — keeps router and routed "
        "experts frozen and fragments small; all-linear for dense models)",
    )
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument(
        "--micro-batch-size",
        type=int_or_auto,
        default="auto",
        help="per-GPU micro batch; 'auto' (default) probes the largest size "
        "that fits VRAM at startup and shrinks --grad-accum to keep the "
        "effective batch constant",
    )
    p.add_argument(
        "--gradient-checkpointing",
        choices=["auto", "on", "off"],
        default="auto",
        help="recompute activations in backward; 'auto' enables it when the "
        "loaded base already occupies more than half of VRAM",
    )
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--fragments", type=int, default=8, help="P (= H, round-robin)")
    p.add_argument(
        "--fragment-pattern",
        choices=["binpack", "strided"],
        default="binpack",
        help="how tensors are grouped into fragments: size-balanced bin-packing "
        "or depth-interleaved transformer layers (layer i -> fragment i mod P)",
    )
    p.add_argument(
        "--merge-alpha",
        type=float,
        default=0.5,
        help="local weight when applying a broadcast fragment: "
        "θ ← α·θ_local + (1−α)·θ_global; 0 overwrites, 0.5 keeps the inner "
        "steps taken while the merge was in flight (Streaming DiLoCo / HALoS)",
    )
    p.add_argument(
        "--wire-dtype",
        choices=["bf16", "f32", "q4"],
        default="bf16",
        help="tensor encoding on the WAN; q4 sends pushes as 4-bit E3M0 "
        "block-quantized deltas (broadcasts stay bf16)",
    )
    p.add_argument("--wan-streams", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=None, help="cap dataset rows per learner")
    p.add_argument(
        "--tokenize",
        choices=["stream", "preload"],
        default="stream",
        help="stream: tokenize asynchronously in DataLoader workers (default); "
        "preload: materialize all blocks before training",
    )
    p.add_argument(
        "--stream-workers",
        type=int,
        default=2,
        help="DataLoader worker processes tokenizing ahead (stream mode)",
    )
    p.add_argument("--max-local-steps", type=int, default=1_000_000, help="safety stop")
    p.add_argument("--output-dir", default="checkpoints/out")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--probe-data",
        default=None,
        help="optional held-out data source for per-fragment utility probes; "
        "when set, rank 0 logs one-step probe utility for answered pulls",
    )
    p.add_argument(
        "--probe-log",
        default=None,
        help="JSONL output for fragment utility probes (default: "
        "<output-dir>/fragment_probe.jsonl)",
    )
    p.add_argument(
        "--probe-every",
        type=int,
        default=1,
        help="evaluate one in every N answered pull candidates",
    )
    p.add_argument(
        "--probe-batches",
        type=int,
        default=2,
        help="number of fixed probe batches used for each utility estimate",
    )
    p.add_argument(
        "--probe-batch-size",
        type=int,
        default=1,
        help="blocks per fixed probe batch",
    )
    p.add_argument(
        "--probe-max-rows",
        type=int,
        default=64,
        help="maximum rows materialized from --probe-data",
    )
    p.add_argument(
        "--probe-outer-lr",
        type=float,
        default=1.0,
        help="scale applied to the candidate fragment delta during probe eval",
    )
    p.add_argument(
        "--probe-freshness-scale",
        type=float,
        default=24.0,
        help="freshness = exp(-age / scale), where age is pull_step - base_version",
    )
    p.add_argument(
        "--debug-step-sleep-ms",
        type=float,
        default=0.0,
        help="debug/stress only: sleep after each optimizer step to emulate a slower learner",
    )
    p.add_argument(
        "--debug-push-delay-ms",
        type=float,
        default=0.0,
        help="debug/stress only: sleep before each fragment push to emulate late fragments",
    )
    p.add_argument(
        "--debug-delay-jitter-ms",
        type=float,
        default=0.0,
        help="debug/stress only: add uniform [0, jitter] ms to step/push sleeps",
    )
    p.add_argument(
        "--fixed-window-tokens",
        type=int,
        default=0,
        help="EXP/stress only: answer pulls from a cached fragment snapshot "
        "taken after at least this many tokens since the fragment reset",
    )
    p.add_argument(
        "--fixed-window-microsteps",
        type=int,
        default=0,
        help="EXP/stress only: answer pulls from a cached fragment snapshot "
        "taken after at least this many optimizer steps since reset",
    )
    p.add_argument(
        "--pad-to-fixed-window-tokens",
        action="store_true",
        help="accepted for experiment configs; token windows are rounded up "
        "to whole optimizer steps because the learner trains packed blocks",
    )
    p.add_argument(
        "--freeze-delta-before-delay",
        action="store_true",
        help="EXP/stress only: materialize payload/probe snapshot before "
        "applying --debug-push-delay-ms",
    )
    return p.parse_args(argv)


def setup_distributed() -> tuple[int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _from_pretrained_offline_first(factory, model_id: str, **kwargs):
    """Try the local cache before touching the Hub.

    A cache hit costs zero API requests; torchrun's 8 ranks otherwise each
    revalidate every config/tokenizer file per crash-loop cycle, which adds
    up against the Hub's per-IP rate limit. A cold cache (fresh spot node)
    falls back to a normal online load.
    """
    try:
        return factory.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        return factory.from_pretrained(model_id, **kwargs)


def load_model_and_tokenizer(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .models import resolve

    # The base uses the fastest native low-precision dtype for the device
    # (fp16 on pre-bf16 CUDA GPUs such as T4, bf16 on newer CUDA devices).
    # Natively-quantized checkpoints (fp8 MoE, mxfp4) are an inference format
    # whose forward kernels have no backward, so training loads dense floating
    # weights and never the packed low-precision ones.
    model_id = resolve(args.model)
    # fsdp+full: originals stay fp32 (uniform dtype for flat-param groups,
    # fp32 optimizer state) and MixedPrecision computes/communicates in bf16.
    # ddp/single and fsdp+lora: frozen base in native low precision; peft
    # leaves LoRA adapters in fp32, which keeps AdamW's exp_avg_sq in fp32.
    # Wire packing casts to the wire dtype either way.
    if (args.shard == "fsdp" and args.tuning == "full") or device.type != "cuda":
        dtype = torch.float32
    else:
        dtype = accelerator_model_dtype(device)
    tokenizer = _from_pretrained_offline_first(AutoTokenizer, model_id, trust_remote_code=True)
    model = _from_pretrained_offline_first(
        AutoModelForCausalLM, model_id, torch_dtype=dtype, trust_remote_code=True
    )
    if args.tuning == "lora":
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=resolve_lora_targets(
                getattr(args, "lora_targets", "auto"), model.config
            ),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    model.to(device)
    return model, tokenizer


# Attention projection names across the architectures we run (Llama/Qwen/
# Gemma-style q/k/v/o, fused qkv variants, DeepSeek's low-rank q_a/q_b and
# kv_a/kv_b split). peft treats a string target as a regex fullmatch on the
# module path.
_ATTENTION_TARGETS = (
    r".*\.(q_proj|k_proj|v_proj|o_proj|qkv_proj|out_proj"
    r"|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)$"
)
# Config attributes that mark a mixture-of-experts architecture.
_MOE_CONFIG_MARKERS = ("n_routed_experts", "num_experts", "num_local_experts")


def is_moe_config(config) -> bool:
    return any(getattr(config, a, None) for a in _MOE_CONFIG_MARKERS)


def resolve_lora_targets(choice: str, config) -> str:
    """Map --lora-targets onto a peft target_modules spec.

    MoE default is attention-only: "all-linear" would put adapters on every
    routed expert (most of a big MoE's parameters — fragments balloon from
    megabytes to gigabytes) AND on the router gate, whose load-balancing
    aux_loss we do not train against. Freezing router + experts is the
    established MoE fine-tuning recipe; anyone overriding gets a loud
    warning rather than a silent foot-gun.
    """
    if choice == "auto":
        return _ATTENTION_TARGETS if is_moe_config(config) else "all-linear"
    if choice == "attention":
        return _ATTENTION_TARGETS
    if choice == "all-linear" and is_moe_config(config):
        log.warning(
            "--lora-targets all-linear on a MoE model adapts every routed "
            "expert and the router gate (whose aux_loss is not trained); "
            "fragments will be huge — prefer --lora-targets attention"
        )
    return "all-linear"


def trainable_params(model) -> dict[str, torch.Tensor]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


# Module-path segments FSDP (and activation checkpointing) may splice into
# parameter FQNs. Fragment layouts are keyed by name, so names must be
# identical across shard modes — an fsdp-lora learner and a ddp-lora learner
# have to be able to join the same syncer.
_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.")


def normalize_param_name(name: str) -> str:
    """Strip wrapper-inserted module path segments from a parameter FQN."""
    for prefix in _WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def allreduce_trainable_grads(params, world: int) -> None:
    """All-reduce (SUM) each param's grad in place and divide by world.

    FSDP leaves params passed via ignored_states unmanaged, so the replicated
    LoRA adapter grads are never reduced; calling this at optimizer-step
    boundaries (before clipping) reproduces DDP-mean semantics — each rank
    normalizes its loss by its own trained-token count, and SUM/world of
    those grads is the cross-rank mean. No-op when world <= 1; params whose
    grad is None are skipped.
    """
    if world <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _debug_sleep(base_ms: float, jitter_ms: float) -> None:
    delay_ms = max(0.0, base_ms)
    if jitter_ms > 0.0:
        delay_ms += random.uniform(0.0, jitter_ms)
    if delay_ms > 0.0:
        time.sleep(delay_ms / 1000.0)


class FragmentUtilityProbe:
    """Opt-in one-step utility probe for asynchronous fragment responses.

    The learner does not own the syncer's exact state after other learners'
    most recent merges. It does, however, know the last global fragment values
    it has applied. The probe therefore evaluates utility against this
    learner-known global state: all trainable fragments are temporarily reset
    to their last broadcast anchors, the candidate fragment delta is applied,
    and the fixed probe loss is compared to the anchor loss.
    """

    def __init__(self, args, model, params, layout, tokenizer, device, compute_loss):
        if args.probe_every < 1:
            raise ValueError("--probe-every must be >= 1")
        if args.probe_batches < 1:
            raise ValueError("--probe-batches must be >= 1")
        if args.probe_batch_size < 1:
            raise ValueError("--probe-batch-size must be >= 1")
        self.args = args
        self.model = model
        self.params = params
        self.layout = layout
        self.device = device
        self.compute_loss = compute_loss
        self.path = args.probe_log or os.path.join(args.output_dir, "fragment_probe.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        # Start fresh for each learner process; append is used so resumed runs
        # can choose a distinct path without file-mode plumbing.
        open(self.path, "w").close()

        ds = build_packed_dataset(
            args.probe_data,
            tokenizer,
            learner_id=0,
            num_learners=1,
            seq_len=args.seq_len,
            max_rows=args.probe_max_rows,
            train_on=args.train_on,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.probe_batch_size, shuffle=False, drop_last=False
        )
        self.batches = []
        for input_ids, weights in loader:
            self.batches.append(
                (input_ids.to(device, non_blocking=True), weights.to(device, non_blocking=True))
            )
            if len(self.batches) >= args.probe_batches:
                break
        if not self.batches:
            raise ValueError("--probe-data produced no batches")

        self.candidates_seen = 0
        self.momentum = [None] * layout.num_fragments
        self.norm_history = [deque(maxlen=96) for _ in range(layout.num_fragments)]
        self.norm_ema = [None] * layout.num_fragments
        self.norm_var = [1e-4] * layout.num_fragments
        log.info(
            "fragment utility probe enabled: %d batch(es), log=%s",
            len(self.batches),
            self.path,
        )

    def note_broadcast(self, fid: int, old_anchor: torch.Tensor | None, new_anchor: torch.Tensor) -> None:
        if old_anchor is None or old_anchor.numel() != new_anchor.numel():
            return
        direction = new_anchor.float().cpu() - old_anchor.float().cpu()
        prev = self.momentum[fid]
        self.momentum[fid] = direction if prev is None else 0.85 * prev + 0.15 * direction

    def maybe_record(
        self,
        *,
        learner_id: int,
        fid: int,
        pull_step: int,
        base_version: int,
        local_step: int,
        c_steps: int,
        c_tokens: int,
        local_flat: torch.Tensor,
        anchors: list[torch.Tensor],
    ) -> None:
        self.candidates_seen += 1
        update = local_flat.detach().float().cpu() - anchors[fid].float().cpu()
        norm = float(update.norm().item())
        hist = self.norm_history[fid]
        if hist:
            h = torch.tensor(list(hist), dtype=torch.float32)
            med = float(h.median().item())
            mad = float((h - med).abs().median().item()) + 1e-8
            norm_anomaly = abs(norm - med) / mad
        else:
            norm_anomaly = 0.0

        prev_mean = self.norm_ema[fid]
        if prev_mean is None:
            uncertainty = 0.0
        else:
            delta = norm - prev_mean
            uncertainty = math.sqrt(self.norm_var[fid]) / (abs(prev_mean) + 1e-8)
            self.norm_var[fid] = 0.85 * self.norm_var[fid] + 0.15 * delta * delta
        self.norm_ema[fid] = norm if prev_mean is None else 0.85 * prev_mean + 0.15 * norm
        hist.append(norm)

        if self.candidates_seen % self.args.probe_every != 0:
            return

        age = max(0, int(pull_step) - int(base_version))
        freshness = math.exp(-age / max(self.args.probe_freshness_scale, 1e-9))
        mom = self.momentum[fid]
        alignment = _cosine(update, mom) if mom is not None else 0.0
        combined_logit = (
            2.25 * alignment
            + 1.35 * freshness
            - 0.55 * math.log1p(norm_anomaly)
            - 0.80 * uncertainty
        )
        combined_score = _sigmoid(combined_logit)
        base_loss, trial_loss, utility_se = self._utility_losses(fid, update, anchors)
        utility = base_loss - trial_loss
        record = {
            "schema": "fragment_probe_v2",
            "oracle_scope": "learner_known_global",
            "learner_id": learner_id,
            "fragment": fid,
            "pull_step": int(pull_step),
            "base_version": int(base_version),
            "local_step": int(local_step),
            "c_steps": int(c_steps),
            "c_tokens": int(c_tokens),
            "age": age,
            "freshness": freshness,
            "alignment": alignment,
            "uncertainty": uncertainty,
            "norm_anomaly": norm_anomaly,
            "combined_score": combined_score,
            "update_norm": norm,
            "base_loss": base_loss,
            "trial_loss": trial_loss,
            "utility": utility,
            "utility_se": utility_se,
            "bad_strict": None if utility_se is None else utility + utility_se < 0.0,
            "probe_batches": len(self.batches),
            "probe_outer_lr": self.args.probe_outer_lr,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _utility_losses(
        self, fid: int, update: torch.Tensor, anchors: list[torch.Tensor]
    ) -> tuple[float, float, float | None]:
        local_snapshot = [
            fragment_flat(frag, self.params).detach().cpu()
            for frag in self.layout.fragments
        ]
        was_training = self.model.training
        try:
            for i, frag in enumerate(self.layout.fragments):
                apply_fragment(frag, anchors[i].to(self.device), self.params)
            base_loss, base_by_batch = self._probe_losses()
            trial = anchors[fid].float().cpu() + self.args.probe_outer_lr * update
            apply_fragment(self.layout.fragments[fid], trial.to(self.device), self.params)
            trial_loss, trial_by_batch = self._probe_losses()
        finally:
            for frag, flat in zip(self.layout.fragments, local_snapshot):
                apply_fragment(frag, flat.to(self.device), self.params)
            self.model.train(was_training)
        utilities = [b - t for b, t in zip(base_by_batch, trial_by_batch)]
        if len(utilities) < 2:
            utility_se = None
        else:
            mean = sum(utilities) / len(utilities)
            var = sum((u - mean) ** 2 for u in utilities) / (len(utilities) - 1)
            utility_se = math.sqrt(var / len(utilities))
        return base_loss, trial_loss, utility_se

    def _probe_losses(self) -> tuple[float, list[float]]:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0.0
        batch_losses = []
        with torch.no_grad():
            for input_ids, weights in self.batches:
                out = self.model(input_ids=input_ids)
                loss, n = self.compute_loss(out.logits, input_ids, weights)
                loss_value = float(loss.item())
                token_count = float(n.item())
                total_loss += loss_value
                total_tokens += token_count
                batch_losses.append(loss_value / max(token_count, 1.0))
        return total_loss / max(total_tokens, 1.0), batch_losses


def main(argv=None) -> None:
    args = parse_args(argv)
    rank, world = setup_distributed()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s learner{args.learner_id}.r{rank} %(levelname)s %(message)s",
    )
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        if os.environ.get("SKYPILOT_NUM_GPUS_PER_NODE", "0") != "0":
            raise RuntimeError(
                "GPUs were provisioned but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}); check the torch wheel's CUDA "
                "version against the node's driver instead of training on CPU"
            )
        device = torch.device("cpu")

    if args.shard == "fsdp" and device.type != "cuda":
        raise RuntimeError(
            "--shard fsdp requires a CUDA accelerator (torch FSDP cannot "
            "shard on cpu); use --shard ddp or run on GPUs"
        )
    if args.probe_data is not None:
        if args.syncer == "none":
            raise RuntimeError("--probe-data requires an async syncer run")
        if world > 1:
            raise RuntimeError(
                "--probe-data currently supports one process per learner; "
                "multi-rank learners need a distributed probe evaluator"
            )
        if args.shard == "fsdp":
            raise RuntimeError(
                "--probe-data is not supported with --shard fsdp yet because "
                "rank-local probe loss would not see full parameters"
            )

    if not 0.0 <= args.merge_alpha < 1.0:
        raise ValueError(f"--merge-alpha must be in [0, 1), got {args.merge_alpha}")

    log.info("loading model %s (%s)", args.model, args.tuning)
    model, tokenizer = load_model_and_tokenizer(args, device)

    grad_ckpt = args.gradient_checkpointing == "on"
    if args.gradient_checkpointing == "auto" and device.type == "cuda":
        # The base is fully on-device here (load ends with model.to(device)),
        # so free memory directly reflects what activations must fit into.
        free, total = torch.cuda.mem_get_info(device)
        grad_ckpt = free < total / 2
    if grad_ckpt:
        # Non-reentrant checkpointing composes with FSDP and peft; the input
        # grad hook keeps the graph alive from the embeddings down to the
        # first adapter when the base is frozen.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if args.tuning == "lora":
            model.enable_input_require_grads()
        log.info("gradient checkpointing enabled")

    params = trainable_params(model)
    layout = build_layout(
        [(n, p.numel()) for n, p in params.items()], args.fragments, args.fragment_pattern
    )
    log.info(
        "%d trainable tensors -> %d fragments (%.1f MB total)",
        len(params),
        layout.num_fragments,
        sum(p.numel() for p in params.values()) * 2 / 1e6,
    )

    peft_model = None  # unwrapped peft handle, kept for fsdp+lora save
    if args.shard == "fsdp":
        if args.tuning == "lora":
            # FSDP2 (fully_shard) shards the frozen bf16 base per-parameter as
            # DTensors — no flat-param groups. The LoRA adapters go in
            # ignored_params: they stay ordinary replicated per-rank fp32
            # tensors, so fragment pack/apply/INIT works on them unchanged and
            # run_inner_loop all-reduces their grads at each optimizer-step
            # boundary (fully_shard never sees them).
            try:
                from torch.distributed.fsdp import fully_shard
            except ImportError as exc:  # pre-2.7 torch (old-driver nodes)
                raise RuntimeError(
                    "--shard fsdp with --tuning lora needs torch>=2.7 "
                    "(fully_shard with ignored_params); this node's driver "
                    "capped torch below that — use --shard ddp here"
                ) from exc

            peft_model = model
            ignored = set(params.values())
            # Shard each transformer block separately (comm/compute overlap,
            # one block unsharded at a time); the root call picks up whatever
            # sits outside the blocks (embeddings, final norm, lm_head). Only
            # the OUTERMOST ModuleLists are treated as blocks — descending
            # would give every MoE expert its own tiny all-gather group.
            def outer_module_lists(root):
                found = []

                def visit(m):
                    for child in m.children():
                        if isinstance(child, torch.nn.ModuleList) and len(child) >= 2:
                            found.append(child)
                        else:
                            visit(child)

                visit(root)
                return found

            blocks = [b for ml in outer_module_lists(model) for b in ml]
            for block in blocks:
                block_ignored = ignored & set(block.parameters())
                fully_shard(block, ignored_params=block_ignored)
            fully_shard(model, ignored_params=ignored)
            log.info(
                "fully_shard: %d blocks sharded per-parameter, %d params replicated",
                len(blocks),
                len(ignored),
            )
            wrapped = {
                normalize_param_name(n): p for n, p in model.named_parameters() if p.requires_grad
            }
            # The fragment layout was built from pre-wrap names; they must
            # survive wrapping bit-identically so this learner speaks the
            # same layout as ddp/single-GPU learners on the same syncer.
            if set(wrapped) != set(params):
                raise RuntimeError(
                    "FSDP wrapping changed trainable parameter names; layout "
                    "would diverge from other learners. Mismatch sample: "
                    f"{sorted(set(wrapped) ^ set(params))[:5]}"
                )
            params = wrapped
        else:
            import functools

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=1_000_000
            )
            if args.syncer != "none":
                raise ValueError(
                    "--shard fsdp with --tuning full: full-parameter sync "
                    "from sharded state is unsupported; use lora or ddp"
                )
            mixed_precision = None
            if device.type == "cuda":
                # fp32 originals (and optimizer state); bf16 compute and
                # all-gather; fp32 gradient reduction.
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                )
            model = FSDP(
                model,
                auto_wrap_policy=wrap_policy,
                use_orig_params=True,  # keep named originals for the optimizer
                mixed_precision=mixed_precision,
                device_id=device if device.type == "cuda" else None,
            )
            params = trainable_params(model)
    elif args.tuning == "lora":
        # Frozen-base LoRA, base replicated per rank and left UNWRAPPED. The
        # base contributes no gradients, so it needs no DDP reducer; the
        # adapters are ordinary replicated tensors whose grads
        # allreduce_trainable_grads averages at each optimizer step. Requires
        # the base to fit per-GPU; --shard fsdp shards it when it does not.
        # peft_model marks the LoRA save path (explicit adapter state dict,
        # no base gather).
        peft_model = model
    elif world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        params = trainable_params(model.module)

    opt = torch.optim.AdamW(params.values(), lr=args.inner_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps))
    )

    # Resolve the micro batch AFTER wrapping and optimizer construction
    # (memory-accurate) and BEFORE the loader/syncer exist (nothing counts
    # the probe). See yeto/autobatch.py.
    requested_mb = args.micro_batch_size
    args.micro_batch_size = resolve_micro_batch_size(
        args, model, params, opt, tokenizer, device, world
    )
    if requested_mb == "auto":
        args.grad_accum = rebalance_grad_accum(args.grad_accum, args.micro_batch_size)
        log.info(
            "auto micro-batch: %d per GPU (grad-accum -> %d)",
            args.micro_batch_size,
            args.grad_accum,
        )

    if args.tokenize == "stream":
        dataset = StreamingPackedBlocks(
            args.data,
            tokenizer,
            args.learner_id,
            args.num_learners,
            args.seq_len,
            args.max_rows,
            rank=rank,
            world=world,
            train_on=args.train_on,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            num_workers=args.stream_workers,
            prefetch_factor=4 if args.stream_workers > 0 else None,
            persistent_workers=args.stream_workers > 0,
        )
        log.info(
            "streaming tokenization: %d worker(s) packing %d-token blocks ahead of training",
            args.stream_workers,
            args.seq_len,
        )
    else:
        dataset = build_packed_dataset(
            args.data,
            tokenizer,
            args.learner_id,
            args.num_learners,
            args.seq_len,
            args.max_rows,
            train_on=args.train_on,
        )
        sampler = None
        if world > 1:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(dataset, num_replicas=world, rank=rank)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
        )
        log.info("dataset ready: %d blocks of %d tokens", len(dataset), args.seq_len)

    wire_dtype = {"bf16": DTYPE_BF16, "f32": DTYPE_F32, "q4": DTYPE_Q4}[args.wire_dtype]
    client = None
    if rank == 0 and args.syncer != "none":
        host, port = args.syncer.rsplit(":", 1)
        client = SyncerClient(
            (host, int(port)), args.learner_id, layout, wire_dtype, args.wan_streams
        )
        client.start()
        log.info("connected to syncer at %s", args.syncer)
        if args.learner_id == 0:
            for fid, frag in enumerate(layout.fragments):
                client.send_init(fid, pack_fragment(frag, params, bulk_dtype(wire_dtype)))
            log.info("sent INIT_PARAMS for %d fragments", layout.num_fragments)

    run_inner_loop(
        args, model, params, layout, opt, sched, loader, client, rank, world, device, tokenizer
    )

    if rank == 0:
        if args.shard == "fsdp" and args.tuning == "full":
            # Gathering a full state dict from shards is not needed for the
            # baseline-comparison use of this mode.
            log.info("skipping checkpoint save in fsdp baseline mode")
        else:
            save_dir = args.output_dir
            os.makedirs(save_dir, exist_ok=True)
            if peft_model is not None:
                # lora (fsdp-sharded or replicated base): the adapters are
                # replicated ordinary tensors in `params`, so hand
                # save_pretrained an explicit state dict through the unwrapped
                # peft handle — the frozen base is never gathered or touched.
                peft_model.save_pretrained(
                    save_dir,
                    state_dict={n: p.detach().cpu() for n, p in params.items()},
                )
            else:
                target = model.module if world > 1 else model
                target.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            log.info("saved model to %s", save_dir)
        if client is not None:
            client.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def run_inner_loop(
    args, model, params, layout, opt, sched, loader, client, rank, world, device, tokenizer
):
    # Counters (Alg. 1): incremented for all fragments each step, reset per
    # fragment on receipt. Tracked as global totals + per-fragment snapshots.
    steps_total = 0
    tokens_total = 0
    steps_at_reset = [0] * layout.num_fragments
    tokens_at_reset = [0] * layout.num_fragments
    fragment_versions = [0] * layout.num_fragments  # last applied version per fragment
    pending_pulls: list = []  # pulls deferred until c_steps >= 1
    global_step = 0
    # Q4 pushes are deltas anchored at the last *received* global value per
    # fragment; the utility probe also needs these anchors for bf16/f32 runs
    # so it can evaluate candidate deltas against the learner-known global
    # state. Before any broadcast the anchor is the base-model value, which
    # every learner loads identically (and learner 0 sends as INIT_PARAMS).
    anchors: list[torch.Tensor] | None = None
    if rank == 0 and client is not None and (client.dtype == DTYPE_Q4 or args.probe_data is not None):
        anchors = [fragment_flat(frag, params).cpu() for frag in layout.fragments]
    # c_tokens counts RAW tokens processed (throughput proxy for merge
    # weighting), not the subset of loss-weighted tokens.
    tokens_per_inner_step = world * args.micro_batch_size * args.grad_accum * args.seq_len
    fixed_window_steps = 1
    if args.fixed_window_microsteps > 0:
        fixed_window_steps = max(fixed_window_steps, args.fixed_window_microsteps)
    if args.fixed_window_tokens > 0:
        fixed_window_steps = max(
            fixed_window_steps,
            math.ceil(args.fixed_window_tokens / max(tokens_per_inner_step, 1)),
        )
    fixed_window_enabled = args.fixed_window_microsteps > 0 or args.fixed_window_tokens > 0
    fixed_window_snapshots: list[dict | None] | None = (
        [None] * layout.num_fragments if fixed_window_enabled else None
    )
    if fixed_window_enabled and rank == 0:
        log.info(
            "fixed response window enabled: %d step(s), %d token(s)/step, "
            "target tokens=%d, target microsteps=%d",
            fixed_window_steps,
            tokens_per_inner_step,
            args.fixed_window_tokens,
            args.fixed_window_microsteps,
        )

    if args.loss_function.startswith("pickle:"):
        compute_loss = load_pickled_loss(args.loss_function)
    elif args.loss_function.startswith("custom:"):
        compute_loss = load_custom_loss(args.loss_function)
    else:
        compute_loss = lambda logits, ids, w: sft_loss(logits, ids, args.loss_function, w)  # noqa: E731

    probe = None
    if rank == 0 and client is not None and args.probe_data is not None:
        probe = FragmentUtilityProbe(args, model, params, layout, tokenizer, device, compute_loss)

    shutdown = False
    epoch = 0
    t_last = time.monotonic()
    while not shutdown and steps_total < args.max_local_steps:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        accum = 0
        opt.zero_grad(set_to_none=True)
        for input_ids, weights in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            out = model(input_ids=input_ids)
            loss, _ = compute_loss(out.logits, input_ids, weights)
            # DDP averages gradients across ranks, so normalizing each rank's
            # sum-over-tokens loss by its *own* trained-token count yields
            # (approximately) the global per-trained-token mean gradient.
            trained_tokens = weights.sum().clamp(min=1.0)
            (loss / (trained_tokens * args.grad_accum)).backward()
            accum += 1
            if accum < args.grad_accum:
                continue
            accum = 0
            if args.tuning == "lora":
                # The adapters are never grad-synced by a wrapper — fsdp+lora
                # ignores them, the replicated path has no wrapper — so average
                # them across ranks before clipping (no-op at world==1). After
                # this the replicated params/grads are identical on every rank,
                # so a plain clip over them is correct (the frozen base
                # contributes no grads).
                allreduce_trainable_grads(params.values(), world)
                torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
            elif args.shard == "fsdp":
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            steps_total += 1
            tokens_total += tokens_per_inner_step
            step_jitter_ms = args.debug_delay_jitter_ms if args.debug_step_sleep_ms > 0.0 else 0.0
            _debug_sleep(args.debug_step_sleep_ms, step_jitter_ms)

            if rank == 0 and fixed_window_snapshots is not None:
                for snap_fid, snap in enumerate(fixed_window_snapshots):
                    if snap is not None:
                        continue
                    snap_steps = steps_total - steps_at_reset[snap_fid]
                    if snap_steps < fixed_window_steps:
                        continue
                    fixed_window_snapshots[snap_fid] = {
                        "flat": fragment_flat(layout.fragments[snap_fid], params).detach().cpu(),
                        "c_steps": snap_steps,
                        "c_tokens": tokens_total - tokens_at_reset[snap_fid],
                        "local_step": steps_total,
                        "base_version": fragment_versions[snap_fid],
                    }

            if steps_total % 10 == 0 and rank == 0:
                dt = time.monotonic() - t_last
                t_last = time.monotonic()
                log.info(
                    "local_step=%d global_step=%d loss/token=%.4f (%.2f s/step)",
                    steps_total,
                    global_step,
                    loss.item() / trained_tokens.item(),  # per trained token
                    dt / 10,
                )

            # --- fragment sync at the step boundary (never blocks) ---
            # Broadcasts are applied BEFORE pulls are answered: with the
            # syncer's pipelined rounds, the pull for a fragment's next
            # round (control stream) can overtake the broadcast that closed
            # its previous round (data streams). Answering first would push
            # a stale base_version; applying first resets the fragment's
            # counters, so the self-clock defers the answer one step and it
            # then carries the fresh anchor.
            actions = []  # (fid, version, flat_f32) applied this boundary
            if rank == 0 and client is not None:
                client.check_health()
                # 1. collect received global fragments
                for bc in client.drain_updates():
                    flat = unpack_fragment(
                        layout.fragments[bc.fragment_id], bc.data, bulk_dtype(client.dtype)
                    )
                    if anchors is not None:
                        if probe is not None:
                            probe.note_broadcast(bc.fragment_id, anchors[bc.fragment_id], flat)
                        # The anchor is the raw global value (pre-blend), so
                        # the syncer can reconstruct pushes from Θ(version)+δ.
                        anchors[bc.fragment_id] = flat.clone()
                    actions.append((bc.fragment_id, bc.version, flat))
                shutdown = client.shutdown.is_set()

            if world > 1:
                meta = [(f, v) for f, v, _ in actions] if rank == 0 else None
                box = [meta, shutdown]
                dist.broadcast_object_list(box, src=0)
                meta, shutdown = box
                if rank != 0:
                    actions = [(f, v, torch.empty(layout.fragments[f].numel)) for f, v in meta]
                for fid, version, flat in actions:
                    flat = flat.to(device)
                    dist.broadcast(flat, src=0)
                    # α-blend: keep a share of the inner steps taken while the
                    # merge was in flight. Ranks hold identical params, so
                    # blending after the broadcast stays consistent.
                    if args.merge_alpha > 0:
                        local = fragment_flat(layout.fragments[fid], params)
                        flat = args.merge_alpha * local + (1.0 - args.merge_alpha) * flat
                    apply_fragment(layout.fragments[fid], flat, params)
                    if rank == 0:
                        steps_at_reset[fid] = steps_total
                        tokens_at_reset[fid] = tokens_total
                        fragment_versions[fid] = version
                        if fixed_window_snapshots is not None:
                            fixed_window_snapshots[fid] = None
                    global_step = max(global_step, version)
            else:
                for fid, version, flat in actions:
                    flat = flat.to(device)
                    if args.merge_alpha > 0:
                        local = fragment_flat(layout.fragments[fid], params)
                        flat = args.merge_alpha * local + (1.0 - args.merge_alpha) * flat
                    apply_fragment(layout.fragments[fid], flat, params)
                    steps_at_reset[fid] = steps_total
                    tokens_at_reset[fid] = tokens_total
                    fragment_versions[fid] = version
                    if fixed_window_snapshots is not None:
                        fixed_window_snapshots[fid] = None
                    global_step = max(global_step, version)

            # 2. answer pulls whose fragment has made progress since the
            # (just-applied) broadcasts.
            if rank == 0 and client is not None:
                pending_pulls.extend(client.drain_pulls())
                still_pending = []
                for pull in pending_pulls:
                    fid = pull.fragment_id
                    if fixed_window_snapshots is not None:
                        snap = fixed_window_snapshots[fid]
                        if snap is None:
                            still_pending.append(pull)
                            continue
                        local_flat = snap["flat"]
                        c_steps = int(snap["c_steps"])
                        c_tokens = int(snap["c_tokens"])
                        local_step_for_push = int(snap["local_step"])
                        base_version_for_push = int(snap["base_version"])
                    else:
                        c_steps = steps_total - steps_at_reset[fid]
                        if c_steps < 1:
                            still_pending.append(pull)
                            continue
                        c_tokens = tokens_total - tokens_at_reset[fid]
                        local_flat = fragment_flat(layout.fragments[fid], params).detach().cpu()
                        local_step_for_push = steps_total
                        base_version_for_push = fragment_versions[fid]
                    if probe is not None:
                        if anchors is None:
                            raise RuntimeError("fragment utility probe requires anchors")
                        probe.maybe_record(
                            learner_id=args.learner_id,
                            fid=fid,
                            pull_step=pull.global_step,
                            base_version=base_version_for_push,
                            local_step=local_step_for_push,
                            c_steps=c_steps,
                            c_tokens=c_tokens,
                            local_flat=local_flat,
                            anchors=anchors,
                        )
                    if anchors is not None and client.dtype == DTYPE_Q4:
                        delta = local_flat - anchors[fid]
                        payload = quantize_q4(delta)
                    else:
                        payload = pack_flat(local_flat, client.dtype)
                    _debug_sleep(args.debug_push_delay_ms, args.debug_delay_jitter_ms)
                    client.push_fragment(
                        fid,
                        pull.global_step,
                        base_version_for_push,
                        local_step_for_push,
                        c_steps,
                        c_tokens,
                        payload,
                    )
                pending_pulls = still_pending

            if shutdown or steps_total >= args.max_local_steps:
                break
        epoch += 1
    log.info("inner loop done at local_step=%d global_step=%d", steps_total, global_step)


if __name__ == "__main__":
    main()
