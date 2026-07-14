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
import uuid
from collections import deque

import torch
import torch.distributed as dist

from .autobatch import int_or_auto, rebalance_grad_accum, resolve_micro_batch_size
from .bcmp_shadow import BCMPShadowTracker
from .data import StreamingPackedBlocks, build_packed_dataset
from .fragments import FragmentLayout, build_layout
from .losses import load_custom_loss, load_pickled_loss, sft_loss
from .protocol import (
    DTYPE_BF16,
    DTYPE_F32,
    DTYPE_Q4,
    PushAudit,
    SyncerClient,
    bulk_dtype,
)
from .scaffold import (
    VersionedControlPairs,
    accumulate_control,
    grad_correction,
    local_control,
)
from .tensor_io import (
    apply_fragment,
    dequantize_q4,
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
    p.add_argument(
        "--model",
        required=True,
        help="HF model id or an alias from yeto/models.py (gemma4, qwen35-9b, llama31-8b, gptoss-120b, ...)",
    )
    p.add_argument("--data", required=True, help="HF dataset id")
    p.add_argument(
        "--syncer",
        required=True,
        help="host:port of the syncer, or 'none' for a standalone DDP "
        "baseline (no async sync; stops at --max-local-steps)",
    )
    p.add_argument("--learner-id", type=int, required=True)
    p.add_argument("--num-learners", type=int, required=True)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base RNG seed; learner id and distributed rank are mixed in deterministically",
    )
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
    p.add_argument(
        "--inner-optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
        help="inner optimizer; plain SGD is required by the SCAFFOLD "
        "zero-sum correctness mode",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="global gradient-norm cap; 0 disables clipping",
    )
    p.add_argument(
        "--bcmp-shadow-path",
        default=None,
        help="optional behavior-preserving JSONL trace of BC-MP ray, "
        "work-clipped slab, and hard-reset AdamW counterfactuals at the "
        "first clipped gradient after sampled fragment broadcasts",
    )
    p.add_argument(
        "--bcmp-shadow-every",
        type=int,
        default=1,
        help="sample every Nth applied fragment broadcast for --bcmp-shadow-path",
    )
    p.add_argument("--fragments", type=int, default=8, help="P (= H, round-robin)")
    p.add_argument(
        "--fragment-pattern",
        choices=["binpack", "strided"],
        default="binpack",
        help="how tensors are grouped into fragments: size-balanced bin-packing "
        "or depth-interleaved transformer layers (layer i -> fragment i mod P)",
    )
    p.add_argument(
        "--matrix-merge",
        choices=["rda", "iso", "worker-snr"],
        default="rda",
        help="syncer aggregation for non-embedding (matrix) fragments: "
        "rda = weighted radial-directional averaging (default); "
        "iso = Iso-C-style isotropic aggregation (IsoLoCo, arXiv 2607.03011) "
        "— average the per-tensor deltas, then flatten each averaged "
        "matrix's singular-value spectrum to its mean; non-2D tensors join "
        "the direct-averaged fragment; "
        "worker-snr = memoryless cross-worker consensus shrink — per tensor "
        "block scale the mean delta by its signal-to-noise confidence "
        "q = (|gbar|^2/d) / (|gbar|^2/d + sigma^2/M), then global norm-match "
        "back to the plain-mean step norm",
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
    p.add_argument(
        "--inner-control-variate",
        choices=["none", "scaffold_lite", "scaffold_full"],
        default="none",
        help="endpoint-derived SCAFFOLD controls. 'scaffold_lite' overwrites "
        "the local control each round; 'scaffold_full' accumulates Option II "
        "controls on the learner. Both apply (c_i - c) to "
        "each inner gradient and require a matching syncer mode.",
    )
    p.add_argument(
        "--scaffold-beta",
        type=float,
        default=1.0,
        help="positive Option-II control accumulation coefficient",
    )
    p.add_argument(
        "--scaffold-control-shuffle",
        action="store_true",
        help="receive a fixed cyclic derangement of full-control residuals; "
        "must match the syncer setting",
    )
    p.add_argument(
        "--scaffold-correctness-mode",
        action="store_true",
        help="require equal-window barrier synchronization, f32 wire, no "
        "reconnect/lag, and constant-LR unclipped plain SGD",
    )
    p.add_argument(
        "--scaffold-audit-path",
        default=None,
        help="correctness-mode output for one real pre-clip correction and "
        "post-step correction-induced displacement",
    )
    p.add_argument(
        "--max-reconnects",
        type=int,
        default=None,
        help="maximum syncer reconnect attempts; 0 disables reconnects",
    )
    p.add_argument("--wan-streams", type=int, default=4)
    p.add_argument(
        "--max-rows", type=int, default=None, help="cap dataset rows per learner"
    )
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
        "--fixed-window-schedule",
        default=None,
        help="EXP: online sync-horizon changes as 'commit1:h1,commit2:h2,...'"
        " — after this learner has answered commitN pulls (local commit"
        " index, counted across fragments) the fixed response window becomes"
        " hN microsteps (hN * tokens-per-step raw tokens). Before the first"
        " entry the --fixed-window-microsteps/--fixed-window-tokens window"
        " applies. Enables fixed-window mode by itself.",
    )
    p.add_argument(
        "--optimizer-state-capture-dir",
        default=None,
        help="EXP: rank-0 directory for exact AdamW/window/push lifecycle captures",
    )
    p.add_argument(
        "--optimizer-state-capture-every",
        type=int,
        default=1,
        help="capture one in every N eligible broadcast/window events",
    )
    p.add_argument(
        "--optimizer-state-capture-max-hmc-events",
        type=int,
        default=32,
        help="maximum admitted first-post-broadcast AdamW events",
    )
    p.add_argument(
        "--optimizer-state-capture-max-midpoint-windows",
        type=int,
        default=32,
        help="maximum admitted exact anchor/H/2/H windows",
    )
    p.add_argument(
        "--optimizer-state-capture-max-bytes",
        type=int,
        default=4 * 1024**3,
        help="maximum finalized capture artifact plus checksum bytes",
    )
    p.add_argument(
        "--optimizer-state-capture-background-writer",
        action="store_true",
        help="EXP qualifier: publish immutable serialized capture artifacts through "
        "one bounded FIFO background writer; default capture remains synchronous",
    )
    p.add_argument(
        "--optimizer-state-capture-writer-max-items",
        type=int,
        default=4,
        help="maximum queued plus in-flight background capture artifacts",
    )
    p.add_argument(
        "--optimizer-state-capture-writer-max-bytes",
        type=int,
        default=4 * 1024**3,
        help="maximum immutable queued plus in-flight background payload bytes",
    )
    p.add_argument(
        "--freeze-delta-before-delay",
        action="store_true",
        help="EXP/stress only: materialize payload/probe snapshot before "
        "applying --debug-push-delay-ms",
    )
    p.add_argument(
        "--debug-broadcast-lag-commits",
        type=int,
        default=0,
        help="EXP/stress only: hold each fragment broadcast in a "
        "per-fragment FIFO and apply it only once this many newer "
        "broadcasts for that fragment have arrived, so local windows are "
        "computed against a K-commits-old global state",
    )
    p.add_argument(
        "--barrier-sync",
        action="store_true",
        help="EXP: true lockstep (barrier-synchronized) DiLoCo. After the "
        "learner pushes a fragment delta in response to a pull it BLOCKS — "
        "takes no further inner optimizer steps — until the syncer's merged "
        "broadcast for that fragment (a strictly newer version) has arrived "
        "and been applied, then resumes the next window from the merged "
        "global. This reproduces original DiLoCo's (arXiv 2311.08105) "
        "worker barrier, versus the default non-barrier strict-quorum "
        "schedule where learners keep local-training while a merge is in "
        "flight. Absent, behavior is byte-identical to the non-barrier loop.",
    )
    args = p.parse_args(argv)
    if args.fixed_window_schedule is not None:
        try:
            args.fixed_window_schedule = parse_fixed_window_schedule(
                args.fixed_window_schedule
            )
        except ValueError as exc:
            p.error(f"--fixed-window-schedule: {exc}")
    if args.debug_broadcast_lag_commits < 0:
        p.error("--debug-broadcast-lag-commits must be >= 0")
    if args.debug_broadcast_lag_commits > 0 and not (
        args.fixed_window_microsteps > 0
        or args.fixed_window_tokens > 0
        or args.fixed_window_schedule is not None
    ):
        p.error(
            "--debug-broadcast-lag-commits requires a fixed response "
            "window so window resets can move to push time"
        )
    if args.debug_broadcast_lag_commits > 0 and args.wire_dtype == "q4":
        # Lag mode pushes deliberately old base_versions; the syncer rejects
        # every q4 delta whose base is not current (validate_push_candidate),
        # so the first lagged push would stall the fragment forever.
        p.error("--debug-broadcast-lag-commits is incompatible with --wire-dtype q4")
    if args.optimizer_state_capture_every < 1:
        p.error("--optimizer-state-capture-every must be >= 1")
    if args.optimizer_state_capture_max_hmc_events < 0:
        p.error("--optimizer-state-capture-max-hmc-events must be >= 0")
    if args.optimizer_state_capture_max_midpoint_windows < 0:
        p.error("--optimizer-state-capture-max-midpoint-windows must be >= 0")
    if args.optimizer_state_capture_max_bytes < 0:
        p.error("--optimizer-state-capture-max-bytes must be >= 0")
    if args.optimizer_state_capture_writer_max_items < 1:
        p.error("--optimizer-state-capture-writer-max-items must be >= 1")
    if args.optimizer_state_capture_writer_max_bytes < 1:
        p.error("--optimizer-state-capture-writer-max-bytes must be >= 1")
    if (
        args.optimizer_state_capture_background_writer
        and args.optimizer_state_capture_dir is None
    ):
        p.error(
            "--optimizer-state-capture-background-writer requires "
            "--optimizer-state-capture-dir"
        )
    return args


def setup_distributed() -> tuple[int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def validate_optimizer_state_capture_args(args) -> None:
    """Fail closed unless a capture run has one exact window/push meaning."""

    if getattr(args, "optimizer_state_capture_dir", None) is None:
        if getattr(args, "optimizer_state_capture_background_writer", False):
            raise RuntimeError(
                "--optimizer-state-capture-background-writer requires "
                "--optimizer-state-capture-dir"
            )
        return
    violations = []
    if args.syncer == "none":
        violations.append("an async --syncer")
    if args.tuning != "lora":
        violations.append("--tuning lora")
    if args.inner_optimizer != "adamw":
        violations.append("--inner-optimizer adamw")
    if args.wire_dtype != "f32":
        violations.append("--wire-dtype f32")
    if args.merge_alpha != 0.0:
        violations.append("--merge-alpha 0")
    if args.inner_control_variate != "none":
        violations.append("--inner-control-variate none")
    if args.debug_broadcast_lag_commits != 0:
        violations.append("--debug-broadcast-lag-commits 0")
    if args.max_reconnects != 0:
        violations.append("--max-reconnects 0")
    if (
        args.fixed_window_microsteps < 2
        or args.fixed_window_microsteps % 2
        or args.fixed_window_tokens != 0
        or args.fixed_window_schedule is not None
    ):
        violations.append("one fixed even --fixed-window-microsteps >= 2")
    if violations:
        raise RuntimeError(
            "--optimizer-state-capture-dir requires native no-scaler fp32 "
            "AdamW with an unambiguous fixed f32 push lifecycle: "
            + ", ".join(violations)
        )


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
    tokenizer = _from_pretrained_offline_first(
        AutoTokenizer, model_id, trust_remote_code=True
    )
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


def apply_control_correction(
    frag,
    params: dict[str, torch.Tensor],
    control_local: torch.Tensor,
    control_mean: torch.Tensor,
    tokens_per_step: float,
    inner_lr: float,
) -> torch.Tensor:
    """SCAFFOLD-lite: add ``(c_i - c) * tokens_per_step / inner_lr`` onto this
    fragment's parameter grads in place (the ``grad - c_i + c`` correction in
    gradient units; see yeto/scaffold.py). Called after backward and before
    grad clipping. Params whose grad is None (no backward this step) are
    skipped. The controls are flat, in the fragment's layout order."""
    delta = grad_correction(control_local, control_mean, tokens_per_step, inner_lr)
    applied = delta.clone()
    off = 0
    for name, numel in frag.tensors:
        p = params[name]
        if p.grad is not None:
            p.grad.add_(delta[off : off + numel].view_as(p).to(p.grad.dtype))
        else:
            applied[off : off + numel].zero_()
        off += numel
    return applied


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


def release_lagged_broadcast(queue: list, item, lag_commits: int):
    """EXP staleness control (--debug-broadcast-lag-commits).

    Append the newly received broadcast to the fragment's FIFO and release
    the one `lag_commits` behind the newest arrival, or None while the queue
    is still warming up. With lag_commits == 0 the item passes straight
    through, preserving current behavior.
    """
    queue.append(item)
    if len(queue) <= lag_commits:
        return None
    return queue.pop(0)


def barrier_release(awaiting: dict[int, int], fid: int, version: int) -> bool:
    """True-lockstep (--barrier-sync) release rule.

    ``awaiting`` maps a fragment id the learner has pushed to the base version
    it pushed from; while the map is non-empty the inner loop takes no steps.
    A broadcast for ``fid`` at ``version`` releases that fragment's barrier iff
    it is strictly newer than the pushed base — i.e. it carries the merge of
    the just-pushed window rather than a stale or duplicate copy (the syncer
    advances a fragment's version on every commit, so the merge of round t is
    always newer than the base t was computed against). Mutates ``awaiting``
    in place and returns whether an entry was released; a no-op (False) when
    the fragment is not awaiting or the broadcast is not newer.
    """
    base = awaiting.get(fid)
    if base is not None and version > base:
        del awaiting[fid]
        return True
    return False


def make_fixed_window_snapshot(
    fragment,
    params: dict[str, torch.Tensor],
    *,
    anchor: torch.Tensor | None,
    c_steps: int,
    c_tokens: int,
    local_step: int,
    base_version: int,
    window_uuid: str | None,
) -> dict:
    """Freeze one immutable response endpoint and its capture lifecycle ID."""

    return {
        "flat": fragment_flat(fragment, params).detach().cpu(),
        "anchor": anchor.clone() if anchor is not None else None,
        "c_steps": int(c_steps),
        "c_tokens": int(c_tokens),
        "local_step": int(local_step),
        "base_version": int(base_version),
        "window_uuid": window_uuid,
    }


def push_audit_from_candidate(candidate: dict) -> PushAudit:
    """Encode the exact local candidate foreign keys into the audited wire."""

    return PushAudit(
        window_uuid=uuid.UUID(candidate["window_uuid"]).bytes,
        attempt_serial=int(candidate["attempt_serial"]),
        payload_sha256=bytes.fromhex(candidate["payload_sha256"]),
    )


def parse_fixed_window_schedule(spec: str) -> list[tuple[int, int]]:
    """Parse ``--fixed-window-schedule`` "commit1:h1,commit2:h2,...".

    Each entry switches the fixed response window to ``h`` microsteps once
    this learner has answered ``commit`` pulls (its LOCAL commit index,
    counted across all fragments). The window's token target follows
    automatically: a window of h microsteps spans h * tokens_per_inner_step
    raw tokens, and every push self-describes its window via c_steps and
    c_tokens, so the syncer needs no schedule of its own.

    Commit indices must be non-negative and strictly increasing; window
    sizes must be >= 1. Raises ValueError on malformed specs.
    """
    entries: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        commit_str, sep, h_str = part.partition(":")
        if not sep:
            raise ValueError(f"schedule entry {part!r} must be 'commit:window'")
        try:
            commit, h = int(commit_str), int(h_str)
        except ValueError as exc:
            raise ValueError(f"schedule entry {part!r} must be integers") from exc
        if commit < 0:
            raise ValueError(f"schedule commit index must be >= 0, got {commit}")
        if h < 1:
            raise ValueError(f"schedule window must be >= 1 microstep, got {h}")
        if entries and commit <= entries[-1][0]:
            raise ValueError(
                f"schedule commit indices must be strictly increasing, "
                f"got {commit} after {entries[-1][0]}"
            )
        entries.append((commit, h))
    if not entries:
        raise ValueError("fixed-window schedule must contain at least one entry")
    return entries


def scheduled_window_steps(
    schedule: list[tuple[int, int]] | None, base_steps: int, local_commits: int
) -> int:
    """Window size (microsteps) in effect after `local_commits` answered
    pulls: the last schedule entry at or before that index, else the base
    window. With no schedule this is always the base — the default path."""
    if not schedule:
        return base_steps
    steps = base_steps
    for commit, h in schedule:
        if commit > local_commits:
            break
        steps = h
    return steps


def invalidate_undersized_snapshots(
    snapshots: list[dict | None], window_steps: int
) -> None:
    """Drop cached fixed-window snapshots that no longer fill the (grown)
    window; they are recaptured once the fragment accumulates enough steps.
    Snapshots that already satisfy the new window stay valid — c_steps and
    c_tokens self-describe every push, so a shrink never strands a pull."""
    for fid, snap in enumerate(snapshots):
        if snap is not None and int(snap["c_steps"]) < window_steps:
            snapshots[fid] = None


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
        self.path = args.probe_log or os.path.join(
            args.output_dir, "fragment_probe.jsonl"
        )
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
                (
                    input_ids.to(device, non_blocking=True),
                    weights.to(device, non_blocking=True),
                )
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

    def note_broadcast(
        self, fid: int, old_anchor: torch.Tensor | None, new_anchor: torch.Tensor
    ) -> None:
        if old_anchor is None or old_anchor.numel() != new_anchor.numel():
            return
        direction = new_anchor.float().cpu() - old_anchor.float().cpu()
        prev = self.momentum[fid]
        self.momentum[fid] = (
            direction if prev is None else 0.85 * prev + 0.15 * direction
        )

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
        self.norm_ema[fid] = (
            norm if prev_mean is None else 0.85 * prev_mean + 0.15 * norm
        )
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
            apply_fragment(
                self.layout.fragments[fid], trial.to(self.device), self.params
            )
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
    process_seed = int(args.seed) + 1009 * int(args.learner_id) + int(rank)
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
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

    validate_optimizer_state_capture_args(args)

    if args.bcmp_shadow_every < 1:
        raise ValueError("--bcmp-shadow-every must be >= 1")
    if args.bcmp_shadow_path is not None:
        if args.syncer == "none":
            raise RuntimeError("--bcmp-shadow-path requires an async syncer run")
        if args.inner_optimizer != "adamw":
            raise RuntimeError("--bcmp-shadow-path requires --inner-optimizer adamw")
        if args.tuning != "lora":
            raise RuntimeError(
                "--bcmp-shadow-path currently requires --tuning lora; full/FSDP "
                "fragment-to-optimizer-state ownership has not been audited"
            )

    scaffold_mode = getattr(args, "inner_control_variate", "none")
    if scaffold_mode != "none":
        if args.syncer == "none":
            raise RuntimeError(
                f"--inner-control-variate {scaffold_mode} requires an async syncer "
                "run (the mean control c is broadcast by the syncer)"
            )
        if world > 1:
            # The control state (c_i, c) and the per-step grad correction live
            # on the syncer-facing rank 0 only; a multi-rank learner would need
            # every rank to hold identical corrected grads before clipping.
            # Not implemented — the scaffold experiments run one process per
            # learner (same constraint as --probe-data / --barrier-sync).
            raise RuntimeError(
                f"--inner-control-variate {scaffold_mode} currently supports "
                f"single-process learners (world size 1); world size is {world}"
            )
        if args.inner_optimizer != "sgd" or args.weight_decay != 0.0:
            log.warning(
                "SCAFFOLD %s with %s/weight_decay=%g is experimental: the "
                "gradient correction is not guaranteed unbiased outside "
                "constant-LR plain SGD",
                scaffold_mode,
                args.inner_optimizer,
                args.weight_decay,
            )
    if args.scaffold_beta <= 0.0:
        raise ValueError(f"--scaffold-beta must be positive, got {args.scaffold_beta}")
    if args.scaffold_control_shuffle and scaffold_mode != "scaffold_full":
        raise ValueError("--scaffold-control-shuffle requires scaffold_full")

    if getattr(args, "scaffold_correctness_mode", False):
        violations = []
        if args.inner_control_variate not in ("scaffold_lite", "scaffold_full"):
            violations.append("a SCAFFOLD --inner-control-variate mode")
        if args.tuning != "lora":
            violations.append("--tuning lora (fp32 trainable parameters)")
        if args.inner_optimizer != "sgd":
            violations.append("--inner-optimizer sgd")
        if args.weight_decay != 0.0:
            violations.append("--weight-decay 0")
        if args.warmup_steps != 0:
            violations.append("--warmup-steps 0")
        if args.grad_clip != 0.0:
            violations.append("--grad-clip 0")
        if args.wire_dtype != "f32":
            violations.append("--wire-dtype f32")
        if args.merge_alpha != 0.0:
            violations.append("--merge-alpha 0")
        if not args.barrier_sync:
            violations.append("--barrier-sync")
        if args.debug_broadcast_lag_commits != 0:
            violations.append("--debug-broadcast-lag-commits 0")
        if (
            args.debug_step_sleep_ms != 0.0
            or args.debug_push_delay_ms != 0.0
            or args.debug_delay_jitter_ms != 0.0
        ):
            violations.append("zero debug step/push delays and jitter")
        if args.max_reconnects != 0:
            violations.append("--max-reconnects 0")
        if not args.scaffold_audit_path:
            violations.append("--scaffold-audit-path")
        if (
            args.fixed_window_microsteps <= 0
            or args.fixed_window_tokens != 0
            or args.fixed_window_schedule is not None
        ):
            violations.append("one fixed positive --fixed-window-microsteps")
        if violations:
            raise RuntimeError(
                "--scaffold-correctness-mode requires " + ", ".join(violations)
            )

    if getattr(args, "barrier_sync", False):
        if args.syncer == "none":
            raise RuntimeError("--barrier-sync requires an async syncer run")
        if world > 1:
            # The block-until-broadcast gate runs on the syncer-facing rank 0
            # only; a multi-rank learner would need every rank to hold the
            # collective in lockstep while rank 0 waits. Not implemented — the
            # barrier experiments run one process per learner.
            raise RuntimeError(
                "--barrier-sync currently supports single-process learners "
                "(world size 1); this learner has world size "
                f"{world}. Use --gpu-slots / one process per learner."
            )

    log.info("loading model %s (%s)", args.model, args.tuning)
    log.info("rng seed=%d", process_seed)
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
        [(n, p.numel()) for n, p in params.items()],
        args.fragments,
        args.fragment_pattern,
        matrix_merge=args.matrix_merge,
        named_shapes={n: tuple(p.shape) for n, p in params.items()},
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
                normalize_param_name(n): p
                for n, p in model.named_parameters()
                if p.requires_grad
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
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index]
        )
        params = trainable_params(model.module)

    if args.inner_optimizer == "sgd":
        opt = torch.optim.SGD(
            params.values(), lr=args.inner_lr, weight_decay=args.weight_decay
        )
    else:
        opt = torch.optim.AdamW(
            params.values(), lr=args.inner_lr, weight_decay=args.weight_decay
        )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            1.0 if args.warmup_steps <= 0 else min(1.0, (s + 1) / args.warmup_steps)
        ),
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
            (host, int(port)),
            args.learner_id,
            layout,
            wire_dtype,
            args.wan_streams,
            max_reconnects=args.max_reconnects,
        )
        client.start()
        log.info("connected to syncer at %s", args.syncer)
        if args.learner_id == 0:
            for fid, frag in enumerate(layout.fragments):
                client.send_init(
                    fid, pack_fragment(frag, params, bulk_dtype(wire_dtype))
                )
            log.info("sent INIT_PARAMS for %d fragments", layout.num_fragments)

    run_inner_loop(
        args,
        model,
        params,
        layout,
        opt,
        sched,
        loader,
        client,
        rank,
        world,
        device,
        tokenizer,
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
    args,
    model,
    params,
    layout,
    opt,
    sched,
    loader,
    client,
    rank,
    world,
    device,
    tokenizer,
):
    # Counters (Alg. 1): incremented for all fragments each step, reset per
    # fragment on receipt. Tracked as global totals + per-fragment snapshots.
    steps_total = 0
    tokens_total = 0
    steps_at_reset = [0] * layout.num_fragments
    tokens_at_reset = [0] * layout.num_fragments
    fragment_versions = [0] * layout.num_fragments  # last applied version per fragment
    lag_commits = max(0, int(getattr(args, "debug_broadcast_lag_commits", 0) or 0))
    lagged_broadcasts: list[list] | None = (
        [[] for _ in range(layout.num_fragments)] if lag_commits > 0 else None
    )
    # Lag-mode determinism (EXP2.29B): post-warmup, a commit must respond
    # with a window trained ENTIRELY on the released (K-stale) base. The
    # apply of a released broadcast resets the window (below), and pulls are
    # held until that apply has happened since the fragment's last push;
    # otherwise a released broadcast racing the window fill leaves rows one
    # extra commit stale (K+1) and windows that straddle the apply.
    lag_released_ever: list[bool] | None = (
        [False] * layout.num_fragments if lag_commits > 0 else None
    )
    lag_applied_since_push: list[bool] | None = (
        [False] * layout.num_fragments if lag_commits > 0 else None
    )
    pending_pulls: list = []  # pulls deferred until c_steps >= 1
    global_step = 0
    # True lockstep DiLoCo (--barrier-sync): after pushing a fragment delta,
    # the learner blocks until the syncer's merged broadcast for that fragment
    # arrives, taking no inner steps in between. `awaiting_broadcast` maps a
    # pushed fragment id to the base version it pushed from; a broadcast for
    # that fragment with a strictly newer version clears the entry, and the
    # inner loop refuses to advance while any entry remains. Empty (and unused)
    # on the default non-barrier path.
    barrier_sync = bool(getattr(args, "barrier_sync", False))
    awaiting_broadcast: dict[int, int] = {}
    awaiting_control: dict[int, int] = {}
    # Q4 pushes are deltas anchored at the last *received* global value per
    # fragment; the utility probe also needs these anchors for bf16/f32 runs
    # so it can evaluate candidate deltas against the learner-known global
    # state. Before any broadcast the anchor is the base-model value, which
    # every learner loads identically (and learner 0 sends as INIT_PARAMS).
    # SCAFFOLD controls are keyed by the completed fragment version so a
    # control that races its parameter broadcast cannot activate early. Lite
    # computes the local half from the pushed endpoint; full Option II keeps the
    # accumulator here and receives the residual mean from the syncer.
    scaffold_mode = getattr(args, "inner_control_variate", "none")
    scaffold_on = scaffold_mode != "none"
    scaffold_full = scaffold_mode == "scaffold_full"
    scaffold_shuffle = bool(getattr(args, "scaffold_control_shuffle", False))
    control_pairs = [VersionedControlPairs() for _ in range(layout.num_fragments)]
    full_residual_pairs = [VersionedControlPairs() for _ in range(layout.num_fragments)]
    full_controls = [
        torch.zeros(frag.numel, dtype=torch.float32, device=device)
        for frag in layout.fragments
    ]
    full_accumulated_versions = [-1] * layout.num_fragments
    anchors: list[torch.Tensor] | None = None
    if (
        rank == 0
        and client is not None
        and (client.dtype == DTYPE_Q4 or args.probe_data is not None or scaffold_on)
    ):
        anchors = [fragment_flat(frag, params).cpu() for frag in layout.fragments]
    # c_tokens counts RAW tokens processed (throughput proxy for merge
    # weighting), not the subset of loss-weighted tokens.
    tokens_per_inner_step = (
        world * args.micro_batch_size * args.grad_accum * args.seq_len
    )
    fixed_window_steps = 1
    if args.fixed_window_microsteps > 0:
        fixed_window_steps = max(fixed_window_steps, args.fixed_window_microsteps)
    if args.fixed_window_tokens > 0:
        fixed_window_steps = max(
            fixed_window_steps,
            math.ceil(args.fixed_window_tokens / max(tokens_per_inner_step, 1)),
        )
    # Online sync-horizon schedule (--fixed-window-schedule): the window in
    # effect is a function of this learner's local commit index (answered
    # pulls, across fragments). None keeps the constant-window path.
    fixed_window_schedule = getattr(args, "fixed_window_schedule", None)
    local_commits = 0
    answered_rounds: set[tuple[int, int]] = set()  # (fid, pull step) dedupe
    base_fixed_window_steps = fixed_window_steps
    fixed_window_steps = scheduled_window_steps(
        fixed_window_schedule, base_fixed_window_steps, local_commits
    )
    fixed_window_enabled = (
        args.fixed_window_microsteps > 0
        or args.fixed_window_tokens > 0
        or fixed_window_schedule is not None
    )
    fixed_window_snapshots: list[dict | None] | None = (
        [None] * layout.num_fragments if fixed_window_enabled else None
    )
    capture_window_uuids: list[str | None] | None = None
    if fixed_window_enabled and rank == 0:
        log.info(
            "fixed response window enabled: %d step(s), %d token(s)/step, "
            "target tokens=%d, target microsteps=%d, schedule=%s",
            fixed_window_steps,
            tokens_per_inner_step,
            args.fixed_window_tokens,
            args.fixed_window_microsteps,
            fixed_window_schedule,
        )

    state_capture = None
    capture_dir = getattr(args, "optimizer_state_capture_dir", None)
    if rank == 0 and client is not None and capture_dir is not None:
        from .optimizer_state_capture import OptimizerStateCapture

        state_capture = OptimizerStateCapture(
            capture_dir,
            params=params,
            layout=layout,
            optimizer=opt,
            scheduler=sched,
            learner_id=args.learner_id,
            rank=rank,
            every=args.optimizer_state_capture_every,
            max_hmc_events=args.optimizer_state_capture_max_hmc_events,
            max_midpoint_windows=args.optimizer_state_capture_max_midpoint_windows,
            max_bytes=args.optimizer_state_capture_max_bytes,
            background_writer=args.optimizer_state_capture_background_writer,
            background_writer_max_items=(args.optimizer_state_capture_writer_max_items),
            background_writer_max_bytes=(args.optimizer_state_capture_writer_max_bytes),
        )
        capture_window_uuids = [None] * layout.num_fragments
        log.info("optimizer-state capture enabled at %s", capture_dir)

    if args.loss_function.startswith("pickle:"):
        compute_loss = load_pickled_loss(args.loss_function)
    elif args.loss_function.startswith("custom:"):
        compute_loss = load_custom_loss(args.loss_function)
    else:
        compute_loss = lambda logits, ids, w: sft_loss(
            logits, ids, args.loss_function, w
        )  # noqa: E731

    probe = None
    if rank == 0 and client is not None and args.probe_data is not None:
        probe = FragmentUtilityProbe(
            args, model, params, layout, tokenizer, device, compute_loss
        )
    bcmp_shadow = None
    if rank == 0 and args.bcmp_shadow_path is not None:
        bcmp_shadow = BCMPShadowTracker(
            args.bcmp_shadow_path,
            every=args.bcmp_shadow_every,
            learner_id=args.learner_id,
            rank=rank,
        )
        log.info(
            "BC-MP shadow enabled: every=%d log=%s",
            args.bcmp_shadow_every,
            args.bcmp_shadow_path,
        )

    def drain_broadcast_actions() -> list:
        """Collect and unpack received global fragments (rank 0), updating
        the probe/anchor and lag-FIFO state exactly as the step boundary
        does. Returns the (fid, version, flat) list still to be applied.
        Shared by the boundary and the --barrier-sync wait so both stay
        bit-identical."""
        acts = []
        for bc in client.drain_updates():
            flat = unpack_fragment(
                layout.fragments[bc.fragment_id], bc.data, bulk_dtype(client.dtype)
            )
            version = bc.version
            if lagged_broadcasts is not None:
                # EXP: explicit version-staleness control. Queue the
                # broadcast; release the one K commits behind the newest
                # arrival for this fragment.
                released = release_lagged_broadcast(
                    lagged_broadcasts[bc.fragment_id],
                    (bc.version, flat),
                    lag_commits,
                )
                if released is None:
                    continue
                version, flat = released
                if lag_released_ever is not None:
                    lag_released_ever[bc.fragment_id] = True
            if anchors is not None:
                if probe is not None:
                    probe.note_broadcast(bc.fragment_id, anchors[bc.fragment_id], flat)
                # The anchor is the raw global value (pre-blend), so the
                # syncer can reconstruct pushes from Θ(version)+δ.
                anchors[bc.fragment_id] = flat.clone()
            acts.append((bc.fragment_id, version, flat))
        return acts

    def apply_broadcast_world1(acts: list) -> None:
        """Apply collected fragments on the single-process path with the
        α-blend, restart each fragment's window, and clear any barrier wait
        the fragment now satisfies (a strictly newer version than the one it
        was pushed from). This is the body of the boundary's world==1 branch,
        reused verbatim by the --barrier-sync wait."""
        nonlocal global_step
        for fid, version, flat in acts:
            flat = flat.to(device)
            if bcmp_shadow is not None:
                bcmp_shadow.note_broadcast(
                    fragment_id=fid,
                    broadcast_version=version,
                    local_step=steps_total,
                    fragment=layout.fragments[fid],
                    params=params,
                    global_flat=flat,
                    merge_alpha=args.merge_alpha,
                )
            if args.merge_alpha > 0:
                local = fragment_flat(layout.fragments[fid], params)
                flat = args.merge_alpha * local + (1.0 - args.merge_alpha) * flat
            apply_fragment(layout.fragments[fid], flat, params)
            fragment_versions[fid] = version
            if scaffold_on:
                control_pairs[fid].discard_before(version)
                if scaffold_full:
                    full_residual_pairs[fid].discard_before(version)
            # Both modes restart the window at apply time. In lag mode this
            # starts the window that must be trained entirely on the
            # just-applied released (K-stale) base; the matching pull is held
            # until this apply.
            steps_at_reset[fid] = steps_total
            tokens_at_reset[fid] = tokens_total
            if fixed_window_snapshots is not None:
                fixed_window_snapshots[fid] = None
            if state_capture is not None:
                capture_window_uuids[fid] = state_capture.note_broadcast(
                    fid,
                    version,
                    local_step=steps_total,
                    tokens_total=tokens_total,
                    window_steps=fixed_window_steps,
                )
            if lag_applied_since_push is not None:
                lag_applied_since_push[fid] = True
            global_step = max(global_step, version)
            # Barrier release: this fragment's merge has come back, so the
            # learner may resume once every pushed fragment is released.
            if awaiting_broadcast:
                barrier_release(awaiting_broadcast, fid, version)
            target = awaiting_control.get(fid)
            if target == version and control_pairs[fid].get(version) is not None:
                awaiting_control.pop(fid, None)

    def activate_full_control(fid: int, version: int) -> None:
        """Accumulate one version exactly once after its residual pair arrives."""
        if not scaffold_full or version <= full_accumulated_versions[fid]:
            return
        pair = full_residual_pairs[fid].get(version)
        if pair is None:
            return
        residual, residual_mean = pair
        full_controls[fid] = accumulate_control(
            full_controls[fid], residual, residual_mean, args.scaffold_beta
        ).to(device)
        control_pairs[fid].add_local(version, full_controls[fid])
        control_pairs[fid].add_mean(version, torch.zeros_like(full_controls[fid]))
        full_accumulated_versions[fid] = version

    def drain_scaffold_controls() -> None:
        """Decode control broadcasts without ever cross-pairing versions."""
        if not scaffold_on:
            return
        if scaffold_shuffle:
            controls = client.drain_control_pairs()
        else:
            controls = client.drain_controls()
        for ctrl in controls:
            frag = layout.fragments[ctrl.fragment_id]
            if scaffold_shuffle:
                residual = unpack_fragment(
                    frag, ctrl.local_data, bulk_dtype(client.dtype)
                ).to(device)
                residual_mean = unpack_fragment(
                    frag, ctrl.mean_data, bulk_dtype(client.dtype)
                ).to(device)
                full_residual_pairs[ctrl.fragment_id].add_local(ctrl.version, residual)
                full_residual_pairs[ctrl.fragment_id].add_mean(
                    ctrl.version, residual_mean
                )
                activate_full_control(ctrl.fragment_id, ctrl.version)
            else:
                mean = unpack_fragment(frag, ctrl.data, bulk_dtype(client.dtype)).to(
                    device
                )
                if scaffold_full:
                    full_residual_pairs[ctrl.fragment_id].add_mean(ctrl.version, mean)
                    activate_full_control(ctrl.fragment_id, ctrl.version)
                else:
                    control_pairs[ctrl.fragment_id].add_mean(ctrl.version, mean)
            target = awaiting_control.get(ctrl.fragment_id)
            if (
                target == ctrl.version
                and fragment_versions[ctrl.fragment_id] == ctrl.version
                and control_pairs[ctrl.fragment_id].get(ctrl.version) is not None
            ):
                awaiting_control.pop(ctrl.fragment_id, None)

    shutdown = False
    scaffold_audit_written = False
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
            audit_capture = bool(
                args.scaffold_correctness_mode
                and not scaffold_audit_written
                and all(
                    control_pairs[fid].get(fragment_versions[fid]) is not None
                    for fid in range(layout.num_fragments)
                )
            )
            audit_corrections: list[torch.Tensor] = []
            if scaffold_on:
                # Correct each fragment's grad by (c_i - c)
                # before clipping. Skips a fragment until both its local
                # control (set at push time) and the syncer's mean control
                # (received via BCAST_CONTROL) exist, so the first window per
                # fragment runs uncorrected — exactly the state SCAFFOLD starts
                # from. world==1 here (validated in main), so no cross-rank
                # grad reconciliation is needed.
                for fid, frag in enumerate(layout.fragments):
                    pair = control_pairs[fid].get(fragment_versions[fid])
                    if pair is not None:
                        c_i, c = pair
                        applied_correction = apply_control_correction(
                            frag, params, c_i, c, tokens_per_inner_step, args.inner_lr
                        )
                        if audit_capture:
                            audit_corrections.append(applied_correction.detach().cpu())
            if audit_capture:
                correction_flat = torch.cat(audit_corrections)
                corrected_grad_flat = torch.cat(
                    [
                        (
                            params[name].grad.detach().reshape(-1).float().cpu()
                            if params[name].grad is not None
                            else torch.zeros(numel)
                        )
                        for frag in layout.fragments
                        for name, numel in frag.tensors
                    ]
                )
                raw_grad_flat = corrected_grad_flat - correction_flat
                params_before_audit = torch.cat(
                    [fragment_flat(frag, params).cpu() for frag in layout.fragments]
                )
                audit_lr = float(opt.param_groups[0]["lr"])
            if args.tuning == "lora":
                # The adapters are never grad-synced by a wrapper — fsdp+lora
                # ignores them, the replicated path has no wrapper — so average
                # them across ranks before clipping (no-op at world==1). After
                # this the replicated params/grads are identical on every rank,
                # so a plain clip over them is correct (the frozen base
                # contributes no grads).
                allreduce_trainable_grads(params.values(), world)
                clip_total_norm = None
                if args.grad_clip > 0.0:
                    clip_total_norm = torch.nn.utils.clip_grad_norm_(
                        params.values(), args.grad_clip
                    )
            elif args.shard == "fsdp":
                clip_total_norm = None
                if args.grad_clip > 0.0:
                    clip_total_norm = model.clip_grad_norm_(args.grad_clip)
            else:
                clip_total_norm = None
                if args.grad_clip > 0.0:
                    clip_total_norm = torch.nn.utils.clip_grad_norm_(
                        params.values(), args.grad_clip
                    )
            if bcmp_shadow is not None:
                bcmp_shadow.before_optimizer_step(
                    layout=layout,
                    params=params,
                    optimizer=opt,
                    local_step=steps_total,
                )
            if state_capture is not None:
                state_capture.capture_first_post_broadcast_gradients(
                    local_step_before_update=steps_total,
                    tokens_total=tokens_total,
                    clip_total_norm=clip_total_norm,
                    clip_max_norm=args.grad_clip if args.grad_clip > 0.0 else None,
                )
            opt.step()
            if audit_capture:
                params_after_audit = torch.cat(
                    [fragment_flat(frag, params).cpu() for frag in layout.fragments]
                )
                actual_displacement = params_after_audit - params_before_audit
                correction_displacement = actual_displacement + audit_lr * raw_grad_flat
                audit_record = {
                    "learner_id": int(args.learner_id),
                    "local_step": int(steps_total),
                    "correction_before_clip": correction_flat,
                    "actual_displacement_after_step": actual_displacement,
                    "correction_displacement_after_step": correction_displacement,
                }
                audit_path = os.path.abspath(args.scaffold_audit_path)
                os.makedirs(os.path.dirname(audit_path), exist_ok=True)
                torch.save(audit_record, audit_path)
                log.info(
                    "SCAFFOLD correctness audit learner=%d local_step=%d "
                    "before_clip_correction_l2=%.9g after_step_correction_displacement_l2=%.9g",
                    args.learner_id,
                    steps_total,
                    correction_flat.norm().item(),
                    correction_displacement.norm().item(),
                )
                scaffold_audit_written = True
            sched.step()
            opt.zero_grad(set_to_none=True)
            steps_total += 1
            tokens_total += tokens_per_inner_step
            if state_capture is not None:
                state_capture.after_optimizer_step(
                    local_step=steps_total,
                    tokens_total=tokens_total,
                    current_window_steps=fixed_window_steps,
                )
            step_jitter_ms = (
                args.debug_delay_jitter_ms if args.debug_step_sleep_ms > 0.0 else 0.0
            )
            _debug_sleep(args.debug_step_sleep_ms, step_jitter_ms)

            if rank == 0 and fixed_window_snapshots is not None:
                for snap_fid, snap in enumerate(fixed_window_snapshots):
                    if snap is not None:
                        continue
                    snap_steps = steps_total - steps_at_reset[snap_fid]
                    if snap_steps < fixed_window_steps:
                        continue
                    fixed_window_snapshots[snap_fid] = make_fixed_window_snapshot(
                        layout.fragments[snap_fid],
                        params,
                        anchor=(anchors[snap_fid] if anchors is not None else None),
                        c_steps=snap_steps,
                        c_tokens=tokens_total - tokens_at_reset[snap_fid],
                        local_step=steps_total,
                        base_version=fragment_versions[snap_fid],
                        window_uuid=(
                            capture_window_uuids[snap_fid]
                            if capture_window_uuids is not None
                            else None
                        ),
                    )

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
                actions = drain_broadcast_actions()
                drain_scaffold_controls()
                shutdown = client.shutdown.is_set()

            if world > 1:
                meta = [(f, v) for f, v, _ in actions] if rank == 0 else None
                box = [meta, shutdown]
                dist.broadcast_object_list(box, src=0)
                meta, shutdown = box
                if rank != 0:
                    actions = [
                        (f, v, torch.empty(layout.fragments[f].numel)) for f, v in meta
                    ]
                for fid, version, flat in actions:
                    flat = flat.to(device)
                    dist.broadcast(flat, src=0)
                    if bcmp_shadow is not None:
                        bcmp_shadow.note_broadcast(
                            fragment_id=fid,
                            broadcast_version=version,
                            local_step=steps_total,
                            fragment=layout.fragments[fid],
                            params=params,
                            global_flat=flat,
                            merge_alpha=args.merge_alpha,
                        )
                    # α-blend: keep a share of the inner steps taken while the
                    # merge was in flight. Ranks hold identical params, so
                    # blending after the broadcast stays consistent.
                    if args.merge_alpha > 0:
                        local = fragment_flat(layout.fragments[fid], params)
                        flat = (
                            args.merge_alpha * local + (1.0 - args.merge_alpha) * flat
                        )
                    apply_fragment(layout.fragments[fid], flat, params)
                    if rank == 0:
                        fragment_versions[fid] = version
                        # Both modes restart the window at apply time. In lag
                        # mode this starts the window that must be trained
                        # entirely on the just-applied released (K-stale)
                        # base; the matching pull is held until this apply.
                        steps_at_reset[fid] = steps_total
                        tokens_at_reset[fid] = tokens_total
                        if fixed_window_snapshots is not None:
                            fixed_window_snapshots[fid] = None
                        if state_capture is not None:
                            capture_window_uuids[fid] = state_capture.note_broadcast(
                                fid,
                                version,
                                local_step=steps_total,
                                tokens_total=tokens_total,
                                window_steps=fixed_window_steps,
                            )
                        if lag_applied_since_push is not None:
                            lag_applied_since_push[fid] = True
                    global_step = max(global_step, version)
            else:
                apply_broadcast_world1(actions)

            # 2. answer pulls whose fragment has made progress since the
            # (just-applied) broadcasts.
            if rank == 0 and client is not None:
                pending_pulls.extend(client.drain_pulls())
                still_pending = []
                for pull in pending_pulls:
                    fid = pull.fragment_id
                    if (
                        lag_applied_since_push is not None
                        and lag_released_ever is not None
                        and lag_released_ever[fid]
                        and not lag_applied_since_push[fid]
                    ):
                        # Lag mode, post-warmup: hold the pull until the
                        # released broadcast for the previous round has been
                        # applied (which restarts the window), so the
                        # answered window is a full fresh window on the
                        # K-stale base with a deterministic base_version.
                        still_pending.append(pull)
                        continue
                    if fixed_window_snapshots is not None:
                        snap = fixed_window_snapshots[fid]
                        if snap is None:
                            still_pending.append(pull)
                            continue
                        local_flat = snap["flat"]
                        base_anchor_for_push = snap["anchor"]
                        c_steps = int(snap["c_steps"])
                        c_tokens = int(snap["c_tokens"])
                        local_step_for_push = int(snap["local_step"])
                        base_version_for_push = int(snap["base_version"])
                        window_uuid_for_push = snap.get("window_uuid")
                    else:
                        c_steps = steps_total - steps_at_reset[fid]
                        if c_steps < 1:
                            still_pending.append(pull)
                            continue
                        c_tokens = tokens_total - tokens_at_reset[fid]
                        local_flat = (
                            fragment_flat(layout.fragments[fid], params).detach().cpu()
                        )
                        base_anchor_for_push = (
                            anchors[fid].clone() if anchors is not None else None
                        )
                        local_step_for_push = steps_total
                        base_version_for_push = fragment_versions[fid]
                        window_uuid_for_push = None
                    if probe is not None:
                        if anchors is None:
                            raise RuntimeError(
                                "fragment utility probe requires anchors"
                            )
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
                    if base_anchor_for_push is not None and client.dtype == DTYPE_Q4:
                        delta = local_flat - base_anchor_for_push
                        payload = quantize_q4(delta)
                    else:
                        payload = pack_flat(local_flat, client.dtype)
                    _debug_sleep(args.debug_push_delay_ms, args.debug_delay_jitter_ms)
                    push_audit = None
                    if state_capture is not None and window_uuid_for_push is not None:
                        candidate = state_capture.note_push(
                            window_uuid=window_uuid_for_push,
                            fragment_id=fid,
                            pull_global_step=pull.global_step,
                            base_version=base_version_for_push,
                            local_step=local_step_for_push,
                            c_steps=c_steps,
                            c_tokens=c_tokens,
                            wire_codec=args.wire_dtype,
                            payload=payload,
                        )
                        push_audit = push_audit_from_candidate(candidate)
                    push_enqueued = client.push_fragment(
                        fid,
                        pull.global_step,
                        base_version_for_push,
                        local_step_for_push,
                        c_steps,
                        c_tokens,
                        payload,
                        audit=push_audit,
                    )
                    if state_capture is not None and not push_enqueued:
                        raise RuntimeError(
                            "audited capture push was not enqueued; reconnect is disabled"
                        )
                    if state_capture is not None and push_audit is not None:
                        state_capture.note_push_enqueued(push_audit.attempt_serial)
                    if scaffold_on and anchors is not None and c_tokens > 0:
                        # Compute c_i from the exact per-push base anchor and
                        # the endpoint after the same wire codec the syncer
                        # consumes. This is essential for the exact-mean identity
                        # (and is exact, not merely close, on the f32 correctness
                        # wire).
                        if base_anchor_for_push is None:
                            raise RuntimeError(
                                "SCAFFOLD push is missing its base anchor"
                            )
                        if client.dtype == DTYPE_Q4:
                            transmitted_endpoint = base_anchor_for_push + dequantize_q4(
                                payload, layout.fragments[fid].numel
                            )
                        else:
                            transmitted_endpoint = unpack_fragment(
                                layout.fragments[fid], payload, client.dtype
                            )
                        residual = local_control(
                            base_anchor_for_push, transmitted_endpoint, c_tokens
                        ).to(device)
                        if scaffold_full:
                            if not scaffold_shuffle:
                                full_residual_pairs[fid].add_local(
                                    pull.global_step, residual
                                )
                                activate_full_control(fid, pull.global_step)
                        else:
                            control_pairs[fid].add_local(pull.global_step, residual)
                    if barrier_sync:
                        # True lockstep: record that this fragment's merge is
                        # in flight. The inner loop will not step again until a
                        # broadcast newer than base_version_for_push lands and
                        # apply_broadcast_world1 clears the entry. Re-pushes of
                        # the same round just refresh the same base version.
                        awaiting_broadcast[fid] = base_version_for_push
                        if scaffold_on:
                            awaiting_control[fid] = pull.global_step
                    if lagged_broadcasts is not None:
                        # Lag mode: restart the fixed window at push time so
                        # every commit carries a fresh full window even while
                        # its broadcast is still queued. Anchor the restart
                        # at the CURRENT local step (matching the non-lag
                        # broadcast-time reset), not at the snapshot's step:
                        # backdating to local_step_for_push let a late pull
                        # leave steps_total - steps_at_reset already past the
                        # next window, materializing oversized windows
                        # (c_steps > target; EXP2.29B k1 gate failure).
                        # Steps trained between snapshot and push drop out of
                        # window accounting, exactly as in non-lag mode.
                        steps_at_reset[fid] = steps_total
                        tokens_at_reset[fid] = tokens_total
                        if fixed_window_snapshots is not None:
                            fixed_window_snapshots[fid] = None
                        if lag_applied_since_push is not None:
                            lag_applied_since_push[fid] = False
                    if fixed_window_schedule is not None:
                        # Advance the local commit index once per distinct
                        # round (re-sent pulls answer again but don't count)
                        # and switch the window when the schedule says so.
                        round_key = (fid, pull.global_step)
                        if round_key not in answered_rounds:
                            answered_rounds.add(round_key)
                            local_commits += 1
                            new_window = scheduled_window_steps(
                                fixed_window_schedule,
                                base_fixed_window_steps,
                                local_commits,
                            )
                            if new_window != fixed_window_steps:
                                log.info(
                                    "fixed-window schedule: local commit %d "
                                    "switches window %d -> %d microstep(s)",
                                    local_commits,
                                    fixed_window_steps,
                                    new_window,
                                )
                                fixed_window_steps = new_window
                                if fixed_window_snapshots is not None:
                                    invalidate_undersized_snapshots(
                                        fixed_window_snapshots, new_window
                                    )
                pending_pulls = still_pending

                # --- true lockstep barrier (--barrier-sync) -----------------
                # Having answered this round's pull(s), HARD-BLOCK until the
                # syncer's merged broadcast for every pushed fragment has
                # arrived and been applied. No inner optimizer steps run while
                # a merge is in flight (original DiLoCo's worker barrier,
                # arXiv 2311.08105); the next window then starts from the
                # merged global. drain/apply reuse the boundary helpers so the
                # applied state is bit-identical to a broadcast picked up at a
                # step boundary. A no-op unless --barrier-sync pushed above.
                while (
                    barrier_sync
                    and (awaiting_broadcast or awaiting_control)
                    and not shutdown
                ):
                    client.check_health()
                    waited = drain_broadcast_actions()
                    drain_scaffold_controls()
                    if waited:
                        apply_broadcast_world1(waited)
                    else:
                        time.sleep(0.002)
                    shutdown = client.shutdown.is_set()

            if shutdown or steps_total >= args.max_local_steps:
                break
        epoch += 1
    if bcmp_shadow is not None:
        bcmp_shadow.close(local_step=steps_total)
    if state_capture is not None:
        state_capture.close()
    log.info(
        "inner loop done at local_step=%d global_step=%d", steps_total, global_step
    )


if __name__ == "__main__":
    main()
