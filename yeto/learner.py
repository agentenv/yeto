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
import logging
import os
import random
import time

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
    pack_fragment,
    pack_tensor,
    quantize_q4,
    unpack_fragment,
)

log = logging.getLogger("learner")

from .models import MODEL_ALIASES  # single source; see yeto/models.py


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
        "--base-quantization",
        choices=["none", "nf4"],
        default="none",
        help="frozen-base storage for LoRA; nf4 enables bitsandbytes QLoRA "
        "and requires CUDA with --shard ddp",
    )
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
    p.add_argument("--seed", type=int, default=None)
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
        choices=["rda", "iso"],
        default="rda",
        help="syncer aggregation for non-embedding (matrix) fragments: "
        "rda = weighted radial-directional averaging (default); "
        "iso = Iso-C-style isotropic aggregation (IsoLoCo, arXiv 2607.03011) "
        "— average the per-tensor deltas, then flatten each averaged "
        "matrix's singular-value spectrum to its mean; non-2D tensors join "
        "the direct-averaged fragment",
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
        help="tensor encoding on the WAN; every push is a base-relative "
        "delta, q4 block-quantizes it as 4-bit E3M0 (broadcasts stay bf16)",
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
    return p.parse_args(argv)


def setup_distributed() -> tuple[int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _derived_training_seed(root: int, learner_id: int, learners: int, rank: int) -> int:
    """Map an island-local rank to its matching baseline-mM rank seed."""
    return root + learner_id + learners * rank


def _stream_seed(root: int, learner_id: int, learners: int, rank: int,
                 workers: int) -> int:
    """Dataset seed before StreamingPackedBlocks adds its consumer index."""
    if workers == 0:
        return root + learner_id + (learners - 1) * rank
    return root + learner_id * 1_000_003


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


def _prepare_nf4_base_for_lora(model) -> None:
    """Freeze an NF4 base without expanding large bf16 tensors to fp32.

    PEFT's generic k-bit helper casts every non-quantized bf16 parameter to
    fp32. On large-vocabulary models that doubles several gigabytes of frozen
    embeddings and lm-head weights. Only normalization weights need the fp32
    stability treatment; checkpointing input gradients are enabled later,
    after LoRA attachment.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if "norm" in module.__class__.__name__.lower():
            module.to(torch.float32)


def load_model_and_tokenizer(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .models import resolve

    base_quantization = getattr(args, "base_quantization", "none")
    if base_quantization != "none":
        if args.tuning != "lora":
            raise ValueError("--base-quantization requires --tuning lora")
        if device.type != "cuda":
            raise ValueError("--base-quantization nf4 requires CUDA")
        if args.shard != "ddp":
            raise ValueError(
                "--base-quantization nf4 keeps one frozen base per rank; "
                "use --shard ddp"
            )

    # The normal path loads the base in bf16 (all-float, so FSDP2 can shard
    # it). NF4 is an explicit QLoRA profile for islands where the frozen bf16
    # base does not fit on one GPU; the trainable adapters remain fp32.
    model_id = resolve(args.model)
    # fsdp+full: originals stay fp32 (uniform dtype for flat-param groups,
    # fp32 optimizer state) and MixedPrecision computes/communicates in bf16.
    # ddp/single and fsdp+lora: frozen base in bf16; peft leaves LoRA
    # adapters in fp32, which keeps AdamW's exp_avg_sq in fp32 — a bf16
    # second moment is too noisy. Wire packing casts to the wire dtype
    # either way.
    if (args.shard == "fsdp" and args.tuning == "full") or device.type != "cuda":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    tokenizer = _from_pretrained_offline_first(AutoTokenizer, model_id, trust_remote_code=True)
    if base_quantization == "nf4":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = _from_pretrained_offline_first(
            AutoModelForCausalLM,
            model_id,
            quantization_config=quantization_config,
            device_map={"": device.index if device.index is not None else 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        model = _from_pretrained_offline_first(
            AutoModelForCausalLM,
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    if args.tuning == "lora":
        from peft import LoraConfig, get_peft_model

        if base_quantization == "nf4":
            _prepare_nf4_base_for_lora(model)

        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=resolve_lora_targets(
                getattr(args, "lora_targets", "auto"), model.config
            ),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    if base_quantization == "none":
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

    if not 0.0 <= args.merge_alpha < 1.0:
        raise ValueError(f"--merge-alpha must be in [0, 1), got {args.merge_alpha}")

    # An explicit root seed gives every rank/island the same LoRA
    # initialization. With no seed, preserve the learner's original ambient
    # RNG behavior outside benchmark runs.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    log.info("loading model %s (%s)", args.model, args.tuning)
    model, tokenizer = load_model_and_tokenizer(args, device)

    # Training randomness may differ after initialization while remaining
    # reproducible for a given benchmark seed and topology.
    training_seed = None
    if args.seed is not None:
        # learner + M*rank is the corresponding rank in baseline-mM. This
        # pairs dropout and zero-worker streaming order across the matching
        # synchronous and DiLoCo topologies.
        training_seed = _derived_training_seed(
            args.seed, args.learner_id, args.num_learners, rank
        )
        random.seed(training_seed)
        torch.manual_seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_seed)

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
        stream_kwargs = {}
        if args.seed is not None:
            # With zero workers, StreamingPackedBlocks adds `rank`
            # internally, yielding seed + learner + M*rank.
            stream_kwargs["seed"] = _stream_seed(
                args.seed,
                args.learner_id,
                args.num_learners,
                rank,
                args.stream_workers,
            )
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
            **stream_kwargs,
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

            sampler_kwargs = {}
            if args.seed is not None:
                sampler_kwargs["seed"] = args.seed + args.learner_id * 1_000_003
            sampler = DistributedSampler(
                dataset,
                num_replicas=world,
                rank=rank,
                **sampler_kwargs,
            )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
            generator=(
                torch.Generator().manual_seed(training_seed)
                if training_seed is not None
                else None
            ),
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

    run_inner_loop(args, model, params, layout, opt, sched, loader, client, rank, world, device)

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
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def run_inner_loop(args, model, params, layout, opt, sched, loader, client, rank, world, device):
    # Counters (Alg. 1): incremented for all fragments each step, reset per
    # fragment on receipt. Tracked as global totals + per-fragment snapshots.
    steps_total = 0
    tokens_total = 0
    target_tokens_total = torch.zeros((), dtype=torch.long, device=device)
    steps_at_reset = [0] * layout.num_fragments
    tokens_at_reset = [0] * layout.num_fragments
    fragment_versions = [0] * layout.num_fragments  # last applied version per fragment
    pending_pulls: list = []  # pulls deferred until c_steps >= 1
    global_step = 0
    # Every PUSH carries local − raw_anchor, where raw_anchor is the exact
    # global fragment from the last broadcast (before optional alpha blending).
    # PULL can overtake the initial striped broadcast, so an unset anchor
    # defers that response instead of guessing from local initialization.
    anchors: list[torch.Tensor | None] | None = None
    if rank == 0 and client is not None:
        anchors = [None] * layout.num_fragments
    # c_tokens counts RAW tokens processed (throughput proxy for merge
    # weighting), not the subset of loss-weighted tokens.
    tokens_per_inner_step = world * args.micro_batch_size * args.grad_accum * args.seq_len

    if args.loss_function.startswith("pickle:"):
        compute_loss = load_pickled_loss(args.loss_function)
    elif args.loss_function.startswith("custom:"):
        compute_loss = load_custom_loss(args.loss_function)
    else:
        compute_loss = lambda logits, ids, w: sft_loss(logits, ids, args.loss_function, w)  # noqa: E731

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
            loss, batch_target_tokens = compute_loss(out.logits, input_ids, weights)
            target_tokens_total.add_(
                batch_target_tokens.detach().to(device=device, dtype=torch.long)
            )
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
                        # Keep the exact raw global value (pre-blend); every
                        # later push is local minus this anchor.
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
                    global_step = max(global_step, version)

            # 2. answer pulls whose fragment has made progress since the
            # (just-applied) broadcasts.
            if rank == 0 and client is not None:
                pending_pulls.extend(client.drain_pulls())
                still_pending = []
                for pull in pending_pulls:
                    fid = pull.fragment_id
                    c_steps = steps_total - steps_at_reset[fid]
                    if c_steps < 1:
                        still_pending.append(pull)
                        continue
                    c_tokens = tokens_total - tokens_at_reset[fid]
                    anchor = anchors[fid] if anchors is not None else None
                    if anchor is None:
                        still_pending.append(pull)
                        continue
                    delta = fragment_flat(layout.fragments[fid], params).cpu() - anchor
                    if client.dtype == DTYPE_Q4:
                        payload = quantize_q4(delta)
                    else:
                        payload = pack_tensor(delta, client.dtype)
                    client.push_fragment(
                        fid,
                        pull.global_step,
                        pull.round_attempt,
                        fragment_versions[fid],
                        steps_total,
                        c_steps,
                        c_tokens,
                        payload,
                    )
                pending_pulls = still_pending

            if shutdown or steps_total >= args.max_local_steps:
                break
        epoch += 1
    raw_tokens_local = (
        steps_total
        * args.micro_batch_size
        * args.grad_accum
        * args.seq_len
    )
    log.info(
        "inner loop done at local_step=%d global_step=%d raw_tokens=%d target_tokens=%d",
        steps_total,
        global_step,
        raw_tokens_local,
        int(target_tokens_total.item()),
    )


if __name__ == "__main__":
    main()
