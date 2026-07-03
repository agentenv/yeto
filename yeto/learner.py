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
import time

import torch
import torch.distributed as dist

from .data import StreamingPackedBlocks, build_packed_dataset
from .fragments import FragmentLayout, build_layout
from .losses import load_custom_loss, load_pickled_loss, sft_loss
from .protocol import DTYPE_BF16, DTYPE_F32, SyncerClient
from .tensor_io import apply_fragment, pack_fragment, unpack_fragment

log = logging.getLogger("learner")

MODEL_ALIASES = {
    "gemma4": "google/gemma-4-12B-it",
    "deepseek4flash": "deepseek-ai/DeepSeek-V4-Flash",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Yeto learner")
    p.add_argument("--model", required=True, help="HF model id or alias (gemma4|deepseek4flash)")
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
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--fragments", type=int, default=8, help="P (= H, round-robin)")
    p.add_argument("--wire-dtype", choices=["bf16", "f32"], default="bf16")
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
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def load_model_and_tokenizer(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_ALIASES.get(args.model, args.model)
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
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    if args.tuning == "lora":
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    model.to(device)
    return model, tokenizer


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

    log.info("loading model %s (%s)", args.model, args.tuning)
    model, tokenizer = load_model_and_tokenizer(args, device)
    params = trainable_params(model)
    layout = build_layout([(n, p.numel()) for n, p in params.items()], args.fragments)
    log.info(
        "%d trainable tensors -> %d fragments (%.1f MB total)",
        len(params),
        layout.num_fragments,
        sum(p.numel() for p in params.values()) * 2 / 1e6,
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

    peft_model = None  # unwrapped peft handle, kept for fsdp+lora save
    if args.shard == "fsdp":
        import functools

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
        if args.tuning == "lora":
            # Hybrid sharding: only the frozen bf16 base is sharded. The LoRA
            # adapter params go in ignored_states, so FSDP never flattens or
            # shards them — they stay ordinary replicated per-rank fp32
            # tensors, and fragment pack/apply/INIT works on them unchanged.
            # No MixedPrecision: the base is already bf16 and frozen; the
            # adapters keep fp32 grads and optimizer state. FSDP also never
            # reduces the ignored params' grads, so run_inner_loop all-reduces
            # them at each optimizer-step boundary.
            peft_model = model
            model = FSDP(
                model,
                auto_wrap_policy=wrap_policy,
                use_orig_params=True,
                ignored_states=list(params.values()),
                device_id=device,
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
    elif world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        params = trainable_params(model.module)

    opt = torch.optim.AdamW(params.values(), lr=args.inner_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps))
    )

    wire_dtype = DTYPE_BF16 if args.wire_dtype == "bf16" else DTYPE_F32
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
                client.send_init(fid, pack_fragment(frag, params, wire_dtype))
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
            if args.shard == "fsdp":
                # fsdp+lora: the adapters are replicated ordinary tensors in
                # `params`, so hand save_pretrained an explicit state dict
                # through the unwrapped peft handle — no sharded base param
                # is ever gathered or touched.
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


def run_inner_loop(args, model, params, layout, opt, sched, loader, client, rank, world, device):
    # Counters (Alg. 1): incremented for all fragments each step, reset per
    # fragment on receipt. Tracked as global totals + per-fragment snapshots.
    steps_total = 0
    tokens_total = 0
    steps_at_reset = [0] * layout.num_fragments
    tokens_at_reset = [0] * layout.num_fragments
    fragment_versions = [0] * layout.num_fragments  # last applied version per fragment
    pending_pulls: list = []  # pulls deferred until c_steps >= 1
    global_step = 0
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
            if args.shard == "fsdp" and args.tuning == "lora":
                # The adapters are FSDP-ignored, so their grads were never
                # reduced; average them across ranks before clipping. After
                # this the replicated params/grads are identical on every
                # rank, so a plain clip over them is correct (the sharded
                # base is frozen and contributes no grads).
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
            actions = []  # (fid, version, flat_f32) applied this boundary
            if rank == 0 and client is not None:
                client.check_health()
                # 1. answer pulls whose fragment has made progress
                pending_pulls.extend(client.drain_pulls())
                still_pending = []
                for pull in pending_pulls:
                    fid = pull.fragment_id
                    c_steps = steps_total - steps_at_reset[fid]
                    if c_steps < 1:
                        still_pending.append(pull)
                        continue
                    c_tokens = tokens_total - tokens_at_reset[fid]
                    client.push_fragment(
                        fid,
                        pull.global_step,
                        fragment_versions[fid],
                        steps_total,
                        c_steps,
                        c_tokens,
                        pack_fragment(layout.fragments[fid], params, client.dtype),
                    )
                pending_pulls = still_pending
                # 2. apply received global fragments
                for bc in client.drain_updates():
                    flat = unpack_fragment(layout.fragments[bc.fragment_id], bc.data, client.dtype)
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
                    apply_fragment(layout.fragments[fid], flat, params)
                    if rank == 0:
                        steps_at_reset[fid] = steps_total
                        tokens_at_reset[fid] = tokens_total
                        fragment_versions[fid] = version
                    global_step = max(global_step, version)
            else:
                for fid, version, flat in actions:
                    apply_fragment(layout.fragments[fid], flat.to(device), params)
                    steps_at_reset[fid] = steps_total
                    tokens_at_reset[fid] = tokens_total
                    fragment_versions[fid] = version
                    global_step = max(global_step, version)

            if shutdown or steps_total >= args.max_local_steps:
                break
        epoch += 1
    log.info("inner loop done at local_step=%d global_step=%d", steps_total, global_step)


if __name__ == "__main__":
    main()
