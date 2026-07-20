"""Megatron-Core island learner — a peer of yeto.learner that distributes
the frozen base with expert/tensor/pipeline parallelism instead of FSDP2,
while speaking the exact same DiLoCo adapter sync to the Rust syncer.

FIRST IMPLEMENTATION, NOT YET HARDWARE-VALIDATED. Megatron-Core is
GPU/multi-node only, so this cannot be unit-tested; it is written against the
researched Megatron-Core / Megatron-Bridge API (see docs/MEGATRON.md) and
needs a live multi-node B200 run to validate and iterate — exactly as the
torch backend needed the gemma4 smokes.

Scope of this first cut: TP=1, PP=1, EP=N (the natural "fill the island with
expert parallelism" default). In that regime, with attention/dense LoRA
targets and share_expert_adapters=True, every trainable adapter is REPLICATED
on every rank, so the DiLoCo fragment layout is identical to the torch
backend's and the sync reuses yeto's primitives unchanged. TP>1 (adapters
TP-sharded) and PP>1 (adapters split across pipeline stages) need cross-
parallel adapter gather before sync — guarded below as an explicit error
rather than silently producing wrong merges.
"""

from __future__ import annotations

import argparse
import logging
import os

log = logging.getLogger("megatron-learner")

# Megatron-parallel module names LoRA-A/B attach to. Fused attention
# (linear_qkv/linear_proj) plus DeepSeek MLA's split projections; the
# ModuleMatcher only matches names that exist, so listing both is safe.
_ATTENTION_TARGETS = [
    "linear_qkv",
    "linear_proj",
    "linear_q_proj",
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
]
# Dense-MLP GEMMs; adding these LoRAs a dense model's MLP. For MoE these are
# the routed-expert GEMMs, which we leave frozen (attention-only is the MoE
# fine-tuning recipe), so they are only added when --lora-targets all-linear.
_MLP_TARGETS = ["linear_fc1", "linear_fc2"]


def parse_args(argv=None):
    p = argparse.ArgumentParser("yeto.megatron.learner")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--syncer", default="none", help="host:port or 'none'")
    p.add_argument("--learner-id", type=int, default=0)
    p.add_argument("--num-learners", type=int, default=1)
    p.add_argument("--loss-function", default="cross_entropy")
    p.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    p.add_argument(
        "--assistant-mask-mode",
        choices=["native", "legacy"],
        default="native",
        help="assistant-only masking: tokenizer-native exact mask or explicit legacy format",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tuning", choices=["lora", "full"], default="lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", choices=["auto", "attention", "all-linear"], default="auto")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch-size", default="1")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--inner-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-local-steps", type=int, default=10_000)
    p.add_argument("--fragments", type=int, default=8)
    p.add_argument("--fragment-pattern", choices=["binpack", "strided"], default="binpack")
    p.add_argument("--merge-alpha", type=float, default=0.5)
    p.add_argument("--tokenize", choices=["stream", "preload"], default="stream")
    p.add_argument("--stream-workers", type=int, default=2)
    p.add_argument("--wire-dtype", choices=["bf16", "f32", "q4"], default="q4")
    p.add_argument("--wan-streams", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--output-dir", default="~/yeto-output")
    # Megatron parallelism.
    p.add_argument("--island-backend", default="megatron")
    p.add_argument("--expert-parallel", type=int, default=1)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--pipeline-parallel", type=int, default=1)
    return p.parse_args(argv)


def _init_distributed(args):
    """torch.distributed first, then Megatron parallel state, then the
    model-parallel RNG (the order Megatron-Core asserts)."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_parallel,
        pipeline_model_parallel_size=args.pipeline_parallel,
        expert_model_parallel_size=args.expert_parallel,
        expert_tensor_parallel_size=1,  # pure EP over experts; never split an expert tensor
    )
    model_parallel_cuda_manual_seed(args.seed)
    if args.tensor_parallel != 1 or args.pipeline_parallel != 1:
        raise NotImplementedError(
            "the Megatron backend's adapter sync currently assumes TP=1, PP=1 "
            "(adapters fully replicated per rank). TP>1 needs a TP all-gather "
            "of linear_in/linear_out and PP>1 a cross-stage gather before the "
            "DiLoCo push — implement those before enabling."
        )
    return dist.get_rank(), dist.get_world_size(), local_rank


def _build_model(args, device):
    """HF bf16 checkpoint -> Megatron-Core model (EP-sharded) -> LoRA."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA

    from ..models import resolve

    model_id = resolve(args.model)
    log.info("importing %s into Megatron-Core (EP=%d)", model_id, args.expert_parallel)
    bridge = AutoBridge.from_hf_pretrained(model_id, trust_remote_code=True)
    model = bridge.to_megatron_model(load_weights=True)  # list[MegatronModule], one per VP chunk

    targets = list(_ATTENTION_TARGETS)
    if args.lora_targets == "all-linear":
        targets += _MLP_TARGETS
    peft = LoRA(
        dim=args.lora_r,               # Bridge names the rank `dim`
        alpha=args.lora_alpha,
        target_modules=targets,
        share_expert_adapters=True,    # one adapter per EP rank -> replicated, syncable
    )
    model = peft(model, training=True)  # freezes base, attaches + trains adapters
    return model, bridge


def _adapter_params(model):
    """Canonical {name: tensor} of the trainable LoRA adapters. Bridge names
    them linear_in (A) / linear_out (B); with TP=1/PP=1 they are replicated on
    every rank, so the local view IS the canonical value."""
    out = {}
    for chunk in model:
        module = getattr(chunk, "module", chunk)  # unwrap mcore DDP
        for name, p in module.named_parameters():
            if p.requires_grad and (
                ".adapter" in name or name.endswith((".linear_in.weight", ".linear_out.weight"))
            ):
                out[name] = p
    return out


def main(argv=None):
    import torch
    import torch.distributed as dist
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel as DDP,
    )
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    from ..fragments import build_layout
    from ..protocol import DTYPE_BF16, DTYPE_F32, DTYPE_Q4, SyncerClient, bulk_dtype
    from ..tensor_io import (
        apply_fragment,
        fragment_flat,
        pack_fragment,
        pack_tensor,
        quantize_q4,
        unpack_fragment,
    )

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s megatron %(levelname)s %(message)s")
    rank, world, local_rank = _init_distributed(args)
    device = torch.device("cuda", local_rank)

    model, bridge = _build_model(args, device)
    params = _adapter_params(model)
    layout = build_layout(
        [(n, p.numel()) for n, p in params.items()],
        args.fragments,
        args.fragment_pattern,
        named_shapes={n: tuple(p.shape) for n, p in params.items()},
    )
    log.info("%d adapter tensors -> %d fragments", len(params), layout.num_fragments)

    # mcore DDP wrap so grads land in the distributed optimizer's buckets.
    ddp_cfg = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_grad_reduce=True, grad_reduce_in_fp32=True
    )
    cfg = getattr(model[0], "config", None) or getattr(getattr(model[0], "module", None), "config")
    cfg.finalize_model_grads_func = finalize_model_grads
    model = [DDP(config=cfg, ddp_config=ddp_cfg, module=m) for m in model]
    opt = get_megatron_optimizer(
        config=OptimizerConfig(
            optimizer="adam",
            lr=args.inner_lr,
            weight_decay=args.weight_decay,
            use_distributed_optimizer=True,
            bf16=True,
            clip_grad=1.0,
        ),
        model_chunks=model,
    )
    forward_backward = get_forward_backward_func()

    # DiLoCo sync setup — identical to the torch learner: rank 0 drives the
    # syncer, sends INIT, and the fragment protocol reuses yeto primitives.
    wire_dtype = {"bf16": DTYPE_BF16, "f32": DTYPE_F32, "q4": DTYPE_Q4}[args.wire_dtype]
    client = None
    if rank == 0 and args.syncer != "none":
        host, port = args.syncer.rsplit(":", 1)
        client = SyncerClient((host, int(port)), args.learner_id, layout, wire_dtype, args.wan_streams)
        client.start()
        log.info("connected to syncer at %s", args.syncer)
        if args.learner_id == 0:
            for fid, frag in enumerate(layout.fragments):
                client.send_init(fid, pack_fragment(frag, params, bulk_dtype(wire_dtype)))
            log.info("sent INIT_PARAMS for %d fragments", layout.num_fragments)

    _run_inner_loop(
        args, model, params, layout, opt, forward_backward, client, rank, world, device,
        fragment_flat=fragment_flat, pack_tensor=pack_tensor, quantize_q4=quantize_q4,
        unpack_fragment=unpack_fragment, apply_fragment=apply_fragment,
        bulk_dtype=bulk_dtype, DTYPE_Q4=DTYPE_Q4,
    )

    if rank == 0:
        save_dir = os.path.expanduser(args.output_dir)
        os.makedirs(save_dir, exist_ok=True)
        # Export adapters back to an HF-loadable checkpoint via the bridge.
        bridge.save_hf_pretrained(model, save_dir)
        log.info("saved adapters to %s", save_dir)
        if client is not None:
            client.close()
    dist.barrier()
    dist.destroy_process_group()


def _run_inner_loop(
    args, model, params, layout, opt, forward_backward, client, rank, world, device,
    *, fragment_flat, pack_tensor, quantize_q4, unpack_fragment, apply_fragment,
    bulk_dtype, DTYPE_Q4,
):
    """N inner Megatron steps, pausing at each step boundary to run the DiLoCo
    fragment sync. The inner step differs from the torch learner (Megatron's
    forward_backward schedule vs a manual loop), but the sync half mirrors
    yeto.learner.run_inner_loop's protocol and counter semantics exactly."""
    import torch
    import torch.distributed as dist

    from ..finalization import finalize_torch_island
    mbs = 1 if args.micro_batch_size == "auto" else int(args.micro_batch_size)
    tokenizer = _load_tokenizer(args)
    dataset = _packed_blocks(args, tokenizer)
    data_iter = _cycle(dataset, micro_batch_size=mbs)

    def forward_step(it, mdl):
        batch = next(it)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        pos = torch.arange(ids.size(1), device=device).unsqueeze(0).expand_as(ids)
        out = mdl(input_ids=ids, position_ids=pos, attention_mask=None, labels=labels)

        def loss_func(output):
            token_losses = output.float().reshape(-1)
            flat_mask = loss_mask.float().reshape(-1)
            if token_losses.numel() != flat_mask.numel():
                raise RuntimeError(
                    "Megatron returned a non-tokenwise LM loss: "
                    f"{token_losses.numel()} losses for {flat_mask.numel()} mask values"
                )
            loss = (token_losses * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
            return loss, {"lm loss": loss.detach()}

        return out, loss_func

    # Counters (Alg. 1): global totals + per-fragment snapshots reset on receipt.
    steps_total = tokens_total = 0
    steps_at_reset = [0] * layout.num_fragments
    tokens_at_reset = [0] * layout.num_fragments
    fragment_versions = [0] * layout.num_fragments
    pending_pulls: list = []
    global_step = 0
    tokens_per_inner_step = world * mbs * args.grad_accum * args.seq_len
    anchors = [None] * layout.num_fragments if rank == 0 and client is not None else None
    shutdown = False

    while not shutdown and steps_total < args.max_local_steps:
        for m in model:
            m.zero_grad_buffer()
        opt.zero_grad()
        forward_backward(
            forward_step_func=forward_step,
            data_iterator=data_iter,
            model=model,
            num_microbatches=args.grad_accum,
            seq_length=args.seq_len,
            micro_batch_size=mbs,
            forward_only=False,
        )
        opt.step()  # grads reduced across DP/EP inside finalize_model_grads
        steps_total += 1
        tokens_total += tokens_per_inner_step

        # --- fragment sync at the step boundary (never blocks) ---
        # Broadcasts apply BEFORE pulls are answered (see yeto/learner.py:
        # a pipelined syncer's next pull can overtake the previous round's
        # broadcast; answering first would push a stale base_version).
        actions = []
        finalizing = False
        if rank == 0 and client is not None:
            client.check_health()
            finalizing = client.finalizing.is_set()
            if not finalizing:
                for bc in client.drain_updates():
                    flat = unpack_fragment(
                        layout.fragments[bc.fragment_id], bc.data, bulk_dtype(client.dtype)
                    )
                    if anchors is not None:
                        anchors[bc.fragment_id] = flat.clone()
                    actions.append((bc.fragment_id, bc.version, flat))
            shutdown = client.shutdown.is_set()

        if world > 1:
            meta = [(f, v) for f, v, _ in actions] if rank == 0 else None
            box = [meta, shutdown, finalizing]
            dist.broadcast_object_list(box, src=0)
            meta, shutdown, finalizing = box
            if rank != 0:
                actions = [(f, v, torch.empty(layout.fragments[f].numel)) for f, v in meta]
        if finalizing:
            manifest = finalize_torch_island(
                client,
                layout,
                params,
                rank=rank,
                world=world,
                device=device,
            )
            global_step = max(global_step, manifest.global_step)
            shutdown = True
            break
        for fid, version, flat in actions:
            flat = flat.to(device)
            if world > 1:
                dist.broadcast(flat, src=0)
            if args.merge_alpha > 0:
                local = fragment_flat(layout.fragments[fid], params)
                flat = args.merge_alpha * local + (1.0 - args.merge_alpha) * flat
            apply_fragment(layout.fragments[fid], flat, params)
            steps_at_reset[fid] = steps_total
            tokens_at_reset[fid] = tokens_total
            fragment_versions[fid] = version
            global_step = max(global_step, version)

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
                    fid, pull.global_step, pull.round_attempt,
                    fragment_versions[fid], steps_total,
                    c_steps, c_tokens, payload,
                )
            pending_pulls = still_pending
    if rank == 0 and client is not None and not client.finalized.is_set():
        raise RuntimeError(
            "Megatron learner stopped before authoritative finalization; "
            "refusing to save local parameters"
        )
    log.info("inner loop done at local_step=%d global_step=%d", steps_total, global_step)


def _load_tokenizer(args):
    """Load the HF tokenizer whose native template defines SFT masking."""
    from transformers import AutoTokenizer

    from ..learner import _from_pretrained_offline_first
    from ..models import resolve

    return _from_pretrained_offline_first(
        AutoTokenizer,
        resolve(args.model),
        trust_remote_code=True,
    )


def _packed_blocks(args, tokenizer):
    """Build Megatron's shared EP input stream with the requested mask mode.

    The currently supported Megatron topology is pure expert parallelism
    (TP=PP=1), so every rank must consume the same tokens. Streaming therefore
    uses a single logical data consumer on every rank instead of rank-sharding
    the rows as the torch data-parallel learner does.
    """
    from ..data import StreamingPackedBlocks, build_packed_dataset

    common = {
        "train_on": args.train_on,
        "assistant_mask_mode": args.assistant_mask_mode,
    }
    if args.tokenize == "stream":
        return StreamingPackedBlocks(
            args.data,
            tokenizer,
            args.learner_id,
            args.num_learners,
            args.seq_len,
            args.max_rows,
            rank=0,
            world=1,
            **common,
        )
    return build_packed_dataset(
        args.data,
        tokenizer,
        args.learner_id,
        args.num_learners,
        args.seq_len,
        args.max_rows,
        **common,
    )


def _cycle(dataset, micro_batch_size: int = 1):
    """Endless shifted LM micro-batches from yeto's weighted packed blocks.

    Megatron computes one loss per supplied label position rather than doing
    the causal shift performed by ``yeto.losses.sft_loss``. Shift labels and
    weights here so weight[t] still controls prediction of token t. The final
    position receives a valid dummy label and zero loss weight.
    """
    import torch

    ids_batch = []
    weights_batch = []
    while True:
        for block in dataset:
            if isinstance(block, dict):
                ids = block["input_ids"]
                weights = block.get("weights")
            else:
                ids, weights = block
            ids = torch.as_tensor(ids)
            weights = (
                torch.ones_like(ids, dtype=torch.float32)
                if weights is None
                else torch.as_tensor(weights, dtype=torch.float32)
            )
            ids_batch.append(ids)
            weights_batch.append(weights)
            if len(ids_batch) < micro_batch_size:
                continue

            input_ids = torch.stack(ids_batch)
            weights = torch.stack(weights_batch)
            labels = torch.zeros_like(input_ids)
            labels[:, :-1] = input_ids[:, 1:]
            loss_mask = torch.zeros_like(weights)
            loss_mask[:, :-1] = weights[:, 1:]
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "loss_mask": loss_mask,
            }
            ids_batch.clear()
            weights_batch.clear()


if __name__ == "__main__":
    main()
