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
import json
import logging
import os
import shutil

log = logging.getLogger("megatron-learner")
MEGATRON_ADAPTER_METADATA_FILE = "yeto_megatron_adapter.json"
MEGATRON_ADAPTER_WEIGHTS_BASENAME = "megatron_adapter_model"

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
    p.add_argument("--model-revision", default=None)
    p.add_argument("--data-revision", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--syncer", default="none", help="host:port or 'none'")
    p.add_argument("--learner-id", type=int, default=0)
    p.add_argument("--num-learners", type=int, default=1)
    p.add_argument("--loss-function", default="cross_entropy")
    p.add_argument("--allow-unsafe-pickled-loss", action="store_true")
    p.add_argument("--loss-sha256", default=None, help=argparse.SUPPRESS)
    p.add_argument("--source-sha256", default=None, help=argparse.SUPPRESS)
    for provenance_flag in (
        "model-requested-identifier",
        "model-requested-revision",
        "data-requested-identifier",
        "data-requested-revision",
    ):
        p.add_argument(f"--{provenance_flag}", default=None, help=argparse.SUPPRESS)
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
    import inspect

    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA

    from ..models import resolve
    from ..provenance import model_load_kwargs

    model_id = resolve(args.model)
    log.info("importing %s into Megatron-Core (EP=%d)", model_id, args.expert_parallel)
    bridge = AutoBridge.from_hf_pretrained(
        model_id,
        use_safetensors=True,
        **model_load_kwargs(args),
    )
    # Yeto wraps with mcore DDP after adapter attachment so trainable LoRA
    # params land in optimizer buckets. Bridge may otherwise require an
    # explicit DDP config before adapters exist.
    model = bridge.to_megatron_model(
        load_weights=True,
        wrap_with_ddp=False,
    )  # list[MegatronModule], one per VP chunk

    targets = list(_ATTENTION_TARGETS)
    if args.lora_targets == "all-linear":
        targets += _MLP_TARGETS
    lora_kwargs = {
        "dim": args.lora_r,  # Bridge names the rank `dim`
        "alpha": args.lora_alpha,
        "target_modules": targets,
    }
    if "share_expert_adapters" in inspect.signature(LoRA).parameters:
        # Newer Megatron-Core uses this for replicated expert adapters. Older
        # containers do not expose it, and dense models like Qwen3-8B do not
        # need it for this smoke path.
        lora_kwargs["share_expert_adapters"] = True
    peft = LoRA(**lora_kwargs)
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


def _save_tensor_state(state, save_dir):
    try:
        from safetensors.torch import save_file

        filename = f"{MEGATRON_ADAPTER_WEIGHTS_BASENAME}.safetensors"
        save_file(state, os.path.join(save_dir, filename))
        return filename, "safetensors"
    except Exception as exc:
        import torch

        filename = f"{MEGATRON_ADAPTER_WEIGHTS_BASENAME}.pt"
        torch.save(state, os.path.join(save_dir, filename))
        log.warning(
            "safetensors save unavailable for Megatron adapter artifact; "
            "used torch.save: %s",
            exc,
        )
        return filename, "torch"


def _save_megatron_adapter_artifact(args, model, output_dir):
    from transformers import AutoTokenizer

    from ..models import resolve
    from ..provenance import model_load_kwargs

    save_dir = os.path.expanduser(output_dir)
    os.makedirs(save_dir, exist_ok=True)
    params = _adapter_params(model)
    state = {name: param.detach().cpu().contiguous() for name, param in params.items()}
    weights_file, weights_format = _save_tensor_state(state, save_dir)

    targets = list(_ATTENTION_TARGETS)
    if args.lora_targets == "all-linear":
        targets += _MLP_TARGETS
    base_model = resolve(args.model)
    metadata = {
        "kind": "yeto.megatron.adapter",
        "schema_version": 1,
        "base_model_name_or_path": base_model,
        "tuning": args.tuning,
        "weights_file": weights_file,
        "weights_format": weights_format,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "targets": args.lora_targets,
            "target_modules": targets,
        },
        "parallelism": {
            "expert": args.expert_parallel,
            "tensor": args.tensor_parallel,
            "pipeline": args.pipeline_parallel,
        },
        "parameter_names": sorted(state),
    }
    with open(os.path.join(save_dir, MEGATRON_ADAPTER_METADATA_FILE), "w") as handle:
        json.dump(metadata, handle, indent=2)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            **model_load_kwargs(args),
        )
        tokenizer.save_pretrained(save_dir)
    except Exception as exc:
        log.warning("could not save tokenizer with Megatron adapter artifact: %s", exc)

    log.info("saved Megatron adapter artifact to %s", save_dir)
    return True


def _save_output_best_effort(bridge, model, output_dir, args=None):
    save_dir = os.path.expanduser(output_dir)
    os.makedirs(save_dir, exist_ok=True)
    if args is not None and args.tuning == "lora":
        log.info("writing Yeto Megatron adapter artifact without Bridge HF export")
        return _save_megatron_adapter_artifact(args, model, save_dir)

    bridge_tmp = os.path.join(save_dir, ".bridge-export-tmp")
    shutil.rmtree(bridge_tmp, ignore_errors=True)
    os.makedirs(bridge_tmp, exist_ok=True)
    try:
        # Export full Megatron models back through Bridge when it supports the
        # selected architecture. Keep partial files out of the final output.
        bridge.save_hf_pretrained(model, bridge_tmp)
    except Exception as exc:
        shutil.rmtree(bridge_tmp, ignore_errors=True)
        log.warning(
            "Megatron-Bridge HF export failed after training; leaving run "
            "successful so validation can proceed: %s",
            exc,
        )
        return False
    for name in os.listdir(bridge_tmp):
        src = os.path.join(bridge_tmp, name)
        dst = os.path.join(save_dir, name)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
    shutil.rmtree(bridge_tmp, ignore_errors=True)
    log.info("saved Megatron output to %s", save_dir)
    return True


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
        pack_fragment,
    )

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s megatron %(levelname)s %(message)s")
    rank, world, local_rank = _init_distributed(args)
    from ..provenance import (
        pin_distributed_runtime_provenance,
        verify_distributed_source_tree_sha256,
    )

    verify_distributed_source_tree_sha256(
        args.source_sha256,
        rank=rank,
        world=world,
    )
    pin_distributed_runtime_provenance(args, rank=rank, world=world)
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
        apply_fragment=apply_fragment,
    )

    # Avoid collectives in the shutdown/save path. Megatron's distributed
    # optimizer may still have bookkeeping collectives in flight on some
    # versions, and an extra barrier here can trip NCCL's watchdog after the
    # tiny validation loop. Rank 0 saves the replicated adapter artifact.
    if rank == 0:
        saved = _save_output_best_effort(bridge, model, args.output_dir, args)
        if saved:
            from ..provenance import write_provenance_manifest

            write_provenance_manifest(
                os.path.expanduser(args.output_dir),
                args,
                artifact_kind="megatron-causal-lm-training-output",
            )
        if client is not None:
            client.close()
    dist.destroy_process_group()


def _run_inner_loop(
    args, model, params, layout, opt, forward_backward, client, rank, world, device,
    *, apply_fragment,
):
    """N inner Megatron steps, pausing at each step boundary to run the DiLoCo
    fragment sync. The inner step differs from the torch learner (Megatron's
    forward_backward schedule vs a manual loop), but the sync half mirrors
    yeto.learner.run_inner_loop's protocol and counter semantics exactly."""
    import torch

    from ..diloco_sync import DiLoCoSyncState, sync_diloco_boundary
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
    tokens_per_inner_step = world * mbs * args.grad_accum * args.seq_len
    sync_state = DiLoCoSyncState.create(
        layout.num_fragments,
        track_anchors=rank == 0 and client is not None,
    )

    def snapshot_sync_params():
        return params

    def apply_sync_flat(fragment, flat):
        apply_fragment(fragment, flat, params)

    def finalize_sync():
        return finalize_torch_island(
            client,
            layout,
            params,
            rank=rank,
            world=world,
            device=device,
        )

    while not sync_state.shutdown and steps_total < args.max_local_steps:
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
        if sync_diloco_boundary(
            client,
            layout,
            sync_state,
            steps_total=steps_total,
            units_total=tokens_total,
            merge_alpha=args.merge_alpha,
            snapshot_params=snapshot_sync_params,
            apply_flat=apply_sync_flat,
            finalize=finalize_sync,
            rank=rank,
            world=world,
            device=device,
            reset_counters_on_replicas=True,
        ):
            break
    if rank == 0 and client is not None and not client.finalized.is_set():
        raise RuntimeError(
            "Megatron learner stopped before authoritative finalization; "
            "refusing to save local parameters"
        )
    log.info(
        "inner loop done at local_step=%d global_step=%d",
        steps_total,
        sync_state.global_step,
    )


def _load_tokenizer(args):
    """Load the HF tokenizer whose native template defines SFT masking."""
    from transformers import AutoTokenizer

    from ..learner import _from_pretrained_offline_first
    from ..models import resolve
    from ..provenance import model_load_kwargs

    return _from_pretrained_offline_first(
        AutoTokenizer,
        resolve(args.model),
        **model_load_kwargs(args),
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
    if getattr(args, "data_revision", None):
        common["revision"] = args.data_revision
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
