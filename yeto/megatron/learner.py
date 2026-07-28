"""Megatron-Core island learner — a peer of yeto.learner that distributes
models with expert/tensor/pipeline parallelism instead of FSDP2, while
speaking the same DiLoCo adapter sync to the Rust syncer.

The full-parameter Qwen3.6-27B path is hardware-validated on two 8xH200 nodes
with TP=8 and PP=2, including distributed Hugging Face checkpoint export.

Scope: TP=1, EP=N with PP=1 for synced DiLoCo runs and model parallelism for
local full-parameter runs. With attention/dense LoRA targets and
share_expert_adapters=True, adapters are replicated across EP ranks and split
across PP stages. The local/no-sync path gathers those PP-stage adapter
tensors for export. Synced PP still needs a global fragment layout and
cross-stage push/pull ownership, so it remains guarded instead of silently
producing wrong merges.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    p.add_argument("--syncer", default="none", help="host:port or 'none'")
    p.add_argument("--learner-id", type=int, default=0)
    p.add_argument("--num-learners", type=int, default=1)
    p.add_argument("--loss-function", default="cross_entropy")
    p.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    p.add_argument("--tuning", choices=["lora", "full"], default="lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", choices=["auto", "attention", "all-linear"], default="auto")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch-size", default="1")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--inner-lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument(
        "--lr-decay-style",
        choices=["constant", "linear", "cosine"],
        default="cosine",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-local-steps", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Dataset passes; when set, overrides --max-local-steps using packed blocks",
    )
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


def _validate_parallelism(args):
    if args.tuning == "full":
        if args.syncer != "none":
            raise NotImplementedError(
                "Megatron full-parameter tuning is currently supported only with "
                "--syncer none. Full-parameter DiLoCo needs a full-model sync protocol."
            )
        return

    if args.tensor_parallel != 1:
        raise NotImplementedError(
            "the Megatron backend currently assumes TP=1 for adapter tensors. "
            "TP>1 needs a TP all-gather of linear_in/linear_out before sync/export."
        )
    if args.pipeline_parallel != 1 and args.syncer != "none":
        raise NotImplementedError(
            "PP>1 is currently supported only for --syncer none validation runs. "
            "DiLoCo sync needs a global PP fragment layout and cross-stage "
            "push/pull ownership before it can be enabled safely."
        )


def _init_distributed(args):
    """torch.distributed first, then Megatron parallel state, then the
    model-parallel RNG (the order Megatron-Core asserts)."""
    _validate_parallelism(args)

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
    model_parallel_cuda_manual_seed(1234)
    return dist.get_rank(), dist.get_world_size(), local_rank


def _maybe_apply_model_specific_parallel_layout(args, bridge):
    """Let Bridge install model-specific PP layouts when it exposes a helper.

    DeepSeek-V4 needs an explicit PP layout because its hash-routed layers must
    co-locate with embeddings. Bridge has grown a helper for this; keep this
    best-effort so older containers still run Qwen/PP smoke tests unchanged.
    """
    if args.pipeline_parallel <= 1 or "deepseek" not in args.model.lower():
        return
    try:
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
            set_deepseek_v4_pipeline_model_parallel_layout,
        )
    except Exception as e:
        log.info("DeepSeek-V4 PP layout helper unavailable: %s", e)
        return

    candidates = [
        bridge,
        getattr(bridge, "provider", None),
        getattr(bridge, "model_provider", None),
        getattr(bridge, "model", None),
        getattr(bridge, "config", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            set_deepseek_v4_pipeline_model_parallel_layout(candidate)
            log.info("applied DeepSeek-V4 pipeline layout for PP=%d", args.pipeline_parallel)
            return
        except Exception:
            continue
    log.warning(
        "DeepSeek-V4 PP>1 requested, but Bridge did not expose a provider/config "
        "that accepted set_deepseek_v4_pipeline_model_parallel_layout"
    )


def _set_trainable(model, trainable: bool):
    for chunk in model:
        module = getattr(chunk, "module", chunk)
        for p in module.parameters():
            p.requires_grad_(trainable)


def _patch_transformers_bridge_compat():
    """Paper over narrow Transformers/Bridge symbol drift.

    Megatron-Bridge 0.5.x imports Ernie-VL registration modules when importing
    megatron.bridge, even for Qwen. Transformers 5.2.0 spells several Ernie
    classes with ``VL_Moe`` while Bridge imports ``VLMoe``. Alias them before
    Bridge registration runs so unrelated Ernie imports do not block Qwen.
    """
    try:
        from transformers.models.ernie4_5_vl_moe import modeling_ernie4_5_vl_moe as ernie
    except Exception:
        return
    for name in dir(ernie):
        if "VL_Moe" not in name:
            continue
        compat_name = name.replace("VL_Moe", "VLMoe")
        if not hasattr(ernie, compat_name):
            setattr(ernie, compat_name, getattr(ernie, name))


def _build_model(args, device):
    """HF bf16 checkpoint -> Megatron-Core model, then optional LoRA."""

    _patch_transformers_bridge_compat()

    from megatron.bridge import AutoBridge

    from ..models import resolve

    model_id = resolve(args.model)
    log.info("importing %s into Megatron-Core (EP=%d, tuning=%s)", model_id, args.expert_parallel, args.tuning)
    bridge = AutoBridge.from_hf_pretrained(model_id, trust_remote_code=True)
    _maybe_apply_model_specific_parallel_layout(args, bridge)
    # Yeto wraps with mcore DDP after adapter attachment so the trainable LoRA
    # params land in the optimizer buckets. Some Bridge versions expose
    # wrap_with_ddp in the signature, others accept it through **kwargs.
    to_megatron_kwargs = {"load_weights": True, "wrap_with_ddp": False}
    model = bridge.to_megatron_model(**to_megatron_kwargs)  # list[MegatronModule], one per VP chunk

    if args.tuning == "full":
        _set_trainable(model, True)
        return model, bridge

    import inspect

    from megatron.bridge.peft.lora import LoRA

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


def _trainable_params(model):
    out = {}
    for chunk in model:
        module = getattr(chunk, "module", chunk)
        for name, p in module.named_parameters():
            if p.requires_grad:
                out[name] = p
    return out


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


def _weighted_token_loss(output, weights):
    """Reduce Megatron's per-token losses using Yeto's assistant mask."""
    if not output.dim():
        return output
    if output.numel() != weights.numel():
        raise ValueError(
            f"loss/mask size mismatch: {output.numel()} losses for {weights.numel()} weights"
        )
    loss_weights = weights.reshape(output.shape).to(output.dtype)
    return (output * loss_weights).sum() / loss_weights.sum().clamp_min(1)


def _prepare_model_config(model, finalize_model_grads, pipeline_parallel):
    """Install Yeto's training hooks and required pipeline metadata."""
    cfg = getattr(model[0], "config", None) or getattr(
        getattr(model[0], "module", None), "config"
    )
    cfg.finalize_model_grads_func = finalize_model_grads
    if pipeline_parallel > 1 and getattr(cfg, "pipeline_dtype", None) is None:
        pipeline_dtype = getattr(cfg, "params_dtype", None)
        if pipeline_dtype is None:
            raise RuntimeError("Megatron PP requires model config.params_dtype")
        cfg.pipeline_dtype = pipeline_dtype
    return cfg


def _build_dataset(args):
    from transformers import AutoTokenizer

    from ..data import build_packed_dataset
    from ..models import resolve

    tokenizer = AutoTokenizer.from_pretrained(resolve(args.model), trust_remote_code=True)
    return build_packed_dataset(
        args.data,
        tokenizer,
        args.learner_id,
        args.num_learners,
        args.seq_len,
        args.max_rows,
        train_on=args.train_on,
    )


def _save_tensor_state(state, save_dir):
    try:
        from safetensors.torch import save_file

        filename = f"{MEGATRON_ADAPTER_WEIGHTS_BASENAME}.safetensors"
        save_file(state, os.path.join(save_dir, filename))
        return filename, "safetensors"
    except Exception as e:
        import torch

        filename = f"{MEGATRON_ADAPTER_WEIGHTS_BASENAME}.pt"
        torch.save(state, os.path.join(save_dir, filename))
        log.warning("safetensors save unavailable for Megatron adapter artifact; used torch.save: %s", e)
        return filename, "torch"


def _save_megatron_adapter_artifact(args, model, output_dir, state_override=None):
    from transformers import AutoTokenizer

    from ..models import resolve

    save_dir = os.path.expanduser(output_dir)
    os.makedirs(save_dir, exist_ok=True)
    params = _adapter_params(model)
    state = (
        {n: t.detach().cpu().contiguous() for n, t in state_override.items()}
        if state_override is not None
        else {n: p.detach().cpu().contiguous() for n, p in params.items()}
    )
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
        "export": {
            "pipeline_stage_gathered": bool(args.pipeline_parallel != 1 and state_override is not None),
        },
        "parameter_names": sorted(state),
    }
    with open(os.path.join(save_dir, MEGATRON_ADAPTER_METADATA_FILE), "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(save_dir)
    except Exception as e:
        log.warning("could not save tokenizer with Megatron adapter artifact: %s", e)

    log.info("saved Megatron adapter artifact to %s", save_dir)
    return True


def _parallel_rank(getter_name, default=0):
    try:
        from megatron.core import parallel_state

        getter = getattr(parallel_state, getter_name, None)
        if getter is None:
            return default
        return int(getter())
    except Exception:
        return default


def _pipeline_stage_roles():
    try:
        from megatron.core import parallel_state

        return (
            bool(parallel_state.is_pipeline_first_stage()),
            bool(parallel_state.is_pipeline_last_stage()),
        )
    except Exception:
        return True, True


def _pipeline_forward_kwargs(input_ids, position_ids, labels, is_last_stage):
    """Build model inputs shared by dense and multimodal Megatron wrappers."""
    return {
        # Qwen3-VL/Qwen3.5 recomputes MRoPE positions from token IDs on every
        # pipeline stage, including stages that do not own the embedding.
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
        "labels": labels if is_last_stage else None,
    }


def _training_steps(args, packed_blocks):
    if args.epochs is None:
        return args.max_local_steps
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be greater than zero")
    return max(1, math.ceil(packed_blocks * args.epochs / args.grad_accum))


def _scheduler_kwargs(args, max_steps):
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    if args.min_lr < 0 or args.min_lr > args.inner_lr:
        raise ValueError("--min-lr must be between zero and --inner-lr")
    warmup_steps = min(args.warmup_steps, max(0, max_steps - 1))
    return {
        "init_lr": 0.0 if warmup_steps else args.inner_lr,
        "max_lr": args.inner_lr,
        "min_lr": args.min_lr,
        "lr_warmup_steps": warmup_steps,
        "lr_decay_steps": max_steps,
        "lr_decay_style": args.lr_decay_style,
        "start_wd": args.weight_decay,
        "end_wd": args.weight_decay,
        "wd_incr_steps": max_steps,
        "wd_incr_style": "constant",
        "use_checkpoint_opt_param_scheduler": False,
        "override_opt_param_scheduler": False,
    }


def _optimizer_config_kwargs(args):
    return {
        "optimizer": "adam",
        "lr": args.inner_lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "use_distributed_optimizer": True,
        "bf16": True,
        "clip_grad": 1.0,
    }


def _adapter_state_for_export(model):
    return {n: p.detach().cpu().contiguous() for n, p in _adapter_params(model).items()}


def _gather_adapter_state_for_export(args, model, rank, world):
    """Return the canonical adapter state on rank 0.

    PP stages own disjoint layer ranges, while attention LoRA is replicated
    across EP ranks when share_expert_adapters=True. For local validation runs,
    gather one representative EP/TP rank per PP stage onto rank 0 for the final
    Yeto adapter artifact. Synced PP runs are still guarded in _init_distributed.
    """
    if args.pipeline_parallel == 1 or world == 1:
        return _adapter_state_for_export(model) if rank == 0 else None

    import torch.distributed as dist

    tp_rank = _parallel_rank("get_tensor_model_parallel_rank")
    ep_rank = _parallel_rank("get_expert_model_parallel_rank")
    pp_rank = _parallel_rank("get_pipeline_model_parallel_rank")
    payload = None
    if tp_rank == 0 and ep_rank == 0:
        payload = {
            "pipeline_rank": pp_rank,
            "state": _adapter_state_for_export(model),
        }

    gathered = [None] * world
    dist.all_gather_object(gathered, payload)
    if rank != 0:
        return None

    merged = {}
    seen_pp = set()
    for item in gathered:
        if not item:
            continue
        pp = item["pipeline_rank"]
        if pp in seen_pp:
            continue
        seen_pp.add(pp)
        for name, tensor in item["state"].items():
            if name in merged:
                if tuple(merged[name].shape) != tuple(tensor.shape):
                    raise RuntimeError(
                        "duplicate Megatron adapter tensor with mismatched shape "
                        f"during PP export: {name}"
                    )
                continue
            merged[name] = tensor
    if len(seen_pp) != args.pipeline_parallel:
        raise RuntimeError(
            "could not gather one adapter shard from every PP stage for export "
            f"(got {sorted(seen_pp)}, expected {args.pipeline_parallel} stages)"
        )
    return merged


def _save_output_best_effort(
    bridge, model, output_dir, args=None, prefer_adapter_artifact=False, state_override=None
):
    import torch.distributed as dist

    save_dir = os.path.expanduser(output_dir)
    os.makedirs(save_dir, exist_ok=True)
    if args is not None and args.tuning == "lora":
        log.info("writing Yeto Megatron adapter artifact without Bridge HF export")
        return _save_megatron_adapter_artifact(args, model, save_dir, state_override=state_override)

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    bridge_tmp = os.path.join(save_dir, ".bridge-export-tmp")
    if rank == 0:
        shutil.rmtree(bridge_tmp, ignore_errors=True)
        os.makedirs(bridge_tmp, exist_ok=True)
    if distributed:
        dist.barrier()
    try:
        # Bridge HF export gathers TP/PP shards and must run on every rank.
        bridge.save_hf_pretrained(model, bridge_tmp)
    except Exception as e:
        if rank == 0:
            shutil.rmtree(bridge_tmp, ignore_errors=True)
        if args is not None and args.tuning == "lora":
            log.warning(
                "Megatron-Bridge HF export failed after training; writing a "
                "Yeto Megatron adapter artifact instead: %s",
                e,
            )
            return _save_megatron_adapter_artifact(
                args, model, save_dir, state_override=state_override
            )
        log.warning(
            "Megatron-Bridge HF export failed after training; leaving run successful "
            "so validation can proceed. adapter-only export still needs wiring: %s",
            e,
        )
        return False
    if rank == 0:
        for name in os.listdir(bridge_tmp):
            src = os.path.join(bridge_tmp, name)
            dst = os.path.join(save_dir, name)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            elif os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(bridge_tmp, ignore_errors=True)
        log.info("saved checkpoint to %s", save_dir)
    if distributed:
        dist.barrier()
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
        fragment_flat,
        pack_fragment,
        quantize_q4,
        unpack_fragment,
    )

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s megatron %(levelname)s %(message)s")
    _validate_parallelism(args)
    rank, world, local_rank = _init_distributed(args)
    device = torch.device("cuda", local_rank)

    model, bridge = _build_model(args, device)
    params = _adapter_params(model) if args.tuning == "lora" else {}
    layout = None
    if args.tuning == "lora":
        layout = build_layout(
            [(n, p.numel()) for n, p in params.items()], args.fragments, args.fragment_pattern
        )
        log.info("%d adapter tensors -> %d fragments", len(params), layout.num_fragments)
    else:
        log.info("%d trainable tensors for full-parameter tuning", len(_trainable_params(model)))

    # mcore DDP wrap so grads land in the distributed optimizer's buckets.
    ddp_cfg = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_grad_reduce=True, grad_reduce_in_fp32=True
    )
    cfg = _prepare_model_config(model, finalize_model_grads, args.pipeline_parallel)
    model = [DDP(config=cfg, ddp_config=ddp_cfg, module=m) for m in model]
    opt = get_megatron_optimizer(
        config=OptimizerConfig(**_optimizer_config_kwargs(args)),
        model_chunks=model,
    )
    forward_backward = get_forward_backward_func()

    # DiLoCo sync setup — identical to the torch learner: rank 0 drives the
    # syncer, sends INIT, and the fragment protocol reuses yeto primitives.
    wire_dtype = {"bf16": DTYPE_BF16, "f32": DTYPE_F32, "q4": DTYPE_Q4}[args.wire_dtype]
    client = None
    if rank == 0 and args.syncer != "none":
        assert layout is not None
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
        fragment_flat=fragment_flat, pack_fragment=pack_fragment, quantize_q4=quantize_q4,
        unpack_fragment=unpack_fragment, apply_fragment=apply_fragment,
        bulk_dtype=bulk_dtype, DTYPE_Q4=DTYPE_Q4,
    )

    # Avoid barriers in the shutdown/save path. For PP local validation, gather
    # one adapter shard per pipeline stage, then rank 0 writes the artifact.
    export_state = (
        _gather_adapter_state_for_export(args, model, rank, world)
        if args.tuning == "lora"
        else None
    )
    if args.tuning == "full" or rank == 0:
        _save_output_best_effort(
            bridge,
            model,
            args.output_dir,
            args,
            state_override=export_state,
        )
        if client is not None:
            client.close()
    dist.destroy_process_group()


def _run_inner_loop(
    args, model, params, layout, opt, forward_backward, client, rank, world, device,
    *, fragment_flat, pack_fragment, quantize_q4, unpack_fragment, apply_fragment,
    bulk_dtype, DTYPE_Q4,
):
    """N inner Megatron steps, pausing at each step boundary to run the DiLoCo
    fragment sync. The inner step differs from the torch learner (Megatron's
    forward_backward schedule vs a manual loop), but the sync half mirrors
    yeto.learner.run_inner_loop's protocol and counter semantics exactly."""
    import torch
    import torch.distributed as dist
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    dataset = _build_dataset(args)
    data_iter = _cycle(dataset)
    mbs = 1 if args.micro_batch_size == "auto" else int(args.micro_batch_size)
    max_steps = _training_steps(args, len(dataset))
    if args.log_every <= 0:
        raise ValueError("--log-every must be greater than zero")
    scheduler = OptimizerParamScheduler(opt, **_scheduler_kwargs(args, max_steps))
    log.info(
        "%d packed blocks; running %d optimizer steps (%s epochs, grad_accum=%d)",
        len(dataset),
        max_steps,
        args.epochs if args.epochs is not None else "step-limited",
        args.grad_accum,
    )

    def forward_step(it, mdl):
        batch = next(it)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        pos = torch.arange(ids.size(1), device=device).unsqueeze(0).expand_as(ids)
        _, is_last_stage = _pipeline_stage_roles()
        out = mdl(**_pipeline_forward_kwargs(ids, pos, labels, is_last_stage))

        def loss_func(output):
            loss = _weighted_token_loss(output, weights)
            return loss, {"lm loss": loss.detach()}

        return out, loss_func

    # Counters (Alg. 1): global totals + per-fragment snapshots reset on receipt.
    steps_total = tokens_total = 0
    steps_at_reset = [0] * layout.num_fragments if layout is not None else []
    tokens_at_reset = [0] * layout.num_fragments if layout is not None else []
    fragment_versions = [0] * layout.num_fragments if layout is not None else []
    pending_pulls: list = []
    global_step = 0
    tokens_per_inner_step = world * mbs * args.grad_accum * args.seq_len
    anchors = None
    if rank == 0 and client is not None and client.dtype == DTYPE_Q4:
        anchors = [fragment_flat(frag, params).cpu() for frag in layout.fragments]
    shutdown = False

    while not shutdown and steps_total < max_steps:
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
        update_successful, _, _ = opt.step()  # grads reduced inside finalize_model_grads
        if update_successful:
            scheduler.step(increment=1)
        steps_total += 1
        tokens_total += tokens_per_inner_step
        if rank == 0 and (steps_total % args.log_every == 0 or steps_total == max_steps):
            log.info("training progress local_step=%d/%d", steps_total, max_steps)

        # --- fragment sync at the step boundary (never blocks) ---
        # Broadcasts apply BEFORE pulls are answered (see yeto/learner.py:
        # a pipelined syncer's next pull can overtake the previous round's
        # broadcast; answering first would push a stale base_version).
        actions = []
        if rank == 0 and client is not None:
            client.check_health()
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
            box = [meta, shutdown]
            dist.broadcast_object_list(box, src=0)
            meta, shutdown = box
            if rank != 0:
                actions = [(f, v, torch.empty(layout.fragments[f].numel)) for f, v in meta]
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
                if anchors is not None:
                    delta = fragment_flat(layout.fragments[fid], params).cpu() - anchors[fid]
                    payload = quantize_q4(delta)
                else:
                    payload = pack_fragment(layout.fragments[fid], params, client.dtype)
                client.push_fragment(
                    fid, pull.global_step, fragment_versions[fid], steps_total,
                    c_steps, c_tokens, payload,
                )
            pending_pulls = still_pending
    log.info("inner loop done at local_step=%d global_step=%d", steps_total, global_step)


def _cycle(dataset):
    """Endless Megatron batches from Yeto's packed token/weight blocks."""
    import torch

    while True:
        for block in dataset:
            if isinstance(block, dict):
                ids = block["input_ids"]
                weights = block.get("weights")
            else:
                ids, weights = block
            ids = torch.as_tensor(ids).unsqueeze(0)
            if weights is None:
                weights = torch.ones_like(ids, dtype=torch.float)
            else:
                weights = torch.as_tensor(weights, dtype=torch.float).unsqueeze(0)
            yield {"input_ids": ids, "labels": ids.clone(), "weights": weights}


if __name__ == "__main__":
    main()
