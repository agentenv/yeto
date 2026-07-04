"""Yeto learner for NAVA fine-tuning.

This module keeps NAVA's own pipeline/dataset/loss code as the source of
truth and only adds Yeto's asynchronous fragment synchronization around the
optimizer-step boundary.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from ..fragments import build_layout
from ..layout_metadata import build_layout_metadata
from .labels import prepare_labels
from .lora import LoRAConfig, patch_lora, save_lora_adapter
from .utils import install_nava_uri_resolver, materialize_uri, sha256_uri
from ..protocol import DTYPE_BF16, DTYPE_F32, SyncerClient
from ..tensor_io import apply_fragment, pack_fragment, unpack_fragment

log = logging.getLogger("nava-learner")
_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Yeto NAVA learner")
    p.add_argument("--syncer", required=True, help="host:port, or none for local baseline")
    p.add_argument("--learner-id", type=int, required=True)
    p.add_argument("--num-learners", type=int, required=True)
    p.add_argument("--nava-root", required=True)
    p.add_argument("--nava-config", required=True)
    p.add_argument("--nava-ckpt", required=True)
    p.add_argument("--nava-data", required=True)
    p.add_argument(
        "--nava-data-format",
        choices=["nava-jsonl", "nava-list", "s3-label-array", "s3-label-prefix"],
        default="nava-jsonl",
    )
    p.add_argument("--nava-data-weights", default=None)
    p.add_argument("--nava-data-cache", default=os.environ.get("YETO_NAVA_DATA_CACHE"))
    p.add_argument("--nava-assets-dir", default=os.environ.get("YETO_NAVA_ASSET_CACHE"))
    p.add_argument("--nava-modality", choices=["text_to_av", "text_to_audio", "text_to_video", "text_to_image"], default="text_to_av")
    p.add_argument("--nava-caption-field", choices=["composed", "dense_lora_caption", "short_lora_caption", "lora_tags"], default="composed")
    p.add_argument("--nava-min-duration", type=float, default=None)
    p.add_argument("--nava-max-duration", type=float, default=None)
    p.add_argument("--nava-probe-labels", action="store_true")

    p.add_argument("--nava-tuning", choices=["lora", "full", "attention", "regex"], default="lora")
    p.add_argument("--nava-trainable-regex", default=None)
    p.add_argument("--nava-lora-r", type=int, default=16)
    p.add_argument("--nava-lora-alpha", type=int, default=32)
    p.add_argument("--nava-lora-dropout", type=float, default=0.0)
    p.add_argument("--nava-lora-targets", default="mmdit-all-linear")
    p.add_argument("--nava-full-sync", choices=["unsupported", "gather"], default="unsupported")
    p.add_argument(
        "--nava-merge-avg-regex",
        default=r"(^|\.)(bias|norm|modulation|scale|shift)(\.|$)",
        help="regex for NAVA trainable names that should use AVG merge instead of RDA",
    )
    p.add_argument("--nava-init-timeout", type=float, default=1800.0)
    p.add_argument("--shard", choices=["ddp", "fsdp"], default="fsdp")

    p.add_argument("--nava-batch-size", type=int, default=None)
    p.add_argument("--nava-grad-accum", type=int, default=None)
    p.add_argument("--nava-lr", type=float, default=None)
    p.add_argument("--nava-weight-decay", type=float, default=None)
    p.add_argument("--nava-warmup-steps", type=int, default=None)
    p.add_argument("--nava-max-local-steps", type=int, default=None)
    p.add_argument("--nava-num-workers", type=int, default=None)
    p.add_argument("--nava-io-workers", type=int, default=None)
    p.add_argument("--nava-disable-ema", action="store_true")
    p.add_argument("--nava-save-every", type=int, default=100, help="rank0 local learner-state checkpoint interval")
    p.add_argument("--nava-learner-state-dir", default=None)

    p.add_argument("--fragments", type=int, default=32)
    p.add_argument("--fragment-policy", choices=["balanced"], default="balanced")
    p.add_argument("--wire-dtype", choices=["bf16", "f32"], default="bf16")
    p.add_argument("--wan-streams", type=int, default=8)
    p.add_argument("--output-dir", default="checkpoints/nava-yeto-out")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def setup_distributed() -> tuple[int, int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def normalize_param_name(name: str) -> str:
    for prefix in _WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def trainable_params(model) -> dict[str, torch.Tensor]:
    return {normalize_param_name(n): p for n, p in model.named_parameters() if p.requires_grad}


def allreduce_grads(params, world: int) -> None:
    if world <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)


def import_from_string(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def resolve_in_nava_root(path: str, nava_root: str) -> str:
    if path.startswith(("s3://", "http://", "https://")):
        return path
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded) or os.path.exists(expanded):
        return expanded
    return os.path.join(os.path.abspath(os.path.expanduser(nava_root)), path)


def load_state_dict(path: str, cache_dir: str | None = None) -> dict[str, torch.Tensor]:
    path = materialize_uri(path, cache_dir or os.environ.get("YETO_NAVA_ASSET_CACHE"))
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    try:
        obj = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj


def _apply_config_overrides(cfg: dict, args) -> dict:
    cfg = json.loads(json.dumps(cfg))  # cheap deep copy of YAML primitives
    if args.nava_batch_size is not None:
        cfg["batch_size"] = args.nava_batch_size
    if args.nava_grad_accum is not None:
        cfg["grad_accum_steps"] = args.nava_grad_accum
    if args.nava_lr is not None:
        cfg["lr"] = args.nava_lr
    if args.nava_weight_decay is not None:
        cfg["weight_decay"] = args.nava_weight_decay
    if args.nava_warmup_steps is not None:
        cfg["warmup_steps"] = args.nava_warmup_steps
    if args.nava_max_local_steps is not None:
        cfg["max_steps"] = args.nava_max_local_steps
    if args.nava_num_workers is not None:
        cfg["num_workers"] = args.nava_num_workers
    if args.nava_io_workers is not None:
        cfg.setdefault("data", {})["io_workers"] = args.nava_io_workers
    data_cfg = cfg.setdefault("data", {})
    if args.nava_min_duration is not None:
        data_cfg["min_audio_duration"] = args.nava_min_duration
        fps = float(data_cfg.get("video_fps", 16))
        data_cfg["video_min_frames"] = int(args.nava_min_duration * fps)
    if args.nava_max_duration is not None:
        data_cfg["max_audio_duration"] = args.nava_max_duration
        fps = float(data_cfg.get("video_fps", 16))
        data_cfg["video_max_frames"] = int(args.nava_max_duration * fps)
    data_cfg["modal_prob"] = {
        "text_to_audio": 1.0 if args.nava_modality == "text_to_audio" else 0.0,
        "text_to_video": 1.0 if args.nava_modality == "text_to_video" else 0.0,
        "text_to_image": 1.0 if args.nava_modality == "text_to_image" else 0.0,
        "text_to_av": 1.0 if args.nava_modality == "text_to_av" else 0.0,
    }
    if getattr(args, "nava_assets_dir", None):
        cfg.setdefault("model", {})["ckpt_dir"] = args.nava_assets_dir
        cfg.setdefault("model", {})["audio_vae_ckpt_dir"] = args.nava_assets_dir
    if args.nava_disable_ema:
        cfg["use_ema"] = False
    return cfg


def _prepare_data(args, rank: int) -> tuple[list[list[str]], dict[str, list]]:
    cache_dir = args.nava_data_cache or tempfile_dir("yeto-nava-data")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    data_uri = args.nava_data
    if args.nava_data_format == "s3-label-prefix":
        data_uri = data_uri.rstrip("/") + "/clip_s3_labels.json.gz"
    if args.nava_data_format in ("s3-label-array", "s3-label-prefix"):
        local_jsonl = str(Path(cache_dir) / f"labels-learner-{args.learner_id}-rank-{rank}.nava.jsonl")
        prepare_labels(
            data_uri,
            local_jsonl,
            caption_field=args.nava_caption_field,
            require_trainable=True,
            strict_age=True,
            min_duration=args.nava_min_duration,
            max_duration=args.nava_max_duration,
            probe=args.nava_probe_labels,
            cache_dir=cache_dir,
        )
        data_uri = local_jsonl
        data_format = "nava-jsonl"
    else:
        data_format = args.nava_data_format

    if data_format == "nava-jsonl":
        local = materialize_uri(data_uri, cache_dir)
        return [["yeto_nava", local]], {"yeto_nava": [1.0, args.nava_modality]}

    # Native list/weight files. Materialize S3 files if necessary.
    list_path = materialize_uri(data_uri, cache_dir)
    weights_path = materialize_uri(args.nava_data_weights, cache_dir) if args.nava_data_weights else None
    data = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                _idx, name, path = parts
                data.append([name, path])
            elif len(parts) == 2:
                _idx, path = parts
                name = Path(path).stem
                data.append([name, path])
            else:
                raise ValueError(f"bad nava list line: {line!r}")
    ratios = {}
    if weights_path:
        with open(weights_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    key, value, modal = parts
                else:
                    key, value = parts
                    modal = args.nava_modality
                ratios[key] = [float(value), modal]
    else:
        ratios = {name_path[0]: [1.0, args.nava_modality] for name_path in data}
    return data, ratios


def tempfile_dir(name: str) -> str:
    path = Path(os.environ.get("TMPDIR", "/tmp")) / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def build_pipeline(args, cfg: dict, device):
    sys.path.insert(0, os.path.abspath(args.nava_root))
    install_nava_uri_resolver(args.nava_data_cache)
    PipelineClass = import_from_string(cfg["pipeline"])
    if "video" in cfg.get("modality", "") and "audio" in cfg.get("modality", ""):
        cfg["init_from_meta"] = True
    pipe = PipelineClass.create(
        model_id=cfg.get("model_id", ""),
        use_bf16=cfg["use_bf16"],
        audio_latent_ch=cfg["audio_latent_ch"],
        video_latent_ch=cfg["video_latent_ch"],
        lambda_ddpm=cfg["lambda_ddpm"],
        cfg=cfg,
        device=device,
    )
    model_dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float16
    pipe.model.to(device=device, dtype=model_dtype)
    missing, unexpected = pipe.model.load_state_dict(load_state_dict(args.nava_ckpt, args.nava_assets_dir), strict=False)
    if missing or unexpected:
        log.info("loaded NAVA ckpt with missing=%d unexpected=%d", len(missing), len(unexpected))
    pipe.switch_training_mode()
    return pipe


def configure_trainables(pipe, args) -> LoRAConfig | None:
    model = pipe.model
    if args.nava_tuning == "lora":
        return patch_lora(
            model,
            r=args.nava_lora_r,
            alpha=args.nava_lora_alpha,
            dropout=args.nava_lora_dropout,
            target=args.nava_lora_targets,
            verbose=False,
        )
    for p in model.parameters():
        p.requires_grad_(False)
    if args.nava_tuning == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return None
    if args.nava_tuning == "attention":
        rx = args.nava_trainable_regex or r"(self_attn|cross_attn|attention|\.q\.|\.k\.|\.v\.|\.o\.)"
    else:
        if not args.nava_trainable_regex:
            raise ValueError("--nava-tuning regex requires --nava-trainable-regex")
        rx = args.nava_trainable_regex
    import re

    pat = re.compile(rx)
    matched = 0
    for name, p in model.named_parameters():
        if pat.search(name):
            p.requires_grad_(True)
            matched += p.numel()
    if matched == 0:
        raise ValueError(f"trainable regex {rx!r} matched no parameters")
    return None


def maybe_wrap_model(pipe, params: dict[str, torch.Tensor], args, rank: int, world: int, device):
    if args.shard == "fsdp":
        if device.type != "cuda":
            raise RuntimeError("NAVA --shard fsdp requires CUDA")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools

        wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
        kwargs = {"auto_wrap_policy": wrap_policy, "use_orig_params": True, "device_id": device}
        if args.nava_tuning == "lora":
            # Keep LoRA adapters replicated and unflattened, shard the frozen base.
            kwargs["ignored_states"] = list(params.values())
        else:
            if args.syncer != "none" and args.nava_full_sync != "gather":
                raise ValueError("NAVA full/regex FSDP sync requires --nava-full-sync gather")
            kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
        pipe.model = FSDP(pipe.model, **kwargs)
        wrapped = trainable_params(pipe.model)
        # For ignored LoRA params names should remain stable; for full-gather,
        # use_orig_params may expose wrapper-normalized names.
        if args.nava_tuning == "lora" and set(wrapped) != set(params):
            diff = sorted(set(wrapped) ^ set(params))[:8]
            raise RuntimeError(f"FSDP changed NAVA LoRA trainable names; layout would diverge: {diff}")
        return wrapped
    if world > 1:
        pipe.model = torch.nn.parallel.DistributedDataParallel(
            pipe.model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
        return trainable_params(pipe.model.module)
    return params


@contextmanager
def full_param_sync_context(pipe, args):
    if args.shard == "fsdp" and args.nava_tuning != "lora" and args.nava_full_sync == "gather":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(pipe.model, writeback=True, rank0_only=False):
            yield
    else:
        yield


def build_dataloader(args, cfg: dict, pipe, rank: int, world: int):
    from torch.utils.data import DataLoader
    from nava_src.data.dataset_train import AudioVideoDataset, DistInfo, collate_fn_batch

    data, ratios = _prepare_data(args, rank)
    global_rank = args.learner_id * world + rank
    global_world = args.num_learners * world
    dist_info = DistInfo(world_rank=global_rank, world_size=global_world)
    data_cfg = cfg["data"]
    ddp_bucket_sync = bool(data_cfg.get("enable_ddp_bucket_sync", False))
    num_workers = 0 if ddp_bucket_sync else int(cfg.get("num_workers", 0))
    ds = AudioVideoDataset(
        batch_size=cfg["batch_size"],
        queue_size=data_cfg.get("queue_size", 5),
        io_workers=data_cfg.get("io_workers", 16),
        jsonl_or_src_list=data,
        src_id2ratios=ratios,
        modal_prob=data_cfg["modal_prob"],
        dist_info=dist_info,
        num_shards=max(1, num_workers) * global_world,
        audio_vae_server=pipe.audio_vae,
        image_vae_server=pipe.image_vae,
        video_vae_server=pipe.video_vae,
        use_aspect_ratio_buckets=data_cfg.get("use_aspect_ratio_buckets", False),
        use_length_buckets=data_cfg.get("use_length_buckets", False),
        num_length_buckets=data_cfg.get("num_length_buckets", 10),
        enable_ddp_bucket_sync=False,  # Cross-learner bucket sync would hang; keep local.
        is_packing=data_cfg.get("is_packing", False),
        audio_tokens_per_sec=data_cfg.get("audio_tokens_per_sec", 31.25),
        min_audio_duration=data_cfg.get("min_audio_duration", 0.5),
        max_audio_duration=data_cfg.get("max_audio_duration", 10.0),
        tgt_audio_duration=data_cfg.get("tgt_audio_duration", -1),
        video_min_frames=data_cfg.get("video_min_frames", 17),
        video_max_frames=data_cfg.get("video_max_frames", 129),
        video_tgt_frames=data_cfg.get("video_tgt_frames", 65),
        video_fps=data_cfg.get("video_fps", 16),
        add_spk_emb=data_cfg.get("add_spk_emb", False),
        spk_emb_prob=data_cfg.get("spk_emb_prob", 0.9),
        use_speech_special_token=data_cfg.get("use_speech_special_token", False),
        data_file_divisor=data_cfg.get("data_file_divisor", 1),
        split_wav_mode=data_cfg.get("split_wav_mode", False),
        enable_perf_log=cfg.get("enable_perf_log", False),
    )
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn_batch,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


def batch_units(batch: dict, world: int) -> int:
    total = 0
    if "text_lens" in batch:
        total += int(sum(batch["text_lens"]))
    for key in ("audio_latents", "video_latents", "image_latents"):
        vals = batch.get(key)
        if vals is None:
            continue
        if hasattr(vals, "numel"):
            total += int(vals.numel())
            continue
        for x in vals:
            if x is not None and hasattr(x, "numel"):
                total += int(x.numel())
    return max(1, total * max(1, world))


def run_sync_boundary(args, pipe, params, layout, client, rank, world, device, counters):
    steps_total, units_total, steps_at_reset, units_at_reset, fragment_versions, pending_pulls, global_step = counters
    actions = []
    shutdown = False
    with full_param_sync_context(pipe, args):
        if rank == 0 and client is not None:
            client.check_health()
            pending_pulls.extend(client.drain_pulls())
            still_pending = []
            for pull in pending_pulls:
                fid = pull.fragment_id
                c_steps = steps_total - steps_at_reset[fid]
                if c_steps < 1:
                    still_pending.append(pull)
                    continue
                c_units = units_total - units_at_reset[fid]
                client.push_fragment(
                    fid,
                    pull.global_step,
                    fragment_versions[fid],
                    steps_total,
                    c_steps,
                    c_units,
                    pack_fragment(layout.fragments[fid], params, client.dtype),
                )
            pending_pulls = still_pending
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
                actions = [(f, v, torch.empty(layout.fragments[f].numel, dtype=torch.float32)) for f, v in meta]
            for fid, version, flat in actions:
                flat = flat.to(device)
                dist.broadcast(flat, src=0)
                apply_fragment(layout.fragments[fid], flat, params)
                steps_at_reset[fid] = steps_total
                units_at_reset[fid] = units_total
                fragment_versions[fid] = version
                global_step = max(global_step, version)
        else:
            for fid, version, flat in actions:
                apply_fragment(layout.fragments[fid], flat.to(device), params)
                steps_at_reset[fid] = steps_total
                units_at_reset[fid] = units_total
                fragment_versions[fid] = version
                global_step = max(global_step, version)
    counters[:] = [steps_total, units_total, steps_at_reset, units_at_reset, fragment_versions, pending_pulls, global_step]
    return shutdown


def wait_for_initial_sync(args, pipe, params, layout, client, rank: int, world: int, device) -> tuple[list[int], int]:
    """Block until the syncer has broadcast every fragment's initial state."""
    if args.syncer == "none":
        return [0] * layout.num_fragments, 0
    seen: set[int] = set()
    seen_versions = [0] * layout.num_fragments
    global_step = 0
    deadline = time.monotonic() + float(args.nava_init_timeout)
    while True:
        actions = []
        shutdown = False
        error = None
        done = False
        if rank == 0:
            try:
                client.check_health()
                for bc in client.drain_updates():
                    flat = unpack_fragment(layout.fragments[bc.fragment_id], bc.data, client.dtype)
                    actions.append((bc.fragment_id, bc.version, flat))
                    seen.add(bc.fragment_id)
                shutdown = client.shutdown.is_set()
                done = len(seen) == layout.num_fragments
                if shutdown and not done:
                    error = "syncer shut down before initial NAVA state broadcast completed"
                elif not done and time.monotonic() > deadline:
                    error = (
                        f"timed out waiting for initial NAVA state broadcast "
                        f"({len(seen)}/{layout.num_fragments} fragments)"
                    )
            except BaseException as e:
                error = repr(e)

        if world > 1:
            meta = [(f, v) for f, v, _ in actions] if rank == 0 else None
            box = [meta, shutdown, done, error]
            dist.broadcast_object_list(box, src=0)
            meta, shutdown, done, error = box
            if rank != 0:
                actions = [(f, v, torch.empty(layout.fragments[f].numel, dtype=torch.float32)) for f, v in meta]

        if error:
            raise RuntimeError(error)

        with full_param_sync_context(pipe, args):
            if world > 1:
                for fid, _version, flat in actions:
                    flat = flat.to(device)
                    dist.broadcast(flat, src=0)
                    apply_fragment(layout.fragments[fid], flat, params)
                    seen_versions[fid] = _version
                    global_step = max(global_step, _version)
            else:
                for fid, _version, flat in actions:
                    apply_fragment(layout.fragments[fid], flat.to(device), params)
                    seen_versions[fid] = _version
                    global_step = max(global_step, _version)

        if done:
            if rank == 0:
                log.info("initial NAVA global state applied (%d fragments)", layout.num_fragments)
            return seen_versions, global_step
        time.sleep(0.1)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _cpu_data_state(batch):
    state = batch.get("data_state") if isinstance(batch, dict) else None
    if state is None:
        return None
    if isinstance(state, list) and len(state) == 1:
        state = state[0]
    return state.detach().cpu() if hasattr(state, "detach") else state


def save_learner_state(args, opt, sched, counters, layout_meta, batch) -> None:
    state_dir = Path(args.nava_learner_state_dir or Path(args.output_dir) / "learner_state")
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "learner_id": args.learner_id,
        "local_step": counters[0],
        "global_step": counters[6],
        "layout_hash": layout_meta.get("layout_hash"),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if hasattr(sched, "state_dict") else None,
        "data_state": _cpu_data_state(batch),
    }
    tmp = state_dir / "learner_state.pt.tmp"
    torch.save(payload, tmp)
    tmp.replace(state_dir / "learner_state.pt")
    (state_dir / "learner_id.json").write_text(
        json.dumps(
            {
                "learner_id": args.learner_id,
                "local_step": counters[0],
                "global_step": counters[6],
                "layout_hash": layout_meta.get("layout_hash"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    rank, world, local_rank = setup_distributed()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s nava{args.learner_id}.r{rank} %(levelname)s %(message)s")
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    import yaml

    args.nava_root = os.path.abspath(os.path.expanduser(args.nava_root))
    args.nava_config = resolve_in_nava_root(args.nava_config, args.nava_root)
    sys.path.insert(0, args.nava_root)
    cfg = _apply_config_overrides(yaml.safe_load(open(args.nava_config, "r", encoding="utf-8")), args)
    pipe = build_pipeline(args, cfg, device)
    lora_config = configure_trainables(pipe, args)
    params = trainable_params(pipe.model)
    if not params:
        raise RuntimeError("NAVA backend has no trainable parameters")
    params = maybe_wrap_model(pipe, params, args, rank, world, device)
    layout = build_layout(
        [(n, p.numel()) for n, p in params.items()],
        args.fragments,
        avg_name_regex=args.nava_merge_avg_regex,
    )
    base_sha = sha256_uri(args.nava_ckpt, args.nava_assets_dir or os.environ.get("YETO_NAVA_ASSET_CACHE"))
    layout_meta = build_layout_metadata(
        task="nava",
        layout=layout,
        params=params,
        backend_version="nava-yeto-v1",
        nava_root=os.path.basename(args.nava_root.rstrip(os.sep)),
        nava_config=args.nava_config,
        base_checkpoint=args.nava_ckpt,
        base_checkpoint_sha256=base_sha,
        trainable_policy=args.nava_tuning,
        trainable_regex=args.nava_trainable_regex,
        lora=(
            {
                "r": args.nava_lora_r,
                "alpha": args.nava_lora_alpha,
                "dropout": args.nava_lora_dropout,
                "targets": args.nava_lora_targets,
            }
            if args.nava_tuning == "lora"
            else None
        ),
        merge_avg_regex=args.nava_merge_avg_regex,
        full_sync=args.nava_full_sync,
    )
    if rank == 0:
        log.info("NAVA trainables: %d tensors -> %d fragments (%.2f MB bf16 wire)", len(params), layout.num_fragments, sum(p.numel() for p in params.values()) * 2 / 1e6)

    wire_dtype = DTYPE_BF16 if args.wire_dtype == "bf16" else DTYPE_F32
    client = None
    if rank == 0 and args.syncer != "none":
        host, port = args.syncer.rsplit(":", 1)
        client = SyncerClient(
            (host, int(port)),
            args.learner_id,
            layout,
            wire_dtype,
            args.wan_streams,
            layout_metadata=layout_meta,
        )
        client.start()
        if args.learner_id == 0:
            with full_param_sync_context(pipe, args):
                for fid, frag in enumerate(layout.fragments):
                    client.send_init(fid, pack_fragment(frag, params, wire_dtype))
            log.info("sent INIT_PARAMS for %d fragments", layout.num_fragments)

    initial_versions, initial_global_step = wait_for_initial_sync(
        args, pipe, params, layout, client, rank, world, device
    )

    loader = build_dataloader(args, cfg, pipe, rank, world)
    opt = torch.optim.AdamW(params.values(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    from nava_src.utils.scheduler import WarmupCosineAnnealingLR

    sched = WarmupCosineAnnealingLR(opt, warmup_steps=int(cfg.get("warmup_steps", 0)), max_steps=int(cfg["max_steps"]), eta_min=float(cfg["lr"]) * 0.05)

    steps_total = 0
    units_total = 0
    steps_at_reset = [0] * layout.num_fragments
    units_at_reset = [0] * layout.num_fragments
    fragment_versions = initial_versions
    pending_pulls = []
    global_step = initial_global_step
    counters = [steps_total, units_total, steps_at_reset, units_at_reset, fragment_versions, pending_pulls, global_step]
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    max_steps = int(cfg["max_steps"])
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    log_every = int(cfg.get("log_every", 20))

    pipe.model.train()
    opt.zero_grad(set_to_none=True)
    accum = 0
    loss_acc = 0.0
    t_last = time.monotonic()
    shutdown = False
    while not shutdown and counters[0] < max_steps:
        for maybe_batch in loader:
            batch = maybe_batch[0] if isinstance(maybe_batch, list) and len(maybe_batch) == 1 else maybe_batch
            loss, logs = pipe.forward(batch, global_step=counters[6])
            (loss / grad_accum).backward()
            loss_acc += float(loss.detach().cpu())
            accum += 1
            if accum < grad_accum:
                continue
            accum = 0
            if args.shard == "fsdp" and args.nava_tuning == "lora":
                allreduce_grads(params.values(), world)
            torch.nn.utils.clip_grad_norm_(params.values(), max_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            counters[0] += 1
            counters[1] += batch_units(batch, world)
            if rank == 0 and counters[0] % log_every == 0:
                dt = max(1e-9, time.monotonic() - t_last)
                t_last = time.monotonic()
                log.info("local_step=%d global_step=%d loss=%.4f units/s=%.1f", counters[0], counters[6], loss_acc / log_every, counters[1] / dt)
                loss_acc = 0.0
            shutdown = run_sync_boundary(args, pipe, params, layout, client, rank, world, device, counters)
            if rank == 0 and args.nava_save_every > 0 and counters[0] % args.nava_save_every == 0:
                save_learner_state(args, opt, sched, counters, layout_meta, batch)
            if shutdown or counters[0] >= max_steps:
                break
        if not hasattr(loader.dataset, "is_cycle"):
            break

    if rank == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if args.nava_tuning == "lora" and lora_config is not None:
            save_lora_adapter(unwrap_model(pipe.model), lora_config, out / "adapter")
        else:
            torch.save({"state_dict": {n: p.detach().cpu() for n, p in params.items()}, "global_step": counters[6]}, out / "trainable_state.pt")
        (out / "layout_manifest.json").write_text(json.dumps(layout_meta, indent=2), encoding="utf-8")
        (out / "train_config.json").write_text(
            json.dumps(
                {
                    "task": "nava",
                    "learner_id": args.learner_id,
                    "num_learners": args.num_learners,
                    "local_steps": counters[0],
                    "global_step": counters[6],
                    "nava_tuning": args.nava_tuning,
                    "nava_data": args.nava_data,
                    "nava_data_format": args.nava_data_format,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if client is not None:
            client.close()
        log.info("saved NAVA Yeto output to %s", out)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
