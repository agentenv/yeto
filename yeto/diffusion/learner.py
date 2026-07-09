"""Diffusion learner: LoRA training with Yeto fragment synchronization.

The distributed/sync half mirrors ``yeto.learner``. The task-specific half is
image/video diffusion: load a diffusers pipeline, freeze VAE/text encoders,
train LoRA adapters on the transformer/UNet denoiser, and compute a denoising
MSE loss on sampled timesteps.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import logging
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from ..autobatch import int_or_auto, rebalance_grad_accum
from ..data import _learner_rows, load_rows
from ..fragments import build_layout
from ..losses import flow_matching_loss
from ..models import resolve
from ..protocol import DTYPE_BF16, DTYPE_F32, DTYPE_Q4, SyncerClient, bulk_dtype
from ..tensor_io import (
    apply_fragment,
    fragment_flat,
    pack_fragment,
    quantize_q4,
    unpack_fragment,
)

log = logging.getLogger("diffusion-learner")

# Cached manifest contract:
#   required with --cache-latents: latents
#   required with --cache-text-embeds: prompt_embeds
#   optional: pooled_prompt_embeds, prompt_attention_mask, latent_num_frames,
#             latent_height, latent_width
_CACHE_TENSOR_SUFFIXES = (".pt", ".pth", ".npy")
DIFFUSION_CACHE_METADATA_FILE = "yeto_diffusion_cache.json"
DIFFUSION_ADAPTER_METADATA_FILE = "yeto_diffusion_adapter.json"
DIFFUSION_CACHE_SCHEMA_VERSION = 1
DIFFUSION_ADAPTER_SCHEMA_VERSION = 1
_MAX_DIFFUSION_MICRO_BATCH = 256
_DIFFUSION_PROBE_ITERATIONS = 2
_MAX_DIFFUSION_PROBE_SHAPES = 8

_ATTENTION_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "q",
    "k",
    "v",
    "o",
]
_MLP_TARGETS = [
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "proj_mlp",
    "proj_out",
    "linear_1",
    "linear_2",
]
_TRAINABLE_ATTRS = ("transformer", "transformer_2", "unet", "model")


@dataclass
class LatentBatch:
    latents: torch.Tensor
    latent_num_frames: int | None = None
    latent_height: int | None = None
    latent_width: int | None = None


@dataclass
class TextConditioning:
    prompt_embeds: torch.Tensor | None
    pooled_prompt_embeds: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _cache_columns(args) -> dict[str, str]:
    return {
        "image": getattr(args, "image_column", "image"),
        "video": getattr(args, "video_column", "video"),
        "prompt": getattr(args, "prompt_column", "prompt"),
        "latents": getattr(args, "latent_column", "latents"),
        "prompt_embeds": getattr(args, "text_embeds_column", "prompt_embeds"),
        "prompt_attention_mask": getattr(
            args,
            "text_attention_mask_column",
            "prompt_attention_mask",
        ),
        "pooled_prompt_embeds": getattr(
            args,
            "pooled_text_embeds_column",
            "pooled_prompt_embeds",
        ),
    }


def _cache_flags(args) -> dict[str, bool]:
    return {
        "latents": bool(getattr(args, "cache_latents", False)),
        "text_embeds": bool(getattr(args, "cache_text_embeds", False)),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_diffusion_cache_metadata(dataset_name) -> dict | None:
    if not isinstance(dataset_name, str):
        return None
    path = Path(os.path.expanduser(dataset_name))
    if path.exists():
        root = path if path.is_dir() else path.parent
    elif path.parent.exists():
        root = path.parent
    else:
        return None
    meta_path = root / DIFFUSION_CACHE_METADATA_FILE
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def validate_diffusion_cache_metadata(meta: dict, args) -> None:
    if meta.get("kind") != "yeto.diffusion.cache":
        raise ValueError(
            f"{DIFFUSION_CACHE_METADATA_FILE}: expected kind 'yeto.diffusion.cache', "
            f"got {meta.get('kind')!r}"
        )
    if meta.get("schema_version") != DIFFUSION_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"{DIFFUSION_CACHE_METADATA_FILE}: unsupported schema_version "
            f"{meta.get('schema_version')!r}"
        )
    cache = meta.get("cache") or {}
    if getattr(args, "cache_latents", False) and not cache.get("latents"):
        raise ValueError(
            f"{DIFFUSION_CACHE_METADATA_FILE}: dataset was not written with latent caches"
        )
    if getattr(args, "cache_text_embeds", False) and not cache.get("text_embeds"):
        raise ValueError(
            f"{DIFFUSION_CACHE_METADATA_FILE}: dataset was not written with text-embed caches"
        )
    columns = meta.get("columns") or {}
    required_keys = []
    if getattr(args, "cache_latents", False):
        required_keys.append("latents")
    if getattr(args, "cache_text_embeds", False):
        required_keys.extend(
            ["prompt_embeds", "prompt_attention_mask", "pooled_prompt_embeds"]
        )
    current_columns = _cache_columns(args)
    for key in required_keys:
        current = current_columns[key]
        recorded = columns.get(key)
        if recorded is not None and recorded != current:
            raise ValueError(
                f"{DIFFUSION_CACHE_METADATA_FILE}: column {key!r} is {recorded!r} "
                f"in the dataset but learner is using {current!r}"
            )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Yeto diffusion learner")
    p.add_argument("--model", required=True, help="diffusers repo id or alias from yeto.models")
    p.add_argument("--data", required=True, help="HF dataset id or local latent/media manifest")
    p.add_argument("--syncer", required=True, help="host:port, or 'none' for standalone")
    p.add_argument("--learner-id", type=int, required=True)
    p.add_argument("--num-learners", type=int, required=True)
    p.add_argument("--loss-function", default="flow_matching")
    p.add_argument("--tuning", choices=["lora", "full"], default="lora")
    p.add_argument("--shard", choices=["ddp", "fsdp"], default="ddp")
    p.add_argument(
        "--diffusion-adapter",
        default=None,
        help="optional module:factory or file.py:factory hook for non-standard diffusion repos",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", choices=["auto", "attention", "all-linear"], default="auto")
    p.add_argument("--micro-batch-size", type=int_or_auto, default="auto")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--fragments", type=int, default=8)
    p.add_argument("--fragment-pattern", choices=["binpack", "strided"], default="binpack")
    p.add_argument("--merge-alpha", type=float, default=0.5)
    p.add_argument("--wire-dtype", choices=["bf16", "f32", "q4"], default="bf16")
    p.add_argument("--wan-streams", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-local-steps", type=int, default=1_000_000)
    p.add_argument("--stream-workers", type=int, default=2)
    p.add_argument(
        "--diffusion-loss-weighting",
        choices=["none", "linear", "sigma", "snr", "min-snr"],
        default="none",
    )
    p.add_argument("--diffusion-min-snr-gamma", type=float, default=5.0)
    p.add_argument("--cache-latents", action="store_true", default=False)
    p.add_argument("--cache-text-embeds", action="store_true", default=False)
    p.add_argument("--image-column", default="image")
    p.add_argument("--video-column", default="video")
    p.add_argument("--prompt-column", default="prompt")
    p.add_argument("--latent-column", default="latents")
    p.add_argument("--text-embeds-column", default="prompt_embeds")
    p.add_argument("--text-attention-mask-column", default="prompt_attention_mask")
    p.add_argument("--pooled-text-embeds-column", default="pooled_prompt_embeds")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--bucket-by-shape", action="store_true", default=False)
    p.add_argument("--output-dir", default="checkpoints/diffusion-out")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def setup_distributed() -> tuple[int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _from_pretrained_offline_first(factory, model_id: str, **kwargs):
    try:
        return factory.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        return factory.from_pretrained(model_id, **kwargs)


def diffusion_torch_dtype(device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    get_capability = getattr(torch.cuda, "get_device_capability", None)
    if get_capability is not None:
        try:
            major, _ = get_capability(device)
        except (AssertionError, RuntimeError, TypeError):
            try:
                major, _ = get_capability()
            except (AssertionError, RuntimeError, TypeError):
                major = None
        if major is not None and major < 8:
            return torch.float16
    if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def resolve_lora_targets(choice: str, model: str | None = None):
    """Map the public LoRA target flag to PEFT target_modules for DiTs."""
    del model
    if choice == "all-linear":
        return "all-linear"
    if choice == "attention":
        return _ATTENTION_TARGETS
    return _ATTENTION_TARGETS + _MLP_TARGETS


def load_diffusion_adapter(spec: str | None):
    """Load an optional user adapter without baking model families into Yeto."""
    if not spec:
        return None
    import importlib
    import importlib.util

    target, sep, factory_name = spec.partition(":")
    if not sep or not target or not factory_name:
        raise ValueError("--diffusion-adapter must be module:factory or file.py:factory")
    if target.endswith(".py") or os.path.sep in target:
        path = Path(os.path.expanduser(target))
        module_spec = importlib.util.spec_from_file_location("yeto_diffusion_adapter", path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot import diffusion adapter from {target!r}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    factory = getattr(module, factory_name)
    return factory() if callable(factory) else factory


def _trainable_module_items(pipe, adapter=None) -> list[tuple[str, torch.nn.Module]]:
    if adapter is not None and hasattr(adapter, "trainable_module_items"):
        return list(adapter.trainable_module_items(pipe))
    return [
        (name, module)
        for name in _TRAINABLE_ATTRS
        if isinstance((module := getattr(pipe, name, None)), torch.nn.Module)
    ]


def _freeze_modules(pipe) -> None:
    for value in getattr(pipe, "components", {}).values():
        if isinstance(value, torch.nn.Module):
            value.requires_grad_(False)
            value.eval()


def load_pipeline(args, device, adapter=None):
    if adapter is not None and hasattr(adapter, "load_pipeline"):
        pipe = adapter.load_pipeline(args, device)
    else:
        from diffusers import DiffusionPipeline

        model_id = resolve(args.model)
        dtype = diffusion_torch_dtype(device)
        pipe = _from_pretrained_offline_first(DiffusionPipeline, model_id, torch_dtype=dtype)
    if adapter is not None and hasattr(adapter, "prepare_model"):
        return adapter.prepare_model(pipe, args, device)
    _freeze_modules(pipe)
    modules = _trainable_module_items(pipe, adapter)
    if not modules:
        model_id = resolve(args.model)
        raise RuntimeError(
            f"{model_id} has no trainable diffusion module named one of {_TRAINABLE_ATTRS}"
        )
    if args.tuning == "lora":
        from peft import LoraConfig, get_peft_model

        peft_kwargs = {}
        try:
            if "autocast_adapter_dtype" in inspect.signature(get_peft_model).parameters:
                peft_kwargs["autocast_adapter_dtype"] = False
        except (TypeError, ValueError):
            pass
        for name, module in modules:
            lora = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=resolve_lora_targets(args.lora_targets, args.model),
            )
            setattr(pipe, name, get_peft_model(module, lora, **peft_kwargs))
    else:
        for _, module in modules:
            module.requires_grad_(True)
    pipe.to(device)
    for _, module in _trainable_module_items(pipe, adapter):
        module.train()
    return pipe


def trainable_params(pipe, adapter=None) -> dict[str, torch.Tensor]:
    if adapter is not None and hasattr(adapter, "trainable_params"):
        return dict(adapter.trainable_params(pipe))
    params: dict[str, torch.Tensor] = {}
    for module_name, module in _trainable_module_items(pipe, adapter):
        for name, p in module.named_parameters():
            if p.requires_grad:
                params[f"{module_name}.{normalize_param_name(name)}"] = p
    return params


_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module.")


def normalize_param_name(name: str) -> str:
    for prefix in _WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def allreduce_trainable_grads(params, world: int) -> None:
    if world <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)


def _outer_module_lists(root):
    found = []

    def visit(m):
        for child in m.children():
            if isinstance(child, torch.nn.ModuleList) and len(child) >= 2:
                found.append(child)
            else:
                visit(child)

    visit(root)
    return found


def maybe_wrap_for_distributed(pipe, args, params, rank: int, world: int, device, adapter=None):
    if args.shard == "fsdp":
        if device.type != "cuda":
            raise RuntimeError("diffusion --shard fsdp requires CUDA")
        if args.tuning == "full" and args.syncer != "none":
            raise ValueError(
                "diffusion --shard fsdp with --tuning full cannot sync sharded "
                "full parameters; use LoRA or --shard ddp"
            )
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError as exc:
            raise RuntimeError("diffusion --shard fsdp needs torch>=2.7") from exc
        ignored = set(params.values()) if args.tuning == "lora" else set()
        for _, module in _trainable_module_items(pipe, adapter):
            blocks = [b for ml in _outer_module_lists(module) for b in ml]
            for block in blocks:
                block_ignored = ignored & set(block.parameters())
                fully_shard(block, ignored_params=block_ignored)
            fully_shard(module, ignored_params=ignored & set(module.parameters()))
        wrapped = trainable_params(pipe, adapter)
        if args.tuning == "lora" and set(wrapped) != set(params):
            diff = sorted(set(wrapped) ^ set(params))[:8]
            raise RuntimeError(f"FSDP changed diffusion adapter names; layout would diverge: {diff}")
        return wrapped
    if world > 1 and args.tuning != "lora":
        for name, module in _trainable_module_items(pipe, adapter):
            setattr(
                pipe,
                name,
                torch.nn.parallel.DistributedDataParallel(
                    module,
                    device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=False,
                ),
            )
        return trainable_params(pipe, adapter)
    return params


class StreamingDiffusionRows(IterableDataset):
    def __init__(
        self,
        dataset_name,
        learner_id: int,
        num_learners: int,
        micro_batch_size: int = 1,
        bucket_by_shape: bool = False,
        max_rows: int | None = None,
        rank: int = 0,
        world: int = 1,
        seed: int = 0,
        split: str = "train",
        cache_latents: bool = False,
        target_num_frames: int | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.learner_id = learner_id
        self.num_learners = num_learners
        self.micro_batch_size = micro_batch_size
        self.bucket_by_shape = bucket_by_shape
        self.max_rows = max_rows
        self.rank = rank
        self.world = world
        self.seed = seed
        self.split = split
        self.cache_latents = cache_latents
        self.num_frames = target_num_frames
        self.height = target_height
        self.width = target_width

    def __iter__(self):
        ds = load_rows(self.dataset_name, self.split)
        data_root = None
        if isinstance(self.dataset_name, str):
            path = Path(os.path.expanduser(self.dataset_name))
            if path.exists():
                data_root = path if path.is_dir() else path.parent
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        shard = _learner_rows(len(ds), self.learner_id, self.num_learners, self.max_rows)
        consumer = self.rank * num_workers + worker_id
        consumers = self.world * num_workers
        my_rows = shard[consumer::consumers]
        if not my_rows:
            raise ValueError(
                f"learner {self.learner_id} rank {self.rank} worker {worker_id}: no rows left"
            )
        rng = random.Random(self.seed + consumer)
        buckets: dict[tuple, list[dict]] = {}
        while True:
            order = my_rows[:]
            rng.shuffle(order)
            for i in order:
                row = dict(ds[i])
                if data_root is not None:
                    row["__yeto_data_root__"] = str(data_root)
                if not self.bucket_by_shape:
                    yield row
                    continue
                key = _shape_key(row, self)
                bucket = buckets.setdefault(key, [])
                bucket.append(row)
                if len(bucket) >= self.micro_batch_size:
                    yield bucket[: self.micro_batch_size]
                    del bucket[: self.micro_batch_size]


def _shape_metadata_int(row: dict, *names: str) -> int | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _row_shape_key(row: dict, args=None) -> tuple[int | None, int | None, int | None]:
    if args is not None and getattr(args, "cache_latents", False):
        return (
            _shape_metadata_int(row, "latent_num_frames", "frames", "num_frames"),
            _shape_metadata_int(row, "latent_height", "height"),
            _shape_metadata_int(row, "latent_width", "width"),
        )
    return (
        _shape_metadata_int(row, "frames", "num_frames", "latent_num_frames"),
        _shape_metadata_int(row, "height", "latent_height"),
        _shape_metadata_int(row, "width", "latent_width"),
    )


def _shape_key(row: dict, args=None) -> tuple[int | None, int | None, int | None]:
    frames, height, width = _row_shape_key(row, args)
    if args is not None:
        frames = getattr(args, "num_frames", None) or frames
        height = getattr(args, "height", None) or height
        width = getattr(args, "width", None) or width
    return frames, height, width


def _batch_shape_key(rows: list[dict], args) -> tuple[int | None, int | None, int | None]:
    if not rows:
        return None, None, None
    key = _shape_key(rows[0], args)
    mismatched = [shape for shape in (_shape_key(row, args) for row in rows[1:]) if shape != key]
    if mismatched:
        shapes = [key, *mismatched[:7]]
        raise ValueError(
            "diffusion rows in a micro-batch must share target shape; "
            f"got {shapes}. Use --bucket-by-shape or set --height/--width/--num-frames."
        )
    return key


def _collate_rows(rows):
    return rows


def _data_root_for(dataset_name) -> str | None:
    if not isinstance(dataset_name, str):
        return None
    path = Path(os.path.expanduser(dataset_name))
    if not path.exists():
        return None
    return str(path if path.is_dir() else path.parent)


def _prepare_probe_batches(args, rank: int, world: int) -> list[list[dict]]:
    ds = load_rows(args.data)
    shard = _learner_rows(
        len(ds),
        getattr(args, "learner_id", 0),
        getattr(args, "num_learners", 1),
        args.max_rows,
    )
    my_rows = shard[rank::world]
    if not my_rows:
        raise ValueError(
            f"learner {getattr(args, 'learner_id', 0)} rank {rank}: "
            "no rows left for autobatch probe"
        )
    data_root = _data_root_for(args.data)
    if not getattr(args, "bucket_by_shape", False):
        rows = [dict(ds[my_rows[0]])]
        if data_root is not None:
            rows[0]["__yeto_data_root__"] = data_root
        return [rows]

    probe_batches: list[list[dict]] = []
    seen_shapes = set()
    for i in my_rows:
        row = dict(ds[i])
        key = _shape_key(row, args)
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        if data_root is not None:
            row["__yeto_data_root__"] = data_root
        probe_batches.append([row])
        if len(probe_batches) >= _MAX_DIFFUSION_PROBE_SHAPES:
            break
    return probe_batches


def _prepare_probe_rows(args, rank: int, world: int) -> list[dict]:
    return _prepare_probe_batches(args, rank, world)[0]


def _repeat_probe_rows(rows: list[dict], micro_batch: int) -> list[dict]:
    return [dict(rows[i % len(rows)]) for i in range(micro_batch)]


def _tensor_from_value(
    value: Any,
    device=None,
    dtype=None,
    base_dir: str | None = None,
    context: str = "cached tensor",
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, (list, tuple)):
        tensor = torch.tensor(value)
    elif isinstance(value, str):
        path = Path(os.path.expanduser(value))
        if not path.is_absolute() and base_dir:
            path = Path(base_dir) / path
        if path.suffix in (".pt", ".pth"):
            if not path.exists():
                raise FileNotFoundError(f"{context}: tensor file {str(path)!r} does not exist")
            tensor = torch.load(path, map_location="cpu")
        elif path.suffix == ".npy":
            if not path.exists():
                raise FileNotFoundError(f"{context}: tensor file {str(path)!r} does not exist")
            import numpy as np

            tensor = torch.from_numpy(np.load(path))
        else:
            raise ValueError(
                f"{context}: unsupported tensor path {value!r}; "
                f"expected one of {_CACHE_TENSOR_SUFFIXES}"
            )
    else:
        raise TypeError(f"{context}: cannot convert {type(value).__name__} to tensor")
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device) if device is not None else tensor


def _stack_column(
    rows,
    column: str,
    device,
    dtype,
    *,
    required: bool = False,
    flag: str | None = None,
) -> torch.Tensor | None:
    values = [row.get(column) for row in rows]
    missing = [i for i, value in enumerate(values) if value is None]
    if missing:
        if required:
            where = ", ".join(str(i) for i in missing[:8])
            raise KeyError(
                f"{flag or 'cached diffusion tensors'} needs column {column!r} "
                f"on every row; missing in batch row(s) {where}"
            )
        if len(missing) == len(rows):
            return None
        where = ", ".join(str(i) for i in missing[:8])
        raise KeyError(
            f"optional cached diffusion column {column!r} is present only on "
            f"some rows; missing in batch row(s) {where}"
        )
    context = f"cached diffusion column {column!r}"
    tensors = [
        _tensor_from_value(
            v,
            dtype=dtype,
            base_dir=row.get("__yeto_data_root__"),
            context=context,
        )
        for row, v in zip(rows, values)
    ]
    try:
        stacked = torch.stack(tensors)
    except RuntimeError as exc:
        shapes = [tuple(t.shape) for t in tensors]
        raise ValueError(
            f"{context} tensors must have identical shapes within a batch; got {shapes}"
        ) from exc
    return stacked.to(device)


def _metadata_int(row: dict, *names: str) -> int | None:
    for name in names:
        value = row.get(name)
        if value is not None:
            return int(value)
    return None


def _open_image(value, base_dir: str | None = None):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        path = Path(os.path.expanduser(value))
        if not path.is_absolute() and base_dir:
            path = Path(base_dir) / path
        return Image.open(path).convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path") is not None:
        return _open_image(value["path"], base_dir)
    raise TypeError(f"unsupported image value {type(value).__name__}")


def _open_video_frames(value, base_dir: str | None = None):
    if isinstance(value, (list, tuple)):
        return [_open_image(v, base_dir) for v in value]
    if isinstance(value, dict) and "path" in value:
        return _open_video_frames(value["path"], base_dir)
    if isinstance(value, str):
        path = Path(os.path.expanduser(value))
        if not path.is_absolute() and base_dir:
            path = Path(base_dir) / path
        if path.is_dir():
            files = sorted(
                p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            )
            return [_open_image(str(p)) for p in files]
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("raw video files need imageio installed, or pass frame paths / --cache-latents") from exc
        from PIL import Image

        return [Image.fromarray(frame).convert("RGB") for frame in iio.imiter(path)]
    raise TypeError(f"unsupported video value {type(value).__name__}")


def _manual_preprocess(images, height: int | None, width: int | None) -> torch.Tensor:
    import numpy as np

    tensors = []
    for img in images:
        if height and width:
            img = img.resize((width, height))
        arr = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
        tensors.append(arr.permute(2, 0, 1))
    return torch.stack(tensors)


def _fit_video_frames(frames, num_frames: int | None):
    if not frames:
        raise ValueError("raw video contains no frames")
    if num_frames is None:
        return list(frames)
    if num_frames <= 0:
        raise ValueError(f"target video frame count must be positive, got {num_frames}")
    source_frames = len(frames)
    if source_frames == num_frames:
        return list(frames)
    if source_frames < num_frames:
        return [*frames, *([frames[-1]] * (num_frames - source_frames))]
    if num_frames == 1:
        return [frames[source_frames // 2]]
    last = source_frames - 1
    return [frames[round(i * last / (num_frames - 1))] for i in range(num_frames)]


def _manual_preprocess_video(
    videos,
    height: int | None,
    width: int | None,
    num_frames: int | None = None,
) -> torch.Tensor:
    # videos: list[list[PIL]] -> (B, C, F, H, W)
    frames = [
        _manual_preprocess(_fit_video_frames(video, num_frames), height, width).permute(1, 0, 2, 3)
        for video in videos
    ]
    try:
        return torch.stack(frames)
    except RuntimeError as exc:
        shapes = [tuple(t.shape) for t in frames]
        raise ValueError(
            "raw video tensors must have identical shapes within a batch; "
            f"got {shapes}. Use --bucket-by-shape or set --height/--width/--num-frames."
        ) from exc


def _preprocess_with_processor(processor, images, *, height: int | None, width: int | None):
    sig = inspect.signature(processor.preprocess)
    params = sig.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {}
    if height is not None and ("height" in params or accepts_kwargs):
        kwargs["height"] = height
    if width is not None and ("width" in params or accepts_kwargs):
        kwargs["width"] = width
    return processor.preprocess(images, **kwargs)


def _extract_latents(encoded) -> torch.Tensor:
    if hasattr(encoded, "latent_dist"):
        return encoded.latent_dist.sample()
    value = getattr(encoded, "latents", None)
    if value is not None:
        return value
    if isinstance(encoded, (tuple, list)):
        return encoded[0]
    if torch.is_tensor(encoded):
        return encoded
    raise TypeError(f"cannot extract latents from VAE output {type(encoded).__name__}")


def _call_vae_encode(vae, pixels: torch.Tensor):
    sig = inspect.signature(vae.encode)
    params = sig.parameters
    if "sample" in params:
        return vae.encode(sample=pixels)
    if "x" in params:
        return vae.encode(x=pixels)
    return vae.encode(pixels)


def _latent_meta(rows, latents: torch.Tensor) -> tuple[int | None, int | None, int | None]:
    if latents.ndim == 5:
        return int(latents.shape[2]), int(latents.shape[3]), int(latents.shape[4])
    first = rows[0] if rows else {}
    if latents.ndim == 4:
        return (
            _metadata_int(first, "latent_num_frames", "frames", "num_frames"),
            int(latents.shape[2]),
            int(latents.shape[3]),
        )
    return (
        _metadata_int(first, "latent_num_frames", "frames", "num_frames"),
        _metadata_int(first, "latent_height", "height"),
        _metadata_int(first, "latent_width", "width"),
    )


def _maybe_pack_latents(pipe, latents: torch.Tensor) -> torch.Tensor:
    pack = getattr(pipe, "_pack_latents", None) or getattr(pipe, "pack_latents", None)
    if pack is None:
        return latents
    sig = inspect.signature(pack)
    params = sig.parameters
    kwargs = {}
    transformer_config = getattr(getattr(pipe, "transformer", None), "config", None)
    if "patch_size" in params:
        kwargs["patch_size"] = getattr(transformer_config, "patch_size", None) or 1
    if "patch_size_t" in params:
        kwargs["patch_size_t"] = getattr(transformer_config, "patch_size_t", None) or 1
    if "batch_size" in params:
        kwargs["batch_size"] = int(latents.shape[0])
    if "num_channels_latents" in params:
        kwargs["num_channels_latents"] = int(latents.shape[1])
    if "height" in params:
        kwargs["height"] = int(latents.shape[-2])
    if "width" in params:
        kwargs["width"] = int(latents.shape[-1])
    if "num_frames" in params and latents.ndim == 5:
        kwargs["num_frames"] = int(latents.shape[2])
    return pack(latents, **kwargs)


def _module_config(module):
    candidates = [module]
    get_base_model = getattr(module, "get_base_model", None)
    if callable(get_base_model):
        try:
            candidates.append(get_base_model())
        except TypeError:
            pass
    base_model = getattr(module, "base_model", None)
    if base_model is not None:
        candidates.append(getattr(base_model, "model", base_model))
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None:
            return config
    return None


def _first_trainable_config(pipe, adapter=None):
    for _, module in _trainable_module_items(pipe, adapter):
        config = _module_config(module)
        if config is not None:
            return config
    return None


def _config_value(config, name: str):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    try:
        return config[name]
    except (KeyError, TypeError):
        return getattr(config, name, None)


def _patch_tuple(value, dims: int) -> tuple[int, ...]:
    if value is None:
        return (1,) * dims
    if isinstance(value, int):
        return (int(value),) * dims
    values = tuple(int(v) for v in value)
    if len(values) >= dims:
        return values[-dims:]
    return ((1,) * (dims - len(values))) + values


def _patchify_latents(latents: torch.Tensor, patch: tuple[int, ...]) -> torch.Tensor | None:
    if latents.ndim == 4:
        ph, pw = patch
        bsz, channels, height, width = latents.shape
        if height % ph or width % pw:
            return None
        return (
            latents.reshape(bsz, channels, height // ph, ph, width // pw, pw)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(bsz, (height // ph) * (width // pw), channels * ph * pw)
        )
    if latents.ndim == 5:
        pt, ph, pw = patch
        bsz, channels, frames, height, width = latents.shape
        if frames % pt or height % ph or width % pw:
            return None
        return (
            latents.reshape(
                bsz,
                channels,
                frames // pt,
                pt,
                height // ph,
                ph,
                width // pw,
                pw,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(
                bsz,
                (frames // pt) * (height // ph) * (width // pw),
                channels * pt * ph * pw,
            )
        )
    return None


def _patchify_for_model_input(pipe, latents: torch.Tensor, adapter=None) -> torch.Tensor | None:
    if latents.ndim == 3:
        return latents
    if latents.ndim not in (4, 5):
        return None
    config = _first_trainable_config(pipe, adapter)
    input_channels = _config_value(config, "in_channels")
    patch = getattr(pipe, "patch_size", None) or _config_value(config, "patch_size")
    dims = latents.ndim - 2
    for candidate in [_patch_tuple(patch, dims), *(_candidate_patch_tuples(latents, input_channels) or [])]:
        packed = _patchify_latents(latents, candidate)
        if packed is None:
            continue
        if input_channels is None or int(packed.shape[-1]) == int(input_channels):
            return packed
    return None


def _candidate_patch_tuples(latents: torch.Tensor, input_channels) -> list[tuple[int, ...]]:
    if input_channels is None or latents.ndim not in (4, 5):
        return []
    channels = int(latents.shape[1])
    target = int(input_channels)
    if target % channels:
        return []
    factor = target // channels
    candidates = []
    if latents.ndim == 4:
        _, _, height, width = latents.shape
        for ph in range(1, min(8, height) + 1):
            if height % ph:
                continue
            for pw in range(1, min(8, width) + 1):
                if width % pw == 0 and ph * pw == factor:
                    candidates.append((ph, pw))
    else:
        _, _, frames, height, width = latents.shape
        for pt in range(1, min(8, frames) + 1):
            if frames % pt:
                continue
            for ph in range(1, min(8, height) + 1):
                if height % ph:
                    continue
                for pw in range(1, min(8, width) + 1):
                    if width % pw == 0 and pt * ph * pw == factor:
                        candidates.append((pt, ph, pw))
    return candidates


def _image_token_mask_from_conditioning(
    cond: TextConditioning,
    batch_size: int,
    seq_len: int,
    image_tokens: int,
    device,
) -> torch.Tensor:
    indicator = cond.extra.get("indicator")
    if torch.is_tensor(indicator) and indicator.ndim == 2 and tuple(indicator.shape) == (batch_size, seq_len):
        last_value = indicator[:, -1:].to(device=indicator.device)
        mask = indicator.eq(last_value)
        if torch.all(mask.sum(dim=1) == image_tokens):
            return mask.to(device=device)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[:, -image_tokens:] = True
    return mask


def _align_latents_to_conditioning_sequence(
    pipe,
    latents: LatentBatch,
    cond: TextConditioning,
    adapter=None,
) -> tuple[LatentBatch, torch.Tensor | None]:
    prompt_embeds = cond.prompt_embeds
    if prompt_embeds is None or prompt_embeds.ndim != 3:
        return latents, None
    tokens = _patchify_for_model_input(pipe, latents.latents, adapter)
    if tokens is None or tokens.ndim != 3:
        return latents, None
    batch_size, image_tokens, token_dim = tokens.shape
    seq_len = int(prompt_embeds.shape[1])
    if int(prompt_embeds.shape[0]) != batch_size or seq_len <= image_tokens:
        return latents, None

    mask = _image_token_mask_from_conditioning(
        cond,
        batch_size,
        seq_len,
        image_tokens,
        tokens.device,
    )
    packed = torch.zeros(
        batch_size,
        seq_len,
        token_dim,
        device=tokens.device,
        dtype=tokens.dtype,
    )
    for idx in range(batch_size):
        packed[idx, mask[idx]] = tokens[idx]
    return (
        LatentBatch(packed, latents.latent_num_frames, latents.latent_height, latents.latent_width),
        mask.to(device=tokens.device, dtype=tokens.dtype),
    )


def encode_latents(pipe, rows, args, device, dtype, adapter=None) -> LatentBatch:
    if adapter is not None and hasattr(adapter, "encode_latents"):
        return adapter.encode_latents(pipe, rows, args, device, dtype)
    if args.cache_latents:
        latents = _stack_column(
            rows,
            args.latent_column,
            device,
            dtype,
            required=True,
            flag="--cache-latents",
        )
        return LatentBatch(latents, *_latent_meta(rows, latents))
    if not hasattr(pipe, "vae") or pipe.vae is None:
        raise RuntimeError("raw diffusion rows need a pipeline VAE, or pass --cache-latents")
    target_frames, target_height, target_width = _batch_shape_key(rows, args)
    if any(args.video_column in row for row in rows):
        videos = [_open_video_frames(row[args.video_column], row.get("__yeto_data_root__")) for row in rows]
        pixels = _manual_preprocess_video(videos, target_height, target_width, target_frames)
    else:
        images = [_open_image(row[args.image_column], row.get("__yeto_data_root__")) for row in rows]
        pixels = (
            _preprocess_with_processor(
                pipe.image_processor,
                images,
                height=target_height,
                width=target_width,
            )
            if hasattr(pipe, "image_processor")
            else _manual_preprocess(images, target_height, target_width)
        )
    pixels = pixels.to(device=device, dtype=dtype)
    with torch.no_grad():
        encoded = _call_vae_encode(pipe.vae, pixels)
        latents = _extract_latents(encoded)
        vae_config = getattr(pipe.vae, "config", None)
        scale = getattr(vae_config, "scaling_factor", None) or 1.0
        shift = getattr(vae_config, "shift_factor", None) or 0.0
        latents = (latents - shift) * scale
    meta = _latent_meta(rows, latents)
    latents = _maybe_pack_latents(pipe, latents)
    return LatentBatch(latents, *meta)


_PROMPT_DICT_STANDARD_KEYS = {
    "prompt_embeds",
    "negative_prompt_embeds",
    "pooled_prompt_embeds",
    "negative_pooled_prompt_embeds",
    "prompt_embeds_mask",
    "prompt_attention_mask",
    "negative_prompt_attention_mask",
    "attention_mask",
    "encoder_attention_mask",
    "encoder_hidden_states_mask",
}

_DENOISER_STANDARD_PARAMS = {
    "hidden_states",
    "sample",
    "timestep",
    "timesteps",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "encoder_hidden_states_mask",
    "attention_mask",
    "hidden_states_masks",
    "pooled_projections",
    "pooled_prompt_embeds",
    "pooled_embeds",
    "img_ids",
    "txt_ids",
    "img_shapes",
    "img_sizes",
    "guidance",
    "attention_kwargs",
    "joint_attention_kwargs",
    "controlnet_block_samples",
    "controlnet_single_block_samples",
    "controlnet_blocks_repeat",
    "additional_t_cond",
    "num_frames",
    "height",
    "width",
    "return_dict",
}

_PROMPT_EXTRA_ALIASES = {
    "prompt_embeds_t5": "encoder_hidden_states_t5",
    "prompt_embeds_llama3": "encoder_hidden_states_llama3",
    "prompt_embeds_llama": "encoder_hidden_states_llama3",
}


def _is_numbered_param(name: str, prefix: str) -> bool:
    return name.startswith(prefix) and name[len(prefix) :].isdigit()


def _prompt_values_for_param(name: str, rows, prompts: list[str]) -> list[str] | None:
    if name == "prompt":
        return prompts
    if _is_numbered_param(name, "prompt_"):
        if not rows:
            return prompts
        return [str(row.get(name, prompt)) for row, prompt in zip(rows, prompts)]
    if name == "negative_prompt" or _is_numbered_param(name, "negative_prompt_"):
        if not any(row.get(name) is not None for row in rows):
            return None
        return [str(row.get(name) or "") for row in rows]
    return None


def _conditioning_column_candidates(name: str):
    yield name
    for source, target in _PROMPT_EXTRA_ALIASES.items():
        if target == name:
            yield source


def _is_numeric_sequence(value) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_numeric_sequence(item) for item in value)
    return False


def _is_tensor_like_conditioning_value(value) -> bool:
    if value is None:
        return False
    if torch.is_tensor(value):
        return True
    if isinstance(value, (list, tuple)):
        return _is_numeric_sequence(value)
    if isinstance(value, str):
        return Path(os.path.expanduser(value)).suffix.lower() in _CACHE_TENSOR_SUFFIXES
    return False


def _is_required_param(params, name: str) -> bool:
    param = params.get(name)
    return param is not None and param.default is inspect.Parameter.empty


def _param_default(fn, name: str):
    try:
        param = inspect.signature(fn).parameters.get(name)
    except (TypeError, ValueError):
        return None
    if param is None or param.default is inspect.Parameter.empty:
        return None
    return param.default


def _tokenizer_max_sequence_length(pipe) -> int | None:
    for name in ("tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4"):
        value = getattr(getattr(pipe, name, None), "model_max_length", None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value
    return None


def _infer_max_sequence_length(pipe, args=None) -> int | None:
    explicit = getattr(args, "max_sequence_length", None)
    if explicit is not None:
        return int(explicit)
    default = _param_default(getattr(pipe, "__call__", None), "max_sequence_length")
    if isinstance(default, int):
        return default
    return _tokenizer_max_sequence_length(pipe)


def _infer_vae_scale_factor(pipe) -> int | None:
    value = getattr(pipe, "vae_scale_factor", None)
    if isinstance(value, int) and value > 0:
        return value
    block_out = getattr(getattr(getattr(pipe, "vae", None), "config", None), "block_out_channels", None)
    if block_out:
        return 2 ** (len(block_out) - 1)
    return None


def _patch_size_2d(pipe) -> tuple[int, int]:
    config = getattr(getattr(pipe, "transformer", None), "config", None)
    patch = getattr(config, "patch_size", None) or 1
    if isinstance(patch, (tuple, list)):
        return int(patch[0] or 1), int((patch[1] if len(patch) > 1 else patch[0]) or 1)
    return int(patch), int(patch)


def _latent_grid_shape(pipe, latents: LatentBatch | None) -> tuple[int | None, int | None]:
    if latents is None or latents.latent_height is None or latents.latent_width is None:
        return None, None
    patch_h, patch_w = _patch_size_2d(pipe)
    return max(1, int(latents.latent_height) // max(1, patch_h)), max(
        1, int(latents.latent_width) // max(1, patch_w)
    )


def _raw_pixel_shape(rows, args, pipe, latents: LatentBatch | None) -> tuple[int | None, int | None]:
    if getattr(args, "height", None) is not None and getattr(args, "width", None) is not None:
        return int(args.height), int(args.width)
    if rows and not getattr(args, "cache_latents", False):
        _, height, width = _batch_shape_key(rows, args)
        if height is not None and width is not None:
            return int(height), int(width)
    scale = _infer_vae_scale_factor(pipe)
    if scale is not None and latents is not None and latents.latent_height is not None and latents.latent_width is not None:
        return int(latents.latent_height) * scale, int(latents.latent_width) * scale
    return None, None


def _denoiser_forward_params(pipe, adapter=None):
    try:
        model = _first_trainable_module(pipe, adapter)
    except RuntimeError:
        return {}
    inspect_model = _signature_model_for_forward(model)
    return inspect.signature(inspect_model.forward).parameters


def _latent_input_param_name(params) -> str | None:
    if "hidden_states" in params:
        return "hidden_states"
    if "sample" in params:
        return "sample"
    for name, param in params.items():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return name
    return None


def _conditioning_extra_param_names(params) -> list[str]:
    reserved = set(_DENOISER_STANDARD_PARAMS)
    latent_name = _latent_input_param_name(params)
    if latent_name is not None:
        reserved.add(latent_name)
    return [
        name
        for name, param in params.items()
        if name not in reserved
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


def _take_first(values: list[Any], predicate):
    for idx, value in enumerate(values):
        if predicate(value):
            return values.pop(idx)
    return None


def _is_float_2d_tensor(value) -> bool:
    return torch.is_tensor(value) and value.ndim == 2 and torch.is_floating_point(value)


def _is_mask_tensor(value) -> bool:
    return torch.is_tensor(value) and value.ndim == 2 and not torch.is_floating_point(value)


def _conditioning_from_tuple(out: tuple, forward_params) -> TextConditioning:
    values = list(out)
    prompt_embeds = values.pop(0) if values else None
    pooled = None
    mask = None
    if any(name in forward_params for name in ("pooled_projections", "pooled_prompt_embeds", "pooled_embeds")):
        pooled = _take_first(values, _is_float_2d_tensor)
    if any(name in forward_params for name in ("encoder_attention_mask", "encoder_hidden_states_mask")):
        mask = _take_first(values, _is_mask_tensor)
    extra = {
        name: value
        for name, value in zip(_conditioning_extra_param_names(forward_params), values)
        if value is not None
    }
    return TextConditioning(prompt_embeds, pooled, mask, extra)


def _conditioning_from_dict(out: dict) -> TextConditioning:
    mask = None
    for name in (
        "prompt_embeds_mask",
        "prompt_attention_mask",
        "attention_mask",
        "encoder_attention_mask",
        "encoder_hidden_states_mask",
        "hidden_states_masks",
    ):
        if out.get(name) is not None:
            mask = out[name]
            break
    extra = {
        _PROMPT_EXTRA_ALIASES.get(name, name): value
        for name, value in out.items()
        if name not in _PROMPT_DICT_STANDARD_KEYS and value is not None
    }
    return TextConditioning(
        out.get("prompt_embeds"),
        out.get("pooled_prompt_embeds"),
        mask,
        extra,
    )


def _move_conditioning_value(value, device, dtype):
    if torch.is_tensor(value):
        if torch.is_floating_point(value) or torch.is_complex(value):
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: _move_conditioning_value(v, device, dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_conditioning_value(v, device, dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_conditioning_value(v, device, dtype) for v in value)
    return value


def _call_encode_prompt(
    pipe,
    prompts,
    device,
    args=None,
    rows=None,
    latents: LatentBatch | None = None,
    dtype=None,
    adapter=None,
):
    if not hasattr(pipe, "encode_prompt"):
        raise RuntimeError("raw prompts need pipeline.encode_prompt(), or pass --cache-text-embeds")
    sig = inspect.signature(pipe.encode_prompt)
    kwargs = {}
    params = sig.parameters
    for name in params:
        values = _prompt_values_for_param(name, rows or [], prompts)
        if values is not None:
            kwargs[name] = values
        elif name == "negative_prompt" or _is_numbered_param(name, "negative_prompt_"):
            if _is_required_param(params, name):
                kwargs[name] = ["" for _ in prompts]
    if "device" in params:
        kwargs["device"] = device
    if "dtype" in params and dtype is not None:
        kwargs["dtype"] = dtype
    if "num_images_per_prompt" in params:
        kwargs["num_images_per_prompt"] = 1
    if "num_videos_per_prompt" in params:
        kwargs["num_videos_per_prompt"] = 1
    if "do_classifier_free_guidance" in params:
        kwargs["do_classifier_free_guidance"] = False
    height, width = _raw_pixel_shape(rows, args, pipe, latents)
    if "height" in params and height is not None:
        kwargs["height"] = height
    if "width" in params and width is not None:
        kwargs["width"] = width
    grid_h, grid_w = _latent_grid_shape(pipe, latents)
    if "grid_h" in params:
        if grid_h is None and _is_required_param(params, "grid_h"):
            raise RuntimeError("pipeline.encode_prompt() requires grid_h, but latent height is unavailable")
        if grid_h is not None:
            kwargs["grid_h"] = grid_h
    if "grid_w" in params:
        if grid_w is None and _is_required_param(params, "grid_w"):
            raise RuntimeError("pipeline.encode_prompt() requires grid_w, but latent width is unavailable")
        if grid_w is not None:
            kwargs["grid_w"] = grid_w
    if "max_sequence_length" in params:
        max_sequence_length = _infer_max_sequence_length(pipe, args)
        if max_sequence_length is not None and (
            _is_required_param(params, "max_sequence_length")
            or getattr(args, "max_sequence_length", None) is not None
        ):
            kwargs["max_sequence_length"] = max_sequence_length
    with torch.no_grad():
        out = pipe.encode_prompt(**kwargs)
    if isinstance(out, dict):
        return _conditioning_from_dict(out)
    if isinstance(out, tuple):
        return _conditioning_from_tuple(out, _denoiser_forward_params(pipe, adapter))
    return TextConditioning(out)


def _stack_row_conditioning_extras(
    pipe,
    rows,
    device,
    dtype,
    adapter=None,
    *,
    existing: set[str] | None = None,
) -> dict[str, Any]:
    extra = {}
    existing = existing or set()
    for name in _conditioning_extra_param_names(_denoiser_forward_params(pipe, adapter)):
        if name in existing:
            continue
        for column in _conditioning_column_candidates(name):
            if not any(_is_tensor_like_conditioning_value(row.get(column)) for row in rows):
                continue
            extra[name] = _stack_column(
                rows,
                column,
                device,
                dtype,
                flag="--cache-text-embeds",
            )
            break
    return extra


def encode_prompt_embeds(pipe, rows, args, device, dtype, adapter=None, latents: LatentBatch | None = None):
    if adapter is not None and hasattr(adapter, "encode_prompt_embeds"):
        return adapter.encode_prompt_embeds(pipe, rows, args, device, dtype)
    if args.cache_text_embeds:
        prompt_embeds = _stack_column(
            rows,
            args.text_embeds_column,
            device,
            dtype,
            required=True,
            flag="--cache-text-embeds",
        )
        pooled = _stack_column(rows, args.pooled_text_embeds_column, device, dtype)
        mask = _stack_column(rows, args.text_attention_mask_column, device, None)
        extra = _stack_row_conditioning_extras(pipe, rows, device, dtype, adapter)
        return TextConditioning(prompt_embeds, pooled, mask, extra)
    prompts = [str(row.get(args.prompt_column, "")) for row in rows]
    cond = _call_encode_prompt(
        pipe,
        prompts,
        device,
        args=args,
        rows=rows,
        latents=latents,
        dtype=dtype,
        adapter=adapter,
    )
    cond.prompt_embeds = cond.prompt_embeds.to(device=device, dtype=dtype) if cond.prompt_embeds is not None else None
    cond.pooled_prompt_embeds = cond.pooled_prompt_embeds.to(device=device, dtype=dtype) if cond.pooled_prompt_embeds is not None else None
    cond.attention_mask = cond.attention_mask.to(device=device) if cond.attention_mask is not None else None
    cond.extra = {k: _move_conditioning_value(v, device, dtype) for k, v in cond.extra.items()}
    cond.extra.update(
        _stack_row_conditioning_extras(
            pipe,
            rows,
            device,
            dtype,
            adapter,
            existing=set(cond.extra),
        )
    )
    return cond


def _num_train_timesteps(scheduler) -> int:
    return int(getattr(getattr(scheduler, "config", None), "num_train_timesteps", 1000))


def _scheduler_tensor(value, device, dtype=None) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, (list, tuple)):
        tensor = torch.tensor(value)
    else:
        return None
    tensor = tensor.to(device=device)
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return tensor


def _scheduler_step_count(scheduler) -> int:
    count = _num_train_timesteps(scheduler)
    for name in ("timesteps", "sigmas"):
        value = getattr(scheduler, name, None)
        if torch.is_tensor(value) or isinstance(value, (list, tuple)):
            count = min(count, len(value))
    return max(1, int(count))


def _sample_scheduler_timesteps(scheduler, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(
        0,
        _scheduler_step_count(scheduler),
        (batch,),
        device=device,
        dtype=torch.long,
    )
    scheduler_timesteps = _scheduler_tensor(getattr(scheduler, "timesteps", None), device)
    if scheduler_timesteps is None:
        return indices, indices
    return indices, scheduler_timesteps[indices.clamp(max=scheduler_timesteps.numel() - 1)]


def _match_dims(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while values.ndim < target.ndim:
        values = values.view(*values.shape, *([1] * (target.ndim - values.ndim)))
    return values


def _call_scheduler_method(fn, positional: tuple, values: dict[str, Any]):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*positional)
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL):
            return fn(*positional)
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if name in values:
            kwargs[name] = values[name]
        elif param.default is inspect.Parameter.empty:
            return fn(*positional)
    try:
        return fn(**kwargs)
    except TypeError:
        return fn(*positional)


def _scale_model_input(scheduler, noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    fn = getattr(scheduler, "scale_model_input", None)
    if not callable(fn):
        return noisy
    return _call_scheduler_method(
        fn,
        (noisy, timesteps),
        {
            "sample": noisy,
            "samples": noisy,
            "latents": noisy,
            "timestep": timesteps,
            "timesteps": timesteps,
            "t": timesteps,
        },
    )


def _scheduler_sigmas_for_timesteps(scheduler, timesteps: torch.Tensor, dtype) -> torch.Tensor | None:
    sigmas = _scheduler_tensor(getattr(scheduler, "sigmas", None), timesteps.device, dtype)
    if sigmas is None:
        return None
    scheduler_timesteps = _scheduler_tensor(getattr(scheduler, "timesteps", None), timesteps.device)
    indices = None
    if scheduler_timesteps is not None:
        matches = scheduler_timesteps.reshape(1, -1) == timesteps.reshape(-1, 1)
        if bool(matches.any(dim=1).all()):
            indices = matches.float().argmax(dim=1).long()
    if indices is None and not torch.is_floating_point(timesteps):
        indices = timesteps.long()
    if indices is None:
        return None
    return sigmas[indices.clamp(min=0, max=sigmas.numel() - 1)]


def diffusion_loss_weights(pipe, timesteps: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor | None:
    scheme = getattr(args, "diffusion_loss_weighting", "none")
    if scheme in (None, "none"):
        return None
    timesteps_f = timesteps.to(device=target.device, dtype=torch.float32)
    if scheme == "linear":
        denom = max(1, _num_train_timesteps(getattr(pipe, "scheduler", None)))
        return (timesteps_f / float(denom)).clamp(min=1e-3)

    scheduler = getattr(pipe, "scheduler", None)
    sigmas = _scheduler_sigmas_for_timesteps(scheduler, timesteps, torch.float32) if scheduler is not None else None
    if sigmas is None:
        raise RuntimeError(
            f"--diffusion-loss-weighting {scheme} needs scheduler.sigmas aligned with timesteps"
        )
    sigmas = sigmas.to(device=target.device, dtype=torch.float32).clamp(min=1e-6)
    if scheme == "sigma":
        return sigmas.square()
    snr = sigmas.reciprocal().square()
    if scheme == "snr":
        return snr
    if scheme == "min-snr":
        gamma = float(getattr(args, "diffusion_min_snr_gamma", 5.0))
        return torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp(min=1e-6)
    raise ValueError(f"unknown --diffusion-loss-weighting {scheme!r}")


def add_noise_and_target(pipe, batch: LatentBatch):
    latents = batch.latents
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("diffusion pipeline has no scheduler")
    noise = torch.randn_like(latents)
    indices, timesteps = _sample_scheduler_timesteps(scheduler, int(latents.shape[0]), latents.device)
    scale_noise = getattr(scheduler, "scale_noise", None)
    if callable(scale_noise):
        noisy = _call_scheduler_method(
            scale_noise,
            (latents, timesteps, noise),
            {
                "sample": latents,
                "samples": latents,
                "latents": latents,
                "original_samples": latents,
                "timestep": timesteps,
                "timesteps": timesteps,
                "noise": noise,
            },
        )
        noisy = _scale_model_input(scheduler, noisy, timesteps)
        return LatentBatch(noisy, batch.latent_num_frames, batch.latent_height, batch.latent_width), noise - latents, timesteps
    sigmas = _scheduler_tensor(getattr(scheduler, "sigmas", None), latents.device, latents.dtype)
    if sigmas is not None:
        indices = indices.clamp(max=sigmas.numel() - 1)
        sigma = _match_dims(sigmas[indices], latents)
        noisy = (1.0 - sigma) * latents + sigma * noise
        noisy = _scale_model_input(scheduler, noisy, timesteps)
        return LatentBatch(noisy, batch.latent_num_frames, batch.latent_height, batch.latent_width), noise - latents, timesteps
    if hasattr(scheduler, "add_noise"):
        noisy = _call_scheduler_method(
            scheduler.add_noise,
            (latents, noise, timesteps),
            {
                "original_samples": latents,
                "sample": latents,
                "samples": latents,
                "latents": latents,
                "noise": noise,
                "timestep": timesteps,
                "timesteps": timesteps,
            },
        )
        noisy = _scale_model_input(scheduler, noisy, timesteps)
        pred_type = getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon")
        if pred_type == "v_prediction" and hasattr(scheduler, "get_velocity"):
            target = _call_scheduler_method(
                scheduler.get_velocity,
                (latents, noise, timesteps),
                {
                    "sample": latents,
                    "samples": latents,
                    "latents": latents,
                    "noise": noise,
                    "timestep": timesteps,
                    "timesteps": timesteps,
                },
            )
        elif pred_type == "sample":
            target = latents
        else:
            target = noise
        return LatentBatch(noisy, batch.latent_num_frames, batch.latent_height, batch.latent_width), target, timesteps
    raise RuntimeError("scheduler must provide sigmas or add_noise()")


def _first_trainable_module(pipe, adapter=None) -> torch.nn.Module:
    modules = _trainable_module_items(pipe, adapter)
    if not modules:
        raise RuntimeError("no trainable diffusion module loaded")
    return modules[0][1]


def _denoiser_boundary_timestep(pipe) -> float | None:
    config = getattr(pipe, "config", None)
    boundary = _config_value(config, "boundary_timestep")
    if boundary is None:
        boundary = getattr(pipe, "boundary_timestep", None)
    if boundary is not None:
        try:
            return float(boundary)
        except (TypeError, ValueError):
            return None

    ratio = _config_value(config, "boundary_ratio")
    if ratio is None:
        ratio = getattr(pipe, "boundary_ratio", None)
    if ratio is None:
        return None
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        return None
    scheduler = getattr(pipe, "scheduler", None)
    return ratio * float(_num_train_timesteps(scheduler))


def _sample_timestep_values(timesteps: torch.Tensor, batch: int) -> torch.Tensor | None:
    if not torch.is_tensor(timesteps):
        return None
    if timesteps.ndim == 0:
        return timesteps.reshape(1).expand(batch)
    if int(timesteps.shape[0]) != batch:
        return None
    if timesteps.ndim == 1:
        return timesteps
    return timesteps.reshape(batch, -1)[:, 0]


def _denoiser_routes(
    pipe,
    modules: list[tuple[str, torch.nn.Module]],
    timesteps: torch.Tensor,
    batch: int,
) -> list[tuple[str, torch.nn.Module, torch.Tensor | None]]:
    if len(modules) < 2:
        return [(modules[0][0], modules[0][1], None)]
    boundary = _denoiser_boundary_timestep(pipe)
    values = _sample_timestep_values(timesteps, batch)
    if boundary is None or values is None:
        return [(modules[0][0], modules[0][1], None)]

    high = torch.nonzero(values >= boundary, as_tuple=False).flatten()
    low = torch.nonzero(values < boundary, as_tuple=False).flatten()
    routes = []
    if high.numel():
        routes.append((modules[0][0], modules[0][1], high))
    if low.numel():
        routes.append((modules[1][0], modules[1][1], low))
    return routes or [(modules[0][0], modules[0][1], None)]


def _is_full_batch_indices(indices: torch.Tensor | None, batch: int) -> bool:
    if indices is None:
        return True
    if int(indices.numel()) != batch:
        return False
    return bool(torch.equal(indices, torch.arange(batch, device=indices.device)))


def _slice_batch_value(value, indices: torch.Tensor, batch: int):
    if torch.is_tensor(value):
        if value.ndim > 0 and int(value.shape[0]) == batch:
            return value.index_select(0, indices.to(device=value.device))
        return value
    if isinstance(value, list) and len(value) == batch:
        return [value[int(i)] for i in indices.detach().cpu().tolist()]
    if isinstance(value, tuple) and len(value) == batch:
        return tuple(value[int(i)] for i in indices.detach().cpu().tolist())
    return value


def _slice_latent_batch(batch: LatentBatch, indices: torch.Tensor) -> LatentBatch:
    return LatentBatch(
        batch.latents.index_select(0, indices.to(device=batch.latents.device)),
        batch.latent_num_frames,
        batch.latent_height,
        batch.latent_width,
    )


def _slice_text_conditioning(cond: TextConditioning, indices: torch.Tensor, batch: int) -> TextConditioning:
    return TextConditioning(
        _slice_batch_value(cond.prompt_embeds, indices, batch),
        _slice_batch_value(cond.pooled_prompt_embeds, indices, batch),
        _slice_batch_value(cond.attention_mask, indices, batch),
        {name: _slice_batch_value(value, indices, batch) for name, value in cond.extra.items()},
    )


def _slice_timesteps(timesteps: torch.Tensor, indices: torch.Tensor, batch: int):
    return _slice_batch_value(timesteps, indices, batch)


def _extract_model_output(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, tuple):
        return out[0]
    for attr in ("sample", "prev_sample"):
        value = getattr(out, attr, None)
        if value is not None:
            return value
    raise TypeError(f"cannot extract tensor from model output {type(out).__name__}")


def _signature_model_for_forward(model: torch.nn.Module) -> torch.nn.Module:
    inspect_model = getattr(model, "module", model)
    candidates = [inspect_model]
    get_base_model = getattr(inspect_model, "get_base_model", None)
    if callable(get_base_model):
        try:
            candidates.append(get_base_model())
        except TypeError:
            pass
    base_model = getattr(inspect_model, "base_model", None)
    if base_model is not None:
        candidates.append(getattr(base_model, "model", base_model))

    for candidate in candidates:
        params = inspect.signature(candidate.forward).parameters
        positional = [
            name
            for name, param in params.items()
            if param.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if "hidden_states" in params or "sample" in params or positional:
            return candidate
    return inspect_model


def _call_optional_pipeline_helper(pipe, names: tuple[str, ...], values: dict[str, Any]):
    for name in names:
        fn = getattr(pipe, name, None)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        kwargs = {}
        missing = False
        for param_name, param in sig.parameters.items():
            if param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            if param_name in values:
                if values[param_name] is None and param.default is inspect.Parameter.empty:
                    missing = True
                    break
                if values[param_name] is not None:
                    kwargs[param_name] = values[param_name]
            elif param.default is inspect.Parameter.empty:
                missing = True
                break
        if missing:
            continue
        try:
            return fn(**kwargs)
        except TypeError:
            log.debug("ignoring incompatible diffusion pipeline helper %s", name, exc_info=True)
    return None


def _latent_token_grid(noisy: LatentBatch) -> tuple[int, int, int] | None:
    latents = noisy.latents
    frames = int(noisy.latent_num_frames or 1)
    height = noisy.latent_height
    width = noisy.latent_width

    if latents.ndim == 5:
        return int(latents.shape[2]), int(latents.shape[3]), int(latents.shape[4])
    if latents.ndim == 4:
        return 1, int(latents.shape[2]), int(latents.shape[3])
    if latents.ndim != 3:
        return None

    tokens = int(latents.shape[1])
    if height and width:
        height = int(height)
        width = int(width)
        if frames * height * width == tokens:
            return frames, height, width
        for scale in range(1, 9):
            if height % scale or width % scale:
                continue
            h = height // scale
            w = width // scale
            if frames * h * w == tokens:
                return frames, h, w

    # Fall back to a 2D factorization when only packed token count is known.
    side = int(tokens**0.5)
    for h in range(side, 0, -1):
        if tokens % h == 0:
            return 1, h, tokens // h
    return None


def _make_image_ids(noisy: LatentBatch, dtype) -> torch.Tensor | None:
    grid = _latent_token_grid(noisy)
    if grid is None:
        return None
    frames, height, width = grid
    frame_ids = torch.arange(frames, device=noisy.latents.device, dtype=dtype)
    row_ids = torch.arange(height, device=noisy.latents.device, dtype=dtype)
    col_ids = torch.arange(width, device=noisy.latents.device, dtype=dtype)
    coords = torch.stack(
        torch.meshgrid(frame_ids, row_ids, col_ids, indexing="ij"),
        dim=-1,
    )
    return coords.reshape(-1, 3)


def _make_batched_image_ids(noisy: LatentBatch, dtype) -> torch.Tensor | None:
    grid = _latent_token_grid(noisy)
    if grid is None:
        return None
    frames, height, width = grid
    frame_ids = torch.arange(frames, device=noisy.latents.device, dtype=dtype)
    row_ids = torch.arange(height, device=noisy.latents.device, dtype=dtype)
    col_ids = torch.arange(width, device=noisy.latents.device, dtype=dtype)
    layer_ids = torch.arange(1, device=noisy.latents.device, dtype=dtype)
    coords = torch.stack(
        torch.meshgrid(frame_ids, row_ids, col_ids, layer_ids, indexing="ij"),
        dim=-1,
    ).reshape(-1, 4)
    return coords.unsqueeze(0).expand(int(noisy.latents.shape[0]), -1, -1)


def _make_text_ids(cond: TextConditioning, fallback, dtype) -> torch.Tensor | None:
    source = cond.prompt_embeds
    if source is None:
        return None
    seq_len = int(source.shape[1] if source.ndim >= 3 else source.shape[0])
    return torch.zeros(seq_len, 3, device=fallback.device, dtype=dtype)


def _shape_records(noisy: LatentBatch, *, image: bool) -> list[tuple[int, ...]] | None:
    grid = _latent_token_grid(noisy)
    if grid is None:
        return None
    frames, height, width = grid
    batch = int(noisy.latents.shape[0])
    shape = (height, width) if image else (frames, height, width)
    return [shape for _ in range(batch)]


def _guidance_value(pipe, args, noisy: LatentBatch) -> torch.Tensor:
    value = getattr(args, "guidance_scale", None)
    if value is None:
        value = _param_default(getattr(pipe, "__call__", None), "guidance_scale")
    if not isinstance(value, (int, float)):
        value = 1.0
    return torch.full(
        (int(noisy.latents.shape[0]),),
        float(value),
        device=noisy.latents.device,
        dtype=noisy.latents.dtype,
    )


def _numeric_attr(obj, *names: str):
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, (int, float)) and value > 0:
            return value
    return None


def _vae_temporal_scale_factor(pipe) -> int:
    value = _numeric_attr(
        pipe,
        "vae_temporal_compression_ratio",
        "vae_scale_factor_temporal",
        "temporal_compression_ratio",
    )
    if value is not None:
        return int(value)
    vae = getattr(pipe, "vae", None)
    value = _numeric_attr(vae, "temporal_compression_ratio", "vae_scale_factor_temporal")
    if value is not None:
        return int(value)
    config = getattr(vae, "config", None)
    value = _numeric_attr(config, "temporal_compression_ratio", "vae_scale_factor_temporal")
    return int(value) if value is not None else 1


def _frame_rate_for_rope(pipe, args) -> float | None:
    value = getattr(args, "fps", None)
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    for name in ("frame_rate", "fps"):
        value = _param_default(getattr(pipe, "__call__", None), name)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _rope_interpolation_scale(pipe, args) -> tuple[float, float, float] | None:
    frame_rate = _frame_rate_for_rope(pipe, args)
    if frame_rate is None:
        return None
    temporal = _vae_temporal_scale_factor(pipe)
    spatial = _vae_scale_factor(pipe)
    return (float(temporal) / frame_rate, float(spatial), float(spatial))


def _shape_denoiser_timesteps(timesteps, noisy: LatentBatch, params):
    if not torch.is_tensor(timesteps) or timesteps.ndim != 1:
        return timesteps
    if noisy.latents.ndim != 3 or int(timesteps.shape[0]) != int(noisy.latents.shape[0]):
        return timesteps
    if "rope_interpolation_scale" not in params:
        return timesteps
    return timesteps.view(-1, 1, 1).expand(-1, int(noisy.latents.shape[1]), -1)


def _auto_fill_denoiser_kwargs(pipe, noisy: LatentBatch, cond: TextConditioning, args, params, kwargs) -> None:
    values = {
        "batch_size": int(noisy.latents.shape[0]),
        "height": (_latent_token_grid(noisy) or (None, None, None))[1],
        "width": (_latent_token_grid(noisy) or (None, None, None))[2],
        "device": noisy.latents.device,
        "dtype": noisy.latents.dtype,
    }

    if "img_ids" in params and "img_ids" not in kwargs:
        helper_ids = _call_optional_pipeline_helper(
            pipe,
            ("_prepare_latent_image_ids", "prepare_latent_image_ids"),
            values,
        )
        kwargs["img_ids"] = helper_ids if helper_ids is not None else _make_image_ids(noisy, noisy.latents.dtype)

    if "txt_ids" in params and "txt_ids" not in kwargs:
        kwargs["txt_ids"] = _make_text_ids(cond, noisy.latents, noisy.latents.dtype)

    if "img_shapes" in params and "img_shapes" not in kwargs:
        kwargs["img_shapes"] = _shape_records(noisy, image=False)

    if "img_sizes" in params and "img_sizes" not in kwargs:
        kwargs["img_sizes"] = _shape_records(noisy, image=True)

    if "guidance" in params and "guidance" not in kwargs:
        kwargs["guidance"] = _guidance_value(pipe, args, noisy)

    if "rope_interpolation_scale" in params and "rope_interpolation_scale" not in kwargs:
        kwargs["rope_interpolation_scale"] = _rope_interpolation_scale(pipe, args)

    if "attention_mask" in params and "attention_mask" not in kwargs:
        kwargs["attention_mask"] = cond.attention_mask

    if "encoder_hidden_states_mask" in params and "encoder_hidden_states_mask" not in kwargs:
        kwargs["encoder_hidden_states_mask"] = cond.attention_mask

    if "hidden_states_masks" in params and "hidden_states_masks" not in kwargs:
        kwargs["hidden_states_masks"] = cond.attention_mask

    for name in list(kwargs):
        if kwargs[name] is None:
            kwargs.pop(name)


def _denoise_forward_one(pipe, model, noisy: LatentBatch, timesteps, cond: TextConditioning, args):
    inspect_model = _signature_model_for_forward(model)
    sig = inspect.signature(inspect_model.forward)
    params = sig.parameters
    kwargs = {}
    if "hidden_states" in params:
        kwargs["hidden_states"] = noisy.latents
    elif "sample" in params:
        kwargs["sample"] = noisy.latents
    else:
        positional = [
            name
            for name, param in params.items()
            if param.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if not positional:
            raise TypeError(f"{type(inspect_model).__name__}.forward has no latent input parameter")
        kwargs[positional[0]] = noisy.latents
    if "timestep" in params:
        kwargs["timestep"] = _shape_denoiser_timesteps(timesteps, noisy, params)
    elif "timesteps" in params:
        kwargs["timesteps"] = _shape_denoiser_timesteps(timesteps, noisy, params)
    if "encoder_hidden_states" in params and cond.prompt_embeds is not None:
        kwargs["encoder_hidden_states"] = cond.prompt_embeds
    if "encoder_attention_mask" in params:
        kwargs["encoder_attention_mask"] = cond.attention_mask
    if "encoder_hidden_states_mask" in params:
        kwargs["encoder_hidden_states_mask"] = cond.attention_mask
    if "pooled_projections" in params and cond.pooled_prompt_embeds is not None:
        kwargs["pooled_projections"] = cond.pooled_prompt_embeds
    if "pooled_prompt_embeds" in params and cond.pooled_prompt_embeds is not None:
        kwargs["pooled_prompt_embeds"] = cond.pooled_prompt_embeds
    if "pooled_embeds" in params and cond.pooled_prompt_embeds is not None:
        kwargs["pooled_embeds"] = cond.pooled_prompt_embeds
    for name, value in cond.extra.items():
        if name in params and name not in kwargs and value is not None:
            kwargs[name] = value
    if "num_frames" in params and noisy.latent_num_frames is not None:
        kwargs["num_frames"] = int(noisy.latent_num_frames)
    if "height" in params and noisy.latent_height is not None:
        kwargs["height"] = int(noisy.latent_height)
    if "width" in params and noisy.latent_width is not None:
        kwargs["width"] = int(noisy.latent_width)
    if "return_dict" in params:
        kwargs["return_dict"] = False
    _auto_fill_denoiser_kwargs(pipe, noisy, cond, args, params, kwargs)
    autocast = nullcontext()
    if noisy.latents.is_cuda and noisy.latents.dtype in (torch.float16, torch.bfloat16):
        autocast = torch.autocast(device_type="cuda", dtype=noisy.latents.dtype)
    with autocast:
        out = model(**kwargs)
    return _extract_model_output(out)


def denoise_forward(pipe, noisy: LatentBatch, timesteps, cond: TextConditioning, args, adapter=None):
    modules = _trainable_module_items(pipe, adapter)
    if not modules:
        raise RuntimeError("no trainable diffusion module loaded")
    batch = int(noisy.latents.shape[0])
    routes = _denoiser_routes(pipe, modules, timesteps, batch)
    if len(routes) == 1 and _is_full_batch_indices(routes[0][2], batch):
        return _denoise_forward_one(pipe, routes[0][1], noisy, timesteps, cond, args)

    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _, model, indices in routes:
        if indices is None:
            indices = torch.arange(batch, device=noisy.latents.device)
        part_noisy = _slice_latent_batch(noisy, indices)
        part_cond = _slice_text_conditioning(cond, indices, batch)
        part_timesteps = _slice_timesteps(timesteps, indices, batch)
        chunks.append((indices, _denoise_forward_one(pipe, model, part_noisy, part_timesteps, part_cond, args)))

    first = chunks[0][1]
    output = first.new_empty((batch, *first.shape[1:]))
    for indices, chunk in chunks:
        output.index_copy_(0, indices.to(device=output.device), chunk)
    return output


def _vae_scale_factor(pipe) -> int:
    for name in ("vae_scale_factor", "vae_scale_factor_spatial", "vae_spatial_compression_ratio"):
        value = getattr(pipe, name, None)
        if isinstance(value, int) and value > 0:
            return value
    vae = getattr(pipe, "vae", None)
    for name in ("spatial_compression_ratio", "vae_scale_factor_spatial"):
        value = getattr(vae, name, None)
        if isinstance(value, int) and value > 0:
            return value
    vae_config = getattr(getattr(pipe, "vae", None), "config", None)
    for name in ("spatial_compression_ratio", "vae_scale_factor_spatial"):
        value = getattr(vae_config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    blocks = getattr(vae_config, "block_out_channels", None)
    if isinstance(blocks, (list, tuple)) and len(blocks) > 1:
        return 2 ** (len(blocks) - 1)
    return 8


def _normalize_unpacked_output(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and value and all(torch.is_tensor(v) for v in value):
        shapes = {tuple(v.shape) for v in value}
        if len(shapes) == 1:
            return torch.stack(list(value), dim=0)
    return None


def _image_ids_for_unpack(noisy: LatentBatch, cond: TextConditioning, dtype) -> torch.Tensor | None:
    for name in ("x_ids", "img_ids", "image_ids", "latent_ids"):
        value = cond.extra.get(name)
        if torch.is_tensor(value):
            return value
    return _make_batched_image_ids(noisy, dtype)


def _unpack_values(pipe, tensor: torch.Tensor, target: torch.Tensor, noisy: LatentBatch, cond: TextConditioning):
    scale = _vae_scale_factor(pipe)
    height = noisy.latent_height
    width = noisy.latent_width
    if height is None and target.ndim >= 4:
        height = int(target.shape[-2])
    if width is None and target.ndim >= 4:
        width = int(target.shape[-1])
    pixel_height = int(height) * scale if height is not None else None
    pixel_width = int(width) * scale if width is not None else None
    x_ids = _image_ids_for_unpack(noisy, cond, tensor.dtype)
    helper_tensor = tensor
    if torch.is_tensor(x_ids) and tensor.ndim == 3 and x_ids.ndim >= 2:
        token_count = int(x_ids.shape[-2])
        if tensor.shape[1] >= token_count:
            helper_tensor = tensor[:, :token_count]
    return {
        "latents": helper_tensor,
        "x": helper_tensor,
        "height": pixel_height,
        "width": pixel_width,
        "vae_scale_factor": scale,
        "x_ids": x_ids,
        "img_ids": x_ids,
        "image_ids": x_ids,
    }


def _try_unpack_to_shape(
    pipe,
    tensor: torch.Tensor,
    target: torch.Tensor,
    noisy: LatentBatch,
    cond: TextConditioning,
) -> torch.Tensor | None:
    values = _unpack_values(pipe, tensor, target, noisy, cond)
    for names in (
        ("_unpack_latents_with_ids", "unpack_latents_with_ids"),
        ("_unpack_latents", "unpack_latents"),
    ):
        try:
            unpacked = _call_optional_pipeline_helper(pipe, names, values)
        except (RuntimeError, ValueError):
            log.debug("ignoring incompatible diffusion unpack helper %s", names, exc_info=True)
            continue
        unpacked = _normalize_unpacked_output(unpacked)
        if torch.is_tensor(unpacked) and tuple(unpacked.shape) == tuple(target.shape):
            return unpacked.to(device=target.device, dtype=target.dtype)
    return None


def _align_prediction_and_target(
    pipe,
    pred: torch.Tensor,
    target: torch.Tensor,
    noisy: LatentBatch,
    cond: TextConditioning,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(pred.shape) == tuple(target.shape):
        return pred, target
    if (
        pred.ndim == target.ndim == 3
        and pred.shape[0] == target.shape[0]
        and pred.shape[2:] == target.shape[2:]
        and pred.shape[1] >= target.shape[1]
    ):
        pred = pred[:, : target.shape[1]]
        if tuple(pred.shape) == tuple(target.shape):
            return pred, target

    unpacked_pred = _try_unpack_to_shape(pipe, pred, target, noisy, cond)
    if unpacked_pred is not None:
        return unpacked_pred, target
    unpacked_target = _try_unpack_to_shape(pipe, target, pred, noisy, cond)
    if unpacked_target is not None:
        return pred, unpacked_target
    return pred, target


def compute_diffusion_loss(pipe, rows, args, device, global_step: int = 0, adapter=None):
    if adapter is not None:
        for name in ("compute_loss", "training_step"):
            fn = getattr(adapter, name, None)
            if fn is None:
                continue
            out = fn(pipe, rows, args, device, global_step)
            if isinstance(out, tuple) and len(out) == 2:
                return out
            return out, torch.ones((), device=device)
    dtype = diffusion_torch_dtype(device)
    latents = encode_latents(pipe, rows, args, device, dtype, adapter)
    cond = encode_prompt_embeds(pipe, rows, args, device, dtype, adapter, latents)
    latents, sequence_loss_mask = _align_latents_to_conditioning_sequence(
        pipe,
        latents,
        cond,
        adapter,
    )
    noisy, target, timesteps = add_noise_and_target(pipe, latents)
    pred = denoise_forward(pipe, noisy, timesteps, cond, args, adapter)
    pred, target = _align_prediction_and_target(pipe, pred, target, noisy, cond)
    if args.loss_function != "flow_matching":
        raise ValueError("diffusion learner currently supports --loss-function flow_matching")
    weights = diffusion_loss_weights(pipe, timesteps, target, args)
    if sequence_loss_mask is not None:
        if weights is None:
            weights = sequence_loss_mask
        else:
            weights = weights.to(device=sequence_loss_mask.device, dtype=sequence_loss_mask.dtype)
            while weights.ndim < sequence_loss_mask.ndim:
                weights = weights.view(*weights.shape, *([1] * (sequence_loss_mask.ndim - weights.ndim)))
            weights = weights * sequence_loss_mask
    return flow_matching_loss(pred, target, timesteps, weights)


def _probe_diffusion_once(pipe, params, opt, rows, args, device, micro_batch: int, adapter=None) -> None:
    for _ in range(_DIFFUSION_PROBE_ITERATIONS):
        probe_rows = _repeat_probe_rows(rows, micro_batch)
        loss, denom = compute_diffusion_loss(pipe, probe_rows, args, device, adapter=adapter)
        (loss / denom.clamp(min=1)).backward()
        with torch.no_grad():
            for p in params.values():
                if p.grad is not None:
                    p.grad.zero_()
        opt.step()
        opt.zero_grad(set_to_none=True)


def _resolve_probe_batch_count(probe_batches: list[list[dict]], device, world: int) -> int:
    count = len(probe_batches)
    if world <= 1:
        return count
    value = torch.tensor([count], device=device, dtype=torch.long)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return int(value.item())


def _probe_max_micro_batch(
    args,
    pipe,
    params,
    opt,
    device,
    rank: int,
    world: int,
    probe_rows: list[dict],
    adapter=None,
) -> int:
    best, size = 0, 1
    while size <= _MAX_DIFFUSION_MICRO_BATCH:
        ok = True
        try:
            _probe_diffusion_once(pipe, params, opt, probe_rows, args, device, size, adapter)
        except torch.cuda.OutOfMemoryError:
            ok = False
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if world > 1:
            flag = torch.tensor([1.0 if ok else 0.0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            ok = flag.item() > 0
        if not ok:
            break
        best = size
        size *= 2
    if best == 0:
        shape = _shape_key(probe_rows[0], args) if probe_rows else None
        raise RuntimeError(
            "diffusion model does not fit even micro-batch 1 for probed "
            f"shape {shape}; lower resolution/frames or use a bigger island"
        )
    if rank == 0 and getattr(args, "bucket_by_shape", False):
        log.info(
            "auto diffusion micro-batch probe shape %s -> %d",
            _shape_key(probe_rows[0], args),
            best,
        )
    return best


def resolve_diffusion_micro_batch_size(
    args,
    pipe,
    params,
    opt,
    device,
    rank: int,
    world: int,
    adapter=None,
) -> int:
    """Probe the largest diffusion micro batch that fits the current latent shape."""
    if args.micro_batch_size != "auto":
        return int(args.micro_batch_size)
    if device.type != "cuda":
        return 1

    probe_batches = _prepare_probe_batches(args, rank, world)
    max_batches = _resolve_probe_batch_count(probe_batches, device, world)
    while len(probe_batches) < max_batches:
        probe_batches.append(probe_batches[-1])
    best = min(
        _probe_max_micro_batch(args, pipe, params, opt, device, rank, world, rows, adapter)
        for rows in probe_batches
    )
    return best


def diffusion_adapter_metadata(
    args,
    pipe,
    adapter=None,
    params: dict[str, torch.Tensor] | None = None,
) -> dict:
    modules = [name for name, _ in _trainable_module_items(pipe, adapter)]
    meta = {
        "kind": "yeto.diffusion.adapter",
        "schema_version": DIFFUSION_ADAPTER_SCHEMA_VERSION,
        "model": getattr(args, "model", None),
        "resolved_model": (
            resolve(getattr(args, "model", "")) if getattr(args, "model", None) else None
        ),
        "diffusion_adapter": getattr(args, "diffusion_adapter", None),
        "tuning": getattr(args, "tuning", None),
        "trainable_modules": modules,
        "cache": _cache_flags(args),
        "columns": _cache_columns(args),
        "loss": {
            "function": getattr(args, "loss_function", None),
            "weighting": getattr(args, "diffusion_loss_weighting", "none"),
            "min_snr_gamma": getattr(args, "diffusion_min_snr_gamma", None),
        },
    }
    if getattr(args, "tuning", None) == "lora":
        meta["lora"] = {
            "r": getattr(args, "lora_r", None),
            "alpha": getattr(args, "lora_alpha", None),
            "targets": getattr(args, "lora_targets", None),
            "resolved_targets": resolve_lora_targets(
                getattr(args, "lora_targets", "auto"),
                getattr(args, "model", None),
            ),
        }
    if params is not None:
        meta["trainable_tensor_count"] = len(params)
        meta["trainable_numel"] = int(sum(p.numel() for p in params.values()))
    return meta


def write_diffusion_adapter_metadata(
    output_dir: str | Path,
    args,
    pipe,
    adapter=None,
    params: dict[str, torch.Tensor] | None = None,
) -> None:
    out = Path(os.path.expanduser(str(output_dir)))
    out.mkdir(parents=True, exist_ok=True)
    _write_json(
        out / DIFFUSION_ADAPTER_METADATA_FILE,
        diffusion_adapter_metadata(args, pipe, adapter, params),
    )


def save_adapters(pipe, output_dir: str, adapter=None, args=None, params=None) -> None:
    out = Path(os.path.expanduser(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    if adapter is not None:
        for name in ("save_adapters", "save"):
            fn = getattr(adapter, name, None)
            if fn is not None:
                fn(pipe, output_dir)
                if args is not None:
                    write_diffusion_adapter_metadata(out, args, pipe, adapter, params)
                return
    saved = False
    for name, module in _trainable_module_items(pipe, adapter):
        module = getattr(module, "module", module)
        if hasattr(module, "save_pretrained"):
            module.save_pretrained(out / name)
            saved = True
    if not saved and hasattr(pipe, "save_lora_weights"):
        pipe.save_lora_weights(str(out))
        saved = True
    if not saved:
        torch.save(
            {n: p.detach().cpu() for n, p in trainable_params(pipe, adapter).items()},
            out / "trainable_state.pt",
        )
    if args is not None:
        write_diffusion_adapter_metadata(out, args, pipe, adapter, params)


def main(argv=None) -> None:
    args = parse_args(argv)
    rank, world = setup_distributed()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s diffusion{args.learner_id}.r{rank} %(levelname)s %(message)s",
    )
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if not 0.0 <= args.merge_alpha < 1.0:
        raise ValueError(f"--merge-alpha must be in [0, 1), got {args.merge_alpha}")
    cache_meta = read_diffusion_cache_metadata(args.data)
    if cache_meta is not None:
        validate_diffusion_cache_metadata(cache_meta, args)
        log.info("loaded diffusion cache metadata from %s", args.data)

    adapter = load_diffusion_adapter(args.diffusion_adapter)
    log.info("loading diffusion model %s (%s)", args.model, args.tuning)
    pipe = load_pipeline(args, device, adapter)
    params = trainable_params(pipe, adapter)
    if not params:
        raise RuntimeError("no trainable diffusion parameters; check --lora-targets")
    params = maybe_wrap_for_distributed(pipe, args, params, rank, world, device, adapter)
    layout = build_layout(
        [(n, p.numel()) for n, p in params.items()], args.fragments, args.fragment_pattern
    )
    log.info(
        "%d trainable tensors -> %d fragments (%.1f MB total)",
        len(params),
        layout.num_fragments,
        sum(p.numel() for p in params.values()) * 2 / 1e6,
    )

    opt = torch.optim.AdamW(params.values(), lr=args.inner_lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps))
    )
    requested_mb = args.micro_batch_size
    args.micro_batch_size = resolve_diffusion_micro_batch_size(
        args, pipe, params, opt, device, rank, world, adapter
    )
    if requested_mb == "auto":
        args.grad_accum = rebalance_grad_accum(args.grad_accum, args.micro_batch_size)
        log.info(
            "auto diffusion micro-batch: %d per GPU (grad-accum -> %d)",
            args.micro_batch_size,
            args.grad_accum,
        )
    else:
        args.grad_accum = rebalance_grad_accum(args.grad_accum, args.micro_batch_size)

    dataset = StreamingDiffusionRows(
        args.data,
        args.learner_id,
        args.num_learners,
        args.micro_batch_size,
        args.bucket_by_shape,
        args.max_rows,
        rank=rank,
        world=world,
        cache_latents=args.cache_latents,
        target_num_frames=args.num_frames,
        target_height=args.height,
        target_width=args.width,
    )
    loader = DataLoader(
        dataset,
        batch_size=None if args.bucket_by_shape else args.micro_batch_size,
        num_workers=args.stream_workers,
        collate_fn=None if args.bucket_by_shape else _collate_rows,
        pin_memory=device.type == "cuda",
    )

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

    steps_total = 0
    units_total = 0
    steps_at_reset = [0] * layout.num_fragments
    units_at_reset = [0] * layout.num_fragments
    fragment_versions = [0] * layout.num_fragments
    pending_pulls: list = []
    global_step = 0
    anchors: list[torch.Tensor] | None = None
    if rank == 0 and client is not None and client.dtype == DTYPE_Q4:
        anchors = [fragment_flat(frag, params).cpu() for frag in layout.fragments]

    shutdown = False
    accum = 0
    t_last = time.monotonic()
    opt.zero_grad(set_to_none=True)
    for rows in loader:
        loss, denom = compute_diffusion_loss(pipe, rows, args, device, global_step=global_step, adapter=adapter)
        (loss / (denom.clamp(min=1) * args.grad_accum)).backward()
        accum += 1
        if accum < args.grad_accum:
            continue
        accum = 0
        if args.tuning == "lora":
            allreduce_trainable_grads(params.values(), world)
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        elif args.shard == "fsdp" and hasattr(_first_trainable_module(pipe, adapter), "clip_grad_norm_"):
            _first_trainable_module(pipe, adapter).clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        steps_total += 1
        units_total += world * args.micro_batch_size * args.grad_accum

        if steps_total % 10 == 0 and rank == 0:
            dt = time.monotonic() - t_last
            t_last = time.monotonic()
            log.info(
                "local_step=%d global_step=%d loss/elem=%.6f (%.2f s/step)",
                steps_total,
                global_step,
                loss.item() / max(1, denom.item()),
                dt / 10,
            )

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
                dist.broadcast(flat, src=0)
                if args.merge_alpha > 0:
                    local = fragment_flat(layout.fragments[fid], params)
                    flat = args.merge_alpha * local + (1.0 - args.merge_alpha) * flat
                apply_fragment(layout.fragments[fid], flat, params)
                if rank == 0:
                    steps_at_reset[fid] = steps_total
                    units_at_reset[fid] = units_total
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
                units_at_reset[fid] = units_total
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
                c_units = units_total - units_at_reset[fid]
                if anchors is not None:
                    delta = fragment_flat(layout.fragments[fid], params).cpu() - anchors[fid]
                    payload = quantize_q4(delta)
                else:
                    payload = pack_fragment(layout.fragments[fid], params, client.dtype)
                client.push_fragment(
                    fid,
                    pull.global_step,
                    fragment_versions[fid],
                    steps_total,
                    c_steps,
                    c_units,
                    payload,
                )
            pending_pulls = still_pending

        if shutdown or steps_total >= args.max_local_steps:
            break

    if rank == 0:
        save_adapters(pipe, args.output_dir, adapter, args=args, params=params)
        if client is not None:
            client.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    log.info("diffusion loop done at local_step=%d global_step=%d", steps_total, global_step)


if __name__ == "__main__":
    main()
