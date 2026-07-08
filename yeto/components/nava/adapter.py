"""NAVA component adapter for the generic diffusion backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..base import DiffusionComponent

REMOTE_PREFIXES = ("http://", "https://")
_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module.")


def _is_local_path(value: str | None) -> bool:
    if not value or value.startswith(REMOTE_PREFIXES):
        return False
    if value.startswith(("/", "./", "../", "~")):
        return True
    return os.path.exists(os.path.expanduser(value))


def _clean_param_name(name: str) -> str:
    for prefix in _WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class NavaComponent(DiffusionComponent):
    name = "nava"
    default_config = "configs/nava.yaml"
    default_lora_targets = "mmdit-all-linear"
    output_dir = "yeto-diffusion-output"

    def add_launch_cli_args(self, parser: argparse.ArgumentParser) -> None:
        # The generic diffusion flags cover NAVA's launch surface. Keeping this
        # hook empty prevents component-specific options from leaking into LM.
        return None

    def add_export_cli_args(self, parser: argparse.ArgumentParser) -> None:
        return None

    def validate(self, args) -> list[str]:
        errors: list[str] = []
        if getattr(args, "adapter", "lora") == "regex" and not getattr(args, "trainable_regex", None):
            errors.append("--adapter regex requires --trainable-regex")
        return errors

    def warnings(self, args) -> list[str]:
        if getattr(args, "adapter", "lora") == "full" and getattr(args, "syncer_memory", 32) < 96:
            return [
                "WARNING: NAVA full trainables keep f32 params+momentum on the syncer; use --syncer-memory 128 or LoRA."
            ]
        return []

    def head_file_mounts(self, args) -> dict[str, str]:
        mounts: dict[str, str] = {}
        if getattr(args, "component_root", None):
            mounts["~/yeto-component"] = os.path.abspath(os.path.expanduser(args.component_root))
        if _is_local_path(getattr(args, "base_checkpoint", None)):
            mounts["~/yeto-base-checkpoint"] = os.path.abspath(os.path.expanduser(args.base_checkpoint))
        return mounts

    def rewrite_for_head(self, args) -> None:
        if getattr(args, "component_root", None):
            args.component_root = os.path.expanduser("~/yeto-component")
        if _is_local_path(getattr(args, "base_checkpoint", None)):
            args.base_checkpoint = os.path.expanduser("~/yeto-base-checkpoint")

    def build_learner_file_mounts(self, args) -> dict[str, str]:
        if not getattr(args, "component_root", None):
            return {}
        return {"~/yeto-component": os.path.abspath(os.path.expanduser(args.component_root))}

    def setup_steps(self, args) -> list[str]:
        return [
            "sudo apt-get update -y >/dev/null 2>&1 && sudo apt-get install -y ffmpeg libsndfile1 >/dev/null 2>&1 || true"
        ]

    def warn_if_wont_fit(self, args, specs) -> None:
        from ... import launcher

        min_vram = 40 if getattr(args, "adapter", "lora") == "lora" and args.shard == "fsdp" else 64
        for spec in specs:
            vram = launcher.GPU_MEM_GB.get(spec.gpu, 0)
            if vram and vram < min_vram:
                print(
                    f"[launcher] WARNING: NAVA {args.adapter}/{args.shard} on {spec.gpu} (~{vram}GB/GPU) may OOM; use a smaller config or larger GPU.",
                    file=sys.stderr,
                )

    def resolve_paths(self, args) -> None:
        args.component_root = args.component_root or os.environ.get("YETO_NAVA_ROOT")
        if args.component_root:
            args.component_root = os.path.abspath(os.path.expanduser(args.component_root))
        args.component_config = resolve_in_component_root(
            args.component_config or self.default_config, args.component_root
        )
        if args.lora_targets is None:
            args.lora_targets = self.default_lora_targets

    def load_config(self, args) -> dict:
        import yaml

        with open(args.component_config, "r", encoding="utf-8") as f:
            return _apply_config_overrides(yaml.safe_load(f), args)

    def build_pipeline(self, args, cfg: dict, device):
        import torch

        if args.component_root:
            sys.path.insert(0, args.component_root)
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
        missing, unexpected = pipe.model.load_state_dict(load_state_dict(args.base_checkpoint), strict=False)
        if missing or unexpected:
            import logging

            logging.getLogger("diffusion-learner").info(
                "loaded NAVA checkpoint with missing=%d unexpected=%d", len(missing), len(unexpected)
            )
        pipe.switch_training_mode()
        return pipe

    def configure_trainables(self, runtime, args):
        from .lora import patch_lora

        model = runtime.model
        if args.adapter == "lora":
            return patch_lora(
                model,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target=args.lora_targets or self.default_lora_targets,
                verbose=False,
            )
        for p in model.parameters():
            p.requires_grad_(False)
        if args.adapter == "full":
            for p in model.parameters():
                p.requires_grad_(True)
            return None
        import re

        pat = re.compile(args.trainable_regex)
        matched = 0
        for name, p in model.named_parameters():
            if pat.search(name):
                p.requires_grad_(True)
                matched += p.numel()
        if matched == 0:
            raise ValueError(f"trainable regex {args.trainable_regex!r} matched no parameters")
        return None

    def trainable_params(self, runtime):
        return {
            _clean_param_name(name): p
            for name, p in runtime.model.named_parameters()
            if p.requires_grad
        }

    def build_dataloader(self, args, cfg: dict, runtime, rank: int, world: int):
        from torch.utils.data import DataLoader
        from nava_src.data.dataset_train import AudioVideoDataset, DistInfo, collate_fn_batch

        data, ratios = _prepare_data(args)
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
            audio_vae_server=runtime.audio_vae,
            image_vae_server=runtime.image_vae,
            video_vae_server=runtime.video_vae,
            use_aspect_ratio_buckets=data_cfg.get("use_aspect_ratio_buckets", False),
            use_length_buckets=data_cfg.get("use_length_buckets", False),
            num_length_buckets=data_cfg.get("num_length_buckets", 10),
            enable_ddp_bucket_sync=False,
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

    def training_step(self, runtime, batch, global_step: int):
        return runtime.forward(batch, global_step=global_step)

    def build_scheduler(self, optimizer, cfg: dict):
        from nava_src.utils.scheduler import WarmupCosineAnnealingLR

        return WarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=int(cfg.get("warmup_steps", 0)),
            max_steps=int(cfg["max_steps"]),
            eta_min=float(cfg["lr"]) * 0.05,
        )

    def batch_units(self, batch, world: int) -> int:
        total = 0
        if isinstance(batch, dict) and "text_lens" in batch:
            total += int(sum(batch["text_lens"]))
        if isinstance(batch, dict):
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

    def save_artifact(self, runtime, args, output_dir: Path, params, adapter_config, metadata: dict) -> None:
        import torch

        from .lora import save_lora_adapter

        output_dir.mkdir(parents=True, exist_ok=True)
        if args.adapter == "lora" and adapter_config is not None:
            save_lora_adapter(_unwrap_model(runtime.model), adapter_config, output_dir / "adapter")
        else:
            torch.save(
                {"state_dict": {n: p.detach().cpu() for n, p in params.items()}, "global_step": metadata.get("global_step", 0)},
                output_dir / "trainable_state.pt",
            )
        (output_dir / "layout_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (output_dir / "train_config.json").write_text(
            json.dumps(
                {
                    "task": "diffusion",
                    "component": self.name,
                    "learner_id": args.learner_id,
                    "num_learners": args.num_learners,
                    "local_steps": metadata.get("local_steps"),
                    "global_step": metadata.get("global_step"),
                    "adapter": args.adapter,
                    "data_format": args.data_format,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def export(self, args, ckpt) -> None:
        import torch

        from ...export import validate_against_layout
        from ...fragments import build_layout
        from ...layout_metadata import validate_layout_metadata
        from ...tensor_io import apply_fragment
        from .lora import LoRAConfig, merge_lora_inplace, save_lora_adapter

        device = torch.device(args.device)
        args.component_root = args.component_root or os.environ.get("YETO_NAVA_ROOT")
        if args.component_root:
            args.component_root = os.path.abspath(os.path.expanduser(args.component_root))
        args.component_config = resolve_in_component_root(
            args.component_config or self.default_config, args.component_root
        )
        meta = ckpt.layout_meta or {}
        adapter_meta = meta.get("adapter") if isinstance(meta.get("adapter"), dict) else {}
        args.adapter = adapter_meta.get("type", meta.get("trainable_policy", args.adapter))
        args.lora_r = int(adapter_meta.get("r", args.lora_r))
        args.lora_alpha = int(adapter_meta.get("alpha", args.lora_alpha))
        args.lora_dropout = float(adapter_meta.get("dropout", args.lora_dropout))
        args.lora_targets = adapter_meta.get("targets", args.lora_targets or self.default_lora_targets)
        args.trainable_regex = meta.get("trainable_regex") or args.trainable_regex
        cfg = self.load_config(args)
        runtime = self.build_pipeline(args, cfg, device)
        lora_cfg = self.configure_trainables(runtime, args)
        params = self.trainable_params(runtime)
        layout = build_layout(
            [(n, p.numel()) for n, p in params.items()],
            args.fragments,
            args.fragment_pattern,
            avg_name_regex=meta.get("merge_avg_regex") or args.merge_avg_regex,
        )
        validate_against_layout(ckpt, layout)
        validate_layout_metadata(
            ckpt.layout_meta,
            layout,
            params,
            expected_task="diffusion",
            expected_component=self.name,
            allow_base_mismatch=args.allow_base_mismatch,
        )
        for frag, (_, flat_params, _) in zip(layout.fragments, ckpt.fragments):
            apply_fragment(frag, flat_params, params)

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if args.adapter == "lora" and args.format in ("lora", "merged", "both"):
            if args.format in ("lora", "both"):
                adapter_out = out if args.format == "lora" else out / "adapter"
                save_lora_adapter(
                    runtime.model,
                    lora_cfg or LoRAConfig(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_targets),
                    adapter_out,
                )
            if args.format in ("merged", "both"):
                merge_lora_inplace(runtime.model)
                merged_state = {k: v.detach().cpu().contiguous() for k, v in runtime.model.state_dict().items()}
                try:
                    from safetensors.torch import save_file

                    save_file(merged_state, str(out / "NAVA_merged.safetensors"))
                except Exception:
                    torch.save({"state_dict": merged_state}, out / "NAVA_merged.ckpt")
        else:
            torch.save(
                {"state_dict": {n: p.detach().cpu() for n, p in params.items()}, "global_step": ckpt.global_step},
                out / "trainable_state.pt",
            )
        (out / "yeto_export_meta.json").write_text(
            json.dumps(
                {
                    "task": "diffusion",
                    "component": self.name,
                    "global_step": ckpt.global_step,
                    "format": args.format,
                    "adapter": args.adapter,
                    "fragments": layout.num_fragments,
                    "trainable_tensors": len(params),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if ckpt.layout_meta:
            (out / "layout_manifest.json").write_text(json.dumps(ckpt.layout_meta, indent=2), encoding="utf-8")
        print(f"exported NAVA {args.format} at global_step={ckpt.global_step} to {out}")


def import_from_string(path: str):
    import importlib

    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def resolve_in_component_root(path: str, component_root: str | None) -> str:
    if path.startswith(REMOTE_PREFIXES):
        return path
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded) or os.path.exists(expanded):
        return expanded
    if not component_root:
        return expanded
    return os.path.join(os.path.abspath(os.path.expanduser(component_root)), path)


def load_state_dict(path: str) -> dict:
    import torch

    path = os.path.expanduser(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    try:
        obj = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj


def _apply_config_overrides(cfg: dict, args) -> dict:
    cfg = json.loads(json.dumps(cfg))
    if getattr(args, "batch_size", None) is not None:
        cfg["batch_size"] = args.batch_size
    if getattr(args, "grad_accum", None) is not None:
        cfg["grad_accum_steps"] = args.grad_accum
    if getattr(args, "lr", None) is not None:
        cfg["lr"] = args.lr
    if getattr(args, "weight_decay", None) is not None:
        cfg["weight_decay"] = args.weight_decay
    if getattr(args, "warmup_steps", None) is not None:
        cfg["warmup_steps"] = args.warmup_steps
    if getattr(args, "max_local_steps", None) is not None:
        cfg["max_steps"] = args.max_local_steps
    if getattr(args, "num_workers", None) is not None:
        cfg["num_workers"] = args.num_workers
    if getattr(args, "io_workers", None) is not None:
        cfg.setdefault("data", {})["io_workers"] = args.io_workers
    data_cfg = cfg.setdefault("data", {})
    modality = getattr(args, "modality", "text_to_av")
    data_cfg["modal_prob"] = {
        "text_to_audio": 1.0 if modality == "text_to_audio" else 0.0,
        "text_to_video": 1.0 if modality == "text_to_video" else 0.0,
        "text_to_image": 1.0 if modality == "text_to_image" else 0.0,
        "text_to_av": 1.0 if modality == "text_to_av" else 0.0,
    }
    if getattr(args, "disable_ema", False):
        cfg["use_ema"] = False
    return cfg


def _prepare_data(args) -> tuple[list[list[str]], dict[str, list]]:
    data_path = os.path.expanduser(args.data)
    if args.data_format == "jsonl":
        return [["yeto_diffusion", data_path]], {"yeto_diffusion": [1.0, args.modality]}

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                _idx, name, path = parts
                data.append([name, path])
            elif len(parts) == 2:
                _idx, path = parts
                data.append([Path(path).stem, path])
            else:
                raise ValueError(f"bad diffusion list line: {line!r}")
    ratios = {name: [1.0, args.modality] for name, _ in data}
    return data, ratios
