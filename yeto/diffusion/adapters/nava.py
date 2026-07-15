"""NAVA adapter for the generic Yeto diffusion learner.

This module intentionally stays behind ``--diffusion-adapter``. The core
diffusion learner remains model-agnostic; NAVA-specific loading, row conversion,
and artifact reload logic live here.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .base import DiffusionAdapter
from ..learner import diffusion_torch_dtype
from ...models import resolve

log = logging.getLogger("diffusion-nava-adapter")


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _expand(value: str | os.PathLike | None) -> str | None:
    return str(Path(os.path.expanduser(str(value))).resolve()) if value else None


def _first_existing(*paths: str | os.PathLike | None) -> str | None:
    for path in paths:
        expanded = _expand(path)
        if expanded and os.path.exists(expanded):
            return expanded
    return None


def _add_python_path(path: str | None) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _patch_nava_tokenizer_padding() -> None:
    from nava_src.models.nava.modules import tokenizers

    cls = tokenizers.HuggingfaceTokenizer
    if getattr(cls, "_yeto_pad_token_patch", False):
        return
    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        tokenizer = self.tokenizer
        if getattr(tokenizer, "pad_token", None) is None:
            for token in (tokenizer.eos_token, tokenizer.unk_token, "<pad>", "</s>", "<unk>"):
                if token is not None and tokenizer.convert_tokens_to_ids(token) is not None:
                    tokenizer.pad_token = token
                    break

    cls.__init__ = patched_init
    cls._yeto_pad_token_patch = True


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def _resolve_relative_to_root(value: str | None, nava_root: str | None) -> str | None:
    if not value or not nava_root:
        return value
    path = Path(os.path.expanduser(value))
    return str(path if path.is_absolute() else Path(nava_root) / path)


def _infer_assets_dir(model_id: str, checkpoint: str | None) -> str | None:
    explicit = _env("YETO_NAVA_ASSETS_DIR") or _env("NAVA_ASSETS_DIR")
    if explicit:
        return _expand(explicit)
    if checkpoint:
        return str(Path(checkpoint).resolve().parent)
    model_path = Path(os.path.expanduser(model_id))
    if model_path.exists():
        return str(model_path.resolve())
    return None


def _infer_checkpoint(model_id: str, assets_dir: str | None) -> str | None:
    explicit = (
        _env("YETO_NAVA_CKPT")
        or _env("YETO_NAVA_CHECKPOINT")
        or _env("NAVA_CKPT")
        or _env("NAVA_CHECKPOINT")
    )
    return _first_existing(
        explicit,
        Path(assets_dir) / "NAVA.safetensors" if assets_dir else None,
        Path(os.path.expanduser(model_id)) / "NAVA.safetensors",
    )


def _module_to_save(module):
    return getattr(module, "module", module)


def _to_local_path(value: Any, base_dir: str | None) -> Any:
    if not isinstance(value, str) or "://" in value:
        return value
    path = Path(os.path.expanduser(value))
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str(Path(base_dir) / path)


def _caption_from_row(row: dict, prompt_column: str) -> str:
    text_list = row.get("text_list")
    if text_list:
        first = text_list[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
        return str(first)
    return str(row.get(prompt_column, row.get("caption", row.get("text", ""))))


def _video_path_from_row(row: dict, video_column: str) -> Any:
    if row.get(video_column) is not None:
        return row[video_column]
    video_info = row.get("video_info")
    if video_info:
        first = video_info[0]
        if isinstance(first, dict):
            return first.get("data_path") or first.get("bos_url") or first.get("path")
    return None


def _video_info_from_row(row: dict, args) -> dict:
    video_info = row.get("video_info")
    if video_info and isinstance(video_info[0], dict):
        info = copy.deepcopy(video_info[0])
    else:
        value = _video_path_from_row(row, getattr(args, "video_column", "video"))
        if value is None:
            raise KeyError(
                "NAVA adapter rows need NAVA video_info or a raw video column "
                f"{getattr(args, 'video_column', 'video')!r}"
            )
        info = {"data_path": value}
    base_dir = row.get("__yeto_data_root__")
    info["data_path"] = _to_local_path(info.get("data_path") or info.get("bos_url"), base_dir)
    fps = float(getattr(args, "fps", None) or row.get("fps") or info.get("fps") or 24.0)
    frames = getattr(args, "num_frames", None) or row.get("frames") or row.get("num_frames")
    duration = row.get("duration") or info.get("duration") or (float(frames) / fps if frames else 0.0)
    info.setdefault("fps", fps)
    info.setdefault("duration", float(duration))
    info.setdefault("image_width", int(row.get("width") or getattr(args, "width", None) or 0))
    info.setdefault("image_height", int(row.get("height") or getattr(args, "height", None) or 0))
    info.setdefault("is_valid", True)
    return info


def _nava_json_row(row: dict, args) -> dict:
    obj = copy.deepcopy(row)
    obj["video_info"] = [_video_info_from_row(row, args)]
    obj["text_list"] = [
        {
            "text": _caption_from_row(row, getattr(args, "prompt_column", "prompt")),
            "is_valid": True,
        }
    ]
    return obj


def _int_or_default(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _latent_channels(pipe, name: str, default: int) -> int:
    model = getattr(pipe, "model", None)
    return _int_or_default(
        getattr(model, name, None) or getattr(pipe, name, None),
        default,
    )


class NavaAdapter(DiffusionAdapter):
    def __init__(
        self,
        *,
        nava_root: str | None = None,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        assets_dir: str | None = None,
    ):
        self.nava_root = _expand(
            nava_root
            or _env("YETO_NAVA_ROOT")
            or _env("NAVA_ROOT")
            or "/home/alexeisie/NAVA"
        )
        self.config_path = _expand(
            config_path
            or _env("YETO_NAVA_CONFIG")
            or _env("NAVA_CONFIG")
            or (Path(self.nava_root) / "configs/nava.yaml" if self.nava_root else None)
        )
        self.checkpoint_path = _expand(checkpoint_path or _env("YETO_NAVA_CKPT") or _env("NAVA_CKPT"))
        self.assets_dir = _expand(assets_dir or _env("YETO_NAVA_ASSETS_DIR") or _env("NAVA_ASSETS_DIR"))
        self._builder = None

    def _config(self, args) -> dict:
        if not self.config_path or not os.path.exists(self.config_path):
            raise FileNotFoundError(
                "NAVA adapter needs a config; set YETO_NAVA_CONFIG or pass "
                "config_path to make_adapter()"
            )
        model_id = resolve(getattr(args, "model", "nava"))
        checkpoint = self.checkpoint_path
        assets_dir = self.assets_dir
        if checkpoint is None:
            checkpoint = _infer_checkpoint(model_id, assets_dir)
        if assets_dir is None:
            assets_dir = _infer_assets_dir(model_id, checkpoint)

        cfg = _load_yaml(self.config_path)
        cfg["model_id"] = model_id
        if _env("YETO_NAVA_MODALITY") or _env("NAVA_MODALITY"):
            cfg["modality"] = _env("YETO_NAVA_MODALITY") or _env("NAVA_MODALITY")

        model_cfg = cfg.setdefault("model", {})
        if assets_dir:
            model_cfg["ckpt_dir"] = assets_dir
            model_cfg["audio_vae_ckpt_dir"] = _expand(
                _env("YETO_NAVA_AUDIO_VAE_DIR")
                or _env("NAVA_AUDIO_VAE_DIR")
                or Path(assets_dir) / "params"
            )

        # Build the base model first, then load state_dict ourselves. NAVA's
        # in-constructor checkpoint loader expects a torch checkpoint and would
        # reject the released NAVA.safetensors file.
        self.checkpoint_path = checkpoint
        model_cfg.pop("checkpoint_path", None)

        for key in ("audio_config", "video_config", "joint_config"):
            if key in model_cfg:
                model_cfg[key] = _resolve_relative_to_root(model_cfg[key], self.nava_root)
        return cfg

    def load_pipeline(self, args, device):
        _add_python_path(self.nava_root)
        _patch_nava_tokenizer_padding()
        from nava_src.pipeline_nava import AudioVideoPipeline

        cfg = self._config(args)
        pipe = AudioVideoPipeline.create(
            model_id=cfg.get("model_id") or resolve(getattr(args, "model", "nava")),
            use_bf16=bool(cfg.get("use_bf16", True)),
            audio_latent_ch=int(cfg.get("audio_latent_ch", 128)),
            video_latent_ch=int(cfg.get("video_latent_ch", 48)),
            lambda_ddpm=float(cfg.get("lambda_ddpm", 1.0)),
            cfg=cfg,
            device=device,
        )
        if self.checkpoint_path:
            state = _load_state_dict(self.checkpoint_path)
            missing, unexpected = pipe.model.load_state_dict(state, strict=False)
            log.info(
                "loaded NAVA checkpoint %s (missing=%d unexpected=%d)",
                self.checkpoint_path,
                len(missing),
                len(unexpected),
            )
        pipe.cfg = cfg
        return pipe

    def prepare_model(self, pipe, args, device):
        for value in getattr(pipe, "components", {}).values():
            if isinstance(value, torch.nn.Module):
                value.requires_grad_(False)
                value.eval()
        pipe.model.requires_grad_(False)
        if hasattr(pipe, "switch_training_mode"):
            pipe.switch_training_mode()

        if getattr(args, "tuning", "lora") == "lora":
            from peft import LoraConfig, get_peft_model

            if getattr(args, "lora_targets", "auto") in ("auto", "all-linear"):
                target_modules = "all-linear"
            else:
                target_modules = [
                    "q",
                    "k",
                    "v",
                    "o",
                    "q_audio",
                    "k_audio",
                    "v_audio",
                    "o_audio",
                    "k_img",
                    "v_img",
                    "k_fusion",
                    "v_fusion",
                ]
            lora = LoraConfig(
                r=int(getattr(args, "lora_r", 16)),
                lora_alpha=int(getattr(args, "lora_alpha", 32)),
                target_modules=target_modules,
            )
            pipe.model = get_peft_model(pipe.model, lora)
        else:
            pipe.model.requires_grad_(True)
        pipe.to(device)
        pipe.model.train()
        return pipe

    def trainable_module_items(self, pipe):
        return [("model", pipe.model)]

    def _dataset_builder(self, pipe, args, device):
        if self._builder is not None:
            return self._builder
        from nava_src.data.dataset_train import AudioVideoDataset, DistInfo

        cfg = getattr(pipe, "cfg", {})
        data_cfg = cfg.get("data", {})
        video_fps = int(getattr(args, "fps", None) or data_cfg.get("video_fps", 24))
        target_frames = int(getattr(args, "num_frames", None) or data_cfg.get("video_tgt_frames", 121))
        batch_size = _int_or_default(getattr(args, "micro_batch_size", 1), 1)
        self._builder = AudioVideoDataset(
            "__yeto_nava_adapter__.jsonl",
            src_id2ratios={"default": [1.0, "text_to_av"]},
            modal_prob={"text_to_audio": 0.0, "text_to_video": 0.0, "text_to_image": 0.0, "text_to_av": 1.0},
            batch_size=batch_size,
            queue_size=1,
            io_workers=1,
            dist_info=DistInfo(world_rank=0, world_size=1),
            num_shards=1,
            audio_vae_server=pipe.audio_vae,
            video_vae_server=pipe.video_vae,
            image_vae_server=getattr(pipe, "image_vae", None),
            min_audio_duration=float(data_cfg.get("min_audio_duration", 0.0)),
            max_audio_duration=float(data_cfg.get("max_audio_duration", 10.0)),
            video_min_frames=int(data_cfg.get("video_min_frames", 1)),
            video_max_frames=int(data_cfg.get("video_max_frames", max(target_frames, 1))),
            video_tgt_frames=target_frames,
            video_fps=video_fps,
            add_spk_emb=bool(data_cfg.get("add_spk_emb", False)),
            spk_emb_prob=float(data_cfg.get("spk_emb_prob", 0.9)),
            use_speech_special_token=bool(data_cfg.get("use_speech_special_token", False)),
            audio_tokens_per_sec=float(data_cfg.get("audio_tokens_per_sec", 25)),
            split_wav_mode=bool(data_cfg.get("split_wav_mode", False)),
        )
        return self._builder

    def _samples_from_rows(self, pipe, rows, args, device) -> list[dict]:
        direct = []
        pending = []
        for row in rows:
            if row.get("audio_latents") is not None or row.get("video_latents") is not None:
                direct.append(dict(row))
            else:
                pending.append(row)

        samples = direct
        if pending:
            builder = self._dataset_builder(pipe, args, device)
            for row in pending:
                built = builder._build_out_av(_nava_json_row(row, args))
                if isinstance(built, list):
                    samples.extend(built)
                elif built:
                    samples.append(built)
        if not samples:
            raise ValueError("NAVA adapter could not build any trainable samples from this batch")
        return samples

    def build_batch(self, pipe, rows, args, device):
        samples = self._samples_from_rows(pipe, rows, args, device)
        batch = {
            "captions": [sample.get("captions", "") for sample in samples],
            "audio_latents": [sample.get("audio_latents") for sample in samples],
            "image_latents": [sample.get("image_latents") for sample in samples],
            "video_latents": [sample.get("video_latents") for sample in samples],
            "spk_embs": [sample.get("spk_embs") or [] for sample in samples],
        }
        if any(x is not None for x in batch["audio_latents"]):
            batch["audio_seq_len"] = [
                int(x.shape[-1]) if x is not None and hasattr(x, "shape") else 0
                for x in batch["audio_latents"]
            ]
        else:
            batch["audio_latents"] = None
        if any(x is not None for x in batch["image_latents"]):
            batch["t_h_w_list"] = [
                tuple(x.shape[:3]) if x is not None and hasattr(x, "shape") else (0, 0, 0)
                for x in batch["image_latents"]
            ]
        else:
            batch["image_latents"] = None
        if any(x is not None for x in batch["video_latents"]):
            batch["t_h_w_list"] = [
                tuple(x.shape[:3]) if x is not None and hasattr(x, "shape") else (0, 0, 0)
                for x in batch["video_latents"]
            ]
        else:
            batch["video_latents"] = None
        return batch

    def training_step(self, pipe, rows, args, device, global_step=0):
        batch = self.build_batch(pipe, rows, args, device)
        loss, _logs = pipe.forward(batch, global_step=global_step)
        return loss, torch.ones((), device=device)

    compute_loss = training_step

    def save_adapters(self, pipe, output_dir):
        out = Path(os.path.expanduser(output_dir))
        out.mkdir(parents=True, exist_ok=True)
        module = _module_to_save(pipe.model)
        if hasattr(module, "save_pretrained"):
            module.save_pretrained(out / "model")
        else:
            torch.save(module.state_dict(), out / "model_state.pt")

    def load_adapters(self, pipe, adapter_dir, meta, args):
        del meta, args
        adapter_dir = Path(adapter_dir)
        model_dir = adapter_dir / "model"
        if model_dir.exists():
            from peft import PeftModel

            pipe.model = PeftModel.from_pretrained(pipe.model, str(model_dir))
            return pipe
        state_path = adapter_dir / "model_state.pt"
        if state_path.exists():
            pipe.model.load_state_dict(torch.load(state_path, map_location="cpu"), strict=False)
            return pipe
        raise FileNotFoundError(f"{adapter_dir}: no NAVA adapter weights found")

    def load_sample_pipeline(self, adapter_dir, meta, args, device):
        values = {
            "model": getattr(args, "model", None) or meta.get("model") or "nava",
            "tuning": "full",
            "lora_r": (meta.get("lora") or {}).get("r") or 16,
            "lora_alpha": (meta.get("lora") or {}).get("alpha") or 32,
            "lora_targets": (meta.get("lora") or {}).get("targets") or "auto",
        }
        pipe = self.load_pipeline(SimpleNamespace(**values), device)
        self.load_adapters(pipe, Path(adapter_dir), meta, args)
        return self.prepare_sample_pipeline(pipe, adapter_dir, meta, args, device)

    def prepare_sample_pipeline(self, pipe, adapter_dir, meta, args, device):
        del adapter_dir, meta, args
        pipe.to(device)
        pipe.model.eval()
        return pipe

    def _sample_batch(self, pipe, args) -> dict:
        prompt = str(getattr(args, "prompt", "") or "")
        cfg = getattr(pipe, "cfg", {})
        patch_size = int(cfg.get("patch_size", 2))
        height = int(getattr(args, "height", None) or cfg.get("image_size", 960))
        width = int(getattr(args, "width", None) or cfg.get("log_width", height))
        fps = int(getattr(args, "fps", None) or cfg.get("data", {}).get("video_fps", 24))
        frames = int(getattr(args, "num_frames", None) or cfg.get("data", {}).get("video_tgt_frames", 121))
        latent_frames = max(1, (frames - 1) // 4 + 1)
        h = height // patch_size
        w = width // patch_size
        audio_len = math.ceil(((latent_frames - 1) * 4 + 1) / fps * cfg.get("data", {}).get("audio_tokens_per_sec", 25))
        return {
            "captions": [prompt.replace("<S>", "<S><extra_id_2>")],
            "video_latents": [torch.randn((latent_frames, h, w, _latent_channels(pipe, "video_latent_ch", 48)))],
            "audio_latents": [torch.randn((audio_len, _latent_channels(pipe, "audio_latent_ch", 48)))],
            "image_latents": None,
            "spk_embs": [[]],
            "t_h_w_list": [(latent_frames, h, w)],
        }

    def sample(self, pipe, args, meta):
        del meta
        device = next(pipe.model.parameters()).device
        dtype = diffusion_torch_dtype(device)
        batch = self._sample_batch(pipe, args)
        batch = {
            key: [
                value.to(device=device, dtype=dtype) if torch.is_tensor(value) else value
                for value in val
            ]
            if isinstance(val, list)
            else val
            for key, val in batch.items()
        }
        with torch.no_grad():
            video, audio = pipe.sample(
                batch,
                num_steps=int(getattr(args, "num_inference_steps", None) or 25),
                video_guidance_scale=float(getattr(args, "guidance_scale", None) or 3.0),
                audio_guidance_scale=float(getattr(args, "audio_guidance_scale", 2.0)),
                decode=True,
            )
        frames = self._pil_frames(video)
        return {"frames": frames, "audio": audio}

    @staticmethod
    def _pil_frames(video):
        from PIL import Image

        tensor = video[0] if isinstance(video, (list, tuple)) else video
        if torch.is_tensor(tensor):
            tensor = tensor.detach().float().cpu()
            if tensor.ndim == 5:
                tensor = tensor[0]
            if tensor.ndim == 4 and tensor.shape[1] in (1, 3):
                tensor = tensor.permute(0, 2, 3, 1)
            tensor = ((tensor.clamp(-1, 1) + 1.0) * 127.5).byte().numpy()
            return [Image.fromarray(frame.squeeze(-1) if frame.shape[-1] == 1 else frame) for frame in tensor]
        return tensor


def make_adapter(**kwargs) -> NavaAdapter:
    return NavaAdapter(**kwargs)
