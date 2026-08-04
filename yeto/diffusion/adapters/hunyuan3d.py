"""Yeto adapter for Tencent Hunyuan3D-2.1 shape diffusion.

Hunyuan3D-2.1 is an image-to-3D asset generator, not a standard image/video
Diffusers pipeline. The public shape pipeline is diffusers-like for inference,
but owns its 3D latent/conditioner/scheduler contract. This adapter keeps that
model-specific behavior outside Yeto core and exposes deterministic trainable
tensors to Yeto's DiLoCo sync loop.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from yeto.diffusion.adapters.base import DiffusionAdapter

if TYPE_CHECKING:
    import torch

HUNYUAN3D_MODEL_ID = "tencent/Hunyuan3D-2.1"
HUNYUAN3D_SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-1"


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("Hunyuan3D adapter requires torch") from exc


def _expand(value: str | os.PathLike | None) -> str | None:
    return str(Path(os.path.expanduser(str(value))).resolve()) if value else None


def _add_python_path(path: str | None) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _maybe_add_hunyuan_paths() -> None:
    root = _expand(os.environ.get("YETO_HUNYUAN3D_ROOT") or os.environ.get("HUNYUAN3D_ROOT"))
    if not root:
        return
    _add_python_path(root)
    _add_python_path(str(Path(root) / "hy3dshape"))
    _add_python_path(str(Path(root) / "hy3dpaint"))


def _device_string(device) -> str:
    return str(device) if device is not None else "cuda"


def _dtype_from_args(args, device):
    torch = _torch()
    dtype = getattr(args, "dtype", "auto")
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "f32":
        return torch.float32
    if getattr(device, "type", str(device)) == "cpu":
        return torch.float32
    return torch.float16


def _resolve_path(value: str | os.PathLike, base_dir: str | os.PathLike | None = None) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute() and base_dir:
        path = Path(base_dir) / path
    return path


def _tensorize(value, device):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _tensorize(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensorize(item, device) for item in value)
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float, bool)) for item in value):
            return torch.tensor(value, device=device)
        return [_tensorize(item, device) for item in value]
    return value


def _maybe_load_image(value, base_dir: str | os.PathLike | None = None):
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        return value
    path = _resolve_path(value, base_dir)
    if not path.exists():
        return str(value)
    from PIL import Image

    return Image.open(path).convert("RGBA")


def _read_first_jsonl_row(path_value: str | os.PathLike | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = _resolve_path(path_value)
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _training_config_from_args(args) -> Path | None:
    explicit = os.environ.get("YETO_HUNYUAN3D_TRAIN_CONFIG")
    if explicit:
        return _resolve_path(explicit).resolve()
    row = _read_first_jsonl_row(getattr(args, "data", None))
    if row is None or not row.get("config"):
        return None
    return _resolve_path(row["config"], row.get("__yeto_data_root__")).resolve()


def _call_training_loss(fn, batch, global_step: int):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "global_step" in params:
        return fn(batch, global_step=global_step)
    if "batch_idx" in params:
        return fn(batch, batch_idx=global_step)
    positional = [
        param
        for param in params.values()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        and param.default is param.empty
    ]
    if len(positional) >= 2:
        return fn(batch, global_step)
    return fn(batch)


class NativeHunyuan3DPipeline:
    """Wrapper around the Hunyuan3D shape pipeline for Yeto hooks."""

    def __init__(self, pipe) -> None:
        self.pipe = pipe
        self.model = pipe.model

    def __getattr__(self, name: str):
        return getattr(self.pipe, name)

    def to(self, device):
        if hasattr(self.pipe, "to"):
            self.pipe.to(device)
        return self

    def build_batch(self, rows, args, device):
        if len(rows) != 1:
            raise RuntimeError(
                "Native Hunyuan3D batches must be pre-collated; use micro-batch 1 "
                "or store a complete Hunyuan3D batch per Yeto row."
            )
        row = rows[0]
        if "hunyuan3d_batch" in row:
            return _tensorize(row["hunyuan3d_batch"], device)
        batch_path = row.get("hunyuan3d_batch_path") or row.get("batch_path")
        if batch_path:
            torch = _torch()
            path = _resolve_path(batch_path, row.get("__yeto_data_root__"))
            return _tensorize(torch.load(path, map_location="cpu", weights_only=True), device)
        raise RuntimeError(
            "Hunyuan3D training rows need a native 'hunyuan3d_batch' or "
            "'hunyuan3d_batch_path'. Use YETO_HUNYUAN3D_WRAPPER for custom "
            "raw-image/mesh feature construction."
        )

    def training_step(self, batch, global_step=0):
        for name in ("training_step", "compute_loss"):
            fn = getattr(self.pipe, name, None)
            if fn is not None:
                return _call_training_loss(fn, batch, global_step)
        raise RuntimeError(
            "Native Hunyuan3D training needs prebatched rows plus a pipeline/wrapper "
            "with training_step(batch, global_step=...) or compute_loss(...)."
        )

    compute_loss = training_step


class NativeHunyuan3DTrainingPipeline(NativeHunyuan3DPipeline):
    """Wrapper around Hunyuan3D's Lightning training module."""

    native_training = True

    def __init__(self, model, config_path: Path) -> None:
        self.pipe = model
        self.model = model
        self.config_path = config_path

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    def to(self, device):
        if hasattr(self.model, "to"):
            self.model.to(device)
        return self

    def training_step(self, batch, global_step=0):
        del global_step
        forward = getattr(self.model, "forward", None)
        if forward is not None:
            return forward(batch)
        compute_loss = getattr(self.model, "compute_loss", None)
        if compute_loss is not None:
            return _call_training_loss(compute_loss, batch, 0)
        raise RuntimeError(
            "Hunyuan3D training module has no forward(batch) or compute_loss(...) "
            "method usable outside a PyTorch Lightning Trainer."
        )

    compute_loss = training_step


class Hunyuan3DAdapter(DiffusionAdapter):
    """Adapter for Hunyuan3D-2.1 shape generation and trainable state sync."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        subfolder: str | None = None,
        backend: Any | None = None,
        wrapper_module: str | None = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("YETO_HUNYUAN3D_MODEL_ID") or HUNYUAN3D_MODEL_ID
        self.subfolder = (
            subfolder
            or os.environ.get("YETO_HUNYUAN3D_SUBFOLDER")
            or HUNYUAN3D_SHAPE_SUBFOLDER
        )
        self.backend = backend
        self.wrapper_module = wrapper_module or os.environ.get("YETO_HUNYUAN3D_WRAPPER")

    def load_pipeline(self, args, device):
        if self.backend is not None:
            return self.backend
        if self.wrapper_module:
            module = importlib.import_module(self.wrapper_module)
            load = getattr(module, "load_pipeline", None)
            if load is None:
                raise RuntimeError(f"{self.wrapper_module} has no load_pipeline() function")
            return load(args, device, model_id=self.model_id, subfolder=self.subfolder)

        _maybe_add_hunyuan_paths()
        training_config = _training_config_from_args(args)
        if training_config is not None:
            try:
                from hy3dshape.utils import get_config_from_file, instantiate_from_config
            except ImportError as exc:
                raise RuntimeError(
                    "Hunyuan3D training config was found, but Hunyuan3D training "
                    "dependencies are not importable. Set YETO_HUNYUAN3D_ROOT/"
                    "HUNYUAN3D_ROOT to the Hunyuan3D-2.1 checkout."
                ) from exc
            config = get_config_from_file(str(training_config))
            model = instantiate_from_config(config.model)
            return NativeHunyuan3DTrainingPipeline(model, training_config)

        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Hunyuan3D is not importable. Install Tencent-Hunyuan/Hunyuan3D-2.1 "
                "or set YETO_HUNYUAN3D_ROOT/HUNYUAN3D_ROOT to its checkout."
            ) from exc
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.model_id,
            device=_device_string(device),
            dtype=_dtype_from_args(args, device),
            subfolder=self.subfolder,
        )
        return NativeHunyuan3DPipeline(pipe)

    def prepare_model(self, pipe, args, device):
        model = getattr(pipe, "model", pipe)
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        for name in ("vae", "conditioner"):
            module = getattr(pipe, name, None)
            if hasattr(module, "requires_grad_"):
                module.requires_grad_(False)
            if hasattr(module, "eval"):
                module.eval()

        tuning = getattr(args, "tuning", "lora")
        if tuning == "lora" and not any(p.requires_grad for p in model.parameters()):
            from peft import LoraConfig, get_peft_model

            target_modules = "all-linear"
            if getattr(args, "lora_targets", "auto") == "attention":
                target_modules = ("q", "k", "v", "q_proj", "k_proj", "v_proj", "to_q", "to_k", "to_v")
            peft_model = get_peft_model(
                model,
                LoraConfig(
                    r=getattr(args, "lora_r", 16),
                    lora_alpha=getattr(args, "lora_alpha", getattr(args, "lora_r", 16)),
                    target_modules=target_modules,
                ),
            )
            setattr(pipe, "model", peft_model)
            if hasattr(pipe, "pipe"):
                setattr(pipe.pipe, "model", peft_model)
        elif tuning == "full":
            model.requires_grad_(True)

        if hasattr(pipe, "to"):
            pipe.to(device)
        trainable = getattr(pipe, "model", pipe)
        if hasattr(trainable, "train"):
            trainable.train()
        return pipe

    def trainable_module_items(self, pipe):
        torch = _torch()
        model = getattr(pipe, "model", pipe)
        if isinstance(model, torch.nn.Module):
            return [("hunyuan3d.model", model)]
        return []

    def trainable_params(self, pipe) -> dict[str, torch.Tensor]:
        custom = getattr(pipe, "trainable_params", None)
        if custom is not None:
            return dict(custom())
        params = {}
        for module_name, module in self.trainable_module_items(pipe):
            for name, param in module.named_parameters():
                if getattr(param, "requires_grad", False):
                    params[f"{module_name}.{name}"] = param
        return params

    def build_batch(self, pipe, rows, args, device):
        build = getattr(pipe, "build_batch", None)
        if build is not None:
            return build(rows, args=args, device=device)
        return rows

    def training_step(self, pipe, rows, args, device, global_step: int = 0):
        for name in ("training_step", "compute_loss"):
            fn = getattr(pipe, name, None)
            if fn is None:
                continue
            out = fn(self.build_batch(pipe, rows, args, device), global_step=global_step)
            return self._normalize_loss(out, device)
        raise RuntimeError(
            "Hunyuan3D adapter needs prebatched rows plus training_step/compute_loss, "
            "or a custom YETO_HUNYUAN3D_WRAPPER that owns the training loss."
        )

    compute_loss = training_step

    def save_adapters(self, pipe, output_dir):
        torch = _torch()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save = getattr(pipe, "save_adapters", None) or getattr(pipe, "save", None)
        if save is not None:
            save(out)
            return
        model = getattr(pipe, "model", pipe)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out / "model")
            return
        state = {name: p.detach().cpu() for name, p in self.trainable_params(pipe).items()}
        torch.save(state, out / "trainable_state.pt")

    save = save_adapters

    def load_adapters(self, pipe, adapter_dir, meta, args):
        torch = _torch()
        del meta, args
        adapter_dir = Path(adapter_dir)
        model = getattr(pipe, "model", pipe)
        peft_dir = adapter_dir / "model"
        if peft_dir.exists() and hasattr(model, "named_parameters"):
            from peft import PeftModel

            loaded = PeftModel.from_pretrained(model, peft_dir)
            setattr(pipe, "model", loaded)
            if hasattr(pipe, "pipe"):
                setattr(pipe.pipe, "model", loaded)
            return pipe
        state_path = adapter_dir / "trainable_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            current = self.trainable_params(pipe)
            for name, tensor in state.items():
                if name in current:
                    current[name].data.copy_(tensor.to(device=current[name].device, dtype=current[name].dtype))
        return pipe

    def load_sample_pipeline(self, adapter_dir, meta, args, device):
        pipe = self.load_pipeline(args, device)
        self.load_adapters(pipe, adapter_dir, meta, args)
        return self.prepare_sample_pipeline(pipe, adapter_dir, meta, args, device)

    load_pipeline_for_sampling = load_sample_pipeline

    def prepare_sample_pipeline(self, pipe, adapter_dir, meta, args, device):
        del adapter_dir, meta, args
        if hasattr(pipe, "to"):
            pipe.to(device)
        return pipe

    def sample(self, pipe, args, meta):
        del meta
        image = self._sample_image_arg(args)
        kwargs = {}
        for name in ("num_inference_steps", "guidance_scale", "octree_resolution", "num_chunks"):
            value = getattr(args, name, None)
            if value is not None:
                kwargs[name] = value
        result = pipe(image=image, **kwargs)
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        return {"meshes": [mesh]}

    @staticmethod
    def _sample_image_arg(args):
        image = getattr(args, "image", None) or getattr(args, "prompt", None)
        if image is None:
            raise RuntimeError("Hunyuan3D sampling expects --prompt to be an input image path")
        return _maybe_load_image(image)

    @staticmethod
    def _normalize_loss(out, device):
        torch = _torch()
        if isinstance(out, tuple) and len(out) == 2:
            return out
        if isinstance(out, dict):
            loss = out.get("loss")
            denom = out.get("denominator", out.get("denom"))
            if loss is None:
                raise RuntimeError("Hunyuan3D loss dict must contain a 'loss' key")
            if denom is None:
                denom = torch.ones((), device=device)
            return loss, denom
        return out, torch.ones((), device=device)


def make_adapter(**kwargs) -> Hunyuan3DAdapter:
    return Hunyuan3DAdapter(**kwargs)
