"""Load a trained diffusion adapter artifact and generate samples."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

from ..models import resolve

DIFFUSION_ADAPTER_METADATA_FILE = "yeto_diffusion_adapter.json"
DIFFUSION_ADAPTER_SCHEMA_VERSION = 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sample from a Yeto diffusion adapter artifact")
    p.add_argument("--adapter-dir", required=True, help="directory written by yeto.diffusion.learner")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True, help="output image file or frame directory")
    p.add_argument("--model", default=None, help="optional base model override")
    p.add_argument(
        "--diffusion-adapter",
        default=None,
        help="optional module:factory or file.py:factory hook for non-standard artifacts",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "f32"], default="auto")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args(argv)


def read_adapter_metadata(adapter_dir: str | Path) -> dict:
    path = Path(os.path.expanduser(str(adapter_dir))) / DIFFUSION_ADAPTER_METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path}: missing diffusion adapter metadata")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("kind") != "yeto.diffusion.adapter":
        raise ValueError(f"{path}: expected kind 'yeto.diffusion.adapter', got {meta.get('kind')!r}")
    if meta.get("schema_version") != DIFFUSION_ADAPTER_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {meta.get('schema_version')!r}")
    return meta


def _adapter_spec(args, meta: dict) -> str | None:
    return getattr(args, "diffusion_adapter", None) or meta.get("diffusion_adapter")


def _load_external_adapter(spec: str | None):
    if not spec:
        return None
    from .learner import load_diffusion_adapter

    return load_diffusion_adapter(spec)


def _select_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_dtype(dtype: str, device):
    import torch

    if dtype == "bf16" or (dtype == "auto" and device.type == "cuda"):
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _artifact_model_id(args, meta: dict) -> str:
    if getattr(args, "model", None):
        return resolve(args.model)
    model = meta.get("resolved_model") or meta.get("model")
    if not model:
        raise ValueError("diffusion adapter metadata has no base model id")
    return resolve(model)


def _load_base_pipeline(model_id: str, device, dtype):
    from diffusers import DiffusionPipeline

    try:
        pipe = DiffusionPipeline.from_pretrained(model_id, local_files_only=True, torch_dtype=dtype)
    except Exception:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    return pipe.to(device) if hasattr(pipe, "to") else pipe


def _looks_like_peft_adapter(path: Path) -> bool:
    return (
        (path / "adapter_config.json").exists()
        or (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )


def _set_pipeline_module(pipe, name: str, module) -> None:
    setattr(pipe, name, module)
    components = getattr(pipe, "components", None)
    if isinstance(components, dict):
        components[name] = module


def _load_default_adapters(pipe, adapter_dir: Path, meta: dict):
    loaded = False
    for name in meta.get("trainable_modules") or []:
        path = adapter_dir / name
        if not path.exists() or not hasattr(pipe, name) or not _looks_like_peft_adapter(path):
            continue
        from peft import PeftModel

        _set_pipeline_module(pipe, name, PeftModel.from_pretrained(getattr(pipe, name), path))
        loaded = True
    if loaded:
        return pipe
    if hasattr(pipe, "load_lora_weights"):
        pipe.load_lora_weights(str(adapter_dir))
        return pipe
    if (adapter_dir / "trainable_state.pt").exists():
        raise RuntimeError(
            f"{adapter_dir}: artifact contains raw trainable_state.pt; "
            "provide --diffusion-adapter with a load_adapters() hook"
        )
    raise FileNotFoundError(f"{adapter_dir}: no loadable diffusion adapter weights found")


def load_artifact_pipeline(adapter_dir: str | Path, args):
    adapter_dir = Path(os.path.expanduser(str(adapter_dir)))
    meta = read_adapter_metadata(adapter_dir)
    device = _select_device(getattr(args, "device", None))
    adapter = _load_external_adapter(_adapter_spec(args, meta))
    if adapter is not None:
        for name in ("load_sample_pipeline", "load_pipeline_for_sampling"):
            fn = getattr(adapter, name, None)
            if fn is not None:
                pipe = fn(adapter_dir, meta, args, device)
                return pipe, meta, adapter

    pipe = _load_base_pipeline(
        _artifact_model_id(args, meta),
        device,
        _torch_dtype(getattr(args, "dtype", "auto"), device),
    )
    if adapter is not None and hasattr(adapter, "load_adapters"):
        loaded = adapter.load_adapters(pipe, adapter_dir, meta, args)
        pipe = pipe if loaded is None else loaded
    else:
        pipe = _load_default_adapters(pipe, adapter_dir, meta)
    if adapter is not None and hasattr(adapter, "prepare_sample_pipeline"):
        prepared = adapter.prepare_sample_pipeline(pipe, adapter_dir, meta, args, device)
        pipe = pipe if prepared is None else prepared
    if hasattr(pipe, "to"):
        pipe = pipe.to(device)
    if hasattr(pipe, "eval"):
        pipe.eval()
    return pipe, meta, adapter


def _accepts(params, name: str) -> bool:
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _pipeline_kwargs(pipe, args):
    sig = inspect.signature(pipe.__call__)
    params = sig.parameters
    kwargs = {}
    if _accepts(params, "prompt"):
        kwargs["prompt"] = args.prompt
    for name in ("num_inference_steps", "guidance_scale", "height", "width", "num_frames"):
        value = getattr(args, name, None)
        if value is not None and _accepts(params, name):
            kwargs[name] = value
    if getattr(args, "seed", None) is not None and _accepts(params, "generator"):
        import torch

        device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs["generator"] = torch.Generator(device=device).manual_seed(args.seed)
    return kwargs


def run_sample(pipe, args, meta: dict, adapter=None):
    if adapter is not None and hasattr(adapter, "sample"):
        return adapter.sample(pipe, args, meta)
    kwargs = _pipeline_kwargs(pipe, args)
    if "prompt" not in kwargs:
        return pipe(args.prompt, **kwargs)
    return pipe(**kwargs)


def _primary_output(value: Any):
    if isinstance(value, dict):
        for key in ("images", "frames", "videos"):
            if key in value:
                return value[key]
    for attr in ("images", "frames", "videos"):
        out = getattr(value, attr, None)
        if out is not None:
            return out
    if isinstance(value, tuple) and value:
        return value[0]
    return value


def _is_pil_image(value) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    return isinstance(value, Image.Image)


def _flatten_frames(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        first = value[0]
        if isinstance(first, (list, tuple)):
            return list(first)
    return list(value) if isinstance(value, (list, tuple)) else None


def save_sample_output(sample, output_path: str | Path, *, fps: int = 8) -> list[Path]:
    value = _primary_output(sample)
    path = Path(os.path.expanduser(str(output_path)))
    if _is_pil_image(value):
        path.parent.mkdir(parents=True, exist_ok=True)
        value.save(path)
        return [path]

    frames = _flatten_frames(value)
    if frames and all(_is_pil_image(frame) for frame in frames):
        if len(frames) == 1 and path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(path)
            return [path]
        if path.suffix.lower() in (".gif", ".mp4", ".webm"):
            import imageio.v3 as iio
            import numpy as np

            path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(path, [np.array(frame) for frame in frames], fps=fps)
            return [path]
        path.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, frame in enumerate(frames):
            frame_path = path / f"frame_{i:06d}.png"
            frame.save(frame_path)
            saved.append(frame_path)
        return saved

    raise TypeError(f"cannot save diffusion output of type {type(value).__name__}")


def main(argv=None) -> int:
    args = parse_args(argv)
    pipe, meta, adapter = load_artifact_pipeline(args.adapter_dir, args)
    output = run_sample(pipe, args, meta, adapter)
    saved = save_sample_output(output, args.output, fps=args.fps)
    print(f"[yeto] wrote {len(saved)} diffusion sample file(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
