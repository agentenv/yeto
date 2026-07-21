"""Load a trained diffusion adapter artifact and generate samples."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..models import resolve

DIFFUSION_ADAPTER_METADATA_FILE = "yeto_diffusion_adapter.json"
DIFFUSION_ADAPTER_SCHEMA_VERSION = 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sample from a Yeto diffusion adapter artifact")
    p.add_argument("--adapter-dir", required=True, help="directory written by yeto.diffusion.learner")
    p.add_argument("--prompt", default=None)
    p.add_argument("--output", default=None, help="single-sample output image file or frame directory")
    p.add_argument("--data", default=None, help="HF/local/cloud-mounted prompt dataset for batch sampling")
    p.add_argument("--data-revision", default=None)
    p.add_argument("--prompt-column", default="prompt")
    p.add_argument("--seed-column", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--output-dir", default=None, help="batch-sampling output directory")
    p.add_argument("--manifest", default=None, help="batch JSONL manifest path")
    p.add_argument("--model", default=None, help="optional base model override")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--source-sha256", default=None, help=argparse.SUPPRESS)
    for provenance_flag in (
        "model-requested-identifier",
        "model-requested-revision",
        "data-requested-identifier",
        "data-requested-revision",
    ):
        p.add_argument(f"--{provenance_flag}", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--diffusion-adapter",
        default=None,
        help="optional module:factory or file.py:factory hook for non-standard artifacts",
    )
    p.add_argument("--diffusion-adapter-sha256", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--allow-unattested-legacy-adapter",
        action="store_true",
        help="allow explicit adapter code for a legacy artifact with no training-time digest",
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


def validate_args(args) -> None:
    if args.data:
        if not args.output_dir:
            raise ValueError("--data batch sampling needs --output-dir")
        if args.output:
            raise ValueError("--data batch sampling writes per-row files under --output-dir; omit --output")
        return
    if not args.prompt or not args.output:
        raise ValueError("single-sample mode needs --prompt and --output, or pass --data --output-dir")


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
    explicit = getattr(args, "diffusion_adapter", None)
    if explicit:
        return explicit
    if meta.get("diffusion_adapter"):
        raise PermissionError(
            "the artifact requests executable diffusion adapter code; pass the "
            "same --diffusion-adapter explicitly after reviewing it"
        )
    return None


def _adapter_specs_match(artifact_spec: str, runtime_spec: str) -> bool:
    artifact_target, artifact_separator, artifact_factory = artifact_spec.partition(":")
    runtime_target, runtime_separator, runtime_factory = runtime_spec.partition(":")
    if not artifact_separator or not runtime_separator:
        return False
    if artifact_factory != runtime_factory:
        return False
    artifact_is_file = artifact_target.endswith(".py") or os.path.sep in artifact_target
    runtime_is_file = runtime_target.endswith(".py") or os.path.sep in runtime_target
    if artifact_is_file or runtime_is_file:
        # Synced/cloud paths legitimately differ. The source digest binds the
        # file bytes, while the factory name binds the selected entry point.
        return artifact_is_file and runtime_is_file
    return artifact_target == runtime_target


def _load_external_adapter(spec: str | None, expected_sha256: str | None = None):
    if not spec:
        return None
    from .learner import load_diffusion_adapter

    return load_diffusion_adapter(spec, expected_sha256=expected_sha256)


def _select_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_dtype(dtype: str, device):
    import torch

    if dtype == "auto":
        from .learner import diffusion_torch_dtype

        return diffusion_torch_dtype(device)
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _artifact_model_id(args, meta: dict) -> str:
    if getattr(args, "model", None):
        return resolve(args.model)
    recorded = (meta.get("provenance") or {}).get("model") or {}
    model = recorded.get("resolved_identifier")
    top_level_models = {
        name: meta.get(name)
        for name in ("model", "resolved_model")
        if meta.get(name)
    }
    if not model or not top_level_models:
        raise ValueError(
            "diffusion adapter metadata has no complete base-model identity; "
            "pass --model and --model-revision to override a legacy artifact"
        )
    resolved_model = resolve(model)
    inconsistent = {}
    requested_model = recorded.get("requested_identifier")
    if "model" in top_level_models:
        top_requested = top_level_models["model"]
        if requested_model is not None:
            if top_requested != requested_model:
                inconsistent["model"] = top_requested
        elif resolve(top_requested) != resolved_model:
            inconsistent["model"] = top_requested
    if (
        "resolved_model" in top_level_models
        and resolve(top_level_models["resolved_model"]) != resolved_model
    ):
        inconsistent["resolved_model"] = top_level_models["resolved_model"]
    if inconsistent:
        raise ValueError(
            "diffusion adapter metadata has inconsistent base-model identities: "
            f"top-level {inconsistent!r}, provenance {model!r}"
        )
    return resolved_model


def _load_base_pipeline(
    model_id: str,
    device,
    dtype,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
):
    from diffusers import DiffusionPipeline

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "use_safetensors": True,
    }
    if revision:
        kwargs["revision"] = revision
    try:
        pipe = DiffusionPipeline.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs)
    return pipe.to(device) if hasattr(pipe, "to") else pipe


def _looks_like_peft_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (
        path / "adapter_model.safetensors"
    ).exists()


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
    unsafe_bins = sorted(adapter_dir.glob("*.bin"))
    if unsafe_bins:
        raise RuntimeError(
            f"{adapter_dir}: legacy binary adapter weights are not accepted; "
            "convert them to safetensors before sampling"
        )
    if (adapter_dir / "trainable_state.safetensors").exists() or (
        adapter_dir / "trainable_state.pt"
    ).exists():
        raise RuntimeError(
            f"{adapter_dir}: artifact contains raw trainable state; "
            "provide --diffusion-adapter with a load_adapters() hook"
        )
    if hasattr(pipe, "load_lora_weights"):
        pipe.load_lora_weights(str(adapter_dir))
        return pipe
    raise FileNotFoundError(f"{adapter_dir}: no loadable diffusion adapter weights found")


def load_artifact_pipeline(adapter_dir: str | Path, args):
    adapter_dir = Path(os.path.expanduser(str(adapter_dir)))
    meta = read_adapter_metadata(adapter_dir)
    device = _select_device(getattr(args, "device", None))
    model_id = _artifact_model_id(args, meta)
    from ..provenance import (
        apply_runtime_provenance,
        is_immutable_commit,
        resolve_reference,
    )

    artifact_model = (meta.get("provenance") or {}).get("model") or {}
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        requested_revision = getattr(args, "model_revision", None)
        original_identifier = (
            getattr(args, "model_requested_identifier", None) or explicit_model
        )
    else:
        recorded_source = artifact_model.get("source")
        requested_revision = artifact_model.get("resolved_revision")
        if recorded_source == "huggingface":
            if not is_immutable_commit(requested_revision):
                raise ValueError(
                    "remote diffusion artifact provenance requires a full immutable "
                    "model commit; pass --model and --model-revision to override a "
                    "legacy artifact"
                )
        elif recorded_source not in {"local", "external-uri"}:
            raise ValueError(
                "diffusion artifact provenance has an invalid model source; pass "
                "--model and --model-revision to override a legacy artifact"
            )
        elif requested_revision is not None:
            raise ValueError(
                "local or external diffusion artifact provenance cannot carry a "
                "model revision"
            )
        original_identifier = (
            artifact_model.get("requested_identifier")
            or meta.get("model")
            or model_id
        )
    model_provenance = resolve_reference(
        model_id,
        requested_revision,
        repo_type="model",
        original_identifier=original_identifier,
    )
    if getattr(args, "model_requested_revision", None) is not None:
        model_provenance["requested_revision"] = args.model_requested_revision
    runtime_provenance = {
        "schema_version": 1,
        "model": model_provenance,
        "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
    }
    apply_runtime_provenance(args, runtime_provenance)
    # Custom sampling hooks receive the same resolved model identity as the
    # default loader, even when the operator did not pass an explicit override.
    args.model = model_id
    adapter_spec = _adapter_spec(args, meta)
    if adapter_spec:
        from ..provenance import python_spec_sha256

        actual_adapter_sha256 = python_spec_sha256(adapter_spec)
        transport_adapter_sha256 = getattr(
            args, "diffusion_adapter_sha256", None
        )
        artifact_adapter_sha256 = meta.get("diffusion_adapter_sha256")
        artifact_adapter_spec = meta.get("diffusion_adapter")
        if (
            artifact_adapter_spec is not None
            and not _adapter_specs_match(artifact_adapter_spec, adapter_spec)
        ):
            raise ValueError(
                "runtime diffusion adapter spec does not match the training artifact"
            )
        if (
            transport_adapter_sha256 is not None
            and artifact_adapter_sha256 is not None
            and transport_adapter_sha256.lower() != artifact_adapter_sha256.lower()
        ):
            raise ValueError(
                "diffusion adapter SHA256 from the launcher does not match the "
                "training artifact"
            )
        expected_adapter_sha256 = (
            artifact_adapter_sha256 or transport_adapter_sha256
        )
        artifact_binding_complete = bool(
            artifact_adapter_spec and artifact_adapter_sha256
        )
        if not artifact_binding_complete and not getattr(
            args, "allow_unattested_legacy_adapter", False
        ):
            raise PermissionError(
                "legacy diffusion artifact has no complete training-time adapter "
                "spec/digest binding; "
                "pass --allow-unattested-legacy-adapter only after separately "
                "verifying that this adapter code matches the artifact"
            )
        if (
            expected_adapter_sha256 is not None
            and actual_adapter_sha256 != expected_adapter_sha256.lower()
        ):
            raise ValueError("diffusion adapter SHA256 mismatch during sampling")
        args.diffusion_adapter_sha256 = actual_adapter_sha256
    adapter = _load_external_adapter(
        adapter_spec,
        getattr(args, "diffusion_adapter_sha256", None),
    )
    if adapter is not None:
        from ..provenance import require_custom_loader_contract

        require_custom_loader_contract(adapter, args)
        for name in ("load_sample_pipeline", "load_pipeline_for_sampling"):
            fn = getattr(adapter, name, None)
            if fn is not None:
                pipe = fn(adapter_dir, meta, args, device)
                return pipe, meta, adapter

    pipe = _load_base_pipeline(
        model_id,
        device,
        _torch_dtype(getattr(args, "dtype", "auto"), device),
        revision=model_provenance["resolved_revision"],
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
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


def resolve_sample_data_arg(data):
    """Match learner data semantics without adding a diffusion storage layer."""
    if not isinstance(data, str):
        return data
    from ..datasource import kind, learner_data_arg

    data_kind = kind(data)
    if data_kind != "cloud":
        return data
    mounted = learner_data_arg(data)
    if os.path.exists(os.path.expanduser(mounted)):
        return mounted
    raise ValueError(
        f"{data!r} is a cloud data source; run sampling in a SkyPilot/Yeto task "
        f"that mounts it at {mounted!r}, or pass a local path/HF dataset id"
    )


def iter_prompt_rows(
    data,
    prompt_column: str,
    max_rows: int | None = None,
    revision: str | None = None,
):
    from ..data import load_rows

    data = resolve_sample_data_arg(data)
    ds = load_rows(data, revision=revision)
    limit = len(ds) if max_rows is None else min(len(ds), max_rows)
    for i in range(limit):
        row = dict(ds[i])
        if prompt_column not in row:
            raise KeyError(f"batch sampling row {i} has no prompt column {prompt_column!r}")
        yield i, row, str(row[prompt_column])


def _row_seed(args, row: dict, index: int) -> int | None:
    if args.seed_column and row.get(args.seed_column) is not None:
        return int(row[args.seed_column])
    return None if args.seed is None else int(args.seed) + index


def _sample_args_for_row(args, prompt: str, seed: int | None):
    values = vars(args).copy()
    values["prompt"] = prompt
    values["seed"] = seed
    return SimpleNamespace(**values)


def generation_args(args) -> dict:
    fields = (
        "num_inference_steps",
        "guidance_scale",
        "height",
        "width",
        "num_frames",
        "fps",
    )
    return {name: getattr(args, name, None) for name in fields if getattr(args, name, None) is not None}


def artifact_summary(meta: dict) -> dict:
    adapter_spec = meta.get("diffusion_adapter")
    adapter_digest = meta.get("diffusion_adapter_sha256")
    if adapter_spec and adapter_digest:
        adapter_status = "attested"
    elif adapter_spec or adapter_digest:
        adapter_status = "legacy-unattested"
    else:
        adapter_status = "not-applicable"
    return {
        "model": meta.get("model"),
        "resolved_model": meta.get("resolved_model"),
        "tuning": meta.get("tuning"),
        "trainable_modules": meta.get("trainable_modules") or [],
        "provenance": meta.get("provenance"),
        "diffusion_adapter": adapter_spec,
        "diffusion_adapter_sha256": adapter_digest,
        "adapter_attestation_status": adapter_status,
    }


def runtime_adapter_summary(args, meta: dict) -> dict:
    spec = getattr(args, "diffusion_adapter", None)
    digest = getattr(args, "diffusion_adapter_sha256", None)
    artifact_spec = meta.get("diffusion_adapter")
    artifact_digest = meta.get("diffusion_adapter_sha256")
    if spec is None:
        binding = "not-used"
    elif artifact_spec and artifact_digest:
        binding = "matched"
    else:
        binding = "legacy-unbound"
    return {
        "spec": spec,
        "sha256": digest,
        "artifact_binding": binding,
    }


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


def _array_frames(value):
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    if not isinstance(value, np.ndarray):
        return None
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3:
        value = value[None]
    if value.ndim != 4:
        return None

    frames = []
    for frame in value:
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.moveaxis(frame, 0, -1)
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0) * 255.0
        else:
            frame = np.clip(frame, 0, 255)
        frame = frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        frames.append(Image.fromarray(frame))
    return frames


def save_sample_output(sample, output_path: str | Path, *, fps: int = 8) -> list[Path]:
    value = _primary_output(sample)
    path = Path(os.path.expanduser(str(output_path)))
    if _is_pil_image(value):
        if not path.suffix:
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        value.save(path)
        return [path]

    frames = _array_frames(value) or _flatten_frames(value)
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
        if path.suffix:
            path = path.with_suffix("")
        path.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, frame in enumerate(frames):
            frame_path = path / f"frame_{i:06d}.png"
            frame.save(frame_path)
            saved.append(frame_path)
        return saved

    raise TypeError(f"cannot save diffusion output of type {type(value).__name__}")


def _relative_paths(paths: list[Path], root: Path) -> list[str]:
    out = []
    for path in paths:
        try:
            out.append(str(path.relative_to(root)))
        except ValueError:
            out.append(str(path))
    return out


def batch_sample(pipe, args, meta: dict, adapter=None) -> list[dict]:
    out_dir = Path(os.path.expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(os.path.expanduser(args.manifest)) if args.manifest else out_dir / "samples.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with manifest.open("w", encoding="utf-8") as f:
        for index, row, prompt in iter_prompt_rows(
            args.data,
            args.prompt_column,
            args.max_rows,
            getattr(args, "data_revision", None),
        ):
            seed = _row_seed(args, row, index)
            row_args = _sample_args_for_row(args, prompt, seed)
            output_prefix = out_dir / f"sample_{index:06d}.png"
            output = run_sample(pipe, row_args, meta, adapter)
            saved = save_sample_output(output, output_prefix, fps=args.fps)
            record = {
                "index": index,
                "prompt": prompt,
                "seed": seed,
                "output_paths": _relative_paths(saved, out_dir),
                "adapter_dir": str(args.adapter_dir),
                "artifact": artifact_summary(meta),
                "runtime_provenance": getattr(args, "_provenance", None),
                "runtime_adapter": runtime_adapter_summary(args, meta),
                "generation": generation_args(args),
                "input_provenance": getattr(args, "_sample_data_provenance", None),
            }
            if args.seed_column and args.seed_column in row:
                record["seed_column"] = args.seed_column
            f.write(json.dumps(record, sort_keys=True) + "\n")
            records.append(record)
    return records


def main(argv=None) -> int:
    args = parse_args(argv)
    validate_args(args)
    from ..provenance import verify_source_tree_sha256

    verify_source_tree_sha256(args.source_sha256)
    if args.data:
        from ..provenance import resolve_reference

        resolved_data = resolve_sample_data_arg(args.data)
        data_provenance = resolve_reference(
            resolved_data,
            args.data_revision,
            repo_type="dataset",
            original_identifier=getattr(
                args, "data_requested_identifier", None
            ) or args.data,
        )
        if getattr(args, "data_requested_revision", None) is not None:
            data_provenance["requested_revision"] = args.data_requested_revision
        args.data = resolved_data
        args.data_revision = data_provenance["resolved_revision"]
        args._sample_data_provenance = data_provenance
    pipe, meta, adapter = load_artifact_pipeline(args.adapter_dir, args)
    if args.data:
        records = batch_sample(pipe, args, meta, adapter)
        manifest = args.manifest or str(Path(os.path.expanduser(args.output_dir)) / "samples.jsonl")
        print(f"[yeto] wrote {len(records)} diffusion sample record(s) to {manifest}")
        return 0
    output = run_sample(pipe, args, meta, adapter)
    saved = save_sample_output(output, args.output, fps=args.fps)
    print(f"[yeto] wrote {len(saved)} diffusion sample file(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
