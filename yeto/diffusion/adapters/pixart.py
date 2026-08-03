"""PixArt-family denoiser and learned-sigma output contracts.

PixArt uses the generic Diffusers loading, encoding, scheduler, LoRA, and
artifact paths. Only the two semantics below belong here: optional resolution
and aspect-ratio conditioning, and the learned-sigma output channel layout.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import DiffusionAdapter


def _config_value(config, name: str):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    try:
        return config[name]
    except (KeyError, TypeError):
        return getattr(config, name, None)


def _declares_pixart(value: Any) -> bool:
    if value is None:
        return False
    declarations: list[Any] = [type(value).__name__, type(value).__module__]
    config = getattr(value, "config", None)
    for name in ("_class_name", "architectures", "model_type"):
        declared = _config_value(config, name)
        if isinstance(declared, (list, tuple)):
            declarations.extend(declared)
        elif declared is not None:
            declarations.append(declared)
    return any("pixart" in str(item).lower() for item in declarations)


def _model_candidates(pipe, model=None):
    pending = [model, pipe]
    pending.extend(
        getattr(pipe, name, None)
        for name in ("transformer", "transformer_2", "unet", "model")
    )
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for name in ("module", "_fsdp_wrapped_module", "_checkpoint_wrapped_module"):
            wrapped = getattr(candidate, name, None)
            if wrapped is not None:
                pending.append(wrapped)
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                pending.append(get_base_model())
            except TypeError:
                pass
        base_model = getattr(candidate, "base_model", None)
        if base_model is not None:
            pending.append(getattr(base_model, "model", base_model))


class PixArtAdapter(DiffusionAdapter):
    """Small behavior adapter layered over Yeto's generic Diffusers path."""

    @classmethod
    def applies_to(cls, pipe, model=None) -> bool:
        """Identify PixArt from public pipeline/model class declarations."""
        return any(_declares_pixart(candidate) for candidate in _model_candidates(pipe, model))

    def denoiser_kwargs(
        self,
        pipe,
        model,
        noisy,
        cond,
        args,
        params,
        kwargs,
        *,
        pixel_height,
        pixel_width,
    ):
        """Supply PixArt's optional micro-conditioning payload."""
        del pipe, cond, args
        additional_conditions = getattr(model, "use_additional_conditions", None)
        if additional_conditions is None:
            additional_conditions = _config_value(
                getattr(model, "config", None),
                "use_additional_conditions",
            )
        if (
            "added_cond_kwargs" not in params
            or "added_cond_kwargs" in kwargs
            or not additional_conditions
        ):
            return {}
        if pixel_height is None or pixel_width is None:
            raise RuntimeError(
                "denoiser requires additional size conditions, but pixel height/width "
                "are unavailable; set --height/--width or provide latent shape metadata"
            )

        batch = int(noisy.latents.shape[0])
        resolution = torch.tensor(
            [[int(pixel_height), int(pixel_width)]],
            device=noisy.latents.device,
            dtype=noisy.latents.dtype,
        ).repeat(batch, 1)
        aspect_ratio = torch.full(
            (batch, 1),
            float(pixel_height) / float(pixel_width),
            device=noisy.latents.device,
            dtype=noisy.latents.dtype,
        )
        return {
            "added_cond_kwargs": {
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
            }
        }

    def align_prediction_and_target(self, pipe, pred, target, noisy, cond):
        """Select the prediction half of PixArt's learned-sigma output."""
        del pipe, noisy, cond
        if (
            pred.ndim == target.ndim == 4
            and pred.shape[0] == target.shape[0]
            and pred.shape[2:] == target.shape[2:]
            and pred.shape[1] == 2 * target.shape[1]
        ):
            return pred[:, : target.shape[1]], target
        return None


def make_adapter() -> PixArtAdapter:
    """Return the in-tree PixArt behavior adapter."""
    return PixArtAdapter()
