"""Copyable templates for model-specific Yeto diffusion adapters.

Use this file as a reference, not as a production adapter. Copy the smallest
class that matches your model and delete hooks you do not need. Yeto's learner
uses duck typing: if a hook exists, Yeto assumes it is intentionally
implemented and may call it.

Launch an external adapter with:

    yeto launch ... --model-kind diffusion --diffusion-adapter path/to/adapter.py:make_adapter

Common modes:

- ``load-only``: Diffusers cannot load the repo directly, but the generic Yeto
  denoising loop is still correct after you return a pipeline.
- ``encoding``: the generic loop is correct, but media/text/audio feature
  construction needs custom code.
- ``full-step``: the model owns the whole rows -> loss training contract.
- ``sampling``: training used a custom artifact or generation API and sampling
  needs custom reload/generation hooks.
"""

from __future__ import annotations

from pathlib import Path

import torch

from yeto.diffusion.adapters.base import DiffusionAdapter


class LoadOnlyAdapter(DiffusionAdapter):
    """Adapter for custom model loading with the generic learner path.

    Implement this when ``DiffusionPipeline.from_pretrained(args.model)`` is not
    sufficient, but the returned object still exposes generic Diffusers-like
    VAE/text/scheduler/denoiser behavior.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load_pipeline(self, args, device):
        del args, device
        raise NotImplementedError("load and return the model-specific pipeline here")


class PreparationAdapter(LoadOnlyAdapter):
    """Adapter for custom trainable module discovery or LoRA attachment."""

    def prepare_model(self, pipe, args, device):
        """Freeze modules, attach LoRA/full trainables, move to device."""
        del args
        pipe.to(device)
        return pipe

    def trainable_module_items(self, pipe):
        """Return stable names for modules whose trainable params are synced."""
        return [("model", pipe.model)]

    # Use trainable_params() instead of trainable_module_items() only when a
    # model needs custom tensor names or non-module trainables.
    # def trainable_params(self, pipe):
    #     return {name: p for name, p in pipe.model.named_parameters() if p.requires_grad}


class EncodingAdapter(PreparationAdapter):
    """Adapter for custom latent/text/audio conditioning construction.

    Use this when the generic VAE or encode_prompt path is not enough, but
    Yeto's scheduler -> denoiser -> flow-matching loss loop is still correct.
    """

    def encode_latents(self, pipe, rows, args, device, dtype):
        """Return ``yeto.diffusion.learner.LatentBatch`` for one micro-batch."""
        del pipe, rows, args, device, dtype
        raise NotImplementedError("build and return LatentBatch here")

    def encode_prompt_embeds(self, pipe, rows, args, device, dtype):
        """Return ``yeto.diffusion.learner.TextConditioning`` for one micro-batch."""
        del pipe, rows, args, device, dtype
        raise NotImplementedError("build and return TextConditioning here")


class FullStepAdapter(PreparationAdapter):
    """Adapter for models whose native training step should be delegated."""

    def build_batch(self, pipe, rows, args, device):
        """Convert Yeto dataset rows into the model's native batch format."""
        del pipe, args, device
        return rows

    def training_step(self, pipe, rows, args, device, global_step=0):
        """Run a complete custom train step and return ``(loss, denominator)``."""
        batch = self.build_batch(pipe, rows, args, device)
        loss = pipe.forward(batch, global_step=global_step)
        if isinstance(loss, tuple):
            return loss
        return loss, torch.ones((), device=device)

    compute_loss = training_step


class ArtifactAdapter(FullStepAdapter):
    """Adapter for custom training artifacts and sample-time reload."""

    def save_adapters(self, pipe, output_dir):
        """Persist custom trainable weights.

        Generic PEFT LoRA adapters do not need this hook; Yeto saves them
        automatically. Use it for custom modules or multiple state dicts.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(pipe.model.state_dict(), out / "model_state.pt")

    save = save_adapters

    def load_adapters(self, pipe, adapter_dir, meta, args):
        """Reload custom weights into an already loaded pipeline for sampling."""
        del meta, args
        state = torch.load(Path(adapter_dir) / "model_state.pt", map_location="cpu")
        pipe.model.load_state_dict(state, strict=False)
        return pipe


class SamplingAdapter(ArtifactAdapter):
    """Adapter for custom sample-time pipeline setup or generation."""

    def load_sample_pipeline(self, adapter_dir, meta, args, device):
        """Load a complete sample-ready pipeline from an adapter artifact.

        Implement this when the generic sampler cannot load the base pipeline
        before adapter weights are applied.
        """
        pipe = self.load_pipeline(args, device)
        self.load_adapters(pipe, adapter_dir, meta, args)
        return self.prepare_sample_pipeline(pipe, adapter_dir, meta, args, device)

    load_pipeline_for_sampling = load_sample_pipeline

    def prepare_sample_pipeline(self, pipe, adapter_dir, meta, args, device):
        """Finalize the pipeline after adapter load and before sampling."""
        del adapter_dir, meta, args
        pipe.to(device)
        pipe.eval()
        return pipe

    def sample(self, pipe, args, meta):
        """Run model-specific generation.

        Return a mapping understood by ``yeto.diffusion.sample`` such as
        ``{"images": [...]}``, ``{"frames": [...]}``, or ``{"audio": ...}``.
        """
        del meta
        return pipe(args.prompt)


_MODES = {
    "load-only": LoadOnlyAdapter,
    "preparation": PreparationAdapter,
    "encoding": EncodingAdapter,
    "full-step": FullStepAdapter,
    "artifact": ArtifactAdapter,
    "sampling": SamplingAdapter,
}


def make_adapter(mode: str = "full-step", **kwargs) -> DiffusionAdapter:
    """Factory used by ``--diffusion-adapter``.

    Keep this function in copied adapters. Yeto loads ``module:make_adapter`` and
    expects an object whose methods are the hooks you intentionally implement.
    """
    try:
        cls = _MODES[mode]
    except KeyError as exc:
        raise ValueError(f"unknown template adapter mode {mode!r}; choices: {sorted(_MODES)}") from exc
    return cls(**kwargs)
