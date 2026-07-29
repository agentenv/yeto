"""Adapter contract for non-standard diffusion training pipelines.

Adapters exist for models whose training or sampling contract cannot be covered
by the generic Diffusers path. They must stay outside Yeto core unless the
behavior is genuinely model-agnostic.

Yeto keeps adapters duck-typed: the learner checks whether a hook exists before
calling it. For that reason ``DiffusionAdapter`` is a marker base class and does
not define hook methods at runtime. If inherited methods existed, optional hooks
would look implemented and the learner would call placeholders.

Use ``DiffusionAdapter`` as the runtime base class. Use
``DiffusionAdapterProtocol`` only for type checking or documentation; do not use
the protocol as the runtime base for adapters passed to Yeto.

Integration modes:

- Load-only adapter: implement ``load_pipeline`` when Diffusers cannot load the
  repo directly, but the generic VAE/text/scheduler/denoiser loop still works.
- Preparation adapter: implement ``prepare_model`` and/or
  ``trainable_module_items`` when LoRA attachment or trainable module discovery
  differs from the generic path.
- Encoding adapter: implement ``encode_latents`` and/or ``encode_prompt_embeds``
  when media/text/audio feature construction is not exposed by standard
  Diffusers helpers.
- Denoiser-contract adapter: implement ``denoiser_kwargs`` and/or
  ``align_prediction_and_target`` when a model family has forward or output
  semantics that cannot be inferred safely from public signatures and tensor
  layouts alone.
- Full-step adapter: implement ``training_step`` or ``compute_loss`` when the
  model owns the whole batch -> loss flow.
- Artifact/sampling adapter: implement save/load/sample hooks when the artifact
  layout or generation API is not a standard PEFT/Diffusers pipeline.

Stable sync contract:

- ``trainable_module_items`` names and ``trainable_params`` keys must be stable
  across all learners, restarts, and syncer-checkpoint export.
- Adapter randomness should use the process RNGs seeded by Yeto rather than
  creating unseeded generators when reproducible diffusion runs are requested.
- Returned loss denominators must be scalar tensors on the training device.
- Adapter hooks should not start launchers, syncers, storage uploads, or
  cross-learner communication. Yeto owns orchestration and DiLoCo sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol

if TYPE_CHECKING:
    import torch


class DiffusionAdapter:
    """Runtime marker base for Yeto diffusion adapters.

    This class deliberately has no hook methods. Add methods directly on your
    adapter subclass only when you want Yeto to call them.
    """


class DiffusionAdapterProtocol(Protocol):
    """Static description of every hook currently recognized by Yeto.

    Do not subclass this protocol for runtime adapters unless every method is
    actually implemented. The learner/sample code uses ``hasattr`` to decide
    which hooks to call.
    """

    def load_pipeline(self, args, device) -> Any:
        """Load a pipeline/model object instead of ``DiffusionPipeline``."""
        ...

    def prepare_model(self, pipe, args, device) -> Any:
        """Freeze, attach adapters, move to device, and return the pipeline."""
        ...

    def trainable_module_items(self, pipe) -> Iterable[tuple[str, torch.nn.Module]]:
        """Return named modules whose trainable params participate in sync."""
        ...

    def trainable_params(self, pipe) -> Mapping[str, torch.Tensor]:
        """Return custom trainable tensor mapping when module traversal is insufficient."""
        ...

    def encode_latents(self, pipe, rows: list[dict], args, device, dtype) -> Any:
        """Return ``yeto.diffusion.learner.LatentBatch`` for a row batch."""
        ...

    def encode_prompt_embeds(self, pipe, rows: list[dict], args, device, dtype) -> Any:
        """Return ``yeto.diffusion.learner.TextConditioning`` for a row batch."""
        ...

    def denoiser_kwargs(
        self,
        pipe,
        model,
        noisy,
        cond,
        args,
        params: Mapping[str, Any],
        kwargs: Mapping[str, Any],
        *,
        pixel_height: int | None,
        pixel_width: int | None,
    ) -> Mapping[str, Any]:
        """Return model-specific denoiser keyword contributions or overrides."""
        ...

    def align_prediction_and_target(
        self,
        pipe,
        pred: torch.Tensor,
        target: torch.Tensor,
        noisy,
        cond,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return a model-specific loss alignment, or ``None`` to use core."""
        ...

    def training_step(self, pipe, rows: list[dict], args, device, global_step: int = 0):
        """Return ``(loss, denominator)`` for a complete custom training step."""
        ...

    def compute_loss(self, pipe, rows: list[dict], args, device, global_step: int = 0):
        """Alias accepted by the learner for ``training_step``."""
        ...

    def save_adapters(self, pipe, output_dir: str | Path) -> None:
        """Persist custom adapter weights under ``output_dir``."""
        ...

    def save(self, pipe, output_dir: str | Path) -> None:
        """Alias accepted by the learner for ``save_adapters``."""
        ...

    def load_sample_pipeline(self, adapter_dir: str | Path, meta: dict, args, device) -> Any:
        """Load a complete sample-ready pipeline from an adapter artifact."""
        ...

    def load_pipeline_for_sampling(self, adapter_dir: str | Path, meta: dict, args, device) -> Any:
        """Alias accepted by the sampler for ``load_sample_pipeline``."""
        ...

    def load_adapters(self, pipe, adapter_dir: str | Path, meta: dict, args) -> Any:
        """Load adapter weights into an already loaded pipeline."""
        ...

    def prepare_sample_pipeline(self, pipe, adapter_dir: str | Path, meta: dict, args, device) -> Any:
        """Finalize a pipeline after adapter load and before sampling."""
        ...

    def sample(self, pipe, args, meta: dict) -> dict:
        """Run model-specific generation and return sample outputs."""
        ...
