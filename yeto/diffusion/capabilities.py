"""Static diffusion capability audit.

This module is intentionally descriptive. It is not a training registry and the
learner must not branch on it for model behavior. The matrix exists to keep the
generic Diffusers support plan explicit before spending GPU time on one model at
a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CapabilityStatus = Literal[
    "generic-covered",
    "generic-gap",
    "adapter-required",
    "needs-real-validation",
]

VALID_CAPABILITY_STATUSES: tuple[CapabilityStatus, ...] = (
    "generic-covered",
    "generic-gap",
    "adapter-required",
    "needs-real-validation",
)


@dataclass(frozen=True)
class DiffusionCapability:
    family: str
    pipeline: str
    denoisers: tuple[str, ...]
    modalities: tuple[str, ...]
    conditioning: tuple[str, ...]
    latent_layout: str
    scheduler: str
    forward_kwargs: tuple[str, ...]
    output_alignment: str
    status: CapabilityStatus
    gaps: tuple[str, ...] = ()
    adapter_boundary: str = ""
    evidence: tuple[str, ...] = ()


def _cap(
    *,
    family: str,
    pipeline: str,
    denoisers: tuple[str, ...] = ("transformer",),
    modalities: tuple[str, ...],
    conditioning: tuple[str, ...],
    latent_layout: str,
    scheduler: str,
    forward_kwargs: tuple[str, ...],
    output_alignment: str,
    status: CapabilityStatus,
    gaps: tuple[str, ...] = (),
    adapter_boundary: str = "",
    evidence: tuple[str, ...] = (),
) -> DiffusionCapability:
    return DiffusionCapability(
        family=family,
        pipeline=pipeline,
        denoisers=denoisers,
        modalities=modalities,
        conditioning=conditioning,
        latent_layout=latent_layout,
        scheduler=scheduler,
        forward_kwargs=forward_kwargs,
        output_alignment=output_alignment,
        status=status,
        gaps=gaps,
        adapter_boundary=adapter_boundary,
        evidence=evidence,
    )


_FLUX_CONDITIONING = ("prompt_embeds", "pooled_prompt_embeds", "txt_ids", "img_ids", "guidance")
_FLOW_MATCH = "flow matching sigmas/timesteps, velocity target"
_PACKED_2D = "packed 2D latent tokens"


DIFFUSION_CAPABILITIES: dict[str, DiffusionCapability] = {
    "flux": _cap(
        family="Flux",
        pipeline="FluxPipeline",
        modalities=("image", "text"),
        conditioning=_FLUX_CONDITIONING,
        latent_layout=_PACKED_2D,
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "pooled_projections", "txt_ids", "img_ids", "guidance"),
        output_alignment="pipeline _unpack_latents helper",
        status="needs-real-validation",
        gaps=("generic output unpack/alignment is implemented; real Flux validation is pending",),
        evidence=("diffusers examples/dreambooth/train_dreambooth_lora_flux.py", "finetrainers FluxModelSpecification"),
    ),
    "flux-schnell": _cap(
        family="Flux",
        pipeline="FluxPipeline",
        modalities=("image", "text"),
        conditioning=_FLUX_CONDITIONING,
        latent_layout=_PACKED_2D,
        scheduler="timestep-distilled Flux variant",
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "pooled_projections", "txt_ids", "img_ids", "guidance"),
        output_alignment="pipeline _unpack_latents helper",
        status="generic-covered",
        gaps=("production-quality target distribution still needs recipe-level validation",),
        evidence=("AWS g6e Flux-Schnell raw-image 20-step LoRA train/backward/save/reload", "finetrainers Flux notes"),
    ),
    "flux2-dev": _cap(
        family="Flux2",
        pipeline="Flux2Pipeline",
        modalities=("image", "text"),
        conditioning=("prompt_embeds", "txt_ids", "img_ids", "guidance"),
        latent_layout="patchified and packed 2D latent tokens",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "txt_ids", "img_ids", "guidance"),
        output_alignment="pipeline _unpack_latents_with_ids helper",
        status="needs-real-validation",
        gaps=("generic output unpack with ids is implemented; real Flux2 validation is pending",),
        evidence=("diffusers examples/dreambooth/train_dreambooth_lora_flux2.py",),
    ),
    "chroma1-base": _cap(
        family="Chroma1",
        pipeline="ChromaPipeline",
        modalities=("image", "text"),
        conditioning=_FLUX_CONDITIONING,
        latent_layout=_PACKED_2D,
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "attention_mask", "txt_ids", "img_ids", "guidance"),
        output_alignment="Flux-like packed output",
        status="needs-real-validation",
        gaps=("contract is Flux-like but not validated on real Chroma weights",),
        evidence=("Diffusers Flux-like forward signatures",),
    ),
    "chroma1-hd": _cap(
        family="Chroma1",
        pipeline="ChromaPipeline",
        modalities=("image", "text"),
        conditioning=_FLUX_CONDITIONING,
        latent_layout=_PACKED_2D,
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "attention_mask", "txt_ids", "img_ids", "guidance"),
        output_alignment="Flux-like packed output",
        status="needs-real-validation",
        gaps=("contract is Flux-like but not validated on real Chroma weights",),
        evidence=("Diffusers Flux-like forward signatures",),
    ),
    "hidream-i1-dev": _cap(
        family="HiDream",
        pipeline="HiDreamImagePipeline",
        modalities=("image", "text"),
        conditioning=("prompt_embeds_t5", "prompt_embeds_llama3", "pooled_prompt_embeds"),
        latent_layout="2D image latents",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states_t5", "encoder_hidden_states_llama3", "pooled_embeds"),
        output_alignment="direct or pipeline helper unpack",
        status="needs-real-validation",
        gaps=("generic multi-encoder prompt tuple/dict mapping is implemented; real component loading needs validation",),
        evidence=("diffusers examples/dreambooth/train_dreambooth_lora_hidream.py",),
    ),
    "hidream-i1-full": _cap(
        family="HiDream",
        pipeline="HiDreamImagePipeline",
        modalities=("image", "text"),
        conditioning=("prompt_embeds_t5", "prompt_embeds_llama3", "pooled_prompt_embeds"),
        latent_layout="2D image latents",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states_t5", "encoder_hidden_states_llama3", "pooled_embeds"),
        output_alignment="direct or pipeline helper unpack",
        status="needs-real-validation",
        gaps=("generic multi-encoder prompt tuple/dict mapping is implemented; real component loading needs validation",),
        evidence=("diffusers examples/dreambooth/train_dreambooth_lora_hidream.py",),
    ),
    "ideogram4": _cap(
        family="Ideogram4",
        pipeline="Ideogram4Pipeline",
        modalities=("image", "text"),
        conditioning=("prompt_embeds", "position_ids", "segment_ids", "indicator"),
        latent_layout="packed text+image sequence",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "position_ids", "segment_ids", "indicator"),
        output_alignment="image-token mask in packed sequence",
        status="generic-covered",
        gaps=("production-quality recipe validation still pending",),
        evidence=("AWS g6e Ideogram4 raw-image 20-step LoRA train/backward/save/reload", "Verda A100 ideogram4 one-step train/backward/save", "diffusers examples/dreambooth/train_dreambooth_lora_ideogram4.py"),
    ),
    "qwen-image": _cap(
        family="Qwen Image",
        pipeline="QwenImagePipeline",
        modalities=("image", "text"),
        conditioning=("prompt_embeds", "prompt_embeds_mask", "img_shapes"),
        latent_layout=_PACKED_2D,
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "encoder_hidden_states_mask", "img_shapes"),
        output_alignment="pipeline _unpack_latents helper",
        status="needs-real-validation",
        gaps=("generic prompt mask forwarding and output unpack are implemented; full Qwen-Image exceeded single L40S 46GB before training",),
        evidence=("AWS g6e.16xlarge Qwen-Image bf16 load OOM at pipe.to(cuda)", "diffusers examples/dreambooth/train_dreambooth_lora_qwen_image.py"),
    ),
    "qwen-image-2512": _cap(
        family="Qwen Image",
        pipeline="QwenImagePipeline",
        modalities=("image", "text"),
        conditioning=("prompt_embeds", "prompt_embeds_mask", "img_shapes"),
        latent_layout=_PACKED_2D,
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "encoder_hidden_states_mask", "img_shapes"),
        output_alignment="pipeline _unpack_latents helper",
        status="needs-real-validation",
        gaps=("generic prompt mask forwarding and output unpack are implemented; real Qwen Image validation is pending",),
        evidence=("diffusers examples/dreambooth/train_dreambooth_lora_qwen_image.py",),
    ),
    "sd35": _cap(
        family="Stable Diffusion 3.5",
        pipeline="StableDiffusion3Pipeline",
        modalities=("image", "text"),
        conditioning=("prompt_embeds", "pooled_prompt_embeds"),
        latent_layout="2D image latents",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "pooled_projections"),
        output_alignment="direct latent output",
        status="generic-covered",
        gaps=("production-quality recipe validation still pending",),
        evidence=("AWS g6e.16xlarge SD3.5 raw-image 20-step LoRA train/backward/save/reload", "diffusers examples/dreambooth/train_dreambooth_lora_sd3.py"),
    ),
    "ltx-video": _cap(
        family="LTX Video",
        pipeline="LTXPipeline",
        modalities=("video", "text"),
        conditioning=("prompt_embeds", "encoder_attention_mask"),
        latent_layout="packed 3D video latent tokens",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask", "rope_interpolation_scale"),
        output_alignment="packed-token target",
        status="generic-covered",
        gaps=("production-quality recipe validation still pending",),
        evidence=("AWS g6e.16xlarge LTX raw-video 49-frame 20-step LoRA train/backward/save/reload", "A6000 LTX raw-video train/save/reload/sample", "finetrainers LTXVideoModelSpecification"),
    ),
    "hunyuan-video": _cap(
        family="Hunyuan Video",
        pipeline="HunyuanVideoPipeline",
        modalities=("video", "text"),
        conditioning=("prompt_embeds", "encoder_attention_mask", "pooled_prompt_embeds", "guidance"),
        latent_layout="3D video latents",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask", "pooled_projections", "guidance"),
        output_alignment="direct latent output",
        status="needs-real-validation",
        gaps=("guidance and pooled field mapping are covered at contract level only",),
        evidence=("finetrainers HunyuanVideoModelSpecification",),
    ),
    "wan21-t2v-1.3b": _cap(
        family="Wan",
        pipeline="WanPipeline",
        modalities=("video", "text"),
        conditioning=("prompt_embeds",),
        latent_layout="3D video latents",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states"),
        output_alignment="direct latent output",
        status="generic-covered",
        gaps=("production-length video validation still pending",),
        evidence=("AWS g6e Wan2.1 1.3B raw-video 20-step LoRA train/backward/save/reload", "finetrainers WanModelSpecification"),
    ),
    "wan21-t2v-14b": _cap(
        family="Wan",
        pipeline="WanPipeline",
        modalities=("video", "text"),
        conditioning=("prompt_embeds",),
        latent_layout="3D video latents",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states"),
        output_alignment="direct latent output",
        status="needs-real-validation",
        gaps=("same contract as Wan2.1 1.3B but large model needs GPU validation",),
        evidence=("finetrainers WanModelSpecification",),
    ),
    "wan22": _cap(
        family="Wan",
        pipeline="WanPipeline",
        denoisers=("transformer", "transformer_2"),
        modalities=("video", "text"),
        conditioning=("prompt_embeds",),
        latent_layout="3D video latents",
        scheduler=_FLOW_MATCH,
        forward_kwargs=("hidden_states", "timestep", "encoder_hidden_states"),
        output_alignment="direct latent output",
        status="needs-real-validation",
        gaps=("generic multi-denoiser routing is implemented; real Wan2.2 validation is pending",),
        evidence=("Wan2.2 config/source inspection",),
    ),
    "nava": _cap(
        family="NAVA",
        pipeline="external adapter",
        denoisers=("adapter-defined",),
        modalities=("video", "audio", "text"),
        conditioning=("adapter-defined",),
        latent_layout="adapter-defined",
        scheduler="adapter-defined",
        forward_kwargs=("adapter-defined",),
        output_alignment="adapter-defined",
        status="adapter-required",
        adapter_boundary="NAVA training uses package-specific audio/video pipeline behavior outside public Diffusers contracts.",
        evidence=("yeto.diffusion.adapters.nava", "AWS g6e.16xlarge NAVA 33-frame r16/a32 20-step LoRA train/backward/save/reload", "A6000/A100 NAVA adapter validation"),
    ),
}


def get_diffusion_capability(alias: str) -> DiffusionCapability | None:
    return DIFFUSION_CAPABILITIES.get(alias)


def aliases_by_status(status: CapabilityStatus) -> tuple[str, ...]:
    if status not in VALID_CAPABILITY_STATUSES:
        raise ValueError(f"unknown diffusion capability status {status!r}")
    return tuple(alias for alias, cap in DIFFUSION_CAPABILITIES.items() if cap.status == status)


def format_capability_table(aliases: tuple[str, ...] | None = None) -> str:
    selected = aliases or tuple(DIFFUSION_CAPABILITIES)
    lines = [
        "| alias | family | pipeline | status | primary gaps |",
        "|---|---|---|---|---|",
    ]
    for alias in selected:
        cap = DIFFUSION_CAPABILITIES[alias]
        gaps = "<br>".join(cap.gaps) if cap.gaps else "-"
        lines.append(f"| `{alias}` | {cap.family} | `{cap.pipeline}` | `{cap.status}` | {gaps} |")
    return "\n".join(lines)
