"""Pinned Miles reinforcement-learning integration."""

# Miles pins moved to ``yeto.rl.adapters.miles.pins`` (decoupling 5.3, audit L4).
# Re-exported lazily so importing the neutral core never loads the adapter.
_MILES_PIN_NAMES = frozenset((
    "MILES_PEFT_VERSION",
    "MILES_NEXT_REPOSITORY",
    "MILES_NEXT_UPSTREAM_COMMIT",
    "MILES_NEXT_COMMIT",
    "SGLANG_NEXT_REPOSITORY",
    "SGLANG_NEXT_UPSTREAM_COMMIT",
    "SGLANG_NEXT_COMMIT",
    "MILES_NEXT_PINS",
    "MILES_NEXT_BASE_IMAGE",
    "MILES_NEXT_IMAGE",
    "MILES_NEXT_IMAGE_MANIFEST",
    "NEBIUS_BAKED_IMAGES",
    "MILES_NEXT_IMAGE_SGLANG_ROOT",
    "MilesRevisionPins",
    "MILES_FORK_LINEAGE",
    "CRITIC_FORK_LINEAGE",
    "miles_fork_lineage_from",
    "MilesLineage",
    "MILES_LINEAGES",
    "CURRENT_LINEAGE",
    "PORTS_V1_FEATURES",
    "K31_FEATURES",
    "lineage_of",
    "current_miles_commit",
    "pin_has",
    "MILES_PORTS_V1_UPSTREAM_COMMIT",
    "MILES_PORTS_V1_COMMIT",
    "MILES_PORTS_V1_PINS",
    "MILES_PORTS_V1_IMAGE",
    "MILES_K31_UPSTREAM_COMMIT",
    "MILES_K31_F1_COMMIT",
    "MILES_K31_B2_COMMIT",
    "MILES_K31_B3B7_COMMIT",
    "MILES_K31_B4_COMMIT",
    "MILES_K31_B8_COMMIT",
    "commits_with",
    "SGLANG_PORTS_V1_REPOSITORY",
    "SGLANG_PORTS_V1_UPSTREAM_COMMIT",
    "SGLANG_PORTS_V1_COMMIT",
    "SGLANG_K31_REPOSITORY",
    "SGLANG_K31_BRANCH",
    "SGLANG_K31_COMMIT",
    "TMS_PRELOAD_PATCH_REPOSITORY",
    "TMS_PRELOAD_PATCH_BASE_COMMIT",
    "TMS_PRELOAD_PATCH_COMMIT",
    "TMS_PRELOAD_BASE_BINARY_SHA256",
    "default_rl_image",
))


# Engine-sampler pins (neutral: ``yeto.rl.sampler_pins``).
_SAMPLER_PIN_NAMES = frozenset((
    "SGLANG_SAMPLER_UPSTREAM_REPOSITORY",
    "SGLANG_SAMPLER_UPSTREAM_COMMIT",
    "SGLANG_SAMPLER_BASE_IMAGE_CUDA",
    "SGLANG_SAMPLER_YETO_COMMIT",
    "SGLANG_SAMPLER_IMAGE_CUDA",
    "SGLANG_SAMPLER_IMAGE_MANIFEST",
    "SGLANG_SAMPLER_ROCM_PR",
    "SGLANG_SAMPLER_ROCM_COMMIT",
    "SGLANG_SAMPLER_IMAGE_ROCM",
    "SGLANG_SAMPLER_IMAGE_ROCM_TAG",
    "SGLANG_SAMPLER_ENGINE_VERSIONS",
    "sampler_sglang_commit",
    "VLLM_SAMPLER_UPSTREAM_REPOSITORY",
    "VLLM_SAMPLER_INTERFACE_COMMIT",
    "VLLM_SAMPLER_IMAGE_ROCM",
    "VLLM_SAMPLER_IMAGE_ROCM_TAG",
    "VLLM_SAMPLER_ROCM_VERSION",
    "VLLM_SAMPLER_IMAGE_ROCM_NIGHTLY",
    "VLLM_SAMPLER_BASE_IMAGE_CUDA",
    "VLLM_SAMPLER_BASE_IMAGE_CUDA_TAG",
    "VLLM_SAMPLER_CUDA_VERSION",
    "VLLM_SAMPLER_YETO_COMMIT",
    "VLLM_SAMPLER_IMAGE_CUDA",
    "VLLM_SAMPLER_IMAGE_MANIFEST",
    "VLLM_SAMPLER_MAIN_COMMIT",
    "VLLM_SAMPLER_MAIN_VERSION",
    "VLLM_SAMPLER_MAIN_WHEEL_CUDA",
    "VLLM_SAMPLER_ENGINE_VERSIONS",
    "sampler_vllm_version",
))


def __getattr__(name: str):
    if name in _SAMPLER_PIN_NAMES:
        from . import sampler_pins

        return getattr(sampler_pins, name)
    if name in _MILES_PIN_NAMES:
        from .adapters.miles import pins

        return getattr(pins, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

