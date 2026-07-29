"""Optional diffusion adapters for non-standard pipelines."""

from .base import DiffusionAdapter, DiffusionAdapterProtocol
from .pixart import PixArtAdapter

_BUILTIN_BEHAVIOR_ADAPTERS = (PixArtAdapter(),)


def diffusion_behavior_adapters(pipe, model=None, adapter=None) -> tuple[DiffusionAdapter, ...]:
    """Return matching in-tree behavior adapters followed by an explicit one.

    Keeping family detection in this package lets the generic learner probe
    hooks without learning model names. An explicitly selected adapter is last
    so its denoiser keyword contributions can override in-tree defaults.
    """
    matched = [
        candidate
        for candidate in _BUILTIN_BEHAVIOR_ADAPTERS
        if not isinstance(adapter, type(candidate)) and candidate.applies_to(pipe, model)
    ]
    if adapter is not None:
        matched.append(adapter)
    return tuple(matched)


__all__ = [
    "DiffusionAdapter",
    "DiffusionAdapterProtocol",
    "PixArtAdapter",
    "diffusion_behavior_adapters",
]
