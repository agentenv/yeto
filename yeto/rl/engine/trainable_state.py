"""Layout-tagged trainable state generalizing ``CanonicalLoraState``.

The LoRA layout is a thin wrapper around :class:`yeto.rl.core.CanonicalLoraState`
and hashes through :func:`yeto.rl.core.policy_hash`, so identities are identical
across the legacy and ports representations. torch is imported lazily.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from yeto.rl.core import CanonicalLoraState

LAYOUT_LORA = "lora"
LAYOUT_FULL = "full"
LAYOUT_FRAGMENTS = "fragments"
TRAINABLE_LAYOUTS = frozenset({LAYOUT_LORA, LAYOUT_FULL, LAYOUT_FRAGMENTS})
IMPLEMENTED_LAYOUTS = frozenset({LAYOUT_LORA})


class UnsupportedLayoutError(ValueError):
    """A representable trainable layout that this engine path does not implement."""

    def __init__(self, layout: str, supported: frozenset[str] = IMPLEMENTED_LAYOUTS):
        super().__init__(
            f"unsupported trainable-state layout {layout!r}; "
            f"supported: {sorted(supported)}"
        )
        self.layout = layout


def require_supported_layout(
    layout: str, supported: frozenset[str] | set[str] = IMPLEMENTED_LAYOUTS
) -> str:
    """Startup check: reject unknown layouts and representable-but-unimplemented ones."""

    if layout not in TRAINABLE_LAYOUTS:
        raise ValueError(
            f"unknown trainable-state layout {layout!r}; "
            f"known: {sorted(TRAINABLE_LAYOUTS)}"
        )
    if layout not in supported:
        raise UnsupportedLayoutError(layout, frozenset(supported))
    return layout


@dataclass(frozen=True)
class TrainableState:
    """Layout id + deterministically ordered tensors + policy version.

    ``config_hash`` is the layout-specific config identity (for LoRA, the
    canonical LoRA config hash); ``layout_hash`` is the tensor-layout identity.
    """

    layout: str
    base_model_revision: str
    config_hash: str
    layout_hash: str
    policy_version: int
    tensors: Mapping[str, Any]
    _lora: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        require_supported_layout(self.layout)
        if self._lora is None:
            object.__setattr__(self, "_lora", self._build_lora())

    def _build_lora(self) -> "CanonicalLoraState":
        from yeto.rl.core import CanonicalLoraState

        return CanonicalLoraState(
            base_model_revision=self.base_model_revision,
            lora_config_hash=self.config_hash,
            layout_hash=self.layout_hash,
            policy_version=self.policy_version,
            tensors=self.tensors,
        )

    @classmethod
    def from_lora(cls, state: "CanonicalLoraState") -> "TrainableState":
        return cls(
            layout=LAYOUT_LORA,
            base_model_revision=state.base_model_revision,
            config_hash=state.lora_config_hash,
            layout_hash=state.layout_hash,
            policy_version=state.policy_version,
            tensors=state.tensors,
            _lora=state,
        )

    def to_lora(self) -> "CanonicalLoraState":
        if self.layout != LAYOUT_LORA:
            raise UnsupportedLayoutError(self.layout)
        return self._lora

    @property
    def tensor_names(self) -> tuple[str, ...]:
        """Deterministic tensor order (sorted by name, as in ``canonical_specs``)."""

        return tuple(sorted(self.tensors))

    def policy_hash(self) -> str:
        from yeto.rl.core import policy_hash

        return policy_hash(self.to_lora())

    def policy_tensor_hash(self) -> str:
        from yeto.rl.core import policy_tensor_hash

        return policy_tensor_hash(self.to_lora())
