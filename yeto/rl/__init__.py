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
    "TMS_PRELOAD_PATCH_REPOSITORY",
    "TMS_PRELOAD_PATCH_BASE_COMMIT",
    "TMS_PRELOAD_PATCH_COMMIT",
    "TMS_PRELOAD_BASE_BINARY_SHA256",
    "default_rl_image",
))


def __getattr__(name: str):
    if name in _MILES_PIN_NAMES:
        from .adapters.miles import pins

        return getattr(pins, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

