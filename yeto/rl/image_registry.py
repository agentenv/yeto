"""Registry of yeto's default container images.

Every default image is pinned by digest; only the repository prefix is
configurable. ``YETO_IMAGE_REGISTRY`` (default ``ghcr.io/agentenv``) names the
registry/organization that hosts ``yeto-miles-k31``, ``yeto-miles-ports`` and
``yeto-sampler-k31``: point it at a mirror that carries the same digests (for
example ``<registry>/<org>``). A digest names the bytes, so a mirror cannot
serve a different image under the same pin. ``--rl-image`` (and a sampler
spec's ``image``) still override the default for one run.
"""

from __future__ import annotations

import os

REGISTRY_ENV = "YETO_IMAGE_REGISTRY"
DEFAULT_REGISTRY = "ghcr.io/agentenv"


def image_registry() -> str:
    value = os.environ.get(REGISTRY_ENV, "").strip().rstrip("/")
    return value or DEFAULT_REGISTRY


def pinned_image(name: str, digest: str, *, docker_prefix: bool = True) -> str:
    """``docker:<registry>/<name>@sha256:<digest>`` (without ``docker:`` on request)."""
    ref = f"{image_registry()}/{name}@sha256:{digest}"
    return f"docker:{ref}" if docker_prefix else ref


IMPORT_REGISTRY = image_registry()  # the registry the module-level pins (pins.MILES_*_IMAGE) were built with


def current_registry(image: str) -> str:
    """A pinned default built at import (:data:`IMPORT_REGISTRY`) on the registry
    ``YETO_IMAGE_REGISTRY`` names now: the same string unless the variable changed since
    (``yeto run up`` resolves its trainer's image under the run config's ``env``, which
    the trainer's own launch process starts with). Any other reference is returned as is."""
    now = image_registry()
    if now == IMPORT_REGISTRY:
        return image
    for prefix in ("docker:", ""):
        old = f"{prefix}{IMPORT_REGISTRY}/"
        if image.startswith(old) and "@sha256:" in image:
            return f"{prefix}{now}/{image[len(old):]}"
    return image


__all__ = ["DEFAULT_REGISTRY", "IMPORT_REGISTRY", "REGISTRY_ENV", "current_registry", "image_registry",
           "pinned_image"]
