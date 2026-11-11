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


__all__ = ["DEFAULT_REGISTRY", "REGISTRY_ENV", "image_registry", "pinned_image"]
