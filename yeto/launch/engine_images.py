"""Engine images by CPU architecture (S25 N0.2, N1.6).

yeto defines one image per (device family, CPU architecture) pair. Each image
has a Dockerfile and a version lock under ``docker/<name>/``; the lock is the
single source of the versions the image asserts at start
(``docker/image-assert/assert_image_versions.py``).

A node of the SSH node pool declares its ``cpu_arch``. The launcher then uses
this catalog in two ways:

* a node that names no image gets the inventory's image for its architecture;
* a node whose image is a catalog image of another architecture or another
  device family is refused before any SSH (an aarch64 node cannot run an
  x86_64 image, and a PPU image has no Ascend runtime).

An image reference that is not in the catalog (a site image) is accepted as
it is: yeto cannot tell its architecture from the name, and the node
preflight (``uname -m``) still checks the node itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The two architectures yeto runs on. Aliases are the names Docker and Go
# use for the same machines.
CPU_ARCHES = ("x86_64", "aarch64")
_ARCH_ALIASES = {"amd64": "x86_64", "x86-64": "x86_64", "arm64": "aarch64"}


class UnknownCpuArch(ValueError):
    """The CPU architecture is not one yeto runs on."""


def canonical_cpu_arch(value: str) -> str:
    """``uname -m`` spelling of ``value`` (``amd64`` -> ``x86_64``)."""
    arch = str(value).strip().lower()
    arch = _ARCH_ALIASES.get(arch, arch)
    if arch not in CPU_ARCHES:
        raise UnknownCpuArch(f"cpu_arch {value!r} is unknown (known: {list(CPU_ARCHES)})")
    return arch


@dataclass(frozen=True)
class ImageSpec:
    """One yeto engine image. ``repository`` is the last path segment of the
    image reference (``REGISTRY/<repository>:TAG``)."""

    name: str
    repository: str
    cpu_arch: str
    device_family: str
    roles: tuple[str, ...]

    @property
    def lock_path(self) -> Path:
        """``docker/<name>/versions.lock.json`` of a source checkout."""
        return Path(__file__).resolve().parents[2] / "docker" / self.name / "versions.lock.json"


IMAGES: dict[str, ImageSpec] = {spec.name: spec for spec in (
    # PPU trainer and SGLang sampler (stage 1): Alibaba PPU nodes are x86_64.
    ImageSpec("ppu-train", "yeto-ppu-train", "x86_64", "ppu", ("trainer", "sampler")),
    # Ascend 910B sampler (stage 2): Kunpeng hosts are aarch64.
    ImageSpec("npu-sampler", "yeto-npu-sampler", "aarch64", "npu", ("sampler",)),
)}


def _repository(reference: str) -> str:
    """``REGISTRY/a/b:tag@sha256:..`` -> ``b``."""
    path = reference.split("@", 1)[0]
    last = path.rsplit("/", 1)[-1]
    return last.split(":", 1)[0]


def catalog_image(reference: str) -> ImageSpec | None:
    """The catalog image ``reference`` points at, or None for a site image."""
    repository = _repository(reference)
    return next((spec for spec in IMAGES.values() if spec.repository == repository), None)


def image_mismatch(reference: str, *, cpu_arch: str, device_family: str) -> str | None:
    """Why ``reference`` cannot run on a node of this kind, or None when it can
    (or when the reference is not a catalog image)."""
    spec = catalog_image(reference)
    if spec is None:
        return None
    if spec.cpu_arch != cpu_arch:
        return (f"image {reference} is the {spec.cpu_arch} image {spec.name}; "
                f"the node is {cpu_arch}")
    if spec.device_family != device_family:
        return (f"image {reference} is the {spec.device_family} image {spec.name}; "
                f"the node is device_family {device_family}")
    return None
