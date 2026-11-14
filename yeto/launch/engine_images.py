"""Engine images by CPU architecture.

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


# --- default engine image per (device family, role, engine) ------------------
#
# hetero-1114 A22. Each entry names a pins constant ("module:NAME") instead of
# copying its digest, so the pins modules stay the single source. An entry
# whose ref is None has no pinned image yet: the caller must name one
# (``reason`` says why). Nothing reads this table unless a config asks for a
# default by device family (``--rl-island-engine`` / a sampler pool's
# ``platform``); the run-wide defaults (``--rl-image`` omitted, the NC/Daytona
# sampler constants) are unchanged.
DEVICE_FAMILIES = ("cuda", "rocm", "ascend", "ppu")
_FAMILY_ALIASES = {"nvidia": "cuda", "gpu": "cuda", "amd": "rocm", "npu": "ascend"}
ROLES = ("trainer", "sampler")


@dataclass(frozen=True)
class DefaultImage:
    device_family: str
    role: str
    engine: str          # trainer: the --rl-backend; sampler: sglang | vllm | vllm-ascend
    ref: str | None      # "module:CONSTANT" of a pinned image, or None (none pinned)
    reason: str = ""     # why there is no pinned image (ref None) / a caveat


DEFAULT_IMAGES: tuple[DefaultImage, ...] = (
    DefaultImage("cuda", "trainer", "miles", "yeto.rl.adapters.miles.pins:MILES_NEXT_IMAGE"),
    DefaultImage("cuda", "trainer", "verl", "yeto.rl.adapters.verl.pins:VERL_IMAGE"),
    DefaultImage("cuda", "sampler", "sglang", "yeto.rl.sampler_pins:SGLANG_SAMPLER_IMAGE_CUDA"),
    DefaultImage("cuda", "sampler", "vllm", "yeto.rl.sampler_pins:VLLM_SAMPLER_IMAGE_CUDA"),
    DefaultImage("rocm", "trainer", "miles", None,
                 "no ROCm trainer image is pinned (ROCm runs samplers only)"),
    DefaultImage("rocm", "sampler", "sglang", "yeto.rl.sampler_pins:SGLANG_SAMPLER_IMAGE_ROCM"),
    DefaultImage("rocm", "sampler", "vllm", "yeto.rl.sampler_pins:VLLM_SAMPLER_IMAGE_ROCM"),
    DefaultImage("ascend", "trainer", "verl", None,
                 "the verl NPU stack (yeto.rl.adapters.verl.pins:VERL_NPU_BASE_IMAGE, CANN "
                 "9.1.0 + vllm-ascend 0.23) has no digest-pinned image verified on hardware; "
                 "name the site image"),
    DefaultImage("ascend", "sampler", "vllm-ascend", None,
                 "no digest-pinned vllm-ascend image; name the site image"),
    DefaultImage("ppu", "trainer", "miles", None,
                 "ppu-train (docker/ppu-train) is built per site; name its digest"),
    DefaultImage("ppu", "sampler", "sglang", None,
                 "ppu-train (docker/ppu-train) is built per site; name its digest"),
)


class NoDefaultImage(ValueError):
    """No pinned default image for this (device family, role, engine)."""


def canonical_device_family(value: str) -> str:
    family = str(value).strip().lower()
    family = _FAMILY_ALIASES.get(family, family)
    if family not in DEVICE_FAMILIES:
        raise ValueError(f"device family {value!r} is unknown (known: {list(DEVICE_FAMILIES)})")
    return family


def default_image_entry(device_family: str, role: str, engine: str | None = None) -> DefaultImage:
    """The table row of (family, role, engine); ``engine`` None takes the
    family's first row for the role."""
    family = canonical_device_family(device_family)
    if role not in ROLES:
        raise ValueError(f"role {role!r} is unknown (known: {list(ROLES)})")
    rows = [r for r in DEFAULT_IMAGES if r.device_family == family and r.role == role
            and (engine is None or r.engine == engine)]
    if not rows:
        known = sorted({r.engine for r in DEFAULT_IMAGES if r.device_family == family and r.role == role})
        raise NoDefaultImage(f"no {role} engine {engine!r} for device family {family} (known: {known})")
    return rows[0]


def resolve_ref(ref: str) -> str:
    """``module:CONSTANT`` -> the constant's value."""
    import importlib

    module, _, name = ref.partition(":")
    value = getattr(importlib.import_module(module), name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{ref} is not an image reference")
    return value


def default_image(device_family: str, role: str, engine: str | None = None) -> str:
    """The pinned default image of (family, role, engine). Raises
    :class:`NoDefaultImage` when none is pinned (the reason says what to pass)."""
    row = default_image_entry(device_family, role, engine)
    if row.ref is None:
        raise NoDefaultImage(f"no pinned {role} image for {row.device_family}/{row.engine}: {row.reason}")
    return resolve_ref(row.ref)
