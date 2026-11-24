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

Engine images as capability records (unified-sampler-pool, bottom of this
module): :class:`EngineImage` records what a pinned sampler image runs on
(device family, CPU architecture, card architectures), its engine and version,
its interpreter, and per model a :class:`Recipe` (precision, the reload modes
it serves, the rollout fields it returns, the smallest node, the cards it was
verified on). :func:`resolve` picks the image of a (node profile, model, engine
preference) or raises :class:`ImageIncompatible` with what would work instead;
:func:`image_mismatch` refuses a named catalog image for a node (and, given a
model, for that model's load mode) before anything is ordered.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..gpu_spec import UNIFIED_FAMILIES, UnknownFamily, unified_family

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


def image_mismatch(reference: str, *, cpu_arch: str, device_family: str, card_arch: str | None = None,
                   model: str | None = None, load: str | None = None, precision: str | None = None,
                   rollout: Iterable[str] = (), hbm_gib_per_node: float | None = None) -> str | None:
    """Why ``reference`` cannot run on a node of this kind, or None when it can
    (or when the reference is not a catalog image: a site image is the site's).

    ``device_family`` may be any spelling (``npu`` and ``ascend`` alike). The
    keywords after it check a pinned :class:`EngineImage`: ``card_arch`` against
    its card architectures, and with ``model`` the model's :class:`Recipe`
    (precision, the reload mode ``load``, the ``rollout`` fields, the node's HBM).
    Without them the check is the one it always was (architecture and family)."""
    spec = catalog_image(reference)
    if spec is not None:
        if spec.cpu_arch != cpu_arch:
            return (f"image {reference} is the {spec.cpu_arch} image {spec.name}; "
                    f"the node is {cpu_arch}")
        if _same_family(spec.device_family, device_family) is False:
            return (f"image {reference} is the {spec.device_family} image {spec.name}; "
                    f"the node is device_family {device_family}")
        return None
    engine = engine_image_of(reference)
    if engine is None:
        return None
    reasons = engine.problems(family=device_family, cpu_arch=cpu_arch, card_arch=card_arch, model=model,
                              load=load, precision=precision, rollout=tuple(rollout),
                              hbm_gib_per_node=hbm_gib_per_node)
    if not reasons:
        return None
    return f"image {reference} is the {engine.name} image ({engine.family} {engine.engine}): " + "; ".join(reasons)


def _same_family(a: str, b: str) -> bool | None:
    """Whether two family spellings name one family (None: one of them is unknown)."""
    try:
        return unified_family(a) == unified_family(b)
    except UnknownFamily:
        return a.strip().lower() == b.strip().lower()


# --- default engine image per (device family, role, engine) ------------------
#
# hetero-1114 A22. Each entry names a pins constant ("module:NAME") instead of
# copying its digest, so the pins modules stay the single source. An entry
# whose ref is None has no pinned image yet: the caller must name one
# (``reason`` says why). Nothing reads this table unless a config asks for a
# default by device family (``--rl-island-engine`` / a sampler pool's
# ``platform``); the run-wide defaults (``--rl-image`` omitted, the NC/Daytona
# sampler constants) are unchanged.
DEVICE_FAMILIES = UNIFIED_FAMILIES  # ("cuda", "rocm", "ascend", "ppu"); aliases: gpu_spec.unified_family
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
    """``cuda`` / ``rocm`` / ``ascend`` / ``ppu`` of any family or vendor spelling
    (:func:`yeto.gpu_spec.unified_family`: ``npu`` -> ``ascend``, ``nvidia`` / ``gpu`` ->
    ``cuda``, ``amd`` -> ``rocm``)."""
    try:
        return unified_family(str(value))
    except UnknownFamily:
        raise ValueError(f"device family {value!r} is unknown (known: {list(DEVICE_FAMILIES)})") from None


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


# --- engine images as capability records (unified-sampler-pool) -------------
#
# One :class:`EngineImage` per pinned sampler image. The digests stay in
# ``yeto.rl.sampler_pins`` (its constants are read by the NC, Daytona and Modal
# routes): a record names its pin as ``"module:CONSTANT"`` and :attr:`EngineImage.ref`
# is exactly that constant, so a route that moves to :func:`resolve` pulls the
# very image it pulled before. What a record adds is what the routes hard-coded:
# the family, CPU and card architectures, engine version and interpreter, and per
# model which precision, reload modes and rollout fields the image serves.

LOAD_MODES = ("in_place", "live", "restart", "drain")
"""Reload modes of a release: ``in_place`` (the server updates its weights in
place: vLLM ``collective_rpc`` / the yeto expert reload, SGLang
``update_weights_from_disk``), ``live`` (SGLang tensor update without a drain),
``restart`` (stop the server, start it on the release) and ``drain`` (finish the
running requests, then restart)."""
ROLLOUT_FIELDS = ("routed_experts", "top_logprobs", "fp32_lm_head")
"""Per-token fields the sampler asks the engine for: R3 routed experts, top-K
log-probs (score_centering) and an fp32 lm-head. The spec keys are
``rollout_fields.routed_experts`` / ``top_logprobs_num`` / ``fp32_lm_head``."""
ENGINES = ("vllm", "sglang", "vllm-ascend")
ANY_MODEL = "*"  # a recipe that applies to any model the image has no own recipe for
MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "kimi_k31": ("kimi_k31", "kimi-k31", "kimi_k3", "kimi-k3", "kimi-k31-pro", "kimi-k3.1", "kimi-k3.1-pro",
                 "moonshotai/kimi-k3.1-pro", "kimik31forconditionalgeneration"),
}
_MODEL_OF_ALIAS = {alias: key for key, aliases in MODEL_ALIASES.items() for alias in aliases}


def model_key(model: str | None) -> str:
    """The catalog key of a model name: ``moonshotai/Kimi-K3.1-Pro``, ``kimi-k31-pro``
    and ``kimi_k31`` are ``kimi_k31``; any other name is itself, lower case
    (it matches only an :data:`ANY_MODEL` recipe). None is :data:`ANY_MODEL`."""
    if model is None or not str(model).strip():
        return ANY_MODEL
    name = str(model).strip().lower().rstrip("/")
    return _MODEL_OF_ALIAS.get(name, _MODEL_OF_ALIAS.get(name.rsplit("/", 1)[-1], name))


def canonical_load(load: str | None) -> str | None:
    if load is None:
        return None
    value = str(load).strip().lower().replace("-", "_")
    value = {"hot": "in_place", "inplace": "in_place", "cold": "restart"}.get(value, value)
    if value not in LOAD_MODES:
        raise ValueError(f"load mode {load!r} is unknown (known: {list(LOAD_MODES)})")
    return value


class ImageIncompatible(ValueError):
    """No catalog image serves the (node profile, model, engine, load) asked for.

    ``reasons``: image name -> why that image does not fit; ``alternatives``: what
    would work instead (another load mode on this node, or another kind of node)."""

    def __init__(self, message: str, *, reasons: Mapping[str, Sequence[str]] | None = None,
                 alternatives: Sequence[str] = ()):
        super().__init__(message)
        self.reasons = {k: list(v) for k, v in (reasons or {}).items()}
        self.alternatives = list(alternatives)


@dataclass(frozen=True)
class Recipe:
    """What an image serves for one model (``model``: a :func:`model_key` or
    :data:`ANY_MODEL`).

    ``loads``: the reload modes it serves (:data:`LOAD_MODES`), first = default;
    ``load_arches``: a load mode limited to these card architectures (e.g. the
    in-place MXFP4 reload of SGLang only on Blackwell); ``rollout``: the
    :data:`ROLLOUT_FIELDS` it returns; ``card_arches``: the card architectures the
    model runs on in this image (empty: every architecture of the image);
    ``min_hbm_gib_per_node``: the smallest node (cards x HBM) it was sized for
    (None: no floor); ``verified_on``: the cards it was run on (empty: inferred from
    the source, not run)."""

    model: str
    precision: str
    loads: tuple[str, ...]
    rollout: tuple[str, ...] = ()
    min_hbm_gib_per_node: float | None = None
    verified_on: tuple[str, ...] = ()
    card_arches: tuple[str, ...] = ()
    load_arches: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    note: str = ""

    def problems(self, *, card_arch: str | None = None, load: str | None = None, precision: str | None = None,
                 rollout: Iterable[str] = (), hbm_gib_per_node: float | None = None) -> list[str]:
        """Why this recipe does not fit (empty: it does). An unknown ``card_arch``
        or ``hbm_gib_per_node`` (None) is not checked."""
        out: list[str] = []
        load = canonical_load(load)
        what = "any model" if self.model == ANY_MODEL else self.model
        if precision is not None and str(precision).lower() != self.precision:
            out.append(f"{what} is served as {self.precision}, not {precision}")
        if card_arch and self.card_arches and card_arch not in self.card_arches:
            out.append(f"{what} runs on {'/'.join(self.card_arches)} cards here, the node is {card_arch}")
        if load is not None and load not in self.loads:
            out.append(f"{what}: load {load} is not served (loads: {', '.join(self.loads)})")
        elif load is not None and card_arch and load in self.load_arches \
                and card_arch not in self.load_arches[load]:
            out.append(f"{what}: load {load} needs {'/'.join(self.load_arches[load])} cards, the node is {card_arch}")
        missing = [f for f in rollout if f not in self.rollout]
        if missing:
            out.append(f"{what}: rollout field(s) {missing} are not returned (returns: {list(self.rollout)})")
        if hbm_gib_per_node is not None and self.min_hbm_gib_per_node is not None \
                and hbm_gib_per_node < self.min_hbm_gib_per_node:
            out.append(f"{what} needs a node of at least {self.min_hbm_gib_per_node:g} GiB HBM, "
                       f"the node has {hbm_gib_per_node:g}")
        return out

    def loads_on(self, card_arch: str | None) -> tuple[str, ...]:
        """The load modes this recipe serves on a card of ``card_arch``."""
        return tuple(m for m in self.loads
                     if not (card_arch and m in self.load_arches and card_arch not in self.load_arches[m]))


def _strip_docker(reference: str) -> str:
    return reference.strip().removeprefix("docker:")


def _digest(reference: str) -> str | None:
    return reference.split("@", 1)[1] if "@" in reference else None


@dataclass(frozen=True)
class EngineImage:
    """A pinned sampler image and what it serves (unified-sampler-pool).

    ``pin``: ``"module:CONSTANT"`` of the reference (:attr:`ref` reads it, so the
    pins module stays the single source of the digest); ``family``: ``cuda`` /
    ``rocm`` / ``ascend`` / ``ppu``; ``card_arches``: the card architectures the
    build targets; ``engine``: :data:`ENGINES`; ``engine_version``: the build's
    version (vLLM ``GET /version``, SGLang commit); ``python``: the interpreter
    the sampler runs with inside it; ``recipes``: per model (:class:`Recipe`)."""

    name: str
    pin: str
    family: str
    cpu_arch: str
    card_arches: tuple[str, ...]
    engine: str
    engine_version: str
    python: str
    recipes: tuple[Recipe, ...]
    roles: tuple[str, ...] = ("sampler",)
    tag: str = ""  # the human tag the digest was pulled as (information only)

    @property
    def ref(self) -> str:
        """The pinned reference, exactly the pins constant (``docker:`` prefix kept)."""
        return resolve_ref(self.pin)

    @property
    def pull_ref(self) -> str:
        """The reference ``docker pull`` / ``podman pull`` take (no ``docker:`` prefix)."""
        return _strip_docker(self.ref)

    def matches(self, reference: str) -> bool:
        """``reference`` names this image: the same digest, or the same reference."""
        mine, theirs = self.pull_ref, _strip_docker(reference)
        if _digest(mine) and _digest(theirs):
            return _digest(mine) == _digest(theirs)
        return mine == theirs

    def recipe_for(self, model: str | None) -> Recipe | None:
        """The model's own recipe, else the :data:`ANY_MODEL` one, else None."""
        key = model_key(model)
        own = next((r for r in self.recipes if r.model == key), None)
        return own or next((r for r in self.recipes if r.model == ANY_MODEL), None)

    def runs_on(self, *, family: str, cpu_arch: str, card_arch: str | None = None) -> list[str]:
        """Why this image cannot start on such a node (empty: it can)."""
        out = []
        if _same_family(self.family, family) is False:
            out.append(f"it is a {self.family} image, the node is {family}")
        if self.cpu_arch != cpu_arch:
            out.append(f"it is a {self.cpu_arch} image, the node is {cpu_arch}")
        if card_arch and self.card_arches and card_arch not in self.card_arches:
            out.append(f"it targets {'/'.join(self.card_arches)} cards, the node is {card_arch}")
        return out

    def problems(self, *, family: str, cpu_arch: str, card_arch: str | None = None, model: str | None = None,
                 load: str | None = None, precision: str | None = None, rollout: Iterable[str] = (),
                 hbm_gib_per_node: float | None = None) -> list[str]:
        """Why this image does not serve that model on that node (empty: it does).
        Without ``model`` only the node is checked."""
        out = self.runs_on(family=family, cpu_arch=cpu_arch, card_arch=card_arch)
        if out or model is None:
            return out
        recipe = self.recipe_for(model)
        if recipe is None:
            return [f"no recipe for {model_key(model)} (recipes: {[r.model for r in self.recipes]})"]
        return recipe.problems(card_arch=card_arch, load=load, precision=precision, rollout=rollout,
                               hbm_gib_per_node=hbm_gib_per_node)

    def record(self) -> dict[str, Any]:
        """Manifest / event row: name, reference, engine and the node kind."""
        return {"image": self.name, "ref": self.ref, "family": self.family, "cpu_arch": self.cpu_arch,
                "card_arches": list(self.card_arches), "engine": self.engine, "engine_version": self.engine_version}


_PINS = "yeto.rl.sampler_pins"
_K31_ROLLOUT = ROLLOUT_FIELDS
# 8 x 288 GiB: the Kimi-K3.1-Pro MXFP4 sampler (1.56 TB of weights, TP8) was only sized and run on
# 288 GiB cards (MI355X, B300); a smaller node is not refused by the engine, it is refused here.
_K31_MIN_NODE_GIB = 8 * 288.0
_BLACKWELL = ("sm100", "sm103")


def _engine_images() -> tuple[EngineImage, ...]:
    from yeto.rl import sampler_pins as pins  # noqa: PLC0415 - stdlib-only constants, read on first use

    return (
        # The customer's K3.1 sampler (NC MI355X, Daytona MI355X): vllm/vllm-openai-rocm:kimi-k31.
        # In-place reload of the trained MXFP4 experts is the yeto worker extension
        # (vllm_expert_reload, backend AITER_MXFP4_BF16: gfx950 only).
        EngineImage(
            "vllm-rocm-k31", f"{_PINS}:VLLM_SAMPLER_IMAGE_ROCM", "rocm", "x86_64", ("gfx950",), "vllm",
            pins.VLLM_SAMPLER_ROCM_VERSION, "/opt/runtime-venv/bin/python3",
            (Recipe("kimi_k31", "mxfp4", ("in_place", "restart"), _K31_ROLLOUT, _K31_MIN_NODE_GIB, ("MI355X",),
                    note="in_place: vllm_expert_reload (AITER_MXFP4_BF16, gfx950)"),
             Recipe(ANY_MODEL, "bf16", ("restart", "in_place"), _K31_ROLLOUT,
                    note="in_place: vLLM reload_weights / the linear and tied reload extensions; not run here")),
            tag=pins.VLLM_SAMPLER_IMAGE_ROCM_TAG),
        # vllm/vllm-openai:kimi-k31 + the yeto overlay (B200 / B300). vLLM's reload_weights corrupts the
        # MXFP4 routed experts and the yeto expert reload is AITER-only: K3.1 restarts per release.
        EngineImage(
            "vllm-cuda-k31", f"{_PINS}:VLLM_SAMPLER_IMAGE_CUDA", "cuda", "x86_64",
            ("sm80", "sm86", "sm89", "sm90", "sm100", "sm103", "sm120"), "vllm",
            pins.VLLM_SAMPLER_CUDA_VERSION, "python3",
            (Recipe("kimi_k31", "mxfp4", ("restart",), _K31_ROLLOUT, _K31_MIN_NODE_GIB, (), card_arches=_BLACKWELL,
                    note="the recipe image targets B200 / B300; no in-place MXFP4 expert reload on CUDA"),
             Recipe(ANY_MODEL, "bf16", ("restart", "in_place"), _K31_ROLLOUT,
                    note="in_place: vLLM reload_weights; not run here")),
            tag=pins.VLLM_SAMPLER_BASE_IMAGE_CUDA_TAG),
        # lmsysorg/sglang nightly-dev 842b4738 + the yeto overlay. In-place MXFP4 reload
        # (sglang_mxfp4_reload, trtllm_sm100) only on Blackwell: SGLang turns it off before SM100
        # (sglang_server.IN_PLACE_RELOAD_MIN_CAPABILITY).
        EngineImage(
            "sglang-cuda-k31", f"{_PINS}:SGLANG_SAMPLER_IMAGE_CUDA", "cuda", "x86_64",
            ("sm80", "sm86", "sm89", "sm90", "sm100", "sm103", "sm120"), "sglang",
            pins.SGLANG_SAMPLER_ENGINE_VERSIONS["cuda"], "/opt/sglang/bin/python3",
            (Recipe("kimi_k31", "mxfp4", ("in_place", "restart"), _K31_ROLLOUT, _K31_MIN_NODE_GIB, (),
                    card_arches=("sm90",) + _BLACKWELL, load_arches={"in_place": _BLACKWELL},
                    note="in_place: sglang_mxfp4_reload (Blackwell only)"),
             Recipe(ANY_MODEL, "bf16", ("in_place", "live", "restart"), _K31_ROLLOUT)),
            tag=f"sampler-k31-{pins.SGLANG_SAMPLER_UPSTREAM_COMMIT[:7]}-{pins.SGLANG_SAMPLER_YETO_COMMIT[:7]}"),
        # lmsysorg/sglang:dev-kimi-k31-pro-mi35x (upstream PR 48427).
        EngineImage(
            "sglang-rocm-k31", f"{_PINS}:SGLANG_SAMPLER_IMAGE_ROCM", "rocm", "x86_64", ("gfx950",), "sglang",
            pins.SGLANG_SAMPLER_ENGINE_VERSIONS["rocm"], "python3",
            (Recipe("kimi_k31", "mxfp4", ("restart", "in_place"), _K31_ROLLOUT, _K31_MIN_NODE_GIB, (),
                    note="in_place: SGLang expert-file update; not run on MI355X by this catalog"),
             Recipe(ANY_MODEL, "bf16", ("in_place", "live", "restart"), _K31_ROLLOUT)),
            tag=pins.SGLANG_SAMPLER_IMAGE_ROCM_TAG),
    )


_CATALOG: tuple[EngineImage, ...] | None = None


def engine_images() -> tuple[EngineImage, ...]:
    """Every pinned sampler image, in preference order (built on first use)."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _engine_images()
    return _CATALOG


def engine_image(name: str) -> EngineImage:
    try:
        return next(i for i in engine_images() if i.name == name)
    except StopIteration:
        raise KeyError(f"no engine image {name!r} (known: {[i.name for i in engine_images()]})") from None


def engine_image_of(reference: str) -> EngineImage | None:
    """The catalog image ``reference`` names (by digest), or None for a site image."""
    if not reference or not str(reference).strip():
        return None
    return next((i for i in engine_images() if i.matches(str(reference))), None)


@dataclass(frozen=True)
class Resolution:
    """The image :func:`resolve` chose for one node.

    ``image``: the catalog record (None: a site image the caller named, which yeto
    cannot check); ``ref``: what to pull and run; ``recipe``: the model's recipe in
    that image (None with a site image); ``load``: the reload mode to use (the one
    asked for, else the recipe's first on this card); ``python``: the interpreter
    (None: the caller's own)."""

    ref: str
    image: EngineImage | None
    recipe: Recipe | None
    load: str | None
    python: str | None

    @property
    def engine(self) -> str | None:
        return self.image.engine if self.image else None

    def record(self) -> dict[str, Any]:
        row = self.image.record() if self.image else {"image": None, "ref": self.ref}
        return {**row, "load": self.load, "precision": self.recipe.precision if self.recipe else None,
                "verified_on": list(self.recipe.verified_on) if self.recipe else []}


def _engine_order(engine_pref: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if engine_pref is None:
        return None
    prefs = (engine_pref,) if isinstance(engine_pref, str) else tuple(engine_pref)
    out = []
    for e in prefs:
        e = str(e).strip().lower()
        e = {"vllm-server": "vllm", "sglang-server": "sglang"}.get(e, e)
        if e not in ENGINES:
            raise ValueError(f"engine {e!r} is unknown (known: {list(ENGINES)})")
        out.append(e)
    return tuple(out)


def _profile_facts(profile: Any) -> dict[str, Any]:
    """family / cpu_arch / card_arch / HBM of a node profile (a NodeProfile, a plain
    Node, or any object with those attributes)."""
    family = getattr(profile, "family", None) or getattr(profile, "device_family")
    hbm = None
    if getattr(profile, "hbm_gib", None) and getattr(profile, "cards_per_node", None):
        hbm = float(profile.hbm_gib) * int(profile.cards_per_node)
    return {"family": unified_family(family), "cpu_arch": canonical_cpu_arch(profile.cpu_arch),
            "card_arch": getattr(profile, "card_arch", None), "hbm_gib_per_node": hbm}


def resolve(profile: Any, model: str | None, engine_pref: str | Sequence[str] | None = None, *,
            load: str | None = None, precision: str | None = None, rollout: Iterable[str] = (),
            image: str | None = None) -> Resolution:
    """The engine image for one node (unified-sampler-pool).

    ``profile``: a :class:`~yeto.launch.node_inventory.NodeProfile` (a plain
    ``Node`` works: no card architecture is then checked); ``model``: a model name
    (:func:`model_key`); ``engine_pref``: None (catalog order), an engine
    (``vllm`` / ``sglang`` / ``vllm-ascend``) or engines in order; ``load``: the
    reload mode the pool needs (:data:`LOAD_MODES`; None: the recipe's default);
    ``precision`` and ``rollout``: what the sampler spec asks for; ``image``: a
    reference the pool or the inventory names (else the profile's ``image``): a
    catalog image is checked like any other and is the only candidate (an
    ``engine_pref`` it does not match is refused, never ignored); a site image is
    returned unchecked.

    Raises :class:`ImageIncompatible` (with every image's reason and what would work
    instead) when nothing fits; the caller refuses before ordering anything."""
    facts = _profile_facts(profile)
    rollout = tuple(rollout)
    load = canonical_load(load)
    named = image or getattr(profile, "image", None)
    if named:
        chosen = engine_image_of(named)
        if chosen is None:
            return Resolution(ref=named, image=None, recipe=None, load=load, python=None)
        candidates: tuple[EngineImage, ...] = (chosen,)
    else:
        candidates = engine_images()
    order = _engine_order(engine_pref)
    if order is not None and named and candidates[0].engine not in order:
        chosen = candidates[0]
        raise ImageIncompatible(
            f"image {named} is the {chosen.engine} image {chosen.name}, the engine asked for is {'/'.join(order)}",
            reasons={chosen.name: [f"engine {chosen.engine}, not {'/'.join(order)}"]})
    if order is not None and not named:
        candidates = tuple(i for e in order for i in candidates if i.engine == e)
    reasons: dict[str, list[str]] = {}
    for candidate in candidates:
        why = candidate.problems(model=model or ANY_MODEL, load=load, precision=precision, rollout=rollout,
                                 **facts)
        if why:
            reasons[candidate.name] = why
            continue
        recipe = candidate.recipe_for(model)
        assert recipe is not None
        chosen_load = load or recipe.loads_on(facts["card_arch"])[0]
        return Resolution(ref=candidate.ref, image=candidate, recipe=recipe, load=chosen_load,
                          python=candidate.python)
    node = f"{facts['family']}/{facts['card_arch'] or '?'}/{facts['cpu_arch']}"
    want = f"{model_key(model)}" + (f" {precision}" if precision else "") + (f" load {load}" if load else "")
    if order:
        want += f" on {'/'.join(order)}"
    if not candidates:
        reasons["(none)"] = [f"no {'/'.join(order or ())} image in the catalog"]
    alternatives = _alternatives(facts, model, load=load, precision=precision, rollout=rollout)
    detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in reasons.items())
    hint = (" Works instead: " + "; ".join(alternatives) + ".") if alternatives else ""
    raise ImageIncompatible(f"no engine image serves {want} on a {node} node ({detail}).{hint}",
                            reasons=reasons, alternatives=alternatives)


def _alternatives(facts: Mapping[str, Any], model: str | None, *, load: str | None, precision: str | None,
                  rollout: tuple[str, ...]) -> list[str]:
    """What would work instead: another load mode on this node, or this load on
    another kind of node."""
    out = []
    for image in engine_images():
        if image.runs_on(family=facts["family"], cpu_arch=facts["cpu_arch"], card_arch=facts["card_arch"]):
            continue
        recipe = image.recipe_for(model)
        if recipe is None or recipe.problems(card_arch=facts["card_arch"], precision=precision, rollout=rollout,
                                             hbm_gib_per_node=facts["hbm_gib_per_node"]):
            continue
        loads = recipe.loads_on(facts["card_arch"])
        if loads:
            out.append(f"{image.engine} ({image.name}) on this node with load {'/'.join(loads)}")
    for image in engine_images():
        recipe = image.recipe_for(model)
        if recipe is None or image.runs_on(family=facts["family"], cpu_arch=facts["cpu_arch"],
                                            card_arch=facts["card_arch"]) == []:
            continue
        if recipe.problems(load=load, precision=precision, rollout=rollout):
            continue
        arches = recipe.load_arches.get(load or "", ()) or recipe.card_arches or image.card_arches
        out.append(f"{image.engine} ({image.name}) on a {image.family} {'/'.join(arches)} node")
    return list(dict.fromkeys(out))


def resolve_pool(profiles: Iterable[Any], model: str | None, engine_pref: str | Sequence[str] | None = None, *,
                 load: str | None = None, precision: str | None = None, rollout: Iterable[str] = (),
                 image: str | None = None) -> dict[str, Resolution]:
    """:func:`resolve` for every node of a (possibly heterogeneous) pool: node name
    -> its image. All nodes share one ``rollout`` / ``load`` / ``precision`` (the
    pool's one sampler spec); ``image`` (the pool's ``container.image``) is checked
    on every node, a node's own inventory image wins over the catalog. Raises
    :class:`ImageIncompatible` naming every node that does not fit."""
    out: dict[str, Resolution] = {}
    failures: dict[str, list[str]] = {}
    alternatives: list[str] = []
    for profile in profiles:
        try:
            out[profile.name] = resolve(profile, model, engine_pref, load=load, precision=precision,
                                        rollout=rollout, image=image or getattr(profile, "image", None))
        except ImageIncompatible as error:
            failures[profile.name] = [str(error)]
            alternatives += error.alternatives
    if failures:
        raise ImageIncompatible("; ".join(f"node {k}: {v[0]}" for k, v in failures.items()), reasons=failures,
                                alternatives=list(dict.fromkeys(alternatives)))
    return out


def spec_needs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """What a sampler spec asks of its image, as :func:`resolve` keywords plus
    ``model`` and ``engine_pref``: the engine (``engine.adapter``), the reload mode
    (``engine.server.in_place_reload``: ``in_place``, else ``restart``), the precision
    (``engine.server.precision``), the rollout fields (``rollout_fields``) and the
    model (``model`` / ``engine.server.model``; a ``k31_preset`` or a K3.1 engine
    version is ``kimi_k31``; None: unknown, any-model recipes apply)."""
    engine = spec.get("engine") or {}
    server = engine.get("server") or {}
    options = engine.get("options") or {}
    adapter = str(engine.get("adapter") or "")
    platform = str(spec.get("platform") or options.get("platform") or "")
    pref = {"vllm-server": "vllm-ascend" if platform == "npu" else "vllm", "sglang-server": "sglang"}.get(adapter)
    fields = spec.get("rollout_fields") or {}
    rollout = []
    if fields.get("routed_experts"):
        rollout.append("routed_experts")
    if int(fields.get("top_logprobs_num") or 0) > 0:
        rollout.append("top_logprobs")
    if fields.get("fp32_lm_head"):
        rollout.append("fp32_lm_head")
    model = spec.get("model") or server.get("model")
    if model is None and (options.get("k31_preset") or "k31" in str(server.get("engine_version") or "")):
        model = "kimi_k31"
    return {"model": model, "engine_pref": pref, "load": "in_place" if server.get("in_place_reload") else "restart",
            "precision": server.get("precision"), "rollout": tuple(rollout)}
