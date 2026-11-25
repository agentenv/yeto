"""A trainer image against the cards of its island, checked before anything is ordered.

Some trainer images only run on cards from a given architecture on: the
``yeto-miles-k31`` images (every Miles image built by ``docker/miles-k31``,
including the default ``--rl-image``) were verified on H100 / H200 / B300 only. On
an A100 (sm80) the container, Ray and the engine all start, and the first forward
step then fails in Megatron's ``_forward_attention`` with Transformer Engine's
"No dot product attention backend is available for the provided inputs" (fused
attention): twelve minutes and the pool's spend later (P5b, 2026-11-24). The launch
preflight and the dry-run refuse that pairing instead.

A floor names the image repository, the lowest card architecture it was verified on,
and the training attention backends that lift the refusal to a warning. The backend is
the one the trainer actually runs (:func:`training_attention`): the Miles trainer on
Transformer Engine's ``fused`` attention, or on ``auto`` (Megatron lets TE pick), is
refused below the floor; ``flash`` and ``unfused`` (the generic recipe under
``--rl-deterministic-trainer``, or an explicit ``--rl-training-attention-backend``)
are only warned about, since they still have to be verified on that card. Other
training backends (verl, ...) run their own images and are not checked, nor is an image
outside this table (a site image, a registry image yeto does not know).

A backend that builds its own trainer image from a pinned engine commit (verl:
``verl-build:<commit>``) checks the image's commit against its pin
(:func:`image_pin_problems`, the backend's ``launch_flags.image_pin_problem``): a trainer
island on another commit is refused and named.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from .node_inventory import card_arch_of


@dataclass(frozen=True)
class ImageFloor:
    repository: str
    min_arch: str
    verified_on: tuple[str, ...]
    failure: str
    lifted_by: tuple[str, ...] = ()


FLOORS: tuple[ImageFloor, ...] = (
    ImageFloor("yeto-miles-k31", "sm90", ("H100", "H200", "B300"),
               'Transformer Engine has no fused attention for it there ("No dot product attention backend is '
               'available"), so the trainer fails at its first step',
               lifted_by=("flash", "unfused")),
)


def _repository(image: str) -> str:
    path = str(image or "").removeprefix("docker:").split("@", 1)[0]
    return path.rsplit("/", 1)[-1].split(":", 1)[0]


def floor_of(image: str) -> ImageFloor | None:
    repo = _repository(image)
    return next((f for f in FLOORS if f.repository == repo), None)


def _sm(arch: str | None) -> int | None:
    """``sm90`` -> 90; None for a non-CUDA or unknown architecture."""
    if not arch or not arch.startswith("sm") or not arch[2:].isdigit():
        return None
    return int(arch[2:])


def image_problem(image: str, gpu: str, *, attention_backend: str | None = "auto") -> tuple[str, bool] | None:
    """``(message, refused)`` when ``image`` is below its floor on ``gpu`` cards (refused
    False: an attention backend that lifts the floor, a warning), else None."""
    floor = floor_of(image)
    if floor is None:
        return None
    try:
        arch = card_arch_of(gpu)
    except ValueError:  # a card outside the catalog: the card checks report it, not this one
        return None
    have, need = _sm(arch), _sm(floor.min_arch)
    if have is None or need is None:
        return None
    if have >= need:
        verified = {card_arch_of(card) for card in floor.verified_on}
        if arch in verified:
            return None
        return (f"trainer image {image} ({floor.repository}) was verified on {'/'.join(floor.verified_on)} "
                f"only; {gpu} ({arch}) is above its {floor.min_arch} floor but was never run: make sure the "
                "trainer runs on this card", False)
    lifted = (attention_backend or "auto") in floor.lifted_by
    message = (f"trainer image {image} ({floor.repository}) was verified on {'/'.join(floor.verified_on)} "
               f"({floor.min_arch}+) only; on {gpu} ({arch}) {floor.failure}")
    if lifted:
        return (message + f"; the {attention_backend} attention backend takes another path: "
                "make sure it was run on this card", False)
    return (message + f". Use {floor.min_arch}+ cards (" + ", ".join(floor.verified_on) + ") for the trainer, or "
            f"pass --rl-training-attention-backend {' / '.join(floor.lifted_by)} after verifying it on {gpu}", True)


# --- overlay-safe pulls on sky VMs ---------------------------------------------------
#
# A pinned trainer image can carry a layer that whites out a path and writes it again
# (the miles pins' OVERLAY_SAFE_DIGESTS / OVERLAY_UNSAFE_UNREPACKED). Modal's builder
# unpacks it; docker on a VM does not (overlay2 and the containerd overlayfs snapshotter:
# "failed to register layer ... not a directory" on `docker pull`). A trainer island on a
# sky VM therefore pulls the image's overlay-safe repack (the same files and Miles
# commit, so the pins still hold), and an unsafe image without a repack is refused before
# anything is ordered. An ssh: island picks the repack itself
# (yeto.launch.ssh_trainer_island.run_image); a Modal island keeps the pin.
NO_SKY_DOCKER_CLOUDS = frozenset({"modal", "ssh"})


def _miles_pins() -> Any:
    from ..rl.engine import backends

    return backends.module("pins", "miles")


def sky_pull_image(image: Any) -> Any:
    """The image a sky VM's docker pulls for the trainer image ``image``: its overlay-safe
    repack when the miles pins name one, else ``image`` itself (also any non-string)."""
    if not isinstance(image, str) or "@sha256:" not in image:
        return image
    try:
        return _miles_pins().overlay_safe_image(image) or image
    except Exception:  # noqa: BLE001 - no pins, no swap: the image as given
        return image


def sky_pull_problem(image: str, cloud: str) -> tuple[str, bool] | None:
    """``(message, refused)`` for a trainer image on a sky VM of ``cloud``: the repack it
    pulls instead (refused False: a note), or an overlay-unsafe image with no repack
    (refused True); None when the image pulls as it is or the island is not a sky VM."""
    if str(cloud or "").lower() in NO_SKY_DOCKER_CLOUDS or not image:
        return None
    pulled = sky_pull_image(image)
    if pulled != image:
        return (f"docker on the {cloud} VM pulls {pulled}, the overlay-safe repack of {image} (the same files "
                "and Miles commit; the image's layer that whites out a path and writes it again does not "
                "unpack on overlay2 / containerd overlayfs)", False)
    message = vm_pull_refusal(image, f"docker on a {cloud} VM")
    return (message, True) if message else None


def vm_pull_refusal(image: Any, where: str) -> str | None:
    """Why ``where`` (a VM's docker: a sky island, a sky pool's node) cannot pull ``image``:
    an image an overlay applier refuses that no swap covers (the miles pins'
    OVERLAY_UNSAFE_UNREPACKED, or an OVERLAY_SAFE_DIGESTS key named where nothing swaps it),
    with the overlay-safe digest to pass instead; None for any other image."""
    if not isinstance(image, str) or "@sha256:" not in image:
        return None
    try:
        pins = _miles_pins()
        if not pins.overlay_unsafe_image(image):
            return None
        alternative = pins.overlay_safe_alternative(image)
    except Exception:  # noqa: BLE001 - no pins: nothing known about the image
        return None
    use = (f"use {alternative[0]} instead ({alternative[1]})" if alternative else
           "pass an image built by docker/miles-k31/build.sh (its layer is overlay-safe), or repack that layer "
           "with docker/miles-ports/make_layer.py repack-whiteouts")
    return (f"image {image} has a layer that whites out a path and writes it again: {where} (overlay2 / "
            "containerd overlayfs) refuses it ('failed to register layer ... not a directory' on pull) and "
            f"the image has no overlay-safe repack: {use}, or run it on Modal")


def training_attention(args: Any) -> tuple[str, str]:
    """``(backend, why)``: the Megatron ``--attention-backend`` the Miles trainer resolves
    to from the launch arguments, as :func:`yeto.rl.engine.run_config.resolve_rl_run_config`
    picks it: ``--rl-training-attention-backend`` when set; else the recipe's own default
    (an explicit ``--rl-model-recipe``, the Qwen3.8-Flash-Next or pinned Qwen3.5 model,
    else the generic recipe: ``fused``, ``unfused`` under ``--rl-deterministic-trainer``).
    A recipe only the model's config selects (a gated-delta-net or Nemotron-H checkpoint
    under another name) reads as the generic one here; its flag says otherwise."""
    from ..rl.engine import run_config as rc

    override = getattr(args, "rl_training_attention_backend", "auto") or "auto"
    if override != "auto":
        return override, "--rl-training-attention-backend"
    recipe = getattr(args, "rl_model_recipe", None) or rc.RECIPE_GENERIC
    if recipe == rc.RECIPE_DEEPSEEK_V4_FLASH:
        return "flash", f"the {recipe} recipe"
    if recipe in (rc.RECIPE_GLM5_NEXT, rc.RECIPE_KIMI_K31, rc.RECIPE_DEEPSEEK_V41):
        return "auto", f"the {recipe} recipe (Megatron picks)"
    if rc.qwen3_8_next_variant(args, None) is not None:
        return "flash", f"the {rc.RECIPE_QWEN3_8_NEXT} recipe"
    from ..rl.codex_backend import QWEN35_MODEL, QWEN35_REVISION

    if getattr(args, "model", None) == QWEN35_MODEL and getattr(args, "model_revision", None) == QWEN35_REVISION:
        return "flash", f"the {rc.RECIPE_QWEN3_5} recipe"
    backend = rc.generic_attention_backend(args)
    return backend, ("the generic recipe under --rl-deterministic-trainer" if backend == "unfused"
                     else "the generic recipe")


def image_pin_problems(args: Any, num_islands: int | None = None) -> list[str]:
    """Each trainer island whose image (``--rl-image``, or its own ``--rl-island-engine``
    image) is not built from its backend's pinned engine commit, named by its ``--gpu``
    island (the backend's ``launch_flags.image_pin_problem``; a backend without one is not
    checked). Empty for an SFT launch."""
    if getattr(args, "training_mode", "sft") != "rl":
        return []
    from ..gpu_spec import parse_gpu_spec
    from ..rl.engine import backends, island_engine, sample_roles

    try:
        specs = parse_gpu_spec(args.gpu)
        count = len(specs) if num_islands is None else int(num_islands)
        samplers = set(sample_roles.sampler_islands(args, count))
    except (TypeError, ValueError, AttributeError):  # the launch's own checks name a bad --gpu
        return []
    found = []
    for m in range(count):
        if m in samplers:
            continue
        try:
            iargs = island_engine.island_args(args, m, count)
            check = getattr(backends.module("launch_flags", getattr(iargs, "rl_backend", None) or "miles"),
                            "image_pin_problem", None)
        except (ValueError, backends.UnknownBackend):  # island_engine.check / the backend check name it
            continue
        problem = check(str(getattr(iargs, "rl_image", None) or "")) if check else None
        if problem:
            found.append(f"--gpu island {m} ({specs[m] if m < len(specs) else '?'}): {problem}")
    return found


def check_vm_pulls(args: Any, specs: Any, *, out: Any = None) -> None:
    """Every island on a sky VM, trainer and sampler islands alike, against the image its
    VM's docker pulls (the island's own image: ``--rl-island-engine N:image=`` or the
    run's ``--rl-image``): the overlay-safe swap is printed, an overlay-unsafe image with
    no repack is refused (ValueError) naming the digest to use instead."""
    from ..rl.engine import island_engine

    count = len(specs)
    for m, spec in enumerate(specs):
        try:
            image = getattr(island_engine.island_args(args, m, count), "rl_image", None)
        except (ValueError, AttributeError, TypeError):  # island_engine.check names a bad island spec
            image = getattr(args, "rl_image", None)
        pull = sky_pull_problem(str(image or ""), str(getattr(spec, "cloud", "") or ""))
        if pull is None:
            continue
        if pull[1]:
            raise ValueError(f"--gpu island {m} ({spec}): {pull[0]}")
        print(f"[launcher] --gpu island {m} ({spec}): {pull[0]}", file=out or sys.stderr, flush=True)


def check_trainer_images(args: Any, specs: Any, *, out: Any = None) -> list[str]:
    """Refuse (ValueError) a trainer island whose image cannot run on its cards; returns
    the warnings it printed. Only the Miles backend's trainer islands are checked against
    the card floors (the sampler islands of a group run their own engine image); every
    backend's against its engine pin (:func:`image_pin_problems`)."""
    if getattr(args, "training_mode", "sft") != "rl":
        return []
    pins = image_pin_problems(args, len(specs))
    if pins:
        raise ValueError("; ".join(pins))
    check_vm_pulls(args, specs, out=out)
    if (getattr(args, "rl_backend", None) or "miles") == "verl":
        return check_verl_trainer_cards(args, specs, out=out)
    if (getattr(args, "rl_backend", None) or "miles") != "miles":
        return []
    image = str(getattr(args, "rl_image", None) or "")
    if not image:
        return []
    from ..rl.engine import sample_roles

    samplers = set(sample_roles.sampler_islands(args, len(specs)))
    backend, why = training_attention(args)
    warnings = []
    for m, spec in enumerate(specs):
        if m in samplers:
            continue
        found = image_problem(image, str(spec.gpu), attention_backend=backend)
        if found is None:
            continue
        message, refused = found
        message += f" (training attention: {backend}, from {why})"
        if refused and why == "the generic recipe":
            message += ("; a recipe only the model's config selects (a renamed gated-delta-net or Nemotron-H "
                        "checkpoint) reads as the generic one here: set --rl-training-attention-backend to the "
                        "backend it runs")
        if refused:
            raise ValueError(f"--gpu island {m} ({spec}): {message}")
        warnings.append(f"--gpu island {m} ({spec}): {message}")
        print(f"[launcher] WARNING: {warnings[-1]}", file=out or sys.stderr, flush=True)
    return warnings


# --- verl FSDP2 trainer cards (fleet-verl-trainer-islands 5.2) ------------------
#
# The verl FSDP2 images (``verl-build`` and its registry image) carry no Transformer
# Engine: the trainer's attention is torch SDPA / flash-attn, built for sm80 and up.
# The allowed architectures and the cards with evidence live in the verl pins
# (VERL_FSDP_CARD_ARCHES, VERL_FSDP_VERIFIED_CARDS), next to the image they describe.

def verl_card_problem(gpu: str) -> tuple[str, bool] | None:
    """``(message, refused)`` for a verl FSDP2 trainer island on ``gpu`` cards: refused
    below the lowest allowed architecture (no flash-attn / bf16 kernels there); an
    "unverified" warning for a card no run has trained on (an allowed architecture
    without evidence, or one outside the list at or above its floor); None for a card
    with evidence, or one outside the CUDA catalog (Ascend, a card the catalog does not
    know: the card checks report it)."""
    from ..rl.engine import backends

    pins = backends.module("pins", "verl")
    try:
        arch = card_arch_of(gpu)
    except ValueError:
        return None
    have = _sm(arch)
    if have is None:
        return None
    allowed = tuple(pins.VERL_FSDP_CARD_ARCHES)
    floor = min(_sm(a) for a in allowed)
    verified = {str(card).upper(): why for card, why in pins.VERL_FSDP_VERIFIED_CARDS.items()}
    if have < floor:
        return (f"the verl FSDP2 trainer needs a card of {'/'.join(allowed)}; {gpu} ({arch}) is below "
                f"sm{floor} (no flash-attn or bf16 kernels): use one of "
                f"{', '.join(sorted(verified)) or 'the allowed architectures'}", True)
    if gpu.upper() in verified:
        return None
    where = "an allowed architecture" if arch in allowed else f"outside the verified list {'/'.join(allowed)}"
    return (f"UNVERIFIED: no verl FSDP2 trainer run on {gpu} ({arch}, {where}) is recorded in the verl pins "
            f"(verified: {', '.join(sorted(verified))}); it is not refused, but watch the first steps", False)


def check_verl_trainer_cards(args: Any, specs: Any, *, out: Any = None) -> list[str]:
    """The verl backend's trainer islands against their cards: a card below the FSDP2
    floor is refused (ValueError), an unverified card is warned about (returned and
    printed). The Megatron engine island and the sampler islands are not checked here."""
    from ..rl.engine import backends, sample_roles

    if backends.module("run_config_rules", "verl").megatron_run(args):
        return []
    samplers = set(sample_roles.sampler_islands(args, len(specs)))
    warnings = []
    for m, spec in enumerate(specs):
        if m in samplers:
            continue
        found = verl_card_problem(str(spec.gpu))
        if found is None:
            continue
        message, refused = found
        if refused:
            raise ValueError(f"--gpu island {m} ({spec}): {message}")
        warnings.append(f"--gpu island {m} ({spec}): {message}")
        print(f"[launcher] WARNING: {warnings[-1]}", file=out or sys.stderr, flush=True)
    return warnings + check_pulled_images(args, specs, samplers, out=out)


def _pulled_image(args: Any, cloud: str | None) -> str | None:
    """The registry image a verl trainer island on ``cloud`` pulls: the one that replaces
    ``verl-build`` there, or an explicit digest-pinned ``--rl-image``; None on Modal / ssh:."""
    import re

    from .rl_backend import _NO_PULL_CLOUDS, pulled_rl_image

    pulled = pulled_rl_image(args, cloud)
    if pulled is not None and pulled.get("problem"):
        raise ValueError(pulled["problem"])
    if pulled is not None:
        return pulled["image"]
    image = str(getattr(args, "rl_image", None) or "")
    if cloud is not None and cloud not in _NO_PULL_CLOUDS and re.fullmatch(
            r"docker:[^\s@]+@sha256:[0-9a-fA-F]{64}", image):
        return image
    return None


def check_pulled_images(args: Any, specs: Any, samplers: Any = (), *, out: Any = None) -> list[str]:
    """A verl trainer island that pulls a registry image (a sky island): the image's
    manifest is asked for before anything is ordered (``image_manifest``, a HEAD with the
    login the island would pull with). Missing or unreadable: refused (ValueError);
    registry not reached: a warning. ``YETO_IMAGE_MANIFEST_CHECK=off`` skips it."""
    import copy

    from ..rl.engine import backends
    from . import image_manifest as im
    from .credentials import registry_login_for

    images = {}
    for m, spec in enumerate(specs):
        if m in samplers:
            continue
        try:  # a configuration problem (no registry named) is refused with the check on or off
            images[m] = _pulled_image(args, getattr(spec, "cloud", None))
        except ValueError as error:
            raise ValueError(f"--gpu island {m} ({spec}): {error}") from None
    if not im.enabled():
        return []
    found: dict[str, tuple[str, str]] = {}
    warnings = []
    for m, spec in enumerate(specs):
        image = images.get(m)
        if image is None:
            continue
        if image not in found:
            iargs = copy.copy(args)
            iargs.rl_image = image
            found[image] = im.manifest_status(image, registry_login_for(iargs))
        status, detail = found[image]
        if status in (im.MISSING, im.DENIED):
            raise ValueError(
                f"--gpu island {m} ({spec}): trainer image {image}: {detail}. Push or mirror that digest there "
                f"(docker/verl-fsdp/build.sh), point {backends.module('pins', 'verl').VERL_REGISTRY_ENV} at a registry "
                "that has it, or give the "
                "launch a registry login (SKYPILOT_DOCKER_USERNAME / SKYPILOT_DOCKER_PASSWORD / "
                "SKYPILOT_DOCKER_SERVER, or --rl-image-private)")
        if status == im.UNKNOWN:
            warnings.append(f"--gpu island {m} ({spec}): trainer image {image} not checked: {detail}")
            print(f"[launcher] WARNING: {warnings[-1]}", file=out or sys.stderr, flush=True)
    return warnings
