"""Card-type catalog and island compatibility group
(yeto-framework-decoupling).

``compat_group`` is ``"<vendor>-<card>"`` in lower case, e.g. ``nvidia-h100``.
With the HBM size it is ``"<vendor>-<card>-<N>g"``, e.g.
``ascend-910b4-64g``: the 910B4 ships with 32 GiB and with 64 GiB (a cluster
can change from one to the other), and the two must not
merge. The vendor names the device family one to one
(:data:`DEVICE_FAMILY_BY_VENDOR`; a PPU is ``thead``), so the group always
carries the device family.
Phase 1 rule: two islands may merge only when their groups are identical
(H100 and H200 are different groups).  The value comes from this catalog,
never from a cloud name; an unknown card is an error before launch (no
default).  Task 7.2 later merges the other card tables into this module.

Driver and CUDA versions are recorded (:func:`runtime_versions`) but are not
part of the group and never cause a refusal (user decision 2026-10-09).
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from typing import Mapping

from .. import gpu_spec

# Canonical card name (``yeto.gpu_spec`` / SkyPilot spelling) -> vendor.
CARD_VENDOR: Mapping[str, str] = {
    "A100": "nvidia", "A100-80GB": "nvidia", "A10G": "nvidia", "L4": "nvidia",
    "L40S": "nvidia", "RTX-6000-Ada": "nvidia", "RTX-PRO-6000": "nvidia",
    "H100": "nvidia", "H200": "nvidia", "B200": "nvidia", "V100": "nvidia",
    "T4": "nvidia",
    # Ascend NPU. 910B4 is the model the user bought (decision 2026-10-10);
    # "910B" stays as the family-level name earlier code already used.
    "910B": "ascend", "910B4": "ascend", "910B2": "ascend", "910B3": "ascend",
    "910C": "ascend",
    # Alibaba T-Head PPU (rl-ssh-node-pool). torch reports device type
    # "cuda" on these cards, so the vendor comes from the card name only.
    "810E": "thead", "M890": "thead",
    # NVIDIA Blackwell Ultra (Modal / National Compute B300 nodes).
    "B300": "nvidia",
    # AMD Instinct (ROCm; the Kimi K3.1 sampler nodes). torch reports device
    # type "cuda" on ROCm too, so the vendor comes from the card name only.
    "MI300X": "amd", "MI325X": "amd", "MI355X": "amd",
}

# Nominal HBM sizes (GiB) a card ships with. Read from our machines: 910B4
# 64 GiB (node inventory) and 32 GiB (ModelArts), M890 144 GiB
# (ppu-smi 147456 MiB). The others are public
# product figures, not read on our nodes. 810E: not known, so any reported
# size is used as is (rounded).
CARD_HBM_GIB: Mapping[str, tuple[int, ...]] = {
    "910B": (32, 64), "910B4": (32, 64), "910B2": (64,), "910B3": (64,), "910C": (128,),
    "810E": (), "M890": (144,),
    "H100": (80,), "H200": (141,), "A100": (40, 80), "A100-80GB": (80,), "B200": (180,),
    "B300": (288,), "MI300X": (192,), "MI325X": (256,), "MI355X": (288,),
    # NVIDIA product figures (the single-card sky offers; the node preflight checks each
    # reported total against them): L40S 48 GB, L4 24 GB, T4 16 GB, A10G 24 GB, V100 16 / 32 GB,
    # RTX 6000 Ada 48 GB, RTX PRO 6000 Blackwell 96 GB. nvidia-smi reports them a few percent
    # lower (an L40S 46068 MiB), within card_tools.HBM_TOLERANCE.
    "L40S": (48,), "L4": (24,), "T4": (16,), "A10G": (24,), "V100": (16, 32),
    "RTX-6000-Ada": (48,), "RTX-PRO-6000": (96,),
}
# The cards whose sizes above came in on 2026-11-25. Before, their compat group took the
# declared size rounded; an inventory that declares a size fitting none of the nominal ones
# keeps that label (it loaded before, and its node preflight still checks the cards) instead
# of being refused at load.
ROUNDED_HBM_CARDS = frozenset({"L40S", "L4", "T4", "A10G", "V100", "RTX-6000-Ada", "RTX-PRO-6000"})

# The tables below are derived from yeto.gpu_spec.HW_FAMILIES, the single
# hardware-family table; the values are the ones this module already had.

# torch device type -> device family in the backend identity. ROCm and PPU are
# absent on purpose: their torch device type is "cuda", so only the card name
# or the launcher's $YETO_DEVICE_FAMILY tells them apart from an NVIDIA GPU
# (see VENDOR_BY_DEVICE_FAMILY and IDENTITY_FROM_ENV_FAMILY).
DEVICE_FAMILY_BY_DEVICE_TYPE: Mapping[str, str] = {"cuda": "nvidia", "npu": "ascend"}

# Launcher device family (yeto.gpu_spec.DEVICE_FAMILIES) -> vendor.
VENDOR_BY_DEVICE_FAMILY: Mapping[str, str] = {
    f.family: f.vendor for f in gpu_spec.HW_FAMILIES if f.family != "cpu"}
DEVICE_FAMILY_BY_VENDOR: Mapping[str, str] = {v: k for k, v in VENDOR_BY_DEVICE_FAMILY.items()}

# Opt-in (default off): read the identity device family from the launcher's
# $YETO_DEVICE_FAMILY instead of the torch device type, so an AMD or PPU island
# records vendor "amd"/"thead" instead of "nvidia" (gap T9). This value feeds
# the backend identity, whose hash is part of the async GRPO delivery, so it
# stays behind a switch: with the switch off every existing island -- NVIDIA
# B300 trainer, MI355X sampler, CPU test runner -- reports exactly what it
# reported before.
IDENTITY_FROM_ENV_FAMILY = "YETO_IDENTITY_FAMILY_FROM_ENV"


def identity_family_from_env_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether $YETO_IDENTITY_FAMILY_FROM_ENV turns the T9 fix on (default: no)."""
    raw = (env if env is not None else os.environ).get(IDENTITY_FROM_ENV_FAMILY, "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def device_family(device_type: str | None = None) -> str:
    """Device family for the backend identity.

    With no argument, the family of the accelerator visible on this node
    (``yeto.accel.available_type``); a CPU-only node reports ``"nvidia"`` so
    that unit tests and dry runs keep the historical value. ROCm and PPU nodes
    also read as ``"nvidia"`` unless :data:`IDENTITY_FROM_ENV_FAMILY` is on,
    which is the historical (wrong, but hash-compatible) behaviour.
    """
    if device_type is None and identity_family_from_env_enabled():
        launcher = os.environ.get(gpu_spec.DEVICE_FAMILY_ENV, "").strip()
        if launcher:
            try:
                return gpu_spec.hw_family(launcher).vendor
            except gpu_spec.UnknownFamily:
                pass
    if device_type is None:
        try:
            from yeto import accel  # noqa: PLC0415

            device_type = accel.available_type()
        except Exception:  # noqa: BLE001 - torch is optional at this call site
            device_type = "cpu"
    return DEVICE_FAMILY_BY_DEVICE_TYPE.get(device_type, "nvidia")


# Ray's accelerator resource name per identity device family.
# Ray counts CUDA cards in its own built-in "GPU" resource, so a CUDA island
# reads cluster_resources()["GPU"] and starts Ray with --num-gpus. An Ascend
# node's cards are the custom resource "NPU" instead: cluster_resources() has
# no "GPU" key at all, `ray start --num-gpus` does not create one that matches
# the cards, and an actor asks for them with resources={"NPU": n}. Verified by
# reading verl's own platform layer at fork acad9875
# (verl/plugin/platform/platform_npu.py ray_resource_name / ray_resource_options,
# verl/single_controller/ray/base.py _check_resource_available).
RAY_RESOURCE_BY_DEVICE_FAMILY: Mapping[str, str] = {
    f.vendor: f.ray_resource for f in gpu_spec.HW_FAMILIES if f.family != "cpu"}


def ray_accelerator_resource(device_family_name: str | None = None) -> str:
    """Ray's resource name for this node's accelerators ("GPU" or "NPU").

    ``device_family_name`` is an identity device family (:func:`device_family`);
    None reads this node's. An unknown family reads as "GPU", the historical
    value, so a new card never silently loses its cards to a KeyError."""
    if device_family_name is None:
        device_family_name = device_family()
    return RAY_RESOURCE_BY_DEVICE_FAMILY.get(device_family_name, "GPU")


def ray_start_accelerator_flag(count: int, device_family_name: str | None = None) -> str:
    """``ray start`` flag that gives a restarted Ray ``count`` accelerators.

    CUDA keeps ``--num-gpus=N``; an Ascend node needs ``--resources`` with the
    custom NPU resource, because ``ray start`` has no ``--num-npus``."""
    resource = ray_accelerator_resource(device_family_name)
    if resource == "GPU":
        return f"--num-gpus={int(count)}"
    return "--resources='" + json.dumps({resource: int(count)}) + "'"


# Phase 2 (design D9a): tolerances per (group, group) pair once calibrated.
# Empty in phase 1: every pair of different groups is "not calibrated".
# Consulted in the default strict mode; under --rl-identity relaxed compat_group
# is provenance (hetero-1114) and any pair is admitted with a warning.
CARD_PAIR_TOLERANCE: dict[frozenset[str], Mapping[str, float]] = {}

COMPAT_GROUP_ENV = "YETO_RL_COMPAT_GROUP"
# Event-tape record with the island's compat_group, driver and CUDA versions
# (dashboard: island "hardware"). Recorded only, never a refusal reason.
HARDWARE_EVENT = "rl_island_hardware"
_GROUP_RE = re.compile(r"^[a-z0-9]+-[a-z0-9][a-z0-9\-]*$")


class UnknownCardType(ValueError):
    """The card type is not in the catalog; the island cannot declare a group."""


class CompatGroupMismatch(ValueError):
    """Two islands are in different compatibility groups; they must not merge."""


def _canonical(card: str) -> str:
    if not isinstance(card, str) or not card.strip():
        raise UnknownCardType("card type is empty; cannot derive compat_group")
    raw = card.strip()
    for name in CARD_VENDOR:
        if name.lower() == raw.lower():
            return name
    raise UnknownCardType(f"card type {card!r} is not in the yeto card catalog "
                          f"(known: {sorted(CARD_VENDOR)}); cannot derive compat_group")


def nominal_hbm_gib(card: str, reported_gib: float) -> int:
    """The nominal HBM size of ``card`` that a reported (usable) size belongs to.

    Card tools report the size minus the driver reservation, so the largest
    nominal size within ``card_tools.HBM_TOLERANCE`` below the report wins.
    A card without known sizes returns the report rounded; a report that fits
    no known size raises :class:`UnknownCardType`."""
    from .card_tools import hbm_matches  # noqa: PLC0415 - card_tools imports gpu_spec

    name = _canonical(card)
    if not reported_gib or reported_gib <= 0:
        raise UnknownCardType(f"{name}: HBM size must be positive, got {reported_gib!r}")
    sizes = CARD_HBM_GIB.get(name, ())
    if not sizes:
        return int(round(reported_gib))
    fitting = [size for size in sizes if hbm_matches(size, reported_gib) and reported_gib <= size * 1.02]
    if not fitting:
        raise UnknownCardType(f"{name} with {reported_gib:g} GiB HBM is not a known variant "
                              f"(known sizes: {list(sizes)} GiB); cannot derive compat_group")
    return max(fitting)


def compat_group(card: str, hbm_gib: float | None = None) -> str:
    """``"H100"`` -> ``"nvidia-h100"``; with ``hbm_gib``: ``("910B4", 61.9)`` ->
    ``"ascend-910b4-64g"``. Unknown card or HBM variant -> :class:`UnknownCardType`."""
    name = _canonical(card)
    group = f"{CARD_VENDOR[name]}-{name.lower()}"
    if hbm_gib is None:
        return group
    try:
        size = nominal_hbm_gib(name, float(hbm_gib))
    except UnknownCardType:
        if name not in ROUNDED_HBM_CARDS or not hbm_gib or float(hbm_gib) <= 0:
            raise
        size = int(round(float(hbm_gib)))  # the label such an inventory had (ROUNDED_HBM_CARDS)
    return f"{group}-{size}g"


def launch_compat_group(card: str, explicit: str | None = None) -> str:
    """The group the launcher exports to an island on ``card``.

    ``explicit`` (``yeto launch --rl-compat-group``) replaces the card group on
    every island of the run: islands on different cards then declare one group
    and the syncer admits them together.  Their numeric tolerance is not
    calibrated (``CARD_PAIR_TOLERANCE`` is empty), so this is an operator
    decision, recorded on the launcher log and on each island's hardware event.
    The card must still be in the catalog: an unknown card fails here as before."""
    derived = compat_group(card)
    if not explicit:
        return derived
    return validate_compat_group(explicit)


def compat_group_family(group: str) -> str:
    """The launcher device family (``cuda``/``npu``/``ppu``) of a compat group."""
    vendor = validate_compat_group(group).split("-", 1)[0]
    try:
        return DEVICE_FAMILY_BY_VENDOR[vendor]
    except KeyError:
        raise ValueError(f"compat_group {group!r}: unknown vendor {vendor!r}") from None


def validate_compat_group(value: str) -> str:
    if not isinstance(value, str) or not _GROUP_RE.match(value):
        raise ValueError(f"compat_group must be '<vendor>-<card>' in lower case, got {value!r}")
    return value


def island_compat_group(explicit: str | None = None) -> str:
    """The group this island declares: the explicit value (``--rl-compat-group``)
    or ``$YETO_RL_COMPAT_GROUP``; neither set -> error (no default)."""
    value = explicit or os.environ.get(COMPAT_GROUP_ENV)
    if not value:
        raise UnknownCardType(f"island compat_group is not set (--rl-compat-group or ${COMPAT_GROUP_ENV}); "
                              "the launcher derives it from the --gpu card type")
    return validate_compat_group(value)


def compat_refusal(local: str | None, peer: str | None, *, strict: bool | None = None) -> str | None:
    """None when the two groups may merge, else the reason.

    ``--rl-identity relaxed`` (``strict`` False): compat_group is provenance,
    every pair may merge (the caller warns).  Strict (default; None reads
    ``$YETO_RL_IDENTITY``): phase-1 comparison, a pair with a calibrated
    tolerance is allowed; the table is empty now, so any difference is refused."""
    from yeto.rl.engine.backend_identity import strict_identity

    if not strict_identity(strict):
        return None
    if local == peer and local is not None:
        return None
    if local is not None and peer is not None and frozenset((local, peer)) in CARD_PAIR_TOLERANCE:
        return None
    return f"compat_group differs: {local or 'undeclared'} vs {peer or 'undeclared'}, tolerance not calibrated"


def check_compat_group(local: str | None, peer: str | None, *, strict: bool | None = None) -> None:
    reason = compat_refusal(local, peer, strict=strict)
    if reason is not None:
        raise CompatGroupMismatch(reason)


def _first_npu_id() -> str:
    """NPU ID of the first card ``npu-smi`` can see.

    A container may expose only a physical card such as ID 4 (ModelArts
    notebooks do); ``npu-smi info -t board -i 0`` then fails. ``npu-smi info -l``
    lists the visible IDs. Falls back to ``ASCEND_VISIBLE_DEVICES``, then "0".
    """
    try:
        res = subprocess.run(["npu-smi", "info", "-l"], capture_output=True, text=True, timeout=10)
        for line in res.stdout.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "NPU ID" and value.strip().isdigit():
                return value.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    visible = os.environ.get("ASCEND_VISIBLE_DEVICES", "").split(",")[0].strip()
    return visible if visible.isdigit() else "0"


def _npu_runtime_versions() -> dict[str, str | None]:
    """Ascend driver, CANN and torch_npu versions (best effort; rl-verl-backend 3.6)."""
    out: dict[str, str | None] = {"npu_driver_version": None, "cann_version": None,
                                  "torch_npu_version": None}
    try:
        res = subprocess.run(["npu-smi", "info", "-t", "board", "-i", _first_npu_id()],
                             capture_output=True, text=True, timeout=10)
        for line in res.stdout.splitlines():
            if "Software Version" in line or "Driver Version" in line:
                out["npu_driver_version"] = line.split(":", 1)[-1].strip()
                break
    except (OSError, subprocess.SubprocessError):
        pass
    for path in ("/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
                 "/usr/local/Ascend/ascend-toolkit/latest/version.info",
                 # CANN 8.x layout (seen on ModelArts snt9b images, CANN 8.5.1).
                 f"/usr/local/Ascend/ascend-toolkit/latest/{platform.machine()}-linux/"
                 "ascend_toolkit_install.info"):
        try:
            with open(path) as fh:
                text = fh.read()
        except OSError:
            continue
        for line in text.splitlines():
            if "version" in line.lower() and "=" in line:
                out["cann_version"] = line.split("=", 1)[1].strip()
                break
        if out["cann_version"]:
            break
    try:
        import torch_npu  # noqa: PLC0415

        out["torch_npu_version"] = getattr(torch_npu, "__version__", None)
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    return out


def runtime_versions(device_family_name: str | None = None) -> dict[str, str | None]:
    """Driver and runtime versions of this host, for logs/events only (best effort,
    never raises, never used to refuse an island).

    The NVIDIA keys stay exactly as before. On an Ascend island
    (``device_family_name="ascend"``, or auto-detected) three NPU keys are
    added: Ascend driver, CANN and torch_npu."""
    if device_family_name is None:
        device_family_name = device_family()
    out: dict[str, str | None] = {"driver_version": None, "cuda_version": None}
    if device_family_name == "ascend":
        out.update(_npu_runtime_versions())
        return out
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            out["driver_version"] = res.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch  # noqa: PLC0415

        out["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    return out
