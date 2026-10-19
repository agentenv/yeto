"""Card-type catalog and island compatibility group
(yeto-framework-decoupling 7.7a, design D9a-1).

``compat_group`` is ``"<vendor>-<card>"`` in lower case, e.g. ``nvidia-h100``.
With the HBM size (S25 N2.1) it is ``"<vendor>-<card>-<N>g"``, e.g.
``ascend-910b4-64g``: the 910B4 ships with 32 GiB and with 64 GiB (the user's
cluster changed from one to the other on 2026-10-16), and the two must not
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

import os
import platform
import re
import subprocess
from typing import Mapping

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
    # Alibaba T-Head PPU (rl-ssh-node-pool D2). torch reports device type
    # "cuda" on these cards, so the vendor comes from the card name only
    # (infra-drafts/SESSION24/T-PPU-gaps.md 2.1).
    "810E": "thead", "M890": "thead",
}

# Nominal HBM sizes (GiB) a card ships with. Read from our machines: 910B4
# 64 GiB (inventory, 2026-10-16) and 32 GiB (ModelArts), M890 144 GiB
# (ppu-smi 147456 MiB, SESSION25/N0.1-ppu-survey.md). The others are public
# product figures, not read on our nodes. 810E: not known, so any reported
# size is used as is (rounded).
CARD_HBM_GIB: Mapping[str, tuple[int, ...]] = {
    "910B": (32, 64), "910B4": (32, 64), "910B2": (64,), "910B3": (64,), "910C": (128,),
    "810E": (), "M890": (144,),
    "H100": (80,), "H200": (141,), "A100": (40, 80), "A100-80GB": (80,), "B200": (180,),
}

# torch device type -> device family in the backend identity. A PPU is absent
# on purpose: its torch device type is "cuda", so only the card name tells it
# apart from an NVIDIA GPU (see VENDOR_BY_DEVICE_FAMILY).
DEVICE_FAMILY_BY_DEVICE_TYPE: Mapping[str, str] = {"cuda": "nvidia", "npu": "ascend"}

# Launcher device family (yeto.gpu_spec.DEVICE_FAMILIES) -> vendor.
VENDOR_BY_DEVICE_FAMILY: Mapping[str, str] = {"cuda": "nvidia", "npu": "ascend",
                                              "ppu": "thead"}
DEVICE_FAMILY_BY_VENDOR: Mapping[str, str] = {v: k for k, v in VENDOR_BY_DEVICE_FAMILY.items()}


def device_family(device_type: str | None = None) -> str:
    """Device family for the backend identity.

    With no argument, the family of the accelerator visible on this node
    (``yeto.accel.available_type``); a CPU-only node reports ``"nvidia"`` so
    that unit tests and dry runs keep the historical value.
    """
    if device_type is None:
        try:
            from yeto import accel  # noqa: PLC0415

            device_type = accel.available_type()
        except Exception:  # noqa: BLE001 - torch is optional at this call site
            device_type = "cpu"
    return DEVICE_FAMILY_BY_DEVICE_TYPE.get(device_type, "nvidia")


# Phase 2 (design D9a): tolerances per (group, group) pair once calibrated.
# Empty in phase 1: every pair of different groups is "not calibrated".
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
    return f"{group}-{nominal_hbm_gib(name, float(hbm_gib))}g"


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


def compat_refusal(local: str | None, peer: str | None) -> str | None:
    """Phase-1 comparison: None when the two groups may merge, else the reason.
    A pair with a calibrated tolerance would be allowed in phase 2; the table
    is empty now, so any difference is refused."""
    if local == peer and local is not None:
        return None
    if local is not None and peer is not None and frozenset((local, peer)) in CARD_PAIR_TOLERANCE:
        return None
    return (f"兼容组不同：{local or '未声明'} 对 {peer or '未声明'}，容差未标定 "
            f"(compat_group differs: {local} vs {peer}, tolerance not calibrated)")


def check_compat_group(local: str | None, peer: str | None) -> None:
    reason = compat_refusal(local, peer)
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
    added: Ascend driver, CANN and torch_npu (task 3.6)."""
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
