"""Parser for the --gpu cluster specification grammar.

Grammar (comma-separated, one entry per learner cluster):

    entry   := cloud ":" [nodes "x"] count "x" gpu ["@" region]
    cloud   := "aws" | "gcp" | "azure" | ...
    nodes   := integer  (number of nodes in the cluster, default 1)
    count   := integer  (GPUs per node)
    gpu     := accelerator name, case-insensitive (a100, l4, h100, ...)
    region  := cloud-specific region string

Examples:
    aws:8xa100@us-east-2          -> 1 node  x 8xA100 in us-east-2
    aws:4x8xa100@us-east-2        -> 4 nodes x 8xA100 in us-east-2
    gcp:8xa100@us-central1        -> 1 node  x 8xA100 on GCP
    aws:1xl4@us-west-2            -> 1 node  x 1xL4 (g6 family)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical accelerator names as SkyPilot expects them.
_GPU_CANONICAL = {
    "a100": "A100",
    "a100-80gb": "A100-80GB",
    "a10g": "A10G",
    "l4": "L4",
    "l40s": "L40S",
    # Verda (sky catalog names; sm_89 like L40S / Blackwell workstation)
    "rtx-6000-ada": "RTX-6000-Ada",
    "rtx-pro-6000": "RTX-PRO-6000",
    "h100": "H100",
    "h200": "H200",
    "b200": "B200",
    "b300": "B300",
    "v100": "V100",
    # AMD Instinct (ROCm): the sampler nodes of the Kimi K3.1 line (MI355X, 288 GiB)
    "mi300x": "MI300X",
    "mi325x": "MI325X",
    "mi355x": "MI355X",
    "t4": "T4",
    # Ascend NPU (910B4 is the model the user bought, 2026-10-10). The
    # launcher spells it like any other card: ``ssh:1x8x910b4``.
    "910b4": "910B4",
    "910b": "910B",
    "910b2": "910B2",
    "910b3": "910B3",
    "910c": "910C",
    # Alibaba T-Head PPU (rl-ssh-node-pool). 810E and M890 are different
    # cards.
    "810e": "810E",
    "m890": "M890",
}

# torch device type per canonical card name. A card absent from this map runs
# on the CUDA device type; PPU cards are in that group on purpose, because
# torch on a PPU reports ``cuda`` (T-PPU-gaps.md 2.1). Only the cards listed
# here need the NPU path (ASCEND_RT_VISIBLE_DEVICES, hccl, npu-smi).
_CARD_DEVICE_TYPE = {"910B": "npu", "910B4": "npu", "910B2": "npu",
                     "910B3": "npu", "910C": "npu"}

# Device family per canonical card name: the launcher dimension that selects
# the card tool (nvidia-smi / npu-smi / ppu-smi / amd-smi, rocm-smi) and the
# collective-communication variables. It is not the torch device type: a PPU
# card has family ``ppu`` and device type ``cuda``; so does an AMD card (torch
# on ROCm reports ``cuda``), with family ``rocm``.
# (derived from HW_FAMILIES below; same tuple and order as before)
# Environment variable the island launcher sets to the node's device family;
# yeto.accel.device_family reads it (torch cannot tell a PPU from a GPU).
DEVICE_FAMILY_ENV = "YETO_DEVICE_FAMILY"
_CARD_DEVICE_FAMILY = {**{card: "npu" for card in _CARD_DEVICE_TYPE},
                       "810E": "ppu", "M890": "ppu",
                       "MI300X": "rocm", "MI325X": "rocm", "MI355X": "rocm"}


# --- hardware family model (T8-T11, S7) -------------------------------------
#
# Three spellings of "which hardware is this" grew side by side: the launcher
# family (cuda/npu/ppu/rocm, this module), the identity vendor
# (nvidia/ascend/thead/amd, ``yeto.hw.catalog``) and the torch device type
# (cuda/npu). Every per-family decision -- collective backend, Ray resource
# name, card tool, visible-device variable, memory collector -- was a separate
# dict somewhere, and each one had its own hole (``accel`` had no ``rocm`` at
# all; ``catalog`` read ROCm and PPU as ``nvidia``).
#
# :class:`HwFamily` is the one table those dicts now derive from. The values
# are exactly the ones already in use, so the NVIDIA and ROCm paths of the
# async GRPO delivery read the same as before (identity hashes included: the
# ``nvidia`` vendor of a CUDA/CPU node is unchanged, see
# ``catalog.device_family``).


class UnknownFamily(ValueError):
    """The hardware family name is not in :data:`HW_FAMILIES`."""


@dataclass(frozen=True)
class HwFamily:
    """One accelerator family: launcher name, identity vendor and its tooling."""

    family: str          # launcher/inventory name, $YETO_DEVICE_FAMILY
    vendor: str          # identity spelling (compat_group prefix)
    device_type: str     # torch device type
    dist_backend: str    # torch.distributed backend
    ray_resource: str    # Ray accelerator resource name
    smi: str             # primary card tool
    visible_devices_env: str | None  # variable that limits visible cards
    telemetry: str       # card-telemetry collector key (card_telemetry.collector_for)

    @property
    def is_npu(self) -> bool:
        return self.device_type == "npu"


HW_FAMILIES: tuple[HwFamily, ...] = (
    HwFamily("cuda", "nvidia", "cuda", "nccl", "GPU", "nvidia-smi",
             "CUDA_VISIBLE_DEVICES", "cuda"),
    HwFamily("npu", "ascend", "npu", "hccl", "NPU", "npu-smi",
             "ASCEND_RT_VISIBLE_DEVICES", "npu"),
    # torch on a PPU reports device type "cuda" and the collective API is
    # NCCL-compatible (served by ACCL-P); inferred from vendor docs, not
    # verified on a PPU machine.
    HwFamily("ppu", "thead", "cuda", "nccl", "GPU", "ppu-smi",
             "CUDA_VISIBLE_DEVICES", "ppu"),
    # torch on ROCm also reports "cuda" and its NCCL backend is RCCL.
    HwFamily("rocm", "amd", "cuda", "nccl", "GPU", "amd-smi",
             "HIP_VISIBLE_DEVICES", "rocm"),
    # Historical values, not a claim about CPU hardware: before this table a
    # CPU rank already reported vendor "nvidia" (identity default) and used the
    # "cuda"/NVML collector, and "GPU" is the Ray default. Kept so no
    # externally visible value changes; only device_type/dist_backend matter.
    HwFamily("cpu", "nvidia", "cpu", "gloo", "GPU", "", None, "cuda"),
)
#: Launcher device families (no ``cpu``): the inventory/``--gpu`` dimension.
DEVICE_FAMILIES = tuple(f.family for f in HW_FAMILIES if f.family != "cpu")
# Both spellings resolve: launcher family and identity vendor.
_HW_BY_NAME: dict[str, HwFamily] = {f.family: f for f in HW_FAMILIES}
_HW_BY_NAME.update({f.vendor: f for f in HW_FAMILIES if f.family != "cpu"})


def hw_family(name: str) -> HwFamily:
    """The :class:`HwFamily` of a family or vendor name (``"npu"``/``"ascend"``).

    Case-insensitive; an unknown name is an error (no silent NVIDIA default).
    """
    key = name.strip().lower() if isinstance(name, str) else ""
    try:
        return _HW_BY_NAME[key]
    except KeyError:
        raise UnknownFamily(f"unknown device family {name!r} "
                            f"(known: {sorted(_HW_BY_NAME)})") from None


def normalize_family(name: str) -> str:
    """The canonical launcher family name of a family or vendor spelling."""
    return hw_family(name).family


def hw_family_or(name: str | None, default: str = "cuda") -> HwFamily:
    """:func:`hw_family`, falling back to ``default`` for a missing/unknown name.

    Used where an unreadable value must not break a running island (telemetry,
    the sampler memory guard): the historical default is ``cuda``.
    """
    if name:
        try:
            return hw_family(name)
        except UnknownFamily:
            pass
    return hw_family(default)


class UnknownCard(ValueError):
    """The card name is not in :data:`_GPU_CANONICAL`; there is no default."""


def canonical_card(card: str) -> str:
    """The catalog spelling of ``card`` (case-insensitive); unknown -> error."""
    if not isinstance(card, str) or not card.strip():
        raise UnknownCard("card name is empty")
    raw = card.strip()
    name = _GPU_CANONICAL.get(raw.lower())
    if name is None:
        raise UnknownCard(f"card {card!r} is not in the yeto card catalog "
                          f"(known: {sorted(_GPU_CANONICAL.values())})")
    return name


def device_type_of(card: str) -> str:
    """torch device type (``"cuda"`` or ``"npu"``) of a card name.

    An unknown card is refused (T-NPU-gaps.md 1, recommendation 2): the old
    silent fallback to ``cuda`` would have run an unknown Ascend or PPU card
    down the NVIDIA path.
    """
    name = canonical_card(card)
    return _CARD_DEVICE_TYPE.get(name, "cuda")


def device_family_of(card: str) -> str:
    """``"cuda"``, ``"npu"``, ``"ppu"`` or ``"rocm"`` of a card name; unknown card -> error."""
    name = canonical_card(card)
    return _CARD_DEVICE_FAMILY.get(name, "cuda")


_ENTRY_RE = re.compile(
    r"^(?P<cloud>[a-z]+):"
    r"(?:(?P<nodes>\d+)x)?"
    r"(?P<count>\d+)x"
    r"(?P<gpu>[a-z0-9\-]+)"
    r"(?:@(?P<region>[a-z0-9\-]+))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClusterSpec:
    """One learner cluster."""

    cloud: str
    region: str | None
    num_nodes: int
    gpus_per_node: int
    gpu: str  # canonical SkyPilot accelerator name

    @property
    def accelerators(self) -> str:
        return f"{self.gpu}:{self.gpus_per_node}"

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    def __str__(self) -> str:
        loc = f"@{self.region}" if self.region else ""
        return f"{self.cloud}:{self.num_nodes}x{self.gpus_per_node}x{self.gpu}{loc}"


def parse_gpu_spec(spec: str) -> list[ClusterSpec]:
    """Parse a --gpu argument into one ClusterSpec per learner."""
    entries = [e.strip() for e in spec.split(",") if e.strip()]
    if not entries:
        raise ValueError("empty --gpu spec")
    clusters = []
    for entry in entries:
        m = _ENTRY_RE.match(entry)
        if not m:
            raise ValueError(
                f"bad --gpu entry {entry!r}; expected cloud:[NxM]xGPU[@region], "
                f"e.g. aws:4x8xa100@us-east-2"
            )
        gpu_raw = m.group("gpu").lower()
        gpu = _GPU_CANONICAL.get(gpu_raw)
        if gpu is None:
            raise ValueError(
                f"unknown GPU {gpu_raw!r} in {entry!r}; known: {sorted(_GPU_CANONICAL)}"
            )
        clusters.append(
            ClusterSpec(
                cloud=m.group("cloud").lower(),
                region=m.group("region"),
                num_nodes=int(m.group("nodes") or 1),
                gpus_per_node=int(m.group("count")),
                gpu=gpu,
            )
        )
    return clusters


def require_min_nodes(spec: ClusterSpec, min_nodes: int) -> ClusterSpec:
    """rl-multinode-island: refuse a learner island smaller than the recipe's
    minimum node count (pure; returns the spec unchanged)."""
    if isinstance(min_nodes, bool) or not isinstance(min_nodes, int) or min_nodes < 1:
        raise ValueError("min_nodes must be a positive int")
    if spec.num_nodes < min_nodes:
        raise ValueError(
            f"{spec} has {spec.num_nodes} node(s); this recipe/partition needs at least "
            f"{min_nodes} node(s) of {spec.gpus_per_node}x{spec.gpu}"
        )
    return spec


# --- npu-smi card assertion ------------------------------------------------
#
# ``npu-smi info`` prints one table row per chip, e.g.
#
#     | 0     910B4                   | OK              | 92.5  47  0 / 0     |
#     | 0                             | 0000:C1:00.0    | 0     0 / 0  3161 / 32768 |
#
# The first row of each pair carries the card name. The parser below reads only
# those rows, so a firmware change in the other columns does not break it.
_NPU_SMI_ROW_RE = re.compile(r"^\|\s*(\d+)\s+(\S+)\s*\|")


class NpuCardMismatch(RuntimeError):
    """``npu-smi`` reports a card name or card count the island did not ask for."""


def parse_npu_smi_names(text: str) -> list[str]:
    """Card names ``npu-smi info`` reports, in device-id order (pure function)."""
    found: dict[int, str] = {}
    for line in text.splitlines():
        m = _NPU_SMI_ROW_RE.match(line.strip())
        if m is None:
            continue
        index, name = int(m.group(1)), m.group(2)
        if name.lower() in ("name", "chip") or ":" in name:
            continue  # header row, or the bus-id row of the same chip
        found.setdefault(index, name)
    return [found[i] for i in sorted(found)]


def assert_npu_cards(text: str, card: str, count: int | None = None) -> list[str]:
    """Refuse before training when ``npu-smi info`` does not show ``count`` x ``card``.

    ``card`` is the canonical name (``"910B4"``); the comparison ignores case
    and the ``-1`` / ``-2`` suffix npu-smi adds for a chip variant (a 910B4-1
    is a 910B4). Returns the names found.
    """
    names = parse_npu_smi_names(text)
    if not names:
        raise NpuCardMismatch(
            "npu-smi info reported no card row; cannot assert the card type "
            f"(expected {card}). Output was: {text.strip()[:200]!r}")
    want = card.strip().lower()
    bad = [n for n in names if n.lower().split("-")[0] != want]
    if bad:
        raise NpuCardMismatch(
            f"npu-smi reports card(s) {sorted(set(bad))} but this island expects {card}; "
            "refusing to start (card type is a numerics boundary)")
    if count is not None and len(names) != count:
        raise NpuCardMismatch(
            f"npu-smi reports {len(names)} x {card}, the island asked for {count}; refusing to start")
    return names
