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
    "v100": "V100",
    "t4": "T4",
    # Ascend NPU (910B4 is the model the user bought, 2026-10-10). The
    # launcher spells it like any other card: ``ssh:1x8x910b4``.
    "910b4": "910B4",
    "910b": "910B",
}

# Accelerator families per canonical card name. A card absent from this map is
# an NVIDIA GPU; only the cards listed here need the NPU launcher path
# (ASCEND_RT_VISIBLE_DEVICES, hccl, npu-smi).
_CARD_DEVICE_TYPE = {"910B": "npu", "910B4": "npu"}


def device_type_of(card: str) -> str:
    """``"cuda"`` or ``"npu"`` for a canonical card name (:data:`_GPU_CANONICAL`)."""
    return _CARD_DEVICE_TYPE.get(card, "cuda")


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
    """rl-multinode-island D8: refuse a learner island smaller than the recipe's
    minimum node count (pure; returns the spec unchanged)."""
    if isinstance(min_nodes, bool) or not isinstance(min_nodes, int) or min_nodes < 1:
        raise ValueError("min_nodes must be a positive int")
    if spec.num_nodes < min_nodes:
        raise ValueError(
            f"{spec} has {spec.num_nodes} node(s); this recipe/partition needs at least "
            f"{min_nodes} node(s) of {spec.gpus_per_node}x{spec.gpu}"
        )
    return spec


# --- npu-smi card assertion (task 3.9) ---------------------------------------
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
