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
    "h100": "H100",
    "h200": "H200",
    "v100": "V100",
    "t4": "T4",
}

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
