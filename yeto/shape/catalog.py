"""GPU capability table + sky catalog adapter.

The planner needs two things per candidate instance type: what it costs
(sky's AWS catalog) and what it can do (peak bf16 TFLOPs, derated by an
MFU heuristic and a spot-goodput factor). Sky is imported lazily so that
importing this module — e.g. from tests or the CLI — never pays sky's
startup cost or requires cloud credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Dense bf16/fp16 peak TFLOPs per GPU (no sparsity), keyed by sky
# accelerator name. Vendor datasheet numbers; kept ⊆ launcher.GPU_MEM_GB
# so every GPU we can plan for is also one the launcher can sanity-check.
PEAK_TFLOPS_BF16: dict[str, float] = {
    "T4": 65.0,
    "V100": 112.0,
    "L4": 121.0,
    "A10G": 125.0,
    "L40S": 362.0,
    "A100": 312.0,
    "A100-80GB": 312.0,
    "H100": 989.0,
    "H200": 989.0,
    "B200": 2250.0,
}


@dataclass(frozen=True)
class Offering:
    """One (GPU, instance type, region) row from a cloud catalog."""

    gpu: str  # sky accelerator name, e.g. "A100-80GB"
    instance_type: str  # e.g. "p4de.24xlarge" / "8x_H100_SECURE"
    gpus_per_node: int
    vcpus: int
    region: str
    spot_price: float | None  # $/hr per node
    on_demand_price: float | None
    gpu_mem_gb: int  # from launcher.GPU_MEM_GB
    cloud: str = "aws"  # lowercase sky cloud name
    # "catalog" (sky's periodic dump) or "live" (the cloud's own pricing
    # API, fetched during this shape); rendering marks live prices.
    price_source: str = "catalog"


# GPUs that predate bf16 (SM80/Ampere): the base is always trained in bf16,
# so these were never viable training targets here and are gated out.
_NO_BF16 = frozenset({"V100", "T4"})


def supports_bf16(gpu: str) -> bool:
    return gpu not in _NO_BF16


def efa_capable(instance_type: str) -> bool:
    """True for the p4d/p4de/p5 families — the only AWS GPU instances with
    EFA fabric, i.e. the only ones where multi-node data-parallel training
    is not bottlenecked on plain ENA networking."""
    return instance_type.startswith(("p4", "p5"))


# Clouds where a multi-node island can be provisioned at all (RunPod pods
# and Verda VMs are single machines to sky), and the subset whose
# multi-node islands get
# an RDMA-class fabric (EFA on AWS, InfiniBand on Nebius GPU clusters,
# RoCE on Modal clustered functions). Both grow as clouds pass the
# multi-node verification in docs/CLOUDS.md.
MULTI_NODE_CLOUDS = frozenset({"aws", "nebius", "modal"})
RDMA_CLOUDS = frozenset({"aws", "nebius", "modal"})
# RL islands run inside a digest-pinned container image and, on spot,
# persist finished rollout groups to an object store the launcher mounts.
# Neither is verified on every cloud; a cloud enters these sets only after
# the live checks in docs/CLOUDS.md pass (Modal: image via its registry
# support, checkpoints via a Modal Volume — both pending verification).
VERIFIED_DOCKER_IMAGE_CLOUDS = frozenset({"aws", "runpod"})
VERIFIED_SPOT_STORAGE_CLOUDS = frozenset({"aws"})


def _modal_gpu_count(instance_type: str) -> tuple[str, int]:
    """Modal 'instance types' are the GPU request string, e.g. 'H100:8'."""
    gpu, _, count = instance_type.partition(":")
    return gpu, int(count or 1)


def rdma_capable(cloud: str, instance_type: str) -> bool:
    """Whether a multi-node island of this shape gets an RDMA fabric
    (decides the multi-node MFU tier). AWS: EFA families. Nebius: the 8-GPU
    SXM presets, which sky places in an InfiniBand GPU cluster; PCIe and
    partial-node presets do not get the fabric. Modal: whole-node
    containers in a clustered function (the only multi-container shape
    Modal schedules) get RoCE."""
    if cloud not in RDMA_CLOUDS:
        return False
    if cloud == "aws":
        return efa_capable(instance_type)
    if cloud == "nebius":
        platform, _, preset = instance_type.partition("_")
        return "sxm" in platform and preset.startswith("8gpu")
    if cloud == "modal":
        from yeto.modal_runner import MODAL_FULL_NODE

        gpu, count = _modal_gpu_count(instance_type)
        return MODAL_FULL_NODE.get(gpu) == count
    return False


def multi_node_rejection(cloud: str, gpu: str, gpus_per_node: int) -> str | None:
    """Why a multi-node island of this per-node shape cannot be planned on
    `cloud`, or None when it can. Single-machine clouds reject every
    multi-node shape; Modal only schedules multi-container groups made of
    whole nodes."""
    if cloud not in MULTI_NODE_CLOUDS:
        return f"multi-node islands unsupported on {cloud}"
    if cloud == "modal":
        from yeto.modal_runner import MODAL_FULL_NODE

        full = MODAL_FULL_NODE.get(gpu)
        if full is None or gpus_per_node != full:
            return (
                "Modal multi-container islands must use whole nodes "
                f"({gpu}:{full} per container)" if full else
                f"Modal multi-container islands need a whole-node GPU, not {gpu}"
            )
    return None


def mfu(nodes: int, efa: bool) -> float:
    """Model FLOPs utilization heuristic. 0.35 single-node (NVLink only,
    typical for tuned fine-tuning stacks); 0.30 multi-node over EFA (fabric
    adds sync stalls); 0.20 multi-node over plain TCP (all-gather/reduce
    dominates). Deliberately conservative — the planner only needs the
    *relative* ranking of candidate shapes to be right."""
    if nodes == 1:
        return 0.35
    return 0.30 if efa else 0.20


def goodput(score: int | None) -> float:
    """Fraction of wall-clock compute a spot island actually delivers,
    from AWS's 1-10 spot placement score (preemption/restart losses eat
    the rest). Linear 0.5 + 0.05*score, capped at 0.98 — even a perfect
    score never means zero interruptions. None (score unavailable, e.g.
    quota-limited API) falls back to 0.85: mid-range optimism so unknown
    regions are neither favored nor written off."""
    if score is None:
        return 0.85
    return min(0.98, 0.5 + 0.05 * score)


def effective_tflops(off: Offering, nodes: int, score: int | None) -> float:
    """Expected delivered TFLOPs of an island of `nodes` nodes of this
    offering: peak, derated by MFU and spot goodput."""
    return (
        nodes
        * off.gpus_per_node
        * PEAK_TFLOPS_BF16[off.gpu]
        * mfu(nodes, rdma_capable(off.cloud, off.instance_type))
        * goodput(score)
    )


def _fetch_raw(
    regions: list[str] | None, gpus: list[str] | None, clouds: tuple[str, ...]
) -> dict[str, list[Any]]:
    """The raw sky catalog call, factored out so tests can monkeypatch it.

    Sky's local catalog is a full dump, so we fetch everything for the
    requested clouds and filter in `_to_rows` — that keeps this function a
    pure I/O boundary (regions/gpus are accepted for signature symmetry
    with the cache key).
    """
    del regions, gpus  # filtering happens downstream, see docstring
    try:
        from sky import catalog as sky_catalog
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "yeto shape needs SkyPilot's catalog to price instances; "
            "install it with `pip install skypilot[aws]`"
        ) from exc
    return sky_catalog.list_accelerators(gpus_only=True, clouds=list(clouds), all_regions=True)


def _to_rows(
    raw: dict[str, list[Any]], regions: list[str], gpus: list[str] | None
) -> list[dict[str, Any]]:
    """Filter + normalize sky's InstanceTypeInfo records into JSON-safe
    dicts (the shape the cache stores, so cached and fresh results take
    the identical path back to Offerings)."""
    from yeto import launcher  # heavy module; sky inside it is lazy

    want_regions = set(regions) if regions is not None else None  # None = all
    want_gpus = set(gpus) if gpus else None
    rows: list[dict[str, Any]] = []
    for gpu, infos in raw.items():
        if gpu not in PEAK_TFLOPS_BF16:
            continue
        if want_gpus is not None and gpu not in want_gpus:
            continue
        for info in infos:
            if info.instance_type is None:
                continue
            cloud = str(getattr(info, "cloud", None) or "aws").lower()
            # --regions means AWS regions; other clouds (RunPod country
            # codes, etc.) have their own geography and pass unfiltered.
            if cloud == "aws" and want_regions is not None and info.region not in want_regions:
                continue
            if not info.accelerator_count or int(info.accelerator_count) < 1:
                continue  # fractional-GPU shapes are useless for training
            rows.append(
                {
                    "gpu": gpu,
                    "instance_type": info.instance_type,
                    "gpus_per_node": int(info.accelerator_count),
                    "vcpus": int(info.cpu_count or 0),
                    "region": info.region,
                    "spot_price": info.spot_price,
                    "on_demand_price": info.price,
                    "gpu_mem_gb": launcher.GPU_MEM_GB[gpu],
                    "cloud": cloud,
                }
            )
    return rows


def list_offerings(
    regions: list[str] | None,
    gpus: list[str] | None = None,
    cache: Any = None,
    clouds: tuple[str, ...] = ("aws",),
) -> list[Offering]:
    """All offerings for the requested clouds/GPUs/regions (regions=None
    means every AWS region; non-AWS clouds are never region-filtered),
    deterministically ordered so plans are reproducible run-to-run.

    `cache` is any object with `.get_or(key, fetch)` (e.g. shape.cache's
    disk cache) — the sky catalog dump is slow to load, so callers doing
    repeated planning pass one; None always fetches fresh.
    """
    def fetch() -> list[dict[str, Any]]:
        return _to_rows(_fetch_raw(regions, gpus, clouds), regions, gpus)

    region_part = ",".join(sorted(regions)) if regions is not None else "*"
    key = (
        f"catalog:v2:{','.join(sorted(clouds))}:{region_part}:"
        + ",".join(sorted(gpus or []))
    )
    rows = cache.get_or(key, fetch) if cache is not None else fetch()
    offerings = [Offering(**row) for row in rows]
    # Biggest nodes first within a (gpu, region) group: the planner prefers
    # fewer, fatter nodes (fewer network hops per island).
    offerings.sort(key=lambda o: (o.cloud, o.gpu, o.region, -o.gpus_per_node, o.instance_type))
    return offerings
