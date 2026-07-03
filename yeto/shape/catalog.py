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
    """One (GPU, instance type, region) row from the AWS catalog."""

    gpu: str  # sky accelerator name, e.g. "A100-80GB"
    instance_type: str  # e.g. "p4de.24xlarge"
    gpus_per_node: int
    vcpus: int
    region: str
    spot_price: float | None  # $/hr per node
    on_demand_price: float | None
    gpu_mem_gb: int  # from launcher.GPU_MEM_GB


def efa_capable(instance_type: str) -> bool:
    """True for the p4d/p4de/p5 families — the only AWS GPU instances with
    EFA fabric, i.e. the only ones where multi-node data-parallel training
    is not bottlenecked on plain ENA networking."""
    return instance_type.startswith(("p4", "p5"))


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
        * mfu(nodes, efa_capable(off.instance_type))
        * goodput(score)
    )


def _fetch_raw(regions: list[str], gpus: list[str] | None) -> dict[str, list[Any]]:
    """The raw sky catalog call, factored out so tests can monkeypatch it.

    Sky's local catalog is a full dump, so we fetch everything for AWS and
    filter in `_to_rows` — that keeps this function a pure I/O boundary
    (regions/gpus are accepted for signature symmetry with the cache key).
    """
    del regions, gpus  # filtering happens downstream, see docstring
    try:
        from sky import catalog as sky_catalog
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "yeto shape needs SkyPilot's catalog to price instances; "
            "install it with `pip install skypilot[aws]`"
        ) from exc
    return sky_catalog.list_accelerators(gpus_only=True, clouds="aws", all_regions=True)


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
            if want_regions is not None and info.region not in want_regions:
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
                }
            )
    return rows


def list_offerings(
    regions: list[str] | None, gpus: list[str] | None = None, cache: Any = None
) -> list[Offering]:
    """All AWS offerings for the requested GPUs/regions (regions=None means
    every region in the catalog), deterministically ordered so plans are
    reproducible run-to-run.

    `cache` is any object with `.get_or(key, fetch)` (e.g. shape.cache's
    disk cache) — the sky catalog dump is slow to load, so callers doing
    repeated planning pass one; None always fetches fresh.
    """
    def fetch() -> list[dict[str, Any]]:
        return _to_rows(_fetch_raw(regions, gpus), regions, gpus)

    region_part = ",".join(sorted(regions)) if regions is not None else "*"
    key = f"catalog:aws:{region_part}:" + ",".join(sorted(gpus or []))
    rows = cache.get_or(key, fetch) if cache is not None else fetch()
    offerings = [Offering(**row) for row in rows]
    # Biggest nodes first within a (gpu, region) group: the planner prefers
    # fewer, fatter nodes (fewer network hops per island).
    offerings.sort(key=lambda o: (o.gpu, o.region, -o.gpus_per_node, o.instance_type))
    return offerings
