"""Orchestrate a shape plan: gather signals in parallel, filter, solve, render.

The expensive part is talking to AWS (quota limits, spot placement scores)
and the HF Hub (weight sizes). Everything goes through a shared 1-hour TTL
disk cache, and the two AWS signal families are fetched concurrently — each
provider also fans out internally — so a cold run is one network wave and a
warm run is zero network.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..gpu_spec import _GPU_CANONICAL
from .cache import TTLCache
from .catalog import Offering, effective_tflops, list_offerings
from .ilp import Candidate, Plan, solve
from .memory import fits, min_nodes, model_weights_gb
from .providers import AwsProviders, QuotaKey

DEFAULT_REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]
HEAD_COST_PER_HOUR = 0.40  # small on-demand CPU VM hosting syncer + controller

# sky accelerator name -> the lowercase name the --gpu grammar accepts.
_GPU_FLAG_NAME = {v: k for k, v in _GPU_CANONICAL.items()}


@dataclass(frozen=True)
class Rejection:
    key: str
    reason: str


@dataclass
class ShapeResult:
    plan: Plan
    candidates: list[Candidate]
    rejections: list[Rejection]
    weights_gb: float
    shard: str  # ddp when the model fits one GPU, else fsdp
    fetch_seconds: float


def _candidate_key(off: Offering, nodes: int) -> str:
    gpu = _GPU_FLAG_NAME.get(off.gpu, off.gpu.lower())
    prefix = f"{nodes}x" if nodes > 1 else ""
    return f"aws:{prefix}{off.gpus_per_node}x{gpu}@{off.region}"


def build_shape(
    model: str,
    budget: float,
    tuning: str = "lora",
    seq_len: int = 2048,
    regions: list[str] | None = None,
    gpus: list[str] | None = None,
    min_score: int = 7,
    max_islands: int = 16,
    weights_gb_override: float | None = None,
    cache_enabled: bool = True,
    providers: AwsProviders | None = None,
) -> ShapeResult:
    """Compute the fleet plan. `providers` is injectable for tests."""
    t0 = time.monotonic()
    cache = TTLCache(enabled=cache_enabled)
    aws = providers or AwsProviders(cache)
    regions = regions or DEFAULT_REGIONS

    # Static-ish facts: weight size (hub, cached) + catalog (sky, cached).
    with ThreadPoolExecutor(max_workers=2) as pool:
        weights_f = pool.submit(model_weights_gb, model, weights_gb_override, cache)
        offerings_f = pool.submit(list_offerings, regions, gpus, cache)
        weights = weights_f.result()
        offerings = offerings_f.result()

    # Several instance types can expose the same (gpu, count, region) shape
    # (e.g. g6e.xlarge vs g6e.2xlarge, both 1xL40S); keep the cheapest —
    # the others are strictly dominated for our purposes.
    cheapest: dict[tuple[str, int, str], Offering] = {}
    for off in offerings:
        if off.spot_price is None:
            continue
        k = (off.gpu, off.gpus_per_node, off.region)
        if k not in cheapest or off.spot_price < cheapest[k].spot_price:
            cheapest[k] = off
    offerings = sorted(cheapest.values(), key=lambda o: (o.gpu, o.region, -o.gpus_per_node))

    # Island shapes: exactly min_nodes per offering — larger islands only
    # lose (worse placement odds, bigger failure blast radius, multi-node
    # MFU discount), so they are dominated by more min-sized islands.
    rejections: list[Rejection] = []
    sized: list[tuple[Offering, int]] = []
    for off in offerings:
        if off.spot_price is None:
            rejections.append(Rejection(_candidate_key(off, 1), "no spot price in catalog"))
            continue
        nodes = min_nodes(weights, tuning, off.gpu_mem_gb, off.gpus_per_node, seq_len)
        if nodes is None:
            rejections.append(
                Rejection(_candidate_key(off, 1), f"model does not fit (≤8 nodes of {off.gpus_per_node}x{off.gpu})")
            )
            continue
        sized.append((off, nodes))

    # One parallel wave for the two AWS signal families.
    quota_keys: dict[tuple[Offering, int], QuotaKey | None] = {}
    for off, nodes in sized:
        code = aws.quota_code(off.instance_type, use_spot=True)
        quota_keys[(off, nodes)] = QuotaKey(off.region, code) if code else None
    unique_quotas = sorted({k for k in quota_keys.values() if k}, key=lambda k: (k.region, k.code))
    unique_asks = sorted({(off.instance_type, nodes) for off, nodes in sized})
    with ThreadPoolExecutor(max_workers=2) as pool:
        quotas_f = pool.submit(aws.quotas, unique_quotas)
        scores_f = pool.submit(aws.placement_scores, unique_asks, regions)
        quotas = quotas_f.result()
        scores = scores_f.result()

    # Filter into solver candidates.
    candidates: list[Candidate] = []
    quota_limits: dict[tuple[str, str], float] = {}
    for off, nodes in sized:
        key = _candidate_key(off, nodes)
        qk = quota_keys[(off, nodes)]
        limit = quotas.get(qk) if qk else None
        if qk is not None:
            if limit is None:
                rejections.append(Rejection(key, f"quota {qk.code}@{qk.region} unavailable"))
                continue
            if limit <= 0:
                rejections.append(Rejection(key, f"zero spot quota ({qk.code}@{qk.region})"))
                continue
            if off.vcpus * nodes > limit:
                rejections.append(
                    Rejection(key, f"one island needs {off.vcpus * nodes} vCPUs > quota {limit:.0f}")
                )
                continue
            quota_limits[(qk.region, qk.code)] = limit
        score = scores.get((off.instance_type, nodes, off.region))
        if min_score > 0:
            if score is None:
                rejections.append(Rejection(key, "placement score unavailable"))
                continue
            if score <= min_score:
                rejections.append(Rejection(key, f"placement score {score} ≤ {min_score}"))
                continue
        candidates.append(
            Candidate(
                key=key,
                region=off.region,
                gpu=off.gpu,
                instance_type=off.instance_type,
                nodes=nodes,
                gpus_per_node=off.gpus_per_node,
                vcpus_per_island=off.vcpus * nodes,
                price_per_hour=off.spot_price * nodes,
                eff_tflops=effective_tflops(off, nodes, score),
                quota_bucket=(qk.region, qk.code) if qk else None,
                score=score,
            )
        )

    plan = solve(candidates, budget, quota_limits, max_islands, HEAD_COST_PER_HOUR)
    # DDP only if the model fits a single GPU on every *planned* island;
    # otherwise the launch line must say fsdp.
    by_key = {c.key: c for c in candidates}
    planned_mems = [
        next(o.gpu_mem_gb for o, _ in sized if o.instance_type == by_key[k].instance_type and o.region == by_key[k].region)
        for k in plan.counts
    ]
    shard = "ddp" if planned_mems and all(fits(weights, tuning, m, 1, seq_len) for m in planned_mems) else "fsdp"
    return ShapeResult(plan, candidates, rejections, weights, shard, time.monotonic() - t0)


def render(result: ShapeResult, model: str, budget: float, tuning: str, min_score: int) -> str:
    """Human-readable plan + the ready-to-run launch line."""
    plan, out = result.plan, []
    by_key = {c.key: c for c in result.candidates}
    if not plan.counts:
        out.append(f"no feasible plan under ${budget:.2f}/hr (weights ~{result.weights_gb:.0f} GB, {tuning})")
    else:
        islands = sum(plan.counts.values())
        out.append(
            f"plan: {islands} island(s), {plan.total_tflops:.1f} effective TFLOPs, "
            f"${plan.total_cost:.2f}/hr (budget ${budget:.2f}/hr)"
        )
        for key, n in sorted(plan.counts.items()):
            c = by_key[key]
            out.append(
                f"  {n}x {key}  spot ${c.price_per_hour:.2f}/hr/island  "
                f"score {c.score}  {c.eff_tflops:.1f} TFLOPs/island"
            )
        out.append(f"  head: on-demand CPU VM  ${HEAD_COST_PER_HOUR:.2f}/hr")
    if plan.binding:
        out.append(f"binding constraints: {', '.join(plan.binding)}")
    if result.rejections:
        out.append("rejected:")
        for line in sorted({f"  {r.key}: {r.reason}" for r in result.rejections}):
            out.append(line)
    if plan.counts:
        entries = []
        for key, n in sorted(plan.counts.items()):
            entries.extend([key] * n)
        out.append(
            "launch: yeto launch"
            f" --gpu {','.join(entries)}"
            f" --model {model} --tuning {tuning} --shard {result.shard}"
            " --data <hf-dataset>"
        )
    out.append(f"(signals fetched in {result.fetch_seconds:.1f}s; cached for 1h)")
    return "\n".join(out)
