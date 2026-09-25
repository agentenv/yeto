"""Orchestrate a shape plan: gather signals in parallel, filter, solve, render.

The expensive part is talking to AWS (quota limits + current usage, spot
placement scores) and the HF Hub (weight sizes). Everything goes through a
shared TTL disk cache, and the AWS signal families are fetched concurrently —
each provider also fans out internally — so a cold run is one network wave
and a warm run is zero network.

Two honesty rules shape the output: catalog spot prices are estimates (they
move), so the budget is enforced against margin-inflated prices; and a plan
that stacks several islands of one shape in one region is re-checked against
the placement score at the *aggregate* capacity, since obtainability decays
with size.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from ..gpu_spec import _GPU_CANONICAL
from .cache import TTLCache
from .catalog import (
    PEAK_TFLOPS_BF16,
    Offering,
    effective_tflops,
    list_offerings,
    multi_node_rejection,
    supports_bf16,
)
from .ilp import Candidate, Plan, solve
from .memory import fits, min_nodes, model_weights_gb
from .providers import (
    CLOUD_SIGNALS,
    AwsProviders,
    CloudSignals,
    QuotaKey,
    credentials_available,
)
from .regions import parse_regions

DEFAULT_REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]
HEAD_COST_PER_HOUR = 0.40  # small on-demand CPU VM hosting syncer + controller
DEFAULT_PRICE_MARGIN = 0.15  # catalog spot prices are stale-ish; budget with headroom
# AWS allows ~50 *distinct* placement-score configurations per day; spend
# them on the shapes most likely to win and say so for the rest.
MAX_SCORE_ASKS = 24
# When a score fetch fails (throttled, daily config budget spent), default
# planning assumes the best score rather than rejecting the shape; the plan
# carries a warning and --strict-capacity-check restores rejection.
ASSUMED_PLACEMENT_SCORE = 10

# sky accelerator name -> the lowercase name the --gpu grammar accepts.
_GPU_FLAG_NAME = {v: k for k, v in _GPU_CANONICAL.items()}


@dataclass(frozen=True)
class Rejection:
    key: str
    reason: str


@dataclass
class ShapeResult:
    plan: Plan  # solved against margin-inflated prices
    candidates: list[Candidate]  # raw (un-inflated) prices
    rejections: list[Rejection]
    warnings: list[str]
    weights_gb: float
    shard: str  # ddp when the model fits one GPU on every planned island
    est_cost: float  # Σ islands at raw catalog prices + head
    price_margin: float
    head_cost: float
    fetch_seconds: float


def _candidate_key(off: Offering, nodes: int) -> str:
    """The `--gpu` entry for this shape. A cloud that does not pin
    placement leaves the region empty, and the key carries no `@`."""
    gpu = _GPU_FLAG_NAME.get(off.gpu, off.gpu.lower())
    prefix = f"{nodes}x" if nodes > 1 else ""
    loc = f"@{off.region}" if off.region else ""
    return f"{off.cloud}:{prefix}{off.gpus_per_node}x{gpu}{loc}"


def build_shape(
    model: str,
    budget: float | None = None,
    tuning: str = "lora",
    seq_len: int = 2048,
    regions: list[str] | None = None,
    gpus: list[str] | None = None,
    min_score: int = 7,
    max_islands: int = 16,
    weights_gb_override: float | None = None,
    cache_enabled: bool = True,
    providers: AwsProviders | None = None,
    price_margin: float = DEFAULT_PRICE_MARGIN,
    head_cost: float = HEAD_COST_PER_HOUR,
    skip_capacity_check: bool = False,
    strict_capacity_check: bool = False,
    clouds: list[str] | None = None,
    signals: dict[str, CloudSignals] | None = None,
    target_tflops: float | None = None,
) -> ShapeResult:
    """Compute the fleet plan. `providers` (AWS) and `signals` (every other
    cloud, keyed by name) are injectable for tests.

    Exactly one objective input is required: `budget` (maximize TFLOPs under
    a $/hr cap) or `target_tflops` (minimize cost to reach a throughput
    target); passing both means "cheapest plan reaching the target, capped
    at the budget".

    clouds: None -> aws, plus every cloud in CLOUD_SIGNALS whose credentials
    are present locally. An explicit list is used as-is: unknown names are
    a ValueError and a listed cloud without credentials is a RuntimeError
    (naming where the credentials are expected). AWS credentials are only
    required when aws is in the list.

    Non-AWS clouds have no quota system; their binding constraint is
    machine stock/capacity, mapped onto the same 1-10 pseudo-score scale as
    AWS placement scores (9 plenty, 6 some, 3 little, 0 sold out) and
    gated/capped accordingly.

    regions: entries of the form `cloud:region` (a bare region means aws;
    `cloud:all` / `all` lift the limit); see shape/regions.py. None ->
    AWS limited to DEFAULT_REGIONS, every other cloud unrestricted. A
    user-named region with no offerings is a warning when the cloud still
    has other named regions with offerings, and a ValueError (listing the
    regions that do have offerings) when none of them do.

    Score-availability policy: measured scores always gate normally. When a
    score cannot be fetched (throttled, daily config budget spent), the
    default assumes ASSUMED_PLACEMENT_SCORE with a warning;
    strict_capacity_check rejects such shapes instead, and
    skip_capacity_check makes no score API calls at all (quota + price only).
    """
    t0 = time.monotonic()
    if budget is None and target_tflops is None:
        raise ValueError("pass --budget and/or --flops: the plan needs an objective")
    cache = TTLCache(enabled=cache_enabled)
    signals = dict(signals or {})
    if clouds is None:
        # Default fleet: AWS plus every registered cloud whose credentials
        # are on this machine. `available()` is a local check (env vars,
        # files) — a cloud that stays out never sees a network request.
        clouds = ["aws"] + [
            name
            for name, factory in CLOUD_SIGNALS.items()
            if name in signals or factory(cache).available()
        ]
    clouds = list(dict.fromkeys(clouds))
    unknown = [c for c in clouds if c != "aws" and c not in CLOUD_SIGNALS and c not in signals]
    if unknown:
        raise ValueError(
            f"unknown cloud(s) {', '.join(unknown)}; known: "
            f"{', '.join(['aws'] + sorted(CLOUD_SIGNALS))}"
        )
    for name in clouds:
        if name == "aws" or name in signals:
            continue
        sig = CLOUD_SIGNALS[name](cache)
        if not sig.available():
            raise RuntimeError(
                f"{name} credentials not found — expected {sig.credential_hint()}"
            )
        signals[name] = sig
    signals = {name: sig for name, sig in signals.items() if name in clouds}
    if "aws" in clouds and providers is None and not credentials_available():
        raise RuntimeError(
            "AWS credentials not found — quota and placement-score signals "
            "need them (run `aws configure` or set AWS_PROFILE, or leave aws "
            "out of --clouds)"
        )
    aws = providers if providers is not None else (AwsProviders(cache) if "aws" in clouds else None)
    region_filter = parse_regions(
        regions, DEFAULT_REGIONS, ["aws", *CLOUD_SIGNALS, *signals]
    )

    def fetch_offerings() -> list[Offering]:
        # Clouds that maintain their own catalog (sky's dump is incomplete
        # for them) hand back rows; everyone else rides one sky call. The
        # catalog comes back unfiltered by region so an unknown region can
        # be reported against the regions that DO exist.
        own: dict[str, list[Offering]] = {}
        for name, sig in signals.items():
            rows = sig.offerings(region_filter.for_cloud(name), gpus, cache)
            if rows is not None:
                own[name] = rows
        sky_clouds = tuple(sorted(c for c in clouds if c not in own))
        rows = list_offerings(None, gpus, cache, sky_clouds) if sky_clouds else []
        return rows + [o for name in sorted(own) for o in own[name]]

    # Static-ish facts: weight size (hub, cached) + catalog (sky, cached).
    with ThreadPoolExecutor(max_workers=2) as pool:
        weights_f = pool.submit(model_weights_gb, model, weights_gb_override, cache)
        offerings_f = pool.submit(fetch_offerings)
        weights = weights_f.result()
        unfiltered = offerings_f.result()

    # Apply the region filter, and be honest about user-named regions with
    # nothing in them (for the requested GPUs).
    region_notes: list[str] = []
    for cloud in clouds:
        wanted = region_filter.for_cloud(cloud)
        if not wanted or not region_filter.is_explicit(cloud):
            continue
        have = {o.region for o in unfiltered if o.cloud == cloud}
        missing = wanted - have
        if not missing:
            continue
        gpu_note = f" for GPUs {', '.join(gpus)}" if gpus else ""
        listing = ", ".join(sorted(have)) or "none"
        if wanted - missing:
            region_notes.append(
                f"{cloud}: no offerings in region(s) {', '.join(sorted(missing))}{gpu_note}; "
                f"planning the rest ({cloud} regions with offerings: {listing})"
            )
        else:
            raise ValueError(
                f"no {cloud} offerings in region(s) {', '.join(sorted(missing))}{gpu_note}; "
                f"{cloud} regions with offerings: {listing}"
            )
    offerings = [o for o in unfiltered if region_filter.allows(o.cloud, o.region)]

    signal_notes: list[str] = []
    # AWS placement-score asks take a stable region list: the user's
    # allowlist, or every catalog region when unrestricted.
    aws_wanted = region_filter.for_cloud("aws")
    regions = (
        sorted(aws_wanted)
        if aws_wanted is not None
        else sorted({o.region for o in offerings if o.cloud == "aws"})
    )

    # Catalog gaps: a cloud can know how to check stock for a GPU its
    # catalog never lists (RunPod/H200). Say so rather than silently never
    # planning that GPU there.
    gap_notes: list[str] = []
    for name, sig in signals.items():
        present = {o.gpu for o in offerings if o.cloud == name}
        gaps = (set(sig.known_gpus()) & set(PEAK_TFLOPS_BF16)) - present
        if gpus:
            gaps &= set(gpus)
        if gaps:
            gap_notes.append(
                f"{name}: catalog has no rows for {', '.join(sorted(gaps))}; "
                f"those GPUs cannot be planned on {name}"
            )

    # Several instance types can expose the same (gpu, count, region) shape
    # (e.g. g6e.xlarge vs g6e.2xlarge, both 1xL40S); keep the cheapest —
    # the others are strictly dominated for our purposes.
    cheapest: dict[tuple[str, str, int, str], Offering] = {}
    for off in offerings:
        if off.spot_price is None:
            continue
        k = (off.cloud, off.gpu, off.gpus_per_node, off.region)
        if k not in cheapest or off.spot_price < cheapest[k].spot_price:
            cheapest[k] = off
    offerings = sorted(cheapest.values(), key=lambda o: (o.cloud, o.gpu, o.region, -o.gpus_per_node))

    # Island shapes: exactly min_nodes per offering — larger islands only
    # lose (worse placement odds, bigger failure blast radius, multi-node
    # MFU discount), so they are dominated by more min-sized islands.
    rejections: list[Rejection] = []
    shape_notes: list[str] = []
    sized_all: list[tuple[Offering, int]] = []
    for off in offerings:
        if not supports_bf16(off.gpu):
            rejections.append(
                Rejection(_candidate_key(off, 1), f"{off.gpu} predates bf16 training")
            )
            continue
        nodes = min_nodes(weights, tuning, off.gpu_mem_gb, off.gpus_per_node, seq_len)
        if nodes is None:
            rejections.append(
                Rejection(_candidate_key(off, 1), f"model does not fit (≤8 nodes of {off.gpus_per_node}x{off.gpu})")
            )
            continue
        sized_all.append((off, nodes))

    # Placement-score asks are a scarce resource (AWS caps *distinct*
    # configurations per day), so shed shapes that cannot win before asking:
    # within a (gpu, region), a fatter-node island with fewer nodes strictly
    # dominates (better MFU, better placement odds, same GPUs); islands
    # pricier than the whole budget can never be planned; and the scores API
    # rejects the p3 family outright.
    best_shape: dict[tuple[str, str, str], tuple[Offering, int]] = {}
    for off, nodes in sized_all:
        k = (off.cloud, off.gpu, off.region)
        cur = best_shape.get(k)
        if cur is None or (nodes, -off.gpus_per_node) < (cur[1], -cur[0].gpus_per_node):
            best_shape[k] = (off, nodes)
    sized = []
    single_node_skips: dict[str, int] = {}
    for off, nodes in sized_all:
        key = _candidate_key(off, nodes)
        # Only AWS asks are scarce enough to prune before measuring; other
        # clouds' stock comes in one cheap call, so they are pruned after it
        # (below) — a sold-out fat node must not take its thinner,
        # in-stock siblings down with it.
        if off.cloud == "aws" and best_shape[(off.cloud, off.gpu, off.region)][0] is not off:
            rejections.append(Rejection(key, "dominated by a fatter-node island of the same GPU"))
        elif budget is not None and off.spot_price * nodes > budget - head_cost:
            rejections.append(Rejection(key, f"one island (${off.spot_price * nodes:.2f}/hr) exceeds the budget"))
        elif nodes > 1 and (why := multi_node_rejection(off.cloud, off.gpu, off.gpus_per_node)):
            rejections.append(Rejection(key, why))
            if why.startswith("multi-node islands unsupported"):
                single_node_skips[off.cloud] = single_node_skips.get(off.cloud, 0) + 1
        elif off.instance_type.startswith("p3."):
            rejections.append(Rejection(key, "placement scores unsupported for the p3 family"))
        else:
            sized.append((off, nodes))

    # Wave 1: quota limits + current usage. These APIs are effectively
    # unlimited, while placement-score *configurations* are capped per day —
    # so quota filtering runs first and scores are only ever requested for
    # shapes that could actually launch.
    quota_keys: dict[tuple[Offering, int], QuotaKey | None] = {}
    for off, nodes in sized:
        if off.cloud != "aws":
            continue  # RunPod has no quota system; stock gates it below
        code = aws.quota_code(off.instance_type, use_spot=True)
        quota_keys[(off, nodes)] = QuotaKey(off.region, code) if code else None
    unique_quotas = sorted({k for k in quota_keys.values() if k}, key=lambda k: (k.region, k.code))
    quotas: dict = {}
    usage: dict = {}
    if aws is not None and unique_quotas:
        with ThreadPoolExecutor(max_workers=2) as pool:
            quotas_f = pool.submit(aws.quotas, unique_quotas)
            usage_f = pool.submit(aws.quota_usage, unique_quotas)
            quotas = quotas_f.result()
            usage = usage_f.result()

    quota_limits: dict[tuple[str, str], float] = {}
    quota_ok: list[tuple[Offering, int]] = []
    for off, nodes in sized:
        key = _candidate_key(off, nodes)
        if off.cloud != "aws":
            quota_ok.append((off, nodes))
            continue
        qk = quota_keys[(off, nodes)]
        if qk is None:
            # Every AWS shape must land in a known quota bucket; treating an
            # unmapped type as unlimited once planned 16 P5 islands against a
            # 128-vCPU quota. (Uncapped buckets are reserved for on-prem.)
            rejections.append(Rejection(key, f"no quota mapping for {off.instance_type}"))
            continue
        limit = quotas.get(qk)
        if limit is None:
            rejections.append(Rejection(key, f"quota {qk.code}@{qk.region} unavailable"))
            continue
        used = usage.get(qk, 0.0)
        room = limit - used
        if room <= 0:
            rejections.append(
                Rejection(key, f"no spot quota room ({used:.0f}/{limit:.0f} vCPUs in use, {qk.code}@{qk.region})")
            )
            continue
        if off.vcpus * nodes > room:
            rejections.append(
                Rejection(
                    key,
                    f"one island needs {off.vcpus * nodes} vCPUs > remaining quota "
                    f"{room:.0f} ({used:.0f}/{limit:.0f} in use)",
                )
            )
            continue
        quota_limits[(qk.region, qk.code)] = room
        quota_ok.append((off, nodes))

    # Ask-budget pruning on the quota survivors: rank asks by the best
    # optimistic TFLOPs/$ any offering gives them, query the top
    # MAX_SCORE_ASKS, and reject the tail explicitly rather than silently
    # burning the daily config budget.
    unique_asks = sorted(
        {(off.instance_type, nodes) for off, nodes in quota_ok if off.cloud == "aws"}
    )
    if not skip_capacity_check and len(unique_asks) > MAX_SCORE_ASKS:
        def ask_value(ask: tuple[str, int]) -> float:
            itype, nodes = ask
            return max(
                effective_tflops(off, n, 10) / (off.spot_price * n)
                for off, n in quota_ok
                if off.instance_type == itype and n == nodes
            )

        ranked = sorted(unique_asks, key=ask_value, reverse=True)
        skipped = set(ranked[MAX_SCORE_ASKS:])
        unique_asks = sorted(ranked[:MAX_SCORE_ASKS])
        kept = []
        for off, nodes in quota_ok:
            if off.cloud == "aws" and (off.instance_type, nodes) in skipped:
                rejections.append(
                    Rejection(_candidate_key(off, nodes), "score not queried (daily config budget; low TFLOPs/$)")
                )
            else:
                kept.append((off, nodes))
        quota_ok = kept

    # Wave 2: capacity signals, spent only on launchable shapes — AWS
    # placement scores (region list stays the caller's full stable list so
    # cache keys and AWS's config identity do not churn) alongside every
    # other cloud's stock/capacity signal, which has no config budget but
    # the same gating semantics. One cloud's signal failing must not take
    # the others down: it degrades to "unavailable" for its own shapes.
    signal_asks = {
        name: sorted({(off.gpu, off.gpus_per_node, off.region) for off, _ in quota_ok if off.cloud == name})
        for name in signals
    }
    scores: dict = {}
    stock: dict[str, dict] = {name: {} for name in signals}

    def one_signal(name: str) -> dict:
        asks = signal_asks[name]
        if not asks:
            return {}
        try:
            return signals[name].scores(asks)
        except Exception as exc:  # noqa: BLE001 - degrade this cloud, keep planning
            signal_notes.append(
                f"{name} capacity signal failed ({exc}); its shapes are treated as score-unavailable"
            )
            return {ask: None for ask in asks}

    if not skip_capacity_check:
        with ThreadPoolExecutor(max_workers=1 + len(signals)) as pool:
            scores_f = (
                pool.submit(aws.placement_scores, unique_asks, regions)
                if aws is not None and unique_asks
                else None
            )
            stock_fs = {name: pool.submit(one_signal, name) for name in signals}
            scores = scores_f.result() if scores_f else {}
            stock = {name: f.result() for name, f in stock_fs.items()}

    candidates: list[Candidate] = []
    assumed_keys: list[str] = []
    stock_caps: dict[str, int | None] = {}
    for off, nodes in quota_ok:
        key = _candidate_key(off, nodes)
        if off.cloud == "aws":
            qk = quota_keys[(off, nodes)]
            score = scores.get((off.instance_type, nodes, off.region))
            kind = "placement"
        else:
            qk = None
            score = stock[off.cloud].get((off.gpu, off.gpus_per_node, off.region))
            kind = "stock"
            if score == 0:
                # Unlike an unfetchable score, sold-out is a measurement.
                rejections.append(Rejection(key, f"{off.cloud} stock: sold out at this GPU count"))
                continue
        assumed = False
        if min_score > 0 and not skip_capacity_check:
            if score is None:
                if strict_capacity_check:
                    rejections.append(Rejection(key, f"{kind} score unavailable (--strict-capacity-check)"))
                    continue
                assumed = True
                assumed_keys.append(key)
            elif score <= min_score:
                rejections.append(Rejection(key, f"{kind} score {score} ≤ {min_score}"))
                continue
        if off.cloud != "aws" and score is not None:
            stock_caps[key] = signals[off.cloud].island_cap(score)
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
                eff_tflops=effective_tflops(
                    off, nodes, ASSUMED_PLACEMENT_SCORE if assumed else score
                ),
                quota_bucket=(qk.region, qk.code) if qk else None,
                score=score,
                assumed=assumed,
                cloud=off.cloud,
            )
        )

    # Non-AWS dominance, among shapes that survived their stock signal.
    best_launchable: dict[tuple[str, str, str], Candidate] = {}
    for c in candidates:
        if c.cloud == "aws":
            continue
        k = (c.cloud, c.gpu, c.region)
        cur = best_launchable.get(k)
        if cur is None or (c.nodes, -c.gpus_per_node) < (cur.nodes, -cur.gpus_per_node):
            best_launchable[k] = c
    kept_candidates = []
    for c in candidates:
        if c.cloud != "aws" and best_launchable[(c.cloud, c.gpu, c.region)] is not c:
            rejections.append(Rejection(c.key, "dominated by a fatter-node island of the same GPU"))
            stock_caps.pop(c.key, None)
            continue
        kept_candidates.append(c)
    candidates = kept_candidates

    # Solve against margin-inflated prices, then verify the placement score
    # at each shape's aggregate planned capacity (obtainability decays with
    # size). A shape whose aggregate score is low or unknown is NOT dropped —
    # its single-island score is verified — it is capped at the largest count
    # whose capacity checked out, and the plan is re-solved (bounded loop).
    by_key = {c.key: c for c in candidates}
    # AWS shapes start verified at one island (their single-island score was
    # measured or assumed); RunPod shapes have no aggregate re-check concept,
    # so they start fully verified and are bounded by their stock caps.
    verified: dict[str, int] = {
        c.key: (1 if c.cloud == "aws" else max_islands) for c in candidates
    }
    caps: dict[str, int | None] = {c.key: stock_caps.get(c.key) for c in candidates}
    cap_notes: list[str] = []
    plan = Plan(counts={}, total_tflops=0.0, total_cost=head_cost, binding=["budget"])
    for _ in range(3):
        inflated = [
            replace(c, price_per_hour=c.price_per_hour * (1 + price_margin), max_count=caps[c.key])
            for c in candidates
        ]
        plan = solve(inflated, budget, quota_limits, max_islands, head_cost, target_tflops)
        if skip_capacity_check:
            break  # no scores were fetched; there is nothing to re-verify
        recheck = sorted(
            {
                (by_key[key].instance_type, n * by_key[key].nodes)
                for key, n in plan.counts.items()
                if n > verified[key] and caps[key] is None
            }
        )
        if not recheck:
            break
        regions_needed = sorted({by_key[key].region for key, n in plan.counts.items() if n > verified[key]})
        agg_scores = aws.placement_scores(recheck, regions_needed) if aws is not None else {}
        for key, n in plan.counts.items():
            c = by_key[key]
            if n <= verified[key] or caps[key] is not None:
                continue
            s = agg_scores.get((c.instance_type, n * c.nodes, c.region))
            if s is None and not strict_capacity_check:
                # Unfetchable at aggregate size: assume the best, say so.
                verified[key] = n
                cap_notes.append(
                    f"{key}: score unavailable at {n * c.nodes}-node aggregate "
                    f"capacity; assumed {ASSUMED_PLACEMENT_SCORE}"
                )
            elif s is not None and s > min_score:
                verified[key] = n
            else:
                caps[key] = verified[key]
                why = f"score {s}" if s is not None else "score unavailable"
                cap_notes.append(
                    f"{key} capped at {verified[key]} island(s): {why} at {n * c.nodes}-node aggregate capacity"
                )

    # Anything still unverified after the bounded loop gets pinned to its
    # verified capacity — a plan must never rely on an unchecked score.
    # (Not applicable when capacity checks were skipped wholesale.)
    leftover = (
        []
        if skip_capacity_check
        else [k for k, n in plan.counts.items() if n > verified[k] and caps[k] is None]
    )
    if leftover:
        for k in leftover:
            caps[k] = verified[k]
            cap_notes.append(f"{k} capped at {verified[k]} island(s): aggregate score not verified")
        inflated = [
            replace(c, price_per_hour=c.price_per_hour * (1 + price_margin), max_count=caps[c.key])
            for c in candidates
        ]
        plan = solve(inflated, budget, quota_limits, max_islands, head_cost, target_tflops)

    est_cost = head_cost + sum(by_key[k].price_per_hour * n for k, n in plan.counts.items())
    planned_mems = [
        next(o.gpu_mem_gb for o, _ in sized if o.instance_type == by_key[k].instance_type and o.region == by_key[k].region)
        for k in plan.counts
    ]
    shard = (
        "ddp"
        if planned_mems
        and all(fits(weights, tuning, m, 1, seq_len) for m in planned_mems)
        else "fsdp"
    )
    return ShapeResult(
        plan=plan,
        candidates=candidates,
        rejections=rejections,
        warnings=list(getattr(aws, "warnings", []))
        + [w for name in sorted(signals) for w in getattr(signals[name], "warnings", [])]
        + region_notes
        + gap_notes
        + shape_notes
        + signal_notes
        + [
            f"{cloud}: single-node islands only in this version; "
            f"{n} shape(s) needing more nodes skipped"
            for cloud, n in sorted(single_node_skips.items())
        ]
        + cap_notes
        + (
            [
                f"placement score unavailable for {len(assumed_keys)} shape(s) "
                f"(e.g. {assumed_keys[0]}); assumed {ASSUMED_PLACEMENT_SCORE} — "
                "pass --strict-capacity-check to reject them instead"
            ]
            if assumed_keys
            else []
        )
        + (
            ["capacity checks skipped: plan is not verified against spot obtainability"]
            if skip_capacity_check
            else []
        ),
        weights_gb=weights,
        shard=shard,
        est_cost=est_cost,
        price_margin=price_margin,
        head_cost=head_cost,
        fetch_seconds=time.monotonic() - t0,
    )


def launch_argv(result: ShapeResult, model: str, tuning: str, data: str) -> list[str]:
    """The `yeto launch` argv realizing the plan (shared by render and --apply
    so what is printed and what runs cannot drift)."""
    entries: list[str] = []
    for key, n in sorted(result.plan.counts.items()):
        entries.extend([key] * n)
    # Learner disk must hold the HF weight cache plus headroom; the 512 GB
    # launch default silently underfits big models.
    disk_gb = max(512, int(result.weights_gb * 1.5) + 100)
    argv = [
        "launch",
        "--gpu", ",".join(entries),
        "--model", model,
        "--tuning", tuning,
        "--shard", result.shard,
        "--disk-size", str(disk_gb),
        "--data", data,
    ]
    return argv


def to_json_dict(result: ShapeResult, model: str, budget: float, tuning: str, data: str | None) -> dict:
    """JSON-safe summary for --json / programmatic consumers."""
    by_key = {c.key: c for c in result.candidates}
    return {
        "model": model,
        "budget_per_hour": budget,
        "weights_gb_bf16": result.weights_gb,
        "shard": result.shard,
        "islands": [
            {
                "key": key,
                "count": n,
                "cloud": by_key[key].cloud,
                "region": by_key[key].region,
                "instance_type": by_key[key].instance_type,
                "nodes": by_key[key].nodes,
                "score": by_key[key].score,
                "score_assumed": by_key[key].assumed,
                "est_price_per_hour": by_key[key].price_per_hour,
                "eff_tflops": by_key[key].eff_tflops,
            }
            for key, n in sorted(result.plan.counts.items())
        ],
        "total_eff_tflops": result.plan.total_tflops,
        "est_cost_per_hour": result.est_cost,
        "budget_cost_with_margin": result.plan.total_cost,
        "price_margin": result.price_margin,
        "head_cost_per_hour": result.head_cost,
        "binding": result.plan.binding,
        "rejections": sorted({f"{r.key}: {r.reason}" for r in result.rejections}),
        "warnings": result.warnings,
        "launch_argv": (["yeto"] + launch_argv(result, model, tuning, data or "<hf-dataset>"))
        if result.plan.counts
        else None,
    }


def render(
    result: ShapeResult,
    model: str,
    budget: float | None,
    tuning: str,
    data: str | None = None,
    target_tflops: float | None = None,
) -> str:
    """Human-readable plan + the ready-to-run launch line."""
    plan, out = result.plan, []
    by_key = {c.key: c for c in result.candidates}
    objective = (
        f"budget ${budget:.2f}/hr" if budget is not None else f"target ≥ {target_tflops:.0f} TFLOPs"
    )
    if budget is not None and target_tflops is not None:
        objective = f"target ≥ {target_tflops:.0f} TFLOPs within ${budget:.2f}/hr"
    if not plan.counts:
        out.append(f"no feasible plan for {objective} (weights ~{result.weights_gb:.0f} GB bf16, {tuning})")
    else:
        islands = sum(plan.counts.values())
        out.append(
            f"plan: {islands} island(s), {plan.total_tflops:.1f} effective TFLOPs, "
            f"est ${result.est_cost:.2f}/hr — ≤ ${plan.total_cost:.2f}/hr with "
            f"{result.price_margin:.0%} spot-price margin ({objective})"
        )
        for key, n in sorted(plan.counts.items()):
            c = by_key[key]
            if c.assumed:
                shown = f"~{ASSUMED_PLACEMENT_SCORE} (assumed)"
            elif c.cloud != "aws":
                shown = f"stock≈{c.score}"
            else:
                shown = str(c.score)
            out.append(
                f"  {n}x {key}  spot est ${c.price_per_hour:.2f}/hr/island  "
                f"score {shown}  {c.eff_tflops:.1f} TFLOPs/island"
            )
        out.append(f"  head: on-demand CPU VM  ${result.head_cost:.2f}/hr")
    if plan.binding:
        out.append(f"binding constraints: {', '.join(plan.binding)}")
    for w in result.warnings:
        out.append(f"warning: {w}")
    if result.rejections:
        out.append("rejected:")
        for line in sorted({f"  {r.key}: {r.reason}" for r in result.rejections}):
            out.append(line)
    if plan.counts:
        argv = launch_argv(result, model, tuning, data or "<hf-dataset>")
        out.append("launch: yeto " + " ".join(argv))
    out.append(f"(signals fetched in {result.fetch_seconds:.1f}s; cached for 1h)")
    return "\n".join(out)
