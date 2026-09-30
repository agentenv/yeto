"""Trainer DP resharding checks (rl-infra-spec 4.6 spike, alignment A4, design D5/D7). Import-light.

A DP change keeps TP/PP/CP/EP, the global batch size, the micro-batch size and
the algorithm fixed and changes only the data-parallel size of the single
trainer cell. The cut (4.2) is keyed by parameter name (fork-M5 format), so the
new ranks gather every DP shard of their (tp, pp) and slice out the range they
own now; nothing here moves tensors.

Everything a DP edge needs to be refused for is refused by
:func:`reshard_problems` BEFORE any file is written (the transition calls it
before ``save_cut``) and again before any rank writes on restore:

* layout: world must be ``tp*pp*cp*dp``; only DP (and so world) may change;
* batch semantics (D7.4): GBS divisible by ``dp * micro_batch_size`` in BOTH
  layouts (Miles computes ``global_batch_size // dp_size``: a floor division
  would silently drop samples); dynamic batch sizes are not certified here;
* loss normalization (A4): only the default sample-mean aggregation, for which
  Miles scales every micro-batch loss by ``num_microbatches / GBS * dp`` and
  Megatron averages over micro-batches and DP, i.e. every sample weighs
  ``1/GBS`` at any DP (:func:`sample_weight`). Token-level aggregation
  (``--calculate-per-token-loss``), the Dr.GRPO constant denominator,
  ``--normalize-advantages`` (DP all-reduce whitening), custom reducers/losses,
  sequence-level variants (GSPO/GMPO) and multi-LoRA are refused until
  certified on their own spec;
* certification: the edge is valid only for the ``algorithm_spec_sha256``
  values listed by the attestation (first round: default GRPO only);
* the cut plugin's own refusals (fp16, precision-aware optimizer, partial
  DistOpt instances, CP>1, EP>1, TP/PP>1 with DistOpt masters).

RNG (4.2a ``keep_on_dp_change``): when DP changes, a new rank keeps the fresh
RNG its process was built with; :func:`rng_mapping` records, per new rank,
where its RNG came from and how the fresh seed is derived (Megatron
``_set_random_seed``: ``seed + 100*pp_rank`` [+ ``10*dp_rank`` with
``--data-parallel-random-init``], CUDA tracker offset ``2718 + tp_rank``). A
DP change without a recorded mapping is refused.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

FIXED_DIMS = ("tp", "pp", "cp", "ep")
RNG_POLICIES = ("exact", "keep_on_dp_change")
# Mechanisms whose loss/advantage normalization depends on the DP/CP layout (A4).
UNCERTIFIED_VARIANTS = frozenset({"gspo", "gmpo", "cispo", "sapo", "sapo_qwen"})


class ReshardRefused(ValueError):
    """The DP edge is not supported; nothing has been written."""


@dataclass(frozen=True)
class ReshardPlan:
    source: Mapping[str, int]  # trainer_layout(): world/tp/pp/cp/ep/dp
    target: Mapping[str, int]
    global_batch_size: int
    micro_batch_size: int
    rng_policy: str = "keep_on_dp_change"

    @property
    def dp_changes(self) -> bool:
        return int(self.source["dp"]) != int(self.target["dp"])

    def to_dict(self) -> dict[str, Any]:
        return {"source": dict(self.source), "target": dict(self.target),
                "global_batch_size": self.global_batch_size, "micro_batch_size": self.micro_batch_size,
                "rng_policy": self.rng_policy}


def layout_problems(source: Mapping[str, int], target: Mapping[str, int]) -> list[str]:
    out = []
    for name, lay in (("source", source), ("target", target)):
        try:
            tp, pp, cp, dp, world = (int(lay[k]) for k in ("tp", "pp", "cp", "dp", "world"))
        except (KeyError, TypeError, ValueError):
            out.append(f"{name} layout incomplete: {dict(lay)}")
            continue
        if min(tp, pp, cp, dp) < 1 or tp * pp * cp * dp != world:
            out.append(f"{name} layout {dict(lay)}: world != tp*pp*cp*dp")
    if out:
        return out
    for dim in FIXED_DIMS:
        if int(source.get(dim, 1)) != int(target.get(dim, 1)):
            out.append(f"{dim} changes {source.get(dim)} -> {target.get(dim)} (only DP may change)")
    return out


def batch_problems(plan: ReshardPlan, args: Any = None) -> list[str]:
    out = []
    gbs, mbs = int(plan.global_batch_size), int(plan.micro_batch_size)
    if gbs <= 0 or mbs <= 0:
        return [f"global/micro batch size must be positive (gbs={gbs}, mbs={mbs})"]
    for name, lay in (("source", plan.source), ("target", plan.target)):
        dp = int(lay["dp"])
        if gbs % (dp * mbs):
            out.append(f"{name}: global batch {gbs} is not divisible by dp {dp} x micro batch {mbs} "
                       "(Miles floor-divides and would drop samples)")
    if args is not None:
        if int(getattr(args, "global_batch_size", gbs) or gbs) != gbs:
            out.append("args.global_batch_size differs from the plan")
        if int(getattr(args, "micro_batch_size", mbs) or mbs) != mbs:
            out.append("args.micro_batch_size differs from the plan")
        for flag in ("use_dynamic_batch_size", "use_dynamic_global_batch_size"):
            if getattr(args, flag, False):
                out.append(f"--{flag.replace('_', '-')} is not certified for a DP change")
        if getattr(args, "calculate_per_token_loss", False):
            out.append("--calculate-per-token-loss (token-level normalization) is not certified (A4)")
        if getattr(args, "normalize_advantages", False):
            out.append("--normalize-advantages whitens within the DP group (A4)")
        if getattr(args, "multi_lora", False) or int(getattr(args, "multi_lora_n_adapters", 0) or 0) > 1:
            out.append("multi-LoRA is not certified for a DP change")
    return out


def algorithm_problems(spec: Any) -> list[str]:
    """Loss/advantage normalization mechanisms that A4 requires to be certified separately."""
    if spec is None:
        return ["no algorithm spec: a DP edge is bound to an algorithm_spec_sha256 (A4)"]
    out = []
    loss = getattr(spec, "loss", None)
    advantage = getattr(spec, "advantage", None)
    aggregation = getattr(loss, "aggregation", "default")
    if aggregation != "default":
        out.append(f"loss.aggregation={aggregation!r} (token-level / constant denominator) is not certified")
    if getattr(loss, "reducer", None) is not None or getattr(loss, "custom_loss", None) is not None:
        out.append("custom loss/reducer plugin: its normalization is not certified")
    variant = getattr(loss, "policy_loss_variant", None)
    if variant in UNCERTIFIED_VARIANTS:
        out.append(f"policy loss variant {variant!r} (sequence-level quantities) is not certified")
    if getattr(advantage, "whiten", False):
        out.append("advantage.whiten (--normalize-advantages, DP all-reduce) is not certified")
    estimator = getattr(advantage, "estimator", None) or getattr(spec, "advantage_estimator", None)
    if estimator != "grpo":
        out.append(f"advantage estimator {estimator!r}: first round certifies default GRPO only")
    return out


def certification_problems(spec_sha256: str | None, certified: Iterable[str]) -> list[str]:
    certified = frozenset(certified)
    if not spec_sha256:
        return ["algorithm_spec_sha256 unknown"]
    if spec_sha256 not in certified:
        return [f"algorithm_spec_sha256 {spec_sha256[:12]}... is not certified for this DP edge"]
    return []


def reshard_problems(
    plan: ReshardPlan,
    *,
    args: Any = None,
    spec: Any = None,
    spec_sha256: str | None = None,
    certified: Iterable[str] | None = None,
) -> list[str]:
    """Everything that refuses the edge; empty means it may proceed. Pure, writes nothing."""
    from .cut_plugin import config_problems  # import-light (torch/miles stay lazy there)

    out = layout_problems(plan.source, plan.target)
    if plan.rng_policy not in RNG_POLICIES:
        out.append(f"unknown RNG policy {plan.rng_policy!r}")
    elif plan.dp_changes and plan.rng_policy == "exact":
        out.append("RNG policy 'exact' cannot change DP")
    if out:
        return out
    out += batch_problems(plan, args)
    out += loss_normalization_problems(plan)
    if args is not None:
        out += config_problems(args, {"cp_size": plan.target["cp"], "ep_size": plan.target["ep"]})
    out += algorithm_problems(spec)
    if certified is not None:
        sha = spec_sha256 or (spec.sha256() if spec is not None and callable(getattr(spec, "sha256", None)) else None)
        out += certification_problems(sha, certified)
    return out


def require_reshard(plan: ReshardPlan, **kwargs: Any) -> None:
    problems = reshard_problems(plan, **kwargs)
    if problems:
        raise ReshardRefused("DP edge refused: " + "; ".join(problems))


# --------------------------------------------------------------------------
# Loss normalization and sample mapping (D7.4)
# --------------------------------------------------------------------------


def sample_weight(*, global_batch_size: int, micro_batch_size: int, dp: int) -> Fraction:
    """Gradient weight of one sample under Miles' default sample-mean loss.

    Miles ``loss_function``: ``loss * num_microbatches / GBS * dp`` per
    micro-batch (sum of per-sample means); Megatron divides the accumulated
    loss by ``num_microbatches`` and the DP gradient reduction averages over
    ``dp`` ranks. ``num_microbatches = GBS / (dp * mbs)``.
    """
    nmb = Fraction(global_batch_size, dp * micro_batch_size)
    if nmb.denominator != 1:
        raise ReshardRefused(f"GBS {global_batch_size} is not divisible by dp {dp} x mbs {micro_batch_size}")
    return nmb / global_batch_size * dp / nmb / dp


def loss_normalization_problems(plan: ReshardPlan) -> list[str]:
    gbs, mbs = plan.global_batch_size, plan.micro_batch_size
    try:
        before = sample_weight(global_batch_size=gbs, micro_batch_size=mbs, dp=int(plan.source["dp"]))
        after = sample_weight(global_batch_size=gbs, micro_batch_size=mbs, dp=int(plan.target["dp"]))
    except ReshardRefused as exc:
        return [str(exc)]
    if before != after or before != Fraction(1, gbs):
        return [f"per-sample loss weight changes {before} -> {after} (expected 1/{gbs})"]
    return []


def dp_partitions(num_samples: int, dp: int) -> list[list[int]]:
    """Miles ``split_train_data_by_dp_raw`` without ``--balance-data``: round-robin."""
    return [list(range(i, num_samples, dp)) for i in range(dp)]


def sample_mapping(num_samples: int, source_dp: int, target_dp: int,
                   partition=dp_partitions) -> dict[str, Any]:
    """Which rank consumes which sample before/after; the consumed SET must be identical."""
    before, after = partition(num_samples, source_dp), partition(num_samples, target_dp)
    flat_before = sorted(i for p in before for i in p)
    flat_after = sorted(i for p in after for i in p)
    problems = []
    for name, flat in (("source", flat_before), ("target", flat_after)):
        if flat != list(range(num_samples)):
            problems.append(f"{name} partition does not consume every sample exactly once")
    return {"source": before, "target": after, "problems": problems}


# --------------------------------------------------------------------------
# RNG mapping (4.2a keep_on_dp_change; design D5 "explainable seed mapping")
# --------------------------------------------------------------------------


def fresh_seed(args: Any, coord: Mapping[str, int]) -> dict[str, int]:
    """Megatron ``_set_random_seed`` derivation for a freshly built rank."""
    seed = int(getattr(args, "seed", 1234) or 1234)
    derived = seed + 100 * int(coord.get("pp", 0))
    if getattr(args, "data_parallel_random_init", False):
        derived += 10 * int(coord.get("dp", 0))
    return {"base_seed": seed, "derived_seed": derived,
            "cuda_tracker_seed": derived + 2718 + int(coord.get("tp", 0))}


def rng_mapping(plan: ReshardPlan, target_coords: Iterable[Mapping[str, int]], args: Any) -> list[dict[str, Any]]:
    """One entry per new rank: RNG source ('restored' from the same coordinate or 'fresh') and derivation."""
    out = []
    for coord in sorted(target_coords, key=lambda c: (c.get("pp", 0), c.get("tp", 0), c.get("dp", 0))):
        key = {k: int(coord[k]) for k in ("tp", "pp", "dp")}
        if not plan.dp_changes:
            out.append({"coord": key, "source": "restored", "from": key})
        elif plan.rng_policy == "keep_on_dp_change":
            out.append({"coord": key, "source": "fresh", "seed": fresh_seed(args, coord)})
        else:
            raise ReshardRefused("RNG policy 'exact' cannot change DP")
    return out
