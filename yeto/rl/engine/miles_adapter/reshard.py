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
* batch semantics (D7.4): the production path is the rollout-side schedule
  (fork ``dp_schedule.build_dp_schedule``: GBS counts rollouts, pack first,
  distribute micro-batch k to rank k % dp), modelled by
  :func:`scheduled_partitions`; GBS/mbs must be a multiple of dp in BOTH
  layouts; dynamic batch, partial steps, --balance-data/--balance-by-flops
  and vpp>1 are refused (their packing/assignment depends on dp);
* dropout: with a DP change every dropout must be 0 (fresh RNG on new ranks);
* loss normalization (A4): only the default sample-mean aggregation. The
  fork scales a micro-batch by ``num_microbatches / num_rollouts * dp``
  (:func:`miles_microbatch_loss_scale`); with Megatron's ``1/num_microbatches``
  and the DP average the weight is ``1/num_rollouts`` at any DP. CPU only
  recomputes this arithmetic; the evidence that the engine does it is A8 G2.
  Token-level aggregation
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


DROPOUT_ARGS = ("lora_dropout", "hidden_dropout", "attention_dropout")


def batch_problems(plan: ReshardPlan, args: Any = None) -> list[str]:
    """Batch-semantics refusals. GBS counts ROLLOUTS (fork ``build_dp_schedule``).

    Static precondition ASSUMING one sample per rollout (the GRPO ports path):
    the per-step micro-batch count ``GBS / mbs`` must be a multiple of dp in
    both layouts (refused here before any write). Because that assumption is
    not guaranteed, every production batch after a DP change is checked again
    by :func:`batch_guard_problems` (``MilesTrainerGroup.train_step``) before
    the engine trains on it.
    """
    out = []
    gbs, mbs = int(plan.global_batch_size), int(plan.micro_batch_size)
    if gbs <= 0 or mbs <= 0:
        return [f"global/micro batch size must be positive (gbs={gbs}, mbs={mbs})"]
    for name, lay in (("source", plan.source), ("target", plan.target)):
        dp = int(lay["dp"])
        if gbs % (dp * mbs):
            out.append(f"{name}: {gbs} rollouts/step are not divisible by dp {dp} x micro batch {mbs} "
                       "(fork static schedule asserts step_size % (dp*mbs) == 0)")
    if args is not None:
        if int(getattr(args, "global_batch_size", gbs) or gbs) != gbs:
            out.append("args.global_batch_size differs from the plan")
        if int(getattr(args, "micro_batch_size", mbs) or mbs) != mbs:
            out.append("args.micro_batch_size differs from the plan")
        for flag in ("use_dynamic_batch_size", "use_dynamic_global_batch_size", "allow_partial_train_step",
                     "balance_data", "balance_by_flops"):
            if getattr(args, flag, False):
                out.append(f"--{flag.replace('_', '-')} is not certified for a DP change "
                           "(packing/assignment depends on dp)")
        if int(getattr(args, "virtual_pipeline_model_parallel_size", 1) or 1) > 1:
            out.append("virtual pipeline (vpp>1) changes the DP alignment; not certified")
        if getattr(args, "indep_dp", False):
            out.append("--indep-dp: the trainer advertises no train_parallel_config and the fork "
                       "falls back to the unscheduled split")
        if getattr(args, "multimodal_keys", None):
            out.append("--multimodal-keys: multimodal batches take the fork's unscheduled split")
        if getattr(args, "calculate_per_token_loss", False):
            out.append("--calculate-per-token-loss (token-level normalization) is not certified (A4)")
        if getattr(args, "normalize_advantages", False):
            out.append("--normalize-advantages whitens within the DP group (A4)")
        if getattr(args, "multi_lora", False) or int(getattr(args, "multi_lora_n_adapters", 0) or 0) > 1:
            out.append("multi-LoRA is not certified for a DP change")
        if plan.dp_changes:
            # New ranks keep fresh RNG (keep_on_dp_change): any dropout makes the next step
            # depend on RNG that the cut does not carry. Missing = unknown (Megatron's
            # hidden/attention dropout default is 0.1) -> refused.
            for name in DROPOUT_ARGS:
                value = getattr(args, name, None)
                if value is None or float(value) != 0.0:
                    out.append(f"{name}={value!r}: a DP change is certified only with every dropout = 0")
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
        out += config_problems(args, {"cp_size": plan.target["cp"], "ep_size": plan.target["ep"]}, reshard=True)
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


def miles_microbatch_loss_scale(*, num_microbatches: int, num_rollouts: int, dp: int, cp: int = 1) -> Fraction:
    """Factor fork ``training_utils/loss.py:loss_function`` multiplies a micro-batch's summed sample means by.

    ``apply_megatron_loss_scaling`` branch, not per-token:
    ``loss * num_microbatches / loss_normalizer * loss_parallel_size`` with
    ``loss_normalizer = num_rollouts`` and ``loss_parallel_size = intra_dp_cp.size``.
    """
    return Fraction(num_microbatches, num_rollouts) * (dp * cp)


def sample_weight(*, num_rollouts: int, num_microbatches: int, dp: int, cp: int = 1) -> Fraction:
    """Gradient weight of one sample mean: Miles scale, then Megatron's ``1/num_microbatches``
    (forward_backward_func) and the DP(xCP) gradient average ``1/(dp*cp)``.

    The Megatron factors are an assumption about Megatron core (not re-read
    here); CPU only checks the arithmetic -- GPU evidence is A8 G2.
    """
    scale = miles_microbatch_loss_scale(num_microbatches=num_microbatches, num_rollouts=num_rollouts, dp=dp, cp=cp)
    return scale / num_microbatches / (dp * cp)


def loss_normalization_problems(plan: ReshardPlan) -> list[str]:
    """Recompute the per-sample weight on both layouts with the fork's formula (one sample per rollout)."""
    gbs, mbs = plan.global_batch_size, plan.micro_batch_size
    weights = {}
    for name, lay in (("source", plan.source), ("target", plan.target)):
        dp, cp = int(lay["dp"]), int(lay.get("cp", 1))
        if gbs % (dp * mbs):
            return []  # reported by batch_problems
        nmb = gbs // mbs // dp
        weights[name] = sample_weight(num_rollouts=gbs, num_microbatches=nmb, dp=dp, cp=cp)
    if weights["source"] != weights["target"]:
        return [f"per-sample loss weight changes {weights['source']} -> {weights['target']}"]
    return []


def scheduled_partitions(rollout_indices: list[int], *, dp: int, global_batch_size: int,
                         micro_batch_size: int) -> dict[str, Any]:
    """Model of fork ``dp_schedule.build_dp_schedule`` on the certified path
    (static micro-batches, no --balance-data/--balance-by-flops, vpp=1).

    Rollouts are taken in first-appearance order, ``global_batch_size`` per
    step (trailing rollouts dropped -- independent of dp); a step's samples
    are chunked into micro-batches of ``micro_batch_size`` in order; micro-batch
    k goes to rank ``k % dp``. Returns per-rank sample positions, per-step
    micro-batch composition, ``num_microbatches`` and ``num_rollouts``.
    """
    by_rollout: dict[int, list[int]] = {}
    for pos, rid in enumerate(rollout_indices):
        by_rollout.setdefault(rid, []).append(pos)
    rids = list(by_rollout)
    steps = len(rids) // global_batch_size
    if steps < 1:
        raise ReshardRefused(f"{len(rids)} rollouts < global batch {global_batch_size}")
    partitions: list[list[int]] = [[] for _ in range(dp)]
    step_micro_batches, num_microbatches = [], []
    for step in range(steps):
        picked = rids[step * global_batch_size:(step + 1) * global_batch_size]
        samples = [p for rid in picked for p in by_rollout[rid]]
        mbs = [samples[i:i + micro_batch_size] for i in range(0, len(samples), micro_batch_size)]
        if len(mbs) % dp:
            raise ReshardRefused(f"step {step}: {len(mbs)} micro-batches not a multiple of dp {dp}")
        step_micro_batches.append(mbs)
        num_microbatches.append(len(mbs) // dp)
        for k, mb in enumerate(mbs):
            partitions[k % dp].extend(mb)
    return {"partitions": partitions, "micro_batches": step_micro_batches,
            "num_microbatches": num_microbatches, "num_rollouts": [global_batch_size] * steps}


def sample_mapping(rollout_indices: list[int], *, source_dp: int, target_dp: int, global_batch_size: int,
                   micro_batch_size: int) -> dict[str, Any]:
    """Before/after assignment; per step the consumed sample set, the micro-batch composition and
    ``num_rollouts`` must be identical (only the rank assignment may differ)."""
    before = scheduled_partitions(rollout_indices, dp=source_dp, global_batch_size=global_batch_size,
                                  micro_batch_size=micro_batch_size)
    after = scheduled_partitions(rollout_indices, dp=target_dp, global_batch_size=global_batch_size,
                                 micro_batch_size=micro_batch_size)
    problems = []
    if before["micro_batches"] != after["micro_batches"]:
        problems.append("micro-batch composition differs between the layouts")
    if before["num_rollouts"] != after["num_rollouts"]:
        problems.append("num_rollouts (loss normalizer) differs between the layouts")
    for name, sched in (("source", before), ("target", after)):
        flat = sorted(p for part in sched["partitions"] for p in part)
        want = sorted(p for step in sched["micro_batches"] for mb in step for p in mb)
        if flat != want or len(set(flat)) != len(flat):
            problems.append(f"{name}: ranks do not consume every scheduled sample exactly once")
    return {"source": before, "target": after, "problems": problems}


def step_problems(rollout_indices: list[int], plan: ReshardPlan) -> list[str]:
    """Per-batch check (harness / A8): the scheduled path must accept the batch on BOTH layouts."""
    try:
        return sample_mapping(rollout_indices, source_dp=int(plan.source["dp"]), target_dp=int(plan.target["dp"]),
                              global_batch_size=plan.global_batch_size,
                              micro_batch_size=plan.micro_batch_size)["problems"]
    except ReshardRefused as exc:
        return [str(exc)]


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


SCHEDULE_CONFIG_KEYS = ("dp_size", "cp_size", "vpp_size", "microbatch_group_size_per_vp_stage")


def batch_guard_problems(plan: ReshardPlan, *, rank_configs: list[Mapping[str, Any]],
                         rollout_indices: list[int], steps: int) -> list[str]:
    """Per-batch production guard after a DP change (review M2/L1).

    The fork silently falls back to ``split_train_data_by_dp_raw`` (floor
    division, round-robin, no ``num_rollouts``) when the trainer advertises an
    incomplete ``train_parallel_config`` (indep-DP), for multimodal batches,
    without ``rollout_ids`` or with fewer distinct rollouts than GBS
    (``train_data_conversion.can_schedule_on_rollout_side``). Here: every rank
    must advertise the full schedule config with the planned dp, and the batch
    (``rollout_indices`` per sample) must hold ``steps * GBS`` rollouts that the
    scheduled path accepts on both layouts. Multimodal is refused statically
    (``--multimodal-keys``); ``rollout_ids`` are always set by the fork's
    ``_convert_samples_to_train_data`` (``rollout_id or index``).
    """
    out = []
    dp = int(plan.target["dp"])
    if not rank_configs:
        out.append("no rank reported its train_parallel_config")
    for i, cfg in enumerate(rank_configs):
        missing = [k for k in SCHEDULE_CONFIG_KEYS if k not in cfg]
        if missing:
            out.append(f"rank {i}: train_parallel_config lacks {missing} (fork would use the unscheduled split)")
        elif int(cfg["dp_size"]) != dp:
            out.append(f"rank {i}: advertises dp_size {cfg['dp_size']}, planned {dp}")
    distinct = len(dict.fromkeys(rollout_indices))
    if distinct < plan.global_batch_size:
        out.append(f"{distinct} rollouts < GBS {plan.global_batch_size} (fork would use the unscheduled split)")
    elif distinct != steps * plan.global_batch_size:
        out.append(f"{distinct} rollouts != {steps} steps x GBS {plan.global_batch_size} (trailing rollouts dropped)")
    else:
        out += step_problems(list(rollout_indices), plan)
    return out


# --------------------------------------------------------------------------
# Argv-level profile (identical check locally and in the container)
# --------------------------------------------------------------------------

# Upstream parse defaults of the fields the checks above read (fork 5c1b49eb:
# Megatron hidden/attention dropout 0.1, Miles micro batch 1, store_true flags False).
ARGV_DEFAULTS: dict[str, Any] = {
    "hidden_dropout": 0.1, "attention_dropout": 0.1, "lora_dropout": 0.0, "micro_batch_size": 1,
    "global_batch_size": None, "virtual_pipeline_model_parallel_size": None, "multimodal_keys": None,
    "multi_lora_n_adapters": 0, "num_distributed_optimizer_instances": 1,
    "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "context_parallel_size": 1,
    "expert_model_parallel_size": 1,
}
ARGV_FLAGS = ("balance_data", "balance_by_flops", "use_dynamic_batch_size", "use_dynamic_global_batch_size",
              "allow_partial_train_step", "calculate_per_token_loss", "normalize_advantages", "multi_lora",
              "indep_dp", "fp16", "bf16", "use_precision_aware_optimizer", "use_distributed_optimizer")


def argv_profile(argv: Iterable[str], overrides: Mapping[str, Any] | None = None) -> Any:
    """The reshard-relevant Miles args as the argv sets them (defaults as upstream parse), plus overrides."""
    from types import SimpleNamespace

    values: dict[str, Any] = dict(ARGV_DEFAULTS)
    values.update({flag: False for flag in ARGV_FLAGS})
    tokens = list(argv)
    for i, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        name, _, inline = token[2:].partition("=")
        key = name.replace("-", "_")
        if key.startswith("no_") and key[3:] in ARGV_FLAGS:
            values[key[3:]] = False
        elif key in ARGV_FLAGS:
            values[key] = True
        elif key in ARGV_DEFAULTS:
            raw = inline or (tokens[i + 1] if i + 1 < len(tokens) else None)
            try:
                values[key] = float(raw) if "dropout" in key else int(raw)
            except (TypeError, ValueError):
                values[key] = raw
    values.update(dict(overrides or {}))
    return SimpleNamespace(**values)


def argv_reshard_problems(argv: Iterable[str], algorithm: Any, overrides: Mapping[str, Any] | None = None,
                          dps: tuple[tuple[int, int], ...] = ((1, 2), (2, 1))) -> dict[str, list[str]]:
    """``reshard_problems`` for each DP edge of the harness, from the Miles argv alone."""
    args = argv_profile(argv, overrides)
    gbs = int(args.global_batch_size or 0)
    mbs = int(args.micro_batch_size or 1)
    lay = lambda dp: {"world": dp, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": dp}  # noqa: E731
    return {f"{s}->{d}": reshard_problems(ReshardPlan(lay(s), lay(d), gbs, mbs), args=args, spec=algorithm)
            for s, d in dps}
