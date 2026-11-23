"""``yeto model check``: can this model train in this form, and how (no weights, no GPU).

One request (trainable form, sync preset, backend, parallel shape) against one
``config.json`` gives a :class:`CheckReport`:

* items, each ``ok`` / ``warn`` / ``fail`` with its reason, a suggestion, and
  its source: ``launcher`` (``yeto launch`` refuses it today), ``island`` (the
  island or its run config refuses it at start), ``upstream`` (the pinned
  Megatron / Megatron-Bridge / Miles stack cannot run it) or ``estimate``;
* recommended DiLoCo settings (``P`` fragments, ``H`` local rounds per sync,
  ``tau`` rounds in flight, fragment bytes) and a GPU-memory estimate
  (:mod:`yeto.memory_estimate`, the launch preflight's model, and the model
  registry's trainer plan);
* the ``yeto launch`` flags of the cell (``launch_argv``) and the recipe or
  mapping templates a new model needs.

The ``launcher`` items mirror ``yeto/launch/rl_args.py`` and
``yeto/rl/engine/selection.py`` on main; ``tests/test_model_check_matrix.py``
runs the launcher on ``launch_argv`` for every fixture model, form, preset and
backend and fails when the two disagree. ``PENDING`` names the cells that open
work changes ("not on main yet"); the matrix test fails when a key no longer
reaches a failing cell. Import-light: stdlib and the model registry only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from . import upstream
from .sync_profile import (
    EXPERT_LORA_SUPPORTED,
    LAYOUT_OFFLINE,
    ModelSyncProfile,
    _derive,
)

FORMS = ("lora", "full", "partial-moe")
PRESETS = ("strict-avg", "decoupled", "full-parameter", "single-island")
BACKENDS = ("miles", "verl")
OK, WARN, FAIL = "ok", "warn", "fail"
LAUNCHER, ISLAND, UPSTREAM, ESTIMATE = "launcher", "island", "upstream", "estimate"
MAX_FRAGMENT_BYTES = 2 << 30  # --rl-sync-fragment-max-bytes default (yeto/launch/rl_args.py defaults)
DEFAULT_P, DEFAULT_TAU, DEFAULT_H = 4, 2, 4  # decoupled runs A/B/C (P=4, tau=2, H=4); H>=4: outer Nesterov
PARTIAL_RECIPES = {"kimi_k31", "deepseek-v4-flash", "generic"}
DSV4_EXPERT_FULL_COUNT = 4  # --expert-full-count of the deepseek-v4-flash expert-full form
# Failing cells that open work turns into runs (printed as "not on main yet"). Remove an
# entry in the change that flips its cell: tests/test_model_check_matrix.py requires every
# key to still reach a failing cell. Open work per key, for maintainers: decoupled/kimi_k31-lora:
# K3.1 offline LoRA specs (the generic ones are on main).
PENDING = {
    "decoupled/kimi_k31-lora": "kimi_k31 LoRA under decoupled, from an offline canonical LoRA layout",
}


@dataclass(frozen=True)
class CheckRequest:
    form: str = "lora"
    preset: str = "decoupled"
    backend: str = "miles"
    recipe: str | None = None  # None: the recipe of the model's registry family
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int | None = None  # None: recommended
    lora_targets: str = "attention"
    lora_rank: int = 16
    islands: int = 2
    gpus_per_island: int = 8
    card: str = "H100"
    seq_len: int = 4096


@dataclass(frozen=True)
class CheckItem:
    name: str
    status: str
    reason: str
    suggestion: str = ""
    source: str = LAUNCHER
    pending: str = ""  # a PENDING key: open work turns this failing cell into a run


@dataclass
class CheckReport:
    model: str
    request: CheckRequest
    recipe: str
    profile: ModelSyncProfile
    items: list[CheckItem]
    recommended: dict[str, Any] = field(default_factory=dict)
    templates: list[str] = field(default_factory=list)
    launch_argv: list[str] = field(default_factory=list)

    @property
    def pending(self) -> list[str]:
        """PENDING keys of the failing items (open work that turns them into runs)."""

        return sorted({i.pending for i in self.items if i.pending and i.status == FAIL})

    def _ok(self, *sources: str) -> bool:
        return not any(i.status == FAIL and i.source in sources for i in self.items)

    @property
    def runnable(self) -> bool:
        return self._ok(LAUNCHER, ISLAND, UPSTREAM)

    @property
    def launcher_accepts(self) -> bool:
        return self._ok(LAUNCHER)

    @property
    def island_accepts(self) -> bool:
        return self._ok(ISLAND)


def family_recipe(hf_config: dict) -> str:
    """The ``--rl-model-recipe`` the model registry pairs with this config (CUDA)."""
    from .match import facts_from_config, match_family

    try:
        entry, _ = match_family(facts_from_config(hf_config))
    except (KeyError, TypeError, ValueError):
        return "generic"
    return next((r.rl_model_recipe for r in entry.recipes if r.rl_model_recipe and "cuda" in r.hardware),
                "generic")


# ----------------------------------------------------------------------------- launcher cells
def _form_items(req: CheckRequest, recipe: str, profile: ModelSyncProfile) -> list[CheckItem]:
    items: list[CheckItem] = []
    form, preset, backend = req.form, req.preset, req.backend

    def fail(reason: str, suggestion: str = "", pend: str | None = None, source: str = LAUNCHER) -> None:
        items.append(CheckItem("form", FAIL, reason, suggestion, source, pend or ""))

    if form == "partial-moe":
        if not profile.moe:
            # the launcher checks the selection against config.json only when it can read it locally
            fail("partial experts need a MoE model (the island checks the selection against the checkpoint)",
                 "--tuning full or LoRA", source=ISLAND)
        if backend == "verl":
            fail("verl trains no expert selection: --rl-expert-selection / --expert-full-count are refused "
                 "(yeto/rl/adapters/verl/run_config_rules.py check_model_recipe)", "--rl-backend miles")
        elif recipe not in PARTIAL_RECIPES:
            fail(f"no partial-expert form for --rl-model-recipe {recipe}: --rl-expert-selection belongs to "
                 "kimi_k31 and the generic recipe, --expert-full-count to deepseek-v4-flash "
                 "(yeto/launch/rl_args.py)", "--rl-model-recipe generic with an in_place selection")
        elif recipe == "generic" and preset == "strict-avg":
            fail("--tuning full behind a syncer needs --rl-sync-preset decoupled or full-parameter "
                 "(yeto/launch/rl_args.py)", "--rl-sync-preset decoupled")
        elif recipe == "deepseek-v4-flash" and preset != "full-parameter":
            fail("expert-full tuning needs --rl-sync-preset full-parameter or decoupled "
                 "(yeto/rl/engine/selection.py)", "--rl-sync-preset decoupled")
        elif recipe == "kimi_k31" and preset == "strict-avg":
            fail("--tuning full behind a syncer needs --rl-sync-preset decoupled or full-parameter "
                 "(yeto/launch/rl_args.py)", "--rl-sync-preset decoupled")
    elif form == "full":
        if recipe == "kimi_k31":
            fail("--rl-model-recipe kimi_k31 trains full parameters on a routed-expert selection only "
                 "(--rl-expert-selection, yeto/launch/rl_args.py)", "--form partial-moe")
        elif recipe == "deepseek-v4-flash":
            fail("the deepseek-v4-flash recipe trains LoRA or expert-full experts, with its TP8/EP8 shape",
                 "--form partial-moe")
        elif preset == "strict-avg":
            fail("--tuning full with --rl-sync-preset strict-avg: strict-avg merges the canonical LoRA layout "
                 "(yeto/launch/rl_args.py)", "--rl-sync-preset decoupled (routed to the full-parameter stack)")
    else:  # lora
        if preset == "full-parameter":
            items.append(CheckItem("form", WARN, "the full-parameter preset syncs and trains every parameter "
                                   "whatever --tuning says", "--rl-sync-preset decoupled for LoRA fragments"))
        if recipe == "kimi_k31" and preset in ("decoupled", "full-parameter"):
            fail(f"kimi_k31 LoRA behind the syncer runs strict-avg (got {preset}): its LoRA layout is learned "
                 "from the first export (yeto/rl/engine/kimi_k31_lora.py)", "--rl-sync-preset strict-avg",
                 "decoupled/kimi_k31-lora" if preset == "decoupled" else None)
        if recipe == "deepseek-v4-flash":
            fail("the deepseek-v4-flash recipe pins its LoRA targets and TP8/EP8 shape; this check covers its "
                 "expert-full form (--form partial-moe)", "--form partial-moe")
    if backend == "verl" and max(req.tensor_parallel, req.pipeline_parallel, _ep(req, profile)) > 1:
        megatron_ok = (form == "full" and preset in ("full-parameter", "single-island"))
        if not megatron_ok:
            fail("verl runs TP/PP/EP > 1 on its Megatron island, which needs full parameters with "
                 "--rl-single-island-no-sync or --rl-sync-preset full-parameter "
                 "(yeto/rl/adapters/verl/run_config_rules.py)", "TP=PP=EP=1 (verl FSDP2 shards the model)")
        elif preset == "full-parameter" and req.pipeline_parallel > 1:
            fail("verl full-parameter sync on the Megatron island is gated to PP 1 "
                 "(yeto/rl/adapters/verl/megatron_engine.py sync_problems)", "--pipeline-parallel 1")
    if not any(i.status == FAIL for i in items):
        items.append(CheckItem("form", OK, f"{form} with {preset} on {backend} (recipe {recipe})"))
    return items


def _ep(req: CheckRequest, profile: ModelSyncProfile) -> int:
    return req.expert_parallel if req.expert_parallel is not None else 1


def island_nodes(req: CheckRequest, recipe: str) -> int:
    return 2 if (recipe == "deepseek-v4-flash" and req.form == "partial-moe" and req.backend == "miles") else 1


def rollout_gpus(req: CheckRequest, recipe: str) -> int | None:
    """``--rl-rollout-gpus`` of the cell: Miles full-parameter islands run fixed-partition
    (K3.1 trainers read their groups from sampler islands instead)."""

    if req.backend == "verl":  # the verl Megatron island (TP/PP/EP > 1) runs fixed-partition
        megatron = max(req.tensor_parallel, req.pipeline_parallel, req.expert_parallel or 1) > 1
        return req.gpus_per_island // 2 if megatron and req.form == "full" else None
    if req.preset != "full-parameter" or recipe == "kimi_k31":
        return None
    return 8 if recipe == "deepseek-v4-flash" else req.gpus_per_island // 2


def actor_gpus(req: CheckRequest, recipe: str) -> int:
    """The trainer's GPUs on one island (the island minus its rollout partition)."""

    return island_nodes(req, recipe) * req.gpus_per_island - (rollout_gpus(req, recipe) or 0)


def _recommended_ep(req: CheckRequest, recipe: str, profile: ModelSyncProfile) -> int:
    if not profile.moe or req.backend == "verl":
        return 1
    if recipe in ("kimi_k31", "deepseek-v4-flash"):
        return 8
    world = actor_gpus(req, recipe) // (req.tensor_parallel * req.pipeline_parallel) or 1
    return max(e for e in range(1, min(world, profile.num_routed_experts, 8) + 1)
               if profile.num_routed_experts % e == 0 and world % e == 0)


# ----------------------------------------------------------------------------- model checks
def _parallel_items(req: CheckRequest, recipe: str, profile: ModelSyncProfile, cfg: dict) -> list[CheckItem]:
    from yeto.rl.moe_geometry import text_config

    text = text_config(cfg)
    out: list[CheckItem] = []
    tp, pp, ep = req.tensor_parallel, req.pipeline_parallel, _ep(req, profile)
    for axis, value in (("tp", tp), ("ep", ep)):
        for what, n in _divides(profile, axis):
            if value > 1 and n % value:
                cap = 64 if axis == "ep" else 8
                out.append(CheckItem("parallel", FAIL, f"{axis.upper()} {value} does not divide {what}={n} "
                                     "(Megatron)", _divisors_hint(axis.upper(), n, cap),
                                     ISLAND if "trainable" in what else UPSTREAM))
    actor = actor_gpus(req, recipe)
    if actor % (tp * pp):
        out.append(CheckItem("parallel", FAIL, f"TP*PP={tp * pp} must divide the trainer's {actor} GPUs "
                             "(run_config_rules.check_trainer_parallel)", "", ISLAND))
    if ep > 1:
        if island_nodes(req, recipe) * req.gpus_per_island % ep:
            out.append(CheckItem("parallel", FAIL, f"EP {ep} must divide every island (yeto/launch/rl_args.py)",
                                 "", LAUNCHER))
        elif actor % ep:
            out.append(CheckItem("parallel", FAIL, f"EP {ep} must divide the trainer's {actor} GPUs "
                                 "(run_config_rules.check_expert_parallel)", "", ISLAND))
        if not profile.moe:
            out.append(CheckItem("parallel", FAIL, "EP>1 requires a MoE model (yeto/rl/engine/run_config.py)",
                                 "--expert-parallel 1", ISLAND))
    if pp > 1 and profile.num_layers % pp:
        out.append(CheckItem("parallel", WARN, f"{profile.num_layers} layers do not split evenly over PP {pp}",
                             "set the first/last stage layer counts", UPSTREAM))
    max_pos = int(text.get("max_position_embeddings") or 0)
    if max_pos and req.seq_len > max_pos:
        out.append(CheckItem("parallel", FAIL, f"--seq-len {req.seq_len} exceeds the model context "
                             f"{max_pos} (yeto/rl/engine/run_config.py)", f"--seq-len {max_pos}", ISLAND))
    if not out:
        out.append(CheckItem("parallel", OK, f"TP {tp}, PP {pp}, EP {ep} fit the model and the island"))
    return out


def _divides(profile: ModelSyncProfile, axis: str) -> list[tuple[str, int]]:
    """``(field, n)`` of the profile's "<axis> divides <field>=<n>" constraints."""

    out = []
    for c in profile.constraints:
        if c.startswith(f"{axis} divides "):
            what, _, rest = c[len(axis) + 9:].partition("=")
            out.append((what, int(rest.split()[0])))
    return out


def _divisors_hint(axis: str, n: int, cap: int) -> str:
    return f"{axis} in " + ", ".join(str(d) for d in range(1, cap + 1) if n % d == 0)


def _lora_items(req: CheckRequest, recipe: str, profile: ModelSyncProfile) -> list[CheckItem]:
    if req.form != "lora":
        return []
    out: list[CheckItem] = []
    targets = req.lora_targets
    kinds = _target_kinds(targets, profile)
    if not any(m.kind in kinds for m in profile.linear_modules):
        out.append(CheckItem("lora-targets", FAIL, f"--lora-targets {targets} selects no module of this model",
                             "--lora-targets attention", ISLAND))
    if profile.lora_target_modules.get("attention") == ("o_proj",):
        out.append(CheckItem("lora-targets", WARN, "gated attention: 'attention' adapts the output projection "
                             "only (Megatron cannot split a gated q_proj, yeto/rl/export.py)", "", ISLAND))
    if targets == "all-linear" and profile.moe:
        if not profile.expert_lora.startswith(EXPERT_LORA_SUPPORTED):
            out.append(CheckItem("lora-targets", FAIL, f"routed-expert LoRA: {profile.expert_lora}",
                                 "--lora-targets attention", ISLAND))
        elif recipe == "generic" and _ep(req, profile) > 1:
            out.append(CheckItem("lora-targets", WARN, "routed-expert LoRA under EP>1 needs grouped-GEMM experts "
                                 "(the provider's moe_grouped_gemm), BF16 experts and no capacity padding: the "
                                 "island refuses it otherwise (run_config.check_generic_expert_lora)",
                                 "--expert-parallel 1, or --lora-targets attention", ISLAND))
        if recipe == "kimi_k31" and req.preset != "single-island":
            out.append(CheckItem("lora-targets", FAIL, "kimi_k31 synced LoRA trainers take --lora-targets "
                                 "attention only (yeto/rl/engine/kimi_k31_lora.py)", "--lora-targets attention",
                                 LAUNCHER))
    if not out:
        out.append(CheckItem("lora-targets", OK, f"--lora-targets {targets}: "
                             + ", ".join(profile.lora_target_modules.get(_kind_key(targets), ())), "", ISLAND))
    return out


def _kind_key(targets: str) -> str:
    return "all-linear" if targets == "all-linear" else "attention"


def _target_kinds(targets: str, profile: ModelSyncProfile) -> set[str]:
    if targets == "all-linear":
        return {"attention", "mlp", "shared_experts", "experts", "mamba"}
    return {"attention"}


def _backend_items(req: CheckRequest, recipe: str, profile: ModelSyncProfile) -> list[CheckItem]:
    out: list[CheckItem] = []
    if req.backend == "miles" and recipe == "generic":
        if not profile.megatron_bridge and not any("Nemotron" in a for a in profile.architectures):
            out.append(CheckItem("backend", FAIL, f"Megatron-Bridge {upstream.MEGATRON_BRIDGE_COMMIT[:8]} has no "
                                 f"bridge for {'/'.join(profile.architectures)}: the Miles trainer cannot load it",
                                 "--rl-backend verl (FSDP2 reads the HF model), or a provider view and Miles "
                                 "recipe (template below)", UPSTREAM))
        if req.form in ("full", "partial-moe") and profile.miles_converter is None:
            name = upstream.config_class_name(profile.architectures[0]) if profile.architectures else "?"
            out.append(CheckItem("backend", FAIL, f"no Miles Megatron->HF converter for {name}: full-parameter "
                                 "publication fails at the first update_weights (miles megatron_to_hf "
                                 "_convert_to_hf_core)", "--rl-backend verl, or --form lora, or a converter in "
                                 "the yeto Miles adapter (template below)", UPSTREAM))
    if req.backend == "verl" and recipe != "generic":
        out.append(CheckItem("backend", FAIL, f"verl trains the generic recipe only: --rl-model-recipe {recipe} "
                             "(and its expert selection) is refused (yeto/rl/adapters/verl/run_config_rules.py "
                             "check_model_recipe)", "--rl-backend miles", LAUNCHER))
    if not out:
        out.append(CheckItem("backend", OK, f"{req.backend} loads {'/'.join(profile.architectures) or '?'}"
                             + (f" (converter {profile.miles_converter})" if req.backend == "miles"
                                and req.form != "lora" and profile.miles_converter else ""), "", UPSTREAM))
    return out


def _layout_item(req: CheckRequest, recipe: str, profile: ModelSyncProfile) -> CheckItem:
    if req.preset == "single-island":
        return CheckItem("layout", OK, "no outer sync: no layout to agree on", "", ESTIMATE)
    if req.preset == "full-parameter" or req.form != "lora":
        if req.backend == "verl":
            return CheckItem("layout", OK, "verl plans the full-parameter layout offline",
                             "python -m yeto.rl.adapters.verl.full_parameter_plan --model <id> --model-revision "
                             "<sha> --fragments P --matrix-merge rda", ESTIMATE)
        return CheckItem("layout", OK, "Miles full-parameter layout: --rl-parameter-layout-sha256 auto (the syncer "
                         "binds it at the first HELLO; each island records its hash to pin later)", "", ESTIMATE)
    source = "offline (from the config)" if profile.layout_source == LAYOUT_OFFLINE else \
        "learned from the first export (recipe provider view)"
    return CheckItem("layout", OK, f"canonical LoRA layout: {source}", "", ESTIMATE)


def _naming_item(profile: ModelSyncProfile, weight_names: list[str] | None) -> CheckItem:
    experts = [m for m in profile.linear_modules if m.kind == "experts"]
    if not experts:
        return CheckItem("expert-naming", OK, "no routed experts" if not profile.moe else
                         f"routed experts not per-expert in the checkpoint view ({profile.expert_lora})",
                         "", ESTIMATE)
    if weight_names is None:
        return CheckItem("expert-naming", OK, f"checkpoint expert names from the config: {experts[0].path}.weight "
                         "(pass --index model.safetensors.index.json to verify)", "", ESTIMATE)
    from yeto.rl.moe_geometry import hf_expert_of

    seen: dict[int, set[int]] = {}
    for name in weight_names:
        hit = hf_expert_of(name)
        if hit and hit[0] < profile.num_layers:  # MTP layers (num_nextn_predict_layers) sit past the decoder
            seen.setdefault(hit[0], set()).add(hit[1])
    leaves = {m.name for m in experts}
    named = {n.rsplit(".", 2)[-2] for n in weight_names if (hit := hf_expert_of(n)) and hit[0] < profile.num_layers}
    problems = []
    if not seen:
        problems.append("the index names no per-expert weight (fused or stacked experts)")
    if seen and set(seen) != set(profile.moe_layers):
        problems.append(f"expert layers {sorted(seen)[:4]}.. differ from the config's MoE layers")
    if seen and any(len(ids) != profile.num_routed_experts for ids in seen.values()):
        problems.append(f"a layer holds a different expert count than {profile.num_routed_experts}")
    if seen and not leaves <= named:
        problems.append(f"expert leaves {sorted(named)} are not {sorted(leaves)}")
    if problems:
        return CheckItem("expert-naming", FAIL, "; ".join(problems), "a name mapping in the recipe (template below)",
                         UPSTREAM)
    return CheckItem("expert-naming", OK, f"index: {len(seen)} MoE layers x {profile.num_routed_experts} experts, "
                     f"leaves {sorted(leaves)}", "", ESTIMATE)


# ----------------------------------------------------------------------------- recommendation
def _lora_numel(profile: ModelSyncProfile, targets: str, rank: int) -> tuple[int, int]:
    kinds = _target_kinds(targets, profile)
    tensors = numel = 0
    for m in profile.linear_modules:
        if m.kind not in kinds:
            continue
        copies = len(m.layers) * max(1, m.experts)
        tensors += 2 * copies
        numel += copies * rank * (m.in_features + m.out_features)
    return tensors, numel


def generic_selection_size(profile: ModelSyncProfile) -> int:
    """Trainable experts per MoE layer the check sizes a generic in_place selection with
    (an eighth of the router, at least the experts per token)."""
    return max(profile.experts_per_token or 1, profile.num_routed_experts // 8)


def _trainable_params(profile: ModelSyncProfile, cfg: dict, form: str) -> float:
    from yeto.memory_estimate import param_breakdown

    parts = param_breakdown(cfg)
    if form == "partial-moe":
        per_layer = (getattr(profile.partial_experts, "num_trainable_per_layer", None)
                     or (generic_selection_size(profile) if profile.recipe == "generic" else 0))
        if per_layer and profile.num_routed_experts:
            return parts["routed_experts"] * per_layer / profile.num_routed_experts
        return parts["routed_experts"] / max(1, profile.num_routed_experts) * DSV4_EXPERT_FULL_COUNT
    return parts["total"]


def recommend_sync(req: CheckRequest, profile: ModelSyncProfile, cfg: dict) -> dict[str, Any]:
    """P / H / tau and fragment bytes for the request's preset."""
    if req.preset == "single-island":
        return {"preset": req.preset, "note": "no outer sync"}
    if req.preset == "strict-avg":
        tensors, numel = _lora_numel(profile, req.lora_targets, req.lora_rank)
        return {"preset": req.preset, "fragments": 1, "local_rl_rounds_per_sync": 1,
                "lora_tensors": tensors, "fragment_bytes": numel * 4}
    if req.form == "lora" and req.preset == "decoupled":
        tensors, numel = _lora_numel(profile, req.lora_targets, req.lora_rank)
        fragments = max(2, min(DEFAULT_P, tensors))
        return {"preset": req.preset, "fragments": fragments, "pipeline": min(DEFAULT_TAU, fragments),
                "local_rl_rounds_per_sync": DEFAULT_H, "lora_tensors": tensors,
                "fragment_bytes": math.ceil(numel * 4 / fragments)}
    params = _trainable_params(profile, cfg, req.form)
    by_size = math.ceil(params * 4 / MAX_FRAGMENT_BYTES)
    fragments = max(DEFAULT_P, math.ceil(1.25 * by_size))
    return {"preset": "full-parameter", "rounds": "async", "fragments": fragments, "pipeline": DEFAULT_TAU,
            "local_rl_rounds_per_sync": DEFAULT_H, "trainable_params": int(params),
            "fragment_bytes": math.ceil(params * 4 / fragments), "fragment_bytes_bound": MAX_FRAGMENT_BYTES,
            "note": "P >= 1.25 x (FP32 bytes / 2 GiB): the island's owner-affine planner then yields exactly P"}


def _memory(req: CheckRequest, recipe: str, profile: ModelSyncProfile, cfg: dict) -> dict[str, Any]:
    """Dense models and generic partial experts: the launch preflight's per-GPU peak
    (:mod:`yeto.memory_estimate`) at the request's shape. Other MoE forms: the registry's
    smallest trainer (EP-sharded experts), which the preflight model does not cover."""

    from yeto import memory_estimate as me

    from .match import facts_from_config
    from .recommend import hardware_facts, plan_trainer
    from .schema import BackendRecipe, QuantPlan

    capacity = me.gpu_capacity_gib(req.card) or 80.0
    actor = actor_gpus(req, recipe)
    tuning = "lora" if req.form == "lora" else "full"
    out: dict[str, Any] = {"card": req.card, "capacity_gib": round(capacity, 1), "trainer_gpus": actor}
    generic_partial = profile.moe and req.form == "partial-moe" and recipe == "generic"
    if not profile.moe or generic_partial:
        dp = max(1, actor // (req.tensor_parallel * req.pipeline_parallel))
        colocated = rollout_gpus(req, recipe) is None
        partial = {}
        if generic_partial:  # the preflight's partial-experts shape (selection of generic_selection_size)
            breakdown = me.param_breakdown(cfg)
            partial = {"ep": req.expert_parallel or 1, "expert_params": float(breakdown["routed_experts"]),
                       "trainable_params": float(breakdown["per_expert"] * len(profile.moe_layers)
                                                 * generic_selection_size(profile)),
                       "recompute": True}
        estimate = me.estimate(me.MemoryRequest(
            arch=me.arch_from_hf_config(cfg, precise=True), gpu=req.card, gpu_mem_gib=capacity, tokens=req.seq_len,
            tuning=tuning, lora_rank=req.lora_rank,
            lora_targets="attention" if req.lora_targets == "attention" else "all",
            tp=req.tensor_parallel, pp=req.pipeline_parallel, dp=dp, distributed_optimizer=tuning == "full",
            colocated=colocated, offload=colocated, **partial))
        out["peak_gib_per_gpu"] = estimate["peak_gib"]
        out["fits"] = estimate["peak_gib"] <= 0.9 * capacity
    # partial experts: K3.1's selection is the registry's partial plan; DSV4 expert-full
    # trains four experts per layer next to frozen ones (close to the LoRA plan)
    plan_tuning = {"partial-moe": "lora" if recipe == "deepseek-v4-flash" else "full"}.get(req.form, tuning)
    try:
        stub = BackendRecipe("check", ("cuda",), "miles", "sglang", recipe, ("lora", "full"), (0, math.inf),
                             "", "", QuantPlan("bf16", "bf16", "bf16"), "inferred", "")
        plan = plan_trainer(facts_from_config(cfg), hardware_facts("nvidia", req.card), stub, plan_tuning)
    except (KeyError, ValueError):
        return out
    out["min_trainer"] = {k: plan[k] for k in ("world", "nodes", "tensor_parallel", "pipeline_parallel",
                                                "expert_parallel", "state_gib_per_card") if k in plan}
    out["sampler"] = sampler_memory(req, cfg, capacity)
    if profile.moe and not generic_partial:
        out["fits"] = bool(plan.get("fits")) and plan["world"] <= actor
        out["note"] = "MoE: the registry's smallest trainer (model state per card, EP-sharded experts)"
    return out


def sampler_memory(req: CheckRequest, cfg: dict, capacity: float) -> dict[str, Any]:
    """A sampler island's SGLang engine (the launch preflight's sampler estimate,
    :func:`yeto.memory_estimate.estimate_engine`): the narrowest engine TP, at the
    default then a raised ``--sglang-mem-fraction-static``, whose weights and one
    context of KV cache fit (bf16 weights: an upper bound for quantized checkpoints)."""

    from yeto import memory_estimate as me

    arch = me.arch_from_hf_config(cfg, precise=True)
    widths = [tp for tp in (1, 2, 4, 8) if tp <= req.gpus_per_island and req.gpus_per_island % tp == 0]
    last: dict[str, Any] = {}
    for fraction in (0.4, 0.8):
        for tp in widths:
            est = me.estimate_engine(me.EngineRequest(
                arch=arch, gpu=req.card, gpu_mem_gib=capacity, tokens=req.seq_len, engine_tp=tp,
                mem_fraction_static=fraction, kv_bytes_per_token=me.kv_bytes_per_token(arch, cfg, tp)))
            last = {"engine_tp": tp, "mem_fraction_static": fraction, "peak_gib": est["peak_gib"],
                    "kv_pool_tokens": est["kv_pool_tokens"],
                    "fits": est["engine_fits"] and est["peak_gib"] <= 0.9 * capacity}
            if last["fits"]:
                return last
    return last


# ----------------------------------------------------------------------------- launch flags and templates
def launch_argv(req: CheckRequest, recipe: str, rec: dict[str, Any]) -> list[str]:
    """The ``yeto launch`` flags of the cell (placeholders in <...>)."""
    argv = ["--rl-model-recipe", recipe]
    if req.backend == "verl":
        argv += ["--rl-backend", "verl"]
    ep = req.expert_parallel
    tuning = {"lora": "lora", "full": "full"}.get(req.form)
    if req.form == "partial-moe":
        if recipe == "deepseek-v4-flash":
            argv += ["--tuning", "lora", "--expert-full-count", str(DSV4_EXPERT_FULL_COUNT),
                     "--lora-targets", "attention",
                     "--expert-selection-sha256", "<64-hex>", "--expert-selection-contract-sha256", "<64-hex>",
                     "--rollout-num-gpus-per-engine", "8", "--sglang-tp-size", "8", "--sglang-ep-size", "8",
                     "--sglang-attention-backend", "dsv4", "--sglang-page-size", "256",
                     "--no-sglang-deterministic-inference"]
        else:
            argv += ["--tuning", "full", "--rl-expert-selection", "<expert_selection.json>"]
    else:
        argv += ["--tuning", tuning]
        if req.form == "lora":
            argv += ["--lora-r", str(req.lora_rank), "--lora-targets", req.lora_targets]
    argv += ["--tensor-parallel", str(req.tensor_parallel), "--pipeline-parallel", str(req.pipeline_parallel)]
    if ep is not None:
        argv += ["--expert-parallel", str(ep)]
    preset = req.preset
    if req.backend == "verl" and rollout_gpus(req, recipe):
        argv += ["--rl-placement", "fixed-partition", "--rl-rollout-gpus", str(rollout_gpus(req, recipe))]
    if preset == "single-island":
        return argv + ["--rl-single-island-no-sync"]
    if preset == "strict-avg":
        return argv + ["--rl-sync-preset", "strict-avg", "--local-rl-rounds-per-sync", "1"]
    if preset == "decoupled":
        return argv + ["--rl-sync-preset", "decoupled", "--fragments", str(rec.get("fragments", DEFAULT_P)),
                       "--pipeline", str(rec.get("pipeline", DEFAULT_TAU)),
                       "--local-rl-rounds-per-sync", str(rec.get("local_rl_rounds_per_sync", DEFAULT_H))]
    argv += ["--rl-sync-preset", "full-parameter", "--rl-sync-full-parameter-rounds", "async",
             "--fragments", str(rec.get("fragments", DEFAULT_P)),
             "--local-rl-rounds-per-sync", str(rec.get("local_rl_rounds_per_sync", DEFAULT_H)),
             "--rl-parameter-layout-sha256", "auto" if req.backend == "miles" else "<verl plan hash>"]
    if req.backend == "miles":
        if recipe == "kimi_k31":  # synced K3.1 trainers read their groups from the store (sampler islands)
            argv += ["--rl-sample-role", "trainer", "--rl-checkpoint-store", "<store>", "--rl-max-policy-age", "2"]
        else:
            argv += ["--rl-placement", "fixed-partition", "--rl-rollout-gpus", str(rollout_gpus(req, recipe))]
    return argv


def node_shape(req: CheckRequest, recipe: str) -> tuple[int, int]:
    """``(nodes, GPUs per node)`` of one island on Modal.

    An island larger than one whole node of the card (``MODAL_FULL_NODE``, the rule
    ``validate_modal_shape`` enforces) is split into whole nodes: 48 B300 -> 6x8. A total
    the card's node size does not divide, or a card without a whole-node size, is left as
    one container (``island_shape_item`` warns)."""
    from yeto.gpu_spec import UnknownCard, canonical_card
    from yeto.modal_runner import MODAL_FULL_NODE

    nodes, per_node = island_nodes(req, recipe), req.gpus_per_island
    total = nodes * per_node
    try:
        full = MODAL_FULL_NODE.get(canonical_card(req.card))
    except UnknownCard:
        full = None
    if full and total > full and total % full == 0:
        return total // full, full
    return nodes, per_node


def island_shape_item(req: CheckRequest, recipe: str) -> CheckItem | None:
    """WARN when the suggested island is not a shape Modal schedules (1-8 GPUs per container,
    whole nodes when there is more than one)."""
    nodes, per_node = node_shape(req, recipe)
    if per_node <= 8:
        return None
    return CheckItem("hardware", WARN, f"{nodes * per_node} GPUs per island is not a whole number of "
                     f"{req.card} nodes (yeto/modal_runner.py validate_modal_shape)",
                     "--gpus-per-island a multiple of the card's GPUs per node", ESTIMATE)


def gpu_spec(req: CheckRequest, recipe: str) -> str:
    """The ``--gpu`` of the cell: Modal islands split into whole nodes (see ``node_shape``)."""
    nodes, per_node = node_shape(req, recipe)
    island = f"modal:{nodes}x{per_node}x{req.card.lower()}"
    return ",".join([island] * (1 if req.preset == "single-island" else req.islands))


TEMPLATE_PROVIDER = """# yeto/rl/adapters/miles/models/<family>/provider.py  (Megatron-Bridge has no bridge)
# Model it on models/deepseek_v41/provider.py: a provider view that maps the HF
# config to a Megatron GPTModelProvider, plus a ports recipe and a
# --rl-model-recipe <family> entry in yeto/rl/engine/run_config.py."""
TEMPLATE_CONVERTER = """# Megatron -> HF full-parameter publication (miles _convert_to_hf_core has none):
# bridge it in the yeto Miles adapter (no upstream PR), e.g. a converter keyed by
# the config class that maps decoder.layers.L.* to model.layers.L.*, selected
# with the recipe's --model-name. Until then: --rl-backend verl or --form lora."""
TEMPLATE_SELECTION = """# the routed experts to train: an in_place selection (any MoE, generic recipe)
python -m yeto.rl.adapters.miles.models.partial_experts.selection --layout in_place \\
  --hf-config <checkpoint>/config.json --num-trainable {trainable} --ep-size {ep} --mode uniform \\
  --output expert_selection.json
python -m yeto.rl.adapters.miles.models.partial_experts.selection --validate expert_selection.json \\
  --hf-config <checkpoint>/config.json
# then: --tuning full --rl-expert-selection expert_selection.json --expert-parallel {ep}"""
TEMPLATE_FAMILY = """# yeto/rl/model_registry/families.py: a FamilyEntry so `yeto adapt` and
# `yeto model check` name the family instead of the generic fallback
{name} = FamilyEntry("{family}", ("{arch}",), ("{model_type}",), _dense_recipes(
    "{family}", cuda_status="inferred", ascend_status="inferred", ascend_evidence="none yet"))"""
TEMPLATE_NAMING = """# The checkpoint's expert weights do not follow {path}.weight: map them in the
# recipe's provider view (Megatron-Bridge MegatronMappingRegistry entries for
# decoder.layers.*.mlp.experts.linear_fc1/linear_fc2) before training."""


def _templates(report: CheckReport) -> list[str]:
    out = []
    reasons = " ".join(i.reason for i in report.items if i.status == FAIL)
    p = report.profile
    if "no bridge" in reasons:
        out.append(TEMPLATE_PROVIDER)
    if "converter" in reasons:
        out.append(TEMPLATE_CONVERTER)
    req = report.request
    if req.form == "partial-moe" and report.recipe == "generic" and p.moe:
        out.append(TEMPLATE_SELECTION.format(trainable=generic_selection_size(p), ep=req.expert_parallel or 1))
    if any(i.name == "expert-naming" and i.status == FAIL for i in report.items):
        experts = next((m for m in p.linear_modules if m.kind == "experts"), None)
        out.append(TEMPLATE_NAMING.format(path=experts.path if experts else "<experts path>"))
    from .match import facts_from_config, match_family

    try:
        _, how = match_family(facts_from_config(report._config))  # type: ignore[attr-defined]
    except (KeyError, TypeError, ValueError, AttributeError):
        how = "fallback"
    if how == "fallback" and p.architectures:
        out.append(TEMPLATE_FAMILY.format(name=p.model_type.upper() or "NEW", family=p.model_type or "new",
                                          arch=p.architectures[0], model_type=p.model_type))
    return out


# ----------------------------------------------------------------------------- entry
def check_model(hf_config: dict, request: CheckRequest, *, model: str = "<model>",
                weight_names: list[str] | None = None) -> CheckReport:
    recipe = request.recipe or family_recipe(hf_config)
    profile = _derive(hf_config, recipe=recipe)
    req = request
    # decoupled with full parameters (dense, partial experts, DSV4 expert-full) runs the
    # full-parameter stack's asynchronous rounds (yeto/rl/engine/sync_route.py); the cell is
    # checked, recommended and flagged as that preset, launched as decoupled
    routed = req.preset == "decoupled" and req.form != "lora"
    if routed:
        req = replace(req, preset="full-parameter")
    if req.expert_parallel is None:
        req = replace(req, expert_parallel=_recommended_ep(req, recipe, profile) if profile.moe else None)
    if recipe == "deepseek-v4-flash" and req.form == "partial-moe" and req.tensor_parallel == 1:
        req = replace(req, tensor_parallel=8)
    items: list[CheckItem] = []
    if routed:
        items.append(CheckItem("route", OK, "--rl-sync-preset decoupled with full parameters runs the full-parameter "
                               "stack's asynchronous rounds (yeto/rl/engine/sync_route.py)", "", LAUNCHER))
    if profile.missing:
        items.append(CheckItem("contract", FAIL, "missing: " + "; ".join(profile.missing),
                               "a --rl-model-recipe that supplies them", UPSTREAM))
    else:
        items.append(CheckItem("contract", OK, f"{profile.model_type}: {profile.num_layers} layers"
                               + (f", {len(profile.moe_layers)} MoE layers x {profile.num_routed_experts} experts"
                                  if profile.moe else ", dense"), "", ESTIMATE))
    items += _form_items(req, recipe, profile)
    items += _backend_items(req, recipe, profile)
    items += _parallel_items(req, recipe, profile, hf_config)
    items += _lora_items(req, recipe, profile)
    items.append(_layout_item(req, recipe, profile))
    items.append(_naming_item(profile, weight_names))
    shape = island_shape_item(req, recipe)
    if shape is not None:
        items.append(shape)
    report = CheckReport(model=model, request=replace(req, preset="decoupled") if routed else req, recipe=recipe,
                         profile=profile, items=items)
    report._config = hf_config  # type: ignore[attr-defined]
    sync = recommend_sync(req, profile, hf_config)
    try:
        memory = _memory(req, recipe, profile, hf_config)
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        memory = {"note": f"not estimated: {error}"}
    report.recommended = {"sync": sync, "memory": memory, "expert_parallel": req.expert_parallel,
                          "gpu": gpu_spec(req, recipe)}
    if memory.get("fits") is False:
        if "peak_gib_per_gpu" in memory:
            reason = (f"estimated peak {memory['peak_gib_per_gpu']} GiB per GPU over 90% of {memory['capacity_gib']}"
                      f" GiB at TP {req.tensor_parallel} / PP {req.pipeline_parallel}")
        else:
            reason = (f"the smallest trainer that holds the model state is {memory.get('min_trainer')}, more than "
                      f"the {memory['trainer_gpus']} trainer GPUs of an island")
        report.items.append(CheckItem("memory", WARN, reason, "more TP/PP/EP, more GPUs per island, a shorter "
                                      "--seq-len, or a larger card (see min_trainer)", ESTIMATE))
    report.launch_argv = launch_argv(req, recipe, sync)
    if routed:
        at = report.launch_argv.index("--rl-sync-preset") + 1
        report.launch_argv[at] = "decoupled"
    report.templates = _templates(report)
    return report


def report_dict(report: CheckReport) -> dict[str, Any]:
    from .sync_profile import profile_dict

    return {"model": report.model, "recipe": report.recipe, "runnable": report.runnable,
            "launcher_accepts": report.launcher_accepts, "request": vars(report.request),
            "items": [vars(i) for i in report.items], "recommended": report.recommended,
            "launch_argv": report.launch_argv, "pending": report.pending, "templates": report.templates,
            "profile": profile_dict(report.profile)}


def render_text(report: CheckReport) -> str:
    r = report.request
    lines = [f"model    {report.model}  recipe={report.recipe}  form={r.form}  preset={r.preset}  backend={r.backend}",
             f"verdict  {'RUNNABLE' if report.runnable else 'NOT RUNNABLE'}"]
    for item in report.items:
        mark = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}[item.status]
        lines.append(f"  [{mark}] {item.name:<13} {item.reason}  ({item.source})")
        if item.suggestion and item.status != OK:
            lines.append(f"         -> {item.suggestion}")
    sync, memory = report.recommended.get("sync", {}), report.recommended.get("memory", {})
    keys = ("fragments", "pipeline", "local_rl_rounds_per_sync", "fragment_bytes", "lora_tensors", "trainable_params")
    lines.append("sync     " + ", ".join(f"{k}={sync[k]}" for k in keys if k in sync))
    peak = f"peak {memory['peak_gib_per_gpu']} GiB/GPU, " if "peak_gib_per_gpu" in memory else ""
    lines.append(f"memory   {memory.get('card')} ({memory.get('capacity_gib')} GiB): {peak}fits={memory.get('fits')}; "
                 f"smallest trainer {memory.get('min_trainer')}")
    sampler = memory.get("sampler") or {}
    if sampler:
        lines.append(f"sampler  engine TP {sampler['engine_tp']}, --sglang-mem-fraction-static "
                     f"{sampler['mem_fraction_static']}: fits={sampler['fits']}, peak {sampler['peak_gib']} GiB/GPU, "
                     f"KV pool {sampler['kv_pool_tokens']} tokens")
    lines.append(f"gpu      {report.recommended.get('gpu')}")
    lines.append("flags    " + " ".join(report.launch_argv))
    lines += [f"not on main yet: {PENDING[key]}" for key in report.pending]
    for template in report.templates:
        lines.append("template\n" + template)
    lines.append("(config.json only; upstream rules from the pinned sources; estimates are upper bounds)")
    return "\n".join(lines)
