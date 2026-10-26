"""Memory estimate of one full-parameter training step (S25 N7.1 FSDP2, N7.1b Megatron).

Every number here is derived from formulas, not measured. The launcher uses
the estimate to refuse a plan before it reaches any node: a plan whose
per-card device total or per-node host total exceeds the inventory stops the
launch with the items that make it too large.

Model of one step (``P`` trained parameters, ``N`` cards in one FSDP2 shard
group, ``T`` tokens per sequence, ``U`` Ulysses degree):

* sharded state, ``P / N`` per card: parameters (``param_bytes``: 4 for an
  FP32 copy that FSDP2 keeps when it computes in BF16, 2 without one), the
  reduced gradients (``grad_bytes``: 4 for FP32 reduction, 2 for BF16) and the
  optimizer state (``state_bytes``: 8 for FP32 Adam m and v, 4 for BF16, 2 for
  8-bit). Where it lives follows ``offload``:

  - ``none`` (plan A): all on the card;
  - ``optimizer`` (plan B): BF16 parameters and gradients on the card, the FP32
    copy and the Adam state on the host, the optimizer step on the host CPU.
    No yeto or verl code runs this split yet (``PLAN_STATUS``);
  - ``full`` (plan C): FSDP2 ``CPUOffloadPolicy``; the card holds only the
    gathered layers and the activations.

* gathered layers: ``prefetch_layers`` unsharded BF16 copies of the largest
  layer, plus that layer's unsharded gradient before the reduce-scatter;
* activations with full recomputation: the layer inputs (``layers x T/U x
  residual_streams x hidden x 2 B``; mHC keeps ``hc_mult`` streams), the
  logits and their gradient in FP32 (unless the loss is chunked), and a
  configured margin for the recomputed layer's working set (KDA chunk state,
  MoE dispatch, attention scores), because no measurement exists;
* the dense terms of the HF reference DSA path (``dsa_path="hf_dense"``): the
  ``[T, T]`` int32 + bool mask and the FP32 indexer scores
  ``[T, index_heads, T / kpool]`` (modeling_glm5_next.py L825, L1236-1246;
  source read, not run at this size). A sparse DSA kernel removes them; none
  exists for training on PPU or NPU yet (N7.0).

Megatron (``framework="megatron"``, N7.1b; the PPU stage-1 backend, A-TRN §4).
``W`` cards, tensor parallel ``TP`` (sequence parallel assumed on), pipeline
``PP``, expert parallel ``EP`` (expert tensor parallel 1), context parallel
``CP``. Figures are for the worst pipeline stage (``ceil(layers / PP)``
layers; every parameter is spread over the layers, embedding and output-layer
skew is not modelled):

* local parameters: dense ``(P - E) x f / TP`` plus experts ``E x f / EP``
  (``E`` = routed-expert parameters, ``f`` = the stage's layer fraction);
* BF16 parameters (2 B) and the main gradients (``grad_bytes``) on every
  local parameter;
* optimizer: ``param_bytes`` 4 adds a 4 B FP32 master copy to the
  ``state_bytes`` Adam state; with the distributed optimizer this is divided
  by the data-parallel size of its group (dense ``W / (TP x PP)``, CP
  included; experts ``W / (EP x PP)``), without it it is per local parameter;
* optimizer host offload (N7.1c; Megatron ``--optimizer-cpu-offload
  --optimizer-offload-fraction F``): the fraction ``F`` of the Adam state
  moves to the host, and with ``offload_master`` also that fraction of the
  FP32 master. Host memory per node is the offloaded bytes of one card times
  ``cards_per_node`` (every card of a node offloads the same share). Not
  modelled: FP32 gradient copies or pinned staging buffers that the CPU
  optimizer may keep on the host (Megatron source not checked for this);
* activations with full recomputation: stage 0 of 1F1B holds ``PP`` micro
  batches, so ``ceil(layers / PP) x PP x mbs x T / (TP x CP) x streams x hidden
  x 2 B``; mHC keeps ``hc_mult`` (4) residual streams, so this term is 4 times
  that of a one-stream model; the last stage adds the logits split by ``TP``;
* Miles' DSA layer does not support CP (``dsa.py`` NotImplementedError), so a
  DSA model with ``CP > 1`` is refused.

Every output carries ``basis: "推算"`` (derived, not measured).

Not modelled: allocator fragmentation, collective buffers, the runtime's own
reserve (covered by ``device_usable_fraction``), CPU optimizer step time.

Status: formulas checked by CPU unit tests; every output needs on-card
measurement (N7.4 / N9.1).

CLI::

    python -m yeto.rl.fullparam.memory --config config.json --cards 64 \\
        --cards-per-node 8 --hbm-gib 144 --host-memory-gib 2048 --seq-len 32768
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, field, replace
from typing import Any

GIB = 1024**3
OFFLOAD_MODES = ("none", "optimizer", "full")
DSA_PATHS = ("none", "hf_dense", "sparse_kernel")
FRAMEWORKS = ("fsdp2", "megatron")
PLAN_STATUS = {
    "none": "FSDP2 standard path",
    "optimizer": "needs a host optimizer step; not implemented in yeto or verl",
    "full": "FSDP2 CPUOffloadPolicy (optimizer step on the host CPU)",
}
LABEL = "derived from formulas, not measured"
BASIS = "推算"  # the dashboard / report word for LABEL


class MemoryPlanError(ValueError):
    """The training plan does not fit the inventory, or is not well formed."""


@dataclass(frozen=True)
class ModelShape:
    """What the estimate needs from the model.

    ``params``: trained parameters; ``largest_layer_params``: the largest
    decoder layer (the unit FSDP2 gathers); ``head_counts``: every head count
    Ulysses must divide; ``dsa_layers`` / ``index_heads`` / ``index_kpool``:
    the DSA indexer (0 layers: no DSA); ``expert_params`` / ``num_experts``:
    the routed experts that Megatron splits by expert parallel (0: dense)."""

    params: int
    largest_layer_params: int
    num_layers: int
    hidden_size: int
    vocab_size: int
    residual_streams: int = 1
    head_counts: tuple[int, ...] = ()
    dsa_layers: int = 0
    index_heads: int = 0
    index_kpool: int = 1
    expert_params: int = 0
    num_experts: int = 0
    source: str = "manual"

    def __post_init__(self) -> None:
        for name in ("params", "largest_layer_params", "num_layers", "hidden_size", "vocab_size",
                     "residual_streams", "index_kpool"):
            if int(getattr(self, name)) <= 0:
                raise MemoryPlanError(f"model shape: {name} must be positive, got {getattr(self, name)}")
        if self.largest_layer_params > self.params:
            raise MemoryPlanError("model shape: largest_layer_params exceeds params")
        if not 0 <= self.expert_params <= self.params or self.num_experts < 0:
            raise MemoryPlanError("model shape: expert_params must be in [0, params], num_experts >= 0")
        if bool(self.expert_params) != bool(self.num_experts):
            raise MemoryPlanError("model shape: expert_params and num_experts go together")


@dataclass(frozen=True)
class TrainPlan:
    """One trainer island and how it trains. ``fsdp_size`` 0 means all cards in
    one shard group; a smaller divisor replicates groups (HSDP). ``tp`` /
    ``pp`` / ``ep`` / ``cp`` / ``distributed_optimizer`` /
    ``optimizer_offload_fraction`` / ``offload_master`` apply to
    ``framework="megatron"`` only."""

    cards: int
    cards_per_node: int
    hbm_gib: float
    host_memory_gib: float
    seq_len: int
    micro_batch_seqs: int = 1
    fsdp_size: int = 0
    ulysses: int = 1
    offload: str = "none"
    param_bytes: int = 4
    grad_bytes: int = 4
    state_bytes: int = 8
    prefetch_layers: int = 2
    logits_fp32: bool = True
    chunked_loss: bool = False
    dsa_path: str = "hf_dense"
    activation_margin_gib: float = 4.0
    device_usable_fraction: float = 0.92
    host_usable_fraction: float = 0.85
    host_reserve_gib: float = 64.0  # per node: OS, data loader, the requant sidecar
    framework: str = "fsdp2"
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    distributed_optimizer: bool = True
    optimizer_offload_fraction: float = 0.0  # share of the optimizer on the host (Megatron)
    offload_master: bool = True  # the offloaded share includes the FP32 master copy

    @property
    def shard_group(self) -> int:
        return self.fsdp_size or self.cards


def _check_megatron_form(plan: TrainPlan, shape: ModelShape) -> None:
    for name in ("tp", "pp", "ep", "cp"):
        if int(getattr(plan, name)) <= 0:
            raise MemoryPlanError(f"{name} must be positive, got {getattr(plan, name)}")
    if plan.offload != "none":
        raise MemoryPlanError("megatron: offload is an FSDP2 mode; use optimizer_offload_fraction "
                              "(Megatron --optimizer-offload-fraction)")
    if not 0.0 <= plan.optimizer_offload_fraction <= 1.0:
        raise MemoryPlanError(f"megatron: optimizer_offload_fraction must be in [0, 1], "
                              f"got {plan.optimizer_offload_fraction}")
    if plan.ulysses != 1 or plan.fsdp_size:
        raise MemoryPlanError("megatron: use cp instead of ulysses, and no fsdp_size")
    W = plan.cards
    if W % (plan.tp * plan.pp * plan.cp):
        raise MemoryPlanError(f"megatron: tp x pp x cp = {plan.tp * plan.pp * plan.cp} does not divide "
                              f"cards {W}")
    if W % (plan.ep * plan.pp):
        raise MemoryPlanError(f"megatron: ep x pp = {plan.ep * plan.pp} does not divide cards {W}")
    if plan.pp > shape.num_layers:
        raise MemoryPlanError(f"megatron: pp {plan.pp} exceeds the {shape.num_layers} layers")
    if plan.ep > 1 and not shape.num_experts:
        raise MemoryPlanError("megatron: ep > 1 needs a model with routed experts")
    if shape.num_experts and shape.num_experts % plan.ep:
        raise MemoryPlanError(f"megatron: ep {plan.ep} does not divide {shape.num_experts} experts")
    if plan.seq_len % (plan.tp * plan.cp):
        raise MemoryPlanError(f"megatron: tp x cp = {plan.tp * plan.cp} does not divide seq_len "
                              f"{plan.seq_len} (sequence parallel)")
    for heads in shape.head_counts:
        if heads % plan.tp:
            raise MemoryPlanError(f"megatron: tp {plan.tp} does not divide a head count {heads}")
    if shape.dsa_layers and plan.cp > 1:
        raise MemoryPlanError("megatron: the Miles DSA layer does not support cp > 1")


def check_plan_form(plan: TrainPlan, shape: ModelShape) -> None:
    """Refuse a plan that is not well formed (raises :class:`MemoryPlanError`)."""
    if plan.framework not in FRAMEWORKS:
        raise MemoryPlanError(f"framework must be one of {FRAMEWORKS}, got {plan.framework!r}")
    if plan.framework == "megatron":
        _check_megatron_form(plan, shape)
    elif (plan.tp, plan.pp, plan.ep, plan.cp) != (1, 1, 1, 1):
        raise MemoryPlanError("tp / pp / ep / cp apply to framework megatron only")
    if plan.framework != "megatron" and plan.optimizer_offload_fraction:
        raise MemoryPlanError("optimizer_offload_fraction applies to framework megatron only "
                              "(FSDP2: offload)")
    if plan.offload not in OFFLOAD_MODES:
        raise MemoryPlanError(f"offload must be one of {OFFLOAD_MODES}, got {plan.offload!r}")
    if plan.dsa_path not in DSA_PATHS:
        raise MemoryPlanError(f"dsa_path must be one of {DSA_PATHS}, got {plan.dsa_path!r}")
    for name in ("cards", "cards_per_node", "seq_len", "micro_batch_seqs", "ulysses", "prefetch_layers"):
        if int(getattr(plan, name)) <= 0:
            raise MemoryPlanError(f"{name} must be positive, got {getattr(plan, name)}")
    if plan.cards % plan.cards_per_node:
        raise MemoryPlanError(f"cards {plan.cards} is not a multiple of cards_per_node {plan.cards_per_node}")
    if plan.cards % plan.shard_group:
        raise MemoryPlanError(f"fsdp_size {plan.shard_group} does not divide cards {plan.cards}")
    if plan.shard_group % plan.ulysses:
        raise MemoryPlanError(f"ulysses {plan.ulysses} does not divide the shard group {plan.shard_group}")
    if plan.seq_len % plan.ulysses:
        raise MemoryPlanError(f"ulysses {plan.ulysses} does not divide seq_len {plan.seq_len}")
    for heads in shape.head_counts:
        if heads % plan.ulysses:
            raise MemoryPlanError(f"ulysses {plan.ulysses} does not divide a head count {heads}")
    if plan.param_bytes not in (2, 4) or plan.grad_bytes not in (2, 4) or plan.state_bytes not in (2, 4, 8):
        raise MemoryPlanError("param_bytes and grad_bytes must be 2 or 4; state_bytes 2, 4 or 8")
    if not (0 < plan.device_usable_fraction <= 1 and 0 < plan.host_usable_fraction <= 1):
        raise MemoryPlanError("usable fractions must be in (0, 1]")


@dataclass
class Item:
    bytes: float
    formula: str
    where: str  # "device" (per card) or "host" (per node)

    @property
    def gib(self) -> float:
        return self.bytes / GIB


@dataclass
class Estimate:
    shape: ModelShape
    plan: TrainPlan
    items: dict[str, Item] = field(default_factory=dict)
    label: str = LABEL

    def total(self, where: str) -> float:
        return sum(i.bytes for i in self.items.values() if i.where == where)

    @property
    def device_limit(self) -> float:
        return self.plan.hbm_gib * GIB * self.plan.device_usable_fraction

    @property
    def host_limit(self) -> float:
        return (self.plan.host_memory_gib - self.plan.host_reserve_gib) * GIB * self.plan.host_usable_fraction

    def problems(self) -> list[str]:
        out = []
        for where, limit, unit in (("device", self.device_limit, "card"), ("host", self.host_limit, "node")):
            total = self.total(where)
            if total > limit:
                big = sorted(((i.bytes, n) for n, i in self.items.items() if i.where == where), reverse=True)[:3]
                out.append(f"{where} {total / GIB:.1f} GiB per {unit} exceeds the usable {limit / GIB:.1f} GiB "
                           f"(largest: {', '.join(f'{n} {b / GIB:.1f} GiB' for b, n in big)})")
        return out

    @property
    def fits(self) -> bool:
        return not self.problems()

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "basis": BASIS,
            "framework": self.plan.framework,
            "plan_status": (PLAN_STATUS[self.plan.offload] if self.plan.framework == "fsdp2"
                            else _megatron_status(self.plan)),
            "device_gib_per_card": round(self.total("device") / GIB, 1),
            "device_usable_gib": round(self.device_limit / GIB, 1),
            "host_gib_per_node": round(self.total("host") / GIB, 1),
            "host_usable_gib": round(self.host_limit / GIB, 1),
            "fits": self.fits,
            "problems": self.problems(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary(), "shape": asdict(self.shape), "plan": asdict(self.plan),
                "items": {n: {"gib": round(i.gib, 3), "where": i.where, "formula": i.formula}
                          for n, i in self.items.items()}}


def estimate(shape: ModelShape, plan: TrainPlan) -> Estimate:
    """The per-card device and per-node host memory of one training step."""
    check_plan_form(plan, shape)
    if plan.framework == "megatron":
        return _estimate_megatron(shape, plan)
    est = Estimate(shape, plan)
    P, N, per_node = shape.params, plan.shard_group, plan.cards_per_node
    sharded = P / N

    def add(name: str, value: float, formula: str, where: str = "device") -> None:
        est.items[name] = Item(float(value), formula, where)

    pb, gb, sb = plan.param_bytes, plan.grad_bytes, plan.state_bytes
    if plan.offload == "none":
        add("params_shard", pb * sharded, f"{pb} B x P / N")
        add("grads_shard", gb * sharded, f"{gb} B x P / N")
        add("optimizer_shard", sb * sharded, f"{sb} B x P / N")
    elif plan.offload == "optimizer":
        add("params_shard_bf16", 2 * sharded, "2 B x P / N")
        add("grads_shard_bf16", 2 * sharded, "2 B x P / N")
        master = 4 if pb == 4 else 0
        add("host_master_and_optimizer", (master + sb) * sharded * per_node,
            f"({master} B master + {sb} B state) x P / N x cards_per_node", "host")
    else:
        add("host_params_grads_optimizer", (pb + gb + sb) * sharded * per_node,
            f"({pb} + {gb} + {sb}) B x P / N x cards_per_node", "host")

    L = shape.largest_layer_params
    add("gathered_layers", plan.prefetch_layers * L * 2 + L * gb,
        f"prefetch({plan.prefetch_layers}) x largest_layer x 2 B + largest_layer x {gb} B unsharded grad")

    tokens = plan.seq_len * plan.micro_batch_seqs / plan.ulysses
    add("layer_inputs", shape.num_layers * tokens * shape.residual_streams * shape.hidden_size * 2,
        f"layers({shape.num_layers}) x T*mbs/U({tokens:.0f}) x streams({shape.residual_streams}) "
        f"x hidden({shape.hidden_size}) x 2 B")
    if not plan.chunked_loss:
        lb = 4 if plan.logits_fp32 else 2
        add("logits_and_grad", tokens * shape.vocab_size * lb * 2, f"T*mbs/U x vocab({shape.vocab_size}) x {lb} B x 2")
    add("working_set_margin", plan.activation_margin_gib * GIB,
        "configured: recomputed layer working set (KDA, MoE dispatch, attention); not measured")
    if shape.dsa_layers and plan.dsa_path == "hf_dense":
        T, b = plan.seq_len, plan.micro_batch_seqs
        mask = b * T * T * 5
        scores = b * T * shape.index_heads * -(-T // shape.index_kpool) * 4
        add("hf_dsa_dense_terms", mask + scores,
            f"mask mbs*T*T*5 B ({mask / GIB:.1f} GiB) + indexer scores mbs*T*{shape.index_heads}"
            f"*T/{shape.index_kpool}*4 B ({scores / GIB:.1f} GiB); not divided by Ulysses")
    return est


def _estimate_megatron(shape: ModelShape, plan: TrainPlan) -> Estimate:
    """Megatron variant of :func:`estimate` (module doc, N7.1b)."""
    est = Estimate(shape, plan)

    def add(name: str, value: float, formula: str, where: str = "device") -> None:
        est.items[name] = Item(float(value), formula, where)

    W, tp, pp, ep, cp = plan.cards, plan.tp, plan.pp, plan.ep, plan.cp
    stage_layers = -(-shape.num_layers // pp)
    f = stage_layers / shape.num_layers
    dense = (shape.params - shape.expert_params) * f / tp
    experts = shape.expert_params * f / ep
    local = dense + experts
    where = f"stage of {stage_layers}/{shape.num_layers} layers"
    add("params_local_bf16", 2 * local, f"2 B x (dense/TP({tp}) + experts/EP({ep})), {where}")
    gb = plan.grad_bytes
    add("main_grads_local", gb * local, f"{gb} B x local params")
    if plan.distributed_optimizer:
        dp_dense, dp_expert = W // (tp * pp), W // (ep * pp)
        owned = dense / dp_dense + experts / dp_expert
        owned_text = (f"(dense local / DP({dp_dense}) + expert local / expert DP({dp_expert})), "
                      "distributed optimizer")
    else:
        owned, owned_text = local, "local params (no distributed optimizer)"
    master = 4 if plan.param_bytes == 4 else 0
    F = plan.optimizer_offload_fraction
    host_per_param = F * (plan.state_bytes + (master if plan.offload_master else 0))
    card_per_param = plan.state_bytes + master - host_per_param
    name = "optimizer_shard" if plan.distributed_optimizer else "optimizer_local"
    add(name, card_per_param * owned, f"{card_per_param:g} B x {owned_text}, on the card")
    if F:
        what = "Adam state + FP32 master" if plan.offload_master and master else "Adam state"
        add("host_optimizer_offload", host_per_param * owned * plan.cards_per_node,
            f"{F:g} x {what} ({host_per_param:g} B) x {owned_text} x cards_per_node", "host")
    tokens = plan.seq_len * plan.micro_batch_seqs / (tp * cp)
    add("layer_inputs", stage_layers * pp * tokens * shape.residual_streams * shape.hidden_size * 2,
        f"stage layers({stage_layers}) x in-flight micro batches PP({pp}) x T*mbs/(TP*CP)({tokens:.0f}) "
        f"x streams({shape.residual_streams}, mHC) x hidden({shape.hidden_size}) x 2 B")
    if not plan.chunked_loss:
        lb = 4 if plan.logits_fp32 else 2
        logit_tokens = plan.seq_len * plan.micro_batch_seqs / cp
        add("logits_and_grad", logit_tokens * shape.vocab_size / tp * lb * 2,
            f"T*mbs/CP x vocab({shape.vocab_size})/TP x {lb} B x 2 (last stage)")
    add("working_set_margin", plan.activation_margin_gib * GIB,
        "configured: recomputed layer working set (KDA, MoE all-to-all, attention); not measured")
    if shape.dsa_layers and plan.dsa_path == "hf_dense":
        T, b = plan.seq_len, plan.micro_batch_seqs
        dense_terms = b * T * T * 5 + b * T * shape.index_heads * -(-T // shape.index_kpool) * 4
        add("dsa_dense_terms", dense_terms,
            "mask mbs*T*T*5 B + indexer scores mbs*T*heads*T/kpool*4 B (dense DSA fallback)")
    return est


def _megatron_status(plan: TrainPlan) -> str:
    if not plan.optimizer_offload_fraction:
        return "Megatron (no optimizer offload)"
    master = " incl. FP32 master" if plan.offload_master and plan.param_bytes == 4 else ""
    return f"Megatron (optimizer offload {plan.optimizer_offload_fraction:g}{master} on the host)"


def check(shape: ModelShape, plan: TrainPlan) -> Estimate:
    """:func:`estimate`, raising :class:`MemoryPlanError` when the plan does not fit."""
    est = estimate(shape, plan)
    if problems := est.problems():
        raise MemoryPlanError(f"the training plan does not fit ({LABEL}): " + "; ".join(problems))
    return est


# ---------------------------------------------------------------------------
# Model shapes
# ---------------------------------------------------------------------------
_HEAD_KEYS = ("num_attention_heads", "num_key_value_heads", "index_n_heads")


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", None) or config


def shape_from_hf_config(config: Mapping[str, Any], *, include_vision: bool = False) -> ModelShape:
    """Count the parameters of an HF config on the meta device (no weights read).

    Needs a transformers build that knows the model type. Vision towers are
    left out unless ``include_vision`` (the trainer does not load them)."""
    import torch
    import transformers

    hf = transformers.AutoConfig.for_model(**dict(config))
    with torch.device("meta"):
        try:
            model = transformers.AutoModelForCausalLM.from_config(hf)
        except ValueError:  # a multimodal wrapper (e.g. glm5_next): count its language model
            model = transformers.AutoModelForImageTextToText.from_config(hf)
    tc = _text_config(hf)
    total, experts, per_layer = 0, 0, Counter()
    for name, p in model.named_parameters():
        parts = name.split(".")
        is_vision = "language_model" not in parts and "lm_head" not in parts and any(
            v in parts for v in ("visual", "vision_tower", "vision_model"))
        if is_vision and not include_vision:
            continue
        total += p.numel()
        if "experts" in parts and "shared_experts" not in parts:
            experts += p.numel()
        if "layers" in parts and not is_vision:
            per_layer[int(parts[parts.index("layers") + 1])] += p.numel()
    layer_types = list(getattr(tc, "layer_types", None) or [])
    linear = getattr(tc, "linear_attn_config", None) or {}
    heads = [int(getattr(tc, k)) for k in _HEAD_KEYS if getattr(tc, k, None)]
    if isinstance(linear, Mapping) and linear.get("num_heads"):
        heads.append(int(linear["num_heads"]))
    return ModelShape(
        params=total, largest_layer_params=max(per_layer.values()), num_layers=int(tc.num_hidden_layers),
        hidden_size=int(tc.hidden_size), vocab_size=int(tc.vocab_size),
        residual_streams=int(getattr(tc, "hc_mult", 1) or 1) if getattr(tc, "mhc", True) else 1,
        head_counts=tuple(sorted(set(heads))),
        dsa_layers=layer_types.count("deepseek_sparse_attention"),
        index_heads=int(getattr(tc, "index_n_heads", 0) or 0),
        index_kpool=int(getattr(tc, "index_kpool", 1) or 1),
        expert_params=experts,
        num_experts=int(getattr(tc, "n_routed_experts", None) or getattr(tc, "num_experts", None) or 0)
        if experts else 0,
        source=f"{type(model).__name__} on meta (transformers {transformers.__version__})",
    )


def plan_for_nodes(nodes: Sequence[Any], **overrides: Any) -> TrainPlan:
    """A :class:`TrainPlan` over inventory nodes (``yeto.launch.node_inventory.Node``).
    The nodes must be one card model with one host memory size."""
    if not nodes:
        raise MemoryPlanError("no trainer nodes")
    kinds = {(n.device_family, n.card, n.cards_per_node, n.hbm_gib, n.host_memory_gib) for n in nodes}
    if len(kinds) != 1:
        raise MemoryPlanError(f"trainer nodes differ (one island needs one node kind): {sorted(kinds)}")
    _, _, per_node, hbm, host = kinds.pop()
    return TrainPlan(cards=per_node * len(nodes), cards_per_node=per_node, hbm_gib=hbm,
                     host_memory_gib=host, **overrides)


def sweep(shape: ModelShape, base: TrainPlan, seq_lens: Iterable[int],
          offloads: Iterable[str] = OFFLOAD_MODES) -> list[dict[str, Any]]:
    """One summary row per (offload, seq_len)."""
    rows = []
    for offload in offloads:
        for seq in seq_lens:
            est = estimate(shape, replace(base, offload=offload, seq_len=int(seq)))
            rows.append({"offload": offload, "seq_len": int(seq), **est.summary()})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="HF config.json")
    ap.add_argument("--include-vision", action="store_true")
    ap.add_argument("--seq-lens", default="", help="comma list: print a sweep over every offload mode")
    types = {"int": int, "float": float, "str": str}
    for name, f in TrainPlan.__dataclass_fields__.items():
        flag = "--" + name.replace("_", "-")
        if f.type == "bool":
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        elif f.default is MISSING:  # cards, cards_per_node, hbm_gib, host_memory_gib
            ap.add_argument(flag, type=types[f.type], required=True)
        elif name == "seq_len":
            ap.add_argument(flag, type=int, default=None)
        else:
            ap.add_argument(flag, type=types[f.type], default=f.default)
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as fh:
        shape = shape_from_hf_config(json.load(fh), include_vision=a.include_vision)
    seqs = [int(s) for s in a.seq_lens.split(",") if s]
    values = {n: getattr(a, n) for n in TrainPlan.__dataclass_fields__}
    values["seq_len"] = values["seq_len"] or (seqs[0] if seqs else None)
    if values["seq_len"] is None:
        ap.error("--seq-len or --seq-lens is required")
    plan = TrainPlan(**values)
    if seqs:
        print(json.dumps({"shape": asdict(shape), "rows": sweep(shape, plan, seqs)}, indent=2))
    else:
        print(json.dumps(estimate(shape, plan).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
