"""Launch-time checks of a Kimi-K3.1 full-parameter (partial-experts) run.

Both run before any cloud resource (``yeto launch`` and ``--dry-run``):

* :func:`check_expert_selection` loads ``--rl-expert-selection`` through the
  same validator the selection tool's ``--validate`` uses and checks it against
  the launch (model, ``--expert-parallel``, the checkpoint's MoE layers when its
  ``config.json`` is readable). Any trainable count that EP divides is
  accepted (64 and 128 of 896 are the usual forms).
* :func:`memory_estimate` is the per-GPU allocated peak of every pipeline
  stage of the full form (weights, full-recompute checkpoints with the
  attention-residual bank, one layer's recompute + backward working set),
  calibrated on measured runs (``MEMORY_CALIBRATION``, within about 6 %). It
  never refuses a launch; it returns warnings the launcher prints when the
  busiest stage nears or passes what runs on the GPU.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

# bytes per trainable parameter: bf16 weight + fp32 grad + fp32 master/m/v (DP 1: no sharding)
TRAINABLE_BYTES = 2 + 4 + 12
# MXFP4: 4-bit values + one 8-bit scale per 32 values
MXFP4_BYTES = 4.25 / 8
# non-expert parameters of one K3.1 decoder layer (attention + shared expert), rough
NON_EXPERT_PER_LAYER = 0.33e9
LONG_CONTEXT = 65536


def _model_config(model: str) -> dict[str, Any] | None:
    path = Path(os.path.expanduser(str(model)))
    candidates = [path / "config.json"] if path.is_dir() else []
    if not candidates:
        try:
            from ..memory_estimate import _hf_cache_configs

            candidates = [Path(p) for p in _hf_cache_configs(str(model))]
        except Exception:  # noqa: BLE001 - the cache lookup is best effort
            candidates = []
    for candidate in candidates:
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    text = cfg.get("text_config")
    return text if isinstance(text, dict) else cfg


def check_expert_selection(args) -> dict[str, Any] | None:
    """Validate ``--rl-expert-selection`` against the launch; returns a summary."""
    path = getattr(args, "rl_expert_selection", None)
    if not path:
        return None
    from ..rl.engine import backends

    k31 = backends.module("kimi_k31_model", "miles")
    sel = backends.module("expert_selection", "miles")

    hint = ("generate one with python -m yeto.rl.adapters.miles.models.partial_experts.selection "
            "--model kimi_k31 --num-trainable 64 --ep-size <EP> ... and check it with --validate")
    try:
        selection = sel.ExpertSelection.load_file(os.path.expanduser(str(path)))
    except FileNotFoundError:
        raise ValueError(f"--rl-expert-selection {path}: no such file; {hint}") from None
    except (sel.SelectionError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"--rl-expert-selection {path} is not a valid selection: {exc}; {hint}") from None
    if selection.model != k31.MODEL_TYPE:
        raise ValueError(f"--rl-expert-selection is for model {selection.model!r}, not {k31.MODEL_TYPE!r}")
    ep = int(getattr(args, "expert_parallel", 0) or 0)
    if ep and selection.ep_size != ep:
        raise ValueError(
            f"--rl-expert-selection was built for EP {selection.ep_size} but the launch uses "
            f"--expert-parallel {ep}; re-derive the slot order with the selection tool "
            f"(--input {path} --ep-size {ep})")
    cfg = _model_config(getattr(args, "model", ""))
    warnings: list[str] = []
    if cfg is not None:
        text = _text_config(cfg)
        total = text.get("num_experts", text.get("n_routed_experts"))
        layers = text.get("num_hidden_layers")
        dense = int(text.get("first_k_dense_replace", 1) or 0)
        if total is not None and int(total) != selection.num_experts:
            raise ValueError(
                f"--rl-expert-selection covers {selection.num_experts} experts per layer; the "
                f"checkpoint has {total}")
        if layers is not None:
            try:
                selection.check_layers(range(dense, int(layers)))
            except sel.SelectionError as exc:
                raise ValueError(f"--rl-expert-selection does not match the checkpoint: {exc}") from None
    else:
        if selection.num_experts != k31.NUM_EXPERTS or len(selection.trainable) != k31.NUM_MOE_LAYERS:
            warnings.append(
                f"--rl-expert-selection covers {len(selection.trainable)} MoE layers x "
                f"{selection.num_experts} experts; Kimi-K3.1-Pro has {k31.NUM_MOE_LAYERS} x {k31.NUM_EXPERTS}. "
                "The model's config.json is not readable here, so the layer check runs on the island")
    return {
        "sha256": selection.sha256,
        "num_trainable": selection.num_trainable,
        "num_experts": selection.num_experts,
        "ep_size": selection.ep_size,
        "moe_layers": len(selection.trainable),
        "warnings": warnings,
    }


# --- per-GPU memory model of the full form (calibrated, see memory_estimate) ----------------------
# non-expert bytes of one layer per GPU, GiB = replicated + split / TP. Measured layer_static of
# trainer-1m d1 (TP8) and c1 (TP4, its expert bytes taken out); the split part follows 1 / TP.
_NON_EXPERT_TP8 = {"dense": 0.2741, "kda": 0.2433, "mla": 0.2311, "emb": 0.2734}
_NON_EXPERT_TP4 = {"dense": 0.5464, "kda": 0.3770, "mla": 0.3124, "emb": 0.5469}
# One layer's recompute + backward working set (MoE dispatch/GEMMs, attention, indexer, the loss
# phase) per 32768 tokens per GPU, linear in tokens per GPU; the frozen experts' bf16 dequantised
# copy of that layer is added on top. RESIDUAL: what every stage holds beyond weights, checkpoints
# and that working set (CUDA/NCCL tensors, CP gathers, p2p buffers, small persistents). Both fitted
# min-max over the allocated peaks in MEMORY_CALIBRATION (max error 5.8 %).
TRANSIENT_GIB_PER_32K = 70.0
RESIDUAL_GIB = 19.5
ATTN_RES_BLOCK = 12  # attn_res_block_size: layer l % 12 == 0 starts a new bank generation
# B300 as torch sees it, and what is not torch's (CUDA context, NCCL): measured (trainer-1m)
TORCH_VISIBLE_GIB = {"B300": 267.68}
NON_TORCH_GIB = 22.3
# the highest allocated peak that ran WITHOUT expandable segments on a B300 (d1: nvidia-smi 264.4 /
# 267.7), as a fraction of the torch-visible memory; with expandable segments the allocator keeps
# about ES_SLACK_GIB above the allocated peak (historical 524k run: reserved - allocated 16.7-21.6)
NO_ES_ALLOC_FRACTION = 166.3 / 267.68
ES_SLACK_GIB = 17.0
ES_MARGIN_GIB = 15.0
THIN_MARGIN = 0.90  # warn from this fraction of the no-ES budget on (model error up to ~6 %)
# (name, num_layers, stage layers [a, b), in-flight microbatches, tokens/GPU, TP, EP, trainable,
#  measured allocated peak GiB): trainer-1m d1/a3/c1, mfu-16l a0b (rank 8, real routing), the
#  historical 524k/CP2 PP3 32/28/33 run (balanced mock routing)
MEMORY_CALIBRATION = (
    ("hist-524k-stage0", 93, (0, 32), 3, 32768, 8, 16, 128, 224.5),
    ("hist-524k-stage1", 93, (32, 60), 2, 32768, 8, 16, 128, 209.5),
    ("hist-524k-stage2", 93, (60, 93), 1, 32768, 8, 16, 128, 209.6),
    ("d1-l16-tp8cp1-ep8", 16, (0, 16), 1, 32768, 8, 8, 64, 165.84),
    ("a0b-l16-tp8cp2-ep16", 16, (0, 16), 1, 16384, 8, 16, 128, 98.92),
    ("a3-l4-tp8cp1-ep8", 4, (0, 4), 1, 32768, 8, 8, 64, 105.61),
    ("c1-l4-tp4cp2-ep8", 4, (0, 4), 1, 32768, 4, 8, 32, 102.8),
)


def _layer_kind(layer: int, num_layers: int) -> str:
    """K3.1 layer layout: layer 0 dense, every 4th (and the last) MLA + DSA, the rest KDA."""
    if layer == 0:
        return "dense"
    return "mla" if (layer + 1) % 4 == 0 or layer == num_layers - 1 else "kda"


def _non_expert_gib(kind: str, tp: int) -> float:
    split = (_NON_EXPERT_TP4[kind] - _NON_EXPERT_TP8[kind]) * 8
    return _NON_EXPERT_TP8[kind] - split / 8 + split / tp


def checkpoint_rows(a: int, b: int) -> float:
    """Checkpointed rows (units of one hidden state of a microbatch) a stage of layers a..b-1
    keeps per in-flight microbatch under full recompute: one hidden per layer, every attention-
    residual bank generation still referenced, and, past the first stage, the received packed
    [prefix, bank] input plus its unpacked bank copy."""
    gens, entry = [], 0.0
    rows_in = math.ceil(a / ATTN_RES_BLOCK)
    if a > 0:
        entry = 1 + rows_in
        gens.append(rows_in)
    gens += [layer // ATTN_RES_BLOCK + 1 for layer in range(a, b) if layer % ATTN_RES_BLOCK == 0]
    return (b - a) + sum(gens) + entry


def stage_memory(a: int, b: int, *, num_layers: int, in_flight: int, tokens_per_gpu: float, tp: int,
                 ep: int, num_trainable: int, trainable_bytes: float = 6.0,
                 num_experts: int = 896, expert_params: int = 33_030_144,
                 hidden: int = 7168) -> dict[str, float]:
    """Allocated peak (GiB) of one pipeline stage holding layers a..b-1 of a ``num_layers`` model."""
    gib = 1024 ** 3
    frozen = (num_experts - num_trainable) / ep
    weights = 0.0
    for layer in range(a, b):
        kind = _layer_kind(layer, num_layers)
        weights += _non_expert_gib(kind, tp)
        if kind != "dense":
            weights += (frozen * expert_params * MXFP4_BYTES
                        + num_trainable / ep * expert_params * trainable_bytes) / gib
    weights += _non_expert_gib("emb", tp) * ((a == 0) + (b == num_layers))
    row = tokens_per_gpu * hidden * 2 / gib
    checkpoints = in_flight * checkpoint_rows(a, b) * row
    transient = TRANSIENT_GIB_PER_32K * tokens_per_gpu / 32768 + frozen * expert_params * 2 / gib
    peak = weights + checkpoints + transient + RESIDUAL_GIB
    return {"layers": b - a, "in_flight": in_flight, "weights": weights, "checkpoints": checkpoints,
            "transient": transient, "residual": RESIDUAL_GIB, "peak_alloc": peak}


def pipeline_split(num_layers: int, pp: int, first: int | None = None,
                   last: int | None = None) -> list[int]:
    """Layers per stage: Megatron's first / last stage layer counts (``first``, ``last``) with
    equal middle stages, else an even split (the remainder on the first stages)."""
    if pp <= 1:
        return [num_layers]
    if first is not None or last is not None:
        if pp == 2:
            first = first if first is not None else num_layers - last
            return [first, num_layers - first]
        f = first if first is not None else 0
        l_ = last if last is not None else 0
        unknown = (first is None) + (last is None)
        middle_stages = pp - 2 + unknown
        rest = num_layers - f - l_
        each = rest // middle_stages if middle_stages else 0
        split = ([f] if first is not None else []) + [each] * middle_stages + ([l_] if last is not None else [])
        split[-1 if last is None else -2] += rest - each * middle_stages
        return split
    base, extra = divmod(num_layers, pp)
    return [base + (1 if s < extra else 0) for s in range(pp)]


def _env_settings(args) -> dict[str, str]:
    """The island environment the launch sets (--modal-env K=V), over this process's."""
    env = {k: v for k, v in os.environ.items()
           if k in ("YETO_MILES_TRAIN_EXTRA_ARGV", "PYTORCH_CUDA_ALLOC_CONF")}
    for item in getattr(args, "modal_env", None) or ():
        key, sep, value = str(item).partition("=")
        if sep:
            env[key] = value
    return env


def _split_flags(raw: str) -> tuple[int | None, int | None]:
    import shlex

    try:
        tokens = shlex.split((raw or "").replace(",", " "))
    except ValueError:
        return None, None
    found: dict[str, int] = {}
    for i, token in enumerate(tokens):
        name, sep, value = token.partition("=")
        if name in ("--decoder-first-pipeline-num-layers", "--decoder-last-pipeline-num-layers"):
            value = value if sep else (tokens[i + 1] if i + 1 < len(tokens) else "")
            try:
                found[name] = int(value)
            except ValueError:
                pass
    return (found.get("--decoder-first-pipeline-num-layers"),
            found.get("--decoder-last-pipeline-num-layers"))


def torch_visible_gib(gpu: str | None) -> float | None:
    if not gpu:
        return None
    key = str(gpu).upper()
    if key in TORCH_VISIBLE_GIB:
        return TORCH_VISIBLE_GIB[key]
    from ..memory_estimate import gpu_capacity_gib

    capacity = gpu_capacity_gib(gpu)
    # B300: catalog 282.2, torch sees 267.68 (0.948); the same share assumed elsewhere
    return None if capacity is None else capacity * 267.68 / 282.24


def memory_estimate(args, *, num_trainable: int, moe_layers: int, gpu: str | None) -> dict[str, Any]:
    """Per-GPU allocated peak of the K3.1 full form on every pipeline stage.

    Model (trainer-1m MODEL.md, the stage model of extrapolate.py): measured non-expert bytes per
    layer and TP, MXFP4 frozen + bf16 trainable experts (with their grads and the device share of
    the Adam state), full-recompute checkpoints (one hidden per layer plus the attention-residual
    bank rows, times the 1F1B microbatches in flight on the stage), one layer's recompute + backward
    working set and a residual, both fitted on MEMORY_CALIBRATION (max error 5.8 %). The busiest
    stage is compared with what runs on the GPU: without expandable segments the highest allocated
    peak seen to run on a B300 (166.3 GiB of 267.7), with them the visible memory less the non-torch
    share, the allocator slack and a margin."""
    tp = max(1, int(getattr(args, "tensor_parallel", 1) or 1))
    pp = max(1, int(getattr(args, "pipeline_parallel", 1) or 1))
    ep = max(1, int(getattr(args, "expert_parallel", 1) or 1))
    cp = max(1, int(getattr(args, "context_parallel", 1) or 1))
    seq = int(getattr(args, "seq_len", 0) or 0)
    response = int(getattr(args, "rollout_max_response_len", 0) or 0)
    per_cp_rank = int(getattr(args, "rl_max_tokens_per_gpu", 0) or 0) or seq / cp
    tokens_per_gpu = per_cp_rank / tp  # sequence parallel over TP
    num_layers = moe_layers + 1  # first_k_dense_replace = 1
    env = _env_settings(args)
    first, last = _split_flags(env.get("YETO_MILES_TRAIN_EXTRA_ARGV", ""))
    split = pipeline_split(num_layers, pp, first, last)
    # --rl-optimizer-offload-fraction F keeps F of the FP32 master/m/v on the host;
    # --rl-precision-aware-optimizer: bf16 main grads (2 bytes) and bf16 moments on the device
    offload = min(1.0, max(0.0, float(getattr(args, "rl_optimizer_offload_fraction", 0.0) or 0.0)))
    precision_aware = bool(getattr(args, "rl_precision_aware_optimizer", False))
    trainable_bytes = 2 + (2 if precision_aware else 4) + (6 if precision_aware else 12) * (1.0 - offload)
    stages, a = [], 0
    for s, layers in enumerate(split):
        stages.append(stage_memory(a, a + layers, num_layers=num_layers, in_flight=pp - s,
                                   tokens_per_gpu=tokens_per_gpu, tp=tp, ep=ep,
                                   num_trainable=num_trainable, trainable_bytes=trainable_bytes))
        a += layers
    busiest = max(range(len(stages)), key=lambda i: stages[i]["peak_alloc"])
    peak = stages[busiest]["peak_alloc"]
    visible = torch_visible_gib(gpu)
    es_on = "expandable_segments:true" in env.get("PYTORCH_CUDA_ALLOC_CONF", "").replace(" ", "").lower()
    budgets = None
    warnings = []
    shape = (f"Kimi-K3.1 full form ({num_trainable} trainable experts/layer, {tokens_per_gpu:.0f} tokens "
             f"per GPU, TP{tp} PP{pp} ({'/'.join(map(str, split))}) EP{ep} CP{cp})")
    if visible is not None:
        budgets = {"no_expandable_segments": visible * NO_ES_ALLOC_FRACTION,
                   "expandable_segments": visible - NON_TORCH_GIB - ES_SLACK_GIB - ES_MARGIN_GIB}
        no_es, with_es = budgets["no_expandable_segments"], budgets["expandable_segments"]
        where = f"stage {busiest} peaks at about {peak:.0f} GiB allocated per GPU"
        if peak > with_es:
            warnings.append(
                f"{shape}: {where}, above what fits on {gpu} even with expandable segments "
                f"({with_es:.0f} GiB); raise --context-parallel or --pipeline-parallel, rebalance the "
                "split, lower --seq-len, or train fewer experts")
        elif peak > no_es and not es_on:
            warnings.append(
                f"{shape}: {where}, above the {no_es:.0f} GiB that ran on {gpu} without expandable "
                "segments; set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (--modal-env) or "
                "shrink the stage")
        elif peak > THIN_MARGIN * (with_es if es_on else no_es):
            limit = with_es if es_on else no_es
            warnings.append(
                f"{shape}: {where}, {limit - peak:+.0f} GiB from the {limit:.0f} GiB budget "
                f"({'with' if es_on else 'without'} expandable segments; the model is within about 6 %); "
                "watch nvidia-smi on that stage and stop before it runs out")
    if seq >= LONG_CONTEXT and response and response * 4 < seq:
        warnings.append(
            f"--seq-len {seq} budgets memory for {seq} tokens but --rollout-max-response-len is "
            f"{response}; size the response length from the sequence length (prompt + response "
            "<= seq-len)")
    if seq >= LONG_CONTEXT and cp == 1:
        warnings.append(
            f"--seq-len {seq} with --context-parallel 1 keeps the whole sequence on every TP group; "
            "the checkpoint and working-set terms above grow linearly with it")
    from ..rl.engine import backends

    k31 = backends.module("kimi_k31_model", "miles")
    moe_on_busiest = sum(1 for layer in range(sum(split[:busiest]), sum(split[:busiest + 1]))
                         if _layer_kind(layer, num_layers) != "dense")
    host_optimizer = moe_on_busiest * num_trainable / ep * k31.EXPERT_PARAMETERS * (12 * offload) / 1024 ** 3
    rnd = lambda v: round(v, 1)  # noqa: E731
    return {
        "host_optimizer_gib_per_gpu": rnd(host_optimizer),
        "estimated": True,
        "rough": False,
        "calibrated": True,
        "per_gpu_gib": rnd(peak),
        "busiest_stage": busiest,
        "pipeline_split": split,
        "tokens_per_gpu": int(tokens_per_gpu),
        "stages_gib": [{k: (rnd(v) if isinstance(v, float) else v) for k, v in st.items()} for st in stages],
        "capacity_gib": None if visible is None else rnd(visible),
        "budget_gib": None if budgets is None else {k: rnd(v) for k, v in budgets.items()},
        "expandable_segments": es_on,
        "parts_gib": {k: rnd(stages[busiest][k]) for k in ("weights", "checkpoints", "transient", "residual")},
        "assumptions": [
            "per_gpu_gib is the torch allocated peak of the busiest pipeline stage (not nvidia-smi: "
            "add the allocator's reserve and about 22 GiB of CUDA/NCCL)",
            "1F1B: stage s holds PP - s microbatches of checkpoints",
            "the working set is linear in tokens per GPU (calibrated at TP4/TP8, 16384-32768 tokens)",
            "layer kinds follow the full K3.1 layout by index (a cut model's attention layers may differ)",
            "no activation offload, no CUDA graphs",
        ],
        "warnings": warnings,
    }


def check_full_form(args, specs) -> dict[str, Any] | None:
    """Selection check + memory estimate of a K3.1 ``--tuning full`` launch.

    Prints every warning to stderr and returns the record the dry-run plan
    carries (``kimi_k31_full``)."""
    import sys

    if getattr(args, "rl_model_recipe", None) != "kimi_k31" or getattr(args, "tuning", None) != "full":
        return None
    summary = check_expert_selection(args)
    if summary is None:
        return None
    gpu = specs[0].gpu if specs else None
    estimate = memory_estimate(args, num_trainable=summary["num_trainable"],
                               moe_layers=summary["moe_layers"], gpu=gpu)
    from ..cost.guard import K31_FULL_LOAD_GRACE_MINUTES, default_load_grace_minutes

    given = getattr(args, "load_grace_minutes", None)
    grace = float(given) if given is not None else default_load_grace_minutes(args)
    load_warnings = []
    if grace < K31_FULL_LOAD_GRACE_MINUTES and not getattr(args, "no_cost_guard", False):
        load_warnings.append(
            f"--load-grace-minutes {grace:g}: a Kimi-K3.1 full-parameter trainer loads silently for "
            f"about 30-50 min before its first trained round; the cost guard's idle rule may stop it "
            f"(default for this recipe {K31_FULL_LOAD_GRACE_MINUTES:g})")
    record = {"expert_selection": summary, "memory": estimate,
              "cost_guard": {"load_grace_minutes": grace, "warnings": load_warnings}}
    for message in (*summary["warnings"], *estimate["warnings"], *load_warnings):
        print(f"[launcher] warning: {message}", file=sys.stderr)
    return record


__all__ = ["check_expert_selection", "check_full_form", "memory_estimate"]
