"""Per-GPU memory peak estimate of an RL island (launch-preflight-guards, design decision 2).

An upper-bound estimate, not a simulation.  Units: GiB per GPU.  Notation:
P = parameters, V = vocabulary, L = layers, H = hidden size, T = tokens of
one micro-batch (context length x micro-batch samples), TP/PP/DP = parallel
degrees, C = logits chunk tokens (None = no chunking).

The six parts (design decision 2):

1. weights            2*P/(TP*PP)                                   (bf16)
2. grads + optimizer  full: 4*P/(TP*PP) + (1-f)*12*P/(TP*PP*DP_shard), f the
                      optimizer offload fraction (``--rl-optimizer-offload-fraction``:
                      that share of the fp32 master weights and Adam moments sits in
                      host memory); LoRA: 16*P_lora
3. activations        K_ACT * L/PP * T/(TP*CP) * H * 2
4. logits block       K_LOGIT * min(T, C) * V/TP * 4
5. inference          SGLang mem_fraction_static * GPU memory, when the rollout
                      engine shares the GPU and is NOT released during training;
                      none on a buffer trainer (``--rl-sample-role trainer`` or
                      ``--rl-buffer-shards``: the samplers run elsewhere, the
                      island has no engine)
6. fixed overhead     R0 + FRAG * (parts 1-4)   (CUDA context, NCCL, residual of
                      released engines, allocator fragmentation)

Colocated islands (train and rollout on the same GPU, the default 1-GPU
island) alternate two phases: training (engine released, offload) and
rollout (trainer offloaded).  The peak is the larger of the two phases;
the rollout phase is ``mem_fraction_static * GPU memory + R0``.

Calibration (data archived outside the repo): calibration runs b, c, d (Qwen3.5-4B LoRA r16,
one GPU per island, colocated).

* K_LOGIT: run c OOM message "Tried to allocate 7.58 GiB" = 16384 x 248320
  x 2 bytes, one bf16 logits block -> K_LOGIT = 0.5 (in units of 4 bytes).
* R0, FRAG: run c OOM message: 133.76 GiB in use, 107.75 GiB allocated by
  PyTorch, 21.04 GiB reserved but unallocated -> FRAG = 21.04/107.75,
  R0 = 133.76 - 107.75 - 21.04.
* K_ACT (no recompute): run d measured peak (tape ``peak_gpu_mem_bytes``,
  max of the two islands, 132.36e9 bytes = 123.27 GiB) solved for K_ACT.

Partial routed experts (``--tuning full`` with an ``in_place``
``--rl-expert-selection``, any MoE model): the routed experts are split over
EP ranks (weights 2*(P - P_exp + P_exp/EP)/(TP*PP)) and only the selected
experts carry grads and optimizer state (P_train/(TP*PP*EP) per GPU, the
optimizer sharded over the expert data-parallel group DP/EP when distributed);
every other parameter is frozen. A fixed partition (``--rl-placement
fixed-partition``) trains on the island's GPUs minus ``--rollout-num-gpus`` and
``--rl-standby-gpus``, with no rollout engine on them.

Only Qwen3.5-4B is calibrated; other models get the same constants and the
launcher treats an estimate for them as advisory (warn, never refuse) until
data exists (design: Risks).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

GIB = float(1 << 30)

# --- calibration constants -----------------------------------------------
K_LOGIT = 0.5
FRAG = 21.04 / 107.75          # 0.1953
R0_GIB = 133.76 - 107.75 - 21.04  # 4.97
K_ACT_NO_RECOMPUTE = 45.62     # solved from run d (see evidence/memory-backtest.json)
K_ACT_RECOMPUTE = 2.0          # unverified: no run with full recompute was measured
CALIBRATED_MODELS = frozenset({"Qwen/Qwen3.5-4B"})

# Usable GPU memory as torch reports it (GiB).  H100/H200: the calibration runs' OOM
# messages ("total capacity of 79.18 GiB" / "139.80 GiB").  Other GPUs:
# the launcher's GPU_MEM_GB nominal value x 0.98 (H100 80 -> 78.4, H200 141 -> 138.2).
MEASURED_CAPACITY_GIB = {"H100": 79.18, "H200": 139.80}


def gpu_capacity_gib(gpu: str) -> float | None:
    key = str(gpu).upper()
    if key in MEASURED_CAPACITY_GIB:
        return MEASURED_CAPACITY_GIB[key]
    from .launcher import GPU_MEM_GB

    for name, gb in GPU_MEM_GB.items():
        if name.upper() == key:
            return gb * 0.98
    return None


# --- model structure ---------------------------------------------------------
@dataclass(frozen=True)
class ModelArch:
    params: float
    layers: int
    hidden: int
    vocab: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int


def param_breakdown(cfg: dict) -> dict[str, float]:
    """Parameter count of an HF ``config.json`` by part, counting what the default
    :func:`arch_from_hf_config` estimate leaves out: MLA low-rank attention
    (``kv_lora_rank``/``q_lora_rank``), shared experts, the dense first layers of a MoE
    (``first_k_dense_replace`` / ``mlp_only_layers`` / ``decoder_sparse_step``) and a
    latent MoE (``routed_expert_hidden_size``). ``active`` counts top-k routed experts.
    MTP layers (``num_nextn_predict_layers``) are not counted."""
    t = cfg.get("text_config") or cfg
    L = int(t["num_hidden_layers"])
    H = int(t["hidden_size"])
    V = int(t["vocab_size"])
    inter = int(t.get("intermediate_size") or 4 * H)
    nh = int(t.get("num_attention_heads") or 1)
    kv = int(t.get("num_key_value_heads") or nh)
    hd = int(t.get("head_dim") or H // nh)
    tied = bool(cfg.get("tie_word_embeddings", t.get("tie_word_embeddings", False)))
    if t.get("kv_lora_rank"):
        rope = int(t.get("qk_rope_head_dim") or 0)
        nope = int(t.get("qk_nope_head_dim") or hd)
        v_hd = int(t.get("v_head_dim") or nope)
        qk = nope + rope
        q_lora = int(t.get("q_lora_rank") or 0)
        kv_lora = int(t["kv_lora_rank"])
        q = H * q_lora + q_lora * nh * qk if q_lora else H * nh * qk
        attn = q + H * (kv_lora + rope) + kv_lora * nh * (nope + v_hd) + nh * v_hd * H
    else:
        attn = H * nh * hd + 2 * H * kv * hd + nh * hd * H
    experts = int(t.get("num_experts") or t.get("n_routed_experts") or 0)
    topk = int(t.get("num_experts_per_tok") or t.get("num_experts_per_token") or 0)
    if experts:
        if t.get("first_k_dense_replace") is not None:
            dense_layers = min(L, int(t["first_k_dense_replace"]))
        elif t.get("mlp_only_layers"):
            dense_layers = len(set(t["mlp_only_layers"]))
        elif int(t.get("decoder_sparse_step") or 1) > 1:
            step = int(t["decoder_sparse_step"])
            dense_layers = L - sum(1 for i in range(L) if (i + 1) % step == 0)
        else:
            dense_layers = 0
    else:
        dense_layers = L
    moe_layers = L - dense_layers
    moe_inter = int(t.get("moe_intermediate_size") or inter)
    latent = int(t.get("routed_expert_hidden_size") or H)
    per_expert = 3 * latent * moe_inter
    latent_proj = 2 * H * latent if latent != H else 0
    n_shared = int(t.get("n_shared_experts") or t.get("num_shared_experts") or 0)
    shared_inter = int(t.get("shared_expert_intermediate_size") or 0) or n_shared * moe_inter
    shared = 3 * H * shared_inter if experts else 0
    router = H * experts
    out = {
        "embed": float((1 if tied else 2) * V * H),
        "attention": float(L * attn),
        "norms": float(L * 2 * H),
        "dense_mlp": float(dense_layers * 3 * H * inter),
        "shared_experts": float(moe_layers * shared),
        "routed_experts": float(moe_layers * per_expert * experts),
        "moe_other": float(moe_layers * (router + latent_proj)),
        "per_expert": float(per_expert),
        "moe_layers": float(moe_layers),
        "dense_layers": float(dense_layers),
    }
    base = out["embed"] + out["attention"] + out["norms"] + out["dense_mlp"] + out["shared_experts"] + out["moe_other"]
    out["total"] = base + out["routed_experts"]
    out["active"] = base + moe_layers * per_expert * min(topk or experts, experts)
    return out


def arch_from_hf_config(cfg: dict, *, precise: bool = False) -> ModelArch:
    """``precise=False`` (default) keeps the historical upper-bound count every caller
    and calibration uses; ``precise=True`` takes the parameter count of
    :func:`param_breakdown` (MLA, shared experts, dense first layers)."""
    t = cfg.get("text_config") or cfg
    L = int(t["num_hidden_layers"])
    H = int(t["hidden_size"])
    V = int(t["vocab_size"])
    inter = int(t.get("intermediate_size") or 4 * H)
    nh = int(t.get("num_attention_heads") or 1)
    kv = int(t.get("num_key_value_heads") or nh)
    hd = int(t.get("head_dim") or H // nh)
    tied = bool(cfg.get("tie_word_embeddings", t.get("tie_word_embeddings", False)))
    attn = H * nh * hd * 2 + 2 * H * kv * hd + nh * hd * H  # q (+gate) k v o, upper bound
    moe_experts = int(t.get("num_experts") or t.get("n_routed_experts") or 0)
    moe_inter = int(t.get("moe_intermediate_size") or inter)
    mlp = 3 * H * (moe_inter * moe_experts if moe_experts else inter)
    params = (1 if tied else 2) * V * H + L * (attn + mlp + 2 * H)
    if precise:
        params = int(param_breakdown(cfg)["total"])
    return ModelArch(params=float(params), layers=L, hidden=H, vocab=V, intermediate=inter,
                     heads=nh, kv_heads=kv, head_dim=hd)


# Text-model fields of the HF config.json of models used on GPU so far (copied
# from the HF cache, same revisions as the calibration runs), so a machine without
# the cache still estimates them.  Other models: local dir or HF cache only.
BUILTIN_CONFIGS: dict[str, dict] = {
    "Qwen/Qwen3-0.6B": {"num_hidden_layers": 28, "hidden_size": 1024, "intermediate_size": 3072,
                        "vocab_size": 151936, "num_attention_heads": 16, "num_key_value_heads": 8,
                        "head_dim": 128, "tie_word_embeddings": True},
    "Qwen/Qwen3.5-0.8B": {"num_hidden_layers": 24, "hidden_size": 1024, "intermediate_size": 3584,
                          "vocab_size": 248320, "num_attention_heads": 8, "num_key_value_heads": 2,
                          "head_dim": 256, "tie_word_embeddings": True},
    "Qwen/Qwen3.5-4B": {"num_hidden_layers": 32, "hidden_size": 2560, "intermediate_size": 9216,
                        "vocab_size": 248320, "num_attention_heads": 16, "num_key_value_heads": 4,
                        "head_dim": 256, "tie_word_embeddings": True},
    "Qwen/Qwen3-8B": {"num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 12288,
                      "vocab_size": 151936, "num_attention_heads": 32, "num_key_value_heads": 8,
                      "head_dim": 128, "tie_word_embeddings": False},
}


def _hf_cache_configs(repo: str) -> list[str]:
    roots = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME") and
             os.path.join(os.environ["HF_HOME"], "hub"), os.path.expanduser("~/.cache/huggingface/hub")]
    out = []
    for root in filter(None, roots):
        out += glob.glob(os.path.join(root, "models--" + repo.replace("/", "--"),
                                      "snapshots", "*", "config.json"))
    return sorted(out)


def resolve_arch(model: str) -> tuple[ModelArch | None, str, str]:
    """(arch, resolved model id, source).  Source: local dir / HF cache / "none"."""
    arch, repo, source, _ = _resolve(model)
    return arch, repo, source


def _resolve(model: str) -> tuple[ModelArch | None, str, str, dict | None]:
    """:func:`resolve_arch` plus the config it read."""
    try:
        from .models import resolve

        repo = resolve(model)
    except Exception:  # noqa: BLE001 - unknown alias: try the name itself
        repo = model
    candidates = []
    if os.path.isdir(os.path.expanduser(str(repo))):
        candidates.append(os.path.join(os.path.expanduser(str(repo)), "config.json"))
    candidates += _hf_cache_configs(str(repo))
    for path in candidates:
        try:
            cfg = json.loads(Path(path).read_text(encoding="utf-8"))
            return arch_from_hf_config(cfg), str(repo), path, cfg
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if str(repo) in BUILTIN_CONFIGS:
        return arch_from_hf_config(BUILTIN_CONFIGS[str(repo)]), str(repo), "builtin", BUILTIN_CONFIGS[str(repo)]
    return None, str(repo), "none", None


# --- request ----------------------------------------------------------------
@dataclass(frozen=True)
class MemoryRequest:
    arch: ModelArch
    gpu: str
    gpu_mem_gib: float
    tokens: int                    # T: one micro-batch, longest sequence
    tuning: str = "lora"           # "lora" | "full"
    lora_rank: int = 16
    lora_targets: str = "attention"
    tp: int = 1
    pp: int = 1
    cp: int = 1
    dp: int = 1
    distributed_optimizer: bool = False
    recompute: bool = False
    logits_chunk: int | None = None
    colocated: bool = True         # train and rollout engine share the GPU
    offload: bool = True           # engine released during training (colocated default)
    mem_fraction_static: float = 0.4
    optimizer_offload_fraction: float = 0.0  # share of the 12 B/param optimizer state in host memory
    # partial routed experts (in_place selection) only; the defaults keep every other estimate
    ep: int = 1                              # expert parallel: routed experts split over EP ranks
    expert_params: float = 0.0               # routed-expert parameters of the model
    trainable_params: float | None = None    # trainable (selected expert) parameters; None = all


def lora_params(arch: ModelArch, rank: int, targets: str) -> float:
    H, nh, kv, hd, I = arch.hidden, arch.heads, arch.kv_heads, arch.head_dim, arch.intermediate
    attn = (H + nh * hd) + 2 * (H + kv * hd) + (nh * hd + H)
    mlp = 2 * (H + I) + (I + H)
    per_layer = attn if targets == "attention" else (mlp if targets == "mlp" else attn + mlp)
    return float(rank * per_layer * arch.layers)


def estimate(req: MemoryRequest) -> dict[str, Any]:
    a = req.arch
    shard = req.tp * req.pp
    parts: dict[str, float] = {}
    partial = req.ep > 1 or req.expert_params > 0 or req.trainable_params is not None
    if partial:
        held = a.params - req.expert_params + req.expert_params / req.ep
        parts["weights"] = 2 * held / shard / GIB
    else:
        parts["weights"] = 2 * a.params / shard / GIB
    if req.tuning == "full" and partial:
        # only the selected experts train: grads and optimizer of this rank's share of them,
        # the optimizer sharded over the expert data-parallel group when distributed
        trainable = (held if req.trainable_params is None else req.trainable_params / req.ep) / shard
        edp_shard = max(1, req.dp // req.ep) if req.distributed_optimizer else 1
        kept = 1.0 - min(1.0, max(0.0, req.optimizer_offload_fraction))
        parts["grads_optimizer"] = (4 * trainable + kept * 12 * trainable / edp_shard) / GIB
    elif req.tuning == "full":
        dp_shard = req.dp if req.distributed_optimizer else 1
        kept = 1.0 - min(1.0, max(0.0, req.optimizer_offload_fraction))
        parts["grads_optimizer"] = (4 * a.params / shard + kept * 12 * a.params / (shard * dp_shard)) / GIB
    else:
        parts["grads_optimizer"] = 16 * lora_params(a, req.lora_rank, req.lora_targets) / GIB
    k_act = K_ACT_RECOMPUTE if req.recompute else K_ACT_NO_RECOMPUTE
    parts["activations"] = (k_act * a.layers / req.pp * req.tokens / (req.tp * req.cp)
                            * a.hidden * 2) / GIB
    block = req.tokens if req.logits_chunk is None else min(req.tokens, req.logits_chunk)
    parts["logits_block"] = K_LOGIT * block * a.vocab / req.tp * 4 / GIB
    train_alloc = sum(parts.values())
    engine = req.mem_fraction_static * req.gpu_mem_gib
    parts["inference"] = engine if (req.colocated and not req.offload) else 0.0
    parts["fixed_overhead"] = R0_GIB + FRAG * train_alloc
    train_phase = sum(parts.values())
    rollout_phase = (engine + R0_GIB) if req.colocated else 0.0
    if req.colocated and req.offload and rollout_phase > train_phase:
        parts["rollout_phase_extra"] = rollout_phase - train_phase
    peak = max(train_phase, rollout_phase)
    return {"parts_gib": {k: round(v, 3) for k, v in parts.items()},
            "train_phase_gib": round(train_phase, 3), "rollout_phase_gib": round(rollout_phase, 3),
            "peak_gib": round(peak, 3)}


def _round_down_1024(n: int) -> int:
    return max(1024, (n // 1024) * 1024)


def suggestions(req: MemoryRequest, limit_frac: float, *, response_len: int | None = None,
                larger_gpus: dict[str, float] | None = None) -> list[dict]:
    """Design decision 2: try three changes; keep the ones that fit."""
    out = []
    # 1. the longest context (multiple of 1024 below the request) that fits;
    #    the response length shrinks in the same ratio (rounded down to 1024).
    tokens = _round_down_1024(req.tokens - 1) if req.tokens > 1024 else None
    while tokens is not None:
        r = replace(req, tokens=tokens)
        peak = estimate(r)["peak_gib"]
        if peak <= limit_frac * req.gpu_mem_gib:
            resp = None if response_len is None else max(
                1024, _round_down_1024(response_len * tokens // req.tokens))
            text = f"context {req.tokens} -> {tokens}" + (
                f", response {response_len} -> {resp}" if resp is not None else "")
            out.append({"change": text, "peak_gib": peak, "seq_len": tokens, "response_len": resp})
            break
        tokens = tokens - 1024 if tokens > 1024 else None
    if req.tp * req.dp < 64:
        r = replace(req, tp=req.tp * 2)
        peak = estimate(r)["peak_gib"]
        if peak <= limit_frac * req.gpu_mem_gib:
            out.append({"change": f"2x GPUs of the same type (TP {req.tp} -> {req.tp * 2})",
                        "peak_gib": peak})
    for gpu, cap in sorted((larger_gpus or {}).items(), key=lambda kv: kv[1]):
        if cap <= req.gpu_mem_gib:
            continue
        r = replace(req, gpu=gpu, gpu_mem_gib=cap)
        peak = estimate(r)["peak_gib"]
        if peak <= limit_frac * cap:
            out.append({"change": f"larger GPU {req.gpu} -> {gpu} ({cap:.0f} GiB)", "peak_gib": peak})
        break  # only the next larger GPU type
    return out


def smallest_parallel_shape(req: MemoryRequest, limit_frac: float, *, max_tp: int = 8,
                            max_pp: int = 128) -> dict | None:
    """The fewest GPUs per model replica (TP <= ``max_tp`` inside a node, then PP) whose
    estimate fits at the requested context: a large model on the default TP 1 / PP 1 is
    otherwise told only "2x GPUs" or nothing. A suggestion only; expert parallelism is not
    modelled (a MoE needs at least this many GPUs, usually fewer with EP)."""
    shapes = sorted(((tp, pp) for tp in (1, 2, 4, 8) if tp <= max_tp
                     for pp in (2 ** i for i in range(8)) if pp <= max_pp),
                    key=lambda s: (s[0] * s[1], -s[0]))
    for tp, pp in shapes:
        if tp * pp <= req.tp * req.pp:
            continue
        peak = estimate(replace(req, tp=tp, pp=pp))["peak_gib"]
        if peak <= limit_frac * req.gpu_mem_gib:
            return {"change": f"model parallel TP {tp} x PP {pp} = {tp * pp} GPUs per replica "
                              f"(--tensor-parallel {tp} --pipeline-parallel {pp}; "
                              f"{-(-tp * pp // 8)} node(s) of 8)",
                    "peak_gib": peak, "tensor_parallel": tp, "pipeline_parallel": pp}
    return None


def _larger_gpus() -> dict[str, float]:
    from .launcher import GPU_MEM_GB

    caps = {}
    for name in GPU_MEM_GB:
        cap = gpu_capacity_gib(name)
        if cap is not None:
            caps[name] = cap
    return caps


def partial_experts_shape(args, cfg: dict | None) -> dict[str, float] | None:
    """``{"ep", "expert_params", "trainable_params"}`` of a generic partial-experts launch
    (``--tuning full`` with an ``in_place`` ``--rl-expert-selection``), else None. Best
    effort: an unreadable selection or config gives None (the launch checks report it)."""
    path = getattr(args, "rl_expert_selection", None)
    if not path or cfg is None or (getattr(args, "tuning", "lora") or "lora") != "full":
        return None
    try:
        from .rl.engine import backends

        selection = backends.module("expert_selection", "miles").ExpertSelection.load_file(
            os.path.expanduser(str(path)))
        if selection.layout != "in_place":
            return None  # kimi_k31 slot_order: estimated by yeto.launch.k31_checks
        breakdown = param_breakdown(cfg)
        selected = sum(len(ids) for ids in selection.trainable.values())
    except Exception:  # noqa: BLE001 - advisory estimate
        return None
    ep = int(getattr(args, "expert_parallel", 0) or 0) or int(selection.ep_size)
    return {"ep": max(1, ep), "expert_params": float(breakdown["routed_experts"]),
            "trainable_params": float(breakdown["per_expert"] * selected)}


def request_from_args(args, spec, arch: ModelArch, cfg: dict | None = None) -> MemoryRequest:
    """Launcher args + one island ClusterSpec -> MemoryRequest (one GPU of the island)."""
    cap = gpu_capacity_gib(spec.gpu)
    total = int(getattr(spec, "total_gpus", 1) or 1)
    # GPUs of the island that do not train: a fixed partition's rollout and standby GPUs
    rollout_gpus = 0
    if getattr(args, "rl_placement", "colocated") == "fixed-partition":
        rollout_gpus = (int(getattr(args, "rollout_num_gpus", 0) or 0)
                        + int(getattr(args, "rl_standby_gpus", 0) or 0))
    train_gpus = max(1, total - rollout_gpus)
    tp = int(getattr(args, "tensor_parallel", 1) or 1)
    pp = int(getattr(args, "pipeline_parallel", 1) or 1)
    seq = int(getattr(args, "seq_len", 2048) or 2048)
    resp = getattr(args, "rollout_max_response_len", None)
    tokens = max(seq, int(resp)) if (resp and getattr(args, "training_mode", "sft") == "rl") else seq
    model = str(getattr(args, "model", ""))
    # a buffer trainer reads groups the samplers wrote elsewhere: no rollout engine on its GPUs
    buffer_trainer = (str(getattr(args, "rl_sample_role", "") or "") == "trainer"
                      or bool(getattr(args, "rl_buffer_shards", None)))
    try:
        offload_fraction = float(getattr(args, "rl_optimizer_offload_fraction", 0.0) or 0.0)
    except (TypeError, ValueError):
        offload_fraction = 0.0
    return MemoryRequest(
        arch=arch, gpu=str(spec.gpu), gpu_mem_gib=float(cap or 0), tokens=tokens,
        tuning=getattr(args, "tuning", "lora") or "lora",
        lora_rank=int(getattr(args, "lora_r", 16) or 16),
        lora_targets=str(getattr(args, "lora_targets", "attention") or "attention"),
        tp=tp, pp=pp, dp=max(1, train_gpus // (tp * pp)),
        recompute="deepseek" in model.lower() or bool(getattr(args, "rl_activation_recompute", False)),
        colocated=rollout_gpus == 0 and not buffer_trainer,
        offload=True if rollout_gpus == 0 or buffer_trainer else bool(getattr(args, "rl_offload_train", False)),
        mem_fraction_static=float(getattr(args, "sglang_mem_fraction_static", 0.4) or 0.4),
        optimizer_offload_fraction=offload_fraction,
        **(partial_experts_shape(args, cfg) or {}),
    )


def estimate_for_launch(args, spec, *, margin: float) -> dict[str, Any]:
    """The per-island record used by the memory preflight, dry-run and manifest."""
    model = str(getattr(args, "model", ""))
    base = {"gpu": str(spec.gpu), "model": model, "margin": margin}
    if getattr(args, "training_mode", "sft") != "rl":
        return {**base, "estimated": False, "reason": "only RL islands are estimated"}
    cap = gpu_capacity_gib(spec.gpu)
    if cap is None:
        reason = f"GPU {spec.gpu!r} memory unknown"
        if str(spec.gpu).upper().startswith("910"):
            reason += (" (Ascend 910x cards are not in the GPU memory table: size the island by "
                       "hand from npu-smi info, e.g. weights 2 bytes/param / (TP x PP) plus "
                       "the vllm-ascend sleep-level-1 engine share)")
        return {**base, "estimated": False, "reason": reason}
    arch, repo, source, cfg = _resolve(model)
    if arch is None:
        return {**base, "estimated": False,
                "reason": f"model structure of {repo!r} unknown (no local or cached config.json)"}
    req = request_from_args(args, spec, arch, cfg)
    est = estimate(req)
    limit = margin * cap
    fits = est["peak_gib"] <= limit
    calibrated = repo in CALIBRATED_MODELS
    rec = {**base, "estimated": True, "model_id": repo, "arch_source": source,
           "calibrated": calibrated, "gpu_mem_gib": cap, "limit_gib": round(limit, 3),
           "tokens": req.tokens, "tuning": req.tuning, "tp": req.tp, "pp": req.pp, "dp": req.dp,
           "colocated": req.colocated, **est, "fits": fits}
    if req.trainable_params is not None:
        rec.update(ep=req.ep, trainable_params=req.trainable_params)
    if not fits:
        resp = getattr(args, "rollout_max_response_len", None)
        rec["suggestions"] = suggestions(req, margin, response_len=int(resp) if resp else None,
                                         larger_gpus=_larger_gpus())
        _print_parallel_shape_hint(req, margin, rec["suggestions"])
    return rec


def _print_parallel_shape_hint(req: MemoryRequest, limit_frac: float, offered: list[dict]) -> None:
    """Say on stderr the smallest TP x PP that fits, when "2x GPUs" is not already offered.
    Text only: kept out of ``suggestions`` so the dry-run JSON and the manifest are unchanged."""
    if any("GPUs" in s["change"] for s in offered):
        return
    shape = smallest_parallel_shape(req, limit_frac)
    if shape is not None:
        print(f"[preflight] hint: {shape['change']} (estimated peak {shape['peak_gib']:.1f} GiB; "
              "heads/layers divisibility not checked)", file=sys.stderr, flush=True)
