"""S25 T-G3 GPU harness: real Miles/Megatron glm5_next (4-layer mini) on 2 GPUs, one torchrun per phase.

Phases (``--phase``):

* ``n75``   -- N7.5: TP2+SP, EP2. Builds the model with the TE spec (``--impl te``) or the TE-free
               fallback (``--impl local``), loads the weights saved by the first TE run (names mapped),
               runs one forward/backward on a fixed packed sequence, saves the gathered logits, the
               label log-probs and every parameter gradient, and times forward+backward (whole step and
               the routed-expert module). Then runs N8.2 on the same weights (``n82`` below).
* ``n71d``  -- N7.1d: TP1, DP2, distributed optimizer + precision-aware CPU offload (real
               HybridDeviceOptimizer: CUDA streams, pinned host memory, d2h/h2d overlap). Sub-modes
               ``cont5`` (5 steps), ``run3`` (3 steps, then the yeto cut plugin saves a cut) and
               ``resume2`` (fresh process: the cut plugin restores, N7.6 assertions, 2 steps).

N8.2 (inside ``n75``): the reference gathers every weight to HF names through Miles' own weight-update
path (``named_params_and_buffers`` -> ``all_gather_params_async`` -> ``convert_to_hf``) and quantizes it
in one process; the distributed path gives each rank's experts through the yeto Megatron adapter
(``megatron_expert_slices``) and the non-expert weights once (rank 0, ``hf_slices``) to
``requantize_distributed``. Both outputs are compared byte for byte, for W4A8 (R2 AWQ) and MXFP4.

Writes everything under ``--out``. Not a test: run by ``modal_g3.py`` on Modal ``H100!:2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
MILES = Path(os.environ.get("G3_MILES", "/g3/miles"))
sys.path.insert(0, str(MILES / "scripts" / "models"))
sys.path.insert(0, str(MILES / "miles" / "utils" / "external_utils"))  # model_args_utils

SEQ = 1024  # tokens of the packed sequence (one sample)
TIMING_SEQ = 4096
VOCAB = 16384
HIDDEN = 1024
EXPERTS = 8


# ----------------------------------------------------------------------------------------------
# Mini HF config (the 45-layer fixture shrunk: 4 layers = 3 KDA + 1 DSA, 1 dense + 3 MoE)
# ----------------------------------------------------------------------------------------------

def write_hf_config(src: Path, dst: Path) -> None:
    cfg = json.loads(src.read_text())
    cfg.pop("quantization_config", None)
    t = cfg["text_config"]
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"]
    t.update(
        hidden_size=HIDDEN, intermediate_size=3 * HIDDEN, moe_intermediate_size=512, num_hidden_layers=4,
        first_k_dense_replace=1, n_routed_experts=EXPERTS, num_experts_per_tok=2, vocab_size=VOCAB,
        index_topk=256, layer_types=layer_types, indexer_types=["full"] * 4,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"], num_nextn_predict_layers=0, pad_token_id=0,
    )
    t["linear_attn_config"] = dict(t["linear_attn_config"], num_heads=16, kda_layers=[0, 1, 2], full_attn_layers=[3])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))


def megatron_argv(impl: str, *, tp: int, ep: int, hf: Path, offload: bool, lr: float) -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("glm53_flash_args", MILES / "scripts" / "models" / "glm5.3-flash.py")
    glm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(glm)
    model = glm.model_args(nlayers=4, first_k_dense_replace=1,
                           transformer_impl="transformer_engine" if impl == "te" else "local").split()
    overrides = [
        "--hidden-size", str(HIDDEN), "--ffn-hidden-size", str(3 * HIDDEN), "--num-experts", str(EXPERTS),
        "--moe-router-topk", "2", "--moe-ffn-hidden-size", "512", "--moe-shared-expert-intermediate-size", "512",
        "--vocab-size", str(VOCAB),
    ]
    par = ["--tensor-model-parallel-size", str(tp), "--pipeline-model-parallel-size", "1",
           "--context-parallel-size", "1", "--expert-model-parallel-size", str(ep),
           "--expert-tensor-parallel-size", "1"]
    if tp > 1:
        par.append("--sequence-parallel")
    misc = [
        "--qkv-format", "thd", "--micro-batch-size", "1", "--seq-length", str(TIMING_SEQ),
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0", "--tokenizer-type", "NullTokenizer",
        "--hf-checkpoint", str(hf), "--load", "/nonexistent-g3", "--seed", "1234",
        "--optimizer", "adam", "--lr", str(lr), "--lr-decay-style", "constant", "--min-lr", "0",
        "--weight-decay", "0.1", "--adam-beta1", "0.9", "--adam-beta2", "0.98", "--clip-grad", "1.0",
        "--attention-backend", "auto", "--no-check-for-nan-in-loss-and-grad", "--distributed-timeout-minutes", "10",
    ]
    if offload:
        misc += ["--optimizer-cpu-offload", "--use-precision-aware-optimizer", "--overlap-cpu-optimizer-d2h-h2d",
                 "--optimizer-offload-fraction", "1.0"]
    return model + overrides + par + misc


def init_miles(argv: list[str], *, global_batch: int, rollouts: int):
    from megatron.training.arguments import parse_args, validate_args
    from miles.backends.megatron_utils.arguments import set_default_megatron_args
    from miles.backends.megatron_utils.initialize import init
    from miles.utils.arguments import get_miles_extra_args_provider, hf_validate_args
    from miles.utils.hf_utils.config import load_hf_config

    sys.argv = ["g3"] + argv + ["--global-batch-size", str(global_batch)]
    args = parse_args(extra_args_provider=get_miles_extra_args_provider())
    hf_config = load_hf_config(args.hf_checkpoint)
    hf_validate_args(args, hf_config)
    text = getattr(hf_config, "text_config", None) or hf_config
    args.indexer_rope_interleave = bool(getattr(text, "indexer_rope_interleave", False))
    args.compress_ratios = None
    # Miles' scheduler sizing (get_optimizer_param_scheduler): train_iters = rollouts*rbs*n // gbs
    args.num_rollout, args.rollout_batch_size, args.n_samples_per_prompt = rollouts, 1, global_batch
    args.__dict__.setdefault("decrease_batch_size_if_needed", False)
    set_default_megatron_args(args)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    validate_args(args)
    init(args)
    return args


def build(args):
    from miles.backends.megatron_utils.model import setup_model_and_optimizer, setup_train_iteration_config

    model, optimizer, scheduler = setup_model_and_optimizer(args, "actor")
    setup_train_iteration_config(args, model, optimizer, False)
    return model, optimizer, scheduler


# ----------------------------------------------------------------------------------------------
# Data, forward/backward
# ----------------------------------------------------------------------------------------------

def tokens_for(step: int, dp_rank: int, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * step + dp_rank)
    return torch.randint(0, VOCAB, (n,), generator=g)


def packed(tokens: torch.Tensor):
    from megatron.core.packed_seq_params import PackedSeqParams

    n = tokens.numel()
    cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    psp = PackedSeqParams(cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=n, max_seqlen_kv=n, qkv_format="thd")
    return tokens.view(1, -1).cuda(), psp


def run_fwd_bwd(model, tokens: torch.Tensor, *, capture: list | None = None, forward_only=False):
    """One micro-batch through Megatron's schedule; loss = mean next-token CE over the sequence."""
    from megatron.core import tensor_parallel
    from megatron.core.pipeline_parallel import get_forward_backward_func

    ids, psp = packed(tokens)
    labels = torch.roll(ids, -1, dims=1)
    mask = torch.ones_like(ids, dtype=torch.float32)
    mask[:, -1] = 0

    def loss_func(output):
        if capture is not None:
            logits = output.detach().float().clone()  # vocab_parallel_cross_entropy edits its input in place
        token_loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
        loss = (token_loss * mask).sum() / mask.sum()
        if capture is not None:
            capture.append((logits, token_loss.detach()))
        return loss, {"loss": loss.detach()}

    def step(data_iter, chunk):
        next(data_iter)
        out = chunk(input_ids=ids, position_ids=None, attention_mask=None, labels=None, packed_seq_params=psp)
        return out, loss_func

    fb = get_forward_backward_func()
    res = fb(forward_step_func=step, data_iterator=iter([None]), model=model, num_microbatches=1,
             seq_length=ids.shape[1], micro_batch_size=1, forward_only=forward_only)
    return float(res[0]["loss"]) if res else float("nan")


def zero(model, optimizer):
    for chunk in model:
        chunk.zero_grad_buffer()
    if optimizer is not None:
        optimizer.zero_grad()


# ----------------------------------------------------------------------------------------------
# Names: TE spec <-> local spec
# ----------------------------------------------------------------------------------------------

def named(model) -> dict[str, torch.Tensor]:
    out = {}
    for n, p in model[0].named_parameters():
        out[n] = p
    for n, b in model[0].named_buffers():
        if "expert_bias" in n:
            out[n] = b
    return out


def local_to_te(name: str) -> str:
    from miles.backends.megatron_utils.named_weights import local_expert_name_to_grouped

    name = te_canonical(local_expert_name_to_grouped(name), set())
    # dense MLP: TE fuses the pre-MLP RMSNorm into linear_fc1 (layer_norm_weight)
    name = re.sub(r"(decoder\.layers\.\d+)\.pre_mlp_layernorm\.weight$", r"\1.PRE_MLP_NORM", name)
    return name


def te_canonical(name: str, te_names: set[str]) -> str:
    return re.sub(r"(decoder\.layers\.\d+)\.mlp\.linear_fc1\.layer_norm_weight$", r"\1.PRE_MLP_NORM", name)


def map_local_to_te(local_names, te_names) -> tuple[dict[str, str], list[str], list[str]]:
    canon_te = {te_canonical(n, te_names): n for n in te_names}
    mapping, missing = {}, []
    for n in local_names:
        c = local_to_te(n)
        if c in canon_te:
            mapping[n] = canon_te[c]
        elif c.endswith("PRE_MLP_NORM") and c.replace("PRE_MLP_NORM", "pre_mlp_layernorm.weight") in canon_te:
            mapping[n] = canon_te[c.replace("PRE_MLP_NORM", "pre_mlp_layernorm.weight")]  # MoE layers keep the name
        else:
            missing.append(n)
    unused = sorted(set(te_names) - set(mapping.values()))
    return mapping, missing, unused


# ----------------------------------------------------------------------------------------------
# N7.5 phase
# ----------------------------------------------------------------------------------------------

def rank_file(out: Path, stem: str) -> Path:
    return out / f"{stem}.rank{dist.get_rank()}.pt"


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] rank{dist.get_rank() if dist.is_initialized() else '-'} {msg}"
    print(line, flush=True)
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(out / "progress.log", "a") as fh:
            fh.write(line + "\n")


def phase_n75(ns) -> None:
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    args = init_miles(megatron_argv(ns.impl, tp=2, ep=2, hf=Path(ns.hf), offload=False, lr=1e-5),
                      global_batch=1, rollouts=1)
    log(out, f"n75 impl={ns.impl} tag={ns.tag} transformer_impl={args.transformer_impl}")
    model, optimizer, _ = build(args)
    params = named(model)
    info = {"impl": ns.impl, "tag": ns.tag, "transformer_impl": args.transformer_impl,
            "module_classes": sorted({type(m).__name__ for m in model[0].modules()}),
            "num_params_local": sum(p.numel() for p in params.values()),
            "gpu": torch.cuda.get_device_name(), "names": sorted(params)}
    core = [type(m).__name__ for n, m in model[0].named_modules() if n.endswith("self_attention.core_attention")]
    info["dsa_core_attention"] = core
    weights_file = Path(ns.weights_from) / f"weights.rank{dist.get_rank()}.pt" if ns.weights_from else None
    if weights_file is None:
        torch.save({n: p.detach().cpu() for n, p in params.items()}, rank_file(out, "weights"))
        info["weights"] = "initialized here (seed 1234)"
    else:
        saved = torch.load(weights_file, map_location="cpu", weights_only=True)
        mapping, missing, unused = map_local_to_te(params, set(saved))
        info["name_map"] = {"mapped": len(mapping), "missing": missing, "unused_te": unused,
                            "renamed": {k: v for k, v in mapping.items() if k != v}}
        if missing or unused:
            raise RuntimeError(f"weight names do not map: missing={missing[:8]} unused={unused[:8]}")
        with torch.no_grad():
            for n, p in params.items():
                src = saved[mapping[n]]
                if tuple(src.shape) != tuple(p.shape):
                    raise RuntimeError(f"{n}: shape {tuple(p.shape)} != saved {tuple(src.shape)}")
                p.copy_(src.to(p.device, p.dtype))
            # the DDP/fp32 main copies are rebuilt from the model weights by the optimizer
            if optimizer is not None and hasattr(optimizer, "reload_model_params"):
                optimizer.reload_model_params()
        info["weights"] = f"loaded from {ns.weights_from}"

    # one forward/backward (no optimizer step)
    zero(model, optimizer)
    cap: list = []
    loss = run_fwd_bwd(model, tokens_for(0, 0, SEQ), capture=cap)
    torch.cuda.synchronize()
    logits_local, token_loss = cap[0]
    from megatron.core import parallel_state as mps
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

    logits = gather_from_tensor_model_parallel_region(logits_local.float())  # [1, T, V]
    grads = {}
    for n, p in params.items():
        g = getattr(p, "main_grad", None)
        if g is None:
            g = p.grad
        if g is not None:
            grads[n] = g.detach().float().cpu()
    torch.save(grads, rank_file(out, f"grads-{ns.tag}"))
    if dist.get_rank() == 0:
        torch.save({"logits": logits[0].cpu(), "token_loss": token_loss.float().cpu().view(-1), "loss": loss},
                   out / f"logits-{ns.tag}.pt")
    info["loss"] = loss
    info["grad_tensors"] = len(grads)
    info["tp_rank"], info["ep_rank"] = mps.get_tensor_model_parallel_rank(), mps.get_expert_model_parallel_rank()
    log(out, f"n75 {ns.tag}: loss={loss:.6f} grads={len(grads)}")

    # timing: forward+backward on TIMING_SEQ tokens; routed experts timed with CUDA events
    events: list = []

    def pre(mod, inp):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        mod._g3_t0 = e

    def post(mod, inp, outp):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        events.append((mod._g3_t0, e))

    hooks = []
    for n, m in model[0].named_modules():
        if n.endswith("mlp.experts"):
            hooks += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    steps = []
    for i in range(7):
        zero(model, optimizer)
        events.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_fwd_bwd(model, tokens_for(100 + i, 0, TIMING_SEQ))
        torch.cuda.synchronize()
        steps.append({"step_s": time.perf_counter() - t0,
                      "experts_fwd_ms": sum(a.elapsed_time(b) for a, b in events)})
    for h in hooks:
        h.remove()
    info["timing"] = {"tokens": TIMING_SEQ, "warmup": 2, "steps": steps,
                      "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
    zero(model, optimizer)

    pre = [None] * dist.get_world_size()
    dist.all_gather_object(pre, info)
    if dist.get_rank() == 0:
        (out / f"info-{ns.tag}.json").write_text(json.dumps(pre, indent=1, default=str))
    # N8.2 on the same (unchanged) weights
    try:
        info["n82"] = n82(args, model, out / f"n82-{ns.tag}")
    except Exception as exc:  # noqa: BLE001 -- recorded; the phase result says what failed
        info["n82"] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
        log(out, f"n82 {ns.tag} FAILED: {exc}")
    all_info = [None] * dist.get_world_size()
    dist.all_gather_object(all_info, info)
    if dist.get_rank() == 0:
        (out / f"info-{ns.tag}.json").write_text(json.dumps(all_info, indent=1, default=str))
    dist.barrier()


# ----------------------------------------------------------------------------------------------
# N8.2
# ----------------------------------------------------------------------------------------------

def nest_language_model(name: str) -> str:
    """GLM-5.3 checkpoint names keep the text model under ``model.language_model.`` (mbridge glm5_next)."""
    if name.startswith("model.") and not name.startswith(("model.language_model.", "model.visual.")):
        return "model.language_model." + name[len("model."):]
    return name


EXPERT_HF = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")


def miles_hf_weights(args, model) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Miles' weight-update path: global names, TP all-gather, convert_to_hf. Returns (non-expert on every
    rank, this EP rank's experts), HF names as Miles emits them (no nesting)."""
    from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
    from miles.backends.megatron_utils.named_weights import named_params_and_buffers
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import (
        all_gather_params_async,
        is_routed_expert_param,
    )
    from miles.utils.types import ParamInfo

    dense, experts = {}, {}
    for name, param in named_params_and_buffers(args, model):
        pinfo = ParamInfo(name=name, dtype=param.dtype, shape=param.shape, attrs={}, size=0, src_rank=dist.get_rank())
        full = all_gather_params_async(args, [(pinfo, param)])[0]
        for hf_name, t in convert_to_hf(args, "glm5_next", name, full):
            (experts if is_routed_expert_param(name) else dense)[hf_name] = t.detach().cpu().clone()
    return dense, experts


def fake_original(source: dict[str, torch.Tensor], scheme: str, out: Path, smooth: str | None) -> Path:
    """An "original" quantized checkpoint (headers matter, values do not) with the source's names."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for n, t in source.items():
        if EXPERT_HF.search(n):
            mod = n[: -len(".weight")]
            rows, cols = t.shape
            if scheme == "w4a8":
                tensors[n] = torch.zeros(rows, cols, dtype=torch.int8)
                tensors[mod + ".weight_scale"] = torch.ones(rows, 1, dtype=torch.bfloat16)
            else:
                tensors[mod + ".weight_packed"] = torch.zeros(rows, cols // 2, dtype=torch.uint8)
                tensors[mod + ".weight_scale"] = torch.zeros(rows, cols // 32, dtype=torch.uint8)
        elif n.endswith("mlp.gate.weight"):
            tensors[n] = t.float()  # the router is stored in F32
        else:
            tensors[n] = t
    shard = lambda n: ("model-00002-of-00002.safetensors" if re.search(r"layers\.[23]\.mlp\.experts", n)  # noqa: E731
                       else "model-00001-of-00002.safetensors")
    files: dict[str, dict] = {}
    for n, t in tensors.items():
        files.setdefault(shard(n), {})[n] = t.contiguous()
    for f, ts in files.items():
        save_file(ts, str(out / f), metadata={"format": "pt"})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {n: shard(n) for n in tensors}}))
    if scheme == "w4a8":
        q = {"quant_method": "compressed-tensors", "format": "int-quantized", "config_groups": {"group_0": {
            "targets": ["re:.*mlp\\.experts\\..*(gate|up|down)_proj$"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "channel", "group_size": None,
                        "dynamic": False, "observer": "memoryless_minmax"},
            "input_activations": {"num_bits": 8, "strategy": "token", "dynamic": True}}}}
    else:
        q = {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized", "config_groups": {"MXFP4A16": {
            "format": "mxfp4-pack-quantized", "targets": ["re:.*mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$"],
            "weights": {"num_bits": 4, "type": "float", "symmetric": True, "strategy": "group", "group_size": 32,
                        "dynamic": False, "scale_dtype": "torch.uint8", "observer": None},
            "input_activations": None}}, "ignore": ["lm_head", "re:^(?!.*mlp\\.experts\\.\\d+\\.).*$"]}
    (out / "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization_config": q}))
    if scheme == "w4a8" and smooth:
        layer = smooth.rsplit(".post_attention_layernorm", 1)[0]
        esc = re.escape(layer)
        (out / "recipe.yaml").write_text(
            "default_stage:\n  default_modifiers:\n    RecordedAWQ:\n      mappings:\n"
            f"      - smooth_layer: {smooth}\n"
            f"        balance_layers: [{layer}.mlp.gate, 're:{esc}\\.mlp\\.(experts\\.\\d+|shared_experts)\\.(gate|up)_proj$']\n"
            "      duo_scaling: true\n    QuantizationModifier:\n      ignore: [lm_head]\n")
    return out


def prepare_reference(source: dict[str, torch.Tensor], scheme: str, d: Path, smooth: str | None) -> dict:
    """Rank 0: template, BF16 source, AWQ factors and the single-process reference output under ``d``."""
    from safetensors.torch import save_file

    from yeto.rl.quant.awq import AwqSmoothing
    from yeto.rl.quant.requant import CheckpointLayout, requantize_checkpoint

    orig = fake_original(source, scheme, d / "original", smooth)
    CheckpointLayout.from_checkpoint(orig).save_template(orig, d / "template")
    (d / "bf16").mkdir(parents=True, exist_ok=True)
    save_file({n: t.contiguous() for n, t in source.items()}, str(d / "bf16" / "model.safetensors"),
              metadata={"format": "pt"})
    if scheme == "w4a8" and smooth:
        (d / "awq-scales").mkdir(exist_ok=True)
        factors = (0.8 + 0.4 * torch.rand(source[smooth + ".weight"].numel(),
                                          generator=torch.Generator().manual_seed(7))).double()
        torch.save(factors, d / "awq-scales" / f"{smooth}.pt")
    smoothing = (AwqSmoothing.from_artefact(d / "original" / "recipe.yaml", d / "awq-scales")
                 if scheme == "w4a8" and smooth else AwqSmoothing.none())
    t0 = time.perf_counter()
    requantize_checkpoint(d / "bf16", d / "template", d / "reference", smoothing)
    return {"single_process_s": time.perf_counter() - t0}


def n82(args, model, out: Path) -> dict:
    from megatron.core import parallel_state as mps

    from yeto.rl.adapters.miles.quant_shards import hf_slices, megatron_expert_slices
    from yeto.rl.quant.awq import AwqSmoothing
    from yeto.rl.quant.shards import SourceSlice, requantize_distributed

    rank, world = dist.get_rank(), dist.get_world_size()
    ep_rank, ep_size = mps.get_expert_model_parallel_rank(), mps.get_expert_model_parallel_world_size()
    if mps.get_data_parallel_world_size() != 1:
        raise RuntimeError("n82 harness expects DP=1 (every rank holds a distinct EP slice)")
    out.mkdir(parents=True, exist_ok=True)
    dense, experts = miles_hf_weights(args, model)
    gathered = [None] * world
    dist.all_gather_object(gathered, experts)
    res: dict = {"miles_hf_names_sample": sorted(dense)[:6] + sorted(experts)[:3],
                 "dense_tensors": len(dense), "expert_tensors_this_rank": len(experts)}
    # reference source: nested names (the checkpoint layout); every rank builds it (same bytes)
    source = {nest_language_model(n): t for n, t in dense.items()}
    for part in gathered:
        source.update({nest_language_model(n): t for n, t in part.items()})
    smooth = next((n[: -len(".weight")] for n in source if n.endswith("layers.1.post_attention_layernorm.weight")), None)
    res["awq_smooth_layer"] = smooth
    local = {n: p for n, p in model[0].named_parameters()}
    num_local = args.num_experts // ep_size
    for scheme in ("w4a8", "mxfp4"):
        d = out / scheme
        r: dict = {}
        try:
            err = None
            if rank == 0:
                try:
                    r.update(prepare_reference(source, scheme, d, smooth))
                except Exception as exc:  # noqa: BLE001 -- re-raised below after telling the peer
                    err = exc
            if err is not None:
                signal_failure()
                raise err
            checked_barrier()  # raises PeerFailed on rank 1 when rank 0 failed above
            smoothing = (AwqSmoothing.from_artefact(d / "original" / "recipe.yaml", d / "awq-scales")
                         if scheme == "w4a8" and smooth else AwqSmoothing.none())
            r["awq"] = bool(scheme == "w4a8" and smooth)
            # distributed: experts from the yeto adapter on Megatron-local names; non-experts once (rank 0)
            mine = list(megatron_expert_slices(((n.removeprefix("module.").removeprefix("module."), p)
                                                for n, p in local.items()),
                                               ep_rank=ep_rank, num_local_experts=num_local))
            mine_cpu = [SourceSlice(x.name, x.tensor.cpu(), x.row_start) for x in mine]
            r["expert_slices_this_rank"] = len(mine)
            r["expert_slice_fields"] = sorted(getattr(mine[0], "__dataclass_fields__", {})) if mine else []
            # raw: convert_to_hf names as they are; yeto: the adapter's hf_slices (renames to checkpoint
            # names), experts first as the GPU tensors a trainer holds, then copied to the host
            for variant in ("raw_convert_to_hf_names", "yeto_hf_slices_gpu_experts", "yeto_hf_slices_cpu_experts"):
                dst = d / f"dist-{variant}"
                slices = list(mine if variant == "yeto_hf_slices_gpu_experts" else mine_cpu)
                if rank == 0:
                    slices += (list(hf_slices(dense.items())) if variant.startswith("yeto_hf_slices")
                               else [SourceSlice(n, t) for n, t in dense.items()])
                try:
                    t0 = time.perf_counter()
                    stats = requantize_distributed(slices, d / "template", dst, smoothing, rank=rank,
                                                   world_size=world, barrier=checked_barrier)
                    r[variant] = {"ok": True, "s": time.perf_counter() - t0, "part": stats.get("part")}
                except PeerFailed as exc:
                    r[variant] = {"ok": False, "error": f"PeerFailed: {exc}"}
                except Exception as exc:  # noqa: BLE001
                    r[variant] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"}
                    signal_failure()  # pairs with the peer's next checked_barrier
                dist.barrier()
                if rank == 0 and r[variant]["ok"]:
                    r[variant]["byte_equal"] = compare_dirs(d / "reference", dst)
        except Exception as exc:  # noqa: BLE001
            r["error"] = f"{type(exc).__name__}: {exc}"
            r["traceback"] = traceback.format_exc()[-3000:]
        res[scheme] = r
    return res


class PeerFailed(RuntimeError):
    pass


def checked_barrier() -> None:
    """Barrier that also carries "a rank failed": every rank either reaches it or calls signal_failure()."""
    flag = torch.zeros(1, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item() > 0:
        raise PeerFailed("another rank failed in requantize_distributed")


def signal_failure() -> None:
    dist.all_reduce(torch.ones(1, device="cuda"), op=dist.ReduceOp.MAX)


def compare_dirs(a: Path, b: Path) -> dict:
    names_a = sorted(p.name for p in a.iterdir() if p.is_file())
    names_b = sorted(p.name for p in b.iterdir() if p.is_file())
    diff = [n for n in names_a if n in names_b and (a / n).read_bytes() != (b / n).read_bytes()]
    return {"files": names_a, "same_file_set": names_a == names_b, "differing": diff,
            "equal": names_a == names_b and not diff,
            "sha256": {n: hashlib.sha256((a / n).read_bytes()).hexdigest()[:16] for n in names_a}}


# ----------------------------------------------------------------------------------------------
# N7.1d phase
# ----------------------------------------------------------------------------------------------

class Actor:
    """The attributes of a Miles MegatronTrainRayActor that the yeto cut plugin reads."""

    def __init__(self, args, model, optimizer, scheduler):
        self.args, self.model, self.optimizer, self.opt_param_scheduler = args, model, optimizer, scheduler


def hdo_report(optimizer) -> dict:
    try:
        return _hdo_report(optimizer)
    except Exception as exc:  # noqa: BLE001 -- diagnostics only
        return {"error": f"{type(exc).__name__}: {exc}"}


def _hdo_report(optimizer) -> dict:
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    leaves = getattr(optimizer, "chained_optimizers", [optimizer])
    rep = []
    for leaf in leaves:
        inner = getattr(leaf, "optimizer", None)
        item = {"leaf": type(leaf).__name__, "inner": type(inner).__name__,
                "is_hdo": isinstance(inner, HybridDeviceOptimizer)}
        if isinstance(inner, HybridDeviceOptimizer):
            fp32 = list(inner.param_to_fp32_param.values()) if getattr(inner, "param_to_fp32_param", None) else []
            copies = list(getattr(inner, "gpu_params_map_cpu_copy", {}).values())
            item.update(
                host_copies=len(copies), host_copies_pinned=sum(bool(c.is_pinned()) for c in copies),
                host_copy_dtypes=sorted({str(c.dtype) for c in copies}),
                fp32_params=len(fp32), fp32_devices=sorted({str(p.device) for p in fp32}),
                fp32_pinned=sum(bool(p.is_pinned()) for p in fp32 if p.device.type == "cpu"),
                fp32_dtypes=sorted({str(p.dtype) for p in fp32}),
                cpu_optimizers=[type(o).__name__ for o in inner.cpu_optimizers],
                overlap=bool(inner.defaults.get("overlap_cpu_optimizer_d2h_h2d")),
                state_dtypes=sorted({str(v.dtype) for o in inner.cpu_optimizers for s in o.state.values()
                                     for v in s.values() if torch.is_tensor(v) and v.numel() > 1}),
                steps=sorted({float(s["step"]) for o in inner.cpu_optimizers for s in o.state.values() if "step" in s}),
            )
        rep.append(item)
    return {"leaves": rep}


def stale_report(optimizer) -> dict:
    """Per HDO parameter: does the tensor the sub-optimizer steps equal the HDO's "master_param"?
    (the old export read the master; N7.6c reads the stepped tensor)"""
    from miles.backends.megatron_utils.lora import dp_invariant_state as dps

    stale, same = [], 0
    for leaf in getattr(optimizer, "chained_optimizers", [optimizer]):
        hdo = getattr(leaf, "optimizer", None)
        if not hasattr(hdo, "sub_optimizers"):
            continue
        for param, state in hdo.state.items():
            master = state.get("master_param")
            if not torch.is_tensor(master):
                continue
            inner = dps._hdo_inner_param(hdo, param)
            if torch.equal(inner.detach().cpu().reshape(-1), master.detach().cpu().reshape(-1)):
                same += 1
            else:
                stale.append({"shape": list(param.shape), "dtype": str(param.dtype)})
    return {"stepped_equals_master": same, "stepped_differs_from_master": len(stale), "differs": stale[:40]}


def param_digest(model) -> dict:
    h = hashlib.sha256()
    norms = {}
    for n, p in sorted(model[0].named_parameters()):
        t = p.detach().float().cpu().contiguous()
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        norms[n] = float(t.norm())
    return {"sha256": h.hexdigest(), "norms": norms}


def train_steps(args, model, optimizer, scheduler, first: int, last: int, out: Path) -> list[dict]:
    from megatron.core import parallel_state as mps

    dp = mps.get_data_parallel_rank()
    rec = []
    for step in range(first, last + 1):
        zero(model, optimizer)
        t0 = time.perf_counter()
        loss = run_fwd_bwd(model, tokens_for(step, dp, SEQ))
        ok, grad_norm, _ = optimizer.step()
        assert ok
        scheduler.step(increment=args.global_batch_size)
        torch.cuda.synchronize()
        lt = torch.tensor([loss], device="cuda")
        dist.all_reduce(lt)
        rec.append({"step": step, "loss_mean_dp": float(lt) / dist.get_world_size(), "loss_rank": loss,
                    "grad_norm": float(grad_norm), "lr": [g["lr"] for g in optimizer.param_groups],
                    "scheduler_num_steps": int(scheduler.num_steps), "s": time.perf_counter() - t0})
        log(out, f"step {step} loss={loss:.6f} grad_norm={float(grad_norm):.6f}")
    zero(model, optimizer)
    return rec


def phase_n71d(ns) -> None:
    from yeto.rl.adapters.miles import cut_plugin
    from yeto.rl.engine.resume_manifest import (
        OptimizerResumeState,
        ResumeManifest,
        first_step_problems,
        require_saveable,
        restore_problems,
    )

    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    args = init_miles(megatron_argv("te", tp=1, ep=1, hf=Path(ns.hf), offload=True, lr=1e-4),
                      global_batch=2, rollouts=10)
    flags = {k: getattr(args, k, None) for k in (
        "use_distributed_optimizer", "optimizer_cpu_offload", "use_precision_aware_optimizer",
        "overlap_cpu_optimizer_d2h_h2d", "optimizer_offload_fraction", "main_params_dtype", "main_grads_dtype",
        "exp_avg_dtype", "exp_avg_sq_dtype", "data_parallel_size", "bf16", "transformer_impl")}
    log(out, f"n71d {ns.mode} flags={flags}")
    model, optimizer, scheduler = build(args)
    actor = Actor(args, model, optimizer, scheduler)
    res: dict = {"mode": ns.mode, "flags": {k: str(v) for k, v in flags.items()}, "gpu": torch.cuda.get_device_name()}
    res["hdo_before"] = hdo_report(optimizer)
    cut_dir = Path(ns.cut_dir)

    def gather(obj):
        lst = [None] * dist.get_world_size()
        dist.all_gather_object(lst, obj)
        return lst

    def manifest(summaries, global_step):
        return ResumeManifest(global_step=global_step, data_watermark=global_step * args.global_batch_size,
                              scheduler_increment=args.global_batch_size,
                              ranks={str(s["path"]): OptimizerResumeState.from_dict(s["resume"]) for s in summaries})

    if ns.mode.startswith("cont5"):
        res["steps"] = train_steps(args, model, optimizer, scheduler, 1, 5, out)
    elif ns.mode.startswith("run3"):
        res["steps"] = train_steps(args, model, optimizer, scheduler, 1, 3, out)
        summary = cut_plugin.save_cut_shard(actor, directory=str(cut_dir), cut_id="g3-cut-3")
        if cut_plugin.REFUSED in summary:
            raise RuntimeError(f"save_cut refused: {summary}")
        summaries = gather(summary)
        res["cut_summary"] = {k: summary[k] for k in ("path", "bytes", "scheduler_samples", "tuning",
                                                      "adapter_tensors", "resume")}
        if dist.get_rank() == 0:
            m = manifest(summaries, 3)
            require_saveable(m)
            (cut_dir / "files.json").write_text(json.dumps(summaries, default=str))
            (cut_dir / "resume_manifest.json").write_text(json.dumps(m.to_dict()))
        res["hdo_after_save"] = hdo_report(optimizer)
    elif ns.mode.startswith("resume2"):
        files = json.loads((cut_dir / "files.json").read_text())
        saved = ResumeManifest.from_dict(json.loads((cut_dir / "resume_manifest.json").read_text()))
        inject = ns.mode.endswith("inj")
        if inject:  # T-C4 N7.6c: put the T-G3 bug back (the restore leaves the HDO stepping its stale host copy)
            from miles.backends.megatron_utils.lora import dp_invariant_state as dps

            dps._bind_masters_to_stepped_tensors = lambda hdo: None
        summary = cut_plugin.restore_cut_shard(actor, directory=str(cut_dir), files=files, cut_id="g3-cut-3")
        if cut_plugin.REFUSED in summary:
            raise RuntimeError(f"restore_cut refused: {summary}")
        if inject:
            res["injected"] = "dp_invariant_state._bind_masters_to_stepped_tensors = no-op"
            res["stale_vs_master"] = stale_report(optimizer)
        res["restore_summary"] = {k: summary.get(k) for k in ("path", "scheduler_samples", "resume", "diff",
                                                              "optimizer_cut_vs_reexport")}
        res["restore_digest_equal"] = summary.get("state_digest") == next(
            f for f in files if f["path"] == summary["path"])["state_digest"]
        res["hdo_after_restore"] = hdo_report(optimizer)
        now = manifest(gather(summary), 3)
        res["restore_problems"] = restore_problems(saved, now, check_data_watermark=True)
        res["steps"] = train_steps(args, model, optimizer, scheduler, 4, 4, out)
        st = cut_plugin.resume_state(actor)
        after1 = manifest(gather(st), 4)
        res["first_step_problems"] = first_step_problems(saved, after1, steps=1)
        res["hdo_after_first_step"] = hdo_report(optimizer)
        res["steps"] += train_steps(args, model, optimizer, scheduler, 5, 5, out)
    else:
        raise ValueError(ns.mode)
    res["final_params"] = param_digest(model)
    res["hdo_final"] = hdo_report(optimizer)
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    allres = gather(res)
    if dist.get_rank() == 0:
        (out / f"n71d-{ns.mode}.json").write_text(json.dumps(allres, indent=1, default=str))
    dist.barrier()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=["hfconfig", "n75", "n71d"])
    p.add_argument("--out", required=True)
    p.add_argument("--hf", default="/tmp/g3-hf")
    p.add_argument("--impl", default="te", choices=["te", "local"])
    p.add_argument("--tag", default="te")
    p.add_argument("--weights-from", default="")
    p.add_argument("--mode", default="cont5")
    p.add_argument("--cut-dir", default="")
    p.add_argument("--fixture", default=str(HERE / "glm53flash_config.json"))
    ns = p.parse_args()
    if ns.phase == "hfconfig":
        write_hf_config(Path(ns.fixture), Path(ns.hf))
        return
    try:
        (phase_n75 if ns.phase == "n75" else phase_n71d)(ns)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
