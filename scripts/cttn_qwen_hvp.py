"""Real-scale CTTN HVP feasibility on Qwen3.5-9B + LoRA (single A100).

Confirms at 9B scale what test_hvp_lora.py proved on a tiny CPU model:
  * double-backward HVPs run on the real model with eager attention,
  * they fit in 40GB alongside bf16 weights (micro-batch 1, seq 128),
  * an 8-HVP block-Lanczos sketch + cttn_step runs and preserves q^Td==||g||,
  * and how long 8 HVPs actually take (the per-outer-round CTTN cost).

Run on the wsub host, GPU 7:  CUDA_VISIBLE_DEVICES=7 python scripts/cttn_qwen_hvp.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from yeto.cttn import block_lanczos, cttn_step, orth, project_out
from yeto.losses import sft_loss

MODEL = "Qwen/Qwen3.5-9B"
DEV = torch.device("cuda:0")
torch.manual_seed(20260713)


def build():
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager",
    )
    lora = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora).to(DEV)
    model.config.use_cache = False
    return model


def main() -> int:
    t0 = time.perf_counter()
    model = build()
    params = [p for p in model.parameters() if p.requires_grad]
    p = sum(x.numel() for x in params)
    vocab = model.config.vocab_size
    print(f"loaded {MODEL} + LoRA r2: {p} trainable params, load {time.perf_counter()-t0:.1f}s")

    B, T = 1, 128
    input_ids = torch.randint(0, vocab, (B, T), device=DEV)
    weights = torch.ones(B, T, device=DEV)

    def hvp_np(X):  # [p, m] numpy -> [p, m] numpy via double-backward
        cols = []
        for j in range(X.shape[1]):
            v = torch.tensor(X[:, j], dtype=torch.float32, device=DEV)
            out = model(input_ids=input_ids, use_cache=False)
            loss, ntok = sft_loss(out.logits, input_ids, "cross_entropy", weights)
            lpt = loss / torch.clamp(ntok.float(), min=1.0)
            g = torch.autograd.grad(lpt, params, create_graph=True)
            gflat = torch.cat([t.reshape(-1).float() for t in g])
            gv = torch.dot(gflat, v)
            Hv = torch.autograd.grad(gv, params, retain_graph=False)
            cols.append(torch.cat([t.reshape(-1).float() for t in Hv]).detach().cpu().numpy())
        return np.stack(cols, axis=1)

    torch.cuda.reset_peak_memory_stats()

    # single-HVP smoke + symmetry
    rng = np.random.default_rng(1)
    u = rng.standard_normal(p); u /= np.linalg.norm(u)
    v = rng.standard_normal(p); v /= np.linalg.norm(v)
    t1 = time.perf_counter()
    Hu = hvp_np(u[:, None])[:, 0]
    dt1 = time.perf_counter() - t1
    Hv = hvp_np(v[:, None])[:, 0]
    uHv = float(u @ Hv); vHu = float(v @ Hu)
    sym_ok = abs(uHv - vHu) < 1e-2 * (abs(uHv) + abs(vHu) + 1e-9)
    print(f"  [{'PASS' if sym_ok else 'FAIL'}] symmetry u^THv={uHv:.5g} v^THu={vHu:.5g}  "
          f"(1 HVP = {dt1:.2f}s, ||Hu||={np.linalg.norm(Hu):.4g})")

    # full 8-HVP block-Lanczos sketch + cttn_step
    g = rng.standard_normal(p); q = g / np.linalg.norm(g)
    buf = rng.standard_normal(p) * 3.0; r = project_out(buf, q)
    Q0 = orth([q, r / np.linalg.norm(r)])
    t2 = time.perf_counter()
    V, T = block_lanczos(hvp_np, Q0, block_steps=4)
    dt_sketch = time.perf_counter() - t2
    res = cttn_step(g, buf, V, T, mu=0.9, rho=0.10)
    qtd = float(q @ res.d)
    e2e_ok = abs(qtd - np.linalg.norm(g)) < 1e-5 * np.linalg.norm(g)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  [{'PASS' if e2e_ok else 'FAIL'}] end-to-end: q^Td={qtd:.5f} ||g||={np.linalg.norm(g):.5f} "
          f"bind={res.bind} retention={res.norm_retention:.4f} ritz_max={res.ritz.max():.4g}")
    print(f"  8-HVP sketch: {dt_sketch:.1f}s  peak_mem={peak:.1f}GB  "
          f"(V:{V.shape}, Ritz modes n90={res.n_modes_90})")
    ok = sym_ok and e2e_ok
    print(f"\n{'ALL PASS — CTTN feasible at 9B on A100-40GB' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
