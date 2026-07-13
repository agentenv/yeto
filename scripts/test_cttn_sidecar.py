"""End-to-end test of the sidecar CTTN step (yeto/cttn_sidecar) on a tiny model.

Validates the retained-graph HVP path used inside the action-probe sidecar:
  * exactly ONE forward + ONE first-backward for the whole 8-HVP sketch
    (efficiency: the fix for the 84.7s/14-HVP regression);
  * retained-graph HVP == naive per-column HVP (correctness);
  * cttn_sidecar_step end-to-end preserves q^T d == ||g|| on real curvature;
  * panel averaging works (multi-panel Hessian).

Run:  python scripts/test_cttn_sidecar.py
"""

from __future__ import annotations

import numpy as np
import torch

from yeto.cttn_sidecar import (
    cttn_sidecar_step,
    flatten_params,
    make_hvp,
    _unflatten_like,
)

torch.manual_seed(20260713)
DEV = torch.device("cpu")


def build_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, attn_implementation="eager",
    )
    model = get_peft_model(LlamaForCausalLM(cfg),
                           LoraConfig(r=4, lora_alpha=8,
                                      target_modules=["q_proj", "v_proj"],
                                      lora_dropout=0.0, bias="none",
                                      task_type="CAUSAL_LM")).to(DEV).float()
    return model


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main() -> int:
    model = build_model()
    params = tuple(p for p in model.parameters() if p.requires_grad)
    p = sum(x.numel() for x in params)

    # two held-out panels
    panels = [(torch.randint(0, 128, (1, 16)), torch.ones(1, 16)) for _ in range(2)]
    rng = np.random.default_rng(11)

    ok = True
    print("CTTN sidecar (tiny LoRA-Llama):")

    # 1. retained-graph HVP == naive per-column HVP, and counts ONE forward.
    fwd_calls = {"n": 0}
    orig_forward = model.forward

    def counting_forward(*a, **k):
        fwd_calls["n"] += 1
        return orig_forward(*a, **k)

    model.forward = counting_forward
    hvp, loss_val, release = make_hvp(model, params, panels)
    n_fwd_after_build = fwd_calls["n"]
    X = torch.tensor(rng.standard_normal((p, 8)), dtype=torch.float32)
    HX = hvp(X)
    n_fwd_after_8hvp = fwd_calls["n"]
    release()
    model.forward = orig_forward

    ok &= check(f"one forward per panel to build graph "
                f"({n_fwd_after_build} == {len(panels)})",
                n_fwd_after_build == len(panels))
    ok &= check(f"8 HVP columns add ZERO forwards (retained graph) "
                f"({n_fwd_after_8hvp - n_fwd_after_build} == 0)",
                n_fwd_after_8hvp == n_fwd_after_build)

    # naive HVP for one column to cross-check
    def naive_hvp_col(v):
        total = None
        from yeto.losses import sft_loss
        for input_ids, weights in panels:
            out = model(input_ids=input_ids, use_cache=False)
            ls, nt = sft_loss(out.logits, input_ids, "cross_entropy", weights)
            lpt = ls / torch.clamp(nt.float(), min=1.0)
            total = lpt if total is None else total + lpt
        loss = total / len(panels)
        g1 = torch.autograd.grad(loss, params, create_graph=True)
        gflat = torch.cat([t.reshape(-1) for t in g1])
        gv = torch.dot(gflat, v)
        hv = torch.autograd.grad(gv, params)
        return torch.cat([t.reshape(-1) for t in hv])

    v0 = X[:, 0]
    naive = naive_hvp_col(v0)
    rel = (HX[:, 0] - naive).norm().item() / (naive.norm().item() + 1e-12)
    ok &= check(f"retained-graph HVP == naive (rel={rel:.2e})", rel < 1e-5)

    # 2. end-to-end sidecar step: q^T d == ||g||, finite, buffer shape.
    g = torch.tensor(rng.standard_normal(p), dtype=torch.float32)
    b = torch.tensor(rng.standard_normal(p) * 3.0, dtype=torch.float32)
    res = cttn_sidecar_step(model, params, panels, g, b, mu=0.9, rho=0.10)
    q = g / g.norm()
    qtd = float(torch.dot(q, res.d))
    ok &= check(f"sidecar q^T d == ||g|| ({qtd:.6f} vs {float(g.norm()):.6f})",
                abs(qtd - float(g.norm())) < 1e-4 * float(g.norm()))
    ok &= check("sidecar d, b_new finite",
                bool(torch.all(torch.isfinite(res.d))) and
                bool(torch.all(torch.isfinite(res.b_new))))
    ok &= check(f"buffer shape preserved ({res.b_new.shape[0]} == {p})",
                res.b_new.shape[0] == p)
    print(f"    diag: bind={res.diag.bind} retention={res.diag.norm_retention:.4f} "
          f"tau={res.diag.tau:.4g} ritz_max={res.diag.ritz.max() if res.diag.ritz.size else 0:.4g} "
          f"loss={res.loss:.4f}")

    # 3. flatten/unflatten round-trip in fragment order
    flat = flatten_params(params)
    parts = _unflatten_like(flat, params)
    rt = torch.cat([x.reshape(-1) for x in parts])
    ok &= check("flatten/unflatten round-trips", bool(torch.allclose(flat, rt)))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
