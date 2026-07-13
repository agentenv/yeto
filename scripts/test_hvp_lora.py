"""De-risk the CTTN HVP: correct double-backward Hessian-vector products through
a LoRA causal-LM on the sidecar's exact loss (yeto.losses.sft_loss).

Validates against central finite differences of the gradient:
    H v  ==  (grad L(theta + eps v) - grad L(theta - eps v)) / (2 eps).
If this passes, the same mechanic drops into yeto/action_probe_server.py
(currently forward-only under inference_mode). CPU + a tiny random Llama so it
runs anywhere with no download and no MPS double-backward quirks.

Run:  python scripts/test_hvp_lora.py
"""

from __future__ import annotations

import numpy as np
import torch

from yeto.losses import sft_loss

torch.manual_seed(20260713)
DEV = torch.device("cpu")          # double-backward is device-independent; CPU is safe


def build_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model

    cfg = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
        # IMPORTANT: flash/efficient SDPA kernels have NO double-backward; HVPs
        # require eager attention. The sidecar HVP path must set this too.
        attn_implementation="eager",
    )
    base = LlamaForCausalLM(cfg)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora).to(DEV).float()
    return model


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def flat(vs):
    return torch.cat([v.reshape(-1) for v in vs])


def set_flat(params, theta):
    off = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(theta[off:off + n].view_as(p))
            off += n


def loss_at(model, params, theta, batch):
    set_flat(params, theta)
    input_ids, weights = batch
    out = model(input_ids=input_ids, use_cache=False)
    loss, ntok = sft_loss(out.logits, input_ids, "cross_entropy", weights)
    return loss / torch.clamp(ntok.float(), min=1.0)


def grad_at(model, params, theta, batch, create_graph=False):
    set_flat(params, theta)
    input_ids, weights = batch
    out = model(input_ids=input_ids, use_cache=False)
    loss, ntok = sft_loss(out.logits, input_ids, "cross_entropy", weights)
    lpt = loss / torch.clamp(ntok.float(), min=1.0)
    g = torch.autograd.grad(lpt, params, create_graph=create_graph)
    return flat(g)


def hvp(model, params, theta, batch, v):
    """H v via double-backward: grad( grad(L).v )."""
    set_flat(params, theta)
    input_ids, weights = batch
    out = model(input_ids=input_ids, use_cache=False)
    loss, ntok = sft_loss(out.logits, input_ids, "cross_entropy", weights)
    lpt = loss / torch.clamp(ntok.float(), min=1.0)
    g = torch.autograd.grad(lpt, params, create_graph=True)
    gv = torch.dot(flat(g), v)
    Hv = torch.autograd.grad(gv, params, retain_graph=False)
    return flat(Hv)


def main() -> int:
    model = build_model()
    params = trainable(model)
    p = sum(x.numel() for x in params)
    theta0 = flat(params).detach().clone()

    B, T = 2, 16
    input_ids = torch.randint(0, 128, (B, T), device=DEV)
    weights = torch.ones(B, T, device=DEV)
    batch = (input_ids, weights)

    print(f"tiny LoRA-Llama: {p} trainable params")

    ok = True
    # a few random probe directions
    rng = np.random.default_rng(7)
    for k in range(3):
        v = torch.tensor(rng.standard_normal(p), dtype=torch.float32, device=DEV)
        v = v / v.norm()
        Hv = hvp(model, params, theta0, batch, v).detach()

        eps = 1e-3
        gp = grad_at(model, params, theta0 + eps * v, batch)
        gm = grad_at(model, params, theta0 - eps * v, batch)
        fd = (gp - gm) / (2 * eps)

        rel = (Hv - fd).norm().item() / (fd.norm().item() + 1e-12)
        sym = "PASS" if rel < 1e-2 else "FAIL"
        ok &= rel < 1e-2
        print(f"  [{sym}] probe {k}: ||Hv - fd|| / ||fd|| = {rel:.3e}  "
              f"(||Hv||={Hv.norm().item():.4g})")

    # symmetry spot-check: u^T H v == v^T H u
    u = torch.tensor(rng.standard_normal(p), dtype=torch.float32).to(DEV); u /= u.norm()
    v = torch.tensor(rng.standard_normal(p), dtype=torch.float32).to(DEV); v /= v.norm()
    uHv = torch.dot(u, hvp(model, params, theta0, batch, v).detach()).item()
    vHu = torch.dot(v, hvp(model, params, theta0, batch, u).detach()).item()
    sym_ok = abs(uHv - vHu) < 1e-4 * (abs(uHv) + abs(vHu) + 1e-9)
    ok &= sym_ok
    print(f"  [{'PASS' if sym_ok else 'FAIL'}] symmetry u^T H v ({uHv:.6g}) == v^T H u ({vHu:.6g})")

    # timing: HVP cost vs a plain forward-backward (the 6.3x accounting)
    import time
    set_flat(params, theta0)
    t0 = time.perf_counter()
    for _ in range(5):
        grad_at(model, params, theta0, batch, create_graph=False)
    t_fb = (time.perf_counter() - t0) / 5
    v = torch.tensor(rng.standard_normal(p), dtype=torch.float32).to(DEV); v /= v.norm()
    t0 = time.perf_counter()
    for _ in range(5):
        hvp(model, params, theta0, batch, v)
    t_hvp = (time.perf_counter() - t0) / 5
    print(f"  [time] fwd-bwd={t_fb*1e3:.1f}ms  hvp={t_hvp*1e3:.1f}ms  ratio={t_hvp/t_fb:.2f}x")

    # END-TO-END: real model -> real HVP -> block-Lanczos (V,T) -> cttn_step.
    # Proves the whole CTTN pipeline runs on true curvature, not just synthetic H.
    from yeto.cttn import block_lanczos, cttn_step, orth, project_out

    def hvp_np(X):  # [p, m] numpy -> [p, m] numpy, columnwise true HVP
        cols = []
        for j in range(X.shape[1]):
            vj = torch.tensor(X[:, j], dtype=torch.float32, device=DEV)
            cols.append(hvp(model, params, theta0, batch, vj).detach().cpu().numpy())
        return np.stack(cols, axis=1)

    g = rng.standard_normal(p).astype(np.float64)          # a merged pseudo-grad
    buf = rng.standard_normal(p).astype(np.float64) * 3.0   # a Nesterov buffer
    qv = g / np.linalg.norm(g)
    rv = project_out(buf, qv)
    Q0 = orth([qv, rv / np.linalg.norm(rv)])
    V, T = block_lanczos(hvp_np, Q0, block_steps=4)
    res = cttn_step(g, buf, V, T, mu=0.9, rho=0.10)
    qtd = float(qv @ res.d)
    e2e_ok = abs(qtd - np.linalg.norm(g)) < 1e-6
    ok &= e2e_ok
    print(f"  [{'PASS' if e2e_ok else 'FAIL'}] end-to-end CTTN on real curvature: "
          f"q^Td={qtd:.6f} vs ||g||={np.linalg.norm(g):.6f}; "
          f"bind={res.bind} retention={res.norm_retention:.4f} "
          f"ritz_max={res.ritz.max():.4g} n90={res.n_modes_90}")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
