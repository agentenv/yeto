"""rl-algo-supplement 2.1: pg_clipfrac compile vs eager, eps_clip == / != eps_clip_high (CPU).

Run (Miles fork checkout on PYTHONPATH, miles-next-venv):
  PYTHONPATH=/home/michael/work/miles-algosup OMP_NUM_THREADS=1 \
    /home/michael/work/miles-next-venv/bin/python check_clipfrac_compile.py
"""
import torch

from miles.backends.training_utils.loss_hub import math_utils as mu

compiled = mu.compute_policy_loss
eager = getattr(compiled, "_torchdynamo_orig_callable", None) or compiled.__wrapped__


def hand(ppo_kl, adv, lo, hi):
    ratio = torch.exp(-ppo_kl)
    l1 = -ratio * adv
    l2 = -ratio.clamp(1 - lo, 1 + hi) * adv
    return torch.maximum(l1, l2), (l2 > l1).float()


def main():
    torch.manual_seed(0)
    n = 4096
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        ppo_kl = (torch.randn(n) * 0.01).to(dtype)
        adv = torch.randn(n).to(dtype)
        cases = [("eq 0.001/0.001", 0.001, 0.001), ("neq 0.001/10", 0.001, 10.0),
                 ("eq 0.2/0.2", 0.2, 0.2), ("neq 0.2/0.28", 0.2, 0.28)]
        # same python float object for both args (the g1e A-arm shape) and two distinct objects
        for name, lo, hi in cases:
            for alias in (True, False):
                hi_arg = lo if (alias and lo == hi) else float(str(hi))
                if not alias and lo == hi:
                    hi_arg = float(repr(hi))  # equal value, distinct object
                hl, hc = hand(ppo_kl.float(), adv.float(), lo, hi)
                el, ec = eager(ppo_kl, adv, lo, hi_arg)
                cl, cc = compiled(ppo_kl, adv, lo, hi_arg)
                upper = ((adv.float() > 0) & (torch.exp(-ppo_kl.float()) > 1 + hi)).float().mean().item()
                rows.append((str(dtype).split(".")[1], name, "same-obj" if alias and lo == hi else "distinct",
                             hc.mean().item(), ec.float().mean().item(), cc.float().mean().item(),
                             upper, torch.equal(el, cl), torch.equal(ec, cc)))
    # call order matters for dynamo specialization: eq first, then neq, on one compiled object
    print(f"torch {torch.__version__}")
    print("dtype | case | eps args | hand | eager | compiled | frac(A>0,ratio>1+hi) | loss eager==compiled | clipfrac eager==compiled")
    for r in rows:
        print(" | ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r))
    # fresh compiled object, neq first then eq (reverse specialization order)
    fresh = torch.compile(eager, dynamic=True)
    ppo_kl = torch.randn(n) * 0.01
    adv = torch.randn(n)
    for lo, hi in ((0.001, 10.0), (0.001, 0.001), (0.2, 0.28), (0.2, 0.2)):
        hl, hc = hand(ppo_kl, adv, lo, hi)
        cl, cc = fresh(ppo_kl, adv, lo, hi)
        print(f"reverse-order fresh compile lo={lo} hi={hi}: hand={hc.mean().item():.6f} compiled={cc.mean().item():.6f} "
              f"clipfrac equal={torch.equal(hc, cc)} loss max|diff|={(hl - cl).abs().max().item():.3e}")
    bad = [r for r in rows if not (r[8] and abs(r[4] - r[3]) < 1e-6 or r[0] == "bfloat16")]
    print("MISMATCH rows (float32):", bad if bad else "none")


if __name__ == "__main__":
    main()
