# tf-diagnosis: compare tfdiag microbatch dumps (policy_loss_function inputs/outputs) across runs
import sys, glob, torch, hashlib, numpy as np
def load(run, isl):
    fs = sorted(glob.glob(f"{run}/work/seed-17/yeto-federated-m2/island-{isl}/tfdiag/*.tfd"))
    out = {}
    for f in fs:
        r = torch.load(f, weights_only=False)
        for j, tok in enumerate(r["unconcat_tokens"]):
            key = hashlib.sha1(tok.numpy().tobytes()).hexdigest()[:16]
            g = lambda k: (r[k][j] if r.get(k) is not None else None)
            out[key] = dict(tokens=tok, resp=r["response_lengths"][j], total=r["total_lengths"][j],
                            mask=g("loss_masks"), adv=g("advantages"), old=g("log_probs"), roll=g("rollout_log_probs"),
                            ref=g("ref_log_probs"), train=r["train_log_probs"][j] if r["train_log_probs"] else None,
                            den=g("rollout_mask_sums"), reward=g("rewards"), mb_loss=r["loss"], file=f)
    return out, len(fs)
def f(x): return x.float() if x is not None else None
a, b = sys.argv[1], sys.argv[2]
for isl in (0, 1):
    A, na = load(a, isl); B, nb = load(b, isl)
    print(f"== island {isl}: {a} ({na} mb, {len(A)} samples) vs {b} ({nb} mb, {len(B)} samples)")
    common = sorted(set(A) & set(B)); print(f"  common samples by token hash: {len(common)}; order same: {list(A)==list(B)}")
    if not common: continue
    stats = {k: [] for k in ("train", "old", "ratioA", "ratioB")}
    adv_eq = mask_eq = den_eq = True; ntok = 0; worst = []
    pa = pb = 0.0
    for k in common:
        x, y = A[k], B[k]; m = f(x["mask"]).bool()
        mask_eq &= torch.equal(x["mask"].float(), y["mask"].float()) and x["resp"] == y["resp"] and x["total"] == y["total"]
        adv_eq &= torch.equal(f(x["adv"]), f(y["adv"]))
        if x["den"] is not None or y["den"] is not None:
            den_eq &= (x["den"] is not None and y["den"] is not None and float(x["den"]) == float(y["den"]))
        ntok += int(m.sum())
        for nm in ("train", "old"):
            if x[nm] is not None and y[nm] is not None:
                stats[nm].append((f(x[nm]) - f(y[nm]))[m].abs())
        stats["ratioA"].append((f(x["train"]) - f(x["old"]))[m].abs()); stats["ratioB"].append((f(y["train"]) - f(y["old"]))[m].abs())
        adv = f(x["adv"])[m].mean().item()
        # per-sample loss contribution / gradient weight: A * mean_t(ratio) / n_i  (sample-mean), in each run
        worst.append((((f(x["train"]) - f(y["train"]))[m].abs().mean().item()), adv, int(m.sum())))
    print(f"  loss_mask/lengths identical: {mask_eq}; advantages bit-identical: {adv_eq}; rollout_mask_sums identical: {den_eq}; tokens={ntok}")
    for nm, lab in (("train", "train-forward logprob |L-P|"), ("old", "forward-only (old) logprob |L-P|"),
                    ("ratioA", f"within {a.split('/')[-1]}: |train-old|"), ("ratioB", f"within {b.split('/')[-1]}: |train-old|")):
        if stats[nm]:
            v = torch.cat(stats[nm]); print(f"  {lab:45s} max={v.max():.4e} mean={v.mean():.4e} p99={v.quantile(0.99) if v.numel()<16_000_000 else float('nan'):.4e} frac_exact0={(v==0).float().mean():.4f}")
    rl = [torch.cat([(f(A[k]["train"]) - f(A[k]["roll"]))[f(A[k]["mask"]).bool()].abs() for k in common if A[k]["roll"] is not None])]
    if rl[0].numel(): print(f"  ref: |train - sglang rollout logprob| in {a.split('/')[-1]}: mean={rl[0].mean():.4e} max={rl[0].max():.4e}")
    ws = sorted(worst, reverse=True)[:3]; print("  worst samples (mean|dlogp|, adv, ntok):", [(round(w[0],5), round(w[1],3), w[2]) for w in ws])
    ma = sorted(set(x["mb_loss"] for x in A.values())); mb = sorted(set(x["mb_loss"] for x in B.values()))
    sa = sum(A[k]["mb_loss"] for k in A); print(f"  microbatch loss sum: {a.split('/')[-1]}={sum(x['mb_loss'] for x in A.values()):.6e} {b.split('/')[-1]}={sum(x['mb_loss'] for x in B.values()):.6e}")
