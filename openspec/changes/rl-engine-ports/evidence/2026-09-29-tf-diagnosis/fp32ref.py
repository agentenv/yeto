# tf-diagnosis: CPU fp32 (float64 logsoftmax) HF reference logprobs for the replayed samples; compare legacy / ports / sglang
import sys, glob, torch, hashlib, json
from transformers import AutoModelForCausalLM
torch.set_num_threads(64)
snap = glob.glob("/home/michael/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")[0]
model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float32, attn_implementation="eager").eval()
def load(run, isl):
    out = {}
    for f in sorted(glob.glob(f"{run}/work/seed-17/yeto-federated-m2/island-{isl}/tfdiag/*.tfd")):
        r = torch.load(f, weights_only=False)
        for j, tok in enumerate(r["unconcat_tokens"]):
            out[hashlib.sha1(tok.numpy().tobytes()).hexdigest()[:16]] = (tok, r["response_lengths"][j], r["loss_masks"][j].bool(), r["train_log_probs"][j].double(), r["rollout_log_probs"][j].double(), r["advantages"][j].double())
    return out
L, P = sys.argv[1], sys.argv[2]; res = {}
for isl in (0, 1):
    A, B = load(L, isl), load(P, isl); acc = {k: [] for k in ("L", "P", "S", "LP")}; sgn = {"L": [], "P": []}
    for k in A:
        tok, R, m, lL, sL, adv = A[k]; lP = B[k][3]
        with torch.no_grad():
            logits = model(tok[None]).logits[0].double()
        lp = torch.log_softmax(logits[-R-1:-1], -1).gather(-1, tok[-R:, None])[:, 0]
        acc["L"].append((lL - lp)[m]); acc["P"].append((lP - lp)[m]); acc["S"].append((sL - lp)[m]); acc["LP"].append((lL - lP)[m])
    r = {}
    for n, lab in (("L", "legacy-bf16 - fp32"), ("P", "ports-bf16 - fp32"), ("S", "sglang - fp32"), ("LP", "legacy - ports")):
        v = torch.cat(acc[n]); r[lab] = dict(mean_abs=v.abs().mean().item(), max_abs=v.abs().max().item(), mean_signed=v.mean().item(), rms=v.pow(2).mean().sqrt().item())
        print(f"island {isl} {lab:20s} " + " ".join(f"{a}={b:.4e}" for a, b in r[lab].items()), flush=True)
    res[isl] = r
json.dump(res, open(sys.argv[3], "w"), indent=1)
