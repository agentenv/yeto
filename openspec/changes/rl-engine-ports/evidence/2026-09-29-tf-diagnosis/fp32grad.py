# tf-diagnosis: CPU fp32 reference LoRA gradient of the GRPO loss on the replayed batch (same initial LoRA from audit base.f32)
import sys, glob, json, torch, numpy as np
from transformers import AutoModelForCausalLM
torch.set_num_threads(96)
EV = "/home/michael/work/gpu-eq/evidence"
snap = "/home/michael/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
isl = int(sys.argv[1]); dumprun = sys.argv[2]
base_dir = f"{EV}/2026-09-29-eq62-v2/tf-legacy/work/seed-17/yeto-federated-m2/island-{isl}/audit/round-00000001"
meta = json.load(open(base_dir + ".json")); base = np.fromfile(base_dir + ".base.f32", dtype="<f4")
m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32, attn_implementation="eager")
for p in m.parameters(): p.requires_grad_(False)
params = {}
T = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
for name, mod in list(m.named_modules()):
    if name.split(".")[-1] in T and isinstance(mod, torch.nn.Linear):
        A = torch.nn.Parameter(torch.zeros(16, mod.in_features)); B = torch.nn.Parameter(torch.zeros(mod.out_features, 16))
        mod.register_parameter("lora_A_w", A); mod.register_parameter("lora_B_w", B)
        mod.register_forward_hook(lambda mo, inp, out: out + (inp[0] @ mo.lora_A_w.t()) @ mo.lora_B_w.t())  # alpha/r = 1
        params["base_model.model." + name + ".lora_A.weight"] = A; params["base_model.model." + name + ".lora_B.weight"] = B
off = 0
for sp in meta["specs"]:
    v = torch.from_numpy(base[off:off+sp["numel"]].copy()).view(sp["shape"]); off += sp["numel"]
    with torch.no_grad(): params[sp["name"]].copy_(v)
assert off == base.size and set(params) == {s["name"] for s in meta["specs"]}
m.train()  # dropout is 0 in Qwen3 config
recs = []
for f in sorted(glob.glob(f"{EV}/2026-09-29-tf-diagnosis/{dumprun}/work/seed-17/yeto-federated-m2/island-{isl}/tfdiag/*.tfd")):
    r = torch.load(f, weights_only=False)
    for j, tok in enumerate(r["unconcat_tokens"]):
        recs.append((tok, r["response_lengths"][j], r["loss_masks"][j].float(), r["advantages"][j].float()))
N = len(recs); assert N == 32, N
for tok, R, mask, adv in recs:
    logits = m(input_ids=tok[None]).logits[0]
    lp = torch.log_softmax(logits[-R-1:-1].float(), -1).gather(-1, tok[-R:, None])[:, 0]
    ratio = torch.exp(lp - lp.detach())
    loss = ((-adv * ratio) * mask).sum() / mask.sum().clamp_min(1) / N
    loss.backward()
names = sorted(params); out = np.concatenate([params[n].grad.detach().numpy().astype("<f4").ravel() for n in names])
gmeta = json.load(open(base_dir + ".grad.json")); assert [s["name"] for s in gmeta["specs"]] == names
import os; d = f"{EV}/2026-09-29-tf-diagnosis/fp32ref/work/seed-17/yeto-federated-m2/island-{isl}/audit"; os.makedirs(d, exist_ok=True)
out.tofile(d + "/round-00000001.grad.f32"); json.dump(gmeta | {"engine": "cpu-fp32-hf-peft"}, open(d + "/round-00000001.grad.json", "w"))
print("island", isl, "fp32 grad norm", float(np.linalg.norm(out.astype(np.float64))))
