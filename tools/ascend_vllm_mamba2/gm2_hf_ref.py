"""G-M2 reference: HF (remote NemotronH code, torch path) on CPU, teacher-forced on vLLM's tokens.
usage: python gm2_hf_ref.py <vllm_json> <out_json>   env MODEL, DTYPE (float32|bfloat16), THREADS"""
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM

def main():
    src, dst = sys.argv[1], sys.argv[2]
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    dtype = getattr(torch, os.environ.get("DTYPE", "float32"))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(os.environ["MODEL"], trust_remote_code=True, torch_dtype=dtype)
    model.eval()
    print("loaded", time.time() - t0, "fast_path", getattr(sys.modules[type(model).__module__], "is_fast_path_available", None), flush=True)
    data = json.load(open(src))
    res = []
    for it in data["items"]:
        p, g = it["prompt_token_ids"], it["token_ids"]
        ids = torch.tensor([p + g])
        t1 = time.time()
        with torch.no_grad():
            logits = model(ids, use_cache=False).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        n = len(p)
        ref = [lp[n - 1 + i, t].item() for i, t in enumerate(g)]
        am = [int(lp[n - 1 + i].argmax()) for i in range(len(g))]
        d = [abs(a - b) for a, b in zip(ref, it["logprobs"])]
        r = {"prompt_len": n, "gen_len": len(g), "ref_logprobs": ref, "vllm_logprobs": it["logprobs"],
             "max_abs": max(d), "mean_abs": sum(d) / len(d), "argmax_agree": sum(int(a == t) for a, t in zip(am, g)) / len(g),
             "seconds": time.time() - t1}
        print({k: v for k, v in r.items() if not k.endswith("logprobs")}, flush=True)
        res.append(r)
    json.dump({"src": src, "dtype": str(dtype), "items": res}, open(dst, "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
