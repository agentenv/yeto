#!/usr/bin/env python3
"""m5 generation-only probe client (head node): greedy /generate of fixed prompts against a running sglang
server, records output token ids + throughput. usage: s1m5gen.py <url> <label> <tp> <nnodes> <out.json>
(merges {label: {...}} into out.json). Prompts are fixed; temperature 0, max_new_tokens 64, ignore_eos."""
import json, os, sys, time, urllib.request

PROMPTS = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
           "Write the first ten prime numbers separated by commas.",
           "Explain in two sentences why the sky is blue.",
           "Translate to French: The weather is nice today and we will go for a walk.",
           "A train travels 60 miles per hour for 3 hours. How far does it travel?",
           "List three uses of a paperclip.",
           "What is 17 multiplied by 23? Show the steps.",
           "Summarize the plot of Romeo and Juliet in one sentence."] * 4


def post(url, body, timeout=600):
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


url, label, tp, nnodes, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
params = {"temperature": 0.0, "max_new_tokens": 64, "ignore_eos": True}
post(url, {"text": PROMPTS[:2], "sampling_params": {**params, "max_new_tokens": 8}})  # warmup
t0 = time.time()
resp = post(url, {"text": PROMPTS, "sampling_params": params})
wall = time.time() - t0
outs, texts, toks = [], [], 0
for r in resp:
    ids = r.get("output_ids")
    outs.append(list(ids) if ids is not None else None)
    texts.append(r.get("text", ""))
    toks += int((r.get("meta_info") or {}).get("completion_tokens") or len(ids or []))
rec = {"tp": tp, "nnodes": nnodes, "n_prompts": len(PROMPTS), "wall_s": round(wall, 3), "completion_tokens": toks,
       "tokens_per_s": round(toks / wall, 1) if wall > 0 else None,
       "outputs": [o if o is not None else t for o, t in zip(outs, texts)], "texts": [t[:200] for t in texts]}
data = json.load(open(out)) if os.path.exists(out) else {}
data[label] = rec
json.dump(data, open(out, "w"))
print(json.dumps({k: v for k, v in rec.items() if k not in ("outputs", "texts")}))
