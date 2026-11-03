import json, sys, urllib.request
from transformers import AutoTokenizer
PROMPTS = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
           "Write the first ten prime numbers separated by commas.",
           "Explain in two sentences why the sky is blue.",
           "Translate to French: The weather is nice today and we will go for a walk.",
           "A train travels 60 miles per hour for 3 hours. How far does it travel?",
           "List three uses of a paperclip.",
           "What is 17 multiplied by 23? Show the steps.",
           "Summarize the plot of Romeo and Juliet in one sentence."] * 4
url, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
ref = json.load(open(sys.argv[4] if len(sys.argv) > 4 else "ref.json"))["ref"]["outputs"]
res = []
for p, o in zip(PROMPTS, ref):
    ids = tok(p).input_ids + o
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True, "logprob_start_len": len(ids) - len(o) - 1}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
    res.append([x[0] for x in r["meta_info"]["input_token_logprobs"]])
try: d = json.load(open(out))
except Exception: d = {}
d[label] = res; json.dump(d, open(out, "w"))
print("LP_DONE", label, len(res))
