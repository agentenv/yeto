#!/usr/bin/env python3
"""Cross-hardware teacher-forced log-prob comparison (sampler vs trainer vs reference).

Measures the numeric noise floor of each scoring side (repeat launches of the same side)
and the bias between sides, in the metrics the RL breakers judge
(``train_rollout_logprob_abs_diff``, ``train_rollout_kl``), so ``yeto breaker`` /
``YETO_RL_XHW_GUARD`` limits for a new card can be set from data.

Subcommands (each writes / reads plain files, so sides can run on different machines):

  prompts  --model M --out prompts.jsonl [--n 64]
      fixed, deterministic chat prompts (math / code / prose), token ids under M's template.
  gen      --model M --prompts prompts.jsonl --out data.jsonl [--max-tokens 256 --seed 0]
      vLLM sampling (T=1) of one completion per prompt; the completions are then frozen.
      Also stores the sampler's own generation-time log-probs of the sampled tokens.
  score    --backend {vllm,hf} --model M --data data.jsonl --out run.pt
           [--device npu|cuda|cpu --dtype bf16|fp32 --batch-size 8 --topk 20]
      teacher-forced scoring of prompt+completion; per completion token: the sampled token's
      log-prob, the argmax id, and the top-k (ids, log-probs). vllm: prompt_logprobs (prefill
      path); hf: transformers forward, log_softmax in fp32 (the trainer path).
  compare  A.pt B.pt [C.pt ...] [--json out.json]
      pairwise metrics over completion tokens:
        abs_diff  mean over samples of the per-sample mean |dlogp| (miles' sum_of_sample_mean)
        k3        mean per-token Schulman k3 of KL(a || b) at the sampled token, clamp [-10, 10]
        topk_kl   truncated KL(a || b) over a's top-k (b's missing ids take b's k-th log-prob)
        p50/p99/max of the per-token |dlogp|, argmax agreement.

Nothing here imports yeto: copy the file next to any engine install and run it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time


# ----------------------------------------------------------------------------- prompts
_MATH = [
    "A train travels {a} km in {b} hours. What is its average speed in km/h? Show your steps.",
    "Tom has {a} apples and gives {b} to each of his 3 friends. How many apples are left?",
    "Compute {a} * {b} - {a} + {b}, explaining each step.",
    "If a rectangle has sides {a} cm and {b} cm, what are its area and perimeter?",
]
_CODE = [
    "Write a Python function that returns the {a}-th Fibonacci number modulo {b}.",
    "Explain what this code prints: `print(sum(i*i for i in range({a})) % {b})`.",
    "Write a bash one-liner that counts lines containing the number {a} in all .log files, "
    "then prints the count times {b}.",
    "Implement binary search in C over an array of {a} sorted ints; return -{b} if absent.",
]
_PROSE = [
    "Write a short story ({a} sentences) about a lighthouse keeper who finds {b} bottles.",
    "Summarize the causes of the French Revolution in {a} bullet points, then give {b} dates.",
    "Translate into French and explain the grammar: 'I have lived here for {a} years and {b} days.'",
    "Give {a} tips for learning a new language, each under {b} words.",
]


def fixed_prompts(n: int) -> list[str]:
    nums = [(3, 7), (12, 5), (48, 9), (100, 13), (7, 64), (25, 4)]
    out = []
    for tpl, (a, b) in itertools.product(_MATH + _CODE + _PROSE, nums):
        out.append(tpl.format(a=a, b=b))
    # interleave so any prefix mixes the three kinds
    out = [out[i] for i in sorted(range(len(out)), key=lambda i: (i % len(nums), i))]
    if n > len(out):
        raise SystemExit(f"--n {n} > {len(out)} fixed prompts")
    return out[:n]


def cmd_prompts(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    with open(a.out, "w") as fh:
        for i, text in enumerate(fixed_prompts(a.n)):
            msgs = [{"role": "user", "content": text}]
            kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)
            if hasattr(ids, "keys"):  # transformers 5 returns a BatchEncoding
                ids = ids["input_ids"]
            fh.write(json.dumps({"id": i, "text": text, "prompt_ids": list(map(int, ids))}) + "\n")
    print(f"wrote {a.n} prompts -> {a.out}")


def _load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ----------------------------------------------------------------------------- vllm
def _llm(a):
    from vllm import LLM
    kw = dict(model=a.model, dtype="bfloat16" if a.dtype == "bf16" else "float32", seed=a.seed,
              gpu_memory_utilization=a.gmu, enable_prefix_caching=False, max_model_len=a.max_model_len)
    if a.enforce_eager:
        kw["enforce_eager"] = True
    return LLM(**kw)


def cmd_gen(a):
    from vllm import SamplingParams
    rows = _load(a.prompts)
    llm = _llm(a)
    sp = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, logprobs=0,
                         seed=a.seed + r["id"]) for r in rows]
    outs = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in rows], sp)
    with open(a.out, "w") as fh:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            lps = [float(d[t].logprob) for t, d in zip(c.token_ids, c.logprobs)]
            fh.write(json.dumps({**r, "completion_ids": list(map(int, c.token_ids)), "gen_logprobs": lps}) + "\n")
    print(f"wrote {len(rows)} completions -> {a.out}")


def _score_vllm(a, rows):
    from vllm import SamplingParams
    llm = _llm(a)
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=a.topk, detokenize=False)
    outs = llm.generate([{"prompt_token_ids": r["prompt_ids"] + r["completion_ids"]} for r in rows], sp)
    res = []
    for r, o in zip(rows, outs):
        p, c = len(r["prompt_ids"]), r["completion_ids"]
        pl = o.prompt_logprobs[p:p + len(c)]
        tok_lp, top_ids, top_lp = [], [], []
        for t, d in zip(c, pl):
            tok_lp.append(float(d[t].logprob))
            items = sorted(d.items(), key=lambda kv: kv[1].rank)
            items = [kv for kv in items if kv[1].rank <= a.topk][:a.topk]
            top_ids.append([int(k) for k, _ in items])
            top_lp.append([float(v.logprob) for _, v in items])
        res.append((tok_lp, top_ids, top_lp))
    return res


# ----------------------------------------------------------------------------- hf
def _score_hf(a, rows):
    import torch
    from transformers import AutoModelForCausalLM
    dev = a.device
    if dev == "npu":
        import torch_npu  # noqa: F401
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt, attn_implementation=a.attn)
    model.to(dev).eval()
    res = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["prompt_ids"]) - len(rows[i]["completion_ids"]))
    with torch.no_grad():
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            seqs = [rows[i]["prompt_ids"] + rows[i]["completion_ids"] for i in idx]
            L = max(map(len, seqs))
            ids = torch.zeros(len(seqs), L, dtype=torch.long)
            att = torch.zeros(len(seqs), L, dtype=torch.long)
            for j, q in enumerate(seqs):
                ids[j, :len(q)] = torch.tensor(q)
                att[j, :len(q)] = 1
            logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits
            for j, i in enumerate(idx):
                p, c = len(rows[i]["prompt_ids"]), rows[i]["completion_ids"]
                lg = logits[j, p - 1:p - 1 + len(c)].float()
                lp = torch.log_softmax(lg, dim=-1)
                ct = torch.tensor(c, device=lp.device)
                tok = lp.gather(1, ct[:, None])[:, 0]
                tv, ti = lp.topk(a.topk, dim=-1)
                res[i] = (tok.cpu().tolist(), ti.cpu().tolist(), tv.cpu().tolist())
    return res


def cmd_score(a):
    import torch
    rows = _load(a.data)
    t0 = time.time()
    res = _score_vllm(a, rows) if a.backend == "vllm" else _score_hf(a, rows)
    meta = {"backend": a.backend, "device": a.device, "dtype": a.dtype, "batch_size": a.batch_size,
            "attn": a.attn, "enforce_eager": a.enforce_eager, "seconds": round(time.time() - t0, 2),
            "label": a.label or os.path.basename(a.out)}
    try:
        import vllm
        meta["vllm"] = vllm.__version__
    except Exception:
        pass
    meta["torch"] = torch.__version__
    torch.save({"meta": meta, "ids": [r["id"] for r in rows],
                "tok": [r[0] for r in res], "top_ids": [r[1] for r in res], "top_lp": [r[2] for r in res]},
               a.out)
    print(json.dumps(meta))


def cmd_from_gen(a):
    """The sampler's generation-time log-probs of a data file as a (top-k-less) run."""
    import torch
    rows = _load(a.data)
    torch.save({"meta": {"backend": "vllm-gen", "label": a.label or os.path.basename(a.out)},
                "ids": [r["id"] for r in rows], "tok": [r["gen_logprobs"] for r in rows],
                "top_ids": None, "top_lp": None}, a.out)


# ----------------------------------------------------------------------------- compare
def _pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = math.floor(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def compare(A, B):
    sample_means, absd, k3s, tkl, agree, n_arg = [], [], [], [], 0, 0
    bi = {i: n for n, i in enumerate(B["ids"])}
    for n, i in enumerate(A["ids"]):
        m = bi[i]
        ta, tb = A["tok"][n], B["tok"][m]
        assert len(ta) == len(tb), f"length mismatch on id {i}"
        d = [abs(x - y) for x, y in zip(ta, tb)]
        absd += d
        if d:
            sample_means.append(sum(d) / len(d))
        for x, y in zip(ta, tb):  # x: "rollout" side a, y: "train" side b; k3 = e^r - 1 - r, r = y - x
            r = max(-10.0, min(10.0, y - x))
            k3s.append(min(10.0, math.exp(r) - 1 - r))
        if A.get("top_ids") and B.get("top_ids"):
            for ia, la, ib, lb in zip(A["top_ids"][n], A["top_lp"][n], B["top_ids"][m], B["top_lp"][m]):
                n_arg += 1
                agree += ia[0] == ib[0]
                qb = dict(zip(ib, lb))
                floor = lb[-1]
                tkl.append(sum(math.exp(p) * (p - qb.get(t, floor)) for t, p in zip(ia, la)))
    out = {"tokens": len(absd), "abs_diff": sum(sample_means) / max(1, len(sample_means)),
           "token_mean": sum(absd) / max(1, len(absd)), "k3": sum(k3s) / max(1, len(k3s)),
           "p50": _pct(absd, 0.5), "p99": _pct(absd, 0.99), "max": max(absd) if absd else float("nan")}
    if n_arg:
        out["argmax_agree"] = agree / n_arg
        out["topk_kl"] = sum(tkl) / len(tkl)
    return out


def cmd_compare(a):
    import torch
    runs = [torch.load(p, weights_only=False) for p in a.runs]
    rows = []
    for (pa, A), (pb, B) in itertools.combinations(zip(a.runs, runs), 2):
        m = compare(A, B)
        m.update(a=A["meta"].get("label", pa), b=B["meta"].get("label", pb))
        rows.append(m)
        print(f"{m['a']:>28} vs {m['b']:<28} abs_diff {m['abs_diff']:.5f}  k3 {m['k3']:.6f}  "
              f"p50 {m['p50']:.5f}  p99 {m['p99']:.4f}  max {m['max']:.3f}  "
              f"argmax {m.get('argmax_agree', float('nan')):.4f}  topk_kl {m.get('topk_kl', float('nan')):.6f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True)
        p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--gmu", type=float, default=0.5)
        p.add_argument("--max-model-len", type=int, default=2048)
        p.add_argument("--enforce-eager", action="store_true")

    p = sub.add_parser("prompts")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=64)
    p.set_defaults(fn=cmd_prompts)
    p = sub.add_parser("gen")
    common(p)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=256)
    p.set_defaults(fn=cmd_gen)
    p = sub.add_parser("score")
    common(p)
    p.add_argument("--backend", required=True, choices=["vllm", "hf"])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--attn", default="eager")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--label", default="")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("from-gen")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="")
    p.set_defaults(fn=cmd_from_gen)
    p = sub.add_parser("compare")
    p.add_argument("runs", nargs="+")
    p.add_argument("--json", default="")
    p.set_defaults(fn=cmd_compare)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
