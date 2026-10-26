"""G-M2: NemotronH (Nano) generation on Ascend with vLLM + yeto Mamba2 patch. Writes JSON to $OUT."""
import json, os, time

def main():
    t0 = time.time()
    from vllm import LLM, SamplingParams
    MODEL = os.environ["MODEL"]; OUT = os.environ["OUT"]
    TP = int(os.environ.get("TP", "4")); MAXTOK = int(os.environ.get("MAXTOK", "64"))
    NBATCH = int(os.environ.get("NBATCH", "0"))
    PROMPTS = [
        "The capital of France is",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        "Question: A train travels 120 km in 1.5 hours. What is its average speed in km/h?\nAnswer:",
        "Write a short poem about the ocean.\n",
    ]
    LONG = os.environ.get("LONG", "1") == "1"
    if LONG:  # one prompt longer than chunk_size (128) so the inter-chunk path is exercised
        PROMPTS.append(("In the beginning of the history of computing, " * 40) + "\nSummarize the text above in one sentence:")
    llm = LLM(model=MODEL, tensor_parallel_size=TP, trust_remote_code=True, dtype="bfloat16",
              max_model_len=int(os.environ.get("MAXLEN", "4096")), max_num_seqs=int(os.environ.get("MAXSEQ", "32")),
              gpu_memory_utilization=float(os.environ.get("GMU", "0.8")), enforce_eager=os.environ.get("EAGER", "1") == "1",
              enable_prefix_caching=False, seed=1234)
    t1 = time.time()
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, logprobs=0, seed=1234)
    outs = llm.generate(PROMPTS, sp)
    t2 = time.time()
    res = {"t_import_init": t1 - t0, "t_gen": t2 - t1, "tp": TP, "items": []}
    for p, o in zip(PROMPTS, outs):
        c = o.outputs[0]
        res["items"].append({"prompt": p, "prompt_token_ids": list(o.prompt_token_ids), "token_ids": list(c.token_ids),
                             "logprobs": [lp[t].logprob for lp, t in zip(c.logprobs, c.token_ids)], "text": c.text})
        print("PROMPT:", repr(p[:60]), "->", repr(c.text[:200]), flush=True)
    if NBATCH:
        bp = [f"Story {i}: Once upon a time in a small village," for i in range(NBATCH)]
        spb = SamplingParams(temperature=0.8, max_tokens=MAXTOK, ignore_eos=True, seed=7)
        llm.generate(bp[:2], SamplingParams(max_tokens=4))
        tb = time.time(); ob = llm.generate(bp, spb); tb2 = time.time()
        ntok = sum(len(o.outputs[0].token_ids) for o in ob)
        res["batch"] = {"n": NBATCH, "out_tokens": ntok, "seconds": tb2 - tb, "tok_per_s": ntok / (tb2 - tb)}
        print("BATCH", res["batch"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("DONE", res["t_import_init"], res["t_gen"], flush=True)

if __name__ == "__main__":
    main()
