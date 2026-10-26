"""Reload test: load the checkpoint a second time through model.load_weights (what verl's sync does)."""
import os

def reload(self):
    from safetensors.torch import load_file
    model = self.model_runner.model
    info = []
    for n, p in model.named_parameters():
        if "experts" in n:
            info.append((n, tuple(p.shape), hasattr(p, "weight_loader"), type(p).__name__))
    sd = load_file(os.environ["MODEL"] + "/model.safetensors")
    try:
        import sys
        sys.path.insert(0, "/data/work/s28-gm2/verl-fork")
        from verl.utils.vllm.vllm_unquant_utils import stage_unquantized_moe_params, fold_unquantized_moe_params
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        staged = stage_unquantized_moe_params(model)
        st = [(n, tuple(p.shape), hasattr(p, "weight_loader"), getattr(getattr(p, "weight_loader", None), "__name__", None))
              for n, p in model.named_parameters() if "experts.w13" in n][:1]
        model.load_weights(list(sd.items()))
        with fold_unquantized_moe_params(staged):
            process_weights_after_loading(model, self.model_runner.model_config, self.device)
        return ("ok", len(staged), st, info[:4])
    except Exception as e:
        return (f"FAIL {type(e).__name__}: {e}", info[:4])

def main():
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from vllm import LLM, SamplingParams
    llm = LLM(model=os.environ["MODEL"], tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.4,
              max_model_len=1024, enable_prefix_caching=False, load_format=os.environ.get("LOADFMT", "auto"), enforce_eager=os.environ.get("EAGER", "1") == "1") if False else LLM(model=os.environ["MODEL"], tensor_parallel_size=1, gpu_memory_utilization=0.4, max_model_len=1024, enable_prefix_caching=False, load_format=os.environ.get("LOADFMT", "auto"), enforce_eager=os.environ.get("EAGER", "1") == "1", enable_sleep_mode=os.environ.get("SLEEP") == "1")
    sp = SamplingParams(temperature=0, max_tokens=8, logprobs=0)
    a = llm.generate(["The capital of France is"], sp)[0].outputs[0]
    if os.environ.get("SLEEP") == "1":
        llm.sleep(level=1)
        llm.wake_up(tags=["weights"])
    print("RELOAD", llm.collective_rpc(reload), flush=True)
    if os.environ.get("SLEEP") == "1":
        llm.wake_up(tags=["kv_cache"])
    b = llm.generate(["The capital of France is"], sp)[0].outputs[0]
    print("SAME_AFTER_RELOAD", a.token_ids == b.token_ids, [x[t].logprob for x, t in zip(a.logprobs, a.token_ids)][:3],
          [x[t].logprob for x, t in zip(b.logprobs, b.token_ids)][:3], flush=True)

if __name__ == "__main__":
    main()
