"""Fix for g1-observe attempt 1 (HF RemoteProtocolError while loading gsm8k): fetch model and
dataset once, with retries, before the runs (cache shared by later runs)."""
import time
for attempt in range(5):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
        import datasets
        datasets.load_dataset("openai/gsm8k", "main", split="train")
        print("predownload ok, attempt", attempt + 1); break
    except Exception as exc:  # noqa: BLE001
        print("predownload attempt", attempt + 1, "failed:", type(exc).__name__); time.sleep(20)
else:
    raise SystemExit(1)
