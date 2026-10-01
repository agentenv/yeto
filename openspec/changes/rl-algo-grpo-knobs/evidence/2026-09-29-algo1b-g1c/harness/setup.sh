set -eo pipefail  # review: a failing cargo build was masked by the pipe (attempt 1/2 ran with plain set -e)
export PATH=$HOME/.cargo/bin:$PATH
# (no-sync single island: no syncer binary needed; the R0 cargo build line ran in the wrong directory and was masked by the pipe before pipefail)
python - <<'PY'
import json
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="train")
import os; os.makedirs("/work/data",exist_ok=True)
with open("/work/data/gsm8k.jsonl","w") as f:
    for r in ds:
        f.write(json.dumps({"messages":[{"role":"user","content":r["question"]+"\nPut the final numeric answer in \\boxed{}."}],"label":r["answer"].split("####")[-1].strip().replace(",","")})+"\n")
print("rows", len(ds))
PY
cd /opt/miles-next && git rev-parse HEAD; cd /opt/sglang-next && git rev-parse HEAD
nvidia-smi -L; nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100" && ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H200 || { echo GPU_ASSERT_FAILED; exit 3; }
echo SETUP_OK
