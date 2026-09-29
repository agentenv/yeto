set -e
export PATH=$HOME/.cargo/bin:$PATH
cd /work/yeto && cargo build --release --locked --quiet 2>&1 | grep -v warning | tail -3; ls target/release | head
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
nvidia-smi -L
echo SETUP_OK
