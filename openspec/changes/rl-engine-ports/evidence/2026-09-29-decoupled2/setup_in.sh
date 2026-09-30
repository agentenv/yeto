set -e
cd /opt
[ -d miles-next ] || git clone -q https://github.com/michaellchung/miles /opt/miles-next
cd /opt/miles-next && git checkout -q --detach 0394715083c91182b5eb0c526eeee4196ac694b9
[ -d /opt/sglang-next ] || git clone -q --filter=blob:none https://github.com/michaellchung/sglang /opt/sglang-next
cd /opt/sglang-next && git checkout -q --detach 9f29303bef1eea38eb613e5f454a52db1326422d
cd /opt/sglang-next/python && pip install --no-deps -e . 2>&1 | tail -1
cd /opt/miles-next && pip install --no-deps -e . 2>&1 | tail -1
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal 2>&1 | tail -1
pip list 2>/dev/null | grep -i -E '^(sglang|miles|torch|peft|megatron) ' || true
mkdir -p /work/data
python - <<'PY'
import json
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="train")
with open("/work/data/gsm8k.jsonl","w") as f:
    for r in ds:
        f.write(json.dumps({"messages":[{"role":"user","content":r["question"]+"\nPut the final numeric answer in \\boxed{}."}],"label":r["answer"].split("####")[-1].strip().replace(",","")})+"\n")
print("rows", len(ds))
PY
echo SETUP_OK
