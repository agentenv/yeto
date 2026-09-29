#!/bin/bash
# Inside the sandbox. usage: run_all.sh "<run names>"
set -u
cd /work
nvidia-smi --query-gpu=name --format=csv,noheader | tee /work/out/gpu.txt
grep -q "H100" /work/out/gpu.txt || { echo "GPU_ASSERT_FAILED"; exit 90; }
cat /opt/yeto/image-manifest.json > /work/out/image-manifest.json 2>/dev/null
( cd /root/miles && git rev-parse HEAD ) > /work/out/miles_head.txt
python - <<'PY'
import json, os
import datasets
ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
os.makedirs("/work/data", exist_ok=True)
rows = [{"messages": [{"role": "user", "content": r["question"] + "\nPut the final numeric answer in \\boxed{}."}],
         "label": r["answer"].split("####")[-1].strip(), "metadata": {"benchmark_prompt_id": i}}
        for i, r in enumerate(ds.select(range(12)))]
open("/work/data/prompts.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
print("prompts", len(rows))
PY
for r in $1; do
  echo "=== RUN $r $(date -u +%FT%TZ)"
  python /work/g1/g1.py $r /work/out/$r > /work/out/$r.run.log 2>&1; echo "rc=$?" | tee /work/out/$r.rc
  tail -3 /work/out/$r.run.log
  ray stop --force >/dev/null 2>&1; pkill -9 -f sglang >/dev/null 2>&1; sleep 5
done
tar czf /tmp/out.tgz --exclude='*.pt' --exclude='*.safetensors' --exclude='*.f32' --exclude='state.ckpt*' -C /work out
echo ALL_DONE
