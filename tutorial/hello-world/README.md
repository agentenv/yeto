# Hello World: Train Gemma on Codex Traces

This tutorial turns local Codex traces into Yeto chat JSONL, optionally backfills
missing reasoning with an OpenAI-compatible teacher, trains a Gemma chat model,
and runs a small Codex-style qualitative eval.

The main recipe is a full SFT of `google/gemma-4-12B-it` on 8x H100 80GB
(assistant-only loss, target-aware packed blocks). Qwen3-0.6B LoRA is a cheap
local smoke test for the pipeline, not a quality target.

## 0. Prereqs

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

You also need:

- local Codex traces, usually under `~/.codex/sessions`
- a teacher endpoint if you want reasoning backfill
- Hugging Face access for `google/gemma-4-12B-it`
- 8x H100 80GB for the full Gemma run below

If you are only testing the pipeline, use the Qwen LoRA smoke command near the
end.

## 1. Configure The Teacher

The teacher only needs to be OpenAI-compatible. It can be DeepSeek, vLLM,
SGLang, or any endpoint that serves `/chat/completions`.

```bash
export BASE_URL="https://api.deepseek.com"
export AUTH_TOKEN="sk-..."

curl "$BASE_URL/models" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -H "Content-Type: application/json"
```

Pass the model name the endpoint serves as `--teacher-model`.

## 2. Convert Codex Traces

This creates a Yeto JSONL file from local Codex sessions. Encrypted reasoning is
never copied directly into the training target. When teacher backfill is enabled,
the converter skips encrypted blobs and asks the teacher for synthetic reasoning
from visible context.

```bash
python -m yeto.codex_traces \
  --input "$HOME/.codex/sessions" \
  --output /tmp/codex-teacher-yeto.jsonl \
  --include-thinking \
  --reasoning-policy teacher-backfill \
  --teacher-base-url "$BASE_URL" \
  --teacher-api-key "$AUTH_TOKEN" \
  --teacher-model <teacher-model> \
  --teacher-api-format chat-completions \
  --teacher-max-output-tokens 96
```

If you do not have a teacher yet, use `--reasoning-policy skip` instead:

```bash
python -m yeto.codex_traces \
  --input "$HOME/.codex/sessions" \
  --output /tmp/codex-skip-reasoning-yeto.jsonl \
  --include-thinking \
  --reasoning-policy skip
```

Sanity-check the output:

```bash
python - <<'PY'
import json
path = "/tmp/codex-teacher-yeto.jsonl"
rows = [json.loads(line) for line in open(path)]
print("rows", len(rows))
print("first metadata", rows[0].get("metadata", {}))
print("first roles", [m.get("role") for m in rows[0].get("messages", [])])
PY
```

## 3. Check Target Packing

Before training, make sure assistant-only packing produces real target tokens.
This catches the class of failures where almost every block has zero loss.

```bash
python tutorial/hello-world/probe_target_packing.py \
  --model google/gemma-4-12B-it \
  --data /tmp/codex-teacher-yeto.jsonl \
  --seq-len 512 \
  --blocks 100
```

A healthy run should report `zero_blocks 0` and a non-trivial target fraction.
If most blocks have zero loss, training looks successful while learning almost
nothing.

## 4. Gemma Full-SFT Smoke

Run a short 20-step smoke first. This confirms the tokenizer, FSDP, target
packing, and checkpoint path all work before spending time on the real run.

```bash
cd /path/to/yeto

MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29511 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=8 --master_port=29511 \
  -m yeto.learner \
  --model google/gemma-4-12B-it \
  --data /tmp/codex-teacher-yeto.jsonl \
  --syncer none \
  --learner-id 0 \
  --num-learners 1 \
  --tuning full \
  --shard fsdp \
  --seq-len 512 \
  --micro-batch-size 1 \
  --grad-accum 8 \
  --inner-lr 1e-6 \
  --weight-decay 0.0 \
  --warmup-steps 20 \
  --train-on assistant \
  --max-local-steps 20 \
  --fragments 8 \
  --gradient-checkpointing on \
  --stream-workers 0 \
  --output-dir /tmp/gemma4-12b-targetpacked-full-sft-smoke
```

You want to see real positive losses and a final `saved model to ...` line.

## 5. Gemma Full SFT

Once the smoke passes, run the 500-step job:

```bash
cd /path/to/yeto

MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29512 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=8 --master_port=29512 \
  -m yeto.learner \
  --model google/gemma-4-12B-it \
  --data /tmp/codex-teacher-yeto.jsonl \
  --syncer none \
  --learner-id 0 \
  --num-learners 1 \
  --tuning full \
  --shard fsdp \
  --seq-len 512 \
  --micro-batch-size 1 \
  --grad-accum 8 \
  --inner-lr 1e-6 \
  --weight-decay 0.0 \
  --warmup-steps 50 \
  --train-on assistant \
  --max-local-steps 500 \
  --fragments 8 \
  --gradient-checkpointing on \
  --stream-workers 0 \
  --output-dir /tmp/gemma4-12b-targetpacked-full-sft-500-lr1e-6
```

The run ends with a `saved model to <output-dir>` line.

## 6. Evaluate

Run the 20-prompt Codex-style eval:

```bash
python tutorial/hello-world/eval_codex_prompts.py \
  --base-model google/gemma-4-12B-it \
  --candidate /tmp/gemma4-12b-targetpacked-full-sft-500-lr1e-6 \
  --candidate-kind full \
  --output /tmp/gemma4-12b-targetpacked-full-sft-500-lr1e-6-eval.jsonl \
  --max-new-tokens 192 \
  --limit 20
```

For a quick readout:

```bash
python - <<'PY'
import json
path = "/tmp/gemma4-12b-targetpacked-full-sft-500-lr1e-6-eval.jsonl"
for i, line in enumerate(open(path), 1):
    row = json.loads(line)
    print(f"\n[{i}] {row['prompt']}")
    print(row["candidate"][:900])
PY
```

What to look for:

- no leaked `<think>` blocks in normal assistant responses
- answers that sound more like code-review/debugging help than generic wiki text
- specific operational advice for traces, adapters, backfills, tests, and evals
- fewer invented commands or fake flags

## 7. Optional Local Smoke: Qwen3-0.6B LoRA

This runs on much smaller hardware and is useful for validating data and code,
but do not treat it as the final quality target.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m yeto.learner \
  --model Qwen/Qwen3-0.6B \
  --data /tmp/codex-teacher-yeto.jsonl \
  --syncer none \
  --learner-id 0 \
  --num-learners 1 \
  --tuning lora \
  --shard ddp \
  --seq-len 256 \
  --micro-batch-size 1 \
  --grad-accum 8 \
  --inner-lr 3e-6 \
  --lora-r 16 \
  --lora-targets all-linear \
  --train-on assistant \
  --max-local-steps 100 \
  --fragments 1 \
  --gradient-checkpointing on \
  --stream-workers 0 \
  --output-dir /tmp/qwen3-0.6b-local-targetpacked-lora-100-r16-lr3e-6
```

If losses stay at `-0.0000`, stop and rerun the target-packing probe. That means
the model is probably not seeing enough weighted assistant target tokens.

## 8. Save Artifacts

The full Gemma checkpoint is large. Prefer keeping it on the cloud volume. If
you must copy it down, use a resumable transfer:

```bash
tar -cf ~/gemma4-12b-targetpacked-full-sft-500-lr1e-6.tar \
  -C /tmp gemma4-12b-targetpacked-full-sft-500-lr1e-6

rsync -avP --inplace \
  <user>@<host>:gemma4-12b-targetpacked-full-sft-500-lr1e-6.tar \
  ~/Downloads/
```

The eval JSONL is tiny and should always be copied or saved with the run.

