#!/usr/bin/env bash

# Runs inside the pinned Miles container. It never starts SGLang or a rollout.
set -euo pipefail

: "${YETO_FULL_PARAMETER_PROBE_EVIDENCE:?missing probe evidence path}"
: "${YETO_FULL_PARAMETER_MODEL_REVISION:?missing model revision}"
: "${YETO_FULL_PARAMETER_CONFIG_HASH:?missing model config hash}"
: "${YETO_FULL_PARAMETER_FRAGMENT_COUNT:?missing fragment count}"
: "${YETO_FULL_PARAMETER_CONVERSION_MANIFEST_SHA256:?missing conversion manifest hash}"
: "${YETO_MILES_IMAGE_DIGEST:?missing Miles image digest}"

test -d /root/miles
test -d /root/yeto
test -d /models/hf
test -f /models/torch_dist/latest_checkpointed_iteration.txt
test "$(cat /models/torch_dist/latest_checkpointed_iteration.txt)" = release
test -f /models/torch_dist/conversion-manifest.json
test -f /root/yeto/tests/fixtures/qwen35_full_parameter_probe.jsonl
test ! -e "${YETO_FULL_PARAMETER_PROBE_EVIDENCE}"
if pgrep -f '(^|/)(raylet|gcs_server)( |$)' >/dev/null; then
  echo "refusing to share an existing Ray runtime" >&2
  exit 1
fi

python /root/yeto/tools/probes/qwen35_conversion_manifest.py verify \
  --model /models/hf \
  --manifest /models/torch_dist/conversion-manifest.json \
  --expected-sha256 "${YETO_FULL_PARAMETER_CONVERSION_MANIFEST_SHA256}" \
  --expected-revision "${YETO_FULL_PARAMETER_MODEL_REVISION}" \
  --expected-config-sha256 "${YETO_FULL_PARAMETER_CONFIG_HASH}" \
  --expected-image-digest "${YETO_MILES_IMAGE_DIGEST}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/miles:/root/yeto:/root/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MILES_EXPERIMENTAL_FT_TRAINER=0
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=0
export YETO_FULL_PARAMETER_YETO_SOURCE_ROOT=/root/yeto
export YETO_FULL_PARAMETER_MILES_SOURCE_ROOT=/root/miles

ray_temp=/evidence/ray
test ! -e "${ray_temp}"
mkdir -m 700 "${ray_temp}"

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

ray start \
  --head \
  --node-ip-address=127.0.0.1 \
  --num-gpus=2 \
  --disable-usage-stats \
  --dashboard-host=127.0.0.1 \
  --dashboard-port=8265 \
  --temp-dir="${ray_temp}"

runtime_env_json="$({
  python -c '
import json
import os

names = (
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
    "PYTHONPATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "MILES_EXPERIMENTAL_FT_TRAINER",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR",
    "YETO_FULL_PARAMETER_PROBE_EVIDENCE",
    "YETO_FULL_PARAMETER_MODEL_REVISION",
    "YETO_FULL_PARAMETER_CONFIG_HASH",
    "YETO_FULL_PARAMETER_FRAGMENT_COUNT",
    "YETO_FULL_PARAMETER_CONVERSION_MANIFEST_SHA256",
    "YETO_MILES_IMAGE_DIGEST",
    "YETO_FULL_PARAMETER_YETO_SOURCE_ROOT",
    "YETO_FULL_PARAMETER_MILES_SOURCE_ROOT",
)
print(json.dumps({"env_vars": {name: os.environ[name] for name in names}}))
'
})"

cd /root/miles
# shellcheck source=/dev/null
source scripts/models/qwen3.5-4B.sh

ray job submit \
  --address=http://127.0.0.1:8265 \
  --runtime-env-json="${runtime_env_json}" \
  -- python3 /root/miles/train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 2 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /models/hf \
  --ref-load /models/torch_dist \
  --prompt-data /root/yeto/tests/fixtures/qwen35_full_parameter_probe.jsonl \
  --input-key prompt \
  --label-key label \
  --rm-type deepscaler \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 64 \
  --global-batch-size 1 \
  --micro-batch-size 1 \
  --debug-train-only \
  --disable-rollout-global-dataset \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --seq-length 512 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --external-policy-sync-path \
  yeto.rl.miles_full_parameter_probe.create_full_parameter_probe

test -s "${YETO_FULL_PARAMETER_PROBE_EVIDENCE}"
