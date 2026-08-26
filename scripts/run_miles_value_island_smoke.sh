#!/usr/bin/env bash
set -euo pipefail

MILES_ROOT=${MILES_ROOT:-/data/miles-values-isoloco-20260825}
YETO_ROOT=${YETO_ROOT:-/data/yeto-isoloco-20260825}
SYNCER_ADDR=${SYNCER_ADDR:?SYNCER_ADDR must be host:port}
OUTPUT_DIR=${OUTPUT_DIR:-/data/local-runs/qwen38-value-island-smoke-20260825}
NUM_ROLLOUT=${NUM_ROLLOUT:-2}
# Do not put the ``{rollout_id}`` placeholder inside a ``${var:-default}``
# expansion: bash treats its closing brace as the end of the parameter
# expansion and appends the remainder (for example ``.pt}``) to overrides.
DATA_TEMPLATE=${DATA_TEMPLATE:-}
if [[ -z "${DATA_TEMPLATE}" ]]; then
  DATA_TEMPLATE='/data/shared-runs/offline-data/qwen38-secrl-tb21-native-256k-dp5-tokenbucket-balanced-v8/data_{rollout_id}.pt'
fi
LOCAL_GLOBAL_BATCH_SIZE=${LOCAL_GLOBAL_BATCH_SIZE:-25}

cd "${MILES_ROOT}"
source scripts/models/qwen3.6-27B.sh
mkdir -p "${OUTPUT_DIR}"

TRAIN_ENV_VARS=$(printf '{"PYTHONPATH":"%s:%s","PYTHONFAULTHANDLER":"1","TORCH_SHOW_CPP_STACKTRACES":"1","TORCH_DISABLE_ADDR2LINE":"1","MILES_OFFLINE_VALIDATION_START_ROLLOUT":"%s","NCCL_DEBUG":"WARN","PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512,garbage_collection_threshold:0.8","YETO_VALUE_SYNCER":"%s","YETO_VALUE_LEARNER_ID":"0","YETO_VALUE_NUM_LEARNERS":"1","YETO_VALUE_MERGE_ALPHA":"0.5","YETO_VALUE_STREAMS":"4","YETO_VALUE_CONNECT_TIMEOUT":"900","YETO_VALUE_FINALIZATION_TIMEOUT":"1800"}' \
  "${YETO_ROOT}" "${MILES_ROOT}" "${NUM_ROLLOUT}" "${SYNCER_ADDR}")

export PYTHONPATH="${YETO_ROOT}:${MILES_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MILES_RAY_TARGET_NODE_IP=${MILES_RAY_TARGET_NODE_IP:-current}

exec python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node 8 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /data/models/Qwen3.8-27B \
  --load /data/models/Qwen3.8-27B_torch_dist \
  --critic-load /data/models/Qwen3.8-27B_torch_dist \
  --ref-load /data/models/Qwen3.8-27B_torch_dist \
  --save "${OUTPUT_DIR}/checkpoints" \
  --critic-save "${OUTPUT_DIR}/critic_checkpoints" \
  --save-interval 1000 \
  --prompt-data /data/rl_data/smoke/all3_24.jsonl \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --load-debug-rollout-data "${DATA_TEMPLATE}" \
  --disable-rollout-global-dataset \
  --rollout-num-gpus 0 \
  --rollout-num-gpus-per-engine 1 \
  --use-dynamic-global-batch-size \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${LOCAL_GLOBAL_BATCH_SIZE}" \
  --global-batch-size "${LOCAL_GLOBAL_BATCH_SIZE}" \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 262144 \
  --seq-length 262144 \
  --max-position-embeddings 262144 \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 2 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --sequence-parallel \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --balance-data \
  --max-tokens-per-gpu 10240 \
  --advantage-estimator gae_adaptive \
  --custom-config-path /data/configs/sao_gae.yaml \
  --critic-lambd 1.0 \
  --critic-lr 1e-6 \
  --critic-lr-warmup-iters 5 \
  --num-critic-only-steps 1000000 \
  --num-critic-epochs 1 \
  --value-loss-type mse \
  --gamma 1.0 \
  --kl-loss-coef 0.0 \
  --entropy-coef 0.0 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style cosine \
  --lr-decay-iters 105 \
  --min-lr 1e-7 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --grad-reduce-in-bf16 \
  --loss-scale 0.0009765625 \
  --use-precision-aware-optimizer \
  --offload-optimizer-states \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --train-env-vars "${TRAIN_ENV_VARS}" \
  --distributed-timeout-minutes 60 \
  --empty-unused-memory-level 2 \
  --colocate-critic \
  --custom-megatron-after-model-init-hook-path yeto.megatron.miles_value_island.after_model_init \
  --custom-megatron-before-train-step-hook-path yeto.megatron.miles_value_island.before_train_step \
  --custom-megatron-after-train-step-hook-path yeto.megatron.miles_value_island.after_train_step
