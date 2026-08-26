#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'run_miles_value_island_smoke.sh: %s\n' "$*" >&2
  exit 2
}

readonly CONTEXT_LENGTH=262144
readonly REQUIRED_MILES_REVISION=277d151c00ef4f6727f01aca06e115b71bd7578c
MILES_ROOT=${MILES_ROOT:-/data/miles-values-isoloco-20260826-v7}
YETO_ROOT=${YETO_ROOT:-/data/yeto-isoloco-20260826-v7}
SYNCER_ADDR=${SYNCER_ADDR:?SYNCER_ADDR must be host:port}
OUTPUT_DIR=${OUTPUT_DIR:-/data/local-runs/qwen38-value-island-smoke-20260826-v7}
readonly SMOKE_BUDGET_STEPS=${SMOKE_BUDGET_STEPS:-2}
readonly NUM_ROLLOUT=${SMOKE_BUDGET_STEPS}
# Do not put the ``{rollout_id}`` placeholder inside a ``${var:-default}``
# expansion: bash treats its closing brace as the end of the parameter
# expansion and appends the remainder (for example ``.pt}``) to overrides.
DATA_TEMPLATE=${DATA_TEMPLATE:-}
if [[ -z "${DATA_TEMPLATE}" ]]; then
  DATA_TEMPLATE='/data/local-runs/qwen38-value-six-islands-fresh-v7/island_0/data_{rollout_id}.pt'
fi
readonly LOCAL_GLOBAL_BATCH_SIZE=5

[[ "${SMOKE_BUDGET_STEPS}" =~ ^[2-4]$ ]] || \
  die "SMOKE_BUDGET_STEPS must be in [2, 4]"
[[ "${DATA_TEMPLATE}" == /* && "${DATA_TEMPLATE}" == *'{rollout_id}'* ]] || \
  die "DATA_TEMPLATE must be absolute and contain {rollout_id}"
template_without_first_placeholder=${DATA_TEMPLATE/'{rollout_id}'/}
[[ "${template_without_first_placeholder}" != *'{rollout_id}'* ]] || \
  die "DATA_TEMPLATE must contain exactly one {rollout_id}"
[[ "${OUTPUT_DIR}" == /* && "${OUTPUT_DIR}" != / ]] || \
  die "OUTPUT_DIR must be an absolute directory other than /"
[[ -f "${MILES_ROOT}/train_async.py" ]] || die "missing ${MILES_ROOT}/train_async.py"
actual_miles_revision=$(git -C "${MILES_ROOT}" rev-parse HEAD 2>/dev/null) || \
  die "MILES_ROOT must be a Git checkout pinned to ${REQUIRED_MILES_REVISION}"
[[ "${actual_miles_revision}" == "${REQUIRED_MILES_REVISION}" ]] || \
  die "MILES_ROOT revision ${actual_miles_revision} != required ${REQUIRED_MILES_REVISION}"
[[ -f "${YETO_ROOT}/yeto/megatron/miles_value_island.py" ]] || \
  die "missing ${YETO_ROOT}/yeto/megatron/miles_value_island.py"
for ((rollout_id = 0; rollout_id < NUM_ROLLOUT; rollout_id++)); do
  rollout_path=${DATA_TEMPLATE/'{rollout_id}'/${rollout_id}}
  [[ -r "${rollout_path}" ]] || die "missing smoke rollout ${rollout_id}: ${rollout_path}"
done

readonly CRITIC_SAVE_DIR="${OUTPUT_DIR}/critic_checkpoints"
if [[ -e "${CRITIC_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  die "smoke requires a fresh critic directory: ${CRITIC_SAVE_DIR}"
fi
if [[ -d "${CRITIC_SAVE_DIR}" ]]; then
  shopt -s nullglob dotglob
  critic_artifacts=("${CRITIC_SAVE_DIR}"/*)
  shopt -u nullglob dotglob
  ((${#critic_artifacts[@]} == 0)) || \
    die "refusing non-fresh critic directory: ${CRITIC_SAVE_DIR}"
fi

cd "${MILES_ROOT}"
# shellcheck source=/dev/null
source scripts/models/qwen3.6-27B.sh
mkdir -p "${OUTPUT_DIR}"

TRAIN_ENV_VARS=$(
  python3 - "${YETO_ROOT}" "${MILES_ROOT}" "${NUM_ROLLOUT}" "${SYNCER_ADDR}" <<'PY'
import json
import sys

yeto_root, miles_root, budget_steps, syncer_addr = sys.argv[1:]
print(
    json.dumps(
        {
            "PYTHONPATH": f"{yeto_root}:{miles_root}",
            "PYTHONFAULTHANDLER": "1",
            "TORCH_SHOW_CPP_STACKTRACES": "1",
            "TORCH_DISABLE_ADDR2LINE": "1",
            "MILES_OFFLINE_VALIDATION_START_ROLLOUT": budget_steps,
            "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.8",
            "YETO_VALUE_SYNCER": syncer_addr,
            "YETO_VALUE_LEARNER_ID": "0",
            "YETO_VALUE_NUM_LEARNERS": "1",
            "YETO_VALUE_MERGE_ALPHA": "0.5",
            "YETO_VALUE_STREAMS": "4",
            "YETO_VALUE_NUM_FRAGMENTS": "96",
            "YETO_VALUE_FRAGMENT_PATTERN": "binpack",
            "YETO_VALUE_CONNECT_TIMEOUT": "900",
            "YETO_VALUE_FINALIZATION_TIMEOUT": "3600",
            "YETO_VALUE_BUDGET_STEPS": budget_steps,
            "YETO_VALUE_LOCAL_STEP_OFFSET": "0",
            "YETO_VALUE_UNIT_OFFSET": "0",
        },
        separators=(",", ":"),
    )
)
PY
) || die "failed to construct smoke worker environment"

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
  --critic-save "${CRITIC_SAVE_DIR}" \
  --ckpt-format torch_dist \
  --save-interval 1000 \
  --save-retain-interval 1000000 \
  --prompt-data /data/rl_data/smoke/all3_24.jsonl \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --load-debug-rollout-data "${DATA_TEMPLATE}" \
  --disable-rollout-global-dataset \
  --rollout-num-gpus 0 \
  --rollout-num-gpus-per-engine 1 \
  --use-dynamic-global-batch-size \
  --start-rollout-id 0 \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${LOCAL_GLOBAL_BATCH_SIZE}" \
  --global-batch-size "${LOCAL_GLOBAL_BATCH_SIZE}" \
  --micro-batch-size 1 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len "${CONTEXT_LENGTH}" \
  --seq-length "${CONTEXT_LENGTH}" \
  --encoder-seq-length "${CONTEXT_LENGTH}" \
  --max-position-embeddings "${CONTEXT_LENGTH}" \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 2 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --sequence-parallel \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
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
