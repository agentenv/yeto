#!/usr/bin/env bash
set -euo pipefail

# One production process launches one node-local TP4 x CP2 x DP1 Miles critic
# island. Five learner islands communicate through one dedicated 8-GPU syncer.
readonly NUM_LEARNERS=5
readonly NUM_ROLLOUT=364
readonly LOCAL_BUDGET_STEPS=364
readonly CONTEXT_LENGTH=262144
readonly REQUIRED_MILES_REVISION=277d151c00ef4f6727f01aca06e115b71bd7578c
readonly SAVE_INTERVAL=${SAVE_INTERVAL:-15}
# Miles converts these iteration counts to sample counts by multiplying the
# nominal GBS. Five islands use fixed nominal size 5 each (total 25), so five
# warmup iterations remain 125 global samples and 105 decay iterations remain
# the original 2,625-sample global cosine horizon.
readonly LR_WARMUP_ITERS=5
readonly LR_DECAY_ITERS=105

die() {
  printf 'run_miles_value_island_prod.sh: %s\n' "$*" >&2
  exit 2
}

LEARNER_ID=${LEARNER_ID:?set LEARNER_ID to an integer in [0, 4]}
SYNCER_ADDR=${SYNCER_ADDR:?set SYNCER_ADDR to the Yeto syncer host:port}
ISLAND_DATA_TEMPLATE=${ISLAND_DATA_TEMPLATE:?set ISLAND_DATA_TEMPLATE to the node-local rollout template}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a node-local output directory}

MILES_ROOT=${MILES_ROOT:-/data/miles-values-isoloco-20260826-v7}
YETO_ROOT=${YETO_ROOT:-/data/yeto-isoloco-20260826-v7}
MODEL_DIR=${MODEL_DIR:-/data/models}
PROMPT_DATA=${PROMPT_DATA:-/data/rl_data/smoke/all3_24.jsonl}
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-/data/configs/sao_gae.yaml}
MILES_RAY_TARGET_NODE_IP=${MILES_RAY_TARGET_NODE_IP:-current}
[[ -n "${MILES_RAY_TARGET_NODE_IP}" ]] || die "MILES_RAY_TARGET_NODE_IP cannot be empty"
export MILES_RAY_TARGET_NODE_IP

[[ "${LEARNER_ID}" =~ ^[0-9]+$ ]] || die "LEARNER_ID must be an integer, got ${LEARNER_ID@Q}"
((LEARNER_ID >= 0 && LEARNER_ID < NUM_LEARNERS)) || die "LEARNER_ID must be in [0, 4], got ${LEARNER_ID}"
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "SAVE_INTERVAL must be a positive integer"
readonly LOCAL_GLOBAL_BATCH_SIZE=5

syncer_host=${SYNCER_ADDR%:*}
syncer_port=${SYNCER_ADDR##*:}
[[ -n "${syncer_host}" && "${syncer_host}" != "${SYNCER_ADDR}" ]] || die "SYNCER_ADDR must be host:port"
[[ "${syncer_port}" =~ ^[0-9]+$ ]] || die "SYNCER_ADDR port must be an integer"
((syncer_port >= 1 && syncer_port <= 65535)) || die "SYNCER_ADDR port must be in [1, 65535]"

readonly ROLLOUT_PLACEHOLDER='{rollout_id}'
[[ "${ISLAND_DATA_TEMPLATE}" == /* ]] || die "ISLAND_DATA_TEMPLATE must be an absolute node-local path"
[[ "${ISLAND_DATA_TEMPLATE}" == *"${ROLLOUT_PLACEHOLDER}"* ]] || die "ISLAND_DATA_TEMPLATE must contain ${ROLLOUT_PLACEHOLDER}"
template_without_first_placeholder=${ISLAND_DATA_TEMPLATE/"${ROLLOUT_PLACEHOLDER}"/}
[[ "${template_without_first_placeholder}" != *"${ROLLOUT_PLACEHOLDER}"* ]] || die "ISLAND_DATA_TEMPLATE must contain exactly one ${ROLLOUT_PLACEHOLDER}"
[[ "${OUTPUT_DIR}" == /* && "${OUTPUT_DIR}" != / ]] || die "OUTPUT_DIR must be an absolute directory other than /"

[[ -f "${MILES_ROOT}/train_async.py" ]] || die "missing ${MILES_ROOT}/train_async.py"
actual_miles_revision=$(git -C "${MILES_ROOT}" rev-parse HEAD 2>/dev/null) || \
  die "MILES_ROOT must be a Git checkout pinned to ${REQUIRED_MILES_REVISION}"
[[ "${actual_miles_revision}" == "${REQUIRED_MILES_REVISION}" ]] || \
  die "MILES_ROOT revision ${actual_miles_revision} != required ${REQUIRED_MILES_REVISION}"
[[ -f "${MILES_ROOT}/scripts/models/qwen3.6-27B.sh" ]] || die "missing Miles Qwen model argument script"
[[ -f "${YETO_ROOT}/yeto/megatron/miles_value_island.py" ]] || die "missing Yeto Miles value-island adapter"
[[ -d "${MODEL_DIR}/Qwen3.8-27B" ]] || die "missing Hugging Face checkpoint ${MODEL_DIR}/Qwen3.8-27B"
[[ -d "${MODEL_DIR}/Qwen3.8-27B_torch_dist" ]] || die "missing distributed checkpoint ${MODEL_DIR}/Qwen3.8-27B_torch_dist"
[[ -r "${PROMPT_DATA}" ]] || die "missing prompt-data shim ${PROMPT_DATA}"
[[ -r "${CUSTOM_CONFIG_PATH}" ]] || die "missing Miles critic config ${CUSTOM_CONFIG_PATH}"

# This launcher is the fresh-run contract.  Never reinterpret an existing or
# partial critic checkpoint as the base model while forcing rollout zero.
readonly CRITIC_SAVE_DIR="${OUTPUT_DIR}/critic_checkpoints"
if [[ -e "${CRITIC_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  die "existing critic checkpoint requires the separate resume contract: ${CRITIC_SAVE_DIR}"
fi
if [[ -d "${CRITIC_SAVE_DIR}" ]]; then
  shopt -s nullglob dotglob
  critic_artifacts=("${CRITIC_SAVE_DIR}"/*)
  shopt -u nullglob dotglob
  ((${#critic_artifacts[@]} == 0)) || die "refusing partial/non-fresh critic save directory: ${CRITIC_SAVE_DIR}"
fi

# Refuse to reserve GPUs when any training bucket is absent locally.  Held-out
# validation is a separate post-finalization job; mixing it into this process
# previously let a skipped data_0 turn the first validation bucket into an
# unintended 365th forward/backward attempt.
for ((rollout_id = 0; rollout_id < NUM_ROLLOUT; rollout_id++)); do
  rollout_path=${ISLAND_DATA_TEMPLATE/"${ROLLOUT_PLACEHOLDER}"/${rollout_id}}
  [[ -r "${rollout_path}" ]] || die "missing island rollout ${rollout_id}: ${rollout_path}"
done

cd "${MILES_ROOT}"
# shellcheck source=/dev/null
source scripts/models/qwen3.6-27B.sh
declare -p MODEL_ARGS >/dev/null 2>&1 || die "Miles model script did not define MODEL_ARGS"
mkdir -p "${OUTPUT_DIR}"

# Serialize only non-secret worker configuration.  W&B credentials remain in
# the existing process/runtime environment and are never copied into argv or
# --train-env-vars.
TRAIN_ENV_VARS=$(
  python3 - "${YETO_ROOT}" "${MILES_ROOT}" \
    "${SYNCER_ADDR}" "${LEARNER_ID}" "${NUM_LEARNERS}" \
    "${LOCAL_BUDGET_STEPS}" <<'PY'
import json
import sys

(
    yeto_root,
    miles_root,
    syncer_addr,
    learner_id,
    num_learners,
    budget_steps,
) = sys.argv[1:]

print(
    json.dumps(
        {
            "PYTHONPATH": f"{yeto_root}:{miles_root}",
            "PYTHONFAULTHANDLER": "1",
            "TORCH_SHOW_CPP_STACKTRACES": "1",
            "TORCH_DISABLE_ADDR2LINE": "1",
            "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.8",
            "YETO_VALUE_SYNCER": syncer_addr,
            "YETO_VALUE_LEARNER_ID": learner_id,
            "YETO_VALUE_NUM_LEARNERS": num_learners,
            "YETO_VALUE_MERGE_ALPHA": "0.5",
            "YETO_VALUE_STREAMS": "4",
            "YETO_VALUE_NUM_FRAGMENTS": "96",
            "YETO_VALUE_FRAGMENT_PATTERN": "binpack",
            "YETO_VALUE_CONNECT_TIMEOUT": "3600",
            "YETO_VALUE_FINALIZATION_TIMEOUT": "3600",
            "YETO_VALUE_BUDGET_STEPS": budget_steps,
            "YETO_VALUE_LOCAL_STEP_OFFSET": "0",
            "YETO_VALUE_UNIT_OFFSET": "0",
        },
        separators=(",", ":"),
    )
)
PY
) || die "failed to construct Miles worker environment"

export PYTHONPATH="${YETO_ROOT}:${MILES_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Supplying both standard W&B naming variables enables logging.  Authentication
# is deliberately left to WANDB_API_KEY/netrc already present in the runtime.
WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" || -n "${WANDB_GROUP:-}" ]]; then
  [[ -n "${WANDB_PROJECT:-}" && -n "${WANDB_GROUP:-}" ]] || die "set both WANDB_PROJECT and WANDB_GROUP, or neither"
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
  )

  wandb_team=${WANDB_TEAM:-${WANDB_ENTITY:-}}
  if [[ -n "${WANDB_TEAM:-}" && -n "${WANDB_ENTITY:-}" && "${WANDB_TEAM}" != "${WANDB_ENTITY}" ]]; then
    die "WANDB_TEAM and WANDB_ENTITY disagree"
  fi
  [[ -z "${wandb_team}" ]] || WANDB_ARGS+=(--wandb-team "${wandb_team}")

  if [[ -n "${WANDB_MODE:-}" ]]; then
    case "${WANDB_MODE}" in
      online | offline | disabled) ;;
      *) die "WANDB_MODE must be online, offline, or disabled" ;;
    esac
    WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
  fi
  [[ -z "${WANDB_DIR:-}" ]] || WANDB_ARGS+=(--wandb-dir "${WANDB_DIR}")
fi

printf 'Launching Miles value island %d/%d: train=%d context=%d data=%s output=%s syncer=%s ray_node=%s\n' \
  "${LEARNER_ID}" "${NUM_LEARNERS}" "${LOCAL_BUDGET_STEPS}" \
  "${CONTEXT_LENGTH}" \
  "${ISLAND_DATA_TEMPLATE}" "${OUTPUT_DIR}" "${SYNCER_ADDR}" \
  "${MILES_RAY_TARGET_NODE_IP}"

exec python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node 8 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}/Qwen3.8-27B" \
  --load "${MODEL_DIR}/Qwen3.8-27B_torch_dist" \
  --critic-load "${MODEL_DIR}/Qwen3.8-27B_torch_dist" \
  --ref-load "${MODEL_DIR}/Qwen3.8-27B_torch_dist" \
  --save "${OUTPUT_DIR}/checkpoints" \
  --critic-save "${CRITIC_SAVE_DIR}" \
  --ckpt-format torch_dist \
  --save-interval "${SAVE_INTERVAL}" \
  --save-retain-interval 1000000 \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --load-debug-rollout-data "${ISLAND_DATA_TEMPLATE}" \
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
  --custom-config-path "${CUSTOM_CONFIG_PATH}" \
  --critic-lambd 1.0 \
  --critic-lr 1e-6 \
  --critic-lr-warmup-iters "${LR_WARMUP_ITERS}" \
  --num-critic-only-steps 1000000 \
  --num-critic-epochs 1 \
  --value-loss-type mse \
  --gamma 1.0 \
  --kl-loss-coef 0.0 \
  --entropy-coef 0.0 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style cosine \
  --lr-decay-iters "${LR_DECAY_ITERS}" \
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
  --custom-megatron-after-train-step-hook-path yeto.megatron.miles_value_island.after_train_step \
  "${WANDB_ARGS[@]}"
