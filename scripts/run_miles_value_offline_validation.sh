#!/usr/bin/env bash
set -euo pipefail

# Offline held-out validation for the contrastive five-island Qwen3.8 critic
# (canary gate #6). Replays validation buckets 240..263 through the normal
# Miles critic forward/backward path with optimizer construction disabled, so
# every bucket is computed against the unchanged step-48 checkpoint. No Yeto
# syncer hooks, no W&B, no checkpoint writes, and no optimizer/RNG load.
# Per-bucket critic-step sufficient statistics
# (train/critic-value_{loss,ev_n,returns_sum,returns_sq_sum,residual_sum,
# residual_sq_sum}) land in the log for exact pooled aggregation with
# scripts/tools/aggregate_offline_validation_ev.py --validation-start-rollout
# 240 --num-rollout 264 over all five island logs.

readonly CONTEXT_LENGTH=262144
readonly REQUIRED_MILES_REVISION=6438fe22d5915c5b60aa81686e854b66ebe6506c
readonly REQUIRED_MILES_CONTRACT_SHA256=d6dad9bb9d41908da0f7d05a09317574892beff8744c0ef8510ffe0970dcfe2c
readonly REQUIRED_DATASET_VERSION=qwen38-value-five-islands-contrastive-20260827-v2
readonly REQUIRED_DATASET_STRATEGY=atomic-thread-reward-contrastive-window-balanced-v2
readonly VALIDATION_START=240
readonly NUM_ROLLOUT=264
readonly EXPECTED_CHECKPOINT_ITER=47
readonly EXPECTED_CHECKPOINT_DIR=iter_0000047
readonly LOCAL_GLOBAL_BATCH_SIZE=5

die() {
  printf 'run_miles_value_offline_validation.sh: %s\n' "$*" >&2
  exit 2
}

ISLAND_ID=${ISLAND_ID:?set ISLAND_ID to the island id in 0..4}
CRITIC_LOAD_DIR=${CRITIC_LOAD_DIR:?set CRITIC_LOAD_DIR to the critic_checkpoints root holding iter_0000047}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a fresh node-local output directory}

MILES_ROOT=${MILES_ROOT:-/data/miles-values-contrastive-20260827-v2}
YETO_ROOT=${YETO_ROOT:-/data/yeto-contrastive-20260827-v2}
MODEL_DIR=${MODEL_DIR:-/data/models}
PROMPT_DATA=${PROMPT_DATA:-/data/rl_data/smoke/all3_24.jsonl}
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-/data/configs/sao_gae.yaml}
MILES_RAY_TARGET_NODE_IP=${MILES_RAY_TARGET_NODE_IP:-current}
[[ -n "${MILES_RAY_TARGET_NODE_IP}" ]] || die "MILES_RAY_TARGET_NODE_IP cannot be empty"
export MILES_RAY_TARGET_NODE_IP
# Do not put the {rollout_id} placeholder inside a ${var:-default} expansion:
# bash treats its closing brace as the end of the parameter expansion.
ISLAND_DATA_TEMPLATE=${ISLAND_DATA_TEMPLATE:-}
if [[ -z "${ISLAND_DATA_TEMPLATE}" ]]; then
  ISLAND_DATA_TEMPLATE="/data/local-runs/qwen38-value-five-islands-contrastive-20260827-v2/island_${ISLAND_ID}/data_{rollout_id}.pt"
fi

[[ "${ISLAND_ID}" =~ ^[0-4]$ ]] || die "ISLAND_ID must be in [0, 4]"
[[ "${OUTPUT_DIR}" == /* && "${OUTPUT_DIR}" != / ]] || die "OUTPUT_DIR must be an absolute directory other than /"
[[ -e "${OUTPUT_DIR}" ]] && die "refusing existing OUTPUT_DIR: ${OUTPUT_DIR}"

readonly ROLLOUT_PLACEHOLDER='{rollout_id}'
[[ "${ISLAND_DATA_TEMPLATE}" == /* ]] || die "ISLAND_DATA_TEMPLATE must be an absolute node-local path"
[[ "${ISLAND_DATA_TEMPLATE}" == *"${ROLLOUT_PLACEHOLDER}"* ]] || die "ISLAND_DATA_TEMPLATE must contain ${ROLLOUT_PLACEHOLDER}"
template_without_first_placeholder=${ISLAND_DATA_TEMPLATE/"${ROLLOUT_PLACEHOLDER}"/}
[[ "${template_without_first_placeholder}" != *"${ROLLOUT_PLACEHOLDER}"* ]] || die "ISLAND_DATA_TEMPLATE must contain exactly one ${ROLLOUT_PLACEHOLDER}"
readonly ISLAND_DATA_DIR=${ISLAND_DATA_TEMPLATE%/*}
readonly DATASET_ROOT=${ISLAND_DATA_DIR%/*}
readonly DATASET_MANIFEST=${DATASET_MANIFEST:-${DATASET_ROOT}/manifest.json}
[[ "${ISLAND_DATA_DIR##*/}" == "island_${ISLAND_ID}" ]] || \
  die "ISLAND_DATA_TEMPLATE must point at island_${ISLAND_ID}"
[[ -r "${DATASET_MANIFEST}" ]] || die "missing dataset manifest ${DATASET_MANIFEST}"

# Fail closed unless the manifest proves this is the audited contrastive pack
# with a contiguous held-out range exactly matching the replay window below.
python3 - "${DATASET_MANIFEST}" "${REQUIRED_DATASET_VERSION}" \
  "${REQUIRED_DATASET_STRATEGY}" "${ISLAND_ID}" \
  "${VALIDATION_START}" "${NUM_ROLLOUT}" <<'PY' || die "dataset manifest failed validation gates"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
required_version = sys.argv[2]
required_strategy = sys.argv[3]
island_id = int(sys.argv[4])
validation_start = int(sys.argv[5])
num_rollout = int(sys.argv[6])
manifest = json.loads(path.read_text(encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(manifest.get("schema_version") == 3, "dataset schema must be 3")
require(manifest.get("dataset_version") == required_version, "dataset version mismatch")
require(manifest.get("strategy") == required_strategy, "dataset strategy mismatch")
require(manifest.get("num_islands") == 5, "dataset must contain five islands")
train_ids = manifest.get("train_rollout_ids")
require(isinstance(train_ids, list), "missing train rollout IDs")
require(train_ids == list(range(validation_start)), "train rollout IDs are not 0..validation_start")
validation_ids = manifest.get("validation_rollout_ids")
require(isinstance(validation_ids, list), "missing validation rollout IDs")
require(
    validation_ids == list(range(validation_start, num_rollout)),
    "validation rollout IDs do not exactly cover the requested range",
)
recipe = manifest.get("critic_recipe") or {}
require(recipe.get("value_loss_type") == "classification", "critic must be bounded")
require(recipe.get("value_num_bins") == 51, "critic must use 51 bins")
require(recipe.get("value_reward_range") == [0.0, 1.0], "critic reward support mismatch")
require(recipe.get("value_target_type") == "hl_gauss", "critic target must be HL-Gauss")
require(float(recipe.get("hl_gauss_sigma_ratio", -1.0)) == 0.75, "HL-Gauss sigma mismatch")
islands = manifest.get("islands") or []
island = next(
    (item for item in islands if isinstance(item, dict) and item.get("island_id") == island_id),
    None,
)
require(island is not None, "dataset manifest is missing this island")
require(
    (island.get("validation") or {}).get("rollouts") == num_rollout - validation_start,
    "island held-out bucket count mismatch",
)
PY

tracker="${CRITIC_LOAD_DIR}/latest_checkpointed_iteration.txt"
[[ -r "${tracker}" ]] || die "missing critic tracker: ${tracker}"
[[ "$(<"${tracker}")" == "${EXPECTED_CHECKPOINT_ITER}" ]] || \
  die "tracker $(<"${tracker}") != ${EXPECTED_CHECKPOINT_ITER}: ${tracker}"
[[ -f "${CRITIC_LOAD_DIR}/${EXPECTED_CHECKPOINT_DIR}/.metadata" ]] || \
  die "missing ${EXPECTED_CHECKPOINT_DIR}/.metadata"
PYTHONPATH="${MILES_ROOT}" python3 - "${CRITIC_LOAD_DIR}" <<'PY' || \
  die "critic checkpoint is not a complete trained model-only artifact"
import sys

from miles.utils.critic_checkpoint import is_trained_model_only_critic

if not is_trained_model_only_critic(sys.argv[1]):
    raise RuntimeError("trained model-only checkpoint marker is absent")
PY

for ((rollout_id = VALIDATION_START; rollout_id < NUM_ROLLOUT; rollout_id++)); do
  rollout_path=${ISLAND_DATA_TEMPLATE/"${ROLLOUT_PLACEHOLDER}"/${rollout_id}}
  [[ -r "${rollout_path}" ]] || die "missing validation bucket ${rollout_id}: ${rollout_path}"
done

[[ -f "${MILES_ROOT}/train_async.py" ]] || die "missing ${MILES_ROOT}/train_async.py"
actual_miles_revision=$(git -C "${MILES_ROOT}" rev-parse HEAD 2>/dev/null) || \
  die "MILES_ROOT must be a Git checkout pinned to ${REQUIRED_MILES_REVISION}"
[[ "${actual_miles_revision}" == "${REQUIRED_MILES_REVISION}" ]] || \
  die "MILES_ROOT revision ${actual_miles_revision} != required ${REQUIRED_MILES_REVISION}"
actual_miles_contract=$(
  python3 - "${MILES_ROOT}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = (
    "train_async.py",
    "miles/backends/megatron_utils/actor.py",
    "miles/utils/critic_checkpoint.py",
    "miles/ray/rollout/train_data_conversion.py",
    "miles/backends/training_utils/cp_utils.py",
    "miles/backends/training_utils/loss.py",
    "miles/backends/megatron_utils/model.py",
    "miles/backends/training_utils/loss_hub/losses.py",
    "scripts/tools/aggregate_offline_validation_ev.py",
)
digest = hashlib.sha256()
for name in files:
    digest.update(name.encode() + b"\0" + (root / name).read_bytes())
print(digest.hexdigest())
PY
) || die "failed to hash the Miles value-training contract"
[[ "${actual_miles_contract}" == "${REQUIRED_MILES_CONTRACT_SHA256}" ]] || \
  die "Miles value-training contract hash mismatch: ${actual_miles_contract}"
[[ -f "${MILES_ROOT}/scripts/models/qwen3.6-27B.sh" ]] || die "missing Miles Qwen model argument script"
[[ -f "${YETO_ROOT}/yeto_value_validation_hook.py" ]] || die "missing validation hook module"
[[ -f "${YETO_ROOT}/yeto/megatron/miles_value_island.py" ]] || die "missing Yeto Miles value-island adapter"
[[ -d "${MODEL_DIR}/Qwen3.8-27B" ]] || die "missing Hugging Face checkpoint ${MODEL_DIR}/Qwen3.8-27B"
[[ -d "${MODEL_DIR}/Qwen3.8-27B_torch_dist" ]] || die "missing distributed checkpoint ${MODEL_DIR}/Qwen3.8-27B_torch_dist"
[[ -r "${PROMPT_DATA}" ]] || die "missing prompt-data shim ${PROMPT_DATA}"
[[ -r "${CUSTOM_CONFIG_PATH}" ]] || die "missing Miles critic config ${CUSTOM_CONFIG_PATH}"

# The eval island needs all eight local GPUs. Refuse to reserve them while any
# compute process (for example a still-running canary learner) holds the cards.
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for the GPU occupancy gate"
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || \
  die "nvidia-smi compute-apps query failed"
[[ -z "${gpu_pids}" ]] || \
  die "refusing to launch while GPU compute processes are present: $(printf '%s' "${gpu_pids}" | tr '\n' ' ')"

cd "${MILES_ROOT}"
# shellcheck source=/dev/null
source scripts/models/qwen3.6-27B.sh
declare -p MODEL_ARGS >/dev/null 2>&1 || die "Miles model script did not define MODEL_ARGS"
mkdir -p "${OUTPUT_DIR}"

TRAIN_ENV_VARS=$(printf '{"PYTHONPATH":"%s:%s","PYTHONFAULTHANDLER":"1","TORCH_SHOW_CPP_STACKTRACES":"1","TORCH_DISABLE_ADDR2LINE":"1","NCCL_DEBUG":"WARN","PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512,garbage_collection_threshold:0.8"}' "${YETO_ROOT}" "${MILES_ROOT}")

export PYTHONPATH="${YETO_ROOT}:${MILES_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

printf 'Launching offline validation island=%d rollouts=[%d,%d) load=%s output=%s\n' \
  "${ISLAND_ID}" "${VALIDATION_START}" "${NUM_ROLLOUT}" \
  "${CRITIC_LOAD_DIR}" "${OUTPUT_DIR}"

exec python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node 8 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}/Qwen3.8-27B" \
  --load "${MODEL_DIR}/Qwen3.8-27B_torch_dist" \
  --critic-load "${CRITIC_LOAD_DIR}" \
  --ref-load "${MODEL_DIR}/Qwen3.8-27B_torch_dist" \
  --ckpt-format torch_dist \
  --debug-disable-optimizer \
  --no-load-optim \
  --no-load-rng \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --load-debug-rollout-data "${ISLAND_DATA_TEMPLATE}" \
  --disable-rollout-global-dataset \
  --rollout-num-gpus 0 \
  --rollout-num-gpus-per-engine 1 \
  --use-dynamic-global-batch-size \
  --start-rollout-id "${VALIDATION_START}" \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${LOCAL_GLOBAL_BATCH_SIZE}" \
  --global-batch-size "${LOCAL_GLOBAL_BATCH_SIZE}" \
  --micro-batch-size 1 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len "${CONTEXT_LENGTH}" \
  --seq-length "${CONTEXT_LENGTH}" \
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
  --critic-lr 1e-30 \
  --num-critic-only-steps 1000000 \
  --num-critic-epochs 1 \
  --value-loss-type classification \
  --value-num-bins 51 \
  --value-reward-range 0.0 1.0 \
  --value-target-type hl_gauss \
  --hl-gauss-sigma-ratio 0.75 \
  --gamma 1.0 \
  --kl-loss-coef 0.0 \
  --entropy-coef 0.0 \
  --optimizer adam \
  --lr 1e-30 \
  --lr-decay-style cosine \
  --lr-decay-iters 264 \
  --min-lr 0.0 \
  --weight-decay 0.0 \
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
  --train-memory-margin-bytes 1073741824 \
  --custom-megatron-after-model-init-hook-path yeto_value_validation_hook.after_model_init
