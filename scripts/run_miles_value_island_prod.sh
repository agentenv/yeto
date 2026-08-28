#!/usr/bin/env bash
set -euo pipefail

# One production process launches one node-local TP4 x CP2 x DP1 Miles critic
# island. Five learner islands communicate through one dedicated 8-GPU syncer.
readonly NUM_LEARNERS=5
readonly LOCAL_BUDGET_STEPS=${LOCAL_BUDGET_STEPS:-240}
readonly NUM_ROLLOUT=${LOCAL_BUDGET_STEPS}
readonly CONTEXT_LENGTH=262144
readonly REQUIRED_DATASET_VERSION=${REQUIRED_DATASET_VERSION:-qwen38-value-five-islands-contrastive-20260827-v2}
readonly REQUIRED_DATASET_STRATEGY=atomic-thread-reward-contrastive-window-balanced-v2
readonly REQUIRED_MILES_REVISION=8eb93f23d2c16d315dc574b6b9ecfd18218bfac4
readonly REQUIRED_MILES_CONTRACT_SHA256=57479fdd9992f7ef175aa358a78597c12271f952a0946aca4002277c456d51e3
readonly SAVE_INTERVAL=${SAVE_INTERVAL:-15}
readonly SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL:-999990}
readonly SYNC_INTERVAL_STEPS=${SYNC_INTERVAL_STEPS:-12}
readonly PHASE2_REJOIN=${PHASE2_REJOIN:-0}
readonly RECOVERY_LOCAL_STEP_OFFSET=${RECOVERY_LOCAL_STEP_OFFSET:-0}
readonly RECOVERY_UNIT_OFFSET=${RECOVERY_UNIT_OFFSET:-0}
readonly TRAIN_MEMORY_MARGIN_BYTES=${TRAIN_MEMORY_MARGIN_BYTES:-1073741824}
# Fresh 51-bin head over a pretrained backbone: the head weight group runs
# at VALUE_HEAD_LR_MULT x the backbone LR (scheduler-scaled max/min lr).
readonly VALUE_HEAD_LR_MULT=${VALUE_HEAD_LR_MULT:-30}
# Miles converts these iteration counts to local-context units by multiplying
# the nominal GBS of five.  The audited pack has exactly 687 train contexts per
# island, so 138 iterations gives a 690-context cosine horizon instead of
# reaching min LR before the final 162 contexts.
readonly LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-5}
readonly LR_DECAY_ITERS=${LR_DECAY_ITERS:-138}

die() {
  printf 'run_miles_value_island_prod.sh: %s\n' "$*" >&2
  exit 2
}

LEARNER_ID=${LEARNER_ID:?set LEARNER_ID to an integer in [0, 4]}
SYNCER_ADDR=${SYNCER_ADDR:?set SYNCER_ADDR to the Yeto syncer host:port}
ISLAND_DATA_TEMPLATE=${ISLAND_DATA_TEMPLATE:?set ISLAND_DATA_TEMPLATE to the node-local rollout template}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a node-local output directory}

MILES_ROOT=${MILES_ROOT:-/data/miles-values-contrastive-20260827-v2}
YETO_ROOT=${YETO_ROOT:-/data/yeto-contrastive-20260827-v2}
MODEL_DIR=${MODEL_DIR:-/data/models}
PROMPT_DATA=${PROMPT_DATA:-/data/rl_data/smoke/all3_24.jsonl}
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-/data/configs/sao_gae.yaml}
MILES_RAY_TARGET_NODE_IP=${MILES_RAY_TARGET_NODE_IP:-current}
[[ -n "${MILES_RAY_TARGET_NODE_IP}" ]] || die "MILES_RAY_TARGET_NODE_IP cannot be empty"
export MILES_RAY_TARGET_NODE_IP

[[ "${LEARNER_ID}" =~ ^[0-9]+$ ]] || die "LEARNER_ID must be an integer, got ${LEARNER_ID@Q}"
((LEARNER_ID >= 0 && LEARNER_ID < NUM_LEARNERS)) || die "LEARNER_ID must be in [0, 4], got ${LEARNER_ID}"
[[ "${LOCAL_BUDGET_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "LOCAL_BUDGET_STEPS must be a positive integer"
[[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ROLLOUT must be a positive integer"
[[ "${LR_WARMUP_ITERS}" =~ ^[0-9]+$ ]] || die "LR_WARMUP_ITERS must be a nonnegative integer"
[[ "${LR_DECAY_ITERS}" =~ ^[1-9][0-9]*$ ]] || die "LR_DECAY_ITERS must be a positive integer"
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "SAVE_INTERVAL must be a positive integer"
[[ "${SAVE_RETAIN_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "SAVE_RETAIN_INTERVAL must be a positive integer"
((SAVE_RETAIN_INTERVAL % SAVE_INTERVAL == 0)) || \
  die "SAVE_RETAIN_INTERVAL must be divisible by SAVE_INTERVAL"
[[ "${SYNC_INTERVAL_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "SYNC_INTERVAL_STEPS must be a positive integer"
[[ "${PHASE2_REJOIN}" =~ ^[01]$ ]] || die "PHASE2_REJOIN must be 0 or 1"
[[ "${TRAIN_MEMORY_MARGIN_BYTES}" =~ ^[0-9]+$ ]] || die "TRAIN_MEMORY_MARGIN_BYTES must be a nonnegative integer"
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
readonly ISLAND_DATA_DIR=${ISLAND_DATA_TEMPLATE%/*}
readonly DATASET_ROOT=${ISLAND_DATA_DIR%/*}
readonly DATASET_MANIFEST=${DATASET_MANIFEST:-${DATASET_ROOT}/manifest.json}
[[ "${ISLAND_DATA_DIR##*/}" == "island_${LEARNER_ID}" ]] || \
  die "ISLAND_DATA_TEMPLATE must point at island_${LEARNER_ID}"
[[ -r "${DATASET_MANIFEST}" ]] || die "missing dataset manifest ${DATASET_MANIFEST}"

# Offline debug replay consumes rollout IDs in numerical order and bypasses
# Miles's normal rollout shuffle.  Refuse any bundle that has not already been
# reward-stratified and H-window-balanced by the audited packer.
python3 - "${DATASET_MANIFEST}" "${REQUIRED_DATASET_VERSION}" \
  "${REQUIRED_DATASET_STRATEGY}" "${LEARNER_ID}" "${LOCAL_BUDGET_STEPS}" \
  "${SYNC_INTERVAL_STEPS}" <<'PY' || die "dataset manifest failed launch gates"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
required_version = sys.argv[2]
required_strategy = sys.argv[3]
island_id = int(sys.argv[4])
budget_steps = int(sys.argv[5])
sync_window = int(sys.argv[6])
manifest = json.loads(path.read_text(encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(manifest.get("schema_version") == 3, "dataset schema must be 3")
require(manifest.get("dataset_version") == required_version, "dataset version mismatch")
require(manifest.get("strategy") == required_strategy, "dataset strategy mismatch")
require(manifest.get("num_islands") == 5, "dataset must contain five islands")
require(manifest.get("sync_window_size") == sync_window, "dataset/sync H mismatch")
train_ids = manifest.get("train_rollout_ids")
require(isinstance(train_ids, list), "missing train rollout IDs")
require(train_ids == list(range(len(train_ids))), "train rollout IDs are not contiguous")
require(budget_steps <= len(train_ids), "learner budget exceeds stratified train data")
verification = manifest.get("verification") or {}
require(verification.get("launch_ready") is True, "dataset launch_ready is not true")
require(verification.get("duplicate_contexts") == 0, "dataset duplicates contexts")
require(verification.get("omitted_contexts") == 0, "dataset omits contexts")
require(verification.get("atomic_group_failures") == 0, "dataset splits atomic groups")
require(verification.get("step_label_failures") == 0, "dataset has a single-label optimizer step")
require(verification.get("window_label_failures") == 0, "dataset has single-label H windows")
require(
    float(verification.get("max_step_positive_rate_deviation", 1.0)) <= 0.10,
    "dataset reward-window drift exceeds 0.10",
)
recipe = manifest.get("critic_recipe") or {}
require(recipe.get("value_loss_type") == "classification", "critic must be bounded")
require(recipe.get("value_num_bins") == 51, "critic must use 51 bins")
require(recipe.get("value_reward_range") == [0.0, 1.0], "critic reward support mismatch")
require(recipe.get("value_target_type") == "hl_gauss", "critic target must be HL-Gauss")
require(float(recipe.get("hl_gauss_sigma_ratio", -1.0)) == 0.75, "HL-Gauss sigma mismatch")
require(
    recipe.get("sample_weighting") == "atomic-group-equal-within-step-v1",
    "critic sample weighting mismatch",
)
islands = manifest.get("islands") or []
island = next(
    (item for item in islands if isinstance(item, dict) and item.get("island_id") == island_id),
    None,
)
require(island is not None, "dataset manifest is missing this island")
require((island.get("train") or {}).get("contexts") == 687, "unexpected train context count")
require((island.get("train") or {}).get("all_steps_contrastive") is True, "island is not fully contrastive")
PY

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
[[ -f "${YETO_ROOT}/yeto/megatron/miles_value_island.py" ]] || die "missing Yeto Miles value-island adapter"
[[ -d "${MODEL_DIR}/Qwen3.8-27B" ]] || die "missing Hugging Face checkpoint ${MODEL_DIR}/Qwen3.8-27B"
[[ -d "${MODEL_DIR}/Qwen3.8-27B_torch_dist" ]] || die "missing distributed checkpoint ${MODEL_DIR}/Qwen3.8-27B_torch_dist"
[[ -r "${PROMPT_DATA}" ]] || die "missing prompt-data shim ${PROMPT_DATA}"
[[ -r "${CUSTOM_CONFIG_PATH}" ]] || die "missing Miles critic config ${CUSTOM_CONFIG_PATH}"

readonly CRITIC_SAVE_DIR="${OUTPUT_DIR}/critic_checkpoints"
RECOVERY_CLI_ARGS=()
if ((PHASE2_REJOIN)); then
  # This is deliberately a narrow terminal-recovery contract. It resumes a
  # full iter359 critic checkpoint, replays the remaining rollouts, joins an
  # already-persisted phase-2 syncer cut, and must never emit BUDGET_DONE.
  ((LOCAL_BUDGET_STEPS == 364)) || die "PHASE2_REJOIN requires LOCAL_BUDGET_STEPS=364"
  [[ "${RECOVERY_LOCAL_STEP_OFFSET}" =~ ^[1-9][0-9]*$ ]] || \
    die "PHASE2_REJOIN requires a positive RECOVERY_LOCAL_STEP_OFFSET"
  ((RECOVERY_LOCAL_STEP_OFFSET < LOCAL_BUDGET_STEPS)) || \
    die "RECOVERY_LOCAL_STEP_OFFSET must be below LOCAL_BUDGET_STEPS"
  [[ "${RECOVERY_UNIT_OFFSET}" =~ ^[1-9][0-9]*$ ]] || \
    die "PHASE2_REJOIN requires a positive RECOVERY_UNIT_OFFSET"
  readonly START_ROLLOUT_ID=${RECOVERY_LOCAL_STEP_OFFSET}
  readonly UNIT_OFFSET=${RECOVERY_UNIT_OFFSET}
  readonly CRITIC_LOAD_DIR=${CRITIC_SAVE_DIR}
  readonly critic_tracker="${CRITIC_LOAD_DIR}/latest_checkpointed_iteration.txt"
  [[ -r "${critic_tracker}" ]] || die "missing resumable critic tracker: ${critic_tracker}"
  checkpoint_iteration=$(<"${critic_tracker}")
  [[ "${checkpoint_iteration}" =~ ^[0-9]+$ ]] || die "invalid critic checkpoint tracker: ${critic_tracker}"
  ((checkpoint_iteration + 1 == START_ROLLOUT_ID)) || \
    die "checkpoint iteration ${checkpoint_iteration} does not resume at rollout ${START_ROLLOUT_ID}"
  # MCore's supported low-memory DCP path allocates TE FusedAdam's temporary
  # sharded-state template on CPU, then moves the loaded states back to CUDA.
  # Without it, resume briefly holds both the template and the real optimizer
  # state on GPU and OOMs even though the steady-state training layout fits.
  RECOVERY_CLI_ARGS+=(--low-memory-resume)
else
  # Fresh runs may not reinterpret an existing or partial critic checkpoint as
  # the base model while forcing rollout zero.
  readonly START_ROLLOUT_ID=0
  readonly UNIT_OFFSET=0
  readonly CRITIC_LOAD_DIR="${MODEL_DIR}/Qwen3.8-27B_torch_dist"
  if [[ -e "${CRITIC_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    die "existing critic checkpoint requires PHASE2_REJOIN=1: ${CRITIC_SAVE_DIR}"
  fi
  if [[ -d "${CRITIC_SAVE_DIR}" ]]; then
    shopt -s nullglob dotglob
    critic_artifacts=("${CRITIC_SAVE_DIR}"/*)
    shopt -u nullglob dotglob
    ((${#critic_artifacts[@]} == 0)) || die "refusing partial/non-fresh critic save directory: ${CRITIC_SAVE_DIR}"
  fi
fi

# Refuse to reserve GPUs when any training bucket is absent locally.  Held-out
# validation is a separate post-finalization job; mixing it into this process
# previously let a skipped data_0 turn the first validation bucket into an
# unintended 365th forward/backward attempt.
for ((rollout_id = START_ROLLOUT_ID; rollout_id < NUM_ROLLOUT; rollout_id++)); do
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
  PHASE2_REJOIN_VALUE="${PHASE2_REJOIN}" \
  LOCAL_STEP_OFFSET_VALUE="${START_ROLLOUT_ID}" \
  UNIT_OFFSET_VALUE="${UNIT_OFFSET}" \
  python3 - "${YETO_ROOT}" "${MILES_ROOT}" \
    "${SYNCER_ADDR}" "${LEARNER_ID}" "${NUM_LEARNERS}" \
    "${LOCAL_BUDGET_STEPS}" "${SYNC_INTERVAL_STEPS}" <<'PY'
import json
import os
import sys

(
    yeto_root,
    miles_root,
    syncer_addr,
    learner_id,
    num_learners,
    budget_steps,
    min_local_steps,
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
            "YETO_VALUE_CONNECT_TIMEOUT": "900",
            "YETO_VALUE_FINALIZATION_TIMEOUT": "900",
            "YETO_VALUE_BUDGET_STEPS": budget_steps,
            "YETO_VALUE_MIN_LOCAL_STEPS": min_local_steps,
            "YETO_VALUE_LOCAL_STEP_OFFSET": os.environ["LOCAL_STEP_OFFSET_VALUE"],
            "YETO_VALUE_UNIT_OFFSET": os.environ["UNIT_OFFSET_VALUE"],
            "YETO_VALUE_PHASE2_REJOIN": os.environ["PHASE2_REJOIN_VALUE"],
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

# Decoder-only Megatron derives encoder_seq_length from seq_length and rejects
# specifying both flags on the command line.
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
  --save "${OUTPUT_DIR}/checkpoints" \
  --critic-save "${CRITIC_SAVE_DIR}" \
  --ckpt-format torch_dist \
  --save-interval "${SAVE_INTERVAL}" \
  --save-retain-interval "${SAVE_RETAIN_INTERVAL}" \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --load-debug-rollout-data "${ISLAND_DATA_TEMPLATE}" \
  --disable-rollout-global-dataset \
  --rollout-num-gpus 0 \
  --rollout-num-gpus-per-engine 1 \
  --use-dynamic-global-batch-size \
  --start-rollout-id "${START_ROLLOUT_ID}" \
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
  --critic-lr 1e-6 \
  --critic-lr-warmup-iters "${LR_WARMUP_ITERS}" \
  --num-critic-only-steps 1000000 \
  --num-critic-epochs 1 \
  --value-loss-type classification \
  --value-num-bins 51 \
  --value-reward-range 0.0 1.0 \
  --value-target-type hl_gauss \
  --hl-gauss-sigma-ratio 0.75 \
  --value-head-lr-mult "${VALUE_HEAD_LR_MULT}" \
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
  --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}" \
  "${RECOVERY_CLI_ARGS[@]}" \
  --colocate-critic \
  --custom-megatron-after-model-init-hook-path yeto.megatron.miles_value_island.after_model_init \
  --custom-megatron-before-train-step-hook-path yeto.megatron.miles_value_island.before_train_step \
  --custom-megatron-after-train-step-hook-path yeto.megatron.miles_value_island.after_train_step \
  "${WANDB_ARGS[@]}"
