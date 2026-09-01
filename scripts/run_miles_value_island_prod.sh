#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'run_miles_value_island_prod.sh: %s\n' "$*" >&2
  exit 2
}

# One process launches one node-local TP4 x CP{1,2} x DP1 Miles critic island.
# Production remains a five-learner full-quorum topology. The explicit
# diagnostic override selects audited source islands 0 and 1. The eight-GPU
# form runs 48 local steps per island; the equal-total-update TP4/CP1 form runs
# 24 so neither can be mistaken for production evidence.
readonly TWO_LEARNER_DIAGNOSTIC=${YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC:-0}
readonly SINGLE_LEARNER_DIAGNOSTIC=${YETO_MILES_VALUE_SINGLE_LEARNER_DIAGNOSTIC:-0}
readonly TP4_CP1_DIAGNOSTIC=${YETO_MILES_VALUE_TP4_CP1:-0}
readonly TP4_CP1_CANARY=${YETO_MILES_VALUE_TP4_CP1_CANARY:-0}
readonly MODEL_ONLY_CHECKPOINT=${YETO_MILES_VALUE_MODEL_ONLY_CHECKPOINT:-0}
readonly CHECKPOINT_OWNER=${YETO_MILES_VALUE_CHECKPOINT_OWNER:-0}
readonly ANCHOR_SPILL=${YETO_MILES_VALUE_ANCHOR_SPILL:-0}
readonly HDO_GRADIENT_STREAM=${YETO_MILES_VALUE_HDO_GRADIENT_STREAM:-0}
[[ "${TWO_LEARNER_DIAGNOSTIC}" =~ ^[01]$ ]] || \
  die "YETO_MILES_VALUE_TWO_LEARNER_DIAGNOSTIC must be 0 or 1"
[[ "${SINGLE_LEARNER_DIAGNOSTIC}" =~ ^[01]$ ]] || \
  die "YETO_MILES_VALUE_SINGLE_LEARNER_DIAGNOSTIC must be 0 or 1"
[[ "${TP4_CP1_DIAGNOSTIC}" =~ ^[01]$ ]] || \
  die "YETO_MILES_VALUE_TP4_CP1 must be 0 or 1"
[[ "${TP4_CP1_CANARY}" =~ ^[01]$ ]] || \
  die "YETO_MILES_VALUE_TP4_CP1_CANARY must be 0 or 1"
[[ "${MODEL_ONLY_CHECKPOINT}" =~ ^[01]$ ]] || \
  die "YETO_MILES_VALUE_MODEL_ONLY_CHECKPOINT must be 0 or 1"
[[ "${ANCHOR_SPILL}" =~ ^[01]$ ]] || \
  die "YETO_MILES_VALUE_ANCHOR_SPILL must be 0 or 1"
[[ "${HDO_GRADIENT_STREAM}" =~ ^[01]$ ]] || \
  die "YETO_MILES_VALUE_HDO_GRADIENT_STREAM must be 0 or 1"
[[ "${CHECKPOINT_OWNER}" =~ ^[01]$ ]] || \
  die "YETO_MILES_VALUE_CHECKPOINT_OWNER must be learner 0 or 1"
((SINGLE_LEARNER_DIAGNOSTIC + TWO_LEARNER_DIAGNOSTIC + TP4_CP1_CANARY <= 1)) || \
  die "single-learner, two-learner, and TP4/CP1 canary modes are mutually exclusive"
((TP4_CP1_DIAGNOSTIC == 0 || SINGLE_LEARNER_DIAGNOSTIC == 1 || \
  TWO_LEARNER_DIAGNOSTIC == 1 || TP4_CP1_CANARY == 1)) || \
  die "YETO_MILES_VALUE_TP4_CP1 requires a diagnostic or canary mode"
readonly TP4_CP1_ENABLED=$((TP4_CP1_DIAGNOSTIC || TP4_CP1_CANARY))
readonly TOPOLOGY_AB_DIAGNOSTIC=$((
  TP4_CP1_DIAGNOSTIC && (SINGLE_LEARNER_DIAGNOSTIC || TWO_LEARNER_DIAGNOSTIC)
))
if ((TP4_CP1_CANARY)); then
  # A single source island is replayed without a syncer. Keep the two-island
  # source roster so an already-staged island_1 pack remains valid; no second
  # learner is launched and the no-sync hook never consumes the roster.
  readonly NUM_LEARNERS=2
  readonly DEFAULT_LOCAL_BUDGET_STEPS=1
elif ((SINGLE_LEARNER_DIAGNOSTIC)); then
  readonly NUM_LEARNERS=1
  readonly DEFAULT_LOCAL_BUDGET_STEPS=48
elif ((TWO_LEARNER_DIAGNOSTIC)); then
  readonly NUM_LEARNERS=2
  if ((TP4_CP1_DIAGNOSTIC)); then
    readonly DEFAULT_LOCAL_BUDGET_STEPS=24
  else
    readonly DEFAULT_LOCAL_BUDGET_STEPS=48
  fi
else
  readonly NUM_LEARNERS=5
  readonly DEFAULT_LOCAL_BUDGET_STEPS=240
fi
readonly LOCAL_BUDGET_STEPS=${LOCAL_BUDGET_STEPS:-${DEFAULT_LOCAL_BUDGET_STEPS}}
readonly CANARY_ROLLOUT_ID=${CANARY_ROLLOUT_ID:-167}
if ((TP4_CP1_CANARY)); then
  [[ "${CANARY_ROLLOUT_ID}" =~ ^[0-9]+$ ]] || \
    die "CANARY_ROLLOUT_ID must be a nonnegative integer"
  readonly NUM_ROLLOUT=$((CANARY_ROLLOUT_ID + 1))
else
  readonly NUM_ROLLOUT=${LOCAL_BUDGET_STEPS}
fi
readonly CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
readonly REQUIRED_DATASET_VERSION=${REQUIRED_DATASET_VERSION:-qwen38-value-five-islands-contrastive-20260827-v2}
readonly REQUIRED_DATASET_STRATEGY=${REQUIRED_DATASET_STRATEGY:-atomic-thread-reward-contrastive-window-balanced-v2}
readonly REQUIRED_MILES_REVISION=6438fe22d5915c5b60aa81686e854b66ebe6506c
readonly REQUIRED_MILES_CONTRACT_SHA256=d6dad9bb9d41908da0f7d05a09317574892beff8744c0ef8510ffe0970dcfe2c
if ((TP4_CP1_ENABLED)); then
  readonly DEFAULT_SAVE_INTERVAL=${LOCAL_BUDGET_STEPS}
  readonly DEFAULT_SAVE_RETAIN_INTERVAL=${LOCAL_BUDGET_STEPS}
else
  readonly DEFAULT_SAVE_INTERVAL=15
  readonly DEFAULT_SAVE_RETAIN_INTERVAL=999990
fi
readonly SAVE_INTERVAL=${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL}}
readonly SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL:-${DEFAULT_SAVE_RETAIN_INTERVAL}}
readonly SYNC_INTERVAL_STEPS=${SYNC_INTERVAL_STEPS:-12}
readonly PHASE2_REJOIN=${PHASE2_REJOIN:-0}
readonly RECOVERY_LOCAL_STEP_OFFSET=${RECOVERY_LOCAL_STEP_OFFSET:-0}
readonly RECOVERY_UNIT_OFFSET=${RECOVERY_UNIT_OFFSET:-0}
readonly TRAIN_MEMORY_MARGIN_BYTES=${TRAIN_MEMORY_MARGIN_BYTES:-1073741824}
readonly TP4_CP1_MIN_HOST_TOTAL_GIB=${TP4_CP1_MIN_HOST_TOTAL_GIB:-1100}
readonly TP4_CP1_MIN_HOST_AVAILABLE_GIB=${TP4_CP1_MIN_HOST_AVAILABLE_GIB:-800}
readonly TP4_CP1_MIN_DISK_AVAILABLE_GIB=${TP4_CP1_MIN_DISK_AVAILABLE_GIB:-550}
readonly TP4_CP1_OVERLAP_CPU_OPTIMIZER=${TP4_CP1_OVERLAP_CPU_OPTIMIZER:-0}
# Fresh 51-bin head over a pretrained backbone: the head weight group runs
# at VALUE_HEAD_LR_MULT x the backbone LR (scheduler-scaled max/min lr).
readonly VALUE_HEAD_LR_MULT=${VALUE_HEAD_LR_MULT:-30}
# Miles converts these iteration counts to local-context units by multiplying
# the nominal GBS of five.  The audited pack has exactly 687 train contexts per
# island, so 138 iterations gives a 690-context cosine horizon instead of
# reaching min LR before the final 162 contexts.
readonly LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-5}
readonly LR_DECAY_ITERS=${LR_DECAY_ITERS:-138}

LEARNER_ID=${LEARNER_ID:?set LEARNER_ID to an integer}
if ((TP4_CP1_CANARY || SINGLE_LEARNER_DIAGNOSTIC)); then
  VALUE_SYNC_MODE=${VALUE_SYNC_MODE:-none}
else
  VALUE_SYNC_MODE=${VALUE_SYNC_MODE:-diloco}
fi
SYNCER_ADDR=${SYNCER_ADDR:-}
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
readonly MAX_LEARNER_ID=$((NUM_LEARNERS - 1))
((LEARNER_ID >= 0 && LEARNER_ID < NUM_LEARNERS)) || \
  die "LEARNER_ID must be in [0, ${MAX_LEARNER_ID}], got ${LEARNER_ID}"
[[ "${VALUE_SYNC_MODE}" == diloco || "${VALUE_SYNC_MODE}" == none ]] || \
  die "VALUE_SYNC_MODE must be diloco or none"
[[ "${CONTEXT_LENGTH}" == 131072 || "${CONTEXT_LENGTH}" == 262144 ]] || \
  die "CONTEXT_LENGTH must be 131072 or 262144"
if ((TP4_CP1_DIAGNOSTIC && (SINGLE_LEARNER_DIAGNOSTIC || TWO_LEARNER_DIAGNOSTIC))); then
  ((CONTEXT_LENGTH == 131072)) || \
    die "the TP4/CP1 A/B diagnostic requires CONTEXT_LENGTH=131072"
fi
[[ "${LOCAL_BUDGET_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "LOCAL_BUDGET_STEPS must be a positive integer"
[[ "${PHASE2_REJOIN}" =~ ^[01]$ ]] || die "PHASE2_REJOIN must be 0 or 1"
((TP4_CP1_ENABLED == 0 || PHASE2_REJOIN == 0)) || \
  die "PHASE2_REJOIN is not supported for TP4/CP1 CPU-offloaded checkpoints"
if ((TP4_CP1_CANARY)); then
  [[ "${VALUE_SYNC_MODE}" == none ]] || \
    die "TP4/CP1 canary requires VALUE_SYNC_MODE=none"
  ((LOCAL_BUDGET_STEPS == 1)) || \
    die "TP4/CP1 canary requires LOCAL_BUDGET_STEPS=1"
fi
if ((SINGLE_LEARNER_DIAGNOSTIC)); then
  ((TP4_CP1_DIAGNOSTIC == 1)) || \
    die "single-learner diagnostic requires YETO_MILES_VALUE_TP4_CP1=1"
  [[ "${VALUE_SYNC_MODE}" == none ]] || \
    die "single-learner diagnostic requires VALUE_SYNC_MODE=none"
  ((LOCAL_BUDGET_STEPS == 48)) || \
    die "single-learner diagnostic requires LOCAL_BUDGET_STEPS=48"
fi
if ((TWO_LEARNER_DIAGNOSTIC)); then
  [[ "${VALUE_SYNC_MODE}" == diloco ]] || \
    die "two-learner diagnostic requires VALUE_SYNC_MODE=diloco"
  if ((TP4_CP1_DIAGNOSTIC)); then
    ((LOCAL_BUDGET_STEPS == 24)) || \
      die "TP4/CP1 diagnostic requires LOCAL_BUDGET_STEPS=24"
  else
    ((LOCAL_BUDGET_STEPS == 48)) || \
      die "two-learner diagnostic requires LOCAL_BUDGET_STEPS=48"
  fi
fi
if ((HDO_GRADIENT_STREAM)); then
  ((TWO_LEARNER_DIAGNOSTIC == 1 && TP4_CP1_DIAGNOSTIC == 1)) || \
    die "HDO gradient streaming is restricted to the TP4/CP1 two-learner diagnostic"
  [[ "${VALUE_SYNC_MODE}" == diloco ]] || \
    die "HDO gradient streaming requires VALUE_SYNC_MODE=diloco"
  ((CONTEXT_LENGTH == 131072 && LOCAL_BUDGET_STEPS == 24 && PHASE2_REJOIN == 0)) || \
    die "HDO gradient streaming requires the fresh 128K 24-step diagnostic contract"
  ((TP4_CP1_OVERLAP_CPU_OPTIMIZER == 0)) || \
    die "HDO gradient streaming requires TP4_CP1_OVERLAP_CPU_OPTIMIZER=0"
fi
[[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ROLLOUT must be a positive integer"
[[ "${LR_WARMUP_ITERS}" =~ ^[0-9]+$ ]] || die "LR_WARMUP_ITERS must be a nonnegative integer"
[[ "${LR_DECAY_ITERS}" =~ ^[1-9][0-9]*$ ]] || die "LR_DECAY_ITERS must be a positive integer"
if ((TP4_CP1_CANARY == 0)); then
  [[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "SAVE_INTERVAL must be a positive integer"
  [[ "${SAVE_RETAIN_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "SAVE_RETAIN_INTERVAL must be a positive integer"
  ((SAVE_RETAIN_INTERVAL % SAVE_INTERVAL == 0)) || \
    die "SAVE_RETAIN_INTERVAL must be divisible by SAVE_INTERVAL"
fi
if ((MODEL_ONLY_CHECKPOINT)); then
  ((TP4_CP1_CANARY == 0)) || \
    die "model-only checkpointing is incompatible with the no-checkpoint TP4/CP1 canary"
  if [[ "${VALUE_SYNC_MODE}" == diloco ]]; then
    ((TWO_LEARNER_DIAGNOSTIC == 1 && TP4_CP1_DIAGNOSTIC == 1 && CONTEXT_LENGTH == 131072)) || \
      die "model-only DiLoCo checkpoints are restricted to the 128K two-learner diagnostic"
  fi
  ((PHASE2_REJOIN == 0)) || \
    die "model-only checkpointing cannot be used for an optimizer-resume run"
  ((SAVE_INTERVAL == LOCAL_BUDGET_STEPS)) || \
    die "model-only diagnostic must checkpoint exactly once at its terminal step"
fi
readonly SHARED_MODEL_ONLY_CHECKPOINT=$((
  MODEL_ONLY_CHECKPOINT && TWO_LEARNER_DIAGNOSTIC && TP4_CP1_DIAGNOSTIC &&
    CONTEXT_LENGTH == 131072
))
if ((SHARED_MODEL_ONLY_CHECKPOINT)); then
  ((CHECKPOINT_OWNER < NUM_LEARNERS)) || \
    die "checkpoint owner is outside the active learner roster"
  readonly CHECKPOINT_WRITER=$((LEARNER_ID == CHECKPOINT_OWNER))
  readonly CHECKPOINT_OWNER_LOG=${CHECKPOINT_OWNER}
else
  ((CHECKPOINT_OWNER == 0)) || \
    die "checkpoint-owner selection is restricted to the model-only 128K two-learner diagnostic"
  # All existing modes, including the one-learner A/B baseline, retain their
  # original local checkpoint behavior.
  readonly CHECKPOINT_WRITER=1
  readonly CHECKPOINT_OWNER_LOG=local
fi
[[ "${SYNC_INTERVAL_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "SYNC_INTERVAL_STEPS must be a positive integer"
if ((SINGLE_LEARNER_DIAGNOSTIC || TWO_LEARNER_DIAGNOSTIC)); then
  ((SYNC_INTERVAL_STEPS == 12)) || \
    die "A/B diagnostics require SYNC_INTERVAL_STEPS=12"
  ((LOCAL_BUDGET_STEPS % SYNC_INTERVAL_STEPS == 0)) || \
    die "diagnostic LOCAL_BUDGET_STEPS must be divisible by SYNC_INTERVAL_STEPS"
fi
[[ "${TRAIN_MEMORY_MARGIN_BYTES}" =~ ^[0-9]+$ ]] || die "TRAIN_MEMORY_MARGIN_BYTES must be a nonnegative integer"
[[ "${TP4_CP1_MIN_HOST_TOTAL_GIB}" =~ ^[1-9][0-9]*$ ]] || \
  die "TP4_CP1_MIN_HOST_TOTAL_GIB must be a positive integer"
[[ "${TP4_CP1_MIN_HOST_AVAILABLE_GIB}" =~ ^[1-9][0-9]*$ ]] || \
  die "TP4_CP1_MIN_HOST_AVAILABLE_GIB must be a positive integer"
[[ "${TP4_CP1_MIN_DISK_AVAILABLE_GIB}" =~ ^[1-9][0-9]*$ ]] || \
  die "TP4_CP1_MIN_DISK_AVAILABLE_GIB must be a positive integer"
[[ "${TP4_CP1_OVERLAP_CPU_OPTIMIZER}" =~ ^[01]$ ]] || \
  die "TP4_CP1_OVERLAP_CPU_OPTIMIZER must be 0 or 1"
readonly LOCAL_GLOBAL_BATCH_SIZE=5

if ((TP4_CP1_ENABLED)); then
  readonly ISLAND_GPU_COUNT=4
  readonly CONTEXT_PARALLEL_SIZE=1
  readonly TOPOLOGY_PROFILE=tp4-cp1
  OPTIMIZER_RESIDENCY_ARGS=(
    --optimizer-cpu-offload
    --optimizer-offload-fraction 1.0
  )
  if ((TP4_CP1_OVERLAP_CPU_OPTIMIZER)); then
    OPTIMIZER_RESIDENCY_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)
  fi
else
  ((TP4_CP1_OVERLAP_CPU_OPTIMIZER == 0)) || \
    die "TP4_CP1_OVERLAP_CPU_OPTIMIZER requires YETO_MILES_VALUE_TP4_CP1=1"
  readonly ISLAND_GPU_COUNT=8
  readonly CONTEXT_PARALLEL_SIZE=2
  readonly TOPOLOGY_PROFILE=tp4-cp2
  OPTIMIZER_RESIDENCY_ARGS=(--offload-optimizer-states)
fi

if [[ "${VALUE_SYNC_MODE}" == diloco ]]; then
  [[ -n "${SYNCER_ADDR}" ]] || die "SYNCER_ADDR must be set in diloco mode"
  syncer_host=${SYNCER_ADDR%:*}
  syncer_port=${SYNCER_ADDR##*:}
  [[ -n "${syncer_host}" && "${syncer_host}" != "${SYNCER_ADDR}" ]] || die "SYNCER_ADDR must be host:port"
  [[ "${syncer_port}" =~ ^[0-9]+$ ]] || die "SYNCER_ADDR port must be an integer"
  ((syncer_port >= 1 && syncer_port <= 65535)) || die "SYNCER_ADDR port must be in [1, 65535]"
else
  [[ -z "${SYNCER_ADDR}" || "${SYNCER_ADDR}" == none ]] || \
    die "SYNCER_ADDR must be unset or 'none' in no-sync mode"
  SYNCER_ADDR=none
  ((PHASE2_REJOIN == 0)) || die "PHASE2_REJOIN is unavailable in no-sync mode"
fi

readonly ROLLOUT_PLACEHOLDER='{rollout_id}'
[[ "${ISLAND_DATA_TEMPLATE}" == /* ]] || die "ISLAND_DATA_TEMPLATE must be an absolute node-local path"
[[ "${ISLAND_DATA_TEMPLATE}" == *"${ROLLOUT_PLACEHOLDER}"* ]] || die "ISLAND_DATA_TEMPLATE must contain ${ROLLOUT_PLACEHOLDER}"
template_without_first_placeholder=${ISLAND_DATA_TEMPLATE/"${ROLLOUT_PLACEHOLDER}"/}
[[ "${template_without_first_placeholder}" != *"${ROLLOUT_PLACEHOLDER}"* ]] || die "ISLAND_DATA_TEMPLATE must contain exactly one ${ROLLOUT_PLACEHOLDER}"
[[ "${OUTPUT_DIR}" == /* && "${OUTPUT_DIR}" != / ]] || die "OUTPUT_DIR must be an absolute directory other than /"
if ((ANCHOR_SPILL)); then
  readonly ANCHOR_RESIDENCY=nvme
  ((TWO_LEARNER_DIAGNOSTIC == 1 && TP4_CP1_DIAGNOSTIC == 1)) || \
    die "anchor spill is restricted to the TP4/CP1 two-learner diagnostic"
  [[ "${VALUE_SYNC_MODE}" == diloco ]] || \
    die "anchor spill requires VALUE_SYNC_MODE=diloco"
  ((CONTEXT_LENGTH == 131072 && LOCAL_BUDGET_STEPS == 24 && PHASE2_REJOIN == 0)) || \
    die "anchor spill requires the fresh 128K 24-step diagnostic contract"
  readonly ANCHOR_SPILL_DIR=${YETO_MILES_VALUE_ANCHOR_SPILL_DIR:-${OUTPUT_DIR}/diloco_anchors}
  [[ "${ANCHOR_SPILL_DIR}" == /* && "${ANCHOR_SPILL_DIR}" != / ]] || \
    die "YETO_MILES_VALUE_ANCHOR_SPILL_DIR must be an absolute path other than /"
  [[ ! -e "${ANCHOR_SPILL_DIR}" ]] || \
    die "anchor spill directory already exists; use a fresh run OUTPUT_DIR: ${ANCHOR_SPILL_DIR}"
elif [[ -n "${YETO_MILES_VALUE_ANCHOR_SPILL_DIR:-}" ]]; then
  die "YETO_MILES_VALUE_ANCHOR_SPILL_DIR requires YETO_MILES_VALUE_ANCHOR_SPILL=1"
else
  readonly ANCHOR_RESIDENCY=ram
  readonly ANCHOR_SPILL_DIR=none
fi
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
  "${REQUIRED_DATASET_STRATEGY}" "${LEARNER_ID}" "${NUM_LEARNERS}" \
  "${LOCAL_BUDGET_STEPS}" \
  "${SYNC_INTERVAL_STEPS}" "${CONTEXT_LENGTH}" \
  "${TOPOLOGY_AB_DIAGNOSTIC}" <<'PY' || die "dataset manifest failed launch gates"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
required_version = sys.argv[2]
required_strategy = sys.argv[3]
island_id = int(sys.argv[4])
num_learners = int(sys.argv[5])
budget_steps = int(sys.argv[6])
sync_window = int(sys.argv[7])
context_length = int(sys.argv[8])
topology_ab = bool(int(sys.argv[9]))
manifest = json.loads(path.read_text(encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(manifest.get("schema_version") == 3, "dataset schema must be 3")
require(manifest.get("dataset_version") == required_version, "dataset version mismatch")
require(manifest.get("strategy") == required_strategy, "dataset strategy mismatch")
if topology_ab:
    require(manifest.get("num_islands") == num_learners, "A/B dataset learner count mismatch")
    require(num_learners in (1, 2), "A/B learner roster must contain one or two islands")
    require(manifest.get("max_sequence_length") == context_length, "A/B context length mismatch")
    require(len(train_ids := manifest.get("train_rollout_ids") or []) == budget_steps, "A/B train budget mismatch")
    expected_view = "baseline" if num_learners == 1 else "diloco"
    require(manifest.get("view") == expected_view, "A/B dataset view mismatch")
    parent = path.parent.parent / "ab_manifest.json"
    require(parent.is_file(), "A/B parent manifest is missing")
    ab = json.loads(parent.read_text(encoding="utf-8"))
    require((ab.get("verification") or {}).get("launch_ready") is True, "A/B parent is not launch-ready")
    require((ab.get("verification") or {}).get("train_union_sha256") == (manifest.get("verification") or {}).get("train_union_sha256"), "A/B train union mismatch")
else:
    require(manifest.get("num_islands") == 5, "dataset must contain five source islands")
    require(num_learners in (2, 5), "active learner roster must contain two or five islands")
require(0 <= island_id < num_learners, "island is outside active learner roster")
require(manifest.get("sync_window_size") == sync_window, "dataset/sync H mismatch")
train_ids = manifest.get("train_rollout_ids")
require(isinstance(train_ids, list), "missing train rollout IDs")
require(train_ids == list(range(len(train_ids))), "train rollout IDs are not contiguous")
if topology_ab:
    require(budget_steps == len(train_ids), "A/B learner budget must exactly consume train data")
else:
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
if topology_ab:
    require((island.get("train") or {}).get("rollouts") == budget_steps, "A/B island train bucket count mismatch")
    require(
        sorted(item.get("island_id") for item in islands if isinstance(item, dict))
        == list(range(num_learners)),
        "A/B manifest island roster mismatch",
    )
else:
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
[[ -f "${YETO_ROOT}/yeto/megatron/miles_value_island.py" ]] || die "missing Yeto Miles value-island adapter"
[[ -d "${MODEL_DIR}/Qwen3.8-27B" ]] || die "missing Hugging Face checkpoint ${MODEL_DIR}/Qwen3.8-27B"
[[ -d "${MODEL_DIR}/Qwen3.8-27B_torch_dist" ]] || die "missing distributed checkpoint ${MODEL_DIR}/Qwen3.8-27B_torch_dist"
[[ -r "${PROMPT_DATA}" ]] || die "missing prompt-data shim ${PROMPT_DATA}"
[[ -r "${CUSTOM_CONFIG_PATH}" ]] || die "missing Miles critic config ${CUSTOM_CONFIG_PATH}"
if ((TP4_CP1_CANARY)); then
  [[ -z "${MILES_OFFLINE_VALIDATION_START_ROLLOUT:-}" ]] || \
    die "TP4/CP1 canary forbids MILES_OFFLINE_VALIDATION_START_ROLLOUT"
  python3 - "${CUSTOM_CONFIG_PATH}" <<'PY' || die "TP4/CP1 canary config contract failed"
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise RuntimeError("custom config must be a mapping")
if set(config) != {"gae_adaptive"}:
    raise RuntimeError(
        "canary custom config may contain only gae_adaptive; "
        f"found {sorted(config)}"
    )
if not isinstance(config["gae_adaptive"], dict):
    raise RuntimeError("gae_adaptive config must be a mapping")
PY
fi

if ((TP4_CP1_ENABLED)); then
  visible_gpu_count=$(python3 - "${ISLAND_GPU_COUNT}" <<'PY'
import sys

import torch

expected = int(sys.argv[1])
count = torch.cuda.device_count()
if count != expected:
    raise RuntimeError(f"expected {expected} visible GPUs, found {count}")
# H200 SXM is marketed as 141 GB and reports about 139.8 binary GiB through
# torch.cuda. Keep the guard far above 80 GB H100 while accepting real H200s.
minimum_hbm = 139 * 1024**3
for index in range(count):
    properties = torch.cuda.get_device_properties(index)
    if properties.total_memory < minimum_hbm:
        raise RuntimeError(
            f"GPU {index} ({properties.name}) has only "
            f"{properties.total_memory / 1024**3:.1f} GiB HBM; "
            "the TP4/CP1 profile requires H200-class >=139 GiB GPUs"
        )
print(count)
PY
  ) || die "failed to inspect visible CUDA devices"
  [[ "${visible_gpu_count}" == "${ISLAND_GPU_COUNT}" ]] || \
    die "TP4/CP1 requires a GPU-scoped container exposing exactly 4 GPUs; found ${visible_gpu_count}"
  python3 - "${ISLAND_GPU_COUNT}" <<'PY' || die "TP4/CP1 Ray isolation preflight failed"
import sys

import ray

expected_gpus = int(sys.argv[1])
ray.init(address="auto", logging_level="ERROR")
try:
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    cluster_gpus = float(ray.cluster_resources().get("GPU", 0.0))
    if len(alive_nodes) != 1:
        raise RuntimeError(
            f"expected one private ALIVE Ray node, found {len(alive_nodes)}"
        )
    if cluster_gpus != expected_gpus:
        raise RuntimeError(
            f"expected private Ray head with {expected_gpus} GPUs, found {cluster_gpus:g}"
        )
finally:
    ray.shutdown()
PY

  read -r host_total_kib host_available_kib < <(
    python3 - /proc/meminfo <<'PY'
import sys
from pathlib import Path

values = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    values[key] = int(value.strip().split()[0])
total_kib = values["MemTotal"]
available_kib = values["MemAvailable"]

# /proc/meminfo can expose host totals even when the container has a much
# smaller cgroup ceiling. Apply the effective v2 limit when present.
cgroup_max = Path("/sys/fs/cgroup/memory.max")
cgroup_current = Path("/sys/fs/cgroup/memory.current")
if cgroup_max.is_file() and cgroup_current.is_file():
    raw_limit = cgroup_max.read_text(encoding="utf-8").strip()
    if raw_limit != "max":
        limit_kib = int(raw_limit) // 1024
        current_kib = int(cgroup_current.read_text(encoding="utf-8").strip()) // 1024
        total_kib = min(total_kib, limit_kib)
        available_kib = min(available_kib, max(0, limit_kib - current_kib))
print(total_kib, available_kib)
PY
  ) || die "failed to inspect host memory"
  readonly kib_per_gib=1048576
  ((host_total_kib >= TP4_CP1_MIN_HOST_TOTAL_GIB * kib_per_gib)) || \
    die "TP4/CP1 host RAM is too small: require at least ${TP4_CP1_MIN_HOST_TOTAL_GIB} GiB total"
  ((host_available_kib >= TP4_CP1_MIN_HOST_AVAILABLE_GIB * kib_per_gib)) || \
    die "TP4/CP1 host RAM headroom is too small: require at least ${TP4_CP1_MIN_HOST_AVAILABLE_GIB} GiB available"
  [[ "$(ulimit -l)" == unlimited ]] || \
    die "TP4/CP1 requires unlimited locked memory; start the container with --ulimit memlock=-1"
  output_parent=${OUTPUT_DIR%/*}
  [[ -n "${output_parent}" ]] || output_parent=/
  mkdir -p "${output_parent}"
  disk_available_kib=$(df -Pk "${output_parent}" | awk 'NR == 2 {print $4}') || \
    die "failed to inspect TP4/CP1 output-volume headroom"
  [[ "${disk_available_kib}" =~ ^[0-9]+$ ]] || \
    die "invalid TP4/CP1 output-volume headroom"
  ((disk_available_kib >= TP4_CP1_MIN_DISK_AVAILABLE_GIB * kib_per_gib)) || \
    die "TP4/CP1 disk headroom is too small: require at least ${TP4_CP1_MIN_DISK_AVAILABLE_GIB} GiB available"
fi

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
  if ((TP4_CP1_CANARY)); then
    readonly START_ROLLOUT_ID=${CANARY_ROLLOUT_ID}
  else
    readonly START_ROLLOUT_ID=0
  fi
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
if ((TOPOLOGY_AB_DIAGNOSTIC)); then
  python3 - "${ISLAND_DATA_DIR}" "${LOCAL_BUDGET_STEPS}" \
    "${CONTEXT_LENGTH}" <<'PY' || die "A/B replay payload preflight failed"
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
budget = int(sys.argv[2])
limit = int(sys.argv[3])
for rollout_id in range(budget):
    path = root / f"data_{rollout_id}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("rollout_id") != rollout_id:
        raise RuntimeError(f"{path}: payload rollout_id mismatch")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"{path}: missing samples")
    rewards = set()
    for position, sample in enumerate(samples):
        tokens = sample.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise RuntimeError(f"{path}: sample {position} has invalid tokens")
        if len(tokens) > limit:
            raise RuntimeError(
                f"{path}: sample {position} has {len(tokens)} tokens > {limit}"
            )
        reward = float(sample.get("reward"))
        if reward not in (0.0, 1.0):
            raise RuntimeError(f"{path}: sample {position} has invalid reward")
        rewards.add(int(reward))
    if rewards != {0, 1}:
        raise RuntimeError(f"{path}: optimizer bucket is not contrastive")
PY
fi

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
  FINALIZATION_TIMEOUT_VALUE="${FINALIZATION_TIMEOUT:-1800}" \
  HDO_GRADIENT_STREAM_VALUE="${HDO_GRADIENT_STREAM}" \
  LOCAL_STEP_OFFSET_VALUE="${START_ROLLOUT_ID}" \
  UNIT_OFFSET_VALUE="${UNIT_OFFSET}" \
  python3 - "${YETO_ROOT}" "${MILES_ROOT}" \
    "${SYNCER_ADDR}" "${LEARNER_ID}" "${NUM_LEARNERS}" \
    "${LOCAL_BUDGET_STEPS}" "${SYNC_INTERVAL_STEPS}" \
    "${TOPOLOGY_PROFILE}" "${ANCHOR_SPILL_DIR}" <<'PY'
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
    topology_profile,
    anchor_spill_dir,
) = sys.argv[1:]

worker_env = {
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
    "YETO_VALUE_FINALIZATION_TIMEOUT": os.environ.get("FINALIZATION_TIMEOUT_VALUE", "1800"),
    "YETO_VALUE_BUDGET_STEPS": budget_steps,
    "YETO_VALUE_MIN_LOCAL_STEPS": min_local_steps,
    "YETO_VALUE_LOCAL_STEP_OFFSET": os.environ["LOCAL_STEP_OFFSET_VALUE"],
    "YETO_VALUE_UNIT_OFFSET": os.environ["UNIT_OFFSET_VALUE"],
    "YETO_VALUE_PHASE2_REJOIN": os.environ["PHASE2_REJOIN_VALUE"],
    "YETO_VALUE_TOPOLOGY_PROFILE": topology_profile,
    "YETO_VALUE_HDO_GRADIENT_STREAM": os.environ["HDO_GRADIENT_STREAM_VALUE"],
}
if anchor_spill_dir != "none":
    worker_env["YETO_VALUE_ANCHOR_SPILL_DIR"] = anchor_spill_dir
print(json.dumps(worker_env, separators=(",", ":")))
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

printf 'Launching Miles value island %d/%d: diagnostic=%d canary=%d topology=%s gpus=%d train=%d rollout_range=[%d,%d) context=%d data=%s output=%s syncer=%s ray_node=%s checkpoint_owner=%s checkpoint_writer=%d anchor_residency=%s anchor_spill_dir=%s\n' \
  "${LEARNER_ID}" "${NUM_LEARNERS}" "${TWO_LEARNER_DIAGNOSTIC}" \
  "${TP4_CP1_CANARY}" "${TOPOLOGY_PROFILE}" "${ISLAND_GPU_COUNT}" \
  "${LOCAL_BUDGET_STEPS}" "${START_ROLLOUT_ID}" "${NUM_ROLLOUT}" \
  "${CONTEXT_LENGTH}" \
  "${ISLAND_DATA_TEMPLATE}" "${OUTPUT_DIR}" "${SYNCER_ADDR}" \
  "${MILES_RAY_TARGET_NODE_IP}" "${CHECKPOINT_OWNER_LOG}" \
  "${CHECKPOINT_WRITER}" "${ANCHOR_RESIDENCY}" \
  "${ANCHOR_SPILL_DIR}"

if [[ "${VALUE_SYNC_MODE}" == diloco ]]; then
  VALUE_HOOK_ARGS=(
    --custom-megatron-after-model-init-hook-path yeto.megatron.miles_value_island.after_model_init
    --custom-megatron-before-train-step-hook-path yeto.megatron.miles_value_island.before_train_step
    --custom-megatron-after-train-step-hook-path yeto.megatron.miles_value_island.after_train_step
  )
else
  # Preserve the BF16 precision-aware optimizer compatibility fixes while
  # bypassing every Yeto fragment/sync boundary for the matched baseline.
  VALUE_HOOK_ARGS=(
    --custom-megatron-after-model-init-hook-path yeto_value_validation_hook.after_model_init
  )
fi

CHECKPOINT_ARGS=()
if ((TP4_CP1_CANARY == 0 && CHECKPOINT_WRITER)); then
  CHECKPOINT_ARGS+=(
    --save "${OUTPUT_DIR}/checkpoints"
    --critic-save "${CRITIC_SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL}"
    --save-retain-interval "${SAVE_RETAIN_INTERVAL}"
  )
  if ((MODEL_ONLY_CHECKPOINT)); then
    CHECKPOINT_ARGS+=(--no-save-optim --no-save-rng)
  fi
fi

CRITIC_PLACEMENT_ARGS=(--colocate-critic)
if ((MODEL_ONLY_CHECKPOINT)); then
  # This is a critic-only job: pinned Miles already allocates no actor. Avoid
  # colocate-critic's redundant CP2 TensorBackuper, which otherwise retains a
  # second ~111-GB host copy while the final model-only checkpoint is staged.
  CRITIC_PLACEMENT_ARGS=()
fi

# Decoder-only Megatron derives encoder_seq_length from seq_length and rejects
# specifying both flags on the command line.
exec python3 train_async.py \
  --num-gpus-per-node "${ISLAND_GPU_COUNT}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ISLAND_GPU_COUNT}" \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node "${ISLAND_GPU_COUNT}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}/Qwen3.8-27B" \
  --load "${MODEL_DIR}/Qwen3.8-27B_torch_dist" \
  --critic-load "${CRITIC_LOAD_DIR}" \
  --ref-load "${MODEL_DIR}/Qwen3.8-27B_torch_dist" \
  --ckpt-format torch_dist \
  "${CHECKPOINT_ARGS[@]}" \
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
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}" \
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
  "${OPTIMIZER_RESIDENCY_ARGS[@]}" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --train-env-vars "${TRAIN_ENV_VARS}" \
  --distributed-timeout-minutes 60 \
  --empty-unused-memory-level 2 \
  --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}" \
  "${RECOVERY_CLI_ARGS[@]}" \
  "${CRITIC_PLACEMENT_ARGS[@]}" \
  "${VALUE_HOOK_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
