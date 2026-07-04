#!/usr/bin/env bash
# Shared defaults for the two-node NAVA L4 smoke test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YETO_ROOT="${YETO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

: "${RUN_DIR:=/tmp/yeto-nava-l4-smoke}"
if [[ -f "${RUN_DIR}/env.sh" && "${YETO_L4_IGNORE_ENV_FILE:-0}" != "1" ]]; then
  # shellcheck disable=SC1090
  source "${RUN_DIR}/env.sh"
fi
: "${NAVA_SMOKE_BACKEND:=mock}"
if [[ "${NAVA_SMOKE_BACKEND}" == "mock" ]]; then
  : "${NAVA_ROOT:=${RUN_DIR}/mock_nava}"
  : "${SKIP_ASSET_CHECK:=1}"
else
  : "${NAVA_ROOT:=/opt/NAVA}"
fi
: "${NAVA_ASSETS_DIR:=/local_nvme/nava_assets}"
: "${LABEL_INPUT:=s3://seahorse-openclip-keepers-533462777468-us-west-2/xv/lora_labels/gemini3_full_v3/}"
: "${NAVA_S3_REGION:=us-west-2}"

: "${NUM_LEARNERS:=2}"
: "${SYNC_PORT:=29400}"
: "${SYNC_QUORUM:=${NUM_LEARNERS}}"
: "${SYNC_GRACE_MS:=1000}"
: "${SYNC_QUORUM_TIMEOUT_S:=900}"
: "${TOTAL_STEPS:=4}"
: "${OUTER_LR:=0.7}"
: "${OUTER_MOMENTUM:=0.9}"
: "${CHECKPOINT_EVERY:=1}"

: "${MAX_OUTPUT_ROWS:=16}"
: "${MIN_OUTPUT_ROWS:=2}"
: "${CAPTION_FIELD:=composed}"
: "${QUALITY:=good,high}"
: "${MIN_DURATION:=1.0}"
: "${MAX_DURATION:=4.0}"
: "${FFPROBE:=0}"

: "${FRAGMENTS:=2}"
: "${WIRE_DTYPE:=bf16}"
: "${WAN_STREAMS:=1}"
: "${NAVA_GPUS_PER_NODE:=1}"
: "${NAVA_BATCH_SIZE:=1}"
: "${NAVA_GRAD_ACCUM:=1}"
: "${NAVA_LOCAL_STEPS:=${TOTAL_STEPS}}"
: "${NAVA_SAVE_EVERY:=1}"
: "${NAVA_LR:=1.0e-4}"
: "${NAVA_WEIGHT_DECAY:=0}"
: "${NAVA_WARMUP_STEPS:=0}"
: "${NAVA_NUM_WORKERS:=0}"
: "${NAVA_IO_WORKERS:=1}"
: "${NAVA_MODALITY:=text_to_video}"
: "${LORA_R:=1}"
: "${LORA_ALPHA:=1}"
: "${LORA_TARGETS:=^backbone\.(double_blocks\.0|single_blocks\.0)\..*(self_attn|cross_attn).*}"

: "${SMOKE_DIM:=256}"
: "${SMOKE_FFN_DIM:=1024}"
: "${SMOKE_HEADS:=4}"
: "${SMOKE_DOUBLE_LAYERS:=1}"
: "${SMOKE_SINGLE_LAYERS:=1}"
: "${SMOKE_IMAGE_SIZE:=256}"
: "${SMOKE_VIDEO_FPS:=8}"
: "${SMOKE_VIDEO_TGT_FRAMES:=17}"
: "${SMOKE_SEED:=1234}"
: "${SMOKE_LATENT_DIM:=128}"
: "${SMOKE_SEQ_LEN:=8}"

SMOKE_CONFIG="${RUN_DIR}/nava_l4_smoke.yaml"
SMOKE_JOINT_CONFIG="${RUN_DIR}/joint_l4_smoke.json"
: "${SMOKE_BASE_CKPT:=${RUN_DIR}/tiny_base.ckpt}"
SMOKE_DATA_JSONL="${RUN_DIR}/train.nava.jsonl"
SMOKE_REPORT="${RUN_DIR}/filter_report.json"
SYNC_CKPT="${RUN_DIR}/yeto-state.ckpt"
FINAL_STATE="${RUN_DIR}/yeto-final-state.bin"
EVENT_TAPE="${RUN_DIR}/yeto-tape.jsonl"
LOG_DIR="${RUN_DIR}/logs"
OUTPUT_DIR="${RUN_DIR}/output"
EXPORT_DIR="${RUN_DIR}/export"
SYNCER_BIN="${YETO_ROOT}/syncer/target/release/yeto-syncer"

export PYTHONPATH="${RUN_DIR}:${YETO_ROOT}:${PYTHONPATH:-}"
export PATH="${HOME}/.local/bin:${PATH}"
export NAVA_S3_REGION
export YETO_NAVA_DATA_CACHE="${YETO_NAVA_DATA_CACHE:-${RUN_DIR}/cache/data}"
export YETO_NAVA_ASSET_CACHE="${YETO_NAVA_ASSET_CACHE:-${NAVA_ASSETS_DIR}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}" "${YETO_NAVA_DATA_CACHE}"

require_file() {
  local path="$1"
  local what="$2"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing ${what}: ${path}" >&2
    return 1
  fi
}

nava_global_batch_size() {
  echo $((NUM_LEARNERS * NAVA_GPUS_PER_NODE * NAVA_BATCH_SIZE * NAVA_GRAD_ACCUM))
}

print_smoke_env() {
  cat <<ENV
YETO_ROOT=${YETO_ROOT}
RUN_DIR=${RUN_DIR}
NAVA_ROOT=${NAVA_ROOT}
NAVA_SMOKE_BACKEND=${NAVA_SMOKE_BACKEND}
NAVA_ASSETS_DIR=${NAVA_ASSETS_DIR}
LABEL_INPUT=${LABEL_INPUT}
SMOKE_CONFIG=${SMOKE_CONFIG}
SMOKE_BASE_CKPT=${SMOKE_BASE_CKPT}
SMOKE_DATA_JSONL=${SMOKE_DATA_JSONL}
SYNC_CKPT=${SYNC_CKPT}
EVENT_TAPE=${EVENT_TAPE}
FRAGMENTS=${FRAGMENTS}
TOTAL_STEPS=${TOTAL_STEPS}
NUM_LEARNERS=${NUM_LEARNERS}
NAVA_GPUS_PER_NODE=${NAVA_GPUS_PER_NODE}
NAVA_BATCH_SIZE=${NAVA_BATCH_SIZE}
NAVA_GRAD_ACCUM=${NAVA_GRAD_ACCUM}
NAVA_GLOBAL_BATCH=$(nava_global_batch_size)
LORA_TARGETS=${LORA_TARGETS}
SMOKE_LATENT_DIM=${SMOKE_LATENT_DIM}
SMOKE_SEQ_LEN=${SMOKE_SEQ_LEN}
ENV
}
