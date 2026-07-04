#!/usr/bin/env bash
# Run one NAVA learner. Start once on each L4 server with LEARNER_ID=0/1.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -z "${LEARNER_ID:-}" || -z "${SYNCER_ADDR:-}" ]]; then
  cat >&2 <<USAGE
Usage:
  LEARNER_ID=0 SYNCER_ADDR=<server0-ip>:${SYNC_PORT} $0
  LEARNER_ID=1 SYNCER_ADDR=<server0-ip>:${SYNC_PORT} $0
USAGE
  exit 2
fi

if [[ "${NAVA_SMOKE_BACKEND}" == "mock" ]]; then
  require_file "${NAVA_ROOT}/nava_src/pipeline_smoke.py" "mock NAVA smoke pipeline"
else
  require_file "${NAVA_ROOT}/nava_src/pipeline_nava.py" "NAVA checkout"
fi
require_file "${SMOKE_CONFIG}" "smoke config"
require_file "${SMOKE_BASE_CKPT}" "tiny base checkpoint"
require_file "${SMOKE_DATA_JSONL}" "NAVA JSONL data"

if [[ "${SKIP_ASSET_CHECK:-0}" != "1" ]]; then
  require_file "${NAVA_ASSETS_DIR}/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" "UMT5 encoder checkpoint"
  require_file "${NAVA_ASSETS_DIR}/Wan2.2-TI2V-5B/google/umt5-xxl" "UMT5 tokenizer"
  case "${NAVA_MODALITY}" in
    text_to_video|text_to_av)
      require_file "${NAVA_ASSETS_DIR}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" "Wan video VAE"
      ;;
  esac
  case "${NAVA_MODALITY}" in
    text_to_audio|text_to_av)
      require_file "${NAVA_ASSETS_DIR}/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors" "LTX audio VAE"
      ;;
  esac
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found; continuing" >&2
fi

if [[ ! "${NAVA_GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || (( NAVA_GPUS_PER_NODE < 1 )); then
  echo "ERROR: NAVA_GPUS_PER_NODE must be a positive integer, got '${NAVA_GPUS_PER_NODE}'" >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((NAVA_GPUS_PER_NODE - 1))")"
else
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#visible_gpus[@]} < NAVA_GPUS_PER_NODE )); then
    echo "ERROR: CUDA_VISIBLE_DEVICES exposes ${#visible_gpus[@]} GPU(s), but NAVA_GPUS_PER_NODE=${NAVA_GPUS_PER_NODE}" >&2
    exit 2
  fi
fi
export CUDA_VISIBLE_DEVICES

master_port="$((29500 + LEARNER_ID))"
out_dir="${OUTPUT_DIR}/learner-${LEARNER_ID}"
log="${LOG_DIR}/learner-${LEARNER_ID}.log"
mkdir -p "${out_dir}"

echo "Starting learner ${LEARNER_ID}; syncer=${SYNCER_ADDR}; gpus=${NAVA_GPUS_PER_NODE}; global_batch=$(nava_global_batch_size); log=${log}"

torchrun --nnodes=1 --nproc_per_node="${NAVA_GPUS_PER_NODE}" \
  --master_addr=127.0.0.1 \
  --master_port="${master_port}" \
  -m yeto.nava.learner \
  --syncer "${SYNCER_ADDR}" \
  --learner-id "${LEARNER_ID}" \
  --num-learners "${NUM_LEARNERS}" \
  --nava-root "${NAVA_ROOT}" \
  --nava-config "${SMOKE_CONFIG}" \
  --nava-ckpt "${SMOKE_BASE_CKPT}" \
  --nava-data "${SMOKE_DATA_JSONL}" \
  --nava-data-format nava-jsonl \
  --nava-data-cache "${YETO_NAVA_DATA_CACHE}" \
  --nava-assets-dir "${NAVA_ASSETS_DIR}" \
  --nava-modality "${NAVA_MODALITY}" \
  --nava-min-duration "${MIN_DURATION}" \
  --nava-max-duration "${MAX_DURATION}" \
  --nava-tuning lora \
  --nava-lora-r "${LORA_R}" \
  --nava-lora-alpha "${LORA_ALPHA}" \
  --nava-lora-targets "${LORA_TARGETS}" \
  --shard ddp \
  --fragments "${FRAGMENTS}" \
  --wire-dtype "${WIRE_DTYPE}" \
  --wan-streams "${WAN_STREAMS}" \
  --nava-batch-size "${NAVA_BATCH_SIZE}" \
  --nava-grad-accum "${NAVA_GRAD_ACCUM}" \
  --nava-lr "${NAVA_LR}" \
  --nava-weight-decay "${NAVA_WEIGHT_DECAY}" \
  --nava-warmup-steps "${NAVA_WARMUP_STEPS}" \
  --nava-max-local-steps "${NAVA_LOCAL_STEPS}" \
  --nava-num-workers "${NAVA_NUM_WORKERS}" \
  --nava-io-workers "${NAVA_IO_WORKERS}" \
  --nava-disable-ema \
  --nava-save-every "${NAVA_SAVE_EVERY}" \
  --output-dir "${out_dir}" 2>&1 | tee "${log}"
