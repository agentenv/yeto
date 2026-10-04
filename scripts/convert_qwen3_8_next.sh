#!/usr/bin/env bash
# HF -> Megatron torch_dist conversion for Qwen3.8-Flash-Next (qwen4_exp).
#
# Runs inside the pinned Miles image (MILES_NEXT_IMAGE + m3-qwen4exp-lora
# overlay) with yeto mounted; renders the exact command Miles CI uses
# (tests/e2e/megatron/model_scripts/test_qwen3_8_next_4layer_ci.py::prepare)
# through yeto.rl.profiles.qwen3_8_next so --dry-run prints what a real run
# executes.  Idempotent: an output whose tracker reads "release" is kept.
#
#   scripts/convert_qwen3_8_next.sh [--variant 4layer|full] [--hf-checkpoint DIR]
#       [--output DIR] [--nproc N] [--miles-root DIR] [--megatron-path DIR]
#       [--yeto-root DIR] [--dry-run]
set -euo pipefail

VARIANT=${YETO_Q38N_VARIANT:-4layer}
HF_CHECKPOINT=""
OUTPUT=""
NPROC=""
MILES_ROOT=${YETO_Q38N_MILES_ROOT:-/root/miles}
MEGATRON_PATH=${YETO_Q38N_MEGATRON_PATH:-/root/Megatron-LM}
YETO_ROOT=${YETO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT=$2; shift 2 ;;
    --hf-checkpoint) HF_CHECKPOINT=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --nproc) NPROC=$2; shift 2 ;;
    --miles-root) MILES_ROOT=$2; shift 2 ;;
    --megatron-path) MEGATRON_PATH=$2; shift 2 ;;
    --yeto-root) YETO_ROOT=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export YETO_Q38N_VARIANT=$VARIANT
export YETO_Q38N_MILES_ROOT=$MILES_ROOT
export YETO_Q38N_MEGATRON_PATH=$MEGATRON_PATH
export PYTHONPATH=${YETO_ROOT}${PYTHONPATH:+:$PYTHONPATH}

profile() { python3 -m yeto.rl.profiles.qwen3_8_next "$@"; }

extra=()
[[ -n $HF_CHECKPOINT ]] && extra+=(--hf-checkpoint "$HF_CHECKPOINT")
[[ -n $OUTPUT ]] && extra+=(--output "$OUTPUT")
[[ -n $NPROC ]] && extra+=(--nproc "$NPROC")

CONVERT_CMD=$(profile convert-command "${extra[@]}")
CONVERT_ENV=$(profile convert-env)
# the --save path is the token after "--save" in the rendered command
SAVE_DIR=$(python3 -c 'import shlex,sys; t=shlex.split(sys.argv[1]); print(t[t.index("--save")+1])' "$CONVERT_CMD")
HF_DIR=$(python3 -c 'import shlex,sys; t=shlex.split(sys.argv[1]); print(t[t.index("--hf-checkpoint")+1])' "$CONVERT_CMD")

echo "# variant=${VARIANT} hf=${HF_DIR} save=${SAVE_DIR}"
echo "# env:"; echo "$CONVERT_ENV" | sed 's/^/#   /'
echo "$CONVERT_CMD"
if [[ $DRY_RUN -eq 1 ]]; then
  exit 0
fi

test -f "${HF_DIR}/config.json" || { echo "missing HF checkpoint ${HF_DIR}/config.json" >&2; exit 1; }
if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]] \
   && [[ "$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")" == release ]]; then
  echo "# ${SAVE_DIR} already converted (tracker=release); skipping"
  exit 0
fi
mkdir -p "$SAVE_DIR"
start=$(date +%s)
# shellcheck disable=SC2086
env $(echo "$CONVERT_ENV" | xargs) bash -c "cd '${MILES_ROOT}' && ${CONVERT_CMD}"
test "$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")" == release
profile manifest > "${SAVE_DIR}/yeto-profile-manifest.json"
echo "# converted in $(( $(date +%s) - start ))s -> ${SAVE_DIR}"
