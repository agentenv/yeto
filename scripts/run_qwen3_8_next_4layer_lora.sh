#!/usr/bin/env bash
# One-command Qwen3.8-Flash-Next-4layer native LoRA GRPO on a single 8xH100 node.
#
# Inside the pinned Miles image (MILES_NEXT_IMAGE rebuilt with the Miles
# m3-qwen4exp-lora and sglang m3-qwen4exp-lora overlays) with yeto mounted:
#   1. download the HF 4-layer slice (pinned revision) and DAPO-Math-17k;
#   2. convert HF -> torch_dist (scripts/convert_qwen3_8_next.sh, idempotent);
#   3. start a local ray head (MILES_SCRIPT_EXTERNAL_RAY=1) and run
#      Miles' scripts/run_qwen3_8_next.py with the M3 LoRA flags, under a hard
#      `timeout` so a wedged run cannot hold the GPUs.
# Every step is rendered by yeto.rl.profiles.qwen3_8_next; --dry-run prints
# the commands and exits.  Override fields with YETO_Q38N_<FIELD> (for example
# YETO_Q38N_NUM_ROLLOUT=20 YETO_Q38N_LORA_EXPERT_RANK=16).
#
#   scripts/run_qwen3_8_next_4layer_lora.sh [--dry-run] [--skip-download]
#       [--skip-convert] [--timeout SECONDS] [--yeto-root DIR] [-- extra train.py args]
set -euo pipefail

DRY_RUN=0
SKIP_DOWNLOAD=0
SKIP_CONVERT=0
HARD_TIMEOUT=${YETO_Q38N_HARD_TIMEOUT:-5400}
YETO_ROOT=${YETO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-convert) SKIP_CONVERT=1; shift ;;
    --timeout) HARD_TIMEOUT=$2; shift 2 ;;
    --yeto-root) YETO_ROOT=$2; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export YETO_Q38N_VARIANT=4layer
export PYTHONPATH=${YETO_ROOT}${PYTHONPATH:+:$PYTHONPATH}
profile() { python3 -m yeto.rl.profiles.qwen3_8_next "$@"; }

MANIFEST=$(profile manifest)
NUM_GPUS=$(python3 -c 'import json,sys; m=json.loads(sys.argv[1]); print(m["parallel"]["ep"]*m["parallel"]["pp"])' "$MANIFEST")
LAUNCH_ENV=$(profile launch-env)
LAUNCH_CMD=$(profile launch-command -- "${EXTRA[@]}")
MILES_ROOT=$(python3 -c 'import os; print(os.environ.get("YETO_Q38N_MILES_ROOT", "/root/miles"))')

echo "# profile manifest:"; echo "$MANIFEST" | sed 's/^/#   /'
echo "# step 1 download:"; profile download-commands | sed 's/^/  /'
echo "# step 2 convert:"; echo "  ${YETO_ROOT}/scripts/convert_qwen3_8_next.sh --variant 4layer --yeto-root ${YETO_ROOT}"
echo "# step 3 launch (hard timeout ${HARD_TIMEOUT}s, ${NUM_GPUS} GPUs):"
echo "$LAUNCH_ENV" | sed 's/^/#   env /'
echo "  timeout --signal=TERM --kill-after=120 ${HARD_TIMEOUT} ${LAUNCH_CMD}"
if [[ $DRY_RUN -eq 1 ]]; then
  exit 0
fi

if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  profile download-commands | while IFS= read -r line; do bash -c "$line"; done
fi
if [[ $SKIP_CONVERT -eq 0 ]]; then
  "${YETO_ROOT}/scripts/convert_qwen3_8_next.sh" --variant 4layer --yeto-root "${YETO_ROOT}"
fi

if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus "${NUM_GPUS}" --disable-usage-stats
fi
# shellcheck disable=SC2086
env $(echo "$LAUNCH_ENV" | xargs) bash -c "cd '${MILES_ROOT}' && timeout --signal=TERM --kill-after=120 ${HARD_TIMEOUT} ${LAUNCH_CMD}"
