#!/usr/bin/env bash
# Run the Yeto syncer on server 0.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_file "${SYNCER_BIN}" "syncer binary; run 00_prepare.sh or cargo build --release"

log="${LOG_DIR}/syncer.log"
echo "Starting syncer on 0.0.0.0:${SYNC_PORT}; log=${log}"
echo "Make sure TCP ${SYNC_PORT} is reachable from server 1."

stdbuf -oL -eL "${SYNCER_BIN}" \
  --port "${SYNC_PORT}" \
  --learners "${NUM_LEARNERS}" \
  --quorum "${SYNC_QUORUM}" \
  --grace-ms "${SYNC_GRACE_MS}" \
  --quorum-timeout-s "${SYNC_QUORUM_TIMEOUT_S}" \
  --total-steps "${TOTAL_STEPS}" \
  --outer-lr "${OUTER_LR}" \
  --outer-momentum "${OUTER_MOMENTUM}" \
  --checkpoint-path "${SYNC_CKPT}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --resume \
  --final-state "${FINAL_STATE}" \
  --event-tape "${EVENT_TAPE}" 2>&1 | tee "${log}"
