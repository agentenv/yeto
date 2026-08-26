#!/usr/bin/env bash
set -euo pipefail

# Supervise the exact two-process learner-budget protocol used by the Miles
# value islands: an unmarked cutoff checkpoint, followed by one full-quorum
# serial consolidation pass over every fragment and authoritative finalization.

die() {
  printf 'run_miles_value_syncer_supervisor.sh: %s\n' "$*" >&2
  exit 2
}

readonly NUM_LEARNERS=6
readonly LOCAL_BUDGET_STEPS=364

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
YETO_ROOT=${YETO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}
SYNCER_BIN=${SYNCER_BIN:-${YETO_ROOT}/syncer/target/release/yeto-syncer}
ISO_WORKER_PYTHON=${ISO_WORKER_PYTHON:-${YETO_ROOT}/scripts/docker_python_iso_worker.sh}
ISO_WORKER_DEVICES=${ISO_WORKER_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}
ISO_WORKER_QUEUE_CAPACITY=${ISO_WORKER_QUEUE_CAPACITY:-16}

RUN_DIR=${RUN_DIR:?set RUN_DIR to a fresh local syncer output directory}
PORT=${PORT:-29400}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${RUN_DIR}/state.ckpt}
EVENT_TAPE=${EVENT_TAPE:-${RUN_DIR}/events.jsonl}
PHASE1_LOG=${PHASE1_LOG:-${RUN_DIR}/syncer-cutoff.log}
PHASE2_LOG=${PHASE2_LOG:-${RUN_DIR}/syncer-finalize.log}
FINAL_MARKER=${CHECKPOINT_PATH}.final

PHASE1_TOTAL_STEPS=${PHASE1_TOTAL_STEPS:-1000000000}
PHASE1_PIPELINE=${PHASE1_PIPELINE:-16}
PHASE2_PIPELINE=${PHASE2_PIPELINE:-16}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-0}
GRACE_MS=${GRACE_MS:-1000}
GRACE_GAMMA=${GRACE_GAMMA:-0.8}
GRACE_TAU=${GRACE_TAU:-2.0}
QUORUM_TIMEOUT_S=${QUORUM_TIMEOUT_S:-7200}
SYNC_INTERVAL_STEPS=${SYNC_INTERVAL_STEPS:-24}
DELTA_CORRECTION=${DELTA_CORRECTION:-heloco}
OUTER_LR=${OUTER_LR:-0.7}
OUTER_MOMENTUM=${OUTER_MOMENTUM:-0.9}

[[ "${RUN_DIR}" == /* && "${RUN_DIR}" != / ]] || die "RUN_DIR must be an absolute directory other than /"
[[ "${CHECKPOINT_PATH}" == /* ]] || die "CHECKPOINT_PATH must be absolute"
[[ "${EVENT_TAPE}" == /* ]] || die "EVENT_TAPE must be absolute"
[[ "${PHASE1_LOG}" == /* && "${PHASE2_LOG}" == /* ]] || die "phase logs must be absolute"
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  die "PORT must be in [1, 65535]"
fi
if [[ ! "${PHASE1_TOTAL_STEPS}" =~ ^[0-9]+$ ]] || ((PHASE1_TOTAL_STEPS <= LOCAL_BUDGET_STEPS)); then
  die "PHASE1_TOTAL_STEPS must exceed ${LOCAL_BUDGET_STEPS}"
fi
if [[ ! "${PHASE1_PIPELINE}" =~ ^[0-9]+$ ]] || ((PHASE1_PIPELINE < 1)); then
  die "PHASE1_PIPELINE must be positive"
fi
if [[ ! "${PHASE2_PIPELINE}" =~ ^[0-9]+$ ]] || ((PHASE2_PIPELINE < 1)); then
  die "PHASE2_PIPELINE must be positive"
fi
if [[ ! "${ISO_WORKER_QUEUE_CAPACITY}" =~ ^[0-9]+$ ]] || ((ISO_WORKER_QUEUE_CAPACITY < 1)); then
  die "ISO_WORKER_QUEUE_CAPACITY must be positive"
fi
[[ "${CHECKPOINT_EVERY}" =~ ^[0-9]+$ ]] || die "CHECKPOINT_EVERY must be a non-negative integer"
[[ "${GRACE_MS}" =~ ^[0-9]+$ ]] || die "GRACE_MS must be a non-negative integer"
if [[ ! "${QUORUM_TIMEOUT_S}" =~ ^[0-9]+$ ]] || ((QUORUM_TIMEOUT_S < 1)); then
  die "QUORUM_TIMEOUT_S must be positive"
fi
[[ "${DELTA_CORRECTION}" == heloco || "${DELTA_CORRECTION}" == none ]] || die "DELTA_CORRECTION must be heloco or none"

[[ -x "${SYNCER_BIN}" ]] || die "syncer binary is not executable: ${SYNCER_BIN}"
[[ -f "${YETO_ROOT}/yeto/iso_worker.py" ]] || die "missing exact Torch SVD worker under ${YETO_ROOT}"

mkdir -p -- "${RUN_DIR}"
for fresh_path in \
  "${CHECKPOINT_PATH}" "${FINAL_MARKER}" "${FINAL_MARKER}.tmp" \
  "${EVENT_TAPE}" "${PHASE1_LOG}" "${PHASE2_LOG}"; do
  [[ ! -e "${fresh_path}" ]] || die "refusing to mix with an existing run artifact: ${fresh_path}"
done

export YETO_ROOT
export PYTHONPATH="${YETO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Fail before learners spend time connecting if this Python/device cannot run
# the exact production backend.  Rust performs its own framed 1x1 probe too.
"${ISO_WORKER_PYTHON}" - "${ISO_WORKER_DEVICES}" <<'PY'
import sys
import torch
import yeto.iso_worker  # noqa: F401

devices = sys.argv[1].split(",")
if not devices or any(not device for device in devices):
    raise RuntimeError("ISO_WORKER_DEVICES must be a non-empty comma-separated list")
if len(set(devices)) != len(devices):
    raise RuntimeError(f"duplicate Torch SVD worker devices: {devices}")
for raw_device in devices:
    device = torch.device(raw_device)
    probe = torch.ones(1, device=device)
    if probe.item() != 1.0:
        raise RuntimeError(f"bad Torch SVD worker device probe on {device}")
PY

active_pid=''
current_phase=''

unpublish_final_marker() {
  rm -f -- "${FINAL_MARKER}" "${FINAL_MARKER}.tmp"
}

forward_signal() {
  local signal=$1
  [[ -z "${active_pid}" ]] || kill "-${signal}" "${active_pid}" 2>/dev/null || true
  [[ "${current_phase}" != phase2 ]] || unpublish_final_marker
  exit 128
}
trap 'forward_signal HUP' HUP
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM

run_syncer() {
  local phase=$1
  local log_path=$2
  shift 2
  current_phase=${phase}
  printf '[%s] starting %s; log=%s\n' "$(date -u +%FT%TZ)" "${phase}" "${log_path}"
  "$@" >"${log_path}" 2>&1 &
  active_pid=$!
  local rc=0
  wait "${active_pid}" || rc=$?
  active_pid=''
  printf '[%s] %s exited rc=%d\n' "$(date -u +%FT%TZ)" "${phase}" "${rc}"
  return "${rc}"
}

show_failure_tail() {
  local log_path=$1
  printf '%s\n' "--- last 80 lines of ${log_path} ---" >&2
  tail -n 80 -- "${log_path}" >&2 || true
}

# Print "global_step fragment_count" after validating the current V3
# checkpoint framing, exact torch-svd backend id, semantic-layout digest,
# every fragment extent, and EOF.  Keep this parser streaming: a full-model
# checkpoint is far too large to materialize in the supervisor merely to read
# its header and extents.
inspect_checkpoint() {
  "${ISO_WORKER_PYTHON}" - "${1}" <<'PY'
import os
import struct
import sys

path = sys.argv[1]
size = os.path.getsize(path)

def read_exact(handle, count, label):
    value = handle.read(count)
    if len(value) != count:
        raise ValueError(f"{path}: truncated {label}")
    return value

with open(path, "rb") as handle:
    magic = struct.unpack("<I", read_exact(handle, 4, "magic"))[0]
    if magic != 0xD1705A80:
        raise ValueError(f"{path}: expected v3 magic 0xD1705A80, got 0x{magic:08X}")
    backend = read_exact(handle, 1, "backend id")[0]
    if backend != 1:
        raise ValueError(f"{path}: expected torch-svd backend id 1, got {backend}")
    layout_fingerprint = read_exact(handle, 32, "semantic layout fingerprint")
    if layout_fingerprint == bytes(32):
        raise ValueError(f"{path}: semantic layout fingerprint is all zero")
    global_step = struct.unpack("<Q", read_exact(handle, 8, "global step"))[0]
    fragments = struct.unpack("<I", read_exact(handle, 4, "fragment count"))[0]
    if fragments == 0:
        raise ValueError(f"{path}: checkpoint has no fragments")
    for fragment_id in range(fragments):
        _version, numel = struct.unpack(
            "<QQ", read_exact(handle, 16, f"fragment {fragment_id} header")
        )
        if numel == 0:
            raise ValueError(f"{path}: fragment {fragment_id} is empty")
        payload_bytes = numel * 8  # f32 params plus f32 momentum
        if handle.tell() + payload_bytes > size:
            raise ValueError(f"{path}: truncated fragment {fragment_id} payload")
        handle.seek(payload_bytes, os.SEEK_CUR)
    ledger_count = struct.unpack("<I", read_exact(handle, 4, "ledger count"))[0]
    ledger_bytes = ledger_count * 28
    if handle.tell() + ledger_bytes != size:
        raise ValueError(
            f"{path}: malformed ledger/trailing bytes at offset {handle.tell()} of {size}"
        )
    handle.seek(ledger_bytes, os.SEEK_CUR)
    if handle.tell() != size:
        raise ValueError(f"{path}: checkpoint length changed while inspecting")

print(global_step, fragments)
PY
}

COMMON_ARGS=(
  --port "${PORT}"
  --learners "${NUM_LEARNERS}"
  --grace-ms "${GRACE_MS}"
  --grace-gamma "${GRACE_GAMMA}"
  --grace-tau "${GRACE_TAU}"
  --quorum-timeout-s "${QUORUM_TIMEOUT_S}"
  --delta-correction "${DELTA_CORRECTION}"
  --outer-lr "${OUTER_LR}"
  --outer-momentum "${OUTER_MOMENTUM}"
  --iso-backend torch-svd
  --iso-worker-python "${ISO_WORKER_PYTHON}"
  --iso-worker-devices "${ISO_WORKER_DEVICES}"
  --iso-worker-queue-capacity "${ISO_WORKER_QUEUE_CAPACITY}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --event-tape "${EVENT_TAPE}"
)

PHASE1_ARGS=(
  "${COMMON_ARGS[@]}"
  --quorum "${NUM_LEARNERS}"
  --pipeline "${PHASE1_PIPELINE}"
  --sync-interval-steps "${SYNC_INTERVAL_STEPS}"
  --total-steps "${PHASE1_TOTAL_STEPS}"
  --learner-budget-steps "${LOCAL_BUDGET_STEPS}"
)

if run_syncer phase1 "${PHASE1_LOG}" "${SYNCER_BIN}" "${PHASE1_ARGS[@]}"; then
  :
else
  rc=$?
  show_failure_tail "${PHASE1_LOG}"
  die "phase 1 exited unexpectedly (rc=${rc}); terminal consolidation was not started"
fi
grep -Fq 'learner-budget cutoff checkpoint written' "${PHASE1_LOG}" || {
  show_failure_tail "${PHASE1_LOG}"
  die "phase 1 returned zero without the learner-budget cutoff event"
}
[[ -s "${CHECKPOINT_PATH}" ]] || die "phase 1 produced no non-empty cutoff checkpoint"
[[ ! -e "${FINAL_MARKER}" ]] || die "cutoff checkpoint was incorrectly marked final"

checkpoint_info=$(inspect_checkpoint "${CHECKPOINT_PATH}") || die "cutoff checkpoint validation failed"
read -r cutoff_step fragment_count extra <<<"${checkpoint_info}"
[[ -z "${extra:-}" && "${cutoff_step}" =~ ^[0-9]+$ && "${fragment_count}" =~ ^[0-9]+$ ]] || die "invalid checkpoint inspection result: ${checkpoint_info}"
((fragment_count > 0)) || die "cutoff checkpoint has no fragments"
terminal_step=$((cutoff_step + fragment_count))
((terminal_step > cutoff_step)) || die "terminal total-step overflow"

printf 'Validated cutoff: local_budget=%d learners=%d global_step=%d fragments=%d; restarting at total_steps=%d\n' \
  "${LOCAL_BUDGET_STEPS}" "${NUM_LEARNERS}" "${cutoff_step}" \
  "${fragment_count}" "${terminal_step}"

PHASE2_ARGS=(
  "${COMMON_ARGS[@]}"
  --quorum "${NUM_LEARNERS}"
  --pipeline "${PHASE2_PIPELINE}"
  --sync-interval-steps 0
  --total-steps "${terminal_step}"
  --resume
  --mark-final-checkpoint
)

if run_syncer phase2 "${PHASE2_LOG}" "${SYNCER_BIN}" "${PHASE2_ARGS[@]}"; then
  :
else
  rc=$?
  unpublish_final_marker
  show_failure_tail "${PHASE2_LOG}"
  die "phase 2 exited unexpectedly (rc=${rc}); removed any final marker"
fi

validation_failed=0
grep -Fq 'all learners acknowledged final cut' "${PHASE2_LOG}" || validation_failed=1
grep -Fq "training complete after ${terminal_step} outer steps" "${PHASE2_LOG}" || validation_failed=1
[[ -s "${CHECKPOINT_PATH}" && -s "${FINAL_MARKER}" ]] || validation_failed=1

final_info=$(inspect_checkpoint "${CHECKPOINT_PATH}") || validation_failed=1
if ((validation_failed == 0)); then
  read -r final_step final_fragments final_extra <<<"${final_info}"
  [[ -z "${final_extra:-}" && "${final_step}" == "${terminal_step}" && "${final_fragments}" == "${fragment_count}" ]] || validation_failed=1
fi

if ((validation_failed == 0)); then
  "${ISO_WORKER_PYTHON}" - "${FINAL_MARKER}" "${terminal_step}" \
    "${EVENT_TAPE}" "${cutoff_step}" "${fragment_count}" \
    "${NUM_LEARNERS}" "${LOCAL_BUDGET_STEPS}" <<'PY' || validation_failed=1
import json
import re
import sys

marker_path, terminal_raw, tape_path, cutoff_raw, fragments_raw, learners_raw, budget_raw = sys.argv[1:]
terminal = int(terminal_raw)
cutoff = int(cutoff_raw)
fragments = int(fragments_raw)
learners = int(learners_raw)
budget = int(budget_raw)

marker = open(marker_path, "rb").read()
expected_marker = f"YETO_FINAL_V1\nglobal_step={terminal}\n".encode()
if marker != expected_marker:
    raise ValueError(f"{marker_path}: malformed or stale final marker")

terminal_records = []
with open(tape_path, "r", encoding="utf-8") as tape:
    for line_number, line in enumerate(tape, 1):
        try:
            record = json.loads(line)
        except Exception as exc:
            raise ValueError(f"{tape_path}:{line_number}: invalid JSON: {exc}") from exc
        if cutoff < int(record.get("step", -1)) <= terminal:
            terminal_records.append(record)

if len(terminal_records) != fragments:
    raise ValueError(
        f"terminal tape has {len(terminal_records)} records, expected {fragments}"
    )
if {int(item["step"]) for item in terminal_records} != set(range(cutoff + 1, terminal + 1)):
    raise ValueError("terminal tape does not contain exactly the consolidation step range")
if {int(item["fragment"]) for item in terminal_records} != set(range(fragments)):
    raise ValueError("terminal consolidation did not cover every fragment exactly once")

expected_ids = set(range(learners))
for record in terminal_records:
    responders = record.get("responders", [])
    if {int(item["id"]) for item in responders} != expected_ids:
        raise ValueError(f"step {record['step']} did not include all {learners} learners")
    if any(int(item["c_steps"]) != budget for item in responders):
        raise ValueError(f"step {record['step']} used a non-budget learner contribution")
    if any(int(item["c_tokens"]) <= 0 for item in responders):
        raise ValueError(f"step {record['step']} used a non-positive token count")
PY
fi

if ((validation_failed != 0)); then
  unpublish_final_marker
  show_failure_tail "${PHASE2_LOG}"
  die "terminal checkpoint/finalization validation failed; removed final marker"
fi

current_phase=complete
printf 'Miles value syncer complete: checkpoint=%s marker=%s cutoff_step=%d final_step=%d fragments=%d learners=%d budget=%d\n' \
  "${CHECKPOINT_PATH}" "${FINAL_MARKER}" "${cutoff_step}" \
  "${terminal_step}" "${fragment_count}" "${NUM_LEARNERS}" \
  "${LOCAL_BUDGET_STEPS}"
