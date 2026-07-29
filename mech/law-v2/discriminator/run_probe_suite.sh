#!/usr/bin/env bash
set -euo pipefail

DISCRIM_ROOT=/mnt/nvme1/yeto-mech-discriminator-20260728
DISCRIM_MECHR3=/home/c/yeto-mechR3-20260727
DISCRIM_REPO="$DISCRIM_MECHR3/repo"
DISCRIM_PYTHON="$DISCRIM_MECHR3/.venv/bin/python"
DISCRIM_ADAPTER="$DISCRIM_REPO/mech/law-v2/discriminator/checkpoint_lambda_probe.py"
DISCRIM_VALIDATOR="$DISCRIM_REPO/mech/law-v2/discriminator/validate_probe_equivalence.py"
DISCRIM_PROTOCOL="$DISCRIM_ROOT/control/protocol.md"
DISCRIM_PAIRS="$DISCRIM_ROOT/control/probe-pairs.tsv"
DISCRIM_MODEL="$DISCRIM_MECHR3/inputs/model"
DISCRIM_EVAL="$DISCRIM_MECHR3/inputs/eval.jsonl"
DISCRIM_OUTPUT="$DISCRIM_ROOT/output"
DISCRIM_LOGS="$DISCRIM_ROOT/logs"
DISCRIM_CACHE="$DISCRIM_MECHR3/scratch/round3/hf-datasets-cache"
DISCRIM_TMP="$DISCRIM_ROOT/tmp"

test "$(hostname)" = dev16
test "$(id -un)" = c
test -e "$DISCRIM_ROOT/checkpoints/archive-extraction.complete"
test "$(sha256sum "$DISCRIM_PROTOCOL" | cut -d' ' -f1)" = cd85592d2594bd513b72ac4ad1d043c795d3f612a3089fdd6cb53e45997fd9c2
test "$(sha256sum "$DISCRIM_ADAPTER" | cut -d' ' -f1)" = fd75968eb091b3282e79eb8ad911d82c07b14938c119411e14ccead308f907b1
test "$(sha256sum "$DISCRIM_VALIDATOR" | cut -d' ' -f1)" = 4be1ec7a5acbd2664cc0a7e900dbfd23178281abdf8eaabe0d370090ed12a876
test "$(sha256sum "$DISCRIM_PAIRS" | cut -d' ' -f1)" = 4a579b33affbc299be486d0aa7cece33f4210c4b34c6d7417e61c40cbb705267
test "$(sha256sum "$DISCRIM_EVAL" | cut -d' ' -f1)" = 533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc
test "$(sha256sum "$DISCRIM_MODEL/model-files.sha256" | cut -d' ' -f1)" = 43f9494fad3335a9237f7a3093ae1401b7c4d3164c7486542070e2cc04837132
test -d "$DISCRIM_CACHE"
test -w "$DISCRIM_CACHE"
test -d "$DISCRIM_TMP"
test -w "$DISCRIM_TMP"
test ! -e "$DISCRIM_OUTPUT/validation"
test ! -e "$DISCRIM_OUTPUT/spectrum"
test ! -e "$DISCRIM_LOGS/validation"
test ! -e "$DISCRIM_LOGS/spectrum"

export HF_DATASETS_CACHE="$DISCRIM_CACHE"
export TMPDIR="$DISCRIM_TMP"
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80

declare -A DISCRIM_CHECKED_HASHES=()

check_checkpoint() {
  local discrim_path=$1
  local discrim_expected=$2
  local discrim_actual
  test -f "$discrim_path"
  test ! -L "$discrim_path"
  if [[ -z ${DISCRIM_CHECKED_HASHES[$discrim_path]+x} ]]; then
    discrim_actual=$(sha256sum "$discrim_path" | cut -d' ' -f1)
    test "$discrim_actual" = "$discrim_expected"
    DISCRIM_CHECKED_HASHES[$discrim_path]=$discrim_actual
  else
    test "${DISCRIM_CHECKED_HASHES[$discrim_path]}" = "$discrim_expected"
  fi
}

DISCRIM_HEADER=$'pair_id\trole\tsource\tconvention\tmomentum_mu\ttraining_seed\tcheckpoint_age\tmomentum_checkpoint\tmomentum_sha256\tcontrol_checkpoint\tcontrol_sha256'
{
  IFS= read -r discrim_observed_header
  test "$discrim_observed_header" = "$DISCRIM_HEADER"
  while IFS=$'\t' read -r discrim_pair discrim_role discrim_source \
    discrim_convention discrim_mu discrim_training_seed discrim_age \
    discrim_momentum discrim_momentum_sha discrim_control discrim_control_sha; do
    test -n "$discrim_pair"
    check_checkpoint "$discrim_momentum" "$discrim_momentum_sha"
    check_checkpoint "$discrim_control" "$discrim_control_sha"
  done
} < "$DISCRIM_PAIRS"

mkdir -p "$DISCRIM_OUTPUT/validation" "$DISCRIM_LOGS/validation"
DISCRIM_VALIDATION_RESULT="$DISCRIM_OUTPUT/validation/default-mode.json"
"$DISCRIM_PYTHON" "$DISCRIM_ADAPTER" \
  --checkpoint "$DISCRIM_MECHR3/trajectory/mu0/state_after_step_00000080.ckpt" \
  --model "$DISCRIM_MODEL" \
  --data "$DISCRIM_EVAL" \
  --output "$DISCRIM_VALIDATION_RESULT" \
  --device cpu \
  --threads 80 \
  --fragments 4 \
  --fragment-pattern binpack \
  --seq-len 128 \
  --train-on assistant \
  --loss-function cross_entropy \
  --probe-panels 4 \
  --probe-batch-size 1 \
  --probe-max-rows 128 \
  --block-steps 4 \
  --seed 20260727 \
  >"$DISCRIM_LOGS/validation/default-mode.log" 2>&1

"$DISCRIM_PYTHON" "$DISCRIM_VALIDATOR" \
  --lane-e "$DISCRIM_MECHR3/output/round3c/spectrum/mu0/age_20_step_00000080.json" \
  --extension "$DISCRIM_VALIDATION_RESULT" \
  --output "$DISCRIM_OUTPUT/validation/equivalence.json" \
  >"$DISCRIM_LOGS/validation/equivalence.log" 2>&1

mkdir -p "$DISCRIM_OUTPUT/spectrum" "$DISCRIM_LOGS/spectrum"
{
  IFS= read -r discrim_observed_header
  test "$discrim_observed_header" = "$DISCRIM_HEADER"
  while IFS=$'\t' read -r discrim_pair discrim_role discrim_source \
    discrim_convention discrim_mu discrim_training_seed discrim_age \
    discrim_momentum discrim_momentum_sha discrim_control discrim_control_sha; do
    for discrim_seed in 20260727 20260728; do
      discrim_pair_output="$DISCRIM_OUTPUT/spectrum/$discrim_pair/seed_$discrim_seed"
      discrim_pair_logs="$DISCRIM_LOGS/spectrum/$discrim_pair/seed_$discrim_seed"
      mkdir -p "$discrim_pair_output" "$discrim_pair_logs"
      discrim_momentum_output="$discrim_pair_output/momentum.json"
      discrim_control_output="$discrim_pair_output/control.json"
      test ! -e "$discrim_momentum_output"
      test ! -e "$discrim_control_output"
      printf 'PROBE_START pair=%s role=%s source=%s seed=%s utc=%s\n' \
        "$discrim_pair" "$discrim_role" "$discrim_source" "$discrim_seed" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      "$DISCRIM_PYTHON" "$DISCRIM_ADAPTER" \
        --checkpoint "$discrim_momentum" \
        --model "$DISCRIM_MODEL" \
        --data "$DISCRIM_EVAL" \
        --output "$discrim_momentum_output" \
        --device cpu \
        --threads 80 \
        --fragments 4 \
        --fragment-pattern binpack \
        --seq-len 128 \
        --train-on assistant \
        --loss-function cross_entropy \
        --probe-panels 4 \
        --probe-batch-size 1 \
        --probe-max-rows 128 \
        --block-steps 4 \
        --seed "$discrim_seed" \
        --randomized-start \
        >"$discrim_pair_logs/momentum.log" 2>&1 &
      discrim_momentum_pid=$!
      "$DISCRIM_PYTHON" "$DISCRIM_ADAPTER" \
        --checkpoint "$discrim_control" \
        --model "$DISCRIM_MODEL" \
        --data "$DISCRIM_EVAL" \
        --output "$discrim_control_output" \
        --device cpu \
        --threads 80 \
        --fragments 4 \
        --fragment-pattern binpack \
        --seq-len 128 \
        --train-on assistant \
        --loss-function cross_entropy \
        --probe-panels 4 \
        --probe-batch-size 1 \
        --probe-max-rows 128 \
        --block-steps 4 \
        --seed "$discrim_seed" \
        --randomized-start \
        >"$discrim_pair_logs/control.log" 2>&1 &
      discrim_control_pid=$!
      discrim_momentum_rc=0
      discrim_control_rc=0
      wait "$discrim_momentum_pid" || discrim_momentum_rc=$?
      wait "$discrim_control_pid" || discrim_control_rc=$?
      if [[ $discrim_momentum_rc -ne 0 || $discrim_control_rc -ne 0 ]]; then
        printf 'PROBE_FAILED pair=%s seed=%s momentum_rc=%s control_rc=%s\n' \
          "$discrim_pair" "$discrim_seed" "$discrim_momentum_rc" "$discrim_control_rc" >&2
        exit 1
      fi
      test -s "$discrim_momentum_output"
      test -s "$discrim_control_output"
      printf 'PROBE_COMPLETE pair=%s seed=%s momentum_sha256=%s control_sha256=%s utc=%s\n' \
        "$discrim_pair" "$discrim_seed" \
        "$(sha256sum "$discrim_momentum_output" | cut -d' ' -f1)" \
        "$(sha256sum "$discrim_control_output" | cut -d' ' -f1)" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    done
  done
} < "$DISCRIM_PAIRS"

DISCRIM_RESULT_COUNT=$(find "$DISCRIM_OUTPUT/spectrum" -type f -name '*.json' | wc -l)
test "$DISCRIM_RESULT_COUNT" -eq 36
find "$DISCRIM_OUTPUT/spectrum" -type f -name '*.json' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum > "$DISCRIM_OUTPUT/spectrum-files.sha256"
touch "$DISCRIM_OUTPUT/probe-suite.complete"
printf 'PROBE_SUITE_COMPLETE results=%s utc=%s\n' \
  "$DISCRIM_RESULT_COUNT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
