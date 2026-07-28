#!/usr/bin/env bash
set -euo pipefail

MECH_CPU_ROOT=/home/c/yeto-mechR3-20260727
MECH_REPO="$MECH_CPU_ROOT/repo"
MECH_PYTHON="$MECH_CPU_ROOT/.venv/bin/python"
MECH_ADAPTER="$MECH_REPO/mech/lane-e/checkpoint_spectrum_probe.py"
MECH_MODEL="$MECH_CPU_ROOT/inputs/model"
MECH_EVAL="$MECH_CPU_ROOT/inputs/eval.jsonl"
MECH_PROTOCOL="$MECH_CPU_ROOT/control/kappa-zeroshot-round3c-protocol.md"
MECH_OUTPUT_ROOT="$MECH_CPU_ROOT/output/round3c/spectrum"
MECH_LOG_ROOT="$MECH_CPU_ROOT/logs/round3c/spectrum"
MECH_CACHE="$MECH_CPU_ROOT/scratch/round3c/hf-datasets-cache"
MECH_TMP="$MECH_CPU_ROOT/scratch/round3c/tmp"

test "$(hostname)" = dev16
test "$(id -un)" = c
test "$(sha256sum "$MECH_PROTOCOL" | awk '{print $1}')" = 7e61343200d5248e24bb35e25c09c6499c9076f8c44f251185f5666c932ef2da
test "$(sha256sum "$MECH_ADAPTER" | awk '{print $1}')" = 857c88c2a227c32f983c5d206c48d43f49792cdda2f797db691df1386e46d8bd
test "$(sha256sum "$MECH_EVAL" | awk '{print $1}')" = 533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc
test "$(sha256sum "$MECH_MODEL/model-files.sha256" | awk '{print $1}')" = 43f9494fad3335a9237f7a3093ae1401b7c4d3164c7486542070e2cc04837132
test -d "$MECH_CACHE"
test -w "$MECH_CACHE"
test -d "$MECH_TMP"
test -w "$MECH_TMP"

export HF_DATASETS_CACHE="$MECH_CACHE"
export TMPDIR="$MECH_TMP"
export OMP_NUM_THREADS=80
export MKL_NUM_THREADS=80

test ! -e "$MECH_OUTPUT_ROOT"
test ! -e "$MECH_LOG_ROOT"
mkdir -p "$MECH_OUTPUT_ROOT" "$MECH_LOG_ROOT"

for MECH_ARM in mu0 corrected; do
  mkdir -p "$MECH_OUTPUT_ROOT/$MECH_ARM" "$MECH_LOG_ROOT/$MECH_ARM"
  for MECH_AGE in 5 10 15 20; do
    MECH_STEP=$((4 * MECH_AGE))
    printf -v MECH_CHECKPOINT \
      '%s/trajectory/%s/state_after_step_%08d.ckpt' \
      "$MECH_CPU_ROOT" "$MECH_ARM" "$MECH_STEP"
    printf -v MECH_OUTPUT \
      '%s/%s/age_%02d_step_%08d.json' \
      "$MECH_OUTPUT_ROOT" "$MECH_ARM" "$MECH_AGE" "$MECH_STEP"
    printf -v MECH_LOG \
      '%s/%s/age_%02d_step_%08d.log' \
      "$MECH_LOG_ROOT" "$MECH_ARM" "$MECH_AGE" "$MECH_STEP"
    test -f "$MECH_CHECKPOINT"
    test ! -L "$MECH_CHECKPOINT"
    test ! -e "$MECH_OUTPUT"
    case "$MECH_ARM:$MECH_STEP" in
      mu0:20) MECH_EXPECTED=80abf3a6528f12c2c0f96ebc9c8f0492c6c149ca2d17a3497e81b5f3c6034d1a ;;
      mu0:40) MECH_EXPECTED=f1ce70021c42c5ef6825a4a56a9051b17b02760a95d50d63f0cb89d95cb519af ;;
      mu0:60) MECH_EXPECTED=ee93ffec6499367ca9851b435ddcc14869622cbffd0ba5fe883cf8f2bea4d4cd ;;
      mu0:80) MECH_EXPECTED=1d7d3a3ee4471e8ae55bfd51d2165823397f1627b2f41186ce7b7625d0fde5b0 ;;
      corrected:20) MECH_EXPECTED=b81d16335ceba79e71dd5ed2131c2015476c00e4540bd4238be3317fd50a07eb ;;
      corrected:40) MECH_EXPECTED=7a27f4c5235ba387919b16e6e53e9b7680f75be22c7338943f0c9cba4c6981e2 ;;
      corrected:60) MECH_EXPECTED=7cd0656fe5f269039bf09de398af31c7163977a432f74009097cca511a6e5d74 ;;
      corrected:80) MECH_EXPECTED=9dc8c5e084aa6ed197375cc0cffe4f0b65ad5026ce7834a18a34794fa0e46697 ;;
      *) exit 97 ;;
    esac
    test "$(sha256sum "$MECH_CHECKPOINT" | awk '{print $1}')" = "$MECH_EXPECTED"
    echo "KAPPA_ROUND3C_PROBE_START arm=$MECH_ARM age=$MECH_AGE step=$MECH_STEP utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$MECH_PYTHON" "$MECH_ADAPTER" \
      --checkpoint "$MECH_CHECKPOINT" \
      --model "$MECH_MODEL" \
      --data "$MECH_EVAL" \
      --output "$MECH_OUTPUT" \
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
      >"$MECH_LOG" 2>&1
    test -s "$MECH_OUTPUT"
    echo "KAPPA_ROUND3C_PROBE_COMPLETE arm=$MECH_ARM age=$MECH_AGE step=$MECH_STEP output_sha256=$(sha256sum "$MECH_OUTPUT" | awk '{print $1}') utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
done

MECH_COUNT="$(find "$MECH_OUTPUT_ROOT" -type f -name '*.json' | wc -l)"
test "$MECH_COUNT" -eq 8
find "$MECH_OUTPUT_ROOT" -type f -name '*.json' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum >"$MECH_CPU_ROOT/output/round3c/spectrum-files.sha256"
echo "KAPPA_ROUND3C_PROBES_COMPLETE count=$MECH_COUNT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
