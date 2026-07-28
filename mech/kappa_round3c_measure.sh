#!/usr/bin/env bash
set -euo pipefail

MECH_CPU_ROOT=/home/c/yeto-mechR3-20260727
MECH_PYTHON="$MECH_CPU_ROOT/.venv/bin/python"
MECH_OUTPUT_ROOT="$MECH_CPU_ROOT/output/round3c"
MECH_SPECTRUM_ROOT="$MECH_OUTPUT_ROOT/spectrum"
MECH_OUTPUT="$MECH_OUTPUT_ROOT/kappa-round3c-results.json"
MECH_ANALYZER="$MECH_CPU_ROOT/control/kappa_round3c_measure.py"
MECH_PROTOCOL="$MECH_CPU_ROOT/control/kappa-zeroshot-round3c-protocol.md"

test "$(sha256sum "$MECH_ANALYZER" | awk '{print $1}')" = e179fd64d38ea44ead60da7643cb11efb6ce7641b1eb04423731ccadbd3fa8e6
test "$(sha256sum "$MECH_PROTOCOL" | awk '{print $1}')" = 7e61343200d5248e24bb35e25c09c6499c9076f8c44f251185f5666c932ef2da
test "$(find "$MECH_CPU_ROOT/trajectory" -type f -name 'state_after_step_*.ckpt' | wc -l)" -eq 8
test "$(find "$MECH_SPECTRUM_ROOT" -type f -name '*.json' | wc -l)" -eq 8
test ! -e "$MECH_OUTPUT"

find "$MECH_CPU_ROOT/trajectory" "$MECH_SPECTRUM_ROOT" \
  -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum >"$MECH_OUTPUT_ROOT/round3c-raw-inputs.sha256"
sha256sum \
  "$MECH_PROTOCOL" \
  "$MECH_ANALYZER" \
  "$MECH_CPU_ROOT/control/kappa_round2_measure.py" \
  "$MECH_CPU_ROOT/control/kappa-round2-results.json" \
  "$MECH_CPU_ROOT/repo/mech/lane-e/checkpoint_spectrum_probe.py" \
  "$MECH_CPU_ROOT/inputs/eval.jsonl" \
  >"$MECH_OUTPUT_ROOT/round3c-provenance.sha256"
sha256sum \
  "$MECH_OUTPUT_ROOT/round3c-raw-inputs.sha256" \
  "$MECH_OUTPUT_ROOT/round3c-provenance.sha256" \
  >"$MECH_OUTPUT_ROOT/round3c-raw-seal.sha256"

set +e
"$MECH_PYTHON" "$MECH_ANALYZER" \
  --protocol "$MECH_PROTOCOL" \
  --round2-analyzer "$MECH_CPU_ROOT/control/kappa_round2_measure.py" \
  --round2-result "$MECH_CPU_ROOT/control/kappa-round2-results.json" \
  --loss-root /home/c/h200-evac/n1/yeto-results-v8 \
  --loss-root /home/c/h200-evac/n2/yeto-results-v8 \
  --trajectory-root "$MECH_CPU_ROOT/trajectory" \
  --spectrum-root "$MECH_SPECTRUM_ROOT" \
  --spectrum-adapter "$MECH_CPU_ROOT/repo/mech/lane-e/checkpoint_spectrum_probe.py" \
  --model-root "$MECH_CPU_ROOT/inputs/model" \
  --eval-data "$MECH_CPU_ROOT/inputs/eval.jsonl" \
  --output "$MECH_OUTPUT" \
  | tee "$MECH_OUTPUT_ROOT/round3c-measurement-stdout.log"
MECH_STATUS=${PIPESTATUS[0]}
set -e

test -f "$MECH_OUTPUT"
sha256sum "$MECH_OUTPUT" >"$MECH_OUTPUT.sha256"
exit "$MECH_STATUS"
