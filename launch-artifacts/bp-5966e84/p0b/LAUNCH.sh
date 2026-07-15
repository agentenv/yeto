#!/bin/bash
set -euo pipefail

cd /private/tmp/yeto-bp-integrator
export CLOUDSDK_CONFIG=/private/tmp/yeto-gcloud-admin-codex
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.

python_bin=/Users/shou/yeto/.venv/bin/python3
state_dir=/tmp/yeto-p0b-state
spec=/private/tmp/yeto-bp-integrator/launch-artifacts/bp-5966e84/p0b/optimizer-harness-p0b.json

test ! -e "$state_dir/bp-p0b-5966e84-20260715a.json"
"$python_bin" /private/tmp/yeto-bp-integrator/launch-artifacts/bp-5966e84/p0b/validate_packet.py
"$python_bin" -m yeto.optimizer_harness --state-dir "$state_dir" \
  launch "$spec" --yes
"$python_bin" -m yeto.optimizer_harness --state-dir "$state_dir" \
  start "$spec"
