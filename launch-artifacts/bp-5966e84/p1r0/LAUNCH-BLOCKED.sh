#!/bin/bash
set -euo pipefail

cd /private/tmp/yeto-bp-integrator
export CLOUDSDK_CONFIG=/private/tmp/yeto-gcloud-admin-codex
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.

/Users/shou/yeto/.venv/bin/python3 \
  /private/tmp/yeto-bp-integrator/launch-artifacts/bp-5966e84/p1r0/validate_wave_plan.py

printf '%s\n' \
  'P1-R0 LAUNCH REFUSED: P0b final replay hashes, amendment-native roster/run IDs,' \
  'and the reviewed parallel executor/partial-manifest/aggregator are not yet bound.' >&2
exit 64
