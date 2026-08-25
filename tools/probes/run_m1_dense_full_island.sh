#!/usr/bin/env bash

# Runs as PID 1 inside exactly one digest-pinned Miles container.
set -euo pipefail
umask 077

if [ "$#" -ne 3 ]; then
  echo "usage: $0 MANIFEST EXPECTED_SHA256 ISLAND_ID" >&2
  exit 64
fi

manifest=$1
expected_sha256=$2
island_id=$3
tool=/root/yeto/tools/probes/m1_dense_full_direct_launch.py

test -f "${manifest}"
test ! -L "${manifest}"
test -x "${tool}" || test -f "${tool}"
if pgrep -f '(^|/)(raylet|gcs_server)( |$)' >/dev/null; then
  echo "refusing to share an existing Ray runtime" >&2
  exit 1
fi

read -r ray_gcs ray_dashboard ray_client evidence_dir < <(
  python3 "${tool}" island-runtime \
    --manifest "${manifest}" \
    --expected-sha256 "${expected_sha256}" \
    --island-id "${island_id}"
)

test "${evidence_dir}" = /evidence
test -d "${evidence_dir}"
test ! -L "${evidence_dir}"
test ! -e "${evidence_dir}/ray"
test ! -e "${evidence_dir}/learner.log"
printf '%s  %s\n' "${expected_sha256}" "${manifest}" \
  > "${evidence_dir}/manifest.sha256"
sha256sum --check "${evidence_dir}/manifest.sha256"

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

ray start \
  --head \
  --node-ip-address=127.0.0.1 \
  --port="${ray_gcs}" \
  --num-gpus=4 \
  --disable-usage-stats \
  --dashboard-host=127.0.0.1 \
  --dashboard-port="${ray_dashboard}" \
  --ray-client-server-port="${ray_client}" \
  --temp-dir="${evidence_dir}/ray" \
  > "${evidence_dir}/ray-start.log" 2>&1

export RAY_ADDRESS="127.0.0.1:${ray_gcs}"
printf '{"island_id":%s,"manifest_sha256":"%s","schema":"yeto-m1-dense-full-island-started-v1"}\n' \
  "${island_id}" "${expected_sha256}" > "${evidence_dir}/container-started.json"

set +e
python3 "${tool}" exec-island \
  --manifest "${manifest}" \
  --expected-sha256 "${expected_sha256}" \
  --island-id "${island_id}" \
  2>&1 | tee "${evidence_dir}/learner.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${status}" > "${evidence_dir}/learner.exit"
exit "${status}"
