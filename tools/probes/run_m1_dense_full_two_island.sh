#!/usr/bin/env bash

# Host-side entrypoint. The manifest builder prints the required SHA256; pass
# it back explicitly so replacing the manifest cannot silently change a run.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 MANIFEST EXPECTED_SHA256" >&2
  exit 64
fi

manifest=$1
expected_sha256=$2
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)

exec python3 "${script_dir}/m1_dense_full_direct_launch.py" launch \
  --manifest "${manifest}" \
  --expected-sha256 "${expected_sha256}"
