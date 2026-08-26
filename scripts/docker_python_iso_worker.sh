#!/usr/bin/env bash
set -euo pipefail

# Keep the large Rust syncer state in a host-systemd process while providing
# its persistent Torch/CUDA Iso worker from the existing Miles image.  Rust
# appends: -m yeto.iso_worker --device <device>.
YETO_ROOT=${YETO_ROOT:?set YETO_ROOT to the versioned Yeto checkout mounted under /data}
[[ "${YETO_ROOT}" == /data/* ]] || {
  printf 'docker_python_iso_worker.sh: YETO_ROOT must be under /data, got %q\n' "${YETO_ROOT}" >&2
  exit 2
}
[[ -f "${YETO_ROOT}/yeto/iso_worker.py" ]] || {
  printf 'docker_python_iso_worker.sh: missing %s/yeto/iso_worker.py\n' "${YETO_ROOT}" >&2
  exit 2
}

exec docker run --rm -i \
  --network host \
  --ipc host \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e PYTHONPATH="${YETO_ROOT}" \
  -v /data:/data \
  -v /root/.local:/root/.local \
  --entrypoint python3 \
  miles-sao:local \
  "$@"
