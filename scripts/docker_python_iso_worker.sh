#!/usr/bin/env bash
set -euo pipefail

# Diagnostic helper only. Killing `docker run` does not prove that Docker
# killed its daemon-owned CUDA container, so this wrapper refuses persistent
# pool-worker launches. Production runs the syncer inside miles_node and uses
# that environment's direct python3 executable.
YETO_ROOT=${YETO_ROOT:?set YETO_ROOT to the versioned Yeto checkout mounted under /data}
[[ "${YETO_ROOT}" == /data/* ]] || {
  printf 'docker_python_iso_worker.sh: YETO_ROOT must be under /data, got %q\n' "${YETO_ROOT}" >&2
  exit 2
}
[[ -f "${YETO_ROOT}/yeto/iso_worker.py" ]] || {
  printf 'docker_python_iso_worker.sh: missing %s/yeto/iso_worker.py\n' "${YETO_ROOT}" >&2
  exit 2
}
if [[ " $* " == *" -m yeto.iso_worker "* ]]; then
  printf '%s\n' \
    'docker_python_iso_worker.sh: refusing persistent worker launch; run the syncer inside miles_node with ISO_WORKER_PYTHON=python3 so timeout cancellation owns the actual Python PID' >&2
  exit 2
fi

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
