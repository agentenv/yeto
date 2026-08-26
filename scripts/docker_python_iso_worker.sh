#!/usr/bin/env bash
set -euo pipefail

# Keep the large Rust syncer state in a host-systemd process while providing
# its persistent Torch/CUDA Iso worker from the existing Miles image.  Rust
# appends: -m yeto.iso_worker --device <device>.
exec docker run --rm -i \
  --network host \
  --ipc host \
  --gpus device=0 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e PYTHONPATH=/data/yeto-isoloco-20260825 \
  -v /data:/data \
  -v /root/.local:/root/.local \
  --entrypoint python3 \
  miles-sao:local \
  "$@"
