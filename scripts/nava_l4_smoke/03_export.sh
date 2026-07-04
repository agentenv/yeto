#!/usr/bin/env bash
# Export the smoke NAVA adapter/merged checkpoint from the syncer checkpoint.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${EXPORT_FORMAT:=both}"
: "${EXPORT_DEVICE:=cpu}"

require_file "${SYNC_CKPT}" "syncer checkpoint"
require_file "${SMOKE_CONFIG}" "smoke config"
require_file "${SMOKE_BASE_CKPT}" "tiny base checkpoint"

# Export does not take --nava-assets-dir, so patch the copied smoke config to
# point at the local asset cache before reconstructing the NAVA pipeline.
SMOKE_CONFIG="${SMOKE_CONFIG}" NAVA_ASSETS_DIR="${NAVA_ASSETS_DIR}" python3 - <<'PY'
import os
from pathlib import Path

import yaml

path = Path(os.environ["SMOKE_CONFIG"])
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
cfg.setdefault("model", {})["ckpt_dir"] = os.environ["NAVA_ASSETS_DIR"]
cfg.setdefault("model", {})["audio_vae_ckpt_dir"] = os.environ["NAVA_ASSETS_DIR"]
path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

mkdir -p "${EXPORT_DIR}"
python3 -m yeto.export \
  --task nava \
  --checkpoint "${SYNC_CKPT}" \
  --nava-root "${NAVA_ROOT}" \
  --nava-config "${SMOKE_CONFIG}" \
  --base-ckpt "${SMOKE_BASE_CKPT}" \
  --fragments "${FRAGMENTS}" \
  --output-dir "${EXPORT_DIR}" \
  --device "${EXPORT_DEVICE}" \
  --format "${EXPORT_FORMAT}" \
  --nava-lora-targets "${LORA_TARGETS}"
