#!/usr/bin/env bash
# CPU-only identity check of a pushed yeto miles-ports image, without docker:
# streams `crane export` of the image and extracts only the identity files,
# then asserts the fork SHAs, the manifest and the sglang version.  Also
# greps the Megatron-LM checkout for the mHC spec slot (M4 D2 risk).
#
#   scripts/check_miles_ports_image.sh <image@sha256:...> <miles_sha> <sglang_sha> <sglang_version> [out_dir]
#
# Registry auth: crane reads ~/.docker/config.json; nothing is printed.
set -euo pipefail
IMAGE=$1; MILES_SHA=$2; SGLANG_SHA=$3; SGLANG_VERSION=$4
OUT=${5:-$(mktemp -d /tmp/img-check-XXXXXX)}
CRANE=${CRANE:-crane}
mkdir -p "$OUT/x"
# One pass over the image (no local docker, no full extraction on disk).
"$CRANE" export "$IMAGE" - | tar -xf - -C "$OUT/x" --wildcards \
  'root/miles/.git/HEAD' 'root/miles/.git/refs/heads/*' 'root/miles/.git/packed-refs' \
  'root/miles/.git/config' \
  'root/miles/miles_plugins/models/qwen3_8_next/*.py' \
  'root/miles/miles/utils/workers/ray_worker_manager.py' \
  'root/miles/miles/utils/workers/worker_handle.py' \
  'root/miles/miles/backends/training_utils/weight_update/protocols/broadcast.py' \
  'sgl-workspace/sglang/.git/HEAD' 'sgl-workspace/sglang/.git/refs/heads/*' 'sgl-workspace/sglang/.git/packed-refs' \
  'sgl-workspace/sglang/.git/config' \
  'sgl-workspace/sglang/python/sglang/_version.py' \
  'sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py' \
  'opt/yeto/image-manifest.json' \
  'opt/sglang/lib/python3.12/site-packages/__editable__.sglang-*.pth' \
  'opt/sglang/lib/python3.12/site-packages/sglang-*.dist-info/METADATA' \
  'root/Megatron-LM/.git/HEAD' 'root/Megatron-LM/.git/refs/heads/*' \
  'root/Megatron-LM/megatron/core/transformer/transformer_block.py' \
  2>"$OUT/tar.err" || true   # tar exits 2 when a wildcard matches nothing
cd "$OUT/x"
resolve() { local ref; ref=$(sed -n 's/^ref: //p' "$1/.git/HEAD"); if [ -n "$ref" ]; then cat "$1/.git/$ref"; else cat "$1/.git/HEAD"; fi; }
fail=0
chk() { if [ "$2" = "$3" ]; then echo "PASS $1: $2"; else echo "FAIL $1: got $2 want $3"; fail=1; fi; }
chk miles_head "$(resolve root/miles)" "$MILES_SHA"
chk sglang_head "$(resolve sgl-workspace/sglang)" "$SGLANG_SHA"
chk sglang_version "$(sed -n 's/^__version__ = version = .\([^"'"'"']*\).*/\1/p' sgl-workspace/sglang/python/sglang/_version.py)" "$SGLANG_VERSION"
chk sglang_dist_info "$(ls opt/sglang/lib/python3.12/site-packages/ | grep -c "^sglang-${SGLANG_VERSION//+/_}.*dist-info$\|^sglang-${SGLANG_VERSION}.dist-info$")" 1
chk manifest_miles "$(python3 -c 'import json;print(json.load(open("opt/yeto/image-manifest.json"))["miles"]["commit"])')" "$MILES_SHA"
chk manifest_sglang "$(python3 -c 'import json;print(json.load(open("opt/yeto/image-manifest.json"))["sglang"]["commit"])')" "$SGLANG_SHA"
chk miles_origin "$(git config -f root/miles/.git/config remote.origin.url)" "https://github.com/michaellchung/miles"
chk sglang_origin "$(git config -f sgl-workspace/sglang/.git/config remote.origin.url)" "https://github.com/michaellchung/sglang"
chk m3_lora_plugin_present "$(test -f root/miles/miles_plugins/models/qwen3_8_next/lora.py && echo yes)" yes
chk qwen4exp_lora_hooks "$(grep -c supported_lora_modules sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py | awk '{print ($1>0)?"yes":"no"}')" yes
chk a27_workers_lost "$(grep -c workers_lost root/miles/miles/utils/workers/ray_worker_manager.py | awk '{print ($1>0)?"yes":"no"}')" yes
chk a27b_external_failure_error "$(grep -c ExternalFailureError root/miles/miles/utils/workers/worker_handle.py | awk '{print ($1>0)?"yes":"no"}')" yes
echo "megatron_head: $(resolve root/Megatron-LM 2>/dev/null || echo unknown)"
chk d2_hc_head_contraction "$(grep -c hc_head_contraction root/Megatron-LM/megatron/core/transformer/transformer_block.py | awk '{print ($1>0)?"yes":"no"}')" yes
echo "out=$OUT"; exit $fail
