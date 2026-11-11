#!/usr/bin/env bash
# Build + push the yeto Kimi K3.1 Miles image without a docker daemon: crane
# appends one overlay layer (docker/miles-ports/make_layer.py) to the upstream
# radixark/miles:kimi-k31 base pinned by digest in versions.lock.json.
#
#   docker/miles-k31/build.sh [--push]
#
# The overlay holds (versions.lock.json "decisions"):
#   * the Miles fork branch over /root/miles            (MILES_BRANCH, default k31/f1)
#   * the SGLang fork branch over /sgl-workspace/sglang (SGLANG_BRANCH, default yeto/k31)
#   * peft and pyrage wheels unpacked into the base's site-packages
#   * the lazy-modelopt patch (Megatron-LM) and Megatron-Bridge __init__
#   * /opt/yeto/image-manifest.json and the image-assert lock under /opt/yeto/image
# No TMS preload: the base's torch_memory_saver already carries the fix.
#
# Inputs (env):
#   MILES_SRC / SGLANG_SRC   required: local clones of the forks (fetch the branches first)
#   MILES_BRANCH             Miles fork branch or commit (default k31/f1)
#   SGLANG_BRANCH            SGLang fork branch or commit (default yeto/k31)
#   MILES_ORIGIN / SGLANG_ORIGIN  fork repository URLs written as the checkouts' origin
#                            (default: yeto.rl.adapters.miles.pins MILES_NEXT_REPOSITORY / SGLANG_NEXT_REPOSITORY)
#   SGLANG_TAG               the SGLang release tag the fork descends from (default v0.5.23)
#   MEGATRON_LM_SRC          clone containing the base's Megatron-LM commit (default: shallow fetch into $OUT)
#   PEFT_VERSION / PYRAGE_VERSION  wheels to add (defaults 0.20.0 / 1.4.0)
#   TAG_SUFFIX               appended to the k31-<miles>-<sglang> tag (rebuilds of the same commits)
#   ALLOW_INSTALL_METADATA_DRIFT=1  accept setup.py/requirements* differences between a fork
#                            and the base's checkout (the base's dist-info is kept; the
#                            paths are recorded in image-manifest.json). k31/integ needs it:
#                            its setup.py adds an extras_require entry and requirements.txt
#                            adds pyrage, which this layer ships as a wheel.
#   DEST_REPO                required for --push: target repository, e.g. ghcr.io/<owner>/yeto-miles-k31
#   CRANE                    crane binary (go-containerregistry >= 0.20)
#   OUT                      work directory (default: mktemp)
# Registry auth: crane reads ~/.docker/config.json; nothing is printed.
# Writes: $OUT/{layer.tar,image-manifest.json,commits.json,build-record.json}
set -euo pipefail
cd "$(dirname "$0")/../.."
PUSH=0; [ "${1:-}" = "--push" ] && PUSH=1
CRANE=${CRANE:-crane}
LOCK=docker/miles-k31/versions.lock.json
MILES_SRC=${MILES_SRC:?set MILES_SRC to a local Miles fork clone}
SGLANG_SRC=${SGLANG_SRC:?set SGLANG_SRC to a local SGLang fork clone}
MILES_BRANCH=${MILES_BRANCH:-k31/f1}
SGLANG_BRANCH=${SGLANG_BRANCH:-yeto/k31}
SGLANG_TAG=${SGLANG_TAG:-v0.5.23}
PEFT_VERSION=${PEFT_VERSION:-0.20.0}
PYRAGE_VERSION=${PYRAGE_VERSION:-1.4.0}
OUT=${OUT:-$(mktemp -d /tmp/img-k31-XXXXXX)}
if [ $PUSH = 1 ]; then DEST_REPO=${DEST_REPO:?set DEST_REPO for --push}; else DEST_REPO=${DEST_REPO:-yeto-miles-k31}; fi

lock() { python3 -c 'import json, sys; d = json.load(open(sys.argv[1]));
for k in sys.argv[2].split("."): d = d[k]
print(d)' "$LOCK" "$1"; }
BASE_REPO=$(lock base_image.name | cut -d: -f1)
BASE_INDEX_DIGEST=$(lock base_image.index_digest)
BASE_AMD64_DIGEST=$(lock base_image.manifest_digest)
BASE="$BASE_REPO@$BASE_AMD64_DIGEST"

# Fork repositories (origin of the shipped checkouts), from the pins unless overridden.
# (pins.py is a stdlib-only constants module; run it instead of grepping, so an
# alias such as SGLANG_NEXT_REPOSITORY = SGLANG_K31_REPOSITORY resolves.)
pins=$(python3 - <<'PY'
import runpy
ns = runpy.run_path("yeto/rl/adapters/miles/pins.py")
for k in ("MILES_NEXT_REPOSITORY", "SGLANG_NEXT_REPOSITORY"):
    print(f"{k}={ns[k]}")
PY
)
eval "$pins"
MILES_ORIGIN=${MILES_ORIGIN:-$MILES_NEXT_REPOSITORY}
SGLANG_ORIGIN=${SGLANG_ORIGIN:-$SGLANG_NEXT_REPOSITORY}

# 1. The base as the registry describes it (read-only): manifest, config, and
#    the build args that name the checkouts the base installed. Nothing here is
#    written into the repository: it is re-read on every build and checked
#    against the lock.
"$CRANE" manifest "$BASE" > "$OUT/base-manifest.json"
"$CRANE" config "$BASE" > "$OUT/base-config.json"
base_arg() {  # value of an ARG the base recorded in its build history
  python3 - "$OUT/base-config.json" "$1" <<'PY'
import json, re, sys
cfg = json.load(open(sys.argv[1])); name = sys.argv[2]
seen = []
for h in cfg.get("history", []):
    for m in re.finditer(rf"\b{name}=(\S+)", h.get("created_by", "")):
        seen.append(m.group(1).rstrip('"'))
print(seen[-1] if seen else "")
PY
}
BASE_MILES_COMMIT=$(base_arg MILES_COMMIT)
BASE_SGLANG_COMMIT=$(base_arg SGLANG_COMMIT)
MEGATRON_LM_COMMIT=$(base_arg MEGATRON_COMMIT)
for v in BASE_MILES_COMMIT BASE_SGLANG_COMMIT MEGATRON_LM_COMMIT; do
  [ -n "${!v}" ] || { echo "$v: not in the base image's build history" >&2; exit 1; }
done
for pair in "SGLANG_COMMIT=$BASE_SGLANG_COMMIT" "MILES_COMMIT=$BASE_MILES_COMMIT" "MEGATRON_COMMIT=$MEGATRON_LM_COMMIT"; do
  want=$(lock "base_build_args.${pair%%=*}")
  [ "$want" = "${pair#*=}" ] || { echo "$LOCK says ${pair%%=*}=$want, the base says ${pair#*=}" >&2; exit 1; }
done
# torch_memory_saver and Megatron-Bridge: the commits pip installed from git
TMS_COMMIT=$(grep -o 'torch_memory_saver\.git@[0-9a-f]\{40\}' "$OUT/base-config.json" | tail -1 | cut -d@ -f2)
MEGATRON_BRIDGE_COMMIT=$(grep -o 'Megatron-Bridge\.git@[0-9a-f]\{40\}' "$OUT/base-config.json" | tail -1 | cut -d@ -f2 || true)
if [ -z "$MEGATRON_BRIDGE_COMMIT" ]; then
  MEGATRON_BRIDGE_COMMIT=$(grep -o 'MEGATRON_BRIDGE_COMMIT=[0-9a-f]\{40\}' "$OUT/base-config.json" | tail -1 | cut -d= -f2 || true)
fi
[ -n "$MEGATRON_BRIDGE_COMMIT" ] || { echo "Megatron-Bridge commit: not in the base image's build history; set MEGATRON_BRIDGE_COMMIT" >&2; exit 1; }
want_tms=$(lock commits.torch_memory_saver)
[ "$want_tms" = "$TMS_COMMIT" ] || { echo "$LOCK pins torch_memory_saver $want_tms, the base installed ${TMS_COMMIT:-?}" >&2; exit 1; }

# 2. Fork commits and the pure-Python check (make_layer refuses anything else).
MILES_COMMIT=$(git -C "$MILES_SRC" rev-parse "$MILES_BRANCH^{commit}")
SGLANG_COMMIT=$(git -C "$SGLANG_SRC" rev-parse "$SGLANG_BRANCH^{commit}")
git -C "$MILES_SRC" cat-file -e "$BASE_MILES_COMMIT^{commit}" || { echo "fetch the base Miles commit $BASE_MILES_COMMIT into $MILES_SRC" >&2; exit 1; }
git -C "$SGLANG_SRC" cat-file -e "$BASE_SGLANG_COMMIT^{commit}" || { echo "fetch the base SGLang commit $BASE_SGLANG_COMMIT into $SGLANG_SRC" >&2; exit 1; }

# 3. The base's sglang editable metadata, from the layer that wrote it (found by
#    scanning; no layer number anywhere).
META="$OUT/base-sglang-meta"
if [ ! -d "$META" ]; then
  layer=""
  for digest in $(python3 docker/miles-ports/make_layer.py find-layer \
      --manifest "$OUT/base-manifest.json" --config "$OUT/base-config.json" \
      | python3 -c 'import json, sys; print("\n".join(l["digest"] for l in json.load(sys.stdin)))'); do
    if "$CRANE" blob "$BASE_REPO@$digest" | tar -tzf - 2>/dev/null \
        | python3 docker/miles-ports/make_layer.py check-listing; then
      layer=$digest; break
    fi
  done
  [ -n "$layer" ] || { echo "no base layer holds the sglang editable-install metadata" >&2; exit 1; }
  mkdir -p "$OUT/meta-x"
  "$CRANE" blob "$BASE_REPO@$layer" | tar -xzf - -C "$OUT/meta-x" \
    --wildcards "opt/sglang/lib/python3.12/site-packages/__editable___sglang_*_finder.py" \
    "opt/sglang/lib/python3.12/site-packages/__editable__.sglang-*.pth" \
    "opt/sglang/lib/python3.12/site-packages/sglang-*.dist-info/*"
  mv "$OUT/meta-x/opt/sglang/lib/python3.12/site-packages" "$META"
fi
BASE_SGLANG_VERSION=$(ls "$META" | sed -n 's/^sglang-\(.*\)\.dist-info$/\1/p' | head -1)
[ -n "$BASE_SGLANG_VERSION" ] || { echo "no sglang-*.dist-info in $META" >&2; exit 1; }

# setuptools-scm version the fork would get from a real `pip install -e`:
# v0.5.23-<n>-g<sha7> -> 0.5.24.dev<n>+g<sha7>; a fork without the tag keeps
# the base's public version plus the local part.
if desc=$(git -C "$SGLANG_SRC" describe --tags --long --match "$SGLANG_TAG" "$SGLANG_COMMIT" 2>/dev/null); then
  n=$(echo "$desc" | awk -F- '{print $(NF-1)}')
  base_public=${SGLANG_TAG#v}
  next=$(python3 -c 'import sys; p = sys.argv[1].split("."); p[-1] = str(int(p[-1]) + 1); print(".".join(p))' "$base_public")
  SGLANG_VERSION="${next}.dev${n}+g${SGLANG_COMMIT:0:7}"
else
  SGLANG_VERSION="${BASE_SGLANG_VERSION%%+*}+g${SGLANG_COMMIT:0:7}"
fi
MS=${MILES_COMMIT:0:7}; SS=${SGLANG_COMMIT:0:7}
TAG="$DEST_REPO:k31-$MS-$SS${TAG_SUFFIX:+-$TAG_SUFFIX}"
MTIME=$(git -C "$MILES_SRC" log -1 --format=%ct "$MILES_COMMIT")

# 4. Megatron-LM files patched in the layer start from the base's checkout commit.
MEGATRON_LM_REPOSITORY=https://github.com/radixark/Megatron-LM
if [ -z "${MEGATRON_LM_SRC:-}" ]; then
  MEGATRON_LM_SRC="$OUT/megatron-lm"
  if [ ! -d "$MEGATRON_LM_SRC" ]; then
    git init -q "$MEGATRON_LM_SRC"
    git -C "$MEGATRON_LM_SRC" fetch -q --depth 1 "$MEGATRON_LM_REPOSITORY" "$MEGATRON_LM_COMMIT"
  fi
fi
git -C "$MEGATRON_LM_SRC" cat-file -e "$MEGATRON_LM_COMMIT^{commit}"

# 5. Wheels: peft (pure Python) and pyrage (manylinux abi3, x86_64), no deps.
PIP=${PIP:-python3 -m pip}
mkdir -p "$OUT/wheels"
$PIP download -q --no-deps --only-binary=:all: --dest "$OUT/wheels" "peft==$PEFT_VERSION"
$PIP download -q --no-deps --only-binary=:all: --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
  --python-version 3.12 --abi abi3 --abi cp312 --dest "$OUT/wheels" "pyrage==$PYRAGE_VERSION"
PEFT_WHEEL=$(ls "$OUT/wheels"/peft-"$PEFT_VERSION"-*.whl)
# the base ships an older peft; its dist-info is whited out (lock packages.peft.replaces_base)
PEFT_BASE_VERSION=$(lock packages.peft.replaces_base 2>/dev/null || true)
PYRAGE_WHEEL=$(ls "$OUT/wheels"/pyrage-"$PYRAGE_VERSION"-*x86_64.whl)

# 6. Manifest, commits.json (image-assert) and the layer.
python3 - "$OUT/image-manifest.json" <<PY
import json, sys
json.dump({
  "schema": 2,
  "image": "$DEST_REPO",
  "tag": "k31-$MS-$SS${TAG_SUFFIX:+-$TAG_SUFFIX}",
  "base": {"repository": "$BASE_REPO", "index_digest": "$BASE_INDEX_DIGEST",
           "platform": "linux/amd64", "manifest_digest": "$BASE_AMD64_DIGEST",
           "miles_commit": "$BASE_MILES_COMMIT", "sglang_commit": "$BASE_SGLANG_COMMIT",
           "sglang_version": "$BASE_SGLANG_VERSION"},
  "miles": {"repository": "$MILES_ORIGIN", "branch": "$MILES_BRANCH", "commit": "$MILES_COMMIT",
            "path": "/root/miles", "install": "base's editable install kept (same path)"},
  "sglang": {"repository": "$SGLANG_ORIGIN", "branch": "$SGLANG_BRANCH", "commit": "$SGLANG_COMMIT",
             "path": "/sgl-workspace/sglang", "version": "$SGLANG_VERSION",
             "install": "pip install -e /sgl-workspace/sglang/python[all] --no-deps (base's editable install, same path)"},
  "wheels": [{"name": "peft", "version": "$PEFT_VERSION", "file": "$(basename "$PEFT_WHEEL")",
              "sha256": "$(sha256sum "$PEFT_WHEEL" | cut -d' ' -f1)"},
             {"name": "pyrage", "version": "$PYRAGE_VERSION", "file": "$(basename "$PYRAGE_WHEEL")",
              "sha256": "$(sha256sum "$PYRAGE_WHEEL" | cut -d' ' -f1)"}],
  "pure_python_overlay": True,
  "megatron_bridge": {"commit": "$MEGATRON_BRIDGE_COMMIT",
                      "patches": ["lazy-modelopt (docker/miles-ports/megatron_bridge_init.py)"]},
  "megatron_lm": {"repository": "$MEGATRON_LM_REPOSITORY", "commit": "$MEGATRON_LM_COMMIT", "path": "/root/Megatron-LM",
                  "patches": ["lazy-modelopt (docker/miles-ports/megatron_lm_lazy_modelopt.patch)"]},
  "torch_memory_saver": {"commit": "$TMS_COMMIT", "preload_patch": None},
  "image_assert": {"lock": "docker/miles-k31/versions.lock.json", "path": "/opt/yeto/image"},
  "builder": "docker/miles-k31/build.sh (crane append)",
}, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY
mkdir -p "$OUT/image-assert"
cp "$LOCK" "$OUT/image-assert/versions.lock.json"
python3 - "$OUT/image-assert/commits.json" <<PY
import json, sys
json.dump({"miles": "$MILES_COMMIT", "sglang": "$SGLANG_COMMIT", "megatron": "$MEGATRON_LM_COMMIT",
           "megatron_bridge": "$MEGATRON_BRIDGE_COMMIT", "torch_memory_saver": "$TMS_COMMIT"},
          open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY

python3 docker/miles-ports/make_layer.py \
  --manifest-json "$OUT/image-manifest.json" \
  --miles-src "$MILES_SRC" --miles-commit "$MILES_COMMIT" \
  --miles-base-commit "$BASE_MILES_COMMIT" --miles-origin "$MILES_ORIGIN" \
  --sglang-src "$SGLANG_SRC" --sglang-commit "$SGLANG_COMMIT" \
  --sglang-base-commit "$BASE_SGLANG_COMMIT" --sglang-origin "$SGLANG_ORIGIN" \
  --sglang-version "$SGLANG_VERSION" --sglang-base-metadata "$META" \
  --wheel "$PEFT_WHEEL" --wheel "$PYRAGE_WHEEL" --wheel-platform x86_64 \
  ${PEFT_BASE_VERSION:+--replace-base-dist "peft=$PEFT_BASE_VERSION"} \
  --megatron-lm-src "$MEGATRON_LM_SRC" --megatron-lm-commit "$MEGATRON_LM_COMMIT" \
  --megatron-bridge-commit "$MEGATRON_BRIDGE_COMMIT" \
  --image-assert-dir "$OUT/image-assert" \
  ${ALLOW_INSTALL_METADATA_DRIFT:+--allow-install-metadata-drift} \
  --mtime "$MTIME" --out "$OUT/layer.tar"
LAYER_SHA=$(sha256sum "$OUT/layer.tar" | cut -d' ' -f1)
echo "layer.tar sha256=$LAYER_SHA"

# 7. Push: append the layer, then set the labels and the image-assert entrypoint
#    (docker runs it before the command; Modal does not run entrypoints, so yeto's
#    own entry points check the lock through yeto.launch.image_check).
DIGEST=""
if [ $PUSH = 1 ]; then
  "$CRANE" append --platform linux/amd64 -b "$BASE" -f "$OUT/layer.tar" -t "$TAG" >/dev/null
  "$CRANE" mutate "$TAG" -t "$TAG" \
    --entrypoint "/opt/yeto/image/entrypoint.sh" \
    --label "org.opencontainers.image.source=$MILES_ORIGIN" \
    --label "org.opencontainers.image.revision=$MILES_COMMIT" \
    --label "org.opencontainers.image.base.name=$BASE_REPO@$BASE_INDEX_DIGEST" \
    --label "ai.yeto.miles.commit=$MILES_COMMIT" \
    --label "ai.yeto.sglang.commit=$SGLANG_COMMIT" \
    --label "ai.yeto.image.lock=docker/miles-k31/versions.lock.json" >/dev/null
  DIGEST=$("$CRANE" digest "$TAG")
  echo "pushed $TAG@$DIGEST"
fi
python3 - "$OUT/build-record.json" <<PY
import json, sys
json.dump({"tag": "$TAG", "digest": "$DIGEST" or None, "layer_tar_sha256": "$LAYER_SHA",
           "base": "$BASE", "base_index": "$BASE_INDEX_DIGEST",
           "miles_commit": "$MILES_COMMIT", "sglang_commit": "$SGLANG_COMMIT",
           "sglang_version": "$SGLANG_VERSION", "megatron_commit": "$MEGATRON_LM_COMMIT",
           "megatron_bridge_commit": "$MEGATRON_BRIDGE_COMMIT", "torch_memory_saver_commit": "$TMS_COMMIT",
           "peft_version": "$PEFT_VERSION", "pyrage_version": "$PYRAGE_VERSION"},
          open(sys.argv[1], "w"), indent=2)
PY
echo "out=$OUT"
