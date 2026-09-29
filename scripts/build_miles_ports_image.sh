#!/usr/bin/env bash
# Build + push the private yeto miles-ports image (MILES_NEXT_IMAGE) without a
# docker daemon: crane appends one overlay layer (docker/miles-ports/
# make_layer.py) to the pinned radixark/miles base.  docker/miles-ports/
# Dockerfile describes the same result for a docker build.
#
#   scripts/build_miles_ports_image.sh [--push]
#
# Inputs (env, defaults = this checkout's pins):
#   MILES_SRC / SGLANG_SRC   local clones containing the pinned commits
#   CRANE                    crane binary (go-containerregistry >= 0.20)
#   DEST_REPO                ghcr.io/michaellchung/yeto-miles-ports (private)
# Registry auth: crane reads ~/.docker/config.json; nothing is printed.
# Writes: $OUT/{layer.tar,image-manifest.json,build-record.json}
set -euo pipefail
cd "$(dirname "$0")/.."
PUSH=0; [ "${1:-}" = "--push" ] && PUSH=1
CRANE=${CRANE:-crane}
MILES_SRC=${MILES_SRC:-$HOME/work/miles-next}
SGLANG_SRC=${SGLANG_SRC:-$HOME/work/sglang-next}
DEST_REPO=${DEST_REPO:-ghcr.io/michaellchung/yeto-miles-ports}
OUT=${OUT:-$(mktemp -d /tmp/img-build-XXXXXX)}

# radixark/miles multi-arch index (upstream docker/Dockerfile at radixark
# 9e4260d) and its linux/amd64 manifest, which is what we extend.
BASE_REPO=docker.io/radixark/miles
BASE_INDEX_DIGEST=sha256:90940828dcd4d54fd907ff668b43537cbd94778047580e4160d6560af548b74d
BASE_AMD64_DIGEST=sha256:4d69750720eb1a4b99976a3fed45ed018e1fc4fff3d240667fa17d258dfc6e0c
# What the base image itself has installed (read from its layers):
BASE_MILES_COMMIT=9e4260de047a704208535c0e90c531929879ab40   # /root/miles
BASE_SGLANG_COMMIT=e5bba1f9c433da5352e521a8bb826bab960e7836  # /sgl-workspace/sglang (sglang-miles head at build)
BASE_SGLANG_VERSION=0.5.21.dev64+ge5bba1f

pins=$(python3 - <<'PY'
import re
src = open("yeto/rl/__init__.py").read()
for k in ("MILES_NEXT_REPOSITORY", "MILES_NEXT_COMMIT", "MILES_NEXT_UPSTREAM_COMMIT",
          "SGLANG_NEXT_REPOSITORY", "SGLANG_NEXT_COMMIT", "SGLANG_NEXT_UPSTREAM_COMMIT"):
    print(f"{k}={re.search(rf'^{k} = \"([^\"]+)\"', src, re.M).group(1)}")
PY
)
eval "$pins"
[ "$(git -C "$MILES_SRC" rev-parse "$MILES_NEXT_COMMIT^{commit}")" = "$MILES_NEXT_COMMIT" ]
[ "$(git -C "$SGLANG_SRC" rev-parse "$SGLANG_NEXT_COMMIT^{commit}")" = "$SGLANG_NEXT_COMMIT" ]
git -C "$SGLANG_SRC" cat-file -e "$BASE_SGLANG_COMMIT^{commit}"
git -C "$MILES_SRC" cat-file -e "$BASE_MILES_COMMIT^{commit}"

# setuptools-scm version the fork would get from a real `pip install -e`
# (sglang's describe: v0.5.20-<n>-g<sha7> -> 0.5.21.dev<n>+g<sha7>).
desc=$(git -C "$SGLANG_SRC" describe --tags --long --match 'v0.5.20' "$SGLANG_NEXT_COMMIT")
n=$(echo "$desc" | cut -d- -f2)
SGLANG_VERSION="0.5.21.dev${n}+g${SGLANG_NEXT_COMMIT:0:7}"
MS=${MILES_NEXT_COMMIT:0:7}; SS=${SGLANG_NEXT_COMMIT:0:7}
TAG="$DEST_REPO:$MS-$SS"
MTIME=$(git -C "$MILES_SRC" log -1 --format=%ct "$MILES_NEXT_COMMIT")

# The base's sglang editable metadata, pulled from the one base layer that
# wrote it (streamed; only the metadata files are extracted).
META="$OUT/base-sglang-meta"
if [ ! -d "$META" ]; then
  layer=$("$CRANE" manifest "$BASE_REPO@$BASE_AMD64_DIGEST" | python3 -c '
import json, sys; print(json.load(sys.stdin)["layers"][104]["digest"])')
  mkdir -p "$OUT/meta-x"
  "$CRANE" blob "$BASE_REPO@$layer" | tar -xzf - -C "$OUT/meta-x" \
    --wildcards "opt/sglang/lib/python3.12/site-packages/*sglang_0_5_21_dev64*finder.py" \
    "opt/sglang/lib/python3.12/site-packages/__editable__.sglang-$BASE_SGLANG_VERSION.pth" \
    "opt/sglang/lib/python3.12/site-packages/sglang-$BASE_SGLANG_VERSION.dist-info/*"
  mv "$OUT/meta-x/opt/sglang/lib/python3.12/site-packages" "$META"
fi

python3 - "$OUT/image-manifest.json" <<PY
import json, sys
json.dump({
  "schema": 1,
  "image": "$DEST_REPO",
  "tag": "$MS-$SS",
  "base": {"repository": "$BASE_REPO", "index_digest": "$BASE_INDEX_DIGEST",
           "platform": "linux/amd64", "manifest_digest": "$BASE_AMD64_DIGEST",
           "miles_commit": "$BASE_MILES_COMMIT", "sglang_commit": "$BASE_SGLANG_COMMIT"},
  "miles": {"repository": "$MILES_NEXT_REPOSITORY", "commit": "$MILES_NEXT_COMMIT",
            "upstream_commit": "$MILES_NEXT_UPSTREAM_COMMIT", "path": "/root/miles",
            "install": "pip install -e /root/miles --no-deps (base's editable install, same path)"},
  "sglang": {"repository": "$SGLANG_NEXT_REPOSITORY", "commit": "$SGLANG_NEXT_COMMIT",
             "upstream_commit": "$SGLANG_NEXT_UPSTREAM_COMMIT", "path": "/sgl-workspace/sglang",
             "version": "$SGLANG_VERSION",
             "install": "pip install -e /sgl-workspace/sglang/python[all] --no-deps (base's editable install, same path)"},
  "pure_python_overlay": True,
  "builder": "scripts/build_miles_ports_image.sh (crane append); docker/miles-ports/Dockerfile equivalent",
}, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY

python3 docker/miles-ports/make_layer.py \
  --manifest-json "$OUT/image-manifest.json" \
  --miles-src "$MILES_SRC" --miles-commit "$MILES_NEXT_COMMIT" \
  --miles-base-commit "$BASE_MILES_COMMIT" --miles-origin "$MILES_NEXT_REPOSITORY" \
  --sglang-src "$SGLANG_SRC" --sglang-commit "$SGLANG_NEXT_COMMIT" \
  --sglang-base-commit "$BASE_SGLANG_COMMIT" --sglang-origin "$SGLANG_NEXT_REPOSITORY" \
  --sglang-base-version "$BASE_SGLANG_VERSION" --sglang-version "$SGLANG_VERSION" \
  --sglang-base-metadata "$META" --mtime "$MTIME" --out "$OUT/layer.tar"
LAYER_SHA=$(sha256sum "$OUT/layer.tar" | cut -d' ' -f1)
echo "layer.tar sha256=$LAYER_SHA"

DIGEST=""
if [ $PUSH = 1 ]; then
  "$CRANE" append --platform linux/amd64 -b "$BASE_REPO@$BASE_AMD64_DIGEST" \
    -f "$OUT/layer.tar" -t "$TAG" >/dev/null
  # (GHCR refuses registry-API tag deletes, so no temporary tag: the
  # label step re-points the same tag.)
  "$CRANE" mutate "$TAG" -t "$TAG" \
    --label "org.opencontainers.image.source=$MILES_NEXT_REPOSITORY" \
    --label "org.opencontainers.image.revision=$MILES_NEXT_COMMIT" \
    --label "org.opencontainers.image.base.name=$BASE_REPO@$BASE_INDEX_DIGEST" \
    --label "ai.yeto.miles.commit=$MILES_NEXT_COMMIT" \
    --label "ai.yeto.sglang.commit=$SGLANG_NEXT_COMMIT" >/dev/null
  DIGEST=$("$CRANE" digest "$TAG")
  echo "pushed $TAG@$DIGEST"
fi
python3 - "$OUT/build-record.json" <<PY
import json, sys
json.dump({"tag": "$TAG", "digest": "$DIGEST" or None, "layer_tar_sha256": "$LAYER_SHA",
           "base": "$BASE_REPO@$BASE_AMD64_DIGEST", "base_index": "$BASE_INDEX_DIGEST",
           "miles_commit": "$MILES_NEXT_COMMIT", "sglang_commit": "$SGLANG_NEXT_COMMIT",
           "sglang_version": "$SGLANG_VERSION"}, open(sys.argv[1], "w"), indent=2)
PY
echo "out=$OUT"
