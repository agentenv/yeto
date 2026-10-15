#!/usr/bin/env bash
# Build the yeto release image (rl-identity-tiering L9.12, release mode): the
# pinned miles-ports image (MILES_NEXT_IMAGE) plus one layer that holds the yeto
# source tree of one clean commit at /opt/yeto/src. In release mode the image
# digest is the run code pack digest (design D2); see docker/yeto-release/README.md.
#
#   scripts/build_yeto_release_image.sh            # layer + local image tarball only
#   scripts/build_yeto_release_image.sh --push     # also push; prints the digest
#
# Inputs (env):
#   CRANE      crane binary (go-containerregistry >= 0.20)
#   DEST_REPO  ghcr.io/michaellchung/yeto-release
#   OUT        output directory (default: a new /tmp/yeto-release-XXXXXX)
# Registry auth: crane reads ~/.docker/config.json; nothing is printed.
# Writes: $OUT/{layer.tar,release-manifest.json,build-record.json[,image.tar]}
#
# Status: written for L9.12, not run yet (no image built or pushed).
set -euo pipefail
cd "$(dirname "$0")/.."
PUSH=0; [ "${1:-}" = "--push" ] && PUSH=1
CRANE=${CRANE:-crane}
DEST_REPO=${DEST_REPO:-ghcr.io/michaellchung/yeto-release}
OUT=${OUT:-$(mktemp -d /tmp/yeto-release-XXXXXX)}
PREFIX=opt/yeto/src

# A release is one commit: refuse a dirty tree (the development-mode code pack
# carries a patch; a release image does not).
if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "release build needs a clean work tree (commit or stash first)" >&2
  exit 1
fi
YETO_COMMIT=$(git rev-parse HEAD)
BASE_IMAGE=$(python3 - <<'PY'
import re
src = open("yeto/rl/adapters/miles/pins.py").read()
body = re.search(r"^MILES_NEXT_IMAGE = \(\s*((?:\"[^\"]*\"\s*)+)\)", src, re.M).group(1)
print("".join(re.findall(r"\"([^\"]*)\"", body)).removeprefix("docker:"))
PY
)
case "$BASE_IMAGE" in *@sha256:*) ;; *) echo "base image is not pinned by digest: $BASE_IMAGE" >&2; exit 1;; esac

# The layer: `git archive` of the commit (tracked files only, file times = the
# commit time, so the same commit gives the same layer bytes) plus the release
# manifest the island reads to know its own code identity.
python3 - "$OUT/release-manifest.json" <<PY
import json, sys
json.dump({"schema": "yeto-release-v1", "yeto_commit": "$YETO_COMMIT",
           "base_image": "$BASE_IMAGE", "source_root": "/$PREFIX"},
          open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
git archive --format=tar --prefix="$PREFIX/" "$YETO_COMMIT" > "$OUT/source.tar"
mkdir -p "$OUT/extra/opt/yeto"
cp "$OUT/release-manifest.json" "$OUT/extra/opt/yeto/release-manifest.json"
touch -d "@$(git log -1 --format=%ct "$YETO_COMMIT")" "$OUT/extra/opt/yeto/release-manifest.json"
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$(git log -1 --format=%ct "$YETO_COMMIT")" \
    -C "$OUT/extra" -cf "$OUT/extra.tar" opt
tar --concatenate --file="$OUT/source.tar" "$OUT/extra.tar"
mv "$OUT/source.tar" "$OUT/layer.tar"
LAYER_SHA=$(sha256sum "$OUT/layer.tar" | cut -d' ' -f1)
TAG="$DEST_REPO:${YETO_COMMIT:0:12}"

if [ "$PUSH" = 1 ]; then
  "$CRANE" append --platform linux/amd64 -b "$BASE_IMAGE" -f "$OUT/layer.tar" -t "$TAG" >/dev/null
  DIGEST=$("$CRANE" digest "$TAG")
else
  "$CRANE" append --platform linux/amd64 -b "$BASE_IMAGE" -f "$OUT/layer.tar" -t "$TAG" \
      -o "$OUT/image.tar" >/dev/null
  DIGEST=""
fi

python3 - "$OUT/build-record.json" <<PY
import json, sys
json.dump({"schema": "yeto-release-build-v1", "yeto_commit": "$YETO_COMMIT",
           "base_image": "$BASE_IMAGE", "tag": "$TAG", "layer_sha256": "$LAYER_SHA",
           "pushed": bool($PUSH), "image_digest": "$DIGEST" or None},
          open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
echo "release layer: $OUT/layer.tar ($LAYER_SHA)"
[ "$PUSH" = 1 ] && echo "release image: $DEST_REPO@$DIGEST (code pack digest = ${DIGEST#sha256:})"
echo "build record: $OUT/build-record.json"
