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
#   DEST_REPO  ghcr.io/agentenv/yeto-release
#   OUT        output directory (default: a new /tmp/yeto-release-XXXXXX)
# Registry auth: crane reads ~/.docker/config.json; nothing is printed.
# Writes: $OUT/{layer.tar,release-manifest.json,build-record.json[,image.tar]}
#
set -euo pipefail
cd "$(dirname "$0")/.."
PUSH=0; [ "${1:-}" = "--push" ] && PUSH=1
CRANE=${CRANE:-crane}
DEST_REPO=${DEST_REPO:-ghcr.io/agentenv/yeto-release}
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
# commit time, so the same commit gives the same layer bytes), the release
# manifest the island checks its tree against (yeto.codepack.verify_release_tree),
# and /root/sky_workdir -> /opt/yeto/src (island scripts `cd ~/sky_workdir`; the
# launcher ships no local tree in release mode).
# One GNU-format tar, written from an unpacked tree: `git archive` output carries a
# pax global header (typeflag 'g', the commit id) that image unpackers (umoci, used
# by Modal) refuse.
COMMIT_TIME=$(git log -1 --format=%ct "$YETO_COMMIT")
rm -rf "$OUT/tree" && mkdir -p "$OUT/tree"
git -c tar.umask=022 archive --format=tar --prefix="$PREFIX/" "$YETO_COMMIT" | tar -x -p -C "$OUT/tree"
SOURCE_TREE_SHA=$(cd "$OUT/tree/$PREFIX" && python3 -c \
  'import sys; sys.path.insert(0, "."); from yeto.provenance import source_tree_sha256; print(source_tree_sha256("yeto"))')
python3 - "$OUT/release-manifest.json" <<PY
import json, sys
json.dump({"schema": "yeto-release-v1", "yeto_commit": "$YETO_COMMIT",
           "base_image": "$BASE_IMAGE", "source_root": "/$PREFIX",
           "source_tree_sha256": "$SOURCE_TREE_SHA"},
          open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
find "$OUT/tree" -name __pycache__ -prune -exec rm -rf {} +  # the hash step above may write them
cp "$OUT/release-manifest.json" "$OUT/tree/opt/yeto/release-manifest.json"
chmod 0644 "$OUT/tree/opt/yeto/release-manifest.json"
mkdir -p "$OUT/tree/root" && ln -s "/$PREFIX" "$OUT/tree/root/sky_workdir"
# fixed directory modes (independent of the local umask); /root stays 0700
find "$OUT/tree" -type d -exec chmod 0755 {} + && chmod 0700 "$OUT/tree/root"
tar --sort=name --format=gnu --owner=0 --group=0 --numeric-owner --mtime="@$COMMIT_TIME" \
    -C "$OUT/tree" -cf "$OUT/layer.tar" opt root
rm -rf "$OUT/tree"
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
