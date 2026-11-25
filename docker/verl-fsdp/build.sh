#!/usr/bin/env bash
# Build + push the verl FSDP2 registry image (pins.VERL_REGISTRY_IMAGE): the recipe of the
# Modal image `verl-build:<VERL_COMMIT>` (yeto/rl/adapters/verl/image.py:recipe_steps) as a
# Dockerfile, for trainer islands that pull an image (sky, ssh docker nodes).
#
#   DEST_REPO=ghcr.io/<owner>/yeto-verl-fsdp docker/verl-fsdp/build.sh [--push]
#
# Needs a docker daemon on linux/amd64 (about 60 GB of free disk) and, for --push, a
# registry login done beforehand (`docker login`); nothing is printed from it.
# Prints the tag and, after --push, the digest: pin both in yeto/rl/adapters/verl/pins.py
# (VERL_REGISTRY_TAG, VERL_REGISTRY_DIGEST, VERL_REGISTRY_RECIPE_SHA256).
set -euo pipefail
cd "$(dirname "$0")/../.."
PUSH=0; [ "${1:-}" = "--push" ] && PUSH=1
PY=${PY:-python3}
DEST_REPO=${DEST_REPO:?set DEST_REPO to the target repository, e.g. ghcr.io/<owner>/yeto-verl-fsdp}

# the checked-in Dockerfile must be the recipe's (the same steps as the Modal image)
if ! diff -q <("$PY" -m yeto.rl.adapters.verl.image --dockerfile) docker/verl-fsdp/Dockerfile >/dev/null; then
  echo "docker/verl-fsdp/Dockerfile is not the recipe's: regenerate it with" >&2
  echo "  $PY -m yeto.rl.adapters.verl.image --dockerfile > docker/verl-fsdp/Dockerfile" >&2
  exit 1
fi
RECIPE=$("$PY" -m yeto.rl.adapters.verl.image --recipe-sha256)
COMMIT=$("$PY" -c 'from yeto.rl.adapters.verl.pins import VERL_COMMIT; print(VERL_COMMIT)')
TAG="fsdp-${COMMIT:0:8}-${RECIPE:0:8}"

CTX=$(mktemp -d /tmp/verl-fsdp-ctx-XXXXXX)
trap 'rm -rf "$CTX"' EXIT
cp docker/verl-fsdp/Dockerfile "$CTX/Dockerfile"
cp yeto/rl/adapters/verl/patch_verl.py "$CTX/patch_verl.py"
docker build --platform linux/amd64 -t "$DEST_REPO:$TAG" "$CTX"
echo "tag: $TAG"
echo "recipe_sha256: $RECIPE"
if [ "$PUSH" = 1 ]; then
  docker push "$DEST_REPO:$TAG" >/dev/null
  echo "digest: $(docker inspect --format '{{index .RepoDigests 0}}' "$DEST_REPO:$TAG" | sed 's/.*@//')"
fi
