# yeto miles-ports image (MILES_NEXT_IMAGE) -- APPROXIMATE, UNVERIFIED
# docker-build equivalent of scripts/build_miles_ports_image.sh (never built:
# the build host has no docker).  The script produces the real image with
# `crane append` (one layer from docker/miles-ports/make_layer.py) because the
# build host has no docker daemon.  Pushed PRIVATE to
# ghcr.io/michaellchung/yeto-miles-ports:<miles7>-<sglang7>.
#
#   docker build --platform linux/amd64 -f docker/miles-ports/Dockerfile \
#     --build-arg MILES_COMMIT=... --build-arg SGLANG_COMMIT=... \
#     --build-context manifest=<dir with image-manifest.json> .
#
# Base: radixark/miles (upstream docker/Dockerfile at radixark 9e4260d), linux/amd64
# manifest of index sha256:90940828dcd4d54fd907ff668b43537cbd94778047580e4160d6560af548b74d.
# It already has Miles 9e4260d editable at /root/miles and SGLang (sglang-miles
# e5bba1f) editable at /sgl-workspace/sglang/python.  The forks differ from those
# checkouts only in pure-Python files, so no native (sgl-kernel / CUDA / rust)
# rebuild is needed; re-running the same editable installs --no-deps swaps in
# the fork sources and regenerates the version metadata.
FROM docker.io/radixark/miles@sha256:4d69750720eb1a4b99976a3fed45ed018e1fc4fff3d240667fa17d258dfc6e0c

ARG MILES_REPOSITORY=https://github.com/michaellchung/miles
ARG MILES_COMMIT=df23a0b5e7937a35f7cea8c77b848805c30a17eb
ARG SGLANG_REPOSITORY=https://github.com/michaellchung/sglang
ARG SGLANG_COMMIT=2fa880182eefdb31e64d8ea70317be27f9f32121
# setuptools-scm version of SGLANG_COMMIT (describe v0.5.20-69-g4e4148f1b);
# pinned because the one-commit clone has no tags to describe.
ARG SGLANG_VERSION=0.5.21.dev70+g2fa8801

# Miles fork -> /root/miles (same path the base installed editable).
RUN rm -rf /root/miles \
 && git init -q /root/miles && cd /root/miles \
 && git remote add origin "$MILES_REPOSITORY" \
 && git fetch -q --depth 1 origin "$MILES_COMMIT" \
 && git checkout -q --detach "$MILES_COMMIT" \
 && test "$(git rev-parse HEAD)" = "$MILES_COMMIT" \
 && pip install --no-deps -e /root/miles

# SGLang fork -> /sgl-workspace/sglang (same path, same `python[all]` extras
# spec as the base, rust extensions off as in the base).  Build artefacts the
# base left in the tree (untracked) are kept, as the crane layer keeps them.
RUN cd /sgl-workspace/sglang \
 && rm -rf .git && git init -q . \
 && git remote add origin "$SGLANG_REPOSITORY" \
 && git fetch -q --depth 1 origin "$SGLANG_COMMIT" \
 && git checkout -q -f --detach "$SGLANG_COMMIT" \
 && test "$(git rev-parse HEAD)" = "$SGLANG_COMMIT" \
 && SETUPTOOLS_SCM_PRETEND_VERSION="$SGLANG_VERSION" \
    SGLANG_BUILD_RUST_EXTS=none pip install -e "python[all]" --no-deps

# Provenance (written by the build script: component SHAs + base digests).
COPY --from=manifest image-manifest.json /opt/yeto/image-manifest.json
