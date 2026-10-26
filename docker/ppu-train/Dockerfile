# yeto-ppu-train: PPU trainer (Megatron + Miles glm5_next, or verl FSDP2) and
# SGLang sampler image, linux/amd64 (S25 N0.2). WRITTEN, NEVER BUILT.
#
# Versions: versions.lock.json in this directory is the single source; the
# build args below must equal it (tests/test_engine_images.py checks this).
#
#   docker build --platform linux/amd64 -f docker/ppu-train/Dockerfile \
#     --build-arg BASE_IMAGE=<registry>/inference-xpu-pytorch-opt@sha256:<digest> \
#     --build-arg MEGATRON_COMMIT=<radixark/Megatron-LM miles-main sha> .
#
# Build context: the yeto repository root. yeto code is NOT copied here: the
# release layer (scripts/build_yeto_release_image.sh, docker/yeto-release) puts
# one clean commit at /opt/yeto/src on top of this image.
#
# Base: the Alibaba PPU image the user runs as `ppu_sgl` on ppu1-4
# (inference-xpu-pytorch-opt:v0.5.17: sglang 0.5.17, torch 2.10.0, triton 3.6.0,
# fla 1.0.0, transformers 5.12.1, ray 2.31.0; SESSION25/N0.1-ppu-survey.md).
# Its registry path and digest are not known yet (read them on ppu1 with
# `docker inspect`), so BASE_IMAGE has no default and must be a digest.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
RUN case "${BASE_IMAGE}" in *@sha256:*) ;; \
      *) echo "BASE_IMAGE must be pinned by digest: '${BASE_IMAGE}'" >&2; exit 1;; esac

# Freeze the base: every later pip install runs with these constraints, so a
# dependency that would replace a base package fails the build. ray is left
# out on purpose (Miles needs >= 2.56; versions.lock.json "decisions").
# Built from package metadata, not `pip freeze`, so editable and URL installs
# (the base's sglang may be one) are constrained too.
RUN mkdir -p /opt/yeto/image \
 && pip freeze > /opt/yeto/image/base-freeze.txt \
 && python3 -c 'import importlib.metadata as m; print("\n".join(sorted({f"{d.name}=={d.version}" for d in m.distributions() if d.name.lower() != "ray"})))' \
      > /opt/yeto/image/base-constraints.txt

ENV PIP_CONSTRAINT=/opt/yeto/image/base-constraints.txt

COPY docker/ppu-train/requirements.txt /opt/yeto/image/requirements.txt
RUN pip install --no-cache-dir -r /opt/yeto/image/requirements.txt

# Megatron-LM: radixark fork, branch miles-main (the only one with
# megatron/core/transformer/hyper_connection.py, needed by glm5_next mHC).
# No commit is pinned yet (Miles' own image follows the branch head), so the
# build refuses to run without one.
ARG MEGATRON_REPOSITORY=https://github.com/radixark/Megatron-LM
ARG MEGATRON_COMMIT
RUN test -n "${MEGATRON_COMMIT}" || { echo "MEGATRON_COMMIT is required" >&2; exit 1; } \
 && git init -q /opt/Megatron-LM && cd /opt/Megatron-LM \
 && git remote add origin "${MEGATRON_REPOSITORY}" \
 && git fetch -q --depth 1 origin "${MEGATRON_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${MEGATRON_COMMIT}" \
 && pip install --no-cache-dir --no-deps -e /opt/Megatron-LM

# mbridge (HF <-> Megatron weights; Miles glm5_next bridge plugin builds on it),
# Megatron-Bridge and two Megatron data extras, all without deps as upstream.
ARG MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
ARG MEGATRON_BRIDGE_COMMIT=2e09c234a3272285140224d0d698593418b55ba5
RUN pip install --no-cache-dir --no-deps \
      "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}" \
      "git+https://github.com/radixark/Megatron-Bridge.git@${MEGATRON_BRIDGE_COMMIT}" \
      megatron-energon multi-storage-client

# tilelang: the DSA SparseMLA kernels of Miles glm5_next need it. Whether it
# runs on PPU is the first PPU probe (N7.5); the lock marks it optional.
ARG TILELANG_VERSION=0.1.14
RUN pip install --no-cache-dir --no-deps "tilelang==${TILELANG_VERSION}" apache-tvm-ffi==0.1.11 \
 || echo "tilelang ${TILELANG_VERSION} did not install; DSA falls back to torch (N7.5)" >&2

# Miles at the GLM 5.3 Flash commit (radixark/miles #2786). T-N77 builds the
# yeto fork branch on this commit; switch MILES_* to it when that is pushed.
ARG MILES_REPOSITORY=https://github.com/radixark/miles
ARG MILES_COMMIT=cc76e23915b2132ecfff65b97331fd83b02637e6
RUN git init -q /opt/miles && cd /opt/miles \
 && git remote add origin "${MILES_REPOSITORY}" \
 && git fetch -q --depth 1 origin "${MILES_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${MILES_COMMIT}" \
 && pip install --no-cache-dir --no-deps -e /opt/miles

# verl fork at the commit yeto pins (yeto/rl/adapters/verl/pins.py). Its
# runtime deps are in requirements.txt; vllm is not installed (the PPU
# sampler is SGLang).
ARG VERL_REPOSITORY=https://github.com/michaellchung/verl
ARG VERL_COMMIT=acad9875a8bdfc81afbcb0a50d146b2630f44093
RUN git init -q /opt/verl && cd /opt/verl \
 && git remote add origin "${VERL_REPOSITORY}" \
 && git fetch -q --depth 1 origin "${VERL_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${VERL_COMMIT}" \
 && pip install --no-cache-dir --no-deps -e /opt/verl

# Commits the start check compares with the lock.
RUN printf '{"megatron": "%s", "mbridge": "%s", "megatron_bridge": "%s", "miles": "%s", "verl": "%s"}\n' \
      "${MEGATRON_COMMIT}" "${MBRIDGE_COMMIT}" "${MEGATRON_BRIDGE_COMMIT}" "${MILES_COMMIT}" "${VERL_COMMIT}" \
      > /opt/yeto/image/commits.json \
 && pip check || echo "pip check reported conflicts (kept in the build log)" >&2

COPY docker/ppu-train/versions.lock.json /opt/yeto/image/versions.lock.json
COPY docker/image-assert/assert_image_versions.py docker/image-assert/entrypoint.sh /opt/yeto/image/
ENV PYTHONPATH=/opt/yeto/src
# Fail at build time too: an image whose own lock does not hold is never pushed.
# (Imports that need a PPU device are not in the lock; the start check repeats
# everything on the node.)
RUN python3 /opt/yeto/image/assert_image_versions.py /opt/yeto/image/versions.lock.json
ENTRYPOINT ["/opt/yeto/image/entrypoint.sh"]
CMD ["bash"]
