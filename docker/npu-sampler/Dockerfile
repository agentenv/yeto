# yeto-npu-sampler: Ascend 910B (A2) vllm-ascend sampler image, linux/arm64
# (S25 N0.2, stage 2). WRITTEN, NEVER BUILT.
#
# Versions: versions.lock.json in this directory is the single source.
#
#   docker build --platform linux/arm64 -f docker/npu-sampler/Dockerfile \
#     --build-arg BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.30.0rc1@sha256:<digest> \
#     --build-context fixes=<dir with apply.sh> .
#
# Base: the official vllm-ascend v0.30.0rc1 A2 image. It carries the CANN
# user-space (toolkit, nnal), torch / torch_npu, vLLM and vllm-ascend; the host
# provides only the driver and firmware, mounted at run time
# (/usr/local/Ascend/driver). The user ran GLM-5.3-Flash W4A8 on ModelArts with
# vllm-ascend 0.30.0rc1 (as a rootfs, CANN 9.1); the vllm-ascend compatibility
# matrix (mkdocs.yml, commit 0ca6a8b) names CANN 9.2.0 and Ascend HDK 26.0.RC1
# for that release. Host npu1 has driver 24.1.0.3 and CANN 8.0.1: see
# versions.lock.json "decisions" for how that gap is handled.
#
# yeto code is not copied: the sampler adapter runs from /opt/yeto/src (the
# release layer, as for the other images). The release build script is
# linux/amd64 only today, so an arm64 release layer is a follow-up.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
RUN case "${BASE_IMAGE}" in *@sha256:*) ;; \
      *) echo "BASE_IMAGE must be pinned by digest: '${BASE_IMAGE}'" >&2; exit 1;; esac

# Freeze the base exactly as docker/ppu-train does: nothing below may replace a
# base package (torch, torch_npu, vllm, vllm_ascend, triton_ascend, ...).
RUN mkdir -p /opt/yeto/image \
 && pip freeze > /opt/yeto/image/base-freeze.txt \
 && python3 -c 'import importlib.metadata as m; print("\n".join(sorted({f"{d.name}=={d.version}" for d in m.distributions()})))' \
      > /opt/yeto/image/base-constraints.txt
ENV PIP_CONSTRAINT=/opt/yeto/image/base-constraints.txt

# yeto sampler-side runtime (HTTP client, weight follower, post-load checks).
RUN pip install --no-cache-dir cloudpickle "huggingface-hub>=0.34" "safetensors>=0.4" pyyaml "httpx[http2]"

# The two run-time fixes the user needed on ModelArts (A-INF items 15 and 17):
# the libprofapi symbol stub (CANN 9.1 HCCL hang; whether CANN 9.2 still needs
# it is unknown) and the W4A8 host-memory staging change (the adapter turns it
# on with VLLM_ASCEND_W4A8_CPU_STAGE=1). Their source is the user's ModelArts
# environment, not this repository, so they arrive as a build context:
# apply.sh installs them and writes /opt/yeto/image/fixes.json
# ({"fixes": [{"name", "sha256"}, ...]}). The PT_INTERP change is for the
# no-root rootfs only; a container brings its own loader. The build stops when
# the context is missing or apply.sh writes nothing.
COPY --from=fixes . /tmp/yeto-fixes
RUN sh /tmp/yeto-fixes/apply.sh \
 && test -s /opt/yeto/image/fixes.json \
 && rm -rf /tmp/yeto-fixes

COPY docker/npu-sampler/versions.lock.json /opt/yeto/image/versions.lock.json
COPY docker/image-assert/assert_image_versions.py docker/image-assert/entrypoint.sh /opt/yeto/image/
ENV PYTHONPATH=/opt/yeto/src
# No build-time check here: torch_npu imports need the driver, which no build
# host has. The entrypoint runs the whole check on the node instead.
ENTRYPOINT ["/opt/yeto/image/entrypoint.sh"]
CMD ["bash"]
