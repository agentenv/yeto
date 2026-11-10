"""Engine-sampler engine pins: the stand-alone SGLang / vLLM servers the engine sampler drives.

Pinned apart from the Miles training image (``yeto.rl.adapters.miles.pins``) and kept neutral, so the
sampler adapters (``yeto.rl.adapters.sampler_engines``) read them without importing a training
adapter. Stdlib-only constants, like the Miles pins.
"""

from __future__ import annotations

from yeto.rl.image_registry import pinned_image

# ---------------------------------------------------------------- sampler-side SGLang
# The engine sampler (yeto.rl.engine.engine_sampler + sampler_engines.sglang_server)
# drives a stand-alone SGLang HTTP server and is pinned SEPARATELY from the Miles
# training image above.  Why: SGLANG_K31_COMMIT is whatever the Miles rollout
# worker links against (branch sglang-miles-kimi-k31, v0.5.23 base, frozen with
# the Miles fork); the sampler only needs the public HTTP surface
# (/generate with return_routed_experts + return_logprob/top_logprobs_num,
# --enable-fp32-lm-head, /update_weights_from_disk, /update_weights_from_tensor,
# /begin_weight_update + /end_weight_update, /server_info), which upstream main
# carries unchanged (field names, the int32 base64 routed_experts wire and the
# flat top-logprob fields were checked identical at SGLANG_SAMPLER_UPSTREAM_COMMIT;
# the sampler performance comparison).  Following main gives the sampler the Kimi-K3.1 inference
# work that lands there weekly (DSA/DCP kernels, MXFP4 decode MoE, prefill CP)
# without rebuilding the training image.  Moving this pin never moves the Miles one.
SGLANG_SAMPLER_UPSTREAM_REPOSITORY = "https://github.com/sgl-project/sglang"
# sgl-project/sglang main as built into lmsysorg/sglang:nightly-dev-20261109-842b4738
# (2026-11-09 00:38 UTC; CUDA 13.0.3, FlashInfer 0.7.3, python 3.12 venv at
# /opt/sglang).  The K3.1 commit list is in the pin comments below.
SGLANG_SAMPLER_UPSTREAM_COMMIT = "842b4738315fed2246b04a6fbe92fd6bb311ac25"
# The upstream NVIDIA base (linux/amd64 manifest of that nightly tag; the tag's
# index digest is sha256:d02579e744662d2cd196d77d7805c208d9f7579ff85c8fc2ef7a27ef636db491).
SGLANG_SAMPLER_BASE_IMAGE_CUDA = (
    "docker:docker.io/lmsysorg/sglang@sha256:"
    "ba9957853ac72aa18c0eb6a043079a40762656befd17b04f8a1173670b894307"
)
# The yeto sampler image: the base plus one overlay layer with yeto (k31/integ
# SGLANG_SAMPLER_YETO_COMMIT) at /opt/yeto/src, its pure-Python sampler
# dependencies (boto3, zstandard, httpx) at /opt/yeto/site, a .pth in the base's
# venv and /opt/yeto/image-manifest.json (a crane append, no docker
# daemon).  Tag sampler-k31-<sglang7>-<yeto7>.  The registry packages are private:
# pulls need registry credentials (docs/CLOUDS.md, "Default container images").
SGLANG_SAMPLER_YETO_COMMIT = "dd344e9fc75a7725ce896c07a0b8b69fbd61b5bb"
SGLANG_SAMPLER_IMAGE_CUDA = pinned_image(
    "yeto-sampler-k31", "c92843a330a1135016379ef3768447863b781bc47a5deccefb19b600e001ec59"
)  # tag sampler-k31-842b473-dd344e9, built 2026-11-09 17:09 CST
SGLANG_SAMPLER_IMAGE_MANIFEST = "/opt/yeto/image-manifest.json"
# AMD MI35x: the official dev image of upstream PR #48427 ([Feat] [AMD] model: Add
# Kimi-K3.1 MI35x support, base kimi-k3.1-day-0), lmsysorg/sglang:dev-kimi-k31-pro-mi35x
# pushed 2026-11-07 00:42 UTC (linux/amd64 manifest below; its label
# ai.radixark.sglang.commit names the PR commit; sgl-amd-mono 25723f1e46, ROCm 10.0
# torch 2.11).  The PR head moved once since (b6bab76558, #49152 gate_mode
# INTERLEAVE, 2026-11-09) and the image was not rebuilt; the node installs yeto
# into it with pip (scripts/sampler_node/run_engine_sampler.sh --pip-install).
SGLANG_SAMPLER_ROCM_PR = 48427
SGLANG_SAMPLER_ROCM_COMMIT = "64507491d93135f772478d7977d8443be7eab826"
SGLANG_SAMPLER_IMAGE_ROCM = (
    "docker:docker.io/lmsysorg/sglang@sha256:"
    "15b057f1a5e7d1954df281d268871f4c77bc2af05edeeb830cb920c977793344"
)
SGLANG_SAMPLER_IMAGE_ROCM_TAG = "lmsysorg/sglang:dev-kimi-k31-pro-mi35x"
# The ``engine_version`` label a sampler spec should carry for each platform
# (the sample's ``metadata.engine.engine_version``; free text, this is the convention).
SGLANG_SAMPLER_ENGINE_VERSIONS = {
    "cuda": f"sglang-main@{SGLANG_SAMPLER_UPSTREAM_COMMIT[:8]}",
    "rocm": f"sglang-pr{SGLANG_SAMPLER_ROCM_PR}@{SGLANG_SAMPLER_ROCM_COMMIT[:8]}",
}


def sampler_sglang_commit(platform: str) -> str | None:
    """The SGLang commit the sampler image of ``platform`` is expected to carry
    (``SGLANG_BUILD_COMMIT`` in the NVIDIA image's environment); None when the
    platform has no pinned image (ppu, npu)."""
    return {"cuda": SGLANG_SAMPLER_UPSTREAM_COMMIT, "rocm": SGLANG_SAMPLER_ROCM_COMMIT}.get(platform)


# ---------------------------------------------------------------- sampler-side vLLM
# The engine sampler's ``vllm-server`` adapter on ``platform`` cuda / rocm drives a
# stand-alone vLLM server (``vllm serve``), pinned like the sampler SGLang: apart from
# any training image. It needs the public surface of upstream vLLM (/v1/completions
# with logprobs K + return_tokens_as_token_ids + return_token_ids and the per-choice
# ``routed_experts`` .npy of --enable-return-routed-experts, --hf-overrides head_dtype
# float32, --max-logprobs, /version; with VLLM_SERVER_DEV_MODE=1 /pause, /resume and
# /collective_rpc reload_weights), checked in the source at VLLM_SAMPLER_INTERFACE_COMMIT
# and VLLM_SAMPLER_MAIN_COMMIT.
#
# No vLLM release serves Kimi-K3.1: v0.33.0 (2026-10-31) predates it and the K3.1
# recipe asks for >= 0.34.0 from the dedicated day-0 images, which are the pins below
# (both pushed 2026-11-03 / 2026-11-06: younger than two weeks, there is no older build
# with K3.1). Both are built from non-public trees; their RL interface files
# (routed_experts_capturer, completion protocol / serving, logits_processor, the RL and
# collective_rpc routers, reload/inplace, aux_output, serial_utils) are byte-identical,
# line endings aside, to public main at VLLM_SAMPLER_INTERFACE_COMMIT (2026-10-30).
VLLM_SAMPLER_UPSTREAM_REPOSITORY = "https://github.com/vllm-project/vllm"
VLLM_SAMPLER_INTERFACE_COMMIT = "335dd77b09161c5d63139b4e7ae51bb9928305b1"
# AMD MI355X: vllm/vllm-openai-rocm:kimi-k31 (AMD's "Kimi K3.1 Pro on AMD Instinct GPUs"
# recipe and vLLM's K3.1 recipe; InferenceX measured this digest), one linux/amd64
# manifest created 2026-11-06 08:30 UTC: ROCm 10.0, torch 2.13.0+rocm10.0.0, vLLM
# 0.0.0+mum.14d1ec2336 in /opt/runtime-venv (labels runtime.mum.commit 14d1ec23,
# runtime.aiter.commit 52846bc2; /opt/runtime-provenance/provenance.json). The node
# installs yeto into it with pip (scripts/sampler_node/run_engine_sampler.sh --pip-install).
# Mirrored byte-identical (same digest) to the yeto registry (yeto.rl.image_registry).
# the upstream vLLM ROCm image on Docker Hub (public; never rebuilt by yeto), so it is not
# under YETO_IMAGE_REGISTRY: that registry holds yeto's own images only
VLLM_SAMPLER_IMAGE_ROCM = (
    "docker:docker.io/vllm/vllm-openai-rocm@sha256:"
    "5f47cc1004d9a7ac65f8a913725edb20c62c99a07e31cbf4ba21741bd0b4f9b1"
)
VLLM_SAMPLER_IMAGE_ROCM_TAG = "vllm/vllm-openai-rocm:kimi-k31"
VLLM_SAMPLER_ROCM_VERSION = "0.0.0+mum.14d1ec2336"  # GET /version of that image
# The upstream ROCm nightly of 2026-11-09 (main 083fbac02, 0.33.0rc1.dev1065+g083fbac02,
# tag nightly-rocm100-083fbac0..., index digest) did NOT start Kimi-K3.1 on 8 x MI355X
# (5 attempts, 2026-11-09): main's AMD model has no DSA indexer (the checkpoint's
# indexer.k_norm.bias has no parameter) and its ROCm sparse MLA backend cannot read the
# fp8_ds_mla KV cache the K3.1 config forces (upstream #66532 / #65685, open). Recorded
# so that nobody pins it again before those land.
VLLM_SAMPLER_IMAGE_ROCM_NIGHTLY = (
    "docker:docker.io/vllm/vllm-openai-rocm@sha256:"
    "07ebdd53fe5b6dfb37f99b6cd084df2734faa2fef17c79317e283a68d30b5d2a"
)
# NVIDIA B200 / B300: vllm/vllm-openai:kimi-k31 (vLLM's K3.1 recipe image; CUDA 13.0.3,
# cu130 build), linux/amd64 manifest of the tag below (index digest
# sha256:904520f3fefc32414abcedf6ba8dadc9c5978db1bcdaf40f58d6b9036ef30a99), created
# 2026-11-03 18:27 UTC; vLLM 0.1.dev24216+g266e88c05 in /usr/local/lib/python3.12/dist-packages.
VLLM_SAMPLER_BASE_IMAGE_CUDA = (
    "docker:docker.io/vllm/vllm-openai@sha256:"
    "88175ddf4f59944ccbb24c6b60c1bc535a160be3c4f09a3b08702c5e301a599e"
)
VLLM_SAMPLER_BASE_IMAGE_CUDA_TAG = "vllm/vllm-openai:kimi-k31"
VLLM_SAMPLER_CUDA_VERSION = "0.1.dev24216+g266e88c05"  # GET /version of that image
# The yeto vLLM sampler image: that base plus one overlay layer with yeto
# (k31/vllm-new VLLM_SAMPLER_YETO_COMMIT) at /opt/yeto/src, its pure-Python sampler
# dependencies (boto3, zstandard, httpx) at /opt/yeto/site, a .pth in the base's
# dist-packages (the base's packages win on a name clash) and /opt/yeto/image-manifest.json
# (a crane append, no docker daemon). Private ghcr
# package like SGLANG_SAMPLER_IMAGE_CUDA. Its server's interpreter is python3.
VLLM_SAMPLER_YETO_COMMIT = "a84f1891b7857bc409437bf2d07da5c735325d18"
VLLM_SAMPLER_IMAGE_CUDA = pinned_image(
    "yeto-sampler-k31", "a869d92f602a0665e643ea549842370e43baa471a7078e887479c85063dc6891"
)  # tag sampler-k31-vllm-266e88c-a84f189, built 2026-11-09 19:18 CST
VLLM_SAMPLER_IMAGE_MANIFEST = "/opt/yeto/image-manifest.json"
# Public main with the merged K3.1 splits 1/5 (#66030), 2/5 (#66028) and 3/5 (#66037;
# served a dummy-weight K3.1 at TP8 on B300) and its cu130 per-commit wheel: the
# public-source alternative on CUDA (its interface was checked too). Not on ROCm (above).
VLLM_SAMPLER_MAIN_COMMIT = "083fbac021348c2c7af8a7bc71ba18199d0dd7a4"
VLLM_SAMPLER_MAIN_VERSION = "0.33.0rc1.dev1065+g083fbac02"
VLLM_SAMPLER_MAIN_WHEEL_CUDA = ("https://wheels.vllm.ai/083fbac021348c2c7af8a7bc71ba18199d0dd7a4/"
                                "vllm-0.33.0rc1.dev1065%2Bg083fbac02-cp38-abi3-manylinux_2_28_x86_64.whl")
VLLM_SAMPLER_ENGINE_VERSIONS = {
    "cuda": f"vllm-kimi-k31@{VLLM_SAMPLER_CUDA_VERSION}",
    "rocm": f"vllm-kimi-k31@{VLLM_SAMPLER_ROCM_VERSION}",
}


def sampler_vllm_version(platform: str) -> str | None:
    """The ``GET /version`` the sampler's vLLM server of ``platform`` is expected to
    report (the pinned image's); None on the legacy line (npu: vllm-ascend, or no platform)."""
    return {"cuda": VLLM_SAMPLER_CUDA_VERSION, "rocm": VLLM_SAMPLER_ROCM_VERSION}.get(platform)
