"""verl pins (rl-verl-backend design D1, D10; user decision S16).

The user's fork https://github.com/michaellchung/verl at acad9875 (the commit
S16 measured on Modal 1xH100, VERL-MODAL-CHECK-S16.md).  Dependencies come
from the fork's own ``uv.lock`` (``uv sync --frozen --all-packages --extra vllm
--extra fsdp``), the same recipe as upstream ``docker/Dockerfile.uv.cu130``.
"""

VERL_REPOSITORY = "https://github.com/michaellchung/verl"
VERL_COMMIT = "acad9875a8bdfc81afbcb0a50d146b2630f44093"
VERL_BASE_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
VERL_PYTHON = "3.12"
VERL_UV_VERSION = "0.9.5"
VERL_ROOT = "/workspace/verl"
# Versions S16 measured from that lock (env.json); the island asserts them.
EXPECTED_VERSIONS = {
    "torch": "2.13.0+cu130",
    "vllm": "0.29.0",
    "transformers": "5.12.1",
    "peft": "0.19.1",
}

# --- Ascend NPU stack (rl-verl-backend 3.11/3.14) ----------------------------
#
# The NPU stack is a second, separate image: vLLM 0.23 + torch 2.10 + CANN
# 9.1.0 cannot share an image with the CUDA stack above (vLLM 0.29 + torch
# 2.13). Sources, both read 2026-10-10:
#   * fork acad9875 ``docker/ascend/Dockerfile.ascend_9.1.0_a2`` -- base image,
#     vLLM tag v0.23.0, vllm-ascend branch releases/v0.23.0 (built with
#     COMPILE_CUSTOM_KERNELS=1), MindSpeed core_r0.18.0, transformers 5.10.4;
#   * fork ``docker/ascend/supported_tags.md`` (updated 09/28/2026) -- the
#     910b / A3 / A5 tags, vLLM backend on CANN 9.1.0;
#   * infra-drafts/S19-NPU-EXPLORE.md C2 and the version table -- torch and
#     torch_npu 2.10.0 / 2.10.0.post4 for vllm-ascend v0.23.0.
# NOT VERIFIED on hardware: no 910B4 machine has run this image yet.
VERL_NPU_BASE_IMAGE = (
    "swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12"
)
VERL_NPU_CANN_VERSION = "9.1.0"
VERL_NPU_SOC_VERSION = "ascend910b1"  # Dockerfile ARG for the 910B (A2) line
VERL_NPU_VLLM_TAG = "v0.23.0"
VERL_NPU_VLLM_ASCEND_BRANCH = "releases/v0.23.0"
VERL_NPU_MINDSPEED_BRANCH = "core_r0.18.0"
# vllm-ascend 0.23 does not support sleep_mode=2
# (fork ``verl/third_party/vllm/__init__.py:41-44``), so the engine sleeps at
# level 1 on an NPU and keeps more HBM than a GPU island does.
VERL_NPU_VLLM_SLEEP_LEVEL = 1
EXPECTED_VERSIONS_NPU = {
    "torch": "2.10.0",
    "torch_npu": "2.10.0.post4",
    "vllm": "0.23.0",
    "vllm_ascend": "0.23.0",
    "transformers": "5.10.4",
    "peft": "0.19.1",
}
EXPECTED_VERSIONS_BY_FAMILY = {"nvidia": EXPECTED_VERSIONS, "ascend": EXPECTED_VERSIONS_NPU}


def expected_versions(device_family: str = "nvidia") -> dict:
    """Versions the island asserts for this device family.

    An NPU island must not assert the CUDA versions: every entry would fail.
    An unknown family is an error before launch, never a silent CUDA default.
    """
    try:
        return dict(EXPECTED_VERSIONS_BY_FAMILY[device_family])
    except KeyError:
        raise ValueError(
            f"no verl version pins for device family {device_family!r}; "
            f"known: {sorted(EXPECTED_VERSIONS_BY_FAMILY)}") from None


# ``--rl-image`` value that selects the Modal-built verl image (no registry
# digest exists: the image is built on Modal from the pinned commit + lock).
VERL_IMAGE = f"verl-build:{VERL_COMMIT}"
