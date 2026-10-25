"""verl pins (rl-verl-backend design D1, D10; user decision S16).

The user's fork https://github.com/michaellchung/verl at acad9875 (the commit
S16 measured on Modal 1xH100, VERL-MODAL-CHECK-S16.md).  Dependencies come
from the fork's own ``uv.lock`` (``uv sync --frozen --all-packages --extra vllm
--extra fsdp``), the same recipe as upstream ``docker/Dockerfile.uv.cu130``.
"""

import os
from pathlib import Path

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

# yeto's object-store client (s3:// and oss:// checkpoint stores: yeto.cloud.s3_store),
# not in the fork's uv.lock; the same pin as yeto's pyproject (a test keeps them equal).
STORE_CLIENT_REQUIREMENTS = ("boto3==1.43.104",)

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

# S27 G-EQ2 (npu1): where the verl checkout lives inside the island's image.
# The island has to start verl with this directory as the working directory,
# because verl's fully_async config names its search path relative to it
# (``fully_async_ppo_trainer.yaml``: ``hydra.searchpath:
# - file://verl/trainer/config``). Our Modal-built image puts the checkout at
# VERL_ROOT; a vendor image does not (the official Ascend verl image
# quay.nju.edu.cn/ascend/verl:verl-8.5.2-910b-ubuntu22.04-py3.11-qwen3-5 has it
# at /verl), and from the wrong directory hydra fails with "Could not load
# 'ppo_trainer'". So the directory is resolved at run time, not assumed.
VERL_ROOT_ENV = "YETO_VERL_ROOT"


def verl_root() -> str | None:
    """The directory to run verl from, or None when none can be found.

    In order: ``$YETO_VERL_ROOT``, the built-in :data:`VERL_ROOT`, then the
    parent of the importable ``verl`` package (a vendor image). Only a
    directory that actually holds ``verl/trainer/config`` is returned, so a
    wrong guess becomes None (verl's own default) instead of a broken cwd."""
    candidates = [os.environ.get(VERL_ROOT_ENV), VERL_ROOT]
    try:
        import verl  # noqa: PLC0415 - absent outside the island image

        if getattr(verl, "__file__", None):
            candidates.append(str(Path(verl.__file__).resolve().parents[1]))
    except Exception:  # noqa: BLE001 - import errors are a missing candidate, not a failure
        pass
    for candidate in candidates:
        if candidate and (Path(candidate) / "verl" / "trainer" / "config").is_dir():
            return candidate
    return None


# --- Megatron engine image (S27 G-DC, V8) ------------------------------------
#
# ``--rl-image verl-megatron-build:<commit>``: the FSDP image above plus the
# fork's ``megatron`` extra (same lock: torch, vLLM and transformers stay the
# versions asserted above) and the Mamba training kernels the extra does not
# list (built from source for Hopper, the fork's Nemotron example installs
# them by hand). Versions the Megatron island asserts on top of
# EXPECTED_VERSIONS (uv.lock of acad9875; mamba-ssm / causal-conv1d are the
# versions S27 G-R's import check passed with, NEMOTRON-VERL.md section 2).
MEGATRON_IMAGE_VARIANT = "megatron"
VERL_MEGATRON_IMAGE = f"verl-{MEGATRON_IMAGE_VARIANT}-build:{VERL_COMMIT}"
MAMBA_SSM_VERSION = "2.3.2.post1"
CAUSAL_CONV1D_VERSION = "1.6.2.post1"
EXPECTED_VERSIONS_MEGATRON = {
    "megatron_core": "0.19.2+4b4acac9a",  # the lock builds Megatron-LM at 4b4acac9a (local version)
    "megatron_bridge": "0.6.2",
    "transformer_engine": "2.16.1",
    "mamba_ssm": MAMBA_SSM_VERSION,
    "causal_conv1d": CAUSAL_CONV1D_VERSION,
}


def expected_versions_for_engine(engine: str, device_family: str = "nvidia") -> dict:
    """Versions an island asserts: the family pins, plus the Megatron stack for the
    Megatron engine island (NVIDIA only; the NPU Megatron path is MindSpeed)."""
    expected = expected_versions(device_family)
    if engine == "megatron":
        if device_family != "nvidia":
            raise ValueError(f"no verl Megatron image pins for device family {device_family!r}")
        expected.update(EXPECTED_VERSIONS_MEGATRON)
    return expected
